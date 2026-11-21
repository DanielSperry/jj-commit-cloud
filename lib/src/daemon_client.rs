// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Local IPC discovery, on-demand process spawning, and Unix socket or loopback TCP
//! transport connector for jj-cc-daemon.
//!
//! # Authority and Crash-Safety Ordering
//! Runtime discovery reads `<state_dir>/daemon.json` after verifying that `<state_dir>` is a
//! directory owned by the current user with mode `0700` on Unix. Cross-process auto-spawn
//! serializes through an advisory `std::fs::File::lock()` on `<state_dir>/daemon.spawn.lock`,
//! which the kernel releases if the spawning client exits or receives `SIGINT`.
//!
//! # Concurrency and Merge Invariants
//! [`DAEMON_ROUTE_CACHE`] records either the connected HTTP/2 [`Channel`] or a bounded
//! negative backoff per `state_dir` so concurrent RPCs inside one process share a single
//! connection and never repeat spawn attempts on every call when the daemon cannot start.
//!
//! # Cross-Process and RPC Boundaries
//! Spawned daemons run in an independent process group (`process_group(0)` on Unix) with
//! stdout and stderr appended to `<state_dir>/daemon.log` (`0600`). Unix domain socket
//! connections verify peer UID and PID via `SO_PEERCRED`, and loopback TCP connections
//! attach [`DAEMON_SECRET_HEADER`] read from `0600` `daemon.json`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex as StdMutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tonic::transport::{Channel, Endpoint};

pub const UPSTREAM_HEADER: &str = "x-jj-cc-upstream";
pub const DAEMON_SECRET_HEADER: &str = "x-jj-cc-daemon-secret";

const DAEMON_UNAVAILABLE_BACKOFF: Duration = Duration::from_secs(30);

static SPAWN_LOCK: Mutex<()> = Mutex::const_new(());
static DAEMON_PREF_CACHE: LazyLock<StdMutex<Option<(PathBuf, bool)>>> =
    LazyLock::new(|| StdMutex::new(None));
static DAEMON_ROUTE_CACHE: LazyLock<StdMutex<HashMap<PathBuf, CachedDaemonState>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

#[derive(Clone)]
enum CachedDaemonState {
    Connected {
        info: DaemonRuntimeInfo,
        channel: Channel,
    },
    Unavailable {
        retry_after: Instant,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserDaemonPreference {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRuntimeInfo {
    pub pid: u32,
    pub socket_path: Option<String>,
    pub tcp_addr: Option<String>,
    pub cache_dir: String,
    pub stats_path: String,
    pub default_upstream: Option<String>,
    pub started_unix_secs: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DaemonStatsView {
    pub l1_hits: u64,
    pub l2_hits: u64,
    pub misses: u64,
    pub singleflight_coalesced: u64,
    pub write_through_objects: u64,
    pub upstream_rpcs: u64,
    pub l1_bytes: usize,
    pub l1_entries: usize,
    #[serde(default)]
    pub l2_write_errors: u64,
}

impl DaemonStatsView {
    pub fn hit_ratio_percent(&self) -> f64 {
        let total = self.l1_hits + self.l2_hits + self.misses;
        if total == 0 {
            0.0
        } else {
            ((self.l1_hits + self.l2_hits) as f64) * 100.0 / (total as f64)
        }
    }
}

pub fn default_daemon_state_dir() -> PathBuf {
    default_daemon_state_dir_from(
        std::env::var("JJ_CC_DAEMON_DIR").ok().as_deref(),
        std::env::var("XDG_RUNTIME_DIR").ok().as_deref(),
        &default_daemon_cache_dir(),
    )
}

fn default_daemon_state_dir_from(
    explicit_dir: Option<&str>,
    xdg_runtime: Option<&str>,
    cache_dir: &Path,
) -> PathBuf {
    if let Some(dir) = explicit_dir
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    if let Some(xdg) = xdg_runtime
        && !xdg.trim().is_empty()
    {
        return PathBuf::from(xdg).join("jj-commit-cloud");
    }
    if cache_dir.as_os_str().is_empty() {
        PathBuf::new()
    } else {
        cache_dir.join("run")
    }
}

pub fn default_daemon_cache_dir() -> PathBuf {
    default_daemon_cache_dir_from(
        std::env::var("JJ_CC_CACHE_DIR").ok().as_deref(),
        std::env::var("XDG_CACHE_HOME").ok().as_deref(),
        std::env::home_dir().as_deref(),
    )
}

fn default_daemon_cache_dir_from(
    explicit_dir: Option<&str>,
    xdg_cache: Option<&str>,
    home: Option<&Path>,
) -> PathBuf {
    if let Some(dir) = explicit_dir
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    if let Some(xdg) = xdg_cache
        && !xdg.trim().is_empty()
    {
        return PathBuf::from(xdg).join("jj-commit-cloud");
    }
    if let Some(home_dir) = home
        && !home_dir.as_os_str().is_empty()
    {
        return home_dir.join(".cache").join("jj-commit-cloud");
    }
    PathBuf::new()
}

pub fn user_daemon_config_path() -> PathBuf {
    if let Ok(p) = std::env::var("JJ_CC_DAEMON_CONFIG")
        && !p.trim().is_empty()
    {
        return PathBuf::from(p);
    }
    if let Ok(xdg_cfg) = std::env::var("XDG_CONFIG_HOME")
        && !xdg_cfg.trim().is_empty()
    {
        return PathBuf::from(xdg_cfg)
            .join("jj-commit-cloud")
            .join("daemon.toml");
    }
    if let Some(home) = std::env::home_dir()
        && !home.as_os_str().is_empty()
    {
        return home
            .join(".config")
            .join("jj-commit-cloud")
            .join("daemon.toml");
    }
    PathBuf::new()
}

pub fn set_user_daemon_enabled(
    enabled: bool,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let cfg_path = user_daemon_config_path();
    if cfg_path.as_os_str().is_empty() {
        return Err(
            "cannot determine user config path; set HOME, XDG_CONFIG_HOME, or JJ_CC_DAEMON_CONFIG"
                .into(),
        );
    }
    if let Some(parent) = cfg_path.parent()
        && !parent.as_os_str().is_empty()
    {
        create_private_dir_all(parent)?;
    }
    let content = toml::to_string_pretty(&UserDaemonPreference { enabled })?;
    fs::write(&cfg_path, content)?;
    *DAEMON_PREF_CACHE.lock().unwrap() = Some((cfg_path.clone(), enabled));
    Ok(cfg_path)
}

pub fn is_daemon_enabled() -> bool {
    if let Ok(raw) = std::env::var("JJ_CC_USE_DAEMON") {
        let normalized = raw.trim().to_ascii_lowercase();
        if matches!(normalized.as_str(), "1" | "true" | "yes" | "on") {
            return true;
        }
        if matches!(normalized.as_str(), "0" | "false" | "no" | "off") {
            return false;
        }
    }
    let cfg_path = user_daemon_config_path();
    if cfg_path.as_os_str().is_empty() {
        return false;
    }
    if let Some((cached_path, cached_enabled)) = &*DAEMON_PREF_CACHE.lock().unwrap()
        && cached_path == &cfg_path
    {
        return *cached_enabled;
    }
    let enabled = fs::read_to_string(&cfg_path)
        .ok()
        .and_then(|raw| toml::from_str::<UserDaemonPreference>(&raw).ok())
        .is_some_and(|pref| pref.enabled);
    *DAEMON_PREF_CACHE.lock().unwrap() = Some((cfg_path, enabled));
    enabled
}

pub fn ensure_private_state_dir(dir: &Path) -> std::io::Result<()> {
    create_private_dir_all(dir)?;
    verify_private_state_dir(dir)
}

fn create_private_dir_all(dir: &Path) -> std::io::Result<()> {
    if dir.as_os_str().is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "cannot determine private per-user daemon directory; set XDG_RUNTIME_DIR, HOME, or JJ_CC_DAEMON_DIR",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        let created_leaf = fs::symlink_metadata(dir).is_err();
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)?;
        if created_leaf {
            fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        fs::create_dir_all(dir)
    }
}

#[cfg(unix)]
pub fn verify_private_state_dir(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let meta = fs::symlink_metadata(dir)?;
    // SAFETY: `geteuid` has no preconditions and cannot fail.
    let euid = unsafe { libc::geteuid() };
    let mode = meta.permissions().mode() & 0o7777;
    if !meta.is_dir() || meta.uid() != euid || mode & 0o077 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "{} must be a directory owned by uid {euid} with mode 0700 (found uid {}, mode {mode:o})",
                dir.display(),
                meta.uid()
            ),
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn verify_private_state_dir(dir: &Path) -> std::io::Result<()> {
    if dir.as_os_str().is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "cannot determine private per-user daemon directory",
        ));
    }
    Ok(())
}

pub fn read_daemon_runtime_info(state_dir: &Path) -> Option<DaemonRuntimeInfo> {
    verify_private_state_dir(state_dir).ok()?;
    let raw = fs::read_to_string(state_dir.join("daemon.json")).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn read_daemon_stats(state_dir: &Path) -> Option<DaemonStatsView> {
    verify_private_state_dir(state_dir).ok()?;
    let raw = fs::read_to_string(state_dir.join("daemon.stats.json")).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn locate_daemon_binary() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("JJ_CC_DAEMON_BIN")
        && !explicit.trim().is_empty()
    {
        let p = PathBuf::from(explicit);
        if p.is_file() {
            return Some(p);
        }
        return None;
    }
    let bin_name = format!("jj-cc-daemon{}", std::env::consts::EXE_SUFFIX);
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let sibling = dir.join(&bin_name);
        if sibling.is_file() {
            return Some(sibling);
        }
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join(&bin_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn spawn_daemon_child(
    state_dir: &Path,
    cache_dir: &Path,
    upstream: Option<&str>,
    socket_override: Option<&Path>,
    listen_override: Option<&str>,
    idle_timeout_secs: Option<u64>,
) -> Result<Child, Box<dyn std::error::Error + Send + Sync>> {
    ensure_private_state_dir(state_dir)?;
    create_private_dir_all(cache_dir)?;
    let bin =
        locate_daemon_binary().ok_or("jj-cc-daemon binary not found alongside jj or in PATH")?;

    let log_path = state_dir.join("daemon.log");
    let mut log_opts = fs::OpenOptions::new();
    log_opts.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        log_opts.mode(0o600);
    }
    let stdout_log = log_opts.open(&log_path)?;
    let stderr_log = stdout_log.try_clone()?;

    let mut cmd = Command::new(bin);
    cmd.arg("--state-dir")
        .arg(state_dir)
        .arg("--cache-dir")
        .arg(cache_dir)
        .env_remove("JJ_CC_AUTH_TOKEN")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_log))
        .stderr(Stdio::from(stderr_log));

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    if let Some(up) = upstream {
        cmd.arg("--upstream").arg(up);
    }
    if let Some(sock) = socket_override {
        cmd.arg("--socket").arg(sock);
    }
    if let Some(addr) = listen_override {
        cmd.arg("--listen").arg(addr);
    }
    if let Some(idle) = idle_timeout_secs {
        cmd.arg("--idle-timeout-secs").arg(idle.to_string());
    }

    Ok(cmd.spawn()?)
}

pub fn spawn_daemon_background(
    state_dir: &Path,
    cache_dir: &Path,
    upstream: Option<&str>,
    socket_override: Option<&Path>,
    listen_override: Option<&str>,
    idle_timeout_secs: Option<u64>,
) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    let child = spawn_daemon_child(
        state_dir,
        cache_dir,
        upstream,
        socket_override,
        listen_override,
        idle_timeout_secs,
    )?;
    Ok(child.id())
}

pub async fn spawn_and_wait_daemon(
    state_dir: &Path,
    cache_dir: &Path,
    upstream: Option<&str>,
    socket_override: Option<&Path>,
    listen_override: Option<&str>,
    idle_timeout_secs: Option<u64>,
) -> Result<(DaemonRuntimeInfo, Channel), Box<dyn std::error::Error + Send + Sync>> {
    let mut child = spawn_daemon_child(
        state_dir,
        cache_dir,
        upstream,
        socket_override,
        listen_override,
        idle_timeout_secs,
    )?;
    let pid = child.id();
    for _ in 0..120 {
        if let Some(status) = child.try_wait()? {
            let log_tail = fs::read_to_string(state_dir.join("daemon.log"))
                .ok()
                .and_then(|s| s.lines().last().map(str::to_owned))
                .unwrap_or_default();
            return Err(format!(
                "jj-cc-daemon (pid {pid}) exited during startup ({status}){}",
                if log_tail.is_empty() {
                    String::new()
                } else {
                    format!(": {log_tail}")
                }
            )
            .into());
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
        if let Some(ready) = connect_existing_daemon(state_dir).await? {
            return Ok(ready);
        }
    }
    Err(format!(
        "timed out waiting for jj-cc-daemon (pid {pid}) in {}",
        state_dir.display()
    )
    .into())
}

fn pid_matches_daemon(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        if pid > i32::MAX as u32 {
            return false;
        }
        // SAFETY: `kill(pid, 0)` performs a kernel existence/permission probe without sending a signal.
        let rc = unsafe { libc::kill(pid as i32, 0) };
        if rc != 0 {
            return false;
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(cmdline) = fs::read(format!("/proc/{pid}/cmdline"))
            && !cmdline.is_empty()
        {
            let text = String::from_utf8_lossy(&cmdline);
            return text.contains("jj-cc-daemon");
        }
    }
    true
}

pub async fn connect_existing_daemon(
    state_dir: &Path,
) -> Result<Option<(DaemonRuntimeInfo, Channel)>, Box<dyn std::error::Error + Send + Sync>> {
    if state_dir.as_os_str().is_empty() {
        return Err("cannot determine private per-user daemon directory".into());
    }
    match fs::symlink_metadata(state_dir) {
        Ok(_) => verify_private_state_dir(state_dir)?,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    }

    let raw = match fs::read_to_string(state_dir.join("daemon.json")) {
        Ok(s) => s,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let Ok(info) = serde_json::from_str::<DaemonRuntimeInfo>(&raw) else {
        return Ok(None);
    };

    #[cfg(unix)]
    if let Some(sock_str) = &info.socket_path {
        let sock_path = PathBuf::from(sock_str);
        if sock_path.exists()
            && let Ok(ch) = connect_unix_channel(&sock_path, Some(info.pid)).await
        {
            return Ok(Some((info, ch)));
        }
        return Ok(None);
    }

    if let Some(tcp_url) = &info.tcp_addr
        && pid_matches_daemon(info.pid)
        && let Ok(ep) = Endpoint::from_shared(tcp_url.clone())
        && let Ok(ch) = ep
            .tcp_nodelay(true)
            .connect_timeout(Duration::from_secs(5))
            .http2_keep_alive_interval(Duration::from_secs(300))
            .keep_alive_timeout(Duration::from_secs(20))
            .connect()
            .await
    {
        return Ok(Some((info, ch)));
    }

    Ok(None)
}

struct CrossProcessSpawnGuard {
    _file: fs::File,
}

async fn acquire_cross_process_spawn_lock(
    state_dir: &Path,
) -> Result<CrossProcessSpawnGuard, Box<dyn std::error::Error + Send + Sync>> {
    ensure_private_state_dir(state_dir)?;
    let lock_path = state_dir.join("daemon.spawn.lock");
    let mut opts = fs::OpenOptions::new();
    opts.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let file = opts.open(&lock_path)?;
    let locked = tokio::time::timeout(
        Duration::from_secs(10),
        tokio::task::spawn_blocking(move || -> std::io::Result<fs::File> {
            file.lock()?;
            Ok(file)
        }),
    )
    .await
    .map_err(|_| {
        format!(
            "timed out waiting for cross-process daemon spawn lock {}",
            lock_path.display()
        )
    })?
    .map_err(|e| format!("daemon spawn lock join error: {e}"))??;
    Ok(CrossProcessSpawnGuard { _file: locked })
}

fn check_cached_route(
    state_dir: &Path,
) -> Option<Result<(DaemonRuntimeInfo, Channel), Box<dyn std::error::Error + Send + Sync>>> {
    let guard = DAEMON_ROUTE_CACHE.lock().unwrap();
    match guard.get(state_dir)? {
        CachedDaemonState::Connected { info, channel } => Some(Ok((info.clone(), channel.clone()))),
        CachedDaemonState::Unavailable { retry_after } => {
            if Instant::now() < *retry_after {
                Some(Err("jj-cc-daemon unavailable (cached backoff)".into()))
            } else {
                None
            }
        }
    }
}

async fn connect_or_spawn_in(
    state_dir: &Path,
    cache_dir: &Path,
    default_upstream: &str,
) -> Result<(DaemonRuntimeInfo, Channel), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(cached) = check_cached_route(state_dir) {
        return cached;
    }

    let _in_proc_guard = SPAWN_LOCK.lock().await;
    if let Some(cached) = check_cached_route(state_dir) {
        return cached;
    }

    let attempt: Result<(DaemonRuntimeInfo, Channel), Box<dyn std::error::Error + Send + Sync>> =
        async {
            ensure_private_state_dir(state_dir)?;
            if let Some(ready) = connect_existing_daemon(state_dir).await? {
                return Ok(ready);
            }
            let _cross_proc_guard = acquire_cross_process_spawn_lock(state_dir).await?;
            if let Some(ready) = connect_existing_daemon(state_dir).await? {
                return Ok(ready);
            }
            spawn_and_wait_daemon(
                state_dir,
                cache_dir,
                Some(default_upstream),
                None,
                None,
                None,
            )
            .await
        }
        .await;

    let mut cache_guard = DAEMON_ROUTE_CACHE.lock().unwrap();
    match attempt {
        Ok((info, ch)) => {
            cache_guard.insert(
                state_dir.to_path_buf(),
                CachedDaemonState::Connected {
                    info: info.clone(),
                    channel: ch.clone(),
                },
            );
            Ok((info, ch))
        }
        Err(err) => {
            tracing::warn!(
                "jj-cc-daemon unavailable in {}, falling back to direct connection: {err}",
                state_dir.display()
            );
            cache_guard.insert(
                state_dir.to_path_buf(),
                CachedDaemonState::Unavailable {
                    retry_after: Instant::now() + DAEMON_UNAVAILABLE_BACKOFF,
                },
            );
            Err(err)
        }
    }
}

pub async fn connect_or_spawn_daemon(
    default_upstream: &str,
) -> Result<(DaemonRuntimeInfo, Channel), Box<dyn std::error::Error + Send + Sync>> {
    let state_dir = default_daemon_state_dir();
    let cache_dir = default_daemon_cache_dir();
    connect_or_spawn_in(&state_dir, &cache_dir, default_upstream).await
}

#[cfg(unix)]
async fn connect_unix_channel(
    socket_path: &Path,
    expected_pid: Option<u32>,
) -> Result<Channel, Box<dyn std::error::Error + Send + Sync>> {
    use hyper_util::rt::TokioIo;
    use tower::service_fn;

    let path = socket_path.to_path_buf();
    let check_initial_pid = Arc::new(AtomicBool::new(true));
    let endpoint = Endpoint::try_from("http://[::]:50051")?
        .connect_timeout(Duration::from_secs(5))
        .http2_keep_alive_interval(Duration::from_secs(300))
        .keep_alive_timeout(Duration::from_secs(20));

    let channel = endpoint
        .connect_with_connector(service_fn(move |_: tonic::transport::Uri| {
            let p = path.clone();
            let verify_pid = Arc::clone(&check_initial_pid);
            async move {
                let stream = tokio::net::UnixStream::connect(p).await?;
                let check_pid_now = verify_pid.swap(false, Ordering::Relaxed);
                if let Ok(cred) = stream.peer_cred() {
                    // SAFETY: `geteuid` has no preconditions and cannot fail.
                    let euid = unsafe { libc::geteuid() };
                    if cred.uid() != euid {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            format!(
                                "daemon socket peer uid {} does not match current uid {euid}",
                                cred.uid()
                            ),
                        ));
                    }
                    if check_pid_now
                        && let (Some(expected), Some(actual)) = (expected_pid, cred.pid())
                        && actual > 0
                        && (actual as u32) != expected
                    {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::ConnectionRefused,
                            format!(
                                "daemon socket peer pid {actual} does not match daemon.json pid {expected}"
                            ),
                        ));
                    }
                }
                Ok::<_, std::io::Error>(TokioIo::new(stream))
            }
        }))
        .await?;
    Ok(channel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_dir_fallback_uses_home_cache_and_never_temp_dir() {
        let home = Path::new("/home/alice");
        let cache = default_daemon_cache_dir_from(None, None, Some(home));
        assert_eq!(cache, PathBuf::from("/home/alice/.cache/jj-commit-cloud"));
        let state = default_daemon_state_dir_from(None, None, &cache);
        assert_eq!(
            state,
            PathBuf::from("/home/alice/.cache/jj-commit-cloud/run")
        );

        let missing_all_cache = default_daemon_cache_dir_from(None, None, None);
        assert!(missing_all_cache.as_os_str().is_empty());
        let missing_all_state = default_daemon_state_dir_from(None, None, &missing_all_cache);
        assert!(missing_all_state.as_os_str().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn state_dir_rejects_group_or_world_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let insecure = dir.path().join("insecure");
        fs::create_dir_all(&insecure).unwrap();
        fs::set_permissions(&insecure, fs::Permissions::from_mode(0o755)).unwrap();

        assert!(ensure_private_state_dir(&insecure).is_err());
        assert!(connect_existing_daemon(&insecure).await.is_err());

        let secure = dir.path().join("secure");
        ensure_private_state_dir(&secure).unwrap();
        let mode = fs::metadata(&secure).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
    }

    #[tokio::test]
    async fn file_lock_blocks_concurrent_holder_and_releases_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let state_dir = dir.path().join("lock_state");
        let guard = acquire_cross_process_spawn_lock(&state_dir).await.unwrap();

        let lock_path = state_dir.join("daemon.spawn.lock");
        let contender = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        assert!(
            matches!(contender.try_lock(), Err(fs::TryLockError::WouldBlock)),
            "second handle must observe active flock"
        );

        drop(guard);
        assert!(
            lock_path.exists(),
            "flock lockfile must stay in place across drops"
        );
        contender
            .try_lock()
            .expect("lock must release as soon as guard drops");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn failed_spawn_caches_negative_verdict_without_repeated_delays() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let bad_state = dir.path().join("bad_state");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&bad_state).unwrap();
        fs::set_permissions(&bad_state, fs::Permissions::from_mode(0o777)).unwrap();

        assert!(
            connect_or_spawn_in(&bad_state, &cache_dir, "http://127.0.0.1:1")
                .await
                .is_err()
        );

        let t0 = Instant::now();
        for _ in 0..25 {
            assert!(
                connect_or_spawn_in(&bad_state, &cache_dir, "http://127.0.0.1:1")
                    .await
                    .is_err()
            );
        }
        assert!(
            t0.elapsed() < Duration::from_millis(200),
            "cached negative verdict must avoid per-RPC spawn retries"
        );
    }
}
