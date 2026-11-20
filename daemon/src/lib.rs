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

//! Local per-user caching gRPC proxy daemon (jj-cc-daemon) for jj-commit-cloud.
//!
//! # Authority and Crash-Safety Ordering
//! jj-cc-daemon exposes the `BackendService`, `OpStoreService`, and `WorkspaceService` gRPC
//! contracts over a `0600` per-user Unix domain socket inside a state directory owned by the
//! current user with mode `0700` (with loopback TCP fallback on platforms without Unix
//! sockets, where TCP requests must carry the random per-daemon secret from `daemon.json`).
//! Runtime discovery metadata (`daemon.json`, mode `0600`) is written to a sibling temp file
//! and renamed into place after the listener socket is bound, and removed on shutdown while
//! `daemon.json` still names this process.
//!
//! # Concurrency and Merge Invariants
//! Content-addressed immutable objects (`Commit`, `Tree`, `Symlink`, `File`, `View`, and
//! `Operation`) are cached across L1 RAM and L2 local disk CAS with singleflight miss
//! coalescing and write-through population, while mutable coordination RPCs (`GetOpHeads`,
//! `UpdateOpHeads`, `ReconcileOpHeads`, `PublishOperation`, and `WorkspaceService`) forward
//! straight to jj-cc-server so multi-agent operation convergence stays linearizable.
//!
//! # Cross-Process and RPC Boundaries
//! Clients specify their target `server_url` per call via the `x-jj-cc-upstream` metadata
//! header (or fall back to `--upstream`), allowing a single background daemon per user to
//! serve multiple local workspaces and jj-vfs mounts across local and cloud servers.

#![allow(clippy::result_large_err)]

pub mod cache;
pub mod service;
pub mod upstream;

use std::fs;
use std::future::Future;
use std::io::Write as _;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use cc_common::backend::backend_service_server::BackendServiceServer;
use cc_common::op_store::op_store_service_server::OpStoreServiceServer;
use cc_common::workspace::workspace_service_server::WorkspaceServiceServer;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tonic::service::interceptor::InterceptedService;
use tonic::transport::Server;

use crate::cache::TwoTierCache;
use crate::service::DaemonProxy;
use crate::upstream::UpstreamPool;

/// Maximum gRPC message size (`64 MiB`) matching `jj-cc-server`.
pub use cc_common::MAX_GRPC_MESSAGE_BYTES;

/// Metadata header carrying [`DaemonRuntimeInfo::secret`], required on the TCP listener.
pub const DAEMON_SECRET_HEADER: &str = "x-jj-cc-daemon-secret";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRuntimeInfo {
    pub pid: u32,
    pub socket_path: Option<String>,
    pub tcp_addr: Option<String>,
    pub cache_dir: String,
    pub stats_path: String,
    pub default_upstream: Option<String>,
    pub started_unix_secs: u64,
    /// Random per-daemon secret that TCP clients send in [`DAEMON_SECRET_HEADER`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DaemonConfig {
    pub state_dir: PathBuf,
    pub socket_path: Option<PathBuf>,
    pub listen_addr: Option<SocketAddr>,
    pub cache_dir: PathBuf,
    pub default_upstream: Option<String>,
    pub memory_cache_mb: usize,
    pub idle_timeout_secs: u64,
}

pub fn default_daemon_state_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("JJ_CC_DAEMON_DIR")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    if let Ok(xdg_runtime) = std::env::var("XDG_RUNTIME_DIR")
        && !xdg_runtime.trim().is_empty()
    {
        return PathBuf::from(xdg_runtime).join("jj-commit-cloud");
    }
    // A per-user directory under the user's cache/home, never a predictable path under `/tmp`.
    let cache = default_daemon_cache_dir();
    if cache.as_os_str().is_empty() {
        PathBuf::new()
    } else {
        cache.join("run")
    }
}

pub fn default_daemon_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("JJ_CC_CACHE_DIR")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME")
        && !xdg_cache.trim().is_empty()
    {
        return PathBuf::from(xdg_cache).join("jj-commit-cloud");
    }
    if let Some(home) = std::env::home_dir()
        && !home.as_os_str().is_empty()
    {
        return home.join(".cache").join("jj-commit-cloud");
    }
    PathBuf::new()
}

pub async fn run_daemon(
    config: DaemonConfig,
    external_shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(addr) = config.listen_addr
        && !addr.ip().is_loopback()
    {
        return Err(format!("refusing to listen on non-loopback address {addr}").into());
    }
    ensure_private_dir(&config.state_dir)?;
    create_private_dir_all(&config.cache_dir)?;

    let info_path = config.state_dir.join("daemon.json");
    let stats_path = config.state_dir.join("daemon.stats.json");
    let tcp_secret: Arc<str> = generate_daemon_secret()?.into();

    let cache = Arc::new(TwoTierCache::new(
        config.cache_dir.clone(),
        config.memory_cache_mb.saturating_mul(1024 * 1024),
    )?);
    let upstream = Arc::new(UpstreamPool::new(config.default_upstream.clone()));
    let proxy = DaemonProxy::new(Arc::clone(&cache), upstream, Some(stats_path.clone()));
    proxy.flush_stats_now();

    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<BackendServiceServer<DaemonProxy>>()
        .await;
    health_reporter
        .set_serving::<OpStoreServiceServer<DaemonProxy>>()
        .await;
    health_reporter
        .set_serving::<WorkspaceServiceServer<DaemonProxy>>()
        .await;

    let idle_timeout_secs = config.idle_timeout_secs;
    let idle_cache = Arc::clone(&cache);
    let shutdown_signal = async move {
        tokio::select! {
            _ = external_shutdown => {}
            _ = async move {
                if idle_timeout_secs == 0 {
                    std::future::pending::<()>().await;
                } else {
                    let limit_ms = idle_timeout_secs.saturating_mul(1000);
                    loop {
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        if idle_cache.idle_millis() >= limit_ms {
                            tracing::info!("jj-cc-daemon idle timeout ({idle_timeout_secs}s) reached; shutting down");
                            break;
                        }
                    }
                }
            } => {}
        }
    };

    let started_unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    #[cfg(unix)]
    if let Some(sock_path) = &config.socket_path {
        // Create a missing parent privately, but never change the mode of an existing
        // directory that `--socket` names.
        if let Some(parent) = sock_path.parent()
            && !parent.as_os_str().is_empty()
        {
            create_private_dir_all(parent)?;
        }
        if sock_path.exists() {
            if tokio::net::UnixStream::connect(sock_path).await.is_ok() {
                return Err(format!(
                    "jj-cc-daemon is already listening on {}",
                    sock_path.display()
                )
                .into());
            }
            let _ = fs::remove_file(sock_path);
        }
        let listener = tokio::net::UnixListener::bind(sock_path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(sock_path, fs::Permissions::from_mode(0o600))?;
        }

        let bound_tcp = if let Some(addr) = config.listen_addr {
            let tcp = tokio::net::TcpListener::bind(addr).await?;
            Some(tcp)
        } else {
            None
        };
        let tcp_addr_str = bound_tcp
            .as_ref()
            .and_then(|t| t.local_addr().ok())
            .map(|a| format!("http://{a}"));

        write_runtime_info(
            &info_path,
            &DaemonRuntimeInfo {
                pid: std::process::id(),
                socket_path: Some(sock_path.to_string_lossy().into_owned()),
                tcp_addr: tcp_addr_str,
                cache_dir: config.cache_dir.to_string_lossy().into_owned(),
                stats_path: stats_path.to_string_lossy().into_owned(),
                default_upstream: config.default_upstream.clone(),
                started_unix_secs,
                secret: bound_tcp.as_ref().map(|_| tcp_secret.to_string()),
            },
        )?;

        let uds_incoming = futures::stream::unfold(listener, |l| async move {
            let res = l.accept().await.map(|(s, _)| UnixConn(s));
            Some((res, l))
        });

        let router = build_router(&proxy, health_service.clone(), None);
        let serve_res = if let Some(tcp_listener) = bound_tcp {
            let (tx, rx) = tokio::sync::watch::channel(false);
            let mut rx_tcp = rx.clone();
            let tcp_incoming = futures::stream::unfold(tcp_listener, |l| async move {
                let res = l.accept().await.map(|(s, _)| TcpConn(s));
                Some((res, l))
            });
            let tcp_router = build_router(&proxy, health_service, Some(tcp_secret));
            let tcp_server = tokio::spawn(async move {
                let _ = tcp_router
                    .serve_with_incoming_shutdown(tcp_incoming, async move {
                        let _ = rx_tcp.changed().await;
                    })
                    .await;
            });
            let res = router
                .serve_with_incoming_shutdown(uds_incoming, async move {
                    shutdown_signal.await;
                    let _ = tx.send(true);
                })
                .await;
            let _ = tcp_server.await;
            res
        } else {
            router
                .serve_with_incoming_shutdown(uds_incoming, shutdown_signal)
                .await
        };

        if !info_path.exists() || runtime_info_is_owned(&info_path) {
            proxy.flush_stats_now();
        }
        remove_runtime_files_if_owned(&info_path, Some(sock_path));
        serve_res?;
        return Ok(());
    }

    let addr = config
        .listen_addr
        .unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 0)));
    let tcp_listener = tokio::net::TcpListener::bind(addr).await?;
    let local_addr = tcp_listener.local_addr()?;

    write_runtime_info(
        &info_path,
        &DaemonRuntimeInfo {
            pid: std::process::id(),
            socket_path: None,
            tcp_addr: Some(format!("http://{local_addr}")),
            cache_dir: config.cache_dir.to_string_lossy().into_owned(),
            stats_path: stats_path.to_string_lossy().into_owned(),
            default_upstream: config.default_upstream.clone(),
            started_unix_secs,
            secret: Some(tcp_secret.to_string()),
        },
    )?;

    let tcp_incoming = futures::stream::unfold(tcp_listener, |l| async move {
        let res = l.accept().await.map(|(s, _)| TcpConn(s));
        Some((res, l))
    });
    let serve_res = build_router(&proxy, health_service, Some(tcp_secret))
        .serve_with_incoming_shutdown(tcp_incoming, shutdown_signal)
        .await;

    if !info_path.exists() || runtime_info_is_owned(&info_path) {
        proxy.flush_stats_now();
    }
    remove_runtime_files_if_owned(&info_path, None);
    serve_res?;
    Ok(())
}

/// Requires [`DAEMON_SECRET_HEADER`] to match `expected` when set (the TCP listener). Unix
/// socket clients are authenticated by the socket and state directory permissions instead.
#[derive(Clone)]
struct DaemonSecretCheck {
    expected: Option<Arc<str>>,
}

impl tonic::service::Interceptor for DaemonSecretCheck {
    fn call(&mut self, request: tonic::Request<()>) -> Result<tonic::Request<()>, tonic::Status> {
        let Some(expected) = &self.expected else {
            return Ok(request);
        };
        let presented = request
            .metadata()
            .get(DAEMON_SECRET_HEADER)
            .map(|v| v.as_bytes());
        if presented.is_some_and(|p| constant_time_eq(p, expected.as_bytes())) {
            Ok(request)
        } else {
            Err(tonic::Status::unauthenticated(format!(
                "missing or invalid {DAEMON_SECRET_HEADER} metadata"
            )))
        }
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn generate_daemon_secret() -> std::io::Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(hex::encode(bytes))
}

fn build_router(
    proxy: &DaemonProxy,
    health_service: tonic_health::pb::health_server::HealthServer<
        impl tonic_health::pb::health_server::Health,
    >,
    required_secret: Option<Arc<str>>,
) -> tonic::transport::server::Router {
    let check = DaemonSecretCheck {
        expected: required_secret,
    };
    Server::builder()
        .add_service(InterceptedService::new(health_service, check.clone()))
        .add_service(InterceptedService::new(
            BackendServiceServer::new(proxy.clone())
                .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES),
            check.clone(),
        ))
        .add_service(InterceptedService::new(
            OpStoreServiceServer::new(proxy.clone())
                .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES),
            check.clone(),
        ))
        .add_service(InterceptedService::new(
            WorkspaceServiceServer::new(proxy.clone())
                .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES),
            check,
        ))
}

/// Creates `dir` (mode `0700`) if missing and verifies that it is a directory owned by the
/// current user without group or other access. Whoever controls this directory controls
/// `daemon.json` and the socket, so the daemon refuses to start otherwise.
fn ensure_private_dir(dir: &Path) -> std::io::Result<()> {
    create_private_dir_all(dir)?;
    verify_private_dir(dir)
}

/// Creates missing directories with mode `0700`; existing directories keep their mode.
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
fn verify_private_dir(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let meta = fs::symlink_metadata(dir)?;
    // SAFETY: `geteuid` has no preconditions and cannot fail.
    let euid = unsafe { libc::geteuid() };
    let mode = meta.permissions().mode() & 0o7777;
    if !meta.is_dir() || meta.uid() != euid || mode & 0o077 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "{} must be a directory owned by uid {euid} with mode 0700 (found uid {}, mode \
                 {mode:o})",
                dir.display(),
                meta.uid()
            ),
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn verify_private_dir(_dir: &Path) -> std::io::Result<()> {
    Ok(())
}

fn write_runtime_info(path: &Path, info: &DaemonRuntimeInfo) -> std::io::Result<()> {
    let bytes = serde_json::to_vec_pretty(info).map_err(std::io::Error::other)?;
    let tmp = path.with_extension(format!("json.tmp.{}", std::process::id()));
    let _ = fs::remove_file(&tmp);
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        // `daemon.json` holds the TCP secret.
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(&tmp)?.write_all(&bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn runtime_info_is_owned(info_path: &Path) -> bool {
    fs::read(info_path)
        .ok()
        .and_then(|raw| serde_json::from_slice::<DaemonRuntimeInfo>(&raw).ok())
        .is_some_and(|info| info.pid == std::process::id())
}

/// Removes `daemon.json` (and the socket) only while `daemon.json` still names this process,
/// so an exiting daemon never deletes the discovery files of a daemon that replaced it.
fn remove_runtime_files_if_owned(info_path: &Path, sock_path: Option<&Path>) {
    if runtime_info_is_owned(info_path) {
        if let Some(sock_path) = sock_path {
            let _ = fs::remove_file(sock_path);
        }
        let _ = fs::remove_file(info_path);
    }
}

#[cfg(unix)]
struct UnixConn(tokio::net::UnixStream);

#[cfg(unix)]
impl tonic::transport::server::Connected for UnixConn {
    type ConnectInfo = ();
    fn connect_info(&self) -> Self::ConnectInfo {}
}

#[cfg(unix)]
impl AsyncRead for UnixConn {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.0).poll_read(cx, buf)
    }
}

#[cfg(unix)]
impl AsyncWrite for UnixConn {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.0).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.0).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.0).poll_shutdown(cx)
    }
}

struct TcpConn(tokio::net::TcpStream);

impl tonic::transport::server::Connected for TcpConn {
    type ConnectInfo = Option<SocketAddr>;
    fn connect_info(&self) -> Self::ConnectInfo {
        self.0.peer_addr().ok()
    }
}

impl AsyncRead for TcpConn {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.0).poll_read(cx, buf)
    }
}

impl AsyncWrite for TcpConn {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.0).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.0).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.0).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cc_common::EMPTY_TREE_ID_BYTES;
    use cc_common::backend::ReadTreeRequest;
    use cc_common::backend::backend_service_client::BackendServiceClient;
    use tonic::transport::Endpoint;
    use tonic::{Code, Request};

    #[tokio::test]
    async fn run_daemon_rejects_non_loopback_listen_address() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = DaemonConfig {
            state_dir: dir.path().join("state"),
            socket_path: None,
            listen_addr: Some("0.0.0.0:0".parse().unwrap()),
            cache_dir: dir.path().join("cache"),
            default_upstream: None,
            memory_cache_mb: 8,
            idle_timeout_secs: 0,
        };
        let err = run_daemon(cfg, std::future::pending()).await.unwrap_err();
        assert!(
            err.to_string().contains("non-loopback"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn tcp_listener_requires_per_daemon_secret_from_0600_daemon_json() {
        let dir = tempfile::tempdir().unwrap();
        let state_dir = dir.path().join("state");
        let cache_dir = dir.path().join("cache");
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let cfg = DaemonConfig {
            state_dir: state_dir.clone(),
            socket_path: None,
            listen_addr: Some("127.0.0.1:0".parse().unwrap()),
            cache_dir,
            default_upstream: Some("http://127.0.0.1:1".to_string()),
            memory_cache_mb: 8,
            idle_timeout_secs: 0,
        };
        let handle = tokio::spawn(async move {
            run_daemon(cfg, async move {
                let _ = stop_rx.await;
            })
            .await
        });

        let info_path = state_dir.join("daemon.json");
        let info: DaemonRuntimeInfo = loop {
            if let Ok(raw) = fs::read(&info_path)
                && let Ok(parsed) = serde_json::from_slice(&raw)
            {
                break parsed;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        };

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&info_path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "daemon.json must be mode 0600");
        }

        let tcp_url = info.tcp_addr.expect("tcp_addr recorded");
        let secret = info.secret.expect("secret recorded for TCP listener");
        let channel = Endpoint::from_shared(tcp_url)
            .unwrap()
            .connect()
            .await
            .unwrap();
        let mut client = BackendServiceClient::new(channel);

        let empty_tree_req = || ReadTreeRequest {
            project_id: "repo".to_string(),
            path: String::new(),
            tree_id: EMPTY_TREE_ID_BYTES.to_vec(),
            page_size: 0,
            page_token: String::new(),
        };

        let missing_err = client
            .read_tree(Request::new(empty_tree_req()))
            .await
            .unwrap_err();
        assert_eq!(missing_err.code(), Code::Unauthenticated);

        let mut wrong_req = Request::new(empty_tree_req());
        wrong_req
            .metadata_mut()
            .insert(DAEMON_SECRET_HEADER, "wrong-secret".parse().unwrap());
        let wrong_err = client.read_tree(wrong_req).await.unwrap_err();
        assert_eq!(wrong_err.code(), Code::Unauthenticated);

        let mut valid_req = Request::new(empty_tree_req());
        valid_req
            .metadata_mut()
            .insert(DAEMON_SECRET_HEADER, secret.parse().unwrap());
        let resp = client.read_tree(valid_req).await.unwrap().into_inner();
        assert_eq!(resp.tree_id, EMPTY_TREE_ID_BYTES);

        let _ = stop_tx.send(());
        handle.await.unwrap().unwrap();
        assert!(!info_path.exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn state_dir_rejects_open_permissions_and_socket_parent_mode_is_preserved() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let open_state_dir = dir.path().join("open_state");
        fs::create_dir_all(&open_state_dir).unwrap();
        fs::set_permissions(&open_state_dir, fs::Permissions::from_mode(0o755)).unwrap();

        let bad_cfg = DaemonConfig {
            state_dir: open_state_dir,
            socket_path: None,
            listen_addr: Some("127.0.0.1:0".parse().unwrap()),
            cache_dir: dir.path().join("cache"),
            default_upstream: None,
            memory_cache_mb: 8,
            idle_timeout_secs: 0,
        };
        let err = run_daemon(bad_cfg, std::future::pending())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("0700"), "unexpected error: {err}");

        let sock_parent = dir.path().join("shared_parent");
        fs::create_dir_all(&sock_parent).unwrap();
        fs::set_permissions(&sock_parent, fs::Permissions::from_mode(0o755)).unwrap();
        let sock_path = sock_parent.join("custom.sock");

        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let ok_cfg = DaemonConfig {
            state_dir: dir.path().join("private_state"),
            socket_path: Some(sock_path.clone()),
            listen_addr: None,
            cache_dir: dir.path().join("cache"),
            default_upstream: None,
            memory_cache_mb: 8,
            idle_timeout_secs: 0,
        };
        let handle = tokio::spawn(async move {
            run_daemon(ok_cfg, async move {
                let _ = stop_rx.await;
            })
            .await
        });
        while !sock_path.exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let parent_mode = fs::metadata(&sock_parent).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            parent_mode, 0o755,
            "existing --socket parent mode must not be changed"
        );

        let _ = stop_tx.send(());
        handle.await.unwrap().unwrap();
    }

    #[test]
    fn shutdown_cleanup_preserves_successor_daemon_files() {
        let dir = tempfile::tempdir().unwrap();
        let info_path = dir.path().join("daemon.json");
        let sock_path = dir.path().join("daemon.sock");
        fs::write(&sock_path, b"").unwrap();
        write_runtime_info(
            &info_path,
            &DaemonRuntimeInfo {
                pid: std::process::id().wrapping_add(1),
                socket_path: Some(sock_path.to_string_lossy().into_owned()),
                tcp_addr: None,
                cache_dir: dir.path().to_string_lossy().into_owned(),
                stats_path: dir
                    .path()
                    .join("daemon.stats.json")
                    .to_string_lossy()
                    .into_owned(),
                default_upstream: None,
                started_unix_secs: 1,
                secret: None,
            },
        )
        .unwrap();

        remove_runtime_files_if_owned(&info_path, Some(&sock_path));
        assert!(info_path.exists(), "must not delete successor daemon.json");
        assert!(sock_path.exists(), "must not delete successor socket");
    }
}
