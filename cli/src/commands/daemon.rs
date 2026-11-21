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

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use cc_lib::daemon_client::{
    connect_existing_daemon, default_daemon_cache_dir, default_daemon_state_dir, is_daemon_enabled,
    read_daemon_runtime_info, read_daemon_stats, set_user_daemon_enabled, spawn_and_wait_daemon,
    user_daemon_config_path,
};
use cc_lib::util::run_on_tokio;
use clap::{Args, Subcommand};
use jj_cli::command_error::{CommandError, user_error};

#[derive(Args, Clone, Debug)]
pub struct DaemonArgs {
    /// Enable automatic daemon routing in user configuration
    #[arg(long, conflicts_with_all = ["disable", "status_flag"])]
    pub enable: bool,

    /// Disable automatic daemon routing in user configuration
    #[arg(long, conflicts_with_all = ["enable", "status_flag"])]
    pub disable: bool,

    /// Print live daemon status and L1/L2 cache hit metrics
    #[arg(long = "status", conflicts_with_all = ["enable", "disable"])]
    pub status_flag: bool,

    #[command(subcommand)]
    pub action: Option<DaemonSubcommand>,
}

#[derive(Subcommand, Clone, Debug)]
pub enum DaemonSubcommand {
    /// Start the local `jj-cc-daemon` background proxy
    Start {
        /// Optional default upstream `jj-cc-server` URL
        #[arg(long)]
        upstream: Option<String>,
        /// Optional Unix domain socket override
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Optional loopback TCP address override (for example `127.0.0.1:50055`)
        #[arg(long)]
        listen: Option<String>,
        /// Optional idle shutdown timeout in seconds (`0` disables idle exit)
        #[arg(long)]
        idle_timeout_secs: Option<u64>,
    },
    /// Stop the running `jj-cc-daemon` process
    Stop,
    /// Show live `jj-cc-daemon` endpoint, PID, and L1/L2 cache hit statistics
    Status,
    /// Enable `jj-cc-daemon` routing for local Commit Cloud commands
    Enable,
    /// Disable `jj-cc-daemon` routing and connect directly to `jj-cc-server`
    Disable,
    /// Purge cached objects in the local L2 disk CAS directory
    ClearCache,
}

pub async fn cmd_cc_daemon(args: &DaemonArgs) -> Result<(), CommandError> {
    if args.enable {
        return handle_enable(true);
    }
    if args.disable {
        return handle_enable(false);
    }
    if args.status_flag {
        return handle_status().await;
    }

    match args.action.as_ref().unwrap_or(&DaemonSubcommand::Status) {
        DaemonSubcommand::Enable => handle_enable(true),
        DaemonSubcommand::Disable => handle_enable(false),
        DaemonSubcommand::Status => handle_status().await,
        DaemonSubcommand::Start {
            upstream,
            socket,
            listen,
            idle_timeout_secs,
        } => {
            handle_start(
                upstream.as_deref(),
                socket.as_deref(),
                listen.as_deref(),
                *idle_timeout_secs,
            )
            .await
        }
        DaemonSubcommand::Stop => handle_stop().await,
        DaemonSubcommand::ClearCache => handle_clear_cache(),
    }
}

fn handle_enable(enabled: bool) -> Result<(), CommandError> {
    let cfg_path = set_user_daemon_enabled(enabled)
        .map_err(|e| user_error(format!("Failed to update daemon preference: {e}")))?;
    println!(
        "Local jj-cc-daemon routing {} ({})",
        if enabled { "enabled" } else { "disabled" },
        cfg_path.display()
    );
    Ok(())
}

async fn handle_start(
    upstream: Option<&str>,
    socket: Option<&Path>,
    listen: Option<&str>,
    idle_timeout_secs: Option<u64>,
) -> Result<(), CommandError> {
    let state_dir = default_daemon_state_dir();
    let st_check = state_dir.clone();
    let already_live = run_on_tokio(async move {
        Ok(connect_existing_daemon(&st_check)
            .await?
            .map(|(info, _)| info))
    })
    .await
    .map_err(|e| user_error(format!("Daemon probe failed: {e}")))?;

    if let Some(info) = already_live {
        println!(
            "jj-cc-daemon is already running (pid {}, endpoint {})",
            info.pid,
            info.socket_path
                .as_deref()
                .or(info.tcp_addr.as_deref())
                .unwrap_or("<unknown>")
        );
        return Ok(());
    }

    let cache_dir = default_daemon_cache_dir();
    let st_wait = state_dir.clone();
    let up_owned = upstream.map(str::to_owned);
    let sock_owned = socket.map(Path::to_path_buf);
    let listen_owned = listen.map(str::to_owned);
    let started = run_on_tokio(async move {
        let (info, _ch) = spawn_and_wait_daemon(
            &st_wait,
            &cache_dir,
            up_owned.as_deref(),
            sock_owned.as_deref(),
            listen_owned.as_deref(),
            idle_timeout_secs,
        )
        .await?;
        Ok(info)
    })
    .await
    .map_err(|e| user_error(format!("Failed to start jj-cc-daemon: {e}")))?;

    println!(
        "Started jj-cc-daemon (pid {}) on {}",
        started.pid,
        started
            .socket_path
            .as_deref()
            .or(started.tcp_addr.as_deref())
            .unwrap_or("<unknown>")
    );
    Ok(())
}

async fn handle_stop() -> Result<(), CommandError> {
    let state_dir = default_daemon_state_dir();
    let Some(recorded) = read_daemon_runtime_info(&state_dir) else {
        println!("jj-cc-daemon is not running.");
        return Ok(());
    };

    let st_check = state_dir.clone();
    let live = run_on_tokio(async move {
        Ok(connect_existing_daemon(&st_check)
            .await?
            .map(|(info, _)| info))
    })
    .await
    .map_err(|e| user_error(format!("Daemon probe failed: {e}")))?;

    let Some(info) = live else {
        // `daemon.json` is stale (for example after SIGKILL or crash); never signal `recorded.pid`
        // because the OS may have recycled it for an unrelated process.
        cleanup_stale_runtime_files(&state_dir, &recorded);
        println!(
            "jj-cc-daemon is not running (removed stale state for pid {}).",
            recorded.pid
        );
        return Ok(());
    };

    terminate_pid(info.pid);
    for _ in 0..120 {
        if !state_dir.join("daemon.json").exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    #[cfg(windows)]
    {
        cleanup_stale_runtime_files(&state_dir, &info);
    }
    println!("Stopped jj-cc-daemon (pid {}).", info.pid);
    Ok(())
}

fn cleanup_stale_runtime_files(
    state_dir: &Path,
    expected: &cc_lib::daemon_client::DaemonRuntimeInfo,
) {
    let still_matches =
        read_daemon_runtime_info(state_dir).is_some_and(|cur| cur.pid == expected.pid);
    if !still_matches {
        return;
    }
    if let Some(sock) = &expected.socket_path {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileTypeExt;
            if fs::symlink_metadata(sock).is_ok_and(|m| m.file_type().is_socket()) {
                let _ = fs::remove_file(sock);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = fs::remove_file(sock);
        }
    }
    let _ = fs::remove_file(state_dir.join("daemon.json"));
}

async fn handle_status() -> Result<(), CommandError> {
    let state_dir = default_daemon_state_dir();
    let cache_dir = default_daemon_cache_dir();
    let enabled = is_daemon_enabled();
    let st_check = state_dir.clone();
    let live = run_on_tokio(async move {
        Ok(connect_existing_daemon(&st_check)
            .await?
            .map(|(info, _)| info))
    })
    .await
    .unwrap_or(None);

    println!(
        "Client routing: {}",
        if enabled {
            "enabled (via jj-cc-daemon)"
        } else {
            "direct (enable with `jj cc daemon enable` or JJ_CC_USE_DAEMON=1)"
        }
    );
    println!("Preference file: {}", user_daemon_config_path().display());

    match live {
        Some(info) => {
            println!("Daemon state:    RUNNING (pid {})", info.pid);
            if let Some(sock) = &info.socket_path {
                println!("Unix socket:     {sock}");
            }
            if let Some(tcp) = &info.tcp_addr {
                println!("TCP listener:    {tcp}");
            }
            println!("L2 cache dir:    {}", info.cache_dir);
            if let Some(stats) = read_daemon_stats(&state_dir) {
                println!(
                    "Cache hits:      {} L1 + {} L2 ({:.1}% hit rate, {} misses, {} coalesced)",
                    stats.l1_hits,
                    stats.l2_hits,
                    stats.hit_ratio_percent(),
                    stats.misses,
                    stats.singleflight_coalesced
                );
                println!(
                    "Write-through:   {} objects | Upstream RPCs: {}",
                    stats.write_through_objects, stats.upstream_rpcs
                );
            }
        }
        None => {
            println!("Daemon state:    STOPPED");
            println!("L2 cache dir:    {}", cache_dir.display());
        }
    }
    Ok(())
}

fn handle_clear_cache() -> Result<(), CommandError> {
    let cache_dir = default_daemon_cache_dir();
    let objects_dir = cache_dir.join("objects");
    if objects_dir.exists() {
        fs::remove_dir_all(&objects_dir)
            .map_err(|e| user_error(format!("Failed to remove {}: {e}", objects_dir.display())))?;
    }
    fs::create_dir_all(&objects_dir)
        .map_err(|e| user_error(format!("Failed to recreate {}: {e}", objects_dir.display())))?;
    println!("Cleared local L2 object cache at {}", objects_dir.display());
    Ok(())
}

fn terminate_pid(pid: u32) {
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status();
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .status();
    }
}
