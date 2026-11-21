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

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use jj_commit_cloud_daemon::{
    DaemonConfig, default_daemon_cache_dir, default_daemon_state_dir, run_daemon,
};

#[derive(Parser, Debug)]
#[command(
    name = "jj-cc-daemon",
    about = "Local two-tier caching gRPC proxy daemon for jj-commit-cloud"
)]
struct Args {
    /// Directory storing runtime discovery metadata (`daemon.json` and `daemon.stats.json`)
    #[arg(long, env = "JJ_CC_DAEMON_DIR")]
    state_dir: Option<PathBuf>,

    /// Unix domain socket path on Linux and macOS
    #[arg(long, env = "JJ_CC_DAEMON_SOCKET")]
    socket: Option<PathBuf>,

    /// Optional loopback TCP bind address (for example `127.0.0.1:0` or `127.0.0.1:50055`);
    /// TCP clients must send the secret recorded in `daemon.json`
    #[arg(long, env = "JJ_CC_DAEMON_LISTEN")]
    listen: Option<SocketAddr>,

    /// Local L2 content-addressed disk cache directory
    #[arg(long, env = "JJ_CC_CACHE_DIR")]
    cache_dir: Option<PathBuf>,

    /// Optional default upstream `jj-cc-server` URL when callers omit `x-jj-cc-upstream`
    #[arg(long, env = "JJ_CC_UPSTREAM")]
    upstream: Option<String>,

    /// Maximum L1 in-memory object cache capacity in MiB
    #[arg(long, default_value_t = 256, env = "JJ_CC_DAEMON_MEMORY_MB")]
    memory_cache_mb: usize,

    /// Idle shutdown timeout in seconds (`0` disables automatic idle exit)
    #[arg(long, default_value_t = 1800, env = "JJ_CC_DAEMON_IDLE_SECS")]
    idle_timeout_secs: u64,
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm =
            signal(SignalKind::terminate()).expect("SIGTERM signal handler should register");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let state_dir = args.state_dir.unwrap_or_else(default_daemon_state_dir);
    let socket_path = if cfg!(unix) {
        Some(args.socket.unwrap_or_else(|| state_dir.join("daemon.sock")))
    } else {
        args.socket
    };
    let listen_addr = if cfg!(unix) {
        args.listen
    } else {
        Some(
            args.listen
                .unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 0))),
        )
    };

    let config = DaemonConfig {
        state_dir,
        socket_path,
        listen_addr,
        cache_dir: args.cache_dir.unwrap_or_else(default_daemon_cache_dir),
        default_upstream: args.upstream,
        memory_cache_mb: args.memory_cache_mb,
        idle_timeout_secs: args.idle_timeout_secs,
    };

    run_daemon(config, shutdown_signal()).await
}
