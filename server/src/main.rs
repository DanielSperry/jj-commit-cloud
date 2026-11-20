#![allow(clippy::result_large_err)]

use clap::{Parser, ValueEnum};
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use cc_common::backend::backend_service_server::BackendServiceServer;
use cc_common::op_store::op_store_service_server::OpStoreServiceServer;

pub mod auth;
mod backend;
pub mod error_util;
mod hash_utils;
mod op_store;
pub mod store;

use auth::BearerAuthInterceptor;
use backend::CommitCloudBackendService;
use op_store::CommitCloudOpStoreService;
use store::{MemoryStore, Store};

// Storage backend to run the server against.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum StoreType {
    Memory,
    Sqlite,
}

#[derive(Parser)]
#[command(name = "jj-cc-server", about = "Jujutsu Commit Cloud Server")]
struct Args {
    /// Host interface to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on (use 0 for ephemeral port assignment)
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Storage backend type
    #[arg(long, value_enum, default_value_t = StoreType::Memory)]
    store_type: StoreType,

    /// Path to SQLite database file (defaults to ~/.jj-cc-server/commit_cloud.db)
    #[arg(long)]
    sqlite_path: Option<std::path::PathBuf>,

    #[arg(
        long = "auth-token",
        env = "JJ_CC_SERVER_AUTH_TOKEN",
        value_delimiter = ',',
        hide_env_values = true
    )]
    auth_tokens: Vec<String>,
}

pub struct CommitCloudServerImpl {
    pub store: Arc<dyn Store>,
}

impl Default for CommitCloudServerImpl {
    fn default() -> Self {
        Self {
            store: Arc::new(MemoryStore::default()),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("jj_cc_server=info".parse()?),
        )
        .init();

    let args = Args::parse();

    let auth = BearerAuthInterceptor::from_tokens(&args.auth_tokens);
    let cli_auth_flag = std::env::args_os().any(|a| {
        let s = a.to_string_lossy();
        s == "--auth-token" || s.starts_with("--auth-token=")
    });
    let auth_requested = !args.auth_tokens.is_empty()
        || cli_auth_flag
        || std::env::var_os("JJ_CC_SERVER_AUTH_TOKEN").is_some();
    if auth_requested && !auth.is_enabled() {
        return Err(
            "--auth-token (or JJ_CC_SERVER_AUTH_TOKEN) is set but contains no non-empty token; unset it to run without authentication"
                .into(),
        );
    }
    if auth.is_enabled() {
        info!("Bearer token authentication enabled for gRPC services");
    } else {
        warn!("Bearer token authentication disabled: gRPC services accept any caller");
    }
    match args.store_type {
        StoreType::Memory => {}
        StoreType::Sqlite => {
            eprintln!("SQLite storage backend is not yet implemented");
            std::process::exit(1);
        }
    }

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_service_status("", tonic_health::ServingStatus::Serving)
        .await;
    health_reporter
        .set_serving::<OpStoreServiceServer<CommitCloudOpStoreService>>()
        .await;

    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;
    info!("jj-cc-server listening on {}", local_addr);

    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    let server_impl = CommitCloudServerImpl::default();
    let backend_service = CommitCloudBackendService::new(server_impl.store.clone());
    let op_store_service = CommitCloudOpStoreService::new(server_impl.store.clone());

    Server::builder()
        .add_service(health_service)
        .add_service(tonic::service::interceptor::InterceptedService::new(
            BackendServiceServer::new(backend_service),
            auth.clone(),
        ))
        .add_service(tonic::service::interceptor::InterceptedService::new(
            OpStoreServiceServer::new(op_store_service),
            auth,
        ))
        .serve_with_incoming_shutdown(incoming, async {
            match tokio::signal::ctrl_c().await {
                Ok(()) => {
                    info!("Shutdown signal received, shutting down gracefully...");
                }
                Err(e) => {
                    warn!(
                        "Failed to listen for shutdown signal: {}, shutting down...",
                        e
                    );
                }
            }
        })
        .await?;

    Ok(())
}
