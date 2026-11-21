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
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use tonic::transport::Server;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use cc_common::backend::backend_service_server::BackendServiceServer;
use cc_common::op_store::op_store_service_server::OpStoreServiceServer;
use cc_common::workspace::workspace_service_server::WorkspaceServiceServer;
use jj_commit_cloud_server::auth::BearerAuthInterceptor;
use jj_commit_cloud_server::backend::CommitCloudBackendService;
use jj_commit_cloud_server::op_store::CommitCloudOpStoreService;
use jj_commit_cloud_server::store::{
    BlobStore, CompositeStore, CompositeStoreConfig, GcsBlobStore, GcsConnectConfig,
    LocalFsBlobStore, MemoryBlobStore, MemoryMetadataStore, MetadataStore, ReplicatedBlobStore,
    ReplicatedBlobStoreConfig, SpannerConnectConfig, SpannerMetadataStore, SqliteMetadataStore,
};
use jj_commit_cloud_server::workspace::CommitCloudWorkspaceService;

/// Metadata storage backend to run the server against.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum StoreType {
    Memory,
    Sqlite,
    Spanner,
}

/// Content-addressed blob storage backend.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum BlobStoreType {
    Memory,
    Local,
    Gcs,
}

// No Debug derive: `auth_tokens` holds secrets.
#[derive(Parser)]
#[command(name = "jj-cc-server", about = "jj Commit Cloud Server")]
struct Args {
    /// Host interface to bind to
    #[arg(long, env = "HOST", default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on (use 0 for ephemeral port assignment)
    #[arg(short, long, env = "PORT", default_value_t = 8080)]
    port: u16,

    /// Metadata storage backend type
    #[arg(long, env = "STORE_TYPE", value_enum, default_value_t = StoreType::Memory)]
    store_type: StoreType,

    /// Blob storage backend type (auto-detected from store-type and bucket/path flags if omitted;
    /// required with --store-type=spanner unless --gcs-bucket or --blob-path is set)
    #[arg(long, env = "BLOB_STORE", value_enum)]
    blob_store: Option<BlobStoreType>,

    /// Maximum blob size in bytes stored inline inside MetadataStore (0 disables inlining)
    #[arg(long, env = "INLINE_BLOB_THRESHOLD_BYTES", default_value_t = 0)]
    inline_blob_threshold_bytes: usize,

    /// Path to SQLite database file (defaults to ~/.jj-cc-server/commit_cloud.db)
    #[arg(long, env = "SQLITE_PATH")]
    sqlite_path: Option<PathBuf>,

    /// Root directory for local content-addressed blob storage
    #[arg(long, env = "BLOB_PATH")]
    blob_path: Option<PathBuf>,

    /// Optional logical location URI override for local blob store (e.g. gs://region-a)
    #[arg(long, env = "BLOB_LOCATION_ID")]
    blob_location_id: Option<String>,

    /// Optional peer local blob directories formatted as `<location_uri>=<dir_path>`
    #[arg(long, env = "LOCAL_PEER_BLOB_DIRS", value_delimiter = ',')]
    local_peer_blob_dirs: Vec<String>,

    /// Cloud Spanner database resource path (projects/<p>/instances/<i>/databases/<d>)
    #[arg(long, visible_alias = "spanner-db", env = "SPANNER_DATABASE")]
    spanner_database: Option<String>,

    /// Optional Spanner gRPC endpoint override (e.g. Spanner Omni or custom emulator URL)
    #[arg(long, env = "SPANNER_ENDPOINT")]
    spanner_endpoint: Option<String>,

    /// Optional Spanner HTTP/REST admin endpoint override for DDL auto-migration
    #[arg(long, env = "SPANNER_EMULATOR_REST_HOST")]
    spanner_admin_endpoint: Option<String>,

    /// Automatically create missing Spanner database and apply spanner_schema.sql on startup
    #[arg(long, env = "SPANNER_AUTO_MIGRATE")]
    spanner_auto_migrate: bool,

    /// Connect to a containerized Spanner Omni endpoint instead of managed Cloud Spanner
    #[arg(long)]
    spanner_omni: bool,

    /// Path to custom Root CA PEM file for Spanner Omni TLS verification
    #[arg(long)]
    spanner_omni_ca_cert: Option<PathBuf>,

    /// Path to client certificate PEM file for Spanner Omni mTLS authentication
    #[arg(long)]
    spanner_omni_client_cert: Option<PathBuf>,

    /// Path to client private key PEM file for Spanner Omni mTLS authentication
    #[arg(long)]
    spanner_omni_client_key: Option<PathBuf>,

    /// Primary Google Cloud Storage bucket (`my-bucket` or `gs://my-bucket`)
    #[arg(long, env = "GCS_BUCKET")]
    gcs_bucket: Option<String>,

    /// Optional comma-separated peer Cloud Storage buckets (`gs://peer` or `gs://peer=http://endpoint`)
    #[arg(long, env = "GCS_PEER_BUCKETS", value_delimiter = ',')]
    gcs_peer_buckets: Vec<String>,

    /// Optional Cloud Storage endpoint override (e.g. local emulator or VPC-SC endpoint)
    #[arg(long, env = "STORAGE_EMULATOR_HOST")]
    gcs_endpoint: Option<String>,

    /// In-memory read-through LRU blob cache size in bytes (0 disables caching). Setting a
    /// non-zero size, or configuring peer blob stores, wraps the blob store in ReplicatedBlobStore
    /// (default cache size with peers: 64 MiB; default for a single store: 0)
    #[arg(long)]
    blob_cache_bytes: Option<usize>,

    /// Retry window in milliseconds for metadata-confirmed blobs that no blob store returns yet
    /// (0 disables). Cloud Storage provides read-after-write consistency within a bucket, and peer
    /// fallback covers lag between buckets, so only object stores without read-after-write
    /// consistency need this
    #[arg(long, default_value_t = 0)]
    gcs_replication_retry_ms: u64,

    /// Replicate uploaded blobs asynchronously to configured peer buckets in the background
    #[arg(long)]
    gcs_async_write_fanout: bool,

    /// Optional Bearer token(s) required on gRPC requests (comma-separated; health checks stay
    /// open)
    #[arg(
        long = "auth-token",
        env = "JJ_CC_SERVER_AUTH_TOKEN",
        value_delimiter = ',',
        hide_env_values = true
    )]
    auth_tokens: Vec<String>,
}

const DEFAULT_BLOB_CACHE_BYTES: usize = 64 * 1024 * 1024;

fn default_server_data_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".jj-cc-server")
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
    // A secret that renders as "" or " , " must not leave the server open without a trace.
    let cli_auth_flag = std::env::args_os().any(|a| {
        let s = a.to_string_lossy();
        s == "--auth-token" || s.starts_with("--auth-token=")
    });
    let auth_requested = !args.auth_tokens.is_empty()
        || cli_auth_flag
        || std::env::var_os("JJ_CC_SERVER_AUTH_TOKEN").is_some();
    if auth_requested && !auth.is_enabled() {
        return Err(
            "--auth-token (or JJ_CC_SERVER_AUTH_TOKEN) is set but contains no non-empty \
                    token; unset it to run without authentication"
                .into(),
        );
    }
    if auth.is_enabled() {
        info!("Bearer token authentication enabled for gRPC services");
    } else {
        warn!("Bearer token authentication disabled: gRPC services accept any caller");
    }

    let resolved_sqlite_path = args
        .sqlite_path
        .clone()
        .unwrap_or_else(|| default_server_data_dir().join("commit_cloud.db"));
    let has_gcs_bucket = args
        .gcs_bucket
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty());
    let has_blob_path = args
        .blob_path
        .as_ref()
        .is_some_and(|p| !p.as_os_str().is_empty());

    let metadata_store: Arc<dyn MetadataStore> = match args.store_type {
        StoreType::Memory => Arc::new(MemoryMetadataStore::new()),
        StoreType::Sqlite => {
            info!(
                "Opening SQLite metadata store at {}",
                resolved_sqlite_path.display()
            );
            Arc::new(SqliteMetadataStore::open(&resolved_sqlite_path)?)
        }
        StoreType::Spanner => {
            let database = args.spanner_database.clone().ok_or(
                "--spanner-database (or SPANNER_DATABASE) is required when --store-type=spanner",
            )?;
            // Durable metadata over a silent in-memory blob default would lose file contents
            // on restart while commits and trees still reference them.
            if args.blob_store.is_none() && !has_gcs_bucket && !has_blob_path {
                return Err("--store-type=spanner requires an explicit blob store: set \
                            --gcs-bucket (GCS_BUCKET) or --blob-path (BLOB_PATH), or pass \
                            --blob-store=memory (BLOB_STORE=memory) for throwaway tests"
                    .into());
            }
            info!("Connecting to Cloud Spanner database '{database}'");
            let cfg = SpannerConnectConfig {
                database,
                endpoint: args.spanner_endpoint.clone(),
                admin_endpoint: args.spanner_admin_endpoint.clone(),
                omni: args.spanner_omni,
                omni_ca_cert: args.spanner_omni_ca_cert.clone(),
                omni_client_cert: args.spanner_omni_client_cert.clone(),
                omni_client_key: args.spanner_omni_client_key.clone(),
                auto_migrate: args.spanner_auto_migrate,
            };
            Arc::new(SpannerMetadataStore::connect(&cfg).await?)
        }
    };

    let effective_blob_store = args.blob_store.unwrap_or(match args.store_type {
        StoreType::Memory => {
            if has_gcs_bucket {
                BlobStoreType::Gcs
            } else {
                BlobStoreType::Memory
            }
        }
        StoreType::Sqlite => BlobStoreType::Local,
        // Spanner startup rejects a missing --blob-store, --gcs-bucket, and --blob-path above, so
        // this default only matters when --gcs-bucket or --blob-path is set.
        StoreType::Spanner => {
            if has_gcs_bucket {
                BlobStoreType::Gcs
            } else {
                BlobStoreType::Local
            }
        }
    });
    if effective_blob_store == BlobStoreType::Memory && args.store_type != StoreType::Memory {
        warn!("Blob payloads are kept in memory and are lost on restart");
    }

    let primary_blob_store: Arc<dyn BlobStore> = match effective_blob_store {
        BlobStoreType::Memory => match &args.blob_location_id {
            Some(loc) => Arc::new(MemoryBlobStore::with_location(loc.clone())),
            None => Arc::new(MemoryBlobStore::new()),
        },
        BlobStoreType::Local => {
            let blob_root = args.blob_path.clone().unwrap_or_else(|| {
                if args.store_type == StoreType::Sqlite {
                    let mut p = resolved_sqlite_path.as_os_str().to_os_string();
                    p.push(".blobs");
                    PathBuf::from(p)
                } else {
                    default_server_data_dir().join("blobs")
                }
            });
            info!(
                "Opening local filesystem blob store at {}",
                blob_root.display()
            );
            match &args.blob_location_id {
                Some(loc) => Arc::new(LocalFsBlobStore::open_with_location_id(&blob_root, loc)?),
                None => Arc::new(LocalFsBlobStore::open(&blob_root)?),
            }
        }
        BlobStoreType::Gcs => {
            let bucket = args
                .gcs_bucket
                .clone()
                .ok_or("--gcs-bucket (or GCS_BUCKET) is required when --blob-store=gcs")?;
            info!("Connecting to Cloud Storage bucket '{bucket}'");
            Arc::new(
                GcsBlobStore::connect(&GcsConnectConfig {
                    bucket,
                    endpoint: args.gcs_endpoint.clone(),
                })
                .await?,
            )
        }
    };

    let mut peer_stores: Vec<Arc<dyn BlobStore>> = Vec::new();
    for spec in args
        .gcs_peer_buckets
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        let (bucket, endpoint) = match spec.split_once('=') {
            Some((b, ep)) => (b.to_string(), Some(ep.to_string())),
            None => (spec.to_string(), args.gcs_endpoint.clone()),
        };
        info!("Registering peer Cloud Storage bucket '{bucket}'");
        peer_stores.push(Arc::new(
            GcsBlobStore::connect(&GcsConnectConfig { bucket, endpoint }).await?,
        ));
    }
    for spec in args
        .local_peer_blob_dirs
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        let (loc_uri, dir_str) = spec.split_once('=').ok_or_else(|| {
            format!("invalid --local-peer-blob-dirs entry '{spec}' (expected <uri>=<path>)")
        })?;
        info!("Registering peer local blob directory '{loc_uri}' -> '{dir_str}'");
        peer_stores.push(Arc::new(LocalFsBlobStore::open_with_location_id(
            dir_str, loc_uri,
        )?));
    }

    // The replication layer serves multi-bucket deployments; a single blob store reads directly
    // (including native ranged reads) unless a non-zero cache or retry window is requested.
    let cache_capacity_bytes = args.blob_cache_bytes.unwrap_or(if peer_stores.is_empty() {
        0
    } else {
        DEFAULT_BLOB_CACHE_BYTES
    });
    let blob_store: Arc<dyn BlobStore> =
        if !peer_stores.is_empty() || cache_capacity_bytes > 0 || args.gcs_replication_retry_ms > 0
        {
            Arc::new(ReplicatedBlobStore::new(
                primary_blob_store,
                peer_stores,
                metadata_store.clone(),
                ReplicatedBlobStoreConfig {
                    cache_capacity_bytes,
                    replication_retry_budget: Duration::from_millis(args.gcs_replication_retry_ms),
                    initial_retry_backoff: Duration::from_millis(20),
                    async_write_fanout: args.gcs_async_write_fanout,
                },
            ))
        } else {
            primary_blob_store
        };

    let store = Arc::new(CompositeStore::with_config(
        metadata_store,
        blob_store,
        CompositeStoreConfig {
            inline_blob_threshold: args.inline_blob_threshold_bytes,
            validate_dependencies: true,
        },
    ));

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_service_status("", tonic_health::ServingStatus::Serving)
        .await;
    health_reporter
        .set_serving::<BackendServiceServer<CommitCloudBackendService>>()
        .await;
    health_reporter
        .set_serving::<OpStoreServiceServer<CommitCloudOpStoreService>>()
        .await;
    health_reporter
        .set_serving::<WorkspaceServiceServer<CommitCloudWorkspaceService>>()
        .await;

    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;
    info!("jj-cc-server listening on {}", local_addr);

    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    let backend_service = CommitCloudBackendService::new(store.clone());
    let op_store_service = CommitCloudOpStoreService::new(store.clone());
    let workspace_service = CommitCloudWorkspaceService::new(store);

    Server::builder()
        .add_service(health_service)
        .add_service(tonic::service::interceptor::InterceptedService::new(
            BackendServiceServer::new(backend_service)
                .max_decoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES),
            auth.clone(),
        ))
        .add_service(tonic::service::interceptor::InterceptedService::new(
            OpStoreServiceServer::new(op_store_service)
                .max_decoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES),
            auth.clone(),
        ))
        .add_service(tonic::service::interceptor::InterceptedService::new(
            WorkspaceServiceServer::new(workspace_service)
                .max_decoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES),
            auth,
        ))
        .serve_with_incoming_shutdown(incoming, shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("installing SIGTERM handler should succeed");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("SIGINT received, shutting down...");
            }
            _ = sigterm.recv() => {
                info!("SIGTERM received, shutting down...");
            }
        }
    }
    #[cfg(not(unix))]
    {
        match tokio::signal::ctrl_c().await {
            Ok(()) => info!("Shutdown signal received, shutting down..."),
            Err(e) => tracing::warn!("Failed to listen for shutdown signal: {e}, shutting down..."),
        }
    }
}
