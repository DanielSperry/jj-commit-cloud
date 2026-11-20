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

use cc_common::backend::backend_service_server::BackendServiceServer;
use cc_common::op_store::op_store_service_server::OpStoreServiceServer;
use cc_common::workspace::workspace_service_server::WorkspaceServiceServer;
use clap::Parser;
use jj_commit_cloud_server::backend::CommitCloudBackendService;
use jj_commit_cloud_server::op_store::CommitCloudOpStoreService;
use jj_commit_cloud_server::store::{
    BlobStore, CompositeStore, CompositeStoreConfig, MemoryBlobStore, MemoryMetadataStore,
    MetadataStore,
};
use jj_commit_cloud_server::workspace::CommitCloudWorkspaceService;
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "jj-cc-server", about = "jj Commit Cloud Server")]
struct Args {
    #[arg(long, env = "HOST", default_value = "127.0.0.1")]
    host: String,
    #[arg(short, long, env = "PORT", default_value_t = 8080)]
    port: u16,
    #[arg(long, env = "INLINE_BLOB_THRESHOLD_BYTES", default_value_t = 0)]
    inline_blob_threshold_bytes: usize,
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

    let metadata_store: Arc<dyn MetadataStore> = Arc::new(MemoryMetadataStore::new());
    let blob_store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());

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
        .add_service(
            BackendServiceServer::new(backend_service)
                .max_decoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES),
        )
        .add_service(
            OpStoreServiceServer::new(op_store_service)
                .max_decoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES),
        )
        .add_service(
            WorkspaceServiceServer::new(workspace_service)
                .max_decoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES),
        )
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
