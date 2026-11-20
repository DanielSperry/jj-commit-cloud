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

//! Shared Tokio runtime bridge, pooled HTTP/2 gRPC channel cache, and client
//! configuration loader for `cc-lib`.
//!
//! # Authority and Crash-Safety Ordering
//! Repository connection configuration (`server_url` and `repo_id`) is persisted in
//! `.jj/repo/store/config.toml` only after remote registration or existence verification
//! succeeds.
//!
//! # Concurrency and Merge Invariants
//! [`SHARED_RUNTIME`] hosts a single process-wide multi-thread Tokio reactor, and
//! [`CHANNEL_POOL`] caches one multiplexed HTTP/2 [`Channel`] per target endpoint so
//! concurrent `Backend`, `OpStore`, and `OpHeadsStore` calls pipeline without spawning
//! per-RPC OS threads or repeating TCP/TLS handshakes.
//!
//! # Cross-Process and RPC Boundaries
//! [`run_on_tokio`] bridges `jj_lib`'s executor-agnostic async traits onto
//! [`SHARED_RUNTIME`] via non-blocking `JoinHandle` awaits, while [`run_async`] bridges
//! synchronous initialization entry points safely from any thread context.

use std::collections::HashMap;
use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use cc_common::backend::backend_service_client::BackendServiceClient;
use cc_common::op_store::op_store_service_client::OpStoreServiceClient;
use cc_common::workspace::workspace_service_client::WorkspaceServiceClient;
use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};
use tonic::{Request, Status};

const UPSTREAM_HEADER: &str = "x-jj-cc-upstream";
const DAEMON_SECRET_HEADER: &str = "x-jj-cc-daemon-secret";

#[derive(Clone, Debug, Default)]
pub struct ClientInterceptor {
    daemon_upstream: Option<MetadataValue<tonic::metadata::Ascii>>,
    daemon_secret: Option<MetadataValue<tonic::metadata::Ascii>>,
}

impl ClientInterceptor {
    pub fn direct() -> Self {
        Self {
            daemon_upstream: None,
            daemon_secret: None,
        }
    }

    pub fn via_daemon(
        upstream_url: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::via_daemon_with_secret(upstream_url, None)
    }

    pub fn via_daemon_with_secret(
        upstream_url: &str,
        secret: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let val = MetadataValue::try_from(upstream_url)?;
        let daemon_secret = match secret {
            Some(s) if !s.is_empty() => {
                let mut sv = MetadataValue::try_from(s)?;
                sv.set_sensitive(true);
                Some(sv)
            }
            _ => None,
        };
        Ok(Self {
            daemon_upstream: Some(val),
            daemon_secret,
        })
    }
}

impl Interceptor for ClientInterceptor {
    fn call(&mut self, mut req: Request<()>) -> Result<Request<()>, Status> {
        if let Some(up) = &self.daemon_upstream {
            req.metadata_mut().insert(UPSTREAM_HEADER, up.clone());
        }
        if let Some(secret) = &self.daemon_secret {
            req.metadata_mut()
                .insert(DAEMON_SECRET_HEADER, secret.clone());
        }
        attach_bearer_token(req)
    }
}

pub type AuthBackendClient = BackendServiceClient<InterceptedService<Channel, ClientInterceptor>>;
pub type AuthOpStoreClient = OpStoreServiceClient<InterceptedService<Channel, ClientInterceptor>>;
pub type AuthWorkspaceClient =
    WorkspaceServiceClient<InterceptedService<Channel, ClientInterceptor>>;

static SHARED_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("jj-cc-client")
        .enable_all()
        .build()
        .expect("shared Commit Cloud Tokio runtime should initialize")
});

static CHANNEL_POOL: LazyLock<Mutex<HashMap<String, Channel>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn attach_bearer_token_from_opt(
    mut req: Request<()>,
    raw_token: Option<&str>,
) -> Result<Request<()>, Status> {
    if let Some(token) = raw_token {
        let trimmed = token.trim();
        if !trimmed.is_empty() {
            let header_val = MetadataValue::try_from(format!("Bearer {trimmed}"))
                .map_err(|e| Status::invalid_argument(format!("invalid JJ_CC_AUTH_TOKEN: {e}")))?;
            req.metadata_mut().insert("authorization", header_val);
        }
    }
    Ok(req)
}

pub fn attach_bearer_token(req: Request<()>) -> Result<Request<()>, Status> {
    let env_val = std::env::var("JJ_CC_AUTH_TOKEN").ok();
    attach_bearer_token_from_opt(req, env_val.as_deref())
}

/// Connects or reuses a pooled HTTP/2 gRPC [`Channel`] for `server_url`.
/// Must be awaited inside [`SHARED_RUNTIME`] (via [`run_on_tokio`] or [`run_async`]).
pub async fn connect_channel(
    server_url: impl Into<String>,
) -> Result<Channel, Box<dyn std::error::Error + Send + Sync>> {
    let url_str = server_url.into();
    if let Some(existing) = CHANNEL_POOL.lock().unwrap().get(&url_str).cloned() {
        return Ok(existing);
    }

    // gRPC servers enforce a default 5-minute minimum ping interval and reject pings when no
    // streams are active (`too_many_pings`), so omit `keep_alive_while_idle`.
    let mut endpoint = Endpoint::from_shared(url_str.clone())?
        .tcp_nodelay(true)
        .connect_timeout(Duration::from_secs(10))
        .http2_keep_alive_interval(Duration::from_secs(300))
        .keep_alive_timeout(Duration::from_secs(20));

    if url_str.starts_with("https://") {
        let tls = ClientTlsConfig::new().with_native_roots();
        endpoint = endpoint.tls_config(tls)?;
    }

    let channel = endpoint.connect().await?;
    let mut pool = CHANNEL_POOL.lock().unwrap();
    Ok(pool
        .entry(url_str)
        .or_insert_with(|| channel.clone())
        .clone())
}

/// Removes any cached HTTP/2 channel for `server_url` from the process-wide pool.
pub fn evict_cached_channel(server_url: &str) {
    CHANNEL_POOL.lock().unwrap().remove(server_url);
}

/// Returns the number of distinct endpoints currently cached in [`CHANNEL_POOL`].
pub fn cached_channel_count() -> usize {
    CHANNEL_POOL.lock().unwrap().len()
}

async fn connect_routed_channel(
    server_url: impl Into<String>,
) -> Result<(Channel, ClientInterceptor), Box<dyn std::error::Error + Send + Sync>> {
    let channel = connect_channel(server_url).await?;
    Ok((channel, ClientInterceptor::direct()))
}

pub async fn connect_backend_client(
    server_url: impl Into<String>,
) -> Result<AuthBackendClient, Box<dyn std::error::Error + Send + Sync>> {
    let (channel, interceptor) = connect_routed_channel(server_url).await?;
    Ok(BackendServiceClient::with_interceptor(channel, interceptor)
        .max_decoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES)
        .max_encoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES))
}

pub async fn connect_op_store_client(
    server_url: impl Into<String>,
) -> Result<AuthOpStoreClient, Box<dyn std::error::Error + Send + Sync>> {
    let (channel, interceptor) = connect_routed_channel(server_url).await?;
    Ok(OpStoreServiceClient::with_interceptor(channel, interceptor)
        .max_decoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES)
        .max_encoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES))
}

pub async fn connect_workspace_client(
    server_url: impl Into<String>,
) -> Result<AuthWorkspaceClient, Box<dyn std::error::Error + Send + Sync>> {
    let (channel, interceptor) = connect_routed_channel(server_url).await?;
    Ok(
        WorkspaceServiceClient::with_interceptor(channel, interceptor)
            .max_decoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES)
            .max_encoding_message_size(cc_common::MAX_GRPC_MESSAGE_BYTES),
    )
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommitCloudConfig {
    pub server_url: String,
    pub repo_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
}

impl CommitCloudConfig {
    pub fn effective_project_id(&self) -> &str {
        self.project_id.as_deref().unwrap_or(&self.repo_id)
    }

    pub fn save_to_store(
        &self,
        store_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let content = toml::to_string_pretty(self)?;
        if let Some(existing) = Self::resolve_config_path(store_path) {
            fs::write(existing, content)?;
            return Ok(());
        }
        let target = store_path.join("config.toml");
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(target, content)?;
        Ok(())
    }
    /// Resolves `.jj/repo/store/config.toml` from `store_path`, sibling `.jj/repo/*`
    /// directories, workspace roots, or secondary `jj workspace add` repo pointer files.
    pub fn resolve_config_path(store_path: &Path) -> Option<PathBuf> {
        let direct = store_path.join("config.toml");
        if direct.is_file() {
            return Some(direct);
        }
        if let Some(parent) = store_path.parent() {
            let sibling_store = parent.join("store").join("config.toml");
            if sibling_store.is_file() {
                return Some(sibling_store);
            }
        }

        let mut curr = store_path.to_path_buf();
        for _ in 0..5 {
            let dot_jj_repo = curr.join(".jj").join("repo");
            if dot_jj_repo.is_dir() {
                let candidate = dot_jj_repo.join("store").join("config.toml");
                if candidate.is_file() {
                    return Some(candidate);
                }
            } else if dot_jj_repo.is_file()
                && let Ok(ptr) = fs::read_to_string(&dot_jj_repo)
            {
                let target_repo = PathBuf::from(ptr.trim());
                let resolved = if target_repo.is_absolute() {
                    target_repo
                } else {
                    curr.join(".jj").join(ptr.trim())
                };
                let candidate = resolved.join("store").join("config.toml");
                if candidate.is_file() {
                    return Some(candidate);
                }
            }

            let store_candidate = curr.join("store").join("config.toml");
            if store_candidate.is_file() {
                return Some(store_candidate);
            }
            if !curr.pop() {
                break;
            }
        }
        None
    }

    pub fn load_from_store(
        store_path: &Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_path = Self::resolve_config_path(store_path).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "store path '{}' should have contained config.toml",
                    store_path.display()
                ),
            )
        })?;
        let content = fs::read_to_string(&config_path)?;
        Ok(toml::from_str(&content)?)
    }
}

/// Schedules `fut` on the process-wide shared multi-thread Tokio runtime and awaits
/// completion asynchronously without spawning OS threads or blocking `pollster`.
pub async fn run_on_tokio<Fut, T>(fut: Fut) -> Result<T, Box<dyn std::error::Error + Send + Sync>>
where
    Fut: Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>> + Send + 'static,
    T: Send + 'static,
{
    SHARED_RUNTIME
        .spawn(fut)
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            format!("Tokio client task join error: {e}").into()
        })?
}

/// Synchronous bridge for non-async initialization entry points.
pub fn run_async<F, Fut, T>(f: F) -> Result<T, Box<dyn std::error::Error + Send + Sync>>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    SHARED_RUNTIME.spawn(async move {
        let _ = tx.send(f().await);
    });
    let received = match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| rx.recv())
        }
        _ => rx.recv(),
    };
    received.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
        format!("client task channel closed: {e}").into()
    })?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_interceptor_passes_through_and_formats_token() {
        let req = attach_bearer_token_from_opt(Request::new(()), Some("  test-id-token  "))
            .expect("valid token");
        assert_eq!(
            req.metadata()
                .get("authorization")
                .and_then(|v| v.to_str().ok()),
            Some("Bearer test-id-token")
        );

        let passthrough = attach_bearer_token_from_opt(Request::new(()), None).expect("none");
        assert!(passthrough.metadata().get("authorization").is_none());
    }

    #[tokio::test]
    async fn shared_runtime_executes_from_inside_tokio_and_sync_contexts() {
        let async_val =
            run_on_tokio(async { Ok::<_, Box<dyn std::error::Error + Send + Sync>>(42) })
                .await
                .expect("run_on_tokio should succeed inside #[tokio::test]");
        assert_eq!(async_val, 42);

        let sync_val = run_async(|| async { Ok(99) })
            .expect("run_async should not panic inside current_thread #[tokio::test]");
        assert_eq!(sync_val, 99);
    }

    #[test]
    fn client_interceptor_attaches_upstream_and_daemon_secret_headers() {
        let mut ic =
            ClientInterceptor::via_daemon_with_secret("http://127.0.0.1:8080", Some("s3cr3t"))
                .unwrap();
        let req = ic.call(Request::new(())).unwrap();
        assert_eq!(
            req.metadata()
                .get(UPSTREAM_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("http://127.0.0.1:8080")
        );
        assert_eq!(
            req.metadata()
                .get(DAEMON_SECRET_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("s3cr3t")
        );
        assert!(
            !format!("{ic:?}").contains("s3cr3t"),
            "sensitive secret must be redacted in Debug"
        );
    }
}
