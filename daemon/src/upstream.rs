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

//! Persistent upstream HTTP/2 channel pool and metadata header forwarding for jj-cc-daemon.
//!
//! # Authority and Crash-Safety Ordering
//! `UpstreamPool` holds stateless HTTP/2 transport channels keyed by normalized upstream
//! `server_url` strings; all persistent state authority remains on jj-cc-server and
//! [`crate::cache::TwoTierCache`].
//!
//! # Concurrency and Merge Invariants
//! Tonic [`Channel`] handles multiplex concurrent HTTP/2 streams without external locking
//! once connected; `channels` caches one connection per normalized target URL.
//!
//! # Cross-Process and RPC Boundaries
//! Inbound calls from cc-lib or jj-vfs pass `x-jj-cc-upstream` metadata so a single
//! per-user daemon routes across multiple local and cloud Commit Cloud endpoints while
//! forwarding each request's own `authorization` Bearer header. The daemon holds no
//! credentials of its own, so a request without a token reaches upstream without one.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use tonic::metadata::MetadataMap;
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};
use tonic::{Request, Status};

/// Request metadata header set by `cc-lib` when routing through a shared per-user daemon
/// so one local daemon can proxy multiple workspaces targeting distinct `server_url`s.
pub const UPSTREAM_HEADER: &str = "x-jj-cc-upstream";

pub struct UpstreamPool {
    default_upstream: Option<String>,
    channels: Mutex<HashMap<String, Channel>>,
}

impl UpstreamPool {
    pub fn new(default_upstream: Option<String>) -> Self {
        Self {
            default_upstream,
            channels: Mutex::new(HashMap::new()),
        }
    }

    pub fn resolve_target_url(&self, metadata: &MetadataMap) -> Result<String, Status> {
        if let Some(val) = metadata.get(UPSTREAM_HEADER) {
            let s = val
                .to_str()
                .map_err(|e| Status::invalid_argument(format!("invalid {UPSTREAM_HEADER}: {e}")))?
                .trim()
                .trim_end_matches('/');
            if !s.is_empty() {
                return Ok(s.to_string());
            }
        }
        if let Some(default_url) = &self.default_upstream {
            return Ok(default_url.trim().trim_end_matches('/').to_string());
        }
        Err(Status::invalid_argument(
            "missing target server URL (provide x-jj-cc-upstream metadata or --upstream)",
        ))
    }

    pub async fn channel_for(&self, metadata: &MetadataMap) -> Result<Channel, Status> {
        let target_url = self.resolve_target_url(metadata)?;
        if let Some(existing) = self.channels.lock().unwrap().get(&target_url).cloned() {
            return Ok(existing);
        }

        // gRPC servers reject keepalive pings sent more often than every 5 minutes (and any
        // ping on an idle connection) with GOAWAY `too_many_pings`.
        let mut endpoint = Endpoint::from_shared(target_url.clone())
            .map_err(|e| {
                Status::invalid_argument(format!("invalid upstream URL {target_url}: {e}"))
            })?
            .tcp_nodelay(true)
            .connect_timeout(Duration::from_secs(10))
            .http2_keep_alive_interval(Duration::from_secs(300))
            .keep_alive_timeout(Duration::from_secs(20));

        if target_url.starts_with("https://") {
            let tls = ClientTlsConfig::new().with_native_roots();
            endpoint = endpoint
                .tls_config(tls)
                .map_err(|e| Status::internal(format!("TLS init failed for {target_url}: {e}")))?;
        }

        let channel = endpoint.connect().await.map_err(|e| {
            Status::unavailable(format!("failed to connect upstream {target_url}: {e}"))
        })?;
        let mut guard = self.channels.lock().unwrap();
        Ok(guard
            .entry(target_url)
            .or_insert_with(|| channel.clone())
            .clone())
    }

    pub fn prepare_outbound<T>(&self, incoming_meta: &MetadataMap, payload: T) -> Request<T> {
        let mut req = Request::new(payload);
        if let Some(auth) = incoming_meta.get("authorization") {
            req.metadata_mut().insert("authorization", auth.clone());
        }
        req
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_outbound_forwards_only_the_callers_own_authorization() {
        let pool = UpstreamPool::new(Some("http://127.0.0.1:1".to_string()));
        let tokenless = pool.prepare_outbound(&MetadataMap::new(), ());
        assert!(tokenless.metadata().get("authorization").is_none());

        let mut incoming = MetadataMap::new();
        incoming.insert("authorization", "Bearer caller-token".parse().unwrap());
        incoming.insert(UPSTREAM_HEADER, "http://127.0.0.1:1".parse().unwrap());
        let forwarded = pool.prepare_outbound(&incoming, ());
        assert_eq!(
            forwarded
                .metadata()
                .get("authorization")
                .unwrap()
                .to_str()
                .unwrap(),
            "Bearer caller-token"
        );
        assert!(forwarded.metadata().get(UPSTREAM_HEADER).is_none());
    }
}
