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

//! Google Cloud Storage `BlobStore` implementation (`GcsBlobStore`).
//!
//! # Authority and Crash-Safety Ordering
//! `CompositeStore` uploads immutable content-addressed blobs via `GcsBlobStore::put_blob`
//! before publishing `BlobRecord` rows in `MetadataStore`. Objects use sharded keys
//! (`blobs/<aa>/<bb>/<full_hex>`) and `Cache-Control: public, max-age=31536000, immutable`
//! so Cloud CDN and LAN caches serve hydrated blobs without origin revalidation.
//!
//! # Concurrency and Merge Invariants
//! `put_blob` sets `if_generation_match(0)` on every upload and treats HTTP `412`
//! (`Code::FailedPrecondition`) as idempotent completion. When hundreds of concurrent
//! agents upload identical content, duplicate writes short-circuit without hitting
//! Cloud Storage's 1-write-per-second per-object overwrite limit.
//!
//! # Cross-Process and RPC Boundaries
//! Generic over `S: google_cloud_storage::stub::Storage` so production HTTP/gRPC
//! transports and deterministic test stubs exercise identical request construction,
//! byte-range `ReadRange` framing, and status-code translation.

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use google_cloud_auth::credentials::anonymous;
use google_cloud_gax::error::rpc::Code as RpcCode;
use google_cloud_storage::client::Storage;
use google_cloud_storage::model_ext::ReadRange;
use google_cloud_storage::stub::{DefaultStorage, Storage as StorageStub};

use super::{BlobStore, StoreError, StoreResult};

pub const IMMUTABLE_CACHE_CONTROL: &str = "public, max-age=31536000, immutable";

#[derive(Clone, Debug, Default)]
pub struct GcsConnectConfig {
    /// Bucket identifier (`my-bucket`, `gs://my-bucket`, or `projects/_/buckets/my-bucket`).
    pub bucket: String,
    /// Optional custom Cloud Storage endpoint (e.g. local emulator or VPC-SC endpoint).
    pub endpoint: Option<String>,
}

#[derive(Clone, Debug)]
pub struct GcsBlobStore<S = DefaultStorage>
where
    S: StorageStub + 'static,
{
    client: Storage<S>,
    bucket_resource: String,
    location_uri: String,
}

impl GcsBlobStore<DefaultStorage> {
    pub async fn connect(config: &GcsConnectConfig) -> StoreResult<Self> {
        let (bucket_resource, location_uri) = normalize_bucket_identifiers(&config.bucket)?;
        let mut builder = Storage::builder();

        let effective_endpoint = config
            .endpoint
            .clone()
            .or_else(|| std::env::var("STORAGE_EMULATOR_HOST").ok())
            .filter(|s| !s.trim().is_empty());

        if let Some(ep) = effective_endpoint {
            let normalized = if ep.starts_with("http://") || ep.starts_with("https://") {
                ep
            } else {
                format!("http://{ep}")
            };
            if let Some(rest) = normalized.strip_prefix("http://") {
                let host_port = rest.split('/').next().unwrap_or(rest);
                let bucket_id = location_uri.strip_prefix("gs://").unwrap_or(&config.bucket);
                ensure_emulator_bucket_http10(host_port, bucket_id).await?;
                builder = builder.with_credentials(anonymous::Builder::new().build());
            }
            builder = builder.with_endpoint(normalized);
        }

        let client = builder.build().await.map_err(|e| {
            StoreError::Unavailable(format!(
                "failed to initialize Cloud Storage client for {location_uri}: {e}"
            ))
        })?;

        Ok(Self {
            client,
            bucket_resource,
            location_uri,
        })
    }
}

impl<S> GcsBlobStore<S>
where
    S: StorageStub + 'static,
{
    pub fn from_storage(bucket: &str, client: Storage<S>) -> StoreResult<Self> {
        let (bucket_resource, location_uri) = normalize_bucket_identifiers(bucket)?;
        Ok(Self {
            client,
            bucket_resource,
            location_uri,
        })
    }

    pub fn object_key(blob_id: &[u8]) -> String {
        let hex_id = hex::encode(blob_id);
        let prefix_a = hex_id.get(0..2).unwrap_or("00");
        let prefix_b = hex_id.get(2..4).unwrap_or("00");
        format!("blobs/{prefix_a}/{prefix_b}/{hex_id}")
    }
}

pub fn normalize_bucket_identifiers(raw_bucket: &str) -> StoreResult<(String, String)> {
    let trimmed = raw_bucket.trim().trim_end_matches('/');
    let bucket_id = if let Some(rest) = trimmed.strip_prefix("gs://") {
        rest
    } else if let Some((_, b)) = trimmed.rsplit_once("/buckets/") {
        b
    } else {
        trimmed
    };

    if bucket_id.is_empty() || bucket_id.contains('/') {
        return Err(StoreError::InvalidArgument(format!(
            "invalid Cloud Storage bucket identifier '{raw_bucket}'"
        )));
    }

    Ok((
        format!("projects/_/buckets/{bucket_id}"),
        format!("gs://{bucket_id}"),
    ))
}

async fn ensure_emulator_bucket_http10(host_port: &str, bucket_id: &str) -> StoreResult<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(15);
    let mut stream = loop {
        match tokio::net::TcpStream::connect(host_port).await {
            Ok(s) => break s,
            Err(_e) if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            }
            Err(e) => {
                return Err(StoreError::Unavailable(format!(
                    "failed to connect to Cloud Storage emulator {host_port}: {e}"
                )));
            }
        }
    };

    let payload = serde_json::json!({ "name": bucket_id }).to_string();
    let req = format!(
        "POST /storage/v1/b?project=local-project HTTP/1.0\r\nHost: {host_port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
        payload.len()
    );
    stream.write_all(req.as_bytes()).await.map_err(|e| {
        StoreError::Unavailable(format!("failed writing to GCS emulator {host_port}: {e}"))
    })?;
    let mut raw = Vec::new();
    let _ = stream.read_to_end(&mut raw).await;
    Ok(())
}

fn is_precondition_failed(err: &google_cloud_gax::error::Error) -> bool {
    err.http_status_code().is_some_and(|code| code == 412)
        || err
            .status()
            .is_some_and(|s| s.code == RpcCode::FailedPrecondition)
}

fn is_not_found(err: &google_cloud_gax::error::Error) -> bool {
    err.http_status_code().is_some_and(|code| code == 404)
        || err.status().is_some_and(|s| s.code == RpcCode::NotFound)
}

fn is_out_of_range(err: &google_cloud_gax::error::Error) -> bool {
    err.http_status_code().is_some_and(|code| code == 416)
        || err.status().is_some_and(|s| s.code == RpcCode::OutOfRange)
}

fn map_gcs_err(err: google_cloud_gax::error::Error) -> StoreError {
    if is_out_of_range(&err) {
        return StoreError::OutOfRange(err.to_string());
    }
    if let Some(status) = err.status() {
        match status.code {
            RpcCode::InvalidArgument => return StoreError::InvalidArgument(err.to_string()),
            RpcCode::Unavailable | RpcCode::DeadlineExceeded => {
                return StoreError::Unavailable(err.to_string());
            }
            _ => {}
        }
    }
    if err
        .http_status_code()
        .is_some_and(|c| c == 429 || c == 502 || c == 503 || c == 504)
    {
        return StoreError::Unavailable(err.to_string());
    }
    StoreError::Backend(format!("Cloud Storage error: {err}"))
}

#[async_trait]
impl<S> BlobStore for GcsBlobStore<S>
where
    S: StorageStub + 'static,
{
    fn location_id(&self) -> &str {
        &self.location_uri
    }

    async fn put_blob(&self, blob_id: &[u8], data: Bytes) -> StoreResult<()> {
        let key = Self::object_key(blob_id);
        let res = self
            .client
            .write_object(&self.bucket_resource, key, data)
            .set_if_generation_match(0)
            .set_cache_control(IMMUTABLE_CACHE_CONTROL)
            .set_content_type("application/octet-stream")
            .send_buffered()
            .await;

        match res {
            Ok(_) => Ok(()),
            // Object already exists with identical content-addressed SHA-1 digest.
            Err(err) if is_precondition_failed(&err) => Ok(()),
            Err(err) => Err(map_gcs_err(err)),
        }
    }

    async fn get_blob(&self, blob_id: &[u8]) -> StoreResult<Option<Bytes>> {
        let key = Self::object_key(blob_id);
        let mut reader = match self
            .client
            .read_object(&self.bucket_resource, key)
            .send()
            .await
        {
            Ok(r) => r,
            Err(err) if is_not_found(&err) => return Ok(None),
            Err(err) => return Err(map_gcs_err(err)),
        };

        let mut buf = BytesMut::new();
        while let Some(chunk) = reader.next().await.transpose().map_err(map_gcs_err)? {
            buf.extend_from_slice(&chunk);
        }
        Ok(Some(buf.freeze()))
    }

    async fn get_blob_range(
        &self,
        blob_id: &[u8],
        offset: u64,
        length: Option<u64>,
    ) -> StoreResult<Option<Bytes>> {
        if offset == 0 && matches!(length, None | Some(0)) {
            return self.get_blob(blob_id).await;
        }

        let key = Self::object_key(blob_id);
        // The client builds the inclusive header `bytes=offset-(offset+length-1)` in i64
        // arithmetic, so read to the end when the requested span reaches past i64::MAX.
        let range = match length {
            Some(max_len) if max_len > 0 && offset.saturating_add(max_len) <= i64::MAX as u64 => {
                ReadRange::segment(offset, max_len)
            }
            _ => ReadRange::offset(offset),
        };

        let mut reader = match self
            .client
            .read_object(&self.bucket_resource, key)
            .set_read_range(range)
            .send()
            .await
        {
            Ok(r) => r,
            Err(err) if is_not_found(&err) => return Ok(None),
            Err(err) => return Err(map_gcs_err(err)),
        };

        let mut buf = BytesMut::new();
        while let Some(chunk) = reader.next().await.transpose().map_err(map_gcs_err)? {
            buf.extend_from_slice(&chunk);
        }
        Ok(Some(buf.freeze()))
    }
}
