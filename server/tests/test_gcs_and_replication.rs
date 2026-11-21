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

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};
use google_cloud_gax::error::Error as GaxError;
use google_cloud_gax::error::rpc::{Code as RpcCode, Status as RpcStatus};
use google_cloud_storage::client::Storage;
use google_cloud_storage::model::{Object, ReadObjectRequest};
use google_cloud_storage::model_ext::{ObjectHighlights, WriteObjectRequest};
use google_cloud_storage::read_object::ReadObjectResponse;
use google_cloud_storage::request_options::RequestOptions;
use google_cloud_storage::streaming_source::StreamingSource;
use jj_commit_cloud_server::store::{
    BlobStore, CompositeStore, GcsBlobStore, IMMUTABLE_CACHE_CONTROL, MemoryMetadataStore,
    MetadataStore, ReplicatedBlobStore, ReplicatedBlobStoreConfig, SpannerConnectConfig,
    SpannerMetadataStore, StoreError,
};

#[derive(Clone, Debug)]
struct StoredObject {
    data: Bytes,
    visible_at: Instant,
    cache_control: String,
    generation: i64,
}

/// Deterministic in-memory `google_cloud_storage::stub::Storage` implementation enforcing
/// `if_generation_match` preconditions, byte-range slicing, and configurable visibility lag.
#[derive(Debug, Default)]
struct FakeGcsStub {
    objects: Mutex<HashMap<(String, String), StoredObject>>,
    visibility_delay: Duration,
    read_delay: Duration,
    write_delay: Duration,
    injected_read_error: Mutex<Option<RpcCode>>,
    read_calls: AtomicUsize,
    bytes_read: AtomicUsize,
    write_calls: AtomicUsize,
    precondition_412_hits: AtomicUsize,
}

impl FakeGcsStub {
    fn with_delays(visibility_delay: Duration, read_delay: Duration) -> Self {
        Self {
            visibility_delay,
            read_delay,
            ..Default::default()
        }
    }
}

impl google_cloud_storage::stub::Storage for FakeGcsStub {
    async fn read_object(
        &self,
        req: ReadObjectRequest,
        _options: RequestOptions,
    ) -> google_cloud_storage::Result<ReadObjectResponse> {
        self.read_calls.fetch_add(1, Ordering::SeqCst);
        if !self.read_delay.is_zero() {
            tokio::time::sleep(self.read_delay).await;
        }
        if let Some(code) = *self.injected_read_error.lock().unwrap() {
            return Err(GaxError::service(
                RpcStatus::default()
                    .set_code(code)
                    .set_message("injected read failure"),
            ));
        }

        // Mirror google-cloud-storage's HTTP `bytes=offset-(offset+limit-1)` i64 arithmetic.
        if req.read_offset >= 0 && req.read_limit > 0 {
            assert!(
                req.read_offset.checked_add(req.read_limit - 1).is_some(),
                "unclamped ReadRange overflowed i64: offset={}, limit={}",
                req.read_offset,
                req.read_limit
            );
        }

        let entry = {
            let guard = self.objects.lock().unwrap();
            guard
                .get(&(req.bucket.clone(), req.object.clone()))
                .cloned()
        };

        let Some(stored) = entry else {
            return Err(GaxError::service(
                RpcStatus::default()
                    .set_code(RpcCode::NotFound)
                    .set_message("object not found"),
            ));
        };

        if Instant::now() < stored.visible_at {
            return Err(GaxError::service(
                RpcStatus::default()
                    .set_code(RpcCode::NotFound)
                    .set_message("object replication still in flight"),
            ));
        }

        let total = stored.data.len() as i64;
        let start = if req.read_offset < 0 {
            (total + req.read_offset).max(0) as usize
        } else {
            req.read_offset as usize
        };
        if total > 0 && start >= stored.data.len() {
            return Err(GaxError::service(
                RpcStatus::default()
                    .set_code(RpcCode::OutOfRange)
                    .set_message("read offset out of range"),
            ));
        }

        let end = if req.read_limit > 0 {
            start
                .saturating_add(req.read_limit as usize)
                .min(stored.data.len())
        } else {
            stored.data.len()
        };

        let slice = stored.data.slice(start..end);
        self.bytes_read.fetch_add(slice.len(), Ordering::SeqCst);
        let mut highlights = ObjectHighlights::default();
        highlights.generation = stored.generation;
        highlights.size = total;
        Ok(ReadObjectResponse::from_source(highlights, slice))
    }

    async fn write_object_buffered<P>(
        &self,
        mut payload: P,
        req: WriteObjectRequest,
        _options: RequestOptions,
    ) -> google_cloud_storage::Result<Object>
    where
        P: StreamingSource + Send + Sync + 'static,
    {
        self.write_calls.fetch_add(1, Ordering::SeqCst);
        if !self.write_delay.is_zero() {
            tokio::time::sleep(self.write_delay).await;
        }
        let mut buf = BytesMut::new();
        while let Some(chunk) = payload.next().await {
            let chunk = chunk.map_err(|e| {
                GaxError::service(
                    RpcStatus::default()
                        .set_code(RpcCode::Internal)
                        .set_message(format!("payload stream error: {e}")),
                )
            })?;
            buf.extend_from_slice(&chunk);
        }
        let data = buf.freeze();

        let resource = req.spec.resource.unwrap_or_default();
        let bucket = resource.bucket.clone();
        let name = resource.name.clone();
        let cache_control = resource.cache_control.clone();

        let mut guard = self.objects.lock().unwrap();
        if req.spec.if_generation_match == Some(0)
            && guard.contains_key(&(bucket.clone(), name.clone()))
        {
            self.precondition_412_hits.fetch_add(1, Ordering::SeqCst);
            return Err(GaxError::service(
                RpcStatus::default()
                    .set_code(RpcCode::FailedPrecondition)
                    .set_message("412 Precondition Failed: if_generation_match=0"),
            ));
        }

        let next_gen = guard
            .get(&(bucket.clone(), name.clone()))
            .map_or(1, |prev| prev.generation + 1);
        guard.insert(
            (bucket, name),
            StoredObject {
                data,
                visible_at: Instant::now() + self.visibility_delay,
                cache_control,
                generation: next_gen,
            },
        );
        Ok(resource)
    }
}

#[tokio::test]
async fn test_gcs_blob_store_conditional_writes_412_idempotency_and_range_reads() {
    let stub = Arc::new(FakeGcsStub::default());
    let gcs_client = Storage::<FakeGcsStub>::from_stub(stub.clone());
    let store = GcsBlobStore::from_storage("gs://jj-blobs-uc1", gcs_client).unwrap();
    assert_eq!(store.location_id(), "gs://jj-blobs-uc1");

    let blob_id = hex::decode("ab12cd3400000000000000000000000000000099").unwrap();
    let payload = Bytes::from_static(b"immutable content-addressed blob payload");

    // First upload writes object with if_generation_match=0 and immutable Cache-Control.
    store.put_blob(&blob_id, payload.clone()).await.unwrap();
    assert_eq!(stub.write_calls.load(Ordering::SeqCst), 1);
    assert_eq!(stub.precondition_412_hits.load(Ordering::SeqCst), 0);

    {
        let guard = stub.objects.lock().unwrap();
        let stored = guard
            .get(&(
                "projects/_/buckets/jj-blobs-uc1".to_string(),
                "blobs/ab/12/ab12cd3400000000000000000000000000000099".to_string(),
            ))
            .expect("sharded object key should exist in bucket");
        assert_eq!(stored.cache_control, IMMUTABLE_CACHE_CONTROL);
    }

    // Duplicate upload hits HTTP 412 / FailedPrecondition and succeeds idempotently!
    store.put_blob(&blob_id, payload.clone()).await.unwrap();
    assert_eq!(stub.precondition_412_hits.load(Ordering::SeqCst), 1);

    // Full read and ranged read.
    assert_eq!(store.get_blob(&blob_id).await.unwrap(), Some(payload));
    assert_eq!(
        store
            .get_blob_range(&blob_id, 10, Some(17))
            .await
            .unwrap()
            .unwrap()
            .as_ref(),
        b"content-addressed"
    );

    // Missing object returns Ok(None).
    assert_eq!(store.get_blob(&[0xff; 20]).await.unwrap(), None);
}

#[tokio::test]
async fn test_replicated_blob_store_cross_region_origin_fallback_and_read_repair() {
    // Use real Cloud Spanner Emulator when available, falling back to MemoryMetadataStore.
    let (_emulator_guard, metadata): (
        Option<testutils::SpannerEmulatorGuard>,
        Arc<dyn MetadataStore>,
    ) = if let Some(emu) = testutils::SpannerEmulatorGuard::start_or_skip(
        "test_replicated_blob_store_cross_region_origin_fallback_and_read_repair",
        false,
    )
    .await
    {
        let cfg = SpannerConnectConfig {
            database: emu.database_uri("gcs_repl_test"),
            endpoint: Some(emu.grpc_endpoint()),
            admin_endpoint: Some(emu.rest_endpoint().to_string()),
            auto_migrate: true,
            ..Default::default()
        };
        let span = Arc::new(SpannerMetadataStore::connect(&cfg).await.unwrap());
        (Some(emu), span)
    } else {
        (None, Arc::new(MemoryMetadataStore::new()))
    };

    let stub_uc1 = Arc::new(FakeGcsStub::default());
    let stub_ew1 = Arc::new(FakeGcsStub::default());

    let gcs_uc1: Arc<dyn BlobStore> = Arc::new(
        GcsBlobStore::from_storage(
            "gs://jj-blobs-uc1",
            Storage::<FakeGcsStub>::from_stub(stub_uc1.clone()),
        )
        .unwrap(),
    );
    let gcs_ew1: Arc<dyn BlobStore> = Arc::new(
        GcsBlobStore::from_storage(
            "gs://jj-blobs-ew1",
            Storage::<FakeGcsStub>::from_stub(stub_ew1.clone()),
        )
        .unwrap(),
    );

    // Region US-Central1 server (writes locally without fanout so EW1 starts lagging!).
    let store_us = CompositeStore::new(
        metadata.clone(),
        Arc::new(ReplicatedBlobStore::new(
            gcs_uc1.clone(),
            vec![gcs_ew1.clone()],
            metadata.clone(),
            ReplicatedBlobStoreConfig::default(),
        )),
    );

    // Region Europe-West1 server (knows UC1 as peer bucket for origin fallback).
    let repl_eu = Arc::new(ReplicatedBlobStore::new(
        gcs_ew1.clone(),
        vec![gcs_uc1.clone()],
        metadata.clone(),
        ReplicatedBlobStoreConfig::default(),
    ));
    let store_eu = CompositeStore::new(metadata.clone(), repl_eu.clone());

    store_us.register_repo("shared-repo", None).await.unwrap();
    let blob_id = vec![0x3a; 20];
    let body = Bytes::from_static(b"written in us-central1 before eu replication");
    store_us
        .put_file("shared-repo", &blob_id, body.clone())
        .await
        .unwrap();

    // Confirm EW1 bucket does NOT yet have the object (simulating cross-region GCS lag).
    assert_eq!(gcs_ew1.get_blob(&blob_id).await.unwrap(), None);

    // Reader in Europe-West1 reads immediately: Spanner metadata points to gs://jj-blobs-uc1,
    // EW1 misses locally, falls back to UC1, returns exact bytes, and schedules read-repair!
    let hydrated = store_eu
        .get_file_range("shared-repo", &blob_id, 0, None)
        .await
        .unwrap();
    assert_eq!(hydrated, body);
    assert_eq!(repl_eu.stats().origin_fallbacks.load(Ordering::Relaxed), 1);

    // Wait briefly for background read-repair into gs://jj-blobs-ew1 and Spanner location update.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if repl_eu
            .stats()
            .read_repairs_completed
            .load(Ordering::Relaxed)
            >= 1
        {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for background read-repair completion"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    assert_eq!(gcs_ew1.get_blob(&blob_id).await.unwrap(), Some(body));
    let updated_record = metadata
        .get_global_blob_record(&blob_id)
        .await
        .unwrap()
        .unwrap();
    assert!(
        updated_record
            .replicated_locations
            .contains(&"gs://jj-blobs-ew1".to_string())
    );
}

#[tokio::test]
async fn test_replicated_blob_store_single_endpoint_visibility_lag_retry_and_missing_fast_fail() {
    let metadata: Arc<dyn MetadataStore> = Arc::new(MemoryMetadataStore::new());
    // Simulate a dual-region single-endpoint bucket with 120ms replication visibility lag!
    let lagging_stub = Arc::new(FakeGcsStub::with_delays(
        Duration::from_millis(120),
        Duration::ZERO,
    ));
    let gcs_dual: Arc<dyn BlobStore> = Arc::new(
        GcsBlobStore::from_storage(
            "gs://jj-blobs-nam4",
            Storage::<FakeGcsStub>::from_stub(lagging_stub.clone()),
        )
        .unwrap(),
    );

    // Writer uploads to nam4 and records metadata; Reader has a separate cold cache instance.
    let writer = CompositeStore::new(metadata.clone(), gcs_dual.clone());
    let reader_repl = Arc::new(ReplicatedBlobStore::new(
        gcs_dual,
        vec![],
        metadata.clone(),
        ReplicatedBlobStoreConfig {
            cache_capacity_bytes: 1024 * 1024,
            replication_retry_budget: Duration::from_millis(1_500),
            initial_retry_backoff: Duration::from_millis(20),
            async_write_fanout: false,
        },
    ));
    let reader = CompositeStore::new(metadata.clone(), reader_repl.clone());

    writer.register_repo("dual-repo", None).await.unwrap();
    let blob_id = vec![0x7c; 20];
    let payload = Bytes::from_static(b"eventually visible dual-region object");
    writer
        .put_file("dual-repo", &blob_id, payload.clone())
        .await
        .unwrap();

    // Immediate read hits initial 404 during 120ms lag, sees Spanner/Metadata confirmation,
    // backs off with jitter, and succeeds once object becomes visible!
    let fetched = reader
        .get_file_range("dual-repo", &blob_id, 0, None)
        .await
        .unwrap();
    assert_eq!(fetched, payload);
    assert!(
        reader_repl
            .stats()
            .visibility_lag_retries
            .load(Ordering::Relaxed)
            >= 1
    );

    // Conversely, probing an unconfirmed missing blob_id via ReplicatedBlobStore::get_blob
    // fast-fails in ZERO retries!
    let retries_before = reader_repl
        .stats()
        .visibility_lag_retries
        .load(Ordering::Relaxed);
    let reads_before = lagging_stub.read_calls.load(Ordering::SeqCst);
    assert_eq!(reader_repl.get_blob(&[0xee; 20]).await.unwrap(), None);
    assert_eq!(
        reader_repl
            .stats()
            .visibility_lag_retries
            .load(Ordering::Relaxed),
        retries_before
    );
    assert_eq!(
        lagging_stub.read_calls.load(Ordering::SeqCst),
        reads_before + 1
    );
    assert!(matches!(
        reader
            .get_file_range("dual-repo", &[0xee; 20], 0, None)
            .await,
        Err(StoreError::ObjectNotFound { .. })
    ));
}

#[tokio::test]
async fn test_replicated_blob_store_singleflight_coalesces_concurrent_misses() {
    let metadata: Arc<dyn MetadataStore> = Arc::new(MemoryMetadataStore::new());
    let origin_stub = Arc::new(FakeGcsStub::with_delays(
        Duration::ZERO,
        Duration::from_millis(60),
    ));
    let local_stub = Arc::new(FakeGcsStub::default());

    let origin_gcs: Arc<dyn BlobStore> = Arc::new(
        GcsBlobStore::from_storage(
            "gs://jj-blobs-origin",
            Storage::<FakeGcsStub>::from_stub(origin_stub.clone()),
        )
        .unwrap(),
    );
    let local_gcs: Arc<dyn BlobStore> = Arc::new(
        GcsBlobStore::from_storage(
            "gs://jj-blobs-edge",
            Storage::<FakeGcsStub>::from_stub(local_stub),
        )
        .unwrap(),
    );

    let writer = CompositeStore::new(metadata.clone(), origin_gcs.clone());
    writer.register_repo("coalesce-repo", None).await.unwrap();
    let blob_id = vec![0x99; 20];
    let payload = Bytes::from(vec![0x5a; 128 * 1024]);
    writer
        .put_file("coalesce-repo", &blob_id, payload.clone())
        .await
        .unwrap();

    let edge_repl = Arc::new(ReplicatedBlobStore::new(
        local_gcs,
        vec![origin_gcs],
        metadata.clone(),
        ReplicatedBlobStoreConfig::default(),
    ));
    let edge_store = Arc::new(CompositeStore::new(metadata, edge_repl.clone()));

    let concurrency = 24usize;
    let barrier = Arc::new(tokio::sync::Barrier::new(concurrency));
    let mut tasks = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let s = edge_store.clone();
        let b = barrier.clone();
        let id = blob_id.clone();
        let expected = payload.clone();
        tasks.push(tokio::spawn(async move {
            b.wait().await;
            let got = s
                .get_file_range("coalesce-repo", &id, 0, None)
                .await
                .unwrap();
            assert_eq!(got, expected);
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }

    // All 24 concurrent callers coalesced onto a SINGLE origin GCS read!
    assert_eq!(origin_stub.read_calls.load(Ordering::SeqCst), 1);
    assert!(
        edge_repl
            .stats()
            .singleflight_coalesced
            .load(Ordering::Relaxed)
            >= 1
    );
}

#[tokio::test]
async fn test_replicated_blob_store_ranged_reads_small_only_cache_and_range_clamp() {
    let metadata: Arc<dyn MetadataStore> = Arc::new(MemoryMetadataStore::new());
    let stub_uc1 = Arc::new(FakeGcsStub::default());
    let stub_ew1 = Arc::new(FakeGcsStub::default());

    let gcs_uc1: Arc<dyn BlobStore> = Arc::new(
        GcsBlobStore::from_storage(
            "gs://jj-blobs-uc1",
            Storage::<FakeGcsStub>::from_stub(stub_uc1.clone()),
        )
        .unwrap(),
    );
    let gcs_ew1: Arc<dyn BlobStore> = Arc::new(
        GcsBlobStore::from_storage(
            "gs://jj-blobs-ew1",
            Storage::<FakeGcsStub>::from_stub(stub_ew1.clone()),
        )
        .unwrap(),
    );

    let store_us = CompositeStore::new(metadata.clone(), gcs_uc1.clone());
    // Cache holds 8 MiB, which is larger than the 2 MiB asset below; ranged reads must still
    // stream only the requested slice rather than downloading or caching the 2 MiB object.
    let repl_eu = Arc::new(ReplicatedBlobStore::new(
        gcs_ew1.clone(),
        vec![gcs_uc1],
        metadata.clone(),
        ReplicatedBlobStoreConfig {
            cache_capacity_bytes: 8 * 1024 * 1024,
            ..ReplicatedBlobStoreConfig::default()
        },
    ));
    let store_eu = CompositeStore::new(metadata, repl_eu.clone());

    store_us.register_repo("range-repo", None).await.unwrap();

    // 1. Small blob (<= 1 MiB) populates the read-through LRU cache on first read.
    let small_id = vec![0x11; 20];
    let small_body = Bytes::from(vec![0x41; 32 * 1024]);
    store_us
        .put_file("range-repo", &small_id, small_body.clone())
        .await
        .unwrap();
    assert_eq!(
        store_eu
            .get_file_range("range-repo", &small_id, 0, Some(16))
            .await
            .unwrap()
            .len(),
        16
    );
    assert_eq!(repl_eu.stats().cache_hits.load(Ordering::Relaxed), 0);
    assert_eq!(
        store_eu
            .get_file_range("range-repo", &small_id, 8, Some(16))
            .await
            .unwrap()
            .len(),
        16
    );
    assert_eq!(repl_eu.stats().cache_hits.load(Ordering::Relaxed), 1);

    // 2. Large blob (2 MiB > SMALL_BLOB_MAX_BYTES): peer fallback ranged read transfers ONLY
    //    the 4 KiB slice, not 2 MiB, and does not spawn whole-object read-repair.
    let large_id = vec![0x22; 20];
    let large_len = 2 * 1024 * 1024;
    let large_body: Bytes = (0..large_len)
        .map(|i| ((i * 7 + 13) & 0xff) as u8)
        .collect();
    store_us
        .put_file("range-repo", &large_id, large_body.clone())
        .await
        .unwrap();

    stub_uc1.bytes_read.store(0, Ordering::SeqCst);
    let slice = store_eu
        .get_file_range("range-repo", &large_id, 500_000, Some(4_096))
        .await
        .unwrap();
    assert_eq!(slice.as_ref(), &large_body[500_000..504_096]);
    assert_eq!(stub_uc1.bytes_read.load(Ordering::SeqCst), 4_096);
    assert_eq!(gcs_ew1.get_blob(&large_id).await.unwrap(), None);

    // 3. Unbounded `max_bytes = Some(u64::MAX)` at `offset >= 2` is clamped to `content_size - offset`
    //    without overflowing `i64` Range header arithmetic (verified both through ReplicatedBlobStore
    //    and directly on GcsBlobStore).
    let tail_offset = (large_len - 128) as u64;
    let tail = store_eu
        .get_file_range("range-repo", &large_id, tail_offset, Some(u64::MAX))
        .await
        .unwrap();
    assert_eq!(tail.as_ref(), &large_body[(large_len - 128)..]);
    gcs_ew1
        .put_blob(&large_id, large_body.clone())
        .await
        .unwrap();
    let direct_tail = gcs_ew1
        .get_blob_range(&large_id, tail_offset, Some(u64::MAX))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(direct_tail.as_ref(), &large_body[(large_len - 128)..]);

    // 4. Full reads of >1 MiB blobs and `put_blob` write bursts do not evict cached small blobs.
    let tiny_cache_repl = Arc::new(ReplicatedBlobStore::new(
        gcs_ew1,
        vec![],
        Arc::new(MemoryMetadataStore::new()),
        ReplicatedBlobStoreConfig {
            cache_capacity_bytes: 48 * 1024,
            ..ReplicatedBlobStoreConfig::default()
        },
    ));
    assert_eq!(
        tiny_cache_repl
            .get_blob(&large_id)
            .await
            .unwrap()
            .unwrap()
            .len(),
        large_len
    );
    let cached_small_id = vec![0x33; 20];
    tiny_cache_repl
        .put_blob(&cached_small_id, Bytes::from(vec![0x52; 32 * 1024]))
        .await
        .unwrap();
    // First get_blob populates cache; subsequent large get_blob and put_blob writes must not evict it.
    let _ = tiny_cache_repl.get_blob(&cached_small_id).await.unwrap();
    assert_eq!(
        tiny_cache_repl.stats().cache_hits.load(Ordering::Relaxed),
        0
    );
    let _ = tiny_cache_repl.get_blob(&large_id).await.unwrap();
    for idx in 0..4u8 {
        tiny_cache_repl
            .put_blob(&[idx; 20], Bytes::from(vec![idx; 40 * 1024]))
            .await
            .unwrap();
    }
    let _ = tiny_cache_repl.get_blob(&cached_small_id).await.unwrap();
    assert_eq!(
        tiny_cache_repl.stats().cache_hits.load(Ordering::Relaxed),
        1
    );
}

#[tokio::test]
async fn test_replicated_blob_store_detached_leader_transient_fallback_and_deduped_repair() {
    let metadata: Arc<dyn MetadataStore> = Arc::new(MemoryMetadataStore::new());
    let origin_stub = Arc::new(FakeGcsStub::with_delays(
        Duration::ZERO,
        Duration::from_millis(80),
    ));
    let local_stub = Arc::new(FakeGcsStub {
        write_delay: Duration::from_millis(80),
        ..Default::default()
    });

    let origin_gcs: Arc<dyn BlobStore> = Arc::new(
        GcsBlobStore::from_storage(
            "gs://jj-blobs-origin",
            Storage::<FakeGcsStub>::from_stub(origin_stub.clone()),
        )
        .unwrap(),
    );
    let local_gcs: Arc<dyn BlobStore> = Arc::new(
        GcsBlobStore::from_storage(
            "gs://jj-blobs-edge",
            Storage::<FakeGcsStub>::from_stub(local_stub.clone()),
        )
        .unwrap(),
    );

    let writer = CompositeStore::new(metadata.clone(), origin_gcs.clone());
    writer.register_repo("resilience-repo", None).await.unwrap();

    let edge_repl = Arc::new(ReplicatedBlobStore::new(
        local_gcs,
        vec![origin_gcs],
        metadata.clone(),
        ReplicatedBlobStoreConfig::default(),
    ));
    let edge_store = Arc::new(CompositeStore::new(metadata.clone(), edge_repl.clone()));

    // 1. Detached singleflight leader: cancelling the first caller's future must not fail
    //    coalesced followers or trigger duplicate origin reads.
    let cancel_id = vec![0xa1; 20];
    let cancel_payload = Bytes::from(vec![0x7b; 64 * 1024]);
    writer
        .put_file("resilience-repo", &cancel_id, cancel_payload.clone())
        .await
        .unwrap();
    origin_stub.read_calls.store(0, Ordering::SeqCst);

    let s_first = edge_store.clone();
    let id_first = cancel_id.clone();
    let first_caller = tokio::spawn(async move {
        tokio::time::timeout(
            Duration::from_millis(15),
            s_first.get_file_range("resilience-repo", &id_first, 0, None),
        )
        .await
    });

    tokio::time::sleep(Duration::from_millis(5)).await;
    let mut followers = Vec::new();
    for _ in 0..6 {
        let s = edge_store.clone();
        let id = cancel_id.clone();
        followers.push(tokio::spawn(async move {
            s.get_file_range("resilience-repo", &id, 0, None).await
        }));
    }

    assert!(
        first_caller.await.unwrap().is_err(),
        "first caller should time out"
    );
    for f in followers {
        assert_eq!(f.await.unwrap().unwrap(), cancel_payload);
    }
    assert_eq!(origin_stub.read_calls.load(Ordering::SeqCst), 1);

    // 2. Transient local Unavailable error falls back to peer replica instead of failing.
    let transient_id = vec![0xb2; 20];
    let transient_payload = Bytes::from_static(b"served from origin while local returns 503");
    writer
        .put_file("resilience-repo", &transient_id, transient_payload.clone())
        .await
        .unwrap();
    *local_stub.injected_read_error.lock().unwrap() = Some(RpcCode::Unavailable);

    let recovered = edge_store
        .get_file_range("resilience-repo", &transient_id, 0, None)
        .await
        .unwrap();
    assert_eq!(recovered, transient_payload);
    *local_stub.injected_read_error.lock().unwrap() = None;

    // 3. Non-transient local error (InvalidArgument) preserves its StoreError variant across singleflight.
    *local_stub.injected_read_error.lock().unwrap() = Some(RpcCode::InvalidArgument);
    let err = edge_repl.get_blob(&[0xc3; 20]).await.unwrap_err();
    assert!(
        matches!(err, StoreError::InvalidArgument(_)),
        "unexpected error variant: {err:?}"
    );
    *local_stub.injected_read_error.lock().unwrap() = None;

    // 4. Deduped read-repair: repeated misses on an uncached >1 MiB blob while local repair
    //    upload is in flight only spawn one repair write.
    let big_payload = Bytes::from(vec![0x3c; 1536 * 1024]);

    // Disable origin read delay so back-to-back full reads complete while local 80ms write_delay runs.
    let fast_origin_stub = Arc::new(FakeGcsStub::default());
    let fast_origin_gcs: Arc<dyn BlobStore> = Arc::new(
        GcsBlobStore::from_storage(
            "gs://jj-blobs-origin-fast",
            Storage::<FakeGcsStub>::from_stub(fast_origin_stub),
        )
        .unwrap(),
    );
    let fast_writer = CompositeStore::new(metadata.clone(), fast_origin_gcs.clone());
    let dedupe_id = vec![0xd5; 20];
    fast_writer
        .put_file("resilience-repo", &dedupe_id, big_payload.clone())
        .await
        .unwrap();

    let slow_local_stub = Arc::new(FakeGcsStub {
        write_delay: Duration::from_millis(100),
        ..Default::default()
    });
    let slow_local_gcs: Arc<dyn BlobStore> = Arc::new(
        GcsBlobStore::from_storage(
            "gs://jj-blobs-edge-slow",
            Storage::<FakeGcsStub>::from_stub(slow_local_stub.clone()),
        )
        .unwrap(),
    );
    let dedupe_repl = Arc::new(ReplicatedBlobStore::new(
        slow_local_gcs,
        vec![fast_origin_gcs],
        metadata.clone(),
        ReplicatedBlobStoreConfig::default(),
    ));
    let dedupe_store = CompositeStore::new(metadata, dedupe_repl.clone());

    assert_eq!(
        dedupe_store
            .get_file_range("resilience-repo", &dedupe_id, 0, None)
            .await
            .unwrap(),
        big_payload
    );
    assert_eq!(
        dedupe_store
            .get_file_range("resilience-repo", &dedupe_id, 0, None)
            .await
            .unwrap(),
        big_payload
    );
    assert!(
        dedupe_repl
            .stats()
            .read_repairs_deduplicated
            .load(Ordering::Relaxed)
            >= 1
    );

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while dedupe_repl
        .stats()
        .read_repairs_completed
        .load(Ordering::Relaxed)
        < 1
    {
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for deduped repair"
        );
        tokio::time::sleep(Duration::from_millis(15)).await;
    }
    assert_eq!(slow_local_stub.write_calls.load(Ordering::SeqCst), 1);
}
