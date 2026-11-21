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

//! Replication-aware `BlobStore` coordinator resolving Cloud Storage replication skew.
//!
//! # Authority and Crash-Safety Ordering
//! `MetadataStore` (Cloud Spanner) commits become visible across regions in milliseconds, while
//! copies between per-region buckets arrive in the background or through read-repair.
//! `ReplicatedBlobStore` writes to the local `BlobStore` before
//! `CompositeStore` publishes `BlobRecord` (`primary_location`, `replicated_locations`) in
//! `MetadataStore`, then bridges cross-region read skew via three mechanisms:
//! 1. Read-through byte-bounded LRU cache for small blobs up to [`SMALL_BLOB_MAX_BYTES`].
//!    Ranged reads of larger blobs go to the local store, then to peers, by range.
//! 2. Cross-region fallback to `BlobRecord.primary_location` (or peer replicas) on local `404`.
//!    Whole-object fallbacks start one background read-repair per blob into the local
//!    `BlobStore` and `MetadataStore::add_blob_replica_location`.
//! 3. Optional bounded exponential-backoff retry on metadata-confirmed blobs, off by default.
//!    Each Cloud Storage bucket provides read-after-write consistency, so the retry only
//!    serves object stores without that guarantee. Unconfirmed hashes fail fast in zero retries.
//!
//! # Concurrency and Merge Invariants
//! Concurrent misses on the same `blob_id` coalesce through a per-digest singleflight
//! channel backed by a detached Tokio task so one cancelled caller cannot abort the fetch for
//! other waiters. Waiters receive the leader's exact `StoreError` variant, and a transient
//! local error counts as a miss so peers can still serve the read.
//!
//! # Cross-Process and RPC Boundaries
//! Background read-repair and optional write fan-out run on Tokio tasks
//! without blocking foreground `ReadFile` or `WriteFileStream` RPC latency.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use tokio::sync::watch;
use tokio::time::Instant;
use tracing::warn;

use super::{BlobRecord, BlobStore, MetadataStore, StoreError, StoreResult, slice_blob_range};

/// Ranged reads of blobs up to this size fetch and cache the whole object. Ranged reads of
/// larger blobs are forwarded by range and bypass the cache.
pub const SMALL_BLOB_MAX_BYTES: u64 = 1024 * 1024;

#[derive(Clone, Debug)]
pub struct ReplicatedBlobStoreConfig {
    /// Maximum total bytes retained in the in-memory read-through LRU cache.
    pub cache_capacity_bytes: usize,
    /// Total retry window for a metadata-confirmed blob that no store returns. Zero disables
    /// the retry: Cloud Storage buckets have read-after-write consistency and lag between
    /// buckets is handled by peer fallback.
    pub replication_retry_budget: Duration,
    /// Initial retry delay before exponential backoff.
    pub initial_retry_backoff: Duration,
    /// Replicate uploaded blobs to configured peer stores in the background.
    pub async_write_fanout: bool,
}

impl Default for ReplicatedBlobStoreConfig {
    fn default() -> Self {
        Self {
            cache_capacity_bytes: 64 * 1024 * 1024,
            replication_retry_budget: Duration::ZERO,
            initial_retry_backoff: Duration::from_millis(20),
            async_write_fanout: false,
        }
    }
}

#[derive(Debug, Default)]
pub struct ReplicationStats {
    pub cache_hits: AtomicU64,
    pub local_hits: AtomicU64,
    pub origin_fallbacks: AtomicU64,
    pub read_repairs_completed: AtomicU64,
    pub read_repairs_deduplicated: AtomicU64,
    pub visibility_lag_retries: AtomicU64,
    pub singleflight_coalesced: AtomicU64,
}

#[derive(Debug)]
struct ByteBoundedLruCache {
    capacity_bytes: usize,
    used_bytes: usize,
    entries: HashMap<Vec<u8>, (Bytes, u64)>,
    order: VecDeque<(Vec<u8>, u64)>,
    tick: u64,
}

impl ByteBoundedLruCache {
    fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
            tick: 0,
        }
    }

    fn get(&mut self, blob_id: &[u8]) -> Option<Bytes> {
        if self.capacity_bytes == 0 {
            return None;
        }
        let data = self.entries.get(blob_id)?.0.clone();
        self.tick = self.tick.wrapping_add(1);
        let gen_id = self.tick;
        if let Some(entry) = self.entries.get_mut(blob_id) {
            entry.1 = gen_id;
        }
        self.order.push_back((blob_id.to_vec(), gen_id));
        if self.order.len() > self.entries.len().saturating_mul(4).max(64) {
            self.compact_order();
        }
        Some(data)
    }

    fn insert(&mut self, blob_id: &[u8], data: Bytes) {
        if self.capacity_bytes == 0 || data.len() > self.capacity_bytes {
            return;
        }
        if let Some((prev, _)) = self.entries.remove(blob_id) {
            self.used_bytes = self.used_bytes.saturating_sub(prev.len());
        }
        while self.used_bytes + data.len() > self.capacity_bytes {
            let Some((evict_key, evict_gen)) = self.order.pop_front() else {
                break;
            };
            if let Some((_, current_gen)) = self.entries.get(&evict_key)
                && *current_gen == evict_gen
                && let Some((removed, _)) = self.entries.remove(&evict_key)
            {
                self.used_bytes = self.used_bytes.saturating_sub(removed.len());
            }
        }
        self.tick = self.tick.wrapping_add(1);
        let gen_id = self.tick;
        self.used_bytes += data.len();
        self.entries.insert(blob_id.to_vec(), (data, gen_id));
        self.order.push_back((blob_id.to_vec(), gen_id));
    }

    fn compact_order(&mut self) {
        self.order.retain(|(k, g)| {
            self.entries
                .get(k)
                .is_some_and(|(_, active_gen)| active_gen == g)
        });
    }
}

type InflightResult = StoreResult<Option<Bytes>>;
type InflightMap = HashMap<Vec<u8>, watch::Receiver<Option<InflightResult>>>;

struct InflightGuard {
    inner: Arc<ReplicatedInner>,
    blob_id: Vec<u8>,
    tx: Option<watch::Sender<Option<InflightResult>>>,
}

impl InflightGuard {
    fn complete(mut self, outcome: InflightResult) {
        self.inner.inflight.lock().unwrap().remove(&self.blob_id);
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Some(outcome));
        }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if let Some(tx) = self.tx.take() {
            self.inner.inflight.lock().unwrap().remove(&self.blob_id);
            let _ = tx.send(Some(Err(StoreError::Backend(
                "singleflight blob fetch task terminated before completion".to_string(),
            ))));
        }
    }
}

/// Removes a blob from the in-flight read-repair set when its repair task ends.
struct RepairGuard {
    inner: Arc<ReplicatedInner>,
    blob_id: Vec<u8>,
}

impl Drop for RepairGuard {
    fn drop(&mut self) {
        self.inner
            .repairs_in_flight
            .lock()
            .unwrap()
            .remove(&self.blob_id);
    }
}

/// Bytes of a blob that a fetch returns.
#[derive(Clone, Copy, Debug)]
enum ReadSpan {
    /// The whole object. Small results are cached, and peer hits start a read-repair.
    Full,
    /// `length` bytes from `offset`, clamped to the object. Results are neither cached nor
    /// repaired.
    Range { offset: u64, length: u64 },
}

impl ReadSpan {
    async fn read_from(self, store: &dyn BlobStore, blob_id: &[u8]) -> StoreResult<Option<Bytes>> {
        match self {
            ReadSpan::Full => store.get_blob(blob_id).await,
            ReadSpan::Range { offset, length } => {
                store.get_blob_range(blob_id, offset, Some(length)).await
            }
        }
    }

    fn slice(self, full: &Bytes) -> StoreResult<Bytes> {
        match self {
            ReadSpan::Full => Ok(full.clone()),
            ReadSpan::Range { offset, length } => slice_blob_range(full, offset, Some(length)),
        }
    }
}

/// Errors after which another attempt or another store may still succeed.
fn is_transient(err: &StoreError) -> bool {
    matches!(err, StoreError::Unavailable(_))
}

struct ReplicatedInner {
    local: Arc<dyn BlobStore>,
    peers: HashMap<String, Arc<dyn BlobStore>>,
    metadata: Arc<dyn MetadataStore>,
    cache: Mutex<ByteBoundedLruCache>,
    inflight: Mutex<InflightMap>,
    repairs_in_flight: Mutex<HashSet<Vec<u8>>>,
    config: ReplicatedBlobStoreConfig,
    stats: Arc<ReplicationStats>,
}

pub struct ReplicatedBlobStore {
    inner: Arc<ReplicatedInner>,
}

impl ReplicatedBlobStore {
    pub fn new(
        local: Arc<dyn BlobStore>,
        peers: Vec<Arc<dyn BlobStore>>,
        metadata: Arc<dyn MetadataStore>,
        config: ReplicatedBlobStoreConfig,
    ) -> Self {
        let local_loc = local.location_id().to_string();
        let mut peer_map = HashMap::new();
        for peer in peers {
            let loc = peer.location_id().to_string();
            if loc != local_loc {
                peer_map.insert(loc, peer);
            }
        }
        Self {
            inner: Arc::new(ReplicatedInner {
                local,
                peers: peer_map,
                metadata,
                cache: Mutex::new(ByteBoundedLruCache::new(config.cache_capacity_bytes)),
                inflight: Mutex::new(HashMap::new()),
                repairs_in_flight: Mutex::new(HashSet::new()),
                config,
                stats: Arc::new(ReplicationStats::default()),
            }),
        }
    }

    pub fn stats(&self) -> &Arc<ReplicationStats> {
        &self.inner.stats
    }

    async fn fetch_full_coalesced(
        &self,
        blob_id: &[u8],
        known_record: Option<&BlobRecord>,
    ) -> StoreResult<Option<Bytes>> {
        if let Some(cached) = self.inner.cache.lock().unwrap().get(blob_id) {
            self.inner.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(cached));
        }

        let mut rx = {
            let mut inflight = self.inner.inflight.lock().unwrap();
            if let Some(cached) = self.inner.cache.lock().unwrap().get(blob_id) {
                self.inner.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Some(cached));
            }
            if let Some(existing_rx) = inflight.get(blob_id) {
                self.inner
                    .stats
                    .singleflight_coalesced
                    .fetch_add(1, Ordering::Relaxed);
                existing_rx.clone()
            } else {
                let (tx, rx) = watch::channel(None);
                let blob_id_owned = blob_id.to_vec();
                inflight.insert(blob_id_owned.clone(), rx.clone());
                let inner = self.inner.clone();
                let record_owned = known_record.cloned();
                // Run the leader fetch in a detached task so cancelling one caller's RPC
                // future does not abort the download for coalesced waiters.
                tokio::spawn(async move {
                    let guard = InflightGuard {
                        inner: inner.clone(),
                        blob_id: blob_id_owned.clone(),
                        tx: Some(tx),
                    };
                    let res = inner
                        .fetch_uncached(&blob_id_owned, record_owned.as_ref(), ReadSpan::Full)
                        .await;
                    guard.complete(res);
                });
                rx
            }
        };

        let _ = rx.wait_for(Option::is_some).await;
        let outcome = rx.borrow().clone();
        outcome.unwrap_or_else(|| {
            Err(StoreError::Unavailable(
                "singleflight leader channel closed without publishing blob state".to_string(),
            ))
        })
    }
}

impl ReplicatedInner {
    fn insert_cache_if_small(&self, blob_id: &[u8], bytes: &Bytes) {
        if bytes.len() as u64 <= SMALL_BLOB_MAX_BYTES {
            self.cache.lock().unwrap().insert(blob_id, bytes.clone());
        }
    }

    async fn fetch_uncached(
        self: &Arc<Self>,
        blob_id: &[u8],
        known_record: Option<&BlobRecord>,
        span: ReadSpan,
    ) -> StoreResult<Option<Bytes>> {
        // 1. Local regional store first. A transient local error is reported only if no other
        //    store has the blob.
        let mut transient_err = None;
        if let Some(bytes) = self.read_local(blob_id, span, &mut transient_err).await? {
            self.stats.local_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(bytes));
        }

        // 2. Local returned 404 (or transient error): consult Spanner/MetadataStore record to
        //    distinguish genuine missing hashes from replication lag.
        let owned_record;
        let record = match known_record {
            Some(r) => r,
            None => {
                owned_record = self.metadata.get_global_blob_record(blob_id).await?;
                match &owned_record {
                    Some(r) => r,
                    // Unconfirmed hash: return None in zero retries!
                    None => return transient_err.map_or(Ok(None), Err),
                }
            }
        };

        if let Some(inline) = &record.inline_content {
            if matches!(span, ReadSpan::Full) {
                self.insert_cache_if_small(blob_id, inline);
            }
            return span.slice(inline).map(Some);
        }

        // 3. Try authoritative primary_location and known peer replicas immediately.
        if let Some(bytes) = self.try_peer_fallback(blob_id, record, span).await? {
            return Ok(Some(bytes));
        }

        // 4. Optional bounded retry with backoff + jitter for object stores without
        //    read-after-write consistency. Cloud Storage buckets need none, and lag between
        //    buckets is handled by step 3, so the default budget is zero.
        if self.config.replication_retry_budget.is_zero() {
            return transient_err.map_or(Ok(None), Err);
        }

        let deadline = Instant::now() + self.config.replication_retry_budget;
        let mut delay = self
            .config
            .initial_retry_backoff
            .max(Duration::from_millis(5));
        let jitter_seed = u64::from(blob_id.first().copied().unwrap_or(0));
        let mut attempt = 0u64;

        while Instant::now() < deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            // Deterministic per-blob + per-attempt jitter in [0ms, 7ms] avoids retry thundering
            // herds.
            let jitter = Duration::from_millis((jitter_seed.wrapping_add(attempt * 13)) % 8);
            let sleep_dur = (delay + jitter).min(remaining);
            if sleep_dur.is_zero() {
                break;
            }
            tokio::time::sleep(sleep_dur).await;
            attempt += 1;
            self.stats
                .visibility_lag_retries
                .fetch_add(1, Ordering::Relaxed);

            if let Some(bytes) = self.read_local(blob_id, span, &mut transient_err).await? {
                return Ok(Some(bytes));
            }
            if let Some(bytes) = self.try_peer_fallback(blob_id, record, span).await? {
                return Ok(Some(bytes));
            }

            delay = (delay * 2).min(Duration::from_millis(250));
        }

        transient_err.map_or(Ok(None), Err)
    }

    /// Reads `span` from the local store. A transient error is stored in `transient_err` and
    /// reported as a miss so the caller can try peers or retry; a clean miss clears it.
    async fn read_local(
        &self,
        blob_id: &[u8],
        span: ReadSpan,
        transient_err: &mut Option<StoreError>,
    ) -> StoreResult<Option<Bytes>> {
        match span.read_from(self.local.as_ref(), blob_id).await {
            Ok(Some(bytes)) => {
                if matches!(span, ReadSpan::Full) {
                    self.insert_cache_if_small(blob_id, &bytes);
                }
                Ok(Some(bytes))
            }
            Ok(None) => {
                *transient_err = None;
                Ok(None)
            }
            Err(err) if is_transient(&err) => {
                warn!(
                    "Local blob read failed for {}, trying peers: {}",
                    hex::encode(blob_id),
                    err
                );
                *transient_err = Some(err);
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }

    async fn try_peer_fallback(
        self: &Arc<Self>,
        blob_id: &[u8],
        record: &BlobRecord,
        span: ReadSpan,
    ) -> StoreResult<Option<Bytes>> {
        if self.peers.is_empty() {
            return Ok(None);
        }

        let local_loc = self.local.location_id();
        let mut candidate_locations: Vec<&str> = Vec::new();
        if let Some(primary) = record.primary_location.as_deref()
            && primary != local_loc
        {
            candidate_locations.push(primary);
        }
        for loc in &record.replicated_locations {
            if loc != local_loc && !candidate_locations.contains(&loc.as_str()) {
                candidate_locations.push(loc.as_str());
            }
        }

        for loc in candidate_locations {
            let Some(peer_store) = self.peers.get(loc) else {
                continue;
            };
            match span.read_from(peer_store.as_ref(), blob_id).await {
                Ok(Some(bytes)) => {
                    self.stats.origin_fallbacks.fetch_add(1, Ordering::Relaxed);
                    if matches!(span, ReadSpan::Full) {
                        self.insert_cache_if_small(blob_id, &bytes);
                        self.spawn_read_repair(blob_id.to_vec(), bytes.clone());
                    }
                    return Ok(Some(bytes));
                }
                Ok(None) => continue,
                Err(err) => {
                    warn!(
                        "Peer blob fallback from {} failed for {}: {}",
                        loc,
                        hex::encode(blob_id),
                        err
                    );
                }
            }
        }

        Ok(None)
    }

    fn spawn_read_repair(self: &Arc<Self>, blob_id: Vec<u8>, bytes: Bytes) {
        // Reads keep missing locally until the repair lands; one upload per blob is enough.
        if !self
            .repairs_in_flight
            .lock()
            .unwrap()
            .insert(blob_id.clone())
        {
            self.stats
                .read_repairs_deduplicated
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        let in_flight = RepairGuard {
            inner: self.clone(),
            blob_id: blob_id.clone(),
        };
        let local = self.local.clone();
        let metadata = self.metadata.clone();
        let stats = self.stats.clone();
        tokio::spawn(async move {
            let _in_flight = in_flight;
            let local_loc = local.location_id().to_string();
            if let Err(err) = local.put_blob(&blob_id, bytes).await {
                warn!(
                    "Background read-repair write to {} failed for {}: {}",
                    local_loc,
                    hex::encode(&blob_id),
                    err
                );
                return;
            }
            if let Err(err) = metadata
                .add_blob_replica_location(&blob_id, &local_loc)
                .await
            {
                warn!(
                    "Background read-repair metadata update failed for {}: {}",
                    hex::encode(&blob_id),
                    err
                );
                return;
            }
            stats.read_repairs_completed.fetch_add(1, Ordering::Relaxed);
        });
    }

    fn spawn_write_fanout(&self, blob_id: Vec<u8>, bytes: Bytes) {
        for (loc, peer) in &self.peers {
            let peer = peer.clone();
            let loc = loc.clone();
            let metadata = self.metadata.clone();
            let blob_id = blob_id.clone();
            let bytes = bytes.clone();
            tokio::spawn(async move {
                if let Err(err) = peer.put_blob(&blob_id, bytes).await {
                    warn!(
                        "Background peer fan-out to {} failed for {}: {}",
                        loc,
                        hex::encode(&blob_id),
                        err
                    );
                    return;
                }
                let _ = metadata.add_blob_replica_location(&blob_id, &loc).await;
            });
        }
    }
}

#[async_trait]
impl BlobStore for ReplicatedBlobStore {
    fn location_id(&self) -> &str {
        self.inner.local.location_id()
    }

    async fn put_blob(&self, blob_id: &[u8], data: Bytes) -> StoreResult<()> {
        // Uploads stay out of the read cache so agent writes do not evict entries readers use.
        self.inner.local.put_blob(blob_id, data.clone()).await?;
        if self.inner.config.async_write_fanout && !self.inner.peers.is_empty() {
            self.inner.spawn_write_fanout(blob_id.to_vec(), data);
        }
        Ok(())
    }

    async fn get_blob(&self, blob_id: &[u8]) -> StoreResult<Option<Bytes>> {
        self.fetch_full_coalesced(blob_id, None).await
    }

    async fn get_blob_range(
        &self,
        blob_id: &[u8],
        offset: u64,
        length: Option<u64>,
    ) -> StoreResult<Option<Bytes>> {
        if offset == 0 && matches!(length, None | Some(0)) {
            return self.fetch_full_coalesced(blob_id, None).await;
        }
        // The content size decides between a whole-object read and a forwarded range.
        match self.inner.metadata.get_global_blob_record(blob_id).await? {
            Some(record) => {
                self.get_blob_range_with_record(&record, offset, length)
                    .await
            }
            None => {
                self.inner
                    .local
                    .get_blob_range(blob_id, offset, length)
                    .await
            }
        }
    }

    async fn get_blob_range_with_record(
        &self,
        record: &BlobRecord,
        offset: u64,
        length: Option<u64>,
    ) -> StoreResult<Option<Bytes>> {
        if record.content_size == 0 {
            if offset == 0 {
                return Ok(Some(Bytes::new()));
            }
            return Err(StoreError::OutOfRange(format!(
                "requested read at offset {offset}, file size is 0"
            )));
        }
        if offset >= record.content_size {
            return Err(StoreError::OutOfRange(format!(
                "requested read at offset {offset}, file size is {}",
                record.content_size
            )));
        }

        // Clamp before calling backends: the Cloud Storage client computes the inclusive Range
        // end as `offset + length - 1` in i64.
        let remaining = record.content_size - offset;
        let length = match length {
            Some(max_len) if max_len > 0 => max_len.min(remaining),
            _ => remaining,
        };

        // Small blobs and whole-object reads go through singleflight; only small blobs are cached.
        if (offset == 0 && length == record.content_size)
            || record.content_size <= SMALL_BLOB_MAX_BYTES
        {
            let Some(full) = self
                .fetch_full_coalesced(&record.blob_id, Some(record))
                .await?
            else {
                return Ok(None);
            };
            return slice_blob_range(&full, offset, Some(length)).map(Some);
        }

        // Ranged reads of larger blobs read only the requested bytes, from the local store
        // first and then from peers.
        if let Some(cached) = self.inner.cache.lock().unwrap().get(&record.blob_id) {
            self.inner.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            return slice_blob_range(&cached, offset, Some(length)).map(Some);
        }
        self.inner
            .fetch_uncached(
                &record.blob_id,
                Some(record),
                ReadSpan::Range { offset, length },
            )
            .await
    }

    async fn has_blob(&self, blob_id: &[u8]) -> StoreResult<bool> {
        if self.inner.cache.lock().unwrap().get(blob_id).is_some() {
            return Ok(true);
        }
        self.inner.local.has_blob(blob_id).await
    }
}
