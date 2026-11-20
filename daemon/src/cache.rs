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

//! Two-tier (L1 byte-bounded memory LRU + L2 local disk CAS) immutable object cache
//! with per-`(upstream_url, repo_id)` authorization scoping and singleflight miss
//! coalescing for jj-cc-daemon.
//!
//! # Authority and Crash-Safety Ordering
//! Every L2 disk CAS entry is written to a process-unique sibling temporary file, flushed
//! with `sync_all`, and renamed onto `objects/<upstream_hex>/<kind>/<aa>/<bb>/<hex>` (with
//! per-repository `View` and `Operation` payloads and authorization markers under
//! `scopes/<scope_hex>/...`) before L1 promotion or caller acknowledgment, so a crash leaves
//! either no entry or a complete one. Reads discard empty or undecodable entries and
//! refetch them, and `File`/`Symlink` payloads that do not hash to their Git blob id are
//! never cached. A failed L2 write is logged and counted; the caller still gets the bytes.
//!
//! # Concurrency and Merge Invariants
//! All cached entities (`Commit`, `Tree`, `Symlink`, `File`, `View`, and `Operation`) are
//! content-addressed and immutable once created. Concurrent cold misses within the same
//! [`CacheScope`] coalesce onto a single upstream gRPC fetch via [`SingleflightGuard`], which
//! hands the fetched bytes to waiters through a `watch` channel. Shared payloads are
//! partitioned per normalized upstream URL, and unverified repositories must prove upstream
//! reachability before reading them (`REQ-CC27`).
//!
//! # Cross-Process and RPC Boundaries
//! File blobs up to [`MAX_L1_OBJECT_BYTES`] are retained in L1 RAM as zero-copy [`Bytes`]
//! slices. Larger blobs are read from L2 starting at `ReadFileRequest.offset`, where the
//! requested range is collected in memory as 64 KiB chunks before streaming, and a cold miss
//! buffers the whole blob fetched from upstream.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::future::Future;
use std::io::{SeekFrom, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use cc_common::backend::{Commit, ReadTreeResponse};
use cc_common::op_store::{Operation, View};
use lru::LruCache;
use prost::Message;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::sync::watch;

/// Maximum single-object payload promoted into L1 RAM (`4 MiB`). Larger blobs live in L2 disk CAS.
pub const MAX_L1_OBJECT_BYTES: usize = 4 * 1024 * 1024;

/// Streaming frame size (`64 KiB`) when serving `ReadFile` hits from L1/L2 cache.
pub const CACHE_STREAM_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectKind {
    Commit,
    Tree,
    Symlink,
    File,
    View,
    Operation,
}

impl ObjectKind {
    pub fn dir_name(self) -> &'static str {
        match self {
            Self::Commit => "commits",
            Self::Tree => "trees",
            Self::Symlink => "symlinks",
            Self::File => "blobs",
            Self::View => "views",
            Self::Operation => "ops",
        }
    }

    pub fn is_repo_scoped(self) -> bool {
        matches!(self, Self::View | Self::Operation)
    }
}

/// Logical `(upstream_url, repo_id)` isolation boundary for cached lookups.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheScope {
    upstream_url: String,
    repo_id: String,
    scope_hex: String,
    /// Partition of the shared CAS for `upstream_url`, so bytes cached for one server are
    /// never served to scopes of another server.
    upstream_hex: String,
}

impl CacheScope {
    pub fn new(upstream_url: impl AsRef<str>, repo_id: impl AsRef<str>) -> Self {
        let normalized_url = upstream_url
            .as_ref()
            .trim()
            .trim_end_matches('/')
            .to_string();
        let repo = repo_id.as_ref().to_string();
        let mut material = Vec::with_capacity(normalized_url.len() + 1 + repo.len() + 8);
        material.extend_from_slice(&(normalized_url.len() as u64).to_be_bytes());
        material.extend_from_slice(normalized_url.as_bytes());
        material.push(0);
        material.extend_from_slice(repo.as_bytes());
        let scope_hex = hex::encode(sha256_bytes(&material));
        let upstream_hex = hex::encode(sha256_bytes(normalized_url.as_bytes()));
        Self {
            upstream_url: normalized_url,
            repo_id: repo,
            scope_hex,
            upstream_hex,
        }
    }

    pub fn upstream_url(&self) -> &str {
        &self.upstream_url
    }

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub fn scope_hex(&self) -> &str {
        &self.scope_hex
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStatsSnapshot {
    pub l1_hits: u64,
    pub l2_hits: u64,
    pub misses: u64,
    pub singleflight_coalesced: u64,
    pub write_through_objects: u64,
    pub upstream_rpcs: u64,
    pub l1_bytes: usize,
    pub l1_entries: usize,
    pub l2_write_errors: u64,
}

impl CacheStatsSnapshot {
    pub fn hit_ratio(&self) -> f64 {
        let total = self.l1_hits + self.l2_hits + self.misses;
        if total == 0 {
            0.0
        } else {
            ((self.l1_hits + self.l2_hits) as f64) / (total as f64)
        }
    }
}

/// `(partition_hex, kind, id)`: the scope for repo-scoped kinds, the upstream otherwise.
type L1Key = (String, ObjectKind, Vec<u8>);

struct ByteBoundedLru {
    map: LruCache<L1Key, Bytes>,
    current_bytes: usize,
    max_bytes: usize,
}

impl ByteBoundedLru {
    fn new(max_bytes: usize) -> Self {
        Self {
            map: LruCache::unbounded(),
            current_bytes: 0,
            max_bytes,
        }
    }

    fn get(&mut self, key: &L1Key) -> Option<Bytes> {
        self.map.get(key).cloned()
    }

    fn peek_len(&self, key: &L1Key) -> Option<u64> {
        self.map.peek(key).map(|b| b.len() as u64)
    }

    fn put(&mut self, key: L1Key, val: Bytes) {
        let weight = val.len();
        if weight > self.max_bytes || weight > MAX_L1_OBJECT_BYTES {
            return;
        }
        if let Some(prev) = self.map.put(key, val) {
            self.current_bytes = self.current_bytes.saturating_sub(prev.len());
        }
        self.current_bytes += weight;
        while self.current_bytes > self.max_bytes {
            let Some((_, evicted)) = self.map.pop_lru() else {
                self.current_bytes = 0;
                break;
            };
            self.current_bytes = self.current_bytes.saturating_sub(evicted.len());
        }
    }
}

type InflightKey = (String, ObjectKind, Vec<u8>);
/// The leader publishes its fetched bytes through the `watch` channel. A `watch` keeps the last
/// value, so a waiter that joined just before the leader finished still observes the result.
type InflightMap = Mutex<HashMap<InflightKey, watch::Receiver<Option<Bytes>>>>;

struct SingleflightGuard<'a> {
    inflight: &'a InflightMap,
    key: InflightKey,
    result: watch::Sender<Option<Bytes>>,
}

impl SingleflightGuard<'_> {
    fn publish(&self, fetched: &Bytes) {
        self.result.send_replace(Some(fetched.clone()));
    }
}

impl Drop for SingleflightGuard<'_> {
    fn drop(&mut self) {
        // The sender drops after this, which closes the channel. Waiters that joined before
        // then see either the published bytes or, if the leader failed or was cancelled, a
        // closed channel, and retry.
        self.inflight.lock().unwrap().remove(&self.key);
    }
}

enum Flight<'a> {
    Leader(SingleflightGuard<'a>),
    Waiter(watch::Receiver<Option<Bytes>>),
}

pub struct TwoTierCache {
    cache_dir: PathBuf,
    l1: Mutex<ByteBoundedLru>,
    authorized: Mutex<HashSet<(String, ObjectKind, Vec<u8>)>>,
    inflight: InflightMap,
    temp_seq: AtomicU64,
    l1_hits: AtomicU64,
    l2_hits: AtomicU64,
    misses: AtomicU64,
    singleflight_coalesced: AtomicU64,
    write_through_objects: AtomicU64,
    upstream_rpcs: AtomicU64,
    l2_write_errors: AtomicU64,
    last_activity_ms: AtomicU64,
}

impl TwoTierCache {
    pub fn new(cache_dir: PathBuf, memory_cache_bytes: usize) -> std::io::Result<Self> {
        fs::create_dir_all(cache_dir.join("objects"))?;
        fs::create_dir_all(cache_dir.join("scopes"))?;
        Ok(Self {
            cache_dir,
            l1: Mutex::new(ByteBoundedLru::new(memory_cache_bytes)),
            authorized: Mutex::new(HashSet::new()),
            inflight: Mutex::new(HashMap::new()),
            temp_seq: AtomicU64::new(1),
            l1_hits: AtomicU64::new(0),
            l2_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            singleflight_coalesced: AtomicU64::new(0),
            write_through_objects: AtomicU64::new(0),
            upstream_rpcs: AtomicU64::new(0),
            l2_write_errors: AtomicU64::new(0),
            last_activity_ms: AtomicU64::new(now_unix_ms()),
        })
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn touch_activity(&self) {
        self.last_activity_ms
            .store(now_unix_ms(), Ordering::Relaxed);
    }

    pub fn idle_millis(&self) -> u64 {
        now_unix_ms().saturating_sub(self.last_activity_ms.load(Ordering::Relaxed))
    }

    pub fn record_upstream_rpc(&self) {
        self.upstream_rpcs.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_l1_hit(&self) {
        self.l1_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn stats(&self) -> CacheStatsSnapshot {
        let l1 = self.l1.lock().unwrap();
        CacheStatsSnapshot {
            l1_hits: self.l1_hits.load(Ordering::Relaxed),
            l2_hits: self.l2_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            singleflight_coalesced: self.singleflight_coalesced.load(Ordering::Relaxed),
            write_through_objects: self.write_through_objects.load(Ordering::Relaxed),
            upstream_rpcs: self.upstream_rpcs.load(Ordering::Relaxed),
            l1_bytes: l1.current_bytes,
            l1_entries: l1.map.len(),
            l2_write_errors: self.l2_write_errors.load(Ordering::Relaxed),
        }
    }

    fn l1_key(scope: &CacheScope, kind: ObjectKind, id_bytes: &[u8]) -> L1Key {
        if kind.is_repo_scoped() {
            (scope.scope_hex.clone(), kind, id_bytes.to_vec())
        } else {
            (scope.upstream_hex.clone(), kind, id_bytes.to_vec())
        }
    }

    pub fn object_path(&self, scope: &CacheScope, kind: ObjectKind, id_bytes: &[u8]) -> PathBuf {
        if kind.is_repo_scoped() {
            self.scoped_path(&scope.scope_hex, kind, id_bytes)
        } else {
            self.shared_object_path(scope, kind, id_bytes)
        }
    }

    fn shared_object_path(&self, scope: &CacheScope, kind: ObjectKind, id_bytes: &[u8]) -> PathBuf {
        let (p1, p2, hex_id) = sharded_hex_parts(id_bytes);
        self.cache_dir
            .join("objects")
            .join(&scope.upstream_hex)
            .join(kind.dir_name())
            .join(p1)
            .join(p2)
            .join(hex_id)
    }

    fn scoped_path(&self, scope_hex: &str, kind: ObjectKind, id_bytes: &[u8]) -> PathBuf {
        let (p1, p2, hex_id) = sharded_hex_parts(id_bytes);
        self.cache_dir
            .join("scopes")
            .join(scope_hex)
            .join(kind.dir_name())
            .join(p1)
            .join(p2)
            .join(hex_id)
    }

    /// Records that `scope` is authorized to read `(kind, id_bytes)` from the shared CAS,
    /// persisting a lightweight marker so warm L2 hits survive daemon restarts.
    pub fn authorize_scope(&self, scope: &CacheScope, kind: ObjectKind, id_bytes: &[u8]) {
        if id_bytes.is_empty() || kind.is_repo_scoped() {
            return;
        }
        let key = (scope.scope_hex.clone(), kind, id_bytes.to_vec());
        let is_new = self.authorized.lock().unwrap().insert(key);
        let marker = self.scoped_path(&scope.scope_hex, kind, id_bytes);
        if is_new
            && !marker.exists()
            && let Some(parent) = marker.parent()
            && fs::create_dir_all(parent).is_ok()
        {
            let _ = fs::write(&marker, b"");
        }
    }

    pub fn is_authorized(&self, scope: &CacheScope, kind: ObjectKind, id_bytes: &[u8]) -> bool {
        if kind.is_repo_scoped() {
            return true;
        }
        let key = (scope.scope_hex.clone(), kind, id_bytes.to_vec());
        if self.authorized.lock().unwrap().contains(&key) {
            return true;
        }
        if self.scoped_path(&scope.scope_hex, kind, id_bytes).exists() {
            self.authorized.lock().unwrap().insert(key);
            return true;
        }
        false
    }

    /// Returns the byte length of `file_id` if already cached and authorized for `scope`
    /// without mutating hit/miss statistics.
    pub fn blob_size(&self, scope: &CacheScope, file_id: &[u8]) -> Option<u64> {
        if !self.is_authorized(scope, ObjectKind::File, file_id) {
            return None;
        }
        let key = Self::l1_key(scope, ObjectKind::File, file_id);
        if let Some(len) = self.l1.lock().unwrap().peek_len(&key) {
            return Some(len);
        }
        let len = fs::metadata(self.shared_object_path(scope, ObjectKind::File, file_id))
            .ok()?
            .len();
        // An empty entry for a non-empty blob is a torn write, not a size.
        (len > 0 || matches_git_blob_id(file_id, b"")).then_some(len)
    }

    pub fn put_write_through(
        &self,
        scope: &CacheScope,
        kind: ObjectKind,
        id_bytes: &[u8],
        payload: Bytes,
    ) {
        if id_bytes.is_empty() {
            return;
        }
        if self.insert_verified(scope, kind, id_bytes, &payload) {
            self.write_through_objects.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Writes `payload` to L2, authorizes `scope`, and promotes the payload to L1. Returns
    /// `false` without caching anything when a `File` or `Symlink` payload does not hash to its
    /// Git blob id, so neither a server nor a client can plant bytes under another id.
    fn insert_verified(
        &self,
        scope: &CacheScope,
        kind: ObjectKind,
        id_bytes: &[u8],
        payload: &Bytes,
    ) -> bool {
        if matches!(kind, ObjectKind::File | ObjectKind::Symlink)
            && !matches_git_blob_id(id_bytes, payload)
        {
            tracing::warn!(
                "not caching {kind:?} {}: payload does not match its Git blob id",
                hex::encode(id_bytes)
            );
            return false;
        }
        let dst = self.object_path(scope, kind, id_bytes);
        if let Err(err) = self.write_l2_atomic(&dst, id_bytes, payload) {
            self.l2_write_errors.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("failed to write L2 cache entry {}: {err}", dst.display());
        }
        self.authorize_scope(scope, kind, id_bytes);
        self.l1
            .lock()
            .unwrap()
            .put(Self::l1_key(scope, kind, id_bytes), payload.clone());
        true
    }

    pub fn lookup_cached(
        &self,
        scope: &CacheScope,
        kind: ObjectKind,
        id_bytes: &[u8],
    ) -> Option<Bytes> {
        if !self.is_authorized(scope, kind, id_bytes) {
            return None;
        }
        let key = Self::l1_key(scope, kind, id_bytes);
        if let Some(hit) = self.l1.lock().unwrap().get(&key) {
            self.l1_hits.fetch_add(1, Ordering::Relaxed);
            return Some(hit);
        }
        let path = self.object_path(scope, kind, id_bytes);
        if let Ok(bytes) = fs::read(&path) {
            if !l2_payload_is_valid(kind, id_bytes, &bytes) {
                discard_invalid_l2_entry(&path);
                return None;
            }
            self.l2_hits.fetch_add(1, Ordering::Relaxed);
            let b = Bytes::from(bytes);
            self.l1.lock().unwrap().put(key, b.clone());
            return Some(b);
        }
        None
    }

    /// Joins the in-flight fetch for `key`, or registers the caller as its leader. The lookup
    /// and the registration happen under one lock, so exactly one caller leads each fetch.
    fn join_flight(&self, key: &InflightKey) -> Flight<'_> {
        let mut map = self.inflight.lock().unwrap();
        if let Some(existing) = map.get(key) {
            return Flight::Waiter(existing.clone());
        }
        let (result, receiver) = watch::channel(None);
        map.insert(key.clone(), receiver);
        Flight::Leader(SingleflightGuard {
            inflight: &self.inflight,
            key: key.clone(),
            result,
        })
    }

    pub async fn get_or_fetch<F, Fut>(
        &self,
        scope: &CacheScope,
        kind: ObjectKind,
        id_bytes: &[u8],
        fetch_fn: F,
    ) -> Result<Bytes, tonic::Status>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Bytes, tonic::Status>>,
    {
        self.touch_activity();
        if let Some(hit) = self.lookup_cached(scope, kind, id_bytes) {
            return Ok(hit);
        }

        let inflight_key = (scope.scope_hex.clone(), kind, id_bytes.to_vec());
        let guard = loop {
            let mut waiter = match self.join_flight(&inflight_key) {
                Flight::Leader(guard) => break guard,
                Flight::Waiter(receiver) => receiver,
            };
            self.singleflight_coalesced.fetch_add(1, Ordering::Relaxed);
            if let Ok(Some(fetched)) = waiter.wait_for(Option::is_some).await.map(|v| (*v).clone())
            {
                return Ok(fetched);
            }
            // The leader failed or was cancelled; retry, leading with this caller's own request
            // if nobody else has taken over.
        };
        // A fetch that completed between the lookup above and `join_flight` already filled
        // the cache.
        if let Some(hit) = self.lookup_cached(scope, kind, id_bytes) {
            guard.publish(&hit);
            return Ok(hit);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        self.record_upstream_rpc();
        let fetched = fetch_fn().await?;
        self.insert_verified(scope, kind, id_bytes, &fetched);
        guard.publish(&fetched);
        Ok(fetched)
    }

    /// Serves a byte-range slice `[offset .. offset + max_bytes]` from L1 or L2 disk CAS
    /// if the blob is already cached and authorized for `scope`, returning `None` on a cold miss.
    pub async fn read_cached_file_range(
        &self,
        scope: &CacheScope,
        file_id: &[u8],
        offset: u64,
        max_bytes: u64,
    ) -> std::io::Result<Option<Vec<Vec<u8>>>> {
        self.touch_activity();
        if !self.is_authorized(scope, ObjectKind::File, file_id) {
            return Ok(None);
        }
        let key = Self::l1_key(scope, ObjectKind::File, file_id);
        if let Some(hit) = self.l1.lock().unwrap().get(&key) {
            self.l1_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(slice_into_chunks(&hit, offset, max_bytes)));
        }

        let path = self.shared_object_path(scope, ObjectKind::File, file_id);
        let Ok(mut file) = tokio::fs::File::open(&path).await else {
            return Ok(None);
        };
        let meta = file.metadata().await?;
        let total_len = meta.len();
        if total_len == 0 && !matches_git_blob_id(file_id, b"") {
            discard_invalid_l2_entry(&path);
            return Ok(None);
        }
        if total_len <= MAX_L1_OBJECT_BYTES as u64 {
            let mut all = Vec::with_capacity(total_len as usize);
            file.read_to_end(&mut all).await?;
            if !matches_git_blob_id(file_id, &all) {
                discard_invalid_l2_entry(&path);
                return Ok(None);
            }
            self.l2_hits.fetch_add(1, Ordering::Relaxed);
            let bytes = Bytes::from(all);
            let chunks = slice_into_chunks(&bytes, offset, max_bytes);
            self.l1.lock().unwrap().put(key, bytes);
            return Ok(Some(chunks));
        }

        self.l2_hits.fetch_add(1, Ordering::Relaxed);

        if offset >= total_len {
            return Ok(Some(Vec::new()));
        }
        file.seek(SeekFrom::Start(offset)).await?;
        let mut remaining = if max_bytes == 0 {
            total_len - offset
        } else {
            max_bytes.min(total_len - offset)
        };
        let mut chunks = Vec::new();
        while remaining > 0 {
            let to_read = (remaining as usize).min(CACHE_STREAM_CHUNK_BYTES);
            let mut buf = vec![0u8; to_read];
            let n = file.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            buf.truncate(n);
            remaining -= n as u64;
            chunks.push(buf);
        }
        Ok(chunks.into())
    }

    fn write_l2_atomic(&self, dst: &Path, id_bytes: &[u8], data: &[u8]) -> std::io::Result<()> {
        if let Ok(meta) = fs::metadata(dst)
            && (meta.len() > 0 || data.is_empty())
        {
            return Ok(());
        }
        let Some(parent) = dst.parent() else {
            return Ok(());
        };
        fs::create_dir_all(parent)?;
        let seq = self.temp_seq.fetch_add(1, Ordering::Relaxed);
        let tmp_name = format!(
            ".tmp.{}.{}.{}",
            std::process::id(),
            seq,
            hex::encode(id_bytes)
        );
        let tmp_path = parent.join(tmp_name);
        // Flush the data before the rename so that an OS crash cannot leave an empty or
        // partial file under the final name.
        let written = write_synced(&tmp_path, data).and_then(|()| fs::rename(&tmp_path, dst));
        if written.is_err() {
            let _ = fs::remove_file(&tmp_path);
        }
        written
    }
}

fn write_synced(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(data)?;
    file.sync_all()
}

/// Returns `false` only when `id_bytes` is a 20-byte SHA-1 that differs from the Git blob hash
/// (`blob <len>\0<content>`) of `content`. jj-cc-server derives `File` and `Symlink` ids this
/// way. Ids of other lengths are not checked.
fn matches_git_blob_id(id_bytes: &[u8], content: &[u8]) -> bool {
    if id_bytes.len() != 20 {
        return true;
    }
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", content.len()).as_bytes());
    hasher.update(content);
    hasher.finalize().as_slice() == id_bytes
}

/// Rejects L2 entries that the daemon cannot have written: empty payloads other than the empty
/// blob, protobuf payloads that fail to decode, and symlink targets that are not UTF-8. The
/// service stores encoded `Commit`, `ReadTreeResponse`, `View`, and `Operation` messages, raw
/// symlink targets, and raw file bytes.
fn l2_payload_is_valid(kind: ObjectKind, id_bytes: &[u8], bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return matches!(kind, ObjectKind::File | ObjectKind::Symlink)
            && matches_git_blob_id(id_bytes, bytes);
    }
    match kind {
        ObjectKind::File => matches_git_blob_id(id_bytes, bytes),
        ObjectKind::Symlink => {
            std::str::from_utf8(bytes).is_ok() && matches_git_blob_id(id_bytes, bytes)
        }
        ObjectKind::Commit => Commit::decode(bytes).is_ok(),
        ObjectKind::Tree => ReadTreeResponse::decode(bytes).is_ok(),
        ObjectKind::View => View::decode(bytes).is_ok(),
        ObjectKind::Operation => Operation::decode(bytes).is_ok(),
    }
}

fn discard_invalid_l2_entry(path: &Path) {
    tracing::warn!("discarding invalid L2 cache entry {}", path.display());
    let _ = fs::remove_file(path);
}

fn sharded_hex_parts(id_bytes: &[u8]) -> (String, String, String) {
    let hex_id = hex::encode(id_bytes);
    let (p1, p2) = if hex_id.len() >= 4 {
        (hex_id[0..2].to_string(), hex_id[2..4].to_string())
    } else {
        ("00".to_string(), "00".to_string())
    };
    (p1, p2, hex_id)
}

pub(crate) fn slice_into_chunks(bytes: &[u8], offset: u64, max_bytes: u64) -> Vec<Vec<u8>> {
    let total = bytes.len() as u64;
    if offset >= total {
        return Vec::new();
    }
    let start = offset as usize;
    let end = if max_bytes == 0 {
        bytes.len()
    } else {
        start.saturating_add(max_bytes as usize).min(bytes.len())
    };
    if start == end {
        return Vec::new();
    }
    bytes[start..end]
        .chunks(CACHE_STREAM_CHUNK_BYTES)
        .map(|s| s.to_vec())
        .collect()
}

pub fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// FIPS 180-4 SHA-256 used to derive collision-resistant 256-bit [`CacheScope`] directory keys.
fn sha256_bytes(input: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((input.len() + 9).div_ceil(64) * 64);
    padded.extend_from_slice(input);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in chunk.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut out = [0u8; 32];
    for (i, word) in state.into_iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    fn tree_payload(label: &str) -> Bytes {
        let tree = ReadTreeResponse {
            tree_id: vec![0xab; 20],
            entries: Vec::new(),
            next_page_token: label.to_string(),
        };
        Bytes::from(tree.encode_to_vec())
    }

    fn git_blob_id(content: &[u8]) -> Vec<u8> {
        let mut hasher = Sha1::new();
        hasher.update(format!("blob {}\0", content.len()).as_bytes());
        hasher.update(content);
        hasher.finalize().to_vec()
    }

    #[test]
    fn sha256_matches_nist_vectors() {
        assert_eq!(
            hex::encode(sha256_bytes(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex::encode(sha256_bytes(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[tokio::test]
    async fn two_tier_cache_promotes_l2_to_l1_coalesces_singleflight_and_isolates_scopes() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Arc::new(TwoTierCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap());
        let calls = Arc::new(AtomicUsize::new(0));
        let digest = vec![0xab; 20];
        let scope_a = CacheScope::new("http://127.0.0.1:8080", "repo-alpha");
        let scope_b = CacheScope::new("http://127.0.0.1:8080", "repo-beta");

        let mut set = tokio::task::JoinSet::new();
        for _ in 0..8 {
            let c = Arc::clone(&cache);
            let cnt = Arc::clone(&calls);
            let id = digest.clone();
            let sc = scope_a.clone();
            set.spawn(async move {
                c.get_or_fetch(&sc, ObjectKind::Tree, &id, || async move {
                    cnt.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                    Ok(tree_payload("tree-payload"))
                })
                .await
                .unwrap()
            });
        }
        while let Some(res) = set.join_next().await {
            assert_eq!(res.unwrap(), tree_payload("tree-payload"));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let st = cache.stats();
        assert_eq!(st.misses, 1);
        assert!(st.singleflight_coalesced >= 1);

        // Unauthorized repository `scope_b` must NOT hit `scope_a`'s cached payload without
        // upstream verification.
        assert!(
            cache
                .lookup_cached(&scope_b, ObjectKind::Tree, &digest)
                .is_none()
        );

        // Drop L1 by opening a fresh TwoTierCache over the same L2 directory and verify `scope_a`
        // L2 hit!
        let cold_l1_cache = TwoTierCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap();
        assert!(
            cold_l1_cache
                .lookup_cached(&scope_b, ObjectKind::Tree, &digest)
                .is_none()
        );
        let hit = cold_l1_cache
            .get_or_fetch(&scope_a, ObjectKind::Tree, &digest, || async {
                panic!("should hit L2 disk CAS without calling upstream");
            })
            .await
            .unwrap();
        assert_eq!(hit, tree_payload("tree-payload"));
        assert_eq!(cold_l1_cache.stats().l2_hits, 1);
    }

    #[tokio::test]
    async fn cached_file_range_serves_partial_slices_across_l1_and_large_l2_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let cache = TwoTierCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap();
        let scope_a = CacheScope::new("http://127.0.0.1:8080", "repo-alpha");
        let scope_b = CacheScope::new("http://127.0.0.1:8080", "repo-beta");
        let payload: Vec<u8> = (0..(MAX_L1_OBJECT_BYTES + 8192))
            .map(|i| (i % 251) as u8)
            .collect();
        let large_id = git_blob_id(&payload);
        cache.put_write_through(
            &scope_a,
            ObjectKind::File,
            &large_id,
            Bytes::from(payload.clone()),
        );
        assert_eq!(cache.stats().l1_entries, 0, ">4 MiB blob stays in L2 only");
        assert_eq!(
            cache.blob_size(&scope_a, &large_id),
            Some(payload.len() as u64)
        );
        assert_eq!(cache.blob_size(&scope_b, &large_id), None);

        let slice_chunks = cache
            .read_cached_file_range(&scope_a, &large_id, 4 * 1024 * 1024, 4096)
            .await
            .unwrap()
            .expect("present in L2");
        let joined: Vec<u8> = slice_chunks.into_iter().flatten().collect();
        assert_eq!(joined, &payload[4 * 1024 * 1024..4 * 1024 * 1024 + 4096]);
    }

    #[tokio::test]
    async fn singleflight_waiter_sees_result_published_before_it_starts_waiting() {
        let dir = tempfile::tempdir().unwrap();
        let cache = TwoTierCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap();
        let scope = CacheScope::new("http://127.0.0.1:8080", "repo-alpha");
        let key = (scope.scope_hex.clone(), ObjectKind::Tree, vec![0x11; 20]);

        let Flight::Leader(leader) = cache.join_flight(&key) else {
            panic!("first caller leads")
        };
        let Flight::Waiter(mut waiter) = cache.join_flight(&key) else {
            panic!("second waits")
        };
        // The leader finishes after the waiter joined but before it starts waiting. A waiter
        // parked on an edge-triggered `Notify` lost its only wakeup in this window.
        leader.publish(&tree_payload("published"));
        drop(leader);
        let seen = tokio::time::timeout(Duration::from_secs(5), waiter.wait_for(Option::is_some))
            .await
            .expect("waiter must not hang")
            .map(|v| (*v).clone())
            .unwrap();
        assert_eq!(seen, Some(tree_payload("published")));
        assert!(cache.inflight.lock().unwrap().is_empty());

        // A leader that fails or is cancelled closes the channel instead of stranding waiters.
        let Flight::Leader(failed) = cache.join_flight(&key) else {
            panic!("new fetch leads")
        };
        let Flight::Waiter(mut orphan) = cache.join_flight(&key) else {
            panic!("second waits")
        };
        drop(failed);
        assert!(orphan.wait_for(Option::is_some).await.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn singleflight_never_strands_waiters_on_multi_thread_runtime() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Arc::new(TwoTierCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap());
        let scope = CacheScope::new("http://127.0.0.1:8080", "repo-alpha");
        let calls = Arc::new(AtomicUsize::new(0));
        let rounds = async {
            for round in 0u32..500 {
                let id = round.to_be_bytes().repeat(5);
                let mut set = tokio::task::JoinSet::new();
                for _ in 0..16 {
                    let (cache, scope, id) = (Arc::clone(&cache), scope.clone(), id.clone());
                    let calls = Arc::clone(&calls);
                    set.spawn(async move {
                        cache
                            .get_or_fetch(&scope, ObjectKind::Tree, &id, || async move {
                                calls.fetch_add(1, Ordering::SeqCst);
                                tokio::task::yield_now().await;
                                Ok(tree_payload("payload"))
                            })
                            .await
                    });
                }
                while let Some(res) = set.join_next().await {
                    assert_eq!(res.unwrap().unwrap(), tree_payload("payload"));
                }
            }
        };
        tokio::time::timeout(Duration::from_secs(120), rounds)
            .await
            .expect("a singleflight waiter never completed");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            500,
            "one upstream fetch per object"
        );
    }

    #[tokio::test]
    async fn singleflight_waiters_retry_with_their_own_fetch_after_leader_failure() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Arc::new(TwoTierCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap());
        let scope = CacheScope::new("http://127.0.0.1:8080", "repo-alpha");
        let id = vec![0x22; 20];
        let key = (scope.scope_hex.clone(), ObjectKind::Tree, id.clone());

        let Flight::Leader(leader) = cache.join_flight(&key) else {
            panic!("test leads")
        };
        let waiter = tokio::spawn({
            let (cache, scope, id) = (Arc::clone(&cache), scope.clone(), id.clone());
            async move {
                cache
                    .get_or_fetch(&scope, ObjectKind::Tree, &id, || async {
                        Ok(tree_payload("retried"))
                    })
                    .await
            }
        });
        while cache.stats().singleflight_coalesced == 0 {
            tokio::task::yield_now().await;
        }
        drop(leader);
        assert_eq!(waiter.await.unwrap().unwrap(), tree_payload("retried"));
        assert_eq!(cache.stats().misses, 1);
    }

    #[tokio::test]
    async fn shared_cas_entries_are_partitioned_per_upstream() {
        let dir = tempfile::tempdir().unwrap();
        let cache = TwoTierCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap();
        let scope_a = CacheScope::new("http://server-a:8080", "repo");
        let scope_b = CacheScope::new("http://server-b:8080/", "repo");
        let tree_id = vec![0x42; 20];

        cache.put_write_through(&scope_a, ObjectKind::Tree, &tree_id, tree_payload("from-a"));
        // Server B proves reachability by serving the id itself and must keep its own bytes.
        let from_b = cache
            .get_or_fetch(&scope_b, ObjectKind::Tree, &tree_id, || async {
                Ok(tree_payload("from-b"))
            })
            .await
            .unwrap();
        assert_eq!(from_b, tree_payload("from-b"));
        assert_eq!(
            cache
                .lookup_cached(&scope_a, ObjectKind::Tree, &tree_id)
                .unwrap(),
            tree_payload("from-a")
        );

        let restarted = TwoTierCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap();
        assert_eq!(
            restarted.lookup_cached(&scope_a, ObjectKind::Tree, &tree_id),
            Some(tree_payload("from-a"))
        );
        assert_eq!(
            restarted.lookup_cached(&scope_b, ObjectKind::Tree, &tree_id),
            Some(tree_payload("from-b"))
        );
    }

    #[tokio::test]
    async fn empty_or_undecodable_l2_entries_are_discarded_and_refetched() {
        let dir = tempfile::tempdir().unwrap();
        let scope = CacheScope::new("http://127.0.0.1:8080", "repo-alpha");
        let tree_id = vec![0x5a; 20];
        let cache = TwoTierCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap();
        cache.put_write_through(&scope, ObjectKind::Tree, &tree_id, tree_payload("good"));
        let path = cache.object_path(&scope, ObjectKind::Tree, &tree_id);

        for torn in [&b""[..], &[0u8; 64][..]] {
            fs::write(&path, torn).unwrap();
            let restarted = TwoTierCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap();
            assert_eq!(
                restarted.lookup_cached(&scope, ObjectKind::Tree, &tree_id),
                None
            );
            assert!(!path.exists(), "invalid entry should be deleted");
            let refetched = restarted
                .get_or_fetch(&scope, ObjectKind::Tree, &tree_id, || async {
                    Ok(tree_payload("good"))
                })
                .await
                .unwrap();
            assert_eq!(refetched, tree_payload("good"));
            assert!(path.exists(), "refetched entry should be rewritten");
        }
    }

    #[tokio::test]
    async fn file_payloads_are_cached_only_under_their_git_blob_id() {
        let dir = tempfile::tempdir().unwrap();
        let cache = TwoTierCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap();
        let scope = CacheScope::new("http://127.0.0.1:8080", "repo-alpha");
        let content = Bytes::from_static(b"hello\n");
        let id = git_blob_id(&content);
        assert_eq!(hex::encode(&id), "ce013625030ba8dba906f756967f9e9ca394464a");
        let forged = Bytes::from_static(b"forged\n");

        cache.put_write_through(&scope, ObjectKind::File, &id, forged.clone());
        assert_eq!(cache.blob_size(&scope, &id), None);
        let fetched = cache
            .get_or_fetch(&scope, ObjectKind::File, &id, || async {
                Ok(forged.clone())
            })
            .await
            .unwrap();
        assert_eq!(
            fetched, forged,
            "the caller still receives the upstream bytes"
        );
        assert!(!cache.object_path(&scope, ObjectKind::File, &id).exists());

        cache.put_write_through(&scope, ObjectKind::File, &id, content.clone());
        assert_eq!(cache.blob_size(&scope, &id), Some(content.len() as u64));

        // An empty L2 entry is valid only for the empty blob.
        let empty_id = git_blob_id(b"");
        cache.put_write_through(&scope, ObjectKind::File, &empty_id, Bytes::new());
        let restarted = TwoTierCache::new(dir.path().to_path_buf(), 1024 * 1024).unwrap();
        assert_eq!(restarted.blob_size(&scope, &empty_id), Some(0));
        let empty_range = restarted
            .read_cached_file_range(&scope, &empty_id, 0, 0)
            .await
            .unwrap();
        assert_eq!(empty_range, Some(Vec::new()));
        fs::write(restarted.object_path(&scope, ObjectKind::File, &id), b"").unwrap();
        assert_eq!(restarted.blob_size(&scope, &id), None);
        assert_eq!(
            restarted
                .read_cached_file_range(&scope, &id, 0, 0)
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn failed_l2_writes_are_counted_and_the_fetched_bytes_still_returned() {
        let dir = tempfile::tempdir().unwrap();
        // L1 is too small for the payload, so after the fetch it is in neither tier.
        let cache = TwoTierCache::new(dir.path().to_path_buf(), 4).unwrap();
        let scope = CacheScope::new("http://127.0.0.1:8080", "repo-alpha");
        // A regular file where the upstream partition belongs makes every L2 write fail.
        fs::write(dir.path().join("objects").join(&scope.upstream_hex), b"").unwrap();
        let content = Bytes::from_static(b"file contents larger than L1\n");
        let id = git_blob_id(&content);

        let fetched = cache
            .get_or_fetch(&scope, ObjectKind::File, &id, || async {
                Ok(content.clone())
            })
            .await
            .unwrap();
        assert_eq!(fetched, content);
        assert_eq!(cache.stats().l2_write_errors, 1);
        assert_eq!(
            cache
                .read_cached_file_range(&scope, &id, 0, 0)
                .await
                .unwrap(),
            None
        );
    }
}
