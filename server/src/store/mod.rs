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

//! Server persistence traits (`MetadataStore` and `BlobStore`) and `CompositeStore`
//! coordinator for jj-commit-cloud.
//!
//! # Authority and Crash-Safety Ordering
//! `CompositeStore` enforces write-before-publish ordering across metadata and object
//! stores:
//! 1. `put_file` writes bytes to `BlobStore` (unless smaller than `inline_blob_threshold`) before
//!    recording `BlobRecord` and `(project_id, blob_id)` ownership in `MetadataStore`.
//! 2. `put_tree`, `put_commit`, `put_operation`, and `update_op_heads` validate that referenced
//!    child blobs, symlinks, subtrees, parent commits, views, and operations already exist in
//!    `MetadataStore` before publishing parent records.
//!
//! # Concurrency and Merge Invariants
//! Content-addressed writes (`Commit`, `Tree`, `Symlink`, `Blob`, `View`, `Operation`) are
//! immutable and idempotent. `MetadataStore::update_op_heads` and
//! `MetadataStore::publish_operation_atomic` execute atomically per `repo_id`, removing
//! superseded `old_ids` while retaining concurrent sibling operation heads so clients and
//! `ReconcileOpHeads` can merge divergent operations without locks.
//!
//! # Cross-Process and RPC Boundaries
//! Errors propagate as typed [`StoreError`] variants that convert deterministically into
//! gRPC `tonic::Status` codes at the `BackendService`, `OpStoreService`, and
//! `WorkspaceService` boundaries.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use bytes::Bytes;
use cc_common::backend::*;
use cc_common::op_store::*;
use cc_common::workspace::WorkspaceState;

pub mod error;
pub mod gcsblob;
pub mod localblob;
pub mod memorystore;
pub mod replicatedblob;
pub mod spannerstore;
pub mod sqlitestore;

pub use error::{StoreError, StoreResult};
pub use gcsblob::{GcsBlobStore, GcsConnectConfig, IMMUTABLE_CACHE_CONTROL};
pub use localblob::LocalFsBlobStore;
pub use memorystore::{
    CommitId, FileId, MemoryBlobStore, MemoryMetadataStore, OpId, ProjectId, RepoId, SymlinkId,
    TreeId, ViewId,
};
pub use replicatedblob::{ReplicatedBlobStore, ReplicatedBlobStoreConfig, ReplicationStats};
pub use spannerstore::{SpannerConnectConfig, SpannerMetadataStore};
pub use sqlitestore::SqliteMetadataStore;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobRecord {
    pub blob_id: Vec<u8>,
    pub content_size: u64,
    pub inline_content: Option<Bytes>,
    pub primary_location: Option<String>,
    pub replicated_locations: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IdPrefixResolution {
    NoMatch,
    SingleMatch(Vec<u8>),
    AmbiguousMatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChangePrefixResolution {
    NoMatch,
    SingleMatch {
        change_id: Vec<u8>,
        commit_ids: Vec<CommitId>,
    },
    AmbiguousMatch,
}

#[async_trait]
pub trait MetadataStore: Send + Sync {
    fn backend_name(&self) -> &'static str;

    async fn register_project(&self, project_id: &str, name: Option<&str>) -> StoreResult<()>;
    async fn is_project_registered(&self, project_id: &str) -> StoreResult<bool>;
    async fn get_repo_project_id(&self, repo_id: &str) -> StoreResult<Option<String>>;
    async fn register_repo_in_project(
        &self,
        repo_id: &str,
        project_id: &str,
        name: Option<&str>,
    ) -> StoreResult<String>;

    async fn is_repo_registered(&self, repo_id: &str) -> StoreResult<bool> {
        Ok(self.get_repo_project_id(repo_id).await?.is_some())
    }

    async fn register_repo(&self, repo_id: &str, name: Option<&str>) -> StoreResult<()> {
        self.register_repo_in_project(repo_id, repo_id, name)
            .await
            .map(|_| ())
    }

    async fn get_commit(&self, project_id: &str, commit_id: &[u8]) -> StoreResult<Option<Commit>>;
    async fn get_commit_generation(
        &self,
        project_id: &str,
        commit_id: &[u8],
    ) -> StoreResult<Option<u64>>;
    /// Returns the generation numbers of the commits in `commit_ids` that exist in
    /// `project_id`; missing commits are omitted. The default implementation runs up to 16
    /// `get_commit_generation` reads concurrently, so backends with a batched read should
    /// override it.
    async fn get_commit_generations(
        &self,
        project_id: &str,
        commit_ids: &[Vec<u8>],
    ) -> StoreResult<HashMap<Vec<u8>, u64>> {
        use futures::{StreamExt as _, TryStreamExt as _};

        let mut lookups = Vec::with_capacity(commit_ids.len());
        for commit_id in commit_ids {
            lookups.push(async move {
                let generation = self.get_commit_generation(project_id, commit_id).await?;
                Ok::<_, StoreError>(generation.map(|g| (commit_id.clone(), g)))
            });
        }
        let found: Vec<_> = futures::stream::iter(lookups)
            .buffer_unordered(16)
            .try_collect()
            .await?;
        Ok(found.into_iter().flatten().collect())
    }
    async fn put_commit(
        &self,
        project_id: &str,
        commit_id: &[u8],
        commit: &Commit,
        generation: u64,
    ) -> StoreResult<()>;
    async fn list_commit_ids(
        &self,
        project_id: &str,
        after_commit_id: Option<&[u8]>,
        limit: usize,
    ) -> StoreResult<Vec<CommitId>>;
    async fn current_commit_watermark(&self, project_id: &str) -> StoreResult<Vec<u8>>;
    async fn list_commit_ids_since(
        &self,
        project_id: &str,
        since_token: &[u8],
        limit: usize,
    ) -> StoreResult<(Vec<CommitId>, Vec<u8>, bool)>;
    async fn resolve_commit_id_prefix(
        &self,
        project_id: &str,
        min_id: &[u8],
        max_id: &[u8],
    ) -> StoreResult<Vec<CommitId>>;
    async fn resolve_change_id_prefix(
        &self,
        project_id: &str,
        min_id: &[u8],
        max_id: &[u8],
    ) -> StoreResult<Vec<(Vec<u8>, Vec<CommitId>)>>;

    async fn get_tree(
        &self,
        project_id: &str,
        tree_id: &[u8],
    ) -> StoreResult<Option<Vec<TreeEntry>>>;
    async fn get_tree_page(
        &self,
        project_id: &str,
        tree_id: &[u8],
        after_name: Option<&str>,
        limit: usize,
    ) -> StoreResult<Option<Vec<TreeEntry>>> {
        let Some(entries) = self.get_tree(project_id, tree_id).await? else {
            return Ok(None);
        };
        let iter = entries.into_iter().filter(|e| {
            after_name
                .filter(|a| !a.is_empty())
                .is_none_or(|after| e.name.as_bytes() > after.as_bytes())
        });
        Ok(Some(iter.take(limit).collect()))
    }
    async fn put_tree(
        &self,
        project_id: &str,
        tree_id: &[u8],
        entries: &[TreeEntry],
    ) -> StoreResult<()>;
    async fn find_existing_trees(
        &self,
        project_id: &str,
        tree_ids: &[Vec<u8>],
    ) -> StoreResult<HashSet<Vec<u8>>>;

    async fn get_symlink(&self, project_id: &str, symlink_id: &[u8])
    -> StoreResult<Option<String>>;
    async fn put_symlink(
        &self,
        project_id: &str,
        symlink_id: &[u8],
        target: &str,
    ) -> StoreResult<()>;
    async fn find_existing_symlinks(
        &self,
        project_id: &str,
        symlink_ids: &[Vec<u8>],
    ) -> StoreResult<HashSet<Vec<u8>>>;

    async fn get_blob_record(
        &self,
        project_id: &str,
        blob_id: &[u8],
    ) -> StoreResult<Option<BlobRecord>>;
    async fn get_blob_sizes_batch(
        &self,
        project_id: &str,
        blob_ids: &[Vec<u8>],
    ) -> StoreResult<HashMap<Vec<u8>, u64>>;
    async fn get_global_blob_record(&self, blob_id: &[u8]) -> StoreResult<Option<BlobRecord>>;
    async fn put_blob_record(&self, project_id: &str, record: &BlobRecord) -> StoreResult<()>;
    async fn add_blob_replica_location(&self, blob_id: &[u8], location: &str) -> StoreResult<()>;

    async fn get_operation(&self, repo_id: &str, op_id: &[u8]) -> StoreResult<Option<Operation>>;
    async fn put_operation(&self, repo_id: &str, op_id: &[u8], op: &Operation) -> StoreResult<()>;
    async fn resolve_operation_id_prefix(
        &self,
        repo_id: &str,
        min_id: &[u8],
        max_id: &[u8],
    ) -> StoreResult<Vec<OpId>>;

    async fn get_view(&self, repo_id: &str, view_id: &[u8]) -> StoreResult<Option<View>>;
    async fn put_view(&self, repo_id: &str, view_id: &[u8], view: &View) -> StoreResult<()>;

    async fn get_op_heads(&self, repo_id: &str) -> StoreResult<Option<Vec<OpId>>>;
    async fn update_op_heads(
        &self,
        repo_id: &str,
        old_ids: &[Vec<u8>],
        new_id: &[u8],
    ) -> StoreResult<Vec<OpId>>;
    async fn publish_operation_atomic(
        &self,
        repo_id: &str,
        view_id: &[u8],
        view: &View,
        op_id: &[u8],
        op: &Operation,
        old_ids: &[Vec<u8>],
    ) -> StoreResult<Vec<OpId>> {
        self.put_view(repo_id, view_id, view).await?;
        self.put_operation(repo_id, op_id, op).await?;
        self.update_op_heads(repo_id, old_ids, op_id).await
    }
    async fn compare_and_swap_op_heads(
        &self,
        repo_id: &str,
        expected_old_ids: &[Vec<u8>],
        new_id: &[u8],
    ) -> StoreResult<()>;

    async fn get_workspace(
        &self,
        repo_id: &str,
        user: &str,
        workspace_name: &str,
    ) -> StoreResult<Option<WorkspaceState>>;
    async fn put_workspace(&self, workspace: &WorkspaceState) -> StoreResult<()>;
    async fn list_workspaces(&self, repo_id: &str) -> StoreResult<Vec<WorkspaceState>>;
    async fn delete_workspace(
        &self,
        repo_id: &str,
        user: &str,
        workspace_name: &str,
    ) -> StoreResult<bool>;
}

#[async_trait]
pub trait BlobStore: Send + Sync {
    fn location_id(&self) -> &str;

    async fn put_blob(&self, blob_id: &[u8], data: Bytes) -> StoreResult<()>;

    async fn get_blob(&self, blob_id: &[u8]) -> StoreResult<Option<Bytes>>;

    async fn get_blob_range(
        &self,
        blob_id: &[u8],
        offset: u64,
        length: Option<u64>,
    ) -> StoreResult<Option<Bytes>> {
        let Some(full) = self.get_blob(blob_id).await? else {
            return Ok(None);
        };
        slice_blob_range(&full, offset, length).map(Some)
    }

    /// Reads a blob slice using an already-fetched [`BlobRecord`] so replication-aware
    /// coordinators can route origin fallback without a second metadata lookup.
    async fn get_blob_range_with_record(
        &self,
        record: &BlobRecord,
        offset: u64,
        length: Option<u64>,
    ) -> StoreResult<Option<Bytes>> {
        self.get_blob_range(&record.blob_id, offset, length).await
    }

    async fn has_blob(&self, blob_id: &[u8]) -> StoreResult<bool> {
        Ok(self.get_blob(blob_id).await?.is_some())
    }
}

pub fn slice_blob_range(data: &Bytes, offset: u64, length: Option<u64>) -> StoreResult<Bytes> {
    let total_len = data.len() as u64;
    if total_len == 0 {
        if offset == 0 {
            return Ok(Bytes::new());
        }
        return Err(StoreError::OutOfRange(format!(
            "requested read at offset {offset}, file size is 0"
        )));
    }
    if offset >= total_len {
        return Err(StoreError::OutOfRange(format!(
            "requested read at offset {offset}, file size is {total_len}"
        )));
    }
    let end = match length {
        Some(0) | None => total_len,
        Some(max_len) => offset.saturating_add(max_len).min(total_len),
    };
    Ok(data.slice((offset as usize)..(end as usize)))
}

/// Computes inclusive `[min_bytes, max_bytes]` lexicographical bounds of length `id_len`
/// matching `hex_prefix` (supports both even and odd nibble lengths). Returns `None` for a
/// valid hex prefix longer than an ID, which matches no ID.
pub fn hex_prefix_byte_bounds(
    hex_prefix: &str,
    id_len: usize,
) -> StoreResult<Option<(Vec<u8>, Vec<u8>)>> {
    if hex_prefix.len() > id_len * 2 {
        for b in hex_prefix.bytes() {
            decode_hex_nibble(b)?;
        }
        return Ok(None);
    }
    let mut min_bytes = vec![0u8; id_len];
    let mut max_bytes = vec![0xffu8; id_len];

    let bytes = hex_prefix.as_bytes();
    let full_bytes = bytes.len() / 2;
    for i in 0..full_bytes {
        let hi = decode_hex_nibble(bytes[2 * i])?;
        let lo = decode_hex_nibble(bytes[2 * i + 1])?;
        let b = (hi << 4) | lo;
        min_bytes[i] = b;
        max_bytes[i] = b;
    }
    if bytes.len() % 2 == 1 {
        let hi = decode_hex_nibble(bytes[bytes.len() - 1])?;
        min_bytes[full_bytes] = hi << 4;
        max_bytes[full_bytes] = (hi << 4) | 0x0f;
    }
    Ok(Some((min_bytes, max_bytes)))
}

/// Normalizes a `ChangeId` prefix given in either jj `z-k` reverse-hex or standard
/// `0-9a-f` hex into lowercase forward hexadecimal digits.
pub fn normalize_change_id_prefix_to_hex(prefix: &str) -> StoreResult<String> {
    if prefix.is_empty() {
        return Ok(String::new());
    }
    let all_rev_hex = prefix
        .bytes()
        .all(|b| matches!(b, b'k'..=b'z' | b'K'..=b'Z'));
    if all_rev_hex {
        let mut out = String::with_capacity(prefix.len());
        for b in prefix.bytes() {
            let lower = b.to_ascii_lowercase();
            let nibble = b'z' - lower;
            out.push(char::from_digit(u32::from(nibble), 16).unwrap());
        }
        return Ok(out);
    }
    let all_std_hex = prefix.bytes().all(|b| b.is_ascii_hexdigit());
    if all_std_hex {
        return Ok(prefix.to_ascii_lowercase());
    }
    Err(StoreError::InvalidArgument(format!(
        "invalid change ID prefix '{prefix}' (expected z-k reverse-hex or 0-9a-f hex)"
    )))
}

fn decode_hex_nibble(b: u8) -> StoreResult<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(StoreError::InvalidArgument(format!(
            "invalid hex character '{}'",
            b as char
        ))),
    }
}

#[derive(Clone, Debug)]
pub struct CompositeStoreConfig {
    /// Blobs with byte length `<= inline_blob_threshold` are stored inline in `MetadataStore`
    /// without an object-store roundtrip. `0` sends all non-empty blobs to `BlobStore`.
    pub inline_blob_threshold: usize,
    /// Enforce bottom-up dependency existence checks on tree, commit, operation, and op-head
    /// writes.
    pub validate_dependencies: bool,
}

impl Default for CompositeStoreConfig {
    fn default() -> Self {
        Self {
            inline_blob_threshold: 0,
            validate_dependencies: true,
        }
    }
}

/// Coordinates `MetadataStore` and `BlobStore` operations with write-before-publish
/// crash safety and bottom-up dependency validation.
#[derive(Clone)]
pub struct CompositeStore {
    metadata: Arc<dyn MetadataStore>,
    blobs: Arc<dyn BlobStore>,
    config: CompositeStoreConfig,
    registered_projects: Arc<RwLock<HashSet<String>>>,
    repo_projects: Arc<RwLock<HashMap<String, String>>>,
}

impl CompositeStore {
    pub fn new(metadata: Arc<dyn MetadataStore>, blobs: Arc<dyn BlobStore>) -> Self {
        Self::with_config(metadata, blobs, CompositeStoreConfig::default())
    }

    pub fn with_config(
        metadata: Arc<dyn MetadataStore>,
        blobs: Arc<dyn BlobStore>,
        config: CompositeStoreConfig,
    ) -> Self {
        Self {
            metadata,
            blobs,
            config,
            registered_projects: Arc::new(RwLock::new(HashSet::new())),
            repo_projects: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn in_memory() -> Self {
        Self::new(
            Arc::new(MemoryMetadataStore::new()),
            Arc::new(MemoryBlobStore::new()),
        )
    }

    pub fn metadata(&self) -> &Arc<dyn MetadataStore> {
        &self.metadata
    }

    pub fn blobs(&self) -> &Arc<dyn BlobStore> {
        &self.blobs
    }

    /// Resolves and caches the immutable `project_id` bound to `repo_id`, returning
    /// `StoreError::RepoNotFound` when `repo_id` has not been registered.
    pub async fn resolve_repo_project(&self, repo_id: &str, action: &str) -> StoreResult<String> {
        if let Some(cached) = self.repo_projects.read().unwrap().get(repo_id).cloned() {
            return Ok(cached);
        }
        let project_id = self
            .metadata
            .get_repo_project_id(repo_id)
            .await?
            .ok_or_else(|| StoreError::RepoNotFound {
                repo_id: repo_id.to_string(),
                action: action.to_string(),
            })?;
        self.repo_projects
            .write()
            .unwrap()
            .insert(repo_id.to_string(), project_id.clone());
        Ok(project_id)
    }

    /// Resolves `namespace_id` for `BackendService` object operations: checks `Projects`
    /// first (`project_id`), then falls back to `Repositories(namespace_id).project_id` so
    /// repository-scoped callers (`jj-vfs`) resolve shared project objects transparently.
    pub async fn resolve_project_namespace(
        &self,
        namespace_id: &str,
        action: &str,
    ) -> StoreResult<String> {
        if self
            .registered_projects
            .read()
            .unwrap()
            .contains(namespace_id)
        {
            return Ok(namespace_id.to_string());
        }
        if self.metadata.is_project_registered(namespace_id).await? {
            self.registered_projects
                .write()
                .unwrap()
                .insert(namespace_id.to_string());
            return Ok(namespace_id.to_string());
        }
        self.resolve_repo_project(namespace_id, action).await
    }

    pub async fn ensure_repo_registered(&self, repo_id: &str, action: &str) -> StoreResult<()> {
        let _ = self.resolve_repo_project(repo_id, action).await?;
        Ok(())
    }

    pub async fn register_project(&self, project_id: &str, name: Option<&str>) -> StoreResult<()> {
        self.metadata.register_project(project_id, name).await?;
        self.registered_projects
            .write()
            .unwrap()
            .insert(project_id.to_string());
        Ok(())
    }

    pub async fn register_repo_in_project(
        &self,
        repo_id: &str,
        project_id: Option<&str>,
        name: Option<&str>,
    ) -> StoreResult<String> {
        let target_project = project_id
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(repo_id);
        let bound_project = self
            .metadata
            .register_repo_in_project(repo_id, target_project, name)
            .await?;
        self.registered_projects
            .write()
            .unwrap()
            .insert(bound_project.clone());
        self.repo_projects
            .write()
            .unwrap()
            .insert(repo_id.to_string(), bound_project.clone());
        Ok(bound_project)
    }

    pub async fn register_repo(&self, repo_id: &str, name: Option<&str>) -> StoreResult<()> {
        self.register_repo_in_project(repo_id, None, name)
            .await
            .map(|_| ())
    }

    pub async fn put_file(
        &self,
        namespace_id: &str,
        file_id: &[u8],
        content: Bytes,
    ) -> StoreResult<()> {
        let project_id = self
            .resolve_project_namespace(namespace_id, "writing files")
            .await?;

        let content_size = content.len() as u64;
        if content.is_empty() || content.len() <= self.config.inline_blob_threshold {
            let record = BlobRecord {
                blob_id: file_id.to_vec(),
                content_size,
                inline_content: Some(content),
                primary_location: None,
                replicated_locations: Vec::new(),
            };
            self.metadata.put_blob_record(&project_id, &record).await?;
            return Ok(());
        }

        let local_location = self.blobs.location_id().to_string();
        // Write-before-publish: persist bytes in BlobStore before exposing BlobRecord in
        // MetadataStore.
        self.blobs.put_blob(file_id, content).await?;

        let existing = self.metadata.get_global_blob_record(file_id).await?;
        let (primary_location, mut replicated_locations) = match existing {
            Some(prev) => (
                prev.primary_location
                    .or_else(|| Some(local_location.clone())),
                prev.replicated_locations,
            ),
            None => (Some(local_location.clone()), vec![local_location.clone()]),
        };
        if !replicated_locations
            .iter()
            .any(|loc| loc == &local_location)
        {
            replicated_locations.push(local_location);
        }

        let record = BlobRecord {
            blob_id: file_id.to_vec(),
            content_size,
            inline_content: None,
            primary_location,
            replicated_locations,
        };
        self.metadata.put_blob_record(&project_id, &record).await
    }

    pub async fn get_file_range(
        &self,
        namespace_id: &str,
        file_id: &[u8],
        offset: u64,
        max_bytes: Option<u64>,
    ) -> StoreResult<Bytes> {
        let project_id = self
            .resolve_project_namespace(namespace_id, "reading files")
            .await?;

        let record = self
            .metadata
            .get_blob_record(&project_id, file_id)
            .await?
            .ok_or_else(|| StoreError::ObjectNotFound {
                message: "file should have been present in cloud database".to_string(),
            })?;

        if let Some(inline) = &record.inline_content {
            return slice_blob_range(inline, offset, max_bytes);
        }

        let bytes = self
            .blobs
            .get_blob_range_with_record(&record, offset, max_bytes)
            .await?
            .ok_or_else(|| StoreError::ObjectNotFound {
                message: format!(
                    "blob payload missing in object store for file {}",
                    hex::encode(file_id)
                ),
            })?;
        Ok(bytes)
    }

    pub async fn put_symlink(
        &self,
        namespace_id: &str,
        symlink_id: &[u8],
        target: &str,
    ) -> StoreResult<()> {
        let project_id = self
            .resolve_project_namespace(namespace_id, "writing symlinks")
            .await?;
        self.metadata
            .put_symlink(&project_id, symlink_id, target)
            .await
    }

    pub async fn get_symlink(&self, namespace_id: &str, symlink_id: &[u8]) -> StoreResult<String> {
        let project_id = self
            .resolve_project_namespace(namespace_id, "reading symlinks")
            .await?;
        self.metadata
            .get_symlink(&project_id, symlink_id)
            .await?
            .ok_or_else(|| StoreError::ObjectNotFound {
                message: "symlink should have been present in cloud database".to_string(),
            })
    }

    pub async fn put_tree(
        &self,
        namespace_id: &str,
        tree_id: &[u8],
        entries: &[TreeEntry],
    ) -> StoreResult<()> {
        let project_id = self
            .resolve_project_namespace(namespace_id, "writing trees")
            .await?;

        let mut unique_file_ids: Vec<Vec<u8>> = Vec::new();
        let mut seen_files: HashSet<&[u8]> = HashSet::new();
        let mut unique_symlink_ids: Vec<Vec<u8>> = Vec::new();
        let mut seen_symlinks: HashSet<&[u8]> = HashSet::new();
        let mut unique_subtree_ids: Vec<Vec<u8>> = Vec::new();
        let mut seen_subtrees: HashSet<&[u8]> = HashSet::new();

        for entry in entries {
            if let Some(val) = entry.value.as_ref().and_then(|v| v.value.as_ref()) {
                match val {
                    tree_value::Value::File(f) => {
                        if seen_files.insert(f.id.as_slice()) {
                            unique_file_ids.push(f.id.clone());
                        }
                    }
                    tree_value::Value::SymlinkId(id) => {
                        if seen_symlinks.insert(id.as_slice()) {
                            unique_symlink_ids.push(id.clone());
                        }
                    }
                    tree_value::Value::TreeId(id) => {
                        if id.as_slice() != cc_common::EMPTY_TREE_ID_BYTES
                            && seen_subtrees.insert(id.as_slice())
                        {
                            unique_subtree_ids.push(id.clone());
                        }
                    }
                    tree_value::Value::ConflictId(_) | tree_value::Value::GitSubmoduleId(_) => {}
                }
            }
        }

        // Trees are immutable: keep the entries of an existing `tree_id` instead of rewriting
        // them. The same batched lookup checks subtree dependencies.
        let mut lookup_tree_ids = vec![tree_id.to_vec()];
        if self.config.validate_dependencies {
            lookup_tree_ids.extend(unique_subtree_ids.iter().cloned());
        }
        let existing_trees = self
            .metadata
            .find_existing_trees(&project_id, &lookup_tree_ids)
            .await?;
        if existing_trees.contains(tree_id) {
            return Ok(());
        }

        let blob_sizes = if unique_file_ids.is_empty() {
            HashMap::new()
        } else {
            self.metadata
                .get_blob_sizes_batch(&project_id, &unique_file_ids)
                .await?
        };

        if self.config.validate_dependencies {
            for fid in &unique_file_ids {
                if !blob_sizes.contains_key(fid.as_slice()) {
                    return Err(StoreError::MissingDependency {
                        parent_kind: "tree",
                        missing_kind: "file",
                        id_hex: hex::encode(fid),
                    });
                }
            }
            if !unique_symlink_ids.is_empty() {
                let found = self
                    .metadata
                    .find_existing_symlinks(&project_id, &unique_symlink_ids)
                    .await?;
                for sid in &unique_symlink_ids {
                    if !found.contains(sid.as_slice()) {
                        return Err(StoreError::MissingDependency {
                            parent_kind: "tree",
                            missing_kind: "symlink",
                            id_hex: hex::encode(sid),
                        });
                    }
                }
            }
            for tid in &unique_subtree_ids {
                if !existing_trees.contains(tid.as_slice()) {
                    return Err(StoreError::MissingDependency {
                        parent_kind: "tree",
                        missing_kind: "subtree",
                        id_hex: hex::encode(tid),
                    });
                }
            }
        }

        // Enrich File entries with uncompressed byte lengths and sort strictly in UTF-8
        // RepoPathComponent order (`"foo" < "foo.rs"`) so `jj_lib::Tree::from_sorted_entries`
        // binary searches never encounter Git's trailing-slash directory order.
        let mut normalized: Vec<TreeEntry> = entries
            .iter()
            .map(|e| {
                let mut cloned = e.clone();
                if let Some(tree_value::Value::File(f)) =
                    cloned.value.as_mut().and_then(|v| v.value.as_mut())
                    && let Some(&sz) = blob_sizes.get(f.id.as_slice())
                {
                    f.size_bytes = Some(sz);
                }
                cloned
            })
            .collect();
        normalized.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));

        self.metadata
            .put_tree(&project_id, tree_id, &normalized)
            .await
    }

    pub async fn get_tree(
        &self,
        namespace_id: &str,
        tree_id: &[u8],
    ) -> StoreResult<Vec<TreeEntry>> {
        let (entries, _) = self.get_tree_page(namespace_id, tree_id, "", 0).await?;
        Ok(entries)
    }

    pub async fn get_tree_page(
        &self,
        namespace_id: &str,
        tree_id: &[u8],
        page_token: &str,
        page_size: u32,
    ) -> StoreResult<(Vec<TreeEntry>, String)> {
        const MAX_TREE_PAGE_SIZE: usize = 20_000;

        let project_id = self
            .resolve_project_namespace(namespace_id, "requesting trees")
            .await?;
        if tree_id == cc_common::EMPTY_TREE_ID_BYTES {
            return Ok((Vec::new(), String::new()));
        }

        if page_size == 0 && page_token.is_empty() {
            let entries = self
                .metadata
                .get_tree(&project_id, tree_id)
                .await?
                .ok_or_else(|| StoreError::ObjectNotFound {
                    message: "tree should have been present in cloud database".to_string(),
                })?;
            return Ok((entries, String::new()));
        }

        let effective_limit = if page_size == 0 {
            MAX_TREE_PAGE_SIZE
        } else {
            (page_size as usize).min(MAX_TREE_PAGE_SIZE)
        };
        let after = (!page_token.is_empty()).then_some(page_token);
        let mut items = self
            .metadata
            .get_tree_page(&project_id, tree_id, after, effective_limit + 1)
            .await?
            .ok_or_else(|| StoreError::ObjectNotFound {
                message: "tree should have been present in cloud database".to_string(),
            })?;

        if items.len() > effective_limit {
            items.truncate(effective_limit);
            let next_token = items.last().map(|e| e.name.clone()).unwrap_or_default();
            Ok((items, next_token))
        } else {
            Ok((items, String::new()))
        }
    }

    pub async fn put_commit(
        &self,
        namespace_id: &str,
        commit_id: &[u8],
        commit: &Commit,
    ) -> StoreResult<()> {
        let project_id = self
            .resolve_project_namespace(namespace_id, "requesting commits")
            .await?;

        let mut seen_parents: HashSet<&[u8]> = HashSet::new();
        let parent_ids: Vec<Vec<u8>> = commit
            .parent_commit_ids
            .iter()
            .filter(|id| {
                id.as_slice() != cc_common::ROOT_COMMIT_ID_BYTES
                    && seen_parents.insert(id.as_slice())
            })
            .cloned()
            .collect();
        let parent_generations = if parent_ids.is_empty() {
            HashMap::new()
        } else {
            self.metadata
                .get_commit_generations(&project_id, &parent_ids)
                .await?
        };
        let mut max_parent_gen = 0u64;
        for parent_id in &parent_ids {
            match parent_generations.get(parent_id.as_slice()) {
                Some(&gen_num) => {
                    max_parent_gen = max_parent_gen.max(gen_num);
                }
                None if self.config.validate_dependencies => {
                    return Err(StoreError::MissingDependency {
                        parent_kind: "commit",
                        missing_kind: "parent commit",
                        id_hex: hex::encode(parent_id),
                    });
                }
                None => {}
            }
        }

        if self.config.validate_dependencies {
            let non_empty_trees: Vec<Vec<u8>> = commit
                .root_tree_id
                .iter()
                .filter(|t| t.as_slice() != cc_common::EMPTY_TREE_ID_BYTES)
                .cloned()
                .collect();
            if !non_empty_trees.is_empty() {
                let found = self
                    .metadata
                    .find_existing_trees(&project_id, &non_empty_trees)
                    .await?;
                for tree_id in &non_empty_trees {
                    if !found.contains(tree_id.as_slice()) {
                        return Err(StoreError::MissingDependency {
                            parent_kind: "commit",
                            missing_kind: "root tree",
                            id_hex: hex::encode(tree_id),
                        });
                    }
                }
            }
        }

        let generation = max_parent_gen.saturating_add(1);
        self.metadata
            .put_commit(&project_id, commit_id, commit, generation)
            .await
    }

    pub async fn get_commit(&self, namespace_id: &str, commit_id: &[u8]) -> StoreResult<Commit> {
        let project_id = self
            .resolve_project_namespace(namespace_id, "requesting commits")
            .await?;
        self.metadata
            .get_commit(&project_id, commit_id)
            .await?
            .ok_or_else(|| StoreError::ObjectNotFound {
                message: "commit should have been present in cloud database".to_string(),
            })
    }

    pub async fn list_project_commits(&self, namespace_id: &str) -> StoreResult<Vec<CommitId>> {
        let mut all_ids = Vec::new();
        let mut page_token = Vec::new();
        loop {
            let (batch, next_token, _) = self
                .list_commits_with_sync(namespace_id, &page_token, 20_000, &[])
                .await?;
            all_ids.extend(batch);
            if next_token.is_empty() {
                break;
            }
            page_token = next_token;
        }
        Ok(all_ids)
    }

    pub async fn list_commits(
        &self,
        namespace_id: &str,
        page_token: &[u8],
        page_size: u32,
    ) -> StoreResult<(Vec<CommitId>, Vec<u8>)> {
        let (items, next_token, _) = self
            .list_commits_with_sync(namespace_id, page_token, page_size, &[])
            .await?;
        Ok((items, next_token))
    }

    pub async fn list_commits_with_sync(
        &self,
        namespace_id: &str,
        page_token: &[u8],
        page_size: u32,
        since_token: &[u8],
    ) -> StoreResult<(Vec<CommitId>, Vec<u8>, Vec<u8>)> {
        const DEFAULT_PAGE_SIZE: usize = 5_000;
        const MAX_PAGE_SIZE: usize = 20_000;

        let project_id = self
            .resolve_project_namespace(namespace_id, "listing project commits")
            .await?;
        let effective_size = if page_size == 0 {
            DEFAULT_PAGE_SIZE
        } else {
            (page_size as usize).min(MAX_PAGE_SIZE)
        };

        if !since_token.is_empty() {
            let cursor = if page_token.is_empty() {
                since_token
            } else {
                page_token
            };
            let (items, new_sync, has_more) = self
                .metadata
                .list_commit_ids_since(&project_id, cursor, effective_size)
                .await?;
            let next_page = if has_more {
                new_sync.clone()
            } else {
                Vec::new()
            };
            return Ok((items, next_page, new_sync));
        }

        // Capture watermark before page 1 so any concurrent commit inserted during a full
        // scan is guaranteed to have watermark > captured_watermark. Page tokens carry it
        // (`len as u32 BE || watermark || last commit id`) so the last page returns it too: a
        // watermark read at the end of the scan would skip commits inserted mid-scan whose ids
        // sort before the cursor.
        let (watermark, after) = if page_token.is_empty() {
            (
                self.metadata.current_commit_watermark(&project_id).await?,
                None,
            )
        } else {
            let malformed =
                || StoreError::InvalidArgument("malformed ListCommits page_token".to_string());
            let (len, rest) = page_token.split_first_chunk::<4>().ok_or_else(malformed)?;
            let len = u32::from_be_bytes(*len) as usize;
            if rest.len() <= len {
                return Err(malformed());
            }
            let (watermark, after) = rest.split_at(len);
            (watermark.to_vec(), Some(after))
        };

        let mut items = self
            .metadata
            .list_commit_ids(&project_id, after, effective_size + 1)
            .await?;

        if items.len() > effective_size {
            items.truncate(effective_size);
            let last = items.last().map(Vec::as_slice).unwrap_or_default();
            let mut next_token = Vec::with_capacity(4 + watermark.len() + last.len());
            next_token.extend_from_slice(&(watermark.len() as u32).to_be_bytes());
            next_token.extend_from_slice(&watermark);
            next_token.extend_from_slice(last);
            Ok((items, next_token, watermark))
        } else {
            Ok((items, Vec::new(), watermark))
        }
    }

    pub async fn resolve_commit_prefix(
        &self,
        namespace_id: &str,
        hex_prefix: &str,
    ) -> StoreResult<IdPrefixResolution> {
        let project_id = self
            .resolve_project_namespace(namespace_id, "resolving commit prefixes")
            .await?;

        let normalized = hex_prefix.to_ascii_lowercase();
        let Some((min_id, max_id)) =
            hex_prefix_byte_bounds(&normalized, cc_common::COMMIT_ID_LENGTH)?
        else {
            return Ok(IdPrefixResolution::NoMatch);
        };

        let mut matches = self
            .metadata
            .resolve_commit_id_prefix(&project_id, &min_id, &max_id)
            .await?;

        let root_id = cc_common::ROOT_COMMIT_ID_BYTES.to_vec();
        if root_id.as_slice() >= min_id.as_slice()
            && root_id.as_slice() <= max_id.as_slice()
            && !matches.contains(&root_id)
        {
            matches.push(root_id);
        }

        match matches.len() {
            0 => Ok(IdPrefixResolution::NoMatch),
            1 => Ok(IdPrefixResolution::SingleMatch(matches.swap_remove(0))),
            _ => Ok(IdPrefixResolution::AmbiguousMatch),
        }
    }

    pub async fn resolve_change_prefix(
        &self,
        namespace_id: &str,
        prefix: &str,
    ) -> StoreResult<ChangePrefixResolution> {
        let project_id = self
            .resolve_project_namespace(namespace_id, "resolving change prefixes")
            .await?;

        let hex_prefix = normalize_change_id_prefix_to_hex(prefix)?;
        let Some((min_id, max_id)) =
            hex_prefix_byte_bounds(&hex_prefix, cc_common::CHANGE_ID_LENGTH)?
        else {
            return Ok(ChangePrefixResolution::NoMatch);
        };

        let mut matches = self
            .metadata
            .resolve_change_id_prefix(&project_id, &min_id, &max_id)
            .await?;

        let root_change = cc_common::ROOT_CHANGE_ID_BYTES.to_vec();
        if root_change.as_slice() >= min_id.as_slice()
            && root_change.as_slice() <= max_id.as_slice()
            && !matches.iter().any(|(ch, _)| ch == &root_change)
        {
            matches.push((root_change, vec![cc_common::ROOT_COMMIT_ID_BYTES.to_vec()]));
        }

        match matches.len() {
            0 => Ok(ChangePrefixResolution::NoMatch),
            1 => {
                let (change_id, commit_ids) = matches.swap_remove(0);
                Ok(ChangePrefixResolution::SingleMatch {
                    change_id,
                    commit_ids,
                })
            }
            _ => Ok(ChangePrefixResolution::AmbiguousMatch),
        }
    }

    pub async fn put_view(&self, repo_id: &str, view_id: &[u8], view: &View) -> StoreResult<()> {
        self.ensure_repo_registered(repo_id, "writing views")
            .await?;
        self.metadata.put_view(repo_id, view_id, view).await
    }

    pub async fn get_view(&self, repo_id: &str, view_id: &[u8]) -> StoreResult<Option<View>> {
        self.ensure_repo_registered(repo_id, "reading views")
            .await?;
        self.metadata.get_view(repo_id, view_id).await
    }

    pub async fn put_operation(
        &self,
        repo_id: &str,
        op_id: &[u8],
        op: &Operation,
    ) -> StoreResult<()> {
        self.ensure_repo_registered(repo_id, "writing operations")
            .await?;
        if self.config.validate_dependencies
            && op.view_id.as_slice() != cc_common::ROOT_VIEW_ID_BYTES
            && self
                .metadata
                .get_view(repo_id, &op.view_id)
                .await?
                .is_none()
        {
            return Err(StoreError::MissingDependency {
                parent_kind: "operation",
                missing_kind: "view",
                id_hex: hex::encode(&op.view_id),
            });
        }
        self.metadata.put_operation(repo_id, op_id, op).await
    }

    pub async fn get_operation(
        &self,
        repo_id: &str,
        op_id: &[u8],
    ) -> StoreResult<Option<Operation>> {
        self.ensure_repo_registered(repo_id, "reading operations")
            .await?;
        self.metadata.get_operation(repo_id, op_id).await
    }

    pub async fn resolve_operation_prefix(
        &self,
        repo_id: &str,
        hex_prefix: &str,
    ) -> StoreResult<IdPrefixResolution> {
        self.ensure_repo_registered(repo_id, "resolving operation prefixes")
            .await?;

        let normalized = hex_prefix.to_ascii_lowercase();
        let Some((min_id, max_id)) =
            hex_prefix_byte_bounds(&normalized, cc_common::OPERATION_ID_LENGTH)?
        else {
            return Ok(IdPrefixResolution::NoMatch);
        };

        let mut matches = self
            .metadata
            .resolve_operation_id_prefix(repo_id, &min_id, &max_id)
            .await?;

        let root_id = cc_common::ROOT_OPERATION_ID_BYTES.to_vec();
        if root_id.as_slice() >= min_id.as_slice()
            && root_id.as_slice() <= max_id.as_slice()
            && !matches.contains(&root_id)
        {
            matches.push(root_id);
        }

        match matches.len() {
            0 => Ok(IdPrefixResolution::NoMatch),
            1 => Ok(IdPrefixResolution::SingleMatch(matches.swap_remove(0))),
            _ => Ok(IdPrefixResolution::AmbiguousMatch),
        }
    }

    pub async fn get_op_heads(&self, repo_id: &str) -> StoreResult<Vec<OpId>> {
        self.ensure_repo_registered(repo_id, "requesting op heads")
            .await?;
        Ok(self
            .metadata
            .get_op_heads(repo_id)
            .await?
            .unwrap_or_else(|| vec![cc_common::ROOT_OPERATION_ID_BYTES.to_vec()]))
    }

    pub async fn update_op_heads(
        &self,
        repo_id: &str,
        old_ids: &[Vec<u8>],
        new_id: &[u8],
    ) -> StoreResult<Vec<OpId>> {
        self.ensure_repo_registered(repo_id, "updating op heads")
            .await?;
        if self.config.validate_dependencies
            && new_id != cc_common::ROOT_OPERATION_ID_BYTES
            && self
                .metadata
                .get_operation(repo_id, new_id)
                .await?
                .is_none()
        {
            return Err(StoreError::MissingDependency {
                parent_kind: "op_heads",
                missing_kind: "operation",
                id_hex: hex::encode(new_id),
            });
        }
        self.metadata
            .update_op_heads(repo_id, old_ids, new_id)
            .await
    }

    pub async fn publish_operation(
        &self,
        repo_id: &str,
        view_id: &[u8],
        view: &View,
        op_id: &[u8],
        op: &Operation,
        old_ids: &[Vec<u8>],
    ) -> StoreResult<Vec<OpId>> {
        self.ensure_repo_registered(repo_id, "publishing operation")
            .await?;
        self.metadata
            .publish_operation_atomic(repo_id, view_id, view, op_id, op, old_ids)
            .await
    }

    pub async fn compare_and_swap_op_heads(
        &self,
        repo_id: &str,
        expected_old_ids: &[Vec<u8>],
        new_id: &[u8],
    ) -> StoreResult<()> {
        self.ensure_repo_registered(repo_id, "comparing and swapping op heads")
            .await?;
        if self.config.validate_dependencies
            && new_id != cc_common::ROOT_OPERATION_ID_BYTES
            && self
                .metadata
                .get_operation(repo_id, new_id)
                .await?
                .is_none()
        {
            return Err(StoreError::MissingDependency {
                parent_kind: "op_heads",
                missing_kind: "operation",
                id_hex: hex::encode(new_id),
            });
        }
        self.metadata
            .compare_and_swap_op_heads(repo_id, expected_old_ids, new_id)
            .await
    }

    pub async fn get_workspace(
        &self,
        repo_id: &str,
        user: &str,
        workspace_name: &str,
    ) -> StoreResult<Option<WorkspaceState>> {
        self.ensure_repo_registered(repo_id, "reading workspaces")
            .await?;
        self.metadata
            .get_workspace(repo_id, user, workspace_name)
            .await
    }

    pub async fn put_workspace(&self, workspace: &WorkspaceState) -> StoreResult<()> {
        if workspace.repo_id.trim().is_empty() || workspace.workspace_name.trim().is_empty() {
            return Err(StoreError::InvalidArgument(
                "workspace repo_id and workspace_name must be non-empty".to_string(),
            ));
        }
        self.ensure_repo_registered(&workspace.repo_id, "updating workspaces")
            .await?;
        let mut normalized = workspace.clone();
        if normalized.tree_ids.is_empty() && !normalized.tree_id.is_empty() {
            normalized.tree_ids = vec![normalized.tree_id.clone()];
        } else if normalized.tree_id.is_empty()
            && let Some(first) = normalized.tree_ids.first()
        {
            normalized.tree_id = first.clone();
        }
        self.metadata.put_workspace(&normalized).await
    }

    pub async fn list_workspaces(&self, repo_id: &str) -> StoreResult<Vec<WorkspaceState>> {
        self.ensure_repo_registered(repo_id, "listing workspaces")
            .await?;
        self.metadata.list_workspaces(repo_id).await
    }

    pub async fn delete_workspace(
        &self,
        repo_id: &str,
        user: &str,
        workspace_name: &str,
    ) -> StoreResult<bool> {
        self.ensure_repo_registered(repo_id, "deleting workspaces")
            .await?;
        self.metadata
            .delete_workspace(repo_id, user, workspace_name)
            .await
    }
}

pub fn op_heads_sets_equal(a: &[Vec<u8>], b: &[Vec<u8>]) -> bool {
    let set_a: std::collections::HashSet<&[u8]> = a.iter().map(Vec::as_slice).collect();
    let set_b: std::collections::HashSet<&[u8]> = b.iter().map(Vec::as_slice).collect();
    set_a == set_b
}

/// Applies the canonical jj Commit Cloud operation-head transition rules to `current_heads`:
/// - Removes all entries listed in `old_ids`.
/// - Preserves any concurrent sibling heads not listed in `old_ids`.
/// - Appends `new_id` so concurrent writers diverge cleanly instead of overwriting each other.
pub fn apply_op_heads_mutation(current_heads: &mut Vec<OpId>, old_ids: &[Vec<u8>], new_id: &[u8]) {
    current_heads.retain(|head| !old_ids.iter().any(|old| old.as_slice() == head.as_slice()));
    if !current_heads.iter().any(|head| head.as_slice() == new_id) {
        current_heads.push(new_id.to_vec());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_prefix_byte_bounds_even_and_odd_nibbles() {
        let (min_even, max_even) = hex_prefix_byte_bounds("4b82", 4).unwrap().unwrap();
        assert_eq!(min_even, vec![0x4b, 0x82, 0x00, 0x00]);
        assert_eq!(max_even, vec![0x4b, 0x82, 0xff, 0xff]);

        let (min_odd, max_odd) = hex_prefix_byte_bounds("4b8", 4).unwrap().unwrap();
        assert_eq!(min_odd, vec![0x4b, 0x80, 0x00, 0x00]);
        assert_eq!(max_odd, vec![0x4b, 0x8f, 0xff, 0xff]);
    }

    #[test]
    fn test_hex_prefix_byte_bounds_rejects_invalid_input() {
        assert!(hex_prefix_byte_bounds("zz", 4).is_err());
        assert!(hex_prefix_byte_bounds("01020304zz", 4).is_err());
        assert_eq!(hex_prefix_byte_bounds("0102030405", 4).unwrap(), None);
    }

    #[test]
    fn test_normalize_change_id_prefix_to_hex() {
        assert_eq!(normalize_change_id_prefix_to_hex("zyxk").unwrap(), "012f");
        assert_eq!(normalize_change_id_prefix_to_hex("01ab").unwrap(), "01ab");
        assert!(normalize_change_id_prefix_to_hex("01zk").is_err());
    }

    #[tokio::test]
    async fn test_composite_store_enforces_dependency_validation_size_enrichment_and_sort_order() {
        let store = CompositeStore::in_memory();
        store.register_repo("repo-a", Some("alpha")).await.unwrap();
        store.register_repo("repo-b", Some("beta")).await.unwrap();

        let blob_id = vec![0x11; 20];
        let sub_tree_id = vec![0x33; 20];
        // Writing a tree referencing an unuploaded file fails with MissingDependency.
        let tree_entries = vec![TreeEntry {
            name: "readme.md".to_string(),
            value: Some(TreeValue {
                value: Some(tree_value::Value::File(File {
                    id: blob_id.clone(),
                    executable: false,
                    copy_id: vec![],
                    size_bytes: None,
                })),
            }),
        }];
        let err = store
            .put_tree("repo-a", &[0x22; 20], &tree_entries)
            .await
            .unwrap_err();
        assert!(matches!(err, StoreError::MissingDependency { .. }));

        // Upload the blob and a subtree into repo-a; write entries in Git trailing-slash order
        // (`"foo.rs"` before directory `"foo"`) and verify ReadTree returns UTF-8 byte order
        // (`"foo"` before `"foo.rs"`) with `size_bytes = Some(11)` populated!
        store
            .put_file("repo-a", &blob_id, Bytes::from_static(b"hello world"))
            .await
            .unwrap();
        store.put_tree("repo-a", &sub_tree_id, &[]).await.unwrap();

        let git_ordered = vec![
            TreeEntry {
                name: "foo.rs".to_string(),
                value: Some(TreeValue {
                    value: Some(tree_value::Value::File(File {
                        id: blob_id.clone(),
                        executable: false,
                        copy_id: vec![],
                        size_bytes: None,
                    })),
                }),
            },
            TreeEntry {
                name: "foo".to_string(),
                value: Some(TreeValue {
                    value: Some(tree_value::Value::TreeId(sub_tree_id.clone())),
                }),
            },
        ];
        store
            .put_tree("repo-a", &[0x22; 20], &git_ordered)
            .await
            .unwrap();

        let fetched = store.get_tree("repo-a", &[0x22; 20]).await.unwrap();
        assert_eq!(fetched.len(), 2);
        assert_eq!(fetched[0].name, "foo");
        assert_eq!(fetched[1].name, "foo.rs");
        match fetched[1].value.as_ref().unwrap().value.as_ref().unwrap() {
            tree_value::Value::File(f) => assert_eq!(f.size_bytes, Some(11)),
            other => panic!("expected File, got {other:?}"),
        }

        // Verify cursor-paginated ReadTree.
        let (page1, tok1) = store
            .get_tree_page("repo-a", &[0x22; 20], "", 1)
            .await
            .unwrap();
        assert_eq!(page1.len(), 1);
        assert_eq!(page1[0].name, "foo");
        assert_eq!(tok1, "foo");
        let (page2, tok2) = store
            .get_tree_page("repo-a", &[0x22; 20], &tok1, 1)
            .await
            .unwrap();
        assert_eq!(page2.len(), 1);
        assert_eq!(page2[0].name, "foo.rs");
        assert!(tok2.is_empty());

        // repo-b cannot probe repo-a's blob until repo-b links it.
        assert!(matches!(
            store.get_file_range("repo-b", &blob_id, 0, None).await,
            Err(StoreError::ObjectNotFound { .. })
        ));
    }

    #[tokio::test]
    async fn test_concurrent_op_heads_preserve_siblings_even_after_race() {
        let store = CompositeStore::in_memory();
        store.register_repo("repo-1", None).await.unwrap();

        let root_op = cc_common::ROOT_OPERATION_ID_BYTES.to_vec();
        let view_id = vec![0xaa; 16];
        store
            .put_view(
                "repo-1",
                &view_id,
                &View {
                    head_ids: vec![cc_common::ROOT_COMMIT_ID_BYTES.to_vec()],
                    wc_commit_ids: Default::default(),
                    local_bookmarks: Default::default(),
                    remote_bookmarks: Default::default(),
                },
            )
            .await
            .unwrap();

        let make_op = || Operation {
            view_id: view_id.clone(),
            parents: vec![root_op.clone()],
            metadata: None,
            commit_predecessors: vec![],
            commit_predecessors_set: false,
        };
        let op_a = vec![0x10; 16];
        let op_b = vec![0x20; 16];
        store
            .put_operation("repo-1", &op_a, &make_op())
            .await
            .unwrap();
        store
            .put_operation("repo-1", &op_b, &make_op())
            .await
            .unwrap();

        // Agent A advances from [root] -> [op_a].
        let heads_after_a = store
            .update_op_heads("repo-1", std::slice::from_ref(&root_op), &op_a)
            .await
            .unwrap();
        assert_eq!(heads_after_a, vec![op_a.clone()]);

        // Agent B concurrently started from [root] and submits [op_b]; both [op_a, op_b] survive!
        let heads_after_b = store
            .update_op_heads("repo-1", std::slice::from_ref(&root_op), &op_b)
            .await
            .unwrap();
        assert_eq!(heads_after_b, vec![op_a.clone(), op_b.clone()]);

        // Reconciling [op_a, op_b] -> [op_c] collapses heads back to [op_c].
        let op_c = vec![0x30; 16];
        store
            .put_operation("repo-1", &op_c, &make_op())
            .await
            .unwrap();
        let heads_after_c = store
            .update_op_heads("repo-1", &[op_a, op_b], &op_c)
            .await
            .unwrap();
        assert_eq!(heads_after_c, vec![op_c]);
    }

    #[tokio::test]
    async fn test_sqlite_and_local_fs_blob_store_range_reads_and_durability() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("meta.db");
        let blob_dir = dir.path().join("blobs_dir");

        let meta = Arc::new(SqliteMetadataStore::open(&db_path).unwrap());
        let blobs = Arc::new(LocalFsBlobStore::open(&blob_dir).unwrap());
        let store = CompositeStore::new(meta, blobs);

        store
            .register_repo("repo-sql", Some("sql-alias"))
            .await
            .unwrap();
        let blob_id = vec![0x42; 20];
        let payload = Bytes::from_static(b"0123456789abcdef");
        store.put_file("repo-sql", &blob_id, payload).await.unwrap();

        let slice = store
            .get_file_range("repo-sql", &blob_id, 4, Some(6))
            .await
            .unwrap();
        assert_eq!(slice.as_ref(), b"456789");

        assert!(matches!(
            store
                .get_file_range("repo-sql", &blob_id, 16, Some(1))
                .await,
            Err(StoreError::OutOfRange(_))
        ));
    }

    #[tokio::test]
    async fn test_put_tree_is_insert_only_per_project() {
        let store = CompositeStore::in_memory();
        store.register_repo("repo-p1", None).await.unwrap();
        store.register_repo("repo-p2", None).await.unwrap();

        let blob_id = vec![0x11; 20];
        let tree_id = vec![0x22; 20];
        store
            .put_file("repo-p1", &blob_id, Bytes::from_static(b"immutable"))
            .await
            .unwrap();

        let original_entry = TreeEntry {
            name: "a.txt".to_string(),
            value: Some(TreeValue {
                value: Some(tree_value::Value::File(File {
                    id: blob_id.clone(),
                    executable: false,
                    copy_id: vec![],
                    size_bytes: None,
                })),
            }),
        };
        store
            .put_tree("repo-p1", &tree_id, std::slice::from_ref(&original_entry))
            .await
            .unwrap();

        // Attempting to rewrite the same `tree_id` in the same project leaves the stored
        // entries unchanged.
        let injected_entry = TreeEntry {
            name: "injected_submodule".to_string(),
            value: Some(TreeValue {
                value: Some(tree_value::Value::GitSubmoduleId(vec![0x99; 20])),
            }),
        };
        store
            .put_tree(
                "repo-p1",
                &tree_id,
                &[original_entry.clone(), injected_entry],
            )
            .await
            .unwrap();

        let stored = store.get_tree("repo-p1", &tree_id).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].name, "a.txt");

        // A separate project does not inherit `tree_id` and still enforces dependency checks.
        let err = store
            .put_tree("repo-p2", &tree_id, std::slice::from_ref(&original_entry))
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            StoreError::MissingDependency {
                missing_kind: "file",
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_put_commit_batches_parent_generations_and_overlong_prefixes_return_no_match() {
        let store = CompositeStore::in_memory();
        store.register_repo("repo-gen", None).await.unwrap();

        let make_commit = |commit_id: Vec<u8>, change_id: Vec<u8>, parents: Vec<Vec<u8>>| Commit {
            commit_id,
            change_id,
            parent_commit_ids: parents,
            root_tree_id: vec![cc_common::EMPTY_TREE_ID_BYTES.to_vec()],
            description: "commit\n".to_string(),
            author: None,
            committer: None,
            predecessors: vec![],
            conflict_labels: vec![],
            secure_sig: None,
            raw_git_commit: vec![],
            external_ids: Default::default(),
        };

        // Build a chain of depth 4 (generations 1..=4) plus 18 sibling root-parented commits.
        let mut chain_prev = cc_common::ROOT_COMMIT_ID_BYTES.to_vec();
        for depth in 1..=4u8 {
            let cid = vec![depth; 20];
            let commit = make_commit(cid.clone(), vec![depth; 16], vec![chain_prev]);
            store.put_commit("repo-gen", &cid, &commit).await.unwrap();
            chain_prev = cid;
        }
        let mut merge_parents = vec![chain_prev];
        for idx in 10..28u8 {
            let cid = vec![idx; 20];
            let commit = make_commit(
                cid.clone(),
                vec![idx; 16],
                vec![cc_common::ROOT_COMMIT_ID_BYTES.to_vec()],
            );
            store.put_commit("repo-gen", &cid, &commit).await.unwrap();
            merge_parents.push(cid);
        }

        let merge_id = vec![0xfe; 20];
        let merge_commit = make_commit(merge_id.clone(), vec![0xfe; 16], merge_parents.clone());
        store
            .put_commit("repo-gen", &merge_id, &merge_commit)
            .await
            .unwrap();
        assert_eq!(
            store
                .metadata
                .get_commit_generation("repo-gen", &merge_id)
                .await
                .unwrap(),
            Some(5)
        );

        let mut missing_parents = merge_parents;
        missing_parents.push(vec![0xee; 20]);
        let bad_commit = make_commit(vec![0xfd; 20], vec![0xfd; 16], missing_parents);
        let err = store
            .put_commit("repo-gen", &[0xfd; 20], &bad_commit)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            StoreError::MissingDependency {
                missing_kind: "parent commit",
                ..
            }
        ));

        // Over-long valid prefixes resolve to NoMatch rather than InvalidArgument.
        assert_eq!(
            store
                .resolve_commit_prefix("repo-gen", &"fe".repeat(32))
                .await
                .unwrap(),
            IdPrefixResolution::NoMatch
        );
        assert_eq!(
            store
                .resolve_change_prefix("repo-gen", &"k".repeat(40))
                .await
                .unwrap(),
            ChangePrefixResolution::NoMatch
        );
        assert_eq!(
            store
                .resolve_operation_prefix("repo-gen", &"00".repeat(64))
                .await
                .unwrap(),
            IdPrefixResolution::NoMatch
        );
    }

    #[tokio::test]
    async fn test_list_commits_full_scan_carries_page1_watermark_across_pages() {
        let dir = tempfile::tempdir().unwrap();
        let sql_meta = Arc::new(SqliteMetadataStore::open(&dir.path().join("scan.db")).unwrap());
        let sql_blobs = Arc::new(LocalFsBlobStore::open(dir.path().join("blobs")).unwrap());
        let stores = [
            CompositeStore::in_memory(),
            CompositeStore::new(sql_meta, sql_blobs),
        ];

        for store in stores {
            store.register_repo("scan-repo", None).await.unwrap();

            let make_commit = |byte: u8| Commit {
                commit_id: vec![byte; 20],
                change_id: vec![byte; 16],
                parent_commit_ids: vec![cc_common::ROOT_COMMIT_ID_BYTES.to_vec()],
                root_tree_id: vec![cc_common::EMPTY_TREE_ID_BYTES.to_vec()],
                description: format!("commit {byte:#04x}\n"),
                author: None,
                committer: None,
                predecessors: vec![],
                conflict_labels: vec![],
                secure_sig: None,
                raw_git_commit: vec![],
                external_ids: Default::default(),
            };

            // Seed 3 commits whose IDs sort above 0x05 (`0x10`, `0x20`, `0x30`).
            for b in [0x10u8, 0x20, 0x30] {
                let c = make_commit(b);
                store
                    .put_commit("scan-repo", &c.commit_id, &c)
                    .await
                    .unwrap();
            }

            // Read page 1 of a full scan (`page_size = 2`): returns `[0x10, 0x20]` and W1.
            let (p1, tok1, sync1) = store
                .list_commits_with_sync("scan-repo", &[], 2, &[])
                .await
                .unwrap();
            assert_eq!(p1, vec![vec![0x10; 20], vec![0x20; 20]]);
            assert!(!tok1.is_empty());
            assert!(!sync1.is_empty());

            // Mid-scan concurrent writer inserts commit `0x05`, which sorts BEFORE page 1's
            // cursor (`0x20`) and therefore will not appear in page 2 of the key-ordered scan.
            let low_commit = make_commit(0x05);
            store
                .put_commit("scan-repo", &low_commit.commit_id, &low_commit)
                .await
                .unwrap();

            // Final page returns `[0x30]` and MUST preserve `sync1` rather than recomputing an
            // end-of-scan watermark past `0x05`.
            let (p2, tok2, sync2) = store
                .list_commits_with_sync("scan-repo", &tok1, 2, &[])
                .await
                .unwrap();
            assert_eq!(p2, vec![vec![0x30; 20]]);
            assert!(tok2.is_empty());
            assert_eq!(sync2, sync1);

            // Subsequent incremental sync from `sync2` discovers `0x05` instead of losing it.
            let (delta, delta_next, _) = store
                .list_commits_with_sync("scan-repo", &[], 10, &sync2)
                .await
                .unwrap();
            assert_eq!(delta, vec![vec![0x05; 20]]);
            assert!(delta_next.is_empty());
        }
    }
}
