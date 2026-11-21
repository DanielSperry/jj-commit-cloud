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

//! Server-side `jj_lib` storage adapters (`ServerBackend`, `ServerOpStore`,
//! `ServerOpHeadsStore`, and `ServerIndexStore`) backed by [`CompositeStore`].
//!
//! # Authority and Crash-Safety Ordering
//! When `jj_lib` resolves divergent operation heads on jj-cc-server, any synthesized
//! merged trees, commits, views, and operations are persisted through [`CompositeStore`]
//! (preserving write-before-publish ordering and bottom-up dependency validation) before
//! [`ServerOpHeadsStore::update_op_heads`] advances the operation head set.
//!
//! # Concurrency and Merge Invariants
//! [`ServerOpHeadsStore::update_op_heads`] delegates to additive
//! [`CompositeStore::update_op_heads`], removing merged or ancestor `old_ids` and adding
//! `new_id` while retaining concurrent sibling heads. [`ServerIndex`] prunes
//! `is_ancestor`, `heads`, and `common_ancestors` walks using commit generation numbers,
//! caches immutable `(project_id, commit_id)` graph nodes across reconciles, and surfaces
//! missing parent commits as corruption errors. [`ServerOpStore`] caches immutable
//! operations across reconciles.
//!
//! # Cross-Process and RPC Boundaries
//! Executes in-process inside jj-cc-server with zero loopback gRPC serialization.

#![allow(elided_lifetimes_in_paths)]

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, Mutex, OnceLock, Weak};
use std::time::SystemTime;

use async_trait::async_trait;
use bytes::Bytes;
use cc_lib::cc_backend::{
    apply_commit_signing, commit_from_proto, commit_to_proto, tree_entry_from_proto,
    tree_entry_to_proto,
};
use cc_lib::cc_op_store::{
    operation_from_proto, operation_to_proto, view_from_proto, view_to_proto,
};
use futures::StreamExt as _;
use futures::stream::{self, BoxStream, LocalBoxStream};
use jj_lib::backend::*;
use jj_lib::commit::Commit as JjCommit;
use jj_lib::graph::GraphNode;
use jj_lib::index::{
    ChangeIdIndex, Index, IndexError, IndexResult, IndexStore, IndexStoreResult, MutableIndex,
    ReadonlyIndex, ResolvedChangeTargets,
};
use jj_lib::object_id::{HexPrefix, ObjectId, PrefixResolution};
use jj_lib::op_heads_store::{OpHeadsStore, OpHeadsStoreError, OpHeadsStoreLock};
use jj_lib::op_store::{
    OpStore, OpStoreError, OpStoreResult, Operation, OperationId, View, ViewId,
};
use jj_lib::operation::Operation as JjOperation;
use jj_lib::repo_path::{RepoPath, RepoPathBuf};
use jj_lib::revset::{ResolvedExpression, Revset, RevsetContainingFn, RevsetEvaluationError};
use jj_lib::store::Store as JjStore;
use jj_lib::submodule_store::SubmoduleStore;

use crate::hash_utils::{
    compute_git_blob_hash, compute_git_commit_hash, compute_git_tree_hash, hash_operation,
    hash_view,
};
use crate::store::{CompositeStore, IdPrefixResolution, MetadataStore, StoreError};

const COMMIT_GRAPH_CACHE_CAPACITY: usize = 65_536;
const OPERATION_CACHE_CAPACITY: usize = 16_384;

struct BoundedCache<K, V> {
    map: HashMap<K, V>,
    order: VecDeque<K>,
    capacity: usize,
}

impl<K: Eq + Hash + Clone, V> BoundedCache<K, V> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    fn insert(&mut self, key: K, value: V) {
        if let std::collections::hash_map::Entry::Occupied(mut occupied) =
            self.map.entry(key.clone())
        {
            occupied.insert(value);
            return;
        }
        if self.map.len() >= self.capacity
            && let Some(oldest) = self.order.pop_front()
        {
            self.map.remove(&oldest);
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }
}

#[derive(Clone)]
struct StoreScopedEntry<V> {
    metadata_ref: Weak<dyn MetadataStore>,
    value: V,
}

fn store_cache_key(store: &CompositeStore) -> usize {
    Arc::as_ptr(store.metadata()).cast::<()>() as usize
}

fn store_entry_matches<V>(entry: &StoreScopedEntry<V>, store: &CompositeStore) -> bool {
    entry
        .metadata_ref
        .upgrade()
        .is_some_and(|arc| std::ptr::addr_eq(Arc::as_ptr(&arc), Arc::as_ptr(store.metadata())))
}

static OPERATION_CACHE: LazyLock<
    Mutex<BoundedCache<(usize, String, OperationId), StoreScopedEntry<Operation>>>,
> = LazyLock::new(|| Mutex::new(BoundedCache::with_capacity(OPERATION_CACHE_CAPACITY)));

static COMMIT_GRAPH_CACHE: LazyLock<
    Mutex<BoundedCache<(usize, String, CommitId), StoreScopedEntry<(u64, Vec<CommitId>)>>>,
> = LazyLock::new(|| Mutex::new(BoundedCache::with_capacity(COMMIT_GRAPH_CACHE_CAPACITY)));

pub struct ServerBackend {
    store: Arc<CompositeStore>,
    repo_id: String,
    root_commit_id: CommitId,
    root_change_id: ChangeId,
    empty_tree_id: TreeId,
}

impl Debug for ServerBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServerBackend")
            .field("repo_id", &self.repo_id)
            .finish()
    }
}

impl ServerBackend {
    pub fn new(store: Arc<CompositeStore>, repo_id: String) -> Self {
        Self {
            store,
            repo_id,
            root_commit_id: CommitId::from_bytes(&cc_common::ROOT_COMMIT_ID_BYTES),
            root_change_id: ChangeId::from_bytes(&cc_common::ROOT_CHANGE_ID_BYTES),
            empty_tree_id: TreeId::from_hex(cc_common::EMPTY_TREE_ID_HEX),
        }
    }
}

#[async_trait]
impl Backend for ServerBackend {
    fn name(&self) -> &str {
        "server_backend"
    }

    fn commit_id_length(&self) -> usize {
        cc_common::COMMIT_ID_LENGTH
    }

    fn change_id_length(&self) -> usize {
        cc_common::CHANGE_ID_LENGTH
    }

    fn root_commit_id(&self) -> &CommitId {
        &self.root_commit_id
    }

    fn root_change_id(&self) -> &ChangeId {
        &self.root_change_id
    }

    fn empty_tree_id(&self) -> &TreeId {
        &self.empty_tree_id
    }

    fn concurrency(&self) -> usize {
        64
    }

    async fn read_file(
        &self,
        _path: &RepoPath,
        id: &FileId,
    ) -> BackendResult<Pin<Box<dyn futures::AsyncRead + Send>>> {
        let content = self
            .store
            .get_file_range(&self.repo_id, id.as_bytes(), 0, None)
            .await
            .map_err(|e| match e {
                StoreError::ObjectNotFound { .. } => BackendError::ObjectNotFound {
                    object_type: "file".into(),
                    hash: id.hex(),
                    source: e.into(),
                },
                other => BackendError::Other(other.into()),
            })?;
        Ok(Box::pin(futures::io::Cursor::new(content.to_vec())))
    }

    async fn write_file(
        &self,
        _path: &RepoPath,
        contents: &mut (dyn futures::AsyncRead + Send + Unpin),
    ) -> BackendResult<FileId> {
        let mut buffer = Vec::new();
        futures::AsyncReadExt::read_to_end(contents, &mut buffer)
            .await
            .map_err(|e| BackendError::Other(e.into()))?;
        let file_id_bytes = compute_git_blob_hash(&buffer);
        self.store
            .put_file(&self.repo_id, &file_id_bytes, Bytes::from(buffer))
            .await
            .map_err(|e| BackendError::Other(e.into()))?;
        Ok(FileId::from_bytes(&file_id_bytes))
    }

    async fn read_symlink(&self, _path: &RepoPath, id: &SymlinkId) -> BackendResult<String> {
        self.store
            .get_symlink(&self.repo_id, id.as_bytes())
            .await
            .map_err(|e| match e {
                StoreError::ObjectNotFound { .. } => BackendError::ObjectNotFound {
                    object_type: "symlink".into(),
                    hash: id.hex(),
                    source: e.into(),
                },
                other => BackendError::Other(other.into()),
            })
    }

    async fn write_symlink(&self, _path: &RepoPath, target: &str) -> BackendResult<SymlinkId> {
        let symlink_id_bytes = compute_git_blob_hash(target.as_bytes());
        self.store
            .put_symlink(&self.repo_id, &symlink_id_bytes, target)
            .await
            .map_err(|e| BackendError::Other(e.into()))?;
        Ok(SymlinkId::from_bytes(&symlink_id_bytes))
    }

    async fn read_copy(&self, _id: &CopyId) -> BackendResult<CopyHistory> {
        Err(BackendError::Unsupported("copies not supported".into()))
    }

    async fn write_copy(&self, _contents: &CopyHistory) -> BackendResult<CopyId> {
        Err(BackendError::Unsupported("copies not supported".into()))
    }

    async fn get_related_copies(&self, _copy_id: &CopyId) -> BackendResult<Vec<RelatedCopy>> {
        Err(BackendError::Unsupported("copies not supported".into()))
    }

    async fn read_tree(&self, _path: &RepoPath, id: &TreeId) -> BackendResult<Tree> {
        if *id == self.empty_tree_id {
            return Ok(Tree::from_sorted_entries(vec![]));
        }

        let proto_entries = self
            .store
            .get_tree(&self.repo_id, id.as_bytes())
            .await
            .map_err(|e| match e {
                StoreError::ObjectNotFound { .. } => BackendError::ObjectNotFound {
                    object_type: "tree".into(),
                    hash: id.hex(),
                    source: e.into(),
                },
                other => BackendError::Other(other.into()),
            })?;

        let mut jj_entries = Vec::with_capacity(proto_entries.len());
        for entry in proto_entries {
            let (comp, val) = tree_entry_from_proto(entry).map_err(BackendError::Other)?;
            jj_entries.push((comp, val));
        }
        jj_entries.sort_by(|(a, _), (b, _)| a.cmp(b));
        Ok(Tree::from_sorted_entries(jj_entries))
    }

    async fn write_tree(&self, _path: &RepoPath, tree: &Tree) -> BackendResult<TreeId> {
        if tree.entries().next().is_none() {
            return Ok(self.empty_tree_id.clone());
        }

        let proto_entries: Result<Vec<_>, _> =
            tree.entries().map(|e| tree_entry_to_proto(&e)).collect();
        let proto_entries = proto_entries?;
        let tree_id_bytes =
            compute_git_tree_hash(&proto_entries).map_err(|e| BackendError::Other(e.into()))?;
        self.store
            .put_tree(&self.repo_id, &tree_id_bytes, &proto_entries)
            .await
            .map_err(|e| BackendError::Other(e.into()))?;
        Ok(TreeId::from_bytes(&tree_id_bytes))
    }

    async fn read_commit(&self, id: &CommitId) -> BackendResult<Commit> {
        if *id == self.root_commit_id {
            return Ok(make_root_commit(
                self.root_change_id.clone(),
                self.empty_tree_id.clone(),
            ));
        }

        let proto_commit = self
            .store
            .get_commit(&self.repo_id, id.as_bytes())
            .await
            .map_err(|e| match e {
                StoreError::ObjectNotFound { .. } => BackendError::ObjectNotFound {
                    object_type: "commit".into(),
                    hash: id.hex(),
                    source: e.into(),
                },
                other => BackendError::Other(other.into()),
            })?;
        Ok(commit_from_proto(proto_commit))
    }

    async fn write_commit(
        &self,
        mut commit: Commit,
        sign_with: Option<&mut SigningFn>,
    ) -> BackendResult<(CommitId, Commit)> {
        apply_commit_signing(&mut commit, sign_with)?;
        let mut proto_commit = commit_to_proto(&commit);
        let commit_id_bytes =
            compute_git_commit_hash(&proto_commit).map_err(|e| BackendError::Other(e.into()))?;
        proto_commit.commit_id = commit_id_bytes.clone();
        self.store
            .put_commit(&self.repo_id, &commit_id_bytes, &proto_commit)
            .await
            .map_err(|e| BackendError::Other(e.into()))?;
        Ok((CommitId::from_bytes(&commit_id_bytes), commit))
    }

    fn get_copy_records(
        &self,
        _paths: Option<&[RepoPathBuf]>,
        _root: &CommitId,
        _head: &CommitId,
    ) -> BackendResult<BoxStream<'_, BackendResult<CopyRecord>>> {
        Ok(stream::empty().boxed())
    }

    fn gc(&self, _index: &dyn Index, _keep_newer: SystemTime) -> BackendResult<()> {
        Ok(())
    }
}

pub struct ServerOpStore {
    store: Arc<CompositeStore>,
    repo_id: String,
    root_operation_id: OperationId,
}

impl Debug for ServerOpStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServerOpStore")
            .field("repo_id", &self.repo_id)
            .finish()
    }
}

impl ServerOpStore {
    pub fn new(store: Arc<CompositeStore>, repo_id: String) -> Self {
        Self {
            store,
            repo_id,
            root_operation_id: OperationId::from_bytes(&cc_common::ROOT_OPERATION_ID_BYTES),
        }
    }
}

#[async_trait]
impl OpStore for ServerOpStore {
    fn name(&self) -> &str {
        "server_op_store"
    }

    fn root_operation_id(&self) -> &OperationId {
        &self.root_operation_id
    }

    async fn read_operation(&self, id: &OperationId) -> OpStoreResult<Operation> {
        let cache_key = (
            store_cache_key(&self.store),
            self.repo_id.clone(),
            id.clone(),
        );
        if let Some(entry) = OPERATION_CACHE.lock().unwrap().get(&cache_key)
            && store_entry_matches(entry, &self.store)
        {
            return Ok(entry.value.clone());
        }

        if let Some(op) = self
            .store
            .get_operation(&self.repo_id, id.as_bytes())
            .await
            .map_err(|e| OpStoreError::Other(e.into()))?
        {
            let converted = operation_from_proto(op);
            OPERATION_CACHE.lock().unwrap().insert(
                cache_key,
                StoreScopedEntry {
                    metadata_ref: Arc::downgrade(self.store.metadata()),
                    value: converted.clone(),
                },
            );
            return Ok(converted);
        }

        if id.as_bytes() == cc_common::ROOT_OPERATION_ID_BYTES {
            let root_op = cc_common::op_store::Operation {
                view_id: cc_common::ROOT_VIEW_ID_BYTES.to_vec(),
                parents: vec![],
                metadata: Some(cc_common::op_store::OperationMetadata {
                    start_time_millis: 0,
                    end_time_millis: 0,
                    description: "root()".to_string(),
                    hostname: String::new(),
                    username: String::new(),
                    is_snapshot: false,
                    workspace_name: None,
                    attributes: HashMap::new(),
                }),
                commit_predecessors: vec![],
                commit_predecessors_set: true,
            };
            let converted = operation_from_proto(root_op);
            OPERATION_CACHE.lock().unwrap().insert(
                cache_key,
                StoreScopedEntry {
                    metadata_ref: Arc::downgrade(self.store.metadata()),
                    value: converted.clone(),
                },
            );
            return Ok(converted);
        }

        Err(OpStoreError::ObjectNotFound {
            object_type: "operation".into(),
            hash: id.hex(),
            source: "operation not found in server store".into(),
        })
    }

    async fn write_operation(&self, operation: &Operation) -> OpStoreResult<OperationId> {
        let op_proto = operation_to_proto(operation);
        let op_id_bytes = hash_operation(&op_proto);
        self.store
            .put_operation(&self.repo_id, &op_id_bytes, &op_proto)
            .await
            .map_err(|e| OpStoreError::Other(e.into()))?;
        let op_id = OperationId::from_bytes(&op_id_bytes);
        OPERATION_CACHE.lock().unwrap().insert(
            (
                store_cache_key(&self.store),
                self.repo_id.clone(),
                op_id.clone(),
            ),
            StoreScopedEntry {
                metadata_ref: Arc::downgrade(self.store.metadata()),
                value: operation.clone(),
            },
        );
        Ok(op_id)
    }

    async fn read_view(&self, id: &ViewId) -> OpStoreResult<View> {
        if let Some(view) = self
            .store
            .get_view(&self.repo_id, id.as_bytes())
            .await
            .map_err(|e| OpStoreError::Other(e.into()))?
        {
            return Ok(view_from_proto(view));
        }

        if id.as_bytes() == cc_common::ROOT_VIEW_ID_BYTES {
            let root_view = cc_common::op_store::View {
                head_ids: vec![cc_common::ROOT_COMMIT_ID_BYTES.to_vec()],
                wc_commit_ids: HashMap::new(),
                local_bookmarks: HashMap::new(),
                remote_bookmarks: HashMap::new(),
            };
            return Ok(view_from_proto(root_view));
        }

        Err(OpStoreError::ObjectNotFound {
            object_type: "view".into(),
            hash: id.hex(),
            source: "view not found in server store".into(),
        })
    }

    async fn write_view(&self, view: &View) -> OpStoreResult<ViewId> {
        let view_proto = view_to_proto(view);
        let view_id_bytes = hash_view(&view_proto);
        self.store
            .put_view(&self.repo_id, &view_id_bytes, &view_proto)
            .await
            .map_err(|e| OpStoreError::Other(e.into()))?;
        Ok(ViewId::from_bytes(&view_id_bytes))
    }

    async fn resolve_operation_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> OpStoreResult<PrefixResolution<OperationId>> {
        let hex_str = prefix.hex();
        let res = self
            .store
            .resolve_operation_prefix(&self.repo_id, &hex_str)
            .await
            .map_err(|e| OpStoreError::Other(e.into()))?;
        Ok(match res {
            IdPrefixResolution::NoMatch => PrefixResolution::NoMatch,
            IdPrefixResolution::SingleMatch(id) => {
                PrefixResolution::SingleMatch(OperationId::from_bytes(&id))
            }
            IdPrefixResolution::AmbiguousMatch => PrefixResolution::AmbiguousMatch,
        })
    }

    async fn gc(
        &self,
        _target_op_ids: &[OperationId],
        _keep_newer: SystemTime,
    ) -> OpStoreResult<()> {
        Ok(())
    }
}

pub struct ServerOpHeadsStore {
    store: Arc<CompositeStore>,
    repo_id: String,
}

impl Debug for ServerOpHeadsStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServerOpHeadsStore")
            .field("repo_id", &self.repo_id)
            .finish()
    }
}

pub struct ServerOpHeadsStoreNoLock;
impl OpHeadsStoreLock for ServerOpHeadsStoreNoLock {}

impl ServerOpHeadsStore {
    pub fn new(store: Arc<CompositeStore>, repo_id: String) -> Self {
        Self { store, repo_id }
    }
}

#[async_trait]
impl OpHeadsStore for ServerOpHeadsStore {
    fn name(&self) -> &str {
        "server_op_heads_store"
    }

    async fn update_op_heads(
        &self,
        old_ids: &[OperationId],
        new_id: &OperationId,
    ) -> Result<(), OpHeadsStoreError> {
        let old_vec: Vec<Vec<u8>> = old_ids.iter().map(|id| id.as_bytes().to_vec()).collect();
        let new_vec = new_id.as_bytes().to_vec();
        let target_id = new_id.clone();

        self.store
            .update_op_heads(&self.repo_id, &old_vec, &new_vec)
            .await
            .map(|_| ())
            .map_err(|e| OpHeadsStoreError::Write {
                new_op_id: target_id,
                source: e.into(),
            })
    }

    async fn get_op_heads(&self) -> Result<Vec<OperationId>, OpHeadsStoreError> {
        let heads = self
            .store
            .get_op_heads(&self.repo_id)
            .await
            .map_err(|e| OpHeadsStoreError::Read(e.into()))?;
        Ok(heads
            .into_iter()
            .map(|b| OperationId::from_bytes(&b))
            .collect())
    }

    async fn lock(&self) -> Result<Box<dyn OpHeadsStoreLock + '_>, OpHeadsStoreError> {
        Ok(Box::new(ServerOpHeadsStoreNoLock))
    }
}

#[derive(Debug)]
pub struct EmptyRevset;

impl Revset for EmptyRevset {
    fn stream<'a>(&self) -> LocalBoxStream<'a, Result<CommitId, RevsetEvaluationError>>
    where
        Self: 'a,
    {
        Box::pin(futures::stream::empty())
    }

    fn commit_change_ids<'a>(
        &self,
    ) -> LocalBoxStream<'a, Result<(CommitId, ChangeId), RevsetEvaluationError>>
    where
        Self: 'a,
    {
        Box::pin(futures::stream::empty())
    }

    fn stream_graph<'a>(
        &self,
    ) -> LocalBoxStream<'a, Result<GraphNode<CommitId>, RevsetEvaluationError>>
    where
        Self: 'a,
    {
        Box::pin(futures::stream::empty())
    }

    fn is_empty(&self) -> bool {
        true
    }

    fn count_estimate(&self) -> Result<(usize, Option<usize>), RevsetEvaluationError> {
        Ok((0, Some(0)))
    }

    fn containing_fn<'a>(&self) -> Box<RevsetContainingFn<'a>>
    where
        Self: 'a,
    {
        Box::new(|_id| Ok(false))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GenCommitNode(u64, CommitId);

impl Ord for GenCommitNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .cmp(&other.0)
            .then_with(|| self.1.as_bytes().cmp(other.1.as_bytes()))
    }
}

impl PartialOrd for GenCommitNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Graph-aware [`Index`] backed by [`CompositeStore`] commit headers and generation
/// numbers so view merges resolve fast-forward bookmark advances (`is_ancestor`) and
/// prune superseded heads (`heads`) without walking full commit history.
#[derive(Clone)]
pub struct ServerIndex {
    store: Arc<CompositeStore>,
    repo_id: String,
    project_id: Arc<OnceLock<String>>,
}

impl Debug for ServerIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServerIndex")
            .field("repo_id", &self.repo_id)
            .finish()
    }
}

impl ServerIndex {
    pub fn new(store: Arc<CompositeStore>, repo_id: String) -> Self {
        Self {
            store,
            repo_id,
            project_id: Arc::new(OnceLock::new()),
        }
    }

    fn resolve_project_id(&self) -> IndexResult<&str> {
        if let Some(pid) = self.project_id.get() {
            return Ok(pid.as_str());
        }
        let store = self.store.clone();
        let repo_id = self.repo_id.clone();
        let resolved = pollster::block_on(async move {
            store
                .resolve_repo_project(&repo_id, "indexing server commits")
                .await
        })
        .map_err(|e| IndexError::Other(e.into()))?;
        let _ = self.project_id.set(resolved);
        Ok(self.project_id.get().unwrap().as_str())
    }

    fn commit_node(&self, commit_id: &CommitId) -> IndexResult<(u64, Vec<CommitId>)> {
        if commit_id.as_bytes() == cc_common::ROOT_COMMIT_ID_BYTES {
            return Ok((0, Vec::new()));
        }
        let project_id = self.resolve_project_id()?;
        let cache_key = (
            store_cache_key(&self.store),
            project_id.to_string(),
            commit_id.clone(),
        );
        if let Some(entry) = COMMIT_GRAPH_CACHE.lock().unwrap().get(&cache_key)
            && store_entry_matches(entry, &self.store)
        {
            return Ok(entry.value.clone());
        }

        let metadata = self.store.metadata().clone();
        let pid_owned = project_id.to_string();
        let cid_bytes = commit_id.as_bytes().to_vec();
        let (commit_res, gen_res) = pollster::block_on(async move {
            futures::future::join(
                metadata.get_commit(&pid_owned, &cid_bytes),
                metadata.get_commit_generation(&pid_owned, &cid_bytes),
            )
            .await
        });

        let proto_commit = commit_res
            .map_err(|e| IndexError::Other(e.into()))?
            .ok_or_else(|| {
                IndexError::Other(
                    format!(
                        "corrupt commit graph: missing commit {} in project {project_id}",
                        commit_id.hex()
                    )
                    .into(),
                )
            })?;
        let generation = gen_res.map_err(|e| IndexError::Other(e.into()))?.ok_or_else(|| {
            IndexError::Other(
                format!(
                    "corrupt commit graph: missing generation for commit {} in project {project_id}",
                    commit_id.hex()
                )
                .into(),
            )
        })?;
        let parents: Vec<CommitId> = proto_commit
            .parent_commit_ids
            .iter()
            .map(|p| CommitId::from_bytes(p))
            .collect();

        COMMIT_GRAPH_CACHE.lock().unwrap().insert(
            cache_key,
            StoreScopedEntry {
                metadata_ref: Arc::downgrade(self.store.metadata()),
                value: (generation, parents.clone()),
            },
        );
        Ok((generation, parents))
    }
}

impl Index for ServerIndex {
    fn shortest_unique_commit_id_prefix_len(&self, _commit_id: &CommitId) -> IndexResult<usize> {
        Ok(12)
    }

    fn resolve_commit_id_prefix(
        &self,
        _prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<CommitId>> {
        Ok(PrefixResolution::NoMatch)
    }

    fn has_id(&self, commit_id: &CommitId) -> IndexResult<bool> {
        if commit_id.as_bytes() == cc_common::ROOT_COMMIT_ID_BYTES {
            return Ok(true);
        }
        let project_id = self.resolve_project_id()?;
        let cache_key = (
            store_cache_key(&self.store),
            project_id.to_string(),
            commit_id.clone(),
        );
        if let Some(entry) = COMMIT_GRAPH_CACHE.lock().unwrap().get(&cache_key)
            && store_entry_matches(entry, &self.store)
        {
            return Ok(true);
        }
        let metadata = self.store.metadata().clone();
        let pid_owned = project_id.to_string();
        let cid_bytes = commit_id.as_bytes().to_vec();
        let (commit_res, gen_res) = pollster::block_on(async move {
            futures::future::join(
                metadata.get_commit(&pid_owned, &cid_bytes),
                metadata.get_commit_generation(&pid_owned, &cid_bytes),
            )
            .await
        });
        match (commit_res, gen_res) {
            (Ok(Some(proto_commit)), Ok(Some(generation))) => {
                let parents: Vec<CommitId> = proto_commit
                    .parent_commit_ids
                    .iter()
                    .map(|p| CommitId::from_bytes(p))
                    .collect();
                COMMIT_GRAPH_CACHE.lock().unwrap().insert(
                    cache_key,
                    StoreScopedEntry {
                        metadata_ref: Arc::downgrade(self.store.metadata()),
                        value: (generation, parents),
                    },
                );
                Ok(true)
            }
            (Ok(None), _) | (_, Ok(None)) => Ok(false),
            (Err(e), _) | (_, Err(e)) => Err(IndexError::Other(e.into())),
        }
    }

    fn is_ancestor(&self, ancestor_id: &CommitId, descendant_id: &CommitId) -> IndexResult<bool> {
        if ancestor_id == descendant_id || ancestor_id.as_bytes() == cc_common::ROOT_COMMIT_ID_BYTES
        {
            return Ok(true);
        }
        if descendant_id.as_bytes() == cc_common::ROOT_COMMIT_ID_BYTES {
            return Ok(false);
        }
        let (target_gen, _) = self.commit_node(ancestor_id)?;
        let (desc_gen, _) = self.commit_node(descendant_id)?;
        if desc_gen <= target_gen && target_gen < u64::MAX {
            return Ok(false);
        }

        let mut visited = HashSet::from([descendant_id.clone()]);
        let mut heap = BinaryHeap::from([GenCommitNode(desc_gen, descendant_id.clone())]);
        while let Some(GenCommitNode(_, curr_id)) = heap.pop() {
            let (_, parents) = self.commit_node(&curr_id)?;
            for p in parents {
                if &p == ancestor_id {
                    return Ok(true);
                }
                if p.as_bytes() == cc_common::ROOT_COMMIT_ID_BYTES || !visited.insert(p.clone()) {
                    continue;
                }
                let (p_gen, _) = self.commit_node(&p)?;
                if p_gen > target_gen || p_gen == u64::MAX {
                    heap.push(GenCommitNode(p_gen, p));
                }
            }
        }
        Ok(false)
    }

    fn common_ancestors(&self, set1: &[CommitId], set2: &[CommitId]) -> IndexResult<Vec<CommitId>> {
        if set1.is_empty() || set2.is_empty() {
            return Ok(Vec::new());
        }
        let mut start1 = Vec::with_capacity(set1.len());
        for c in set1 {
            start1.push(GenCommitNode(self.commit_node(c)?.0, c.clone()));
        }
        let mut start2 = Vec::with_capacity(set2.len());
        for c in set2 {
            start2.push(GenCommitNode(self.commit_node(c)?.0, c.clone()));
        }
        let root_id = CommitId::from_bytes(&cc_common::ROOT_COMMIT_ID_BYTES);
        let common_nodes = pollster::block_on(jj_lib::dag_walk_async::closest_common_nodes(
            start1,
            start2,
            |n: &GenCommitNode| n.1.clone(),
            async |n: &GenCommitNode| -> IndexResult<Vec<GenCommitNode>> {
                if n.1 == root_id {
                    return Ok(Vec::new());
                }
                let (_, mut parents) = self.commit_node(&n.1)?;
                if parents.is_empty() {
                    parents.push(root_id.clone());
                }
                let mut out = Vec::with_capacity(parents.len());
                for p in parents {
                    out.push(GenCommitNode(self.commit_node(&p)?.0, p));
                }
                Ok(out)
            },
        ))?;
        let common_ids: Vec<CommitId> = common_nodes.into_iter().map(|n| n.1).collect();
        self.heads(&mut common_ids.iter())
    }

    fn all_heads_for_gc(&self) -> IndexResult<Box<dyn Iterator<Item = CommitId> + '_>> {
        Err(IndexError::AllHeadsForGcUnsupported)
    }

    fn heads(&self, candidates: &mut dyn Iterator<Item = &CommitId>) -> IndexResult<Vec<CommitId>> {
        let mut unique = Vec::new();
        let mut seen = HashSet::new();
        for c in candidates {
            if seen.insert(c.clone()) {
                unique.push(c.clone());
            }
        }
        if unique.len() <= 1 {
            return Ok(unique);
        }
        unique.retain(|c| c.as_bytes() != cc_common::ROOT_COMMIT_ID_BYTES);
        if unique.len() <= 1 {
            return Ok(unique);
        }

        let mut cand_nodes = Vec::with_capacity(unique.len());
        for c in unique {
            let (gen_num, _) = self.commit_node(&c)?;
            cand_nodes.push(GenCommitNode(gen_num, c));
        }
        let min_gen = cand_nodes.iter().map(|n| n.0).min().unwrap_or(0);
        cand_nodes.sort_by(|a, b| b.cmp(a));

        let mut proper_ancestors = HashSet::new();
        let mut frontier: BinaryHeap<GenCommitNode> = BinaryHeap::new();
        let mut heads = Vec::new();

        let enqueue_parents = |commit_id: &CommitId,
                               proper_ancestors: &mut HashSet<CommitId>,
                               frontier: &mut BinaryHeap<GenCommitNode>|
         -> IndexResult<()> {
            let (_, parents) = self.commit_node(commit_id)?;
            for p in parents {
                if proper_ancestors.insert(p.clone())
                    && p.as_bytes() != cc_common::ROOT_COMMIT_ID_BYTES
                {
                    let (p_gen, _) = self.commit_node(&p)?;
                    if p_gen > min_gen || p_gen == u64::MAX {
                        frontier.push(GenCommitNode(p_gen, p));
                    }
                }
            }
            Ok(())
        };

        for GenCommitNode(cand_gen, cand_id) in cand_nodes {
            while frontier
                .peek()
                .is_some_and(|top| top.0 > cand_gen || top.0 == u64::MAX)
            {
                let GenCommitNode(curr_gen, curr_id) = frontier.pop().unwrap();
                if curr_gen > min_gen || curr_gen == u64::MAX {
                    enqueue_parents(&curr_id, &mut proper_ancestors, &mut frontier)?;
                }
            }
            if proper_ancestors.contains(&cand_id) {
                continue;
            }
            if cand_gen > min_gen || cand_gen == u64::MAX {
                enqueue_parents(&cand_id, &mut proper_ancestors, &mut frontier)?;
            }
            heads.push(cand_id);
        }

        heads.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        Ok(heads)
    }

    fn changed_paths_in_commit(
        &self,
        _commit_id: &CommitId,
    ) -> IndexResult<Option<Box<dyn Iterator<Item = RepoPathBuf> + '_>>> {
        Ok(None)
    }

    fn evaluate_revset(
        &self,
        _expression: &ResolvedExpression,
        _store: &Arc<JjStore>,
    ) -> Result<Box<dyn Revset + '_>, RevsetEvaluationError> {
        Ok(Box::new(EmptyRevset))
    }
}

pub struct ServerChangeIdIndex;

impl ChangeIdIndex for ServerChangeIdIndex {
    fn resolve_prefix(
        &self,
        _prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<ResolvedChangeTargets>> {
        Ok(PrefixResolution::NoMatch)
    }

    fn shortest_unique_prefix_len(&self, _change_id: &ChangeId) -> IndexResult<usize> {
        Ok(12)
    }
}

impl ReadonlyIndex for ServerIndex {
    fn as_index(&self) -> &dyn Index {
        self
    }

    fn change_id_index(
        &self,
        _heads: &mut dyn Iterator<Item = &CommitId>,
    ) -> Box<dyn ChangeIdIndex> {
        Box::new(ServerChangeIdIndex)
    }

    fn start_modification(&self) -> Box<dyn MutableIndex> {
        Box::new(self.clone())
    }
}

#[async_trait]
impl MutableIndex for ServerIndex {
    fn as_index(&self) -> &dyn Index {
        self
    }

    fn change_id_index(
        &self,
        _heads: &mut dyn Iterator<Item = &CommitId>,
    ) -> Box<dyn ChangeIdIndex + '_> {
        Box::new(ServerChangeIdIndex)
    }

    async fn add_commit(&mut self, commit: &JjCommit) -> IndexResult<()> {
        let project_id = self.resolve_project_id()?.to_string();
        let mut max_parent_gen = 0u64;
        for parent_id in commit.parent_ids() {
            let (p_gen, _) = self.commit_node(parent_id)?;
            max_parent_gen = max_parent_gen.max(p_gen);
        }
        let generation = max_parent_gen.saturating_add(1);
        COMMIT_GRAPH_CACHE.lock().unwrap().insert(
            (
                store_cache_key(&self.store),
                project_id,
                commit.id().clone(),
            ),
            StoreScopedEntry {
                metadata_ref: Arc::downgrade(self.store.metadata()),
                value: (generation, commit.parent_ids().to_vec()),
            },
        );
        Ok(())
    }

    fn merge_in(&mut self, _other: &dyn ReadonlyIndex) -> IndexResult<()> {
        Ok(())
    }
}

pub struct ServerIndexStore {
    store: Arc<CompositeStore>,
    repo_id: String,
}

impl Debug for ServerIndexStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServerIndexStore")
            .field("repo_id", &self.repo_id)
            .finish()
    }
}

impl ServerIndexStore {
    pub fn new(store: Arc<CompositeStore>, repo_id: String) -> Self {
        Self { store, repo_id }
    }
}

#[async_trait(?Send)]
impl IndexStore for ServerIndexStore {
    fn name(&self) -> &str {
        "server"
    }

    async fn get_index_at_op(
        &self,
        _op: &JjOperation,
        _store: &Arc<JjStore>,
    ) -> IndexStoreResult<Box<dyn ReadonlyIndex>> {
        Ok(Box::new(ServerIndex::new(
            self.store.clone(),
            self.repo_id.clone(),
        )))
    }

    fn write_index(
        &self,
        index: Box<dyn MutableIndex>,
        _op: &JjOperation,
    ) -> IndexStoreResult<Box<dyn ReadonlyIndex>> {
        let _ = index;
        Ok(Box::new(ServerIndex::new(
            self.store.clone(),
            self.repo_id.clone(),
        )))
    }
}

#[derive(Debug, Default)]
pub struct ServerSubmoduleStore;

impl SubmoduleStore for ServerSubmoduleStore {
    fn name(&self) -> &str {
        "server"
    }
}

#[cfg(test)]
mod tests {
    use cc_common::backend::Commit as ProtoCommit;
    use cc_common::op_store::Operation as ProtoOperation;

    use super::*;
    use crate::store::{CompositeStoreConfig, MemoryBlobStore, MemoryMetadataStore};

    fn dummy_proto_commit(cid: &[u8], parents: Vec<Vec<u8>>) -> ProtoCommit {
        ProtoCommit {
            commit_id: cid.to_vec(),
            change_id: cid[..16].to_vec(),
            parent_commit_ids: parents,
            root_tree_id: vec![cc_common::EMPTY_TREE_ID_BYTES.to_vec()],
            description: "test\n".to_string(),
            author: None,
            committer: None,
            predecessors: vec![],
            conflict_labels: vec![],
            secure_sig: None,
            raw_git_commit: vec![],
            external_ids: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_server_index_prunes_by_generation_and_errors_on_missing_parents() {
        let meta = Arc::new(MemoryMetadataStore::new());
        let blobs = Arc::new(MemoryBlobStore::new());
        let store = Arc::new(CompositeStore::with_config(
            meta.clone(),
            blobs,
            CompositeStoreConfig {
                inline_blob_threshold: 0,
                validate_dependencies: false,
            },
        ));
        store.register_repo("prune-repo", None).await.unwrap();

        let missing_anc = vec![0xaa; 20];
        let c1 = vec![0x01; 20];
        let c198 = vec![0xc6; 20];
        let c199 = vec![0xc7; 20];

        // Persist `c1` (gen 1), `c198` (gen 199 -> parent `missing_anc`), and `c199` (gen 200 ->
        // parent `c198`), intentionally leaving `missing_anc` absent from storage.
        meta.put_commit(
            "prune-repo",
            &c1,
            &dummy_proto_commit(&c1, vec![cc_common::ROOT_COMMIT_ID_BYTES.to_vec()]),
            1,
        )
        .await
        .unwrap();
        meta.put_commit(
            "prune-repo",
            &c198,
            &dummy_proto_commit(&c198, vec![missing_anc.clone()]),
            199,
        )
        .await
        .unwrap();
        meta.put_commit(
            "prune-repo",
            &c199,
            &dummy_proto_commit(&c199, vec![c198.clone()]),
            200,
        )
        .await
        .unwrap();

        let jj_c1 = CommitId::from_bytes(&c1);
        let jj_c198 = CommitId::from_bytes(&c198);
        let jj_c199 = CommitId::from_bytes(&c199);

        let idx = ServerIndex::new(store.clone(), "prune-repo".to_string());
        // `heads([c198, c199])` and `is_ancestor(c198, c199)` stop at generation 199 without
        // walking into `missing_anc`!
        assert!(idx.is_ancestor(&jj_c198, &jj_c199).unwrap());
        assert!(!idx.is_ancestor(&jj_c199, &jj_c198).unwrap());
        assert_eq!(
            idx.heads(&mut [&jj_c198, &jj_c199].into_iter()).unwrap(),
            vec![jj_c199.clone()]
        );

        // Walking past generation 199 toward `c1` (gen 1) encounters `missing_anc` and surfaces
        // a corruption error instead of silently treating `c198` as a root.
        let err = idx.is_ancestor(&jj_c1, &jj_c199).unwrap_err();
        assert!(
            err.to_string().contains("corrupt commit graph"),
            "expected corruption error for missing parent, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_cross_reconcile_caches_retain_commits_and_operations() {
        let store = Arc::new(CompositeStore::in_memory());
        store.register_repo("cache-repo", None).await.unwrap();

        let cid = vec![0x44; 20];
        store
            .put_commit(
                "cache-repo",
                &cid,
                &dummy_proto_commit(&cid, vec![cc_common::ROOT_COMMIT_ID_BYTES.to_vec()]),
            )
            .await
            .unwrap();

        let op_proto = ProtoOperation {
            view_id: cc_common::ROOT_VIEW_ID_BYTES.to_vec(),
            parents: vec![cc_common::ROOT_OPERATION_ID_BYTES.to_vec()],
            metadata: None,
            commit_predecessors: vec![],
            commit_predecessors_set: true,
        };
        let op_id_bytes = hash_operation(&op_proto);
        store
            .put_operation("cache-repo", &op_id_bytes, &op_proto)
            .await
            .unwrap();

        let idx1 = ServerIndex::new(store.clone(), "cache-repo".to_string());
        let jj_cid = CommitId::from_bytes(&cid);
        assert!(idx1.has_id(&jj_cid).unwrap());

        let op_store1 = ServerOpStore::new(store.clone(), "cache-repo".to_string());
        let jj_op_id = OperationId::from_bytes(&op_id_bytes);
        let read1 = op_store1.read_operation(&jj_op_id).await.unwrap();

        // Fresh adapter instances constructed in a subsequent reconcile hit the shared caches.
        let key_c = (store_cache_key(&store), "cache-repo".to_string(), jj_cid);
        assert!(COMMIT_GRAPH_CACHE.lock().unwrap().get(&key_c).is_some());
        let key_op = (
            store_cache_key(&store),
            "cache-repo".to_string(),
            jj_op_id.clone(),
        );
        assert!(OPERATION_CACHE.lock().unwrap().get(&key_op).is_some());

        let op_store2 = ServerOpStore::new(store, "cache-repo".to_string());
        let read2 = op_store2.read_operation(&jj_op_id).await.unwrap();
        assert_eq!(read1.view_id, read2.view_id);
    }
}
