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

//! Ephemeral in-memory `MetadataStore` and `BlobStore` implementations for unit tests and
//! volatile local development stacks (`--store-type=memory`).
//!
//! # Authority and Crash-Safety Ordering
//! `MemoryMetadataStore` mirrors the two-scope `Projects` / `Repositories` relational
//! boundary of the SQLite and Cloud Spanner backends in `Mutex`-guarded maps, keeping
//! immutable content-addressed commits (`or_insert`) and UTF-8 name-sorted directory
//! entries consistent with persistent backends.
//!
//! # Concurrency and Merge Invariants
//! `update_op_heads`, `compare_and_swap_op_heads`, and `publish_operation` execute under
//! serialized per-map mutexes so concurrent sibling operation heads are preserved without
//! read-modify-write races.
//!
//! # Cross-Process and RPC Boundaries
//! State lives strictly inside the `jj-cc-server` process address space and resets on
//! process exit.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Bound;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use cc_common::backend::*;
use cc_common::op_store::*;
use cc_common::workspace::WorkspaceState;

use super::{
    BlobRecord, BlobStore, MetadataStore, StoreError, StoreResult, apply_op_heads_mutation,
    op_heads_sets_equal,
};

pub type ProjectId = String;
pub type RepoId = String;
pub type CommitId = Vec<u8>;
pub type TreeId = Vec<u8>;
pub type FileId = Vec<u8>;
pub type SymlinkId = Vec<u8>;
pub type OpId = Vec<u8>;
pub type ViewId = Vec<u8>;

#[derive(Clone, Debug)]
struct StoredCommit {
    commit: Commit,
    generation: u64,
    seq: u64,
}

#[derive(Clone, Debug)]
struct RepoMeta {
    project_id: ProjectId,
    name: Option<String>,
}

#[derive(Debug, Default)]
pub struct MemoryMetadataStore {
    next_commit_seq: AtomicU64,
    projects: Mutex<HashMap<ProjectId, Option<String>>>,
    project_names: Mutex<HashMap<String, ProjectId>>,
    repos: Mutex<HashMap<RepoId, RepoMeta>>,
    repo_names: Mutex<HashMap<String, RepoId>>,
    commits: Mutex<HashMap<ProjectId, BTreeMap<CommitId, StoredCommit>>>,
    trees: Mutex<HashMap<ProjectId, HashMap<TreeId, Vec<TreeEntry>>>>,
    symlinks: Mutex<HashMap<ProjectId, HashMap<SymlinkId, String>>>,
    // Methods that hold both blob locks take `project_blobs` before `blobs`, so concurrent
    // blob writes and tree or file reads cannot deadlock.
    blobs: Mutex<HashMap<FileId, BlobRecord>>,
    project_blobs: Mutex<HashMap<ProjectId, HashSet<FileId>>>,
    ops: Mutex<HashMap<RepoId, BTreeMap<OpId, Operation>>>,
    views: Mutex<HashMap<RepoId, HashMap<ViewId, View>>>,
    op_heads: Mutex<HashMap<RepoId, Vec<OpId>>>,
    workspaces: Mutex<HashMap<RepoId, BTreeMap<(String, String), WorkspaceState>>>,
}

impl MemoryMetadataStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl MetadataStore for MemoryMetadataStore {
    fn backend_name(&self) -> &'static str {
        "memory"
    }

    async fn register_project(&self, project_id: &str, name: Option<&str>) -> StoreResult<()> {
        let mut projects = self.projects.lock().unwrap();
        let mut project_names = self.project_names.lock().unwrap();
        if let Some(alias) = name {
            if let Some(existing) = project_names.get(alias)
                && existing != project_id
            {
                return Err(StoreError::ProjectAlreadyExists(alias.to_string()));
            }
            project_names.insert(alias.to_string(), project_id.to_string());
        }
        let entry = projects.entry(project_id.to_string()).or_insert(None);
        if name.is_some() {
            *entry = name.map(ToOwned::to_owned);
        }
        Ok(())
    }

    async fn is_project_registered(&self, project_id: &str) -> StoreResult<bool> {
        Ok(self.projects.lock().unwrap().contains_key(project_id))
    }

    async fn get_repo_project_id(&self, repo_id: &str) -> StoreResult<Option<String>> {
        Ok(self
            .repos
            .lock()
            .unwrap()
            .get(repo_id)
            .map(|m| m.project_id.clone()))
    }

    async fn register_repo_in_project(
        &self,
        repo_id: &str,
        project_id: &str,
        name: Option<&str>,
    ) -> StoreResult<String> {
        {
            let mut projects = self.projects.lock().unwrap();
            projects.entry(project_id.to_string()).or_insert(None);
        }
        let mut repos = self.repos.lock().unwrap();
        let mut repo_names = self.repo_names.lock().unwrap();

        if let Some(existing_meta) = repos.get(repo_id)
            && existing_meta.project_id != project_id
        {
            return Err(StoreError::InvalidArgument(format!(
                "repository '{repo_id}' is already bound to project '{}', cannot rebind to '{project_id}'",
                existing_meta.project_id
            )));
        }

        if let Some(alias) = name {
            if let Some(existing) = repo_names.get(alias)
                && existing != repo_id
            {
                return Err(StoreError::RepoAlreadyExists(alias.to_string()));
            }
            repo_names.insert(alias.to_string(), repo_id.to_string());
        }

        let effective_name = name
            .map(ToOwned::to_owned)
            .or_else(|| repos.get(repo_id).and_then(|m| m.name.clone()));
        repos.insert(
            repo_id.to_string(),
            RepoMeta {
                project_id: project_id.to_string(),
                name: effective_name,
            },
        );
        Ok(project_id.to_string())
    }

    async fn get_commit(&self, project_id: &str, commit_id: &[u8]) -> StoreResult<Option<Commit>> {
        let commits = self.commits.lock().unwrap();
        Ok(commits
            .get(project_id)
            .and_then(|m| m.get(commit_id))
            .map(|sc| sc.commit.clone()))
    }

    async fn get_commit_generation(
        &self,
        project_id: &str,
        commit_id: &[u8],
    ) -> StoreResult<Option<u64>> {
        let commits = self.commits.lock().unwrap();
        Ok(commits
            .get(project_id)
            .and_then(|m| m.get(commit_id))
            .map(|sc| sc.generation))
    }

    async fn get_commit_generations(
        &self,
        project_id: &str,
        commit_ids: &[Vec<u8>],
    ) -> StoreResult<HashMap<Vec<u8>, u64>> {
        let commits = self.commits.lock().unwrap();
        let Some(proj_commits) = commits.get(project_id) else {
            return Ok(HashMap::new());
        };
        Ok(commit_ids
            .iter()
            .filter_map(|id| proj_commits.get(id).map(|sc| (id.clone(), sc.generation)))
            .collect())
    }

    async fn put_commit(
        &self,
        project_id: &str,
        commit_id: &[u8],
        commit: &Commit,
        generation: u64,
    ) -> StoreResult<()> {
        let mut commits = self.commits.lock().unwrap();
        commits
            .entry(project_id.to_string())
            .or_default()
            .entry(commit_id.to_vec())
            .or_insert_with(|| {
                let seq = self.next_commit_seq.fetch_add(1, Ordering::Relaxed) + 1;
                StoredCommit {
                    commit: commit.clone(),
                    generation,
                    seq,
                }
            });
        Ok(())
    }

    async fn list_commit_ids(
        &self,
        project_id: &str,
        after_commit_id: Option<&[u8]>,
        limit: usize,
    ) -> StoreResult<Vec<CommitId>> {
        let commits = self.commits.lock().unwrap();
        let Some(proj_commits) = commits.get(project_id) else {
            return Ok(Vec::new());
        };
        match after_commit_id {
            Some(cursor) if !cursor.is_empty() => Ok(proj_commits
                .range((Bound::Excluded(cursor.to_vec()), Bound::Unbounded))
                .take(limit)
                .map(|(k, _)| k.clone())
                .collect()),
            _ => Ok(proj_commits.keys().take(limit).cloned().collect()),
        }
    }

    async fn current_commit_watermark(&self, project_id: &str) -> StoreResult<Vec<u8>> {
        let commits = self.commits.lock().unwrap();
        let max_seq = commits
            .get(project_id)
            .and_then(|m| m.values().map(|sc| sc.seq).max())
            .unwrap_or(0);
        Ok(max_seq.to_be_bytes().to_vec())
    }

    async fn list_commit_ids_since(
        &self,
        project_id: &str,
        since_token: &[u8],
        limit: usize,
    ) -> StoreResult<(Vec<CommitId>, Vec<u8>, bool)> {
        let since_seq = if since_token.len() == 8 {
            u64::from_be_bytes(since_token.try_into().unwrap())
        } else {
            return Err(StoreError::InvalidArgument(
                "invalid memory commit sync token".to_string(),
            ));
        };
        let commits = self.commits.lock().unwrap();
        let Some(proj_commits) = commits.get(project_id) else {
            return Ok((Vec::new(), since_token.to_vec(), false));
        };
        let mut ordered: Vec<(u64, CommitId)> = proj_commits
            .iter()
            .filter(|(_, sc)| sc.seq > since_seq)
            .map(|(cid, sc)| (sc.seq, cid.clone()))
            .collect();
        ordered.sort_by_key(|(seq, _)| *seq);
        let has_more = ordered.len() > limit;
        if has_more {
            ordered.truncate(limit);
        }
        let Some(&(last_seq, _)) = ordered.last() else {
            return Ok((Vec::new(), since_token.to_vec(), false));
        };
        let ids = ordered.into_iter().map(|(_, cid)| cid).collect();
        Ok((ids, last_seq.to_be_bytes().to_vec(), has_more))
    }

    async fn resolve_commit_id_prefix(
        &self,
        project_id: &str,
        min_id: &[u8],
        max_id: &[u8],
    ) -> StoreResult<Vec<CommitId>> {
        let commits = self.commits.lock().unwrap();
        let Some(proj_commits) = commits.get(project_id) else {
            return Ok(Vec::new());
        };
        Ok(proj_commits
            .range(min_id.to_vec()..=max_id.to_vec())
            .take(2)
            .map(|(k, _)| k.clone())
            .collect())
    }

    async fn resolve_change_id_prefix(
        &self,
        project_id: &str,
        min_id: &[u8],
        max_id: &[u8],
    ) -> StoreResult<Vec<(Vec<u8>, Vec<CommitId>)>> {
        let commits = self.commits.lock().unwrap();
        let Some(proj_commits) = commits.get(project_id) else {
            return Ok(Vec::new());
        };
        let mut by_change: BTreeMap<Vec<u8>, Vec<CommitId>> = BTreeMap::new();
        for (cid, sc) in proj_commits {
            let ch = sc.commit.change_id.as_slice();
            if ch >= min_id && ch <= max_id {
                by_change
                    .entry(sc.commit.change_id.clone())
                    .or_default()
                    .push(cid.clone());
            }
        }
        Ok(by_change.into_iter().take(2).collect())
    }

    async fn get_tree(
        &self,
        project_id: &str,
        tree_id: &[u8],
    ) -> StoreResult<Option<Vec<TreeEntry>>> {
        let trees = self.trees.lock().unwrap();
        Ok(trees.get(project_id).and_then(|m| m.get(tree_id)).cloned())
    }

    async fn get_tree_page(
        &self,
        project_id: &str,
        tree_id: &[u8],
        after_name: Option<&str>,
        limit: usize,
    ) -> StoreResult<Option<Vec<TreeEntry>>> {
        let trees = self.trees.lock().unwrap();
        let Some(entries) = trees.get(project_id).and_then(|m| m.get(tree_id)) else {
            return Ok(None);
        };
        let iter = entries.iter().filter(|e| {
            after_name
                .filter(|a| !a.is_empty())
                .is_none_or(|after| e.name.as_bytes() > after.as_bytes())
        });
        Ok(Some(iter.take(limit).cloned().collect()))
    }

    async fn put_tree(
        &self,
        project_id: &str,
        tree_id: &[u8],
        entries: &[TreeEntry],
    ) -> StoreResult<()> {
        let mut sorted = entries.to_vec();
        sorted.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
        let mut trees = self.trees.lock().unwrap();
        trees
            .entry(project_id.to_string())
            .or_default()
            .insert(tree_id.to_vec(), sorted);
        Ok(())
    }

    async fn find_existing_trees(
        &self,
        project_id: &str,
        tree_ids: &[Vec<u8>],
    ) -> StoreResult<HashSet<Vec<u8>>> {
        let trees = self.trees.lock().unwrap();
        let Some(proj_trees) = trees.get(project_id) else {
            return Ok(HashSet::new());
        };
        Ok(tree_ids
            .iter()
            .filter(|id| proj_trees.contains_key(id.as_slice()))
            .cloned()
            .collect())
    }

    async fn get_symlink(
        &self,
        project_id: &str,
        symlink_id: &[u8],
    ) -> StoreResult<Option<String>> {
        let symlinks = self.symlinks.lock().unwrap();
        Ok(symlinks
            .get(project_id)
            .and_then(|m| m.get(symlink_id))
            .cloned())
    }

    async fn put_symlink(
        &self,
        project_id: &str,
        symlink_id: &[u8],
        target: &str,
    ) -> StoreResult<()> {
        let mut symlinks = self.symlinks.lock().unwrap();
        symlinks
            .entry(project_id.to_string())
            .or_default()
            .insert(symlink_id.to_vec(), target.to_string());
        Ok(())
    }

    async fn find_existing_symlinks(
        &self,
        project_id: &str,
        symlink_ids: &[Vec<u8>],
    ) -> StoreResult<HashSet<Vec<u8>>> {
        let symlinks = self.symlinks.lock().unwrap();
        let Some(proj_symlinks) = symlinks.get(project_id) else {
            return Ok(HashSet::new());
        };
        Ok(symlink_ids
            .iter()
            .filter(|id| proj_symlinks.contains_key(id.as_slice()))
            .cloned()
            .collect())
    }

    async fn get_blob_record(
        &self,
        project_id: &str,
        blob_id: &[u8],
    ) -> StoreResult<Option<BlobRecord>> {
        let project_blobs = self.project_blobs.lock().unwrap();
        let authorized = project_blobs
            .get(project_id)
            .is_some_and(|set| set.contains(blob_id));
        if !authorized {
            return Ok(None);
        }
        let blobs = self.blobs.lock().unwrap();
        Ok(blobs.get(blob_id).cloned())
    }

    async fn get_blob_sizes_batch(
        &self,
        project_id: &str,
        blob_ids: &[Vec<u8>],
    ) -> StoreResult<HashMap<Vec<u8>, u64>> {
        let project_blobs = self.project_blobs.lock().unwrap();
        let Some(proj_set) = project_blobs.get(project_id) else {
            return Ok(HashMap::new());
        };
        let blobs = self.blobs.lock().unwrap();
        let mut out = HashMap::with_capacity(blob_ids.len());
        for id in blob_ids {
            if proj_set.contains(id.as_slice())
                && let Some(rec) = blobs.get(id.as_slice())
            {
                out.insert(id.clone(), rec.content_size);
            }
        }
        Ok(out)
    }

    async fn get_global_blob_record(&self, blob_id: &[u8]) -> StoreResult<Option<BlobRecord>> {
        let blobs = self.blobs.lock().unwrap();
        Ok(blobs.get(blob_id).cloned())
    }

    async fn put_blob_record(&self, project_id: &str, record: &BlobRecord) -> StoreResult<()> {
        let mut project_blobs = self.project_blobs.lock().unwrap();
        let mut blobs = self.blobs.lock().unwrap();
        blobs.insert(record.blob_id.clone(), record.clone());
        project_blobs
            .entry(project_id.to_string())
            .or_default()
            .insert(record.blob_id.clone());
        Ok(())
    }

    async fn add_blob_replica_location(&self, blob_id: &[u8], location: &str) -> StoreResult<()> {
        let mut blobs = self.blobs.lock().unwrap();
        if let Some(rec) = blobs.get_mut(blob_id)
            && !rec.replicated_locations.iter().any(|l| l == location)
        {
            rec.replicated_locations.push(location.to_string());
        }
        Ok(())
    }

    async fn get_operation(&self, repo_id: &str, op_id: &[u8]) -> StoreResult<Option<Operation>> {
        let ops = self.ops.lock().unwrap();
        Ok(ops.get(repo_id).and_then(|m| m.get(op_id)).cloned())
    }

    async fn put_operation(&self, repo_id: &str, op_id: &[u8], op: &Operation) -> StoreResult<()> {
        let mut ops = self.ops.lock().unwrap();
        ops.entry(repo_id.to_string())
            .or_default()
            .insert(op_id.to_vec(), op.clone());
        Ok(())
    }

    async fn resolve_operation_id_prefix(
        &self,
        repo_id: &str,
        min_id: &[u8],
        max_id: &[u8],
    ) -> StoreResult<Vec<OpId>> {
        let ops = self.ops.lock().unwrap();
        let Some(repo_ops) = ops.get(repo_id) else {
            return Ok(Vec::new());
        };
        Ok(repo_ops
            .range(min_id.to_vec()..=max_id.to_vec())
            .take(2)
            .map(|(k, _)| k.clone())
            .collect())
    }

    async fn get_view(&self, repo_id: &str, view_id: &[u8]) -> StoreResult<Option<View>> {
        let views = self.views.lock().unwrap();
        Ok(views.get(repo_id).and_then(|m| m.get(view_id)).cloned())
    }

    async fn put_view(&self, repo_id: &str, view_id: &[u8], view: &View) -> StoreResult<()> {
        let mut views = self.views.lock().unwrap();
        views
            .entry(repo_id.to_string())
            .or_default()
            .insert(view_id.to_vec(), view.clone());
        Ok(())
    }

    async fn get_op_heads(&self, repo_id: &str) -> StoreResult<Option<Vec<OpId>>> {
        let op_heads = self.op_heads.lock().unwrap();
        Ok(op_heads.get(repo_id).cloned())
    }

    async fn update_op_heads(
        &self,
        repo_id: &str,
        old_ids: &[Vec<u8>],
        new_id: &[u8],
    ) -> StoreResult<Vec<OpId>> {
        let mut op_heads = self.op_heads.lock().unwrap();
        let current_heads = op_heads
            .entry(repo_id.to_string())
            .or_insert_with(|| vec![cc_common::ROOT_OPERATION_ID_BYTES.to_vec()]);
        apply_op_heads_mutation(current_heads, old_ids, new_id);
        Ok(current_heads.clone())
    }

    async fn publish_operation_atomic(
        &self,
        repo_id: &str,
        view_id: &[u8],
        view: &View,
        op_id: &[u8],
        op: &Operation,
        old_ids: &[Vec<u8>],
    ) -> StoreResult<Vec<OpId>> {
        {
            let mut views = self.views.lock().unwrap();
            views
                .entry(repo_id.to_string())
                .or_default()
                .insert(view_id.to_vec(), view.clone());
        }
        {
            let mut ops = self.ops.lock().unwrap();
            ops.entry(repo_id.to_string())
                .or_default()
                .insert(op_id.to_vec(), op.clone());
        }
        let mut op_heads = self.op_heads.lock().unwrap();
        let current_heads = op_heads
            .entry(repo_id.to_string())
            .or_insert_with(|| vec![cc_common::ROOT_OPERATION_ID_BYTES.to_vec()]);
        apply_op_heads_mutation(current_heads, old_ids, op_id);
        Ok(current_heads.clone())
    }

    async fn compare_and_swap_op_heads(
        &self,
        repo_id: &str,
        expected_old_ids: &[Vec<u8>],
        new_id: &[u8],
    ) -> StoreResult<()> {
        let mut op_heads = self.op_heads.lock().unwrap();
        let current_heads = op_heads
            .entry(repo_id.to_string())
            .or_insert_with(|| vec![cc_common::ROOT_OPERATION_ID_BYTES.to_vec()]);
        if !op_heads_sets_equal(current_heads, expected_old_ids) {
            return Err(StoreError::CasConflict(format!(
                "op heads changed concurrently in repo '{repo_id}'"
            )));
        }
        *current_heads = vec![new_id.to_vec()];
        Ok(())
    }

    async fn get_workspace(
        &self,
        repo_id: &str,
        user: &str,
        workspace_name: &str,
    ) -> StoreResult<Option<WorkspaceState>> {
        let workspaces = self.workspaces.lock().unwrap();
        Ok(workspaces
            .get(repo_id)
            .and_then(|m| m.get(&(user.to_string(), workspace_name.to_string())))
            .cloned())
    }

    async fn put_workspace(&self, workspace: &WorkspaceState) -> StoreResult<()> {
        let mut workspaces = self.workspaces.lock().unwrap();
        workspaces
            .entry(workspace.repo_id.clone())
            .or_default()
            .insert(
                (workspace.user.clone(), workspace.workspace_name.clone()),
                workspace.clone(),
            );
        Ok(())
    }

    async fn list_workspaces(&self, repo_id: &str) -> StoreResult<Vec<WorkspaceState>> {
        let workspaces = self.workspaces.lock().unwrap();
        Ok(workspaces
            .get(repo_id)
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default())
    }

    async fn delete_workspace(
        &self,
        repo_id: &str,
        user: &str,
        workspace_name: &str,
    ) -> StoreResult<bool> {
        let mut workspaces = self.workspaces.lock().unwrap();
        Ok(workspaces
            .get_mut(repo_id)
            .and_then(|m| m.remove(&(user.to_string(), workspace_name.to_string())))
            .is_some())
    }
}

#[derive(Debug)]
pub struct MemoryBlobStore {
    location: String,
    blobs: Mutex<HashMap<FileId, Bytes>>,
}

impl Default for MemoryBlobStore {
    fn default() -> Self {
        Self::with_location("memory://default")
    }
}

impl MemoryBlobStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_location(location: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            blobs: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl BlobStore for MemoryBlobStore {
    fn location_id(&self) -> &str {
        &self.location
    }

    async fn put_blob(&self, blob_id: &[u8], data: Bytes) -> StoreResult<()> {
        self.blobs
            .lock()
            .unwrap()
            .entry(blob_id.to_vec())
            .or_insert(data);
        Ok(())
    }

    async fn get_blob(&self, blob_id: &[u8]) -> StoreResult<Option<Bytes>> {
        Ok(self.blobs.lock().unwrap().get(blob_id).cloned())
    }

    async fn has_blob(&self, blob_id: &[u8]) -> StoreResult<bool> {
        Ok(self.blobs.lock().unwrap().contains_key(blob_id))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;
    use crate::store::CompositeStore;

    #[test]
    fn test_memory_store_concurrent_blob_writes_and_tree_reads_do_not_deadlock() {
        let store = Arc::new(CompositeStore::in_memory());
        let seed_blob = vec![0xaa; 20];
        futures::executor::block_on(async {
            store.register_repo("repo-stress", None).await.unwrap();
            store
                .put_file("repo-stress", &seed_blob, Bytes::from_static(b"seed"))
                .await
                .unwrap();
        });

        let num_threads = 6;
        let iterations = 400u32;
        let start = Barrier::new(num_threads);

        std::thread::scope(|s| {
            for worker in 0..num_threads {
                let store = Arc::clone(&store);
                let start = &start;
                let seed_blob = seed_blob.clone();
                s.spawn(move || {
                    start.wait();
                    for i in 0..iterations {
                        futures::executor::block_on(async {
                            match worker % 3 {
                                0 => {
                                    // Acquires `project_blobs` then `blobs` in `put_blob_record`.
                                    let mut blob_id = vec![0u8; 20];
                                    blob_id[0] = worker as u8;
                                    blob_id[1..5].copy_from_slice(&i.to_be_bytes());
                                    store
                                        .put_file(
                                            "repo-stress",
                                            &blob_id,
                                            Bytes::from_static(b"payload"),
                                        )
                                        .await
                                        .unwrap();
                                }
                                1 => {
                                    // Acquires `project_blobs` then `blobs` in `get_blob_sizes_batch`.
                                    let mut tree_id = vec![0x10u8; 20];
                                    tree_id[0] = worker as u8;
                                    tree_id[1..5].copy_from_slice(&i.to_be_bytes());
                                    let entry = TreeEntry {
                                        name: "seed.txt".to_string(),
                                        value: Some(TreeValue {
                                            value: Some(tree_value::Value::File(File {
                                                id: seed_blob.clone(),
                                                executable: false,
                                                copy_id: vec![],
                                                size_bytes: None,
                                            })),
                                        }),
                                    };
                                    store
                                        .put_tree("repo-stress", &tree_id, &[entry])
                                        .await
                                        .unwrap();
                                }
                                _ => {
                                    // Acquires `project_blobs` then `blobs` in `get_blob_record`.
                                    let bytes = store
                                        .get_file_range("repo-stress", &seed_blob, 0, None)
                                        .await
                                        .unwrap();
                                    assert_eq!(bytes.as_ref(), b"seed");
                                }
                            }
                        });
                    }
                });
            }
        });
    }
}
