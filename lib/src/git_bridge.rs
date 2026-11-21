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

//! Bidirectional Git history bridge (`GitImporter` and `GitExporter`) preserving 1:1
//! 20-byte Git SHA-1 commit, tree, symlink, and blob digests.
//!
//! # Authority and Crash-Safety Ordering
//! - [`GitImporter`] uploads file blobs (`WriteFileStream` in 64 KiB chunks) and symlinks
//!   (`WriteSymlink`) first, then bottom-up directory trees (`WriteTree`), then commits
//!   (`WriteCommit`) in strict post-order topological order (parents before children), and finally
//!   publishes the updated `View`, `Operation`, and `OpHeads` in one atomic `PublishOperation`
//!   call on top of the current `View`; failing to read that `View` aborts the import. Like
//!   `jj git import`, it reads only `HEAD`, `refs/heads/*`, `refs/remotes/*`, and `refs/tags/*`,
//!   and it never writes to the source repository.
//! - [`GitExporter`] writes referenced blobs, symlinks, trees, and commits into the target
//!   `.git/objects/` store and verifies every written SHA-1 digest before updating `refs/heads/*`
//!   and `HEAD` through gix ref transactions (lock file, then rename). Bookmark names come from the
//!   shared `View`, so names that are not valid Git ref names are skipped and reported.
//!
//! # Concurrency and Merge Invariants
//! File and symlink uploads pipeline up to `CONCURRENT_UPLOADS` (16) streams over the
//! shared pooled HTTP/2 channel; each in-flight file is held in memory whole. Imported commits
//! without an existing `change-id` header get the synthetic `ChangeId` that jj's `GitBackend`
//! derives from the Git SHA-1, and preserve `raw_git_commit` bytes so round-trips
//! back to Git reproduce non-standard Git headers, encodings, and signatures bit-for-bit.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use cc_common::backend::{
    Commit, File, ReadCommitRequest, ReadFileRequest, ReadSymlinkRequest, ReadTreeRequest,
    RegisterRepositoryRequest, SecureSig, Signature, Timestamp, TreeEntry, TreeValue,
    WriteCommitRequest, WriteFileChunk, WriteFileRequest, WriteSymlinkRequest, WriteTreeRequest,
    tree_value,
};
use cc_common::op_store::{
    Operation, OperationMetadata, PublishOperationRequest, ReadOperationRequest, ReadViewRequest,
    ReconcileOpHeadsRequest, RefTarget, RefTargetTerm, RemoteRef,
};
use cc_common::{
    CHANGE_ID_LENGTH, COMMIT_ID_LENGTH, EMPTY_TREE_ID_BYTES, FILE_STREAM_CHUNK_SIZE,
    ROOT_COMMIT_ID_BYTES,
};
use futures::StreamExt as _;
use gix::objs::WriteTo as _;
use gix::prelude::Write as _;
use jj_lib::backend::CommitId;
use jj_lib::git_backend::synthetic_change_id_from_git_commit_id;
use jj_lib::object_id::ObjectId as _;

use crate::util::{AuthBackendClient, connect_backend_client, connect_op_store_client};

const CONCURRENT_UPLOADS: usize = 16;
const EMPTY_STRING_PLACEHOLDER: &str = "JJ_EMPTY_STRING";
const REVERSE_HEX_CHARS: &[u8; 16] = b"zyxwvutsrqponmlk";

#[derive(Clone, Debug)]
pub struct GitImportOptions {
    pub git_dir: PathBuf,
    pub server_url: String,
    pub project_id: Option<String>,
    pub repo_id: Option<String>,
    pub create_repo: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GitImportReport {
    pub repo_id: String,
    pub project_id: String,
    pub commits_imported: usize,
    pub trees_imported: usize,
    pub blobs_imported: usize,
    pub symlinks_imported: usize,
    pub bookmarks_imported: usize,
}

#[derive(Clone, Debug)]
pub struct GitExportOptions {
    pub git_dir: PathBuf,
    pub server_url: String,
    pub repo_id: String,
    pub project_id: Option<String>,
    pub bare: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GitExportReport {
    pub commits_exported: usize,
    pub trees_exported: usize,
    pub blobs_exported: usize,
    pub bookmarks_exported: usize,
    /// Bookmarks left out because their names are not valid Git branch names.
    pub skipped_bookmarks: Vec<String>,
}

pub struct GitImporter {
    opts: GitImportOptions,
}

impl GitImporter {
    pub fn new(opts: GitImportOptions) -> Self {
        Self { opts }
    }

    pub async fn run(&self) -> Result<GitImportReport, Box<dyn std::error::Error + Send + Sync>> {
        let dot_git = resolve_dot_git_dir(&self.opts.git_dir);
        let reftable_objects = ReftableObjectsDir::create_if_reftable(&dot_git)?;
        let open_dir = reftable_objects
            .as_ref()
            .map_or(dot_git.as_path(), |dir| dir.path.as_path());

        let open_opts = gix::open::Options::isolated();
        let thread_safe_repo = gix::ThreadSafeRepository::open_opts(open_dir, open_opts.clone())
            .or_else(|_| gix::ThreadSafeRepository::open_opts(&self.opts.git_dir, open_opts))
            .or_else(|_| gix::ThreadSafeRepository::open(&dot_git))
            .or_else(|_| gix::ThreadSafeRepository::open(&self.opts.git_dir))
            .or_else(|_| gix::ThreadSafeRepository::discover(&self.opts.git_dir))
            .map_err(|e| {
                format!(
                    "Failed to open Git repository at {}: {e}",
                    self.opts.git_dir.display()
                )
            })?;

        let mut backend_client = connect_backend_client(self.opts.server_url.clone()).await?;
        let mut op_client = connect_op_store_client(self.opts.server_url.clone()).await?;

        let should_register = self.opts.create_repo || self.opts.repo_id.is_none();
        let (repo_id, project_id) = if should_register {
            let effective_repo_id = self
                .opts
                .repo_id
                .clone()
                .filter(|id| !id.trim().is_empty())
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            let effective_project_id = self
                .opts
                .project_id
                .clone()
                .filter(|id| !id.trim().is_empty())
                .unwrap_or_else(|| effective_repo_id.clone());
            let res = backend_client
                .register_repository(RegisterRepositoryRequest {
                    project_id: effective_project_id,
                    repo_id: Some(effective_repo_id),
                    name: None,
                    hash_algorithm: cc_common::backend::HashAlgorithm::Sha1 as i32,
                })
                .await?
                .into_inner();
            (res.repo_id, res.project_id)
        } else {
            let existing_repo = self
                .opts
                .repo_id
                .clone()
                .expect("repo_id must be present when create_repo is false");
            op_client
                .reconcile_op_heads(ReconcileOpHeadsRequest {
                    repo_id: existing_repo.clone(),
                })
                .await
                .map_err(|e| {
                    format!(
                        "Target repository '{existing_repo}' is not registered on {}: {e}",
                        self.opts.server_url
                    )
                })?;
            let proj = self
                .opts
                .project_id
                .clone()
                .unwrap_or_else(|| existing_repo.clone());
            (existing_repo, proj)
        };

        let collected = collect_git_graph(
            &thread_safe_repo,
            &dot_git,
            &self.opts.git_dir,
            reftable_objects.is_some(),
        )?;

        // 1. Stream all unique file blobs concurrently in 64 KiB chunks. Each in-flight blob is
        // read into memory whole, so peak memory grows with the largest blobs in flight.
        let blobs_imported = collected.blob_oids.len();
        upload_blobs_streaming(
            &thread_safe_repo,
            &backend_client,
            &project_id,
            collected.blob_oids,
        )
        .await?;

        // 2. Upload all unique symlinks concurrently.
        let symlinks_imported = collected.symlinks.len();
        upload_symlinks(&backend_client, &project_id, collected.symlinks).await?;

        // 3. Upload trees bottom-up and commits topological oldest-first.
        let mut tree_cache: HashSet<gix::hash::ObjectId> = HashSet::new();
        for commit_oid in &collected.topo_commits {
            import_single_commit(
                &thread_safe_repo,
                *commit_oid,
                &mut backend_client,
                &project_id,
                &mut tree_cache,
            )
            .await?;
        }

        // 4. Publish updated View + Operation + OpHeads in one atomic RPC.
        let bookmarks_imported = publish_imported_refs(
            &mut op_client,
            &repo_id,
            &collected.head_commit_ids,
            &collected.local_branches,
            &collected.remote_branches,
        )
        .await?;

        Ok(GitImportReport {
            repo_id,
            project_id,
            commits_imported: collected.topo_commits.len(),
            trees_imported: tree_cache.len(),
            blobs_imported,
            symlinks_imported,
            bookmarks_imported,
        })
    }
}

pub struct GitExporter {
    opts: GitExportOptions,
}

impl GitExporter {
    pub fn new(opts: GitExportOptions) -> Self {
        Self { opts }
    }

    pub async fn run(&self) -> Result<GitExportReport, Box<dyn std::error::Error + Send + Sync>> {
        let mut backend_client = connect_backend_client(self.opts.server_url.clone()).await?;
        let mut op_client = connect_op_store_client(self.opts.server_url.clone()).await?;

        let op_head = op_client
            .reconcile_op_heads(ReconcileOpHeadsRequest {
                repo_id: self.opts.repo_id.clone(),
            })
            .await?
            .into_inner()
            .op_head;

        let op = op_client
            .read_operation(ReadOperationRequest {
                repo_id: self.opts.repo_id.clone(),
                operation_id: op_head,
            })
            .await?
            .into_inner()
            .operation
            .ok_or("server returned empty operation during export-git")?;

        let view = op_client
            .read_view(ReadViewRequest {
                repo_id: self.opts.repo_id.clone(),
                view_id: op.view_id,
            })
            .await?
            .into_inner()
            .view
            .ok_or("server returned empty view during export-git")?;

        let thread_safe_repo = open_or_init_git_repo(&self.opts.git_dir, self.opts.bare)?;

        let object_project_id = self
            .opts
            .project_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(&self.opts.repo_id)
            .to_string();

        // Seed roots from all resolved bookmarks and non-root view heads.
        let wc_set: HashSet<Vec<u8>> = view.wc_commit_ids.values().cloned().collect();
        let mut root_targets: Vec<Vec<u8>> = Vec::new();
        let mut exported_branches: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut skipped_bookmarks: Vec<String> = Vec::new();

        for (bm_name, target) in &view.local_bookmarks {
            if target.removes.is_empty()
                && let Some(first_add) = target.adds.first()
                && first_add.commit_id.as_slice() != ROOT_COMMIT_ID_BYTES
            {
                if git_branch_ref_name(bm_name).is_none() {
                    skipped_bookmarks.push(bm_name.clone());
                    continue;
                }
                exported_branches.insert(bm_name.clone(), first_add.commit_id.clone());
                if !root_targets.contains(&first_add.commit_id) {
                    root_targets.push(first_add.commit_id.clone());
                }
            }
        }
        for head_id in &view.head_ids {
            if head_id.as_slice() == ROOT_COMMIT_ID_BYTES {
                continue;
            }
            // Skip synthetic empty working-copy commits unless explicitly pointed to by a bookmark.
            if wc_set.contains(head_id) && !exported_branches.values().any(|b| b == head_id) {
                let c = backend_client
                    .read_commit(ReadCommitRequest {
                        project_id: object_project_id.clone(),
                        commit_id: head_id.clone(),
                    })
                    .await?
                    .into_inner()
                    .commit;
                if let Some(ref proto_c) = c
                    && proto_c.description.is_empty()
                    && proto_c.root_tree_id.len() == 1
                {
                    let parent_tree = if let Some(first_parent) = proto_c.parent_commit_ids.first()
                    {
                        if first_parent.as_slice() == ROOT_COMMIT_ID_BYTES {
                            Some(EMPTY_TREE_ID_BYTES.to_vec())
                        } else {
                            backend_client
                                .read_commit(ReadCommitRequest {
                                    project_id: object_project_id.clone(),
                                    commit_id: first_parent.clone(),
                                })
                                .await
                                .ok()
                                .and_then(|r| r.into_inner().commit)
                                .and_then(|pc| pc.root_tree_id.first().cloned())
                        }
                    } else {
                        None
                    };
                    if parent_tree.as_ref() == proto_c.root_tree_id.first() {
                        continue;
                    }
                }
            }
            if !root_targets.contains(head_id) {
                root_targets.push(head_id.clone());
            }
        }

        // Fetch reachable commit graph from Commit Cloud topological oldest-first.
        let mut commit_map: HashMap<Vec<u8>, Commit> = HashMap::new();
        let mut topo_commits: Vec<Vec<u8>> = Vec::new();
        let mut state: HashMap<Vec<u8>, u8> = HashMap::new();
        for target in &root_targets {
            let mut stack = vec![(target.clone(), false)];
            while let Some((cid, expanded)) = stack.pop() {
                if cid.as_slice() == ROOT_COMMIT_ID_BYTES {
                    continue;
                }
                if expanded {
                    if state.get(&cid) != Some(&2) {
                        state.insert(cid.clone(), 2);
                        topo_commits.push(cid);
                    }
                    continue;
                }
                if state.contains_key(&cid) {
                    continue;
                }
                state.insert(cid.clone(), 1);
                let commit = backend_client
                    .read_commit(ReadCommitRequest {
                        project_id: object_project_id.clone(),
                        commit_id: cid.clone(),
                    })
                    .await?
                    .into_inner()
                    .commit
                    .ok_or("missing commit payload during export-git")?;
                stack.push((cid.clone(), true));
                for p in commit.parent_commit_ids.iter().rev() {
                    if p.as_slice() != ROOT_COMMIT_ID_BYTES && !state.contains_key(p) {
                        stack.push((p.clone(), false));
                    }
                }
                commit_map.insert(cid, commit);
            }
        }

        let mut written_blobs: HashSet<Vec<u8>> = HashSet::new();
        let mut written_trees: HashSet<Vec<u8>> = HashSet::new();

        for cid in &topo_commits {
            let commit = commit_map.get(cid).expect("commit loaded");
            for tree_id in &commit.root_tree_id {
                export_tree_recursive(
                    &thread_safe_repo,
                    &mut backend_client,
                    &object_project_id,
                    tree_id,
                    &mut written_blobs,
                    &mut written_trees,
                )
                .await?;
            }
            export_single_commit(&thread_safe_repo, cid, commit)?;
        }

        // Write `refs/heads/<bookmark>` and `HEAD`.
        let git_repo = thread_safe_repo.to_thread_local();
        for (branch, cid) in &exported_branches {
            write_git_branch_ref(&git_repo, branch, cid)?;
        }
        if exported_branches.is_empty()
            && let Some(last_head) = topo_commits.last()
        {
            write_git_branch_ref(&git_repo, "main", last_head)?;
            exported_branches.insert("main".to_string(), last_head.clone());
        }
        if let Some(default_branch) = exported_branches
            .contains_key("main")
            .then_some("main")
            .or_else(|| exported_branches.keys().next().map(String::as_str))
        {
            let head_target =
                git_branch_ref_name(default_branch).ok_or("invalid default branch")?;
            write_git_ref(
                &git_repo,
                "HEAD".try_into()?,
                gix::refs::Target::Symbolic(head_target),
            )?;
        }
        skipped_bookmarks.sort();

        Ok(GitExportReport {
            commits_exported: topo_commits.len(),
            trees_exported: written_trees.len(),
            blobs_exported: written_blobs.len(),
            bookmarks_exported: exported_branches.len(),
            skipped_bookmarks,
        })
    }
}

struct CollectedGitGraph {
    head_commit_ids: Vec<Vec<u8>>,
    local_branches: BTreeMap<String, Vec<u8>>,
    remote_branches: BTreeMap<String, Vec<u8>>,
    topo_commits: Vec<gix::hash::ObjectId>,
    blob_oids: Vec<gix::hash::ObjectId>,
    symlinks: Vec<(gix::hash::ObjectId, String)>,
}

fn resolve_dot_git_dir(git_dir: &Path) -> PathBuf {
    if git_dir.file_name() == Some(std::ffi::OsStr::new(".git")) || git_dir.join("HEAD").is_file() {
        git_dir.to_path_buf()
    } else {
        git_dir.join(".git")
    }
}

/// Temporary bare git dir that exposes a reftable repository's objects to gix 0.68, which
/// rejects the `ref: refs/heads/.invalid` HEAD stub that reftable repositories keep for older
/// clients. Objects are borrowed through `objects/info/alternates` and refs are listed with
/// `git for-each-ref` (see [`discover_git_refs`]), so the source repository is never written.
struct ReftableObjectsDir {
    path: PathBuf,
}

impl ReftableObjectsDir {
    fn create_if_reftable(dot_git: &Path) -> std::io::Result<Option<Self>> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let is_reftable = dot_git.join("reftable").is_dir()
            || fs::read_to_string(dot_git.join("HEAD"))
                .is_ok_and(|head| head.trim_end() == "ref: refs/heads/.invalid");
        if !is_reftable {
            return Ok(None);
        }
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let seq = NEXT_ID.fetch_add(1, AtomicOrdering::Relaxed);
        let dir = Self {
            path: std::env::temp_dir().join(format!(
                "jj-cc-import-git-{}-{unique}-{seq}.git",
                std::process::id()
            )),
        };
        fs::create_dir_all(dir.path.join("objects").join("info"))?;
        fs::create_dir(dir.path.join("refs"))?;
        fs::write(dir.path.join("HEAD"), "ref: refs/heads/main\n")?;
        let mut alternates = fs::canonicalize(dot_git.join("objects"))?
            .into_os_string()
            .into_encoded_bytes();
        alternates.push(b'\n');
        fs::write(
            dir.path.join("objects").join("info").join("alternates"),
            alternates,
        )?;
        Ok(Some(dir))
    }
}

impl Drop for ReftableObjectsDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn collect_git_graph(
    thread_safe_repo: &gix::ThreadSafeRepository,
    dot_git: &Path,
    work_dir: &Path,
    is_reftable: bool,
) -> Result<CollectedGitGraph, Box<dyn std::error::Error + Send + Sync>> {
    let repo = thread_safe_repo.to_thread_local();
    let (head_commit_ids, local_branches, remote_branches) =
        discover_git_refs(&repo, dot_git, work_dir, is_reftable);

    // Post-order DFS so every parent strictly precedes all of its descendants, even on diamond
    // DAGs.
    let mut topo_commits = Vec::new();
    let mut visit_state: HashMap<gix::hash::ObjectId, u8> = HashMap::new();
    for head in &head_commit_ids {
        let head_oid = gix::hash::ObjectId::try_from(head.as_slice())?;
        let mut stack = vec![(head_oid, false)];
        while let Some((oid, expanded)) = stack.pop() {
            if expanded {
                if visit_state.get(&oid) != Some(&2) {
                    visit_state.insert(oid, 2);
                    topo_commits.push(oid);
                }
                continue;
            }
            if visit_state.contains_key(&oid) {
                continue;
            }
            visit_state.insert(oid, 1);
            let obj = repo.find_object(oid)?;
            let commit_ref = obj.to_commit_ref();
            stack.push((oid, true));
            let parents: Vec<gix::hash::ObjectId> = commit_ref.parents().collect();
            for p in parents.into_iter().rev() {
                if !visit_state.contains_key(&p) {
                    stack.push((p, false));
                }
            }
        }
    }

    let mut seen_blobs = HashSet::new();
    let mut seen_symlinks = HashSet::new();
    let mut seen_trees = HashSet::new();
    let mut blob_oids = Vec::new();
    let mut symlinks = Vec::new();

    for commit_oid in &topo_commits {
        let obj = repo.find_object(*commit_oid)?;
        let commit_ref = obj.to_commit_ref();
        let mut tree_stack = vec![commit_ref.tree()];
        while let Some(tree_oid) = tree_stack.pop() {
            if !seen_trees.insert(tree_oid) {
                continue;
            }
            let tree_obj = repo.find_object(tree_oid)?;
            let tree_ref = gix::objs::TreeRef::from_bytes(&tree_obj.data, gix::hash::Kind::Sha1)
                .map_err(|e| format!("Failed to parse Git tree {tree_oid}: {e}"))?;
            for entry in tree_ref.entries {
                let entry_oid = entry.oid.to_owned();
                if entry.mode.is_tree() {
                    tree_stack.push(entry_oid);
                } else if entry.mode.is_link() {
                    if seen_symlinks.insert(entry_oid) {
                        let link_obj = repo.find_object(entry_oid)?;
                        let target = std::str::from_utf8(&link_obj.data)
                            .map_err(|e| format!("Non-UTF8 Git symlink {entry_oid}: {e}"))?
                            .to_string();
                        symlinks.push((entry_oid, target));
                    }
                } else if entry.mode.is_blob() && seen_blobs.insert(entry_oid) {
                    blob_oids.push(entry_oid);
                }
            }
        }
    }

    Ok(CollectedGitGraph {
        head_commit_ids,
        local_branches,
        remote_branches,
        topo_commits,
        blob_oids,
        symlinks,
    })
}

fn discover_git_refs(
    repo: &gix::Repository,
    dot_git: &Path,
    work_dir: &Path,
    is_reftable: bool,
) -> (
    Vec<Vec<u8>>,
    BTreeMap<String, Vec<u8>>,
    BTreeMap<String, Vec<u8>>,
) {
    let mut heads: Vec<Vec<u8>> = Vec::new();
    let mut local_branches: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut remote_branches: BTreeMap<String, Vec<u8>> = BTreeMap::new();

    if !is_reftable {
        if let Ok(platform) = repo.references()
            && let Ok(iter) = platform.all()
        {
            for mut r in iter.flatten() {
                let full_name = r.name().as_bstr().to_string();
                if let Ok(id) = r.peel_to_id() {
                    register_git_ref(
                        repo,
                        &full_name,
                        id.detach(),
                        &mut heads,
                        &mut local_branches,
                        &mut remote_branches,
                    );
                }
            }
        }

        if let Ok(head_commit) = repo.head_commit() {
            let bytes = head_commit.id.as_bytes().to_vec();
            if !heads.contains(&bytes) {
                heads.push(bytes);
            }
        }
    } else {
        // gix 0.68 only reads the files ref backend, so reftable repositories query `git`
        // read-only for refs and `HEAD` while reading objects through `ReftableObjectsDir`.
        let cwd = if work_dir.exists() { work_dir } else { dot_git };
        if let Ok(out) = std::process::Command::new("git")
            .arg("--git-dir")
            .arg(dot_git)
            .args(["for-each-ref", "--format=%(refname) %(objectname)"])
            .current_dir(cwd)
            .output()
            && out.status.success()
        {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                if let Some((name, hex_str)) = line.trim().split_once(' ')
                    && let Ok(oid) = gix::hash::ObjectId::from_hex(hex_str.as_bytes())
                {
                    register_git_ref(
                        repo,
                        name,
                        oid,
                        &mut heads,
                        &mut local_branches,
                        &mut remote_branches,
                    );
                }
            }
        }
        if let Ok(out) = std::process::Command::new("git")
            .arg("--git-dir")
            .arg(dot_git)
            .args(["rev-parse", "--verify", "--quiet", "HEAD^{commit}"])
            .current_dir(cwd)
            .output()
            && out.status.success()
            && let Ok(oid) = gix::hash::ObjectId::from_hex(out.stdout.trim_ascii())
        {
            let bytes = oid.as_bytes().to_vec();
            if !heads.contains(&bytes) {
                heads.push(bytes);
            }
        }
    }

    (heads, local_branches, remote_branches)
}

fn register_git_ref(
    repo: &gix::Repository,
    full_name: &str,
    oid: gix::hash::ObjectId,
    heads: &mut Vec<Vec<u8>>,
    local_branches: &mut BTreeMap<String, Vec<u8>>,
    remote_branches: &mut BTreeMap<String, Vec<u8>>,
) {
    if !is_importable_git_ref(full_name) {
        return;
    }
    if let Some(commit) = repo
        .find_object(oid)
        .ok()
        .and_then(|o| o.peel_to_commit().ok())
    {
        let bytes = commit.id.as_bytes().to_vec();
        if !heads.contains(&bytes) {
            heads.push(bytes.clone());
        }
        if let Some(branch) = full_name.strip_prefix("refs/heads/")
            && !branch.is_empty()
            && branch != ".invalid"
        {
            local_branches.insert(branch.to_string(), bytes);
        } else if let Some(rem) = full_name.strip_prefix("refs/remotes/")
            && let Some((remote_name, branch_name)) = rem.split_once('/')
            && branch_name != "HEAD"
        {
            remote_branches.insert(format!("{branch_name}@{remote_name}"), bytes);
        }
    }
}

/// Mirrors `jj git import`: only branches, remote-tracking branches (except `<remote>/HEAD`), and
/// tags make commits visible. `refs/jj/keep/*`, `refs/stash`, `refs/notes/*`, `refs/pull/*`, and
/// other namespaces hold commits that jj does not show. Tags only keep their commits visible
/// because the cloud `View` has no tags yet.
fn is_importable_git_ref(full_name: &str) -> bool {
    if let Some(remote_ref) = full_name.strip_prefix("refs/remotes/") {
        return remote_ref.split_once('/').is_some_and(|(remote, branch)| {
            !remote.is_empty() && remote != "git" && !branch.is_empty() && branch != "HEAD"
        });
    }
    full_name
        .strip_prefix("refs/heads/")
        .is_some_and(|b| !b.is_empty() && b != ".invalid")
        || full_name
            .strip_prefix("refs/tags/")
            .is_some_and(|t| !t.is_empty())
}

async fn upload_blobs_streaming(
    thread_safe_repo: &gix::ThreadSafeRepository,
    backend_client: &AuthBackendClient,
    project_id: &str,
    blob_oids: Vec<gix::hash::ObjectId>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut stream = futures::stream::iter(blob_oids.into_iter().map(|oid| {
        let mut client = backend_client.clone();
        let p_id = project_id.to_string();
        let ts_repo = thread_safe_repo.clone();
        async move {
            let data = {
                let repo = ts_repo.to_thread_local();
                let obj = repo.find_object(oid)?;
                obj.data.to_vec()
            };
            let mut frames = Vec::new();
            if data.is_empty() {
                frames.push(WriteFileChunk {
                    project_id: p_id.clone(),
                    chunk: Vec::new(),
                });
            } else {
                for (idx, slice) in data.chunks(FILE_STREAM_CHUNK_SIZE).enumerate() {
                    frames.push(WriteFileChunk {
                        project_id: if idx == 0 {
                            p_id.clone()
                        } else {
                            String::new()
                        },
                        chunk: slice.to_vec(),
                    });
                }
            }
            let res = match client
                .write_file_stream(futures::stream::iter(frames))
                .await
            {
                Ok(r) => r.into_inner(),
                Err(status) if status.code() == tonic::Code::Unimplemented => client
                    .write_file(WriteFileRequest {
                        project_id: p_id,
                        content: data,
                    })
                    .await?
                    .into_inner(),
                Err(status) => return Err(status.into()),
            };
            if res.file_id.as_slice() != oid.as_bytes() {
                return Err::<(), Box<dyn std::error::Error + Send + Sync>>(
                    format!(
                        "Git blob SHA-1 mismatch: expected {oid}, server computed {}",
                        hex::encode(&res.file_id)
                    )
                    .into(),
                );
            }
            Ok(())
        }
    }))
    .buffer_unordered(CONCURRENT_UPLOADS);

    while let Some(res) = stream.next().await {
        res?;
    }
    Ok(())
}

async fn upload_symlinks(
    backend_client: &AuthBackendClient,
    project_id: &str,
    symlinks: Vec<(gix::hash::ObjectId, String)>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut stream = futures::stream::iter(symlinks.into_iter().map(|(oid, target)| {
        let mut client = backend_client.clone();
        let p_id = project_id.to_string();
        async move {
            let res = client
                .write_symlink(WriteSymlinkRequest {
                    project_id: p_id,
                    target,
                })
                .await?
                .into_inner();
            if res.symlink_id.as_slice() != oid.as_bytes() {
                return Err::<(), Box<dyn std::error::Error + Send + Sync>>(
                    format!(
                        "Git symlink SHA-1 mismatch: expected {oid}, server computed {}",
                        hex::encode(&res.symlink_id)
                    )
                    .into(),
                );
            }
            Ok(())
        }
    }))
    .buffer_unordered(CONCURRENT_UPLOADS);

    while let Some(res) = stream.next().await {
        res?;
    }
    Ok(())
}

async fn import_tree_recursive(
    thread_safe_repo: &gix::ThreadSafeRepository,
    tree_oid: gix::hash::ObjectId,
    backend_client: &mut AuthBackendClient,
    project_id: &str,
    tree_cache: &mut HashSet<gix::hash::ObjectId>,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    if tree_oid.as_bytes() == EMPTY_TREE_ID_BYTES {
        return Ok(EMPTY_TREE_ID_BYTES.to_vec());
    }
    if tree_cache.contains(&tree_oid) {
        return Ok(tree_oid.as_bytes().to_vec());
    }

    let raw_entries = {
        let repo = thread_safe_repo.to_thread_local();
        let tree_obj = repo.find_object(tree_oid)?;
        let tree_ref = gix::objs::TreeRef::from_bytes(&tree_obj.data, gix::hash::Kind::Sha1)
            .map_err(|e| format!("Failed to parse Git tree {tree_oid}: {e}"))?;
        tree_ref
            .entries
            .into_iter()
            .map(|e| {
                (
                    e.filename.to_string(),
                    e.oid.to_owned(),
                    e.mode.is_tree(),
                    e.mode.is_link(),
                    e.mode.is_commit(),
                    e.mode.is_executable(),
                )
            })
            .collect::<Vec<_>>()
    };

    let mut proto_entries = Vec::with_capacity(raw_entries.len());
    for (name, child_oid, is_tree, is_link, is_submodule, is_exec) in raw_entries {
        let val = if is_tree {
            let child_bytes = Box::pin(import_tree_recursive(
                thread_safe_repo,
                child_oid,
                backend_client,
                project_id,
                tree_cache,
            ))
            .await?;
            tree_value::Value::TreeId(child_bytes)
        } else if is_link {
            tree_value::Value::SymlinkId(child_oid.as_bytes().to_vec())
        } else if is_submodule {
            tree_value::Value::GitSubmoduleId(child_oid.as_bytes().to_vec())
        } else {
            tree_value::Value::File(File {
                id: child_oid.as_bytes().to_vec(),
                executable: is_exec,
                copy_id: Vec::new(),
                size_bytes: None,
            })
        };
        proto_entries.push(TreeEntry {
            name,
            value: Some(TreeValue { value: Some(val) }),
        });
    }

    let res = backend_client
        .write_tree(WriteTreeRequest {
            project_id: project_id.to_string(),
            path: String::new(),
            entries: proto_entries,
        })
        .await?
        .into_inner();

    if res.tree_id.as_slice() != tree_oid.as_bytes() {
        return Err(format!(
            "Git tree SHA-1 mismatch: expected {tree_oid}, server computed {}",
            hex::encode(&res.tree_id)
        )
        .into());
    }
    tree_cache.insert(tree_oid);
    Ok(res.tree_id)
}

async fn import_single_commit(
    thread_safe_repo: &gix::ThreadSafeRepository,
    commit_oid: gix::hash::ObjectId,
    backend_client: &mut AuthBackendClient,
    project_id: &str,
    tree_cache: &mut HashSet<gix::hash::ObjectId>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (
        raw_commit_bytes,
        tree_oid,
        mut parent_ids,
        author,
        committer,
        message,
        parsed_change_id,
        extra_tree_ids,
        conflict_labels,
        secure_sig,
    ) = {
        let repo = thread_safe_repo.to_thread_local();
        let object = repo.find_object(commit_oid)?;
        let raw_bytes = object.data.to_vec();
        let commit_ref = object.to_commit_ref();

        let tree_oid = commit_ref.tree();
        let author_sig = commit_ref
            .author()
            .map_err(|e| format!("Invalid Git author in {commit_oid}: {e}"))?;
        let author_time = author_sig
            .time()
            .map_err(|e| format!("Invalid Git author timestamp in {commit_oid}: {e}"))?;
        let committer_sig = commit_ref
            .committer()
            .map_err(|e| format!("Invalid Git committer in {commit_oid}: {e}"))?;
        let committer_time = committer_sig
            .time()
            .map_err(|e| format!("Invalid Git committer timestamp in {commit_oid}: {e}"))?;
        let parent_ids: Vec<Vec<u8>> = commit_ref
            .parents()
            .map(|p| p.as_bytes().to_vec())
            .collect();

        let mut parsed_change_id: Option<Vec<u8>> = None;
        let mut extra_tree_ids: Vec<gix::hash::ObjectId> = Vec::new();
        let mut conflict_labels: Vec<String> = Vec::new();
        let mut secure_sig: Option<SecureSig> = None;

        for (k, v) in &commit_ref.extra_headers {
            let key_bytes: &[u8] = k.as_ref();
            let val_bytes: &[u8] = v.as_ref();
            match key_bytes {
                b"change-id" => {
                    if let Ok(s) = std::str::from_utf8(val_bytes) {
                        parsed_change_id = decode_change_id_header(s.trim());
                    }
                }
                b"jj:trees" => {
                    if let Ok(s) = std::str::from_utf8(val_bytes) {
                        for hex_tok in s.split_whitespace() {
                            if let Ok(tid) = gix::hash::ObjectId::from_hex(hex_tok.as_bytes()) {
                                extra_tree_ids.push(tid);
                            }
                        }
                    }
                }
                b"jj:conflict-labels" => {
                    if let Ok(s) = std::str::from_utf8(val_bytes) {
                        let trimmed = s.strip_suffix('\n').unwrap_or(s);
                        conflict_labels = trimmed.split('\n').map(ToOwned::to_owned).collect();
                    }
                }
                b"gpgsig" => {
                    secure_sig = Some(SecureSig {
                        data: Vec::new(),
                        sig: val_bytes.to_vec(),
                    });
                }
                _ => {}
            }
        }

        (
            raw_bytes,
            tree_oid,
            parent_ids,
            Signature {
                name: author_sig.name.to_string(),
                email: author_sig.email.to_string(),
                timestamp: Some(Timestamp {
                    millis_since_epoch: author_time.seconds * 1000,
                    tz_offset: author_time.offset / 60,
                }),
            },
            Signature {
                name: committer_sig.name.to_string(),
                email: committer_sig.email.to_string(),
                timestamp: Some(Timestamp {
                    millis_since_epoch: committer_time.seconds * 1000,
                    tz_offset: committer_time.offset / 60,
                }),
            },
            commit_ref.message.to_string(),
            parsed_change_id,
            extra_tree_ids,
            conflict_labels,
            secure_sig,
        )
    };

    if parent_ids.is_empty() {
        parent_ids.push(ROOT_COMMIT_ID_BYTES.to_vec());
    }

    let root_tree_ids = if extra_tree_ids.len() > 1 {
        let mut ids = Vec::with_capacity(extra_tree_ids.len());
        for tid in extra_tree_ids {
            ids.push(
                import_tree_recursive(
                    thread_safe_repo,
                    tid,
                    backend_client,
                    project_id,
                    tree_cache,
                )
                .await?,
            );
        }
        ids
    } else {
        vec![
            import_tree_recursive(
                thread_safe_repo,
                tree_oid,
                backend_client,
                project_id,
                tree_cache,
            )
            .await?,
        ]
    };

    let change_id = parsed_change_id.unwrap_or_else(|| {
        synthetic_change_id_from_git_commit_id(&CommitId::from_bytes(commit_oid.as_bytes()))
            .to_bytes()
    });

    let mut external_ids = HashMap::new();
    external_ids.insert("git-sha1".to_string(), commit_oid.as_bytes().to_vec());

    let proto_commit = Commit {
        commit_id: commit_oid.as_bytes().to_vec(),
        change_id,
        parent_commit_ids: parent_ids,
        root_tree_id: root_tree_ids,
        description: message,
        author: Some(author),
        committer: Some(committer),
        predecessors: Vec::new(),
        conflict_labels,
        secure_sig,
        raw_git_commit: raw_commit_bytes,
        external_ids,
    };

    let res = backend_client
        .write_commit(WriteCommitRequest {
            project_id: project_id.to_string(),
            commit: Some(proto_commit),
        })
        .await?
        .into_inner();
    if res.commit_id.as_slice() != commit_oid.as_bytes() {
        return Err(format!(
            "Git commit SHA-1 mismatch: expected {commit_oid}, server returned {}",
            hex::encode(&res.commit_id)
        )
        .into());
    }
    Ok(())
}

async fn publish_imported_refs(
    op_client: &mut crate::util::AuthOpStoreClient,
    repo_id: &str,
    head_commit_ids: &[Vec<u8>],
    local_branches: &BTreeMap<String, Vec<u8>>,
    remote_branches: &BTreeMap<String, Vec<u8>>,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let reconciled_head = op_client
        .reconcile_op_heads(ReconcileOpHeadsRequest {
            repo_id: repo_id.to_string(),
        })
        .await?
        .into_inner()
        .op_head;

    let op = op_client
        .read_operation(ReadOperationRequest {
            repo_id: repo_id.to_string(),
            operation_id: reconciled_head.clone(),
        })
        .await?
        .into_inner()
        .operation
        .ok_or("server returned empty operation during import-git")?;
    let mut view = op_client
        .read_view(ReadViewRequest {
            repo_id: repo_id.to_string(),
            view_id: op.view_id,
        })
        .await?
        .into_inner()
        .view
        .ok_or("server returned empty view during import-git")?;

    if !head_commit_ids.is_empty() {
        view.head_ids.retain(|h| {
            h.as_slice() != ROOT_COMMIT_ID_BYTES || view.wc_commit_ids.values().any(|wc| wc == h)
        });
    }
    for head in head_commit_ids {
        if !view.head_ids.contains(head) {
            view.head_ids.push(head.clone());
        }
    }

    let mut bookmark_count = 0usize;
    for (name, cid) in local_branches {
        view.local_bookmarks.insert(
            name.clone(),
            RefTarget {
                removes: Vec::new(),
                adds: vec![RefTargetTerm {
                    commit_id: cid.clone(),
                }],
            },
        );
        bookmark_count += 1;
    }
    for (remote_key, cid) in remote_branches {
        // Like `jj git import`, new remote bookmarks start untracked and existing ones keep their
        // tracking state.
        let is_tracked = view
            .remote_bookmarks
            .get(remote_key)
            .is_some_and(|r| r.is_tracked);
        view.remote_bookmarks.insert(
            remote_key.clone(),
            RemoteRef {
                target: Some(RefTarget {
                    removes: Vec::new(),
                    adds: vec![RefTargetTerm {
                        commit_id: cid.clone(),
                    }],
                }),
                is_tracked,
            },
        );
    }

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let operation = Operation {
        view_id: Vec::new(),
        parents: vec![reconciled_head.clone()],
        metadata: Some(OperationMetadata {
            start_time_millis: now_ms,
            end_time_millis: now_ms,
            description: "import git refs and commits".to_string(),
            is_snapshot: false,
            workspace_name: None,
            hostname: "localhost".to_string(),
            username: "git-bridge".to_string(),
            attributes: HashMap::new(),
        }),
        commit_predecessors: Vec::new(),
        commit_predecessors_set: false,
    };

    op_client
        .publish_operation(PublishOperationRequest {
            repo_id: repo_id.to_string(),
            view: Some(view),
            operation: Some(operation),
            old_op_head_ids: vec![reconciled_head],
        })
        .await?;

    Ok(bookmark_count)
}

async fn export_tree_recursive(
    thread_safe_repo: &gix::ThreadSafeRepository,
    backend_client: &mut AuthBackendClient,
    project_id: &str,
    tree_id: &[u8],
    written_blobs: &mut HashSet<Vec<u8>>,
    written_trees: &mut HashSet<Vec<u8>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if tree_id == EMPTY_TREE_ID_BYTES || written_trees.contains(tree_id) {
        return Ok(());
    }

    let entries = backend_client
        .read_tree(ReadTreeRequest {
            project_id: project_id.to_string(),
            tree_id: tree_id.to_vec(),
            path: String::new(),
            page_size: 0,
            page_token: String::new(),
        })
        .await?
        .into_inner()
        .entries;

    let mut gix_entries = Vec::with_capacity(entries.len());
    for entry in entries {
        let Some(val) = entry.value.as_ref().and_then(|v| v.value.as_ref()) else {
            continue;
        };
        let (kind, oid_bytes) = match val {
            tree_value::Value::File(f) => {
                if !written_blobs.contains(&f.id) {
                    let mut stream = backend_client
                        .read_file(ReadFileRequest {
                            project_id: project_id.to_string(),
                            file_id: f.id.clone(),
                            offset: 0,
                            max_bytes: 0,
                        })
                        .await?
                        .into_inner();
                    let mut bytes = Vec::new();
                    while let Some(chunk) = stream.next().await {
                        bytes.extend_from_slice(&chunk?.chunk);
                    }
                    let written = {
                        let git_repo = thread_safe_repo.to_thread_local();
                        git_repo.objects.write_buf(gix::objs::Kind::Blob, &bytes)?
                    };
                    if written.as_bytes() != f.id.as_slice() {
                        return Err(format!(
                            "Exported Git blob SHA-1 mismatch: expected {}, wrote {written}",
                            hex::encode(&f.id)
                        )
                        .into());
                    }
                    written_blobs.insert(f.id.clone());
                }
                let mode = if f.executable {
                    gix::objs::tree::EntryKind::BlobExecutable
                } else {
                    gix::objs::tree::EntryKind::Blob
                };
                (mode, f.id.clone())
            }
            tree_value::Value::SymlinkId(sym_id) => {
                if !written_blobs.contains(sym_id) {
                    let target = backend_client
                        .read_symlink(ReadSymlinkRequest {
                            project_id: project_id.to_string(),
                            symlink_id: sym_id.clone(),
                        })
                        .await?
                        .into_inner()
                        .target;
                    let written = {
                        let git_repo = thread_safe_repo.to_thread_local();
                        git_repo
                            .objects
                            .write_buf(gix::objs::Kind::Blob, target.as_bytes())?
                    };
                    if written.as_bytes() != sym_id.as_slice() {
                        return Err(format!(
                            "Exported Git symlink SHA-1 mismatch: expected {}, wrote {written}",
                            hex::encode(sym_id)
                        )
                        .into());
                    }
                    written_blobs.insert(sym_id.clone());
                }
                (gix::objs::tree::EntryKind::Link, sym_id.clone())
            }
            tree_value::Value::TreeId(sub_id) => {
                Box::pin(export_tree_recursive(
                    thread_safe_repo,
                    backend_client,
                    project_id,
                    sub_id,
                    written_blobs,
                    written_trees,
                ))
                .await?;
                (gix::objs::tree::EntryKind::Tree, sub_id.clone())
            }
            tree_value::Value::GitSubmoduleId(submod_cid) => {
                (gix::objs::tree::EntryKind::Commit, submod_cid.clone())
            }
            tree_value::Value::ConflictId(_) => {
                return Err("Cannot export unresolved conflict entry to native Git tree".into());
            }
        };

        let oid = gix::hash::ObjectId::try_from(oid_bytes.as_slice())?;
        gix_entries.push(gix::objs::tree::Entry {
            mode: kind.into(),
            filename: entry.name.as_bytes().into(),
            oid,
        });
    }

    gix_entries.sort_unstable();
    let gix_tree = gix::objs::Tree {
        entries: gix_entries,
    };
    let mut buf = Vec::new();
    gix_tree.write_to(&mut buf)?;
    let written_tree = {
        let git_repo = thread_safe_repo.to_thread_local();
        git_repo.objects.write_buf(gix::objs::Kind::Tree, &buf)?
    };
    if written_tree.as_bytes() != tree_id {
        return Err(format!(
            "Exported Git tree SHA-1 mismatch: expected {}, wrote {written_tree}",
            hex::encode(tree_id)
        )
        .into());
    }
    written_trees.insert(tree_id.to_vec());
    Ok(())
}

fn export_single_commit(
    thread_safe_repo: &gix::ThreadSafeRepository,
    expected_id: &[u8],
    commit: &Commit,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let git_repo = thread_safe_repo.to_thread_local();
    if !commit.raw_git_commit.is_empty() {
        let digest = gix::objs::compute_hash(
            gix::hash::Kind::Sha1,
            gix::objs::Kind::Commit,
            &commit.raw_git_commit,
        )?;
        if digest.as_bytes() == expected_id {
            let written = git_repo
                .objects
                .write_buf(gix::objs::Kind::Commit, &commit.raw_git_commit)?;
            debug_assert_eq!(written.as_bytes(), expected_id);
            return Ok(());
        }
    }

    let with_change_id = serialize_proto_commit_to_git(commit, true)?;
    let hash_with = gix::objs::compute_hash(
        gix::hash::Kind::Sha1,
        gix::objs::Kind::Commit,
        &with_change_id,
    )?;
    let chosen_buf = if hash_with.as_bytes() == expected_id {
        with_change_id
    } else {
        let without_change_id = serialize_proto_commit_to_git(commit, false)?;
        let hash_without = gix::objs::compute_hash(
            gix::hash::Kind::Sha1,
            gix::objs::Kind::Commit,
            &without_change_id,
        )?;
        if hash_without.as_bytes() == expected_id {
            without_change_id
        } else {
            return Err(format!(
                "Exported Git commit SHA-1 mismatch for {}: neither with-change-id ({hash_with}) nor native ({hash_without}) matched",
                hex::encode(expected_id)
            )
            .into());
        }
    };

    let written = git_repo
        .objects
        .write_buf(gix::objs::Kind::Commit, &chosen_buf)?;
    if written.as_bytes() != expected_id {
        return Err(format!(
            "Exported Git commit SHA-1 mismatch: expected {}, wrote {written}",
            hex::encode(expected_id)
        )
        .into());
    }
    Ok(())
}

fn serialize_proto_commit_to_git(
    commit: &Commit,
    include_change_id: bool,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let tree_id = commit
        .root_tree_id
        .first()
        .and_then(|id| gix::hash::ObjectId::try_from(id.as_slice()).ok())
        .unwrap_or_else(|| gix::hash::ObjectId::empty_tree(gix::hash::Kind::Sha1));

    let parents: Vec<gix::hash::ObjectId> = commit
        .parent_commit_ids
        .iter()
        .filter(|id| id.as_slice() != ROOT_COMMIT_ID_BYTES)
        .filter_map(|id| gix::hash::ObjectId::try_from(id.as_slice()).ok())
        .collect();

    let mut extra_headers = Vec::new();
    if commit.conflict_labels.len() > 1 {
        let mut joined = commit.conflict_labels.join("\n");
        joined.push('\n');
        extra_headers.push(("jj:conflict-labels".into(), joined.into()));
    }
    if commit.root_tree_id.len() > 1 {
        let trees_hex: Vec<String> = commit.root_tree_id.iter().map(hex::encode).collect();
        extra_headers.push(("jj:trees".into(), trees_hex.join(" ").into()));
    }
    if include_change_id && !commit.change_id.is_empty() {
        extra_headers.push((
            "change-id".into(),
            encode_reverse_hex(&commit.change_id).into(),
        ));
    }
    if let Some(sig) = &commit.secure_sig
        && !sig.sig.is_empty()
    {
        extra_headers.push(("gpgsig".into(), sig.sig.clone().into()));
    }

    let gix_commit = gix::objs::Commit {
        tree: tree_id,
        parents: parents.into(),
        author: signature_proto_to_gix(commit.author.as_ref()),
        committer: signature_proto_to_gix(commit.committer.as_ref()),
        encoding: None,
        extra_headers,
        message: commit.description.as_bytes().into(),
    };

    let mut buf = Vec::new();
    gix_commit.write_to(&mut buf)?;
    Ok(buf)
}

fn signature_proto_to_gix(sig: Option<&Signature>) -> gix::actor::Signature {
    let (name, email) = match sig {
        Some(s) => (
            if s.name.is_empty() {
                EMPTY_STRING_PLACEHOLDER
            } else {
                &s.name
            },
            if s.email.is_empty() {
                EMPTY_STRING_PLACEHOLDER
            } else {
                &s.email
            },
        ),
        None => (EMPTY_STRING_PLACEHOLDER, EMPTY_STRING_PLACEHOLDER),
    };
    let (secs, offset_mins) = sig.and_then(|s| s.timestamp.as_ref()).map_or((0, 0), |t| {
        (t.millis_since_epoch.div_euclid(1000), t.tz_offset)
    });

    gix::actor::Signature {
        name: name.into(),
        email: email.into(),
        time: gix::date::Time::new(secs, offset_mins * 60),
    }
}

fn open_or_init_git_repo(
    git_dir: &Path,
    bare: bool,
) -> Result<gix::ThreadSafeRepository, Box<dyn std::error::Error + Send + Sync>> {
    if git_dir.exists()
        && (git_dir.join(".git").exists() || git_dir.join("HEAD").exists())
        && let Ok(existing) = gix::open(git_dir)
    {
        return Ok(existing.into_sync());
    }
    fs::create_dir_all(git_dir)?;
    let repo = if bare {
        gix::init_bare(git_dir)?
    } else {
        gix::init(git_dir)?
    };
    Ok(repo.into_sync())
}

fn write_git_branch_ref(
    repo: &gix::Repository,
    branch: &str,
    commit_id: &[u8],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if commit_id.len() != COMMIT_ID_LENGTH {
        return Err(format!("invalid commit ID length {}", commit_id.len()).into());
    }
    let name = git_branch_ref_name(branch)
        .ok_or_else(|| format!("bookmark {branch:?} is not a valid Git branch name"))?;
    let target = gix::refs::Target::Object(gix::hash::ObjectId::try_from(commit_id)?);
    write_git_ref(repo, name, target)
}

/// Returns `refs/heads/<name>` if `name` is a valid Git branch name. Bookmark names in the `View`
/// are arbitrary strings, so this check keeps names such as `../config` or `/home/user/file` from
/// reaching the ref store.
fn git_branch_ref_name(name: &str) -> Option<gix::refs::FullName> {
    gix::refs::FullName::try_from(format!("refs/heads/{name}")).ok()
}

/// Updates one ref through a gix ref transaction, which writes a lock file and renames it.
fn write_git_ref(
    repo: &gix::Repository,
    name: gix::refs::FullName,
    new: gix::refs::Target,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use gix::refs::transaction::{Change, LogChange, PreviousValue, RefEdit, RefLog};

    let edit = RefEdit {
        change: Change::Update {
            log: LogChange {
                mode: RefLog::AndReference,
                force_create_reflog: false,
                message: "jj cc export-git".into(),
            },
            expected: PreviousValue::Any,
            new,
        },
        name,
        deref: false,
    };
    // Reflog entries need a committer, and Git may have no identity configured.
    let mut fallback_time_buf = gix::date::parse::TimeBuf::default();
    let fallback_time_str = gix::date::Time::now_utc().to_str(&mut fallback_time_buf);
    let committer = repo
        .committer()
        .transpose()?
        .unwrap_or(gix::actor::SignatureRef {
            name: "jj cc export-git".into(),
            email: "".into(),
            time: fallback_time_str,
        });
    let fail_mode = gix::lock::acquire::Fail::Immediately;
    repo.refs
        .transaction()
        .prepare([edit], fail_mode, fail_mode)?
        .commit(Some(committer))?;
    Ok(())
}

fn encode_reverse_hex(data: &[u8]) -> String {
    let encoded: Vec<u8> = data
        .iter()
        .flat_map(|b| {
            [
                REVERSE_HEX_CHARS[usize::from(b >> 4)],
                REVERSE_HEX_CHARS[usize::from(b & 0x0f)],
            ]
        })
        .collect();
    String::from_utf8(encoded).expect("ASCII z-k digits must be valid UTF-8")
}

fn decode_change_id_header(raw: &str) -> Option<Vec<u8>> {
    if raw.len() != CHANGE_ID_LENGTH * 2 {
        return None;
    }
    if let Ok(std_hex) = hex::decode(raw) {
        return Some(std_hex);
    }
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(CHANGE_ID_LENGTH);
    for pair in bytes.as_chunks::<2>().0 {
        let hi = REVERSE_HEX_CHARS.iter().position(|&c| c == pair[0])? as u8;
        let lo = REVERSE_HEX_CHARS.iter().position(|&c| c == pair[1])? as u8;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use cc_common::op_store::op_store_service_server::{OpStoreService, OpStoreServiceServer};
    use cc_common::op_store::{
        GetOpHeadsRequest, GetOpHeadsResponse, PublishOperationResponse, ReadOperationResponse,
        ReadViewResponse, ReconcileOpHeadsResponse, ResolveOperationPrefixRequest,
        ResolveOperationPrefixResponse, UpdateOpHeadsRequest, UpdateOpHeadsResponse,
        WriteOperationRequest, WriteOperationResponse, WriteViewRequest, WriteViewResponse,
    };
    use tonic::{Request, Response, Status};

    use super::*;
    use crate::util::run_on_tokio;

    #[test]
    fn git_branch_ref_name_rejects_paths_and_invalid_git_names() {
        for name in ["main", "feature/auth", "release-1.2", "user@host"] {
            assert!(
                git_branch_ref_name(name).is_some(),
                "{name:?} should be exported"
            );
        }
        for name in [
            "../x",
            "../../config",
            "a/../../b",
            "/home/alice/.ssh/authorized_keys",
            "a//b",
            "trailing/",
            ".hidden",
            "has space",
            "tilde~1",
            "caret^",
            "colon:name",
            "question?",
            "star*",
            "bracket[",
            "back\\slash",
            "ends.lock",
            "ends.",
            "at@{1}",
            "",
        ] {
            assert!(
                git_branch_ref_name(name).is_none(),
                "{name:?} should be skipped"
            );
        }
    }

    #[test]
    fn importable_git_refs_match_jj_git_import() {
        for name in [
            "refs/heads/main",
            "refs/heads/feature/auth",
            "refs/remotes/origin/main",
            "refs/remotes/origin/feature/auth",
            "refs/tags/v1.0",
        ] {
            assert!(is_importable_git_ref(name), "{name} should be imported");
        }
        for name in [
            "refs/remotes/origin/HEAD",
            "refs/jj/keep/0123456789abcdef0123456789abcdef01234567",
            "refs/stash",
            "refs/notes/commits",
            "refs/pull/1/head",
            "refs/changes/01/1/1",
            "HEAD",
        ] {
            assert!(!is_importable_git_ref(name), "{name} should be ignored");
        }
    }

    /// Op store whose `ReadOperation` or `ReadView` fails and that records publish attempts.
    #[derive(Default)]
    struct FailingReadOpStore {
        fail_read_op: bool,
        published: Arc<AtomicBool>,
    }

    #[tonic::async_trait]
    impl OpStoreService for FailingReadOpStore {
        async fn read_operation(
            &self,
            _request: Request<ReadOperationRequest>,
        ) -> Result<Response<ReadOperationResponse>, Status> {
            if self.fail_read_op {
                return Err(Status::unavailable("injected ReadOperation failure"));
            }
            let operation = Operation {
                view_id: vec![1; 64],
                ..Default::default()
            };
            Ok(Response::new(ReadOperationResponse {
                operation: Some(operation),
            }))
        }

        async fn write_operation(
            &self,
            _request: Request<WriteOperationRequest>,
        ) -> Result<Response<WriteOperationResponse>, Status> {
            Err(Status::unimplemented("write_operation"))
        }

        async fn resolve_operation_prefix(
            &self,
            _request: Request<ResolveOperationPrefixRequest>,
        ) -> Result<Response<ResolveOperationPrefixResponse>, Status> {
            Err(Status::unimplemented("resolve_operation_prefix"))
        }

        async fn read_view(
            &self,
            _request: Request<ReadViewRequest>,
        ) -> Result<Response<ReadViewResponse>, Status> {
            Err(Status::unavailable("injected ReadView failure"))
        }

        async fn write_view(
            &self,
            _request: Request<WriteViewRequest>,
        ) -> Result<Response<WriteViewResponse>, Status> {
            Err(Status::unimplemented("write_view"))
        }

        async fn get_op_heads(
            &self,
            _request: Request<GetOpHeadsRequest>,
        ) -> Result<Response<GetOpHeadsResponse>, Status> {
            Err(Status::unimplemented("get_op_heads"))
        }

        async fn update_op_heads(
            &self,
            _request: Request<UpdateOpHeadsRequest>,
        ) -> Result<Response<UpdateOpHeadsResponse>, Status> {
            Err(Status::unimplemented("update_op_heads"))
        }

        async fn reconcile_op_heads(
            &self,
            _request: Request<ReconcileOpHeadsRequest>,
        ) -> Result<Response<ReconcileOpHeadsResponse>, Status> {
            Ok(Response::new(ReconcileOpHeadsResponse {
                op_head: vec![2; 64],
            }))
        }

        async fn publish_operation(
            &self,
            _request: Request<PublishOperationRequest>,
        ) -> Result<Response<PublishOperationResponse>, Status> {
            self.published.store(true, Ordering::SeqCst);
            Ok(Response::new(PublishOperationResponse::default()))
        }
    }

    #[tokio::test]
    async fn publish_imported_refs_fails_without_publishing_when_op_or_view_read_fails() {
        for (fail_read_op, expected_msg) in [
            (true, "injected ReadOperation failure"),
            (false, "injected ReadView failure"),
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let op_store = FailingReadOpStore {
                fail_read_op,
                ..Default::default()
            };
            let published = op_store.published.clone();
            let incoming = futures::stream::unfold(listener, |listener| async move {
                Some((listener.accept().await.map(|(stream, _)| stream), listener))
            });
            tokio::spawn(
                tonic::transport::Server::builder()
                    .add_service(OpStoreServiceServer::new(op_store))
                    .serve_with_incoming(incoming),
            );

            let result = run_on_tokio(async move {
                let mut op_client = connect_op_store_client(url).await?;
                let heads = vec![vec![3; COMMIT_ID_LENGTH]];
                let no_branches = BTreeMap::new();
                publish_imported_refs(&mut op_client, "repo", &heads, &no_branches, &no_branches)
                    .await
            })
            .await;

            let err = result.expect_err("a failed read must abort the import");
            assert!(
                err.to_string().contains(expected_msg),
                "unexpected error: {err}"
            );
            assert!(
                !published.load(Ordering::SeqCst),
                "nothing may be published after a failed read"
            );
        }
    }
}
