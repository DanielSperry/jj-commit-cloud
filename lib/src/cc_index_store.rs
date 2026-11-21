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

//! `jj_lib::index::IndexStore` wrapper synchronizing shared project commits into the
//! local revset index using incremental server watermarks.
//!
//! # Authority and Crash-Safety Ordering
//! `CommitCloudIndexStore` wraps `jj_lib::default_index::DefaultIndexStore`. On
//! `get_index_at_op`, it queries remote `ListCommits` using the persisted `since_token`
//! watermark in `.jj/repo/index/cc_sync_cursor.json` (falling back to a full project scan
//! if no cursor exists yet or if the local index was rewound past the cursor's anchor
//! commit), fetches each missing commit and missing ancestor once, inserts parents before
//! children, and persists the updated index segment before advancing the local cursor.
//!
//! # Concurrency and Merge Invariants
//! Commits written in sibling repositories sharing the same `project_id` (or imported
//! from Git) become resolvable in local revsets (`jj log <id>`, `jj new <id>`) in
//! $O(\Delta \text{commits})$ rather than rescanning the full project history on every
//! command. Setting `commit-cloud.sync-project-commits = false` or
//! `JJ_CC_SYNC_PROJECT_COMMITS=0` skips project-wide sync and indexes only commits
//! reachable from the operation view plus any full commit IDs passed on the command line.
//!
//! # Cross-Process and RPC Boundaries
//! Uses cursor-paginated `ListCommits` frames (`LIST_COMMITS_PAGE_SIZE = 5,000`) and
//! 16-way pipelined `get_commit_async` reads over the pooled HTTP/2 channel. When the
//! server returns `UNIMPLEMENTED`, `UNAVAILABLE`, or `DEADLINE_EXCEEDED`, the sync logs a
//! warning and returns the local index.

use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt as _;
use jj_lib::backend::{BackendError, CommitId};
use jj_lib::commit::Commit;
use jj_lib::config::{ConfigGetError, ConfigGetResultExt as _};
use jj_lib::dag_walk;
use jj_lib::default_index::DefaultIndexStore;
use jj_lib::index::{
    Index, IndexStore, IndexStoreError, IndexStoreResult, MutableIndex, ReadonlyIndex,
};
use jj_lib::object_id::ObjectId as _;
use jj_lib::operation::Operation;
use jj_lib::repo::{ReadonlyRepo, Repo as _};
use jj_lib::settings::UserSettings;
use jj_lib::store::Store;
use serde::{Deserialize, Serialize};

use crate::cc_backend::CLIENT_BACKEND_CONCURRENCY;
use crate::util::{CommitCloudConfig, connect_backend_client, run_on_tokio};

const LIST_COMMITS_PAGE_SIZE: u32 = 5_000;
const SYNC_CURSOR_FILENAME: &str = "cc_sync_cursor.json";

/// jj config key (boolean, default `true`) that syncs commits written by other
/// repositories in the same project into the local index on every repository load.
pub const SYNC_PROJECT_COMMITS_CONFIG_KEY: &str = "commit-cloud.sync-project-commits";

/// Environment variable (`0`/`false` disables, `1`/`true` enables) overriding the default
/// project-wide commit sync when `commit-cloud.sync-project-commits` is unset in jj config.
pub const SYNC_PROJECT_COMMITS_ENV: &str = "JJ_CC_SYNC_PROJECT_COMMITS";

static CLI_COMMIT_HINTS: Mutex<Vec<CommitId>> = Mutex::new(Vec::new());

/// Records any full 40-character hexadecimal commit IDs appearing in process CLI arguments
/// so `CommitCloudIndexStore` can fetch and index them on demand (`O(depth)`) when
/// project-wide commit sync is disabled.
pub fn register_cli_commit_hints<I, S>(args: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let hints = extract_cli_commit_hints(args);
    *CLI_COMMIT_HINTS.lock().unwrap() = hints;
}

fn extract_cli_commit_hints<I, S>(args: I) -> Vec<CommitId>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    const HEX_LEN: usize = cc_common::COMMIT_ID_LENGTH * 2;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for arg in args.into_iter().skip(1) {
        let Some(s) = arg.as_ref().to_str() else {
            continue;
        };
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i].is_ascii_hexdigit() {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                    i += 1;
                }
                if i - start == HEX_LEN
                    && let Some(cid) = CommitId::try_from_hex(&s[start..i])
                    && cid.as_bytes().len() == cc_common::COMMIT_ID_LENGTH
                    && seen.insert(cid.clone())
                {
                    out.push(cid);
                }
            } else {
                i += 1;
            }
        }
    }
    out
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexSyncCursor {
    server_url: String,
    project_id: String,
    sync_token_hex: String,
    anchor_commit_hex: Option<String>,
}

#[derive(Debug)]
pub struct CommitCloudIndexStore {
    inner: DefaultIndexStore,
    config: CommitCloudConfig,
    index_dir: PathBuf,
    sync_project_commits: bool,
}

impl CommitCloudIndexStore {
    pub fn name() -> &'static str {
        "commit_cloud"
    }

    pub fn init(
        settings: &UserSettings,
        store_path: &Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let inner = DefaultIndexStore::init(store_path)?;
        let config = CommitCloudConfig::load_from_store(store_path)?;
        let sync_project_commits = sync_project_commits_setting(settings)?;
        Ok(Self {
            inner,
            config,
            index_dir: store_path.to_path_buf(),
            sync_project_commits,
        })
    }

    pub fn load(
        settings: &UserSettings,
        store_path: &Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let inner = DefaultIndexStore::load(store_path);
        let config = CommitCloudConfig::load_from_store(store_path)?;
        let sync_project_commits = sync_project_commits_setting(settings)?;
        Ok(Self {
            inner,
            config,
            index_dir: store_path.to_path_buf(),
            sync_project_commits,
        })
    }

    fn load_valid_cursor(&self, index: &dyn ReadonlyIndex) -> Option<(Vec<u8>, Option<String>)> {
        let raw = fs::read_to_string(self.index_dir.join(SYNC_CURSOR_FILENAME)).ok()?;
        let cursor: IndexSyncCursor = serde_json::from_str(&raw).ok()?;
        if cursor.server_url != self.config.server_url
            || cursor.project_id != self.config.effective_project_id()
            || cursor.sync_token_hex.is_empty()
        {
            return None;
        }
        if let Some(anchor_hex) = &cursor.anchor_commit_hex {
            let anchor_id = CommitId::try_from_hex(anchor_hex)?;
            if !index.as_index().has_id(&anchor_id).ok()? {
                return None;
            }
        }
        let token = hex::decode(&cursor.sync_token_hex).ok()?;
        Some((token, cursor.anchor_commit_hex))
    }

    fn save_cursor(&self, sync_token: &[u8], anchor_commit_hex: Option<String>) {
        if sync_token.is_empty() {
            return;
        }
        let cursor = IndexSyncCursor {
            server_url: self.config.server_url.clone(),
            project_id: self.config.effective_project_id().to_string(),
            sync_token_hex: hex::encode(sync_token),
            anchor_commit_hex,
        };
        if let Ok(bytes) = serde_json::to_vec(&cursor) {
            let tmp = self
                .index_dir
                .join(format!("{SYNC_CURSOR_FILENAME}.tmp.{}", std::process::id()));
            if fs::write(&tmp, bytes).is_ok() {
                let _ = fs::rename(&tmp, self.index_dir.join(SYNC_CURSOR_FILENAME));
            }
        }
    }
}

#[async_trait(?Send)]
impl IndexStore for CommitCloudIndexStore {
    fn name(&self) -> &str {
        Self::name()
    }

    async fn get_index_at_op(
        &self,
        op: &Operation,
        store: &Arc<Store>,
    ) -> IndexStoreResult<Box<dyn ReadonlyIndex>> {
        let index = self.inner.get_index_at_op(op, store).await?;
        if !self.sync_project_commits {
            return self.index_cli_hinted_commits(index, op, store).await;
        }

        let (since_token, prev_anchor) = self
            .load_valid_cursor(index.as_ref())
            .unwrap_or_else(|| (Vec::new(), None));

        let server_url = self.config.server_url.clone();
        let project_id = self.config.effective_project_id().to_string();
        let listed: Result<(Vec<CommitId>, Vec<u8>), _> = run_on_tokio(async move {
            let mut client = connect_backend_client(server_url).await?;
            let mut all_ids = Vec::new();
            let mut page_token = Vec::new();
            let mut final_sync_token = Vec::new();
            loop {
                match client
                    .list_commits(tonic::Request::new(
                        cc_common::backend::ListCommitsRequest {
                            project_id: project_id.clone(),
                            page_token,
                            page_size: LIST_COMMITS_PAGE_SIZE,
                            since_token: since_token.clone(),
                        },
                    ))
                    .await
                {
                    Ok(resp) => {
                        let res = resp.into_inner();
                        // A full scan's watermark is read before its first page, so keep the first
                        // one; each incremental page advances the watermark, so keep the latest.
                        if !res.sync_token.is_empty()
                            && (final_sync_token.is_empty() || !since_token.is_empty())
                        {
                            final_sync_token = res.sync_token;
                        }
                        all_ids
                            .extend(res.commit_ids.into_iter().map(|b| CommitId::from_bytes(&b)));
                        if res.next_page_token.is_empty() {
                            break;
                        }
                        page_token = res.next_page_token;
                    }
                    Err(status) if status.code() == tonic::Code::Unimplemented => {
                        let fallback = client
                            .list_project_commits(tonic::Request::new(
                                cc_common::backend::ListProjectCommitsRequest { project_id },
                            ))
                            .await?
                            .into_inner();
                        all_ids = fallback
                            .commit_ids
                            .into_iter()
                            .map(|b| CommitId::from_bytes(&b))
                            .collect();
                        final_sync_token.clear();
                        break;
                    }
                    Err(status) => return Err(status.into()),
                }
            }
            Ok((all_ids, final_sync_token))
        })
        .await;
        let (remote_commit_ids, new_sync_token) = match listed {
            Ok(listed) => listed,
            Err(err) => return keep_local_index_or_fail(index, err),
        };

        let mut missing_ids = Vec::new();
        for cid in &remote_commit_ids {
            if !index
                .as_index()
                .has_id(cid)
                .map_err(|e| IndexStoreError::Read(e.into()))?
            {
                missing_ids.push(cid.clone());
            }
        }

        if missing_ids.is_empty() {
            let anchor = remote_commit_ids.last().map(|c| c.hex()).or(prev_anchor);
            self.save_cursor(&new_sync_token, anchor);
            return Ok(index);
        }

        let anchor = missing_ids.last().map(|c| c.hex()).or(prev_anchor);
        let fetched = match fetch_missing_commits(store, index.as_index(), missing_ids).await {
            Ok(fetched) => fetched,
            Err(err) => return keep_local_index_or_fail(index, err),
        };

        let mut mut_index = index.start_modification();
        add_commits_parents_first(&mut *mut_index, &fetched).await?;

        let written = self.inner.write_index(mut_index, op)?;
        self.save_cursor(&new_sync_token, anchor);
        Ok(written)
    }

    fn write_index(
        &self,
        index: Box<dyn MutableIndex>,
        op: &Operation,
    ) -> IndexStoreResult<Box<dyn ReadonlyIndex>> {
        self.inner.write_index(index, op)
    }
}

/// Indexes `commit_ids` and their ancestors that are missing from the index at `repo`'s
/// operation, so they resolve in revsets when `commit-cloud.sync-project-commits` is off.
/// Indexes of later operations build on this one, so the commits stay resolvable.
pub async fn index_commits(repo: &ReadonlyRepo, commit_ids: &[CommitId]) -> IndexStoreResult<()> {
    let index = repo.readonly_index();
    let mut missing_ids = Vec::new();
    for cid in commit_ids {
        if !index
            .as_index()
            .has_id(cid)
            .map_err(|e| IndexStoreError::Read(e.into()))?
        {
            missing_ids.push(cid.clone());
        }
    }
    if missing_ids.is_empty() {
        return Ok(());
    }
    let fetched = fetch_missing_commits(repo.store(), index.as_index(), missing_ids)
        .await
        .map_err(IndexStoreError::Read)?;
    let mut mut_index = index.start_modification();
    add_commits_parents_first(&mut *mut_index, &fetched).await?;
    repo.loader()
        .index_store()
        .write_index(mut_index, repo.operation())?;
    Ok(())
}

impl CommitCloudIndexStore {
    async fn index_cli_hinted_commits(
        &self,
        index: Box<dyn ReadonlyIndex>,
        op: &Operation,
        store: &Arc<Store>,
    ) -> IndexStoreResult<Box<dyn ReadonlyIndex>> {
        let hints = CLI_COMMIT_HINTS.lock().unwrap().clone();
        if hints.is_empty() {
            return Ok(index);
        }
        let mut candidates = Vec::new();
        for cid in hints {
            if !index
                .as_index()
                .has_id(&cid)
                .map_err(|e| IndexStoreError::Read(e.into()))?
            {
                candidates.push(cid);
            }
        }
        if candidates.is_empty() {
            return Ok(index);
        }

        // Probe hinted full commit IDs first, ignoring `ObjectNotFound` so `present(<id>)`
        // and cross-project isolation behave identically to a local index miss.
        let mut seeded_roots = Vec::new();
        for cid in candidates {
            match store.get_commit_async(&cid).await {
                Ok(_) => seeded_roots.push(cid),
                Err(BackendError::ObjectNotFound { .. }) => {}
                Err(err) => return keep_local_index_or_fail(index, err.into()),
            }
        }
        if seeded_roots.is_empty() {
            return Ok(index);
        }

        let fetched = match fetch_missing_commits(store, index.as_index(), seeded_roots).await {
            Ok(fetched) => fetched,
            Err(err) => return keep_local_index_or_fail(index, err),
        };
        let mut mut_index = index.start_modification();
        add_commits_parents_first(&mut *mut_index, &fetched).await?;
        self.inner.write_index(mut_index, op)
    }
}

fn sync_project_commits_setting(settings: &UserSettings) -> Result<bool, ConfigGetError> {
    if let Some(configured) = settings
        .get_bool(SYNC_PROJECT_COMMITS_CONFIG_KEY)
        .optional()?
    {
        return Ok(configured);
    }
    if let Ok(raw) = std::env::var(SYNC_PROJECT_COMMITS_ENV)
        && let Some(parsed) = parse_env_bool(&raw)
    {
        return Ok(parsed);
    }
    Ok(true)
}

fn parse_env_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Returns `index` without the synced commits when `err` shows that the server cannot sync
/// right now, so read-only commands keep working with the commits already indexed locally.
fn keep_local_index_or_fail(
    index: Box<dyn ReadonlyIndex>,
    err: Box<dyn std::error::Error + Send + Sync>,
) -> IndexStoreResult<Box<dyn ReadonlyIndex>> {
    if is_sync_unavailable(err.as_ref()) {
        tracing::warn!("Skipping Commit Cloud project commit sync: {err}");
        Ok(index)
    } else {
        Err(IndexStoreError::Read(err))
    }
}

/// Returns whether `err` carries a gRPC status meaning the server lacks `ListCommits` or is
/// unreachable or too slow, as opposed to a failure that the user should see.
fn is_sync_unavailable(err: &(dyn std::error::Error + 'static)) -> bool {
    let mut next = Some(err);
    while let Some(err) = next {
        if let Some(status) = err.downcast_ref::<tonic::Status>() {
            return matches!(
                status.code(),
                tonic::Code::Unimplemented
                    | tonic::Code::Unavailable
                    | tonic::Code::DeadlineExceeded
            );
        }
        // `BackendError::Other` is `#[error(transparent)]`, so `source()` would skip the status.
        next = match err.downcast_ref::<BackendError>() {
            Some(BackendError::Other(inner)) => Some(inner.as_ref()),
            _ => err.source(),
        };
    }
    false
}

/// Fetches `missing_ids` and their ancestors that `index` lacks, each once and at most
/// `CLIENT_BACKEND_CONCURRENCY` at a time. jj-lib's commit cache holds only 100 commits, so
/// the commits are kept here until they are indexed.
async fn fetch_missing_commits(
    store: &Arc<Store>,
    index: &dyn Index,
    missing_ids: Vec<CommitId>,
) -> Result<HashMap<CommitId, Commit>, Box<dyn std::error::Error + Send + Sync>> {
    let mut fetched = HashMap::new();
    let mut queued: HashSet<CommitId> = missing_ids.iter().cloned().collect();
    let mut frontier = missing_ids;
    while !frontier.is_empty() {
        let mut commits =
            futures::stream::iter(std::mem::take(&mut frontier).into_iter().map(|cid| {
                let store = store.clone();
                async move { store.get_commit_async(&cid).await }
            }))
            .buffer_unordered(CLIENT_BACKEND_CONCURRENCY);
        while let Some(commit) = commits.next().await {
            let commit = commit?;
            for parent_id in commit.parent_ids() {
                if !queued.contains(parent_id) && !index.has_id(parent_id)? {
                    queued.insert(parent_id.clone());
                    frontier.push(parent_id.clone());
                }
            }
            fetched.insert(commit.id().clone(), commit);
        }
    }
    Ok(fetched)
}

/// Adds `commits` to `mut_index`, each after its parents, whose other ancestors must
/// already be indexed.
async fn add_commits_parents_first(
    mut_index: &mut dyn MutableIndex,
    commits: &HashMap<CommitId, Commit>,
) -> IndexStoreResult<()> {
    let parents: HashMap<CommitId, Vec<CommitId>> = commits
        .iter()
        .map(|(id, commit)| (id.clone(), commit.parent_ids().to_vec()))
        .collect();
    for id in parents_first_order(&parents)? {
        mut_index
            .add_commit(&commits[&id])
            .await
            .map_err(|e| IndexStoreError::Write(e.into()))?;
    }
    Ok(())
}

/// Orders the keys of `parents` (commit id to parent ids) so each commit follows its parents
/// that are also keys. The walk keeps an explicit stack, so deep histories cannot overflow
/// the thread stack.
fn parents_first_order(
    parents: &HashMap<CommitId, Vec<CommitId>>,
) -> IndexStoreResult<Vec<CommitId>> {
    dag_walk::topo_order_forward(
        parents.keys().cloned(),
        |id: &CommitId| id.clone(),
        |id: &CommitId| {
            parents[id]
                .iter()
                .filter(|parent| parents.contains_key(*parent))
                .cloned()
                .collect::<Vec<_>>()
        },
        |id: CommitId| {
            IndexStoreError::Read(format!("commit graph has a cycle at {}", id.hex()).into())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parents_first_order_handles_deep_chain_without_stack_overflow() {
        const DEPTH: usize = 20_000;
        let make_id = |idx: usize| {
            let mut b = [0u8; 20];
            b[..8].copy_from_slice(&(idx as u64 + 1).to_be_bytes());
            CommitId::from_bytes(&b)
        };

        let mut parents = HashMap::with_capacity(DEPTH);
        let root = CommitId::from_bytes(&cc_common::ROOT_COMMIT_ID_BYTES);
        for idx in 0..DEPTH {
            let cid = make_id(idx);
            let parent = if idx == 0 {
                root.clone()
            } else {
                make_id(idx - 1)
            };
            parents.insert(cid, vec![parent]);
        }

        let ordered = parents_first_order(&parents).unwrap();
        assert_eq!(ordered.len(), DEPTH);
        for (idx, cid) in ordered.iter().enumerate() {
            assert_eq!(cid, &make_id(idx));
        }
    }

    #[test]
    fn test_extract_cli_commit_hints_parses_exact_full_commit_ids() {
        let cid1 = "0123456789abcdef0123456789abcdef01234567";
        let cid2 = "fedcba9876543210fedcba9876543210fedcba98";
        let too_short = "0123456789abcdef";
        let too_long = "0123456789abcdef0123456789abcdef0123456789";

        let hints = extract_cli_commit_hints([
            "jj",
            "new",
            cid1,
            &format!("present({cid2})|{cid1}"),
            too_short,
            too_long,
        ]);
        assert_eq!(
            hints,
            vec![CommitId::from_hex(cid1), CommitId::from_hex(cid2)]
        );
    }

    #[test]
    fn test_is_sync_unavailable_recognizes_transient_and_unimplemented_statuses() {
        let unimpl = BackendError::Other(Box::new(tonic::Status::unimplemented("no ListCommits")));
        let unavail =
            BackendError::Other(Box::new(tonic::Status::unavailable("server restarting")));
        let deadline = tonic::Status::deadline_exceeded("timed out");
        let auth = BackendError::Other(Box::new(tonic::Status::unauthenticated("bad token")));

        assert!(is_sync_unavailable(&unimpl));
        assert!(is_sync_unavailable(&unavail));
        assert!(is_sync_unavailable(&deadline));
        assert!(!is_sync_unavailable(&auth));
    }

    #[test]
    fn test_parse_env_bool() {
        assert_eq!(parse_env_bool("0"), Some(false));
        assert_eq!(parse_env_bool("false"), Some(false));
        assert_eq!(parse_env_bool("OFF"), Some(false));
        assert_eq!(parse_env_bool("1"), Some(true));
        assert_eq!(parse_env_bool("true"), Some(true));
        assert_eq!(parse_env_bool("invalid"), None);
    }
}
