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

//! Server-side operation-head reconciliation engine executing `jj_lib`'s 3-way view and
//! tree merge algebra against [`CompositeStore`].
//!
//! # Authority and Crash-Safety Ordering
//! `reconcile_repo_op_heads` writes the merged `View` and `Operation` through
//! [`ServerOpStore`] before advancing the repository head set via additive
//! [`ServerOpHeadsStore::update_op_heads`]. Server-authored merge operations carry
//! `"jj-cc-server"` hostname and username metadata.
//!
//! # Concurrency and Merge Invariants
//! In-process callers targeting the same `repo_id` serialize behind a per-repository
//! async mutex so concurrent `ReconcileOpHeads` requests coalesce into one merge pass.
//! When divergent operations rewrite a commit whose descendants, bookmarks, or other
//! workspace pointers changed on another branch, `reconcile_repo_op_heads` returns an
//! empty `op_head` so the client resolves the rewrites against `DefaultIndexStore`.
//!
//! # Cross-Process and RPC Boundaries
//! Runs `RepoLoader::load_at_head()` (`!Send` due to `dyn Repo`) on Tokio's
//! `spawn_blocking` pool via `pollster::block_on`, catching worker panics as retryable
//! attempts with bounded exponential backoff.

#![allow(elided_lifetimes_in_paths)]

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use futures::TryStreamExt as _;
use jj_lib::backend::CommitId;
use jj_lib::object_id::ObjectId as _;
use jj_lib::op_heads_store::OpHeadsStore;
use jj_lib::op_store::{OpStore, OperationId};
use jj_lib::operation::Operation as JjOperation;
use jj_lib::repo::RepoLoader;
use jj_lib::settings::UserSettings;
use jj_lib::signing::Signer;
use jj_lib::store::Store as JjStore;
use jj_lib::tree_merge::MergeOptions;

use crate::jj_lib_adapters::{
    ServerBackend, ServerIndexStore, ServerOpHeadsStore, ServerOpStore, ServerSubmoduleStore,
};
use crate::store::CompositeStore;

static RECONCILER_ENV: LazyLock<(UserSettings, MergeOptions)> = LazyLock::new(|| {
    let mut config = jj_lib::config::StackedConfig::with_defaults();
    config.add_layer(
        jj_lib::config::ConfigLayer::parse(
            jj_lib::config::ConfigSource::Default,
            "operation.hostname = \"jj-cc-server\"\noperation.username = \"jj-cc-server\"\n",
        )
        .expect("reconciler config layer should parse"),
    );
    let user_settings =
        UserSettings::from_config(config).expect("default user settings should load");
    let merge_options = MergeOptions::from_settings(&user_settings)
        .expect("default merge options should initialize");
    (user_settings, merge_options)
});

static REPO_RECONCILE_LOCKS: LazyLock<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn repo_reconcile_mutex(repo_id: &str) -> Arc<tokio::sync::Mutex<()>> {
    REPO_RECONCILE_LOCKS
        .lock()
        .unwrap()
        .entry(repo_id.to_string())
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

pub async fn reconcile_repo_op_heads(
    store: Arc<CompositeStore>,
    repo_id: &str,
) -> Result<Vec<u8>, tonic::Status> {
    const MAX_RETRIES: usize = 6;
    const BASE_BACKOFF_MS: u64 = 20;
    const MAX_JITTER_MS: u64 = 15;

    store
        .ensure_repo_registered(repo_id, "reconciling op heads")
        .await?;

    // Fast path when heads are already converged.
    let initial_heads = store.get_op_heads(repo_id).await?;
    if initial_heads.len() <= 1 {
        return Ok(initial_heads
            .into_iter()
            .next()
            .unwrap_or_else(|| cc_common::ROOT_OPERATION_ID_BYTES.to_vec()));
    }

    // Coalesce concurrent in-process reconcile calls for the same `repo_id`.
    let repo_lock = repo_reconcile_mutex(repo_id);
    let _guard = repo_lock.lock().await;

    let mut attempt = 0usize;
    loop {
        let heads = store.get_op_heads(repo_id).await?;
        if heads.len() <= 1 {
            return Ok(heads
                .into_iter()
                .next()
                .unwrap_or_else(|| cc_common::ROOT_OPERATION_ID_BYTES.to_vec()));
        }

        let store_clone = store.clone();
        let repo_id_owned = repo_id.to_string();

        let handle = tokio::task::spawn_blocking(move || {
            pollster::block_on(async move {
                let op_store: Arc<dyn OpStore> = Arc::new(ServerOpStore::new(
                    store_clone.clone(),
                    repo_id_owned.clone(),
                ));
                if requires_client_rewrite_resolution(&op_store, &heads).await? {
                    return Ok(None);
                }

                let (ref user_settings, ref merge_options) = *RECONCILER_ENV;
                let signer = Signer::from_settings(user_settings)
                    .map_err(|e| format!("failed to initialize signer: {e}"))?;
                let index_store = Arc::new(ServerIndexStore::new(
                    store_clone.clone(),
                    repo_id_owned.clone(),
                ));
                let submodule_store = Arc::new(ServerSubmoduleStore);

                let server_backend = Box::new(ServerBackend::new(
                    store_clone.clone(),
                    repo_id_owned.clone(),
                ));
                let jj_store = JjStore::new(server_backend, signer, merge_options.clone());
                let op_heads_store: Arc<dyn OpHeadsStore> =
                    Arc::new(ServerOpHeadsStore::new(store_clone, repo_id_owned));

                let repo_loader = RepoLoader::new(
                    user_settings.clone(),
                    jj_store,
                    op_store,
                    op_heads_store,
                    index_store,
                    submodule_store,
                );

                repo_loader
                    .load_at_head()
                    .await
                    .map(|repo| Some(repo.op_id().as_bytes().to_vec()))
                    .map_err(|e| e.to_string())
            })
        });

        let reconcile_result = run_reconcile_join(handle).await;

        match reconcile_result {
            Ok(Some(op_id)) => return Ok(op_id),
            Ok(None) => return Ok(Vec::new()),
            Err(e) if attempt + 1 < MAX_RETRIES => {
                tracing::warn!(
                    repo_id = %repo_id,
                    attempt = attempt + 1,
                    error = %e,
                    "Transient op-head reconciliation failure; retrying"
                );
                let jitter = rand::random::<u64>() % MAX_JITTER_MS;
                let sleep_ms = (BASE_BACKOFF_MS << attempt.min(5)) + jitter;
                tokio::time::sleep(Duration::from_millis(sleep_ms)).await;
                attempt += 1;
            }
            Err(e) => {
                return Err(tonic::Status::internal(format!(
                    "Failed to reconcile op heads for repo {repo_id}: {e}"
                )));
            }
        }
    }
}

async fn run_reconcile_join<T>(
    handle: tokio::task::JoinHandle<Result<T, String>>,
) -> Result<T, String> {
    match handle.await {
        Ok(res) => res,
        Err(join_err) => Err(format!("reconciliation worker failed: {join_err}")),
    }
}

/// Returns `true` when divergent operation branches contain commit rewrites that interact
/// with another branch's commits, bookmarks, or working copies and therefore require
/// `DefaultIndexStore` descendant rebasing on the client.
async fn requires_client_rewrite_resolution(
    op_store: &Arc<dyn OpStore>,
    raw_head_ids: &[Vec<u8>],
) -> Result<bool, String> {
    let mut head_ops = Vec::with_capacity(raw_head_ids.len());
    for raw in raw_head_ids {
        let id = OperationId::from_bytes(raw);
        let data = op_store
            .read_operation(&id)
            .await
            .map_err(|e| e.to_string())?;
        head_ops.push(JjOperation::new(op_store.clone(), id, data));
    }

    let filtered = jj_lib::dag_walk_async::heads(
        head_ops,
        |op: &JjOperation| op.id().clone(),
        async |op: &JjOperation| op.parents().await,
    )
    .await
    .map_err(|e| e.to_string())?;
    let filtered_vec: Vec<JjOperation> = filtered.into_iter().collect();
    if filtered_vec.len() <= 1 {
        return Ok(false);
    }

    let mut base_ops = vec![filtered_vec[0].clone()];
    for h in &filtered_vec[1..] {
        base_ops = jj_lib::op_walk::closest_common_ancestors(base_ops, std::iter::once(h.clone()))
            .await
            .map_err(|e| e.to_string())?;
    }

    let mut branch_rewritten_preds: Vec<HashSet<CommitId>> = Vec::with_capacity(filtered_vec.len());
    let mut any_external_rewrite = false;
    for head_op in &filtered_vec {
        let branch_ops: Vec<JjOperation> =
            jj_lib::op_walk::walk_ancestors_range(std::slice::from_ref(head_op), &base_ops)
                .try_collect()
                .await
                .map_err(|e| e.to_string())?;
        let mut created_in_branch = HashSet::new();
        let mut all_preds = HashSet::new();
        for op in &branch_ops {
            if let Some(cp_map) = &op.store_operation().commit_predecessors {
                for (new_cid, preds) in cp_map {
                    created_in_branch.insert(new_cid.clone());
                    for p in preds {
                        all_preds.insert(p.clone());
                    }
                }
            }
        }
        all_preds.retain(|p| !created_in_branch.contains(p));
        if !all_preds.is_empty() {
            any_external_rewrite = true;
        }
        branch_rewritten_preds.push(all_preds);
    }

    if !any_external_rewrite {
        return Ok(false);
    }

    let mut head_views = Vec::with_capacity(filtered_vec.len());
    for h in &filtered_vec {
        head_views.push(h.view().await.map_err(|e| e.to_string())?);
    }
    let mut base_views = Vec::with_capacity(base_ops.len());
    for b in &base_ops {
        base_views.push(b.view().await.map_err(|e| e.to_string())?);
    }

    for (i, preds) in branch_rewritten_preds.iter().enumerate() {
        for pred in preds {
            for (j, other_view) in head_views.iter().enumerate() {
                if i == j {
                    continue;
                }
                if !other_view.heads().contains(pred) {
                    return Ok(true);
                }
                if other_view.local_bookmarks().any(|(_, target)| {
                    target.added_ids().any(|id| id == pred)
                        || target.removed_ids().any(|id| id == pred)
                }) {
                    return Ok(true);
                }
                if other_view.all_remote_bookmarks().any(|(_, remote_ref)| {
                    remote_ref.target.added_ids().any(|id| id == pred)
                        || remote_ref.target.removed_ids().any(|id| id == pred)
                }) {
                    return Ok(true);
                }
                for (ws_name, wc_id) in other_view.wc_commit_ids() {
                    if wc_id == pred
                        && (!base_views
                            .iter()
                            .all(|bv| bv.wc_commit_ids().get(ws_name) == Some(pred))
                            || head_views[i].wc_commit_ids().get(ws_name) == Some(pred))
                    {
                        return Ok(true);
                    }
                }
            }
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use cc_common::backend::Commit as ProtoCommit;
    use cc_common::op_store::{
        Operation as ProtoOperation, OperationMetadata as ProtoOperationMetadata, RefTarget,
        RefTargetTerm, View as ProtoView,
    };
    use cc_lib::cc_op_store::{operation_from_proto, view_from_proto};
    use jj_lib::backend::CommitId;
    use jj_lib::op_store::OperationId;

    use super::*;

    fn make_test_proto_op(view_id: Vec<u8>, parents: Vec<Vec<u8>>, desc: &str) -> ProtoOperation {
        ProtoOperation {
            view_id,
            parents,
            metadata: Some(ProtoOperationMetadata {
                start_time_millis: 1000,
                end_time_millis: 1000,
                description: desc.to_string(),
                is_snapshot: false,
                workspace_name: None,
                hostname: "test-host".to_string(),
                username: "test-user".to_string(),
                attributes: HashMap::new(),
            }),
            commit_predecessors: vec![],
            commit_predecessors_set: true,
        }
    }

    #[tokio::test]
    async fn test_reconcile_single_head_is_noop() {
        let store = Arc::new(CompositeStore::in_memory());
        store.register_repo("single-head-repo", None).await.unwrap();
        let res = reconcile_repo_op_heads(store, "single-head-repo")
            .await
            .unwrap();
        assert_eq!(res, cc_common::ROOT_OPERATION_ID_BYTES.to_vec());
    }

    #[tokio::test]
    async fn test_reconcile_merges_divergent_bookmarks_and_prunes_ancestor_heads() {
        let store = Arc::new(CompositeStore::in_memory());
        let repo_id = "reconcile-ancestry-repo";
        store.register_repo(repo_id, None).await.unwrap();

        let root_op = cc_common::ROOT_OPERATION_ID_BYTES.to_vec();
        let c1_id = vec![0x01; 20];
        let c2_id = vec![0x02; 20];
        let c3_id = vec![0x03; 20];

        // Build commit chain: root -> c1 -> c2, and sibling root -> c3.
        for (cid, parent) in [
            (&c1_id, cc_common::ROOT_COMMIT_ID_BYTES.to_vec()),
            (&c2_id, c1_id.clone()),
            (&c3_id, cc_common::ROOT_COMMIT_ID_BYTES.to_vec()),
        ] {
            store
                .put_commit(
                    repo_id,
                    cid,
                    &ProtoCommit {
                        commit_id: cid.clone(),
                        change_id: cid[..16].to_vec(),
                        parent_commit_ids: vec![parent],
                        root_tree_id: vec![cc_common::EMPTY_TREE_ID_BYTES.to_vec()],
                        description: "test\n".to_string(),
                        author: None,
                        committer: None,
                        predecessors: vec![],
                        conflict_labels: vec![],
                        secure_sig: None,
                        raw_git_commit: vec![],
                        external_ids: Default::default(),
                    },
                )
                .await
                .unwrap();
        }

        // Base operation Op0 has bookmark `main = c1` and `head_ids = [c1]`.
        let mut bm_base = HashMap::new();
        bm_base.insert(
            "main".to_string(),
            RefTarget {
                removes: vec![],
                adds: vec![RefTargetTerm {
                    commit_id: c1_id.clone(),
                }],
            },
        );
        let view0 = ProtoView {
            head_ids: vec![c1_id.clone()],
            wc_commit_ids: HashMap::new(),
            local_bookmarks: bm_base.clone(),
            remote_bookmarks: HashMap::new(),
        };
        let view0_id = crate::hash_utils::hash_view(&view0);
        store.put_view(repo_id, &view0_id, &view0).await.unwrap();
        let op0 = make_test_proto_op(view0_id, vec![root_op.clone()], "base");
        let op0_id = crate::hash_utils::hash_operation(&op0);
        store.put_operation(repo_id, &op0_id, &op0).await.unwrap();
        store
            .update_op_heads(repo_id, &[root_op], &op0_id)
            .await
            .unwrap();

        // Branch A fast-forwards `main` from c1 -> c2 (`head_ids = [c2]`).
        let mut bm_a = HashMap::new();
        bm_a.insert(
            "main".to_string(),
            RefTarget {
                removes: vec![],
                adds: vec![RefTargetTerm {
                    commit_id: c2_id.clone(),
                }],
            },
        );
        let view_a = ProtoView {
            head_ids: vec![c2_id.clone()],
            wc_commit_ids: HashMap::new(),
            local_bookmarks: bm_a,
            remote_bookmarks: HashMap::new(),
        };
        let view_a_id = crate::hash_utils::hash_view(&view_a);
        store.put_view(repo_id, &view_a_id, &view_a).await.unwrap();
        let op_a = make_test_proto_op(view_a_id, vec![op0_id.clone()], "advance main");
        let op_a_id = crate::hash_utils::hash_operation(&op_a);
        store.put_operation(repo_id, &op_a_id, &op_a).await.unwrap();

        // Branch B keeps `main = c1` and adds bookmark `feature = c3` (`head_ids = [c1, c3]`).
        let mut bm_b = bm_base;
        bm_b.insert(
            "feature".to_string(),
            RefTarget {
                removes: vec![],
                adds: vec![RefTargetTerm {
                    commit_id: c3_id.clone(),
                }],
            },
        );
        let view_b = ProtoView {
            head_ids: vec![c1_id.clone(), c3_id.clone()],
            wc_commit_ids: HashMap::new(),
            local_bookmarks: bm_b,
            remote_bookmarks: HashMap::new(),
        };
        let view_b_id = crate::hash_utils::hash_view(&view_b);
        store.put_view(repo_id, &view_b_id, &view_b).await.unwrap();
        let op_b = make_test_proto_op(view_b_id, vec![op0_id.clone()], "add feature");
        let op_b_id = crate::hash_utils::hash_operation(&op_b);
        store.put_operation(repo_id, &op_b_id, &op_b).await.unwrap();

        store
            .update_op_heads(repo_id, std::slice::from_ref(&op0_id), &op_a_id)
            .await
            .unwrap();
        store
            .update_op_heads(repo_id, std::slice::from_ref(&op0_id), &op_b_id)
            .await
            .unwrap();

        let merged_op_id = reconcile_repo_op_heads(store.clone(), repo_id)
            .await
            .unwrap();
        assert_eq!(
            store.get_op_heads(repo_id).await.unwrap(),
            vec![merged_op_id.clone()]
        );

        let merged_op = operation_from_proto(
            store
                .get_operation(repo_id, &merged_op_id)
                .await
                .unwrap()
                .unwrap(),
        );
        assert_eq!(merged_op.metadata.hostname, "jj-cc-server");
        assert_eq!(merged_op.metadata.username, "jj-cc-server");
        let parent_set: HashSet<_> = merged_op.parents.into_iter().collect();
        assert_eq!(
            parent_set,
            HashSet::from([
                OperationId::from_bytes(&op_a_id),
                OperationId::from_bytes(&op_b_id),
            ])
        );

        let merged_view = view_from_proto(
            store
                .get_view(repo_id, merged_op.view_id.as_bytes())
                .await
                .unwrap()
                .unwrap(),
        );
        // Because c1 is an ancestor of c2, merged `head_ids` contains only `{c2, c3}` (no dangling
        // c1), and `main` fast-forwarded cleanly to `c2` alongside `feature = c3`.
        assert_eq!(
            merged_view.head_ids,
            HashSet::from([CommitId::from_bytes(&c2_id), CommitId::from_bytes(&c3_id)])
        );
        assert_eq!(
            merged_view
                .local_bookmarks
                .get(jj_lib::ref_name::RefName::new("main"))
                .and_then(|t| t.as_normal()),
            Some(&CommitId::from_bytes(&c2_id))
        );
        assert_eq!(
            merged_view
                .local_bookmarks
                .get(jj_lib::ref_name::RefName::new("feature"))
                .and_then(|t| t.as_normal()),
            Some(&CommitId::from_bytes(&c3_id))
        );
    }

    #[tokio::test]
    async fn test_reconcile_prunes_ancestor_op_head_without_cas_wedge() {
        let store = Arc::new(CompositeStore::in_memory());
        let repo_id = "ancestor-head-repo";
        store.register_repo(repo_id, None).await.unwrap();

        let root_op = cc_common::ROOT_OPERATION_ID_BYTES.to_vec();
        let c1_id = vec![0x11; 20];
        store
            .put_commit(
                repo_id,
                &c1_id,
                &ProtoCommit {
                    commit_id: c1_id.clone(),
                    change_id: c1_id[..16].to_vec(),
                    parent_commit_ids: vec![cc_common::ROOT_COMMIT_ID_BYTES.to_vec()],
                    root_tree_id: vec![cc_common::EMPTY_TREE_ID_BYTES.to_vec()],
                    description: "c1\n".to_string(),
                    author: None,
                    committer: None,
                    predecessors: vec![],
                    conflict_labels: vec![],
                    secure_sig: None,
                    raw_git_commit: vec![],
                    external_ids: Default::default(),
                },
            )
            .await
            .unwrap();

        let view0 = ProtoView {
            head_ids: vec![c1_id],
            wc_commit_ids: HashMap::new(),
            local_bookmarks: HashMap::new(),
            remote_bookmarks: HashMap::new(),
        };
        let view0_id = crate::hash_utils::hash_view(&view0);
        store.put_view(repo_id, &view0_id, &view0).await.unwrap();

        let op0 = make_test_proto_op(view0_id.clone(), vec![root_op.clone()], "op0");
        let op0_id = crate::hash_utils::hash_operation(&op0);
        store.put_operation(repo_id, &op0_id, &op0).await.unwrap();

        let op1 = make_test_proto_op(view0_id, vec![op0_id.clone()], "op1 child of op0");
        let op1_id = crate::hash_utils::hash_operation(&op1);
        store.put_operation(repo_id, &op1_id, &op1).await.unwrap();

        // Advance to `op1`, then simulate an idempotent retry re-adding ancestor `op0` so
        // `OpHeads = [op1, op0]`. Exact-set CAS previously wedged permanently here.
        store
            .update_op_heads(repo_id, std::slice::from_ref(&root_op), &op1_id)
            .await
            .unwrap();
        store.update_op_heads(repo_id, &[], &op0_id).await.unwrap();
        assert_eq!(store.get_op_heads(repo_id).await.unwrap().len(), 2);

        let resolved = reconcile_repo_op_heads(store.clone(), repo_id)
            .await
            .unwrap();
        assert_eq!(resolved, op1_id);
        assert_eq!(store.get_op_heads(repo_id).await.unwrap(), vec![op1_id]);
    }

    #[tokio::test]
    async fn test_reconcile_defers_cross_branch_commit_rewrites_to_client() {
        let store = Arc::new(CompositeStore::in_memory());
        let repo_id = "rewrite-defer-repo";
        store.register_repo(repo_id, None).await.unwrap();

        let root_op = cc_common::ROOT_OPERATION_ID_BYTES.to_vec();
        let x_id = vec![0x21; 20];
        let x_rewritten_id = vec![0x22; 20];
        let y_child_id = vec![0x23; 20];

        for (cid, change_byte, parent) in [
            (&x_id, 0x21u8, cc_common::ROOT_COMMIT_ID_BYTES.to_vec()),
            (
                &x_rewritten_id,
                0x21u8,
                cc_common::ROOT_COMMIT_ID_BYTES.to_vec(),
            ),
            (&y_child_id, 0x23u8, x_id.clone()),
        ] {
            store
                .put_commit(
                    repo_id,
                    cid,
                    &ProtoCommit {
                        commit_id: cid.clone(),
                        change_id: vec![change_byte; 16],
                        parent_commit_ids: vec![parent],
                        root_tree_id: vec![cc_common::EMPTY_TREE_ID_BYTES.to_vec()],
                        description: "test\n".to_string(),
                        author: None,
                        committer: None,
                        predecessors: vec![],
                        conflict_labels: vec![],
                        secure_sig: None,
                        raw_git_commit: vec![],
                        external_ids: Default::default(),
                    },
                )
                .await
                .unwrap();
        }

        let view0 = ProtoView {
            head_ids: vec![x_id.clone()],
            wc_commit_ids: HashMap::new(),
            local_bookmarks: HashMap::new(),
            remote_bookmarks: HashMap::new(),
        };
        let view0_id = crate::hash_utils::hash_view(&view0);
        store.put_view(repo_id, &view0_id, &view0).await.unwrap();
        let op0 = make_test_proto_op(view0_id, vec![root_op.clone()], "base");
        let op0_id = crate::hash_utils::hash_operation(&op0);
        store.put_operation(repo_id, &op0_id, &op0).await.unwrap();
        store
            .update_op_heads(repo_id, &[root_op], &op0_id)
            .await
            .unwrap();

        // Branch A rewrites X -> X'.
        let view_a = ProtoView {
            head_ids: vec![x_rewritten_id.clone()],
            wc_commit_ids: HashMap::new(),
            local_bookmarks: HashMap::new(),
            remote_bookmarks: HashMap::new(),
        };
        let view_a_id = crate::hash_utils::hash_view(&view_a);
        store.put_view(repo_id, &view_a_id, &view_a).await.unwrap();
        let mut op_a = make_test_proto_op(view_a_id, vec![op0_id.clone()], "rewrite X");
        op_a.commit_predecessors = vec![cc_common::op_store::CommitPredecessors {
            commit_id: x_rewritten_id,
            predecessor_ids: vec![x_id.clone()],
        }];
        let op_a_id = crate::hash_utils::hash_operation(&op_a);
        store.put_operation(repo_id, &op_a_id, &op_a).await.unwrap();

        // Branch B creates child Y on top of X (replacing X in head_ids).
        let view_b = ProtoView {
            head_ids: vec![y_child_id],
            wc_commit_ids: HashMap::new(),
            local_bookmarks: HashMap::new(),
            remote_bookmarks: HashMap::new(),
        };
        let view_b_id = crate::hash_utils::hash_view(&view_b);
        store.put_view(repo_id, &view_b_id, &view_b).await.unwrap();
        let op_b = make_test_proto_op(view_b_id, vec![op0_id.clone()], "child of X");
        let op_b_id = crate::hash_utils::hash_operation(&op_b);
        store.put_operation(repo_id, &op_b_id, &op_b).await.unwrap();

        store
            .update_op_heads(repo_id, std::slice::from_ref(&op0_id), &op_a_id)
            .await
            .unwrap();
        store
            .update_op_heads(repo_id, std::slice::from_ref(&op0_id), &op_b_id)
            .await
            .unwrap();

        let res = reconcile_repo_op_heads(store.clone(), repo_id)
            .await
            .unwrap();
        assert!(
            res.is_empty(),
            "cross-branch rewrite must return empty op_head for client rebase"
        );
        assert_eq!(store.get_op_heads(repo_id).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_run_reconcile_join_catches_worker_panic_as_err() {
        let handle = tokio::task::spawn_blocking(|| -> Result<Option<Vec<u8>>, String> {
            panic!("simulated transient worker panic");
        });
        let err = run_reconcile_join(handle).await.unwrap_err();
        assert!(err.contains("reconciliation worker failed"));
    }
}
