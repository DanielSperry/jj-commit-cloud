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

//! `jj_lib::working_copy::WorkingCopy` wrapper synchronizing local checkout state with
//! remote `WorkspaceService`.
//!
//! # Authority and Crash-Safety Ordering
//! `LockedCommitCloudWorkingCopy::finish` completes the local working-copy state commit
//! (`LocalWorkingCopy::finish`) *before* publishing `WorkspaceState` (`commit_id`,
//! `operation_id`, `tree_id` / `tree_ids`, and `sparse_patterns`) to `WorkspaceService`.
//! The server derives `commit_id` from `operation_id` (the operation's
//! `View.wc_commit_ids[workspace_name]`), so snapshots that rewrite `@` without a checkout keep
//! the record current. The client only sends the commit of a successful `check_out`, `reset`, or
//! `recover` in the same mutation, which the server keeps when it cannot read that operation.
//!
//! # Concurrency and Merge Invariants
//! Keyed by `(repo_id, user, workspace_name)`. Renaming a workspace removes the previous
//! `(repo_id, user, old_name)` entry before upserting the new workspace name, and
//! conflicted checkouts preserve all `Merge<TreeId>` sides in `tree_ids`.
//!
//! # Cross-Process and RPC Boundaries
//! Syncs over pooled HTTP/2 channels via [`crate::util::connect_workspace_client`],
//! logging structured `tracing::warn!` diagnostics if the remote workspace service is
//! temporarily unreachable so local working-copy durability is never lost.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use cc_common::workspace::{DeleteWorkspaceRequest, UpdateWorkspaceRequest, WorkspaceState};
use jj_lib::commit::Commit;
use jj_lib::local_working_copy::LocalWorkingCopyFactory;
use jj_lib::merged_tree::MergedTree;
use jj_lib::object_id::ObjectId;
use jj_lib::op_store::OperationId;
use jj_lib::ref_name::{WorkspaceName, WorkspaceNameBuf};
use jj_lib::repo_path::RepoPathBuf;
use jj_lib::settings::UserSettings;
use jj_lib::store::Store;
use jj_lib::working_copy::{
    CheckoutError, CheckoutStats, LockedWorkingCopy, ResetError, SnapshotError, SnapshotOptions,
    SnapshotStats, WorkingCopy, WorkingCopyFactory, WorkingCopyStateError,
};

use crate::util::{CommitCloudConfig, connect_workspace_client, run_async, run_on_tokio};

pub struct CommitCloudWorkingCopy {
    inner: Box<dyn WorkingCopy>,
    config: CommitCloudConfig,
    state_path: PathBuf,
    user: String,
}

impl CommitCloudWorkingCopy {
    pub fn name() -> &'static str {
        "commit_cloud"
    }

    pub fn new(
        inner: Box<dyn WorkingCopy>,
        config: CommitCloudConfig,
        state_path: PathBuf,
        user: String,
    ) -> Self {
        Self {
            inner,
            config,
            state_path,
            user,
        }
    }

    pub fn config(&self) -> &CommitCloudConfig {
        &self.config
    }

    pub fn user(&self) -> &str {
        &self.user
    }
}

#[async_trait(?Send)]
impl WorkingCopy for CommitCloudWorkingCopy {
    fn name(&self) -> &str {
        Self::name()
    }

    fn workspace_name(&self) -> &WorkspaceName {
        self.inner.workspace_name()
    }

    fn operation_id(&self) -> &OperationId {
        self.inner.operation_id()
    }

    fn tree(&self) -> Result<&MergedTree, WorkingCopyStateError> {
        self.inner.tree()
    }

    fn sparse_patterns(&self) -> Result<&[RepoPathBuf], WorkingCopyStateError> {
        self.inner.sparse_patterns()
    }

    async fn start_mutation(&self) -> Result<Box<dyn LockedWorkingCopy>, WorkingCopyStateError> {
        let initial_ws_name = self.inner.workspace_name().as_str().to_string();
        let locked_inner = self.inner.start_mutation().await?;
        Ok(Box::new(LockedCommitCloudWorkingCopy {
            locked_inner,
            config: self.config.clone(),
            state_path: self.state_path.clone(),
            user: self.user.clone(),
            current_commit_id: None,
            initial_workspace_name: initial_ws_name,
            renamed_from: None,
        }))
    }
}

pub struct LockedCommitCloudWorkingCopy {
    locked_inner: Box<dyn LockedWorkingCopy>,
    config: CommitCloudConfig,
    state_path: PathBuf,
    user: String,
    current_commit_id: Option<Vec<u8>>,
    initial_workspace_name: String,
    renamed_from: Option<String>,
}

#[async_trait]
impl LockedWorkingCopy for LockedCommitCloudWorkingCopy {
    fn old_operation_id(&self) -> &OperationId {
        self.locked_inner.old_operation_id()
    }

    fn old_tree(&self) -> &MergedTree {
        self.locked_inner.old_tree()
    }

    async fn snapshot(
        &mut self,
        options: &SnapshotOptions,
    ) -> Result<(MergedTree, SnapshotStats), SnapshotError> {
        self.locked_inner.snapshot(options).await
    }

    async fn check_out(&mut self, commit: &Commit) -> Result<CheckoutStats, CheckoutError> {
        let stats = self.locked_inner.check_out(commit).await?;
        self.current_commit_id = Some(commit.id().as_bytes().to_vec());
        Ok(stats)
    }

    fn rename_workspace(&mut self, new_workspace_name: WorkspaceNameBuf) {
        if new_workspace_name.as_str() != self.initial_workspace_name {
            self.renamed_from = Some(self.initial_workspace_name.clone());
        }
        self.locked_inner.rename_workspace(new_workspace_name);
    }

    async fn reset(&mut self, commit: &Commit) -> Result<(), ResetError> {
        self.locked_inner.reset(commit).await?;
        self.current_commit_id = Some(commit.id().as_bytes().to_vec());
        Ok(())
    }

    async fn recover(&mut self, commit: &Commit) -> Result<(), ResetError> {
        self.locked_inner.recover(commit).await?;
        self.current_commit_id = Some(commit.id().as_bytes().to_vec());
        Ok(())
    }

    fn sparse_patterns(&self) -> Result<&[RepoPathBuf], WorkingCopyStateError> {
        self.locked_inner.sparse_patterns()
    }

    async fn set_sparse_patterns(
        &mut self,
        new_sparse_patterns: Vec<RepoPathBuf>,
    ) -> Result<CheckoutStats, CheckoutError> {
        self.locked_inner
            .set_sparse_patterns(new_sparse_patterns)
            .await
    }

    async fn finish(
        self: Box<Self>,
        new_operation_id: OperationId,
    ) -> Result<Box<dyn WorkingCopy>, WorkingCopyStateError> {
        let config = self.config;
        let state_path = self.state_path;
        let user = self.user;
        let current_commit_id = self.current_commit_id;
        let renamed_from = self.renamed_from;

        let finished_inner = self.locked_inner.finish(new_operation_id.clone()).await?;

        let ws_name = finished_inner.workspace_name().as_str().to_string();
        let op_id_bytes = new_operation_id.as_bytes().to_vec();
        let (tree_id_bytes, tree_ids_bytes) = extract_tree_ids_bytes(finished_inner.as_ref());
        let sparse_patterns = extract_sparse_patterns(finished_inner.as_ref());

        let server_url = config.server_url.clone();
        let repo_id = config.repo_id.clone();
        let user_clone = user.clone();

        if let Err(err) = run_on_tokio(async move {
            // Empty unless this mutation checked out a commit; the server derives `commit_id`
            // from `op_id_bytes` and keeps this value only when it cannot read that operation.
            let commit_id_bytes = current_commit_id.unwrap_or_default();

            let mut client = connect_workspace_client(server_url).await?;
            if let Some(old_name) = renamed_from
                && old_name != ws_name
                && let Err(del_err) = client
                    .delete_workspace(DeleteWorkspaceRequest {
                        repo_id: repo_id.clone(),
                        user: user_clone.clone(),
                        workspace_name: old_name.clone(),
                    })
                    .await
                {
                    tracing::warn!(
                        "Failed to remove renamed workspace '{old_name}' from WorkspaceService: {del_err}"
                    );
                }
            client
                .update_workspace(UpdateWorkspaceRequest {
                    workspace: Some(WorkspaceState {
                        repo_id,
                        user: user_clone,
                        workspace_name: ws_name,
                        commit_id: commit_id_bytes,
                        operation_id: op_id_bytes,
                        tree_id: tree_id_bytes,
                        sparse_patterns,
                        tree_ids: tree_ids_bytes,
                    }),
                })
                .await?;
            Ok(())
        })
        .await
        {
            tracing::warn!("Failed to sync working copy state to WorkspaceService: {err}");
        }

        Ok(Box::new(CommitCloudWorkingCopy::new(
            finished_inner,
            config,
            state_path,
            user,
        )))
    }
}

#[derive(Default)]
pub struct CommitCloudWorkingCopyFactory;

impl CommitCloudWorkingCopyFactory {
    pub fn new() -> Self {
        Self
    }
}

pub fn resolve_workspace_user(settings: &UserSettings) -> String {
    let email = settings.user_email().trim();
    if !email.is_empty() {
        email.to_string()
    } else if let Ok(env_email) = std::env::var("JJ_EMAIL")
        && !env_email.trim().is_empty()
    {
        env_email.trim().to_string()
    } else {
        "anonymous@localhost".to_string()
    }
}

fn extract_tree_ids_bytes(wc: &dyn WorkingCopy) -> (Vec<u8>, Vec<Vec<u8>>) {
    let all_ids: Vec<Vec<u8>> = wc
        .tree()
        .ok()
        .map(|t| {
            t.tree_ids()
                .iter()
                .map(|id| id.as_bytes().to_vec())
                .collect::<Vec<_>>()
        })
        .filter(|ids| !ids.is_empty())
        .unwrap_or_else(|| vec![cc_common::EMPTY_TREE_ID_BYTES.to_vec()]);
    let primary = all_ids
        .first()
        .cloned()
        .unwrap_or_else(|| cc_common::EMPTY_TREE_ID_BYTES.to_vec());
    (primary, all_ids)
}

fn extract_sparse_patterns(wc: &dyn WorkingCopy) -> Vec<String> {
    wc.sparse_patterns()
        .ok()
        .map(|pats| {
            pats.iter()
                .map(|p| p.as_internal_file_string().to_string())
                .collect()
        })
        .unwrap_or_default()
}

impl WorkingCopyFactory for CommitCloudWorkingCopyFactory {
    fn init_working_copy(
        &self,
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        operation_id: OperationId,
        workspace_name: WorkspaceNameBuf,
        settings: &UserSettings,
    ) -> Result<Box<dyn WorkingCopy>, WorkingCopyStateError> {
        let user = resolve_workspace_user(settings);
        let local_factory = LocalWorkingCopyFactory {};
        let local_wc = local_factory.init_working_copy(
            store,
            working_copy_path,
            state_path.clone(),
            operation_id.clone(),
            workspace_name.clone(),
            settings,
        )?;

        let config =
            CommitCloudConfig::load_from_store(&state_path).map_err(|e| WorkingCopyStateError {
                message: format!("Failed to load Commit Cloud config: {e}"),
                err: e,
            })?;

        let ws_name = workspace_name.as_str().to_string();
        let op_id_bytes = operation_id.as_bytes().to_vec();
        let (tree_id_bytes, tree_ids_bytes) = extract_tree_ids_bytes(local_wc.as_ref());
        let sparse_patterns = extract_sparse_patterns(local_wc.as_ref());
        let server_url = config.server_url.clone();
        let repo_id = config.repo_id.clone();
        let user_clone = user.clone();

        if let Err(err) = run_async(move || async move {
            let mut client = connect_workspace_client(server_url).await?;
            client
                .update_workspace(UpdateWorkspaceRequest {
                    workspace: Some(WorkspaceState {
                        repo_id,
                        user: user_clone,
                        workspace_name: ws_name,
                        // Derived by the server from the operation that added this workspace.
                        commit_id: Vec::new(),
                        operation_id: op_id_bytes,
                        tree_id: tree_id_bytes,
                        sparse_patterns,
                        tree_ids: tree_ids_bytes,
                    }),
                })
                .await?;
            Ok(())
        }) {
            tracing::warn!("Failed to register initial workspace in WorkspaceService: {err}");
        }

        Ok(Box::new(CommitCloudWorkingCopy::new(
            local_wc, config, state_path, user,
        )))
    }

    fn load_working_copy(
        &self,
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        settings: &UserSettings,
    ) -> Result<Box<dyn WorkingCopy>, WorkingCopyStateError> {
        let user = resolve_workspace_user(settings);
        let local_factory = LocalWorkingCopyFactory {};
        let local_wc = local_factory.load_working_copy(
            store,
            working_copy_path,
            state_path.clone(),
            settings,
        )?;

        let config =
            CommitCloudConfig::load_from_store(&state_path).map_err(|e| WorkingCopyStateError {
                message: format!("Failed to load Commit Cloud config: {e}"),
                err: e,
            })?;

        Ok(Box::new(CommitCloudWorkingCopy::new(
            local_wc, config, state_path, user,
        )))
    }
}
