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

//! gRPC `WorkspaceService` implementation persisting per-workspace checkout state in
//! [`CompositeStore`].
//!
//! # Authority and Crash-Safety Ordering
//! Clients update `WorkspaceState` (`commit_id`, `operation_id`, `tree_id` / `tree_ids`,
//! and `sparse_patterns`) after committing the underlying `Operation`, `Commit`, and `Tree`
//! objects in `OpStoreService` and `BackendService`. `UpdateWorkspace` derives `commit_id`
//! from `operation_id` (`View.wc_commit_ids[workspace_name]`) and keeps the client value only
//! when that operation or view cannot be read.
//!
//! # Concurrency and Merge Invariants
//! Keyed by `(repo_id, user, workspace_name)`. Upserts are idempotent across retries, and
//! `DeleteWorkspace` is advisory because active workspaces upsert their record on finish.
//!
//! # Cross-Process and RPC Boundaries
//! `CheckWorkingCopyChanges` resolves `repo_id -> project_id` through `CompositeStore`
//! and compares `WorkspaceState.tree_ids` against `Commit.root_tree_id` (preserving full
//! `Merge<TreeId>` algebraic conflict sides) without client I/O.

use std::sync::Arc;

use cc_common::workspace::workspace_service_server::WorkspaceService;
use cc_common::workspace::*;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::store::CompositeStore;

#[derive(Clone)]
pub struct CommitCloudWorkspaceService {
    store: Arc<CompositeStore>,
}

impl CommitCloudWorkspaceService {
    pub fn new(store: Arc<CompositeStore>) -> Self {
        Self { store }
    }
}

#[tonic::async_trait]
impl WorkspaceService for CommitCloudWorkspaceService {
    async fn get_workspace(
        &self,
        request: Request<GetWorkspaceRequest>,
    ) -> Result<Response<GetWorkspaceResponse>, Status> {
        let req = request.into_inner();
        let workspace = self
            .store
            .get_workspace(&req.repo_id, &req.user, &req.workspace_name)
            .await?;
        Ok(Response::new(GetWorkspaceResponse { workspace }))
    }

    async fn update_workspace(
        &self,
        request: Request<UpdateWorkspaceRequest>,
    ) -> Result<Response<UpdateWorkspaceResponse>, Status> {
        let req = request.into_inner();
        let mut ws = req
            .workspace
            .ok_or_else(|| Status::invalid_argument("WorkspaceState payload is required"))?;
        // A snapshot rewrites `@` without a checkout, so the client may not know the new commit.
        // Take it from the recorded operation's view and keep the client value only when that
        // operation, its view, or the workspace entry cannot be read.
        if let Ok(Some(op)) = self
            .store
            .get_operation(&ws.repo_id, &ws.operation_id)
            .await
            && let Ok(Some(view)) = self.store.get_view(&ws.repo_id, &op.view_id).await
            && let Some(wc_commit_id) = view.wc_commit_ids.get(&ws.workspace_name)
        {
            ws.commit_id = wc_commit_id.clone();
        }
        info!(
            "Update workspace '{}' for user '{}' in repo '{}'",
            ws.workspace_name, ws.user, ws.repo_id
        );
        self.store.put_workspace(&ws).await?;
        Ok(Response::new(UpdateWorkspaceResponse { success: true }))
    }

    async fn list_workspaces(
        &self,
        request: Request<ListWorkspacesRequest>,
    ) -> Result<Response<ListWorkspacesResponse>, Status> {
        let req = request.into_inner();
        let workspaces = self.store.list_workspaces(&req.repo_id).await?;
        Ok(Response::new(ListWorkspacesResponse { workspaces }))
    }

    async fn check_working_copy_changes(
        &self,
        request: Request<CheckWorkingCopyChangesRequest>,
    ) -> Result<Response<CheckWorkingCopyChangesResponse>, Status> {
        let req = request.into_inner();
        let ws = self
            .store
            .get_workspace(&req.repo_id, &req.user, &req.workspace_name)
            .await?
            .ok_or_else(|| {
                Status::not_found(format!(
                    "workspace '{}' for user '{}' not found in repo '{}'",
                    req.workspace_name, req.user, req.repo_id
                ))
            })?;

        let current_tree_ids = if !ws.tree_ids.is_empty() {
            ws.tree_ids
        } else if !ws.tree_id.is_empty() {
            vec![ws.tree_id.clone()]
        } else {
            vec![cc_common::EMPTY_TREE_ID_BYTES.to_vec()]
        };

        let commit_tree_ids = if ws.commit_id.is_empty()
            || ws.commit_id.as_slice() == cc_common::ROOT_COMMIT_ID_BYTES
        {
            vec![cc_common::EMPTY_TREE_ID_BYTES.to_vec()]
        } else {
            let commit = self.store.get_commit(&req.repo_id, &ws.commit_id).await?;
            if commit.root_tree_id.is_empty() {
                vec![cc_common::EMPTY_TREE_ID_BYTES.to_vec()]
            } else {
                commit.root_tree_id
            }
        };

        let current_tree_id = current_tree_ids
            .first()
            .cloned()
            .unwrap_or_else(|| cc_common::EMPTY_TREE_ID_BYTES.to_vec());
        let commit_tree_id = commit_tree_ids
            .first()
            .cloned()
            .unwrap_or_else(|| cc_common::EMPTY_TREE_ID_BYTES.to_vec());
        let has_changes = current_tree_ids != commit_tree_ids;

        Ok(Response::new(CheckWorkingCopyChangesResponse {
            has_changes,
            current_tree_id,
            commit_tree_id,
            current_tree_ids,
            commit_tree_ids,
        }))
    }

    async fn delete_workspace(
        &self,
        request: Request<DeleteWorkspaceRequest>,
    ) -> Result<Response<DeleteWorkspaceResponse>, Status> {
        let req = request.into_inner();
        info!(
            "Delete workspace '{}' for user '{}' in repo '{}'",
            req.workspace_name, req.user, req.repo_id
        );
        let success = self
            .store
            .delete_workspace(&req.repo_id, &req.user, &req.workspace_name)
            .await?;
        Ok(Response::new(DeleteWorkspaceResponse { success }))
    }
}
