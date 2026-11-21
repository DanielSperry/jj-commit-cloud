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

//! `jj_lib::op_heads_store::OpHeadsStore` adapter over `OpStoreService` gRPC RPCs.
//!
//! # Authority and Crash-Safety Ordering
//! `update_op_heads` is called only after the target `Operation` and its `View` have
//! been persisted via [`crate::cc_op_store::CommitCloudOpStore`].
//!
//! # Concurrency and Merge Invariants
//! `lock()` is an intentional client-side no-op. `get_op_heads` invokes `ReconcileOpHeads`
//! so jj-cc-server merges divergent operation heads without cross-client locks. When
//! divergent operations carry cross-branch commit rewrites that require descendant
//! rebasing, `ReconcileOpHeads` returns an empty `op_head` and the client falls back to
//! `GetOpHeads` so `jj_lib` resolves the rewrites against `DefaultIndexStore`.
//! `UpdateOpHeads` removes `old_ids` and appends `new_id` while retaining concurrent
//! sibling heads.
//!
//! # Cross-Process and RPC Boundaries
//! Connects to `OpStoreService` using the `server_url` and `repo_id` stored in
//! `.jj/repo/store/config.toml`.

use std::fmt::Debug;
use std::path::Path;

use async_trait::async_trait;
use cc_common::op_store::*;
use jj_lib::object_id::ObjectId;
use jj_lib::op_heads_store::{OpHeadsStore, OpHeadsStoreError, OpHeadsStoreLock};
use jj_lib::op_store::OperationId;

use crate::util::{CommitCloudConfig, connect_op_store_client, run_on_tokio};

#[derive(Debug)]
pub struct CommitCloudOpHeadsStore {
    server_url: String,
    repo_id: String,
}

struct CommitCloudOpHeadsStoreNoLock;
impl OpHeadsStoreLock for CommitCloudOpHeadsStoreNoLock {}

impl CommitCloudOpHeadsStore {
    pub fn name() -> &'static str {
        "commit_cloud"
    }

    pub fn new(repo_id: String, server_url: String) -> Self {
        Self {
            server_url,
            repo_id,
        }
    }

    pub fn load(store_path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config = CommitCloudConfig::load_from_store(store_path)?;
        Ok(Self::new(config.repo_id, config.server_url))
    }

    pub async fn get_op_heads_without_reconciliation(
        &self,
    ) -> Result<Vec<OperationId>, OpHeadsStoreError> {
        let server_url = self.server_url.clone();
        let repo_id = self.repo_id.clone();

        run_on_tokio(async move {
            let mut client = connect_op_store_client(server_url).await?;
            let response = client.get_op_heads(GetOpHeadsRequest { repo_id }).await?;
            let head_ids: Vec<OperationId> = response
                .into_inner()
                .op_head_ids
                .into_iter()
                .map(|b| OperationId::from_bytes(&b))
                .collect();
            Ok(head_ids)
        })
        .await
        .map_err(OpHeadsStoreError::Read)
    }

    pub async fn get_op_heads_with_reconciliation(
        &self,
    ) -> Result<Vec<OperationId>, OpHeadsStoreError> {
        let server_url = self.server_url.clone();
        let repo_id = self.repo_id.clone();

        run_on_tokio(async move {
            let mut client = connect_op_store_client(server_url).await?;
            let response = client
                .reconcile_op_heads(ReconcileOpHeadsRequest {
                    repo_id: repo_id.clone(),
                })
                .await?;
            let op_head = response.into_inner().op_head;
            if op_head.is_empty() {
                let heads_res = client.get_op_heads(GetOpHeadsRequest { repo_id }).await?;
                return Ok(heads_res
                    .into_inner()
                    .op_head_ids
                    .into_iter()
                    .map(|b| OperationId::from_bytes(&b))
                    .collect());
            }
            Ok(vec![OperationId::from_bytes(&op_head)])
        })
        .await
        .map_err(OpHeadsStoreError::Read)
    }
}

#[async_trait]
impl OpHeadsStore for CommitCloudOpHeadsStore {
    fn name(&self) -> &str {
        Self::name()
    }

    async fn update_op_heads(
        &self,
        old_ids: &[OperationId],
        new_id: &OperationId,
    ) -> Result<(), OpHeadsStoreError> {
        let server_url = self.server_url.clone();
        let repo_id = self.repo_id.clone();
        let old_op_head_ids = old_ids.iter().map(|id| id.as_bytes().to_vec()).collect();
        let new_op_head_id = new_id.as_bytes().to_vec();
        let target_new_id = new_id.clone();

        run_on_tokio(async move {
            let mut client = connect_op_store_client(server_url).await?;
            client
                .update_op_heads(UpdateOpHeadsRequest {
                    repo_id,
                    old_op_head_ids,
                    new_op_head_id,
                })
                .await?;
            Ok(())
        })
        .await
        .map_err(|e| OpHeadsStoreError::Write {
            new_op_id: target_new_id,
            source: e,
        })
    }

    async fn get_op_heads(&self) -> Result<Vec<OperationId>, OpHeadsStoreError> {
        self.get_op_heads_with_reconciliation().await
    }

    async fn lock(&self) -> Result<Box<dyn OpHeadsStoreLock + '_>, OpHeadsStoreError> {
        Ok(Box::new(CommitCloudOpHeadsStoreNoLock))
    }
}
