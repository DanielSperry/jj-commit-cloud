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

//! gRPC `OpStoreService` implementation backed by [`CompositeStore`].
//!
//! # Authority and Crash-Safety Ordering
//! Clients write `View` objects (`WriteView`) before referencing them in `Operation`
//! nodes (`WriteOperation`), and persist `Operation` nodes before advancing repository
//! heads via `UpdateOpHeads`. `CompositeStore` enforces these dependencies before
//! exposing new operation heads to readers.
//!
//! # Concurrency and Merge Invariants
//! `UpdateOpHeads` delegates to `MetadataStore::update_op_heads` under serializable
//! isolation, removing superseded `old_op_head_ids` while preserving concurrent sibling
//! heads for lock-free client-side reconciliation.
//!
//! # Cross-Process and RPC Boundaries
//! Synthesizes the all-zero root `Operation` and root `View` on read without requiring
//! pre-seeded rows, and resolves hex prefixes via byte-range scans in `ResolveOperationPrefix`.

use std::collections::HashMap;
use std::sync::Arc;

use tracing::info;

use cc_common::op_store::op_store_service_server::OpStoreService;
use cc_common::op_store::*;

use crate::hash_utils::{hash_operation, hash_view};
use crate::reconcile_op_heads::reconcile_repo_op_heads;
use crate::store::{CompositeStore, IdPrefixResolution};

#[derive(Clone)]
pub struct CommitCloudOpStoreService {
    store: Arc<CompositeStore>,
}

impl CommitCloudOpStoreService {
    pub fn new(store: Arc<CompositeStore>) -> Self {
        Self { store }
    }
}

#[tonic::async_trait]
impl OpStoreService for CommitCloudOpStoreService {
    async fn read_operation(
        &self,
        request: tonic::Request<ReadOperationRequest>,
    ) -> Result<tonic::Response<ReadOperationResponse>, tonic::Status> {
        let req = request.into_inner();
        info!(
            "Reading operation: {} for repo: {}",
            hex::encode(&req.operation_id),
            req.repo_id
        );

        if let Some(op) = self
            .store
            .get_operation(&req.repo_id, &req.operation_id)
            .await?
        {
            return Ok(tonic::Response::new(ReadOperationResponse {
                operation: Some(op),
            }));
        }

        if req.operation_id == cc_common::ROOT_OPERATION_ID_BYTES {
            let root_op = Operation {
                view_id: cc_common::ROOT_VIEW_ID_BYTES.to_vec(),
                parents: vec![],
                metadata: Some(OperationMetadata {
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
            return Ok(tonic::Response::new(ReadOperationResponse {
                operation: Some(root_op),
            }));
        }

        Err(tonic::Status::not_found(format!(
            "operation should have been present in cloud database: {} in repo: {}",
            hex::encode(&req.operation_id),
            req.repo_id
        )))
    }

    async fn write_operation(
        &self,
        request: tonic::Request<WriteOperationRequest>,
    ) -> Result<tonic::Response<WriteOperationResponse>, tonic::Status> {
        let req = request.into_inner();
        let op = req.operation.ok_or_else(|| {
            tonic::Status::invalid_argument("request should have contained an operation object")
        })?;

        info!("Writing operation for repo: {}", req.repo_id);
        let op_id = hash_operation(&op);
        self.store.put_operation(&req.repo_id, &op_id, &op).await?;

        Ok(tonic::Response::new(WriteOperationResponse {
            operation_id: op_id,
        }))
    }

    async fn resolve_operation_prefix(
        &self,
        request: tonic::Request<ResolveOperationPrefixRequest>,
    ) -> Result<tonic::Response<ResolveOperationPrefixResponse>, tonic::Status> {
        let req = request.into_inner();
        let resolution = self
            .store
            .resolve_operation_prefix(&req.repo_id, &req.hex_prefix)
            .await?;

        let (status, matched_operation_id) = match resolution {
            IdPrefixResolution::NoMatch => {
                (PrefixResolutionStatus::PrefixResolutionNoMatch, vec![])
            }
            IdPrefixResolution::SingleMatch(id) => {
                (PrefixResolutionStatus::PrefixResolutionSingleMatch, id)
            }
            IdPrefixResolution::AmbiguousMatch => (
                PrefixResolutionStatus::PrefixResolutionAmbiguousMatch,
                vec![],
            ),
        };

        Ok(tonic::Response::new(ResolveOperationPrefixResponse {
            status: status as i32,
            matched_operation_id,
        }))
    }

    async fn read_view(
        &self,
        request: tonic::Request<ReadViewRequest>,
    ) -> Result<tonic::Response<ReadViewResponse>, tonic::Status> {
        let req = request.into_inner();
        info!(
            "Reading view: {} for repo: {}",
            hex::encode(&req.view_id),
            req.repo_id
        );

        if let Some(v) = self.store.get_view(&req.repo_id, &req.view_id).await? {
            return Ok(tonic::Response::new(ReadViewResponse {
                view_id: req.view_id,
                view: Some(v),
            }));
        }

        if req.view_id == cc_common::ROOT_VIEW_ID_BYTES {
            let root_view = View {
                head_ids: vec![cc_common::ROOT_COMMIT_ID_BYTES.to_vec()],
                wc_commit_ids: HashMap::new(),
                local_bookmarks: HashMap::new(),
                remote_bookmarks: HashMap::new(),
            };
            return Ok(tonic::Response::new(ReadViewResponse {
                view_id: req.view_id,
                view: Some(root_view),
            }));
        }

        Err(tonic::Status::not_found(format!(
            "view should have been present in cloud database: {} in repo: {}",
            hex::encode(&req.view_id),
            req.repo_id
        )))
    }

    async fn write_view(
        &self,
        request: tonic::Request<WriteViewRequest>,
    ) -> Result<tonic::Response<WriteViewResponse>, tonic::Status> {
        let req = request.into_inner();
        let view = req.view.ok_or_else(|| {
            tonic::Status::invalid_argument("request should have contained a view object")
        })?;

        info!("Writing view for repo: {}", req.repo_id);
        let view_id = hash_view(&view);
        self.store.put_view(&req.repo_id, &view_id, &view).await?;

        Ok(tonic::Response::new(WriteViewResponse { view_id }))
    }

    async fn get_op_heads(
        &self,
        request: tonic::Request<GetOpHeadsRequest>,
    ) -> Result<tonic::Response<GetOpHeadsResponse>, tonic::Status> {
        let req = request.into_inner();
        info!("Get op heads for repo: {}", req.repo_id);
        let heads = self.store.get_op_heads(&req.repo_id).await?;
        Ok(tonic::Response::new(GetOpHeadsResponse {
            op_head_ids: heads,
        }))
    }

    async fn update_op_heads(
        &self,
        request: tonic::Request<UpdateOpHeadsRequest>,
    ) -> Result<tonic::Response<UpdateOpHeadsResponse>, tonic::Status> {
        let req = request.into_inner();
        info!("Update op heads for repo: {}", req.repo_id);
        let current_heads = self
            .store
            .update_op_heads(&req.repo_id, &req.old_op_head_ids, &req.new_op_head_id)
            .await?;

        Ok(tonic::Response::new(UpdateOpHeadsResponse {
            current_op_head_ids: current_heads,
        }))
    }

    async fn reconcile_op_heads(
        &self,
        request: tonic::Request<ReconcileOpHeadsRequest>,
    ) -> Result<tonic::Response<ReconcileOpHeadsResponse>, tonic::Status> {
        let req = request.into_inner();
        info!("Reconcile op heads for repo: {}", req.repo_id);
        let op_head = reconcile_repo_op_heads(self.store.clone(), &req.repo_id).await?;
        Ok(tonic::Response::new(ReconcileOpHeadsResponse { op_head }))
    }

    async fn publish_operation(
        &self,
        request: tonic::Request<PublishOperationRequest>,
    ) -> Result<tonic::Response<PublishOperationResponse>, tonic::Status> {
        let req = request.into_inner();
        let view = req.view.ok_or_else(|| {
            tonic::Status::invalid_argument("PublishOperationRequest must include view")
        })?;
        let mut operation = req.operation.ok_or_else(|| {
            tonic::Status::invalid_argument("PublishOperationRequest must include operation")
        })?;

        let view_id = hash_view(&view);
        if operation.view_id.is_empty() {
            operation.view_id = view_id.clone();
        } else if operation.view_id != view_id {
            return Err(tonic::Status::invalid_argument(
                "operation.view_id does not match canonical hash of supplied view",
            ));
        }

        let operation_id = hash_operation(&operation);
        let current_op_head_ids = self
            .store
            .publish_operation(
                &req.repo_id,
                &view_id,
                &view,
                &operation_id,
                &operation,
                &req.old_op_head_ids,
            )
            .await?;

        Ok(tonic::Response::new(PublishOperationResponse {
            view_id,
            operation_id,
            current_op_head_ids,
        }))
    }
}
