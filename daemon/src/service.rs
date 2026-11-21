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

//! Protocol-transparent gRPC handlers for `BackendService`, `OpStoreService`, and
//! `WorkspaceService` backed by [`TwoTierCache`] and [`UpstreamPool`].
//!
//! # Authority and Crash-Safety Ordering
//! Every write RPC (`WriteFile`, `WriteFileStream`, `WriteSymlink`, `WriteTree`,
//! `WriteCommit`, `WriteView`, `WriteOperation`, and `PublishOperation`) succeeds on
//! jj-cc-server before its content-addressed payload is populated into local L1/L2
//! cache, guaranteeing that local cache state never diverges ahead of server durability.
//!
//! # Concurrency and Merge Invariants
//! Cached `TreeEntry` lists are persisted in `RepoPathComponent` UTF-8 byte order
//! (`"foo" < "foo.rs"`) required by `jj_lib::backend::Tree::from_sorted_entries`, while
//! mutable head and workspace RPCs forward straight through to jj-cc-server.
//!
//! # Cross-Process and RPC Boundaries
//! Outbound upstream clients match `cc_common::MAX_GRPC_MESSAGE_BYTES` (`64 MiB`), and
//! all cache reads/writes bind to a [`CacheScope`] derived from `(upstream_url, namespace_id)`
//! (`project_id` on `BackendService`, `repo_id` on `OpStoreService`) with upstream-partitioned
//! shared CAS objects so unauthorized projects cannot probe cached objects (`REQ-CC27`).

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use cc_common::backend::backend_service_client::BackendServiceClient;
use cc_common::backend::backend_service_server::BackendService;
use cc_common::backend::*;
use cc_common::op_store::op_store_service_client::OpStoreServiceClient;
use cc_common::op_store::op_store_service_server::OpStoreService;
use cc_common::op_store::*;
use cc_common::workspace::workspace_service_client::WorkspaceServiceClient;
use cc_common::workspace::workspace_service_server::WorkspaceService;
use cc_common::workspace::*;
use cc_common::{
    EMPTY_TREE_ID_BYTES, MAX_GRPC_MESSAGE_BYTES, ROOT_COMMIT_ID_BYTES, ROOT_OPERATION_ID_BYTES,
    ROOT_VIEW_ID_BYTES,
};
use futures::{Stream, StreamExt};
use prost::Message;
use tonic::metadata::MetadataMap;
use tonic::transport::Channel;
use tonic::{Request, Response, Status, Streaming};

use crate::cache::{CacheScope, ObjectKind, TwoTierCache, now_unix_ms, slice_into_chunks};
use crate::upstream::UpstreamPool;

#[derive(Clone)]
pub struct DaemonProxy {
    inner: Arc<DaemonProxyInner>,
}

pub struct DaemonProxyInner {
    cache: Arc<TwoTierCache>,
    upstream: Arc<UpstreamPool>,
    stats_path: Option<PathBuf>,
    last_stats_flush_ms: AtomicU64,
}

impl std::ops::Deref for DaemonProxy {
    type Target = DaemonProxyInner;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DaemonProxy {
    pub fn new(
        cache: Arc<TwoTierCache>,
        upstream: Arc<UpstreamPool>,
        stats_path: Option<PathBuf>,
    ) -> Self {
        Self {
            inner: Arc::new(DaemonProxyInner {
                cache,
                upstream,
                stats_path,
                last_stats_flush_ms: AtomicU64::new(0),
            }),
        }
    }

    pub fn flush_stats_now(&self) {
        let Some(path) = &self.stats_path else {
            return;
        };
        if let Ok(json) = serde_json::to_vec_pretty(&self.cache.stats()) {
            let tmp = path.with_extension(format!("json.tmp.{}", std::process::id()));
            if std::fs::write(&tmp, json).is_ok() {
                let _ = std::fs::rename(&tmp, path);
            }
        }
    }

    fn maybe_flush_stats(&self) {
        if self.stats_path.is_none() {
            return;
        }
        let now = now_unix_ms();
        let prev = self.last_stats_flush_ms.load(Ordering::Relaxed);
        if now.saturating_sub(prev) >= 25
            && self
                .last_stats_flush_ms
                .compare_exchange(prev, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            self.flush_stats_now();
        }
    }

    fn scope_for(&self, meta: &MetadataMap, repo_id: &str) -> Result<CacheScope, Status> {
        let target_url = self.upstream.resolve_target_url(meta)?;
        Ok(CacheScope::new(target_url, repo_id))
    }

    fn authorize_commit_children(&self, scope: &CacheScope, commit: &Commit) {
        for tree_id in &commit.root_tree_id {
            if tree_id.as_slice() != EMPTY_TREE_ID_BYTES {
                self.cache.authorize_scope(scope, ObjectKind::Tree, tree_id);
            }
        }
    }

    fn authorize_tree_children(&self, scope: &CacheScope, entries: &[TreeEntry]) {
        for entry in entries {
            let Some(val) = entry.value.as_ref().and_then(|v| v.value.as_ref()) else {
                continue;
            };
            match val {
                tree_value::Value::File(f) => {
                    self.cache.authorize_scope(scope, ObjectKind::File, &f.id);
                }
                tree_value::Value::SymlinkId(id) => {
                    self.cache.authorize_scope(scope, ObjectKind::Symlink, id);
                }
                tree_value::Value::TreeId(id) => {
                    if id.as_slice() != EMPTY_TREE_ID_BYTES {
                        self.cache.authorize_scope(scope, ObjectKind::Tree, id);
                    }
                }
                tree_value::Value::GitSubmoduleId(_) | tree_value::Value::ConflictId(_) => {}
            }
        }
    }
}

fn backend_client(channel: Channel) -> BackendServiceClient<Channel> {
    BackendServiceClient::new(channel)
        .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES)
}

fn op_store_client(channel: Channel) -> OpStoreServiceClient<Channel> {
    OpStoreServiceClient::new(channel)
        .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES)
}

fn workspace_client(channel: Channel) -> WorkspaceServiceClient<Channel> {
    WorkspaceServiceClient::new(channel)
        .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES)
}

fn slice_tree_page(
    mut full: ReadTreeResponse,
    page_size: u32,
    page_token: &str,
) -> ReadTreeResponse {
    if page_size == 0 && page_token.is_empty() {
        full.next_page_token.clear();
        return full;
    }
    let start_idx = if page_token.is_empty() {
        0
    } else {
        full.entries
            .partition_point(|e| e.name.as_bytes() <= page_token.as_bytes())
    };
    let slice = &full.entries[start_idx..];
    let limit = if page_size == 0 {
        slice.len()
    } else {
        (page_size as usize).min(slice.len())
    };
    let page_entries = slice[..limit].to_vec();
    let next_page_token = if limit < slice.len() {
        page_entries
            .last()
            .map(|e| e.name.clone())
            .unwrap_or_default()
    } else {
        String::new()
    };
    ReadTreeResponse {
        tree_id: full.tree_id,
        entries: page_entries,
        next_page_token,
    }
}

#[tonic::async_trait]
impl BackendService for DaemonProxy {
    async fn register_project(
        &self,
        request: Request<RegisterProjectRequest>,
    ) -> Result<Response<RegisterProjectResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = backend_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.register_project(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn register_repository(
        &self,
        request: Request<RegisterRepositoryRequest>,
    ) -> Result<Response<RegisterRepositoryResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = backend_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.register_repository(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn read_commit(
        &self,
        request: Request<ReadCommitRequest>,
    ) -> Result<Response<ReadCommitResponse>, Status> {
        let meta = request.metadata().clone();
        let req = request.into_inner();
        let scope = self.scope_for(&meta, &req.project_id)?;
        let commit_id = req.commit_id.clone();
        let upstream = Arc::clone(&self.upstream);
        let raw = self
            .cache
            .get_or_fetch(&scope, ObjectKind::Commit, &commit_id, || async move {
                let channel = upstream.channel_for(&meta).await?;
                let mut client = backend_client(channel);
                let out = upstream.prepare_outbound(&meta, req);
                let resp = client.read_commit(out).await?.into_inner();
                let commit = resp
                    .commit
                    .ok_or_else(|| Status::not_found("upstream returned empty commit"))?;
                Ok(Bytes::from(commit.encode_to_vec()))
            })
            .await?;
        let decoded = Commit::decode(raw)
            .map_err(|e| Status::internal(format!("cached commit decode failed: {e}")))?;
        self.authorize_commit_children(&scope, &decoded);
        self.maybe_flush_stats();
        Ok(Response::new(ReadCommitResponse {
            commit: Some(decoded),
        }))
    }

    async fn write_commit(
        &self,
        request: Request<WriteCommitRequest>,
    ) -> Result<Response<WriteCommitResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let req = request.into_inner();
        let scope = self.scope_for(&meta, &req.project_id)?;
        let commit_clone = req.commit.clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = backend_client(channel);
        let out = self.upstream.prepare_outbound(&meta, req);
        let res = client.write_commit(out).await?.into_inner();
        if let Some(mut c) = commit_clone {
            c.commit_id = res.commit_id.clone();
            self.authorize_commit_children(&scope, &c);
            self.cache.put_write_through(
                &scope,
                ObjectKind::Commit,
                &res.commit_id,
                Bytes::from(c.encode_to_vec()),
            );
        }
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn list_project_commits(
        &self,
        request: Request<ListProjectCommitsRequest>,
    ) -> Result<Response<ListProjectCommitsResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = backend_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.list_project_commits(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn list_commits(
        &self,
        request: Request<ListCommitsRequest>,
    ) -> Result<Response<ListCommitsResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = backend_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.list_commits(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn resolve_commit_prefix(
        &self,
        request: Request<ResolveCommitPrefixRequest>,
    ) -> Result<Response<ResolveCommitPrefixResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = backend_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.resolve_commit_prefix(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn resolve_change_prefix(
        &self,
        request: Request<ResolveChangePrefixRequest>,
    ) -> Result<Response<ResolveChangePrefixResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = backend_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.resolve_change_prefix(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn read_tree(
        &self,
        request: Request<ReadTreeRequest>,
    ) -> Result<Response<ReadTreeResponse>, Status> {
        let meta = request.metadata().clone();
        let req = request.into_inner();
        if req.tree_id.as_slice() == EMPTY_TREE_ID_BYTES {
            self.cache.touch_activity();
            self.cache.record_l1_hit();
            self.maybe_flush_stats();
            return Ok(Response::new(ReadTreeResponse {
                tree_id: req.tree_id,
                entries: Vec::new(),
                next_page_token: String::new(),
            }));
        }
        let scope = self.scope_for(&meta, &req.project_id)?;
        let tree_id = req.tree_id.clone();
        let fetch_tree_id = req.tree_id.clone();
        let project_id = req.project_id.clone();
        let tree_path = req.path.clone();
        let page_size = req.page_size;
        let page_token = req.page_token.clone();
        let upstream = Arc::clone(&self.upstream);
        let raw = self
            .cache
            .get_or_fetch(&scope, ObjectKind::Tree, &tree_id, || async move {
                let channel = upstream.channel_for(&meta).await?;
                let mut client = backend_client(channel);
                let out = upstream.prepare_outbound(
                    &meta,
                    ReadTreeRequest {
                        project_id,
                        path: tree_path,
                        tree_id: fetch_tree_id,
                        page_size: 0,
                        page_token: String::new(),
                    },
                );
                let mut resp = client.read_tree(out).await?.into_inner();
                resp.entries
                    .sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
                resp.next_page_token.clear();
                Ok(Bytes::from(resp.encode_to_vec()))
            })
            .await?;
        let decoded = ReadTreeResponse::decode(raw)
            .map_err(|e| Status::internal(format!("cached tree decode failed: {e}")))?;
        self.authorize_tree_children(&scope, &decoded.entries);
        let sliced = slice_tree_page(decoded, page_size, &page_token);
        self.maybe_flush_stats();
        Ok(Response::new(sliced))
    }

    async fn write_tree(
        &self,
        request: Request<WriteTreeRequest>,
    ) -> Result<Response<WriteTreeResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let req = request.into_inner();
        let scope = self.scope_for(&meta, &req.project_id)?;
        let entries_clone = req.entries.clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = backend_client(channel);
        let out = self.upstream.prepare_outbound(&meta, req);
        let res = client.write_tree(out).await?.into_inner();
        if res.tree_id.as_slice() != EMPTY_TREE_ID_BYTES {
            let mut sorted_entries = entries_clone;
            // Must store entries in strict `RepoPathComponent` UTF-8 byte order
            // (`"foo" < "foo.rs"`) expected by `jj_lib::backend::Tree::from_sorted_entries`.
            sorted_entries.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
            let mut all_file_sizes_known = true;
            for entry in &mut sorted_entries {
                if let Some(tree_value::Value::File(f)) =
                    entry.value.as_mut().and_then(|v| v.value.as_mut())
                    && f.size_bytes.is_none()
                {
                    if let Some(sz) = self.cache.blob_size(&scope, &f.id) {
                        f.size_bytes = Some(sz);
                    } else {
                        all_file_sizes_known = false;
                    }
                }
            }
            self.authorize_tree_children(&scope, &sorted_entries);
            if all_file_sizes_known {
                let cached_tree = ReadTreeResponse {
                    tree_id: res.tree_id.clone(),
                    entries: sorted_entries,
                    next_page_token: String::new(),
                };
                self.cache.put_write_through(
                    &scope,
                    ObjectKind::Tree,
                    &res.tree_id,
                    Bytes::from(cached_tree.encode_to_vec()),
                );
            }
        }
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    type ReadFileStream =
        Pin<Box<dyn Stream<Item = Result<ReadFileResponse, Status>> + Send + 'static>>;

    async fn read_file(
        &self,
        request: Request<ReadFileRequest>,
    ) -> Result<Response<Self::ReadFileStream>, Status> {
        let meta = request.metadata().clone();
        let req = request.into_inner();
        let scope = self.scope_for(&meta, &req.project_id)?;
        let offset = req.offset;
        let max_bytes = req.max_bytes;

        if let Some(chunks) = self
            .cache
            .read_cached_file_range(&scope, &req.file_id, offset, max_bytes)
            .await
            .map_err(|e| Status::internal(format!("local L2 cache read error: {e}")))?
        {
            self.maybe_flush_stats();
            let stream = futures::stream::iter(
                chunks
                    .into_iter()
                    .map(|chunk| Ok(ReadFileResponse { chunk })),
            );
            return Ok(Response::new(Box::pin(stream)));
        }

        // Cold miss: coalesce the full-blob fetch into L1/L2 CAS, then slice the requested range
        // from the fetched bytes. They may be in neither tier (too large for L1, failed L2 write).
        let fetch_file_id = req.file_id.clone();
        let project_id = req.project_id.clone();
        let upstream = Arc::clone(&self.upstream);
        let fetched = self
            .cache
            .get_or_fetch(&scope, ObjectKind::File, &req.file_id, || async move {
                let channel = upstream.channel_for(&meta).await?;
                let mut client = backend_client(channel);
                let out = upstream.prepare_outbound(
                    &meta,
                    ReadFileRequest {
                        project_id,
                        file_id: fetch_file_id,
                        offset: 0,
                        max_bytes: 0,
                    },
                );
                let mut stream = client.read_file(out).await?.into_inner();
                let mut full = Vec::new();
                while let Some(frame) = stream.next().await {
                    full.extend_from_slice(&frame?.chunk);
                }
                Ok(Bytes::from(full))
            })
            .await?;

        let chunks = slice_into_chunks(&fetched, offset, max_bytes);
        self.maybe_flush_stats();
        let stream = futures::stream::iter(
            chunks
                .into_iter()
                .map(|chunk| Ok(ReadFileResponse { chunk })),
        );
        Ok(Response::new(Box::pin(stream)))
    }

    async fn write_file(
        &self,
        request: Request<WriteFileRequest>,
    ) -> Result<Response<WriteFileResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let req = request.into_inner();
        let scope = self.scope_for(&meta, &req.project_id)?;
        let content_bytes = Bytes::from(req.content.clone());
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = backend_client(channel);
        let out = self.upstream.prepare_outbound(&meta, req);
        let res = client.write_file(out).await?.into_inner();
        self.cache
            .put_write_through(&scope, ObjectKind::File, &res.file_id, content_bytes);
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn write_file_stream(
        &self,
        request: Request<Streaming<WriteFileChunk>>,
    ) -> Result<Response<WriteFileResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let mut inbound = request.into_inner();
        let mut project_id = String::new();
        let mut frames = Vec::new();
        let mut assembled = Vec::new();
        while let Some(frame) = inbound.next().await {
            let chunk = frame?;
            if project_id.is_empty() && !chunk.project_id.is_empty() {
                project_id = chunk.project_id.clone();
            }
            assembled.extend_from_slice(&chunk.chunk);
            frames.push(chunk);
        }
        let scope = self.scope_for(&meta, &project_id)?;
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = backend_client(channel);
        let out = self
            .upstream
            .prepare_outbound(&meta, futures::stream::iter(frames));
        let res = client.write_file_stream(out).await?.into_inner();
        self.cache.put_write_through(
            &scope,
            ObjectKind::File,
            &res.file_id,
            Bytes::from(assembled),
        );
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn read_symlink(
        &self,
        request: Request<ReadSymlinkRequest>,
    ) -> Result<Response<ReadSymlinkResponse>, Status> {
        let meta = request.metadata().clone();
        let req = request.into_inner();
        let scope = self.scope_for(&meta, &req.project_id)?;
        let symlink_id = req.symlink_id.clone();
        let upstream = Arc::clone(&self.upstream);
        let raw = self
            .cache
            .get_or_fetch(&scope, ObjectKind::Symlink, &symlink_id, || async move {
                let channel = upstream.channel_for(&meta).await?;
                let mut client = backend_client(channel);
                let out = upstream.prepare_outbound(&meta, req);
                let resp = client.read_symlink(out).await?.into_inner();
                Ok(Bytes::from(resp.target.into_bytes()))
            })
            .await?;
        let target = String::from_utf8(raw.to_vec())
            .map_err(|e| Status::internal(format!("cached symlink UTF-8 error: {e}")))?;
        self.maybe_flush_stats();
        Ok(Response::new(ReadSymlinkResponse { target }))
    }

    async fn write_symlink(
        &self,
        request: Request<WriteSymlinkRequest>,
    ) -> Result<Response<WriteSymlinkResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let req = request.into_inner();
        let scope = self.scope_for(&meta, &req.project_id)?;
        let target_bytes = Bytes::from(req.target.clone().into_bytes());
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = backend_client(channel);
        let out = self.upstream.prepare_outbound(&meta, req);
        let res = client.write_symlink(out).await?.into_inner();
        self.cache
            .put_write_through(&scope, ObjectKind::Symlink, &res.symlink_id, target_bytes);
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }
}

#[tonic::async_trait]
impl OpStoreService for DaemonProxy {
    async fn read_operation(
        &self,
        request: Request<ReadOperationRequest>,
    ) -> Result<Response<ReadOperationResponse>, Status> {
        let meta = request.metadata().clone();
        let req = request.into_inner();
        if req.operation_id.as_slice() == ROOT_OPERATION_ID_BYTES {
            self.cache.touch_activity();
            self.cache.record_l1_hit();
            self.maybe_flush_stats();
            return Ok(Response::new(ReadOperationResponse {
                operation: Some(Operation {
                    view_id: ROOT_VIEW_ID_BYTES.to_vec(),
                    parents: Vec::new(),
                    metadata: Some(OperationMetadata {
                        description: "root()".to_string(),
                        ..Default::default()
                    }),
                    commit_predecessors: Vec::new(),
                    commit_predecessors_set: true,
                }),
            }));
        }
        let scope = self.scope_for(&meta, &req.repo_id)?;
        let op_id = req.operation_id.clone();
        let upstream = Arc::clone(&self.upstream);
        let raw = self
            .cache
            .get_or_fetch(&scope, ObjectKind::Operation, &op_id, || async move {
                let channel = upstream.channel_for(&meta).await?;
                let mut client = op_store_client(channel);
                let out = upstream.prepare_outbound(&meta, req);
                let resp = client.read_operation(out).await?.into_inner();
                let op = resp
                    .operation
                    .ok_or_else(|| Status::not_found("upstream returned empty operation"))?;
                Ok(Bytes::from(op.encode_to_vec()))
            })
            .await?;
        let decoded = Operation::decode(raw)
            .map_err(|e| Status::internal(format!("cached operation decode failed: {e}")))?;
        self.maybe_flush_stats();
        Ok(Response::new(ReadOperationResponse {
            operation: Some(decoded),
        }))
    }

    async fn write_operation(
        &self,
        request: Request<WriteOperationRequest>,
    ) -> Result<Response<WriteOperationResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let req = request.into_inner();
        let scope = self.scope_for(&meta, &req.repo_id)?;
        let op_clone = req.operation.clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = op_store_client(channel);
        let out = self.upstream.prepare_outbound(&meta, req);
        let res = client.write_operation(out).await?.into_inner();
        if let Some(op) = op_clone {
            self.cache.put_write_through(
                &scope,
                ObjectKind::Operation,
                &res.operation_id,
                Bytes::from(op.encode_to_vec()),
            );
        }
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn resolve_operation_prefix(
        &self,
        request: Request<ResolveOperationPrefixRequest>,
    ) -> Result<Response<ResolveOperationPrefixResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = op_store_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.resolve_operation_prefix(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn read_view(
        &self,
        request: Request<ReadViewRequest>,
    ) -> Result<Response<ReadViewResponse>, Status> {
        let meta = request.metadata().clone();
        let req = request.into_inner();
        if req.view_id.as_slice() == ROOT_VIEW_ID_BYTES {
            self.cache.touch_activity();
            self.cache.record_l1_hit();
            self.maybe_flush_stats();
            return Ok(Response::new(ReadViewResponse {
                view_id: req.view_id,
                view: Some(View {
                    head_ids: vec![ROOT_COMMIT_ID_BYTES.to_vec()],
                    ..Default::default()
                }),
            }));
        }
        let scope = self.scope_for(&meta, &req.repo_id)?;
        let view_id = req.view_id.clone();
        let upstream = Arc::clone(&self.upstream);
        let raw = self
            .cache
            .get_or_fetch(&scope, ObjectKind::View, &view_id, || async move {
                let channel = upstream.channel_for(&meta).await?;
                let mut client = op_store_client(channel);
                let out = upstream.prepare_outbound(&meta, req);
                let resp = client.read_view(out).await?.into_inner();
                let view = resp
                    .view
                    .ok_or_else(|| Status::not_found("upstream returned empty view"))?;
                Ok(Bytes::from(view.encode_to_vec()))
            })
            .await?;
        let decoded = View::decode(raw)
            .map_err(|e| Status::internal(format!("cached view decode failed: {e}")))?;
        self.maybe_flush_stats();
        Ok(Response::new(ReadViewResponse {
            view_id,
            view: Some(decoded),
        }))
    }

    async fn write_view(
        &self,
        request: Request<WriteViewRequest>,
    ) -> Result<Response<WriteViewResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let req = request.into_inner();
        let scope = self.scope_for(&meta, &req.repo_id)?;
        let view_clone = req.view.clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = op_store_client(channel);
        let out = self.upstream.prepare_outbound(&meta, req);
        let res = client.write_view(out).await?.into_inner();
        if let Some(v) = view_clone {
            self.cache.put_write_through(
                &scope,
                ObjectKind::View,
                &res.view_id,
                Bytes::from(v.encode_to_vec()),
            );
        }
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn get_op_heads(
        &self,
        request: Request<GetOpHeadsRequest>,
    ) -> Result<Response<GetOpHeadsResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = op_store_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.get_op_heads(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn update_op_heads(
        &self,
        request: Request<UpdateOpHeadsRequest>,
    ) -> Result<Response<UpdateOpHeadsResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = op_store_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.update_op_heads(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn reconcile_op_heads(
        &self,
        request: Request<ReconcileOpHeadsRequest>,
    ) -> Result<Response<ReconcileOpHeadsResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = op_store_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.reconcile_op_heads(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn publish_operation(
        &self,
        request: Request<PublishOperationRequest>,
    ) -> Result<Response<PublishOperationResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let req = request.into_inner();
        let scope = self.scope_for(&meta, &req.repo_id)?;
        let view_clone = req.view.clone();
        let op_clone = req.operation.clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = op_store_client(channel);
        let out = self.upstream.prepare_outbound(&meta, req);
        let res = client.publish_operation(out).await?.into_inner();
        if let Some(v) = view_clone {
            self.cache.put_write_through(
                &scope,
                ObjectKind::View,
                &res.view_id,
                Bytes::from(v.encode_to_vec()),
            );
        }
        if let Some(mut op) = op_clone {
            op.view_id = res.view_id.clone();
            self.cache.put_write_through(
                &scope,
                ObjectKind::Operation,
                &res.operation_id,
                Bytes::from(op.encode_to_vec()),
            );
        }
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }
}

#[tonic::async_trait]
impl WorkspaceService for DaemonProxy {
    async fn get_workspace(
        &self,
        request: Request<GetWorkspaceRequest>,
    ) -> Result<Response<GetWorkspaceResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = workspace_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.get_workspace(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn update_workspace(
        &self,
        request: Request<UpdateWorkspaceRequest>,
    ) -> Result<Response<UpdateWorkspaceResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = workspace_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.update_workspace(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn list_workspaces(
        &self,
        request: Request<ListWorkspacesRequest>,
    ) -> Result<Response<ListWorkspacesResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = workspace_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.list_workspaces(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn check_working_copy_changes(
        &self,
        request: Request<CheckWorkingCopyChangesRequest>,
    ) -> Result<Response<CheckWorkingCopyChangesResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = workspace_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.check_working_copy_changes(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }

    async fn delete_workspace(
        &self,
        request: Request<DeleteWorkspaceRequest>,
    ) -> Result<Response<DeleteWorkspaceResponse>, Status> {
        self.cache.touch_activity();
        self.cache.record_upstream_rpc();
        let meta = request.metadata().clone();
        let channel = self.upstream.channel_for(&meta).await?;
        let mut client = workspace_client(channel);
        let out = self.upstream.prepare_outbound(&meta, request.into_inner());
        let res = client.delete_workspace(out).await?.into_inner();
        self.maybe_flush_stats();
        Ok(Response::new(res))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use cc_common::backend::backend_service_server::BackendServiceServer;
    use sha1::{Digest, Sha1};

    use super::*;
    use crate::upstream::UPSTREAM_HEADER;

    /// Upstream that serves one file and rejects every other RPC.
    struct FileOnlyUpstream {
        content: Bytes,
    }

    fn unused<T>() -> Result<Response<T>, Status> {
        Err(Status::unimplemented("not used by this test"))
    }

    #[tonic::async_trait]
    impl BackendService for FileOnlyUpstream {
        async fn register_project(
            &self,
            _: Request<RegisterProjectRequest>,
        ) -> Result<Response<RegisterProjectResponse>, Status> {
            unused()
        }
        async fn register_repository(
            &self,
            _: Request<RegisterRepositoryRequest>,
        ) -> Result<Response<RegisterRepositoryResponse>, Status> {
            unused()
        }
        async fn read_commit(
            &self,
            _: Request<ReadCommitRequest>,
        ) -> Result<Response<ReadCommitResponse>, Status> {
            unused()
        }
        async fn write_commit(
            &self,
            _: Request<WriteCommitRequest>,
        ) -> Result<Response<WriteCommitResponse>, Status> {
            unused()
        }
        async fn list_project_commits(
            &self,
            _: Request<ListProjectCommitsRequest>,
        ) -> Result<Response<ListProjectCommitsResponse>, Status> {
            unused()
        }
        async fn list_commits(
            &self,
            _: Request<ListCommitsRequest>,
        ) -> Result<Response<ListCommitsResponse>, Status> {
            unused()
        }
        async fn resolve_commit_prefix(
            &self,
            _: Request<ResolveCommitPrefixRequest>,
        ) -> Result<Response<ResolveCommitPrefixResponse>, Status> {
            unused()
        }
        async fn resolve_change_prefix(
            &self,
            _: Request<ResolveChangePrefixRequest>,
        ) -> Result<Response<ResolveChangePrefixResponse>, Status> {
            unused()
        }
        async fn read_tree(
            &self,
            _: Request<ReadTreeRequest>,
        ) -> Result<Response<ReadTreeResponse>, Status> {
            unused()
        }
        async fn write_tree(
            &self,
            _: Request<WriteTreeRequest>,
        ) -> Result<Response<WriteTreeResponse>, Status> {
            unused()
        }

        type ReadFileStream =
            Pin<Box<dyn Stream<Item = Result<ReadFileResponse, Status>> + Send + 'static>>;

        async fn read_file(
            &self,
            _: Request<ReadFileRequest>,
        ) -> Result<Response<Self::ReadFileStream>, Status> {
            let frames: Vec<_> = self
                .content
                .chunks(1024)
                .map(|chunk| {
                    Ok(ReadFileResponse {
                        chunk: chunk.to_vec(),
                    })
                })
                .collect();
            Ok(Response::new(Box::pin(futures::stream::iter(frames))))
        }
        async fn write_file(
            &self,
            _: Request<WriteFileRequest>,
        ) -> Result<Response<WriteFileResponse>, Status> {
            unused()
        }
        async fn write_file_stream(
            &self,
            _: Request<Streaming<WriteFileChunk>>,
        ) -> Result<Response<WriteFileResponse>, Status> {
            unused()
        }
        async fn read_symlink(
            &self,
            _: Request<ReadSymlinkRequest>,
        ) -> Result<Response<ReadSymlinkResponse>, Status> {
            unused()
        }
        async fn write_symlink(
            &self,
            _: Request<WriteSymlinkRequest>,
        ) -> Result<Response<WriteSymlinkResponse>, Status> {
            unused()
        }
    }

    #[tokio::test]
    async fn cold_read_file_serves_fetched_bytes_that_neither_cache_tier_kept() {
        let content = Bytes::from((0..10_000u32).map(|i| (i % 251) as u8).collect::<Vec<u8>>());
        let file_id = {
            let mut hasher = Sha1::new();
            hasher.update(format!("blob {}\0", content.len()).as_bytes());
            hasher.update(&content);
            hasher.finalize().to_vec()
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_url = format!("http://{}", listener.local_addr().unwrap());
        let incoming =
            tonic::transport::server::TcpIncoming::from_listener(listener, true, None).unwrap();
        let upstream = FileOnlyUpstream {
            content: content.clone(),
        };
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(BackendServiceServer::new(upstream))
                .serve_with_incoming(incoming),
        );

        // L1 holds at most 4 bytes and `objects` is a regular file, so every L2 write fails
        // and the fetched blob ends up in neither tier.
        let dir = tempfile::tempdir().unwrap();
        let cache = Arc::new(TwoTierCache::new(dir.path().to_path_buf(), 4).unwrap());
        fs::remove_dir(dir.path().join("objects")).unwrap();
        fs::write(dir.path().join("objects"), b"").unwrap();
        let proxy = DaemonProxy::new(Arc::clone(&cache), Arc::new(UpstreamPool::new(None)), None);

        for (offset, max_bytes) in [(0u64, 0u64), (100, 4096)] {
            let mut request = Request::new(ReadFileRequest {
                project_id: "repo".to_string(),
                file_id: file_id.clone(),
                offset,
                max_bytes,
            });
            request
                .metadata_mut()
                .insert(UPSTREAM_HEADER, upstream_url.parse().unwrap());
            let frames: Vec<_> = BackendService::read_file(&proxy, request)
                .await
                .unwrap()
                .into_inner()
                .collect()
                .await;
            let body: Vec<u8> = frames
                .into_iter()
                .flat_map(|frame| frame.unwrap().chunk)
                .collect();
            let end = if max_bytes == 0 {
                content.len()
            } else {
                (offset + max_bytes) as usize
            };
            assert_eq!(body, &content[offset as usize..end]);
        }
        assert_eq!(cache.stats().l2_write_errors, 2);
    }
}
