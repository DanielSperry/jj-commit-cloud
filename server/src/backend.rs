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

//! gRPC `BackendService` implementation backed by [`CompositeStore`].
//!
//! # Authority and Crash-Safety Ordering
//! `WriteFile` and `WriteFileStream` persist file payloads to `BlobStore` before
//! publishing `BlobRecord` rows in `MetadataStore`. `WriteTree` and `WriteCommit`
//! delegate batched dependency checks to `CompositeStore` before exposing parent
//! directory or commit graph records, and `WriteCommit` verifies canonical Git SHA-1
//! digest parity on any client-supplied `commit_id` or `raw_git_commit`.
//!
//! # Concurrency and Merge Invariants
//! All object mutations (`WriteFile`, `WriteFileStream`, `WriteSymlink`, `WriteTree`,
//! `WriteCommit`) are idempotent content-addressed writes keyed by 20-byte Git SHA-1
//! digests, allowing parallel client uploads without server-side locks.
//!
//! # Cross-Process and RPC Boundaries
//! `ReadFile` and `WriteFileStream` stream payloads in 64 KiB frames (`FILE_STREAM_CHUNK_SIZE`),
//! while `ReadTree` returns entries sorted in UTF-8 `RepoPathComponent` byte order with
//! `File.size_bytes` populated from `MetadataStore` and optional cursor pagination.

use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use tokio_stream::StreamExt;
use tracing::info;

use cc_common::backend::backend_service_server::BackendService;
use cc_common::backend::*;

use crate::hash_utils::{
    compute_git_blob_hash, compute_git_tree_hash, verify_or_compute_git_commit_hash,
};
use crate::store::{ChangePrefixResolution, CompositeStore, IdPrefixResolution};

#[derive(Clone)]
pub struct CommitCloudBackendService {
    store: Arc<CompositeStore>,
}

impl CommitCloudBackendService {
    pub fn new(store: Arc<CompositeStore>) -> Self {
        Self { store }
    }
}

#[tonic::async_trait]
impl BackendService for CommitCloudBackendService {
    async fn register_project(
        &self,
        request: tonic::Request<RegisterProjectRequest>,
    ) -> Result<tonic::Response<RegisterProjectResponse>, tonic::Status> {
        let req = request.into_inner();
        let project_id = req
            .project_id
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        info!("Registering project: {}", project_id);
        self.store
            .register_project(&project_id, req.name.as_deref())
            .await?;
        Ok(tonic::Response::new(RegisterProjectResponse { project_id }))
    }

    async fn register_repository(
        &self,
        request: tonic::Request<RegisterRepositoryRequest>,
    ) -> Result<tonic::Response<RegisterRepositoryResponse>, tonic::Status> {
        let req = request.into_inner();
        match HashAlgorithm::try_from(req.hash_algorithm).unwrap_or(HashAlgorithm::Unspecified) {
            HashAlgorithm::Unspecified | HashAlgorithm::Sha1 => {}
            HashAlgorithm::Sha256 => {
                return Err(tonic::Status::unimplemented(
                    "SHA-256 repositories are not yet enabled in Milestone 1",
                ));
            }
        }
        let repo_id = req
            .repo_id
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let requested_project =
            (!req.project_id.trim().is_empty()).then_some(req.project_id.as_str());
        info!(
            "Registering repository: {} (requested project: {:?})",
            repo_id, requested_project
        );
        let project_id = self
            .store
            .register_repo_in_project(&repo_id, requested_project, req.name.as_deref())
            .await?;
        Ok(tonic::Response::new(RegisterRepositoryResponse {
            repo_id,
            project_id,
        }))
    }

    async fn read_commit(
        &self,
        request: tonic::Request<ReadCommitRequest>,
    ) -> Result<tonic::Response<ReadCommitResponse>, tonic::Status> {
        let req = request.into_inner();
        let commit = self
            .store
            .get_commit(&req.project_id, &req.commit_id)
            .await?;
        Ok(tonic::Response::new(ReadCommitResponse {
            commit: Some(commit),
        }))
    }

    async fn write_commit(
        &self,
        request: tonic::Request<WriteCommitRequest>,
    ) -> Result<tonic::Response<WriteCommitResponse>, tonic::Status> {
        let req = request.into_inner();
        let mut commit = req.commit.ok_or_else(|| {
            tonic::Status::invalid_argument("request should have contained commit data")
        })?;
        if commit.parent_commit_ids.len() > 1
            && commit
                .parent_commit_ids
                .iter()
                .any(|p| p.as_slice() == cc_common::ROOT_COMMIT_ID_BYTES)
        {
            return Err(tonic::Status::invalid_argument(
                "The Commit Cloud backend does not support creating merge commits with the root commit as one of the parents.",
            ));
        }

        let commit_id = verify_or_compute_git_commit_hash(&mut commit)
            .map_err(tonic::Status::invalid_argument)?;
        commit.commit_id = commit_id.clone();
        info!(
            "Writing commit {} for project {}",
            hex::encode(&commit_id),
            req.project_id
        );

        self.store
            .put_commit(&req.project_id, &commit_id, &commit)
            .await?;
        Ok(tonic::Response::new(WriteCommitResponse { commit_id }))
    }

    async fn list_project_commits(
        &self,
        request: tonic::Request<ListProjectCommitsRequest>,
    ) -> Result<tonic::Response<ListProjectCommitsResponse>, tonic::Status> {
        let req = request.into_inner();
        let commit_ids = self.store.list_project_commits(&req.project_id).await?;
        Ok(tonic::Response::new(ListProjectCommitsResponse {
            commit_ids,
        }))
    }

    async fn list_commits(
        &self,
        request: tonic::Request<ListCommitsRequest>,
    ) -> Result<tonic::Response<ListCommitsResponse>, tonic::Status> {
        let req = request.into_inner();
        let (commit_ids, next_page_token, sync_token) = self
            .store
            .list_commits_with_sync(
                &req.project_id,
                &req.page_token,
                req.page_size,
                &req.since_token,
            )
            .await?;
        Ok(tonic::Response::new(ListCommitsResponse {
            commit_ids,
            next_page_token,
            sync_token,
        }))
    }

    async fn resolve_commit_prefix(
        &self,
        request: tonic::Request<ResolveCommitPrefixRequest>,
    ) -> Result<tonic::Response<ResolveCommitPrefixResponse>, tonic::Status> {
        let req = request.into_inner();
        let resolution = self
            .store
            .resolve_commit_prefix(&req.project_id, &req.hex_prefix)
            .await?;
        let (status, matched_commit_id) = match resolution {
            IdPrefixResolution::NoMatch => (ObjectPrefixStatus::ObjectPrefixNoMatch, Vec::new()),
            IdPrefixResolution::SingleMatch(id) => {
                (ObjectPrefixStatus::ObjectPrefixSingleMatch, id)
            }
            IdPrefixResolution::AmbiguousMatch => {
                (ObjectPrefixStatus::ObjectPrefixAmbiguousMatch, Vec::new())
            }
        };
        Ok(tonic::Response::new(ResolveCommitPrefixResponse {
            status: status as i32,
            matched_commit_id,
        }))
    }

    async fn resolve_change_prefix(
        &self,
        request: tonic::Request<ResolveChangePrefixRequest>,
    ) -> Result<tonic::Response<ResolveChangePrefixResponse>, tonic::Status> {
        let req = request.into_inner();
        let resolution = self
            .store
            .resolve_change_prefix(&req.project_id, &req.prefix)
            .await?;
        let (status, matched_change_id, commit_ids) = match resolution {
            ChangePrefixResolution::NoMatch => (
                ObjectPrefixStatus::ObjectPrefixNoMatch,
                Vec::new(),
                Vec::new(),
            ),
            ChangePrefixResolution::SingleMatch {
                change_id,
                commit_ids,
            } => (
                ObjectPrefixStatus::ObjectPrefixSingleMatch,
                change_id,
                commit_ids,
            ),
            ChangePrefixResolution::AmbiguousMatch => (
                ObjectPrefixStatus::ObjectPrefixAmbiguousMatch,
                Vec::new(),
                Vec::new(),
            ),
        };
        Ok(tonic::Response::new(ResolveChangePrefixResponse {
            status: status as i32,
            matched_change_id,
            commit_ids,
        }))
    }

    async fn read_tree(
        &self,
        request: tonic::Request<ReadTreeRequest>,
    ) -> Result<tonic::Response<ReadTreeResponse>, tonic::Status> {
        let req = request.into_inner();
        let (entries, next_page_token) = self
            .store
            .get_tree_page(
                &req.project_id,
                &req.tree_id,
                &req.page_token,
                req.page_size,
            )
            .await?;
        Ok(tonic::Response::new(ReadTreeResponse {
            tree_id: req.tree_id,
            entries,
            next_page_token,
        }))
    }

    async fn write_tree(
        &self,
        request: tonic::Request<WriteTreeRequest>,
    ) -> Result<tonic::Response<WriteTreeResponse>, tonic::Status> {
        let req = request.into_inner();
        let tree_id =
            compute_git_tree_hash(&req.entries).map_err(tonic::Status::invalid_argument)?;
        self.store
            .put_tree(&req.project_id, &tree_id, &req.entries)
            .await?;
        Ok(tonic::Response::new(WriteTreeResponse { tree_id }))
    }

    type ReadFileStream =
        tokio_stream::wrappers::ReceiverStream<Result<ReadFileResponse, tonic::Status>>;

    async fn read_file(
        &self,
        request: tonic::Request<ReadFileRequest>,
    ) -> Result<tonic::Response<Self::ReadFileStream>, tonic::Status> {
        let req = request.into_inner();
        let max_bytes = (req.max_bytes > 0).then_some(req.max_bytes);
        let content = self
            .store
            .get_file_range(&req.project_id, &req.file_id, req.offset, max_bytes)
            .await?;

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            if content.is_empty() {
                let _ = tx.send(Ok(ReadFileResponse { chunk: Vec::new() })).await;
                return;
            }
            for chunk in content.chunks(cc_common::FILE_STREAM_CHUNK_SIZE) {
                if tx
                    .send(Ok(ReadFileResponse {
                        chunk: chunk.to_vec(),
                    }))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });

        Ok(tonic::Response::new(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
    }

    async fn write_file(
        &self,
        request: tonic::Request<WriteFileRequest>,
    ) -> Result<tonic::Response<WriteFileResponse>, tonic::Status> {
        let req = request.into_inner();
        let file_id = compute_git_blob_hash(&req.content);
        self.store
            .put_file(&req.project_id, &file_id, Bytes::from(req.content))
            .await?;
        Ok(tonic::Response::new(WriteFileResponse { file_id }))
    }

    async fn write_file_stream(
        &self,
        request: tonic::Request<tonic::Streaming<WriteFileChunk>>,
    ) -> Result<tonic::Response<WriteFileResponse>, tonic::Status> {
        let mut stream = request.into_inner();
        let mut project_id: Option<String> = None;
        let mut payload = BytesMut::new();

        while let Some(frame) = stream.next().await {
            let frame = frame?;
            match &project_id {
                None => {
                    if frame.project_id.is_empty() {
                        return Err(tonic::Status::invalid_argument(
                            "first WriteFileChunk frame must specify project_id",
                        ));
                    }
                    project_id = Some(frame.project_id);
                }
                Some(expected) if !frame.project_id.is_empty() && &frame.project_id != expected => {
                    return Err(tonic::Status::invalid_argument(
                        "WriteFileChunk project_id changed mid-stream",
                    ));
                }
                Some(_) => {}
            }
            payload.extend_from_slice(&frame.chunk);
        }

        let project_id = project_id.ok_or_else(|| {
            tonic::Status::invalid_argument("WriteFileStream requires at least one chunk frame")
        })?;
        let content = payload.freeze();
        let file_id = compute_git_blob_hash(&content);
        self.store.put_file(&project_id, &file_id, content).await?;
        Ok(tonic::Response::new(WriteFileResponse { file_id }))
    }

    async fn read_symlink(
        &self,
        request: tonic::Request<ReadSymlinkRequest>,
    ) -> Result<tonic::Response<ReadSymlinkResponse>, tonic::Status> {
        let req = request.into_inner();
        let target = self
            .store
            .get_symlink(&req.project_id, &req.symlink_id)
            .await?;
        Ok(tonic::Response::new(ReadSymlinkResponse { target }))
    }

    async fn write_symlink(
        &self,
        request: tonic::Request<WriteSymlinkRequest>,
    ) -> Result<tonic::Response<WriteSymlinkResponse>, tonic::Status> {
        let req = request.into_inner();
        let symlink_id = compute_git_blob_hash(req.target.as_bytes());
        self.store
            .put_symlink(&req.project_id, &symlink_id, &req.target)
            .await?;
        Ok(tonic::Response::new(WriteSymlinkResponse { symlink_id }))
    }
}

#[cfg(test)]
mod tests {
    use tonic::Request;

    use super::*;
    use crate::hash_utils::serialize_canonical_git_commit;

    fn make_service() -> CommitCloudBackendService {
        CommitCloudBackendService::new(Arc::new(CompositeStore::in_memory()))
    }

    fn base_commit() -> Commit {
        Commit {
            commit_id: vec![],
            change_id: vec![0x42; 16],
            parent_commit_ids: vec![cc_common::ROOT_COMMIT_ID_BYTES.to_vec()],
            root_tree_id: vec![cc_common::EMPTY_TREE_ID_BYTES.to_vec()],
            description: "honest commit\n".to_string(),
            author: Some(Signature {
                name: "Alice".to_string(),
                email: "alice@example.com".to_string(),
                timestamp: Some(Timestamp {
                    millis_since_epoch: 1_730_000_000_000,
                    tz_offset: 0,
                }),
            }),
            committer: Some(Signature {
                name: "Alice".to_string(),
                email: "alice@example.com".to_string(),
                timestamp: Some(Timestamp {
                    millis_since_epoch: 1_730_000_000_000,
                    tz_offset: 0,
                }),
            }),
            predecessors: vec![],
            conflict_labels: vec![],
            secure_sig: None,
            raw_git_commit: vec![],
            external_ids: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_write_tree_and_write_commit_reject_malformed_inputs_with_invalid_argument() {
        let svc = make_service();
        svc.register_repository(Request::new(RegisterRepositoryRequest {
            project_id: "repo-1".to_string(),
            repo_id: Some("repo-1".to_string()),
            name: None,
            hash_algorithm: HashAlgorithm::Sha1 as i32,
        }))
        .await
        .unwrap();

        let file_id = svc
            .write_file(Request::new(WriteFileRequest {
                project_id: "repo-1".to_string(),
                content: b"hello\n".to_vec(),
            }))
            .await
            .unwrap()
            .into_inner()
            .file_id;

        let valid_entry = TreeEntry {
            name: "hello.txt".to_string(),
            value: Some(TreeValue {
                value: Some(tree_value::Value::File(File {
                    id: file_id,
                    executable: false,
                    copy_id: vec![],
                    size_bytes: None,
                })),
            }),
        };
        let tree_id = svc
            .write_tree(Request::new(WriteTreeRequest {
                project_id: "repo-1".to_string(),
                path: String::new(),
                entries: vec![valid_entry.clone()],
            }))
            .await
            .unwrap()
            .into_inner()
            .tree_id;

        // Trying to inject a zero-byte submodule entry alongside `valid_entry` is rejected
        // with INVALID_ARGUMENT rather than dropped from the hash and persisted under `tree_id`.
        let err = svc
            .write_tree(Request::new(WriteTreeRequest {
                project_id: "repo-1".to_string(),
                path: String::new(),
                entries: vec![
                    valid_entry,
                    TreeEntry {
                        name: "phantom".to_string(),
                        value: Some(TreeValue {
                            value: Some(tree_value::Value::GitSubmoduleId(vec![])),
                        }),
                    },
                ],
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);

        let stored_entries = svc
            .read_tree(Request::new(ReadTreeRequest {
                project_id: "repo-1".to_string(),
                path: String::new(),
                tree_id,
                page_size: 0,
                page_token: String::new(),
            }))
            .await
            .unwrap()
            .into_inner()
            .entries;
        assert_eq!(stored_entries.len(), 1);
        assert_eq!(stored_entries[0].name, "hello.txt");

        // Malformed commits (empty root_tree_id, NUL/angle-bracket in author) return
        // INVALID_ARGUMENT instead of panicking or poisoning repository readers.
        let mut empty_trees = base_commit();
        empty_trees.root_tree_id.clear();
        let err = svc
            .write_commit(Request::new(WriteCommitRequest {
                project_id: "repo-1".to_string(),
                commit: Some(empty_trees),
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);

        let mut bad_author = base_commit();
        bad_author.author.as_mut().unwrap().name = "Bad <Actor>".to_string();
        let err = svc
            .write_commit(Request::new(WriteCommitRequest {
                project_id: "repo-1".to_string(),
                commit: Some(bad_author),
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn test_write_commit_blocks_signature_spoofing_and_raw_git_commit_forging() {
        let svc = make_service();
        svc.register_repository(Request::new(RegisterRepositoryRequest {
            project_id: "repo-sig".to_string(),
            repo_id: Some("repo-sig".to_string()),
            name: None,
            hash_algorithm: HashAlgorithm::Sha1 as i32,
        }))
        .await
        .unwrap();

        let mut commit_a = base_commit();
        let data_a = serialize_canonical_git_commit(&commit_a, false).unwrap();
        let sig_a = b"-----BEGIN SSH SIGNATURE-----\nsigA\n-----END SSH SIGNATURE-----\n".to_vec();
        commit_a.secure_sig = Some(SecureSig {
            data: data_a.clone(),
            sig: sig_a.clone(),
        });

        let id_a = svc
            .write_commit(Request::new(WriteCommitRequest {
                project_id: "repo-sig".to_string(),
                commit: Some(commit_a),
            }))
            .await
            .unwrap()
            .into_inner()
            .commit_id;
        let read_a = svc
            .read_commit(Request::new(ReadCommitRequest {
                project_id: "repo-sig".to_string(),
                commit_id: id_a,
            }))
            .await
            .unwrap()
            .into_inner()
            .commit
            .unwrap();

        // Replaying commit A's `(data_A, sig_A)` onto commit M is rejected.
        let mut commit_m = base_commit();
        commit_m.description = "tampered commit\n".to_string();
        commit_m.secure_sig = read_a.secure_sig.clone();
        let err = svc
            .write_commit(Request::new(WriteCommitRequest {
                project_id: "repo-sig".to_string(),
                commit: Some(commit_m.clone()),
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);

        // Sending `raw_git_commit` bytes from commit A with commit M's description is rejected.
        let raw_a = serialize_canonical_git_commit(&read_a, true).unwrap();
        commit_m.raw_git_commit = raw_a;
        commit_m.commit_id.clear();
        let err = svc
            .write_commit(Request::new(WriteCommitRequest {
                project_id: "repo-sig".to_string(),
                commit: Some(commit_m),
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn test_shared_project_object_rpcs_accept_both_project_id_and_repo_id() {
        let svc = make_service();
        svc.register_project(Request::new(RegisterProjectRequest {
            project_id: Some("shared-proj".to_string()),
            name: Some("Shared Project".to_string()),
        }))
        .await
        .unwrap();
        svc.register_repository(Request::new(RegisterRepositoryRequest {
            project_id: "shared-proj".to_string(),
            repo_id: Some("actor-1".to_string()),
            name: None,
            hash_algorithm: HashAlgorithm::Sha1 as i32,
        }))
        .await
        .unwrap();

        // Write commit addressing `shared-proj` directly (where no repo named `shared-proj` exists).
        let cid = svc
            .write_commit(Request::new(WriteCommitRequest {
                project_id: "shared-proj".to_string(),
                commit: Some(base_commit()),
            }))
            .await
            .unwrap()
            .into_inner()
            .commit_id;

        // Read and list via both `shared-proj` and `actor-1`.
        let by_proj = svc
            .read_commit(Request::new(ReadCommitRequest {
                project_id: "shared-proj".to_string(),
                commit_id: cid.clone(),
            }))
            .await
            .unwrap()
            .into_inner()
            .commit
            .unwrap();
        let by_repo = svc
            .read_commit(Request::new(ReadCommitRequest {
                project_id: "actor-1".to_string(),
                commit_id: cid.clone(),
            }))
            .await
            .unwrap()
            .into_inner()
            .commit
            .unwrap();
        assert_eq!(by_proj.commit_id, by_repo.commit_id);

        let listed_proj = svc
            .list_project_commits(Request::new(ListProjectCommitsRequest {
                project_id: "shared-proj".to_string(),
            }))
            .await
            .unwrap()
            .into_inner()
            .commit_ids;
        let listed_repo = svc
            .list_commits(Request::new(ListCommitsRequest {
                project_id: "actor-1".to_string(),
                page_token: Vec::new(),
                page_size: 100,
                since_token: Vec::new(),
            }))
            .await
            .unwrap()
            .into_inner()
            .commit_ids;
        assert_eq!(listed_proj, vec![cid.clone()]);
        assert_eq!(listed_repo, vec![cid]);
    }
}
