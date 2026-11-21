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

//! `jj_lib::backend::Backend` adapter over `BackendService` gRPC RPCs.
//!
//! # Authority and Crash-Safety Ordering
//! File payloads stream through `WriteFileStream` in 64 KiB chunks (`FILE_STREAM_CHUNK_SIZE`)
//! and symlink targets persist via `WriteSymlink` before referencing `Tree` and `Commit`
//! records are uploaded. Repository attachment (`init_with_options` with `create = false`)
//! validates remote `repo_id` registration before writing `.jj/repo/store/config.toml`.
//!
//! # Concurrency and Merge Invariants
//! All `Backend` writes are immutable, content-addressed, and safe under concurrent
//! execution. Conflicted `Merge<TreeId>` root trees, `conflict_labels`, and `SecureSig`
//! payloads round-trip without loss.
//!
//! # Cross-Process and RPC Boundaries
//! Translates gRPC `Code::NotFound` responses into `BackendError::ObjectNotFound` and
//! avoids unary 4 MiB frame limits on both `read_file` and `write_file`.

#![allow(elided_lifetimes_in_paths)]

use std::fmt::Debug;
use std::fs;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use async_trait::async_trait;
use futures::AsyncReadExt as _;
use futures::StreamExt as _;
use futures::stream::{self, BoxStream};
use jj_lib::backend::*;
use jj_lib::index::Index;
use jj_lib::object_id::ObjectId;
use jj_lib::repo_path::{RepoPath, RepoPathBuf, RepoPathComponentBuf};

use crate::util::{CommitCloudConfig, run_async, run_on_tokio};

/// Default client concurrency reported to `jj_lib` so tree and file hydration
/// pipelines up to 16 concurrent RPCs over the pooled HTTP/2 channel.
pub const CLIENT_BACKEND_CONCURRENCY: usize = 16;

#[derive(Debug)]
pub struct CommitCloudBackend {
    server_url: String,
    repo_id: String,
    project_id: String,
    prefer_unary_write_file: AtomicBool,
    root_commit_id: CommitId,
    root_change_id: ChangeId,
    empty_tree_id: TreeId,
}

impl CommitCloudBackend {
    pub fn name() -> &'static str {
        "commit_cloud"
    }

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    pub fn init(
        store_path: &Path,
        server_url: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::init_with_options(store_path, server_url, None, None, true)
    }

    pub fn init_with_options(
        store_path: &Path,
        server_url: &str,
        name: Option<&str>,
        explicit_repo_id: Option<&str>,
        create: bool,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::init_in_project(store_path, server_url, name, None, explicit_repo_id, create)
    }

    pub fn init_in_project(
        store_path: &Path,
        server_url: &str,
        name: Option<&str>,
        explicit_project_id: Option<&str>,
        explicit_repo_id: Option<&str>,
        create: bool,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let root_commit_id = CommitId::from_bytes(&cc_common::ROOT_COMMIT_ID_BYTES);
        let root_change_id = ChangeId::from_bytes(&cc_common::ROOT_CHANGE_ID_BYTES);
        let empty_tree_id = TreeId::from_hex(cc_common::EMPTY_TREE_ID_HEX);

        let server_url_cloned = server_url.to_string();
        let name_opt = name.map(ToOwned::to_owned);
        let project_id_opt = explicit_project_id.map(ToOwned::to_owned);
        let repo_id_opt = explicit_repo_id.map(ToOwned::to_owned);

        let (repo_id, project_id) = match (create, repo_id_opt) {
            (false, Some(existing_repo_id)) => run_async(move || async move {
                let mut op_client = crate::util::connect_op_store_client(server_url_cloned).await?;
                op_client
                    .get_op_heads(tonic::Request::new(
                        cc_common::op_store::GetOpHeadsRequest {
                            repo_id: existing_repo_id.clone(),
                        },
                    ))
                    .await?;
                let proj = project_id_opt.unwrap_or_else(|| existing_repo_id.clone());
                Ok((existing_repo_id, proj))
            })?,
            (_, maybe_repo_id) => run_async(move || async move {
                let mut client = crate::util::connect_backend_client(server_url_cloned).await?;
                let effective_repo_id = maybe_repo_id
                    .filter(|id| !id.trim().is_empty())
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                let effective_project_id = project_id_opt
                    .filter(|id| !id.trim().is_empty())
                    .unwrap_or_else(|| effective_repo_id.clone());
                let register_repo_response = client
                    .register_repository(tonic::Request::new(
                        cc_common::backend::RegisterRepositoryRequest {
                            project_id: effective_project_id,
                            repo_id: Some(effective_repo_id),
                            name: name_opt,
                            hash_algorithm: cc_common::backend::HashAlgorithm::Sha1 as i32,
                        },
                    ))
                    .await?
                    .into_inner();
                Ok((
                    register_repo_response.repo_id,
                    register_repo_response.project_id,
                ))
            })?,
        };

        let config_path = store_path.join("config.toml");
        let config = CommitCloudConfig {
            server_url: server_url.to_string(),
            repo_id: repo_id.clone(),
            project_id: Some(project_id.clone()),
        };
        fs::write(&config_path, toml::to_string_pretty(&config)?)?;

        Ok(Self {
            server_url: server_url.to_string(),
            repo_id,
            project_id,
            prefer_unary_write_file: AtomicBool::new(false),
            root_commit_id,
            root_change_id,
            empty_tree_id,
        })
    }

    pub fn load(store_path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config = CommitCloudConfig::load_from_store(store_path)?;
        let project_id = config.effective_project_id().to_string();
        let root_commit_id = CommitId::from_bytes(&cc_common::ROOT_COMMIT_ID_BYTES);
        let root_change_id = ChangeId::from_bytes(&cc_common::ROOT_CHANGE_ID_BYTES);
        let empty_tree_id = TreeId::from_hex(cc_common::EMPTY_TREE_ID_HEX);
        Ok(Self {
            server_url: config.server_url,
            repo_id: config.repo_id,
            project_id,
            prefer_unary_write_file: AtomicBool::new(false),
            root_commit_id,
            root_change_id,
            empty_tree_id,
        })
    }
}

fn signature_to_proto(sig: &Signature) -> cc_common::backend::Signature {
    cc_common::backend::Signature {
        name: sig.name.clone(),
        email: sig.email.clone(),
        timestamp: Some(cc_common::backend::Timestamp {
            millis_since_epoch: sig.timestamp.timestamp.0,
            tz_offset: sig.timestamp.tz_offset,
        }),
    }
}

fn signature_from_proto(proto: cc_common::backend::Signature) -> Signature {
    let ts = proto.timestamp.unwrap_or_default();
    Signature {
        name: proto.name,
        email: proto.email,
        timestamp: Timestamp {
            timestamp: MillisSinceEpoch(ts.millis_since_epoch),
            tz_offset: ts.tz_offset,
        },
    }
}

pub fn commit_to_proto(commit: &Commit) -> cc_common::backend::Commit {
    cc_common::backend::Commit {
        commit_id: vec![],
        change_id: commit.change_id.to_bytes().to_vec(),
        parent_commit_ids: commit
            .parents
            .iter()
            .map(|id| id.to_bytes().to_vec())
            .collect(),
        root_tree_id: commit
            .root_tree
            .iter()
            .map(|id| id.to_bytes().to_vec())
            .collect(),
        description: commit.description.clone(),
        author: Some(signature_to_proto(&commit.author)),
        committer: Some(signature_to_proto(&commit.committer)),
        predecessors: commit
            .predecessors
            .iter()
            .map(|id| id.to_bytes().to_vec())
            .collect(),
        conflict_labels: commit.conflict_labels.as_slice().to_owned(),
        secure_sig: commit
            .secure_sig
            .as_ref()
            .map(|s| cc_common::backend::SecureSig {
                data: s.data.clone(),
                sig: s.sig.clone(),
            }),
        raw_git_commit: Vec::new(),
        external_ids: Default::default(),
    }
}

/// Formats the canonical unsigned Git commit payload matching `server::hash_utils` and
/// invokes `sign_with` when cryptographic commit signing is requested.
pub fn apply_commit_signing(
    commit: &mut Commit,
    sign_with: Option<&mut SigningFn>,
) -> BackendResult<()> {
    const EMPTY_STRING_PLACEHOLDER: &str = "JJ_EMPTY_STRING";
    const REVERSE_HEX_CHARS: &[u8; 16] = b"zyxwvutsrqponmlk";
    const FORWARD_HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

    let Some(sign_fn) = sign_with else {
        return Ok(());
    };
    if commit.secure_sig.is_some() {
        return Ok(());
    }

    use gix::objs::WriteTo as _;
    let proto = commit_to_proto(commit);
    let tree_id = proto
        .root_tree_id
        .first()
        .and_then(|id| gix::hash::ObjectId::try_from(id.as_slice()).ok())
        .unwrap_or_else(|| gix::hash::ObjectId::empty_tree(gix::hash::Kind::Sha1));
    let parents: Vec<gix::hash::ObjectId> = proto
        .parent_commit_ids
        .iter()
        .filter(|id| id.as_slice() != cc_common::ROOT_COMMIT_ID_BYTES)
        .filter_map(|id| gix::hash::ObjectId::try_from(id.as_slice()).ok())
        .collect();

    let mut extra_headers = Vec::new();
    if proto.conflict_labels.len() > 1 {
        let mut joined = proto.conflict_labels.join("\n");
        joined.push('\n');
        extra_headers.push(("jj:conflict-labels".into(), joined.into()));
    }
    if proto.root_tree_id.len() > 1 {
        let trees_hex: Vec<String> = proto
            .root_tree_id
            .iter()
            .map(|id| {
                let bytes: Vec<u8> = id
                    .iter()
                    .flat_map(|b| {
                        [
                            FORWARD_HEX_CHARS[usize::from(b >> 4)],
                            FORWARD_HEX_CHARS[usize::from(b & 0x0f)],
                        ]
                    })
                    .collect();
                String::from_utf8(bytes).unwrap()
            })
            .collect();
        extra_headers.push(("jj:trees".into(), trees_hex.join(" ").into()));
    }
    if !proto.change_id.is_empty() {
        let rev_bytes: Vec<u8> = proto
            .change_id
            .iter()
            .flat_map(|b| {
                [
                    REVERSE_HEX_CHARS[usize::from(b >> 4)],
                    REVERSE_HEX_CHARS[usize::from(b & 0x0f)],
                ]
            })
            .collect();
        extra_headers.push((
            "change-id".into(),
            String::from_utf8(rev_bytes).unwrap().into(),
        ));
    }

    let sig_to_git = |s: &Signature| -> gix::actor::Signature {
        let name = if s.name.is_empty() {
            EMPTY_STRING_PLACEHOLDER
        } else {
            &s.name
        };
        let email = if s.email.is_empty() {
            EMPTY_STRING_PLACEHOLDER
        } else {
            &s.email
        };
        gix::actor::Signature {
            name: name.into(),
            email: email.into(),
            time: gix::date::Time::new(
                s.timestamp.timestamp.0.div_euclid(1000),
                s.timestamp.tz_offset * 60,
            ),
        }
    };

    let gix_commit = gix::objs::Commit {
        tree: tree_id,
        parents: parents.into(),
        author: sig_to_git(&commit.author),
        committer: sig_to_git(&commit.committer),
        encoding: None,
        extra_headers,
        message: commit.description.as_bytes().into(),
    };
    let mut unsigned_data = Vec::new();
    gix_commit
        .write_to(&mut unsigned_data)
        .map_err(|e| BackendError::Other(Box::new(e)))?;

    let sig = sign_fn(&unsigned_data).map_err(|e| BackendError::Other(Box::new(e)))?;
    commit.secure_sig = Some(SecureSig {
        data: unsigned_data,
        sig,
    });
    Ok(())
}

pub fn commit_from_proto(proto_commit: cc_common::backend::Commit) -> Commit {
    let author = signature_from_proto(proto_commit.author.unwrap_or_default());
    let committer = signature_from_proto(proto_commit.committer.unwrap_or_default());

    let merge_builder: jj_lib::merge::MergeBuilder<_> = proto_commit
        .root_tree_id
        .into_iter()
        .map(|b| TreeId::from_bytes(&b))
        .collect();

    let root_tree = merge_builder.build();
    let conflict_labels =
        jj_lib::conflict_labels::ConflictLabels::from_vec(proto_commit.conflict_labels)
            .into_merge();

    Commit {
        parents: proto_commit
            .parent_commit_ids
            .iter()
            .map(|b| CommitId::from_bytes(b))
            .collect(),
        predecessors: proto_commit
            .predecessors
            .iter()
            .map(|b| CommitId::from_bytes(b))
            .collect(),
        root_tree,
        change_id: ChangeId::from_bytes(&proto_commit.change_id),
        description: proto_commit.description,
        author,
        committer,
        conflict_labels,
        secure_sig: proto_commit.secure_sig.map(|s| SecureSig {
            data: s.data,
            sig: s.sig,
        }),
    }
}

pub fn tree_entry_to_proto(
    entry: &jj_lib::backend::TreeEntry,
) -> BackendResult<cc_common::backend::TreeEntry> {
    let value = match entry.value() {
        TreeValue::File {
            id,
            executable,
            copy_id,
        } => cc_common::backend::TreeValue {
            value: Some(cc_common::backend::tree_value::Value::File(
                cc_common::backend::File {
                    id: id.to_bytes().to_vec(),
                    executable: *executable,
                    copy_id: copy_id.to_bytes().to_vec(),
                    size_bytes: None,
                },
            )),
        },
        TreeValue::Symlink(id) => cc_common::backend::TreeValue {
            value: Some(cc_common::backend::tree_value::Value::SymlinkId(
                id.to_bytes().to_vec(),
            )),
        },
        TreeValue::Tree(id) => cc_common::backend::TreeValue {
            value: Some(cc_common::backend::tree_value::Value::TreeId(
                id.to_bytes().to_vec(),
            )),
        },
        TreeValue::GitSubmodule(id) => cc_common::backend::TreeValue {
            value: Some(cc_common::backend::tree_value::Value::GitSubmoduleId(
                id.to_bytes().to_vec(),
            )),
        },
    };

    Ok(cc_common::backend::TreeEntry {
        name: entry.name().as_internal_str().to_string(),
        value: Some(value),
    })
}

pub fn tree_entry_from_proto(
    proto_entry: cc_common::backend::TreeEntry,
) -> Result<(RepoPathComponentBuf, TreeValue), Box<dyn std::error::Error + Send + Sync>> {
    let component = RepoPathComponentBuf::new(proto_entry.name)?;
    let proto_val = proto_entry
        .value
        .ok_or("tree entry should have contained a TreeValue")?;
    let val = match proto_val
        .value
        .ok_or("TreeValue should have contained an inner value")?
    {
        cc_common::backend::tree_value::Value::File(f) => TreeValue::File {
            id: FileId::from_bytes(&f.id),
            executable: f.executable,
            copy_id: CopyId::from_bytes(if f.copy_id.len() == cc_common::COMMIT_ID_LENGTH {
                &f.copy_id
            } else {
                &cc_common::ROOT_COMMIT_ID_BYTES
            }),
        },
        cc_common::backend::tree_value::Value::SymlinkId(id) => {
            TreeValue::Symlink(SymlinkId::from_bytes(&id))
        }
        cc_common::backend::tree_value::Value::TreeId(id) => {
            TreeValue::Tree(TreeId::from_bytes(&id))
        }
        cc_common::backend::tree_value::Value::GitSubmoduleId(id) => {
            TreeValue::GitSubmodule(CommitId::from_bytes(&id))
        }
        cc_common::backend::tree_value::Value::ConflictId(_) => {
            return Err("tree entry should not have contained a ConflictId".into());
        }
    };
    Ok((component, val))
}

#[async_trait]
impl Backend for CommitCloudBackend {
    fn name(&self) -> &str {
        Self::name()
    }

    fn commit_id_length(&self) -> usize {
        cc_common::COMMIT_ID_LENGTH
    }

    fn change_id_length(&self) -> usize {
        cc_common::CHANGE_ID_LENGTH
    }

    fn root_commit_id(&self) -> &CommitId {
        &self.root_commit_id
    }

    fn root_change_id(&self) -> &ChangeId {
        &self.root_change_id
    }

    fn empty_tree_id(&self) -> &TreeId {
        &self.empty_tree_id
    }

    fn concurrency(&self) -> usize {
        CLIENT_BACKEND_CONCURRENCY
    }

    async fn read_file(
        &self,
        _path: &RepoPath,
        id: &FileId,
    ) -> BackendResult<Pin<Box<dyn futures::AsyncRead + Send>>> {
        let server_url = self.server_url.clone();
        let project_id = self.project_id.clone();
        let file_id_bytes = id.to_bytes().to_vec();
        let file_id_hex = id.hex();

        let content = run_on_tokio(async move {
            let mut client = crate::util::connect_backend_client(server_url).await?;
            let res = client
                .read_file(tonic::Request::new(cc_common::backend::ReadFileRequest {
                    project_id,
                    file_id: file_id_bytes,
                    offset: 0,
                    max_bytes: 0,
                }))
                .await;

            let mut stream = match res {
                Ok(r) => r.into_inner(),
                Err(status) if status.code() == tonic::Code::NotFound => {
                    return Err(Box::new(BackendError::ObjectNotFound {
                        object_type: "file".into(),
                        hash: file_id_hex,
                        source: status.into(),
                    })
                        as Box<dyn std::error::Error + Send + Sync>);
                }
                Err(status) => {
                    return Err(Box::new(status) as Box<dyn std::error::Error + Send + Sync>);
                }
            };

            let mut content = Vec::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| BackendError::Other(e.into()))?;
                content.extend_from_slice(&chunk.chunk);
            }
            Ok(content)
        })
        .await
        .map_err(|e| match e.downcast::<BackendError>() {
            Ok(err) => *err,
            Err(e) => BackendError::Other(e),
        })?;

        let cursor = futures::io::Cursor::new(content);
        Ok(Box::pin(cursor) as Pin<Box<dyn futures::AsyncRead + Send>>)
    }

    async fn write_file(
        &self,
        _path: &RepoPath,
        contents: &mut (dyn futures::AsyncRead + Send + Unpin),
    ) -> BackendResult<FileId> {
        let mut raw_bytes = Vec::new();
        let mut scratch = vec![0u8; cc_common::FILE_STREAM_CHUNK_SIZE];
        loop {
            let n = contents
                .read(&mut scratch)
                .await
                .map_err(|e| BackendError::Other(e.into()))?;
            if n == 0 {
                break;
            }
            raw_bytes.extend_from_slice(&scratch[..n]);
        }

        let server_url = self.server_url.clone();
        let project_id = self.project_id.clone();
        let try_unary_first = self.prefer_unary_write_file.load(Ordering::Relaxed);

        let (file_id_bytes, switched_to_unary) = run_on_tokio(async move {
            let mut client = crate::util::connect_backend_client(server_url).await?;
            if !try_unary_first {
                let mut chunks: Vec<cc_common::backend::WriteFileChunk> = Vec::new();
                if raw_bytes.is_empty() {
                    chunks.push(cc_common::backend::WriteFileChunk {
                        project_id: project_id.clone(),
                        chunk: Vec::new(),
                    });
                } else {
                    for (idx, slice) in raw_bytes
                        .chunks(cc_common::FILE_STREAM_CHUNK_SIZE)
                        .enumerate()
                    {
                        chunks.push(cc_common::backend::WriteFileChunk {
                            project_id: if idx == 0 {
                                project_id.clone()
                            } else {
                                String::new()
                            },
                            chunk: slice.to_vec(),
                        });
                    }
                }
                match client
                    .write_file_stream(tonic::Request::new(futures::stream::iter(chunks)))
                    .await
                {
                    Ok(res) => return Ok((res.into_inner().file_id, false)),
                    Err(status) if status.code() == tonic::Code::Unimplemented => {}
                    Err(status) => return Err(status.into()),
                }
            }

            let res = client
                .write_file(tonic::Request::new(cc_common::backend::WriteFileRequest {
                    project_id,
                    content: raw_bytes,
                }))
                .await?;
            Ok((res.into_inner().file_id, true))
        })
        .await
        .map_err(BackendError::Other)?;

        if switched_to_unary && !try_unary_first {
            self.prefer_unary_write_file.store(true, Ordering::Relaxed);
        }

        Ok(FileId::from_bytes(&file_id_bytes))
    }

    async fn read_symlink(&self, _path: &RepoPath, id: &SymlinkId) -> BackendResult<String> {
        let server_url = self.server_url.clone();
        let project_id = self.project_id.clone();
        let symlink_id_bytes = id.to_bytes().to_vec();
        let symlink_id_hex = id.hex();

        run_on_tokio(async move {
            let mut client = crate::util::connect_backend_client(server_url).await?;
            let res = client
                .read_symlink(tonic::Request::new(
                    cc_common::backend::ReadSymlinkRequest {
                        project_id,
                        symlink_id: symlink_id_bytes,
                    },
                ))
                .await;

            match res {
                Ok(r) => Ok(r.into_inner().target),
                Err(status) if status.code() == tonic::Code::NotFound => {
                    Err(Box::new(BackendError::ObjectNotFound {
                        object_type: "symlink".into(),
                        hash: symlink_id_hex,
                        source: status.into(),
                    })
                        as Box<dyn std::error::Error + Send + Sync>)
                }
                Err(status) => Err(Box::new(status) as Box<dyn std::error::Error + Send + Sync>),
            }
        })
        .await
        .map_err(|e| match e.downcast::<BackendError>() {
            Ok(err) => *err,
            Err(e) => BackendError::Other(e),
        })
    }

    async fn write_symlink(&self, _path: &RepoPath, target: &str) -> BackendResult<SymlinkId> {
        let server_url = self.server_url.clone();
        let project_id = self.project_id.clone();
        let target_owned = target.to_string();

        let symlink_id_bytes = run_on_tokio(async move {
            let mut client = crate::util::connect_backend_client(server_url).await?;
            let res = client
                .write_symlink(tonic::Request::new(
                    cc_common::backend::WriteSymlinkRequest {
                        project_id,
                        target: target_owned,
                    },
                ))
                .await?;
            Ok(res.into_inner().symlink_id)
        })
        .await
        .map_err(BackendError::Other)?;

        Ok(SymlinkId::from_bytes(&symlink_id_bytes))
    }

    async fn read_copy(&self, _id: &CopyId) -> BackendResult<CopyHistory> {
        Err(BackendError::Unsupported(
            "copies not supported".to_string(),
        ))
    }

    async fn write_copy(&self, _contents: &CopyHistory) -> BackendResult<CopyId> {
        Err(BackendError::Unsupported(
            "copies not supported".to_string(),
        ))
    }

    async fn get_related_copies(&self, _copy_id: &CopyId) -> BackendResult<Vec<RelatedCopy>> {
        Err(BackendError::Unsupported(
            "copies not supported".to_string(),
        ))
    }

    async fn read_tree(&self, path: &RepoPath, id: &TreeId) -> BackendResult<Tree> {
        if *id == self.empty_tree_id {
            return Ok(Tree::from_sorted_entries(vec![]));
        }

        let server_url = self.server_url.clone();
        let project_id = self.project_id.clone();
        let tree_id_bytes = id.to_bytes().to_vec();
        let tree_id_hex = id.hex();
        let path_str = path.as_internal_file_string().to_string();

        let proto_entries = run_on_tokio(async move {
            let mut client = crate::util::connect_backend_client(server_url).await?;
            let res = client
                .read_tree(tonic::Request::new(cc_common::backend::ReadTreeRequest {
                    project_id,
                    tree_id: tree_id_bytes,
                    path: path_str,
                    page_size: 0,
                    page_token: String::new(),
                }))
                .await;

            match res {
                Ok(r) => Ok(r.into_inner().entries),
                Err(status) if status.code() == tonic::Code::NotFound => {
                    Err(Box::new(BackendError::ObjectNotFound {
                        object_type: "tree".into(),
                        hash: tree_id_hex,
                        source: status.into(),
                    })
                        as Box<dyn std::error::Error + Send + Sync>)
                }
                Err(status) => Err(Box::new(status) as Box<dyn std::error::Error + Send + Sync>),
            }
        })
        .await
        .map_err(|e| match e.downcast::<BackendError>() {
            Ok(err) => *err,
            Err(e) => BackendError::Other(e),
        })?;

        let mut jj_entries = Vec::new();
        for entry in proto_entries {
            let (comp, val) = tree_entry_from_proto(entry).map_err(BackendError::Other)?;
            jj_entries.push((comp, val));
        }
        jj_entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        Ok(Tree::from_sorted_entries(jj_entries))
    }

    async fn write_tree(&self, path: &RepoPath, tree: &Tree) -> BackendResult<TreeId> {
        if tree.entries().next().is_none() {
            return Ok(self.empty_tree_id.clone());
        }

        let proto_entries: Result<Vec<_>, _> =
            tree.entries().map(|e| tree_entry_to_proto(&e)).collect();
        let proto_entries = proto_entries?;

        let server_url = self.server_url.clone();
        let project_id = self.project_id.clone();
        let path_str = path.as_internal_file_string().to_string();

        let tree_id_bytes = run_on_tokio(async move {
            let mut client = crate::util::connect_backend_client(server_url).await?;
            let res = client
                .write_tree(tonic::Request::new(cc_common::backend::WriteTreeRequest {
                    project_id,
                    path: path_str,
                    entries: proto_entries,
                }))
                .await?;
            Ok(res.into_inner().tree_id)
        })
        .await
        .map_err(BackendError::Other)?;

        Ok(TreeId::from_bytes(&tree_id_bytes))
    }

    async fn read_commit(&self, id: &CommitId) -> BackendResult<Commit> {
        if *id == self.root_commit_id {
            return Ok(make_root_commit(
                self.root_change_id().clone(),
                self.empty_tree_id.clone(),
            ));
        }

        let server_url = self.server_url.clone();
        let project_id = self.project_id.clone();
        let commit_id_bytes = id.to_bytes().to_vec();
        let commit_id_hex = id.hex();

        let proto_commit = run_on_tokio(async move {
            let mut client = crate::util::connect_backend_client(server_url).await?;
            let res = client
                .read_commit(tonic::Request::new(cc_common::backend::ReadCommitRequest {
                    project_id,
                    commit_id: commit_id_bytes,
                }))
                .await;

            match res {
                Ok(r) => r.into_inner().commit.ok_or_else(|| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "server response should have contained commit data",
                    )) as Box<dyn std::error::Error + Send + Sync>
                }),
                Err(status) if status.code() == tonic::Code::NotFound => {
                    Err(Box::new(BackendError::ObjectNotFound {
                        object_type: "commit".into(),
                        hash: commit_id_hex,
                        source: status.into(),
                    })
                        as Box<dyn std::error::Error + Send + Sync>)
                }
                Err(status) => Err(Box::new(status) as Box<dyn std::error::Error + Send + Sync>),
            }
        })
        .await
        .map_err(|e| match e.downcast::<BackendError>() {
            Ok(err) => *err,
            Err(e) => BackendError::Other(e),
        })?;

        Ok(commit_from_proto(proto_commit))
    }

    async fn write_commit(
        &self,
        mut commit: Commit,
        sign_with: Option<&mut SigningFn>,
    ) -> BackendResult<(CommitId, Commit)> {
        if commit.parents.contains(&self.root_commit_id) && commit.parents.len() > 1 {
            return Err(BackendError::Unsupported(
                "The Commit Cloud backend does not support creating merge commits with the root commit as one of the parents."
                    .to_owned(),
            ));
        }

        apply_commit_signing(&mut commit, sign_with)?;
        let proto_commit = commit_to_proto(&commit);
        let server_url = self.server_url.clone();
        let project_id = self.project_id.clone();

        let returned_id_bytes = run_on_tokio(async move {
            let mut client = crate::util::connect_backend_client(server_url).await?;
            let res = client
                .write_commit(tonic::Request::new(
                    cc_common::backend::WriteCommitRequest {
                        project_id,
                        commit: Some(proto_commit),
                    },
                ))
                .await?;
            Ok(res.into_inner().commit_id)
        })
        .await
        .map_err(BackendError::Other)?;

        let returned_id = CommitId::from_bytes(&returned_id_bytes);
        Ok((returned_id, commit))
    }

    fn get_copy_records(
        &self,
        _paths: Option<&[RepoPathBuf]>,
        _root: &CommitId,
        _head: &CommitId,
    ) -> BackendResult<BoxStream<'_, BackendResult<CopyRecord>>> {
        Ok(stream::empty().boxed())
    }

    fn gc(&self, _index: &dyn Index, _keep_newer: SystemTime) -> BackendResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    use cc_common::backend::backend_service_server::{BackendService, BackendServiceServer};
    use cc_common::backend::*;
    use sha1::{Digest as _, Sha1};
    use tonic::{Request, Response, Status, Streaming};

    use super::*;

    struct UnaryOnlyBackendServer {
        stream_attempts: Arc<AtomicUsize>,
        unary_writes: Arc<AtomicUsize>,
        expected_project_id: String,
    }

    fn unimpl<T>() -> Result<Response<T>, Status> {
        Err(Status::unimplemented("unimplemented in mock"))
    }

    #[tonic::async_trait]
    impl BackendService for UnaryOnlyBackendServer {
        async fn register_project(
            &self,
            _: Request<RegisterProjectRequest>,
        ) -> Result<Response<RegisterProjectResponse>, Status> {
            unimpl()
        }
        async fn register_repository(
            &self,
            req: Request<RegisterRepositoryRequest>,
        ) -> Result<Response<RegisterRepositoryResponse>, Status> {
            let r = req.into_inner();
            if r.project_id.trim().is_empty() {
                return Err(Status::invalid_argument("project_id required"));
            }
            let repo_id = r.repo_id.unwrap_or_else(|| "alloc-repo".to_string());
            Ok(Response::new(RegisterRepositoryResponse {
                repo_id,
                project_id: r.project_id,
            }))
        }
        async fn read_commit(
            &self,
            _: Request<ReadCommitRequest>,
        ) -> Result<Response<ReadCommitResponse>, Status> {
            unimpl()
        }
        async fn write_commit(
            &self,
            _: Request<WriteCommitRequest>,
        ) -> Result<Response<WriteCommitResponse>, Status> {
            unimpl()
        }
        async fn list_project_commits(
            &self,
            _: Request<ListProjectCommitsRequest>,
        ) -> Result<Response<ListProjectCommitsResponse>, Status> {
            Ok(Response::new(ListProjectCommitsResponse {
                commit_ids: Vec::new(),
            }))
        }
        async fn list_commits(
            &self,
            _: Request<ListCommitsRequest>,
        ) -> Result<Response<ListCommitsResponse>, Status> {
            unimpl()
        }
        async fn resolve_commit_prefix(
            &self,
            _: Request<ResolveCommitPrefixRequest>,
        ) -> Result<Response<ResolveCommitPrefixResponse>, Status> {
            unimpl()
        }
        async fn resolve_change_prefix(
            &self,
            _: Request<ResolveChangePrefixRequest>,
        ) -> Result<Response<ResolveChangePrefixResponse>, Status> {
            unimpl()
        }
        async fn read_tree(
            &self,
            _: Request<ReadTreeRequest>,
        ) -> Result<Response<ReadTreeResponse>, Status> {
            unimpl()
        }
        async fn write_tree(
            &self,
            _: Request<WriteTreeRequest>,
        ) -> Result<Response<WriteTreeResponse>, Status> {
            unimpl()
        }
        type ReadFileStream =
            Pin<Box<dyn futures::Stream<Item = Result<ReadFileResponse, Status>> + Send + 'static>>;
        async fn read_file(
            &self,
            _: Request<ReadFileRequest>,
        ) -> Result<Response<Self::ReadFileStream>, Status> {
            unimpl()
        }
        async fn write_file(
            &self,
            req: Request<WriteFileRequest>,
        ) -> Result<Response<WriteFileResponse>, Status> {
            let r = req.into_inner();
            assert_eq!(r.project_id, self.expected_project_id);
            self.unary_writes.fetch_add(1, Ordering::SeqCst);
            let mut hasher = Sha1::new();
            hasher.update(format!("blob {}\0", r.content.len()).as_bytes());
            hasher.update(&r.content);
            Ok(Response::new(WriteFileResponse {
                file_id: hasher.finalize().to_vec(),
            }))
        }
        async fn write_file_stream(
            &self,
            _: Request<Streaming<WriteFileChunk>>,
        ) -> Result<Response<WriteFileResponse>, Status> {
            self.stream_attempts.fetch_add(1, Ordering::SeqCst);
            Err(Status::unimplemented("WriteFileStream not supported"))
        }
        async fn read_symlink(
            &self,
            _: Request<ReadSymlinkRequest>,
        ) -> Result<Response<ReadSymlinkResponse>, Status> {
            unimpl()
        }
        async fn write_symlink(
            &self,
            _: Request<WriteSymlinkRequest>,
        ) -> Result<Response<WriteSymlinkResponse>, Status> {
            unimpl()
        }
    }

    #[test]
    fn test_write_file_falls_back_to_unary_write_file_and_caches_preference() {
        let stream_attempts = Arc::new(AtomicUsize::new(0));
        let unary_writes = Arc::new(AtomicUsize::new(0));
        let sa = Arc::clone(&stream_attempts);
        let uw = Arc::clone(&unary_writes);

        let url = run_async(move || async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let addr = listener.local_addr()?;
            let incoming =
                tonic::transport::server::TcpIncoming::from_listener(listener, true, None)
                    .map_err(|e| format!("incoming error: {e}"))?;
            let svc = UnaryOnlyBackendServer {
                stream_attempts: sa,
                unary_writes: uw,
                expected_project_id: "shared-proj".to_string(),
            };
            tokio::spawn(
                tonic::transport::Server::builder()
                    .add_service(BackendServiceServer::new(svc))
                    .serve_with_incoming(incoming),
            );
            Ok(format!("http://{addr}"))
        })
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let backend = CommitCloudBackend::init_in_project(
            dir.path(),
            &url,
            None,
            Some("shared-proj"),
            Some("actor-1"),
            true,
        )
        .unwrap();
        assert_eq!(backend.project_id(), "shared-proj");
        assert_eq!(backend.repo_id(), "actor-1");

        futures::executor::block_on(async {
            let mut r1 = futures::io::Cursor::new(b"first payload".to_vec());
            let _ = backend.write_file(RepoPath::root(), &mut r1).await.unwrap();
            let mut r2 = futures::io::Cursor::new(b"second payload".to_vec());
            let _ = backend.write_file(RepoPath::root(), &mut r2).await.unwrap();
        });

        assert_eq!(
            stream_attempts.load(Ordering::SeqCst),
            1,
            "second write_file must skip WriteFileStream once UNIMPLEMENTED was observed"
        );
        assert_eq!(unary_writes.load(Ordering::SeqCst), 2);
    }
}
