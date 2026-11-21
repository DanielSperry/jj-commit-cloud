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

//! Local SQLite `MetadataStore` implementation running in `WAL` journal mode with a
//! concurrent read-connection pool.
//!
//! # Authority and Crash-Safety Ordering
//! `SqliteMetadataStore` persists relational commit, tree, symlink, view, operation, and
//! blob-location rows after `CompositeStore` has completed any out-of-line `BlobStore`
//! upload. Multi-row writes (`put_tree`, `put_blob_record`, `update_op_heads`, and
//! `publish_operation_atomic`) execute inside explicit SQLite `IMMEDIATE` transactions so
//! partial tree entry lists or orphaned `project_blobs` links never appear after a crash.
//!
//! # Concurrency and Merge Invariants
//! File-backed databases open a dedicated writer connection alongside a pool of read-only
//! `WAL` snapshot connections so concurrent agent reads never contend on the writer mutex,
//! while `update_op_heads` and `publish_operation_atomic` acquire `TransactionBehavior::Immediate`
//! write locks to apply [`super::apply_op_heads_mutation`] without read-modify-write races.
//!
//! # Cross-Process and RPC Boundaries
//! Blocking `rusqlite` operations run on Tokio's `spawn_blocking` pool and map SQLite
//! constraint and I/O failures into [`StoreError`].

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use cc_common::backend::*;
use cc_common::op_store::*;
use cc_common::workspace::WorkspaceState;
use prost::Message;
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};

use super::{
    BlobRecord, CommitId, MetadataStore, OpId, StoreError, StoreResult, apply_op_heads_mutation,
    op_heads_sets_equal,
};

const SQLITE_SCHEMA: &str = include_str!("../../sql/sqlite_schema.sql");
const WAL_READER_POOL_SIZE: usize = 8;

#[derive(Clone)]
pub struct SqliteMetadataStore {
    writer: Arc<Mutex<Connection>>,
    readers: Arc<Vec<Arc<Mutex<Connection>>>>,
    next_reader: Arc<AtomicUsize>,
}

impl SqliteMetadataStore {
    pub fn open(db_path: &Path) -> StoreResult<Self> {
        if let Some(parent) = db_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                StoreError::Backend(format!(
                    "failed to create SQLite parent directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        let writer_conn = Connection::open(db_path).map_err(|e| {
            StoreError::Backend(format!(
                "failed to open SQLite database {}: {e}",
                db_path.display()
            ))
        })?;
        configure_writer_connection(&writer_conn)?;

        let mut reader_conns = Vec::with_capacity(WAL_READER_POOL_SIZE);
        let read_flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        for _ in 0..WAL_READER_POOL_SIZE {
            let rconn = Connection::open_with_flags(db_path, read_flags).map_err(|e| {
                StoreError::Backend(format!(
                    "failed to open SQLite WAL reader {}: {e}",
                    db_path.display()
                ))
            })?;
            rconn
                .execute_batch("PRAGMA busy_timeout = 5000; PRAGMA query_only = ON;")
                .map_err(|e| {
                    StoreError::Backend(format!("failed to configure SQLite reader pragmas: {e}"))
                })?;
            reader_conns.push(Arc::new(Mutex::new(rconn)));
        }

        Ok(Self {
            writer: Arc::new(Mutex::new(writer_conn)),
            readers: Arc::new(reader_conns),
            next_reader: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn in_memory() -> StoreResult<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| StoreError::Backend(format!("failed to open in-memory SQLite: {e}")))?;
        configure_writer_connection(&conn)?;
        let shared = Arc::new(Mutex::new(conn));
        Ok(Self {
            writer: Arc::clone(&shared),
            readers: Arc::new(vec![shared]),
            next_reader: Arc::new(AtomicUsize::new(0)),
        })
    }

    async fn with_write_conn<F, T>(&self, f: F) -> StoreResult<T>
    where
        F: FnOnce(&mut Connection) -> StoreResult<T> + Send + 'static,
        T: Send + 'static,
    {
        let conn = Arc::clone(&self.writer);
        tokio::task::spawn_blocking(move || {
            let mut guard = conn
                .lock()
                .map_err(|_| StoreError::Backend("SQLite writer mutex poisoned".to_string()))?;
            f(&mut guard)
        })
        .await
        .map_err(|e| StoreError::Backend(format!("SQLite blocking task panicked: {e}")))?
    }

    async fn with_read_conn<F, T>(&self, f: F) -> StoreResult<T>
    where
        F: FnOnce(&mut Connection) -> StoreResult<T> + Send + 'static,
        T: Send + 'static,
    {
        let idx = self.next_reader.fetch_add(1, Ordering::Relaxed) % self.readers.len();
        let conn = Arc::clone(&self.readers[idx]);
        tokio::task::spawn_blocking(move || {
            let mut guard = conn
                .lock()
                .map_err(|_| StoreError::Backend("SQLite reader mutex poisoned".to_string()))?;
            f(&mut guard)
        })
        .await
        .map_err(|e| StoreError::Backend(format!("SQLite blocking task panicked: {e}")))?
    }
}

fn configure_writer_connection(conn: &Connection) -> StoreResult<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 5000;
         PRAGMA foreign_keys = ON;",
    )
    .map_err(|e| StoreError::Backend(format!("failed to configure SQLite pragmas: {e}")))?;

    conn.execute_batch(SQLITE_SCHEMA)
        .map_err(|e| StoreError::Backend(format!("failed to initialize SQLite schema: {e}")))?;

    // Upgrade pre-existing SQLite workspaces tables transparently if needed.
    let _ = conn.execute(
        "ALTER TABLE workspaces ADD COLUMN tree_ids_json TEXT NOT NULL DEFAULT '[]'",
        [],
    );
    Ok(())
}

fn encode_op_heads_json(heads: &[OpId]) -> StoreResult<String> {
    let hex_ids: Vec<String> = heads.iter().map(hex::encode).collect();
    serde_json::to_string(&hex_ids)
        .map_err(|e| StoreError::Corruption(format!("failed to serialize op_heads JSON: {e}")))
}

fn decode_op_heads_json(raw: &str) -> StoreResult<Vec<OpId>> {
    let hex_ids: Vec<String> = serde_json::from_str(raw)
        .map_err(|e| StoreError::Corruption(format!("failed to parse op_heads JSON: {e}")))?;
    hex_ids
        .into_iter()
        .map(|h| {
            hex::decode(&h).map_err(|e| {
                StoreError::Corruption(format!("invalid hex in persisted op_heads '{h}': {e}"))
            })
        })
        .collect()
}

#[async_trait]
impl MetadataStore for SqliteMetadataStore {
    fn backend_name(&self) -> &'static str {
        "sqlite"
    }

    async fn register_project(&self, project_id: &str, name: Option<&str>) -> StoreResult<()> {
        let project_id = project_id.to_string();
        let name = name.map(ToOwned::to_owned);
        self.with_write_conn(move |conn| {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|e| StoreError::Backend(format!("SQLite begin project tx failed: {e}")))?;

            if let Some(alias) = &name {
                let existing: Option<String> = tx
                    .query_row(
                        "SELECT project_id FROM projects WHERE name = ?1",
                        params![alias],
                        |row| row.get(0),
                    )
                    .optional()
                    .map_err(|e| {
                        StoreError::Backend(format!("SQLite project alias lookup failed: {e}"))
                    })?;
                if let Some(other_id) = existing
                    && other_id != project_id
                {
                    return Err(StoreError::ProjectAlreadyExists(alias.clone()));
                }
            }

            tx.execute(
                "INSERT INTO projects (project_id, name)
                 VALUES (?1, ?2)
                 ON CONFLICT(project_id) DO UPDATE SET name = COALESCE(excluded.name, projects.name)",
                params![project_id, name],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite register_project failed: {e}")))?;

            tx.commit().map_err(|e| {
                StoreError::Backend(format!("SQLite commit register_project failed: {e}"))
            })
        })
        .await
    }

    async fn is_project_registered(&self, project_id: &str) -> StoreResult<bool> {
        let project_id = project_id.to_string();
        self.with_read_conn(move |conn| {
            let mut stmt = conn
                .prepare("SELECT 1 FROM projects WHERE project_id = ?1")
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite is_project_registered prepare failed: {e}"))
                })?;
            stmt.exists(params![project_id]).map_err(|e| {
                StoreError::Backend(format!("SQLite is_project_registered failed: {e}"))
            })
        })
        .await
    }

    async fn get_repo_project_id(&self, repo_id: &str) -> StoreResult<Option<String>> {
        let repo_id = repo_id.to_string();
        self.with_read_conn(move |conn| {
            conn.query_row(
                "SELECT project_id FROM repositories WHERE repo_id = ?1",
                params![repo_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| StoreError::Backend(format!("SQLite get_repo_project_id failed: {e}")))
        })
        .await
    }

    async fn register_repo_in_project(
        &self,
        repo_id: &str,
        project_id: &str,
        name: Option<&str>,
    ) -> StoreResult<String> {
        let repo_id = repo_id.to_string();
        let project_id = project_id.to_string();
        let name = name.map(ToOwned::to_owned);
        self.with_write_conn(move |conn| {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|e| StoreError::Backend(format!("SQLite begin tx failed: {e}")))?;

            let existing_project: Option<String> = tx
                .query_row(
                    "SELECT project_id FROM repositories WHERE repo_id = ?1",
                    params![repo_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite existing repo lookup failed: {e}"))
                })?;
            if let Some(bound_proj) = &existing_project
                && bound_proj != &project_id
            {
                return Err(StoreError::InvalidArgument(format!(
                    "repository '{repo_id}' is already bound to project '{bound_proj}', cannot rebind to '{project_id}'"
                )));
            }

            if let Some(alias) = &name {
                let existing: Option<String> = tx
                    .query_row(
                        "SELECT repo_id FROM repositories WHERE name = ?1",
                        params![alias],
                        |row| row.get(0),
                    )
                    .optional()
                    .map_err(|e| StoreError::Backend(format!("SQLite alias lookup failed: {e}")))?;
                if let Some(other_id) = existing
                    && other_id != repo_id
                {
                    return Err(StoreError::RepoAlreadyExists(alias.clone()));
                }
            }

            tx.execute(
                "INSERT OR IGNORE INTO projects (project_id, name) VALUES (?1, NULL)",
                params![project_id],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite ensure project failed: {e}")))?;

            let initial_heads =
                encode_op_heads_json(&[cc_common::ROOT_OPERATION_ID_BYTES.to_vec()])?;
            tx.execute(
                "INSERT INTO repositories (repo_id, project_id, name, op_heads_json)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(repo_id) DO UPDATE SET name = COALESCE(excluded.name, repositories.name)",
                params![repo_id, project_id, name, initial_heads],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite register_repo failed: {e}")))?;

            tx.commit().map_err(|e| {
                StoreError::Backend(format!("SQLite commit register_repo failed: {e}"))
            })?;
            Ok(project_id)
        })
        .await
    }

    async fn get_commit(&self, project_id: &str, commit_id: &[u8]) -> StoreResult<Option<Commit>> {
        let project_id = project_id.to_string();
        let commit_id = commit_id.to_vec();
        self.with_read_conn(move |conn| {
            let raw: Option<Vec<u8>> = conn
                .query_row(
                    "SELECT proto_bytes FROM commits WHERE project_id = ?1 AND commit_id = ?2",
                    params![project_id, commit_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| StoreError::Backend(format!("SQLite get_commit failed: {e}")))?;

            raw.map(|bytes| {
                Commit::decode(bytes.as_slice()).map_err(|e| {
                    StoreError::Corruption(format!("failed to decode Commit protobuf: {e}"))
                })
            })
            .transpose()
        })
        .await
    }

    async fn get_commit_generation(
        &self,
        project_id: &str,
        commit_id: &[u8],
    ) -> StoreResult<Option<u64>> {
        let project_id = project_id.to_string();
        let commit_id = commit_id.to_vec();
        self.with_read_conn(move |conn| {
            let generation: Option<i64> = conn
                .query_row(
                    "SELECT generation FROM commits WHERE project_id = ?1 AND commit_id = ?2",
                    params![project_id, commit_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite get_commit_generation failed: {e}"))
                })?;
            Ok(generation.map(|g| g as u64))
        })
        .await
    }

    async fn put_commit(
        &self,
        project_id: &str,
        commit_id: &[u8],
        commit: &Commit,
        generation: u64,
    ) -> StoreResult<()> {
        let project_id = project_id.to_string();
        let commit_id = commit_id.to_vec();
        let change_id = commit.change_id.clone();
        let proto_bytes = commit.encode_to_vec();
        let gen_i64 = generation as i64;

        self.with_write_conn(move |conn| {
            conn.execute(
                "INSERT OR IGNORE INTO commits (project_id, commit_id, change_id, generation, proto_bytes)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![project_id, commit_id, change_id, gen_i64, proto_bytes],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite put_commit failed: {e}")))?;
            Ok(())
        })
        .await
    }

    async fn list_commit_ids(
        &self,
        project_id: &str,
        after_commit_id: Option<&[u8]>,
        limit: usize,
    ) -> StoreResult<Vec<CommitId>> {
        let project_id = project_id.to_string();
        let cursor = after_commit_id
            .filter(|c| !c.is_empty())
            .map(|c| c.to_vec());
        let limit_i64 = limit as i64;

        self.with_read_conn(move |conn| {
            let mut ids = Vec::new();
            if let Some(after) = cursor {
                let mut stmt = conn
                    .prepare(
                        "SELECT commit_id FROM commits
                         WHERE project_id = ?1 AND commit_id > ?2
                         ORDER BY commit_id ASC
                         LIMIT ?3",
                    )
                    .map_err(|e| {
                        StoreError::Backend(format!("SQLite prepare list_commit_ids failed: {e}"))
                    })?;
                let rows = stmt
                    .query_map(params![project_id, after, limit_i64], |r| {
                        r.get::<_, Vec<u8>>(0)
                    })
                    .map_err(|e| {
                        StoreError::Backend(format!("SQLite query list_commit_ids failed: {e}"))
                    })?;
                for r in rows {
                    ids.push(r.map_err(|e| {
                        StoreError::Backend(format!("SQLite read commit_id failed: {e}"))
                    })?);
                }
            } else {
                let mut stmt = conn
                    .prepare(
                        "SELECT commit_id FROM commits
                         WHERE project_id = ?1
                         ORDER BY commit_id ASC
                         LIMIT ?2",
                    )
                    .map_err(|e| {
                        StoreError::Backend(format!("SQLite prepare list_commit_ids failed: {e}"))
                    })?;
                let rows = stmt
                    .query_map(params![project_id, limit_i64], |r| r.get::<_, Vec<u8>>(0))
                    .map_err(|e| {
                        StoreError::Backend(format!("SQLite query list_commit_ids failed: {e}"))
                    })?;
                for r in rows {
                    ids.push(r.map_err(|e| {
                        StoreError::Backend(format!("SQLite read commit_id failed: {e}"))
                    })?);
                }
            }
            Ok(ids)
        })
        .await
    }

    async fn current_commit_watermark(&self, project_id: &str) -> StoreResult<Vec<u8>> {
        let project_id = project_id.to_string();
        self.with_read_conn(move |conn| {
            let max_rowid: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(rowid), 0) FROM commits WHERE project_id = ?1",
                    params![project_id],
                    |r| r.get(0),
                )
                .map_err(|e| StoreError::Backend(format!("SQLite commit watermark failed: {e}")))?;
            Ok(max_rowid.to_be_bytes().to_vec())
        })
        .await
    }

    async fn list_commit_ids_since(
        &self,
        project_id: &str,
        since_token: &[u8],
        limit: usize,
    ) -> StoreResult<(Vec<CommitId>, Vec<u8>, bool)> {
        let since_rowid = if since_token.len() == 8 {
            i64::from_be_bytes(since_token.try_into().unwrap())
        } else {
            return Err(StoreError::InvalidArgument(
                "invalid SQLite commit sync token".to_string(),
            ));
        };
        let project_id = project_id.to_string();
        let fallback_tok = since_token.to_vec();
        let fetch_limit = (limit + 1) as i64;

        self.with_read_conn(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT rowid, commit_id FROM commits
                     WHERE project_id = ?1 AND rowid > ?2
                     ORDER BY rowid ASC
                     LIMIT ?3",
                )
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite prepare list_commit_ids_since failed: {e}"))
                })?;
            let rows = stmt
                .query_map(params![project_id, since_rowid, fetch_limit], |r| {
                    Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
                })
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite query list_commit_ids_since failed: {e}"))
                })?;
            let mut items = Vec::new();
            for r in rows {
                items.push(r.map_err(|e| {
                    StoreError::Backend(format!("SQLite read incremental commit failed: {e}"))
                })?);
            }
            let has_more = items.len() > limit;
            if has_more {
                items.truncate(limit);
            }
            let Some(&(last_rowid, _)) = items.last() else {
                return Ok((Vec::new(), fallback_tok, false));
            };
            let ids = items.into_iter().map(|(_, cid)| cid).collect();
            Ok((ids, last_rowid.to_be_bytes().to_vec(), has_more))
        })
        .await
    }

    async fn resolve_commit_id_prefix(
        &self,
        project_id: &str,
        min_id: &[u8],
        max_id: &[u8],
    ) -> StoreResult<Vec<CommitId>> {
        let project_id = project_id.to_string();
        let min_id = min_id.to_vec();
        let max_id = max_id.to_vec();
        self.with_read_conn(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT commit_id FROM commits
                     WHERE project_id = ?1 AND commit_id >= ?2 AND commit_id <= ?3
                     ORDER BY commit_id ASC
                     LIMIT 2",
                )
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite prepare commit prefix query failed: {e}"))
                })?;
            let rows = stmt
                .query_map(params![project_id, min_id, max_id], |row| {
                    row.get::<_, Vec<u8>>(0)
                })
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite execute commit prefix query failed: {e}"))
                })?;
            let mut matches = Vec::new();
            for row in rows {
                matches.push(row.map_err(|e| {
                    StoreError::Backend(format!("SQLite read commit prefix match failed: {e}"))
                })?);
            }
            Ok(matches)
        })
        .await
    }

    async fn resolve_change_id_prefix(
        &self,
        project_id: &str,
        min_id: &[u8],
        max_id: &[u8],
    ) -> StoreResult<Vec<(Vec<u8>, Vec<CommitId>)>> {
        let project_id = project_id.to_string();
        let min_id = min_id.to_vec();
        let max_id = max_id.to_vec();
        self.with_read_conn(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT DISTINCT change_id FROM commits
                     WHERE project_id = ?1 AND change_id >= ?2 AND change_id <= ?3
                     ORDER BY change_id ASC
                     LIMIT 2",
                )
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite prepare change prefix query failed: {e}"))
                })?;
            let ch_rows = stmt
                .query_map(params![project_id, min_id, max_id], |r| {
                    r.get::<_, Vec<u8>>(0)
                })
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite execute change prefix query failed: {e}"))
                })?;
            let mut change_ids = Vec::new();
            for r in ch_rows {
                change_ids.push(r.map_err(|e| {
                    StoreError::Backend(format!("SQLite read change_id match failed: {e}"))
                })?);
            }
            if change_ids.len() != 1 {
                return Ok(change_ids.into_iter().map(|ch| (ch, Vec::new())).collect());
            }
            let matched_ch = change_ids.swap_remove(0);
            let mut c_stmt = conn
                .prepare(
                    "SELECT commit_id FROM commits
                     WHERE project_id = ?1 AND change_id = ?2
                     ORDER BY commit_id ASC",
                )
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite prepare change commits query failed: {e}"))
                })?;
            let c_rows = c_stmt
                .query_map(params![project_id, matched_ch], |r| r.get::<_, Vec<u8>>(0))
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite execute change commits query failed: {e}"))
                })?;
            let mut by_change = BTreeMap::new();
            let mut cids = Vec::new();
            for r in c_rows {
                cids.push(r.map_err(|e| {
                    StoreError::Backend(format!("SQLite read change commit_id failed: {e}"))
                })?);
            }
            by_change.insert(matched_ch, cids);
            Ok(by_change.into_iter().collect())
        })
        .await
    }

    async fn get_tree(
        &self,
        project_id: &str,
        tree_id: &[u8],
    ) -> StoreResult<Option<Vec<TreeEntry>>> {
        let project_id = project_id.to_string();
        let tree_id = tree_id.to_vec();
        self.with_read_conn(move |conn| {
            let exists: Option<i64> = conn
                .query_row(
                    "SELECT 1 FROM trees WHERE project_id = ?1 AND tree_id = ?2",
                    params![project_id, tree_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| StoreError::Backend(format!("SQLite get_tree header failed: {e}")))?;

            if exists.is_none() {
                return Ok(None);
            }

            let mut stmt = conn
                .prepare(
                    "SELECT entry_proto FROM tree_entries
                     WHERE project_id = ?1 AND tree_id = ?2
                     ORDER BY name ASC",
                )
                .map_err(|e| StoreError::Backend(format!("SQLite prepare get_tree failed: {e}")))?;

            let rows = stmt
                .query_map(params![project_id, tree_id], |row| row.get::<_, Vec<u8>>(0))
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite query tree_entries failed: {e}"))
                })?;

            let mut entries = Vec::new();
            for row in rows {
                let bytes = row.map_err(|e| {
                    StoreError::Backend(format!("SQLite read tree_entry row failed: {e}"))
                })?;
                let entry = TreeEntry::decode(bytes.as_slice()).map_err(|e| {
                    StoreError::Corruption(format!("failed to decode TreeEntry protobuf: {e}"))
                })?;
                entries.push(entry);
            }
            Ok(Some(entries))
        })
        .await
    }

    async fn get_tree_page(
        &self,
        project_id: &str,
        tree_id: &[u8],
        after_name: Option<&str>,
        limit: usize,
    ) -> StoreResult<Option<Vec<TreeEntry>>> {
        let project_id = project_id.to_string();
        let tree_id = tree_id.to_vec();
        let after = after_name.unwrap_or("").to_string();
        let limit_i64 = limit as i64;
        self.with_read_conn(move |conn| {
            let exists: Option<i64> = conn
                .query_row(
                    "SELECT 1 FROM trees WHERE project_id = ?1 AND tree_id = ?2",
                    params![project_id, tree_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| StoreError::Backend(format!("SQLite get_tree_page header: {e}")))?;
            if exists.is_none() {
                return Ok(None);
            }
            let mut stmt = conn
                .prepare(
                    "SELECT entry_proto FROM tree_entries
                     WHERE project_id = ?1 AND tree_id = ?2 AND name > ?3
                     ORDER BY name ASC
                     LIMIT ?4",
                )
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite prepare get_tree_page failed: {e}"))
                })?;
            let rows = stmt
                .query_map(params![project_id, tree_id, after, limit_i64], |row| {
                    row.get::<_, Vec<u8>>(0)
                })
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite query get_tree_page failed: {e}"))
                })?;
            let mut entries = Vec::new();
            for row in rows {
                let bytes = row.map_err(|e| {
                    StoreError::Backend(format!("SQLite read get_tree_page row failed: {e}"))
                })?;
                let entry = TreeEntry::decode(bytes.as_slice()).map_err(|e| {
                    StoreError::Corruption(format!("failed to decode TreeEntry protobuf: {e}"))
                })?;
                entries.push(entry);
            }
            Ok(Some(entries))
        })
        .await
    }

    async fn put_tree(
        &self,
        project_id: &str,
        tree_id: &[u8],
        entries: &[TreeEntry],
    ) -> StoreResult<()> {
        let project_id = project_id.to_string();
        let tree_id = tree_id.to_vec();
        let entries = entries.to_vec();

        self.with_write_conn(move |conn| {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite begin put_tree tx failed: {e}"))
                })?;

            tx.execute(
                "INSERT OR IGNORE INTO trees (project_id, tree_id) VALUES (?1, ?2)",
                params![project_id, tree_id],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite insert tree failed: {e}")))?;

            for entry in &entries {
                let entry_proto = entry.encode_to_vec();
                tx.execute(
                    "INSERT OR REPLACE INTO tree_entries (project_id, tree_id, name, entry_proto)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![project_id, tree_id, entry.name, entry_proto],
                )
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite insert tree_entry failed: {e}"))
                })?;
            }

            tx.commit()
                .map_err(|e| StoreError::Backend(format!("SQLite commit put_tree failed: {e}")))
        })
        .await
    }

    async fn find_existing_trees(
        &self,
        project_id: &str,
        tree_ids: &[Vec<u8>],
    ) -> StoreResult<HashSet<Vec<u8>>> {
        if tree_ids.is_empty() {
            return Ok(HashSet::new());
        }
        let project_id = project_id.to_string();
        let ids = tree_ids.to_vec();
        self.with_read_conn(move |conn| {
            let mut stmt = conn
                .prepare("SELECT 1 FROM trees WHERE project_id = ?1 AND tree_id = ?2")
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite prepare find_existing_trees failed: {e}"))
                })?;
            let mut found = HashSet::with_capacity(ids.len());
            for id in ids {
                let exists: Option<i64> = stmt
                    .query_row(params![project_id, id], |r| r.get(0))
                    .optional()
                    .map_err(|e| {
                        StoreError::Backend(format!("SQLite find_existing_trees failed: {e}"))
                    })?;
                if exists.is_some() {
                    found.insert(id);
                }
            }
            Ok(found)
        })
        .await
    }

    async fn get_symlink(
        &self,
        project_id: &str,
        symlink_id: &[u8],
    ) -> StoreResult<Option<String>> {
        let project_id = project_id.to_string();
        let symlink_id = symlink_id.to_vec();
        self.with_read_conn(move |conn| {
            conn.query_row(
                "SELECT target FROM symlinks WHERE project_id = ?1 AND symlink_id = ?2",
                params![project_id, symlink_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| StoreError::Backend(format!("SQLite get_symlink failed: {e}")))
        })
        .await
    }

    async fn put_symlink(
        &self,
        project_id: &str,
        symlink_id: &[u8],
        target: &str,
    ) -> StoreResult<()> {
        let project_id = project_id.to_string();
        let symlink_id = symlink_id.to_vec();
        let target = target.to_string();
        self.with_write_conn(move |conn| {
            conn.execute(
                "INSERT OR REPLACE INTO symlinks (project_id, symlink_id, target)
                 VALUES (?1, ?2, ?3)",
                params![project_id, symlink_id, target],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite put_symlink failed: {e}")))?;
            Ok(())
        })
        .await
    }

    async fn find_existing_symlinks(
        &self,
        project_id: &str,
        symlink_ids: &[Vec<u8>],
    ) -> StoreResult<HashSet<Vec<u8>>> {
        if symlink_ids.is_empty() {
            return Ok(HashSet::new());
        }
        let project_id = project_id.to_string();
        let ids = symlink_ids.to_vec();
        self.with_read_conn(move |conn| {
            let mut stmt = conn
                .prepare("SELECT 1 FROM symlinks WHERE project_id = ?1 AND symlink_id = ?2")
                .map_err(|e| {
                    StoreError::Backend(format!(
                        "SQLite prepare find_existing_symlinks failed: {e}"
                    ))
                })?;
            let mut found = HashSet::with_capacity(ids.len());
            for id in ids {
                let exists: Option<i64> = stmt
                    .query_row(params![project_id, id], |r| r.get(0))
                    .optional()
                    .map_err(|e| {
                        StoreError::Backend(format!("SQLite find_existing_symlinks failed: {e}"))
                    })?;
                if exists.is_some() {
                    found.insert(id);
                }
            }
            Ok(found)
        })
        .await
    }

    async fn get_blob_record(
        &self,
        project_id: &str,
        blob_id: &[u8],
    ) -> StoreResult<Option<BlobRecord>> {
        let project_id = project_id.to_string();
        let blob_id = blob_id.to_vec();
        self.with_read_conn(move |conn| {
            let row = conn
                .query_row(
                    "SELECT b.blob_id, b.content_size, b.inline_content, b.primary_location, b.replicated_locations_json
                     FROM blobs b
                     INNER JOIN project_blobs pb ON pb.blob_id = b.blob_id
                     WHERE pb.project_id = ?1 AND b.blob_id = ?2",
                    params![project_id, blob_id],
                    |r| {
                        Ok((
                            r.get::<_, Vec<u8>>(0)?,
                            r.get::<_, i64>(1)?,
                            r.get::<_, Option<Vec<u8>>>(2)?,
                            r.get::<_, Option<String>>(3)?,
                            r.get::<_, String>(4)?,
                        ))
                    },
                )
                .optional()
                .map_err(|e| StoreError::Backend(format!("SQLite get_blob_record failed: {e}")))?;

            row.map(|(id, size, inline, primary, repl_json)| {
                let replicated_locations: Vec<String> = serde_json::from_str(&repl_json)
                    .map_err(|e| {
                        StoreError::Corruption(format!("invalid replicated_locations_json: {e}"))
                    })?;
                Ok(BlobRecord {
                    blob_id: id,
                    content_size: size as u64,
                    inline_content: inline.map(Bytes::from),
                    primary_location: primary,
                    replicated_locations,
                })
            })
            .transpose()
        })
        .await
    }

    async fn get_blob_sizes_batch(
        &self,
        project_id: &str,
        blob_ids: &[Vec<u8>],
    ) -> StoreResult<HashMap<Vec<u8>, u64>> {
        if blob_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let project_id = project_id.to_string();
        let ids = blob_ids.to_vec();
        self.with_read_conn(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT b.content_size
                     FROM blobs b
                     INNER JOIN project_blobs pb ON pb.blob_id = b.blob_id
                     WHERE pb.project_id = ?1 AND pb.blob_id = ?2",
                )
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite prepare get_blob_sizes_batch failed: {e}"))
                })?;
            let mut sizes = HashMap::with_capacity(ids.len());
            for id in ids {
                let sz: Option<i64> = stmt
                    .query_row(params![project_id, id], |r| r.get(0))
                    .optional()
                    .map_err(|e| {
                        StoreError::Backend(format!("SQLite get_blob_sizes_batch failed: {e}"))
                    })?;
                if let Some(s) = sz {
                    sizes.insert(id, s as u64);
                }
            }
            Ok(sizes)
        })
        .await
    }

    async fn get_global_blob_record(&self, blob_id: &[u8]) -> StoreResult<Option<BlobRecord>> {
        let blob_id = blob_id.to_vec();
        self.with_read_conn(move |conn| {
            let row = conn
                .query_row(
                    "SELECT blob_id, content_size, inline_content, primary_location, replicated_locations_json
                     FROM blobs WHERE blob_id = ?1",
                    params![blob_id],
                    |r| {
                        Ok((
                            r.get::<_, Vec<u8>>(0)?,
                            r.get::<_, i64>(1)?,
                            r.get::<_, Option<Vec<u8>>>(2)?,
                            r.get::<_, Option<String>>(3)?,
                            r.get::<_, String>(4)?,
                        ))
                    },
                )
                .optional()
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite get_global_blob_record failed: {e}"))
                })?;

            row.map(|(id, size, inline, primary, repl_json)| {
                let replicated_locations: Vec<String> = serde_json::from_str(&repl_json)
                    .map_err(|e| {
                        StoreError::Corruption(format!("invalid replicated_locations_json: {e}"))
                    })?;
                Ok(BlobRecord {
                    blob_id: id,
                    content_size: size as u64,
                    inline_content: inline.map(Bytes::from),
                    primary_location: primary,
                    replicated_locations,
                })
            })
            .transpose()
        })
        .await
    }

    async fn put_blob_record(&self, project_id: &str, record: &BlobRecord) -> StoreResult<()> {
        let project_id = project_id.to_string();
        let record = record.clone();
        self.with_write_conn(move |conn| {
            let repl_json = serde_json::to_string(&record.replicated_locations).map_err(|e| {
                StoreError::Corruption(format!("failed to serialize replicated_locations: {e}"))
            })?;
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite begin put_blob_record tx failed: {e}"))
                })?;

            tx.execute(
                "INSERT OR REPLACE INTO blobs
                 (blob_id, content_size, inline_content, primary_location, replicated_locations_json)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    record.blob_id,
                    record.content_size as i64,
                    record.inline_content.as_ref().map(|b| b.as_ref()),
                    record.primary_location,
                    repl_json
                ],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite upsert blobs failed: {e}")))?;

            tx.execute(
                "INSERT OR IGNORE INTO project_blobs (project_id, blob_id) VALUES (?1, ?2)",
                params![project_id, record.blob_id],
            )
            .map_err(|e| {
                StoreError::Backend(format!("SQLite insert project_blobs failed: {e}"))
            })?;

            tx.commit().map_err(|e| {
                StoreError::Backend(format!("SQLite commit put_blob_record failed: {e}"))
            })
        })
        .await
    }

    async fn add_blob_replica_location(&self, blob_id: &[u8], location: &str) -> StoreResult<()> {
        let blob_id = blob_id.to_vec();
        let location = location.to_string();
        self.with_write_conn(move |conn| {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|e| {
                    StoreError::Backend(format!(
                        "SQLite begin add_blob_replica_location tx failed: {e}"
                    ))
                })?;

            let existing_json: Option<String> = tx
                .query_row(
                    "SELECT replicated_locations_json FROM blobs WHERE blob_id = ?1",
                    params![blob_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite select blob locations failed: {e}"))
                })?;

            if let Some(raw_json) = existing_json {
                let mut locations: Vec<String> = serde_json::from_str(&raw_json).map_err(|e| {
                    StoreError::Corruption(format!("invalid replicated_locations_json: {e}"))
                })?;
                if !locations.iter().any(|l| l == &location) {
                    locations.push(location);
                    let updated_json = serde_json::to_string(&locations).map_err(|e| {
                        StoreError::Corruption(format!("failed to encode locations: {e}"))
                    })?;
                    tx.execute(
                        "UPDATE blobs SET replicated_locations_json = ?1 WHERE blob_id = ?2",
                        params![updated_json, blob_id],
                    )
                    .map_err(|e| {
                        StoreError::Backend(format!("SQLite update blob locations failed: {e}"))
                    })?;
                }
            }

            tx.commit().map_err(|e| {
                StoreError::Backend(format!(
                    "SQLite commit add_blob_replica_location failed: {e}"
                ))
            })
        })
        .await
    }

    async fn get_operation(&self, repo_id: &str, op_id: &[u8]) -> StoreResult<Option<Operation>> {
        let repo_id = repo_id.to_string();
        let op_id = op_id.to_vec();
        self.with_read_conn(move |conn| {
            let raw: Option<Vec<u8>> = conn
                .query_row(
                    "SELECT proto_bytes FROM operations WHERE repo_id = ?1 AND op_id = ?2",
                    params![repo_id, op_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| StoreError::Backend(format!("SQLite get_operation failed: {e}")))?;

            raw.map(|bytes| {
                Operation::decode(bytes.as_slice()).map_err(|e| {
                    StoreError::Corruption(format!("failed to decode Operation protobuf: {e}"))
                })
            })
            .transpose()
        })
        .await
    }

    async fn put_operation(&self, repo_id: &str, op_id: &[u8], op: &Operation) -> StoreResult<()> {
        let repo_id = repo_id.to_string();
        let op_id = op_id.to_vec();
        let proto_bytes = op.encode_to_vec();
        self.with_write_conn(move |conn| {
            conn.execute(
                "INSERT OR IGNORE INTO operations (repo_id, op_id, proto_bytes)
                 VALUES (?1, ?2, ?3)",
                params![repo_id, op_id, proto_bytes],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite put_operation failed: {e}")))?;
            Ok(())
        })
        .await
    }

    async fn resolve_operation_id_prefix(
        &self,
        repo_id: &str,
        min_id: &[u8],
        max_id: &[u8],
    ) -> StoreResult<Vec<OpId>> {
        let repo_id = repo_id.to_string();
        let min_id = min_id.to_vec();
        let max_id = max_id.to_vec();
        self.with_read_conn(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT op_id FROM operations
                     WHERE repo_id = ?1 AND op_id >= ?2 AND op_id <= ?3
                     ORDER BY op_id ASC
                     LIMIT 2",
                )
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite prepare prefix query failed: {e}"))
                })?;

            let rows = stmt
                .query_map(params![repo_id, min_id, max_id], |row| {
                    row.get::<_, Vec<u8>>(0)
                })
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite execute prefix query failed: {e}"))
                })?;

            let mut matches = Vec::new();
            for row in rows {
                matches.push(row.map_err(|e| {
                    StoreError::Backend(format!("SQLite read prefix match failed: {e}"))
                })?);
            }
            Ok(matches)
        })
        .await
    }

    async fn get_view(&self, repo_id: &str, view_id: &[u8]) -> StoreResult<Option<View>> {
        let repo_id = repo_id.to_string();
        let view_id = view_id.to_vec();
        self.with_read_conn(move |conn| {
            let raw: Option<Vec<u8>> = conn
                .query_row(
                    "SELECT proto_bytes FROM views WHERE repo_id = ?1 AND view_id = ?2",
                    params![repo_id, view_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| StoreError::Backend(format!("SQLite get_view failed: {e}")))?;

            raw.map(|bytes| {
                View::decode(bytes.as_slice()).map_err(|e| {
                    StoreError::Corruption(format!("failed to decode View protobuf: {e}"))
                })
            })
            .transpose()
        })
        .await
    }

    async fn put_view(&self, repo_id: &str, view_id: &[u8], view: &View) -> StoreResult<()> {
        let repo_id = repo_id.to_string();
        let view_id = view_id.to_vec();
        let proto_bytes = view.encode_to_vec();
        self.with_write_conn(move |conn| {
            conn.execute(
                "INSERT OR IGNORE INTO views (repo_id, view_id, proto_bytes)
                 VALUES (?1, ?2, ?3)",
                params![repo_id, view_id, proto_bytes],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite put_view failed: {e}")))?;
            Ok(())
        })
        .await
    }

    async fn get_op_heads(&self, repo_id: &str) -> StoreResult<Option<Vec<OpId>>> {
        let repo_id = repo_id.to_string();
        self.with_read_conn(move |conn| {
            let raw: Option<String> = conn
                .query_row(
                    "SELECT op_heads_json FROM repositories WHERE repo_id = ?1",
                    params![repo_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| StoreError::Backend(format!("SQLite get_op_heads failed: {e}")))?;

            raw.map(|json| decode_op_heads_json(&json)).transpose()
        })
        .await
    }

    async fn update_op_heads(
        &self,
        repo_id: &str,
        old_ids: &[Vec<u8>],
        new_id: &[u8],
    ) -> StoreResult<Vec<OpId>> {
        let repo_id = repo_id.to_string();
        let old_ids = old_ids.to_vec();
        let new_id = new_id.to_vec();

        self.with_write_conn(move |conn| {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite begin update_op_heads tx failed: {e}"))
                })?;

            let raw: String = tx
                .query_row(
                    "SELECT op_heads_json FROM repositories WHERE repo_id = ?1",
                    params![repo_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| StoreError::Backend(format!("SQLite select op_heads failed: {e}")))?
                .ok_or_else(|| StoreError::RepoNotFound {
                    repo_id: repo_id.clone(),
                    action: "updating op heads".to_string(),
                })?;

            let mut current_heads = decode_op_heads_json(&raw)?;
            apply_op_heads_mutation(&mut current_heads, &old_ids, &new_id);
            let encoded = encode_op_heads_json(&current_heads)?;

            tx.execute(
                "UPDATE repositories SET op_heads_json = ?1 WHERE repo_id = ?2",
                params![encoded, repo_id],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite update op_heads failed: {e}")))?;

            tx.commit().map_err(|e| {
                StoreError::Backend(format!("SQLite commit update_op_heads failed: {e}"))
            })?;

            Ok(current_heads)
        })
        .await
    }

    async fn publish_operation_atomic(
        &self,
        repo_id: &str,
        view_id: &[u8],
        view: &View,
        op_id: &[u8],
        op: &Operation,
        old_ids: &[Vec<u8>],
    ) -> StoreResult<Vec<OpId>> {
        let repo_id = repo_id.to_string();
        let view_id = view_id.to_vec();
        let view_bytes = view.encode_to_vec();
        let op_id = op_id.to_vec();
        let op_bytes = op.encode_to_vec();
        let old_ids = old_ids.to_vec();

        self.with_write_conn(move |conn| {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite begin publish_operation tx failed: {e}"))
                })?;

            tx.execute(
                "INSERT OR IGNORE INTO views (repo_id, view_id, proto_bytes) VALUES (?1, ?2, ?3)",
                params![repo_id, view_id, view_bytes],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite publish view failed: {e}")))?;

            tx.execute(
                "INSERT OR IGNORE INTO operations (repo_id, op_id, proto_bytes) VALUES (?1, ?2, ?3)",
                params![repo_id, op_id, op_bytes],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite publish operation failed: {e}")))?;

            let raw: String = tx
                .query_row(
                    "SELECT op_heads_json FROM repositories WHERE repo_id = ?1",
                    params![repo_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| StoreError::Backend(format!("SQLite select op_heads failed: {e}")))?
                .ok_or_else(|| StoreError::RepoNotFound {
                    repo_id: repo_id.clone(),
                    action: "publishing operation".to_string(),
                })?;

            let mut current_heads = decode_op_heads_json(&raw)?;
            apply_op_heads_mutation(&mut current_heads, &old_ids, &op_id);
            let encoded = encode_op_heads_json(&current_heads)?;

            tx.execute(
                "UPDATE repositories SET op_heads_json = ?1 WHERE repo_id = ?2",
                params![encoded, repo_id],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite update op_heads failed: {e}")))?;

            tx.commit().map_err(|e| {
                StoreError::Backend(format!("SQLite commit publish_operation failed: {e}"))
            })?;

            Ok(current_heads)
        })
        .await
    }

    async fn compare_and_swap_op_heads(
        &self,
        repo_id: &str,
        expected_old_ids: &[Vec<u8>],
        new_id: &[u8],
    ) -> StoreResult<()> {
        let repo_id = repo_id.to_string();
        let expected_old_ids = expected_old_ids.to_vec();
        let new_id = new_id.to_vec();

        self.with_write_conn(move |conn| {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite begin cas_op_heads tx failed: {e}"))
                })?;

            let raw: String = tx
                .query_row(
                    "SELECT op_heads_json FROM repositories WHERE repo_id = ?1",
                    params![repo_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| StoreError::Backend(format!("SQLite select op_heads failed: {e}")))?
                .ok_or_else(|| StoreError::RepoNotFound {
                    repo_id: repo_id.clone(),
                    action: "comparing and swapping op heads".to_string(),
                })?;

            let current_heads = decode_op_heads_json(&raw)?;
            if !op_heads_sets_equal(&current_heads, &expected_old_ids) {
                return Err(StoreError::CasConflict(format!(
                    "op heads changed concurrently in repo '{repo_id}'"
                )));
            }

            let encoded = encode_op_heads_json(&[new_id])?;
            tx.execute(
                "UPDATE repositories SET op_heads_json = ?1 WHERE repo_id = ?2",
                params![encoded, repo_id],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite cas op_heads failed: {e}")))?;

            tx.commit()
                .map_err(|e| StoreError::Backend(format!("SQLite commit cas_op_heads failed: {e}")))
        })
        .await
    }

    async fn get_workspace(
        &self,
        repo_id: &str,
        user: &str,
        workspace_name: &str,
    ) -> StoreResult<Option<WorkspaceState>> {
        let repo_id = repo_id.to_string();
        let user = user.to_string();
        let workspace_name = workspace_name.to_string();
        self.with_read_conn(move |conn| {
            let row = conn
                .query_row(
                    "SELECT commit_id, operation_id, tree_id, tree_ids_json, sparse_patterns_json
                     FROM workspaces
                     WHERE repo_id = ?1 AND user_email = ?2 AND workspace_name = ?3",
                    params![repo_id, user, workspace_name],
                    |r| {
                        Ok((
                            r.get::<_, Vec<u8>>(0)?,
                            r.get::<_, Vec<u8>>(1)?,
                            r.get::<_, Vec<u8>>(2)?,
                            r.get::<_, String>(3)?,
                            r.get::<_, String>(4)?,
                        ))
                    },
                )
                .optional()
                .map_err(|e| StoreError::Backend(format!("SQLite get_workspace failed: {e}")))?;

            row.map(
                |(commit_id, operation_id, tree_id, tree_ids_json, sparse_json)| {
                    let mut tree_ids = decode_op_heads_json(&tree_ids_json)?;
                    if tree_ids.is_empty() && !tree_id.is_empty() {
                        tree_ids.push(tree_id.clone());
                    }
                    let sparse_patterns: Vec<String> =
                        serde_json::from_str(&sparse_json).map_err(|e| {
                            StoreError::Corruption(format!("invalid sparse_patterns_json: {e}"))
                        })?;
                    Ok(WorkspaceState {
                        repo_id,
                        user,
                        workspace_name,
                        commit_id,
                        operation_id,
                        tree_id,
                        sparse_patterns,
                        tree_ids,
                    })
                },
            )
            .transpose()
        })
        .await
    }

    async fn put_workspace(&self, workspace: &WorkspaceState) -> StoreResult<()> {
        let ws = workspace.clone();
        self.with_write_conn(move |conn| {
            let tree_ids_json = encode_op_heads_json(&ws.tree_ids)?;
            let sparse_json = serde_json::to_string(&ws.sparse_patterns).map_err(|e| {
                StoreError::Corruption(format!("failed to serialize sparse_patterns: {e}"))
            })?;
            conn.execute(
                "INSERT OR REPLACE INTO workspaces
                 (repo_id, user_email, workspace_name, commit_id, operation_id, tree_id, tree_ids_json, sparse_patterns_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    ws.repo_id,
                    ws.user,
                    ws.workspace_name,
                    ws.commit_id,
                    ws.operation_id,
                    ws.tree_id,
                    tree_ids_json,
                    sparse_json
                ],
            )
            .map_err(|e| StoreError::Backend(format!("SQLite put_workspace failed: {e}")))?;
            Ok(())
        })
        .await
    }

    async fn list_workspaces(&self, repo_id: &str) -> StoreResult<Vec<WorkspaceState>> {
        let repo_id = repo_id.to_string();
        self.with_read_conn(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT user_email, workspace_name, commit_id, operation_id, tree_id, tree_ids_json, sparse_patterns_json
                     FROM workspaces
                     WHERE repo_id = ?1
                     ORDER BY user_email ASC, workspace_name ASC",
                )
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite prepare list_workspaces failed: {e}"))
                })?;

            let rows = stmt
                .query_map(params![repo_id], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Vec<u8>>(2)?,
                        r.get::<_, Vec<u8>>(3)?,
                        r.get::<_, Vec<u8>>(4)?,
                        r.get::<_, String>(5)?,
                        r.get::<_, String>(6)?,
                    ))
                })
                .map_err(|e| {
                    StoreError::Backend(format!("SQLite query list_workspaces failed: {e}"))
                })?;

            let mut out = Vec::new();
            for row in rows {
                let (
                    user,
                    workspace_name,
                    commit_id,
                    operation_id,
                    tree_id,
                    tree_ids_json,
                    sparse_json,
                ) = row.map_err(|e| {
                    StoreError::Backend(format!("SQLite read workspace row failed: {e}"))
                })?;
                let mut tree_ids = decode_op_heads_json(&tree_ids_json)?;
                if tree_ids.is_empty() && !tree_id.is_empty() {
                    tree_ids.push(tree_id.clone());
                }
                let sparse_patterns: Vec<String> = serde_json::from_str(&sparse_json)
                    .map_err(|e| {
                        StoreError::Corruption(format!("invalid sparse_patterns_json: {e}"))
                    })?;
                out.push(WorkspaceState {
                    repo_id: repo_id.clone(),
                    user,
                    workspace_name,
                    commit_id,
                    operation_id,
                    tree_id,
                    sparse_patterns,
                    tree_ids,
                });
            }
            Ok(out)
        })
        .await
    }

    async fn delete_workspace(
        &self,
        repo_id: &str,
        user: &str,
        workspace_name: &str,
    ) -> StoreResult<bool> {
        let repo_id = repo_id.to_string();
        let user = user.to_string();
        let workspace_name = workspace_name.to_string();
        self.with_write_conn(move |conn| {
            let removed = conn
                .execute(
                    "DELETE FROM workspaces WHERE repo_id = ?1 AND user_email = ?2 AND workspace_name = ?3",
                    params![repo_id, user, workspace_name],
                )
                .map_err(|e| StoreError::Backend(format!("SQLite delete_workspace failed: {e}")))?;
            Ok(removed > 0)
        })
        .await
    }
}
