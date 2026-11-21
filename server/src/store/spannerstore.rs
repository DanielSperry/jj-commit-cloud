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

//! Cloud Spanner and Spanner Omni `MetadataStore` implementation.
//!
//! # Authority and Crash-Safety Ordering
//! `SpannerMetadataStore` shares one long-lived `Spanner` client and multiplexed
//! `DatabaseClient` across all RPCs. Content-addressed writes (`Commits`, `Trees` +
//! `TreeEntries`, `Symlinks`, `Views`, `Operations`, `Blobs` + `ProjectBlobs`) commit
//! atomically per call after `CompositeStore` has completed any external `BlobStore`
//! upload and batched dependency validation. `Commits` rows use idempotent insert-only
//! mutations so published commits cannot be overwritten.
//!
//! # Concurrency and Merge Invariants
//! Point reads, batched `UNNEST` dependency checks, and prefix range scans execute as
//! lock-free strong `single_use()` reads. Mutable coordination points (`register_repo`,
//! `update_op_heads`, `publish_operation_atomic`, and `add_blob_replica_location`)
//! execute inside `DatabaseClient::read_write_transaction` (`TransactionRunner`),
//! retrying automatically on Spanner `ABORTED` conflicts so concurrent sibling operation
//! heads are never overwritten.
//!
//! # Cross-Process and RPC Boundaries
//! Supports Google Cloud Spanner via Application Default Credentials, containerized
//! Spanner Omni via custom endpoints and CA/mTLS certificates, and the Cloud Spanner
//! Emulator (`SPANNER_EMULATOR_HOST`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use cc_common::backend::*;
use cc_common::op_store::*;
use cc_common::workspace::WorkspaceState;
use google_cloud_gax::error::rpc::{Code as RpcCode, Status as RpcStatus};
use google_cloud_spanner::channel_pool::StaticChannelPoolConfig;
use google_cloud_spanner::client::{DatabaseClient, Spanner, SpannerBuilderExt};
use google_cloud_spanner::key::{Key, KeySet};
use google_cloud_spanner::mutation::Mutation;
use google_cloud_spanner::omni::{InstanceType, TlsConfig};
use google_cloud_spanner::statement::Statement;
use prost::Message;
use tracing::{info, warn};

use super::{
    BlobRecord, CommitId, MetadataStore, OpId, StoreError, StoreResult, apply_op_heads_mutation,
    op_heads_sets_equal,
};

const SPANNER_SCHEMA: &str = include_str!("../../sql/spanner_schema.sql");
const UNNEST_BATCH_SIZE: usize = 5_000;

#[derive(Clone, Debug, Default)]
pub struct SpannerConnectConfig {
    /// Full database resource path: `projects/<project>/instances/<instance>/databases/<db>`.
    pub database: String,
    /// Optional gRPC endpoint override (e.g. `http://127.0.0.1:9010` or Spanner Omni URL).
    pub endpoint: Option<String>,
    /// Optional HTTP/REST admin endpoint override for emulator DDL administration.
    pub admin_endpoint: Option<String>,
    /// Connect in Spanner Omni mode (`InstanceType::Omni`).
    pub omni: bool,
    /// Optional Root CA PEM path for Spanner Omni TLS verification.
    pub omni_ca_cert: Option<PathBuf>,
    /// Optional client certificate PEM path for Spanner Omni mTLS.
    pub omni_client_cert: Option<PathBuf>,
    /// Optional client private key PEM path for Spanner Omni mTLS.
    pub omni_client_key: Option<PathBuf>,
    /// Create the database if missing on the instance and apply `spanner_schema.sql`.
    pub auto_migrate: bool,
}

#[derive(Clone)]
pub struct SpannerMetadataStore {
    _spanner: Spanner,
    db_client: DatabaseClient,
}

impl SpannerMetadataStore {
    pub async fn connect(config: &SpannerConnectConfig) -> StoreResult<Self> {
        let mut builder = Spanner::builder();

        if let Some(ep) = &config.endpoint
            && !ep.trim().is_empty()
        {
            let normalized = normalize_endpoint(ep);
            if normalized.starts_with("http://") {
                builder = builder.with_channel_pool(StaticChannelPoolConfig::new(1));
            }
            builder = builder.with_endpoint(normalized);
        }

        if config.omni
            || config.omni_ca_cert.is_some()
            || config.omni_client_cert.is_some()
            || config.omni_client_key.is_some()
        {
            builder = builder.with_instance_type(InstanceType::Omni);
            if config.omni_ca_cert.is_some()
                || config.omni_client_cert.is_some()
                || config.omni_client_key.is_some()
            {
                let mut tls = TlsConfig::new();
                if let Some(ca_path) = &config.omni_ca_cert {
                    tls = tls.with_root_certificate_file(ca_path).map_err(|e| {
                        StoreError::InvalidArgument(format!(
                            "failed to load Spanner Omni CA certificate {}: {e}",
                            ca_path.display()
                        ))
                    })?;
                }
                match (&config.omni_client_cert, &config.omni_client_key) {
                    (Some(cert), Some(key)) => {
                        tls = tls.with_client_certificate_file(cert, key).map_err(|e| {
                            StoreError::InvalidArgument(format!(
                                "failed to load Spanner Omni mTLS identity: {e}"
                            ))
                        })?;
                    }
                    (Some(_), None) | (None, Some(_)) => {
                        return Err(StoreError::InvalidArgument(
                            "both --spanner-omni-client-cert and --spanner-omni-client-key must be set together"
                                .to_string(),
                        ));
                    }
                    (None, None) => {}
                }
                builder = builder.with_omni_tls(tls);
            }
        }

        let spanner = builder
            .build()
            .await
            .map_err(|e| StoreError::Unavailable(format!("failed to build Spanner client: {e}")))?;

        if config.auto_migrate {
            ensure_database_and_schema(&spanner, config).await?;
        }

        let db_client = spanner
            .database_client(&config.database)
            .build()
            .await
            .map_err(|e| {
                StoreError::Unavailable(format!(
                    "failed to initialize Spanner DatabaseClient for '{}': {e}",
                    config.database
                ))
            })?;

        Ok(Self {
            _spanner: spanner,
            db_client,
        })
    }
}

fn normalize_endpoint(ep: &str) -> String {
    let trimmed = ep.trim();
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    }
}

fn parse_ddl_statements(sql: &str) -> Vec<String> {
    sql.split(';')
        .map(|stmt| {
            stmt.lines()
                .filter(|line| !line.trim_start().starts_with("--"))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn split_database_resource(database: &str) -> StoreResult<(&str, &str)> {
    let (instance_path, db_id) = database.rsplit_once("/databases/").ok_or_else(|| {
        StoreError::InvalidArgument(format!(
            "invalid Spanner database path '{database}' (expected projects/<p>/instances/<i>/databases/<d>)"
        ))
    })?;
    if !instance_path.starts_with("projects/")
        || !instance_path.contains("/instances/")
        || db_id.is_empty()
    {
        return Err(StoreError::InvalidArgument(format!(
            "invalid Spanner database path '{database}' (expected projects/<p>/instances/<i>/databases/<d>)"
        )));
    }
    Ok((instance_path, db_id))
}

async fn ensure_database_and_schema(
    spanner: &Spanner,
    config: &SpannerConnectConfig,
) -> StoreResult<()> {
    let (instance_path, db_id) = split_database_resource(&config.database)?;
    let ddl_statements = parse_ddl_statements(SPANNER_SCHEMA);

    let resolved_admin_ep = config
        .admin_endpoint
        .clone()
        .or_else(|| std::env::var("SPANNER_EMULATOR_REST_HOST").ok())
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            std::env::var("SPANNER_EMULATOR_HOST")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .map(|h| {
                    let clean = h.trim().trim_start_matches("http://");
                    clean
                        .strip_suffix(":9010")
                        .map(|p| format!("http://{p}:9020"))
                        .unwrap_or_else(|| format!("http://{clean}"))
                })
        });

    if let Some(ep) = &resolved_admin_ep {
        let normalized = normalize_endpoint(ep);
        if let Some(host_port) = normalized.strip_prefix("http://") {
            return ensure_database_and_schema_emulator_http10(
                host_port,
                instance_path,
                db_id,
                &config.database,
                &ddl_statements,
            )
            .await;
        }
    }

    let mut admin_builder = spanner.database_admin_builder();
    if let Some(ep) = resolved_admin_ep {
        admin_builder = admin_builder.with_endpoint(normalize_endpoint(&ep));
    }

    let admin_client = admin_builder.build().await.map_err(|e| {
        StoreError::Unavailable(format!("failed to build Spanner DatabaseAdmin client: {e}"))
    })?;

    match admin_client
        .get_database()
        .set_name(&config.database)
        .send()
        .await
    {
        Ok(_) => {
            if !ddl_statements.is_empty() {
                info!("Applying idempotent Spanner DDL to {}", config.database);
                let op = admin_client
                    .update_database_ddl()
                    .set_database(&config.database)
                    .set_statements(ddl_statements)
                    .send()
                    .await
                    .map_err(map_spanner_admin_err)?;
                let op_error = op.error().map(|s| RpcStatus::from(&**s));
                wait_admin_op(&admin_client, op.name, op.done, op_error).await?;
            }
        }
        Err(err) if is_not_found(&err) => {
            info!(
                "Creating Spanner database {} under {}",
                db_id, instance_path
            );
            let create_res = admin_client
                .create_database()
                .set_parent(instance_path)
                .set_create_statement(format!("CREATE DATABASE `{db_id}`"))
                .set_extra_statements(ddl_statements.clone())
                .send()
                .await;
            match create_res {
                Ok(op) => {
                    let op_error = op.error().map(|s| RpcStatus::from(&**s));
                    wait_admin_op(&admin_client, op.name, op.done, op_error).await?;
                }
                Err(e) if is_already_exists(&e) => {
                    warn!(
                        "Database {} was created concurrently; ensuring DDL",
                        config.database
                    );
                    let op = admin_client
                        .update_database_ddl()
                        .set_database(&config.database)
                        .set_statements(ddl_statements)
                        .send()
                        .await
                        .map_err(map_spanner_admin_err)?;
                    let op_error = op.error().map(|s| RpcStatus::from(&**s));
                    wait_admin_op(&admin_client, op.name, op.done, op_error).await?;
                }
                Err(e) => return Err(map_spanner_admin_err(e)),
            }
        }
        Err(err) => return Err(map_spanner_admin_err(err)),
    }

    Ok(())
}

async fn ensure_database_and_schema_emulator_http10(
    host_port: &str,
    instance_path: &str,
    db_id: &str,
    database_path: &str,
    ddl_statements: &[String],
) -> StoreResult<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn http10_json(
        host_port: &str,
        method: &str,
        path: &str,
        body: &str,
    ) -> StoreResult<(u16, String)> {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(15);
        let mut stream = loop {
            match tokio::net::TcpStream::connect(host_port).await {
                Ok(s) => break s,
                Err(_e) if tokio::time::Instant::now() < deadline => {
                    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                }
                Err(e) => {
                    return Err(StoreError::Unavailable(format!(
                        "failed to connect to Spanner emulator admin {host_port}: {e}"
                    )));
                }
            }
        };
        // HTTP/1.0 avoids chunked binary gRPC server-stats trailers emitted by gateway_main.
        let req = format!(
            "{method} {path} HTTP/1.0\r\nHost: {host_port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(req.as_bytes()).await.map_err(|e| {
            StoreError::Unavailable(format!("failed writing to emulator admin {host_port}: {e}"))
        })?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).await.map_err(|e| {
            StoreError::Unavailable(format!(
                "failed reading from emulator admin {host_port}: {e}"
            ))
        })?;
        let resp = String::from_utf8_lossy(&raw).into_owned();
        let code = resp
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|c| c.parse::<u16>().ok())
            .unwrap_or(0);
        Ok((code, resp))
    }

    if let Some((project_path, instance_id)) = instance_path.rsplit_once("/instances/") {
        let inst_payload = serde_json::json!({
            "instanceId": instance_id,
            "instance": {
                "config": format!("{project_path}/instanceConfigs/emulator-config"),
                "displayName": instance_id,
                "nodeCount": 1,
            }
        })
        .to_string();
        let inst_url_path = format!("/v1/{project_path}/instances");
        let (inst_status, inst_resp) =
            http10_json(host_port, "POST", &inst_url_path, &inst_payload).await?;
        if inst_status != 200 && inst_status != 409 {
            return Err(StoreError::Backend(format!(
                "Spanner emulator create_instance failed (HTTP {inst_status}): {inst_resp}"
            )));
        }
    }

    let create_payload = serde_json::json!({
        "createStatement": format!("CREATE DATABASE `{db_id}`"),
        "extraStatements": ddl_statements,
    })
    .to_string();
    let create_path = format!("/v1/{instance_path}/databases");
    let (status, resp) = http10_json(host_port, "POST", &create_path, &create_payload).await?;
    if status == 200 {
        return check_emulator_lro_json(&resp);
    }
    if status == 409 {
        if ddl_statements.is_empty() {
            return Ok(());
        }
        let patch_payload = serde_json::json!({ "statements": ddl_statements }).to_string();
        let patch_path = format!("/v1/{database_path}/ddl");
        let (ddl_status, ddl_resp) =
            http10_json(host_port, "PATCH", &patch_path, &patch_payload).await?;
        if ddl_status == 200 {
            return check_emulator_lro_json(&ddl_resp);
        }
        return Err(StoreError::Backend(format!(
            "Spanner emulator update_database_ddl failed (HTTP {ddl_status}): {ddl_resp}"
        )));
    }
    Err(StoreError::Backend(format!(
        "Spanner emulator create_database failed (HTTP {status}): {resp}"
    )))
}

fn check_emulator_lro_json(http_resp: &str) -> StoreResult<()> {
    let body = http_resp
        .split_once("\r\n\r\n")
        .map_or(http_resp, |(_, b)| b);
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body)
        && let Some(op_err) = json.get("error").filter(|v| !v.is_null())
    {
        return Err(StoreError::Backend(format!(
            "Spanner emulator DDL operation failed: {op_err}"
        )));
    }
    Ok(())
}

async fn wait_admin_op(
    admin_client: &google_cloud_spanner::client::DatabaseAdmin,
    op_name: String,
    already_done: bool,
    op_error: Option<RpcStatus>,
) -> StoreResult<()> {
    if let Some(outcome) = finished_admin_op(already_done, op_error) {
        return outcome;
    }
    if op_name.is_empty() {
        return Ok(());
    }
    for _ in 0..120 {
        let op = admin_client
            .get_operation()
            .set_name(&op_name)
            .send()
            .await
            .map_err(map_spanner_admin_err)?;
        if let Some(outcome) = finished_admin_op(op.done, op.error().map(|s| RpcStatus::from(&**s)))
        {
            return outcome;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    Err(StoreError::Unavailable(format!(
        "timed out waiting for Spanner DDL operation '{op_name}'"
    )))
}

/// Returns the outcome of a long-running admin operation once it has finished. DDL statements
/// that fail while they run, rather than during validation, are reported only through the
/// operation's error result.
fn finished_admin_op(done: bool, op_error: Option<RpcStatus>) -> Option<StoreResult<()>> {
    match op_error {
        Some(status) => Some(Err(map_spanner_admin_err(
            google_cloud_gax::error::Error::service(status),
        ))),
        None if done => Some(Ok(())),
        None => None,
    }
}

fn is_not_found(err: &google_cloud_gax::error::Error) -> bool {
    err.status().is_some_and(|s| s.code == RpcCode::NotFound)
        || err.http_status_code().is_some_and(|c| c == 404)
}

fn is_already_exists(err: &google_cloud_gax::error::Error) -> bool {
    err.status()
        .is_some_and(|s| s.code == RpcCode::AlreadyExists)
        || err.http_status_code().is_some_and(|c| c == 409)
}

fn map_spanner_admin_err(err: google_cloud_gax::error::Error) -> StoreError {
    map_spanner_err(err)
}

fn map_spanner_err(err: google_cloud_spanner::Error) -> StoreError {
    if let Some(status) = err.status() {
        match status.code {
            RpcCode::InvalidArgument => StoreError::InvalidArgument(err.to_string()),
            RpcCode::OutOfRange => StoreError::OutOfRange(err.to_string()),
            // An ABORTED transaction that reaches this point can be retried as a whole call,
            // which is what UNAVAILABLE tells gRPC clients.
            RpcCode::Unavailable | RpcCode::DeadlineExceeded | RpcCode::Aborted => {
                StoreError::Unavailable(err.to_string())
            }
            RpcCode::ResourceExhausted => StoreError::ResourceExhausted(err.to_string()),
            RpcCode::PermissionDenied => StoreError::PermissionDenied(err.to_string()),
            RpcCode::Unauthenticated => StoreError::Unauthenticated(err.to_string()),
            _ => StoreError::Backend(err.to_string()),
        }
    } else {
        StoreError::Backend(err.to_string())
    }
}

fn encode_spanner_watermark(ts_micros: i64, commit_id: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + commit_id.len());
    out.extend_from_slice(&ts_micros.to_be_bytes());
    out.extend_from_slice(commit_id);
    out
}

fn decode_spanner_watermark(token: &[u8]) -> StoreResult<(i64, Vec<u8>)> {
    if token.len() < 8 {
        return Err(StoreError::InvalidArgument(
            "invalid Spanner commit sync token".to_string(),
        ));
    }
    let ts_micros = i64::from_be_bytes(token[..8].try_into().unwrap());
    Ok((ts_micros, token[8..].to_vec()))
}

#[async_trait]
impl MetadataStore for SpannerMetadataStore {
    fn backend_name(&self) -> &'static str {
        "spanner"
    }

    async fn register_project(&self, project_id: &str, name: Option<&str>) -> StoreResult<()> {
        let project_id = project_id.to_string();
        let name = name.map(ToOwned::to_owned);
        let runner = self.db_client.read_write_transaction().build();

        runner
            .run(
                async move |tx| -> google_cloud_spanner::Result<StoreResult<()>> {
                    if let Some(alias) = &name {
                        let alias_stmt =
                            Statement::builder("SELECT ProjectId FROM Projects WHERE Name = @name")
                                .add_param("name", alias)
                                .build();
                        let mut alias_rs = tx.execute_query(alias_stmt).await?;
                        if let Some(row) = alias_rs.next().await.transpose()? {
                            let existing_id: String = row.get("ProjectId")?;
                            if existing_id != project_id {
                                return Ok(Err(StoreError::ProjectAlreadyExists(alias.clone())));
                            }
                        }
                    }

                    let check_stmt = Statement::builder(
                        "SELECT Name FROM Projects WHERE ProjectId = @project_id",
                    )
                    .add_param("project_id", &project_id)
                    .build();
                    let mut check_rs = tx.execute_query(check_stmt).await?;
                    let effective_name: Option<String> =
                        if let Some(row) = check_rs.next().await.transpose()? {
                            let prev_name: Option<String> = row.get("Name")?;
                            name.clone().or(prev_name)
                        } else {
                            name.clone()
                        };

                    let mutation = Mutation::new_insert_or_update_builder("Projects")
                        .set("ProjectId")
                        .to(&project_id)
                        .set("Name")
                        .to(&effective_name)
                        .build();
                    tx.buffer([mutation])?;
                    Ok(Ok(()))
                },
            )
            .await
            .map_err(map_spanner_err)?
            .into_inner()
    }

    async fn is_project_registered(&self, project_id: &str) -> StoreResult<bool> {
        let stmt =
            Statement::builder("SELECT 1 FROM Projects WHERE ProjectId = @project_id LIMIT 1")
                .add_param("project_id", project_id)
                .build();
        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        Ok(rs
            .next()
            .await
            .transpose()
            .map_err(map_spanner_err)?
            .is_some())
    }

    async fn get_repo_project_id(&self, repo_id: &str) -> StoreResult<Option<String>> {
        let stmt = Statement::builder("SELECT ProjectId FROM Repositories WHERE RepoId = @repo_id")
            .add_param("repo_id", repo_id)
            .build();
        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? else {
            return Ok(None);
        };
        let project_id: String = row.get("ProjectId").map_err(map_spanner_err)?;
        Ok(Some(project_id))
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
        let runner = self.db_client.read_write_transaction().build();

        runner
            .run(
                async move |tx| -> google_cloud_spanner::Result<StoreResult<String>> {
                    if let Some(alias) = &name {
                        let alias_stmt = Statement::builder(
                            "SELECT RepoId FROM Repositories WHERE Name = @name",
                        )
                        .add_param("name", alias)
                        .build();
                        let mut alias_rs = tx.execute_query(alias_stmt).await?;
                        if let Some(row) = alias_rs.next().await.transpose()? {
                            let existing_id: String = row.get("RepoId")?;
                            if existing_id != repo_id {
                                return Ok(Err(StoreError::RepoAlreadyExists(alias.clone())));
                            }
                        }
                    }

                    let check_stmt = Statement::builder(
                        "SELECT ProjectId, Name, OpHeads FROM Repositories WHERE RepoId = @repo_id",
                    )
                    .add_param("repo_id", &repo_id)
                    .build();
                    let mut check_rs = tx.execute_query(check_stmt).await?;

                    let (effective_name, op_heads): (Option<String>, Vec<Vec<u8>>) =
                        if let Some(row) = check_rs.next().await.transpose()? {
                            let bound_project: String = row.get("ProjectId")?;
                            if bound_project != project_id {
                                return Ok(Err(StoreError::InvalidArgument(format!(
                                    "repository '{repo_id}' is already bound to project '{bound_project}', cannot rebind to '{project_id}'"
                                ))));
                            }
                            let prev_name: Option<String> = row.get("Name")?;
                            let prev_heads: Vec<Vec<u8>> = row.get("OpHeads")?;
                            (name.clone().or(prev_name), prev_heads)
                        } else {
                            (
                                name.clone(),
                                vec![cc_common::ROOT_OPERATION_ID_BYTES.to_vec()],
                            )
                        };

                    let proj_stmt = Statement::builder(
                        "SELECT 1 FROM Projects WHERE ProjectId = @project_id",
                    )
                    .add_param("project_id", &project_id)
                    .build();
                    let mut proj_rs = tx.execute_query(proj_stmt).await?;
                    let mut mutations = Vec::with_capacity(2);
                    if proj_rs.next().await.transpose()?.is_none() {
                        let none_name: Option<String> = None;
                        mutations.push(
                            Mutation::new_insert_builder("Projects")
                                .set("ProjectId")
                                .to(&project_id)
                                .set("Name")
                                .to(&none_name)
                                .build(),
                        );
                    }

                    mutations.push(
                        Mutation::new_insert_or_update_builder("Repositories")
                            .set("RepoId")
                            .to(&repo_id)
                            .set("ProjectId")
                            .to(&project_id)
                            .set("Name")
                            .to(&effective_name)
                            .set("OpHeads")
                            .to(&op_heads)
                            .build(),
                    );
                    tx.buffer(mutations)?;
                    Ok(Ok(project_id.clone()))
                },
            )
            .await
            .map_err(map_spanner_err)?
            .into_inner()
    }

    async fn get_commit(&self, project_id: &str, commit_id: &[u8]) -> StoreResult<Option<Commit>> {
        let stmt = Statement::builder(
            "SELECT ProtoBytes FROM Commits WHERE ProjectId = @project_id AND CommitId = @commit_id",
        )
        .add_param("project_id", project_id)
        .add_param("commit_id", commit_id)
        .build();
        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? else {
            return Ok(None);
        };
        let proto_bytes: Vec<u8> = row.get("ProtoBytes").map_err(map_spanner_err)?;
        let commit = Commit::decode(proto_bytes.as_slice())
            .map_err(|e| StoreError::Corruption(format!("failed to decode Spanner Commit: {e}")))?;
        Ok(Some(commit))
    }

    async fn get_commit_generation(
        &self,
        project_id: &str,
        commit_id: &[u8],
    ) -> StoreResult<Option<u64>> {
        let stmt = Statement::builder(
            "SELECT Generation FROM Commits WHERE ProjectId = @project_id AND CommitId = @commit_id",
        )
        .add_param("project_id", project_id)
        .add_param("commit_id", commit_id)
        .build();
        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? else {
            return Ok(None);
        };
        let gen_i64: i64 = row.get("Generation").map_err(map_spanner_err)?;
        Ok(Some(gen_i64 as u64))
    }

    async fn put_commit(
        &self,
        project_id: &str,
        commit_id: &[u8],
        commit: &Commit,
        generation: u64,
    ) -> StoreResult<()> {
        let proto_bytes = commit.encode_to_vec();
        let mutation = Mutation::new_insert_builder("Commits")
            .set("ProjectId")
            .to(project_id)
            .set("CommitId")
            .to(commit_id)
            .set("ChangeId")
            .to(&commit.change_id)
            .set("Generation")
            .to(generation as i64)
            .set("ProtoBytes")
            .to(&proto_bytes)
            .set("InsertedAt")
            .to("spanner.commit_timestamp()")
            .build();

        match self
            .db_client
            .write_only_transaction()
            .build()
            .write(vec![mutation])
            .await
        {
            Ok(_) => Ok(()),
            Err(err) if is_already_exists(&err) => Ok(()),
            Err(err) => Err(map_spanner_err(err)),
        }
    }

    async fn list_commit_ids(
        &self,
        project_id: &str,
        after_commit_id: Option<&[u8]>,
        limit: usize,
    ) -> StoreResult<Vec<CommitId>> {
        let stmt = match after_commit_id.filter(|c| !c.is_empty()) {
            Some(after) => Statement::builder(
                "SELECT CommitId FROM Commits
                 WHERE ProjectId = @project_id AND CommitId > @after_id
                 ORDER BY CommitId ASC
                 LIMIT @limit",
            )
            .add_param("project_id", project_id)
            .add_param("after_id", after)
            .add_param("limit", limit as i64)
            .build(),
            None => Statement::builder(
                "SELECT CommitId FROM Commits
                 WHERE ProjectId = @project_id
                 ORDER BY CommitId ASC
                 LIMIT @limit",
            )
            .add_param("project_id", project_id)
            .add_param("limit", limit as i64)
            .build(),
        };

        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let mut ids = Vec::new();
        while let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? {
            let cid: Vec<u8> = row.get("CommitId").map_err(map_spanner_err)?;
            ids.push(cid);
        }
        Ok(ids)
    }

    async fn current_commit_watermark(&self, project_id: &str) -> StoreResult<Vec<u8>> {
        let stmt = Statement::builder(
            "SELECT UNIX_MICROS(InsertedAt) AS TsMicros, CommitId
             FROM Commits@{FORCE_INDEX=CommitsByInsertedAt}
             WHERE ProjectId = @project_id AND InsertedAt IS NOT NULL
             ORDER BY InsertedAt DESC, CommitId DESC
             LIMIT 1",
        )
        .add_param("project_id", project_id)
        .build();
        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        if let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? {
            let ts_micros: i64 = row.get("TsMicros").map_err(map_spanner_err)?;
            let cid: Vec<u8> = row.get("CommitId").map_err(map_spanner_err)?;
            Ok(encode_spanner_watermark(ts_micros, &cid))
        } else {
            Ok(encode_spanner_watermark(0, &[]))
        }
    }

    async fn list_commit_ids_since(
        &self,
        project_id: &str,
        since_token: &[u8],
        limit: usize,
    ) -> StoreResult<(Vec<CommitId>, Vec<u8>, bool)> {
        let (ts_micros, after_id) = decode_spanner_watermark(since_token)?;
        let fetch_limit = (limit + 1) as i64;
        let stmt = Statement::builder(
            "SELECT UNIX_MICROS(InsertedAt) AS TsMicros, CommitId
             FROM Commits@{FORCE_INDEX=CommitsByInsertedAt}
             WHERE ProjectId = @project_id
               AND InsertedAt IS NOT NULL
               AND (
                 InsertedAt > TIMESTAMP_MICROS(@ts_micros)
                 OR (InsertedAt = TIMESTAMP_MICROS(@ts_micros) AND CommitId > @after_id)
               )
             ORDER BY InsertedAt ASC, CommitId ASC
             LIMIT @limit",
        )
        .add_param("project_id", project_id)
        .add_param("ts_micros", ts_micros)
        .add_param("after_id", &after_id)
        .add_param("limit", fetch_limit)
        .build();

        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let mut items = Vec::new();
        while let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? {
            let ts: i64 = row.get("TsMicros").map_err(map_spanner_err)?;
            let cid: Vec<u8> = row.get("CommitId").map_err(map_spanner_err)?;
            items.push((ts, cid));
        }

        let has_more = items.len() > limit;
        if has_more {
            items.truncate(limit);
        }
        let Some((last_ts, last_cid)) = items.last() else {
            return Ok((Vec::new(), since_token.to_vec(), false));
        };
        let next_tok = encode_spanner_watermark(*last_ts, last_cid);
        let ids = items.into_iter().map(|(_, cid)| cid).collect();
        Ok((ids, next_tok, has_more))
    }

    async fn resolve_commit_id_prefix(
        &self,
        project_id: &str,
        min_id: &[u8],
        max_id: &[u8],
    ) -> StoreResult<Vec<CommitId>> {
        let stmt = Statement::builder(
            "SELECT CommitId FROM Commits
             WHERE ProjectId = @project_id AND CommitId >= @min_id AND CommitId <= @max_id
             ORDER BY CommitId ASC
             LIMIT 2",
        )
        .add_param("project_id", project_id)
        .add_param("min_id", min_id)
        .add_param("max_id", max_id)
        .build();

        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let mut matches = Vec::new();
        while let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? {
            let cid: Vec<u8> = row.get("CommitId").map_err(map_spanner_err)?;
            matches.push(cid);
        }
        Ok(matches)
    }

    async fn resolve_change_id_prefix(
        &self,
        project_id: &str,
        min_id: &[u8],
        max_id: &[u8],
    ) -> StoreResult<Vec<(Vec<u8>, Vec<CommitId>)>> {
        let ro_tx = self
            .db_client
            .read_only_transaction()
            .build()
            .await
            .map_err(map_spanner_err)?;

        let ch_stmt = Statement::builder(
            "SELECT DISTINCT ChangeId FROM Commits@{FORCE_INDEX=CommitsByChangeId}
             WHERE ProjectId = @project_id AND ChangeId >= @min_id AND ChangeId <= @max_id
             ORDER BY ChangeId ASC
             LIMIT 2",
        )
        .add_param("project_id", project_id)
        .add_param("min_id", min_id)
        .add_param("max_id", max_id)
        .build();

        let mut ch_rs = ro_tx
            .execute_query(ch_stmt)
            .await
            .map_err(map_spanner_err)?;
        let mut change_ids = Vec::new();
        while let Some(row) = ch_rs.next().await.transpose().map_err(map_spanner_err)? {
            let ch: Vec<u8> = row.get("ChangeId").map_err(map_spanner_err)?;
            change_ids.push(ch);
        }
        if change_ids.len() != 1 {
            return Ok(change_ids.into_iter().map(|ch| (ch, Vec::new())).collect());
        }

        let matched_ch = change_ids.swap_remove(0);
        let c_stmt = Statement::builder(
            "SELECT CommitId FROM Commits@{FORCE_INDEX=CommitsByChangeId}
             WHERE ProjectId = @project_id AND ChangeId = @change_id
             ORDER BY CommitId ASC",
        )
        .add_param("project_id", project_id)
        .add_param("change_id", &matched_ch)
        .build();
        let mut c_rs = ro_tx.execute_query(c_stmt).await.map_err(map_spanner_err)?;
        let mut by_change = BTreeMap::new();
        let mut cids = Vec::new();
        while let Some(row) = c_rs.next().await.transpose().map_err(map_spanner_err)? {
            let cid: Vec<u8> = row.get("CommitId").map_err(map_spanner_err)?;
            cids.push(cid);
        }
        by_change.insert(matched_ch, cids);
        Ok(by_change.into_iter().collect())
    }

    async fn get_tree(
        &self,
        project_id: &str,
        tree_id: &[u8],
    ) -> StoreResult<Option<Vec<TreeEntry>>> {
        let ro_tx = self
            .db_client
            .read_only_transaction()
            .build()
            .await
            .map_err(map_spanner_err)?;

        let tree_stmt = Statement::builder(
            "SELECT 1 FROM Trees WHERE ProjectId = @project_id AND TreeId = @tree_id AND Ready = TRUE",
        )
        .add_param("project_id", project_id)
        .add_param("tree_id", tree_id)
        .build();
        let mut tree_rs = ro_tx
            .execute_query(tree_stmt)
            .await
            .map_err(map_spanner_err)?;
        if tree_rs
            .next()
            .await
            .transpose()
            .map_err(map_spanner_err)?
            .is_none()
        {
            return Ok(None);
        }

        let entries_stmt = Statement::builder(
            "SELECT EntryProto FROM TreeEntries
             WHERE ProjectId = @project_id AND TreeId = @tree_id
             ORDER BY Name ASC",
        )
        .add_param("project_id", project_id)
        .add_param("tree_id", tree_id)
        .build();
        let mut entries_rs = ro_tx
            .execute_query(entries_stmt)
            .await
            .map_err(map_spanner_err)?;

        let mut entries = Vec::new();
        while let Some(row) = entries_rs
            .next()
            .await
            .transpose()
            .map_err(map_spanner_err)?
        {
            let bytes: Vec<u8> = row.get("EntryProto").map_err(map_spanner_err)?;
            let entry = TreeEntry::decode(bytes.as_slice()).map_err(|e| {
                StoreError::Corruption(format!("failed to decode Spanner TreeEntry: {e}"))
            })?;
            entries.push(entry);
        }
        Ok(Some(entries))
    }

    async fn get_tree_page(
        &self,
        project_id: &str,
        tree_id: &[u8],
        after_name: Option<&str>,
        limit: usize,
    ) -> StoreResult<Option<Vec<TreeEntry>>> {
        let ro_tx = self
            .db_client
            .read_only_transaction()
            .build()
            .await
            .map_err(map_spanner_err)?;

        let tree_stmt = Statement::builder(
            "SELECT 1 FROM Trees WHERE ProjectId = @project_id AND TreeId = @tree_id AND Ready = TRUE",
        )
        .add_param("project_id", project_id)
        .add_param("tree_id", tree_id)
        .build();
        let mut tree_rs = ro_tx
            .execute_query(tree_stmt)
            .await
            .map_err(map_spanner_err)?;
        if tree_rs
            .next()
            .await
            .transpose()
            .map_err(map_spanner_err)?
            .is_none()
        {
            return Ok(None);
        }

        let after = after_name.unwrap_or("");
        let entries_stmt = Statement::builder(
            "SELECT EntryProto FROM TreeEntries
             WHERE ProjectId = @project_id AND TreeId = @tree_id AND Name > @after_name
             ORDER BY Name ASC
             LIMIT @limit",
        )
        .add_param("project_id", project_id)
        .add_param("tree_id", tree_id)
        .add_param("after_name", after)
        .add_param("limit", limit as i64)
        .build();
        let mut entries_rs = ro_tx
            .execute_query(entries_stmt)
            .await
            .map_err(map_spanner_err)?;

        let mut entries = Vec::new();
        while let Some(row) = entries_rs
            .next()
            .await
            .transpose()
            .map_err(map_spanner_err)?
        {
            let bytes: Vec<u8> = row.get("EntryProto").map_err(map_spanner_err)?;
            let entry = TreeEntry::decode(bytes.as_slice()).map_err(|e| {
                StoreError::Corruption(format!("failed to decode Spanner TreeEntry: {e}"))
            })?;
            entries.push(entry);
        }
        Ok(Some(entries))
    }

    async fn put_tree(
        &self,
        project_id: &str,
        tree_id: &[u8],
        entries: &[TreeEntry],
    ) -> StoreResult<()> {
        const MAX_ENTRIES_PER_BATCH: usize = 4_000;

        if entries.len() <= MAX_ENTRIES_PER_BATCH {
            let mut mutations = Vec::with_capacity(entries.len() + 1);
            // Parent row in Trees MUST precede interleaved child rows in TreeEntries.
            mutations.push(
                Mutation::new_insert_or_update_builder("Trees")
                    .set("ProjectId")
                    .to(project_id)
                    .set("TreeId")
                    .to(tree_id)
                    .set("Ready")
                    .to(true)
                    .build(),
            );
            for entry in entries {
                let entry_proto = entry.encode_to_vec();
                mutations.push(
                    Mutation::new_insert_or_update_builder("TreeEntries")
                        .set("ProjectId")
                        .to(project_id)
                        .set("TreeId")
                        .to(tree_id)
                        .set("Name")
                        .to(&entry.name)
                        .set("EntryProto")
                        .to(&entry_proto)
                        .build(),
                );
            }
            self.db_client
                .write_only_transaction()
                .build()
                .write(mutations)
                .await
                .map_err(map_spanner_err)?;
            return Ok(());
        }

        // Large directory (>4,000 entries): stage parent with Ready=false, stream chunks under
        // Spanner's 80,000-mutation transaction cap, and flip Ready=true on the final batch.
        // Tree ids are content hashes, so a ready tree is complete; skip it rather than hiding
        // it from readers while the chunks are rewritten.
        let ready_stmt = Statement::builder(
            "SELECT 1 FROM Trees WHERE ProjectId = @project_id AND TreeId = @tree_id AND Ready = TRUE",
        )
        .add_param("project_id", project_id)
        .add_param("tree_id", tree_id)
        .build();
        let ready_tx = self.db_client.single_use().build();
        let mut ready_rs = ready_tx
            .execute_query(ready_stmt)
            .await
            .map_err(map_spanner_err)?;
        if ready_rs
            .next()
            .await
            .transpose()
            .map_err(map_spanner_err)?
            .is_some()
        {
            return Ok(());
        }

        // Insert-only staging never turns a row that a concurrent upload just finished back
        // to Ready=false; an existing row (ready or left by an interrupted upload) is kept.
        let stage_parent = Mutation::new_insert_builder("Trees")
            .set("ProjectId")
            .to(project_id)
            .set("TreeId")
            .to(tree_id)
            .set("Ready")
            .to(false)
            .build();
        match self
            .db_client
            .write_only_transaction()
            .build()
            .write(vec![stage_parent])
            .await
        {
            Ok(_) => {}
            Err(err) if is_already_exists(&err) => {}
            Err(err) => return Err(map_spanner_err(err)),
        }

        let chunks: Vec<&[TreeEntry]> = entries.chunks(MAX_ENTRIES_PER_BATCH).collect();
        for (idx, chunk) in chunks.iter().enumerate() {
            let is_last = idx + 1 == chunks.len();
            let mut mutations = Vec::with_capacity(chunk.len() + usize::from(is_last));
            for entry in *chunk {
                let entry_proto = entry.encode_to_vec();
                mutations.push(
                    Mutation::new_insert_or_update_builder("TreeEntries")
                        .set("ProjectId")
                        .to(project_id)
                        .set("TreeId")
                        .to(tree_id)
                        .set("Name")
                        .to(&entry.name)
                        .set("EntryProto")
                        .to(&entry_proto)
                        .build(),
                );
            }
            if is_last {
                mutations.push(
                    Mutation::new_insert_or_update_builder("Trees")
                        .set("ProjectId")
                        .to(project_id)
                        .set("TreeId")
                        .to(tree_id)
                        .set("Ready")
                        .to(true)
                        .build(),
                );
            }
            self.db_client
                .write_only_transaction()
                .build()
                .write(mutations)
                .await
                .map_err(map_spanner_err)?;
        }
        Ok(())
    }

    async fn find_existing_trees(
        &self,
        project_id: &str,
        tree_ids: &[Vec<u8>],
    ) -> StoreResult<HashSet<Vec<u8>>> {
        if tree_ids.is_empty() {
            return Ok(HashSet::new());
        }
        let mut found = HashSet::with_capacity(tree_ids.len());
        for chunk in tree_ids.chunks(UNNEST_BATCH_SIZE) {
            let ids_vec = chunk.to_vec();
            let stmt = Statement::builder(
                "SELECT TreeId FROM Trees
                 WHERE ProjectId = @project_id AND TreeId IN UNNEST(@ids) AND Ready = TRUE",
            )
            .add_param("project_id", project_id)
            .add_param("ids", ids_vec)
            .build();
            let tx = self.db_client.single_use().build();
            let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
            while let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? {
                let tid: Vec<u8> = row.get("TreeId").map_err(map_spanner_err)?;
                found.insert(tid);
            }
        }
        Ok(found)
    }

    async fn get_symlink(
        &self,
        project_id: &str,
        symlink_id: &[u8],
    ) -> StoreResult<Option<String>> {
        let stmt = Statement::builder(
            "SELECT Target FROM Symlinks WHERE ProjectId = @project_id AND SymlinkId = @symlink_id",
        )
        .add_param("project_id", project_id)
        .add_param("symlink_id", symlink_id)
        .build();
        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? else {
            return Ok(None);
        };
        let target: String = row.get("Target").map_err(map_spanner_err)?;
        Ok(Some(target))
    }

    async fn put_symlink(
        &self,
        project_id: &str,
        symlink_id: &[u8],
        target: &str,
    ) -> StoreResult<()> {
        let mutation = Mutation::new_insert_or_update_builder("Symlinks")
            .set("ProjectId")
            .to(project_id)
            .set("SymlinkId")
            .to(symlink_id)
            .set("Target")
            .to(target)
            .build();
        self.db_client
            .write_only_transaction()
            .build()
            .write(vec![mutation])
            .await
            .map_err(map_spanner_err)?;
        Ok(())
    }

    async fn find_existing_symlinks(
        &self,
        project_id: &str,
        symlink_ids: &[Vec<u8>],
    ) -> StoreResult<HashSet<Vec<u8>>> {
        if symlink_ids.is_empty() {
            return Ok(HashSet::new());
        }
        let mut found = HashSet::with_capacity(symlink_ids.len());
        for chunk in symlink_ids.chunks(UNNEST_BATCH_SIZE) {
            let ids_vec = chunk.to_vec();
            let stmt = Statement::builder(
                "SELECT SymlinkId FROM Symlinks
                 WHERE ProjectId = @project_id AND SymlinkId IN UNNEST(@ids)",
            )
            .add_param("project_id", project_id)
            .add_param("ids", ids_vec)
            .build();
            let tx = self.db_client.single_use().build();
            let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
            while let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? {
                let sid: Vec<u8> = row.get("SymlinkId").map_err(map_spanner_err)?;
                found.insert(sid);
            }
        }
        Ok(found)
    }

    async fn get_blob_record(
        &self,
        project_id: &str,
        blob_id: &[u8],
    ) -> StoreResult<Option<BlobRecord>> {
        let stmt = Statement::builder(
            "SELECT b.BlobId, b.ContentSize, b.InlineContent, b.PrimaryLocation, b.ReplicatedLocations
             FROM Blobs b
             INNER JOIN ProjectBlobs pb ON pb.BlobId = b.BlobId
             WHERE pb.ProjectId = @project_id AND b.BlobId = @blob_id",
        )
        .add_param("project_id", project_id)
        .add_param("blob_id", blob_id)
        .build();

        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? else {
            return Ok(None);
        };
        decode_blob_row(&row)
    }

    async fn get_blob_sizes_batch(
        &self,
        project_id: &str,
        blob_ids: &[Vec<u8>],
    ) -> StoreResult<HashMap<Vec<u8>, u64>> {
        if blob_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let mut sizes = HashMap::with_capacity(blob_ids.len());
        for chunk in blob_ids.chunks(UNNEST_BATCH_SIZE) {
            let ids_vec = chunk.to_vec();
            let stmt = Statement::builder(
                "SELECT b.BlobId, b.ContentSize
                 FROM Blobs b
                 INNER JOIN ProjectBlobs pb ON pb.BlobId = b.BlobId
                 WHERE pb.ProjectId = @project_id AND pb.BlobId IN UNNEST(@ids)",
            )
            .add_param("project_id", project_id)
            .add_param("ids", ids_vec)
            .build();
            let tx = self.db_client.single_use().build();
            let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
            while let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? {
                let bid: Vec<u8> = row.get("BlobId").map_err(map_spanner_err)?;
                let sz: i64 = row.get("ContentSize").map_err(map_spanner_err)?;
                sizes.insert(bid, sz as u64);
            }
        }
        Ok(sizes)
    }

    async fn get_global_blob_record(&self, blob_id: &[u8]) -> StoreResult<Option<BlobRecord>> {
        let stmt = Statement::builder(
            "SELECT BlobId, ContentSize, InlineContent, PrimaryLocation, ReplicatedLocations
             FROM Blobs WHERE BlobId = @blob_id",
        )
        .add_param("blob_id", blob_id)
        .build();

        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? else {
            return Ok(None);
        };
        decode_blob_row(&row)
    }

    async fn put_blob_record(&self, project_id: &str, record: &BlobRecord) -> StoreResult<()> {
        let inline_vec: Option<Vec<u8>> = record.inline_content.as_ref().map(|b| b.to_vec());
        let blob_mutation = Mutation::new_insert_or_update_builder("Blobs")
            .set("BlobId")
            .to(&record.blob_id)
            .set("ContentSize")
            .to(record.content_size as i64)
            .set("InlineContent")
            .to(&inline_vec)
            .set("PrimaryLocation")
            .to(&record.primary_location)
            .set("ReplicatedLocations")
            .to(&record.replicated_locations)
            .build();

        let project_blob_mutation = Mutation::new_insert_or_update_builder("ProjectBlobs")
            .set("ProjectId")
            .to(project_id)
            .set("BlobId")
            .to(&record.blob_id)
            .build();

        self.db_client
            .write_only_transaction()
            .build()
            .write(vec![blob_mutation, project_blob_mutation])
            .await
            .map_err(map_spanner_err)?;
        Ok(())
    }

    async fn add_blob_replica_location(&self, blob_id: &[u8], location: &str) -> StoreResult<()> {
        let blob_id = blob_id.to_vec();
        let location = location.to_string();
        let runner = self.db_client.read_write_transaction().build();

        runner
            .run(
                async move |tx| -> google_cloud_spanner::Result<StoreResult<()>> {
                    let stmt = Statement::builder(
                        "SELECT ReplicatedLocations FROM Blobs WHERE BlobId = @blob_id",
                    )
                    .add_param("blob_id", &blob_id)
                    .build();
                    let mut rs = tx.execute_query(stmt).await?;
                    if let Some(row) = rs.next().await.transpose()? {
                        let mut locations: Vec<String> = row.get("ReplicatedLocations")?;
                        if !locations.iter().any(|l| l == &location) {
                            locations.push(location.clone());
                            let update = Mutation::new_update_builder("Blobs")
                                .set("BlobId")
                                .to(&blob_id)
                                .set("ReplicatedLocations")
                                .to(&locations)
                                .build();
                            tx.buffer([update])?;
                        }
                    }
                    Ok(Ok(()))
                },
            )
            .await
            .map_err(map_spanner_err)?
            .into_inner()
    }

    async fn get_operation(&self, repo_id: &str, op_id: &[u8]) -> StoreResult<Option<Operation>> {
        let stmt = Statement::builder(
            "SELECT ProtoBytes FROM Operations WHERE RepoId = @repo_id AND OperationId = @op_id",
        )
        .add_param("repo_id", repo_id)
        .add_param("op_id", op_id)
        .build();
        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? else {
            return Ok(None);
        };
        let proto_bytes: Vec<u8> = row.get("ProtoBytes").map_err(map_spanner_err)?;
        let op = Operation::decode(proto_bytes.as_slice()).map_err(|e| {
            StoreError::Corruption(format!("failed to decode Spanner Operation: {e}"))
        })?;
        Ok(Some(op))
    }

    async fn put_operation(&self, repo_id: &str, op_id: &[u8], op: &Operation) -> StoreResult<()> {
        let proto_bytes = op.encode_to_vec();
        let mutation = Mutation::new_insert_or_update_builder("Operations")
            .set("RepoId")
            .to(repo_id)
            .set("OperationId")
            .to(op_id)
            .set("ProtoBytes")
            .to(&proto_bytes)
            .build();
        self.db_client
            .write_only_transaction()
            .build()
            .write(vec![mutation])
            .await
            .map_err(map_spanner_err)?;
        Ok(())
    }

    async fn resolve_operation_id_prefix(
        &self,
        repo_id: &str,
        min_id: &[u8],
        max_id: &[u8],
    ) -> StoreResult<Vec<OpId>> {
        let stmt = Statement::builder(
            "SELECT OperationId FROM Operations
             WHERE RepoId = @repo_id AND OperationId >= @min_id AND OperationId <= @max_id
             ORDER BY OperationId ASC
             LIMIT 2",
        )
        .add_param("repo_id", repo_id)
        .add_param("min_id", min_id)
        .add_param("max_id", max_id)
        .build();

        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let mut matches = Vec::new();
        while let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? {
            let op_id: Vec<u8> = row.get("OperationId").map_err(map_spanner_err)?;
            matches.push(op_id);
        }
        Ok(matches)
    }

    async fn get_view(&self, repo_id: &str, view_id: &[u8]) -> StoreResult<Option<View>> {
        let stmt = Statement::builder(
            "SELECT ProtoBytes FROM Views WHERE RepoId = @repo_id AND ViewId = @view_id",
        )
        .add_param("repo_id", repo_id)
        .add_param("view_id", view_id)
        .build();
        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? else {
            return Ok(None);
        };
        let proto_bytes: Vec<u8> = row.get("ProtoBytes").map_err(map_spanner_err)?;
        let view = View::decode(proto_bytes.as_slice())
            .map_err(|e| StoreError::Corruption(format!("failed to decode Spanner View: {e}")))?;
        Ok(Some(view))
    }

    async fn put_view(&self, repo_id: &str, view_id: &[u8], view: &View) -> StoreResult<()> {
        let proto_bytes = view.encode_to_vec();
        let mutation = Mutation::new_insert_or_update_builder("Views")
            .set("RepoId")
            .to(repo_id)
            .set("ViewId")
            .to(view_id)
            .set("ProtoBytes")
            .to(&proto_bytes)
            .build();
        self.db_client
            .write_only_transaction()
            .build()
            .write(vec![mutation])
            .await
            .map_err(map_spanner_err)?;
        Ok(())
    }

    async fn get_op_heads(&self, repo_id: &str) -> StoreResult<Option<Vec<OpId>>> {
        let stmt = Statement::builder("SELECT OpHeads FROM Repositories WHERE RepoId = @repo_id")
            .add_param("repo_id", repo_id)
            .build();
        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? else {
            return Ok(None);
        };
        let heads: Vec<Vec<u8>> = row.get("OpHeads").map_err(map_spanner_err)?;
        Ok(Some(heads))
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
        let runner = self.db_client.read_write_transaction().build();

        runner
            .run(
                async move |tx| -> google_cloud_spanner::Result<StoreResult<Vec<OpId>>> {
                    let stmt = Statement::builder(
                        "SELECT OpHeads FROM Repositories WHERE RepoId = @repo_id",
                    )
                    .add_param("repo_id", &repo_id)
                    .build();
                    let mut rs = tx.execute_query(stmt).await?;
                    let Some(row) = rs.next().await.transpose()? else {
                        return Ok(Err(StoreError::RepoNotFound {
                            repo_id: repo_id.clone(),
                            action: "updating op heads".to_string(),
                        }));
                    };

                    let mut current_heads: Vec<Vec<u8>> = row.get("OpHeads")?;
                    apply_op_heads_mutation(&mut current_heads, &old_ids, &new_id);

                    let mutation = Mutation::new_update_builder("Repositories")
                        .set("RepoId")
                        .to(&repo_id)
                        .set("OpHeads")
                        .to(&current_heads)
                        .build();
                    tx.buffer([mutation])?;
                    Ok(Ok(current_heads))
                },
            )
            .await
            .map_err(map_spanner_err)?
            .into_inner()
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
        let runner = self.db_client.read_write_transaction().build();

        runner
            .run(
                async move |tx| -> google_cloud_spanner::Result<StoreResult<Vec<OpId>>> {
                    let stmt = Statement::builder(
                        "SELECT OpHeads FROM Repositories WHERE RepoId = @repo_id",
                    )
                    .add_param("repo_id", &repo_id)
                    .build();
                    let mut rs = tx.execute_query(stmt).await?;
                    let Some(row) = rs.next().await.transpose()? else {
                        return Ok(Err(StoreError::RepoNotFound {
                            repo_id: repo_id.clone(),
                            action: "publishing operation".to_string(),
                        }));
                    };

                    let mut current_heads: Vec<Vec<u8>> = row.get("OpHeads")?;
                    apply_op_heads_mutation(&mut current_heads, &old_ids, &op_id);

                    let view_mut = Mutation::new_insert_or_update_builder("Views")
                        .set("RepoId")
                        .to(&repo_id)
                        .set("ViewId")
                        .to(&view_id)
                        .set("ProtoBytes")
                        .to(&view_bytes)
                        .build();
                    let op_mut = Mutation::new_insert_or_update_builder("Operations")
                        .set("RepoId")
                        .to(&repo_id)
                        .set("OperationId")
                        .to(&op_id)
                        .set("ProtoBytes")
                        .to(&op_bytes)
                        .build();
                    let heads_mut = Mutation::new_update_builder("Repositories")
                        .set("RepoId")
                        .to(&repo_id)
                        .set("OpHeads")
                        .to(&current_heads)
                        .build();
                    tx.buffer([view_mut, op_mut, heads_mut])?;
                    Ok(Ok(current_heads))
                },
            )
            .await
            .map_err(map_spanner_err)?
            .into_inner()
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
        let runner = self.db_client.read_write_transaction().build();

        runner
            .run(
                async move |tx| -> google_cloud_spanner::Result<StoreResult<()>> {
                    let stmt = Statement::builder(
                        "SELECT OpHeads FROM Repositories WHERE RepoId = @repo_id",
                    )
                    .add_param("repo_id", &repo_id)
                    .build();
                    let mut rs = tx.execute_query(stmt).await?;
                    let Some(row) = rs.next().await.transpose()? else {
                        return Ok(Err(StoreError::RepoNotFound {
                            repo_id: repo_id.clone(),
                            action: "comparing and swapping op heads".to_string(),
                        }));
                    };

                    let current_heads: Vec<Vec<u8>> = row.get("OpHeads")?;
                    if !op_heads_sets_equal(&current_heads, &expected_old_ids) {
                        return Ok(Err(StoreError::CasConflict(format!(
                            "op heads changed concurrently in repo '{repo_id}'"
                        ))));
                    }

                    let updated = vec![new_id.clone()];
                    let mutation = Mutation::new_update_builder("Repositories")
                        .set("RepoId")
                        .to(&repo_id)
                        .set("OpHeads")
                        .to(&updated)
                        .build();
                    tx.buffer([mutation])?;
                    Ok(Ok(()))
                },
            )
            .await
            .map_err(map_spanner_err)?
            .into_inner()
    }

    async fn get_workspace(
        &self,
        repo_id: &str,
        user: &str,
        workspace_name: &str,
    ) -> StoreResult<Option<WorkspaceState>> {
        let stmt = Statement::builder(
            "SELECT CommitId, OperationId, TreeId, TreeIds, SparsePatterns
             FROM Workspaces
             WHERE RepoId = @repo_id AND UserEmail = @user AND WorkspaceName = @ws",
        )
        .add_param("repo_id", repo_id)
        .add_param("user", user)
        .add_param("ws", workspace_name)
        .build();

        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? else {
            return Ok(None);
        };
        let commit_id: Vec<u8> = row.get("CommitId").map_err(map_spanner_err)?;
        let operation_id: Vec<u8> = row.get("OperationId").map_err(map_spanner_err)?;
        let tree_id: Vec<u8> = row.get("TreeId").map_err(map_spanner_err)?;
        let mut tree_ids: Vec<Vec<u8>> = row.get("TreeIds").map_err(map_spanner_err)?;
        if tree_ids.is_empty() && !tree_id.is_empty() {
            tree_ids.push(tree_id.clone());
        }
        let sparse_patterns: Vec<String> = row.get("SparsePatterns").map_err(map_spanner_err)?;

        Ok(Some(WorkspaceState {
            repo_id: repo_id.to_string(),
            user: user.to_string(),
            workspace_name: workspace_name.to_string(),
            commit_id,
            operation_id,
            tree_id,
            sparse_patterns,
            tree_ids,
        }))
    }

    async fn put_workspace(&self, workspace: &WorkspaceState) -> StoreResult<()> {
        let tree_ids = if workspace.tree_ids.is_empty() && !workspace.tree_id.is_empty() {
            vec![workspace.tree_id.clone()]
        } else {
            workspace.tree_ids.clone()
        };
        let mutation = Mutation::new_insert_or_update_builder("Workspaces")
            .set("RepoId")
            .to(&workspace.repo_id)
            .set("UserEmail")
            .to(&workspace.user)
            .set("WorkspaceName")
            .to(&workspace.workspace_name)
            .set("CommitId")
            .to(&workspace.commit_id)
            .set("OperationId")
            .to(&workspace.operation_id)
            .set("TreeId")
            .to(&workspace.tree_id)
            .set("TreeIds")
            .to(&tree_ids)
            .set("SparsePatterns")
            .to(&workspace.sparse_patterns)
            .build();

        self.db_client
            .write_only_transaction()
            .build()
            .write(vec![mutation])
            .await
            .map_err(map_spanner_err)?;
        Ok(())
    }

    async fn list_workspaces(&self, repo_id: &str) -> StoreResult<Vec<WorkspaceState>> {
        let stmt = Statement::builder(
            "SELECT UserEmail, WorkspaceName, CommitId, OperationId, TreeId, TreeIds, SparsePatterns
             FROM Workspaces
             WHERE RepoId = @repo_id
             ORDER BY UserEmail ASC, WorkspaceName ASC",
        )
        .add_param("repo_id", repo_id)
        .build();

        let tx = self.db_client.single_use().build();
        let mut rs = tx.execute_query(stmt).await.map_err(map_spanner_err)?;
        let mut out = Vec::new();
        while let Some(row) = rs.next().await.transpose().map_err(map_spanner_err)? {
            let user: String = row.get("UserEmail").map_err(map_spanner_err)?;
            let workspace_name: String = row.get("WorkspaceName").map_err(map_spanner_err)?;
            let commit_id: Vec<u8> = row.get("CommitId").map_err(map_spanner_err)?;
            let operation_id: Vec<u8> = row.get("OperationId").map_err(map_spanner_err)?;
            let tree_id: Vec<u8> = row.get("TreeId").map_err(map_spanner_err)?;
            let mut tree_ids: Vec<Vec<u8>> = row.get("TreeIds").map_err(map_spanner_err)?;
            if tree_ids.is_empty() && !tree_id.is_empty() {
                tree_ids.push(tree_id.clone());
            }
            let sparse_patterns: Vec<String> =
                row.get("SparsePatterns").map_err(map_spanner_err)?;
            out.push(WorkspaceState {
                repo_id: repo_id.to_string(),
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
        let runner = self.db_client.read_write_transaction().build();

        runner
            .run(
                async move |tx| -> google_cloud_spanner::Result<StoreResult<bool>> {
                    let stmt = Statement::builder(
                        "SELECT 1 FROM Workspaces
                         WHERE RepoId = @repo_id AND UserEmail = @user AND WorkspaceName = @ws",
                    )
                    .add_param("repo_id", &repo_id)
                    .add_param("user", &user)
                    .add_param("ws", &workspace_name)
                    .build();
                    let mut rs = tx.execute_query(stmt).await?;
                    if rs.next().await.transpose()?.is_none() {
                        return Ok(Ok(false));
                    }
                    let key = Key::new(vec![
                        repo_id.clone().into(),
                        user.clone().into(),
                        workspace_name.clone().into(),
                    ]);
                    let key_set = KeySet::builder().add_key(key).build();
                    let del_mutation = Mutation::delete("Workspaces", key_set);
                    tx.buffer([del_mutation])?;
                    Ok(Ok(true))
                },
            )
            .await
            .map_err(map_spanner_err)?
            .into_inner()
    }
}

fn decode_blob_row(row: &google_cloud_spanner::result::Row) -> StoreResult<Option<BlobRecord>> {
    let blob_id: Vec<u8> = row.get("BlobId").map_err(map_spanner_err)?;
    let content_size: i64 = row.get("ContentSize").map_err(map_spanner_err)?;
    let inline_content: Option<Vec<u8>> = row.get("InlineContent").map_err(map_spanner_err)?;
    let primary_location: Option<String> = row.get("PrimaryLocation").map_err(map_spanner_err)?;
    let replicated_locations: Vec<String> =
        row.get("ReplicatedLocations").map_err(map_spanner_err)?;

    Ok(Some(BlobRecord {
        blob_id,
        content_size: content_size as u64,
        inline_content: inline_content.map(Bytes::from),
        primary_location,
        replicated_locations,
    }))
}

#[allow(dead_code)]
fn _assert_arc_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<Arc<SpannerMetadataStore>>();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn service_error(code: RpcCode) -> google_cloud_spanner::Error {
        google_cloud_gax::error::Error::service(
            RpcStatus::default().set_code(code).set_message("injected"),
        )
    }

    #[test]
    fn spanner_errors_map_to_retryable_and_specific_codes() {
        let cases = [
            (RpcCode::Aborted, tonic::Code::Unavailable),
            (RpcCode::Unavailable, tonic::Code::Unavailable),
            (RpcCode::DeadlineExceeded, tonic::Code::Unavailable),
            (RpcCode::ResourceExhausted, tonic::Code::ResourceExhausted),
            (RpcCode::PermissionDenied, tonic::Code::PermissionDenied),
            (RpcCode::Unauthenticated, tonic::Code::Unauthenticated),
            (RpcCode::InvalidArgument, tonic::Code::InvalidArgument),
            (RpcCode::Internal, tonic::Code::Internal),
        ];
        for (spanner_code, grpc_code) in cases {
            let status = tonic::Status::from(map_spanner_err(service_error(spanner_code)));
            assert_eq!(
                status.code(),
                grpc_code,
                "Spanner {spanner_code:?}: {status:?}"
            );
        }
    }

    #[test]
    fn finished_admin_op_surfaces_operation_errors() {
        assert!(finished_admin_op(false, None).is_none());
        assert!(matches!(finished_admin_op(true, None), Some(Ok(()))));

        let failed = RpcStatus::default()
            .set_code(RpcCode::FailedPrecondition)
            .set_message("Duplicate name in schema: Commits.");
        let outcome = finished_admin_op(true, Some(failed)).expect("finished operation");
        let err = outcome.expect_err("DDL failure must not report success");
        assert!(
            err.to_string().contains("Duplicate name in schema"),
            "{err}"
        );

        assert!(check_emulator_lro_json("HTTP/1.0 200 OK\r\n\r\n{\"done\":true}").is_ok());
        let emu_failed =
            "HTTP/1.0 200 OK\r\n\r\n{\"done\":true,\"error\":{\"code\":9,\"message\":\"bad ddl\"}}";
        assert!(check_emulator_lro_json(emu_failed).is_err());
    }
}
