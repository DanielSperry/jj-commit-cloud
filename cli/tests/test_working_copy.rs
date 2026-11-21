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

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use cc_common::workspace::{
    CheckWorkingCopyChangesRequest, GetWorkspaceRequest, UpdateWorkspaceRequest, WorkspaceState,
};
use cc_lib::util::{CommitCloudConfig, connect_workspace_client, run_on_tokio};
use predicates::prelude::*;

fn jj_cmd(cwd: &Path, xdg: &Path) -> Command {
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("jj");
    cmd.current_dir(cwd)
        .env("XDG_CONFIG_HOME", xdg)
        .env("JJ_USER", "Workspace Tester")
        .env("JJ_EMAIL", "ws.tester@example.com");
    cmd
}

#[tokio::test]
async fn test_commit_cloud_working_copy_syncs_state_sparse_patterns_and_cli_commands() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xdg = dir.path().join("xdg");
    fs::create_dir_all(&xdg).unwrap();
    let server = testutils::spawn_sqlite_server(&dir.path().join("wc.db")).await;

    let ws_main = dir.path().join("ws_main");
    jj_cmd(dir.path(), &xdg)
        .args([
            "cc",
            "init",
            "--server",
            server.url(),
            "--project-id",
            "proj-wc",
            "--create",
            ws_main.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Default working copy type is `commit_cloud`.
    let wc_type = fs::read_to_string(ws_main.join(".jj/working_copy/type")).unwrap();
    assert_eq!(wc_type.trim(), "commit_cloud");

    let cfg = CommitCloudConfig::load_from_store(&ws_main).unwrap();
    let repo_id = cfg.repo_id.clone();
    let server_url = cfg.server_url.clone();

    // Edit files, configure sparse patterns, and describe the commit.
    fs::create_dir_all(ws_main.join("src")).unwrap();
    fs::write(ws_main.join("src/main.rs"), "fn main() {}\n").unwrap();
    jj_cmd(&ws_main, &xdg)
        .args(["sparse", "set", "--clear", "--add", "src"])
        .assert()
        .success();
    jj_cmd(&ws_main, &xdg)
        .args(["describe", "-m", "sparse wc commit"])
        .assert()
        .success();

    let commit_hex = String::from_utf8(
        jj_cmd(&ws_main, &xdg)
            .args(["log", "--no-graph", "-r", "@", "-T", "commit_id"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone(),
    )
    .unwrap()
    .trim()
    .to_string();

    // Verify `WorkspaceService` persisted commit_id, operation_id, tree_id, and sparse_patterns.
    let url_clone = server_url.clone();
    let repo_clone = repo_id.clone();
    let (ws_state, change_check) = run_on_tokio(async move {
        let mut client = connect_workspace_client(url_clone).await?;
        let ws = client
            .get_workspace(GetWorkspaceRequest {
                repo_id: repo_clone.clone(),
                user: "ws.tester@example.com".to_string(),
                workspace_name: "default".to_string(),
            })
            .await?
            .into_inner()
            .workspace
            .expect("workspace state should exist on server");
        let ch = client
            .check_working_copy_changes(CheckWorkingCopyChangesRequest {
                repo_id: repo_clone,
                user: "ws.tester@example.com".to_string(),
                workspace_name: "default".to_string(),
            })
            .await?
            .into_inner();
        Ok((ws, ch))
    })
    .await
    .unwrap();

    assert_eq!(hex::encode(&ws_state.commit_id), commit_hex);
    assert!(!ws_state.operation_id.is_empty());
    assert!(!ws_state.tree_id.is_empty());
    assert!(ws_state.sparse_patterns.iter().any(|p| p == "src"));
    assert!(!change_check.has_changes);

    // Clone a second workspace (`ws_clone`) and exercise `jj cc workspace list/status/delete`.
    let ws_clone = dir.path().join("ws_clone");
    jj_cmd(dir.path(), &xdg)
        .args([
            "cc",
            "clone",
            "--server",
            server.url(),
            "--workspace-name",
            "agent-sandbox",
            &repo_id,
            ws_clone.to_str().unwrap(),
        ])
        .assert()
        .success();

    jj_cmd(&ws_main, &xdg)
        .args(["cc", "workspace", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("default (ws.tester@example.com)"))
        .stdout(predicate::str::contains(
            "agent-sandbox (ws.tester@example.com)",
        ));

    jj_cmd(&ws_main, &xdg)
        .args(["cc", "workspace", "status", "--workspace-name", "default"])
        .assert()
        .success()
        .stdout(predicate::str::contains(&commit_hex))
        .stdout(predicate::str::contains("Modified:  false"));

    jj_cmd(&ws_main, &xdg)
        .args(["cc", "workspace", "delete", "agent-sandbox"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Deleted cloud workspace 'agent-sandbox'",
        ));
}

#[tokio::test]
async fn test_cc_init_supports_explicit_local_working_copy_type() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xdg = dir.path().join("xdg");
    fs::create_dir_all(&xdg).unwrap();
    let server = testutils::spawn_server().await;

    let ws_local = dir.path().join("ws_local");
    jj_cmd(dir.path(), &xdg)
        .args([
            "cc",
            "init",
            "--server",
            server.url(),
            "--working-copy-type",
            "local",
            "--create",
            ws_local.to_str().unwrap(),
        ])
        .assert()
        .success();

    let wc_type = fs::read_to_string(ws_local.join(".jj/working_copy/type")).unwrap();
    assert_eq!(wc_type.trim(), "local");

    fs::write(ws_local.join("note.txt"), "local mode\n").unwrap();
    jj_cmd(&ws_local, &xdg)
        .args(["describe", "-m", "local wc mode"])
        .assert()
        .success();
}

#[tokio::test]
async fn test_commit_cloud_working_copy_preserves_conflicted_merge_trees_in_workspace_service() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xdg = dir.path().join("xdg");
    fs::create_dir_all(&xdg).unwrap();
    let server = testutils::spawn_sqlite_server(&dir.path().join("wc_conflict.db")).await;

    let ws_dir = dir.path().join("ws_conflict");
    jj_cmd(dir.path(), &xdg)
        .args([
            "cc",
            "init",
            "--server",
            server.url(),
            "--project-id",
            "proj-wc-conflict",
            "--create",
            ws_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Base commit
    fs::write(ws_dir.join("conflict.txt"), "base\n").unwrap();
    jj_cmd(&ws_dir, &xdg)
        .args(["bookmark", "create", "base", "-r", "@"])
        .assert()
        .success();
    jj_cmd(&ws_dir, &xdg)
        .args(["describe", "-m", "base"])
        .assert()
        .success();

    // Left branch
    jj_cmd(&ws_dir, &xdg)
        .args(["new", "base"])
        .assert()
        .success();
    fs::write(ws_dir.join("conflict.txt"), "left side\n").unwrap();
    jj_cmd(&ws_dir, &xdg)
        .args(["bookmark", "create", "left", "-r", "@"])
        .assert()
        .success();
    jj_cmd(&ws_dir, &xdg)
        .args(["describe", "-m", "left"])
        .assert()
        .success();

    // Right branch
    jj_cmd(&ws_dir, &xdg)
        .args(["new", "base"])
        .assert()
        .success();
    fs::write(ws_dir.join("conflict.txt"), "right side\n").unwrap();
    jj_cmd(&ws_dir, &xdg)
        .args(["bookmark", "create", "right", "-r", "@"])
        .assert()
        .success();
    jj_cmd(&ws_dir, &xdg)
        .args(["describe", "-m", "right"])
        .assert()
        .success();

    // Conflicted merge checkout (`Merge<TreeId>` with 3 tree terms: `[left, base, right]`)!
    jj_cmd(&ws_dir, &xdg)
        .args(["new", "left", "right", "-m", "conflicted merge"])
        .assert()
        .success();

    let cfg = CommitCloudConfig::load_from_store(&ws_dir).unwrap();
    let (ws_state, change_check) = run_on_tokio(async move {
        let mut client = connect_workspace_client(cfg.server_url).await?;
        let ws = client
            .get_workspace(GetWorkspaceRequest {
                repo_id: cfg.repo_id.clone(),
                user: "ws.tester@example.com".to_string(),
                workspace_name: "default".to_string(),
            })
            .await?
            .into_inner()
            .workspace
            .expect("workspace state should exist");
        let ch = client
            .check_working_copy_changes(CheckWorkingCopyChangesRequest {
                repo_id: cfg.repo_id,
                user: "ws.tester@example.com".to_string(),
                workspace_name: "default".to_string(),
            })
            .await?
            .into_inner();
        Ok((ws, ch))
    })
    .await
    .unwrap();

    assert_eq!(
        ws_state.tree_ids.len(),
        3,
        "conflicted checkout must persist all 3 Merge<TreeId> sides instead of EMPTY_TREE_ID"
    );
    assert!(
        !change_check.has_changes,
        "fresh conflicted merge checkout must match its checked-out commit's Merge<TreeId>"
    );
    assert_eq!(change_check.current_tree_ids, change_check.commit_tree_ids);
}

#[tokio::test]
async fn test_snapshot_only_command_keeps_workspace_commit_current() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xdg = dir.path().join("xdg");
    fs::create_dir_all(&xdg).unwrap();
    let server = testutils::spawn_sqlite_server(&dir.path().join("wc_snapshot.db")).await;

    let ws_dir = dir.path().join("ws_snapshot");
    jj_cmd(dir.path(), &xdg)
        .args([
            "cc",
            "init",
            "--server",
            server.url(),
            "--project-id",
            "proj-wc-snapshot",
            "--create",
            ws_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    jj_cmd(&ws_dir, &xdg)
        .args(["describe", "-m", "checked out by describe"])
        .assert()
        .success();

    // Snapshot-only commands (`jj st`, `jj diff`, `jj log`) rewrite `@` without a checkout.
    for (idx, snapshot_cmd) in [
        (1, &["st"][..]),
        (2, &["diff"][..]),
        (3, &["log", "-n", "1"][..]),
    ] {
        fs::write(ws_dir.join("edited.txt"), format!("edited step {idx}\n")).unwrap();
        jj_cmd(&ws_dir, &xdg).args(snapshot_cmd).assert().success();

        let commit_hex = String::from_utf8(
            jj_cmd(&ws_dir, &xdg)
                .args([
                    "log",
                    "--ignore-working-copy",
                    "--no-graph",
                    "-r",
                    "@",
                    "-T",
                    "commit_id",
                ])
                .assert()
                .success()
                .get_output()
                .stdout
                .clone(),
        )
        .unwrap()
        .trim()
        .to_string();

        let cfg = CommitCloudConfig::load_from_store(&ws_dir).unwrap();
        let (ws_state, change_check) = run_on_tokio(async move {
            let mut client = connect_workspace_client(cfg.server_url).await?;
            let ws = client
                .get_workspace(GetWorkspaceRequest {
                    repo_id: cfg.repo_id.clone(),
                    user: "ws.tester@example.com".to_string(),
                    workspace_name: "default".to_string(),
                })
                .await?
                .into_inner()
                .workspace
                .expect("workspace state should exist");
            let ch = client
                .check_working_copy_changes(CheckWorkingCopyChangesRequest {
                    repo_id: cfg.repo_id,
                    user: "ws.tester@example.com".to_string(),
                    workspace_name: "default".to_string(),
                })
                .await?
                .into_inner();
            Ok((ws, ch))
        })
        .await
        .unwrap();

        assert_eq!(
            hex::encode(&ws_state.commit_id),
            commit_hex,
            "record must follow snapshot from {snapshot_cmd:?}"
        );
        assert!(
            !change_check.has_changes,
            "snapshotted tree must match the recorded commit"
        );

        jj_cmd(&ws_dir, &xdg)
            .args(["cc", "workspace", "status", "--workspace-name", "default"])
            .assert()
            .success()
            .stdout(predicate::str::contains(format!("Commit:    {commit_hex}")))
            .stdout(predicate::str::contains("Modified:  false"));
    }
}

#[tokio::test]
async fn test_workspace_status_surfaces_check_working_copy_changes_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xdg = dir.path().join("xdg");
    fs::create_dir_all(&xdg).unwrap();
    let server = testutils::spawn_server().await;

    let ws_dir = dir.path().join("ws_err");
    jj_cmd(dir.path(), &xdg)
        .args([
            "cc",
            "init",
            "--server",
            server.url(),
            "--create",
            ws_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let cfg = CommitCloudConfig::load_from_store(&ws_dir).unwrap();
    run_on_tokio(async move {
        let mut client = connect_workspace_client(cfg.server_url).await?;
        client
            .update_workspace(UpdateWorkspaceRequest {
                workspace: Some(WorkspaceState {
                    repo_id: cfg.repo_id,
                    user: "ws.tester@example.com".to_string(),
                    workspace_name: "broken".to_string(),
                    commit_id: vec![0xde; 20],
                    operation_id: Vec::new(),
                    tree_id: cc_common::EMPTY_TREE_ID_BYTES.to_vec(),
                    sparse_patterns: Vec::new(),
                    tree_ids: vec![cc_common::EMPTY_TREE_ID_BYTES.to_vec()],
                }),
            })
            .await?;
        Ok(())
    })
    .await
    .unwrap();

    jj_cmd(&ws_dir, &xdg)
        .args(["cc", "workspace", "status", "--workspace-name", "broken"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Failed to query cloud workspace status",
        ));
}
