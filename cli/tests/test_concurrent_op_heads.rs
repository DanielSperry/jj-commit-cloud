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
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::Barrier;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_agents_commit_and_reconcile_server_side_without_locks() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xdg_dir = dir.path().join("xdg_config");
    fs::create_dir_all(&xdg_dir).unwrap();
    let server = testutils::spawn_sqlite_server(&dir.path().join("concurrent_agents.db")).await;
    let server_url = server.url().to_string();

    // Initialize shared repository via Workspace 0.
    let ws0 = dir.path().join("ws-0");
    let init_status = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("jj"))
        .args([
            "cc",
            "init",
            "--server",
            &server_url,
            "--create",
            ws0.to_str().unwrap(),
        ])
        .env("XDG_CONFIG_HOME", &xdg_dir)
        .env("JJ_USER", "Agent 0")
        .env("JJ_EMAIL", "agent0@example.com")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .unwrap();
    assert!(init_status.success());

    let cfg: cc_lib::util::CommitCloudConfig =
        cc_lib::util::CommitCloudConfig::load_from_store(&ws0).unwrap();
    let repo_id = cfg.repo_id;

    // Clone 6 independent agent workspaces attached to the same `repo_id`.
    let agent_count = 6usize;
    let mut ws_paths: Vec<PathBuf> = Vec::with_capacity(agent_count);
    for idx in 1..=agent_count {
        let ws = dir.path().join(format!("agent-{idx}"));
        let st = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("jj"))
            .args([
                "cc",
                "clone",
                "--server",
                &server_url,
                "--workspace-name",
                &format!("agent-{idx}"),
                &repo_id,
                ws.to_str().unwrap(),
            ])
            .env("XDG_CONFIG_HOME", &xdg_dir)
            .env("JJ_USER", format!("Agent {idx}"))
            .env("JJ_EMAIL", format!("agent{idx}@example.com"))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .unwrap();
        assert!(st.success());
        ws_paths.push(ws);
    }

    // All 6 agents write distinct files and bookmarks concurrently behind a barrier.
    let barrier = Arc::new(Barrier::new(agent_count));
    let mut handles = Vec::with_capacity(agent_count);
    for (idx, ws) in ws_paths.iter().cloned().enumerate() {
        let b = barrier.clone();
        let xdg = xdg_dir.clone();
        handles.push(tokio::spawn(async move {
            let agent_num = idx + 1;
            fs::write(
                ws.join(format!("agent_{agent_num}.txt")),
                format!("work by agent {agent_num}\n"),
            )
            .unwrap();
            b.wait().await;
            let st1 = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("jj"))
                .current_dir(&ws)
                .args(["describe", "-m", &format!("agent-{agent_num} commit")])
                .env("XDG_CONFIG_HOME", &xdg)
                .env("JJ_USER", format!("Agent {agent_num}"))
                .env("JJ_EMAIL", format!("agent{agent_num}@example.com"))
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await
                .unwrap();
            assert!(st1.success());

            let st2 = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("jj"))
                .current_dir(&ws)
                .args([
                    "bookmark",
                    "create",
                    &format!("bm-agent-{agent_num}"),
                    "-r",
                    "@",
                ])
                .env("XDG_CONFIG_HOME", &xdg)
                .env("JJ_USER", format!("Agent {agent_num}"))
                .env("JJ_EMAIL", format!("agent{agent_num}@example.com"))
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await
                .unwrap();
            assert!(st2.success());
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    // Next read invokes server-side `ReconcileOpHeads`, merging all 6 concurrent agent views
    // into a single operation head with zero lost commits or bookmarks!
    let out = tokio::process::Command::new(assert_cmd::cargo::cargo_bin!("jj"))
        .current_dir(&ws0)
        .args(["bookmark", "list", "--no-pager"])
        .env("XDG_CONFIG_HOME", &xdg_dir)
        .output()
        .await
        .unwrap();
    assert!(out.status.success());
    let bm_stdout = String::from_utf8_lossy(&out.stdout);
    for idx in 1..=agent_count {
        assert!(
            bm_stdout.contains(&format!("bm-agent-{idx}")),
            "missing bookmark bm-agent-{idx} in reconciled view:\n{bm_stdout}"
        );
    }

    // Verify server `GetOpHeads` now reports a single converged operation head.
    let heads_store = cc_lib::cc_op_heads_store::CommitCloudOpHeadsStore::new(repo_id, server_url);
    let raw_heads = heads_store
        .get_op_heads_without_reconciliation()
        .await
        .unwrap();
    assert_eq!(raw_heads.len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_rewrite_rebases_descendant_without_divergent_change() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xdg_dir = dir.path().join("xdg_config");
    fs::create_dir_all(&xdg_dir).unwrap();
    let server = testutils::spawn_sqlite_server(&dir.path().join("rewrite_reconcile.db")).await;
    let server_url = server.url().to_string();

    let run_jj = |cwd: &std::path::Path, args: &[&str]| {
        let mut cmd = std::process::Command::new(assert_cmd::cargo::cargo_bin!("jj"));
        cmd.current_dir(cwd)
            .args(args)
            .env("XDG_CONFIG_HOME", &xdg_dir)
            .env("JJ_USER", "Rewrite Tester")
            .env("JJ_EMAIL", "tester@example.com");
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "jj {:?} failed:\nstdout: {}\nstderr: {}",
            args,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    };

    let ws1 = dir.path().join("ws-1");
    let ws2 = dir.path().join("ws-2");
    run_jj(
        dir.path(),
        &[
            "cc",
            "init",
            "--server",
            &server_url,
            "--create",
            ws1.to_str().unwrap(),
        ],
    );
    let cfg = cc_lib::util::CommitCloudConfig::load_from_store(&ws1).unwrap();
    let repo_id = cfg.repo_id;

    // Attach ws-2 before X is authored so ws-2 starts on root().
    run_jj(
        dir.path(),
        &[
            "cc",
            "init",
            "--server",
            &server_url,
            "--repo-id",
            &repo_id,
            "--workspace-name",
            "ws-2",
            ws2.to_str().unwrap(),
        ],
    );

    // Author commit X in ws-1 and capture the shared fork operation `op_sync`.
    fs::write(ws1.join("shared.txt"), "v1\n").unwrap();
    run_jj(&ws1, &["describe", "-m", "commit X v1"]);
    let x_cid = run_jj(&ws1, &["log", "--no-graph", "-r", "@", "-T", "commit_id"])
        .trim()
        .to_string();
    let x_change_id = run_jj(&ws1, &["log", "--no-graph", "-r", "@", "-T", "change_id"])
        .trim()
        .to_string();
    let op_sync = run_jj(
        &ws1,
        &[
            "op",
            "log",
            "--no-graph",
            "--limit",
            "1",
            "-T",
            "id.short(32)",
        ],
    )
    .trim()
    .to_string();

    // Sync ws-2's local index up to `op_sync`.
    run_jj(&ws2, &["status"]);

    // Branch 1 (ws-1): rewrite X -> X' (`shared.txt = "v2\n"`, description `"commit X v2"`).
    fs::write(ws1.join("shared.txt"), "v2\n").unwrap();
    run_jj(&ws1, &["describe", "-m", "commit X v2"]);
    let x_rewritten_cid = run_jj(&ws1, &["log", "--no-graph", "-r", "@", "-T", "commit_id"])
        .trim()
        .to_string();
    assert_ne!(x_cid, x_rewritten_cid);

    // Branch 2 (ws-2): starting from `op_sync` (before X -> X'), create a child Y on top of X!
    run_jj(&ws2, &["--at-op", &op_sync, "new", &x_cid, "-m", "child Y"]);

    // Confirm both divergent operation heads exist on the server prior to reconciliation.
    let heads_store = cc_lib::cc_op_heads_store::CommitCloudOpHeadsStore::new(repo_id, server_url);
    let divergent_heads = heads_store
        .get_op_heads_without_reconciliation()
        .await
        .unwrap();
    assert_eq!(
        divergent_heads.len(),
        2,
        "expected 2 divergent op heads before reconciliation"
    );

    // Reconciling via any workspace defers cross-branch rewrites to `DefaultIndex`, rebasing Y
    // onto X' cleanly so X's change_id resolves unambiguously with zero divergence!
    let resolved_x = run_jj(
        &ws1,
        &["log", "--no-graph", "-r", &x_change_id, "-T", "commit_id"],
    )
    .trim()
    .to_string();
    assert_eq!(resolved_x, x_rewritten_cid);

    let y_parent_cid = run_jj(
        &ws1,
        &[
            "log",
            "--no-graph",
            "-r",
            "description(substring:\"child Y\")-",
            "-T",
            "commit_id",
        ],
    )
    .trim()
    .to_string();
    assert_eq!(
        y_parent_cid, x_rewritten_cid,
        "child Y must be rebased onto rewritten X'"
    );

    let y_file = run_jj(
        &ws1,
        &[
            "file",
            "show",
            "shared.txt",
            "-r",
            "description(substring:\"child Y\")",
        ],
    );
    assert_eq!(y_file, "v2\n");

    let converged_heads = heads_store
        .get_op_heads_without_reconciliation()
        .await
        .unwrap();
    assert_eq!(converged_heads.len(), 1);
}
