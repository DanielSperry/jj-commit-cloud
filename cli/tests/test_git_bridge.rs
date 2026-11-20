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

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use cc_common::op_store::{ReadOperationRequest, ReadViewRequest, ReconcileOpHeadsRequest, View};
use jj_lib::backend::CommitId;
use jj_lib::git_backend::synthetic_change_id_from_git_commit_id;
use jj_lib::object_id::ObjectId as _;

fn run_git(repo_dir: &Path, args: &[&str]) -> String {
    let out = StdCommand::new("git")
        .current_dir(repo_dir)
        .args(args)
        .env("GIT_AUTHOR_NAME", "Git Author")
        .env("GIT_AUTHOR_EMAIL", "author@example.com")
        .env("GIT_AUTHOR_DATE", "1730000000 +0000")
        .env("GIT_COMMITTER_NAME", "Git Committer")
        .env("GIT_COMMITTER_EMAIL", "committer@example.com")
        .env("GIT_COMMITTER_DATE", "1730000000 +0000")
        .output()
        .expect("git CLI should execute");
    assert!(
        out.status.success(),
        "git {args:?} failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

fn run_git_bare(bare_dir: &Path, args: &[&str]) -> String {
    let out = StdCommand::new("git")
        .arg("--git-dir")
        .arg(bare_dir)
        .args(args)
        .output()
        .expect("git CLI should execute");
    assert!(
        out.status.success(),
        "git --git-dir {} {args:?} failed:\nstdout: {}\nstderr: {}",
        bare_dir.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

fn jj_cmd(cwd: &Path, xdg: &Path) -> Command {
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("jj");
    cmd.current_dir(cwd)
        .env("XDG_CONFIG_HOME", xdg)
        .env("JJ_USER", "Cloud Developer")
        .env("JJ_EMAIL", "dev@example.com");
    cmd
}

/// Runs `git init` with the given ref storage format. Returns false if this Git cannot create
/// that format (`--ref-format` needs Git 2.45; older versions only have the files format).
fn git_init_with_ref_format(repo_dir: &Path, ref_format: &str) -> bool {
    fs::create_dir_all(repo_dir).unwrap();
    let out = StdCommand::new("git")
        .current_dir(repo_dir)
        .args(["init", "-b", "main", &format!("--ref-format={ref_format}")])
        .output()
        .expect("git CLI should execute");
    if out.status.success() {
        return true;
    }
    if ref_format == "files" {
        run_git(repo_dir, &["init", "-b", "main"]);
        return true;
    }
    false
}

async fn read_head_view(server_url: &str, repo_id: &str) -> View {
    let server_url = server_url.to_string();
    let repo_id = repo_id.to_string();
    cc_lib::util::run_on_tokio(async move {
        let mut client = cc_lib::util::connect_op_store_client(server_url).await?;
        let op_head = client
            .reconcile_op_heads(ReconcileOpHeadsRequest {
                repo_id: repo_id.clone(),
            })
            .await?
            .into_inner()
            .op_head;
        let op = client
            .read_operation(ReadOperationRequest {
                repo_id: repo_id.clone(),
                operation_id: op_head,
            })
            .await?
            .into_inner()
            .operation
            .expect("operation present");
        let view = client
            .read_view(ReadViewRequest {
                repo_id,
                view_id: op.view_id,
            })
            .await?
            .into_inner()
            .view
            .expect("view present");
        Ok(view)
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn test_git_import_and_export_preserve_exact_sha1_commits_trees_blobs_symlinks_and_merges() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xdg = dir.path().join("xdg");
    fs::create_dir_all(&xdg).unwrap();
    let server = testutils::spawn_sqlite_server(&dir.path().join("git_bridge.db")).await;

    // 1. Construct a multi-branch Git repository with nested dirs, >4 MiB blob, executable,
    //    symlink, and merge.
    let src_git = dir.path().join("src_git");
    fs::create_dir_all(src_git.join("src")).unwrap();
    fs::create_dir_all(src_git.join("bin")).unwrap();
    run_git(&src_git, &["init", "-b", "main"]);

    fs::write(src_git.join("src/lib.rs"), "pub fn base() -> u32 { 1 }\n").unwrap();
    let script_path = src_git.join("bin/deploy.sh");
    fs::write(&script_path, "#!/bin/sh\necho ok\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755)).unwrap();
        std::os::unix::fs::symlink("../src/lib.rs", src_git.join("bin/link.rs")).unwrap();
    }
    let large_payload: Vec<u8> = (0..(4 * 1024 * 1024 + 128 * 1024))
        .map(|i| ((i * 17 + 3) & 0xff) as u8)
        .collect();
    fs::write(src_git.join("model.bin"), &large_payload).unwrap();

    run_git(&src_git, &["add", "."]);
    run_git(&src_git, &["commit", "-m", "initial git commit"]);
    let git_c1 = run_git(&src_git, &["rev-parse", "HEAD"]);

    // Feature branch commit
    run_git(&src_git, &["checkout", "-b", "feature/auth"]);
    fs::write(
        src_git.join("src/auth.rs"),
        "pub fn auth() -> bool { true }\n",
    )
    .unwrap();
    run_git(&src_git, &["add", "."]);
    run_git(&src_git, &["commit", "-m", "add auth module"]);
    let git_feature = run_git(&src_git, &["rev-parse", "HEAD"]);

    // Mainline commit + merge commit (diamond DAG!)
    run_git(&src_git, &["checkout", "main"]);
    fs::write(src_git.join("README.md"), "# Monorepo\n").unwrap();
    run_git(&src_git, &["add", "."]);
    run_git(&src_git, &["commit", "-m", "add readme on main"]);
    run_git(
        &src_git,
        &[
            "merge",
            "--no-ff",
            "feature/auth",
            "-m",
            "merge feature/auth",
        ],
    );
    let git_main_merge = run_git(&src_git, &["rev-parse", "main"]);
    let git_main_tree = run_git(&src_git, &["rev-parse", "main^{tree}"]);

    // 2. Import into Commit Cloud via `jj cc import-git`.
    let import_stdout = String::from_utf8(
        jj_cmd(dir.path(), &xdg)
            .args([
                "cc",
                "import-git",
                "--server",
                server.url(),
                "--project-id",
                "proj-git",
                "--repo-id",
                "repo-git-1",
                "--create",
                "--git-dir",
                src_git.to_str().unwrap(),
            ])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone(),
    )
    .unwrap();
    assert!(import_stdout.contains("Commits: 4"));

    // 3. Clone into a Commit Cloud workspace and verify exact Git SHA-1 preservation.
    let ws = dir.path().join("cc_ws");
    jj_cmd(dir.path(), &xdg)
        .args([
            "cc",
            "clone",
            "--server",
            server.url(),
            "-r",
            "main",
            "repo-git-1",
            ws.to_str().unwrap(),
        ])
        .assert()
        .success();

    let resolved_main = String::from_utf8(
        jj_cmd(&ws, &xdg)
            .args(["log", "--no-graph", "-r", "main", "-T", "commit_id"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone(),
    )
    .unwrap()
    .trim()
    .to_string();
    assert_eq!(resolved_main, git_main_merge);

    let resolved_feature = String::from_utf8(
        jj_cmd(&ws, &xdg)
            .args(["log", "--no-graph", "-r", "feature/auth", "-T", "commit_id"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone(),
    )
    .unwrap()
    .trim()
    .to_string();
    assert_eq!(resolved_feature, git_feature);
    assert_eq!(fs::read(ws.join("model.bin")).unwrap(), large_payload);
    assert_eq!(
        fs::read_to_string(ws.join("src/auth.rs")).unwrap(),
        "pub fn auth() -> bool { true }\n"
    );

    // 4. Author a new Commit Cloud commit stacked on `main` and advance `main`.
    fs::write(ws.join("src/cloud.rs"), "pub fn cloud() {}\n").unwrap();
    jj_cmd(&ws, &xdg)
        .args(["describe", "-m", "authored in Commit Cloud"])
        .assert()
        .success();
    jj_cmd(&ws, &xdg)
        .args(["bookmark", "set", "main", "-r", "@"])
        .assert()
        .success();
    let cc_new_head = String::from_utf8(
        jj_cmd(&ws, &xdg)
            .args(["log", "--no-graph", "-r", "main", "-T", "commit_id"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone(),
    )
    .unwrap()
    .trim()
    .to_string();

    // 5. Export back to a fresh bare Git repository via `jj cc export-git` and verify `git fsck` +
    //    1:1 SHA-1s!
    let dst_git = dir.path().join("exported.git");
    jj_cmd(&ws, &xdg)
        .args([
            "cc",
            "export-git",
            "--bare",
            "--git-dir",
            dst_git.to_str().unwrap(),
        ])
        .assert()
        .success();

    run_git_bare(&dst_git, &["fsck", "--full"]);
    assert_eq!(run_git_bare(&dst_git, &["rev-parse", "main"]), cc_new_head);
    assert_eq!(
        run_git_bare(&dst_git, &["rev-parse", "main~1"]),
        git_main_merge
    );
    assert_eq!(
        run_git_bare(&dst_git, &["rev-parse", "main~1^{tree}"]),
        git_main_tree
    );
    assert_eq!(
        run_git_bare(&dst_git, &["rev-parse", "feature/auth"]),
        git_feature
    );
    assert_eq!(
        run_git_bare(&dst_git, &["rev-parse", "feature/auth~1"]),
        git_c1
    );
}

#[tokio::test]
async fn test_git_import_and_export_roundtrip_submodule_gitlinks_and_external_ids() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xdg = dir.path().join("xdg");
    fs::create_dir_all(&xdg).unwrap();
    // Exercise `MemoryMetadataStore` too (verifying P0-1 UTF-8 sort order on Git-imported trees!).
    let server = testutils::spawn_server().await;

    // 1. Create a standalone Git repository commit to reference as a `160000` gitlink submodule.
    let sub_git = dir.path().join("sub_git");
    fs::create_dir_all(&sub_git).unwrap();
    run_git(&sub_git, &["init", "-b", "main"]);
    fs::write(sub_git.join("sub.txt"), "submodule v1\n").unwrap();
    run_git(&sub_git, &["add", "."]);
    run_git(&sub_git, &["commit", "-m", "submodule commit"]);
    let sub_commit_hex = run_git(&sub_git, &["rev-parse", "HEAD"]);

    // 2. Create parent Git repo with `vendor/sub` (`160000`) AND `src/foo/mod.rs` + `src/foo.rs`!
    let parent_git = dir.path().join("parent_git");
    fs::create_dir_all(parent_git.join("src/foo")).unwrap();
    run_git(&parent_git, &["init", "-b", "main"]);
    fs::write(parent_git.join("src/foo/mod.rs"), "pub mod inner;\n").unwrap();
    fs::write(parent_git.join("src/foo.rs"), "pub fn top() {}\n").unwrap();
    fs::write(
        parent_git.join(".gitmodules"),
        "[submodule \"vendor/sub\"]\n\tpath = vendor/sub\n\turl = https://example.com/sub.git\n",
    )
    .unwrap();
    run_git(&parent_git, &["add", "."]);
    run_git(
        &parent_git,
        &[
            "update-index",
            "--add",
            "--cacheinfo",
            &format!("160000,{sub_commit_hex},vendor/sub"),
        ],
    );
    run_git(
        &parent_git,
        &[
            "commit",
            "-m",
            "parent commit with submodule and foo/foo.rs",
        ],
    );
    let expected_commit = run_git(&parent_git, &["rev-parse", "main"]);
    let expected_tree = run_git(&parent_git, &["rev-parse", "main^{tree}"]);

    // 3. Import into Commit Cloud (`MemoryMetadataStore`).
    jj_cmd(dir.path(), &xdg)
        .args([
            "cc",
            "import-git",
            "--server",
            server.url(),
            "--project-id",
            "proj-submod",
            "--repo-id",
            "repo-submod",
            "--create",
            "--git-dir",
            parent_git.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Verify `Commit.external_ids["git-sha1"]` was recorded on import!
    let expected_commit_bytes = hex::decode(&expected_commit).unwrap();
    let server_url = server.url().to_string();
    let stored_commit = cc_lib::util::run_on_tokio(async move {
        let mut client = cc_lib::util::connect_backend_client(server_url).await?;
        let resp = client
            .read_commit(cc_common::backend::ReadCommitRequest {
                project_id: "repo-submod".to_string(),
                commit_id: expected_commit_bytes.clone(),
            })
            .await?
            .into_inner();
        Ok(resp.commit.expect("commit present"))
    })
    .await
    .unwrap();
    assert_eq!(
        stored_commit.external_ids.get("git-sha1").map(hex::encode),
        Some(expected_commit.clone())
    );

    // 4. Clone workspace and export back to bare Git, verifying `160000` gitlink and 1:1 SHA-1!
    let ws = dir.path().join("ws_submod");
    jj_cmd(dir.path(), &xdg)
        .args([
            "cc",
            "clone",
            "--server",
            server.url(),
            "-r",
            "main",
            "repo-submod",
            ws.to_str().unwrap(),
        ])
        .assert()
        .success();

    let exported_git = dir.path().join("exported_submod.git");
    jj_cmd(&ws, &xdg)
        .args([
            "cc",
            "export-git",
            "--bare",
            "--git-dir",
            exported_git.to_str().unwrap(),
        ])
        .assert()
        .success();

    run_git_bare(&exported_git, &["fsck", "--full"]);
    assert_eq!(
        run_git_bare(&exported_git, &["rev-parse", "main"]),
        expected_commit
    );
    assert_eq!(
        run_git_bare(&exported_git, &["rev-parse", "main^{tree}"]),
        expected_tree
    );
    let ls_sub = run_git_bare(&exported_git, &["ls-tree", "main", "vendor/sub"]);
    assert!(
        ls_sub.contains(&format!("160000 commit {sub_commit_hex}\tvendor/sub")),
        "expected 160000 gitlink entry in exported tree, got: {ls_sub}"
    );
}

#[tokio::test]
async fn test_export_git_skips_bookmarks_that_are_not_valid_git_branch_names() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xdg = dir.path().join("xdg");
    fs::create_dir_all(&xdg).unwrap();
    let server = testutils::spawn_server().await;

    let ws = dir.path().join("ws_names");
    jj_cmd(dir.path(), &xdg)
        .args([
            "cc",
            "init",
            "--server",
            server.url(),
            "--create",
            ws.to_str().unwrap(),
        ])
        .assert()
        .success();
    fs::write(ws.join("file.txt"), "bookmarked\n").unwrap();
    jj_cmd(&ws, &xdg)
        .args(["describe", "-m", "bookmarked commit"])
        .assert()
        .success();
    let commit_hex = String::from_utf8(
        jj_cmd(&ws, &xdg)
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

    // Files that a path-joined bookmark name would overwrite: an absolute path, and
    // `<dst>/refs/heads/../../../victim_rel`, which resolves to `<tempdir>/victim_rel`.
    let victim_abs = dir.path().join("victim_abs");
    let victim_rel = dir.path().join("victim_rel");
    fs::write(&victim_abs, "original\n").unwrap();
    fs::write(&victim_rel, "original\n").unwrap();
    let invalid_names = [
        victim_abs.to_str().unwrap().to_string(),
        "../x".to_string(),
        "../../../victim_rel".to_string(),
        "has space".to_string(),
        "tilde~1".to_string(),
        "ends.lock".to_string(),
    ];
    for name in invalid_names.iter().map(String::as_str).chain(["good"]) {
        jj_cmd(&ws, &xdg)
            .args(["bookmark", "create", "-r", "@", &format!("{name:?}")])
            .assert()
            .success();
    }

    let dst_git = dir.path().join("exported_names.git");
    let export_stderr = String::from_utf8(
        jj_cmd(&ws, &xdg)
            .args([
                "cc",
                "export-git",
                "--bare",
                "--git-dir",
                dst_git.to_str().unwrap(),
            ])
            .assert()
            .success()
            .get_output()
            .stderr
            .clone(),
    )
    .unwrap();

    for name in &invalid_names {
        assert!(
            export_stderr.contains(&format!("Skipped bookmark {name:?}")),
            "expected a warning for {name:?}, got: {export_stderr}"
        );
    }
    assert_eq!(fs::read_to_string(&victim_abs).unwrap(), "original\n");
    assert_eq!(fs::read_to_string(&victim_rel).unwrap(), "original\n");
    assert!(!dst_git.join("refs/heads/has space").exists());
    run_git_bare(&dst_git, &["fsck", "--full"]);
    assert_eq!(
        run_git_bare(&dst_git, &["for-each-ref", "--format=%(refname)"]),
        "refs/heads/good"
    );
    assert_eq!(
        run_git_bare(&dst_git, &["symbolic-ref", "HEAD"]),
        "refs/heads/good"
    );
    assert_eq!(run_git_bare(&dst_git, &["rev-parse", "good"]), commit_hex);
}

#[tokio::test]
async fn test_import_git_reads_only_jj_visible_refs_and_leaves_source_untouched() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xdg = dir.path().join("xdg");
    fs::create_dir_all(&xdg).unwrap();
    let server = testutils::spawn_server().await;

    for ref_format in ["files", "reftable"] {
        let src_git = dir.path().join(format!("src_{ref_format}"));
        if !git_init_with_ref_format(&src_git, ref_format) {
            eprintln!("skipping {ref_format}: this Git cannot create it");
            continue;
        }
        fs::write(src_git.join("a.txt"), "a\n").unwrap();
        run_git(&src_git, &["add", "."]);
        run_git(&src_git, &["commit", "-m", "main commit"]);
        let main = run_git(&src_git, &["rev-parse", "HEAD"]);
        let tree = run_git(&src_git, &["rev-parse", "HEAD^{tree}"]);
        let child = |msg: &str| run_git(&src_git, &["commit-tree", &tree, "-p", &main, "-m", msg]);
        let tagged = child("reachable only from a lightweight tag");
        let annotated = child("reachable only from an annotated tag");
        let remote = child("reachable only from origin/feature");
        let kept = child("reachable only from refs/jj/keep");
        let pulled = child("reachable only from refs/pull");
        run_git(&src_git, &["tag", "v1", &tagged]);
        run_git(
            &src_git,
            &["tag", "-a", "v2", "-m", "annotated v2", &annotated],
        );
        run_git(
            &src_git,
            &["update-ref", "refs/remotes/origin/feature", &remote],
        );
        run_git(
            &src_git,
            &[
                "symbolic-ref",
                "refs/remotes/origin/HEAD",
                "refs/remotes/origin/feature",
            ],
        );
        run_git(
            &src_git,
            &["update-ref", &format!("refs/jj/keep/{kept}"), &kept],
        );
        run_git(&src_git, &["update-ref", "refs/pull/1/head", &pulled]);
        run_git(&src_git, &["notes", "add", "-m", "a note", "HEAD"]);
        fs::write(src_git.join("a.txt"), "stashed edit\n").unwrap();
        run_git(&src_git, &["stash"]);

        let refs_before = run_git(
            &src_git,
            &["for-each-ref", "--format=%(refname) %(objectname)"],
        );
        let head_file_before = fs::read(src_git.join(".git/HEAD")).unwrap();

        let repo_id = format!("repo-refs-{ref_format}");
        let import_stdout = String::from_utf8(
            jj_cmd(dir.path(), &xdg)
                .args([
                    "cc",
                    "import-git",
                    "--server",
                    server.url(),
                    "--repo-id",
                    &repo_id,
                    "--create",
                    "--git-dir",
                    src_git.to_str().unwrap(),
                ])
                .assert()
                .success()
                .get_output()
                .stdout
                .clone(),
        )
        .unwrap();
        assert!(
            import_stdout.contains("Commits: 4"),
            "{ref_format}: {import_stdout}"
        );

        // The importer must not rewrite HEAD (reftable keeps a `refs/heads/.invalid` stub) or refs.
        assert_eq!(
            run_git(
                &src_git,
                &["for-each-ref", "--format=%(refname) %(objectname)"]
            ),
            refs_before
        );
        assert_eq!(
            fs::read(src_git.join(".git/HEAD")).unwrap(),
            head_file_before
        );

        let view = read_head_view(server.url(), &repo_id).await;
        let heads: HashSet<String> = view.head_ids.iter().map(hex::encode).collect();
        assert_eq!(
            heads,
            HashSet::from([
                main.clone(),
                tagged.clone(),
                annotated.clone(),
                remote.clone()
            ])
        );
        assert_eq!(view.local_bookmarks.keys().collect::<Vec<_>>(), ["main"]);
        assert_eq!(
            view.remote_bookmarks.keys().collect::<Vec<_>>(),
            ["feature@origin"]
        );
        assert!(
            !view.remote_bookmarks["feature@origin"].is_tracked,
            "jj git import leaves new remote bookmarks untracked"
        );

        // Creating a local `feature` bookmark must succeed because `feature@origin` is untracked.
        let ws = dir.path().join(format!("ws_{ref_format}"));
        jj_cmd(dir.path(), &xdg)
            .args([
                "cc",
                "clone",
                "--server",
                server.url(),
                "-r",
                "main",
                &repo_id,
                ws.to_str().unwrap(),
            ])
            .assert()
            .success();
        jj_cmd(&ws, &xdg)
            .args(["bookmark", "create", "feature", "-r", "@"])
            .assert()
            .success();

        // Commits without a `change-id` header get the same change ID as in jj's GitBackend.
        let server_url = server.url().to_string();
        let main_id = hex::decode(&main).unwrap();
        let stored = cc_lib::util::run_on_tokio(async move {
            let mut client = cc_lib::util::connect_backend_client(server_url).await?;
            let resp = client
                .read_commit(cc_common::backend::ReadCommitRequest {
                    project_id: repo_id,
                    commit_id: main_id,
                })
                .await?
                .into_inner();
            Ok(resp.commit.expect("commit present"))
        })
        .await
        .unwrap();
        let expected_change_id =
            synthetic_change_id_from_git_commit_id(&CommitId::try_from_hex(&main).unwrap());
        assert_eq!(stored.change_id, expected_change_id.as_bytes());
    }
}

#[tokio::test]
async fn test_import_git_from_jj_colocated_repo_excludes_keep_refs_and_fabricates_no_main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xdg = dir.path().join("xdg");
    fs::create_dir_all(&xdg).unwrap();
    let server = testutils::spawn_server().await;

    let colocated = dir.path().join("colocated_src");
    jj_cmd(dir.path(), &xdg)
        .args(["git", "init", "--colocate", colocated.to_str().unwrap()])
        .assert()
        .success();
    fs::write(colocated.join("note.txt"), "obsolete v1\n").unwrap();
    jj_cmd(&colocated, &xdg)
        .args(["describe", "-m", "v1"])
        .assert()
        .success();
    fs::write(colocated.join("note.txt"), "live v2\n").unwrap();
    jj_cmd(&colocated, &xdg)
        .args(["describe", "-m", "v2"])
        .assert()
        .success();
    // In a colocated jj workspace, `.git/HEAD` tracks `@-` while `refs/jj/keep/*` retains `@`
    // and every rewritten predecessor (`v1`).
    jj_cmd(&colocated, &xdg).args(["new"]).assert().success();

    let live_commit = String::from_utf8(
        jj_cmd(&colocated, &xdg)
            .args(["log", "--no-graph", "-r", "@-", "-T", "commit_id"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone(),
    )
    .unwrap()
    .trim()
    .to_string();
    let keep_refs = run_git(&colocated, &["for-each-ref", "refs/jj/keep"]);
    assert!(
        keep_refs.lines().count() >= 2,
        "rewriting `@` in a colocated jj repo should leave multiple refs/jj/keep/* entries: {keep_refs}"
    );
    let head_before = fs::read(colocated.join(".git/HEAD")).unwrap();

    let import_stdout = String::from_utf8(
        jj_cmd(dir.path(), &xdg)
            .args([
                "cc",
                "import-git",
                "--server",
                server.url(),
                "--repo-id",
                "repo-colocated",
                "--create",
                "--git-dir",
                colocated.to_str().unwrap(),
            ])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone(),
    )
    .unwrap();
    assert!(import_stdout.contains("Commits: 1"), "got: {import_stdout}");
    assert_eq!(fs::read(colocated.join(".git/HEAD")).unwrap(), head_before);

    let view = read_head_view(server.url(), "repo-colocated").await;
    let heads: HashSet<String> = view.head_ids.iter().map(hex::encode).collect();
    assert_eq!(heads, HashSet::from([live_commit.clone()]));
    assert!(
        view.local_bookmarks.is_empty(),
        "importer must not fabricate a `main` bookmark when none exist: {:?}",
        view.local_bookmarks.keys().collect::<Vec<_>>()
    );

    let cloned = dir.path().join("cloned_colocated");
    jj_cmd(dir.path(), &xdg)
        .args([
            "cc",
            "clone",
            "--server",
            server.url(),
            "repo-colocated",
            cloned.to_str().unwrap(),
        ])
        .assert()
        .success();
    assert_eq!(
        fs::read_to_string(cloned.join("note.txt")).unwrap(),
        "live v2\n"
    );
}
