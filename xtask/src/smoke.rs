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
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::local::resolve_bearer_token;
use crate::state::{AuthMode, BoxError, cargo_target_dir, load_state, workspace_root};

pub struct SmokeOptions {
    pub env_name: String,
    pub server_override: Option<String>,
    pub peer_server_override: Option<String>,
    pub jj_bin_override: Option<PathBuf>,
    pub keep_workspaces: bool,
}

pub fn run_smoke(opts: &SmokeOptions) -> Result<(), BoxError> {
    let root = workspace_root();
    let (primary_url, peer_url, auth_mode, server_token, mode_label) = if let Some(
        explicit_server,
    ) = &opts.server_override
    {
        (
            explicit_server.clone(),
            opts.peer_server_override.clone(),
            AuthMode::None,
            None,
            "explicit".to_string(),
        )
    } else {
        let state = load_state(&root, &opts.env_name)?.ok_or_else(|| {
                format!(
                    "no active environment state found in .jj-cc for '{}' (start a server first with `cargo xtask up <mode>`)",
                    opts.env_name
                )
            })?;
        (
            state.endpoints.primary,
            opts.peer_server_override.clone().or(state.endpoints.peer),
            state.auth,
            state.auth_token,
            format!("{} ({})", state.mode, state.runtime),
        )
    };

    let auth_token = resolve_bearer_token(auth_mode, server_token.as_deref())?;

    let jj_bin = match &opts.jj_bin_override {
        Some(p) => p.clone(),
        None => ensure_host_jj_binary(&root)?,
    };

    let temp_holder = tempfile::Builder::new().prefix("jj-cc-smoke-").tempdir()?;
    let base_dir = if opts.keep_workspaces {
        temp_holder.keep()
    } else {
        temp_holder.path().to_path_buf()
    };

    let ws_writer = base_dir.join("ws-writer");
    let ws_reader = base_dir.join("ws-reader");
    let checkout_url = peer_url.as_deref().unwrap_or(&primary_url);

    println!("==> Smoke test against {mode_label}");
    println!("    Writer endpoint:   {primary_url}");
    if peer_url.is_some() {
        println!("    Checkout endpoint: {checkout_url} (cross-region replica)");
    }
    let steps = if peer_url.is_some() { 7 } else { 6 };

    // 1. Initialize repository in ws-writer
    let init_out = run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &[
            "cc",
            "init",
            "--server",
            &primary_url,
            "--create",
            ws_writer.to_str().unwrap(),
        ],
    )?;
    let repo_id = extract_repo_id(&ws_writer, &init_out)?;
    println!("    [1/{steps}] Initialized remote repository {repo_id}");

    // 2. Commit Version 1
    fs::create_dir_all(ws_writer.join("src"))?;
    let v1_code = "pub fn version() -> &'static str { \"v1.0.0\" }\n";
    fs::write(ws_writer.join("src/lib.rs"), v1_code)?;
    run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &[
            "-R",
            ws_writer.to_str().unwrap(),
            "describe",
            "-m",
            "release v1.0.0",
        ],
    )?;
    let v1_commit = run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &[
            "-R",
            ws_writer.to_str().unwrap(),
            "log",
            "--no-graph",
            "-r",
            "@",
            "-T",
            "commit_id",
        ],
    )?
    .trim()
    .to_string();
    println!("    [2/{steps}] Published v1 commit {}", &v1_commit[..12]);

    // 3. Commit Version 2 (with >4 MiB streamed binary + Unix symlink)
    run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &["-R", ws_writer.to_str().unwrap(), "new"],
    )?;
    let v2_code = "pub fn version() -> &'static str { \"v2.0.0\" }\n";
    fs::write(ws_writer.join("src/lib.rs"), v2_code)?;
    fs::create_dir_all(ws_writer.join("assets"))?;
    let large_blob: Vec<u8> = (0..(4 * 1024 * 1024 + 512 * 1024))
        .map(|i| ((i * 29 + 7) & 0xff) as u8)
        .collect();
    fs::write(ws_writer.join("assets/weights.bin"), &large_blob)?;
    #[cfg(unix)]
    std::os::unix::fs::symlink("src/lib.rs", ws_writer.join("entry.rs"))?;

    run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &[
            "-R",
            ws_writer.to_str().unwrap(),
            "describe",
            "-m",
            "release v2.0.0 with streamed asset",
        ],
    )?;
    let v2_commit = run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &[
            "-R",
            ws_writer.to_str().unwrap(),
            "log",
            "--no-graph",
            "-r",
            "@",
            "-T",
            "commit_id",
        ],
    )?
    .trim()
    .to_string();
    println!(
        "    [3/{steps}] Published v2 commit {} (4.5 MiB streamed blob)",
        &v2_commit[..12]
    );

    // 4. Clone ws-reader from checkout_url and verify latest v2 head is checked out on disk
    run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &[
            "cc",
            "clone",
            "--server",
            checkout_url,
            &repo_id,
            ws_reader.to_str().unwrap(),
        ],
    )?;
    let cloned_v2 = fs::read_to_string(ws_reader.join("src/lib.rs"))?;
    if cloned_v2 != v2_code {
        return Err(format!("expected cloned v2 code, got {cloned_v2:?}").into());
    }
    let cloned_blob = fs::read(ws_reader.join("assets/weights.bin"))?;
    if cloned_blob != large_blob {
        return Err("cloned 4.5 MiB binary payload mismatch".into());
    }
    #[cfg(unix)]
    {
        let link_target = fs::read_link(ws_reader.join("entry.rs"))?;
        if link_target.to_str() != Some("src/lib.rs") {
            return Err(format!("unexpected symlink target {link_target:?}").into());
        }
    }
    println!("    [4/{steps}] Cloned second workspace and verified v2 files on disk");

    // 5. Check out historical v1 in ws-reader, verify contents, then check out v2 + short op prefix
    run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &["-R", ws_reader.to_str().unwrap(), "new", &v1_commit],
    )?;
    let checked_out_v1 = fs::read_to_string(ws_reader.join("src/lib.rs"))?;
    if checked_out_v1 != v1_code {
        return Err(format!("expected historical v1 code, got {checked_out_v1:?}").into());
    }
    if ws_reader.join("assets/weights.bin").exists() {
        return Err("assets/weights.bin should not exist when v1 is checked out".into());
    }

    run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &["-R", ws_reader.to_str().unwrap(), "new", &v2_commit],
    )?;
    let op_prefix = run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &[
            "-R",
            ws_reader.to_str().unwrap(),
            "op",
            "log",
            "--no-graph",
            "-n",
            "1",
            "-T",
            "id.short(6)",
        ],
    )?
    .trim()
    .to_string();
    run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &[
            "-R",
            ws_reader.to_str().unwrap(),
            "op",
            "show",
            &op_prefix,
            "--no-op-diff",
        ],
    )?;
    println!(
        "    [5/{steps}] Checked out historical v1 -> v2 and verified short op prefix ({op_prefix})"
    );

    // 6. Export Commit Cloud history to a bare Git repo and re-import into a fresh repo, verifying
    //    1:1 SHA-1 parity
    run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &[
            "-R",
            ws_writer.to_str().unwrap(),
            "bookmark",
            "set",
            "main",
            "-r",
            &v2_commit,
        ],
    )?;
    let exported_git = base_dir.join("roundtrip.git");
    run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &[
            "cc",
            "export-git",
            "--bare",
            "--server",
            checkout_url,
            "--repo-id",
            &repo_id,
            "--git-dir",
            exported_git.to_str().unwrap(),
        ],
    )?;
    let exported_main = fs::read_to_string(exported_git.join("refs/heads/main"))?
        .trim()
        .to_string();
    if exported_main != v2_commit {
        return Err(format!(
            "Git export SHA-1 mismatch: expected {v2_commit}, got {exported_main}"
        )
        .into());
    }
    run_jj(
        &jj_bin,
        auth_token.as_deref(),
        &[
            "cc",
            "import-git",
            "--server",
            &primary_url,
            "--create",
            "--git-dir",
            exported_git.to_str().unwrap(),
        ],
    )?;
    println!(
        "    [6/{steps}] Verified 1:1 Git SHA-1 export and re-import round-trip ({})",
        &exported_main[..12]
    );

    if peer_url.is_some() {
        // 7. Reverse-direction check: publish a new blob on the peer server and verify checkout on
        //    the primary server reads it via peer origin fallback.
        let v3_code = "pub fn peer_region() -> &'static str { \"authored-in-peer\" }\n";
        fs::write(ws_reader.join("src/peer.rs"), v3_code)?;
        run_jj(
            &jj_bin,
            auth_token.as_deref(),
            &[
                "-R",
                ws_reader.to_str().unwrap(),
                "describe",
                "-m",
                "peer-authored v3",
            ],
        )?;
        let v3_commit = run_jj(
            &jj_bin,
            auth_token.as_deref(),
            &[
                "-R",
                ws_reader.to_str().unwrap(),
                "log",
                "--no-graph",
                "-r",
                "@",
                "-T",
                "commit_id",
            ],
        )?
        .trim()
        .to_string();
        run_jj(
            &jj_bin,
            auth_token.as_deref(),
            &["-R", ws_writer.to_str().unwrap(), "new", &v3_commit],
        )?;
        let checked_out_peer = fs::read_to_string(ws_writer.join("src/peer.rs"))?;
        if checked_out_peer != v3_code {
            return Err(format!(
                "expected peer-authored file on primary, got {checked_out_peer:?}"
            )
            .into());
        }
        println!(
            "    [7/{steps}] Verified reverse peer -> primary blob read ({})",
            &v3_commit[..12]
        );
    }

    println!("==> Smoke verification PASSED");
    println!(
        "    Clone this repository interactively:\n      cargo run -p cli -- cc clone --server {checkout_url} {repo_id} ./my-checkout"
    );
    if opts.keep_workspaces {
        println!("    Preserved workspaces at: {}", base_dir.display());
    }
    Ok(())
}

pub fn ensure_host_jj_binary(root: &Path) -> Result<PathBuf, BoxError> {
    let exe_name = if cfg!(windows) { "jj.exe" } else { "jj" };
    let status = Command::new("cargo")
        .current_dir(root)
        .args(["build", "--quiet", "-p", "cli", "--bin", "jj"])
        .status()?;
    if !status.success() {
        return Err("`cargo build -p cli --bin jj` failed".into());
    }
    Ok(cargo_target_dir(root).join("debug").join(exe_name))
}

fn run_jj(jj_bin: &Path, auth_token: Option<&str>, args: &[&str]) -> Result<String, BoxError> {
    let mut cmd = Command::new(jj_bin);
    cmd.arg("--no-pager").args(args);
    if let Some(tok) = auth_token {
        cmd.env("JJ_CC_AUTH_TOKEN", tok);
    }
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(format!(
            "jj {:?} failed ({}):\nstdout: {}\nstderr: {}",
            args,
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn extract_repo_id(ws_dir: &Path, init_stdout: &str) -> Result<String, BoxError> {
    let cfg_path = ws_dir
        .join(".jj")
        .join("repo")
        .join("store")
        .join("config.toml");
    if cfg_path.exists() {
        let raw = fs::read_to_string(&cfg_path)?;
        for line in raw.lines() {
            if let Some(rest) = line.trim().strip_prefix("repo_id")
                && let Some((_, quoted)) = rest.split_once('"')
                && let Some((id, _)) = quoted.split_once('"')
            {
                return Ok(id.to_string());
            }
        }
    }
    Err(format!("failed to read repo_id after init ({init_stdout})").into())
}
