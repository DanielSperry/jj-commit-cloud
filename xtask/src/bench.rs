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

//! Multi-agent concurrency, throughput, and warm-cache hit-ratio benchmark (`cargo xtask bench`).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Stdio};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command as TokioCommand};
use tokio::sync::Barrier;
use tokio::task::JoinSet;

use crate::gcp::mint_gcloud_identity_token;
use crate::smoke::ensure_host_jj_binary;
use crate::state::{AuthMode, BoxError, cargo_target_dir, load_state, workspace_root};

pub struct BenchOptions {
    pub env_name: String,
    pub server_override: Option<String>,
    pub ephemeral: bool,
    pub agents: usize,
    pub commits_per_agent: usize,
    pub isolated_repos: bool,
    pub use_daemon: bool,
    pub jj_bin_override: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
struct DaemonStatsJson {
    l1_hits: u64,
    l2_hits: u64,
    misses: u64,
    singleflight_coalesced: u64,
    upstream_rpcs: u64,
}

struct EphemeralServer {
    child: Child,
    url: String,
}

impl Drop for EphemeralServer {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

struct BenchDaemonGuard {
    jj_bin: PathBuf,
    cwd: PathBuf,
    xdg_dir: PathBuf,
    daemon_bin: PathBuf,
    state_dir: PathBuf,
    cache_dir: PathBuf,
    active: bool,
}

impl BenchDaemonGuard {
    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for BenchDaemonGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let _ = StdCommand::new(&self.jj_bin)
            .current_dir(&self.cwd)
            .args(["--no-pager", "cc", "daemon", "stop"])
            .env("XDG_CONFIG_HOME", &self.xdg_dir)
            .env("JJ_CC_USE_DAEMON", "1")
            .env("JJ_CC_DAEMON_BIN", &self.daemon_bin)
            .env("JJ_CC_DAEMON_DIR", &self.state_dir)
            .env("JJ_CC_CACHE_DIR", &self.cache_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

pub async fn run_bench(opts: &BenchOptions) -> Result<(), BoxError> {
    if opts.agents == 0 || opts.commits_per_agent == 0 {
        return Err("`--agents` and `--commits-per-agent` must both be >= 1".into());
    }

    let root = workspace_root();
    let temp_dir = tempfile::Builder::new().prefix("jj-cc-bench-").tempdir()?;
    let xdg_dir = temp_dir.path().join("xdg");
    fs::create_dir_all(&xdg_dir)?;

    let mut _ephemeral_guard: Option<EphemeralServer> = None;
    let (target_url, auth_mode, mode_label) = if let Some(explicit) = &opts.server_override {
        (explicit.clone(), AuthMode::None, "explicit".to_string())
    } else if opts.ephemeral {
        let srv = spawn_ephemeral_sqlite_server(&root, &temp_dir.path().join("bench.db")).await?;
        let url = srv.url.clone();
        _ephemeral_guard = Some(srv);
        (url, AuthMode::None, "ephemeral-sqlite".to_string())
    } else if let Some(state) = load_state(&root, &opts.env_name)? {
        (
            state.endpoints.primary,
            state.auth,
            format!("{} ({})", state.mode, state.runtime),
        )
    } else {
        return Err(format!(
            "no active environment state found for '{}' (start a stack with `cargo xtask up <mode>` or pass `--ephemeral`)",
            opts.env_name
        )
        .into());
    };

    let auth_token = match auth_mode {
        AuthMode::None => std::env::var("JJ_CC_AUTH_TOKEN").ok(),
        AuthMode::GcloudIdentityToken => match std::env::var("JJ_CC_AUTH_TOKEN") {
            Ok(t) if !t.trim().is_empty() => Some(t),
            _ => Some(mint_gcloud_identity_token()?),
        },
    };

    let jj_bin = match &opts.jj_bin_override {
        Some(p) => p.clone(),
        None => ensure_host_jj_binary(&root)?,
    };

    let daemon_state_dir = temp_dir.path().join("daemon_state");
    let daemon_cache_dir = temp_dir.path().join("daemon_cache");
    let daemon_bin = if opts.use_daemon {
        Some(ensure_host_daemon_binary(&root)?)
    } else {
        None
    };

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let project_id = format!("bench-proj-{ts}");
    let shared_repo_id = format!("bench-repo-{ts}");

    let total_expected_commits = opts.agents * opts.commits_per_agent;
    println!("==> Multi-agent scale benchmark against {mode_label}");
    println!("    Endpoint:          {target_url}");
    println!(
        "    Topology:          {} agents x {} commits = {} commits ({})",
        opts.agents,
        opts.commits_per_agent,
        total_expected_commits,
        if opts.isolated_repos {
            "isolated repos sharing project_id"
        } else {
            "shared repo_id with server-side ReconcileOpHeads"
        }
    );
    println!(
        "    Local daemon:      {}",
        if opts.use_daemon {
            "enabled (L1 memory + L2 disk CAS)"
        } else {
            "disabled (direct HTTP/2)"
        }
    );

    let mut daemon_guard = if let Some(dbin) = &daemon_bin {
        run_jj_async(
            &jj_bin,
            temp_dir.path(),
            &xdg_dir,
            auth_token.as_deref(),
            Some((dbin, &daemon_state_dir, &daemon_cache_dir)),
            &[
                "cc",
                "daemon",
                "start",
                "--upstream",
                &target_url,
                "--idle-timeout-secs",
                "60",
            ],
        )
        .await?;
        Some(BenchDaemonGuard {
            jj_bin: jj_bin.clone(),
            cwd: temp_dir.path().to_path_buf(),
            xdg_dir: xdg_dir.clone(),
            daemon_bin: dbin.clone(),
            state_dir: daemon_state_dir.clone(),
            cache_dir: daemon_cache_dir.clone(),
            active: true,
        })
    } else {
        None
    };

    // Seed project + initial baseline commit so all agents branch from a realistic tree.
    let seed_ws = temp_dir.path().join("seed-ws");
    run_jj_async(
        &jj_bin,
        temp_dir.path(),
        &xdg_dir,
        auth_token.as_deref(),
        daemon_bin
            .as_deref()
            .map(|b| (b, daemon_state_dir.as_path(), daemon_cache_dir.as_path())),
        &[
            "cc",
            "init",
            "--server",
            &target_url,
            "--project-id",
            &project_id,
            "--repo-id",
            &shared_repo_id,
            "--create",
            seed_ws.to_str().unwrap(),
        ],
    )
    .await?;
    fs::create_dir_all(seed_ws.join("src"))?;
    fs::write(
        seed_ws.join("src/common.rs"),
        "pub const BENCH_BASELINE: u64 = 1;\n",
    )?;
    run_jj_async(
        &jj_bin,
        &seed_ws,
        &xdg_dir,
        auth_token.as_deref(),
        daemon_bin
            .as_deref()
            .map(|b| (b, daemon_state_dir.as_path(), daemon_cache_dir.as_path())),
        &["commit", "-m", "benchmark baseline commit"],
    )
    .await?;

    // Phase 1: set up all N agent workspaces before starting the throughput clock.
    let shared_jj = Arc::new(jj_bin.clone());
    let shared_url = Arc::new(target_url.clone());
    let shared_proj = Arc::new(project_id.clone());
    let shared_repo = Arc::new(shared_repo_id.clone());
    let shared_xdg = Arc::new(xdg_dir.clone());
    let shared_auth = Arc::new(auth_token.clone());
    let shared_daemon = Arc::new(
        daemon_bin
            .clone()
            .map(|b| (b, daemon_state_dir.clone(), daemon_cache_dir.clone())),
    );

    let mut setup_set = JoinSet::new();
    for agent_idx in 0..opts.agents {
        let ws_dir = temp_dir.path().join(format!("agent-{agent_idx}"));
        let jj = Arc::clone(&shared_jj);
        let url = Arc::clone(&shared_url);
        let proj = Arc::clone(&shared_proj);
        let repo = Arc::clone(&shared_repo);
        let xdg = Arc::clone(&shared_xdg);
        let auth = Arc::clone(&shared_auth);
        let daemon_ctx = Arc::clone(&shared_daemon);
        let isolated = opts.isolated_repos;
        let parent_tmp = temp_dir.path().to_path_buf();

        setup_set.spawn(async move {
            let d_ref = daemon_ctx
                .as_ref()
                .as_ref()
                .map(|(b, s, c)| (b.as_path(), s.as_path(), c.as_path()));

            let agent_repo = if isolated {
                let r = format!("{repo}-a{agent_idx}");
                run_jj_async(
                    &jj,
                    &parent_tmp,
                    &xdg,
                    auth.as_deref(),
                    d_ref,
                    &[
                        "cc",
                        "init",
                        "--server",
                        &url,
                        "--project-id",
                        &proj,
                        "--repo-id",
                        &r,
                        "--create",
                        ws_dir.to_str().unwrap(),
                    ],
                )
                .await?;
                r
            } else {
                let ws_name = format!("agent-{agent_idx}");
                run_jj_async(
                    &jj,
                    &parent_tmp,
                    &xdg,
                    auth.as_deref(),
                    d_ref,
                    &[
                        "cc",
                        "clone",
                        "--server",
                        &url,
                        "--workspace-name",
                        &ws_name,
                        &repo,
                        ws_dir.to_str().unwrap(),
                    ],
                )
                .await?;
                (*repo).clone()
            };

            fs::create_dir_all(ws_dir.join("agents")).map_err(|e| e.to_string())?;
            Ok::<(usize, String, PathBuf), String>((agent_idx, agent_repo, ws_dir))
        });
    }

    let mut prepared_agents = Vec::with_capacity(opts.agents);
    while let Some(joined) = setup_set.join_next().await {
        let ready = joined.map_err(|e| format!("agent setup task panicked: {e}"))??;
        prepared_agents.push(ready);
    }

    // Phase 2: release all prepared agents behind a barrier so wall-clock throughput excludes
    // workspace init/clone setup.
    let barrier = Arc::new(Barrier::new(opts.agents));
    let commit_start = Arc::new(OnceLock::<Instant>::new());
    let mut join_set = JoinSet::new();

    for (agent_idx, agent_repo, ws_dir) in prepared_agents {
        let jj = Arc::clone(&shared_jj);
        let xdg = Arc::clone(&shared_xdg);
        let auth = Arc::clone(&shared_auth);
        let daemon_ctx = Arc::clone(&shared_daemon);
        let commits_k = opts.commits_per_agent;
        let barrier = Arc::clone(&barrier);
        let commit_start = Arc::clone(&commit_start);

        join_set.spawn(async move {
            let d_ref = daemon_ctx
                .as_ref()
                .as_ref()
                .map(|(b, s, c)| (b.as_path(), s.as_path(), c.as_path()));

            barrier.wait().await;
            let _ = commit_start.get_or_init(Instant::now);

            let mut latencies_ms = Vec::with_capacity(commits_k);
            let mut markers = Vec::with_capacity(commits_k);
            let agent_subdir = ws_dir.join("agents");
            for step in 0..commits_k {
                let file_path = agent_subdir.join(format!("agent_{agent_idx}_step_{step}.rs"));
                fs::write(
                    &file_path,
                    format!("pub fn agent_{agent_idx}_step_{step}() -> usize {{ {step} }}\n"),
                )
                .map_err(|e| e.to_string())?;
                let msg = format!("bench-commit:a{agent_idx}:s{step}");
                let t0 = Instant::now();
                run_jj_async(
                    &jj,
                    &ws_dir,
                    &xdg,
                    auth.as_deref(),
                    d_ref,
                    &["commit", "-m", &msg],
                )
                .await?;
                latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                markers.push(msg);
            }
            Ok::<(String, PathBuf, Vec<f64>, Vec<String>), String>((
                agent_repo,
                ws_dir,
                latencies_ms,
                markers,
            ))
        });
    }

    let mut all_latencies_ms = Vec::with_capacity(total_expected_commits);
    let mut agent_workspaces = Vec::with_capacity(opts.agents);
    let mut agent_results = Vec::with_capacity(opts.agents);
    while let Some(joined) = join_set.join_next().await {
        let (agent_repo, ws_dir, lats, markers) =
            joined.map_err(|e| format!("agent task panicked: {e}"))??;
        agent_workspaces.push(ws_dir.clone());
        all_latencies_ms.extend(lats);
        agent_results.push((agent_repo, ws_dir, markers));
    }
    let write_elapsed = commit_start.get().map(Instant::elapsed).unwrap_or_default();

    // Collect each agent's authored commit IDs and verify all commits per repo through fresh
    // direct workspaces (bypassing the local daemon cache) so both shared and isolated topologies
    // prove zero lost commits on the server.
    let mut expected_by_repo: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let mut collect_set = JoinSet::new();
    for (agent_repo, ws_dir, markers) in agent_results {
        let jj = Arc::clone(&shared_jj);
        let xdg = Arc::clone(&shared_xdg);
        let auth = Arc::clone(&shared_auth);
        let daemon_ctx = Arc::clone(&shared_daemon);
        collect_set.spawn(async move {
            let d_ref = daemon_ctx
                .as_ref()
                .as_ref()
                .map(|(b, s, c)| (b.as_path(), s.as_path(), c.as_path()));
            let log_out = run_jj_async(
                &jj,
                &ws_dir,
                &xdg,
                auth.as_deref(),
                d_ref,
                &[
                    "log",
                    "--no-graph",
                    "-r",
                    "all()",
                    "-T",
                    "commit_id ++ \"\\t\" ++ description.first_line() ++ \"\\n\"",
                ],
            )
            .await?;
            let collected = extract_agent_commit_ids(&log_out, &markers)?;
            Ok::<(String, Vec<(String, String)>), String>((agent_repo, collected))
        });
    }
    while let Some(joined) = collect_set.join_next().await {
        let (repo_id, pairs) =
            joined.map_err(|e| format!("commit collection task panicked: {e}"))??;
        expected_by_repo.entry(repo_id).or_default().extend(pairs);
    }

    let verify_root = temp_dir.path().join("verify");
    fs::create_dir_all(&verify_root)?;
    let mut verify_set = JoinSet::new();
    for (idx, (repo_id, expected_pairs)) in expected_by_repo.into_iter().enumerate() {
        let jj = Arc::clone(&shared_jj);
        let url = Arc::clone(&shared_url);
        let xdg = Arc::clone(&shared_xdg);
        let auth = Arc::clone(&shared_auth);
        let verify_parent = verify_root.clone();
        let verify_ws = verify_root.join(format!("repo-{idx}"));
        verify_set.spawn(async move {
            run_jj_async(
                &jj,
                &verify_parent,
                &xdg,
                auth.as_deref(),
                None,
                &[
                    "cc",
                    "clone",
                    "--server",
                    &url,
                    &repo_id,
                    verify_ws.to_str().unwrap(),
                ],
            )
            .await?;
            let log_out = run_jj_async(
                &jj,
                &verify_ws,
                &xdg,
                auth.as_deref(),
                None,
                &[
                    "log",
                    "--no-graph",
                    "-r",
                    "all()",
                    "-T",
                    "commit_id ++ \"\\t\" ++ description.first_line() ++ \"\\n\"",
                ],
            )
            .await?;
            verify_repo_commits(&repo_id, &log_out, &expected_pairs)
        });
    }

    let mut verified_count = 0usize;
    while let Some(joined) = verify_set.join_next().await {
        let count = joined.map_err(|e| format!("verification task panicked: {e}"))??;
        verified_count += count;
    }
    let verified_commits: Option<usize> = Some(verified_count);

    // Optional warm-read sweep when `--daemon` is enabled.
    let mut daemon_stats: Option<DaemonStatsJson> = None;
    if let Some(dbin) = &daemon_bin {
        for ws in agent_workspaces.iter().take(4) {
            for _ in 0..2 {
                run_jj_async(
                    &jj_bin,
                    ws,
                    &xdg_dir,
                    auth_token.as_deref(),
                    Some((dbin, &daemon_state_dir, &daemon_cache_dir)),
                    &[
                        "log",
                        "--no-graph",
                        "-r",
                        "all()",
                        "-T",
                        "commit_id ++ \"\\n\"",
                    ],
                )
                .await?;
                run_jj_async(
                    &jj_bin,
                    ws,
                    &xdg_dir,
                    auth_token.as_deref(),
                    Some((dbin, &daemon_state_dir, &daemon_cache_dir)),
                    &["diff", "-r", "@-"],
                )
                .await?;
            }
        }
        if run_jj_async(
            &jj_bin,
            temp_dir.path(),
            &xdg_dir,
            auth_token.as_deref(),
            Some((dbin, &daemon_state_dir, &daemon_cache_dir)),
            &["cc", "daemon", "stop"],
        )
        .await
        .is_ok()
            && let Some(guard) = &mut daemon_guard
        {
            guard.disarm();
        }
        if let Ok(raw) = fs::read_to_string(daemon_state_dir.join("daemon.stats.json")) {
            daemon_stats = serde_json::from_str(&raw).ok();
        }
    }

    all_latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let throughput = (total_expected_commits as f64) / write_elapsed.as_secs_f64().max(0.001);
    let p50 = percentile(&all_latencies_ms, 0.50);
    let p95 = percentile(&all_latencies_ms, 0.95);
    let p99 = percentile(&all_latencies_ms, 0.99);

    println!("==> Benchmark Results");
    println!(
        "    Commits verified:  {}",
        format_verification_summary(verified_commits, total_expected_commits)
    );
    println!("    Wall-clock time:   {:.2}s", write_elapsed.as_secs_f64());
    println!("    Throughput:        {throughput:.2} commits/s");
    println!("    Commit latency:    p50={p50:.1}ms  p95={p95:.1}ms  p99={p99:.1}ms");
    if let Some(st) = daemon_stats {
        let hits = st.l1_hits + st.l2_hits;
        let denom = (hits + st.misses).max(1);
        let pct = (hits as f64) * 100.0 / (denom as f64);
        println!(
            "    Daemon cache:      {} L1 + {} L2 hits ({pct:.1}% hit rate, {} coalesced, {} upstream RPCs)",
            st.l1_hits, st.l2_hits, st.singleflight_coalesced, st.upstream_rpcs
        );
    }

    Ok(())
}

fn format_verification_summary(verified: Option<usize>, expected: usize) -> String {
    match verified {
        Some(n) => format!(
            "{n}/{expected} ({} lost commits)",
            expected.saturating_sub(n)
        ),
        None => "not verified".to_string(),
    }
}

fn parse_commit_log_lines(log_out: &str) -> HashMap<&str, &str> {
    let mut by_desc = HashMap::new();
    for line in log_out.lines() {
        if let Some((commit_id, desc)) = line.split_once('\t') {
            let cid = commit_id.trim();
            let d = desc.trim();
            if !cid.is_empty() && !d.is_empty() {
                by_desc.insert(d, cid);
            }
        }
    }
    by_desc
}

fn extract_agent_commit_ids(
    log_out: &str,
    expected_markers: &[String],
) -> Result<Vec<(String, String)>, String> {
    let by_desc = parse_commit_log_lines(log_out);
    let mut pairs = Vec::with_capacity(expected_markers.len());
    for marker in expected_markers {
        let Some(&commit_id) = by_desc.get(marker.as_str()) else {
            return Err(format!(
                "agent workspace is missing authored commit '{marker}'"
            ));
        };
        pairs.push((commit_id.to_string(), marker.clone()));
    }
    Ok(pairs)
}

fn verify_repo_commits(
    repo_id: &str,
    log_out: &str,
    expected_pairs: &[(String, String)],
) -> Result<usize, String> {
    let present: HashSet<(&str, &str)> = log_out
        .lines()
        .filter_map(|line| {
            let (cid, desc) = line.split_once('\t')?;
            Some((cid.trim(), desc.trim()))
        })
        .collect();
    for (commit_id, marker) in expected_pairs {
        if !present.contains(&(commit_id.as_str(), marker.as_str())) {
            return Err(format!(
                "repository '{repo_id}' lost commit {commit_id} ('{marker}')"
            ));
        }
    }
    Ok(expected_pairs.len())
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn ensure_host_daemon_binary(root: &Path) -> Result<PathBuf, BoxError> {
    let exe_name = if cfg!(windows) {
        "jj-cc-daemon.exe"
    } else {
        "jj-cc-daemon"
    };
    let status = StdCommand::new("cargo")
        .current_dir(root)
        .args([
            "build",
            "--quiet",
            "-p",
            "jj-commit-cloud-daemon",
            "--bin",
            "jj-cc-daemon",
        ])
        .status()?;
    if !status.success() {
        return Err("`cargo build -p jj-commit-cloud-daemon` failed".into());
    }
    Ok(cargo_target_dir(root).join("debug").join(exe_name))
}

async fn spawn_ephemeral_sqlite_server(
    root: &Path,
    db_path: &Path,
) -> Result<EphemeralServer, BoxError> {
    let exe_name = if cfg!(windows) {
        "jj-cc-server.exe"
    } else {
        "jj-cc-server"
    };
    let status = StdCommand::new("cargo")
        .current_dir(root)
        .args([
            "build",
            "--quiet",
            "-p",
            "jj-commit-cloud-server",
            "--bin",
            "jj-cc-server",
        ])
        .status()?;
    if !status.success() {
        return Err("`cargo build -p jj-commit-cloud-server` failed".into());
    }
    let server_bin = cargo_target_dir(root).join("debug").join(exe_name);

    let mut child = TokioCommand::new(&server_bin)
        .kill_on_drop(true)
        .arg("--port=0")
        .arg("--store-type=sqlite")
        .arg(format!("--sqlite-path={}", db_path.display()))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    let stdout = child
        .stdout
        .take()
        .ok_or("failed to capture jj-cc-server stdout")?;
    let mut lines = BufReader::new(stdout).lines();
    let deadline = tokio::time::sleep(Duration::from_secs(10));
    tokio::pin!(deadline);

    let bound_addr = loop {
        tokio::select! {
            line = lines.next_line() => {
                match line? {
                    Some(text) => {
                        if let Some((_, rest)) = text.split_once("listening on ") {
                            let addr = rest.trim().to_string();
                            if !addr.is_empty() {
                                break addr;
                            }
                        }
                    }
                    None => return Err("ephemeral jj-cc-server exited before binding".into()),
                }
            }
            _ = &mut deadline => return Err("timed out waiting for ephemeral jj-cc-server".into()),
        }
    };
    tokio::spawn(async move { while let Ok(Some(_)) = lines.next_line().await {} });

    Ok(EphemeralServer {
        child,
        url: format!("http://{bound_addr}"),
    })
}

async fn run_jj_async(
    jj_bin: &Path,
    cwd: &Path,
    xdg_dir: &Path,
    auth_token: Option<&str>,
    daemon_ctx: Option<(&Path, &Path, &Path)>,
    args: &[&str],
) -> Result<String, String> {
    let mut cmd = TokioCommand::new(jj_bin);
    cmd.kill_on_drop(true)
        .current_dir(cwd)
        .arg("--no-pager")
        .args(args)
        .env("XDG_CONFIG_HOME", xdg_dir)
        .env("JJ_USER", "Benchmark Agent")
        .env("JJ_EMAIL", "bench@example.com");

    if let Some(tok) = auth_token {
        cmd.env("JJ_CC_AUTH_TOKEN", tok);
    }
    if let Some((daemon_bin, state_dir, cache_dir)) = daemon_ctx {
        cmd.env("JJ_CC_USE_DAEMON", "1")
            .env("JJ_CC_DAEMON_BIN", daemon_bin)
            .env("JJ_CC_DAEMON_DIR", state_dir)
            .env("JJ_CC_CACHE_DIR", cache_dir);
    } else {
        cmd.env("JJ_CC_USE_DAEMON", "0");
    }

    let out = cmd
        .output()
        .await
        .map_err(|e| format!("failed to spawn jj {args:?}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "jj {args:?} failed ({}):\nstdout: {}\nstderr: {}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_line_matching_rejects_prefix_collision_between_s1_and_s10() {
        let log_out = "\
aaaaaaaa11111111\tbench-commit:a1:s0\n\
bbbbbbbb22222222\tbench-commit:a1:s10\n";
        let markers = vec![
            "bench-commit:a1:s0".to_string(),
            "bench-commit:a1:s1".to_string(),
        ];
        let err = extract_agent_commit_ids(log_out, &markers).unwrap_err();
        assert!(
            err.contains("bench-commit:a1:s1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn verify_repo_commits_checks_both_commit_id_and_exact_marker() {
        let log_out = "\
aaaaaaaa11111111\tbench-commit:a0:s0\n\
cccccccc33333333\tbench-commit:a0:s1\n";
        let pairs = vec![
            (
                "aaaaaaaa11111111".to_string(),
                "bench-commit:a0:s0".to_string(),
            ),
            (
                "cccccccc33333333".to_string(),
                "bench-commit:a0:s1".to_string(),
            ),
        ];
        assert_eq!(verify_repo_commits("repo-a0", log_out, &pairs).unwrap(), 2);

        let wrong_id = vec![(
            "ffffffff99999999".to_string(),
            "bench-commit:a0:s1".to_string(),
        )];
        assert!(verify_repo_commits("repo-a0", log_out, &wrong_id).is_err());
    }

    #[test]
    fn verification_summary_prints_not_verified_when_skipped() {
        assert_eq!(format_verification_summary(None, 36), "not verified");
        assert_eq!(
            format_verification_summary(Some(36), 36),
            "36/36 (0 lost commits)"
        );
        assert_eq!(
            format_verification_summary(Some(35), 36),
            "35/36 (1 lost commits)"
        );
    }
}
