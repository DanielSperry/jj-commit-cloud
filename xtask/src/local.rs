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
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use tonic::metadata::MetadataValue;
use tonic::transport::{Channel, ClientTlsConfig};
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_client::HealthClient;

use crate::gcp::mint_gcloud_identity_token;
use crate::state::{AuthMode, BoxError, EnvState, ManagedProcess, cargo_target_dir, save_state};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum LocalDriver {
    Auto,
    Container,
    Native,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ContainerRuntimeChoice {
    Auto,
    Docker,
    Podman,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalMode {
    Memory,
    Sqlite,
    Spanner,
    Multiregion,
}

impl LocalMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::Sqlite => "sqlite",
            Self::Spanner => "spanner",
            Self::Multiregion => "multiregion",
        }
    }
}

pub fn detect_container_runtime(choice: ContainerRuntimeChoice) -> Result<String, BoxError> {
    let candidates: &[&str] = match choice {
        ContainerRuntimeChoice::Docker => &["docker"],
        ContainerRuntimeChoice::Podman => &["podman"],
        ContainerRuntimeChoice::Auto => &["docker", "podman"],
    };
    for bin in candidates {
        if Command::new(bin)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            return Ok((*bin).to_string());
        }
    }
    Err("neither `docker` nor `podman` was found on PATH".into())
}

pub fn ensure_server_container_image(
    root: &Path,
    runtime: &str,
    force_rebuild: bool,
) -> Result<(), BoxError> {
    let image_exists = Command::new(runtime)
        .args(["image", "inspect", "localhost/jj-cc-server:local"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());

    if !image_exists || force_rebuild {
        println!("==> Building container image localhost/jj-cc-server:local via {runtime}...");
        let status = Command::new(runtime)
            .current_dir(root)
            .args([
                "build",
                "--target",
                "server",
                "-t",
                "localhost/jj-cc-server:local",
                "-t",
                "jj-cc-server:local",
                ".",
            ])
            .status()?;
        if !status.success() {
            return Err(format!("`{runtime} build` failed").into());
        }
    }
    Ok(())
}

pub fn build_and_export_binaries(
    root: &Path,
    container: bool,
    export_bin: bool,
    runtime_choice: ContainerRuntimeChoice,
) -> Result<(), BoxError> {
    if container {
        let rt = detect_container_runtime(runtime_choice)?;
        ensure_server_container_image(root, &rt, true)?;
        println!("==> Building container image jj-cc-cli:local via {rt}...");
        let status = Command::new(&rt)
            .current_dir(root)
            .args(["build", "--target", "cli", "-t", "jj-cc-cli:local", "."])
            .status()?;
        if !status.success() {
            return Err(format!("`{rt} build --target cli` failed").into());
        }
        if export_bin {
            let out_dir = root.join("bin");
            fs::create_dir_all(&out_dir)?;
            let status = Command::new(&rt)
                .current_dir(root)
                .args([
                    "build",
                    "--target",
                    "bins",
                    "--output",
                    &format!("type=local,dest={}", out_dir.display()),
                    ".",
                ])
                .status()?;
            if !status.success() {
                return Err("container binary export failed".into());
            }
            println!("==> Exported binaries to {}", out_dir.display());
        }
    } else {
        println!("==> Building workspace binaries with Cargo...");
        let status = Command::new("cargo")
            .current_dir(root)
            .args([
                "build",
                "-p",
                "jj-commit-cloud-server",
                "--bin",
                "jj-cc-server",
                "-p",
                "jj-commit-cloud-daemon",
                "--bin",
                "jj-cc-daemon",
                "-p",
                "cli",
                "--bin",
                "jj",
            ])
            .status()?;
        if !status.success() {
            return Err("`cargo build` failed".into());
        }
        if export_bin {
            let out_dir = root.join("bin");
            fs::create_dir_all(&out_dir)?;
            for name in ["jj-cc-server", "jj-cc-daemon", "jj"] {
                let exe = if cfg!(windows) {
                    format!("{name}.exe")
                } else {
                    name.to_string()
                };
                fs::copy(
                    cargo_target_dir(root).join("debug").join(&exe),
                    out_dir.join(&exe),
                )?;
            }
            println!("==> Copied binaries to {}", out_dir.display());
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn up_local(
    root: &Path,
    env_name: &str,
    mode: LocalMode,
    driver: LocalDriver,
    runtime_choice: ContainerRuntimeChoice,
    port: u16,
    peer_port: u16,
    rebuild_image: bool,
) -> Result<EnvState, BoxError> {
    let use_container = match driver {
        LocalDriver::Container => true,
        LocalDriver::Native => false,
        LocalDriver::Auto => detect_container_runtime(runtime_choice).is_ok(),
    };

    if use_container {
        let rt = detect_container_runtime(runtime_choice)?;
        up_local_containers(root, env_name, mode, &rt, port, peer_port, rebuild_image).await
    } else {
        up_local_native(root, env_name, mode, port, peer_port).await
    }
}

async fn up_local_containers(
    root: &Path,
    env_name: &str,
    mode: LocalMode,
    rt: &str,
    port: u16,
    peer_port: u16,
    rebuild_image: bool,
) -> Result<EnvState, BoxError> {
    let primary_port = if mode == LocalMode::Multiregion && port == 8080 {
        8081
    } else {
        port
    };
    let primary_url = format!("http://127.0.0.1:{primary_port}");
    let peer_url =
        (mode == LocalMode::Multiregion).then(|| format!("http://127.0.0.1:{peer_port}"));

    let prefix = format!("jj-cc-{env_name}");
    let net_name = format!("{prefix}-net");
    let mut state = EnvState::new(
        env_name,
        mode.as_str(),
        rt,
        &primary_url,
        peer_url,
        AuthMode::None,
    );
    state.resources.network = Some(net_name.clone());
    save_state(root, &state)?;

    ensure_server_container_image(root, rt, rebuild_image)?;

    let _ = Command::new(rt)
        .args(["network", "create", &net_name])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    match mode {
        LocalMode::Memory => {
            let cname = format!("{prefix}-server");
            state.resources.containers.push(cname.clone());
            save_state(root, &state)?;
            run_detached_container(
                rt,
                &cname,
                &net_name,
                &[(primary_port, 8080)],
                &[("STORE_TYPE", "memory"), ("BLOB_STORE", "memory")],
                &[],
                "localhost/jj-cc-server:local",
                &[],
            )?;
        }
        LocalMode::Sqlite => {
            let vol = format!("{prefix}-sqlite-data");
            let cname = format!("{prefix}-server");
            state.resources.volumes.push(vol.clone());
            state.resources.containers.push(cname.clone());
            save_state(root, &state)?;
            let _ = Command::new(rt)
                .args(["volume", "create", &vol])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            run_detached_container(
                rt,
                &cname,
                &net_name,
                &[(primary_port, 8080)],
                &[
                    ("STORE_TYPE", "sqlite"),
                    ("BLOB_STORE", "local"),
                    ("SQLITE_PATH", "/data/commit_cloud.db"),
                    ("BLOB_PATH", "/data/blobs"),
                ],
                &[(&vol, "/data")],
                "localhost/jj-cc-server:local",
                &[],
            )?;
        }
        LocalMode::Spanner | LocalMode::Multiregion => {
            let sp_name = format!("{prefix}-spanner");
            let gcs_name = format!("{prefix}-gcs");
            state.resources.containers.push(sp_name.clone());
            state.resources.containers.push(gcs_name.clone());
            save_state(root, &state)?;

            run_detached_container(
                rt,
                &sp_name,
                &net_name,
                &[],
                &[],
                &[],
                "gcr.io/cloud-spanner-emulator/emulator:latest",
                &[],
            )?;
            run_detached_container(
                rt,
                &gcs_name,
                &net_name,
                &[],
                &[],
                &[],
                "docker.io/fsouza/fake-gcs-server:latest",
                &[
                    "-scheme",
                    "http",
                    "-port",
                    "4443",
                    "-external-url",
                    &format!("http://{gcs_name}:4443"),
                    "-backend",
                    "memory",
                ],
            )?;

            let sp_grpc = format!("{sp_name}:9010");
            let sp_rest = format!("http://{sp_name}:9020");
            let gcs_ep = format!("http://{gcs_name}:4443");

            if mode == LocalMode::Spanner {
                let cname = format!("{prefix}-server");
                state.resources.containers.push(cname.clone());
                save_state(root, &state)?;
                run_detached_container(
                    rt,
                    &cname,
                    &net_name,
                    &[(primary_port, 8080)],
                    &[
                        ("STORE_TYPE", "spanner"),
                        ("SPANNER_EMULATOR_HOST", &sp_grpc),
                        ("SPANNER_EMULATOR_REST_HOST", &sp_rest),
                        (
                            "SPANNER_DATABASE",
                            "projects/local-project/instances/local-instance/databases/commit-cloud",
                        ),
                        ("SPANNER_AUTO_MIGRATE", "true"),
                        ("BLOB_STORE", "gcs"),
                        ("STORAGE_EMULATOR_HOST", &gcs_ep),
                        ("GCS_BUCKET", "gs://blobs-uc1"),
                    ],
                    &[],
                    "localhost/jj-cc-server:local",
                    &[],
                )?;
            } else {
                let us_name = format!("{prefix}-server-us");
                let eu_name = format!("{prefix}-server-eu");
                state.resources.containers.push(us_name.clone());
                state.resources.containers.push(eu_name.clone());
                save_state(root, &state)?;

                // Each region lists the other's bucket as a peer, so a blob written through
                // either server can be read through the other.
                run_detached_container(
                    rt,
                    &us_name,
                    &net_name,
                    &[(primary_port, 8080)],
                    &[
                        ("STORE_TYPE", "spanner"),
                        ("SPANNER_EMULATOR_HOST", &sp_grpc),
                        ("SPANNER_EMULATOR_REST_HOST", &sp_rest),
                        (
                            "SPANNER_DATABASE",
                            "projects/local-project/instances/local-instance/databases/commit-cloud",
                        ),
                        ("SPANNER_AUTO_MIGRATE", "true"),
                        ("BLOB_STORE", "gcs"),
                        ("STORAGE_EMULATOR_HOST", &gcs_ep),
                        ("GCS_BUCKET", "gs://blobs-uc1"),
                        ("GCS_PEER_BUCKETS", "gs://blobs-ew1"),
                    ],
                    &[],
                    "localhost/jj-cc-server:local",
                    &[],
                )?;
                run_detached_container(
                    rt,
                    &eu_name,
                    &net_name,
                    &[(peer_port, 8080)],
                    &[
                        ("STORE_TYPE", "spanner"),
                        ("SPANNER_EMULATOR_HOST", &sp_grpc),
                        ("SPANNER_EMULATOR_REST_HOST", &sp_rest),
                        (
                            "SPANNER_DATABASE",
                            "projects/local-project/instances/local-instance/databases/commit-cloud",
                        ),
                        ("SPANNER_AUTO_MIGRATE", "true"),
                        ("BLOB_STORE", "gcs"),
                        ("STORAGE_EMULATOR_HOST", &gcs_ep),
                        ("GCS_BUCKET", "gs://blobs-ew1"),
                        ("GCS_PEER_BUCKETS", "gs://blobs-uc1"),
                    ],
                    &[],
                    "localhost/jj-cc-server:local",
                    &[],
                )?;
            }
        }
    }

    wait_until_healthy(
        &state.endpoints.primary,
        state.auth,
        None,
        Duration::from_secs(30),
    )
    .await?;
    if let Some(peer) = &state.endpoints.peer {
        wait_until_healthy(peer, state.auth, None, Duration::from_secs(30)).await?;
    }
    Ok(state)
}

#[allow(clippy::too_many_arguments)]
fn run_detached_container(
    rt: &str,
    name: &str,
    net_name: &str,
    ports: &[(u16, u16)],
    envs: &[(&str, &str)],
    volumes: &[(&str, &str)],
    image: &str,
    extra_args: &[&str],
) -> Result<(), BoxError> {
    let _ = Command::new(rt)
        .args(["rm", "-f", name])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    let mut cmd = Command::new(rt);
    cmd.args(["run", "-d", "--name", name, "--network", net_name]);
    for (host_p, cont_p) in ports {
        cmd.args(["-p", &format!("127.0.0.1:{host_p}:{cont_p}")]);
    }
    for (k, v) in envs {
        cmd.args(["-e", &format!("{k}={v}")]);
    }
    for (vol, target) in volumes {
        cmd.args(["-v", &format!("{vol}:{target}")]);
    }
    cmd.arg(image).args(extra_args);
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(format!(
            "`{rt} run {name}` failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(())
}

async fn up_local_native(
    root: &Path,
    env_name: &str,
    mode: LocalMode,
    port: u16,
    peer_port: u16,
) -> Result<EnvState, BoxError> {
    let exe_name = if cfg!(windows) {
        "jj-cc-server.exe"
    } else {
        "jj-cc-server"
    };
    build_and_export_binaries(root, false, false, ContainerRuntimeChoice::Auto)?;
    let server_bin = cargo_target_dir(root).join("debug").join(exe_name);

    let primary_port = if mode == LocalMode::Multiregion && port == 8080 {
        8081
    } else {
        port
    };
    ensure_loopback_port_free(primary_port)?;
    if mode == LocalMode::Multiregion {
        ensure_loopback_port_free(peer_port)?;
    }
    let primary_url = format!("http://127.0.0.1:{primary_port}");
    let peer_url =
        (mode == LocalMode::Multiregion).then(|| format!("http://127.0.0.1:{peer_port}"));

    let data_dir = root.join(".jj-cc").join(format!("data-{env_name}"));
    fs::create_dir_all(&data_dir)?;
    let log_path = root.join(".jj-cc").join(format!("{env_name}.log"));
    let peer_log_path = root.join(".jj-cc").join(format!("{env_name}-peer.log"));

    let mut state = EnvState::new(
        env_name,
        mode.as_str(),
        "native",
        &primary_url,
        peer_url,
        AuthMode::None,
    );
    state
        .resources
        .data_dirs
        .push(data_dir.display().to_string());
    save_state(root, &state)?;
    println!(
        "==> Starting native servers (log: {})...",
        log_path.display()
    );

    let mut children = Vec::new();
    match mode {
        LocalMode::Memory => {
            let child = spawn_bg_child(
                &server_bin,
                &[&format!("--port={primary_port}"), "--store-type=memory"],
                &[],
                &log_path,
            )?;
            state
                .resources
                .processes
                .push(ManagedProcess::new(child.id(), SERVER_PROCESS_MARKER));
            children.push(child);
            save_state(root, &state)?;
        }
        LocalMode::Sqlite => {
            let db = data_dir.join("commit_cloud.db");
            let blobs = data_dir.join("blobs");
            let child = spawn_bg_child(
                &server_bin,
                &[
                    &format!("--port={primary_port}"),
                    "--store-type=sqlite",
                    &format!("--sqlite-path={}", db.display()),
                    &format!("--blob-path={}", blobs.display()),
                ],
                &[],
                &log_path,
            )?;
            state
                .resources
                .processes
                .push(ManagedProcess::new(child.id(), SERVER_PROCESS_MARKER));
            children.push(child);
            save_state(root, &state)?;
        }
        LocalMode::Spanner | LocalMode::Multiregion => {
            let emu_log = root
                .join(".jj-cc")
                .join(format!("{env_name}-spanner-emulator.log"));
            let (sp_grpc, sp_rest, emu_pid) = ensure_native_spanner_emulator(&emu_log)?;
            if let Some(pid) = emu_pid {
                state
                    .resources
                    .processes
                    .push(ManagedProcess::new(pid, "gateway_main"));
                save_state(root, &state)?;
            }
            let uc1_dir = data_dir.join("blobs-uc1");
            fs::create_dir_all(&uc1_dir)?;
            let ew1_dir = data_dir.join("blobs-ew1");
            let us_port = format!("--port={primary_port}");
            let us_blob_path = format!("--blob-path={}", uc1_dir.display());
            let us_peer = format!(
                "--local-peer-blob-dirs=gs://blobs-ew1={}",
                ew1_dir.display()
            );
            let mut us_args: Vec<&str> = vec![
                &us_port,
                "--store-type=spanner",
                "--spanner-database=projects/local-project/instances/local-instance/databases/commit-cloud",
                "--spanner-auto-migrate",
                "--blob-store=local",
                &us_blob_path,
                "--blob-location-id=gs://blobs-uc1",
            ];
            if mode == LocalMode::Multiregion {
                // Each region lists the other's directory as a peer, so a blob written through
                // either server can be read through the other.
                fs::create_dir_all(&ew1_dir)?;
                us_args.push(&us_peer);
            }
            let child_us = spawn_bg_child(
                &server_bin,
                &us_args,
                &[
                    ("SPANNER_EMULATOR_HOST", &sp_grpc),
                    ("SPANNER_EMULATOR_REST_HOST", &sp_rest),
                ],
                &log_path,
            )?;
            state
                .resources
                .processes
                .push(ManagedProcess::new(child_us.id(), SERVER_PROCESS_MARKER));
            children.push(child_us);
            save_state(root, &state)?;

            if mode == LocalMode::Multiregion {
                let child_eu = spawn_bg_child(
                    &server_bin,
                    &[
                        &format!("--port={peer_port}"),
                        "--store-type=spanner",
                        "--spanner-database=projects/local-project/instances/local-instance/databases/commit-cloud",
                        "--spanner-auto-migrate",
                        "--blob-store=local",
                        &format!("--blob-path={}", ew1_dir.display()),
                        "--blob-location-id=gs://blobs-ew1",
                        &format!(
                            "--local-peer-blob-dirs=gs://blobs-uc1={}",
                            uc1_dir.display()
                        ),
                    ],
                    &[
                        ("SPANNER_EMULATOR_HOST", &sp_grpc),
                        ("SPANNER_EMULATOR_REST_HOST", &sp_rest),
                    ],
                    &peer_log_path,
                )?;
                state
                    .resources
                    .processes
                    .push(ManagedProcess::new(child_eu.id(), SERVER_PROCESS_MARKER));
                children.push(child_eu);
                save_state(root, &state)?;
            }
        }
    }

    wait_until_healthy_with_children(
        &state.endpoints.primary,
        state.auth,
        None,
        Duration::from_secs(20),
        &mut children,
        &log_path,
    )
    .await?;
    if let Some(peer) = &state.endpoints.peer {
        wait_until_healthy_with_children(
            peer,
            state.auth,
            None,
            Duration::from_secs(20),
            &mut children,
            &peer_log_path,
        )
        .await?;
    }
    Ok(state)
}

fn ensure_loopback_port_free(port: u16) -> Result<(), BoxError> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(100)).is_ok() {
        return Err(format!(
            "127.0.0.1:{port} is already accepting connections; stop the existing server or pass `--port`"
        )
        .into());
    }
    Ok(())
}

fn ensure_native_spanner_emulator(
    log_path: &Path,
) -> Result<(String, String, Option<u32>), BoxError> {
    if let Ok(existing) = std::env::var("SPANNER_EMULATOR_HOST")
        && !existing.trim().is_empty()
    {
        let rest = std::env::var("SPANNER_EMULATOR_REST_HOST")
            .unwrap_or_else(|_| "http://127.0.0.1:9020".to_string());
        return Ok((existing, rest, None));
    }

    let gw = PathBuf::from("/google/bin/releases/spanner/cloud/emulator/daily/gateway_main");
    let grpc = PathBuf::from("/google/bin/releases/spanner/cloud/emulator/daily/emulator_main");
    if gw.exists() && grpc.exists() {
        let pid = spawn_bg(
            &gw,
            &[
                &format!("-grpc_binary={}", grpc.display()),
                "-hostname=127.0.0.1",
                "-grpc_port=9010",
                "-http_port=9020",
                "-disable_switch_user_and_group=true",
            ],
            &[],
            log_path,
        )?;
        return Ok((
            "127.0.0.1:9010".to_string(),
            "http://127.0.0.1:9020".to_string(),
            Some(pid),
        ));
    }
    Err(
        "native Spanner mode requires SPANNER_EMULATOR_HOST or a container runtime (`docker`/`podman`)"
            .into(),
    )
}

/// Starts `bin` in the background, appending its stdout and stderr to `log_path` so startup
/// failures stay diagnosable after `up` returns.
pub fn spawn_bg(
    bin: &Path,
    args: &[&str],
    envs: &[(&str, &str)],
    log_path: &Path,
) -> Result<u32, BoxError> {
    Ok(spawn_bg_child(bin, args, envs, log_path)?.id())
}

fn spawn_bg_child(
    bin: &Path,
    args: &[&str],
    envs: &[(&str, &str)],
    log_path: &Path,
) -> Result<std::process::Child, BoxError> {
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let mut cmd = Command::new(bin);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    Ok(cmd.spawn()?)
}

fn remove_container_resource(rt: &str, args: &[&str]) -> Result<(), String> {
    let out = Command::new(rt)
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    let lower = stderr.to_ascii_lowercase();
    if out.status.success() || lower.contains("no such") || lower.contains("not found") {
        Ok(())
    } else {
        Err(stderr.trim().to_string())
    }
}

pub fn down_local(state: &EnvState, purge: bool) -> Result<(), BoxError> {
    let mut failures = Vec::new();
    if state.runtime == "native" {
        failures.extend(stop_managed_processes(state));
        if purge && failures.is_empty() {
            for d in &state.resources.data_dirs {
                if Path::new(d).exists()
                    && let Err(e) = fs::remove_dir_all(d)
                {
                    failures.push(format!("data dir {d}: {e}"));
                }
            }
        }
    } else {
        let rt = &state.runtime;
        for c in state.resources.containers.iter().rev() {
            if let Err(e) = remove_container_resource(rt, &["rm", "-f", c]) {
                failures.push(format!("container {c}: {e}"));
            }
        }
        if let Some(net) = &state.resources.network
            && let Err(e) = remove_container_resource(rt, &["network", "rm", "-f", net])
        {
            failures.push(format!("network {net}: {e}"));
        }
        if purge {
            for v in &state.resources.volumes {
                if let Err(e) = remove_container_resource(rt, &["volume", "rm", "-f", v]) {
                    failures.push(format!("volume {v}: {e}"));
                }
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "could not tear down local resources: {}",
            failures.join("; ")
        )
        .into())
    }
}

fn kill_pid(pid: u32) -> bool {
    #[cfg(windows)]
    let status = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    #[cfg(not(windows))]
    let status = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    status.is_ok_and(|s| s.success())
}

/// Command-line marker of native jj-cc-server processes.
pub const SERVER_PROCESS_MARKER: &str = "jj-cc-server";

/// Markers accepted for bare PIDs recorded by earlier xtask versions.
const LEGACY_PROCESS_MARKERS: &[&str] = &[
    SERVER_PROCESS_MARKER,
    "gateway_main",
    "start-iap-tunnel",
    "port-forward",
];

/// Stops the background processes recorded in `state`, newest first. A PID is signaled only
/// while its command line still contains the marker recorded when xtask started it, so PIDs
/// that exited or were reused by another program after a crash or reboot are left alone.
/// Returns one message per process that is still running.
pub fn stop_managed_processes(state: &EnvState) -> Vec<String> {
    let tracked = state
        .resources
        .processes
        .iter()
        .map(|p| (p.pid, vec![p.command_marker.as_str()]));
    let legacy = state
        .resources
        .pids
        .iter()
        .map(|&pid| (pid, LEGACY_PROCESS_MARKERS.to_vec()));
    let all: Vec<_> = tracked.chain(legacy).collect();
    let mut failures = Vec::new();
    for (pid, markers) in all.iter().rev() {
        if let Err(e) = stop_process(*pid, markers) {
            failures.push(format!("PID {pid}: {e}"));
        }
    }
    failures
}

fn stop_process(pid: u32, markers: &[&str]) -> Result<(), String> {
    let Some(cmdline) = process_command_line(pid)? else {
        return Ok(());
    };
    if !markers.iter().any(|m| cmdline.contains(m)) {
        println!(
            "    Leaving PID {pid} alone: it now runs `{cmdline}`, not a process xtask started"
        );
        return Ok(());
    }
    if kill_pid(pid) || process_command_line(pid)?.is_none() {
        Ok(())
    } else {
        Err(format!("failed to stop `{cmdline}`"))
    }
}

/// Returns the command line of `pid`, or `None` when no such process is running.
fn process_command_line(pid: u32) -> Result<Option<String>, String> {
    #[cfg(target_os = "linux")]
    let cmdline = match fs::read(format!("/proc/{pid}/cmdline")) {
        Ok(raw) => String::from_utf8_lossy(&raw).replace('\0', " "),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("cannot read /proc/{pid}/cmdline: {e}")),
    };
    #[cfg(all(unix, not(target_os = "linux")))]
    let cmdline = {
        let out = Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "command="])
            .output()
            .map_err(|e| format!("cannot run `ps`: {e}"))?;
        // `ps -p` exits non-zero when no process has that PID.
        if !out.status.success() {
            return Ok(None);
        }
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    #[cfg(windows)]
    let cmdline = {
        let query =
            format!("(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine");
        let out = Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &query])
            .output()
            .map_err(|e| format!("cannot run `powershell`: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "cannot inspect PID {pid}: {}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let cmdline = cmdline.trim().to_string();
    // Zombies and kernel threads have empty command lines; neither is a process to stop.
    Ok((!cmdline.is_empty()).then_some(cmdline))
}

/// Returns the bearer token to send: the token `up` configured on the server, else
/// `JJ_CC_AUTH_TOKEN`, else (Cloud Run) a fresh `gcloud` identity token.
pub fn resolve_bearer_token(
    auth: AuthMode,
    server_token: Option<&str>,
) -> Result<Option<String>, BoxError> {
    if let Some(token) = server_token {
        return Ok(Some(token.to_string()));
    }
    Ok(match auth {
        AuthMode::None => std::env::var("JJ_CC_AUTH_TOKEN").ok(),
        AuthMode::GcloudIdentityToken => match std::env::var("JJ_CC_AUTH_TOKEN") {
            Ok(t) if !t.trim().is_empty() => Some(t),
            _ => Some(mint_gcloud_identity_token()?),
        },
    })
}

pub async fn check_grpc_health(
    url: &str,
    auth: AuthMode,
    server_token: Option<&str>,
) -> Result<bool, BoxError> {
    let mut ep = Channel::from_shared(url.to_string())?
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(5));
    if url.starts_with("https://") {
        ep = ep.tls_config(ClientTlsConfig::new().with_native_roots())?;
    }
    let channel = ep.connect().await?;

    let token = resolve_bearer_token(auth, server_token)?;

    let mut req = tonic::Request::new(HealthCheckRequest {
        service: String::new(),
    });
    if let Some(tok) = token {
        let header_val: MetadataValue<_> = format!("Bearer {}", tok.trim()).parse()?;
        req.metadata_mut().insert("authorization", header_val);
    }

    let mut client = HealthClient::new(channel);
    let resp = client.check(req).await?;
    Ok(resp.into_inner().status == ServingStatus::Serving as i32)
}

pub async fn wait_until_healthy(
    url: &str,
    auth: AuthMode,
    server_token: Option<&str>,
    timeout: Duration,
) -> Result<(), BoxError> {
    let deadline = Instant::now() + timeout;
    let mut last_err = String::new();
    while Instant::now() < deadline {
        match check_grpc_health(url, auth, server_token).await {
            Ok(true) => return Ok(()),
            Ok(false) => last_err = "service reported non-SERVING status".to_string(),
            Err(e) => last_err = e.to_string(),
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    Err(format!("timed out waiting for gRPC health at {url}: {last_err}").into())
}

async fn wait_until_healthy_with_children(
    url: &str,
    auth: AuthMode,
    server_token: Option<&str>,
    timeout: Duration,
    children: &mut [std::process::Child],
    log_path: &Path,
) -> Result<(), BoxError> {
    let deadline = Instant::now() + timeout;
    let mut last_err = String::new();
    while Instant::now() < deadline {
        for child in children.iter_mut() {
            if let Some(status) = child.try_wait()? {
                return Err(format!(
                    "native jj-cc-server (PID {}) exited during startup with {status} (log: {})",
                    child.id(),
                    log_path.display()
                )
                .into());
            }
        }
        match check_grpc_health(url, auth, server_token).await {
            Ok(true) => return Ok(()),
            Ok(false) => last_err = "service reported non-SERVING status".to_string(),
            Err(e) => last_err = e.to_string(),
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    Err(format!(
        "timed out waiting for gRPC health at {url}: {last_err} (log: {})",
        log_path.display()
    )
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_managed_processes_skips_unrelated_live_process() {
        let self_pid = std::process::id();
        let mut st = EnvState::new(
            "default",
            "memory",
            "native",
            "http://127.0.0.1:8080",
            None,
            AuthMode::None,
        );
        st.resources.processes.push(ManagedProcess::new(
            self_pid,
            "jj-cc-server-marker-never-matches-unit-test",
        ));
        st.resources.pids.push(self_pid);
        let failures = stop_managed_processes(&st);
        assert!(
            failures.is_empty(),
            "expected no failures, got {failures:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn spawn_bg_logs_output_and_stop_managed_processes_terminates_matching_child() {
        let tmp = tempfile::tempdir().unwrap();
        let log_path = tmp.path().join(".jj-cc").join("dev.log");
        let mut child = spawn_bg_child(
            Path::new("sh"),
            &["-c", "echo native-server-booted; exec sleep 30"],
            &[],
            &log_path,
        )
        .unwrap();
        let pid = child.id();

        for _ in 0..200 {
            let logged = fs::read_to_string(&log_path).unwrap_or_default();
            let cmdline = process_command_line(pid).unwrap().unwrap_or_default();
            if logged.contains("native-server-booted") && cmdline.contains("sleep") {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let logged = fs::read_to_string(&log_path).unwrap();
        assert!(
            logged.contains("native-server-booted"),
            "unexpected log: {logged:?}"
        );

        let mut st = EnvState::new(
            "dev",
            "memory",
            "native",
            "http://127.0.0.1:8080",
            None,
            AuthMode::None,
        );
        st.resources
            .processes
            .push(ManagedProcess::new(pid, "sleep"));
        let failures = stop_managed_processes(&st);
        assert!(failures.is_empty(), "expected clean stop, got {failures:?}");
        let exit = child.wait().unwrap();
        assert!(!exit.success());
    }

    #[test]
    fn resolve_bearer_token_prefers_recorded_server_token() {
        let tok = resolve_bearer_token(AuthMode::None, Some("generated-secret")).unwrap();
        assert_eq!(tok.as_deref(), Some("generated-secret"));
    }
}
