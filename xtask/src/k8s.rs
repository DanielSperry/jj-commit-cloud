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
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::local::{
    ContainerRuntimeChoice, detect_container_runtime, ensure_server_container_image,
    wait_until_healthy,
};
use crate::state::{AuthMode, BoxError, EnvState, save_state};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum K8sMode {
    Memory,
    Sqlite,
    Spanner,
    Multiregion,
}

impl K8sMode {
    pub fn mode_name(self) -> &'static str {
        match self {
            Self::Memory => "k8s-memory",
            Self::Sqlite => "k8s-sqlite",
            Self::Spanner => "k8s-spanner",
            Self::Multiregion => "k8s-multiregion",
        }
    }

    pub fn manifest_rel_path(self) -> &'static str {
        match self {
            Self::Memory => "deploy/k8s/memory.yaml",
            Self::Sqlite => "deploy/k8s/sqlite.yaml",
            Self::Spanner => "deploy/k8s/spanner.yaml",
            Self::Multiregion => "deploy/k8s/multiregion.yaml",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum K8sProvider {
    Auto,
    Kind,
    PodmanKube,
}

fn has_tool(bin: &str) -> bool {
    Command::new(bin)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
        || Command::new(bin)
            .args(["version", "--client"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
}

pub async fn up_k8s(
    root: &Path,
    env_name: &str,
    mode: K8sMode,
    provider: K8sProvider,
    runtime_choice: ContainerRuntimeChoice,
    rebuild_image: bool,
) -> Result<EnvState, BoxError> {
    let resolved_provider = match provider {
        K8sProvider::Kind => K8sProvider::Kind,
        K8sProvider::PodmanKube => K8sProvider::PodmanKube,
        K8sProvider::Auto => {
            if has_tool("kind") && has_tool("kubectl") {
                K8sProvider::Kind
            } else if has_tool("podman") {
                K8sProvider::PodmanKube
            } else {
                return Err(
                    "local Kubernetes mode requires either `kind` + `kubectl` or `podman` (`podman kube play`)"
                        .into(),
                );
            }
        }
    };

    let manifest = root.join(mode.manifest_rel_path());
    if !manifest.exists() {
        return Err(format!("missing Kubernetes manifest {}", manifest.display()).into());
    }

    let (primary_url, peer_url) = if mode == K8sMode::Multiregion {
        (
            "http://127.0.0.1:8081".to_string(),
            Some("http://127.0.0.1:8082".to_string()),
        )
    } else {
        ("http://127.0.0.1:8080".to_string(), None)
    };

    match resolved_provider {
        K8sProvider::PodmanKube => {
            let mut state = EnvState::new(
                env_name,
                mode.mode_name(),
                "podman-kube",
                &primary_url,
                peer_url,
                AuthMode::None,
            );
            state.resources.kube_manifest = Some(mode.manifest_rel_path().to_string());
            save_state(root, &state)?;

            ensure_server_container_image(root, "podman", rebuild_image)?;

            println!(
                "==> Applying Kubernetes manifest {} via `podman kube play`...",
                mode.manifest_rel_path()
            );
            let out = Command::new("podman")
                .current_dir(root)
                .args(["kube", "play", "--replace", mode.manifest_rel_path()])
                .output()?;
            if !out.status.success() {
                return Err(format!(
                    "`podman kube play` failed: {}",
                    String::from_utf8_lossy(&out.stderr)
                )
                .into());
            }

            wait_until_healthy(
                &state.endpoints.primary,
                state.auth,
                None,
                Duration::from_secs(35),
            )
            .await?;
            if let Some(peer) = &state.endpoints.peer {
                wait_until_healthy(peer, state.auth, None, Duration::from_secs(35)).await?;
            }
            Ok(state)
        }
        K8sProvider::Kind | K8sProvider::Auto => {
            let rt = detect_container_runtime(runtime_choice)?;
            let cluster_name = format!("jj-cc-{env_name}");
            let mut state = EnvState::new(
                env_name,
                mode.mode_name(),
                "kind",
                &primary_url,
                peer_url,
                AuthMode::None,
            );
            state.resources.kind_cluster = Some(cluster_name.clone());
            state.resources.kube_context = Some(format!("kind-{cluster_name}"));
            state.resources.kube_manifest = Some(mode.manifest_rel_path().to_string());
            save_state(root, &state)?;

            ensure_server_container_image(root, &rt, rebuild_image)?;
            ensure_kind_cluster(root, &cluster_name, mode)?;

            println!(
                "==> Loading localhost/jj-cc-server:local into kind cluster {cluster_name}..."
            );
            let load_status = Command::new("kind")
                .args([
                    "load",
                    "docker-image",
                    "localhost/jj-cc-server:local",
                    "--name",
                    &cluster_name,
                ])
                .status()?;
            if !load_status.success() {
                return Err("`kind load docker-image` failed".into());
            }

            let ctx = format!("kind-{cluster_name}");
            let apply_status = Command::new("kubectl")
                .current_dir(root)
                .args(["--context", &ctx, "apply", "-f", mode.manifest_rel_path()])
                .status()?;
            if !apply_status.success() {
                return Err("`kubectl apply` failed".into());
            }

            wait_until_healthy(
                &state.endpoints.primary,
                state.auth,
                None,
                Duration::from_secs(45),
            )
            .await?;
            if let Some(peer) = &state.endpoints.peer {
                wait_until_healthy(peer, state.auth, None, Duration::from_secs(45)).await?;
            }
            Ok(state)
        }
    }
}

fn ensure_kind_cluster(root: &Path, cluster_name: &str, mode: K8sMode) -> Result<(), BoxError> {
    let existing = Command::new("kind").args(["get", "clusters"]).output()?;
    let clusters_out = String::from_utf8_lossy(&existing.stdout);
    if clusters_out.lines().any(|l| l.trim() == cluster_name) {
        return Ok(());
    }

    let port_mappings = if mode == K8sMode::Multiregion {
        "  - containerPort: 30081\n    hostPort: 8081\n    listenAddress: \"127.0.0.1\"\n  - containerPort: 30082\n    hostPort: 8082\n    listenAddress: \"127.0.0.1\""
    } else {
        "  - containerPort: 30080\n    hostPort: 8080\n    listenAddress: \"127.0.0.1\""
    };
    let cfg = format!(
        "kind: Cluster\napiVersion: kind.x-k8s.io/v1alpha4\nnodes:\n- role: control-plane\n  extraPortMappings:\n{port_mappings}\n"
    );
    let cfg_path = root
        .join(".jj-cc")
        .join(format!("{cluster_name}-kind.yaml"));
    fs::write(&cfg_path, cfg)?;

    println!("==> Creating local kind Kubernetes cluster {cluster_name}...");
    let status = Command::new("kind")
        .args([
            "create",
            "cluster",
            "--name",
            cluster_name,
            "--config",
            cfg_path.to_str().unwrap(),
        ])
        .status()?;
    if !status.success() {
        return Err("`kind create cluster` failed".into());
    }
    Ok(())
}

pub fn down_k8s(root: &Path, state: &EnvState, purge: bool) -> Result<(), BoxError> {
    if state.runtime == "podman-kube" {
        if let Some(manifest) = &state.resources.kube_manifest {
            let mut cmd = Command::new("podman");
            cmd.current_dir(root).args(["kube", "down"]);
            if purge {
                cmd.arg("--force");
            }
            cmd.arg(manifest);
            let out = cmd.output()?;
            let stderr = String::from_utf8_lossy(&out.stderr);
            let lower = stderr.to_ascii_lowercase();
            if !out.status.success() && !lower.contains("no such") && !lower.contains("not found") {
                return Err(format!("`podman kube down` failed: {}", stderr.trim()).into());
            }
        }
    } else if state.runtime == "kind"
        && let Some(cluster) = &state.resources.kind_cluster
    {
        if !purge && let Some(manifest) = &state.resources.kube_manifest {
            let default_ctx = format!("kind-{cluster}");
            let ctx = state
                .resources
                .kube_context
                .as_deref()
                .unwrap_or(&default_ctx);
            let _ = Command::new("kubectl")
                .current_dir(root)
                .args([
                    "--context",
                    ctx,
                    "delete",
                    "--ignore-not-found",
                    "-f",
                    manifest,
                ])
                .status();
        }
        let out = Command::new("kind")
            .args(["delete", "cluster", "--name", cluster])
            .output()?;
        if !out.status.success() {
            return Err(format!(
                "`kind delete cluster --name {cluster}` failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )
            .into());
        }
    }
    Ok(())
}
