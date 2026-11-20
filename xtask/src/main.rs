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

//! Cross-platform workflow driver (`cargo xtask`) for building, deploying, smoke-testing,
//! and tearing down `jj-commit-cloud` across local containers, local Kubernetes, and GCP.
//!
//! # Authority and Crash-Safety Ordering
//! `xtask up` writes `.jj-cc/state.json` (or `.jj-cc/<env>.json`) *before* launching
//! containers, `kind` clusters, background processes, or Google Cloud services, and
//! appends managed resource identifiers incrementally. If `up` is interrupted partway
//! through, `xtask down` reads the persisted manifest and cleans up partial state. `down`
//! deletes only cloud storage that `up` created, drops each resource from the manifest once it
//! is gone, and keeps the file when a deletion fails.
//!
//! # Concurrency and Merge Invariants
//! Separate named environments (`--env <name>`) isolate state files, container prefixes,
//! and default GCP resource names so a developer can keep a GCP deployment active alongside a
//! local stack; pass `--port` to run two container or native stacks at once.
//!
//! # Cross-Process and RPC Boundaries
//! Readiness checks probe `tonic_health` `Health::Check` over HTTP/2 gRPC (attaching TLS
//! roots and `gcloud` IAM identity tokens on Cloud Run) before reporting `SERVING`.

mod gcp;
mod k8s;
mod local;
mod smoke;
mod state;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};

use gcp::{GcpDeployOptions, GcpMode, down_gcp, env_resource_name, up_gcp};
use k8s::{K8sMode, K8sProvider, down_k8s, up_k8s};
use local::{
    ContainerRuntimeChoice, LocalDriver, LocalMode, build_and_export_binaries, check_grpc_health,
    detect_container_runtime, down_local, up_local,
};
use smoke::{SmokeOptions, run_smoke};
use state::{BoxError, EnvState, load_state, remove_state, state_file_path, workspace_root};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum DeployMode {
    /// Ephemeral single-node server in memory
    Memory,
    /// Durable single-node server backed by SQLite + local content-addressed blobs
    Sqlite,
    /// Local Cloud Spanner Emulator + Cloud Storage Emulator (`fake-gcs-server`)
    Spanner,
    /// Two local regional servers (`:8081` US, `:8082` EU) sharing Spanner with isolated GCS
    /// buckets
    Multiregion,
    /// Local Kubernetes stack (alias for `k8s-spanner`) via `kind` or `podman kube play`
    K8s,
    /// Local Kubernetes in-memory server pod
    K8sMemory,
    /// Local Kubernetes SQLite server pod with PersistentVolumeClaim
    K8sSqlite,
    /// Local Kubernetes Spanner + GCS emulator + server pod
    K8sSpanner,
    /// Local Kubernetes two-region (`:8081` US, `:8082` EU) pod
    K8sMultiregion,
    /// Google Cloud Run serverless HTTP/2 deployment backed by Cloud Spanner + Cloud Storage
    CloudRun,
    /// Google Compute Engine Container-Optimized OS server VM backed by Cloud Spanner + Cloud
    /// Storage
    Gce,
    /// Google Kubernetes Engine Autopilot deployment backed by Cloud Spanner + Cloud Storage
    Gke,
}

#[derive(Parser, Debug)]
#[command(
    name = "cargo xtask",
    about = "Build, deploy, and smoke-test jj-commit-cloud across local, Kubernetes, and GCP modes"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Build server and custom jj CLI (natively or inside multi-stage container)
    Build {
        /// Build inside the multi-stage Dockerfile via docker/podman
        #[arg(long)]
        container: bool,
        /// Export compiled binaries into `./bin/`
        #[arg(long)]
        export_bin: bool,
        /// Container runtime when `--container` is set
        #[arg(long, value_enum, default_value_t = ContainerRuntimeChoice::Auto)]
        runtime: ContainerRuntimeChoice,
    },

    /// Start a local, Kubernetes, or Google Cloud Commit Cloud environment
    Up {
        /// Target deployment topology
        #[arg(value_enum, default_value_t = DeployMode::Sqlite)]
        mode: DeployMode,

        /// Environment state slot name (stored at `.jj-cc/state.json` for `default`)
        #[arg(long, default_value = "default")]
        env: String,

        /// Tear down any existing environment in the same `--env` slot before starting
        #[arg(long)]
        replace: bool,

        /// Force rebuilding the local container image before starting
        #[arg(long)]
        rebuild: bool,

        /// Driver for local non-Kubernetes modes (`auto` prefers docker/podman containers)
        #[arg(long, value_enum, default_value_t = LocalDriver::Auto)]
        driver: LocalDriver,

        /// Container runtime preference
        #[arg(long, value_enum, default_value_t = ContainerRuntimeChoice::Auto)]
        runtime: ContainerRuntimeChoice,

        /// Local Kubernetes provider (`auto` prefers `kind`+`kubectl`, then `podman kube play`)
        #[arg(long, value_enum, default_value_t = K8sProvider::Auto)]
        k8s_provider: K8sProvider,

        /// Primary localhost port for local modes (`8081` automatically in `multiregion`)
        #[arg(long, default_value_t = 8080)]
        port: u16,

        /// Secondary peer localhost port for local `multiregion` mode
        #[arg(long, default_value_t = 8082)]
        peer_port: u16,

        /// Google Cloud project ID (required for `cloud-run`, `gce`, and `gke`)
        #[arg(long, env = "PROJECT_ID")]
        project: Option<String>,

        /// Primary Google Cloud region
        #[arg(long, default_value = "us-central1")]
        region: String,

        /// Optional secondary Google Cloud region for multi-region cloud deployments
        #[arg(long)]
        peer_region: Option<String>,

        /// Cloud Spanner instance ID for GCP modes (shared by environments; `down --purge`
        /// deletes it only if this environment created it and no databases remain)
        #[arg(long, default_value = "jj-cc-dev")]
        spanner_instance: String,

        /// Cloud Spanner database ID for GCP modes [default: `commit-cloud`, or
        /// `commit-cloud-<env>` for named environments]
        #[arg(long)]
        spanner_database: Option<String>,

        /// GKE Autopilot cluster name when `mode = gke` [default: `jj-cc-cluster`, or
        /// `jj-cc-cluster-<env>` for named environments]
        #[arg(long)]
        gke_cluster: Option<String>,

        /// Optional pre-built container image URI for GCP modes (skips `gcloud builds submit`)
        #[arg(long)]
        image: Option<String>,

        /// Assign public external IPs to GCE VMs instead of tunneling over Identity-Aware Proxy
        /// (IAP); requires `--allow-cidr`, and the server then requires a generated bearer token
        #[arg(long, requires = "allow_cidr")]
        gce_public_ip: bool,

        /// Source range allowed to reach public GCE VMs (repeatable or comma-separated, e.g.
        /// `203.0.113.0/24`)
        #[arg(long, requires = "gce_public_ip", value_delimiter = ',', value_parser = gcp::parse_allow_cidr)]
        allow_cidr: Vec<String>,
    },

    /// Inspect the active environment and probe live gRPC server health
    Status {
        /// Environment state slot name
        #[arg(long, default_value = "default")]
        env: String,
    },

    /// Run an end-to-end repo init, version commit, clone, and historical checkout test
    Smoke {
        /// Environment state slot name
        #[arg(long, default_value = "default")]
        env: String,

        /// Optional explicit primary server URL (overrides `.jj-cc/state.json`)
        #[arg(long, env = "SMOKE_SERVER_URL")]
        server: Option<String>,

        /// Optional explicit peer replica server URL for cross-region checkout testing
        #[arg(long, env = "SMOKE_PEER_SERVER_URL")]
        peer_server: Option<String>,

        /// Optional path to compiled custom `jj` binary
        #[arg(long)]
        jj_bin: Option<PathBuf>,

        /// Keep temporary smoke workspaces on disk after completion
        #[arg(long)]
        keep: bool,
    },

    /// Tear down the active environment recorded in `.jj-cc/state.json`
    Down {
        /// Optional explicit mode fallback if `.jj-cc/state.json` was manually removed
        #[arg(value_enum)]
        mode: Option<DeployMode>,

        /// Environment state slot name
        #[arg(long, default_value = "default")]
        env: String,

        /// Also delete data volumes, plus the Spanner database and instance, Cloud Storage
        /// buckets, and GKE cluster that this environment's `up` created
        #[arg(long)]
        purge: bool,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), BoxError> {
    let cli = Cli::parse();
    let root = workspace_root();

    match cli.command {
        Commands::Build {
            container,
            export_bin,
            runtime,
        } => {
            build_and_export_binaries(&root, container, export_bin, runtime)?;
        }
        Commands::Up {
            mode,
            env,
            replace,
            rebuild,
            driver,
            runtime,
            k8s_provider,
            port,
            peer_port,
            project,
            region,
            peer_region,
            spanner_instance,
            spanner_database,
            gke_cluster,
            image,
            gce_public_ip,
            allow_cidr,
        } => {
            if let Some(mut existing) = load_state(&root, &env)? {
                if replace {
                    println!(
                        "==> Replacing existing '{}' environment (mode: {})...",
                        env, existing.mode
                    );
                    teardown_state(&root, &mut existing, false)?;
                    remove_state(&root, &env)?;
                } else {
                    return Err(format!(
                        "environment '{}' is already active (mode: {}, primary: {}); run `cargo xtask down` first or pass `--replace`",
                        env, existing.mode, existing.endpoints.primary
                    )
                    .into());
                }
            }

            let state = match mode {
                DeployMode::Memory => {
                    up_local(
                        &root,
                        &env,
                        LocalMode::Memory,
                        driver,
                        runtime,
                        port,
                        peer_port,
                        rebuild,
                    )
                    .await?
                }
                DeployMode::Sqlite => {
                    up_local(
                        &root,
                        &env,
                        LocalMode::Sqlite,
                        driver,
                        runtime,
                        port,
                        peer_port,
                        rebuild,
                    )
                    .await?
                }
                DeployMode::Spanner => {
                    up_local(
                        &root,
                        &env,
                        LocalMode::Spanner,
                        driver,
                        runtime,
                        port,
                        peer_port,
                        rebuild,
                    )
                    .await?
                }
                DeployMode::Multiregion => {
                    up_local(
                        &root,
                        &env,
                        LocalMode::Multiregion,
                        driver,
                        runtime,
                        port,
                        peer_port,
                        rebuild,
                    )
                    .await?
                }
                DeployMode::K8s | DeployMode::K8sSpanner => {
                    up_k8s(
                        &root,
                        &env,
                        K8sMode::Spanner,
                        k8s_provider,
                        runtime,
                        rebuild,
                    )
                    .await?
                }
                DeployMode::K8sMemory => {
                    up_k8s(&root, &env, K8sMode::Memory, k8s_provider, runtime, rebuild).await?
                }
                DeployMode::K8sSqlite => {
                    up_k8s(&root, &env, K8sMode::Sqlite, k8s_provider, runtime, rebuild).await?
                }
                DeployMode::K8sMultiregion => {
                    up_k8s(
                        &root,
                        &env,
                        K8sMode::Multiregion,
                        k8s_provider,
                        runtime,
                        rebuild,
                    )
                    .await?
                }
                DeployMode::CloudRun | DeployMode::Gce | DeployMode::Gke => {
                    let project_id = project.ok_or(
                        "`--project <PROJECT_ID>` (or `PROJECT_ID` env var) is required for GCP modes",
                    )?;
                    let gcp_mode = match mode {
                        DeployMode::CloudRun => GcpMode::CloudRun,
                        DeployMode::Gce => GcpMode::Gce,
                        DeployMode::Gke => GcpMode::Gke,
                        _ => unreachable!(),
                    };
                    up_gcp(
                        &root,
                        &env,
                        gcp_mode,
                        &GcpDeployOptions {
                            project_id,
                            region,
                            peer_region,
                            spanner_instance,
                            spanner_database: spanner_database
                                .unwrap_or_else(|| env_resource_name("commit-cloud", &env)),
                            gke_cluster: gke_cluster
                                .unwrap_or_else(|| env_resource_name("jj-cc-cluster", &env)),
                            image_override: image,
                            local_port: port,
                            local_peer_port: peer_port,
                            gce_public_ip,
                            allow_cidrs: allow_cidr,
                        },
                    )
                    .await?
                }
            };

            let state_path = state_file_path(&root, &env);
            println!(
                "==> Environment '{}' is SERVING (mode: {})",
                env, state.mode
            );
            println!("    Primary endpoint: {}", state.endpoints.primary);
            if let Some(peer) = &state.endpoints.peer {
                println!("    Peer endpoint:    {peer}");
            }
            println!("    State recorded:   {}", state_path.display());
            if state.auth_token.is_some() {
                println!(
                    "    Bearer token:     `auth_token` in the state file; export it as JJ_CC_AUTH_TOKEN for jj commands"
                );
            }
            println!("    Next steps:       `cargo xtask smoke` or `cargo xtask down`");
        }
        Commands::Status { env } => {
            let Some(state) = load_state(&root, &env)? else {
                println!("No active environment in slot '{env}'.");
                return Ok(());
            };
            let token = state.auth_token.as_deref();
            let primary_ok = check_grpc_health(&state.endpoints.primary, state.auth, token)
                .await
                .unwrap_or(false);
            println!("Environment: {}", state.env_name);
            println!("Mode:        {} (runtime: {})", state.mode, state.runtime);
            println!(
                "Primary:     {} [{}]",
                state.endpoints.primary,
                if primary_ok { "SERVING" } else { "UNREACHABLE" }
            );
            if let Some(peer) = &state.endpoints.peer {
                let peer_ok = check_grpc_health(peer, state.auth, token)
                    .await
                    .unwrap_or(false);
                println!(
                    "Peer:        {peer} [{}]",
                    if peer_ok { "SERVING" } else { "UNREACHABLE" }
                );
            }
        }
        Commands::Smoke {
            env,
            server,
            peer_server,
            jj_bin,
            keep,
        } => {
            run_smoke(&SmokeOptions {
                env_name: env,
                server_override: server,
                peer_server_override: peer_server,
                jj_bin_override: jj_bin,
                keep_workspaces: keep,
            })?;
        }
        Commands::Down { mode, env, purge } => {
            if let Some(mut state) = load_state(&root, &env)? {
                println!(
                    "==> Tearing down environment '{}' (mode: {}, runtime: {})...",
                    env, state.mode, state.runtime
                );
                teardown_state(&root, &mut state, purge)?;
                remove_state(&root, &env)?;
                println!("==> Environment '{env}' stopped.");
            } else if let Some(fallback_mode) = mode {
                println!(
                    "==> No state file for '{env}'; attempting fallback cleanup for {fallback_mode:?}..."
                );
                let mut fallback = synthesize_fallback_state(&env, fallback_mode);
                teardown_state(&root, &mut fallback, purge)?;
            } else {
                println!("No active environment recorded for '{env}'.");
            }
        }
    }

    Ok(())
}

fn teardown_state(
    root: &std::path::Path,
    state: &mut EnvState,
    purge: bool,
) -> Result<(), BoxError> {
    match state.runtime.as_str() {
        "gcloud" => down_gcp(root, state, purge),
        "kind" | "podman-kube" => down_k8s(root, state, purge),
        _ => down_local(state, purge),
    }
}

fn synthesize_fallback_state(env_name: &str, mode: DeployMode) -> EnvState {
    let rt = detect_container_runtime(ContainerRuntimeChoice::Auto)
        .unwrap_or_else(|_| "podman".to_string());
    let prefix = format!("jj-cc-{env_name}");
    let mut st = EnvState::new(
        env_name,
        &format!("{mode:?}").to_lowercase(),
        &rt,
        "http://127.0.0.1:8080",
        None,
        state::AuthMode::None,
    );
    match mode {
        DeployMode::K8s
        | DeployMode::K8sMemory
        | DeployMode::K8sSqlite
        | DeployMode::K8sSpanner
        | DeployMode::K8sMultiregion => {
            st.runtime = "podman-kube".to_string();
            let kmode = match mode {
                DeployMode::K8sMemory => K8sMode::Memory,
                DeployMode::K8sSqlite => K8sMode::Sqlite,
                DeployMode::K8sMultiregion => K8sMode::Multiregion,
                _ => K8sMode::Spanner,
            };
            st.resources.kube_manifest = Some(kmode.manifest_rel_path().to_string());
        }
        _ => {
            st.resources.containers = vec![
                format!("{prefix}-server"),
                format!("{prefix}-server-us"),
                format!("{prefix}-server-eu"),
                format!("{prefix}-gcs"),
                format!("{prefix}-spanner"),
            ];
            st.resources.network = Some(format!("{prefix}-net"));
            st.resources.volumes = vec![format!("{prefix}-sqlite-data")];
        }
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gce_public_ip_requires_explicit_non_zero_allow_cidr() {
        assert!(
            Cli::try_parse_from([
                "cargo xtask",
                "up",
                "gce",
                "--project",
                "p",
                "--gce-public-ip"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "cargo xtask",
                "up",
                "gce",
                "--project",
                "p",
                "--allow-cidr",
                "203.0.113.0/24",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "cargo xtask",
                "up",
                "gce",
                "--project",
                "p",
                "--gce-public-ip",
                "--allow-cidr",
                "0.0.0.0/0",
            ])
            .is_err()
        );
        let parsed = Cli::try_parse_from([
            "cargo xtask",
            "up",
            "gce",
            "--project",
            "p",
            "--gce-public-ip",
            "--allow-cidr",
            "203.0.113.0/24,198.51.100.7/32",
        ])
        .unwrap();
        match parsed.command {
            Commands::Up {
                gce_public_ip,
                allow_cidr,
                ..
            } => {
                assert!(gce_public_ip);
                assert_eq!(allow_cidr, vec!["203.0.113.0/24", "198.51.100.7/32"]);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }
}
