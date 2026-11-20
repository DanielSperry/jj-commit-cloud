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
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::local::{spawn_bg, stop_managed_processes, wait_until_healthy};
use crate::state::{AuthMode, BoxError, EnvState, ManagedProcess, save_state, state_file_path};

/// Command-line marker of the `kubectl port-forward` that reaches a GKE deployment.
const GKE_PORT_FORWARD_MARKER: &str = "port-forward service/jj-cc-server";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GcpMode {
    CloudRun,
    Gce,
    Gke,
}

impl GcpMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CloudRun => "cloud-run",
            Self::Gce => "gce",
            Self::Gke => "gke",
        }
    }
}

pub struct GcpDeployOptions {
    pub project_id: String,
    pub region: String,
    pub peer_region: Option<String>,
    pub spanner_instance: String,
    pub spanner_database: String,
    pub gke_cluster: String,
    pub image_override: Option<String>,
    pub local_port: u16,
    pub local_peer_port: u16,
    pub gce_public_ip: bool,
    /// Source ranges allowed to reach a public GCE server (`--allow-cidr`).
    pub allow_cidrs: Vec<String>,
}

/// Default name of an env-scoped GCP resource: `base` for the `default` env, which keeps the
/// names earlier versions used, and `base-<env>` for named envs, so separate `--env` slots in
/// one project never share or purge each other's databases, buckets, clusters, or services.
pub fn env_resource_name(base: &str, env_name: &str) -> String {
    if env_name == "default" {
        base.to_string()
    } else {
        format!("{base}-{env_name}")
    }
}

/// Rejects env names that cannot appear in GCP resource names. The tightest limit is the
/// 30-character Spanner database ID `commit-cloud-<env>`.
fn validate_gcp_env_name(env_name: &str) -> Result<(), BoxError> {
    let valid = (1..=17).contains(&env_name.len())
        && env_name.starts_with(|c: char| c.is_ascii_lowercase())
        && !env_name.ends_with('-')
        && env_name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if valid {
        Ok(())
    } else {
        Err(format!(
            "`--env {env_name}` cannot name GCP resources: use 1-17 lowercase letters, digits, or hyphens, starting with a letter"
        )
        .into())
    }
}

fn blob_bucket_name(project_id: &str, env_name: &str, region: &str) -> Result<String, BoxError> {
    let bucket = format!("{project_id}-jj-blobs-{env_name}-{region}");
    if bucket.len() > 63 {
        return Err(format!(
            "bucket name `{bucket}` is longer than 63 characters; use a shorter `--env`"
        )
        .into());
    }
    Ok(bucket)
}

/// Parses an `--allow-cidr` value. A zero-length prefix admits every address, so it is
/// rejected: a public GCE server only accepts connections from ranges named explicitly.
pub fn parse_allow_cidr(value: &str) -> Result<String, String> {
    let (addr, prefix) = value
        .split_once('/')
        .ok_or_else(|| format!("`{value}` is not a CIDR range such as 203.0.113.0/24"))?;
    let ip: std::net::IpAddr = addr
        .parse()
        .map_err(|_| format!("`{addr}` is not an IP address"))?;
    let max_prefix = if ip.is_ipv4() { 32 } else { 128 };
    match prefix.parse::<u8>() {
        Ok(len) if (1..=max_prefix).contains(&len) => Ok(value.to_string()),
        _ => Err(format!(
            "prefix `/{prefix}` must be between /1 and /{max_prefix}; xtask never opens the server to every address"
        )),
    }
}

/// Generates the bearer token a public GCE server requires: 244 random bits from two v4 UUIDs.
fn generate_auth_token() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

/// Kubernetes context that `gcloud container clusters get-credentials` writes for a cluster.
fn gke_kube_context(project_id: &str, region: &str, cluster: &str) -> String {
    format!("gke_{project_id}_{region}_{cluster}")
}

pub fn locate_gcloud() -> Result<PathBuf, BoxError> {
    let candidates = if cfg!(windows) {
        vec!["gcloud.cmd", "gcloud"]
    } else {
        vec!["gcloud", "/google/bin/releases/cloud-sdk-build/gcloud.par"]
    };
    for cand in candidates {
        let p = PathBuf::from(cand);
        if (p.is_absolute() && p.exists())
            || Command::new(&p)
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        {
            return Ok(p);
        }
    }
    Err("`gcloud` CLI was not found; install Google Cloud SDK before running GCP modes".into())
}

pub fn mint_gcloud_identity_token() -> Result<String, BoxError> {
    let gcloud = locate_gcloud()?;
    let out = Command::new(gcloud)
        .args(["auth", "print-identity-token"])
        .output()?;
    if !out.status.success() {
        return Err(format!(
            "`gcloud auth print-identity-token` failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn run_gcloud(gcloud: &Path, args: &[&str]) -> Result<String, BoxError> {
    let out = Command::new(gcloud).args(args).output()?;
    if !out.status.success() {
        return Err(format!(
            "`gcloud {}` failed:\n{}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn gcloud_ok(gcloud: &Path, args: &[&str]) -> bool {
    Command::new(gcloud)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Fails when an env-named resource already exists. State files record only resources xtask
/// created, so `up` refuses to adopt one left behind by a run whose state file was lost.
fn ensure_absent(
    gcloud: &Path,
    describe_args: &[&str],
    what: &str,
    delete_cmd: &str,
) -> Result<(), BoxError> {
    if gcloud_ok(gcloud, describe_args) {
        return Err(format!(
            "{what} already exists but is not recorded in this environment's state; delete it with `{delete_cmd}` or use another `--env`"
        )
        .into());
    }
    Ok(())
}

pub async fn up_gcp(
    root: &Path,
    env_name: &str,
    mode: GcpMode,
    opts: &GcpDeployOptions,
) -> Result<EnvState, BoxError> {
    validate_gcp_env_name(env_name)?;
    if opts.gce_public_ip && mode != GcpMode::Gce {
        return Err("`--gce-public-ip` only applies to `up gce`".into());
    }
    if opts.gce_public_ip && opts.allow_cidrs.is_empty() {
        return Err(
            "`--gce-public-ip` requires `--allow-cidr <CIDR>` naming the networks that may connect"
                .into(),
        );
    }
    let primary_bucket = blob_bucket_name(&opts.project_id, env_name, &opts.region)?;
    let peer_bucket = match &opts.peer_region {
        Some(peer_reg) => Some(blob_bucket_name(&opts.project_id, env_name, peer_reg)?),
        None => None,
    };
    let gcloud = locate_gcloud()?;
    let auth = match mode {
        GcpMode::CloudRun => AuthMode::GcloudIdentityToken,
        GcpMode::Gce | GcpMode::Gke => AuthMode::None,
    };

    let mut state = EnvState::new(env_name, mode.as_str(), "gcloud", "pending", None, auth);
    state.project_id = Some(opts.project_id.clone());
    state.resources.spanner_instance = Some(opts.spanner_instance.clone());
    state.resources.spanner_database = Some(opts.spanner_database.clone());
    save_state(root, &state)?;

    // 1. Enable required APIs
    let mut apis = vec![
        "spanner.googleapis.com",
        "storage.googleapis.com",
        "artifactregistry.googleapis.com",
        "cloudbuild.googleapis.com",
        "iam.googleapis.com",
    ];
    match mode {
        GcpMode::CloudRun => apis.push("run.googleapis.com"),
        GcpMode::Gce => {
            apis.push("compute.googleapis.com");
            apis.push("iap.googleapis.com");
        }
        GcpMode::Gke => apis.push("container.googleapis.com"),
    }
    println!(
        "==> Enabling Google Cloud APIs in project {}...",
        opts.project_id
    );
    let mut enable_args = vec!["services", "enable"];
    enable_args.extend(apis);
    let proj_flag = format!("--project={}", opts.project_id);
    enable_args.push(&proj_flag);
    run_gcloud(&gcloud, &enable_args)?;

    // 2. Ensure Spanner instance & database, recording which ones this environment creates so
    //    `down --purge` never deletes pre-existing ones.
    if !gcloud_ok(
        &gcloud,
        &[
            "spanner",
            "instances",
            "describe",
            &opts.spanner_instance,
            &proj_flag,
        ],
    ) {
        let sp_cfg = if opts.peer_region.is_some() {
            "nam-eur-asia1".to_string()
        } else {
            format!("regional-{}", opts.region)
        };
        println!(
            "==> Creating Cloud Spanner instance {} ({sp_cfg}, 100 PUs)...",
            opts.spanner_instance
        );
        run_gcloud(
            &gcloud,
            &[
                "spanner",
                "instances",
                "create",
                &opts.spanner_instance,
                &proj_flag,
                &format!("--config={sp_cfg}"),
                "--description=jj-commit-cloud metadata",
                "--processing-units=100",
            ],
        )?;
        state.resources.spanner_instance_created = true;
        save_state(root, &state)?;
    }
    if !gcloud_ok(
        &gcloud,
        &[
            "spanner",
            "databases",
            "describe",
            &opts.spanner_database,
            &format!("--instance={}", opts.spanner_instance),
            &proj_flag,
        ],
    ) {
        let ddl_file = root.join("server/sql/spanner_schema.sql");
        println!(
            "==> Creating Cloud Spanner database {} from {}...",
            opts.spanner_database,
            ddl_file.display()
        );
        run_gcloud(
            &gcloud,
            &[
                "spanner",
                "databases",
                "create",
                &opts.spanner_database,
                &format!("--instance={}", opts.spanner_instance),
                &proj_flag,
                &format!("--ddl-file={}", ddl_file.display()),
            ],
        )?;
        state.resources.spanner_database_created = true;
        save_state(root, &state)?;
    }

    // 3. Ensure runtime service account & Spanner IAM
    let sa_name = "jj-cc-server-sa";
    let sa_email = format!("{sa_name}@{}.iam.gserviceaccount.com", opts.project_id);
    if !gcloud_ok(
        &gcloud,
        &["iam", "service-accounts", "describe", &sa_email, &proj_flag],
    ) {
        run_gcloud(
            &gcloud,
            &[
                "iam",
                "service-accounts",
                "create",
                sa_name,
                &proj_flag,
                "--display-name=jj-cc-server runtime",
            ],
        )?;
    }
    run_gcloud(
        &gcloud,
        &[
            "spanner",
            "databases",
            "add-iam-policy-binding",
            &opts.spanner_database,
            &format!("--instance={}", opts.spanner_instance),
            &proj_flag,
            &format!("--member=serviceAccount:{sa_email}"),
            "--role=roles/spanner.databaseUser",
        ],
    )?;

    // 4. Ensure secure-by-default Cloud Storage bucket(s)
    ensure_gcs_bucket(
        &gcloud,
        root,
        &mut state,
        &opts.project_id,
        &primary_bucket,
        &opts.region,
        &sa_email,
    )?;
    if let (Some(b), Some(peer_reg)) = (&peer_bucket, &opts.peer_region) {
        ensure_gcs_bucket(
            &gcloud,
            root,
            &mut state,
            &opts.project_id,
            b,
            peer_reg,
            &sa_email,
        )?;
    }

    // 5. Ensure Artifact Registry & container image
    let image_uri = if let Some(img) = &opts.image_override {
        img.clone()
    } else {
        let ar_repo = "jj-commit-cloud";
        if !gcloud_ok(
            &gcloud,
            &[
                "artifacts",
                "repositories",
                "describe",
                ar_repo,
                &format!("--location={}", opts.region),
                &proj_flag,
            ],
        ) {
            run_gcloud(
                &gcloud,
                &[
                    "artifacts",
                    "repositories",
                    "create",
                    ar_repo,
                    &proj_flag,
                    "--repository-format=docker",
                    &format!("--location={}", opts.region),
                ],
            )?;
        }
        let uri = format!(
            "{}-docker.pkg.dev/{}/{ar_repo}/jj-cc-server:latest",
            opts.region, opts.project_id
        );
        println!("==> Building server image via Cloud Build ({uri})...");
        let status = Command::new(&gcloud)
            .current_dir(root)
            .args(["builds", "submit", &proj_flag, &format!("--tag={uri}"), "."])
            .status()?;
        if !status.success() {
            return Err("`gcloud builds submit` failed".into());
        }
        uri
    };

    let spanner_db_uri = format!(
        "projects/{}/instances/{}/databases/{}",
        opts.project_id, opts.spanner_instance, opts.spanner_database
    );
    let primary_gcs_uri = format!("gs://{primary_bucket}");
    let peer_gcs_uri = peer_bucket.as_ref().map(|b| format!("gs://{b}"));

    // 6. Deploy target compute topology
    match mode {
        GcpMode::CloudRun => {
            let service = env_resource_name("jj-cc-server", env_name);
            ensure_cloud_run_service_absent(&gcloud, &opts.project_id, &opts.region, &service)?;
            state
                .resources
                .cloud_run_services
                .push(format!("{}/{service}", opts.region));
            save_state(root, &state)?;
            let url_primary = deploy_cloud_run_region(
                &gcloud,
                &opts.project_id,
                &opts.region,
                &service,
                &image_uri,
                &sa_email,
                &spanner_db_uri,
                &primary_gcs_uri,
                peer_gcs_uri.as_deref(),
            )?;
            state.endpoints.primary = url_primary;

            if let (Some(peer_reg), Some(peer_gcs)) = (&opts.peer_region, &peer_gcs_uri) {
                ensure_cloud_run_service_absent(&gcloud, &opts.project_id, peer_reg, &service)?;
                state
                    .resources
                    .cloud_run_services
                    .push(format!("{peer_reg}/{service}"));
                save_state(root, &state)?;
                let url_peer = deploy_cloud_run_region(
                    &gcloud,
                    &opts.project_id,
                    peer_reg,
                    &service,
                    &image_uri,
                    &sa_email,
                    &spanner_db_uri,
                    peer_gcs,
                    Some(&primary_gcs_uri),
                )?;
                state.endpoints.peer = Some(url_peer);
            }
            save_state(root, &state)?;
        }
        GcpMode::Gce => {
            let fw_name = format!("jj-cc-{env_name}-allow-grpc");
            let vm_tag = format!("jj-cc-{env_name}-server");
            ensure_absent(
                &gcloud,
                &[
                    "compute",
                    "firewall-rules",
                    "describe",
                    &fw_name,
                    &proj_flag,
                ],
                &format!("firewall rule {fw_name}"),
                &format!("gcloud compute firewall-rules delete {fw_name} {proj_flag}"),
            )?;
            // The server port is open only to IAP's TCP forwarding range or to the ranges
            // passed with `--allow-cidr`, and only on this environment's VMs; xtask never
            // creates a rule without source ranges, which gcloud would open to 0.0.0.0/0.
            let source_ranges = if opts.gce_public_ip {
                opts.allow_cidrs.join(",")
            } else {
                "35.235.240.0/20".to_string()
            };
            if source_ranges.is_empty() {
                return Err("refusing to create a firewall rule without source ranges".into());
            }
            state.resources.gce_firewall_rule = Some(fw_name.clone());
            save_state(root, &state)?;
            run_gcloud(
                &gcloud,
                &[
                    "compute",
                    "firewall-rules",
                    "create",
                    &fw_name,
                    &proj_flag,
                    "--allow=tcp:8080",
                    &format!("--target-tags={vm_tag}"),
                    &format!("--source-ranges={source_ranges}"),
                ],
            )?;
            if opts.gce_public_ip {
                // A public IP needs authentication on top of the source ranges.
                state.auth_token = Some(generate_auth_token());
                save_state(root, &state)?;
            }

            let primary_local_port = if opts.peer_region.is_some() && opts.local_port == 8080 {
                8081
            } else {
                opts.local_port
            };
            let zone_primary = format!("{}-a", opts.region);
            let vm_primary = format!("jj-cc-{env_name}-{}", opts.region);
            ensure_gce_vm_absent(&gcloud, &opts.project_id, &zone_primary, &vm_primary)?;
            state
                .resources
                .gce_instances
                .push(format!("{zone_primary}/{vm_primary}"));
            save_state(root, &state)?;
            let ip_primary = deploy_gce_vm(
                &gcloud,
                &opts.project_id,
                &opts.region,
                &zone_primary,
                &vm_primary,
                &vm_tag,
                &image_uri,
                &sa_email,
                &spanner_db_uri,
                &primary_gcs_uri,
                peer_gcs_uri.as_deref(),
                state.auth_token.as_deref(),
            )?;
            if opts.gce_public_ip {
                state.endpoints.primary = format!("http://{ip_primary}:8080");
            } else {
                let tunnel = spawn_iap_tunnel(
                    &gcloud,
                    root,
                    &opts.project_id,
                    &zone_primary,
                    &vm_primary,
                    primary_local_port,
                )?;
                state.resources.processes.push(tunnel);
                state.endpoints.primary = format!("http://127.0.0.1:{primary_local_port}");
            }
            save_state(root, &state)?;

            if let (Some(peer_reg), Some(peer_gcs)) = (&opts.peer_region, &peer_gcs_uri) {
                let zone_peer = format!("{peer_reg}-b");
                let vm_peer = format!("jj-cc-{env_name}-{peer_reg}");
                ensure_gce_vm_absent(&gcloud, &opts.project_id, &zone_peer, &vm_peer)?;
                state
                    .resources
                    .gce_instances
                    .push(format!("{zone_peer}/{vm_peer}"));
                save_state(root, &state)?;
                let ip_peer = deploy_gce_vm(
                    &gcloud,
                    &opts.project_id,
                    peer_reg,
                    &zone_peer,
                    &vm_peer,
                    &vm_tag,
                    &image_uri,
                    &sa_email,
                    &spanner_db_uri,
                    peer_gcs,
                    Some(&primary_gcs_uri),
                    state.auth_token.as_deref(),
                )?;
                if opts.gce_public_ip {
                    state.endpoints.peer = Some(format!("http://{ip_peer}:8080"));
                } else {
                    let tunnel = spawn_iap_tunnel(
                        &gcloud,
                        root,
                        &opts.project_id,
                        &zone_peer,
                        &vm_peer,
                        opts.local_peer_port,
                    )?;
                    state.resources.processes.push(tunnel);
                    state.endpoints.peer =
                        Some(format!("http://127.0.0.1:{}", opts.local_peer_port));
                }
                save_state(root, &state)?;
            }
        }
        GcpMode::Gke => {
            let cluster_exists = gcloud_ok(
                &gcloud,
                &[
                    "container",
                    "clusters",
                    "describe",
                    &opts.gke_cluster,
                    &format!("--region={}", opts.region),
                    &proj_flag,
                ],
            );
            state.resources.gke_cluster = Some(opts.gke_cluster.clone());
            state.resources.gke_region = Some(opts.region.clone());
            save_state(root, &state)?;
            if !cluster_exists {
                println!(
                    "==> Creating GKE Autopilot cluster {} in {}...",
                    opts.gke_cluster, opts.region
                );
                run_gcloud(
                    &gcloud,
                    &[
                        "container",
                        "clusters",
                        "create-auto",
                        &opts.gke_cluster,
                        &format!("--region={}", opts.region),
                        &proj_flag,
                    ],
                )?;
                state.resources.gke_cluster_created = true;
                save_state(root, &state)?;
            }

            run_gcloud(
                &gcloud,
                &[
                    "container",
                    "clusters",
                    "get-credentials",
                    &opts.gke_cluster,
                    &format!("--region={}", opts.region),
                    &proj_flag,
                ],
            )?;
            // Pin every kubectl call (here and in `down`) to this cluster: the current context
            // can change at any time, for example when `up k8s` creates a kind cluster.
            let kube_context = gke_kube_context(&opts.project_id, &opts.region, &opts.gke_cluster);
            state.resources.kube_context = Some(kube_context.clone());
            save_state(root, &state)?;
            run_gcloud(
                &gcloud,
                &[
                    "iam",
                    "service-accounts",
                    "add-iam-policy-binding",
                    &sa_email,
                    &proj_flag,
                    "--role=roles/iam.workloadIdentityUser",
                    &format!(
                        "--member=serviceAccount:{}.svc.id.goog[default/jj-cc-server-ksa]",
                        opts.project_id
                    ),
                ],
            )?;

            let template = fs::read_to_string(root.join("deploy/k8s/gcp/gke.yaml"))?;
            let rendered = template
                .replace("__GCP_SERVICE_ACCOUNT__", &sa_email)
                .replace("__IMAGE_URI__", &image_uri)
                .replace("__SPANNER_DATABASE__", &spanner_db_uri)
                .replace("__GCS_BUCKET__", &primary_gcs_uri)
                .replace(
                    "__GCS_PEER_BUCKETS__",
                    peer_gcs_uri.as_deref().unwrap_or(""),
                );
            let rendered_path = root
                .join(".jj-cc")
                .join(format!("gke-rendered-{env_name}.yaml"));
            fs::write(&rendered_path, rendered)?;
            state.resources.kube_manifest = Some(rendered_path.display().to_string());
            save_state(root, &state)?;

            let status = Command::new("kubectl")
                .args([
                    "--context",
                    &kube_context,
                    "apply",
                    "-f",
                    rendered_path.to_str().unwrap(),
                ])
                .status()?;
            if !status.success() {
                return Err("`kubectl apply` to GKE failed".into());
            }
            let status = Command::new("kubectl")
                .args([
                    "--context",
                    &kube_context,
                    "rollout",
                    "status",
                    "deployment/jj-cc-server",
                    "--timeout=10m",
                ])
                .status()?;
            if !status.success() {
                return Err("GKE deployment jj-cc-server did not become ready".into());
            }

            // The Service is ClusterIP-only, so the server is never on a public IP. Reach it
            // through a local `kubectl port-forward`, managed like the GCE IAP tunnel.
            println!(
                "==> Starting kubectl port-forward 127.0.0.1:{} -> service/jj-cc-server:8080...",
                opts.local_port
            );
            let pid = spawn_bg(
                Path::new("kubectl"),
                &[
                    "--context",
                    &kube_context,
                    "port-forward",
                    "service/jj-cc-server",
                    &format!("{}:8080", opts.local_port),
                    "--address=127.0.0.1",
                ],
                &[],
                &root
                    .join(".jj-cc")
                    .join(format!("{env_name}-port-forward.log")),
            )?;
            state
                .resources
                .processes
                .push(ManagedProcess::new(pid, GKE_PORT_FORWARD_MARKER));
            state.endpoints.primary = format!("http://127.0.0.1:{}", opts.local_port);
            save_state(root, &state)?;
        }
    }

    let token = state.auth_token.as_deref();
    wait_until_healthy(
        &state.endpoints.primary,
        state.auth,
        token,
        Duration::from_secs(90),
    )
    .await?;
    if let Some(peer) = &state.endpoints.peer {
        wait_until_healthy(peer, state.auth, token, Duration::from_secs(90)).await?;
    }
    Ok(state)
}

fn ensure_cloud_run_service_absent(
    gcloud: &Path,
    project_id: &str,
    region: &str,
    service: &str,
) -> Result<(), BoxError> {
    let proj_flag = format!("--project={project_id}");
    let reg_flag = format!("--region={region}");
    ensure_absent(
        gcloud,
        &[
            "run", "services", "describe", service, &reg_flag, &proj_flag,
        ],
        &format!("Cloud Run service {service} in {region}"),
        &format!("gcloud run services delete {service} {reg_flag} {proj_flag}"),
    )
}

fn ensure_gce_vm_absent(
    gcloud: &Path,
    project_id: &str,
    zone: &str,
    vm_name: &str,
) -> Result<(), BoxError> {
    let proj_flag = format!("--project={project_id}");
    let zone_flag = format!("--zone={zone}");
    ensure_absent(
        gcloud,
        &[
            "compute",
            "instances",
            "describe",
            vm_name,
            &zone_flag,
            &proj_flag,
        ],
        &format!("VM {vm_name}"),
        &format!("gcloud compute instances delete {vm_name} {zone_flag} {proj_flag}"),
    )
}

/// Creates `bucket` unless it exists, records it (and whether this environment created it) in
/// `state`, then grants the runtime service account object access.
fn ensure_gcs_bucket(
    gcloud: &Path,
    root: &Path,
    state: &mut EnvState,
    project_id: &str,
    bucket: &str,
    location: &str,
    sa_email: &str,
) -> Result<(), BoxError> {
    let gs_uri = format!("gs://{bucket}");
    let proj_flag = format!("--project={project_id}");
    state.resources.gcs_buckets.push(bucket.to_string());
    save_state(root, state)?;
    if !gcloud_ok(
        gcloud,
        &["storage", "buckets", "describe", &gs_uri, &proj_flag],
    ) {
        println!("==> Creating Cloud Storage bucket {gs_uri} in {location}...");
        run_gcloud(
            gcloud,
            &[
                "storage",
                "buckets",
                "create",
                &gs_uri,
                &proj_flag,
                &format!("--location={location}"),
                "--uniform-bucket-level-access",
                "--public-access-prevention",
                "--soft-delete-duration=7d",
            ],
        )?;
        state.resources.gcs_buckets_created.push(bucket.to_string());
        save_state(root, state)?;
    }
    run_gcloud(
        gcloud,
        &[
            "storage",
            "buckets",
            "add-iam-policy-binding",
            &gs_uri,
            &format!("--member=serviceAccount:{sa_email}"),
            "--role=roles/storage.objectUser",
        ],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn deploy_cloud_run_region(
    gcloud: &Path,
    project_id: &str,
    region: &str,
    service: &str,
    image_uri: &str,
    sa_email: &str,
    spanner_db_uri: &str,
    gcs_uri: &str,
    peer_gcs_uri: Option<&str>,
) -> Result<String, BoxError> {
    println!("==> Deploying Cloud Run service {service} in {region}...");
    let peer_str = peer_gcs_uri.unwrap_or("");
    let env_arg = format!(
        "--set-env-vars=^@^STORE_TYPE=spanner@SPANNER_DATABASE={spanner_db_uri}@BLOB_STORE=gcs@GCS_BUCKET={gcs_uri}@GCS_PEER_BUCKETS={peer_str}"
    );
    let proj_flag = format!("--project={project_id}");
    let reg_flag = format!("--region={region}");
    run_gcloud(
        gcloud,
        &[
            "run",
            "deploy",
            service,
            &proj_flag,
            &reg_flag,
            &format!("--image={image_uri}"),
            &format!("--service-account={sa_email}"),
            "--no-allow-unauthenticated",
            "--use-http2",
            &env_arg,
        ],
    )?;
    run_gcloud(
        gcloud,
        &[
            "run",
            "services",
            "describe",
            service,
            &proj_flag,
            &reg_flag,
            "--format=value(status.url)",
        ],
    )
}

/// Creates a Container-Optimized OS VM running the server image. With `auth_token`, the VM gets
/// a public IP and the server requires that bearer token; the token goes through a private
/// env file so it never appears in the local process list.
#[allow(clippy::too_many_arguments)]
fn deploy_gce_vm(
    gcloud: &Path,
    project_id: &str,
    region: &str,
    zone: &str,
    vm_name: &str,
    network_tag: &str,
    image_uri: &str,
    sa_email: &str,
    spanner_db_uri: &str,
    gcs_uri: &str,
    peer_gcs_uri: Option<&str>,
    auth_token: Option<&str>,
) -> Result<String, BoxError> {
    let public_ip = auth_token.is_some();
    let proj_flag = format!("--project={project_id}");
    let zone_flag = format!("--zone={zone}");
    if !public_ip {
        let _ = run_gcloud(
            gcloud,
            &[
                "compute",
                "networks",
                "subnets",
                "update",
                "default",
                &format!("--region={region}"),
                &proj_flag,
                "--enable-private-ip-google-access",
            ],
        );
    }
    println!("==> Launching GCE Container-Optimized OS VM {vm_name} in {zone}...");
    let peer_str = peer_gcs_uri.unwrap_or("");
    let env_spanner = format!("--container-env=SPANNER_DATABASE={spanner_db_uri}");
    let env_gcs = format!("--container-env=GCS_BUCKET={gcs_uri},GCS_PEER_BUCKETS={peer_str}");
    let sa_flag = format!("--service-account={sa_email}");
    let img_flag = format!("--container-image={image_uri}");
    let tags_flag = format!("--tags={network_tag}");
    let mut create_args = vec![
        "compute",
        "instances",
        "create-with-container",
        vm_name,
        &proj_flag,
        &zone_flag,
        "--machine-type=e2-standard-2",
        &tags_flag,
        &sa_flag,
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        &img_flag,
        "--container-env=HOST=0.0.0.0,PORT=8080,STORE_TYPE=spanner,BLOB_STORE=gcs",
        &env_spanner,
        &env_gcs,
    ];
    let mut token_env = tempfile::NamedTempFile::new()?;
    let token_env_flag = format!("--container-env-file={}", token_env.path().display());
    if let Some(token) = auth_token {
        writeln!(token_env, "JJ_CC_SERVER_AUTH_TOKEN={token}")?;
        create_args.push(&token_env_flag);
    } else {
        create_args.push("--no-address");
    }
    run_gcloud(gcloud, &create_args)?;
    if public_ip {
        run_gcloud(
            gcloud,
            &[
                "compute",
                "instances",
                "describe",
                vm_name,
                &zone_flag,
                &proj_flag,
                "--format=get(networkInterfaces[0].accessConfigs[0].natIP)",
            ],
        )
    } else {
        Ok(String::new())
    }
}

fn spawn_iap_tunnel(
    gcloud: &Path,
    root: &Path,
    project_id: &str,
    zone: &str,
    vm_name: &str,
    local_port: u16,
) -> Result<ManagedProcess, BoxError> {
    println!("==> Starting IAP TCP tunnel 127.0.0.1:{local_port} -> {vm_name}:8080...");
    let pid = spawn_bg(
        gcloud,
        &[
            "compute",
            "start-iap-tunnel",
            vm_name,
            "8080",
            &format!("--local-host-port=127.0.0.1:{local_port}"),
            &format!("--zone={zone}"),
            &format!("--project={project_id}"),
        ],
        &[],
        &root
            .join(".jj-cc")
            .join(format!("{vm_name}-iap-tunnel.log")),
    )?;
    Ok(ManagedProcess::new(
        pid,
        &format!("start-iap-tunnel {vm_name}"),
    ))
}

/// Returns true when gcloud's error output says the resource does not exist.
fn is_not_found(stderr: &str) -> bool {
    let stderr = stderr.to_ascii_lowercase();
    [
        "not found",
        "not_found",
        "could not find",
        "cannot find",
        "does not exist",
        "matched no",
    ]
    .iter()
    .any(|needle| stderr.contains(needle))
}

/// Runs a gcloud delete command. A resource that is already gone counts as deleted, so a
/// retried `down` converges after a partial failure.
fn delete_gcloud_resource(gcloud: &Path, args: &[&str]) -> Result<(), String> {
    let out = Command::new(gcloud)
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    if out.status.success() || is_not_found(&stderr) {
        Ok(())
    } else {
        Err(stderr.trim().to_string())
    }
}

fn run_kubectl(kubectl: &Path, args: &[&str]) -> Result<(), String> {
    let out = Command::new(kubectl)
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// Records a failed deletion in `failures`; returns whether the resource is gone.
fn deleted(failures: &mut Vec<String>, what: &str, result: Result<(), String>) -> bool {
    match result {
        Ok(()) => true,
        Err(e) => {
            failures.push(format!("{what}: {e}"));
            false
        }
    }
}

pub fn down_gcp(root: &Path, state: &mut EnvState, purge: bool) -> Result<(), BoxError> {
    let failures = stop_managed_processes(state);
    if state.project_id.is_none() {
        return teardown_result(root, state, failures);
    }
    let gcloud = locate_gcloud()?;
    down_gcp_resources(&gcloud, Path::new("kubectl"), root, state, purge, failures)
}

/// Deletes the cloud resources recorded in `state`. Each resource leaves the state file as soon
/// as it is deleted, so after a failure the file lists exactly what still exists. Shared
/// storage (Spanner instance and database, buckets, GKE cluster) is only deleted with `purge`,
/// and only when this environment's `up` created it.
fn down_gcp_resources(
    gcloud: &Path,
    kubectl: &Path,
    root: &Path,
    state: &mut EnvState,
    purge: bool,
    mut failures: Vec<String>,
) -> Result<(), BoxError> {
    let proj_flag = format!("--project={}", state.project_id.clone().unwrap_or_default());

    for svc in state.resources.cloud_run_services.clone() {
        let Some((region, name)) = svc.split_once('/') else {
            continue;
        };
        let result = delete_gcloud_resource(
            gcloud,
            &[
                "run",
                "services",
                "delete",
                name,
                &format!("--region={region}"),
                &proj_flag,
                "--quiet",
            ],
        );
        if deleted(&mut failures, &format!("Cloud Run service {svc}"), result) {
            state.resources.cloud_run_services.retain(|s| s != &svc);
            save_state(root, state)?;
        }
    }

    for vm in state.resources.gce_instances.clone() {
        let Some((zone, name)) = vm.split_once('/') else {
            continue;
        };
        let result = delete_gcloud_resource(
            gcloud,
            &[
                "compute",
                "instances",
                "delete",
                name,
                &format!("--zone={zone}"),
                &proj_flag,
                "--quiet",
            ],
        );
        if deleted(&mut failures, &format!("VM {vm}"), result) {
            state.resources.gce_instances.retain(|v| v != &vm);
            save_state(root, state)?;
        }
    }
    if let Some(fw) = state.resources.gce_firewall_rule.clone() {
        let result = delete_gcloud_resource(
            gcloud,
            &[
                "compute",
                "firewall-rules",
                "delete",
                &fw,
                &proj_flag,
                "--quiet",
            ],
        );
        if deleted(&mut failures, &format!("firewall rule {fw}"), result) {
            state.resources.gce_firewall_rule = None;
            save_state(root, state)?;
        }
    }

    if purge
        && state.resources.gke_cluster_created
        && let (Some(cluster), Some(region)) = (
            state.resources.gke_cluster.clone(),
            state.resources.gke_region.clone(),
        )
    {
        let result = delete_gcloud_resource(
            gcloud,
            &[
                "container",
                "clusters",
                "delete",
                &cluster,
                &format!("--region={region}"),
                &proj_flag,
                "--quiet",
            ],
        );
        if deleted(&mut failures, &format!("GKE cluster {cluster}"), result) {
            // The cluster's workloads went with it.
            if let Some(manifest) = state.resources.kube_manifest.take() {
                let _ = fs::remove_file(manifest);
            }
            state.resources.gke_cluster = None;
            state.resources.gke_region = None;
            state.resources.gke_cluster_created = false;
            state.resources.kube_context = None;
            save_state(root, state)?;
        }
    }
    if let Some(manifest) = state.resources.kube_manifest.clone() {
        // State files from earlier versions did not record the context; derive the one
        // `get-credentials` wrote rather than acting on whatever cluster is current.
        let context = state.resources.kube_context.clone().or_else(|| {
            match (
                &state.project_id,
                &state.resources.gke_region,
                &state.resources.gke_cluster,
            ) {
                (Some(p), Some(r), Some(c)) => Some(gke_kube_context(p, r, c)),
                _ => None,
            }
        });
        let result = match &context {
            Some(ctx) => run_kubectl(
                kubectl,
                &[
                    "--context",
                    ctx,
                    "delete",
                    "--ignore-not-found",
                    "-f",
                    &manifest,
                ],
            ),
            None => Err(
                "no recorded kube context; not running kubectl against the current one".to_string(),
            ),
        };
        if deleted(
            &mut failures,
            &format!("Kubernetes objects in {manifest}"),
            result,
        ) {
            let _ = fs::remove_file(&manifest);
            state.resources.kube_manifest = None;
            save_state(root, state)?;
        }
    }

    if purge {
        for bucket in state.resources.gcs_buckets_created.clone() {
            let result = delete_gcloud_resource(
                gcloud,
                &[
                    "storage",
                    "rm",
                    "--recursive",
                    &format!("gs://{bucket}"),
                    &proj_flag,
                    "--quiet",
                ],
            );
            if deleted(&mut failures, &format!("bucket gs://{bucket}"), result) {
                state.resources.gcs_buckets.retain(|b| b != &bucket);
                state.resources.gcs_buckets_created.retain(|b| b != &bucket);
                save_state(root, state)?;
            }
        }
        if state.resources.spanner_database_created
            && let (Some(db), Some(inst)) = (
                state.resources.spanner_database.clone(),
                state.resources.spanner_instance.clone(),
            )
        {
            let result = delete_gcloud_resource(
                gcloud,
                &[
                    "spanner",
                    "databases",
                    "delete",
                    &db,
                    &format!("--instance={inst}"),
                    &proj_flag,
                    "--quiet",
                ],
            );
            if deleted(&mut failures, &format!("Spanner database {db}"), result) {
                state.resources.spanner_database = None;
                state.resources.spanner_database_created = false;
                save_state(root, state)?;
            }
        }
        // Other environments may have added databases to an instance this one created, so
        // delete the instance only once it is empty.
        if state.resources.spanner_instance_created
            && state.resources.spanner_database.is_none()
            && let Some(inst) = state.resources.spanner_instance.clone()
        {
            let instance_flag = format!("--instance={inst}");
            let listed = run_gcloud(
                gcloud,
                &[
                    "spanner",
                    "databases",
                    "list",
                    &instance_flag,
                    &proj_flag,
                    "--format=value(name)",
                ],
            )
            .map_err(|e| e.to_string());
            match listed {
                Ok(remaining) if remaining.trim().is_empty() => {
                    let result = delete_gcloud_resource(
                        gcloud,
                        &[
                            "spanner",
                            "instances",
                            "delete",
                            &inst,
                            &proj_flag,
                            "--quiet",
                        ],
                    );
                    if deleted(&mut failures, &format!("Spanner instance {inst}"), result) {
                        state.resources.spanner_instance = None;
                        state.resources.spanner_instance_created = false;
                        save_state(root, state)?;
                    }
                }
                Ok(remaining) => {
                    let names: Vec<&str> = remaining
                        .lines()
                        .filter_map(|l| l.rsplit('/').next())
                        .collect();
                    println!(
                        "    Kept Spanner instance {inst}: it still holds databases {}.",
                        names.join(", ")
                    );
                }
                Err(e) => failures.push(format!("Spanner instance {inst}: {e}")),
            }
        }
    }

    report_kept_resources(state, purge);
    teardown_result(root, state, failures)
}

/// Prints the shared resources `down` left in place.
fn report_kept_resources(state: &EnvState, purge: bool) {
    let res = &state.resources;
    let mut created = Vec::new();
    let mut preexisting = Vec::new();
    let mut kept = |was_created: bool, what: String| {
        if was_created {
            created.push(what)
        } else {
            preexisting.push(what)
        }
    };
    if let Some(cluster) = &res.gke_cluster {
        kept(res.gke_cluster_created, format!("GKE cluster {cluster}"));
    }
    if let Some(inst) = &res.spanner_instance {
        kept(
            res.spanner_instance_created,
            format!("Spanner instance {inst}"),
        );
    }
    if let Some(db) = &res.spanner_database {
        kept(
            res.spanner_database_created,
            format!("Spanner database {db}"),
        );
    }
    for bucket in &res.gcs_buckets {
        kept(
            res.gcs_buckets_created.contains(bucket),
            format!("bucket gs://{bucket}"),
        );
    }
    if !purge && !created.is_empty() {
        println!(
            "    Kept {} (pass `--purge` to delete them too).",
            created.join(", ")
        );
    }
    if !preexisting.is_empty() {
        println!(
            "    Kept {}: they existed before `up`, so `down` never deletes them.",
            preexisting.join(", ")
        );
    }
}

fn teardown_result(root: &Path, state: &EnvState, failures: Vec<String>) -> Result<(), BoxError> {
    if failures.is_empty() {
        return Ok(());
    }
    Err(format!(
        "teardown left {} resource(s) running; {} still records them, so rerun `cargo xtask down` after fixing the cause:\n  {}",
        failures.len(),
        state_file_path(root, &state.env_name).display(),
        failures.join("\n  ")
    )
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{load_state, workspace_root};

    #[test]
    fn allow_cidr_rejects_open_world_and_accepts_explicit_ranges() {
        assert!(parse_allow_cidr("0.0.0.0/0").is_err());
        assert!(parse_allow_cidr("::/0").is_err());
        assert!(parse_allow_cidr("203.0.113.5").is_err());
        assert!(parse_allow_cidr("203.0.113.0/33").is_err());
        assert!(parse_allow_cidr("2001:db8::/129").is_err());
        assert!(parse_allow_cidr("not-an-ip/24").is_err());

        assert_eq!(
            parse_allow_cidr("203.0.113.0/24").unwrap(),
            "203.0.113.0/24"
        );
        assert_eq!(
            parse_allow_cidr("198.51.100.7/32").unwrap(),
            "198.51.100.7/32"
        );
        assert_eq!(parse_allow_cidr("2001:db8::/32").unwrap(), "2001:db8::/32");
    }

    #[test]
    fn default_bucket_and_resource_names_include_env_name() {
        assert_eq!(
            blob_bucket_name("example-proj", "default", "us-central1").unwrap(),
            "example-proj-jj-blobs-default-us-central1"
        );
        assert_eq!(
            blob_bucket_name("example-proj", "alice", "europe-west1").unwrap(),
            "example-proj-jj-blobs-alice-europe-west1"
        );
        assert_eq!(env_resource_name("commit-cloud", "default"), "commit-cloud");
        assert_eq!(
            env_resource_name("commit-cloud", "alice"),
            "commit-cloud-alice"
        );

        assert!(validate_gcp_env_name("default").is_ok());
        assert!(validate_gcp_env_name("alice-1").is_ok());
        assert!(validate_gcp_env_name("").is_err());
        assert!(validate_gcp_env_name("Upper").is_err());
        assert!(validate_gcp_env_name("trailing-").is_err());
        assert!(validate_gcp_env_name("way-too-long-environment-name").is_err());

        let t1 = generate_auth_token();
        let t2 = generate_auth_token();
        assert_eq!(t1.len(), 64);
        assert_ne!(t1, t2);
    }

    #[cfg(unix)]
    fn write_executable(path: &Path, script: &str) {
        use std::os::unix::fs::PermissionsExt;
        fs::write(path, script).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn down_purge_only_deletes_created_resources_and_passes_kube_context() {
        let tmp = tempfile::tempdir().unwrap();
        let calls_log = tmp.path().join("calls.log");
        let fake_gcloud = tmp.path().join("fake-gcloud.sh");
        let fake_kubectl = tmp.path().join("fake-kubectl.sh");
        write_executable(
            &fake_gcloud,
            &format!(
                "#!/bin/sh\necho \"gcloud $*\" >> \"{}\"\nexit 0\n",
                calls_log.display()
            ),
        );
        write_executable(
            &fake_kubectl,
            &format!(
                "#!/bin/sh\necho \"kubectl $*\" >> \"{}\"\nexit 0\n",
                calls_log.display()
            ),
        );

        let manifest_path = tmp.path().join("rendered-gke.yaml");
        fs::write(&manifest_path, "kind: Service\n").unwrap();

        let mut st = EnvState::new(
            "alice",
            "gke",
            "gcloud",
            "http://127.0.0.1:8080",
            None,
            AuthMode::None,
        );
        st.project_id = Some("example-proj".to_string());
        st.resources.gke_cluster = Some("preexisting-cluster".to_string());
        st.resources.gke_region = Some("us-central1".to_string());
        st.resources.gke_cluster_created = false;
        st.resources.kube_context =
            Some("gke_example-proj_us-central1_preexisting-cluster".to_string());
        st.resources.kube_manifest = Some(manifest_path.display().to_string());
        st.resources.spanner_instance = Some("preexisting-inst".to_string());
        st.resources.spanner_instance_created = false;
        st.resources.spanner_database = Some("commit-cloud-alice".to_string());
        st.resources.spanner_database_created = true;
        st.resources.gcs_buckets = vec![
            "preexisting-bkt".to_string(),
            "example-proj-jj-blobs-alice-us-central1".to_string(),
        ];
        st.resources.gcs_buckets_created =
            vec!["example-proj-jj-blobs-alice-us-central1".to_string()];
        save_state(tmp.path(), &st).unwrap();

        down_gcp_resources(
            &fake_gcloud,
            &fake_kubectl,
            tmp.path(),
            &mut st,
            true,
            Vec::new(),
        )
        .unwrap();

        let calls = fs::read_to_string(&calls_log).unwrap();
        assert!(
            calls.contains(
                "kubectl --context gke_example-proj_us-central1_preexisting-cluster delete --ignore-not-found -f"
            ),
            "expected explicit --context in kubectl call, got:\n{calls}"
        );
        assert!(
            calls.contains(
                "gcloud spanner databases delete commit-cloud-alice --instance=preexisting-inst"
            ),
            "expected created Spanner database deletion, got:\n{calls}"
        );
        assert!(
            !calls.contains("spanner instances delete"),
            "must never delete pre-existing Spanner instance:\n{calls}"
        );
        assert!(
            !calls.contains("container clusters delete"),
            "must never delete pre-existing GKE cluster:\n{calls}"
        );
        assert!(
            calls.contains("storage rm --recursive gs://example-proj-jj-blobs-alice-us-central1"),
            "expected created bucket deletion, got:\n{calls}"
        );
        assert!(
            !calls.contains("preexisting-bkt"),
            "must never delete pre-existing bucket:\n{calls}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn down_preserves_remaining_state_when_delete_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let fake_gcloud = tmp.path().join("fake-gcloud.sh");
        let fake_kubectl = tmp.path().join("fake-kubectl.sh");
        write_executable(
            &fake_gcloud,
            "#!/bin/sh\ncase \"$*\" in\n  *--region=europe-west1*)\n    echo \"permission denied\" >&2\n    exit 1\n    ;;\n  *)\n    exit 0\n    ;;\nesac\n",
        );
        write_executable(&fake_kubectl, "#!/bin/sh\nexit 0\n");

        let mut st = EnvState::new(
            "alice",
            "cloud-run",
            "gcloud",
            "https://jj-cc.a.run.app",
            None,
            AuthMode::GcloudIdentityToken,
        );
        st.project_id = Some("example-proj".to_string());
        st.resources.cloud_run_services = vec![
            "us-central1/jj-cc-server-alice".to_string(),
            "europe-west1/jj-cc-server-alice".to_string(),
        ];
        save_state(tmp.path(), &st).unwrap();

        let err = down_gcp_resources(
            &fake_gcloud,
            &fake_kubectl,
            tmp.path(),
            &mut st,
            false,
            Vec::new(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("europe-west1/jj-cc-server-alice"),
            "unexpected err: {err}"
        );

        let persisted = load_state(tmp.path(), "alice")
            .unwrap()
            .expect("state file must remain");
        assert_eq!(
            persisted.resources.cloud_run_services,
            vec!["europe-west1/jj-cc-server-alice".to_string()]
        );
    }

    #[test]
    fn manifests_and_dockerfile_follow_security_and_storage_invariants() {
        let root = workspace_root();

        let gke = fs::read_to_string(root.join("deploy/k8s/gcp/gke.yaml")).unwrap();
        assert!(gke.contains("type: ClusterIP"));
        assert!(!gke.contains("LoadBalancer"));
        assert!(gke.contains("name: BLOB_STORE\n              value: \"gcs\""));

        let compose = fs::read_to_string(root.join("compose.yaml")).unwrap();
        let mut in_ports = false;
        let mut checked_ports = 0usize;
        for line in compose.lines() {
            let trimmed = line.trim();
            if trimmed == "ports:" {
                in_ports = true;
                continue;
            }
            if in_ports {
                if let Some(item) = trimmed.strip_prefix("- ") {
                    let unquoted = item.trim_matches('"');
                    assert!(
                        unquoted.starts_with("127.0.0.1:"),
                        "compose port binding must be loopback-bound: {trimmed}"
                    );
                    checked_ports += 1;
                } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
                    in_ports = false;
                }
            }
        }
        assert!(
            checked_ports >= 7,
            "expected at least 7 port bindings, got {checked_ports}"
        );
        assert!(compose.contains("GCS_PEER_BUCKETS: \"gs://blobs-ew1\""));
        assert!(compose.contains("GCS_PEER_BUCKETS: \"gs://blobs-uc1\""));

        for rel in [
            "deploy/k8s/memory.yaml",
            "deploy/k8s/sqlite.yaml",
            "deploy/k8s/spanner.yaml",
            "deploy/k8s/multiregion.yaml",
        ] {
            let body = fs::read_to_string(root.join(rel)).unwrap();
            let host_ports = body
                .lines()
                .filter(|l| l.trim_start().starts_with("hostPort:"))
                .count();
            let loopback_ips = body
                .lines()
                .filter(|l| l.trim() == "hostIP: 127.0.0.1")
                .count();
            assert_eq!(
                host_ports, loopback_ips,
                "{rel} must set hostIP: 127.0.0.1 on every hostPort"
            );
        }
        let multiregion = fs::read_to_string(root.join("deploy/k8s/multiregion.yaml")).unwrap();
        assert!(multiregion.contains("value: \"gs://blobs-ew1\""));
        assert!(
            multiregion.contains("name: GCS_PEER_BUCKETS\n          value: \"gs://blobs-ew1\"")
        );

        let dockerfile = fs::read_to_string(root.join("Dockerfile")).unwrap();
        assert!(!dockerfile.contains(".fingerprint/cc-*"));
        assert!(dockerfile.contains(".fingerprint/cc-common-*"));
        assert!(dockerfile.contains(".fingerprint/cc-lib-*"));
        assert!(dockerfile.contains(
            "ENV STORE_TYPE=sqlite SQLITE_PATH=/data/commit_cloud.db BLOB_PATH=/data/blobs"
        ));
    }
}
