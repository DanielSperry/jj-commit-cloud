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
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum AuthMode {
    None,
    GcloudIdentityToken,
}

/// A background process started by `xtask up` (native server, emulator, IAP tunnel, or
/// `kubectl port-forward`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManagedProcess {
    pub pid: u32,
    /// Text that must appear in the process command line before `down` signals `pid`, so a PID
    /// that an unrelated process reused after a crash or reboot is never killed.
    pub command_marker: String,
}

impl ManagedProcess {
    pub fn new(pid: u32, command_marker: &str) -> Self {
        Self {
            pid,
            command_marker: command_marker.to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Endpoints {
    pub primary: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub peer: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManagedResources {
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub containers: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub volumes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub network: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub kube_manifest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub kind_cluster: Option<String>,
    /// Kubernetes context `up` applied manifests to; `down` passes it to every `kubectl` call.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub kube_context: Option<String>,
    /// Bare PIDs written by earlier xtask versions, before `processes` recorded a marker.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub pids: Vec<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub processes: Vec<ManagedProcess>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub data_dirs: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub cloud_run_services: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub gce_instances: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub gce_firewall_rule: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub gke_cluster: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub gke_region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub spanner_instance: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub spanner_database: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub gcs_buckets: Vec<String>,
    // `down --purge` deletes shared storage only when this environment's `up` created it. The
    // flags default to false, so state files from earlier versions never delete storage.
    #[serde(skip_serializing_if = "std::ops::Not::not", default)]
    pub gke_cluster_created: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not", default)]
    pub spanner_instance_created: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not", default)]
    pub spanner_database_created: bool,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub gcs_buckets_created: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnvState {
    pub env_name: String,
    pub mode: String,
    pub runtime: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub project_id: Option<String>,
    pub endpoints: Endpoints,
    pub auth: AuthMode,
    /// Bearer token `up` configured on the server (`JJ_CC_SERVER_AUTH_TOKEN`); `status` and
    /// `smoke` send it.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub auth_token: Option<String>,
    pub created_unix_secs: u64,
    pub resources: ManagedResources,
}

impl EnvState {
    pub fn new(
        env_name: &str,
        mode: &str,
        runtime: &str,
        primary_url: &str,
        peer_url: Option<String>,
        auth: AuthMode,
    ) -> Self {
        let created_unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            env_name: env_name.to_string(),
            mode: mode.to_string(),
            runtime: runtime.to_string(),
            project_id: None,
            endpoints: Endpoints {
                primary: primary_url.to_string(),
                peer: peer_url,
            },
            auth,
            auth_token: None,
            created_unix_secs,
            resources: ManagedResources::default(),
        }
    }
}

pub fn workspace_root() -> PathBuf {
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(manifest_dir);
        if let Some(parent) = p.parent() {
            return parent.to_path_buf();
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

pub fn cargo_target_dir(root: &Path) -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target"))
}

pub fn state_file_path(root: &Path, env_name: &str) -> PathBuf {
    let dir = root.join(".jj-cc");
    if env_name == "default" {
        dir.join("state.json")
    } else {
        dir.join(format!("{env_name}.json"))
    }
}

pub fn save_state(root: &Path, state: &EnvState) -> Result<PathBuf, BoxError> {
    let path = state_file_path(root, &state.env_name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(state)?;
    write_private_file(&path, format!("{body}\n").as_bytes())?;
    Ok(path)
}

/// Writes `contents` so only the current user can read it, because state files can hold a
/// server bearer token.
fn write_private_file(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    // `mode` only applies to new files; also tighten files written by earlier versions.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(contents)
}

pub fn load_state(root: &Path, env_name: &str) -> Result<Option<EnvState>, BoxError> {
    let path = state_file_path(root, env_name);
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&path)?;
    let parsed: EnvState = serde_json::from_str(&raw)?;
    Ok(Some(parsed))
}

pub fn remove_state(root: &Path, env_name: &str) -> Result<(), BoxError> {
    let path = state_file_path(root, env_name);
    if path.exists() {
        fs::remove_file(&path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_env_uses_state_json_and_roundtrips() {
        let tmp = tempfile::tempdir().unwrap();
        let mut st = EnvState::new(
            "default",
            "multiregion",
            "podman",
            "http://127.0.0.1:8081",
            Some("http://127.0.0.1:8082".to_string()),
            AuthMode::None,
        );
        st.resources.containers = vec!["jj-cc-default-us".to_string()];

        let saved_path = save_state(tmp.path(), &st).unwrap();
        assert_eq!(saved_path, tmp.path().join(".jj-cc").join("state.json"));

        let loaded = load_state(tmp.path(), "default").unwrap().unwrap();
        assert_eq!(loaded, st);

        remove_state(tmp.path(), "default").unwrap();
        assert!(load_state(tmp.path(), "default").unwrap().is_none());
    }

    #[test]
    fn named_env_uses_dedicated_file() {
        let tmp = tempfile::tempdir().unwrap();
        let st = EnvState::new(
            "gcp",
            "cloud-run",
            "gcloud",
            "https://jj-cc-server.a.run.app",
            None,
            AuthMode::GcloudIdentityToken,
        );
        let saved_path = save_state(tmp.path(), &st).unwrap();
        assert_eq!(saved_path, tmp.path().join(".jj-cc").join("gcp.json"));
        assert_eq!(load_state(tmp.path(), "gcp").unwrap(), Some(st));
    }

    #[test]
    fn ownership_processes_and_token_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let mut st = EnvState::new(
            "gce",
            "gce",
            "gcloud",
            "http://203.0.113.7:8080",
            None,
            AuthMode::None,
        );
        st.auth_token = Some("secret-token".to_string());
        st.resources.processes = vec![ManagedProcess {
            pid: 4242,
            command_marker: "start-iap-tunnel jj-cc-gce-us-central1".to_string(),
        }];
        st.resources.kube_context = Some("gke_p_us-central1_jj-cc-cluster-gce".to_string());
        st.resources.spanner_instance = Some("jj-cc-dev".to_string());
        st.resources.spanner_database = Some("commit-cloud-gce".to_string());
        st.resources.spanner_database_created = true;
        st.resources.gcs_buckets = vec!["shared".to_string(), "mine".to_string()];
        st.resources.gcs_buckets_created = vec!["mine".to_string()];
        save_state(tmp.path(), &st).unwrap();
        assert_eq!(load_state(tmp.path(), "gce").unwrap(), Some(st));
    }

    #[test]
    fn legacy_state_without_ownership_fields_never_claims_storage() {
        let tmp = tempfile::tempdir().unwrap();
        let legacy = r#"{
            "env_name": "old",
            "mode": "cloud-run",
            "runtime": "gcloud",
            "project_id": "p",
            "endpoints": { "primary": "https://jj-cc-server.a.run.app" },
            "auth": "gcloud-identity-token",
            "created_unix_secs": 1,
            "resources": {
                "pids": [123],
                "spanner_instance": "jj-cc-dev",
                "spanner_database": "commit-cloud",
                "gcs_buckets": ["p-jj-blobs-us-central1"]
            }
        }"#;
        fs::create_dir_all(tmp.path().join(".jj-cc")).unwrap();
        fs::write(state_file_path(tmp.path(), "old"), legacy).unwrap();
        let st = load_state(tmp.path(), "old").unwrap().unwrap();
        assert_eq!(st.resources.pids, vec![123]);
        assert!(st.resources.processes.is_empty());
        assert!(!st.resources.spanner_instance_created);
        assert!(!st.resources.spanner_database_created);
        assert!(st.resources.gcs_buckets_created.is_empty());
        assert_eq!(st.auth_token, None);
    }

    #[cfg(unix)]
    #[test]
    fn state_file_is_readable_only_by_owner() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let path = state_file_path(tmp.path(), "default");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "{}").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();

        let st = EnvState::new("default", "gce", "gcloud", "http://x", None, AuthMode::None);
        save_state(tmp.path(), &st).unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
