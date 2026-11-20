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

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};

pub struct ServerGuard {
    child: Child,
    url: String,
}

impl ServerGuard {
    pub fn url(&self) -> &str {
        &self.url
    }
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

fn extract_listening_address(line: &str) -> Option<String> {
    let marker = "listening on ";
    if let Some(idx) = line.find(marker) {
        let addr_part = &line[idx + marker.len()..];
        let addr = addr_part.trim().to_string();
        if !addr.is_empty() {
            return Some(addr);
        }
    }
    None
}

pub fn server_binary_path() -> PathBuf {
    PathBuf::from(env!("JJ_CC_SERVER_BIN"))
}

pub async fn spawn_server() -> ServerGuard {
    spawn_server_with_args(&[]).await
}

pub async fn spawn_server_with_args(extra_args: &[&str]) -> ServerGuard {
    spawn_server_with_env(&[], extra_args).await
}

pub const SANITIZED_SERVER_ENV_VARS: &[&str] = &[
    "HOST",
    "PORT",
    "STORE_TYPE",
    "BLOB_STORE",
    "INLINE_BLOB_THRESHOLD_BYTES",
    "SQLITE_PATH",
    "BLOB_PATH",
    "BLOB_LOCATION_ID",
    "LOCAL_PEER_BLOB_DIRS",
    "SPANNER_DATABASE",
    "SPANNER_ENDPOINT",
    "SPANNER_EMULATOR_HOST",
    "SPANNER_EMULATOR_REST_HOST",
    "SPANNER_AUTO_MIGRATE",
    "GCS_BUCKET",
    "GCS_PEER_BUCKETS",
    "STORAGE_EMULATOR_HOST",
    "JJ_CC_SERVER_AUTH_TOKEN",
];

pub async fn spawn_server_with_env(env_vars: &[(&str, &str)], extra_args: &[&str]) -> ServerGuard {
    let server_binary_path = server_binary_path();

    let mut cmd = Command::new(server_binary_path);
    cmd.arg("--port=0");
    for var in SANITIZED_SERVER_ENV_VARS {
        cmd.env_remove(var);
    }
    for (k, v) in env_vars {
        cmd.env(k, v);
    }
    for arg in extra_args {
        cmd.arg(arg);
    }

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("The jj-cc-server process should have spawned");

    let stdout = child
        .stdout
        .take()
        .expect("The stdout pipe should have been captured from the child process");

    let mut reader = BufReader::new(stdout).lines();

    let timeout = tokio::time::sleep(Duration::from_secs(10));
    tokio::pin!(timeout);

    let server_addr = loop {
        tokio::select! {
            line_result = reader.next_line() => {
                match line_result {
                    Ok(Some(line)) => {
                        if let Some(addr) = extract_listening_address(&line) {
                            break addr;
                        }
                    }
                    Ok(None) => panic!("The stdout pipe should not have closed before the server listening address was printed"),
                    Err(e) => panic!("A line from the server stdout should have been read: {:?}", e),
                }
            }
            _ = &mut timeout => {
                panic!("The server listening address should have appeared in stdout before timeout");
            }
        }
    };

    // Keep draining stdout in the background so the child process never blocks on a full pipe.
    tokio::spawn(async move { while let Ok(Some(_)) = reader.next_line().await {} });

    let url = format!("http://{}", server_addr);

    ServerGuard { child, url }
}

pub async fn spawn_sqlite_server(db_path: &Path) -> ServerGuard {
    let sqlite_arg = format!("--sqlite-path={}", db_path.display());
    spawn_server_with_args(&["--store-type=sqlite", &sqlite_arg]).await
}

fn allocate_ephemeral_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("ephemeral port allocation should succeed")
        .local_addr()
        .expect("local_addr should resolve")
        .port()
}

fn locate_spanner_emulator_binaries() -> Option<(PathBuf, PathBuf)> {
    if let (Ok(gw), Ok(grpc)) = (
        std::env::var("SPANNER_EMULATOR_GATEWAY_BIN"),
        std::env::var("SPANNER_EMULATOR_GRPC_BIN"),
    ) {
        let gw_path = PathBuf::from(gw);
        let grpc_path = PathBuf::from(grpc);
        if gw_path.exists() && grpc_path.exists() {
            return Some((gw_path, grpc_path));
        }
    }

    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let gw = dir.join("gateway_main");
            let grpc = dir.join("emulator_main");
            if gw.exists() && grpc.exists() {
                return Some((gw, grpc));
            }
        }
    }

    let release_dir = Path::new("/google/bin/releases/spanner/cloud/emulator/daily");
    let gw = release_dir.join("gateway_main");
    let grpc = release_dir.join("emulator_main");
    if gw.exists() && grpc.exists() {
        return Some((gw, grpc));
    }

    None
}

async fn send_http_json(
    host_port: &str,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect(host_port).await?;
    let payload = body.unwrap_or("");
    let request = format!(
        "{method} {path} HTTP/1.0\r\nHost: {host_port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
        payload.len()
    );
    stream.write_all(request.as_bytes()).await?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await?;
    let response = String::from_utf8_lossy(&raw).into_owned();

    let status_code = response
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .unwrap_or(0);
    Ok((status_code, response))
}

/// Manages a hermetic Cloud Spanner Emulator (`gateway_main` + `emulator_main`)
/// instance on ephemeral ports (or attaches to an existing `SPANNER_EMULATOR_HOST`).
pub struct SpannerEmulatorGuard {
    child: Option<Child>,
    grpc_host_port: String,
    rest_url: String,
    project_id: String,
    instance_id: String,
}

impl SpannerEmulatorGuard {
    /// Starts an emulator like [`Self::start`], or prints a skip notice and returns `None`
    /// when none is available. With `JJ_CC_REQUIRE_SPANNER=1` a missing emulator fails the
    /// test instead, so CI jobs that provide the emulator cannot pass without running it.
    pub async fn start_or_skip(test_name: &str, enable_fault_injection: bool) -> Option<Self> {
        if let Some(guard) = Self::start(enable_fault_injection).await {
            return Some(guard);
        }
        on_missing_spanner_emulator(
            test_name,
            std::env::var("JJ_CC_REQUIRE_SPANNER").ok().as_deref(),
        );
        None
    }

    pub async fn start(enable_fault_injection: bool) -> Option<Self> {
        let project_id = "test-project".to_string();
        let instance_id = "test-instance".to_string();

        if !enable_fault_injection
            && let Ok(existing_grpc) = std::env::var("SPANNER_EMULATOR_HOST")
            && !existing_grpc.trim().is_empty()
        {
            let grpc_host_port = existing_grpc
                .trim()
                .trim_start_matches("http://")
                .to_string();
            let rest_host_port = std::env::var("SPANNER_EMULATOR_REST_HOST")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim_start_matches("http://").to_string())
                .unwrap_or_else(|| {
                    grpc_host_port
                        .strip_suffix(":9010")
                        .map(|h| format!("{h}:9020"))
                        .unwrap_or_else(|| grpc_host_port.clone())
                });
            let guard = Self {
                child: None,
                grpc_host_port,
                rest_url: format!("http://{rest_host_port}"),
                project_id,
                instance_id,
            };
            guard.ensure_instance(&rest_host_port).await;
            return Some(guard);
        }

        let (gateway_bin, grpc_bin) = locate_spanner_emulator_binaries()?;
        let grpc_port = allocate_ephemeral_port();
        let http_port = allocate_ephemeral_port();
        let grpc_host_port = format!("127.0.0.1:{grpc_port}");
        let rest_host_port = format!("127.0.0.1:{http_port}");

        let mut cmd = Command::new(gateway_bin);
        cmd.arg(format!("-grpc_binary={}", grpc_bin.display()))
            .arg("-hostname=127.0.0.1")
            .arg(format!("-grpc_port={grpc_port}"))
            .arg(format!("-http_port={http_port}"))
            .arg("-disable_switch_user_and_group=true");
        if enable_fault_injection {
            cmd.arg("-enable_fault_injection=true");
        }

        let child = cmd
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("Cloud Spanner emulator gateway_main should spawn");

        let guard = Self {
            child: Some(child),
            grpc_host_port,
            rest_url: format!("http://{rest_host_port}"),
            project_id,
            instance_id,
        };

        let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
        loop {
            if let Ok((status, _)) = send_http_json(
                &rest_host_port,
                "GET",
                "/v1/projects/test-project/instances",
                None,
            )
            .await
                && status == 200
            {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!("Timed out waiting for Cloud Spanner Emulator REST port {rest_host_port}");
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        guard.ensure_instance(&rest_host_port).await;
        Some(guard)
    }

    async fn ensure_instance(&self, rest_host_port: &str) {
        let create_body = format!(
            r#"{{"instanceId":"{inst}","instance":{{"config":"projects/{proj}/instanceConfigs/emulator-config","displayName":"{inst}","nodeCount":1}}}}"#,
            inst = self.instance_id,
            proj = self.project_id,
        );
        let path = format!("/v1/projects/{}/instances", self.project_id);
        let (status, resp) = send_http_json(rest_host_port, "POST", &path, Some(&create_body))
            .await
            .expect("POST /instances to Spanner emulator should succeed");
        // 200 OK (created) or 409 Conflict (already exists)
        assert!(
            status == 200 || status == 409,
            "Unexpected Spanner emulator create_instance status {status}: {resp}"
        );
    }

    pub fn grpc_host_port(&self) -> &str {
        &self.grpc_host_port
    }

    pub fn grpc_endpoint(&self) -> String {
        format!("http://{}", self.grpc_host_port)
    }

    pub fn rest_endpoint(&self) -> &str {
        &self.rest_url
    }

    pub fn database_uri(&self, db_id: &str) -> String {
        format!(
            "projects/{}/instances/{}/databases/{db_id}",
            self.project_id, self.instance_id
        )
    }
}

impl Drop for SpannerEmulatorGuard {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.start_kill();
        }
    }
}

fn on_missing_spanner_emulator(test_name: &str, require_spanner_env: Option<&str>) {
    let hint = "set SPANNER_EMULATOR_HOST, put gateway_main and emulator_main on PATH, or set \
                SPANNER_EMULATOR_GATEWAY_BIN and SPANNER_EMULATOR_GRPC_BIN";
    if require_spanner_env == Some("1") {
        panic!("{test_name}: JJ_CC_REQUIRE_SPANNER=1 but no Cloud Spanner Emulator found; {hint}");
    }
    // Write to the stderr handle: the test harness captures `eprintln!` of passing tests.
    let _ = writeln!(
        std::io::stderr(),
        "SKIPPED {test_name}: no Cloud Spanner Emulator found; {hint}, or set \
         JJ_CC_REQUIRE_SPANNER=1 to fail instead of skipping"
    );
}

/// Struct to hold the jj-cc-server instance and the temporary test directory where changes are made
/// to the working directory for cli integration tests
pub struct TestWorkspace {
    server: ServerGuard,
    temp_dir: tempfile::TempDir,
}

impl TestWorkspace {
    /// Spawns a new dynamic-port `jj-cc-server` instance and initializes a temporary
    /// Commit Cloud repository workspace using `jj cc init`.
    pub async fn init() -> Self {
        let server = spawn_server().await;
        Self::init_against_server(server).await
    }

    pub async fn init_sqlite(db_path: &Path) -> Self {
        let server = spawn_sqlite_server(db_path).await;
        Self::init_against_server(server).await
    }

    pub async fn init_against_server(server: ServerGuard) -> Self {
        let temp_dir =
            tempfile::tempdir().expect("temporary directory should have been created for testing");
        let repo_path = temp_dir.path();

        let mut init_cmd =
            assert_cmd::Command::cargo_bin("jj").expect("The jj CLI binary should have compiled");

        init_cmd.current_dir(repo_path).args([
            "cc",
            "init",
            "--server",
            server.url(),
            "--create",
            ".",
        ]);

        init_cmd.assert().success();

        TestWorkspace { server, temp_dir }
    }

    pub fn repo_path(&self) -> &Path {
        self.temp_dir.path()
    }

    pub fn server_url(&self) -> &str {
        self.server.url()
    }

    pub fn jj_cmd(&self) -> assert_cmd::Command {
        let mut cmd =
            assert_cmd::Command::cargo_bin("jj").expect("The jj CLI binary should have compiled");
        cmd.current_dir(self.repo_path());
        cmd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "JJ_CC_REQUIRE_SPANNER=1")]
    fn require_spanner_env_fails_when_emulator_is_missing() {
        on_missing_spanner_emulator("sample_spanner_test", Some("1"));
    }

    #[test]
    fn unset_require_spanner_env_skips_without_panicking() {
        on_missing_spanner_emulator("sample_spanner_test", None);
        on_missing_spanner_emulator("sample_spanner_test", Some("0"));
    }
}
