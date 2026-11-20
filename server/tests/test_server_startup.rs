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

use cc_common::backend::RegisterRepositoryRequest;
use cc_common::backend::backend_service_client::BackendServiceClient;
use cc_common::op_store::GetOpHeadsRequest;
use cc_common::op_store::op_store_service_client::OpStoreServiceClient;
use tonic::{Code, Request};
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_client::HealthClient;

#[tokio::test]
async fn test_server_startup_succeeds() {
    let server = testutils::spawn_server().await;

    let channel = tonic::transport::Endpoint::from_shared(server.url().to_string())
        .expect("The gRPC endpoint URI should be valid")
        .connect()
        .await
        .expect("The gRPC connection to the health service should have succeeded");

    let mut client = HealthClient::new(channel);

    let response = client
        .check(HealthCheckRequest {
            service: "".to_string(),
        })
        .await
        .expect("The gRPC health check request should have succeeded");

    assert_eq!(
        response.into_inner().status(),
        tonic_health::pb::health_check_response::ServingStatus::Serving,
        "Server is not in SERVING state"
    );
}

#[tokio::test]
async fn test_server_bearer_auth_enforces_tokens_while_keeping_health_open() {
    let server = testutils::spawn_server_with_env(
        &[("JJ_CC_SERVER_AUTH_TOKEN", "active-token-1,rotated-token-2")],
        &[],
    )
    .await;

    let channel = tonic::transport::Endpoint::from_shared(server.url().to_string())
        .expect("valid endpoint")
        .connect()
        .await
        .expect("connect succeeded");

    // Kubernetes / Cloud Run readiness probe stays unauthenticated.
    let mut health = HealthClient::new(channel.clone());
    let health_res = health
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .expect("health check should succeed without auth token");
    assert_eq!(
        health_res.into_inner().status(),
        tonic_health::pb::health_check_response::ServingStatus::Serving
    );

    // Unauthenticated BackendService and OpStoreService calls are rejected.
    let mut backend = BackendServiceClient::new(channel.clone());
    let unauth_err = backend
        .register_repository(Request::new(RegisterRepositoryRequest {
            project_id: String::new(),
            repo_id: None,
            name: None,
            hash_algorithm: 0,
        }))
        .await
        .expect_err("unauthenticated RegisterRepository must fail");
    assert_eq!(unauth_err.code(), Code::Unauthenticated);

    let mut op_store = OpStoreServiceClient::new(channel);
    let mut bad_req = Request::new(GetOpHeadsRequest {
        repo_id: "any".to_string(),
    });
    bad_req
        .metadata_mut()
        .insert("authorization", "Bearer wrong-token".parse().unwrap());
    let bad_err = op_store
        .get_op_heads(bad_req)
        .await
        .expect_err("wrong Bearer token must fail");
    assert_eq!(bad_err.code(), Code::Unauthenticated);

    // Both active and rotated tokens succeed.
    let mut req1 = Request::new(RegisterRepositoryRequest {
        project_id: String::new(),
        repo_id: None,
        name: None,
        hash_algorithm: 0,
    });
    req1.metadata_mut()
        .insert("authorization", "Bearer active-token-1".parse().unwrap());
    let repo_id = backend
        .register_repository(req1)
        .await
        .expect("primary token should succeed")
        .into_inner()
        .repo_id;

    // SHA-256 registration is explicitly rejected with UNIMPLEMENTED in M1.
    let mut sha256_req = Request::new(RegisterRepositoryRequest {
        project_id: String::new(),
        repo_id: None,
        name: None,
        hash_algorithm: cc_common::backend::HashAlgorithm::Sha256 as i32,
    });
    sha256_req
        .metadata_mut()
        .insert("authorization", "Bearer active-token-1".parse().unwrap());
    let sha256_err = backend
        .register_repository(sha256_req)
        .await
        .expect_err("SHA-256 repository registration must return UNIMPLEMENTED in M1");
    assert_eq!(sha256_err.code(), Code::Unimplemented);

    let mut req2 = Request::new(GetOpHeadsRequest { repo_id });
    req2.metadata_mut()
        .insert("authorization", "Bearer rotated-token-2".parse().unwrap());
    let heads = op_store
        .get_op_heads(req2)
        .await
        .expect("rotated token should succeed")
        .into_inner()
        .op_head_ids;
    assert_eq!(heads.len(), 1);
}

fn run_server_cli(env_vars: &[(&str, &str)], args: &[&str]) -> std::process::Output {
    let mut cmd = std::process::Command::new(testutils::server_binary_path());
    for var in testutils::SANITIZED_SERVER_ENV_VARS {
        cmd.env_remove(var);
    }
    for (k, v) in env_vars {
        cmd.env(k, v);
    }
    cmd.args(args)
        .output()
        .expect("jj-cc-server binary should execute")
}

#[test]
fn test_server_spanner_refuses_to_start_without_explicit_blob_store() {
    let db_arg = "--spanner-database=projects/p/instances/i/databases/d";

    let out = run_server_cli(&[], &["--port=0", "--store-type=spanner", db_arg]);
    assert!(
        !out.status.success(),
        "spanner without blob store must fail at startup"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("requires an explicit blob store"),
        "unexpected stderr: {stderr}"
    );

    // Empty env var placeholders must not bypass the durable blob store requirement.
    let out_empty_env = run_server_cli(
        &[("GCS_BUCKET", ""), ("BLOB_PATH", "")],
        &["--port=0", "--store-type=spanner", db_arg],
    );
    assert!(!out_empty_env.status.success());
}

#[test]
fn test_server_auth_rejects_empty_tokens_and_hides_env_in_help() {
    let secret = "super-secret-env-token-xyz";
    let help = run_server_cli(&[("JJ_CC_SERVER_AUTH_TOKEN", secret)], &["--help"]);
    assert!(help.status.success());
    let help_stdout = String::from_utf8_lossy(&help.stdout);
    let help_stderr = String::from_utf8_lossy(&help.stderr);
    assert!(
        !help_stdout.contains(secret) && !help_stderr.contains(secret),
        "--help leaked JJ_CC_SERVER_AUTH_TOKEN value"
    );

    for bad_env in ["", " , "] {
        let out = run_server_cli(&[("JJ_CC_SERVER_AUTH_TOKEN", bad_env)], &["--port=0"]);
        assert!(
            !out.status.success(),
            "empty JJ_CC_SERVER_AUTH_TOKEN={bad_env:?} must be rejected"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("contains no non-empty token"),
            "unexpected stderr: {stderr}"
        );
    }

    for bad_arg in ["--auth-token=", "--auth-token= , "] {
        let out = run_server_cli(&[], &["--port=0", bad_arg]);
        assert!(!out.status.success(), "{bad_arg:?} must be rejected");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("contains no non-empty token"),
            "unexpected stderr: {stderr}"
        );
    }
}
