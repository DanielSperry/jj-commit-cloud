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

//! Optional constant-time Bearer token authentication interceptor for `jj-cc-server`.
//!
//! # Authority and Crash-Safety Ordering
//! `BearerAuthInterceptor` runs at the `tonic` service boundary before any `CompositeStore`,
//! `MetadataStore`, or `BlobStore` read or write executes, rejecting unauthenticated
//! requests with `Code::Unauthenticated` prior to database or object-storage access.
//!
//! # Concurrency and Merge Invariants
//! Supports multiple active tokens simultaneously (comma-separated or repeated
//! `--auth-token` flags) so operators can rotate credentials across fleets of agents
//! without downtime. Token comparison inspects every configured token in constant time
//! relative to candidate byte length.
//!
//! # Cross-Process and RPC Boundaries
//! Applied to `BackendService`, `OpStoreService`, and `WorkspaceService` while leaving
//! `tonic_health` open so Kubernetes, GKE, GCE, and Cloud Run readiness probes succeed without
//! credentials.

use std::sync::Arc;

use tonic::{Request, Status, service::Interceptor};

/// Constant-time gRPC Bearer token validator.
///
/// When `tokens` is `None` (no `--auth-token` or `JJ_CC_SERVER_AUTH_TOKEN` configured),
/// all requests pass through unchanged so local development and IAM/IAP-terminated
/// deployments require zero extra configuration.
#[derive(Clone, Default)]
pub struct BearerAuthInterceptor {
    tokens: Option<Arc<Vec<Vec<u8>>>>,
}

/// Prints the number of configured tokens, never their values.
impl std::fmt::Debug for BearerAuthInterceptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let token_count = self.tokens.as_ref().map_or(0, |tokens| tokens.len());
        f.debug_struct("BearerAuthInterceptor")
            .field("token_count", &token_count)
            .finish()
    }
}

impl BearerAuthInterceptor {
    pub fn from_tokens<I, S>(raw_tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let normalized: Vec<Vec<u8>> = raw_tokens
            .into_iter()
            .flat_map(|s| {
                s.as_ref()
                    .split(',')
                    .map(|part| part.trim().as_bytes().to_vec())
                    .collect::<Vec<_>>()
            })
            .filter(|b| !b.is_empty())
            .collect();

        if normalized.is_empty() {
            Self { tokens: None }
        } else {
            Self {
                tokens: Some(Arc::new(normalized)),
            }
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.tokens.is_some()
    }

    pub fn validate_request<T>(&self, req: Request<T>) -> Result<Request<T>, Status> {
        let Some(expected_tokens) = &self.tokens else {
            return Ok(req);
        };

        let raw_header = req
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| Status::unauthenticated("missing authorization metadata header"))?;

        let provided = extract_bearer_token(raw_header).ok_or_else(|| {
            Status::unauthenticated("authorization header must use Bearer <token> scheme")
        })?;

        let mut matched = 0u8;
        for expected in expected_tokens.iter() {
            matched |= constant_time_eq(provided.as_bytes(), expected);
        }

        if matched == 1 {
            Ok(req)
        } else {
            Err(Status::unauthenticated("invalid bearer token"))
        }
    }
}

impl Interceptor for BearerAuthInterceptor {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        self.validate_request(request)
    }
}

fn extract_bearer_token(header_val: &str) -> Option<&str> {
    // Authentication schemes are case-insensitive (RFC 7235 section 2.1).
    let (scheme, rest) = header_val.trim().split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let token = rest.trim();
    (!token.is_empty()).then_some(token)
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> u8 {
    if a.len() != b.len() {
        return 0;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    u8::from(diff == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::Code;

    #[test]
    fn disabled_interceptor_allows_unauthenticated_requests() {
        let mut auth = BearerAuthInterceptor::from_tokens(Vec::<String>::new());
        assert!(!auth.is_enabled());
        assert!(auth.call(Request::new(())).is_ok());
    }

    #[test]
    fn enabled_interceptor_enforces_tokens_and_supports_rotation() {
        let mut auth = BearerAuthInterceptor::from_tokens([" token-v1 ", "token-v2"]);
        assert!(auth.is_enabled());

        let err = auth.call(Request::new(())).unwrap_err();
        assert_eq!(err.code(), Code::Unauthenticated);

        let mut wrong = Request::new(());
        wrong
            .metadata_mut()
            .insert("authorization", "Bearer wrong-secret".parse().unwrap());
        assert_eq!(auth.call(wrong).unwrap_err().code(), Code::Unauthenticated);

        for valid_token in ["token-v1", "token-v2"] {
            let mut ok_req = Request::new(());
            ok_req.metadata_mut().insert(
                "authorization",
                format!("Bearer {valid_token}").parse().unwrap(),
            );
            assert!(auth.call(ok_req).is_ok());
        }
    }

    #[test]
    fn bearer_scheme_is_case_insensitive() {
        let mut auth = BearerAuthInterceptor::from_tokens(["secret-token"]);
        for header in [
            "Bearer secret-token",
            "bearer secret-token",
            "BEARER  secret-token",
        ] {
            let mut req = Request::new(());
            req.metadata_mut()
                .insert("authorization", header.parse().unwrap());
            assert!(auth.call(req).is_ok(), "{header}");
        }
        for header in ["Basic secret-token", "Bearersecret-token", "Bearer "] {
            let mut req = Request::new(());
            req.metadata_mut()
                .insert("authorization", header.parse().unwrap());
            assert_eq!(
                auth.call(req).unwrap_err().code(),
                Code::Unauthenticated,
                "{header}"
            );
        }
    }

    #[test]
    fn whitespace_only_tokens_leave_auth_disabled_and_debug_hides_tokens() {
        assert!(!BearerAuthInterceptor::from_tokens(["", " ", " , "]).is_enabled());

        let auth = BearerAuthInterceptor::from_tokens(["super-secret-value"]);
        let debug = format!("{auth:?}");
        assert!(!debug.contains("super-secret-value"), "{debug}");
        assert!(debug.contains("token_count: 1"), "{debug}");
    }
}
