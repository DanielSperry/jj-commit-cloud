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

//! Client-side `jj_lib` storage trait implementations (`Backend`, `OpStore`, and
//! `OpHeadsStore`) backed by remote jj-commit-cloud gRPC services.
//!
//! # Authority and Crash-Safety Ordering
//! `jj_lib` transactions persist file blobs, symlinks, trees, and commits through
//! [`cc_backend::CommitCloudBackend`] before writing `View` and `Operation` nodes via
//! [`cc_op_store::CommitCloudOpStore`] and advancing heads in
//! [`cc_op_heads_store::CommitCloudOpHeadsStore`]. Interrupted client mutations leave
//! unreferenced content-addressed objects on the server without corrupting active heads.
//!
//! # Concurrency and Merge Invariants
//! `CommitCloudOpHeadsStore` uses lock-free optimistic updates (`UpdateOpHeads`). When
//! concurrent workspaces advance from the same parent operation, the server retains
//! both sibling heads and `jj_lib` reconciles them via 3-way view merging on next load.
//!
//! # Cross-Process and RPC Boundaries
//! Each store reads `.jj/repo/store/config.toml` (`server_url` and `repo_id`), reuses
//! pooled HTTP/2 gRPC channels per target endpoint, and bridges `jj_lib`'s async traits
//! onto a shared multi-thread Tokio runtime via [`util::run_on_tokio`].

#![allow(
    clippy::large_enum_variant,
    clippy::result_large_err,
    clippy::type_complexity
)]

pub mod cc_backend;
pub mod cc_op_heads_store;
pub mod cc_op_store;
pub mod repo;
pub mod util;

pub use repo::StoreFactoriesExt;
