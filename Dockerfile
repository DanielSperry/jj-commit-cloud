# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Multi-target container build for jj-cc-server, custom jj CLI, and smoke runner.
# - Default target (`server`): minimal runtime image for Cloud Run, GKE, GCE, and Compose.
# - `--target cli`: runtime image with custom `jj` and `xtask` for containerized client workflows.
# - `--target bins`: scratch export stage for extracting compiled Linux binaries (`--output ./bin`).
FROM docker.io/library/rust:bookworm AS builder

WORKDIR /app
COPY rust-toolchain.toml Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo
COPY common/Cargo.toml ./common/Cargo.toml
COPY lib/Cargo.toml ./lib/Cargo.toml
COPY server/Cargo.toml ./server/Cargo.toml
COPY daemon/Cargo.toml ./daemon/Cargo.toml
COPY cli/Cargo.toml ./cli/Cargo.toml
COPY testutils/Cargo.toml ./testutils/Cargo.toml
COPY xtask/Cargo.toml ./xtask/Cargo.toml

# Pre-build and cache third-party Cargo dependencies before copying workspace sources.
RUN mkdir -p common/src lib/src server/src daemon/src cli/src testutils/src xtask/src \
    && printf 'fn main() {}\n' > common/build.rs \
    && touch common/src/lib.rs lib/src/lib.rs server/src/lib.rs testutils/src/lib.rs \
    && touch daemon/src/lib.rs \
    && printf 'fn main() {}\n' > daemon/src/main.rs \
    && printf 'fn main() {}\n' > server/src/main.rs \
    && printf 'fn main() {}\n' > cli/src/main.rs \
    && printf 'fn main() {}\n' > xtask/src/main.rs \
    && cargo build --locked --release \
        -p jj-commit-cloud-server --bin jj-cc-server \
        -p jj-commit-cloud-daemon --bin jj-cc-daemon \
        -p cli --bin jj \
        -p xtask --bin xtask \
    && rm -rf common/build.rs common/src lib/src server/src daemon/src cli/src testutils/src xtask/src \
        target/release/build/cc-common-* \
        target/release/.fingerprint/cc-common-* \
        target/release/.fingerprint/cc-lib-* \
        target/release/.fingerprint/jj-commit-cloud-* \
        target/release/.fingerprint/cli-* \
        target/release/.fingerprint/xtask-* \
        target/release/deps/libcc_* \
        target/release/deps/libjj_commit_cloud_* \
        target/release/deps/jj* \
        target/release/deps/xtask* \
        target/release/jj* \
        target/release/xtask*

COPY common ./common
COPY lib ./lib
COPY server ./server
COPY daemon ./daemon
COPY cli ./cli
COPY testutils ./testutils
COPY xtask ./xtask

# common/build.rs falls back to vendored protoc, so external protobuf-compiler is optional.
RUN rm -rf target/release/build/cc-common-* target/release/build/cc-lib-* target/release/build/jj-commit-cloud-* target/release/build/cli* target/release/build/xtask* target/release/.fingerprint/cc-common-* target/release/.fingerprint/cc-lib-* target/release/.fingerprint/jj-commit-cloud-* target/release/.fingerprint/cli* target/release/.fingerprint/xtask* \
    && find common lib server daemon cli testutils xtask -type f -exec touch {} + \
    && cargo build --locked --release \
    -p jj-commit-cloud-server --bin jj-cc-server \
    -p jj-commit-cloud-daemon --bin jj-cc-daemon \
    -p cli --bin jj \
    -p xtask --bin xtask

FROM docker.io/library/debian:bookworm-slim AS cli
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/jj /usr/local/bin/jj
COPY --from=builder /app/target/release/jj-cc-daemon /usr/local/bin/jj-cc-daemon
COPY --from=builder /app/target/release/xtask /usr/local/bin/xtask
ENTRYPOINT ["jj"]

FROM scratch AS bins
COPY --from=builder /app/target/release/jj-cc-server /jj-cc-server
COPY --from=builder /app/target/release/jj-cc-daemon /jj-cc-daemon
COPY --from=builder /app/target/release/jj /jj

FROM docker.io/library/debian:bookworm-slim AS server
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && mkdir -p /data \
    && chown nobody:nogroup /data \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/jj-cc-server /usr/local/bin/jj-cc-server

USER nobody:nogroup
VOLUME ["/data"]
ENV HOST=0.0.0.0
ENV PORT=8080
ENV STORE_TYPE=sqlite SQLITE_PATH=/data/commit_cloud.db BLOB_PATH=/data/blobs
EXPOSE 8080

# HOST, PORT, STORE_TYPE, BLOB_STORE, SQLITE_PATH, BLOB_PATH, SPANNER_DATABASE,
# GCS_BUCKET, and GCS_PEER_BUCKETS are read directly from environment variables by clap.
ENTRYPOINT ["jj-cc-server"]
