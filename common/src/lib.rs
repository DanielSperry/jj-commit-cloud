// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[allow(clippy::result_large_err)]
pub mod backend {
    tonic::include_proto!("commit_cloud.backend");
}

#[allow(clippy::result_large_err)]
pub mod op_store {
    tonic::include_proto!("commit_cloud.op_store");
}

#[allow(clippy::result_large_err)]
pub mod workspace {
    tonic::include_proto!("commit_cloud.workspace");
}

pub mod conversions;

pub const COMMIT_ID_LENGTH: usize = 20;
pub const TREE_ID_LENGTH: usize = 20;
pub const FILE_ID_LENGTH: usize = 20;
pub const SYMLINK_ID_LENGTH: usize = 20;
pub const CHANGE_ID_LENGTH: usize = 16;
pub const OPERATION_ID_LENGTH: usize = 16;
pub const VIEW_ID_LENGTH: usize = 16;

/// Maximum gRPC message size (64 MiB) reserved for large-payload and streaming channels.
pub const MAX_GRPC_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_REPO_PATH_BYTES: usize = 4096;

pub const ROOT_COMMIT_ID_BYTES: [u8; COMMIT_ID_LENGTH] = [0u8; COMMIT_ID_LENGTH];
pub const ROOT_CHANGE_ID_BYTES: [u8; CHANGE_ID_LENGTH] = [0u8; CHANGE_ID_LENGTH];
pub const ROOT_OPERATION_ID_BYTES: [u8; OPERATION_ID_LENGTH] = [0u8; OPERATION_ID_LENGTH];
pub const ROOT_VIEW_ID_BYTES: [u8; VIEW_ID_LENGTH] = [0u8; VIEW_ID_LENGTH];
pub const EMPTY_TREE_ID_HEX: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";
pub const EMPTY_TREE_ID_BYTES: [u8; TREE_ID_LENGTH] = [
    0x4b, 0x82, 0x5d, 0xc6, 0x42, 0xcb, 0x6e, 0xb9, 0xa0, 0x60, 0xe5, 0x4b, 0xf8, 0xd6, 0x92, 0x88,
    0xfb, 0xee, 0x49, 0x04,
];

/// Validates an optional canonical repository-relative path carried on tree,
/// file, and symlink RPCs.
///
/// Empty paths (`""`) are accepted (representing the repository root on tree
/// RPCs or an omitted path). Non-empty paths must be at most
/// [`MAX_REPO_PATH_BYTES`] bytes and must not contain NUL bytes (`\0`),
/// leading or trailing `/`, empty segments (`//`), or `.` / `..` segments.
pub fn validate_optional_repo_path(path: &str) -> Result<(), &'static str> {
    if path.is_empty() {
        return Ok(());
    }
    if path.len() > MAX_REPO_PATH_BYTES {
        return Err("path exceeds maximum length of 4096 bytes");
    }
    if path.as_bytes().contains(&0) {
        return Err("path must not contain NUL bytes");
    }
    if path.starts_with('/') || path.ends_with('/') {
        return Err("path must not start or end with '/'");
    }
    for segment in path.split('/') {
        if segment.is_empty() {
            return Err("path must not contain empty segments ('//')");
        }
        if segment == "." || segment == ".." {
            return Err("path must not contain '.' or '..' segments");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message as _;

    #[test]
    fn validate_optional_repo_path_accepts_empty_and_canonical_paths() {
        for valid in [
            "",
            "a",
            "src/lib.rs",
            "a/.hidden",
            "a/foo..bar",
            "a/..hidden",
            "a/foo.",
            "docs/日本語.md",
        ] {
            assert!(
                validate_optional_repo_path(valid).is_ok(),
                "expected valid path {valid:?}"
            );
        }

        let max_len = "a".repeat(MAX_REPO_PATH_BYTES);
        assert!(validate_optional_repo_path(&max_len).is_ok());

        let max_utf8_len = "é".repeat(MAX_REPO_PATH_BYTES / 2);
        assert_eq!(max_utf8_len.len(), MAX_REPO_PATH_BYTES);
        assert!(validate_optional_repo_path(&max_utf8_len).is_ok());
    }

    #[test]
    fn validate_optional_repo_path_rejects_malformed_and_oversized_paths() {
        for invalid in [
            "/", "/abs", "dir/", "a//b", ".", "..", "./a", "a/.", "a/./b", "../a", "a/..",
            "a/../b", "a\0b", "\0",
        ] {
            assert!(
                validate_optional_repo_path(invalid).is_err(),
                "expected invalid path {invalid:?}"
            );
        }

        let too_long = "a".repeat(MAX_REPO_PATH_BYTES + 1);
        assert!(validate_optional_repo_path(&too_long).is_err());

        let too_long_utf8 = format!("{}a", "é".repeat(MAX_REPO_PATH_BYTES / 2));
        assert_eq!(too_long_utf8.len(), MAX_REPO_PATH_BYTES + 1);
        assert!(too_long_utf8.chars().count() < MAX_REPO_PATH_BYTES);
        assert!(validate_optional_repo_path(&too_long_utf8).is_err());
    }

    #[test]
    fn file_and_symlink_rpc_messages_round_trip_optional_path() {
        let read_file = backend::ReadFileRequest {
            project_id: "proj-1".to_string(),
            file_id: vec![0x11; FILE_ID_LENGTH],
            offset: 10,
            max_bytes: 20,
            path: "src/main.rs".to_string(),
        };
        assert_eq!(
            backend::ReadFileRequest::decode(read_file.encode_to_vec().as_slice()).unwrap(),
            read_file
        );

        let write_file = backend::WriteFileRequest {
            project_id: "proj-1".to_string(),
            content: b"hello".to_vec(),
            path: "src/main.rs".to_string(),
        };
        assert_eq!(
            backend::WriteFileRequest::decode(write_file.encode_to_vec().as_slice()).unwrap(),
            write_file
        );

        let write_chunk = backend::WriteFileChunk {
            project_id: "proj-1".to_string(),
            chunk: b"hello".to_vec(),
            total_size: Some(5),
            expected_file_id: Some(vec![0x22; FILE_ID_LENGTH]),
            path: "src/main.rs".to_string(),
        };
        assert_eq!(
            backend::WriteFileChunk::decode(write_chunk.encode_to_vec().as_slice()).unwrap(),
            write_chunk
        );

        let read_symlink = backend::ReadSymlinkRequest {
            project_id: "proj-1".to_string(),
            symlink_id: vec![0x33; SYMLINK_ID_LENGTH],
            path: "docs/latest".to_string(),
        };
        assert_eq!(
            backend::ReadSymlinkRequest::decode(read_symlink.encode_to_vec().as_slice()).unwrap(),
            read_symlink
        );

        let write_symlink = backend::WriteSymlinkRequest {
            project_id: "proj-1".to_string(),
            target: "../v2".to_string(),
            path: "docs/latest".to_string(),
        };
        assert_eq!(
            backend::WriteSymlinkRequest::decode(write_symlink.encode_to_vec().as_slice()).unwrap(),
            write_symlink
        );
    }
}
