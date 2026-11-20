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

//! Local content-addressed `BlobStore` backed by two-level sharded filesystem directories.
//!
//! # Authority and Crash-Safety Ordering
//! `put_blob` writes content to a unique temporary file inside the target shard directory
//! (`blobs/<aa>/<bb>/.tmp-<uuid>`) with blocking `std::fs` calls, checks the written length,
//! fsyncs the file, renames it onto `blobs/<aa>/<bb>/<full_hex>`, and fsyncs the shard
//! directory on unix. Readers see either no file or a complete blob, and failed writes
//! remove their temporary file.
//!
//! # Concurrency and Merge Invariants
//! Because keys are 20-byte Git blob SHA-1 digests, concurrent writers uploading the
//! same `blob_id` produce identical byte streams; an existing file with the expected length
//! short-circuits without rewriting disk blocks, a file with any other length is a damaged
//! earlier write and is replaced, and racing renames replace the target with identical
//! bytes.
//!
//! # Cross-Process and RPC Boundaries
//! `get_blob_range` seeks directly to `offset` and reads at most `length` bytes so partial
//! hydration does not read entire multi-gigabyte blobs into server RAM.

use std::io::{SeekFrom, Write as _};
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use super::{BlobStore, StoreError, StoreResult};

#[derive(Clone, Debug)]
pub struct LocalFsBlobStore {
    root: PathBuf,
    location_uri: String,
}

impl LocalFsBlobStore {
    pub fn open(root: impl AsRef<Path>) -> StoreResult<Self> {
        let root_path = root.as_ref().to_path_buf();
        std::fs::create_dir_all(root_path.join("blobs")).map_err(|e| {
            StoreError::Backend(format!(
                "failed to create local blob root {}: {e}",
                root_path.display()
            ))
        })?;
        let canonical = std::fs::canonicalize(&root_path).unwrap_or_else(|_| root_path.clone());
        let location_uri = format!("file://{}", canonical.display());
        Ok(Self {
            root: root_path,
            location_uri,
        })
    }

    pub fn open_with_location_id(
        root: impl AsRef<Path>,
        location_uri: impl Into<String>,
    ) -> StoreResult<Self> {
        let mut store = Self::open(root)?;
        store.location_uri = location_uri.into();
        Ok(store)
    }

    fn object_path(&self, blob_id: &[u8]) -> PathBuf {
        let hex_id = hex::encode(blob_id);
        let prefix_a = hex_id.get(0..2).unwrap_or("00");
        let prefix_b = hex_id.get(2..4).unwrap_or("00");
        self.root
            .join("blobs")
            .join(prefix_a)
            .join(prefix_b)
            .join(hex_id)
    }
}

/// Stores `data` at `target` unless a file with the same length is already there. The blob id
/// is a hash over the length and the content, so a length mismatch means a damaged file.
fn write_blob_file(target: &Path, data: &[u8]) -> StoreResult<()> {
    match std::fs::metadata(target) {
        Ok(meta) if meta.is_file() && meta.len() == data.len() as u64 => return Ok(()),
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(StoreError::Backend(format!("blob stat failed: {e}"))),
    }

    let shard_dir = target.parent().expect("blob path must have shard parent");
    std::fs::create_dir_all(shard_dir)
        .map_err(|e| StoreError::Backend(format!("failed to create blob shard dir: {e}")))?;

    let tmp_path = shard_dir.join(format!(".tmp-{}", uuid::Uuid::new_v4()));
    let result = write_temp_and_rename(&tmp_path, target, data);
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    result
}

fn write_temp_and_rename(tmp_path: &Path, target: &Path, data: &[u8]) -> StoreResult<()> {
    let mut tmp_file = std::fs::File::create_new(tmp_path)
        .map_err(|e| StoreError::Backend(format!("failed to create temp blob file: {e}")))?;
    tmp_file
        .write_all(data)
        .and_then(|()| tmp_file.flush())
        .map_err(|e| StoreError::Backend(format!("failed to write temp blob file: {e}")))?;
    let written = tmp_file
        .metadata()
        .map_err(|e| StoreError::Backend(format!("failed to stat temp blob file: {e}")))?
        .len();
    if written != data.len() as u64 {
        return Err(StoreError::Backend(format!(
            "temp blob file has {written} bytes, expected {}",
            data.len()
        )));
    }
    tmp_file
        .sync_all()
        .map_err(|e| StoreError::Backend(format!("failed to sync temp blob file: {e}")))?;
    drop(tmp_file);

    std::fs::rename(tmp_path, target)
        .map_err(|e| StoreError::Backend(format!("failed to rename temp blob file: {e}")))?;
    sync_dir(target.parent().expect("blob path must have shard parent"))
        .map_err(|e| StoreError::Backend(format!("failed to sync blob shard dir: {e}")))
}

/// Makes a completed rename durable across power loss.
#[cfg(unix)]
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> std::io::Result<()> {
    Ok(())
}

#[async_trait]
impl BlobStore for LocalFsBlobStore {
    fn location_id(&self) -> &str {
        &self.location_uri
    }

    async fn put_blob(&self, blob_id: &[u8], data: Bytes) -> StoreResult<()> {
        let target = self.object_path(blob_id);
        // tokio::fs::File reports some write errors only on flush, so write with blocking
        // std::fs calls, where every error comes back from the call that hit it.
        tokio::task::spawn_blocking(move || write_blob_file(&target, &data))
            .await
            .map_err(|e| StoreError::Backend(format!("blob write task failed: {e}")))?
    }

    async fn get_blob(&self, blob_id: &[u8]) -> StoreResult<Option<Bytes>> {
        let target = self.object_path(blob_id);
        match tokio::fs::read(&target).await {
            Ok(buf) => Ok(Some(Bytes::from(buf))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(StoreError::Backend(format!(
                "failed to read local blob {}: {e}",
                target.display()
            ))),
        }
    }

    async fn get_blob_range(
        &self,
        blob_id: &[u8],
        offset: u64,
        length: Option<u64>,
    ) -> StoreResult<Option<Bytes>> {
        let target = self.object_path(blob_id);
        let mut file = match tokio::fs::File::open(&target).await {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(StoreError::Backend(format!(
                    "failed to open local blob {}: {e}",
                    target.display()
                )));
            }
        };

        let total_len = file
            .metadata()
            .await
            .map_err(|e| StoreError::Backend(format!("failed to stat local blob: {e}")))?
            .len();

        if total_len == 0 {
            if offset == 0 {
                return Ok(Some(Bytes::new()));
            }
            return Err(StoreError::OutOfRange(format!(
                "requested read at offset {offset}, file size is 0"
            )));
        }
        if offset >= total_len {
            return Err(StoreError::OutOfRange(format!(
                "requested read at offset {offset}, file size is {total_len}"
            )));
        }

        file.seek(SeekFrom::Start(offset))
            .await
            .map_err(|e| StoreError::Backend(format!("failed to seek local blob: {e}")))?;

        let to_read = match length {
            Some(0) | None => total_len - offset,
            Some(max_len) => max_len.min(total_len - offset),
        };

        let mut buf = Vec::with_capacity(to_read as usize);
        file.take(to_read)
            .read_to_end(&mut buf)
            .await
            .map_err(|e| StoreError::Backend(format!("failed to read local blob range: {e}")))?;
        Ok(Some(Bytes::from(buf)))
    }

    async fn has_blob(&self, blob_id: &[u8]) -> StoreResult<bool> {
        let target = self.object_path(blob_id);
        tokio::fs::try_exists(&target)
            .await
            .map_err(|e| StoreError::Backend(format!("failed to stat local blob: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shard_dir_entries(store: &LocalFsBlobStore, blob_id: &[u8]) -> Vec<String> {
        let shard_dir = store.object_path(blob_id).parent().unwrap().to_path_buf();
        let mut names: Vec<String> = std::fs::read_dir(shard_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[tokio::test]
    async fn put_blob_replaces_existing_file_with_wrong_length() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalFsBlobStore::open(dir.path()).unwrap();
        let blob_id = [0x5e; 20];
        let payload = Bytes::from_static(b"complete blob payload");

        // Zero-length or truncated files from interrupted writes must be replaced on retry.
        let target = store.object_path(&blob_id);
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        for damaged in [&[][..], &payload[..5]] {
            std::fs::write(&target, damaged).unwrap();
            store.put_blob(&blob_id, payload.clone()).await.unwrap();
            assert_eq!(
                store.get_blob(&blob_id).await.unwrap(),
                Some(payload.clone())
            );
            assert_eq!(
                shard_dir_entries(&store, &blob_id),
                vec![hex::encode(blob_id)]
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn put_blob_keeps_existing_file_with_expected_length() {
        use std::os::unix::fs::MetadataExt as _;

        let dir = tempfile::tempdir().unwrap();
        let store = LocalFsBlobStore::open(dir.path()).unwrap();
        let blob_id = [0x6f; 20];
        let payload = Bytes::from_static(b"stored once");

        store.put_blob(&blob_id, payload.clone()).await.unwrap();
        let inode = std::fs::metadata(store.object_path(&blob_id))
            .unwrap()
            .ino();
        store.put_blob(&blob_id, payload).await.unwrap();
        assert_eq!(
            std::fs::metadata(store.object_path(&blob_id))
                .unwrap()
                .ino(),
            inode
        );
    }

    #[tokio::test]
    async fn put_blob_removes_temp_file_when_write_fails() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalFsBlobStore::open(dir.path()).unwrap();
        let blob_id = [0x7a; 20];

        // A directory at the target path makes the final rename fail.
        let target = store.object_path(&blob_id);
        std::fs::create_dir_all(target.join("occupied")).unwrap();

        let err = store
            .put_blob(&blob_id, Bytes::from_static(b"payload"))
            .await
            .unwrap_err();
        assert!(matches!(err, StoreError::Backend(_)), "{err:?}");
        assert_eq!(
            shard_dir_entries(&store, &blob_id),
            vec![hex::encode(blob_id)]
        );
    }
}
