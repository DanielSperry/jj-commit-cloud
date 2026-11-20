-- Copyright 2026 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     https://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- GoogleSQL DDL for jj-commit-cloud on Cloud Spanner and Spanner Omni.
-- Shared immutable DAG objects (Commits, Trees, TreeEntries, Symlinks, ProjectBlobs)
-- interleave under Projects(ProjectId) so sibling agent repositories deduplicate object
-- storage, while per-repository mutable coordination (OpHeads, Views, Operations)
-- interleaves under Repositories(RepoId) so concurrent agents scale across independent
-- Spanner splits without hotspotting.

CREATE TABLE IF NOT EXISTS Projects(
  ProjectId STRING(128) NOT NULL,
  Name STRING(256))
  PRIMARY KEY(ProjectId);

CREATE INDEX IF NOT EXISTS ProjectsByName
ON
  Projects(Name);

CREATE TABLE IF NOT EXISTS Commits(
  ProjectId STRING(128) NOT NULL,
  CommitId BYTES(64) NOT NULL,
  ChangeId BYTES(64) NOT NULL,
  Generation INT64 NOT NULL,
  ProtoBytes BYTES(MAX) NOT NULL,
  InsertedAt TIMESTAMP OPTIONS (allow_commit_timestamp = TRUE))
  PRIMARY KEY(ProjectId, CommitId),
  INTERLEAVE IN PARENT Projects
ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS CommitsByChangeId
ON
  Commits(ProjectId, ChangeId),
INTERLEAVE IN Projects;

CREATE INDEX IF NOT EXISTS CommitsByInsertedAt
ON
  Commits(ProjectId, InsertedAt),
INTERLEAVE IN Projects;

CREATE TABLE IF NOT EXISTS Trees(
  ProjectId STRING(128) NOT NULL,
  TreeId BYTES(64) NOT NULL,
  Ready BOOL NOT NULL)
  PRIMARY KEY(ProjectId, TreeId),
  INTERLEAVE IN PARENT Projects
ON DELETE CASCADE;

CREATE TABLE IF NOT EXISTS TreeEntries(
  ProjectId STRING(128) NOT NULL,
  TreeId BYTES(64) NOT NULL,
  Name STRING(1024) NOT NULL,
  EntryProto BYTES(MAX) NOT NULL)
  PRIMARY KEY(ProjectId, TreeId, Name),
  INTERLEAVE IN PARENT Trees
ON DELETE CASCADE;

CREATE TABLE IF NOT EXISTS Symlinks(
  ProjectId STRING(128) NOT NULL,
  SymlinkId BYTES(64) NOT NULL,
  Target STRING(MAX) NOT NULL)
  PRIMARY KEY(ProjectId, SymlinkId),
  INTERLEAVE IN PARENT Projects
ON DELETE CASCADE;

CREATE TABLE IF NOT EXISTS ProjectBlobs(
  ProjectId STRING(128) NOT NULL,
  BlobId BYTES(64) NOT NULL)
  PRIMARY KEY(ProjectId, BlobId),
  INTERLEAVE IN PARENT Projects
ON DELETE CASCADE;

CREATE TABLE IF NOT EXISTS Repositories(
  RepoId STRING(128) NOT NULL,
  ProjectId STRING(128) NOT NULL,
  Name STRING(256),
  OpHeads ARRAY < BYTES(64) > NOT NULL)
  PRIMARY KEY(RepoId);

CREATE INDEX IF NOT EXISTS RepositoriesByName
ON
  Repositories(Name);

CREATE INDEX IF NOT EXISTS RepositoriesByProject
ON
  Repositories(ProjectId);

CREATE TABLE IF NOT EXISTS Views(
  RepoId STRING(128) NOT NULL,
  ViewId BYTES(64) NOT NULL,
  ProtoBytes BYTES(MAX) NOT NULL)
  PRIMARY KEY(RepoId, ViewId),
  INTERLEAVE IN PARENT Repositories
ON DELETE CASCADE;

CREATE TABLE IF NOT EXISTS Operations(
  RepoId STRING(128) NOT NULL,
  OperationId BYTES(64) NOT NULL,
  ProtoBytes BYTES(MAX) NOT NULL)
  PRIMARY KEY(RepoId, OperationId),
  INTERLEAVE IN PARENT Repositories
ON DELETE CASCADE;

CREATE TABLE IF NOT EXISTS Workspaces(
  RepoId STRING(128) NOT NULL,
  UserEmail STRING(256) NOT NULL,
  WorkspaceName STRING(256) NOT NULL,
  CommitId BYTES(64) NOT NULL,
  OperationId BYTES(64) NOT NULL,
  TreeId BYTES(64) NOT NULL,
  TreeIds ARRAY < BYTES(64) > NOT NULL,
  SparsePatterns ARRAY < STRING(1024) > NOT NULL)
  PRIMARY KEY(RepoId, UserEmail, WorkspaceName),
  INTERLEAVE IN PARENT Repositories
ON DELETE CASCADE;

CREATE TABLE IF NOT EXISTS Blobs(
  BlobId BYTES(64) NOT NULL,
  ContentSize INT64 NOT NULL,
  InlineContent BYTES(MAX),
  PrimaryLocation STRING(1024),
  ReplicatedLocations ARRAY < STRING(1024) > NOT NULL)
  PRIMARY KEY(BlobId);
