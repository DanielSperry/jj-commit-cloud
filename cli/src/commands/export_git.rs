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

use std::path::PathBuf;

use cc_lib::git_bridge::{GitExportOptions, GitExporter};
use cc_lib::util::{CommitCloudConfig, run_on_tokio};
use clap::Parser;
use jj_cli::cli_util::CommandHelper;
use jj_cli::command_error::{CommandError, user_error};

#[derive(Parser, Clone, Debug)]
pub struct ExportGitArgs {
    /// Destination Git repository directory (initialized automatically if absent)
    #[arg(long)]
    pub git_dir: PathBuf,

    /// Remote Commit Cloud gRPC server URL (defaults to current workspace's `server_url`)
    #[arg(long)]
    pub server: Option<String>,

    /// Source Commit Cloud repository ID (defaults to current workspace's `repo_id`)
    #[arg(long)]
    pub repo_id: Option<String>,

    /// Optional shared Commit Cloud project ID owning immutable objects
    #[arg(long)]
    pub project_id: Option<String>,

    /// Initialize destination as a bare Git repository when creating a new directory
    #[arg(long)]
    pub bare: bool,
}

pub async fn cmd_export_git(
    command_helper: &CommandHelper,
    args: &ExportGitArgs,
) -> Result<(), CommandError> {
    let (server_url, repo_id, project_id) = match (&args.server, &args.repo_id) {
        (Some(s), Some(r)) => (s.clone(), r.clone(), args.project_id.clone()),
        (s_opt, r_opt) => {
            let store_hint = command_helper
                .workspace_loader()
                .map(|w| w.repo_path().to_path_buf())
                .unwrap_or_else(|_| command_helper.cwd().to_path_buf());
            let cfg = CommitCloudConfig::load_from_store(&store_hint).map_err(|e| {
                user_error(format!(
                    "Pass --server and --repo-id or run inside a Commit Cloud workspace: {e}"
                ))
            })?;
            let proj = args
                .project_id
                .clone()
                .or_else(|| Some(cfg.effective_project_id().to_string()));
            (
                s_opt.clone().unwrap_or(cfg.server_url),
                r_opt.clone().unwrap_or(cfg.repo_id),
                proj,
            )
        }
    };

    let opts = GitExportOptions {
        git_dir: args.git_dir.clone(),
        server_url,
        repo_id: repo_id.clone(),
        project_id,
        bare: args.bare,
    };

    let report = run_on_tokio(async move { GitExporter::new(opts).run().await })
        .await
        .map_err(|e| user_error(format!("Git export failed: {e}")))?;

    for name in &report.skipped_bookmarks {
        eprintln!("Warning: Skipped bookmark {name:?}: not a valid Git branch name");
    }
    println!(
        "Exported Commit Cloud repository {} to {}",
        repo_id,
        args.git_dir.display()
    );
    println!(
        "  Commits: {}, Trees: {}, Blobs: {}, Bookmarks: {}",
        report.commits_exported,
        report.trees_exported,
        report.blobs_exported,
        report.bookmarks_exported
    );
    Ok(())
}
