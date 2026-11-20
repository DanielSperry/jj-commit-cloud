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

use cc_common::workspace::{
    CheckWorkingCopyChangesRequest, DeleteWorkspaceRequest, GetWorkspaceRequest,
    ListWorkspacesRequest,
};
use cc_lib::cc_working_copy::resolve_workspace_user;
use cc_lib::util::{CommitCloudConfig, connect_workspace_client, run_on_tokio};
use clap::{Parser, Subcommand};
use jj_cli::cli_util::CommandHelper;
use jj_cli::command_error::{CommandError, user_error};

#[derive(Parser, Clone, Debug)]
pub struct WorkspaceArgs {
    #[command(subcommand)]
    pub subcommand: WorkspaceSubcommands,
}

#[derive(Subcommand, Clone, Debug)]
pub enum WorkspaceSubcommands {
    /// List cloud-tracked workspaces registered in this repository
    List {
        /// Optional workspace directory (defaults to current directory)
        #[arg(long)]
        repo_dir: Option<PathBuf>,
    },
    /// Show cloud-tracked checkout and working-copy dirty status for a workspace
    Status {
        /// Workspace name to inspect (defaults to "default")
        #[arg(long, default_value = "default")]
        workspace_name: String,
        /// Optional user email owner (defaults to current `user.email`)
        #[arg(long)]
        user: Option<String>,
        /// Optional workspace directory (defaults to current directory)
        #[arg(long)]
        repo_dir: Option<PathBuf>,
    },
    /// Remove a cloud-tracked workspace entry from the remote server
    Delete {
        /// Workspace name to remove
        workspace_name: String,
        /// Optional user email owner (defaults to current `user.email`)
        #[arg(long)]
        user: Option<String>,
        /// Optional workspace directory (defaults to current directory)
        #[arg(long)]
        repo_dir: Option<PathBuf>,
    },
}

fn load_repo_config(
    command_helper: &CommandHelper,
    repo_dir: Option<&PathBuf>,
) -> Result<CommitCloudConfig, CommandError> {
    let base = match repo_dir {
        Some(p) => p.clone(),
        None => command_helper
            .workspace_loader()
            .map(|w| w.repo_path().to_path_buf())
            .unwrap_or_else(|_| command_helper.cwd().to_path_buf()),
    };
    CommitCloudConfig::load_from_store(&base)
        .map_err(|e| user_error(format!("Failed to load Commit Cloud config: {e}")))
}

pub async fn cmd_cc_workspace(
    command_helper: &CommandHelper,
    args: &WorkspaceArgs,
) -> Result<(), CommandError> {
    match &args.subcommand {
        WorkspaceSubcommands::List { repo_dir } => {
            let cfg = load_repo_config(command_helper, repo_dir.as_ref())?;
            let server_url = cfg.server_url;
            let repo_id = cfg.repo_id;
            let workspaces = run_on_tokio(async move {
                let mut client = connect_workspace_client(server_url).await?;
                let res = client
                    .list_workspaces(ListWorkspacesRequest { repo_id })
                    .await?
                    .into_inner();
                Ok(res.workspaces)
            })
            .await
            .map_err(|e| user_error(format!("Failed to list cloud workspaces: {e}")))?;

            if workspaces.is_empty() {
                println!("No cloud-tracked workspaces registered.");
            } else {
                for ws in workspaces {
                    println!(
                        "{} ({}): commit={} op={} tree={}",
                        ws.workspace_name,
                        ws.user,
                        short_hex(&ws.commit_id, 12),
                        short_hex(&ws.operation_id, 12),
                        short_hex(&ws.tree_id, 12),
                    );
                }
            }
            Ok(())
        }
        WorkspaceSubcommands::Status {
            workspace_name,
            user,
            repo_dir,
        } => {
            let cfg = load_repo_config(command_helper, repo_dir.as_ref())?;
            let resolved_user = user
                .clone()
                .unwrap_or_else(|| resolve_workspace_user(command_helper.settings()));
            let ws_name = workspace_name.clone();
            let server_url = cfg.server_url;
            let repo_id = cfg.repo_id;

            let (ws_opt, changes_opt) = run_on_tokio(async move {
                let mut client = connect_workspace_client(server_url).await?;
                let ws = client
                    .get_workspace(GetWorkspaceRequest {
                        repo_id: repo_id.clone(),
                        user: resolved_user.clone(),
                        workspace_name: ws_name.clone(),
                    })
                    .await?
                    .into_inner()
                    .workspace;
                let changes = if ws.is_some() {
                    client
                        .check_working_copy_changes(CheckWorkingCopyChangesRequest {
                            repo_id,
                            user: resolved_user,
                            workspace_name: ws_name,
                        })
                        .await
                        .map(|r| Some(r.into_inner()))?
                } else {
                    None
                };
                Ok((ws, changes))
            })
            .await
            .map_err(|e| user_error(format!("Failed to query cloud workspace status: {e}")))?;

            let Some(ws) = ws_opt else {
                return Err(user_error(format!(
                    "Cloud workspace '{workspace_name}' is not registered"
                )));
            };
            println!("Workspace: {}", ws.workspace_name);
            println!("  User:      {}", ws.user);
            println!("  Commit:    {}", hex::encode(&ws.commit_id));
            println!("  Operation: {}", hex::encode(&ws.operation_id));
            println!("  Tree:      {}", hex::encode(&ws.tree_id));
            if !ws.sparse_patterns.is_empty() {
                println!("  Sparse:    {}", ws.sparse_patterns.join(", "));
            }
            if let Some(ch) = changes_opt {
                println!("  Modified:  {}", ch.has_changes);
            }
            Ok(())
        }
        WorkspaceSubcommands::Delete {
            workspace_name,
            user,
            repo_dir,
        } => {
            let cfg = load_repo_config(command_helper, repo_dir.as_ref())?;
            let resolved_user = user
                .clone()
                .unwrap_or_else(|| resolve_workspace_user(command_helper.settings()));
            let ws_name = workspace_name.clone();
            let server_url = cfg.server_url;
            let repo_id = cfg.repo_id;

            let deleted = run_on_tokio(async move {
                let mut client = connect_workspace_client(server_url).await?;
                let res = client
                    .delete_workspace(DeleteWorkspaceRequest {
                        repo_id,
                        user: resolved_user,
                        workspace_name: ws_name,
                    })
                    .await?
                    .into_inner();
                Ok(res.success)
            })
            .await
            .map_err(|e| user_error(format!("Failed to delete cloud workspace: {e}")))?;

            if deleted {
                println!("Deleted cloud workspace '{workspace_name}'");
            } else {
                println!("Cloud workspace '{workspace_name}' was not present");
            }
            Ok(())
        }
    }
}

fn short_hex(bytes: &[u8], max_chars: usize) -> String {
    let full = hex::encode(bytes);
    if full.len() > max_chars {
        full[..max_chars].to_string()
    } else {
        full
    }
}
