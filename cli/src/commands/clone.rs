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

use std::fs;
use std::path::Path;

use cc_lib::cc_backend::CommitCloudBackend;
use cc_lib::cc_index_store::CommitCloudIndexStore;
use cc_lib::cc_op_heads_store::CommitCloudOpHeadsStore;
use cc_lib::cc_op_store::CommitCloudOpStore;
use cc_lib::cc_working_copy::CommitCloudWorkingCopyFactory;
use clap::Parser;
use jj_cli::cli_util::{CommandHelper, RevisionArg};
use jj_cli::command_error::{CommandError, user_error};
use jj_cli::ui::Ui;
use jj_lib::backend::BackendInitError;
use jj_lib::commit::Commit;
use jj_lib::object_id::ObjectId as _;
use jj_lib::op_store::RootOperationData;
use jj_lib::ref_name::WorkspaceNameBuf;
use jj_lib::repo::{ReadonlyRepo, Repo as _};
use jj_lib::settings::UserSettings;
use jj_lib::signing::Signer;
use jj_lib::workspace::Workspace;
use uuid::Uuid;

#[derive(Parser, Clone, Debug)]
pub struct CcCloneArgs {
    /// Remote gRPC server URL (e.g. http://127.0.0.1:8080)
    #[arg(long)]
    pub server: String,

    /// Existing remote repository ID to clone
    pub repo_id: String,

    /// Destination directory for the cloned workspace (defaults to <repo_id>)
    pub destination: Option<String>,

    /// Optional workspace identifier (defaults to a unique ws-<short_id>)
    #[arg(long)]
    pub workspace_name: Option<String>,

    /// Revision (commit ID, change ID, or revset) to check out on clone
    #[arg(short = 'r', long)]
    pub revision: Option<String>,

    /// Working copy implementation ("commit_cloud" syncs state to WorkspaceService, or "local")
    #[arg(long, default_value = "commit_cloud")]
    pub working_copy_type: String,
}

pub async fn cmd_cc_clone(
    ui: &mut Ui,
    command_helper: &CommandHelper,
    args: &CcCloneArgs,
) -> Result<(), CommandError> {
    let dest_str = args.destination.as_deref().unwrap_or(&args.repo_id);
    let dest_path = Path::new(dest_str);
    let dest_existed = dest_path.exists();
    if dest_existed {
        let is_non_empty = fs::read_dir(dest_path)
            .map_err(|e| {
                user_error(format!(
                    "Failed to inspect clone directory {}: {e}",
                    dest_path.display()
                ))
            })?
            .next()
            .is_some();
        if is_non_empty {
            return Err(user_error(
                "Destination path exists and is not an empty directory",
            ));
        }
    } else {
        fs::create_dir_all(dest_path).map_err(|e| {
            user_error(format!(
                "Failed to create clone directory {}: {e}",
                dest_path.display()
            ))
        })?;
    }
    let dot_jj_existed = dest_path.join(".jj").exists();

    let ws_name = args.workspace_name.clone().unwrap_or_else(|| {
        let short = &Uuid::new_v4().simple().to_string()[..8];
        format!("ws-{short}")
    });

    let (init_settings, _) = command_helper.settings_for_new_workspace(ui, dest_path)?;
    let signer = Signer::from_settings(&init_settings)
        .map_err(|e| user_error(format!("Failed to initialize signature signer: {e:?}")))?;

    let backend_initializer = |_settings: &UserSettings, store_path: &Path| {
        let backend = CommitCloudBackend::init_with_options(
            store_path,
            &args.server,
            None,
            Some(&args.repo_id),
            false,
        )
        .map_err(BackendInitError)?;
        Ok(Box::new(backend) as Box<dyn jj_lib::backend::Backend>)
    };

    let op_store_initializer =
        |_settings: &UserSettings, store_path: &Path, _root_op_data: RootOperationData| {
            let op_store = CommitCloudOpStore::load(store_path).map_err(BackendInitError)?;
            Ok(Box::new(op_store) as Box<dyn jj_lib::op_store::OpStore>)
        };

    let op_heads_store_initializer =
        |_settings: &UserSettings,
         store_path: &Path,
         _root_op_id: &jj_lib::op_store::OperationId| {
            let op_heads_store =
                CommitCloudOpHeadsStore::load(store_path).map_err(BackendInitError)?;
            Ok(Box::new(op_heads_store) as Box<dyn jj_lib::op_heads_store::OpHeadsStore>)
        };

    let index_store_initializer = |settings: &UserSettings, store_path: &Path| {
        let index_store =
            CommitCloudIndexStore::init(settings, store_path).map_err(BackendInitError)?;
        Ok(Box::new(index_store) as Box<dyn jj_lib::index::IndexStore>)
    };

    let working_copy_factory: Box<dyn jj_lib::working_copy::WorkingCopyFactory> =
        match args.working_copy_type.as_str() {
            "commit_cloud" => Box::new(CommitCloudWorkingCopyFactory::new()),
            "local" => Box::new(jj_lib::local_working_copy::LocalWorkingCopyFactory {}),
            other => {
                return Err(user_error(format!(
                    "Unsupported working copy type '{other}' (expected 'commit_cloud' or 'local')"
                )));
            }
        };

    let clone_res: Result<(), CommandError> = async {
        Workspace::init_with_factories(
            &init_settings,
            dest_path,
            &backend_initializer,
            signer,
            &op_store_initializer,
            &op_heads_store_initializer,
            &index_store_initializer,
            ReadonlyRepo::default_submodule_store_initializer(),
            working_copy_factory.as_ref(),
            WorkspaceNameBuf::from(ws_name.clone()),
        )
        .await
        .map_err(|e| user_error(format!("Failed to initialize cloned workspace: {e:?}")))?;

        // Reload settings and workspace at dest_path so CWD's repository config never leaks into
        // the clone.
        let (clone_settings, _) = command_helper.settings_for_new_workspace(ui, dest_path)?;
        let workspace = command_helper.load_workspace_at(dest_path, &clone_settings)?;
        let op_head = command_helper.resolve_operation(
            ui,
            workspace.repo_loader(),
            workspace.workspace_name(),
        )?;
        let repo = workspace.repo_loader().load_at(&op_head).await?;
        let mut workspace_command = command_helper.for_workable_repo(ui, workspace, repo)?;

        let target_res: Result<Option<Commit>, CommandError> = if let Some(rev_str) = &args.revision
        {
            let rev_arg = RevisionArg::from(rev_str.clone());
            workspace_command
                .resolve_single_rev(ui, &rev_arg)
                .await
                .map(Some)
        } else {
            select_default_clone_head(
                workspace_command.repo(),
                workspace_command.get_wc_commit_id(),
            )
            .await
        };

        let target_commit = match target_res {
            Ok(c) => c,
            Err(err) => {
                let mut forget_tx = workspace_command.repo().start_transaction();
                if forget_tx
                    .repo_mut()
                    .remove_wc_commit(workspace_command.workspace_name())
                    .await
                    .is_ok()
                {
                    let _ = forget_tx
                        .write(format!("forget failed clone workspace {ws_name}"))
                        .await;
                }
                return Err(err);
            }
        };

        if let Some(commit) = target_commit {
            let mut tx = workspace_command.start_transaction();
            tx.check_out(&commit)
                .map_err(|e| user_error(format!("Failed to check out cloned commit: {e:?}")))?;
            tx.finish(
                ui,
                format!("check out cloned Commit Cloud commit {}", commit.id().hex()),
            )
            .await?;
        }
        Ok(())
    }
    .await;

    if let Err(err) = clone_res {
        if !dot_jj_existed {
            let _ = fs::remove_dir_all(dest_path.join(".jj"));
        }
        if !dest_existed {
            let _ = fs::remove_dir(dest_path);
        }
        return Err(err);
    }

    println!(
        "Cloned Commit Cloud repository {} into {}",
        args.repo_id,
        dest_path.display()
    );
    Ok(())
}

async fn select_default_clone_head(
    repo: &ReadonlyRepo,
    own_wc_id: Option<&jj_lib::backend::CommitId>,
) -> Result<Option<Commit>, CommandError> {
    let root_id = repo.store().root_commit_id();
    let mut candidates = Vec::new();
    for head_id in repo.view().heads() {
        if head_id == root_id || Some(head_id) == own_wc_id {
            continue;
        }
        let commit = repo
            .store()
            .get_commit_async(head_id)
            .await
            .map_err(|e| user_error(format!("Failed to read remote head commit: {e:?}")))?;
        let is_non_empty = !commit
            .is_empty(repo)
            .await
            .map_err(|e| user_error(format!("Failed to inspect head tree: {e:?}")))?;
        // If a peer workspace already ran `jj new` atop a described/non-empty commit,
        // prefer checking out its non-empty parent when the head itself is an empty WC commit.
        if !is_non_empty
            && commit.description().is_empty()
            && commit.parent_ids().len() == 1
            && &commit.parent_ids()[0] != root_id
        {
            let parent = repo
                .store()
                .get_commit_async(&commit.parent_ids()[0])
                .await
                .map_err(|e| user_error(format!("Failed to read parent commit: {e:?}")))?;
            candidates.push((true, parent.committer().timestamp.timestamp.0, parent));
        } else {
            candidates.push((
                is_non_empty || !commit.description().is_empty(),
                commit.committer().timestamp.timestamp.0,
                commit,
            ));
        }
    }

    candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    Ok(candidates.pop().map(|(_, _, c)| c))
}
