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

use std::process::ExitCode;

use cc_lib::StoreFactoriesExt;
use jj_cli::cli_util::{CliRunner, CommandHelper};
use jj_cli::command_error::CommandError;
use jj_lib::config::{ConfigLayer, ConfigSource};
use jj_lib::repo::StoreFactories;

mod commands;

#[derive(clap::Subcommand, Clone, Debug)]
enum CustomCommands {
    Cc {
        #[command(subcommand)]
        subcommand: CcCommands,
    },
}

#[derive(clap::Subcommand, Clone, Debug)]
enum CcCommands {
    Init(commands::init::CcInitArgs),
}

fn create_store_factories() -> StoreFactories {
    let mut factories = StoreFactories::empty();
    factories.add_commit_cloud();
    factories
}

fn default_commit_cloud_config() -> ConfigLayer {
    ConfigLayer::parse(
        ConfigSource::Default,
        "snapshot.max-new-file-size = \"64MiB\"\n",
    )
    .expect("default Commit Cloud config layer should parse")
}

async fn run_custom_command(
    ui: &mut jj_cli::ui::Ui,
    command_helper: &CommandHelper,
    command: CustomCommands,
) -> Result<(), CommandError> {
    match command {
        CustomCommands::Cc { subcommand } => match subcommand {
            CcCommands::Init(args) => commands::init::cmd_cc_init(ui, command_helper, &args).await,
        },
    }
}

fn main() -> ExitCode {
    CliRunner::init()
        .add_extra_config(default_commit_cloud_config())
        .add_store_factories(create_store_factories())
        .add_subcommand(run_custom_command)
        .run()
        .into()
}
