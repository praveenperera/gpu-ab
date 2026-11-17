use std::{path::PathBuf, process::Command};

use clap::{Parser, Subcommand, ValueEnum};
use color_eyre::eyre::{Result, bail};

mod bump;
mod release;

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Install gpu-ab from this checkout, or publish a GitHub release
    Release {
        /// Where to put the release
        #[arg(value_enum)]
        place: Place,
    },
    /// Bump the workspace version
    Bump {
        /// Semver component to increment
        #[arg(value_enum)]
        part: bump::Part,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Place {
    /// Build this checkout and install it to ~/.local/bin/gpu-ab
    Local,
    /// Tag the committed Cargo version and push it to GitHub
    Github,
    /// Alias for github
    Gh,
    /// Alias for github
    Public,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let cli = Cli::parse();

    match cli.command {
        Commands::Release { place } => match place {
            Place::Local => release::local(),
            Place::Github | Place::Gh | Place::Public => release::github(),
        },
        Commands::Bump { part } => bump::run(part),
    }
}

pub(crate) fn workspace_root() -> Result<PathBuf> {
    let output = Command::new("cargo")
        .args(["locate-project", "--workspace", "--message-format=plain"])
        .output()?;

    if !output.status.success() {
        bail!("Failed to locate workspace root");
    }

    let path = String::from_utf8(output.stdout)?;
    let cargo_toml = PathBuf::from(path.trim());
    let Some(root) = cargo_toml.parent() else {
        bail!("Cargo.toml should have a parent directory");
    };

    Ok(root.to_path_buf())
}
