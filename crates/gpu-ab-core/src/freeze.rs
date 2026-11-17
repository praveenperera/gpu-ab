//! Frozen copies of the binaries under test
//!
//! Runs time copies inside the run directory, never `target/`, so a rebuild while the job
//! waits in the queue can't change what it measures

use std::{
    fs::{self, File},
    io::Read,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
};

use eyre::{Result, WrapErr, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A binary copied into the run directory
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Frozen {
    /// Where it was copied from
    pub source: PathBuf,
    /// The copy that runs
    pub path: PathBuf,
    pub sha256: String,
    /// Commit of the checkout it was built from, when given
    pub git: Option<GitSource>,
}

/// The source checkout a binary was built from
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitSource {
    pub dir: PathBuf,
    pub commit: String,
    /// Whether the checkout had uncommitted changes at freeze time
    pub dirty: bool,
}

/// Copies `source` to `dest`, makes the copy executable, and returns it with its hash
pub fn freeze(source: &Path, dest: &Path, git: Option<GitSource>) -> Result<Frozen> {
    let source = source
        .canonicalize()
        .wrap_err_with(|| format!("binary {}", source.display()))?;
    if !source.is_file() {
        bail!("{} is not a file", source.display());
    }

    fs::copy(&source, dest).wrap_err_with(|| format!("copying {} to {}", source.display(), dest.display()))?;
    fs::set_permissions(dest, fs::Permissions::from_mode(0o755))?;

    let sha256 = sha256_file(dest)?;
    if sha256_file(&source)? != sha256 {
        bail!("{} changed while it was being copied", source.display());
    }

    Ok(Frozen {
        source,
        path: dest.to_owned(),
        sha256,
        git,
    })
}

/// Hex sha256 of a file
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).wrap_err_with(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }

        hasher.update(&buf[..n]);
    }

    Ok(hex(&hasher.finalize()))
}

/// Lowercase hex of some bytes
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Reads `HEAD` and the dirty flag of a git checkout
pub fn git_source(dir: &Path) -> Result<GitSource> {
    let git = |args: &[&str]| -> Result<String> {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .wrap_err("running git")?;
        if !out.status.success() {
            bail!(
                "git {} in {}: {}",
                args.join(" "),
                dir.display(),
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }

        Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };

    Ok(GitSource {
        dir: dir.canonicalize()?,
        commit: git(&["rev-parse", "HEAD"])?,
        dirty: !git(&["status", "--porcelain", "--untracked-files=no"])?.is_empty(),
    })
}
