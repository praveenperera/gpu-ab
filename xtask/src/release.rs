use std::{
    env, fs,
    path::Path,
    process::{Command, Output},
};

use color_eyre::eyre::{Result, WrapErr, bail};

pub(crate) fn local() -> Result<()> {
    let root = crate::workspace_root()?;
    let status = Command::new("cargo")
        .args(["build", "--release", "--locked", "--package", "gpu-ab"])
        .current_dir(&root)
        .status()
        .wrap_err("failed to run cargo build --release")?;
    if !status.success() {
        bail!("cargo build --release failed");
    }

    // the same place install.sh puts a release build, so one replaces the other
    let home = env::var("HOME").wrap_err("HOME is not set")?;
    let bin_dir = format!("{home}/.local/bin");
    fs::create_dir_all(&bin_dir)?;
    let dest = format!("{bin_dir}/gpu-ab");

    // stage beside the destination, then rename, so a failed copy leaves the old binary in place
    let staged = format!("{bin_dir}/.gpu-ab.new");
    fs::copy(root.join("target/release/gpu-ab"), &staged)?;
    fs::rename(&staged, &dest)?;

    println!("Installed gpu-ab to {dest}");
    Ok(())
}

pub(crate) fn github() -> Result<()> {
    let root = crate::workspace_root()?;
    if !root.join(".github/workflows/release.yml").is_file() {
        bail!("missing .github/workflows/release.yml; cannot publish GitHub release assets");
    }

    // the tag must name the version in the commit it points at, not an
    // uncommitted bump in the working tree
    let dirty = git_stdout(&root, &["status", "--porcelain", "--untracked-files=no"])?;
    if !dirty.is_empty() {
        bail!("commit or stash tracked changes before tagging a release");
    }

    let tag = format!("v{}", crate::bump::package_version(&root)?);

    // release.yml builds the tagged commit, so it must already be on origin
    let on_remote = git_stdout(&root, &["branch", "--remotes", "--contains", "HEAD"])?;
    if on_remote.is_empty() {
        bail!("HEAD is not on origin; push it before tagging {tag}");
    }

    let tag_ref = format!("refs/tags/{tag}");
    let remote_tag = git_stdout(&root, &["ls-remote", "--tags", "origin", &tag_ref])?;
    if !remote_tag.is_empty() {
        bail!("{tag} is already on origin; run `just bump patch` and commit first");
    }

    // a local tag left by a failed push is reused when it names HEAD
    let head = git_stdout(&root, &["rev-parse", "HEAD"])?;
    let local_tag = git(
        &root,
        &["rev-parse", "--quiet", "--verify", &format!("{tag_ref}^{{commit}}")],
    )?;
    if !local_tag.status.success() {
        git_stdout(&root, &["tag", &tag])?;
    } else if String::from_utf8_lossy(&local_tag.stdout).trim() != head {
        bail!("local tag {tag} points at another commit; delete it or bump the version");
    }

    git_stdout(&root, &["push", "origin", &tag_ref])?;

    println!("Pushed {tag}; the Release workflow builds and publishes it");
    println!("Watch: just release-watch");
    println!("Then on each GPU host: just install");
    Ok(())
}

fn git_stdout(root: &Path, args: &[&str]) -> Result<String> {
    let output = git(root, args)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git {} failed: {}", args.join(" "), stderr.trim());
    }

    String::from_utf8(output.stdout)
        .wrap_err("git output was not utf-8")
        .map(|stdout| stdout.trim().to_string())
}

fn git(root: &Path, args: &[&str]) -> Result<Output> {
    Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .wrap_err_with(|| format!("failed to run git {}", args.join(" ")))
}
