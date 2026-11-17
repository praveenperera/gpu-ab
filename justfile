set shell := ["bash", "-uc"]

[default]
list:
    @just --list

# ------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------

[group('build')]
build:
    cargo build

[group('build')]
build-release:
    cargo build --release

# Bump the workspace version: major, minor, or patch
[group('build')]
bump part:
    cargo xtask bump {{part}}

# After `just bump` is committed and pushed, tag that version so the Release
# workflow publishes it for install.sh; `local` installs a dev build instead
#
# Tag and publish through GitHub Actions: github (default) | gh | public | local
[group('build')]
release place="github":
    cargo xtask release {{place}}

# Follow the newest Release workflow run until it finishes
[group('build')]
release-watch:
    gh run watch --exit-status "$(gh run list --workflow Release --limit 1 --json databaseId --jq '.[0].databaseId')"

[private]
alias br := build-release

# ------------------------------------------------------------------------------
# run
# ------------------------------------------------------------------------------

# Install or update the latest release into ~/.local/bin
[group('run')]
install tag="":
    ./install.sh {{tag}}

# ------------------------------------------------------------------------------
# test
# ------------------------------------------------------------------------------

[group('test')]
test *args="":
    cargo test --workspace {{args}}

# ------------------------------------------------------------------------------
# lint
# ------------------------------------------------------------------------------

[group('lint')]
fmt *args="":
    cargo fmt --all {{args}}

[group('lint')]
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

[group('ci')]
[script('bash')]
ci:
    set -euo pipefail
    just fmt -- --check
    just clippy
    just test
