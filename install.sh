#!/usr/bin/env bash
# Install or update gpu-ab from a GitHub release
#
#   ./install.sh                 # latest release into ~/.local/bin
#   ./install.sh v0.1.0          # a specific tag
#   curl -fsSL https://github.com/praveenperera/gpu-ab/releases/latest/download/install.sh | bash
#
# Downloads go through `gh` when it is installed and logged in, and otherwise
# through the GitHub API with curl, sending GH_TOKEN when it is set so a private
# fork works too. Release builds are static x86_64 Linux, for NVIDIA GPU hosts,
# and arm64 macOS, for Apple silicon

set -euo pipefail

repo="${GPU_AB_REPO:-praveenperera/gpu-ab}"
install_dir="${GPU_AB_INSTALL_DIR:-$HOME/.local/bin}"
tag="${1:-${GPU_AB_VERSION:-}}"

die() {
    echo "install.sh: $*" >&2
    exit 1
}

target_triple() {
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) echo x86_64-unknown-linux-musl ;;
        Darwin-arm64) echo aarch64-apple-darwin ;;
        *) die "no release build for $(uname -s) $(uname -m)" ;;
    esac
}

use_gh() {
    command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1
}

# GET a GitHub API path with curl; extra args go to curl
api() {
    local path="$1"
    shift
    if [ -n "${GH_TOKEN:-}" ]; then
        set -- -H "Authorization: Bearer $GH_TOKEN" "$@"
    fi
    curl -fsSL -H "X-GitHub-Api-Version: 2022-11-28" "$@" "https://api.github.com/$path"
}

# Put each JSON key/value pair on its own line, pretty-printed or not; release
# names and asset names never contain commas or braces
json_fields() {
    tr ',{}[]' '\n' | sed 's/^[[:space:]]*//; /^$/d'
}

latest_tag() {
    if use_gh; then
        gh release view --repo "$repo" --json tagName --jq .tagName
    else
        # avoid a jq dependency; tag_name appears once in a release object
        api "repos/$repo/releases/latest" -H "Accept: application/vnd.github+json" |
            json_fields | sed -n 's/^"tag_name": *"\([^"]*\)"$/\1/p' | head -n 1
    fi
}

# Download a named asset of $tag into the current directory
download() {
    local name="$1"

    if use_gh; then
        gh release download "$tag" --repo "$repo" --pattern "$name" --clobber
        return
    fi

    local release id
    release="$(api "repos/$repo/releases/tags/$tag" -H "Accept: application/vnd.github+json")"
    # each asset object lists its API url, ending in the asset id, before its name
    id="$(printf '%s\n' "$release" | json_fields |
        awk -v want="\"name\":\"$name\"" '
            /^"url": *"https:\/\/api\.github\.com\/repos\/.*\/releases\/assets\/[0-9]+"$/ {
                candidate = $0; sub(/.*\/assets\//, "", candidate); sub(/"$/, "", candidate)
            }
            { line = $0; gsub(/ /, "", line) }
            line == want { print candidate; exit }
        ')"
    [ -n "$id" ] || die "release $tag has no asset $name"
    api "repos/$repo/releases/assets/$id" -H "Accept: application/octet-stream" -o "$name"
}

verify() {
    local archive="$1" sums="$2" expected actual
    expected="$(awk '{ print $1; exit }' "$sums")"

    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$archive" | awk '{ print $1 }')"
    else
        actual="$(shasum -a 256 "$archive" | awk '{ print $1 }')"
    fi

    [ "$expected" = "$actual" ] || die "checksum mismatch for $archive"
}

main() {
    local target archive bin
    target="$(target_triple)"
    [ -n "$tag" ] || tag="$(latest_tag)"
    [ -n "$tag" ] || die "could not resolve the latest release of $repo"

    # global so the EXIT trap still sees it after main returns
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    cd "$work"

    archive="gpu-ab-$tag-$target.tar.gz"
    echo "downloading $archive from $repo" >&2
    download "$archive"
    download "gpu-ab-$tag-$target.sha256"
    verify "$archive" "gpu-ab-$tag-$target.sha256"

    mkdir extract
    tar -xzf "$archive" -C extract
    bin="$(find extract -type f -name gpu-ab -perm -u+x | head -n 1)"
    [ -n "$bin" ] || die "$archive does not contain a gpu-ab binary"

    # stage beside the destination, then rename, so a run already started keeps its inode
    mkdir -p "$install_dir"
    cp "$bin" "$install_dir/.gpu-ab.new"
    chmod 755 "$install_dir/.gpu-ab.new"
    mv -f "$install_dir/.gpu-ab.new" "$install_dir/gpu-ab"

    echo "installed $("$install_dir/gpu-ab" --version 2>/dev/null || echo "gpu-ab $tag") to $install_dir/gpu-ab" >&2

    # an older `cargo install` copy in ~/.cargo/bin usually comes first on PATH
    local found
    found="$(command -v gpu-ab || true)"
    if [ -n "$found" ] && [ "$found" != "$install_dir/gpu-ab" ]; then
        echo "warning: $found comes first on PATH; remove it to use this install" >&2
    fi
}

main
