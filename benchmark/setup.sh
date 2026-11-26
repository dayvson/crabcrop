#!/usr/bin/env bash
# End to end setup for the benchmark harness. Run from anywhere:
#
#   bash benchmark/setup.sh
#   bash benchmark/setup.sh --check   # settle the crabcrop pin only
#
# It prepares benchmark/.work/ with the three servers at the pinned
# versions below, then prints how to run the benchmark. Nothing is
# installed system wide except the apt packages in step 1, and the
# Go toolchain goes under ~/.local (no sudo for that step). With
# --check it says which crabcrop commit it would build, or why it
# refuses to, and installs nothing.
#
# Pinned versions (the single place to change them). The crabcrop
# version and tag are release version sites: scripts/bump_version.py
# moves them, and empties CRABCROP_COMMIT, which a bump cannot know
# because the tag comes after the release commit (issue #711). The
# refresh that publishes new results pins the commit again; until
# then the tag alone names it.
GO_VERSION=1.27.4
GO_SHA256=7226736c2339719cda28fbf3a54d71aa63bcefe376e98cef00034268dab24840
LIBVIPS_APT_VERSION=8.15.1
IMGPROXY_VERSION=4.0.24
IMGPROXY_TAG=v4.0.24
IMGPROXY_COMMIT=4735a54944a53a3d572c166b38649c841862520c
THUMBOR_VERSION=7.8.0
CRABCROP_VERSION=0.16.0
CRABCROP_TAG=v0.16.0
CRABCROP_COMMIT=

# imgproxy 4.0.24 is the current imgproxy release. Its go.mod
# requires Go 1.27, which the pin above provides. It builds
# against libvips 8.15.1 (the Ubuntu 24.04 apt version) with
# one adjustment: Go 1.27's external link refuses the
# unresolved private dependencies of the shared libvips, so
# the build passes -Wl,--allow-shlib-undefined, the semantics
# the 3.31.4 build linked with under Go 1.26. The official
# Docker image instead builds from the imgproxy-base image
# (v4.0.15 for this release), which carries its own newer
# libvips built from source, so encoder behavior could differ
# slightly from this source build. See
# results/2026-11-15-imgproxy4.md for the full version table
# and the fidelity caveat.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORK="$HERE/.work"
CRABCROP_TREE="$WORK/crabcrop-${CRABCROP_TAG}"

case "${1:-}" in
  "" | --check) ;;
  *)
    echo "usage: bash benchmark/setup.sh [--check]" >&2
    exit 2
    ;;
esac

refuse() {
  echo "setup.sh: $*" >&2
  exit 1
}

# The crabcrop release to build, settled before anything is
# installed (issue #711). A bump used to leave the previous
# release's commit in CRABCROP_COMMIT, and step 5 fell back to it
# when the new tag was missing, so it built the previous release
# under the new tag's name. Now the tag decides: its commit is the
# one built, and a pinned commit must equal it. Without the tag in
# this clone, only a pinned commit whose Cargo.toml carries
# CRABCROP_VERSION is built.
# awk reads to the end, so git never writes into a closed pipe, and
# a commit with no readable Cargo.toml prints nothing: the caller
# then refuses with its own message instead of stopping on set -e.
crabcrop_version_at() {
  { git -C "$REPO" show "$1:Cargo.toml" 2>/dev/null || true; } \
    | awk '/^\[/ { in_package = ($0 == "[package]") }
           in_package && !found && /^version *= *"/ { sub(/^version *= *"/, ""); sub(/".*$/, ""); print; found = 1 }'
}

case "$CRABCROP_COMMIT" in
  "") ;;
  *[!0-9a-f]*) refuse "CRABCROP_COMMIT '$CRABCROP_COMMIT' is not a full commit SHA" ;;
  *) [ "${#CRABCROP_COMMIT}" -eq 40 ] || refuse "CRABCROP_COMMIT '$CRABCROP_COMMIT' is not a full commit SHA" ;;
esac
tagged="$(git -C "$REPO" rev-parse -q --verify "refs/tags/${CRABCROP_TAG}^{commit}" 2>/dev/null || true)"
if [ -n "$tagged" ]; then
  if [ -n "$CRABCROP_COMMIT" ] && [ "$CRABCROP_COMMIT" != "$tagged" ]; then
    refuse "${CRABCROP_TAG} is commit $tagged, but CRABCROP_COMMIT pins $CRABCROP_COMMIT;" \
      "pin the tag's commit, or fetch the tag again if it moved (git fetch --force origin tag ${CRABCROP_TAG})"
  fi
  crabcrop_commit="$tagged"
  how="the ${CRABCROP_TAG} tag"
elif [ -n "$CRABCROP_COMMIT" ]; then
  git -C "$REPO" cat-file -e "${CRABCROP_COMMIT}^{commit}" 2>/dev/null \
    || refuse "this clone has neither the ${CRABCROP_TAG} tag nor the pinned commit $CRABCROP_COMMIT; fetch them (git fetch --tags origin)"
  crabcrop_commit="$CRABCROP_COMMIT"
  how="CRABCROP_COMMIT (this clone has no ${CRABCROP_TAG} tag)"
else
  refuse "this clone has no ${CRABCROP_TAG} tag, and CRABCROP_COMMIT is empty, as a version bump leaves it;" \
    "fetch the tags (git fetch --tags origin), or wait until ${CRABCROP_TAG} is tagged"
fi
found_version="$(crabcrop_version_at "$crabcrop_commit")"
if [ "$found_version" != "$CRABCROP_VERSION" ]; then
  refuse "commit $crabcrop_commit is crabcrop ${found_version:-of no readable version}, not ${CRABCROP_VERSION};" \
    "pin the commit ${CRABCROP_TAG} names"
fi
# A worktree left by an earlier run is reused only at that commit.
if [ -e "$CRABCROP_TREE" ]; then
  at="$(git -C "$CRABCROP_TREE" rev-parse -q --verify HEAD 2>/dev/null || true)"
  if [ "$at" != "$crabcrop_commit" ]; then
    refuse "$CRABCROP_TREE holds ${at:-no checkout}, not $crabcrop_commit;" \
      "remove it (git worktree remove --force '$CRABCROP_TREE') and run setup.sh again"
  fi
fi
echo "crabcrop ${CRABCROP_VERSION}: commit $crabcrop_commit, from $how"
if [ "${1:-}" = "--check" ]; then
  exit 0
fi

mkdir -p "$WORK"
export PATH="$HOME/.cargo/bin:$PATH"

echo "== 1/5 system packages (libvips ${LIBVIPS_APT_VERSION}, build tools) =="
# libvips-dev is what imgproxy compiles against. The version above
# is the Ubuntu 24.04 apt candidate; on other distributions install
# the libvips development package for 8.15.x the same way.
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libvips-dev libdav1d-dev pkg-config gcc git curl python3 python3-venv nasm \
  cmake make
pkg-config --modversion vips

echo "== 2/5 Go ${GO_VERSION} =="
GO_DIR="$HOME/.local/go-${GO_VERSION}"
if [ ! -x "$GO_DIR/bin/go" ]; then
  tmp_tgz="$(mktemp)"
  curl -fsSL -o "$tmp_tgz" "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
  echo "${GO_SHA256}  $tmp_tgz" | sha256sum -c -
  mkdir -p "$HOME/.local"
  tar -C "$HOME/.local" -xzf "$tmp_tgz"
  mv "$HOME/.local/go" "$GO_DIR"
  rm -f "$tmp_tgz"
fi
"$GO_DIR/bin/go" version

echo "== 3/5 imgproxy ${IMGPROXY_VERSION} (source build) =="
if [ ! -x "$WORK/imgproxy" ]; then
  src="$WORK/imgproxy-src"
  rm -rf "$src"
  git clone --depth 1 --branch "$IMGPROXY_TAG" https://github.com/imgproxy/imgproxy.git "$src"
  test "$(git -C "$src" rev-parse HEAD)" = "$IMGPROXY_COMMIT"
  # GOTMPDIR points into .work because Go builds spill gigabytes of
  # temp files and some hosts mount a small /tmp. The main package
  # is ./cli in imgproxy 4.x, and the extldflags entry is the
  # libvips link adjustment described with the pins above.
  mkdir -p "$WORK/gotmp"
  (cd "$src" && PATH="$GO_DIR/bin:$PATH" CGO_ENABLED=1 \
    CGO_LDFLAGS_ALLOW='-Wl,--allow-shlib-undefined' \
    GOPATH="$WORK/gopath" GOCACHE="$WORK/gocache" GOTMPDIR="$WORK/gotmp" \
    go build -ldflags "-s -w -extldflags '-Wl,--allow-shlib-undefined'" \
    -o "$WORK/imgproxy" ./cli)
fi
"$WORK/imgproxy" --version || true

echo "== 4/5 Thumbor ${THUMBOR_VERSION} (venv) =="
if [ ! -x "$WORK/thumbor-venv/bin/thumbor" ]; then
  python3 -m venv "$WORK/thumbor-venv"
  "$WORK/thumbor-venv/bin/pip" install -q "thumbor==${THUMBOR_VERSION}"
fi
"$WORK/thumbor-venv/bin/thumbor" --version

echo "== 5/5 crabcrop ${CRABCROP_VERSION} (release build of commit ${crabcrop_commit}) =="
# A worktree of this clone checked out at the release's commit, as
# settled above, so the benchmark measures the named release, not
# the current branch.
if [ ! -e "$CRABCROP_TREE" ]; then
  git -C "$REPO" worktree add --detach "$CRABCROP_TREE" "$crabcrop_commit"
fi
if [ ! -x "$CRABCROP_TREE/target/release/crabcrop" ]; then
  (cd "$CRABCROP_TREE" && CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}" \
    cargo build --release --locked)
fi
ls -lh "$CRABCROP_TREE/target/release/crabcrop"

echo
echo "Setup complete. Run the benchmark with:"
echo "  python3 benchmark/bench.py"
echo "Raw results land in benchmark/.work/results.json"
