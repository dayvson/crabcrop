#!/usr/bin/env bash
# End to end setup for the benchmark harness. Run from anywhere:
#
#   bash benchmark/setup.sh
#
# It prepares benchmark/.work/ with the three servers at the pinned
# versions below, then prints how to run the benchmark. Nothing is
# installed system wide except the apt packages in step 1, and the
# Go toolchain goes under ~/.local (no sudo for that step).
#
# Pinned versions (the single place to change them):
GO_VERSION=1.26.5
GO_SHA256=5c2c3b16caefa1d968a94c1daca04a7ca301a496d9b086e17ad77bb81393f053
LIBVIPS_APT_VERSION=8.15.1
IMGPROXY_VERSION=3.31.4
IMGPROXY_TAG=v3.31.4
IMGPROXY_COMMIT=8b0aa78279f191081054ba709b39f317f66ace78
THUMBOR_VERSION=7.8.0
CRABCROP_VERSION=0.9.5
CRABCROP_TAG=v0.9.5
CRABCROP_COMMIT=5a1674f4908ccc8ca83f64c134e98a60744fcd24

# Why imgproxy 3.31.4 and not the newer 4.0.x line: imgproxy 4.0.x
# needs Go 1.27 and a newer libvips than Ubuntu 24.04 ships. The
# 3.31.4 release is the newest release whose CI tests libvips
# 8.14, 8.15, and 8.16 (see .github/ci-docker in its tree), and
# Ubuntu 24.04 apt provides libvips 8.15.1. That pairing builds
# and runs unmodified. See results/2026-10-31.md for the full
# version table and the fidelity caveat.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORK="$HERE/.work"
mkdir -p "$WORK"
export PATH="$HOME/.cargo/bin:$PATH"

echo "== 1/5 system packages (libvips ${LIBVIPS_APT_VERSION}, build tools) =="
# libvips-dev is what imgproxy compiles against. The version above
# is the Ubuntu 24.04 apt candidate; on other distributions install
# the libvips development package for 8.15.x the same way.
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libvips-dev libdav1d-dev pkg-config gcc git curl python3 python3-venv nasm
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
  # temp files and some hosts mount a small /tmp.
  mkdir -p "$WORK/gotmp"
  (cd "$src" && PATH="$GO_DIR/bin:$PATH" CGO_ENABLED=1 \
    GOPATH="$WORK/gopath" GOCACHE="$WORK/gocache" GOTMPDIR="$WORK/gotmp" \
    go build -ldflags "-s -w" -o "$WORK/imgproxy" .)
fi
"$WORK/imgproxy" --version || true

echo "== 4/5 Thumbor ${THUMBOR_VERSION} (venv) =="
if [ ! -x "$WORK/thumbor-venv/bin/thumbor" ]; then
  python3 -m venv "$WORK/thumbor-venv"
  "$WORK/thumbor-venv/bin/pip" install -q "thumbor==${THUMBOR_VERSION}"
fi
"$WORK/thumbor-venv/bin/thumbor" --version

echo "== 5/5 crabcrop ${CRABCROP_VERSION} (release build from the tag) =="
if [ ! -x "$WORK/crabcrop-${CRABCROP_TAG}/target/release/crabcrop" ]; then
  # A worktree of this clone checked out at the release tag, so the
  # benchmark measures the named release, not the current branch.
  git -C "$REPO" worktree add --detach "$WORK/crabcrop-${CRABCROP_TAG}" "$CRABCROP_TAG" 2>/dev/null \
    || git -C "$REPO" worktree add --detach "$WORK/crabcrop-${CRABCROP_TAG}" "$CRABCROP_COMMIT"
  (cd "$WORK/crabcrop-${CRABCROP_TAG}" && CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}" \
    cargo build --release --locked)
fi
ls -lh "$WORK/crabcrop-${CRABCROP_TAG}/target/release/crabcrop"

echo
echo "Setup complete. Run the benchmark with:"
echo "  python3 benchmark/bench.py"
echo "Raw results land in benchmark/.work/results.json"
