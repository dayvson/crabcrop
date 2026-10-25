#!/usr/bin/env bash
# Stage the published release binaries for the container image build.
#
# Usage: bash scripts/stage_image_binaries.sh <version> <out-dir>
#
# Downloads the Linux release tarballs for <version> from the GitHub
# Release, verifies each against its published SHA-256 sidecar, and
# places the bare binary at <out-dir>/amd64/crabcrop and
# <out-dir>/arm64/crabcrop, the layout Dockerfile.release copies per
# TARGETARCH. The image build therefore ships exactly the binaries
# the release published, with the checksum proof carried into CI.
set -euo pipefail

VERSION="${1:?usage: stage_image_binaries.sh <version> <out-dir>}"
OUT="${2:?usage: stage_image_binaries.sh <version> <out-dir>}"
BASE="https://github.com/dayvson/crabcrop/releases/download/v${VERSION}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

stage() {
  local target="$1" arch="$2"
  local pkg="crabcrop-${target}.tar.xz"
  curl -fsSL "${BASE}/${pkg}" -o "${work}/${pkg}"
  curl -fsSL "${BASE}/${pkg}.sha256" -o "${work}/${pkg}.sha256"
  (cd "$work" && sha256sum -c "${pkg}.sha256")
  tar -xJf "${work}/${pkg}" -C "$work" --no-same-owner "crabcrop-${target}/crabcrop"
  mkdir -p "${OUT}/${arch}"
  install -m 0755 "${work}/crabcrop-${target}/crabcrop" "${OUT}/${arch}/crabcrop"
  echo "staged ${arch}: ${OUT}/${arch}/crabcrop"
}

stage x86_64-unknown-linux-gnu amd64
stage aarch64-unknown-linux-gnu arm64
