#!/usr/bin/env bash
# Build dav1d, the AV1 decoder every shipped binary decodes AVIF with,
# from a pinned and checksummed release tarball (issue #596).
#
# dav1d is a C library. The release binaries, CI's macOS job and the
# nightly fuzzing link it statically. This script downloads the dav1d
# release tarball for DAV1D_VERSION, checks it against DAV1D_SHA256
# before anything reads it, and builds a static library with the meson
# and ninja pinned in scripts/requirements/dav1d-build.txt. dav1d-sys
# then finds it through pkg-config and links it statically
# (SYSTEM_DEPS_DAV1D_LINK=static).
#
# It replaces SYSTEM_DEPS_DAV1D_BUILD_INTERNAL=always, which makes
# dav1d-sys clone dav1d from code.videolan.org by ref name at build
# time, with no commit or hash check.
#
# The version is the one Debian trixie ships as libdav1d7, which the
# from-source Dockerfile links, so both image routes decode with the
# same dav1d release. RELEASING.md says how to move it; the version
# tools check that docs/deployment.md names the same version
# (scripts/version_sites.py).
#
# Usage:
#   bash scripts/build_dav1d.sh DIR
#       Build into DIR (DIR/prefix holds the library). In GitHub
#       Actions the two variables cargo needs go to $GITHUB_ENV;
#       elsewhere they are printed as export lines, for eval.
#   bash scripts/build_dav1d.sh --check-binary FILE...
#       Fail when a FILE links libdav1d dynamically (ldd, or otool -L
#       on macOS), so a release cannot ship a binary that needs a
#       system libdav1d.
#
# Needs curl, python3 (with venv, or pip), a C compiler, pkg-config, and nasm
# on x86_64. DAV1D_TARBALL=PATH reads a local copy of the tarball
# instead of downloading it; the checksum check still runs.
set -euo pipefail

DAV1D_VERSION=1.5.1
DAV1D_SHA256=401813f1f89fa8fd4295805aa5284d9aed9bc7fc1fdbe554af4292f64cbabe21
DAV1D_URL="https://downloads.videolan.org/pub/videolan/dav1d/$DAV1D_VERSION/dav1d-$DAV1D_VERSION.tar.xz"

REPO=$(cd "$(dirname "$0")/.." && pwd)

die() {
  echo "build_dav1d.sh: $*" >&2
  exit 1
}

sha256_check() { # reads "HASH  FILE" lines on stdin
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -c -
  else
    sha256sum -c -
  fi
}

check_binary() {
  [ "$#" -gt 0 ] || die "--check-binary needs at least one binary"
  for bin in "$@"; do
    [ -f "$bin" ] || die "no binary at $bin"
    # otool's first line names the binary itself, not a library.
    if [ "$(uname -s)" = Darwin ]; then
      deps=$(otool -L "$bin") || die "otool -L $bin failed"
      deps=$(printf '%s\n' "$deps" | sed 1d)
    else
      deps=$(ldd "$bin") || die "ldd $bin failed"
    fi
    if printf '%s\n' "$deps" | grep -i libdav1d >&2; then
      die "$bin links libdav1d dynamically; it must carry the dav1d this script builds"
    fi
    echo "$bin: dav1d is linked statically (no libdav1d among its shared libraries)"
  done
}

if [ "${1:-}" = "--check-binary" ]; then
  shift
  check_binary "$@"
  exit 0
fi

[ "$#" -eq 1 ] || die "usage: build_dav1d.sh DIR | --check-binary FILE..."
# Only the export lines go to stdout, so `eval "$(...)"` reads nothing
# else; the build's own output goes to stderr.
exec 3>&1 1>&2
mkdir -p "$1"
dir=$(cd "$1" && pwd)
tarball="dav1d-$DAV1D_VERSION.tar.xz"

# Fetch and check the tarball before anything reads it.
rm -f "$dir/$tarball"
if [ -n "${DAV1D_TARBALL:-}" ]; then
  cp "$DAV1D_TARBALL" "$dir/$tarball"
else
  curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$dir/$tarball" "$DAV1D_URL"
fi
if ! (cd "$dir" && echo "$DAV1D_SHA256  $tarball" | sha256_check); then
  rm -f "$dir/$tarball"
  die "$tarball does not match the pinned sha256 $DAV1D_SHA256; nothing was built"
fi

rm -rf "$dir/src" "$dir/build" "$dir/prefix" "$dir/tools"
mkdir -p "$dir/src"
tar -xJf "$dir/$tarball" -C "$dir/src" --strip-components=1

# meson and ninja, pinned by hash, in a virtual environment of their
# own. Where Python has no venv module (Ubuntu without python3-venv),
# they go into a directory on PYTHONPATH instead.
requirements="$REPO/scripts/requirements/dav1d-build.txt"
if python3 -m venv "$dir/tools" >/dev/null 2>&1; then
  "$dir/tools/bin/python" -m pip install --quiet --disable-pip-version-check \
    --require-hashes -r "$requirements"
else
  rm -rf "$dir/tools"
  python3 -m pip install --quiet --disable-pip-version-check \
    --require-hashes -r "$requirements" --target "$dir/tools"
  export PYTHONPATH="$dir/tools${PYTHONPATH:+:$PYTHONPATH}"
fi
export PATH="$dir/tools/bin:$PATH"

# The static library only: no tools, tests or examples. --libdir=lib
# keeps Debian's multiarch default out of the path pkg-config reads.
meson setup "$dir/build" "$dir/src" \
  --prefix "$dir/prefix" --libdir lib --buildtype release \
  --default-library static \
  -Denable_tools=false -Denable_tests=false -Denable_examples=false
ninja -C "$dir/build"
meson install -C "$dir/build" --quiet

# pkg-config must resolve this build: the pinned version, and a static
# library with no shared one beside it.
pc="$dir/prefix/lib/pkgconfig"
found=$(PKG_CONFIG_PATH="$pc" pkg-config --modversion dav1d) || die "pkg-config cannot find dav1d in $pc"
[ "$found" = "$DAV1D_VERSION" ] || die "pkg-config resolves dav1d $found, not $DAV1D_VERSION"
[ -f "$dir/prefix/lib/libdav1d.a" ] || die "no static libdav1d.a in $dir/prefix/lib"
for shared in "$dir"/prefix/lib/libdav1d.so* "$dir"/prefix/lib/libdav1d*.dylib; do
  if [ -e "$shared" ]; then
    die "a shared library, $shared, is in $dir/prefix/lib; only the static library may be there"
  fi
done

pkg_path="$pc${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
if [ -n "${GITHUB_ENV:-}" ]; then
  {
    echo "PKG_CONFIG_PATH=$pkg_path"
    echo "SYSTEM_DEPS_DAV1D_LINK=static"
  } >>"$GITHUB_ENV"
  echo "dav1d $DAV1D_VERSION built in $dir/prefix; PKG_CONFIG_PATH and SYSTEM_DEPS_DAV1D_LINK=static set for later steps" >&2
else
  printf 'export PKG_CONFIG_PATH=%q\nexport SYSTEM_DEPS_DAV1D_LINK=static\n' "$pkg_path" >&3
  echo "dav1d $DAV1D_VERSION built in $dir/prefix" >&2
fi
