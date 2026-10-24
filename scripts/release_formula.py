#!/usr/bin/env python3
"""Generate the Homebrew formula for a crabcrop release.

Usage: python3 scripts/release_formula.py X.Y.Z sha256.sum > crabcrop.rb

Reads the release's sha256.sum (published by cargo-dist with the release)
and fills the macOS and Linux blocks for both architectures. The formula
installs the prebuilt binary; see RELEASING.md for how it reaches a tap.
"""
import re
import sys

BASE = "https://github.com/dayvson/crabcrop/releases/download/v{version}"


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: release_formula.py X.Y.Z sha256.sum")
    version, sums_path = sys.argv[1], sys.argv[2]
    sums = {}
    for line in open(sums_path, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0]

    def sha(target):
        name = f"crabcrop-{target}.tar.xz"
        if name not in sums:
            raise SystemExit(f"{name} not in {sums_path}")
        return name, sums[name]

    def block(target, indent="    "):
        name, digest = sha(target)
        url = f"{BASE.format(version=version)}/{name}"
        return f'{indent}url "{url}"\n{indent}sha256 "{digest}"'

    print(f'''class Crabcrop < Formula
  desc "On-demand image optimization service"
  homepage "https://crabcrop.org"
  version "{version}"
  license "MIT"

  on_macos do
    on_arm do
{block("aarch64-apple-darwin", "      ")}
    end
    on_intel do
{block("x86_64-apple-darwin", "      ")}
    end
  end

  on_linux do
    on_arm do
{block("aarch64-unknown-linux-gnu", "      ")}
    end
    on_intel do
{block("x86_64-unknown-linux-gnu", "      ")}
    end
  end

  def install
    bin.install "crabcrop"
  end

  test do
    assert_match "crabcrop", shell_output("#{{bin}}/crabcrop --help")
  end
end
''')


if __name__ == "__main__":
    main()
