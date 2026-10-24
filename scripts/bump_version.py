#!/usr/bin/env python3
"""Set every artifact version to X.Y.Z in one pass.

Usage: python3 scripts/bump_version.py 0.2.0 [--dry-run]

Edits each version site in place (the same sites check_versions.py reads),
then runs the check. Cargo.lock files are refreshed by the next cargo
command; the release workflow builds with --locked after the tag's own
lockfile update commit (see RELEASING.md).
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CARGO_FILES = ["Cargo.toml", "crates/crabcrop-url/Cargo.toml"]
JSON_FILES = [
    "sdks/typescript/package.json",
    "sdks/mcp/package.json",
    "frameworks/nextjs/package.json",
    "frameworks/nuxt/package.json",
    "frameworks/hydrogen/package.json",
]


def sub_once(rel, pattern, repl, flags=0):
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    new, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise SystemExit(f"{rel}: version pattern matched {count} times, expected 1")
    return new


def read_text(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def edits(version):
    out = {}
    for rel in CARGO_FILES:
        out[rel] = sub_once(
            rel,
            r'(\[package\][^\[]*?^version\s*=\s*")[^"]+(")',
            rf"\g<1>{version}\g<2>",
            re.S | re.M,
        )
    for rel in JSON_FILES:
        out[rel] = sub_once(rel, r'("version":\s*")[^"]+(")', rf"\g<1>{version}\g<2>")
    out["sdks/python/pyproject.toml"] = sub_once(
        "sdks/python/pyproject.toml",
        r'(\[project\][^\[]*?^version\s*=\s*")[^"]+(")',
        rf"\g<1>{version}\g<2>",
        re.S | re.M,
    )
    # server.json carries the version twice: the manifest
    # itself and its npm package entry. Both must move.
    manifest = read_text("sdks/mcp/server.json")
    manifest, count = re.subn(
        r'("version":\s*")[^"]+(")', rf"\g<1>{version}\g<2>", manifest
    )
    if count != 2:
        raise SystemExit(f"sdks/mcp/server.json: version matched {count} times, expected 2")
    out["sdks/mcp/server.json"] = manifest
    chart = sub_once(
        "deploy/helm/crabcrop/Chart.yaml", r"^version:\s*\S+", f"version: {version}", re.M
    )
    path = ROOT / "deploy/helm/crabcrop/Chart.yaml"
    chart, count = re.subn(
        r'^appVersion:\s*"[^"]+"', f'appVersion: "{version}"', chart, count=1, flags=re.M
    )
    if count != 1:
        raise SystemExit("Chart.yaml: appVersion matched 0 times")
    out["deploy/helm/crabcrop/Chart.yaml"] = chart
    return out


def main():
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry = "--dry-run" in sys.argv
    if len(args) != 1 or not re.fullmatch(r"\d+\.\d+\.\d+", args[0]):
        raise SystemExit("usage: bump_version.py X.Y.Z [--dry-run]")
    version = args[0]
    for rel, new in edits(version).items():
        if dry:
            print(f"would update {rel}")
        else:
            (ROOT / rel).write_text(new, encoding="utf-8")
            print(f"updated {rel}")
    if dry:
        return 0
    return subprocess.call([sys.executable, str(ROOT / "scripts/check_versions.py")])


if __name__ == "__main__":
    sys.exit(main())
