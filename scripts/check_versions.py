#!/usr/bin/env python3
"""Verify that every shipped artifact carries the same version.

crabcrop releases one version number across all artifacts (RELEASING.md).
This script reads each version site and fails if any disagrees. It is run
by scripts/bump_version.py after an edit, and belongs in CI.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def cargo_package_version(rel):
    text = read(rel)
    match = re.search(r'\[package\][^\[]*?^version\s*=\s*"([^"]+)"', text, re.S | re.M)
    return match.group(1) if match else None


def json_version(rel):
    return json.loads(read(rel)).get("version")


def mcp_manifest_versions(rel):
    data = json.loads(read(rel))
    packages = data.get("packages") or [{}]
    return data.get("version"), packages[0].get("version")


def pyproject_version(rel):
    match = re.search(r'\[project\][^\[]*?^version\s*=\s*"([^"]+)"', read(rel), re.S | re.M)
    return match.group(1) if match else None


def chart_versions(rel):
    text = read(rel)
    version = re.search(r"^version:\s*(\S+)", text, re.M)
    app = re.search(r'^appVersion:\s*"([^"]+)"', text, re.M)
    return (version.group(1) if version else None, app.group(1) if app else None)


def mcp_server_info_version(rel):
    match = re.search(r'const SERVER = \{ name: "crabcrop", version: "([^"]+)" \}', read(rel))
    return match.group(1) if match else None


def sites():
    found = {
        "Cargo.toml (server)": cargo_package_version("Cargo.toml"),
        "crates/crabcrop-url/Cargo.toml": cargo_package_version("crates/crabcrop-url/Cargo.toml"),
        "sdks/typescript/package.json": json_version("sdks/typescript/package.json"),
        "sdks/mcp/package.json": json_version("sdks/mcp/package.json"),
        "frameworks/nextjs/package.json": json_version("frameworks/nextjs/package.json"),
        "frameworks/nuxt/package.json": json_version("frameworks/nuxt/package.json"),
        "frameworks/hydrogen/package.json": json_version("frameworks/hydrogen/package.json"),
        "sdks/python/pyproject.toml": pyproject_version("sdks/python/pyproject.toml"),
    }
    found["sdks/mcp/src/index.ts SERVER.version"] = mcp_server_info_version("sdks/mcp/src/index.ts")
    manifest, manifest_pkg = mcp_manifest_versions("sdks/mcp/server.json")
    found["sdks/mcp/server.json version"] = manifest
    found["sdks/mcp/server.json packages[0].version"] = manifest_pkg
    chart, app = chart_versions("deploy/helm/crabcrop/Chart.yaml")
    found["deploy/helm/crabcrop/Chart.yaml version"] = chart
    found["deploy/helm/crabcrop/Chart.yaml appVersion"] = app
    return found


def main():
    found = sites()
    values = set(found.values())
    for name, value in found.items():
        print(f"{value or 'MISSING':>10}  {name}")
    if None in values or len(values) != 1:
        print("\nversion mismatch across artifacts", file=sys.stderr)
        return 1
    print(f"\nall artifacts at {values.pop()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
