#!/usr/bin/env python3
"""Publish the MCP server's registry entry for a release, each version once.

The release workflow's publish-mcp-registry job runs this (issue #711)
once publish-mcp has put @crabcrop/mcp on npm: the official MCP registry
lists a version only after it finds the npm package and the entry's name
in that package's `mcpName`. It publishes sdks/mcp/server.json with the
registry's own publisher CLI, mcp-publisher, which the job installs
pinned by version and SHA-256, so the listing follows every release
instead of the one version submitted by hand.

- A prerelease is not listed. The registry marks the newest version of
  an entry as its latest, and a prerelease such as 0.14.0-rc.1 sorts
  above 0.13.0, so clients that install the latest listing would get the
  release candidate, which npm keeps under its `next` dist-tag
  (scripts/npm_dist_tag.sh applies the same rule).
- server.json must describe the release: its version and its npm
  package's version are the tag's, the package is the one
  sdks/mcp/package.json builds, and that package's `mcpName` is the
  entry's name.
- A version the registry already lists, in any status, is skipped.
  Registry versions are immutable, so an earlier attempt of the job
  published it, and re-running the failed job has nothing left to do.
  Any other answer than listed or not found stops the job.
- The login is GitHub OIDC when the entry's namespace is the repository
  owner's (`io.github.<owner>/`): the job's `id-token: write` is all it
  needs. The registry grants an OIDC token only its repository owner's
  namespace, so an entry under another account, such as an organization
  the repository does not belong to, logs in with a GitHub token of an
  Owner of that organization instead: MCP_GITHUB_TOKEN, a secret of the
  `release` environment (RELEASING.md, Repository settings). Without one
  the job stops and says so.
- Once published, the version must be listed.

Usage: publish_mcp_registry.py --tag vX.Y.Z --publisher PATH
       [--server-json FILE] [--package-json FILE] [--registry URL]
       [--owner OWNER] [--wait SECONDS]
The owner defaults to GITHUB_REPOSITORY_OWNER, which Actions sets, and
the token comes from MCP_GITHUB_TOKEN.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = "https://registry.modelcontextprotocol.io"
TAG = re.compile(r"v(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)")
GITHUB_NAME = re.compile(r"io\.github\.([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)/[A-Za-z0-9._-]+")


class Refusal(Exception):
    """A reason to stop before, or instead of, the publish."""


def is_prerelease(version):
    """A SemVer pre-release part (before any build metadata) makes a prerelease."""
    return "-" in version.split("+", 1)[0]


def http_get(url):
    """GET `url`; (status, body text). A failed request raises Refusal."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, ""
    except (urllib.error.URLError, OSError) as err:
        raise Refusal(f"could not read {url}: {err}") from err


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise Refusal(f"cannot read {path}: {err}") from err


def manifest_problems(server, package, version):
    """Why server.json does not describe this release of the npm package; empty when it does."""
    if not isinstance(server, dict) or not isinstance(package, dict):
        return ["server.json and package.json must each be a JSON object"]
    problems = []
    name = server.get("name")
    if server.get("version") != version:
        problems.append(f"server.json's version is {server.get('version')!r}, not {version!r}, the version the tag names")
    npm = [p for p in server.get("packages") or [] if isinstance(p, dict) and p.get("registryType") == "npm"]
    if len(npm) != 1:
        problems.append(f"server.json lists {len(npm)} npm packages, not one")
    else:
        if npm[0].get("identifier") != package.get("name"):
            problems.append(f"server.json's npm package is {npm[0].get('identifier')!r}, not {package.get('name')!r}")
        if npm[0].get("version") != version:
            problems.append(f"server.json's npm package version is {npm[0].get('version')!r}, not {version!r}")
    if package.get("version") != version:
        problems.append(f"package.json's version is {package.get('version')!r}, not {version!r}")
    if package.get("mcpName") != name:
        problems.append(f"package.json's mcpName is {package.get('mcpName')!r}, not the entry's name {name!r}")
    return problems


def version_url(registry, name, version):
    quote = urllib.parse.quote
    return f"{registry.rstrip('/')}/v0.1/servers/{quote(name, safe='')}/versions/{quote(version, safe='')}?include_deleted=true"


def listed(registry, name, version, fetch):
    """Whether the registry lists `version` of `name`, in any status."""
    url = version_url(registry, name, version)
    status, body = fetch(url)
    if status == 404:
        return False
    if status != 200:
        raise Refusal(f"the MCP registry answered {status} for {url}, so it is unknown whether {name} {version} is listed")
    try:
        found = json.loads(body)["server"]["version"]
    except (ValueError, KeyError, TypeError) as err:
        raise Refusal(f"the MCP registry's answer for {name} {version} names no version") from err
    if found != version:
        raise Refusal(f"the MCP registry answered version {found!r} when asked for {version!r}")
    return True


def login_args(name, owner, token, registry):
    """(publisher arguments, how) for the login that can publish `name`."""
    match = GITHUB_NAME.fullmatch(name or "")
    if not match:
        raise Refusal(f"the entry's name {name!r} is not io.github.<account>/<server>, which a GitHub login publishes")
    account = match.group(1)
    if owner and account.lower() == owner.lower():
        return ["login", "github-oidc", "--registry", registry], f"GitHub OIDC (io.github.{account}/ is the repository owner's namespace)"
    if token:
        return (
            ["login", "github", "--token", token, "--registry", registry],
            f"MCP_GITHUB_TOKEN (io.github.{account}/ is not the repository owner's namespace, so OIDC cannot publish it)",
        )
    raise Refusal(
        f"the entry {name} is in the io.github.{account}/ namespace, and the registry grants a GitHub OIDC token only "
        f"its repository owner's ({'io.github.' + owner + '/' if owner else 'unknown here'}); add MCP_GITHUB_TOKEN, a "
        f"GitHub token of an Owner of {account} that can read its members (classic: read:org), as a secret of the "
        "release environment (RELEASING.md, Repository settings), and re-run this job"
    )


def run_command(args):
    return subprocess.run(args)


def publish(
    version,
    publisher,
    server_json,
    package_json,
    registry=REGISTRY,
    owner="",
    token="",
    fetch=http_get,
    run=run_command,
    wait=60,
    sleep=time.sleep,
    out=print,
):
    """Publish server.json for `version` unless that is not needed; True when this run published it."""
    if is_prerelease(version):
        out(f"{version} is a prerelease: not listing it in the MCP registry, where it would become the latest version")
        return False
    server, package = read_json(server_json), read_json(package_json)
    problems = manifest_problems(server, package, version)
    if problems:
        raise Refusal("; ".join(problems))
    name = server["name"]
    if listed(registry, name, version, fetch):
        out(f"the MCP registry already lists {name} {version}; nothing to publish")
        return False
    args, how = login_args(name, owner, token, registry)
    out(f"logging in to {registry} with {how}")
    done = run([publisher, *args])
    if done.returncode != 0:
        raise Refusal(f"mcp-publisher login failed (exit {done.returncode})")
    out(f"publishing {name} {version}")
    done = run([publisher, "publish", str(server_json)])
    if done.returncode != 0:
        raise Refusal(f"mcp-publisher publish failed (exit {done.returncode})")
    deadline = time.monotonic() + wait
    while not listed(registry, name, version, fetch):
        if time.monotonic() >= deadline:
            raise Refusal(f"mcp-publisher published {name} {version}, but the registry has not listed it within {wait} s")
        sleep(5)
    out(f"the MCP registry lists {name} {version}")
    return True


def fail(message):
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::error title=MCP registry publish::{message}")
    else:
        print(f"publish_mcp_registry.py: {message}", file=sys.stderr)


def main(argv=None, **hooks):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="the release tag, vX.Y.Z")
    parser.add_argument("--publisher", required=True, help="the mcp-publisher binary")
    parser.add_argument("--server-json", default=str(ROOT / "sdks/mcp/server.json"))
    parser.add_argument("--package-json", default=str(ROOT / "sdks/mcp/package.json"))
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--owner", default=os.environ.get("GITHUB_REPOSITORY_OWNER", ""))
    parser.add_argument("--wait", type=int, default=60, help="seconds to wait for the registry to list the publish")
    args = parser.parse_args(argv)
    match = TAG.fullmatch(args.tag)
    if not match or args.wait < 0:
        fail(f"usage: --tag must be vX.Y.Z, got {args.tag!r}")
        return 2
    try:
        publish(
            match.group(1),
            args.publisher,
            args.server_json,
            args.package_json,
            registry=args.registry,
            owner=args.owner,
            token=os.environ.get("MCP_GITHUB_TOKEN", ""),
            wait=args.wait,
            **hooks,
        )
    except Refusal as err:
        fail(str(err))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
