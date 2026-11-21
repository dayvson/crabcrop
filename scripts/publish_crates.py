#!/usr/bin/env python3
"""Publish the release's crates to crates.io in order, each version once.

The release workflow's publish-crate job runs this (issue #711) with the
release tag and the workspace's crates, each after the ones it depends
on:

    python3 scripts/publish_crates.py --tag v0.14.0 --no-verify crabcrop-url crabcrop

crabcrop-url went up from the workflow since the first release; the
crabcrop crate, which `cargo install --locked crabcrop` builds, was
published by hand, when at all. Before it publishes anything, the script
checks that every crate named carries the tag's version (cargo publishes
the version in Cargo.toml, whatever the tag says) and comes after the
crates it depends on. Then, for each crate:

- When the crates.io index already lists the version, the crate is
  skipped. An earlier attempt of the job published it, and crates.io
  takes each version once, so re-running the failed job publishes only
  what that attempt did not.
- Otherwise `cargo publish -p CRATE --locked` packages the crate, builds
  the packaged source, and uploads it. With --no-verify it builds
  nothing: the workflow's package-crates job built both packages before
  the release approval, in a job with no secrets, so no build script or
  procedural macro of a dependency runs beside the crates.io token. The
  script then waits until the index lists the version. cargo waits too,
  but gives up after a minute and still succeeds, and the next crate
  needs the version listed: the crabcrop crate's packaged Cargo.lock
  names crabcrop-url from crates.io, so packaging it fails until the
  index has that version, or, for a patch release, names the previous
  one, which the server's `0.y` requirement also allows, and nothing
  would notice.
- So before it publishes a crate that depends on one named before it,
  the script packages that crate with `cargo package --no-verify`,
  which builds nothing, and reads the Cargo.lock in the package: it
  must name the dependency at the release's version. Until it does
  (the index cargo reads can lag the one the script reads), the script
  waits and packages again, and past the wait it stops without
  publishing the crate.

The index is the sparse one cargo reads (https://index.crates.io), and
its answer decides: a crate it does not know (404) has no version yet,
and any other failure stops the job before it publishes, since the
script cannot tell what is there.

Usage: publish_crates.py --tag vX.Y.Z CRATE... [--no-verify]
       [--index URL] [--wait SECONDS] [--poll SECONDS]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

INDEX = "https://index.crates.io"
TAG = re.compile(r"v(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)")
CRATE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
# A Cargo.lock entry's first two lines, as cargo writes them. The
# release job's python3 may predate tomllib.
LOCK_ENTRY = re.compile(r'^name = "([^"]+)"\nversion = "([^"]+)"$', re.M)


class Refusal(Exception):
    """A reason to stop before, or instead of, the next publish."""


def index_path(name):
    """The crate's file in the sparse index, as cargo names it."""
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def http_get(url):
    """GET `url`; (status, body text). A status that is no answer raises Refusal."""
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, ""
    except (urllib.error.URLError, OSError) as err:
        raise Refusal(f"could not read {url}: {err}") from err


def listed(name, version, index, fetch):
    """Whether the index lists `version` of `name`, yanked or not."""
    url = f"{index.rstrip('/')}/{index_path(name)}"
    status, body = fetch(url)
    if status == 404:
        return False
    if status != 200:
        raise Refusal(f"the crates.io index answered {status} for {url}, so it is unknown whether {name} {version} is published")
    for line in body.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError as err:
            raise Refusal(f"the crates.io index file for {name} has a line that is not JSON") from err
        if isinstance(entry, dict) and entry.get("vers") == version:
            return True
    return False


def workspace(run):
    """({name: (version, [names of its path dependencies])}, target directory) for the workspace."""
    done = run(["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"], capture=True)
    if done.returncode != 0:
        raise Refusal(f"cargo metadata failed: {done.stderr.strip()[:300]}")
    try:
        metadata = json.loads(done.stdout)
        packages = {
            package["name"]: (
                package["version"],
                [dep["name"] for dep in package.get("dependencies", []) if dep.get("path") and dep.get("kind") in (None, "normal", "build")],
            )
            for package in metadata["packages"]
        }
        target = metadata["target_directory"]
    except (ValueError, KeyError, TypeError) as err:
        raise Refusal("cargo metadata printed no package list and target directory") from err
    if not isinstance(target, str) or not target:
        raise Refusal("cargo metadata printed no target directory")
    return packages, target


def packaged_lock(name, version, target, run):
    """{crate: {versions}} that the Cargo.lock in `name`'s package names.

    `cargo package --no-verify` makes the package as `cargo publish`
    would, its Cargo.lock resolved against the index cargo reads, and
    builds nothing.
    """
    done = run(["cargo", "package", "-p", name, "--locked", "--no-verify"], capture=True)
    if done.returncode != 0:
        raise Refusal(f"cargo package -p {name} failed (exit {done.returncode}): {done.stderr.strip()[-300:]}")
    crate = Path(target) / "package" / f"{name}-{version}.crate"
    try:
        with tarfile.open(crate, "r:gz") as archive:
            member = archive.extractfile(f"{name}-{version}/Cargo.lock")
            text = member.read().decode("utf-8") if member is not None else ""
    except (OSError, KeyError, EOFError, tarfile.TarError, UnicodeDecodeError) as err:
        raise Refusal(f"cannot read the Cargo.lock in {crate}: {err}") from err
    locked = {}
    for entry, entry_version in LOCK_ENTRY.findall(text):
        locked.setdefault(entry, set()).add(entry_version)
    return locked


def wait_until_locked(name, version, deps, target, run, wait, poll, sleep):
    """Wait until `name`'s packaged Cargo.lock names each of `deps` at `version` alone.

    A package cargo cannot make yet counts as one that names another
    version: a requirement that allows no earlier version (a minor
    release's) fails to resolve until the index cargo reads lists the
    release, where one that does (a patch release's) resolves to it.
    """
    deadline = time.monotonic() + wait
    while True:
        try:
            locked = packaged_lock(name, version, target, run)
        except Refusal as err:
            problem = str(err)
        else:
            stale = [f"{dep} {', '.join(sorted(locked.get(dep, ()))) or '(none)'}" for dep in deps if locked.get(dep) != {version}]
            if not stale:
                return
            problem = f"{name}'s package locks {'; '.join(stale)}, not {version}"
        if time.monotonic() >= deadline:
            raise Refusal(
                f"{problem}; {name} is not published. After {wait} s the index cargo reads may still not list "
                f"{' and '.join(deps)} {version}, though the one this script reads does: re-run the failed job"
            )
        sleep(poll)


def plan_problems(crates, version, packages):
    """Why the crates cannot be published for `version` in this order; empty when they can."""
    problems, seen = [], set()
    for name in crates:
        if name not in packages:
            problems.append(f"{name} is not a package of this workspace")
            continue
        have, deps = packages[name]
        if have != version:
            problems.append(f"{name} is at {have}, not {version}, the version the tag names; bump it (RELEASING.md, step 2)")
        for dep in deps:
            if dep in crates and dep not in seen:
                problems.append(f"{name} depends on {dep}, so {dep} must come before it")
        seen.add(name)
    if len(set(crates)) != len(crates):
        problems.append("a crate is named twice")
    return problems


def run_command(args, capture=False):
    return subprocess.run(args, capture_output=capture, text=True)


def wait_until_listed(name, version, index, fetch, wait, poll, sleep):
    deadline = time.monotonic() + wait
    while True:
        if listed(name, version, index, fetch):
            return
        if time.monotonic() >= deadline:
            raise Refusal(
                f"{name} {version} was published, but the crates.io index has not listed it within {wait} s; "
                "re-run the failed job, which skips it once it is listed"
            )
        sleep(poll)


def publish(
    crates, version, index=INDEX, fetch=http_get, run=run_command, wait=600, poll=10, sleep=time.sleep, out=print, verify=True
):
    """Publish each crate not yet listed; the names of those published."""
    packages, target = workspace(run)
    problems = plan_problems(crates, version, packages)
    if problems:
        raise Refusal("; ".join(problems))
    published = []
    for name in crates:
        if listed(name, version, index, fetch):
            out(f"{name} {version} is already on crates.io; skipping it")
            continue
        deps = [dep for dep in packages[name][1] if dep in crates]
        if deps:
            wait_until_locked(name, version, deps, target, run, wait, poll, sleep)
            out(f"{name}'s package locks {', '.join(deps)} at {version}")
        out(f"publishing {name} {version}")
        done = run(["cargo", "publish", "-p", name, "--locked", *([] if verify else ["--no-verify"])])
        if done.returncode != 0:
            later = crates[crates.index(name) + 1 :]
            raise Refusal(
                f"cargo publish -p {name} failed (exit {done.returncode})"
                + (f"; {', '.join(later)} not published" if later else "")
            )
        wait_until_listed(name, version, index, fetch, wait, poll, sleep)
        out(f"{name} {version} is on crates.io")
        published.append(name)
    return published


def fail(message):
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::error title=crates.io publish::{message}")
    else:
        print(f"publish_crates.py: {message}", file=sys.stderr)


def main(argv=None, **hooks):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="the release tag, vX.Y.Z")
    parser.add_argument("crates", nargs="+", help="the crates, each after the ones it depends on")
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="publish without building the packages (cargo publish --no-verify), once a job without the token built them",
    )
    parser.add_argument("--index", default=INDEX, help="the sparse index (default %(default)s)")
    parser.add_argument("--wait", type=int, default=600, help="seconds to wait for the index to list a publish")
    parser.add_argument("--poll", type=int, default=10, help="seconds between index reads while waiting")
    args = parser.parse_args(argv)
    match = TAG.fullmatch(args.tag)
    bad = [name for name in args.crates if not CRATE.fullmatch(name)]
    if not match or bad or args.wait < 0 or args.poll < 1:
        fail(f"usage: --tag must be vX.Y.Z (got {args.tag!r}) and each crate a package name (got {bad or 'ok'})")
        return 2
    try:
        published = publish(args.crates, match.group(1), args.index, wait=args.wait, poll=args.poll, verify=args.verify, **hooks)
    except Refusal as err:
        fail(str(err))
        return 1
    hooks.get("out", print)(f"crates.io has {', '.join(args.crates)} {match.group(1)} ({len(published)} published by this run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
