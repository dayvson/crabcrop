#!/usr/bin/env python3
"""Fail when the upgrade corpus was re-baselined without a rendering version bump.

tests/upgrade/corpus.json records what the previous release served for each
frozen URL. Re-baselining an entry (changing its recorded status, content type,
byte length, or digest) admits that the build now renders that URL differently.
Every cache key carries the configuration fingerprint, and its only
output-epoch input is RENDERING_VERSION (src/main.rs), so a release that
re-baselines without raising it keeps serving the previous release's renders
from the disk and shared tiers after the upgrade (issue #412).

The check compares the working tree with the last release tag reachable from
HEAD: when an entry present at the tag records a different outcome now,
RENDERING_VERSION must be greater than it was at the tag. Added and removed
entries are not re-baselines, and neither is a re-minted path whose recorded
outcome is unchanged (the sealed entry gets a fresh nonce on every freeze).

A clone without the tag (CI's shallow checkout fetches no tags) cannot make
the comparison; the check says so and exits 0. Run it from a full clone, as
the release steps in RELEASING.md do.

Usage: scripts/check_rendering_version.py [--root DIR]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = "tests/upgrade/corpus.json"
GOLDEN_FIELDS = ("status", "content_type", "bytes", "sha256")
CONSTANT = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?const RENDERING_VERSION: u32 = (\d+);", re.M)


class CheckError(Exception):
    """A state the check cannot judge: the gate fails with this message."""


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )


def last_release_tag(root):
    """The nearest vX.Y.Z tag reachable from HEAD, or None with the reason."""
    inside = git(root, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return None, "not a git checkout"
    found = git(root, "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", "HEAD")
    if found.returncode == 0:
        return found.stdout.strip(), None
    shallow = git(root, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    if shallow:
        return None, (
            "no release tag is reachable from HEAD in this shallow clone; "
            "fetch the history and tags (git fetch --tags --unshallow) to run it"
        )
    return None, "no release tag (v*) is reachable from HEAD"


def versions_in(texts, where):
    """The RENDERING_VERSION values declared across the given sources."""
    found = [int(m.group(1)) for text in texts for m in CONSTANT.finditer(text)]
    if len(found) > 1:
        raise CheckError(f"RENDERING_VERSION is declared {len(found)} times in {where}")
    return found[0] if found else None


def current_version(root):
    sources = sorted((root / "src").rglob("*.rs"))
    texts = [p.read_text(encoding="utf-8") for p in sources]
    version = versions_in(texts, "src/")
    if version is None:
        raise CheckError(
            "RENDERING_VERSION was not found under src/; "
            "update scripts/check_rendering_version.py if it moved"
        )
    return version


def tag_version(root, tag):
    """The version declared at the tag; 0 when the tag predates the constant."""
    found = git(root, "grep", "-h", "-E", r"const RENDERING_VERSION: u32 = [0-9]+;", tag, "--", "src")
    if found.returncode not in (0, 1):
        raise CheckError(f"git grep at {tag} failed: {found.stderr.strip()}")
    version = versions_in([found.stdout], f"src/ at {tag}")
    return 0 if version is None else version


def goldens(text, where):
    try:
        entries = json.loads(text)["entries"]
        by_id = {e["id"]: tuple(e.get(f) for f in GOLDEN_FIELDS) for e in entries}
    except (ValueError, KeyError, TypeError) as err:
        raise CheckError(f"{CORPUS} at {where} is not a readable corpus: {err!r}") from err
    if len(by_id) != len(entries):
        raise CheckError(f"{CORPUS} at {where} repeats an entry id")
    return by_id


def check(root):
    """Return (exit code, message)."""
    tag, reason = last_release_tag(root)
    if tag is None:
        return 0, f"rendering version check skipped: {reason}"
    before = git(root, "show", f"{tag}:{CORPUS}")
    if before.returncode != 0:
        return 0, f"rendering version check: {tag} has no {CORPUS}; nothing to compare"
    corpus_path = root / CORPUS
    if not corpus_path.is_file():
        raise CheckError(f"{CORPUS} is missing from the working tree")
    old = goldens(before.stdout, tag)
    new = goldens(corpus_path.read_text(encoding="utf-8"), "the working tree")
    rebaselined = sorted(i for i in old.keys() & new.keys() if old[i] != new[i])
    was, now = tag_version(root, tag), current_version(root)

    if now < was:
        return 1, (
            f"RENDERING_VERSION went backwards: {now} now, {was} at {tag}. "
            "A lower value can reuse the cache keys of an older epoch; raise it instead."
        )
    if not rebaselined:
        return 0, (
            f"rendering version check: no corpus entry re-baselined since {tag} "
            f"(RENDERING_VERSION {now})"
        )
    names = ", ".join(rebaselined)
    if now > was:
        return 0, (
            f"rendering version check: {len(rebaselined)} corpus entries re-baselined since "
            f"{tag} ({names}), covered by RENDERING_VERSION {was} -> {now}"
        )
    return 1, (
        f"{len(rebaselined)} upgrade corpus entries were re-baselined since {tag} "
        f"({names}), but RENDERING_VERSION is still {now}. A re-baseline means existing "
        "URLs render differently, and without a new rendering version upgraded servers "
        "keep serving the old renders from the persistent tiers. Raise RENDERING_VERSION "
        "in src/main.rs and say in CHANGELOG.md that the release starts a new cache epoch "
        "(see RELEASING.md)."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="repository checkout to check")
    args = parser.parse_args(argv)
    try:
        code, message = check(args.root.resolve())
    except CheckError as err:
        code, message = 1, f"rendering version check failed: {err}"
    print(message, file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())
