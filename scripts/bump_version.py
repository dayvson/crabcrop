#!/usr/bin/env python3
"""Set every artifact version to X.Y.Z in one pass, and assemble the CHANGELOG.

Usage: python3 scripts/bump_version.py 0.2.0 [--date YYYY-MM-DD] [--dry-run]

Edits each site listed in scripts/version_sites.py in place, the list
check_versions.py reads, then runs the check. The sites include the
lockfiles' own entries (the crabcrop and crabcrop-url packages in
Cargo.lock and fuzz/Cargo.lock, the version and linked @crabcrop/url
fields of every package-lock.json), so no lockfile refresh follows, and
on a minor bump the server's crabcrop-url requirement. A site that is
not where the list says (a pin count that changed, a field gone) stops
the bump before anything is written.

It also writes the release's CHANGELOG section with
scripts/assemble_changelog.py: the ## [Unreleased] entries and every
changelog.d/ fragment move into ## [X.Y.Z] - YYYY-MM-DD (today unless
--date says otherwise), an empty ## [Unreleased] stays above it, and the
fragments are deleted (RELEASING.md step 1). A malformed fragment stops
the bump before anything is written, like a misplaced site.
"""
import argparse
import re
import subprocess
import sys

import assemble_changelog
import version_sites


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("version", help="the release version, X.Y.Z")
    parser.add_argument("--date", help="the CHANGELOG section's date, YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="say what would change; write nothing")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        raise SystemExit("usage: bump_version.py X.Y.Z [--date YYYY-MM-DD] [--dry-run]")
    version = args.version
    changed, problems = version_sites.bump(version)
    try:
        changelog = assemble_changelog.plan(version_sites.ROOT, version, args.date)
    except assemble_changelog.AssembleError as e:
        problems = [*problems, *e.problems]
    if problems:
        print("cannot bump; nothing was written:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    for rel, text in changed.items():
        if args.dry_run:
            print(f"would update {rel}")
        else:
            (version_sites.ROOT / rel).write_text(text, encoding="utf-8")
            print(f"updated {rel}")
    for line in assemble_changelog.apply(version_sites.ROOT, changelog, args.dry_run):
        print(line)
    if args.dry_run:
        return 0
    return subprocess.call([sys.executable, str(version_sites.ROOT / "scripts/check_versions.py")])


if __name__ == "__main__":
    sys.exit(main())
