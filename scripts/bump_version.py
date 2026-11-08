#!/usr/bin/env python3
"""Set every artifact version to X.Y.Z in one pass.

Usage: python3 scripts/bump_version.py 0.2.0 [--dry-run]

Edits each site listed in scripts/version_sites.py in place, the list
check_versions.py reads, then runs the check. The sites include the
lockfiles' own entries (the crabcrop and crabcrop-url packages in
Cargo.lock and fuzz/Cargo.lock, the version and linked @crabcrop/url
fields of every package-lock.json), so no lockfile refresh follows, and
on a minor bump the server's crabcrop-url requirement. A site that is
not where the list says (a pin count that changed, a field gone) stops
the bump before anything is written.
"""
import re
import subprocess
import sys

import version_sites


def main():
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry = "--dry-run" in sys.argv
    if len(args) != 1 or not re.fullmatch(r"\d+\.\d+\.\d+", args[0]):
        raise SystemExit("usage: bump_version.py X.Y.Z [--dry-run]")
    version = args[0]
    changed, problems = version_sites.bump(version)
    if problems:
        print("cannot bump; nothing was written:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    for rel, text in changed.items():
        if dry:
            print(f"would update {rel}")
        else:
            (version_sites.ROOT / rel).write_text(text, encoding="utf-8")
            print(f"updated {rel}")
    if dry:
        return 0
    return subprocess.call([sys.executable, str(version_sites.ROOT / "scripts/check_versions.py")])


if __name__ == "__main__":
    sys.exit(main())
