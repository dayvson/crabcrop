#!/usr/bin/env python3
"""Verify that every shipped artifact carries the same version.

crabcrop releases one version number across all artifacts (RELEASING.md).
This script reads each site listed in scripts/version_sites.py, the list
scripts/bump_version.py writes, and fails if any disagrees with the
server's own version or cannot be read as listed (a pin count that
changed, a lockfile entry gone). It is run by scripts/bump_version.py
after an edit, and belongs in CI.
"""
import sys

import version_sites


def main():
    version, rows, problems = version_sites.check()
    for values, site in rows:
        shown = ",".join(dict.fromkeys(values)) if values else "MISSING"
        print(f"{shown:>10}  {site.path}: {site.what}")
    if problems:
        print("\nversion mismatch across artifacts:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"\nall artifacts at {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
