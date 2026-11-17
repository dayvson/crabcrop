#!/usr/bin/env python3
"""Verify that every shipped artifact carries the same version.

crabcrop releases one version number across all artifacts (RELEASING.md).
This script reads each site listed in scripts/version_sites.py, the list
scripts/bump_version.py writes, and fails if any disagrees with the
server's own version or cannot be read as listed (a pin count that
changed, a lockfile entry gone). It is run by scripts/bump_version.py
after an edit, and belongs in CI.

It also reads the dav1d pin in scripts/build_dav1d.sh, which no release
bump moves, and fails when a doc that names the shipped dav1d names
another version (version_sites.DAV1D_MENTIONS, issue #596), and the
release commit pins (version_sites.RELEASE_COMMITS, issue #711), which
must hold nothing, as a bump leaves them, or a full commit SHA, and in a
clone that has that commit, one of the release the tree carries.
"""
import sys

import version_sites


def main():
    version, rows, problems = version_sites.check()
    dav1d, dav1d_rows, dav1d_problems = version_sites.check_dav1d()
    commit_rows, commit_problems = version_sites.check_commits(version=version)
    problems += dav1d_problems + commit_problems
    for values, site in rows + dav1d_rows:
        shown = ",".join(dict.fromkeys(values)) if values else "MISSING"
        print(f"{shown:>10}  {site.path}: {site.what}")
    for values, site in commit_rows:
        shown = "MISSING" if values is None else ",".join(value[:12] or "(empty)" for value in values)
        print(f"{shown:>10}  {site.path}: {site.what}; a bump empties it")
    if dav1d:
        print(f"{dav1d:>10}  {version_sites.DAV1D_PIN}: dav1d pin (not a release version; a bump leaves it)")
    if problems:
        print("\nversion mismatch across artifacts:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"\nall artifacts at {version}; dav1d pinned at {dav1d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
