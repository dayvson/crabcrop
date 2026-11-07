#!/usr/bin/env bash
# Print the npm dist-tag a version publishes under (issue #413).
#
# npm puts every publish on the `latest` dist-tag unless told
# otherwise, prerelease or not, and `latest` is what a bare
# `npm install @crabcrop/url` or `npx -y @crabcrop/mcp` resolves.
# A prerelease (a SemVer version with a pre-release part, as in
# 0.10.0-rc.1) therefore publishes under `next`, and only a plain
# release moves `latest`: the same rule the release workflow
# applies to the container images' :latest tags.
#
# Usage: bash scripts/npm_dist_tag.sh <version>
set -euo pipefail

version="${1:?usage: npm_dist_tag.sh <version>}"
semver='^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$'
if ! [[ "$version" =~ $semver ]]; then
  echo "npm_dist_tag.sh: not a SemVer version: $version" >&2
  exit 1
fi

# Build metadata (after `+`) may contain hyphens and never makes
# a version a prerelease; only the part before it decides.
case "${version%%+*}" in
  *-*) echo next ;;
  *) echo latest ;;
esac
