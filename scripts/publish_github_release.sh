#!/usr/bin/env bash
# Create the GitHub Release for a tag, or update the one that
# already exists (issue #457).
#
# The release workflow's host job runs this in place of
# cargo-dist's plain `gh release create`, which fails with "a
# release with the same tag name already exists" whenever a
# release for the tag is already there: a draft made by hand, or
# the release an earlier attempt of the run created before a later
# step failed. Every tag run from v0.9.8 to v0.9.11 failed that way
# on its first attempt. With no release yet, this creates it
# exactly as the generated step did. With one already there, it
# uploads the assets (replacing any of the same name) and then sets
# the title, notes, target, and prerelease flag and publishes it,
# so the release ends up as a fresh create would leave it. Assets
# with other names, such as a Homebrew formula uploaded by hand,
# stay.
#
# `gh release view` finds drafts as well as published releases. If
# it fails for any other reason, the create runs as before and
# reports the real error.
#
# Usage: bash scripts/publish_github_release.sh TAG COMMIT TITLE \
#          NOTES_FILE PRERELEASE FILE...
#   PRERELEASE is `true` or `false`; FILE... are the assets.
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "usage: publish_github_release.sh TAG COMMIT TITLE NOTES_FILE true|false FILE..." >&2
  exit 2
fi
tag=$1 commit=$2 title=$3 notes=$4 prerelease=$5
shift 5

if [ -z "$tag" ] || [ -z "$commit" ] || [ -z "$title" ]; then
  echo "publish_github_release.sh: TAG, COMMIT, and TITLE must not be empty" >&2
  exit 2
fi
case "$prerelease" in
  true | false) ;;
  *)
    echo "publish_github_release.sh: PRERELEASE must be true or false, not '$prerelease'" >&2
    exit 2
    ;;
esac
if [ ! -f "$notes" ]; then
  echo "publish_github_release.sh: no notes file at '$notes'" >&2
  exit 2
fi
for asset in "$@"; do
  if [ ! -f "$asset" ]; then
    echo "publish_github_release.sh: asset '$asset' is not a file" >&2
    exit 2
  fi
done

# Every positional argument follows `--`, so a tag or an asset
# name that starts with a dash is never read as a flag.
if gh release view --json tagName,isDraft -- "$tag" >/dev/null; then
  echo "A release for $tag already exists; uploading the assets and updating it in place."
  gh release upload --clobber -- "$tag" "$@"
  gh release edit --target "$commit" --title "$title" --notes-file "$notes" \
    --prerelease="$prerelease" --draft=false -- "$tag"
else
  echo "No release for $tag yet; creating it."
  gh release create --target "$commit" --prerelease="$prerelease" --title "$title" \
    --notes-file "$notes" -- "$tag" "$@"
fi
