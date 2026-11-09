#!/usr/bin/env bash
# Create the GitHub Release for a tag, or publish the draft that
# already exists (issue #457). Refuse a release that is already
# published (issue #552).
#
# The release workflow's host job runs this in place of
# cargo-dist's plain `gh release create`, which fails with "a
# release with the same tag name already exists" whenever a
# release for the tag is already there. Every tag run from v0.9.8
# to v0.9.11 failed that way on its first attempt. With no release
# yet, this creates it exactly as the generated step did. With a
# draft already there (made by hand, or left by an earlier
# attempt), it uploads the assets (replacing any of the same name)
# and then sets the title, notes, target, and prerelease flag and
# publishes it, so the release ends up as a fresh create would
# leave it. Assets with other names, such as a Homebrew formula
# uploaded by hand, stay.
#
# A release that is already published is refused. Either it was
# published by hand, so its page went public before this run's
# gates passed (v0.10.0's was, which also created the tag, and the
# page had no installer for 37 minutes until the run caught up),
# or an earlier attempt of this workflow published it, and
# re-running the job would replace assets people may already have
# downloaded. The script stops and prints the fix instead. With
# ADOPT_PUBLISHED_RELEASE=1 (the workflow passes the repository
# variable of that name) it adopts the release anyway, with a
# warning.
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
if is_draft=$(gh release view --json isDraft --jq .isDraft -- "$tag"); then
  case "$is_draft" in
    true) ;;
    false)
      if [ "${ADOPT_PUBLISHED_RELEASE:-}" != 1 ]; then
        echo "::error title=Release $tag is already published::Refusing to adopt it: the release workflow is the only way to publish a release. The log says how to recover."
        {
          echo "publish_github_release.sh: the release for $tag is already published; refusing to adopt it."
          echo "Publishing a release by hand makes its page public before the gates pass:"
          echo "the release workflow is the only way to publish one (RELEASING.md)."
          echo "If it was published by hand, turn it back into a draft, then re-run this job:"
          echo "  gh release edit $tag --draft=true"
          echo "If an earlier attempt of this workflow published it, this job already passed:"
          echo "re-run only the failed jobs instead."
          echo "To adopt it as it is, set the repository variable ADOPT_PUBLISHED_RELEASE"
          echo "to 1 for the re-run, and delete it afterwards."
        } >&2
        exit 1
      fi
      echo "::warning title=Adopting published release $tag::ADOPT_PUBLISHED_RELEASE is 1, so this run uploads to and updates the release for $tag, which was already published."
      ;;
    *)
      echo "publish_github_release.sh: cannot tell whether the release for $tag is a draft (gh printed '$is_draft')" >&2
      exit 1
      ;;
  esac
  echo "A release for $tag already exists; uploading the assets and updating it in place."
  gh release upload --clobber -- "$tag" "$@"
  gh release edit --target "$commit" --title "$title" --notes-file "$notes" \
    --prerelease="$prerelease" --draft=false -- "$tag"
else
  echo "No release for $tag yet; creating it."
  gh release create --target "$commit" --prerelease="$prerelease" --title "$title" \
    --notes-file "$notes" -- "$tag" "$@"
fi
