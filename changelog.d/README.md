# Changelog fragments

A pull request with a user-facing change adds its CHANGELOG entry here, as a
file of its own, instead of a line in [CHANGELOG.md](../CHANGELOG.md). No two
pull requests edit the same file, so they merge in either order without a
conflict (issue #539). The release moves the fragments into CHANGELOG.md and
deletes them.

- Name the file `<issue>.<section>.md`, for example `521.changed.md`. The
  section is one of `added`, `changed`, `deprecated`, `removed`, `fixed`,
  `security`. A slug is always allowed, `521-chained-resize.changed.md`: use
  one for a second entry with the same issue and section, and whenever
  another open pull request for that issue may add one to that section.
- Write one paragraph in the style of CHANGELOG.md, naming the issue as the
  entries there do: `(issue #521)`. Leave out the leading `- `.
- A change that cannot name a public issue, such as the fix for a private
  security advisory, writes a line under `## [Unreleased]` in CHANGELOG.md
  instead.
- Preview `## [Unreleased]` with the fragments in place:
  `python3 scripts/assemble_changelog.py --check`.

[CONTRIBUTING.md](../CONTRIBUTING.md#changelog) has the full rules and
RELEASING.md step 1 the release side. This README is not a fragment: the
tools skip it.
