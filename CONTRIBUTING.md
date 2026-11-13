# Contributing to crabcrop

Thanks for your interest in improving crabcrop. This guide covers how to build,
test, and propose changes.

## Proposing a change

- For anything beyond a small fix, open an issue first so the approach can be
  discussed before you spend time on it.
- Submit changes as a pull request against `main`.
- Keep each pull request focused on one logical change; it is much easier to
  review.
- Reference the related issue in the pull request. The commit history uses a
  `short summary (#123)` style, where `#123` is the issue or pull request.

## Building

```bash
cargo build
```

AVIF source decoding is behind the `avif-decode` Cargo feature (off by default).
Building with it needs dav1d 1.3 or newer reachable through pkg-config: a
system libdav1d (`libdav1d-dev` on Debian and Ubuntu), or the static dav1d the
release links, which `scripts/build_dav1d.sh` builds from a pinned and
checksummed tarball:

```bash
eval "$(bash scripts/build_dav1d.sh "$HOME/.cache/crabcrop-dav1d")"
cargo build --features avif-decode
```

Running the server locally needs only a secret:

```bash
CRABCROP_SECRET="$(openssl rand -hex 16)" cargo run
```

## Testing

Rust tests:

```bash
cargo test
cargo test -p crabcrop-url   # the signing crate
```

Both include property tests ([proptest](https://docs.rs/proptest)):
`tests/proptest_url.rs` (generated operation chains parse back to the
operations they spell and canonicalize to a fixed point),
`tests/proptest_crop.rs` (every crop window is non-empty, inside the image,
no larger than the fill window, and at the target aspect within a pixel), and
`crates/crabcrop-url/tests/proptest_sign.rs` (signatures and sealed tokens
round-trip, and any one changed byte fails them). Each property runs 64 cases
by default; `PROPTEST_CASES` runs more:

```bash
PROPTEST_CASES=10000 cargo test --workspace --test proptest_url --test proptest_crop --test proptest_sign
```

A failing case is saved as a seed in the `.proptest-regressions` file next to
its suite and replayed first on every later run. Commit the seed with the fix.

The full cross-language suite (Rust, the TypeScript and Python SDKs, the MCP
server, the Nuxt module, and the playground, plus a fuzz smoke test) runs from
one script, which starts and stops the local servers the integration tests
need:

```bash
scripts/test_all.sh              # everything
scripts/test_all.sh --skip-rust  # just the non-Rust suites
```

The JavaScript suites need Node 22.18 or newer: they run the TypeScript sources directly, and type stripping is on by default from 22.18.

A unit test that stands a script in for an external tool (FFmpeg,
pdftocairo, pdfinfo) writes it with `crate::test_support::write_script`, not
`std::fs::write`. A script written through the test process's own file
descriptor can fail to start with "Text file busy" when another test thread
spawns a child at the same moment (issue #523).

A test that writes files makes them in a directory of its own from
`crate::test_support::scratch_dir(tag)`: it starts empty under
`std::env::temp_dir()` and is removed when dropped. The binary's unit tests
and `tests/*.rs` cannot see the library's test modules, so they include the
same file, `src/test_support/scratch.rs`, with `#[path]`: the binary's tests
directly, and `tests/*.rs` through the module they share, `tests/common`
(`common::scratch_dir`). Never build a temp name from the process ID alone:
IDs repeat, and a later run inherits what an earlier one left (issue #526).
Quote a path you splice into a fake tool's script (`>> '{}'`), so a temp dir
with a space in it still works.

An integration test that boots the server starts it with `common::serve`,
which runs the binary with `PORT=0` and returns the port its listening line
names. It reads the server's stdout and stderr itself, so leave both alone;
`ServerGuard::stderr` returns what the server wrote there. Never pick a free
port in the test and hand it to the server: another process can take it
before the server binds, or another test's server can answer on it
(issue #536).

## Code style

CI enforces formatting and lints. Run both before pushing:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

## The URL grammar is a shared contract

The URL grammar and signing scheme are specified in
[`spec/url-signing.md`](spec/url-signing.md), with canonical test vectors in
[`spec/test-vectors.json`](spec/test-vectors.json) that every implementation
runs as its test suite. If you change the grammar or the signing scheme, update
the spec and the vectors, and keep the SDKs (Rust, TypeScript, Python) and the
playground in sync so they all still pass the shared vectors.

The grammar has one JavaScript implementation:
[`sdks/typescript/src/explain.ts`](sdks/typescript/src/explain.ts). The typed
builders in `operations.ts` import its checks, and the playground serves its
build from `playground/sdk/`. To change what JavaScript accepts (a new filter,
say), edit `explain.ts` and the builders, add the cases to
`spec/generate_operation_vectors.py` or `spec/explain-verdicts.json`, and
refresh the playground's copy:

```bash
(cd sdks/typescript && npm ci)   # once
python3 scripts/sync_playground_sdk.py
```

`scripts/test_all.sh` runs the script with `--check`, which fails when
`playground/sdk/` is stale or when a second copy of the grammar appears in the
JavaScript sources. The copy is `tsc` output, so a TypeScript bump in
`sdks/typescript` (a Dependabot pull request, say) can change it: run the
script on that branch too.

## Changelog

Give user-facing changes a changelog entry. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Write the entry as a
fragment: a file of its own in [`changelog.d/`](changelog.d/), which the
release moves into [CHANGELOG.md](CHANGELOG.md). Every pull request adds a
different file, so entries no longer conflict when another pull request
merges first (issue #539).

- **Name** the file `changelog.d/<issue>.<section>.md`, for example
  `changelog.d/521.changed.md`. The section is one of `added`, `changed`,
  `deprecated`, `removed`, `fixed` or `security`. A short slug after the
  number, in lowercase words joined by hyphens, is always allowed:
  `changelog.d/521-chained-resize.changed.md`. Use one when the plain name is
  taken (a second entry for the same issue and section), and also when
  another pull request for the same issue may add an entry to that section
  while yours is open: two pull requests that both add `521.changed.md`
  conflict, as two CHANGELOG.md lines would.
- **Name an issue.** Every fragment names one. A change with no issue opens
  one first, unless it cannot name a public issue, such as the fix for a
  private security advisory: that change keeps its entry as a line under
  `## [Unreleased]` in CHANGELOG.md (below), which needs no issue number.
- **Write** one paragraph in the style of the entries in CHANGELOG.md, naming
  the issue the same way, `(issue #521)` or `(issues #604 and #613)`. The
  issue in the file name must be one of those the text names. Leave out the
  leading `- `: the release adds it. You may wrap the paragraph; its lines are
  joined with a space, but no line may open with `- `, `* `, `+ ` or `# `,
  which would start a second entry or a heading: rewrap it instead. A blank
  line, a second bullet or a heading is refused, since an entry is one
  paragraph and a second entry is a second fragment.
- **Preview** `## [Unreleased]` as the release will write it. Each section
  lists the entries already written under `## [Unreleased]` first, as they
  are, then the fragments in issue order:

  ```bash
  python3 scripts/assemble_changelog.py --check
  ```

A pull request that raises `RENDERING_VERSION` also says that the release
starts a new cache epoch (RELEASING.md step 4), in an entry that opens with
"This release starts a new cache epoch". As a fragment, name it
`<issue>.changed.md`: the release puts it first under Changed, ahead of the
other entries, since it describes the changes below it. A release has one
such note, so a later pull request whose change belongs to the same epoch adds
its change to that note, the fragment or the CHANGELOG.md line, instead of
writing a second one.

A line under `## [Unreleased]` in CHANGELOG.md still counts as the entry, so
a pull request opened before fragments existed needs no change, and a change
that cannot name a public issue uses one. Such lines conflict with each
other, so otherwise prefer a fragment.

The changelog workflow enforces this: a pull request that touches `src/`,
`sdks/`, `frameworks/`, or `plugins/` fails until it adds a fragment or
changes one's entry, or adds a line under `## [Unreleased]`. When nothing
user-facing changed (a refactor, a test, a dependency bump), a maintainer adds
the `no-changelog` label instead, and the check passes. A fragment the pull
request adds or edits must be well formed even then, a second cache-epoch note
fails, and a file named like a fragment outside `changelog.d/` fails, so a
fragment the release could not assemble never reaches `main`;
`scripts/check_docs.py`, which `scripts/test_all.sh` runs, checks every
fragment in the tree too. To run the same check locally, against the branch
point with `main`:

```bash
python3 scripts/check_docs.py --changelog-entry origin/main
```

## Reference docs

The reference pages in `docs/` move with the code. `scripts/check_docs.py`,
which `scripts/test_all.sh` runs, fails when an environment variable, a
default, an error code or status, or a filter is in the code but not the docs
(or the other way round), and when a documented error code is named in no test.
Its output names the file and the entry to fix.

## Security

Please do not report security vulnerabilities as public issues. See
[SECURITY.md](SECURITY.md) for private reporting.

## Releases

Releases are tag-driven and handled by the maintainer; see
[RELEASING.md](RELEASING.md).
