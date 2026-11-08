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
Building with it needs dav1d 1.3 or newer reachable through pkg-config, or a
vendored build:

```bash
SYSTEM_DEPS_DAV1D_BUILD_INTERNAL=always cargo build --features avif-decode
```

Running the server locally needs only a secret:

```bash
CRABCROP_SECRET=local-dev-secret cargo run
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
same file, `src/test_support/scratch.rs`, with `#[path]`. Never build a temp
name from the process ID alone: IDs repeat, and a later run inherits what an
earlier one left (issue #526). Quote a path you splice into a fake tool's
script (`>> '{}'`), so a temp dir with a space in it still works.

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

Give user-facing changes an entry under `## [Unreleased]` in
[CHANGELOG.md](CHANGELOG.md). The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The changelog workflow enforces this: a pull request that touches `src/`,
`sdks/`, `frameworks/`, or `plugins/` fails until it adds a line under
`## [Unreleased]`. When nothing user-facing changed (a refactor, a test, a
dependency bump), a maintainer adds the `no-changelog` label instead, and the
check passes. To run the same check locally, against the branch point with
`main`:

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
