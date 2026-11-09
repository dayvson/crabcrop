# AGENTS.md

Guidance for AI agents working in this repository. For human-facing
contribution rules see CONTRIBUTING.md; for the release process see
RELEASING.md.

## What this project is

crabcrop is an on-demand image optimization service written in Rust. One
binary serves signed image URLs of the form
`/<signature>/<operations>/<source>`: the URL is the API, operations run in
path order, and renders are cached under a canonical key re-serialized from
the parsed operations. README.md is the product contract; when behavior and
the README disagree, one of them has a bug. The URL grammar is a published
spec (see `spec/`) with cross-language test vectors that the Rust crate and
both SDKs must all pass.

## Layout

- `src/`: the server. `lib.rs` declares the modules as a library
  target, which the fuzz harness and the integration tests import;
  `main.rs` is the binary, reading configuration and booting the
  server. The operation parser and canonical forms in `url.rs`
  (a filter is one row of the table in `url/filters.rs`, which
  generates both its parser and its canonical spelling),
  pipeline in `engine.rs` (encoders in `engine/encode.rs`, pipeline
  state in `engine/state.rs`), HTTP and caching in `server.rs`
  (request coalescing in `server/process.rs`, metadata responses in
  `server/meta.rs`), source fetching policy in `loader.rs`, AI
  features in `ai/` (provider traits in `ai/mod.rs`, one file per
  provider family, all re-exported from `crate::ai`) and
  `studio.rs`. A large module keeps its unit tests in
  `<module>/tests.rs` behind `#[cfg(test)] mod tests;`, as
  `server.rs`, `engine.rs` and the files under `ai/` do.
- `crates/crabcrop-url`: the client-side library: signing, sealing,
  verification, and URL building. It does not parse operations; the
  server's parser is `src/url.rs`.
- `sdks/`: TypeScript and Python SDKs, plus the MCP server (`sdks/mcp`).
- `frameworks/`, `playground/`: integration starters and the live playground.
- `fuzz/`: cargo-fuzz targets for the URL parser, the token
  opener, image decode (AVIF through dav1d included), metadata,
  and operation chains through the engine, built on the server
  library and `crabcrop-url` (the opener's home). The smoke runs
  in `scripts/test_all.sh`; `.github/workflows/fuzz.yml` fuzzes
  every target nightly.
- `tests/functional/`: the HTTP functional suite, tiered; the local tier
  runs without network or model files.

## Build and test

- Build: `cargo build --locked`
- Full gate: `bash scripts/test_all.sh` (binary suite, crabcrop-url, SDKs,
  MCP, frameworks, playground, fuzz). Everything must pass before a change
  lands.
- CI also enforces `cargo fmt --check` and
  `cargo clippy --locked --workspace --all-targets -- -D warnings`,
  which test_all.sh does not run. Run both locally before pushing or
  the first CI run fails on style alone.
- The real-model AI tests are `#[ignore]`d in the normal gate: they need
  model files and an ONNX Runtime library, which are deployment artifacts,
  never repo content. `.github/workflows/ai-models.yml` downloads the
  pinned artifacts and runs them (`cargo test -- --ignored`) on pull
  requests and main pushes that touch `src/ai/`, `src/tokenizer.rs`,
  `tests/fixtures/portrait.jpg` or the workflow itself, on every release
  tag before anything publishes, weekly, and by hand. When you cannot run
  them locally, that pull-request run is where an AI change is verified.
  A change outside those paths can still break them (the real-model tests
  in `src/main.rs` and `src/engine.rs`, an `ort` bump in `Cargo.lock`)
  without starting it: run the workflow by hand on the branch
  (`gh workflow run ai-models.yml --ref <branch>`).
- The AI functional groups (`tests/functional`, tier `local`) run in no CI
  workflow: only a local run with the models verifies them.

## Invariants

- Signed URLs are a compatibility surface. A URL signed by an earlier
  release must keep verifying, and a grammar change is a breaking change:
  it resets cache keys and invalidates existing URLs. Treat grammar work
  accordingly.
- `Cargo.lock` moves with any version or dependency change. Version bumps
  run through `scripts/bump_version.py`, which moves every site listed in
  `scripts/version_sites.py` (the lockfiles' own entries included) and
  then runs `scripts/check_versions.py` over the same list. Never move a
  site by hand: a file that starts carrying the release version joins
  that list, or `scripts/test_version_tools.py` fails the gate on the
  old version a bump leaves in it.
  `fuzz/Cargo.lock` moves too: the fuzz package builds the server
  library, so a dependency change that the root lockfile records needs
  the refresh in `fuzz/README.md`, or the fuzz smoke's `--locked` build
  in CI fails. `scripts/check_locks.py` fails CI when the two lockfiles
  disagree on a package both carry, or when either locks a decoder at
  two versions: one decoder version is what ships, pre-checks and
  fuzzes.
- One binary, configuration-gated: AI and provenance features are enabled
  by environment configuration in the same binary, not by separate builds
  or cargo features. Never commit model files, certificates, or keys.
  The exception is throwaway C2PA test material: the test identity
  (`tests/functional/c2pa_cert.pem` and `c2pa_key.pem`, the same pair
  as `TEST_CHAIN` / `TEST_KEY` in `src/provenance.rs`) and the
  negative-case key and certificate in that module's tests. None of
  it chains to a trust list or is a deployment credential.
- Source fetching stays inside the loader's policy (host allowlist,
  private-address protection, fetch and decode bounds). Do not widen what
  the server will fetch as a side effect of an unrelated change.

## Conventions

- Docs style: simple, clear, example-driven; README first. Behavior,
  README, and `docs/` move together in the same change. Code comments
  explain the why, not the what; no gratuitous comments.
- Dependencies: active, maintained crates. A dependency swap that changes
  output bytes needs before-and-after evidence (sizes, quality, timing),
  not just a benchmark win.
- New behavior ships with tests. Anything that parses untrusted input
  (URLs, images, uploads) also gets hostile-input tests that prove the
  process survives.
- Errors are part of the API: responses carry structured error codes, and
  the error lists in the README and docs stay accurate.

## Releases

See RELEASING.md for the full choreography. The hard rule: never tag or
publish a release unless main CI is fully green on the release head, the
cargo audit job included. The release workflow's preflight job enforces
the `rust.yml` part: a tag whose commit has no passing `rust.yml` push run
on main stops before anything builds. It does not look at the other
workflows that run on main, such as functional.yml; check those yourself.
