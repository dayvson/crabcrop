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

- `src/`: the server binary. Pipeline in `engine.rs`, HTTP, caching, and
  request coalescing in `server.rs`, source fetching policy in `loader.rs`,
  AI features in `ai.rs` and `studio.rs`.
- `crates/crabcrop-url`: the URL grammar and signing library.
- `sdks/`: TypeScript and Python SDKs, plus the MCP server (`sdks/mcp`).
- `frameworks/`, `playground/`: integration starters and the live playground.
- `fuzz/`: cargo-fuzz targets for the parsers.
- `tests/functional/`: the HTTP functional suite, tiered; the local tier
  runs without network or model files.

## Build and test

- Build: `cargo build --locked`
- Full gate: `bash scripts/test_all.sh` (binary suite, crabcrop-url, SDKs,
  MCP, frameworks, playground, fuzz). Everything must pass before a change
  lands.
- CI also enforces `cargo fmt --check` and
  `cargo clippy --locked --all-targets -- -D warnings`, which test_all.sh
  does not run. Run both locally before pushing or the first CI run fails
  on style alone.
- AI functional groups need local model files and an ONNX Runtime library.
  Models are deployment artifacts, never repo content; a group that cannot
  run locally for lack of models is verified in CI instead.

## Invariants

- Signed URLs are a compatibility surface. A URL signed by an earlier
  release must keep verifying, and a grammar change is a breaking change:
  it resets cache keys and invalidates existing URLs. Treat grammar work
  accordingly.
- `Cargo.lock` moves with any version or dependency change. Version bumps
  run through `scripts/bump_version.py` plus a lockfile refresh; a bump
  without the lockfile update fails every `--locked` build.
- One binary, configuration-gated: AI and provenance features are enabled
  by environment configuration in the same binary, not by separate builds
  or cargo features. Never commit model files, certificates, or keys.
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
cargo audit job included.
