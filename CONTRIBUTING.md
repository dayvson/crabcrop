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

The full cross-language suite (Rust, the TypeScript and Python SDKs, the MCP
server, the Nuxt module, and the playground, plus a fuzz smoke test) runs from
one script, which starts and stops the local servers the integration tests
need:

```bash
scripts/test_all.sh              # everything
scripts/test_all.sh --skip-rust  # just the non-Rust suites
```

The JavaScript suites need Node 22 or newer.

## Code style

CI enforces formatting and lints. Run both before pushing:

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
```

## The URL grammar is a shared contract

The URL grammar and signing scheme are specified in
[`spec/url-signing.md`](spec/url-signing.md), with canonical test vectors in
[`spec/test-vectors.json`](spec/test-vectors.json) that every implementation
runs as its test suite. If you change the grammar or the signing scheme, update
the spec and the vectors, and keep the SDKs (Rust, TypeScript, Python) and the
playground in sync so they all still pass the shared vectors.

## Changelog

Give user-facing changes an entry under `## [Unreleased]` in
[CHANGELOG.md](CHANGELOG.md). The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Security

Please do not report security vulnerabilities as public issues. See
[SECURITY.md](SECURITY.md) for private reporting.

## Releases

Releases are tag-driven and handled by the maintainer; see
[RELEASING.md](RELEASING.md).
