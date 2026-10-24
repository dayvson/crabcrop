# Fuzzing the URL parser

The URL parser is the server's attack surface: every request
starts as untrusted text. This package fuzzes it (issue #146).

## The target

`parse_request` (fuzz target `parse_request`) checks the
round-trip invariants in `src/lib.rs` for every input:

- nothing panics, on any entry point over untrusted text;
- a path that parses canonicalizes to a fixed point
  (canonical form, re-parsed and re-canonicalized, is
  byte-identical, signature and key ID included);
- the canonical key form is a fixed point.

The parser lives in the server binary crate, so the harness
includes `src/url.rs` by path (it has no crate-internal
dependencies). This package is detached from the main
workspace so fuzzing dependencies never enter a normal
build.

## The smoke (runs everywhere)

`scripts/test_all.sh` runs the deterministic smoke:
every seed in `corpus/parse_request/`, every truncation of
a grammar-rich seed, and single-byte mutations of it,
through the same harness. On its own:

```sh
cargo test --manifest-path fuzz/Cargo.toml --lib --test smoke
```

## Real fuzzing (nightly)

Open-ended runs need a nightly toolchain and cargo-fuzz:

```sh
rustup toolchain install nightly
cargo +nightly install cargo-fuzz
cargo +nightly fuzz run parse_request -- -max_total_time=300
```

Crashes land in `fuzz/artifacts/`. A crash becomes its own
fix issue with the minimized input, and the input joins
this corpus and the parser's regression tests before the
fix ships.

## The corpus

Seeds come from the grammar corpora: the canonicalization
test strings, the filter accept/reject batches, preset and
expiry shapes, keyed signatures, and malformed spellings.
Add a seed file per interesting shape; keep them small.
