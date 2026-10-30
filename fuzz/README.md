# Fuzzing the untrusted-text surfaces

Every request starts as untrusted text, so the text that
reaches the server is its attack surface. This package fuzzes
the two entry points: the URL parser (issue #146) and the
sealed-token opener (issue #255).

## The targets

`parse_request` checks the parser's round-trip invariants in
`src/lib.rs` for every input:

- nothing panics, on any entry point over untrusted text;
- a path that parses canonicalizes to a fixed point
  (canonical form, re-parsed and re-canonicalized, is
  byte-identical, signature and key ID included);
- the canonical key form is a fixed point.

`unseal` checks the sealed-token opener (`check_seal`) for
every input:

- opening a token segment under a secret is total, however
  malformed the bytes, keyed or bare;
- a remainder within the cap seals and reopens to itself
  under the minting secret, keyed and bare (round trip);
- a token never opens to its remainder under a different
  secret, so a wrong key or a tampered body is a refusal,
  never another plaintext.

The parser lives in the server binary crate, so the harness
includes `src/url.rs` by path (it has no crate-internal
dependencies). The opener lives in `crabcrop-url`, a direct
dependency. This package is detached from the main workspace
so fuzzing dependencies never enter a normal build.

## The smoke (runs everywhere)

`scripts/test_all.sh` runs the deterministic smoke: every
seed in `corpus/parse_request/` and `corpus/unseal/`, every
truncation of a grammar-rich seed and a well-formed token,
and single-byte mutations of each, through the same harness.
On its own:

```sh
cargo test --manifest-path fuzz/Cargo.toml --lib --test smoke
```

## Real fuzzing (nightly)

Open-ended runs need a nightly toolchain and cargo-fuzz:

```sh
rustup toolchain install nightly
cargo +nightly install cargo-fuzz
cargo +nightly fuzz run parse_request -- -max_total_time=300
cargo +nightly fuzz run unseal -- -max_total_time=300
```

Crashes land in `fuzz/artifacts/`. A crash becomes its own
fix issue with the minimized input, and the input joins
this corpus and the regression tests before the fix ships.

## The corpus

`corpus/parse_request/` seeds come from the grammar corpora:
the canonicalization test strings, the filter accept/reject
batches, preset and expiry shapes, keyed signatures, and
malformed spellings. `corpus/unseal/` seeds the token shapes:
prefix-only, truncated, keyed, wrong-version, bad base64, and
non-sealed segments. Add a seed file per interesting shape;
keep them small.
