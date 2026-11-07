# Fuzzing the untrusted surfaces

Every request starts as untrusted text, and the body it
names is untrusted bytes, so both are the server's attack
surface. This package fuzzes the text entry points, the
URL parser (issue #146) and the sealed-token opener
(issue #255), and the byte surfaces behind them (issue
#387): image decode, source metadata extraction, and
filter chains.

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

`decode` checks the decode path (`check_decode`) for
every input: nothing panics and no allocation runs away,
whether the bytes are an image the decoders accept, an
SVG the sanitizer and rasterizer process, or garbage.
Non-SVG bytes go through the server's decode dispatch
(`engine::validate_source_image`: HEIC, animated GIF and
WebP frames under the animation caps, the JPEG XL header
probe) and through `engine::decode_image`, the bounded
still decode with the server's dimension and allocation
limits and its JPEG and WebP fast paths.

`source_meta` checks metadata handling (`check_meta`):
extracting a source's ICC profile, EXIF stream, XMP
packet, and PNG text chunks, patching the TIFF
orientation, re-injecting metadata into JPEG and PNG
containers, and converting through an extracted profile
are all total over untrusted bytes.

`op_chain` checks filter chains (`check_chain`): a
request that parses has its `filters:` segments applied
to a small synthetic image; application is total, and
the canvas keeps its dimensions (or its rotation swap)
under every chain.

The harness depends on the server's library target
(`crabcrop`, issue #417), so each target runs the
server's own code: the parser, the decode dispatch, the
metadata code, the filters, and the SVG pipeline, with
no copies or stand-ins. The opener lives in
`crabcrop-url`, a direct dependency. This package is
detached from the main workspace so fuzzing dependencies
never enter a normal build.

`Cargo.lock` here pins the same versions as the root
lockfile. Because the package builds the server library,
a dependency added or bumped in the root manifest (or a
release version bump) can leave it stale, and the
smoke's `--locked` build (in `scripts/test_all.sh` and
CI) then fails until it is refreshed. Refresh it from
the root lockfile:

```sh
cp Cargo.lock fuzz/Cargo.lock
cargo update -w --manifest-path fuzz/Cargo.toml
```

## The smoke (runs everywhere)

`scripts/test_all.sh` runs the deterministic smoke: every
seed in the corpora, every truncation of a representative
seed per target, and single-byte mutations of each, through
the same harness. On its own:

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
cargo +nightly fuzz run decode -- -max_total_time=300
cargo +nightly fuzz run source_meta -- -max_total_time=300
cargo +nightly fuzz run op_chain -- -max_total_time=300
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
non-sealed segments. `corpus/decode/` and
`corpus/source_meta/` seed real bodies from the functional
corpus and the fixtures (small images per format, an SVG,
EXIF and ICC carriers, and truncated prefixes of the
animated formats). `corpus/op_chain/` seeds filter-chain
remainders, plain and face-shaped. Add a seed file per
interesting shape; keep them small.
