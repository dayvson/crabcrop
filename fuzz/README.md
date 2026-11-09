# Fuzzing the untrusted surfaces

Every request starts as untrusted text, and the body it
names is untrusted bytes, so both are the server's attack
surface. This package fuzzes the text entry points, the
URL parser (issue #146) and the sealed-token opener
(issue #255), and the byte surfaces behind them (issue
#387): image decode, source metadata extraction, and
operation chains through the render pipeline (issue
#476).

## The targets

`parse_request` checks the parser's round-trip invariants in
`src/lib.rs` for every input:

- nothing panics, on any entry point over untrusted text;
- a path that parses canonicalizes to a fixed point
  (canonical form, re-parsed and re-canonicalized, is
  byte-identical, signature and key ID included);
- the canonical key form is a fixed point;
- preset expansion against a small fixed preset map (a
  stand-in for operator configuration, like the source
  root map) is total, with and without the presets-only
  policy, and an expanded path that parses meets the same
  fixed point.

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

`op_chain` checks operation chains (`check_chain`) on a
small synthetic source (24x18), in two legs, for every
request that parses:

- The filter leg applies the filters of the `filters:`
  segments one at a time, in URL order. Application is
  total, and each filter lands on the canvas its kind
  predicts: a rotation on the canvas
  `filters::rotated_canvas` computes (quarter turns swap
  the axes, other angles grow to the rotated bounding
  box), a trim never grows either axis, every other
  filter keeps the dimensions, and no dimension is ever
  zero. A rotation whose canvas would exceed the
  harness's pixel budget, or a blur past its blur
  budget (below), ends the leg.
- The engine leg runs the whole operation list through
  `engine::Engine::process` on the source encoded as a
  PNG: resize, crop (`src/crop.rs`: center, directional,
  smart, focal), fit-in, pad, crop-rect, the debug view,
  the filters, and the encode. It uses the default
  context, so the AI filters, watermarks, text, frame and
  page selection refuse with an `EngineError`. The engine
  either refuses or renders an image whose axes are
  within 1 and `url::MAX_DIMENSION` (16383). Before the
  engine runs, `canvas_bound` walks the chain with
  conservative bounds on every canvas it builds (DPR,
  proportional and `orig` axes, the adaptive fit-in
  swap, rotation, trim); a request that could pass the
  harness budget (`CHAIN_MAX_PIXELS`, 2^18 pixels) at any
  step skips this leg, so the harness, not the engine's
  100-megapixel output budget, bounds a run's
  allocations. The render must also stay inside that
  bound, which checks the guard itself.

Blurs get a budget of their own, `CHAIN_MAX_BLUR_WORK`
(2^24 pixels times kernel taps, summed over the chain).
A blur's cost grows with its sigma as well as its canvas,
and the fuzzing build (address sanitizer plus comparison
tracing) runs its inner loop over a hundred times slower
than a debug build: `blur(100)` on a 500x500 canvas took
three minutes there and about a second in the smoke,
which the fuzzer would report as a hang. A blur past the budget ends the filter
leg, and a request whose blurs (blur filters, blurred
regions, `pad:...:blur` bars) could pass it skips the
engine leg. Small sigmas and canvases still reach every
blur path.

The harness depends on the server's library target
(`crabcrop`, issue #417), so each target runs the
server's own code: the parser, the decode dispatch, the
metadata code, the filters, the render pipeline, and the
SVG pipeline, with no copies or stand-ins.

The package's default `avif-decode` feature turns on the
server's feature of the same name, so `decode` (and the
AVIF seed in its corpus) reaches the dav1d decoder that
the release binaries, the Docker image, and the Homebrew
install ship. dav1d is a C library: pkg-config finds a
system libdav1d (`libdav1d-dev` on Debian and Ubuntu), or
`SYSTEM_DEPS_DAV1D_BUILD_INTERNAL=always` builds it from
vendored sources, as the release does (that needs meson,
ninja, and nasm, and network access to
code.videolan.org). `--no-default-features` builds
without it. The opener lives in
`crabcrop-url`, a direct dependency. This package is
detached from the main workspace so fuzzing dependencies
never enter a normal build.

`Cargo.lock` here pins the same versions as the root
lockfile. Because the package builds the server library,
a dependency added or bumped in the root manifest can
leave it stale (a release version bump does not:
`scripts/bump_version.py` moves this lockfile's
`crabcrop` and `crabcrop-url` entries), and the
smoke's `--locked` build (in `scripts/test_all.sh` and
CI) then fails until it is refreshed. A root lock
refresh can also leave it behind while that build still
passes, so the fuzzer runs decoder versions the server
does not ship: `scripts/check_locks.py` (CI's build job
and `scripts/test_all.sh`) fails whenever the two
lockfiles disagree on a package both carry, or either
locks a decoder at two versions (issue #552). A
Dependabot cargo pull request moves the root lockfile
alone, so one that moves a package both carry fails
that check until this refresh is pushed to its branch;
Dependabot's own entry for this directory updates only
`libfuzzer-sys`, which the root lockfile does not
carry. Refresh it from the root lockfile:

```sh
cp Cargo.lock fuzz/Cargo.lock
cargo update -w --manifest-path fuzz/Cargo.toml
```

## The smoke (runs everywhere)

`scripts/test_all.sh` runs the deterministic smoke: every
seed in the corpora, every truncation of a representative
seed per target, and single-byte mutations of each, through
the same harness. `op_chain` also checks that every seed
parses and that the seeds render through the engine, and
runs a short fixed-seed random mutation pass
(substitutions, insertions, and deletions over the grammar
alphabet) as a stand-in for a real fuzz run. Fixed
requests prove the engine leg runs: crops, fit-ins, pads,
and resizes render at sizes only the engine's geometry
produces, oversized requests skip it, and unconfigured
operations refuse. `parse_request` checks that its seeds
cover every operation family the grammar has (each
operation, crop alignment kind, and filter, plus `exp:`,
`maxage:`, `preset:`, and `root:`); the family match is
exhaustive, so a new operation or filter does not compile
there until it has a family and a seed. On its own,
without dav1d (as `scripts/test_all.sh` runs it):

```sh
cargo test --manifest-path fuzz/Cargo.toml --no-default-features --lib --test smoke
```

With the default features it also checks that the AVIF
seed decodes through dav1d.

## Real fuzzing (nightly)

`.github/workflows/fuzz.yml` (mirrored at
`docs/fuzz-workflow.yml`) runs every night and by hand:
each of the five targets for 15 minutes with the default
features (dav1d built from vendored sources), the corpus
each run grows cached for the next one, and any crashing
input uploaded as the `fuzz-artifacts-<target>` artifact
of the failed job. The same workflow runs the smoke with
the AVIF decoder linked, and the workspace tests with
`PROPTEST_CASES=10000` (issue #478).

Locally, open-ended runs need a nightly toolchain,
cargo-fuzz, and dav1d (see above):

```sh
rustup toolchain install nightly --profile minimal
cargo +nightly install cargo-fuzz --locked
cargo +nightly fuzz run op_chain -- -max_total_time=300 -rss_limit_mb=2048 -malloc_limit_mb=512 -timeout=600
```

The same for `parse_request`, `unseal`, `decode`, and
`source_meta`. The limits are the workflow's: 2 GiB for
the process, and 512 MiB for any one allocation, twice the
engine's 256 MiB decode budget (one decoded buffer may
reach the budget, and a same-size copy or one capacity
doubling of it stays within design; anything larger
escaped the budget). The 600-second hang timeout is
generous on purpose: the sanitizer build is slow enough
that a tight one reports bounded work as a hang.

Crashes land in `fuzz/artifacts/`. A crash becomes its own
fix issue with the minimized input
(`cargo +nightly fuzz tmin <target> <input>`), and the
input joins this corpus and the regression tests before
the fix ships.

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
animated formats); `corpus/decode/` also seeds shapes
the server refuses: an OpenEXR file, whose
decoder is not compiled in, and an AV1 still under a
generic HEIF brand, which the HEIC path refuses before
decoding (issue #466); and AVIF containers the item-table
screen refuses before decoding (issue #522): one whose
`iloc` extents sum past the data the file holds, one with
more extents than the cap, and one whose primary `ispe`
declares a size past the decode limits. `corpus/op_chain/` seeds filter-chain
remainders: plain, face-shaped, quarter and arbitrary-angle
rotations (`rotate(91)`, `rotate(45)`, negative and
fractional angles), and trims that shrink the canvas. Add a
seed file per interesting shape; keep them small, and make
sure it parses (the smoke rejects a seed that does not).

Beyond those, `corpus/parse_request/` holds a
well-formed seed for every operation family (the smoke
checks the coverage), `corpus/decode/` an AVIF the engine
encoded (it reaches dav1d with the default features), and
`corpus/op_chain/` the engine's geometry: center, smart,
focal and focal-point crops, fit-in, pad, crop-rect, the
debug view, DPR, and encodes to other formats, all small
enough to render inside the harness budget.
