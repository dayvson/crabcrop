# Upgrade test: previous-release URL corpus

Issue #311. Signed URLs are a compatibility surface: a URL minted
under one release must keep verifying, and keep rendering the same
bytes, under the next one. This suite freezes that promise into a
corpus and replays it against every candidate build.

`corpus.json` holds the frozen corpus: URLs minted with the repo
Python SDK under test-fixture secrets, rendered by the previous
release (see the `baseline` block in the file for the release, the
tag commit, and how the goldens were produced). Each entry records
the full frozen path plus the expected status, content type, byte
length, and SHA-256 of the baseline render.

## What the corpus covers

Resize (exact and proportional), crop (plain and with a gravity),
rotate, flip, format conversion to WebP, AVIF, JPEG XL, PNG, and
GIF, a quality override, grayscale, a multi-filter chain, blur,
sharpen, brightness, contrast and saturation adjustment, invert,
hue rotate, pixelate, rounded radius, pad, metadata strip and
metadata preserve on the EXIF source, a watermark overlay, an
animated GIF resize, two preset references, the source-root short
form, a sealed URL, `exp:` in the far future, `exp:` in the past
(expects 403), keyed URLs under the current and previous key IDs,
an unkeyed URL signed with the previous secret (the rotation
window), and a purge entry that round-trips a purge and rejects a
corrupted purge signature.

Sources are the small files in `tests/functional/corpus/`. The
corpus instance runs on ports 8991 (origin) and 8992 (server) with
the fixture secrets recorded in `corpus.json`, the local presets
in `presets.txt`, and a source roots file generated at runtime
that points `photos` at the local origin.

The metadata preserve entry is the one entry whose variable is
configuration rather than the URL: it repeats the metadata strip
entry's path and is rendered by an instance booted with
`CRABCROP_METADATA=preserve`. Both runners boot that second
configuration for it.

## Running the check

```bash
cargo build --release --locked
python3 tests/upgrade/run_upgrade.py --bin target/release/crabcrop
```

`run_upgrade.py` is standard library only: the paths in
`corpus.json` are frozen, so replaying them needs no SDK. It boots
the origin and the candidate, compares every entry byte for byte,
prints one PASS or FAIL line per entry (a failure names the URL
and the expected and actual hashes), and exits 1 on any failure.
`--out DIR` saves the actual body of each mismatching entry under
DIR for diffing. The default `--bin` is `target/debug/crabcrop`,
which is the form `scripts/test_all.sh` runs.

Byte differences fail. That is the point: a difference is either
an incompatibility to fix, or a deliberate output change to
re-baseline (below). There is no tolerance band.

## Re-baselining

When a release deliberately changes output bytes (an encoder swap,
a filter fix), re-freeze the corpus against the new release once
it ships:

```bash
cargo build --release --locked   # in a checkout of the new tag
python3 tests/upgrade/freeze.py --bin target/release/crabcrop \
    --release vX.Y.Z
```

`freeze.py` mints the corpus with the Python SDK (the sealed entry
needs the SDK seal extra, `cryptography`), renders every entry
twice from two separate server boots, and keeps only entries whose
two renders are byte-identical: a non-deterministic entry is
dropped and reported rather than frozen as a golden that would
flake forever.

For every entry whose bytes change in a re-baseline, record the
reason in that entry's `note` field in `corpus.json` and in the
CHANGELOG. A re-baseline with no recorded reason is not a
re-baseline, it is a silent compatibility break.

## Release wiring

RELEASING.md carries the step: run this check against the release
candidate build before tagging. `scripts/test_all.sh` also runs it
(against the debug build) after the Rust suites, so an
incompatibility shows up in the ordinary gate, not only at release
time.

## Scope exclusions

These surfaces are deliberately not in the corpus:

- C2PA: manifests embed the signing time, so the bytes are not
  stable by design.
- Text overlay: output depends on host font rasterization.
- Video frames and PDF pages: output depends on the external
  ffmpeg and poppler versions on the host.
- AI features: models are deployment artifacts, never repo
  content, so no model is guaranteed present at check time.
- AVIF as a source format: the decoder varies with build
  features, so AVIF sources are not a stable baseline. AVIF as an
  output format is covered.
