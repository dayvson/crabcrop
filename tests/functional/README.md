# Functional tests

The permanent functional test workflow for crabcrop (issue #232): the
feature-campaign harness, productized. Where the unit suites check
functions, this suite checks the shipped behavior: it boots a locally
built server, renders real images from a frozen in-repo corpus, and
verifies the results the way a client would see them.

## Assertion philosophy

Property assertions, never byte goldens.

- Dimensions exact.
- Format and magic bytes correct.
- Channel statistics within tolerance (means, spreads, hue shifts,
  region statistics, pixel samples with tolerances).
- Output bytes within a band of the blessed baseline (about
  +-15 percent), and channel means within a few points.

Encoder bumps change bytes legitimately; a suite that fails on every
dependency update gets ignored, then deleted. This one is built to
stay green through them and to fail on real behavior changes:
wrong dimensions, wrong format, a filter that stops moving pixels,
a cache that stops hitting, a signature that stops verifying.

Timings are collected into every report as trend data (per-group
median and p95 server time, largest byte deltas). A timing never
fails a run: shared runners are too noisy, and the 10x regressions
this suite exists to catch stand out in the trend table anyway.

## Layout

| path | what |
|---|---|
| `run.py` | the entry point: builds or takes a binary, boots the instances, runs a tier, writes results |
| `harness.py` | server management, signing (via the repo Python SDK), HTTP, the assertion DSL evaluator, recording |
| `cases_core.json` | data-driven cases: core imaging (wave 1 of the campaign) |
| `cases_overlay.json` | data-driven cases: overlays, text, geometry (wave 2) |
| `cases_platform.json` | index of the scripted platform cases (wave 4) |
| `platform_cases.py` | the scripted cells: presets, purge, explain, upload, source kinds, SDK/MCP/framework parity |
| `cases_ai.json` | index of the AI cases (wave 3), local tier only |
| `ai_cases.py` | the scripted AI cells |
| `corpus/` + `corpus.json` | the frozen corpus and its manifest (origin, format, dimensions, bytes, frames per file) |
| `baselines.json` | blessed per-case output bytes and channel means |
| `origin.py` | the hit-counting origin server (static corpus plus redirect/slow/error traps) |
| `presets.txt`, `badpresets/` | preset fixtures, valid and boot-rejecting |
| `emit_py.py`, `emit_ts.mjs`, `emit_fw.mjs`, `fw_register.mjs`, `parity-rs/` | the SDK parity emitters |
| `mcp_client.py` | a minimal stdio JSON-RPC client for the repo MCP server |
| `make_corpus.py` | the one-time freeze that produced `corpus/` from the campaign corpora (provenance record; not run by CI) |

Case tables mark every case with a tier: `smoke`, `full`, or
`local`. Data-driven cases assert through the DSL in `harness.py`
(status, dims, pixel and region statistics, meta fields, cache
behavior, EXIF/ICC handling, determinism, the baseline band, and
more). Scripted cases are Python functions registered by dashed
name in `platform_cases.py` / `ai_cases.py`; the JSON tables are
their index.

## Running it

Prerequisites: Python 3 with Pillow (`pip install pillow`), a Rust
toolchain, and (full tier only) Node 22+ for the parity and MCP
cells.

```bash
# PR smoke tier (about 4 minutes of budget; far less in practice)
python3 tests/functional/run.py --tier smoke

# the full CI matrix
python3 tests/functional/run.py --tier full

# against an already-built binary, results somewhere specific
python3 tests/functional/run.py --tier full --release --no-build \
    --bin target/release/crabcrop --out /tmp/functional-out

# subsets while developing
python3 tests/functional/run.py --tier full --ids preset-basic,fx-grayscale
python3 tests/functional/run.py --tier full --groups purge,explain
```

`run.py` boots everything it needs on scratch ports (8971 to 8983)
and stops it at the end: an origin server over `corpus/`, and
crabcrop instances per configuration profile (default with a font,
no-font, metadata-preserve, presets, bare, uploads, only-presets,
localhost-only allowlist). Nothing is fetched from the network; the
corpus is the only fixture source.

Results land in the out directory after every case, so a crashed
run keeps its data: `results.json`, `results.csv`, `summary.md`
(the same Markdown the CI job summary publishes), and
`observations.jsonl` (one raw fact record per request).

Exit code is 1 when any case verdict is FAIL or ERROR. NOTE is a
recorded observation (a documented quirk, a known-issue sighting,
a timing sample), never a failure.

## Tiers and CI

The workflow lives at `.github/workflows/functional.yml`; its
mirror is `docs/functional-workflow.yml` (the repo's established
pattern, landed through the GitHub web UI like the other workflow
files).

- **Smoke on pull requests**, with `--enforce-budget`: the run
  fails if the tier's wall clock passes 240 s. An overrun is a bug
  in the suite: cut cells or shrink corpus images rather than
  letting it grow.
- **Full on schedule (nightly) and on release tags.**
- **Full instead of smoke on a pull request** whose diff touches
  `Cargo.lock`, `Cargo.toml`, a crate manifest, the encode
  pipeline modules under `src/` (engine, filters, quality, color,
  crop, heic, svg, text, video, studio, ai), or this harness.

## Adding a case

Data-driven first: if the behavior is "request X, then the output
has property Y", add a row to the matching `cases_*.json` table.
Pick the tier honestly (smoke is for the top operations and must
stay fast), set `ops` and `src` (a corpus file), and write
assertions in the DSL. Cross-case references (`vs`, `cell`) point
at earlier rows in the same table; keep referenced base renders in
the same tier as their dependents.

If the behavior is a sequence (prime, purge, re-render; upload then
render; boot with a bad config), write a scripted cell in
`platform_cases.py` (or `ai_cases.py` for AI behavior), register it
with the `@cell` decorator, and add its index row to the matching
JSON table.

New corpus files: add the file to `corpus/`, record it in
`corpus.json` (origin, format, dimensions, bytes, frames), and keep
the total small (the whole corpus is about 5 MB; downscale photos
to the smallest size the cases still meaningfully exercise).

## Re-blessing baselines

Baselines are reviewed like a lockfile change: a real regression
must not be rubber-stamped away, so re-bless deliberately, read the
diff, and say why in the commit.

```bash
python3 tests/functional/run.py --tier full --bless
git diff tests/functional/baselines.json
```

`--bless` runs the full tier with band checks off and records each
passing render's bytes and channel means into `baselines.json`.
On normal runs, a render whose bytes leave the +-15 percent band,
or whose channel means move more than 7 points, fails with the
delta in its evidence line. If a case's output legitimately changed
(an encoder bump, an intentional filter change), the diff shows
exactly which cases moved and by how much.

## The local AI tier

The AI cases never run in CI: the models are deployment artifacts,
never repo content. To run them locally, stage the campaign model
files in one directory and point the runner at them:

```bash
CRABCROP_FUNC_MODELS_DIR=/path/to/models \
CRABCROP_ORT_DYLIB=/path/to/libonnxruntime.so \
python3 tests/functional/run.py --tier local
```

The directory must hold, under these names: `u2netp.onnx`,
`yunet.onnx`, `clip_vision.onnx`, `clip_text.onnx`,
`clip_tokenizer.json`, `clip_labels.txt`, `moderation.onnx`,
`moderation_labels.txt`, `moderation_signal_labels.txt`,
`aesthetic_b32.pth`, `aesthetic_l14_v1.pth`, `yolo.onnx`,
`detectors.json`, `upscale_swin2sr.onnx`. The runner boots three
instances: AI (everything configured), AIN (nothing configured),
AIC (the B/32 encoder with the mismatched L/14 aesthetic head and
label-only moderation signals). Without the environment variables,
every AI case is recorded as a NOTE (skipped) and the run succeeds,
so the tier stays visible in local summaries.

Two cells are known-defect probes from the campaign and assert the
fixes hold: `st-nosubject-gradient` (#243) and `upscale-oom-probe`
(#244). The #227 lossy-WebP alpha encode cells record sightings as
NOTEs with their timings, the trend data for that issue.
