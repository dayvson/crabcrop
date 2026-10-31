# Benchmark harness

A reproducible comparison of three self-hosted image servers on the
same host, the same origin, and the same operations:

- crabcrop 0.9.5 (release build from the `v0.9.5` tag)
- Thumbor 7.8.0 (pip, venv)
- imgproxy 3.31.4 (official source build, see the version notes)

This harness began as the October 2026 comparison that measured
crabcrop against Thumbor only (imgproxy could not be measured then:
no Docker on the benchmark host, and no Go or libvips installed).
It lives in the repo now so the comparison can be re-run for future
releases instead of living in chat notes.

## Reproduce

```bash
bash benchmark/setup.sh
python3 benchmark/bench.py
```

`setup.sh` pins every version in one place at the top of the file
and prepares `benchmark/.work/` with the three servers. It installs
system packages with apt (libvips and build tools), installs a Go
toolchain under `~/.local`, builds imgproxy from its release tag,
creates the Thumbor venv, and builds crabcrop in release mode from
the `v0.9.5` tag in a detached worktree.

`bench.py` then starts a local origin server for the fixtures,
starts each service in turn, measures it alone, stops it, and
writes raw results to `benchmark/.work/results.json`. Committed
runs are copied into `benchmark/results/` next to a write-up.

Set `BENCH_SKIP_SERVICES=crabcrop,thumbor,imgproxy` (any subset)
to run only some services while debugging.

## Fixtures

`fixtures/` holds 25 generated photo-like JPEGs, 2400x1600,
quality 90, about 205 KB each (5.1 MB total). They are the same
sources the October runs used, so numbers stay comparable across
runs.

## Method

- One local origin server (Python `http.server` threads) serves
  the fixtures over HTTP on 127.0.0.1. Every service fetches from
  that same origin.
- Services run one at a time, sequentially, on an otherwise idle
  host. Nothing else is benchmarked concurrently.
- Cold means the first request for each of the 25 sources:
  fetch, decode, transform, and encode all happen. Warm means the
  same 25 URLs again immediately after, so each product answers
  from its own result caching if it has any.
- Warm throughput is one extra pass on the resize operation:
  the 25 warm URLs at 8 concurrent workers.
- Every request is a fresh connection, timed client side with a
  monotonic clock, in milliseconds.
- Code paths are warmed up before measuring with a throwaway
  17x17 size that no measured URL uses, so no cold sample is
  pre-warmed. (The October harness had that blemish and disclosed
  it; this harness removes it.)
- Per operation the harness records mean, p50, p95, min, and max
  latency for cold and warm from the 25 samples, plus the average
  output size in bytes.
- An operation counts as unsupported when the service has no
  encoder or filter for it, or when the response is not a 200
  with the expected image Content-Type. Unsupported operations
  are recorded as unsupported in the results, never approximated
  silently. See the mapping table for the per-operation reasons.

## Configuration

The exact configuration for each service is in `configs/`:

- `configs/crabcrop.env`: secret set, unsafe mode on, port 3310.
  Everything else is the server default, including its in-memory
  result cache.
- `configs/thumbor.conf`: the October config. HTTP loader, unsafe
  URLs, everything else at Thumbor defaults (no result storage).
- `configs/imgproxy.env`: bind on 127.0.0.1:8080,
  `IMGPROXY_WORKERS=4` (one per core on the benchmark host; the
  imgproxy docs size workers to cores), no key or salt so
  signature checking is off, and
  `IMGPROXY_ALLOW_LOOPBACK_SOURCE_ADDRESSES=true` so imgproxy
  will fetch from the local origin at all (it refuses loopback
  sources by default; that is a fetching policy, not processing).

All three systems run with signature verification effectively
open (crabcrop and Thumbor unsafe mode, imgproxy without
key/salt). This measures processing, not HMAC verification.

## URL mappings

Origin below means `http://127.0.0.1:8911/srcNN.jpg` (the local
origin serving the fixtures). crabcrop and Thumbor URLs are
path suffixes on their own ports; imgproxy uses its plain
(unsigned) URL form.

| Operation | crabcrop | Thumbor | imgproxy |
|---|---|---|---|
| resize_w800 | `/unsafe/800x0/<origin>` | `/unsafe/800x0/<origin>` | `/insecure/rs:fit:800:0/plain/<origin>@jpg` |
| crop_400x300 | `/unsafe/crop:400x300/<origin>` | `/unsafe/400x300/<origin>` | `/insecure/rs:fill:400:300/plain/<origin>@jpg` |
| webp800_q80 | `/unsafe/800x0/filters:format(webp):quality(80)/<origin>` | same form as crabcrop | `/insecure/rs:fit:800:0/q:80/plain/<origin>@webp` |
| gray800 | `/unsafe/800x0/filters:grayscale/<origin>` | `/unsafe/800x0/filters:grayscale()/<origin>` | unsupported |
| avif800_q80 | `/unsafe/800x0/filters:format(avif):quality(80)/<origin>` | unsupported | `/insecure/rs:fit:800:0/q:80/plain/<origin>@avif` |
| jxl800_q80 | `/unsafe/800x0/filters:format(jxl):quality(80)/<origin>` | unsupported | `/insecure/rs:fit:800:0/q:80/plain/<origin>@jxl` |
| jpeg800_q60 | `/unsafe/800x0/filters:quality(60)/<origin>` | same form as crabcrop | `/insecure/rs:fit:800:0/q:60/plain/<origin>@jpg` |
| jpeg800_q95 | `/unsafe/800x0/filters:quality(95)/<origin>` | same form as crabcrop | `/insecure/rs:fit:800:0/q:95/plain/<origin>@jpg` |
| webp800_q60 | `/unsafe/800x0/filters:format(webp):quality(60)/<origin>` | same form as crabcrop | `/insecure/rs:fit:800:0/q:60/plain/<origin>@webp` |
| webp800_q95 | `/unsafe/800x0/filters:format(webp):quality(95)/<origin>` | same form as crabcrop | `/insecure/rs:fit:800:0/q:95/plain/<origin>@webp` |

Resize and crop use each product's default quality (80 for all
three at these settings) so the October rows stay comparable.

Mapping notes, where a mapping is not exact:

- Crop: crabcrop `crop:400x300` is an exact-box center crop.
  Thumbor `400x300` is a cover resize with center crop (its
  smart detection is off in both cases). imgproxy `rs:fill`
  scales to cover and center-crops. All three produce exactly
  400x300 from a center weighted crop; the pixel selection can
  differ by a rounding pixel.
- imgproxy grayscale: imgproxy OSS has no grayscale, saturation,
  or desaturation processing option (those adjustments are
  imgproxy Pro features). The operation is recorded as
  unsupported for imgproxy. It is not approximated.
- Thumbor AVIF and JPEG XL: Thumbor 7.8.0 core has no AV1 or
  JPEG XL encoder, so both operations are unsupported there.
- imgproxy AVIF depends on the host libheif having an AV1
  encoder plugin. See the results document for what happened on
  the benchmark host.
- Quality numbers are nominal. Each encoder maps the same
  number to different quantisation choices, so equal quality
  numbers do not promise equal fidelity or equal bytes. The
  byte tables are part of the results for exactly that reason.

## Version notes

- crabcrop 0.9.5, tag `v0.9.5`, commit
  5a1674f4908ccc8ca83f64c134e98a60744fcd24. Built with
  `cargo build --release --locked` from the tag.
- Thumbor 7.8.0, pip package `thumbor==7.8.0`, Python venv.
- imgproxy 3.31.4, tag `v3.31.4`, commit
  8b0aa78279f191081054ba709b39f317f66ace78, built from the
  official source with Go 1.26.5 against libvips 8.15.1
  (Ubuntu 24.04 apt). Why not the newer imgproxy 4.0.x line:
  4.0.x needs Go 1.27 and a newer libvips than Ubuntu 24.04
  ships; 3.31.4 is the newest release whose CI tests libvips
  8.14, 8.15, and 8.16, so 8.15.1 is inside its supported
  range. The official Docker image instead builds from the
  imgproxy-base image (v3.15.9 for this release), which carries
  its own newer libvips built from source, so encoder behavior
  could differ slightly from this source build. The results
  document repeats this caveat with the numbers.

## Limits

- Single host topology: services, origin, and client share one
  machine and its loopback. Absolute numbers include loopback
  and client overhead; relative comparisons are the point.
- The host is small (see the results document for its exact
  spec). Rankings can move on bigger hardware, especially for
  the multi-process and multi-worker topologies.
- Warm numbers measure each product's own result caching as
  configured in `configs/`. Thumbor's production caching story
  is normally a CDN in front, which this harness does not
  simulate for any product.
- One run per operation set per service, 25 samples each.
  Treat small deltas as noise.
