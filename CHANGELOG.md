# Changelog

All notable changes to crabcrop are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and crabcrop uses [Semantic Versioning](https://semver.org/) with one version number across every artifact (server binaries, container images, the `crabcrop-url` crate, the TypeScript and Python SDKs, the MCP server, and the Helm chart). See RELEASING.md for how a release is made.

## [Unreleased]

### Fixed

- One aggregate output budget for animated and video outputs: output pixels are charged per frame against the shared 100M px budget instead of per frame only, final dimensions are rechecked after the operation chain (with the rotation canvas checked before it allocates), requests are capped at 100 executed operations, and animated WebP encoding no longer clones every frame buffer (issue #182).
- `blur(...)` sigma is capped at 512 and must be a normal float: larger or subnormal values are rejected with `invalid_filters` instead of sizing multi-gigabyte Gaussian kernel allocations or panicking inside the image crate, and the same bound is enforced where blur is applied (issue #183).
- HEIC/HEIF sources are held to the shared decode budget before decoding starts: the announced size of every item the decode can reach (primary tree, alpha and depth auxiliaries, gain map, `altr` alternatives) is checked against the 16384 px per axis and 256 MiB budgets, every HEVC SPS (configuration record and in band) is checked against its item's declared size with a small block-alignment tolerance, and deflated embedded metadata is no longer inflated (the decoder is built without its deflate feature) while decoded metadata is held to the same 4 MiB ICC and 16 MiB metadata caps as other formats. A crafted HEIC can no longer hide a multi-gigabyte coded picture behind a small declared size (issue #184).

## [0.1.0] - 2026-10-24

The first public release.

### Added

- crabcrop is an on-demand image optimization service written in Rust: the operations are encoded in the URL, the source is fetched and transformed per request, and the result is returned. There are no pre-generated thumbnails and no build step; the URL is the API.

**URL grammar and signing**

- Signed image URLs: every request is `/<signature>/<operations>/<source-url>`, where the signature is the hex HMAC-SHA256 of everything after it, computed over the operations and source exactly as they appear. A bad signature answers `403`; the literal signature `unsafe` skips verification only when the server runs with `CRABCROP_ALLOW_UNSAFE` on.

- Key IDs and secret rotation: a signature segment may carry a key ID (`<key-id>:<tag>`). During a rotation window the server verifies the current secret and the previous one (`CRABCROP_SECRET_PREVIOUS`) side by side, checking a keyed URL against its named key only, so URLs already in the wild keep working until the window closes. An unknown key ID is indistinguishable from a bad signature.

- Expiring URLs: an `exp:<unix-seconds>` segment anywhere among the operations makes a signed URL expire; the HMAC covers it, past the expiry the request answers `403`, and expired URLs never serve from the cache.

- Presets: a preset names an operation string in a server-side file (`CRABCROP_PRESETS_FILE`), and requests reference it as `preset:<name>` anywhere an operation fits. The signature covers the reference, the server expands it deterministically after verification, and cache entries key on the expanded operations. Definitions are validated at startup and cannot contain a source segment, an `exp:`, or another preset; an unknown name answers `400`.

- Presets-only mode: with `CRABCROP_ONLY_PRESETS=1`, free-form operations are rejected with `400`; requests may use preset references, `exp:` metadata, and bare sources only, so public deployments can let browsers and CMSs build URLs from preset names alone.

- Server-wide default operations: `CRABCROP_DEFAULT_OPS` holds an operations string the server prepends to every request after verification, for a house format, quality, or watermark. For the singleton directives the URL wins over the default, and cache entries key on the merged operations, so a bare request and the same request spelled out in full share one entry.

- Domain-separated purge signatures: `POST /purge/<signature>/<path>` and `POST /purge-group/<signature>/<group>` sign under a purge subkey derived from the secret, never the secret itself, so an image signature can never be replayed as a purge signature. The purge target is canonicalized the same way entries are keyed, so any equivalent spelling, a preset reference included, evicts the shared entry.

- Upload endpoint: an optional `PUT /upload/<signature>/<name>` stores originals in an operator-configured destination (a local directory, `CRABCROP_UPLOAD_DIR`, or an S3 bucket, `CRABCROP_UPLOAD_BUCKET`), off by default and absent (404) when not configured. Uploads are signed in their own domain (an upload subkey derived from the secret, like purge signatures) via new `upload_signature` / `uploadSignature` helpers in the Rust crate and the TypeScript and Python SDKs, plus a `crabcrop upload` CLI verb. Bodies are capped by `CRABCROP_UPLOAD_MAX_MB` (default 10, tied to the source fetch cap), must decode as an image the pipeline can serve before they are stored, and never overwrite an existing object unless `CRABCROP_UPLOAD_ALLOW_OVERWRITE` is on (409 otherwise). The response is the exact `file://` or `s3://` source reference to embed in transform URLs. There is no listing and no delete.

**Sources and loading**

- Source kinds: `http(s)://` URLs; `s3://bucket/key`, fetched from AWS S3 or an S3-compatible store (`CRABCROP_S3_ENDPOINT`) and signed with SigV4 when AWS credentials are in the environment (unsigned fetches of public buckets work with none); `gs://bucket/key`, against Google Cloud Storage's S3-compatible XML API, signed with an HMAC key pair when set; and `file:///path`, a local file confined to `CRABCROP_FILE_ROOT`, canonicalized per request so `..` segments and symlink escapes are refused.

- Fetch limits: 10 seconds total per fetch including redirects, a 10 MiB maximum body, and up to 5 redirects for HTTP sources with the source policy re-checked on every hop. If a response carries a `Content-Type` header, it must start with `image/`.

- Source policy: `CRABCROP_ALLOWED_SOURCES` lists host suffixes the loader may fetch from. `CRABCROP_BLOCK_PRIVATE_SOURCES` refuses hosts that are, or resolve to, loopback, private, link-local, or unspecified addresses, and pins the connection to the validated address so the address checked is the address connected. Source URLs with userinfo are always refused. Bucket sources are gated by `CRABCROP_ALLOWED_BUCKETS` instead, and the allowlist is required whenever bucket signing credentials are configured.

- Decode and output bounds: decoding is capped at 16384 px per axis and 256 MiB of decoder allocations, and animated sources at 200 frames and 100 million decoded pixels, so a small file cannot expand into gigabytes. Requested outputs are capped at 16383 px per axis and 100 megapixels.

**Geometry**

- Resize: `<width>x<height>` produces exactly that size; a `0` axis scales proportionally, `orig` takes the source's size on that axis, and a negative dimension flips the image on that axis while resizing.

- Fit inside a box: `fit-in:<width>x<height>` scales the image proportionally so it fits inside the box, with a `0` axis unbounded.

- Orientation-adaptive fit-in: `fit-in:<w>x<h>:adaptive` fits inside the transposed box when the source orientation and the box orientation disagree (portrait source, landscape box, or the reverse), and behaves exactly like plain fit-in when they agree; square sources, square boxes, and boxes without two pixel axes never swap. `pad:` has no adaptive form, since its canvas is explicit. The Thumbor converter now maps `adaptive-fit-in` to it exactly instead of downgrading with a warning.

- Pad to an exact size: `pad:<width>x<height>[:<fill>]` fits the image inside the box, centers it, and fills the rest of the canvas so the output is exactly the requested size. The fill is a 6-digit hex color or `blur`, ambient bars made from a cover-scaled, heavily blurred copy of the image itself; with no fill named, the request's `fill(...)` filter color is used, or black.

- Crop to an aspect ratio: `crop:<width>x<height>[:<align>]` crops to the ratio and resizes to exactly that size. Alignments are `center` (the default), `top`, `bottom`, `left`, `right`, the four corners, `smart` (a window centered on the focal point), and `face` (a window around the largest detected face).

- Corner crop alignments: `crop:` accepts `top-left`, `top-right`, `bottom-left`, and `bottom-right` alongside the single-axis alignments (each corner decomposes per axis: its horizontal component positions a width crop, its vertical component a height crop). The Thumbor converter maps corner alignment pairs to them exactly instead of approximating to the vertical side.

- Caller-supplied focal rectangle for crops: a `focal:<left>x<top>:<right>x<bottom>` segment names the source rectangle the next `crop:` must keep (center and smart crops position on it; directional alignments ignore it). The Thumbor converter now maps the `focal(...)` filter to it.

- Crop a rectangle: `crop-rect:<left>x<top>:<right>x<bottom>` cuts the exact pixel rectangle between the two corners; a rectangle outside the source image fails with `400`.

- Smart-crop debug view: a `debug:focal` segment answers with a diagnostic image instead of the transform result: the canvas the next crop operates on, with the chosen focal point (magenta crosshair), the focal rectangle when one is in play, detected face boxes (amber), and the crop window outline (blue) drawn on it; with no crop it shows the focal analysis of the final image. Debug responses are never cached and are served `Cache-Control: no-store`; animated sources and video output formats are refused with 400. Unknown debug views report their own error code, `invalid_debug`.

**Filters**

- The `filters:` catalog holds 42 names. The 25 pixel filters are: `blur`, `blur_region`, `border`, `brighten`, `contrast`, `convolution`, `curve`, `duotone`, `equalize`, `flip`, `grayscale`, `huerotate`, `invert`, `mask` (circle or ellipse), `monochrome`, `noise`, `pixelate`, `pixelate_region`, `radius`, `rgb`, `rotate`, `saturation`, `sepia`, `sharpen`, and `trim`. Filters chain inside one `filters:` segment in the order written, the segment may repeat in a path, and a repeated directive resolves to the last one.

- Tone curves: a `curve(<channel>,<x>,<y>,...)` filter remaps the `rgb`, `r`, `g`, or `b` channel through 2 to 16 control points (each 0–255, x strictly increasing), linearly interpolated with the ends clamped, for per-channel tone control the additive filters only approximate. Malformed curves report their own error code, `invalid_curve`.

- Convolution: a `convolution(<v1>,...,<vN>,<normalize>)` filter applies a caller-supplied kernel: 9 values (3x3) or 25 (5x5) in row-major order, each a plain decimal from −1024 to 1024, once over the image with edge replication. The trailing flag (0 or 1) scales the kernel by the sum of its entries first; a zero sum applies the kernel as written. Malformed kernels report their own error code, `invalid_convolution`.

- Film grain: a `noise(<amount>)` filter adds monochrome grain: one noise value per pixel applied equally to all three channels (alpha untouched), the value uniform within plus or minus the amount, an integer 0 to 100 percent of full scale. The field is deterministic, seeded from the image content and the amount, so the same URL always produces the same bytes; `noise(0)` is a byte-identical no-op, and animated sources grain per frame from each frame's own content. Malformed amounts report their own error code, `invalid_noise`.

- Encoding directives (6): `format` chooses the output format and overrides negotiation; `quality` sets encoding quality 1 to 100 (default 80) or `auto`; `max_bytes` re-encodes at decreasing quality until the output fits a byte budget; `fill` names the background color used to flatten transparency for JPEG; `no_upscale` caps every resize in the request at the source dimensions; `dpr` scales size and crop targets by the device pixel ratio (1, 1.5, 2, or 3). When crabcrop picks the quality, through `quality(auto)`, a `max_bytes` reduction, or a `Save-Data` step, the response carries an `X-Crabcrop-Quality` header with the value used.

- Verified auto quality: `quality(auto)` picks a quality per image from content analysis, and for JPEG and JPEG XL verifies the pick: the encode is decoded and scored against the processed source with SSIMULACRA2, adjusting quality until the score lands at the target (75 by default, `CRABCROP_SSIM_TARGET`). The achieved score is reported in meta responses as `ssimulacra2`.

- Watermark: `watermark(<url>,<x>,<y>,<alpha>)` fetches another image through the same loader, limits, and source policy as sources and composites it at a position with an opacity; negative coordinates anchor from the right and bottom edges. A watermark that fails to load or decode fails the request (`502`) instead of being skipped.

- Overlay: `overlay(<url>,<x>,<y>,<alpha>,<blend>[,<width>])` generalizes the watermark with the CSS blend modes `normal`, `multiply`, `screen`, and `overlay`, and an optional width as a percentage of the base image's width.

- Text overlay: `text(<text>,<size>,<color>,<anchor>)` draws a label at one of nine grid anchors with a font configured on the server (`CRABCROP_FONT_PATH`); with no font configured, text requests fail with `501`.

- Keep profile: `keep_profile` keeps the source ICC profile instead of converting to sRGB; the profile is embedded in JPEG and PNG outputs.

**Formats and codecs**

- Decode set: JPEG, PNG, WebP, GIF, and JPEG XL sources decode in every build, with EXIF orientation applied before any operation runs, so phone photos come out upright.

- SVG sources: detected by content rather than declared type, sanitized before rendering (script elements, event handlers, and references to external resources are stripped), then rasterized at the requested size and run through the normal pipeline. Output defaults to PNG for SVG sources.

- HEIC/HEIF source decoding: HEIC sources decode in every build through the pure-Rust oxideav stack (`oxideav-heif` + `oxideav-h265`), with no feature flag, configuration, or external tool. The primary image is decoded with grid and overlay compositions resolved, EXIF orientation is applied at decode, embedded ICC profiles feed the usual convert-to-sRGB path, and 10-bit sources flatten to the 8-bit working space. A file whose only colour description lives in the bitstream (no `nclx` colour box) is converted per its SPS VUI, limited range included, matching other decoders instead of the container default. Corrupt HEIC sources keep failing with a specific 502.

- AVIF decoding: AVIF sources decode in builds that include AVIF decoding (the Docker image, the release binaries, and the Homebrew install) via dav1d; animated AVIF sources decode their first frame. Builds without it fail AVIF sources with `502` and an error naming the feature.

- Encode set: JPEG, PNG, WebP, JPEG XL, AVIF, and GIF. WebP encodes lossy at the requested quality and lossless at `quality(100)`; AVIF encoding speed is configurable (`CRABCROP_AVIF_SPEED`), as is the JPEG XL speed tier (`CRABCROP_JXL_SPEED`).

- Animated handling: animated GIFs keep their frames when the output format is GIF; animated WebPs keep their frames, delays, and loop count when the output is WebP or GIF; converting an animated source to a still format uses the first frame. Animated WebP output is always encoded losslessly.

- Video output (gifv): an animated GIF or WebP source can be served as H.264 MP4 or VP9 WebM through an explicit `format(mp4)` / `format(webm)` (never negotiated). The encode runs in an operator-configured external program (`CRABCROP_VIDEO_ENCODER`) over a fixed pipe contract (raw RGBA in on stdin, video out on stdout), bounded by a two-permit gate, a wall-clock timeout (`CRABCROP_VIDEO_TIMEOUT_SECS`, default 30), and an output cap (`CRABCROP_VIDEO_MAX_MB`, default 64). Unconfigured servers answer 501, still sources 400, encoder failures 502; crabcrop ships no codec, so codec licensing stays with the operator.

- Color management: when a source embeds an ICC profile, pixels are converted to sRGB before any operation runs, so sizes, crops, and filters all work in one known space; a source with no profile is assumed to be sRGB. CMYK JPEGs decode through the JPEG decoder's own CMYK conversion.

- Metadata policy: `CRABCROP_METADATA` is `strip` by default (outputs carry no source metadata) or `preserve`, which copies the source's EXIF and XMP segments (and Photoshop/IPTC for JPEG) into JPEG and PNG outputs, with the EXIF orientation rewritten to 1 because the pixels were already rotated at decode.

**Delivery and meta responses**

- Format negotiation: when the URL names no `format(...)`, the request's `Accept` header picks among AVIF, JPEG XL, and WebP by q-value, with ties broken by server preference in that order; a format refused with `q=0` is never picked and wildcards do not count as an offer. An explicit `format(...)` always wins. Animated sources never negotiate to JPEG XL, which would drop their frames. Responses `Vary` on `Accept` and the client hints.

- Client hints: when a URL sizes nothing itself, the server honors the width client hints (`Sec-CH-Width`, `Width`, and the viewport-width variants) and resizes to that width, quantized up to a multiple of `CRABCROP_WIDTH_STEP` (default 100) so nearby viewports share one cache entry. The `DPR` hint applies when the URL carries no `dpr`. A `Save-Data: on` request gets the default quality stepped down from 80 to 70 on the formats that carry a quality.

- Per-request cache lifetime: a `maxage:<seconds>` segment (0 to 31536000) overrides the response `Cache-Control` max-age for that request only. Header-only: it never enters the cache key, so URLs differing only in `maxage` share one stored image; `maxage:0` answers `max-age=0` without `immutable`.

- Meta responses: `Accept: application/json` on any image URL returns metadata about the processed image instead of the bytes: output `width`, `height`, `format`, and `bytes`, plus `source_width` and `source_height`. Meta responses are never cached and are served `Cache-Control: private, no-store`.

- Placeholders: with the `placeholders` media-type parameter, meta adds `dominant_color` (the average color as `#rrggbb`), `placeholder` (a 16 px-wide blurred JPEG preview as a base64 data URI), `palette` (the five most prominent colors), and `thumbhash` (a ThumbHash string client libraries decode into an aspect-correct blurred preview).

- Fallback image: with `CRABCROP_FALLBACK_IMAGE` set, a source that cannot be fetched or decoded degrades to the fallback, which runs through the same operations and is served with `200` and `X-Crabcrop-Fallback: 1`. Fallback responses are never cached. Signature and parse failures never fall back, and neither does a source-policy refusal.

- Cache headers: every image response reports `Cache-Status` (RFC 9211: memory hit, disk hit, or origin fetch) and, for each preset its URL referenced, a `Cache-Group` header (RFC 9875) that makes preset-wide invalidation one `/purge-group` call.

- Errors: error bodies are plain text with a short message; when JSON is requested, a grammar failure instead answers with the v2 JSON envelope carrying one of the parser's frozen, named error codes. Statuses distinguish malformed requests (`400`), bad signatures, expired URLs, and source-policy refusals (`403`), failed encodes and inference (`500`), valid URLs the server is not configured for (`501`), and source fetch or decode failures (`502`, `504` on fetch timeout).

**Caching**

- Memory tier: an in-memory LRU result cache bounded by bytes, 256 MB by default (`CRABCROP_CACHE_MAX_MB`); `0` disables it.

- Disk tier: with `CRABCROP_RESULT_STORAGE_DIR` set, a disk-backed second tier (512 MB default budget) sits under memory: results survive restarts, and disk hits are promoted into memory.

- Shared S3 tier: with `CRABCROP_RESULT_BUCKET` set, a third tier in one S3 bucket (or S3-compatible endpoint), namespaced by `CRABCROP_RESULT_PREFIX`, is shared by every replica. A shared hit promotes into disk and memory, new results are written through to the bucket in the background, and the tier is best-effort: a bucket error reads as a miss, never a failed request.

- Canonical keys: entries are keyed by the request's canonical form, so equivalent spellings of the same request share one entry. Negotiated responses are cached per negotiated format class, per effective DPR value, per quantized width-hint class, and per Save-Data class. Only `200` responses are cached; errors never are.

- Purge fan-out: `/purge` invalidates every stored variant of one result, and `/purge-group` clears a preset's whole recorded output set, across all cache tiers.

**AI features**

- Model posture: AI features run ONNX models in-process, from files the operator supplies; crabcrop never downloads a model, at startup or at request time. Models load at boot, before the server reports ready, and a configured model that cannot load is a startup error with a one-line diagnostic. With no model configured, the feature is off and requests that use it fail cleanly with `501`; there is no silent fallback. Inference runs behind a server-wide gate of 2 concurrent inferences. The runtime library comes from `CRABCROP_ORT_DYLIB` or the system library search path.

- Background removal: `removebg` runs a U²-Net-family model and keeps the subject over transparency; the output format must carry alpha (`png`, `webp`, `jxl`, or `avif`).

- Upscale: `upscale(2)` and `upscale(4)` run a super-resolution model (Real-ESRGAN-family exports follow the contract), with output capped at 4096 px per dimension. The model's native factor is probed at load with a test image and a non-integer factor is rejected at boot; a native 2x model composes two runs to land an exact `upscale(4)`, and source transparency is carried across even though the model only sees RGB.

- Face detection: a YuNet model (`CRABCROP_FACE_MODEL`) powers face-aware smart crops (centered on the area-weighted center of the faces found), `crop:...:face` around the largest face with `facepad` padding, and the `blurfaces` and `pixelatefaces` redaction filters. Chained face operations in one request share a single detection. No model, no faces, or a detection failure falls back to the detail-based focal point for crops; the redaction filters instead require the model (`501` without one) and a detection failure fails the request with `500` rather than serve an unredacted image.

- Class redaction and detections: a YOLOv8-family model (`CRABCROP_DETECT_MODEL`) powers `blurclass(<class>)` and `pixelateclass(<class>)`, which redact every detected object of a class (repeat the filter for more classes; matching is case-insensitive), and meta `detections` with the `detections=1` parameter. The class vocabulary comes from the model's own `names` metadata, or from `CRABCROP_DETECT_LABELS` for exports without it; `CRABCROP_DETECT_THRESHOLD` sets the score floor. A model whose output is not the YOLOv8 layout is rejected at boot.

- CLIP tags and alt text: a CLIP model pair with its tokenizer and a label file adds `tags` (zero-shot labels scored against the processed image, above `CRABCROP_CLIP_THRESHOLD`) and `alt_text` composed from the best labels to meta responses. Labels are embedded once at boot, so a meta request runs only the image encoder; a tagging failure omits the fields instead of failing the response.

- Declared detectors: `CRABCROP_DETECTORS_FILE` points at a JSON file wiring operator ONNX models to two built-in contracts, with no operator code loaded: `boxes` detectors (YOLOv8-family, class names from the declaration or the model's metadata) join class redaction (resolved built-in first, then declared in file order) and meta detections (aggregated, each box attributed with a `detector` field), and `focal` heatmap models feed smart crops between the face model and saliency (first declaration wins, later ones stand by). Every declaration is validated, loaded, and probed at boot; a broken one is a startup error naming it.

- AI image variant: `Dockerfile.ai` builds on the base image with a second layer adding the pinned ONNX Runtime, the U2Netp model, and the YuNet model, each hash-checked at image build time, and enables the features by setting the variables above.

**Client libraries and loaders**

- Signing spec and test vectors: the URL grammar and signing scheme are specified in `spec/url-signing.md`, an unversioned grammar where a URL minted today keeps its meaning, with a canonical set of test vectors in `spec/test-vectors.json` that every implementation runs as its test suite.

- Rust crate: `crabcrop-url` provides `sign`, `verify`, `build_url`, and `build_srcset`, plus the `purge_signature`, `purge_group_signature`, and `upload_signature` helpers. The server verifies with this same crate, so client and server cannot drift.

- TypeScript SDK: `@crabcrop/url` provides `sign`, `verifyUrlSignature`, `buildURL`, and `buildSrcSet`, the purge and upload signature helpers, and typed operation builders (`ops`, `filter`) that validate against the grammar and compile to the same canonical strings. Standard library only.

- Python SDK: `crabcrop_url` provides `sign`, `verify`, `build_url`, and `build_srcset`, plus the purge, purge-group, and upload signature helpers. Standard library only.

- Framework loaders: a Next.js custom loader for `next/image`, a Nuxt provider for `@nuxt/image`, and a Hydrogen loader for external (non-Shopify-CDN) images. All three recommend presets as the configuration, so client-side URL building never needs the secret.

- Playground: a static page (no build step, no server state) for building, signing, previewing, and debugging URLs in the browser; the secret never leaves the page, signatures verify locally, and pasted URLs are explained segment by segment with errors highlighted.

- MCP server: a first-party MCP server (stdio, no dependencies) exposing `sign_url`, `explain_url`, `fetch_meta`, and `purge_result` over the same SDK and HTTP surface, for coding agents.

**Operations and deployment**

- Operator CLI: the `crabcrop` binary doubles as its own CLI. The verbs are `sign` (sign a URL, validated against the server grammar first), `explain` (explain a URL segment by segment, verifying its signature), `meta` (fetch the meta JSON, with `--placeholders` and `--detections`), `purge`, `purge-group`, `upload` (store an original and print the source reference for transform URLs), and `convert-thumbor` (convert a Thumbor URL, re-signed, with losses reported on stderr).

- Health and metrics: `GET /health` answers `200` with the body `ok`, and `GET /metrics` serves Prometheus metrics: request counts by status class, request, load, and process durations, cache hit and miss counters, loader and engine error counters, and byte totals. Both are unauthenticated and served `Cache-Control: no-store`, so they belong on the internal network behind the proxy.

- Structured request logs: every image request writes one structured JSON log line to stdout with the operation summary, status, per-stage timings, and cache result, never the full source URL.

- Overload behavior: the processing, AI, and video gates shed overload with `503` rather than queue without bound.

- Upload metrics: accepted and rejected uploads count into `crabcrop_uploads_total` on the metrics endpoint.

- Container images: a multi-stage `Dockerfile` whose runtime image carries only the binary and CA certificates, and the `Dockerfile.ai` variant on top of it. Releases publish both to GHCR as `ghcr.io/dayvson/crabcrop` and `ghcr.io/dayvson/crabcrop-ai`, tagged with the version and `latest`.

- Kubernetes and cloud recipes: `deploy/helm` is a Helm chart and `deploy/kustomize` the same workload in kustomize form (base, components for presets, the disk tier, and the AI variant, plus single and multi-replica overlays); `deploy/compose` runs the workload under Docker Compose. Per-cloud recipes for ECS Fargate, Cloud Run, and Container Apps each carry their secret wiring and a smoke checklist.

- Release artifacts: cargo-dist builds the server for Linux and macOS on both architectures, plus a shell installer and SHA-256 checksums, attached to the GitHub Release; the Homebrew formula is generated from the release checksums.

- Migration guides: `docs/` carries guides for moving existing URL templates from Thumbor, from imgproxy, and from Cloudinary.

### Fixed

- Boot with an AI model configured but no loadable ONNX Runtime (no `CRABCROP_ORT_DYLIB` and no system runtime) died on a panic inside the `ort` crate instead of failing startup. The runtime is now loaded explicitly from the configured path, or from the path ort itself would search (`ORT_DYLIB_PATH`, then the system library name), and a load failure is the usual `cannot load ...` boot diagnostic with exit code 1.
