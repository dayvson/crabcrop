# crabcrop

crabcrop is an on-demand image optimization service written in Rust. You request an image with the operations encoded in the URL — resize, crop, convert, filter — and crabcrop fetches the source image, applies the operations in order, and returns the result. There are no pre-generated thumbnails and no build step: the URL is the API.

crabcrop is for sites and apps that serve the same images at many sizes and want signed, cacheable image URLs in front of an origin image store.

## Quickstart

**1. Start the server.** A secret is required. For local development, also allow unsigned URLs:

```bash
CRABCROP_SECRET=dev-secret CRABCROP_ALLOW_UNSAFE=1 cargo run --release
# crabcrop listening on port 3000
```

**2. Request an image.** This resizes a source photo to 300×200:

```bash
curl "http://localhost:3000/unsafe/300x200/https://example.com/photo.jpg" --output photo.jpg
```

Replace `example.com/photo.jpg` with the URL of an image you control. You get back a 300×200 JPEG.

**3. Sign your URLs in production.** Drop `CRABCROP_ALLOW_UNSAFE` and put an HMAC signature in the first path segment:

```bash
curl "http://localhost:3000/3c1f6c180116661877bd4c85fd201a178087d74c1cc188cc584e819d901308c6/300x200/https://example.com/photo.jpg" --output photo.jpg
```

That signature is HMAC-SHA256 of `300x200/https://example.com/photo.jpg` under the secret `your-secret`. See [Sign a request](#sign-a-request).

## How requests work

Every image request has three parts:

```text
GET /<signature>/<operations>/<source-url>
```

| Part | Description |
| --- | --- |
| `signature` | An HMAC-SHA256 (hex) of everything after it, or the literal `unsafe`. |
| `operations` | Zero or more operation segments, applied in path order. |
| `source-url` | The full URL of the source image, scheme included. It may contain slashes; the first segment that is not an operation starts it. |

```text
/unsafe/crop:1200x630:smart/filters:grayscale:quality(80)/https://example.com/photo.jpg
```

### Sign a request

Compute HMAC-SHA256 over the remainder of the path — the operations and the source URL exactly as they appear, with no leading slash — keyed with your `CRABCROP_SECRET`, and hex-encode the result:

```bash
printf '%s' '300x200/https://example.com/photo.jpg' | openssl dgst -sha256 -hmac 'your-secret'
# SHA2-256(stdin)= 3c1f6c180116661877bd4c85fd201a178087d74c1cc188cc584e819d901308c6
```

Use that hex string as the first path segment. Anyone can verify the scheme against this test vector: with secret `Jefe`, the HMAC-SHA256 of `what do ye want for nothing?` is `ba5396e10039ec002bf1fde117eee48df8c2f06bc47b5f2d3272ea77f6862da9`.

A request with a bad signature gets `403` and the body `bad signature`. The literal signature `unsafe` skips verification, but only when the server runs with `CRABCROP_ALLOW_UNSAFE=1` (or `true`). Keep it off in production.

### Source images

The loader fetches the source URL under fixed limits:

- HTTP and HTTPS only.
- 10 seconds total per fetch, including redirects.
- 10 MiB maximum response body; larger sources fail with `413`.
- Up to 5 redirects. The [source policy](#configuration) is re-checked on every hop.
- If the response carries a `Content-Type` header, it must start with `image/`.

JPEG, PNG, WebP, GIF, and JPEG XL sources are decoded. EXIF orientation is applied before any operation runs, so phone photos come out upright. Animated GIFs keep their frames when the output format is GIF; converting an animated GIF to a still format uses the first frame. HEIC/HEIF sources are rejected, and AVIF sources cannot be decoded in this build (AVIF is an output-only format here); both fail with `502`.

SVG sources are supported and detected by content, not by the declared content type. An SVG is sanitized before it is rendered: script elements, event handlers, and references to external resources are stripped, so a source SVG can only draw what it carries with it. The sanitized document is rasterized at the size the URL asks for (or its intrinsic size when the URL asks for none) and then runs through the normal pipeline. Output defaults to PNG for SVG sources.

## Operations

Operations run in the order they appear in the path.

### Resize

```text
<width>x<height>
```

| Form | Result |
| --- | --- |
| `300x200` | Exactly 300×200 px. |
| `300x0` or `300x` | Width 300 px, height scaled proportionally. |
| `0x200` or `x200` | Height 200 px, width scaled proportionally. |
| `0x0` | Original size. |
| `origx100` | Source width, height 100 px. `orig` works on either axis. |
| `-300x200` | Resize to 300×200 and flip horizontally. |
| `300x-200` | Resize to 300×200 and flip vertically. |

```bash
curl "http://localhost:3000/unsafe/300x0/https://example.com/photo.jpg"
```

### Fit inside a box

```text
fit-in:<width>x<height>
```

Scales the image proportionally so it fits inside the box. A `0` axis is unbounded. Flips are not allowed here.

```bash
curl "http://localhost:3000/unsafe/fit-in:800x600/https://example.com/photo.jpg"
```

### Crop to an aspect ratio

```text
crop:<width>x<height>[:<align>]
```

Crops the image to the width:height aspect ratio, then resizes to exactly that width and height. When the aspect ratios differ, `align` chooses which part survives:

| Align | Keeps |
| --- | --- |
| `center` | The middle. This is the default. |
| `top` | The top edge. |
| `bottom` | The bottom edge. |
| `left` | The left edge. |
| `right` | The right edge. |
| `smart` | A window centered on the focal point: a detected face when a face model is configured (see [AI features](#ai-features)), otherwise the point of greatest local detail in the image. |

```bash
curl "http://localhost:3000/unsafe/crop:1200x630:top/https://example.com/photo.jpg"
```

### Crop a rectangle

```text
crop-rect:<left>x<top>:<right>x<bottom>
```

Cuts the exact pixel rectangle between the two corners. `right` must be greater than `left`, and `bottom` greater than `top`. A rectangle that falls outside the source image fails with `400`.

```bash
curl "http://localhost:3000/unsafe/crop-rect:100x50:700x450/https://example.com/photo.jpg"
```

### Filters

```text
filters:<name>(<params>):<name>(<params>):...
```

Filters chain inside one `filters:` segment, separated by `:`, and run in the order written. A filter with no parameters may omit the parentheses: `grayscale` and `grayscale()` are the same. You may also repeat the `filters:` segment in a path. When a directive repeats, the last one wins.

**Pixel filters** change the image itself:

| Filter | Parameters | Description | Example |
| --- | --- | --- | --- |
| `grayscale` | none | Desaturate to gray. | `filters:grayscale` |
| `invert` | none | Invert every channel. | `filters:invert` |
| `blur` | `blur(2.5)` — sigma, a positive number | Gaussian blur. | `filters:blur(2.5)` |
| `brighten` | `brighten(40)` — integer added to every channel, clamped to 0–255 | Brighten (or darken, with a negative value). | `filters:brighten(40)` |
| `contrast` | `contrast(25)` — number; 0 is unchanged, negative lowers contrast, positive raises it | Adjust contrast. | `filters:contrast(25)` |
| `huerotate` | `huerotate(90)` — degrees, integer | Rotate the hue. | `filters:huerotate(90)` |
| `sharpen` | `sharpen(1.5)` — strength, a positive number | Unsharp-mask sharpening. | `filters:sharpen(1.5)` |
| `saturation` | `saturation(1.5)` — 0 is grayscale, 1 is unchanged | Scale color saturation. | `filters:saturation(0.5)` |
| `rgb` | `rgb(10,-5,0)` — per-channel offsets, each −255 to 255 | Add offsets to the red, green, and blue channels. | `filters:rgb(10,-5,0)` |
| `equalize` | none | Histogram equalization, per channel. | `filters:equalize` |
| `rotate` | `rotate(90)` — a multiple of 90; negatives allowed | Rotate the image. | `filters:rotate(-90)` |
| `trim` | `trim` or `trim(20)` — tolerance 0–255, default 10 | Crop away the border that matches the top-left corner color within the tolerance. | `filters:trim(20)` |

```bash
curl "http://localhost:3000/unsafe/600x400/filters:sharpen(1.2):saturation(1.1)/https://example.com/photo.jpg"
```

**Encoding directives** leave the pixels alone and steer the encoder:

| Directive | Parameters | Description | Example |
| --- | --- | --- | --- |
| `format` | `format(webp)` — `jpeg` (or `jpg`), `png`, `webp`, `jxl`, `avif`, `gif` | Output format. Overrides `Accept` negotiation. | `filters:format(avif)` |
| `quality` | `quality(80)` — 0 to 100, or `quality(auto)` | Encoding quality for JPEG, JPEG XL, and AVIF. Default 80. | `filters:quality(auto)` |
| `max_bytes` | `max_bytes(50000)` — byte budget, at least 1 | Re-encode at decreasing quality until the output fits the budget. | `filters:max_bytes(50000)` |
| `fill` | `fill(ff0000)` — six hex digits, RGB | Background color used to flatten transparency when encoding to JPEG. | `filters:fill(ffffff)` |
| `no_upscale` | none | Cap every resize in the request at the source dimensions. | `filters:no_upscale` |
| `dpr` | `dpr(2)` — one of `1`, `1.5`, `2`, `3` | Scale size and crop targets by the device pixel ratio, so one URL serves standard and retina displays. | `filters:dpr(2)` |

Notes on the directives:

- `quality(auto)` picks a quality per image, from 50 for flat images to 95 for detailed ones, based on content analysis. When crabcrop picks the quality — for `quality(auto)`, or when `max_bytes` steps quality down — the response carries an `X-Crabcrop-Quality` header with the value used.
- `max_bytes` can only steer formats with a quality setting: JPEG, JPEG XL, and AVIF. If quality reaches its floor and the output is still over budget, you get the smallest encode rather than an error. PNG, WebP, and GIF ignore the budget.
- WebP output is lossless only, so `quality` has no effect on WebP, PNG, or GIF output.
- JPEG has no alpha channel. With `fill`, transparent pixels are composited over the fill color; without it, the stored RGB values are used as-is.
- `no_upscale` applies to the whole request wherever it appears: a `300x200` resize of a 200×150 source returns 200×150, scaled proportionally.
- `dpr` also applies request-wide, and it scales targets *before* the `no_upscale` cap sees them, so `dpr` never invents pixels past the source when both are set. Targets round to whole pixels. `crop-rect` coordinates address source pixels and are not scaled. When the URL carries no `dpr`, the server honors the `DPR` client hint header (same set of values; an invalid hint is ignored). An explicit `dpr` in the URL always beats the header.

```bash
curl "http://localhost:3000/unsafe/1200x0/filters:format(webp):quality(auto)/https://example.com/photo.jpg"
```

**Watermark** overlays another image:

```text
filters:watermark(<url>,<x>,<y>,<alpha>)
```

| Parameter | Description |
| --- | --- |
| `url` | The watermark image URL, percent-encoded (it lives inside a path segment, so encode at least `:` and `/`). It may itself contain commas. |
| `x`, `y` | Position of the watermark's top-left corner, in pixels. Negative values anchor from the right and bottom edges. |
| `alpha` | Opacity, 0 (invisible) to 100 (opaque). The watermark's own alpha is honored. |

The server fetches each watermark through the same loader, limits, and source policy as source images. A watermark that fails to load or decode fails the request (`502`) instead of being skipped.

```bash
# Overlay https://cdn.example.com/logo.png at the bottom right, 80% opaque
curl "http://localhost:3000/unsafe/filters:watermark(https%3A%2F%2Fcdn.example.com%2Flogo.png,-10,-10,80)/https://example.com/photo.jpg"
```

**Text overlay** draws a label with a font configured on the server:

```text
filters:text(<text>,<size>,<color>,<anchor>)
```

| Parameter | Description |
| --- | --- |
| `text` | The label, percent-encoded. It may contain commas and unicode once encoded. |
| `size` | Font size in pixels, 1 to 1000. |
| `color` | `rrggbb`, or `rrggbbaa` to include alpha. |
| `anchor` | One of nine grid points: `top-left`, `top-center`, `top-right`, `center-left`, `center`, `center-right`, `bottom-left`, `bottom-center`, `bottom-right`. The label is padded from the edges by half the font size. |

The text is drawn when the filter runs, at whatever resolution the image has at that point in the path, so put `text` after your resize to keep it sharp. The font comes from `CRABCROP_FONT_PATH` on the server; with no font configured, `text` requests fail with `400`.

```bash
curl "http://localhost:3000/unsafe/800x600/filters:text(Sale%20ends%20soon,32,ffffff,bottom-right)/https://example.com/photo.jpg"
```

**AI filters** need a model configured on the server. See [AI features](#ai-features).

| Filter | Parameters | Description | Example |
| --- | --- | --- | --- |
| `removebg` | none | Remove the background, keeping the subject over transparency. Output format must carry alpha: `png`, `webp`, `jxl`, or `avif`. | `filters:removebg:format(png)` |
| `upscale` | `upscale(2)` or `upscale(4)` | AI super-resolution by that factor. Output is capped at 4096 px per dimension. | `filters:upscale(2)` |

```bash
curl "http://localhost:3000/unsafe/filters:removebg:format(png)/https://example.com/product.jpg"
```

## Format negotiation

When the URL names no `format(...)`, crabcrop picks the output format from the request's `Accept` header:

1. `image/avif` → AVIF
2. `image/webp` → WebP
3. Otherwise → the source image's format (JPEG if the format cannot be detected)

A format refused with `q=0` is never picked, and wildcards such as `*/*` do not count as an offer. An explicit `format(...)` in the URL always wins over negotiation. Responses include `Vary: Accept, DPR` so downstream caches key this correctly (the `DPR` hint likewise changes sizing; see `dpr` above).

## Meta responses

Send `Accept: application/json` with any image URL to get metadata about the processed image instead of the image bytes:

```bash
curl -H "Accept: application/json" "http://localhost:3000/unsafe/300x200/https://example.com/photo.jpg"
```

```json
{"width":300,"height":200,"format":"jpeg","bytes":18234,"source_width":1200,"source_height":800}
```

| Field | Description |
| --- | --- |
| `width`, `height` | Output dimensions in pixels. |
| `format` | Output format: `jpeg`, `png`, `webp`, `jxl`, `avif`, or `gif`. |
| `bytes` | Size of the processed image in bytes. |
| `source_width`, `source_height` | Source dimensions in pixels, after EXIF orientation. |

Add the `placeholders` parameter to the media type to also get placeholder data, computed from the final processed pixels:

```bash
curl -H "Accept: application/json; placeholders=1" "http://localhost:3000/unsafe/300x200/https://example.com/photo.jpg"
```

This adds two fields: `dominant_color`, the image's average color as `#rrggbb`, and `placeholder`, a 16 px-wide blurred JPEG preview as a base64 data URI — handy for rendering a preview while the real image loads. Meta requests are never cached, and errors in meta mode keep the usual status codes and plain-text bodies.

## Responses

A successful image response carries:

| Header | Value |
| --- | --- |
| `Content-Type` | The output format's media type, e.g. `image/webp`. |
| `Cache-Control` | `public, max-age=31536000, immutable` |
| `Vary` | `Accept, DPR` |
| `X-Crabcrop-Quality` | Present only when crabcrop chose the quality (`quality(auto)` or a `max_bytes` reduction); the value used. |
| `X-Crabcrop-Fallback` | `1` when the response is the configured fallback image rather than the source (see [Fallback image](#fallback-image)). |

### Caching

Signed URLs are content-addressed, so successful responses are cached and served without re-fetching or re-processing. The first tier is an in-memory LRU bounded by bytes: 256 MB by default, configurable with `CRABCROP_CACHE_MAX_MB` (`0` disables it). With `CRABCROP_RESULT_STORAGE_DIR` set, a second, disk-backed tier (bounded by `CRABCROP_RESULT_STORAGE_MAX_MB`, default 512 MB) sits underneath: results survive restarts, and disk hits are promoted into memory. Only `200` responses are cached; errors never are. Because the response can depend on the request headers, negotiated responses are cached per negotiated format class and per effective DPR value.

### Fallback image

With `CRABCROP_FALLBACK_IMAGE` set (a URL, or a path to a local file read at startup), a source that cannot be fetched or decoded degrades to the fallback instead of an error: the fallback runs through the same operations and is served with `200` and the header `X-Crabcrop-Fallback: 1`. Signature and parse failures never fall back — a `403` or `400` stays honest — and neither does a source-policy refusal. Fallback responses are never cached, so a recovered origin is picked up on the next request. With no fallback configured, error behavior is unchanged.

## Errors

Error bodies are plain text with a short message. The statuses you can get:

| Status | Meaning |
| --- | --- |
| `400` | The path or operations do not parse, or the operation is not supported: a malformed segment, an unknown filter or parameter, `removebg` into a format without alpha, `upscale` past the 4096 px cap, a `crop-rect` outside the image — or an AI filter used on a server with no model configured. |
| `403` | Bad signature, or the source URL was refused by the source policy (host not in `CRABCROP_ALLOWED_SOURCES`, or resolving to a non-public address while `CRABCROP_BLOCK_PRIVATE_SOURCES` is on). |
| `413` | The source image is larger than the 10 MiB fetch limit. |
| `500` | Encoding the output failed, or an AI model failed at inference time. |
| `502` | The source could not be fetched (network error, a non-2xx answer from the origin, or a non-image `Content-Type`), or the fetched bytes could not be decoded as an image — including HEIC sources and undecodable formats. A watermark that fails to load also lands here. |
| `504` | The source fetch timed out (10 seconds). |

For example, a `removebg` request to a server with no background-removal model returns `400` with the body `removebg is not configured on this server`.

## Endpoints

| Route | Description |
| --- | --- |
| `GET /health` | Health check. Returns `200` with the body `ok`. |
| `GET /metrics` | Prometheus metrics in text format: request counts by status class (`crabcrop_requests_total`), request/load/process durations, cache hit and miss counters, loader and engine error counters, and byte totals. |
| `POST /purge/<signature>/<path>` | Invalidate every stored variant of one result, in both cache tiers. See below. |
| `GET /<signature>/...` | Image requests, as documented above. |

### Purge a cached result

```bash
curl -X POST "http://localhost:3000/purge/<signature>/300x200/https://example.com/photo.jpg"
```

The `<path>` is the original request path without its signature, and the purge signature is the HMAC-SHA256 of the literal string `purge/<path>` under the same secret — only the secret holder can invalidate. (This differs from image signatures, which cover the path alone.) The response is `200` with a body like `purged 2`: the count covers the entry and all of its variants (per format class and DPR). A bad purge signature is a `403` and removes nothing; purging a path that is not stored answers `purged 0`.

Every image request also writes one structured JSON log line to stdout, with the operation summary, status, per-stage timings, and cache result (never the full source URL).

## Configuration

All configuration comes from environment variables. Invalid values for the numeric settings are startup errors — the server refuses to start rather than run with a wrong port or cache budget.

| Variable | Default | Description |
| --- | --- | --- |
| `CRABCROP_SECRET` | — (required) | The HMAC secret used to verify signed URLs. The server will not start without it. |
| `PORT` | `3000` | Listen port. |
| `CRABCROP_ALLOW_UNSAFE` | off | `1` or `true` honors `unsafe` URLs. Anything else, including unset, means off. |
| `CRABCROP_CACHE_MAX_MB` | `256` | Result cache budget in megabytes. `0` disables the cache. |
| `CRABCROP_ALLOWED_SOURCES` | unset (all hosts allowed) | Comma-separated host suffixes the loader may fetch from. A suffix matches the host itself or any subdomain of it. |
| `CRABCROP_BLOCK_PRIVATE_SOURCES` | off | `1` or `true` refuses sources whose host is, or resolves to, a loopback, private, link-local, or unspecified address. Checked before connecting and again on every redirect hop. |
| `CRABCROP_REMOVEBG_MODEL` | unset | Path to a U²-Net-family ONNX model for the `removebg` filter. See [AI features](#ai-features). |
| `CRABCROP_UPSCALE_MODEL` | unset | Path to a super-resolution ONNX model for the `upscale` filter. See [AI features](#ai-features). |
| `CRABCROP_FACE_MODEL` | unset | Path to a YuNet ONNX model for face-aware `smart` crops. See [AI features](#ai-features). |
| `CRABCROP_ORT_DYLIB` | unset | Path to the ONNX Runtime shared library, when it is not on the library search path. |
| `CRABCROP_FONT_PATH` | unset | Path to a TTF/OTF font for the `text(...)` overlay filter, loaded at startup. Unset disables the filter. |
| `CRABCROP_FALLBACK_IMAGE` | unset | URL or local file path of the fallback image served when a source cannot be fetched or decoded. See [Fallback image](#fallback-image). |
| `CRABCROP_RESULT_STORAGE_DIR` | unset | Directory for the disk-backed result store. Unset disables the second cache tier. |
| `CRABCROP_RESULT_STORAGE_MAX_MB` | `512` | Disk result store budget in megabytes. `0` disables storage even with a directory set. |

One known limit: with `CRABCROP_BLOCK_PRIVATE_SOURCES`, the loader resolves the host to check its addresses, and the HTTP client resolves again when it connects. A hostile DNS server could answer differently the second time (DNS rebinding). The check still stops direct internal URLs, which is the common exposure.

## AI features

Three features run ONNX models in-process: `removebg` (background removal), `upscale` (super-resolution), and face detection for `smart` crops. The models are files you supply — deployment artifacts, like the `Dockerfile.ai` image variant in this repo. crabcrop never downloads a model, at startup or at request time.

How configuration works:

- Point `CRABCROP_REMOVEBG_MODEL` at a U²-Net-family model file (320×320 input; the U2Netp export is a good fit) and/or `CRABCROP_UPSCALE_MODEL` at a super-resolution model. The upscale contract is dynamic-size RGB in (values 0..1), the same image out at an integer factor; Real-ESRGAN-family ONNX exports follow it.
- Point `CRABCROP_FACE_MODEL` at a YuNet model to make `crop:...:smart` center on detected faces (the area-weighted center of the faces found). Face detection only ever improves a crop: no model, no faces, or a detection failure all fall back to the detail-based focal point, and a smart crop without the model behaves exactly as before.
- Models load at boot, before the server reports ready. A configured model that cannot be loaded — missing file, wrong shape — is a startup error with a one-line diagnostic, not a surprise for the first request. The upscale model's native factor is probed at load with a 16×16 test image; a model that does not enlarge by an integer factor is rejected at boot.
- With no model configured, the filter is off: requests that use it fail cleanly with `400` (`removebg is not configured on this server`, `upscale is not configured on this server`). There is no silent fallback — `upscale` never degrades into a plain resize.
- Inference runs behind a server-wide gate of 2 concurrent inferences, so a burst of AI requests cannot starve the normal transform pipeline.
- `upscale(4)` on a server whose model is natively 2× composes two model runs and lands the exact requested size; when the native factor does not divide the request, a final high-quality resize trims the difference. Source transparency is carried across an upscale even though the model only sees RGB.
- The ONNX Runtime library itself comes from `CRABCROP_ORT_DYLIB`, or the system library search path when that is unset.

The `Dockerfile.ai` variant in this repo shows the intended shape: the base image stays free of AI payload, and a second layer adds the pinned ONNX Runtime 1.31.0, the U2Netp model, and the YuNet model, each hash-checked at image build time, and enables the features by setting the variables above. Layer your own super-resolution model the same way and set `CRABCROP_UPSCALE_MODEL` to enable `upscale`.

## Development

```bash
cargo build
cargo test
```

Running the server locally takes only a secret:

```bash
CRABCROP_SECRET=dev-secret cargo run
```

## License

MIT. See [LICENSE](LICENSE).
