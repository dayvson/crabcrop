# crabcrop

<p align="center">
  <img src="assets/crabcrop-mascot.webp" alt="The crabcrop mascot: a crochet crab waving hello and holding a photo" width="260">
</p>

<p align="center">
  <a href="https://crabcrop.org/">Website</a> · <a href="https://crabcrop.org/docs.html">Documentation</a>
</p>

**Blazing fast, on-demand images.**

crabcrop is a Rust-built, lightning-fast on-demand image API supporting all major formats in and out, with over 42 filters and AI features like background removal. It resizes, crops, converts, and filters your images straight from a signed URL. One binary, no pre-generated thumbnails, no build step: the URL is the API.

Everything an image pipeline does, in one signed URL: resize, crop, adjust, shape, overlay, and encode, chained in the order written, plus background removal, studio mode, smart face-safe crops, redaction, and content provenance. Ask for JSON instead of an image and the meta endpoint describes it: dimensions, detections, OCR text, and more.

crabcrop is for sites and apps that serve the same images at many sizes and want signed, cacheable image URLs in front of an origin image store.

## Install

**Shell installer** (macOS and Linux):

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/dayvson/crabcrop/releases/latest/download/crabcrop-installer.sh | sh
```

**Homebrew:** download `crabcrop.rb` from the [latest release](https://github.com/dayvson/crabcrop/releases/latest), then:

```bash
brew install --formula ./crabcrop.rb
```

**Docker:** the base image, and the AI variant with background removal and face detection on board:

```bash
docker pull ghcr.io/dayvson/crabcrop:0.9.3
docker pull ghcr.io/dayvson/crabcrop-ai:0.9.3
docker run -e CRABCROP_SECRET=your-secret -p 3000:3000 ghcr.io/dayvson/crabcrop:0.9.3
```

**Release binaries:** prebuilt tarballs for macOS and Linux, on x86_64 and arm64, are on the [releases page](https://github.com/dayvson/crabcrop/releases). To build from source instead, see [Deployment](#deployment).

## Quickstart

**1. Start the server.** A secret is required. For local development, also allow unsigned URLs:

```bash
CRABCROP_SECRET=local-dev-secret CRABCROP_ALLOW_UNSAFE=1 cargo run --release
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
| `source-url` | The source locator: a full URL with its scheme, a bucket or file locator, or a `root:<name>/<path>` short source naming a [source root](#source-roots). It may contain slashes; the first segment that is not an operation starts it. |

```text
/unsafe/crop:1200x630:smart/filters:grayscale:quality(80)/https://example.com/photo.jpg
```

### Sign a request

Compute HMAC-SHA256 over the remainder of the path, the operations and the source URL exactly as they appear, with no leading slash, keyed with your `CRABCROP_SECRET`, and hex-encode the result:

```bash
printf '%s' '300x200/https://example.com/photo.jpg' | openssl dgst -sha256 -hmac 'your-secret'
# SHA2-256(stdin)= 3c1f6c180116661877bd4c85fd201a178087d74c1cc188cc584e819d901308c6
```

Use that hex string as the first path segment. Anyone can verify the scheme against this test vector: with secret `Jefe`, the HMAC-SHA256 of `what do ye want for nothing?` is `ba5396e10039ec002bf1fde117eee48df8c2f06bc47b5f2d3272ea77f6862da9`.

A request with a bad signature gets `403` and the body `bad signature`. The literal signature `unsafe` skips verification, but only when the server runs with `CRABCROP_ALLOW_UNSAFE=1` (or `true`, `yes`, `on`). Keep it off in production. Purges are the exception that never bends: `/purge` and `/purge-group` always require an HMAC signature made with the secret, even in unsafe mode.

### Sealed URLs

Signing proves a URL was minted by someone holding the secret, but it hides nothing: the operations and the source sit in the path for anyone to read. That readability is the design (`The URL is the API` cuts both ways, and it is what makes crabcrop URLs debuggable), and for most deployments it is the right default. When it is not, the same remainder can be **sealed**: encrypted into an opaque token that sits in the signature slot.

```text
https://images.example.com/s1.AAECAwQFBgcICQoL4rtOMQ/.../photo.jpg
```

The token is `s1.` plus base64url(nonce, ciphertext, tag): AES-256-GCM over the whole remainder, under a key derived from the same secret, so there is no second secret to manage. The text after the token (`photo.jpg` above) is a friendly name: purely cosmetic, never parsed or authenticated, present so sealed URLs still read as images in logs, analytics, and page source. A sealed URL serves exactly what its plain spelling serves: the server decrypts the remainder and runs the same pipeline, so both forms share one cache entry, one ETag, and one purge target. Mint one with the builders (`build_sealed_url` in Rust, `buildSealedURL` in TypeScript, `build_sealed_url` in Python, where sealing alone needs the `crabcrop-url[seal]` extra) or the CLI:

```bash
crabcrop seal 'filters:format(webp):quality(80)/https://media.example.com/heroes/dog.jpg' dog.jpg
```

Know what sealing does and does not do:

- **It hides the recipe from page readers.** Anyone who can see the URL (a page visitor, a competitor scraping your pages) learns nothing about the operations or the origin. Anyone who can observe the request itself (your CDN, a network intermediary) sees the token in the request line regardless; sealing is not transport privacy, and TLS is the answer there.
- **It is not access control.** A sealed URL is a bearer artifact anyone can replay, exactly like a signed one. Authorization is still the source allowlist, expiry (`exp:` works inside a sealed payload, unchanged), and whatever fronts the deployment.
- **It trades away debuggability at the server.** `/explain` refuses sealed URLs: a server endpoint that decrypts would be an oracle for anyone who lifts a token off a public page. To debug one, unseal it locally (`crabcrop unseal <url>`, under your own secret) and explain the plain form. The meta document for a sealed request likewise reports the render, never the recipe. Sealed URLs are for links that must be public and unreadable at the same time: most links are not, and the readable form stays the default and the recommendation.

Failures are uniform: a malformed token, a token minted under another secret, a token whose payload does not parse, all answer `403` with the body `bad signature`, indistinguishable from each other and from a bad HMAC. (A token that opens but has expired answers `signature expired`, exactly like the plain form.) Rotation works the way it does for signatures, inherited rather than reinvented: a token may carry a key ID (`k2:s1.…`), the server opens keyed tokens against the named key only, and during a rotation window an unkeyed token opens under the previous secret until that secret is retired. Minting refuses a remainder longer than 2,048 bytes; a recipe that long belongs in a preset. The full construction is in `spec/url-signing.md`.

### Expiring URLs

Add an `exp:<unix-seconds>` segment anywhere among the operations to make a signed URL expire:

```text
/<signature>/exp:1893456000/300x200/https://example.com/photo.jpg
```

The segment is inside the signed remainder, so the HMAC covers it: extending the expiry without the secret breaks the signature. Past the expiry the request gets `403` with the body `signature expired`, and expired URLs never serve from the cache. The comparison is exact: the URL is valid strictly before its `exp` second, with no built-in allowance for clock skew. Keep the server's clock synchronized (NTP), and mint expiries with whatever margin your own clock situation needs. URLs without `exp` never expire, exactly as before.

### Per-request cache lifetime

Image responses are cached by clients and CDNs for a year: `Cache-Control: public, max-age=31536000, immutable`, because a signed URL's bytes never change. When one response should live shorter, add a `maxage:<seconds>` segment anywhere among the operations:

```text
/<signature>/maxage:60/300x200/https://example.com/photo.jpg
```

The value runs from 0 to 31536000. The response to that request carries `Cache-Control: public, max-age=60, immutable` instead of the year; every other response is unchanged. `maxage:0` answers `Cache-Control: public, max-age=0` with no `immutable`: the response is stale on arrival, so caches must revalidate before reusing it.

The directive changes the response header only. It never enters the cache key, so URLs that differ only in `maxage` share one stored image, and purging is unaffected. The shared result cache (S3) stores bodies, not headers, so the per-response header composes there too: each request's lifetime is applied as its response is built, wherever the body came from.

Like `exp`, the segment sits inside the signed remainder and repeated segments resolve to the last one. A malformed or out-of-range value is a `400`, never a silent fallback to the default. Presets stay immutable by default; putting `maxage` on a preset URL is the caller's explicit choice for that response, and preset definitions cannot contain it (it is per-request metadata, like `exp`).

### Revalidation

Every cacheable image response also carries a strong `ETag`, derived from the response's canonical cache key, so it identifies the exact bytes of the variant that key selects. A client or CDN holding a copy can revalidate with `If-None-Match`: when the value matches (an exact match, a weak `W/` spelling, one entry of a comma list, or `*`), the answer is `304 Not Modified` with no body, on cache hits and fresh renders alike; anything else gets the full `200`. The validator applies only to public `200` image responses: meta responses (`private, no-store`), fallbacks, and errors carry no `ETag` and are never answered with a `304`. It pairs with `maxage:0` for URLs whose copies should be checked before every reuse.

### Source images

The loader fetches the source under fixed limits, for every source kind:

- 10 seconds total per fetch, including redirects.
- 10 MiB maximum body; larger sources fail with `502` (the request was fine; the origin object is too big).
- If an HTTP response carries a `Content-Type` header, it must start with `image/`. The exceptions are the media selections: when the URL carries a `frame:` segment, a `video/*` Content-Type is accepted too (see [Video frames](#video-frames)), and when it carries a `page:` segment, `application/pdf` is accepted (see [PDF pages](#pdf-pages)). Both admissions belong to the request, never to the response alone.

Decoding is bounded too, so a small file cannot expand into gigabytes: at most 16384 px per axis and 256 MiB of decoder allocations, and animated GIF and WebP sources at most 200 frames and 100 million decoded pixels in total. Sources past the bounds fail with `502`.

HTTP and HTTPS sources additionally allow up to 5 redirects, with the [source policy](#configuration) re-checked on every hop.

Three more source kinds cover originals that do not live behind a public URL:

- `s3://bucket/key`: fetched from AWS S3, or from an S3-compatible store (MinIO, R2, Spaces) when `CRABCROP_S3_ENDPOINT` is set. Requests are signed with SigV4 when AWS credentials are in the environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`); with none, the fetch is unsigned and public buckets just work.
- `gs://bucket/key`: the same, against Google Cloud Storage's S3-compatible XML API, signed with the HMAC pair `CRABCROP_GCS_ACCESS_KEY` / `CRABCROP_GCS_SECRET_KEY` when set.
- `file:///path`: a local file under `CRABCROP_FILE_ROOT`. The path is canonicalized per request and must stay under the root, so `..` segments and symlink escapes are refused. Unset root disables file sources.

Bucket sources are gated by `CRABCROP_ALLOWED_BUCKETS` rather than the host policy, since the endpoint is operator configuration, not request input. When signing credentials are configured, the allowlist is required: the server refuses to boot without it, because signed fetches would otherwise reach every bucket the credentials can read. List `*` to deliberately allow all buckets. A bucket or object key carrying a dot segment (a `.` or `..` path component) is refused, since the request path would collapse it past the allowlisted bucket; dots inside a segment name are fine. Bucket names must also follow the S3 naming rules (3 to 63 characters of lowercase letters, digits, dots, and hyphens, starting and ending with a letter or digit, no adjacent dots, not an IP address spelling): the name is interpolated into the request host, so a name carrying a `?` or another host-breaking character is refused at parse time, whatever the allowlist says. Store responses are not redirected: a 3xx from a store surfaces as its status code.

JPEG, PNG, WebP, GIF, and JPEG XL sources are decoded. EXIF orientation is applied before any operation runs, so phone photos come out upright. Animated GIFs keep their frames when the output format is GIF; converting an animated GIF to a still format uses the first frame. Animated WebPs keep their frames when the output format is WebP or GIF, with frame delays and the loop count preserved; converting one to a still format uses the first frame. Animated WebP output is always encoded losslessly: the pure-Rust WebP encoder has no lossy animation mode, so a quality setting does not apply to it (the same is true of GIF output). That lossless encode is also the slowest work the server does, so animated WebP renders run behind a dedicated gate of one encode at a time: while one is running, another animated WebP request waits up to 10 seconds for the slot and then sheds with a `503` rather than queueing behind a minutes-long encode, and still renders keep their own processing pool throughout. AVIF sources are decoded in builds that include AVIF decoding: the Docker image, the release binaries, and the Homebrew install (which ships the release binary). A source build includes it with `--features avif-decode`, which links dav1d (a system libdav1d 1.3 or newer, or a vendored build); without it, AVIF sources fail with `502` and an error naming the feature. Animated AVIF sources decode their first frame. HEIC/HEIF sources are decoded in every build by the pure-Rust oxideav stack (`oxideav-heif` + `oxideav-h265`): the primary image only, with grid and overlay compositions resolved, EXIF orientation honored, and 10-bit sources flattened to the pipeline's 8-bit working space, as AVIF sources are.

SVG sources are supported and detected by content, not by the declared content type. An SVG is sanitized before it is rendered: script elements, event handlers, and references to external resources are stripped, so a source SVG can only draw what it carries with it. The sanitized document is rasterized at the size the URL asks for (or its intrinsic size when the URL asks for none) and then runs through the normal pipeline. Output defaults to PNG for SVG sources.

## Operations

Operations run in the order they appear in the path. Requested outputs are capped at 16383 px per axis and 100 megapixels in total; a request past either bound fails with `400` before anything is allocated.

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
fit-in:<width>x<height>[:adaptive]
```

Scales the image proportionally so it fits inside the box. A `0` axis is unbounded. Flips are not allowed here.

With `:adaptive`, the box axes swap when the source orientation disagrees with the box's: a portrait photo fitted into a landscape box fits inside the swapped (portrait) box instead, so it is not squeezed by the tighter axis. When the orientations agree, adaptive changes nothing. A square source, a square box, and a box with a `0` or `orig` axis have no orientation to disagree with and never swap.

```bash
curl "http://localhost:3000/unsafe/fit-in:800x600/https://example.com/photo.jpg"
curl "http://localhost:3000/unsafe/fit-in:800x600:adaptive/https://example.com/photo.jpg"
```

### Pad to an exact size

```text
pad:<width>x<height>[:<fill>]
```

Fits the image inside the box (the fit-in rules, including `no_upscale`), centers it, and fills the rest of the canvas so the output is exactly the requested size. There is no adaptive form of `pad:`: the canvas is explicit, so it is never swapped. Both dimensions are required in pixels: a proportional (`0`) axis, `orig`, and flipped sizes are rejected, the same exact-box contract as `crop:`. The fill is a 6-digit hex color, or `blur` for ambient bars: a cover-scaled, heavily blurred copy of the image itself, the standard treatment for user-generated content in fixed-aspect slots. With no fill named, the request's `fill(...)` filter color is used, or black when there is none.

```bash
curl "http://localhost:3000/unsafe/pad:1200x630:blur/https://example.com/photo.jpg"
```

### Crop to an aspect ratio

```text
crop:<width>x<height>[:<align>]
```

Crops the image to the width:height aspect ratio, then resizes to exactly that width and height. Both dimensions are required in pixels: a proportional (`0`) axis, `orig`, and flipped sizes are rejected when the URL is parsed (`fit-in:` owns proportional boxes, and `filters:flip` flips). When the aspect ratios differ, `align` chooses which part survives:

| Align | Keeps |
| --- | --- |
| `center` | The middle. This is the default. |
| `top` | The top edge. |
| `bottom` | The bottom edge. |
| `left` | The left edge. |
| `right` | The right edge. |
| `top-left` | The top-left corner: the left edge when the width is cropped, the top edge when the height is. |
| `top-right` | The top-right corner. |
| `bottom-left` | The bottom-left corner. |
| `bottom-right` | The bottom-right corner. |
| `smart` | A window centered on the focal point: a detected face when a face model is configured (see [AI features](#ai-features)), otherwise the point of greatest local detail in the image. When faces are detected, the window shifts as needed to keep them whole: a position keeping every face fully inside wins, otherwise the position keeping the most face area whole (larger faces win), and a face larger than the window stays centered. The crop record behind `debug:focal` counts the faces detected and the faces the window keeps whole. |
| `face` | A window around the largest detected face, expanded by the `facepad` padding (below). Requires a face model; with no faces (or a detection failure) it falls back to the `smart` window. |

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

### Focal rectangle

```text
focal:<left>x<top>:<right>x<bottom>
```

Names the rectangle the next `crop:` in the URL must keep, in source pixels. The segment produces no output of its own; the crop consumes it. A `center` or `smart` crop positions its window on the rectangle: centered on the rectangle's center, and clamped so a rectangle that fits inside the window stays fully inside it. With a focal rectangle pending, `smart` skips face and detail detection entirely, because the caller has already said what matters. The directional alignments (`top`, `bottom`, `left`, `right`, and the four corners) and `face` are explicit instructions of their own and ignore the rectangle. `right` must be greater than `left`, and `bottom` greater than `top`. A rectangle that falls outside the source image fails with `400`. Coordinates address source pixels and are not scaled by `dpr`.

```bash
curl "http://localhost:3000/unsafe/focal:100x50:700x450/crop:800x600:smart/https://example.com/photo.jpg"
```

### Debug view

```text
debug:focal
```

Renders the smart-crop debug view: instead of the transform result, the response is a diagnostic image, the canvas the next `crop:` in the URL operates on, with the decision that crop actually makes drawn on it. A magenta crosshair marks the chosen focal point, and the focal rectangle outline (magenta) is drawn when one is in play: a caller `focal:` rectangle, or the face a `face` crop centers on. Detected face boxes that fed the decision are amber, and the crop window outline is blue. Line width and crosshair size scale with the image. With no crop in the URL, the view shows the focal analysis of the final image. Saliency alone drives it when no AI models are configured; any other `debug:` value fails with `400`.

Debug responses are never stored in, or served from, any cache tier, and are served with `Cache-Control: no-store`: the diagnostic is computed from the request's own processing, every time. Because the view runs the face and focal providers, a `debug:focal` request takes an AI-gate permit like the AI work it runs. An animated source, or a chain whose resolved format is a video format, fails with `400` rather than serving a half-applied debug view.

```bash
curl "http://localhost:3000/unsafe/debug:focal/crop:800x600:smart/https://example.com/photo.jpg"
```

### Video frame

```text
frame:poster
frame:<seconds>
frame:<percent>%
```

Selects one frame of a video source, which becomes the source image the rest of the URL operates on: the poster frame (`frame:poster`, the video's first frame), the frame at a timestamp in seconds (`frame:1.5`), or the frame at a percentage of the duration (`frame:50%`). The selection applies at decode, wherever the segment sits in the path, and the extracted frame then flows through crops (including face-safe smart crops), filters, studio mode, and the format negotiation like any other source image. When a URL carries several `frame:` segments, the last one wins.

The semantics are strict about the edges:

- Frame selection is video-only. A `frame:` segment on a still image or an animated GIF or WebP source fails with `502`, the way a bad source fails; animated sources keep their existing behavior whenever no `frame:` segment is present.
- A timestamp past the end of the video clamps to the last frame, as does `frame:100%`; nothing about the request is an error. A percentage needs the video's duration: a container that reports none fails the percentage form with `400`.
- A malformed selector (a sign, an exponent, a percentage over 100, a timestamp over 86400) fails with `400` and the `invalid_frame` code.

Frame extraction is an operator feature: it runs through the external decoder configured with `CRABCROP_VIDEO_DECODER`, and a server without one answers `501`. See [Video frames](#video-frames) for the decoder contract and the failure table.

```bash
curl "http://localhost:3000/unsafe/frame:50%/800x450/filters:format(webp)/https://example.com/clip.mp4"
```

### PDF page

```text
page:<n>
```

Selects one page of a PDF source, rendered as the source image the rest of the URL operates on: `page:1` is the document's first page. The selection applies at decode, wherever the segment sits in the path, and the rendered page then flows through crops, filters, studio mode, and the format negotiation like any other source image. When a URL carries several `page:` segments, the last one wins. One page per request: there are no ranges and no contact sheets.

The semantics are strict about the edges:

- Page selection is PDF-only. A `page:` segment on an image, a video, or any other source fails with `502`, the way a bad source fails.
- Pages are 1-based and the number is exact: `page:9` on a three-page document fails with `400`. It never clamps to the last page, the deliberate difference from `frame:` timestamps: a frame is a position inside a stream, while a page is a numbered object a caller picks by identity, so a wrong number is a wrong request, not a position past the end.
- A malformed selector (zero, a sign, letters, more than six digits) fails with `400` and the `invalid_page` code. Leading zeros are fine: `page:007` is `page:7`, and both spellings share one cache entry.
- A document that needs a password to open fails with `400`. A document that is merely permission-encrypted and opens without a password renders normally.

Page rendering is an operator feature: it runs through the external renderer configured with `CRABCROP_PDF_RENDERER`, and a server without one answers `501`. See [PDF pages](#pdf-pages) for the renderer contract and the failure table.

```bash
curl "http://localhost:3000/unsafe/page:2/800x1000/filters:format(webp)/https://example.com/report.pdf"
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
| `blur` | `blur(2.5)`, sigma, a positive number up to 512 | Gaussian blur. | `filters:blur(2.5)` |
| `brighten` | `brighten(40)`, integer from −255 to 255, added to every channel and clamped to 0–255 | Brighten (or darken, with a negative value). | `filters:brighten(40)` |
| `contrast` | `contrast(25)`, percent, −100 to 100; 0 is unchanged, −100 is flat gray, positive raises contrast | Adjust contrast. | `filters:contrast(25)` |
| `huerotate` | `huerotate(90)`, degrees, integer | Rotate the hue. | `filters:huerotate(90)` |
| `sharpen` | `sharpen(1.5)`, strength, a positive number | Unsharp-mask sharpening. | `filters:sharpen(1.5)` |
| `saturation` | `saturation(1.5)`, 0 to 255; 0 is grayscale, 1 is unchanged | Scale color saturation. | `filters:saturation(0.5)` |
| `rgb` | `rgb(10,-5,0)`, per-channel offsets, each −255 to 255 | Add offsets to the red, green, and blue channels. | `filters:rgb(10,-5,0)` |
| `curve` | `curve(rgb,0,0,255,255)`, a channel (`rgb`, `r`, `g`, or `b`), then 2 to 16 `x,y` control points, each 0–255 with x strictly increasing | Remap tones through the control points, linearly interpolated; inputs outside the points' x range clamp to the nearest end value. Repeat the filter to shape channels independently. | `filters:curve(r,0,64,255,255)` |
| `convolution` | `convolution(0,-1,0,-1,5,-1,0,-1,0,0)`, 9 (3×3) or 25 (5×5) kernel values in row-major order, each a plain decimal from −1024 to 1024, then a normalize flag (0 or 1) | Apply the kernel once over the image; edges replicate. With the flag on, the kernel is scaled by the sum of its entries first (an all-ones kernel averages); a zero sum applies the kernel as written. | `filters:convolution(0,-1,0,-1,5,-1,0,-1,0,0)` |
| `noise` | `noise(40)`, strength, an integer from 0 to 100, percent of full scale | Monochrome film grain: one noise value per pixel, uniform within ±strength percent of full scale, added equally to the red, green, and blue channels (alpha untouched). The field is deterministic, seeded from the image content and the amount, so the same request always returns the same bytes and caches soundly. `noise(0)` changes nothing. On animated sources it applies per frame, seeded from each frame's own content, so the frames grain independently. | `filters:noise(40)` |
| `equalize` | none | Histogram equalization, per channel. | `filters:equalize` |
| `rotate` | `rotate(45)`, any angle in degrees, decimals allowed; negatives allowed | Rotate the image. Multiples of 90 are exact quarter turns; other angles expand the canvas to fit, filling the uncovered corners with the `fill` color (white by default) or transparency when the image carries alpha. | `filters:rotate(-90)` |
| `trim` | `trim` or `trim(20)`, tolerance 0–255, default 10 | Crop away the border that matches the top-left corner color within the tolerance. | `filters:trim(20)` |
| `sepia` | none | The classic warm sepia tone. | `filters:sepia` |
| `duotone` | `duotone(1a1a40,ffd9a0)`, two colors, six hex digits each | Map shadows to the first color and highlights to the second, by luminance. | `filters:duotone(1a1a40,ffd9a0)` |
| `monochrome` | `monochrome(ff8000)`, six hex digits | Single-hue rendering: each pixel's luminance scaled by the color, so full luminance renders exactly that color. | `filters:monochrome(ff8000)` |
| `pixelate` | `pixelate(8)`, block size in pixels, at least 1 | Mosaic the image into square blocks, each painted with the average of its pixels. | `filters:pixelate(8)` |
| `flip` | `flip(h)` or `flip(v)` | Mirror the image horizontally or vertically, at this point in the chain. | `filters:flip(h)` |
| `blur_region` | `blur_region(x,y,w,h)`, pixels | Gaussian-blur one rectangle (license plates, screens, documents). Coordinates address the image at this point in the chain; blur strength scales with the region. A region that overflows the image is an error, not a partial redaction. | `filters:blur_region(120,40,180,60)` |
| `pixelate_region` | `pixelate_region(x,y,w,h)`, pixels | Pixelate one rectangle, with the same bounds rule as `blur_region`. Block size scales with the region. | `filters:pixelate_region(120,40,180,60)` |
| `remove_region` | `remove_region(x,y,w,h)`, pixels | Erase one rectangle and inpaint it from the surrounding pixels (AI, LaMa-family model). Same bounds rule as `blur_region`. The fill is fabricated by the model: plausible on smooth surroundings, visibly wrong on structured content. See [AI features](#ai-features). | `filters:remove_region(120,40,180,60)` |
| `radius` | `radius(24)`, pixels; or four values `radius(24,24,0,0)` in CSS corner order (top-left, top-right, bottom-right, bottom-left) | Round the corners, antialiased. Radii larger than the frame shrink proportionally so corners never overlap. The output format must carry alpha: `png`, `webp`, `jxl`, or `avif`. | `filters:radius(24):format(png)` |
| `border` | `border(4,000000)`, width from 0 to 16383 pixels, color as six hex digits | Paint an opaque band of the color inside the frame. A width of 0 is a no-op. Dimensions do not change. Works on any format. | `filters:border(4,000000)` |
| `mask` | `mask(circle)` or `mask(ellipse)` | Keep only the inscribed shape (the circle's diameter is the smaller dimension); the rest becomes transparent. The output format must carry alpha. | `filters:mask(circle):format(png)` |

The `curve` filter is the fine tone control the additive filters only approximate. Each control point says "an input of x becomes y"; between points the mapping is a straight line, and inputs below the first point's x take its y (likewise above the last point's), so a curve need not span the whole range. `filters:curve(rgb,0,20,128,160,255,255)` lifts the shadows and midtones of every channel at once, and a second `curve` with `r`, `g`, or `b` then shapes one channel at a time. The identity curve `curve(rgb,0,0,255,255)` changes nothing.

The `convolution` filter covers the sharpen, blur, and edge styles the built-ins do not: you supply the kernel yourself. Values are written in row-major order (the first value is the window's top-left tap, the last its bottom-right), each a plain decimal such as `0.0625` (no exponent forms), from −1024 to 1024. The trailing flag turns normalization on or off: with `1`, the kernel is scaled by the sum of its entries before it applies, so the all-ones kernel `convolution(1,1,1,1,1,1,1,1,1,1)` is a box average; when that sum is zero (edge kernels, typically), the kernel applies exactly as written, flag or not. Taps that fall outside the frame read the nearest edge pixel, the same edge rule as `blur` and `sharpen`. `filters:convolution(0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625,0)` is a 3×3 gaussian blur.

```bash
curl "http://localhost:3000/unsafe/600x400/filters:sharpen(1.2):saturation(1.1)/https://example.com/photo.jpg"
```

**Encoding directives** leave the pixels alone and steer the encoder:

| Directive | Parameters | Description | Example |
| --- | --- | --- | --- |
| `format` | `format(webp)`, `jpeg` (or `jpg`), `png`, `webp`, `jxl`, `avif`, `gif`, `mp4`, `webm` | Output format. Overrides `Accept` negotiation. The video formats are explicit-only (never negotiated) and need an animated source plus a configured encoder; see [Video output](#video-output). | `filters:format(avif)` |
| `quality` | `quality(80)`, 1 to 100, or `quality(auto)` | Encoding quality for JPEG, JPEG XL, AVIF, and WebP. Default 80. | `filters:quality(auto)` |
| `max_bytes` | `max_bytes(50000)`, byte budget, at least 1 | Re-encode at decreasing quality until the output fits the budget. | `filters:max_bytes(50000)` |
| `fill` | `fill(ff0000)`, six hex digits, RGB | Background color used to flatten transparency when encoding to JPEG. | `filters:fill(ffffff)` |
| `no_upscale` | none | Cap every resize in the request at the source dimensions. | `filters:no_upscale` |
| `dpr` | `dpr(2)`, one of `1`, `1.5`, `2`, `3` | Scale size and crop targets by the device pixel ratio, so one URL serves standard and retina displays. | `filters:dpr(2)` |

Notes on the directives:

- `quality(auto)` picks a quality per image, from 50 for flat images to 95 for detailed ones, based on content analysis. For JPEG and JPEG XL the pick is then verified: crabcrop encodes, decodes the result, and scores it against the processed source with SSIMULACRA2, adjusting quality (at most 4 encodes) until the score lands at the target, 75 by default and configurable with `CRABCROP_SSIM_TARGET`. The achieved score is reported in meta responses as `ssimulacra2`. Verification happens on cache misses only, but it is real work: scoring a full-size image takes about a second. When crabcrop picks the quality, for `quality(auto)`, or when `max_bytes` steps quality down, the response carries an `X-Crabcrop-Quality` header with the value used.
- `max_bytes` can only steer formats with a quality setting: JPEG, JPEG XL, AVIF, and WebP. If quality reaches its floor of 1 and the output is still over budget, you get the smallest encode rather than an error. PNG and GIF ignore the budget.
- WebP encodes lossy at the requested quality (default 80); `quality(100)` selects the lossless encode. `quality` has no effect on PNG or GIF output.
- JPEG has no alpha channel. With `fill`, transparent pixels are composited over the fill color; without it, the stored RGB values are used as-is.
- `no_upscale` applies to the whole request wherever it appears. A plain resize clamps each axis at the source dimension: a `300x200` resize of a 200×150 source returns 200×150, and a `300x100` resize returns 200×100 (the axis that shrinks is honored as written). Fit-in and crop targets shrink proportionally instead, since their dimensions already carry the image's proportions.
- `dpr` also applies request-wide, and it scales targets *before* the `no_upscale` cap sees them, so `dpr` never invents pixels past the source when both are set. Targets round to whole pixels. `crop-rect` and `focal` coordinates address source pixels and are not scaled. When the URL carries no `dpr`, the server honors the `DPR` client hint header (same set of values; an invalid hint is ignored). An explicit `dpr` in the URL always beats the header.

#### Client hints

When a URL sizes nothing itself (no size, crop, fit-in, pad, or crop-rect operation), the server honors the width client hints so responsive pages do not have to mint a URL per width. It reads, in order, `Sec-CH-Width`, `Width`, `Sec-CH-Viewport-Width`, and `Viewport-Width`, and resizes to that width as if the URL had carried the width-only size op (the `DPR` hint still multiplies on top). Hinted widths are quantized up to a multiple of `CRABCROP_WIDTH_STEP` (default 100), so nearby viewports share one cache entry, and the quantized width joins the cache key. Image responses advertise `Accept-CH` for these hints and vary on them.

A request with `Save-Data: on` gets the default encode quality stepped down one notch, from 80 to 70, on the formats that carry a quality (JPEG, JPEG XL, AVIF, WebP); the stepped value is reported in `X-Crabcrop-Quality`. URLs that set `quality(...)` themselves are unaffected, and stepped responses are cached separately.

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

**Overlay** generalizes the watermark with blend modes and relative sizing:

```text
filters:overlay(<url>,<x>,<y>,<alpha>,<blend>)
filters:overlay(<url>,<x>,<y>,<alpha>,<blend>,<width>)
```

| Parameter | Description |
| --- | --- |
| `url`, `x`, `y`, `alpha` | Same as `watermark`: percent-encoded URL (may contain commas), position with negative values anchoring from the right and bottom edges, opacity 0 to 100. |
| `blend` | How the overlay's colors mix with the image below: `normal` (the watermark behavior), `multiply`, `screen`, or `overlay`, the CSS formulas per channel. The mix is composited at the effective alpha (the overlay's own alpha times `alpha`), and the base image's alpha channel is preserved. |
| `width` | Optional. Scale the overlay to this percentage of the base image's width (1 to 400), preserving its aspect ratio, before compositing. |

Overlay images are fetched through the same prefetch, loader, and failure semantics as watermarks: a failed overlay fails the request.

```bash
curl "http://localhost:3000/unsafe/filters:overlay(https%3A%2F%2Fcdn.example.com%2Ftexture.png,0,0,60,multiply,100)/https://example.com/photo.jpg"
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

The text is drawn when the filter runs, at whatever resolution the image has at that point in the path, so put `text` after your resize to keep it sharp. The font comes from `CRABCROP_FONT_PATH` on the server; with no font configured, `text` requests fail with `501`.

```bash
curl "http://localhost:3000/unsafe/800x600/filters:text(Sale%20ends%20soon,32,ffffff,bottom-right)/https://example.com/photo.jpg"
```

**AI filters** need a model configured on the server. See [AI features](#ai-features).

| Filter | Parameters | Description | Example |
| --- | --- | --- | --- |
| `removebg` | none | Remove the background, keeping the subject over transparency. Output format must carry alpha: `png`, `webp`, `jxl`, or `avif`. | `filters:removebg:format(png)` |
| `studio` | `studio(product,...)` or `studio(scene,...)` | Studio mode: `product` composites the subject onto a backdrop with a synthesized contact shadow, scaled and centered to a pad margin; `scene` keeps the photograph's scene, reframing it on the detected subject and correcting light and color. See [Studio mode](#studio-mode). Requires the removebg model. | `filters:studio(product,bg:white)` |
| `keep_profile` | none | Keep the source ICC profile instead of converting to sRGB; the profile is embedded in JPEG and PNG outputs. Other output formats convert to sRGB (pixels labeled as nothing would be wrong colors). | `filters:keep_profile` |
| `blurfaces` | none | Blur every detected face (privacy redaction). Requires a face model. | `filters:blurfaces` |
| `pixelatefaces` | none | Pixelate every detected face (privacy redaction). Requires a face model. | `filters:pixelatefaces` |
| `blurclass` | `blurclass(person)` | Blur every detected object of the class (privacy redaction). Repeat for more classes. Requires a detection model. With a segmentation model, the blur follows the object's silhouette instead of its box. | `filters:blurclass(person)` |
| `pixelateclass` | `pixelateclass(car)` | Pixelate every detected object of the class (privacy redaction). Repeat for more classes. Requires a detection model. With a segmentation model, the pixelation follows the object's silhouette instead of its box. | `filters:pixelateclass(car)` |
| `recolorclass` | `recolorclass(car,ff0000)` | Paint every detected object of the class a solid color, alpha preserved. Repeat for more classes. Requires a detection model. With a segmentation model, the paint follows the object's silhouette instead of its box. | `filters:recolorclass(car,ff0000)` |
| `facepad` | `facepad(0.5)`, 0 to 3 | Padding for `crop:...:face`: that fraction of the face's size added on every side of the crop window. Default 0.5. | `crop:200x200:face/filters:facepad(1)` |
| `upscale` | `upscale(2)` or `upscale(4)` | AI super-resolution by that factor. Output is capped at 4096 px per dimension. One model run takes at most 262,144 input pixels (the image itself, or the intermediate between the composed runs of an `upscale(4)`); a larger input is a `400`, so resize the source first. | `filters:upscale(2)` |

```bash
curl "http://localhost:3000/unsafe/filters:removebg:format(png)/https://example.com/product.jpg"
```

#### Studio mode

`studio(product,...)` names an outcome instead of a mechanism: a listing-ready product shot. One filter expands into a fixed pipeline, run at the filter's position in the chain:

1. **Mask**: the subject is cut out with the same segmentation model `removebg` uses (without one configured, `studio` fails with `501`, the same way `removebg` does).
2. **Defringe**: edge colors are decontaminated against the local background, so no halo of the old backdrop survives the cut.
3. **Backdrop and shadow**: the canvas (the image box in effect at that point in the chain) is filled with the resolved backdrop, and a contact shadow is synthesized from the subject's alpha, blurred, darkened, and offset down, beneath the subject.
4. **Pad**: the subject is scaled and centered so the pad margin holds on every side.

```bash
# The defaults: brand backdrop (white when none is configured), soft shadow, pad 10
curl "http://localhost:3000/unsafe/filters:studio(product)/https://example.com/shoe.jpg"
# Every argument named
curl "http://localhost:3000/unsafe/filters:studio(product,bg:f5f0e8,shadow:strong,pad:25)/https://example.com/shoe.jpg"
```

| Argument | Values | Default |
| --- | --- | --- |
| `bg` | `white`, a six-digit hex color, `brand`, or `none` | the brand color when configured, else `white` |
| `shadow` | `soft`, `strong`, or `none` | `soft` |
| `pad` | 0 to 40, percent margin per side | `10` |

The fine print:

- `bg:brand` paints the operator's brand color, set with `CRABCROP_BRAND_COLOR`. With none configured it resolves to white. `bg:none` keeps the subject's transparency and skips the shadow.
- If the mask finds no subject, the fallback serves the original image centered and padded on the resolved backdrop (no shadow), rather than an empty canvas.
- The result is a normal image in the chain, valid in every output format (JPEG included): unlike `removebg`, nothing transparent survives a backdrop. Later operations apply on top, so `filters:studio(product)/1200x1200` sizes the composite, and a `pad:` operation after `studio` pads the composite like any image. The encode-time `fill` directive only matters when transparency remains, which after `studio` means `bg:none` chains.
- Meta responses report the run in a `studio` field: the resolved backdrop and where it came from (`argument`, `brand`, or `default`), the shadow and pad applied, the steps that ran, any fallbacks taken (the brand resolution, the no-subject fallback), and the mask confidence when the provider exposes one. See [Meta responses](#meta-responses).

`studio(scene)` is the second mode: it keeps the photograph's scene instead of cutting the subject out. The expansion is a smart crop on the detected subject plus a per-image light and color correction, run at the filter's position in the chain:

1. **Mask**: the same segmentation model product mode runs (without one configured, `studio(scene)` fails with `501`, the same way).
2. **Crop**: the photograph is reframed on the subject. The subject is picked from the mask's connected regions at the subject threshold: the largest wins, ties break toward the frame center, then reading order. The frame is cropped to the smallest window with the frame's own aspect ratio that holds the subject with its pad margin, then resized back to the canvas in effect, so the output keeps the size the chain expects and later operations compose unchanged.
3. **Correct**: a light and color correction driven by the reframed image's own histogram: a per-channel levels stretch between the 0.5th and 99.5th percentiles (a channel whose span is under 16 values is too flat to stretch and is left alone), then a saturation factor that steers the image's mean saturation toward 0.30, capped at 1.35 and never below 1, so muted photographs gain color and vivid ones are never desaturated.

```bash
# Reframe on the subject and correct, default pad
curl "http://localhost:3000/unsafe/filters:studio(scene)/https://example.com/shoe.jpg"
# A wider margin around the subject
curl "http://localhost:3000/unsafe/filters:studio(scene,pad:25)/https://example.com/shoe.jpg"
```

Scene takes one argument: `pad`, 0 to 40, default `10`, the margin the subject keeps in the reframed window. The product arguments (`bg`, `shadow`) are bad arguments under scene. If the mask finds no subject (or only a speck not worth believing), the fallback skips the crop and serves the original framing, corrected. Meta reports scene runs in the same `studio` field, with `mode: "scene"` and a `scene` object: the winning subject's box, how many candidate subjects the mask carried, the crop window applied, the per-channel levels, and the saturation factor, alongside the steps and fallbacks.

## Presets

A preset names an operation string on the server, so URLs stay short and the transformation menu stays operator-controlled. Define presets in a file, one per line:

```text
# /etc/crabcrop/presets.txt
og-card = crop:1200x630:smart/filters:format(webp):quality(80)
thumb = 300x200
```

Point `CRABCROP_PRESETS_FILE` at it, and requests can use `preset:<name>` anywhere an operation fits:

```bash
curl "http://localhost:3000/<signature>/preset:og-card/https://example.com/photo.jpg"
```

The signature covers the preset reference; the server expands it deterministically after verification, and cache entries key on the expanded operations. Treat a preset as a stable template: **never edit a definition in place**. Roll a change out under a new name (`thumb2`) and update your templates to reference it. Editing in place changes the bytes behind URLs already signed and shared, and image responses travel with `Cache-Control: immutable` for a year, so browsers and CDNs holding the old response keep serving it, beyond the reach of any purge. (This server's own caches are not the problem: they key on the expanded form, so they would pick the edit up. Downstream copies are the ones stranded.) Purging accepts the preset URL itself, exactly as an operator holds it; the server expands it before matching entries, and `/purge-group` clears a preset's whole output set at once. An unknown preset name is a `400`. Preset definitions are validated at startup: malformed entries refuse the boot, and presets cannot contain a source segment, an `exp:`, or another preset.

With `CRABCROP_ONLY_PRESETS=1`, free-form operations are rejected with `400`: requests may use preset references, `exp:` and `maxage:` metadata, and bare sources only. The mode does not change signature rules. Combined with `CRABCROP_ALLOW_UNSAFE=1` it makes a public deployment where browsers and CMSs can build URLs from preset names alone, with no secret to hold and no way to invent a transform.

## Source roots

A source root names an origin on the server, so a URL can carry a short source instead of spelling out the full source URL: the origin layout stays off the URL, and URLs get shorter. Define roots in a file, one per line:

```text
# /etc/crabcrop/roots.txt
photos = https://cdn.example.com/photos
media = https://assets.example.com
```

Point `CRABCROP_SOURCE_ROOTS_FILE` at it, and a request can use `root:<name>/<path>` as its source:

```bash
curl "http://localhost:3000/<signature>/300x200/root:photos/cat.jpg"
```

That request fetches `https://cdn.example.com/photos/cat.jpg`. The signature covers the short form exactly as written; the server expands it after verification and before cache keying, so the short spelling and the full URL it expands to are one render: they share a cache entry, and purging either spelling evicts it. Root definitions are validated at startup: names are letters, digits, `_`, or `-`, a duplicate name refuses the boot, and a base must be an absolute http(s) URL with no userinfo, query, or fragment.

The safety rules:

- The source allowlist (`CRABCROP_ALLOWED_SOURCES`) and every other loader check apply to the expanded URL, exactly as if the client had spelled it out. A root never widens what the server will fetch.
- The path must stay under the root. Dot segments are normalized, and a path that climbs above the root, plainly or percent-encoded, is a `400` (`invalid_root`).
- An unknown root name is a `400` (`unknown_root`), including on a server with no roots configured.
- Root base URLs are operator configuration and stay private: `explain` reports the source as written plus the root's name, and never prints a base URL.

## Client libraries

The URL grammar and signing scheme are specified in [`spec/url-signing.md`](spec/url-signing.md), with a canonical set of test vectors in [`spec/test-vectors.json`](spec/test-vectors.json) that every implementation runs as its test suite. Builders on top of the spec:

- **Rust**: the `crabcrop-url` crate in this workspace (`crates/crabcrop-url`) with `sign`, `verify`, `build_url`, `build_srcset`, and the purge helpers `purge_signature` and `purge_group_signature`. The server verifies with this same crate, so client and server cannot drift.
- **TypeScript**: [`sdks/typescript`](sdks/typescript), `@crabcrop/url`, with `sign`, `verifyUrlSignature`, `buildURL`, `buildSrcSet`, `purgeSignature`, and `purgeGroupSignature`, plus typed operation builders (`ops`, `filter`) that validate against the grammar and compile to the same canonical strings. Standard library only; tests run on `node --test`.
- **Python**: [`sdks/python`](sdks/python), `crabcrop_url`, with `sign`, `verify`, `build_url`, `build_srcset`, `purge_signature`, and `purge_group_signature`. Standard library only; tests run on `python3 -m unittest`.

All three take an optional expiry (emitted as a leading `exp:` segment) and a null secret for the `unsafe` form used by unsafe and presets-only deployments.

Framework integrations in [`frameworks/`](frameworks) sit on the TypeScript builder: a [Next.js](frameworks/nextjs) custom loader for `next/image`, a [Nuxt](frameworks/nuxt) provider for `@nuxt/image`, and a [Hydrogen](frameworks/hydrogen) loader for external (non-Shopify-CDN) images. All three recommend presets as the configuration, so client-side URL building never needs the secret.

The [playground](playground) is a static page (no build step, no server state) for building, signing, previewing, and debugging URLs in the browser: the secret never leaves the page, signatures verify locally, and pasted URLs are explained segment by segment with errors highlighted.

For coding agents, [`sdks/mcp`](sdks/mcp) is a first-party MCP server (stdio, no dependencies) exposing `sign_url`, `explain_url`, `fetch_meta`, and `purge_result` over the same SDK and HTTP surface, configured with `CRABCROP_BASE` / `CRABCROP_SECRET`. Client setup snippets are in its README.

## Format negotiation

When the URL names no `format(...)`, crabcrop picks the output format from the request's `Accept` header. Among the negotiable formats (AVIF, JPEG XL, WebP), the one with the highest q-value wins; a missing q counts as 1. Ties break by server preference, in this order:

1. `image/avif` → AVIF
2. `image/jxl` → JPEG XL
3. `image/webp` → WebP

So `Accept: image/avif;q=0.5, image/webp` produces WebP: the client's stronger preference beats the server's. If none of the three is offered, the source image's format is kept (JPEG if the format cannot be detected). A format refused with `q=0` is never picked, and wildcards such as `*/*` do not count as an offer. An explicit `format(...)` in the URL always wins over negotiation. Animated GIF and WebP sources are the one exception: they never negotiate to JPEG XL, which would drop their frames; a GIF source keeps GIF instead, and a WebP source keeps WebP. Video is the other exception: `mp4` and `webm` are never negotiated at all, and only ever come from an explicit `format(...)` (see [Video output](#video-output)). Responses include a `Vary` covering `Accept` and the client hints so downstream caches key this correctly (the hints likewise change sizing and quality; see [Client hints](#client-hints)).

## Video output

An animated GIF or WebP can also be served as real video: H.264 MP4 or VP9 WebM, the gifv pattern that plays inline everywhere and weighs a fraction of the GIF. Ask with an explicit format filter:

```
/<signature>/filters:format(mp4)/https://example.com/animation.gif
```

Crabcrop ships no codec. The encode runs in an external program you configure, `CRABCROP_VIDEO_ENCODER`: the path to an encoder executable on the server, with FFmpeg as the reference. The path is validated at startup (it must exist and be executable, or the server refuses to boot). With the variable unset, video is off and a video URL answers `501`, exactly like an AI filter with no model. And there is no silent fallback: a failed encode is an error response, never the GIF.

### The encoder contract

Crabcrop runs the program directly, with no shell and an argv it constructs; only the program path is yours. Frames go in on stdin as raw RGBA and the video comes out on stdout:

```
<encoder> -nostdin -v error -f rawvideo -pix_fmt rgba -s <W>x<H> -r <fps> -i pipe:0 -an -vf format=yuv420p <codec args> pipe:1
```

For MP4 the codec args are `-c:v libx264 -preset veryfast -crf 23 -movflags frag_keyframe+empty_moov+default_base_moof -f mp4`. The fragmented-MP4 flags are what make MP4 writable to a pipe at all; a plain MP4 needs a seekable file to finish its index. For WebM they are `-c:v libvpx-vp9 -b:v 0 -crf 30 -f webm`. A wrapper script that passes its arguments through to a real encoder (one ending in `exec ffmpeg "$@"`, say) satisfies the contract, which is how you bring an encoder build of your own. Because the codec lives outside crabcrop, so does codec licensing: H.264 patent pools and friends are the operator's concern, and that separation is the point. Output bytes vary with the encoder build, so byte-for-byte reproducibility across deployments is not promised; the URL grammar and the errors are.

Details an encoder setup has to match:

- Frames are composited over black and padded up to even dimensions (H.264 and VP9 need them) by replicating the edge pixels. The response describes the padded dimensions.
- Frame timing: the pipe contract carries one constant rate, so crabcrop picks it from the fastest frame, `ceil(1000 / shortest delay)` clamped to 1-100 fps (10 fps when no frame carries a delay), and repeats each frame `round(delay × fps / 1000)` times, at least once. Total duration is preserved to within one frame period per source frame. A source whose delays total more than 600 s is refused with a `400`.
- At most two encodes run at a time, each under a hard wall-clock timeout (`CRABCROP_VIDEO_TIMEOUT_SECS`, default 30 s) after which the encoder is killed, and a hard output cap (`CRABCROP_VIDEO_MAX_MB`, default 64 MB) that aborts the encode the moment it is crossed. The encoder runs with an empty environment; nothing is inherited from the server's.

### Errors

| Situation | Answer |
| --- | --- |
| No encoder configured | `501`: `video output is not configured on this server` |
| Still source (a still image, or a one-frame GIF or WebP) | `400`: the source is a still image, and video needs an animated source |
| Animation over the 600 s cap | `400` |
| Encoder exits nonzero, produces nothing, exceeds the cap, or times out | `502`: `video encoding failed` (the cause, including the encoder's stderr excerpt, is in the server log) |
| Encode queue full (both slots busy past a 10 s wait) | `503` |

Video results cache like any other response, under the video content type (`video/mp4` or `video/webm`), and every operation in the URL (resize, crop, filters) applies per frame before the encode. A meta request (`Accept: application/json`) on a video URL answers `400`: meta describes images, and a video result is not one.

## Video frames

A video the server can fetch is also an image source: a `frame:` segment in the URL selects one frame (see [Video frame](#video-frame) in the operations), and from there the request is an ordinary image request. The frame is the poster's first frame, the frame at a timestamp, or the frame at a share of the duration; out-of-range selections clamp to the last frame. Extraction happens once per unique URL, on a cache miss, and the result caches (and purges) like any other render, under the same key rules as every spelling of the URL.

Crabcrop ships no video decoder either. Extraction runs in an external program you configure, `CRABCROP_VIDEO_DECODER`: the path to a decoder executable, with FFmpeg as the reference. The path is validated at startup (it must exist and be executable, or the server refuses to boot). With the variable unset, frame selection is off and a `frame:` URL answers `501`. The contract between crabcrop and the program is fixed, and crabcrop builds every argument itself; the program never sees a shell:

- **Probe**: argv is `ffmpeg -nostdin -v info -i pipe:0`, with the source bytes on stdin and no output target. The program reports the input on stderr (duration, stream list); crabcrop reads the duration, the first video stream's dimensions, and its frame rate from that report. A source whose report holds no video stream (a non-video, an audio-only file, a corrupt container) is a bad source.
- **Extract**: argv is `ffmpeg -nostdin -v info -i pipe:0 -ss <seconds> -frames:v 1 -f image2pipe -vcodec png pipe:1`, again with the source bytes on stdin. One PNG frame comes back on stdout and becomes the source image, so every decode and output budget applies to it unchanged.

The bounds mirror the encoder's, because the work is the same class:

- At most two extractions run at a time behind the video gate (shared with encodes), on top of the process gate; a request that waits more than 10 s for a slot sheds with `503`.
- Each run (probe and extraction alike) is bounded by the same wall-clock timeout as encodes (`CRABCROP_VIDEO_TIMEOUT_SECS`, default 30 s), after which the program's whole process group is killed; the program runs with an empty environment, and its output is capped (a frame whose PNG passes 256 MiB, the decode allocation budget, is refused).
- The source itself flows through the ordinary loader first, with the same allowlist, size cap, and redirect rules as any source. Only a URL carrying a `frame:` segment may fetch a `video/*` Content-Type; every other request keeps the image-only rule.

### Errors

| Situation | Answer |
| --- | --- |
| No decoder configured | `501`: `frame extraction is not configured on this server` |
| Source is an image (still or animated) | `502`: frame selection needs a video source |
| Source has no video stream, or is not a readable video | `502`: the bad-source failure every undecodable source gets |
| Percentage selection on a video that reports no duration | `400` |
| Malformed `frame:` selector | `400` with the `invalid_frame` code, before any fetch |
| Decoder exits nonzero, produces no frame, exceeds the cap, or times out | `502`: `video frame extraction failed` (the cause, including the decoder's stderr excerpt, is in the server log) |
| Video queue full (both slots busy past a 10 s wait) | `503` |

A frame render's [meta response](#meta-responses) carries a `video` object: the source's `width` and `height`, its `duration` in seconds when the container reports one, and the `frame_timestamp` the frame was selected at, after clamping. The explain endpoint labels a video source's kind as `video`; its dimensions stay unknown there, because explain never runs the decoder.

## PDF pages

A PDF the server can fetch is also an image source: a `page:` segment in the URL selects one page (see [PDF page](#pdf-page) in the operations), and from there the request is an ordinary image request. The page renders at a fixed scale of 2 (a 144 dpi raster), clamped down for very large pages so neither raster axis passes the 16384 px decode cap; the rendered PNG re-enters the pipeline, so every decode budget, filter, and output rule applies to it unchanged. Rendering happens once per unique URL, on a cache miss, and the result caches (and purges) like any other render, under the same key rules as every spelling of the URL: `page:7` and `page:07` are one entry.

Crabcrop ships no PDF renderer. Rendering runs in an external program you configure, `CRABCROP_PDF_RENDERER`: the path to a renderer executable, with Poppler's `pdftocairo` as the reference. The renderer does not come alone: crabcrop derives the `pdfinfo` program next to it (same directory, same extension) and uses it to probe documents. Both paths are validated at startup (they must exist and be executable, or the server refuses to boot). With the variable unset, page selection is off and a `page:` URL answers `501`. The contract between crabcrop and the programs is fixed, and crabcrop builds every argument itself; the programs never see a shell, a file path, or the source URL:

- **Probe**: argv is `pdfinfo -f <page> -l <page> -`, with the PDF bytes on stdin. Crabcrop reads the page count, the encryption state, and the selected page's size in points from the report. When the probe refuses the range itself, which it does past the end of the document, a whole-document probe confirms the count for the refusal.
- **Render**: argv is `pdftocairo -png -singlefile -f <page> -l <page> -r <dpi> - -`, again with the PDF bytes on stdin. One PNG page comes back on stdout and becomes the source image.

Because the tools receive only bytes over stdin, nothing inside a document has anything to resolve: external references, embedded files, and document actions or scripts have no target and no effect. The source itself flows through the ordinary loader first, with the same allowlist and redirect rules as any source and the same 10 MiB fetch cap; only a URL carrying a `page:` segment may fetch an `application/pdf` Content-Type. Documents over 1000 pages are refused.

The bounds mirror the frame extractor's, because the work is the same class:

- Page work takes the video gate (at most two runs at a time, shared with encodes and frame extractions), on top of the process gate; a request that waits more than 10 s for a slot sheds with `503`.
- Each run is bounded by `CRABCROP_PDF_TIMEOUT_SECS` (default 30 s), after which the program's whole process group is killed; the program runs with an empty environment, and its output is capped (a page whose PNG passes 256 MiB, the decode allocation budget, is refused).

### Errors

| Situation | Answer |
| --- | --- |
| No renderer configured | `501`: `page rendering is not configured on this server` |
| Source is an image, a video, or anything else that is not a PDF | `502`: page selection needs a PDF source |
| Source is not a readable PDF (junk bytes, a corrupt or truncated document) | `502`: the bad-source failure every undecodable source gets |
| Document over the 1000-page cap | `502`, naming the count and the cap |
| Page past the end of the document | `400`, naming the page and the document's page count |
| Document needs a password to open | `400`: the document is password-protected |
| Malformed `page:` selector | `400` with the `invalid_page` code, before any fetch |
| Renderer or probe exits nonzero, produces no page, exceeds the cap, or times out | `502`: `PDF rendering failed` (the cause, including the tool's stderr excerpt, is in the server log) |
| Video queue full (both slots busy past a 10 s wait) | `503` |

A page render's [meta response](#meta-responses) carries a `pdf` object: the document's `pages` count, the `page` rendered, and the page's `page_width` and `page_height` in points. The explain endpoint labels a PDF source's kind as `pdf`; its dimensions stay unknown there, because explain never runs the renderer.

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
| `ssimulacra2` | The verified perceptual score of the output, present when `quality(auto)` verified a JPEG or JPEG XL encode. 100 is identical. |
| `studio` | Present when a `studio(...)` filter ran, with a `mode` (`product` or `scene`). Product runs report the resolved backdrop (`bg`, spelled `#rrggbb`, or `none` for transparency) and its source (`bg_from`: `argument`, `brand`, or `default`), the `shadow` and `pad` applied. Scene runs report the `pad` and a `scene` object: the winning `subject` box (or `null`), the candidate `subjects` count, the `crop` window applied, the per-channel `levels` stretch, and the `saturation` factor. Both report the expansion `steps` that ran, the `fallbacks` taken, and `mask_confidence` when the mask provider exposes one. See [Studio mode](#studio-mode). |
| `video` | Present when the source was a video and a `frame:` segment selected the frame: the video stream's `width` and `height`, its `duration` in seconds when the container reports one, and the `frame_timestamp` the frame was selected at, after clamping. See [Video frames](#video-frames). |
| `pdf` | Present when the source was a PDF and a `page:` segment selected the page: the document's `pages` count, the `page` rendered, and the page size as `page_width` and `page_height` in points. See [PDF pages](#pdf-pages). |
| `provenance` | The C2PA facts for this render: whether output signing is configured on this server (`signing_configured`), and a summary of the manifest the source carried and the manifest the output carries (`source` and `output`, each `null` when the asset carries none, else `{state, signer, manifests, ingredients}` with `state` one of `valid`, `trusted`, `invalid`). See [Provenance (C2PA)](#provenance-c2pa). |

Meta responses are served with `Cache-Control: private, no-store`: a URL has two representations (image and JSON), and intermediaries must never pin one in place of the other.

Add the `placeholders` parameter to the media type to also get placeholder data, computed from the final processed pixels. The parameter opts in when it is bare (`; placeholders`), set to `1`, or set to `true` (case-insensitive); any other value, including `0` and `false`, does not. The `detections` parameter below follows the same rule. On the CLI, `crabcrop meta --placeholders` and `crabcrop meta --detections` set them for you:

```bash
curl -H "Accept: application/json; placeholders=1" "http://localhost:3000/unsafe/300x200/https://example.com/photo.jpg"
```

This adds four fields: `dominant_color`, the image's average color as `#rrggbb`; `placeholder`, a 16 px-wide blurred JPEG preview as a base64 data URI; `palette`, the five most prominent colors as `#rrggbb`, most prominent first; and `thumbhash`, a ThumbHash string that ThumbHash client libraries decode into an aspect-correct blurred preview. Handy for rendering a preview while the real image loads. When a CLIP model is configured (see [AI features](#ai-features)), meta responses also carry `tags` (labels with zero-shot scores) and `alt_text` composed from the best labels, whether or not placeholders were asked for. With a detection model configured, adding `detections=1` to the same `Accept` entry adds `detections`: the objects found in the processed image as `{label, score, x, y, w, h, detector}` in pixels, best first per detector; `detector` names the provider that produced the box (`builtin`, or a declared detector's name, see [Declared detectors](#declared-detectors)). With moderation configured (see [AI features](#ai-features)), adding `moderation=1` to the same `Accept` entry adds `moderation`: the configured model's per-category verdict scores as `{model, scores: [{label, score}]}`, or, when only a labels file is configured, zero-shot `signals` scored by CLIP and marked `review_only`, because signals are leads for review, never verdicts. When the opt-in cannot be answered (nothing configured, a busy inference gate, a failed run), the field reports `{"available": false, "reason": ...}` and the meta request still succeeds. Verdicts annotate meta only: no score ever changes what the image URL serves. With a CLIP model configured, adding `embeddings=1` to the same `Accept` entry adds the tagger's image embedding: `embedding`, the vector packed as float16 little-endian and base64-encoded, with `embedding_model`, `embedding_dim`, `embedding_encoding` (`f16le-base64`), and `embedding_normalized: true` alongside, so a stored vector stays interpretable after the encoder is swapped. When that opt-in cannot be answered, `embedding` is `null` with an `embedding_reason`, and the meta request still succeeds. With `CRABCROP_AESTHETIC_HEAD` set (see [AI features](#ai-features)), meta responses also carry `aesthetic`: the head's score of that same embedding as `{head, model, score}`, or a `null` score with a reason when the head and the encoder disagree on dimension (the field then names `expected` and `got`) or no vector could be produced. The aesthetic score is descriptive only: it never gates, ranks, or alters what a URL serves. With OCR configured (see [AI features](#ai-features)), adding `ocr=1` to the same `Accept` entry adds `ocr`: the text read from the processed image as `{model, language, text: [{text, score, x, y, w, h}], text_content}`, the lines in reading order with their boxes in the image's pixels and a confidence score each, plus the lines joined into one `text_content`. `model` names the recognition model (it fixes the script the text is read in) and `language` is the operator-declared language that model serves, so the field describes itself. When the opt-in cannot be answered (nothing configured, a busy inference gate, a failed run), the field reports `{"available": false, "reason": ...}` like the moderation field, and the meta request still succeeds. The text annotates meta only: an OCR request builds and caches exactly the image entry any other request for the URL would, byte for byte. Meta requests are never cached. One refusal: a meta request on a video URL answers `400`, because a video result is not an image (see [Video output](#video-output)). Errors in meta mode keep the usual status codes, and grammar failures answer with the JSON error envelope described in [Errors](#errors).

## Provenance (C2PA)

Crabcrop reads and writes [C2PA](https://c2pa.org/) Content Credentials. Two halves, independently useful:

**Verify, always on.** Every meta response carries a `provenance` object reporting what the source asset carried: whether it has a manifest, whether that manifest validates (`valid`, `trusted`, or `invalid`), who signed it, and how the store is composed. No configuration is needed, and verification runs on the c2pa-rs 0.91 line, the first that enforces the specification's exclusion rules.

**Sign, when you configure an identity.** Set `CRABCROP_C2PA_CERT` and `CRABCROP_C2PA_KEY` to a PEM certificate chain and its private key, and still JPEG, PNG, WebP, and AVIF outputs are signed as they are produced. A manifest cannot survive a transform byte for byte (decoding and re-encoding drop the segments it lives in), so the source itself is recorded in the output's manifest as a parent ingredient: its hash always, and its own manifest store whenever it carries one, so the chain stays walkable end to end. Outputs of the AI edits (background removal, studio mode, region removal, upscale, class and face edits) record the edit with the IPTC `compositeWithTrainedAlgorithmicMedia` source type. GIF, JPEG XL, video, and animated WebP outputs are served unsigned in this version.

The identity is yours, like the models and the secret: the server holds no certificate of its own. It is proven at boot with a sign-and-verify round trip, so a broken identity (a bare self-signed certificate, a key that does not match, a chain outside the C2PA certificate profile) is a startup error with the reason named, never a first-request surprise. `CRABCROP_C2PA_TSA_URL` optionally adds a timestamp authority.

Two honesty rules, stated plainly:

- A valid credential proves **origin, not untouched pixels**. It attests that this server produced the output from the ingredient it names, under the identity that signed it. It does not claim the image is unedited; the manifest itself records that it was transformed.
- Whether verifiers call the result **trusted** depends on your certificate, not on this software. A chain issued by a CA on the C2PA trust list (commercial CAs sell these, and at least one offers a free tier) reads as trusted. A private CA chain signs and validates, and verifiers show it as valid but not trusted, with the signer named. Both are useful; pick with your eyes open.

## Responses

A successful image response carries:

| Header | Value |
| --- | --- |
| `Content-Type` | The output format's media type, e.g. `image/webp`. |
| `Cache-Control` | `public, max-age=31536000, immutable` |
| `Vary` | `Accept, DPR, Sec-CH-DPR, Sec-CH-Width, Sec-CH-Viewport-Width, Width, Viewport-Width, Save-Data` |
| `X-Crabcrop-Quality` | Present only when crabcrop chose the quality (`quality(auto)`, a `max_bytes` reduction, or a `Save-Data` step); the value used. |
| `X-Crabcrop-Fallback` | `1` when the response is the configured fallback image rather than the source (see [Fallback image](#fallback-image)). |

### Caching

Signed URLs are content-addressed, so successful responses are cached and served without re-fetching or re-processing. Entries are keyed by the request's canonical form (the parsed operations in their one spelling), so equivalent spellings of the same request share one entry. The first tier is an in-memory LRU bounded by bytes: 256 MB by default, configurable with `CRABCROP_CACHE_MAX_MB` (`0` disables it). With `CRABCROP_RESULT_STORAGE_DIR` set, a second, disk-backed tier (bounded by `CRABCROP_RESULT_STORAGE_MAX_MB`, default 512 MB) sits underneath: results survive restarts, and disk hits are promoted into memory. With `CRABCROP_RESULT_BUCKET` set, a third tier sits behind the local ones: one S3 bucket (or an S3-compatible endpoint, via `CRABCROP_S3_ENDPOINT` and `CRABCROP_S3_REGION`, signed with the ambient AWS credentials) shared by every replica. A shared hit promotes into disk and memory; new results are written through to the bucket in the background; purges fan out to all three tiers. `CRABCROP_RESULT_PREFIX` namespaces the bucket's objects. Shared entries are stored in the disk tier's format, and the tier is best-effort by design: a bucket error reads as a miss, never a failed request. Only `200` responses are cached; errors never are. Because the response can depend on the request headers, negotiated responses are cached per negotiated format class, per effective DPR value, per quantized width-hint class, and per Save-Data class.

Concurrent requests for the same uncached URL share one fetch and one encode: the first requester does the work while the rest wait, then serve the stored entry. Cache keys also carry a fingerprint of the configuration that shapes output (the metadata policy, the quality tuning knobs, the width-hint step, and the model, font, encoder, and detector files in use), printed at startup as `configuration fingerprint: ...`. Restarting with any of those changed starts a fresh cache epoch: the persistent tiers keep the old entries under their old keys, unreachable, until they age out, instead of serving renders produced under the previous configuration. Presets and default operations need no fingerprint: both already change the canonical form every key is built from.

Entries in the persistent tiers are authenticated: each stored entry carries an HMAC-SHA256 tag under a key derived from the server secret, verified before the entry is served or indexed at startup. A file or object planted in the cache directory or bucket by anyone without the secret, or one byte of a stored entry altered on disk, reads as a miss and is re-rendered, never served. Rotating the secret therefore also invalidates the persistent tiers' contents: entries written under the old secret stop verifying and are replaced on demand. (The in-memory tier is process-local and needs no tag.)

### Color management and metadata

Pixels are decoded in the source profile's space. When the source embeds an ICC profile (JPEG and PNG sources are read), crabcrop converts to sRGB with an embedded-profile color transform before any operation runs, so sizes, crops, and filters all work in one known space; a source with no profile is assumed to be sRGB already. `filters:keep_profile` opts out per request: no conversion, and the source profile is embedded into JPEG and PNG outputs instead. CMYK JPEGs decode through the JPEG decoder's own CMYK conversion (embedded CMYK profiles are not honored yet); grayscale profiles are not transformed.

Metadata follows an operator policy, `CRABCROP_METADATA`:

| Value | Behavior |
| --- | --- |
| `strip` (default) | Outputs carry no source metadata. |
| `preserve` | JPEG outputs gain the source's EXIF, XMP, and Photoshop (IPTC) segments; PNG outputs gain EXIF and XMP. Other output formats do not carry preserved metadata. |

Preservation copies the source segments rather than re-serializing them, with one patch: the EXIF orientation is rewritten to 1, because the pixels were already rotated at decode and the original value would double-rotate in viewers. EXIF orientation itself is always honored at decode, under either policy.

### Fallback image

With `CRABCROP_FALLBACK_IMAGE` set (a URL, or a path to a local file read at startup), a source that cannot be fetched or decoded degrades to the fallback instead of an error: the fallback runs through the same operations and is served with `200` and the header `X-Crabcrop-Fallback: 1`. Signature and parse failures never fall back, a `403` or `400` stays honest, and neither does a source-policy refusal. Fallback responses are never cached, so a recovered origin is picked up on the next request; they are also served with `Cache-Control: no-store`, so a CDN or browser never pins the placeholder in place of the real image. With no fallback configured, error behavior is unchanged.

## Errors

Error bodies are plain text with a short message. Send `Accept: application/json` and a grammar failure (a `400` below) instead answers with the v2 JSON envelope, `{"error":{"code":"invalid_filters","message":"invalid filters segment: filters:bogus(1)"}}`. The codes are the parser's stable names, frozen as published: `empty_path`, `missing_signature`, `malformed_signature`, `invalid_size`, `invalid_crop`, `invalid_crop_rect`, `invalid_pad`, `invalid_expiry`, `unknown_preset`, `invalid_preset`, `unknown_root`, `invalid_root`, `presets_only`, `invalid_filters`, `missing_source_url`, `invalid_focal_rect`, `invalid_maxage`, `invalid_debug`, `invalid_curve`, `invalid_convolution`, `invalid_noise`, `invalid_frame`, `invalid_page`. The statuses you can get:

| Status | Meaning |
| --- | --- |
| `400` | The path or operations do not parse, or the operation is not supported: a malformed segment, an unknown filter or parameter, `removebg` into a format without alpha, `upscale` past the 4096 px output cap or the 262,144 px per-run input budget, a `crop-rect` or `remove_region` outside the image, a video format asked of a still source. |
| `403` | Bad signature, an expired `exp:` URL, or the source URL was refused by the source policy (host not in `CRABCROP_ALLOWED_SOURCES`, or resolving to a non-public address while `CRABCROP_BLOCK_PRIVATE_SOURCES` is on). |
| `404` | A `PUT` to `/upload/...` when uploads are not configured, or a `PUT` to any path that is not the upload route. |
| `413` | An upload body larger than the `CRABCROP_UPLOAD_MAX_MB` cap. |
| `500` | Encoding the output failed, or an AI model failed at inference time (the runtime errored, or a model session lock was poisoned by an earlier failure). A model that cannot be loaded at all fails startup instead of answering requests. |
| `501` | The URL is valid but this server was not configured for it: an AI filter with no model, a text overlay with no font, a video format with no encoder configured. |
| `502` | The source could not be fetched (network error, a non-2xx answer from the origin, or a non-image `Content-Type`), the source is larger than the 10 MiB fetch limit, or the fetched bytes could not be decoded as an image, including AVIF sources in builds without AVIF decoding, corrupt HEIC sources, and undecodable formats. A watermark that fails to load also lands here, and so does a configured AI model that ran but produced output the server cannot use (no outputs, an unreadable tensor, an unexpected shape). A configured video encoder that fails, times out, or exceeds its output cap also lands here. |
| `503` | The server shed the request because a processing, AI, video, or animation gate waited past its timeout; the response carries `Retry-After`. |
| `504` | The source fetch timed out (10 seconds). |

For example, a `removebg` request to a server with no background-removal model returns `501` with the body `removebg is not configured on this server`.

## Endpoints

| Route | Description |
| --- | --- |
| `GET /health` | Health check. Returns `200` with the body `ok`. |
| `GET /metrics` | Prometheus metrics in text format: request counts by status class (`crabcrop_requests_total`), request/load/process durations, cache hit and miss counters, loader and engine error counters, and byte totals. |
| `POST /purge/<signature>/<path>` | Invalidate every stored variant of one result, across all cache tiers. See below. |
| `POST /purge-group/<signature>/<group>` | Invalidate every stored result in a cache group, across all cache tiers. See below. |
| `PUT /upload/<signature>/<name>` | Store one original in the configured upload destination. Only exists when uploads are configured. See [Upload an original](#upload-an-original). |
| `GET /explain/<signature>/...` | A report on the URL instead of the image. See [Self-explaining URLs](#self-explaining-urls). |
| `GET /<signature>/...` | Image requests, as documented above. |

### Purge a cached result

```bash
curl -X POST "http://localhost:3000/purge/<signature>/300x200/https://example.com/photo.jpg"
```

The `<path>` is the original request path without its signature. The signature covers the target exactly as presented; the target is then canonicalized the same way entries are keyed, so purging any equivalent spelling of a path (the `jpg` alias, leading zeros, a preset reference) evicts the shared entry, `exp:` variants included. Compute the signature with your client library (`purge_signature` in the Rust crate and the Python SDK, `purgeSignature` in TypeScript): the HMAC-SHA256 of the path under a purge subkey derived from the secret, so only the secret holder can invalidate, and an image signature can never be replayed as a purge signature. The response is `200` with a body like `purged 2`: the count covers the entry and all of its variants (per format class and DPR). A bad purge signature is a `403` and removes nothing; purging a path that is not stored answers `purged 0`.

### Upload an original

crabcrop is a fetch-only service by design: sources live somewhere else, and transforms reference them by URL. The upload endpoint is the one deliberate exception, and it is off unless the operator turns it on. Configure exactly one destination:

- `CRABCROP_UPLOAD_DIR=/srv/originals` stores originals as files in a local directory.
- `CRABCROP_UPLOAD_BUCKET=originals` stores them in an S3 bucket, reusing the source S3 configuration (endpoint, region, and ambient AWS credentials); `CRABCROP_UPLOAD_PREFIX` namespaces the objects. With credentials configured, the bucket must also appear in `CRABCROP_ALLOWED_BUCKETS`, because uploaded objects are served back as `s3://` sources. There is no GCS destination: `gs://` sources are read-only in crabcrop.

With neither variable set, the endpoint does not exist: a `PUT` is a plain `404`. With one set, an upload is one signed `PUT`:

```bash
curl -X PUT --data-binary @photo.jpg "http://localhost:3000/upload/<signature>/heroes/photo.jpg"
```

The `<name>` is the object name, `/`-separated segments of `[A-Za-z0-9._-]` (no empty, `.`, or `..` segments, nothing starting with a dot, at most 255 bytes). The signature covers the name exactly as presented, under an upload subkey derived from the secret, the same domain-separation construction purge signatures use: only the secret holder can upload, an image signature or a purge signature can never be replayed here, and `unsafe` never authorizes an upload. Compute it with your client library (`upload_signature` in the Rust crate and the Python SDK, `uploadSignature` in TypeScript), or let the CLI do the whole job:

```bash
crabcrop upload ./photo.jpg heroes/photo.jpg
```

An upload token can carry an expiry, and a delegated token should: with `PUT /upload/<signature>/exp:<unix-seconds>/<name>` the signature covers the `exp:` segment along with the name, so the token stops working at that second and can neither be stripped of its expiry nor extended. The client libraries mint this form with an `expires` argument, and the CLI with `--expires <unix-seconds>`.

The body must decode as an image this server can serve (the same decode paths a transform request uses, size and dimension caps included); anything else is a `415` and is never stored. The validation decode runs under the same process gate as transforms, so a saturated server sheds an upload with a `503` and `Retry-After` instead of decoding ungated. `CRABCROP_UPLOAD_MAX_MB` caps one upload (default 10 MB, the source fetch cap, so a stored original is always one the loader will fetch back; larger configurations are refused at startup). What is stored is the validated form: for an SVG that is the sanitized document (scripts and event handlers stripped at store time, not only when crabcrop serves it back), and the name's extension must match the media kind the body validated as, so a PNG cannot be stored as `photo.jpg` or as `promo.html` (a name with no extension is accepted as-is). An upload to an existing name is a `409`, answered before the body is decoded, and the final write is a conditional create, so the guarantee holds even against a racing upload for the same new name; set `CRABCROP_UPLOAD_ALLOW_OVERWRITE=1` to let a `PUT` replace it (a `200` instead of the new-object `201`). A replacing `PUT` also purges every cached derivative of the replaced original, in all tiers, so the next transform request re-renders from the new bytes instead of serving a stale result. There is no listing and no delete.

The response body is the exact source reference to embed in transform URLs (`file:///heroes/photo.jpg` for a local destination, `s3://originals/heroes/photo.jpg` for a bucket), also returned in the `X-Crabcrop-Source` header. Serving works through the normal loader: a local destination doubles as the `file://` source root when no `CRABCROP_FILE_ROOT` is configured (and must sit under one that is), so the uploaded original is transformable the moment the `PUT` answers.

One posture note, stated plainly: enabling uploads changes this server from fetch-only to a service that accepts and stores caller-supplied bytes. The mitigations are the ones above (off by default, its own signature domain with an expiring form, decode-before-store, sanitized SVG storage, extension binding, a size cap, no overwrite by default, no listing or delete), and the destination should still be a directory or bucket dedicated to uploads, watched like any other store of user-supplied originals. If you serve the destination directly, serve it as images: the stored bytes are validated images under bound names, not arbitrary content, and a dedicated host keeps even a hypothetical confusion off your main domain.

### Self-explaining URLs

Every crabcrop URL can explain itself. Take any image URL, put `explain/` in front of the signature segment, and the server answers with a report on that URL instead of the image:

```bash
curl "http://localhost:3000/explain/<signature>/crop:1200x630:smart/filters:format(avif):quality(auto)/https://example.com/photo.jpg"
```

Browsers get a readable HTML page; send `Accept: application/json` for the same report as JSON. The report carries the signature verdict (valid, which key, when the link expires), the source and its probed facts (media kind, pixel dimensions, encoded size), every operation glossed in plain language in order, the referenced presets with their expansions, a forecast (output dimensions when the chain determines them, the output format, the output pixel count against the aggregate budget), the canonical form that gets cached with its cache group, and links back to the image itself and to the debug focal remainder when the chain contains a crop. The CLI's `explain` verb does the offline half of this (grammar and signature against a secret you pass); the server route adds the live facts only a server can know.

Explain is safe to expose to the same callers who hold signed URLs: the signature and expiry are verified exactly as the image path verifies them, before any source access, so a bad or expired URL is refused with the image path's own response and costs zero source fetches, and the one probe a valid URL pays for is a single bounded fetch through the loader whose facts come from headers alone, with no decode, no detection, and nothing written to any cache tier. The report contains only facts derivable from the URL and the source itself: no secrets or key material, no configuration values, paths, or fingerprints, no redirect or address details, and errors are the usual frozen codes and generic messages.

### Cache headers and groups

Image responses carry two standard headers. `Cache-Status` (RFC 9211) reports what the server's own caches did: `crabcrop; hit` for a memory hit, `crabcrop; hit; detail="disk"` for a disk hit, `crabcrop; hit; detail="shared"` for a shared-tier hit, and `crabcrop; fwd=uri-miss` when the response was produced from the origin.

A response also belongs to a cache group (RFC 9875) for each preset its URL referenced, listed in the `Cache-Group` header:

```http
Cache-Group: "small"
```

Groups make preset-wide invalidation one call instead of a script over every URL. Group membership accumulates: an entry stored before a group was ever referenced joins the group the first time any grouped spelling of its URL is served, so a group purge reaches entries that were first rendered through a plain, preset-free URL. The signature comes from the same libraries (`purge_group_signature` / `purgeGroupSignature`), which derive the purge-group endpoint's own subkey and sign the group name on your behalf, so nobody constructs it by hand:

```bash
curl -X POST "http://localhost:3000/purge-group/<signature>/small"
```

The response is `200` with a body like `purged 3`: every entry recorded in the group, across all cache tiers. A bad signature is a `403` and removes nothing; an unknown group answers `purged 0`.

Every image request also writes one structured JSON log line to stdout, with the operation summary, status, per-stage timings, and cache result (never the full source URL).

## Command line

The `crabcrop` binary doubles as its own operator CLI: when the first argument is a verb, it runs the verb and exits instead of starting the server. Configuration comes from flags over environment variables, the same convention as the MCP server: `--base` / `--secret` over `CRABCROP_BASE` / `CRABCROP_SECRET`, with the base defaulting to `http://localhost:3000`.

```bash
export CRABCROP_BASE=http://localhost:3000
export CRABCROP_SECRET=your-secret

# Sign a URL (validated against the server grammar first)
crabcrop sign "300x200/filters:grayscale/https://example.com/photo.jpg"

# Explain a URL segment by segment, verifying its signature
crabcrop explain "http://localhost:3000/<signature>/300x200/filters:grayscale/https://example.com/photo.jpg"

# Fetch the meta JSON for an image (a bare remainder is signed first)
crabcrop meta "300x200/https://example.com/photo.jpg"

# ... with placeholder fields and detection boxes included
crabcrop meta --placeholders --detections "300x200/https://example.com/photo.jpg"

# Purge one URL, or a whole preset group (Cache-Group)
crabcrop purge "300x200/https://example.com/photo.jpg"
crabcrop purge-group thumb

# Upload an original to the server's upload store (prints the
# source reference for transform URLs; see Upload an original)
crabcrop upload ./photo.jpg heroes/photo.jpg

# Convert a Thumbor URL (re-signed; losses are reported on stderr)
crabcrop convert-thumbor "unsafe/300x200/smart/https://example.com/photo.jpg"
```

## Configuration

All configuration comes from environment variables. Invalid values are startup errors, booleans included, the server refuses to start rather than run with a wrong port, cache budget, or safety flag. Boolean variables accept `1`, `true`, `yes`, `on` (any case) for on and `0`, `false`, `no`, `off` for off; anything else fails the boot with the variable named. Names follow one pattern: `*_MODEL` is an AI model artifact, `*_FILE` is a file read at boot, `*_DIR` is a directory, and `*_MAX_MB` is a size budget in megabytes.

| Variable | Default | Description |
| --- | --- | --- |
| `CRABCROP_SECRET` | required | The HMAC secret used to verify signed URLs, at least 16 bytes. The server will not start without it. |
| `CRABCROP_SECRET_KEY_ID` | `current` | The current secret's key ID, named by keyed signature segments (`<key-id>:<tag>`). |
| `CRABCROP_SECRET_PREVIOUS` | unset | The outgoing secret during a rotation window; the server verifies it alongside the current one. Same minimum length. |
| `CRABCROP_SECRET_PREVIOUS_KEY_ID` | `previous` | The previous secret's key ID. Must differ from the current one. |
| `CRABCROP_DEFAULT_OPS` | unset | Operations prepended to every request (a house format, quality, or watermark), validated at boot. See [Server-wide default operations](#server-wide-default-operations). |
| `PORT` | `3000` | Listen port. `CRABCROP_PORT` is the namespaced alias and wins when both are set. |
| `CRABCROP_ALLOW_UNSAFE` | off | Honors `unsafe` URLs (for the boolean spellings, see above). |
| `CRABCROP_CACHE_MAX_MB` | `256` | Result cache budget in megabytes. `0` disables the cache. |
| `CRABCROP_SSIM_TARGET` | `75` | SSIMULACRA2 target (0 to 100) that `quality(auto)` verifies against for JPEG and JPEG XL output. |
| `CRABCROP_METADATA` | `strip` | `preserve` copies source EXIF/XMP (and IPTC for JPEG) into JPEG and PNG outputs; see [Color management and metadata](#color-management-and-metadata). |
| `CRABCROP_WIDTH_STEP` | `100` | Quantization step in pixels for width client hints: hinted widths round up to a multiple of this. |
| `CRABCROP_AVIF_SPEED` | `9` | ravif speed tier for AVIF encodes, 1 (slowest, smallest output) to 10 (fastest). At 800w and quality 80, the default speed 9 encodes about 1.8x faster than speed 8 for about 15% more bytes; speed 10 is roughly 3x quicker than speed 8 for about 44% more bytes. Set 8 for the smallest files of the fast tiers. |
| `CRABCROP_JXL_SPEED` | `fastest` | jixel speed tier for JPEG XL encodes: `ultrafast`, `fastest`, `fast`, `medium`, `slow`, or `extraslow`. The default `fastest` encodes about 2x faster than `fast` for about 18% more bytes at 800w. Slower tiers shrink output with steep latency costs (`slow` is about 8x the latency of `fast` for about 18% fewer bytes at 800w). |
| `CRABCROP_PRESETS_FILE` | unset | Path to a presets file. See [Presets](#presets). |
| `CRABCROP_ONLY_PRESETS` | off | Rejects free-form operations in requests; only preset references (and `exp:` / `maxage:` metadata) are allowed. Requires at least one preset. |
| `CRABCROP_SOURCE_ROOTS_FILE` | unset | Path to a source roots file. See [Source roots](#source-roots). |
| `CRABCROP_ALLOWED_SOURCES` | unset (all hosts allowed) | Comma-separated host suffixes the loader may fetch from. A suffix matches the host itself or any subdomain of it. |
| `CRABCROP_BLOCK_PRIVATE_SOURCES` | off | Refuses sources whose host is, or resolves to, a non-public address: loopback, private, link-local, unspecified, shared (CGNAT), reserved, multicast, broadcast, documentation, benchmarking, and the other special-purpose ranges, including IPv4-mapped, IPv4-compatible, and 6to4 spellings of them. Checked before connecting and again on every redirect hop. |
| `CRABCROP_S3_ENDPOINT` | unset (AWS) | Base URL of an S3-compatible store for `s3://` sources, path-style. `CRABCROP_S3_REGION` sets the AWS region (default `us-east-1`). Signing credentials come from `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`; unset means unsigned. |
| `CRABCROP_GCS_ENDPOINT` | `https://storage.googleapis.com` | Base URL of the GCS XML API for `gs://` sources. `CRABCROP_GCS_REGION` (default `auto`) is the signing region; the HMAC pair is `CRABCROP_GCS_ACCESS_KEY` / `CRABCROP_GCS_SECRET_KEY`. |
| `CRABCROP_ALLOWED_BUCKETS` | unset | Comma-separated bucket names the `s3://` and `gs://` schemes may read; `*` allows all. Required when bucket credentials are configured (the server refuses to boot without it); without credentials, unset means all buckets, fetched unsigned. |
| `CRABCROP_FILE_ROOT` | unset | Directory that `file://` sources must stay under, canonicalized at boot and per request. Unset disables file sources. |
| `CRABCROP_REMOVEBG_MODEL` | unset | Path to a U²-Net-family ONNX model for the `removebg` filter. See [AI features](#ai-features). |
| `CRABCROP_BRAND_COLOR` | unset | Brand color for `studio(...)` composites, six hex digits (a leading `#` is allowed). An unnamed backdrop and `bg:brand` resolve to it; unset means white. Changing it starts a fresh cache epoch, like the other output-shaping settings. See [Studio mode](#studio-mode). |
| `CRABCROP_UPSCALE_MODEL` | unset | Path to a super-resolution ONNX model for the `upscale` filter. See [AI features](#ai-features). |
| `CRABCROP_INPAINT_MODEL` | unset | Path to a LaMa-family ONNX model (image plus mask inputs, fixed 512×512) for the `remove_region` filter. See [AI features](#ai-features). |
| `CRABCROP_SEGMENT_ENCODER_MODEL` / `CRABCROP_SEGMENT_DECODER_MODEL` | unset | A MobileSAM ONNX pair (image encoder, box-prompt decoder) for mask-tight class operations. Both paths are required together. See [AI features](#ai-features). |
| `CRABCROP_FACE_MODEL` | unset | Path to a YuNet ONNX model for face-aware `smart` crops, `crop:...:face`, and the face redaction filters. See [AI features](#ai-features). |
| `CRABCROP_CLIP_IMAGE_MODEL` / `CRABCROP_CLIP_TEXT_MODEL` | unset | A CLIP ONNX model pair for meta `tags` and `alt_text`. All four CLIP paths are required together. See [AI features](#ai-features). |
| `CRABCROP_CLIP_TOKENIZER` | unset | The CLIP export's `tokenizer.json`. |
| `CRABCROP_CLIP_LABELS` | unset | Text file with one tag label per line. |
| `CRABCROP_CLIP_THRESHOLD` | `0.05` | Minimum zero-shot score for a label to appear in `tags`. |
| `CRABCROP_MODERATION_MODEL` | unset | Path to a ViT classifier ONNX model (the Falconsai nsfw_image_detection shape) for meta moderation verdicts. See [AI features](#ai-features). |
| `CRABCROP_MODERATION_LABELS` | unset | Text file with one label per line: the verdict model's output names (in output order) when a model is set, or zero-shot prompt labels for the CLIP fallback when it is not. |
| `CRABCROP_AESTHETIC_HEAD` | unset | Path to a LAION-style linear aesthetic head (the PyTorch save) scored against the CLIP embedding in meta. Requires the `CRABCROP_CLIP_*` set. See [AI features](#ai-features). |
| `CRABCROP_OCR_DETECT_MODEL` / `CRABCROP_OCR_RECOGNIZE_MODEL` | unset | A PP-OCR-family ONNX model pair (DB detection plus CTC recognition) for meta OCR text. All three OCR paths are required together. See [AI features](#ai-features). |
| `CRABCROP_OCR_DICT` | unset | The recognition model's character dictionary: one character per line, in class order. |
| `CRABCROP_OCR_LANGUAGE` | `und` | The language tag the recognition model serves, echoed in meta OCR responses. |
| `CRABCROP_DETECT_MODEL` | unset | Path to a YOLOv8-family ONNX model for `blurclass` / `pixelateclass` / `recolorclass` and meta detections. See [AI features](#ai-features). |
| `CRABCROP_DETECT_LABELS` | unset | Class names file (one per line, in index order) for detection exports that carry no `names` metadata. |
| `CRABCROP_DETECT_THRESHOLD` | `0.5` | Minimum detection score. |
| `CRABCROP_DETECTORS_FILE` | unset | Path to a JSON file declaring additional detectors (`boxes` and `focal` contracts) from your own models. See [Declared detectors](#declared-detectors). |
| `CRABCROP_ORT_DYLIB` | unset | Path to the ONNX Runtime shared library, when it is not on the library search path. |
| `CRABCROP_FONT_PATH` | unset | Path to a TTF/OTF font for the `text(...)` overlay filter, loaded at startup. Unset disables the filter. |
| `CRABCROP_VIDEO_ENCODER` | unset | Path to the external video encoder program for `format(mp4)` / `format(webm)`, validated at startup. Unset disables video output. See [Video output](#video-output). |
| `CRABCROP_VIDEO_DECODER` | unset | Path to the external video decoder program for `frame:` extraction from video sources, validated at startup. Unset disables frame selection. See [Video frames](#video-frames). |
| `CRABCROP_VIDEO_TIMEOUT_SECS` | `30` | Hard wall-clock limit per external video tool run (one encode, or one frame probe/extraction), 1 to 600 seconds. |
| `CRABCROP_PDF_RENDERER` | unset | Path to the external PDF renderer program for `page:` rendering from PDF sources (its `pdfinfo` sibling is derived and validated alongside), validated at startup. Unset disables page selection. See [PDF pages](#pdf-pages). |
| `CRABCROP_C2PA_CERT` | unset | Path to a PEM certificate chain: the C2PA signing identity for output provenance. Set together with `CRABCROP_C2PA_KEY`; a partial set is a startup error, and the identity is proven at boot. Unset disables output signing. See [Provenance (C2PA)](#provenance-c2pa). |
| `CRABCROP_C2PA_KEY` | unset | Path to the private key for `CRABCROP_C2PA_CERT`, in PKCS#8 PEM form (`BEGIN PRIVATE KEY`; convert a traditional key with `openssl pkcs8 -topk8 -nocrypt`). |
| `CRABCROP_C2PA_TSA_URL` | unset | Optional timestamp authority URL recorded in output signatures. Requires the certificate pair. |
| `CRABCROP_PDF_TIMEOUT_SECS` | `30` | Hard wall-clock limit per external PDF tool run (one probe or one page render), 1 to 600 seconds. |
| `CRABCROP_VIDEO_MAX_MB` | `64` | Output cap per video encode in megabytes, 1 to 1024. The encode aborts the moment it is crossed. |
| `CRABCROP_FALLBACK_IMAGE` | unset | URL or local file path of the fallback image served when a source cannot be fetched or decoded. See [Fallback image](#fallback-image). |
| `CRABCROP_RESULT_STORAGE_DIR` | unset | Directory for the disk-backed result store. Unset disables the second cache tier. |
| `CRABCROP_RESULT_BUCKET` | unset | S3 bucket for the shared result store (multi-replica third tier). Unset disables it. |
| `CRABCROP_RESULT_PREFIX` | empty | Object prefix namespacing the shared result store's entries. |
| `CRABCROP_RESULT_STORAGE_MAX_MB` | `512` | Disk result store budget in megabytes. `0` disables storage even with a directory set. |
| `CRABCROP_UPLOAD_DIR` | unset | Local directory destination for the upload endpoint. Exactly one of this and `CRABCROP_UPLOAD_BUCKET` may be set; neither disables uploads (the route is a 404). Also becomes the `file://` source root when no `CRABCROP_FILE_ROOT` is set, and must sit under one that is. See [Upload an original](#upload-an-original). |
| `CRABCROP_UPLOAD_BUCKET` | unset | S3 bucket destination for the upload endpoint, reusing the source S3 endpoint, region, and ambient AWS credentials. |
| `CRABCROP_UPLOAD_PREFIX` | empty | Object prefix namespacing the upload bucket's objects. |
| `CRABCROP_UPLOAD_MAX_MB` | `10` | Cap on one upload in megabytes, 1 to 10 (the source fetch cap: a stored original must stay servable). |
| `CRABCROP_UPLOAD_ALLOW_OVERWRITE` | off | Lets an upload `PUT` replace an existing object. Off, an existing name is a `409`. |

With `CRABCROP_BLOCK_PRIVATE_SOURCES` on, the loader resolves the host to check its addresses and pins the connection to the validated address, so the address that was checked is the address that gets connected (no DNS-rebinding gap). Source URLs with userinfo (`http://user:pass@host/...`) are always refused: the loader never presents Basic credentials to an origin.

### Server-wide default operations

`CRABCROP_DEFAULT_OPS` holds an operations string (the same grammar as a URL's operations, `preset:` references included) that the server prepends to every request, for a house format, quality, or watermark you do not want baked into every URL. The decisions, once:

- **The signature never covers the defaults.** They are operator configuration, applied after verification, like preset expansion.
- **Defaults sit before the URL's own operations**, and for the singleton directives (format, quality, fill, dpr, max_bytes) the URL wins by the default being dropped from the merged list. A bare request under defaults and the same request spelled out in full therefore produce the same operations and share a cache entry. Everything else (sizes, filters, watermarks) applies from both, in order: a default blur and a URL blur stack.
- **Cache entries key on the merged operations**, exactly like presets, so changing the defaults changes the keys and old entries age out unused. Purges account for the merge too: purging a URL removes its merged entry.
- **Presets-only mode is unaffected in what it rejects** (the URL's own operations), and the defaults still apply: they are not URL content.
- Cache groups stay the URL's own presets; a preset referenced only by the defaults does not join entries to its group.

### Rotating the secret

Rotation never invalidates a URL that is already in the wild, in three steps:

1. **Open the window.** Generate the new secret. Set `CRABCROP_SECRET` to it with `CRABCROP_SECRET_KEY_ID` naming it (say `2026-10`), and move the outgoing secret to `CRABCROP_SECRET_PREVIOUS` with `CRABCROP_SECRET_PREVIOUS_KEY_ID` naming it (say `2026-08`). Restart. The server now verifies both keys: a URL whose signature segment carries a key ID (`2026-08:<tag>`) is checked against that key only, and a URL with no ID is checked against both, so everything minted before key IDs existed keeps working too. Purge signatures route the same way.
2. **Mint with the new key.** Point your builders at the new secret and stamp its ID: `buildURL(..., { keyId: "2026-10" })` in TypeScript, `build_url(..., key_id="2026-10")` in Python, `sign_with_key_id` / `build_url_with_key_id` in Rust, or `crabcrop sign --key-id 2026-10` on the CLI. New cache entries accumulate under the new key from here on.
3. **Close the window.** When the old key's URLs have aged out of your caches and pages (the result cache's `max-age` is one year, but in practice your own TTLs and page lifetimes govern), remove `CRABCROP_SECRET_PREVIOUS` and restart. From that moment the old secret verifies nothing: its remaining URLs answer `403`, which is the point of rotating.

An unknown key ID is indistinguishable from a bad signature, by design: `403`, no hint about which IDs exist.

## AI features

The features that run ONNX models in-process: `removebg` (background removal), `studio` (product-shot composites and scene reframing on the same mask model, see [Studio mode](#studio-mode)), `upscale` (super-resolution), `remove_region` (region removal by inpainting), face detection for crops and redaction, and per-object segmentation that tightens the class filters to silhouettes. The models are files you supply, deployment artifacts, like the `Dockerfile.ai` image variant in this repo. crabcrop never downloads a model, at startup or at request time.

How configuration works:

- Point `CRABCROP_REMOVEBG_MODEL` at a U²-Net-family model file (320×320 input; the U2Netp export is a good fit) and/or `CRABCROP_UPSCALE_MODEL` at a super-resolution model. The upscale contract is dynamic-size RGB in (values 0..1), the same image out at an integer factor; Real-ESRGAN-family ONNX exports follow it.
- Point `CRABCROP_INPAINT_MODEL` at a LaMa-family inpainting model to enable the `remove_region(x,y,w,h)` filter: the rectangle is erased and inpainted from the surrounding pixels, with the same bounds rule as `blur_region` (a rectangle that overflows the image is a `400`). The contract is a two-input session (image plus mask) at a fixed 512×512: image in as RGB values 0..1, mask 0..1 with 1 the hole, output back in 0..255. The OpenCV zoo export of big-lama (`inpainting_lama_2025jan`, Apache-2.0, from the OpenCV zoo model collection) follows it. Each removal runs the model once on a square crop window around the rectangle (twice the rectangle's larger axis, so the model sees context as well as hole) resized to 512×512, and only the rectangle's pixels are taken from the result: everything outside it comes back bit-identical. Meta responses for a removal carry an `inpaint` object listing the erased rectangles. Know the limits before you rely on it: the model *fabricates* what goes in the hole, plausibly on smooth or textured surroundings and visibly wrong on structured content (text, faces, repeating patterns, straight edges that should continue). A removal takes seconds of inference (behind the shared AI gate), and the result cache absorbs repeat requests for the same URL, so a given removal is normally paid for once.
- Point `CRABCROP_FACE_MODEL` at a YuNet model to make `crop:...:smart` center on detected faces (the area-weighted center of the faces found). Face detection only ever improves a smart crop: no model, no faces, or a detection failure all fall back to the detail-based focal point, and a smart crop without the model behaves exactly as before. The same model powers the explicit face operations: `crop:...:face` (crop around the largest face; falls back to the smart window when detection fails or finds nothing), and the `blurfaces` / `pixelatefaces` redaction filters. Those require the model, without one they return `501`, and a detection failure fails a redaction request with `500` rather than serve an unredacted image. Chained face operations in one request share a single detection.
- Point the `CRABCROP_CLIP_*` set at a CLIP model pair (image and text ONNX exports with the projection head), its `tokenizer.json`, and a label file with one label per line, and meta responses gain `tags` and `alt_text`: zero-shot labels scored against the processed image, plus a short alt text composed from the best three. Labels are embedded once at boot, so a meta request runs only the image encoder, behind the same inference gate; a tagging failure omits the fields instead of failing the response. Labels are embedded exactly as written, so phrase-style labels (`aerial view of a beach`) work as well as single words.
- Point `CRABCROP_MODERATION_MODEL` at a ViT classifier export (the Falconsai nsfw_image_detection shape: a 224x224 image input, one logits row out) and meta requests that opt in with `moderation=1` gain a `moderation` object with the model's per-category verdict scores, softmaxed into probabilities and reported with the model identity. `CRABCROP_MODERATION_LABELS` names the model's outputs, one per line in output order, with the count checked against the model at boot; without it, a two-output model answers as `normal` and `nsfw`, and any other output count is a startup error. With no model set, the labels file instead names zero-shot prompt labels scored by the CLIP pair (which must then be configured), reported as review `signals`, never as verdicts. Verdicts are annotations only: nothing is blocked, blurred, or altered at serve time, and a failed or unconfigured moderation reports itself unavailable in the field instead of failing the meta request. Treat the scores as a model's opinion, not ground truth: the reference model's training data is undisclosed and no subgroup evaluation is published, and NSFW classifiers as a class over-flag swimwear, medical imagery, and art. Validate on your own corpus before acting on a score.
- Point `CRABCROP_AESTHETIC_HEAD` at a LAION-style linear aesthetic head (the PyTorch save, e.g. `sa_0_4_vit_b_32_linear.pth` for a ViT-B/32 encoder) and meta responses gain an `aesthetic` object: the head's score of the CLIP embedding, reported with the head's and the encoder's identities. The head must match the configured encoder: a head built for a 768-dimensional encoder (ViT-L/14) scored against a 512-dimensional embedding (ViT-B/32) reports a null score with `reason: "dimension_mismatch"` and the expected and actual dimensions, per request, never as a startup failure; a malformed head file is a startup error, like every other configured artifact. The same embedding is available on its own: meta requests that opt in with `embeddings=1` gain `embedding` (base64-packed float16 little-endian) plus `embedding_model`, `embedding_dim`, `embedding_encoding`, and `embedding_normalized`, ready to store in a vector index or a DAM. The score is descriptive only, never a gate: it encodes the taste of the LAION aesthetics data, itself scored against human ratings of mostly conventional photography, so it favors sharp, brightly lit, conventionally composed photos and underrates plenty of good work. Do not rank, filter, or price by it without validating on your own corpus.
- Point `CRABCROP_OCR_DETECT_MODEL` and `CRABCROP_OCR_RECOGNIZE_MODEL` at a PP-OCR-family model pair (a DB text detector and a CTC recognizer) and `CRABCROP_OCR_DICT` at the recognition model's character dictionary, and meta requests that opt in with `ocr=1` gain an `ocr` object: the lines the pipeline reads in the processed image, in reading order, each with its text, a confidence score, and its box, plus the lines joined as one `text_content`. The reference artifacts are the RapidOCR exports of PP-OCRv6 small (Apache-2.0, from the RapidOCR model collection): a detection model of about 10 MB and a recognition model of about 21 MB, with the `ppocrv6_dict.txt` dictionary (18,708 characters: Chinese, Japanese, Latin scripts and more in one model). The dictionary is operator configuration, not a download at runtime: it must be the table the recognition model was trained with, and the boot probe checks its character count against the model's output width, so a mismatched set fails startup rather than returning misaligned text. The recognition model fixes the script and the language coverage; `CRABCROP_OCR_LANGUAGE` declares which language the deployment serves (a BCP-47 tag, default `und`) and meta echoes it beside the model identity, so a stored response stays interpretable after the model is swapped. Text is an annotation only: it never alters, blocks, or re-renders the image, and an OCR request builds and caches exactly the image entry any other request for the URL would. One pipeline guard worth knowing: a detection box spanning essentially the whole frame is dropped, because on smooth backgrounds (a gradient, a clear sky) the detector fires as one giant component, and a real text line always keeps margins on some side. Expect photo-text variance: clean rendered or printed text reads essentially perfectly, while small, stylized, or low-contrast text in photos will misread characters (a 0 read as an O is the classic), merge or split lines, and miss text the detector never fires on. Scores are the model's own confidences, useful for triage, not proof. Validate on your own corpus before building on the text.
- Point `CRABCROP_DETECT_MODEL` at a YOLOv8-family detection export and two more surfaces unlock. The class filters `blurclass(<class>)` and `pixelateclass(<class>)` redact every detected object of a class, and `recolorclass(<class>,<hex>)` paints every detected object of a class a solid color, alpha preserved (repeat the filter for more classes; matching is case-insensitive). They inherit the face redaction rules: no model is a `501`, an unknown class is a `400`, and a detection failure is a `500`, never an unredacted response. And meta requests with `detections=1` return the detection boxes for debugging. The class vocabulary comes from the model's own `names` metadata, which Ultralytics exports carry; for exports without it, `CRABCROP_DETECT_LABELS` supplies one class per line in index order. A model whose output is not the YOLOv8 layout is rejected at boot.
- Point `CRABCROP_SEGMENT_ENCODER_MODEL` and `CRABCROP_SEGMENT_DECODER_MODEL` at a MobileSAM export pair (an image encoder and a box-prompt decoder; MobileSAM is Apache-2.0, and the pair is about 28 MB plus 17 MB as ONNX) and the class filters change shape: `blurclass` and `pixelateclass` redact each object's silhouette instead of its rectangle, and `recolorclass` paints the silhouette. Segmentation is box-prompted: each detection is prompted to the decoder as its own box, and the mask that comes back decides, pixel by pixel, what the operation touches; the mask's soft edge blends, so silhouettes antialias instead of stair-stepping. The image is encoded once per request however many class operations chain, and each box costs one small decode. Meta responses for a request that segmented carry a `segments` object: per object, the label and detection score and box, the mask's tight box (the silhouette's own extent, often well inside the detection), its coverage (the share of the box the silhouette fills), and its centroid. Without the pair, the class filters keep their rectangle behavior exactly; with the pair, a segmentation failure fails the request like a detection failure does, never a silently rectangular redaction. Class operations still need a detection model: the segmenter refines boxes, it does not find them.
- Models load at boot, before the server reports ready. A configured model that cannot be loaded, missing file, wrong shape, is a startup error with a one-line diagnostic, not a surprise for the first request. The upscale model's native factor is probed at load with a 16×16 test image; a model that does not enlarge by an integer factor is rejected at boot. The mask model is probed at load too: an export whose output is not the 320x320 mask field is rejected at boot rather than failing on the first `removebg` request. The inpainting model is probed the same way (a small removal must succeed, and a model without the image-plus-mask input pair is rejected) rather than failing on the first `remove_region` request. The segmentation pair is probed the same way again (an encode and a box-prompted decode must succeed end to end, and a decoder without the prompt inputs is rejected) rather than failing on the first class operation. The OCR set is probed the same way too (detection on a blank image and recognition on a blank crop must both succeed, and the dictionary's character count must match the recognition model's output width) rather than failing on the first `ocr=1` request.
- With no model configured, the filter is off: requests that use it fail cleanly with `501` (`removebg is not configured on this server`, `upscale is not configured on this server`, `remove_region is not configured on this server`). There is no silent fallback, `upscale` never degrades into a plain resize.
- Inference runs behind a server-wide gate of 2 concurrent inferences, so a burst of AI requests cannot starve the normal transform pipeline.

### Declared detectors

Beyond the one built-in detection model, `CRABCROP_DETECTORS_FILE` points at a JSON file that declares more of your own models against two built-in contracts. The file is configuration only: crabcrop never loads or runs operator code; it wires each named model to a contract it already implements.

```json
[
  {
    "name": "crates",
    "model": "/models/crates.onnx",
    "kind": "boxes",
    "labels": "/models/crates.txt",
    "threshold": 0.4
  },
  {
    "name": "logos",
    "model": "/models/logos.onnx",
    "kind": "boxes",
    "labels": ["nike", "adidas", "puma"]
  },
  {
    "name": "saliency",
    "model": "/models/saliency.onnx",
    "kind": "focal"
  }
]
```

- `name` (required) identifies the declaration: letters, digits, `_`, `-`; duplicate names are a startup error, and `builtin` is reserved for the built-in detector (it is the source name meta detections report for its boxes). `model` (required) is the ONNX file path. `kind` (required) is `"boxes"` or `"focal"`; anything else is a startup error naming the declaration.
- The file is bounded configuration: at most 1 MiB, at most 16 declarations (each one loads a model at boot and joins every meta request's detections, so the count bounds both costs). A declaration's vocabulary is capped at 4096 classes, inline labels of at most 256 characters each.
- **`boxes`** is a YOLOv8-family detector, decoded by the same post-processing as `CRABCROP_DETECT_MODEL`. Its class vocabulary comes from the declaration's `labels`, a path to a labels file (one class per line, in index order) or an inline array of class names, when given, and from the model's `names` metadata otherwise; declared labels win when both exist. `threshold` (0 to 1) sets its score floor, default 0.5 like the built-in detector. A `boxes` declaration with no vocabulary from either source is a startup error, as is a model whose output the decoder cannot read.
- **`focal`** is a heatmap model: one fixed `[1, C, H, W]` input (C is 1 for luma, 3 for RGB) of at most 409,600 pixels (the detection contract's 640x640 frame; the input tensor and the boot probe are allocated from the declared shape, so a larger declared frame is a startup error) and exactly one output shaped `[H, W]`, `[1, H, W]`, or `[1, 1, H, W]`. The focal point is the heatmap's hottest cell, ties going to the first in row-major order. `labels` and `threshold` do not apply to this kind.
- Class redaction resolves a class across providers in a fixed order: the built-in detector first, then declared `boxes` detectors in file order; the first provider whose vocabulary knows the class runs it. Chained class filters still share one run per provider per request.
- Meta detections aggregate every configured detector: the built-in's boxes first, then each declared detector's in file order, and every box carries a `detector` field naming its source (`"builtin"`, or the declaration's name). One detector's failure omits only its own boxes.
- Smart crops consult focal sources in this order: a caller `focal:` rectangle, then the face model, then the first declared `focal` provider, then the built-in saliency estimate. Only the first `focal` declaration is consulted; later ones are loaded and probed at boot, then stand by.
- Every declaration is loaded and probed at boot, like every other model: a missing model file, an unreadable or malformed declarations file, an unknown kind, or an output that does not fit the contract is a startup error naming the declaration.
- `upscale(4)` on a server whose model is natively 2× composes two model runs and lands the exact requested size; when the native factor does not divide the request, a final high-quality resize trims the difference. Source transparency is carried across an upscale even though the model only sees RGB.
- The ONNX Runtime library itself comes from `CRABCROP_ORT_DYLIB`, or from the standard `ORT_DYLIB_PATH` variable, or the system library search path when neither is set.

The `Dockerfile.ai` variant in this repo shows the intended shape: the base image stays free of AI payload, and a second layer adds the pinned ONNX Runtime 1.31.0, the U2Netp model, and the YuNet model, each hash-checked at image build time, and enables the features by setting the variables above. Layer your own super-resolution model the same way and set `CRABCROP_UPSCALE_MODEL` to enable `upscale`.

## Migrating

Guides for moving existing URL templates over:

- [From Thumbor](docs/migrating-from-thumbor.md): concept and operation mapping, plus the `convert-thumbor` verb, which converts and re-signs URLs mechanically.
- [From imgproxy](docs/migrating-from-imgproxy.md): option-by-option mapping tables.
- [From Cloudinary](docs/migrating-from-cloudinary.md): transformation mapping, and what changes when assets become origin URLs.
- [From imgix](docs/migrating-from-imgix.md): parameter-by-parameter mapping tables, including the adjustment scales that change meaning in translation.
- [From Cloudflare Images](docs/migrating-from-cloudflare-images.md): delivery-model and option mapping, from named variants and flexible variants to presets and signed operations, including the adjustment scales that change meaning in translation.
- [From ImageKit](docs/migrating-from-imagekit.md): transformation-chain and parameter mapping, from named transformations to presets, including the values that change meaning in translation.
- [From Gumlet](docs/migrating-from-gumlet.md): parameter-by-parameter mapping tables, from source default parameters to presets, including the adjustment scales that change meaning in translation.

## Deployment

The repo ships a multi-stage `Dockerfile` (the runtime image carries the binary, CA certificates, and the dav1d library for AVIF decoding):

```bash
docker build -t crabcrop:latest .
docker run -p 3000:3000 -e CRABCROP_SECRET=your-secret crabcrop:latest
```

For the AI features, build `Dockerfile.ai` on top of that image; see [AI features](#ai-features).

On Kubernetes, `deploy/helm` is a Helm chart and `deploy/kustomize` the same workload in kustomize form (base, components for presets, the disk tier, and the AI variant, plus single, multi-replica, and full overlays); `deploy/compose` runs it under Docker Compose. Without Kubernetes, `deploy/ecs-fargate`, `deploy/cloud-run`, and `deploy/container-apps` are per-cloud recipes (task definition, service manifest, and app manifest, each with its secret wiring and a smoke checklist). For a CDN in front of the origin, `deploy/cloudflare` is the Cloudflare recipe (cache rules, purge mapping with cache groups bridged to cache tags, format negotiation and expiry caveats) and `deploy/cloudfront` is the CloudFront recipe (a cache policy that respects origin headers, purge by path and by cache tag, and an Accept-normalizing edge function), and `deploy/fastly` is the Fastly recipe (origin cache headers left in charge, cache groups bridged to surrogate keys for group purges, and Image Optimizer kept off), and `deploy/bunnycdn` is the BunnyCDN recipe (origin Cache-Control left in charge, cache groups bridged to CDN tags by a middleware script for group purges, and Bunny Optimizer kept off), and `deploy/akamai` is the Akamai recipe (origin cache headers put in charge through the Caching behavior, cache groups bridged to cache tags by per-preset rules for group purges, and Image and Video Manager kept off).

Run crabcrop behind a reverse proxy or CDN in production. The application bounds its own work (the processing, AI, video, and animation gates shed overload with `503`), but connection-level protections belong at the edge: request and header timeouts, connection limits, rate limiting, and TLS termination. Restrict `/health` and `/metrics` to your internal network at the proxy: both are unauthenticated (they are served with `Cache-Control: no-store`), and the metrics expose traffic and error counters.

## Development

```bash
cargo build
cargo test
```

AVIF source decoding is behind the `avif-decode` cargo feature (off by default, on in the Docker image and the release builds; see the source-format notes above). Building with it needs dav1d 1.3 or newer reachable through pkg-config, or a vendored build:

```bash
SYSTEM_DEPS_DAV1D_BUILD_INTERNAL=always cargo build --features avif-decode
```

Running the server locally takes only a secret:

```bash
CRABCROP_SECRET=local-dev-secret cargo run
```

## Security

Found a security vulnerability? Please report it privately — see
[SECURITY.md](SECURITY.md). Please don't open a public issue, pull request, or
discussion for security problems.

## License

MIT. See [LICENSE](LICENSE).
