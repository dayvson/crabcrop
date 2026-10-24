# crabcrop

<p align="center">
  <img src="assets/crabcrop-mascot.webp" alt="The crabcrop mascot: a crochet crab waving hello and holding a photo" width="260">
</p>

crabcrop is an on-demand image optimization service written in Rust. You request an image with the operations encoded in the URL — resize, crop, convert, filter — and crabcrop fetches the source image, applies the operations in order, and returns the result. There are no pre-generated thumbnails and no build step: the URL is the API.

crabcrop is for sites and apps that serve the same images at many sizes and want signed, cacheable image URLs in front of an origin image store.

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

A request with a bad signature gets `403` and the body `bad signature`. The literal signature `unsafe` skips verification, but only when the server runs with `CRABCROP_ALLOW_UNSAFE=1` (or `true`, `yes`, `on`). Keep it off in production. Purges are the exception that never bends: `/purge` and `/purge-group` always require an HMAC signature made with the secret, even in unsafe mode.

### Expiring URLs

Add an `exp:<unix-seconds>` segment anywhere among the operations to make a signed URL expire:

```text
/<signature>/exp:1893456000/300x200/https://example.com/photo.jpg
```

The segment is inside the signed remainder, so the HMAC covers it: extending the expiry without the secret breaks the signature. Past the expiry the request gets `403` with the body `signature expired`, and expired URLs never serve from the cache. The comparison is exact: the URL is valid strictly before its `exp` second, with no built-in allowance for clock skew. Keep the server's clock synchronized (NTP), and mint expiries with whatever margin your own clock situation needs. URLs without `exp` never expire, exactly as before.

### Source images

The loader fetches the source under fixed limits, for every source kind:

- 10 seconds total per fetch, including redirects.
- 10 MiB maximum body; larger sources fail with `502` (the request was fine; the origin object is too big).
- If an HTTP response carries a `Content-Type` header, it must start with `image/`.

Decoding is bounded too, so a small file cannot expand into gigabytes: at most 16384 px per axis and 256 MiB of decoder allocations, and animated GIF and WebP sources at most 200 frames and 100 million decoded pixels in total. Sources past the bounds fail with `502`.

HTTP and HTTPS sources additionally allow up to 5 redirects, with the [source policy](#configuration) re-checked on every hop.

Three more source kinds cover originals that do not live behind a public URL:

- `s3://bucket/key`: fetched from AWS S3, or from an S3-compatible store (MinIO, R2, Spaces) when `CRABCROP_S3_ENDPOINT` is set. Requests are signed with SigV4 when AWS credentials are in the environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`); with none, the fetch is unsigned and public buckets just work.
- `gs://bucket/key`: the same, against Google Cloud Storage's S3-compatible XML API, signed with the HMAC pair `CRABCROP_GCS_ACCESS_KEY` / `CRABCROP_GCS_SECRET_KEY` when set.
- `file:///path`: a local file under `CRABCROP_FILE_ROOT`. The path is canonicalized per request and must stay under the root, so `..` segments and symlink escapes are refused. Unset root disables file sources.

Bucket sources are gated by `CRABCROP_ALLOWED_BUCKETS` rather than the host policy, since the endpoint is operator configuration, not request input. When signing credentials are configured, the allowlist is required: the server refuses to boot without it, because signed fetches would otherwise reach every bucket the credentials can read. List `*` to deliberately allow all buckets. Store responses are not redirected: a 3xx from a store surfaces as its status code.

JPEG, PNG, WebP, GIF, and JPEG XL sources are decoded. EXIF orientation is applied before any operation runs, so phone photos come out upright. Animated GIFs keep their frames when the output format is GIF; converting an animated GIF to a still format uses the first frame. Animated WebPs keep their frames when the output format is WebP or GIF, with frame delays and the loop count preserved; converting one to a still format uses the first frame. Animated WebP output is always encoded losslessly: the pure-Rust WebP encoder has no lossy animation mode, so a quality setting does not apply to it (the same is true of GIF output). AVIF sources are decoded in builds that include AVIF decoding: the Docker image, the release binaries, and the Homebrew install (which ships the release binary). A source build includes it with `--features avif-decode`, which links dav1d (a system libdav1d 1.3 or newer, or a vendored build); without it, AVIF sources fail with `502` and an error naming the feature. Animated AVIF sources decode their first frame. HEIC/HEIF sources are rejected and also fail with `502`.

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
fit-in:<width>x<height>
```

Scales the image proportionally so it fits inside the box. A `0` axis is unbounded. Flips are not allowed here.

```bash
curl "http://localhost:3000/unsafe/fit-in:800x600/https://example.com/photo.jpg"
```

### Pad to an exact size

```text
pad:<width>x<height>[:<fill>]
```

Fits the image inside the box (the fit-in rules, including `no_upscale`), centers it, and fills the rest of the canvas so the output is exactly the requested size. Both dimensions are required in pixels: a proportional (`0`) axis, `orig`, and flipped sizes are rejected, the same exact-box contract as `crop:`. The fill is a 6-digit hex color, or `blur` for ambient bars: a cover-scaled, heavily blurred copy of the image itself, the standard treatment for user-generated content in fixed-aspect slots. With no fill named, the request's `fill(...)` filter color is used, or black when there is none.

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
| `smart` | A window centered on the focal point: a detected face when a face model is configured (see [AI features](#ai-features)), otherwise the point of greatest local detail in the image. |
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
| `brighten` | `brighten(40)` — integer from −255 to 255, added to every channel and clamped to 0–255 | Brighten (or darken, with a negative value). | `filters:brighten(40)` |
| `contrast` | `contrast(25)` — percent, −100 to 100; 0 is unchanged, −100 is flat gray, positive raises contrast | Adjust contrast. | `filters:contrast(25)` |
| `huerotate` | `huerotate(90)` — degrees, integer | Rotate the hue. | `filters:huerotate(90)` |
| `sharpen` | `sharpen(1.5)` — strength, a positive number | Unsharp-mask sharpening. | `filters:sharpen(1.5)` |
| `saturation` | `saturation(1.5)` — 0 to 255; 0 is grayscale, 1 is unchanged | Scale color saturation. | `filters:saturation(0.5)` |
| `rgb` | `rgb(10,-5,0)` — per-channel offsets, each −255 to 255 | Add offsets to the red, green, and blue channels. | `filters:rgb(10,-5,0)` |
| `equalize` | none | Histogram equalization, per channel. | `filters:equalize` |
| `rotate` | `rotate(45)` — any angle in degrees, decimals allowed; negatives allowed | Rotate the image. Multiples of 90 are exact quarter turns; other angles expand the canvas to fit, filling the uncovered corners with the `fill` color (white by default) or transparency when the image carries alpha. | `filters:rotate(-90)` |
| `trim` | `trim` or `trim(20)` — tolerance 0–255, default 10 | Crop away the border that matches the top-left corner color within the tolerance. | `filters:trim(20)` |
| `sepia` | none | The classic warm sepia tone. | `filters:sepia` |
| `duotone` | `duotone(1a1a40,ffd9a0)` — two colors, six hex digits each | Map shadows to the first color and highlights to the second, by luminance. | `filters:duotone(1a1a40,ffd9a0)` |
| `monochrome` | `monochrome(ff8000)` — six hex digits | Single-hue rendering: each pixel's luminance scaled by the color, so full luminance renders exactly that color. | `filters:monochrome(ff8000)` |
| `pixelate` | `pixelate(8)` — block size in pixels, at least 1 | Mosaic the image into square blocks, each painted with the average of its pixels. | `filters:pixelate(8)` |
| `flip` | `flip(h)` or `flip(v)` | Mirror the image horizontally or vertically, at this point in the chain. | `filters:flip(h)` |
| `blur_region` | `blur_region(x,y,w,h)` — pixels | Gaussian-blur one rectangle (license plates, screens, documents). Coordinates address the image at this point in the chain; blur strength scales with the region. A region that overflows the image is an error, not a partial redaction. | `filters:blur_region(120,40,180,60)` |
| `pixelate_region` | `pixelate_region(x,y,w,h)` — pixels | Pixelate one rectangle, with the same bounds rule as `blur_region`. Block size scales with the region. | `filters:pixelate_region(120,40,180,60)` |
| `radius` | `radius(24)` — pixels; or four values `radius(24,24,0,0)` in CSS corner order (top-left, top-right, bottom-right, bottom-left) | Round the corners, antialiased. Radii larger than the frame shrink proportionally so corners never overlap. The output format must carry alpha: `png`, `webp`, `jxl`, or `avif`. | `filters:radius(24):format(png)` |
| `border` | `border(4,000000)` — width in pixels, color as six hex digits | Paint an opaque band of the color inside the frame. Dimensions do not change. Works on any format. | `filters:border(4,000000)` |
| `mask` | `mask(circle)` or `mask(ellipse)` | Keep only the inscribed shape (the circle's diameter is the smaller dimension); the rest becomes transparent. The output format must carry alpha. | `filters:mask(circle):format(png)` |

```bash
curl "http://localhost:3000/unsafe/600x400/filters:sharpen(1.2):saturation(1.1)/https://example.com/photo.jpg"
```

**Encoding directives** leave the pixels alone and steer the encoder:

| Directive | Parameters | Description | Example |
| --- | --- | --- | --- |
| `format` | `format(webp)` — `jpeg` (or `jpg`), `png`, `webp`, `jxl`, `avif`, `gif` | Output format. Overrides `Accept` negotiation. | `filters:format(avif)` |
| `quality` | `quality(80)` — 1 to 100, or `quality(auto)` | Encoding quality for JPEG, JPEG XL, AVIF, and WebP. Default 80. | `filters:quality(auto)` |
| `max_bytes` | `max_bytes(50000)` — byte budget, at least 1 | Re-encode at decreasing quality until the output fits the budget. | `filters:max_bytes(50000)` |
| `fill` | `fill(ff0000)` — six hex digits, RGB | Background color used to flatten transparency when encoding to JPEG. | `filters:fill(ffffff)` |
| `no_upscale` | none | Cap every resize in the request at the source dimensions. | `filters:no_upscale` |
| `dpr` | `dpr(2)` — one of `1`, `1.5`, `2`, `3` | Scale size and crop targets by the device pixel ratio, so one URL serves standard and retina displays. | `filters:dpr(2)` |

Notes on the directives:

- `quality(auto)` picks a quality per image, from 50 for flat images to 95 for detailed ones, based on content analysis. For JPEG and JPEG XL the pick is then verified: crabcrop encodes, decodes the result, and scores it against the processed source with SSIMULACRA2, adjusting quality (at most 4 encodes) until the score lands at the target, 75 by default and configurable with `CRABCROP_SSIM_TARGET`. The achieved score is reported in meta responses as `ssimulacra2`. Verification happens on cache misses only, but it is real work: scoring a full-size image takes about a second. When crabcrop picks the quality — for `quality(auto)`, or when `max_bytes` steps quality down — the response carries an `X-Crabcrop-Quality` header with the value used.
- `max_bytes` can only steer formats with a quality setting: JPEG, JPEG XL, AVIF, and WebP. If quality reaches its floor and the output is still over budget, you get the smallest encode rather than an error. PNG and GIF ignore the budget.
- WebP encodes lossy at the requested quality (default 80); `quality(100)` selects the lossless encode. `quality` has no effect on PNG or GIF output.
- JPEG has no alpha channel. With `fill`, transparent pixels are composited over the fill color; without it, the stored RGB values are used as-is.
- `no_upscale` applies to the whole request wherever it appears: a `300x200` resize of a 200×150 source returns 200×150, scaled proportionally.
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
| `keep_profile` | none | Keep the source ICC profile instead of converting to sRGB; the profile is embedded in JPEG and PNG outputs. Other output formats convert to sRGB (pixels labeled as nothing would be wrong colors). | `filters:keep_profile` |
| `blurfaces` | none | Blur every detected face (privacy redaction). Requires a face model. | `filters:blurfaces` |
| `pixelatefaces` | none | Pixelate every detected face (privacy redaction). Requires a face model. | `filters:pixelatefaces` |
| `blurclass` | `blurclass(person)` | Blur every detected object of the class (privacy redaction). Repeat for more classes. Requires a detection model. | `filters:blurclass(person)` |
| `pixelateclass` | `pixelateclass(car)` | Pixelate every detected object of the class (privacy redaction). Repeat for more classes. Requires a detection model. | `filters:pixelateclass(car)` |
| `facepad` | `facepad(0.5)` — 0 to 3 | Padding for `crop:...:face`: that fraction of the face's size added on every side of the crop window. Default 0.5. | `crop:200x200:face/filters:facepad(1)` |
| `upscale` | `upscale(2)` or `upscale(4)` | AI super-resolution by that factor. Output is capped at 4096 px per dimension. | `filters:upscale(2)` |

```bash
curl "http://localhost:3000/unsafe/filters:removebg:format(png)/https://example.com/product.jpg"
```

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

With `CRABCROP_ONLY_PRESETS=1`, free-form operations are rejected with `400`: requests may use preset references, `exp:` metadata, and bare sources only. The mode does not change signature rules. Combined with `CRABCROP_ALLOW_UNSAFE=1` it makes a public deployment where browsers and CMSs can build URLs from preset names alone, with no secret to hold and no way to invent a transform.

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

So `Accept: image/avif;q=0.5, image/webp` produces WebP: the client's stronger preference beats the server's. If none of the three is offered, the source image's format is kept (JPEG if the format cannot be detected). A format refused with `q=0` is never picked, and wildcards such as `*/*` do not count as an offer. An explicit `format(...)` in the URL always wins over negotiation. Animated GIF and WebP sources are the one exception: they never negotiate to JPEG XL, which would drop their frames; a GIF source keeps GIF instead, and a WebP source keeps WebP. Responses include a `Vary` covering `Accept` and the client hints so downstream caches key this correctly (the hints likewise change sizing and quality; see [Client hints](#client-hints)).

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

Meta responses are served with `Cache-Control: private, no-store`: a URL has two representations (image and JSON), and intermediaries must never pin one in place of the other.

Add the `placeholders` parameter to the media type to also get placeholder data, computed from the final processed pixels. The parameter opts in when it is bare (`; placeholders`), set to `1`, or set to `true` (case-insensitive); any other value, including `0` and `false`, does not. The `detections` parameter below follows the same rule. On the CLI, `crabcrop meta --placeholders` and `crabcrop meta --detections` set them for you:

```bash
curl -H "Accept: application/json; placeholders=1" "http://localhost:3000/unsafe/300x200/https://example.com/photo.jpg"
```

This adds four fields: `dominant_color`, the image's average color as `#rrggbb`; `placeholder`, a 16 px-wide blurred JPEG preview as a base64 data URI; `palette`, the five most prominent colors as `#rrggbb`, most prominent first; and `thumbhash`, a ThumbHash string that ThumbHash client libraries decode into an aspect-correct blurred preview. Handy for rendering a preview while the real image loads. When a CLIP model is configured (see [AI features](#ai-features)), meta responses also carry `tags` (labels with zero-shot scores) and `alt_text` composed from the best labels, whether or not placeholders were asked for. With a detection model configured, adding `detections=1` to the same `Accept` entry adds `detections`: the objects found in the processed image as `{label, score, x, y, w, h}` in pixels, best first. Meta requests are never cached. Errors in meta mode keep the usual status codes, and grammar failures answer with the JSON error envelope described in [Errors](#errors).

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

### Color management and metadata

Pixels are decoded in the source profile's space. When the source embeds an ICC profile (JPEG, PNG, and WebP sources are read), crabcrop converts to sRGB with an embedded-profile color transform before any operation runs, so sizes, crops, and filters all work in one known space; a source with no profile is assumed to be sRGB already. `filters:keep_profile` opts out per request: no conversion, and the source profile is embedded into JPEG and PNG outputs instead. CMYK JPEGs decode through the JPEG decoder's own CMYK conversion (embedded CMYK profiles are not honored yet); grayscale profiles are not transformed.

Metadata follows an operator policy, `CRABCROP_METADATA`:

| Value | Behavior |
| --- | --- |
| `strip` (default) | Outputs carry no source metadata. |
| `preserve` | JPEG outputs gain the source's EXIF, XMP, and Photoshop (IPTC) segments; PNG outputs gain EXIF and XMP. Other output formats do not carry preserved metadata. |

Preservation copies the source segments rather than re-serializing them, with one patch: the EXIF orientation is rewritten to 1, because the pixels were already rotated at decode and the original value would double-rotate in viewers. EXIF orientation itself is always honored at decode, under either policy.

### Fallback image

With `CRABCROP_FALLBACK_IMAGE` set (a URL, or a path to a local file read at startup), a source that cannot be fetched or decoded degrades to the fallback instead of an error: the fallback runs through the same operations and is served with `200` and the header `X-Crabcrop-Fallback: 1`. Signature and parse failures never fall back — a `403` or `400` stays honest — and neither does a source-policy refusal. Fallback responses are never cached, so a recovered origin is picked up on the next request; they are also served with `Cache-Control: no-store`, so a CDN or browser never pins the placeholder in place of the real image. With no fallback configured, error behavior is unchanged.

## Errors

Error bodies are plain text with a short message. Send `Accept: application/json` and a grammar failure (a `400` below) instead answers with the v2 JSON envelope, `{"error":{"code":"invalid_filters","message":"invalid filters segment: filters:bogus(1)"}}`. The codes are the parser's stable names, frozen as published: `empty_path`, `missing_signature`, `malformed_signature`, `invalid_size`, `invalid_crop`, `invalid_crop_rect`, `invalid_pad`, `invalid_expiry`, `unknown_preset`, `invalid_preset`, `presets_only`, `invalid_filters`, `missing_source_url`. The statuses you can get:

| Status | Meaning |
| --- | --- |
| `400` | The path or operations do not parse, or the operation is not supported: a malformed segment, an unknown filter or parameter, `removebg` into a format without alpha, `upscale` past the 4096 px cap, a `crop-rect` outside the image. |
| `403` | Bad signature, an expired `exp:` URL, or the source URL was refused by the source policy (host not in `CRABCROP_ALLOWED_SOURCES`, or resolving to a non-public address while `CRABCROP_BLOCK_PRIVATE_SOURCES` is on). |
| `500` | Encoding the output failed, or an AI model failed at inference time (the runtime errored, or a model session lock was poisoned by an earlier failure). A model that cannot be loaded at all fails startup instead of answering requests. |
| `501` | The URL is valid but this server was not configured for it: an AI filter with no model, a text overlay with no font. |
| `502` | The source could not be fetched (network error, a non-2xx answer from the origin, or a non-image `Content-Type`), the source is larger than the 10 MiB fetch limit, or the fetched bytes could not be decoded as an image — including HEIC sources, AVIF sources in builds without AVIF decoding, and undecodable formats. A watermark that fails to load also lands here, and so does a configured AI model that ran but produced output the server cannot use (no outputs, an unreadable tensor, an unexpected shape). |
| `504` | The source fetch timed out (10 seconds). |

For example, a `removebg` request to a server with no background-removal model returns `501` with the body `removebg is not configured on this server`.

## Endpoints

| Route | Description |
| --- | --- |
| `GET /health` | Health check. Returns `200` with the body `ok`. |
| `GET /metrics` | Prometheus metrics in text format: request counts by status class (`crabcrop_requests_total`), request/load/process durations, cache hit and miss counters, loader and engine error counters, and byte totals. |
| `POST /purge/<signature>/<path>` | Invalidate every stored variant of one result, in both cache tiers. See below. |
| `POST /purge-group/<signature>/<group>` | Invalidate every stored result in a cache group, in both cache tiers. See below. |
| `GET /<signature>/...` | Image requests, as documented above. |

### Purge a cached result

```bash
curl -X POST "http://localhost:3000/purge/<signature>/300x200/https://example.com/photo.jpg"
```

The `<path>` is the original request path without its signature. The signature covers the target exactly as presented; the target is then canonicalized the same way entries are keyed, so purging any equivalent spelling of a path (the `jpg` alias, leading zeros, a preset reference) evicts the shared entry, `exp:` variants included. Compute the signature with your client library (`purge_signature` in the Rust crate and the Python SDK, `purgeSignature` in TypeScript): the HMAC-SHA256 of the path under a purge subkey derived from the secret, so only the secret holder can invalidate, and an image signature can never be replayed as a purge signature. The response is `200` with a body like `purged 2`: the count covers the entry and all of its variants (per format class and DPR). A bad purge signature is a `403` and removes nothing; purging a path that is not stored answers `purged 0`.

### Cache headers and groups

Image responses carry two standard headers. `Cache-Status` (RFC 9211) reports what the server's own caches did: `crabcrop; hit` for a memory hit, `crabcrop; hit; detail="disk"` for a disk hit, and `crabcrop; fwd=uri-miss` when the response was produced from the origin.

A response also belongs to a cache group (RFC 9875) for each preset its URL referenced, listed in the `Cache-Group` header:

```http
Cache-Group: "small"
```

Groups make preset-wide invalidation one call instead of a script over every URL. The signature comes from the same libraries (`purge_group_signature` / `purgeGroupSignature`), which sign a synthetic string on your behalf, so nobody constructs it by hand:

```bash
curl -X POST "http://localhost:3000/purge-group/<signature>/small"
```

The response is `200` with a body like `purged 3`: every entry recorded in the group, across both tiers. A bad signature is a `403` and removes nothing; an unknown group answers `purged 0`.

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

# Convert a Thumbor URL (re-signed; losses are reported on stderr)
crabcrop convert-thumbor "unsafe/300x200/smart/https://example.com/photo.jpg"
```

## Configuration

All configuration comes from environment variables. Invalid values are startup errors, booleans included — the server refuses to start rather than run with a wrong port, cache budget, or safety flag. Boolean variables accept `1`, `true`, `yes`, `on` (any case) for on and `0`, `false`, `no`, `off` for off; anything else fails the boot with the variable named. Names follow one pattern: `*_MODEL` is an AI model artifact, `*_FILE` is a file read at boot, `*_DIR` is a directory, and `*_MAX_MB` is a size budget in megabytes.

| Variable | Default | Description |
| --- | --- | --- |
| `CRABCROP_SECRET` | — (required) | The HMAC secret used to verify signed URLs. The server will not start without it. |
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
| `CRABCROP_AVIF_SPEED` | `8` | ravif speed tier for AVIF encodes, 1 (slowest, smallest output) to 10 (fastest). At 800w and quality 80, speed 8 encodes in about 1 s cold and produces the smallest files of the fast tiers; speed 10 is roughly 3x quicker for about 47% more bytes. |
| `CRABCROP_JXL_SPEED` | `fast` | jixel speed tier for JPEG XL encodes: `ultrafast`, `fastest`, `fast`, `medium`, `slow`, or `extraslow`. Slower tiers shrink output with steep latency costs (`slow` is about 8x the latency of `fast` for about 18% fewer bytes at 800w). |
| `CRABCROP_PRESETS_FILE` | unset | Path to a presets file. See [Presets](#presets). |
| `CRABCROP_ONLY_PRESETS` | off | Rejects free-form operations in requests; only preset references (and `exp:` metadata) are allowed. Requires at least one preset. |
| `CRABCROP_ALLOWED_SOURCES` | unset (all hosts allowed) | Comma-separated host suffixes the loader may fetch from. A suffix matches the host itself or any subdomain of it. |
| `CRABCROP_BLOCK_PRIVATE_SOURCES` | off | Refuses sources whose host is, or resolves to, a loopback, private, link-local, or unspecified address. Checked before connecting and again on every redirect hop. |
| `CRABCROP_S3_ENDPOINT` | unset (AWS) | Base URL of an S3-compatible store for `s3://` sources, path-style. `CRABCROP_S3_REGION` sets the AWS region (default `us-east-1`). Signing credentials come from `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`; unset means unsigned. |
| `CRABCROP_GCS_ENDPOINT` | `https://storage.googleapis.com` | Base URL of the GCS XML API for `gs://` sources. `CRABCROP_GCS_REGION` (default `auto`) is the signing region; the HMAC pair is `CRABCROP_GCS_ACCESS_KEY` / `CRABCROP_GCS_SECRET_KEY`. |
| `CRABCROP_ALLOWED_BUCKETS` | unset | Comma-separated bucket names the `s3://` and `gs://` schemes may read; `*` allows all. Required when bucket credentials are configured (the server refuses to boot without it); without credentials, unset means all buckets, fetched unsigned. |
| `CRABCROP_FILE_ROOT` | unset | Directory that `file://` sources must stay under, canonicalized at boot and per request. Unset disables file sources. |
| `CRABCROP_REMOVEBG_MODEL` | unset | Path to a U²-Net-family ONNX model for the `removebg` filter. See [AI features](#ai-features). |
| `CRABCROP_UPSCALE_MODEL` | unset | Path to a super-resolution ONNX model for the `upscale` filter. See [AI features](#ai-features). |
| `CRABCROP_FACE_MODEL` | unset | Path to a YuNet ONNX model for face-aware `smart` crops, `crop:...:face`, and the face redaction filters. See [AI features](#ai-features). |
| `CRABCROP_CLIP_IMAGE_MODEL` / `CRABCROP_CLIP_TEXT_MODEL` | unset | A CLIP ONNX model pair for meta `tags` and `alt_text`. All four CLIP paths are required together. See [AI features](#ai-features). |
| `CRABCROP_CLIP_TOKENIZER` | unset | The CLIP export's `tokenizer.json`. |
| `CRABCROP_CLIP_LABELS` | unset | Text file with one tag label per line. |
| `CRABCROP_CLIP_THRESHOLD` | `0.05` | Minimum zero-shot score for a label to appear in `tags`. |
| `CRABCROP_DETECT_MODEL` | unset | Path to a YOLOv8-family ONNX model for `blurclass` / `pixelateclass` and meta detections. See [AI features](#ai-features). |
| `CRABCROP_DETECT_LABELS` | unset | Class names file (one per line, in index order) for detection exports that carry no `names` metadata. |
| `CRABCROP_DETECT_THRESHOLD` | `0.5` | Minimum detection score. |
| `CRABCROP_ORT_DYLIB` | unset | Path to the ONNX Runtime shared library, when it is not on the library search path. |
| `CRABCROP_FONT_PATH` | unset | Path to a TTF/OTF font for the `text(...)` overlay filter, loaded at startup. Unset disables the filter. |
| `CRABCROP_FALLBACK_IMAGE` | unset | URL or local file path of the fallback image served when a source cannot be fetched or decoded. See [Fallback image](#fallback-image). |
| `CRABCROP_RESULT_STORAGE_DIR` | unset | Directory for the disk-backed result store. Unset disables the second cache tier. |
| `CRABCROP_RESULT_BUCKET` | unset | S3 bucket for the shared result store (multi-replica third tier). Unset disables it. |
| `CRABCROP_RESULT_PREFIX` | empty | Object prefix namespacing the shared result store's entries. |
| `CRABCROP_RESULT_STORAGE_MAX_MB` | `512` | Disk result store budget in megabytes. `0` disables storage even with a directory set. |

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

Three features run ONNX models in-process: `removebg` (background removal), `upscale` (super-resolution), and face detection for crops and redaction. The models are files you supply — deployment artifacts, like the `Dockerfile.ai` image variant in this repo. crabcrop never downloads a model, at startup or at request time.

How configuration works:

- Point `CRABCROP_REMOVEBG_MODEL` at a U²-Net-family model file (320×320 input; the U2Netp export is a good fit) and/or `CRABCROP_UPSCALE_MODEL` at a super-resolution model. The upscale contract is dynamic-size RGB in (values 0..1), the same image out at an integer factor; Real-ESRGAN-family ONNX exports follow it.
- Point `CRABCROP_FACE_MODEL` at a YuNet model to make `crop:...:smart` center on detected faces (the area-weighted center of the faces found). Face detection only ever improves a smart crop: no model, no faces, or a detection failure all fall back to the detail-based focal point, and a smart crop without the model behaves exactly as before. The same model powers the explicit face operations: `crop:...:face` (crop around the largest face; falls back to the smart window when detection fails or finds nothing), and the `blurfaces` / `pixelatefaces` redaction filters. Those require the model — without one they return `501` — and a detection failure fails a redaction request with `500` rather than serve an unredacted image. Chained face operations in one request share a single detection.
- Point the `CRABCROP_CLIP_*` set at a CLIP model pair (image and text ONNX exports with the projection head), its `tokenizer.json`, and a label file with one label per line, and meta responses gain `tags` and `alt_text`: zero-shot labels scored against the processed image, plus a short alt text composed from the best three. Labels are embedded once at boot, so a meta request runs only the image encoder, behind the same inference gate; a tagging failure omits the fields instead of failing the response. Labels are embedded exactly as written, so phrase-style labels (`aerial view of a beach`) work as well as single words.
- Point `CRABCROP_DETECT_MODEL` at a YOLOv8-family detection export and two more surfaces unlock. The class redaction filters `blurclass(<class>)` and `pixelateclass(<class>)` redact every detected object of a class (repeat the filter for more classes; matching is case-insensitive). They inherit the face redaction rules: no model is a `501`, an unknown class is a `400`, and a detection failure is a `500`, never an unredacted response. And meta requests with `detections=1` return the detection boxes for debugging. The class vocabulary comes from the model's own `names` metadata, which Ultralytics exports carry; for exports without it, `CRABCROP_DETECT_LABELS` supplies one class per line in index order. A model whose output is not the YOLOv8 layout is rejected at boot.
- Models load at boot, before the server reports ready. A configured model that cannot be loaded — missing file, wrong shape — is a startup error with a one-line diagnostic, not a surprise for the first request. The upscale model's native factor is probed at load with a 16×16 test image; a model that does not enlarge by an integer factor is rejected at boot.
- With no model configured, the filter is off: requests that use it fail cleanly with `501` (`removebg is not configured on this server`, `upscale is not configured on this server`). There is no silent fallback — `upscale` never degrades into a plain resize.
- Inference runs behind a server-wide gate of 2 concurrent inferences, so a burst of AI requests cannot starve the normal transform pipeline.
- `upscale(4)` on a server whose model is natively 2× composes two model runs and lands the exact requested size; when the native factor does not divide the request, a final high-quality resize trims the difference. Source transparency is carried across an upscale even though the model only sees RGB.
- The ONNX Runtime library itself comes from `CRABCROP_ORT_DYLIB`, or the system library search path when that is unset.

The `Dockerfile.ai` variant in this repo shows the intended shape: the base image stays free of AI payload, and a second layer adds the pinned ONNX Runtime 1.31.0, the U2Netp model, and the YuNet model, each hash-checked at image build time, and enables the features by setting the variables above. Layer your own super-resolution model the same way and set `CRABCROP_UPSCALE_MODEL` to enable `upscale`.

## Migrating

Guides for moving existing URL templates over:

- [From Thumbor](docs/migrating-from-thumbor.md): concept and operation mapping, plus the `convert-thumbor` verb, which converts and re-signs URLs mechanically.
- [From imgproxy](docs/migrating-from-imgproxy.md): option-by-option mapping tables.
- [From Cloudinary](docs/migrating-from-cloudinary.md): transformation mapping, and what changes when assets become origin URLs.

## Deployment

The repo ships a multi-stage `Dockerfile` (the runtime image carries only the binary and CA certificates):

```bash
docker build -t crabcrop:latest .
docker run -p 3000:3000 -e CRABCROP_SECRET=your-secret crabcrop:latest
```

For the AI features, build `Dockerfile.ai` on top of that image; see [AI features](#ai-features).

On Kubernetes, `deploy/helm` is a Helm chart and `deploy/kustomize` the same workload in kustomize form (base, components for presets, the disk tier, and the AI variant, plus single and multi-replica overlays); `deploy/compose` runs it under Docker Compose. Without Kubernetes, `deploy/ecs-fargate`, `deploy/cloud-run`, and `deploy/container-apps` are per-cloud recipes (task definition, service manifest, and app manifest, each with its secret wiring and a smoke checklist).

Run crabcrop behind a reverse proxy or CDN in production. The application bounds its own work (the processing and AI gates shed overload with `503`), but connection-level protections belong at the edge: request and header timeouts, connection limits, rate limiting, and TLS termination. Restrict `/health` and `/metrics` to your internal network at the proxy: both are unauthenticated (they are served with `Cache-Control: no-store`), and the metrics expose traffic and error counters.

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

## License

MIT. See [LICENSE](LICENSE).
