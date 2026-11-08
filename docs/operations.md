# Operations

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
| `smart` | A window centered on the focal point: a detected face when a face model is configured (see [AI features](ai-features.md#ai-features)), otherwise the point of greatest local detail in the image. When faces are detected, the window shifts as needed to keep them whole: a position keeping every face fully inside wins, otherwise the position keeping the most face area whole (larger faces win), and a face larger than the window stays centered. The crop record behind `debug:focal` counts the faces detected and the faces the window keeps whole. |
| `face` | A window around the largest detected face, expanded by the `facepad` padding (below). Requires a face model; with no faces (or a detection failure) it falls back to the `smart` window. |
| `objects(<class>[,<class>...])[:<scale>]` | A window around the union of every detected object of the named classes, the union box scaled by `<scale>` about its center (below). Requires a detection model; with none of the classes detected (or a detection failure) it falls back to the plain `center` crop of the same size. |

```bash
curl "http://localhost:3000/unsafe/crop:1200x630:top/https://example.com/photo.jpg"
```

The objects form names one or more detection classes (the class filters' vocabulary, matched case-insensitively, duplicates folded), up to 32 distinct classes. A repeat counts once, and a list that names a 33rd distinct class is a `400 invalid_crop`, never a crop to the first 32:

```text
crop:800x800:objects(dog)
crop:1200x630:objects(person,car):1.5
```

The crop window is the smallest region of the requested aspect that contains the union box scaled by `<scale>` (default 1, the tight box; above 1 adds margin around the subject, below 1 tightens into it), centered on the union's center and clamped into the frame. The canonical form always spells the scale. The mode needs a detection model (`CRABCROP_DETECT_MODEL`, see [AI features](ai-features.md#ai-features)): with none configured it answers `501`, never a silent plain crop, and a class outside the model's vocabulary is a `400`. The fallback is only for a model that ran and found nothing matching. Meta responses report the decision under `objects_crop`: the classes, the scale, the path taken (`objects` or `fallback`), the detections used, and the union box. The record is omitted if any value in it has no JSON representation (a non-finite score or scale), rather than printing a malformed body. Under `debug:focal`, the detections that fed the crop are amber and the union box is the magenta focal rectangle.

```bash
curl "http://localhost:3000/unsafe/crop:800x800:objects(dog):1.5/https://example.com/pets.jpg"
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

### Focal point

```text
focalpoint:<x>x<y>[:<zoom>]
```

Names the point the next `crop:` in the URL centers on, as fractions of the image dimensions (`0.75x0.4` is three quarters across, two fifths down), with an optional zoom factor. Like the focal rectangle, the segment produces no output of its own; the crop consumes it, a `center` or `smart` crop positions on it, and the directional alignments and `face` ignore it. When a URL carries both focal forms, the last one before the crop wins. At zoom 1 the crop is the ordinary fill crop positioned on the point instead of the frame center. Above 1 the crop window tightens around the point (zoom 2 halves each axis, keeping the requested output size, so the crop renders from fewer source pixels); below 1 it loosens toward the fill window, which it can never exceed, so a small zoom converges on the zoom-1 crop. The zoom runs from above 0 to 100 and defaults to 1; the canonical form always spells it. Because the point is fractional, it addresses whatever image is in effect at that point in the chain and can never fall outside it, so there is no bounds failure. This is the form to migrate imgix focal-point crops (`fp-x`, `fp-y`, `fp-z`) to: the fractions and the zoom carry over as written.

```bash
curl "http://localhost:3000/unsafe/focalpoint:0.75x0.4:2/crop:800x600/https://example.com/photo.jpg"
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

Frame extraction is an operator feature: it runs through the external decoder configured with `CRABCROP_VIDEO_DECODER`, and a server without one answers `501`. See [Video frames](media.md#video-frames) for the decoder contract and the failure table.

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

Page rendering is an operator feature: it runs through the external renderer configured with `CRABCROP_PDF_RENDERER`, and a server without one answers `501`. See [PDF pages](media.md#pdf-pages) for the renderer contract and the failure table.

```bash
curl "http://localhost:3000/unsafe/page:2/800x1000/filters:format(webp)/https://example.com/report.pdf"
```

## Presets

A preset names an operation string on the server, so URLs stay short and the transformation menu stays operator-controlled. Define presets in a file, one per line (blank lines and lines starting with `#` are skipped):

```text
# /etc/crabcrop/presets.txt
og-card = crop:1200x630:smart/filters:format(webp):quality(80)
thumb = 300x200
```

Point `CRABCROP_PRESETS_FILE` at it, and requests can use `preset:<name>` anywhere an operation fits:

```bash
curl "http://localhost:3000/<signature>/preset:og-card/https://example.com/photo.jpg"
```

The signature covers the preset reference; the server expands it deterministically after verification, and cache entries key on the expanded operations. Treat a preset as a stable template: **never edit a definition in place**. Roll a change out under a new name (`thumb2`) and update your templates to reference it. Editing in place changes the bytes behind URLs already signed and shared, and image responses travel with `Cache-Control: immutable` for a year, so browsers and CDNs holding the old response keep serving it, beyond the reach of any purge. (This server's own caches are not the problem: they key on the expanded form, so they would pick the edit up. Downstream copies are the ones stranded.) Purging accepts the preset URL itself, exactly as an operator holds it; the server expands it before matching entries, and `/purge-group` clears a preset's whole output set at once. An unknown preset name is a `400`. Preset definitions are validated at startup: malformed entries refuse the boot, and presets cannot contain a source segment, an `exp:`, a `maxage:`, another preset, or more than the 100 operations a request may run (each filter in a `filters:` segment counts once; see [the URL grammar](url-grammar.md#how-requests-work)).

With `CRABCROP_ONLY_PRESETS=1`, free-form operations are rejected with `400`: requests may use preset references, `exp:` and `maxage:` metadata, and bare sources only. The mode does not change signature rules, and it applies to every request, signed ones included. Combined with `CRABCROP_ALLOW_UNSAFE=1` it makes a public deployment where browsers and CMSs can build URLs from preset names alone, with no secret to hold and no way to invent a transform.

Unsafe mode without presets-only is not a public mode. With `CRABCROP_ALLOW_UNSAFE=1` alone, the server honors any unsigned operation chain on any source the loader may fetch (every public host, unless `CRABCROP_ALLOWED_SOURCES` sets an allowlist): configured presets constrain nothing, because a client can leave the preset out or append any operation after it. That deployment is an open image proxy and a CPU sink, so keep it to local development or behind access control of your own.

Which mode fits depends on where URLs are built:

| Server mode | Accepts | Fits |
| --- | --- | --- |
| Signed (`CRABCROP_ALLOW_UNSAFE` and `CRABCROP_ONLY_PRESETS` off) | Any operations, with a valid signature | URLs built where the secret lives: server code and build steps |
| Presets-only (`CRABCROP_ONLY_PRESETS=1` with `CRABCROP_ALLOW_UNSAFE=1`) | Preset references, `exp:` and `maxage:`, and bare sources; anything else answers `400`, signed or not | URLs built anywhere from preset names alone, with the size inside the preset (`card-640 = 640x0/filters:format(webp)`) |
| Unsafe alone (`CRABCROP_ALLOW_UNSAFE=1`, presets-only off) | Any operations, signed or not | Local development only |

Because presets-only mode applies to signed requests too, one server cannot take both presets-only browser URLs and signed URLs with free-form sizing. The framework integrations in [`frameworks/`](../frameworks) append per-width sizing after the preset, which presets-only mode rejects; their READMEs describe the signed setup for production.
