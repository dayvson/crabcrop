# Filters

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
| `gamma` | `gamma(2.2)`, divisor from above 0 to 10; 1 is unchanged, above 1 brightens midtones, below 1 darkens them | Gamma correction: each channel value `v` becomes `255 * (v / 255)^(1 / g)`, the levels-style midtone gamma. Black and white stay fixed at every setting, and alpha is untouched. Unlike `curve`, no control points need translating: the gamma value from another tool carries over as written. | `filters:gamma(2.2)` |
| `exposure` | `exposure(1.5)`, stops from −10 to 10; 0 is unchanged, positive brightens, negative darkens | Exposure in photographic stops: each channel is decoded to linear light, multiplied by 2 raised to the stops, and encoded back. A stop multiplies, so it behaves the same at every brightness, where `brighten` adds a fixed shift and lifts black along with everything else. Black stays black at every setting, and alpha is untouched. | `filters:exposure(1.5)` |
| `highlights_shadows` | `highlights_shadows(30,-20)`, two percents, each −100 to 100; 0 is unchanged | Adjust the bright and dark ranges independently: each pixel shifts by one offset weighted by its luminance, (1 − Y)² for the shadows argument and Y² for the highlights, up to a quarter of full scale at the extremes. `contrast` cannot do this; it moves both ends together. The shift is shared by all three channels, so hue survives, and alpha is untouched. | `filters:highlights_shadows(30,-20)` |
| `huerotate` | `huerotate(90)`, degrees, integer | Rotate the hue. | `filters:huerotate(90)` |
| `sharpen` | `sharpen(1.5)`, strength, a positive number | Unsharp-mask sharpening. | `filters:sharpen(1.5)` |
| `saturation` | `saturation(1.5)`, 0 to 255; 0 is grayscale, 1 is unchanged | Scale color saturation. | `filters:saturation(0.5)` |
| `vibrance` | `vibrance(40)`, percent, −100 to 100; 0 is unchanged | Saturation weighted toward the muted colors: each pixel's factor is `1 + (percent / 100) × (1 − s)` for its own saturation `s`, so dull colors gain the most and vivid colors (and skin tones) barely move, where `saturation` scales every pixel by the same factor. Negative values drain the muted colors first. | `filters:vibrance(40)` |
| `rgb` | `rgb(10,-5,0)`, per-channel offsets, each −255 to 255 | Add offsets to the red, green, and blue channels. | `filters:rgb(10,-5,0)` |
| `curve` | `curve(rgb,0,0,255,255)`, a channel (`rgb`, `r`, `g`, or `b`), then 2 to 16 `x,y` control points, each 0–255 with x strictly increasing | Remap tones through the control points, linearly interpolated; inputs outside the points' x range clamp to the nearest end value. Repeat the filter to shape channels independently. | `filters:curve(r,0,64,255,255)` |
| `convolution` | `convolution(0,-1,0,-1,5,-1,0,-1,0,0)`, 9 (3×3) or 25 (5×5) kernel values in row-major order, each a plain decimal from −1024 to 1024, then a normalize flag (0 or 1) | Apply the kernel once over the image; edges replicate. With the flag on, the kernel is scaled by the sum of its entries first (an all-ones kernel averages); a zero sum applies the kernel as written. | `filters:convolution(0,-1,0,-1,5,-1,0,-1,0,0)` |
| `noise` | `noise(40)`, strength, an integer from 0 to 100, percent of full scale | Monochrome film grain: one noise value per pixel, uniform within ±strength percent of full scale, added equally to the red, green, and blue channels (alpha untouched). The field is deterministic, seeded from the image content and the amount, so the same request always returns the same bytes and caches soundly. `noise(0)` changes nothing. On animated sources it applies per frame, seeded from each frame's own content, so the frames grain independently. | `filters:noise(40)` |
| `equalize` | none | Histogram equalization, per channel. | `filters:equalize` |
| `enhance` | none | Gentle per-channel auto-levels: each channel's 0.5th percentile maps to 0 and its 99.5th to 255, stretching a compressed range toward the ends; a channel whose percentile span is under 16 levels is left alone. Deliberately weaker than `equalize`, which redistributes the whole histogram. | `filters:enhance` |
| `rotate` | `rotate(45)`, angle in degrees, from -36000 to 36000, decimals allowed | Rotate the image. Multiples of 90 are exact quarter turns; other angles expand the canvas to fit, filling the uncovered corners with the `fill` color (white by default) or transparency when the image carries alpha. | `filters:rotate(-90)` |
| `trim` | `trim`, `trim(20)`, or `trim(20,5,0,5,0)`: tolerance 0–255, default 10; the five-argument form adds pixels to remove from the left, top, right, and bottom | Crop away the border that matches the top-left corner color within the tolerance. With five arguments, the named pixels come off each side of what the tolerance pass leaves, clamped so at least one pixel survives on each axis. On an animated source every frame takes the same cut: the smallest box holding what each frame's own pass would keep, so the frames share one canvas and stay in register. | `filters:trim(20)` |
| `sepia` | none | The classic warm sepia tone. | `filters:sepia` |
| `duotone` | `duotone(1a1a40,ffd9a0)`, two colors, six hex digits each | Map shadows to the first color and highlights to the second, by luminance. | `filters:duotone(1a1a40,ffd9a0)` |
| `monochrome` | `monochrome(ff8000)`, six hex digits | Single-hue rendering: each pixel's luminance scaled by the color, so full luminance renders exactly that color. | `filters:monochrome(ff8000)` |
| `pixelate` | `pixelate(8)`, block size in pixels, from 1 to 16383 | Mosaic the image into square blocks, each painted with the average of its pixels. | `filters:pixelate(8)` |
| `flip` | `flip(h)` or `flip(v)` | Mirror the image horizontally or vertically, at this point in the chain. | `filters:flip(h)` |
| `blur_region` | `blur_region(x,y,w,h)`, pixels | Gaussian-blur one rectangle (license plates, screens, documents). Coordinates address the image at this point in the chain; blur strength scales with the region. A region that overflows the image is an error, not a partial redaction. | `filters:blur_region(120,40,180,60)` |
| `pixelate_region` | `pixelate_region(x,y,w,h)`, pixels | Pixelate one rectangle, with the same bounds rule as `blur_region`. Block size scales with the region. | `filters:pixelate_region(120,40,180,60)` |
| `remove_region` | `remove_region(x,y,w,h)`, pixels | Erase one rectangle and inpaint it from the surrounding pixels (AI, LaMa-family model). Same bounds rule as `blur_region`. The fill is fabricated by the model: plausible on smooth surroundings, visibly wrong on structured content. See [AI features](ai-features.md#ai-features). | `filters:remove_region(120,40,180,60)` |
| `radius` | `radius(24)`, pixels, each radius from 0 to 16383; or four values `radius(24,24,0,0)` in CSS corner order (top-left, top-right, bottom-right, bottom-left) | Round the corners, antialiased. Radii larger than the frame shrink proportionally so corners never overlap. The output format must carry alpha: `png`, `webp`, `jxl`, or `avif`. | `filters:radius(24):format(png)` |
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
| `format` | `format(webp)`, `jpeg` (or `jpg`), `png`, `webp`, `jxl`, `avif`, `gif`, `mp4`, `webm` | Output format. Overrides `Accept` negotiation. The video formats are explicit-only (never negotiated) and need an animated source plus a configured encoder; see [Video output](media.md#video-output). A format also takes its own encoder options as extra arguments: `format(jpeg,progressive)`, `format(jpeg,subsampling:444)` (444, 422, or 420; 420 is the standing behavior), `format(png,interlace)` (Adam7 interlacing; combines with `palette`), `format(webp,method:6)` (libwebp compression method 0 to 6; 4 is the standing behavior; on animated output method 6 takes about 16x the encode time of method 4, and method 2 is about twice as fast as method 4 for about 9% more bytes, see [Source images](url-grammar.md#source-images)), `format(webp,sns:50)` (smart subsampling strength 0 to 100). AVIF, JPEG XL, GIF, and the video formats take no options: the AVIF encoder writes 4:4:4 only and exposes no subsampling control. An option an encoder cannot honour is a 400, never silently ignored, and a format with no options renders byte for byte as it always has. | `filters:format(avif)` |
| `quality` | `quality(80)`, 1 to 100, or `quality(auto)` | Encoding quality for JPEG, JPEG XL, AVIF, and WebP. Default 80. One URL can carry a value per format: `quality(80,jpeg:90,webp:75)` takes a bare fallback value plus one `format:value` entry per format (`format:auto` works too), resolved after format negotiation, so the same signed URL serves every negotiated format at its intended quality. A format with no entry and no fallback behaves as if no quality was named, except that `Save-Data` does not step it (see [Client hints](#client-hints)). | `filters:quality(auto)` |
| `palette` | `palette(64)`, colors from 2 to 256 | Write PNG output as an indexed (palette) image with at most that many colors, quantized and dithered with the same quantizer the GIF encoder uses. Palette entries keep their alpha (carried in the file's `tRNS` chunk), so translucency survives, unlike GIF's binarized transparency. Other formats ignore the directive, the same stance `quality` takes on PNG. This is the PNG8 form other tools select: `palette(256)` is a full PNG8 palette. | `filters:format(png):palette(64)` |
| `max_bytes` | `max_bytes(50000)`, byte budget, at least 1 | Re-encode at decreasing quality until the output fits the budget. | `filters:max_bytes(50000)` |
| `fill` | `fill(ff0000)`, six hex digits, RGB | Background color used to flatten transparency when encoding to JPEG. | `filters:fill(ffffff)` |
| `no_upscale` | none | Cap every resize in the request at the source dimensions. | `filters:no_upscale` |
| `dpr` | `dpr(2)`, one of `1`, `1.5`, `2`, `3` | Scale size and crop targets by the device pixel ratio, so one URL serves standard and retina displays. | `filters:dpr(2)` |

Notes on the directives:

- `quality(auto)` picks a quality per image, from 50 for flat images to 95 for detailed ones, based on content analysis. For JPEG and JPEG XL the pick is then verified: crabcrop encodes, decodes the result, and scores it against the processed source with SSIMULACRA2, adjusting quality (at most 4 encodes) until the score lands at the target, 75 by default and configurable with `CRABCROP_SSIM_TARGET`. The achieved score is reported in meta responses as `ssimulacra2`. Verification happens on cache misses only, but it is real work: scoring a full-size image takes about a second. When crabcrop picks the quality, for `quality(auto)`, or when `max_bytes` steps quality down, the response carries an `X-Crabcrop-Quality` header with the value used.
- `max_bytes` can only steer formats with a quality setting: JPEG, JPEG XL, AVIF, and WebP. If quality reaches its floor of 1 and the output is still over budget, you get the smallest encode rather than an error. PNG and GIF ignore the budget. On a server that signs outputs (see [Provenance (C2PA)](meta.md#provenance-c2pa)), the manifest is added after the budget is met. Its size depends on your certificate chain and, when `CRABCROP_C2PA_TSA_URL` is set, on the timestamp (about 4 KB with a small EC chain and no timestamp), not on the source, so leave that much room in a tight budget.
- WebP encodes lossy at the requested quality (default 80); `quality(100)` selects the lossless encode. `quality` has no effect on PNG or GIF output.
- JPEG has no alpha channel. With `fill`, transparent pixels are composited over the fill color; without it, the stored RGB values are used as-is.
- `no_upscale` applies to the whole request wherever it appears. A plain resize clamps each axis at the source dimension: a `300x200` resize of a 200×150 source returns 200×150, and a `300x100` resize returns 200×100 (the axis that shrinks is honored as written). Fit-in and crop targets shrink proportionally instead, since their dimensions already carry the image's proportions.
- `dpr` also applies request-wide, and it scales targets *before* the `no_upscale` cap sees them, so `dpr` never invents pixels past the source when both are set. Targets round to whole pixels. `crop-rect` and `focal` coordinates address the image as it stands at their segment (source pixels when nothing runs before them) and are not scaled. When the URL carries no `dpr`, the server honors the `DPR` client hint header (same set of values; an invalid hint is ignored). An explicit `dpr` in the URL always beats the header. A hinted factor that would push a size past the output limits steps down instead of failing; an explicit one does not (see [Client hints](#client-hints)).

### Client hints

When a URL sizes nothing itself (no size, crop, fit-in, pad, or crop-rect operation), the server honors the width client hints so responsive pages do not have to mint a URL per width. It reads, in order, `Sec-CH-Width`, `Width`, `Sec-CH-Viewport-Width`, and `Viewport-Width`, and resizes to that width as if the URL had carried the width-only size op plus `no_upscale` (the `DPR` hint still multiplies on top, under the same cap). The size op, and the `no_upscale` when the URL has none of its own, count toward the [100-operation cap](url-grammar.md#how-requests-work). When they would take the request past it, the server ignores the width hint and answers as it would without one, so a width hint never turns a URL the cap accepts into a `400`. Only a URL with 99 or 100 operations, counted after presets and default operations, is affected. A hint never enlarges a raster source: on a 640×480 source, `Sec-CH-Width: 3000` returns 640×480, with or without `Sec-CH-DPR: 2`. An SVG source still renders at the hinted width, because it is rasterized at the size the request asks for and a vector has no pixels to invent (the same as with an explicit `no_upscale`). Hinted widths are quantized up to a multiple of `CRABCROP_WIDTH_STEP` (default 100) and stop at 3840, so nearby viewports share one cache entry and every hint past 3840 shares the 3840 entry; the quantized width joins the cache key. Image responses advertise `Accept-CH` for these hints and vary on them, unless `CRABCROP_CLIENT_HINTS=0` turns hints off.

The `Sec-CH-DPR` and `DPR` hints scale every pixel size in the URL when it carries no `dpr(...)`, and take the same values: 1, 1.5, 2, or 3. The value is read as a number and must equal one of them; any other value is ignored, so a device ratio such as 2.625 gets no factor at all rather than the nearest one. A DPR hint is advisory and never fails a URL on the output size limits: when the hinted factor alone would push a size past the [output limits](url-grammar.md#output-size) (16383 px per axis, 100 million pixels per step), the server steps it down to the largest of those values whose sizes fit both, and serves the image at that factor:

```text
/unsafe/6000x1/<source>     DPR: 3     200, 12000x2: 18000x3 is past the per-axis limit, so the factor steps to 2
/unsafe/8192x1/<source>     DPR: 2     200, 12288x2: 16384x2 is past it, so the factor steps to 1.5
/unsafe/8000x8000/<source>  DPR: 1.5   200, 8000x8000: 144 million pixels is past the budget, so the factor steps to 1
```

If even factor 1 does not fit, the URL itself is over the limits, and it answers the same `400` with or without the hint. A `dpr(...)` written in the URL, in a preset, or in the server's [default operations](configuration.md#server-wide-default-operations) is never stepped down: it is the request's own factor, so a size it pushes past the limits answers `400` (`unsupported_operation`). The response is cached under the factor the hint asked for, so a stepped-down render answers that hint again and never the same URL without a hint (responses vary on both DPR headers). The server sends no `Content-DPR` header: a client that needs the factor applied reads it off the image's dimensions. The server settles the factor from the source's header before it decodes anything, so a factor the header already shows past the limits costs nothing. A size the header cannot show (after a `trim`, from an SVG or a source with no stated size, an overlay, or the budget an animation's frames share) is found by rendering: each such try decodes the source again and stops at the step that asks for the oversized image, before it is allocated. Limits an operation keeps for itself are not output size limits and are not stepped: a hinted factor that takes an `upscale(n)` input past that filter's own caps answers its `400`. [`/explain`](endpoints.md#self-explaining-urls) reads the DPR hint too and forecasts the factor the image request applies, up to the sizes only a render finds.

A request with `Save-Data: on` gets the default encode quality stepped down one notch, from 80 to 70, on the formats that carry a quality (JPEG, JPEG XL, AVIF, WebP); the stepped value is reported in `X-Crabcrop-Quality`. A URL with any `quality(...)` in force (its own, a preset's, or the default operations') is unaffected, a per-format list included, even one with no value for the output format; stepped responses are cached separately.

A response any of these hints applies to (one the hint gives its own cache entry, even where its bytes come out the same) is served with `Cache-Control: private` instead of `public`, with the same lifetime, so a shared cache that keys on the URL alone never hands one requester's hinted variant to everyone else (see [Client hints and shared caches](responses.md#client-hints-and-shared-caches)). `CRABCROP_CLIENT_HINTS=0` switches all of them off: the headers are ignored, responses vary on `Accept` alone, and `Accept-CH` is not sent.

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

The server fetches each watermark through the same loader, limits, and source policy as source images. A watermark that fails to load or decode fails the request (`502`, or `400 invalid_source` when its URL is not one the server can fetch at all) instead of being skipped.

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

**AI filters** need a model configured on the server. See [AI features](ai-features.md#ai-features).

| Filter | Parameters | Description | Example |
| --- | --- | --- | --- |
| `removebg` | none | Remove the background, keeping the subject over transparency. Output format must carry alpha: `png`, `webp`, `jxl`, or `avif`. | `filters:removebg:format(png)` |
| `studio` | `studio(product,...)` or `studio(scene,...)` | Studio mode: `product` composites the subject onto a backdrop with a synthesized contact shadow, scaled and centered to a pad margin; `scene` keeps the photograph's scene, reframing it on the detected subject and correcting light and color. See [Studio mode](#studio-mode). Requires the removebg model. | `filters:studio(product,bg:white)` |
| `keep_profile` | none | Keep the source ICC profile instead of converting to sRGB; the profile is embedded in JPEG and PNG outputs. Other output formats convert to sRGB (pixels labeled as nothing would be wrong colors), and so does a profile that is malformed or larger than 256 KiB, which is never embedded. | `filters:keep_profile` |
| `blurfaces` | none | Blur every detected face (privacy redaction). Requires a face model. | `filters:blurfaces` |
| `pixelatefaces` | none | Pixelate every detected face (privacy redaction): each face becomes a mosaic of averaged cells, at most 8 across its longer side and never under 2 px, so even a face a few pixels wide is redacted. Requires a face model. | `filters:pixelatefaces` |
| `blurclass` | `blurclass(person)` | Blur every detected object of the class (privacy redaction). Repeat for more classes. Requires a detection model. With a segmentation model, the blur follows the object's silhouette instead of its box. | `filters:blurclass(person)` |
| `pixelateclass` | `pixelateclass(car)` | Pixelate every detected object of the class (privacy redaction). Repeat for more classes. Requires a detection model. With a segmentation model, the pixelation follows the object's silhouette instead of its box. | `filters:pixelateclass(car)` |
| `recolorclass` | `recolorclass(car,ff0000)` | Paint every detected object of the class a solid color, alpha preserved. Repeat for more classes. Requires a detection model. With a segmentation model, the paint follows the object's silhouette instead of its box. | `filters:recolorclass(car,ff0000)` |
| `facepad` | `facepad(0.5)`, 0 to 3 | Padding for `crop:...:face`: that fraction of the face's size added on every side of the crop window. Default 0.5. | `crop:200x200:face/filters:facepad(1)` |
| `upscale` | `upscale(2)` or `upscale(4)` | AI super-resolution by that factor. Output is capped at 4096 px per dimension. One model run takes at most 262,144 input pixels (the image itself, or the intermediate between the composed runs of an `upscale(4)`); a larger input is a `400`, so resize the source first. | `filters:upscale(2)` |

```bash
curl "http://localhost:3000/unsafe/filters:removebg:format(png)/https://example.com/product.jpg"
```

Redaction replaces pixels. `blurfaces`, `pixelatefaces`, `blurclass`, `pixelateclass`, `blur_region`, and `pixelate_region` write the blurred or pixelated result, alpha included, in place of the pixels they cover; they never composite it over the original, so the source's alpha cannot let the original through. (With a segmentation model, a silhouette's antialiased edge mixes the redacted and the original pixels by the mask's strength, as it does for an opaque source.) This matters for sources with transparency: detection reads color only, so a face stored at alpha 0 is still found, and JPEG output, which has no alpha, writes the stored color. Under any alpha the stored color is the redacted one, the same an opaque source gets. An area of uniform alpha keeps that alpha exactly, so an opaque image stays opaque; where the alpha varies, as at a cut-out's edge, it is blurred or pixelated along with the color. Under `CRABCROP_METADATA=preserve`, the metadata copied into a JPEG or PNG output carries no EXIF or Photoshop thumbnail of the source. The XMP packet is copied as is, so a thumbnail in it stays and can still show what a redaction covered: see [Color management and metadata](responses.md#color-management-and-metadata).

### Studio mode

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
- Meta responses report the run in a `studio` field: the resolved backdrop and where it came from (`argument`, `brand`, or `default`), the shadow and pad applied, the steps that ran, any fallbacks taken (the brand resolution, the no-subject fallback), and the mask confidence when the provider exposes one. See [Meta responses](meta.md#meta-responses).

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
