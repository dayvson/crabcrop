# Migrating from Gumlet

Gumlet is a hosted service: you point a source at your storage, and transformations ride on the delivery URL as query parameters. crabcrop is a server you run in front of the same storage, and transformations are path segments in a signed URL. The assets do not move; the URL templates do. There is no mechanical Gumlet converter (the `convert-thumbor` verb covers Thumbor), so templates translate by hand using the tables below. Each row is marked direct, approximate, or unsupported, and the adjustment rows say exactly when a number changes meaning in translation, because copying those verbatim is how a migration ships wrong renders nobody notices.

Gumlet's parameters have long and short spellings (`width` and `w`, `quality` and `q`, `format` and `fm`, `brightness` and `bri`). The tables use whichever spelling Gumlet's own examples use; the mapping is the same for both.

## Concepts

| Gumlet | crabcrop |
| --- | --- |
| A source (S3, GCS, Azure, Web Folder, Web Proxy) bound to a Gumlet subdomain or a custom domain; the URL path is the asset's key under the source | The full source URL spelled out in every request, fetched from your origin at request time |
| Default parameters on a source: a parameter set in the dashboard that applies to every URL from that source, with query parameters overriding it | No per-source defaults. The closest shapes are presets (named operation strings, referenced per URL as `preset:<name>`) and server configuration |
| Named presets | Gumlet has none: there is no way to name a transformation and reference it from a URL. Dashboard default parameters are the only reuse mechanism, and they are invisible in the URL |
| An Imgix-type source, which accepts imgix spellings (`fit`, `txt`, `txt-align`, and the rest of Gumlet's imgix list) | Those templates follow the imgix guide in this series, not this one; this guide covers Gumlet's native parameters |
| Parameters in the query string, order-free, applied in Gumlet's own pipeline order (`extract` before resize, `pad` after) | Operations as path segments, applied in URL order: geometry first, `filters:` last |
| Secure URLs optional per source: MD5 of the source's secure token and the URL, appended as `s=` | Every URL signed with hex HMAC-SHA256 (`CRABCROP_SECRET`), or `unsafe` behind `CRABCROP_ALLOW_UNSAFE` |
| `expires=<unix-seconds>` query parameter; past it, Gumlet answers 403 | An `exp:<unix-seconds>` segment inside the signed path; past it, crabcrop answers 403 |
| No `format` parameter: the source format is served. `format=auto`: Gumlet negotiates, JPEG XL first | Negotiation is built in and always on unless `format(...)` pins one: AVIF, JPEG XL, then WebP, by `Accept` q-values |
| Gumlet.js, which rewrites image URLs in the page and sizes them from the rendered element and device pixel ratio | The crabcrop SDKs and the `crabcrop-url` crate (`build_url`, `build_srcset`); client hints are read natively, so hint-driven sizing needs no script |
| Purge from the dashboard or API, per URL; purging an entire source is plan-gated | `POST /purge/...` per URL, `POST /purge-group/...` per preset, or the CLI verbs, clear crabcrop's own cache tiers |
| The video platform, uploads, and source management | Out of scope: crabcrop is delivery-time image transformation only |

## Signing

The schemes do not convert, in either direction. A Gumlet signature is an MD5 of the source's secure token, the path, and the query string exactly as written, carried in the URL as `s=`. A crabcrop signature is an HMAC-SHA256 hex string over the path remainder, carried as the first path segment. Pick a `CRABCROP_SECRET` and re-sign templates with the SDKs or `crabcrop sign`.

Three operational differences matter. First, enforcement: Gumlet's Secure URLs is a per-source toggle, off by default, and turning it on makes every unsigned or altered URL on that source answer 403; crabcrop signs everything unless the server deliberately runs with `CRABCROP_ALLOW_UNSAFE=1`. Second, expiry behaves alike for once: Gumlet's `expires` timestamp sits inside the signed query string, so it copies exactly into `exp:`, and both systems answer 403 past it. Gumlet also caps the response's cache lifetime at the remaining time to expiry on valid requests; crabcrop leaves caching to its normal rules. Third, rotation: the Gumlet token is per source, and rotating it invalidates that source's URLs with no overlap; crabcrop's optional `<key-id>:` signature prefix exists so rotation can overlap two secrets instead.

## Resizing and cropping

Gumlet splits the sizing decision across three parameters: `mode` chooses the strategy (`fit` by default), `crop` chooses the anchor when `mode=crop`, and `enlarge` decides whether a smaller source may be scaled up. One Gumlet behavior drives several rows below: `enlarge` defaults to `false` for `crop` and `stretch` and cannot be overridden there, so a Gumlet crop or stretch never upscales. crabcrop's `crop:` and exact `WxH` do upscale, so faithful translations of those modes add `filters:no_upscale`.

| Gumlet | crabcrop | Mapping |
| --- | --- | --- |
| `w=300` alone (or `h=` alone) | `300x0` | Direct: one axis set, the other proportional |
| `w` + `h` with `mode=fit` (the default) | `fit-in:WxH` | Direct: fit inside the box, aspect kept |
| `mode=max` | `fit-in:WxH` | Direct: Gumlet documents `fit` and `max` as the same fit-inside behavior |
| `mode=fit` or `mode=max` with `enlarge=false` | `fit-in:WxH` plus `filters:no_upscale` | Direct: fit inside, never upscale |
| `mode=stretch` | `WxH` plus `filters:no_upscale` | Direct, with the flag doing real work: `enlarge` is forced off for stretch, so a smaller source keeps its size on the short axis instead of being distorted up to the box |
| `mode=crop` (centered) | `crop:WxH` plus `filters:no_upscale` | Direct: crop-to-fill in both, and `no_upscale` reproduces the forced `enlarge=false` |
| `crop=top`, `bottom`, `left`, `right`, `center` | `crop:WxH:top` and the same-named alignments | Direct |
| `crop=topleft`, `topright`, `bottomleft`, `bottomright` | `crop:WxH:top-left` and the same-named alignments | Direct, with a spelling change: crabcrop alignments use hyphens, Gumlet does not |
| `crop=entropy` | `crop:WxH:smart` | Direct in intent: both anchor on the point of greatest detail. With a face model configured, crabcrop's `smart` also keeps detected faces whole |
| `crop=smart` | `crop:WxH:smart` | Direct in intent: Gumlet anchors on luminance frequency and skin tones, crabcrop on saliency, plus faces when a face model is configured. Expect close, not identical, windows |
| `crop=faces` | `crop:WxH:face` | Approximate: Gumlet centers the crop on all detected faces; crabcrop frames the largest detected face and requires `CRABCROP_FACE_MODEL`. `crop:WxH:smart` is the alternative when keeping every face whole matters more than centering on them |
| `crop=focalpoint` with `fp-x`, `fp-y` | `focal:<rect>` before `crop:WxH:smart` | Approximate, and only with source dimensions in hand: the fractions become pixels, and a point becomes a small rectangle around it (see the worked example). `fp-z` (1 to 10) shrinks that rectangle to zoom |
| `mode=min` (cover the box, no cropping, output may exceed it) | No equivalent | Unsupported: crabcrop has no cover-without-crop size. Compute the covering size in the template when the source aspect is known, or accept `crop:WxH`, which crops: different pixels |
| `mode=fill` with `fill=solid` and `fill-color` | `pad:WxH:<color>` | Approximate: Gumlet may upscale the image into the box first (fill is not a no-enlarge mode); crabcrop's `pad` never upscales. `fillmax` below is the exact match |
| `mode=fillmax` with `fill=solid` and `fill-color` | `pad:WxH:<color>` | Direct: fit without upscaling, centered on the exact canvas |
| `mode=fill` or `fillmax` with `fill=blur` | `pad:WxH:blur` | Direct: ambient blurred bars in both. Gumlet's `fill-blur` intensity (1 to 100) has no equivalent; the blur strength is not a knob here |
| `mode=fill` or `fillmax` with `fill=background` (the top-left corner color) | No equivalent | Unsupported: a `pad` fill is a hex color known when the template is written, or `blur`. A corner color depends on the image |
| `ar=16:9` with `w` or `h` (only with `mode=crop`) | Compute the other axis, or `crop:WxH` at that ratio | Approximate: the ratio is template arithmetic; there is no `ar` parameter |
| `extract=100,300,300,300` (`rect`), as `left,top,width,height` | `crop-rect:100x300:400x600` | Direct, after the corner math: crabcrop takes two corners, Gumlet a corner and a size. Both address the original image, before resizing |
| `pad=30` or `pad=40,20,40,30` (per-edge pixels, colored by `background`) | Resize smaller, then `pad:WxH:<color>` | Approximate: Gumlet adds padding around the result; crabcrop pads to an exact canvas, so the box math is yours |
| `mask=ellipse` | `filters:mask(ellipse)` | Direct; `mask=circle` is the equal-axes case. Pin an alpha-carrying `format(...)` if the page needs the cutout transparent; Gumlet fills the masked area with the background color on formats without alpha, crabcrop makes it transparent and lets the format decide |
| `mask=corners` with `rpixels=50` | `filters:radius(50)` | Direct: pixels on both sides |
| `dpr=2` | `filters:dpr(2)`, or the `DPR` / `Sec-CH-DPR` request headers | Approximate: Gumlet accepts any float from 1.0 to 3.0; crabcrop accepts exactly 1, 1.5, 2, and 3. In-between values like `dpr=2.5` are parse errors here, not renders (see the differences below) |
| Client hints (`Sec-CH-Width`, `Sec-CH-DPR`, `Sec-CH-Viewport-Width`) | Nothing to translate | Direct: when a URL sizes nothing itself, crabcrop honors the width and DPR hints natively, quantized by `CRABCROP_WIDTH_STEP` (default 100), and reads `Save-Data` as well |

## Format and output

| Gumlet | crabcrop | Mapping |
| --- | --- | --- |
| No `format` (the source format is served) | Pin `filters:format(<the source format>)` | Approximate: dropping `format` does not keep the source format here, because negotiation then chooses. Name the format on templates that relied on staying put |
| `format=auto` (`fm=auto`) | Nothing | Direct: negotiation is built in. The winner can differ: Gumlet's order is JPEG XL, AVIF, WebP, then JPEG/GIF; crabcrop's is AVIF, JPEG XL, then WebP by q-value, and it honors a client's `q=0` refusal, which Gumlet's fixed order does not express |
| `fm=jpeg`, `jpg`, `png`, `webp`, `jxl`, `avif`, `gif` | `filters:format(...)` | Direct |
| `fm=tiff`, `fm=raw` | No equivalent | Unsupported as output formats |
| PDF, TIFF, RAW, and JPEG 2000 sources | No equivalent | Unsupported as sources: crabcrop decodes JPEG, PNG, WebP, GIF, JPEG XL, AVIF, HEIC/HEIF, and SVG (sanitized). Convert those sources before migrating them |
| `format=json` (metadata as JSON) | Fetch the image URL with `Accept: application/json`, or `crabcrop meta` | Direct for the image facts. The `faces=true` addition (detected face coordinates in the JSON) has no equivalent; meta responses carry no face coordinates |
| `q=80` | `filters:quality(80)` | Direct for JPEG and WebP: the values copy. The defaults do not (Gumlet 85, crabcrop 80), Gumlet `q=0` has no crabcrop spelling (1 is the floor), and Gumlet's `q` is documented as JPEG and WebP only, while crabcrop's quality also drives AVIF and JPEG XL |
| `lossless=true` | `filters:quality(100)` for WebP | Approximate: 100 selects crabcrop's lossless WebP encode; PNG is lossless at any setting |
| `compress=true` (extra lossy optimization of JPEG and PNG output) | No equivalent | Unsupported as a switch: there is no second recompression pass. Set `quality(...)` explicitly if templates relied on the size drop |
| `metadata=true` (keep all metadata) | Server configuration, not a URL option | Approximate: the operator policy `CRABCROP_METADATA=preserve` keeps EXIF, XMP, and IPTC on JPEG outputs and EXIF and XMP on PNG outputs, for every response. `metadata=false` and the default need nothing: stripping is crabcrop's default |
| `progressive=false` | `filters:format(jpeg)` | Approximate: progressive encoding is not a separate switch |
| `subsample=true` (`chromasub`, force 4:2:0) | No equivalent | Unsupported: chroma subsampling is not exposed |
| `pngcomp=9` | No equivalent | Unsupported: PNG compression level is not a knob |
| `page=<n>` on PDF sources | No equivalent | Unsupported: crabcrop does not take PDF sources. On animated GIF and WebP sources, page selection has no equivalent either; animations keep their frames |
| `dl=<filename>` (force download) | No equivalent | Unsupported: a `Content-Disposition` concern for your app or CDN layer |

## Adjustments and effects: where numbers change meaning

Gumlet's adjustment family runs on imgix-style scales: -100 to 100 with 0 as unchanged for `bri`, `sat`, and `con`. crabcrop's family was not built on one scale, so this is the section to translate carefully. Saturation is the trap, and it leads the table.

| Gumlet | crabcrop | Mapping |
| --- | --- | --- |
| `sat=-30` | `filters:saturation(0.7)` | Value changes meaning: Gumlet is a percent (0 unchanged, -100 grayscale), crabcrop a factor (1 unchanged, 0 grayscale). Convert with `1 + sat/100`; copying the number renders a different image |
| `con=20` | `filters:contrast(20)` | Provisionally direct: both are documented as percents, -100 to 100. Gumlet's own page contradicts itself, stating that range and a default of `1`, which belongs to a factor scale. Verify one mid-range `con` render before trusting a bulk copy (see the differences below) |
| `bri=10` | `filters:brighten(26)` | Approximate: rescale with `bri x 2.55` onto crabcrop's additive channel shift (-255 to 255). Gumlet's docs give the range but never say whether `bri` scales or shifts, so expect a close match, not an identical one |
| `hue=90` | `filters:huerotate(90)` | Direct: degrees on both. Gumlet accepts 0 to 360 only; crabcrop also takes negative degrees. Small rendering differences between any two hue implementations are normal |
| `sharp=40` | `filters:sharpen(0.4)` | Approximate: a 0 to 100 strength becomes an unsharp strength, rescaled by 100. Gumlet sharpens the LAB L channel; the kernels differ, so treat the number as a starting point |
| `blur=10` (Gaussian radius, 0 to 100, floats allowed) | `filters:blur(10)` | Approximate: both numbers are Gaussian radii in pixels, so the value carries as a starting point, unlike imgix's 0 to 2000 strength scale. Radius and sigma are not the same definition in every implementation; compare renders |
| `gamma=1.8` (`gam`) | No equivalent | Unsupported: Gumlet's gamma is a resize-time round trip (encode at `1/gamma` before resizing, decode at `gamma` after), not an output look. Dropping it loses no appearance for most URLs, but there is no crabcrop knob that reproduces it |
| `threshold=128` | `filters:curve(rgb,0,0,127,0,128,255,255,255)` | Approximate: there is no threshold filter. A two-step curve (flat 0 up to the threshold, flat 255 above it) reproduces the binarize on the combined channel |
| `opacity=50` (`o`) | No equivalent | Unsupported: opacity exists here only per overlay (`watermark` and `overlay` alpha), never on the base image |
| `tint=<color>` | `filters:monochrome(<hex>)` | Approximate at best: monochrome is luminance scaled by one color at full strength. A Gumlet tint is a lighter touch over the colors, and CSS color names do not carry; crabcrop colors are RRGGBB hex |
| `enhance=true` | `filters:equalize` | Approximate at best: equalize is per-channel histogram equalization, a stronger and different operation than stretching luminance to the full range. Treat it as the blunt substitute and compare renders |
| `greyscale=true` (or `grayscale=true`) | `filters:grayscale` | Direct |
| `invert=true` | `filters:invert` | Direct |
| `rotate=90` | `filters:rotate(90)` | Approximate, on one unverified point: Gumlet accepts only 0, 90, 180, and 270, and its docs never state the direction. crabcrop counts clockwise and takes any angle, expanding the canvas off the quarter turns. Verify one asymmetric render before bulk translation |
| `flip=h`, `v`, `hv` | `filters:flip(h)`, `flip(v)`, or both chained | Direct |
| `trim=50` (similarity percent, 1 to 99) | `filters:trim(<tolerance>)` | Value changes meaning: crabcrop's argument is a per-channel color-difference tolerance from 0 to 255 (default 10), not a percentage. The number does not copy; a proportional starting point is `n x 255 / 99`, then compare renders. Both trim the top-left corner color by default |
| `trim-color=<hex>` (with `trim`) | No equivalent | Unsupported: crabcrop always trims the top-left corner color; the reference color cannot be named |
| `background=<color>` (`bg`), as the flatten and pad color | `filters:fill(<hex>)`, or the `pad:` fill | Direct for hex colors. CSS color names do not carry, and the defaults differ: Gumlet's unnamed background is white, an unnamed crabcrop `pad` fill is black |
| `colorspace=b-w` (`cs`) | `filters:grayscale` | Direct for that one value. The other colorspaces (`cmyk`, `lab`, `hsv`, and the rest) are unsupported: outputs are sRGB |
| `palette=json` / `palette=css`, with `colors` and `prefix` | No equivalent | Unsupported: palette extraction is analysis, not rendering, and meta responses carry no palette. Compute palettes offline if you need them |

## AI operations

Gumlet's AI operations are plan-gated (upscale on paid plans, the rest on Enterprise). crabcrop's counterparts are operator-configured models on your own server: configured, they render; unconfigured, the request answers 501 instead of falling back to a plain render.

| Gumlet | crabcrop | Mapping |
| --- | --- | --- |
| `remove_bg=true` | `filters:removebg` plus an alpha-carrying `format(...)` | Direct counterpart: both cut the subject to transparency with a segmentation model. crabcrop's model is operator-configured (`CRABCROP_REMOVEBG_MODEL`), and the masks will differ in detail |
| `add_shadow=true`, with `az`, `el`, `st` | No standalone equivalent | Unsupported as a filter: `filters:studio(product,shadow:soft)` and `shadow:strong` synthesize a contact shadow, but only inside studio mode's product pipeline, with named strengths instead of light angles |
| `extend_bg=true` (outpainting to the target size) | No equivalent | Unsupported: generative. crabcrop renders from the source image only |
| `change_bg=true` with `prompt` | No equivalent | Unsupported: generative. Studio mode's backdrop is a color, the brand color, or none, never a generated scene |
| `upscale=true` | `filters:upscale(2)` or `filters:upscale(4)` | Approximate: Gumlet picks the factor itself, for sources under 512x512, up to 4x, and removes artifacts as it goes. crabcrop takes the factor explicitly, behind an operator-configured upscale model, and caps output at 4096 px per dimension |

## Overlays and text

Gumlet has two overlay systems: an image overlay (`overlay` and its companions) and a text overlay (`text` and its companions). crabcrop's surface is three flat filters, so simple watermarks and captions translate and compositions do not.

| Gumlet | crabcrop | Mapping |
| --- | --- | --- |
| `overlay=<url>` (centered by default) | `filters:watermark(<url>,<x>,<y>,<alpha>)` | Approximate: the overlay input is a full URL on both systems (percent-encoded into the crabcrop segment), but crabcrop positions by pixel offsets, so a position becomes offsets. Gumlet documents no overlay opacity, so use alpha 100 unless the overlay asset carries its own transparency |
| `overlay_position=bottomright` (the nine grid values) | Negative `x`, `y` in `watermark` or `overlay`, once a pad is chosen | Approximate: negative offsets anchor from the right and bottom edges on crabcrop. A grid position with no pad has no exact spelling; the position becomes computed pixel offsets, which needs the output dimensions |
| `overlay_top=24`, `overlay_left=24` | The `x`, `y` arguments | Direct in convention: pixel offsets from the top-left on both systems |
| `overlay_width_pct=0.25` | `filters:overlay(<url>,<x>,<y>,<alpha>,normal,25)` | Value changes form: a fraction of the output width (0 to 1) becomes a percent of the base width (1 to 400), so multiply by 100. `overlay_height_pct` has no equivalent; width is the only size argument |
| `overlay_tile=true` | No equivalent | Unsupported: overlays do not tile |
| An overlay URL carrying its own query string (Gumlet's own examples nest one, unencoded) | Pre-size the overlay asset, or accept overlay's width percent | Approximate: the URL is percent-encoded whole into the crabcrop segment, and its parameters are not applied separately |
| `text=<text>` with `text_font_size`, `text_color`, `text_position` | `filters:text(<text>,<size>,<color>,<anchor>)` | Partial: one caption, percent-encoded directly, in the one font configured on the server (`CRABCROP_FONT_PATH`), at one of nine grid anchors, padded from the edges by half the font size. The nine `text_position` values map to the anchors with hyphen spellings (`topleft` becomes `top-left`) |
| `text_font`, `text_align`, `text_line_height`, `text_wrap`, `text_rotate` | No equivalent | Unsupported: font choice, multi-line layout, and rotated text do not map. Multi-line text via `\n` in the Gumlet value does not carry; the caption is one line |
| `text_bg_color` with `text_padding` (a background pill behind the text) | No equivalent | Unsupported: the text filter has no background |
| `text_top`, `text_left` (pixel offsets) | No equivalent | Unsupported: the caption sits at its anchor; there are no per-caption pixel offsets |
| `text_width_pct`, `text_height_pct` | No equivalent | Unsupported: the caption sizes from its font size, not a box |

## The differences that bite

**Saturation is the silent one.** `sat` and `con` sit side by side in Gumlet URLs on the same -100 to 100 scale, but only `con` has a chance of surviving the move. Gumlet `sat` is a percent with 0 as unchanged; crabcrop `saturation` is a factor with 1 as unchanged. `sat=-100` copied into a crabcrop URL is a parse error (the factor floor is 0); `sat=50` copied as `saturation(50)` parses fine and renders at fifty times the intended saturation, the worst kind of migration bug, because every page looks plausible at a glance. Convert with `1 + sat/100`.

**Gumlet's contrast documentation contradicts itself.** The operations page gives `con` a range of -100 to 100 and a default of `1`. A percent scale's neutral point is 0; `1` is a factor scale's neutral point pasted into the wrong row, most likely, but the docs never resolve it. If the range is the truth, `con` copies verbatim and this guide's table is right. Translate one known image at a mid-range `con` value and compare before bulk-copying thousands of templates on a documentation ambiguity.

**Crop and stretch never enlarge on Gumlet; their translations do.** `enlarge` is forced off for `mode=crop` and `mode=stretch`, so Gumlet serves a small source at its own size where the template asked for more. crabcrop's `crop:` and `WxH` upscale to the box unless `filters:no_upscale` rides along. Every crop and stretch row in the table above carries the flag for this reason; drop it from a template and small sources come back bigger than Gumlet ever served them.

**Dropping `format` changes the format.** A Gumlet URL with no `format` serves the source format. A crabcrop URL with no `format(...)` negotiates from `Accept`, which can turn a JPEG source into AVIF, JPEG XL, or WebP. Templates that relied on staying in the source format need the format named.

**A source's default parameters are invisible in its URLs.** A Gumlet URL under-reports its own rendering: whatever the dashboard defaults add (a width, a quality, a format) applies on top of the query string and appears nowhere in it. Auditing templates alone misses them. Read each source's default parameters before translating, and fold what you find into the templates or into presets, or renders change silently at cutover.

**DPR values between the stops do not degrade, they fail.** Gumlet accepts any `dpr` float from 1.0 to 3.0, and its own srcset tooling emits values like 2.5. crabcrop accepts exactly 1, 1.5, 2, and 3, so a copied `dpr=2.5` is a 400, loudly, on every request that uses it. Bake the extra density into the size instead: a 400 px slot at `dpr=2.5` becomes a 1000x0 request.

**Unnamed backgrounds are white on Gumlet and black here.** Gumlet's `background` and `fill-color` both default to white. An unnamed crabcrop `pad` fill is black, and an unnamed flatten has no color until `filters:fill(...)` names one. Any template that pads or flattens without naming a color changes appearance at cutover; name the color.

**Rotate direction is undocumented.** Gumlet's docs list the accepted angles (0, 90, 180, 270) and never say which way is positive. crabcrop counts clockwise. If Gumlet counts counter-clockwise, every `rotate=90` template comes back mirrored in orientation, and nothing errors anywhere. One asymmetric test image settles it; run it before the bulk translation, not after.

**The source hides its base.** For an S3 or Web Folder source, the Gumlet path is a key, and the bucket and prefix live in the source configuration, not the URL. crabcrop needs the whole source URL, so migration joins prefix and key per asset, in your application or templates. Nothing in the Gumlet URL alone tells you the prefix.

**Purging moves in-house.** A Gumlet purge clears Gumlet's cache layers (whole-source purges are plan-gated there). On crabcrop, purge the server's tiers with `purge` or `purge-group`; if you also keep a CDN in front of crabcrop, purge crabcrop first and the CDN second, so a concurrent request cannot refill the edge from stale tiers (the recipes in `deploy/` cover the major CDNs).

## Converter feasibility

Could a `convert-gumlet` verb sit next to `convert-thumbor`? The mechanical core could: Gumlet's native URLs parse cleanly (a path plus a flat query string of `key=value` pairs), most size, format, quality, and adjustment parameters have the equivalents above, and the Thumbor converter's pattern, convert what is faithful and drop the rest with a named warning rather than guessing, fits.

Four things keep it from being a weekend verb. First, the source: a Thumbor URL carries its source in full, but a Gumlet URL's path is a key under a source whose base lives in the dashboard, so the verb needs the source base supplied out of band before it can emit a crabcrop URL at all (the same blocker as imgix bucket Sources). Worse, a source's default parameters are also dashboard state the URL does not carry, so even a perfect conversion of the visible parameters can miss part of the render. Second, a Gumlet sizing decision is three parameters, not one: `mode`, `crop`, and `enlarge` combine into the crabcrop operation, and the forced `enlarge=false` cases have to be detected to place `no_upscale` correctly. Third, fraction-valued parameters (`fp-x` and `fp-y`, `overlay_width_pct`, the text percentages) need source or output dimensions a string converter does not have, and the overlay value can nest a whole second URL with its own unencoded query string inside the first one's query string, which breaks naive query parsing. Fourth, the estate may not be in Gumlet's grammar at all: URLs served from an Imgix-type source use imgix spellings, so the verb would need to detect and route two grammars. None of that blocks a useful verb for the flat native core; all of it argues for shipping the tables first, which is what this guide is.

## Rolling out

Run both side by side. crabcrop fetches the same origins Gumlet does, so templates can switch per page, per route, or behind a flag, and either system can serve while the other warms up. Translate templates, not individual asset URLs: the template set is finite, and the SDKs re-sign everything they build. Where a source's default parameters were doing real work, turn them into presets or fold them into the templates as you go; there is no per-source default to carry over.

If Gumlet.js is in the pages, it has to go: it rewrites URLs toward the Gumlet host and sizes them client-side. Replace it with server-built URLs from the SDKs (or `build_srcset` families), and keep the client-hint delegation pointing at the crabcrop host; the hints themselves are read natively on this side.

Expect cold caches. crabcrop renders on first request and then serves from its memory and disk tiers; pre-warm the hot templates by requesting them once after cutover, or let a CDN in front absorb the first-hit cost. The AI rows invert their economics at the same time: on Gumlet they are plan entitlements, here they are your configured models and your compute, and an unconfigured model answers 501 rather than rendering.

Old Gumlet URLs age out on the source and its token, not on the migration. An unsigned URL stays valid while the source serves it, which can be forever. A signed URL without `expires` also never expires: the signature stands until the source's secure token rotates. A signed URL with `expires` dies at its timestamp (403). The practical sequence: switch the templates, keep the source deployed while shared and bookmarked links drain, rotate the secure token last if signing was on (so a missed template fails loudly instead of serving forever), then decommission the source.

## Worked examples

A hero crop. `format=auto` drops out, because negotiation is built in, and `no_upscale` rides along because a Gumlet crop never enlarges:

```
Gumlet:
  https://yourbrand.gumlet.io/hero.jpg?mode=crop&crop=smart&w=1600&h=900&q=80&format=auto

crabcrop:
  /<signature>/crop:1600x900:smart/filters:no_upscale:quality(80)/https://images.example.com/hero.jpg
```

A face thumbnail. Gumlet centers on all detected faces; crabcrop frames the largest one:

```
Gumlet:
  https://yourbrand.gumlet.io/portrait.jpg?mode=crop&crop=faces&w=200&h=200

crabcrop:
  /<signature>/crop:200x200:face/filters:no_upscale/https://images.example.com/portrait.jpg
```

A product shot padded onto a brand color, using `fillmax`, the no-enlarge fill that `pad` matches exactly:

```
Gumlet:
  https://yourbrand.gumlet.io/shoe.png?mode=fillmax&fill=solid&fill-color=F5F0E8&w=1200&h=1200&format=webp

crabcrop:
  /<signature>/pad:1200x1200:f5f0e8/filters:format(webp)/https://images.example.com/shoe.png
```

An adjustment stack. Contrast copies (subject to the documentation caveat above); brightness rescales; saturation converts from percent to factor:

```
Gumlet:
  https://yourbrand.gumlet.io/street.jpg?w=800&bri=10&con=20&sat=-30&hue=90&sharp=40

crabcrop:
  /<signature>/800x0/filters:brighten(26):contrast(20):saturation(0.7):huerotate(90):sharpen(0.4)/https://images.example.com/street.jpg
```

A focal-point crop. The source is 2000x1333, so `fp-x=0.75` and `fp-y=0.4` land on the point (1500, 533), which becomes a 100px focal rectangle around it:

```
Gumlet:
  https://yourbrand.gumlet.io/puppy.jpg?mode=crop&crop=focalpoint&fp-x=0.75&fp-y=0.4&w=600&h=600

crabcrop:
  /<signature>/focal:1450x483:1550x583/crop:600x600:smart/filters:no_upscale/https://images.example.com/puppy.jpg
```

An extract at original scale, then a resize. The corner and size become two corners:

```
Gumlet:
  https://yourbrand.gumlet.io/food.jpg?extract=100,300,300,300&w=600

crabcrop:
  /<signature>/crop-rect:100x300:400x600/600x0/https://images.example.com/food.jpg
```

A watermarked card. Gumlet's overlay has no opacity parameter, so alpha is 100; its width fraction (0.25 of the output width) becomes a width percent (25 of the base width, the same 1200 px here):

```
Gumlet:
  https://yourbrand.gumlet.io/card.jpg?w=1200&h=630&mode=crop&overlay=https://assets.example.com/logo.png&overlay_top=24&overlay_left=24&overlay_width_pct=0.25

crabcrop:
  /<signature>/crop:1200x630/filters:no_upscale:overlay(https%3A%2F%2Fassets.example.com%2Flogo.png,24,24,100,normal,25)/https://images.example.com/card.jpg
```

A text badge. The caption, size, color, and position carry; a `text_bg_color` pill would not:

```
Gumlet:
  https://yourbrand.gumlet.io/hoodie.jpg?w=394&h=540&mode=crop&text=SALE&text_font_size=30&text_color=ffffff&text_position=topleft

crabcrop:
  /<signature>/crop:394x540/filters:no_upscale:text(SALE,30,ffffff,top-left)/https://images.example.com/hoodie.jpg
```

Background removal, the direct counterpart. The output format is pinned to PNG so the transparency has somewhere to live:

```
Gumlet:
  https://yourbrand.gumlet.io/product.png?remove_bg=true&format=png

crabcrop:
  /<signature>/filters:removebg:format(png)/https://images.example.com/product.png
```

A signed, expiring URL. The timestamp copies exactly; what changes is the scheme (MD5 in a query parameter becomes HMAC-SHA256 as a path segment). Both systems answer 403 past the deadline:

```
Gumlet:
  https://yourbrand.gumlet.io/report.jpg?w=1024&expires=1893456000&s=<md5>

crabcrop:
  /<signature>/exp:1893456000/1024x0/https://images.example.com/report.jpg
```
