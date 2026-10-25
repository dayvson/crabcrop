# Migrating from ImageKit

Draft placeholder.

ImageKit is a hosted service: you attach an origin (its Media Library, or your own storage) to a URL endpoint, and transformations ride on the delivery URL as a `tr:` chain in the path or a `?tr=` query parameter. crabcrop is a server you run in front of the same storage, and transformations are path segments in a signed URL. The assets do not move; the URL templates do. There is no mechanical ImageKit converter (the `convert-thumbor` verb covers Thumbor), so templates translate by hand using the tables below. Each row is marked direct, approximate, or unsupported, and the rows whose numbers change meaning in translation say so exactly, because copying those verbatim is how a migration ships wrong renders nobody notices.

## Concepts

| ImageKit | crabcrop |
| --- | --- |
| A URL endpoint bound to one origin: the Media Library, or external storage (S3, GCS, Azure, a web folder, another ImageKit account) | The full source URL spelled out in every request, fetched from your origin at request time |
| An asset path under the endpoint; Media Library files also carry a `fileId`, but delivery URLs use the path, never the ID | The source URL itself; there is no asset registry or ID namespace |
| Transformations as `tr:` in the path or `?tr=` in the query: steps of comma-joined `key-value` parameters, steps chained with `:` | Operations as path segments, applied in URL order: geometry segments and `filters:` chains |
| A named transformation (`n-<name>`), defined in the dashboard as a whole transformation string | A preset: an operation string defined on the server (`CRABCROP_PRESETS_FILE`), referenced as `preset:<name>` |
| The restrict-unnamed-transformations account setting: only named transformations are served | `CRABCROP_ONLY_PRESETS=1`: only preset references, `exp:` and `maxage:` metadata, and bare sources are accepted |
| Signing optional, enforced per account (restrict unsigned URLs): HMAC-SHA1 in the `ik-s` and `ik-t` query parameters | Every URL signed with hex HMAC-SHA256 (`CRABCROP_SECRET`), or `unsafe` behind `CRABCROP_ALLOW_UNSAFE` |
| Automatic format conversion and quality by default, from dashboard image settings | Negotiation is built in (AVIF, JPEG XL, then WebP, by `Accept` q-values); quality defaults to 80 on both systems |
| Purge from the dashboard or API clears ImageKit's CDN copy of a URL | `POST /purge/...` per URL, `POST /purge-group/...` per preset, or the CLI verbs, clear crabcrop's own cache tiers |
| The Media Library and DAM, the upload API, AI tags and metadata, the video transformation pipeline | Out of scope: crabcrop is delivery-time image transformation only. Its `mp4` and `webm` are output formats for animated image sources, not a video pipeline |

## Signing

The schemes do not convert, in either direction. An ImageKit signature is a lowercase hex HMAC-SHA1 keyed with the account's private API key, computed over the URL path after the endpoint with the expiry timestamp concatenated onto it, and carried in the query string as `ik-s`; the optional expiry travels beside it as `ik-t` (unix seconds). A crabcrop signature is an HMAC-SHA256 hex string over the path remainder, carried as the first path segment. Pick a `CRABCROP_SECRET` and re-sign templates with the SDKs or `crabcrop sign`.

Four operational differences matter. First, enforcement: ImageKit signing is an account setting with three modes (off, AI transformations only, all requests), and it is off by default; crabcrop signs everything unless the server deliberately runs with `CRABCROP_ALLOW_UNSAFE=1`. Second, the failure answer: an invalid or expired ImageKit URL returns 401, crabcrop returns 403, for a bad signature and for an expired `exp:` alike. Third, expiry is optional on both: an ImageKit URL with no `ik-t` never expires, and a crabcrop signature with no `exp:` segment never expires either; when an `ik-t` timestamp exists, it copies exactly into `exp:`. Fourth, rotation: the ImageKit signing key is the account's private API key, so rotating it also rotates upload and admin API access, with no overlap period; crabcrop's optional `<key-id>:` signature prefix exists so rotation can overlap two secrets instead.

## Resizing and cropping

ImageKit splits the cropping decision across two parameters: `c` chooses the size strategy (the default `maintain_ratio`, plus `force`, `at_max`, `at_max_enlarge`, `at_least`, `maintain_ratio_no_enlarge`) and `cm` chooses the crop mode (`pad_resize`, `pad_resize_no_enlarge`, `extract`, `pad_extract`, `pad_extract_no_shrink`). crabcrop spells each strategy as its own operation.

| ImageKit | crabcrop | Mapping |
| --- | --- | --- |
| `w-300` alone (or `h-` alone) | `300x0` | Direct: one axis set, the other proportional |
| `w-0.5` (a value between 0 and 1 is a fraction of the source) | No equivalent | Unsupported mechanically: pixels need the source dimensions |
| `w-auto` (read `Sec-CH-Width`), `dpr-auto` | Nothing to translate | Direct: when a URL sizes nothing itself, crabcrop honors the width and DPR client hints natively, quantized by `CRABCROP_WIDTH_STEP` (default 100) |
| `w` + `h` (default `c-maintain_ratio`, centered) | `crop:WxH` | Direct: crop-to-fill in both |
| `c-maintain_ratio_no_enlarge` | `crop:WxH` plus `filters:no_upscale` | Approximate: ImageKit returns a smaller source untouched, at its original size and aspect; crabcrop still crops to the target aspect, at source scale |
| `c-force` | `WxH` | Direct: exact resize, aspect not kept, in both |
| `c-at_max` | `fit-in:WxH` plus `filters:no_upscale` | Direct: fit inside the box, never upscale |
| `c-at_max_enlarge` | `fit-in:WxH` | Direct: fit inside the box, upscaling allowed |
| `c-at_least` (cover the box, no cropping, output may exceed it) | No equivalent | Unsupported: crabcrop has no cover-without-crop size. Compute the covering size in the template when the source aspect is known, or accept `crop:WxH`, which crops: different pixels |
| `cm-pad_resize` with `bg` | `pad:WxH:<color>` | Approximate: `pad_resize` upscales a smaller source to fill the box before padding; crabcrop's `pad` never upscales |
| `cm-pad_resize_no_enlarge` with `bg` | `pad:WxH:<color>` | Direct: fit without upscaling, centered on the exact canvas. The unnamed fill defaults agree for once: black on both |
| `cm-extract` with `x`, `y` | `crop-rect:<x>x<y>:<x+w>x<y+h>` | Direct, after the corner math: crabcrop takes two corners, ImageKit a corner and a size |
| `cm-extract` with an `fo` alignment | `crop-rect` with computed corners, or the matching `crop:` alignment | Approximate: the alignment positions need the source dimensions, and an alignment crop resizes its window to WxH where extract keeps native-scale pixels |
| `cm-pad_extract`, `cm-pad_extract_no_shrink` | No equivalent as written | Unsupported: a `crop-rect` followed by a `pad:` composes the shape when the coordinates are known |
| `ar-16-9` with `w` or `h` | Compute the other axis, or `crop:WxH` at that ratio | Approximate: the ratio is template arithmetic; there is no `ar` parameter |
| `fo-center`, `top`, `left`, `bottom`, `right`, `top_left`, `top_right`, `bottom_left`, `bottom_right` | `crop:WxH:center` and the same-named alignments | Direct, with a spelling change: crabcrop alignments use hyphens (`top-left`), ImageKit underscores |
| `fo-auto` | `crop:WxH:smart` | Direct in intent: both anchor on the most important region. With a face model configured, crabcrop's `smart` also keeps detected faces whole |
| `fo-face` | `crop:WxH:face` | Approximate: crabcrop frames the largest detected face in its own window and requires `CRABCROP_FACE_MODEL`. The zoom parameter `z` becomes `filters:facepad(...)`, translated by eye: `z` above 1 zooms in (tighter), while facepad 0 is the tightest framing and larger values add background |
| `fo-<object>` (a COCO class name, or a fallback list like `fo-car_dog`) | No crop equivalent | Unsupported: crabcrop's detectors (`CRABCROP_DETECTORS_FILE`) feed the redaction filters (`blurclass`, `pixelateclass`), not crop focus. `fo-auto` and `fo-face` cover the saliency and face cases |
| `fo-custom` (a focus area stored on the asset, set at upload or in the Media Library) | Export the stored rectangle, then `focal:<rect>` before `crop:WxH:smart` | Approximate, and the data is the work: the rectangle lives in ImageKit, not in the URL, so it has to be exported per asset |
| `x`, `y` with `cm-extract` | `crop-rect` corners | Direct: both address the original image |
| `xc`, `yc` with `c-maintain_ratio` | `focal:<rect>` around the point, rescaled | Approximate, and a trap: these coordinates address the image after it is resized to cover the target box, not the source. Scale them back with the source and target dimensions before they become a focal rectangle (see the differences below) |
| `dpr-2` | `filters:dpr(2)`, or the `DPR` / `Sec-CH-DPR` request headers | Approximate: crabcrop accepts exactly 1, 1.5, 2, and 3. ImageKit accepts 0.1 to 5, so `dpr-2.5`, `dpr-4`, and `dpr-5` are parse errors here, not renders. Note that ImageKit applies DPR at every chained step unless an account setting moves it to the last step; crabcrop's `dpr` applies once, request-wide |
| Chained steps (`tr:w-300,h-200:r-24`) | Operations in path order | Direct in spirit: a chain step becomes one or more segments. Within one ImageKit step, parameters apply in ImageKit's own pipeline order; crabcrop applies the order written, so write a radius after the resize it rounds |

## Format and output

| ImageKit | crabcrop | Mapping |
| --- | --- | --- |
| `f-auto` (the dashboard default) | Nothing | Direct: negotiation is built in. Two behaviors widen: crabcrop also negotiates JPEG XL, and it honors a client's `q=0` refusal |
| `f-jpg`, `jpeg`, `png`, `webp`, `avif` | `filters:format(...)` | Direct |
| `f-tiff`, `f-tif` | No equivalent | Unsupported |
| `f-orig` | Pin `filters:format(<the source format>)` | Approximate: dropping `f` does not keep the source format here, because negotiation then chooses. Name the format |
| `q-80` | `filters:quality(80)` | Direct: the values copy, and the defaults agree (80 on both), the first guide in this series where they do |
| `lo-true` (lossless WebP and PNG) | `filters:quality(100)` for WebP | Approximate: 100 selects crabcrop's lossless WebP encode; PNG is lossless at any setting |
| `pr-true` (progressive JPEG) | `filters:format(jpeg)` | Approximate: progressive encoding is not a separate switch |
| `md-true` (preserve all metadata, per URL) | Server configuration, not a URL option | Approximate: the operator policy `CRABCROP_METADATA=preserve` keeps EXIF, XMP, and IPTC on JPEG outputs and EXIF and XMP on PNG outputs, for every response. `md-false` and the default need nothing: stripping is crabcrop's default |
| `cp-true` (embed the color profile) | `filters:keep_profile` | Approximate: the profile is kept and embedded in JPEG and PNG outputs. ImageKit also converts a non-sRGB source to sRGB with a standard profile embedded; `keep_profile` leaves the pixels unconverted |
| `orig-true` (deliver the file as-is, ignoring other parameters) | A bare signed source URL | Direct in effect |
| `di-<path>` (per-URL default image on a miss) | `CRABCROP_FALLBACK_IMAGE` | Approximate: the fallback is one server-wide image, not a per-URL choice; it runs through the same operations and is never cached |
| `ik-attachment=true` (force download) | No equivalent | Unsupported: a `Content-Disposition` concern for your app or CDN layer |
| `dn-<dpi>` (output density; rasterization DPI for vector sources) | No equivalent | Unsupported: a print and vector concern. Page selection (`pg`) on multi-page sources is likewise unsupported: crabcrop does not take PDF or PSD sources |

## Adjustments and effects: where numbers change meaning

ImageKit's adjustment surface is mostly flags, not dials: there is no numeric brightness, contrast, saturation, or hue parameter anywhere in `tr:`. The one contrast control is a stretch flag, and the traps live in trim, DPR, coordinates, and color spelling instead.

| ImageKit | crabcrop | Mapping |
| --- | --- | --- |
| `e-contrast` (flag: stretch contrast to the full intensity range) | `filters:equalize` | Approximate at best: equalize is per-channel histogram equalization, a stronger and different operation. A true stretch depends on each image's own extremes, so there is no faithful URL-level conversion; treat equalize as the blunt substitute and compare renders |
| `e-grayscale` | `filters:grayscale` | Direct |
| `e-sharpen`, `e-sharpen-<n>` | `filters:sharpen(<n>)` | Approximate: the number carries as a starting point, but the kernels differ, and the unnumbered form's default extent is an ImageKit choice you now have to make explicitly |
| `e-usm-<radius>-<sigma>-<amount>-<threshold>` | `filters:sharpen(<amount>)` | Approximate: only the amount carries, as a starting point. The radius and threshold have no equivalents; crabcrop's kernel is fixed |
| `bl-<n>` (Gaussian radius, 1 to 100) | `filters:blur(<sigma>)` | Approximate: a radius becomes a Gaussian sigma, and the numbers do not convert by a constant; start near `bl / 10` and compare renders |
| `t-true`, `t-<n>` (threshold level, 1 to 99) | `filters:trim(<tolerance>)` | Value changes meaning: crabcrop's argument is a per-channel color-difference tolerance from 0 to 255 (default 10), not a 1 to 99 background level. The number does not copy; a proportional starting point is `n x 255 / 99`, then compare renders. The target differs too: crabcrop always trims the top-left corner color |
| `rt-<deg>` (clockwise), `rt-N<deg>` (counter-clockwise) | `filters:rotate(<deg>)`, `filters:rotate(-<deg>)` | Direct for the value and sign: both grammars count clockwise as positive, and the `N` prefix becomes a minus. Approximate at non-quarter angles: crabcrop expands the canvas and fills corners with the `fill` color or transparency |
| `rt-auto` (rotate by EXIF orientation) | Nothing | Direct: crabcrop always honors EXIF orientation at decode |
| `fl-h`, `fl-v`, `fl-h_v` | `filters:flip(h)`, `flip(v)`, or both chained | Direct |
| `r-<px>`, `r-<tl>_<tr>_<br>_<bl>` | `filters:radius(...)` | Direct: pixels on both sides, the same corner order. One behavior differs: crabcrop needs an alpha-carrying output, so pin `format(png)` (or webp, jxl, avif) beside it; ImageKit switches the output for you |
| `r-max` | No literal equivalent | Approximate: a radius of half the shorter side needs the source dimensions, or use `filters:mask(circle)` / `mask(ellipse)` when the whole shape is the point |
| `b-<width>-<hex>` | `filters:border(<width>,<hex>)` | Approximate: crabcrop paints the band inside the frame and the dimensions do not change; check a render if your layout depends on the bordered size |
| `bg-<hex>` (with the pad modes, or flattening transparency) | The `pad:` fill, or `filters:fill(<hex>)` | Direct for six-digit hex. Color names do not carry, and RGBA hex does not copy: ImageKit's last two characters are a decimal opacity (00 to 99), not a hex alpha byte |
| `bg-blurred` | `pad:WxH:blur` | Direct: ambient blurred bars in both, and ImageKit likewise restricts it to `cm-pad_resize` |
| `bg-dominant`, `bg-gradient` | No equivalent | Unsupported |
| `o-<n>` (whole-image opacity, 0 to 100) | No equivalent | Unsupported: opacity exists here only per overlay (`watermark` and `overlay` alpha), never on the base image |
| `e-colorize-co-<color>_in-<n>` | `filters:monochrome(<hex>)` | Approximate: monochrome is the full-strength version (luminance scaled by the color). An `in` below 100 blends the tint back toward the original, which has no equivalent; color names do not carry |
| `e-shadow` (drop shadow under a transparent subject, with `bl`, `st`, `x`, `y`) | No standalone equivalent | Unsupported as a filter: `filters:studio(product,shadow:soft)` and `shadow:strong` synthesize a contact shadow, but only inside studio mode's product pipeline |
| `e-gradient` (linear gradient overlay) | No equivalent | Unsupported: `duotone` and `monochrome` are different looks, not a gradient |
| `e-distort` (perspective and arc) | No equivalent | Unsupported |
| `cr-<to>_<tolerance>_<from>` (color replace) | No equivalent | Unsupported: `rgb`, `curve`, and `monochrome` are blunt substitutes, not a hue-preserving recolor |
| Arithmetic expressions in values (`w-iw_mul_0.8`, `lx-bw_div_4`) | No equivalent | Unsupported as written: evaluate them when the template is built. Your application, or a meta request, knows the source dimensions the expression reads |

## AI transformations

| ImageKit | crabcrop | Mapping |
| --- | --- | --- |
| `e-bgremove` (ImageKit's own background removal) | `filters:removebg` plus an alpha-carrying `format(...)` | Direct counterpart: both cut the subject to transparency with a segmentation model. crabcrop's model is operator-configured (`CRABCROP_REMOVEBG_MODEL`), and the masks will differ in detail |
| `e-removedotbg` (the same job, run by remove.bg as a third party) | `filters:removebg` | Approximate: the provider changes, not the intent. There is no per-request provider choice here; the metered third party becomes your own model's compute |
| `e-dropshadow` (AI drop shadow, `az`, `el`, `st`) | No standalone equivalent | Unsupported as a filter: studio mode's synthesized contact shadow (`filters:studio(product,shadow:...)`) covers the product-shot case only, with named strengths instead of light angles |
| `e-retouch` | No equivalent | Unsupported: there is no quality-enhancement model. `upscale` raises resolution only |
| `e-upscale` (factor chosen for you; the output targets 16 MP) | `filters:upscale(2)` or `filters:upscale(4)` | Approximate: crabcrop takes the factor explicitly, behind an operator-configured upscale model, and caps output at 4096 px per dimension |
| `e-edit-prompt-...`, `e-changebg-prompt-...`, `e-genvar`, `ik-genimg-prompt-...`, `bg-genfill` | No equivalent | Unsupported: the generative family. crabcrop renders from the source image only; studio mode's backdrop is a color, the brand color, or none, never a generated scene |

## Overlays and text

ImageKit overlays are layers: `l-image`, `l-text`, and solid-color blocks opened inside a chain step and closed with `l-end`, nestable three deep, each with its own transformations, positions (`lx`, `ly`, `lxc`, `lyc`, `lfo`, `lap`, with `N` for negative values), and layer modes. The older `oi` and `ot` parameter families spell the same two ideas without layers. crabcrop's overlay surface is three flat filters, so simple watermarks and captions translate and compositions do not.

| ImageKit | crabcrop | Mapping |
| --- | --- | --- |
| `l-image,i-<path>` (a Media Library path, folders joined with `@@`), or `ie-<base64>` for a full URL | `filters:watermark(<url>,<x>,<y>,<alpha>)` | Approximate: the layer input becomes a percent-encoded URL, fetched through the same source policy as source images. A failed overlay load fails the request (502) instead of being skipped |
| Layer position: `lx`, `ly` pixels; `N` negatives anchor from the right and bottom; `lfo` and `lap` anchor grids | Negative `x`, `y` in `watermark` or `overlay` | Direct in convention: negative offsets anchor from the right and bottom edges on both systems. A named anchor plus offsets becomes computed pixel offsets |
| The layer's own `w`, `h`, and effects (`e-grayscale` inside a layer, and friends) | `filters:overlay(<url>,<x>,<y>,<alpha>,<blend>,<width>)` | Partial: overlay's width is a percent of the base width (1 to 400), so a pixel width becomes a percentage of the base. There is no way to pre-transform the overlay image itself |
| `lm-multiply` | `overlay(...,multiply)` | Direct: one of the four blend modes crabcrop has (`normal`, `multiply`, `screen`, `overlay`) |
| `lm-displace`, `lm-cutout`, `lm-cutter` | No equivalent | Unsupported |
| `oi-<path>` with `ox`, `oy`, `ofo`, `ow`, `oh`, `oa` (the legacy image overlay parameters) | `filters:watermark(...)` or `overlay(...)` | Approximate, as the layer row above; `oa` is the overlay opacity, which becomes the `alpha` argument (0 to 100 on both) |
| `l-text,i-<text>` or `ie-<base64 text>`, with `fs`, `co`, `ff`, `tg`, `ia`, `pa`, `bg`, `lh` | `filters:text(<text>,<size>,<color>,<anchor>)` | Partial: one caption, percent-encoded directly (no base64 twin), in the one font configured on the server (`CRABCROP_FONT_PATH`), at one of nine grid anchors, padded from the edges by half the font size. Font choice, bold and italic, wrapping width, line height, inner alignment, and the background pill do not map |
| `ot-<text>` with `ots`, `otc`, `otf`, `oa`, `otbg`, `otp` (the legacy text parameters) | `filters:text(...)` | Partial, as the layer row above |
| Solid color blocks and canvases (`l-image,i-ik_canvas,...`) with nested child layers | No equivalent | Unsupported as a composition system: a plain band is a `border` or `pad` job |

## The differences that bite

**Trim's number is not a tolerance.** `t` and `trim` look like the same parameter with the same kind of argument, and nothing errors when you copy one: the render just trims more, or less, than the original did. ImageKit's number is a background threshold level from 1 to 99; crabcrop's is a color-difference tolerance from 0 to 255 with a default of 10. Start from `n x 255 / 99` and compare renders; do not copy.

**Two coordinate parameters, two coordinate spaces.** With `cm-extract`, `x` and `y` address the original image, and they become `crop-rect` corners unchanged. With `c-maintain_ratio`, `xc` and `yc` address the image after it has been resized to cover the target box, so the same numbers copied into a `focal:` rectangle land in the wrong place by exactly the resize ratio. Convert them back through the source and target dimensions first.

**DPR values past 3 do not degrade, they fail.** ImageKit accepts `dpr` from 0.1 to 5, and its own named-transformation examples for icon sets use `dpr-4`. crabcrop accepts exactly 1, 1.5, 2, and 3, so a copied `dpr-4` is a 400, loudly, on every request that uses it. Bake the extra density into the size instead: an 18 px icon slot at `dpr-4` becomes a 72x72 request.

**RGBA hex is decimal where it looks hexadecimal.** In an ImageKit color like `FF000050`, the last two characters are an opacity written as a decimal number from 00 to 99: that is red at 50 percent. Read as the hex alpha byte crabcrop's eight-digit text colors use, `50` means 0x50, about 31 percent. Any template carrying an eight-digit ImageKit color needs its last pair converted, not copied.

**The endpoint hides its origin.** For a Media Library asset or an external storage origin, the ImageKit path is relative to the endpoint's configured origin, and the origin's base lives in the endpoint configuration, not the URL. crabcrop needs the whole source URL, so migration joins origin base and path per asset, in your application or templates. The `fileId` does not help: it never appears in delivery URLs.

**Named transformations can be edited in place; presets must not be.** Changing a named transformation's definition in the dashboard re-renders every URL that names it, once ImageKit's cache is purged. A crabcrop preset is an operator file, and the discipline is the opposite: never edit a definition in place, because image responses travel as immutable for a year and downstream copies of the old render sit beyond any purge. Roll the change out under a new preset name, update templates, and `purge-group` the old name to clear this server's tiers.

**Dropping `f` changes the format.** An ImageKit template with no `f` still runs under the account's automatic format setting, and `f-orig` pins the source format. A crabcrop URL with no `format(...)` negotiates from `Accept`, which can turn a JPEG source into AVIF or WebP. Templates that relied on staying in the source format need the format named.

**Purging moves in-house.** An ImageKit purge clears ImageKit's CDN copy. On crabcrop, purge the server's tiers with `purge` or `purge-group`; if you also keep a CDN in front of crabcrop, purge crabcrop first and the CDN second, so a concurrent request cannot refill the edge from stale tiers (the recipes in `deploy/` cover the major CDNs).

## Converter feasibility

Could a `convert-imagekit` verb sit next to `convert-thumbor`? The flat core is the most mechanical of any source in this guide series: the `tr:` grammar is regular (steps split on `:`, parameters on `,`, each a `key-value` pair split on the first `-`), the asset path travels in the URL in full, and most resize, format, quality, and effect parameters have the equivalents above. The Thumbor converter's pattern, convert what is faithful and drop the rest with a named warning rather than guessing, fits it well.

Four things keep it from being a weekend verb. First, the source: the URL carries the path under an endpoint, but the endpoint's origin base is configuration, not URL text, so the verb needs an endpoint-to-base mapping supplied out of band before it can emit a crabcrop URL at all (the same blocker as imgix bucket Sources). Second, named transformations are names: expanding `n-thumb` needs the dashboard definition, which lives behind the account's API keys. Third, value semantics need data a string converter does not have: arithmetic expressions read source dimensions, `xc` and `yc` need the cover-resize math, and `fo-custom` reads a rectangle stored on the asset. Fourth, layers nest recursively, three deep, with their own input encodings (`@@` paths, base64 `ie` values) and per-layer transformation lists; that is a second parser inside the first. None of that blocks a useful verb for the flat parameter core; all of it argues for shipping the tables first, which is what this guide is.

## Rolling out

Run both side by side. crabcrop fetches the same origins ImageKit does, so templates can switch per page, per route, or behind a flag, and either system can serve while the other warms up. Translate templates, not individual asset URLs: the template set is finite, and the SDKs re-sign everything they build. Turn each named transformation into a preset of the same name as you go, so `purge-group` lines up with the name it replaces.

Media Library assets need one decision: delivery is by path, so keep the paths when you copy originals into storage crabcrop can fetch, and templates change only in their base. Nothing in a delivery URL references the `fileId`, so there is nothing to translate there; application logic keyed on IDs (renames, versioning, AI tags) stays behind in ImageKit or moves to your own metadata.

Expect cold caches. crabcrop renders on first request and then serves from its memory and disk tiers; pre-warm the hot templates by requesting them once after cutover, or let a CDN in front absorb the first-hit cost. The AI rows invert their economics at the same time: on ImageKit they are metered extension units, here they are your configured models and your compute, and an unconfigured model answers 501 rather than rendering.

Old ImageKit URLs age out on the signing setting, not on the migration. An unsigned URL stays valid while the endpoint serves it, which can be forever. A signed URL without `ik-t` also never expires: the signature stands until the private key rotates. A signed URL with `ik-t` dies at its timestamp (401). The practical sequence: switch the templates, keep the endpoint deployed while shared and bookmarked links drain, and rotate the private key last, remembering that the rotation is an account-wide event that also replaces the key your upload and admin integrations use.

## Worked examples

A hero crop with smart focus. `fo-auto` becomes the `smart` alignment, and `q` copies:

```
ImageKit:
  https://ik.imagekit.io/yourid/hero.jpg?tr=w-1600,h-900,fo-auto,q-80

crabcrop:
  /<signature>/crop:1600x900:smart/filters:quality(80)/https://images.example.com/hero.jpg
```

A named transformation becomes a preset of the same name. The dashboard definition (`w-300,h-200,cm-pad_resize,bg-F5F0E8`) becomes a line in the presets file; note the preset as written behaves like `pad_resize_no_enlarge`, because crabcrop's `pad` never upscales:

```
ImageKit:
  https://ik.imagekit.io/yourid/tr:n-thumb/photos/team.jpg

presets file:
  thumb = pad:300x200:f5f0e8

crabcrop:
  /<signature>/preset:thumb/https://images.example.com/photos/team.jpg
```

A face thumbnail. `fo-face` becomes the `face` alignment, and `z-0.5` becomes a facepad, translated by eye because the scales run in opposite directions:

```
ImageKit:
  https://ik.imagekit.io/yourid/portrait.jpg?tr=w-200,h-200,fo-face,z-0.5

crabcrop:
  /<signature>/crop:200x200:face/filters:facepad(1)/https://images.example.com/portrait.jpg
```

An extract at original scale. The corner and size become two corners:

```
ImageKit:
  https://ik.imagekit.io/yourid/food.jpg?tr=w-300,h-300,cm-extract,x-100,y-300

crabcrop:
  /<signature>/crop-rect:100x300:400x600/https://images.example.com/food.jpg
```

A product shot padded onto a brand color:

```
ImageKit:
  https://ik.imagekit.io/yourid/shoe.png?tr=w-1200,h-1200,cm-pad_resize,bg-F5F0E8,f-webp

crabcrop:
  /<signature>/pad:1200x1200:f5f0e8/filters:format(webp)/https://images.example.com/shoe.png
```

A chained step: resize first, then round and desaturate. The radius is why the second step exists on ImageKit, and the format is pinned because radius needs alpha:

```
ImageKit:
  https://ik.imagekit.io/yourid/badge.png?tr=w-800:r-24,e-grayscale

crabcrop:
  /<signature>/800x0/filters:grayscale:radius(24):format(png)/https://images.example.com/badge.png
```

Background removal, the direct counterpart. The output format is pinned to PNG so the transparency has somewhere to live:

```
ImageKit:
  https://ik.imagekit.io/yourid/product.png?tr=e-bgremove

crabcrop:
  /<signature>/filters:removebg:format(png)/https://images.example.com/product.png
```

A logo layer. The layer's path in the Media Library becomes a full URL, its pixel width becomes a percent of the base width (250 of 1000), and the bottom-right anchor with `N` offsets becomes negative pixel offsets:

```
ImageKit:
  https://ik.imagekit.io/yourid/card.jpg?tr=w-1000,h-1000:l-image,i-assets@@logo.png,w-250,lfo-bottom_right,lx-N20,ly-N20,l-end

crabcrop:
  /<signature>/crop:1000x1000/filters:overlay(https%3A%2F%2Fassets.example.com%2Flogo.png,-20,-20,100,normal,25)/https://images.example.com/card.jpg
```

A text badge. The caption carries; the background pill and the exact 10 px offsets do not (the anchor pads by half the font size instead):

```
ImageKit:
  https://ik.imagekit.io/yourid/hoodie.jpg?tr=w-394,h-540:l-text,i-SALE,fs-30,co-FFFFFF,bg-EF6823,pa-8_8,lfo-top_left,lx-10,ly-10,l-end

crabcrop:
  /<signature>/crop:394x540/filters:text(SALE,30,ffffff,top-left)/https://images.example.com/hoodie.jpg
```

A signed, expiring URL. The timestamp copies exactly; what changes is the scheme (HMAC-SHA1 in query parameters becomes HMAC-SHA256 as a path segment) and the failure code past expiry (401 becomes 403):

```
ImageKit:
  https://ik.imagekit.io/yourid/report.jpg?tr=w-1024&ik-t=1893456000&ik-s=<hmac-sha1>

crabcrop:
  /<signature>/exp:1893456000/1024x0/https://images.example.com/report.jpg
```
