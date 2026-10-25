# Migrating from Cloudflare Images

Cloudflare Images is a hosted service with two delivery shapes: images stored in Cloudflare and served by ID from `imagedelivery.net`, and images on your own origin transformed through `/cdn-cgi/image/` on your zone. crabcrop is a server you run in front of your own storage, and transformations are path segments in a signed URL. The zone shape moves by translating URL templates; the hosted shape first needs its originals back in your storage (see Rolling out). There is no mechanical Cloudflare converter (the `convert-thumbor` verb covers Thumbor), so templates translate by hand using the tables below. Each row is marked direct, approximate, or unsupported, and the adjustment rows say exactly when a number changes meaning in translation, because copying those verbatim is how a migration ships wrong renders nobody notices.

## Concepts

| Cloudflare Images | crabcrop |
| --- | --- |
| A hosted image: an original uploaded to Cloudflare storage, addressed by account hash and image ID (a UUID, or a custom ID in your own path shape) | The full source URL spelled out in every request, fetched from your origin at request time; there is no asset registry or ID namespace |
| A named variant (up to 100 per account), defined in the dashboard or API with sizing, fit, and metadata options | A preset: an operation string defined on the server (`CRABCROP_PRESETS_FILE`), referenced as `preset:<name>` |
| Flexible variants: arbitrary transformation options in the `imagedelivery.net` URL path, enabled account-wide | Operations as path segments in a signed URL; see Signing for what replaces unsigned flexible variants |
| Zone transformations: `/cdn-cgi/image/<options>/<source>` on your domain, the source an origin path or a full URL | The same source URL carried in the crabcrop path; the options become operation segments |
| Workers `cf.image` options and the Images binding, including `draw` overlays | The crabcrop SDKs and the `crabcrop-url` crate (`build_url`, `build_srcset`); `draw` has filter equivalents in the effects table |
| Signing optional and per image: `requireSignedURLs` on the image, with per-variant public exemptions | Every URL signed with hex HMAC-SHA256 (`CRABCROP_SECRET`), or `unsafe` behind `CRABCROP_ALLOW_UNSAFE` |
| Expiry carried inside the signed token, as an `exp` query parameter | An `exp:<unix-seconds>` segment inside the signed path; past it, crabcrop answers 403 |
| Polish: zone-level compression and WebP conversion applied outside the URL | No equivalent layer: format negotiation and quality live in the URL and are built in. If Polish supplied your WebP conversion or your effective quality, that behavior has no URL to translate; state it in the crabcrop URL or accept crabcrop's defaults |
| Purge from the Cloudflare dashboard or API; `/cdn-cgi/` URLs cannot be purged individually, purging the source URL purges its derivatives | `POST /purge/...` per URL, `POST /purge-group/...` per preset, or the CLI verbs, clear crabcrop's own cache tiers |
| Image storage, the upload and batch APIs, the dashboard, Workers AI features | Out of scope: crabcrop is delivery-time transformation only |

## Signing

The schemes are cousins, not conversions. A Cloudflare signed URL carries two query parameters: `exp` (a unix timestamp) and `sig`, an HMAC-SHA256 hex over the URL path plus `?exp=<timestamp>`, keyed with the account signing key. The signature covers the path, so it covers the variant name or the flexible-variant options inside it, but not the host. A crabcrop signature is an HMAC-SHA256 hex over the path remainder, carried as the first path segment, with `exp:` inside the signed remainder. Pick a `CRABCROP_SECRET` and re-sign templates with the SDKs or `crabcrop sign`.

Three operational differences matter. First, scope: Cloudflare signing is opt-in per image (set at upload, changeable later), and any variant can be marked "always allow public access" to exempt itself. crabcrop signs everything unless the server deliberately runs with `CRABCROP_ALLOW_UNSAFE=1`. Second, the unsigned case: a public Cloudflare image with flexible variants enabled accepts any options from anyone, which is often load-bearing in existing templates. crabcrop's deliberate equivalent is `CRABCROP_ONLY_PRESETS=1` with unsafe URLs: browsers and CMSs can build URLs from preset names alone, but nobody can invent a transform that has no preset. Third, rotation: rotating the Cloudflare signing key invalidates every outstanding token at once, and rotating `CRABCROP_SECRET` does the same; crabcrop's optional `<key-id>:` signature prefix exists so rotation can overlap two secrets instead.

## Resizing and cropping

| Cloudflare Images | crabcrop | Mapping |
| --- | --- | --- |
| `width=400` alone (or `height=` alone) | `400x0` | Direct: one axis set, the other proportional |
| `width` + `height` with no `fit` (the default `fit=scale-down`) | `fit-in:WxH` plus `filters:no_upscale` | Direct: fit inside the box, never upscale |
| `fit=scale-down` | `fit-in:WxH` plus `filters:no_upscale` | Direct |
| `fit=contain` | `fit-in:WxH` | Direct: contain upscales a smaller source, and plain `fit-in` does too |
| `fit=cover` | `crop:WxH` | Direct: fill the box, crop the overflow, upscale if needed |
| `fit=crop` | `crop:WxH` plus `filters:no_upscale` | Approximate: for a source smaller than the box, Cloudflare returns the source untouched, original size and aspect; crabcrop still crops to the target aspect, at source scale |
| `fit=aspect-crop` | `crop:WxH` plus `filters:no_upscale` | Direct in outcome: the target aspect is enforced and nothing upscales |
| `fit=pad` with `background` | `pad:WxH:<color>` | Approximate: Cloudflare upscales a smaller image to fill the box; crabcrop fits without upscaling and centers the result on the exact canvas. A default flips: an unnamed Cloudflare pad fill is white (its `background` default), an unnamed crabcrop pad fill is black, so name the color |
| `fit=squeeze` | `WxH` | Direct: exact resize, aspect not kept, in both |
| `fit=scale-up` | `fit-in:WxH` | Approximate: scale-up never downscales, so an oversized source comes back at original size; `fit-in` downscales it into the box. Identical when sources are smaller than the box |
| `gravity=left`, `right`, `top`, `bottom` (with `fit=cover` or `fit=crop`) | `crop:WxH:left` and the same-named alignments | Direct; crabcrop also has the four corner alignments |
| `gravity=auto` | `crop:WxH:smart` | Direct in intent: both anchor on a saliency point. With a face model configured, crabcrop's `smart` also keeps detected faces whole |
| `gravity=face` | `crop:WxH:face` | Approximate: Cloudflare anchors the crop window on the detected face point; crabcrop frames the largest detected face in its own window and requires `CRABCROP_FACE_MODEL` |
| `gravity=0.5x1` (relative XxY coordinates) | `focal:<rect>` before `crop:WxH:smart` | Approximate, and only with source dimensions in hand: the fractions become pixels, and the point becomes a small rectangle around it (see the worked example) |
| `zoom` / `face-zoom` (with `gravity=face`) | `filters:facepad(...)` | Approximate, and the direction inverts: zoom 1.0 is the tightest face crop, while facepad 0 is crabcrop's tightest framing and larger values add background. Translate by eye |
| `dpr=2` | `filters:dpr(2)`, or the `DPR` / `Sec-CH-DPR` request headers | Direct for 1, 1.5, and 2 (Cloudflare caps at 2; crabcrop accepts exactly 1, 1.5, 2, and 3, so values like 1.25 do not convert) |
| `width=auto` with `wbreakpoints`, `wmobile`, `wdesktop` | Nothing to translate | Approximate: when a URL sizes nothing itself, crabcrop honors the `Sec-CH-Width` and `Width` client hints natively, quantized by `CRABCROP_WIDTH_STEP` (default 100). The breakpoint lists and the user-agent fallbacks do not carry |
| `rotate=90`, `180`, `270` | `filters:rotate(90)` and so on | Approximate: the value copies (clockwise, quarter turns), but Cloudflare rotates before resizing, so its width and height describe the rotated image; crabcrop geometry runs before filters, so the crop box describes the unrotated image. Swap the box axes when translating a rotate plus cover or crop combination |
| `flip=h`, `v`, `hv` | `filters:flip(h)`, `flip(v)`, or both chained | Direct. Cloudflare flips before rotation; in a crabcrop filter chain, the order is the order written |

## Format and output

| Cloudflare Images | crabcrop | Mapping |
| --- | --- | --- |
| `format=auto` | Nothing | Direct: negotiation is built in. Two behaviors widen: crabcrop also negotiates JPEG XL, and it honors a client's `q=0` refusal, where Cloudflare's `auto` picks between AVIF, WebP, and JPEG by browser support alone |
| `format=avif`, `format=webp` | `filters:format(...)` | Direct, with one behavioral note: Cloudflare falls back to WebP or JPEG when an image is too large to encode to AVIF quickly; crabcrop encodes what was asked |
| `format=jpeg` | `filters:format(jpeg)` | Approximate: Cloudflare's `jpeg` is progressive encoding |
| `format=baseline-jpeg` | `filters:format(jpeg)` | Approximate: the progressive versus baseline distinction does not exist in crabcrop |
| `format=json` | Fetch the image URL with `Accept: application/json`, or `crabcrop meta` | Direct |
| `quality=80` | `filters:quality(80)` | Direct: the values copy. The defaults do not (Cloudflare 85, crabcrop 80) |
| `quality=high`, `medium-high`, `medium-low`, `low` (perceptual levels) | `filters:quality(auto)`, or pick a number | Approximate: the names do not convert to numbers (Cloudflare publishes no equivalents). `quality(auto)` is the nearest behavior: a per-image choice made by content analysis |
| `quality` set on PNG output (selects the PNG8 palette variant) | No equivalent | Unsupported: crabcrop PNG is full-color and lossless at any setting |
| `slow-connection-quality` / `scq` | Nothing | Approximate at best: crabcrop steps quality down when a client sends `Save-Data: on` and the URL carries no quality of its own. The `rtt`, `ect`, and `downlink` triggers, and a named override value, do not exist |
| `metadata=none` | Nothing (the default) | Direct: crabcrop strips source metadata unless told otherwise |
| `metadata=copyright` | No equivalent | Unsupported as spelled: crabcrop's strip drops the copyright tag too, and its preserve mode keeps everything. Copyright-only retention has no setting on either the URL or the server |
| `metadata=keep` | Server configuration, not a URL option | Approximate: the server-wide metadata mode `preserve` keeps EXIF, XMP, and IPTC on JPEG outputs and EXIF and XMP on PNG outputs, for every response. Cloudflare's `keep` is per URL and JPEG-only (other formats discard metadata there too) |
| `anim=true` | Keep an animated output format | Direct in effect: frames survive only into animated outputs on both systems (`format(gif)`, or `format(webp)` for an animated WebP source) |
| `anim=false` | Any explicit still `format(...)` | Direct: a still output uses the first frame |
| `compression=fast` | No equivalent | Unsupported: the encoder speed versus quality trade-off is not exposed. Note that `compression=fast` usually overrides `format` toward JPEG, so check whether a URL's real format intent is hiding in this option |

## Adjustments: where numbers change meaning

Cloudflare's adjustment options are multipliers with 1 as unchanged. crabcrop's adjustment family was not built on one scale, so this is the section to translate carefully. The headline: from Cloudflare, saturation copies and contrast does not, which is the reverse of the imgix situation.

| Cloudflare Images | crabcrop | Mapping |
| --- | --- | --- |
| `saturation=0.5` | `filters:saturation(0.5)` | Direct, and the only adjustment that copies verbatim: both are factors, 1 unchanged, 0 grayscale |
| `contrast=1.5` | `filters:contrast(50)` | Value changes meaning: Cloudflare is a multiplier (1 unchanged), crabcrop a percent (-100 to 100, 0 unchanged). Convert with `(factor - 1) x 100`. A factor of 0 to 2 spans the whole percent range; above 2, clamp at 100 |
| `brightness=0.9` | `filters:brighten(-26)` as a starting point | Approximate, because the operations differ in shape: Cloudflare multiplies luminance (black stays black, whites scale), crabcrop adds a fixed channel shift from -255 to 255 (black lifts too). There is no faithful conversion; `brighten((factor - 1) x 255)`, rounded, matches near the highlights and is a reasonable place to start tuning |
| `gamma=0.5` | `filters:curve(rgb,0,0,64,128,128,180,192,221,255,255)` | Approximate: crabcrop has no gamma filter, so the power curve is sampled into `curve(...)` control points (this example is `gamma=0.5`, where `out = in^0.5`). Mind the convention: Cloudflare gamma below 1 lightens midtones, above 1 darkens them, and 0 and 1 are both no change |
| `sharpen=2` | `filters:sharpen(2)` | Approximate: both are unsharp-mask strengths and the number carries as a starting point (Cloudflare's scale runs 0 to 10; crabcrop's is any positive number), but the kernels differ |
| `blur=50` | `filters:blur(5)` as a starting point | Approximate: Cloudflare's value is a radius from 0 to 250, crabcrop's a Gaussian sigma. The numbers do not convert by a constant; sigma near radius / 10 is a reasonable first guess, then compare renders |

## Effects, overlays, and text

| Cloudflare Images | crabcrop | Mapping |
| --- | --- | --- |
| `background=%23F5F0E8` (fill transparent pixels) | `filters:fill(f5f0e8)` for flattening, or the `pad:` fill | Direct for hex colors. CSS color names and `rgba()` values with alpha do not carry: crabcrop colors are RRGGBB. Where a fill goes unnamed, the defaults differ: Cloudflare fills white, a crabcrop `pad:` fills black |
| `border` (chiefly a Workers `cf.image` option: uniform or per-side widths, any CSS color) | `filters:border(4,000000)` | Approximate: the uniform-width, hex-color case maps. Per-side widths and alpha colors do not |
| `segment=foreground` | `filters:removebg` plus an alpha-carrying `format(...)` | Approximate: both isolate the subject onto transparency with a segmentation model (Cloudflare runs BiRefNet through Workers AI; crabcrop's model is operator-configured, `CRABCROP_REMOVEBG_MODEL`), and the masks will differ in detail |
| `upscale=generate` (with an upscaling `fit`) | `filters:upscale(2)` or `filters:upscale(4)` | Approximate: Cloudflare's ESRGAN pass picks the nearest supported scale from the enlargement asked for; crabcrop takes the factor explicitly, behind an operator-configured upscale model, and the filter's position in the chain is yours to place |
| `upscale=interpolate` | Nothing | Direct: interpolation is the default enlargement on both |
| `draw` overlays and text (Workers `cf.image` only, never a delivery URL option) | `filters:watermark(...)`, `filters:overlay(...)`, `filters:text(...)` | Partial: positions become pixel offsets or grid anchors, and text renders in the one font configured on the server (`CRABCROP_FONT_PATH`). `draw`'s repeat, rotation, and per-overlay opacity curves do not map |
| `trim=border` with `trim.border.color`, `trim.border.tolerance`, `trim.border.keep` | `filters:trim(<tolerance>)` | Approximate twice: crabcrop always trims the top-left corner color (an auto-detect or a named `trim.border.color` has no other spelling), and the tolerance scale inverts: Cloudflare's 255 means exact match only and 0 means match anything, crabcrop's tolerance is the allowed difference, so convert with `255 - value`. `trim.border.keep` has no equivalent |
| `trim=10;20;10;20`, `trim.top`, `trim.width`, and friends (pixel or fraction cuts per side) | `crop-rect:<left>x<top>:<right>x<bottom>` | Approximate: the corner math needs the source dimensions (fractions especially), and the cut is a geometry segment, so it composes with the rest of the chain as a crop |
| `onerror=redirect` | No URL equivalent | Unsupported: crabcrop's server-configured fallback image serves a substitute image when a source fails, which covers part of the intent, but nothing redirects to the original |

## The differences that bite

**Contrast is the silent one.** `contrast` exists in both grammars, so it looks copyable, and it is the one adjustment that is not. Cloudflare's is a multiplier: `contrast=1.5` means half again as much contrast. crabcrop's is a percent: `contrast(1.5)` parses fine and means a 1.5 percent lift, almost no change. Copied across a whole template library, every image comes out flatter than intended and nothing errors. Convert with `(factor - 1) x 100`, and audit any template that already crossed over by copying.

**Saturation inverts the usual warning.** Teams migrating from imgix learn that saturation is the trap and contrast is safe. From Cloudflare it is exactly backwards: `saturation` is a factor on both systems and copies, `contrast` changes meaning. A migration runbook shared across sources will get one of the two wrong; keep the per-source tables separate.

**Signing is opt-in there, default here.** Cloudflare serves unsigned by default: protection is a flag on each image, and flexible variants on a public image let anyone mint transforms. crabcrop signs everything unless the operator chose otherwise. Inventory which of your URLs rely on being unsigned (hot-linked variants, CMS-built URLs) before cutover, and decide deliberately between signing them with the SDKs and running presets-only mode.

**Rotate runs first there, last here.** Cloudflare applies `rotate` before `width` and `height` are interpreted, so its sizes describe the rotated image. crabcrop applies geometry segments before filters, so `crop:200x300` describes the unrotated image and a following `rotate(90)` turns the result. Translated rotate plus crop templates need their box axes swapped, or the output framing transposes.

**Hosted images are storage, not just URLs.** An `imagedelivery.net` ID names an object in Cloudflare's storage; there is no origin URL inside it to recover (custom IDs are the exception, since they are your paths). crabcrop fetches from your origin, so hosted originals have to be copied back to your storage first. Zone transformations have no such move: their sources already live on your origin.

**Variant edits are in place there; preset edits should not be here.** Changing a Cloudflare variant definition re-renders every URL that names it, once cached copies age out. A crabcrop preset is an operator file, and the discipline is the opposite: never edit a definition in place, because image responses travel as immutable for a year and downstream copies of the old render sit beyond any purge. Roll the change out under a new preset name, update templates, and `purge-group` the old name to clear this server's tiers.

**Defaults differ quietly.** Quality defaults to 85 on Cloudflare and 80 on crabcrop. An unnamed pad fill is white on Cloudflare and black on crabcrop. Cloudflare keeps the EXIF copyright tag by default; crabcrop strips all metadata by default.

**Purging gets more precise, and moves in-house.** Cloudflare cannot purge an individual `/cdn-cgi/` URL; purging the source URL takes its derivatives with it. On crabcrop, purge per URL or per preset group on the server itself. If a CDN also sits in front of crabcrop, purge crabcrop first and the CDN second, so a concurrent request cannot refill the edge from stale tiers (the recipes in `deploy/` cover the major CDNs, including Cloudflare itself).

## Converter feasibility

Could a `convert-cloudflare` verb sit next to `convert-thumbor`? For zone transformation URLs, the mechanical core could: the options are comma-separated `key=value` pairs in a single path segment, the source follows in full, and the Thumbor converter's pattern, convert what is faithful and drop the rest with a named warning rather than guessing, fits. The fit, gravity, format, quality, and copying saturation rows are table lookups; contrast and brightness become conversions with warnings; gamma and the trims become warnings and drops.

Two things keep it from being a weekend verb. First, the hosted form does not carry its source: an account hash and a UUID image ID name nothing on your origin, so the verb needs an ID to URL manifest (exportable from the Images API) or a custom-ID base supplied out of band before it can emit a crabcrop URL at all, the same blocker as imgix bucket Sources. Second, variant names are opaque to the URL: expanding `thumbnail` into operations needs the account's variant definitions, fetched from the API and translated by these tables, after which a variant is just a preset definition waiting to be written. None of that blocks a useful verb for zone URLs; all of it argues for shipping the tables first, which is what this guide is.

## Rolling out

The two delivery shapes migrate differently. Zone transformations need no asset move at all: crabcrop fetches the same origins Cloudflare proxied, so templates can switch per page, per route, or behind a flag, and either system can serve while the other warms up. Hosted images need their originals back first: download each image's original blob through the Images API into your own storage, keeping the ID as the path where custom IDs were used, and mint crabcrop URLs against those paths.

Translate templates, not individual asset URLs: the template set is finite, and the SDKs re-sign everything they build. Turn each named variant into a preset of the same name as you go, so `purge-group` lines up with the variant it replaces: changing what a variant means becomes rolling out a new preset name and purging the old group, as the differences section describes.

Expect cold caches. crabcrop renders on first request and then serves from its memory and disk tiers; pre-warm the hot templates by requesting them once after cutover, or let a CDN in front absorb the first-hit cost.

Old Cloudflare URLs age out on their own schedules, so plan for both. An unsigned imagedelivery or zone URL stays valid as long as the image or zone serves it, which can be forever; shared and bookmarked links keep working until you delete the hosted image or turn the feature off, so drain them deliberately. A signed URL carries its own deadline in `exp` and dies there; there is no per-token revocation, only rotating the account signing key, which kills every outstanding token at once. The practical sequence: switch the templates, copy hosted originals back and delete the hosted copies once links drain, and rotate the signing key last to cut off any stragglers still holding tokens.
## Worked examples

A hero crop on a zone. `format=auto` drops out, because negotiation is built in, and `gravity=auto` becomes the `smart` alignment:

```
Cloudflare:
  https://example.com/cdn-cgi/image/width=1600,height=900,fit=cover,gravity=auto,format=auto,quality=80/images/hero.jpg

crabcrop:
  /<signature>/crop:1600x900:smart/filters:quality(80)/https://images.example.com/hero.jpg
```

A named variant becomes a preset of the same name. The variant definition (`fit: scale-down`, `width: 200`, `height: 200`, `metadata: none`) becomes a line in the presets file; metadata stripping is crabcrop's default, so it needs no spelling:

```
Cloudflare:
  https://imagedelivery.net/<account-hash>/<image-id>/thumbnail

presets file:
  thumbnail = fit-in:200x200/filters:no_upscale

crabcrop:
  /<signature>/preset:thumbnail/https://images.example.com/photos/team.jpg
```

A face crop. `gravity=face` becomes the `face` alignment, and `zoom=0.5` becomes a facepad, translated by eye because the scales run in opposite directions:

```
Cloudflare (flexible variant):
  https://imagedelivery.net/<account-hash>/<image-id>/width=400,height=400,fit=cover,gravity=face,zoom=0.5

crabcrop:
  /<signature>/crop:400x400:face/filters:facepad(1)/https://images.example.com/portrait.jpg
```

A focal-point crop. The source is 2000x1333, so `gravity=0.75x0.4` lands on the point (1500, 533), which becomes a 100px focal rectangle around it:

```
Cloudflare (flexible variant):
  https://imagedelivery.net/<account-hash>/<image-id>/width=600,height=600,fit=cover,gravity=0.75x0.4

crabcrop:
  /<signature>/focal:1450x483:1550x583/crop:600x600:smart/https://images.example.com/puppy.jpg
```

An adjustment stack, the section to translate slowly. Contrast converts from factor to percent; saturation copies; sharpening carries as a starting point; brightness starts from the rescaled value and gets tuned by eye:

```
Cloudflare:
  https://example.com/cdn-cgi/image/width=800,brightness=0.9,contrast=1.5,saturation=0.5,sharpen=2/images/street.jpg

crabcrop:
  /<signature>/800x0/filters:brighten(-26):contrast(50):saturation(0.5):sharpen(2)/https://images.example.com/street.jpg
```

A product shot padded onto a brand color. The `background` hex becomes the pad fill, named explicitly because the unnamed defaults differ:

```
Cloudflare:
  https://example.com/cdn-cgi/image/width=1200,height=1200,fit=pad,background=%23F5F0E8,format=webp/images/shoe.png

crabcrop:
  /<signature>/pad:1200x1200:f5f0e8/filters:format(webp)/https://images.example.com/shoe.png
```

Background removal. `segment=foreground` becomes `removebg`, and the output format is pinned to PNG so the transparency has somewhere to live:

```
Cloudflare (flexible variant):
  https://imagedelivery.net/<account-hash>/<image-id>/width=800,segment=foreground

crabcrop:
  /<signature>/800x0/filters:removebg:format(png)/https://images.example.com/product.png
```

A gamma lift, sampled into a tone curve:

```
Cloudflare:
  https://example.com/cdn-cgi/image/gamma=0.5/images/street.jpg

crabcrop:
  /<signature>/filters:curve(rgb,0,0,64,128,128,180,192,221,255,255)/https://images.example.com/street.jpg
```

A signed, expiring URL. The timestamp copies exactly; what changes is that the expiry now sits inside the signature instead of beside it, and scale-down becomes fit-in with the no-upscale cap:

```
Cloudflare:
  https://imagedelivery.net/<account-hash>/<image-id>/w=800,fit=scale-down?exp=1893456000&sig=<hmac>

crabcrop:
  /<signature>/exp:1893456000/fit-in:800x0/filters:no_upscale/https://images.example.com/report.jpg
```
