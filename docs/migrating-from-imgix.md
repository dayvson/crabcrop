# Migrating from imgix

imgix is a hosted service: you point a Source at your storage, and transformations ride on the delivery URL as query parameters. crabcrop is a server you run in front of the same storage, and transformations are path segments in a signed URL. The assets do not move; the URL templates do. There is no mechanical imgix converter (the `convert-thumbor` verb covers Thumbor), so templates translate by hand using the tables below. Each row is marked direct, approximate, or unsupported, and the adjustment rows say exactly when a number changes meaning in translation, because copying those verbatim is how a migration ships wrong renders nobody notices.

## Concepts

| imgix | crabcrop |
| --- | --- |
| A Source (S3, GCS, Azure, Web Folder, Web Proxy) bound to an imgix subdomain; the URL path is the asset's key under the Source | The full source URL spelled out in every request, fetched from your origin at request time |
| Web Proxy Sources, whose path holds the whole source URL | The closest imgix shape to crabcrop's model; bucket and folder Sources need their configured prefix joined back onto each path |
| Parameters in the query string, order-free, applied in imgix's own pipeline order | Operations as path segments, applied in URL order: geometry first, `filters:` last |
| Signing optional per Source: MD5 of the per-Source token and the URL, appended as `s=` | Every URL signed with hex HMAC-SHA256 (`CRABCROP_SECRET`), or `unsafe` behind `CRABCROP_ALLOW_UNSAFE` |
| `expires=<unix-seconds>` query parameter; past it, imgix answers 404 | An `exp:<unix-seconds>` segment inside the signed path; past it, crabcrop answers 403 |
| `auto=format` for per-browser format negotiation | Negotiation is built in (AVIF, JPEG XL, then WebP, by `Accept` q-values); no parameter needed |
| imgix.js and the other URL libraries, which add an `ixlib` parameter | The crabcrop SDKs and the `crabcrop-url` crate (`build_url`, `build_srcset`); `ixlib` has no equivalent and is dropped |
| Purge from the imgix dashboard or API clears the imgix CDN | `POST /purge/...` per URL, `POST /purge-group/...` per preset, or the CLI verbs, clear crabcrop's own cache tiers |
| Asset Manager, uploads, the Video API, the DAM | Out of scope: crabcrop is delivery-time transformation only |

## Signing

The schemes do not convert, in either direction. An imgix signature is an MD5 over the Source's token and the full URL (path plus query string), carried in the URL as `s=`. A crabcrop signature is an HMAC-SHA256 hex string over the path remainder, carried as the first path segment. Pick a `CRABCROP_SECRET` and re-sign templates with the SDKs or `crabcrop sign`.

Two operational differences matter. First, imgix signing is opt-in per Source (Web Proxy Sources require it); crabcrop signs everything unless the server deliberately runs with `CRABCROP_ALLOW_UNSAFE=1`. Second, expiry: imgix's `expires` is a plain query parameter anyone can edit unless the URL is also signed, and expired URLs answer 404. crabcrop's `exp:` segment sits inside the signed remainder, so extending it without the secret breaks the signature, and expired URLs answer 403. Rotating a secret invalidates outstanding URLs on both systems; crabcrop's optional `<key-id>:` signature prefix exists so rotation can overlap two secrets instead.

## Resizing and cropping

| imgix | crabcrop | Mapping |
| --- | --- | --- |
| `w=300` alone (or `h=` alone) | `300x0` | Direct: one axis set, the other proportional |
| `w` + `h` with no `fit` (the default `fit=clip`) | `fit-in:WxH` | Direct: fit inside the box, aspect kept |
| `fit=scale` | `WxH` | Direct: exact resize, aspect not kept, in both |
| `fit=crop` (centered) | `crop:WxH` | Direct: crop-to-fill in both |
| `crop=top`, `bottom`, `left`, `right` | `crop:WxH:top` and the same-named alignments | Direct; crabcrop also has the four corner alignments |
| `crop=entropy`, or `fit=entcrop` | `crop:WxH:smart` | Direct in intent: both anchor on the point of greatest detail. With a face model configured, crabcrop's `smart` also keeps detected faces whole |
| `crop=edges` | `crop:WxH:smart` | Approximate: edge detection becomes saliency-plus-faces |
| `crop=faces` | `crop:WxH:smart` | Approximate: `smart` keeps faces whole inside the window rather than centering on them; `crop:WxH:face` centers the largest face |
| `fit=facearea`, with `facepad=1.5` | `crop:WxH:face` plus `filters:facepad(1.5)` | Approximate: crabcrop frames the largest detected face and requires `CRABCROP_FACE_MODEL`; `faceindex` (pick the Nth face) has no equivalent |
| `crop=focalpoint` with `fp-x`, `fp-y` | `focal:<rect>` before `crop:WxH:smart` | Approximate, and only with source dimensions in hand: the fractions become pixels, and a point becomes a small rectangle around it (see the worked example). `fp-z` shrinks that rectangle to zoom; `fp-debug` maps to the `debug:focal` operation |
| `crop` fallback chains (`crop=faces,entropy`) and the object-class values (`crop=car`, subject detection) | One alignment per crop | Unsupported as written: pick the first chain entry that matters and use it |
| `fit=max` | `fit-in:WxH` plus `filters:no_upscale` | Direct: fit inside, never upscale |
| `fit=min` | `crop:WxH` plus `filters:no_upscale` | Direct: crop to the aspect, capped at the source size |
| `fit=fill` with `fill=solid` and `fill-color` | `pad:WxH:<color>` | Direct, with one default flipped: an unnamed imgix fill is white or transparent, an unnamed crabcrop pad fill is black, so name the color. On sources smaller than the box, imgix may upscale the image into the box and crabcrop's `pad` never does |
| `fit=fill` with `fill=blur` | `pad:WxH:blur` | Direct: ambient blurred bars in both |
| `fit=fillmax` | `pad:WxH:<color>` | Direct: pad already refuses to upscale |
| `fit=clamp` (edge pixels extended) | No equivalent | Unsupported: `pad:WxH:blur` is the closest look, not the same pixels |
| `ar=16:9` with `w` or `h` | Compute the other axis, or `crop:WxH` at that ratio | Approximate: the ratio is template arithmetic; there is no `ar` parameter |
| `rect=x,y,w,h` | `crop-rect:<x>x<y>:<x+w>x<y+h>` | Direct, after the corner math: crabcrop takes two corners, imgix a corner and a size |
| Fractional `w` / `h` (0 to 1, a ratio of the source) | No equivalent | Unsupported mechanically: pixels need the source dimensions |
| `dpr=2` | `filters:dpr(2)`, or the `DPR` / `Sec-CH-DPR` request headers | Approximate: crabcrop accepts exactly 1, 1.5, 2, and 3, so imgix's 4, 5, and in-between values do not convert |
| `ch=Width,DPR,...` (client hints opt-in) | Nothing to translate | Direct: crabcrop reads the width, DPR, and Save-Data hints from request headers natively; drop `ch` |

## Format and output

| imgix | crabcrop | Mapping |
| --- | --- | --- |
| `fm=jpg`, `png`, `webp`, `avif`, `jxl`, `gif` | `filters:format(...)` | Direct; or drop `fm` and let negotiation choose |
| `fm=pjpg` | `filters:format(jpeg)` | Approximate: progressive encoding is not a separate switch |
| `fm=png8`, `png32`, `jp2`, `jxr` | No equivalent | Unsupported |
| `q=80` | `filters:quality(80)` | Direct: the values copy. The defaults do not (imgix 75, crabcrop 80), and imgix `q=0` has no crabcrop spelling (1 is the floor) |
| `lossless=true` | `filters:quality(100)` for WebP | Approximate: 100 selects crabcrop's lossless WebP encode; PNG is lossless at any setting |
| `auto=compress` | `filters:quality(45)` captures its pinned quality | Approximate: its metadata stripping is crabcrop's default, but its habit of overriding `fm` to JPEG for opaque images does not carry; an explicit `format(...)` in crabcrop stays |
| `auto=format` | Nothing | Direct: negotiation is built in |
| `cs=strip`, `cs=srgb` | Nothing (the default) | Direct: crabcrop converts to sRGB and drops profiles unless `filters:keep_profile` is set, which is the inverse of `cs` preservation values |
| `fm=json` (metadata as JSON) | Fetch the image URL with `Accept: application/json`, or `crabcrop meta` | Direct |
| `dl=<filename>` (force download) | No equivalent | Unsupported: a `Content-Disposition` concern for your app or CDN layer |

## Adjustments: where numbers change meaning

imgix adjustments mostly run -100 to 100 with 0 as unchanged. crabcrop's adjustment family was not built on one scale, so this is the section to translate carefully.

| imgix | crabcrop | Mapping |
| --- | --- | --- |
| `con=20` | `filters:contrast(20)` | Direct, and the only adjustment that copies verbatim: both are percents, -100 to 100, 0 unchanged |
| `sat=-30` | `filters:saturation(0.7)` | Value changes meaning: imgix is a percent (0 unchanged, -100 grayscale), crabcrop a factor (1 unchanged, 0 grayscale). Convert with `1 + sat/100`; copying the number renders a different image |
| `bri=10` | `filters:brighten(26)` | Approximate: imgix scales brightness, crabcrop adds a channel shift, so rescale with `bri x 2.55` and expect a close match, not an identical one |
| `hue=15` | `filters:huerotate(15)` | Direct: degrees on both; small rendering differences between any two hue implementations are normal |
| `exp=20` (exposure) | No equivalent | Unsupported: `filters:curve(...)` is the closest tool for an exposure-like lift. Note the name collision in the differences below |
| `gam`, `shad`, `high` | No equivalent | Unsupported as parameters: all three are tone-curve shapes, approximated per image with `filters:curve(...)` |
| `vib` | No equivalent | Unsupported: `saturation(...)` is the blunt substitute and does not protect skin tones |
| `sharp=40` | `filters:sharpen(0.4)` | Approximate: a 0 to 100 luminance-sharpening strength becomes an unsharp strength, rescaled by 100 |
| `usm=2` with `usmrad` | `filters:sharpen(2)` | Approximate: both are unsharp masks, but the radius has no equivalent; crabcrop's kernel is fixed |
| `invert=true` | `filters:invert` | Direct |
| `auto=enhance` | No equivalent | Unsupported: `filters:equalize` is a different, stronger operation, not a match |
| `auto=redeye` | No equivalent | Unsupported: crabcrop's face filters (`blurfaces`, `pixelatefaces`) are redaction, not correction |

## Effects, overlays, and text

| imgix | crabcrop | Mapping |
| --- | --- | --- |
| `blur=200` (0 to 2000 strength) | `filters:blur(2)` | Approximate: the strength scale becomes a Gaussian sigma, roughly value / 100 |
| `rot=90` | `filters:rotate(-90)` | Different twice: imgix counts counter-clockwise and zooms to cover the frame; crabcrop counts clockwise and expands the canvas, filling corners with the `fill` color or transparency. The sign flips and the framing differs |
| `flip=h`, `v`, `hv` | `filters:flip(h)`, `flip(v)`, or both chained | Direct |
| `trim=auto` / `trim=color` with `trim-color`, `trim-tol` | `filters:trim(<tolerance>)` | Approximate: crabcrop trims the top-left corner color within a 0 to 255 tolerance (default 10), and the tolerances are different scales, so `trim-tol` does not copy numerically. `trim=alpha` has no exact equivalent |
| `bg=<color>` | `filters:fill(<hex>)`, or the `pad:` fill | Direct for hex colors; color keywords and alpha hex do not carry, crabcrop colors are RRGGBB |
| `pad=20` (and per-side `pad-top` and friends) | Resize smaller, then `pad:WxH:<color>` | Approximate: imgix adds padding around the result; crabcrop pads to an exact canvas, so the box math is yours |
| `border=4,000000` | `filters:border(4,000000)` | Direct: the border overlaps the image in both. Per-side widths and alpha colors do not map |
| `corner-radius`, `mask=corners` | `filters:radius(20)` | Direct; `mask=ellipse` maps to `filters:mask(ellipse)`. Other mask shapes (hearts, stars) are unsupported |
| `duotone=<dark>,<light>` | `filters:duotone(<dark>,<light>)` | Direct for two hex colors; `duotone-alpha` (blending the effect back) has no equivalent |
| `monochrome=<color>` | `filters:monochrome(<color>)` | Direct for hex colors; the alpha-as-intensity form does not carry |
| `mark=<url>` with `mark-align`, `mark-pad`, `mark-alpha` | `filters:watermark(<url>,<x>,<y>,<alpha>)` | Approximate: the nine grid positions become pixel offsets, and negative offsets anchor from the right and bottom edges, which covers `mark-align=bottom,right` with a pad |
| `mark-scale=25` | `filters:overlay(<url>,<x>,<y>,<alpha>,normal,25)` | Approximate: overlay's width is a percent of the base width. `mark-rot` and `mark-tile` are unsupported |
| `blend=<image>` with `blend-mode` | `filters:overlay(<url>,<x>,<y>,<alpha>,<mode>)` | Partial: only `normal`, `multiply`, `screen`, and `overlay` of imgix's blend modes exist; the rest are unsupported, as is blending with a color |
| `txt` and the typesetting family (`txt-font`, `txt-size`, `txt-align`, `txt-w`, `txt-pad`, `txt-shad`, ...) | `filters:text(<text>,<size>,<color>,<anchor>)` | Partial: one caption, in the font configured on the server (`CRABCROP_FONT_PATH`), at a grid anchor. Font choice, alignment, text wrapping, line height, and shadows do not map |
| `px` (pixellate) | `filters:pixelate(<n>)` | Approximate: crabcrop's argument is a block size in pixels |
| Background removal | `filters:removebg` | Approximate: both cut the subject to transparency with a segmentation model; crabcrop's model is operator-configured (`CRABCROP_REMOVEBG_MODEL`) and the masks will differ in detail |
| `upscale=true` (super resolution) | `filters:upscale(2)` or `filters:upscale(4)` | Approximate: imgix picks the factor from the source size; crabcrop takes it explicitly, behind an operator-configured upscale model |
| `palette=json` / `palette=css`, `colors` | No equivalent | Unsupported: palette extraction is analysis, not rendering, and meta responses carry no palette. Compute palettes offline if you need them |
| `page=<n>` on PDF sources | No equivalent | Unsupported: crabcrop does not take PDF sources |
| Generative fill, background replace, text-to-image | No equivalent | Unsupported: crabcrop renders from the source image only |

## The differences that bite

**`exp` means exposure on imgix and expiry on crabcrop.** An imgix URL with `exp=20` is asking for an exposure adjustment; a crabcrop URL with `exp:1893456000` is a deadline. imgix's actual expiry parameter is `expires`. Any translator, human or script, that maps `exp` to `exp:` corrupts both the render and the link lifetime.

**Saturation is the silent one.** `sat` and `con` sit side by side in imgix URLs on the same -100 to 100 scale, but only `con` survives copying. `sat=-100` copied into a crabcrop URL is a parse error; `sat=50` copied as `saturation(50)` parses fine and renders at fifty times the intended saturation, the worst kind of migration bug, because every page looks plausible at a glance.

**Bucket Sources hide their base.** For an S3 or Web Folder Source, the imgix path is a key, and the bucket and prefix live in the Source configuration, not the URL. crabcrop needs the whole source URL, so migration joins prefix and key per asset, in your application or templates. Nothing in the imgix URL alone tells you the prefix.

**Parameters are order-free on imgix, ordered on crabcrop.** imgix runs its own pipeline order regardless of how the query string is written. crabcrop applies operations in path order, so a translated template that puts `filters:` before the crop resizes and crops different pixels than imgix did.

**Defaults differ quietly.** Quality defaults to 75 on imgix and 80 on crabcrop. An unnamed pad fill is white or transparent on imgix and black on crabcrop. Blur and trim tolerances are different scales on the same-looking numbers.

**Purging moves in-house.** An imgix purge clears imgix's CDN copy. On crabcrop, purge the server's tiers with `purge` or `purge-group`; if you also keep a CDN in front of crabcrop, purge crabcrop first and the CDN second, so a concurrent request cannot refill the edge from stale tiers (the recipes in `deploy/` cover the major CDNs).

## Converter feasibility

Could a `convert-imgix` verb sit next to `convert-thumbor`? The mechanical core could: imgix URLs parse cleanly (a path plus a query string), most size, format, quality, and adjustment parameters have the equivalents above, and the Thumbor converter's pattern, convert what is faithful and drop the rest with a named warning rather than guessing, fits.

Three things keep it from being a weekend verb. First, the source: a Thumbor URL carries its source in full, but a bucket Source's imgix URL does not, so the verb needs the Source base supplied out of band before it can emit a crabcrop URL at all. Second, fraction-valued parameters (`fp-x` and `fp-y`, fractional `w` and `h`, the relative layout values in the `txt` and `mark` families) need the source's pixel dimensions, which a converter that does not fetch the source cannot know; those become warnings and dropped parameters, not conversions. Third, the surface doubles itself: every parameter has a Base64 twin (`txt64`, `mark64`, `blend64`, and so on) that must be decoded first, `mark` and `blend` values can nest whole imgix URLs with their own parameters inside them, and `crop` fallback chains have no target spelling. None of that blocks a useful verb; all of it argues for shipping the tables first, which is what this guide is.

## Rolling out

Run both side by side. crabcrop fetches the same origins imgix does, so templates can switch per page, per route, or behind a flag, and either system can serve while the other warms up. Translate templates, not individual asset URLs: the template set is finite, and the SDKs re-sign everything they build.

Expect cold caches. crabcrop renders on first request and then serves from its memory and disk tiers; pre-warm the hot templates by requesting them once after cutover, or let a CDN in front absorb the first-hit cost.

Old imgix URLs age out slowly, so plan for it. A signed imgix URL stays valid on imgix until its Source is disabled or the token rotates; signing alone sets no deadline (only `expires=` does, answering 404 past it). The practical sequence: switch the templates, keep the Source deployed while shared and bookmarked links drain, then disable the Source (disabled Sources answer 410 Gone for anything not still cached) and rotate its token last, so a missed template fails loudly instead of serving forever.

## Worked examples

A hero crop. `auto=format` drops out, because negotiation is built in, and `crop=entropy` becomes the `smart` alignment:

```
imgix:
  https://yourbrand.imgix.net/hero.jpg?w=1600&h=900&fit=crop&crop=entropy&auto=format&q=80

crabcrop:
  /<signature>/crop:1600x900:smart/filters:quality(80)/https://images.example.com/hero.jpg
```

A face thumbnail, using `fit=facearea` and its padding:

```
imgix:
  https://yourbrand.imgix.net/portrait.jpg?w=200&h=200&fit=facearea&facepad=1.5

crabcrop:
  /<signature>/crop:200x200:face/filters:facepad(1.5)/https://images.example.com/portrait.jpg
```

A product shot padded onto a brand color:

```
imgix:
  https://yourbrand.imgix.net/shoe.png?w=1200&h=1200&fit=fill&fill=solid&fill-color=F5F0E8&fm=webp

crabcrop:
  /<signature>/pad:1200x1200:f5f0e8/filters:format(webp)/https://images.example.com/shoe.png
```

An adjustment stack. Contrast copies; brightness rescales; saturation converts from percent to factor:

```
imgix:
  https://yourbrand.imgix.net/street.jpg?w=800&bri=10&con=20&sat=-30&hue=15&sharp=40

crabcrop:
  /<signature>/800x0/filters:brighten(26):contrast(20):saturation(0.7):huerotate(15):sharpen(0.4)/https://images.example.com/street.jpg
```

A focal-point crop. The source is 2000x1333, so `fp-x=0.75` and `fp-y=0.4` land on the point (1500, 533), which becomes a 100px focal rectangle around it:

```
imgix:
  https://yourbrand.imgix.net/puppy.jpg?w=600&h=600&fit=crop&crop=focalpoint&fp-x=0.75&fp-y=0.4

crabcrop:
  /<signature>/focal:1450x483:1550x583/crop:600x600:smart/https://images.example.com/puppy.jpg
```

A watermarked share card. The grid position and pad become negative pixel offsets, and `mark-scale` moves to the overlay filter's width argument:

```
imgix:
  https://yourbrand.imgix.net/card.jpg?w=1200&h=630&fit=crop&mark=https%3A%2F%2Fassets.example.com%2Flogo.png&mark-align=bottom,right&mark-pad=24&mark-alpha=80&mark-scale=25

crabcrop:
  /<signature>/crop:1200x630/filters:overlay(https%3A%2F%2Fassets.example.com%2Flogo.png,-24,-24,80,normal,25)/https://images.example.com/card.jpg
```

A signed, expiring URL. The timestamp copies exactly; what changes is that the expiry now sits inside the signature instead of beside it:

```
imgix:
  https://yourbrand.imgix.net/report.jpg?w=1024&expires=1893456000&s=<md5>

crabcrop:
  /<signature>/exp:1893456000/1024x0/https://images.example.com/report.jpg
```

One member of a DPR srcset. The imgix libraries generate these in families (dpr 1 through 5, quality stepped down as dpr rises); build the same families with `build_srcset` and per-member quality:

```
imgix:
  https://yourbrand.imgix.net/bluehat.jpg?w=400&dpr=2&fit=max&q=50

crabcrop:
  /<signature>/400x0/filters:no_upscale:dpr(2):quality(50)/https://images.example.com/bluehat.jpg
```
