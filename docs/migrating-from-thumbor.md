# Migrating from Thumbor

Thumbor's URL grammar is crabcrop's closest relative, and most Thumbor URLs convert mechanically. The `convert-thumbor` CLI verb does the conversion, including re-signing: it maps what has an equivalent, and reports everything it had to drop or approximate instead of guessing.

```bash
export CRABCROP_BASE=https://img.example.com
export CRABCROP_SECRET=<your secret>

crabcrop convert-thumbor "https://thumbor.example.com/unsafe/300x200/smart/filters:quality(80):format(webp)/https://images.example.com/photo.jpg"
# https://img.example.com/<signature>/crop:300x200:smart/filters:quality(80):format(webp)/https://images.example.com/photo.jpg
```

Losses are printed to stderr, one per line, so the URL on stdout stays clean:

```bash
crabcrop convert-thumbor "unsafe/filters:brightness(20):focal(10x10:50x50)/https://images.example.com/photo.jpg"
# warning: brightness(20) approximated as brighten(51): Thumbor scales brightness, crabcrop shifts it
# https://img.example.com/<signature>/focal:10x10:50x50/filters:brighten(51)/https://images.example.com/photo.jpg
```

## Concepts

| Thumbor | crabcrop |
| --- | --- |
| `/<signature>/<operations>/<source>` | Same shape: `/<signature>/<operations>/<source>` |
| Signature: URL-safe base64 HMAC-SHA1 of the operations and source | Hex HMAC-SHA256, over the same kind of remainder |
| `unsafe` URLs, gated by `ALLOW_UNSAFE` | `unsafe` URLs, gated by `CRABCROP_ALLOW_UNSAFE` |
| Security key in the server config | `CRABCROP_SECRET` (one secret, minimum 16 bytes) |
| `/meta/...` URL prefix | The same URL fetched with `Accept: application/json`, or `crabcrop meta` |
| Result caching is your CDN's job; Thumbor has no built-in result cache | Two built-in cache tiers (memory, then disk), `Cache-Status` on every image response |
| No cache purge endpoint | `POST /purge/...` for one URL, `POST /purge-group/...` for a preset group, or the `purge` / `purge-group` verbs |
| Presets: none; every URL spells its operations | Optional named presets (`preset:<name>`), expandable server-side and usable as cache groups |
| Filters run after the resize | Operations run in URL order; a `filters:` segment can appear anywhere |

## Operations

| Thumbor | crabcrop | Notes |
| --- | --- | --- |
| `trim` | `filters:trim` | crabcrop's default tolerance applies |
| `trim:<corner>:<tolerance>` | `filters:trim(<tolerance>)` | The corner is dropped, with a warning: crabcrop trim reads the border itself |
| `<l>x<t>:<r>x<b>` crop | `crop-rect:<l>x<t>:<r>x<b>` | A rectangle in both; cut before the size |
| `<W>x<H>`, both set | `crop:<W>x<H>` | Thumbor's fixed size is a crop-to-fill, and so is crabcrop's `crop:` |
| `<W>x0` or `0x<H>` | `<W>x0` / `0x<H>` | A proportional resize in both grammars |
| `orig` as a dimension | `orig` | Same meaning |
| Negative dimension (`-300x200`) | Inside a plain size; a `flip(h)` / `flip(v)` filter where the size takes no flips (`crop:`, `fit-in:`) | A flip on that axis in both |
| `fit-in` | `fit-in:` | Flips become `flip(h)` / `flip(v)` filters, since crabcrop's fit-in size takes none |
| `adaptive-fit-in` | `fit-in:<W>x<H>:adaptive` | Converted exactly; the box swaps when the source orientation disagrees with it |
| `full-fit-in` | `fit-in:` | Converted with a warning; the whole image stays inside the box |
| `left` / `center` / `right`, `top` / `middle` / `bottom` | The crop alignment (`crop:300x200:left`) | Corner pairs map to the corner alignments (`crop:300x200:top-left`, `top-right`, `bottom-left`, `bottom-right`); a centered side collapses to the plain edge |
| `smart` | `crop:<W>x<H>:smart` | Focal-point crop in both |
| `filters:focal(<l>x<t>:<r>x<b>)` | `focal:<l>x<t>:<r>x<b>` | A filter in Thumbor, an operation in crabcrop; the converter places the segment ahead of the crop it steers |

## Filters

Same-name filters with the same meaning: `quality`, `format` (jpeg, png, gif, webp, avif), `max_bytes`, `no_upscale`, `grayscale`, `equalize`, `rotate` (90 degree steps), `sharpen`, `watermark` (the URL argument is percent-encoded into its segment), and `fill` with a 6-digit hex color.

Filters whose scale differs:

| Thumbor | crabcrop | Conversion |
| --- | --- | --- |
| `contrast(20)` | `contrast(1.2)` | Percent becomes a factor: `1 + n/100` |
| `saturation(-50)` | `saturation(0.5)` | Same conversion |
| `brightness(20)` | `brighten(51)` | Approximate, and the converter says so: Thumbor scales brightness multiplicatively, crabcrop shifts it additively |
| `rgb(10,0,-10)` | `rgb(26,0,-26)` | Approximate, same reason: channel scaling becomes additive offsets |
| `blur(2)` | `blur(2)` | The value carries over; the two blurs weight radius and sigma slightly differently |

Filters with no equivalent are dropped and named in the warnings: `frame(...)`, `background_color(...)`, per-format encoder options (`webp_options`, `jpeg_options`, and friends), and anything else the converter does not know. `fill(auto)` is dropped too. `strip_exif()` and `strip_icc()` need no conversion: metadata handling is server policy in crabcrop (`CRABCROP_METADATA`, `strip` by default), not a per-URL option.

## The differences that bite

**Signatures are incompatible.** Thumbor signs with HMAC-SHA1 and base64; crabcrop signs with HMAC-SHA256 and hex. No Thumbor signature can be carried over, so every URL template must be re-signed. That is why the migration bridge is crabcrop's unsafe mode or presets-only mode (`CRABCROP_ONLY_PRESETS`): run crabcrop unsigned behind your existing access control while templates are converted, then turn signing on.

**A fixed size is a crop.** If you have been treating Thumbor's `300x200` as a plain resize, it never was one: it crops to fill. crabcrop makes the distinction explicit (`crop:300x200` fills, `300x200` resizes to exactly 300x200, `fit-in:300x200` fits inside). The converter maps Thumbor's behavior, not its spelling.

**Metadata is policy, not URL.** Per-URL `strip_exif()` calls become one server setting. Color profiles are converted to sRGB on the way through unless a URL asks for `filters:keep_profile`.

## A cutover recipe

1. Run crabcrop next to Thumbor, pointed at the same origins, with `CRABCROP_ALLOW_UNSAFE=1` (or presets-only) while you convert.
2. Convert your URL *templates* with `convert-thumbor`, one per layout, and read every warning. Signed output needs the target secret; the converter validates each result against the server grammar before printing it.
3. Compare outputs on a sample of real images. Expect pixel differences in `brightness` and `rgb`; those are the documented approximations.
4. Flip traffic, then set `CRABCROP_SECRET` enforcement by removing the unsafe allowance once every template is signed.

## Worked examples

```
unsafe/300x200/filters:quality(80):grayscale()/https://images.example.com/photo.jpg
-> /<signature>/crop:300x200/filters:quality(80):grayscale/https://images.example.com/photo.jpg

unsafe/10x20:110x220/fit-in/100x100/https://images.example.com/photo.jpg
-> /<signature>/crop-rect:10x20:110x220/fit-in:100x100/https://images.example.com/photo.jpg

unsafe/-300x200/smart/https://images.example.com/photo.jpg
-> /<signature>/filters:flip(h)/crop:300x200:smart/https://images.example.com/photo.jpg
```
