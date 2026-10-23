# Migrating from Cloudinary

Cloudinary is a hosted media service: assets are uploaded to it, and delivery URLs transform them on the way out. crabcrop is delivery only: it fetches an origin URL at request time and transforms that. Migrating means mapping your assets' public IDs to origin URLs once, then translating the transformation strings with the tables below.

## Concepts

| Cloudinary | crabcrop |
| --- | --- |
| `res.cloudinary.com/<cloud>/image/upload/<transformations>/<public_id>.<ext>` | `/<signature>/<operations>/<source URL>` on your own host |
| Transformations: comma-joined `key_value` pairs, chained with `/` | Operations: path segments, applied in order |
| Delivery URLs usually unsigned; signed URLs and the upload API use SHA-1 over sorted parameters with the API secret | Every URL signed with hex HMAC-SHA256 (`CRABCROP_SECRET`), or `unsafe` behind `CRABCROP_ALLOW_UNSAFE` |
| Assets live in Cloudinary (public IDs); `image/fetch` wraps a remote URL | The source is always a URL crabcrop can fetch from your origin |
| The `v1234567` version segment busts caches | Meaningless here; drop it. Cache lifetime is your `Cache-Control` and crabcrop's cache tiers |
| `f_auto` negotiates format per request | Format negotiation is built in (AVIF, JXL, then WebP by `Accept`); no URL option needed |
| Media library, upload API, moderation, and the rest of the DAM | Out of scope: crabcrop does delivery-time transformation only |
| Cache invalidation from the console or API | `POST /purge/...` per URL, `POST /purge-group/...` per preset, or the CLI verbs |

## Resizing and cropping

| Cloudinary | crabcrop | Notes |
| --- | --- | --- |
| `c_fill,w_300,h_200` | `crop:300x200` | Crop-to-fill in both |
| `c_fill,g_auto` | `crop:300x200:smart` | `g_auto` becomes the smart alignment |
| `c_fill,g_face` / `c_thumb,g_face` | `crop:300x200:face` | Face-anchored crop in both; crabcrop's requires `CRABCROP_FACE_MODEL` (see AI features in the README) |
| `c_fit,w_300,h_200` | `fit-in:300x200` | Fit inside the box in both |
| `c_scale,w_300,h_200` | `300x200` | An exact resize in both |
| `c_limit` / `c_imagga_*` | `fit-in:` / `crop:` plus `filters:no_upscale` | The never-upscale half is a filter here |
| `w_300` alone | `300x0` | Proportional in both |
| `x_`, `y_` with `c_crop` (custom rectangle) | `crop-rect:<l>x<t>:<r>x<b>` | Cloudinary takes a corner and a size; crabcrop takes two corners |
| `dpr_2.0` | Send `DPR: 2` or `Sec-CH-DPR: 2` | Device pixel ratio is a request signal, not a URL option |
| `r_20` / `r_max` | `filters:radius(20)` / `filters:mask(circle)` | Rounded corners; `r_max`'s circular crop is the circle mask |
| `bo_4px_solid_rgb:000000` | `filters:border(4,000000)` | Border width and color |

## Effects and output

| Cloudinary | crabcrop | Notes |
| --- | --- | --- |
| `q_80` / `q_auto` | `filters:quality(80)` / `filters:quality(auto)` | |
| `f_webp` | `filters:format(webp)` | Or drop it and let negotiation choose |
| `e_grayscale` | `filters:grayscale` | |
| `e_sepia` | `filters:sepia` | |
| `e_blur:200` | `filters:blur(2)` | Approximate: Cloudinary's 1-2000 strength scale becomes a sigma, roughly value / 100 |
| `e_sharpen:100` | `filters:sharpen(1)` | Approximate on the same basis |
| `e_brightness:20` | `filters:brighten(51)` | Approximate: Cloudinary scales brightness, crabcrop shifts it |
| `e_contrast:20` | `filters:contrast(1.2)` | Percent becomes a factor: `1 + n/100` |
| `e_saturation:-50` | `filters:saturation(0.5)` | Same conversion |
| `l_<image>` overlay with `g_`, `x_`, `y_`, `o_` | `filters:overlay(<url>,<x>,<y>,<alpha>,normal)` | Overlay image, position, and opacity; blend modes are available too |
| `b_rgb:000000` (background) | `filters:fill(000000)`, or `pad:WxH:000000` | Flatten onto a color, or pad onto one |
| `e_trim` | `filters:trim` | Border trimming |

## The differences that bite

**You need an origin.** Cloudinary URLs name stored assets; crabcrop URLs name fetchable sources. During the transition, the Cloudinary delivery URL itself (without transformations) can serve as the origin, so URLs can move before the assets do. Afterwards, point sources at your own storage.

**Public IDs become a mapping you own.** A public ID like `samples/animals/cat` has no meaning to crabcrop. Keep the mapping in your application: public ID in, origin URL out, then sign with an SDK or `crabcrop sign`.

**No upload pipeline.** Ingest, tagging, and moderation stay wherever they are now. A common split is: keep uploads and the library in the DAM, and move delivery (the hot path) to crabcrop in front of the same assets.

**Defaults differ.** Cloudinary's `q_auto` and `f_auto` are conventions you add per URL. In crabcrop, format negotiation is automatic, and the quality default is a server-side property of each format unless a URL sets `filters:quality(...)`.

## Worked examples

```
Cloudinary:
  https://res.cloudinary.com/demo/image/upload/c_fill,g_auto,h_200,w_300/q_auto/f_webp/sample.jpg

crabcrop:
  /<signature>/crop:300x200:smart/filters:quality(auto):format(webp)/https://images.example.com/sample.jpg
```

```
Cloudinary:
  https://res.cloudinary.com/demo/image/upload/c_thumb,g_face,h_150,w_150/r_max/sample.jpg

crabcrop:
  /<signature>/crop:150x150:face/filters:mask(circle)/https://images.example.com/sample.jpg
```
