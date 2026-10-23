# Migrating from imgproxy

imgproxy and crabcrop sign and shape their URLs differently enough that there is no mechanical converter (the `convert-thumbor` verb covers Thumbor). Most imgproxy processing options have a direct crabcrop equivalent, though, so templates translate by hand using the tables below.

## Concepts

| imgproxy | crabcrop |
| --- | --- |
| `/<signature>/<options>/<source>` | `/<signature>/<operations>/<source>` |
| Signature: URL-safe base64 HMAC-SHA256, key and salt as two hex strings (`IMGPROXY_KEY`, `IMGPROXY_SALT`) | Hex HMAC-SHA256 with one secret (`CRABCROP_SECRET`) |
| Source usually base64url-encoded; `plain/` for a literal URL | The source URL spelled literally, always |
| `insecure` URLs | `unsafe` URLs, gated by `CRABCROP_ALLOW_UNSAFE` |
| Options are `name:argument` segments (`rs:fit:300:200`, `q:80`) | Operations are segments (`fit-in:300x200`, `filters:quality(80)`) |
| No built-in result cache; caching is the CDN's job | Two built-in cache tiers (memory, then disk), `Cache-Status` on every image response, `Cache-Group` for presets |
| No purge endpoint | `POST /purge/...` per URL, `POST /purge-group/...` per preset, or the CLI verbs |
| Watermark image configured on the server | Per-URL `filters:watermark(...)`, or the `overlay` filter with blend modes |
| Info processing options | The image URL fetched with `Accept: application/json`, or `crabcrop meta` |

## Resizing

| imgproxy | crabcrop | Notes |
| --- | --- | --- |
| `rs:fit:W:H` (or `rt:fit` with `w:` / `h:`) | `fit-in:WxH` | Fit inside the box in both |
| `rs:fill:W:H` | `crop:WxH` | Crop-to-fill in both |
| `rs:force:W:H` | `WxH` | An exact resize in both (aspect is not kept) |
| `rs:fill-down` / `rs:fit-down` | The same, plus `filters:no_upscale` | The `-down` variants never upscale |
| `w:300` alone | `300x0` | One axis set, the other proportional, in both |
| `g:sm` | `:smart` on the crop | Smart gravity becomes the crop alignment |
| `g:no`, `g:ea`, ... | `:top`, `:right`, ... | Compass gravities collapse to crabcrop's single-axis alignment; corners are approximated |
| `g:fp:0.5:0.25` | No equivalent | Focal-point coordinates do not convert; `smart` and `face` alignments are the alternatives |
| `dpr:2` | Send `DPR: 2` or `Sec-CH-DPR: 2` | crabcrop treats device pixel ratio as a request signal, not a URL option |
| `el:1` (enlarge off) | `filters:no_upscale` | Same switch, opposite spelling |
| `trim:...` | `filters:trim(...)` | Border trimming in both |
| `pd:...` (padding) with `bg:` | `pad:WxH:<color>` | Padding onto a background color |

## Filters and output

| imgproxy | crabcrop | Notes |
| --- | --- | --- |
| `q:80` | `filters:quality(80)` | |
| `f:webp` (or the source extension) | `filters:format(webp)` | Without it, crabcrop negotiates AVIF, JXL, then WebP from the `Accept` header |
| `mb:200000` | `filters:max_bytes(200000)` | Byte budget in both |
| `bl:2` | `filters:blur(2)` | A Gaussian sigma in both |
| `sh:1` | `filters:sharpen(1)` | |
| `sa:1.2` | `filters:saturation(1.2)` | A factor in both |
| `co:1.2` | `filters:contrast(1.2)` | A factor in both |
| `br:20` | `filters:brighten(51)` | Approximate: imgproxy scales brightness, crabcrop shifts it (the amount is rescaled accordingly) |
| `bg:ff0000` | `filters:fill(ff0000)` | Flatten onto a color |
| `wm:...` | `filters:watermark(...)` / `filters:overlay(...)` | imgproxy's watermark is one server-configured image; crabcrop takes the image per URL |
| `ex:1` (extend) with gravity | `pad:` with an alignment story | Padding behavior maps onto the pad operation |

## The differences that bite

**Two-part keys vs one secret.** imgproxy signs with an HMAC keyed by a key and salt pair; crabcrop has a single secret. There is nothing to convert: pick a `CRABCROP_SECRET` and re-sign your templates with the SDKs or `crabcrop sign`.

**Encoded sources.** imgproxy's default source form is base64url, which hides the origin URL. crabcrop sources are literal URLs. Decoded templates are a one-time rewrite; the upside is that crabcrop URLs stay readable in logs, and `crabcrop explain` can verify them segment by segment.

**Options are order-free in imgproxy, ordered in crabcrop.** imgproxy collects its options and applies them in its own pipeline order. crabcrop applies operations in URL order, so put geometry first and `filters:` last, matching what imgproxy would have done.

**Caching moves in-house.** An imgproxy deployment usually relies on the CDN for result caching. crabcrop caches results itself (memory, then disk) and reports `Cache-Status` per response; keep the CDN if you have one, but purge through crabcrop (`purge`, `purge-group`) when an origin image changes under a stable URL.

## Worked example

```
imgproxy:
  /insecure/rs:fill:300:200/g:sm/q:80/f:webp/plain/https://images.example.com/photo.jpg

crabcrop:
  /<signature>/crop:300x200:smart/filters:quality(80):format(webp)/https://images.example.com/photo.jpg
```

And one with fit and a byte budget:

```
imgproxy:
  /insecure/rs:fit:800:600/mb:150000/plain/https://images.example.com/photo.jpg

crabcrop:
  /<signature>/fit-in:800x600/filters:max_bytes(150000)/https://images.example.com/photo.jpg
```
