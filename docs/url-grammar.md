# URL grammar and signing

## How requests work

Every image request has three parts:

```text
GET /<signature>/<operations>/<source-url>
```

| Part | Description |
| --- | --- |
| `signature` | An HMAC-SHA256 (hex) of everything after it, or the literal `unsafe`. |
| `operations` | Zero or more operation segments, applied in path order. |
| `source-url` | The source locator: a full URL with its scheme, a bucket or file locator, or a `root:<name>/<path>` short source naming a [source root](#source-roots). It may contain slashes; the first segment that is not an operation starts it. |

```text
/unsafe/crop:1200x630:smart/filters:grayscale:quality(80)/https://example.com/photo.jpg
```

A request runs at most 100 operations. Each segment counts once, except `filters:`, where each filter counts once: the example above is three operations. The count is taken on what the server will run, after `preset:` references expand and with any server-wide default operations (`CRABCROP_DEFAULT_OPS`) merged in; `exp:` and `maxage:` are not operations and do not count. A request over the cap answers `400` with the code `unsupported_operation` and a body that names the count, such as `request has 101 operations, over the 100 operation limit`. The check runs right after parsing, on the image and `/explain` routes alike, so an over-cap URL never fetches its source. A URL whose own operations already pass the cap is refused before its presets expand: each `preset:` reference counts as the operations its definition holds, settled when the presets load, and the body then names the URL's own count, without the defaults. Preset references may expand to at most 256 KiB of text; a request past that answers the same `400`, with a body such as `request expands to 300000 bytes, over the 262144 byte limit`. Real chains sit far below the cap; it exists so a URL that repeats cheap operations, or repeats a preset, cannot buy unbounded processing time.

### Sign a request

Compute HMAC-SHA256 over the remainder of the path, the operations and the source URL exactly as they appear, with no leading slash, keyed with your `CRABCROP_SECRET`, and hex-encode the result:

```bash
printf '%s' '300x200/https://example.com/photo.jpg' | openssl dgst -sha256 -hmac 'your-secret'
# SHA2-256(stdin)= 3c1f6c180116661877bd4c85fd201a178087d74c1cc188cc584e819d901308c6
```

Use that hex string as the first path segment. Anyone can verify the scheme against this test vector: with secret `Jefe`, the HMAC-SHA256 of `what do ye want for nothing?` is `ba5396e10039ec002bf1fde117eee48df8c2f06bc47b5f2d3272ea77f6862da9`.

A request with a bad signature gets `403` and the body `bad signature`. The literal signature `unsafe` skips verification, but only when the server runs with `CRABCROP_ALLOW_UNSAFE=1` (or `true`, `yes`, `on`). Keep it off in production. Purges are the exception that never bends: `/purge` and `/purge-group` always require an HMAC signature made with the secret, even in unsafe mode.

### Sealed URLs

Signing proves a URL was minted by someone holding the secret, but it hides nothing: the operations and the source sit in the path for anyone to read. That readability is the design (`The URL is the API` cuts both ways, and it is what makes crabcrop URLs debuggable), and for most deployments it is the right default. When it is not, the same remainder can be **sealed**: encrypted into an opaque token that sits in the signature slot.

```text
https://images.example.com/s1.AAECAwQFBgcICQoL4rtOMQ/.../photo.jpg
```

The token is `s1.` plus base64url(nonce, ciphertext, tag): AES-256-GCM over the whole remainder, under a key derived from the same secret, so there is no second secret to manage. The text after the token (`photo.jpg` above) is a friendly name: purely cosmetic, never parsed or authenticated, present so sealed URLs still read as images in logs, analytics, and page source. A sealed URL serves exactly what its plain spelling serves: the server decrypts the remainder and runs the same pipeline, so both forms share one cache entry, one ETag, and one purge target. Mint one with the builders (`build_sealed_url` in Rust, `buildSealedURL` in TypeScript, `build_sealed_url` in Python, where sealing alone needs the `crabcrop-url[seal]` extra) or the CLI:

```bash
crabcrop seal 'filters:format(webp):quality(80)/https://media.example.com/heroes/dog.jpg' dog.jpg
```

Know what sealing does and does not do:

- **It hides the recipe from page readers.** Anyone who can see the URL (a page visitor, a competitor scraping your pages) learns nothing about the operations or the origin from the URL itself. Nor does the response name its presets: a sealed URL's response carries no `Cache-Group` header (see [Cache headers and groups](endpoints.md#cache-headers-and-groups)). Anyone who can observe the request itself (your CDN, a network intermediary) sees the token in the request line regardless; sealing is not transport privacy, and TLS is the answer there.
- **It is not access control.** A sealed URL is a bearer artifact anyone can replay, exactly like a signed one. Authorization is still the source allowlist, expiry (`exp:` works inside a sealed payload, unchanged), and whatever fronts the deployment.
- **It trades away debuggability at the server.** `/explain` refuses sealed URLs: a server endpoint that decrypts would be an oracle for anyone who lifts a token off a public page. To debug one, unseal it locally (`crabcrop unseal <url>`, under your own secret) and explain the plain form. The meta document for a sealed request likewise reports the render, never the recipe. Sealed URLs are for links that must be public and unreadable at the same time: most links are not, and the readable form stays the default and the recommendation.

Failures are uniform: a malformed token, a token minted under another secret, a token whose payload does not parse, all answer `403` with the body `bad signature`, indistinguishable from each other and from a bad HMAC. (A token that opens but has expired answers `signature expired`, exactly like the plain form.) Rotation works the way it does for signatures, inherited rather than reinvented: a token may carry a key ID (`k2:s1.…`), the server opens keyed tokens against the named key only, and during a rotation window an unkeyed token opens under the previous secret until that secret is retired. Minting refuses a remainder longer than 2,048 bytes; a recipe that long belongs in a preset. The full construction is in `spec/url-signing.md`.

### Expiring URLs

Add an `exp:<unix-seconds>` segment anywhere among the operations to make a signed URL expire:

```text
/<signature>/exp:1893456000/300x200/https://example.com/photo.jpg
```

The segment is inside the signed remainder, so the HMAC covers it: extending the expiry without the secret breaks the signature. Past the expiry the request gets `403` with the body `signature expired`, and expired URLs never serve from the cache. The comparison is exact: the URL is valid strictly before its `exp` second, with no built-in allowance for clock skew. Keep the server's clock synchronized (NTP), and mint expiries with whatever margin your own clock situation needs. URLs without `exp` never expire, exactly as before.

### Per-request cache lifetime

Image responses are cached by clients and CDNs for a year: `Cache-Control: public, max-age=31536000, immutable`, because a signed URL's bytes never change. When one response should live shorter, add a `maxage:<seconds>` segment anywhere among the operations:

```text
/<signature>/maxage:60/300x200/https://example.com/photo.jpg
```

The value runs from 0 to 31536000. The response to that request carries `Cache-Control: public, max-age=60, immutable` instead of the year; every other response is unchanged. `maxage:0` answers `Cache-Control: public, max-age=0` with no `immutable`: the response is stale on arrival, so caches must revalidate before reusing it.

The directive changes the response header only. It never enters the cache key, so URLs that differ only in `maxage` share one stored image, and purging is unaffected. The shared result cache (S3) stores bodies, not headers, so the per-response header composes there too: each request's lifetime is applied as its response is built, wherever the body came from.

Like `exp`, the segment sits inside the signed remainder and repeated segments resolve to the last one. A malformed or out-of-range value is a `400`, never a silent fallback to the default. Presets stay immutable by default; putting `maxage` on a preset URL is the caller's explicit choice for that response, and preset definitions cannot contain it (it is per-request metadata, like `exp`).

### Revalidation

Every cacheable image response also carries a strong `ETag`: a digest of the response's exact bytes, computed when the render is produced and stored with its cache entry. If a re-render under the same URL produces different bytes (the source changed and the old entry was purged), the ETag changes with them. A client or CDN holding a copy can revalidate with `If-None-Match`: when the value matches (an exact match, a weak `W/` spelling, one entry of a comma list, or `*`), the answer is `304 Not Modified` with no body, on cache hits and fresh renders alike; anything else gets the full `200`. The validator applies only to public `200` image responses: meta responses (`private, no-store`), fallbacks, and errors carry no `ETag` and are never answered with a `304`. It pairs with `maxage:0` for URLs whose copies should be checked before every reuse.

### Output size

A size written in the URL is at most 16383 px per axis, the strictest encoder limit in the pipeline (WebP's): `16384x100` does not parse, and fails with `400` (`invalid_size`). The same limit holds for every size the server works out from the request rather than reads from it: a proportional or `orig` side, a `fit-in` scale, a `crop` or `pad` target, any of these scaled by `dpr(...)` or a DPR client hint, and the larger canvas a `rotate(...)` filter draws on. Each step's output is also held to 100 megapixels in total. A step whose size passes either bound fails the whole request with `400` (`unsupported_operation`) before the step allocates its image, wherever the step sits in the chain and whatever follows it; the size is never clamped. With `no_upscale`, the cap shrinks a resize, `fit-in`, or `crop` first, and the limit applies to what is left; a `pad` canvas is never shrunk, so a pad past the limit is refused with or without it.

A factor that comes only from a DPR client hint (`DPR` or `Sec-CH-DPR`) is the exception, because a hint is advisory: when the hint alone pushes a size past either bound, the server steps the factor down to the largest of 1, 1.5, 2, and 3 whose sizes fit both, and answers `200` at that factor instead of `400`. `/unsafe/6000x1/<source>` with `DPR: 3` asks for 18000x3 and is served at factor 2, as 12000x2. A stepped factor past 1 also holds the output itself to 16383 px per axis, so it never serves a 16384 px source's side as decoded: `/unsafe/8192x1/<16384x2 source>` with `DPR: 3` asks for the source's own size at factor 2 and is served at 1.5, as 12288x2. Only when even factor 1 does not fit is the URL itself over, and it answers its `400` under any hint. A factor written as `dpr(...)`, in the URL, a preset, or the server's default operations, is never stepped down: `/unsafe/6000x1/filters:dpr(3)/<source>` answers `400` with or without the hint (see [Client hints](filters.md#client-hints)).

```text
/unsafe/16383x1/0x2/https://example.com/photo.jpg   400: the second step keeps the 16383:1 aspect ratio at height 2, so it asks for 32766x2
/unsafe/5461x1/0x3/https://example.com/photo.jpg    200: 16383x3, exactly the limit
```

The refusal names the size it refused: `requested output 32766x2 exceeds the 16383 pixel limit per axis`.

A source can decode at up to 16384 px per axis (see below), one past the limit. A step that ends at the size it started from works nothing out, so the per-axis limit does not refuse the step itself: `0x0`, `fit-in:0x0`, `rotate(180)`, and a resize that `no_upscale` or a width client hint holds at the source pass such a source on as decoded, and a later step can still bring it within the limit. A step that changes the size is held to the limit at the step, a quarter turn included: `origx1` on a 16384x2 source asks for 16384x1, and `rotate(90)` for 2x16384, so both are refused. An SVG source is rasterized before the chain runs, at the size its first resize, `fit-in`, or `crop` asks for and at most 4096 px per axis, and the chain's sizes resolve against that raster. A PDF page renders within 16383 px per axis, unless its media box is larger than its crop box (see [PDF pages](media.md#pdf-pages)).

The image the server encodes is held to the limit too, whatever the chain did. A chain that leaves a side past 16383 px, such as a 16384 px source with no sizing step or no operations at all, fails with `400` (`unsupported_operation`) in every output format, including a WebP negotiated from `Accept`, and the `/explain` forecast reports it as `over_dimension: true`:

```text
/unsafe/https://example.com/16384x1.png          400: requested output 16384x1 exceeds the 16383 pixel limit per axis; resize or crop it to fit
/unsafe/8192x0/https://example.com/16384x1.png   200: 8192x1
```

Video output needs even sides (H.264 and VP9 subsample color in 2x2 blocks). An odd side pads up by one pixel, except a side at the limit: 16383 px crops its last row or column to 16382 instead of padding past the limit (see [Video output](media.md#video-output)).

The [`/explain`](endpoints.md#self-explaining-urls) forecast applies the same rule to resize, `fit-in`, `crop`, `pad`, and `rotate(...)` steps, and reports `over_dimension: true` for such a chain. It works from the source's stored size, the URL, and the DPR hint (stepped down as the image request steps it) alone, so it does not see an EXIF orientation turn the source, it reports no dimensions from a `trim` on (the trimmed size depends on the pixels), it does not count an animated source's frames (so it cannot see the budget they share, which the image request may step a hinted factor down for), and it leaves out the width and Save-Data hints.

### Source images

The loader fetches the source under bounded limits, for every source kind:

- 10 seconds per source fetch in total by default (`CRABCROP_SOURCE_TIMEOUT_SECS`): one budget spans the whole redirect chain and the body read, so a chain of fast hops cannot stretch a fetch past it.
- 10 MiB maximum body by default (`CRABCROP_SOURCE_MAX_MB`); larger sources fail with `502` (the request was fine; the origin object is too big).
- If an HTTP response carries a `Content-Type` header, it must start with `image/`. The exceptions are the media selections: when the URL carries a `frame:` segment, a `video/*` Content-Type is accepted too (see [Video frames](media.md#video-frames)), and when it carries a `page:` segment, `application/pdf` is accepted (see [PDF pages](media.md#pdf-pages)). Both admissions belong to the request, never to the response alone.

Decoding is bounded too, so a small file cannot expand into gigabytes: at most 16384 px per axis and 256 MiB of decoder allocations, and animated GIF and WebP sources at most 200 frames, 100 million decoded pixels, and 256 MiB of decoded frames in total. Sources past the bounds fail with `502`. An AVIF source's container is checked before it is parsed, since the parser sets memory aside for what the file declares: a box declaring more bytes than the file (or the box around it) holds, or an item count its box has no room for, fails with `502` before anything is set aside.

HTTP and HTTPS sources additionally allow up to 5 redirects, with the [source policy](configuration.md#configuration) re-checked on every hop.

When `CRABCROP_ALLOWED_SOURCES` is set, only the listed host suffixes may be fetched (a suffix matches the host itself or any subdomain of it); unset, every host may be fetched, subject to the private-address refusal below. Source URLs with userinfo (`http://user:pass@host/...`) are always refused: the loader never presents Basic credentials to an origin.

`CRABCROP_BLOCK_PRIVATE_SOURCES` is on by default: the loader refuses sources whose host is, or resolves to, a non-public address: loopback, private, link-local, unspecified, shared (CGNAT), reserved, multicast, broadcast, documentation, benchmarking, and the other special-purpose ranges, including IPv4-mapped, IPv4-compatible, and 6to4 spellings of them. A host on the `CRABCROP_ALLOWED_SOURCES` allowlist is exempt, which is how a deployment serves loopback or internal sources. For every other host, the host is resolved and its addresses checked before connecting and again on every redirect hop, and the connection is pinned to the validated address, so the address that was checked is the address that gets connected (no DNS-rebinding gap). Set `CRABCROP_BLOCK_PRIVATE_SOURCES=0`, or `CRABCROP_ALLOW_ANY_SOURCE=1`, to switch the refusal off. A policy refusal answers `403`.

Beyond a plain HTTP or HTTPS URL, a source can take four more forms:

- `s3://bucket/key`: fetched from AWS S3, or from an S3-compatible store (MinIO, R2, Spaces) when `CRABCROP_S3_ENDPOINT` is set. Requests are signed with SigV4 when AWS credentials are in the environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`); with none, the fetch is unsigned and public buckets just work.
- `gs://bucket/key`: the same, against Google Cloud Storage's S3-compatible XML API, signed with the HMAC pair `CRABCROP_GCS_ACCESS_KEY` / `CRABCROP_GCS_SECRET_KEY` when set.
- `file:///path`: a local file under `CRABCROP_FILE_ROOT`. The path is canonicalized per request and must stay under the root, so `..` segments and symlink escapes are refused. Unset root disables file sources.
- `root:<name>/<path>`: a short spelling of an HTTP or HTTPS source, instead of the full URL. The server expands it against the origin configured for that name in `CRABCROP_SOURCE_ROOTS_FILE` (see [Source roots](#source-roots)): the signature covers the short form, expansion runs after verification and before cache keying so both spellings are one render, and every loader check in this section applies to the expanded URL.

A source in none of these forms is a `400` with the code `invalid_source`: text that does not parse as a URL, another scheme (`ftp://`, `data:`), or a malformed locator (`s3://` with no bucket or key, a file path carrying a NUL). The fault is the URL's own, so it never reaches the network and never falls back. A redirect to such a URL is different: that is the origin's answer, and it fails as a source fetch (`502`).

Bucket sources are gated by `CRABCROP_ALLOWED_BUCKETS` rather than the host policy, since the endpoint is operator configuration, not request input. The allowlist is required with or without signing credentials: an unset list refuses every bucket fetch, and when credentials are configured the server also refuses to boot without it, because signed fetches would otherwise reach every bucket the credentials can read. List `*` to deliberately allow all buckets. A bucket or object key carrying a dot segment (a `.` or `..` path component) is refused, since the request path would collapse it past the allowlisted bucket; dots inside a segment name are fine. Bucket names must also follow the S3 naming rules (3 to 63 characters of lowercase letters, digits, dots, and hyphens, starting and ending with a letter or digit, no adjacent dots, no dot beside a hyphen, not an IP address spelling): the name is interpolated into the request host, so a name carrying a `?` or another host-breaking character is refused at parse time, whatever the allowlist says. Store responses are not redirected: a 3xx from a store surfaces as its status code.

JPEG, PNG, WebP, GIF, JPEG XL, and TIFF sources are decoded. TIFF decoding is the pure-Rust `tiff` decoder under the same decode bounds as every other format; a multipage TIFF renders its first page. A floating-point TIFF has its samples clamped to the 0 to 1 range and then renders like a 16-bit source: PNG output keeps 16 bits per channel unless a step works in 8 bits (an aspect crop, a pad, and some filters do), and a `palette` PNG and every other format encode 8. EXIF orientation is applied before any operation runs, so phone photos come out upright. Animated GIFs keep their frames when the output format is GIF or WebP; converting an animated GIF to a still format uses the first frame. Animated WebPs keep their frames when the output format is WebP or GIF, with frame delays and the loop count preserved; converting one to a still format uses the first frame. Animated WebP output is encoded through libwebp with the request quality, the same mapping as still WebP: the default is 80, a lower quality is a smaller lossy encode, and `quality(100)` encodes losslessly. GIF output has no quality setting. Every render that encodes animated WebP output runs behind a dedicated gate of one encode at a time: an animated WebP kept as WebP, and an animated GIF converted to WebP, whether `format(webp)` names it or `Accept` negotiates it. While one is running, another such request waits up to 10 seconds for the slot and then sheds with a `503` rather than queueing behind a long encode, and still renders keep their own processing pool throughout. AVIF sources are decoded in builds that include AVIF decoding: the Docker image and the release binaries. A source build includes it with `--features avif-decode`, which links dav1d (a system libdav1d 1.3 or newer, or a vendored build); without it, AVIF sources fail with `502` and an error naming the feature. Animated AVIF sources decode their first frame. HEIC/HEIF sources are decoded in every build by the pure-Rust oxideav stack (`oxideav-heif` + `oxideav-h265`): the primary image only, with grid and overlay compositions resolved, EXIF orientation honored, and 10-bit sources flattened to the pipeline's 8-bit working space, as AVIF sources are.

BMP and ICO sources are decoded too, under the same bounds; an ICO renders one of the images it carries: the one with the deepest color, the largest if several tie. No other raster format is decoded: OpenEXR, Radiance HDR, QOI, PNM, DDS, TGA, and farbfeld sources fail with `502`, and as uploads with `415`. A HEIC/HEIF source decodes only when every image it is built from is HEVC-coded: a HEIF file that would need another codec (an AV1, H.264, or H.266 item, say, behind a generic `mif1` brand) fails with `502` before any decoding starts. An AVIF file, which names itself with the `avif` brand, is an AVIF source, not a HEIF one.

SVG sources are supported and detected by content, not by the declared content type. An SVG is sanitized before it is rendered: script elements, event handlers, and references to external resources are stripped, so a source SVG can only draw what it carries with it. What it carries is bounded before anything renders: the document may draw at most 100,000 nodes once `<use>` references, patterns, masks, clip paths, markers, and filters (including an `feImage` that draws another element) are expanded, and every raster image it embeds counts against the 256 MiB decode budget at its own decoded size, once for each time it is drawn, wherever it sits (a plain `<image>`, a pattern, a mask, a filter's `feImage`, or an SVG image nested inside it). Both bounds are checked from the document's structure before the renderer builds anything, so a few kilobytes of references that would expand exponentially are refused in milliseconds. That check counts conservatively: every element and every character of text once for each time it is drawn, and every embedded payload's bytes once for each copy. References may nest at most 128 levels deep, counting element nesting and every reference followed, and a cycle of references is refused. SVG images nested in the document are held to the same bounds, and share them with it; a gzip-compressed (svgz) nested image is not drawn. A document past any of these bounds fails with `502`. The sanitized document is rasterized at the size the URL asks for (or its intrinsic size when the URL asks for none) and then runs through the normal pipeline. Output defaults to PNG for SVG sources.

An SVG is parsed and rendered in a separate worker process, never in the server itself: for each SVG the server starts its own binary as a worker, which caps its own CPU time and, on Linux, its address space (4 GiB) as it starts, parses and renders on a bounded stack, and is killed at the request's processing deadline (`CRABCROP_PROCESS_TIMEOUT_SECS`). So a document that gets past the bounds and still crashes or exhausts the renderer, such as a 130-byte path whose arc radii flatten into billions of segments, fails its own request with `502`, and one still rendering at the deadline fails with `504`, while the server and every other request carry on. Upload validation and the explain report's size probe parse SVG in a worker the same way (the probe waits on a blocking thread, never on the server's request threads, and gives up after 5 seconds and reports no size). The worker costs one process start per SVG: measured on a debug build on a 4-core host, about 9 ms on a small icon (2.5 ms in-process before, 12 ms now), about 40 ms more when the document has text, because a fresh worker loads the system fonts for it, and 2 to 6% on a 290 KB illustration rendered at 800x600 (about 4 s either way).

A source that cannot be fetched or decoded can degrade to a configured fallback image instead of an error; see [Fallback image](responses.md#fallback-image). Short `root:` sources expand to their configured origin before any of these checks run, and every check applies to the expanded URL; see [Source roots](#source-roots).

## Source roots

A source root names an origin on the server, so a URL can carry a short source instead of spelling out the full source URL: the origin layout stays off the URL, and URLs get shorter. Define roots in a file, one per line (blank lines and lines starting with `#` are skipped):

```text
# /etc/crabcrop/roots.txt
photos = https://cdn.example.com/photos
media = https://assets.example.com
```

Point `CRABCROP_SOURCE_ROOTS_FILE` at it, and a request can use `root:<name>/<path>` as its source:

```bash
curl "http://localhost:3000/<signature>/300x200/root:photos/cat.jpg"
```

That request fetches `https://cdn.example.com/photos/cat.jpg`. The signature covers the short form exactly as written; the server expands it after verification and before cache keying, so the short spelling and the full URL it expands to are one render: they share a cache entry, and purging either spelling evicts it. Root definitions are validated at startup: names are letters, digits, `_`, or `-`, a duplicate name refuses the boot, and a base must be an absolute http(s) URL with no userinfo, query, or fragment.

The safety rules:

- The source allowlist (`CRABCROP_ALLOWED_SOURCES`) and every other loader check apply to the expanded URL, exactly as if the client had spelled it out. A root never widens what the server will fetch.
- The path must stay under the root. Dot segments are normalized, and a path that climbs above the root, plainly or percent-encoded, is a `400` (`invalid_root`).
- An unknown root name is a `400` (`unknown_root`), including on a server with no roots configured.
- Root base URLs are operator configuration and stay private: `explain` reports the source as written plus the root's name, and never prints a base URL.
