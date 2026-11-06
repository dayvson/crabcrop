# Responses and errors

## Responses

A successful image response carries:

| Header | Value |
| --- | --- |
| `Content-Type` | The output format's media type, e.g. `image/webp`. |
| `Cache-Control` | `public, max-age=31536000, immutable` |
| `ETag` | A strong validator: a digest of the response's exact bytes (see [Caching](#caching)). |
| `Vary` | `Accept, DPR, Sec-CH-DPR, Sec-CH-Width, Sec-CH-Viewport-Width, Width, Viewport-Width, Save-Data` |
| `X-Crabcrop-Quality` | Present only when crabcrop chose the quality (`quality(auto)`, a `max_bytes` reduction, or a `Save-Data` step); the value used. |
| `X-Crabcrop-Fallback` | `1` when the response is the configured fallback image rather than the source (see [Fallback image](#fallback-image)). |

### Caching

Successful responses are cached and served without re-fetching or re-processing. Entries are keyed by the request's canonical form (the parsed operations in their one spelling), so equivalent spellings of the same request share one entry. A key names a transformation of a source locator, not the source's contents: after a purge, a changed source re-renders under the same URL, so a source that must stay fixed under its URL should be versioned or immutable itself. The response `ETag` digests the rendered bytes, so a re-render whose bytes differ carries a different ETag and a revalidation is never answered `304` for a render the client does not hold. The first tier is an in-memory LRU bounded by bytes: 256 MB by default, configurable with `CRABCROP_CACHE_MAX_MB` (`0` disables it). With `CRABCROP_RESULT_STORAGE_DIR` set, a second, disk-backed tier (bounded by `CRABCROP_RESULT_STORAGE_MAX_MB`, default 512 MB) sits underneath: results survive restarts, and disk hits are promoted into memory. With `CRABCROP_RESULT_BUCKET` set, a third tier sits behind the local ones: one S3 bucket (or an S3-compatible endpoint, via `CRABCROP_S3_ENDPOINT` and `CRABCROP_S3_REGION`, signed with the ambient AWS credentials) shared by every replica. A shared hit promotes into disk and memory; new results are written through to the bucket in the background; purges fan out to all three tiers of the receiving process. `CRABCROP_RESULT_PREFIX` namespaces the bucket's objects. Shared entries are stored in the disk tier's format, and the tier is best-effort by design: a bucket error reads as a miss, never a failed request. Only `200` responses are cached; errors never are. Because the response can depend on the request headers, negotiated responses are cached per negotiated format class, per effective DPR value, per quantized width-hint class, and per Save-Data class.

Concurrent requests for the same uncached URL share one fetch and one encode: the first requester does the work while the rest wait, then serve the stored entry. Cache keys also carry a fingerprint of the configuration that shapes output (the metadata policy, the quality tuning knobs, the width-hint step, the AI behavior thresholds, and the model, font, encoder, and detector files in use, each identified by its contents rather than its path), printed at startup as `configuration fingerprint: ...`. Restarting with any of those changed starts a fresh cache epoch: the persistent tiers keep the old entries under their old keys, unreachable, until they age out, instead of serving renders produced under the previous configuration. Presets and default operations need no fingerprint: both already change the canonical form every key is built from.

Entries in the persistent tiers are authenticated: each stored entry carries an HMAC-SHA256 tag under a key derived from the server secret, verified before the entry is served or indexed at startup. A file or object planted in the cache directory or bucket by anyone without the secret, or one byte of a stored entry altered on disk, reads as a miss and is re-rendered, never served. Rotating the secret therefore also invalidates the persistent tiers' contents: entries written under the old secret stop verifying and are replaced on demand. (The in-memory tier is process-local and needs no tag.)

### Color management and metadata

Pixels are decoded in the source profile's space. When the source embeds an ICC profile (JPEG and PNG sources are read), crabcrop converts to sRGB with an embedded-profile color transform before any operation runs, so sizes, crops, and filters all work in one known space; a source with no profile is assumed to be sRGB already. `filters:keep_profile` opts out per request: no conversion, and the source profile is embedded into JPEG and PNG outputs instead. CMYK JPEGs decode through the JPEG decoder's own CMYK conversion (embedded CMYK profiles are not honored yet); grayscale profiles are not transformed.

Metadata follows an operator policy, `CRABCROP_METADATA`:

| Value | Behavior |
| --- | --- |
| `strip` (default) | Outputs carry no source metadata. |
| `preserve` | JPEG outputs gain the source's EXIF, XMP, and Photoshop (IPTC) segments; PNG outputs gain EXIF and XMP. Other output formats do not carry preserved metadata. |

Preservation copies the source segments rather than re-serializing them, with one patch: the EXIF orientation is rewritten to 1, because the pixels were already rotated at decode and the original value would double-rotate in viewers. EXIF orientation itself is always honored at decode, under either policy.

### Fallback image

With `CRABCROP_FALLBACK_IMAGE` set (a URL, or a path to a local file read at startup), a source that cannot be fetched or decoded degrades to the fallback instead of an error: the fallback runs through the same operations and is served with `200` and the header `X-Crabcrop-Fallback: 1`. Signature and parse failures never fall back, a `403` or `400` stays honest, and neither does a source-policy refusal. Fallback responses are never cached, so a recovered origin is picked up on the next request; they are also served with `Cache-Control: no-store`, so a CDN or browser never pins the placeholder in place of the real image. With no fallback configured, error behavior is unchanged.

## Errors

Error bodies are plain text with a short message. Send `Accept: application/json` and a grammar failure (a `400` below) instead answers with the v2 JSON envelope, `{"error":{"code":"invalid_filters","message":"invalid filters segment: filters:bogus(1)"}}`. The codes are the parser's stable names, frozen as published: `empty_path`, `missing_signature`, `malformed_signature`, `invalid_size`, `invalid_crop`, `invalid_crop_rect`, `invalid_pad`, `invalid_expiry`, `unknown_preset`, `invalid_preset`, `unknown_root`, `invalid_root`, `presets_only`, `invalid_filters`, `missing_source_url`, `invalid_focal_rect`, `invalid_focal_point`, `invalid_maxage`, `invalid_debug`, `invalid_curve`, `invalid_convolution`, `invalid_noise`, `invalid_frame`, `invalid_page`. The statuses you can get:

| Status | Meaning |
| --- | --- |
| `400` | The path or operations do not parse, or the operation is not supported: a malformed segment, an unknown filter or parameter, `removebg` into a format without alpha, `upscale` past the 4096 px output cap or the 262,144 px per-run input budget, a `crop-rect` or `remove_region` outside the image, a video format asked of a still source. |
| `401` | `GET /metrics` with a missing or wrong `CRABCROP_METRICS_TOKEN` bearer. |
| `403` | Bad signature, an expired `exp:` URL, or the source URL was refused by the source policy (host not in `CRABCROP_ALLOWED_SOURCES`, or resolving to a non-public address refused under `CRABCROP_BLOCK_PRIVATE_SOURCES`). |
| `404` | The source was not found: a file missing under a source root, a source path that resolves outside every root, or an origin (an object store included) that answered 404. All three answer with the same `source not found` body, so the response does not reveal what exists. Also a `PUT` to `/upload/...` when uploads are not configured, or a `PUT` to any path that is not the upload route. |
| `409` | An upload to a name that already exists, while `CRABCROP_UPLOAD_ALLOW_OVERWRITE` is off. |
| `413` | An upload body larger than the `CRABCROP_UPLOAD_MAX_MB` cap. |
| `415` | An upload body that does not decode as an image, or whose media kind does not match the name's extension. |
| `500` | Encoding the output failed, or an AI model failed at inference time (the runtime errored, or a model session lock was poisoned by an earlier failure). A model that cannot be loaded at all fails startup instead of answering requests. |
| `501` | The URL is valid but this server was not configured for it: an AI filter with no model, a text overlay with no font, a video format with no encoder configured. |
| `502` | The source could not be fetched (network error, a non-2xx answer from the origin other than a 404, or a non-image `Content-Type`), the source is larger than the 10 MiB fetch limit, or the fetched bytes could not be decoded as an image, including AVIF sources in builds without AVIF decoding, corrupt HEIC sources, and undecodable formats. A watermark that fails to load also lands here, and so does a configured AI model that ran but produced output the server cannot use (no outputs, an unreadable tensor, an unexpected shape). A configured video encoder that fails, times out, or exceeds its output cap also lands here. |
| `503` | The server shed the request because a processing, slow-lane, AI, video, or animation gate waited past its timeout; the response carries `Retry-After`. |
| `504` | The source fetch timed out (10 seconds per request), or processing ran past the `CRABCROP_PROCESS_TIMEOUT_SECS` limit (30 seconds by default). |

For example, a `removebg` request to a server with no background-removal model returns `501` with the body `removebg is not configured on this server`.
