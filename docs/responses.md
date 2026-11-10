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

Successful responses are cached and served without re-fetching or re-processing. Entries are keyed by the request's canonical form (the parsed operations in their one spelling), so equivalent spellings of the same request share one entry. A key names a transformation of a source locator, not the source's contents: after a purge, a changed source re-renders under the same URL, so a source that must stay fixed under its URL should be versioned or immutable itself. The response `ETag` digests the rendered bytes, so a re-render whose bytes differ carries a different ETag and a revalidation is never answered `304` for a render the client does not hold. The first tier is an in-memory LRU bounded by bytes: 256 MB by default, configurable with `CRABCROP_CACHE_MAX_MB` (`0` disables it). An entry counts against the budget with its cache key and metadata as well as its body. A key whose canonical form is longer than 4096 bytes is stored under a SHA-256 digest of that form, followed by the source when it fits, so a very long URL still caches and purges while its key stays short; keys of 4096 bytes or less are the canonical form itself. With `CRABCROP_RESULT_STORAGE_DIR` set, a second, disk-backed tier (bounded by `CRABCROP_RESULT_STORAGE_MAX_MB`, default 512 MB) sits underneath: results survive restarts, and disk hits are promoted into memory. With `CRABCROP_RESULT_BUCKET` set, a third tier sits behind the local ones: one S3 bucket (or an S3-compatible endpoint, via `CRABCROP_S3_ENDPOINT` and `CRABCROP_S3_REGION`, signed with the AWS credentials from the environment) shared by every replica. A shared hit promotes into disk and memory; new results are written through to the bucket in the background; purges fan out to all three tiers of the receiving process. `CRABCROP_RESULT_PREFIX` namespaces the bucket's objects. Shared entries are stored in the disk tier's format, and the tier is best-effort by design: a bucket error reads as a miss, never a failed request. Only `200` responses are cached; errors never are. Because the response can depend on the request headers, negotiated responses are cached per negotiated format class, per effective DPR value, per quantized width-hint class, and per Save-Data class.

Concurrent requests for the same uncached URL share one fetch and one encode: the first requester does the work while the rest wait, then serve the stored entry. When nothing is stored (an error, a `503` shed included, the fallback image, or a render with every cache tier turned off), the waiting requests that would get the same response as the first one (the same path, negotiated format, DPR, quantized width, and Save-Data) receive its response instead, each answering its own `If-None-Match`, so a burst against a failing origin costs one fetch rather than one per request. A request that would get a different response, such as the sealed spelling of the same URL, makes its own attempt, and so does every waiter when a purge of the URL lands while the first request is still working, so a request sent after a purge never receives a render begun before it. Meta requests coalesce the same way, errors included. Cache keys also carry a fingerprint of the configuration that shapes output (the metadata policy, the quality tuning knobs, the width-hint step, the AI behavior thresholds, and the model, font, encoder, and detector files in use, each identified by its contents rather than its path), printed at startup as `configuration fingerprint: ...`. Restarting with any of those changed starts a fresh cache epoch: the persistent tiers keep the old entries under their old keys, unreachable, until they age out, instead of serving renders produced under the previous configuration. The fingerprint also carries a rendering version that a release raises when it changes rendered output for URLs that already exist, so the first start on such a release begins a fresh epoch the same way; the CHANGELOG says when a release does. Presets and default operations need no fingerprint: both already change the canonical form every key is built from.

Entries in the persistent tiers are authenticated: each stored entry carries an HMAC-SHA256 tag under a key derived from the server secret, verified before the entry is served or indexed at startup. A file or object planted in the cache directory or bucket by anyone without the secret, or one byte of a stored entry altered on disk, reads as a miss and is re-rendered, never served. Rotating the secret therefore also invalidates the persistent tiers' contents: entries written under the old secret stop verifying and are replaced on demand. (The in-memory tier is process-local and needs no tag.)

### Color management and metadata

Pixels are decoded in the source profile's space. When the source embeds an ICC profile (JPEG and PNG sources are read), crabcrop converts to sRGB with an embedded-profile color transform before any operation runs, so sizes, crops, and filters all work in one known space; a source with no profile is assumed to be sRGB already. `filters:keep_profile` opts out per request: no conversion, and the source profile is embedded into JPEG and PNG outputs instead. Only a well-formed profile of at most 256 KiB is embedded: its header's size field must match its length, and the `acsp` signature must be in place. Any other profile is converted to sRGB even under `keep_profile`, as for output formats that cannot carry a profile, since pixels left in a space nothing labels would show wrong colors. CMYK JPEGs decode through the JPEG decoder's own CMYK conversion (embedded CMYK profiles are not honored yet); grayscale profiles are not transformed.

Metadata follows an operator policy, `CRABCROP_METADATA`:

| Value | Behavior |
| --- | --- |
| `strip` (default) | Outputs carry no source metadata. |
| `preserve` | JPEG outputs gain the source's EXIF, XMP, and Photoshop (IPTC) segments; PNG outputs gain EXIF and XMP (the XMP packet compressed if the source stored it compressed, plain text otherwise). Other output formats do not carry preserved metadata. |

Preservation copies the source segments rather than re-serializing them, with one patch: the EXIF orientation is rewritten to 1, because the pixels were already rotated at decode and the original value would double-rotate in viewers. EXIF orientation itself is always honored at decode, under either policy.

Reading a source's profile and metadata is bounded, because PNG stores the profile, and can store the XMP packet, compressed. Only the first profile (`iCCP`), EXIF (`eXIf`), and XMP packet of a PNG are read; repeats are skipped unread (libpng likewise honors only the first profile). A profile that inflates past 4 MiB, or an XMP packet past 256 KiB, is dropped rather than truncated, and all of one source's metadata inflates within a single 4.25 MiB budget. Under `strip`, the EXIF, XMP, and Photoshop segments of JPEG and PNG sources are not extracted at all (the EXIF orientation is still read and honored at decode); the profile always is, since it drives the color conversion.

Writing metadata back out is bounded too. A PNG output stores the profile compressed, and the XMP packet compressed whenever the source did, so each weighs at most about what it weighed in the source. A JPEG output stores both uncompressed, so the caps set the ceiling there: the profile adds at most 256 KiB, and an XMP packet that does not fit one 64 KiB segment is left out. EXIF and Photoshop segments are copied at the size they had in the source.

### Fallback image

With `CRABCROP_FALLBACK_IMAGE` set (a URL, or a path to a local file read at startup), a source that cannot be fetched or decoded degrades to the fallback instead of an error: the fallback runs through the same operations and is served with `200` and the header `X-Crabcrop-Fallback: 1`. Signature and parse failures never fall back, a `403` or `400` stays honest, and neither does a source-policy refusal or a source URL the server cannot fetch at all (`invalid_source`). Fallback responses are never cached, so a recovered origin is picked up on the next request; they are also served with `Cache-Control: no-store`, so a CDN or browser never pins the placeholder in place of the real image. With no fallback configured, error behavior is unchanged.

## Errors

Error bodies are plain text with a short message. Send `Accept: application/json` and any failure instead answers with the v2 JSON envelope, `{"error":{"code":"invalid_filters","message":"invalid filters segment: filters:bogus(1)"}}`: the same status and the same message, plus a stable machine-readable `code` that clients can branch on. The codes are frozen as published: a failure class may be renamed in code, but its code never changes. One exception: a sealed request's failures stay the uniform `403` refusal, with no envelope and no code, so a lifted token reveals nothing about the payload it carries; an opened token whose payload has expired answers `signature_expired`, exactly as the plain form does. The same header also asks for meta. With meta mode off (the default), a request that passes the signature, grammar, and expiry checks answers `403 meta_disabled`, so a later failure (fetching or rendering the source) reports its own code in the envelope only on a server with meta mode on.

Grammar failures (a `400` below) carry the parser's stable names: `empty_path`, `missing_signature`, `malformed_signature`, `invalid_size`, `invalid_crop`, `invalid_crop_rect`, `invalid_pad`, `invalid_expiry`, `unknown_preset`, `invalid_preset`, `unknown_root`, `invalid_root`, `presets_only`, `invalid_filters`, `missing_source_url`, `invalid_focal_rect`, `invalid_focal_point`, `invalid_maxage`, `invalid_debug`, `invalid_curve`, `invalid_convolution`, `invalid_noise`, `invalid_frame`, `invalid_page`. Signature-segment failures on the purge, purge-group, and upload routes carry these codes too.

Operational failures carry these codes:

| Code | Status | Failure |
| --- | --- | --- |
| `bad_signature` | `403` | The signature does not verify (image, explain, purge, purge-group, and upload routes). |
| `signature_expired` | `403` | The signed URL's `exp:` second has been reached. |
| `reserved_segment` | `400` | An image URL begins with a purge or upload segment. |
| `invalid_path` | `400` | An explain, purge, purge-group, or upload path does not have the route's shape. |
| `meta_not_available` | `400` | Meta was asked for a video or debug output, which serves no image to describe. |
| `meta_disabled` | `403` | Meta mode is off on this server, which is the default (`CRABCROP_META_MODE` unset or `0`). |
| `not_found` | `404` | A `PUT` to a path that is not the upload route. |
| `metrics_disabled` | `403` | `GET /metrics` when no `CRABCROP_METRICS_TOKEN` is configured. |
| `metrics_unauthorized` | `401` | `GET /metrics` with a missing or wrong bearer token. |
| `invalid_source` | `400` | The source URL is not one the server can fetch: it does not parse, names a scheme other than `http`, `https`, `s3`, `gs`, or `file`, or is a malformed bucket or file locator (`s3://` with no bucket or key, a file path carrying a NUL). A redirect to such a URL is the origin's answer and stays `source_fetch_failed`. |
| `source_forbidden` | `403` | The source policy refused the source URL. |
| `source_not_found` | `404` | The source does not exist. |
| `source_bad_status` | `502` | The origin answered a non-2xx status other than 404. |
| `source_fetch_failed` | `502` | The source fetch failed (a DNS, connect, TLS, or stream error). |
| `source_not_an_image` | `502` | The origin answered with a non-image `Content-Type`. |
| `source_too_large` | `502` | The source is larger than the fetch limit. |
| `source_timeout` | `504` | The source fetch timed out. |
| `decode_failed` | `502` | The fetched bytes could not be decoded as an image, including an SVG whose render worker crashed or ran out of memory, and an AVIF whose container the screen refused or whose decode worker failed. |
| `encode_failed` | `500` | Encoding the output failed. |
| `unsupported_operation` | `400` | The operation is not supported for this source or output. Also a request over the 100-operation cap, counted after presets expand and with the default operations merged in (see [the URL grammar](url-grammar.md#how-requests-work)); that one answers before any source fetch, with the body `request has <n> operations, over the 100 operation limit`. So does a request whose preset references would expand past 256 KiB of text, with the body `request expands to <n> bytes, over the 262144 byte limit`. |
| `not_configured` | `501` | The operation needs a model or resource this server was not configured with. |
| `inference_bad_output` | `502` | A model ran but produced output the server cannot use. |
| `inference_failed` | `500` | An AI model failed at inference time. |
| `render_failed` | `500` | A text overlay could not be rendered. |
| `video_encode_failed` | `502` | The configured video encoder failed. |
| `frame_extract_failed` | `502` | The configured frame extractor failed. |
| `pdf_render_failed` | `502` | The configured PDF renderer failed. |
| `processing_timeout` | `504` | Processing ran past the process deadline. An SVG still rendering, or an AVIF still decoding, is killed with its worker process. |
| `processing_failed` | `500` | The processing job itself failed. |
| `server_busy` | `503` | A processing, slow-lane, AI, video, or animation gate, or the source buffer budget (`CRABCROP_SOURCE_BUFFER_MAX_MB`), waited past its timeout; the response carries `Retry-After`. |
| `provenance_unavailable` | `503` | Signing the output failed because the C2PA timestamp authority (`CRABCROP_C2PA_TSA_URL`) did not answer within its bounds, refused the connection, or answered with something that is not a timestamp. The response carries `Retry-After: 30`. Any other signing failure is `encode_failed`. |
| `invalid_purge_group` | `400` | A purge-group name is empty or contains a slash. |
| `uploads_not_configured` | `404` | A `PUT` to `/upload/...` on a server with no upload destination configured, whatever its signature. |
| `invalid_upload_name` | `400` | The upload name is not valid. |
| `upload_too_large` | `413` | The upload body is larger than the cap. |
| `upload_unreadable` | `400` | The upload body could not be read to its end: the client aborted it, or its chunked framing is malformed. |
| `upload_not_an_image` | `415` | The upload body is empty or does not decode as an image. |
| `upload_media_mismatch` | `415` | The upload name's extension does not match the body's media kind. |
| `upload_already_exists` | `409` | An upload to a name that already exists, while overwriting is off. |
| `validation_failed` | `500` | The upload's image validation itself failed. |
| `upload_store_unavailable` | `502` | The upload store's backend is unavailable. |
| `upload_store_failed` | `500` | The upload could not be stored. |

The statuses you can get:

| Status | Meaning |
| --- | --- |
| `400` | The path or operations do not parse, or the operation is not supported: a malformed segment, an unknown filter or parameter, `removebg` into a format without alpha, `upscale` past the 4096 px output cap or the 262,144 px per-run input budget, a `crop-rect` or `remove_region` outside the image, a video format asked of a still source. Also a source or watermark URL the server cannot fetch at all (it does not parse, or names a scheme the server does not fetch), and an upload body that could not be read to its end. |
| `401` | `GET /metrics` with a missing or wrong `CRABCROP_METRICS_TOKEN` bearer. |
| `403` | Bad signature, an expired `exp:` URL, or the source URL was refused by the source policy (host not in `CRABCROP_ALLOWED_SOURCES`, or resolving to a non-public address refused under `CRABCROP_BLOCK_PRIVATE_SOURCES`). |
| `404` | The source was not found: a file missing under a source root, a source path that resolves outside every root, or an origin (an object store included) that answered 404. All three answer with the same `source not found` body, so the response does not reveal what exists. Also a `PUT` to `/upload/...` when uploads are not configured (`uploads_not_configured`), or a `PUT` to any path that is not the upload route (`not_found`). |
| `409` | An upload to a name that already exists, while `CRABCROP_UPLOAD_ALLOW_OVERWRITE` is off. |
| `413` | An upload body larger than the `CRABCROP_UPLOAD_MAX_MB` cap. |
| `415` | An upload body that does not decode as an image, or whose media kind does not match the name's extension. |
| `500` | Encoding the output failed, or an AI model failed at inference time (the runtime errored, or a model session lock was poisoned by an earlier failure). A model that cannot be loaded at all fails startup instead of answering requests. |
| `501` | The URL is valid but this server was not configured for it: an AI filter with no model, a text overlay with no font, a video format with no encoder configured. |
| `502` | The source could not be fetched (network error, a non-2xx answer from the origin other than a 404, or a non-image `Content-Type`), the source is larger than the fetch limit (`CRABCROP_SOURCE_MAX_MB`, 10 MiB by default), or the fetched bytes could not be decoded as an image, including AVIF sources in builds without AVIF decoding, corrupt HEIC sources, and undecodable formats. A watermark that fails to load also lands here, and so does a configured AI model that ran but produced output the server cannot use (no outputs, an unreadable tensor, an unexpected shape). A configured video encoder that fails, times out, or exceeds its output cap also lands here. |
| `503` | The server shed the request because a processing, slow-lane, AI, video, or animation gate waited past its timeout, or the C2PA timestamp authority failed while signing the output; the response carries `Retry-After`. |
| `504` | The source fetch timed out (`CRABCROP_SOURCE_TIMEOUT_SECS`, 10 seconds per fetch by default), or processing ran past the `CRABCROP_PROCESS_TIMEOUT_SECS` limit (30 seconds by default). |

For example, a `removebg` request to a server with no background-removal model returns `501` with the body `removebg is not configured on this server`.
