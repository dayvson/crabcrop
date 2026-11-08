# Endpoints and CLI

## Endpoints

| Route | Description |
| --- | --- |
| `GET /health` | Health check. Returns `200` with the body `ok`. |
| `GET /metrics` | Prometheus metrics in text format: request counts by status class (`crabcrop_requests_total`), request/load/process durations, cache hit and miss counters, loader and engine error counters, and byte totals, per-lane occupancy (`crabcrop_lane_in_use` against `crabcrop_lane_capacity`, with `lane="process"` for the stills lane and `lane="slow"` for the slow lane), source buffer occupancy (`crabcrop_source_buffer_bytes` against `crabcrop_source_buffer_capacity_bytes`, see `CRABCROP_SOURCE_BUFFER_MAX_MB`), and cache tier failure counters (`crabcrop_cache_tier_errors_total`, with `tier="disk"` and `tier="shared"` for the configured tiers). Gated by `CRABCROP_METRICS_TOKEN`: with no token configured it answers `403`; with one, requests must carry it as a bearer token, and a missing or wrong token answers `401`. |
| `POST /purge/<signature>/<path>` | Invalidate every stored variant of one result, across this server's cache tiers. See below. |
| `POST /purge-group/<signature>/<group>` | Invalidate every stored result in a cache group, across this server's cache tiers. See below. |
| `PUT /upload/<signature>/<name>` | Store one original in the configured upload destination. Only exists when uploads are configured. See [Upload an original](#upload-an-original). |
| `GET /explain/<signature>/...` | A report on the URL instead of the image. See [Self-explaining URLs](#self-explaining-urls). |
| `GET /<signature>/...` | Image requests, as documented above. |

### Purge a cached result

```bash
curl -X POST "http://localhost:3000/purge/<signature>/300x200/https://example.com/photo.jpg"
```

The `<path>` is the original request path without its signature. The signature covers the target exactly as presented; the target is then canonicalized the same way entries are keyed, so purging any equivalent spelling of a path (the `jpg` alias, leading zeros, a preset reference) evicts the shared entry, `exp:` variants included. Compute the signature with your client library (`purge_signature` in the Rust crate and the Python SDK, `purgeSignature` in TypeScript): the HMAC-SHA256 of the path under a purge subkey derived from the secret, so only the secret holder can invalidate, and an image signature can never be replayed as a purge signature. The response is `200` with a body like `purged 2`: the count covers the entry and all of its variants (per format class and DPR). Against a shared bucket, the scan that finds older spellings of the entry is bounded: the request scans the first entries itself and any remainder finishes in a background task, so on a very large bucket the count reports what the request itself removed, the background completion is logged with its own count, and on the receiving server the purge is effective immediately either way: until the scan finishes, that server treats any disk or bucket copy the purge covers as a miss and deletes it on sight, so a request can neither be served that copy nor restore it to a faster tier. A bad purge signature is a `403` and removes nothing; purging a path that is not stored answers `purged 0`.

The guarantee behind that count: the removal covers the receiving server's memory, its disk tier, and the shared bucket when one is configured. It does not reach the memory of any other replica: a replica that already holds the entry can keep serving it until it evicts the entry or restarts, so on a multi-replica deployment a purge takes effect everywhere only as each replica drops its own copy, unless you send the purge to each replica. A replica that misses in its own memory finds nothing underneath in the tiers the purge reached, because the removals reach the receiving server's disk tier and the bucket too, and in those tiers a purged entry is never resurrected from a lower tier. One timing caveat applies while a background scan runs: a bucket copy the scan has not reached yet is still readable by other replicas, and a replica that reads it keeps it in its own memory and disk like a copy it already held. The per-replica limit applies to disk as well as memory: where each replica keeps its own disk directory, one replica's purge does not reach another replica's disk tier, so that replica can still serve the entry from its own disk until it receives the purge itself or evicts the entry. [Deployment](deployment.md#the-cache-tiers-and-what-replicas-change) states the same guarantee with the operational guidance.

### Upload an original

crabcrop is a fetch-only service by design: sources live somewhere else, and transforms reference them by URL. The upload endpoint is the one deliberate exception, and it is off unless the operator turns it on. Configure exactly one destination:

- `CRABCROP_UPLOAD_DIR=/srv/originals` stores originals as files in a local directory.
- `CRABCROP_UPLOAD_BUCKET=originals` stores them in an S3 bucket, reusing the source S3 configuration (endpoint, region, and the AWS credentials from the environment); `CRABCROP_UPLOAD_PREFIX` namespaces the objects. The bucket must also appear in `CRABCROP_ALLOWED_BUCKETS`, because uploaded objects are served back as `s3://` sources. There is no GCS destination: `gs://` sources are read-only in crabcrop.

With neither variable set, uploads are off: every `PUT` to `/upload/...` answers `404` with the code `uploads_not_configured`, whatever its signature. With one set, an upload is one signed `PUT`:

```bash
curl -X PUT --data-binary @photo.jpg "http://localhost:3000/upload/<signature>/heroes/photo.jpg"
```

The `<name>` is the object name, `/`-separated segments of `[A-Za-z0-9._-]` (no empty, `.`, or `..` segments, nothing starting with a dot, at most 255 bytes). The signature covers the name exactly as presented, under an upload subkey derived from the secret, the same domain-separation construction purge signatures use: only the secret holder can upload, an image signature or a purge signature can never be replayed here, and `unsafe` never authorizes an upload. Compute it with your client library (`upload_signature` in the Rust crate and the Python SDK, `uploadSignature` in TypeScript), or let the CLI do the whole job:

```bash
crabcrop upload ./photo.jpg heroes/photo.jpg
```

An upload token can carry an expiry, and a delegated token should: with `PUT /upload/<signature>/exp:<unix-seconds>/<name>` the signature covers the `exp:` segment along with the name, so the token stops working at that second and can neither be stripped of its expiry nor extended. The client libraries mint this form with an `expires` argument, and the CLI with `--expires <unix-seconds>`.

The body must decode as an image this server can serve (the same decode paths a transform request uses, size and dimension caps included); anything else is a `415` and is never stored. The validation decode runs under the same process gate as transforms, so a saturated server sheds an upload with a `503` and `Retry-After` instead of decoding ungated. The body is charged as it arrives to the source buffer budget (`CRABCROP_SOURCE_BUFFER_MAX_MB`) that cache misses share, so a client that stalls mid-body holds only what it sent, and an upload that cannot get budget within 5 seconds sheds with the same `503`. `CRABCROP_UPLOAD_MAX_MB` caps one upload (default 10 MB, the source fetch cap, so a stored original is always one the loader will fetch back; larger configurations are refused at startup): a body past it is a `413`. A body the server cannot read to its end, because the client aborted it or framed it badly, is a `400` (`upload_unreadable`) and is never stored. What is stored is the validated form: for an SVG that is the sanitized document (scripts and event handlers stripped at store time, not only when crabcrop serves it back), and the name's extension must match the media kind the body validated as, so a PNG cannot be stored as `photo.jpg` or as `promo.html` (a name with no extension is accepted as-is). An upload to an existing name is a `409`, answered before the body is decoded, and the final write is a conditional create, so the guarantee holds even against a racing upload for the same new name; set `CRABCROP_UPLOAD_ALLOW_OVERWRITE=1` to let a `PUT` replace it (a `200` instead of the new-object `201`). A replacing `PUT` also purges every cached derivative of the replaced original, in all of this server's tiers, so the next transform request re-renders from the new bytes instead of serving a stale result. There is no listing and no delete.

The response body is the exact source reference to embed in transform URLs (`file:///heroes/photo.jpg` for a local destination, `s3://originals/heroes/photo.jpg` for a bucket), also returned in the `X-Crabcrop-Source` header. Serving works through the normal loader: a local destination doubles as the `file://` source root when no `CRABCROP_FILE_ROOT` is configured (and must sit under one that is), so the uploaded original is transformable the moment the `PUT` answers.

One posture note, stated plainly: enabling uploads changes this server from fetch-only to a service that accepts and stores caller-supplied bytes. The mitigations are the ones above (off by default, its own signature domain with an expiring form, decode-before-store, sanitized SVG storage, extension binding, a size cap, no overwrite by default, no listing or delete), and the destination should still be a directory or bucket dedicated to uploads, watched like any other store of user-supplied originals. If you serve the destination directly, serve it as images: the stored bytes are validated images under bound names, not arbitrary content, and a dedicated host keeps even a hypothetical confusion off your main domain.

### Self-explaining URLs

Every crabcrop URL can explain itself. Take any image URL, put `explain/` in front of the signature segment, and the server answers with a report on that URL instead of the image:

```bash
curl "http://localhost:3000/explain/<signature>/crop:1200x630:smart/filters:format(avif):quality(auto)/https://example.com/photo.jpg"
```

Browsers get a readable HTML page; send `Accept: application/json` for the same report as JSON. The report carries the signature verdict (valid, which key, when the link expires), the source and its probed facts (media kind, pixel dimensions, and the encoded size when the fetch states it), every operation glossed in plain language in order, the referenced presets with their expansions, a forecast (output dimensions when the chain determines them, the output format, the output pixel count against the aggregate budget), the canonical form that gets cached with its cache group, and links back to the image itself and to the debug focal remainder when the chain contains a crop. The CLI's `explain` verb does the offline half of this (grammar and signature against a secret you pass); the server route adds the live facts only a server can know.

Explain is safe to expose to the same callers who hold signed URLs: the signature and expiry are verified exactly as the image path verifies them, before any source access, so a bad or expired URL is refused with the image path's own response and costs zero source fetches, and the one probe a valid URL pays for reads only the source's head through the loader, stopping as soon as the head identifies the source, with no decode, no detection, and nothing written to any cache tier. The encoded size is reported when the fetch itself states it (a declared length, or a body small enough to end inside the head); the probe never downloads a body just to count it, so a large source can report no size. The report contains only facts derivable from the URL and the source itself: no secrets or key material, no configuration values, paths, or fingerprints, no redirect or address details, and errors are the usual frozen codes and generic messages.

### Cache headers and groups

Image responses carry two standard headers. `Cache-Status` (RFC 9211) reports what the server's own caches did: `crabcrop; hit` for a memory hit, `crabcrop; hit; detail="disk"` for a disk hit, `crabcrop; hit; detail="shared"` for a shared-tier hit, and `crabcrop; fwd=uri-miss` when the response was produced from the origin.

A response also belongs to a cache group (RFC 9875) for each preset its URL referenced, listed in the `Cache-Group` header:

```http
Cache-Group: "small"
```

Groups make preset-wide invalidation one call instead of a script over every URL. Group membership accumulates: an entry stored before a group was ever referenced joins the group the first time any grouped spelling of its URL is served, so a group purge reaches entries that were first rendered through a plain, preset-free URL. The signature comes from the same libraries (`purge_group_signature` / `purgeGroupSignature`), which derive the purge-group endpoint's own subkey and sign the group name on your behalf, so nobody constructs it by hand:

```bash
curl -X POST "http://localhost:3000/purge-group/<signature>/small"
```

The response is `200` with a body like `purged 3`: every entry recorded in the group, across this server's cache tiers, with the same reach as a per-URL purge (this server's tiers and the shared bucket, not other replicas' memory) and the same bounded shared-bucket scan: on a very large bucket the count reports what the request itself removed and the remainder finishes in the background, logged on completion, while the receiving server treats any disk or bucket copy of a group member as a miss until the scan ends. A bad signature is a `403` and removes nothing; an unknown group answers `purged 0`.

A [sealed URL](url-grammar.md#sealed-urls)'s responses carry no `Cache-Group`: its preset names are inside the token, and the header would show them to anyone who fetches the URL. Its cache entry still joins those groups, so `purge-group` reaches sealed renders exactly as it reaches plain ones. A CDN group purge does not. The group purge bridges in the [CDN recipes](deployment.md#put-a-cdn-in-front) for Cloudflare (the Worker), Fastly, BunnyCDN, and CloudFront build the edge's tags from this header, and the path-rule bridges (Cloudflare's Cache Response Rule, Akamai) match the `preset:` segment the token hides, so the edge copy of a sealed response carries no tag. Purge sealed URLs at the CDN by URL.

Every image request also writes one structured JSON log line to stdout, with the operation summary, status, per-stage timings, and cache result (never the full source URL).

## Command line

The `crabcrop` binary doubles as its own operator CLI: when the first argument is a verb, it runs the verb and exits instead of starting the server. Configuration comes from flags over environment variables, the same convention as the MCP server: `--base` / `--secret` over `CRABCROP_BASE` / `CRABCROP_SECRET`, with the base defaulting to `http://localhost:3000`.

```bash
export CRABCROP_BASE=http://localhost:3000
export CRABCROP_SECRET=your-secret

# Sign a URL (validated against the server grammar first)
crabcrop sign "300x200/filters:grayscale/https://example.com/photo.jpg"

# Explain a URL segment by segment, verifying its signature
crabcrop explain "http://localhost:3000/<signature>/300x200/filters:grayscale/https://example.com/photo.jpg"

# Fetch the meta JSON for an image (a bare remainder is signed first).
# The server needs meta mode on (CRABCROP_META_MODE=1); it is off by default.
crabcrop meta "300x200/https://example.com/photo.jpg"

# ... with placeholder fields and detection boxes included
crabcrop meta --placeholders --detections "300x200/https://example.com/photo.jpg"

# Purge one URL, or a whole preset group (Cache-Group)
crabcrop purge "300x200/https://example.com/photo.jpg"
crabcrop purge-group thumb

# Seal a remainder into an opaque token URL (with an optional friendly
# name), or recover the remainder a sealed URL carries
crabcrop seal "300x200/https://example.com/photo.jpg" photo.jpg
crabcrop unseal "http://localhost:3000/s1.<token>/photo.jpg"

# Upload an original to the server's upload store (prints the
# source reference for transform URLs; see Upload an original)
crabcrop upload ./photo.jpg heroes/photo.jpg

# Convert a Thumbor URL (re-signed; losses are reported on stderr)
crabcrop convert-thumbor "unsafe/300x200/smart/https://example.com/photo.jpg"
```
