# Endpoints and CLI

## Endpoints

| Route | Description |
| --- | --- |
| `GET /health` | Health check. Returns `200` with the body `ok`. |
| `GET /metrics` | Prometheus metrics in text format: request counts by status class (`crabcrop_requests_total`), request/load/process durations, cache hit and miss counters, loader and engine error counters, and byte totals, and per-lane occupancy (`crabcrop_lane_in_use` against `crabcrop_lane_capacity`, with `lane="process"` for the stills lane and `lane="slow"` for the slow lane). Gated by `CRABCROP_METRICS_TOKEN`: with no token configured it answers `403`; with one, requests must carry it as a bearer token, and a missing or wrong token answers `401`. |
| `POST /purge/<signature>/<path>` | Invalidate every stored variant of one result, across this server's cache tiers. See below. |
| `POST /purge-group/<signature>/<group>` | Invalidate every stored result in a cache group, across this server's cache tiers. See below. |
| `PUT /upload/<signature>/<name>` | Store one original in the configured upload destination. Only exists when uploads are configured. See [Upload an original](#upload-an-original). |
| `GET /explain/<signature>/...` | A report on the URL instead of the image. See [Self-explaining URLs](#self-explaining-urls). |
| `GET /<signature>/...` | Image requests, as documented above. |

### Purge a cached result

```bash
curl -X POST "http://localhost:3000/purge/<signature>/300x200/https://example.com/photo.jpg"
```

The `<path>` is the original request path without its signature. The signature covers the target exactly as presented; the target is then canonicalized the same way entries are keyed, so purging any equivalent spelling of a path (the `jpg` alias, leading zeros, a preset reference) evicts the shared entry, `exp:` variants included. Compute the signature with your client library (`purge_signature` in the Rust crate and the Python SDK, `purgeSignature` in TypeScript): the HMAC-SHA256 of the path under a purge subkey derived from the secret, so only the secret holder can invalidate, and an image signature can never be replayed as a purge signature. The response is `200` with a body like `purged 2`: the count covers the entry and all of its variants (per format class and DPR). A bad purge signature is a `403` and removes nothing; purging a path that is not stored answers `purged 0`.

The guarantee behind that count: the removal covers the receiving server's memory, its disk tier, and the shared bucket when one is configured. It does not reach the memory of any other replica: a replica that already holds the entry can keep serving it until it evicts the entry or restarts, so on a multi-replica deployment a purge takes effect everywhere only as each replica drops its own copy, unless you send the purge to each replica. A replica that misses in its own memory finds nothing underneath, because the removals reach the disk tier and the bucket too; a purged entry is never resurrected from a lower tier. [Deployment](deployment.md#the-cache-tiers-and-what-replicas-change) states the same guarantee with the operational guidance.

### Upload an original

crabcrop is a fetch-only service by design: sources live somewhere else, and transforms reference them by URL. The upload endpoint is the one deliberate exception, and it is off unless the operator turns it on. Configure exactly one destination:

- `CRABCROP_UPLOAD_DIR=/srv/originals` stores originals as files in a local directory.
- `CRABCROP_UPLOAD_BUCKET=originals` stores them in an S3 bucket, reusing the source S3 configuration (endpoint, region, and ambient AWS credentials); `CRABCROP_UPLOAD_PREFIX` namespaces the objects. The bucket must also appear in `CRABCROP_ALLOWED_BUCKETS`, because uploaded objects are served back as `s3://` sources. There is no GCS destination: `gs://` sources are read-only in crabcrop.

With neither variable set, the endpoint does not exist: a `PUT` is a plain `404`. With one set, an upload is one signed `PUT`:

```bash
curl -X PUT --data-binary @photo.jpg "http://localhost:3000/upload/<signature>/heroes/photo.jpg"
```

The `<name>` is the object name, `/`-separated segments of `[A-Za-z0-9._-]` (no empty, `.`, or `..` segments, nothing starting with a dot, at most 255 bytes). The signature covers the name exactly as presented, under an upload subkey derived from the secret, the same domain-separation construction purge signatures use: only the secret holder can upload, an image signature or a purge signature can never be replayed here, and `unsafe` never authorizes an upload. Compute it with your client library (`upload_signature` in the Rust crate and the Python SDK, `uploadSignature` in TypeScript), or let the CLI do the whole job:

```bash
crabcrop upload ./photo.jpg heroes/photo.jpg
```

An upload token can carry an expiry, and a delegated token should: with `PUT /upload/<signature>/exp:<unix-seconds>/<name>` the signature covers the `exp:` segment along with the name, so the token stops working at that second and can neither be stripped of its expiry nor extended. The client libraries mint this form with an `expires` argument, and the CLI with `--expires <unix-seconds>`.

The body must decode as an image this server can serve (the same decode paths a transform request uses, size and dimension caps included); anything else is a `415` and is never stored. The validation decode runs under the same process gate as transforms, so a saturated server sheds an upload with a `503` and `Retry-After` instead of decoding ungated. `CRABCROP_UPLOAD_MAX_MB` caps one upload (default 10 MB, the source fetch cap, so a stored original is always one the loader will fetch back; larger configurations are refused at startup). What is stored is the validated form: for an SVG that is the sanitized document (scripts and event handlers stripped at store time, not only when crabcrop serves it back), and the name's extension must match the media kind the body validated as, so a PNG cannot be stored as `photo.jpg` or as `promo.html` (a name with no extension is accepted as-is). An upload to an existing name is a `409`, answered before the body is decoded, and the final write is a conditional create, so the guarantee holds even against a racing upload for the same new name; set `CRABCROP_UPLOAD_ALLOW_OVERWRITE=1` to let a `PUT` replace it (a `200` instead of the new-object `201`). A replacing `PUT` also purges every cached derivative of the replaced original, in all of this server's tiers, so the next transform request re-renders from the new bytes instead of serving a stale result. There is no listing and no delete.

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

The response is `200` with a body like `purged 3`: every entry recorded in the group, across this server's cache tiers, with the same reach as a per-URL purge (this server's tiers and the shared bucket, not other replicas' memory). A bad signature is a `403` and removes nothing; an unknown group answers `purged 0`.

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

# Fetch the meta JSON for an image (a bare remainder is signed first)
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
