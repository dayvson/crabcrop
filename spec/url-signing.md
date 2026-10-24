# crabcrop URL signing spec

Version 1. This document is the contract between a crabcrop server and anything that builds URLs for it: the Rust builder crate (`crates/crabcrop-url`), the TypeScript package (`sdks/typescript`), the Python package (`sdks/python`), and the framework loaders built on them. It is descriptive of the server implementation; where this document and the server disagree, the server wins and the document is wrong.

## URL structure

```text
{base}/{signature}/{operations...}/{source}
```

- `base` is the deployment origin, e.g. `https://images.example.com`, with no trailing slash.
- `signature` is either the lowercase hex HMAC described below, or the literal `unsafe` (honored only when the server runs with `CRABCROP_ALLOW_UNSAFE`; in presets-only deployments it is the normal form).
- `operations` is zero or more operation segments separated by `/`, in the grammar the README documents, plus the two metadata segment kinds below.
- `source` is the source locator with its slashes intact: an `http(s)://` URL, an `s3://bucket/key`, a `gs://bucket/key`, or a `file://` path. The first segment that is not an operation starts the source; everything from there on is source text, even segments that look like operations.

### Metadata segments

- `exp:<unix-seconds>` may appear anywhere among the operations. It is not an operation. The URL is valid strictly before that second; at or past it the server answers `403`. The segment is inside the signed region, so it cannot be extended without the secret.
- `preset:<name>` may appear anywhere an operation fits. The server expands it to the preset's operation string after verification. The signature covers the reference, not the expansion.

## Signing

```text
signature = lowercase_hex( HMAC_SHA256( key = secret, message = remainder ) )
remainder = everything after the signature segment, exactly as it appears
            in the URL: the operations and the source joined by `/`,
            with no leading slash and no decoding or normalization
```

- The secret is the server's `CRABCROP_SECRET`, encoded as UTF-8 when it is text.
- The message is the remainder encoded as UTF-8. Sign the bytes you send: any percent-encoding, case change, or added/removed character is a different message and fails verification.
- Signatures have one canonical spelling: lowercase hex. Servers reject a signature segment that is not lowercase hex at parse time (`400`); builders always emit lowercase.
- A signature segment may carry a key ID: `<key-id>:<hex>`, where the ID is 1 to 32 characters of `[A-Za-z0-9_-]`. The ID is routing metadata, not secret material: it names which of the server's keys must verify the tag, which is what makes secret rotation possible without invalidating outstanding URLs. A server verifies a keyed segment against the named key only (an unknown ID fails as a bad signature, `403`), and an unkeyed segment against every key it holds. Purge signature segments accept the same prefix and routing. Builders stamp an ID only when asked; verifiers that hold a single secret ignore the ID and check the tag.
- The signature covers the path only. A query string is not signed, is ignored by the server, and must not be relied on for anything.
- Verification failure is `403` with the body `bad signature`. An expired URL is `403` with the body `signature expired`. An unknown preset, a malformed segment, or free-form operations in presets-only mode is a `400`.

## Purge signatures

The purge endpoints are signed under a derived subkey, never the secret itself:

```text
subkey = HMAC_SHA256( key = secret, message = "crabcrop/purge-key" )   (raw bytes)
signature for POST /purge/<signature>/<path>        = lowercase_hex( HMAC_SHA256( subkey, path ) )
signature for POST /purge-group/<signature>/<group> = lowercase_hex( HMAC_SHA256( subkey, "purge-group/" + group ) )
```

`path` is the request remainder exactly as presented. The server canonicalizes it before matching entries, so any equivalent spelling signed this way evicts the shared entry. Domain separation is the point: an image signature can never be replayed as a purge signature, and vice versa. Purges never honor `unsafe`: without a secret there is no purge, in every mode. Client libraries expose these as `purge_signature` / `purge_group_signature` helpers so the signed strings are never assembled by hand.

## Upload signatures

The upload endpoint (issue #168) is signed under its own derived subkey, the same construction in a third domain:

```text
subkey = HMAC_SHA256( key = secret, message = "crabcrop/upload-key" )   (raw bytes)
signature for PUT /upload/<signature>/<name> = lowercase_hex( HMAC_SHA256( subkey, name ) )
```

`name` is the object name exactly as presented after the signature segment (it may contain `/`-separated segments). Domain separation is the point, as with purges: an image signature or a purge signature can never be replayed as an upload signature, and an upload signature authorizes nothing else. Uploads never honor `unsafe`. Signature segments accept the same `<key-id>:` prefix and routing as purge segments. Client libraries expose this as an `upload_signature` / `uploadSignature` helper so the signed string is never assembled by hand.

## Building URLs

A builder assembles, in order:

1. The expiry segment `exp:<unix-seconds>`, when an expiry is requested.
2. The caller's operation string, verbatim (it may itself contain several segments and `preset:` references).
3. The source.

then signs the joined remainder and prefixes `{base}/{signature}/`. With a null secret the signature segment is the literal `unsafe` and no HMAC is computed.

Builders reject inputs that can never verify, at build time: an expiry that is not a non-negative integer, and operations or sources containing `?`, `#`, whitespace, control, or non-ASCII characters. The remainder travels as a raw path and those characters cannot be represented in it (percent-encoding them produces a different remainder, which fails verification just the same). Only a null secret means `unsafe`: an empty-string secret is a real key and signs.

## Srcset building

For responsive images, a builder emits one URL per width, each with the single operation `{width}x0` (proportional resize to that width), formatted as an `srcset` attribute value: entries `{url} {width}w` joined by `, `. An expiry, when requested, applies to every entry.

## Test vectors

`test-vectors.json` in this directory is the canonical set: each entry gives a `secret`, a `remainder`, and the expected `signature`. Every shipped implementation runs these vectors as its test suite. The first entry is the long-standing known-answer vector (secret `Jefe`); the second is the worked example from the README, computed with OpenSSL. The rest were computed with Python's `hmac` module, independently of the server code, and are verified by the Rust suite against the server's own verification path.

## Versioning

The grammar is unversioned: there is no version segment, and a URL minted today keeps its meaning. Additive change is tolerated asymmetrically, and the asymmetry matters:

- An unknown filter inside `filters:`, or a malformed payload on a known prefix (`crop:300x0`, `pad:xyz`), is rejected with `400`: the shape is recognized and fails to parse.
- Anything else unrecognized is read as source text, never rejected: the first segment that does not parse as an operation is where the source URL begins. A new segment kind (a new prefix, a new bare form) and a size-like segment that fails to parse (`300xabc`) all take this path on a server that predates them: the request fails later, as a source error, not a `400`.

A breaking change, if one is ever unavoidable, arrives as a new first segment: the signature is an HMAC over the whole remainder, so a future scheme can introduce its own leading segment and a server can accept both spellings during migration without disturbing URLs already signed.
