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
- A signature segment that does not hex-decode to a 32-byte tag fails verification. (Hex decoding is case-insensitive; builders always emit lowercase.) Servers compare in constant time.
- Verification failure is `403` with the body `bad signature`. An expired URL is `403` with the body `signature expired`. An unknown preset, a malformed segment, or free-form operations in presets-only mode is a `400`.

## Building URLs

A builder assembles, in order:

1. The expiry segment `exp:<unix-seconds>`, when an expiry is requested.
2. The caller's operation string, verbatim (it may itself contain several segments and `preset:` references).
3. The source.

then signs the joined remainder and prefixes `{base}/{signature}/`. With a null secret the signature segment is the literal `unsafe` and no HMAC is computed.

## Srcset building

For responsive images, a builder emits one URL per width, each with the single operation `{width}x0` (proportional resize to that width), formatted as an `srcset` attribute value: entries `{url} {width}w` joined by `, `. An expiry, when requested, applies to every entry.

## Test vectors

`test-vectors.json` in this directory is the canonical set: each entry gives a `secret`, a `remainder`, and the expected `signature`. Every shipped implementation runs these vectors as its test suite. The first entry is the long-standing known-answer vector (secret `Jefe`); the second is the worked example from the README, computed with OpenSSL. The rest were computed with Python's `hmac` module, independently of the server code, and are verified by the Rust suite against the server's own verification path.

## Versioning

Changes to the grammar or the signing scheme that would invalidate existing URLs require a new spec version and a server that accepts both during migration. Additive changes (new operation kinds, new metadata segments) do not: old builders keep working because unknown operation segments are rejected by the server with `400`, never silently misread.
