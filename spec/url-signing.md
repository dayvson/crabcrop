# crabcrop URL signing spec

Version 1. This document is the contract between a crabcrop server and anything that builds URLs for it: the Rust builder crate (`crates/crabcrop-url`), the TypeScript package (`sdks/typescript`), the Python package (`sdks/python`), and the framework loaders built on them. It is descriptive of the server implementation; where this document and the server disagree, the server wins and the document is wrong.

## URL structure

```text
{base}/{signature}/{operations...}/{source}
```

- `base` is the deployment origin, e.g. `https://images.example.com`, with no trailing slash.
- `signature` is either the lowercase hex HMAC described below, or the literal `unsafe` (honored only when the server runs with `CRABCROP_ALLOW_UNSAFE`; in presets-only deployments it is the normal form).
- `operations` is zero or more operation segments separated by `/`, in the grammar the README documents, plus the two metadata segment kinds below.
- `source` is the source locator with its slashes intact: an `http(s)://` URL, an `s3://bucket/key`, a `gs://bucket/key`, a `file://` path, or the short form `root:<name>/<path>` naming a server-configured source root. The first segment that is not an operation starts the source; everything from there on is source text, even segments that look like operations.

### Metadata segments

- `exp:<unix-seconds>` may appear anywhere among the operations. It is not an operation. The URL is valid strictly before that second; at or past it the server answers `403`. The segment is inside the signed region, so it cannot be extended without the secret.
- `preset:<name>` may appear anywhere an operation fits. The server expands it to the preset's operation string after verification. The signature covers the reference, not the expansion.
- `root:<name>/<path>` is a source, not an operation: it is valid only as the whole source. The server expands it against the named root's base URL after verification, the same way a preset reference expands; the signature covers the short form exactly as written, and a short source and the full source it expands to are one render (one cache entry, one purge target). A name the server has not configured is a `400` (`unknown_root`); a path that climbs above the root is a `400` (`invalid_root`).

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
- Verification failure is `403` with the body `bad signature`. An expired URL is `403` with the body `signature expired`. An unknown preset, an unknown or invalid source root, a malformed segment, or free-form operations in presets-only mode is a `400`.

## Purge signatures

The purge endpoints are signed under derived subkeys, never the secret itself, one subkey per endpoint:

```text
purge subkey       = HMAC_SHA256( key = secret, message = "crabcrop/purge-key" )        (raw bytes)
purge-group subkey = HMAC_SHA256( key = secret, message = "crabcrop/purge-group-key" )  (raw bytes)
signature for POST /purge/<signature>/<path>        = lowercase_hex( HMAC_SHA256( purge subkey, path ) )
signature for POST /purge-group/<signature>/<group> = lowercase_hex( HMAC_SHA256( purge-group subkey, group ) )
```

`path` is the request remainder exactly as presented. The server canonicalizes it before matching entries, so any equivalent spelling signed this way evicts the shared entry. Domain separation is the point: an image signature can never be replayed as a purge signature, and vice versa, and because each purge endpoint has its own subkey, a per-target purge signature can never be transplanted to `/purge-group` (or the reverse), whatever string it was minted over. Purges never honor `unsafe`: without a secret there is no purge, in every mode. Client libraries expose these as `purge_signature` / `purge_group_signature` helpers so the signed strings are never assembled by hand.

## Upload signatures

The upload endpoint (issue #168) is signed under its own derived subkey, the same construction in a third domain:

```text
subkey = HMAC_SHA256( key = secret, message = "crabcrop/upload-key" )   (raw bytes)
signature for PUT /upload/<signature>/<name> = lowercase_hex( HMAC_SHA256( subkey, name ) )
```

`name` is the object name exactly as presented after the signature segment (it may contain `/`-separated segments). Domain separation is the point, as with purges: an image signature or a purge signature can never be replayed as an upload signature, and an upload signature authorizes nothing else. Uploads never honor `unsafe`. Signature segments accept the same `<key-id>:` prefix and routing as purge segments. Client libraries expose this as an `upload_signature` / `uploadSignature` helper so the signed string is never assembled by hand.

An upload signature may carry an expiry, the grammar's `exp:` segment in its upload spelling (issue #192):

```text
signature for PUT /upload/<signature>/exp:<unix-seconds>/<name> = lowercase_hex( HMAC_SHA256( subkey, "exp:<unix-seconds>/" + name ) )
```

The expiry is part of the signed string, so it can neither be stripped (the remainder no longer matches) nor extended (ditto), and the name stays bound exactly as in the plain form. The server refuses the upload with `403` once the current second reaches the expiry, the same comparison image URLs use. A leading remainder segment that opens with `exp:` but does not parse is a grammar error (`invalid_expiry`), never a name; upload names cannot contain `:` in any case. Delegated upload tokens should always be minted in the expiring form: a plain token authorizes its name until the secret rotates.

## Building URLs

A builder assembles, in order:

1. The expiry segment `exp:<unix-seconds>`, when an expiry is requested.
2. The caller's operation string, verbatim (it may itself contain several segments and `preset:` references).
3. The source.

then signs the joined remainder and prefixes `{base}/{signature}/`. With a null secret the signature segment is the literal `unsafe` and no HMAC is computed.

Builders reject inputs that can never verify, at build time: an expiry that is not a non-negative integer, and operations or sources containing `?`, `#`, whitespace, control, or non-ASCII characters. The remainder travels as a raw path and those characters cannot be represented in it (percent-encoding them produces a different remainder, which fails verification just the same). Only a null secret means `unsafe`: an empty-string secret is a real key and signs.

## Srcset building

For responsive images, a builder emits one URL per width, each with the single operation `{width}x0` (proportional resize to that width), formatted as an `srcset` attribute value: entries `{url} {width}w` joined by `, `. An expiry, when requested, applies to every entry.

## Sealed URLs

The sealed form replaces the signature segment with an encrypted token that carries the whole remainder:

```
{base}/[<key-id>:]s1.<token>/{friendly-name}
```

The token is `s1.` (the scheme version marker) followed by base64url without padding of `nonce (12 bytes) || ciphertext || tag (16 bytes)`: AES-256-GCM over the remainder exactly as written (pre-expansion, no inner signature; the AEAD tag replaces the HMAC), under the key `HMAC_SHA256(secret, "crabcrop/seal-key")` as raw bytes, with the fixed associated data `crabcrop/sealed/1`. Detection is a prefix test on the first segment after the optional `<key-id>:` prefix is split off: a plain signature is 64 lowercase hex characters or `unsafe`, and a key ID is `[A-Za-z0-9_-]`, so no valid plain segment can begin `s1.`. The key ID is routing metadata (as with signatures): it is not part of the ciphertext or the AAD, and a keyed token opens only under the named key, an unkeyed one under current-then-previous. There is no separate seal secret; each keyring entry derives its own seal subkey, so the signature rotation window is the token rotation window.

After a successful open, the decrypted remainder enters the serve pipeline exactly where a verified plain remainder enters (reserved-segment guard, preset expansion, parsing, source-root expansion, server defaults, expiry, canonical cache key). The sealed and plain forms of one render are therefore one cache entry, one ETag, and one purge target. The seal builder refuses a remainder longer than 2,048 bytes; the server refuses a token segment longer than 4,096 characters before attempting decode. The friendly name is one opaque cosmetic suffix after the token: it may be empty (the URL may end at the token) and may contain slashes; it is never parsed, decoded, normalized, validated, or authenticated, is not part of the AAD, and is excluded from the cache key, so its extension means nothing for format negotiation or anything else.

Every sealed failure is `403` with body `bad signature`: an unparseable token, bad base64url, a wrong-length body, an unknown version marker in a sealed-looking segment (the `s<digits>.` shape), an unknown key ID, a tag mismatch under every candidate key, an oversized token, a sealed spelling on a deployment that cannot open tokens, and any payload failure after a successful open (malformed operations, unknown preset or root, a presets-only violation, a reserved segment at the payload head). The sole exception: a successfully opened token whose payload is expired returns `403` with body `signature expired`, exactly as the plain form. `/explain` never unseals: it answers a sealed spelling with the same uniform refusal and performs no source fetch, because a server endpoint that opens tokens would be an unsealing oracle; operators unseal locally (the CLI `unseal` verb, the SDK `unseal`) under their own secret and then explain the plain form. Meta for a request that arrived sealed carries the render facts and the image-derived sections, omits every field that reports operation or source text, and carries `sealed: true`. Purge canonicalization unseals a sealed target first (after the purge signature verifies over the target as presented), falling back to the presented target when the token does not open; group purge is unchanged. There is no `unsafe` spelling of a token, and sealing without a secret is an error.

The adversary is the page reader: visitors and competitors scraping public pages, who should learn nothing about the operations or the origin from a URL they can see. The intermediary is out of scope by construction (a CDN sees the request line whatever it contains; TLS is the answer there), replay is inherent and identical to plain signed URLs, and a token minted under one deployment opens on any deployment sharing its secret, exactly like a signature. Sealing is not access control: authorization on crabcrop remains the operator's allowlist, expiry, and whatever fronts the deployment.

Implementations: `seal` / `seal_with_nonce` (the deterministic seam the vectors use; a nonce must never repeat under one secret) / `unseal` / `build_sealed_url` in the `crabcrop-url` crate; `seal` / `sealWithNonce` / `unseal` / `buildSealedURL` in the TypeScript package; `seal_url` / `seal_with_nonce` / `unseal` / `build_sealed_url` in the Python package, where sealing alone needs the `cryptography` package from the `crabcrop-url[seal]` extra (the rest of the package stays zero-dependency).

## Test vectors

`test-vectors.json` in this directory is the canonical set: each signing entry gives a `secret`, a `remainder`, and the expected `signature`; each sealed entry (marked `"sealed": true`) gives a `secret`, an optional `key_id`, a `nonce` (hex), a `remainder`, and the expected `token`, which every implementation must reproduce byte for byte with the injected nonce and open back. Every shipped implementation runs these vectors as its test suite. The first entry is the long-standing known-answer vector (secret `Jefe`); the second is the worked example from the README, computed with OpenSSL. The rest were computed with Python's `hmac` module, independently of the server code, and are verified by the Rust suite against the server's own verification path.

## Versioning

The grammar is unversioned: there is no version segment, and a URL minted today keeps its meaning. Additive change is tolerated asymmetrically, and the asymmetry matters:

- An unknown filter inside `filters:`, or a malformed payload on a known prefix (`crop:300x0`, `pad:xyz`), is rejected with `400`: the shape is recognized and fails to parse.
- Anything else unrecognized is read as source text, never rejected: the first segment that does not parse as an operation is where the source URL begins. A new segment kind (a new prefix, a new bare form) and a size-like segment that fails to parse (`300xabc`) all take this path on a server that predates them: the request fails later, as a source error, not a `400`.

A breaking change, if one is ever unavoidable, arrives as a new first segment: the signature is an HMAC over the whole remainder, so a future scheme can introduce its own leading segment and a server can accept both spellings during migration without disturbing URLs already signed.
