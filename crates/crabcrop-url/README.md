# crabcrop-url

Build, sign, seal, and verify [crabcrop](https://github.com/dayvson/crabcrop) URLs from Rust.

crabcrop is an on-demand image optimization service: a URL names an image and the operations to apply to it, and a signature (or a sealed token) proves the URL was minted by someone holding the secret. This crate is the client half of that scheme, and it is the same code the server verifies with, so client and server cannot drift. The scheme is specified in [`spec/url-signing.md`](../../spec/url-signing.md), and the crate runs the canonical [`spec/test-vectors.json`](../../spec/test-vectors.json) as part of its test suite.

The same scheme is implemented for other languages, byte for byte against the same test vectors: [`@crabcrop/url`](https://www.npmjs.com/package/@crabcrop/url) for TypeScript and [`crabcrop-url`](https://pypi.org/project/crabcrop-url/) for Python.

## Signed URLs

`build_url` assembles a URL and signs it. The operation string is passed verbatim and may contain several segments and `preset:` references.

```rust
use crabcrop_url::build_url;

let url = build_url(
    "https://images.example.com",
    Some(b"your-secret"),
    "crop:1200x630:smart/filters:format(webp)",
    "https://example.com/photo.jpg",
    None,
)
.unwrap();
// https://images.example.com/<signature>/crop:1200x630:smart/filters:format(webp)/https://example.com/photo.jpg
```

Pass `Some(unix_seconds)` as the last argument to mint an expiring URL. The expiry becomes a leading `exp:` segment inside the signed remainder, so it cannot be stripped or extended without the secret, and the server refuses the URL once the expiry second is reached.

```rust
let expiring = build_url(
    "https://images.example.com",
    Some(b"your-secret"),
    "300x200",
    "https://example.com/photo.jpg",
    Some(1893456000),
)
.unwrap();
```

## Key IDs and secret rotation

A key ID stamps the signature segment with the name of the key that produced it: `<key-id>:<signature>`. The server verifies a keyed segment against the named key only, which makes rotation a two-step overlap instead of a flag day: publish the new key under a new ID, mint with it, and retire the old key once its URLs expire.

```rust
use crabcrop_url::build_url_with_key_id;

let url = build_url_with_key_id(
    "https://images.example.com",
    Some((b"your-new-secret", "k2026")),
    "300x200",
    "https://example.com/photo.jpg",
    None,
)
.unwrap();
// https://images.example.com/k2026:<signature>/300x200/https://example.com/photo.jpg
```

Key IDs are 1 to 32 characters of `[A-Za-z0-9_-]`; builders reject anything else instead of minting a URL the server would refuse.

## Responsive srcset

`build_srcset` produces a `srcset` attribute value: one `{width}x0` URL per width, each signed, with a shared expiry when one is given.

```rust
use crabcrop_url::build_srcset;

let srcset = build_srcset(
    "https://images.example.com",
    Some(b"your-secret"),
    "https://example.com/photo.jpg",
    &[400, 800, 1200],
    None,
)
.unwrap();
// <url> 400w, <url> 800w, <url> 1200w
```

## Sealed URLs

A signed URL shows its recipe to anyone who sees it. A sealed URL encrypts the whole remainder (expiry, operations, and source) into an opaque token segment, so the URL reveals nothing about what it renders or where the source lives:

```
https://images.example.com/s1.<token>/hero.jpg
```

The trailing name is cosmetic: the server never parses or authenticates it, and it is excluded from the token and the cache key, so the sealed and plain forms of one render are one cache entry and one purge target.

```rust
use crabcrop_url::build_sealed_url;

let url = build_sealed_url(
    "https://images.example.com",
    Some(b"your-secret"),
    "crop:1200x630:smart/filters:format(webp)",
    "https://origin.internal/photo.jpg",
    Some(1893456000),
    Some("hero.jpg"),
    None,
)
.unwrap();
```

Tokens are AES-256-GCM under a subkey derived from the secret (see below), with a fresh random nonce per mint. Key IDs work for sealed URLs exactly as for signed ones: pass `Some("k2026")` as the last argument and the segment becomes `k2026:s1.<token>`. There is no `unsafe` spelling of a token; sealing without a secret is an error.

The primitives are public for servers and tests that open tokens directly:

```rust
use crabcrop_url::{seal, unseal};

let token = seal(b"your-secret", "300x200/https://example.com/photo.jpg").unwrap();
assert!(token.starts_with("s1."));
assert_eq!(
    unseal(b"your-secret", &token).unwrap(),
    "300x200/https://example.com/photo.jpg"
);
```

Every open failure (wrong secret, tampered token, malformed segment) is the same `SealError::OpenFailed`, so callers cannot be probed for which check failed. Remainders are capped at 2048 bytes: a recipe longer than that is a preset asking to happen.

## Verification and primitives

If you assemble or check URLs yourself, `sign` and `verify` are the raw HMAC-SHA256 pair. Verification compares in constant time, and a malformed signature segment fails cleanly instead of panicking.

```rust
use crabcrop_url::{sign, verify};

let signature = sign(b"your-secret", "300x200/https://example.com/photo.jpg");
assert!(verify(b"your-secret", &signature, "300x200/https://example.com/photo.jpg"));
```

Builders always emit the canonical lowercase hex tag, and the server accepts only that spelling: an uppercase tag is well-formed hex but not a valid signature segment there, so hand-assembled URLs should reuse builder output rather than re-casing a tag.

## Purge signatures

Cache purge endpoints take their own signatures, domain-separated under a derived subkey, so a signature minted for an image URL can never be replayed against a purge endpoint (or the other way around). The helpers sign the right strings, so callers never assemble them by hand.

```rust
use crabcrop_url::{purge_group_signature, purge_signature};

let purge = purge_signature(b"your-secret", "300x200/https://example.com/photo.jpg");
let group = purge_group_signature(b"your-secret", "thumb");
```

## Upload signatures

Uploads to the server's `PUT /upload/<signature>/<name>` endpoint are signed under their own derived subkey, in the same domain-separated style.

```rust
use crabcrop_url::{upload_signature, upload_signature_with_expiry};

let plain = upload_signature(b"your-secret", "photo.jpg");
```

For delegated upload tokens (handing a client a one-shot right to upload), always use the expiring form: the expiry sits inside the signed string, and the server refuses the upload once it passes. A plain upload signature replays until the secret rotates.

```rust
let expiring = upload_signature_with_expiry(b"your-secret", "photo.jpg", 1893456000);
```

## Unsigned mode

A `None` secret produces the literal `unsafe` signature segment, for servers running with `CRABCROP_ALLOW_UNSAFE` or in presets-only mode. Sealed URLs are the exception: they require a secret, as described above.

## Errors

Builders fail loudly at build time rather than minting a URL that would be refused at fetch time. Failures are typed: `BuildError` for signed URL construction (a character that cannot travel in a URL path, an out-of-range srcset width, an invalid key ID) and `SealError` for sealing and opening (remainder too long, invalid key ID, no secret, and the uniform `OpenFailed`).

## Links

- The crabcrop server and full documentation: https://github.com/dayvson/crabcrop
- The signing scheme, specified: [`spec/url-signing.md`](../../spec/url-signing.md)
- API documentation: https://docs.rs/crabcrop-url
- TypeScript: [`@crabcrop/url`](https://www.npmjs.com/package/@crabcrop/url)
- Python: [`crabcrop-url`](https://pypi.org/project/crabcrop-url/)
