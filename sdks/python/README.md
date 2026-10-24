# crabcrop_url

Build and sign [crabcrop](https://github.com/dayvson/crabcrop) URLs from Python. Standard library only; the signing scheme is specified in [`spec/url-signing.md`](../../spec/url-signing.md) and this package runs the canonical [`spec/test-vectors.json`](../../spec/test-vectors.json) as its test suite, so it cannot drift from the server.

## Usage

```python
from crabcrop_url import build_url, build_srcset, purge_group_signature, purge_signature, sign, upload_signature, verify

# A signed URL. The operation string is passed verbatim and may
# contain several segments and preset: references.
url = build_url(
    "https://images.example.com",
    "your-secret",
    "crop:1200x630:smart/filters:format(webp)",
    "https://example.com/photo.jpg",
)

# An expiring URL (unix seconds), emitted as a leading exp: segment.
expiring = build_url(
    "https://images.example.com",
    "your-secret",
    "300x200",
    "https://example.com/photo.jpg",
    expires=1893456000,
)

# A responsive srcset: one {width}x0 URL per width.
srcset = build_srcset(
    "https://images.example.com",
    "your-secret",
    "https://example.com/photo.jpg",
    [400, 800, 1200],
)

# The raw signature for a remainder, if you assemble URLs yourself.
signature = sign("your-secret", "300x200/https://example.com/photo.jpg")

# Verify a signature in constant time (malformed tags return False).
ok = verify("your-secret", signature, "300x200/https://example.com/photo.jpg")

# Purge signatures for the server's /purge and /purge-group endpoints
# are domain-separated under a derived subkey; the helpers sign the
# right strings so callers never assemble them by hand.
purge = purge_signature("your-secret", "300x200/https://example.com/photo.jpg")
group = purge_group_signature("your-secret", "thumb")

# Upload signatures for the server's PUT /upload/<signature>/<name>
# endpoint live in a third domain, under their own derived subkey.
upload = upload_signature("your-secret", "heroes/photo.jpg")
```

Pass `None` as the secret to produce the literal `unsafe` signature segment, for servers running with `CRABCROP_ALLOW_UNSAFE` or in presets-only mode. The secret may be `str` or `bytes`.

## Tests

```bash
python3 -m unittest discover -s tests
```
