# crabcrop_url

Build and sign [crabcrop](https://github.com/dayvson/crabcrop) URLs from Python. Standard library only; the signing scheme is specified in [`spec/url-signing.md`](../../spec/url-signing.md) and this package runs the canonical [`spec/test-vectors.json`](../../spec/test-vectors.json) as its test suite, so it cannot drift from the server.

## Usage

```python
from crabcrop_url import build_sealed_url, build_url, build_srcset, purge_group_signature, purge_signature, seal, sign, unseal, upload_signature, verify

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

# A responsive srcset: one {width}x0 URL per width. An expiry and a
# key_id, when given, apply to every entry.
srcset = build_srcset(
    "https://images.example.com",
    "your-secret",
    "https://example.com/photo.jpg",
    [400, 800, 1200],
)

# The signature for a remainder, if you assemble URLs yourself. It
# raises ValueError for a remainder that begins with the reserved
# segment "crabcrop" (is_reserved_remainder), as every builder does:
# the derived subkeys are signatures of labels under that segment.
signature = sign("your-secret", "300x200/https://example.com/photo.jpg")

# Verify a signature in constant time (malformed tags return False).
ok = verify("your-secret", signature, "300x200/https://example.com/photo.jpg")

# Purge signatures for the server's /purge and /purge-group endpoints
# are domain-separated under a derived subkey; the helpers sign the
# right strings so callers never assemble them by hand. Pass an
# expiry (unix seconds) for a purge URL that stops working then;
# its path is /purge/<signature>/exp:<expiry>/<path> (or
# /purge-group/<signature>/exp:<expiry>/<group>).
purge = purge_signature("your-secret", "300x200/https://example.com/photo.jpg")
group = purge_group_signature("your-secret", "thumb")
expiring_purge = purge_signature("your-secret", "300x200/https://example.com/photo.jpg", 1893456000)
expiring_group = purge_group_signature("your-secret", "thumb", 1893456000)

# Upload signatures for the server's PUT /upload/<signature>/<name>
# endpoint live in a third domain, under their own derived subkey.
# Pass an expiry (unix seconds) for a token that stops working
# then; its path is /upload/<signature>/exp:<expiry>/<name>.
upload = upload_signature("your-secret", "heroes/photo.jpg")
expiring = upload_signature("your-secret", "heroes/photo.jpg", 1893456000)

# Sealed URLs (issue #255): the whole remainder encrypted into an
# opaque token in the signature slot, for public links whose recipe
# should stay unreadable. The server serves the sealed and plain
# forms of one render from the same cache entry. Sealing alone
# needs the cryptography package: pip install crabcrop-url[seal]
sealed_url = build_sealed_url(
    "https://images.example.com",
    "your-secret",
    "300x200",
    "https://example.com/photo.jpg",
    name="photo.jpg",
)
remainder = unseal("your-secret", seal("your-secret", "300x200/https://example.com/photo.jpg"))
```

## Typed builders

The operation string is the spec, and passing it verbatim stays the primary API. For callers who would rather build it, `ops` and `filter` are typed builders, the Python spelling of the TypeScript SDK's: each call checks its arguments against the server's grammar and returns the canonical spelling, so the typed and hand-written forms sign byte-identical URLs. A value the server would refuse raises `ValueError` at build time. Both SDKs run the builder cases in [`spec/operation-vectors.json`](../../spec/operation-vectors.json), so they spell every operation alike: a decimal argument is written the way the server prints it, the shortest decimal that reads back as the same 32-bit float (`ops.focal_point(0.1 + 0.2, 0.4, 2)` is `focalpoint:0.3x0.4:2`), and a class name is written trimmed (`ops.crop_objects(100, 100, [" person "])` is `crop:100x100:objects(person):1`).

```python
import crabcrop_url as cc

operations = cc.join_ops(
    cc.ops.crop(1200, 630, "smart"),
    cc.ops.filters(cc.filter.format("webp", method=6), cc.filter.quality(80)),
)
# "crop:1200x630:smart/filters:format(webp,method:6):quality(80)"
url = cc.build_url("https://images.example.com", "your-secret", operations, "https://example.com/photo.jpg")

cc.ops.crop_objects(400, 300, ["person", "car"], 1.5)  # "crop:400x300:objects(person,car):1.5"
cc.MAX_DIMENSION  # 16383, the server's per-axis pixel cap
```

Each TypeScript name has its Python spelling: `buildURL` is `build_url`, `ops.cropObjects` is `ops.crop_objects`, `filter.qualityPerFormat` is `filter.quality_per_format`, and an options object (`filter.format("webp", { method: 6 })`, `filter.studio("product", { bg: "white" })`) is keyword arguments. `seal` is the TypeScript and Rust name for sealing a remainder; `seal_url`, its earlier Python name, still works. Integer arguments take an `int`: a float, even `300.0`, and a `bool` raise, as `build_srcset` widths do. Imported by name, `filter` shadows the builtin `filter` in your module; `import crabcrop_url as cc` keeps both.

The package ships its type annotations (`py.typed`), so a type checker reads the builders' argument types, such as the `Literal` alignments of `ops.crop`, and `crabcrop_url.__version__` is the release version.

Pass `None` as the secret to produce the literal `unsafe` signature segment, for servers running with `CRABCROP_ALLOW_UNSAFE` or in presets-only mode. The secret may be `str` or `bytes`.

The builders raise `ValueError` on operations or a source that can never verify: `?`, `#`, whitespace, control and non-ASCII characters, and `"` `<` `>` `\` `^` `` ` `` `{` `|` `}`, which a browser rewrites before an `<img src>` or `fetch()` request leaves the page (`\` becomes `/`, the rest are percent-encoded). Percent-encode those in the source first (`x{y}.jpg` as `x%7By%7D.jpg`); the error names the encoding. The full rules are in [`spec/url-signing.md`](../../spec/url-signing.md#building-urls).

## Tests

```bash
python3 -m unittest discover -s tests
```
