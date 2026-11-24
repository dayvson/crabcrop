# crabcrop_url

Build, sign, verify and explain [crabcrop](https://github.com/dayvson/crabcrop) URLs from Python. Standard library only; the signing scheme is specified in [`spec/url-signing.md`](../../spec/url-signing.md) and this package runs the canonical [`spec/test-vectors.json`](../../spec/test-vectors.json) as its test suite, so it cannot drift from the server. The explainer runs the grammar corpora [`spec/explain-verdicts.json`](../../spec/explain-verdicts.json) and [`spec/operation-vectors.json`](../../spec/operation-vectors.json), as the TypeScript SDK's does.

## Usage

```python
from crabcrop_url import build_sealed_url, build_url, build_srcset, purge_group_signature, purge_signature, seal, sign, unseal, upload_signature, verify, verify_url_signature

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

# Verify a signed URL's signature, in constant time (unsigned or
# malformed URLs return None, and so do URLs longer than 64 KiB).
ok = verify_url_signature("your-secret", url)

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

## Explaining and verifying URLs

`explain_url` and `verify_url_signature` are the TypeScript SDK's `explainUrl` and `verifyUrlSignature`, with the same answers and, for `explain_url`, the same messages: the test suite runs the two explainers, and the two verifiers, against each other on generated inputs (see [Tests](#tests)).

| TypeScript | Python |
|---|---|
| `explainUrl(input, nowSeconds?)` | `explain_url(input, now_seconds=None)` |
| `verifyUrlSignature(secret, input)`, where `secret` is a string or a `Record<string, string>` keyring | `verify_url_signature(secret_or_keyring, input)`, where the secret is a `str` or `bytes`, and a keyring is a mapping of key ID to secret (`Keyring`) |
| `Explanation`, `ExplainedSegment`, `SegmentKind`, `SignatureKind` | the same names: `Explanation` and `ExplainedSegment` are frozen dataclasses, and the signature is an `ExplainedSignature` |
| an absent `detail`, `error`, `expired` or `keyId` | `None` in `detail`, `error`, `expired` or `key_id` |
| `null` from `verifyUrlSignature` | `None` |
| `FILTER_DESCRIPTIONS` | `FILTER_DESCRIPTIONS`, read only |

```python
from crabcrop_url import explain_url, verify_url_signature

url = "https://images.example.com/62d8085a0734c070c0f42e45b3c5e1f818f50afd56e38ad0de3a451b10d660e5/crop:1200x630:smart/filters:blur(600)/https://example.com/photo.jpg"
out = explain_url(url)
out.signature.kind      # "hmac"
out.segments[0].detail  # "crop to the aspect of 1200x630 and resize to it; alignment: smart"
out.segments[1].error   # "the server rejects this filter: blur(600)"
out.source              # "https://example.com/photo.jpg"
out.errors              # ["the server rejects this filter: blur(600)"], a 400 (invalid_filters) from the server

verify_url_signature("your-secret", url)  # True
verify_url_signature({"2026-10": "new-secret", "2026-04": "your-secret"}, url)  # True
```

`explain_url(url)` parses a URL, or a bare path, back into its signature, its explained segments and its source, and lists every error. Each segment carries its `raw` text, a `kind` (`expiry`, `maxage`, `debug`, `preset` or `operation`), a `title`, and a `detail` or an `error`; a well-formed `exp:` segment also says whether the URL had `expired` at `now_seconds` (the current time unless you pass one). `remainder` is the text the signature covers, and an empty `errors` is the server's verdict that the request parses. Its verdict is the server's: it reads each number as the server's types read it (a decimal as the 32-bit float the server reads, or the 64-bit one for a `convolution` value or a `frame:` selector, its range applied to the rounded value), and it reports what the server refuses before and after the parse: a remainder that begins with `crabcrop`, `purge`, `purge-group` or `upload`, more than 100 operations, and more than 16 distinct watermark and overlay URLs. It cannot see preset definitions or the server's default operations, so it counts each `preset:` reference as one operation. A class name's letters and digits are Unicode 17.0's, as the server's Rust reads them, whatever the Unicode version of the interpreter's `unicodedata`; the typed builders read them the same way. It reads the path as written, not through a URL parser, so dot segments stay literal and a query or fragment is cut off, as the server does. It reads at most 64 KiB (65,536 bytes of UTF-8), which bounds the time one call takes: a longer input gets a single error and no signature. A string that holds surrogate code points, as `surrogateescape` decoding makes for bytes that are not UTF-8, reads as JavaScript reads the same UTF-16: a surrogate pair is the character it encodes.

`verify_url_signature` reads the URL the same way and answers `True` or `False`, or `None` when the URL is unsigned (`unsafe`), sealed, malformed, or longer than 64 KiB. A remainder that begins with the reserved segment `crabcrop` is `False` whatever its tag. With one secret, the key ID a URL carries is ignored: the question is whether this secret made the tag. With a keyring, the server's rotation rules apply: a keyed URL (`<key-id>:<tag>`) consults its named key only, so an ID the keyring does not hold is `False`, and an unkeyed URL consults every key. Tags compare in constant time (`hmac.compare_digest`), as in `verify`. A lone surrogate, in the URL or in a secret, is hashed as U+FFFD, the bytes the TypeScript SDK's HMAC hashes for it, so no URL and no secret of text makes it raise.

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

With `expires`, `build_url`, `build_srcset` and `build_sealed_url` also raise `ExpiryOverrideError`, a `ValueError`, for operations that carry an `exp:` or `maxage:` segment, and for a source whose first segment the server reads as an operation, `exp:`, `maxage:` or `preset:`, such as `300x200/https://example.com/a.jpg` passed as the source. The server keeps the last `exp:` of a URL and reads everything before the source as operations, so such text, from page content say, would set the URL's lifetime whatever `expires` says, or sign operations nobody passed as operations. The error's `input` names the input and `segment` the segment. Without `expires` these inputs build as before.

## Tests

```bash
python3 -m unittest discover -s tests
```

`tests/test_explain_differential.py` runs this package's explainer and the TypeScript SDK's on 6,000 inputs generated from the grammar corpora (their remainders with segments and filters dropped, repeated and swapped, characters changed, and numbers at the edges of their ranges), asks the two verifiers about each input as generated and re-signed, under one secret and under keyrings, and fails on any difference in any field or answer. It runs the TypeScript SDK from its sources with Node 22.18 or newer (or 23.6 or newer), which needs no build or `npm install`, and is skipped when no such `node` is on `PATH`, unless `CRABCROP_DIFF_REQUIRE_NODE=1`, which `scripts/test_all.sh` sets: then it fails. The inputs follow from a seed, so a failure repeats; to run more of them, or another set:

```bash
CRABCROP_DIFF_CASES=100000 CRABCROP_DIFF_SEED=7 python3 -m unittest tests.test_explain_differential -v
```
