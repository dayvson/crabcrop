# @crabcrop/url

Build and sign [crabcrop](https://github.com/dayvson/crabcrop) URLs from TypeScript or JavaScript. Standard library only (`node:crypto`); the signing scheme is specified in [`spec/url-signing.md`](../../spec/url-signing.md). Signing is pinned by the canonical [`spec/test-vectors.json`](../../spec/test-vectors.json), which this package runs as tests, and the explainer's verdicts are pinned by a grammar corpus mirrored in the playground's test suite.

## Usage

```ts
import { buildSealedURL, buildURL, buildSrcSet, purgeGroupSignature, purgeSignature, seal, sign, unseal, uploadSignature, verifyUrlSignature } from "@crabcrop/url";

// A signed URL. The operation string is passed verbatim and may
// contain several segments and preset: references.
const url = buildURL(
  "https://images.example.com",
  "your-secret",
  "crop:1200x630:smart/filters:format(webp)",
  "https://example.com/photo.jpg",
);

// An expiring URL (unix seconds), emitted as a leading exp: segment.
const expiring = buildURL(
  "https://images.example.com",
  "your-secret",
  "300x200",
  "https://example.com/photo.jpg",
  { expires: 1893456000 },
);

// A responsive srcset: one {width}x0 URL per width.
const srcset = buildSrcSet(
  "https://images.example.com",
  "your-secret",
  "https://example.com/photo.jpg",
  [400, 800, 1200],
);

// The raw signature for a remainder, if you assemble URLs yourself.
const signature = sign("your-secret", "300x200/https://example.com/photo.jpg");

// Verify a signed URL's signature locally, in constant time
// (unsigned or malformed URLs return null).
const ok = verifyUrlSignature("your-secret", url);

// Purge signatures for the server's /purge and /purge-group endpoints
// are domain-separated under a derived subkey; the helpers sign the
// right strings so callers never assemble them by hand.
const purge = purgeSignature("your-secret", "300x200/https://example.com/photo.jpg");
const group = purgeGroupSignature("your-secret", "thumb");

// Upload signatures for the server's PUT /upload/<signature>/<name>
// endpoint live in a third domain, under their own derived subkey.
// Pass an expiry (unix seconds) for a token that stops working
// then; its path is /upload/<signature>/exp:<expiry>/<name>.
const upload = uploadSignature("your-secret", "heroes/photo.jpg");
const expiring = uploadSignature("your-secret", "heroes/photo.jpg", 1893456000);

// Sealed URLs (issue #255): the whole remainder encrypted into an
// opaque token in the signature slot, for public links whose recipe
// should stay unreadable. The server serves the sealed and plain
// forms of one render from the same cache entry.
const sealedUrl = buildSealedURL(
  "https://images.example.com",
  "your-secret",
  "300x200",
  "https://example.com/photo.jpg",
  { name: "photo.jpg" },
);
const remainder = unseal("your-secret", seal("your-secret", "300x200/https://example.com/photo.jpg"));
```

Pass `null` as the secret to produce the literal `unsafe` signature segment, for servers running with `CRABCROP_ALLOW_UNSAFE` or in presets-only mode.

## Typed operations

The operation string is the spec, and passing it verbatim stays the primary API. For callers who would rather build it, `ops` and `filter` are typed builders that validate each argument against the server's grammar and return the canonical spelling, so the typed and hand-written forms sign byte-identical URLs (the test suite pins this against the canonical corpus):

```ts
import { buildURL, filter, joinOps, ops } from "@crabcrop/url";

const operations = joinOps(
  ops.crop(1200, 630, "smart"),
  ops.filters(filter.format("webp"), filter.quality(80)),
);
// operations === "crop:1200x630:smart/filters:format(webp):quality(80)"
const url = buildURL("https://images.example.com", "your-secret", operations, "https://example.com/photo.jpg");
```

A value the server would reject (`filter.quality(0)`, `filter.brighten(256)`, `ops.crop(0, 100)`) throws at build time instead of minting a URL that fails at the edge. The closed sets are exported as types (`ImageFormat`, `CropAlign`, `Anchor`, `BlendMode`, `MaskShape`, `Dimension`), and `FILTER_DESCRIPTIONS` is the filter metadata table the explainer uses.

## Runtime

The package is TypeScript and runs directly on Node >= 22 (type stripping), which is also how the tests run:

```bash
node --test
```

In this repository the framework loaders and the MCP server resolve `@crabcrop/url` to these sources; a standalone install comes from npm once published, or from the repository with `"@crabcrop/url": "file:../typescript"` as the MCP package in this repo does.
