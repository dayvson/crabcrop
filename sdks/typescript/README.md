# @crabcrop/url

Build and sign [crabcrop](https://github.com/dayvson/crabcrop) URLs from TypeScript or JavaScript, on Node or in the browser. No dependencies: Node signs and seals with `node:crypto`, and browser bundles get a built-in TypeScript implementation of the same functions (see [Browsers and bundlers](#browsers-and-bundlers)). The signing scheme is specified in [`spec/url-signing.md`](../../spec/url-signing.md). Signing is pinned by the canonical [`spec/test-vectors.json`](../../spec/test-vectors.json), which this package runs as tests, and the explainer's verdicts and the typed builders are pinned by the shared grammar corpora in [`spec/explain-verdicts.json`](../../spec/explain-verdicts.json) and [`spec/operation-vectors.json`](../../spec/operation-vectors.json), which the playground's suite and the server crate's differential tests run as well.

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
const expiringUpload = uploadSignature("your-secret", "heroes/photo.jpg", 1893456000);

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

## Browsers and bundlers

The package imports its crypto through one internal specifier, `#crypto`, which `package.json` maps by condition:

| Condition | Backend | Who resolves it |
|---|---|---|
| `browser` | `dist/crypto/portable.js`: SHA-256, HMAC-SHA-256, and AES-256-GCM in plain TypeScript | client builds from bundlers that read `imports` conditions (Vite, webpack 5, esbuild, Rollup) |
| `node` | `dist/crypto/node.js`: `node:crypto` (OpenSSL) | Node, and server builds that target it |
| `default` | the portable backend | any other runtime, such as an edge worker without Node compatibility |

The rest of the package uses no Node globals (no `Buffer`, no `process`), so a client bundle never pulls in a Node built-in. Every function stays synchronous on both backends; WebCrypto's `crypto.subtle` is asynchronous and could not keep the API as it is. Both backends produce identical bytes: the test suite checks the portable one against `node:crypto` on thousands of random inputs and runs every spec vector on each, and [`tests/browser-bundles`](../../tests/browser-bundles) builds the framework integrations with Vite and runs the bundles without Node globals.

The portable backend is slower: about 5 µs per signature against about 2 µs on `node:crypto` (Node 22), and about a quarter of a millisecond to seal a 2 KB remainder, the largest allowed. Sealing needs `crypto.getRandomValues`, which every browser and edge runtime provides. Its AES rounds index the S-box by key-dependent bytes, which a cache-timing observer could in principle measure; Node keeps OpenSSL for that reason as well as speed.

A secret in browser code is public: anyone who loads the page can read it. The browser build exists so that client code can import the package at all: to build unsigned and preset URLs, explain URLs, and sign where the secret belongs to the person using the page (a playground-style tool). Sign with a deployment secret only in server code; the [framework integrations](../../frameworks) never sign in the browser.

## Runtime

The published package is compiled JavaScript (`dist/`) for Node >= 22. The sources are TypeScript and run directly on Node 22.18 or newer (or 23.6 or newer), where type stripping is on by default, which is also how the tests run:

```bash
npm test
```

`npm test` builds `dist/` first (the `#crypto` import resolves there), then runs the suite twice: on `node:crypto`, and again under `--conditions=browser` on the portable backend.

In this repository the framework loaders and the MCP server resolve `@crabcrop/url` to these sources, so build the package once (`npm run build`) before running their tests; a standalone install comes from npm once published, or from the repository with `"@crabcrop/url": "file:../typescript"` as the MCP package in this repo does.
