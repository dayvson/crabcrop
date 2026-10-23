# @crabcrop/url

Build and sign [crabcrop](https://github.com/dayvson/crabcrop) URLs from TypeScript or JavaScript. Standard library only (`node:crypto`); the signing scheme is specified in [`spec/url-signing.md`](../../spec/url-signing.md). Signing is pinned by the canonical [`spec/test-vectors.json`](../../spec/test-vectors.json), which this package runs as tests, and the explainer's verdicts are pinned by a grammar corpus mirrored in the playground's test suite.

## Usage

```ts
import { buildURL, buildSrcSet, sign } from "@crabcrop/url";

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
```

Pass `null` as the secret to produce the literal `unsafe` signature segment, for servers running with `CRABCROP_ALLOW_UNSAFE` or in presets-only mode.

## Runtime

The package is TypeScript and runs directly on Node >= 22 (type stripping), which is also how the tests run:

```bash
node --test
```

In this repository the framework loaders and the MCP server resolve `@crabcrop/url` to these sources; a standalone install comes from npm once published, or from the repository with `"@crabcrop/url": "file:../typescript"` as the MCP package in this repo does.
