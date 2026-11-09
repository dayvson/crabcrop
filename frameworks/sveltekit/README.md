# crabcrop for SvelteKit

A loader factory for SvelteKit apps that routes image URLs through your crabcrop deployment. SvelteKit has no built-in image component, so the loader is a small object your app creates once and calls from components and load functions to build `src` and `srcset` values. It is a thin wrapper over [`@crabcrop/url`](../../sdks/typescript): no signing code of its own.

## Setup

1. Install the builder: `npm install @crabcrop/url` (0.10.0 or later, which exports the `MAX_DIMENSION` this file imports).
2. Copy [`crabcrop-loader.ts`](crabcrop-loader.ts) into your project, for example under `src/lib/`.
3. Create the loader in a shared module:

```ts
// src/lib/images.ts
import { createCrabcropLoader } from "./crabcrop-loader";

export const images = createCrabcropLoader({
  base: "https://images.example.com",
  preset: "card",
});
```

4. Call it from any component:

```svelte
<img
  src={images.url(photo.url, { width: 800 })}
  srcset={images.srcset(photo.url, [320, 640, 960])}
  sizes="(min-width: 48em) 33vw, 100vw"
  alt={photo.alt}
/>
```

This setup builds unsigned URLs; read [Signing and server modes](#signing-and-server-modes) before you deploy it.

## Configuration

`createCrabcropLoader({ base, preset?, secret?, expires?, origin? })`:

- `base` (required): the crabcrop origin.
- `preset`: a server-side preset name. Recommended: the transform policy stays on the server and the URLs carry only what varies per image. See [Signing and server modes](#signing-and-server-modes) for the server configuration it pairs with.
- `secret`: signs the URLs. Create the loader with it only in server-side code (a `$lib/server` module or a load function reading `$env/static/private`); the loader additionally forces the secret to null in a browser context, so a factory created with one still cannot sign client-side.
- `expires`: a unix-seconds expiry applied to every URL.
- `origin`: origin prepended to root-relative `src` values such as `/uploads/a.jpg`, which crabcrop cannot fetch on their own.

## URL mapping

| Call | crabcrop operations |
|---|---|
| `url(src, { width, height })` | `{width}x{height}` |
| `url(src, { width })` / `url(src, { height })` | `{width}x0` / `0x{height}` |
| `format` (`jpg` becomes `jpeg`) | `filters:format(...)` |
| `quality` | `filters:quality(...)` |
| `srcset(src, widths)` | one `{width}x0` URL per width, as `{url} {width}w` entries |

With a preset configured, `preset:<name>` leads and the geometry and filters above append after it. A `srcset` varies the width alone, carrying any `format` and `quality` options through to every entry. Unsupported formats and out-of-range qualities throw where the mistake is made, instead of minting a URL the server would reject.

Dimensions follow one rule: a fraction rounds to whole pixels, a value that is not a size (0, negative, or not a number) is left out so the URL serves the source's own size, and a value past the server's per-axis cap (`MAX_DIMENSION` in `@crabcrop/url`) throws, as does `createCrabcropLoader` without a `base`, instead of minting a URL that cannot load. All five crabcrop integrations share this rule. A `srcset` width is also spelled into its `w` descriptor, so it must be a positive whole number no larger than the cap, or the call throws.

The [`example/`](example) directory is a minimal loader module and page that render an image through a local crabcrop using only a preset name: the page names no operations at all. It builds unsigned URLs for a local server in unsafe mode, a development setup (see below).

## Signing and server modes

A loader created without a secret, or running in the browser, builds unsigned URLs. That decides which server mode fits:

- **Signed (production).** Run the server without `CRABCROP_ALLOW_UNSAFE`. Create the loader with `secret` in server code (a `$lib/server` module, or a `+page.server` / `+layout.server` load function reading `$env/static/private`), build `src` and `srcset` there, and return the strings to the page. A loader running in the browser cannot sign, and a signed server answers its URLs with `403`.
- **Presets-only (no secret anywhere).** `CRABCROP_ONLY_PRESETS=1` with `CRABCROP_ALLOW_UNSAFE=1` accepts URLs that carry a preset reference alone. Call `url(src)` with no options and put the size in the preset (`card-800 = 800x0/filters:format(webp)`): sized calls, `format`, `quality`, and `srcset` append operations, which this mode rejects with `400`. A responsive srcset then needs one preset per width.
- **Unsafe alone (local development only).** With `CRABCROP_ALLOW_UNSAFE=1` and presets-only off, every loader call works as is, but the server then honors any unsigned operation chain on any public source: the preset constrains nothing, because a client can drop it or append anything after it. That is an open image proxy and CPU sink. Use it to try the loader locally, never as the public deployment.

[Presets](../../docs/operations.md#presets) compares the three modes.

## Tests

`npm test` in this directory runs the loader against exact expected URLs; the signed cases reproduce the canonical spec test vectors byte for byte, and the remaining signatures are cross-checked with `node:crypto` directly. It also runs the dimension table all five integrations share ([`dimension-table.mjs`](../dimension-table.mjs)), so one mistake has one outcome in each. The tests map the `@crabcrop/url` import to the in-repo SDK through `test/register.mjs` (build the SDK first: `npm run build` in `sdks/typescript`); in a real project the installed package resolves it.

[`tests/browser-bundles`](../../tests/browser-bundles) builds this loader into a browser bundle with Vite and runs it with a browser's globals and none of Node's.
