# crabcrop for SvelteKit

A loader factory for SvelteKit apps that routes image URLs through your crabcrop deployment. SvelteKit has no built-in image component, so the loader is a small object your app creates once and calls from components and load functions to build `src` and `srcset` values. It is a thin wrapper over [`@crabcrop/url`](../../sdks/typescript): no signing code of its own.

## Setup

1. Install the builder: `npm install @crabcrop/url` (or copy `sdks/typescript/src/index.ts` into your project and adjust the import).
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

## Configuration

`createCrabcropLoader({ base, preset?, secret?, expires?, origin? })`:

- `base` (required): the crabcrop origin.
- `preset`: a server-side preset name. Recommended when URLs are generated anywhere near the client: the transform policy stays on the server and no secret is involved. Pair with the server running `CRABCROP_ALLOW_UNSAFE=1`. (`CRABCROP_ONLY_PRESETS=1` goes further and rejects the sizing appended after the preset; it fits URLs that carry a preset reference alone.)
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

With a preset configured, `preset:<name>` leads and the geometry and filters above append after it. A `srcset` varies the width alone, carrying any `format` and `quality` options through to every entry. Dimensions past the server's 16383 cap, unsupported formats, and out-of-range qualities throw where the mistake is made, instead of minting a URL the server would reject.

The [`example/`](example) directory is a minimal loader module and page that render an image through a local crabcrop using only a preset name: the page names no operations at all.

## Tests

`npm test` in this directory runs the loader against exact expected URLs; the signed cases reproduce the canonical spec test vectors byte for byte, and the remaining signatures are cross-checked with `node:crypto` directly. The tests map the `@crabcrop/url` import to the in-repo SDK through `test/register.mjs`; in a real project the installed package resolves it.
