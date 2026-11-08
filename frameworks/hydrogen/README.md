# crabcrop for Hydrogen

A loader for `@shopify/hydrogen`'s `Image` component, for images that do **not** live on Shopify's CDN. Shopify product media keeps its native CDN loader; external originals (editorial photography, UGC, a DAM) route through crabcrop with this loader. Built on [`@crabcrop/url`](../../sdks/typescript).

## Setup

1. Install the builder: `npm install @crabcrop/url` (0.9.12 or later, which exports the `MAX_DIMENSION` this file imports).
2. Copy [`crabcrop-loader.ts`](crabcrop-loader.ts) into your project.
3. Create the loader where you have configuration access (a route loader or a shared module fed from `context.env`), and pass it to `Image`:

```tsx
import { Image } from "@shopify/hydrogen";
import { createCrabcropLoader } from "./crabcrop-loader";

const loader = createCrabcropLoader({
  base: "https://images.example.com",
  preset: "card",
});

<Image
  loader={loader}
  data={{ url: externalImageUrl, altText: "Editorial photo" }}
  widths={[320, 640, 960]}
  sizes="(min-width: 48em) 33vw, 100vw"
/>
```

This setup builds unsigned URLs; read [Signing and server modes](#signing-and-server-modes) before you deploy it.

## Configuration

`createCrabcropLoader({ base, preset?, secret?, expires? })`:

- `base` (required): the crabcrop origin.
- `preset`: a server-side preset name. Recommended: the transform policy stays on the server. See [Signing and server modes](#signing-and-server-modes) for the server configuration it pairs with.
- `secret`: signs the URLs. Only pass it from server-side code.
- `expires`: a unix-seconds expiry applied to every URL.

The loader maps Hydrogen's `{src, width, height, scale}` calls onto the grammar: width and height together become `{w}x{h}` (with `scale` multiplied in, for device-pixel-ratio variants), width alone becomes `{w}x0`, height alone `0x{h}`. With a preset, `preset:<name>` leads and the sizing appends.

Dimensions are multiplied by `scale` first, then follow one rule: a fraction rounds to whole pixels, a value that is not a size (0, negative, or not a number) is left out so the URL serves the source's own size, and a value past the server's per-axis cap (`MAX_DIMENSION` in `@crabcrop/url`) throws, as does `createCrabcropLoader` without a `base`, instead of minting a URL that cannot load. All five crabcrop integrations share this rule.

## Signing and server modes

The setup above builds unsigned URLs. Hydrogen's `Image` calls the loader once per width, during server rendering and again in the browser, and the browser calls cannot sign. That decides which server mode fits:

- **Signed (production).** Run the server without `CRABCROP_ALLOW_UNSAFE`. Build the URLs in a route `loader` (server-only) with a loader created from `context.env.CRABCROP_SECRET`, return the `src` and `srcset` strings in the loader data, and render them with a plain `<img>`. Passing the loader to `Image` makes the browser rebuild the URLs unsigned, which a signed server answers with `403`.
- **Presets-only (no secret anywhere).** `CRABCROP_ONLY_PRESETS=1` with `CRABCROP_ALLOW_UNSAFE=1` accepts URLs that carry a preset reference alone and rejects the sizing this loader appends with `400`, so `Image` with this loader cannot target it. A preset that carries the size (`card-640 = 640x0/filters:format(webp)`) works from a plain `<img>`.
- **Unsafe alone (local development only).** With `CRABCROP_ALLOW_UNSAFE=1` and presets-only off, the setup above works as is, but the server then honors any unsigned operation chain on any public source: the preset constrains nothing, because a client can drop it or append anything after it. That is an open image proxy and CPU sink. Use it to try the loader locally, never as the public deployment.

[Presets](../../docs/operations.md#presets) compares the three modes.

## Tests

`npm test` in this directory pins the factory's output to exact URLs (signatures cross-checked with `node:crypto` directly). It also runs the dimension table all five integrations share ([`dimension-table.mjs`](../dimension-table.mjs)), so one mistake has one outcome in each. The tests map the `@crabcrop/url` import to the in-repo SDK through `test/register.mjs` (build the SDK first: `npm run build` in `sdks/typescript`); in a real project the installed package resolves it.

[`tests/browser-bundles`](../../tests/browser-bundles) builds this loader into a browser bundle with Vite and runs it with a browser's globals and none of Node's.
