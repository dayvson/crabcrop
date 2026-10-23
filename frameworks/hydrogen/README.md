# crabcrop for Hydrogen

A loader for `@shopify/hydrogen`'s `Image` component, for images that do **not** live on Shopify's CDN. Shopify product media keeps its native CDN loader; external originals (editorial photography, UGC, a DAM) route through crabcrop with this loader. Built on [`@crabcrop/url`](../../sdks/typescript).

## Setup

1. Install the builder: `npm install @crabcrop/url` (or copy `sdks/typescript/src/index.ts` into your project and adjust the import).
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

## Configuration

`createCrabcropLoader({ base, preset?, secret?, expires? })`:

- `base` (required): the crabcrop origin.
- `preset`: a server-side preset name. Recommended when URLs are generated anywhere near the client: the transform policy stays on the server and no secret is involved. Pair with the server's unsafe or presets-only mode.
- `secret`: signs the URLs. Only pass it from server-side code.
- `expires`: a unix-seconds expiry applied to every URL.

The loader maps Hydrogen's `{src, width, height, scale}` calls onto the grammar: width and height together become `{w}x{h}` (with `scale` multiplied in, for device-pixel-ratio variants), width alone becomes `{w}x0`, height alone `0x{h}`. With a preset, `preset:<name>` leads and the sizing appends.

## Tests

`npm test` in this directory pins the factory's output to exact URLs (signatures cross-checked with `node:crypto` directly). The tests map the `@crabcrop/url` import to the in-repo SDK through `test/register.mjs`; in a real project the installed package resolves it.
