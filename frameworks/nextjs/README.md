# crabcrop for Next.js

A custom loader for `next/image` that routes every image through your crabcrop deployment. It is a thin wrapper over [`@crabcrop/url`](../../sdks/typescript): no signing code of its own.

## Setup

1. Install the builder: `npm install @crabcrop/url` (or copy `sdks/typescript/src/index.ts` into your project and adjust the import).
2. Copy [`crabcrop-loader.ts`](crabcrop-loader.ts) into your project root.
3. Point Next at it:

```js
// next.config.mjs
export default {
  images: { loader: "custom", loaderFile: "./crabcrop-loader.ts" },
};
```

4. Configure by environment:

| Variable | Meaning |
|---|---|
| `NEXT_PUBLIC_CRABCROP_BASE` | The crabcrop origin, e.g. `https://images.example.com`. Required. |
| `NEXT_PUBLIC_CRABCROP_PRESET` | A preset name defined on the server. Recommended (see below). |
| `NEXT_PUBLIC_CRABCROP_ORIGIN` | Origin prepended to relative `src` values such as `/uploads/a.jpg`. |
| `CRABCROP_SECRET` | Used only for loader calls during server rendering. Never exposed to the browser. |

## Presets are the recommended configuration

A custom loader also runs in the browser, where a secret must never ship. Define the transform policy as a server-side [preset](../../README.md#presets) (crop, format, quality) and set `NEXT_PUBLIC_CRABCROP_PRESET` to its name; run the server with `CRABCROP_ALLOW_UNSAFE=1` or `CRABCROP_ONLY_PRESETS=1`. The loader then emits `preset:<name>` plus the `{width}x0` sizing `next/image` asks for per srcset entry, so widths still vary while the policy stays on the server.

Without a preset, the loader emits `{width}x0` and maps the `quality` prop to `filters:quality(...)`; sign with `CRABCROP_SECRET` for server-rendered calls if your deployment requires signatures.

## Tests

`npm test` in this directory runs the loader against exact expected URLs (signatures cross-checked with `node:crypto` directly). The tests map the `@crabcrop/url` import to the in-repo SDK through `test/register.mjs`; in a real project the installed package resolves it.
