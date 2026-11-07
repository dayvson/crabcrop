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

The loader builds unsigned URLs whenever it runs in the browser; read [Signing and server modes](#signing-and-server-modes) before you deploy it.

## Presets keep the policy on the server

Define the transform policy as a server-side [preset](../../docs/operations.md#presets) (crop, format, quality) and set `NEXT_PUBLIC_CRABCROP_PRESET` to its name. The loader then emits `preset:<name>` plus the `{width}x0` sizing `next/image` asks for per srcset entry, so widths still vary while the policy stays on the server. Without a preset, the loader emits `{width}x0` and maps the `quality` prop to `filters:quality(...)`.

## Signing and server modes

`next/image` calls the loader once per srcset width, during server rendering and again in the browser (client-side renders and navigations). A secret must never ship to the browser, so the loader signs only its server-rendering calls, with `CRABCROP_SECRET`; its browser calls produce unsigned URLs. That decides which server mode fits:

- **Signed (production).** Run the server without `CRABCROP_ALLOW_UNSAFE`, and build the URLs where the secret lives: in Server Components, Route Handlers, or `getServerSideProps` / `getStaticProps`, with `CRABCROP_SECRET` from server-only environment. Called there, the loader function itself signs and keeps the preset (one call per srcset width); `buildURL` from `@crabcrop/url` with `preset:<name>/<width>x0` does the same. (`buildSrcSet` emits bare `{width}x0` entries, so it drops the preset.) Render them with a plain `<img>`, or with `next/image` and `unoptimized`, so the browser never rebuilds them. Do not rely on the custom loader against a signed server: its browser calls answer `403`.
- **Presets-only (no secret anywhere).** `CRABCROP_ONLY_PRESETS=1` with `CRABCROP_ALLOW_UNSAFE=1` accepts URLs that carry a preset reference alone and rejects any appended operation with `400`. The loader always appends `{width}x0`, so it cannot target this mode. Put the size in the preset instead (`card-640 = 640x0/filters:format(webp):quality(80)`) and reference it from a plain `<img>` (`/unsafe/preset:card-640/<source>`).
- **Unsafe alone (local development only).** With `CRABCROP_ALLOW_UNSAFE=1` and presets-only off, the loader works as is, but the server then honors any unsigned operation chain on any public source: the preset constrains nothing, because a client can drop it or append anything after it. That is an open image proxy and CPU sink. Use it to try the loader locally, never as the public deployment.

[Presets](../../docs/operations.md#presets) compares the three modes.

## Tests

`npm test` in this directory runs the loader against exact expected URLs (signatures cross-checked with `node:crypto` directly). The tests map the `@crabcrop/url` import to the in-repo SDK through `test/register.mjs`; in a real project the installed package resolves it.
