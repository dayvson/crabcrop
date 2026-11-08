# crabcrop for Nuxt (@nuxt/image provider)

A provider module for `@nuxt/image` that routes images through your crabcrop deployment, built on [`@crabcrop/url`](../../sdks/typescript).

## Setup

1. Install the builder: `npm install @crabcrop/url` (with `@nuxt/image` 1.8 or later, on Nuxt 3 or 4).
2. Copy [`crabcrop.ts`](crabcrop.ts) into your project, for example as `providers/crabcrop.ts`.
3. Register it in `nuxt.config`:

```ts
export default defineNuxtConfig({
  modules: ["@nuxt/image"],
  runtimeConfig: {
    // Server-only. Leave it empty here and set NUXT_CRABCROP_SECRET
    // in the server's environment; omit it to build unsigned URLs.
    crabcropSecret: "",
  },
  image: {
    providers: {
      crabcrop: {
        provider: "~/providers/crabcrop",
        options: {
          base: "https://images.example.com",
          preset: "card",
        },
      },
    },
    provider: "crabcrop",
  },
});
```

Provider options: `base` (required) and `preset` (recommended: the transform policy stays on the server).

The secret is not a provider option. `@nuxt/image` writes provider options verbatim into the generated `#build/image-options.mjs`, which the client bundle imports, so anything placed there ships to every browser. The provider reads the secret from private runtime config instead (`runtimeConfig.crabcropSecret`, set at runtime with `NUXT_CRABCROP_SECRET`): Nuxt sends the browser only the `public` part of runtime config, and the provider signs only when it runs on the server. If it finds a `secret` provider option it throws. An earlier version of this file accepted one, so if you set it there, it has been in your client bundle: rotate it.

The provider builds unsigned URLs whenever it runs in the browser; read [Signing and server modes](#signing-and-server-modes) before you deploy it.

## Signing and server modes

`@nuxt/image` calls the provider during server rendering and again in the browser (client-side navigation and re-renders), and every sized image appends its geometry after the preset (one width per srcset entry with `sizes`). The browser calls cannot sign. That decides which server mode fits:

- **Signed (production).** Run the server without `CRABCROP_ALLOW_UNSAFE`. With `crabcropSecret` set, the provider signs its server-rendering calls, and the server-side `useImage()` in Nitro routes signs too, but the browser calls produce unsigned URLs, which a signed server answers with `403`. So for pages the browser may re-render, build the URLs in server-only code (a Nitro route under `server/`, reading the secret from private runtime config) with `buildURL` from `@crabcrop/url`, one call per srcset width with `preset:<name>/<width>x0` so the preset still applies (`buildSrcSet` emits bare `{width}x0` entries, without it), and render them with a plain `<img>`.
- **Presets-only (no secret anywhere).** `CRABCROP_ONLY_PRESETS=1` with `CRABCROP_ALLOW_UNSAFE=1` accepts URLs that carry a preset reference alone and rejects the geometry and filters the provider appends with `400`. It fits only images rendered without size, format, or quality modifiers, with the size inside the preset (`card-640 = 640x0/filters:format(webp)`).
- **Unsafe alone (local development only).** With `CRABCROP_ALLOW_UNSAFE=1` and presets-only off, the provider works as is, but the server then honors any unsigned operation chain on any public source: the preset constrains nothing, because a client can drop it or append anything after it. That is an open image proxy and CPU sink. Use it to try the provider locally, never as the public deployment.

[Presets](../../docs/operations.md#presets) compares the three modes.

## Modifier mapping

| Modifiers | crabcrop operations |
|---|---|
| `width` + `height`, fit `cover` (default) | `crop:WxH:smart` |
| `width` + `height`, fit `contain` | `pad:WxH:blur` (blurred ambient bars) |
| `width` + `height`, fit `inside` | `fit-in:WxH` |
| `width` + `height`, fit `outside` | `crop:WxH` (approximation, see below) |
| `width` + `height`, fit `fill` | `WxH` |
| `width` only / `height` only | `Wx0` / `0xH` |
| `format` (`jpg` becomes `jpeg`) | `filters:format(...)` |
| `quality` | `filters:quality(...)` |

With a preset configured, `preset:<name>` leads and the geometry and filters above append after it.

About `outside`: in @nuxt/image semantics, `outside` means the result covers the box (at least the box on both axes, aspect preserved) without cropping. crabcrop has no uncropped cover resize, so the provider maps `outside` to a centered `crop:WxH`: both requested dimensions are honored exactly, but the overflow is cropped away instead of kept. (The earlier mapping to `fit-in:WxH` was the opposite geometry: its result could be smaller than the box on one axis.) If keeping the whole frame matters more than the exact size, use `fit: "inside"` and let your layout clip the overflow.

## Registry status

This provider is maintained in the crabcrop repository. Submission to the Nuxt Image provider registry is a separate step (a pull request to the Nuxt project) and has not been made yet; the module works today through the custom-provider registration above, which is the same mechanism built-in providers use.

## Tests

`npm test` in this directory pins the modifier mapping to exact signed URLs (signatures cross-checked with `node:crypto` directly), the refusal of a `secret` provider option, and the browser-side guard. The tests map the `@crabcrop/url` import to the in-repo SDK through `test/register.mjs` (build the SDK first: `npm run build` in `sdks/typescript`); in a real project the installed package resolves it.

[`tests/browser-bundles`](../../tests/browser-bundles) builds a real Nuxt app with `@nuxt/image` and this provider: it checks that the generated `image-options.mjs` and every file under `.output/public` are free of the secret, that the client build pulls in no Node built-in, and that server rendering signs with the secret from `NUXT_CRABCROP_SECRET`.
