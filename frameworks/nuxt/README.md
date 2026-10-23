# crabcrop for Nuxt (@nuxt/image provider)

A provider module for `@nuxt/image` that routes images through your crabcrop deployment, built on [`@crabcrop/url`](../../sdks/typescript).

## Setup

1. Install the builder: `npm install @crabcrop/url`.
2. Copy [`crabcrop.ts`](crabcrop.ts) into your project, for example as `providers/crabcrop.ts`.
3. Register it in `nuxt.config`:

```ts
export default defineNuxtConfig({
  modules: ["@nuxt/image"],
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

Provider options: `base` (required), `preset` (recommended: the transform policy stays server-side and no secret is needed client-side; pair with the server's unsafe or presets-only mode), `secret` (optional, server-only contexts).

## Modifier mapping

| Modifiers | crabcrop operations |
|---|---|
| `width` + `height`, fit `cover` (default) | `crop:WxH:smart` |
| `width` + `height`, fit `contain` | `pad:WxH:blur` (blurred ambient bars) |
| `width` + `height`, fit `inside` / `outside` | `fit-in:WxH` |
| `width` + `height`, fit `fill` | `WxH` |
| `width` only / `height` only | `Wx0` / `0xH` |
| `format` (`jpg` becomes `jpeg`) | `filters:format(...)` |
| `quality` | `filters:quality(...)` |

With a preset configured, `preset:<name>` leads and the geometry and filters above append after it.

## Registry status

This provider is maintained in the crabcrop repository. Submission to the Nuxt Image provider registry is a separate step (a pull request to the Nuxt project) and has not been made yet; the module works today through the custom-provider registration above, which is the same mechanism built-in providers use.

## Tests

`npm test` in this directory pins the modifier mapping to exact signed URLs (signatures cross-checked with `node:crypto` directly). The tests map the `@crabcrop/url` import to the in-repo SDK through `test/register.mjs`; in a real project the installed package resolves it.
