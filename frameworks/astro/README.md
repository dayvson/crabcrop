# crabcrop for Astro

An [Astro image service](https://docs.astro.build/en/reference/image-service-reference/) that routes `astro:assets` images through your crabcrop deployment: Astro calls the service's hooks at build time and during on-demand rendering, and the emitted `<img>` points straight at crabcrop, which does the transforming. It is a thin wrapper over [`@crabcrop/url`](../../sdks/typescript): no signing code of its own.

## Setup

1. Install the builder: `npm install @crabcrop/url` (or copy `index.ts`, `operations.ts`, and `explain.ts` from `sdks/typescript/src/` into your project, since `index.ts` imports the other two, and adjust the import).
2. Copy [`crabcrop-service.ts`](crabcrop-service.ts) into your project.
3. Register it as the image service in `astro.config`:

```js
// astro.config.mjs
import { defineConfig } from "astro/config";

export default defineConfig({
  image: {
    service: {
      entrypoint: "./crabcrop-service.ts",
      config: {
        base: "https://images.example.com",
        preset: "card",
        secret: process.env.CRABCROP_SECRET,
      },
    },
  },
});
```

4. Use Astro's components as usual; the service turns their props into crabcrop URLs:

```astro
---
import { Image } from "astro:assets";
---

<Image src="https://cdn.example.com/photo.jpg" width={800} height={600} alt="..." />
```

## Configuration

Service config (under `image.service.config`):

| Key | Meaning |
|---|---|
| `base` | The crabcrop origin, e.g. `https://images.example.com`. Required. |
| `preset` | A preset name defined on the server. Recommended (see below). |
| `origin` | Origin prepended to root-relative sources such as `/_astro/hero.jpg` (local imports) or `/uploads/a.jpg`. |
| `secret` | Signs the URLs. Safe here in a way a client loader cannot offer: the service runs only on the server (build and SSR), so the secret never enters client code. Read it from the environment in `astro.config` (`process.env.CRABCROP_SECRET`), never hardcode it. |
| `expires` | A unix-seconds expiry applied to every generated URL. |

## Sign in the service, keep the policy in a preset

The service runs only on the server, at build time and during on-demand rendering, so it can sign every URL: set `secret` from the environment, as in the setup above, and run the server signed, without `CRABCROP_ALLOW_UNSAFE`. A server-side [preset](../../docs/operations.md#presets) is the recommended place for the transform policy (crop, format, quality): name it in the service config, and the service emits `preset:<name>` plus the geometry Astro asks for, so widths still vary per `srcset` entry while the policy stays on the server.

The two unsigned modes do not fit this service in production. Presets-only mode (`CRABCROP_ONLY_PRESETS=1` with `CRABCROP_ALLOW_UNSAFE=1`) accepts URLs that carry a preset reference alone, and rejects the geometry and filters this service appends after the preset with `400`. Unsafe mode alone (`CRABCROP_ALLOW_UNSAFE=1` without presets-only) accepts the unsigned URLs, but the server then honors any unsigned operation chain on any public source: the preset constrains nothing, because a client can drop it or append anything after it. That is an open image proxy and CPU sink; use it only for local development, as the example below does.

## Transform mapping

| Astro props | crabcrop operations |
|---|---|
| `width` + `height` | `{width}x{height}` |
| `width` only / `height` only | `{width}x0` / `0x{height}` |
| `format` (`jpg` becomes `jpeg`) | `filters:format(...)` |
| `quality` | `filters:quality(...)` |

With a preset configured, `preset:<name>` leads and the geometry and filters above append after it.

Astro also accepts named quality presets; `low`, `mid`, `high`, and `max` map to 25, 50, 80, 100, the values Astro's own services use. Formats outside the server's grammar throw rather than minting a URL the server would reject; that includes `svg`, which needs no raster pipeline: use a plain `<img>` for SVGs.

The `getSrcSet` hook maps the `widths` prop to one transform per width (a requested height scales proportionally, keeping the box's aspect) with `w` descriptors, and the `densities` prop to width-multiplied transforms with `x` descriptors. The `getHTMLAttributes` hook keeps Astro's contract: transform fields are stripped, a missing dimension is filled from a local import's aspect ratio, and `loading`/`decoding` default to `lazy`/`async`.

The [`example/`](example) directory is a minimal page and config that render an image through a local crabcrop using only a preset name: the page names no operations at all. It leaves `secret` out and targets a local server in unsafe mode, a development setup.

## Tests

`npm test` in this directory runs the service hooks against exact expected URLs; the signed cases reproduce the canonical spec test vectors byte for byte, and the remaining signatures are cross-checked with `node:crypto` directly. The tests map the `@crabcrop/url` import to the in-repo SDK through `test/register.mjs`; in a real project the installed package resolves it.
