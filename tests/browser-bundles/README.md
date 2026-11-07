# Browser bundle tests

Real bundler builds of the client code the framework integrations ship, for issue #381. Each integration that runs in the browser is used as its README says (the file copied into a project that installs `@crabcrop/url`) and built for the client:

- `@crabcrop/url` itself, the [SvelteKit](../../frameworks/sveltekit), [Hydrogen](../../frameworks/hydrogen), and [Next.js](../../frameworks/nextjs) loaders: a Vite client build (library mode, one IIFE), which must log no Node built-in being externalized, then run in a `vm` sandbox with a browser's globals (`window`, `TextEncoder`, `TextDecoder`, `crypto.getRandomValues`) and none of Node's. The SDK bundle reproduces every signing and sealing vector in [`spec/test-vectors.json`](../../spec/test-vectors.json); the loaders build their unsigned browser URLs. Next.js bundles with webpack or Turbopack, not Vite; the Vite build stands in for any bundler that resolves the `browser` condition.
- The [Nuxt](../../frameworks/nuxt) provider: a full `nuxi build` of the minimal app in [`nuxt-app/`](nuxt-app) with `@nuxt/image`. The test requires that the secret set in `runtimeConfig` be absent from the generated `image-options.mjs` (the module client code imports) and from every file under `.output/public`, while present in the server build; it then starts the built server with `NUXT_CRABCROP_SECRET` and checks that every server-rendered image URL is signed with it.

The [Astro](../../frameworks/astro) service is not built here: Astro runs image services on the server only.

```bash
(cd ../../sdks/typescript && npm ci && npm run build)
npm ci
npm test
```

Builds go to `.work/`, which is ignored. [`scripts/test_all.sh`](../../scripts/test_all.sh) runs this suite after the framework modules.
