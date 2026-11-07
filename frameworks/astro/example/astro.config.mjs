// Minimal Astro configuration for the crabcrop service
// example. Run a local crabcrop with a preset named "card"
// first (CRABCROP_ALLOW_UNSAFE=1 plus a preset definition),
// then `npm run dev` in an Astro
// project with these files and the page in src/pages/.
// Unsafe mode alone is an open image proxy, so this is a
// local development setup: production sets `secret` and
// runs the server signed (see the README).
import { defineConfig } from "astro/config";

export default defineConfig({
  image: {
    service: {
      entrypoint: "./crabcrop-service.ts",
      config: {
        // The local crabcrop from the repository README's
        // quick start. The preset carries the whole
        // transform policy; these pages name no operations.
        base: "http://localhost:3000",
        preset: "card",
      },
    },
  },
});
