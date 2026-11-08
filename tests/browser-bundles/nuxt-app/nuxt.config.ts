// The setup frameworks/nuxt/README.md documents, plus a
// sentinel secret set at build time: the test finds it in the
// server build (so its search works) and requires that it be
// absent from everything a browser loads.
export default defineNuxtConfig({
  modules: ["@nuxt/image"],
  compatibilityDate: "2025-07-15",
  devtools: { enabled: false },
  telemetry: false,
  runtimeConfig: {
    crabcropSecret: "browser-bundles-sentinel-5f2c9e",
  },
  hooks: {
    // Nuxt keeps generated templates in memory; write
    // @nuxt/image's options module (the one client code
    // imports) to .nuxt/ so the test can read it.
    "app:templates"(app) {
      for (const template of app.templates) {
        if (template.filename === "image-options.mjs") {
          template.write = true;
        }
      }
    },
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
