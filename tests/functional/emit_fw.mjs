// Framework parity emitter: runs the three framework loaders /
// provider with fixed inputs and prints {out, expected}. Expected
// values are computed through the TypeScript SDK's buildURL with the
// documented operation construction, then byte-compared by the driver.
//
// The driver passes the live instance base, secret and source through
// the environment: FUNC_FW_BASE, FUNC_FW_SECRET, FUNC_FW_SRC.
import "./fw_register.mjs";

const sdkSrc = new URL("../../sdks/typescript/src/index.ts", import.meta.url);
const { buildURL } = await import(sdkSrc);

const BASE = process.env.FUNC_FW_BASE;
const SECRET = process.env.FUNC_FW_SECRET;
const SRC = process.env.FUNC_FW_SRC;
const out = {};
const expected = {};

// --- Next.js loader: env is read at import time ---
process.env.NEXT_PUBLIC_CRABCROP_BASE = BASE;
delete process.env.NEXT_PUBLIC_CRABCROP_PRESET;
delete process.env.NEXT_PUBLIC_CRABCROP_ORIGIN;
process.env.CRABCROP_SECRET = SECRET;
const nextLoader = (await import(
  new URL("../../frameworks/nextjs/crabcrop-loader.ts", import.meta.url)
)).default;

out["next-secret-q"] = nextLoader({ src: SRC, width: 640, quality: 75 });
expected["next-secret-q"] = buildURL(BASE, SECRET, "640x0/filters:quality(75)", SRC);
out["next-secret-noq"] = nextLoader({ src: SRC, width: 320 });
expected["next-secret-noq"] = buildURL(BASE, SECRET, "320x0", SRC);

// --- Hydrogen loader: factory config ---
const { createCrabcropLoader } = await import(
  new URL("../../frameworks/hydrogen/crabcrop-loader.ts", import.meta.url)
);
const hydroPreset = createCrabcropLoader({ base: BASE, preset: "card" });
out["hydro-preset-wh"] = hydroPreset({ src: SRC, width: 800, height: 450 });
expected["hydro-preset-wh"] = buildURL(BASE, null, "preset:card/800x450", SRC);
const hydroSecret = createCrabcropLoader({ base: BASE, secret: SECRET });
out["hydro-secret-scale"] = hydroSecret({ src: SRC, width: 300, scale: 2 });
expected["hydro-secret-scale"] = buildURL(BASE, SECRET, "600x0", SRC);
const hydroExp = createCrabcropLoader({ base: BASE, secret: SECRET, expires: 1893456000 });
out["hydro-secret-exp"] = hydroExp({ src: SRC, width: 100 });
expected["hydro-secret-exp"] = buildURL(BASE, SECRET, "100x0", SRC, { expires: 1893456000 });

// --- Nuxt provider ---
const { getImage } = await import(
  new URL("../../frameworks/nuxt/crabcrop.ts", import.meta.url)
);
const nuxtOpts = { options: { base: BASE, secret: SECRET } };
const nuxt = (src, mods, opts) => getImage(src, { modifiers: mods }, opts).url;
out["nuxt-cover"] = nuxt(SRC, { width: 800, height: 450 }, nuxtOpts);
expected["nuxt-cover"] = buildURL(BASE, SECRET, "crop:800x450:smart", SRC);
out["nuxt-contain"] = nuxt(SRC, { width: 800, height: 450, fit: "contain" }, nuxtOpts);
expected["nuxt-contain"] = buildURL(BASE, SECRET, "pad:800x450:blur", SRC);
out["nuxt-inside"] = nuxt(SRC, { width: 800, height: 450, fit: "inside" }, nuxtOpts);
expected["nuxt-inside"] = buildURL(BASE, SECRET, "fit-in:800x450", SRC);
out["nuxt-fill-fmt"] = nuxt(SRC, { width: 800, height: 450, fit: "fill", format: "jpg", quality: 70 }, nuxtOpts);
expected["nuxt-fill-fmt"] = buildURL(BASE, SECRET, "800x450/filters:format(jpeg)/filters:quality(70)", SRC);
out["nuxt-preset-w"] = nuxt(SRC, { width: 500 }, { options: { base: BASE, preset: "thumb" } });
expected["nuxt-preset-w"] = buildURL(BASE, null, "preset:thumb/500x0", SRC);

console.log(JSON.stringify({ out, expected }));
