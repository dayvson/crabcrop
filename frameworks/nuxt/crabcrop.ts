/**
 * A crabcrop provider for @nuxt/image (1.8 or later, Nuxt 3
 * or 4).
 *
 * Copy this file into your project (for example as
 * `providers/crabcrop.ts`) and register it in nuxt.config:
 *
 *   export default defineNuxtConfig({
 *     runtimeConfig: {
 *       // Server-only: set by NUXT_CRABCROP_SECRET at runtime.
 *       crabcropSecret: "",
 *     },
 *     image: {
 *       providers: {
 *         crabcrop: {
 *           provider: "~/providers/crabcrop",
 *           options: {
 *             base: "https://images.example.com",
 *             preset: "card",
 *           },
 *         },
 *       },
 *       provider: "crabcrop",
 *     },
 *   });
 *
 * Provider options: `base` (required) and `preset`
 * (recommended: the transform policy stays server-side).
 * @nuxt/image compiles provider options verbatim into
 * `#build/image-options.mjs`, which the client bundle
 * imports, so the secret is never a provider option: the
 * provider throws if it finds one. It reads the secret from
 * private runtime config (`crabcropSecret`), which Nuxt keeps
 * out of the client payload, and signs only when it runs on
 * the server; in the browser it builds unsigned URLs.
 * Unsigned URLs with appended geometry need the server's
 * unsafe mode without presets-only, an open image proxy fit
 * for local development only; production builds signed URLs
 * in server code (see the README's server modes).
 *
 * Modifier mapping:
 *   width + height, fit cover (default) -> crop:WxH:smart
 *   width + height, fit contain         -> pad:WxH:blur
 *   width + height, fit inside          -> fit-in:WxH
 *   width + height, fit outside         -> crop:WxH (see below)
 *   width + height, fit fill            -> WxH
 *   width only -> Wx0 ; height only -> 0xH
 *   format (jpg -> jpeg)                -> filters:format(...)
 *   quality                             -> filters:quality(...)
 */
import { buildURL } from "@crabcrop/url";

export interface CrabcropProviderOptions {
  base: string;
  preset?: string;
}

/** The part of @nuxt/image's context this provider reads. */
interface ImageContext {
  options?: { runtimeConfig?: { crabcropSecret?: unknown } };
}

interface Modifiers {
  width?: number | string;
  height?: number | string;
  fit?: string;
  format?: string;
  quality?: number | string;
  [key: string]: unknown;
}

/** The server's per-axis dimension cap (the URL grammar's MAX_DIMENSION). */
const MAX_DIMENSION = 16383;

const dim = (v: number | string | undefined, name: string): number | undefined => {
  const n = v === undefined ? NaN : Number(v);
  if (!Number.isFinite(n) || n <= 0) return undefined;
  const rounded = Math.round(n);
  // An over-cap dimension is a certain dead URL (the
  // server rejects the size at parse): throw here, where
  // the mistake is visible, instead of signing it.
  if (rounded > MAX_DIMENSION) {
    throw new Error(
      `crabcrop provider: ${name} ${rounded} exceeds the maximum dimension ${MAX_DIMENSION}`,
    );
  }
  return rounded;
};

/** The output formats the server's grammar accepts. */
const FORMATS = new Set(["jpeg", "png", "webp", "avif", "jxl", "gif"]);

/**
 * Quality is an integer 1-100 in the grammar (0 has no
 * meaning distinct from the minimum for any encoder, so
 * the server rejects it at parse). It is not a dimension,
 * so it gets its own validation instead of riding dim():
 * a present but invalid value throws rather than minting
 * filters:quality(undefined) or quality(150).
 */
const quality = (v: number | string): number => {
  const n = Number(v);
  if (!Number.isInteger(n) || n < 1 || n > 100) {
    throw new Error(
      `crabcrop provider: quality must be an integer between 1 and 100, got ${v}`,
    );
  }
  return n;
};

/**
 * The signing secret, on the server only. Private runtime
 * config never reaches the browser (Nuxt sends the client
 * only its `public` part), and the window check keeps a
 * client call unsigned even if a secret were somehow there.
 */
const serverSecret = (ctx?: ImageContext): string | null => {
  if (typeof window !== "undefined") return null;
  const secret = ctx?.options?.runtimeConfig?.crabcropSecret;
  return typeof secret === "string" && secret !== "" ? secret : null;
};

/**
 * @nuxt/image calls this with the provider options merged
 * into the second argument beside the modifiers, and its own
 * context (global options and runtime config) as the third.
 */
export const getImage = (
  src: string,
  { modifiers = {}, ...options }: { modifiers?: Modifiers } & Partial<CrabcropProviderOptions>,
  ctx?: ImageContext,
) => {
  if ((options as { secret?: unknown }).secret != null) {
    throw new Error(
      "crabcrop provider: `secret` is not a provider option: @nuxt/image compiles provider " +
        "options into the client bundle, so it has shipped to browsers. Rotate it, and set it " +
        "in private runtime config instead (runtimeConfig.crabcropSecret, NUXT_CRABCROP_SECRET)",
    );
  }
  if (!options.base) {
    throw new Error("crabcrop provider: the `base` option is required");
  }
  const width = dim(modifiers.width, "width");
  const height = dim(modifiers.height, "height");
  const operations: string[] = [];
  if (options.preset) {
    operations.push(`preset:${options.preset}`);
  }
  if (width && height) {
    switch (modifiers.fit) {
      case "contain":
        operations.push(`pad:${width}x${height}:blur`);
        break;
      case "inside":
        operations.push(`fit-in:${width}x${height}`);
        break;
      case "outside":
        // Nuxt's outside promises the result COVERS the box
        // (>= the box on both axes) without cropping.
        // crabcrop has no uncropped cover resize, and the
        // old mapping to fit-in was the opposite geometry
        // (result <= box). The honest approximation is a
        // centered crop to exactly WxH: both dimensions are
        // honored, at the price of the overflow the cover
        // promise would have kept (issue #153, documented
        // in the provider README).
        operations.push(`crop:${width}x${height}`);
        break;
      case "fill":
        operations.push(`${width}x${height}`);
        break;
      default:
        operations.push(`crop:${width}x${height}:smart`);
    }
  } else if (width) {
    operations.push(`${width}x0`);
  } else if (height) {
    operations.push(`0x${height}`);
  }
  if (modifiers.format) {
    const format = modifiers.format === "jpg" ? "jpeg" : modifiers.format;
    if (!FORMATS.has(format)) {
      throw new Error(
        `crabcrop provider: unsupported format ${modifiers.format} (expected one of ${[...FORMATS].join(", ")}, or jpg)`,
      );
    }
    operations.push(`filters:format(${format})`);
  }
  if (modifiers.quality !== undefined) {
    operations.push(`filters:quality(${quality(modifiers.quality)})`);
  }
  return { url: buildURL(options.base, serverSecret(ctx), operations.join("/"), src) };
};

/**
 * @nuxt/image 2 imports a provider's default export and calls
 * it for the provider; 1.x reads the named `getImage` above.
 */
export default () => ({ getImage });
