/**
 * A crabcrop provider for @nuxt/image.
 *
 * Copy this file into your project (for example as
 * `providers/crabcrop.ts`) and register it in nuxt.config:
 *
 *   export default defineNuxtConfig({
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
 * Provider options: `base` (required), `preset` (recommended:
 * the transform policy stays server-side and the browser
 * never needs the secret), `secret` (optional; honored only
 * when this code runs server-side. Nuxt compiles provider
 * options into the app, so in the browser the secret is
 * forced to null and can never ship in the client bundle;
 * prefer a preset for client-side use).
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
  secret?: string | null;
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

export const getImage = (
  src: string,
  { modifiers = {} }: { modifiers?: Modifiers },
  input?: { options?: CrabcropProviderOptions },
) => {
  const options = input?.options;
  if (!options?.base) {
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
  // The provider also runs client-side, where options from
  // nuxt.config are part of the bundle: never sign there
  // (audit PL-11, the Next.js loader's guard).
  const secret = typeof window === "undefined" ? (options.secret ?? null) : null;
  return { url: buildURL(options.base, secret, operations.join("/"), src) };
};
