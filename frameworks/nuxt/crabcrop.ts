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
 * never needs the secret), `secret` (optional; only for
 * providers configured in server-only contexts).
 *
 * Modifier mapping:
 *   width + height, fit cover (default) -> crop:WxH:smart
 *   width + height, fit contain         -> pad:WxH:blur
 *   width + height, fit inside/outside  -> fit-in:WxH
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

const dim = (v: number | string | undefined): number | undefined => {
  const n = v === undefined ? NaN : Number(v);
  return Number.isFinite(n) && n > 0 ? Math.round(n) : undefined;
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
  const width = dim(modifiers.width);
  const height = dim(modifiers.height);
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
      case "outside":
        operations.push(`fit-in:${width}x${height}`);
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
    operations.push(`filters:format(${format})`);
  }
  if (modifiers.quality !== undefined) {
    operations.push(`filters:quality(${dim(modifiers.quality)})`);
  }
  return { url: buildURL(options.base, options.secret ?? null, operations.join("/"), src) };
};
