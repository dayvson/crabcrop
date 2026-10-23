/**
 * A crabcrop loader for next/image.
 *
 * Copy this file into your project (for example as
 * `crabcrop-loader.ts` at the project root) and point
 * next.config at it:
 *
 *   // next.config.mjs
 *   export default {
 *     images: { loader: "custom", loaderFile: "./crabcrop-loader.ts" },
 *   };
 *
 * Configuration is by environment variable:
 *
 * - NEXT_PUBLIC_CRABCROP_BASE (required): the crabcrop
 *   deployment origin, e.g. https://images.example.com.
 * - NEXT_PUBLIC_CRABCROP_PRESET (recommended): a preset name
 *   defined on the server. The preset carries the transform
 *   policy (crop, format, quality); this loader appends only
 *   the per-width sizing next/image asks for.
 * - NEXT_PUBLIC_CRABCROP_ORIGIN: origin prepended to relative
 *   `src` values ("/uploads/a.jpg"), which crabcrop cannot
 *   fetch on their own.
 *
 * Signing: a custom loader also runs in the browser, where a
 * secret must never ship. Use a preset (with the server's
 * unsafe or presets-only mode) as the normal configuration.
 * If CRABCROP_SECRET is set, it is used for the loader calls
 * that happen during server rendering only; browser-side
 * calls fall back to unsigned URLs.
 */
import { buildURL } from "@crabcrop/url";

const base = process.env.NEXT_PUBLIC_CRABCROP_BASE ?? "";
const preset = process.env.NEXT_PUBLIC_CRABCROP_PRESET;
const origin = process.env.NEXT_PUBLIC_CRABCROP_ORIGIN ?? "";
const secret =
  typeof window === "undefined" ? (process.env.CRABCROP_SECRET ?? null) : null;

export default function crabcropLoader({
  src,
  width,
  quality,
}: {
  src: string;
  width: number;
  quality?: number;
}): string {
  const source = src.startsWith("/") && origin ? `${origin}${src}` : src;
  const operations: string[] = [];
  if (preset) {
    operations.push(`preset:${preset}`);
  }
  operations.push(`${width}x0`);
  if (!preset && quality) {
    operations.push(`filters:quality(${quality})`);
  }
  return buildURL(base, secret, operations.join("/"), source);
}
