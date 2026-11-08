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
 * secret must never ship. If CRABCROP_SECRET is set, it is
 * used for the loader calls that happen during server
 * rendering only; browser-side calls fall back to unsigned
 * URLs. Unsigned URLs with appended sizing need the server's
 * unsafe mode without presets-only, an open image proxy fit
 * for local development only; production builds signed URLs
 * in server code (see the README's server modes).
 *
 * Widths follow the rule every crabcrop integration shares:
 * a fraction rounds to whole pixels, a width that is not a
 * size (0, negative, NaN) is left out, and a width past the
 * server's cap throws, as does a missing
 * NEXT_PUBLIC_CRABCROP_BASE, instead of minting a dead URL.
 */
import { buildURL, MAX_DIMENSION } from "@crabcrop/url";

const base = process.env.NEXT_PUBLIC_CRABCROP_BASE ?? "";
const preset = process.env.NEXT_PUBLIC_CRABCROP_PRESET;
const origin = process.env.NEXT_PUBLIC_CRABCROP_ORIGIN ?? "";
const secret =
  typeof window === "undefined" ? (process.env.CRABCROP_SECRET ?? null) : null;

const dim = (v: number | undefined, name: string): number | undefined => {
  if (v === undefined) return undefined;
  if (!Number.isFinite(v) || v <= 0) return undefined;
  const rounded = Math.round(v);
  // An over-cap dimension is a certain dead URL (the
  // server rejects the size at parse): throw here, where
  // the mistake is visible, instead of signing it.
  if (rounded > MAX_DIMENSION) {
    throw new Error(
      `crabcrop loader: ${name} ${rounded} exceeds the maximum dimension ${MAX_DIMENSION}`,
    );
  }
  return rounded;
};

export default function crabcropLoader({
  src,
  width,
  quality,
}: {
  src: string;
  width: number;
  quality?: number;
}): string {
  // Without a base the URL would be a relative /unsafe/...
  // path on the site's own origin: an image that never
  // loads, with nothing to say why.
  if (!base) {
    throw new Error(
      "crabcrop loader: NEXT_PUBLIC_CRABCROP_BASE is required (the crabcrop deployment origin)",
    );
  }
  const w = dim(width, "width");
  const source = src.startsWith("/") && origin ? `${origin}${src}` : src;
  const operations: string[] = [];
  if (preset) {
    operations.push(`preset:${preset}`);
  }
  if (w) {
    operations.push(`${w}x0`);
  }
  if (!preset && quality) {
    operations.push(`filters:quality(${quality})`);
  }
  return buildURL(base, secret, operations.join("/"), source);
}
