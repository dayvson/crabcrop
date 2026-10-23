/**
 * A crabcrop loader for Hydrogen (@shopify/hydrogen), for
 * images that do NOT live on Shopify's CDN: product media
 * from Shopify keeps its native loader, while external
 * originals (editorial photos, UGC, a DAM) route through
 * crabcrop with this loader.
 *
 *   import { Image } from "@shopify/hydrogen";
 *   import { createCrabcropLoader } from "./crabcrop-loader";
 *
 *   const loader = createCrabcropLoader({
 *     base: "https://images.example.com",
 *     preset: "card",
 *   });
 *
 *   <Image loader={loader} data={{ url: externalUrl, altText: "..." }}
 *          widths={[320, 640, 960]} sizes="(min-width: 48em) 33vw, 100vw" />
 *
 * Configuration is a factory argument because Hydrogen apps
 * read configuration from the request context, not from
 * build-time env: pass `secret` only from server-side code
 * (a loader function in a route), or prefer a preset, which
 * needs no secret anywhere.
 */
import { buildURL } from "@crabcrop/url";

export interface CrabcropLoaderConfig {
  base: string;
  preset?: string;
  secret?: string | null;
  /** Unix-seconds expiry applied to every generated URL. */
  expires?: number;
}

export interface CrabcropImageParams {
  src: string;
  width?: number;
  height?: number;
  /** Device pixel ratio multiplier Hydrogen asks for. */
  scale?: number;
}

export function createCrabcropLoader(config: CrabcropLoaderConfig) {
  return ({ src, width, height, scale }: CrabcropImageParams): string => {
    const factor = scale ?? 1;
    const w = width ? Math.round(width * factor) : undefined;
    const h = height ? Math.round(height * factor) : undefined;
    const operations: string[] = [];
    if (config.preset) {
      operations.push(`preset:${config.preset}`);
    }
    if (w && h) {
      operations.push(`${w}x${h}`);
    } else if (w) {
      operations.push(`${w}x0`);
    } else if (h) {
      operations.push(`0x${h}`);
    }
    return buildURL(config.base, config.secret ?? null, operations.join("/"), src, {
      expires: config.expires,
    });
  };
}
