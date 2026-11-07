/**
 * A crabcrop loader for SvelteKit. SvelteKit has no built-in
 * image component, so the loader is a small factory your app
 * creates once and calls from components and load functions:
 *
 *   // src/lib/images.ts
 *   import { createCrabcropLoader } from "./crabcrop-loader";
 *
 *   export const images = createCrabcropLoader({
 *     base: "https://images.example.com",
 *     preset: "card",
 *   });
 *
 *   // in a component
 *   <img src={images.url(photo.url, { width: 800 })}
 *        srcset={images.srcset(photo.url, [320, 640, 960])}
 *        sizes="(min-width: 48em) 33vw, 100vw" alt={photo.alt} />
 *
 * Configuration is a factory argument because SvelteKit apps
 * read configuration from the environment per module: build
 * the loader in a server module (for example a `+layout.server`
 * load or a `$lib/server` module reading
 * `$env/static/private`) when a secret is involved. Unsigned
 * URLs with appended sizing need the server's unsafe mode
 * without presets-only, an open image proxy fit for local
 * development only (see the README's server modes). The
 * returned loader enforces the boundary: in a browser context the
 * secret is forced to null, so a factory created with one
 * still cannot sign (or leak it) client-side.
 */
import { buildURL } from "@crabcrop/url";

export interface CrabcropLoaderConfig {
  base: string;
  preset?: string;
  secret?: string | null;
  /** Unix-seconds expiry applied to every generated URL. */
  expires?: number;
  /** Origin prepended to root-relative `src` values ("/uploads/a.jpg"). */
  origin?: string;
}

export interface CrabcropImageOptions {
  width?: number;
  height?: number;
  format?: string;
  quality?: number | string;
}

export interface CrabcropLoader {
  url(src: string, options?: CrabcropImageOptions): string;
  srcset(src: string, widths: number[], options?: CrabcropImageOptions): string;
}

/** The server's per-axis dimension cap (the URL grammar's MAX_DIMENSION). */
const MAX_DIMENSION = 16383;

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

/** A srcset width must be an exact positive integer: it is spelled into the URL. */
const srcsetWidth = (v: number): number => {
  if (!Number.isSafeInteger(v) || v < 1 || v > MAX_DIMENSION) {
    throw new Error(
      `crabcrop loader: srcset width must be a positive integer no larger than ${MAX_DIMENSION}, got ${v}`,
    );
  }
  return v;
};

/** The output formats the server's grammar accepts. */
const FORMATS = new Set(["jpeg", "png", "webp", "avif", "jxl", "gif"]);

/**
 * Quality is an integer 1-100 in the grammar, so a present
 * but invalid value throws rather than minting
 * filters:quality(undefined) or quality(150).
 */
const quality = (v: number | string): number => {
  const n = Number(v);
  if (!Number.isInteger(n) || n < 1 || n > 100) {
    throw new Error(
      `crabcrop loader: quality must be an integer between 1 and 100, got ${v}`,
    );
  }
  return n;
};

export function createCrabcropLoader(config: CrabcropLoaderConfig): CrabcropLoader {
  if (!config?.base) {
    throw new Error("crabcrop loader: the `base` config is required");
  }
  const build = (src: string, options: CrabcropImageOptions): string => {
    const width = dim(options.width, "width");
    const height = dim(options.height, "height");
    const operations: string[] = [];
    if (config.preset) {
      operations.push(`preset:${config.preset}`);
    }
    if (width && height) {
      operations.push(`${width}x${height}`);
    } else if (width) {
      operations.push(`${width}x0`);
    } else if (height) {
      operations.push(`0x${height}`);
    }
    if (options.format) {
      const format = options.format === "jpg" ? "jpeg" : options.format;
      if (!FORMATS.has(format)) {
        throw new Error(
          `crabcrop loader: unsupported format ${options.format} (expected one of ${[...FORMATS].join(", ")}, or jpg)`,
        );
      }
      operations.push(`filters:format(${format})`);
    }
    if (options.quality !== undefined) {
      operations.push(`filters:quality(${quality(options.quality)})`);
    }
    const source = src.startsWith("/") && config.origin ? `${config.origin}${src}` : src;
    // In a browser context the secret is forced to null, so
    // it can never be used (or extracted) client-side
    // (audit PL-11, the Next.js loader's guard).
    const secret = typeof window === "undefined" ? (config.secret ?? null) : null;
    return buildURL(config.base, secret, operations.join("/"), source, {
      expires: config.expires,
    });
  };
  return {
    url: (src, options = {}) => build(src, options),
    // A srcset varies the width alone: each entry is the
    // same render at one width, spelled "{url} {width}w",
    // with any format and quality options carried through.
    srcset: (src, widths, options = {}) =>
      widths
        .map((w) => `${build(src, { ...options, width: srcsetWidth(w), height: undefined })} ${w}w`)
        .join(", "),
  };
}
