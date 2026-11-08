/**
 * A crabcrop image service for Astro (astro:assets), following
 * Astro's Image Service API as an external service: Astro calls
 * the hooks below at build time and during on-demand rendering,
 * and the emitted <img> points straight at your crabcrop
 * deployment, which does the transforming.
 *
 * Copy this file into your project and register it in
 * astro.config:
 *
 *   // astro.config.mjs
 *   import { defineConfig } from "astro/config";
 *
 *   export default defineConfig({
 *     image: {
 *       service: {
 *         entrypoint: "./crabcrop-service.ts",
 *         config: {
 *           base: "https://images.example.com",
 *           preset: "card",
 *         },
 *       },
 *     },
 *   });
 *
 * Service config (read from `image.service.config` on every
 * hook call, the documented Astro pattern):
 *
 * - base (required): the crabcrop deployment origin.
 * - preset (recommended): a preset name defined on the
 *   server. The preset carries the transform policy (crop,
 *   format, quality); this service appends only the geometry
 *   and the per-image format and quality Astro asks for.
 * - origin: origin prepended to root-relative sources
 *   ("/_astro/hero.jpg" from a local import, or a plain
 *   "/uploads/a.jpg"), which crabcrop cannot fetch on their
 *   own.
 * - secret: signs the URLs. Unlike a client-side loader, an
 *   Astro service runs only on the server (build and SSR),
 *   so the secret never enters client code; read it from the
 *   environment in astro.config rather than hardcoding it.
 * - expires: a unix-seconds expiry applied to every URL.
 *
 * Transform mapping:
 *   width + height -> {width}x{height}
 *   width only -> {width}x0 ; height only -> 0x{height}
 *   format (jpg -> jpeg)   -> filters:format(...)
 *   quality                -> filters:quality(...)
 * Astro also accepts named quality presets ("low", "mid",
 * "high", "max"); they map to 25, 50, 80, 100, the values
 * Astro's own services use. Formats outside the server's
 * grammar (including "svg": use a plain <img> for those)
 * throw, rather than minting a URL the server would reject.
 */
import { buildURL, MAX_DIMENSION } from "@crabcrop/url";

export interface CrabcropServiceConfig {
  base: string;
  preset?: string;
  origin?: string;
  secret?: string | null;
  /** Unix-seconds expiry applied to every generated URL. */
  expires?: number;
}

/** The subset of Astro's ImageMetadata this service reads. */
export interface ImageMetadataLike {
  src: string;
  width: number;
  height: number;
  format: string;
}

/** The subset of Astro's ImageTransform this service reads. */
export interface ImageTransformLike {
  src: string | ImageMetadataLike;
  width?: number;
  height?: number;
  format?: string;
  quality?: number | string;
  widths?: number[];
  densities?: (number | `${number}x`)[];
  [key: string]: unknown;
}

/** The shape of the imageConfig argument Astro passes to every hook. */
export interface ImageConfigLike {
  service: { config?: CrabcropServiceConfig };
}

/** One entry of Astro's UnresolvedSrcSetValue. */
export interface SrcSetValueLike {
  transform: ImageTransformLike;
  descriptor?: string;
  attributes?: Record<string, unknown>;
}

const dim = (v: number | undefined, name: string): number | undefined => {
  if (v === undefined) return undefined;
  if (!Number.isFinite(v) || v <= 0) return undefined;
  const rounded = Math.round(v);
  // An over-cap dimension is a certain dead URL (the
  // server rejects the size at parse): throw here, where
  // the mistake is visible, instead of signing it.
  if (rounded > MAX_DIMENSION) {
    throw new Error(
      `crabcrop service: ${name} ${rounded} exceeds the maximum dimension ${MAX_DIMENSION}`,
    );
  }
  return rounded;
};

/** The output formats the server's grammar accepts. */
const FORMATS = new Set(["jpeg", "png", "webp", "avif", "jxl", "gif"]);

/** Astro's named quality presets, at Astro's own values. */
const QUALITY_PRESETS: Record<string, number> = {
  low: 25,
  mid: 50,
  high: 80,
  max: 100,
};

/**
 * Quality is an integer 1-100 in the grammar, so a present
 * but invalid value throws rather than minting
 * filters:quality(undefined) or quality(150). Astro's
 * named presets resolve through QUALITY_PRESETS first.
 */
const quality = (v: number | string): number => {
  const n = typeof v === "string" && v in QUALITY_PRESETS ? QUALITY_PRESETS[v] : Number(v);
  if (!Number.isInteger(n) || n < 1 || n > 100) {
    throw new Error(
      `crabcrop service: quality must be an integer between 1 and 100 (or one of ${Object.keys(QUALITY_PRESETS).join(", ")}), got ${v}`,
    );
  }
  return n;
};

const configOf = (imageConfig: ImageConfigLike): CrabcropServiceConfig => {
  const config = imageConfig?.service?.config;
  if (!config?.base) {
    throw new Error(
      "crabcrop service: the `base` service config is required (set it under image.service.config in astro.config)",
    );
  }
  return config;
};

const operationsFor = (options: ImageTransformLike, config: CrabcropServiceConfig): string => {
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
        `crabcrop service: unsupported format ${options.format} (expected one of ${[...FORMATS].join(", ")}, or jpg)`,
      );
    }
    operations.push(`filters:format(${format})`);
  }
  if (options.quality !== undefined) {
    operations.push(`filters:quality(${quality(options.quality)})`);
  }
  return operations.join("/");
};

const sourceOf = (options: ImageTransformLike, config: CrabcropServiceConfig): string => {
  const src = typeof options.src === "string" ? options.src : options.src.src;
  return src.startsWith("/") && config.origin ? `${config.origin}${src}` : src;
};

const service = {
  validateOptions(options: ImageTransformLike, imageConfig: ImageConfigLike) {
    // Fail at the first image, with the config path named,
    // rather than at the first getURL call.
    configOf(imageConfig);
    return options;
  },

  getURL(options: ImageTransformLike, imageConfig: ImageConfigLike): string {
    const config = configOf(imageConfig);
    return buildURL(config.base, config.secret ?? null, operationsFor(options, config), sourceOf(options, config), {
      expires: config.expires,
    });
  },

  getHTMLAttributes(options: ImageTransformLike, imageConfig: ImageConfigLike) {
    void imageConfig;
    // Fill the missing dimension from the source's aspect
    // ratio when Astro knows it (a local import), so the
    // browser can reserve the right box before loading.
    let targetWidth = options.width;
    let targetHeight = options.height;
    if (typeof options.src === "object") {
      const aspectRatio = options.src.width / options.src.height;
      if (targetHeight && !targetWidth) {
        targetWidth = Math.round(targetHeight * aspectRatio);
      } else if (targetWidth && !targetHeight) {
        targetHeight = Math.round(targetWidth / aspectRatio);
      }
    }
    const { src, format, quality: _quality, widths, densities, ...attributes } = options;
    void src;
    void format;
    void widths;
    void densities;
    return {
      ...attributes,
      width: targetWidth,
      height: targetHeight,
      loading: attributes.loading ?? "lazy",
      decoding: attributes.decoding ?? "async",
    };
  },

  getSrcSet(options: ImageTransformLike, imageConfig: ImageConfigLike): SrcSetValueLike[] {
    void imageConfig;
    // A variant keeps the requested box's aspect: when the
    // transform names both dimensions, the height scales
    // with the width.
    const variant = (width: number): ImageTransformLike => {
      const transform: ImageTransformLike = { ...options, width };
      if (options.width && options.height) {
        transform.height = Math.round((width * options.height) / options.width);
      }
      return transform;
    };
    const values: SrcSetValueLike[] = [];
    if (options.widths?.length) {
      for (const width of options.widths) {
        values.push({ transform: variant(width), descriptor: `${width}w` });
      }
    } else if (options.densities?.length && options.width) {
      for (const density of options.densities) {
        const factor = typeof density === "number" ? density : parseFloat(density);
        values.push({
          transform: variant(Math.round(options.width * factor)),
          descriptor: `${factor}x`,
        });
      }
    }
    return values;
  },
};

export default service;
