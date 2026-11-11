/**
 * Typed builders for crabcrop operation strings (AP-24).
 *
 * `buildURL` takes operations verbatim, by design: the string
 * is the spec. This module is the optional typed surface over
 * it: each call validates its arguments against the server's
 * grammar (the same bounds the explainer enforces) and returns
 * the operation segment in its canonical spelling, so a typed
 * pipeline and the equivalent hand-written string produce
 * byte-identical URLs. Validation throws at build time, like
 * `buildSrcSet` does for widths: a value the server would 400
 * never gets signed. The closed sets (formats, aligns, anchors,
 * modes, factors, shapes) and the class-name checks come from
 * the explainer (issue #482), so the builders and the
 * explainer read one copy of them. The numeric bounds are
 * still written on each side: the builders check numbers, the
 * explainer checks their spellings.
 */
import {
  BLEND_MODES,
  classFoldKey,
  CROP_ALIGNS,
  CURVE_CHANNELS,
  DPR_FACTORS,
  FLIP_AXES,
  IMAGE_FORMATS,
  isClassName,
  isOneOf,
  JPEG_SUBSAMPLINGS,
  MASK_SHAPES,
  MAX_OBJECTS_CLASSES,
  QUALITY_FORMATS,
  STUDIO_BACKDROPS,
  STUDIO_MODES,
  STUDIO_SHADOWS,
  TEXT_ANCHORS,
  trimLikeServer,
  UPSCALE_FACTORS,
} from "./explain.ts";
import { MAX_DIMENSION } from "./limits.ts";

/** Output formats the `format(...)` filter accepts. */
export type ImageFormat = "jpeg" | "jpg" | "png" | "webp" | "jxl" | "avif" | "gif" | "mp4" | "webm";

/** Formats a per-format quality entry can name (the video formats take no quality). */
export type PerFormatQualityFormat = "jpeg" | "png" | "webp" | "jxl" | "avif" | "gif";

/** Per-format encoder options for the `format(...)` filter (issue #319). */
export interface FormatOptions {
  /** JPEG: progressive (multi-scan) encoding. */
  progressive?: boolean;
  /** JPEG: chroma subsampling, 444, 422, or 420. */
  subsampling?: 444 | 422 | 420;
  /** PNG: Adam7 interlacing. */
  interlace?: boolean;
  /** WebP: libwebp compression method, 0..6. */
  method?: number;
  /** WebP: smart subsampling strength, 0..100. */
  sns?: number;
}

/** Crop alignments; `center` is the bare `crop:WxH` spelling. */
export type CropAlign = "center" | "smart" | "face" | "top" | "bottom" | "left" | "right" | "top-left" | "top-right" | "bottom-left" | "bottom-right";

/** Text-overlay anchor points. */
export type Anchor =
  | "top-left"
  | "top-center"
  | "top-right"
  | "center-left"
  | "center"
  | "center-right"
  | "bottom-left"
  | "bottom-center"
  | "bottom-right";

/** Overlay blend modes. */
export type BlendMode = "normal" | "multiply" | "screen" | "overlay";

/** Mask shapes for the `mask(...)` filter. */
export type MaskShape = "circle" | "ellipse";

/** Channels the `curve(...)` filter remaps. */
export type CurveChannel = "rgb" | "r" | "g" | "b";

/** Modes the `studio(...)` filter runs. */
export type StudioMode = "product" | "scene";

/** Contact shadow strengths for `studio(product, ...)`. */
export type StudioShadow = "soft" | "strong" | "none";

/**
 * Named arguments for the `studio(...)` filter. `bg` and
 * `shadow` are product arguments only: the server rejects
 * them under `scene`, so the builder throws for them there.
 */
export interface StudioOptions {
  /**
   * The product backdrop: `"white"`, `"brand"` (the
   * operator brand color, white when none is configured),
   * `"none"` (the subject keeps its transparency and no
   * shadow is synthesized), or a RRGGBB hex color. Unset
   * means the unnamed default: the brand color when one
   * is configured, else white. Naming `"white"` is not
   * the default: the two can resolve differently, so they
   * spell differently.
   */
  bg?: string;
  /** Contact shadow strength; the default is `"soft"`. */
  shadow?: StudioShadow;
  /**
   * Pad margin in percent, 0..40; the default is 10.
   * Under `scene` it is the margin the reframed subject
   * keeps.
   */
  pad?: number;
}

/**
 * One side of a size: a pixel count (1..{@link MAX_DIMENSION}),
 * 0 for proportional, a negative count to flip that axis, or
 * `"orig"` for the source's own dimension.
 */
export type Dimension = number | "orig";

function fail(what: string, value: unknown, rule: string): never {
  throw new Error(`${what} must be ${rule}, got ${JSON.stringify(value)}`);
}

/** A closed set spelled for an error message: `2 or 4`, `1, 1.5, 2, or 3`. */
function eitherOf(values: readonly (string | number)[], quote = false): string {
  const words = values.map((v) => (quote ? `"${v}"` : String(v)));
  const last = words.pop();
  return words.length === 1 ? `${words[0]} or ${last}` : `${words.join(", ")}, or ${last}`;
}

function intIn(what: string, value: number, min: number, max: number): number {
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    fail(what, value, `an integer from ${min} to ${max}`);
  }
  return value;
}

function floatIn(what: string, value: number, min: number, max: number): number {
  if (!Number.isFinite(value) || value < min || value > max) {
    fail(what, value, `a number from ${min} to ${max}`);
  }
  return value;
}

function positiveFloat(what: string, value: number): number {
  // The server reads a 32-bit float: past its range the value
  // is infinite there, and a positive value below it is zero.
  const f32 = Math.fround(value);
  if (!Number.isFinite(f32) || f32 <= 0) {
    fail(what, value, "a positive number");
  }
  return value;
}

/** Spell `digits * 10^exponent` in plain decimal, no exponent. */
function plainDecimal(digits: bigint, exponent: number): string {
  const text = digits.toString();
  if (exponent >= 0) return text + "0".repeat(exponent);
  const padded = text.padStart(1 - exponent, "0");
  const whole = padded.slice(0, padded.length + exponent);
  const fraction = padded.slice(padded.length + exponent).replace(/0+$/, "");
  return fraction ? `${whole}.${fraction}` : whole;
}

/**
 * The canonical spelling of a 32-bit float parameter (issue
 * #472). The server parses these arguments as `f32` and
 * prints them back as Rust's float `Display` does: the
 * shortest decimal that reads back as the same f32 (the
 * closer of two, a tie going up), never an exponent. So the
 * double `0.1 + 0.2` spells `0.3`, the form the server keys
 * its cache on, not its 17-digit double spelling. The search
 * is exact, on the float's binary value in BigInt
 * arithmetic, so no double rounding can pick a spelling Rust
 * would not.
 */
function num(value: number): string {
  const f = Math.fround(value);
  if (!Number.isFinite(f)) {
    throw new Error(`${value} is outside the 32-bit float range the server parses`);
  }
  if (f === 0) return "0";
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, Math.abs(f));
  const bits = view.getUint32(0);
  const biased = bits >>> 23;
  const fraction = bits & 0x7fffff;
  const mantissa = BigInt(biased === 0 ? fraction : fraction | 0x800000);
  // In units of a quarter ulp, 2^unit: the float sits at
  // 4 * mantissa and reads back from any decimal strictly
  // between the midpoints to its neighbors, which lie half as
  // far below at a power of two (the binade below is finer).
  // A decimal exactly on a midpoint rounds to the even
  // mantissa, so the ends count only for an even one.
  const unit = (biased === 0 ? 1 : biased) - 152;
  const center = 4n * mantissa;
  const low = fraction === 0 && biased > 1 ? center - 1n : center - 2n;
  const high = center + 2n;
  const inclusive = mantissa % 2n === 0n;
  const sign = f < 0 ? "-" : "";
  // From one decade above the value down, the first decimal
  // exponent with a candidate in the interval gives the
  // shortest spelling. Only the two neighbors of the value at
  // that exponent can qualify, and an f32 needs at most nine
  // significant digits, so the loop is short.
  for (let k = Math.floor(Math.log10(Math.abs(f))) + 2; ; k--) {
    // Compare digits * 10^k with units * 2^unit as integers.
    const binary = (unit > 0 ? 2n ** BigInt(unit) : 1n) * (k < 0 ? 10n ** BigInt(-k) : 1n);
    const step = (unit < 0 ? 2n ** BigInt(-unit) : 1n) * (k > 0 ? 10n ** BigInt(k) : 1n);
    const at = center * binary;
    const below = (at / step) * step;
    const above = below + step;
    const lowEnd = low * binary;
    const highEnd = high * binary;
    const downFits = inclusive ? below >= lowEnd : below > lowEnd;
    const upFits = inclusive ? above <= highEnd : above < highEnd;
    if (downFits || upFits) {
      const down = downFits && (!upFits || 2n * (at - below) < step);
      return sign + plainDecimal((down ? below : above) / step, k);
    }
  }
}

/**
 * The canonical spelling of a 64-bit float parameter (the
 * convolution kernel's values): integers without a fraction,
 * other values in plain decimal (the grammar takes no
 * exponent forms). JavaScript's shortest round-trip spelling
 * agrees with the server's for the magnitudes the grammar
 * allows; exponent spellings are expanded to plain decimal.
 */
function kernelNum(value: number): string {
  if (Number.isSafeInteger(value)) return String(value);
  const s = String(value);
  if (!s.includes("e")) return s;
  return value
    .toFixed(12)
    .replace(/0+$/, "")
    .replace(/\.$/, "");
}

function dimension(what: string, value: Dimension): string {
  if (value === "orig") return "orig";
  if (!Number.isSafeInteger(value) || Math.abs(value) > MAX_DIMENSION) {
    fail(what, value, `a pixel count up to ${MAX_DIMENSION}, 0, or "orig"`);
  }
  return String(value);
}

function pixelBox(what: string, width: number, height: number): string {
  intIn(`${what} width`, width, 1, MAX_DIMENSION);
  intIn(`${what} height`, height, 1, MAX_DIMENSION);
  return `${width}x${height}`;
}

function hex6(what: string, value: string): string {
  if (!/^[0-9a-fA-F]{6}$/.test(value)) {
    fail(what, value, "six hex digits (RRGGBB)");
  }
  return value.toLowerCase();
}

function hex6or8(what: string, value: string): string {
  if (/^[0-9a-fA-F]{6}$/.test(value)) return `${value.toLowerCase()}ff`;
  if (/^[0-9a-fA-F]{8}$/.test(value)) return value.toLowerCase();
  return fail(what, value, "six or eight hex digits (RRGGBB or RRGGBBAA)");
}

/**
 * The canonical percent-encoding for filter arguments that
 * carry text (watermark and overlay URLs, class names, text
 * labels): everything outside the RFC 3986 unreserved set is
 * encoded, byte-wise, exactly as the server's canonical form
 * spells it.
 */
function encodeArg(text: string): string {
  return encodeURIComponent(text).replace(
    /[!'()*]/g,
    (c) => `%${c.charCodeAt(0).toString(16).toUpperCase().padStart(2, "0")}`,
  );
}

function className(what: string, name: string): string {
  const trimmed = trimLikeServer(name);
  if (!isClassName(trimmed)) {
    fail(what, name, "1 to 64 UTF-8 bytes of letters, digits, space, `_`, `.`, `-`, `/`");
  }
  // The server trims the name before it validates and keys
  // on it, so the trimmed name is the canonical one.
  return encodeArg(trimmed);
}

/** Operation segments, one typed call per segment kind. */
export const ops = {
  /** `300x200`; each side a Dimension (0 proportional, negative flips). */
  size(width: Dimension, height: Dimension): string {
    return `${dimension("width", width)}x${dimension("height", height)}`;
  },

  /**
   * `fit-in:300x200`; no flips (the server rejects them here).
   * With `adaptive`, `fit-in:300x200:adaptive`: the box axes
   * swap when the source orientation disagrees with the box's.
   */
  fitIn(width: Dimension, height: Dimension, adaptive = false): string {
    for (const [what, v] of [
      ["width", width],
      ["height", height],
    ] as const) {
      if (typeof v === "number" && v < 0) {
        fail(`fit-in ${what}`, v, "not a flip: fit-in takes no negative sizes");
      }
    }
    const base = `fit-in:${dimension("width", width)}x${dimension("height", height)}`;
    return adaptive ? `${base}:adaptive` : base;
  },

  /** `crop:300x200` or `crop:300x200:smart`; an exact pixel box. */
  crop(width: number, height: number, align: CropAlign = "center"): string {
    if (!isOneOf(CROP_ALIGNS, align)) {
      fail("crop align", align, `one of ${CROP_ALIGNS.join(", ")}`);
    }
    const box = pixelBox("crop", width, height);
    return align === "center" ? `crop:${box}` : `crop:${box}:${align}`;
  },

  /**
   * `crop:300x200:objects(person,car):1.5`; crops to the
   * union of every detection of the named classes, the
   * union box scaled about its center by `scale` (1 is the
   * tight box; the crop stays centered on the union).
   * Needs a detection model configured on the server; when
   * none of the classes is detected the render falls back
   * to the plain center crop of the same size. Duplicate
   * classes (ASCII case-insensitive, as the server compares
   * them: `E` repeats `e`, `É` does not repeat `é`) are
   * spelled once, as the server's canonical form spells
   * them; at most 32 distinct classes, and a 33rd throws,
   * since the server rejects it with `invalid_crop`.
   */
  cropObjects(width: number, height: number, classes: string[], scale = 1): string {
    if (!Array.isArray(classes) || classes.length < 1) {
      fail("crop objects classes", classes, "an array of 1 to 32 distinct class names");
    }
    // The cap counts distinct names, as the server's parser
    // does (issue #408): repeats fold first and cost nothing.
    const seen = new Set<string>();
    const names: string[] = [];
    for (const name of classes) {
      const encoded = className("crop objects class", name);
      const key = classFoldKey(trimLikeServer(name));
      if (!seen.has(key)) {
        if (names.length === MAX_OBJECTS_CLASSES) {
          fail("crop objects classes", classes, "an array of 1 to 32 distinct class names");
        }
        seen.add(key);
        names.push(encoded);
      }
    }
    // Checked as the f32 the server reads (as are zoom, blur
    // and gamma below): a positive double that rounds to f32
    // zero would otherwise be spelled `0`, which it refuses.
    if (!Number.isFinite(scale) || Math.fround(scale) <= 0 || scale > 100) {
      fail("crop objects scale", scale, "a number above 0 and at most 100");
    }
    const box = pixelBox("crop", width, height);
    return `crop:${box}:objects(${names.join(",")}):${num(scale)}`;
  },

  /** `crop-rect:10x20:110x220` (left x top : right x bottom). */
  cropRect(left: number, top: number, right: number, bottom: number): string {
    intIn("crop-rect left", left, 0, 4294967295);
    intIn("crop-rect top", top, 0, 4294967295);
    intIn("crop-rect right", right, 0, 4294967295);
    intIn("crop-rect bottom", bottom, 0, 4294967295);
    if (right <= left || bottom <= top) {
      fail("crop-rect", `${left}x${top}:${right}x${bottom}`, "right > left and bottom > top");
    }
    return `crop-rect:${left}x${top}:${right}x${bottom}`;
  },

  /** `focal:10x20:110x220` (left x top : right x bottom); steers the next crop. */
  focal(left: number, top: number, right: number, bottom: number): string {
    intIn("focal left", left, 0, 4294967295);
    intIn("focal top", top, 0, 4294967295);
    intIn("focal right", right, 0, 4294967295);
    intIn("focal bottom", bottom, 0, 4294967295);
    if (right <= left || bottom <= top) {
      fail("focal", `${left}x${top}:${right}x${bottom}`, "right > left and bottom > top");
    }
    return `focal:${left}x${top}:${right}x${bottom}`;
  },

  /** `focalpoint:0.75x0.4:2` (fractions of the image dimensions, zoom optional); steers the next crop. */
  focalPoint(x: number, y: number, zoom = 1): string {
    const frac = (what: string, v: number): number => {
      if (!Number.isFinite(v) || v < 0 || v > 1) {
        fail(what, v, "a fraction from 0 to 1");
      }
      return v;
    };
    if (!Number.isFinite(zoom) || Math.fround(zoom) <= 0 || zoom > 100) {
      fail("focalpoint zoom", zoom, "a number above 0 and at most 100");
    }
    return `focalpoint:${num(frac("focalpoint x", x))}x${num(frac("focalpoint y", y))}:${num(zoom)}`;
  },

  /** `pad:300x200`, optionally `:blur` or a RRGGBB fill color. */
  pad(width: number, height: number, fill?: "blur" | string): string {
    const box = pixelBox("pad", width, height);
    if (fill === undefined) return `pad:${box}`;
    if (fill === "blur") return `pad:${box}:blur`;
    return `pad:${box}:${hex6("pad fill", fill)}`;
  },

  /** `preset:thumb`; expands server-side. */
  preset(name: string): string {
    // The server can only configure names from this
    // charset (its preset file validation), so anything
    // outside it is a reference that can only ever answer
    // unknown_preset; refuse it here instead.
    if (!/^[A-Za-z0-9_-]+$/.test(name)) {
      fail("preset name", name, "letters, digits, `_`, `-` (the server's preset name charset)");
    }
    return `preset:${name}`;
  },

  /**
   * `maxage:60`; this response's Cache-Control lifetime in
   * seconds (0 to 31536000), overriding the default year for
   * this request only. Header-only: it never enters the
   * cache key, so spellings that differ only here share one
   * stored image.
   */
  maxage(seconds: number): string {
    return `maxage:${intIn("maxage seconds", seconds, 0, 31536000)}`;
  },

  /**
   * `debug:focal`; the smart-crop debug view. Place it
   * before the crop it annotates: the response is a
   * diagnostic image of the focal decision (focal point,
   * focal rectangle, crop window drawn on the crop's
   * canvas), never the transform result, and it is never
   * cached (served with Cache-Control: no-store).
   */
  debugFocal(): string {
    return "debug:focal";
  },

  /** Join filter calls into one `filters:` segment. */
  filters(...calls: string[]): string {
    if (calls.length === 0) {
      fail("filters", calls, "at least one filter call");
    }
    return `filters:${calls.join(":")}`;
  },
};

/** Join operation segments into the operations string `buildURL` takes. */
export function joinOps(...segments: string[]): string {
  return segments.join("/");
}

/** Filter calls for `ops.filters(...)`; each returns its canonical spelling. */
export const filter = {
  grayscale(): string {
    return "grayscale";
  },
  invert(): string {
    return "invert";
  },
  noUpscale(): string {
    return "no_upscale";
  },
  equalize(): string {
    return "equalize";
  },
  enhance(): string {
    return "enhance";
  },
  removeBg(): string {
    return "removebg";
  },
  keepProfile(): string {
    return "keep_profile";
  },
  sepia(): string {
    return "sepia";
  },
  blurFaces(): string {
    return "blurfaces";
  },
  pixelateFaces(): string {
    return "pixelatefaces";
  },

  /** Gaussian blur by sigma: positive, at most the server's 512 cap. */
  blur(sigma: number): string {
    // The server also refuses a subnormal f32 sigma.
    if (!Number.isFinite(sigma) || Math.fround(sigma) < 2 ** -126 || sigma > 512) {
      fail("blur sigma", sigma, "a positive number up to 512");
    }
    return `blur(${num(sigma)})`;
  },
  /** Unsharp-mask strength (positive). */
  sharpen(amount: number): string {
    return `sharpen(${num(positiveFloat("sharpen amount", amount))})`;
  },
  /** Add to every channel: -255..255. */
  brighten(amount: number): string {
    return `brighten(${intIn("brighten amount", amount, -255, 255)})`;
  },
  /** Percent contrast: -100..100 (0 unchanged, -100 flat). */
  contrast(percent: number): string {
    return `contrast(${num(floatIn("contrast", percent, -100, 100))})`;
  },
  /** Gamma correction divisor: 0 < g <= 10 (1 unchanged; above 1 brightens midtones). */
  gamma(value: number): string {
    if (!Number.isFinite(value) || Math.fround(value) <= 0 || value > 10) {
      fail("gamma", value, "a number above 0 and at most 10");
    }
    return `gamma(${num(value)})`;
  },
  /** Exposure in stops: -10..10 (0 unchanged; +1 doubles the light, -1 halves it). */
  exposure(stops: number): string {
    if (!Number.isFinite(stops) || stops < -10 || stops > 10) {
      fail("exposure", stops, "a number between -10 and 10");
    }
    return `exposure(${num(stops)})`;
  },
  /** Highlights and shadows: two percents, each -100..100 (0 unchanged). */
  highlightsShadows(highlights: number, shadows: number): string {
    return `highlights_shadows(${num(floatIn("highlights_shadows highlights", highlights, -100, 100))},${num(floatIn("highlights_shadows shadows", shadows, -100, 100))})`;
  },
  /** Vibrance percent: -100..100 (0 unchanged; muted colors move most, vivid colors are protected). */
  vibrance(percent: number): string {
    return `vibrance(${num(floatIn("vibrance", percent, -100, 100))})`;
  },
  /** Rotate the hue by integer degrees. */
  hueRotate(degrees: number): string {
    return `huerotate(${intIn("huerotate degrees", degrees, -2147483648, 2147483647)})`;
  },
  /** Saturation factor: 0..255 (0 grayscale, 1 identity). */
  saturation(factor: number): string {
    return `saturation(${num(floatIn("saturation", factor, 0, 255))})`;
  },
  /** Per-channel offsets, each -255..255. */
  rgb(r: number, g: number, b: number): string {
    return `rgb(${intIn("rgb red", r, -255, 255)},${intIn("rgb green", g, -255, 255)},${intIn("rgb blue", b, -255, 255)})`;
  },
  /**
   * Tone curve: remap the channel through 2 to 16 [x, y]
   * control points, each coordinate 0..255 with x strictly
   * increasing. Between points the server interpolates
   * linearly; inputs outside the point range clamp to the
   * nearest end value.
   */
  curve(channel: CurveChannel, points: [number, number][]): string {
    if (!isOneOf(CURVE_CHANNELS, channel)) {
      fail("curve channel", channel, `one of ${CURVE_CHANNELS.join(", ")}`);
    }
    if (points.length < 2 || points.length > 16) {
      fail("curve points", points, "2 to 16 control points");
    }
    const flat: number[] = [];
    let prevX = -1;
    for (const [x, y] of points) {
      intIn("curve point x", x, 0, 255);
      intIn("curve point y", y, 0, 255);
      if (x <= prevX) {
        fail("curve points", points, "control points with strictly increasing x");
      }
      prevX = x;
      flat.push(x, y);
    }
    return `curve(${channel},${flat.join(",")})`;
  },
  /**
   * Convolution: apply a caller-supplied kernel, 9 values
   * (3x3) or 25 (5x5) in row-major order, each a plain
   * decimal from -1024 to 1024. With normalize set the
   * server scales the kernel by the sum of its entries
   * first (a zero sum applies the kernel as written). The
   * edges replicate. `normalize` is a boolean or the 0/1
   * flag itself.
   */
  convolution(values: number[], normalize: boolean | 0 | 1 = false): string {
    if (values.length !== 9 && values.length !== 25) {
      fail("convolution kernel", values, "9 (3x3) or 25 (5x5) values");
    }
    for (const v of values) {
      floatIn("convolution kernel value", v, -1024, 1024);
      if (!/^-?\d+(\.\d+)?$/.test(kernelNum(v))) {
        fail(
          "convolution kernel value",
          v,
          "a plain decimal (the grammar takes no exponent forms)",
        );
      }
    }
    if (
      normalize !== true &&
      normalize !== false &&
      normalize !== 0 &&
      normalize !== 1
    ) {
      fail("convolution normalize", normalize, "a boolean or 0/1");
    }
    const flag = normalize === true || normalize === 1 ? 1 : 0;
    return `convolution(${values.map(kernelNum).join(",")},${flag})`;
  },
  /**
   * Film grain: one noise value per pixel, added equally
   * to all three channels (alpha untouched), the value
   * uniform in plus or minus the given percent of full
   * scale. The field is deterministic, seeded from the
   * image content and the amount, so the same request
   * always renders the same bytes; 0 is a no-op.
   */
  noise(amount: number): string {
    return `noise(${intIn("noise amount", amount, 0, 100)})`;
  },
  /**
   * Rotate by an angle in degrees (decimals allowed, no
   * exponent spelling); multiples of 90 are exact quarter
   * turns on the server.
   */
  rotate(angle: number): string {
    floatIn("rotate angle", angle, -36000, 36000);
    if (!/^-?\d+(\.\d+)?$/.test(num(angle))) {
      fail("rotate angle", angle, "a plain decimal angle (the grammar takes no exponent forms)");
    }
    return `rotate(${num(angle)})`;
  },
  /** Trim uniform borders; tolerance 0..255, default 10. */
  trim(tolerance = 10): string {
    return `trim(${intIn("trim tolerance", tolerance, 0, 255)})`;
  },
  /** Trim uniform borders, then remove pixels per side (left, top, right, bottom). */
  trimSides(tolerance: number, left: number, top: number, right: number, bottom: number): string {
    const side = (what: string, v: number) => intIn(what, v, 0, 4294967295);
    return `trim(${intIn("trim tolerance", tolerance, 0, 255)},${side("trim left", left)},${side("trim top", top)},${side("trim right", right)},${side("trim bottom", bottom)})`;
  },
  /** Encoding quality 1..100, or `"auto"` for perceptual targeting. */
  quality(value: number | "auto"): string {
    if (value === "auto") return "quality(auto)";
    return `quality(${intIn("quality", value, 1, 100)})`;
  },
  /**
   * Per-format quality in one URL: a map of format to value
   * (1..100 or "auto"), plus an optional fallback for
   * formats with no entry. Resolved by the server after
   * format negotiation; entries spell in canonical
   * (format-name) order with the fallback first.
   */
  qualityPerFormat(
    entries: Partial<Record<PerFormatQualityFormat, number | "auto">>,
    fallback?: number | "auto",
  ): string {
    const names = Object.keys(entries) as PerFormatQualityFormat[];
    if (names.length === 0 && fallback === undefined) {
      fail("quality", "{}", "at least one entry or a fallback");
    }
    const parts: string[] = [];
    if (fallback !== undefined) {
      parts.push(fallback === "auto" ? "auto" : String(intIn("quality", fallback, 1, 100)));
    }
    for (const name of names.sort()) {
      if (!isOneOf(QUALITY_FORMATS, name)) {
        fail("quality", name, `one of ${QUALITY_FORMATS.join(", ")}`);
      }
      const value = entries[name]!;
      parts.push(`${name}:${value === "auto" ? "auto" : intIn("quality", value, 1, 100)}`);
    }
    return `quality(${parts.join(",")})`;
  },
  /** Indexed PNG output with at most this many colors, 2..256; other formats ignore it. */
  palette(colors: number): string {
    return `palette(${intIn("palette colors", colors, 2, 256)})`;
  },
  /**
   * Output format, with its encoder options (issue #319):
   * `progressive` and `subsampling` (444, 422, or 420) for
   * JPEG, `interlace` for PNG, `method` (0..6) and `sns`
   * (smart subsampling, 0..100) for WebP. An option its
   * format cannot honour throws here, like the server's 400.
   */
  format(format: ImageFormat, options: FormatOptions = {}): string {
    if (!isOneOf(IMAGE_FORMATS, format)) {
      fail("format", format, `one of ${IMAGE_FORMATS.join(", ")}`);
    }
    // The output segment is the canonical spelling: the
    // `jpg` input alias spells as `jpeg`, like the server's
    // canonical form.
    const canonical = format === "jpg" ? "jpeg" : format;
    const args: string[] = [canonical];
    const used = (key: keyof FormatOptions) => options[key] !== undefined;
    if (format === "jpeg" || format === "jpg") {
      if (used("interlace") || used("method") || used("sns")) {
        fail("format", JSON.stringify(options), "jpeg takes progressive and subsampling only");
      }
      if (options.progressive) args.push("progressive");
      if (options.subsampling !== undefined) {
        if (!isOneOf(JPEG_SUBSAMPLINGS, options.subsampling)) {
          fail("subsampling", options.subsampling, `one of ${JPEG_SUBSAMPLINGS.join(", ")}`);
        }
        args.push(`subsampling:${options.subsampling}`);
      }
    } else if (format === "png") {
      if (used("progressive") || used("subsampling") || used("method") || used("sns")) {
        fail("format", JSON.stringify(options), "png takes interlace only");
      }
      if (options.interlace) args.push("interlace");
    } else if (format === "webp") {
      if (used("progressive") || used("subsampling") || used("interlace")) {
        fail("format", JSON.stringify(options), "webp takes method and sns only");
      }
      if (options.method !== undefined) {
        args.push(`method:${intIn("method", options.method, 0, 6)}`);
      }
      if (options.sns !== undefined) {
        args.push(`sns:${intIn("sns", options.sns, 0, 100)}`);
      }
    } else if (
      used("progressive") ||
      used("subsampling") ||
      used("interlace") ||
      used("method") ||
      used("sns")
    ) {
      fail("format", JSON.stringify(options), `${format} takes no encoder options`);
    }
    return `format(${args.join(",")})`;
  },
  /** The background color other operations pad with (RRGGBB). */
  fill(color: string): string {
    return `fill(${hex6("fill color", color)})`;
  },
  /** How much a face crop expands around the face: 0..3. */
  facePad(amount: number): string {
    return `facepad(${num(floatIn("facepad", amount, 0, 3))})`;
  },
  /**
   * Studio mode: `studio(product)` composites the subject
   * the server's segmentation model finds onto a backdrop
   * with a synthesized contact shadow, scaled and centered
   * to the pad margin; `studio(scene)` keeps the
   * photograph's scene, reframing it on the detected
   * subject and correcting light and color. See
   * {@link StudioOptions} for the named arguments.
   * Arguments spelled at their defaults (shadow `soft`,
   * pad 10) are omitted, matching the server's canonical
   * spelling, so `studio("product", { pad: 10 })` and
   * `studio("product")` are the same string.
   */
  studio(mode: StudioMode, options: StudioOptions = {}): string {
    if (!isOneOf(STUDIO_MODES, mode)) {
      fail("studio mode", mode, `one of ${STUDIO_MODES.join(", ")}`);
    }
    const { bg, shadow, pad } = options;
    if (mode === "scene" && (bg !== undefined || shadow !== undefined)) {
      fail("studio scene options", options, "pad only: bg and shadow are product arguments");
    }
    const parts: string[] = [mode];
    if (mode === "product") {
      if (bg !== undefined) {
        const value = isOneOf(STUDIO_BACKDROPS, bg) ? bg : hex6("studio bg", bg);
        parts.push(`bg:${value}`);
      }
      if (shadow !== undefined) {
        if (!isOneOf(STUDIO_SHADOWS, shadow)) {
          fail("studio shadow", shadow, `one of ${STUDIO_SHADOWS.join(", ")}`);
        }
        if (shadow !== "soft") parts.push(`shadow:${shadow}`);
      }
    }
    if (pad !== undefined) {
      intIn("studio pad", pad, 0, 40);
      if (pad !== 10) parts.push(`pad:${pad}`);
    }
    return `studio(${parts.join(",")})`;
  },
  /** AI upscale by the given factor. */
  upscale(factor: 2 | 4): string {
    if (!isOneOf(UPSCALE_FACTORS, factor)) {
      fail("upscale factor", factor, eitherOf(UPSCALE_FACTORS));
    }
    return `upscale(${factor})`;
  },
  /** Device pixel ratio the sizes are multiplied by. */
  dpr(factor: 1 | 1.5 | 2 | 3): string {
    if (!isOneOf(DPR_FACTORS, factor)) {
      fail("dpr", factor, eitherOf(DPR_FACTORS));
    }
    return `dpr(${num(factor)})`;
  },
  /** Cap the output bytes, lowering quality to fit: at least 1. */
  maxBytes(bytes: number): string {
    if (!Number.isSafeInteger(bytes) || bytes < 1) {
      fail("max_bytes", bytes, "a positive integer byte budget");
    }
    return `max_bytes(${bytes})`;
  },
  /** Round the corners: one radius for all, or [tl, tr, br, bl]. */
  radius(value: number | [number, number, number, number]): string {
    const corners = Array.isArray(value) ? value : [value, value, value, value];
    for (const c of corners) intIn("radius", c, 0, MAX_DIMENSION);
    return `radius(${corners.join(",")})`;
  },
  /** Paint an opaque border inside the frame (RRGGBB); a width of 0 is a no-op. */
  border(width: number, color: string): string {
    return `border(${intIn("border width", width, 0, MAX_DIMENSION)},${hex6("border color", color)})`;
  },
  mask(shape: MaskShape): string {
    if (!isOneOf(MASK_SHAPES, shape)) {
      fail("mask shape", shape, eitherOf(MASK_SHAPES, true));
    }
    return `mask(${shape})`;
  },
  duotone(shadow: string, highlight: string): string {
    return `duotone(${hex6("duotone shadow", shadow)},${hex6("duotone highlight", highlight)})`;
  },
  monochrome(color: string): string {
    return `monochrome(${hex6("monochrome color", color)})`;
  },
  /** Mosaic block size in pixels: 1..{@link MAX_DIMENSION}. */
  pixelate(blockSize: number): string {
    return `pixelate(${intIn("pixelate size", blockSize, 1, MAX_DIMENSION)})`;
  },
  flip(axis: "h" | "v"): string {
    if (!isOneOf(FLIP_AXES, axis)) {
      fail("flip axis", axis, eitherOf(FLIP_AXES, true));
    }
    return `flip(${axis})`;
  },
  /** Overlay a watermark image at (x, y), opacity 0..100. */
  watermark(url: string, x: number, y: number, alpha: number): string {
    intIn("watermark x", x, -2147483648, 2147483647);
    intIn("watermark y", y, -2147483648, 2147483647);
    intIn("watermark alpha", alpha, 0, 100);
    if (url === "") fail("watermark url", url, "a non-empty URL");
    return `watermark(${encodeArg(url)},${x},${y},${alpha})`;
  },
  /**
   * Composite another image over the result at (x, y) with
   * opacity 0..100 and a blend mode; `widthPct` optionally
   * scales it to that percent of the base width (1..400).
   */
  overlay(
    url: string,
    x: number,
    y: number,
    alpha: number,
    blend: BlendMode,
    widthPct?: number,
  ): string {
    intIn("overlay x", x, -2147483648, 2147483647);
    intIn("overlay y", y, -2147483648, 2147483647);
    intIn("overlay alpha", alpha, 0, 100);
    if (!isOneOf(BLEND_MODES, blend)) {
      fail("overlay blend", blend, `one of ${BLEND_MODES.join(", ")}`);
    }
    if (url === "") fail("overlay url", url, "a non-empty URL");
    const width = widthPct === undefined ? "" : `,${intIn("overlay width", widthPct, 1, 400)}`;
    return `overlay(${encodeArg(url)},${x},${y},${alpha},${blend}${width})`;
  },
  /**
   * Draw a text overlay. Color is RRGGBB or RRGGBBAA; the
   * canonical spelling is always eight lowercase hex digits.
   */
  text(content: string, size: number, color: string, anchor: Anchor): string {
    if (!isOneOf(TEXT_ANCHORS, anchor)) {
      fail("text anchor", anchor, `one of ${TEXT_ANCHORS.join(", ")}`);
    }
    if (content === "") fail("text", content, "a non-empty label");
    intIn("text size", size, 1, 1000);
    return `text(${encodeArg(content)},${size},${hex6or8("text color", color)},${anchor})`;
  },
  /** Blur every detected object of the named class. */
  blurClass(name: string): string {
    return `blurclass(${className("blurclass name", name)})`;
  },
  /** Pixelate every detected object of the named class. */
  pixelateClass(name: string): string {
    return `pixelateclass(${className("pixelateclass name", name)})`;
  },
  /** Recolor every detected object of the named class to the given RRGGBB color. */
  recolorClass(name: string, color: string): string {
    return `recolorclass(${className("recolorclass name", name)},${hex6("recolorclass color", color)})`;
  },
  /** Blur the rectangle (x, y, w, h) in pixels. */
  blurRegion(x: number, y: number, w: number, h: number): string {
    return `blur_region(${region(x, y, w, h)})`;
  },
  /** Pixelate the rectangle (x, y, w, h) in pixels. */
  pixelateRegion(x: number, y: number, w: number, h: number): string {
    return `pixelate_region(${region(x, y, w, h)})`;
  },
  /** Erase the rectangle (x, y, w, h) in pixels and inpaint it (AI). */
  removeRegion(x: number, y: number, w: number, h: number): string {
    return `remove_region(${region(x, y, w, h)})`;
  },
};

function region(x: number, y: number, w: number, h: number): string {
  intIn("region x", x, 0, MAX_DIMENSION);
  intIn("region y", y, 0, MAX_DIMENSION);
  intIn("region width", w, 1, MAX_DIMENSION);
  intIn("region height", h, 1, MAX_DIMENSION);
  return `${x},${y},${w},${h}`;
}
