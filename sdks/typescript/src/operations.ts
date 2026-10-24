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
 * never gets signed.
 */

/** Output formats the `format(...)` filter accepts. */
export type ImageFormat = "jpeg" | "jpg" | "png" | "webp" | "jxl" | "avif" | "gif";

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

/**
 * One side of a size: a pixel count (1..16383), 0 for
 * proportional, a negative count to flip that axis, or
 * `"orig"` for the source's own dimension.
 */
export type Dimension = number | "orig";

const MAX_DIMENSION = 16383;

function fail(what: string, value: unknown, rule: string): never {
  throw new Error(`${what} must be ${rule}, got ${JSON.stringify(value)}`);
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
  if (!Number.isFinite(value) || value <= 0) {
    fail(what, value, "a positive number");
  }
  return value;
}

/**
 * The canonical number spelling: integers without a fraction,
 * other values in plain decimal (the grammar takes no
 * exponent forms). JavaScript's shortest round-trip spelling
 * agrees with the server's for the magnitudes the grammar
 * allows; exponent spellings are expanded to plain decimal.
 */
function num(value: number): string {
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
  const trimmed = name.trim();
  if (
    trimmed.length < 1 ||
    trimmed.length > 64 ||
    !/^[\p{L}\p{N} _.\-/]+$/u.test(trimmed)
  ) {
    fail(what, name, "1 to 64 characters of letters, digits, space, `_`, `.`, `-`, `/`");
  }
  return encodeArg(name);
}

const CROP_ALIGNS: readonly string[] = ["center", "smart", "face", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right"];
const ANCHORS: readonly string[] = [
  "top-left",
  "top-center",
  "top-right",
  "center-left",
  "center",
  "center-right",
  "bottom-left",
  "bottom-center",
  "bottom-right",
];
const BLEND_MODES: readonly string[] = ["normal", "multiply", "screen", "overlay"];
const CURVE_CHANNELS: readonly string[] = ["rgb", "r", "g", "b"];
const IMAGE_FORMATS: readonly string[] = ["jpeg", "jpg", "png", "webp", "jxl", "avif", "gif"];

/** Operation segments, one typed call per segment kind. */
export const ops = {
  /** `300x200`; each side a Dimension (0 proportional, negative flips). */
  size(width: Dimension, height: Dimension): string {
    return `${dimension("width", width)}x${dimension("height", height)}`;
  },

  /** `fit-in:300x200`; no flips (the server rejects them here). */
  fitIn(width: Dimension, height: Dimension): string {
    for (const [what, v] of [
      ["width", width],
      ["height", height],
    ] as const) {
      if (typeof v === "number" && v < 0) {
        fail(`fit-in ${what}`, v, "not a flip: fit-in takes no negative sizes");
      }
    }
    return `fit-in:${dimension("width", width)}x${dimension("height", height)}`;
  },

  /** `crop:300x200` or `crop:300x200:smart`; an exact pixel box. */
  crop(width: number, height: number, align: CropAlign = "center"): string {
    if (!CROP_ALIGNS.includes(align)) {
      fail("crop align", align, `one of ${CROP_ALIGNS.join(", ")}`);
    }
    const box = pixelBox("crop", width, height);
    return align === "center" ? `crop:${box}` : `crop:${box}:${align}`;
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

  /** `pad:300x200`, optionally `:blur` or a RRGGBB fill color. */
  pad(width: number, height: number, fill?: "blur" | string): string {
    const box = pixelBox("pad", width, height);
    if (fill === undefined) return `pad:${box}`;
    if (fill === "blur") return `pad:${box}:blur`;
    return `pad:${box}:${hex6("pad fill", fill)}`;
  },

  /** `preset:thumb`; expands server-side. */
  preset(name: string): string {
    if (name === "" || name.includes("/") || name.includes(":")) {
      fail("preset name", name, "a non-empty preset name without `/` or `:`");
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

  /** Gaussian blur by sigma (positive). */
  blur(sigma: number): string {
    return `blur(${num(positiveFloat("blur sigma", sigma))})`;
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
    if (!CURVE_CHANNELS.includes(channel)) {
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
  /** Encoding quality 1..100, or `"auto"` for perceptual targeting. */
  quality(value: number | "auto"): string {
    if (value === "auto") return "quality(auto)";
    return `quality(${intIn("quality", value, 1, 100)})`;
  },
  format(format: ImageFormat): string {
    if (!IMAGE_FORMATS.includes(format)) {
      fail("format", format, `one of ${IMAGE_FORMATS.join(", ")}`);
    }
    return `format(${format})`;
  },
  /** The background color other operations pad with (RRGGBB). */
  fill(color: string): string {
    return `fill(${hex6("fill color", color)})`;
  },
  /** How much a face crop expands around the face: 0..3. */
  facePad(amount: number): string {
    return `facepad(${num(floatIn("facepad", amount, 0, 3))})`;
  },
  /** AI upscale by the given factor. */
  upscale(factor: 2 | 4): string {
    if (factor !== 2 && factor !== 4) {
      fail("upscale factor", factor, "2 or 4");
    }
    return `upscale(${factor})`;
  },
  /** Device pixel ratio the sizes are multiplied by. */
  dpr(factor: 1 | 1.5 | 2 | 3): string {
    if (![1, 1.5, 2, 3].includes(factor)) {
      fail("dpr", factor, "1, 1.5, 2, or 3");
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
  /** Paint an opaque border inside the frame (RRGGBB). */
  border(width: number, color: string): string {
    return `border(${intIn("border width", width, 1, MAX_DIMENSION)},${hex6("border color", color)})`;
  },
  mask(shape: MaskShape): string {
    if (shape !== "circle" && shape !== "ellipse") {
      fail("mask shape", shape, '"circle" or "ellipse"');
    }
    return `mask(${shape})`;
  },
  duotone(shadow: string, highlight: string): string {
    return `duotone(${hex6("duotone shadow", shadow)},${hex6("duotone highlight", highlight)})`;
  },
  monochrome(color: string): string {
    return `monochrome(${hex6("monochrome color", color)})`;
  },
  /** Mosaic block size in pixels: 1..16383. */
  pixelate(blockSize: number): string {
    return `pixelate(${intIn("pixelate size", blockSize, 1, MAX_DIMENSION)})`;
  },
  flip(axis: "h" | "v"): string {
    if (axis !== "h" && axis !== "v") {
      fail("flip axis", axis, '"h" or "v"');
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
    if (!BLEND_MODES.includes(blend)) {
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
    if (!ANCHORS.includes(anchor)) {
      fail("text anchor", anchor, `one of ${ANCHORS.join(", ")}`);
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
  /** Blur the rectangle (x, y, w, h) in pixels. */
  blurRegion(x: number, y: number, w: number, h: number): string {
    return `blur_region(${region(x, y, w, h)})`;
  },
  /** Pixelate the rectangle (x, y, w, h) in pixels. */
  pixelateRegion(x: number, y: number, w: number, h: number): string {
    return `pixelate_region(${region(x, y, w, h)})`;
  },
};

function region(x: number, y: number, w: number, h: number): string {
  intIn("region x", x, 0, MAX_DIMENSION);
  intIn("region y", y, 0, MAX_DIMENSION);
  intIn("region width", w, 1, MAX_DIMENSION);
  intIn("region height", h, 1, MAX_DIMENSION);
  return `${x},${y},${w},${h}`;
}
