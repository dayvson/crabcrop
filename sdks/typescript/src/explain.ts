/**
 * Parse a crabcrop URL (or bare path) back into its pieces and
 * explain each segment, mirroring the server's grammar. This is
 * the one JavaScript implementation of the grammar (issue #482):
 * the typed builders in operations.ts validate with the checks
 * exported here, and the playground serves this module's build
 * (playground/sdk/, refreshed by scripts/sync_playground_sdk.py,
 * whose --check fails on drift or on a second copy). Every
 * surface is pinned to the same verdict corpora (audit PL-10,
 * P-3).
 */
import { isReservedRemainder, MAX_DIMENSION } from "./limits.ts";

/** The segment kinds an explanation is made of. */
export type SegmentKind = "expiry" | "maxage" | "debug" | "preset" | "operation";

/** The signature kinds an explanation can report. */
export type SignatureKind = "hmac" | "unsafe" | "sealed" | "invalid";

export interface ExplainedSegment {
  raw: string;
  kind: SegmentKind;
  title: string;
  detail?: string;
  error?: string;
  expired?: boolean;
}

export interface Explanation {
  signature: { kind: SignatureKind; value: string; detail: string; keyId?: string } | null;
  segments: ExplainedSegment[];
  source: string;
  remainder: string;
  errors: string[];
}

export const FILTER_DESCRIPTIONS: Record<string, string> = {
  grayscale: "convert to grayscale",
  invert: "invert the colors",
  no_upscale: "never enlarge the source",
  blur: "gaussian blur, argument is the sigma",
  blurfaces: "blur detected faces (needs a face model on the server)",
  brighten: "brighten by the given amount (-255..255)",
  contrast: "contrast in percent, -100..100 (0 is unchanged, -100 is flat)",
  gamma:
    "gamma correction with divisor g, 0 < g <= 10: each channel becomes 255 * (v / 255)^(1 / g) (1 is unchanged, above 1 brightens midtones)",
  exposure:
    "exposure in stops, -10..10: multiply the linear light by 2^stops (0 is unchanged, positive brightens, negative darkens)",
  highlights_shadows:
    "adjust the highlights and shadows ranges independently: two percents, each -100..100 (0 is unchanged)",
  vibrance:
    "saturation weighted toward the muted colors: percent, -100..100 (0 is unchanged; vivid colors and skin tones are protected)",
  huerotate: "rotate the hue by the given degrees",
  rotate: "rotate by the given angle in degrees; the canvas expands to fit",
  quality:
    "output quality 1..100, or `auto` for perceptual targeting; or per-format values in one list (quality(80,jpeg:90,webp:75)), resolved after format negotiation",
  palette:
    "write PNG output as an indexed (palette) image with at most this many colors, 2..256; ignored by the other formats",
  format:
    "force the output format; a format also takes its encoder options (format(jpeg,progressive), format(png,interlace), format(webp,method:6,sns:50)); an option its encoder cannot honour is refused",
  fill: "the background color other operations pad with",
  facepad: "how much a face crop expands around the face",
  pixelatefaces: "pixelate detected faces (needs a face model on the server)",
  blurclass: "blur every detected object of the named class (needs a detection model on the server)",
  pixelateclass:
    "pixelate every detected object of the named class (needs a detection model on the server)",
  recolorclass:
    "recolor every detected object of the named class to the given color (needs a detection model on the server)",
  removebg: "remove the background (needs an AI model on the server)",
  studio:
    "studio mode: `product` composites the subject onto a backdrop (bg: white, a RRGGBB color, brand, or none) with a synthesized contact shadow (shadow: soft, strong, or none), scaled and centered to a pad margin (pad: 0..40, default 10); `scene` keeps the photograph's scene, reframing it on the detected subject with light and color correction (pad only) (needs an AI model on the server)",
  keep_profile: "keep the source color profile instead of converting to sRGB",
  upscale: "AI upscale by the given factor (needs an AI model on the server)",
  equalize: "histogram equalization",
  enhance: "gentle per-channel auto-levels (0.5th percentile to 0, 99.5th to 255)",
  saturation: "saturation factor, 0..255 (0 is grayscale, 1 is identity)",
  sharpen: "sharpen by the given amount",
  text: "draw a text overlay (needs a font on the server)",
  rgb: "tint toward the given red/green/blue amounts",
  curve: "tone curve: remap the named channel (rgb, r, g, or b) through 2 to 16 x,y control points, linearly interpolated, ends clamped",
  convolution: "convolve with a 3x3 or 5x5 kernel: 9 or 25 row-major values (plain decimals, -1024..1024), then a normalize flag (0 or 1); edges replicate",
  noise: "add monochrome film grain: one deterministic noise value per pixel on all channels, strength 0..100 percent of full scale (0 is a no-op)",
  trim: "trim uniform borders; the five-argument form (tolerance,left,top,right,bottom) also removes that many pixels per side after the tolerance pass",
  dpr: "device pixel ratio the sizes are multiplied by",
  watermark: "overlay a watermark image",
  max_bytes: "cap the output size, lowering quality to fit",
  radius: "round the corners by the given radius in pixels (one value, or four in CSS corner order); needs an alpha-capable format",
  border: "paint a border of the given width and color (RRGGBB) inside the frame",
  mask: "keep only the inscribed `circle` or `ellipse`; the rest becomes transparent",
  sepia: "the classic warm sepia tone",
  duotone: "map shadows to the first color and highlights to the second, by luminance",
  monochrome: "single-hue rendering in the given color (RRGGBB)",
  pixelate: "mosaic into blocks of the given pixel size, each block averaged",
  flip: "mirror the image: `h` horizontal, `v` vertical",
  blur_region: "gaussian-blur the rectangle (x,y,w,h) in pixels at this point in the chain (redaction)",
  overlay: "composite another image over the result: (url,x,y,alpha,blend[,width %]) with blend normal, multiply, screen, or overlay",
  pixelate_region: "pixelate the rectangle (x,y,w,h) in pixels at this point in the chain (redaction)",
  remove_region: "erase the rectangle (x,y,w,h) in pixels at this point in the chain and inpaint it from the surrounding pixels (AI region removal)",
};

/**
 * The server's size grammar (url.rs parse_size/parse_dimension),
 * ported (audit P-3): each side is an optional `-` flip marker
 * plus `` (proportional), `0`, `orig`, or a pixel count up to
 * the dimension cap. Returns null when the segment is not a
 * size, which in the server's dispatch means the source starts.
 */
interface ParsedDimension {
  kind: "proportional" | "original" | "pixels";
  value: number;
  flip: boolean;
}

interface ParsedSize {
  w: ParsedDimension;
  h: ParsedDimension;
}

function parseDimension(part: string): ParsedDimension | null {
  let flip = false;
  let digits = part;
  if (digits.startsWith("-")) {
    flip = true;
    digits = digits.slice(1);
  }
  if (digits === "" || digits === "0") return { kind: "proportional", value: 0, flip };
  if (digits === "orig") return { kind: "original", value: 0, flip };
  if (!/^\+?\d+$/.test(digits)) return null;
  if (BigInt(digits) > BigInt(MAX_DIMENSION)) return null;
  return { kind: "pixels", value: Number(digits), flip };
}

function parseSizeSegment(seg: string): ParsedSize | null {
  const x = seg.indexOf("x");
  if (x === -1) return null;
  const w = parseDimension(seg.slice(0, x));
  const h = parseDimension(seg.slice(x + 1));
  if (!w || !h) return null;
  return { w, h };
}

function isExactBox(size: ParsedSize): boolean {
  const px = (d: ParsedDimension) => d.kind === "pixels" && d.value >= 1 && !d.flip;
  return px(size.w) && px(size.h);
}

function sizeBlurb(seg: string): string {
  const size = parseSizeSegment(seg);
  if (!size) return `resize per ${seg}`;
  const describe = (d: ParsedDimension, axis: string) =>
    d.kind === "original"
      ? `the original ${axis}`
      : d.kind === "proportional"
        ? `proportional ${axis}`
        : `${d.value}px ${axis}`;
  const flips = [
    size.w.flip ? "flipped horizontally" : null,
    size.h.flip ? "flipped vertically" : null,
  ].filter(Boolean);
  return `resize to ${describe(size.w, "width")} by ${describe(size.h, "height")}${flips.length ? ` (${flips.join(", ")})` : ""}`;
}

// The grammar's closed sets, declared once for the explainer
// and the typed builders (issue #482). Each is `as const` so
// test/grammar-sets.test.ts can check it against the union
// type operations.ts gives callers: a value added to one side
// only fails the type check.

/** Crop alignments (url.rs parse_align); `center` is the bare `crop:WxH`. */
export const CROP_ALIGNS = ["center", "smart", "face", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right"] as const;

/** Text overlay anchors (url.rs Anchor::parse). */
export const TEXT_ANCHORS = [
  "top-left",
  "top-center",
  "top-right",
  "center-left",
  "center",
  "center-right",
  "bottom-left",
  "bottom-center",
  "bottom-right",
] as const;

/** The most characters a `text` label may hold (url.rs MAX_TEXT_GLYPHS, issue #606). */
export const MAX_TEXT_GLYPHS = 1000;

/** The most a `text` label's character count times its size may come to (url.rs MAX_TEXT_EXTENT). */
export const MAX_TEXT_EXTENT = 100000;

/**
 * Whether a decoded `text` label at `size` is within the
 * server's caps. Characters are Unicode code points, as the
 * server counts them, not bytes or UTF-16 code units.
 */
export function textLabelFits(label: string, size: number): boolean {
  let glyphs = 0;
  for (const _ of label) {
    glyphs += 1;
    if (glyphs > MAX_TEXT_GLYPHS) return false;
  }
  return glyphs * size <= MAX_TEXT_EXTENT;
}

/** Output formats `format(...)` names; `jpg` is an input alias of `jpeg`. */
export const IMAGE_FORMATS = ["jpeg", "jpg", "png", "webp", "jxl", "avif", "gif", "mp4", "webm"] as const;

/**
 * Formats a per-format `quality(...)` entry names, in
 * canonical (name) order: the still formats, since a video
 * format's quality is not the server's to set. `jpg` is
 * accepted as the input alias of `jpeg`.
 */
export const QUALITY_FORMATS = ["avif", "gif", "jpeg", "jxl", "png", "webp"] as const;

/** JPEG chroma subsamplings `format(jpeg,subsampling:...)` takes. */
export const JPEG_SUBSAMPLINGS = [444, 422, 420] as const;

/** Overlay blend modes. */
export const BLEND_MODES = ["normal", "multiply", "screen", "overlay"] as const;

/** Channels `curve(...)` remaps. */
export const CURVE_CHANNELS = ["rgb", "r", "g", "b"] as const;

/** Modes `studio(...)` runs. */
export const STUDIO_MODES = ["product", "scene"] as const;

/** Named backdrops for `studio(product,bg:...)`; any RRGGBB color also works. */
export const STUDIO_BACKDROPS = ["white", "brand", "none"] as const;

/** Contact shadow strengths for `studio(product, ...)`. */
export const STUDIO_SHADOWS = ["soft", "strong", "none"] as const;

/** Factors `upscale(...)` takes. */
export const UPSCALE_FACTORS = [2, 4] as const;

/** Device pixel ratios `dpr(...)` takes. */
export const DPR_FACTORS = [1, 1.5, 2, 3] as const;

/** Shapes `mask(...)` cuts. */
export const MASK_SHAPES = ["circle", "ellipse"] as const;

/** Axes `flip(...)` mirrors across. */
export const FLIP_AXES = ["h", "v"] as const;

/** Whether `value` is a member of the closed set `list`. */
export function isOneOf<T extends string | number>(list: readonly T[], value: unknown): value is T {
  return (list as readonly unknown[]).includes(value);
}

/**
 * The objects crop's cap on distinct class names (url.rs
 * MAX_OBJECTS_CLASSES).
 */
export const MAX_OBJECTS_CLASSES = 32;

/**
 * Whether a UTF-16 code unit is in the Unicode White_Space
 * set, the set Rust's `str::trim` strips. Every member is in
 * the Basic Multilingual Plane, so one code unit is one
 * character here, and a surrogate is never white space.
 */
function isServerWhiteSpace(unit: number): boolean {
  return (
    (unit >= 0x0009 && unit <= 0x000d) ||
    unit === 0x0020 ||
    unit === 0x0085 ||
    unit === 0x00a0 ||
    unit === 0x1680 ||
    (unit >= 0x2000 && unit <= 0x200a) ||
    unit === 0x2028 ||
    unit === 0x2029 ||
    unit === 0x202f ||
    unit === 0x205f ||
    unit === 0x3000
  );
}

/**
 * Trim like the server's Rust `str::trim`: exactly the
 * Unicode White_Space set, from both ends. JavaScript's
 * `String.prototype.trim` disagrees on two characters:
 * it also strips U+FEFF, which the server keeps (so a
 * class name wrapped in it is no class name), and it
 * keeps U+0085, which the server strips (issue #537).
 * Every place this package mirrors a server-side trim
 * uses this instead of the built-in. It walks in from
 * each end, so it takes time linear in the input, however
 * long a run of white space is.
 */
export function trimLikeServer(text: string): string {
  let start = 0;
  let end = text.length;
  while (start < end && isServerWhiteSpace(text.charCodeAt(start))) start++;
  while (end > start && isServerWhiteSpace(text.charCodeAt(end - 1))) end--;
  return text.slice(start, end);
}

/**
 * A detection class name as the server's parse_class_name
 * takes it, already trimmed: 1 to 64 UTF-8 bytes (bytes, not
 * UTF-16 code units, as on the server) of letters, digits,
 * space, `_`, `.`, `-`, `/`. A letter or digit is what Rust's
 * `char::is_alphanumeric` calls one: the Unicode Alphabetic
 * property, or a number category. Alphabetic holds more than
 * the letter categories: the vowel signs of Indic scripts
 * (`हिंदी`), combining marks such as U+0345, and circled
 * letters such as `Ⓐ` (issue #712).
 */
export function isClassName(name: string): boolean {
  return (
    name.length >= 1 &&
    new TextEncoder().encode(name).length <= 64 &&
    /^[\p{Alphabetic}\p{N} _.\-/]+$/u.test(name)
  );
}

/**
 * The key a class name folds to when the objects crop drops
 * repeats. The server compares names with
 * `eq_ignore_ascii_case` (url.rs parse_crop_align), so only
 * A to Z fold: `E` and `e` are one class, `É` and `é` are two
 * (issue #482).
 */
export function classFoldKey(name: string): string {
  return name.replace(/[A-Z]/g, (c) => c.toLowerCase());
}

const NO_ARG_FILTERS = new Set([
  "grayscale",
  "invert",
  "no_upscale",
  "equalize",
  "enhance",
  "removebg",
  "keep_profile",
  "sepia",
  "blurfaces",
  "pixelatefaces",
]);
const FORMAT_NAMES = new Set<string>(IMAGE_FORMATS);
const ANCHOR_NAMES = new Set<string>(TEXT_ANCHORS);

// Numbers (issue #712). The server reads every numeric
// argument with its Rust type's `FromStr`, and the explainer
// reads each one the same way, so the two give one verdict on
// every spelling. spec/url-signing.md (Argument spellings)
// states the rules for every implementation.

/**
 * Whether `s` is an integer in `min..=max` as Rust's unsigned
 * `FromStr` reads one: an optional `+`, then ASCII digits,
 * leading zeros allowed. A `-` is refused, even in `-0`.
 */
function isUintIn(s: string | null, min: number | bigint, max: number | bigint): boolean {
  return s !== null && /^\+?\d+$/.test(s) && integerIn(s, min, max);
}

/**
 * Whether `s` is an integer in `min..=max` as Rust's signed
 * `FromStr` reads one: an optional `+` or `-`, then ASCII
 * digits. `-0` is 0.
 */
function isIntIn(s: string | null, min: number | bigint, max: number | bigint): boolean {
  return s !== null && /^[+-]?\d+$/.test(s) && integerIn(s, min, max);
}

function integerIn(s: string, min: number | bigint, max: number | bigint): boolean {
  // Past 20 significant digits a value is outside every
  // integer type the grammar reads, and BigInt need not read
  // a digit run as long as the URL.
  if (s.replace(/^[+-]?0*/, "").length > 20) return false;
  const value = BigInt(s);
  return value >= BigInt(min) && value <= BigInt(max);
}

/**
 * Rust's float syntax: an optional sign, digits with an
 * optional fraction (`2`, `2.`, `.5`), and an optional
 * exponent. Each part matches one way, so a long run of
 * digits costs linear time, matched or not (issue #594).
 */
const DECIMAL = /^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?)(\d+))?$/;

/**
 * The most significant digits {@link readDecimal} keeps. Every
 * midpoint between two 64-bit floats has at most 767
 * significant digits, so a decimal cut to this many, with one
 * nonzero digit appended when any cut digit is nonzero, rounds
 * to the float the whole decimal rounds to.
 */
const KEPT_DIGITS = 1000;

/** The precision and range of the two binary float formats. */
const BINARY = {
  32: { mantissa: 24, minExponent: -126, maxExponent: 127, overflow: 39, underflow: -46 },
  64: { mantissa: 53, minExponent: -1022, maxExponent: 1023, overflow: 309, underflow: -325 },
} as const;

/** The largest finite 32-bit float, Rust's `f32::MAX`. */
const F32_MAX = (2 - 2 ** -23) * 2 ** 127;

/**
 * The float a decimal argument reads as on the server (issue
 * #712): Rust's `f32` or `f64` `FromStr`, which takes an
 * optional sign, digits with an optional fraction (`2`, `2.`,
 * `.5`), and an optional exponent (`e` or `E`, an optional
 * sign, digits), and rounds the decimal to the nearest float of
 * `bits` bits, a tie to the even one. Past the float's range the
 * value is an infinity, and below half its smallest step it is
 * zero, signed as the decimal is. Returns null for any other
 * text, Rust's `inf` and `NaN` spellings included: no range the
 * grammar checks admits them.
 *
 * The rounding is exact, in BigInt arithmetic. Rounding to a
 * double first and then to 32 bits (`Math.fround(Number(s))`)
 * rounds twice, and gets a decimal near the midpoint between
 * two 32-bit floats wrong.
 */
export function readDecimal(text: string, bits: 32 | 64): number | null {
  const match = DECIMAL.exec(text);
  if (match === null) return null;
  const [, sign, whole, fraction = "", expSign = "", expDigits = "0"] = match;
  if (whole === "" && fraction === "") return null;
  const negative = sign === "-";
  const zero = negative ? -0 : 0;
  const infinity = negative ? -Infinity : Infinity;
  const format = BINARY[bits];
  // value = digits * 10^e10. The exponent stays a safe integer:
  // past nine digits it only has to stay past both pre-screens.
  let digits = (whole + fraction).replace(/^0+/, "");
  const expTrimmed = expDigits.replace(/^0+/, "");
  const magnitude = expTrimmed.length > 9 ? 1e9 : Number(expTrimmed || "0");
  let e10 = (expSign === "-" ? -magnitude : magnitude) - fraction.length;
  let end = digits.length;
  while (end > 0 && digits.charCodeAt(end - 1) === 48) end--;
  e10 += digits.length - end;
  digits = digits.slice(0, end);
  if (digits === "") return zero;
  const n = digits.length;
  if (n - 1 + e10 >= format.overflow) return infinity;
  if (n + e10 <= format.underflow) return zero;
  if (n > KEPT_DIGITS) {
    const sticky = /[1-9]/.test(digits.slice(KEPT_DIGITS));
    e10 += n - KEPT_DIGITS - (sticky ? 1 : 0);
    digits = digits.slice(0, KEPT_DIGITS) + (sticky ? "1" : "");
  }
  const d = BigInt(digits);
  const num = e10 >= 0 ? d * 10n ** BigInt(e10) : d;
  const den = e10 >= 0 ? 1n : 10n ** BigInt(-e10);
  // e = floor(log2(num / den)), from the bit lengths.
  let e = num.toString(2).length - den.toString(2).length;
  if (e >= 0 ? num < den << BigInt(e) : num << BigInt(-e) < den) e -= 1;
  let q = Math.max(e, format.minExponent) - (format.mantissa - 1);
  const top = q >= 0 ? num : num << BigInt(-q);
  const bottom = q >= 0 ? den << BigInt(q) : den;
  let mantissa = top / bottom;
  const twice = 2n * (top - mantissa * bottom);
  if (twice > bottom || (twice === bottom && (mantissa & 1n) === 1n)) mantissa += 1n;
  if (mantissa === 1n << BigInt(format.mantissa)) {
    mantissa >>= 1n;
    q += 1;
  }
  if (q > format.maxExponent - (format.mantissa - 1)) return infinity;
  const value = Number(mantissa) * 2 ** q;
  return negative ? -value : value;
}

/**
 * Whether `s` is a decimal whose 32-bit float is in
 * `min..=max`, or above `min` with `above`: the bounds apply to
 * the rounded value, as the server applies them, so
 * `blur(512.00001)` is `blur(512)`.
 */
function f32In(s: string | null, min: number, max: number, above = false): boolean {
  if (s === null) return false;
  const value = readDecimal(s, 32);
  return value !== null && (above ? value > min : value >= min) && value <= max;
}

/**
 * The strict decimal of the `rotate` angle, the `convolution`
 * kernel and the `frame:` selector (url.rs parse_angle): an
 * optional minus where the value may be negative, digits, an
 * optional fraction; no `+`, no exponent, no leading or
 * trailing dot.
 */
const STRICT_DECIMAL = /^-?\d+(\.\d+)?$/;

function isKernelValue(s: string): boolean {
  // url.rs parse_kernel_value: a strict decimal, read as an
  // f64, at most 1024 in absolute value.
  if (!STRICT_DECIMAL.test(s)) return false;
  const value = readDecimal(s, 64);
  return value !== null && Math.abs(value) <= 1024;
}

function percentDecodable(s: string): string | null {
  // The server's escapes (url.rs decode_arg): `%` and two hex
  // digits, decoding to UTF-8. `decodeURIComponent` refuses
  // the rest, `%+1` included, as the server does (issue #712).
  try {
    return decodeURIComponent(s);
  } catch {
    return null;
  }
}

/**
 * Validate one filter against the server's rules (url.rs
 * parse_filters, ported). `args` is null when the filter has
 * no parentheses (empty parens count as none, like the
 * server) and the raw argument string otherwise. Returns an
 * error string, or null when the server would accept it.
 */
function validateFilter(name: string, args: string | null): string | null {
  const bad = `the server rejects this filter: ${name}${args === null ? "" : `(${args})`}`;
  if (NO_ARG_FILTERS.has(name)) {
    return args === null ? null : bad;
  }
  switch (name) {
    case "trim": {
      if (args === null) return null;
      const parts = args.split(",");
      if (parts.length === 1) return isUintIn(parts[0], 0, 255) ? null : bad;
      return parts.length === 5 &&
        isUintIn(parts[0], 0, 255) &&
        parts.slice(1).every((p) => isUintIn(p, 0, 4294967295))
        ? null
        : bad;
    }
    case "blur":
      // The server's blur bound (url.rs MAX_BLUR_SIGMA): a
      // normal f32, so at least 2^-126 (a subnormal sigma
      // panics the image crate's Gaussian setup), and at most
      // 512.
      return f32In(args, 2 ** -126, 512) ? null : bad;
    case "sharpen":
      return f32In(args, 0, F32_MAX, true) ? null : bad;
    case "brighten":
      return isIntIn(args, -255, 255) ? null : bad;
    case "huerotate":
      return isIntIn(args, -2147483648, 2147483647) ? null : bad;
    case "contrast":
      return f32In(args, -100, 100) ? null : bad;
    case "gamma":
      return f32In(args, 0, 10, true) ? null : bad;
    case "exposure":
      return f32In(args, -10, 10) ? null : bad;
    case "highlights_shadows": {
      if (args === null) return bad;
      const parts = args.split(",");
      return parts.length === 2 && parts.every((p) => f32In(p, -100, 100)) ? null : bad;
    }
    case "vibrance":
      return f32In(args, -100, 100) ? null : bad;
    case "saturation":
      return f32In(args, 0, 255) ? null : bad;
    case "quality": {
      if (args === null) return bad;
      if (args === "auto" || isUintIn(args, 1, 100)) return null;
      // Per-format list (issue #319): at most one bare
      // fallback, at most one entry per format, no video
      // formats (their quality is not the server's to set).
      const seen = new Set<string>();
      let fallbackSeen = false;
      for (const part of args.split(",")) {
        const colon = part.indexOf(":");
        if (colon === -1) {
          if (fallbackSeen || !(part === "auto" || isUintIn(part, 1, 100))) return bad;
          fallbackSeen = true;
        } else {
          const name = part.slice(0, colon);
          const value = part.slice(colon + 1);
          const canonical = name === "jpg" ? "jpeg" : name;
          if (!isOneOf(QUALITY_FORMATS, canonical)) return bad;
          if (seen.has(canonical)) return bad;
          seen.add(canonical);
          if (!(value === "auto" || isUintIn(value, 1, 100))) return bad;
        }
      }
      return seen.size + (fallbackSeen ? 1 : 0) > 0 ? null : bad;
    }
    case "palette":
      return isUintIn(args, 2, 256) ? null : bad;
    case "format": {
      if (args === null) return bad;
      // A format name plus its encoder options (issue #319),
      // each validated against the format it belongs to.
      const parts = args.split(",");
      const [name, ...options] = parts;
      if (!FORMAT_NAMES.has(name)) return bad;
      const seen = new Set<string>();
      for (const option of options) {
        const key = option.includes(":") ? option.slice(0, option.indexOf(":")) : option;
        if (seen.has(key)) return bad;
        seen.add(key);
        if (name === "jpeg" || name === "jpg") {
          if (option === "progressive") continue;
          if (JPEG_SUBSAMPLINGS.some((s) => option === `subsampling:${s}`)) continue;
          return bad;
        }
        if (name === "png") {
          if (option === "interlace") continue;
          return bad;
        }
        if (name === "webp") {
          // Each value is a u8 on the server (url.rs
          // parse_format_spec), so `method:+6` and `sns:050`
          // are spellings of 6 and 50.
          if (key === "method" && isUintIn(option.slice(7), 0, 6)) continue;
          if (key === "sns" && isUintIn(option.slice(4), 0, 100)) continue;
          return bad;
        }
        return bad;
      }
      return null;
    }
    case "rgb": {
      if (args === null) return bad;
      const parts = args.split(",");
      return parts.length === 3 && parts.every((p) => isIntIn(p, -255, 255)) ? null : bad;
    }
    case "curve": {
      if (args === null) return bad;
      const parts = args.split(",");
      if (!isOneOf(CURVE_CHANNELS, parts[0])) return bad;
      const coords = parts.slice(1);
      if (coords.length < 4 || coords.length > 32 || coords.length % 2 !== 0) return bad;
      if (!coords.every((p) => isUintIn(p, 0, 255))) return bad;
      for (let i = 2; i < coords.length; i += 2) {
        if (Number(coords[i]) <= Number(coords[i - 2])) return bad;
      }
      return null;
    }
    case "convolution": {
      if (args === null) return bad;
      const parts = args.split(",");
      if (parts.length !== 10 && parts.length !== 26) return bad;
      const flag = parts[parts.length - 1];
      if (flag !== "0" && flag !== "1") return bad;
      return parts.slice(0, -1).every((p) => isKernelValue(p)) ? null : bad;
    }
    case "noise":
      return isUintIn(args, 0, 100) ? null : bad;
    case "facepad":
      return f32In(args, 0, 3) ? null : bad;
    case "upscale":
      // A u32, then one of the factors.
      return isUintIn(args, 0, 4294967295) && isOneOf(UPSCALE_FACTORS, Number(args))
        ? null
        : bad;
    case "studio": {
      // The server's studio grammar (url.rs parse_studio):
      // the mode (`product` or `scene`) first, then named
      // `key:value` arguments, each key at most once.
      // Product takes `bg`, `shadow`, and `pad`; scene
      // takes `pad` only, because a backdrop and a shadow
      // are product concepts (naming one under scene is a
      // bad argument, not a silent no-op).
      if (args === null) return bad;
      const parts = args.split(",");
      const mode = parts[0];
      if (!isOneOf(STUDIO_MODES, mode)) return bad;
      const seen = new Set<string>();
      for (const part of parts.slice(1)) {
        const cut = part.indexOf(":");
        if (cut === -1) return bad;
        const key = part.slice(0, cut);
        const value = part.slice(cut + 1);
        if (seen.has(key)) return bad;
        seen.add(key);
        if (key === "pad") {
          // The pad parses as a Rust u8 (an optional `+`,
          // then digits; no minus sign, not even `-0`) and
          // is bounded to 40.
          if (!isUintIn(value, 0, 40)) return bad;
        } else if (mode === "product" && key === "bg") {
          if (!isOneOf(STUDIO_BACKDROPS, value) && !/^[0-9a-fA-F]{6}$/.test(value)) return bad;
        } else if (mode === "product" && key === "shadow") {
          if (!isOneOf(STUDIO_SHADOWS, value)) return bad;
        } else {
          return bad;
        }
      }
      return null;
    }
    case "dpr": {
      // An f32, then one of the factors: `dpr(1.50000001)`
      // rounds to 1.5.
      const factor = args === null ? null : readDecimal(args, 32);
      return factor !== null && isOneOf(DPR_FACTORS, factor) ? null : bad;
    }
    case "rotate":
      // url.rs parse_angle: a strict decimal, read as an f32,
      // at most 100 turns either way.
      return args !== null && STRICT_DECIMAL.test(args) && f32In(args, -36000, 36000)
        ? null
        : bad;
    case "fill":
      return args !== null && /^[0-9a-fA-F]{6}$/.test(args) ? null : bad;
    case "radius": {
      if (args === null) return bad;
      const parts = args.split(",");
      return (parts.length === 1 || parts.length === 4) &&
        parts.every((p) => isUintIn(p, 0, MAX_DIMENSION))
        ? null
        : bad;
    }
    case "border": {
      if (args === null) return bad;
      const parts = args.split(",");
      return parts.length === 2 &&
        isUintIn(parts[0], 0, MAX_DIMENSION) &&
        /^[0-9a-fA-F]{6}$/.test(parts[1])
        ? null
        : bad;
    }
    case "mask":
      return isOneOf(MASK_SHAPES, args) ? null : bad;
    case "duotone": {
      if (args === null) return bad;
      const parts = args.split(",");
      return parts.length === 2 && parts.every((p) => /^[0-9a-fA-F]{6}$/.test(p)) ? null : bad;
    }
    case "monochrome":
      return args !== null && /^[0-9a-fA-F]{6}$/.test(args) ? null : bad;
    case "pixelate":
      return isUintIn(args, 1, MAX_DIMENSION) ? null : bad;
    case "flip":
      return isOneOf(FLIP_AXES, args) ? null : bad;
    case "overlay": {
      if (args === null) return bad;
      // The server splits from the right (rsplitn(6, ',')):
      // with six or more fields the rightmost IS the width
      // and must parse, and the URL keeps any commas left
      // over. Splitting any other way misreads both.
      const fields = args.split(",");
      if (fields.length < 5) return bad;
      let rest = fields;
      if (fields.length >= 6) {
        if (!isUintIn(fields[fields.length - 1], 1, 400)) return bad;
        rest = fields.slice(0, -1);
      }
      const blend = rest[rest.length - 1];
      if (!isOneOf(BLEND_MODES, blend)) return bad;
      if (!isUintIn(rest[rest.length - 2], 0, 100)) return bad;
      if (!isIntIn(rest[rest.length - 3], -2147483648, 2147483647)) return bad;
      if (!isIntIn(rest[rest.length - 4], -2147483648, 2147483647)) return bad;
      const url = percentDecodable(rest.slice(0, -4).join(","));
      return url !== null && url !== "" ? null : bad;
    }
    case "blurclass":
    case "pixelateclass": {
      if (args === null) return bad;
      const decoded = percentDecodable(args);
      return decoded !== null && isClassName(trimLikeServer(decoded)) ? null : bad;
    }
    case "recolorclass": {
      if (args === null) return bad;
      const cut = args.lastIndexOf(",");
      if (cut < 0) return bad;
      if (!/^[0-9a-fA-F]{6}$/.test(args.slice(cut + 1))) return bad;
      const decoded = percentDecodable(args.slice(0, cut));
      return decoded !== null && isClassName(trimLikeServer(decoded)) ? null : bad;
    }
    case "blur_region":
    case "pixelate_region":
    case "remove_region": {
      if (args === null) return bad;
      const parts = args.split(",");
      return parts.length === 4 &&
        isUintIn(parts[0], 0, MAX_DIMENSION) &&
        isUintIn(parts[1], 0, MAX_DIMENSION) &&
        isUintIn(parts[2], 1, MAX_DIMENSION) &&
        isUintIn(parts[3], 1, MAX_DIMENSION)
        ? null
        : bad;
    }
    case "max_bytes":
      return isUintIn(args, 1, 18446744073709551615n) ? null : bad;
    case "text": {
      if (args === null) return bad;
      const parts = args.split(",");
      if (parts.length < 4) return bad;
      const anchor = parts[parts.length - 1];
      const color = parts[parts.length - 2];
      const size = parts[parts.length - 3];
      const text = percentDecodable(parts.slice(0, -3).join(","));
      return ANCHOR_NAMES.has(anchor) &&
        /^[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$/.test(color) &&
        isUintIn(size, 1, 1000) &&
        text !== null &&
        text !== "" &&
        textLabelFits(text, Number(size))
        ? null
        : bad;
    }
    case "watermark": {
      if (args === null) return bad;
      const parts = args.split(",");
      if (parts.length < 4) return bad;
      const alpha = parts[parts.length - 1];
      const y = parts[parts.length - 2];
      const x = parts[parts.length - 3];
      const url = percentDecodable(parts.slice(0, -3).join(","));
      return isUintIn(alpha, 0, 100) &&
        isIntIn(y, -2147483648, 2147483647) &&
        isIntIn(x, -2147483648, 2147483647) &&
        url !== null &&
        url !== ""
        ? null
        : bad;
    }
    default:
      return `unknown filter: ${name}`;
  }
}

/**
 * The filters of a `filters:` segment's list. The server splits
 * the list on the ':' separators that sit outside parentheses
 * (url.rs split_filter_chain): an argument list may itself
 * contain ':' (the studio filter's named arguments are
 * `key:value`), so a naive split would cut those filters apart.
 * Unbalanced parentheses leave the tail in one part, which then
 * fails its own parse, as on the server.
 */
function filterChain(rest: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < rest.length; i++) {
    const c = rest[i];
    if (c === "(") {
      depth += 1;
    } else if (c === ")") {
      depth = Math.max(0, depth - 1);
    } else if (c === ":" && depth === 0) {
      parts.push(rest.slice(start, i));
      start = i + 1;
    }
  }
  parts.push(rest.slice(start));
  return parts;
}

function explainFilters(rest: string): { detail?: string; error?: string } {
  const blurbs = [];
  for (const part of filterChain(rest)) {
    const match = part.match(/^([a-z_0-9]+)(?:\((.*)\))?$/);
    if (!match) {
      return { error: `malformed filter: ${part}` };
    }
    const [, name, rawArgs] = match;
    const args = rawArgs === undefined || rawArgs === "" ? null : rawArgs;
    const blurb = FILTER_DESCRIPTIONS[name];
    if (!blurb) {
      return { error: `unknown filter: ${name}` };
    }
    const invalid = validateFilter(name, args);
    if (invalid) {
      return { error: invalid };
    }
    blurbs.push(args !== null ? `${name}(${args}): ${blurb}` : `${name}: ${blurb}`);
  }
  return { detail: blurbs.join("; ") };
}

/** The most operations one request runs (url.rs MAX_OPERATIONS). */
export const MAX_OPERATIONS = 100;

/**
 * The most distinct watermark and overlay URLs one request
 * names (url.rs MAX_WATERMARK_URLS, issue #613).
 */
export const MAX_WATERMARK_URLS = 16;

/**
 * The decoded image URL a valid `watermark(...)` or
 * `overlay(...)` filter composites, or null for any other
 * filter. Both split from the right, as validateFilter reads
 * them.
 */
function compositedUrl(filter: string): string | null {
  const match = filter.match(/^(watermark|overlay)\((.*)\)$/);
  if (!match) return null;
  const fields = match[2].split(",");
  const numeric = match[1] === "watermark" ? 3 : fields.length >= 6 ? 5 : 4;
  return percentDecodable(fields.slice(0, -numeric).join(","));
}

/**
 * The caps the server applies once a request parses (url.rs
 * check_operation_count and check_watermark_count, issues #469
 * and #613): at most {@link MAX_OPERATIONS} operations, each
 * filter in a `filters:` segment counting once, and at most
 * {@link MAX_WATERMARK_URLS} distinct watermark and overlay
 * URLs, compared as decoded. A preset reference counts as one
 * operation: its definition holds at least one, and the
 * explainer sees neither the definition nor the server's
 * default operations, so both counts are lower bounds, and a
 * cap they pass is one the server refuses too.
 */
function capErrors(segments: ExplainedSegment[]): string[] {
  let operations = 0;
  let presets = false;
  const urls = new Set<string>();
  for (const { raw, kind } of segments) {
    if (kind === "expiry" || kind === "maxage") continue;
    presets ||= kind === "preset";
    if (kind === "operation" && raw.startsWith("filters:")) {
      for (const filter of filterChain(raw.slice(8))) {
        operations += 1;
        const url = compositedUrl(filter);
        if (url !== null) urls.add(url);
      }
    } else {
      operations += 1;
    }
  }
  const errors: string[] = [];
  const atLeast = presets ? "at least " : "";
  if (operations > MAX_OPERATIONS) {
    errors.push(
      `the request has ${atLeast}${operations} operations, over the ${MAX_OPERATIONS} operation limit: the server answers 400 (unsupported_operation)`,
    );
  } else if (urls.size > MAX_WATERMARK_URLS) {
    errors.push(
      `the request names ${urls.size} distinct watermark and overlay URLs, over the ${MAX_WATERMARK_URLS} URL limit: the server answers 400 (unsupported_operation)`,
    );
  }
  return errors;
}

/**
 * `<left>x<top>:<right>x<bottom>` as url.rs parse_rect_points
 * reads it (crop-rect and focal): four u32s, with the right
 * edge past the left and the bottom past the top.
 */
function isRectPoints(rest: string): boolean {
  const m = rest.match(/^(\+?\d+)x(\+?\d+):(\+?\d+)x(\+?\d+)$/);
  if (!m || !m.slice(1).every((n) => isUintIn(n, 0, 4294967295))) return false;
  return BigInt(m[3]) > BigInt(m[1]) && BigInt(m[4]) > BigInt(m[2]);
}

const ALIGN_NAMES = new Set<string>(CROP_ALIGNS);
const U64_MAX = 18446744073709551615n;

/**
 * The objects crop alignment (url.rs parse_objects, ported):
 * `objects(<class>[,<class>...])[:<scale>]`. Returns the
 * decoded class names (duplicates folded ASCII-only, first
 * spelling kept, as the server's canonical form folds them)
 * and the scale, or null when the server would reject the
 * form. The 32-class cap counts distinct names, as the
 * server's does: repeats are free, and a 33rd distinct name
 * rejects the form (issue #408).
 */
function parseObjectsAlign(align: string): { classes: string[]; scale: number } | null {
  if (!align.startsWith("objects(")) return null;
  const inner = align.slice("objects(".length);
  const close = inner.indexOf(")");
  if (close < 0) return null;
  const list = inner.slice(0, close);
  const tail = inner.slice(close + 1);
  let scale = 1;
  if (tail !== "") {
    if (!tail.startsWith(":")) return null;
    const text = tail.slice(1);
    // An f32 above 0 and at most 100, the bound applied to
    // the rounded value, as on the server.
    if (!f32In(text, 0, 100, true)) return null;
    scale = Number(text);
  }
  const parts = list.split(",");
  const classes: string[] = [];
  const seen = new Set<string>();
  for (const part of parts) {
    const decoded = percentDecodable(part);
    if (decoded === null) return null;
    const name = trimLikeServer(decoded);
    if (!isClassName(name)) return null;
    const key = classFoldKey(name);
    if (!seen.has(key)) {
      if (classes.length === MAX_OBJECTS_CLASSES) return null;
      seen.add(key);
      classes.push(name);
    }
  }
  return classes.length > 0 ? { classes, scale } : null;
}

function explainSegment(seg: string, nowSeconds: number): Omit<ExplainedSegment, "raw"> | null {
  if (seg.startsWith("exp:")) {
    const value = seg.slice(4);
    if (!/^\d+$/.test(value)) {
      return { kind: "expiry", title: "Expiry", error: `malformed expiry: ${seg}` };
    }
    // Parse as BigInt: the server reads a u64, and a huge
    // value must report its clean 400, not crash the date
    // arithmetic (audit P-3).
    const seconds = BigInt(value);
    if (seconds > U64_MAX) {
      return {
        kind: "expiry",
        title: "Expiry",
        error: `expiry overflows the server's u64 parser: ${seg} (the server answers 400)`,
      };
    }
    const expired = BigInt(nowSeconds) >= seconds;
    const asNumber = Number(seconds);
    const when =
      asNumber <= 8640000000000
        ? new Date(asNumber * 1000).toISOString().replace("T", " ").replace(".000Z", " UTC")
        : "a date beyond representation";
    return {
      kind: "expiry",
      title: "Expiry",
      detail: `valid strictly before ${when} (${value})${expired ? " — EXPIRED at the reference time" : ""}`,
      expired,
    };
  }
  if (seg.startsWith("maxage:")) {
    const value = seg.slice(7);
    if (!/^\d+$/.test(value)) {
      return {
        kind: "maxage",
        title: "Max-age",
        error: `malformed maxage: ${seg} (want maxage:<seconds> from 0 to 31536000)`,
      };
    }
    if (BigInt(value) > 31536000n) {
      return {
        kind: "maxage",
        title: "Max-age",
        error: `maxage out of range: ${seg} (0 to 31536000 seconds; the server answers 400)`,
      };
    }
    return {
      kind: "maxage",
      title: "Max-age",
      detail:
        value === "0"
          ? "this response is served with Cache-Control: public, max-age=0 and no immutable flag: caches must revalidate before reuse; the stored image is unchanged"
          : `this response is served with Cache-Control max-age ${value} seconds instead of the default year; the stored image is unchanged`,
    };
  }
  if (seg.startsWith("debug:")) {
    const value = seg.slice(6);
    if (value !== "focal") {
      return {
        kind: "debug",
        title: "Debug view",
        error: `unsupported debug view: ${seg} (want debug:focal; the server answers 400)`,
      };
    }
    return {
      kind: "debug",
      title: "Debug view",
      detail:
        "the response is a diagnostic image instead of the transform result: the canvas the next crop operates on, with the chosen focal point (magenta crosshair), the focal rectangle when one is in play, the detection boxes that fed the decision (amber), and the crop window outline (blue); with no crop it shows the focal analysis of the final image; debug responses are never cached and are served with Cache-Control: no-store; animated sources and video formats are refused with 400",
    };
  }
  if (seg.startsWith("frame:")) {
    const rest = seg.slice(6);
    // The server's frame grammar (url.rs parse_frame): the
    // poster keyword, or a strict decimal (no sign, no
    // exponent, no leading dot) read as an f64, as seconds up
    // to 86400 or, with a % suffix, as a percent up to 100.
    const decimal = (text: string, max: number): boolean => {
      if (!/^\d+(\.\d+)?$/.test(text)) return false;
      const value = readDecimal(text, 64);
      return value !== null && value <= max;
    };
    if (rest === "poster") {
      return { kind: "operation", title: "Frame", detail: "render the video source's poster frame" };
    }
    if (rest.endsWith("%") && decimal(rest.slice(0, -1), 100)) {
      return { kind: "operation", title: "Frame", detail: `render the frame ${rest} through the video source` };
    }
    if (!rest.endsWith("%") && decimal(rest, 86400)) {
      return { kind: "operation", title: "Frame", detail: `render the frame at ${rest} seconds into the video source` };
    }
    return {
      kind: "operation",
      title: "Frame",
      error: `malformed frame segment: ${seg} (want frame:poster, frame:<seconds> up to 86400, or frame:<percent>% up to 100; the server answers 400)`,
    };
  }
  if (seg.startsWith("page:")) {
    const rest = seg.slice(5);
    // The server's page grammar (url.rs parse_page): ASCII
    // digits, leading zeros allowed, 1 to 999999.
    const digits = rest.replace(/^0+/, "");
    if (!/^\d+$/.test(rest) || digits === "" || digits.length > 6) {
      return {
        kind: "operation",
        title: "Page",
        error: `malformed page segment: ${seg} (want page:<n> with n from 1 to 999999; the server answers 400)`,
      };
    }
    return { kind: "operation", title: "Page", detail: `render page ${Number(digits)} of the PDF source` };
  }
  if (seg.startsWith("preset:")) {
    const name = seg.slice(7);
    if (!name) {
      return { kind: "preset", title: "Preset", error: "empty preset name" };
    }
    return {
      kind: "preset",
      title: "Preset",
      detail: `expands server-side to the operations the operator named "${name}"; the signature covers this reference, not the expansion`,
    };
  }
  if (seg.startsWith("crop-rect:")) {
    const rest = seg.slice(10);
    if (!isRectPoints(rest)) {
      return {
        kind: "operation",
        title: "Crop rectangle",
        error: `malformed crop-rect segment: ${seg} (want crop-rect:<l>x<t>:<r>x<b> with right > left and bottom > top)`,
      };
    }
    return { kind: "operation", title: "Crop rectangle", detail: `crop the exact rectangle ${rest} (left x top : right x bottom)` };
  }
  if (seg.startsWith("focal:")) {
    const rest = seg.slice(6);
    if (!isRectPoints(rest)) {
      return {
        kind: "operation",
        title: "Focal rectangle",
        error: `malformed focal segment: ${seg} (want focal:<l>x<t>:<r>x<b> with right > left and bottom > top)`,
      };
    }
    return { kind: "operation", title: "Focal rectangle", detail: `set the focal rectangle ${rest} for the next crop (left x top : right x bottom)` };
  }
  if (seg.startsWith("focalpoint:")) {
    const rest = seg.slice(11);
    const colon = rest.indexOf(":");
    const point = colon === -1 ? rest : rest.slice(0, colon);
    const zoomText = colon === -1 ? null : rest.slice(colon + 1);
    const xy = point.split("x");
    // Each number an f32, the bounds applied to the rounded
    // value (url.rs parse_focal_point).
    const ok =
      xy.length === 2 &&
      f32In(xy[0], 0, 1) &&
      f32In(xy[1], 0, 1) &&
      (zoomText === null || f32In(zoomText, 0, 100, true));
    if (!ok) {
      return {
        kind: "operation",
        title: "Focal point",
        error: `malformed focalpoint segment: ${seg} (want focalpoint:<x>x<y>[:<zoom>] with x and y in 0..1 and zoom in (0, 100])`,
      };
    }
    return {
      kind: "operation",
      title: "Focal point",
      detail: `set the focal point ${point} at zoom ${zoomText ?? "1"} for the next crop (fractions of the image dimensions; zoom above 1 tightens the crop around the point)`,
    };
  }
  if (seg.startsWith("crop:")) {
    const rest = seg.slice(5);
    const idx = rest.indexOf(":");
    const dims = idx === -1 ? rest : rest.slice(0, idx);
    const align = idx === -1 ? "center" : rest.slice(idx + 1);
    const cropSize = parseSizeSegment(dims);
    const objects = align.startsWith("objects") ? parseObjectsAlign(align) : null;
    if (!cropSize || !isExactBox(cropSize) || (!ALIGN_NAMES.has(align) && objects === null)) {
      return {
        kind: "operation",
        title: "Crop",
        error: `malformed crop segment: ${seg} (want crop:<w>x<h>[:center|smart|face|top|bottom|left|right|top-left|top-right|bottom-left|bottom-right|objects(<class>[,<class>...])[:<scale>]], pixels only, no flips)`,
      };
    }
    if (objects !== null) {
      return {
        kind: "operation",
        title: "Crop",
        detail: `crop to the aspect of ${dims} and resize to it; alignment: objects (${objects.classes.join(", ")}): frame the union of every detection of the named classes, the union box scaled by ${objects.scale} about its center and the crop centered on it; needs a detection model on the server, and falls back to the plain center crop of the same size when none of the classes is detected`,
      };
    }
    return { kind: "operation", title: "Crop", detail: `crop to the aspect of ${dims} and resize to it; alignment: ${align === "center" ? "center (or the focal point for smart)" : align}` };
  }
  if (seg.startsWith("fit-in:")) {
    const rest = seg.slice(7);
    const idx = rest.indexOf(":");
    const dims = idx === -1 ? rest : rest.slice(0, idx);
    const modifier = idx === -1 ? null : rest.slice(idx + 1);
    const size = parseSizeSegment(dims);
    if (!size || size.w.flip || size.h.flip || (modifier !== null && modifier !== "adaptive")) {
      return { kind: "operation", title: "Fit in", error: `malformed fit-in segment: ${seg} (want fit-in:<w>x<h>[:adaptive], flips are not allowed here)` };
    }
    if (modifier === "adaptive") {
      return { kind: "operation", title: "Fit inside a box", detail: `scale proportionally to fit inside ${dims}, swapping the box axes when the source orientation disagrees (adaptive)` };
    }
    return { kind: "operation", title: "Fit inside a box", detail: `scale proportionally to fit inside ${rest}` };
  }
  if (seg.startsWith("pad:")) {
    const rest = seg.slice(4);
    const idx = rest.indexOf(":");
    const dims = idx === -1 ? rest : rest.slice(0, idx);
    const fill = idx === -1 ? null : rest.slice(idx + 1);
    const padSize = parseSizeSegment(dims);
    if (!padSize || !isExactBox(padSize) || (fill !== null && fill !== "blur" && !/^[0-9a-fA-F]{6}$/.test(fill))) {
      return { kind: "operation", title: "Pad", error: `malformed pad segment: ${seg} (want pad:<w>x<h>[:<hex>|blur], pixels only, no flips)` };
    }
    return { kind: "operation", title: "Pad", detail: `fit inside ${dims} and fill the canvas to exactly that size, fill: ${fill ?? "the fill(...) color or black"}` };
  }
  if (seg.startsWith("filters:")) {
    const { detail, error } = explainFilters(seg.slice(8));
    if (error) {
      return { kind: "operation", title: "Filters", error };
    }
    return { kind: "operation", title: "Filters", detail };
  }
  if (parseSizeSegment(seg)) {
    return { kind: "operation", title: "Resize", detail: sizeBlurb(seg) };
  }
  if (isSizeShaped(seg)) {
    // The server's dispatch (url.rs size_shaped, issue #228):
    // a segment shaped like a size that fails the size
    // grammar is a malformed size, not the start of the
    // source, so the request is a 400 rather than a fetch
    // of a nonsense source.
    return {
      kind: "operation",
      title: "Resize",
      error: `malformed size segment: ${seg} (each side is a pixel count up to ${MAX_DIMENSION}, 0, or orig; the server answers 400)`,
    };
  }
  return null; // Not an operation: the source starts here.
}

/**
 * Whether the server reads `seg` ahead of the source (issue
 * #782): as `exp:` or `maxage:` metadata, a `preset:`
 * reference, or an operation, valid or not. The first segment
 * that is none of these starts the source. The builders refuse
 * a source that begins with one when they are given an expiry.
 */
export function readsBeforeSource(seg: string): boolean {
  return seg !== "" && explainSegment(seg, 0) !== null;
}

/**
 * The server's size-shape test (url.rs size_shaped), ported:
 * an `x` joining two parts, at least one of them a dimension
 * spelling (digits, `orig`, or empty, with an optional flip
 * sign), and no `.` or `:` anywhere, the punctuation that
 * marks URL text.
 */
function isSizeShaped(seg: string): boolean {
  const x = seg.indexOf("x");
  if (x === -1) return false;
  const left = seg.slice(0, x);
  const right = seg.slice(x + 1);
  if (left.includes(".") || right.includes(".") || seg.includes(":")) return false;
  const spelling = (part: string): boolean => {
    const digits = part.startsWith("-") ? part.slice(1) : part;
    return digits === "" || digits === "orig" || /^\d+$/.test(digits);
  };
  return spelling(left) || spelling(right);
}

/**
 * The most input explainUrl reads, in UTF-8 bytes: 64 KiB. It
 * bounds the work one call does, and it sits above the URL
 * limits of common CDNs and proxies.
 */
const MAX_EXPLAIN_BYTES = 64 * 1024;

function utf8Length(text: string): number {
  // UTF-8 never takes fewer bytes than UTF-16 code units, so
  // the caller checks .length first and this only runs on
  // input of at most MAX_EXPLAIN_BYTES code units.
  return new TextEncoder().encode(text).length;
}

/**
 * Parse a crabcrop URL (or bare path) back into its pieces,
 * explaining each segment. Mirrors the server's dispatch:
 * segments are operations until the first segment that is not
 * one; everything from there is the source.
 *
 * Returns { signature, segments, source, remainder, errors }
 * where segments carry { raw, kind, title, detail?, error? }.
 *
 * Input longer than MAX_EXPLAIN_BYTES is not read: the answer
 * is one error and no signature, so verifyUrlSignature returns
 * null for it.
 */
export function explainUrl(
  input: string,
  nowSeconds: number = Math.floor(Date.now() / 1000),
): Explanation {
  if (input.length > MAX_EXPLAIN_BYTES || utf8Length(input) > MAX_EXPLAIN_BYTES) {
    return {
      signature: null,
      segments: [],
      source: "",
      remainder: "",
      errors: [`the URL is longer than 64 KiB (${MAX_EXPLAIN_BYTES} bytes): the explainer does not read it`],
    };
  }
  // Extract the path by string surgery, never via the URL
  // parser: WHATWG parsing resolves dot segments (.. and
  // %2e%2e) that the server treats as literal path bytes
  // (audit P-3). A query or fragment is not part of the
  // signed path, on either side.
  let path = trimLikeServer(input);
  const scheme = path.indexOf("://");
  if (scheme > 0 && /^[a-zA-Z][a-zA-Z0-9+.-]*$/.test(path.slice(0, scheme))) {
    const afterAuthority = path.slice(scheme + 3);
    const slash = afterAuthority.indexOf("/");
    path = slash === -1 ? "" : afterAuthority.slice(slash + 1);
  } else if (path.startsWith("/")) {
    path = path.slice(1);
  }
  const cut = path.search(/[?#]/);
  if (cut !== -1) {
    path = path.slice(0, cut);
  }
  const errors: string[] = [];
  // A sealed URL is a single segment: the token carries
  // the whole remainder encrypted, so there is no slash
  // and nothing visible to itemize (issue #388).
  const sealedWhole = path.match(/^(?:([A-Za-z0-9_-]{1,32}):)?s1\.([A-Za-z0-9_-]+)$/);
  if (sealedWhole) {
    const signature: Explanation["signature"] = {
      kind: "sealed",
      value: `s1.${sealedWhole[2]}`,
      detail:
        "sealed token (AES-256-GCM): the operations and source are encrypted inside it, so there is nothing visible to itemize; the server opens it and serves the URL it carries",
    };
    if (sealedWhole[1]) {
      signature.keyId = sealedWhole[1];
    }
    return { signature, segments: [], source: "", remainder: "", errors: [] };
  }
  const slash = path.indexOf("/");
  if (slash === -1) {
    return { signature: null, segments: [], source: "", remainder: "", errors: ["no signature segment: the path is a single segment"] };
  }
  const signatureRaw = path.slice(0, slash);
  const remainder = path.slice(slash + 1);
  let signature: Explanation["signature"];
  const keyed = signatureRaw.match(/^([A-Za-z0-9_-]{1,32}):([0-9a-f]{64})$/);
  const sealedTok = signatureRaw.match(/^(?:([A-Za-z0-9_-]{1,32}):)?s1\.([A-Za-z0-9_-]+)$/);
  if (sealedTok) {
    // A sealed token with a trailing path (issue #388):
    // the server opens the token and serves the URL it
    // carries; the visible remainder behind it is not
    // part of that URL, so there is nothing to itemize.
    signature = {
      kind: "sealed",
      value: `s1.${sealedTok[2]}`,
      detail:
        "sealed token (AES-256-GCM): the operations and source are encrypted inside it; the server opens it and serves the URL it carries, and anything visible after the token is not part of that URL",
    };
    if (sealedTok[1]) {
      signature.keyId = sealedTok[1];
    }
    return { signature, segments: [], source: "", remainder, errors };
  }
  if (signatureRaw === "unsafe") {
    signature = { kind: "unsafe", value: signatureRaw, detail: "unsigned; the server honors it only with CRABCROP_ALLOW_UNSAFE or in presets-only deployments" };
  } else if (keyed) {
    signature = { kind: "hmac", value: keyed[2], keyId: keyed[1], detail: `HMAC-SHA256 of the remainder, signed with key "${keyed[1]}"; the server verifies it against that key only (secret rotation)` };
  } else if (/^[0-9a-f]{64}$/.test(signatureRaw)) {
    signature = { kind: "hmac", value: signatureRaw, detail: "HMAC-SHA256 of the remainder; paste the secret to verify it below" };
  } else if (/^[0-9a-f]+$/.test(signatureRaw)) {
    signature = { kind: "invalid", value: signatureRaw, detail: "lowercase hex, but not a full 64-character HMAC tag: the server will answer 403" };
    errors.push("malformed signature segment");
  } else if (/^(?:[A-Za-z0-9_-]{1,32}:)?s\d+\./.test(signatureRaw)) {
    signature = { kind: "invalid", value: signatureRaw, detail: "a sealed-token shape this scheme does not open (an unknown version marker, or a malformed token body): the server answers the uniform 403 refusal, indistinguishable from a token that fails to open" };
    errors.push("malformed signature segment");
  } else {
    signature = { kind: "invalid", value: signatureRaw, detail: "not `unsafe` and not lowercase hex: the server rejects it as malformed (400)" };
    errors.push("malformed signature segment");
  }
  // Remainders the server refuses before it reads an
  // operation, whatever the signature (server/admission.rs
  // resolve_request): the segment key derivation owns, and
  // the endpoints' own segments.
  const head = remainder.split("/", 1)[0];
  if (isReservedRemainder(remainder)) {
    errors.push(
      "the remainder begins with the reserved segment crabcrop, which key derivation owns: the server answers 400 (reserved_segment)",
    );
  } else if (head === "purge" || head === "purge-group" || head === "upload") {
    errors.push(
      "an image URL may not begin with a purge or upload segment: the server answers 400 (reserved_segment)",
    );
  }

  const segments: ExplainedSegment[] = [];
  let rest = remainder;
  let source = "";
  for (;;) {
    const idx = rest.indexOf("/");
    const seg = idx === -1 ? rest : rest.slice(0, idx);
    if (seg === "" && idx !== -1) {
      // An empty segment in the operations run (a doubled
      // slash): the server answers 400 (missing source URL)
      // instead of treating the rest as a source with a
      // leading slash (AP-23).
      const error = "empty segment in the path: the server answers 400 (missing source URL)";
      segments.push({ raw: seg, kind: "operation", title: "Empty segment", error });
      errors.push(error);
      rest = rest.slice(idx + 1);
      continue;
    }
    const explained = seg === "" ? null : explainSegment(seg, nowSeconds);
    if (!explained) {
      source = rest;
      break;
    }
    segments.push({ raw: seg, ...explained });
    if (explained.error) {
      errors.push(explained.error);
    }
    if (idx === -1) {
      errors.push("the URL ends after the operations: no source");
      rest = "";
      break;
    }
    rest = rest.slice(idx + 1);
  }
  if (!source && rest) {
    source = rest;
  }
  // The caps apply to a request that parses, as on the server.
  if (errors.length === 0) {
    errors.push(...capErrors(segments));
  }
  return { signature, segments, source, remainder, errors };
}
