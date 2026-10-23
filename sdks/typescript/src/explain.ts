/**
 * Parse a crabcrop URL (or bare path) back into its pieces and
 * explain each segment, mirroring the server's grammar. This is
 * the canonical implementation: the playground keeps a plain-JS
 * copy for its no-build static page, and both are pinned to the
 * same verdict corpus (audit PL-10, P-3).
 */

export interface ExplainedSegment {
  raw: string;
  kind: string;
  title: string;
  detail?: string;
  error?: string;
  expired?: boolean;
}

export interface Explanation {
  signature: { kind: string; value: string; detail: string } | null;
  segments: ExplainedSegment[];
  source: string;
  remainder: string;
  errors: string[];
}

const FILTER_BLURBS = {
  grayscale: "convert to grayscale",
  invert: "invert the colors",
  no_upscale: "never enlarge the source",
  blur: "gaussian blur, argument is the sigma",
  blurfaces: "blur detected faces (needs a face model on the server)",
  brighten: "brighten by the given amount (-255..255)",
  contrast: "contrast multiplier (1 is identity)",
  huerotate: "rotate the hue by the given degrees",
  rotate: "rotate by a multiple of 90 degrees",
  quality: "output quality 0..100, or `auto` for perceptual targeting",
  format: "force the output format",
  fill: "the background color other operations pad with",
  facepad: "how much a face crop expands around the face",
  pixelatefaces: "pixelate detected faces (needs a face model on the server)",
  blurclass: "blur every detected object of the named class (needs a detection model on the server)",
  pixelateclass:
    "pixelate every detected object of the named class (needs a detection model on the server)",
  removebg: "remove the background (needs an AI model on the server)",
  upscale: "AI upscale by the given factor (needs an AI model on the server)",
  equalize: "histogram equalization",
  saturation: "saturation factor (0 is grayscale, 1 is identity)",
  sharpen: "sharpen by the given amount",
  text: "draw a text overlay (needs a font on the server)",
  rgb: "tint toward the given red/green/blue amounts",
  trim: "trim uniform borders",
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
};

const MAX_DIMENSION = 16383;

/**
 * The server's size grammar (url.rs parse_size/parse_dimension),
 * ported (audit P-3): each side is an optional `-` flip marker
 * plus `` (proportional), `0`, `orig`, or a pixel count up to
 * the dimension cap. Returns null when the segment is not a
 * size, which in the server's dispatch means the source starts.
 */
function parseDimension(part) {
  let flip = false;
  let digits = part;
  if (digits.startsWith("-")) {
    flip = true;
    digits = digits.slice(1);
  }
  if (digits === "" || digits === "0") return { kind: "proportional", flip };
  if (digits === "orig") return { kind: "original", flip };
  if (!/^\d+$/.test(digits)) return null;
  if (BigInt(digits) > BigInt(MAX_DIMENSION)) return null;
  return { kind: "pixels", value: Number(digits), flip };
}

function parseSizeSegment(seg) {
  const x = seg.indexOf("x");
  if (x === -1) return null;
  const w = parseDimension(seg.slice(0, x));
  const h = parseDimension(seg.slice(x + 1));
  if (!w || !h) return null;
  return { w, h };
}

function sizeBlurb(seg) {
  const size = parseSizeSegment(seg);
  if (!size) return `resize per ${seg}`;
  const describe = (d, axis) =>
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

const NO_ARG_FILTERS = new Set([
  "grayscale",
  "invert",
  "no_upscale",
  "equalize",
  "removebg",
  "sepia",
  "blurfaces",
  "pixelatefaces",
]);
const FORMAT_NAMES = new Set(["jpeg", "jpg", "png", "webp", "jxl", "avif", "gif"]);
const ANCHOR_NAMES = new Set([
  "top-left",
  "top-center",
  "top-right",
  "center-left",
  "center",
  "center-right",
  "bottom-left",
  "bottom-center",
  "bottom-right",
]);

function isIntIn(s, min, max) {
  return /^-?\d+$/.test(s) && BigInt(s) >= BigInt(min) && BigInt(s) <= BigInt(max);
}

function isFloat(s) {
  return s !== "" && Number.isFinite(Number(s));
}

function percentDecodable(s) {
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
function validateFilter(name, args) {
  const bad = `the server rejects this filter: ${name}${args === null ? "" : `(${args})`}`;
  if (NO_ARG_FILTERS.has(name)) {
    return args === null ? null : bad;
  }
  switch (name) {
    case "trim":
      return args === null || isIntIn(args, 0, 255) ? null : bad;
    case "blur":
    case "sharpen":
      return args !== null && isFloat(args) && Number(args) > 0 ? null : bad;
    case "brighten":
    case "huerotate":
      return args !== null && isIntIn(args, -2147483648, 2147483647) ? null : bad;
    case "contrast":
      return args !== null && isFloat(args) ? null : bad;
    case "saturation":
      return args !== null && isFloat(args) && Number(args) >= 0 ? null : bad;
    case "quality":
      return args === "auto" || (args !== null && isIntIn(args, 0, 100)) ? null : bad;
    case "format":
      return FORMAT_NAMES.has(args) ? null : bad;
    case "rgb": {
      if (args === null) return bad;
      const parts = args.split(",");
      return parts.length === 3 && parts.every((p) => isIntIn(p, -255, 255)) ? null : bad;
    }
    case "facepad":
      return args !== null && isFloat(args) && Number(args) >= 0 && Number(args) <= 3
        ? null
        : bad;
    case "upscale":
      return args !== null && /^\d+$/.test(args) && (Number(args) === 2 || Number(args) === 4)
        ? null
        : bad;
    case "dpr":
      return args !== null && isFloat(args) && [1, 1.5, 2, 3].includes(Number(args))
        ? null
        : bad;
    case "rotate":
      return args !== null && isIntIn(args, -2147483648, 2147483647) && Number(args) % 90 === 0
        ? null
        : bad;
    case "fill":
      return args !== null && /^[0-9a-fA-F]{6}$/.test(args) ? null : bad;
    case "radius": {
      if (args === null) return bad;
      const parts = args.split(",");
      return (parts.length === 1 || parts.length === 4) &&
        parts.every((p) => /^\d+$/.test(p) && BigInt(p) <= 16383n)
        ? null
        : bad;
    }
    case "border": {
      if (args === null) return bad;
      const parts = args.split(",");
      return parts.length === 2 &&
        /^\d+$/.test(parts[0]) &&
        BigInt(parts[0]) >= 1n &&
        BigInt(parts[0]) <= 16383n &&
        /^[0-9a-fA-F]{6}$/.test(parts[1])
        ? null
        : bad;
    }
    case "mask":
      return args === "circle" || args === "ellipse" ? null : bad;
    case "duotone": {
      if (args === null) return bad;
      const parts = args.split(",");
      return parts.length === 2 && parts.every((p) => /^[0-9a-fA-F]{6}$/.test(p)) ? null : bad;
    }
    case "monochrome":
      return args !== null && /^[0-9a-fA-F]{6}$/.test(args) ? null : bad;
    case "pixelate":
      return args !== null && /^\d+$/.test(args) && BigInt(args) >= 1n && BigInt(args) <= 16383n
        ? null
        : bad;
    case "flip":
      return args === "h" || args === "v" ? null : bad;
    case "overlay": {
      if (args === null) return bad;
      const rest = args.split(",");
      if (rest.length < 5) return bad;
      let width: string | null = null;
      if (rest.length >= 6 && /^\d+$/.test(rest[rest.length - 1])) width = rest.pop()!;
      const blend = rest.pop();
      if (!["normal", "multiply", "screen", "overlay"].includes(blend!)) return bad;
      if (width !== null && (BigInt(width) < 1n || BigInt(width) > 400n)) return bad;
      const alpha = rest.pop()!;
      if (!/^\d+$/.test(alpha) || BigInt(alpha) > 100n) return bad;
      const y = rest.pop()!;
      const x = rest.pop()!;
      if (!/^-?\d+$/.test(x) || !/^-?\d+$/.test(y)) return bad;
      return rest.join(",").length > 0 ? null : bad;
    }
    case "blurclass":
    case "pixelateclass": {
      if (args === null) return bad;
      const decoded = percentDecodable(args);
      if (decoded === null) return bad;
      const name = decoded.trim();
      return name.length >= 1 &&
        name.length <= 64 &&
        /^[\p{L}\p{N} _.\-/]+$/u.test(name)
        ? null
        : bad;
    }
    case "blur_region":
    case "pixelate_region": {
      if (args === null) return bad;
      const parts = args.split(",");
      return parts.length === 4 &&
        parts.every((p) => /^\d+$/.test(p) && BigInt(p) <= 16383n) &&
        BigInt(parts[2]) >= 1n &&
        BigInt(parts[3]) >= 1n
        ? null
        : bad;
    }
    case "max_bytes":
      return args !== null &&
        /^\d+$/.test(args) &&
        BigInt(args) >= 1n &&
        BigInt(args) <= 18446744073709551615n
        ? null
        : bad;
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
        isIntIn(size, 1, 1000) &&
        text !== null &&
        text !== ""
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
      return isIntIn(alpha, 0, 100) &&
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

function explainFilters(rest) {
  // The server splits the list naively on ':'; a ':' inside
  // parentheses is not special there either.
  const parts = rest.split(":");
  const blurbs = [];
  for (const part of parts) {
    const match = part.match(/^([a-z_0-9]+)(?:\((.*)\))?$/);
    if (!match) {
      return { error: `malformed filter: ${part}` };
    }
    const [, name, rawArgs] = match;
    const args = rawArgs === undefined || rawArgs === "" ? null : rawArgs;
    const blurb = FILTER_BLURBS[name];
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

const ALIGN_NAMES = new Set(["center", "smart", "face", "top", "bottom", "left", "right"]);
const U64_MAX = 18446744073709551615n;

function explainSegment(seg, nowSeconds) {
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
    const m = rest.match(/^(\d+)x(\d+):(\d+)x(\d+)$/);
    if (!m || m.slice(1).some((n) => BigInt(n) > 4294967295n)) {
      return {
        kind: "operation",
        title: "Crop rectangle",
        error: `malformed crop-rect segment: ${seg} (want crop-rect:<l>x<t>:<r>x<b>)`,
      };
    }
    return { kind: "operation", title: "Crop rectangle", detail: `crop the exact rectangle ${rest} (left x top : right x bottom)` };
  }
  if (seg.startsWith("crop:")) {
    const rest = seg.slice(5);
    const idx = rest.indexOf(":");
    const dims = idx === -1 ? rest : rest.slice(0, idx);
    const align = idx === -1 ? "center" : rest.slice(idx + 1);
    if (!parseSizeSegment(dims) || !ALIGN_NAMES.has(align)) {
      return {
        kind: "operation",
        title: "Crop",
        error: `malformed crop segment: ${seg} (want crop:<w>x<h>[:center|smart|face|top|bottom|left|right])`,
      };
    }
    return { kind: "operation", title: "Crop", detail: `crop to the aspect of ${dims} and resize to it; alignment: ${align === "center" ? "center (or the focal point for smart)" : align}` };
  }
  if (seg.startsWith("fit-in:")) {
    const rest = seg.slice(7);
    const size = parseSizeSegment(rest);
    if (!size || size.w.flip || size.h.flip) {
      return { kind: "operation", title: "Fit in", error: `malformed fit-in segment: ${seg} (flips are not allowed here)` };
    }
    return { kind: "operation", title: "Fit inside a box", detail: `scale proportionally to fit inside ${rest}` };
  }
  if (seg.startsWith("pad:")) {
    const rest = seg.slice(4);
    const idx = rest.indexOf(":");
    const dims = idx === -1 ? rest : rest.slice(0, idx);
    const fill = idx === -1 ? null : rest.slice(idx + 1);
    if (!parseSizeSegment(dims) || (fill !== null && fill !== "blur" && !/^[0-9a-fA-F]{6}$/.test(fill))) {
      return { kind: "operation", title: "Pad", error: `malformed pad segment: ${seg} (want pad:<w>x<h>[:<hex>|blur])` };
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
  return null; // Not an operation: the source starts here.
}

/**
 * Parse a crabcrop URL (or bare path) back into its pieces,
 * explaining each segment. Mirrors the server's dispatch:
 * segments are operations until the first segment that is not
 * one; everything from there is the source.
 *
 * Returns { signature, segments, source, remainder, errors }
 * where segments carry { raw, kind, title, detail?, error? }.
 */
export function explainUrl(
  input: string,
  nowSeconds: number = Math.floor(Date.now() / 1000),
): Explanation {
  // Extract the path by string surgery, never via the URL
  // parser: WHATWG parsing resolves dot segments (.. and
  // %2e%2e) that the server treats as literal path bytes
  // (audit P-3). A query or fragment is not part of the
  // signed path, on either side.
  let path = input.trim();
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
  const errors = [];
  const slash = path.indexOf("/");
  if (slash === -1) {
    return { signature: null, segments: [], source: "", remainder: "", errors: ["no signature segment: the path is a single segment"] };
  }
  const signatureRaw = path.slice(0, slash);
  const remainder = path.slice(slash + 1);
  let signature;
  if (signatureRaw === "unsafe") {
    signature = { kind: "unsafe", value: signatureRaw, detail: "unsigned; the server honors it only with CRABCROP_ALLOW_UNSAFE or in presets-only deployments" };
  } else if (/^[0-9a-f]{64}$/.test(signatureRaw)) {
    signature = { kind: "hmac", value: signatureRaw, detail: "HMAC-SHA256 of the remainder; paste the secret to verify it below" };
  } else if (/^[0-9a-f]+$/.test(signatureRaw)) {
    signature = { kind: "invalid", value: signatureRaw, detail: "lowercase hex, but not a full 64-character HMAC tag: the server will answer 403" };
    errors.push("malformed signature segment");
  } else {
    signature = { kind: "invalid", value: signatureRaw, detail: "not `unsafe` and not lowercase hex: the server rejects it as malformed (400)" };
    errors.push("malformed signature segment");
  }

  const segments = [];
  let rest = remainder;
  let source = "";
  for (;;) {
    const idx = rest.indexOf("/");
    const seg = idx === -1 ? rest : rest.slice(0, idx);
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
  return { signature, segments, source, remainder, errors };
}
