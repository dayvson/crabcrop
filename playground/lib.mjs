/**
 * The playground's core: signing (WebCrypto) and URL
 * explanation, implementing spec/url-signing.md in the
 * browser. The scheme is the same one the SDK implements for
 * Node; the test suite pins both to the canonical vectors in
 * spec/test-vectors.json and to each other's outputs.
 *
 * Nothing here makes a network request, and a secret passed
 * to these functions is used only for the HMAC computation.
 */

const encoder = new TextEncoder();

/** Lowercase hex HMAC-SHA256 of `remainder` under `secret`. */
export async function sign(secret, remainder) {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const mac = await crypto.subtle.sign("HMAC", key, encoder.encode(remainder));
  return [...new Uint8Array(mac)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Assemble a crabcrop URL exactly as the SDK's buildURL does:
 * an optional leading `exp:` segment, the operations verbatim,
 * the source, and either the HMAC of the remainder or the
 * literal `unsafe` when the secret is empty.
 */
export async function buildPlaygroundUrl({ base, secret, operations, source, expires }) {
  const trimmedBase = base.replace(/\/+$/, "");
  let remainder = "";
  if (expires !== undefined && expires !== null) {
    remainder += `exp:${expires}/`;
  }
  if (operations) {
    remainder += `${operations}/`;
  }
  remainder += source;
  const signature = secret ? await sign(secret, remainder) : "unsafe";
  return `${trimmedBase}/${signature}/${remainder}`;
}

// ----------------------------------------------------------------
// URL explanation

const FILTER_BLURBS = {
  grayscale: "convert to grayscale",
  invert: "invert the colors",
  no_upscale: "never enlarge the source",
  blur: "gaussian blur, argument is the sigma",
  blurfaces: "blur detected faces (needs a face model on the server)",
  brighten: "brighten by the given amount (-255..255)",
  contrast: "contrast multiplier (1 is identity)",
  huerotate: "rotate the hue by the given degrees",
  quality: "output quality 0..100, or `auto` for perceptual targeting",
  format: "force the output format",
  fill: "the background color other operations pad with",
  facepad: "how much a face crop expands around the face",
  pixelatefaces: "pixelate detected faces (needs a face model on the server)",
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
};

const SIZE_RE = /^-?(\d+|orig)x-?(\d+|orig)$/;

function sizeBlurb(seg) {
  const [, w, h] = seg.match(SIZE_RE);
  return `resize to ${w === "orig" ? "the original width" : `${w}px wide`} by ${
    h === "orig" ? "the original height" : `${h}px high`
  } (0 keeps the aspect ratio; a leading - flips)`;
}

function explainFilters(rest) {
  const parts = rest.split(":");
  const blurbs = [];
  for (let i = 0; i < parts.length; i++) {
    // Rejoin parenthesized arguments, which may contain ':'.
    let part = parts[i];
    while (part.includes("(") && !part.includes(")") && i + 1 < parts.length) {
      part += `:${parts[++i]}`;
    }
    const match = part.match(/^([a-z_0-9]+)(?:\((.*)\))?$/);
    if (!match) {
      return { error: `malformed filter: ${part}` };
    }
    const [, name, args] = match;
    const blurb = FILTER_BLURBS[name];
    if (!blurb) {
      return { error: `unknown filter: ${name}` };
    }
    blurbs.push(args !== undefined ? `${name}(${args}): ${blurb}` : `${name}: ${blurb}`);
  }
  return { detail: blurbs.join("; ") };
}

function explainSegment(seg, nowSeconds) {
  if (seg.startsWith("exp:")) {
    const value = seg.slice(4);
    if (!/^\d+$/.test(value)) {
      return { kind: "expiry", title: "Expiry", error: `malformed expiry: ${seg}` };
    }
    const seconds = Number(value);
    const when = new Date(seconds * 1000).toISOString().replace("T", " ").replace(".000Z", " UTC");
    const expired = nowSeconds >= seconds;
    return {
      kind: "expiry",
      title: "Expiry",
      detail: `valid strictly before ${when} (${seconds})${expired ? " — EXPIRED at the reference time" : ""}`,
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
    return { kind: "operation", title: "Crop rectangle", detail: `crop the exact rectangle ${seg.slice(10)} (left x top : right x bottom)` };
  }
  if (seg.startsWith("crop:")) {
    const rest = seg.slice(5);
    if (!/^\d+x\d+(:[a-z]+)?$/.test(rest)) {
      return { kind: "operation", title: "Crop", error: `malformed crop segment: ${seg} (want crop:<w>x<h>[:align])` };
    }
    return { kind: "operation", title: "Crop", detail: `crop to the aspect of ${rest.split(":")[0]} and resize to it; alignment: ${rest.split(":")[1] ?? "center (or the focal point for smart)"}` };
  }
  if (seg.startsWith("fit-in:")) {
    const rest = seg.slice(7);
    if (!SIZE_RE.test(rest) || rest.includes("-")) {
      return { kind: "operation", title: "Fit in", error: `malformed fit-in segment: ${seg}` };
    }
    return { kind: "operation", title: "Fit inside a box", detail: `scale proportionally to fit inside ${rest}` };
  }
  if (seg.startsWith("pad:")) {
    const rest = seg.slice(4);
    const m = rest.match(/^(\d+x\d+)(?::([0-9a-fA-F]{6}|blur))?$/);
    if (!m) {
      return { kind: "operation", title: "Pad", error: `malformed pad segment: ${seg} (want pad:<w>x<h>[:<hex>|blur])` };
    }
    return { kind: "operation", title: "Pad", detail: `fit inside ${m[1]} and fill the canvas to exactly that size, fill: ${m[2] ?? "the fill(...) color or black"}` };
  }
  if (seg.startsWith("filters:")) {
    const { detail, error } = explainFilters(seg.slice(8));
    if (error) {
      return { kind: "operation", title: "Filters", error };
    }
    return { kind: "operation", title: "Filters", detail };
  }
  if (SIZE_RE.test(seg)) {
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
export function explainUrl(input, nowSeconds = Math.floor(Date.now() / 1000)) {
  let path = input.trim();
  try {
    path = new URL(path).pathname;
  } catch {
    // A bare path, not a full URL: use it as-is.
  }
  path = path.replace(/^\//, "");
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
  } else if (/^[0-9a-fA-F]{64}$/.test(signatureRaw)) {
    signature = { kind: "hmac", value: signatureRaw, detail: "HMAC-SHA256 of the remainder; paste the secret to verify it below" };
  } else {
    signature = { kind: "invalid", value: signatureRaw, detail: "not `unsafe` and not a 64-character hex tag: the server will answer 403" };
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

/**
 * Verify a URL's signature locally against a secret the user
 * pasted. Returns true/false, or null when the URL is unsigned
 * or malformed.
 */
export async function verifyUrlSignature(secret, input) {
  const { signature, remainder } = explainUrl(input);
  if (!signature || signature.kind !== "hmac") {
    return null;
  }
  const expected = await sign(secret, remainder);
  return expected === signature.value.toLowerCase();
}
