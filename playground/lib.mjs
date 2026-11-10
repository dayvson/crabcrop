/**
 * The playground's core: signing (WebCrypto) and URL
 * assembly, implementing spec/url-signing.md in the browser,
 * plus the TypeScript SDK's own URL explainer (./sdk/). The
 * scheme is the same one the SDK implements for Node; the
 * test suite pins both to the canonical vectors in
 * spec/test-vectors.json and to each other's outputs.
 *
 * Nothing here makes a network request, and a secret passed
 * to these functions is used only for the HMAC computation.
 */

import { explainUrl } from "./sdk/explain.js";

const encoder = new TextEncoder();

/**
 * The first remainder segment reserved for key derivation, as
 * in the SDK: every derived subkey (purge, purge-group, upload,
 * seal, cache entry) is the HMAC of a label such as
 * `crabcrop/upload-key` under the secret, which is the image
 * signature of that label. The server refuses an image URL
 * that begins with this segment, and no signer signs one.
 */
export const RESERVED_SEGMENT = "crabcrop";

/** Whether `remainder` is the reserved segment or starts with `crabcrop/`. */
export function isReservedRemainder(remainder) {
  return remainder === RESERVED_SEGMENT || remainder.startsWith(`${RESERVED_SEGMENT}/`);
}

function assertNotReserved(remainder) {
  if (isReservedRemainder(remainder)) {
    throw new Error(
      `remainder begins with the reserved segment "${RESERVED_SEGMENT}": it is kept for key derivation, and no image URL may begin with it`,
    );
  }
}

/**
 * Lowercase hex HMAC-SHA256 of `remainder` under `secret`. A
 * remainder that begins with the reserved segment throws: its
 * signature could be a derived subkey.
 */
export async function sign(secret, remainder) {
  assertNotReserved(remainder);
  // WebCrypto refuses a zero-length HMAC key. By the HMAC
  // definition, a key shorter than the 64-byte block size is
  // padded with zeros to the block size, so an empty key is
  // exactly a 64-byte all-zero key: importing the padded form
  // is the same key, not an approximation (audit P-6).
  const keyBytes = encoder.encode(secret);
  const key = await crypto.subtle.importKey(
    "raw",
    keyBytes.length === 0 ? new Uint8Array(64) : keyBytes,
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
 * literal `unsafe`. Only a null/undefined secret means unsafe:
 * an empty string is a real (if weak) key and signs, matching
 * every other implementation (audit P-6). Inputs that can
 * never verify throw instead of minting a dead URL (P-4, P-5).
 */
export async function buildPlaygroundUrl({ base, secret, operations, source, expires }) {
  if (expires !== undefined && expires !== null) {
    if (!Number.isSafeInteger(expires) || expires < 0) {
      throw new Error(`expires must be a non-negative integer of unix seconds, got ${expires}`);
    }
  }
  assertUrlText(operations, "operations");
  assertUrlText(source, "source");
  const trimmedBase = base.replace(/\/+$/, "");
  let remainder = "";
  if (expires !== undefined && expires !== null) {
    remainder += `exp:${expires}/`;
  }
  if (operations) {
    remainder += `${operations}/`;
  }
  remainder += source;
  // Refused with or without a secret, as the SDK refuses it:
  // the server refuses the URL whatever its signature.
  assertNotReserved(remainder);
  const signature =
    secret === null || secret === undefined ? "unsafe" : await sign(secret, remainder);
  return `${trimmedBase}/${signature}/${remainder}`;
}

// Characters a browser rewrites in a URL path (issue #472):
// `\` becomes `/` and the rest are percent-encoded, so a URL
// signed over one of them answers 403 when a page requests it.
// The SDK's assertUrlText refuses the same set.
const BROWSER_REWRITTEN = "\"<>\\^`{|}";

function assertUrlText(text, what) {
  for (const c of text) {
    const code = c.codePointAt(0);
    const rewritten = BROWSER_REWRITTEN.includes(c);
    if (c === "?" || c === "#" || code <= 0x20 || code === 0x7f || code > 0x7e || rewritten) {
      const hint = rewritten
        ? ` (browsers rewrite it; percent-encode it as %${code.toString(16).toUpperCase()})`
        : "";
      throw new Error(
        `${what} contains a character that cannot appear in a crabcrop URL: ${JSON.stringify(c)}${hint}`,
      );
    }
  }
}

// ----------------------------------------------------------------
// URL explanation
//
// The explainer is the TypeScript SDK's own (issue #482):
// ./sdk/ holds the SDK's build of sdks/typescript/src/explain.ts,
// copied by scripts/sync_playground_sdk.py, so the page stays a
// static directory and the grammar has one JavaScript copy.

export { explainUrl };

/**
 * Constant-time string equality for MAC comparison: the
 * loop always scans the whole tag, so how much of a
 * guessed tag is right does not leak through timing. The
 * Rust, TypeScript, and Python verifiers all compare this
 * way; the playground is the same code in the browser,
 * where `node:crypto` is unavailable.
 */
function macEqual(a, b) {
  if (a.length !== b.length) {
    return false;
  }
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
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
  // No signature over a reserved remainder is ever valid, and
  // this check must not confirm a derived subkey either.
  if (isReservedRemainder(remainder)) {
    return false;
  }
  const expected = await sign(secret, remainder);
  // No case-folding: the explainer already classifies a
  // non-lowercase tag as invalid, and the SDK verifiers
  // compare the canonical spelling as-is.
  return macEqual(expected, signature.value);
}
