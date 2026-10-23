/**
 * Build and sign crabcrop URLs, per `spec/url-signing.md` in
 * the crabcrop repository. Standard library only.
 */
import { createHmac, timingSafeEqual } from "node:crypto";

/**
 * Compute the signature segment for a remainder: the lowercase
 * hex HMAC-SHA256 of `remainder` under `secret`. The remainder
 * is everything after the signature segment in the URL,
 * exactly as it will appear.
 */
export function sign(secret: string, remainder: string): string {
  return createHmac("sha256", secret).update(remainder, "utf8").digest("hex");
}

export interface BuildOptions {
  /** Unix-seconds expiry, emitted as a leading `exp:` segment. */
  expires?: number;
}

/**
 * Fail loudly on inputs that can never produce a verifiable URL
 * (audit P-4, P-5): the remainder travels as a raw URL path, so
 * `?`/`#` would truncate it and whitespace, control, and
 * non-ASCII characters cannot be sent as-is; an expiry must be
 * a non-negative integer the server's u64 parser accepts.
 */
function assertUrlText(text: string, what: string): void {
  for (const c of text) {
    const code = c.codePointAt(0) as number;
    if (c === "?" || c === "#" || code <= 0x20 || code === 0x7f || code > 0x7e) {
      throw new Error(
        `${what} contains a character that cannot appear in a crabcrop URL: ${JSON.stringify(c)}`,
      );
    }
  }
}

function assertExpires(expires: number): void {
  if (!Number.isSafeInteger(expires) || expires < 0) {
    throw new Error(
      `expires must be a non-negative integer number of unix seconds, got ${expires}`,
    );
  }
}

/**
 * A srcset width becomes a `{width}x0` size segment, so it
 * must satisfy the grammar's dimension rules: a positive
 * integer within the server cap. Anything else mints a
 * URL the server rejects (or worse, `-300` reads as a
 * flip), so it throws here, at build time.
 */
function assertWidth(width: number): void {
  if (!Number.isSafeInteger(width) || width < 1 || width > 16383) {
    throw new Error(
      `srcset width must be a positive integer no larger than 16383, got ${width}`,
    );
  }
}

/**
 * Assemble a signed (or `unsafe`) crabcrop URL. `operations`
 * is the operation string verbatim (several `/`-separated
 * segments and `preset:` references included). A null secret
 * produces the literal `unsafe` signature segment, for servers
 * running with unsafe or presets-only configurations.
 */
export function buildURL(
  base: string,
  secret: string | null,
  operations: string,
  source: string,
  options: BuildOptions = {},
): string {
  if (options.expires !== undefined) {
    assertExpires(options.expires);
  }
  assertUrlText(operations, "operations");
  assertUrlText(source, "source");
  const trimmedBase = base.replace(/\/+$/, "");
  let remainder = "";
  if (options.expires !== undefined) {
    remainder += `exp:${options.expires}/`;
  }
  if (operations) {
    remainder += `${operations}/`;
  }
  remainder += source;
  const signature = secret === null ? "unsafe" : sign(secret, remainder);
  return `${trimmedBase}/${signature}/${remainder}`;
}

/**
 * Build an `srcset` attribute value: one `{width}x0` URL per
 * width, entries formatted `{url} {width}w` and joined by
 * `, `. An expiry, when given, applies to every entry.
 */
export function buildSrcSet(
  base: string,
  secret: string | null,
  source: string,
  widths: number[],
  options: BuildOptions = {},
): string {
  for (const w of widths) {
    assertWidth(w);
  }
  return widths
    .map((w) => `${buildURL(base, secret, `${w}x0`, source, options)} ${w}w`)
    .join(", ");
}

export { explainUrl } from "./explain.ts";
export type { ExplainedSegment, Explanation } from "./explain.ts";

import { explainUrl } from "./explain.ts";

/**
 * Verify a URL's signature against a secret, locally. Returns
 * true or false, or null when the URL is unsigned or malformed.
 */
export function verifyUrlSignature(secret: string, input: string): boolean | null {
  const { signature, remainder } = explainUrl(input);
  if (!signature || signature.kind !== "hmac") {
    return null;
  }
  // Constant-time, like the Rust and Python verifiers: a
  // plain === leaks how much of the tag was right.
  const expected = Buffer.from(sign(secret, remainder), "utf8");
  const given = Buffer.from(signature.value, "utf8");
  return expected.length === given.length && timingSafeEqual(expected, given);
}
