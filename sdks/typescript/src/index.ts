/**
 * Build and sign crabcrop URLs, per `spec/url-signing.md` in
 * the crabcrop repository. Standard library only.
 */
import { createHmac } from "node:crypto";

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
  return widths
    .map((w) => `${buildURL(base, secret, `${w}x0`, source, options)} ${w}w`)
    .join(", ");
}
