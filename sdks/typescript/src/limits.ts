/**
 * The server's per-axis pixel cap (`MAX_DIMENSION` in the
 * server's `url.rs`): the largest value a size side, a crop
 * or pad box, a region, a border, a radius, or a pixelate
 * block can name. A larger value is a parse error on the
 * server, so the builders, the explainer, and the framework
 * integrations all check against this one constant.
 */
export const MAX_DIMENSION = 16383;

/**
 * The first remainder segment reserved for key derivation.
 * Every derived subkey (purge, purge-group, upload, seal, and
 * the server's cache entry key) is the HMAC of a label such as
 * `crabcrop/purge-key` under the secret, which is the image
 * signature of that label. No image URL may begin with this
 * segment (the server refuses one with `400 reserved_segment`),
 * and `sign` and every builder refuse a remainder that does,
 * so no image signature is ever a subkey. The explainer reports
 * such a remainder as the server answers it.
 */
export const RESERVED_SEGMENT = "crabcrop";

/**
 * Whether `remainder` begins with {@link RESERVED_SEGMENT}: it
 * is that segment alone or starts with `crabcrop/`. The test is
 * on the text as written, as the HMAC is: `exp:1/crabcrop/x`
 * and `CRABCROP/x` are not reserved.
 */
export function isReservedRemainder(remainder: string): boolean {
  return remainder === RESERVED_SEGMENT || remainder.startsWith(`${RESERVED_SEGMENT}/`);
}
