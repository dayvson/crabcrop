/**
 * Build and sign crabcrop URLs, per `spec/url-signing.md` in
 * the crabcrop repository. Standard library only.
 */
import { createCipheriv, createDecipheriv, createHmac, randomBytes, timingSafeEqual } from "node:crypto";

/**
 * Compute the signature segment for a remainder: the lowercase
 * hex HMAC-SHA256 of `remainder` under `secret`. The remainder
 * is everything after the signature segment in the URL,
 * exactly as it will appear.
 */
export function sign(secret: string, remainder: string): string {
  return createHmac("sha256", secret).update(remainder, "utf8").digest("hex");
}

/**
 * The purge signature for a target path (the request
 * remainder without its signature): HMAC-SHA256 of the
 * path under a subkey derived from the secret (the raw
 * HMAC of the secret over the label `crabcrop/purge-key`),
 * so an image signature can never be replayed as a purge
 * signature. This is the signature segment for
 * `POST /purge/<signature>/<path>`.
 */
export function purgeSignature(secret: string, target: string): string {
  const subkey = Buffer.from(sign(secret, "crabcrop/purge-key"), "hex");
  return createHmac("sha256", subkey).update(target, "utf8").digest("hex");
}

/**
 * The purge-group signature for a cache group: HMAC-SHA256
 * of the group name under a subkey derived from the secret
 * (the raw HMAC of the secret over the label
 * `crabcrop/purge-group-key`), its own domain beside the
 * purge subkey, so a per-target purge signature can never
 * be transplanted to the purge-group endpoint. The helper
 * exists so nobody has to know the signed string. This is
 * the signature segment for
 * `POST /purge-group/<signature>/<group>`.
 */
export function purgeGroupSignature(secret: string, group: string): string {
  const subkey = Buffer.from(sign(secret, "crabcrop/purge-group-key"), "hex");
  return createHmac("sha256", subkey).update(group, "utf8").digest("hex");
}

/**
 * The upload signature for an object name: HMAC-SHA256 of
 * the name under a subkey derived from the secret (the raw
 * HMAC of the secret over the label `crabcrop/upload-key`),
 * so neither an image signature nor a purge signature can
 * ever be replayed as an upload signature. This is the
 * signature segment for `PUT /upload/<signature>/<name>`.
 *
 * With `expires` (unix seconds), the signature covers
 * `exp:<expires>/<name>` instead, and the request path is
 * `PUT /upload/<signature>/exp:<expires>/<name>`: the
 * expiry sits inside the signed string, so it can neither
 * be stripped nor extended, and the server refuses the
 * upload once that second is reached. Delegated upload
 * tokens should always use the expiring form; without it
 * a token replays until the secret rotates.
 */
export function uploadSignature(secret: string, name: string, expires?: number): string {
  const subkey = Buffer.from(sign(secret, "crabcrop/upload-key"), "hex");
  const signed = expires === undefined ? name : `exp:${expires}/${name}`;
  return createHmac("sha256", subkey).update(signed, "utf8").digest("hex");
}

export interface BuildOptions {
  /** Unix-seconds expiry, emitted as a leading `exp:` segment. */
  expires?: number;
  /**
   * Key ID to stamp on the signature segment (`<key-id>:<tag>`),
   * for servers running a secret rotation: they verify a keyed
   * segment against the named key only. Requires a non-null
   * secret (an `unsafe` URL has no key to name).
   */
  keyId?: string;
}

/** Whether `id` is a well-formed key ID (1-32 of `[A-Za-z0-9_-]`). */
export function isValidKeyId(id: string): boolean {
  return /^[A-Za-z0-9_-]{1,32}$/.test(id);
}

/**
 * The signature segment for a keyed signature: `<key-id>:`
 * followed by the hex tag `sign` computes.
 */
export function signWithKeyId(secret: string, keyId: string, remainder: string): string {
  if (!isValidKeyId(keyId)) {
    throw new Error(`invalid key ID ${JSON.stringify(keyId)}: 1 to 32 characters of [A-Za-z0-9_-]`);
  }
  return `${keyId}:${sign(secret, remainder)}`;
}

/**
 * The largest remainder {@link seal} will encrypt, in bytes
 * (issue #255). A recipe longer than this belongs in a
 * preset; the cap also bounds the token a server must
 * decode.
 */
export const MAX_SEALED_REMAINDER = 2048;

/** The AAD every sealed token is bound to. */
const SEAL_AAD = Buffer.from("crabcrop/sealed/1", "utf8");

/**
 * The seal subkey: the raw bytes of the HMAC-SHA256 of the
 * label `crabcrop/seal-key` under the secret, its own domain
 * beside purge, purge-group, and upload, so there is no
 * second secret to configure or rotate: each keyring entry
 * yields its own seal subkey, and the rotation window for
 * signatures is the rotation window for tokens.
 */
function sealKey(secret: string): Buffer {
  return Buffer.from(sign(secret, "crabcrop/seal-key"), "hex");
}

/**
 * Seal a remainder into a token segment with a
 * caller-supplied nonce: `s1.<base64url(nonce ||
 * ciphertext || tag)>`, AES-256-GCM under the derived seal
 * subkey with the fixed associated data `crabcrop/sealed/1`.
 *
 * DANGER: a nonce must never repeat under one secret.
 * Reusing a nonce with AES-GCM destroys the construction's
 * confidentiality and integrity guarantees. This function
 * is the deterministic seam the spec's test vectors use;
 * production minting calls {@link seal}, which draws a
 * fresh random nonce every time.
 */
export function sealWithNonce(
  secret: string,
  remainder: string,
  nonce: Uint8Array,
  keyId?: string,
): string {
  if (nonce.length !== 12) {
    throw new Error(`sealed token nonce must be 12 bytes, got ${nonce.length}`);
  }
  if (Buffer.byteLength(remainder, "utf8") > MAX_SEALED_REMAINDER) {
    throw new Error(
      `remainder is over the ${MAX_SEALED_REMAINDER}-byte sealing cap: a recipe that long belongs in a preset`,
    );
  }
  const cipher = createCipheriv("aes-256-gcm", sealKey(secret), Buffer.from(nonce));
  cipher.setAAD(SEAL_AAD);
  const ciphertext = Buffer.concat([cipher.update(remainder, "utf8"), cipher.final()]);
  const body = Buffer.concat([Buffer.from(nonce), ciphertext, cipher.getAuthTag()]);
  const token = `s1.${body.toString("base64url")}`;
  if (keyId === undefined) {
    return token;
  }
  if (!isValidKeyId(keyId)) {
    throw new Error(`invalid key ID ${JSON.stringify(keyId)}: 1 to 32 characters of [A-Za-z0-9_-]`);
  }
  return `${keyId}:${token}`;
}

/**
 * Seal a remainder into a token segment (issue #255), with
 * a fresh random 96-bit nonce. The token is the whole
 * signature segment; pass `keyId` to stamp `<key-id>:`
 * in front of it, so servers in a rotation window open it
 * against the named key only.
 */
export function seal(secret: string, remainder: string, keyId?: string): string {
  return sealWithNonce(secret, remainder, randomBytes(12), keyId);
}

/**
 * Open a token segment, with or without a `<key-id>:`
 * prefix, under one secret, returning the remainder it
 * seals. The key ID is routing metadata and is not
 * consulted: the question is whether this secret opens the
 * token. Every failure (malformed body, wrong secret,
 * tampered bytes, unknown version) throws the one error,
 * `the token did not open under this secret`: openers must
 * never learn which stage failed.
 */
export function unseal(secret: string, segment: string): string {
  const openFailed = (): Error => new Error("the token did not open under this secret");
  let token = segment;
  const colon = segment.indexOf(":");
  if (colon >= 0) {
    if (!isValidKeyId(segment.slice(0, colon))) {
      throw openFailed();
    }
    token = segment.slice(colon + 1);
  }
  if (!token.startsWith("s1.")) {
    throw openFailed();
  }
  const body = token.slice(3);
  // Node's base64 decoder is lenient (it skips characters
  // outside the alphabet), so canonicity is checked by
  // re-encoding: anything but the exact minted spelling
  // fails here.
  const raw = Buffer.from(body, "base64url");
  if (raw.length < 12 + 16 || raw.toString("base64url") !== body) {
    throw openFailed();
  }
  const decipher = createDecipheriv("aes-256-gcm", sealKey(secret), raw.subarray(0, 12));
  decipher.setAAD(SEAL_AAD);
  decipher.setAuthTag(raw.subarray(raw.length - 16));
  let plain: Buffer;
  try {
    plain = Buffer.concat([
      decipher.update(raw.subarray(12, raw.length - 16)),
      decipher.final(),
    ]);
  } catch {
    throw openFailed();
  }
  const text = plain.toString("utf8");
  if (!Buffer.from(text, "utf8").equals(plain)) {
    throw openFailed();
  }
  return text;
}

/** Options for {@link buildSealedURL}. */
export interface SealOptions {
  /** Unix-seconds expiry, sealed inside the token as a leading `exp:` segment. */
  expires?: number;
  /** Key ID to stamp on the token segment (`<key-id>:s1.…`), for rotation. */
  keyId?: string;
  /**
   * The cosmetic friendly name after the token. The server
   * never parses, validates, or authenticates it; one
   * filename-like segment is the dependable shape. Omit it
   * and the URL ends at the token.
   */
  name?: string;
}

/**
 * Assemble a sealed crabcrop URL:
 * `{base}/[<key-id>:]s1.<token>/{friendly-name}` (issue
 * #255). The remainder is assembled exactly as
 * {@link buildURL} assembles it (expiry segment,
 * operations, source) and then sealed whole, pre-expansion,
 * so the sealed and plain forms of one render are one cache
 * entry and one purge target on the server. A secret is
 * required: there is no `unsafe` spelling of a token.
 */
export function buildSealedURL(
  base: string,
  secret: string,
  operations: string,
  source: string,
  options: SealOptions = {},
): string {
  if (secret === null || secret === undefined) {
    throw new Error("sealing requires a secret: there is no unsafe token");
  }
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
  const segment = seal(secret, remainder, options.keyId);
  return options.name ? `${trimmedBase}/${segment}/${options.name}` : `${trimmedBase}/${segment}`;
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
  let signature: string;
  if (secret === null) {
    if (options.keyId !== undefined) {
      throw new Error("keyId requires a secret: an unsafe URL has no key to name");
    }
    signature = "unsafe";
  } else if (options.keyId !== undefined) {
    signature = signWithKeyId(secret, options.keyId, remainder);
  } else {
    signature = sign(secret, remainder);
  }
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
export { FILTER_DESCRIPTIONS } from "./explain.ts";
export type { ExplainedSegment, Explanation, SegmentKind, SignatureKind } from "./explain.ts";
export { filter, joinOps, ops } from "./operations.ts";
export type {
  Anchor,
  BlendMode,
  CropAlign,
  CurveChannel,
  Dimension,
  ImageFormat,
  MaskShape,
  StudioMode,
  StudioOptions,
  StudioShadow,
} from "./operations.ts";

import { explainUrl } from "./explain.ts";

/**
 * Verify a URL's signature locally. Returns true or false, or
 * null when the URL is unsigned or malformed.
 *
 * With a single secret string, the key ID (when the URL
 * carries one) is ignored: the question is whether this
 * secret made the tag. With a keyring (a map of key ID to
 * secret), the server's rotation semantics apply: a keyed
 * URL consults exactly its named key (an unknown ID is
 * false), and an unkeyed URL consults every key in the map.
 */
export function verifyUrlSignature(
  secret: string | Record<string, string>,
  input: string,
): boolean | null {
  const { signature, remainder } = explainUrl(input);
  if (!signature || signature.kind !== "hmac") {
    return null;
  }
  // Constant-time, like the Rust and Python verifiers: a
  // plain === leaks how much of the tag was right.
  const given = Buffer.from(signature.value, "utf8");
  const matches = (candidate: string): boolean => {
    const expected = Buffer.from(sign(candidate, remainder), "utf8");
    return expected.length === given.length && timingSafeEqual(expected, given);
  };
  if (typeof secret === "string") {
    return matches(secret);
  }
  if (signature.keyId !== undefined) {
    // Own properties only: key IDs are attacker-shaped text
    // (`constructor`, `toString`, `__proto__` are all legal
    // IDs), and an inherited lookup would hand a non-string
    // to the HMAC and throw instead of answering false.
    const candidate = Object.hasOwn(secret, signature.keyId)
      ? secret[signature.keyId]
      : undefined;
    return typeof candidate === "string" ? matches(candidate) : false;
  }
  return Object.values(secret).some(matches);
}
