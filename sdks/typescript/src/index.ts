/**
 * Build and sign crabcrop URLs, per `spec/url-signing.md` in
 * the crabcrop repository. No dependencies: `#crypto` resolves
 * to `node:crypto` on Node and to a plain TypeScript backend
 * in browser bundles (see package.json `imports`), and this
 * module uses no Node globals, so it bundles for either.
 */
import {
  aes256GcmOpen,
  aes256GcmSeal,
  hmacSha256,
  randomBytes,
  timingSafeEqual,
} from "#crypto";
import { fromBase64Url, toBase64Url, toHex, utf8, utf8Strict } from "./bytes.ts";
import { MAX_DIMENSION } from "./limits.ts";

/**
 * The first remainder segment reserved for key derivation.
 * Every derived subkey (purge, purge-group, upload, seal, and
 * the server's cache entry key) is the HMAC of a label such as
 * `crabcrop/purge-key` under the secret, which is the image
 * signature of that label. No image URL may begin with this
 * segment (the server refuses one with `400 reserved_segment`),
 * and {@link sign} and every builder refuse a remainder that
 * does, so no image signature is ever a subkey.
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

function assertNotReserved(remainder: string): void {
  if (isReservedRemainder(remainder)) {
    throw new Error(
      `remainder begins with the reserved segment "${RESERVED_SEGMENT}": it is kept for key derivation, and no image URL may begin with it`,
    );
  }
}

/**
 * Compute the signature segment for a remainder: the lowercase
 * hex HMAC-SHA256 of `remainder` under `secret`. The remainder
 * is everything after the signature segment in the URL,
 * exactly as it will appear. A remainder that begins with
 * {@link RESERVED_SEGMENT} throws: its signature could be a
 * derived subkey.
 */
export function sign(secret: string, remainder: string): string {
  assertNotReserved(remainder);
  return toHex(hmacSha256(secret, remainder));
}

/**
 * A subkey derived from the secret: the raw HMAC-SHA256 of
 * `label` under it, one per signing domain. Every label begins
 * with {@link RESERVED_SEGMENT}, which is why {@link sign}
 * refuses to compute the same value as a signature.
 */
function subkey(secret: string, label: string): Uint8Array {
  return hmacSha256(secret, label);
}

/**
 * The purge signature for a target path (the request
 * remainder without its signature): HMAC-SHA256 of the
 * path under a subkey derived from the secret (the raw
 * HMAC of the secret over the label `crabcrop/purge-key`),
 * so an image signature does not verify as a purge
 * signature: no signer signs a label (see
 * {@link RESERVED_SEGMENT}). This is the signature segment
 * for `POST /purge/<signature>/<path>`.
 */
export function purgeSignature(secret: string, target: string): string {
  return toHex(hmacSha256(subkey(secret, "crabcrop/purge-key"), target));
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
  return toHex(hmacSha256(subkey(secret, "crabcrop/purge-group-key"), group));
}

/**
 * The upload signature for an object name: HMAC-SHA256 of
 * the name under a subkey derived from the secret (the raw
 * HMAC of the secret over the label `crabcrop/upload-key`),
 * so neither an image signature nor a purge signature
 * verifies as an upload signature: a purge signature is
 * under another subkey, and no signer signs a label (see
 * {@link RESERVED_SEGMENT}). This is the signature segment
 * for `PUT /upload/<signature>/<name>`.
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
  const signed = expires === undefined ? name : `exp:${expires}/${name}`;
  return toHex(hmacSha256(subkey(secret, "crabcrop/upload-key"), signed));
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
const SEAL_AAD = utf8("crabcrop/sealed/1");

/**
 * The seal subkey: the raw bytes of the HMAC-SHA256 of the
 * label `crabcrop/seal-key` under the secret, its own domain
 * beside purge, purge-group, and upload, so there is no
 * second secret to configure or rotate: each keyring entry
 * yields its own seal subkey, and the rotation window for
 * signatures is the rotation window for tokens.
 */
function sealKey(secret: string): Uint8Array {
  return subkey(secret, "crabcrop/seal-key");
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
  const plaintext = utf8(remainder);
  if (plaintext.length > MAX_SEALED_REMAINDER) {
    throw new Error(
      `remainder is over the ${MAX_SEALED_REMAINDER}-byte sealing cap: a recipe that long belongs in a preset`,
    );
  }
  const sealed = aes256GcmSeal(sealKey(secret), nonce, SEAL_AAD, plaintext);
  const body = new Uint8Array(nonce.length + sealed.length);
  body.set(nonce);
  body.set(sealed, nonce.length);
  const token = `s1.${toBase64Url(body)}`;
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
  // Canonicity is checked by re-encoding: anything but the
  // exact minted spelling (padding, stray characters, nonzero
  // trailing bits) fails here.
  const raw = fromBase64Url(body);
  if (raw === null || raw.length < 12 + 16 || toBase64Url(raw) !== body) {
    throw openFailed();
  }
  const plain = aes256GcmOpen(sealKey(secret), raw.subarray(0, 12), SEAL_AAD, raw.subarray(12));
  // The plaintext must be valid UTF-8: a remainder is text.
  const text = plain === null ? null : utf8Strict(plain);
  if (text === null) {
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
  assertNotReserved(remainder);
  // The inputs are checked before the secret, as in the Rust
  // and Python builders, so a refused input gets the same
  // error with or without a secret.
  if (secret === null || secret === undefined) {
    throw new Error("sealing requires a secret: there is no unsafe token");
  }
  const segment = seal(secret, remainder, options.keyId);
  return options.name ? `${trimmedBase}/${segment}/${options.name}` : `${trimmedBase}/${segment}`;
}

/**
 * Characters a browser rewrites in a URL path (issue #472): an
 * `<img src>` or `fetch()` URL is parsed by WHATWG rules, which
 * turn `\` into `/` and percent-encode the rest (Chrome encodes
 * `^` and `|` too). The server verifies the raw path, so a URL
 * signed over one of them answers 403 from a page.
 */
const BROWSER_REWRITTEN = "\"<>\\^`{|}";

/**
 * Fail loudly on inputs that can never produce a verifiable URL
 * (audit P-4, P-5): the remainder travels as a raw URL path, so
 * `?`/`#` would truncate it, whitespace, control, and non-ASCII
 * characters cannot be sent as-is, and a browser rewrites the
 * {@link BROWSER_REWRITTEN} set; an expiry must be a
 * non-negative integer the server's u64 parser accepts.
 */
function assertUrlText(text: string, what: string): void {
  for (const c of text) {
    const code = c.codePointAt(0) as number;
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
  if (!Number.isSafeInteger(width) || width < 1 || width > MAX_DIMENSION) {
    throw new Error(
      `srcset width must be a positive integer no larger than ${MAX_DIMENSION}, got ${width}`,
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
  // Refused with or without a secret: the server refuses the
  // URL whatever its signature.
  assertNotReserved(remainder);
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
export { MAX_DIMENSION } from "./limits.ts";
export { filter, joinOps, ops } from "./operations.ts";
export type {
  Anchor,
  BlendMode,
  CropAlign,
  CurveChannel,
  Dimension,
  FormatOptions,
  ImageFormat,
  MaskShape,
  PerFormatQualityFormat,
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
  // No signature over a reserved remainder is ever valid: the
  // server refuses it whatever the signature, and this check
  // must not confirm a derived subkey either.
  if (isReservedRemainder(remainder)) {
    return false;
  }
  // Constant-time, like the Rust and Python verifiers: a
  // plain === leaks how much of the tag was right.
  const given = utf8(signature.value);
  const matches = (candidate: string): boolean =>
    timingSafeEqual(utf8(sign(candidate, remainder)), given);
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
