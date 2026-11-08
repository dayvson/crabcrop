/**
 * The Node crypto backend: OpenSSL through `node:crypto`.
 * Package resolution picks it through the `#crypto` import's
 * `node` condition; bundles for browsers take `./portable.ts`,
 * which exports the same functions.
 */
import {
  createCipheriv,
  createDecipheriv,
  createHmac,
  randomBytes as nodeRandomBytes,
  timingSafeEqual as nodeTimingSafeEqual,
} from "node:crypto";

export const backend = "node";

/** HMAC-SHA-256 of the UTF-8 `message` under `key`. */
export function hmacSha256(key: string | Uint8Array, message: string): Uint8Array {
  return createHmac("sha256", key).update(message, "utf8").digest();
}

/** AES-256-GCM: ciphertext || 16-byte tag. */
export function aes256GcmSeal(
  key: Uint8Array,
  nonce: Uint8Array,
  aad: Uint8Array,
  plaintext: Uint8Array,
): Uint8Array {
  const cipher = createCipheriv("aes-256-gcm", key, nonce);
  cipher.setAAD(aad);
  return Buffer.concat([cipher.update(plaintext), cipher.final(), cipher.getAuthTag()]);
}

/** AES-256-GCM: the plaintext, or null when the tag does not verify. */
export function aes256GcmOpen(
  key: Uint8Array,
  nonce: Uint8Array,
  aad: Uint8Array,
  sealed: Uint8Array,
): Uint8Array | null {
  if (sealed.length < 16) {
    return null;
  }
  const decipher = createDecipheriv("aes-256-gcm", key, nonce);
  decipher.setAAD(aad);
  decipher.setAuthTag(sealed.subarray(sealed.length - 16));
  try {
    return Buffer.concat([decipher.update(sealed.subarray(0, sealed.length - 16)), decipher.final()]);
  } catch {
    return null;
  }
}

/** `n` bytes from OpenSSL's CSPRNG. */
export function randomBytes(n: number): Uint8Array {
  return nodeRandomBytes(n);
}

/** Constant-time equality of two byte strings of equal length. */
export function timingSafeEqual(a: Uint8Array, b: Uint8Array): boolean {
  return a.length === b.length && nodeTimingSafeEqual(a, b);
}
