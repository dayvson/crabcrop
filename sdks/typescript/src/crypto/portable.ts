/**
 * The portable crypto backend: plain TypeScript, no Node
 * built-ins. Package resolution picks it through the
 * `#crypto` import's `browser` and `default` conditions, so
 * browser bundles (Vite, webpack, esbuild) and edge runtimes
 * get a working, synchronous SDK; Node takes `./node.ts`.
 * Both modules export the same functions with the same
 * results, which the test suite checks against `node:crypto`.
 */
import { aes256GcmOpen as open, aes256GcmSeal as seal } from "./aes-gcm.ts";
import { hmacSha256 as hmac } from "./sha256.ts";

export const backend = "portable";

const encoder = new TextEncoder();

/**
 * HMAC-SHA-256 of the UTF-8 `message` under `key` (a string
 * is UTF-8 encoded, as `node:crypto` does).
 */
export function hmacSha256(key: string | Uint8Array, message: string): Uint8Array {
  let keyBytes: Uint8Array;
  if (typeof key === "string") {
    keyBytes = encoder.encode(key);
  } else if (key instanceof Uint8Array) {
    keyBytes = key;
  } else {
    // node:crypto refuses a missing or non-string key the
    // same way, rather than signing with "undefined".
    throw new TypeError(`the HMAC key must be a string or a Uint8Array, got ${typeof key}`);
  }
  return hmac(keyBytes, encoder.encode(message));
}

/** AES-256-GCM: ciphertext || 16-byte tag. */
export function aes256GcmSeal(
  key: Uint8Array,
  nonce: Uint8Array,
  aad: Uint8Array,
  plaintext: Uint8Array,
): Uint8Array {
  return seal(key, nonce, aad, plaintext);
}

/** AES-256-GCM: the plaintext, or null when the tag does not verify. */
export function aes256GcmOpen(
  key: Uint8Array,
  nonce: Uint8Array,
  aad: Uint8Array,
  sealed: Uint8Array,
): Uint8Array | null {
  return open(key, nonce, aad, sealed);
}

/** `n` bytes from the platform's CSPRNG (`crypto.getRandomValues`). */
export function randomBytes(n: number): Uint8Array {
  const source = (globalThis as { crypto?: { getRandomValues?: <T extends Uint8Array>(a: T) => T } })
    .crypto;
  if (typeof source?.getRandomValues !== "function") {
    throw new Error("no secure random source: crypto.getRandomValues is unavailable here");
  }
  return source.getRandomValues(new Uint8Array(n));
}

/** Constant-time equality of two byte strings of equal length. */
export function timingSafeEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) {
    return false;
  }
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a[i] ^ b[i];
  }
  return diff === 0;
}
