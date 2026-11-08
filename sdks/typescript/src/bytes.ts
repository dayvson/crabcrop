/**
 * Byte encodings the SDK needs, without Node's `Buffer`, so
 * the same code runs in browsers: UTF-8 through the
 * TextEncoder and TextDecoder globals, hex, and unpadded
 * base64url.
 */

const encoder = new TextEncoder();
// fatal: invalid UTF-8 is an error, not U+FFFD. ignoreBOM:
// a leading U+FEFF is text like any other, kept as decoded.
const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

/** The UTF-8 bytes of `text` (lone surrogates become U+FFFD, as in Node). */
export function utf8(text: string): Uint8Array {
  return encoder.encode(text);
}

/** Decode UTF-8, or null when the bytes are not valid UTF-8. */
export function utf8Strict(bytes: Uint8Array): string | null {
  try {
    return decoder.decode(bytes);
  } catch {
    return null;
  }
}

const HEX = Array.from({ length: 256 }, (_, i) => i.toString(16).padStart(2, "0"));

/** Lowercase hex. */
export function toHex(bytes: Uint8Array): string {
  let out = "";
  for (let i = 0; i < bytes.length; i++) {
    out += HEX[bytes[i]];
  }
  return out;
}

const B64URL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
// Character code to sextet, -1 outside the alphabet.
const B64URL_INDEX = new Int8Array(128).fill(-1);
for (let i = 0; i < B64URL.length; i++) {
  B64URL_INDEX[B64URL.charCodeAt(i)] = i;
}

/** Unpadded base64url (RFC 4648 section 5). */
export function toBase64Url(bytes: Uint8Array): string {
  let out = "";
  let i = 0;
  for (; i + 3 <= bytes.length; i += 3) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
    out += B64URL[n >>> 18] + B64URL[(n >>> 12) & 63] + B64URL[(n >>> 6) & 63] + B64URL[n & 63];
  }
  const rest = bytes.length - i;
  if (rest === 1) {
    const n = bytes[i] << 16;
    out += B64URL[n >>> 18] + B64URL[(n >>> 12) & 63];
  } else if (rest === 2) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8);
    out += B64URL[n >>> 18] + B64URL[(n >>> 12) & 63] + B64URL[(n >>> 6) & 63];
  }
  return out;
}

/**
 * Decode unpadded base64url, or null on any character outside
 * the alphabet (padding included) or an impossible length.
 * Callers that need the canonical spelling compare the input
 * with `toBase64Url` of the result.
 */
export function fromBase64Url(text: string): Uint8Array | null {
  if (text.length % 4 === 1) {
    return null;
  }
  const out = new Uint8Array(Math.floor((text.length * 3) / 4));
  let bits = 0;
  let acc = 0;
  let o = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    const v = code < 128 ? B64URL_INDEX[code] : -1;
    if (v < 0) {
      return null;
    }
    acc = ((acc << 6) | v) & 0xffffff;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out[o++] = (acc >>> bits) & 0xff;
    }
  }
  return out;
}
