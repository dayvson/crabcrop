/**
 * SHA-256 (FIPS 180-4) and HMAC-SHA-256 (RFC 2104) in plain
 * TypeScript, for runtimes without `node:crypto`. WebCrypto
 * cannot stand in: `crypto.subtle` is asynchronous, and the
 * SDK's signing functions are synchronous.
 *
 * All arithmetic is on 32-bit words with no data-dependent
 * branches or table indices, so the time taken depends only
 * on the input lengths.
 */

// Int32Array keeps every word a small integer to V8 (a
// Uint32Array would hand back doubles above 2^31).
const K = new Int32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const BLOCK = 64;

// Scratch space, reused across calls: the functions here are
// synchronous and never re-entered, and reusing it keeps a
// signature from allocating a dozen typed arrays.
const W = new Int32Array(64);
const TAIL = new Uint8Array(2 * BLOCK);
const TAIL_VIEW = new DataView(TAIL.buffer);
const PAD = new Uint8Array(BLOCK);

/** One compression round over the 64-byte block at `offset`. */
function compress(state: Int32Array, view: DataView, offset: number): void {
  const w = W;
  for (let i = 0; i < 16; i++) {
    w[i] = view.getInt32(offset + i * 4);
  }
  for (let i = 16; i < 64; i++) {
    const a = w[i - 15];
    const b = w[i - 2];
    const s0 = ((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3);
    const s1 = ((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10);
    w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
  }
  let a = state[0];
  let b = state[1];
  let c = state[2];
  let d = state[3];
  let e = state[4];
  let f = state[5];
  let g = state[6];
  let h = state[7];
  for (let i = 0; i < 64; i++) {
    const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
    const ch = (e & f) ^ (~e & g);
    const t1 = (h + S1 + ch + K[i] + w[i]) | 0;
    const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
    const maj = (a & b) ^ (a & c) ^ (b & c);
    const t2 = (S0 + maj) | 0;
    h = g;
    g = f;
    f = e;
    e = (d + t1) | 0;
    d = c;
    c = b;
    b = a;
    a = (t1 + t2) | 0;
  }
  state[0] = (state[0] + a) | 0;
  state[1] = (state[1] + b) | 0;
  state[2] = (state[2] + c) | 0;
  state[3] = (state[3] + d) | 0;
  state[4] = (state[4] + e) | 0;
  state[5] = (state[5] + f) | 0;
  state[6] = (state[6] + g) | 0;
  state[7] = (state[7] + h) | 0;
}

/**
 * SHA-256 of `prefix || data`, where `prefix` is one whole
 * block or absent: HMAC's padded keys go in without copying
 * the message behind them.
 */
function digest(prefix: Uint8Array | null, data: Uint8Array): Uint8Array {
  const state = Int32Array.of(
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  );
  if (prefix !== null) {
    compress(state, new DataView(prefix.buffer, prefix.byteOffset, BLOCK), 0);
  }
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const whole = data.length - (data.length % BLOCK);
  for (let offset = 0; offset < whole; offset += BLOCK) {
    compress(state, view, offset);
  }
  // The tail: the leftover bytes, a 0x80 byte, zeros, and the
  // message length in bits as a big-endian 64-bit integer,
  // split into words without leaving the safe-integer range.
  const rest = data.length - whole;
  const tailLength = rest < BLOCK - 8 ? BLOCK : 2 * BLOCK;
  const total = data.length + (prefix === null ? 0 : BLOCK);
  TAIL.fill(0);
  TAIL.set(data.subarray(whole));
  TAIL[rest] = 0x80;
  TAIL_VIEW.setUint32(tailLength - 8, Math.floor(total / 0x20000000));
  TAIL_VIEW.setUint32(tailLength - 4, (total << 3) >>> 0);
  compress(state, TAIL_VIEW, 0);
  if (tailLength === 2 * BLOCK) {
    compress(state, TAIL_VIEW, BLOCK);
  }
  const out = new Uint8Array(32);
  const outView = new DataView(out.buffer);
  for (let i = 0; i < 8; i++) {
    outView.setInt32(i * 4, state[i]);
  }
  return out;
}

/** The SHA-256 digest of `data`. */
export function sha256(data: Uint8Array): Uint8Array {
  return digest(null, data);
}

/** HMAC-SHA-256 of `message` under `key` (RFC 2104). */
export function hmacSha256(key: Uint8Array, message: Uint8Array): Uint8Array {
  const k = key.length > BLOCK ? sha256(key) : key;
  PAD.fill(0x36);
  for (let i = 0; i < k.length; i++) {
    PAD[i] ^= k[i];
  }
  const inner = digest(PAD, message);
  // opad = ipad ^ (0x36 ^ 0x5c), byte for byte.
  for (let i = 0; i < BLOCK; i++) {
    PAD[i] ^= 0x36 ^ 0x5c;
  }
  return digest(PAD, inner);
}
