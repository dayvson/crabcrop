/**
 * AES-256 (FIPS 197) in GCM mode (NIST SP 800-38D) with a
 * 96-bit nonce and a 128-bit tag, in plain TypeScript, for
 * runtimes without `node:crypto`. Like SHA-256 here, it backs
 * synchronous functions (`seal`, `unseal`), which WebCrypto's
 * asynchronous `subtle.encrypt` cannot.
 *
 * GHASH is computed with masks rather than branches. The AES
 * rounds index the S-box by key-dependent bytes, so they are
 * not hardened against cache-timing observers; Node keeps
 * OpenSSL through the package's `node` condition, and this
 * implementation serves the browser and edge builds.
 */

const SBOX = new Uint8Array(256);
{
  // The S-box is the multiplicative inverse in GF(2^8)
  // followed by the FIPS 197 affine map. Walk p over the
  // powers of the generator 3 while q walks the matching
  // inverses (powers of 3^-1), so sbox[p] = affine(q).
  let p = 1;
  let q = 1;
  do {
    p = p ^ ((p << 1) & 0xff) ^ (p & 0x80 ? 0x1b : 0);
    q ^= q << 1;
    q ^= q << 2;
    q ^= q << 4;
    q &= 0xff;
    if (q & 0x80) {
      q ^= 0x09;
    }
    const rotl = (x: number, n: number) => ((x << n) | (x >>> (8 - n))) & 0xff;
    SBOX[p] = q ^ rotl(q, 1) ^ rotl(q, 2) ^ rotl(q, 3) ^ rotl(q, 4) ^ 0x63;
  } while (p !== 1);
  SBOX[0] = 0x63;
}

/** Multiply by x in GF(2^8), without a branch on the high bit. */
const xtime = (x: number): number => ((x << 1) ^ ((x >>> 7) * 0x1b)) & 0xff;

const ROUNDS = 14;

/** The AES-256 key schedule: 15 round keys, 240 bytes. */
function expandKey(key: Uint8Array): Uint8Array {
  if (key.length !== 32) {
    throw new Error(`AES-256 needs a 32-byte key, got ${key.length}`);
  }
  const w = new Uint8Array(16 * (ROUNDS + 1));
  w.set(key);
  let rcon = 1;
  for (let i = 32; i < w.length; i += 4) {
    let t0 = w[i - 4];
    let t1 = w[i - 3];
    let t2 = w[i - 2];
    let t3 = w[i - 1];
    if (i % 32 === 0) {
      // RotWord, SubWord, and the round constant.
      const first = t0;
      t0 = SBOX[t1] ^ rcon;
      t1 = SBOX[t2];
      t2 = SBOX[t3];
      t3 = SBOX[first];
      rcon = xtime(rcon);
    } else if (i % 32 === 16) {
      t0 = SBOX[t0];
      t1 = SBOX[t1];
      t2 = SBOX[t2];
      t3 = SBOX[t3];
    }
    w[i] = w[i - 32] ^ t0;
    w[i + 1] = w[i - 31] ^ t1;
    w[i + 2] = w[i - 30] ^ t2;
    w[i + 3] = w[i - 29] ^ t3;
  }
  return w;
}

/** Encrypt one 16-byte block from `input` into `output`. */
function encryptBlock(roundKeys: Uint8Array, input: Uint8Array, output: Uint8Array): void {
  // The state is column-major: byte r of column c is s[r + 4c].
  const s = new Uint8Array(16);
  const t = new Uint8Array(16);
  for (let i = 0; i < 16; i++) {
    s[i] = input[i] ^ roundKeys[i];
  }
  for (let round = 1; round <= ROUNDS; round++) {
    // SubBytes and ShiftRows: row r rotates left by r columns.
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        t[r + 4 * c] = SBOX[s[r + 4 * ((c + r) & 3)]];
      }
    }
    if (round < ROUNDS) {
      for (let c = 0; c < 16; c += 4) {
        const a0 = t[c];
        const a1 = t[c + 1];
        const a2 = t[c + 2];
        const a3 = t[c + 3];
        const all = a0 ^ a1 ^ a2 ^ a3;
        // MixColumns: b_i = a_i ^ all ^ 2·(a_i ^ a_{i+1}).
        t[c] = a0 ^ all ^ xtime(a0 ^ a1);
        t[c + 1] = a1 ^ all ^ xtime(a1 ^ a2);
        t[c + 2] = a2 ^ all ^ xtime(a2 ^ a3);
        t[c + 3] = a3 ^ all ^ xtime(a3 ^ a0);
      }
    }
    const k = round * 16;
    for (let i = 0; i < 16; i++) {
      s[i] = t[i] ^ roundKeys[k + i];
    }
  }
  output.set(s);
}

/**
 * Multiply two GF(2^128) elements in GCM's bit order, each as
 * four big-endian 32-bit words (SP 800-38D, Algorithm 1). The
 * loop runs all 128 steps with masks, never a branch on a
 * secret bit.
 */
function gmul(x: Int32Array, h: Int32Array): void {
  let z0 = 0;
  let z1 = 0;
  let z2 = 0;
  let z3 = 0;
  let v0 = h[0];
  let v1 = h[1];
  let v2 = h[2];
  let v3 = h[3];
  for (let i = 0; i < 128; i++) {
    const bit = (x[i >>> 5] >>> (31 - (i & 31))) & 1;
    const take = -bit;
    z0 ^= v0 & take;
    z1 ^= v1 & take;
    z2 ^= v2 & take;
    z3 ^= v3 & take;
    const reduce = -(v3 & 1);
    v3 = (v3 >>> 1) | (v2 << 31);
    v2 = (v2 >>> 1) | (v1 << 31);
    v1 = (v1 >>> 1) | (v0 << 31);
    v0 = (v0 >>> 1) ^ (0xe1000000 & reduce);
  }
  x[0] = z0;
  x[1] = z1;
  x[2] = z2;
  x[3] = z3;
}

function readWords(bytes: Uint8Array): Int32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, 16);
  return Int32Array.of(view.getInt32(0), view.getInt32(4), view.getInt32(8), view.getInt32(12));
}

/** GHASH_H over the AAD and ciphertext, then their bit lengths. */
function ghash(h: Uint8Array, aad: Uint8Array, ciphertext: Uint8Array): Uint8Array {
  const hw = readWords(h);
  const x = new Int32Array(4);
  const block = new Uint8Array(16);
  const absorb = (data: Uint8Array): void => {
    for (let off = 0; off < data.length; off += 16) {
      block.fill(0);
      block.set(data.subarray(off, off + 16));
      const bw = readWords(block);
      for (let i = 0; i < 4; i++) {
        x[i] ^= bw[i];
      }
      gmul(x, hw);
    }
  };
  absorb(aad);
  absorb(ciphertext);
  const lengths = new DataView(block.buffer);
  lengths.setUint32(0, Math.floor(aad.length / 0x20000000));
  lengths.setUint32(4, (aad.length << 3) >>> 0);
  lengths.setUint32(8, Math.floor(ciphertext.length / 0x20000000));
  lengths.setUint32(12, (ciphertext.length << 3) >>> 0);
  const lw = readWords(block);
  for (let i = 0; i < 4; i++) {
    x[i] ^= lw[i];
  }
  gmul(x, hw);
  const out = new Uint8Array(16);
  const outView = new DataView(out.buffer);
  for (let i = 0; i < 4; i++) {
    outView.setInt32(i * 4, x[i]);
  }
  return out;
}

/** CTR mode from inc32(J0), XORing `input` into `output`. */
function ctr(roundKeys: Uint8Array, j0: Uint8Array, input: Uint8Array, output: Uint8Array): void {
  const counter = j0.slice();
  const view = new DataView(counter.buffer);
  const stream = new Uint8Array(16);
  for (let off = 0; off < input.length; off += 16) {
    view.setUint32(12, (view.getUint32(12) + 1) >>> 0);
    encryptBlock(roundKeys, counter, stream);
    const end = Math.min(off + 16, input.length);
    for (let i = off; i < end; i++) {
      output[i] = input[i] ^ stream[i - off];
    }
  }
}

interface GcmSetup {
  roundKeys: Uint8Array;
  h: Uint8Array;
  j0: Uint8Array;
}

function setup(key: Uint8Array, nonce: Uint8Array): GcmSetup {
  if (nonce.length !== 12) {
    throw new Error(`AES-GCM here takes a 12-byte nonce, got ${nonce.length}`);
  }
  const roundKeys = expandKey(key);
  const h = new Uint8Array(16);
  encryptBlock(roundKeys, new Uint8Array(16), h);
  const j0 = new Uint8Array(16);
  j0.set(nonce);
  j0[15] = 1;
  return { roundKeys, h, j0 };
}

/** The tag: E(K, J0) XOR GHASH(A, C). */
function tagFor(g: GcmSetup, aad: Uint8Array, ciphertext: Uint8Array): Uint8Array {
  const tag = ghash(g.h, aad, ciphertext);
  const mask = new Uint8Array(16);
  encryptBlock(g.roundKeys, g.j0, mask);
  for (let i = 0; i < 16; i++) {
    tag[i] ^= mask[i];
  }
  return tag;
}

/** AES-256-GCM encryption: returns ciphertext || 16-byte tag. */
export function aes256GcmSeal(
  key: Uint8Array,
  nonce: Uint8Array,
  aad: Uint8Array,
  plaintext: Uint8Array,
): Uint8Array {
  const g = setup(key, nonce);
  const out = new Uint8Array(plaintext.length + 16);
  ctr(g.roundKeys, g.j0, plaintext, out);
  out.set(tagFor(g, aad, out.subarray(0, plaintext.length)), plaintext.length);
  return out;
}

/**
 * AES-256-GCM decryption of ciphertext || tag. Returns the
 * plaintext, or null when the tag does not verify; nothing is
 * decrypted before the tag is checked.
 */
export function aes256GcmOpen(
  key: Uint8Array,
  nonce: Uint8Array,
  aad: Uint8Array,
  sealed: Uint8Array,
): Uint8Array | null {
  if (sealed.length < 16) {
    return null;
  }
  const g = setup(key, nonce);
  const ciphertext = sealed.subarray(0, sealed.length - 16);
  const expected = tagFor(g, aad, ciphertext);
  let diff = 0;
  for (let i = 0; i < 16; i++) {
    diff |= expected[i] ^ sealed[ciphertext.length + i];
  }
  if (diff !== 0) {
    return null;
  }
  const plaintext = new Uint8Array(ciphertext.length);
  ctr(g.roundKeys, g.j0, ciphertext, plaintext);
  return plaintext;
}
