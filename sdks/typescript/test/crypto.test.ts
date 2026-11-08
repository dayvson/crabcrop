// The portable crypto backend (the one browser bundles get)
// checked against node:crypto: SHA-256, HMAC-SHA-256, and
// AES-256-GCM on thousands of random inputs across the
// block-boundary lengths, plus the byte encodings and the
// unseal decision. `npm test` also reruns the whole suite
// under --conditions=browser, so the spec vectors in
// vectors.test.ts pass on both backends.
import assert from "node:assert/strict";
import {
  createCipheriv,
  createDecipheriv,
  createHash,
  createHmac,
  randomBytes,
  randomInt,
} from "node:crypto";
import { test } from "node:test";
import * as active from "#crypto";
import { fromBase64Url, toBase64Url, toHex, utf8Strict } from "../src/bytes.ts";
import { aes256GcmOpen, aes256GcmSeal } from "../src/crypto/aes-gcm.ts";
import * as portable from "../src/crypto/portable.ts";
import { hmacSha256, sha256 } from "../src/crypto/sha256.ts";
import { sealWithNonce, unseal } from "../src/index.ts";

const hex = (b: Uint8Array) => Buffer.from(b).toString("hex");

// Lengths around SHA-256's 64-byte block and its 55/56-byte
// padding split, around AES's 16-byte block, and multi-block.
const EDGES = [0, 1, 15, 16, 17, 31, 32, 33, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 191, 192, 193, 1000, 4096];

test("the active backend follows the resolution condition", () => {
  const browserRun = process.execArgv.includes("--conditions=browser");
  assert.equal(active.backend, browserRun ? "portable" : "node");
});

test("sha256 matches node:crypto for every length to 300, the edges, and random lengths", () => {
  const lengths = [...Array.from({ length: 301 }, (_, i) => i), ...EDGES];
  for (let i = 0; i < 1000; i++) {
    lengths.push(randomInt(0, 20000));
  }
  for (const n of lengths) {
    const data = randomBytes(n);
    assert.equal(hex(sha256(data)), createHash("sha256").update(data).digest("hex"), `length ${n}`);
  }
  assert.equal(
    hex(sha256(new TextEncoder().encode("abc"))),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  );
});

test("hmacSha256 matches node:crypto on random keys and messages", () => {
  const keyLengths = [0, 1, 32, 63, 64, 65, 100, 128, 129, 300];
  for (let i = 0; i < 4000; i++) {
    const keyLength = i < 400 ? keyLengths[i % keyLengths.length] : randomInt(0, 200);
    const messageLength = i < 400 ? EDGES[i % EDGES.length] : randomInt(0, 600);
    const key = randomBytes(keyLength);
    const message = randomBytes(messageLength);
    assert.equal(
      hex(hmacSha256(key, message)),
      createHmac("sha256", key).update(message).digest("hex"),
      `key ${keyLength} message ${messageLength}`,
    );
  }
});

/** A random string mixing ASCII, multibyte, astral, and lone surrogates. */
function randomText(length: number): string {
  const pieces = ["a", "/", ":", "é", "€", "😀", "\u0000", "\ud800", "\udfff", "﻿"];
  let out = "";
  for (let i = 0; i < length; i++) {
    out += pieces[randomInt(0, pieces.length)];
  }
  return out;
}

test("string keys and messages encode as node:crypto encodes them (lone surrogates included)", () => {
  for (let i = 0; i < 2000; i++) {
    const key = randomText(randomInt(0, 80));
    const message = randomText(randomInt(0, 120));
    assert.equal(
      hex(portable.hmacSha256(key, message)),
      createHmac("sha256", key).update(message, "utf8").digest("hex"),
    );
    assert.equal(hex(active.hmacSha256(key, message)), hex(portable.hmacSha256(key, message)));
  }
  // A missing key is an error on both backends, never a
  // signature under the string "undefined".
  assert.throws(() => portable.hmacSha256(undefined as unknown as string, "x"), TypeError);
  assert.throws(() => createHmac("sha256", undefined as unknown as string), TypeError);
});

function nodeSeal(key: Uint8Array, nonce: Uint8Array, aad: Uint8Array, plaintext: Uint8Array) {
  const cipher = createCipheriv("aes-256-gcm", key, nonce);
  cipher.setAAD(aad);
  return Buffer.concat([cipher.update(plaintext), cipher.final(), cipher.getAuthTag()]);
}

test("AES-256-GCM matches node:crypto on random keys, nonces, AAD, and plaintexts", () => {
  for (let i = 0; i < 2000; i++) {
    const key = randomBytes(32);
    const nonce = randomBytes(12);
    const aad = randomBytes(i % 3 === 0 ? 17 : randomInt(0, 40));
    const length = i < EDGES.length * 4 ? EDGES[i % EDGES.length] : randomInt(0, 2100);
    const plaintext = randomBytes(length);
    const sealed = aes256GcmSeal(key, nonce, aad, plaintext);
    assert.equal(hex(sealed), hex(nodeSeal(key, nonce, aad, plaintext)), `length ${length}`);
    assert.equal(hex(aes256GcmOpen(key, nonce, aad, sealed) as Uint8Array), hex(plaintext));
    // Any flipped bit, in the ciphertext, the tag, or the AAD,
    // fails to open.
    const tampered = Uint8Array.from(sealed);
    tampered[randomInt(0, tampered.length)] ^= 1 << randomInt(0, 8);
    assert.equal(aes256GcmOpen(key, nonce, aad, tampered), null);
    const otherAad = Uint8Array.from([...aad, 0]);
    assert.equal(aes256GcmOpen(key, nonce, otherAad, sealed), null);
  }
  assert.equal(aes256GcmOpen(randomBytes(32), randomBytes(12), new Uint8Array(0), randomBytes(15)), null);
});

test("AES-256-GCM reproduces the GCM specification's AES-256 test cases", () => {
  // McGrew and Viega, "The Galois/Counter Mode of Operation",
  // test cases 13 and 14: the all-zero key and nonce.
  const zero = (n: number) => new Uint8Array(n);
  assert.equal(
    hex(aes256GcmSeal(zero(32), zero(12), zero(0), zero(0))),
    "530f8afbc74536b9a963b4f1c4cb738b",
  );
  assert.equal(
    hex(aes256GcmSeal(zero(32), zero(12), zero(0), zero(16))),
    "cea7403d4d606b6e074ec5d3baf39d18" + "d0d1c8a799996bf0265b98b5d48ab919",
  );
});

test("the two backends agree on sealing and opening", () => {
  for (let i = 0; i < 300; i++) {
    const key = randomBytes(32);
    const nonce = randomBytes(12);
    const aad = randomBytes(randomInt(0, 24));
    const plaintext = randomBytes(randomInt(0, 300));
    const sealed = portable.aes256GcmSeal(key, nonce, aad, plaintext);
    assert.equal(hex(active.aes256GcmSeal(key, nonce, aad, plaintext)), hex(sealed));
    assert.equal(hex(active.aes256GcmOpen(key, nonce, aad, sealed) as Uint8Array), hex(plaintext));
    sealed[0] ^= 0x80;
    assert.equal(active.aes256GcmOpen(key, nonce, aad, sealed), null);
    assert.equal(portable.aes256GcmOpen(key, nonce, aad, sealed), null);
  }
});

test("timingSafeEqual and randomBytes behave like node:crypto's", () => {
  for (let i = 0; i < 500; i++) {
    const a = randomBytes(randomInt(0, 40));
    const b = i % 2 === 0 ? Uint8Array.from(a) : randomBytes(randomInt(0, 40));
    const same = Buffer.compare(a, b) === 0;
    assert.equal(portable.timingSafeEqual(a, b), same);
    assert.equal(active.timingSafeEqual(a, b), same);
  }
  const nonce = portable.randomBytes(12);
  assert.equal(nonce.length, 12);
  assert.notEqual(hex(nonce), hex(portable.randomBytes(12)));
});

test("hex and base64url match Buffer's encodings", () => {
  for (let n = 0; n < 200; n++) {
    const data = randomBytes(n);
    assert.equal(toHex(data), Buffer.from(data).toString("hex"));
    const text = toBase64Url(data);
    assert.equal(text, Buffer.from(data).toString("base64url"));
    assert.equal(hex(fromBase64Url(text) as Uint8Array), hex(data));
  }
  // Strict: padding, the standard alphabet's + and /, spaces,
  // and an impossible length are refused outright.
  for (const bad of ["AAA=", "AA==", "A+AA", "A/AA", "AA AA", "A", "AAAAA", "AAÀA"]) {
    assert.equal(fromBase64Url(bad), null, bad);
  }
});

test("strict UTF-8 decoding accepts exactly what round-trips through Buffer", () => {
  const samples = [
    [0xef, 0xbb, 0xbf, 0x61], // a BOM is kept, as Buffer keeps it
    [0xed, 0xa0, 0x80], // an encoded surrogate
    [0xc0, 0x80], // overlong
    [0xf4, 0x8f, 0xbf, 0xbf], // U+10FFFF
    [0xf4, 0x90, 0x80, 0x80], // past U+10FFFF
    [0xe2, 0x82], // truncated
  ].map((b) => Uint8Array.from(b));
  for (let i = 0; i < 3000; i++) {
    samples.push(randomBytes(randomInt(0, 12)).map((b) => (i % 2 ? b : b & 0x7f) | (i % 5 === 0 ? 0x80 : 0)));
  }
  for (const bytes of samples) {
    const lossless = Buffer.from(Buffer.from(bytes).toString("utf8"), "utf8").equals(bytes);
    const decoded = utf8Strict(bytes);
    assert.equal(decoded !== null, lossless, hex(bytes));
    if (decoded !== null) {
      assert.equal(decoded, Buffer.from(bytes).toString("utf8"));
    }
  }
});

/**
 * The unseal of the release before the portable backend,
 * on node:crypto and Buffer directly: the reference the new
 * decision must reproduce.
 */
function referenceUnseal(secret: string, segment: string): string | null {
  let token = segment;
  const colon = segment.indexOf(":");
  if (colon >= 0) {
    if (!/^[A-Za-z0-9_-]{1,32}$/.test(segment.slice(0, colon))) {
      return null;
    }
    token = segment.slice(colon + 1);
  }
  if (!token.startsWith("s1.")) {
    return null;
  }
  const body = token.slice(3);
  const raw = Buffer.from(body, "base64url");
  if (raw.length < 28 || raw.toString("base64url") !== body) {
    return null;
  }
  const key = createHmac("sha256", secret).update("crabcrop/seal-key").digest();
  const decipher = createDecipheriv("aes-256-gcm", key, raw.subarray(0, 12));
  decipher.setAAD(Buffer.from("crabcrop/sealed/1", "utf8"));
  decipher.setAuthTag(raw.subarray(raw.length - 16));
  let plain: Buffer;
  try {
    plain = Buffer.concat([decipher.update(raw.subarray(12, raw.length - 16)), decipher.final()]);
  } catch {
    return null;
  }
  const text = plain.toString("utf8");
  return Buffer.from(text, "utf8").equals(plain) ? text : null;
}

test("unseal accepts and refuses exactly what the node:crypto implementation did", () => {
  const secret = "unseal-parity";
  const key = createHmac("sha256", secret).update("crabcrop/seal-key").digest();
  const aad = Buffer.from("crabcrop/sealed/1");
  const tokens: string[] = [];
  for (let i = 0; i < 400; i++) {
    // Valid tokens over text, and over raw bytes that are not
    // always valid UTF-8 (sealed directly under the seal key).
    const nonce = randomBytes(12);
    if (i % 2 === 0) {
      tokens.push(sealWithNonce(secret, randomText(randomInt(0, 40)).replace(/[\ud800-\udfff]/g, "x"), nonce));
    } else {
      const body = Buffer.concat([nonce, nodeSeal(key, nonce, aad, randomBytes(randomInt(0, 20)))]);
      tokens.push(`s1.${body.toString("base64url")}`);
    }
  }
  const variants: string[] = [];
  for (const t of tokens) {
    const at = randomInt(3, t.length);
    variants.push(
      t,
      `k1:${t}`,
      `bad key:${t}`,
      `${t}=`,
      `${t.slice(0, at)} ${t.slice(at)}`,
      `${t.slice(0, at)}+${t.slice(at + 1)}`,
      `${t.slice(0, at)}${t[at] === "A" ? "B" : "A"}${t.slice(at + 1)}`,
      t.slice(0, -1),
      `${t}A`,
      `s2.${t.slice(3)}`,
    );
  }
  for (const segment of variants) {
    const expected = referenceUnseal(secret, segment);
    let got: string | null;
    try {
      got = unseal(secret, segment);
    } catch (error) {
      assert.match(String(error), /did not open/);
      got = null;
    }
    assert.equal(got, expected, segment);
  }
});
