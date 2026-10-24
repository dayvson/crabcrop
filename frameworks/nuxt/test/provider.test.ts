import "./register.mjs";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { test } from "node:test";

const { getImage } = await import("../crabcrop.ts");

const SRC = "https://cdn.example.com/a.jpg";
const ctx = { options: { base: "https://images.example.com", secret: "test-secret" } };
const sig = (remainder: string) =>
  createHmac("sha256", "test-secret").update(remainder, "utf8").digest("hex");
const url = (remainder: string) => `https://images.example.com/${sig(remainder)}/${remainder}`;

test("cover is the default fit and smart-crops", () => {
  assert.equal(
    getImage(SRC, { modifiers: { width: 800, height: 600 } }, ctx).url,
    url(`crop:800x600:smart/${SRC}`),
  );
});

test("outside crops to exactly the box (documented approximation)", () => {
  // Nuxt promises an uncropped cover for outside; crabcrop
  // has no such resize, so the provider crops to exactly
  // WxH instead of the old fit-in mapping, whose result
  // could be SMALLER than the box (issue #153).
  assert.equal(
    getImage(SRC, { modifiers: { width: 800, height: 600, fit: "outside" } }, ctx).url,
    url(`crop:800x600/${SRC}`),
  );
});

test("contain pads with blur bars; inside fits; fill stretches", () => {
  assert.equal(
    getImage(SRC, { modifiers: { width: 800, height: 600, fit: "contain" } }, ctx).url,
    url(`pad:800x600:blur/${SRC}`),
  );
  assert.equal(
    getImage(SRC, { modifiers: { width: 800, height: 600, fit: "inside" } }, ctx).url,
    url(`fit-in:800x600/${SRC}`),
  );
  assert.equal(
    getImage(SRC, { modifiers: { width: 800, height: 600, fit: "fill" } }, ctx).url,
    url(`800x600/${SRC}`),
  );
});

test("single dimensions, format, and quality", () => {
  assert.equal(
    getImage(SRC, { modifiers: { width: 800 } }, ctx).url,
    url(`800x0/${SRC}`),
  );
  assert.equal(
    getImage(SRC, { modifiers: { width: 800, format: "jpg", quality: 70 } }, ctx).url,
    url(`800x0/filters:format(jpeg)/filters:quality(70)/${SRC}`),
  );
});

test("a configured preset leads the operations", () => {
  const presetCtx = {
    options: { base: "https://images.example.com", preset: "card", secret: null },
  };
  assert.equal(
    getImage(SRC, { modifiers: { width: 800, height: 600 } }, presetCtx).url,
    `https://images.example.com/unsafe/preset:card/crop:800x600:smart/${SRC}`,
  );
});

test("a missing base is a configuration error", () => {
  assert.throws(() => getImage(SRC, { modifiers: {} }, { options: {} as never }), /base/);
});

test("implausible modifiers throw instead of minting dead URLs", () => {
  // These used to sign URLs the server rejects at parse:
  // quality(undefined), quality(150), format(tiff), and a
  // width past the dimension cap. Quality 0 joined the list
  // when the grammar made it a parse error (issue #147):
  // no encoder gives 0 a meaning distinct from the minimum.
  assert.throws(() => getImage(SRC, { modifiers: { quality: 0 } }, ctx), /quality/);
  assert.throws(() => getImage(SRC, { modifiers: { quality: "abc" } }, ctx), /quality/);
  assert.throws(() => getImage(SRC, { modifiers: { quality: 150 } }, ctx), /quality/);
  assert.throws(() => getImage(SRC, { modifiers: { quality: 80.5 } }, ctx), /quality/);
  assert.throws(() => getImage(SRC, { modifiers: { format: "tiff" } }, ctx), /format/);
  assert.throws(() => getImage(SRC, { modifiers: { width: 99999 } }, ctx), /maximum dimension/);
  assert.throws(
    () => getImage(SRC, { modifiers: { width: 800, height: 20000 } }, ctx),
    /maximum dimension/,
  );
});

test("in a browser context the secret is never used (audit PL-11)", () => {
  const ctx = {
    options: { base: "https://images.example.com", secret: "browser-secret-0123" },
  };
  (globalThis as { window?: unknown }).window = {};
  try {
    assert.equal(
      getImage(SRC, { modifiers: { width: 800 } }, ctx).url,
      `https://images.example.com/unsafe/800x0/${SRC}`,
    );
  } finally {
    delete (globalThis as { window?: unknown }).window;
  }
});
