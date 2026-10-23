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
