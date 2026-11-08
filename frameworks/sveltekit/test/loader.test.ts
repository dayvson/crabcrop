import "./register.mjs";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { test } from "node:test";

const { createCrabcropLoader } = await import("../crabcrop-loader.ts");
const table = await import("../../dimension-table.mjs");

const SRC = "https://cdn.example.com/a.jpg";
const images = createCrabcropLoader({ base: "https://images.example.com" });

test("width and height map to an exact-size operation (unsigned without a secret)", () => {
  assert.equal(
    images.url(SRC, { width: 640, height: 480 }),
    "https://images.example.com/unsafe/640x480/https://cdn.example.com/a.jpg",
  );
});

test("single dimensions keep the other axis automatic; no options means the bare source", () => {
  assert.equal(
    images.url(SRC, { width: 320 }),
    "https://images.example.com/unsafe/320x0/https://cdn.example.com/a.jpg",
  );
  assert.equal(
    images.url(SRC, { height: 200 }),
    "https://images.example.com/unsafe/0x200/https://cdn.example.com/a.jpg",
  );
  assert.equal(images.url(SRC), "https://images.example.com/unsafe/https://cdn.example.com/a.jpg");
});

test("format and quality map to filters; jpg becomes jpeg", () => {
  assert.equal(
    images.url(SRC, { width: 800, format: "jpg", quality: 70 }),
    "https://images.example.com/unsafe/800x0/filters:format(jpeg)/filters:quality(70)/https://cdn.example.com/a.jpg",
  );
});

test("a configured preset leads the operations", () => {
  const preset = createCrabcropLoader({ base: "https://images.example.com", preset: "card" });
  assert.equal(
    preset.url(SRC, { width: 640 }),
    "https://images.example.com/unsafe/preset:card/640x0/https://cdn.example.com/a.jpg",
  );
});

test("root-relative sources gain the configured origin", () => {
  const withOrigin = createCrabcropLoader({
    base: "https://images.example.com",
    origin: "https://site.example.com",
  });
  assert.equal(
    withOrigin.url("/uploads/a.jpg", { width: 640 }),
    "https://images.example.com/unsafe/640x0/https://site.example.com/uploads/a.jpg",
  );
});

// The signing checks run against spec/test-vectors.json:
// the remainders are ones this loader emits verbatim, so
// the expected URLs carry the vectors' signatures byte for
// byte (with the vectors' sources, under this base).
test("signed output reproduces the spec test vector for the README worked example", () => {
  const signed = createCrabcropLoader({ base: "https://images.example.com", secret: "your-secret" });
  assert.equal(
    signed.url("https://example.com/photo.jpg", { width: 300, height: 200 }),
    "https://images.example.com/3c1f6c180116661877bd4c85fd201a178087d74c1cc188cc584e819d901308c6/300x200/https://example.com/photo.jpg",
  );
});

test("signed output reproduces the spec test vector for an expiring URL", () => {
  const signed = createCrabcropLoader({
    base: "https://images.example.com",
    secret: "s3cr3t",
    expires: 1893456000,
  });
  assert.equal(
    signed.url("https://example.com/photo.jpg", { width: 300, height: 200 }),
    "https://images.example.com/878f5cb69f5ac5770161176099fcdd64dcce17d6e6f846a3108ffacfa4644386/exp:1893456000/300x200/https://example.com/photo.jpg",
  );
});

// Expected signatures are computed here with node:crypto
// directly, not through the SDK, so the loader's remainder
// construction is what is actually under test.
const sig = (remainder: string) =>
  createHmac("sha256", "test-secret").update(remainder, "utf8").digest("hex");

test("preset and sizing sign over the assembled remainder", () => {
  const signed = createCrabcropLoader({
    base: "https://images.example.com",
    preset: "card",
    secret: "test-secret",
  });
  const remainder = `preset:card/640x0/${SRC}`;
  assert.equal(
    signed.url(SRC, { width: 640 }),
    `https://images.example.com/${sig(remainder)}/${remainder}`,
  );
});

test("srcset emits one signed-or-unsigned entry per width, options carried through", () => {
  assert.equal(
    images.srcset(SRC, [320, 640]),
    "https://images.example.com/unsafe/320x0/https://cdn.example.com/a.jpg 320w, " +
      "https://images.example.com/unsafe/640x0/https://cdn.example.com/a.jpg 640w",
  );
  const preset = createCrabcropLoader({ base: "https://images.example.com", preset: "card" });
  assert.equal(
    preset.srcset(SRC, [320], { format: "webp" }),
    "https://images.example.com/unsafe/preset:card/320x0/filters:format(webp)/https://cdn.example.com/a.jpg 320w",
  );
});

test("the factory requires a base; invalid options throw instead of minting dead URLs", () => {
  assert.throws(() => createCrabcropLoader({ base: "" }), /`base` config is required/);
  assert.throws(() => images.url(SRC, { width: 20000 }), /maximum dimension/);
  assert.throws(() => images.url(SRC, { format: "bmp" }), /unsupported format/);
  assert.throws(() => images.url(SRC, { quality: 0 }), /quality must be/);
  assert.throws(() => images.srcset(SRC, [0]), /srcset width must be/);
  assert.throws(() => images.srcset(SRC, [640.5]), /srcset width must be/);
});

test("dimensions and the base follow the rule all five integrations share (issue #460)", () => {
  const loader = createCrabcropLoader({ base: table.BASE });
  table.checkDimensionTable({
    width: (width: number) => loader.url(table.SRC, { width }),
    height: (height: number) => loader.url(table.SRC, { height }),
    emptyBase: () => createCrabcropLoader({ base: "" }),
  });
});

test("format and quality follow the SDK's grammar (issue #482)", async () => {
  const grammar = await import("../../grammar-table.mjs");
  const loader = createCrabcropLoader({ base: table.BASE });
  grammar.checkFormatTable({
    format: (format: string) => loader.url(table.SRC, { format }),
    quality: (quality: number) => loader.url(table.SRC, { quality }),
  });
});
