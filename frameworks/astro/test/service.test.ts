import "./register.mjs";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { test } from "node:test";

const { default: service } = await import("../crabcrop-service.ts");
const table = await import("../../dimension-table.mjs");

const SRC = "https://cdn.example.com/a.jpg";
const cfg = (config: Record<string, unknown>) => ({ service: { config } });
const unsigned = cfg({ base: "https://images.example.com" });

test("width and height map to an exact-size operation (unsigned without a secret)", () => {
  assert.equal(
    service.getURL({ src: SRC, width: 640, height: 480 }, unsigned),
    "https://images.example.com/unsafe/640x480/https://cdn.example.com/a.jpg",
  );
});

test("single dimensions keep the other axis automatic", () => {
  assert.equal(
    service.getURL({ src: SRC, width: 320 }, unsigned),
    "https://images.example.com/unsafe/320x0/https://cdn.example.com/a.jpg",
  );
  assert.equal(
    service.getURL({ src: SRC, height: 200 }, unsigned),
    "https://images.example.com/unsafe/0x200/https://cdn.example.com/a.jpg",
  );
});

test("format and quality map to filters; jpg becomes jpeg", () => {
  assert.equal(
    service.getURL({ src: SRC, width: 800, format: "jpg", quality: 70 }, unsigned),
    "https://images.example.com/unsafe/800x0/filters:format(jpeg)/filters:quality(70)/https://cdn.example.com/a.jpg",
  );
});

test("Astro's named quality presets resolve to their documented values", () => {
  for (const [name, value] of [["low", 25], ["mid", 50], ["high", 80], ["max", 100]] as const) {
    assert.equal(
      service.getURL({ src: SRC, width: 800, quality: name }, unsigned),
      `https://images.example.com/unsafe/800x0/filters:quality(${value})/https://cdn.example.com/a.jpg`,
    );
  }
});

test("a configured preset leads the operations", () => {
  assert.equal(
    service.getURL({ src: SRC, width: 640 }, cfg({ base: "https://images.example.com", preset: "card" })),
    "https://images.example.com/unsafe/preset:card/640x0/https://cdn.example.com/a.jpg",
  );
});

test("root-relative sources gain the configured origin; metadata sources resolve", () => {
  const withOrigin = cfg({ base: "https://images.example.com", origin: "https://site.example.com" });
  assert.equal(
    service.getURL({ src: "/uploads/a.jpg", width: 640 }, withOrigin),
    "https://images.example.com/unsafe/640x0/https://site.example.com/uploads/a.jpg",
  );
  assert.equal(
    service.getURL(
      { src: { src: "/_astro/hero.jpg", width: 1600, height: 900, format: "jpg" }, width: 800 },
      withOrigin,
    ),
    "https://images.example.com/unsafe/800x0/https://site.example.com/_astro/hero.jpg",
  );
});

// The signing checks run against spec/test-vectors.json:
// the remainders are ones this service emits verbatim, so
// the expected URLs carry the vectors' signatures byte for
// byte (with the vectors' sources, under this base).
test("signed output reproduces the spec test vector for the README worked example", () => {
  const signed = cfg({ base: "https://images.example.com", secret: "your-secret" });
  assert.equal(
    service.getURL({ src: "https://example.com/photo.jpg", width: 300, height: 200 }, signed),
    "https://images.example.com/3c1f6c180116661877bd4c85fd201a178087d74c1cc188cc584e819d901308c6/300x200/https://example.com/photo.jpg",
  );
});

test("signed output reproduces the spec test vector for an expiring URL", () => {
  const signed = cfg({
    base: "https://images.example.com",
    secret: "s3cr3t",
    expires: 1893456000,
  });
  assert.equal(
    service.getURL({ src: "https://example.com/photo.jpg", width: 300, height: 200 }, signed),
    "https://images.example.com/878f5cb69f5ac5770161176099fcdd64dcce17d6e6f846a3108ffacfa4644386/exp:1893456000/300x200/https://example.com/photo.jpg",
  );
});

// Expected signatures are computed here with node:crypto
// directly, not through the SDK, so the service's remainder
// construction is what is actually under test.
const sig = (remainder: string) =>
  createHmac("sha256", "test-secret").update(remainder, "utf8").digest("hex");

test("preset and sizing sign over the assembled remainder", () => {
  const signed = cfg({ base: "https://images.example.com", preset: "card", secret: "test-secret" });
  const remainder = `preset:card/640x0/${SRC}`;
  assert.equal(
    service.getURL({ src: SRC, width: 640 }, signed),
    `https://images.example.com/${sig(remainder)}/${remainder}`,
  );
});

test("validateOptions requires the base config and otherwise passes options through", () => {
  assert.throws(() => service.validateOptions({ src: SRC }, cfg({})), /`base` service config is required/);
  assert.throws(() => service.getURL({ src: SRC, width: 100 }, cfg({})), /`base` service config is required/);
  const options = { src: SRC, width: 100 };
  assert.equal(service.validateOptions(options, unsigned), options);
});

test("invalid dimensions, formats, and qualities throw instead of minting dead URLs", () => {
  assert.throws(() => service.getURL({ src: SRC, width: 20000 }, unsigned), /maximum dimension/);
  assert.throws(() => service.getURL({ src: SRC, width: 100, format: "svg" }, unsigned), /unsupported format/);
  assert.throws(() => service.getURL({ src: SRC, width: 100, quality: 150 }, unsigned), /quality must be/);
  assert.throws(() => service.getURL({ src: SRC, width: 100, quality: "ultra" }, unsigned), /quality must be/);
});

test("getHTMLAttributes strips the transform fields and defaults loading and decoding", () => {
  const attrs = service.getHTMLAttributes(
    { src: SRC, width: 640, height: 480, format: "webp", quality: 80, alt: "A photo" },
    unsigned,
  );
  assert.deepEqual(attrs, {
    alt: "A photo",
    width: 640,
    height: 480,
    loading: "lazy",
    decoding: "async",
  });
});

test("getHTMLAttributes fills the missing dimension from a metadata source's aspect", () => {
  const attrs = service.getHTMLAttributes(
    { src: { src: "/_astro/hero.jpg", width: 1600, height: 900, format: "jpg" }, width: 800, alt: "" },
    unsigned,
  );
  assert.equal(attrs.width, 800);
  assert.equal(attrs.height, 450);
});

test("getSrcSet maps widths to transforms with w descriptors, scaling the box height", () => {
  const values = service.getSrcSet(
    { src: SRC, width: 800, height: 600, widths: [400, 800] },
    unsigned,
  );
  assert.deepEqual(
    values.map((v) => [v.transform.width, v.transform.height, v.descriptor]),
    [
      [400, 300, "400w"],
      [800, 600, "800w"],
    ],
  );
  // Each variant is a URL this service itself would emit.
  assert.equal(
    service.getURL(values[0].transform, unsigned),
    "https://images.example.com/unsafe/400x300/https://cdn.example.com/a.jpg",
  );
});

test("getSrcSet maps densities to transforms with x descriptors", () => {
  const values = service.getSrcSet({ src: SRC, width: 400, densities: [1, 2] }, unsigned);
  assert.deepEqual(
    values.map((v) => [v.transform.width, v.descriptor]),
    [
      [400, "1x"],
      [800, "2x"],
    ],
  );
  assert.deepEqual(service.getSrcSet({ src: SRC, width: 400 }, unsigned), []);
});

test("dimensions and the base follow the rule all five integrations share (issue #460)", () => {
  const config = cfg({ base: table.BASE });
  table.checkDimensionTable({
    width: (width: number) => service.getURL({ src: table.SRC, width }, config),
    height: (height: number) => service.getURL({ src: table.SRC, height }, config),
    emptyBase: () => service.getURL({ src: table.SRC, width: 100 }, cfg({ base: "" })),
  });
});

test("format and quality follow the SDK's grammar (issue #482)", async () => {
  const grammar = await import("../../grammar-table.mjs");
  const config = cfg({ base: table.BASE });
  grammar.checkFormatTable({
    format: (format: string) => service.getURL({ src: table.SRC, format }, config),
    quality: (quality: number) => service.getURL({ src: table.SRC, quality }, config),
  });
});
