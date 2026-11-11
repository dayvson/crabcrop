import "./register.mjs";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { test } from "node:test";

const provider = await import("../crabcrop.ts");
const { getImage } = provider;
const table = await import("../../dimension-table.mjs");

// @nuxt/image passes the provider options merged with the
// modifiers as the second argument, and its context, whose
// options carry the runtime config, as the third. On the
// server the runtime config holds the private keys.
const SRC = "https://cdn.example.com/a.jpg";
const options = { base: "https://images.example.com" };
const server = { options: { runtimeConfig: { crabcropSecret: "test-secret", public: {} } } };
const sig = (remainder: string) =>
  createHmac("sha256", "test-secret").update(remainder, "utf8").digest("hex");
const url = (remainder: string) => `https://images.example.com/${sig(remainder)}/${remainder}`;
const image = (modifiers: Record<string, unknown>, ctx: object = server, opts: object = options) =>
  getImage(SRC, { modifiers, ...opts }, ctx).url;

test("cover is the default fit and smart-crops", () => {
  assert.equal(image({ width: 800, height: 600 }), url(`crop:800x600:smart/${SRC}`));
});

test("outside crops to exactly the box (documented approximation)", () => {
  // Nuxt promises an uncropped cover for outside; crabcrop
  // has no such resize, so the provider crops to exactly
  // WxH instead of the old fit-in mapping, whose result
  // could be SMALLER than the box (issue #153).
  assert.equal(image({ width: 800, height: 600, fit: "outside" }), url(`crop:800x600/${SRC}`));
});

test("contain pads with blur bars; inside fits; fill stretches", () => {
  assert.equal(image({ width: 800, height: 600, fit: "contain" }), url(`pad:800x600:blur/${SRC}`));
  assert.equal(image({ width: 800, height: 600, fit: "inside" }), url(`fit-in:800x600/${SRC}`));
  assert.equal(image({ width: 800, height: 600, fit: "fill" }), url(`800x600/${SRC}`));
});

test("single dimensions, format, and quality", () => {
  assert.equal(image({ width: 800 }), url(`800x0/${SRC}`));
  assert.equal(
    image({ width: 800, format: "jpg", quality: 70 }),
    url(`800x0/filters:format(jpeg)/filters:quality(70)/${SRC}`),
  );
});

test("a configured preset leads the operations; no secret means unsafe", () => {
  const publicOnly = { options: { runtimeConfig: { public: {} } } };
  assert.equal(
    image({ width: 800, height: 600 }, publicOnly, { ...options, preset: "card" }),
    `https://images.example.com/unsafe/preset:card/crop:800x600:smart/${SRC}`,
  );
  // @nuxt/image before 1.8 passes no runtime config at all.
  assert.equal(image({ width: 800 }, {}), `https://images.example.com/unsafe/800x0/${SRC}`);
});

test("a missing base is a configuration error", () => {
  assert.throws(() => getImage(SRC, { modifiers: {} }, server), /base/);
});

test("the secret is never a provider option: provider options ship to the browser (issue #381)", () => {
  // @nuxt/image serializes provider options verbatim into
  // #build/image-options.mjs, which client code imports, so
  // a secret there has already leaked: refuse it loudly.
  // (tests/browser-bundles builds a real Nuxt app and checks
  // that module and the client bundle for the secret.)
  for (const secret of ["test-secret", ""]) {
    assert.throws(
      () => getImage(SRC, { modifiers: { width: 800 }, ...options, secret } as never, server),
      /not a provider option.*runtimeConfig\.crabcropSecret/s,
    );
  }
});

test("implausible modifiers throw instead of minting dead URLs", () => {
  // These used to sign URLs the server rejects at parse:
  // quality(undefined), quality(150), format(tiff), and a
  // width past the dimension cap. Quality 0 joined the list
  // when the grammar made it a parse error (issue #147):
  // no encoder gives 0 a meaning distinct from the minimum.
  assert.throws(() => image({ quality: 0 }), /quality/);
  assert.throws(() => image({ quality: "abc" }), /quality/);
  assert.throws(() => image({ quality: 150 }), /quality/);
  assert.throws(() => image({ quality: 80.5 }), /quality/);
  assert.throws(() => image({ format: "tiff" }), /format/);
  assert.throws(() => image({ width: 99999 }), /maximum dimension/);
  assert.throws(() => image({ width: 800, height: 20000 }), /maximum dimension/);
});

test("in a browser context the secret is never used (audit PL-11)", () => {
  // Nuxt's client runtime config has no private keys; even if
  // one were there, a browser-side call stays unsigned.
  (globalThis as { window?: unknown }).window = {};
  try {
    assert.equal(image({ width: 800 }), `https://images.example.com/unsafe/800x0/${SRC}`);
  } finally {
    delete (globalThis as { window?: unknown }).window;
  }
});

test("both @nuxt/image provider shapes resolve to the same getImage", () => {
  // 2.x imports the default export and calls it; 1.x reads
  // the named export.
  assert.equal(provider.default().getImage, getImage);
});

test("dimensions and the base follow the rule all five integrations share (issue #460)", () => {
  // No runtime config in the context: unsigned, like the other suites' rows.
  const unsigned = (modifiers: Record<string, unknown>, base = table.BASE) =>
    getImage(table.SRC, { modifiers, base }, {}).url;
  table.checkDimensionTable({
    width: (width: number) => unsigned({ width }),
    height: (height: number) => unsigned({ height }),
    emptyBase: () => unsigned({ width: 100 }, ""),
  });
});

test("format and quality follow the SDK's grammar (issue #482)", async () => {
  const grammar = await import("../../grammar-table.mjs");
  const unsigned = (modifiers: Record<string, unknown>) =>
    getImage(table.SRC, { modifiers, base: table.BASE }, {}).url;
  grammar.checkFormatTable({
    format: (format: string) => unsigned({ format }),
    quality: (quality: number) => unsigned({ quality }),
  });
});

test("a source under the reserved segment is refused, never signed", () => {
  // With no modifiers the provider forwards src as the whole
  // remainder, and every derived subkey is the signature of a
  // label such as crabcrop/upload-key: the SDK refuses it, on
  // the server and in the browser alike.
  for (const src of ["crabcrop/upload-key", "crabcrop/purge-key", "crabcrop"]) {
    assert.throws(() => getImage(src, { modifiers: {}, ...options }, server), /reserved segment "crabcrop"/);
    assert.throws(() => getImage(src, { modifiers: {}, ...options }, {}), /reserved segment "crabcrop"/);
  }
  // A path that only names a label further in still signs.
  const named = "https://cdn.example.com/crabcrop/upload-key.jpg";
  assert.equal(getImage(named, { modifiers: {}, ...options }, server).url, url(named));
});
