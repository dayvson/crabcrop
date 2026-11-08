import "./register.mjs";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { test } from "node:test";

const { createCrabcropLoader } = await import("../crabcrop-loader.ts");
const table = await import("../../dimension-table.mjs");

const SRC = "https://cdn.example.com/a.jpg";
const sig = (remainder: string) =>
  createHmac("sha256", "test-secret").update(remainder, "utf8").digest("hex");

test("preset plus scaled dimensions", () => {
  const loader = createCrabcropLoader({
    base: "https://images.example.com",
    preset: "card",
  });
  assert.equal(
    loader({ src: SRC, width: 300, height: 200, scale: 2 }),
    `https://images.example.com/unsafe/preset:card/600x400/${SRC}`,
  );
});

test("width only, signed, with an expiry", () => {
  const loader = createCrabcropLoader({
    base: "https://images.example.com",
    secret: "test-secret",
    expires: 1893456000,
  });
  const remainder = `exp:1893456000/300x0/${SRC}`;
  assert.equal(
    loader({ src: SRC, width: 300 }),
    `https://images.example.com/${sig(remainder)}/${remainder}`,
  );
});

test("no dimensions produces the bare source URL", () => {
  const loader = createCrabcropLoader({ base: "https://images.example.com" });
  assert.equal(loader({ src: SRC }), `https://images.example.com/unsafe/${SRC}`);
});

test("in a browser context the secret is never used (audit PL-11)", () => {
  const loader = createCrabcropLoader({
    base: "https://images.example.com",
    secret: "browser-secret-0123",
  });
  (globalThis as { window?: unknown }).window = {};
  try {
    assert.equal(
      loader({ src: SRC, width: 300 }),
      `https://images.example.com/unsafe/300x0/${SRC}`,
    );
  } finally {
    delete (globalThis as { window?: unknown }).window;
  }
});

test("dimensions and the base follow the rule all five integrations share (issue #460)", () => {
  const loader = createCrabcropLoader({ base: table.BASE });
  table.checkDimensionTable({
    width: (width: number) => loader({ src: table.SRC, width }),
    height: (height: number) => loader({ src: table.SRC, height }),
    emptyBase: () => createCrabcropLoader({ base: "" }),
  });
});

test("the shared rule applies after scale: the URL carries the scaled pixel count", () => {
  const loader = createCrabcropLoader({ base: "https://images.example.com" });
  assert.equal(
    loader({ src: SRC, width: 0.75, height: 100.25, scale: 2 }),
    `https://images.example.com/unsafe/2x201/${SRC}`,
  );
  // A scale that is not a positive number leaves the size out
  // rather than emitting a flip or a 0x0.
  assert.equal(loader({ src: SRC, width: 300, scale: -1 }), `https://images.example.com/unsafe/${SRC}`);
  assert.equal(loader({ src: SRC, width: 300, scale: 0 }), `https://images.example.com/unsafe/${SRC}`);
  assert.throws(
    () => loader({ src: SRC, width: 10000, scale: 2 }),
    /crabcrop loader: width 20000 exceeds the maximum dimension/,
  );
});
