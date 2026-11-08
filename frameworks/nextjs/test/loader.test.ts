import "./register.mjs";
import assert from "node:assert/strict";
import { test } from "node:test";

delete process.env.NEXT_PUBLIC_CRABCROP_PRESET;
delete process.env.CRABCROP_SECRET;

// The loader reads its configuration at import, so the
// unconfigured instance is a second import under its own URL.
process.env.NEXT_PUBLIC_CRABCROP_BASE = "";
const { default: unconfigured } = await import("../crabcrop-loader.ts?unconfigured");
process.env.NEXT_PUBLIC_CRABCROP_BASE = "https://images.example.com";
const { default: loader } = await import("../crabcrop-loader.ts");
const table = await import("../../dimension-table.mjs");

test("width and quality map to operations (unsigned without a secret)", () => {
  assert.equal(
    loader({ src: "https://cdn.example.com/a.jpg", width: 640, quality: 75 }),
    "https://images.example.com/unsafe/640x0/filters:quality(75)/https://cdn.example.com/a.jpg",
  );
});

test("no quality means no quality filter", () => {
  assert.equal(
    loader({ src: "https://cdn.example.com/a.jpg", width: 320 }),
    "https://images.example.com/unsafe/320x0/https://cdn.example.com/a.jpg",
  );
});

test("dimensions and the base follow the rule all five integrations share (issue #460)", () => {
  table.checkDimensionTable({
    width: (width: number) => loader({ src: table.SRC, width }),
    emptyBase: () => unconfigured({ src: table.SRC, width: 640 }),
  });
});
