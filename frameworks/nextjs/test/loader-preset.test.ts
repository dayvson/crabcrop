import "./register.mjs";
import assert from "node:assert/strict";
import { test } from "node:test";

process.env.NEXT_PUBLIC_CRABCROP_BASE = "https://images.example.com";
process.env.NEXT_PUBLIC_CRABCROP_PRESET = "card";
process.env.NEXT_PUBLIC_CRABCROP_ORIGIN = "https://site.example.com";
delete process.env.CRABCROP_SECRET;

const { default: loader } = await import("../crabcrop-loader.ts");

test("preset first, then per-width sizing; quality is the preset's business", () => {
  assert.equal(
    loader({ src: "https://cdn.example.com/a.jpg", width: 640, quality: 75 }),
    "https://images.example.com/unsafe/preset:card/640x0/https://cdn.example.com/a.jpg",
  );
});

test("relative sources gain the configured origin", () => {
  assert.equal(
    loader({ src: "/uploads/a.jpg", width: 640 }),
    "https://images.example.com/unsafe/preset:card/640x0/https://site.example.com/uploads/a.jpg",
  );
});
