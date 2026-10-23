import "./register.mjs";
import assert from "node:assert/strict";
import { test } from "node:test";

process.env.NEXT_PUBLIC_CRABCROP_BASE = "https://images.example.com";
delete process.env.NEXT_PUBLIC_CRABCROP_PRESET;
delete process.env.CRABCROP_SECRET;

const { default: loader } = await import("../crabcrop-loader.ts");

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
