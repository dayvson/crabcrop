import "./register.mjs";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { test } from "node:test";

const { createCrabcropLoader } = await import("../crabcrop-loader.ts");

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
