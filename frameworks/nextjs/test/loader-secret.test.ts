import "./register.mjs";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { test } from "node:test";

process.env.NEXT_PUBLIC_CRABCROP_BASE = "https://images.example.com";
process.env.CRABCROP_SECRET = "test-secret";
delete process.env.NEXT_PUBLIC_CRABCROP_PRESET;

const { default: loader } = await import("../crabcrop-loader.ts");

// Expected signatures are computed here with node:crypto
// directly, not through the SDK, so the loader's remainder
// construction is what is actually under test.
const sig = (remainder: string) =>
  createHmac("sha256", "test-secret").update(remainder, "utf8").digest("hex");

test("server-side calls sign (the spec vector secret agrees)", () => {
  const remainder = "640x0/filters:quality(75)/https://cdn.example.com/a.jpg";
  assert.equal(
    loader({ src: "https://cdn.example.com/a.jpg", width: 640, quality: 75 }),
    `https://images.example.com/${sig(remainder)}/${remainder}`,
  );
});

test("a source under the reserved segment is refused, never signed", () => {
  // A width the loader drops leaves no operations, so src is
  // the whole remainder; every derived subkey is the signature
  // of a label such as crabcrop/upload-key.
  for (const src of ["crabcrop/upload-key", "crabcrop/seal-key", "crabcrop"]) {
    assert.throws(() => loader({ src, width: 0 }), /reserved segment "crabcrop"/);
  }
  const named = "https://cdn.example.com/crabcrop/upload-key.jpg";
  assert.equal(loader({ src: named, width: 0 }), `https://images.example.com/${sig(named)}/${named}`);
});
