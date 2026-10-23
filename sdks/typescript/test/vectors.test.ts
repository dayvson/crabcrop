import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { buildSrcSet, buildURL, sign } from "../src/index.ts";

// The canonical cross-language vectors: this package's whole
// signing suite is the shared file from the spec directory.
const vectors = JSON.parse(
  readFileSync(new URL("../../../spec/test-vectors.json", import.meta.url), "utf8"),
) as { description: string; secret: string; remainder: string; signature: string }[];

for (const v of vectors) {
  test(`vector: ${v.description}`, () => {
    assert.equal(sign(v.secret, v.remainder), v.signature);
  });
}

test("buildURL reproduces the spec worked example", () => {
  assert.equal(
    buildURL(
      "https://images.example.com/",
      "your-secret",
      "300x200",
      "https://example.com/photo.jpg",
    ),
    "https://images.example.com/3c1f6c180116661877bd4c85fd201a178087d74c1cc188cc584e819d901308c6/300x200/https://example.com/photo.jpg",
  );
});

test("buildURL with expiry and a null secret", () => {
  assert.equal(
    buildURL("https://img.example.com", null, "preset:thumb", "s3://media/a.jpg", {
      expires: 1893456000,
    }),
    "https://img.example.com/unsafe/exp:1893456000/preset:thumb/s3://media/a.jpg",
  );
});

test("buildSrcSet formats entries", () => {
  assert.equal(
    buildSrcSet("https://img.example.com", null, "https://example.com/a.jpg", [320, 640]),
    "https://img.example.com/unsafe/320x0/https://example.com/a.jpg 320w, " +
      "https://img.example.com/unsafe/640x0/https://example.com/a.jpg 640w",
  );
});

test("buildURL rejects inputs that can never verify (audit P-4, P-5)", () => {
  const base = "https://img.example.com";
  assert.throws(() => buildURL(base, "k", "300x200", "https://example.com/a.jpg?w=1"));
  assert.throws(() => buildURL(base, "k", "300x200", "https://example.com/a.jpg#f"));
  assert.throws(() => buildURL(base, "k", "300x200", "https://example.com/my photo.jpg"));
  assert.throws(() => buildURL(base, "k", "300x200", "https://example.com/café.jpg"));
  assert.throws(() => buildURL(base, "k", "300x200", "https://example.com/a.jpg", { expires: 1.5 }));
  assert.throws(() => buildURL(base, "k", "300x200", "https://example.com/a.jpg", { expires: -1 }));
});
