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

import { explainUrl } from "../src/index.ts";

test("explainUrl verdicts match the server grammar (audit P-3 corpus)", () => {
  const SIG = "a".repeat(64);
  const SRC = "https://example.com/a.jpg";
  const cases: Array<[string, boolean, string]> = [
    ["exp:99999999999999999999999999/300x200/" + SRC, true, SRC],
    ["300x200/https://example.com/%2e%2e/a.jpg", false, "https://example.com/%2e%2e/a.jpg"],
    ["300x/" + SRC, false, SRC],
    ["x200/" + SRC, false, SRC],
    ["fit-in:300x/" + SRC, false, SRC],
    ["fit-in:-300x200/" + SRC, true, SRC],
    ["crop:100x100:sideways/" + SRC, true, SRC],
    ["crop:100x100:smart/" + SRC, false, SRC],
    ["filters:quality(999)/" + SRC, true, SRC],
    ["filters:quality(80):blur(2)/" + SRC, false, SRC],
    ["filters:blur(-1)/" + SRC, true, SRC],
    ["filters:dpr(7)/" + SRC, true, SRC],
    ["filters:dpr(1.5)/" + SRC, false, SRC],
    ["pad:100x100:blur/" + SRC, false, SRC],
    ["pad:100x100:ff00aa/" + SRC, false, SRC],
    ["pad:100x100:red/" + SRC, true, SRC],
    ["crop-rect:0x0:50x50/" + SRC, false, SRC],
    ["crop-rect:0x0:50/" + SRC, true, SRC],
    ["16384x10/" + SRC, false, "16384x10/" + SRC],
    ["exp:1893456000/300x200/" + SRC, false, SRC],
    ["filters:radius(12)/" + SRC, false, SRC],
    ["filters:radius(1,2,3,4):mask(circle)/" + SRC, false, SRC],
    ["filters:radius(1,2)/" + SRC, true, SRC],
    ["filters:border(2,00ff00)/" + SRC, false, SRC],
    ["filters:border(0,00ff00)/" + SRC, true, SRC],
    ["filters:mask(square)/" + SRC, true, SRC],
    ["filters:sepia:duotone(000000,ffffff):flip(h)/" + SRC, false, SRC],
    ["filters:duotone(000000)/" + SRC, true, SRC],
    ["filters:monochrome(ff8000):pixelate(8)/" + SRC, false, SRC],
    ["filters:pixelate(0)/" + SRC, true, SRC],
    ["filters:flip(x)/" + SRC, true, SRC],
    ["filters:blur_region(1,2,30,40):pixelate_region(0,0,5,5)/" + SRC, false, SRC],
    ["filters:blur_region(1,2,0,40)/" + SRC, true, SRC],
    ["filters:pixelate_region(0,0,10,16384)/" + SRC, true, SRC],
    ["filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,80,multiply)/" + SRC, false, SRC],
    ["filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,screen,25)/" + SRC, false, SRC],
    ["filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,darken)/" + SRC, true, SRC],
    ["filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,101,normal)/" + SRC, true, SRC],
  ];
  for (const [remainder, expectError, expectSource] of cases) {
    const out = explainUrl(`https://img.example.com/${SIG}/${remainder}`);
    assert.equal(out.source, expectSource, remainder);
    assert.equal(out.errors.length > 0, expectError, `${remainder}: ${JSON.stringify(out.errors)}`);
  }
});
