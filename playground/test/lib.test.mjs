import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { buildPlaygroundUrl, explainUrl, sign, verifyUrlSignature } from "../lib.mjs";

// The SDK itself, imported to prove byte-for-byte parity of
// the playground's assembly with the reference builder.
const sdk = await import("../../sdks/typescript/src/index.ts");

const vectors = JSON.parse(
  readFileSync(new URL("../../spec/test-vectors.json", import.meta.url), "utf8"),
);

for (const v of vectors) {
  test(`sign vector: ${v.description}`, async () => {
    assert.equal(await sign(v.secret, v.remainder), v.signature);
  });
}

test("URL assembly matches the SDK byte for byte", async () => {
  const cases = [
    { base: "https://images.example.com/", secret: "your-secret", operations: "300x200", source: "https://example.com/photo.jpg" },
    { base: "https://img.example.com", secret: "", operations: "preset:thumb", source: "s3://media/a.jpg", expires: 1893456000 },
    { base: "https://img.example.com", secret: "k", operations: "", source: "https://example.com/a.jpg" },
    { base: "https://img.example.com", secret: "k", operations: "crop:10x10:smart/filters:grayscale", source: "gs://b/k.png", expires: 42 },
  ];
  for (const c of cases) {
    const expected = sdk.buildURL(c.base, c.secret === "" ? null : c.secret, c.operations, c.source, {
      expires: c.expires,
    });
    const actual = await buildPlaygroundUrl(c);
    assert.equal(actual, expected);
  }
});

const SIG = "a".repeat(64);

test("explains a full URL", () => {
  const out = explainUrl(
    `https://images.example.com/${SIG}/exp:1893456000/preset:og-card/crop:1200x630:smart/filters:format(webp):quality(80)/https://example.com/photo.jpg`,
    1700000000,
  );
  assert.equal(out.signature.kind, "hmac");
  assert.deepEqual(
    out.segments.map((s) => s.kind),
    ["expiry", "preset", "operation", "operation"],
  );
  assert.equal(out.source, "https://example.com/photo.jpg");
  assert.equal(out.errors.length, 0);
  assert.match(out.segments[0].detail, /2030-01-01/);
});

test("expired expiries and malformed segments are flagged", () => {
  const expired = explainUrl(`https://x/${SIG}/exp:1/300x200/https://e.com/a.jpg`, 1700000000);
  assert.equal(expired.segments[0].expired, true);
  assert.match(expired.segments[0].detail, /EXPIRED/);

  const badCrop = explainUrl(`https://x/${SIG}/crop:banana/https://e.com/a.jpg`);
  assert.ok(badCrop.errors.some((e) => e.includes("malformed crop")));

  const badFilter = explainUrl(`https://x/${SIG}/filters:sparkles/https://e.com/a.jpg`);
  assert.ok(badFilter.errors.some((e) => e.includes("unknown filter: sparkles")));

  const badSig = explainUrl("https://x/zzz/300x200/https://e.com/a.jpg");
  assert.equal(badSig.signature.kind, "invalid");
  assert.ok(badSig.errors.length > 0);
});

test("preset-looking text inside the source stays source", () => {
  const out = explainUrl(`https://x/${SIG}/300x200/example.com/preset:thumb/img.jpg`);
  assert.equal(out.source, "example.com/preset:thumb/img.jpg");
  assert.equal(out.segments.length, 1);
});

test("signature verification round-trips locally", async () => {
  const url = await buildPlaygroundUrl({
    base: "https://images.example.com",
    secret: "s3cr3t",
    operations: "300x200",
    source: "https://example.com/photo.jpg",
  });
  assert.equal(await verifyUrlSignature("s3cr3t", url), true);
  assert.equal(await verifyUrlSignature("wrong", url), false);
  assert.equal(await verifyUrlSignature("s3cr3t", url.replace("300x200", "400x200")), false);
  const unsigned = await buildPlaygroundUrl({
    base: "https://images.example.com",
    secret: "",
    operations: "300x200",
    source: "https://example.com/photo.jpg",
  });
  assert.equal(await verifyUrlSignature("s3cr3t", unsigned), null);
});
