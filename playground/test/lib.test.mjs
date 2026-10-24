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
    { base: "https://img.example.com", secret: null, operations: "preset:thumb", source: "s3://media/a.jpg", expires: 1893456000 },
    { base: "https://img.example.com", secret: "k", operations: "", source: "https://example.com/a.jpg" },
    { base: "https://img.example.com", secret: "k", operations: "crop:10x10:smart/filters:grayscale", source: "gs://b/k.png", expires: 42 },
  ];
  for (const c of cases) {
    const expected = sdk.buildURL(c.base, c.secret, c.operations, c.source, {
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
    secret: null,
    operations: "300x200",
    source: "https://example.com/photo.jpg",
  });
  assert.equal(await verifyUrlSignature("s3cr3t", unsigned), null);
});

test("buildPlaygroundUrl rejects inputs that can never verify (audit P-4, P-5)", async () => {
  const base = { base: "https://img.example.com", secret: "k", operations: "300x200" };
  await assert.rejects(() => buildPlaygroundUrl({ ...base, source: "https://example.com/a.jpg?w=1" }));
  await assert.rejects(() => buildPlaygroundUrl({ ...base, source: "https://example.com/my photo.jpg" }));
  await assert.rejects(() =>
    buildPlaygroundUrl({ ...base, source: "https://example.com/a.jpg", expires: 1893456000.5 }),
  );
});

test("empty-string secret signs; only null means unsafe (audit P-6)", async () => {
  const signed = await buildPlaygroundUrl({
    base: "https://img.example.com",
    secret: "",
    operations: "300x200",
    source: "https://example.com/a.jpg",
  });
  const segment = signed.split("/")[3];
  assert.match(segment, /^[0-9a-f]{64}$/);
  const unsigned = await buildPlaygroundUrl({
    base: "https://img.example.com",
    secret: null,
    operations: "300x200",
    source: "https://example.com/a.jpg",
  });
  assert.ok(unsigned.includes("/unsafe/"));
});

test("explainer verdicts match the server grammar (audit P-3 corpus)", () => {
  const SRC = "https://example.com/a.jpg";
  // [remainder, expectError, expectSource]
  const cases = [
    ["exp:99999999999999999999999999/300x200/" + SRC, true, SRC], // u64 overflow: a clean error, not a crash
    ["300x200/https://example.com/%2e%2e/a.jpg", false, "https://example.com/%2e%2e/a.jpg"], // dot segments stay literal
    ["300x/" + SRC, false, SRC], // empty height is proportional
    ["x200/" + SRC, false, SRC], // empty width is proportional
    ["fit-in:300x/" + SRC, false, SRC],
    ["fit-in:-300x200/" + SRC, true, SRC], // flips are invalid in fit-in
    ["crop:100x100:sideways/" + SRC, true, SRC], // align is an enumerated set
    ["crop:100x100:smart/" + SRC, false, SRC],
    ["crop:300x0/" + SRC, true, SRC], // crop is an exact box: pixels on both axes
    ["crop:-300x200/" + SRC, true, SRC], // flips are invalid in crop
    ["crop:origx100/" + SRC, true, SRC], // orig is invalid in crop
    ["pad:300xorig/" + SRC, true, SRC], // pad is an exact box too
    ["pad:-200x100/" + SRC, true, SRC], // flips are invalid in pad
    ["filters:quality(999)/" + SRC, true, SRC], // quality caps at 100
    ["filters:quality(80):blur(2)/" + SRC, false, SRC],
    ["filters:blur(-1)/" + SRC, true, SRC], // sigma must be positive
    ["filters:dpr(7)/" + SRC, true, SRC], // dpr is an enumerated set
    ["filters:dpr(1.5)/" + SRC, false, SRC],
    ["pad:100x100:blur/" + SRC, false, SRC],
    ["pad:100x100:ff00aa/" + SRC, false, SRC],
    ["pad:100x100:red/" + SRC, true, SRC],
    ["crop-rect:0x0:50x50/" + SRC, false, SRC],
    ["crop-rect:0x0:50/" + SRC, true, SRC],
    ["16384x10/" + SRC, false, "16384x10/" + SRC], // oversize bare size is not an op: the source starts there
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
    ["filters:blurclass(person)/" + SRC, false, SRC],
    ["filters:pixelateclass(traffic%20light)/" + SRC, false, SRC],
    ["filters:blurclass()/" + SRC, true, SRC],
    ["filters:blurclass(per%24on)/" + SRC, true, SRC],
    ["filters:pixelateclass/" + SRC, true, SRC],
    ["filters:keep_profile/" + SRC, false, SRC],
    // AP-23 parity cases: each verdict below is the Rust
    // parser's, verified against src/url.rs.
    ["crop-rect:50x50:0x0/" + SRC, true, SRC],
    ["crop-rect:50x50:51x51/" + SRC, false, SRC],
    ["filters:blur(0x10)/" + SRC, true, SRC],
    ["filters:contrast(0x2)/" + SRC, true, SRC],
    ["filters:blur( 1.5)/" + SRC, true, SRC],
    ["filters:blur(1e2)/" + SRC, false, SRC],
    ["filters:blur(.5)/" + SRC, false, SRC],
    ["filters:overlay(a,b.png,0,0,80,normal)/" + SRC, true, SRC],
    ["filters:overlay(a,b.png,0,0,80,normal,25)/" + SRC, false, SRC],
    ["300x200//" + SRC, true, SRC],
    ["filters:blurclass(" + "é".repeat(40) + ")/" + SRC, true, SRC],
    ["filters:blurclass(" + "é".repeat(32) + ")/" + SRC, false, SRC],
    ["filters:brighten(+5)/" + SRC, false, SRC],
    ["filters:quality(+80)/" + SRC, false, SRC],
    ["filters:text(Hello,+12,ff8800,center)/" + SRC, false, SRC],
    ["+300x200/" + SRC, false, SRC],
    ["filters:radius(+12)/" + SRC, false, SRC],
    ["filters:pixelate(+8)/" + SRC, false, SRC],
  ];
  for (const [remainder, expectError, expectSource] of cases) {
    const out = explainUrl(`https://img.example.com/${SIG}/${remainder}`);
    assert.equal(out.source, expectSource, remainder);
    assert.equal(out.errors.length > 0, expectError, `${remainder}: ${JSON.stringify(out.errors)}`);
  }
});
