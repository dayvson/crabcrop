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
  if (v.refuses !== undefined) {
    // Issue #472: the playground refuses what every builder
    // refuses, with the SDK's message, signed or not.
    test(`refusal vector: ${v.description}`, async () => {
      const { input, character } = v.refuses;
      const base = "https://img.example.com";
      let expected;
      try {
        sdk.buildURL(base, "s3cr3t", v.operations, v.source);
      } catch (e) {
        expected = e.message;
      }
      assert.ok(expected?.startsWith(`${input} contains`), expected);
      assert.ok(expected.includes(JSON.stringify(character)), expected);
      for (const secret of ["s3cr3t", null]) {
        await assert.rejects(
          buildPlaygroundUrl({ base, secret, operations: v.operations, source: v.source }),
          { message: expected },
        );
      }
    });
    continue;
  }
  if (v.signature === undefined) {
    // Sealed entries carry a token, not a signature; the
    // playground does not seal (issue #255 lives in the
    // SDKs and the CLI).
    continue;
  }
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

test("a sealed token explains as sealed, not malformed (issue #388)", () => {
  // Minted by the SDK (the playground itself does not
  // seal), with a fixed nonce so the shape is stable.
  const token = sdk.sealWithNonce(
    "explain-sealed-secret",
    "300x200/https://example.com/a.jpg",
    new Uint8Array(12).fill(7),
  );
  const out = explainUrl(`https://images.example.com/${token}`);
  assert.equal(out.signature.kind, "sealed");
  assert.equal(out.signature.value, token);
  assert.deepEqual(out.errors, []);
  assert.deepEqual(out.segments, []);
  assert.equal(out.source, "");

  const keyed = sdk.sealWithNonce(
    "explain-sealed-secret",
    "300x200/https://example.com/a.jpg",
    new Uint8Array(12).fill(9),
    "k2",
  );
  const keyedOut = explainUrl(`https://images.example.com/${keyed}`);
  assert.equal(keyedOut.signature.kind, "sealed");
  assert.equal(keyedOut.signature.keyId, "k2");

  const imitation = explainUrl("https://x/s2.AAAA/300x200/https://e.com/a.jpg");
  assert.equal(imitation.signature.kind, "invalid");
  assert.match(imitation.signature.detail, /uniform 403 refusal/);
});

test("studio filters explain, including inside a chain", () => {
  const product = explainUrl(
    `https://x/${SIG}/filters:studio(product,bg:f5f0e8,shadow:strong,pad:25)/https://e.com/a.jpg`,
  );
  assert.equal(product.errors.length, 0);

  const chained = explainUrl(`https://x/${SIG}/filters:studio(scene,pad:25):quality(80)/https://e.com/a.jpg`);
  assert.equal(chained.errors.length, 0);

  const sceneBg = explainUrl(`https://x/${SIG}/filters:studio(scene,bg:white)/https://e.com/a.jpg`);
  assert.ok(sceneBg.errors.some((e) => e.includes("the server rejects this filter: studio")));

  const padTooBig = explainUrl(`https://x/${SIG}/filters:studio(product,pad:41)/https://e.com/a.jpg`);
  assert.ok(padTooBig.errors.some((e) => e.includes("the server rejects this filter: studio")));
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
  // The comparison scans the whole tag (constant time): a
  // tag differing only in its final character still fails,
  // as does one differing only in its first.
  const tag = url.split("/")[3];
  const lastFlipped = tag.slice(0, -1) + (tag.endsWith("0") ? "1" : "0");
  assert.equal(await verifyUrlSignature("s3cr3t", url.replace(tag, lastFlipped)), false);
  const firstFlipped = (tag.startsWith("0") ? "1" : "0") + tag.slice(1);
  assert.equal(await verifyUrlSignature("s3cr3t", url.replace(tag, firstFlipped)), false);
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

// The explainer verdict corpus (audit P-3 / AP-23) is a
// shared spec file, not a copy in this suite: the TypeScript
// SDK suite and the server crate's differential tests run
// the same entries (issue #353), so a verdict can only
// change in one place and every surface re-checks it.
const verdictCorpus = JSON.parse(
  readFileSync(new URL("../../spec/explain-verdicts.json", import.meta.url), "utf8"),
);

test("explainer verdicts match the server grammar (audit P-3 corpus)", () => {
  const SIG = "a".repeat(64);
  for (const { remainder, error, source } of verdictCorpus.cases) {
    const out = explainUrl(`https://img.example.com/${SIG}/${remainder}`);
    assert.equal(out.source, source, remainder);
    assert.equal(out.errors.length > 0, error, `${remainder}: ${JSON.stringify(out.errors)}`);
  }
});

// The generated-combination corpus (issue #353): generated
// valid and invalid remainders whose verdicts the generator
// pins from the grammar. The server crate checks the same
// file against the Rust parser, including the canonical
// forms and error codes this surface does not produce.
const operationVectors = JSON.parse(
  readFileSync(new URL("../../spec/operation-vectors.json", import.meta.url), "utf8"),
);

test("explainer verdicts match the generated combination corpus", () => {
  for (const { remainder, ok, source } of operationVectors.cases) {
    const out = explainUrl(
      `https://img.example.com/${operationVectors.signature}/${remainder}`,
      1700000000,
    );
    assert.equal(out.errors.length === 0, ok, `${remainder}: ${JSON.stringify(out.errors)}`);
    assert.equal(out.source, source, remainder);
  }
});

test("encoder options and per-format quality explain (issue #319)", () => {
  const good = explainUrl(
    `https://x/${SIG}/filters:format(jpeg,progressive,subsampling:444):quality(80,webp:75,jpeg:90)/https://e.com/a.jpg`,
  );
  assert.equal(good.errors.length, 0);
  const webp = explainUrl(
    `https://x/${SIG}/filters:format(webp,method:6,sns:20)/https://e.com/a.jpg`,
  );
  assert.equal(webp.errors.length, 0);
  const avif = explainUrl(
    `https://x/${SIG}/filters:format(avif,subsampling:420)/https://e.com/a.jpg`,
  );
  assert.ok(avif.errors.length > 0);
  const dup = explainUrl(
    `https://x/${SIG}/filters:quality(jpeg:80,jpeg:90)/https://e.com/a.jpg`,
  );
  assert.ok(dup.errors.length > 0);
  const video = explainUrl(
    `https://x/${SIG}/filters:quality(mp4:80)/https://e.com/a.jpg`,
  );
  assert.ok(video.errors.length > 0);
});
