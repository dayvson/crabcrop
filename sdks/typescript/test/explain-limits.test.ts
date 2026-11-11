// Issue #594: a float argument costs time linear in its length,
// and explainUrl reads at most 64 KiB of input, so neither it
// nor verifyUrlSignature can be held up by a long URL.
import assert from "node:assert/strict";
import { test } from "node:test";
import { explainUrl, sign, verifyUrlSignature } from "../src/index.ts";

// Any 64-hex tag reaches the grammar; it need not be valid.
const TAG = "a".repeat(64);
const SRC = "https://e.com/a.jpg";
const CAP = 64 * 1024;
const CAP_ERROR = /longer than 64 KiB/;

// Every place the grammar reads a float, as an operation segment
// with `#` standing for the float.
const FLOAT_SITES = [
  "filters:blur(#)",
  "filters:sharpen(#)",
  "filters:contrast(#)",
  "filters:gamma(#)",
  "filters:exposure(#)",
  "filters:highlights_shadows(#,0)",
  "filters:vibrance(#)",
  "filters:saturation(#)",
  "filters:facepad(#)",
  "filters:dpr(#)",
  "focalpoint:#x0.5",
  "focalpoint:0.5x#",
  "focalpoint:0.5x0.5:#",
  "crop:100x100:objects(person):#",
];

function timed<T>(run: () => T): [T, number] {
  const started = performance.now();
  const out = run();
  return [out, performance.now() - started];
}

test("a long digit run in a float argument costs linear time", () => {
  // The old pattern could split one run of digits between
  // `\d+` and `\d*` at any point, so a run that ends in a
  // character no float can end in took quadratic time: about
  // 10 s for this run. It fits under the cap, so the grammar
  // reads all of it.
  const digits = "1".repeat(65_000);
  for (const site of FLOAT_SITES) {
    for (const tail of ["z", "e", ".z"]) {
      const url = `https://x/${TAG}/${site.replace("#", digits + tail)}/${SRC}`;
      assert.ok(url.length <= CAP, `${url.length} bytes`);
      const [out, explainMs] = timed(() => explainUrl(url));
      assert.ok(explainMs < 1000, `${site} with ${tail}: explainUrl took ${explainMs.toFixed(0)} ms`);
      assert.equal(out.errors.length, 1, `${site} with ${tail}`);
      assert.doesNotMatch(out.errors[0], CAP_ERROR);
      const [verdict, verifyMs] = timed(() => verifyUrlSignature("secret", url));
      assert.ok(verifyMs < 1000, `${site} with ${tail}: verifyUrlSignature took ${verifyMs.toFixed(0)} ms`);
      assert.equal(verdict, false);
    }
  }
});

test("the float grammar accepts and rejects what it did before", () => {
  // The pattern this replaced, kept as the reference on short
  // strings, where its backtracking costs nothing. sharpen
  // reads a float above 0 and contrast one in -100..100, so
  // the two together show the verdict on every string whose
  // value is above -100; below that, the sign is the only
  // difference, and the pattern reads the sign the same way.
  const OLD = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
  const accepts = (s: string): boolean => {
    const ok = (segment: string) => explainUrl(`https://x/${TAG}/${segment}/${SRC}`).errors.length === 0;
    return ok(`filters:sharpen(${s})`) || ok(`filters:contrast(${s})`);
  };
  const reference = (s: string): boolean => {
    const n = Number(s);
    return OLD.test(s) && Number.isFinite(n) && n >= -100;
  };
  const alphabet = "10.eE+-z";
  let strings = [""];
  let checked = 0;
  for (let length = 1; length <= 5; length++) {
    strings = strings.flatMap((s) => [...alphabet].map((c) => s + c));
    for (const s of strings) {
      assert.equal(accepts(s), reference(s), JSON.stringify(s));
      checked++;
    }
  }
  assert.equal(checked, 37_448);
  for (const s of ["1", "1.", ".5", "1.5", "1e5", "1.e5", ".5e-3", "+1", "-1", "007", "1E+2"]) {
    assert.ok(accepts(s), s);
  }
  for (const s of ["", ".", "e5", "1e", "1.5.5", "+-1", " 1", "1 ", "0x10", "inf", "NaN", "1e999", "1_0"]) {
    assert.ok(!accepts(s), JSON.stringify(s));
  }
});

test("explainUrl reads at most 64 KiB of input", () => {
  const secret = "secret";
  // A signed URL of exactly the cap explains and verifies.
  const head = `https://x/${"0".repeat(64)}/300x200/https://e.com/`;
  const fill = (bytes: number) => `${head}${"a".repeat(bytes - head.length - 4)}.jpg`;
  const signed = (url: string) => {
    const remainder = url.slice("https://x/".length + 65);
    return url.replace("0".repeat(64), sign(secret, remainder));
  };
  const atCap = signed(fill(CAP));
  assert.equal(atCap.length, CAP);
  const ok = explainUrl(atCap);
  assert.deepEqual(ok.errors, []);
  assert.equal(ok.signature?.kind, "hmac");
  assert.equal(verifyUrlSignature(secret, atCap), true);

  // One byte over: one error, nothing else read, and the
  // verifier answers null, as for any malformed URL, even
  // though the tag is right for its remainder.
  const over = signed(fill(CAP + 1));
  const out = explainUrl(over);
  assert.equal(out.errors.length, 1);
  assert.match(out.errors[0], CAP_ERROR);
  assert.deepEqual(
    { signature: out.signature, segments: out.segments, source: out.source, remainder: out.remainder },
    { signature: null, segments: [], source: "", remainder: "" },
  );
  assert.equal(verifyUrlSignature(secret, over), null);
  assert.equal(verifyUrlSignature({ k: secret }, over), null);

  // The cap counts UTF-8 bytes, as the server reads them, not
  // UTF-16 code units: one `é` is two bytes.
  const wide = `${fill(CAP - 1).slice(0, -4)}é.jpg`;
  assert.equal(wide.length, CAP);
  assert.match(explainUrl(wide).errors[0] ?? "", CAP_ERROR);
  const wideOk = `${fill(CAP - 1).slice(0, -5)}é.jpg`;
  assert.equal(new TextEncoder().encode(wideOk).length, CAP);
  assert.doesNotMatch(explainUrl(wideOk).errors.join("\n"), CAP_ERROR);

  // The issue's own input, 100,000 digits: refused at once.
  const long = `https://x/${TAG}/filters:blur(${"1".repeat(100_000)}z)/${SRC}`;
  const [refused, ms] = timed(() => explainUrl(long));
  assert.ok(ms < 1000, `${ms.toFixed(0)} ms`);
  assert.equal(refused.errors.length, 1);
  assert.match(refused.errors[0], CAP_ERROR);
  assert.equal(verifyUrlSignature(secret, long), null);
});
