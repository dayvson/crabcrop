// Issue #712: the explainer reads every number as the server's
// Rust types read it, so the two give one verdict on every
// spelling. The cross-language cases live in
// spec/explain-verdicts.json; these pin the reader itself.
import assert from "node:assert/strict";
import { test } from "node:test";
import { explainUrl } from "../src/index.ts";
import { isClassName, readDecimal } from "../src/explain.ts";

const TAG = "a".repeat(64);
const SRC = "https://example.com/a.jpg";
const errors = (ops: string): string[] => explainUrl(`https://x/${TAG}/${ops}/${SRC}`).errors;

test("readDecimal rounds a decimal to the nearest 32-bit float, exactly", () => {
  // Each decimal sits next to the midpoint between two 32-bit
  // floats. Rounding to a double first lands on the midpoint,
  // and the tie then goes to the even float, the wrong one:
  // Math.fround(Number(s)) misreads all of these. The expected
  // values are the exact rounding, from rational arithmetic.
  const misread: Array<[string, number]> = [
    ["-967.109100341796874999999999999999990000000000000000000000000000", -967.1090698242188],
    ["-134.516242980957031250000000100000000000000000000000000000000000", -134.51625061035156],
    ["566.857391357421874999999999999999900000000000000000000000000000", 566.8573608398438],
    ["966.323883056640625000000000000001000000000000000000000000000000", 966.3239135742188],
    ["701.345001220703125000000000100000000000000000000000000000000000", 701.3450317382812],
    // Just above the midpoint between 512 and the next float:
    // the server reads 512.00006, past the blur bound.
    ["512.0000305175781250000000000000001", 512.00006103515625],
    // Just below the midpoint past the largest float: the
    // server reads f32::MAX, a sharpen amount, not infinity.
    ["3.4028235677973366e38", 3.4028234663852886e38],
  ];
  for (const [text, want] of misread) {
    assert.ok(!Object.is(Math.fround(Number(text)), want), `${text} needs no exact reading`);
    assert.equal(readDecimal(text, 32), want, text);
  }
  // The other side of the same midpoints.
  assert.equal(readDecimal("512.0000305175781249999999999999999", 32), 512);
  assert.equal(readDecimal("3.4028235677973367e38", 32), Infinity);
});

test("readDecimal reads Rust's float syntax and nothing else", () => {
  const read: Array<[string, number]> = [
    ["2", 2],
    ["+2", 2],
    ["2.", 2],
    [".5", 0.5],
    ["+.5e1", 5],
    ["2e0", 2],
    ["1E+2", 100],
    ["007", 7],
    ["-0", -0],
    ["-1e-50", -0],
    ["1e-46", 0],
    ["1e-45", 1.401298464324817e-45],
    ["1e39", Infinity],
    ["-1e39", -Infinity],
    ["1e999999999999999999999", Infinity],
    ["1e-999999999999999999999", 0],
  ];
  for (const [text, want] of read) {
    assert.ok(Object.is(readDecimal(text, 32), want), `${text}: ${readDecimal(text, 32)}`);
  }
  for (const text of ["", ".", "+", "e5", "1e", "1e+", "1.5.5", "+-1", " 1", "1 ", "0x10", "1_0", "inf", "-infinity", "NaN"]) {
    assert.equal(readDecimal(text, 32), null, JSON.stringify(text));
  }
  // The 64-bit reading is the double JavaScript's own parser
  // gives, which is correctly rounded.
  for (const text of ["0.1", "86400.000000000000000001", "1024.0000000000001", "2.2250738585072011e-308", "1e309"]) {
    assert.equal(readDecimal(text, 64), Number(text), text);
  }
});

test("readDecimal reads a decimal as long as the URL in linear time", () => {
  // The digits after the thousandth only decide a sticky digit,
  // so the BigInt work is bounded whatever the length.
  const long = [`1.${"1".repeat(64_000)}`, `0.${"0".repeat(32_000)}1${"0".repeat(32_000)}e32010`, `${"9".repeat(64_000)}e-63990`];
  for (const text of long) {
    const started = performance.now();
    const value = readDecimal(text, 32);
    const ms = performance.now() - started;
    assert.ok(ms < 1000, `${text.length} characters took ${ms.toFixed(0)} ms`);
    assert.ok(value !== null && Number.isFinite(value), `${value}`);
  }
  assert.equal(readDecimal(`1.${"1".repeat(64_000)}`, 32), Math.fround(1.1111111111111112));
});

test("the explainer holds a number's bounds to its rounded value", () => {
  for (const ok of ["filters:blur(512.00001)", "filters:dpr(1.50000001)", "focalpoint:1.00000001x0.5", "filters:rotate(36000.001)", "filters:sharpen(3.4028235e38)", "filters:format(webp,method:+6)"]) {
    assert.deepEqual(errors(ok), [], ok);
  }
  for (const bad of ["filters:blur(512.0001)", "filters:blur(1e-40)", "filters:gamma(1e-46)", "filters:sharpen(1e39)", "filters:trim(-0)", "crop-rect:-0x0:10x10"]) {
    assert.equal(errors(bad).length, 1, bad);
  }
});

test("a class name is letters or digits as Rust's is_alphanumeric reads them", () => {
  // The Unicode Alphabetic property holds more than the letter
  // categories: Devanagari vowel signs, U+0345, circled letters.
  for (const name of ["हिंदी", "aͅ", "Ⓐ", "Ⅰ", "traffic light", "e-bike_2.0/x"]) {
    assert.ok(isClassName(name), name);
  }
  for (const name of ["✓", "a‍b", "a*b", "", "é".repeat(33)]) {
    assert.ok(!isClassName(name), name);
  }
});

test("the explainer refuses what the server refuses before the parse and after it", () => {
  const reserved = (remainder: string) => explainUrl(`https://x/${TAG}/${remainder}`).errors;
  assert.match(reserved("crabcrop/purge-key")[0], /reserved segment crabcrop.*400 \(reserved_segment\)/);
  assert.match(reserved("purge/300x200/a.jpg")[0], /purge or upload segment.*400 \(reserved_segment\)/);
  assert.deepEqual(reserved("crabcrop-assets/a.jpg"), []);
  const inverts = (n: number) => `filters:${Array(n).fill("invert").join(":")}`;
  assert.deepEqual(errors(inverts(100)), []);
  assert.deepEqual(errors(inverts(101)), [
    "the request has 101 operations, over the 100 operation limit: the server answers 400 (unsupported_operation)",
  ]);
  // A preset adds at least one operation the explainer cannot see.
  assert.match(errors(`preset:thumb/${inverts(100)}`)[0], /at least 101 operations/);
  const watermarks = (n: number) =>
    `filters:${Array.from({ length: n }, (_, i) => `watermark(w${i}.png,0,0,50)`).join(":")}`;
  assert.deepEqual(errors(watermarks(16)), []);
  assert.deepEqual(errors(`${watermarks(16)}:watermark(w%30.png,9,9,50)`), []);
  assert.deepEqual(errors(watermarks(17)), [
    "the request names 17 distinct watermark and overlay URLs, over the 16 URL limit: the server answers 400 (unsupported_operation)",
  ]);
});
