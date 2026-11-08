// The dimension and base rule the five framework integrations
// share (issue #460), as one table every framework suite runs,
// so one mistake has one outcome in all five:
//
// - a fraction rounds to whole pixels;
// - a value that is not a size (0, negative, NaN, infinite) is
//   left out, so the URL serves the source's own size rather
//   than a flip (-5x0) or a URL the server rejects (NaNx0);
// - a value past the server's cap, and a missing base, throw
//   where the mistake is made instead of minting a dead URL.
//
// Test support only: the integrations are single files that
// users copy, and none of them imports this.
import assert from "node:assert/strict";
import { MAX_DIMENSION } from "@crabcrop/url";

export const BASE = "https://images.example.com";
export const SRC = "https://cdn.example.com/a.jpg";

/** [requested value, the pixel count it becomes (null: left out) or the error it throws]. */
const CASES = [
  [0, null],
  [-5, null],
  [Number.NaN, null],
  [Number.POSITIVE_INFINITY, null],
  [1.5, 2],
  [MAX_DIMENSION, MAX_DIMENSION],
  [MAX_DIMENSION + 1, "throws"],
];

/**
 * Run the table. `width(value)` and `height(value)` build an
 * unsigned URL under BASE for SRC with no preset and only that
 * one dimension set (`height` is omitted for an integration
 * with no height, Next.js); `emptyBase()` uses an empty base.
 */
export function checkDimensionTable({ width, height, emptyBase }) {
  const axes = [["width", width, (n) => `${n}x0`]];
  if (height) axes.push(["height", height, (n) => `0x${n}`]);
  for (const [axis, build, segment] of axes) {
    for (const [value, expected] of CASES) {
      const label = `${axis} ${value}`;
      if (expected === "throws") {
        const message = `${axis} ${value} exceeds the maximum dimension ${MAX_DIMENSION}`;
        assert.throws(() => build(value), { message: new RegExp(`^crabcrop \\w+: ${message}$`) }, label);
      } else {
        const operations = expected === null ? "" : `${segment(expected)}/`;
        assert.equal(build(value), `${BASE}/unsafe/${operations}${SRC}`, label);
      }
    }
  }
  assert.throws(emptyBase, { message: /^crabcrop \w+: .*base.* is required/i }, "empty base");
}
