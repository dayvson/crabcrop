// Issue #537: trimLikeServer strips what Rust's str::trim strips,
// the Unicode White_Space set, and takes time linear in its input.
import assert from "node:assert/strict";
import { test } from "node:test";
import { trimLikeServer } from "../src/explain.ts";

test("trimLikeServer strips the White_Space set and nothing else", () => {
  // The engine's own White_Space table is the reference.
  const whiteSpace = /^\p{White_Space}$/u;
  for (let unit = 0; unit <= 0xffff; unit++) {
    const c = String.fromCharCode(unit);
    const want = whiteSpace.test(c) ? "x" : `${c}x${c}`;
    assert.equal(trimLikeServer(`${c}x${c}`), want, `U+${unit.toString(16).padStart(4, "0")}`);
  }
});

test("trimLikeServer edge cases", () => {
  const cases: [string, string][] = [
    ["", ""],
    [" ", ""],
    [" \t\n\v\f\r\u0085        　", ""],
    ["person", "person"],
    [" traffic light ", "traffic light"],
    // U+FEFF is not white space to the server: it stays.
    ["﻿person﻿", "﻿person﻿"],
    [" ﻿ person ﻿ ", "﻿ person ﻿"],
    // U+0085 and U+3000 are: they go.
    ["\u0085person\u0085", "person"],
    ["　person　", "person"],
    // A character outside the Basic Multilingual Plane stays whole.
    [" \u{1f600} ", "\u{1f600}"],
  ];
  for (const [input, want] of cases) {
    assert.equal(trimLikeServer(input), want, JSON.stringify(input));
  }
});

test("trimLikeServer is linear in a long run of white space", () => {
  // The regular expression this replaced took about 2 seconds on
  // the run between two `x`s: at each position in the run, its
  // trailing alternative scanned to the run's end and failed.
  const run = " ".repeat(50_000);
  const cases: [string, string][] = [
    [`${run}x`, "x"],
    [`x${run}`, "x"],
    [`x${run}x`, `x${run}x`],
    [`x${run}x${run}`, `x${run}x`],
    [run, ""],
  ];
  for (const [input, want] of cases) {
    const started = performance.now();
    const got = trimLikeServer(input);
    const elapsed = performance.now() - started;
    assert.equal(got, want);
    assert.ok(elapsed < 250, `${elapsed.toFixed(0)} ms for ${input.length} characters`);
  }
});
