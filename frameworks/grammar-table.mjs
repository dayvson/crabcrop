// The format and quality rule of the integrations that take them
// (Astro, Nuxt, SvelteKit), checked against the SDK's grammar
// (issue #482). Each integration is a single file copied into a
// project that installs @crabcrop/url, so it keeps its own short
// format list and quality check, with its own messages; this
// table keeps both from drifting from the grammar the SDK
// implements:
//
// - every still-image format the grammar takes is accepted, spelled
//   as the SDK's typed builder spells it (jpg as jpeg), and the
//   URL explains cleanly in the SDK;
// - the video formats (those the grammar takes no quality for) and
//   names outside the grammar throw: an <img> cannot show a video;
// - a quality from 1 to 100 spells as the SDK's builder spells it,
//   and 0, 101 and fractions throw.
//
// Test support only, like dimension-table.mjs.
import assert from "node:assert/strict";
import { explainUrl, filter, ops } from "@crabcrop/url";
import { IMAGE_FORMATS } from "../sdks/typescript/src/explain.ts";

const explains = (segment) =>
  explainUrl(`/unsafe/${segment}/https://cdn.example.com/a.jpg`).errors.length === 0;
const STILL = IMAGE_FORMATS.filter((name) => explains(`filters:quality(${name}:80)`));
const VIDEO = IMAGE_FORMATS.filter((name) => !STILL.includes(name));

/**
 * Run the table. `format(name)` and `quality(value)` build a URL
 * with only that option set.
 */
export function checkFormatTable({ format, quality }) {
  assert.ok(STILL.includes("jpeg") && VIDEO.includes("mp4"), `still ${STILL}, video ${VIDEO}`);
  for (const name of STILL) {
    const url = format(name);
    assert.ok(url.includes(`/${ops.filters(filter.format(name))}/`), `${name}: ${url}`);
    assert.deepEqual(explainUrl(url).errors, [], url);
  }
  for (const name of [...VIDEO, "svg", "bmp", "JPEG"]) {
    assert.throws(() => format(name), /unsupported format/, name);
  }
  for (const value of [1, 80, 100]) {
    const url = quality(value);
    assert.ok(url.includes(`/${ops.filters(filter.quality(value))}/`), `${value}: ${url}`);
    assert.deepEqual(explainUrl(url).errors, [], url);
  }
  for (const value of [0, 101, 1.5, Number.NaN]) {
    assert.throws(() => quality(value), /quality must be/, String(value));
  }
}
