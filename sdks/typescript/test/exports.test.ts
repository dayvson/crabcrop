// The public surface as a consumer imports it (issue #460). The
// package is imported by its own name, so the imports resolve
// through package.json `exports` to the built `dist/`, and
// `pretest` type-checks this file (`tsconfig.test.json`)
// against the shipped declarations: a type the index does not
// re-export fails the build there, not in a user's project.
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";
import type { FormatOptions, PerFormatQualityFormat } from "@crabcrop/url";
import { buildSrcSet, explainUrl, filter, MAX_DIMENSION, ops } from "@crabcrop/url";

const BASE = "https://images.example.com";
const SRC = "https://example.com/a.jpg";

test("MAX_DIMENSION is the server's per-axis cap, and every check uses it", () => {
  assert.equal(MAX_DIMENSION, 16383);
  const over = MAX_DIMENSION + 1;
  assert.equal(ops.size(MAX_DIMENSION, 0), `${MAX_DIMENSION}x0`);
  assert.throws(() => ops.size(over, 0), /a pixel count up to 16383/);
  assert.equal(filter.pixelate(MAX_DIMENSION), `pixelate(${MAX_DIMENSION})`);
  assert.throws(() => filter.pixelate(over), /from 1 to 16383/);
  assert.doesNotThrow(() => buildSrcSet(BASE, null, SRC, [MAX_DIMENSION]));
  assert.throws(() => buildSrcSet(BASE, null, SRC, [over]), /no larger than 16383, got 16384/);
  assert.equal(explainUrl(`/unsafe/${MAX_DIMENSION}x0/${SRC}`).segments[0].error, undefined);
  assert.match(explainUrl(`/unsafe/filters:pixelate(${over})/${SRC}`).segments[0].error ?? "", /pixelate/);
});

test("the format option types are importable by name", () => {
  const options: FormatOptions = { progressive: true, subsampling: 420 };
  const formats: PerFormatQualityFormat[] = ["avif", "webp"];
  assert.equal(filter.format("jpeg", options), "format(jpeg,progressive,subsampling:420)");
  const entries: Partial<Record<PerFormatQualityFormat, number>> = {};
  for (const format of formats) entries[format] = 60;
  assert.equal(filter.qualityPerFormat(entries), "quality(avif:60,webp:60)");
});

test("MAX_DIMENSION is declared once in the sources", () => {
  const src = new URL("../src/", import.meta.url);
  const literals = readdirSync(src, { recursive: true, encoding: "utf8" })
    .filter((file) => file.endsWith(".ts"))
    .flatMap((file) =>
      readFileSync(new URL(file, src), "utf8")
        .split("\n")
        .filter((line) => /\b16383\b/.test(line))
        .map((line) => `${file}: ${line.trim()}`),
    );
  assert.deepEqual(literals, ["limits.ts: export const MAX_DIMENSION = 16383;"]);
});
