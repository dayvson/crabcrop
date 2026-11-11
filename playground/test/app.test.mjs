/**
 * Unit test: drives the page's own app.mjs with a minimal DOM stub,
 * the field values index.html ships with, and a fetch that records
 * each request instead of sending it. Nothing leaves the process.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { installDom } from "./dom-stub.mjs";

const elements = installDom();

// The defaults a visitor gets: each <input> with an id and a value.
const page = readFileSync(new URL("../index.html", import.meta.url), "utf8");
for (const [input] of page.matchAll(/<input\b[^>]*>/g)) {
  const id = input.match(/\bid="([^"]*)"/)?.[1];
  const value = input.match(/\bvalue="([^"]*)"/)?.[1];
  if (id && value !== undefined) elements[id].value = value;
}
assert.ok(elements.source.value, "index.html prefills a source");

const fetched = [];
globalThis.fetch = async (url) => {
  fetched.push(String(url));
  return new Response(new Uint8Array(64));
};

async function until(condition) {
  for (let i = 0; i < 100 && !condition(); i++) {
    await new Promise((r) => setTimeout(r, 20));
  }
  // Let any request started alongside the condition land too.
  await new Promise((r) => setTimeout(r, 50));
}

await import("../app.mjs");
await until(() => elements.built.value);

test("on load the page builds and previews the URL but fetches nothing (issue #603)", () => {
  // The page says it contacts no source host until you edit a
  // field: the preview is the only request, and it goes to the
  // endpoint through the <img>, not through fetch.
  assert.ok(elements.built.value.endsWith(`/${elements.source.value}`), elements.built.value);
  assert.equal(elements.preview.src, elements.built.value);
  assert.deepEqual(fetched, []);
  assert.match(elements.sizes.textContent, /first edit/);
});

test("the first edit runs the size check, which fetches the source and the URL", async () => {
  await elements.operations.dispatch("input");
  await until(() => /Original/.test(elements.sizes.textContent));
  assert.deepEqual(fetched.sort(), [elements.built.value, elements.source.value].sort());
  assert.match(elements.sizes.textContent, /^Original 64 B -> processed 64 B \(0% smaller\)\.$/);
});
