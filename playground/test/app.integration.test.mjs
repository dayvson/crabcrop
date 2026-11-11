/**
 * Integration test: drives the page's own app.mjs with a
 * minimal DOM stub, against a locally running crabcrop.
 *
 * Preconditions (this is not part of the hermetic suite):
 *   - crabcrop on 127.0.0.1:3321 with CRABCROP_SECRET=playground-secret
 *   - any image source on 127.0.0.1:8909 (the repo's test-fixtures/ serve this)
 * Run with: npm run test:integration
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { installDom } from "./dom-stub.mjs";

const elements = installDom();

elements.base.value = "http://127.0.0.1:3321";
elements.secret.value = "playground-secret";
elements.source.value = "http://127.0.0.1:8909/src00.jpg";
elements.operations.value = "300x200";
elements.expiry.value = "";

const health = await fetch("http://127.0.0.1:3321/health").catch(() => null);
assert.ok(health?.ok, "precondition: crabcrop must be running on 127.0.0.1:3321");

await import("../app.mjs");

// The build is async; wait for it. The size check waits for the
// first edit (issue #603), so make one, then wait for the sizes.
for (let i = 0; i < 100 && !elements.built.value; i++) {
  await new Promise((r) => setTimeout(r, 100));
}
await elements.operations.dispatch("input");
for (let i = 0; i < 100 && !/Original|unavailable/.test(elements.sizes.textContent); i++) {
  await new Promise((r) => setTimeout(r, 100));
}

test("the builder produces a signed URL the server accepts", () => {
  const built = elements.built.value;
  assert.match(
    built,
    /^http:\/\/127\.0\.0\.1:3321\/[0-9a-f]{64}\/300x200\/http:\/\/127\.0\.0\.1:8909\/src00\.jpg$/,
  );
  assert.equal(elements.preview.src, built);
  assert.match(elements.sizes.textContent, /Original .+ -> processed .+ \(\d+% smaller\)/);
});

async function verdict() {
  // The verify path is async (WebCrypto); poll for the line.
  for (let i = 0; i < 50; i++) {
    const text = elements["verify-line"].innerHTML || elements["verify-line"].textContent;
    if (text) return text;
    await new Promise((r) => setTimeout(r, 50));
  }
  return "";
}

test("the explainer verifies the built URL, then catches a tamper", async () => {
  elements["explain-input"].value = elements.built.value;
  await elements["explain-built"].dispatch("click");
  assert.ok(elements.explanation.children.length >= 3, "signature, operation, and source rows");
  assert.match(await verdict(), /Signature verifies/);

  elements["verify-line"].innerHTML = "";
  elements["verify-line"].textContent = "";
  elements["explain-input"].value = elements.built.value.replace("300x200", "400x200");
  await elements["explain-input"].dispatch("input");
  assert.match(await verdict(), /does NOT verify/);
});

test("a builder error is surfaced and clears the stale build", async () => {
  // A source carrying a query string is routine to paste
  // and the builder throws on it by design: the page must
  // say so and drop the previous URL and preview, not
  // silently keep showing them.
  assert.ok(elements.built.value, "a valid build exists before the error");
  elements.source.value = "http://127.0.0.1:8909/src00.jpg?w=300";
  await elements.source.dispatch("input");
  for (let i = 0; i < 50 && elements.built.value; i++) {
    await new Promise((r) => setTimeout(r, 50));
  }
  assert.equal(elements.built.value, "");
  assert.equal(elements.preview.src, undefined);
  assert.match(elements.sizes.textContent, /Cannot build a URL from these inputs/);
});
