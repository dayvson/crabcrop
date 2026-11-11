import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
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
  if (v.reserved === true) {
    // A remainder under the reserved segment: the playground
    // refuses to sign it or build it, with the SDK's message,
    // signed or not.
    test(`reserved vector: ${v.description}`, async () => {
      const base = "https://img.example.com";
      let expected;
      try {
        sdk.sign("s3cr3t", v.remainder);
      } catch (e) {
        expected = e.message;
      }
      assert.match(expected ?? "", /reserved segment "crabcrop"/);
      await assert.rejects(sign("s3cr3t", v.remainder), { message: expected });
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

// Issue #482: the page's explainer is the SDK's own build
// (../sdk/, written by scripts/sync_playground_sdk.py, whose
// --check pins the bytes to a fresh build). This pins the
// behavior as well: on every remainder of both corpora, the
// page answers exactly what the SDK's source answers.
test("the explainer is the SDK's: every corpus answer matches it exactly", () => {
  const inputs = [
    ...verdictCorpus.cases.map(({ remainder }) => `https://img.example.com/${SIG}/${remainder}`),
    ...operationVectors.cases.map(
      ({ remainder }) => `https://img.example.com/${operationVectors.signature}/${remainder}`,
    ),
  ];
  assert.ok(inputs.length > 900, `only ${inputs.length} corpus remainders`);
  for (const input of inputs) {
    assert.deepEqual(explainUrl(input, 1700000000), sdk.explainUrl(input, 1700000000), input);
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

test("the playground never signs a key label, and a subkey never verifies", async () => {
  // Every derived subkey is HMAC(secret, "crabcrop/<x>-key"),
  // the same function as the playground's signer, so the
  // signer refuses every label.
  const secret = "label-test-secret-0123456789";
  const base = "https://img.example.com";
  for (const label of ["crabcrop/purge-key", "crabcrop/upload-key", "crabcrop/seal-key"]) {
    await assert.rejects(
      buildPlaygroundUrl({ base, secret, operations: "", source: label }),
      /reserved segment "crabcrop"/,
    );
    const subkey = createHmac("sha256", secret).update(label).digest("hex");
    assert.equal(await verifyUrlSignature(secret, `${base}/${subkey}/${label}`), false);
  }
});

// Issue #594: the page explains on every edit, so a long digit
// run in a float argument must cost linear time, and input over
// 64 KiB is refused before the grammar reads it.
test("a long float argument explains in linear time, and input over 64 KiB is refused", async () => {
  const url = (digits) => `https://x/${SIG}/filters:blur(${"1".repeat(digits)}z)/https://e.com/a.jpg`;
  const cases = [
    // Under the cap: the grammar reads it all (about 10 s
    // with the old float pattern).
    [65_000, /^the server rejects this filter: blur\(1+z\)$/, false],
    // Over the cap: refused at once, and nothing to verify.
    [100_000, /longer than 64 KiB/, null],
  ];
  for (const [digits, error, verdict] of cases) {
    const started = performance.now();
    const out = explainUrl(url(digits));
    const verified = await verifyUrlSignature("secret", url(digits));
    const elapsed = performance.now() - started;
    assert.ok(elapsed < 2000, `${digits} digits took ${elapsed.toFixed(0)} ms`);
    assert.equal(out.errors.length, 1, `${digits} digits`);
    assert.match(out.errors[0], error);
    assert.equal(verified, verdict, `${digits} digits`);
  }
});

// ---------------------------------------------------------------- the page's policy
//
// Issue #603: the page takes the signing secret, so it carries a
// Content-Security-Policy that lets only its own scripts run. A meta
// policy covers only what follows it, so it comes first in <head>.
// The page must also need nothing looser: no inline script, no
// inline event handler, no javascript: URL.

const PAGE = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const POLICY = "script-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'";

const TAG = /<([a-zA-Z][\w-]*)((?:\s+[^\s"'>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'>]+))?)*)\s*\/?>/g;
const ATTR = /([^\s"'>\/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?/g;

function pageProblems(html) {
  // A commented-out tag does not apply, so it does not count.
  const text = html.replace(/<!--[\s\S]*?-->/g, "");
  const tags = [...text.matchAll(TAG)].map((m) => ({
    name: m[1].toLowerCase(),
    attrs: new Map(
      [...m[2].matchAll(ATTR)].map((a) => [a[1].toLowerCase(), a[2] ?? a[3] ?? a[4] ?? ""]),
    ),
  }));
  const problems = [];
  const at = tags.findIndex(
    (t) => t.name === "meta" && t.attrs.get("http-equiv")?.toLowerCase() === "content-security-policy",
  );
  if (at < 0) {
    problems.push("no Content-Security-Policy meta tag");
  } else {
    const before = tags
      .slice(0, at)
      .find((t) => !["html", "head"].includes(t.name) && !(t.name === "meta" && t.attrs.has("charset")));
    if (before) {
      problems.push(`the policy follows <${before.name}>: it must come first in <head>, after <meta charset>`);
    }
    const policy = tags[at].attrs.get("content") ?? "";
    const directives = new Map();
    for (const part of policy.split(";")) {
      const [name, ...sources] = part.trim().toLowerCase().split(/\s+/);
      if (name && !directives.has(name)) directives.set(name, sources);
    }
    const scripts = directives.get("script-src")?.join(" ");
    if (scripts !== "'self'") {
      problems.push(`script-src is ${JSON.stringify(scripts)}: it must be 'self' alone`);
    }
    for (const word of policy.match(/'[\w-]*unsafe[\w-]*'/gi) ?? []) {
      problems.push(`the policy allows ${word}`);
    }
    for (const name of directives.keys()) {
      if (name.startsWith("script-src-")) problems.push(`the policy sets ${name}, which overrides script-src`);
    }
    for (const name of ["object-src", "base-uri", "form-action"]) {
      const sources = directives.get(name)?.join(" ");
      if (sources !== "'none'") problems.push(`${name} is ${JSON.stringify(sources)}: it must be 'none'`);
    }
  }
  for (const tag of tags) {
    if (tag.name === "script") {
      const src = tag.attrs.get("src");
      if (src === undefined) {
        problems.push("an inline <script>");
      } else if (/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(src.trim())) {
        problems.push(`a <script> from another origin: ${src}`);
      }
    }
    for (const [name, value] of tag.attrs) {
      if (/^on[a-z]+$/.test(name)) problems.push(`an inline event handler: <${tag.name} ${name}>`);
      if (/^\s*javascript:/i.test(value)) problems.push(`a javascript: URL in <${tag.name} ${name}>`);
    }
  }
  return problems;
}

test("the page carries its CSP first in <head>, and needs nothing looser (issue #603)", () => {
  assert.deepEqual(pageProblems(PAGE), []);
  // The policy is the one playground/README.md documents, word for word.
  const readme = readFileSync(new URL("../README.md", import.meta.url), "utf8");
  assert.ok(PAGE.includes(`content="${POLICY}"`), "index.html does not carry the documented policy");
  assert.ok(readme.includes(POLICY), "playground/README.md does not document the policy");
});

test("the CSP check catches a page without the policy or with a looser one (issue #603)", () => {
  const meta = PAGE.match(/<meta http-equiv="Content-Security-Policy"[^>]*>\n/)?.[0] ?? "";
  const swap = (from, to) => (p) => p.replace(from, to);
  const move = (after) => (p) => p.replace(meta, "").replace(after, `${after}${meta}`);
  const add = (html) => swap("</body>", `${html}\n</body>`);
  const self = "script-src 'self'";
  const loosened = [
    ["policy removed", swap(meta, ""), /no Content-Security-Policy/],
    ["policy commented out", swap(meta, `<!-- ${meta.trim()} -->\n`), /no Content-Security-Policy/],
    ["policy after the title", move("</title>\n"), /first in <head>/],
    ["policy after the script", move("</script>\n"), /first in <head>/],
    ["'unsafe-inline'", swap(self, `${self} 'unsafe-inline'`), /allows 'unsafe-inline'/],
    ["'unsafe-eval'", swap(self, `${self} 'unsafe-eval'`), /allows 'unsafe-eval'/],
    ["another script host", swap(self, `${self} https://cdn.example.com`), /script-src is/],
    ["a wildcard", swap(self, "script-src *"), /script-src is/],
    ["script-src dropped", swap(`${self}; `, ""), /script-src is undefined/],
    ["script-src-elem", swap("object-src", "script-src-elem https://cdn.example.com; object-src"), /sets script-src-elem/],
    ["object-src dropped", swap(" object-src 'none';", ""), /object-src is undefined/],
    ["base-uri 'self'", swap("base-uri 'none'", "base-uri 'self'"), /base-uri is "'self'"/],
    ["form-action dropped", swap("; form-action 'none'", ""), /form-action is undefined/],
    ["an inline script", add("<script>void 0</script>"), /an inline <script>/],
    ["a foreign script", add('<script src="https://evil.example/x.js"></script>'), /another origin/],
    ["a protocol-relative script", add('<script src="//evil.example/x.js"></script>'), /another origin/],
    ["an onclick", swap('id="copy"', 'id="copy" onclick="copy()"'), /event handler: <button onclick>/],
    ["an unquoted ONERROR", swap('id="preview"', 'id="preview" ONERROR=x()'), /event handler: <img onerror>/],
    ["a javascript: link", add('<a href="JavaScript:void 0">x</a>'), /javascript: URL/],
  ];
  for (const [name, mutate, expected] of loosened) {
    const page = mutate(PAGE);
    assert.ok(page !== PAGE, `the "${name}" edit does not apply to index.html`);
    const problems = pageProblems(page);
    assert.ok(problems.some((p) => expected.test(p)), `${name}: ${JSON.stringify(problems)}`);
  }
});

test("the page's scripts import only same-origin modules and need no eval (issue #603)", () => {
  // script-src 'self' blocks a module from another origin and eval,
  // so neither may be needed for the page to work under its policy.
  const dir = new URL("../", import.meta.url);
  const files = ["app.mjs", "lib.mjs", ...readdirSync(new URL("sdk/", dir)).map((f) => `sdk/${f}`)];
  for (const file of files) {
    const source = readFileSync(new URL(file, dir), "utf8");
    const imports = [
      ...source.matchAll(/^(?:import|export)\s+(?:[\w$*\s{},]*?\s*from\s*)?["']([^"']+)["']/gm),
      ...source.matchAll(/\bimport\s*\(\s*["']([^"']+)["']/g),
    ];
    for (const [, specifier] of imports) {
      assert.match(specifier, /^\.{1,2}\//, `${file} imports ${specifier}`);
    }
    assert.doesNotMatch(source, /\beval\s*\(|\bFunction\s*\(/, file);
  }
});
