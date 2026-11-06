import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  buildSealedURL,
  buildSrcSet,
  buildURL,
  purgeGroupSignature,
  purgeSignature,
  seal,
  sealWithNonce,
  sign,
  unseal,
  uploadSignature,
  signWithKeyId,
  verifyUrlSignature,
} from "../src/index.ts";

// The canonical cross-language vectors: this package's whole
// signing suite is the shared file from the spec directory.
const vectors = JSON.parse(
  readFileSync(new URL("../../../spec/test-vectors.json", import.meta.url), "utf8"),
) as {
  description: string;
  secret: string;
  remainder: string;
  signature?: string;
  sealed?: boolean;
  key_id?: string;
  nonce?: string;
  token?: string;
}[];

for (const v of vectors) {
  if (v.signature === undefined) {
    continue;
  }
  const signature = v.signature;
  test(`vector: ${v.description}`, () => {
    assert.equal(sign(v.secret, v.remainder), signature);
  });
}

// The sealed vectors (issue #255): AES-GCM is a deterministic
// function of (key, nonce, plaintext, AAD), so an injected
// nonce pins the token byte for byte, and every token opens
// back to its remainder. A wrong secret opens none of them.
for (const v of vectors) {
  if (v.sealed !== true) {
    continue;
  }
  const token = v.token as string;
  test(`sealed vector: ${v.description}`, () => {
    const nonce = Buffer.from(v.nonce as string, "hex");
    assert.equal(sealWithNonce(v.secret, v.remainder, nonce, v.key_id), token);
    assert.equal(unseal(v.secret, token), v.remainder);
    assert.throws(() => unseal("wrong-secret", token), /did not open/);
  });
}

test("sealed round trip and builders", () => {
  const remainder = "300x200/https://example.com/photo.jpg";
  const token = seal("test-secret", remainder);
  assert.ok(token.startsWith("s1."));
  assert.equal(unseal("test-secret", token), remainder);
  const keyed = seal("test-secret", remainder, "k2");
  assert.ok(keyed.startsWith("k2:s1."));
  assert.equal(unseal("test-secret", keyed), remainder);
  // Two seals of one remainder differ (fresh random nonce);
  // a tampered token opens nowhere.
  assert.notEqual(seal("test-secret", remainder), token);
  const damaged = token.slice(0, -1) + (token.endsWith("A") ? "B" : "A");
  assert.throws(() => unseal("test-secret", damaged), /did not open/);
  assert.throws(() => unseal("test-secret", "s2.AAAA"), /did not open/);
  // The assembled URL: token in the signature slot, the
  // friendly name after it.
  const url = buildSealedURL(
    "https://img.example.com/",
    "test-secret",
    "300x200",
    "https://example.com/photo.jpg",
    { name: "photo.jpg" },
  );
  const path = url.slice("https://img.example.com/".length);
  const [segment, name] = path.split("/", 2);
  assert.equal(name, "photo.jpg");
  assert.equal(unseal("test-secret", segment), remainder);
  // No secret: no unsafe token.
  assert.throws(
    () =>
      buildSealedURL(
        "https://img.example.com",
        null as unknown as string,
        "300x200",
        "https://example.com/photo.jpg",
      ),
    /no unsafe token/,
  );
  // Over the cap: refused at mint time.
  assert.throws(() => seal("test-secret", "a".repeat(2049)), /sealing cap/);
});

test("buildURL reproduces the spec worked example", () => {
  assert.equal(
    buildURL(
      "https://images.example.com/",
      "your-secret",
      "300x200",
      "https://example.com/photo.jpg",
    ),
    "https://images.example.com/3c1f6c180116661877bd4c85fd201a178087d74c1cc188cc584e819d901308c6/300x200/https://example.com/photo.jpg",
  );
});

test("buildURL with expiry and a null secret", () => {
  assert.equal(
    buildURL("https://img.example.com", null, "preset:thumb", "s3://media/a.jpg", {
      expires: 1893456000,
    }),
    "https://img.example.com/unsafe/exp:1893456000/preset:thumb/s3://media/a.jpg",
  );
});

test("buildURL signs a root source exactly as written", () => {
  // A short source (issue #248) is signed as written; the
  // signature is the spec's canonical vector for this
  // remainder.
  assert.equal(
    buildURL("https://img.example.com", "s3cr3t", "300x200", "root:photos/cat.jpg"),
    "https://img.example.com/47e002f609cfbfc252eb1357bbee780bbbaf4c336847610caf44617b06453824/300x200/root:photos/cat.jpg",
  );
  // The character refusals still bite a short source.
  assert.throws(() =>
    buildURL("https://img.example.com", "k", "300x200", "root:photos/cat.jpg?w=1"),
  );
});

test("buildSrcSet formats entries", () => {
  assert.equal(
    buildSrcSet("https://img.example.com", null, "https://example.com/a.jpg", [320, 640]),
    "https://img.example.com/unsafe/320x0/https://example.com/a.jpg 320w, " +
      "https://img.example.com/unsafe/640x0/https://example.com/a.jpg 640w",
  );
});

test("buildSrcSet rejects widths the grammar cannot express", () => {
  const base = "https://img.example.com";
  const src = "https://example.com/a.jpg";
  // 320.5 is not an integer, -300 is the flip spelling,
  // 0 is not a size, and 16384 is past the server cap:
  // each would mint a URL the server rejects.
  for (const bad of [320.5, -300, 0, 16384, NaN]) {
    assert.throws(() => buildSrcSet(base, null, src, [bad]), /srcset width/);
  }
  assert.doesNotThrow(() => buildSrcSet(base, null, src, [1, 16383]));
});

test("buildURL rejects inputs that can never verify (audit P-4, P-5)", () => {
  const base = "https://img.example.com";
  assert.throws(() => buildURL(base, "k", "300x200", "https://example.com/a.jpg?w=1"));
  assert.throws(() => buildURL(base, "k", "300x200", "https://example.com/a.jpg#f"));
  assert.throws(() => buildURL(base, "k", "300x200", "https://example.com/my photo.jpg"));
  assert.throws(() => buildURL(base, "k", "300x200", "https://example.com/café.jpg"));
  assert.throws(() => buildURL(base, "k", "300x200", "https://example.com/a.jpg", { expires: 1.5 }));
  assert.throws(() => buildURL(base, "k", "300x200", "https://example.com/a.jpg", { expires: -1 }));
});

import { explainUrl } from "../src/index.ts";

// The explainer verdict corpus (audit P-3 / AP-23) is a
// shared spec file, not a copy in this suite: the playground
// suite and the server crate's differential tests run the
// same entries (issue #353), so a verdict can only change in
// one place and every surface re-checks it.
const verdictCorpus = JSON.parse(
  readFileSync(new URL("../../../spec/explain-verdicts.json", import.meta.url), "utf8"),
) as { cases: Array<{ remainder: string; error: boolean; source: string }> };

test("explainUrl verdicts match the server grammar (audit P-3 corpus)", () => {
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
  readFileSync(new URL("../../../spec/operation-vectors.json", import.meta.url), "utf8"),
) as {
  signature: string;
  cases: Array<{ remainder: string; ok: boolean; source: string }>;
};

test("explainUrl verdicts match the generated combination corpus", () => {
  for (const { remainder, ok, source } of operationVectors.cases) {
    const out = explainUrl(
      `https://img.example.com/${operationVectors.signature}/${remainder}`,
      1700000000,
    );
    assert.equal(out.errors.length === 0, ok, `${remainder}: ${JSON.stringify(out.errors)}`);
    assert.equal(out.source, source, remainder);
  }
});

test("explainUrl reports a sealed token as sealed, not malformed (issue #388)", () => {
  const secret = "explain-sealed-secret";
  const remainder = "300x200/https://example.com/a.jpg";
  const token = sealWithNonce(secret, remainder, new Uint8Array(12).fill(7));
  const out = explainUrl(`https://img.example.com/${token}`);
  assert.equal(out.signature?.kind, "sealed");
  assert.equal(out.signature?.value, token);
  assert.equal(out.signature?.keyId, undefined);
  assert.deepEqual(out.errors, []);
  assert.deepEqual(out.segments, []);
  assert.equal(out.source, "");
  // A keyed token reports its key ID.
  const keyed = sealWithNonce(secret, remainder, new Uint8Array(12).fill(9), "k2");
  const keyedOut = explainUrl(`https://img.example.com/${keyed}`);
  assert.equal(keyedOut.signature?.kind, "sealed");
  assert.equal(keyedOut.signature?.keyId, "k2");
  assert.deepEqual(keyedOut.errors, []);
  // A trailing path behind a token is not part of the
  // URL the server serves; the token still classifies.
  const trailed = explainUrl(`https://img.example.com/${token}/300x200/https://example.com/a.jpg`);
  assert.equal(trailed.signature?.kind, "sealed");
  assert.deepEqual(trailed.segments, []);
  // An unknown version marker is not a malformed
  // signature either: the server refuses it uniformly.
  const imitation = explainUrl(
    "https://img.example.com/s2.AAAA/300x200/https://example.com/a.jpg",
  );
  assert.equal(imitation.signature?.kind, "invalid");
  assert.match(imitation.signature?.detail ?? "", /uniform 403 refusal/);
});

test("explainUrl explains studio filters instead of calling them unknown", () => {
  const SIG = "a".repeat(64);
  const SRC = "https://example.com/a.jpg";
  const out = explainUrl(
    `https://img.example.com/${SIG}/filters:studio(product,bg:white)/${SRC}`,
  );
  assert.deepEqual(out.errors, []);
  assert.equal(out.source, SRC);
  assert.match(out.segments[0].detail ?? "", /studio\(product,bg:white\): studio mode/);
  // A chain neighbour of a studio filter still parses.
  const chained = explainUrl(
    `https://img.example.com/${SIG}/filters:grayscale:studio(scene,pad:25)/${SRC}`,
  );
  assert.deepEqual(chained.errors, []);
  // And a product-only argument under scene is an error,
  // not a silent no-op.
  const bad = explainUrl(
    `https://img.example.com/${SIG}/filters:studio(scene,bg:white)/${SRC}`,
  );
  assert.ok(bad.errors.length > 0);
});

test("explainUrl explains objects crops and rejects their malformed forms", () => {
  const SIG = "a".repeat(64);
  const SRC = "https://example.com/a.jpg";
  const good = explainUrl(
    `https://img.example.com/${SIG}/crop:400x300:objects(person,car):1.5/${SRC}`,
  );
  assert.deepEqual(good.errors, []);
  assert.equal(good.source, SRC);
  assert.match(good.segments[0].detail ?? "", /alignment: objects \(person, car\)/);
  assert.match(good.segments[0].detail ?? "", /scaled by 1\.5/);
  // The scale may be omitted (it defaults to 1), and a
  // percent-encoded class name decodes in the explanation.
  const tight = explainUrl(
    `https://img.example.com/${SIG}/crop:400x300:objects(traffic%20light)/${SRC}`,
  );
  assert.deepEqual(tight.errors, []);
  assert.match(tight.segments[0].detail ?? "", /objects \(traffic light\)/);
  // Malformed objects forms are errors, not silent crops.
  for (const bad of [
    "crop:400x300:objects()",
    "crop:400x300:objects(person,)",
    "crop:400x300:objects(person",
    "crop:400x300:objects(person)junk",
    "crop:400x300:objects(person):0",
    "crop:400x300:objects(person):101",
    "crop:400x300:objects(bad;name)",
    "crop:400x300:objects",
    "crop:-400x300:objects(person)",
  ]) {
    const out = explainUrl(`https://img.example.com/${SIG}/${bad}/${SRC}`);
    assert.ok(out.errors.length > 0, bad);
  }
});

test("encoder options and per-format quality explain (issue #319)", () => {
  const SIG = "a".repeat(64);
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

test("purge signatures match the published vectors", () => {
  // The same pinned literals the Rust crate asserts: one
  // wire scheme across every implementation.
  assert.equal(
    purgeSignature("test-secret", "300x200/https://example.com/a.jpg"),
    "9b58a43a984e47cbde3ec08333290acad06c48a5e9b85b99124ae725218e90e8",
  );
  assert.equal(
    purgeGroupSignature("test-secret", "thumb"),
    "256634813bd6237179d251ba4526685f05db8a42ae056e4a04566f00042f0439",
  );
  // Purge-group signs under its own subkey: a per-target
  // purge signature never transplants to it.
  assert.notEqual(
    purgeGroupSignature("test-secret", "thumb"),
    purgeSignature("test-secret", "purge-group/thumb"),
  );
  assert.notEqual(
    purgeGroupSignature("test-secret", "thumb"),
    purgeSignature("test-secret", "thumb"),
  );
});

test("upload signatures match the published vectors", () => {
  // The same pinned literals the Rust crate asserts: one
  // wire scheme across every implementation.
  assert.equal(
    uploadSignature("test-secret", "heroes/cat.jpg"),
    "81d87b588203fdad32f5431c0a893bf7ddad8188f5cca123a1b057a439b34b84",
  );
  // Domain separation from both other signature kinds.
  assert.notEqual(uploadSignature("test-secret", "heroes/cat.jpg"), sign("test-secret", "heroes/cat.jpg"));
  assert.notEqual(
    uploadSignature("test-secret", "heroes/cat.jpg"),
    purgeSignature("test-secret", "heroes/cat.jpg"),
  );
  // The expiring form signs `exp:<expires>/<name>`: the
  // same pinned value the Rust crate asserts, bound to
  // both the expiry and the name.
  assert.equal(
    uploadSignature("test-secret", "heroes/cat.jpg", 1893456000),
    "391061e9bd262811f9bac647b5ff535f987e2478394675407f895823f45e59d2",
  );
  assert.notEqual(
    uploadSignature("test-secret", "heroes/cat.jpg", 1893456000),
    uploadSignature("test-secret", "heroes/cat.jpg"),
  );
  assert.notEqual(
    uploadSignature("test-secret", "heroes/cat.jpg", 1893456000),
    uploadSignature("test-secret", "heroes/dog.jpg", 1893456000),
  );
});

test("signWithKeyId stamps the segment and buildURL uses it", () => {
  const remainder = "300x200/example.com/i.jpg";
  assert.equal(
    signWithKeyId("test-secret", "k2", remainder),
    `k2:${sign("test-secret", remainder)}`,
  );
  assert.equal(
    buildURL("https://img.example.com", "test-secret", "300x200", "example.com/i.jpg", { keyId: "k2" }),
    `https://img.example.com/k2:${sign("test-secret", remainder)}/300x200/example.com/i.jpg`,
  );
  assert.throws(() => signWithKeyId("test-secret", "bad id", remainder));
  assert.throws(() =>
    buildURL("https://img.example.com", null, "300x200", "example.com/i.jpg", { keyId: "k2" }),
  );
});

test("verifyUrlSignature routes by key ID with a keyring", () => {
  const remainder = "300x200/example.com/i.jpg";
  const keyed = `https://img.example.com/${signWithKeyId("old-secret", "k-old", remainder)}/${remainder}`;
  const unkeyed = `https://img.example.com/${sign("old-secret", remainder)}/${remainder}`;
  const ring = { "k-old": "old-secret", "k-new": "new-secret" };
  // Keyed: exactly the named key.
  assert.equal(verifyUrlSignature(ring, keyed), true);
  assert.equal(verifyUrlSignature({ "k-new": "new-secret" }, keyed), false);
  assert.equal(verifyUrlSignature({ "k-old": "new-secret" }, keyed), false);
  // Unkeyed: any key in the ring.
  assert.equal(verifyUrlSignature(ring, unkeyed), true);
  assert.equal(verifyUrlSignature({ "k-new": "other" }, unkeyed), false);
  // A bare secret ignores the ID and checks the tag.
  assert.equal(verifyUrlSignature("old-secret", keyed), true);
  assert.equal(verifyUrlSignature("new-secret", keyed), false);
});

test("verifyUrlSignature answers false for prototype-named key IDs", () => {
  // Key IDs may be any [A-Za-z0-9_-]{1,32}, so an attacker
  // can name one `constructor`, `toString`, or `__proto__`.
  // The keyring lookup must consult own properties only:
  // an inherited value is not a secret, and verification
  // must answer false, never throw.
  const remainder = "300x200/example.com/i.jpg";
  const ring = { "k-old": "old-secret" };
  for (const keyId of ["constructor", "toString", "__proto__", "hasOwnProperty"]) {
    const url = `https://img.example.com/${signWithKeyId("old-secret", keyId, remainder)}/${remainder}`;
    assert.equal(verifyUrlSignature(ring, url), false, keyId);
  }
  // An own property with one of those names is a real key
  // and verifies as usual.
  const url = `https://img.example.com/${signWithKeyId("old-secret", "toString", remainder)}/${remainder}`;
  assert.equal(verifyUrlSignature({ toString: "old-secret" }, url), true);
});
