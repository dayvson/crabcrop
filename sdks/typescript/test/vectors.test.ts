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

test("explainUrl verdicts match the server grammar (audit P-3 corpus)", () => {
  const SIG = "a".repeat(64);
  const SRC = "https://example.com/a.jpg";
  const cases: Array<[string, boolean, string]> = [
    ["exp:99999999999999999999999999/300x200/" + SRC, true, SRC],
    ["300x200/https://example.com/%2e%2e/a.jpg", false, "https://example.com/%2e%2e/a.jpg"],
    ["300x/" + SRC, false, SRC],
    ["x200/" + SRC, false, SRC],
    ["fit-in:300x/" + SRC, false, SRC],
    ["fit-in:-300x200/" + SRC, true, SRC],
    ["fit-in:300x200:adaptive/" + SRC, false, SRC],
    ["fit-in:300x200:sideways/" + SRC, true, SRC],
    ["fit-in:300x200:adaptive:extra/" + SRC, true, SRC],
    ["fit-in:adaptive/" + SRC, true, SRC],
    ["pad:100x100:adaptive/" + SRC, true, SRC],
    ["crop:100x100:sideways/" + SRC, true, SRC],
    ["crop:100x100:smart/" + SRC, false, SRC],
    ["crop:100x100:top-left/" + SRC, false, SRC],
    ["crop:100x100:bottom-right/" + SRC, false, SRC],
    ["crop:100x100:left-top/" + SRC, true, SRC],
    ["crop:300x0/" + SRC, true, SRC],
    ["crop:-300x200/" + SRC, true, SRC],
    ["crop:origx100/" + SRC, true, SRC],
    ["pad:300xorig/" + SRC, true, SRC],
    ["pad:-200x100/" + SRC, true, SRC],
    ["filters:quality(999)/" + SRC, true, SRC],
    ["filters:quality(80):blur(2)/" + SRC, false, SRC],
    ["filters:blur(-1)/" + SRC, true, SRC],
    ["filters:dpr(7)/" + SRC, true, SRC],
    ["filters:dpr(1.5)/" + SRC, false, SRC],
    ["pad:100x100:blur/" + SRC, false, SRC],
    ["pad:100x100:ff00aa/" + SRC, false, SRC],
    ["pad:100x100:red/" + SRC, true, SRC],
    ["crop-rect:0x0:50x50/" + SRC, false, SRC],
    ["crop-rect:0x0:50/" + SRC, true, SRC],
    ["focal:0x0:50x50/" + SRC, false, SRC],
    ["focal:0x0:50/" + SRC, true, SRC],
    ["16384x10/" + SRC, false, "16384x10/" + SRC],
    ["exp:1893456000/300x200/" + SRC, false, SRC],
    ["maxage:60/300x200/" + SRC, false, SRC],
    ["maxage:0/300x200/" + SRC, false, SRC],
    ["maxage:31536000/300x200/" + SRC, false, SRC],
    ["maxage:31536001/300x200/" + SRC, true, SRC],
    ["maxage:soon/300x200/" + SRC, true, SRC],
    ["debug:focal/crop:100x100:smart/" + SRC, false, SRC],
    ["debug:focal/" + SRC, false, SRC],
    ["debug:banana/300x200/" + SRC, true, SRC],
    ["debug:/300x200/" + SRC, true, SRC],
    ["filters:radius(12)/" + SRC, false, SRC],
    ["filters:radius(1,2,3,4):mask(circle)/" + SRC, false, SRC],
    ["filters:radius(1,2)/" + SRC, true, SRC],
    ["filters:border(2,00ff00)/" + SRC, false, SRC],
    ["filters:border(0,00ff00)/" + SRC, true, SRC],
    ["filters:mask(square)/" + SRC, true, SRC],
    ["filters:sepia:duotone(000000,ffffff):flip(h)/" + SRC, false, SRC],
    ["filters:duotone(000000)/" + SRC, true, SRC],
    ["filters:curve(rgb,0,0,255,255)/" + SRC, false, SRC],
    ["filters:curve(r,0,64,255,255):curve(b,32,10,224,245)/" + SRC, false, SRC],
    ["filters:curve(x,0,0,255,255)/" + SRC, true, SRC],
    ["filters:curve(rgb,0,0)/" + SRC, true, SRC],
    ["filters:curve(rgb,0,0,255)/" + SRC, true, SRC],
    ["filters:curve(rgb,0,0,256,255)/" + SRC, true, SRC],
    ["filters:curve(rgb,128,0,64,255)/" + SRC, true, SRC],
    ["filters:curve(rgb,0,0,0,255)/" + SRC, true, SRC],
    ["filters:convolution(0,-1,0,-1,5,-1,0,-1,0,0)/" + SRC, false, SRC],
    ["filters:convolution(0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625,1)/" + SRC, false, SRC],
    ["filters:convolution(0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0)/" + SRC, false, SRC],
    ["filters:convolution(0,0,0,0,1,0,0,0,0)/" + SRC, true, SRC],
    ["filters:convolution(0,0,0,0,1,0,0,0,0,0,0)/" + SRC, true, SRC],
    ["filters:convolution(0,-1,0,-1,5,-1,0,-1,0,2)/" + SRC, true, SRC],
    ["filters:convolution(0,-1,0,-1,1e2,-1,0,-1,0,0)/" + SRC, true, SRC],
    ["filters:convolution(0,-1,0,-1,.5,-1,0,-1,0,0)/" + SRC, true, SRC],
    ["filters:convolution(0,-1,0,-1,1025,-1,0,-1,0,0)/" + SRC, true, SRC],
    ["filters:noise(0)/" + SRC, false, SRC],
    ["filters:noise(40)/" + SRC, false, SRC],
    ["filters:noise(100)/" + SRC, false, SRC],
    ["filters:noise(101)/" + SRC, true, SRC],
    ["filters:noise(256)/" + SRC, true, SRC],
    ["filters:noise(-1)/" + SRC, true, SRC],
    ["filters:noise(1.5)/" + SRC, true, SRC],
    ["filters:monochrome(ff8000):pixelate(8)/" + SRC, false, SRC],
    ["filters:pixelate(0)/" + SRC, true, SRC],
    ["filters:flip(x)/" + SRC, true, SRC],
    ["filters:blur_region(1,2,30,40):pixelate_region(0,0,5,5)/" + SRC, false, SRC],
    ["filters:blur_region(1,2,0,40)/" + SRC, true, SRC],
    ["filters:pixelate_region(0,0,10,16384)/" + SRC, true, SRC],
    ["filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,80,multiply)/" + SRC, false, SRC],
    ["filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,screen,25)/" + SRC, false, SRC],
    ["filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,darken)/" + SRC, true, SRC],
    ["filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,101,normal)/" + SRC, true, SRC],
    ["filters:blurclass(person)/" + SRC, false, SRC],
    ["filters:pixelateclass(traffic%20light)/" + SRC, false, SRC],
    ["filters:blurclass()/" + SRC, true, SRC],
    ["filters:blurclass(per%24on)/" + SRC, true, SRC],
    ["filters:pixelateclass/" + SRC, true, SRC],
    ["filters:keep_profile/" + SRC, false, SRC],
    ["filters:format(mp4)/" + SRC, false, SRC],
    ["filters:format(webm)/" + SRC, false, SRC],
    ["filters:format(mov)/" + SRC, true, SRC],
    // AP-23 parity cases: each verdict below is the Rust
    // parser's, verified against src/url.rs.
    ["crop-rect:50x50:0x0/" + SRC, true, SRC],
    ["crop-rect:50x50:51x51/" + SRC, false, SRC],
    ["focal:50x50:0x0/" + SRC, true, SRC],
    ["focal:50x50:51x51/" + SRC, false, SRC],
    ["filters:blur(0x10)/" + SRC, true, SRC],
    ["filters:contrast(0x2)/" + SRC, true, SRC],
    ["filters:blur( 1.5)/" + SRC, true, SRC],
    ["filters:blur(1e2)/" + SRC, false, SRC],
    ["filters:blur(.5)/" + SRC, false, SRC],
    ["filters:overlay(a,b.png,0,0,80,normal)/" + SRC, true, SRC],
    ["filters:overlay(a,b.png,0,0,80,normal,25)/" + SRC, false, SRC],
    ["300x200//" + SRC, true, SRC],
    ["filters:blurclass(" + "é".repeat(40) + ")/" + SRC, true, SRC],
    ["filters:blurclass(" + "é".repeat(32) + ")/" + SRC, false, SRC],
    ["filters:brighten(+5)/" + SRC, false, SRC],
    ["filters:quality(+80)/" + SRC, false, SRC],
    ["filters:text(Hello,+12,ff8800,center)/" + SRC, false, SRC],
    ["+300x200/" + SRC, false, SRC],
    ["filters:radius(+12)/" + SRC, false, SRC],
    ["filters:pixelate(+8)/" + SRC, false, SRC],
  ];
  for (const [remainder, expectError, expectSource] of cases) {
    const out = explainUrl(`https://img.example.com/${SIG}/${remainder}`);
    assert.equal(out.source, expectSource, remainder);
    assert.equal(out.errors.length > 0, expectError, `${remainder}: ${JSON.stringify(out.errors)}`);
  }
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
