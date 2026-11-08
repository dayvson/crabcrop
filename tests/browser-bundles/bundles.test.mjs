// Real bundler builds of the client code the framework
// integrations ship (issue #381). Each integration that runs
// in the browser is used the way its README says (the file
// copied into a project that installs @crabcrop/url), built
// with Vite's client resolution, and run in a sandbox with a
// browser's globals and none of Node's. Nuxt gets a full
// `nuxi build` with @nuxt/image, which also proves the signing
// secret stays out of everything a browser loads while server
// rendering still signs.
//
// Next.js bundles its loader with webpack or Turbopack rather
// than Vite; the Vite build stands in for any bundler that
// resolves the `browser` condition. The Astro service is not
// here: Astro runs image services on the server only.
//
// Needs the SDK built first (npm run build in sdks/typescript).
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { createHmac, webcrypto } from "node:crypto";
import {
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:net";
import { dirname, join } from "node:path";
import { before, test } from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import { build, createLogger } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..", "..");
const work = join(here, ".work");
const BASE = "https://images.example.com";
const SRC = "https://cdn.example.com/a.jpg";

before(() => {
  assert.ok(
    existsSync(join(repo, "sdks/typescript/dist/index.js")),
    "build the SDK first: npm run build in sdks/typescript",
  );
  rmSync(work, { recursive: true, force: true });
  mkdirSync(work, { recursive: true });
});

/**
 * A Vite client build of `entry` (written into a fresh
 * project directory beside copies of `copy`), as one IIFE.
 * Returns the bundle and every warning Vite logged.
 */
async function viteBundle(name, { copy = [], entry, define = {} }) {
  const dir = join(work, name);
  mkdirSync(dir, { recursive: true });
  for (const [from, to] of copy) {
    cpSync(join(repo, from), join(dir, to));
  }
  writeFileSync(join(dir, "entry.ts"), entry);
  const warnings = [];
  const logger = createLogger("warn");
  logger.warn = (msg) => warnings.push(msg);
  logger.warnOnce = (msg) => warnings.push(msg);
  await build({
    configFile: false,
    root: dir,
    customLogger: logger,
    define,
    build: {
      outDir: join(dir, "dist"),
      emptyOutDir: true,
      minify: false,
      lib: {
        entry: join(dir, "entry.ts"),
        formats: ["iife"],
        name: "bundle",
        fileName: () => "bundle.js",
      },
    },
  });
  return { code: readFileSync(join(dir, "dist", "bundle.js"), "utf8"), warnings };
}

/** No Node built-in reached the client graph. */
function assertBrowserClean({ code, warnings }) {
  // Vite turns a node: import in client code into an empty
  // stub and warns; the call then fails in the browser.
  assert.deepEqual(
    warnings.filter((w) => /externalized for browser compatibility/.test(w)),
    [],
  );
  assert.doesNotMatch(code, /["']node:/);
}

/**
 * Run an IIFE bundle with a browser's globals (window, self,
 * TextEncoder, TextDecoder, crypto.getRandomValues) and none
 * of Node's (no Buffer, process, or require), returning its
 * exports.
 */
function runInBrowserSandbox(code) {
  const sandbox = {
    console,
    TextEncoder,
    TextDecoder,
    crypto: { getRandomValues: (array) => webcrypto.getRandomValues(array) },
  };
  sandbox.window = sandbox;
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  return vm.runInContext(`${code}\n;bundle`, sandbox);
}

test("@crabcrop/url: a Vite client build signs and seals with no Node built-ins", async () => {
  const bundle = await viteBundle("sdk", {
    entry: `export { buildURL, seal, sealWithNonce, sign, unseal, verifyUrlSignature } from "@crabcrop/url";\n`,
  });
  assertBrowserClean(bundle);
  const sdk = runInBrowserSandbox(bundle.code);
  const vectors = JSON.parse(readFileSync(join(repo, "spec/test-vectors.json"), "utf8"));
  let signed = 0;
  let sealed = 0;
  for (const v of vectors) {
    if (v.signature !== undefined) {
      assert.equal(sdk.sign(v.secret, v.remainder), v.signature, v.description);
      signed++;
    }
    if (v.sealed === true) {
      const nonce = Uint8Array.from(Buffer.from(v.nonce, "hex"));
      assert.equal(sdk.sealWithNonce(v.secret, v.remainder, nonce, v.key_id), v.token, v.description);
      assert.equal(sdk.unseal(v.secret, v.token), v.remainder);
      sealed++;
    }
  }
  assert.ok(signed > 0 && sealed > 0);
  const remainder = "300x200/https://example.com/photo.jpg";
  assert.equal(sdk.unseal("k", sdk.seal("k", remainder)), remainder);
  const url = sdk.buildURL(BASE, "k", "300x200", "https://example.com/photo.jpg");
  assert.equal(sdk.verifyUrlSignature("k", url), true);
});

test("SvelteKit loader: a Vite client build runs in the browser and never signs there", async () => {
  const bundle = await viteBundle("sveltekit", {
    copy: [["frameworks/sveltekit/crabcrop-loader.ts", "crabcrop-loader.ts"]],
    entry: `import { createCrabcropLoader } from "./crabcrop-loader.ts";
export const images = createCrabcropLoader({ base: "${BASE}", preset: "card" });
`,
  });
  assertBrowserClean(bundle);
  const { images } = runInBrowserSandbox(bundle.code);
  assert.equal(images.url(SRC, { width: 800 }), `${BASE}/unsafe/preset:card/800x0/${SRC}`);
  assert.equal(
    images.srcset(SRC, [320, 640]),
    `${BASE}/unsafe/preset:card/320x0/${SRC} 320w, ${BASE}/unsafe/preset:card/640x0/${SRC} 640w`,
  );
});

test("Hydrogen loader: a Vite client build runs in the browser and never signs there", async () => {
  const bundle = await viteBundle("hydrogen", {
    copy: [["frameworks/hydrogen/crabcrop-loader.ts", "crabcrop-loader.ts"]],
    entry: `import { createCrabcropLoader } from "./crabcrop-loader.ts";
export const loader = createCrabcropLoader({ base: "${BASE}", preset: "card", secret: "server-only" });
`,
  });
  assertBrowserClean(bundle);
  const { loader } = runInBrowserSandbox(bundle.code);
  assert.equal(loader({ src: SRC, width: 400, scale: 2 }), `${BASE}/unsafe/preset:card/800x0/${SRC}`);
});

test("Next.js loader: a client build runs in the browser and never signs there", async () => {
  const bundle = await viteBundle("nextjs", {
    copy: [["frameworks/nextjs/crabcrop-loader.ts", "crabcrop-loader.ts"]],
    entry: `export { default as loader } from "./crabcrop-loader.ts";\n`,
    // Next.js inlines NEXT_PUBLIC_* variables into client code.
    define: {
      "process.env.NEXT_PUBLIC_CRABCROP_BASE": JSON.stringify(BASE),
      "process.env.NEXT_PUBLIC_CRABCROP_PRESET": JSON.stringify("card"),
      "process.env.NEXT_PUBLIC_CRABCROP_ORIGIN": JSON.stringify(""),
    },
  });
  assertBrowserClean(bundle);
  const { loader } = runInBrowserSandbox(bundle.code);
  assert.equal(loader({ src: SRC, width: 800 }), `${BASE}/unsafe/preset:card/800x0/${SRC}`);
});

/** Every file under `dir`. */
function walk(dir) {
  return readdirSync(dir, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => join(entry.parentPath, entry.name));
}

function freePort() {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

async function fetchWhenReady(url) {
  for (let attempt = 0; attempt < 100; attempt++) {
    try {
      const response = await fetch(url);
      return { status: response.status, body: await response.text() };
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  throw new Error(`server never answered at ${url}`);
}

test("Nuxt provider: a full nuxi build keeps the secret server-side, and server rendering signs", async () => {
  const SENTINEL = "browser-bundles-sentinel-5f2c9e"; // nuxt-app/nuxt.config.ts
  const RUNTIME_SECRET = "runtime-secret-from-the-environment";
  const dir = join(work, "nuxt");
  cpSync(join(here, "nuxt-app"), dir, { recursive: true });
  mkdirSync(join(dir, "providers"));
  cpSync(join(repo, "frameworks/nuxt/crabcrop.ts"), join(dir, "providers/crabcrop.ts"));
  const result = spawnSync(join(here, "node_modules/.bin/nuxi"), ["build", dir], {
    encoding: "utf8",
    env: { ...process.env, NUXT_TELEMETRY_DISABLED: "1" },
  });
  const log = `${result.stdout}\n${result.stderr}`;
  assert.equal(result.status, 0, log);
  assert.doesNotMatch(log, /externalized for browser compatibility/);

  // The options module @nuxt/image generates for client code
  // carries the provider options, and no secret.
  const generated = readFileSync(join(dir, ".nuxt/image-options.mjs"), "utf8");
  assert.match(generated, /images\.example\.com/);
  assert.ok(!generated.includes(SENTINEL), generated);
  // Nothing the browser can load carries it either, while the
  // server build does (so the search can find it).
  for (const file of walk(join(dir, ".output/public"))) {
    assert.ok(!readFileSync(file, "latin1").includes(SENTINEL), `secret in ${file}`);
  }
  assert.ok(walk(join(dir, ".output/server")).some((f) => readFileSync(f, "latin1").includes(SENTINEL)));

  // Server rendering signs with the secret from private
  // runtime config, set by the environment as in production;
  // the page sent to the browser does not contain it.
  const port = await freePort();
  const server = spawn(process.execPath, [join(dir, ".output/server/index.mjs")], {
    env: { ...process.env, HOST: "127.0.0.1", PORT: String(port), NUXT_CRABCROP_SECRET: RUNTIME_SECRET },
    stdio: "ignore",
  });
  try {
    const { status, body } = await fetchWhenReady(`http://127.0.0.1:${port}/`);
    assert.equal(status, 200, body);
    assert.ok(!body.includes(RUNTIME_SECRET) && !body.includes(SENTINEL), body);
    const img = body.match(/<img[^>]*>/)?.[0] ?? "";
    const urls = [
      ...img.matchAll(/(?:src|srcset)="([^"]*)"/g),
    ].flatMap((m) => m[1].split(",").map((entry) => entry.trim().split(" ")[0]));
    assert.ok(urls.length > 0, body);
    for (const url of urls) {
      const [, signature, remainder] = url.match(/^https:\/\/images\.example\.com\/([^/]+)\/(.*)$/) ?? [];
      assert.match(remainder ?? "", /^preset:card\/\d+x0\/https:\/\/cdn\.example\.com\/a\.jpg$/, url);
      assert.equal(signature, createHmac("sha256", RUNTIME_SECRET).update(remainder).digest("hex"), url);
    }
  } finally {
    server.kill();
  }
});
