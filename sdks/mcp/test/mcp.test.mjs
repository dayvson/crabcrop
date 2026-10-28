/**
 * End-to-end test: spawns the MCP server and a real crabcrop
 * binary and runs the full loop (sign, fetch, meta, purge).
 *
 * Precondition: the debug server binary is built
 * (`cargo build` in the repository root). The image source is
 * the repository's shared test fixture (test-fixtures/),
 * served locally; CRABCROP_TEST_FIXTURES overrides its
 * directory.
 */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHmac } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../..", import.meta.url));
const binary = `${root}target/debug/crabcrop`;

function waitForHealth(port, path = "/health", tries = 50) {
  return new Promise((resolve, reject) => {
    const attempt = async (n) => {
      try {
        const resp = await fetch(`http://127.0.0.1:${port}${path}`);
        if (resp.status < 500) return resolve();
        throw new Error("not ok");
      } catch {
        if (n <= 0) return reject(new Error(`server on ${port} never became healthy`));
        setTimeout(() => void attempt(n - 1), 200);
      }
    };
    void attempt(tries);
  });
}

class McpClient {
  constructor(child) {
    this.child = child;
    this.nextId = 1;
    this.pending = new Map();
    const rl = createInterface({ input: child.stdout });
    rl.on("line", (line) => {
      const msg = JSON.parse(line);
      const resolve = this.pending.get(msg.id);
      if (resolve) {
        this.pending.delete(msg.id);
        resolve(msg.result);
      }
    });
  }
  request(method, params) {
    const id = this.nextId++;
    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      this.child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
    });
  }
  notify(method, params) {
    this.child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", method, params })}\n`);
  }
  async call(name, args) {
    const result = await this.request("tools/call", { name, arguments: args });
    return result.content.map((c) => c.text).join("\n");
  }
}

test("the MCP tools work end to end against a live server", { timeout: 90000 }, async () => {
  assert.ok(existsSync(binary), "precondition: cargo build must have produced target/debug/crabcrop");

  const server = spawn(binary, [], {
    env: { ...process.env, CRABCROP_SECRET: "mcp-test-secret-0123456", PORT: "3323" },
    stdio: "ignore",
  });
  const fixtureServer = spawn("python3", ["-m", "http.server", "8912"], {
    cwd: process.env.CRABCROP_TEST_FIXTURES ?? `${root}test-fixtures`,
    stdio: "ignore",
  });
  const mcp = spawn(
    process.execPath,
    ["--import", `${root}sdks/mcp/test/register.mjs`, `${root}sdks/mcp/src/index.ts`],
    {
      env: {
        ...process.env,
        CRABCROP_BASE: "http://127.0.0.1:3323",
        CRABCROP_SECRET: "mcp-test-secret-0123456",
      },
      stdio: ["pipe", "pipe", "inherit"],
    },
  );
  try {
    await waitForHealth(3323);
    await waitForHealth(8912, "/src00.jpg");
    const client = new McpClient(mcp);

    const init = await client.request("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "mcp-test", version: "0" },
    });
    assert.equal(init.serverInfo.name, "crabcrop");
    // The reported version is a release version site: it must
    // match the package version (scripts/check_versions.py).
    const pkg = JSON.parse(readFileSync(`${root}sdks/mcp/package.json`, "utf8"));
    assert.equal(init.serverInfo.version, pkg.version);
    client.notify("notifications/initialized", {});

    const list = await client.request("tools/list", {});
    assert.deepEqual(
      list.tools.map((t) => t.name).sort(),
      ["explain_url", "fetch_meta", "purge_result", "sign_url"],
    );
    // Every schema is closed: misspelled argument names are
    // rejected by clients instead of silently ignored.
    for (const tool of list.tools) {
      assert.equal(tool.inputSchema.additionalProperties, false, tool.name);
    }

    const source = "http://127.0.0.1:8912/src00.jpg";
    // Arguments are validated, not String(...)-coerced: a
    // missing operations argument errors instead of signing
    // a URL over the literal segment "undefined", and a
    // numeric one errors instead of stringifying.
    const missing = await client.call("sign_url", { source });
    assert.match(missing, /missing required argument: operations/);
    const wrongType = await client.call("sign_url", { operations: 42, source });
    assert.match(wrongType, /argument operations must be a string/);

    const url = await client.call("sign_url", { operations: "300x200", source });
    const remainder = `300x200/${source}`;
    const expectedSig = createHmac("sha256", "mcp-test-secret-0123456").update(remainder).digest("hex");
    assert.equal(url, `http://127.0.0.1:3323/${expectedSig}/${remainder}`);

    // The signed URL really serves, populating the cache.
    const img = await fetch(url);
    assert.equal(img.status, 200);
    await img.arrayBuffer();

    const meta = JSON.parse(await client.call("fetch_meta", { url }));
    assert.equal(meta.width, 300);
    assert.equal(meta.height, 200);

    // fetch_meta refuses URLs outside the configured origin
    // (audit PL-1): the fixture server is a different origin.
    const refused = await client.call("fetch_meta", { url: source });
    assert.match(refused, /refusing to fetch/);

    // fetch_meta also refuses the system endpoints on the
    // configured origin (issue #271): they are not meta
    // targets, and /metrics is token-gated besides.
    const sysRefused = await client.call("fetch_meta", {
      url: "http://127.0.0.1:3323/metrics",
    });
    assert.match(sysRefused, /system endpoint/);

    const explanation = await client.call("explain_url", { url });
    assert.match(explanation, /signature \(hmac\)/);
    assert.match(explanation, /resize to 300px width by 200px height/);
    assert.match(explanation, new RegExp(`source: ${source.replaceAll("/", "\\/")}`));

    assert.equal(await client.call("purge_result", { path: remainder }), "purged 1");
    assert.equal(await client.call("purge_result", { path: remainder }), "purged 0");
  } finally {
    mcp.kill();
    fixtureServer.kill();
    server.kill();
  }
});

test("purge_result only purges on the configured origin and refuses dot segments", { timeout: 60000 }, async () => {
  // Recording servers stand in for the configured server
  // and for an internal target: any request that lands on
  // one proves the tool actually fetched that origin, so
  // the oracle question is answered by counting hits, not
  // by trusting the response text. Before the R2-10 fix,
  // purge_result POSTed to any caller-chosen base and
  // returned the body, which made it an SSRF oracle: a
  // steered caller could probe internal endpoints and read
  // the responses back.
  const { createServer } = await import("node:http");
  const hits = { configured: [], internal: [] };
  const recorder = (bucket) =>
    createServer((req, res) => {
      bucket.push(`${req.method} ${req.url}`);
      res.writeHead(200, { "content-type": "text/plain" });
      res.end("purged 1");
    });
  const configuredServer = recorder(hits.configured);
  const internalServer = recorder(hits.internal);
  await new Promise((resolve) => configuredServer.listen(0, "127.0.0.1", resolve));
  await new Promise((resolve) => internalServer.listen(0, "127.0.0.1", resolve));
  const configuredOrigin = `http://127.0.0.1:${configuredServer.address().port}`;
  const internalOrigin = `http://127.0.0.1:${internalServer.address().port}`;

  const spawnMcp = (env) =>
    spawn(
      process.execPath,
      ["--import", `${root}sdks/mcp/test/register.mjs`, `${root}sdks/mcp/src/index.ts`],
      { env, stdio: ["pipe", "pipe", "inherit"] },
    );
  const secretEnv = { ...process.env, CRABCROP_SECRET: "mcp-test-secret-0123456" };
  const mcp = spawnMcp({ ...secretEnv, CRABCROP_BASE: configuredOrigin });
  const noBaseEnv = { ...secretEnv };
  delete noBaseEnv.CRABCROP_BASE;
  const mcpNoBase = spawnMcp(noBaseEnv);
  const init = {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "mcp-test", version: "0" },
  };
  try {
    const client = new McpClient(mcp);
    await client.request("initialize", init);
    client.notify("notifications/initialized", {});

    const path = "300x200/https://example.com/a.jpg";

    // A legitimate purge still works: the default base is
    // the configured origin, so the POST goes out and the
    // server's answer comes back.
    assert.equal(await client.call("purge_result", { path }), "purged 1");
    assert.equal(hits.configured.length, 1);
    assert.match(
      hits.configured[0],
      /^POST \/purge\/[0-9a-f]{64}\/300x200\/https:\/\/example\.com\/a\.jpg$/,
    );

    // A per-call base naming the same origin explicitly is
    // equally fine.
    assert.equal(await client.call("purge_result", { path, base: configuredOrigin }), "purged 1");
    assert.equal(hits.configured.length, 2);

    // Every other origin is refused before any fetch, and
    // the refusal has the same shape whatever state the
    // target is in: listening, refusing connections, not
    // resolving at all, or a link-local metadata address.
    // No request reaches any of them, so there is no
    // timing, status, or body signal to tell them apart.
    const refusal = new RegExp(
      `^purge_result: refusing to purge .*; only the configured server origin \\(${configuredOrigin.replaceAll(".", "\\.")}\\) is allowed$`,
    );
    for (const base of [
      internalOrigin,
      "http://127.0.0.1:1",
      "http://nonexistent.invalid",
      "http://169.254.169.254",
    ]) {
      assert.match(await client.call("purge_result", { path, base }), refusal);
    }
    assert.equal(hits.internal.length, 0);
    assert.equal(hits.configured.length, 2);

    // Dot segments in the path are refused even against
    // the configured origin: parsed as a URL they resolve
    // out of the /purge/ prefix onto arbitrary paths, and
    // in an s3:// target they would traverse past the
    // bucket the signature was computed for.
    for (const badPath of [
      "300x200/../../admin/hook",
      "300x200/./x.jpg",
      "300x200/%2e%2e/%2e%2e/admin/hook",
      "300x200/s3://bucket/../other-bucket/secret.jpg",
    ]) {
      const out = await client.call("purge_result", { path: badPath });
      assert.match(out, /refusing a path with dot segments/);
    }
    assert.equal(hits.configured.length, 2);

    // Dots inside a segment name are ordinary content, not
    // segments: the purge goes through.
    assert.equal(
      await client.call("purge_result", { path: "300x200/https://example.com/my.photo..final.jpg" }),
      "purged 1",
    );
    assert.equal(hits.configured.length, 3);

    // With no CRABCROP_BASE configured there is no origin
    // to check a per-call base against, so the tool refuses
    // outright instead of trusting the argument.
    const clientNoBase = new McpClient(mcpNoBase);
    await clientNoBase.request("initialize", init);
    clientNoBase.notify("notifications/initialized", {});
    const out = await clientNoBase.call("purge_result", { path, base: internalOrigin });
    assert.match(out, /purge_result requires CRABCROP_BASE to be set/);
    assert.equal(hits.internal.length, 0);
  } finally {
    mcp.kill();
    mcpNoBase.kill();
    configuredServer.close();
    internalServer.close();
  }
});

test("purge_result gives up on a hung endpoint", { timeout: 60000 }, async () => {
  // An endpoint that accepts the connection and never
  // answers: the purge fetch must time out (10 s) and
  // report, not hang the tool call forever.
  const { createServer } = await import("node:http");
  const hung = createServer(() => {});
  await new Promise((resolve) => hung.listen(0, "127.0.0.1", resolve));
  const mcp = spawn(
    process.execPath,
    ["--import", `${root}sdks/mcp/test/register.mjs`, `${root}sdks/mcp/src/index.ts`],
    {
      env: {
        ...process.env,
        CRABCROP_BASE: `http://127.0.0.1:${hung.address().port}`,
        CRABCROP_SECRET: "mcp-test-secret-0123456",
      },
      stdio: ["pipe", "pipe", "inherit"],
    },
  );
  try {
    const client = new McpClient(mcp);
    await client.request("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "mcp-test", version: "0" },
    });
    client.notify("notifications/initialized", {});
    const out = await client.call("purge_result", { path: "300x200/https://example.com/a.jpg" });
    assert.match(out, /purge request failed/);
  } finally {
    mcp.kill();
    hung.close();
  }
});
