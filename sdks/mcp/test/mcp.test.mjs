/**
 * End-to-end test: spawns the MCP server and a real crabcrop
 * binary and runs the full loop (sign, fetch, meta, purge).
 *
 * Precondition: the debug server binary is built
 * (`cargo build` in the repository root). The image source is
 * the repository's bench fixture, served locally.
 */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHmac } from "node:crypto";
import { existsSync } from "node:fs";
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
    cwd: `${root}../spikes/bench/fixtures`,
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
