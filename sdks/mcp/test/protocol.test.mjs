/**
 * JSON-RPC 2.0 and MCP conformance of the stdio transport
 * (issue #716): the error codes, protocol version
 * negotiation, batches, and a server that keeps answering
 * after any line. These tests talk to the MCP server alone,
 * one raw line at a time; none needs the crabcrop binary.
 */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { createInterface } from "node:readline";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../..", import.meta.url));

/** The stdio server, written to and read from as raw lines. */
class LineClient {
  constructor(env = {}) {
    this.child = spawn(
      process.execPath,
      ["--import", `${root}sdks/mcp/test/register.mjs`, `${root}sdks/mcp/src/index.ts`],
      {
        env: { ...process.env, CRABCROP_BASE: "http://127.0.0.1:9", ...env },
        stdio: ["pipe", "pipe", "inherit"],
      },
    );
    this.lines = [];
    this.waiting = [];
    this.closed = false;
    const rl = createInterface({ input: this.child.stdout });
    rl.on("line", (line) => {
      const waiter = this.waiting.shift();
      if (waiter) waiter.resolve(line);
      else this.lines.push(line);
    });
    // A server that exits (the old one did, on a bare `null`)
    // fails the read in progress instead of leaving it waiting.
    rl.on("close", () => {
      this.closed = true;
      for (const waiter of this.waiting.splice(0)) {
        waiter.reject(new Error("the MCP server closed its output"));
      }
    });
    this.child.stdin.on("error", () => {});
    this.sentinel = 0;
  }

  send(text) {
    this.child.stdin.write(`${text}\n`);
  }

  next() {
    if (this.lines.length > 0) return Promise.resolve(this.lines.shift());
    if (this.closed) return Promise.reject(new Error("the MCP server closed its output"));
    return new Promise((resolve, reject) => this.waiting.push({ resolve, reject }));
  }

  /**
   * Send `text`, then a ping, and return every message the
   * server wrote before the ping's answer. The server answers
   * lines in order, so this is everything `text` got: an empty
   * list when it got no answer, without waiting on a timeout.
   */
  async exchange(text) {
    const id = `sentinel-${(this.sentinel += 1)}`;
    this.send(text);
    this.send(JSON.stringify({ jsonrpc: "2.0", id, method: "ping" }));
    const before = [];
    for (;;) {
      const message = JSON.parse(await this.next());
      if (!Array.isArray(message) && message?.id === id) {
        assert.deepEqual(message, { jsonrpc: "2.0", id, result: {} });
        return before;
      }
      before.push(message);
    }
  }

  /** The one answer `text` got. */
  async answer(text) {
    const answers = await this.exchange(text);
    assert.equal(answers.length, 1, `${text} got ${answers.length} answers: ${JSON.stringify(answers)}`);
    return answers[0];
  }

  close() {
    this.child.kill();
  }
}

const request = (id, method, params) =>
  JSON.stringify(params === undefined ? { jsonrpc: "2.0", id, method } : { jsonrpc: "2.0", id, method, params });

/** Assert `reply` is the JSON-RPC error `code` for request `id`. */
function assertError(reply, id, code, text) {
  assert.equal(reply.jsonrpc, "2.0", text);
  assert.equal(reply.id, id, text);
  assert.equal(reply.error?.code, code, `${text}: ${JSON.stringify(reply)}`);
  assert.equal(typeof reply.error.message, "string", text);
  assert.equal("result" in reply, false, text);
}

test("an unknown method gets -32601, and a notification gets no answer (issue #716)", async () => {
  // The old server answered an unknown method with an empty
  // result, as if it had succeeded.
  const client = new LineClient();
  try {
    for (const [id, method] of [
      [1, "resources/list"],
      ["two", "prompts/get"],
      [3, "tools/unknown"],
      [4, "notifications/initialized"],
    ]) {
      const reply = await client.answer(request(id, method));
      assertError(reply, id, -32601, method);
      assert.equal(reply.error.message, `Method not found: ${method}`);
    }
    for (const method of ["notifications/initialized", "notifications/cancelled", "no/such/method"]) {
      assert.deepEqual(await client.exchange(JSON.stringify({ jsonrpc: "2.0", method })), [], method);
    }
  } finally {
    client.close();
  }
});

test("a line that is not JSON gets -32700 with a null id (issue #716)", async () => {
  // The old server wrote nothing back, so the client waited
  // on a request it could not name.
  const client = new LineClient();
  try {
    for (const line of ["not json", "{", request(1, "ping").slice(0, -1), "\u0000", "{'jsonrpc': '2.0'}"]) {
      assertError(await client.answer(line), null, -32700, line);
    }
  } finally {
    client.close();
  }
});

test("JSON that is not a request object gets -32600, and the server keeps answering (issue #716)", async () => {
  // A bare `null` crashed the old server: reading its id threw
  // inside the request chain, and the unhandled rejection
  // ended the process. The id comes back when it is usable.
  const client = new LineClient();
  try {
    for (const [line, id] of [
      ["null", null],
      ["42", null],
      ['"ping"', null],
      ["true", null],
      ["{}", null],
      ['{"id":1,"method":"ping"}', 1],
      ['{"jsonrpc":"1.0","id":2,"method":"ping"}', 2],
      ['{"jsonrpc":"2.0","id":3}', 3],
      ['{"jsonrpc":"2.0","id":4,"method":7}', 4],
      ['{"jsonrpc":"2.0","id":null,"method":"ping"}', null],
      ['{"jsonrpc":"2.0","id":{"n":5},"method":"ping"}', null],
      ['{"jsonrpc":"2.0","id":true,"method":"ping"}', null],
      ['{"jsonrpc":"2.0","id":1e999,"method":"ping"}', null],
      ['{"jsonrpc":"2.0","id":6,"method":"ping","params":"x"}', 6],
      ['{"jsonrpc":"2.0","id":7,"method":"ping","params":null}', 7],
      ['{"jsonrpc":"2.0","method":"ping","params":8}', null],
    ]) {
      assertError(await client.answer(line), id, -32600, line);
    }
    // Structured params that are not an object are the
    // method's invalid params, not an invalid request.
    assertError(await client.answer(request(9, "ping", [])), 9, -32602, "array params");
    assert.equal(client.closed, false);
  } finally {
    client.close();
  }
});

test("initialize answers the client's protocol version when the server speaks it, else its newest (issue #716)", async () => {
  // The old server answered 2024-11-05 whatever the client
  // asked for.
  const supported = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
  const client = new LineClient();
  const initialize = (id, protocolVersion) =>
    request(id, "initialize", {
      protocolVersion,
      capabilities: {},
      clientInfo: { name: "protocol-test", version: "0" },
    });
  try {
    let id = 0;
    for (const version of supported) {
      const reply = await client.answer(initialize((id += 1), version));
      assert.equal(reply.id, id);
      assert.equal(reply.result.protocolVersion, version);
      assert.deepEqual(reply.result.capabilities, { tools: {} });
      assert.equal(reply.result.serverInfo.name, "crabcrop");
    }
    for (const version of ["2099-01-01", "2024-10-07", "1.0", ""]) {
      const reply = await client.answer(initialize((id += 1), version));
      assert.equal(reply.result.protocolVersion, supported[0], version);
    }
    for (const params of [{}, { protocolVersion: 20250618 }, { protocolVersion: null }]) {
      const reply = await client.answer(request((id += 1), "initialize", params));
      assertError(reply, id, -32602, JSON.stringify(params));
    }
    assertError(await client.answer(request((id += 1), "initialize")), id, -32602, "no params");
  } finally {
    client.close();
  }
});

test("a batch gets one array of answers, and a batch of notifications gets none (issue #716)", async () => {
  // 2025-03-26 requires a server to accept batches. The old
  // server read a batch as a notification and dropped it.
  const client = new LineClient();
  try {
    const reply = await client.answer(
      JSON.stringify([
        { jsonrpc: "2.0", id: 1, method: "ping" },
        { jsonrpc: "2.0", method: "notifications/initialized" },
        { jsonrpc: "2.0", id: 2, method: "resources/list" },
        null,
        [],
        { jsonrpc: "2.0", id: 3, method: "tools/list" },
      ]),
    );
    assert.ok(Array.isArray(reply), JSON.stringify(reply));
    assert.equal(reply.length, 5);
    assert.deepEqual(reply[0], { jsonrpc: "2.0", id: 1, result: {} });
    assertError(reply[1], 2, -32601, "batched unknown method");
    assertError(reply[2], null, -32600, "batched null");
    assertError(reply[3], null, -32600, "batched array");
    assert.equal(reply[4].id, 3);
    assert.equal(reply[4].result.tools.length, 4);

    // An empty batch is one invalid request, not an empty array.
    assertError(await client.answer("[]"), null, -32600, "empty batch");
    assert.deepEqual(
      await client.exchange(
        JSON.stringify([
          { jsonrpc: "2.0", method: "notifications/initialized" },
          { jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: 1 } },
        ]),
      ),
      [],
    );
  } finally {
    client.close();
  }
});

test("tools/call answers -32602 for an unknown tool or malformed params, and a tool error for a bad argument (issue #716)", async () => {
  // An unknown tool is a protocol error in the spec's tools
  // section; the old server answered it with an error result.
  // A bad argument value stays the tool's own error result,
  // which the model reads and can correct.
  const client = new LineClient();
  try {
    const unknown = await client.answer(request(1, "tools/call", { name: "nope", arguments: {} }));
    assertError(unknown, 1, -32602, "unknown tool");
    assert.equal(unknown.error.message, "Unknown tool: nope");
    for (const [id, params] of [
      [2, {}],
      [3, { name: 42 }],
      [4, { name: "sign_url", arguments: "operations=300x200" }],
      [5, { name: "sign_url", arguments: ["300x200"] }],
    ]) {
      assertError(await client.answer(request(id, "tools/call", params)), id, -32602, JSON.stringify(params));
    }
    const bad = await client.answer(request(6, "tools/call", { name: "sign_url", arguments: { source: "https://e.com/a.jpg" } }));
    assert.equal(bad.id, 6);
    assert.equal(bad.result.isError, true);
    assert.match(bad.result.content[0].text, /missing required argument: operations/);
  } finally {
    client.close();
  }
});

test("a response from the client gets no answer, and a tools/call without an id runs nothing (issue #716)", async () => {
  // This server sends no requests, so a response is
  // unsolicited; the old server answered it with a result. A
  // tools/call is a request, and without an id it is not one
  // this server runs: the old server ran it, purge and all,
  // and told nobody. The recorder counts what reaches it.
  const hits = [];
  const recorder = createServer((req, res) => {
    hits.push(`${req.method} ${req.url}`);
    res.writeHead(200, { "content-type": "text/plain" });
    res.end("purged 1");
  });
  await new Promise((resolve) => recorder.listen(0, "127.0.0.1", resolve));
  const client = new LineClient({
    CRABCROP_BASE: `http://127.0.0.1:${recorder.address().port}`,
    CRABCROP_SECRET: "mcp-test-secret-0123456",
  });
  const purge = { name: "purge_result", arguments: { path: "300x200/https://example.com/a.jpg" } };
  try {
    for (const line of [
      '{"jsonrpc":"2.0","id":1,"result":{}}',
      '{"jsonrpc":"2.0","id":"x","error":{"code":-32601,"message":"Method not found"}}',
      JSON.stringify({ jsonrpc: "2.0", method: "tools/call", params: purge }),
    ]) {
      assert.deepEqual(await client.exchange(line), [], line);
    }
    assert.deepEqual(hits, []);
    // The same call as a request does purge, so the silence
    // above is not a purge that could not have worked.
    const reply = await client.answer(request(2, "tools/call", purge));
    assert.deepEqual(reply.result.content, [{ type: "text", text: "purged 1" }]);
    assert.equal(hits.length, 1);
  } finally {
    client.close();
    recorder.close();
  }
});

test("the source header names the dependency the package bundles (issue #716)", () => {
  // The header said "No dependencies" while the package
  // depends on, and bundles, @crabcrop/url.
  const pkg = JSON.parse(readFileSync(`${root}sdks/mcp/package.json`, "utf8"));
  const source = readFileSync(`${root}sdks/mcp/src/index.ts`, "utf8");
  const header = /^#![^\n]*\n\/\*\*([\s\S]*?)\*\//.exec(source)?.[1];
  assert.ok(header, "src/index.ts opens with its doc comment");
  const dependencies = Object.keys(pkg.dependencies ?? {});
  assert.ok(dependencies.length > 0);
  assert.doesNotMatch(header, /\bno dependencies\b/i);
  for (const name of dependencies) {
    assert.ok(header.includes(name), `the header does not name ${name}`);
    assert.ok(pkg.bundleDependencies.includes(name), `${name} is not bundled, as the header says`);
  }
  assert.match(header, /bundle/);
});
