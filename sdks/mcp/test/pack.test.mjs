import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { mkdtempSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const mcpDir = join(dirname(fileURLToPath(import.meta.url)), "..");

test("the packed MCP server installs and runs outside the repo (audit PL-10, issue #154)", async () => {
  // Pack exactly what would be published. The package's
  // prepack builds the SDK and then the server, and
  // bundleDependencies puts the SDK's published form
  // (package.json + dist) inside the tarball: the declared
  // file: dependency can never resolve from a registry, so
  // the tarball has to be self-contained (AP-27).
  const packDir = mkdtempSync(join(tmpdir(), "crabcrop-pack-"));
  execFileSync("npm", ["pack", "--silent", mcpDir], { cwd: packDir });
  const tarball = readdirSync(packDir).find((f) => f.endsWith(".tgz"));
  assert.ok(tarball, "npm pack produced a tarball");

  // Install the tarball into a fresh project with npm, the
  // way a registry consumer would. Nothing else is
  // available to resolve: if the bundled SDK were missing
  // or the layout wrong, the install or the run below
  // would fail. No hand-simulated node_modules.
  const consumer = mkdtempSync(join(tmpdir(), "crabcrop-consumer-"));
  writeFileSync(
    join(consumer, "package.json"),
    JSON.stringify({ name: "consumer", version: "1.0.0", type: "module" }),
  );
  execFileSync(
    "npm",
    ["install", "--no-audit", "--no-fund", join(packDir, tarball)],
    { cwd: consumer, stdio: "pipe" },
  );
  const bin = join(consumer, "node_modules", ".bin", "crabcrop-mcp");

  const child = spawn(bin, [], {
    env: {
      ...process.env,
      CRABCROP_SECRET: "playground-secret",
      CRABCROP_BASE: "http://127.0.0.1:9",
    },
    stdio: ["pipe", "pipe", "inherit"],
  });
  try {
    const request = {
      jsonrpc: "2.0",
      id: 1,
      method: "tools/call",
      params: {
        name: "explain_url",
        arguments: {
          url: "https://img.example.com/unsafe/300x200/https://example.com/a.jpg",
        },
      },
    };
    const line = await new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error("installed server did not answer in 15s")),
        15000,
      );
      let buf = "";
      child.stdout.on("data", (d) => {
        buf += d;
        const i = buf.indexOf("\n");
        if (i >= 0) {
          clearTimeout(timer);
          resolve(buf.slice(0, i));
        }
      });
      child.on("error", (err) => {
        clearTimeout(timer);
        reject(err);
      });
      child.on("exit", (code) => {
        clearTimeout(timer);
        reject(new Error(`installed server exited early (${code})`));
      });
      child.stdin.write(JSON.stringify(request) + "\n");
    });
    const parsed = JSON.parse(line);
    assert.match(parsed.result.content[0].text, /300px width/);
  } finally {
    child.kill();
  }
});
