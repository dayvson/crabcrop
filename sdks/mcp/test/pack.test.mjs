import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const mcpDir = join(dirname(fileURLToPath(import.meta.url)), "..");
const sdkDir = join(mcpDir, "..", "typescript");

test("the packed MCP server runs outside the repo layout (audit PL-10)", async () => {
  // Build the SDK first: a registry install of @crabcrop/url
  // carries its compiled dist, and the MCP's own prepack build
  // type-resolves against it.
  execFileSync("npm", ["run", "--silent", "build"], { cwd: sdkDir });
  const work = mkdtempSync(join(tmpdir(), "crabcrop-pack-"));
  execFileSync("npm", ["pack", "--silent", mcpDir], { cwd: work });
  const tarball = readdirSync(work).find((f) => f.endsWith(".tgz"));
  assert.ok(tarball, "npm pack produced a tarball");
  execFileSync("tar", ["-xzf", join(work, tarball), "-C", work]);
  const pkgDir = join(work, "package");
  // Simulate a registry install's node_modules: the SDK in its
  // published form (package.json + dist), and nothing else from
  // the repo around it. In particular there is no
  // ../../../playground to import from, and Node will not
  // type-strip sources under node_modules, so only the
  // compiled layout can work here.
  const nm = join(pkgDir, "node_modules", "@crabcrop", "url");
  mkdirSync(nm, { recursive: true });
  cpSync(join(sdkDir, "package.json"), join(nm, "package.json"));
  cpSync(join(sdkDir, "dist"), join(nm, "dist"), { recursive: true });

  const child = spawn("node", [join(pkgDir, "dist", "index.js")], {
    env: {
      ...process.env,
      CRABCROP_SECRET: "pack-smoke-secret-0123",
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
        () => reject(new Error("packed server did not answer in 15s")),
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
        reject(new Error(`packed server exited early (${code})`));
      });
      child.stdin.write(JSON.stringify(request) + "\n");
    });
    const parsed = JSON.parse(line);
    assert.match(parsed.result.content[0].text, /300px width/);
  } finally {
    child.kill();
  }
});
