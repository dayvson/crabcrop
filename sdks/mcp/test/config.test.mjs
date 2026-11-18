/**
 * The MCP server and the TypeScript SDK it bundles build with
 * one compiler and one set of checks (issue #716). The server
 * had `strict: false` and TypeScript 5.9 while the SDK had
 * `strict: true` and TypeScript 7, and nothing caught the
 * unused import that hid behind the looser settings. The pack
 * test builds both packages through `prepack`, so a type error
 * in either fails it.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const sdks = fileURLToPath(new URL("../..", import.meta.url));
const read = (relative) => JSON.parse(readFileSync(`${sdks}${relative}`, "utf8"));

test("the MCP package uses the SDK's TypeScript, locked to the same release (issue #716)", () => {
  const sdk = read("typescript/package.json");
  const mcp = read("mcp/package.json");
  assert.equal(mcp.devDependencies.typescript, sdk.devDependencies.typescript);
  const locked = (relative) => read(relative).packages["node_modules/typescript"].version;
  assert.equal(locked("mcp/package-lock.json"), locked("typescript/package-lock.json"));
});

test("the MCP package compiles with the SDK's checks (issue #716)", () => {
  const sdk = read("typescript/tsconfig.json").compilerOptions;
  const mcp = read("mcp/tsconfig.json").compilerOptions;
  // The options that decide what compiles and how it runs.
  // `declaration` is the SDK's alone: it ships types, and the
  // server ships a bin.
  for (const option of [
    "module",
    "moduleResolution",
    "noUnusedLocals",
    "rewriteRelativeImportExtensions",
    "skipLibCheck",
    "strict",
    "target",
    "types",
  ]) {
    assert.deepEqual(mcp[option], sdk[option], option);
  }
  assert.equal(mcp.strict, true);
  assert.equal(mcp.noUnusedLocals, true);
});
