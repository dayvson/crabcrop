# @crabcrop/mcp

A first-party MCP (Model Context Protocol) server for crabcrop, so coding agents can sign URLs, debug them, inspect results, and purge the cache without hand-rolling the scheme. It is a thin wrapper over [`@crabcrop/url`](../typescript) and the server's HTTP surface. The explainer comes from `@crabcrop/url`; the [playground](../../playground) ships its own copy of the same grammar, and both are pinned by a mirrored grammar corpus in their test suites, while signing everywhere runs the canonical `spec/test-vectors.json`.

## Tools

| Tool | What it does |
|---|---|
| `sign_url` | Build and sign a URL from an operation string and a source (optional expiry). |
| `explain_url` | Parse a URL back into its signature kind, explained segments, and source; malformed segments are flagged. |
| `fetch_meta` | Fetch the server's meta JSON for a URL (dimensions, format, bytes, quality, placeholders on request). Only fetches URLs on the configured `CRABCROP_BASE` origin, never follows redirects, and caps the response at 256 KiB. |
| `purge_result` | Evict a cached result by its URL remainder; a preset URL works as-is (the server expands it before matching). Only purges on the configured `CRABCROP_BASE` origin (a per-call `base` must match it) and refuses paths with dot segments. |

## Setup

The server runs over stdio with Node >= 22. Once published, point your MCP client at the installed package:

```json
{
  "mcpServers": {
    "crabcrop": {
      "command": "npx",
      "args": ["-y", "@crabcrop/mcp"],
      "env": {
        "CRABCROP_BASE": "https://images.example.com",
        "CRABCROP_SECRET": "your-secret"
      }
    }
  }
}
```

From a checkout, it also runs directly from source (Node type-strips the TypeScript):

```json
{
  "mcpServers": {
    "crabcrop": {
      "command": "node",
      "args": ["/path/to/crabcrop/sdks/mcp/src/index.ts"],
      "env": {
        "CRABCROP_BASE": "https://images.example.com",
        "CRABCROP_SECRET": "your-secret"
      }
    }
  }
}
```

For Claude Code:

```bash
claude mcp add crabcrop \
  --env CRABCROP_BASE=https://images.example.com \
  --env CRABCROP_SECRET=your-secret \
  -- node /path/to/crabcrop/sdks/mcp/src/index.ts
```

`CRABCROP_BASE` and `CRABCROP_SECRET` are the defaults, and the environment is the recommended home for the secret: `sign_url` and `purge_result` still accept per-call `base`/`secret` overrides, but the per-call secret is deprecated because agentic clients persist tool arguments in their transcripts. With no secret configured, `sign_url` produces `unsafe` URLs for servers that allow them. `purge_result` always requires a secret, and it only ever sends its POST to the configured `CRABCROP_BASE` origin: a per-call `base` must match that origin, with no `CRABCROP_BASE` set the tool refuses outright, and paths carrying dot segments are refused, the same origin discipline `fetch_meta` follows.

Both this package and `@crabcrop/url` ship compiled `dist/` builds (their `prepack` runs `tsc`). The published tarball is self-contained: the declared `@crabcrop/url` dependency is a `file:` path, which no registry install could resolve, so the package bundles it (`bundleDependencies`); the tarball carries the SDK's published form under its own `node_modules`, and installing the package is all a consumer does. In this repository's tests (no install step), a register hook maps `@crabcrop/url` to the SDK sources, and the pack test does the real thing instead of simulating it: `npm pack`, `npm install` the tarball into a fresh project, and a JSON-RPC round trip against the installed bin. The end-to-end test (`npm test`) spawns this server plus a real crabcrop binary and runs sign -> fetch -> meta -> purge against it; it needs `cargo build` to have produced `target/debug/crabcrop` and uses the bench fixture photo as its source.

`server.json` in this directory is the MCP registry manifest (name `io.github.crabcrop/crabcrop`, matching the `mcpName` field in `package.json`); its versions are bumped by the same one-version tooling as every other artifact. The registry submission itself is a separate step and has not been made yet.
