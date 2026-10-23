# @crabcrop/mcp

A first-party MCP (Model Context Protocol) server for crabcrop, so coding agents can sign URLs, debug them, inspect results, and purge the cache without hand-rolling the scheme. It is a thin wrapper over [`@crabcrop/url`](../typescript) and the server's HTTP surface; the explainer is the same module the [playground](../../playground) ships, so the tools cannot drift from the server or the spec.

## Tools

| Tool | What it does |
|---|---|
| `sign_url` | Build and sign a URL from an operation string and a source (optional expiry). |
| `explain_url` | Parse a URL back into its signature kind, explained segments, and source; malformed segments are flagged. |
| `fetch_meta` | Fetch the server's meta JSON for a URL (dimensions, format, bytes, quality, placeholders on request). |
| `purge_result` | Evict a cached result by its URL remainder (the expanded form for preset URLs). |

## Setup

The server runs over stdio with Node >= 22 (it is TypeScript, run directly). Point your MCP client at it:

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

`CRABCROP_BASE` and `CRABCROP_SECRET` are defaults: `sign_url` and `purge_result` accept per-call `base`/`secret` overrides, and with no secret configured `sign_url` produces `unsafe` URLs for servers that allow them. `purge_result` always requires a secret.

In this repository (no install step), `@crabcrop/url` resolves through the SDK sources; in a standalone install, `npm install @crabcrop/url` provides it. The end-to-end test (`npm test`) spawns this server plus a real crabcrop binary and runs sign -> fetch -> meta -> purge against it; it needs `cargo build` to have produced `target/debug/crabcrop` and uses the bench fixture photo as its source.
