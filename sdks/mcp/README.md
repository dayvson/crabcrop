# @crabcrop/mcp

A first-party MCP (Model Context Protocol) server for crabcrop, so coding agents can sign URLs, debug them, inspect results, and purge the cache without hand-rolling the scheme. It is a thin wrapper over [`@crabcrop/url`](../typescript) and the server's HTTP surface. The explainer comes from `@crabcrop/url`, as does the [playground](../../playground)'s, so every JavaScript surface reads one implementation of the grammar, pinned by the shared grammar corpora in `spec/`, while signing everywhere runs the canonical `spec/test-vectors.json`.

## Tools

| Tool | What it does |
|---|---|
| `sign_url` | Build and sign a URL from an operation string and a source (optional expiry). |
| `explain_url` | Parse a URL back into its signature kind, explained segments, and source; malformed segments are flagged. |
| `fetch_meta` | Fetch the server's meta JSON for a URL (dimensions, format, bytes, quality, placeholders on request). The server needs meta mode on (`CRABCROP_META_MODE=1`); it is off by default, and the tool then reports the server's `403 meta_disabled`. Only fetches URLs on the configured `CRABCROP_BASE` origin, never follows redirects, and caps the response at 256 KiB. |
| `purge_result` | Evict a cached result by its URL remainder; a preset URL works as-is (the server expands it before matching). Only POSTs to `/purge/` under the configured `CRABCROP_BASE` (a per-call `base` must equal it) and refuses paths with dot segments. |

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

From a checkout, it also runs directly from source on Node 22.18 or newer, which type-strips the TypeScript. Build the SDK and install the package's link to it once first:

```bash
cd /path/to/crabcrop/sdks/typescript && npm ci && npm run build
cd ../mcp && npm ci
```

Then point the client at the source:

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

`CRABCROP_BASE` and `CRABCROP_SECRET` are the defaults, and the environment is the recommended home for the secret: `sign_url` still accepts per-call `base`/`secret` overrides and `purge_result` a per-call `secret`, but the per-call secret is deprecated because agentic clients persist tool arguments in their transcripts. With no secret configured, `sign_url` produces `unsafe` URLs for servers that allow them. `purge_result` always requires a secret, and it only ever sends its POST to `/purge/` under the configured `CRABCROP_BASE`, which may carry a path for a server mounted under a prefix (`https://example.com/img` purges at `https://example.com/img/purge/...`). The request URL is built from `CRABCROP_BASE` alone: a per-call `base` is only checked, and must equal `CRABCROP_BASE` (trailing slashes aside), so a path, query, or fragment in it cannot move the POST elsewhere on the host. `CRABCROP_BASE` itself must be an http(s) URL with no userinfo, query, or fragment; with it unset or malformed the tool refuses outright. Paths carrying dot segments are refused, and a `?` or `#` in a path is sent percent-encoded inside the path. This is the origin discipline `fetch_meta` follows, held to the configured path as well.

Both this package and `@crabcrop/url` ship compiled `dist/` builds (their `prepack` runs `tsc`). The published tarball is self-contained: the declared `@crabcrop/url` dependency is a `file:` path, which no registry install could resolve, so the package bundles it (`bundleDependencies`); the tarball carries the SDK's published form under its own `node_modules`, and installing the package is all a consumer does. In this repository's tests (no install step), a register hook maps `@crabcrop/url` to the SDK sources, and the pack test does the real thing instead of simulating it: `npm pack`, `npm install` the tarball into a fresh project, and a JSON-RPC round trip against the installed bin. The end-to-end test (`npm test`) spawns this server plus a real crabcrop binary and runs sign -> fetch -> meta -> purge against it; it needs `cargo build` to have produced `target/debug/crabcrop` and uses the bench fixture photo as its source.

`server.json` in this directory is the MCP registry manifest (name `io.github.crabcrop/crabcrop`, matching the `mcpName` field in `package.json`); its versions are bumped by the same one-version tooling as every other artifact. The registry submission itself is a separate step and has not been made yet.
