#!/usr/bin/env node
/**
 * A first-party MCP server for crabcrop: four tools over
 * stdio, thin wrappers on the TypeScript SDK and the HTTP
 * surface. No dependencies.
 *
 *   sign_url     build and sign a crabcrop URL
 *   explain_url  parse a URL back into its segments, flagging errors
 *   fetch_meta   the server's JSON meta for a URL
 *   purge_result evict a cached result
 *
 * Configuration: CRABCROP_BASE and CRABCROP_SECRET in the
 * environment are the defaults for sign_url and purge_result,
 * so agents do not pass the secret on every call. Per-call
 * secret arguments exist as an override but are deprecated:
 * agentic clients persist tool arguments in transcripts, so
 * the environment is the right home for the secret.
 *
 * Run: node src/index.ts   (Node >= 22, for type stripping)
 */
import { buildURL, explainUrl, purgeSignature, sign } from "@crabcrop/url";

const SERVER = { name: "crabcrop", version: "0.8.3" };
const PROTOCOL = "2024-11-05";

const envBase = process.env.CRABCROP_BASE ?? "";
const envSecret = process.env.CRABCROP_SECRET ?? null;

function textResult(text: string, isError = false) {
  return { content: [{ type: "text", text }], ...(isError ? { isError: true } : {}) };
}

/**
 * Tool arguments arrive as untyped JSON. Validate each one
 * instead of coercing with String(...): a missing argument
 * would otherwise be signed into a URL as the literal
 * segment "undefined", and numbers and objects would
 * stringify silently.
 */
function stringArg(args: Record<string, unknown>, name: string): string {
  const value = args[name];
  if (typeof value !== "string") {
    throw new Error(
      value === undefined
        ? `missing required argument: ${name}`
        : `argument ${name} must be a string, got ${typeof value}`,
    );
  }
  return value;
}

function optionalStringArg(args: Record<string, unknown>, name: string): string | undefined {
  const value = args[name];
  if (value === undefined) return undefined;
  if (typeof value !== "string") {
    throw new Error(`argument ${name} must be a string, got ${typeof value}`);
  }
  return value;
}

function numberArg(args: Record<string, unknown>, name: string): number | undefined {
  const value = args[name];
  if (value === undefined) return undefined;
  if (typeof value !== "number") {
    throw new Error(`argument ${name} must be a number, got ${typeof value}`);
  }
  return value;
}

function boolArg(args: Record<string, unknown>, name: string): boolean | undefined {
  const value = args[name];
  if (value === undefined) return undefined;
  if (typeof value !== "boolean") {
    throw new Error(`argument ${name} must be a boolean, got ${typeof value}`);
  }
  return value;
}

/** Read a response body with a hard byte cap; null when exceeded. */
async function readCapped(resp: Response, cap: number): Promise<string | null> {
  if (!resp.body) return "";
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > cap) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
  }
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(merged);
}

/**
 * Whether any path component of `path` is a dot segment.
 * Mirrors has_dot_segment in the Rust loader: a component
 * exactly `.` or `..`, with dots inside a name (`a.b`,
 * `c..d`) counting as ordinary content. Because the string
 * is later parsed as a WHATWG URL, the percent-encoded
 * spellings (`%2e`, `.%2e`, `%2e.`, `%2e%2e`) count too, and
 * backslashes are separators for http(s) URLs, so components
 * are split on both.
 */
function hasDotSegment(path: string): boolean {
  return path.split(/[/\\]/).some((segment) => {
    const decoded = segment.toLowerCase().replaceAll("%2e", ".");
    return decoded === "." || decoded === "..";
  });
}

const TOOLS = [
  {
    name: "sign_url",
    description:
      "Build and sign a crabcrop URL from an operation string and a source. " +
      "Returns the full URL. Uses CRABCROP_BASE / CRABCROP_SECRET from the " +
      "environment unless base/secret arguments override them; with no secret " +
      "the URL carries the literal `unsafe` signature segment. Prefer the environment for the secret: per-call secret arguments are deprecated.",
    inputSchema: {
      type: "object",
      properties: {
        operations: {
          type: "string",
          description: "Operation string, e.g. \"crop:1200x630:smart/filters:format(webp)\"; may be empty and may contain preset: references.",
        },
        source: {
          type: "string",
          description: "Source locator: an http(s) URL, s3://bucket/key, gs://bucket/key, or file:// path.",
        },
        base: { type: "string", description: "crabcrop endpoint origin. Defaults to CRABCROP_BASE." },
        secret: { type: "string", description: "Signing secret. Defaults to CRABCROP_SECRET; omit for unsafe URLs. Sensitive, deprecated: prefer the CRABCROP_SECRET environment variable; a per-call secret is kept in client transcripts." },
        expires: { type: "number", description: "Optional unix-seconds expiry, emitted as an exp: segment." },
        keyId: { type: "string", description: "Optional key ID stamped on the signature segment (`<key-id>:<tag>`) for servers running a secret rotation; they verify a keyed segment against the named key only." },
      },
      required: ["operations", "source"],
      additionalProperties: false,
    },
    async run(args: Record<string, unknown>) {
      const base = optionalStringArg(args, "base") ?? envBase;
      if (!base) throw new Error("no base: pass the base argument or set CRABCROP_BASE");
      const secret = optionalStringArg(args, "secret") ?? envSecret;
      const url = buildURL(base, secret, stringArg(args, "operations"), stringArg(args, "source"), {
        expires: numberArg(args, "expires"),
        keyId: optionalStringArg(args, "keyId"),
      });
      return textResult(url);
    },
  },
  {
    name: "explain_url",
    description:
      "Parse a crabcrop URL back into its pieces: the signature kind, each " +
      "operation segment with a plain-language explanation, exp: rendered as " +
      "a UTC instant with an expired verdict, and the source. Malformed " +
      "segments and unknown filters are reported as errors.",
    inputSchema: {
      type: "object",
      properties: {
        url: { type: "string", description: "The full crabcrop URL or bare path." },
      },
      required: ["url"],
      additionalProperties: false,
    },
    async run(args: Record<string, unknown>) {
      const out = explainUrl(stringArg(args, "url"));
      const lines: string[] = [];
      if (out.signature) {
        lines.push(`signature (${out.signature.kind}): ${out.signature.value}`);
        lines.push(`  ${out.signature.detail}`);
      }
      for (const seg of out.segments) {
        lines.push(`${seg.kind}: ${seg.raw}`);
        lines.push(`  ${seg.error ? `ERROR: ${seg.error}` : seg.detail ?? ""}`);
      }
      if (out.source) {
        lines.push(`source: ${out.source}`);
      }
      for (const error of out.errors) {
        lines.push(`ERROR: ${error}`);
      }
      return textResult(lines.join("\n"), out.errors.length > 0);
    },
  },
  {
    name: "fetch_meta",
    description:
      "Fetch the crabcrop meta JSON for a URL (the processed result's width, " +
      "height, format, byte size, quality, and placeholder data on request). " +
      "The URL must already be signed if the server requires signatures. " +
      "For safety the URL's origin must match CRABCROP_BASE: this tool will " +
      "not fetch arbitrary URLs.",
    inputSchema: {
      type: "object",
      properties: {
        url: { type: "string", description: "The full crabcrop URL." },
        placeholders: { type: "boolean", description: "Also request placeholder data (default false)." },
      },
      required: ["url"],
      additionalProperties: false,
    },
    async run(args: Record<string, unknown>) {
      if (!envBase) {
        return textResult(
          "fetch_meta requires CRABCROP_BASE to be set: it only fetches URLs on the configured server's origin",
          true,
        );
      }
      const urlArg = stringArg(args, "url");
      let target: URL;
      try {
        target = new URL(urlArg);
      } catch {
        return textResult(`fetch_meta: not a valid URL: ${urlArg}`, true);
      }
      if (target.origin !== new URL(envBase).origin) {
        return textResult(
          `fetch_meta: refusing to fetch ${target.origin}; only the configured server origin (${new URL(envBase).origin}) is allowed`,
          true,
        );
      }
      // The system endpoints are not meta targets (issue
      // #271): this tool's contract is the meta JSON of an
      // image URL, and /metrics is token-gated on the
      // server besides, so the tool is not a route around
      // the gate.
      if (target.pathname === "/metrics" || target.pathname === "/health") {
        return textResult(
          `fetch_meta: refusing to fetch the system endpoint ${target.pathname}; this tool reads meta for image URLs`,
          true,
        );
      }
      const accept = boolArg(args, "placeholders")
        ? "application/json; placeholders=1"
        : "application/json";
      let resp: Response;
      try {
        resp = await fetch(target, {
          headers: { accept },
          redirect: "error",
          signal: AbortSignal.timeout(10_000),
        });
      } catch (err) {
        return textResult(
          `meta request failed: ${err instanceof Error ? err.message : String(err)}`,
          true,
        );
      }
      const body = await readCapped(resp, 256 * 1024);
      if (body === null) {
        return textResult("meta request failed: response exceeded the 256 KiB cap", true);
      }
      if (!resp.ok) {
        return textResult(`meta request failed: HTTP ${resp.status}: ${body}`, true);
      }
      return textResult(body);
    },
  },
  {
    name: "purge_result",
    description:
      "Evict a cached result from a crabcrop server. The path is the cached " +
      "URL's remainder after its signature (operations and source; a preset " +
      "URL works as-is, the server expands it). Requires the signing secret (argument or " +
      "CRABCROP_SECRET). Returns how many entries were removed. " +
      "For safety the request only goes to the configured CRABCROP_BASE " +
      "origin (a base argument must match it) and paths with dot segments " +
      "are refused: this tool will not POST to arbitrary endpoints.",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "Remainder of the cached URL, e.g. \"300x200/https://example.com/a.jpg\"." },
        base: { type: "string", description: "crabcrop endpoint origin. Defaults to CRABCROP_BASE; its origin must match CRABCROP_BASE." },
        secret: { type: "string", description: "Signing secret. Defaults to CRABCROP_SECRET. Sensitive, deprecated: prefer the CRABCROP_SECRET environment variable; a per-call secret is kept in client transcripts." },
      },
      required: ["path"],
      additionalProperties: false,
    },
    async run(args: Record<string, unknown>) {
      // The fetch_meta treatment (audit PL-1, extended by
      // R2-10): the POST may only ever go to the configured
      // server's origin. Without the check, a caller-chosen
      // base made the tool an SSRF oracle: it POSTed to
      // internal endpoints and read the responses back, and
      // dot segments in the path resolved out of the
      // /purge/ prefix onto arbitrary paths on that host.
      // Refusing before any fetch keeps every disallowed or
      // unreachable target indistinguishable: no request is
      // sent, so there is no timing, status, or body signal
      // to tell them apart.
      if (!envBase) {
        return textResult(
          "purge_result requires CRABCROP_BASE to be set: it only purges on the configured server's origin",
          true,
        );
      }
      const configuredOrigin = new URL(envBase).origin;
      const base = (optionalStringArg(args, "base") ?? envBase).replace(/\/+$/, "");
      let baseOrigin: string;
      try {
        baseOrigin = new URL(base).origin;
      } catch {
        return textResult(`purge_result: not a valid base URL: ${base}`, true);
      }
      if (baseOrigin !== configuredOrigin) {
        return textResult(
          `purge_result: refusing to purge ${baseOrigin}; only the configured server origin (${configuredOrigin}) is allowed`,
          true,
        );
      }
      const secret = optionalStringArg(args, "secret") ?? envSecret;
      if (!secret) throw new Error("purge requires a secret: pass the secret argument or set CRABCROP_SECRET");
      const target = stringArg(args, "path").replace(/^\//, "");
      if (hasDotSegment(target)) {
        return textResult(
          "purge_result: refusing a path with dot segments: they would resolve out of the purge prefix",
          true,
        );
      }
      // The SDK's helper: purge signatures are
      // domain-separated from image signatures (audit P-1),
      // and the derivation lives in exactly one place.
      const signature = purgeSignature(secret, target);
      // The same network hygiene as fetch_meta: no
      // redirects, a bounded wait, and a capped body, so a
      // hung or hostile endpoint cannot hang the tool call
      // or stream an unbounded response.
      let resp: Response;
      try {
        resp = await fetch(`${base}/purge/${signature}/${target}`, {
          method: "POST",
          redirect: "error",
          signal: AbortSignal.timeout(10_000),
        });
      } catch (err) {
        return textResult(
          `purge request failed: ${err instanceof Error ? err.message : String(err)}`,
          true,
        );
      }
      const body = await readCapped(resp, 256 * 1024);
      if (body === null) {
        return textResult("purge failed: response exceeded the 256 KiB cap", true);
      }
      if (!resp.ok) {
        return textResult(`purge failed: HTTP ${resp.status}: ${body.trim()}`, true);
      }
      return textResult(body.trim());
    },
  },
];

interface JsonRpcRequest {
  jsonrpc: string;
  id?: string | number | null;
  method: string;
  params?: Record<string, unknown>;
}

async function handle(req: JsonRpcRequest): Promise<unknown> {
  switch (req.method) {
    case "initialize":
      return {
        protocolVersion: PROTOCOL,
        capabilities: { tools: {} },
        serverInfo: SERVER,
      };
    case "ping":
      return {};
    case "tools/list":
      return {
        tools: TOOLS.map(({ name, description, inputSchema }) => ({
          name,
          description,
          inputSchema,
        })),
      };
    case "tools/call": {
      const params = req.params ?? {};
      const tool = TOOLS.find((t) => t.name === params.name);
      if (!tool) {
        return textResult(`unknown tool: ${String(params.name)}`, true);
      }
      try {
        return await tool.run((params.arguments as Record<string, unknown>) ?? {});
      } catch (err) {
        return textResult(`error: ${err instanceof Error ? err.message : String(err)}`, true);
      }
    }
    default:
      return undefined; // Notifications and unknown methods: no reply.
  }
}

// Transport: newline-delimited JSON-RPC over stdio. Lines
// are capped (readline would buffer an unbounded line in
// memory) and requests are handled one at a time through a
// promise chain, so side-effecting calls cannot race
// (audit PL-9).
const MAX_LINE_BYTES = 1024 * 1024;
let chain: Promise<void> = Promise.resolve();
let buffered = "";
let discarding = false;

function enqueue(line: string): void {
  chain = chain.then(async () => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let req: JsonRpcRequest;
    try {
      req = JSON.parse(trimmed) as JsonRpcRequest;
    } catch {
      return;
    }
    if (req.id === undefined || req.id === null) {
      await handle(req); // A notification: handle silently.
      return;
    }
    const result = await handle(req);
    process.stdout.write(
      `${JSON.stringify({ jsonrpc: "2.0", id: req.id, result: result ?? {} })}\n`,
    );
  });
}

process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk: string) => {
  buffered += chunk;
  for (;;) {
    const nl = buffered.indexOf("\n");
    if (discarding) {
      if (nl === -1) {
        buffered = "";
        return;
      }
      buffered = buffered.slice(nl + 1);
      discarding = false;
      continue;
    }
    if (nl === -1) {
      if (buffered.length > MAX_LINE_BYTES) {
        // An over-long line: drop it and everything up to
        // its newline, and answer with a protocol error.
        buffered = "";
        discarding = true;
        process.stdout.write(
          `${JSON.stringify({ jsonrpc: "2.0", id: null, error: { code: -32600, message: "request line too large" } })}\n`,
        );
      }
      return;
    }
    const line = buffered.slice(0, nl);
    buffered = buffered.slice(nl + 1);
    enqueue(line);
  }
});
