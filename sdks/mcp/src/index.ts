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
 * arguments override them.
 *
 * Run: node src/index.ts   (Node >= 22, for type stripping)
 */
import { createHmac } from "node:crypto";
import { createInterface } from "node:readline";
import { buildURL, sign } from "@crabcrop/url";
// The explainer is the playground's module: one implementation
// of URL explanation, shared by the debugger and this server.
import { explainUrl } from "../../../playground/lib.mjs";

const SERVER = { name: "crabcrop", version: "0.1.0" };
const PROTOCOL = "2024-11-05";

const envBase = process.env.CRABCROP_BASE ?? "";
const envSecret = process.env.CRABCROP_SECRET ?? null;

function textResult(text: string, isError = false) {
  return { content: [{ type: "text", text }], ...(isError ? { isError: true } : {}) };
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

const TOOLS = [
  {
    name: "sign_url",
    description:
      "Build and sign a crabcrop URL from an operation string and a source. " +
      "Returns the full URL. Uses CRABCROP_BASE / CRABCROP_SECRET from the " +
      "environment unless base/secret arguments override them; with no secret " +
      "the URL carries the literal `unsafe` signature segment.",
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
        secret: { type: "string", description: "Signing secret. Defaults to CRABCROP_SECRET; omit for unsafe URLs." },
        expires: { type: "number", description: "Optional unix-seconds expiry, emitted as an exp: segment." },
      },
      required: ["operations", "source"],
    },
    async run(args: Record<string, unknown>) {
      const base = String(args.base ?? envBase);
      if (!base) throw new Error("no base: pass the base argument or set CRABCROP_BASE");
      const secret = (args.secret as string | undefined) ?? envSecret;
      const url = buildURL(
        base,
        secret,
        String(args.operations),
        String(args.source),
        { expires: args.expires as number | undefined },
      );
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
    },
    async run(args: Record<string, unknown>) {
      const out = explainUrl(String(args.url));
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
    },
    async run(args: Record<string, unknown>) {
      if (!envBase) {
        return textResult(
          "fetch_meta requires CRABCROP_BASE to be set: it only fetches URLs on the configured server's origin",
          true,
        );
      }
      let target: URL;
      try {
        target = new URL(String(args.url));
      } catch {
        return textResult(`fetch_meta: not a valid URL: ${String(args.url)}`, true);
      }
      if (target.origin !== new URL(envBase).origin) {
        return textResult(
          `fetch_meta: refusing to fetch ${target.origin}; only the configured server origin (${new URL(envBase).origin}) is allowed`,
          true,
        );
      }
      const accept = args.placeholders
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
      "URL's remainder after its signature (operations and source, in expanded " +
      "form for preset URLs). Requires the signing secret (argument or " +
      "CRABCROP_SECRET). Returns how many entries were removed.",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "Remainder of the cached URL, e.g. \"300x200/https://example.com/a.jpg\"." },
        base: { type: "string", description: "crabcrop endpoint origin. Defaults to CRABCROP_BASE." },
        secret: { type: "string", description: "Signing secret. Defaults to CRABCROP_SECRET." },
      },
      required: ["path"],
    },
    async run(args: Record<string, unknown>) {
      const base = String(args.base ?? envBase).replace(/\/+$/, "");
      const secret = (args.secret as string | undefined) ?? envSecret;
      if (!base) throw new Error("no base: pass the base argument or set CRABCROP_BASE");
      if (!secret) throw new Error("purge requires a secret: pass the secret argument or set CRABCROP_SECRET");
      const target = String(args.path).replace(/^\//, "");
      // Purge signatures are domain-separated (audit P-1): the
      // key is the raw HMAC of the secret over a fixed label,
      // and the signature is the HMAC of the target under it.
      const subkey = Buffer.from(sign(secret, "crabcrop/purge-key"), "hex");
      const signature = createHmac("sha256", subkey).update(target, "utf8").digest("hex");
      const resp = await fetch(`${base}/purge/${signature}/${target}`, { method: "POST" });
      const body = (await resp.text()).trim();
      if (!resp.ok) {
        return textResult(`purge failed: HTTP ${resp.status}: ${body}`, true);
      }
      return textResult(body);
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

const rl = createInterface({ input: process.stdin, terminal: false });
rl.on("line", async (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let req: JsonRpcRequest;
  try {
    req = JSON.parse(trimmed) as JsonRpcRequest;
  } catch {
    return;
  }
  if (req.id === undefined || req.id === null) {
    void handle(req); // A notification: handle silently.
    return;
  }
  const result = await handle(req);
  process.stdout.write(
    `${JSON.stringify({ jsonrpc: "2.0", id: req.id, result: result ?? {} })}\n`,
  );
});
