#!/usr/bin/env python3
"""Minimal stdio JSON-RPC client for the repo MCP server.

The server entry point is CRABCROP_MCP_ENTRY when set, else the repo's
built sdks/mcp/dist/index.js.
"""
import json, os, subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
MCP = os.environ.get(
    "CRABCROP_MCP_ENTRY",
    os.path.join(_HERE, "..", "..", "sdks", "mcp", "dist", "index.js"))


class Mcp:
    def __init__(self, base, secret):
        env = dict(os.environ, CRABCROP_BASE=base,
                   NO_PROXY="127.0.0.1,localhost")
        if secret is not None:
            env["CRABCROP_SECRET"] = secret
        self.p = subprocess.Popen(["node", MCP], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, env=env)
        self.n = 0
        self.rpc("initialize", {"protocolVersion": "2025-03-26",
                                "capabilities": {},
                                "clientInfo": {"name": "functional",
                                               "version": "1"}})
        self.notify("notifications/initialized")

    def _send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def notify(self, method):
        self._send({"jsonrpc": "2.0", "method": method})

    def rpc(self, method, params=None):
        self.n += 1
        self._send({"jsonrpc": "2.0", "id": self.n, "method": method,
                    "params": params or {}})
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("mcp server closed stdout")
            msg = json.loads(line)
            if msg.get("id") == self.n:
                return msg

    def tools(self):
        r = self.rpc("tools/list")
        return r["result"]["tools"]

    def call(self, name, args):
        r = self.rpc("tools/call", {"name": name, "arguments": args})
        res = r.get("result", {})
        texts = [c.get("text", "") for c in res.get("content", [])
                 if c.get("type") == "text"]
        return {"isError": bool(res.get("isError")),
                "text": "\n".join(texts), "raw": res}

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()
