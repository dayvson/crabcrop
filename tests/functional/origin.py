#!/usr/bin/env python3
"""Functional-test origin: static corpus plus trap endpoints.

Every request line (method, path incl. query, host header) is appended
to the hits log so cases can assert exactly what the loader fetched
(and, for the explain cases, what it did NOT fetch).

Configuration comes from the environment, set by harness.start_origin:
  FUNC_ORIGIN_PORT  listen port
  FUNC_CORPUS       directory of static files (the frozen corpus)
  FUNC_HITS         hits log path
  FUNC_FRESH        path of the swappable fresh.jpg body
  FUNC_OUT          output dir (the /big.jpg body is staged here)

Endpoints:
  /<file>            static file from the corpus
  /fresh.jpg         bytes of FUNC_FRESH, re-read per request
  /redir/<n>         302 to /redir/<n-1>; /redir/0 is a 302 to /dog.jpg
  /redir-out         302 to http://example.org/dog.jpg (not allowlisted)
  /redir-loop        302 to itself
  /slow.jpg          sleeps 12 s, then serves dog.jpg
  /big.jpg           11 MiB body, image/jpeg
  /notimage          text/html body
  /badbytes.jpg      image/jpeg content type, non-image body
  /missing.jpg       404
"""
import os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ["FUNC_ORIGIN_PORT"])
SRC = os.environ["FUNC_CORPUS"]
FRESH = os.environ["FUNC_FRESH"]
HITS = open(os.environ["FUNC_HITS"], "a", buffering=1)
LOCK = threading.Lock()

CT = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
      ".svg": "image/svg+xml", ".webp": "image/webp", ".gif": "image/gif",
      ".mp4": "video/mp4", ".pdf": "application/pdf"}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _hit(self):
        with LOCK:
            HITS.write(f"{self.command} {self.path} "
                       f"host={self.headers.get('Host', '')}\n")

    def _send(self, code, body=b"", ct="text/plain", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        self._hit()
        path = self.path.split("?", 1)[0]
        if path == "/fresh.jpg":
            body = open(FRESH, "rb").read()
            return self._send(200, body, "image/jpeg")
        if path.startswith("/redir/"):
            try:
                n = int(path.split("/")[2])
            except Exception:
                return self._send(400, b"bad redir")
            loc = "/dog.jpg" if n <= 0 else f"/redir/{n - 1}"
            return self._send(302, b"", extra={"Location": loc})
        if path == "/redir-out":
            return self._send(302, b"",
                              extra={"Location": "http://example.org/dog.jpg"})
        if path == "/redir-loop":
            return self._send(302, b"", extra={"Location": "/redir-loop"})
        if path == "/slow.jpg":
            time.sleep(12)
            return self._send(200, open(f"{SRC}/dog.jpg", "rb").read(),
                              "image/jpeg")
        if path == "/big.jpg":
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            total = 11 * 1024 * 1024
            self.send_header("Content-Length", str(total))
            self.end_headers()
            chunk = b"\x00" * 65536
            sent = 0
            try:
                while sent < total:
                    self.wfile.write(chunk)
                    sent += len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if path == "/notimage":
            return self._send(200, b"<html><body>not an image</body></html>",
                              "text/html")
        if path == "/badbytes.jpg":
            return self._send(200, b"this is not an image at all",
                              "image/jpeg")
        if path == "/missing.jpg":
            return self._send(404, b"nope")
        name = path.lstrip("/")
        full = os.path.join(SRC, name)
        if os.path.isfile(full) and ".." not in name:
            ext = os.path.splitext(name)[1].lower()
            return self._send(200, open(full, "rb").read(),
                              CT.get(ext, "application/octet-stream"))
        return self._send(404, b"nope")

    do_HEAD = do_GET


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
