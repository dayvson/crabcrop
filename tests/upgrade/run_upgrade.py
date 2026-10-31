#!/usr/bin/env python3
"""Check a candidate build against the frozen upgrade corpus.

Issue #311: replays tests/upgrade/corpus.json (URLs minted and
rendered under the previous release, see freeze.py) against the
binary given with --bin, using the same origin, secrets, presets,
and source roots the corpus was frozen with. Every render entry
must match the frozen status, content type, byte length, and
SHA-256 exactly. Any difference fails the run and is reported by
URL: a silent byte change behind a signed URL is exactly the
incompatibility this suite exists to catch.

Standard library only, on purpose: the frozen paths need no SDK to
replay, and the check must run anywhere python3 does.

Exit status is 0 when every entry passes and 1 otherwise. With
--out DIR, the actual body of every mismatching entry is saved
under DIR for diffing.
"""
import argparse, hashlib, http.client, json, os, subprocess, sys
import tempfile, time, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FUNC_DIR = os.path.join(REPO, "tests", "functional")
CORPUS_DIR = os.path.join(FUNC_DIR, "corpus")
PRESETS = os.path.join(HERE, "presets.txt")
DEFAULT_CORPUS = os.path.join(HERE, "corpus.json")


def fetch(base, path, method="GET", timeout=60):
    parts = urllib.parse.urlsplit(base + path)
    conn = http.client.HTTPConnection(parts.hostname, parts.port,
                                      timeout=timeout)
    try:
        conn.request(method, parts.path)
        r = conn.getresponse()
        body = r.read()
        hdrs = {k.lower(): v for k, v in r.getheaders()}
        return r.status, hdrs, body
    finally:
        conn.close()


def wait_health(port, proc, timeout_s=60):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            st, _, body = fetch(f"http://127.0.0.1:{port}", "/health",
                                timeout=3)
            if st == 200 and body.strip() == b"ok":
                return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


class Runner:
    def __init__(self, binary, doc, tmp):
        self.binary = binary
        self.tmp = tmp
        self.secrets = doc["secrets"]
        server = doc["server"]
        self.origin_port = server["origin_port"]
        self.server_port = server["server_port"]
        self.origin = f"http://127.0.0.1:{self.origin_port}"
        self.base = f"http://127.0.0.1:{self.server_port}"
        roots = os.path.join(tmp, "roots.txt")
        with open(roots, "w") as f:
            f.write(f"photos = {self.origin}\n")
        self.roots = roots
        self.procs = {}
        self.mode = None

    def start_origin(self):
        env = dict(os.environ, NO_PROXY="127.0.0.1,localhost",
                   FUNC_ORIGIN_PORT=str(self.origin_port),
                   FUNC_CORPUS=CORPUS_DIR,
                   FUNC_HITS=os.path.join(self.tmp, "origin-hits.log"),
                   FUNC_FRESH=os.path.join(CORPUS_DIR, "crab-photo.jpg"),
                   FUNC_OUT=self.tmp)
        log = open(os.path.join(self.tmp, "origin.log"), "w")
        p = subprocess.Popen([sys.executable,
                              os.path.join(FUNC_DIR, "origin.py")],
                             env=env, stdout=log, stderr=subprocess.STDOUT)
        self.procs["origin"] = p
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                st, _, _ = fetch(self.origin, "/crab-photo.jpg", timeout=3)
                if st == 200:
                    return
            except Exception:
                pass
            time.sleep(0.3)
        raise RuntimeError("origin did not become ready")

    def ensure_server(self, preserve):
        mode = "preserve" if preserve else "strip"
        if self.mode == mode:
            return
        self.stop_server()
        s = self.secrets
        env = dict(os.environ, NO_PROXY="127.0.0.1,localhost",
                   CRABCROP_SECRET=s["current"],
                   CRABCROP_SECRET_KEY_ID=s["current_key_id"],
                   CRABCROP_SECRET_PREVIOUS=s["previous"],
                   CRABCROP_SECRET_PREVIOUS_KEY_ID=s["previous_key_id"],
                   CRABCROP_PRESETS_FILE=PRESETS,
                   CRABCROP_SOURCE_ROOTS_FILE=self.roots,
                   CRABCROP_ALLOWED_SOURCES="127.0.0.1,localhost",
                   CRABCROP_PORT=str(self.server_port))
        if preserve:
            env["CRABCROP_METADATA"] = "preserve"
        log = open(os.path.join(self.tmp, f"server-{mode}.log"), "w")
        p = subprocess.Popen([self.binary], env=env, stdout=log,
                             stderr=subprocess.STDOUT)
        self.procs["server"] = p
        if not wait_health(self.server_port, p):
            raise RuntimeError(f"crabcrop ({mode}) did not become "
                               f"healthy; see server-{mode}.log")
        self.mode = mode

    def stop_server(self):
        p = self.procs.pop("server", None)
        if p:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        self.mode = None

    def stop_all(self):
        self.stop_server()
        p = self.procs.pop("origin", None)
        if p:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()


def compare(entry, st, hdrs, body):
    """None when the render matches the frozen golden, else a reason."""
    digest = hashlib.sha256(body).hexdigest()
    if st != entry["status"]:
        return f"status {st}, want {entry['status']}"
    ct = hdrs.get("content-type", "")
    if ct != entry["content_type"]:
        return f"content-type {ct!r}, want {entry['content_type']!r}"
    if len(body) != entry["bytes"] or digest != entry["sha256"]:
        return (f"sha256 {digest}, want {entry['sha256']} "
                f"(bytes {len(body)}, want {entry['bytes']})")
    return None


def check_purge(runner, entry):
    """The purge round trip: render, purge, re-render, bad purge."""
    st, hdrs, body = fetch(runner.base, entry["path"])
    bad = compare(entry, st, hdrs, body)
    if bad:
        return f"initial render: {bad}", body
    sig = entry["purge_signature"]
    target = entry["remainder"]
    st, _, body = fetch(runner.base, f"/purge/{sig}/{target}",
                        method="POST")
    text = body.decode("utf-8", "replace").strip()
    if st != 200 or text != "purged 1":
        return f"purge answered {st} {text!r}, want 200 'purged 1'", body
    st, hdrs, body = fetch(runner.base, entry["path"])
    bad = compare(entry, st, hdrs, body)
    if bad:
        return f"render after purge: {bad}", body
    corrupted = ("0" if sig[0] != "0" else "1") + sig[1:]
    st, _, _ = fetch(runner.base, f"/purge/{corrupted}/{target}",
                     method="POST")
    if st != 403:
        return f"purge with a corrupted signature answered {st}, " \
               f"want 403", body
    return None, body


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin", default=os.path.join(REPO, "target",
                                                  "debug", "crabcrop"),
                    help="candidate crabcrop binary "
                         "(default: target/debug/crabcrop)")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help="corpus.json to replay")
    ap.add_argument("--out", default="",
                    help="save mismatching bodies under this directory")
    args = ap.parse_args()

    binary = os.path.abspath(args.bin)
    if not os.path.isfile(binary):
        print(f"upgrade: no such binary: {binary}", file=sys.stderr)
        return 1
    with open(args.corpus) as f:
        doc = json.load(f)
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    passed, failed = 0, 0
    with tempfile.TemporaryDirectory(prefix="crabcrop-upgrade-") as tmp:
        runner = Runner(binary, doc, tmp)
        try:
            runner.start_origin()
            for entry in doc["entries"]:
                runner.ensure_server(entry["kind"] == "render-preserve")
                body = b""
                try:
                    if entry["kind"] == "expired":
                        st, _, body = fetch(runner.base, entry["path"])
                        bad = None if st == 403 else \
                            f"status {st}, want 403 (expired URL)"
                    elif entry["kind"] == "purge":
                        bad, body = check_purge(runner, entry)
                    else:
                        st, hdrs, body = fetch(runner.base, entry["path"])
                        bad = compare(entry, st, hdrs, body)
                except Exception as e:
                    bad = f"request errored: {type(e).__name__}: {e}"
                if bad is None:
                    passed += 1
                    print(f"PASS {entry['id']}: {entry['path']}")
                else:
                    failed += 1
                    print(f"FAIL {entry['id']}: {entry['path']}: {bad}")
                    if args.out and body:
                        with open(os.path.join(args.out,
                                               entry["id"] + ".bin"),
                                  "wb") as f:
                            f.write(body)
        finally:
            runner.stop_all()
    total = passed + failed
    print(f"upgrade corpus: {passed} passed, {failed} failed, "
          f"{total} total (baseline {doc['baseline']['release']})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
