#!/usr/bin/env python3
"""Freeze the upgrade corpus against a previous-release binary.

Issue #311: URLs minted under one release must keep rendering the
same bytes under the next one. This tool is the baseline half of
that check. It boots the given binary with the corpus configuration
(see README.md in this directory), mints the corpus URLs with the
repo Python SDK, renders every entry twice from two separate server
boots, and writes corpus.json with the expected status, content
type, byte length, and SHA-256 per entry.

An entry whose two renders differ is not self-deterministic, so it
cannot be a byte golden: it is dropped from corpus.json and
reported on stdout. Re-running this tool against a newer release
is the re-baseline path; record the reason for any entry whose
bytes change in that entry's "note" field (see README.md).

Sealing one corpus entry needs the SDK's seal extra
(cryptography), the same optional dependency the SDK documents.
Everything else here is the standard library plus the SDK itself.
"""
import argparse, hashlib, http.client, json, os, subprocess, sys
import tempfile, time, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FUNC_DIR = os.path.join(REPO, "tests", "functional")
CORPUS_DIR = os.path.join(FUNC_DIR, "corpus")
PRESETS = os.path.join(HERE, "presets.txt")

ORIGIN_PORT = 8991
SERVER_PORT = 8992
ORIGIN = f"http://127.0.0.1:{ORIGIN_PORT}"
BASE = f"http://127.0.0.1:{SERVER_PORT}"

# Test fixtures only. These secrets exist so URLs minted at freeze
# time stay verifiable for the life of the corpus; they protect
# nothing and must never be used anywhere else.
SECRET = "upgrade-corpus-current-secret"
SECRET_KEY_ID = "2026-08"
SECRET_PREVIOUS = "upgrade-corpus-previous-secret"
SECRET_PREVIOUS_KEY_ID = "2025-11"

EXP_FUTURE = 1893456000
EXP_PAST = 946684800
SEAL_NONCE = b"upg311nonce1"  # fixed, so the sealed entry is reproducible

sys.path.insert(0, os.path.join(REPO, "sdks", "python", "src"))
import crabcrop_url as sdk  # noqa: E402


def src(name):
    return f"{ORIGIN}/{name}"


def signed(ops, source, secret=SECRET, **kw):
    url = sdk.build_url("", secret, ops, source, **kw)
    return url  # build_url with an empty base returns the bare path


def corpus_entries():
    """The corpus: (id, kind, path, extra fields) tuples.

    Kinds: render (plain byte golden), render-preserve (rendered by
    an instance with CRABCROP_METADATA=preserve), expired (expects
    403), purge (byte golden plus a purge round trip in the check).
    """
    entries = []

    def add(eid, kind, path, **extra):
        entries.append(dict(id=eid, kind=kind, path=path, **extra))

    add("resize-exact", "render", signed("300x200", src("crab-photo.jpg")))
    add("resize-proportional", "render", signed("300x0", src("gradient.png")))
    add("crop-plain", "render", signed("crop:400x300", src("crab-photo.jpg")))
    add("crop-gravity", "render",
        signed("crop:300x300:bottom-right", src("crab-photo.jpg")))
    add("rotate-90", "render",
        signed("filters:rotate(90):format(png)", src("logo.png")))
    add("flip-horizontal", "render",
        signed("400x0/filters:flip(h):format(png)", src("gradient.png")))
    add("format-webp", "render",
        signed("300x0/filters:format(webp)", src("crab-photo.jpg")))
    add("format-avif", "render",
        signed("300x0/filters:format(avif)", src("crab-photo.jpg")))
    add("format-jxl", "render",
        signed("300x0/filters:format(jxl)", src("crab-photo.jpg")))
    add("format-png", "render",
        signed("300x0/filters:format(png)", src("crab-photo.jpg")))
    add("format-gif", "render",
        signed("300x0/filters:format(gif)", src("crab-photo.jpg")))
    add("quality-override", "render",
        signed("300x0/filters:quality(55)", src("crab-photo.jpg")))
    add("grayscale", "render",
        signed("300x0/filters:grayscale", src("crab-photo.jpg")))
    add("chain-multi-filter", "render",
        signed("400x0/filters:blur(1.5):saturation(1.4):sharpen(1.2)"
               ":format(webp)", src("crab-photo.jpg")))
    add("blur", "render",
        signed("300x0/filters:blur(2.5):format(png)", src("gradient.png")))
    add("sharpen", "render",
        signed("300x0/filters:sharpen(1.5):format(png)",
               src("gradient.png")))
    add("adjust-bright-contrast-saturation", "render",
        signed("filters:brighten(30):contrast(25):saturation(0.5)"
               ":format(png)", src("base-rgb.png")))
    add("invert", "render",
        signed("filters:invert:format(png)", src("base-rgb.png")))
    add("hue-rotate", "render",
        signed("filters:huerotate(90):format(png)", src("base-rgb.png")))
    add("pixelate", "render",
        signed("400x0/filters:pixelate(10):format(png)",
               src("crab-photo.jpg")))
    add("radius-rounded", "render",
        signed("300x0/filters:radius(40):format(png)",
               src("alpha-base.png")))
    add("pad-fill", "render",
        signed("pad:500x400:00ff00/filters:format(png)", src("logo.png")))
    add("metadata-strip", "render",
        signed("0x0/filters:format(jpeg)", src("exif-orient.jpg")))
    # The same remainder as metadata-strip, rendered by an instance
    # whose operator policy preserves metadata: the URL is identical,
    # the configuration is the variable under test.
    add("metadata-preserve", "render-preserve",
        signed("0x0/filters:format(jpeg)", src("exif-orient.jpg")))
    wm = urllib.parse.quote(src("logo.png"), safe="")
    add("watermark-overlay", "render",
        signed(f"600x0/filters:watermark({wm},-10,-10,80):format(png)",
               src("crab-photo.jpg")))
    add("animated-gif-resize", "render", signed("240x0", src("anim.gif")))
    add("preset-thumb", "render",
        signed("preset:thumb", src("crab-photo.jpg")))
    add("preset-card", "render",
        signed("preset:card", src("crab-photo.jpg")))
    add("source-root-short", "render",
        signed("300x200", "root:photos/crab-photo.jpg"))
    sealed_rem = f"300x200/{src('crab-photo.jpg')}"
    token = sdk.seal_with_nonce(SECRET, sealed_rem, SEAL_NONCE)
    add("sealed-url", "render", f"/{token}/crab-photo.jpg")
    add("exp-future", "render",
        signed("200x0", src("gradient.png"), expires=EXP_FUTURE))
    add("exp-past", "expired",
        signed("200x0", src("gradient.png"), expires=EXP_PAST))
    add("keyid-current", "render",
        signed("250x0", src("logo.png"), key_id=SECRET_KEY_ID))
    add("keyid-previous", "render",
        signed("250x0", src("gradient.png"), secret=SECRET_PREVIOUS,
               key_id=SECRET_PREVIOUS_KEY_ID))
    add("unkeyed-previous-secret", "render",
        signed("250x0/filters:format(png)", src("alpha-base.png"),
               secret=SECRET_PREVIOUS))
    purge_rem = f"275x0/{src('base-rgb.png')}"
    add("purge-roundtrip", "purge", signed("275x0", src("base-rgb.png")),
        remainder=purge_rem,
        purge_signature=sdk.purge_signature(SECRET, purge_rem))
    return entries


# ---------------------------------------------------------------- servers
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


class Servers:
    def __init__(self, binary, tmp):
        self.binary = binary
        self.tmp = tmp
        self.procs = {}
        roots = os.path.join(tmp, "roots.txt")
        with open(roots, "w") as f:
            f.write(f"photos = {ORIGIN}\n")
        self.roots = roots

    def start_origin(self):
        env = dict(os.environ, NO_PROXY="127.0.0.1,localhost",
                   FUNC_ORIGIN_PORT=str(ORIGIN_PORT),
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
                st, _, _ = fetch(ORIGIN, "/crab-photo.jpg", timeout=3)
                if st == 200:
                    return
            except Exception:
                pass
            time.sleep(0.3)
        raise RuntimeError("origin did not become ready")

    def start_server(self, preserve=False):
        env = dict(os.environ, NO_PROXY="127.0.0.1,localhost",
                   CRABCROP_SECRET=SECRET,
                   CRABCROP_SECRET_KEY_ID=SECRET_KEY_ID,
                   CRABCROP_SECRET_PREVIOUS=SECRET_PREVIOUS,
                   CRABCROP_SECRET_PREVIOUS_KEY_ID=SECRET_PREVIOUS_KEY_ID,
                   CRABCROP_PRESETS_FILE=PRESETS,
                   CRABCROP_SOURCE_ROOTS_FILE=self.roots,
                   CRABCROP_ALLOWED_SOURCES="127.0.0.1,localhost",
                   CRABCROP_PORT=str(SERVER_PORT))
        if preserve:
            env["CRABCROP_METADATA"] = "preserve"
        name = "server-preserve" if preserve else "server"
        log = open(os.path.join(self.tmp, f"{name}.log"), "w")
        p = subprocess.Popen([self.binary], env=env, stdout=log,
                             stderr=subprocess.STDOUT)
        self.procs[name] = p
        if not wait_health(SERVER_PORT, p):
            raise RuntimeError(f"crabcrop ({name}) did not become healthy; "
                               f"see {name}.log in the temp dir")

    def stop_server(self):
        for name in ("server", "server-preserve"):
            p = self.procs.pop(name, None)
            if p:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except Exception:
                    p.kill()

    def stop_all(self):
        self.stop_server()
        p = self.procs.pop("origin", None)
        if p:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()


def render_all(binary, tmp, entries):
    """One freeze round: every entry rendered once, on fresh boots."""
    srv = Servers(binary, tmp)
    out = {}
    try:
        srv.start_origin()
        srv.start_server(preserve=False)
        for e in entries:
            if e["kind"] == "render-preserve":
                continue
            st, hdrs, body = fetch(BASE, e["path"])
            out[e["id"]] = (st, hdrs.get("content-type", ""), body)
        srv.stop_server()
        srv.start_server(preserve=True)
        for e in entries:
            if e["kind"] != "render-preserve":
                continue
            st, hdrs, body = fetch(BASE, e["path"])
            out[e["id"]] = (st, hdrs.get("content-type", ""), body)
        srv.stop_server()
    finally:
        srv.stop_all()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin", required=True,
                    help="previous-release crabcrop binary")
    ap.add_argument("--release", default="v0.9.5",
                    help="release tag the binary was built from")
    ap.add_argument("--tag-commit", default="",
                    help="commit the tag points at (default: resolved "
                         "with git from the release name)")
    ap.add_argument("--out", default=os.path.join(HERE, "corpus.json"),
                    help="where to write corpus.json")
    args = ap.parse_args()

    binary = os.path.abspath(args.bin)
    if not os.path.isfile(binary):
        print(f"freeze: no such binary: {binary}", file=sys.stderr)
        return 1
    tag_commit = args.tag_commit
    if not tag_commit:
        r = subprocess.run(["git", "-C", REPO, "rev-parse",
                            f"{args.release}^{{commit}}"],
                           capture_output=True, text=True)
        tag_commit = r.stdout.strip() if r.returncode == 0 else "unknown"

    entries = corpus_entries()
    with tempfile.TemporaryDirectory(prefix="crabcrop-upgrade-freeze-") \
            as tmp:
        print(f"freeze: round 1 against {binary}")
        round1 = render_all(binary, tmp, entries)
        print("freeze: round 2 (fresh boots)")
        round2 = render_all(binary, tmp, entries)

    frozen, dropped = [], []
    for e in entries:
        st1, ct1, body1 = round1[e["id"]]
        st2, ct2, body2 = round2[e["id"]]
        want = 403 if e["kind"] == "expired" else 200
        if st1 != want:
            print(f"freeze: entry {e['id']} answered {st1}, want {want}: "
                  f"the corpus definition or the baseline is broken, "
                  f"refusing to freeze", file=sys.stderr)
            return 1
        digest1 = hashlib.sha256(body1).hexdigest()
        digest2 = hashlib.sha256(body2).hexdigest()
        if (st1, ct1, digest1) != (st2, ct2, digest2):
            dropped.append(e["id"])
            print(f"DROP {e['id']}: renders differ across boots "
                  f"(sha256 {digest1[:12]} vs {digest2[:12]}), "
                  f"not self-deterministic")
            continue
        row = {"id": e["id"], "kind": e["kind"], "path": e["path"],
               "status": st1, "content_type": ct1, "bytes": len(body1),
               "sha256": digest1, "note": ""}
        for k in ("remainder", "purge_signature"):
            if k in e:
                row[k] = e[k]
        frozen.append(row)

    doc = {
        "baseline": {
            "release": args.release,
            "tag_commit": tag_commit,
            "how": "Goldens produced by tests/upgrade/freeze.py against "
                   "a release build (cargo build --release --locked) of "
                   "the baseline tag, rendering each entry twice from "
                   "two separate server boots and keeping only "
                   "byte-identical entries.",
        },
        "freeze_date": time.strftime("%Y-%m-%d"),
        "secrets": {
            "current": SECRET,
            "current_key_id": SECRET_KEY_ID,
            "previous": SECRET_PREVIOUS,
            "previous_key_id": SECRET_PREVIOUS_KEY_ID,
            "comment": "Test fixtures only, minted for this corpus. "
                       "They protect nothing and appear here so the "
                       "frozen URLs stay verifiable.",
        },
        "server": {
            "origin_port": ORIGIN_PORT,
            "server_port": SERVER_PORT,
            "presets_file": "tests/upgrade/presets.txt",
            "source_root": f"photos = {ORIGIN} (generated at runtime)",
        },
        "entries": frozen,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"freeze: wrote {args.out}: {len(frozen)} entries frozen, "
          f"{len(dropped)} dropped")
    if dropped:
        print("freeze: dropped for nondeterminism: " + ", ".join(dropped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
