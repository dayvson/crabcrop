#!/usr/bin/env python3
"""Functional test harness for crabcrop (issue #232).

Generalizes the feature-campaign harnesses (waves 1 to 4) into one
permanent in-repo runner support library:

- boots a locally built crabcrop binary in several configuration
  profiles ("instances") on scratch ports, plus a hit-counting origin
  server that serves the frozen corpus (tests/functional/corpus);
- signs with the repo's own Python SDK, so the harness dogfoods the
  client it also parity-checks;
- prime / purge / timed re-request measurement, with the server log
  confirming the cache state of the timed request;
- a declarative assertion DSL for the data-driven case tables
  (cases_core.json, cases_overlay.json) and recording primitives for
  the scripted tables (platform_cases.py, ai_cases.py).

Assertion philosophy (issue #232): property assertions, never byte
goldens. Dimensions exact, format and magic correct, channel
statistics within tolerance, output bytes within a band of the
blessed baseline (baselines.json, about +-15 percent).
"""
import hashlib, http.client, json, math, os, shutil, subprocess, sys, time
import urllib.parse
from PIL import Image, ImageStat, ImageChops

FUNC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(FUNC_DIR, "..", ".."))
CORPUS = os.path.join(FUNC_DIR, "corpus")
SECRET = "functional-test-secret-0123456789"

# Scratch ports. Nothing here may collide with scripts/test_all.sh
# (8909/8912/3321) or a developer's own server (default 8904 family).
PORTS = {"origin": 8971, "A": 8972, "AN": 8973, "AP": 8974, "P": 8975,
         "N": 8976, "U": 8977, "OP": 8978, "E": 8979, "BOOT": 8980,
         "AI": 8981, "AIN": 8982, "AIC": 8983, "R": 8984, "V": 8985,
         "D": 8986, "C": 8987}
BASES = {k: f"http://127.0.0.1:{p}" for k, p in PORTS.items() if k != "origin"}
ORIGIN = f"http://127.0.0.1:{PORTS['origin']}"
SRC127 = ORIGIN
SRCLOC = f"http://localhost:{PORTS['origin']}"

# Set by configure() from run.py.
BIN = None
OUT = os.path.join(FUNC_DIR, "out")
UPLOAD_DIR = None
HITS_PATH = None
FRESH_BIN = None
LOGS = {}
RESULTS = {}
OBS = None

sys.path.insert(0, os.path.join(REPO, "sdks", "python", "src"))
import crabcrop_url as sdk  # noqa: E402


def configure(bin_path, out_dir):
    global BIN, OUT, UPLOAD_DIR, HITS_PATH, FRESH_BIN, OBS
    BIN = bin_path
    OUT = out_dir
    UPLOAD_DIR = os.path.join(OUT, "uploads")
    HITS_PATH = os.path.join(OUT, "origin-hits.log")
    FRESH_BIN = os.path.join(OUT, "fresh.bin")
    os.makedirs(os.path.join(OUT, "renders"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for name in list(BASES) + ["BOOT", "origin"]:
        LOGS[name] = os.path.join(OUT, "logs", f"server-{name.lower()}.log")
    OBS = open(os.path.join(OUT, "observations.jsonl"), "a")


# ---------------------------------------------------------------- servers
_PROCS = {}


def _wait_health(port, timeout_s, proc=None):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            st, _, payload, _ = fetch(f"http://127.0.0.1:{port}/health",
                                      timeout=3)
            if st == 200 and payload.strip() == b"ok":
                return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def start_server(name, extra_env=None, health_timeout=60):
    """Boot one crabcrop instance profile. Returns True when healthy."""
    port = PORTS[name]
    env = dict(os.environ, CRABCROP_SECRET=SECRET,
               CRABCROP_PORT=str(port), NO_PROXY="127.0.0.1,localhost")
    env.update(extra_env or {})
    logf = open(LOGS[name], "w")
    p = subprocess.Popen([BIN], env=env, stdout=logf,
                         stderr=subprocess.STDOUT)
    _PROCS[name] = p
    return _wait_health(port, health_timeout, p)


def start_origin():
    env = dict(os.environ, NO_PROXY="127.0.0.1,localhost",
               FUNC_ORIGIN_PORT=str(PORTS["origin"]), FUNC_CORPUS=CORPUS,
               FUNC_HITS=HITS_PATH, FUNC_FRESH=FRESH_BIN, FUNC_OUT=OUT)
    logf = open(LOGS["origin"], "w")
    p = subprocess.Popen([sys.executable, os.path.join(FUNC_DIR, "origin.py")],
                         env=env, stdout=logf, stderr=subprocess.STDOUT)
    _PROCS["origin"] = p
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            st, _, _, _ = fetch(f"{ORIGIN}/dog.jpg", timeout=3)
            if st == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def stop_servers():
    for p in _PROCS.values():
        try:
            p.terminate()
        except Exception:
            pass
    for p in _PROCS.values():
        try:
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    _PROCS.clear()


FONT = os.path.join(REPO, "tests", "fixtures", "font.ttf")
ALLOW = {"CRABCROP_ALLOWED_SOURCES": "127.0.0.1,localhost"}

# Instance profiles for the CI tiers. AI instances (AI/AIN/AIC) are
# built by run.py for the local tier only, from operator-supplied
# model paths; models are deployment artifacts, never repo content.
INSTANCE_ENV = {
    "A": dict(CRABCROP_ALLOW_UNSAFE="1", CRABCROP_FONT_PATH=FONT, **ALLOW),
    "AN": dict(CRABCROP_ALLOW_UNSAFE="1", **ALLOW),
    "AP": dict(CRABCROP_ALLOW_UNSAFE="1", CRABCROP_FONT_PATH=FONT,
               CRABCROP_METADATA="preserve", **ALLOW),
    "P": dict(CRABCROP_ALLOW_UNSAFE="1",
              CRABCROP_PRESETS_FILE=os.path.join(FUNC_DIR, "presets.txt"),
              **ALLOW),
    "N": dict(**ALLOW),
    "OP": dict(CRABCROP_ALLOW_UNSAFE="1",
               CRABCROP_PRESETS_FILE=os.path.join(FUNC_DIR, "presets.txt"),
               CRABCROP_ONLY_PRESETS="1", **ALLOW),
    "E": dict(CRABCROP_ALLOW_UNSAFE="1",
              CRABCROP_ALLOWED_SOURCES="localhost"),
    "R": dict(CRABCROP_ALLOW_UNSAFE="1",
              CRABCROP_SOURCE_ROOTS_FILE=os.path.join(FUNC_DIR, "roots.txt"),
              **ALLOW),
    # V carries the video frame decoder (issue #214): the
    # ffmpeg on PATH, resolved here. Without one the env
    # value is empty, the instance boots decoder-less, and
    # the frames cells record NOTEs instead of asserting.
    "V": dict(CRABCROP_ALLOW_UNSAFE="1",
              CRABCROP_VIDEO_DECODER=shutil.which("ffmpeg") or "",
              **ALLOW),
    # D carries the PDF renderer (issue #215): the
    # pdftocairo on PATH, resolved here, with a short
    # renderer timeout so a pathological document sheds
    # fast. Without one the env value is empty, the
    # instance boots renderer-less, and the pages cells
    # record NOTEs instead of asserting.
    "D": dict(CRABCROP_ALLOW_UNSAFE="1",
              CRABCROP_PDF_RENDERER=shutil.which("pdftocairo") or "",
              CRABCROP_PDF_TIMEOUT_SECS="5",
              **ALLOW),
    # C carries the C2PA signing identity (issue #173):
    # the suite's test-only chain and key, the fixtures
    # the Rust suites use, so signed output validates as
    # "crabcrop test signer".
    "C": dict(CRABCROP_ALLOW_UNSAFE="1",
              CRABCROP_C2PA_CERT=os.path.join(FUNC_DIR, "c2pa_cert.pem"),
              CRABCROP_C2PA_KEY=os.path.join(FUNC_DIR, "c2pa_key.pem"),
              **ALLOW),
}
# U needs the runtime upload dir; filled in by start_instances().
def start_instances(names):
    ok = {}
    for name in names:
        if name == "U":
            env = dict(CRABCROP_ALLOW_UNSAFE="1",
                       CRABCROP_UPLOAD_DIR=UPLOAD_DIR, **ALLOW)
        else:
            env = dict(INSTANCE_ENV[name])
        ok[name] = start_server(name, env)
    return ok


# ---------------------------------------------------------------- signing
def sign(rem):
    return sdk.sign(SECRET, rem)


def signed_path(rem):
    return f"/{sign(rem)}/{rem}"


def unsafe_path(rem):
    return f"/unsafe/{rem}"


def purge_sig(target):
    return sdk.purge_signature(SECRET, target)


def group_sig(group):
    return sdk.purge_group_signature(SECRET, group)


def upload_sig(name, expires=None):
    return sdk.upload_signature(SECRET, name, expires)


# ---------------------------------------------------------------- HTTP
def fetch(url, headers=None, timeout=60, method="GET", body=None,
          path_as_is=False):
    """One raw HTTP request. Returns (status, headers, payload, ms).

    http.client sends the path exactly as given (no normalization),
    which the traversal and encoded-remainder cases rely on.
    """
    parts = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port,
                                      timeout=timeout)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    t0 = time.time()
    try:
        conn.request(method, path, body=body, headers=headers or {})
        r = conn.getresponse()
        payload = r.read()
        hdrs = {}
        for k, v in r.getheaders():
            hdrs[k.lower()] = v
        return r.status, hdrs, payload, (time.time() - t0) * 1000
    finally:
        conn.close()


def request(base, method, path, headers=None, body=None, timeout=60,
            path_as_is=False):
    url = BASES[base] + path if base else path
    if isinstance(body, str) and body.startswith("@"):
        body = open(body[1:], "rb").read()
    return fetch(url, headers, timeout, method=method, body=body)


def get(base, path, headers=None, timeout=60, path_as_is=False):
    return request(base, "GET", path, headers, None, timeout, path_as_is)


# ---------------------------------------------------------------- CLI
def cli(verb, arg, base="P", extra=None):
    env = dict(os.environ, CRABCROP_BASE=BASES[base], CRABCROP_SECRET=SECRET,
               NO_PROXY="127.0.0.1,localhost")
    cmd = [BIN, verb] + (extra or []) + [arg]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=60)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "cli-timeout"


def purge_cli(remainder, base="P"):
    return cli("purge", remainder, base)


def purge(remainder, base="A"):
    return purge_cli(remainder, base)


# ---------------------------------------------------------------- logs
def log_size(base):
    try:
        return os.path.getsize(LOGS[base])
    except OSError:
        return 0


def new_log_lines(pos, base):
    try:
        with open(LOGS[base], "rb") as f:
            f.seek(pos)
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    out = []
    for line in data.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def raw_log_tail(pos, base):
    try:
        with open(LOGS[base], "rb") as f:
            f.seek(pos)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


# ---------------------------------------------------------------- origin
def origin_hits(substr=None):
    try:
        lines = open(HITS_PATH).read().splitlines()
    except OSError:
        return 0
    if substr is None:
        return len(lines)
    return sum(1 for ln in lines if substr in ln)


# ---------------------------------------------------------------- facts
IMG_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp",
           "image/gif": "gif", "image/avif": "avif", "image/jxl": "jxl",
           "application/json": "json", "image/svg+xml": "svg"}


def fact_of(cell, base, method, path, status, hdrs, payload, client_ms):
    fact = {"cell": cell, "base": base, "method": method, "path": path[:220],
            "status": status, "ct": hdrs.get("content-type", ""),
            "bytes": len(payload), "client_ms": client_ms,
            "cache_status": hdrs.get("cache-status", ""),
            "cache_group": hdrs.get("cache-group", ""),
            "cache_control": hdrs.get("cache-control", ""),
            "etag": hdrs.get("etag", ""),
            "headers": hdrs,
            "md5": hashlib.md5(payload).hexdigest(),
            "body": payload[:400].decode("utf-8", "replace")
            if status != 200 else ""}
    OBS.write(json.dumps(fact) + "\n")
    OBS.flush()
    return fact


def timed_render(cell, base, path, headers=None, timeout=120, save=True):
    """One GET with facts recorded; the caller drives priming/purging."""
    pos = log_size(base)
    status, hdrs, payload, ms = get(base, path, headers, timeout)
    lines = [l for l in new_log_lines(pos, base) if l.get("method") == "GET"]
    logline = lines[-1] if lines else {}
    fact = fact_of(cell, base, "GET", path, status, hdrs, payload, ms)
    fact["server_ms"] = logline.get("total_ms")
    fact["cache"] = logline.get("cache")
    ext = IMG_EXT.get(hdrs.get("content-type", "").split(";")[0], "bin")
    if save and status == 200:
        p = os.path.join(OUT, "renders", f"{cell}.{ext}")
        with open(p, "wb") as f:
            f.write(payload)
        fact["path"] = p
    else:
        fact["path"] = None
    return fact


def prime_purge_render(cell, base, path, remainder, headers=None,
                       timeout=120):
    """Campaign pattern: prime, purge via CLI, timed re-request (a miss)."""
    get(base, path, headers, timeout)
    purge_cli(remainder, base)
    return timed_render(cell, base, path, headers, timeout)


# ---------------------------------------------------------------- records
_save_hook = None


def set_save_hook(fn):
    global _save_hook
    _save_hook = fn


def rec(cell, group, ops, src, verdict, fact, evidence):
    RESULTS[cell] = {"id": cell, "group": group, "ops": ops, "src": src,
                     "verdict": verdict,
                     "status": (fact or {}).get("status"),
                     "content_type": (fact or {}).get("ct", ""),
                     "bytes": (fact or {}).get("bytes"),
                     "server_ms": (fact or {}).get("server_ms"),
                     "client_ms": (fact or {}).get("client_ms"),
                     "cache": (fact or {}).get("cache"),
                     "evidence": evidence}
    if _save_hook:
        _save_hook()
    print(f"[{verdict:4s}] {cell}: {evidence}", flush=True)


def check(cell, group, ops, src, cond, fact, evidence):
    rec(cell, group, ops, src, "PASS" if cond else "FAIL", fact, evidence)
    return cond


def note(cell, group, ops, src, fact, evidence):
    rec(cell, group, ops, src, "NOTE", fact, evidence)


# ================================================================ DSL side
# Data-driven cases (cases_core.json, cases_overlay.json): the campaign
# assertion DSL, waves 1 and 2 merged. Facts come from an /unsafe/
# render on a profile instance: default -> A, preserve -> AP,
# nofont -> AN.
PROFILE_INSTANCE = {"default": "A", "preserve": "AP", "nofont": "AN"}
REG = {}            # case id -> fact dict (this process)
BASELINES = {}      # case id -> {"bytes": n, "mean": [r, g, b]}
BLESS = None        # dict collecting fresh baselines when blessing


def instance_of(case):
    return case.get("instance") or PROFILE_INSTANCE[case.get("profile",
                                                             "default")]


def remainder_of(ops, src):
    return f"{ops}/{ORIGIN}/{src}" if ops else f"{ORIGIN}/{src}"


def render_dsl(cell, ops, src, instance="A", headers=None, accept=None,
               save=True, timeout=180):
    """Prime, purge via CLI, timed re-request. Returns a fact dict."""
    rem = remainder_of(ops, src)
    url = f"{BASES[instance]}/unsafe/{rem}"
    hdrs = dict(headers or {})
    if accept:
        hdrs["Accept"] = accept
    fetch(url, hdrs, timeout=timeout)          # prime
    purge_cli(rem, instance)
    pos = log_size(instance)
    status, rsp_headers, payload, client_ms = fetch(url, hdrs,
                                                    timeout=timeout)
    lines = [l for l in new_log_lines(pos, instance)
             if l.get("method") == "GET"]
    logline = lines[-1] if lines else {}
    ext = IMG_EXT.get(rsp_headers.get("content-type", "").split(";")[0],
                      "bin")
    path = os.path.join(OUT, "renders", f"{cell}.{ext}")
    if save and status == 200:
        with open(path, "wb") as f:
            f.write(payload)
    fact = {"cell": cell, "ops": ops, "src": src, "status": status,
            "ct": rsp_headers.get("content-type", ""),
            "bytes": len(payload), "client_ms": round(client_ms, 1),
            "server_ms": logline.get("total_ms"),
            "cache": logline.get("cache"),
            "x_quality": rsp_headers.get("x-crabcrop-quality"),
            "headers": rsp_headers,
            "cache_status": rsp_headers.get("cache-status", ""),
            "md5": hashlib.md5(payload).hexdigest(),
            "path": path if (save and status == 200) else None,
            "body": payload[:300].decode("utf-8", "replace")
            if status != 200 else ""}
    OBS.write(json.dumps(fact) + "\n")
    OBS.flush()
    return fact


# Wave 3 scripted cells use this name/signature (base A/B/C -> AI/AIN/AIC).
AI_BASE_MAP = {"A": "AI", "B": "AIN", "C": "AIC"}
AI_BASES = {"A": BASES["AI"], "B": BASES["AIN"], "C": BASES["AIC"]}


def render(cell, ops, src, base="A", headers=None, accept=None, save=True,
           timeout=300):
    return render_dsl(cell, ops, src, AI_BASE_MAP[base], headers, accept,
                      save, timeout)


def meta_dsl(ops, src, instance="A", headers=None):
    rem = remainder_of(ops, src)
    hdrs = {"Accept": "application/json"}
    hdrs.update(headers or {})
    status, h, payload, ms = fetch(f"{BASES[instance]}/unsafe/{rem}", hdrs)
    try:
        j = json.loads(payload)
    except Exception:
        j = {"_raw": payload[:300].decode("utf-8", "replace")}
    return status, j


def meta(ops, src, base="A", accept_params=None, timeout=300):
    """Wave 3 style meta: accept_params appended to the JSON Accept."""
    rem = remainder_of(ops, src)
    instance = AI_BASE_MAP[base]
    accept = "application/json" + (f"; {accept_params}"
                                   if accept_params else "")
    status, h, payload, ms = fetch(f"{BASES[instance]}/unsafe/{rem}",
                                   {"Accept": accept}, timeout=timeout)
    try:
        j = json.loads(payload)
    except Exception:
        j = {"_raw": payload[:300].decode("utf-8", "replace")}
    fact = {"meta": True, "base": base, "ops": ops, "src": src,
            "status": status, "accept_params": accept_params,
            "client_ms": round(ms, 1), "json": j}
    OBS.write(json.dumps(fact) + "\n")
    OBS.flush()
    return status, j


# ------------------------------------------------------------ PIL helpers
def open_img(path):
    im = Image.open(path)
    im.load()
    return im


def stats(im):
    im = im.convert("RGB")
    st = ImageStat.Stat(im)
    return {"mean": [round(m, 2) for m in st.mean],
            "std": [round(s, 2) for s in st.stddev]}


def small(im, n=64):
    return im.convert("RGB").resize((n, n))


def channel_spread(im):
    """Max per-pixel |R-G|,|G-B|,|R-B| over a downscale: ~0 means gray."""
    px = list(small(im).getdata())
    return max(max(abs(r - g), abs(g - b), abs(r - b)) for r, g, b in px)


def mean_hue(im):
    hsv = im.convert("RGB").resize((96, 96)).convert("HSV")
    px = [p for p in hsv.getdata() if p[1] > 60 and p[2] > 40]
    if not px:
        return None
    sx = sum(math.cos(p[0] / 255 * 2 * math.pi) for p in px)
    sy = sum(math.sin(p[0] / 255 * 2 * math.pi) for p in px)
    return round((math.atan2(sy, sx) % (2 * math.pi)) / (2 * math.pi)
                 * 360, 1)


def local_var(im):
    g = im.convert("L").resize((128, 128))
    px = list(g.getdata())
    w = 128
    diffs = [abs(px[i] - px[i + 1]) for i in range(len(px) - 1)
             if (i + 1) % w]
    return round(sum(diffs) / len(diffs), 2)


def alpha_stats(path):
    """Fractions of fully opaque / transparent / partial alpha pixels."""
    im = open_img(path)
    if im.mode in ("RGBA", "LA"):
        a = im.getchannel("A")
    elif "transparency" in im.info:
        a = im.convert("RGBA").getchannel("A")
    else:
        return {"has_alpha": False, "opaque": 1.0, "transparent": 0.0,
                "partial": 0.0}
    hist = a.histogram()
    total = sum(hist)
    return {"has_alpha": True,
            "opaque": round(hist[255] / total, 4),
            "transparent": round(hist[0] / total, 4),
            "partial": round(sum(hist[1:255]) / total, 4)}


def pixel(path, x, y):
    return open_img(path).convert("RGBA").getpixel((x, y))


def region_mean_p(path, box):
    st = ImageStat.Stat(open_img(path).convert("RGBA").crop(box)
                        .convert("RGB"))
    return [round(m, 1) for m in st.mean]


def local_var_p(path, box=None):
    im = open_img(path).convert("L")
    if box:
        im = im.crop(box)
    im = im.resize((min(im.width, 256), min(im.height, 256)))
    px = list(im.getdata())
    w = im.width
    diffs = [abs(px[i] - px[i + 1]) for i in range(len(px) - 1)
             if (i + 1) % w]
    return round(sum(diffs) / len(diffs), 2) if diffs else 0.0


def distinct_colors(path, box=None, cap=200000):
    im = open_img(path).convert("RGB")
    if box:
        im = im.crop(box)
    im.thumbnail((256, 256))
    return len(set(im.getdata()))


def color_count(path, pred, sample=4):
    im = open_img(path).convert("RGB")
    im = im.resize((im.width // sample, im.height // sample))
    return sum(1 for p in im.getdata() if pred(*p))


def mean_diff(path_a, path_b):
    a = open_img(path_a).convert("RGB")
    b = open_img(path_b).convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size)
    return round(sum(ImageStat.Stat(ImageChops.difference(a, b)).mean)
                 / 3, 2)


def decode_embedding(b64):
    import base64, struct
    raw = base64.b64decode(b64)
    n = len(raw) // 2
    vals = list(struct.unpack(f"<{n}e", raw))
    norm = math.sqrt(sum(v * v for v in vals))
    return n, round(norm, 4)


# ------------------------------------------------------------ evaluator
def img_of(cid):
    return open_img(REG[cid]["path"])


def rgb(im):
    return im.convert("RGB")


def lum_std(im):
    return ImageStat.Stat(im.convert("L")).stddev[0]


def sat_mean(im):
    return ImageStat.Stat(
        im.convert("RGB").resize((64, 64)).convert("HSV")).mean[1]


def hue_of(im):
    return mean_hue(im)


def region(im, box):
    return im.crop(tuple(box))


def local_var_box(im, box):
    g = region(im, box).convert("L")
    g = g.resize((min(g.width, 128), min(g.height, 128))) \
        if g.width > 128 else g
    px = list(g.getdata())
    w = g.width
    diffs = [abs(px[i] - px[i + 1]) for i in range(len(px) - 1)
             if (i + 1) % w]
    return sum(diffs) / max(len(diffs), 1)


PREDS = {
    "r_dom": lambda p: p[0] > p[1] + 50 and p[0] > p[2] + 50,
    "g_dom": lambda p: p[1] > p[0] + 50 and p[1] > p[2] + 50,
    "b_dom": lambda p: p[2] > p[0] + 50 and p[2] > p[1] + 50,
    "yellow": lambda p: p[0] > 180 and p[1] > 160 and p[2] < 110,
    "white": lambda p: min(p) > 240,
    "black": lambda p: max(p) < 15,
}


def ev(a, fact, case):
    """Evaluate one DSL assertion; return (ok, detail)."""
    t = a["t"]
    if t == "status":
        return fact["status"] == a["eq"], \
            f"status={fact['status']} want {a['eq']}"
    if t == "ct":
        return fact["ct"].startswith(a["eq"]), f"ct={fact['ct']}"
    if t == "body_contains":
        return a["s"] in fact["body"], f"body={fact['body'][:70]!r}"
    if t == "header":
        v = (fact.get("headers") or {}).get(a["k"])
        return v == a["eq"], f"{a['k']}={v} want {a['eq']}"
    if t == "header_present":
        v = (fact.get("headers") or {}).get(a["k"])
        return v is not None, f"{a['k']}={v}"
    if t == "header_contains":
        v = (fact.get("headers") or {}).get(a["k"], "")
        return a["s"] in v, f"header {a['k']}={v!r} contains {a['s']!r}"
    if t == "bytes_lte":
        return fact["bytes"] <= a["v"], f"bytes={fact['bytes']} <= {a['v']}"
    if t == "bytes_gt_cell":
        other = REG[a["cell"]]["bytes"]
        return fact["bytes"] > other, \
            f"bytes={fact['bytes']} > {a['cell']} {other}"
    if t == "meta":
        mj = case["_meta"]
        return mj.get(a["field"]) == a["eq"], \
            f"meta.{a['field']}={mj.get(a['field'])} want {a['eq']}"
    if t == "meta_field_present":
        return a["field"] in case["_meta"], \
            f"meta has {a['field']}: {a['field'] in case['_meta']}"
    if t == "meta_field_len":
        v = case["_meta"].get(a["field"])
        return isinstance(v, list) and len(v) == a["eq"], \
            f"meta.{a['field']} len={len(v) if isinstance(v, list) else v}"
    if t == "meta_bytes_match":
        return case["_meta"].get("bytes") == fact["bytes"], \
            f"meta.bytes={case['_meta'].get('bytes')} vs image {fact['bytes']}"
    # ---- image assertions below need a decoded output ----
    im = img_of(case["id"])
    if t == "dims":
        tol = a.get("tol", 0)
        want = a["eq"]
        ok = all(abs(im.size[i] - want[i]) <= tol for i in (0, 1))
        return ok, f"dims={im.size} want {tuple(want)} tol {tol}"
    if t == "dims_gt":
        return all(im.size[i] > a["eq"][i] for i in (0, 1)), \
            f"dims={im.size} > {tuple(a['eq'])}"
    if t == "pixel":
        p = rgb(im).getpixel(tuple(a["xy"]))
        return PREDS[a["pred"]](p), f"pixel{a['xy']}={p} pred {a['pred']}"
    if t == "pixel_eq":
        p = rgb(im).getpixel(tuple(a["xy"]))
        ok = all(abs(p[i] - a["rgb"][i]) <= a.get("tol", 8) for i in range(3))
        return ok, f"pixel{a['xy']}={p} want ~{tuple(a['rgb'])}"
    if t == "pixel_rgba":
        p = im.convert("RGBA").getpixel(tuple(a["xy"]))
        ok = all(abs(p[i] - a["rgba"][i]) <= a.get("tol", 8)
                 for i in range(4))
        return ok, f"rgba{a['xy']}={p} want ~{tuple(a['rgba'])}"
    if t == "pixel_alpha":
        al = im.convert("RGBA").getpixel(tuple(a["xy"]))[3]
        return abs(al - a["eq"]) <= a.get("tol", 8), \
            f"alpha{a['xy']}={al} want {a['eq']}"
    if t == "gray":
        worst = 0
        n = getattr(im, "n_frames", 1)
        for fr in {0, min(5, n - 1)}:
            try:
                im.seek(fr)
            except Exception:
                pass
            worst = max(worst, channel_spread(im))
        return worst <= a["max_spread"], \
            f"max channel spread={worst} <= {a['max_spread']}"
    if t == "frames":
        return getattr(im, "n_frames", 1) == a["eq"], \
            f"frames={getattr(im, 'n_frames', 1)}"
    if t == "diff_mean":
        other = rgb(img_of(a["vs"]))
        if other.size != im.size:
            other = other.resize(im.size)
        d = ImageStat.Stat(
            ImageChops.difference(rgb(im), other).convert("L")).mean[0]
        if "lt" in a:
            return d < a["lt"], f"diff vs {a['vs']} mean={d:.2f} < {a['lt']}"
        return d > a["gt"], f"diff vs {a['vs']} mean={d:.2f} > {a['gt']}"
    if t == "diff_file":
        ref = Image.open(os.path.join(CORPUS, a["file"])).convert("RGB")
        if ref.size != im.size:
            ref = ref.resize(im.size)
        d = ImageStat.Stat(
            ImageChops.difference(rgb(im), ref).convert("L")).mean[0]
        return d < a["lt"], f"diff vs corpus/{a['file']} mean={d:.2f} < {a['lt']}"
    if t == "invert_of":
        base = rgb(img_of(a["cell"]))
        if base.size != im.size:
            base = base.resize(im.size)
        inv = base.point(lambda v: 255 - v)
        d = ImageStat.Stat(
            ImageChops.difference(rgb(im), inv).convert("L")).mean[0]
        return d < 3, f"diff vs inverted {a['cell']} mean={d:.2f} < 3"
    if t == "mean_shift":
        mine = stats(im)["mean"][a["ch"]]
        theirs = stats(img_of(a["vs"]))["mean"][a["ch"]]
        d = mine - theirs
        if "between" in a:
            lo, hi = a["between"]
            return lo <= d <= hi, \
                f"ch{a['ch']} mean shift={d:+.1f} in [{lo},{hi}]"
        if a["dir"] == "up":
            return d >= a["min"], \
                f"ch{a['ch']} mean shift={d:+.1f} >= {a['min']}"
        return d <= -a["min"], \
            f"ch{a['ch']} mean shift={d:+.1f} <= -{a['min']}"
    if t == "std_cmp":
        mine, theirs = lum_std(im), lum_std(img_of(a["vs"]))
        return (mine > theirs) if a["dir"] == "gt" else (mine < theirs), \
            f"lum std {mine:.1f} vs {a['vs']} {theirs:.1f}"
    if t == "std_lte":
        return lum_std(im) <= a["v"], f"lum std={lum_std(im):.2f} <= {a['v']}"
    if t == "sat_cmp":
        mine, theirs = sat_mean(im), sat_mean(img_of(a["vs"]))
        return (mine > theirs) if a["dir"] == "gt" else (mine < theirs), \
            f"sat mean {mine:.1f} vs {a['vs']} {theirs:.1f}"
    if t == "hue_shift":
        h1, h0 = hue_of(im), hue_of(img_of(a["vs"]))
        d = (h1 - h0) % 360
        return a["deg"][0] <= d <= a["deg"][1], \
            f"hue {h0} -> {h1} (shift {d:.0f} deg)"
    if t == "local_var_cmp":
        mine, theirs = local_var(im), local_var(img_of(a["vs"]))
        return (mine > theirs) if a["dir"] == "gt" else (mine < theirs), \
            f"local var {mine:.2f} vs {a['vs']} {theirs:.2f}"
    if t == "channel_order":
        m = stats(im)["mean"]
        return m[0] > m[1] > m[2], f"channel means={m} (want R>G>B)"
    if t == "extreme_near":
        sm = rgb(im)
        if sm.width * sm.height > 2_000_000:
            sm = sm.resize((sm.width // 2, sm.height // 2), Image.NEAREST)
        px = list(sm.getdata())
        lums = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in px]
        p = px[lums.index(min(lums))] if a["which"] == "min" \
            else px[lums.index(max(lums))]
        d = max(abs(p[i] - a["rgb"][i]) for i in range(3))
        return d <= a["tol"], \
            f"{a['which']} pixel={p} want ~{tuple(a['rgb'])} (max ch diff {d})"
    if t == "unique_colors":
        cols = rgb(im).getcolors(1 << 22)
        n = len(cols) if cols else 1 << 22
        return n <= a["lte"], f"unique colors={n} <= {a['lte']}"
    if t == "region_std":
        r = region(rgb(im), a["box"])
        s = ImageStat.Stat(r).stddev[a["ch"]]
        return s > a["gt"], f"region std ch{a['ch']}={s:.1f} > {a['gt']}"
    if t == "region_local_var":
        mine = local_var_box(im, a["box"])
        theirs = local_var_box(img_of(a["vs"]), a["box"])
        return (mine < theirs) if a["dir"] == "lt" else (mine > theirs), \
            f"region local var {mine:.2f} vs {a['vs']} {theirs:.2f}"
    if t == "region_diff":
        other = rgb(img_of(a["vs"]))
        mine = region(rgb(im), a["box"])
        theirs = region(other, a["box"])
        d = ImageStat.Stat(
            ImageChops.difference(mine, theirs).convert("L")).mean[0]
        if "lt" in a:
            return d < a["lt"], f"region diff mean={d:.2f} < {a['lt']}"
        return d > a["gt"], f"region diff mean={d:.2f} > {a['gt']}"
    if t == "region_mean":
        r = region(rgb(im), a["box"])
        m = ImageStat.Stat(r).mean
        ok = all(abs(m[i] - a["rgb"][i]) <= a.get("tol", 8) for i in range(3))
        return ok, \
            f"region mean={[round(v, 1) for v in m]} want ~{a['rgb']} tol {a.get('tol', 8)}"
    if t == "region_alpha_mean":
        r = region(im.convert("RGBA"), a["box"]).getchannel("A")
        m = ImageStat.Stat(r).mean[0]
        return abs(m - a["eq"]) <= a.get("tol", 8), \
            f"region alpha mean={m:.1f} want {a['eq']} tol {a.get('tol', 8)}"
    if t == "bright_count":
        r = rgb(region(im, a["box"]))
        px = list(r.getdata())
        n = sum(1 for p in px if max(p) >= a["thresh"])
        if "gte" in a:
            return n >= a["gte"], f"bright px in {a['box']}={n} >= {a['gte']}"
        return n <= a["lte"], f"bright px in {a['box']}={n} <= {a['lte']}"
    if t == "bright_mean_pred":
        r = rgb(region(im, a["box"]))
        px = [p for p in r.getdata() if max(p) >= a["thresh"]]
        if not px:
            return False, f"no bright px in {a['box']}"
        m = tuple(sum(p[i] for p in px) // len(px) for i in range(3))
        return PREDS[a["pred"]](m), \
            f"bright mean in {a['box']}={m} (n={len(px)}) pred {a['pred']}"
    if t == "bright_mean_val":
        r = rgb(region(im, a["box"]))
        px = [p for p in r.getdata() if max(p) >= a["thresh"]]
        if not px:
            return False, f"no bright px in {a['box']}"
        m = sum(max(p) for p in px) / len(px)
        lo, hi = a["between"]
        return lo <= m <= hi, \
            f"bright mean max-channel={m:.1f} in [{lo},{hi}] (n={len(px)})"
    if t == "bright_bbox":
        sc = rgb(im)
        if "box" in a:
            sc = region(sc, a["box"])
        px = sc.load()
        found = []
        step = a.get("step", 2)
        for yy in range(0, sc.height, step):
            for xx in range(0, sc.width, step):
                if max(px[xx, yy]) >= a["thresh"]:
                    found.append((xx, yy))
        if not found:
            return False, "no bright pixels found"
        xs = [p[0] for p in found]
        ys = [p[1] for p in found]
        bb = {"minx": min(xs), "miny": min(ys),
              "maxx": max(xs), "maxy": max(ys)}
        ok = True
        det = []
        for k, v in bb.items():
            if k in a:
                lo, hi = a[k]
                good = lo <= v <= hi
                ok &= good
                det.append(f"{k}={v} in [{lo},{hi}] {good}")
        return ok, "bbox " + ", ".join(det)
    if t == "alpha_partial":
        r = region(im.convert("RGBA"), a["box"]).getchannel("A")
        vals = set(r.getdata())
        has = any(0 < v < 255 for v in vals)
        return has, f"partial alpha present in {a['box']}: {has}"
    if t == "pixel_eq_file":
        ref = Image.open(os.path.join(CORPUS, a["file"])) \
            .convert("RGBA").getpixel(tuple(a["oxy"]))
        p = rgb(im).getpixel(tuple(a["xy"]))
        ok = all(abs(p[i] - ref[i]) <= a.get("tol", 8) for i in range(3))
        return ok, \
            f"pixel{a['xy']}={p} vs {a['file']}{a['oxy']}={ref[:3]} tol {a.get('tol', 8)}"
    if t == "pixel_blend_file":
        ref = Image.open(os.path.join(CORPUS, a["file"])) \
            .convert("RGBA").getpixel(tuple(a["oxy"]))
        base = rgb(img_of(a["cell"])).getpixel(tuple(a["xy"]))
        eff = (ref[3] / 255.0) * (a["alpha"] / 100.0)
        exp = tuple(round(base[i] * (1 - eff) + ref[i] * eff)
                    for i in range(3))
        p = rgb(im).getpixel(tuple(a["xy"]))
        ok = all(abs(p[i] - exp[i]) <= a.get("tol", 10) for i in range(3))
        return ok, \
            f"pixel{a['xy']}={p} want blend~{exp} (base {base}, ov {ref}, eff {eff:.2f}) tol {a.get('tol', 10)}"
    if t == "pixel_eq_cell":
        other = rgb(img_of(a["cell"])).getpixel(tuple(a["oxy"]))
        p = rgb(im).getpixel(tuple(a["xy"]))
        ok = all(abs(p[i] - other[i]) <= a.get("tol", 8) for i in range(3))
        return ok, \
            f"pixel{a['xy']}={p} vs {a['cell']}{a['oxy']}={other} tol {a.get('tol', 8)}"
    if t == "bbox_height_ratio":
        def bb(img):
            sc = rgb(img)
            px = sc.load()
            found = []
            for yy in range(0, sc.height, 2):
                for xx in range(0, sc.width, 2):
                    if max(px[xx, yy]) >= a["thresh"]:
                        found.append(yy)
            return (max(found) - min(found)) if found else 0
        mine, theirs = bb(im), bb(img_of(a["cell"]))
        if not theirs:
            return False, "reference cell has no bright pixels"
        ratio = mine / theirs
        lo, hi = a["between"]
        return lo <= ratio <= hi, \
            f"bbox height {mine} vs {a['cell']} {theirs}: ratio {ratio:.2f} in [{lo},{hi}]"
    if t == "second_hit":
        s = fact.get("second") or {}
        cs = (s.get("cache_status") or "").lower()
        return "hit" in cs, \
            f"second fetch cache-status={s.get('cache_status')!r} status={s.get('status')}"
    if t == "second_same":
        s = fact.get("second") or {}
        return s.get("md5") == fact.get("md5"), \
            f"second md5 == first md5: {s.get('md5') == fact.get('md5')}"
    if t == "md5_diff_cell":
        other = REG[a["cell"]].get("md5")
        return fact.get("md5") != other, \
            f"md5 differs from {a['cell']}: {fact.get('md5') != other}"
    if t == "deterministic":
        inst = instance_of(case)
        rem = remainder_of(case["ops"], case["src"])
        purge_cli(rem, inst)
        st2, h2, payload2, _ = fetch(f"{BASES[inst]}/unsafe/{rem}")
        same = hashlib.md5(payload2).hexdigest() == \
            hashlib.md5(open(fact["path"], "rb").read()).hexdigest()
        return same, f"re-render after purge byte-identical: {same}"
    if t == "exif_absent":
        raw = open(fact["path"], "rb").read()
        ex = Image.open(fact["path"]).getexif()
        return len(ex) == 0 and b"Wave1Cam" not in raw, \
            f"exif tags={len(ex)}, Make in bytes={b'Wave1Cam' in raw}"
    if t == "exif_orientation":
        ex = Image.open(fact["path"]).getexif()
        return ex.get(274) == a["eq"], \
            f"exif orientation={ex.get(274)} want {a['eq']}"
    if t == "exif_field":
        ex = Image.open(fact["path"]).getexif()
        return ex.get(a["tag"]) == a["eq"], \
            f"exif[{a['tag']}]={ex.get(a['tag'])!r}"
    if t == "icc_eq_source":
        src_icc = Image.open(os.path.join(CORPUS, case["src"])) \
            .info.get("icc_profile")
        out_icc = Image.open(fact["path"]).info.get("icc_profile")
        return out_icc is not None and out_icc == src_icc, \
            f"output icc {'== source' if out_icc == src_icc else 'differs/absent'}"
    if t == "icc_ne_source":
        src_icc = Image.open(os.path.join(CORPUS, case["src"])) \
            .info.get("icc_profile")
        out_icc = Image.open(fact["path"]).info.get("icc_profile")
        return out_icc != src_icc, \
            f"output icc present={out_icc is not None}, equal to source={out_icc == src_icc}"
    return False, f"unknown assertion {t}"


MAGIC = {"jpeg": b"\xff\xd8\xff", "png": b"\x89PNG", "gif": b"GIF8"}


def magic_check(path, fmt):
    with open(path, "rb") as fh:
        head = fh.read(16)
    if fmt == "webp":
        return head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    if fmt == "avif":
        return b"avif" in head or b"avis" in head
    if fmt == "jxl":
        return head[:2] == b"\xff\x0a" or b"jxl" in head
    return head[:3] == MAGIC[fmt][:3]


def channel_mean_of(path):
    im = open_img(path).convert("RGB")
    im.thumbnail((256, 256))
    return [round(m, 2) for m in ImageStat.Stat(im).mean]


def run_dsl_case(case):
    """Execute one data-driven case; record and return its result row."""
    cid = case["id"]
    inst = instance_of(case)
    try:
        if case.get("kind") == "meta":
            accept = case.get("accept", "application/json")
            st, mj = meta_dsl(case["ops"], case["src"], inst,
                              headers={"Accept": accept}
                              if ";" in accept else None)
            if ";" in accept:
                rem = remainder_of(case["ops"], case["src"])
                st, h, payload, ms = fetch(
                    f"{BASES[inst]}/unsafe/{rem}", {"Accept": accept})
                try:
                    mj = json.loads(payload)
                except Exception:
                    mj = {}
                fact = {"cell": cid, "status": st,
                        "ct": h.get("content-type", ""),
                        "bytes": len(payload), "client_ms": round(ms, 1),
                        "server_ms": None, "cache": "n/a (meta)",
                        "x_quality": None, "headers": h, "path": None,
                        "body": "", "md5": hashlib.md5(payload).hexdigest()}
            else:
                fact = {"cell": cid, "status": st,
                        "ct": "application/json",
                        "bytes": len(json.dumps(mj)), "client_ms": None,
                        "server_ms": None, "cache": "n/a (meta)",
                        "x_quality": None, "headers": {}, "path": None,
                        "body": "", "md5": ""}
            case["_meta"] = mj
            REG[cid] = fact
            if any(a["t"] == "meta_bytes_match"
                   for a in case.get("assert", [])):
                rem = remainder_of(case["ops"], case["src"])
                st2, h2, payload2, _ = fetch(
                    f"{BASES[inst]}/unsafe/{rem}")
                fact["bytes_img"] = len(payload2)
        else:
            fact = render_dsl(cid, case["ops"], case["src"], inst,
                              headers=case.get("headers"),
                              accept=case.get("accept"),
                              timeout=case.get("timeout", 180))
            REG[cid] = fact
            if any(a["t"].startswith("second_")
                   for a in case.get("assert", [])):
                rem = remainder_of(case["ops"], case["src"])
                hdrs = dict(case.get("headers") or {})
                if case.get("accept"):
                    hdrs["Accept"] = case["accept"]
                st2, h2, p2, ms2 = fetch(f"{BASES[inst]}/unsafe/{rem}",
                                         hdrs)
                fact["second"] = {
                    "status": st2, "md5": hashlib.md5(p2).hexdigest(),
                    "cache_status": h2.get("cache-status", ""),
                    "client_ms": round(ms2, 1)}
            if any(a["t"] in ("meta", "meta_field_present",
                              "meta_field_len", "meta_bytes_match")
                   for a in case.get("assert", [])):
                _, case["_meta"] = meta_dsl(case["ops"], case["src"], inst)
    except Exception as e:
        row = {"id": cid, "group": case["group"], "ops": case["ops"],
               "src": case["src"], "verdict": "ERROR", "status": None,
               "content_type": "", "bytes": None, "server_ms": None,
               "client_ms": None, "cache": "",
               "evidence": f"runner exception: {type(e).__name__}: {e}",
               "why": case.get("why", "")}
        RESULTS[cid] = row
        if _save_hook:
            _save_hook()
        print(f"[ERR ] {cid}: {e}", flush=True)
        return row
    details, all_ok = [], True
    for a in case.get("assert", []):
        if a["t"] == "magic":
            ok = fact["path"] is not None and \
                magic_check(fact["path"], a["fmt"])
            details.append(f"magic {a['fmt']} ok={ok}")
            all_ok &= ok
            continue
        if a["t"] == "meta_bytes_match" and case.get("kind") == "meta":
            ok = case["_meta"].get("bytes") == fact.get("bytes_img")
            details.append(
                f"meta.bytes={case['_meta'].get('bytes')} vs image "
                f"{fact.get('bytes_img')}")
            all_ok &= ok
            continue
        try:
            ok, det = ev(a, fact, case)
        except Exception as e:
            ok, det = False, f"{a['t']} errored: {e}"
        details.append(det)
        all_ok &= ok
    if case.get("samples_brand") and fact.get("path"):
        with open(fact["path"], "rb") as fh:
            head = fh.read(32)
        brand = head[8:16].decode("ascii", "replace")
        details.append(f"ftyp brand={brand!r} bytes={fact['bytes']}")
    if case.get("samples") and fact.get("path"):
        im = rgb(open_img(fact["path"]))
        pts = {}
        for p in case["samples"]:
            x, y = min(p[0], im.width - 1), min(p[1], im.height - 1)
            pts[(x, y)] = im.getpixel((x, y))
        details.append(f"samples={pts}")
    # ---- baseline band: bytes within +-15%, channel means within tol --
    if fact.get("path") and case.get("kind") != "meta" \
            and not case.get("no_baseline"):
        if BLESS is not None:
            try:
                BLESS[cid] = {"bytes": fact["bytes"],
                              "mean": channel_mean_of(fact["path"])}
            except Exception:
                pass
        elif cid in BASELINES and not case.get("observe"):
            base = BASELINES[cid]
            if base.get("bytes"):
                ratio = fact["bytes"] / base["bytes"]
                ok = 0.85 <= ratio <= 1.15
                details.append(
                    f"bytes {fact['bytes']} vs baseline {base['bytes']} "
                    f"(x{ratio:.3f}, band 0.85..1.15) ok={ok}")
                all_ok &= ok
            if base.get("mean"):
                try:
                    mean = channel_mean_of(fact["path"])
                    ok = all(abs(mean[i] - base["mean"][i]) <= 7
                             for i in range(3))
                    details.append(
                        f"channel mean {mean} vs baseline "
                        f"{base['mean']} (tol 7) ok={ok}")
                    all_ok &= ok
                except Exception as e:
                    details.append(f"channel mean check errored: {e}")
                    all_ok = False
    verdict = "NOTE" if case.get("observe") else \
        ("PASS" if all_ok else "FAIL")
    row = {"id": cid, "group": case["group"], "ops": case["ops"],
           "src": case["src"], "verdict": verdict,
           "status": fact["status"], "content_type": fact["ct"],
           "bytes": fact["bytes"], "server_ms": fact["server_ms"],
           "client_ms": fact["client_ms"], "cache": fact["cache"],
           "evidence": "; ".join(details), "why": case.get("why", "")}
    RESULTS[cid] = row
    if _save_hook:
        _save_hook()
    print(f"[{verdict:4s}] {cid}: {row['evidence'][:150]}", flush=True)
    return row
