#!/usr/bin/env python3
"""crabcrop vs Thumbor vs imgproxy: same origin, same operations, sequential latency.

Cold = first request for each of 25 distinct sources (result caches miss,
source is fetched and processed). Warm = the same 25 again immediately
(each product answers from its own result caching if it has any).
Plus a warm-throughput pass at 8 workers for the resize op.
Everything runs one service at a time on this host.

The harness starts and stops each service itself. Binaries and configs
come from setup.sh and configs/ (see README.md). Set BENCH_SKIP_SERVICES
to a comma list (crabcrop,thumbor,imgproxy) to run a subset while
debugging; a full published run leaves it unset.

Fix vs the October harness: validation fetches use a throwaway size that
no measured URL uses, so no cold sample is pre-warmed. October noted
that blemish in its write-up; this harness removes it.

Ports: origin 8911, crabcrop 3310, Thumbor 8888, imgproxy 8080.
"""
import base64
import http.client
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.environ.get("BENCH_WORK", os.path.join(HERE, ".work"))
FIXTURES = os.path.join(HERE, "fixtures")
ORIGIN_PORT = 8911
N = 25

# Operation set. Order matters for readability of the results JSON only.
OPS = [
    "resize_w800",
    "crop_400x300",
    "webp800_q80",
    "gray800",
    "avif800_q80",
    "jxl800_q80",
    "jpeg800_q60",
    "jpeg800_q95",
    "webp800_q60",
    "webp800_q95",
]

# Expected Content-Type prefix per op, used to mark unsupported ops
# instead of recording an error page as a measurement.
OP_CONTENT_TYPE = {
    "resize_w800": "image/jpeg",
    "crop_400x300": "image/jpeg",
    "webp800_q80": "image/webp",
    "gray800": "image/jpeg",
    "avif800_q80": "image/avif",
    "jxl800_q80": "image/jxl",
    "jpeg800_q60": "image/jpeg",
    "jpeg800_q95": "image/jpeg",
    "webp800_q60": "image/webp",
    "webp800_q95": "image/webp",
}


def origin_url(src):
    return f"http://127.0.0.1:{ORIGIN_PORT}/{src}"


def crab_url(op, src):
    base = origin_url(src)
    return {
        "resize_w800": f"/unsafe/800x0/{base}",
        "crop_400x300": f"/unsafe/crop:400x300/{base}",
        "webp800_q80": f"/unsafe/800x0/filters:format(webp):quality(80)/{base}",
        "gray800": f"/unsafe/800x0/filters:grayscale/{base}",
        "avif800_q80": f"/unsafe/800x0/filters:format(avif):quality(80)/{base}",
        "jxl800_q80": f"/unsafe/800x0/filters:format(jxl):quality(80)/{base}",
        "jpeg800_q60": f"/unsafe/800x0/filters:quality(60)/{base}",
        "jpeg800_q95": f"/unsafe/800x0/filters:quality(95)/{base}",
        "webp800_q60": f"/unsafe/800x0/filters:format(webp):quality(60)/{base}",
        "webp800_q95": f"/unsafe/800x0/filters:format(webp):quality(95)/{base}",
    }[op]


def thumb_url(op, src):
    base = origin_url(src)
    return {
        "resize_w800": f"/unsafe/800x0/{base}",
        "crop_400x300": f"/unsafe/400x300/{base}",
        "webp800_q80": f"/unsafe/800x0/filters:format(webp):quality(80)/{base}",
        "gray800": f"/unsafe/800x0/filters:grayscale()/{base}",
        # Thumbor core has no AV1 or JPEG XL encoder: unsupported, not faked.
        "avif800_q80": None,
        "jxl800_q80": None,
        "jpeg800_q60": f"/unsafe/800x0/filters:quality(60)/{base}",
        "jpeg800_q95": f"/unsafe/800x0/filters:quality(95)/{base}",
        "webp800_q60": f"/unsafe/800x0/filters:format(webp):quality(60)/{base}",
        "webp800_q95": f"/unsafe/800x0/filters:format(webp):quality(95)/{base}",
    }[op]


def imgproxy_url(op, src):
    # Plain (unsigned) URL form: /insecure/<options>/plain/<url>@<ext>
    # rs:fit:800:0 resizes to width 800 keeping aspect. rs:fill:400:300
    # scales to cover and center-crops to exactly 400x300.
    base = origin_url(src)
    opts = {
        "resize_w800": ("rs:fit:800:0", "jpg"),
        "crop_400x300": ("rs:fill:400:300", "jpg"),
        "webp800_q80": ("rs:fit:800:0/q:80", "webp"),
        # imgproxy OSS has no grayscale or saturation option at all
        # (desaturation lives in imgproxy Pro): unsupported, not faked.
        "gray800": None,
        "avif800_q80": ("rs:fit:800:0/q:80", "avif"),
        "jxl800_q80": ("rs:fit:800:0/q:80", "jxl"),
        "jpeg800_q60": ("rs:fit:800:0/q:60", "jpg"),
        "jpeg800_q95": ("rs:fit:800:0/q:95", "jpg"),
        "webp800_q60": ("rs:fit:800:0/q:60", "webp"),
        "webp800_q95": ("rs:fit:800:0/q:95", "webp"),
    }[op]
    if opts is None:
        return None
    return f"/insecure/{opts[0]}/plain/{base}@{opts[1]}"


def fetch(port, path, timeout=120):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    dt = (time.perf_counter() - t0) * 1000.0
    conn.close()
    return resp.status, resp.getheader("Content-Type") or "", body, dt


def stats(samples):
    s = sorted(samples)
    return {
        "mean_ms": round(statistics.mean(s), 1),
        "p50_ms": round(s[len(s) // 2], 1),
        "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 1),
        "min_ms": round(s[0], 1),
        "max_ms": round(s[-1], 1),
    }


def wait_healthy(port, path, tries=120):
    for _ in range(tries):
        try:
            status, _, _, _ = fetch(port, path, timeout=3)
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_service(name, port, urlfn, warmup_path, start, stop, health="/health"):
    print(f"== {name} ==", flush=True)
    proc = start()
    try:
        if not wait_healthy(port, health):
            print(f"  !! {name} never became healthy", flush=True)
            return {"error": "unhealthy"}
        # Warm up code paths on a throwaway size so no measured URL is cached.
        for i in range(3):
            try:
                fetch(port, warmup_path(f"src{i:02d}.jpg"))
            except Exception as e:
                print("  warmup:", e, flush=True)
        out = {}
        for op in OPS:
            if urlfn(op, "src00.jpg") is None:
                out[op] = {"unsupported": "no encoder or filter for this operation"}
                print(f"  {op}: UNSUPPORTED (by construction)", flush=True)
                continue
            srcs = [f"src{i:02d}.jpg" for i in range(N)]
            cold, sizes, warm = [], [], []
            err = None
            for s in srcs:
                st, ct, body, dt = fetch(port, urlfn(op, s))
                if st != 200 or not ct.startswith(OP_CONTENT_TYPE[op]):
                    err = f"status {st} ct {ct}"
                    break
                cold.append(dt)
                sizes.append(len(body))
            if err:
                out[op] = {"unsupported": err}
                print(f"  {op}: UNSUPPORTED ({err})", flush=True)
                continue
            for s in srcs:
                st, _, _, dt = fetch(port, urlfn(op, s))
                assert st == 200, (op, s, st)
                warm.append(dt)
            out[op] = {
                "cold": stats(cold),
                "warm": stats(warm),
                "avg_bytes": int(statistics.mean(sizes)),
            }
            print(
                f"  {op}: cold p50 {out[op]['cold']['p50_ms']}ms mean {out[op]['cold']['mean_ms']}ms | "
                f"warm p50 {out[op]['warm']['p50_ms']}ms | avg {out[op]['avg_bytes']} B",
                flush=True,
            )
        # Warm throughput on the resize op, 8 concurrent workers.
        urls = [urlfn("resize_w800", f"src{i:02d}.jpg") for i in range(N)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(lambda u: fetch(port, u), urls))
        total = time.perf_counter() - t0
        assert all(r[0] == 200 for r in results)
        out["throughput_warm_resize"] = {
            "requests": N,
            "workers": 8,
            "total_s": round(total, 2),
            "req_per_s": round(N / total, 1),
        }
        print(f"  throughput warm resize: {out['throughput_warm_resize']['req_per_s']} req/s", flush=True)
        return out
    finally:
        stop(proc)
        time.sleep(1)


def main():
    os.makedirs(WORK, exist_ok=True)
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = lambda *a, **kw: QuietHandler(*a, directory=FIXTURES, **kw)
    origin = ThreadingHTTPServer(("127.0.0.1", ORIGIN_PORT), handler)
    threading.Thread(target=origin.serve_forever, daemon=True).start()
    print(f"origin on :{ORIGIN_PORT} serving {FIXTURES}", flush=True)

    skip = {s.strip() for s in os.environ.get("BENCH_SKIP_SERVICES", "").split(",") if s.strip()}
    results = {
        "meta": {
            "sources": N,
            "source_dims": "2400x1600 JPEG, 25 files, see fixtures/",
            "origin_port": ORIGIN_PORT,
            "ops": OPS,
        }
    }

    if "crabcrop" not in skip:
        crab_bin = os.environ.get(
            "CRABCROP_BIN", os.path.join(WORK, "crabcrop-v0.9.5", "target", "release", "crabcrop")
        )

        def start_crab():
            env = dict(
                os.environ,
                CRABCROP_SECRET="bench-secret-0123456789",
                CRABCROP_ALLOW_UNSAFE="1",
                PORT="3310",
            )
            return subprocess.Popen(
                [crab_bin], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

        results["crabcrop"] = run_service(
            "crabcrop", 3310, crab_url,
            lambda s: f"/unsafe/17x17/{origin_url(s)}",
            start_crab, lambda p: p.terminate(),
        )

    if "thumbor" not in skip:
        thumbor_bin = os.environ.get("THUMBOR_BIN", os.path.join(WORK, "thumbor-venv", "bin", "thumbor"))
        conf = os.path.join(HERE, "configs", "thumbor.conf")
        # Fresh result storage so the cold pass is honest.
        subprocess.run(["rm", "-rf", "/tmp/thumbor-bench-storage"], check=False)

        def start_thumb():
            return subprocess.Popen(
                [thumbor_bin, "-c", conf, "-i", "127.0.0.1", "-p", "8888"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        results["thumbor"] = run_service(
            "thumbor", 8888, thumb_url,
            lambda s: f"/unsafe/17x17/{origin_url(s)}",
            start_thumb, lambda p: p.terminate(), health="/healthcheck",
        )

    if "imgproxy" not in skip:
        imgproxy_bin = os.environ.get("IMGPROXY_BIN", os.path.join(WORK, "imgproxy"))
        env_file = os.path.join(HERE, "configs", "imgproxy.env")

        def start_imgproxy():
            env = dict(os.environ)
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env[k.strip()] = v.strip()
            return subprocess.Popen(
                [imgproxy_bin], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

        results["imgproxy"] = run_service(
            "imgproxy", 8080, imgproxy_url,
            lambda s: f"/insecure/rs:fit:17:17/plain/{origin_url(s)}@jpg",
            start_imgproxy, lambda p: p.terminate(),
        )

    origin.shutdown()
    out_path = os.environ.get("BENCH_OUT", os.path.join(WORK, "results.json"))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
