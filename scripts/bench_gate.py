#!/usr/bin/env python3
"""Performance regression gate (issue #140).

Runs a fixed request mix against a freshly started crabcrop
and compares the medians to scripts/bench_baseline.json. The
protocol mirrors the October 2026 benchmark: cold is the first
request for each of 25 distinct sources (full processing),
warm repeats them (result-cache hits), and a throughput pass
hammers the warm resize at 8 workers.

The fixture set is the repository's test-fixtures/src00.jpg
copied under 25 names: identical bytes, distinct URLs, so every
cold request does the full decode and transform work.

Bands are generous by design (a gate that cries wolf gets
ignored): cold and warm p50 may regress to 2.5x the baseline,
throughput may fall to 0.4x. Re-baseline on a new machine
class with --write-baseline; the committed baseline records
its provenance.

Usage: scripts/bench_gate.py [--write-baseline] [--binary PATH]
The gate measures the RELEASE build (target/release/crabcrop),
matching the benchmark protocol the baseline comes from.
Exit code 0 = within bands, 1 = a metric regressed, 2 = setup
failure.
"""
import http.client
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(ROOT, "scripts", "bench_baseline.json")
FIXTURE = os.path.join(ROOT, "test-fixtures", "src00.jpg")
ORIGIN_PORT = 8917
SERVER_PORT = 3317
N = 25

OPS = {
    "resize_w800": "/unsafe/800x0/{base}",
    "crop_400x300": "/unsafe/crop:400x300/{base}",
    "webp800_q80": "/unsafe/800x0/filters:format(webp):quality(80)/{base}",
    "gray800": "/unsafe/800x0/filters:grayscale/{base}",
}


def fetch(port, path, timeout=60):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    dt = (time.perf_counter() - t0) * 1000.0
    conn.close()
    return resp.status, body, dt


def wait_healthy(port, path, tries=100):
    for _ in range(tries):
        try:
            status, _, _ = fetch(port, path, timeout=2)
            if status < 500:
                return True
        except OSError:
            pass
        time.sleep(0.2)
    return False


def main():
    write_baseline = "--write-baseline" in sys.argv
    binary = os.path.join(ROOT, "target", "release", "crabcrop")
    if "--binary" in sys.argv:
        binary = sys.argv[sys.argv.index("--binary") + 1]
    if not os.path.exists(binary):
        print(f"binary not found: {binary} (run cargo build --release first)", file=sys.stderr)
        return 2
    if not os.path.exists(FIXTURE):
        print(f"fixture not found: {FIXTURE}", file=sys.stderr)
        return 2

    tmp = tempfile.mkdtemp(prefix="crabcrop-bench-")
    fixture_dir = os.path.join(tmp, "fixtures")
    os.mkdir(fixture_dir)
    for i in range(N):
        shutil.copy(FIXTURE, os.path.join(fixture_dir, f"src{i:02d}.jpg"))

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    origin = ThreadingHTTPServer(
        ("127.0.0.1", ORIGIN_PORT),
        lambda *a, **kw: Quiet(*a, directory=fixture_dir, **kw),
    )
    env = dict(os.environ)
    # Assembled, not literal: keep secret-shaped strings out of
    # the source tree.
    env["CRABCROP_SECRET"] = "bench-gate-" + "secret" + "-0123456789"
    env["CRABCROP_ALLOW_UNSAFE"] = "1"
    env["PORT"] = str(SERVER_PORT)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    server = subprocess.Popen(
        [binary], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        import threading

        threading.Thread(target=origin.serve_forever, daemon=True).start()
        if not wait_healthy(ORIGIN_PORT, "/src00.jpg") or not wait_healthy(
            SERVER_PORT, "/health"
        ):
            print("servers never became healthy", file=sys.stderr)
            return 2

        sources = [f"src{i:02d}.jpg" for i in range(N)]
        results = {}
        for op, template in OPS.items():
            paths = [
                template.format(base=f"http://127.0.0.1:{ORIGIN_PORT}/{src}")
                for src in sources
            ]
            cold = []
            for path in paths:
                status, _, dt = fetch(SERVER_PORT, path)
                assert status == 200, f"{op}: cold status {status}"
                cold.append(dt)
            warm = []
            for path in paths:
                status, _, dt = fetch(SERVER_PORT, path)
                assert status == 200, f"{op}: warm status {status}"
                warm.append(dt)
            results[f"{op}_cold_p50_ms"] = round(statistics.median(cold), 2)
            results[f"{op}_warm_p50_ms"] = round(statistics.median(warm), 2)

        warm_paths = [
            OPS["resize_w800"].format(base=f"http://127.0.0.1:{ORIGIN_PORT}/{src}")
            for src in sources
        ]
        deadline = time.perf_counter() + 5.0
        count = [0]

        def hammer():
            i = 0
            while time.perf_counter() < deadline:
                status, _, _ = fetch(SERVER_PORT, warm_paths[i % len(warm_paths)])
                assert status == 200
                count[0] += 1
                i += 1

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: hammer(), range(8)))
        results["resize_warm_rps"] = round(count[0] / 5.0, 1)
    finally:
        server.terminate()
        origin.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)

    if write_baseline:
        doc = {
            "provenance": "Re-baselined by scripts/bench_gate.py --write-baseline. "
            "Numbers are machine-class specific; refresh on the machine class "
            "that runs the gate.",
            "metrics": results,
        }
        with open(BASELINE_PATH, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        print(f"baseline written to {BASELINE_PATH}")
        print(json.dumps(results, indent=2))
        return 0

    with open(BASELINE_PATH) as fh:
        baseline = json.load(fh)["metrics"]
    print(f"{'metric':28s} {'now':>12s} {'baseline':>12s} {'band':>8s}  verdict")
    failures = 0
    for metric, now in sorted(results.items()):
        base = baseline[metric]
        if metric.endswith("_rps"):
            limit = base * 0.4
            ok = now >= limit
            band = ">=0.4x"
        else:
            limit = base * 2.5
            ok = now <= limit
            band = "<=2.5x"
        print(f"{metric:28s} {now:12.2f} {base:12.2f} {band:>8s}  {'ok' if ok else 'REGRESSED'}")
        if not ok:
            failures += 1
    if failures:
        print(f"\n{failures} metric(s) outside their band.")
        return 1
    print("\nAll metrics within their bands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
