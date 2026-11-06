#!/usr/bin/env python3
"""Cache behavior scenarios for crabcrop: churn and memory accounting.

The comparison harness (bench.py) measures cold and warm latency on
a working set that fits in the result cache. This script measures
the two things that set does not reach, against crabcrop alone
(Thumbor and imgproxy run with no result cache in the comparison
configs, so there is nothing equivalent to measure there):

- Churn: a rotating working set several times larger than a small
  cache budget, so the cache turns over constantly. Reports the
  hit rate the rotation actually gets, miss latency under churn
  against a cache-disabled baseline (the difference is the
  insert and evict work on the miss path), warm-hit latency, and
  the cost of per-key purges interleaved with the workload.

- Memory accounting: the cache budget bounds payload bytes only.
  These scenarios fill the cache with known payloads and compare
  the process RSS growth against the payload total, for many
  small entries, for entries carrying a cache group, and for a
  few large entries, so the memory held by keys, group names,
  and the cache's own indexes is measured rather than assumed.
  A budget-plateau run then drives several budgets' worth of
  small entries through a small budget and watches RSS level
  off, and a purge run measures what the purge tombstone maps
  retain.

Hit and miss are read from the response `Cache-Status` header
(RFC 9875), counted client side; the server's /metrics counters
are scraped once as a cross-check. RSS is the server process's
VmRSS from /proc. Purges are signed with the repository's own
Python SDK (sdks/python), the same helpers an operator uses.

Ports: origin 8911, crabcrop 3310 (same as bench.py; run the
two scripts sequentially, not at once). The binary comes from
CRABCROP_BIN, defaulting to the release build setup.sh prepares.
Raw results land in benchmark/.work/cache-results.json (or
BENCH_OUT). Set BENCH_SKIP_SCENARIOS to a comma list (churn,
memory) to run a subset while debugging.
"""
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
ORIGIN_DIR = os.path.join(WORK, "origin")
ORIGIN_PORT = 8911
PORT = 3310
SECRET = "bench-secret-0123456789"
METRICS_TOKEN = "bench-metrics-token"

sys.path.insert(0, os.path.join(REPO, "sdks", "python", "src"))
import crabcrop_url  # noqa: E402


def origin_url(src):
    return f"http://127.0.0.1:{ORIGIN_PORT}/{src}"


def resize_url(src, width):
    return f"/unsafe/{width}x0/{origin_url(src)}"


def stats(samples):
    s = sorted(samples)
    return {
        "mean_ms": round(statistics.mean(s), 1),
        "p50_ms": round(s[len(s) // 2], 1),
        "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 1),
        "min_ms": round(s[0], 1),
        "max_ms": round(s[-1], 1),
    }


def fetch(path, method="GET", port=PORT, timeout=120):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request(method, path)
    resp = conn.getresponse()
    body = resp.read()
    dt = (time.perf_counter() - t0) * 1000.0
    headers = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    return resp.status, headers, body, dt


def is_hit(headers):
    return headers.get("cache-status", "").strip().endswith("hit")


def rss_kib(pid):
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS not found")


def wait_healthy(tries=120):
    for _ in range(tries):
        try:
            status, _, _, _ = fetch("/health", timeout=3)
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


class CrabServer:
    """One crabcrop process with a scenario-specific environment."""

    def __init__(self, binary, **env_extra):
        env = dict(
            os.environ,
            CRABCROP_SECRET=SECRET,
            CRABCROP_ALLOW_UNSAFE="1",
            # 0.9.9 refuses non-public sources by default; the
            # benchmark origin is loopback, so allowlist it.
            CRABCROP_ALLOWED_SOURCES="127.0.0.1",
            CRABCROP_METRICS_TOKEN=METRICS_TOKEN,
            PORT=str(PORT),
        )
        env.update({k: str(v) for k, v in env_extra.items()})
        self.proc = subprocess.Popen(
            [binary], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if not wait_healthy():
            self.stop()
            raise RuntimeError("crabcrop never became healthy")

    def rss_kib(self):
        return rss_kib(self.proc.pid)

    def warmup(self):
        # Throwaway size, as in bench.py, so no measured URL is cached.
        for i in range(3):
            fetch(f"/unsafe/17x17/{origin_url(f'src{i:02d}.jpg')}")

    def metrics(self):
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
        conn.request("GET", "/metrics", headers={"Authorization": f"Bearer {METRICS_TOKEN}"})
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        out = {}
        for line in body.splitlines():
            for key in ("crabcrop_cache_hits_total", "crabcrop_cache_misses_total"):
                if line.startswith(key + " "):
                    out[key] = int(float(line.split()[1]))
        return out

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        time.sleep(0.5)


def run_pass(urls, workers=1):
    """One pass over urls; returns per-request records."""
    def one(u):
        status, headers, body, dt = fetch(u)
        assert status == 200, (u, status)
        return {"dt": dt, "hit": is_hit(headers), "bytes": len(body)}

    if workers == 1:
        return [one(u) for u in urls]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, urls))


def summarize(records):
    hits = [r["dt"] for r in records if r["hit"]]
    misses = [r["dt"] for r in records if not r["hit"]]
    out = {
        "requests": len(records),
        "hits": len(hits),
        "hit_rate": round(len(hits) / len(records), 3) if records else 0,
    }
    if hits:
        out["hit_latency"] = stats(hits)
    if misses:
        out["miss_latency"] = stats(misses)
    return out


def scenario_churn(binary):
    print("== churn ==", flush=True)
    srcs = [f"src{i:02d}.jpg" for i in range(25)]
    widths = list(range(300, 348))
    urls = [resize_url(s, w) for w in widths for s in srcs]
    out = {"distinct_urls": len(urls)}

    # Fill once on the measured budget to learn the payload size
    # and the capacity the budget actually holds.
    server = CrabServer(binary, CRABCROP_CACHE_MAX_MB=4)
    try:
        server.warmup()
        fill = run_pass(urls)
        payload = sum(r["bytes"] for r in fill)
        out["working_set_payload_bytes"] = payload
        out["fill"] = summarize(fill)
        # Probe what survived: immediate re-request of the first
        # and last 50 tells where the LRU boundary fell.
        probe_first = run_pass(urls[:50])
        probe_last = run_pass(urls[-50:])
        out["probe_first50_after_fill"] = summarize(probe_first)
        out["probe_last50_after_fill"] = summarize(probe_last)
        # Sequential rotation: the LRU worst case, three passes.
        out["rotation_sequential"] = [summarize(run_pass(urls)) for _ in range(3)]
        # Concurrent rotation at the comparison harness's width.
        conc = []
        for _ in range(2):
            t0 = time.perf_counter()
            recs = run_pass(urls, workers=8)
            total = time.perf_counter() - t0
            s = summarize(recs)
            s["req_per_s"] = round(len(urls) / total, 1)
            conc.append(s)
        out["rotation_8_workers"] = conc
        # Purge cost against a live cache. Rotation has evicted
        # the original set, so first make a control set and a
        # target set resident (both miss, both are stored as
        # most recently used), then purge only the targets.
        control, targets = urls[600:650], urls[650:700]
        run_pass(control + targets)
        purge_dts, removed = [], 0
        for u in targets:
            target = u[len("/unsafe/"):]
            sig = crabcrop_url.purge_signature(SECRET, target)
            status, _, body, dt = fetch(f"/purge/{sig}/{target}", method="POST")
            assert status == 200, (status, body)
            removed += int(body.decode().split()[1])
            purge_dts.append(dt)
        out["purge_50_keys"] = {"latency": stats(purge_dts), "removed_total": removed}
        out["purged_keys_recheck"] = summarize(run_pass(targets))
        out["control_keys_recheck"] = summarize(run_pass(control))
        out["metrics_crosscheck"] = server.metrics()
    finally:
        server.stop()

    # Cache-disabled baseline over the same set: a miss here is
    # pure fetch and render, with no insert or eviction work, so
    # churn miss latency minus this is the turnover overhead.
    server = CrabServer(binary, CRABCROP_CACHE_MAX_MB=0)
    try:
        server.warmup()
        base = run_pass(urls)
        out["cache_disabled_baseline"] = summarize(base)
    finally:
        server.stop()
    print(json.dumps(out)[:400], "...", flush=True)
    return out


def fill_distinct(server, urls):
    """Request every url once; returns (payload_bytes, records)."""
    recs = run_pass(urls)
    return sum(r["bytes"] for r in recs), recs


def memory_fill(binary, label, urls, budget_mb=256, env_extra=None):
    server = CrabServer(binary, CRABCROP_CACHE_MAX_MB=budget_mb, **(env_extra or {}))
    try:
        server.warmup()
        idle = server.rss_kib()
        payload, recs = fill_distinct(server, urls)
        time.sleep(1.5)
        after = server.rss_kib()
        result = {
            "entries": len(urls),
            "payload_bytes": payload,
            "rss_idle_kib": idle,
            "rss_after_kib": after,
            "rss_delta_kib": after - idle,
            "overhead_bytes_per_entry": round(
                ((after - idle) * 1024 - payload) / len(urls)
            ),
            "fill_summary": summarize(recs),
        }
        print(f"  {label}: {json.dumps(result)}", flush=True)
        return server, result
    except Exception:
        server.stop()
        raise


def scenario_memory(binary):
    print("== memory accounting ==", flush=True)
    out = {}
    srcs = [f"src{i:02d}.jpg" for i in range(25)]
    many = [f"m{i:03d}.jpg" for i in range(200)]

    # Many small entries, no groups: widths 100..219 over the
    # 25 fixtures is 3000 distinct renders of a few KB each.
    small = [resize_url(s, w) for w in range(100, 220) for s in srcs]
    server, res = memory_fill(binary, "small, no groups", small)
    out["small_no_groups"] = res
    server.stop()

    # The same shape through a preset, so every entry also
    # carries a group name: 200 sources x 15 chained widths.
    presets = os.path.join(WORK, "presets.txt")
    with open(presets, "w") as f:
        f.write("thumb = 200x0\n")
    grouped = [
        f"/unsafe/preset:thumb/{w}x0/{origin_url(s)}"
        for w in range(150, 165)
        for s in many
    ]
    server, res = memory_fill(
        binary, "small, one group", grouped,
        env_extra={"CRABCROP_PRESETS_FILE": presets},
    )
    out["small_one_group"] = res
    # Group purge cost at this scale, and what it hands back.
    sig = crabcrop_url.purge_group_signature(SECRET, "thumb")
    status, _, body, dt = fetch(f"/purge-group/{sig}/thumb", method="POST")
    assert status == 200, (status, body)
    time.sleep(1.5)
    out["group_purge_3000"] = {
        "latency_ms": round(dt, 1),
        "body": body.decode().strip(),
        "rss_after_purge_kib": server.rss_kib(),
    }
    print(f"  group purge: {json.dumps(out['group_purge_3000'])}", flush=True)
    server.stop()

    # A few large entries: payload dominates, bookkeeping fades.
    # Measured as two back-to-back batches inside one server,
    # after a throwaway large render has warmed the encode path:
    # a bare before/after RSS difference at this entry count is
    # under this host's RSS noise (shared pages move by
    # megabytes), so the second batch's growth over the first
    # is the number that bounds the per-entry overhead.
    server = CrabServer(binary, CRABCROP_CACHE_MAX_MB=256)
    try:
        server.warmup()
        fetch(f"/unsafe/1601x0/filters:format(webp):quality(80)/{origin_url(srcs[0])}")
        time.sleep(1.5)
        idle = server.rss_kib()
        batches = []
        prev_rss, prev_payload = idle, 0
        for widths in ((1600, 1599, 1598, 1597), (1596, 1595, 1594, 1593)):
            urls = [
                f"/unsafe/{w}x0/filters:format(webp):quality(80)/{origin_url(s)}"
                for w in widths
                for s in srcs
            ]
            payload, recs = fill_distinct(server, urls)
            time.sleep(1.5)
            now = server.rss_kib()
            batches.append({
                "entries": len(urls),
                "payload_bytes": payload,
                "rss_delta_kib": now - prev_rss,
                "overhead_bytes_per_entry": round(
                    ((now - prev_rss) * 1024 - payload) / len(urls)
                ),
                "fill_summary": summarize(recs),
            })
            prev_rss, prev_payload = now, payload
        out["large"] = {"rss_idle_kib": idle, "batches": batches}
        print(f"  large batches: {json.dumps(batches)}", flush=True)
    finally:
        server.stop()

    # Budget plateau: several budgets' worth of the small set
    # through a 4 MiB budget, RSS sampled along the way.
    server = CrabServer(binary, CRABCROP_CACHE_MAX_MB=4)
    try:
        server.warmup()
        idle = server.rss_kib()
        samples = []
        payload_seen = 0
        for i in range(0, len(small), 250):
            chunk = run_pass(small[i:i + 250])
            payload_seen += sum(r["bytes"] for r in chunk)
            samples.append({"after_requests": i + len(chunk), "rss_kib": server.rss_kib()})
        time.sleep(1.5)
        first = run_pass(small[:100])
        last = run_pass(small[-100:])
        # A second full rotation over the same set: entry count
        # is already capped by the budget, so any further RSS
        # growth here is allocator behavior under turnover, not
        # more entries. This is what separates "levels off" from
        # "grows forever".
        samples2 = []
        for i in range(0, len(small), 250):
            run_pass(small[i:i + 250])
            samples2.append({"after_requests": i + 250, "rss_kib": server.rss_kib()})
        time.sleep(1.5)
        out["budget_plateau_4mb"] = {
            "budget_bytes": 4 * 1024 * 1024,
            "payload_presented_bytes": payload_seen,
            "rss_idle_kib": idle,
            "rss_samples": samples,
            "rss_samples_second_rotation": samples2,
            "rss_final_kib": server.rss_kib(),
            "first100_recheck": summarize(first),
            "last100_recheck": summarize(last),
        }
        print(f"  plateau: final RSS {server.rss_kib()} KiB, "
              f"first100 hits {sum(1 for r in first if r['hit'])}, "
              f"last100 hits {sum(1 for r in last if r['hit'])}", flush=True)
    finally:
        server.stop()

    # Purge tombstones: thousands of purges of targets that were
    # never stored. The guard's maps are capped per kind, so the
    # retained memory must level off instead of growing per purge.
    server = CrabServer(binary, CRABCROP_CACHE_MAX_MB=256)
    try:
        server.warmup()
        idle = server.rss_kib()
        dts = []
        for i in range(3000):
            target = f"{1000 + i}x0/{origin_url(f'src{i % 25:02d}.jpg')}"
            sig = crabcrop_url.purge_signature(SECRET, target)
            status, _, _, dt = fetch(f"/purge/{sig}/{target}", method="POST")
            assert status == 200
            dts.append(dt)
            if i % 2 == 0:
                g = f"benchgroup{i}"
                gsig = crabcrop_url.purge_group_signature(SECRET, g)
                status, _, _, dt = fetch(f"/purge-group/{gsig}/{g}", method="POST")
                assert status == 200
                dts.append(dt)
        time.sleep(1.5)
        out["purge_tombstones"] = {
            "purges": 4500,
            "latency": stats(dts),
            "rss_idle_kib": idle,
            "rss_after_kib": server.rss_kib(),
            "rss_delta_kib": server.rss_kib() - idle,
        }
        print(f"  tombstones: {json.dumps(out['purge_tombstones'])}", flush=True)
    finally:
        server.stop()
    return out


def prepare_origin():
    """Origin directory: the 25 fixtures plus 200 copies under
    distinct names, so grouped fills can have many sources."""
    os.makedirs(ORIGIN_DIR, exist_ok=True)
    import shutil
    for i in range(25):
        dst = os.path.join(ORIGIN_DIR, f"src{i:02d}.jpg")
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(FIXTURES, f"src{i:02d}.jpg"), dst)
    for i in range(200):
        dst = os.path.join(ORIGIN_DIR, f"m{i:03d}.jpg")
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(FIXTURES, f"src{i % 25:02d}.jpg"), dst)


def main():
    binary = os.environ.get(
        "CRABCROP_BIN", os.path.join(WORK, "crabcrop-v0.9.9", "target", "release", "crabcrop")
    )
    if not os.path.exists(binary):
        print(f"crabcrop binary not found: {binary} (run setup.sh first)", file=sys.stderr)
        return 2
    prepare_origin()

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = lambda *a, **kw: QuietHandler(*a, directory=ORIGIN_DIR, **kw)
    origin = ThreadingHTTPServer(("127.0.0.1", ORIGIN_PORT), handler)
    threading.Thread(target=origin.serve_forever, daemon=True).start()
    print(f"origin on :{ORIGIN_PORT} serving {ORIGIN_DIR}", flush=True)

    skip = {s.strip() for s in os.environ.get("BENCH_SKIP_SCENARIOS", "").split(",") if s.strip()}
    results = {"meta": {"binary": binary, "origin_port": ORIGIN_PORT}}
    if "churn" not in skip:
        results["churn"] = scenario_churn(binary)
    if "memory" not in skip:
        results["memory"] = scenario_memory(binary)
    origin.shutdown()
    out_path = os.environ.get("BENCH_OUT", os.path.join(WORK, "cache-results.json"))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
