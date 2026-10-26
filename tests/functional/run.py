#!/usr/bin/env python3
"""Functional test runner for crabcrop (issue #232).

Usage:
  python3 run.py --tier smoke|full|local [options]

Tiers:
  smoke   PR tier. The top operations across every area (core imaging,
          overlays/geometry, presets/purge/explain/upload basics, SDK
          parity vectors). Hard budget: about 4 minutes wall clock
          including server boots (enforced with --enforce-budget).
  full    The whole CI matrix: every case in cases_core.json,
          cases_overlay.json and cases_platform.json. Nightly and on
          release tags; also on PRs whose diff touches Cargo.lock or
          the encode pipeline (see docs/functional-workflow.yml).
  local   The AI tier (cases_ai.json). Never runs in CI: models are
          deployment artifacts. Needs CRABCROP_FUNC_MODELS_DIR and
          CRABCROP_ORT_DYLIB; without them every AI case is recorded
          as a NOTE (skipped) and the run still succeeds.

Options:
  --bin PATH        crabcrop binary (default: build debug, use
                    target/debug/crabcrop)
  --release         build/use the release binary instead
  --no-build        do not build; --bin must point at a real binary
  --out DIR         results directory (default: tests/functional/out)
  --ids a,b         run only these case ids
  --groups a,b      run only these groups
  --bless           re-bless baselines.json from this run (full tier)
  --enforce-budget  fail the run if the smoke tier exceeds 240 s
  --keep-servers    leave the booted instances running at the end

Results land in <out>/results.json, results.csv, summary.md and
observations.jsonl after every case, so a crashed run keeps its data.
Timings are collected as trend data in the summary; they never gate.
Exit code is 1 when any case verdict is FAIL or ERROR, else 0.
"""
import argparse, csv, json, os, statistics, subprocess, sys, time
import urllib.parse

FUNC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FUNC_DIR)
import harness as hz

SMOKE_BUDGET_S = 240


def load_cases(name):
    with open(os.path.join(FUNC_DIR, name)) as f:
        return json.load(f)


def substitute(case):
    """Fill the origin placeholders in a DSL case's ops string."""
    enc = urllib.parse.quote(hz.ORIGIN, safe="")
    case = dict(case)
    if case.get("ops"):
        case["ops"] = case["ops"].replace("{ORIGIN_ENC}", enc) \
                                  .replace("{ORIGIN_HOST}", hz.ORIGIN)
    return case


def save_outputs(out_dir):
    rows = list(hz.RESULTS.values())
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(hz.RESULTS, f, indent=1)
    cols = ["id", "group", "ops", "src", "verdict", "status",
            "content_type", "bytes", "server_ms", "client_ms", "cache",
            "evidence"]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def group_stats(rows):
    groups = {}
    for r in rows:
        g = groups.setdefault(r["group"], {"rows": [], "server_ms": []})
        g["rows"].append(r)
        if isinstance(r.get("server_ms"), (int, float)):
            g["server_ms"].append(r["server_ms"])
    return groups


def write_summary(out_dir, tier, wall_s, boot_s, bin_version):
    rows = list(hz.RESULTS.values())
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    lines = [f"## Functional tests: {tier} tier", "",
             f"Binary: `{bin_version}` · wall clock {wall_s:.1f} s "
             f"(server boots {boot_s:.1f} s) · "
             + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())),
             "",
             "| group | cases | pass | fail | note | error | "
             "median server ms | p95 server ms |",
             "|---|---|---|---|---|---|---|---|"]
    for gname, g in sorted(group_stats(rows).items()):
        c = {}
        for r in g["rows"]:
            c[r["verdict"]] = c.get(r["verdict"], 0) + 1
        ms = sorted(g["server_ms"])
        med = f"{statistics.median(ms):.1f}" if ms else "-"
        p95 = f"{ms[min(len(ms) - 1, int(len(ms) * 0.95))]:.1f}" \
            if ms else "-"
        lines.append(f"| {gname} | {len(g['rows'])} | {c.get('PASS', 0)} "
                     f"| {c.get('FAIL', 0)} | {c.get('NOTE', 0)} "
                     f"| {c.get('ERROR', 0)} | {med} | {p95} |")
    fails = [r for r in rows if r["verdict"] in ("FAIL", "ERROR")]
    if fails:
        lines += ["", "### Failures", ""]
        for r in fails:
            lines.append(f"- **{r['id']}** ({r['group']}): "
                         f"{r['evidence'][:400]}")
    deltas = []
    for r in rows:
        base = hz.BASELINES.get(r["id"]) or {}
        if base.get("bytes") and r.get("bytes"):
            deltas.append((abs(r["bytes"] / base["bytes"] - 1), r["id"],
                           r["bytes"], base["bytes"]))
    if deltas:
        deltas.sort(reverse=True)
        lines += ["", "### Largest output byte deltas vs baseline "
                      "(trend data)", "",
                      "| case | bytes | baseline | delta |",
                      "|---|---|---|---|"]
        for d, cid, b, bb in deltas[:10]:
            lines.append(f"| {cid} | {b} | {bb} | {d * 100:.1f}% |")
    lines += ["", "Timings are trend data only; they never gate a run."]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return counts


def build_binary(release):
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    cmd = ["cargo", "build"] + (["--release"] if release else [])
    r = subprocess.run(cmd, cwd=hz.REPO, env=env, capture_output=True,
                       text=True, timeout=1800)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("cargo build failed")
    return os.path.join(hz.REPO, "target",
                        "release" if release else "debug", "crabcrop")


def build_parity_rs(out_dir):
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    target = os.path.join(out_dir, "parity-rs", "target")
    r = subprocess.run(
        ["cargo", "build", "--release",
         "--manifest-path", os.path.join(FUNC_DIR, "parity-rs",
                                         "Cargo.toml"),
         "--target-dir", target],
        env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise SystemExit("parity-rs build failed")


def ai_instance_env(models, ort):
    common = dict(CRABCROP_ALLOW_UNSAFE="1", **hz.ALLOW)
    if ort:
        common["CRABCROP_ORT_DYLIB"] = ort
        common["ORT_DYLIB_PATH"] = ort
    ai = dict(common,
              CRABCROP_REMOVEBG_MODEL=f"{models}/u2netp.onnx",
              CRABCROP_FACE_MODEL=f"{models}/yunet.onnx",
              CRABCROP_CLIP_IMAGE_MODEL=f"{models}/clip_vision.onnx",
              CRABCROP_CLIP_TEXT_MODEL=f"{models}/clip_text.onnx",
              CRABCROP_CLIP_TOKENIZER=f"{models}/clip_tokenizer.json",
              CRABCROP_CLIP_LABELS=f"{models}/clip_labels.txt",
              CRABCROP_MODERATION_MODEL=f"{models}/moderation.onnx",
              CRABCROP_MODERATION_LABELS=f"{models}/moderation_labels.txt",
              CRABCROP_AESTHETIC_HEAD=f"{models}/aesthetic_b32.pth",
              CRABCROP_DETECT_MODEL=f"{models}/yolo.onnx",
              CRABCROP_DETECTORS_FILE=f"{models}/detectors.json",
              CRABCROP_UPSCALE_MODEL=f"{models}/upscale_swin2sr.onnx",
              CRABCROP_BRAND_COLOR="0E7C7B", CRABCROP_STUDIO_TIMING="1",
              ORT_DISABLE_TELEMETRY="1")
    aic = dict(common,
               CRABCROP_CLIP_IMAGE_MODEL=f"{models}/clip_vision.onnx",
               CRABCROP_CLIP_TEXT_MODEL=f"{models}/clip_text.onnx",
               CRABCROP_CLIP_TOKENIZER=f"{models}/clip_tokenizer.json",
               CRABCROP_CLIP_LABELS=f"{models}/clip_labels.txt",
               CRABCROP_MODERATION_LABELS=
               f"{models}/moderation_signal_labels.txt",
               CRABCROP_AESTHETIC_HEAD=f"{models}/aesthetic_l14_v1.pth",
               ORT_DISABLE_TELEMETRY="1")
    return {"AI": ai, "AIN": dict(common), "AIC": aic}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True,
                    choices=["smoke", "full", "local"])
    ap.add_argument("--bin")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--out", default=os.path.join(FUNC_DIR, "out"))
    ap.add_argument("--ids", default="")
    ap.add_argument("--groups", default="")
    ap.add_argument("--bless", action="store_true")
    ap.add_argument("--enforce-budget", action="store_true")
    ap.add_argument("--keep-servers", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    bin_path = args.bin
    if not bin_path:
        if args.no_build:
            raise SystemExit("--no-build needs --bin")
        bin_path = build_binary(args.release)
    bin_path = os.path.abspath(bin_path)
    # The binary has no --version verb; the crate version is the stamp.
    ver = "unknown"
    try:
        import re as _re
        m = _re.search(r'^version\s*=\s*"([^"]+)"',
                       open(os.path.join(hz.REPO, "Cargo.toml")).read(),
                       _re.M)
        if m:
            ver = f"crabcrop {m.group(1)}"
    except OSError:
        pass

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    hz.configure(bin_path, out_dir)
    hz.set_save_hook(lambda: save_outputs(out_dir))

    baselines_path = os.path.join(FUNC_DIR, "baselines.json")
    if args.bless:
        hz.BLESS = {}
    elif os.path.exists(baselines_path):
        hz.BASELINES = json.load(open(baselines_path))

    want_ids = {s for s in args.ids.split(",") if s}
    want_groups = {s for s in args.groups.split(",") if s}

    def wanted(case):
        if want_ids and case["id"] not in want_ids:
            return False
        if want_groups and case["group"] not in want_groups:
            return False
        return True

    # ---------------- case selection ----------------
    dsl_cases, platform_rows, ai_rows = [], [], []
    models = os.environ.get("CRABCROP_FUNC_MODELS_DIR")
    ort = os.environ.get("CRABCROP_ORT_DYLIB")
    if args.tier in ("smoke", "full"):
        tiers = {"smoke"} if args.tier == "smoke" else {"smoke", "full"}
        for name in ("cases_core.json", "cases_overlay.json"):
            dsl_cases += [c for c in load_cases(name)
                          if c["tier"] in tiers]
        platform_rows = [c for c in load_cases("cases_platform.json")
                         if c["tier"] in tiers]
    else:
        ai_rows = load_cases("cases_ai.json")
    dsl_cases = [substitute(c) for c in dsl_cases if wanted(c)]
    platform_rows = [c for c in platform_rows if wanted(c)]
    ai_rows = [c for c in ai_rows if wanted(c)]

    # ---------------- servers ----------------
    t_boot = time.time()
    try:
        if not hz.start_origin():
            raise SystemExit("origin server did not come up")
        if args.tier in ("smoke", "full"):
            need = {"A"}
            for c in dsl_cases:
                need.add(hz.instance_of(c))
            for c in platform_rows:
                need |= {"P", "N", "U", "OP", "E"}
                break
            if args.tier == "smoke":
                need &= {"A", "P", "U"}
            ok = hz.start_instances(sorted(need))
            bad = [k for k, v in ok.items() if not v]
            if bad:
                raise SystemExit(f"instances failed to boot: {bad}")
        elif models:
            envs = ai_instance_env(models, ort)
            for name in ("AI", "AIN", "AIC"):
                if not hz.start_server(name, envs[name],
                                       health_timeout=240):
                    raise SystemExit(f"AI instance {name} failed to boot")
    except BaseException:
        if not args.keep_servers:
            hz.stop_servers()
        raise
    boot_s = time.time() - t_boot

    exit_code = 0
    try:
        # ---------------- DSL cases ----------------
        for case in dsl_cases:
            hz.run_dsl_case(case)
        # ---------------- scripted platform cases ----------------
        if platform_rows:
            need_rs = any(c["id"] in ("parity-build-url-and-srcset",
                                      "parity-signatures-and-vectors")
                          for c in platform_rows)
            if need_rs:
                build_parity_rs(out_dir)
            import platform_cases
            for row in platform_rows:
                fn = platform_cases.CELLS.get(row["id"])
                if fn is None:
                    hz.rec(row["id"], row["group"], "-", "-", "ERROR",
                           None, "no scripted cell registered for this id")
                    continue
                try:
                    fn()
                except Exception as e:
                    hz.rec(row["id"], row["group"], "-", "-", "ERROR",
                           None,
                           f"harness exception: {type(e).__name__}: {e}")
        # ---------------- scripted AI cases (local tier) ----------------
        if ai_rows:
            import ai_cases
            for row in ai_rows:
                if not models:
                    hz.rec(row["id"], row["group"], "-", "-", "NOTE", None,
                           "skipped: local tier needs "
                           "CRABCROP_FUNC_MODELS_DIR (models are "
                           "deployment artifacts, never repo content)")
                    continue
                fn = ai_cases.CELLS.get(row["id"])
                if fn is None:
                    hz.rec(row["id"], row["group"], "-", "-", "ERROR",
                           None, "no scripted cell registered for this id")
                    continue
                try:
                    fn()
                except Exception as e:
                    hz.rec(row["id"], row["group"], "-", "-", "ERROR",
                           None,
                           f"harness exception: {type(e).__name__}: {e}")
    finally:
        wall_s = time.time() - t_start
        counts = write_summary(out_dir, args.tier, wall_s, boot_s, ver)
        save_outputs(out_dir)
        if args.bless:
            merged = {}
            if os.path.exists(baselines_path):
                merged = json.load(open(baselines_path))
            merged.update(hz.BLESS or {})
            with open(baselines_path, "w") as f:
                json.dump(merged, f, indent=1, sort_keys=True)
            print(f"blessed {len(hz.BLESS or {})} baselines into "
                  f"{baselines_path}")
        if not args.keep_servers:
            hz.stop_servers()
    bad = [r for r in hz.RESULTS.values()
           if r["verdict"] in ("FAIL", "ERROR")]
    print(f"\n{args.tier} tier: {len(hz.RESULTS)} cases, "
          f"{len(bad)} failing, wall {wall_s:.1f}s (boots {boot_s:.1f}s)")
    if bad:
        exit_code = 1
    if args.tier == "smoke" and args.enforce_budget \
            and wall_s > SMOKE_BUDGET_S:
        print(f"smoke budget exceeded: {wall_s:.1f}s > {SMOKE_BUDGET_S}s "
              f"(an overrun is a bug in the suite: cut cells or shrink "
              f"images, do not raise the budget)")
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
