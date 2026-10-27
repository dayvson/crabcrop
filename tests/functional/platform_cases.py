#!/usr/bin/env python3
"""Scripted platform cases for the functional harness (issue #232).

Ported from wave 4 of the feature campaign: presets, purge, explain,
upload, source kinds, SDK / MCP / framework parity. Each cell is a
function registered in CELLS by dashed name; cases_platform.json is
the index (id, group, tier) the runner drives from.

Imports harness after harness.configure() has run (run.py does this),
so the runtime paths (BIN, OUT, UPLOAD_DIR, HITS_PATH, FRESH_BIN) are
already set when this module binds them.
"""
import argparse, io, json, os, re, shutil, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from harness import (FUNC_DIR, REPO, CORPUS, OUT, BIN, SECRET, PORTS,
                     SRC127, SRCLOC, BASES, UPLOAD_DIR, HITS_PATH,
                     FRESH_BIN, get, request, sign, signed_path,
                     unsafe_path, purge_sig, group_sig, upload_sig,
                     purge_cli, cli, timed_render, fact_of, origin_hits,
                     log_size, raw_log_tail, rec, check, note, RESULTS)
from mcp_client import Mcp

CELLS = {}
def cell(fn):
    CELLS[fn.__name__.replace("_", "-")] = fn
    return fn

NOW = int(time.time())
FUTURE = NOW + 86400
PAST = NOW - 3600
DOG = f"{SRC127}/dog.jpg"
CAT = f"{SRC127}/cat.jpg"
CAR = f"{SRC127}/car.jpg"
LOGO = f"{SRC127}/logo.png"
GRAD = f"{SRC127}/gradient.png"
BLUE = f"{SRC127}/blue.jpg"


def dims(fact):
    if not fact.get("path"):
        return None
    im = Image.open(fact["path"])
    return im.size


def body_text(fact):
    return fact.get("body", "")


def purge_http(target, sig, base="P"):
    st, h, payload, ms = request(base, "POST", f"/purge/{sig}/{target}",
                                 path_as_is=True)
    return st, payload.decode("utf-8", "replace").strip(), h


def purge_group_http(group, sig, base="P"):
    st, h, payload, ms = request(base, "POST", f"/purge-group/{sig}/{group}")
    return st, payload.decode("utf-8", "replace").strip(), h


def render_signed(cell, base, rem, headers=None, timeout=120):
    """Purge then timed render of the signed URL for rem (a clean miss)."""
    purge_cli(rem, base)
    return timed_render(cell, base, signed_path(rem), headers, timeout=timeout)


# ---------------------------------------------------------------- presets
@cell
def preset_basic():
    rem_p = f"preset:thumb/{DOG}"
    rem_e = f"300x200/{DOG}"
    fe = render_signed("preset-basic", "P", rem_e)
    fp = render_signed("preset-basic", "P", rem_p)
    ok = (fp["status"] == 200 and fe["status"] == 200
          and fp["md5"] == fe["md5"] and dims(fp) == (300, 200)
          and fp["cache_group"] == '"thumb"')
    check("preset-basic", "presets", "preset:thumb", "dog.jpg", ok, fp,
          f"status={fp['status']} dims={dims(fp)} bytes_equal_expanded={fp['md5'] == fe['md5']} "
          f"cache_group={fp['cache_group']} server_ms={fp['server_ms']}")


@cell
def preset_cache_expanded_first():
    rem_e = f"300x200/{CAT}"
    rem_p = f"preset:thumb/{CAT}"
    purge_cli(rem_e, "P")
    f1 = timed_render("preset-cache-expanded-first", "P", signed_path(rem_e))
    f2 = timed_render("preset-cache-expanded-first", "P", signed_path(rem_p))
    ok = (f1["cache_status"].endswith("fwd=uri-miss")
          and "hit" in f2["cache_status"] and f1["md5"] == f2["md5"])
    check("preset-cache-expanded-first", "presets", "preset:thumb after 300x200",
          "cat.jpg", ok, f2,
          f"expanded cache={f1['cache_status']} then preset cache={f2['cache_status']} "
          f"bytes_equal={f1['md5'] == f2['md5']}")


@cell
def preset_cache_preset_first():
    rem_e = f"300x200/{CAR}"
    rem_p = f"preset:thumb/{CAR}"
    purge_cli(rem_p, "P")
    f1 = timed_render("preset-cache-preset-first", "P", signed_path(rem_p))
    f2 = timed_render("preset-cache-preset-first", "P", signed_path(rem_e))
    ok = (f1["cache_status"].endswith("fwd=uri-miss")
          and "hit" in f2["cache_status"] and f1["md5"] == f2["md5"])
    check("preset-cache-preset-first", "presets", "300x200 after preset:thumb",
          "car.jpg", ok, f2,
          f"preset cache={f1['cache_status']} then expanded cache={f2['cache_status']} "
          f"bytes_equal={f1['md5'] == f2['md5']}")


@cell
def preset_compose_inline():
    rem_p = f"preset:thumb/filters:grayscale/{DOG}"
    rem_e = f"300x200/filters:grayscale/{DOG}"
    fe = render_signed("preset-compose-inline", "P", rem_e)
    fp = render_signed("preset-compose-inline", "P", rem_p)
    ok = fp["status"] == 200 and fp["md5"] == fe["md5"]
    check("preset-compose-inline", "presets", "preset:thumb/filters:grayscale",
          "dog.jpg", ok, fp,
          f"status={fp['status']} bytes_equal_expanded_plus_inline={fp['md5'] == fe['md5']}")


@cell
def preset_unknown():
    rem = f"preset:nope/{DOG}"
    pos = log_size("P")
    st, h, payload, ms = get("P", signed_path(rem), {"Accept": "application/json"})
    f = fact_of("preset-unknown", "P", "GET", signed_path(rem), st, h, payload, ms)
    code = None
    try:
        code = json.loads(payload)["error"]["code"]
    except Exception:
        pass
    ok = st == 400 and code == "unknown_preset"
    check("preset-unknown", "presets", "preset:nope", "dog.jpg", ok, f,
          f"status={st} json_code={code} body={payload[:100]!r}")


@cell
def preset_contexts():
    rem = f"preset:thumb/{LOGO}"
    fa = timed_render("preset-contexts", "P", unsafe_path(rem))
    fb = timed_render("preset-contexts", "N", unsafe_path(rem))
    bad = "0" * 64
    fc = timed_render("preset-contexts", "P", f"/{bad}/{rem}")
    fd = timed_render("preset-contexts", "N", signed_path(rem))
    ok = (fa["status"] == 200 and fb["status"] == 403
          and fc["status"] == 403 and fd["status"] == 400)
    check("preset-contexts", "presets", "preset:thumb", "logo.png", ok, fa,
          f"unsafe_on_P={fa['status']} unsafe_on_N={fb['status']} body={fb['body'][:30]!r} "
          f"badsig_on_P={fc['status']} signed_on_N_no_presets_file={fd['status']} body={fd['body'][:40]!r}")


@cell
def preset_exp():
    rem_f = f"exp:{FUTURE}/preset:thumb/{GRAD}"
    rem_p = f"exp:{PAST}/preset:thumb/{GRAD}"
    rem_p2 = f"preset:thumb/exp:{PAST}/{GRAD}"
    ff = timed_render("preset-exp", "P", signed_path(rem_f))
    fp = timed_render("preset-exp", "P", signed_path(rem_p))
    fp2 = timed_render("preset-exp", "P", signed_path(rem_p2))
    ok = (ff["status"] == 200 and fp["status"] == 403
          and "expired" in fp["body"] and fp2["status"] == 403)
    check("preset-exp", "presets", "exp + preset:thumb", "gradient.png", ok, ff,
          f"future={ff['status']} past_leading={fp['status']} body={fp['body'][:30]!r} "
          f"past_after_preset={fp2['status']}")


@cell
def preset_maxage():
    rem = f"preset:thumb/maxage:60/{BLUE}"
    f = render_signed("preset-maxage", "P", rem)
    rem0 = f"preset:thumb/maxage:0/{BLUE}"
    f0 = render_signed("preset-maxage", "P", rem0)
    ok = (f["status"] == 200 and "max-age=60" in f["cache_control"]
          and f0["status"] == 200 and "max-age=0" in f0["cache_control"]
          and "immutable" not in f0["cache_control"])
    check("preset-maxage", "presets", "preset:thumb/maxage:60", "blue.jpg", ok, f,
          f"cc60={f['cache_control']!r} cc0={f0['cache_control']!r}")


def boot_with_presets(fname):
    env = dict(os.environ, CRABCROP_SECRET=SECRET, CRABCROP_PORT=str(PORTS["BOOT"]),
               CRABCROP_PRESETS_FILE=os.path.join(FUNC_DIR, "badpresets", fname),
               NO_PROXY="127.0.0.1,localhost")
    logf = open(os.path.join(OUT, "logs", f"boot-{fname}.log"), "w")
    p = subprocess.Popen([BIN], env=env, stdout=logf, stderr=subprocess.STDOUT)
    try:
        rc = p.wait(timeout=6)
        out = open(os.path.join(OUT, "logs", f"boot-{fname}.log")).read()
        return rc, out
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        return None, "SERVER BOOTED AND KEPT RUNNING"


def _boot_cell(cell_id, fname, label):
    rc, out = boot_with_presets(fname)
    ok = rc is not None and rc != 0 and "preset" in out.lower()
    rec(cell_id, "presets", label, "-", "PASS" if ok else "FAIL", None,
        f"exit={rc} log_tail={out.strip()[-160:]!r}")


@cell
def preset_boot_exp():
    _boot_cell("preset-boot-exp", "exp.txt", "definition containing exp:")


@cell
def preset_boot_nested():
    _boot_cell("preset-boot-nested", "nested.txt", "definition containing preset:")


@cell
def preset_boot_source():
    _boot_cell("preset-boot-source", "source.txt", "definition containing a source")


@cell
def preset_boot_maxage():
    _boot_cell("preset-boot-maxage", "maxage.txt", "definition containing maxage:")


@cell
def preset_boot_malformed():
    _boot_cell("preset-boot-malformed", "malformed.txt", "definition line without '='")


@cell
def preset_boot_badfilter():
    _boot_cell("preset-boot-badfilter", "badfilter.txt", "definition with filters:bogus(1)")


@cell
def only_presets_freeform():
    rem = f"300x200/{DOG}"
    st, h, payload, ms = get("OP", signed_path(rem), {"Accept": "application/json"})
    f = fact_of("only-presets-freeform", "OP", "GET", signed_path(rem), st, h, payload, ms)
    code = None
    try:
        code = json.loads(payload)["error"]["code"]
    except Exception:
        pass
    ok = st == 400 and code == "presets_only"
    check("only-presets-freeform", "presets", "300x200 (only-presets)", "dog.jpg",
          ok, f, f"status={st} json_code={code}")


@cell
def only_presets_allowed_forms():
    rem_p = f"preset:thumb/{DOG}"
    rem_b = DOG
    rem_e = f"exp:{FUTURE}/preset:thumb/{DOG}"
    fp = timed_render("only-presets-allowed-forms", "OP", signed_path(rem_p))
    fb = timed_render("only-presets-allowed-forms", "OP", signed_path(rem_b))
    fe = timed_render("only-presets-allowed-forms", "OP", signed_path(rem_e))
    fu = timed_render("only-presets-allowed-forms", "OP", unsafe_path(rem_p))
    ok = (fp["status"] == 200 and fb["status"] == 200
          and fe["status"] == 200 and fu["status"] == 200)
    check("only-presets-allowed-forms", "presets", "preset / bare / exp+preset / unsafe preset",
          "dog.jpg", ok, fp,
          f"preset={fp['status']} bare={fb['status']} exp={fe['status']} unsafe={fu['status']}")


# ---------------------------------------------------------------- roots
@cell
def roots_valid_render():
    rem = "30x20/root:photos/dog.jpg"
    f = render_signed("roots-valid-render", "R", rem)
    ok = f["status"] == 200 and dims(f) == (30, 20)
    check("roots-valid-render", "roots", "30x20/root:photos/dog.jpg",
          "dog.jpg", ok, f,
          f"status={f['status']} dims={dims(f)} server_ms={f['server_ms']}")


@cell
def roots_cache_and_purge():
    short = "300x200/root:photos/cat.jpg"
    full = f"300x200/{CAT}"
    purge_cli(short, "R")
    hits0 = origin_hits("/cat.jpg")
    f1 = timed_render("roots-cache-and-purge", "R", signed_path(short))
    f2 = timed_render("roots-cache-and-purge", "R", signed_path(full))
    hits1 = origin_hits("/cat.jpg")
    st, txt, _ = purge_http(short, purge_sig(short), base="R")
    f3 = timed_render("roots-cache-and-purge", "R", signed_path(full))
    hits2 = origin_hits("/cat.jpg")
    ok = (f1["status"] == 200 and f2["status"] == 200
          and f1["md5"] == f2["md5"] and hits1 - hits0 == 1
          and "purged 1" in txt and f3["status"] == 200
          and hits2 - hits1 == 1)
    check("roots-cache-and-purge", "roots",
          "300x200 short + full spellings", "cat.jpg", ok, f1,
          f"short={f1['status']} full={f2['status']} bytes_equal={f1['md5'] == f2['md5']} "
          f"fetches_for_both={hits1 - hits0} purge={txt!r} "
          f"after_purge={f3['status']} refetch={hits2 - hits1}")


@cell
def roots_refusals_no_fetch():
    """Issue #250: adversarial root spellings against the real
    binary. Every one is its documented 400, none reaches the
    origin, and the same process renders afterwards."""
    cases = [
        ("30x20/root:nope/dog.jpg", "unknown_root"),
        ("30x20/root:PHOTOS/dog.jpg", "unknown_root"),
        ("30x20/root:photo/dog.jpg", "unknown_root"),
        ("30x20/root:/dog.jpg", "invalid_root"),
        ("30x20/root:photos/../secret.jpg", "invalid_root"),
        ("30x20/root:photos/%2e%2e/secret.jpg", "invalid_root"),
        ("30x20/root:photos/..%5csecret.jpg", "invalid_root"),
    ]
    hits0 = origin_hits()
    results = []
    for rem, want in cases:
        st, h, payload, ms = get("R", signed_path(rem),
                                 {"Accept": "application/json"})
        code = None
        try:
            code = json.loads(payload)["error"]["code"]
        except Exception:
            pass
        results.append((rem, st, code, want))
    hits1 = origin_hits()
    bad = [r for r in results if r[1] != 400 or r[2] != r[3]]
    f = render_signed("roots-refusals-no-fetch", "R",
                      "30x20/root:photos/dog.jpg")
    ok = not bad and hits1 == hits0 and f["status"] == 200
    check("roots-refusals-no-fetch", "roots", "adversarial root spellings",
          "dog.jpg", ok, f,
          " | ".join(f"{r[0]} -> {r[1]}/{r[2]}" for r in results)[:420]
          + f" origin_hits_delta={hits1 - hits0} render_after={f['status']}")


@cell
def roots_explain():
    rem = "400x300/root:photos/dog.jpg"
    st, h, payload, ms = explain_get(rem, base="R")
    f = fact_of("roots-explain", "R", "GET", f"/explain/{rem}",
                st, h, payload, ms)
    text = payload.decode("utf-8", "replace")
    j = {}
    try:
        j = json.loads(payload)
    except Exception:
        pass
    srcj = j.get("source") or {}
    hits0 = origin_hits("/dog.jpg")
    rem2 = "400x300/root:nope/dog.jpg"
    st2, h2, p2, _ = explain_get(rem2, base="R")
    code2 = None
    try:
        code2 = json.loads(p2)["error"]["code"]
    except Exception:
        pass
    hits1 = origin_hits("/dog.jpg")
    ok = (st == 200 and srcj.get("url") == "root:photos/dog.jpg"
          and srcj.get("root") == "photos" and "8971" not in text
          and st2 == 400 and code2 == "unknown_root" and hits1 == hits0)
    check("roots-explain", "roots", "400x300/root:photos/dog.jpg",
          "dog.jpg", ok, f,
          f"status={st} source_url={srcj.get('url')} root={srcj.get('root')} "
          f"base_leaked={'8971' in text} unknown: status={st2} "
          f"code={code2} fetches={hits1 - hits0}")


def boot_with_roots(fname):
    env = dict(os.environ, CRABCROP_SECRET=SECRET, CRABCROP_PORT=str(PORTS["BOOT"]),
               CRABCROP_SOURCE_ROOTS_FILE=os.path.join(FUNC_DIR, "badroots", fname),
               NO_PROXY="127.0.0.1,localhost")
    logf = open(os.path.join(OUT, "logs", f"boot-roots-{fname}.log"), "w")
    p = subprocess.Popen([BIN], env=env, stdout=logf, stderr=subprocess.STDOUT)
    try:
        rc = p.wait(timeout=6)
        out = open(os.path.join(OUT, "logs", f"boot-roots-{fname}.log")).read()
        return rc, out
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        return None, "SERVER BOOTED AND KEPT RUNNING"


def _roots_boot_cell(cell_id, fname, label):
    rc, out = boot_with_roots(fname)
    ok = rc is not None and rc != 0 and "root" in out.lower()
    rec(cell_id, "roots", label, "-", "PASS" if ok else "FAIL", None,
        f"exit={rc} log_tail={out.strip()[-160:]!r}")


@cell
def roots_boot_duplicate():
    _roots_boot_cell("roots-boot-duplicate", "duplicate.txt",
                     "roots file with a duplicate name")


@cell
def roots_boot_bad_name():
    _roots_boot_cell("roots-boot-bad-name", "badname.txt",
                     "roots file with a dotted name")


@cell
def roots_boot_bad_base():
    _roots_boot_cell("roots-boot-bad-base", "badbase.txt",
                     "roots file with a non-http base")


# ---------------------------------------------------------------- purge
@cell
def purge_basic():
    rem = f"311x211/{CAT}"
    purge_cli(rem, "P")
    f1 = timed_render("purge-basic", "P", signed_path(rem))
    f2 = timed_render("purge-basic", "P", signed_path(rem))
    st, txt, _ = purge_http(rem, purge_sig(rem))
    f3 = timed_render("purge-basic", "P", signed_path(rem))
    ok = ("hit" in f2["cache_status"] and st == 200
          and re.fullmatch(r"purged [1-9]\d*", txt or "") is not None
          and f3["cache_status"].endswith("fwd=uri-miss"))
    check("purge-basic", "purge", "311x211", "cat.jpg", ok, f3,
          f"purge_status={st} body={txt!r} before={f2['cache_status']} after={f3['cache_status']}")


@cell
def purge_cli_verb():
    rem = f"312x212/{CAR}"
    purge_cli(rem, "P")
    timed_render("purge-cli-verb", "P", signed_path(rem))
    f2 = timed_render("purge-cli-verb", "P", signed_path(rem))
    rc, out = purge_cli(rem, "P")
    f3 = timed_render("purge-cli-verb", "P", signed_path(rem))
    ok = (rc == 0 and "purged" in out and "hit" in f2["cache_status"]
          and f3["cache_status"].endswith("fwd=uri-miss"))
    check("purge-cli-verb", "purge", "312x212", "car.jpg", ok, f3,
          f"cli_rc={rc} cli_out={out[:60]!r} after={f3['cache_status']}")


@cell
def purge_never_rendered():
    rem = f"313x213/{GRAD}"
    st, txt, _ = purge_http(rem, purge_sig(rem))
    ok = st == 200 and txt == "purged 0"
    rec("purge-never-rendered", "purge", "313x213", "gradient.png",
        "PASS" if ok else "FAIL", None, f"status={st} body={txt!r}")


@cell
def purge_bad_sig():
    rem = f"314x214/{DOG}"
    purge_cli(rem, "P")
    timed_render("purge-bad-sig", "P", signed_path(rem))
    st, txt, _ = purge_http(rem, "0" * 64)
    f = timed_render("purge-bad-sig", "P", signed_path(rem))
    ok = st == 403 and "hit" in f["cache_status"]
    check("purge-bad-sig", "purge", "314x214", "dog.jpg", ok, f,
          f"purge_status={st} body={txt!r} entry_after={f['cache_status']}")


@cell
def purge_wrong_kind():
    rem = f"315x215/{CAT}"
    purge_cli(rem, "P")
    timed_render("purge-wrong-kind", "P", signed_path(rem))
    st1, t1, _ = purge_http(rem, sign(rem))            # image signature
    st2, t2, _ = purge_http(rem, upload_sig(rem))      # upload signature
    st3, h3, p3, _ = get("P", f"/{purge_sig(rem)}/{rem}")  # purge sig as image sig
    f = timed_render("purge-wrong-kind", "P", signed_path(rem))
    ok = (st1 == 403 and st2 == 403 and st3 == 403
          and "hit" in f["cache_status"])
    check("purge-wrong-kind", "purge", "315x215", "cat.jpg", ok, f,
          f"image_sig_as_purge={st1} upload_sig_as_purge={st2} purge_sig_as_image={st3} "
          f"entry_after={f['cache_status']}")


@cell
def purge_preset_spelling():
    rem_e = f"300x200/{LOGO}"
    rem_p = f"preset:thumb/{LOGO}"
    purge_cli(rem_e, "P")
    timed_render("purge-preset-spelling", "P", signed_path(rem_e))
    f2 = timed_render("purge-preset-spelling", "P", signed_path(rem_e))
    st, txt, _ = purge_http(rem_p, purge_sig(rem_p))
    f3 = timed_render("purge-preset-spelling", "P", signed_path(rem_e))
    ok = ("hit" in f2["cache_status"] and st == 200
          and re.fullmatch(r"purged [1-9]\d*", txt or "") is not None
          and f3["cache_status"].endswith("fwd=uri-miss"))
    check("purge-preset-spelling", "purge", "preset:thumb spelling", "logo.png",
          ok, f3, f"purge_status={st} body={txt!r} after={f3['cache_status']}")


@cell
def purge_equivalent_spelling():
    # leading zeros
    rem_a = f"0300x200/{BLUE}"
    rem_b = f"300x200/{BLUE}"
    purge_cli(rem_b, "P")
    fa = timed_render("purge-equivalent-spelling", "P", signed_path(rem_a))
    st1, t1, _ = purge_http(rem_b, purge_sig(rem_b))
    fa2 = timed_render("purge-equivalent-spelling", "P", signed_path(rem_a))
    # jpg alias
    rem_j1 = f"filters:format(jpg)/{GRAD}"
    rem_j2 = f"filters:format(jpeg)/{GRAD}"
    purge_cli(rem_j2, "P")
    fj = timed_render("purge-equivalent-spelling", "P", signed_path(rem_j1))
    st2, t2, _ = purge_http(rem_j2, purge_sig(rem_j2))
    fj2 = timed_render("purge-equivalent-spelling", "P", signed_path(rem_j1))
    ok = (fa["status"] == 200 and st1 == 200
          and fa2["cache_status"].endswith("fwd=uri-miss")
          and fj["status"] == 200 and st2 == 200
          and fj2["cache_status"].endswith("fwd=uri-miss"))
    check("purge-equivalent-spelling", "purge", "0300x200 / format(jpg)",
          "blue.jpg + gradient.png", ok, fa2,
          f"zeros: render={fa['status']} purge={t1!r} re_render={fa2['cache_status']} | "
          f"jpg-alias: render={fj['status']} purge={t2!r} re_render={fj2['cache_status']}")


@cell
def purge_exp_variants():
    rem_plain = f"320x240/{CAR}"
    rem_exp = f"exp:{FUTURE}/320x240/{CAR}"
    purge_cli(rem_plain, "P")
    timed_render("purge-exp-variants", "P", signed_path(rem_plain))
    timed_render("purge-exp-variants", "P", signed_path(rem_exp))
    st, txt, _ = purge_http(rem_plain, purge_sig(rem_plain))
    f_exp = timed_render("purge-exp-variants", "P", signed_path(rem_exp))
    # reverse direction: exp-spelled purge target evicts the plain entry
    timed_render("purge-exp-variants", "P", signed_path(rem_plain))
    st2, txt2, _ = purge_http(rem_exp, purge_sig(rem_exp))
    f_plain = timed_render("purge-exp-variants", "P", signed_path(rem_plain))
    ok = (f_exp["cache_status"].endswith("fwd=uri-miss")
          and f_plain["cache_status"].endswith("fwd=uri-miss"))
    check("purge-exp-variants", "purge", "320x240 +/- exp", "car.jpg", ok, f_exp,
          f"purge_plain={txt!r} exp_rerender={f_exp['cache_status']} | "
          f"purge_exp_spelling={txt2!r} plain_rerender={f_plain['cache_status']}")


@cell
def purge_group_basic():
    purge_group_http("thumb", group_sig("thumb"))
    fd = render_signed("purge-group-basic", "P", f"preset:thumb/{DOG}")
    fc = render_signed("purge-group-basic", "P", f"preset:thumb/{CAT}")
    fb = render_signed("purge-group-basic", "P", f"300x200/{BLUE}")
    st, txt, _ = purge_group_http("thumb", group_sig("thumb"))
    fd2 = timed_render("purge-group-basic", "P", signed_path(f"preset:thumb/{DOG}"))
    fc2 = timed_render("purge-group-basic", "P", signed_path(f"preset:thumb/{CAT}"))
    fb2 = timed_render("purge-group-basic", "P", signed_path(f"300x200/{BLUE}"))
    n = int(txt.split()[1]) if txt and txt.startswith("purged") else -1
    ok = (st == 200 and n >= 2
          and fd2["cache_status"].endswith("fwd=uri-miss")
          and fc2["cache_status"].endswith("fwd=uri-miss")
          and "hit" in fb2["cache_status"])
    check("purge-group-basic", "purge", "purge-group thumb", "dog+cat via preset, blue plain",
          ok, fd2, f"status={st} body={txt!r} dog={fd2['cache_status']} cat={fc2['cache_status']} "
          f"ungrouped_blue={fb2['cache_status']}")


@cell
def purge_group_unknown_and_badsig():
    st0, t0, _ = purge_group_http("nosuchgroup", group_sig("nosuchgroup"))
    rem = f"preset:thumb/{GRAD}"
    render_signed("purge-group-unknown-and-badsig", "P", rem)
    st1, t1, _ = purge_group_http("thumb", "0" * 64)
    st2, t2, _ = purge_group_http("thumb", purge_sig("thumb"))  # wrong kind
    f = timed_render("purge-group-unknown-and-badsig", "P", signed_path(rem))
    ok = (st0 == 200 and t0 == "purged 0" and st1 == 403 and st2 == 403
          and "hit" in f["cache_status"])
    check("purge-group-unknown-and-badsig", "purge", "purge-group edge cases",
          "gradient.png", ok, f,
          f"unknown={st0}/{t0!r} badsig={st1} single_purge_sig_on_group={st2} "
          f"entry_after={f['cache_status']}")


@cell
def purge_freshness():
    fresh = f"{SRC127}/fresh.jpg"
    rem = f"300x200/{fresh}"
    with open(FRESH_BIN, "wb") as fh:
        fh.write(open(os.path.join(CORPUS, "dog.jpg"), "rb").read())
    f1 = render_signed("purge-freshness", "P", rem)
    with open(FRESH_BIN, "wb") as fh:
        fh.write(open(os.path.join(CORPUS, "blue.jpg"), "rb").read())
    f2 = timed_render("purge-freshness", "P", signed_path(rem))
    st, txt, _ = purge_http(rem, purge_sig(rem))
    f3 = timed_render("purge-freshness", "P", signed_path(rem))
    fb = render_signed("purge-freshness", "P", f"300x200/{BLUE}")
    ok = (f1["md5"] != f3["md5"] and f2["md5"] == f1["md5"]
          and f3["md5"] == fb["md5"])
    check("purge-freshness", "purge", "300x200", "fresh.jpg (swapped origin bytes)",
          ok, f3, f"before_md5={f1['md5'][:8]} cached_after_swap={f2['md5'][:8]} "
          f"after_purge={f3['md5'][:8]} blue_render={fb['md5'][:8]}")


@cell
def purge_unsafe_mode_still_signed():
    rem = f"316x216/{DOG}"
    st1, t1, _ = purge_http(rem, "unsafe")
    st2, t2, _ = purge_group_http("thumb", "unsafe")
    ok = st1 == 403 and st2 == 403
    rec("purge-unsafe-mode-still-signed", "purge", "purge with literal 'unsafe'",
        "dog.jpg", "PASS" if ok else "FAIL", None,
        f"purge_unsafe={st1} body={t1!r} purge_group_unsafe={st2} body={t2!r}")


# ---------------------------------------------------------------- explain
def explain_get(rem, base="P", sig_path=None, accept_json=True):
    path = f"/explain{sig_path if sig_path is not None else signed_path(rem)}"
    hdrs = {"Accept": "application/json"} if accept_json else {}
    st, h, payload, ms = get(base, path, hdrs)
    return st, h, payload, ms


@cell
def explain_valid_json():
    rem = f"400x300/{DOG}"
    hits0 = origin_hits("/dog.jpg")
    st, h, payload, ms = explain_get(rem)
    f = fact_of("explain-valid-json", "P", "GET", f"/explain/{rem}", st, h, payload, ms)
    hits1 = origin_hits("/dog.jpg")
    j = {}
    try:
        j = json.loads(payload)
    except Exception:
        pass
    fr = timed_render("explain-valid-json", "P", signed_path(rem))
    srcj = j.get("source") or {}
    ok = (st == 200 and (j.get("signature") or {}).get("status") == "valid"
          and srcj.get("width") == 1000 and srcj.get("height") == 750
          and srcj.get("kind") == "jpeg" and bool(j.get("canonical"))
          and bool(j.get("operations")) and hits1 - hits0 == 1
          and fr["cache_status"].endswith("fwd=uri-miss"))
    check("explain-valid-json", "explain", "400x300", "dog.jpg", ok, f,
          f"status={st} sig={(j.get('signature') or {}).get('status')} "
          f"source={srcj.get('width')}x{srcj.get('height')}/{srcj.get('kind')} "
          f"origin_fetches_for_explain={hits1 - hits0} image_after={fr['cache_status']}")


@cell
def explain_valid_html():
    rem = f"400x300/{CAT}"
    st, h, payload, ms = explain_get(rem, accept_json=False)
    f = fact_of("explain-valid-html", "P", "GET", f"/explain/{rem}", st, h, payload, ms)
    text = payload.decode("utf-8", "replace")
    ok = (st == 200 and h.get("content-type", "").startswith("text/html")
          and "cat.jpg" in text and len(text) > 400)
    check("explain-valid-html", "explain", "400x300", "cat.jpg", ok, f,
          f"status={st} ct={h.get('content-type')} bytes={len(payload)}")


@cell
def explain_bad_sig():
    rem = f"400x300/{CAR}"
    good = sign(rem)
    bad = ("0" if good[0] != "0" else "1") + good[1:]
    hits0 = origin_hits("/car.jpg")
    st, h, payload, ms = explain_get(rem, sig_path=f"/{bad}/{rem}")
    f = fact_of("explain-bad-sig", "P", "GET", f"/explain/{rem}", st, h, payload, ms)
    hits1 = origin_hits("/car.jpg")
    ok = st == 403 and hits1 == hits0
    check("explain-bad-sig", "explain", "400x300 (tampered sig)", "car.jpg", ok, f,
          f"status={st} body={payload[:40]!r} origin_fetches={hits1 - hits0}")


@cell
def explain_expired():
    rem = f"exp:{PAST}/400x300/{LOGO}"
    hits0 = origin_hits("/logo.png")
    st, h, payload, ms = explain_get(rem)
    f = fact_of("explain-expired", "P", "GET", f"/explain/{rem}", st, h, payload, ms)
    hits1 = origin_hits("/logo.png")
    ok = st == 403 and b"expired" in payload and hits1 == hits0
    check("explain-expired", "explain", "exp:<past>/400x300", "logo.png", ok, f,
          f"status={st} body={payload[:40]!r} origin_fetches={hits1 - hits0}")


@cell
def explain_malformed():
    rem = f"filters:bogus(1)/{DOG}"
    hits0 = origin_hits("/dog.jpg")
    st, h, payload, ms = explain_get(rem)
    f = fact_of("explain-malformed", "P", "GET", f"/explain/{rem}", st, h, payload, ms)
    hits1 = origin_hits("/dog.jpg")
    code = None
    try:
        code = json.loads(payload)["error"]["code"]
    except Exception:
        pass
    rem2 = "300x200"
    st2, h2, p2, _ = explain_get(rem2)
    code2 = None
    try:
        code2 = json.loads(p2)["error"]["code"]
    except Exception:
        pass
    ok = (st == 400 and code == "invalid_filters" and hits1 == hits0
          and st2 == 400 and code2 == "missing_source_url")
    check("explain-malformed", "explain", "filters:bogus(1) / no source", "dog.jpg",
          ok, f, f"bogus: status={st} code={code} fetches={hits1 - hits0} | "
          f"no-source: status={st2} code={code2}")


@cell
def explain_nonallowlisted_no_fetch():
    rem = f"400x300/{CAT}"  # CAT is on 127.0.0.1; E allows localhost only
    hits0 = origin_hits("/cat.jpg")
    st, h, payload, ms = explain_get(rem, base="E")
    f = fact_of("explain-nonallowlisted-no-fetch", "E", "GET", f"/explain/{rem}",
                st, h, payload, ms)
    hits1 = origin_hits("/cat.jpg")
    ok = st == 403 and hits1 == hits0
    check("explain-nonallowlisted-no-fetch", "explain", "400x300", "cat.jpg via E",
          ok, f, f"status={st} body={payload[:60]!r} origin_fetches={hits1 - hits0}")


@cell
def explain_no_private_leak():
    rem = f"preset:card/{DOG}"
    st, h, payload, ms = explain_get(rem)
    f = fact_of("explain-no-private-leak", "P", "GET", f"/explain/{rem}", st, h, payload, ms)
    text = payload.decode("utf-8", "replace")
    leaks = [s for s in [SECRET, "fingerprint", os.path.expanduser("~"), "CRABCROP_",
                         "presets.txt"] if s in text]
    j = {}
    try:
        j = json.loads(payload)
    except Exception:
        pass
    ok = st == 200 and not leaks
    check("explain-no-private-leak", "explain", "preset:card", "dog.jpg", ok, f,
          f"status={st} leaked_tokens={leaks} top_keys={sorted(j.keys())}")


@cell
def explain_preset_report():
    rem = f"preset:card/{CAR}"
    st, h, payload, ms = explain_get(rem)
    f = fact_of("explain-preset-report", "P", "GET", f"/explain/{rem}", st, h, payload, ms)
    j = {}
    try:
        j = json.loads(payload)
    except Exception:
        pass
    presets = j.get("presets") or []
    segs = []
    if presets:
        segs = [o.get("segment") for o in presets[0].get("operations", [])]
    ok = (st == 200 and j.get("cache_group") == "card" and presets
          and presets[0].get("name") == "card"
          and "crop:800x450" in (j.get("canonical") or "")
          and "crop:800x450" in segs)
    check("explain-preset-report", "explain", "preset:card", "car.jpg", ok, f,
          f"status={st} cache_group={j.get('cache_group')} preset_segs={segs} "
          f"canonical={j.get('canonical')}")


@cell
def explain_unsafe_form():
    rem = f"400x300/{GRAD}"
    st, h, payload, ms = explain_get(rem, sig_path=f"/unsafe/{rem}")
    f = fact_of("explain-unsafe-form", "P", "GET", f"/explain/unsafe/{rem}",
                st, h, payload, ms)
    j = {}
    try:
        j = json.loads(payload)
    except Exception:
        pass
    ok = st == 200 and bool(j.get("operations"))
    check("explain-unsafe-form", "explain", "400x300 via /explain/unsafe/",
          "gradient.png", ok, f,
          f"status={st} signature_field={j.get('signature')}")


# ---------------------------------------------------------------- upload
UPDIR = UPLOAD_DIR
DOG_BYTES = open(os.path.join(CORPUS, "dog.jpg"), "rb").read()
PNG_BYTES = open(os.path.join(CORPUS, "tiny.png"), "rb").read()
SVG_BYTES = open(os.path.join(CORPUS, "probe.svg"), "rb").read()


def put_upload(base, sigpath, body, path_as_is=False):
    st, h, payload, ms = request(base, "PUT", f"/upload/{sigpath}", body=body,
                                 timeout=120, path_as_is=path_as_is)
    return st, h, payload, ms


def uploads_tree():
    out = []
    for root, _, files in os.walk(UPDIR):
        for fn in files:
            out.append(os.path.relpath(os.path.join(root, fn), UPDIR))
    return sorted(out)


@cell
def upload_basic():
    name = "heroes/dog2.jpg"
    st, h, payload, ms = put_upload("U", f"{upload_sig(name)}/{name}", DOG_BYTES)
    f = fact_of("upload-basic", "U", "PUT", name, st, h, payload, ms)
    stored = os.path.join(UPDIR, name)
    stored_ok = os.path.isfile(stored) and open(stored, "rb").read() == DOG_BYTES
    rem = "200x0/file:///heroes/dog2.jpg"
    fr = render_signed("upload-basic", "U", rem)
    ok = (st == 201 and payload.decode().strip() == "file:///heroes/dog2.jpg"
          and h.get("x-crabcrop-source") == "file:///heroes/dog2.jpg"
          and stored_ok and fr["status"] == 200 and dims(fr) == (200, 150))
    check("upload-basic", "upload", name, "dog.jpg bytes", ok, f,
          f"status={st} body={payload[:40]!r} header={h.get('x-crabcrop-source')} "
          f"stored_identical={stored_ok} render={fr['status']} dims={dims(fr)}")


@cell
def upload_cli_verb():
    # CLI signature: crabcrop upload <file> [<name>] with --base/--secret env
    import subprocess as sp
    env = dict(os.environ, CRABCROP_BASE=BASES["U"], CRABCROP_SECRET=SECRET,
               NO_PROXY="127.0.0.1,localhost")
    r = sp.run([BIN, "upload", os.path.join(CORPUS, "cat.jpg"), "cli/cat.jpg"],
               env=env, capture_output=True, text=True, timeout=60)
    out = (r.stdout + r.stderr).strip()
    rem = "150x0/file:///cli/cat.jpg"
    fr = render_signed("upload-cli-verb", "U", rem)
    ok = (r.returncode == 0 and "file:///cli/cat.jpg" in out
          and fr["status"] == 200 and dims(fr) == (150, 103))
    rec("upload-cli-verb", "upload", "cli/cat.jpg", "cat.jpg",
        "PASS" if ok else "FAIL", fr,
        f"cli_rc={r.returncode} cli_out={out[:70]!r} render={fr['status']} dims={dims(fr)}")


@cell
def upload_wrong_sig_and_replays():
    name = "heroes/nope.jpg"
    st1, h1, p1, _ = put_upload("U", f"{'0' * 64}/{name}", DOG_BYTES)
    st2, h2, p2, _ = put_upload("U", f"{sign(name)}/{name}", DOG_BYTES)
    st3, h3, p3, _ = put_upload("U", f"{purge_sig(name)}/{name}", DOG_BYTES)
    st4, h4, p4, _ = put_upload("U", f"unsafe/{name}", DOG_BYTES)
    stored = os.path.exists(os.path.join(UPDIR, name))
    ok = st1 == 403 and st2 == 403 and st3 == 403 and st4 == 403 and not stored
    rec("upload-wrong-sig-and-replays", "upload", name, "dog.jpg bytes",
        "PASS" if ok else "FAIL", None,
        f"garbage={st1} image_sig={st2} purge_sig={st3} unsafe={st4} stored={stored}")


@cell
def upload_disabled_404():
    name = "heroes/disabled.jpg"
    st, h, payload, ms = put_upload("P", f"{upload_sig(name)}/{name}", DOG_BYTES)
    f = fact_of("upload-disabled-404", "P", "PUT", name, st, h, payload, ms)
    ok = st == 404
    check("upload-disabled-404", "upload", name, "dog.jpg bytes", ok, f,
          f"status={st} (uploads not configured on P)")


@cell
def upload_oversized():
    name = "big/huge.jpg"
    body = b"\x00" * (11 * 1024 * 1024)
    st, h, payload, ms = put_upload("U", f"{upload_sig(name)}/{name}", body)
    f = fact_of("upload-oversized", "U", "PUT", name, st, h, payload, ms)
    stored = os.path.exists(os.path.join(UPDIR, name))
    ok = st == 413 and not stored
    check("upload-oversized", "upload", name, "11 MiB zeros", ok, f,
          f"status={st} body={payload[:60]!r} stored={stored}")


@cell
def upload_traversal():
    names = ["../evil.jpg", "a/../../evil.jpg", "/abs.jpg", "a//b.jpg",
             ".hidden.jpg", "a/./b.jpg", "a/%2e%2e/evil.jpg", "a\\b.jpg",
             "..", "a/b/../../../evil.jpg"]
    before = uploads_tree()
    results = {}
    for nm in names:
        st, h, payload, ms = put_upload("U", f"{upload_sig(nm)}/{nm}",
                                        DOG_BYTES, path_as_is=True)
        results[nm] = st
    after = uploads_tree()
    escaped = [p for p in [os.path.join(OUT, "evil.jpg"), "/abs.jpg"] if os.path.exists(p)]
    # Control (case-design correction): "trailingdot./x.jpg" obeys every
    # rule the README states (segment is not empty, not "." or "..", does
    # not start with a dot), so a 201 storing it literally inside the
    # upload root is the documented behavior, not an escape.
    ctl = "trailingdot./x.jpg"
    stc, _, _, _ = put_upload("U", f"{upload_sig(ctl)}/{ctl}", DOG_BYTES,
                              path_as_is=True)
    ctl_stored = os.path.isfile(os.path.join(UPDIR, ctl))
    import shutil
    shutil.rmtree(os.path.join(UPDIR, "trailingdot."), ignore_errors=True)
    ok = (all(s in (400, 403) for s in results.values())
          and before == after and not escaped
          and stc == 201 and ctl_stored)
    rec("upload-traversal", "upload", "10 hostile names + trailing-dot control",
        "dog.jpg bytes", "PASS" if ok else "FAIL", None,
        f"statuses={results} tree_unchanged={before == after} escaped={escaped} "
        f"trailing_dot_segment_control={stc} stored_inside_root={ctl_stored}")


@cell
def upload_nonimage_and_ext_mismatch():
    st1, h1, p1, _ = put_upload("U", f"{upload_sig('x.jpg')}/x.jpg", b"hello world")
    st2, h2, p2, _ = put_upload("U", f"{upload_sig('fake.jpg')}/fake.jpg", PNG_BYTES)
    st3, h3, p3, _ = put_upload("U", f"{upload_sig('real.png')}/real.png", PNG_BYTES)
    st4, h4, p4, _ = put_upload("U", f"{upload_sig('noext')}/noext", PNG_BYTES)
    st5, h5, p5, _ = put_upload("U", f"{upload_sig('page.html')}/page.html", PNG_BYTES)
    tree = uploads_tree()
    ok = (st1 == 415 and st2 == 415 and st3 == 201 and st4 == 201 and st5 == 415
          and "x.jpg" not in tree and "fake.jpg" not in tree)
    rec("upload-nonimage-and-ext-mismatch", "upload",
        "text-as-jpg / png-as-jpg / png-as-png / png-noext / png-as-html", "-",
        "PASS" if ok else "FAIL", None,
        f"text_jpg={st1} png_as_jpg={st2} png_as_png={st3} png_noext={st4} "
        f"png_as_html={st5} bodies={[p2[:40], p5[:40]]}")


@cell
def upload_duplicate_409():
    name = "dup/once.jpg"
    st1, _, _, _ = put_upload("U", f"{upload_sig(name)}/{name}", DOG_BYTES)
    st2, h2, p2, _ = put_upload("U", f"{upload_sig(name)}/{name}", PNG_BYTES)
    ok = st1 == 201 and st2 == 409
    rec("upload-duplicate-409", "upload", name, "dog then png, same name",
        "PASS" if ok else "FAIL", None,
        f"first={st1} second={st2} body={p2[:50]!r}")


@cell
def upload_expired_token():
    name = "exp/old.jpg"
    sig_p = upload_sig(name, PAST)
    st1, h1, p1, _ = put_upload("U", f"{sig_p}/exp:{PAST}/{name}", DOG_BYTES)
    name2 = "exp/new.jpg"
    sig_f = upload_sig(name2, FUTURE)
    st2, h2, p2, _ = put_upload("U", f"{sig_f}/exp:{FUTURE}/{name2}", DOG_BYTES)
    # expiry stripped: present the expiring token's signature without exp:
    st3, h3, p3, _ = put_upload("U", f"{sig_f}/{name2}", DOG_BYTES)
    ok = st1 == 403 and st2 == 201 and st3 == 403
    rec("upload-expired-token", "upload", "exp:<past> / exp:<future> / stripped",
        "dog.jpg bytes", "PASS" if ok else "FAIL", None,
        f"expired={st1} valid={st2} stripped_expiry_replay={st3}")


@cell
def upload_svg_sanitized():
    name = "vec/probe.svg"
    st, h, payload, ms = put_upload("U", f"{upload_sig(name)}/{name}", SVG_BYTES)
    stored_path = os.path.join(UPDIR, name)
    stored = open(stored_path).read() if os.path.isfile(stored_path) else ""
    fr = render_signed("upload-svg-sanitized", "U", "200x0/file:///vec/probe.svg")
    ok = (st == 201 and "<script" not in stored and "onclick" not in stored
          and "<svg" in stored and fr["status"] == 200
          and fr["ct"].startswith("image/png") and dims(fr) == (200, 100))
    check("upload-svg-sanitized", "upload", name, "probe.svg (script + onclick)",
          ok, fr, f"status={st} stored_has_script={'<script' in stored} "
          f"stored_has_onclick={'onclick' in stored} render={fr['status']} "
          f"ct={fr['ct']} dims={dims(fr)}")


# ---------------------------------------------------------------- sources
@cell
def src_localhost_and_127():
    rem_l = f"300x200/{SRCLOC}/dog.jpg"
    rem_i = f"300x200/{DOG}"
    fl = render_signed("src-localhost-and-127", "P", rem_l)
    fi = timed_render("src-localhost-and-127", "P", signed_path(rem_i))
    ok = fl["status"] == 200 and fi["status"] == 200 and fl["md5"] == fi["md5"]
    check("src-localhost-and-127", "sources", "300x200",
          "dog.jpg via localhost and via 127.0.0.1", ok, fl,
          f"localhost={fl['status']} ip={fi['status']} bytes_equal={fl['md5'] == fi['md5']}")


@cell
def src_query_string():
    enc_src = f"{SRC127}/dog.jpg%3Fv%3D42%26x%3D1"
    rem_enc = f"300x200/{enc_src}"
    hits0 = origin_hits("dog.jpg")
    fe = timed_render("src-query-string", "P", signed_path(rem_enc))
    hits1 = origin_hits("dog.jpg")
    # raw form: query on the crabcrop request itself, plain remainder
    rem_plain = f"300x200/{DOG}"
    st, h, payload, ms = get("P", signed_path(rem_plain) + "?v=42&x=1")
    hits2 = origin_hits("dog.jpg")
    lines = open(HITS_PATH).read().splitlines()
    enc_line = [ln for ln in lines if "v=42" in ln]
    note("src-query-string", "sources", "300x200", "dog.jpg?v=42&x=1", fe,
         f"encoded_form: status={fe['status']} origin_saw_query={enc_line[-1] if enc_line else None} | "
         f"request_query_form: status={st} new_origin_hits={hits2 - hits1} "
         f"last_hit={lines[-1] if lines else None}")


@cell
def src_encoded_remainder():
    hits0 = origin_hits()
    f1 = timed_render("src-encoded-remainder", "P", signed_path(
        f"300x200/http%3A%2F%2F127.0.0.1%3A{PORTS['origin']}%2Fdog.jpg"))
    f2 = timed_render("src-encoded-remainder", "P", signed_path(
        f"300x200/http%3A//127.0.0.1:{PORTS['origin']}/dog.jpg"))
    hits1 = origin_hits()
    note("src-encoded-remainder", "sources", "300x200",
         "percent-encoded source spellings", f1,
         f"fully_encoded={f1['status']} scheme_encoded={f2['status']} "
         f"origin_fetches={hits1 - hits0}: the source sits literally in the "
         f"path and percent-encoded spellings are not decoded (documented "
         f"grammar presents the source as-is; no fetch is attempted)")


@cell
def src_redirect_within():
    rem_r = f"300x200/{SRC127}/redir/3"
    rem_d = f"300x200/{DOG}"
    fr = render_signed("src-redirect-within", "P", rem_r)
    fd = render_signed("src-redirect-within", "P", rem_d)
    ok = fr["status"] == 200 and fr["md5"] == fd["md5"]
    check("src-redirect-within", "sources", "300x200", "redir/3 -> dog.jpg", ok, fr,
          f"status={fr['status']} bytes_equal_direct={fr['md5'] == fd['md5']} "
          f"server_ms={fr['server_ms']}")


@cell
def src_redirect_beyond_and_loop():
    fb = render_signed("src-redirect-beyond-and-loop", "P",
                       f"300x200/{SRC127}/redir/6")
    fl = render_signed("src-redirect-beyond-and-loop", "P",
                       f"300x200/{SRC127}/redir-loop")
    ok = fb["status"] == 502 and fl["status"] == 502
    check("src-redirect-beyond-and-loop", "sources", "300x200",
          "redir/6 and redir-loop", ok, fb,
          f"six_hops={fb['status']} body={fb['body'][:50]!r} loop={fl['status']}")


@cell
def src_redirect_out_denied():
    f = render_signed("src-redirect-out-denied", "P",
                      f"300x200/{SRC127}/redir-out")
    ok = f["status"] == 403
    check("src-redirect-out-denied", "sources", "300x200",
          "redir-out -> example.org (not allowlisted)", ok, f,
          f"status={f['status']} body={f['body'][:60]!r}")


@cell
def src_origin_errors():
    f404 = render_signed("src-origin-errors", "P", f"300x200/{SRC127}/missing.jpg")
    fbig = render_signed("src-origin-errors", "P", f"300x200/{SRC127}/big.jpg")
    fct = render_signed("src-origin-errors", "P", f"300x200/{SRC127}/notimage")
    fbb = render_signed("src-origin-errors", "P", f"300x200/{SRC127}/badbytes.jpg")
    ok = (f404["status"] == 502 and fbig["status"] == 502
          and fct["status"] == 502 and fbb["status"] == 502)
    check("src-origin-errors", "sources", "300x200",
          "missing / big / notimage / badbytes", ok, f404,
          f"404->{f404['status']} 11MiB->{fbig['status']} html_ct->{fct['status']} "
          f"bad_bytes->{fbb['status']}")


@cell
def src_slow_timeout():
    f = render_signed("src-slow-timeout", "P", f"300x200/{SRC127}/slow.jpg",
                      timeout=60)
    ok = f["status"] == 504 and 9000 < f["client_ms"] < 25000
    check("src-slow-timeout", "sources", "300x200", "slow.jpg (12 s origin)",
          ok, f, f"status={f['status']} client_ms={f['client_ms']} body={f['body'][:50]!r}")


@cell
def src_allowlist_mismatch():
    rem = f"300x200/{CAT}"
    hits0 = origin_hits("/cat.jpg")
    fe = timed_render("src-allowlist-mismatch", "E", signed_path(rem))
    hits1 = origin_hits("/cat.jpg")
    rem_l = f"300x200/{SRCLOC}/cat.jpg"
    fl = render_signed("src-allowlist-mismatch", "E", rem_l)
    ok = fe["status"] == 403 and hits1 == hits0 and fl["status"] == 200
    check("src-allowlist-mismatch", "sources", "300x200",
          "cat.jpg via 127.0.0.1 on E (localhost only)", ok, fe,
          f"denied={fe['status']} body={fe['body'][:50]!r} origin_fetches={hits1 - hits0} "
          f"localhost_control={fl['status']}")


@cell
def src_unsafe_with_secret():
    f = timed_render("src-unsafe-with-secret", "N", unsafe_path(f"300x200/{DOG}"))
    ok = f["status"] == 403 and "bad signature" in f["body"]
    check("src-unsafe-with-secret", "sources", "300x200", "dog.jpg on N (no unsafe)",
          ok, f, f"status={f['status']} body={f['body'][:40]!r}")


@cell
def src_userinfo_refused():
    rem = f"300x200/http://user:pass@127.0.0.1:{PORTS['origin']}/dog.jpg"
    f = timed_render("src-userinfo-refused", "P", signed_path(rem))
    ok = f["status"] == 403
    check("src-userinfo-refused", "sources", "300x200",
          "http://user:pass@127.0.0.1:{PORTS['origin']}/dog.jpg", ok, f,
          f"status={f['status']} body={f['body'][:60]!r}")


@cell
def src_file_traversal():
    f1 = timed_render("src-file-traversal", "U",
                      signed_path("200x0/file:///../../etc/passwd"))
    f2 = timed_render("src-file-traversal", "U",
                      signed_path("200x0/file:///heroes/../../presets.txt"))
    leaked = b"root:" in (f1.get("body", "").encode() or b"")
    ok = f1["status"] in (400, 403, 404, 502) and f2["status"] in (400, 403, 404, 502) \
        and not leaked
    check("src-file-traversal", "sources", "200x0",
          "file:// escapes from the upload root", ok, f1,
          f"etc_passwd={f1['status']} presets_escape={f2['status']}")


@cell
def src_svg_source():
    f = render_signed("src-svg-source", "P", f"{SRC127}/probe.svg")
    ok = (f["status"] == 200 and f["ct"].startswith("image/png")
          and dims(f) == (200, 100))
    check("src-svg-source", "sources", "(bare)", "probe.svg via HTTP", ok, f,
          f"status={f['status']} ct={f['ct']} dims={dims(f)}")


# ---------------------------------------------------------------- parity
PARITY_BASE = "https://images.example.com"


def parity_cases():
    S = SECRET
    cases = [
        {"id": "bu-basic", "kind": "build_url", "base": PARITY_BASE, "secret": S,
         "operations": "300x200", "source": "https://example.com/photo.jpg"},
        {"id": "bu-chain", "kind": "build_url", "base": PARITY_BASE, "secret": S,
         "operations": "crop:1200x630:smart/filters:format(webp):quality(80)",
         "source": "s3://bucket/key.jpg"},
        {"id": "bu-empty-ops", "kind": "build_url", "base": PARITY_BASE, "secret": S,
         "operations": "", "source": "https://example.com/a.jpg"},
        {"id": "bu-preset", "kind": "build_url", "base": PARITY_BASE, "secret": S,
         "operations": "preset:thumb", "source": "file:///x/y.png"},
        {"id": "bu-exp", "kind": "build_url", "base": PARITY_BASE, "secret": S,
         "operations": "300x200/filters:grayscale", "source": "https://example.com/photo.jpg",
         "expires": 1893456000},
        {"id": "bu-unsafe", "kind": "build_url", "base": PARITY_BASE, "secret": None,
         "operations": "preset:thumb/640x0", "source": "https://example.com/photo.jpg"},
        {"id": "bu-keyid", "kind": "build_url", "base": PARITY_BASE, "secret": S,
         "operations": "300x200", "source": "https://example.com/photo.jpg",
         "key_id": "2026-10"},
        {"id": "bu-base-slash", "kind": "build_url", "base": PARITY_BASE + "/",
         "secret": S, "operations": "100x0", "source": "gs://bkt/obj.png"},
        {"id": "ss-basic", "kind": "srcset", "base": PARITY_BASE, "secret": S,
         "source": "https://example.com/photo.jpg", "widths": [320, 640, 1280]},
        {"id": "ss-exp", "kind": "srcset", "base": PARITY_BASE, "secret": S,
         "source": "https://example.com/photo.jpg", "widths": [480],
         "expires": 1893456000},
        {"id": "ss-unsafe", "kind": "srcset", "base": PARITY_BASE, "secret": None,
         "source": "https://example.com/photo.jpg", "widths": [320, 640]},
        {"id": "sig-purge", "kind": "purge_sig", "secret": S,
         "target": "300x200/https://example.com/photo.jpg"},
        {"id": "sig-purge-preset", "kind": "purge_sig", "secret": S,
         "target": "preset:thumb/https://example.com/photo.jpg"},
        {"id": "sig-group", "kind": "group_sig", "secret": S, "group": "thumb"},
        {"id": "sig-upload", "kind": "upload_sig", "secret": S, "name": "heroes/photo.jpg"},
        {"id": "sig-upload-exp", "kind": "upload_sig", "secret": S,
         "name": "heroes/photo.jpg", "expires": 1893456000},
    ]
    vectors = json.load(open(os.path.join(REPO, "spec", "test-vectors.json")))
    for i, v in enumerate(vectors):
        if v.get("sealed"):
            cases.append({"id": f"vec-seal-{i}", "kind": "seal",
                          "secret": v["secret"],
                          "remainder": v["remainder"], "nonce": v["nonce"],
                          "key_id": v.get("key_id"),
                          "expect": v["token"]})
        else:
            cases.append({"id": f"vec-{i}", "kind": "sign", "secret": v["secret"],
                          "remainder": v["remainder"],
                          "expect": v["signature"]})
    return cases


def run_emitters(cases):
    path = os.path.join(OUT, "parity-cases.json")
    with open(path, "w") as fh:
        json.dump(cases, fh)
    env = dict(os.environ, NO_PROXY="127.0.0.1,localhost")
    py = json.loads(subprocess.run(
        [sys.executable, os.path.join(FUNC_DIR, "emit_py.py"), path],
        capture_output=True, text=True, timeout=120).stdout)
    ts = json.loads(subprocess.run(
        ["node", os.path.join(FUNC_DIR, "emit_ts.mjs"), path],
        capture_output=True, text=True, timeout=120).stdout)
    rs = json.loads(subprocess.run(
        [os.path.join(OUT, "parity-rs", "target", "release", "functional-parity-rs"), path],
        capture_output=True, text=True, timeout=120).stdout)
    return py, ts, rs


@cell
def parity_build_url_and_srcset():
    cases = [c for c in parity_cases() if c["kind"] in ("build_url", "srcset")]
    py, ts, rs = run_emitters(cases)
    diffs = []
    for c in cases:
        vals = {py.get(c["id"]), ts.get(c["id"]), rs.get(c["id"])}
        if len(vals) != 1:
            diffs.append(f"{c['id']}: py={py.get(c['id'])!r} ts={ts.get(c['id'])!r} "
                         f"rs={rs.get(c['id'])!r}")
    ok = not diffs
    rec("parity-build-url-and-srcset", "parity",
        f"{len(cases)} build_url/srcset cases x TS/Py/Rust", "-",
        "PASS" if ok else "FAIL", None,
        "byte-identical across all three SDKs" if ok else " | ".join(diffs)[:600])


@cell
def parity_signatures_and_vectors():
    cases = [c for c in parity_cases() if c["kind"] not in ("build_url", "srcset")]
    py, ts, rs = run_emitters(cases)
    diffs = []
    skipped_seal = 0
    for c in cases:
        if c["kind"] == "seal" and py.get(c["id"]) == "SKIP:no-extra":
            # The Python extra is absent in this run: the
            # sealed byte-parity rests on TS and Rust, and
            # the skip is reported, not hidden.
            skipped_seal += 1
            if ts.get(c["id"]) != c["expect"] or rs.get(c["id"]) != c["expect"]:
                diffs.append(f"{c['id']}: ts={ts.get(c['id'])!r} "
                             f"rs={rs.get(c['id'])!r} expect={c['expect']!r}")
            continue
        vals = {py.get(c["id"]), ts.get(c["id"]), rs.get(c["id"])}
        if len(vals) != 1:
            diffs.append(f"{c['id']}: py={py.get(c['id'])!r} ts={ts.get(c['id'])!r} "
                         f"rs={rs.get(c['id'])!r}")
        elif "expect" in c and py.get(c["id"]) != c["expect"]:
            diffs.append(f"{c['id']}: got {py.get(c['id'])}, spec expects {c['expect']}")
    ok = not diffs
    rec("parity-signatures-and-vectors", "parity",
        f"{len(cases)} sign/purge/group/upload/seal cases incl. spec vectors", "-",
        "PASS" if ok else "FAIL", None,
        ("byte-identical across all three SDKs and matching spec/test-vectors.json"
         + (f"; {skipped_seal} sealed vectors skipped for Python (no cryptography extra)"
            if skipped_seal else ""))
        if ok else " | ".join(diffs)[:600])


@cell
def parity_cli_sign():
    import crabcrop_url as sdk
    rem = f"300x200/filters:grayscale/{DOG}"
    rc, out = cli("sign", rem, "P")
    want = sdk.build_url(BASES["P"], SECRET, "300x200/filters:grayscale", DOG)
    ok = rc == 0 and out.splitlines()[0].strip() == want
    rec("parity-cli-sign", "parity", "crabcrop sign vs Python build_url", "dog.jpg",
        "PASS" if ok else "FAIL", None,
        f"cli={out[:110]!r} sdk={want[:110]!r} equal={out.splitlines()[0].strip() == want if out else False}")


# ---------------------------------------------------------------- sealed
def seal_path(rem, base="A", name=None, extra=None):
    """Mint a sealed path for a remainder with the CLI `seal`
    verb (issue #255). The harness mints through the CLI on
    purpose: the cells exercise the verb itself and never
    depend on the Python `cryptography` extra."""
    rc, out = cli("seal", rem, base, extra=extra)
    if rc != 0:
        raise AssertionError(f"cli seal failed: {out}")
    url = out.splitlines()[0].strip()
    path = url[len(BASES[base]):]
    if name:
        path = f"{path}/{name}"
    return path


@cell
def sealed_serve_parity():
    # One render, two spellings, one cache entry: the
    # second request is a hit whichever order they arrive
    # in, and the bytes and ETag are identical.
    rem = f"300x200/{DOG}"
    purge_cli(rem, "A")
    sp = seal_path(rem, "A", name="dog.jpg")
    fp = timed_render("sealed-serve-parity", "A", signed_path(rem))
    fs = timed_render("sealed-serve-parity", "A", sp)
    ok = (fp["status"] == 200 and fs["status"] == 200
          and fp["md5"] == fs["md5"] and fp["etag"] == fs["etag"]
          and fs["cache_status"].endswith("hit"))
    rem2 = f"240x160/{CAT}"
    purge_cli(rem2, "A")
    sp2 = seal_path(rem2, "A")
    fs2 = timed_render("sealed-serve-parity", "A", sp2)
    fp2 = timed_render("sealed-serve-parity", "A", signed_path(rem2))
    ok = (ok and fs2["status"] == 200 and fp2["status"] == 200
          and fs2["md5"] == fp2["md5"]
          and fp2["cache_status"].endswith("hit"))
    check("sealed-serve-parity", "sealed",
          "plain then sealed, sealed then plain", "dog.jpg/cat.jpg",
          ok, fs,
          f"plain={fp['status']}/{fp['cache_status']} sealed={fs['status']}/{fs['cache_status']} "
          f"md5_equal={fp['md5'] == fs['md5']} etag_equal={fp['etag'] == fs['etag']} "
          f"reverse: sealed={fs2['cache_status']} plain={fp2['cache_status']}")


@cell
def sealed_uniform_errors():
    rem = f"300x200/{DOG}"
    sp = seal_path(rem, "A", name="dog.jpg")
    token = sp.split("/")[1]
    flipped = token[:10] + ("B" if token[10] != "B" else "A") + token[11:]
    cases = {
        "flipped-ciphertext": flipped,
        "unknown-key-id": seal_path(rem, "A", extra=["--key-id", "zz"]),
        "garbage-segment": "s1.!!!",
        "unknown-version": "s2.AAAA",
        "oversized-segment": "s1." + "A" * 5000,
    }
    got = {}
    for label, path in cases.items():
        st, h, payload, ms = get("A", path if path.startswith("/") else f"/{path}")
        got[label] = (st, payload.decode("utf-8", "replace").strip())
    ok = all(v == (403, "bad signature") for v in got.values())
    # The server is healthy after the refusals, and the
    # genuine token still serves.
    st, h, payload, ms = get("A", sp)
    ok = ok and st == 200
    check("sealed-uniform-errors", "sealed",
          "flipped / unknown key id / garbage / unknown version / oversized",
          "dog.jpg", ok,
          fact_of("sealed-uniform-errors", "A", "GET", sp, st, h, payload, ms),
          f"answers={got} genuine_after={st}")


@cell
def sealed_keyed_current():
    # The default current key ID (no CRABCROP_SECRET_KEY_ID
    # is set on these instances, so it is `current`): a
    # token stamped with it opens; a plain signature does
    # not suddenly need a key ID.
    rem = f"300x200/{DOG}"
    purge_cli(rem, "A")
    sp = seal_path(rem, "A", extra=["--key-id", "current"])
    f = timed_render("sealed-keyed-current", "A", sp)
    ok = f["status"] == 200 and f["cache_status"].endswith("fwd=uri-miss")
    check("sealed-keyed-current", "sealed", "keyed token under the current key id",
          "dog.jpg", ok, f, f"status={f['status']} cache={f['cache_status']}")


@cell
def sealed_purge_cross_form():
    # Purge in one spelling evicts the entry the other
    # spelling serves: the canonicalization chain unseals
    # sealed targets before keying.
    rem = f"100x50/{DOG}"
    sp = seal_path(rem, "A", name="dog.jpg")
    target = sp[1:]
    purge_cli(rem, "A")
    f1 = timed_render("sealed-purge-cross-form", "A", signed_path(rem))
    st, body, h = purge_http(target, purge_sig(target), "A")
    ok1 = st == 200 and "purged 1" in body
    f2 = timed_render("sealed-purge-cross-form", "A", signed_path(rem))
    ok1 = ok1 and f2["cache_status"].endswith("fwd=uri-miss")
    purge_cli(rem, "A")
    fs1 = timed_render("sealed-purge-cross-form", "A", sp)
    st, body, h = purge_http(rem, purge_sig(rem), "A")
    ok2 = st == 200 and "purged 1" in body
    fs2 = timed_render("sealed-purge-cross-form", "A", sp)
    ok2 = ok2 and fs2["cache_status"].endswith("fwd=uri-miss")
    check("sealed-purge-cross-form", "sealed",
          "purge sealed evicts plain, purge plain evicts sealed", "dog.jpg",
          ok1 and ok2, fs2,
          f"plain-side: {f1['cache_status']} -> {f2['cache_status']} "
          f"sealed-side: {fs1['cache_status']} -> {fs2['cache_status']}")


@cell
def sealed_meta_redaction():
    rem = f"300x200/{DOG}"
    sp = seal_path(rem, "A", name="dog.jpg")
    st, h, payload, ms = get("A", sp, {"Accept": "application/json"})
    j = {}
    try:
        j = json.loads(payload)
    except Exception:
        pass
    text = payload.decode("utf-8", "replace")
    ok = (st == 200 and j.get("sealed") is True and j.get("width") == 300
          and j.get("height") == 200 and "dog.jpg" not in text
          and "300x200" not in text)
    # The plain meta of the same render carries no marker.
    st2, h2, payload2, ms2 = get("A", signed_path(rem), {"Accept": "application/json"})
    j2 = {}
    try:
        j2 = json.loads(payload2)
    except Exception:
        pass
    ok = ok and st2 == 200 and "sealed" not in j2 and j2.get("width") == 300
    check("sealed-meta-redaction", "sealed", "meta: sealed:true, no recipe text",
          "dog.jpg", ok, fact_of("sealed-meta-redaction", "A", "GET", sp, st, h, payload, ms),
          f"sealed_meta={j.get('sealed')} plain_has_sealed={'sealed' in j2} "
          f"sealed_doc={text[:220]}")


@cell
def sealed_explain_refusal():
    hits0 = origin_hits("/dog.jpg")
    rem = f"300x200/{DOG}"
    sp = seal_path(rem, "A", name="dog.jpg")
    st, h, payload, ms = get("A", f"/explain{sp}")
    hits1 = origin_hits("/dog.jpg")
    ok = (st == 403 and payload.decode("utf-8", "replace").strip() == "bad signature"
          and hits1 == hits0)
    check("sealed-explain-refusal", "sealed", "explain refuses, zero source fetches",
          "dog.jpg", ok, fact_of("sealed-explain-refusal", "A", "GET", f"/explain{sp}",
                                  st, h, payload, ms),
          f"status={st} origin_fetches={hits1 - hits0}")


@cell
def sealed_expired():
    rem = f"exp:{PAST}/300x200/{DOG}"
    sp = seal_path(rem, "A")
    st, h, payload, ms = get("A", sp)
    ok = (st == 403 and payload.decode("utf-8", "replace").strip() == "signature expired")
    check("sealed-expired", "sealed", "opened token, expired payload", "dog.jpg",
          ok, fact_of("sealed-expired", "A", "GET", sp, st, h, payload, ms),
          f"status={st} body={payload.decode('utf-8', 'replace').strip()}")


@cell
def sealed_presets_only_uniform():
    # The presets-only instance answers a freeform sealed
    # payload with the uniform refusal, not the plain
    # form's 400 presets_only, while a sealed preset URL
    # serves.
    rem = f"300x200/{DOG}"
    sp = seal_path(rem, "OP")
    st, h, payload, ms = get("OP", sp)
    ok = (st == 403 and payload.decode("utf-8", "replace").strip() == "bad signature")
    rem_p = f"preset:thumb/{DOG}"
    sp_p = seal_path(rem_p, "OP")
    st_p, h_p, payload_p, ms_p = get("OP", sp_p)
    ok = ok and st_p == 200
    check("sealed-presets-only-uniform", "sealed",
          "presets-only: sealed freeform refused, sealed preset serves", "dog.jpg",
          ok, fact_of("sealed-presets-only-uniform", "OP", "GET", sp, st, h, payload, ms),
          f"freeform={st} preset={st_p}")


@cell
def sealed_payload_failure_uniform():
    # Payload failures after a successful open: unknown
    # preset and unknown source root answer the uniform
    # refusal. A pure grammar failure is minted with the
    # Python SDK instead of the CLI (the CLI validates
    # remainders at mint time, by design, so it refuses to
    # seal one), and is skipped with a note when the
    # cryptography extra is absent.
    got = {}
    last = None
    for label, rem, base in [("unknown-preset", f"preset:nope/{DOG}", "A"),
                             ("unknown-root", "300x200/root:nope/cat.jpg", "R")]:
        sp = seal_path(rem, base)
        st, h, payload, ms = get(base, sp)
        got[label] = (st, payload.decode("utf-8", "replace").strip())
        last = (sp, st, h, payload, ms)
    grammar_note = "grammar case skipped (no cryptography extra)"
    try:
        import crabcrop_url as sdk
        token = sdk.seal_url(SECRET, f"filters:quality(abc)/{DOG}")
        st, h, payload, ms = get("A", f"/{token}")
        got["grammar"] = (st, payload.decode("utf-8", "replace").strip())
        grammar_note = f"grammar={got['grammar']}"
    except Exception as e:
        if "cryptography" not in str(e):
            raise
    ok = all(v == (403, "bad signature") for v in got.values())
    sp, st, h, payload, ms = last
    check("sealed-payload-failure-uniform", "sealed",
          "unknown preset / unknown root / grammar after open", "dog.jpg",
          ok, fact_of("sealed-payload-failure-uniform", "A", "GET", sp, st, h, payload, ms),
          f"answers={got} {grammar_note}")


# ---------------------------------------------------------------- MCP
@cell
def mcp_tools_and_sign():
    import crabcrop_url as sdk
    m = Mcp(BASES["P"], SECRET)
    try:
        tools = m.tools()
        names = sorted(t["name"] for t in tools)
        r1 = m.call("sign_url", {"operations": "300x200", "source": DOG})
        want1 = sdk.build_url(BASES["P"], SECRET, "300x200", DOG)
        r2 = m.call("sign_url", {"operations": "preset:thumb", "source": DOG,
                                 "expires": FUTURE})
        want2 = sdk.build_url(BASES["P"], SECRET, "preset:thumb", DOG,
                              expires=FUTURE)
        ok = (names == ["explain_url", "fetch_meta", "purge_result", "sign_url"]
              and not r1["isError"] and r1["text"] == want1
              and not r2["isError"] and r2["text"] == want2)
        rec("mcp-tools-and-sign", "mcp", "tools/list + sign_url x2", "dog.jpg",
            "PASS" if ok else "FAIL", None,
            f"tools={names} sign_equal={r1['text'] == want1} "
            f"sign_exp_equal={r2['text'] == want2} r1={r1['text'][:90]!r}")
    finally:
        m.close()


@cell
def mcp_explain_and_meta():
    import crabcrop_url as sdk
    url = sdk.build_url(BASES["P"], SECRET, "300x200", DOG)
    m = Mcp(BASES["P"], SECRET)
    try:
        r1 = m.call("explain_url", {"url": url})
        r2 = m.call("explain_url", {"url": "http://example.com/not/a/url"})
        r3 = m.call("fetch_meta", {"url": url})
        r4 = m.call("fetch_meta", {"url": f"{SRC127}/dog.jpg"})
        j = {}
        try:
            j = json.loads(r3["text"])
        except Exception:
            pass
        ok = (not r1["isError"] and "300x200" in r1["text"]
              and r2["isError"]
              and not r3["isError"] and j.get("width") == 300
              and j.get("height") == 200 and j.get("format") == "jpeg"
              and r4["isError"])
        rec("mcp-explain-and-meta", "mcp", "explain_url + fetch_meta", "dog.jpg",
            "PASS" if ok else "FAIL", None,
            f"explain_ok={not r1['isError']} explain_bad_is_error={r2['isError']} "
            f"meta={j.get('width')}x{j.get('height')}/{j.get('format')} "
            f"foreign_origin_refused={r4['isError']}")
    finally:
        m.close()


@cell
def mcp_purge_result():
    rem = f"317x217/{CAR}"
    purge_cli(rem, "P")
    timed_render("mcp-purge-result", "P", signed_path(rem))
    f2 = timed_render("mcp-purge-result", "P", signed_path(rem))
    m = Mcp(BASES["P"], SECRET)
    try:
        r = m.call("purge_result", {"path": rem})
        rdot = m.call("purge_result", {"path": "../secret"})
    finally:
        m.close()
    f3 = timed_render("mcp-purge-result", "P", signed_path(rem))
    ok = ("hit" in f2["cache_status"] and not r["isError"]
          and f3["cache_status"].endswith("fwd=uri-miss") and rdot["isError"])
    check("mcp-purge-result", "mcp", "purge_result", "car.jpg", ok, f3,
          f"purge_text={r['text'][:60]!r} dot_segment_refused={rdot['isError']} "
          f"after={f3['cache_status']}")


# ---------------------------------------------------------------- frameworks
@cell
def framework_parity():
    env = dict(os.environ, FUNC_FW_BASE=BASES["P"], FUNC_FW_SECRET=SECRET,
               FUNC_FW_SRC=DOG, NO_PROXY="127.0.0.1,localhost")
    r = subprocess.run(["node", os.path.join(FUNC_DIR, "emit_fw.mjs")],
                       capture_output=True, text=True, timeout=120, env=env)
    try:
        data = json.loads(r.stdout)
    except Exception:
        rec("framework-parity", "frameworks", "next/nuxt/hydrogen loaders", "-",
            "FAIL", None, f"emitter failed: rc={r.returncode} err={r.stderr[:300]!r}")
        return
    groups = {"next": [], "nuxt": [], "hydro": []}
    for k, want in data["expected"].items():
        fw = k.split("-")[0]
        got = data["out"].get(k)
        groups[fw].append((k, got == want, got, want))
    for fw, items in groups.items():
        bad = [i for i in items if not i[1]]
        rec(f"framework-{fw}", "frameworks", f"{len(items)} loader cases", "-",
            "PASS" if not bad else "FAIL", None,
            "all outputs byte-identical to TS SDK buildURL" if not bad
            else "; ".join(f"{k}: got {g!r} want {w!r}" for k, _, g, w in bad)[:500])


# ---------------------------------------------------------------- timing
@cell
def timing_families():
    import statistics
    miss_ms, hit_ms = [], []
    for wpx in (901, 902, 903, 904, 905):
        rem = f"{wpx}x0/{DOG}"
        f = render_signed("timing-families", "P", rem)
        if f["server_ms"] is not None:
            miss_ms.append(f["server_ms"])
    rem = f"906x0/{DOG}"
    render_signed("timing-families", "P", rem)
    for _ in range(5):
        f = timed_render("timing-families", "P", signed_path(rem))
        hit_ms.append(f["client_ms"])
    med_miss = statistics.median(miss_ms) if miss_ms else None
    med_hit = statistics.median(hit_ms) if hit_ms else None
    ok = med_hit is not None and med_hit < 25
    note("timing-families", "timing", "9xx-width renders + repeats", "dog.jpg",
         None, f"warm render (cache miss) server_ms samples={miss_ms} median={med_miss} | "
         f"cache hit client_ms samples={hit_ms} median={med_hit} | "
         f"hit_median_under_25ms={ok}")




@cell
def parity_vectors_py():
    """Smoke-tier parity: the repo's published spec vectors through the
    Python SDK emitter only (no node, no cargo). The full three-SDK
    byte comparison is parity-signatures-and-vectors (full tier)."""
    cases = [c for c in parity_cases() if c["kind"] in ("sign", "seal")]
    path = os.path.join(OUT, "parity-cases-py.json")
    with open(path, "w") as fh:
        json.dump(cases, fh)
    env = dict(os.environ, NO_PROXY="127.0.0.1,localhost")
    r = subprocess.run(
        [sys.executable, os.path.join(FUNC_DIR, "emit_py.py"), path],
        capture_output=True, text=True, timeout=120, env=env)
    try:
        py = json.loads(r.stdout)
    except Exception:
        rec("parity-vectors-py", "parity", "spec vectors via Python SDK",
            "-", "FAIL", None,
            f"emitter failed: rc={r.returncode} err={r.stderr[:300]!r}")
        return
    diffs = [f"{c['id']}: got {py.get(c['id'])}, spec expects {c['expect']}"
             for c in cases
             if py.get(c["id"]) != c["expect"] and py.get(c["id"]) != "SKIP:no-extra"]
    skipped = sum(1 for c in cases if py.get(c["id"]) == "SKIP:no-extra")
    ok = not diffs
    rec("parity-vectors-py", "parity",
        f"{len(cases)} published sign and sealed vectors", "-",
        "PASS" if ok else "FAIL", None,
        ("python SDK matches spec/test-vectors.json on every vector"
         + (f"; {skipped} sealed vectors skipped (no cryptography extra)"
            if skipped else ""))
        if ok else " | ".join(diffs)[:600])


# ---------------------------------------------------------------- video frames
CLIP = f"{SRC127}/clip.mp4"
AUDIO = f"{SRC127}/audio.mp4"
BROKEN = f"{SRC127}/broken.mp4"


def _frame_note(cell_id, ops, src):
    note(cell_id, "frames", ops, src, None,
         "no ffmpeg on PATH: the V instance has no decoder configured, "
         "so frame cells record instead of asserting")


def _center_pixel(fact):
    im = Image.open(fact["path"]).convert("RGB")
    return im.getpixel((im.width // 2, im.height // 2))


def _is_red(px):
    return px[0] > 200 and px[1] < 60 and px[2] < 60


def _is_green(px):
    return px[1] > 100 and px[0] < 60 and px[2] < 60


def _is_blue(px):
    return px[2] > 200 and px[0] < 60 and px[1] < 60


@cell
def frames_poster_and_timestamps():
    # The corpus clip is a solid color per second (red,
    # green, blue at 64x48), so the selected frame is
    # provable from its center pixel (issue #214).
    if not shutil.which("ffmpeg"):
        return _frame_note("frames-poster-and-timestamps", "frame:*", "clip.mp4")
    cases = [("frame:poster", _is_red, "red"),
             ("frame:0.5", _is_red, "red"),
             ("frame:1.5", _is_green, "green"),
             ("frame:2.5", _is_blue, "blue")]
    facts = []
    ok = True
    for seg, pred, name in cases:
        f = render_signed("frames-poster-and-timestamps", "V", f"{seg}/{CLIP}")
        px = _center_pixel(f) if f["status"] == 200 else None
        facts.append(f"{seg}: status={f['status']} px={px} ({name})")
        ok = ok and f["status"] == 200 and dims(f) == (64, 48) \
            and px is not None and pred(px)
    # An operation chain runs on the frame like any source.
    f = render_signed("frames-poster-and-timestamps", "V",
                      f"frame:1.5/32x24/filters:format(webp)/{CLIP}")
    facts.append(f"chain: status={f['status']} dims={dims(f)} ct={f['ct']}")
    ok = ok and f["status"] == 200 and dims(f) == (32, 24) \
        and f["ct"].startswith("image/webp")
    check("frames-poster-and-timestamps", "frames",
          "frame:poster|frame:<seconds>", "clip.mp4", ok, f, " | ".join(facts))


@cell
def frames_percent_and_clamp():
    if not shutil.which("ffmpeg"):
        return _frame_note("frames-percent-and-clamp", "frame:<percent>%", "clip.mp4")
    mid = render_signed("frames-percent-and-clamp", "V", f"frame:50%/{CLIP}")
    mid_px = _center_pixel(mid) if mid["status"] == 200 else None
    end = render_signed("frames-percent-and-clamp", "V", f"frame:100%/{CLIP}")
    end_px = _center_pixel(end) if end["status"] == 200 else None
    past = render_signed("frames-percent-and-clamp", "V", f"frame:999/{CLIP}")
    past_px = _center_pixel(past) if past["status"] == 200 else None
    ok = (mid["status"] == 200 and mid_px is not None and _is_green(mid_px)
          and end["status"] == 200 and end_px is not None and _is_blue(end_px)
          and past["status"] == 200 and past_px is not None and _is_blue(past_px))
    check("frames-percent-and-clamp", "frames", "frame:50%|frame:100%|frame:999",
          "clip.mp4", ok, past,
          f"50%: status={mid['status']} px={mid_px} (green) | "
          f"100%: status={end['status']} px={end_px} (blue, clamped) | "
          f"999s: status={past['status']} px={past_px} (blue, clamped)")


@cell
def frames_meta():
    if not shutil.which("ffmpeg"):
        return _frame_note("frames-meta", "frame:1.5 + meta", "clip.mp4")
    rem = f"frame:1.5/{CLIP}"
    st, h, payload, ms = get("V", signed_path(rem),
                             {"Accept": "application/json"})
    meta = json.loads(payload) if st == 200 else {}
    video = meta.get("video") or {}
    ok = (st == 200 and video.get("duration") == 3.0
          and video.get("width") == 64 and video.get("height") == 48
          and video.get("frame_timestamp") == 1.5
          and meta.get("source_width") == 64
          and meta.get("source_height") == 48)
    check("frames-meta", "frames", "frame:1.5 (meta)", "clip.mp4", ok,
          {"status": st},
          f"status={st} video={video} source={meta.get('source_width')}x"
          f"{meta.get('source_height')}")


@cell
def frames_refusals():
    if not shutil.which("ffmpeg"):
        return _frame_note("frames-refusals", "frame:*", "clip.mp4 et al")
    results = []
    # A frame segment on an image source: 502, video-only.
    f = timed_render("frames-refusals", "V", signed_path(f"frame:poster/{DOG}"))
    results.append(f"image-source: {f['status']} (want 502)")
    ok = f["status"] == 502
    # An audio-only file and a truncated container: no
    # video stream, the bad-source 502.
    for src, label in ((AUDIO, "audio-only"), (BROKEN, "broken")):
        f = timed_render("frames-refusals", "V", signed_path(f"frame:poster/{src}"))
        results.append(f"{label}: {f['status']} (want 502)")
        ok = ok and f["status"] == 502
    # A malformed selector is a grammar 400 with the
    # frozen code, and costs no origin fetch.
    hits0 = origin_hits("clip.mp4")
    st, h, payload, ms = get("V", signed_path(f"frame:banana/{CLIP}"),
                             {"Accept": "application/json"})
    code = (json.loads(payload).get("error") or {}).get("code")
    results.append(f"malformed: {st} code={code} "
                   f"new_hits={origin_hits('clip.mp4') - hits0} (want 400 invalid_frame 0)")
    ok = ok and st == 400 and code == "invalid_frame" \
        and origin_hits("clip.mp4") == hits0
    # No decoder configured (instance A): 501.
    f = timed_render("frames-refusals", "A", signed_path(f"frame:poster/{CLIP}"))
    results.append(f"no-decoder: {f['status']} (want 501)")
    ok = ok and f["status"] == 501
    check("frames-refusals", "frames", "frame:poster|frame:banana",
          "dog.jpg|audio.mp4|broken.mp4|clip.mp4", ok, f, " | ".join(results))


@cell
def frames_cache_and_purge():
    if not shutil.which("ffmpeg"):
        return _frame_note("frames-cache-and-purge", "frame:2.5", "clip.mp4")
    rem = f"frame:2.5/32x24/{CLIP}"
    purge_cli(rem, "V")
    hits0 = origin_hits("clip.mp4")
    f1 = timed_render("frames-cache-and-purge", "V", signed_path(rem))
    hits1 = origin_hits("clip.mp4")
    f2 = timed_render("frames-cache-and-purge", "V", signed_path(rem))
    hits2 = origin_hits("clip.mp4")
    st, body, _h = purge_http(rem, purge_sig(rem), base="V")
    f3 = timed_render("frames-cache-and-purge", "V", signed_path(rem))
    hits3 = origin_hits("clip.mp4")
    ok = (f1["status"] == 200 and hits1 - hits0 == 1
          and f2["status"] == 200 and hits2 == hits1
          and f2["cache"] == "hit"
          and st == 200 and f3["status"] == 200 and hits3 - hits2 == 1)
    check("frames-cache-and-purge", "frames", "frame:2.5/32x24", "clip.mp4",
          ok, f3,
          f"miss: hits+{hits1 - hits0} cache={f1['cache']} | "
          f"repeat: hits+{hits2 - hits1} cache={f2['cache']} | "
          f"purge: {st} {body} | after: hits+{hits3 - hits2}")


# ---------------------------------------------------------------- PDF pages
PAGES = f"{SRC127}/pages.pdf"
LOCKED = f"{SRC127}/locked.pdf"
TRUNCATED = f"{SRC127}/truncated.pdf"


def _page_note(cell_id, ops, src):
    note(cell_id, "pages", ops, src, None,
         "no pdftocairo on PATH: the D instance has no renderer "
         "configured, so pages cells record instead of asserting")


def _band_color(i):
    """The corpus band color of page i (the generator's
    formula), as an RGB triple."""
    return tuple(round(c * 255) for c in
                 (((i * 37) % 100) / 100, ((i * 53) % 100) / 100,
                  ((i * 91) % 100) / 100))


def _band_pixel(fact):
    # The band sits at (72..540, 560..620) points; at the
    # fixed render scale its center lands at (612, 404)
    # of the 1224x1584 page raster.
    im = Image.open(fact["path"]).convert("RGB")
    return im.getpixel((612, 404))


def _near(px, want, tol=14):
    return all(abs(a - b) <= tol for a, b in zip(px, want))


@cell
def pages_render_and_chain():
    # The corpus document carries a per-page hue band,
    # so the selected page is provable from the band's
    # center pixel (issue #215).
    if not shutil.which("pdftocairo"):
        return _page_note("pages-render-and-chain", "page:*", "pages.pdf")
    cases = [(1, _band_color(1)), (3, _band_color(3)), (5, _band_color(5))]
    facts = []
    ok = True
    for page, want in cases:
        f = render_signed("pages-render-and-chain", "D", f"page:{page}/{PAGES}")
        px = _band_pixel(f) if f["status"] == 200 else None
        facts.append(f"page:{page}: status={f['status']} px={px} (want {want})")
        ok = ok and f["status"] == 200 and dims(f) == (1224, 1584) \
            and px is not None and _near(px, want)
    # An operation chain runs on the page like any source.
    f = render_signed("pages-render-and-chain", "D",
                      f"page:2/306x396/filters:format(webp)/{PAGES}")
    facts.append(f"chain: status={f['status']} dims={dims(f)} ct={f['ct']}")
    ok = ok and f["status"] == 200 and dims(f) == (306, 396) \
        and f["ct"].startswith("image/webp")
    check("pages-render-and-chain", "pages",
          "page:1|page:3|page:5", "pages.pdf", ok, f, " | ".join(facts))


@cell
def pages_meta():
    if not shutil.which("pdftocairo"):
        return _page_note("pages-meta", "page:3 + meta", "pages.pdf")
    rem = f"page:3/{PAGES}"
    st, h, payload, ms = get("D", signed_path(rem),
                             {"Accept": "application/json"})
    meta = json.loads(payload) if st == 200 else {}
    pdf = meta.get("pdf") or {}
    ok = (st == 200 and pdf.get("pages") == 5
          and pdf.get("page") == 3
          and pdf.get("page_width") == 612
          and pdf.get("page_height") == 792
          and meta.get("source_width") == 1224
          and meta.get("source_height") == 1584)
    check("pages-meta", "pages", "page:3 (meta)", "pages.pdf", ok,
          {"status": st},
          f"status={st} pdf={pdf} source={meta.get('source_width')}x"
          f"{meta.get('source_height')}")


@cell
def pages_refusals():
    if not shutil.which("pdftocairo"):
        return _page_note("pages-refusals", "page:*", "pages.pdf et al")
    results = []
    ok = True
    # Out of range: a 400 refusal, never the last page
    # served under the asked number.
    st, h, payload, ms = get("D", signed_path(f"page:6/{PAGES}"), {})
    results.append(f"out-of-range: {st} (want 400)")
    ok = ok and st == 400 and b"page 6" in payload
    # A password-protected document: its own 400.
    st, h, payload, ms = get("D", signed_path(f"page:1/{LOCKED}"), {})
    results.append(f"locked: {st} (want 400)")
    ok = ok and st == 400 and b"password" in payload
    # A page segment on an image source: 502, PDF-only.
    f = timed_render("pages-refusals", "D", signed_path(f"page:1/{DOG}"))
    results.append(f"image-source: {f['status']} (want 502)")
    ok = ok and f["status"] == 502
    # A truncated document the probe cannot read, and a
    # video source (not a PDF at all): the bad-source 502.
    for src, label in ((TRUNCATED, "truncated"), (CLIP, "video-source")):
        f = timed_render("pages-refusals", "D", signed_path(f"page:1/{src}"))
        results.append(f"{label}: {f['status']} (want 502)")
        ok = ok and f["status"] == 502
    # A malformed selector is a grammar 400 with the
    # frozen code, and costs no origin fetch.
    hits0 = origin_hits("pages.pdf")
    st, h, payload, ms = get("D", signed_path(f"page:banana/{PAGES}"),
                             {"Accept": "application/json"})
    code = (json.loads(payload).get("error") or {}).get("code")
    results.append(f"malformed: {st} code={code} "
                   f"new_hits={origin_hits('pages.pdf') - hits0} "
                   f"(want 400 invalid_page 0)")
    ok = ok and st == 400 and code == "invalid_page" \
        and origin_hits("pages.pdf") == hits0
    # No renderer configured (instance A): 501.
    f = timed_render("pages-refusals", "A", signed_path(f"page:1/{PAGES}"))
    results.append(f"no-renderer: {f['status']} (want 501)")
    ok = ok and f["status"] == 501
    check("pages-refusals", "pages", "page:6|page:banana|page:1",
          "pages.pdf|locked.pdf|truncated.pdf|dog.jpg|clip.mp4", ok, f,
          " | ".join(results))


@cell
def pages_cache_and_purge():
    if not shutil.which("pdftocairo"):
        return _page_note("pages-cache-and-purge", "page:2", "pages.pdf")
    rem = f"page:2/306x396/{PAGES}"
    purge_cli(rem, "D")
    hits0 = origin_hits("pages.pdf")
    f1 = timed_render("pages-cache-and-purge", "D", signed_path(rem))
    hits1 = origin_hits("pages.pdf")
    f2 = timed_render("pages-cache-and-purge", "D", signed_path(rem))
    hits2 = origin_hits("pages.pdf")
    # The zero-padded spelling is the same request after
    # canonicalization: same entry, no new fetch.
    alt = f"page:02/306x396/{PAGES}"
    f2b = timed_render("pages-cache-and-purge", "D", signed_path(alt))
    hits2b = origin_hits("pages.pdf")
    st, body, _h = purge_http(rem, purge_sig(rem), base="D")
    f3 = timed_render("pages-cache-and-purge", "D", signed_path(rem))
    hits3 = origin_hits("pages.pdf")
    ok = (f1["status"] == 200 and hits1 - hits0 == 1
          and f2["status"] == 200 and hits2 == hits1
          and f2["cache"] == "hit"
          and f2b["status"] == 200 and f2b["cache"] == "hit"
          and hits2b == hits1
          and st == 200 and f3["status"] == 200 and hits3 - hits2 == 1)
    check("pages-cache-and-purge", "pages", "page:2/306x396", "pages.pdf",
          ok, f3,
          f"miss: hits+{hits1 - hits0} cache={f1['cache']} | "
          f"repeat: hits+{hits2 - hits1} cache={f2['cache']} | "
          f"page:02: cache={f2b['cache']} hits+{hits2b - hits1} | "
          f"purge: {st} {body} | after: hits+{hits3 - hits2}")
