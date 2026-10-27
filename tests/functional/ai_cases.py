#!/usr/bin/env python3
"""Scripted AI cases for the functional harness (issue #232): the
LOCAL tier. Ported from wave 3 of the feature campaign (removebg,
studio, face, meta AI surfaces, detectors, moderation signals, the
AI concurrency gate, upscale).

These cells never run in CI: the models are deployment artifacts,
never repo content. Run them locally with models configured:

    CRABCROP_FUNC_MODELS_DIR=<dir with the campaign model files> \
    CRABCROP_ORT_DYLIB=<libonnxruntime path> \
    python3 run.py --tier local

Instances: AI (all models configured), AIN (nothing configured),
AIC (CLIP B/32 encoder with the mismatched L/14 aesthetic head and
label-only moderation signals). See README.md for the model file
list. Two cells encode known-defect probes from the campaign and are
expected to pass on current builds: rb-2000-webp and st-bg-none-webp
record #227 sightings as NOTEs; st-nosubject-gradient and
upscale-oom-probe assert the #243 and #244 fixes hold (a process
kill fails the cell and the rest of the AI instance's cells with
connection errors).
"""
import hashlib, json, os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (render, meta, rec, check, note, open_img,
                     alpha_stats, pixel, region_mean_p as region_mean,
                     local_var_p as local_var, distinct_colors,
                     color_count, mean_diff, decode_embedding, RESULTS,
                     CORPUS, LOGS)

CELLS = {}
def cell(fn):
    CELLS[fn.__name__.replace("_", "-")] = fn
    return fn

BRAND = (14, 124, 123)  # #0E7C7B

def corners_bg(path, want, tol=12):
    im = open_img(path); w, h = im.size
    pts = [(4, 4), (w - 5, 4), (4, h - 5), (w - 5, h - 5)]
    got = [pixel(path, x, y)[:3] for x, y in pts]
    ok = all(all(abs(g - wn) <= tol for g, wn in zip(px, want)) for px in got)
    return ok, got

# ---------------------------------------------------------------- removebg
@cell
def rb_png_portrait():
    f = render("rb-png-portrait", "800x0/filters:removebg:format(png)", "person-portrait.jpg")
    a = alpha_stats(f["path"]) if f["path"] else {}
    ok = f["status"] == 200 and a.get("has_alpha") and 0.15 < a["opaque"] < 0.95 and a["partial"] > 0.005
    check("rb-png-portrait", "removebg", "800x0/filters:removebg:format(png)", "person-portrait.jpg",
          ok, f, f"status={f['status']} server_ms={f['server_ms']} alpha={a}")

@cell
def rb_png_dog():
    f = render("rb-png-dog", "filters:removebg:format(png)", "dog.jpg")
    a = alpha_stats(f["path"]) if f["path"] else {}
    ok = f["status"] == 200 and a.get("has_alpha") and 0.2 < a["opaque"] < 0.9
    check("rb-png-dog", "removebg", "filters:removebg:format(png)", "dog.jpg",
          ok, f, f"status={f['status']} server_ms={f['server_ms']} alpha={a}")

@cell
def rb_webp_lossless():
    f = render("rb-webp-lossless", "600x0/filters:removebg:format(webp):quality(100)", "dog.jpg")
    a = alpha_stats(f["path"]) if f["path"] else {}
    ok = f["status"] == 200 and f["ct"].startswith("image/webp") and a.get("has_alpha") and a["opaque"] > 0.2
    check("rb-webp-lossless", "removebg", "600x0/filters:removebg:format(webp):quality(100)", "dog.jpg",
          ok, f, f"status={f['status']} server_ms={f['server_ms']} alpha={a}")

@cell
def rb_webp_lossy_small():
    # #227 family at a small size: expected to complete, slowly. Timing is the data point.
    f = render("rb-webp-lossy-small", "300x0/filters:removebg:format(webp)", "dog.jpg", timeout=600)
    a = alpha_stats(f["path"]) if f["path"] else {}
    if f["status"] == 200 and a.get("has_alpha"):
        note("rb-webp-lossy-small", "removebg", "300x0/filters:removebg:format(webp)", "dog.jpg", f,
             f"#227 known family: status=200 server_ms={f['server_ms']} client_ms={f['client_ms']} alpha={a}")
    else:
        check("rb-webp-lossy-small", "removebg", "300x0/filters:removebg:format(webp)", "dog.jpg",
              False, f, f"status={f['status']} body={f['body'][:120]} alpha={a}")

@cell
def rb_jpeg():
    # Documented error class (same as radius/mask into JPEG, Waves 1-2): a cutout
    # cannot be a JPEG; the server refuses with a plain-text 400 naming the formats.
    f = render("rb-jpeg", "filters:removebg:format(jpeg)", "dog.jpg")
    ok = f["status"] == 400 and "alpha-capable" in f["body"]
    check("rb-jpeg", "removebg", "filters:removebg:format(jpeg)", "dog.jpg", ok, f,
          f"status={f['status']} body={f['body'][:110]}")

@cell
def rb_avif():
    f = render("rb-avif", "600x0/filters:removebg:format(avif)", "dog.jpg")
    s, j = meta("600x0/filters:removebg:format(avif)", "dog.jpg")
    ok = f["status"] == 200 and f["ct"].startswith("image/avif") and j.get("format") == "avif"
    check("rb-avif", "removebg", "600x0/filters:removebg:format(avif)", "dog.jpg",
          ok, f, f"status={f['status']} ct={f['ct']} meta_format={j.get('format')} server_ms={f['server_ms']}")

@cell
def rb_chain_resize_after():
    f = render("rb-chain-resize-after", "filters:removebg/400x0/filters:format(png)", "dog.jpg")
    a = alpha_stats(f["path"]) if f["path"] else {}
    dims = open_img(f["path"]).size if f["path"] else None
    ok = f["status"] == 200 and dims == (400, 300) and a.get("has_alpha")
    check("rb-chain-resize-after", "removebg", "filters:removebg/400x0/filters:format(png)", "dog.jpg",
          ok, f, f"status={f['status']} dims={dims} alpha={a}")

@cell
def rb_chain_resize_before():
    f = render("rb-chain-resize-before", "400x0/filters:removebg:format(png)", "dog.jpg")
    a = alpha_stats(f["path"]) if f["path"] else {}
    dims = open_img(f["path"]).size if f["path"] else None
    ok = f["status"] == 200 and dims == (400, 300) and a.get("has_alpha")
    check("rb-chain-resize-before", "removebg", "400x0/filters:removebg:format(png)", "dog.jpg",
          ok, f, f"status={f['status']} dims={dims} alpha={a}")

@cell
def rb_chain_grayscale():
    f = render("rb-chain-grayscale", "filters:removebg:grayscale:format(png)", "dog.jpg")
    p = f["path"]; a = alpha_stats(p) if p else {}
    spread = None
    if p:
        im = open_img(p).convert("RGBA")
        px = [q for q in im.resize((64, 48)).getdata() if q[3] > 250]
        spread = max((max(q[:3]) - min(q[:3])) for q in px) if px else None
    ok = f["status"] == 200 and a.get("has_alpha") and spread is not None and spread <= 4
    check("rb-chain-grayscale", "removebg", "filters:removebg:grayscale:format(png)", "dog.jpg",
          ok, f, f"status={f['status']} alpha={a} opaque_channel_spread={spread}")

@cell
def rb_2000_webp_known227():
    # The #227 signature case: transparent render at 2000x1334 into lossy WebP.
    f = render("rb-2000-webp", "2000x0/filters:removebg:format(webp)", "large.jpg", timeout=900)
    note("rb-2000-webp", "removebg", "2000x0/filters:removebg:format(webp)", "large.jpg", f,
         f"#227 known sighting: status={f['status']} client_ms={f['client_ms']} bytes={f['bytes']} body={f['body'][:120]}")

@cell
def rb_health_after_227():
    f = render("rb-health-after-227", "300x0/filters:removebg:format(png)", "dog.jpg")
    check("rb-health-after-227", "removebg", "300x0/filters:removebg:format(png)", "dog.jpg",
          f["status"] == 200, f, f"status={f['status']} server_ms={f['server_ms']} (server survival after the 2000px #227 probe)")

@cell
def rb_then_studio():
    f = render("rb-then-studio", "filters:removebg:studio(product,bg:white):format(png)", "product-shoe.jpg")
    p = f["path"]
    okc, got = corners_bg(p, (255, 255, 255), tol=14) if p else (False, None)
    check("rb-then-studio", "removebg", "filters:removebg:studio(product,bg:white):format(png)", "product-shoe.jpg",
          f["status"] == 200 and okc, f, f"status={f['status']} corners={got} server_ms={f['server_ms']}")

# ---------------------------------------------------------------- studio
def studio_cell(cid, ops, src="product-shoe.jpg", want_bg=None, bg_tol=14):
    f = render(cid, ops, src)
    p = f["path"]
    if want_bg and p:
        okc, got = corners_bg(p, want_bg, tol=bg_tol)
        return f, okc, got
    return f, None, None

@cell
def st_default_brand():
    f, okc, got = studio_cell("st-default-brand", "filters:studio(product):format(png)", want_bg=BRAND)
    check("st-default-brand", "studio", "filters:studio(product):format(png)", "product-shoe.jpg",
          f["status"] == 200 and okc, f, f"status={f['status']} corners={got} want~{BRAND} server_ms={f['server_ms']}")

@cell
def st_bg_white():
    f, okc, got = studio_cell("st-bg-white", "filters:studio(product,bg:white):format(png)", want_bg=(255, 255, 255))
    check("st-bg-white", "studio", "filters:studio(product,bg:white):format(png)", "product-shoe.jpg",
          f["status"] == 200 and okc, f, f"status={f['status']} corners={got}")

@cell
def st_bg_hex():
    f, okc, got = studio_cell("st-bg-hex", "filters:studio(product,bg:f5f0e8):format(png)", want_bg=(245, 240, 232))
    check("st-bg-hex", "studio", "filters:studio(product,bg:f5f0e8):format(png)", "product-shoe.jpg",
          f["status"] == 200 and okc, f, f"status={f['status']} corners={got}")

@cell
def st_bg_brand_arg():
    f, okc, got = studio_cell("st-bg-brand-arg", "filters:studio(product,bg:brand):format(png)", want_bg=BRAND)
    check("st-bg-brand-arg", "studio", "filters:studio(product,bg:brand):format(png)", "product-shoe.jpg",
          f["status"] == 200 and okc, f, f"status={f['status']} corners={got}")

@cell
def st_bg_none_png():
    f = render("st-bg-none-png", "filters:studio(product,bg:none):format(png)", "product-shoe.jpg")
    a = alpha_stats(f["path"]) if f["path"] else {}
    ok = f["status"] == 200 and a.get("has_alpha") and a["transparent"] > 0.2 and a["opaque"] > 0.1
    check("st-bg-none-png", "studio", "filters:studio(product,bg:none):format(png)", "product-shoe.jpg",
          ok, f, f"status={f['status']} alpha={a} server_ms={f['server_ms']}")

@cell
def st_bg_none_webp():
    # Transparent backdrop into lossy WebP: #227 family, timing is the data point.
    f = render("st-bg-none-webp", "filters:studio(product,bg:none):format(webp)", "product-shoe.jpg", timeout=900)
    a = alpha_stats(f["path"]) if f["path"] else {}
    if f["status"] == 200:
        note("st-bg-none-webp", "studio", "filters:studio(product,bg:none):format(webp)", "product-shoe.jpg", f,
             f"#227 known family: status=200 server_ms={f['server_ms']} client_ms={f['client_ms']} alpha={a}")
    else:
        note("st-bg-none-webp", "studio", "filters:studio(product,bg:none):format(webp)", "product-shoe.jpg", f,
             f"#227 known family: status={f['status']} body={f['body'][:120]}")

@cell
def st_opaque_webp_fast():
    # The #226 opaque shortcut: color backdrop into WebP must stay fast.
    f = render("st-opaque-webp-fast", "filters:studio(product,bg:white):format(webp)", "product-shoe.jpg", timeout=120)
    ok = f["status"] == 200 and (f["server_ms"] or 999999) < 10000
    check("st-opaque-webp-fast", "studio", "filters:studio(product,bg:white):format(webp)", "product-shoe.jpg",
          ok, f, f"status={f['status']} server_ms={f['server_ms']} (#226 path: expect ~1 s class, not the #227 alpha path)")

def shadow_strip_mean(path):
    im = open_img(path); w, h = im.size
    return region_mean(path, (int(w * 0.35), int(h * 0.80), int(w * 0.65), int(h * 0.88)))

def shadow_stats(path_a, path_b):
    """(footprint px, mean depth, max depth) of b's darkening vs a, lower half."""
    a = open_img(path_a).convert("RGB"); b = open_img(path_b).convert("RGB")
    w, h = a.size
    a = a.crop((0, h // 2, w, h)).resize((220, 74)); b = b.crop((0, h // 2, w, h)).resize((220, 74))
    pa, pb = list(a.getdata()), list(b.getdata())
    depths = [(x[0] + x[1] + x[2]) - (y[0] + y[1] + y[2]) for x, y in zip(pa, pb)]
    foot = [d for d in depths if d > 36]
    return len(foot), (round(sum(foot) / len(foot), 1) if foot else 0), max(depths)

@cell
def st_shadow_variants():
    fn = render("st-shadow-none", "filters:studio(product,bg:white,shadow:none):format(png)", "product-shoe.jpg")
    fs = render("st-shadow-soft", "filters:studio(product,bg:white,shadow:soft):format(png)", "product-shoe.jpg")
    ft = render("st-shadow-strong", "filters:studio(product,bg:white,shadow:strong):format(png)", "product-shoe.jpg")
    if all(x["path"] for x in (fn, fs, ft)):
        fp_s, md_s, mx_s = shadow_stats(fn["path"], fs["path"])
        fp_t, md_t, mx_t = shadow_stats(fn["path"], ft["path"])
        # By design (studio.rs): soft = wide blur, peak 0.35; strong = tight blur, peak 0.55.
        ok = fp_s > 40 and fp_t > 40 and md_t > md_s and mx_t > mx_s
        check("st-shadow-variants", "studio", "filters:studio(product,bg:white,shadow:*):format(png)", "product-shoe.jpg",
              ok, fs, f"soft: footprint={fp_s} mean_depth={md_s} max={mx_s}; strong: footprint={fp_t} mean_depth={md_t} max={mx_t} (expect both present, strong deeper)")
    else:
        check("st-shadow-variants", "studio", "filters:studio(product,bg:white,shadow:*):format(png)", "product-shoe.jpg",
              False, fs, f"statuses none={fn['status']} soft={fs['status']} strong={ft['status']}")

def subject_fraction(path, bg, tol=16):
    im = open_img(path).convert("RGB")
    im = im.resize((220, 147))
    px = list(im.getdata())
    off = sum(1 for q in px if any(abs(q[i] - bg[i]) > tol for i in range(3)))
    return round(off / len(px), 4)

@cell
def st_pad_variants():
    f0 = render("st-pad0", "filters:studio(product,bg:white,pad:0):format(png)", "product-shoe.jpg")
    f10 = render("st-pad10", "filters:studio(product,bg:white,pad:10):format(png)", "product-shoe.jpg")
    f40 = render("st-pad40", "filters:studio(product,bg:white,pad:40):format(png)", "product-shoe.jpg")
    if all(x["path"] for x in (f0, f10, f40)):
        s0 = subject_fraction(f0["path"], (255, 255, 255))
        s10 = subject_fraction(f10["path"], (255, 255, 255))
        s40 = subject_fraction(f40["path"], (255, 255, 255))
        ok = s0 > s10 > s40
        check("st-pad-variants", "studio", "filters:studio(product,bg:white,pad:*):format(png)", "product-shoe.jpg",
              ok, f10, f"subject fraction pad0={s0} pad10={s10} pad40={s40} (expect decreasing)")
    else:
        check("st-pad-variants", "studio", "filters:studio(product,bg:white,pad:*):format(png)", "product-shoe.jpg",
              False, f10, f"statuses {f0['status']}/{f10['status']}/{f40['status']}")

@cell
def st_chain_size():
    f = render("st-chain-size", "filters:studio(product)/1200x1200/filters:format(png)", "product-shoe.jpg")
    dims = open_img(f["path"]).size if f["path"] else None
    check("st-chain-size", "studio", "filters:studio(product)/1200x1200", "product-shoe.jpg",
          f["status"] == 200 and dims == (1200, 1200), f, f"status={f['status']} dims={dims}")

@cell
def st_jpeg():
    f, okc, got = studio_cell("st-jpeg", "filters:studio(product):format(jpeg)", want_bg=BRAND, bg_tol=20)
    check("st-jpeg", "studio", "filters:studio(product):format(jpeg)", "product-shoe.jpg",
          f["status"] == 200 and f["ct"].startswith("image/jpeg") and okc, f,
          f"status={f['status']} corners={got}")

@cell
def st_nosubject_gradient():
    # Documented behavior (Studio fine print): when the mask finds no usable subject,
    # the fallback serves the original image centered and padded on the backdrop,
    # recorded as a no_subject fallback in meta. On this gradient the mask's
    # threshold-128 subject is 6 px in a 3x2 corner box over a full-frame faint
    # region. RUN LAST: on v0.4.0 this request aborts the server (issue #243).
    f = render("st-nosubject", "filters:studio(product,bg:white):format(png)", "gradient.png")
    ok = f["status"] == 200 and f["path"] is not None
    check("st-nosubject", "studio", "filters:studio(product,bg:white):format(png)", "gradient.png", ok, f,
          f"status={f['status']} (defect #243: on v0.4.0 the process aborts in place() with an 87 GB allocation; status 0 = connection died with the server)")

@cell
def st_logo_alpha_src():
    f = render("st-logo", "filters:studio(product,bg:white):format(png)", "logo.png")
    p = f["path"]
    okc, got = corners_bg(p, (255, 255, 255), tol=14) if p else (False, None)
    check("st-logo", "studio", "filters:studio(product,bg:white):format(png)", "logo.png",
          f["status"] == 200 and okc, f, f"status={f['status']} corners={got} server_ms={f['server_ms']}")

@cell
def st_meta_fields():
    s, j = meta("filters:studio(product,bg:f5f0e8,shadow:strong,pad:25):format(png)", "product-shoe.jpg")
    stu = j.get("studio") or {}
    ok = (s == 200 and stu.get("bg") == "#f5f0e8" and stu.get("bg_from") == "argument"
          and stu.get("shadow") == "strong" and stu.get("pad") == 25 and stu.get("steps"))
    rec("st-meta-fields", "studio", "meta filters:studio(product,bg:f5f0e8,shadow:strong,pad:25)", "product-shoe.jpg",
        "PASS" if ok else "FAIL", {"status": s}, f"meta.studio={json.dumps(stu)[:300]}")

@cell
def st_meta_default_brand():
    s, j = meta("filters:studio(product):format(png)", "product-shoe.jpg")
    stu = j.get("studio") or {}
    ok = s == 200 and stu.get("bg") == "#0e7c7b" and stu.get("bg_from") in ("brand", "default")
    rec("st-meta-default-brand", "studio", "meta filters:studio(product)", "product-shoe.jpg",
        "PASS" if ok else "FAIL", {"status": s}, f"meta.studio={json.dumps(stu)[:260]}")

# ---------------------------------------------------------------- studio scene (#198)
@cell
def sc_meta_fields():
    s, j = meta("filters:studio(scene):format(png)", "product-shoe.jpg")
    stu = j.get("studio") or {}
    scn = stu.get("scene") or {}
    ok = (s == 200 and stu.get("mode") == "scene" and stu.get("pad") == 10
          and stu.get("steps") == ["mask", "crop", "correct"] and stu.get("fallbacks") == []
          and "bg" not in stu
          and isinstance(scn.get("subject"), list) and len(scn["subject"]) == 4
          and scn.get("subjects", 0) >= 1
          and isinstance(scn.get("crop"), list) and len(scn["crop"]) == 4
          and isinstance(scn.get("levels"), list) and len(scn["levels"]) == 3
          and isinstance(scn.get("saturation"), (int, float)) and scn["saturation"] >= 1)
    rec("sc-meta-fields", "studio", "meta filters:studio(scene)", "product-shoe.jpg",
        "PASS" if ok else "FAIL", {"status": s}, f"meta.studio={json.dumps(stu)[:340]}")

@cell
def sc_canvas_kept():
    # The reframe crops and resizes back: the canvas in
    # effect (the 1100x733 source) never changes size.
    f = render("sc-canvas-kept", "filters:studio(scene):format(png)", "product-shoe.jpg")
    dims = open_img(f["path"]).size if f["path"] else None
    check("sc-canvas-kept", "studio", "filters:studio(scene):format(png)", "product-shoe.jpg",
          f["status"] == 200 and dims == (1100, 733), f,
          f"status={f['status']} dims={dims} server_ms={f['server_ms']}")

@cell
def sc_pad_window():
    s0, j0 = meta("filters:studio(scene,pad:0):format(png)", "product-shoe.jpg")
    s4, j4 = meta("filters:studio(scene,pad:40):format(png)", "product-shoe.jpg")
    c0 = (j0.get("studio") or {}).get("scene", {}).get("crop") or [0, 0, 0, 0]
    c4 = (j4.get("studio") or {}).get("scene", {}).get("crop") or [0, 0, 0, 0]
    ok = s0 == 200 and s4 == 200 and c4[2] * c4[3] > c0[2] * c0[3]
    rec("sc-pad-window", "studio", "meta filters:studio(scene,pad:0|40)", "product-shoe.jpg",
        "PASS" if ok else "FAIL", {"status": s0},
        f"crop pad0={c0} pad40={c4} (expect the pad:40 window the larger)")

@cell
def sc_nosubject_gradient():
    # The gradient's mask story is the st-nosubject one:
    # a speck over faint coverage. Scene records the
    # no_subject fallback, a null subject, and the full
    # frame as the crop, and still corrects.
    s, j = meta("filters:studio(scene):format(png)", "gradient.png")
    stu = j.get("studio") or {}
    scn = stu.get("scene") or {}
    ok = (s == 200 and stu.get("mode") == "scene"
          and stu.get("steps") == ["mask", "fallback", "correct"]
          and stu.get("fallbacks") == ["no_subject"]
          and scn.get("subject") is None and scn.get("crop") == [0, 0, 800, 600])
    rec("sc-nosubject-gradient", "studio", "meta filters:studio(scene)", "gradient.png",
        "PASS" if ok else "FAIL", {"status": s}, f"meta.studio={json.dumps(stu)[:300]}")

@cell
def sc_correction_stretches():
    fp = render("sc-plain", "filters:format(png)", "gradient.png")
    fs = render("sc-corrected", "filters:studio(scene):format(png)", "gradient.png")
    def extremes(path):
        im = open_img(path).convert("RGB").resize((200, 150))
        px = list(im.getdata())
        return (min(q[0] for q in px), max(q[0] for q in px),
                min(q[1] for q in px), max(q[1] for q in px))
    if fp["path"] and fs["path"]:
        p = extremes(fp["path"]); c = extremes(fs["path"])
        ok = fs["status"] == 200 and c[0] < p[0] and c[1] > p[1] and c[2] < p[2] and c[3] > p[3]
        check("sc-correction-stretches", "studio", "filters:studio(scene):format(png)", "gradient.png",
              ok, fs, f"plain R {p[0]}..{p[1]} G {p[2]}..{p[3]}; corrected R {c[0]}..{c[1]} G {c[2]}..{c[3]} (expect both ranges wider)")
    else:
        check("sc-correction-stretches", "studio", "filters:studio(scene):format(png)", "gradient.png",
              False, fs, f"statuses plain={fp['status']} scene={fs['status']}")

# ---------------------------------------------------------------- face
AMBER = lambda r, g, b: r > 200 and 110 < g < 215 and b < 90
BLUE = lambda r, g, b: b > 130 and r < 120 and g < 190
MAGENTA = lambda r, g, b: r > 200 and g < 80 and 80 < b < 210  # measured draw color ~ (255, 0, 140)

def drawn_colors(debug_path, plain_path):
    """Count drawing colors among pixels that DIFFER from the plain render.

    The debug overlay is the only difference between the two renders, so the
    changed-pixel set isolates the drawn boxes/cross from photo content
    (a raw color count false-positives on warm photo tones)."""
    d = open_img(debug_path).convert("RGB"); p = open_img(plain_path).convert("RGB")
    assert d.size == p.size, f"debug {d.size} vs plain {p.size}: debug:focal draws on the full frame"
    dd = list(d.getdata()); pp = list(p.getdata())
    counts = {"amber": 0, "blue": 0, "magenta": 0, "changed": 0}
    for a, b in zip(dd, pp):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) > 60:
            counts["changed"] += 1
            if AMBER(*a): counts["amber"] += 1
            elif BLUE(*a): counts["blue"] += 1
            elif MAGENTA(*a): counts["magenta"] += 1
    return counts

@cell
def face_debug_portrait():
    f = render("face-debug-portrait", "debug:focal/crop:700x700:smart/filters:format(png)", "person-portrait.jpg")
    fp = render("face-plain-portrait", "filters:format(png)", "person-portrait.jpg")
    if f["path"] and fp["path"]:
        c = drawn_colors(f["path"], fp["path"])
        ok = f["status"] == 200 and c["amber"] > 20 and c["blue"] > 20 and c["magenta"] >= 1
        check("face-debug-portrait", "face", "debug:focal/crop:700x700:smart", "person-portrait.jpg",
              ok, f, f"status={f['status']} drawn={c} (amber face box, blue window, magenta focal cross expected)")
    else:
        check("face-debug-portrait", "face", "debug:focal/crop:700x700:smart", "person-portrait.jpg",
              False, f, f"status={f['status']} body={f['body'][:120]}")

@cell
def face_debug_group():
    f = render("face-debug-group", "debug:focal/crop:700x700:smart/filters:format(png)", "group-faces.jpg")
    fp = render("face-plain-group", "filters:format(png)", "group-faces.jpg")
    if f["path"] and fp["path"]:
        c = drawn_colors(f["path"], fp["path"])
        ok = f["status"] == 200 and c["amber"] > 20
        check("face-debug-group", "face", "debug:focal/crop:700x700:smart", "group-faces.jpg",
              ok, f, f"status={f['status']} drawn={c} (two faces in frame; amber boxes expected)")
    else:
        check("face-debug-group", "face", "debug:focal/crop:700x700:smart", "group-faces.jpg",
              False, f, f"status={f['status']}")

@cell
def face_debug_fullbody():
    f = render("face-debug-fullbody", "debug:focal/crop:700x700:smart/filters:format(png)", "person-fullbody.jpg")
    fp = render("face-plain-fullbody", "filters:format(png)", "person-fullbody.jpg")
    if f["path"] and fp["path"]:
        c = drawn_colors(f["path"], fp["path"])
        note("face-debug-fullbody", "face", "debug:focal/crop:700x700:smart", "person-fullbody.jpg", f,
             f"status={f['status']} drawn={c} (spike found YuNet misses this tilted/sunglasses face; amber>0 would be a detection, 0 means saliency fallback)")
    else:
        note("face-debug-fullbody", "face", "debug:focal/crop:700x700:smart", "person-fullbody.jpg", f,
             f"status={f['status']}")

@cell
def face_debug_dog_noface():
    f = render("face-debug-dog", "debug:focal/crop:700x700:smart/filters:format(png)", "dog.jpg")
    fp = render("face-plain-dog", "filters:format(png)", "dog.jpg")
    if f["path"] and fp["path"]:
        c = drawn_colors(f["path"], fp["path"])
        check("face-debug-dog", "face", "debug:focal/crop:700x700:smart", "dog.jpg",
              f["status"] == 200 and c["amber"] == 0 and c["magenta"] >= 1, f,
              f"status={f['status']} drawn={c} (expect no amber face box, magenta focal cross present)")
    else:
        check("face-debug-dog", "face", "debug:focal/crop:700x700:smart", "dog.jpg", False, f,
              f"status={f['status']}")

@cell
def face_crop_face():
    fc = render("face-crop-face", "crop:300x300:face/filters:format(png)", "person-portrait.jpg")
    fz = render("face-crop-center", "crop:300x300:center/filters:format(png)", "person-portrait.jpg")
    ok = fc["status"] == 200 and fc["md5"] != fz["md5"]
    check("face-crop-face", "face", "crop:300x300:face", "person-portrait.jpg", ok, fc,
          f"status={fc['status']} differs_from_center={fc['md5'] != fz['md5']} server_ms={fc['server_ms']}")

@cell
def face_crop_face_noface():
    f = render("face-crop-face-noface", "crop:300x300:face/filters:format(png)", "dog.jpg")
    check("face-crop-face-noface", "face", "crop:300x300:face", "dog.jpg",
          f["status"] == 200, f, f"status={f['status']} (docs: falls back to the smart window when no face is found)")

@cell
def face_facepad():
    f0 = render("face-facepad0", "crop:200x200:face/filters:facepad(0):format(png)", "person-portrait.jpg")
    f1 = render("face-facepad1", "crop:200x200:face/filters:facepad(1):format(png)", "person-portrait.jpg")
    ok = f0["status"] == 200 and f1["status"] == 200 and f0["md5"] != f1["md5"]
    check("face-facepad", "face", "crop:200x200:face + facepad(0|1)", "person-portrait.jpg", ok,
          f1, f"pad0_status={f0['status']} pad1_status={f1['status']} outputs_differ={f0['md5'] != f1['md5']}")

FACE_BOX_PORTRAIT = (270, 120, 540, 440)   # approx face rect in the 800x1200 portrait
OUTSIDE_PORTRAIT = (80, 900, 720, 1150)    # torso/background rect, no face

@cell
def face_blur():
    fb = render("face-base-png", "filters:format(png)", "person-portrait.jpg")
    f = render("face-blur", "filters:blurfaces:format(png)", "person-portrait.jpg")
    if fb["path"] and f["path"]:
        vin_b = local_var(fb["path"], FACE_BOX_PORTRAIT); vin_f = local_var(f["path"], FACE_BOX_PORTRAIT)
        dout = mean_diff(fb["path"], f["path"])  # whole-frame includes the face; use outside rect below
        ob = region_mean(fb["path"], OUTSIDE_PORTRAIT); of = region_mean(f["path"], OUTSIDE_PORTRAIT)
        odiff = max(abs(a - b) for a, b in zip(ob, of))
        ok = f["status"] == 200 and vin_f < vin_b * 0.6 and odiff < 6
        check("face-blur", "face", "filters:blurfaces:format(png)", "person-portrait.jpg", ok, f,
              f"face_rect var base={vin_b} blurred={vin_f}; outside mean diff={odiff}")
    else:
        check("face-blur", "face", "filters:blurfaces:format(png)", "person-portrait.jpg", False, f,
              f"status={f['status']}")

@cell
def face_pixelate():
    fb = render("face-base-png2", "filters:format(png)", "person-portrait.jpg")
    f = render("face-pixelate", "filters:pixelatefaces:format(png)", "person-portrait.jpg")
    if fb["path"] and f["path"]:
        cb = distinct_colors(fb["path"], FACE_BOX_PORTRAIT); cf = distinct_colors(f["path"], FACE_BOX_PORTRAIT)
        ok = f["status"] == 200 and cf < cb * 0.5
        check("face-pixelate", "face", "filters:pixelatefaces:format(png)", "person-portrait.jpg", ok, f,
              f"face_rect distinct colors base={cb} pixelated={cf}")
    else:
        check("face-pixelate", "face", "filters:pixelatefaces:format(png)", "person-portrait.jpg", False, f,
              f"status={f['status']}")

@cell
def face_blur_group():
    fb = render("face-group-base", "filters:format(png)", "group-faces.jpg")
    f = render("face-blur-group", "filters:blurfaces:format(png)", "group-faces.jpg")
    L = (230, 170, 490, 350); R = (970, 170, 1230, 350)
    if fb["path"] and f["path"]:
        vl_b, vl_f = local_var(fb["path"], L), local_var(f["path"], L)
        vr_b, vr_f = local_var(fb["path"], R), local_var(f["path"], R)
        ok = f["status"] == 200 and vl_f < vl_b * 0.65 and vr_f < vr_b * 0.65
        check("face-blur-group", "face", "filters:blurfaces:format(png)", "group-faces.jpg", ok, f,
              f"left var {vl_b}->{vl_f}; right var {vr_b}->{vr_f} (both faces should blur)")
    else:
        check("face-blur-group", "face", "filters:blurfaces:format(png)", "group-faces.jpg", False, f,
              f"status={f['status']}")

@cell
def face_blur_noface():
    fb = render("face-noface-base", "filters:format(png)", "dog.jpg")
    f = render("face-blur-noface", "filters:blurfaces:format(png)", "dog.jpg")
    if fb["path"] and f["path"]:
        d = mean_diff(fb["path"], f["path"])
        check("face-blur-noface", "face", "filters:blurfaces:format(png)", "dog.jpg",
              f["status"] == 200 and d < 1.5, f,
              f"status={f['status']} mean_diff_vs_base={d} (no faces: expect a no-op 200, not an error)")
    else:
        check("face-blur-noface", "face", "filters:blurfaces:format(png)", "dog.jpg",
              f["status"] == 200, f, f"status={f['status']} body={f['body'][:120]}")

@cell
def face_chain_shared():
    f = render("face-chain", "crop:300x300:face/filters:blurfaces:format(png)", "person-portrait.jpg")
    check("face-chain", "face", "crop:300x300:face/filters:blurfaces", "person-portrait.jpg",
          f["status"] == 200, f, f"status={f['status']} server_ms={f['server_ms']} (chained face ops share one detection)")

# ---------------------------------------------------------------- meta AI (server A)
def tags_of(j):
    t = j.get("tags")
    if isinstance(t, list) and t and isinstance(t[0], dict):
        return [(x.get("label"), x.get("score")) for x in t]
    if isinstance(t, list):
        return [(x, None) for x in t]
    return []

@cell
def meta_tags_dog():
    s, j = meta("300x200", "dog.jpg")
    tg = tags_of(j)
    labels = [l for l, _ in tg]
    top3 = labels[:3]
    ok = s == 200 and tg and any("dog" in (l or "") for l in top3) and j.get("alt_text")
    rec("meta-tags-dog", "meta-ai", "meta 300x200", "dog.jpg", "PASS" if ok else "FAIL",
        {"status": s}, f"tags={tg[:5]} alt_text={j.get('alt_text')!r}")

@cell
def meta_tags_car():
    s, j = meta("300x200", "car.jpg")
    tg = tags_of(j)
    top3 = [l for l, _ in tg[:3]]
    ok = s == 200 and any("car" in (l or "") for l in top3)
    rec("meta-tags-car", "meta-ai", "meta 300x200", "car.jpg", "PASS" if ok else "FAIL",
        {"status": s}, f"tags={tg[:5]} alt_text={j.get('alt_text')!r}")

@cell
def meta_tags_screenshot():
    s, j = meta("300x200", "screenshot.png")
    tg = tags_of(j)
    top5 = [l for l, _ in tg[:5]]
    ok = s == 200 and any("interface" in (l or "") or "screenshot" in (l or "") for l in top5)
    rec("meta-tags-screenshot", "meta-ai", "meta 300x200", "screenshot.png", "PASS" if ok else "FAIL",
        {"status": s}, f"tags={tg[:6]}")

@cell
def meta_tags_processed():
    s, j = meta("300x200/filters:grayscale", "dog.jpg")
    tg = tags_of(j)
    ok = s == 200 and tg and any("dog" in (l or "") for l, _ in tg[:3])
    rec("meta-tags-processed", "meta-ai", "meta 300x200/filters:grayscale", "dog.jpg",
        "PASS" if ok else "FAIL", {"status": s}, f"tags on processed (grayscale) image={tg[:4]}")

@cell
def meta_embeddings():
    s, j = meta("300x200", "dog.jpg", accept_params="embeddings=1")
    emb = j.get("embedding")
    if emb:
        n, norm = decode_embedding(emb)
        ok = (s == 200 and j.get("embedding_encoding") == "f16le-base64"
              and j.get("embedding_normalized") is True and n == j.get("embedding_dim")
              and abs(norm - 1.0) < 0.05)
        rec("meta-embeddings", "meta-ai", "meta; embeddings=1", "dog.jpg", "PASS" if ok else "FAIL",
            {"status": s}, f"dim={j.get('embedding_dim')} decoded={n} norm={norm} model={j.get('embedding_model')} encoding={j.get('embedding_encoding')} normalized={j.get('embedding_normalized')}")
    else:
        rec("meta-embeddings", "meta-ai", "meta; embeddings=1", "dog.jpg", "FAIL",
            {"status": s}, f"embedding missing; reason={j.get('embedding_reason')} keys={sorted(j.keys())}")

@cell
def meta_aesthetic_score():
    s, j = meta("300x200", "dog.jpg")
    ae = j.get("aesthetic")
    if isinstance(ae, dict):
        sc = ae.get("score")
        ok = (s == 200 and isinstance(sc, (int, float)) and 0.0 <= sc <= 12.0
              and ae.get("head") and ae.get("model"))
        rec("meta-aesthetic-score", "meta-ai", "meta 300x200", "dog.jpg", "PASS" if ok else "FAIL",
            {"status": s}, f"aesthetic={json.dumps(ae)[:240]} (genuine LAION v1 ViT-B/32 head, matching the 512-dim encoder)")
    else:
        rec("meta-aesthetic-score", "meta-ai", "meta 300x200", "dog.jpg", "FAIL",
            {"status": s}, f"aesthetic field={ae!r} keys={sorted(j.keys())}")

@cell
def meta_aesthetic_mismatch():
    # Server C pairs the ViT-B/32 encoder (512) with the genuine LAION ViT-L/14 head (768).
    s, j = meta("300x200", "dog.jpg", base="C")
    ae = j.get("aesthetic")
    if isinstance(ae, dict):
        ok = (ae.get("score") is None and ae.get("reason") == "dimension_mismatch"
              and ae.get("expected") == 768 and ae.get("got") == 512)
        rec("meta-aesthetic-mismatch", "meta-ai", "meta 300x200 (server C)", "dog.jpg", "PASS" if ok else "FAIL",
            {"status": s}, f"aesthetic={json.dumps(ae)[:240]} (documented per-request mismatch shape expected)")
    else:
        rec("meta-aesthetic-mismatch", "meta-ai", "meta 300x200 (server C)", "dog.jpg", "FAIL",
            {"status": s}, f"aesthetic field={ae!r} keys={sorted(j.keys())}")

BENIGN = ["dog.jpg", "person-portrait.jpg", "car.jpg", "product-shoe.jpg", "screenshot.png"]
@cell
def meta_moderation_benign():
    worst = []
    allok = True
    for src in BENIGN:
        s, j = meta("300x200", src, accept_params="moderation=1")
        mod = j.get("moderation") or {}
        scores = {x.get("label"): x.get("score") for x in (mod.get("scores") or [])}
        nsfw = scores.get("nsfw")
        ok = s == 200 and mod.get("model") and nsfw is not None and nsfw < 0.5
        allok = allok and ok
        worst.append(f"{src}: nsfw={nsfw} scores={scores} model={mod.get('model')}")
    rec("meta-moderation-benign", "meta-ai", "meta; moderation=1", ",".join(BENIGN),
        "PASS" if allok else "FAIL", {"status": 200 if allok else 0},
        " | ".join(worst))

@cell
def meta_detections_dog():
    s, j = meta("0x0", "dog.jpg", accept_params="detections=1")
    dets = j.get("detections") or []
    dogs = [d for d in dets if d.get("label") == "dog"]
    ok = s == 200 and dogs and all(d.get("detector") for d in dets) and dogs[0].get("score", 0) >= 0.5
    rec("meta-detections-dog", "meta-ai", "meta; detections=1", "dog.jpg", "PASS" if ok else "FAIL",
        {"status": s}, f"detections={json.dumps(dets)[:300]}")

@cell
def meta_detections_car_person():
    s1, j1 = meta("0x0", "car.jpg", accept_params="detections=1")
    s2, j2 = meta("0x0", "person-portrait.jpg", accept_params="detections=1")
    d1 = j1.get("detections") or []; d2 = j2.get("detections") or []
    ok = (any(d.get("label") == "car" for d in d1) and any(d.get("label") == "person" for d in d2))
    rec("meta-detections-car-person", "meta-ai", "meta; detections=1", "car.jpg,person-portrait.jpg",
        "PASS" if ok else "FAIL", {"status": s1},
        f"car->{[ (d.get('label'), d.get('score'), d.get('detector')) for d in d1][:4]} person->{[(d.get('label'), d.get('score'), d.get('detector')) for d in d2][:4]}")

@cell
def meta_detections_declared():
    s, j = meta("0x0", "dog.jpg", accept_params="detections=1")
    dets = j.get("detections") or []
    providers = sorted({d.get("detector") for d in dets})
    ok = s == 200 and "yolo-echo" in providers and "builtin" in providers
    rec("meta-detections-declared", "meta-ai", "meta; detections=1", "dog.jpg", "PASS" if ok else "FAIL",
        {"status": s}, f"providers={providers} n={len(dets)} (declared detector 'yolo-echo' should aggregate alongside builtin)")

# ---------------------------------------------------------------- not configured (server B)
@cell
def nc_meta_surfaces():
    s1, jm = meta("300x200", "dog.jpg", base="B", accept_params="moderation=1")
    s2, je = meta("300x200", "dog.jpg", base="B", accept_params="embeddings=1")
    s3, jt = meta("300x200", "dog.jpg", base="B")
    s4, jd = meta("300x200", "dog.jpg", base="B", accept_params="detections=1")
    mod = jm.get("moderation") or {}
    ok = (s1 == 200 and mod.get("available") is False and mod.get("reason")
          and s2 == 200 and je.get("embedding") is None and je.get("embedding_reason")
          and s3 == 200 and "tags" not in jt and "aesthetic" not in jt)
    rec("nc-meta-surfaces", "not-configured", "meta opt-ins on bare server", "dog.jpg",
        "PASS" if ok else "FAIL", {"status": s1},
        f"moderation={json.dumps(mod)[:140]} embedding={je.get('embedding')} reason={je.get('embedding_reason')!r} tags_present={'tags' in jt} detections_field={json.dumps(jd.get('detections'))[:120]}")

@cell
def nc_redaction_501s():
    outs = {}
    for cid, ops in [("blurfaces", "filters:blurfaces"), ("pixelatefaces", "filters:pixelatefaces"),
                     ("blurclass", "filters:blurclass(person)"), ("pixelateclass", "filters:pixelateclass(car)")]:
        f = render(f"nc-{cid}", ops, "dog.jpg", base="B")
        outs[cid] = (f["status"], f["body"][:60])
    ok = all(v[0] == 501 for v in outs.values())
    rec("nc-redaction-501s", "not-configured", "redaction filters on bare server", "dog.jpg",
        "PASS" if ok else "FAIL", {"status": 501 if ok else 0}, f"{outs}")

# ---------------------------------------------------------------- moderation signals (server C: CLIP + labels, no model)
@cell
def sig_moderation_signals():
    s, j = meta("300x200", "dog.jpg", base="C", accept_params="moderation=1")
    mod = j.get("moderation") or {}
    sig = mod.get("signals") or {}
    scores = sig.get("scores") or []
    ok = (s == 200 and scores and sig.get("review_only") is True
          and all(0.0 <= (x.get("score") or -1) <= 1.0 for x in scores)
          and not mod.get("scores"))
    rec("sig-moderation-signals", "moderation-signals", "meta; moderation=1 (labels only)", "dog.jpg",
        "PASS" if ok else "FAIL", {"status": s}, f"moderation={json.dumps(mod)[:320]}")

# ---------------------------------------------------------------- detect filters (server A)
@cell
def det_blurclass_person():
    fb = render("det-base-portrait", "filters:format(png)", "person-portrait.jpg")
    f = render("det-blurclass-person", "filters:blurclass(person):format(png)", "person-portrait.jpg")
    box = (300, 550, 520, 950)
    if fb["path"] and f["path"]:
        vb, vf = local_var(fb["path"], box), local_var(f["path"], box)
        ok = f["status"] == 200 and vf < vb * 0.6
        check("det-blurclass-person", "detect", "filters:blurclass(person):format(png)", "person-portrait.jpg",
              ok, f, f"torso var base={vb} blurred={vf}")
    else:
        check("det-blurclass-person", "detect", "filters:blurclass(person):format(png)", "person-portrait.jpg",
              False, f, f"status={f['status']} body={f['body'][:120]}")

@cell
def det_pixelateclass_car():
    fb = render("det-base-car", "filters:format(png)", "car.jpg")
    f = render("det-pixelateclass-car", "filters:pixelateclass(car):format(png)", "car.jpg")
    box = (200, 250, 900, 650)
    if fb["path"] and f["path"]:
        cb, cf = distinct_colors(fb["path"], box), distinct_colors(f["path"], box)
        ok = f["status"] == 200 and cf < cb * 0.6
        check("det-pixelateclass-car", "detect", "filters:pixelateclass(car):format(png)", "car.jpg",
              ok, f, f"car region distinct colors base={cb} pixelated={cf}")
    else:
        check("det-pixelateclass-car", "detect", "filters:pixelateclass(car):format(png)", "car.jpg",
              False, f, f"status={f['status']}")

@cell
def det_blurclass_unknown():
    f = render("det-blurclass-unknown", "filters:blurclass(hovercraft):format(png)", "dog.jpg")
    check("det-blurclass-unknown", "detect", "filters:blurclass(hovercraft)", "dog.jpg",
          f["status"] == 400, f, f"status={f['status']} body={f['body'][:110]} (docs: unknown class is a 400)")

@cell
def det_blurclass_nomatch():
    fb = render("det-nomatch-base", "filters:format(png)", "dog.jpg")
    f = render("det-blurclass-nomatch", "filters:blurclass(person):format(png)", "dog.jpg")
    if fb["path"] and f["path"]:
        d = mean_diff(fb["path"], f["path"])
        check("det-blurclass-nomatch", "detect", "filters:blurclass(person):format(png)", "dog.jpg",
              f["status"] == 200 and d < 1.5, f,
              f"status={f['status']} mean_diff_vs_base={d} (class absent: expect a no-op 200)")
    else:
        check("det-blurclass-nomatch", "detect", "filters:blurclass(person):format(png)", "dog.jpg",
              f["status"] == 200, f, f"status={f['status']} body={f['body'][:120]}")

@cell
def det_blurclass_case():
    f1 = render("det-blurclass-lower", "filters:blurclass(person):format(png)", "person-portrait.jpg")
    f2 = render("det-blurclass-case", "filters:blurclass(Person):format(png)", "person-portrait.jpg")
    same = f1["path"] and f2["path"] and mean_diff(f1["path"], f2["path"]) == 0.0
    check("det-blurclass-case", "detect", "filters:blurclass(Person)", "person-portrait.jpg",
          f2["status"] == 200 and bool(same), f2,
          f"status={f2['status']} pixel_identical_to_lowercase={same} (docs: class matching is case-insensitive)")

# ---------------------------------------------------------------- AI gate
@cell
def gate_ai_concurrent():
    from harness import fetch, AI_BASES, remainder_of
    jobs = []
    def ai_job(ops, src, tag):
        url = f"{AI_BASES['A']}/unsafe/{remainder_of(ops, src)}"
        st, h, body, ms = fetch(url, timeout=300)
        jobs.append((tag, st, round(ms)))
    def still_job(ops, src, tag):
        url = f"{AI_BASES['A']}/unsafe/{remainder_of(ops, src)}"
        st, h, body, ms = fetch(url, timeout=120)
        jobs.append((tag, st, round(ms)))
    threads = [
        threading.Thread(target=ai_job, args=("filters:removebg:format(png)", "dog.jpg", "ai-removebg-dog")),
        threading.Thread(target=ai_job, args=("filters:removebg:format(png)", "cat.jpg", "ai-removebg-cat")),
        threading.Thread(target=ai_job, args=("filters:removebg:format(png)", "car.jpg", "ai-removebg-car")),
        threading.Thread(target=ai_job, args=("filters:studio(product,bg:white):format(png)", "product-shoe.jpg", "ai-studio-shoe")),
        threading.Thread(target=still_job, args=("300x200/filters:format(jpeg)", "dog.jpg", "still-jpeg")),
        threading.Thread(target=still_job, args=("400x0/filters:format(jxl)", "car.jpg", "still-jxl")),
    ]
    for t in threads: t.start()
    for t in threads: t.join(timeout=360)
    stills = [j for j in jobs if j[0].startswith("still")]
    ais = [j for j in jobs if j[0].startswith("ai")]
    ok = (len(jobs) == 6 and all(s == 200 for _, s, _ in stills)
          and all(s in (200, 503) for _, s, _ in ais) and len(ais) == 4)
    rec("gate-ai-concurrent", "gate", "4 AI + 2 still concurrent", "mixed",
        "PASS" if ok else "FAIL", {"status": 200 if ok else 0},
        f"results={sorted(jobs)} (stills must all be 200; AI may be 200 or an explained 503, never 500)")

# ---------------------------------------------------------------- upscale
@cell
def upscale_x2():
    f = render("upscale-x2", "filters:upscale(2):format(png)", "dog-480.jpg", timeout=300)
    dims = open_img(f["path"]).size if f["path"] else None
    ok = f["status"] == 200 and dims == (960, 720)
    check("upscale-x2", "upscale", "filters:upscale(2):format(png)", "dog-480.jpg", ok, f,
          f"status={f['status']} dims={dims} want (960, 720) server_ms={f['server_ms']} (Swin2SR export: correct but slow, ~58 s at this size)")

@cell
def upscale_alpha_carry():
    f = render("upscale-alpha", "filters:upscale(2):format(png)", "logo-256.png", timeout=300)
    dims = open_img(f["path"]).size if f["path"] else None
    a = alpha_stats(f["path"]) if f["path"] else {}
    src_a = alpha_stats(os.path.join(CORPUS, "logo-256.png"))
    ok = (f["status"] == 200 and dims == (512, 320) and a.get("has_alpha")
          and abs(a["transparent"] - src_a["transparent"]) < 0.06)
    check("upscale-alpha", "upscale", "filters:upscale(2):format(png)", "logo-256.png", ok, f,
          f"status={f['status']} dims={dims} alpha={a} source_alpha={src_a} (docs: source transparency is carried across an upscale)")

@cell
def upscale_x4_compose():
    f = render("upscale-x4", "filters:upscale(4):format(png)", "dog-256.jpg", timeout=600)
    dims = open_img(f["path"]).size if f["path"] else None
    ok = f["status"] == 200 and dims == (1024, 768)
    check("upscale-x4", "upscale", "filters:upscale(4):format(png)", "dog-256.jpg", ok, f,
          f"status={f['status']} dims={dims} want (1024, 768): a native 2x model must compose two runs server_ms={f['server_ms']}")

@cell
def upscale_factor3():
    # Case-design correction (disclosed in REPORT.md): the first version of this
    # cell expected upscale(3) to succeed via the README trim sentence. The
    # grammar accepts only factors 2 and 4 (README filter table; url.rs test:
    # "upscale takes a factor of 2 or 4, nothing else", parse of upscale(3) is
    # an error). The trim sentence covers a native factor that does not divide
    # a 2 or 4 request. Correct expectation: an instant, clean 400 parse error.
    f = render("upscale-x3", "filters:upscale(3):format(png)", "dog-256.jpg", timeout=60)
    ok = f["status"] == 400 and "invalid filters segment" in f["body"]
    check("upscale-x3", "upscale", "filters:upscale(3):format(png)", "dog-256.jpg", ok, f,
          f"status={f['status']} body={f['body'][:110]} (grammar accepts factors 2 and 4 only; clean parse 400 is the documented behavior)")

@cell
def upscale_oom_probe():
    # Issue #244: at 900x560 (504,000 px) this export's activation
    # memory OOM-killed the server on v0.4.0. The fix (v0.4.1) caps
    # upscale input at 262,144 px, checked before inference: the
    # request now answers a clean 400 and the server stays alive,
    # which the follow-up render proves.
    f = render("upscale-oom", "filters:upscale(2):format(png)", "logo.png", timeout=60)
    alive = render("upscale-oom-alive", "300x200/filters:format(jpeg)", "dog.jpg", timeout=60)
    check("upscale-oom-probe", "upscale", "filters:upscale(2):format(png)", "logo.png",
          f["status"] == 400 and alive["status"] == 200, f,
          f"status={f['status']} (expect the #244 budget 400, pre-inference) "
          f"followup_render={alive['status']} body={f['body'][:110]}")

# ---------------------------------------------------------------- remove_region
@cell
def rr_erases_region():
    ops = "600x0/filters:remove_region(60,60,180,120):format(png)"
    base = render("rr-base", "600x0/filters:format(png)", "product-shoe.jpg")
    f = render("rr-erases-region", ops, "product-shoe.jpg")
    s, j = meta(ops, "product-shoe.jpg")
    ok = (f["status"] == 200 and base["status"] == 200 and s == 200
          and (j.get("inpaint") or {}).get("regions") == [[60, 60, 180, 120]])
    evidence = (f"status={f['status']} server_ms={f['server_ms']} "
                f"meta_inpaint={j.get('inpaint')}")
    if ok:
        # Outside the rectangle the pixels are bit-identical
        # to the plain render (PNG is lossless); inside, the
        # fill differs measurably from what was there.
        outside_same = (pixel(f["path"], 10, 10) == pixel(base["path"], 10, 10)
                        and pixel(f["path"], 590, 390)
                        == pixel(base["path"], 590, 390))
        inside = region_mean(f["path"], (60, 60, 240, 180))
        inside_base = region_mean(base["path"], (60, 60, 240, 180))
        changed = sum(abs(a - b) for a, b in zip(inside, inside_base)) > 12
        ok = outside_same and changed
        evidence += (f" outside_bit_identical={outside_same} "
                     f"region_mean base={inside_base} removed={inside}")
    check("rr-erases-region", "remove-region", ops, "product-shoe.jpg",
          ok, f, evidence)


@cell
def rr_out_of_bounds():
    f = render("rr-out-of-bounds",
               "600x0/filters:remove_region(500,300,200,200):format(png)",
               "product-shoe.jpg")
    check("rr-out-of-bounds", "remove-region",
          "filters:remove_region(500,300,200,200)", "product-shoe.jpg",
          f["status"] == 400 and "exceeds image bounds" in f["body"], f,
          f"status={f['status']} body={f['body'][:110]}")


@cell
def rr_malformed():
    f = render("rr-malformed", "filters:remove_region(1,2,3):format(png)",
               "product-shoe.jpg")
    check("rr-malformed", "remove-region", "filters:remove_region(1,2,3)",
          "product-shoe.jpg", f["status"] == 400, f,
          f"status={f['status']} body={f['body'][:110]}")


@cell
def rr_unconfigured_501():
    f = render("rr-unconfigured-501",
               "filters:remove_region(10,10,50,50):format(png)", "dog.jpg",
               base="B")
    check("rr-unconfigured-501", "remove-region",
          "filters:remove_region(10,10,50,50)", "dog.jpg",
          f["status"] == 501 and "remove_region is not configured" in f["body"],
          f, f"status={f['status']} body={f['body'][:110]}")


# ---------------------------------------------------------------- segmentation
def _person_segment():
    """The segments record for the portrait's person, from
    a blurclass meta request against the AI instance."""
    s, j = meta("filters:blurclass(person):format(png)", "person-portrait.jpg")
    objects = ((j.get("segments") or {}).get("objects")) or []
    person = next((o for o in objects if o.get("label") == "person"), None)
    return s, j, person


def _corners(box):
    x, y, w, h = box
    return [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]


@cell
def sg_meta_segments():
    s, j, person = _person_segment()
    ok = s == 200 and person is not None
    evidence = f"status={s} segments={json.dumps(j.get('segments'))[:220]}"
    if ok:
        x, y, w, h = person["box"]
        tx, ty, tw, th = person["tight_box"]
        cov = person["coverage"]
        inside = tx >= x and ty >= y and tx + tw <= x + w and ty + th <= y + h
        tighter = tw * th < w * h
        ok = inside and tighter and 0.05 < cov < 0.95
        evidence += (f" box={person['box']} tight={person['tight_box']} "
                     f"coverage={cov} centroid={person['centroid']}")
    rec("sg-meta-segments", "segment", "filters:blurclass(person) meta",
        "person-portrait.jpg", "PASS" if ok else "FAIL", {"status": s},
        evidence)


@cell
def sg_blurclass_mask_tight():
    s, j, person = _person_segment()
    fb = render("sg-base-portrait", "filters:format(png)", "person-portrait.jpg")
    f = render("sg-blurclass-mask", "filters:blurclass(person):format(png)",
               "person-portrait.jpg")
    ok = False
    evidence = f"meta_status={s} render_status={f['status']}"
    if person and fb["path"] and f["path"]:
        # The box's corners are background in a portrait:
        # a mask-tight blur leaves them bit-identical to
        # the plain render (the rectangle form blurs them
        # too). One corner may legitimately carry hair, so
        # three of four must survive exactly.
        same = sum(1 for pt in _corners(person["box"])
                   if pixel(f["path"], *pt) == pixel(fb["path"], *pt))
        # Inside the silhouette the blur bites: the tight
        # box's local variance collapses like the det
        # cell's torso check (local_var crops by corners).
        tx, ty, tw, th = person["tight_box"]
        crop = (tx, ty, tx + tw, ty + th)
        vb = local_var(fb["path"], crop)
        vf = local_var(f["path"], crop)
        ok = same >= 3 and vf < vb * 0.6
        evidence += (f" box={person['box']} tight={person['tight_box']} "
                     f"corners_bit_identical={same}/4 tight_var base={vb} "
                     f"blurred={vf}")
    check("sg-blurclass-mask-tight", "segment",
          "filters:blurclass(person):format(png)", "person-portrait.jpg",
          ok, f, evidence)


@cell
def sg_recolorclass():
    s, j, person = _person_segment()
    fb = render("sg-base-portrait2", "filters:format(png)", "person-portrait.jpg")
    f = render("sg-recolorclass", "filters:recolorclass(person,ff0000):format(png)",
               "person-portrait.jpg")
    ok = False
    evidence = f"meta_status={s} render_status={f['status']}"
    if person and fb["path"] and f["path"]:
        cx, cy = person["centroid"]
        centre = pixel(f["path"], cx, cy)
        painted = centre[0] > 200 and centre[1] < 60 and centre[2] < 60
        same = sum(1 for pt in _corners(person["box"])
                   if pixel(f["path"], *pt) == pixel(fb["path"], *pt))
        ok = painted and same >= 3
        evidence += (f" centroid={person['centroid']} centroid_px={centre} "
                     f"corners_bit_identical={same}/4")
    check("sg-recolorclass", "segment",
          "filters:recolorclass(person,ff0000):format(png)",
          "person-portrait.jpg", ok, f, evidence)


@cell
def sg_recolorclass_unconfigured_501():
    f = render("sg-recolor-unconfigured-501",
               "filters:recolorclass(person,ff0000):format(png)", "dog.jpg",
               base="B")
    check("sg-recolorclass-unconfigured-501", "segment",
          "filters:recolorclass(person,ff0000)", "dog.jpg",
          f["status"] == 501 and "not configured" in f["body"],
          f, f"status={f['status']} body={f['body'][:110]}")


# ---------------------------------------------------------------- studio timing log
@cell
def studio_timing_log():
    path = LOGS["AI"]
    hits = []
    try:
        with open(path, "rb") as fh:
            for line in fh.read().decode("utf-8", "replace").splitlines():
                if "studio" in line.lower() and ("ms" in line or "timing" in line.lower()) and not line.strip().startswith("{"):
                    hits.append(line.strip()[:160])
    except OSError as e:
        hits = [f"log unreadable: {e}"]
    note("studio-timing-log", "studio", "CRABCROP_STUDIO_TIMING=1 server log", "-",
         {"status": 200 if hits else 0},
         f"timing lines found={len(hits)} sample={hits[:2]}")


# ---------------------------------------------------------------- ocr
@cell
def oc_meta_text():
    # Issue #177: ocr=1 meta on text-bearing corpus content
    # (screenshot.png carries the functional UI text). The
    # pipeline must read the strong lines, name its model,
    # echo the configured language, and keep every box in
    # bounds with scores in range.
    s, j = meta("0x0", "screenshot.png", accept_params="ocr=1")
    ocr = j.get("ocr") or {}
    content = ocr.get("text_content") or ""
    lines = ocr.get("text") or []
    want = ["General", "Image pipeline", "sharp edges stress",
            "jumps over the lazy dog", "Save changes"]
    found = [w for w in want if w in content]
    w_img, h_img = j.get("width") or 0, j.get("height") or 0
    shapes = all(
        isinstance(ln.get("text"), str) and bool(ln["text"])
        and 0.0 <= ln.get("score", -1) <= 1.0
        and 0 <= ln.get("x", -1) and 0 <= ln.get("y", -1)
        and ln.get("x", 0) + ln.get("w", 0) <= w_img
        and ln.get("y", 0) + ln.get("h", 0) <= h_img
        for ln in lines)
    joined = "\n".join(ln["text"] for ln in lines) == content
    ok = (s == 200 and bool(ocr.get("model"))
          and ocr.get("language") == "en"
          and len(found) == len(want) and shapes and joined)
    rec("oc-meta-text", "ocr", 'meta "0x0" ocr=1', "screenshot.png",
        "PASS" if ok else "FAIL", {"status": s},
        f"model={ocr.get('model')} language={ocr.get('language')} "
        f"lines={len(lines)} found={found} shapes={shapes} joined={joined} "
        f"content_head={content[:80]!r}")


@cell
def oc_notext():
    # A gradient carries no text: the report is present and
    # empty, not an error and not an unavailable field.
    s, j = meta("0x0", "gradient.png", accept_params="ocr=1")
    ocr = j.get("ocr") or {}
    ok = (s == 200 and ocr.get("text") == []
          and ocr.get("text_content") == "" and bool(ocr.get("model")))
    rec("oc-notext", "ocr", 'meta "0x0" ocr=1', "gradient.png",
        "PASS" if ok else "FAIL", {"status": s}, f"ocr={ocr}")


@cell
def oc_unconfigured():
    # The AIN instance has no OCR set: the field reports
    # itself unavailable and the meta request succeeds.
    s, j = meta("0x0", "screenshot.png", base="B", accept_params="ocr=1")
    ocr = j.get("ocr") or {}
    ok = (s == 200 and ocr.get("available") is False
          and ocr.get("reason") == "not_configured")
    rec("oc-unconfigured", "ocr", 'meta "0x0" ocr=1 (AIN)', "screenshot.png",
        "PASS" if ok else "FAIL", {"status": s}, f"ocr={ocr}")


@cell
def oc_bytes_stable():
    # OCR never touches the image entry (issue #177): two
    # raw renders of one URL around an ocr=1 meta call are
    # byte-identical, and the second is served from the
    # same cache entry. Raw fetches on purpose: render()
    # purges before its timed request by design, so it can
    # never observe the hit this cell asserts.
    from harness import fetch as raw_fetch, BASES as HB, ORIGIN as HO
    url = f"{HB['AI']}/unsafe/filters:format(png)/{HO}/screenshot.png"
    s1, h1, p1, _ = raw_fetch(url)
    meta("0x0", "screenshot.png", accept_params="ocr=1")
    s2, h2, p2, _ = raw_fetch(url)
    ok = (s1 == 200 and s2 == 200 and p1 == p2
          and "hit" in h2.get("cache-status", ""))
    rec("oc-bytes-stable", "ocr", "raw render, ocr meta, raw render",
        "screenshot.png", "PASS" if ok else "FAIL", {"status": s2},
        f"md5_a={hashlib.md5(p1).hexdigest()} "
        f"md5_b={hashlib.md5(p2).hexdigest()} "
        f"cache2={h2.get('cache-status')}")
