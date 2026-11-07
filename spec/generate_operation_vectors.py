#!/usr/bin/env python3
"""Generate spec/operation-vectors.json, the generated-combination
differential corpus for crabcrop's operation grammar (issue #353).

Operation knowledge is implemented independently in the Rust
parser and canonical serializer (src/url.rs), the explainer
(src/explain.rs and the CLI gloss table), the TypeScript SDK's
typed builders and explainer (sdks/typescript), and the
playground's plain-JS explainer copy (playground/lib.mjs). The
hand-written verdict corpus (spec/explain-verdicts.json) samples
that space; this corpus generates combinations across it and
pins, for every generated remainder, the verdict every surface
must agree on:

- `cases`: a remainder, whether the server's parser accepts it,
  and either its canonical form (the re-serialization the cache
  keys on) or the stable error code of its rejection.
- `builder_cases`: a TypeScript builder call, the segment it
  must produce (or that it must throw), and the canonical form
  the server's parser gives that segment.

The tables below encode the grammar as documented in
docs/url-grammar.md and docs/filters.md; the differential tests
in each surface's suite check the implementations against them,
so a table entry that disagrees with an implementation fails
the gate and the disagreement gets adjudicated against the
docs, not silently absorbed.

Deterministic: fixed tables plus a fixed-seed sampler. Run
directly to regenerate the committed file; run with --check to
fail when the committed file differs from a fresh generation
(scripts/test_all.sh does this).

The Python SDK has no operation surface to check: it builds
and signs URLs from verbatim operation strings, exactly like
the crabcrop-url crate, and is pinned by spec/test-vectors.json.
The MCP server delegates building and explaining to the
TypeScript SDK, so it inherits that surface's coverage.
"""

import json
import random
import sys

SRC = "https://example.com/a.jpg"
SRC_S3 = "s3://media/photo.jpg"
SRC_DEEP = "https://cdn.example.com/a/b/c.png"

CASES = []
BUILDER_CASES = []
_seen = {}


def fnum(spelling):
    """The canonical spelling of a decimal the way Rust's float
    Display prints it: whole values as integers, fractions in
    shortest round-trip form, never an exponent."""
    value = float(spelling)
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _emit(remainder, expect):
    if remainder in _seen:
        if _seen[remainder] != expect:
            raise SystemExit(f"conflicting expectations for {remainder}: "
                             f"{_seen[remainder]} vs {expect}")
        return
    _seen[remainder] = expect
    CASES.append({"remainder": remainder, **expect})


def ok(ops, canonical_ops=None, source=SRC):
    """A remainder the parser accepts; `canonical_ops` defaults
    to `ops` (most spellings are already canonical)."""
    if canonical_ops is None:
        canonical_ops = ops
    remainder = f"{ops}/{source}" if ops else source
    canonical = f"{canonical_ops}/{source}" if canonical_ops else source
    _emit(remainder, {"ok": True, "canonical": canonical, "source": source})


def err(ops, code, source=SRC):
    remainder = f"{ops}/{source}" if ops else source
    _emit(remainder, {"ok": False, "error": code, "source": source})


def ok_raw(remainder, canonical, source):
    _emit(remainder, {"ok": True, "canonical": canonical, "source": source})


def seg_cases(table):
    """Singles from a table of (input, canonical-or-None,
    error-or-None) segment spellings."""
    for spelling, canonical, error in table:
        if error:
            err(spelling, error)
        else:
            ok(spelling, canonical)


# ----------------------------------------------------------------
# Operation segments. Canonical notes: leading zeros and `+`
# signs are spelling, not meaning; a size side of `0` (or empty)
# is proportional and spells `0`; the multi-digit zero is the
# one spelling of an exact zero pixel count and spells `00`.

SIZE = [
    ("300x200", None, None),
    ("0x200", None, None),
    ("300x", "300x0", None),
    ("x200", "0x200", None),
    ("x", "0x0", None),
    ("0x0", None, None),
    ("00x100", None, None),
    ("000x100", "00x100", None),
    ("0300x0200", "300x200", None),
    ("+300x200", "300x200", None),
    ("-300x200", None, None),
    ("300x-200", None, None),
    ("-300x-200", None, None),
    ("-0x100", None, None),
    ("0x-100", None, None),
    ("origx100", None, None),
    ("100xorig", None, None),
    ("origxorig", None, None),
    ("-origx100", None, None),
    ("16383x16383", None, None),
    ("1x1", None, None),
    ("16384x10", None, "invalid_size"),
    ("10x16384", None, "invalid_size"),
    ("16384x16384", None, "invalid_size"),
    ("300x200x100", None, "invalid_size"),
    ("abcx10", None, "invalid_size"),
    ("10xabc", None, "invalid_size"),
]

# The objects class cap (issue #408) counts distinct names: 32
# parse, a 33rd distinct name is a bad crop (never a silent
# truncation), and repeats in any case fold before they count.
OBJ32 = ",".join(f"c{i}" for i in range(32))
OBJ33 = ",".join(f"c{i}" for i in range(33))

CROP = [
    ("crop:300x200", None, None),
    ("crop:300x200:center", "crop:300x200", None),
    ("crop:300x200:smart", None, None),
    ("crop:300x200:face", None, None),
    ("crop:300x200:top", None, None),
    ("crop:300x200:bottom", None, None),
    ("crop:300x200:left", None, None),
    ("crop:300x200:right", None, None),
    ("crop:300x200:top-left", None, None),
    ("crop:300x200:top-right", None, None),
    ("crop:300x200:bottom-left", None, None),
    ("crop:300x200:bottom-right", None, None),
    ("crop:0300x0200", "crop:300x200", None),
    ("crop:100x100:objects(person)", "crop:100x100:objects(person):1", None),
    ("crop:100x100:objects(person,car):1.5", None, None),
    ("crop:100x100:objects(person,car)", "crop:100x100:objects(person,car):1", None),
    ("crop:100x100:objects(traffic%20light)", "crop:100x100:objects(traffic%20light):1", None),
    ("crop:100x100:objects(person,PERSON)", "crop:100x100:objects(person):1", None),
    ("crop:100x100:objects(person):2.0", "crop:100x100:objects(person):2", None),
    (f"crop:100x100:objects({OBJ32})", f"crop:100x100:objects({OBJ32}):1", None),
    (f"crop:100x100:objects({OBJ32},C0,c31)", f"crop:100x100:objects({OBJ32}):1", None),
    ("crop:300x0", None, "invalid_crop"),
    ("crop:0x200", None, "invalid_crop"),
    ("crop:-300x200", None, "invalid_crop"),
    ("crop:origx100", None, "invalid_crop"),
    ("crop:300x200:sideways", None, "invalid_crop"),
    ("crop:300x200:left-top", None, "invalid_crop"),
    ("crop:300x200:smart:extra", None, "invalid_crop"),
    ("crop:100x100:objects()", None, "invalid_crop"),
    ("crop:100x100:objects(person,)", None, "invalid_crop"),
    ("crop:100x100:objects(person):0", None, "invalid_crop"),
    ("crop:100x100:objects(person):101", None, "invalid_crop"),
    ("crop:100x100:objects(bad;name)", None, "invalid_crop"),
    (f"crop:100x100:objects({OBJ33})", None, "invalid_crop"),
    ("crop:300", None, "invalid_crop"),
    ("crop:", None, "invalid_crop"),
]

for prefix, code in [("crop-rect", "invalid_crop_rect"), ("focal", "invalid_focal_rect")]:
    seg_cases([
        (f"{prefix}:0x0:50x50", None, None),
        (f"{prefix}:50x50:51x51", None, None),
        (f"{prefix}:007x008:50x50", f"{prefix}:7x8:50x50", None),
        (f"{prefix}:50x50:0x0", None, code),
        (f"{prefix}:0x0:0x50", None, code),
        (f"{prefix}:0x50:50x50", None, code),
        (f"{prefix}:0x0:50", None, code),
        (f"{prefix}:-1x0:50x50", None, code),
        (f"{prefix}:0x0:4294967296x50", None, code),
        (f"{prefix}:0x0:50x50:60x60", None, code),
    ])

FOCALPOINT = [
    ("focalpoint:0.75x0.4", "focalpoint:0.75x0.4:1", None),
    ("focalpoint:0.75x0.4:2", None, None),
    ("focalpoint:0x0:100", None, None),
    ("focalpoint:1x1", "focalpoint:1x1:1", None),
    ("focalpoint:0.75x0.40:2.50", "focalpoint:0.75x0.4:2.5", None),
    ("focalpoint:1.5x0.4", None, "invalid_focal_point"),
    ("focalpoint:0.75x-0.1", None, "invalid_focal_point"),
    ("focalpoint:0.75x0.4:0", None, "invalid_focal_point"),
    ("focalpoint:0.75x0.4:101", None, "invalid_focal_point"),
    ("focalpoint:0.75", None, "invalid_focal_point"),
    ("focalpoint:0.75x0.4:2:3", None, "invalid_focal_point"),
]

FIT_IN = [
    ("fit-in:300x200", None, None),
    ("fit-in:300x", "fit-in:300x0", None),
    ("fit-in:300x200:adaptive", None, None),
    ("fit-in:origx100", None, None),
    ("fit-in:0300x200", "fit-in:300x200", None),
    ("fit-in:-300x200", None, "invalid_size"),
    ("fit-in:300x-200", None, "invalid_size"),
    ("fit-in:300x200:sideways", None, "invalid_size"),
    ("fit-in:300x200:adaptive:extra", None, "invalid_size"),
    ("fit-in:adaptive", None, "invalid_size"),
    ("fit-in:16384x10", None, "invalid_size"),
]

PAD = [
    ("pad:100x100", None, None),
    ("pad:100x100:blur", None, None),
    ("pad:100x100:ff00aa", None, None),
    ("pad:100x100:FF00AA", "pad:100x100:ff00aa", None),
    ("pad:300xorig", None, "invalid_pad"),
    ("pad:-200x100", None, "invalid_pad"),
    ("pad:100x0", None, "invalid_pad"),
    ("pad:100x100:red", None, "invalid_pad"),
    ("pad:100x100:adaptive", None, "invalid_pad"),
    ("pad:100x100:ff00a", None, "invalid_pad"),
    ("pad:100x100:ff00aa:extra", None, "invalid_pad"),
]

FRAME_PAGE = [
    ("frame:poster", None, None),
    ("frame:0", None, None),
    ("frame:90", None, None),
    ("frame:1.50", "frame:1.5", None),
    ("frame:86400", None, None),
    ("frame:50%", None, None),
    ("frame:100%", None, None),
    ("frame:0.5%", None, None),
    ("frame:86401", None, "invalid_frame"),
    ("frame:100.1%", None, "invalid_frame"),
    ("frame:1e2", None, "invalid_frame"),
    ("frame:.5", None, "invalid_frame"),
    ("frame:-1", None, "invalid_frame"),
    ("frame:", None, "invalid_frame"),
    ("frame:abc", None, "invalid_frame"),
    ("page:1", None, None),
    ("page:007", "page:7", None),
    ("page:999999", None, None),
    ("page:0", None, "invalid_page"),
    ("page:1000000", None, "invalid_page"),
    ("page:-1", None, "invalid_page"),
    ("page:1.5", None, "invalid_page"),
    ("page:", None, "invalid_page"),
]

DIRECTIVES = [
    # exp and maxage are metadata: they parse out of the
    # operation list, so the canonical form drops them.
    ("exp:1893456000/300x200", "300x200", None),
    ("maxage:60/300x200", "300x200", None),
    ("maxage:0/300x200", "300x200", None),
    ("maxage:31536000/300x200", "300x200", None),
    ("debug:focal", None, None),
    ("debug:focal/crop:100x100:smart", None, None),
]

for table in (SIZE, CROP, FOCALPOINT, FIT_IN, PAD, FRAME_PAGE):
    seg_cases(table)
for spelling, canonical, _ in DIRECTIVES:
    ok(spelling, canonical)

for ops, code in [
    ("exp:99999999999999999999999999/300x200", "invalid_expiry"),
    ("exp:abc/300x200", "invalid_expiry"),
    ("exp:/300x200", "invalid_expiry"),
    ("exp:-1/300x200", "invalid_expiry"),
    ("maxage:31536001/300x200", "invalid_maxage"),
    ("maxage:soon/300x200", "invalid_maxage"),
    ("maxage:/300x200", "invalid_maxage"),
    ("debug:banana", "invalid_debug"),
    ("debug:", "invalid_debug"),
]:
    err(ops, code)


# ----------------------------------------------------------------
# Filters. A filter case is (spelling, canonical-or-None,
# error-or-None) inside one `filters:` segment. Curve,
# convolution, and noise failures carry their own codes; every
# other filter failure is invalid_filters.

def filter_cases(table):
    for spelling, canonical, error in table:
        if error:
            err(f"filters:{spelling}", error)
        else:
            ok(f"filters:{spelling}",
               f"filters:{canonical}" if canonical else None)


NO_ARG = ["grayscale", "invert", "no_upscale", "equalize", "enhance",
          "removebg", "keep_profile", "sepia", "blurfaces", "pixelatefaces"]
filter_cases([(name, None, None) for name in NO_ARG])
filter_cases([
    ("grayscale()", "grayscale", None),  # empty parens mean no parameters
    ("grayscale(1)", None, "invalid_filters"),
    ("frobnicate", None, "invalid_filters"),
    ("blur", None, "invalid_filters"),  # a filter that takes an argument, bare
])

filter_cases([
    ("blur(2)", None, None),
    ("blur(0.5)", None, None),
    ("blur(.5)", "blur(0.5)", None),
    ("blur(1e2)", "blur(100)", None),
    ("blur(+2)", "blur(2)", None),
    ("blur(512)", None, None),
    ("blur(0)", None, "invalid_filters"),
    ("blur(-1)", None, "invalid_filters"),
    ("blur(512.5)", None, "invalid_filters"),
    ("blur(abc)", None, "invalid_filters"),
    ("blur()", None, "invalid_filters"),
    ("blur( 2)", None, "invalid_filters"),
    ("sharpen(1.5)", None, None),
    ("sharpen(1e1)", "sharpen(10)", None),
    ("sharpen(0)", None, "invalid_filters"),
    ("sharpen(-2)", None, "invalid_filters"),
    ("brighten(5)", None, None),
    ("brighten(+5)", "brighten(5)", None),
    ("brighten(-255)", None, None),
    ("brighten(255)", None, None),
    ("brighten(256)", None, "invalid_filters"),
    ("brighten(1.5)", None, "invalid_filters"),
    ("contrast(-100)", None, None),
    ("contrast(100)", None, None),
    ("contrast(1e2)", "contrast(100)", None),
    ("contrast(0.5)", None, None),
    ("contrast(100.1)", None, "invalid_filters"),
    ("contrast(nan)", None, "invalid_filters"),
    ("gamma(2.2)", None, None),
    ("gamma(1e1)", "gamma(10)", None),
    ("gamma(10)", None, None),
    ("gamma(0)", None, "invalid_filters"),
    ("gamma(11)", None, "invalid_filters"),
    ("gamma(-1)", None, "invalid_filters"),
    ("exposure(1.5)", None, None),
    ("exposure(-1e1)", "exposure(-10)", None),
    ("exposure(0)", None, None),
    ("exposure(10)", None, None),
    ("exposure(10.5)", None, "invalid_filters"),
    ("exposure(nan)", None, "invalid_filters"),
    ("highlights_shadows(30,-20)", None, None),
    ("highlights_shadows(0,0)", None, None),
    ("highlights_shadows(+30,20)", "highlights_shadows(30,20)", None),
    ("highlights_shadows(101,0)", None, "invalid_filters"),
    ("highlights_shadows(30)", None, "invalid_filters"),
    ("highlights_shadows(1,2,3)", None, "invalid_filters"),
    ("vibrance(40)", None, None),
    ("vibrance(-100)", None, None),
    ("vibrance(101)", None, "invalid_filters"),
    ("vibrance(nan)", None, "invalid_filters"),
    ("huerotate(90)", None, None),
    ("huerotate(-90)", None, None),
    ("huerotate(+90)", "huerotate(90)", None),
    ("huerotate(2147483647)", None, None),
    ("huerotate(-2147483648)", None, None),
    ("huerotate(2147483648)", None, "invalid_filters"),
    ("huerotate(1.5)", None, "invalid_filters"),
    ("saturation(0)", None, None),
    ("saturation(1)", None, None),
    ("saturation(255)", None, None),
    ("saturation(0.5)", None, None),
    ("saturation(256)", None, "invalid_filters"),
    ("saturation(-1)", None, "invalid_filters"),
    ("rgb(255,-255,0)", None, None),
    ("rgb(+1,2,3)", "rgb(1,2,3)", None),
    ("rgb(256,0,0)", None, "invalid_filters"),
    ("rgb(1,2)", None, "invalid_filters"),
    ("rgb(1,2,3,4)", None, "invalid_filters"),
])

filter_cases([
    ("quality(80)", None, None),
    ("quality(auto)", None, None),
    ("quality(+80)", "quality(80)", None),
    ("quality(1)", None, None),
    ("quality(100)", None, None),
    ("quality(80,jpeg:90,webp:75)", None, None),
    ("quality(webp:75,80,jpeg:90)", "quality(80,jpeg:90,webp:75)", None),
    ("quality(jpg:90)", "quality(jpeg:90)", None),
    ("quality(jpeg:auto)", None, None),
    ("quality(0)", None, "invalid_filters"),
    ("quality(101)", None, "invalid_filters"),
    ("quality(mp4:80)", None, "invalid_filters"),
    ("quality(jpeg:90,jpg:80)", None, "invalid_filters"),
    ("quality(80,90)", None, "invalid_filters"),
    ("quality(jpeg:0)", None, "invalid_filters"),
    ("quality(banana:80)", None, "invalid_filters"),
    ("quality()", None, "invalid_filters"),
    ("palette(2)", None, None),
    ("palette(64)", None, None),
    ("palette(256)", None, None),
    ("palette(1)", None, "invalid_filters"),
    ("palette(257)", None, "invalid_filters"),
    ("palette(0)", None, "invalid_filters"),
])

filter_cases([
    ("format(jpeg)", None, None),
    ("format(jpg)", "format(jpeg)", None),
    ("format(png)", None, None),
    ("format(webp)", None, None),
    ("format(jxl)", None, None),
    ("format(avif)", None, None),
    ("format(gif)", None, None),
    ("format(mp4)", None, None),
    ("format(webm)", None, None),
    ("format(jpeg,progressive)", None, None),
    ("format(jpeg,subsampling:420)", None, None),
    ("format(jpg,subsampling:444)", "format(jpeg,subsampling:444)", None),
    ("format(jpeg,subsampling:420,progressive)", "format(jpeg,progressive,subsampling:420)", None),
    ("format(png,interlace)", None, None),
    ("format(webp,method:6)", None, None),
    ("format(webp,sns:50,method:0)", "format(webp,method:0,sns:50)", None),
    ("format(webp,method:0,sns:0)", None, None),
    ("format(mov)", None, "invalid_filters"),
    ("format(avif,subsampling:420)", None, "invalid_filters"),
    ("format(jpeg,interlace)", None, "invalid_filters"),
    ("format(png,progressive)", None, "invalid_filters"),
    ("format(webp,method:7)", None, "invalid_filters"),
    ("format(webp,sns:101)", None, "invalid_filters"),
    ("format(jpeg,progressive,progressive)", None, "invalid_filters"),
    ("format(webp,method:6,method:4)", None, "invalid_filters"),
    ("format(jpeg,subsampling:421)", None, "invalid_filters"),
])

_pts16 = ",".join(f"{i * 17},{i * 17}" for i in range(15)) + ",255,255"
_pts17 = ",".join(f"{i * 15},{i * 15}" for i in range(17))
filter_cases([
    ("curve(rgb,0,0,255,255)", None, None),
    ("curve(r,0,64,255,255)", None, None),
    ("curve(b,32,10,224,245)", None, None),
    (f"curve(rgb,{_pts16})", None, None),
    ("curve(x,0,0,255,255)", None, "invalid_curve"),
    ("curve(rgb,0,0)", None, "invalid_curve"),
    ("curve(rgb,0,0,255)", None, "invalid_curve"),
    ("curve(rgb,0,0,256,255)", None, "invalid_curve"),
    ("curve(rgb,128,0,64,255)", None, "invalid_curve"),
    ("curve(rgb,0,0,0,255)", None, "invalid_curve"),
    (f"curve(rgb,{_pts17})", None, "invalid_curve"),
    ("convolution(0,-1,0,-1,5,-1,0,-1,0,0)", None, None),
    ("convolution(0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625,1)", None, None),
    ("convolution(0.50,0,0,0,0,0,0,0,0,0)", "convolution(0.5,0,0,0,0,0,0,0,0,0)", None),
    ("convolution(-1024,0,0,0,0,0,0,0,1024,1)", None, None),
    ("convolution(" + ",".join(["0"] * 12 + ["1"] + ["0"] * 13) + ")", None, None),
    ("convolution(0,0,0,0,1,0,0,0,0)", None, "invalid_convolution"),
    ("convolution(0,0,0,0,1,0,0,0,0,0,0)", None, "invalid_convolution"),
    ("convolution(0,-1,0,-1,5,-1,0,-1,0,2)", None, "invalid_convolution"),
    ("convolution(0,-1,0,-1,1e2,-1,0,-1,0,0)", None, "invalid_convolution"),
    ("convolution(0,-1,0,-1,.5,-1,0,-1,0,0)", None, "invalid_convolution"),
    ("convolution(0,-1,0,-1,1025,-1,0,-1,0,0)", None, "invalid_convolution"),
    ("noise(0)", None, None),
    ("noise(40)", None, None),
    ("noise(100)", None, None),
    ("noise(101)", None, "invalid_noise"),
    ("noise(256)", None, "invalid_noise"),
    ("noise(-1)", None, "invalid_noise"),
    ("noise(1.5)", None, "invalid_noise"),
])

filter_cases([
    ("studio(product)", None, None),
    ("studio(product,bg:white)", None, None),
    ("studio(product,bg:brand)", None, None),
    ("studio(product,bg:none)", None, None),
    ("studio(product,bg:F5F0E8)", "studio(product,bg:f5f0e8)", None),
    ("studio(product,pad:10)", "studio(product)", None),
    ("studio(product,bg:white,shadow:soft,pad:10)", "studio(product,bg:white)", None),
    ("studio(product,pad:25,shadow:strong,bg:f5f0e8)", "studio(product,bg:f5f0e8,shadow:strong,pad:25)", None),
    ("studio(product,pad:+10)", "studio(product)", None),
    ("studio(product,pad:010)", "studio(product)", None),
    ("studio(product,shadow:none)", None, None),
    ("studio(scene)", None, None),
    ("studio(scene,pad:25)", None, None),
    ("studio(scene,pad:10)", "studio(scene)", None),
    ("studio(portrait)", None, "invalid_filters"),
    ("studio(product,bg:blur)", None, "invalid_filters"),
    ("studio(product,bg:fffff)", None, "invalid_filters"),
    ("studio(product,bg:White)", None, "invalid_filters"),
    ("studio(product,shadow:hard)", None, "invalid_filters"),
    ("studio(product,pad:41)", None, "invalid_filters"),
    ("studio(product,pad:-1)", None, "invalid_filters"),
    ("studio(product,pad:1.5)", None, "invalid_filters"),
    ("studio(scene,bg:white)", None, "invalid_filters"),
    ("studio(scene,shadow:soft)", None, "invalid_filters"),
    ("studio(product,pad:5,pad:6)", None, "invalid_filters"),
    ("studio(product,)", None, "invalid_filters"),
    ("studio()", None, "invalid_filters"),
    ("studio", None, "invalid_filters"),
])

filter_cases([
    ("blurclass(person)", None, None),
    ("pixelateclass(traffic%20light)", None, None),
    # The canonical form re-encodes the decoded name, so a
    # raw non-ASCII spelling canonicalizes to its
    # percent-encoded form (the verdict corpus's boundary
    # case: 32 two-byte characters are exactly the 64-byte
    # class-name bound).
    ("blurclass(" + "é" * 32 + ")", "blurclass(" + "%C3%A9" * 32 + ")", None),
    ("recolorclass(person,FF0000)", "recolorclass(person,ff0000)", None),
    ("blurclass()", None, "invalid_filters"),
    ("blurclass(per%24on)", None, "invalid_filters"),
    ("blurclass(" + "é" * 40 + ")", None, "invalid_filters"),
    ("recolorclass(person,ff000)", None, "invalid_filters"),
    ("recolorclass(,ff0000)", None, "invalid_filters"),
    ("facepad(0)", None, None),
    ("facepad(1.5)", None, None),
    ("facepad(3)", None, None),
    ("facepad(3.1)", None, "invalid_filters"),
    ("facepad(-1)", None, "invalid_filters"),
    ("upscale(2)", None, None),
    ("upscale(4)", None, None),
    ("upscale(3)", None, "invalid_filters"),
    ("upscale(0)", None, "invalid_filters"),
    ("upscale(2.0)", None, "invalid_filters"),
    ("dpr(1)", None, None),
    ("dpr(1.5)", None, None),
    ("dpr(2)", None, None),
    ("dpr(3)", None, None),
    ("dpr(1.0)", "dpr(1)", None),
    ("dpr(2.0)", "dpr(2)", None),
    ("dpr(7)", None, "invalid_filters"),
    ("dpr(2.5)", None, "invalid_filters"),
    ("dpr(0)", None, "invalid_filters"),
])

filter_cases([
    ("text(Hello,12,ff8800,center)", "text(Hello,12,ff8800ff,center)", None),
    ("text(Hello,+12,ff8800,center)", "text(Hello,12,ff8800ff,center)", None),
    ("text(Hi%20there,12,ff8800ff,top-left)", None, None),
    ("text(Hello,12,FF8800,center)", "text(Hello,12,ff8800ff,center)", None),
    ("text(Hello,0,ff8800,center)", None, "invalid_filters"),
    ("text(Hello,1001,ff8800,center)", None, "invalid_filters"),
    ("text(Hello,12,ff8800,middle)", None, "invalid_filters"),
    ("text(,12,ff8800,center)", None, "invalid_filters"),
    ("text(Hello,12,ff880,center)", None, "invalid_filters"),
    ("trim", "trim(10)", None),
    ("trim()", "trim(10)", None),
    ("trim(20)", None, None),
    ("trim(0)", None, None),
    ("trim(255)", None, None),
    ("trim(20,5,0,5,0)", None, None),
    ("trim(256)", None, "invalid_filters"),
    ("trim(20,5)", None, "invalid_filters"),
    ("trim(20,5,0,5,-1)", None, "invalid_filters"),
    ("trim(-1)", None, "invalid_filters"),
    ("watermark(https%3A%2F%2Fx%2Fwm.png,0,0,80)", None, None),
    ("watermark(a.png,-5,10,100)", None, None),
    ("watermark(a.png,0,0,101)", None, "invalid_filters"),
    ("watermark(,0,0,80)", None, "invalid_filters"),
    ("watermark(a.png,0,0)", None, "invalid_filters"),
    ("rotate(45)", None, None),
    ("rotate(-45)", None, None),
    ("rotate(45.5)", None, None),
    ("rotate(90.0)", "rotate(90)", None),
    ("rotate(36000)", None, None),
    ("rotate(36001)", None, "invalid_filters"),
    ("rotate(1e2)", None, "invalid_filters"),
    ("rotate(.5)", None, "invalid_filters"),
    ("rotate(abc)", None, "invalid_filters"),
    ("overlay(https%3A%2F%2Fx%2Fov.png,0,0,80,multiply)", None, None),
    ("overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,screen,25)", None, None),
    ("overlay(a%2Cb.png,0,0,80,normal,25)", None, None),
    ("overlay(a.png,0,0,101,normal)", None, "invalid_filters"),
    ("overlay(a.png,0,0,80,darken)", None, "invalid_filters"),
    ("overlay(a.png,0,0,80,normal,0)", None, "invalid_filters"),
    ("overlay(a.png,0,0,80,normal,401)", None, "invalid_filters"),
])

filter_cases([
    ("radius(12)", "radius(12,12,12,12)", None),
    ("radius(0)", "radius(0,0,0,0)", None),
    ("radius(1,2,3,4)", None, None),
    ("radius(1,2)", None, "invalid_filters"),
    ("radius(16384)", None, "invalid_filters"),
    ("border(2,00ff00)", None, None),
    ("border(0,00ff00)", None, None),
    ("border(2,FF00AA)", "border(2,ff00aa)", None),
    ("border(16384,00ff00)", None, "invalid_filters"),
    ("border(2,00ff0)", None, "invalid_filters"),
    ("border(-1,00ff00)", None, "invalid_filters"),
    ("mask(circle)", None, None),
    ("mask(ellipse)", None, None),
    ("mask(square)", None, "invalid_filters"),
    ("mask()", None, "invalid_filters"),
    ("duotone(000000,ffffff)", None, None),
    ("duotone(FF0000,00FF00)", "duotone(ff0000,00ff00)", None),
    ("duotone(000000)", None, "invalid_filters"),
    ("monochrome(ff8000)", None, None),
    ("monochrome(ff800)", None, "invalid_filters"),
    ("pixelate(8)", None, None),
    ("pixelate(1)", None, None),
    ("pixelate(16383)", None, None),
    ("pixelate(0)", None, "invalid_filters"),
    ("pixelate(16384)", None, "invalid_filters"),
    ("flip(h)", None, None),
    ("flip(v)", None, None),
    ("flip(x)", None, "invalid_filters"),
    ("flip(H)", None, "invalid_filters"),
    ("fill(ff00aa)", None, None),
    ("fill(FF00AA)", "fill(ff00aa)", None),
    ("fill(red)", None, "invalid_filters"),
    ("fill(ff00a)", None, "invalid_filters"),
    ("blur_region(1,2,30,40)", None, None),
    ("pixelate_region(0,0,5,5)", None, None),
    ("remove_region(0,0,16383,16383)", None, None),
    ("blur_region(1,2,0,40)", None, "invalid_filters"),
    ("pixelate_region(0,0,10,16384)", None, "invalid_filters"),
    ("blur_region(1,2,3)", None, "invalid_filters"),
    ("max_bytes(1000)", None, None),
    ("max_bytes(18446744073709551615)", None, None),
    ("max_bytes(0)", None, "invalid_filters"),
    ("max_bytes(-1)", None, "invalid_filters"),
    ("max_bytes(18446744073709551616)", None, "invalid_filters"),
])


# ----------------------------------------------------------------
# Combinations. Filter chains inside one segment preserve order
# (order is meaning, not spelling); adjacent `filters:`
# segments merge in the canonical form. Segment chains sample
# pairs and triples from a representative pool with a fixed
# seed. An invalid segment inside a valid chain rejects the
# whole remainder with that segment's code: the parser reads
# segments in path order and reports the first failure.

FILTER_POOL = [
    "grayscale", "blur(2)", "quality(80)", "format(webp)", "sepia",
    "contrast(20)", "noise(10)", "flip(h)", "rotate(90)", "sharpen(1.5)",
]
for a in FILTER_POOL:
    for b in FILTER_POOL:
        if a != b:
            ok(f"filters:{a}:{b}")
for chain, canonical in [
    ("grayscale:blur(2):quality(80)", None),
    ("format(png):radius(4):border(1,000000)",
     "format(png):radius(4,4,4,4):border(1,000000)"),
    ("sepia:duotone(000000,ffffff):flip(h)", None),
    ("curve(rgb,0,0,255,255):convolution(0,-1,0,-1,5,-1,0,-1,0,0):noise(5)", None),
]:
    ok(f"filters:{chain}", f"filters:{canonical}" if canonical else None)
# Split segments merge into one canonical segment.
ok("filters:blur(2)/filters:grayscale", "filters:blur(2):grayscale")
ok("filters:grayscale/filters:blur(2)/filters:invert",
   "filters:grayscale:blur(2):invert")
ok("300x200/filters:blur(2)/filters:grayscale",
   "300x200/filters:blur(2):grayscale")
# A bad filter anywhere in the chain fails the segment with
# that filter family's code.
err("filters:grayscale:blur(0):quality(80)", "invalid_filters")
err("filters:quality(80):curve(x,0,0,255,255)", "invalid_curve")
err("filters:noise(101):grayscale", "invalid_noise")
err("filters:", "invalid_filters")
err("filters:blur(2):", "invalid_filters")
err("filters::blur(2)", "invalid_filters")

SEGMENT_POOL = [
    "300x200", "crop:100x100:smart", "fit-in:200x200", "pad:100x100:blur",
    "crop-rect:0x0:50x50", "focalpoint:0.5x0.5:2", "frame:poster",
    "page:2", "debug:focal", "filters:grayscale", "filters:blur(2):quality(80)",
    "origx100", "-300x200", "crop:100x100:objects(person):1",
]
def merged(ops):
    """The canonical form of a composed chain, at the
    spelling level: adjacent `filters:` segments merge into
    one. (Per-operation respellings are covered by the
    singles tables; the pool's spellings are canonical.)"""
    out = []
    for seg in ops.split("/"):
        if out and seg.startswith("filters:") and out[-1].startswith("filters:"):
            out[-1] += ":" + seg[len("filters:"):]
        else:
            out.append(seg)
    return "/".join(out)


rng = random.Random(353353)
pairs = [(a, b) for a in SEGMENT_POOL for b in SEGMENT_POOL if a != b]
rng.shuffle(pairs)
for a, b in pairs[:48]:
    ops = f"{a}/{b}"
    ok(ops, merged(ops))
triples = set()
while len(triples) < 40:
    triples.add(tuple(rng.sample(SEGMENT_POOL, 3)))
for a, b, c in sorted(triples):
    ops = f"{a}/{b}/{c}"
    ok(ops, merged(ops))

INVALID_SEGMENTS = [
    ("crop:300x0", "invalid_crop"),
    ("filters:blur(0)", "invalid_filters"),
    ("16384x10", "invalid_size"),
    ("page:0", "invalid_page"),
    ("pad:100x100:red", "invalid_pad"),
    ("debug:banana", "invalid_debug"),
]
base_chains = [
    ["300x200", "filters:grayscale"],
    ["crop:100x100", "filters:quality(80)", "pad:50x50"],
    ["filters:sepia"],
]
for chain in base_chains:
    for bad, code in INVALID_SEGMENTS:
        for at in range(len(chain) + 1):
            err("/".join(chain[:at] + [bad] + chain[at:]), code)

# The first segment that is not an operation starts the
# source, and the source is taken verbatim from there.
ok_raw(f"banana/{SRC}", f"banana/{SRC}", f"banana/{SRC}")
ok_raw(f"300x200/banana/{SRC}", f"300x200/banana/{SRC}", f"banana/{SRC}")
ok_raw(f"300x200.5/{SRC}", f"300x200.5/{SRC}", f"300x200.5/{SRC}")
ok_raw(f"100x200.jpg/{SRC}", f"100x200.jpg/{SRC}", f"100x200.jpg/{SRC}")
# Other source kinds ride along unchanged.
ok("300x200/filters:grayscale", source=SRC_S3)
ok("crop:1200x630:smart/filters:quality(80)", source=SRC_DEEP)
ok("300x200", source="root:photos/cat.jpg")
ok("", source=SRC)

# ----------------------------------------------------------------
# Builder cases: one TypeScript builder call, the segment it
# must produce (or that it must throw), and the canonical form
# the server's parser gives that segment. `call` is a dotted
# path into the SDK's `ops` / `filter` objects; an arg may
# itself be a {"call": ..., "args": [...]} descriptor.

def builder(call, args, segment, canonical=None, throws=False, code=None,
            rust=True):
    """One builder case. `rust=False` marks a case with no
    parser counterpart (a preset reference is not a parseable
    segment; a non-integer size has no URL spelling): the Rust
    differential skips those and only the SDK suite checks
    them."""
    entry = {"call": call, "args": args}
    if throws:
        entry["throws"] = True
        if segment is not None:
            entry["segment"] = segment
            entry["error"] = code
    else:
        entry["segment"] = segment
        entry["canonical"] = canonical if canonical is not None else segment
    if not rust:
        entry["rust"] = False
    BUILDER_CASES.append(entry)


builder("ops.size", [300, 200], "300x200")
builder("ops.size", [-300, 0], "-300x0")
builder("ops.size", ["orig", 100], "origx100")
builder("ops.size", [0, 200], "0x200")
builder("ops.size", [16384, 10], "16384x10", throws=True, code="invalid_size")
builder("ops.size", [300.5, 200], None, throws=True, rust=False)
builder("ops.fitIn", [300, 200], "fit-in:300x200")
builder("ops.fitIn", [300, 200, True], "fit-in:300x200:adaptive")
builder("ops.fitIn", [-300, 200], "fit-in:-300x200", throws=True, code="invalid_size")
builder("ops.crop", [300, 200], "crop:300x200")
builder("ops.crop", [300, 200, "smart"], "crop:300x200:smart")
builder("ops.crop", [0, 200], "crop:0x200", throws=True, code="invalid_crop")
builder("ops.crop", [300, 200, "sideways"], "crop:300x200:sideways",
        throws=True, code="invalid_crop")
builder("ops.cropObjects", [100, 100, ["person", "car"], 1.5],
        "crop:100x100:objects(person,car):1.5")
builder("ops.cropObjects", [100, 100, ["person", "PERSON"]],
        "crop:100x100:objects(person):1")
builder("ops.cropObjects", [100, 100, OBJ32.split(",") + ["C0"]],
        f"crop:100x100:objects({OBJ32}):1")
builder("ops.cropObjects", [100, 100, OBJ33.split(",")],
        f"crop:100x100:objects({OBJ33}):1", throws=True, code="invalid_crop")
builder("ops.cropRect", [0, 0, 50, 50], "crop-rect:0x0:50x50")
builder("ops.cropRect", [50, 50, 0, 0], "crop-rect:50x50:0x0",
        throws=True, code="invalid_crop_rect")
builder("ops.focal", [0, 0, 50, 50], "focal:0x0:50x50")
builder("ops.focalPoint", [0.75, 0.4], "focalpoint:0.75x0.4:1")
builder("ops.focalPoint", [0.75, 0.4, 2], "focalpoint:0.75x0.4:2")
builder("ops.focalPoint", [1.5, 0.4], "focalpoint:1.5x0.4:1",
        throws=True, code="invalid_focal_point")
builder("ops.pad", [100, 100], "pad:100x100")
builder("ops.pad", [100, 100, "blur"], "pad:100x100:blur")
builder("ops.pad", [100, 100, "FF00AA"], "pad:100x100:ff00aa")
builder("ops.pad", [0, 100], "pad:0x100", throws=True, code="invalid_pad")
builder("ops.preset", ["thumb"], "preset:thumb", rust=False)
builder("ops.maxage", [60], "maxage:60", canonical="")
builder("ops.maxage", [31536001], "maxage:31536001",
        throws=True, code="invalid_maxage")
builder("ops.debugFocal", [], "debug:focal")
builder("ops.filters",
        [{"call": "filter.grayscale", "args": []},
         {"call": "filter.blur", "args": [2]}],
        "filters:grayscale:blur(2)")

builder("filter.grayscale", [], "grayscale")
builder("filter.blur", [2], "blur(2)")
builder("filter.blur", [0.5], "blur(0.5)")
builder("filter.blur", [0], "blur(0)", throws=True, code="invalid_filters")
builder("filter.blur", [-1], "blur(-1)", throws=True, code="invalid_filters")
builder("filter.blur", [600], "blur(600)", throws=True, code="invalid_filters")
builder("filter.sharpen", [1.5], "sharpen(1.5)")
builder("filter.brighten", [-255], "brighten(-255)")
builder("filter.brighten", [256], "brighten(256)", throws=True, code="invalid_filters")
builder("filter.contrast", [100], "contrast(100)")
builder("filter.contrast", [101], "contrast(101)", throws=True, code="invalid_filters")
builder("filter.gamma", [2.2], "gamma(2.2)")
builder("filter.gamma", [0], "gamma(0)", throws=True, code="invalid_filters")
builder("filter.exposure", [-10], "exposure(-10)")
builder("filter.highlightsShadows", [30, -20], "highlights_shadows(30,-20)")
builder("filter.vibrance", [40], "vibrance(40)")
builder("filter.hueRotate", [-90], "huerotate(-90)")
builder("filter.saturation", [0], "saturation(0)")
builder("filter.rgb", [1, 2, 3], "rgb(1,2,3)")
builder("filter.curve", ["rgb", [[0, 0], [255, 255]]], "curve(rgb,0,0,255,255)")
builder("filter.curve", ["rgb", [[0, 0]]], "curve(rgb,0,0)",
        throws=True, code="invalid_curve")
builder("filter.convolution", [[0, -1, 0, -1, 5, -1, 0, -1, 0], 0],
        "convolution(0,-1,0,-1,5,-1,0,-1,0,0)")
builder("filter.noise", [100], "noise(100)")
builder("filter.noise", [101], "noise(101)", throws=True, code="invalid_noise")
builder("filter.rotate", [45], "rotate(45)")
builder("filter.rotate", [45.5], "rotate(45.5)")
builder("filter.trim", [], "trim(10)")
builder("filter.trim", [20], "trim(20)")
builder("filter.trimSides", [20, 5, 0, 5, 0], "trim(20,5,0,5,0)")
builder("filter.quality", [80], "quality(80)")
builder("filter.quality", ["auto"], "quality(auto)")
builder("filter.quality", [0], "quality(0)", throws=True, code="invalid_filters")
builder("filter.qualityPerFormat", [{"webp": 75, "jpeg": 90}, 80],
        "quality(80,jpeg:90,webp:75)")
builder("filter.palette", [64], "palette(64)")
builder("filter.palette", [1], "palette(1)", throws=True, code="invalid_filters")
builder("filter.format", ["jpg"], "format(jpeg)")
builder("filter.format", ["webp", {"method": 6, "sns": 50}], "format(webp,method:6,sns:50)")
builder("filter.format", ["avif", {"subsampling": 420}], "format(avif,subsampling:420)",
        throws=True, code="invalid_filters")
builder("filter.fill", ["FF00AA"], "fill(ff00aa)")
builder("filter.facePad", [1.5], "facepad(1.5)")
builder("filter.studio", ["product"], "studio(product)")
builder("filter.studio", ["product", {"bg": "white"}], "studio(product,bg:white)")
builder("filter.studio", ["product", {"pad": 10}], "studio(product)")
builder("filter.studio", ["scene", {"pad": 25}], "studio(scene,pad:25)")
builder("filter.studio", ["scene", {"bg": "white"}], "studio(scene,bg:white)",
        throws=True, code="invalid_filters")
builder("filter.upscale", [2], "upscale(2)")
builder("filter.upscale", [3], "upscale(3)", throws=True, code="invalid_filters")
builder("filter.dpr", [1.5], "dpr(1.5)")
builder("filter.dpr", [7], "dpr(7)", throws=True, code="invalid_filters")
builder("filter.maxBytes", [1000], "max_bytes(1000)")
builder("filter.radius", [12], "radius(12,12,12,12)")
builder("filter.radius", [[1, 2, 3, 4]], "radius(1,2,3,4)")
builder("filter.border", [0, "00ff00"], "border(0,00ff00)")
builder("filter.border", [2, "00FF00"], "border(2,00ff00)")
builder("filter.mask", ["circle"], "mask(circle)")
builder("filter.mask", ["square"], "mask(square)", throws=True, code="invalid_filters")
builder("filter.duotone", ["000000", "ffffff"], "duotone(000000,ffffff)")
builder("filter.monochrome", ["ff8000"], "monochrome(ff8000)")
builder("filter.pixelate", [8], "pixelate(8)")
builder("filter.pixelate", [0], "pixelate(0)", throws=True, code="invalid_filters")
builder("filter.flip", ["h"], "flip(h)")
builder("filter.watermark", ["https://x/wm.png", 0, 0, 80],
        "watermark(https%3A%2F%2Fx%2Fwm.png,0,0,80)")
builder("filter.overlay", ["https://x/ov.png", 0, 0, 80, "multiply"],
        "overlay(https%3A%2F%2Fx%2Fov.png,0,0,80,multiply)")
builder("filter.text", ["Hello", 12, "ff8800", "center"], "text(Hello,12,ff8800ff,center)")
builder("filter.blurClass", ["person"], "blurclass(person)")
builder("filter.pixelateClass", ["traffic light"], "pixelateclass(traffic%20light)")
builder("filter.recolorClass", ["person", "FF0000"], "recolorclass(person,ff0000)")
builder("filter.blurRegion", [1, 2, 30, 40], "blur_region(1,2,30,40)")
builder("filter.removeRegion", [0, 0, 5, 5], "remove_region(0,0,5,5)")


def render():
    doc = {
        "description": (
            "Generated-combination differential corpus for the operation "
            "grammar (issue #353): remainders with the verdict every "
            "surface must agree on, and TypeScript builder calls with the "
            "segments they must produce. Generated by "
            "spec/generate_operation_vectors.py; do not edit by hand. "
            "Consumed by the server crate's tests, the TypeScript SDK "
            "tests, and the playground tests. The hand-written audit "
            "corpus lives in spec/explain-verdicts.json."
        ),
        "version": 1,
        "signature": "a" * 64,
        "cases": CASES,
        "builder_cases": BUILDER_CASES,
    }
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    out = render()
    if "--check" in sys.argv:
        try:
            current = open("spec/operation-vectors.json").read()
        except FileNotFoundError:
            current = ""
        if current != out:
            print("spec/operation-vectors.json is stale; regenerate with "
                  "python3 spec/generate_operation_vectors.py", file=sys.stderr)
            raise SystemExit(1)
        print(f"operation-vectors.json is current ({len(CASES)} cases, "
              f"{len(BUILDER_CASES)} builder cases)")
    else:
        with open("spec/operation-vectors.json", "w") as fh:
            fh.write(out)
        print(f"wrote spec/operation-vectors.json: {len(CASES)} cases, "
              f"{len(BUILDER_CASES)} builder cases")
