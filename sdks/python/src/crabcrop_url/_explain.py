"""Parse a crabcrop URL (or bare path) back into its pieces and
explain each segment, as the server's grammar reads it: the Python
port of the TypeScript SDK's ``explainUrl`` (``explain.ts``, issue
#716).

The two explainers are to give one answer, message for message: both
run ``spec/explain-verdicts.json`` and ``spec/operation-vectors.json``,
and when Node is installed the test suite compares this one with the
TypeScript one, field by field, on thousands of generated inputs.
The closed sets, caps and class-name rules come from
``_grammar.py``, the copy the typed builders check against; what is
here is what only the explainer has: the dispatch, the number
readers, the messages and the request caps.

The port keeps the JavaScript semantics wherever Python's differ:
``\\d`` is ASCII only, ``$`` does not match before a final newline,
``.`` stops at every JavaScript line terminator, a long digit run is
never handed to ``int`` whole (by default Python converts at most
4300 digits), and
percent-escapes decode as ``decodeURIComponent`` decodes them.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Dict, List, Literal, Mapping, NamedTuple, Optional, Set, Tuple

from ._grammar import (
    BLEND_MODES,
    CROP_ALIGNS,
    CURVE_CHANNELS,
    DPR_FACTORS,
    FLIP_AXES,
    IMAGE_FORMATS,
    JPEG_SUBSAMPLINGS,
    MASK_SHAPES,
    MAX_DIMENSION,
    MAX_OBJECTS_CLASSES,
    QUALITY_FORMATS,
    STUDIO_BACKDROPS,
    STUDIO_MODES,
    STUDIO_SHADOWS,
    TEXT_ANCHORS,
    UPSCALE_FACTORS,
    class_fold_key,
    is_class_name,
    is_reserved_remainder,
    text_label_fits,
    trim_like_server,
)
from ._operations import _HEX6, _HEX8, _INT32, _PLAIN_DECIMAL, _UINT32_MAX, _js_number_string

#: The segment kinds an explanation is made of.
SegmentKind = Literal["expiry", "maxage", "debug", "preset", "operation"]

#: The signature kinds an explanation can report.
SignatureKind = Literal["hmac", "unsafe", "sealed", "invalid"]


@dataclass(frozen=True)
class ExplainedSignature:
    """The signature segment of an explained URL: its kind, the tag
    or token it carries, and what the server does with it. ``key_id``
    is the ``<key-id>:`` prefix of a keyed tag or token, and ``None``
    without one."""

    kind: SignatureKind
    value: str
    detail: str
    key_id: Optional[str] = None


@dataclass(frozen=True)
class ExplainedSegment:
    """One segment of the operations run, as written (``raw``), with
    what it does (``detail``) or why the server refuses it
    (``error``). ``expired`` is set on a well-formed ``exp:`` segment
    only: whether the URL had expired at the reference time."""

    raw: str
    kind: SegmentKind
    title: str
    detail: Optional[str] = None
    error: Optional[str] = None
    expired: Optional[bool] = None


@dataclass(frozen=True)
class Explanation:
    """What `explain_url` reads in a URL: the signature (``None`` when
    there is none to read), the operation segments, the source, the
    remainder the signature covers, and every error. An empty
    ``errors`` is the server's verdict that the request parses."""

    signature: Optional[ExplainedSignature]
    segments: List[ExplainedSegment]
    source: str
    remainder: str
    errors: List[str]


#: What each filter does, by name: the TypeScript SDK's
#: ``FILTER_DESCRIPTIONS``, read only. A filter the table does not
#: name is an unknown filter to the explainer.
FILTER_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "grayscale": "convert to grayscale",
        "invert": "invert the colors",
        "no_upscale": "never enlarge the source",
        "blur": "gaussian blur, argument is the sigma",
        "blurfaces": "blur detected faces (needs a face model on the server)",
        "brighten": "brighten by the given amount (-255..255)",
        "contrast": "contrast in percent, -100..100 (0 is unchanged, -100 is flat)",
        "gamma": (
            "gamma correction with divisor g, 0 < g <= 10: each channel becomes 255 * (v / "
            "255)^(1 / g) (1 is unchanged, above 1 brightens midtones)"
        ),
        "exposure": (
            "exposure in stops, -10..10: multiply the linear light by 2^stops (0 is unchanged, "
            "positive brightens, negative darkens)"
        ),
        "highlights_shadows": (
            "adjust the highlights and shadows ranges independently: two percents, each "
            "-100..100 (0 is unchanged)"
        ),
        "vibrance": (
            "saturation weighted toward the muted colors: percent, -100..100 (0 is unchanged; "
            "vivid colors and skin tones are protected)"
        ),
        "huerotate": "rotate the hue by the given degrees",
        "rotate": "rotate by the given angle in degrees; the canvas expands to fit",
        "quality": (
            "output quality 1..100, or `auto` for perceptual targeting; or per-format values in "
            "one list (quality(80,jpeg:90,webp:75)), resolved after format negotiation"
        ),
        "palette": (
            "write PNG output as an indexed (palette) image with at most this many colors, "
            "2..256; ignored by the other formats"
        ),
        "format": (
            "force the output format; a format also takes its encoder options "
            "(format(jpeg,progressive), format(png,interlace), format(webp,method:6,sns:50)); an "
            "option its encoder cannot honour is refused"
        ),
        "fill": "the background color other operations pad with",
        "facepad": "how much a face crop expands around the face",
        "pixelatefaces": "pixelate detected faces (needs a face model on the server)",
        "blurclass": (
            "blur every detected object of the named class (needs a detection model on the server)"
        ),
        "pixelateclass": (
            "pixelate every detected object of the named class (needs a detection model on the server)"
        ),
        "recolorclass": (
            "recolor every detected object of the named class to the given color (needs a "
            "detection model on the server)"
        ),
        "removebg": "remove the background (needs an AI model on the server)",
        "studio": (
            "studio mode: `product` composites the subject onto a backdrop (bg: white, a RRGGBB "
            "color, brand, or none) with a synthesized contact shadow (shadow: soft, strong, or "
            "none), scaled and centered to a pad margin (pad: 0..40, default 10); `scene` keeps "
            "the photograph's scene, reframing it on the detected subject with light and color "
            "correction (pad only) (needs an AI model on the server)"
        ),
        "keep_profile": "keep the source color profile instead of converting to sRGB",
        "upscale": "AI upscale by the given factor (needs an AI model on the server)",
        "equalize": "histogram equalization",
        "enhance": "gentle per-channel auto-levels (0.5th percentile to 0, 99.5th to 255)",
        "saturation": "saturation factor, 0..255 (0 is grayscale, 1 is identity)",
        "sharpen": "sharpen by the given amount",
        "text": "draw a text overlay (needs a font on the server)",
        "rgb": "tint toward the given red/green/blue amounts",
        "curve": (
            "tone curve: remap the named channel (rgb, r, g, or b) through 2 to 16 x,y control "
            "points, linearly interpolated, ends clamped"
        ),
        "convolution": (
            "convolve with a 3x3 or 5x5 kernel: 9 or 25 row-major values (plain decimals, "
            "-1024..1024), then a normalize flag (0 or 1); edges replicate"
        ),
        "noise": (
            "add monochrome film grain: one deterministic noise value per pixel on all "
            "channels, strength 0..100 percent of full scale (0 is a no-op)"
        ),
        "trim": (
            "trim uniform borders; the five-argument form (tolerance,left,top,right,bottom) "
            "also removes that many pixels per side after the tolerance pass"
        ),
        "dpr": "device pixel ratio the sizes are multiplied by",
        "watermark": "overlay a watermark image",
        "max_bytes": "cap the output size, lowering quality to fit",
        "radius": (
            "round the corners by the given radius in pixels (one value, or four in CSS corner "
            "order); needs an alpha-capable format"
        ),
        "border": "paint a border of the given width and color (RRGGBB) inside the frame",
        "mask": "keep only the inscribed `circle` or `ellipse`; the rest becomes transparent",
        "sepia": "the classic warm sepia tone",
        "duotone": "map shadows to the first color and highlights to the second, by luminance",
        "monochrome": "single-hue rendering in the given color (RRGGBB)",
        "pixelate": "mosaic into blocks of the given pixel size, each block averaged",
        "flip": "mirror the image: `h` horizontal, `v` vertical",
        "blur_region": (
            "gaussian-blur the rectangle (x,y,w,h) in pixels at this point in the chain (redaction)"
        ),
        "overlay": (
            "composite another image over the result: (url,x,y,alpha,blend[,width %]) with "
            "blend normal, multiply, screen, or overlay"
        ),
        "pixelate_region": (
            "pixelate the rectangle (x,y,w,h) in pixels at this point in the chain (redaction)"
        ),
        "remove_region": (
            "erase the rectangle (x,y,w,h) in pixels at this point in the chain and inpaint it "
            "from the surrounding pixels (AI region removal)"
        ),
    }
)

#: The most operations one request runs (url.rs MAX_OPERATIONS).
MAX_OPERATIONS = 100

#: The most distinct watermark and overlay URLs one request names
#: (url.rs MAX_WATERMARK_URLS, issue #613).
MAX_WATERMARK_URLS = 16

#: The most input `explain_url` reads, in UTF-8 bytes: 64 KiB. It
#: bounds the work one call does, and it sits above the URL limits of
#: common CDNs and proxies.
MAX_EXPLAIN_BYTES = 64 * 1024

_U64_MAX = 2**64 - 1

# ---------------------------------------------------------------------------
# Numbers (issue #712). The server reads every numeric argument with
# its Rust type's `FromStr`, and the explainer reads each one the same
# way, so the two give one verdict on every spelling.
# spec/url-signing.md (Argument spellings) states the rules.
# ---------------------------------------------------------------------------

_UINT = re.compile(r"\+?[0-9]+")
_INT = re.compile(r"[+-]?[0-9]+")
_DIGITS = re.compile(r"[0-9]+")

# Past 20 significant digits a value is outside every integer type the
# grammar reads (u64 is 20 digits), so it reads as this, signed.
_PAST_EVERY_TYPE = 10**21


def _integer(text: str) -> int:
    """The value of a digit run with an optional sign, already
    matched. Only the significant digits reach ``int``: by default
    Python refuses to convert a run past 4300 digits, and a URL can
    hold a longer one. A value past 20 significant digits reads as
    ``_PAST_EVERY_TYPE``, which no range the grammar checks holds."""
    negative = text.startswith("-")
    body = text[1:] if text[:1] in ("+", "-") else text
    significant = body.lstrip("0")
    value = _PAST_EVERY_TYPE if len(significant) > 20 else int(significant or "0")
    return -value if negative else value


def _is_uint_in(text: Optional[str], low: int, high: int) -> bool:
    """Whether ``text`` is an integer in ``low..=high`` as Rust's
    unsigned ``FromStr`` reads one: an optional ``+``, then ASCII
    digits, leading zeros allowed. A ``-`` is refused, even in
    ``-0``."""
    return text is not None and _UINT.fullmatch(text) is not None and low <= _integer(text) <= high


def _is_int_in(text: Optional[str], low: int, high: int) -> bool:
    """Whether ``text`` is an integer in ``low..=high`` as Rust's
    signed ``FromStr`` reads one: an optional ``+`` or ``-``, then
    ASCII digits. ``-0`` is 0."""
    return text is not None and _INT.fullmatch(text) is not None and low <= _integer(text) <= high


# Rust's float syntax: an optional sign, digits with an optional
# fraction (`2`, `2.`, `.5`), and an optional exponent. Each part
# matches one way, so a long run of digits costs linear time, matched
# or not (issue #594).
_DECIMAL = re.compile(r"([+-]?)([0-9]*)(?:\.([0-9]*))?(?:[eE]([+-]?)([0-9]+))?")

# The most significant digits `read_decimal` keeps. Every midpoint
# between two 64-bit floats has at most 767 significant digits, so a
# decimal cut to this many, with one nonzero digit appended when any
# cut digit is nonzero, rounds to the float the whole decimal rounds
# to.
_KEPT_DIGITS = 1000


class _Binary(NamedTuple):
    """The precision and range of one binary float format, and the
    decimal exponents past which a value is certainly infinite or
    zero there."""

    mantissa: int
    min_exponent: int
    max_exponent: int
    overflow: int
    underflow: int


_BINARY = {
    32: _Binary(24, -126, 127, 39, -46),
    64: _Binary(53, -1022, 1023, 309, -325),
}

#: The largest finite 32-bit float, Rust's ``f32::MAX``.
_F32_MAX = (2 - 2**-23) * 2.0**127


def read_decimal(text: str, bits: Literal[32, 64]) -> Optional[float]:
    """The float a decimal argument reads as on the server (issue
    #712): Rust's ``f32`` or ``f64`` ``FromStr``, which takes an
    optional sign, digits with an optional fraction (``2``, ``2.``,
    ``.5``), and an optional exponent (``e`` or ``E``, an optional
    sign, digits), and rounds the decimal to the nearest float of
    ``bits`` bits, a tie to the even one. Past the float's range the
    value is an infinity, and below half its smallest step it is
    zero, signed as the decimal is. Returns ``None`` for any other
    text, Rust's ``inf`` and ``NaN`` spellings included: no range the
    grammar checks admits them.

    The rounding is exact, in integer arithmetic: rounding to a
    double first and then to 32 bits rounds twice, and gets a decimal
    near the midpoint between two 32-bit floats wrong. A 32-bit
    result comes back as the double of the same value."""
    match = _DECIMAL.fullmatch(text)
    if match is None:
        return None
    sign, whole, fraction, exp_sign, exp_digits = match.groups()
    fraction = fraction or ""
    exp_sign = exp_sign or ""
    exp_digits = exp_digits or "0"
    if whole == "" and fraction == "":
        return None
    negative = sign == "-"
    zero = -0.0 if negative else 0.0
    infinity = -math.inf if negative else math.inf
    fmt = _BINARY[bits]
    # value = digits * 10**e10. Past nine digits the exponent only
    # has to stay past both pre-screens, as in the TypeScript reader.
    digits = (whole + fraction).lstrip("0")
    exp_trimmed = exp_digits.lstrip("0")
    magnitude = 10**9 if len(exp_trimmed) > 9 else int(exp_trimmed or "0")
    e10 = (-magnitude if exp_sign == "-" else magnitude) - len(fraction)
    stripped = digits.rstrip("0")
    e10 += len(digits) - len(stripped)
    digits = stripped
    if digits == "":
        return zero
    n = len(digits)
    if n - 1 + e10 >= fmt.overflow:
        return infinity
    if n + e10 <= fmt.underflow:
        return zero
    if n > _KEPT_DIGITS:
        sticky = digits[_KEPT_DIGITS:].strip("0") != ""
        e10 += n - _KEPT_DIGITS - (1 if sticky else 0)
        digits = digits[:_KEPT_DIGITS] + ("1" if sticky else "")
    d = int(digits)
    num = d * 10**e10 if e10 >= 0 else d
    den = 1 if e10 >= 0 else 10**-e10
    # e = floor(log2(num / den)), from the bit lengths.
    e = num.bit_length() - den.bit_length()
    if (num < den << e) if e >= 0 else (num << -e < den):
        e -= 1
    q = max(e, fmt.min_exponent) - (fmt.mantissa - 1)
    top = num if q >= 0 else num << -q
    bottom = den << q if q >= 0 else den
    mantissa = top // bottom
    twice = 2 * (top - mantissa * bottom)
    if twice > bottom or (twice == bottom and mantissa & 1):
        mantissa += 1
    if mantissa == 1 << fmt.mantissa:
        mantissa >>= 1
        q += 1
    if q > fmt.max_exponent - (fmt.mantissa - 1):
        return infinity
    value = math.ldexp(mantissa, q)
    return -value if negative else value


def _f32_in(text: Optional[str], low: float, high: float, above: bool = False) -> bool:
    """Whether ``text`` is a decimal whose 32-bit float is in
    ``low..=high``, or above ``low`` with ``above``: the bounds apply
    to the rounded value, as the server applies them, so
    ``blur(512.00001)`` is ``blur(512)``."""
    if text is None:
        return False
    value = read_decimal(text, 32)
    return value is not None and (value > low if above else value >= low) and value <= high


# The strict decimal of the `rotate` angle and the `convolution`
# kernel (url.rs parse_angle) is the builders' own `_PLAIN_DECIMAL`:
# an optional minus, digits, an optional fraction; no `+`, no
# exponent, no leading or trailing dot. A `frame:` selector takes no
# minus.
_FRAME_DECIMAL = re.compile(r"[0-9]+(\.[0-9]+)?")


def _is_kernel_value(text: str) -> bool:
    # url.rs parse_kernel_value: a strict decimal, read as an f64, at
    # most 1024 in absolute value.
    if _PLAIN_DECIMAL.fullmatch(text) is None:
        return False
    value = read_decimal(text, 64)
    return value is not None and abs(value) <= 1024


_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _percent_decode(text: str) -> Optional[str]:
    """The text the server's escapes decode to (url.rs decode_arg),
    as JavaScript's ``decodeURIComponent`` decodes it: ``%`` and two
    hex digits, the bytes UTF-8. Returns ``None`` where
    ``decodeURIComponent`` throws: a ``%`` without two hex digits
    after it (``%+1`` included, as on the server, issue #712), or
    bytes that are not UTF-8. ``urllib.parse.unquote`` passes both
    through, so it is not used."""
    if "%" not in text:
        return text
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "%":
            nxt = text.find("%", i)
            end = n if nxt == -1 else nxt
            out.append(text[i:end])
            i = end
            continue
        run = bytearray()
        while i < n and text[i] == "%":
            if i + 2 >= n or text[i + 1] not in _HEX_DIGITS or text[i + 2] not in _HEX_DIGITS:
                return None
            run.append(int(text[i + 1 : i + 3], 16))
            i += 3
        try:
            out.append(run.decode("utf-8"))
        except UnicodeDecodeError:
            return None
    return "".join(out)


# ---------------------------------------------------------------------------
# Sizes (url.rs parse_size and parse_dimension, audit P-3): each side
# is an optional `-` flip marker plus `` (proportional), `0`, `orig`,
# or a pixel count up to the dimension cap.
# ---------------------------------------------------------------------------


class _Dimension(NamedTuple):
    kind: Literal["proportional", "original", "pixels"]
    value: int
    flip: bool


def _parse_dimension(part: str) -> Optional[_Dimension]:
    flip = part.startswith("-")
    digits = part[1:] if flip else part
    if digits in ("", "0"):
        return _Dimension("proportional", 0, flip)
    if digits == "orig":
        return _Dimension("original", 0, flip)
    if _UINT.fullmatch(digits) is None:
        return None
    value = _integer(digits)
    if value > MAX_DIMENSION:
        return None
    return _Dimension("pixels", value, flip)


def _parse_size(seg: str) -> Optional[Tuple[_Dimension, _Dimension]]:
    """The two sides of a size segment, or ``None`` when the segment
    is not a size, which in the server's dispatch means the source
    starts."""
    x = seg.find("x")
    if x == -1:
        return None
    w = _parse_dimension(seg[:x])
    h = _parse_dimension(seg[x + 1 :])
    if w is None or h is None:
        return None
    return w, h


def _is_exact_box(size: Tuple[_Dimension, _Dimension]) -> bool:
    return all(d.kind == "pixels" and d.value >= 1 and not d.flip for d in size)


def _size_blurb(size: Tuple[_Dimension, _Dimension]) -> str:
    def describe(d: _Dimension, axis: str) -> str:
        if d.kind == "original":
            return f"the original {axis}"
        if d.kind == "proportional":
            return f"proportional {axis}"
        return f"{d.value}px {axis}"

    w, h = size
    flips = [
        text for flip, text in ((w.flip, "flipped horizontally"), (h.flip, "flipped vertically")) if flip
    ]
    tail = f" ({', '.join(flips)})" if flips else ""
    return f"resize to {describe(w, 'width')} by {describe(h, 'height')}{tail}"


def _is_size_shaped(seg: str) -> bool:
    """The server's size-shape test (url.rs size_shaped): an ``x``
    joining two parts, at least one of them a dimension spelling
    (digits, ``orig``, or empty, with an optional flip sign), and no
    ``.`` or ``:`` anywhere, the punctuation that marks URL text."""
    x = seg.find("x")
    if x == -1:
        return False
    left, right = seg[:x], seg[x + 1 :]
    if "." in left or "." in right or ":" in seg:
        return False

    def spelling(part: str) -> bool:
        digits = part[1:] if part.startswith("-") else part
        return digits in ("", "orig") or _DIGITS.fullmatch(digits) is not None

    return spelling(left) or spelling(right)


# ---------------------------------------------------------------------------
# Filters (url.rs parse_filters, ported).
# ---------------------------------------------------------------------------

_NO_ARG_FILTERS = frozenset(
    {
        "grayscale",
        "invert",
        "no_upscale",
        "equalize",
        "enhance",
        "removebg",
        "keep_profile",
        "sepia",
        "blurfaces",
        "pixelatefaces",
    }
)


def _hex6(text: Optional[str]) -> bool:
    return text is not None and _HEX6.fullmatch(text) is not None


def _trim(args: Optional[str]) -> bool:
    if args is None:
        return True
    parts = args.split(",")
    if len(parts) == 1:
        return _is_uint_in(parts[0], 0, 255)
    return (
        len(parts) == 5
        and _is_uint_in(parts[0], 0, 255)
        and all(_is_uint_in(p, 0, _UINT32_MAX) for p in parts[1:])
    )


def _highlights_shadows(args: Optional[str]) -> bool:
    if args is None:
        return False
    parts = args.split(",")
    return len(parts) == 2 and all(_f32_in(p, -100, 100) for p in parts)


def _quality(args: Optional[str]) -> bool:
    if args is None:
        return False
    if args == "auto" or _is_uint_in(args, 1, 100):
        return True
    # Per-format list (issue #319): at most one bare fallback, at most
    # one entry per format, no video formats (their quality is not the
    # server's to set).
    seen: Set[str] = set()
    fallback_seen = False
    for part in args.split(","):
        colon = part.find(":")
        if colon == -1:
            if fallback_seen or not (part == "auto" or _is_uint_in(part, 1, 100)):
                return False
            fallback_seen = True
        else:
            name, value = part[:colon], part[colon + 1 :]
            canonical = "jpeg" if name == "jpg" else name
            if canonical not in QUALITY_FORMATS or canonical in seen:
                return False
            seen.add(canonical)
            if not (value == "auto" or _is_uint_in(value, 1, 100)):
                return False
    return len(seen) + (1 if fallback_seen else 0) > 0


def _format(args: Optional[str]) -> bool:
    if args is None:
        return False
    # A format name plus its encoder options (issue #319), each
    # validated against the format it belongs to.
    name, *options = args.split(",")
    if name not in IMAGE_FORMATS:
        return False
    seen: Set[str] = set()
    for option in options:
        key = option[: option.find(":")] if ":" in option else option
        if key in seen:
            return False
        seen.add(key)
        if name in ("jpeg", "jpg"):
            if option == "progressive" or option in (f"subsampling:{s}" for s in JPEG_SUBSAMPLINGS):
                continue
            return False
        if name == "png":
            if option == "interlace":
                continue
            return False
        if name == "webp":
            # Each value is a u8 on the server (url.rs
            # parse_format_spec), so `method:+6` and `sns:050` are
            # spellings of 6 and 50.
            if key == "method" and _is_uint_in(option[7:], 0, 6):
                continue
            if key == "sns" and _is_uint_in(option[4:], 0, 100):
                continue
            return False
        return False
    return True


def _rgb(args: Optional[str]) -> bool:
    if args is None:
        return False
    parts = args.split(",")
    return len(parts) == 3 and all(_is_int_in(p, -255, 255) for p in parts)


def _curve(args: Optional[str]) -> bool:
    if args is None:
        return False
    parts = args.split(",")
    if parts[0] not in CURVE_CHANNELS:
        return False
    coords = parts[1:]
    if len(coords) < 4 or len(coords) > 32 or len(coords) % 2 != 0:
        return False
    if not all(_is_uint_in(p, 0, 255) for p in coords):
        return False
    return all(_integer(coords[i]) > _integer(coords[i - 2]) for i in range(2, len(coords), 2))


def _convolution(args: Optional[str]) -> bool:
    if args is None:
        return False
    parts = args.split(",")
    if len(parts) not in (10, 26) or parts[-1] not in ("0", "1"):
        return False
    return all(_is_kernel_value(p) for p in parts[:-1])


def _upscale(args: Optional[str]) -> bool:
    # A u32, then one of the factors.
    return args is not None and _is_uint_in(args, 0, _UINT32_MAX) and _integer(args) in UPSCALE_FACTORS


def _studio(args: Optional[str]) -> bool:
    # The server's studio grammar (url.rs parse_studio): the mode
    # (`product` or `scene`) first, then named `key:value` arguments,
    # each key at most once. Product takes `bg`, `shadow`, and `pad`;
    # scene takes `pad` only, because a backdrop and a shadow are
    # product concepts (naming one under scene is a bad argument, not
    # a silent no-op).
    if args is None:
        return False
    mode, *rest = args.split(",")
    if mode not in STUDIO_MODES:
        return False
    seen: Set[str] = set()
    for part in rest:
        cut = part.find(":")
        if cut == -1:
            return False
        key, value = part[:cut], part[cut + 1 :]
        if key in seen:
            return False
        seen.add(key)
        if key == "pad":
            # A Rust u8 (an optional `+`, then digits; no minus sign,
            # not even `-0`), bounded to 40.
            if not _is_uint_in(value, 0, 40):
                return False
        elif mode == "product" and key == "bg":
            if value not in STUDIO_BACKDROPS and not _hex6(value):
                return False
        elif mode == "product" and key == "shadow":
            if value not in STUDIO_SHADOWS:
                return False
        else:
            return False
    return True


def _dpr(args: Optional[str]) -> bool:
    # An f32, then one of the factors: `dpr(1.50000001)` rounds to 1.5.
    factor = None if args is None else read_decimal(args, 32)
    return factor is not None and factor in DPR_FACTORS


def _rotate(args: Optional[str]) -> bool:
    # url.rs parse_angle: a strict decimal, read as an f32, at most 100
    # turns either way.
    return args is not None and _PLAIN_DECIMAL.fullmatch(args) is not None and _f32_in(args, -36000, 36000)


def _radius(args: Optional[str]) -> bool:
    if args is None:
        return False
    parts = args.split(",")
    return len(parts) in (1, 4) and all(_is_uint_in(p, 0, MAX_DIMENSION) for p in parts)


def _border(args: Optional[str]) -> bool:
    if args is None:
        return False
    parts = args.split(",")
    return len(parts) == 2 and _is_uint_in(parts[0], 0, MAX_DIMENSION) and _hex6(parts[1])


def _duotone(args: Optional[str]) -> bool:
    if args is None:
        return False
    parts = args.split(",")
    return len(parts) == 2 and all(_hex6(p) for p in parts)


def _overlay(args: Optional[str]) -> bool:
    if args is None:
        return False
    # The server splits from the right (rsplitn(6, ',')): with six or
    # more fields the rightmost IS the width and must parse, and the
    # URL keeps any commas left over. Splitting any other way misreads
    # both.
    fields = args.split(",")
    if len(fields) < 5:
        return False
    rest = fields
    if len(fields) >= 6:
        if not _is_uint_in(fields[-1], 1, 400):
            return False
        rest = fields[:-1]
    if (
        rest[-1] not in BLEND_MODES
        or not _is_uint_in(rest[-2], 0, 100)
        or not _is_int_in(rest[-3], *_INT32)
        or not _is_int_in(rest[-4], *_INT32)
    ):
        return False
    url = _percent_decode(",".join(rest[:-4]))
    return url is not None and url != ""


def _class_filter(args: Optional[str]) -> bool:
    if args is None:
        return False
    decoded = _percent_decode(args)
    return decoded is not None and is_class_name(trim_like_server(decoded))


def _recolor_class(args: Optional[str]) -> bool:
    if args is None:
        return False
    cut = args.rfind(",")
    if cut < 0 or not _hex6(args[cut + 1 :]):
        return False
    decoded = _percent_decode(args[:cut])
    return decoded is not None and is_class_name(trim_like_server(decoded))


def _region(args: Optional[str]) -> bool:
    if args is None:
        return False
    parts = args.split(",")
    return (
        len(parts) == 4
        and _is_uint_in(parts[0], 0, MAX_DIMENSION)
        and _is_uint_in(parts[1], 0, MAX_DIMENSION)
        and _is_uint_in(parts[2], 1, MAX_DIMENSION)
        and _is_uint_in(parts[3], 1, MAX_DIMENSION)
    )


def _text(args: Optional[str]) -> bool:
    if args is None:
        return False
    parts = args.split(",")
    if len(parts) < 4:
        return False
    anchor, color, size = parts[-1], parts[-2], parts[-3]
    text = _percent_decode(",".join(parts[:-3]))
    return (
        anchor in TEXT_ANCHORS
        and (_HEX6.fullmatch(color) is not None or _HEX8.fullmatch(color) is not None)
        and _is_uint_in(size, 1, 1000)
        and text is not None
        and text != ""
        and text_label_fits(text, _integer(size))
    )


def _watermark(args: Optional[str]) -> bool:
    if args is None:
        return False
    parts = args.split(",")
    if len(parts) < 4:
        return False
    alpha, y, x = parts[-1], parts[-2], parts[-3]
    url = _percent_decode(",".join(parts[:-3]))
    return (
        _is_uint_in(alpha, 0, 100)
        and _is_int_in(y, *_INT32)
        and _is_int_in(x, *_INT32)
        and url is not None
        and url != ""
    )


# Each filter that takes arguments, and whether the server accepts an
# argument string (None for a filter written without parentheses).
_FILTER_ARGUMENTS: Dict[str, Callable[[Optional[str]], bool]] = {
    "trim": _trim,
    # The server's blur bound (url.rs MAX_BLUR_SIGMA): a normal f32, so
    # at least 2^-126 (a subnormal sigma panics the image crate's
    # Gaussian setup), and at most 512.
    "blur": lambda args: _f32_in(args, 2.0**-126, 512),
    "sharpen": lambda args: _f32_in(args, 0, _F32_MAX, above=True),
    "brighten": lambda args: _is_int_in(args, -255, 255),
    "huerotate": lambda args: _is_int_in(args, *_INT32),
    "contrast": lambda args: _f32_in(args, -100, 100),
    "gamma": lambda args: _f32_in(args, 0, 10, above=True),
    "exposure": lambda args: _f32_in(args, -10, 10),
    "highlights_shadows": _highlights_shadows,
    "vibrance": lambda args: _f32_in(args, -100, 100),
    "saturation": lambda args: _f32_in(args, 0, 255),
    "quality": _quality,
    "palette": lambda args: _is_uint_in(args, 2, 256),
    "format": _format,
    "rgb": _rgb,
    "curve": _curve,
    "convolution": _convolution,
    "noise": lambda args: _is_uint_in(args, 0, 100),
    "facepad": lambda args: _f32_in(args, 0, 3),
    "upscale": _upscale,
    "studio": _studio,
    "dpr": _dpr,
    "rotate": _rotate,
    "fill": _hex6,
    "radius": _radius,
    "border": _border,
    "mask": lambda args: args in MASK_SHAPES,
    "duotone": _duotone,
    "monochrome": _hex6,
    "pixelate": lambda args: _is_uint_in(args, 1, MAX_DIMENSION),
    "flip": lambda args: args in FLIP_AXES,
    "overlay": _overlay,
    "blurclass": _class_filter,
    "pixelateclass": _class_filter,
    "recolorclass": _recolor_class,
    "blur_region": _region,
    "pixelate_region": _region,
    "remove_region": _region,
    "max_bytes": lambda args: _is_uint_in(args, 1, _U64_MAX),
    "text": _text,
    "watermark": _watermark,
}


def _validate_filter(name: str, args: Optional[str]) -> Optional[str]:
    """Validate one filter against the server's rules. ``args`` is
    ``None`` when the filter has no parentheses (empty parentheses
    count as none, as on the server) and the raw argument string
    otherwise. Returns an error string, or ``None`` when the server
    would accept it."""
    bad = f"the server rejects this filter: {name}{'' if args is None else f'({args})'}"
    if name in _NO_ARG_FILTERS:
        return None if args is None else bad
    check = _FILTER_ARGUMENTS.get(name)
    if check is None:
        return f"unknown filter: {name}"
    return None if check(args) else bad


def _filter_chain(rest: str) -> List[str]:
    """The filters of a ``filters:`` segment's list. The server splits
    the list on the ':' separators that sit outside parentheses
    (url.rs split_filter_chain): an argument list may itself contain
    ':' (the studio filter's named arguments are ``key:value``), so a
    naive split would cut those filters apart. Unbalanced parentheses
    leave the tail in one part, which then fails its own parse, as on
    the server."""
    parts: List[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(rest):
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == ":" and depth == 0:
            parts.append(rest[start:i])
            start = i + 1
    parts.append(rest[start:])
    return parts


# JavaScript's `.` stops at four line terminators; Python's `.` only at
# `\n`. The argument list is matched with JavaScript's set, so a `\r`
# or a U+2028 in it makes the filter malformed, as in the TypeScript
# explainer.
_FILTER_CALL = re.compile("([a-z_0-9]+)(?:\\(([^\n\r\u2028\u2029]*)\\))?")
_COMPOSITE_CALL = re.compile("(watermark|overlay)\\(([^\n\r\u2028\u2029]*)\\)")


def _explain_filters(rest: str) -> Tuple[Optional[str], Optional[str]]:
    blurbs: List[str] = []
    for part in _filter_chain(rest):
        match = _FILTER_CALL.fullmatch(part)
        if match is None:
            return None, f"malformed filter: {part}"
        name, raw_args = match.groups()
        args = None if raw_args is None or raw_args == "" else raw_args
        blurb = FILTER_DESCRIPTIONS.get(name)
        if not blurb:
            return None, f"unknown filter: {name}"
        invalid = _validate_filter(name, args)
        if invalid:
            return None, invalid
        blurbs.append(f"{name}({args}): {blurb}" if args is not None else f"{name}: {blurb}")
    return "; ".join(blurbs), None


def _composited_url(call: str) -> Optional[str]:
    """The decoded image URL a valid ``watermark(...)`` or
    ``overlay(...)`` filter composites, or ``None`` for any other
    filter. Both split from the right, as their checks read them."""
    match = _COMPOSITE_CALL.fullmatch(call)
    if match is None:
        return None
    fields = match.group(2).split(",")
    numeric = 3 if match.group(1) == "watermark" else 5 if len(fields) >= 6 else 4
    return _percent_decode(",".join(fields[:-numeric]))


def _cap_errors(segments: List[ExplainedSegment]) -> List[str]:
    """The caps the server applies once a request parses (url.rs
    check_operation_count and check_watermark_count, issues #469 and
    #613): at most `MAX_OPERATIONS` operations, each filter in a
    ``filters:`` segment counting once, and at most
    `MAX_WATERMARK_URLS` distinct watermark and overlay URLs, compared
    as decoded. A preset reference counts as one operation: its
    definition holds at least one, and the explainer sees neither the
    definition nor the server's default operations, so both counts
    are lower bounds, and a cap they pass is one the server refuses
    too."""
    operations = 0
    presets = False
    urls: Set[str] = set()
    for segment in segments:
        if segment.kind in ("expiry", "maxage"):
            continue
        presets = presets or segment.kind == "preset"
        if segment.kind == "operation" and segment.raw.startswith("filters:"):
            for call in _filter_chain(segment.raw[8:]):
                operations += 1
                url = _composited_url(call)
                if url is not None:
                    urls.add(url)
        else:
            operations += 1
    at_least = "at least " if presets else ""
    if operations > MAX_OPERATIONS:
        return [
            f"the request has {at_least}{operations} operations, over the {MAX_OPERATIONS} "
            "operation limit: the server answers 400 (unsupported_operation)"
        ]
    if len(urls) > MAX_WATERMARK_URLS:
        return [
            f"the request names {len(urls)} distinct watermark and overlay URLs, over the "
            f"{MAX_WATERMARK_URLS} URL limit: the server answers 400 (unsupported_operation)"
        ]
    return []


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

_RECT_POINTS = re.compile(r"(\+?[0-9]+)x(\+?[0-9]+):(\+?[0-9]+)x(\+?[0-9]+)")


def _is_rect_points(rest: str) -> bool:
    """``<left>x<top>:<right>x<bottom>`` as url.rs parse_rect_points
    reads it (crop-rect and focal): four u32s, with the right edge
    past the left and the bottom past the top."""
    match = _RECT_POINTS.fullmatch(rest)
    if match is None or not all(_is_uint_in(n, 0, _UINT32_MAX) for n in match.groups()):
        return False
    left, top, right, bottom = (_integer(n) for n in match.groups())
    return right > left and bottom > top


def _parse_objects_align(align: str) -> Optional[Tuple[List[str], float]]:
    """The objects crop alignment (url.rs parse_objects):
    ``objects(<class>[,<class>...])[:<scale>]``. Returns the decoded
    class names (duplicates folded ASCII-only, first spelling kept, as
    the server's canonical form folds them) and the scale, or ``None``
    when the server would reject the form. The 32-class cap counts
    distinct names, as the server's does: repeats are free, and a 33rd
    distinct name rejects the form (issue #408)."""
    if not align.startswith("objects("):
        return None
    inner = align[len("objects(") :]
    close = inner.find(")")
    if close < 0:
        return None
    names, tail = inner[:close], inner[close + 1 :]
    scale = 1.0
    if tail != "":
        if not tail.startswith(":"):
            return None
        text = tail[1:]
        # An f32 above 0 and at most 100, the bound applied to the
        # rounded value, as on the server. The explanation shows the
        # decimal as JavaScript's Number reads it: the 64-bit float.
        wide = read_decimal(text, 64)
        if wide is None or not _f32_in(text, 0, 100, above=True):
            return None
        scale = wide
    classes: List[str] = []
    seen: Set[str] = set()
    for part in names.split(","):
        decoded = _percent_decode(part)
        if decoded is None:
            return None
        name = trim_like_server(decoded)
        if not is_class_name(name):
            return None
        key = class_fold_key(name)
        if key not in seen:
            if len(classes) == MAX_OBJECTS_CLASSES:
                return None
            seen.add(key)
            classes.append(name)
    return (classes, scale) if classes else None


def _civil_from_days(days: int) -> Tuple[int, int, int]:
    """The proleptic Gregorian (year, month, day) ``days`` after
    1970-01-01 (Howard Hinnant's civil_from_days). ``datetime`` stops
    at the year 9999, and JavaScript's dates run to 275760."""
    z = days + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    day = doy - (153 * mp + 2) // 5 + 1
    month = mp + 3 if mp < 10 else mp - 9
    return yoe + era * 400 + (1 if month <= 2 else 0), month, day


#: The last second a JavaScript Date represents.
_LAST_DATE_SECOND = 8640000000000


def _utc_text(seconds: int) -> str:
    """``seconds`` after the epoch, spelled as the TypeScript explainer
    spells it: ``Date.toISOString`` with the ``T`` a space and ``Z``
    ``UTC``, so a year past 9999 is the six-digit ``+YYYYYY``."""
    days, rest = divmod(seconds, 86400)
    year, month, day = _civil_from_days(days)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    spelled = f"{year:04d}" if year <= 9999 else f"+{year:06d}"
    return f"{spelled}-{month:02d}-{day:02d} {hours:02d}:{minutes:02d}:{secs:02d} UTC"


_CROP_ALIGN_NAMES = frozenset(CROP_ALIGNS)

_DEBUG_DETAIL = (
    "the response is a diagnostic image instead of the transform result: the canvas the next "
    "crop operates on, with the chosen focal point (magenta crosshair), the focal rectangle "
    "when one is in play, the detection boxes that fed the decision (amber), and the crop "
    "window outline (blue); with no crop it shows the focal analysis of the final image; debug "
    "responses are never cached and are served with Cache-Control: no-store; animated sources "
    "and video formats are refused with 400"
)


def _explain_expiry(seg: str, now_seconds: int) -> ExplainedSegment:
    value = seg[4:]
    if _DIGITS.fullmatch(value) is None:
        return ExplainedSegment(seg, "expiry", "Expiry", error=f"malformed expiry: {seg}")
    # The server reads a u64, and a huge value must report its clean
    # 400 (audit P-3).
    seconds = _integer(value)
    if seconds > _U64_MAX:
        return ExplainedSegment(
            seg,
            "expiry",
            "Expiry",
            error=f"expiry overflows the server's u64 parser: {seg} (the server answers 400)",
        )
    expired = now_seconds >= seconds
    when = _utc_text(seconds) if seconds <= _LAST_DATE_SECOND else "a date beyond representation"
    flag = " — EXPIRED at the reference time" if expired else ""
    return ExplainedSegment(
        seg, "expiry", "Expiry", detail=f"valid strictly before {when} ({value}){flag}", expired=expired
    )


def _explain_maxage(seg: str) -> ExplainedSegment:
    value = seg[7:]
    if _DIGITS.fullmatch(value) is None:
        return ExplainedSegment(
            seg,
            "maxage",
            "Max-age",
            error=f"malformed maxage: {seg} (want maxage:<seconds> from 0 to 31536000)",
        )
    if _integer(value) > 31536000:
        return ExplainedSegment(
            seg,
            "maxage",
            "Max-age",
            error=f"maxage out of range: {seg} (0 to 31536000 seconds; the server answers 400)",
        )
    if value == "0":
        detail = (
            "this response is served with Cache-Control: public, max-age=0 and no immutable "
            "flag: caches must revalidate before reuse; the stored image is unchanged"
        )
    else:
        detail = (
            f"this response is served with Cache-Control max-age {value} seconds instead of the "
            "default year; the stored image is unchanged"
        )
    return ExplainedSegment(seg, "maxage", "Max-age", detail=detail)


def _explain_frame(seg: str) -> ExplainedSegment:
    rest = seg[6:]

    # The server's frame grammar (url.rs parse_frame): the poster
    # keyword, or a strict decimal (no sign, no exponent, no leading
    # dot) read as an f64, as seconds up to 86400 or, with a % suffix,
    # as a percent up to 100.
    def decimal(text: str, high: float) -> bool:
        if _FRAME_DECIMAL.fullmatch(text) is None:
            return False
        value = read_decimal(text, 64)
        return value is not None and value <= high

    if rest == "poster":
        return ExplainedSegment(seg, "operation", "Frame", detail="render the video source's poster frame")
    if rest.endswith("%") and decimal(rest[:-1], 100):
        return ExplainedSegment(
            seg, "operation", "Frame", detail=f"render the frame {rest} through the video source"
        )
    if not rest.endswith("%") and decimal(rest, 86400):
        return ExplainedSegment(
            seg, "operation", "Frame", detail=f"render the frame at {rest} seconds into the video source"
        )
    return ExplainedSegment(
        seg,
        "operation",
        "Frame",
        error=(
            f"malformed frame segment: {seg} (want frame:poster, frame:<seconds> up to 86400, or "
            "frame:<percent>% up to 100; the server answers 400)"
        ),
    )


def _explain_page(seg: str) -> ExplainedSegment:
    rest = seg[5:]
    # The server's page grammar (url.rs parse_page): ASCII digits,
    # leading zeros allowed, 1 to 999999.
    digits = rest.lstrip("0")
    if _DIGITS.fullmatch(rest) is None or digits == "" or len(digits) > 6:
        return ExplainedSegment(
            seg,
            "operation",
            "Page",
            error=(
                f"malformed page segment: {seg} (want page:<n> with n from 1 to 999999; the "
                "server answers 400)"
            ),
        )
    return ExplainedSegment(seg, "operation", "Page", detail=f"render page {int(digits)} of the PDF source")


def _explain_focal_point(seg: str) -> ExplainedSegment:
    rest = seg[11:]
    colon = rest.find(":")
    point = rest if colon == -1 else rest[:colon]
    zoom = None if colon == -1 else rest[colon + 1 :]
    xy = point.split("x")
    # Each number an f32, the bounds applied to the rounded value
    # (url.rs parse_focal_point).
    ok = (
        len(xy) == 2
        and _f32_in(xy[0], 0, 1)
        and _f32_in(xy[1], 0, 1)
        and (zoom is None or _f32_in(zoom, 0, 100, above=True))
    )
    if not ok:
        return ExplainedSegment(
            seg,
            "operation",
            "Focal point",
            error=(
                f"malformed focalpoint segment: {seg} (want focalpoint:<x>x<y>[:<zoom>] with x "
                "and y in 0..1 and zoom in (0, 100])"
            ),
        )
    return ExplainedSegment(
        seg,
        "operation",
        "Focal point",
        detail=(
            f"set the focal point {point} at zoom {'1' if zoom is None else zoom} for the next "
            "crop (fractions of the image dimensions; zoom above 1 tightens the crop around the "
            "point)"
        ),
    )


def _explain_crop(seg: str) -> ExplainedSegment:
    rest = seg[5:]
    idx = rest.find(":")
    dims = rest if idx == -1 else rest[:idx]
    align = "center" if idx == -1 else rest[idx + 1 :]
    size = _parse_size(dims)
    objects = _parse_objects_align(align) if align.startswith("objects") else None
    if size is None or not _is_exact_box(size) or (align not in _CROP_ALIGN_NAMES and objects is None):
        return ExplainedSegment(
            seg,
            "operation",
            "Crop",
            error=(
                f"malformed crop segment: {seg} (want crop:<w>x<h>[:center|smart|face|top|bottom|"
                "left|right|top-left|top-right|bottom-left|bottom-right|objects(<class>[,<class>"
                "...])[:<scale>]], pixels only, no flips)"
            ),
        )
    if objects is not None:
        classes, scale = objects
        return ExplainedSegment(
            seg,
            "operation",
            "Crop",
            detail=(
                f"crop to the aspect of {dims} and resize to it; alignment: objects "
                f"({', '.join(classes)}): frame the union of every detection of the named "
                f"classes, the union box scaled by {_js_number_string(scale)} about its center "
                "and the crop centered on it; needs a detection model on the server, and falls "
                "back to the plain center crop of the same size when none of the classes is "
                "detected"
            ),
        )
    shown = "center (or the focal point for smart)" if align == "center" else align
    return ExplainedSegment(
        seg, "operation", "Crop", detail=f"crop to the aspect of {dims} and resize to it; alignment: {shown}"
    )


def _explain_fit_in(seg: str) -> ExplainedSegment:
    rest = seg[7:]
    idx = rest.find(":")
    dims = rest if idx == -1 else rest[:idx]
    modifier = None if idx == -1 else rest[idx + 1 :]
    size = _parse_size(dims)
    if size is None or size[0].flip or size[1].flip or (modifier is not None and modifier != "adaptive"):
        return ExplainedSegment(
            seg,
            "operation",
            "Fit in",
            error=(
                f"malformed fit-in segment: {seg} (want fit-in:<w>x<h>[:adaptive], flips are not "
                "allowed here)"
            ),
        )
    if modifier == "adaptive":
        return ExplainedSegment(
            seg,
            "operation",
            "Fit inside a box",
            detail=(
                f"scale proportionally to fit inside {dims}, swapping the box axes when the "
                "source orientation disagrees (adaptive)"
            ),
        )
    return ExplainedSegment(
        seg, "operation", "Fit inside a box", detail=f"scale proportionally to fit inside {rest}"
    )


def _explain_pad(seg: str) -> ExplainedSegment:
    rest = seg[4:]
    idx = rest.find(":")
    dims = rest if idx == -1 else rest[:idx]
    fill = None if idx == -1 else rest[idx + 1 :]
    size = _parse_size(dims)
    if size is None or not _is_exact_box(size) or (fill is not None and fill != "blur" and not _hex6(fill)):
        return ExplainedSegment(
            seg,
            "operation",
            "Pad",
            error=f"malformed pad segment: {seg} (want pad:<w>x<h>[:<hex>|blur], pixels only, no flips)",
        )
    shown = "the fill(...) color or black" if fill is None else fill
    return ExplainedSegment(
        seg,
        "operation",
        "Pad",
        detail=f"fit inside {dims} and fill the canvas to exactly that size, fill: {shown}",
    )


def _explain_rect(seg: str, name: str, title: str, detail: str) -> ExplainedSegment:
    """A ``crop-rect:`` or ``focal:`` segment: its points, or the
    error that names the form."""
    if not _is_rect_points(seg[len(name) + 1 :]):
        return ExplainedSegment(
            seg,
            "operation",
            title,
            error=(
                f"malformed {name} segment: {seg} (want {name}:<l>x<t>:<r>x<b> with right > left "
                "and bottom > top)"
            ),
        )
    return ExplainedSegment(seg, "operation", title, detail=detail)


def _explain_segment(seg: str, now_seconds: int) -> Optional[ExplainedSegment]:
    """The explanation of one operations-run segment, or ``None`` when
    it is not an operation, which in the server's dispatch means the
    source starts here."""
    if seg.startswith("exp:"):
        return _explain_expiry(seg, now_seconds)
    if seg.startswith("maxage:"):
        return _explain_maxage(seg)
    if seg.startswith("debug:"):
        if seg[6:] != "focal":
            return ExplainedSegment(
                seg,
                "debug",
                "Debug view",
                error=f"unsupported debug view: {seg} (want debug:focal; the server answers 400)",
            )
        return ExplainedSegment(seg, "debug", "Debug view", detail=_DEBUG_DETAIL)
    if seg.startswith("frame:"):
        return _explain_frame(seg)
    if seg.startswith("page:"):
        return _explain_page(seg)
    if seg.startswith("preset:"):
        name = seg[7:]
        if not name:
            return ExplainedSegment(seg, "preset", "Preset", error="empty preset name")
        return ExplainedSegment(
            seg,
            "preset",
            "Preset",
            detail=(
                f'expands server-side to the operations the operator named "{name}"; the '
                "signature covers this reference, not the expansion"
            ),
        )
    if seg.startswith("crop-rect:"):
        rest = seg[10:]
        return _explain_rect(
            seg,
            "crop-rect",
            "Crop rectangle",
            f"crop the exact rectangle {rest} (left x top : right x bottom)",
        )
    if seg.startswith("focal:"):
        rest = seg[6:]
        return _explain_rect(
            seg,
            "focal",
            "Focal rectangle",
            f"set the focal rectangle {rest} for the next crop (left x top : right x bottom)",
        )
    if seg.startswith("focalpoint:"):
        return _explain_focal_point(seg)
    if seg.startswith("crop:"):
        return _explain_crop(seg)
    if seg.startswith("fit-in:"):
        return _explain_fit_in(seg)
    if seg.startswith("pad:"):
        return _explain_pad(seg)
    if seg.startswith("filters:"):
        detail, error = _explain_filters(seg[8:])
        if error:
            return ExplainedSegment(seg, "operation", "Filters", error=error)
        return ExplainedSegment(seg, "operation", "Filters", detail=detail)
    size = _parse_size(seg)
    if size is not None:
        return ExplainedSegment(seg, "operation", "Resize", detail=_size_blurb(size))
    if _is_size_shaped(seg):
        # The server's dispatch (url.rs size_shaped, issue #228): a
        # segment shaped like a size that fails the size grammar is a
        # malformed size, not the start of the source, so the request
        # is a 400 rather than a fetch of a nonsense source.
        return ExplainedSegment(
            seg,
            "operation",
            "Resize",
            error=(
                f"malformed size segment: {seg} (each side is a pixel count up to {MAX_DIMENSION}, "
                "0, or orig; the server answers 400)"
            ),
        )
    return None


# ---------------------------------------------------------------------------
# The URL
# ---------------------------------------------------------------------------

_SCHEME = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*")
_QUERY = re.compile(r"[?#]")
_SEALED = re.compile(r"(?:([A-Za-z0-9_-]{1,32}):)?s1\.([A-Za-z0-9_-]+)")
_KEYED = re.compile(r"([A-Za-z0-9_-]{1,32}):([0-9a-f]{64})")
_TAG = re.compile(r"[0-9a-f]{64}")
_LOWER_HEX = re.compile(r"[0-9a-f]+")
_SEALED_SHAPE = re.compile(r"(?:[A-Za-z0-9_-]{1,32}:)?s[0-9]+\.")


def _signature_of(raw: str) -> ExplainedSignature:
    if raw == "unsafe":
        return ExplainedSignature(
            "unsafe",
            raw,
            "unsigned; the server honors it only with CRABCROP_ALLOW_UNSAFE or in presets-only deployments",
        )
    keyed = _KEYED.fullmatch(raw)
    if keyed is not None:
        key_id, tag = keyed.groups()
        return ExplainedSignature(
            "hmac",
            tag,
            (
                f'HMAC-SHA256 of the remainder, signed with key "{key_id}"; the server verifies it '
                "against that key only (secret rotation)"
            ),
            key_id,
        )
    if _TAG.fullmatch(raw) is not None:
        return ExplainedSignature(
            "hmac", raw, "HMAC-SHA256 of the remainder; paste the secret to verify it below"
        )
    if _LOWER_HEX.fullmatch(raw) is not None:
        return ExplainedSignature(
            "invalid", raw, "lowercase hex, but not a full 64-character HMAC tag: the server will answer 403"
        )
    if _SEALED_SHAPE.match(raw) is not None:
        return ExplainedSignature(
            "invalid",
            raw,
            (
                "a sealed-token shape this scheme does not open (an unknown version marker, or a "
                "malformed token body): the server answers the uniform 403 refusal, "
                "indistinguishable from a token that fails to open"
            ),
        )
    return ExplainedSignature(
        "invalid", raw, "not `unsafe` and not lowercase hex: the server rejects it as malformed (400)"
    )


def _as_javascript_reads(text: str) -> str:
    """``text`` as JavaScript reads the same UTF-16 code units: a high
    surrogate followed by a low one is the one character the pair
    encodes, and any other surrogate stays a lone surrogate, as it
    would in a JavaScript string."""
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")


def _too_long() -> Explanation:
    return Explanation(
        None,
        [],
        "",
        "",
        [f"the URL is longer than 64 KiB ({MAX_EXPLAIN_BYTES} bytes): the explainer does not read it"],
    )


def explain_url(input: str, now_seconds: Optional[int] = None) -> Explanation:
    """Parse a crabcrop URL (or bare path) back into its pieces,
    explaining each segment, as the TypeScript SDK's ``explainUrl``
    does. It mirrors the server's dispatch: segments are operations
    until the first segment that is not one, and everything from there
    is the source. ``now_seconds`` (unix seconds, the current time by
    default) is the reference time an ``exp:`` segment is checked
    against.

    The answer is the server's verdict: it reads each number as the
    server's types read it, and it reports what the server refuses
    before and after the parse (a remainder that begins with
    ``crabcrop``, ``purge``, ``purge-group`` or ``upload``, more than
    100 operations, and more than 16 distinct watermark and overlay
    URLs). It cannot see preset definitions or the server's default
    operations, so it counts each ``preset:`` reference as one
    operation.

    Input longer than 64 KiB of UTF-8 is not read: the answer is one
    error and no signature, so `verify_url_signature` returns
    ``None`` for it.

    A string that holds surrogate code points, which no UTF-8 text
    does, is read as JavaScript reads the same UTF-16: a surrogate
    pair is the one character it encodes, and a lone surrogate counts
    the three bytes TextEncoder writes for it."""
    if not isinstance(input, str):
        raise TypeError(f"explain_url takes the URL as a str, got {type(input).__name__}")
    if now_seconds is None:
        now_seconds = int(time.time())
    # A UTF-8 byte per character at least, so a long string is refused
    # before it is encoded.
    if len(input) > MAX_EXPLAIN_BYTES:
        return _too_long()
    try:
        size = len(input.encode("utf-8"))
    except UnicodeEncodeError:
        input = _as_javascript_reads(input)
        size = len(input.encode("utf-8", "surrogatepass"))
    if size > MAX_EXPLAIN_BYTES:
        return _too_long()
    # Extract the path by string surgery, never with a URL parser:
    # WHATWG parsing resolves dot segments (.. and %2e%2e) that the
    # server treats as literal path bytes (audit P-3). A query or
    # fragment is not part of the signed path, on either side.
    path = trim_like_server(input)
    scheme = path.find("://")
    if scheme > 0 and _SCHEME.fullmatch(path[:scheme]) is not None:
        after_authority = path[scheme + 3 :]
        slash = after_authority.find("/")
        path = "" if slash == -1 else after_authority[slash + 1 :]
    elif path.startswith("/"):
        path = path[1:]
    cut = _QUERY.search(path)
    if cut is not None:
        path = path[: cut.start()]
    # A sealed URL is a single segment: the token carries the whole
    # remainder encrypted, so there is no slash and nothing visible to
    # itemize (issue #388).
    sealed_whole = _SEALED.fullmatch(path)
    if sealed_whole is not None:
        key_id, body = sealed_whole.groups()
        signature = ExplainedSignature(
            "sealed",
            f"s1.{body}",
            (
                "sealed token (AES-256-GCM): the operations and source are encrypted inside it, so "
                "there is nothing visible to itemize; the server opens it and serves the URL it "
                "carries"
            ),
            key_id,
        )
        return Explanation(signature, [], "", "", [])
    slash = path.find("/")
    if slash == -1:
        return Explanation(None, [], "", "", ["no signature segment: the path is a single segment"])
    signature_raw, remainder = path[:slash], path[slash + 1 :]
    sealed = _SEALED.fullmatch(signature_raw)
    if sealed is not None:
        # A sealed token with a trailing path (issue #388): the server
        # opens the token and serves the URL it carries; the visible
        # remainder behind it is not part of that URL, so there is
        # nothing to itemize.
        key_id, body = sealed.groups()
        signature = ExplainedSignature(
            "sealed",
            f"s1.{body}",
            (
                "sealed token (AES-256-GCM): the operations and source are encrypted inside it; "
                "the server opens it and serves the URL it carries, and anything visible after "
                "the token is not part of that URL"
            ),
            key_id,
        )
        return Explanation(signature, [], "", remainder, [])
    errors: List[str] = []
    signature = _signature_of(signature_raw)
    if signature.kind == "invalid":
        errors.append("malformed signature segment")
    # Remainders the server refuses before it reads an operation,
    # whatever the signature (server/admission.rs resolve_request):
    # the segment key derivation owns, and the endpoints' own segments.
    if is_reserved_remainder(remainder):
        errors.append(
            "the remainder begins with the reserved segment crabcrop, which key derivation owns: "
            "the server answers 400 (reserved_segment)"
        )
    elif remainder.split("/", 1)[0] in ("purge", "purge-group", "upload"):
        errors.append(
            "an image URL may not begin with a purge or upload segment: the server answers 400 "
            "(reserved_segment)"
        )

    segments: List[ExplainedSegment] = []
    rest = remainder
    source = ""
    while True:
        idx = rest.find("/")
        seg = rest if idx == -1 else rest[:idx]
        if seg == "" and idx != -1:
            # An empty segment in the operations run (a doubled slash):
            # the server answers 400 (missing source URL) instead of
            # treating the rest as a source with a leading slash
            # (AP-23).
            error = "empty segment in the path: the server answers 400 (missing source URL)"
            segments.append(ExplainedSegment(seg, "operation", "Empty segment", error=error))
            errors.append(error)
            rest = rest[idx + 1 :]
            continue
        explained = None if seg == "" else _explain_segment(seg, now_seconds)
        if explained is None:
            source = rest
            break
        segments.append(explained)
        if explained.error:
            errors.append(explained.error)
        if idx == -1:
            errors.append("the URL ends after the operations: no source")
            rest = ""
            break
        rest = rest[idx + 1 :]
    # The caps apply to a request that parses, as on the server.
    if not errors:
        errors.extend(_cap_errors(segments))
    return Explanation(signature, segments, source, remainder, errors)
