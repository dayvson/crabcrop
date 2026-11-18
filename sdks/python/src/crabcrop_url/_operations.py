"""Typed builders for crabcrop operation strings: ``ops``,
``filter`` and ``join_ops``, the Python mirror of the TypeScript
SDK's ``ops``, ``filter`` and ``joinOps`` (issue #716).

``build_url`` takes operations verbatim, by design: the string is
the spec. This module is the optional typed surface over it. Each
call validates its arguments against the server's grammar and
returns the operation segment in its canonical spelling, so a typed
pipeline and the equivalent hand-written string sign byte-identical
URLs. A value the server would refuse raises ``ValueError`` here,
at build time, as ``build_srcset`` does for widths, so it never
gets signed. Both SDKs run the builder cases of
``spec/operation-vectors.json``, which pin each call's output.
"""

from __future__ import annotations

import math
import re
import struct
from decimal import Decimal
from typing import Dict, List, Literal, NoReturn, Optional, Sequence, Set, Tuple, TypedDict, Union
from urllib.parse import quote

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
    MAX_TEXT_EXTENT,
    MAX_TEXT_GLYPHS,
    QUALITY_FORMATS,
    STUDIO_BACKDROPS,
    STUDIO_MODES,
    STUDIO_SHADOWS,
    TEXT_ANCHORS,
    UPSCALE_FACTORS,
    Anchor,
    BlendMode,
    CropAlign,
    CurveChannel,
    Dimension,
    FlipAxis,
    ImageFormat,
    JpegSubsampling,
    MaskShape,
    PerFormatQualityFormat,
    StudioMode,
    StudioShadow,
    UpscaleFactor,
    class_fold_key,
    is_class_name,
    text_label_fits,
    trim_like_server,
)


class FormatOptions(TypedDict, total=False):
    """Per-format encoder options for the ``format(...)`` filter
    (issue #319), as keyword arguments to `filter.format`."""

    #: JPEG: progressive (multi-scan) encoding.
    progressive: bool
    #: JPEG: chroma subsampling, 444, 422, or 420.
    subsampling: JpegSubsampling
    #: PNG: Adam7 interlacing.
    interlace: bool
    #: WebP: libwebp compression method, 0 to 6.
    method: int
    #: WebP: smart subsampling strength, 0 to 100.
    sns: int


class StudioOptions(TypedDict, total=False):
    """Named arguments for the ``studio(...)`` filter, as keyword
    arguments to `filter.studio`. ``bg`` and ``shadow`` are product
    arguments only: the server rejects them under ``scene``, so the
    builder raises for them there."""

    #: The product backdrop: ``"white"``, ``"brand"`` (the operator
    #: brand color, white when none is configured), ``"none"`` (the
    #: subject keeps its transparency and no shadow is synthesized),
    #: or a RRGGBB hex color. Unset means the unnamed default: the
    #: brand color when one is configured, else white. Naming
    #: ``"white"`` is not the default: the two can resolve
    #: differently, so they spell differently.
    bg: str
    #: Contact shadow strength; the default is ``"soft"``.
    shadow: StudioShadow
    #: Pad margin in percent, 0 to 40; the default is 10. Under
    #: ``scene`` it is the margin the reframed subject keeps.
    pad: int


def _fail(what: str, value: object, rule: str) -> NoReturn:
    raise ValueError(f"{what} must be {rule}, got {value!r}")


def _either_of(values: Sequence[object], quote_them: bool = False) -> str:
    """A closed set spelled for an error message: ``2 or 4``,
    ``1, 1.5, 2, or 3``."""
    words = [f'"{v}"' if quote_them else str(v) for v in values]
    last = words.pop()
    return f"{words[0]} or {last}" if len(words) == 1 else f"{', '.join(words)}, or {last}"


def _is_finite(value: Union[int, float]) -> bool:
    # An int is finite however large, and math.isfinite would
    # overflow converting a huge one to a float.
    return isinstance(value, int) or math.isfinite(value)


# A bool is an int in Python, and True would read as 1, so every
# numeric check below refuses one first.


def _int_in(what: str, value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        _fail(what, value, f"an integer from {low} to {high}")
    return value


def _float_in(what: str, value: object, low: float, high: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _is_finite(value)
        or not low <= value <= high
    ):
        _fail(what, value, f"a number from {low} to {high}")
    return value


def _f32(value: Union[int, float]) -> float:
    """``value`` rounded to the nearest 32-bit float, as the server
    parses it (JavaScript's ``Math.fround``): infinite past the
    float range, and zero below its smallest subnormal."""
    # float() first: struct reports an int too large for a double as
    # a struct.error, where float() raises the OverflowError below.
    try:
        f: float = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except OverflowError:
        return math.inf if value > 0 else -math.inf
    return f


def _f32_in(what: str, value: object, rule: str, smallest: float, high: float) -> float:
    """A float parameter the server reads as an f32: finite, at most
    ``high``, and, as the f32 it becomes, at least ``smallest`` (a
    positive double that rounds to zero there would otherwise be
    spelled ``0``, which the server refuses)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _is_finite(value)
        or _f32(value) < smallest
        or value > high
    ):
        _fail(what, value, rule)
    return value


def _positive_float(what: str, value: object) -> float:
    # The server reads a 32-bit float: past its range the value is
    # infinite there, and a positive value below it is zero.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(what, value, "a positive number")
    f32 = _f32(value)
    if not math.isfinite(f32) or f32 <= 0:
        _fail(what, value, "a positive number")
    return value


def _plain_decimal(digits: int, exponent: int) -> str:
    """Spell ``digits * 10**exponent`` in plain decimal, with no
    exponent."""
    text = str(digits)
    if exponent >= 0:
        return text + "0" * exponent
    padded = text.rjust(1 - exponent, "0")
    whole = padded[: len(padded) + exponent]
    fraction = padded[len(padded) + exponent :].rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def _num(value: Union[int, float]) -> str:
    """The canonical spelling of a 32-bit float parameter (issue
    #472). The server parses these arguments as ``f32`` and prints
    them back as Rust's float ``Display`` does: the shortest decimal
    that reads back as the same f32 (the closer of two, a tie going
    up), never an exponent. So ``0.1 + 0.2`` spells ``0.3``, the form
    the server keys its cache on. The search is exact, on the
    float's binary value in integer arithmetic, and is the TypeScript
    SDK's own, so the two SDKs spell every float alike."""
    f = _f32(value)
    if not math.isfinite(f):
        raise ValueError(f"{value} is outside the 32-bit float range the server parses")
    if f == 0:
        return "0"
    bits = struct.unpack("<I", struct.pack("<f", abs(f)))[0]
    biased = bits >> 23
    fraction = bits & 0x7FFFFF
    mantissa = fraction if biased == 0 else fraction | 0x800000
    # In units of a quarter ulp, 2**unit: the float sits at
    # 4 * mantissa and reads back from any decimal strictly between
    # the midpoints to its neighbors, which lie half as far below at
    # a power of two (the binade below is finer). A decimal exactly
    # on a midpoint rounds to the even mantissa, so the ends count
    # only for an even one.
    unit = (1 if biased == 0 else biased) - 152
    center = 4 * mantissa
    low = center - 1 if fraction == 0 and biased > 1 else center - 2
    high = center + 2
    inclusive = mantissa % 2 == 0
    sign = "-" if f < 0 else ""
    # From one decade above the value down, the first decimal
    # exponent with a candidate in the interval gives the shortest
    # spelling. Only the two neighbors of the value at that exponent
    # can qualify, and an f32 needs at most nine significant digits,
    # so the loop is short.
    k = math.floor(math.log10(abs(f))) + 2
    while True:
        # Compare digits * 10**k with units * 2**unit as integers.
        binary = (1 << unit if unit > 0 else 1) * (10 ** -k if k < 0 else 1)
        step = (1 << -unit if unit < 0 else 1) * (10**k if k > 0 else 1)
        at = center * binary
        below = at // step * step
        above = below + step
        low_end = low * binary
        high_end = high * binary
        down_fits = below >= low_end if inclusive else below > low_end
        up_fits = above <= high_end if inclusive else above < high_end
        if down_fits or up_fits:
            down = down_fits and (not up_fits or 2 * (at - below) < step)
            return sign + _plain_decimal((below if down else above) // step, k)
        k -= 1


def _js_number_string(value: float) -> str:
    """JavaScript's ``String(value)`` for a finite double: the
    shortest decimal that reads back as ``value`` (the digits
    ``repr`` gives), laid out by ECMAScript's Number::toString."""
    if value == 0:
        return "0"
    if value < 0:
        return "-" + _js_number_string(-value)
    parts = Decimal(repr(value)).normalize().as_tuple()
    digits = "".join(map(str, parts.digits))
    k = len(digits)
    n = int(parts.exponent) + k
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return f"{digits[:n]}.{digits[n:]}"
    if -6 < n <= 0:
        return "0." + "0" * -n + digits
    mantissa = digits if k == 1 else f"{digits[0]}.{digits[1:]}"
    return f"{mantissa}e{'+' if n - 1 >= 0 else '-'}{abs(n - 1)}"


def _kernel_num(value: Union[int, float]) -> str:
    """The canonical spelling of a 64-bit float parameter (the
    convolution kernel's values), as the TypeScript SDK writes it:
    integers without a fraction, other values as JavaScript's
    shortest round-trip spelling, which agrees with the server's for
    the magnitudes the grammar allows, and an exponent spelling
    (below 1e-6) written out to twelve places."""
    if isinstance(value, int) or (value.is_integer() and abs(value) <= 2**53 - 1):
        return str(int(value))
    text = _js_number_string(value)
    if "e" not in text:
        return text
    return f"{value:.12f}".rstrip("0").rstrip(".")


_PLAIN_DECIMAL = re.compile(r"-?\d+(\.\d+)?", re.ASCII)
_HEX6 = re.compile(r"[0-9a-fA-F]{6}")
_HEX8 = re.compile(r"[0-9a-fA-F]{8}")
_PRESET_NAME = re.compile(r"[A-Za-z0-9_-]+")
_INT32 = (-(2**31), 2**31 - 1)
_UINT32_MAX = 2**32 - 1


#: The smallest positive 32-bit float: a value "above 0" is at least
#: this once the server reads it as an f32.
_F32_TINY = 2.0**-149


def _dimension(what: str, value: object) -> str:
    if value == "orig":
        return "orig"
    if isinstance(value, bool) or not isinstance(value, int) or abs(value) > MAX_DIMENSION:
        _fail(what, value, f'a pixel count up to {MAX_DIMENSION}, 0, or "orig"')
    return str(value)


def _pixel_box(what: str, width: object, height: object) -> str:
    _int_in(f"{what} width", width, 1, MAX_DIMENSION)
    _int_in(f"{what} height", height, 1, MAX_DIMENSION)
    return f"{width}x{height}"


def _hex6(what: str, value: object) -> str:
    if not isinstance(value, str) or not _HEX6.fullmatch(value):
        _fail(what, value, "six hex digits (RRGGBB)")
    return value.lower()


def _hex6or8(what: str, value: object) -> str:
    if isinstance(value, str) and _HEX6.fullmatch(value):
        return f"{value.lower()}ff"
    if isinstance(value, str) and _HEX8.fullmatch(value):
        return value.lower()
    _fail(what, value, "six or eight hex digits (RRGGBB or RRGGBBAA)")


def _encode_arg(what: str, text: str) -> str:
    """The canonical percent-encoding for filter arguments that carry
    text (watermark and overlay URLs, class names, text labels):
    everything outside the RFC 3986 unreserved set is encoded,
    byte-wise, exactly as the server's canonical form spells it."""
    try:
        return quote(text, safe="")
    except UnicodeEncodeError:
        raise ValueError(f"{what} is not valid Unicode text (it holds a lone surrogate)") from None


def _class_name(what: str, name: object) -> str:
    rule = "1 to 64 UTF-8 bytes of letters, digits, space, `_`, `.`, `-`, `/`"
    if not isinstance(name, str):
        _fail(what, name, rule)
    trimmed = trim_like_server(name)
    if not is_class_name(trimmed):
        _fail(what, name, rule)
    # The server trims the name before it validates and keys on it,
    # so the trimmed name is the canonical one.
    return _encode_arg(what, trimmed)


def _region(x: int, y: int, w: int, h: int) -> str:
    _int_in("region x", x, 0, MAX_DIMENSION)
    _int_in("region y", y, 0, MAX_DIMENSION)
    _int_in("region width", w, 1, MAX_DIMENSION)
    _int_in("region height", h, 1, MAX_DIMENSION)
    return f"{x},{y},{w},{h}"


def _check_segments(what: str, parts: Sequence[object]) -> List[str]:
    checked: List[str] = []
    for part in parts:
        if not isinstance(part, str):
            _fail(what, part, "a string")
        checked.append(part)
    return checked


class _Ops:
    """Operation segments, one typed call per segment kind
    (``ops`` in the TypeScript SDK). Each returns the segment's
    canonical spelling; `join_ops` joins them."""

    @staticmethod
    def size(width: Dimension, height: Dimension) -> str:
        """``300x200``; each side a `Dimension` (0 proportional,
        negative flips)."""
        return f"{_dimension('width', width)}x{_dimension('height', height)}"

    @staticmethod
    def fit_in(width: Dimension, height: Dimension, adaptive: bool = False) -> str:
        """``fit-in:300x200``; no flips (the server rejects them
        here). With ``adaptive``, ``fit-in:300x200:adaptive``: the
        box axes swap when the source orientation disagrees with the
        box's."""
        for what, value in (("width", width), ("height", height)):
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
                _fail(f"fit-in {what}", value, "not a flip: fit-in takes no negative sizes")
        base = f"fit-in:{_dimension('width', width)}x{_dimension('height', height)}"
        return f"{base}:adaptive" if adaptive else base

    @staticmethod
    def crop(width: int, height: int, align: CropAlign = "center") -> str:
        """``crop:300x200`` or ``crop:300x200:smart``; an exact pixel
        box."""
        if align not in CROP_ALIGNS:
            _fail("crop align", align, f"one of {', '.join(CROP_ALIGNS)}")
        box = _pixel_box("crop", width, height)
        return f"crop:{box}" if align == "center" else f"crop:{box}:{align}"

    @staticmethod
    def crop_objects(width: int, height: int, classes: Sequence[str], scale: float = 1) -> str:
        """``crop:300x200:objects(person,car):1.5``; crops to the
        union of every detection of the named classes, the union box
        scaled about its center by ``scale`` (1 is the tight box; the
        crop stays centered on the union). Needs a detection model
        configured on the server; when none of the classes is
        detected the render falls back to the plain center crop of
        the same size. Duplicate classes (ASCII case-insensitive, as
        the server compares them: ``E`` repeats ``e``, ``É`` does not
        repeat ``é``) are spelled once, as the server's canonical form
        spells them; at most 32 distinct classes, and a 33rd raises,
        since the server rejects it with ``invalid_crop``."""
        rule = "a list of 1 to 32 distinct class names"
        if not isinstance(classes, (list, tuple)) or len(classes) < 1:
            _fail("crop objects classes", classes, rule)
        # The cap counts distinct names, as the server's parser does
        # (issue #408): repeats fold first and cost nothing.
        seen: Set[str] = set()
        names: List[str] = []
        for name in classes:
            encoded = _class_name("crop objects class", name)
            key = class_fold_key(trim_like_server(name))
            if key not in seen:
                if len(names) == MAX_OBJECTS_CLASSES:
                    _fail("crop objects classes", classes, rule)
                seen.add(key)
                names.append(encoded)
        # Checked as the f32 the server reads (as are zoom, blur and
        # gamma): a positive double that rounds to f32 zero would
        # otherwise be spelled `0`, which it refuses.
        _f32_in("crop objects scale", scale, "a number above 0 and at most 100", _F32_TINY, 100)
        box = _pixel_box("crop", width, height)
        return f"crop:{box}:objects({','.join(names)}):{_num(scale)}"

    @staticmethod
    def crop_rect(left: int, top: int, right: int, bottom: int) -> str:
        """``crop-rect:10x20:110x220`` (left x top : right x bottom)."""
        for what, value in (("left", left), ("top", top), ("right", right), ("bottom", bottom)):
            _int_in(f"crop-rect {what}", value, 0, _UINT32_MAX)
        if right <= left or bottom <= top:
            _fail("crop-rect", f"{left}x{top}:{right}x{bottom}", "right > left and bottom > top")
        return f"crop-rect:{left}x{top}:{right}x{bottom}"

    @staticmethod
    def focal(left: int, top: int, right: int, bottom: int) -> str:
        """``focal:10x20:110x220`` (left x top : right x bottom);
        steers the next crop."""
        for what, value in (("left", left), ("top", top), ("right", right), ("bottom", bottom)):
            _int_in(f"focal {what}", value, 0, _UINT32_MAX)
        if right <= left or bottom <= top:
            _fail("focal", f"{left}x{top}:{right}x{bottom}", "right > left and bottom > top")
        return f"focal:{left}x{top}:{right}x{bottom}"

    @staticmethod
    def focal_point(x: float, y: float, zoom: float = 1) -> str:
        """``focalpoint:0.75x0.4:2`` (fractions of the image
        dimensions, zoom optional); steers the next crop."""

        def fraction(what: str, value: object) -> float:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not _is_finite(value)
                or not 0 <= value <= 1
            ):
                _fail(what, value, "a fraction from 0 to 1")
            return value

        _f32_in("focalpoint zoom", zoom, "a number above 0 and at most 100", _F32_TINY, 100)
        return (
            f"focalpoint:{_num(fraction('focalpoint x', x))}"
            f"x{_num(fraction('focalpoint y', y))}:{_num(zoom)}"
        )

    @staticmethod
    def pad(width: int, height: int, fill: Optional[str] = None) -> str:
        """``pad:300x200``, optionally ``:blur`` or a RRGGBB fill
        color."""
        box = _pixel_box("pad", width, height)
        if fill is None:
            return f"pad:{box}"
        if fill == "blur":
            return f"pad:{box}:blur"
        return f"pad:{box}:{_hex6('pad fill', fill)}"

    @staticmethod
    def preset(name: str) -> str:
        """``preset:thumb``; expands server-side."""
        # The server can only configure names from this charset (its
        # preset file validation), so anything outside it is a
        # reference that can only ever answer unknown_preset.
        if not isinstance(name, str) or not _PRESET_NAME.fullmatch(name):
            _fail("preset name", name, "letters, digits, `_`, `-` (the server's preset name charset)")
        return f"preset:{name}"

    @staticmethod
    def maxage(seconds: int) -> str:
        """``maxage:60``; this response's Cache-Control lifetime in
        seconds (0 to 31536000), overriding the default year for this
        request only. Header-only: it never enters the cache key, so
        spellings that differ only here share one stored image."""
        return f"maxage:{_int_in('maxage seconds', seconds, 0, 31536000)}"

    @staticmethod
    def debug_focal() -> str:
        """``debug:focal``; the smart-crop debug view. Place it before
        the crop it annotates: the response is a diagnostic image of
        the focal decision, never the transform result, and it is
        never cached (served with Cache-Control: no-store)."""
        return "debug:focal"

    @staticmethod
    def filters(*calls: str) -> str:
        """Join filter calls into one ``filters:`` segment."""
        if not calls:
            _fail("filters", calls, "at least one filter call")
        return "filters:" + ":".join(_check_segments("filter call", calls))


def join_ops(*segments: str) -> str:
    """Join operation segments into the operations string
    `build_url` takes (``joinOps`` in the TypeScript SDK)."""
    return "/".join(_check_segments("operation segment", segments))


class _Filter:
    """Filter calls for `ops.filters`, one per filter (``filter`` in
    the TypeScript SDK); each returns its canonical spelling."""

    @staticmethod
    def grayscale() -> str:
        return "grayscale"

    @staticmethod
    def invert() -> str:
        return "invert"

    @staticmethod
    def no_upscale() -> str:
        return "no_upscale"

    @staticmethod
    def equalize() -> str:
        return "equalize"

    @staticmethod
    def enhance() -> str:
        return "enhance"

    @staticmethod
    def remove_bg() -> str:
        return "removebg"

    @staticmethod
    def keep_profile() -> str:
        return "keep_profile"

    @staticmethod
    def sepia() -> str:
        return "sepia"

    @staticmethod
    def blur_faces() -> str:
        return "blurfaces"

    @staticmethod
    def pixelate_faces() -> str:
        return "pixelatefaces"

    @staticmethod
    def blur(sigma: float) -> str:
        """Gaussian blur by sigma: positive, at most the server's 512
        cap. The server also refuses a subnormal f32 sigma."""
        _f32_in("blur sigma", sigma, "a positive number up to 512", 2.0**-126, 512)
        return f"blur({_num(sigma)})"

    @staticmethod
    def sharpen(amount: float) -> str:
        """Unsharp-mask strength (positive)."""
        return f"sharpen({_num(_positive_float('sharpen amount', amount))})"

    @staticmethod
    def brighten(amount: int) -> str:
        """Add to every channel: -255 to 255."""
        return f"brighten({_int_in('brighten amount', amount, -255, 255)})"

    @staticmethod
    def contrast(percent: float) -> str:
        """Percent contrast: -100 to 100 (0 unchanged, -100 flat)."""
        return f"contrast({_num(_float_in('contrast', percent, -100, 100))})"

    @staticmethod
    def gamma(value: float) -> str:
        """Gamma correction divisor: above 0 and at most 10 (1
        unchanged; above 1 brightens midtones)."""
        _f32_in("gamma", value, "a number above 0 and at most 10", _F32_TINY, 10)
        return f"gamma({_num(value)})"

    @staticmethod
    def exposure(stops: float) -> str:
        """Exposure in stops: -10 to 10 (0 unchanged; +1 doubles the
        light, -1 halves it)."""
        if (
            isinstance(stops, bool)
            or not isinstance(stops, (int, float))
            or not _is_finite(stops)
            or not -10 <= stops <= 10
        ):
            _fail("exposure", stops, "a number between -10 and 10")
        return f"exposure({_num(stops)})"

    @staticmethod
    def highlights_shadows(highlights: float, shadows: float) -> str:
        """Highlights and shadows: two percents, each -100 to 100 (0
        unchanged)."""
        h = _num(_float_in("highlights_shadows highlights", highlights, -100, 100))
        s = _num(_float_in("highlights_shadows shadows", shadows, -100, 100))
        return f"highlights_shadows({h},{s})"

    @staticmethod
    def vibrance(percent: float) -> str:
        """Vibrance percent: -100 to 100 (0 unchanged; muted colors
        move most, vivid colors are protected)."""
        return f"vibrance({_num(_float_in('vibrance', percent, -100, 100))})"

    @staticmethod
    def hue_rotate(degrees: int) -> str:
        """Rotate the hue by integer degrees."""
        return f"huerotate({_int_in('huerotate degrees', degrees, *_INT32)})"

    @staticmethod
    def saturation(factor: float) -> str:
        """Saturation factor: 0 to 255 (0 grayscale, 1 identity)."""
        return f"saturation({_num(_float_in('saturation', factor, 0, 255))})"

    @staticmethod
    def rgb(r: int, g: int, b: int) -> str:
        """Per-channel offsets, each -255 to 255."""
        red = _int_in("rgb red", r, -255, 255)
        green = _int_in("rgb green", g, -255, 255)
        blue = _int_in("rgb blue", b, -255, 255)
        return f"rgb({red},{green},{blue})"

    @staticmethod
    def curve(channel: CurveChannel, points: Sequence[Tuple[int, int]]) -> str:
        """Tone curve: remap the channel through 2 to 16 ``(x, y)``
        control points, each coordinate 0 to 255 with x strictly
        increasing. Between points the server interpolates linearly;
        inputs outside the point range clamp to the nearest end
        value."""
        if channel not in CURVE_CHANNELS:
            _fail("curve channel", channel, f"one of {', '.join(CURVE_CHANNELS)}")
        if not isinstance(points, (list, tuple)) or not 2 <= len(points) <= 16:
            _fail("curve points", points, "2 to 16 control points")
        flat: List[int] = []
        prev_x = -1
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                _fail("curve point", point, "an (x, y) pair")
            x, y = point
            _int_in("curve point x", x, 0, 255)
            _int_in("curve point y", y, 0, 255)
            if x <= prev_x:
                _fail("curve points", points, "control points with strictly increasing x")
            prev_x = x
            flat += [x, y]
        return f"curve({channel},{','.join(map(str, flat))})"

    @staticmethod
    def convolution(values: Sequence[float], normalize: Union[bool, int] = False) -> str:
        """Convolution: apply a caller-supplied kernel, 9 values (3x3)
        or 25 (5x5) in row-major order, each a plain decimal from
        -1024 to 1024. With ``normalize`` set the server scales the
        kernel by the sum of its entries first (a zero sum applies the
        kernel as written). The edges replicate. ``normalize`` is a
        boolean or the 0/1 flag itself."""
        if not isinstance(values, (list, tuple)) or len(values) not in (9, 25):
            _fail("convolution kernel", values, "9 (3x3) or 25 (5x5) values")
        for value in values:
            _float_in("convolution kernel value", value, -1024, 1024)
            if not _PLAIN_DECIMAL.fullmatch(_kernel_num(value)):
                _fail(
                    "convolution kernel value",
                    value,
                    "a plain decimal (the grammar takes no exponent forms)",
                )
        # 0 and 1 compare equal to False and True, so this is the
        # TypeScript check's set: a boolean or the 0/1 flag.
        if normalize not in (True, False):
            _fail("convolution normalize", normalize, "a boolean or 0/1")
        flag = 1 if normalize else 0
        return f"convolution({','.join(_kernel_num(v) for v in values)},{flag})"

    @staticmethod
    def noise(amount: int) -> str:
        """Film grain: one noise value per pixel, added equally to all
        three channels (alpha untouched), the value uniform in plus
        or minus the given percent of full scale. The field is
        deterministic, seeded from the image content and the amount,
        so the same request always renders the same bytes; 0 is a
        no-op."""
        return f"noise({_int_in('noise amount', amount, 0, 100)})"

    @staticmethod
    def rotate(angle: float) -> str:
        """Rotate by an angle in degrees (decimals allowed, no exponent
        spelling); multiples of 90 are exact quarter turns on the
        server."""
        _float_in("rotate angle", angle, -36000, 36000)
        if not _PLAIN_DECIMAL.fullmatch(_num(angle)):
            _fail("rotate angle", angle, "a plain decimal angle (the grammar takes no exponent forms)")
        return f"rotate({_num(angle)})"

    @staticmethod
    def trim(tolerance: int = 10) -> str:
        """Trim uniform borders; tolerance 0 to 255, default 10."""
        return f"trim({_int_in('trim tolerance', tolerance, 0, 255)})"

    @staticmethod
    def trim_sides(tolerance: int, left: int, top: int, right: int, bottom: int) -> str:
        """Trim uniform borders, then remove pixels per side (left,
        top, right, bottom)."""
        t = _int_in("trim tolerance", tolerance, 0, 255)
        sides = [
            _int_in(f"trim {what}", value, 0, _UINT32_MAX)
            for what, value in (("left", left), ("top", top), ("right", right), ("bottom", bottom))
        ]
        return f"trim({t},{','.join(map(str, sides))})"

    @staticmethod
    def quality(value: Union[int, Literal["auto"]]) -> str:
        """Encoding quality 1 to 100, or ``"auto"`` for perceptual
        targeting."""
        if value == "auto":
            return "quality(auto)"
        return f"quality({_int_in('quality', value, 1, 100)})"

    @staticmethod
    def quality_per_format(
        entries: Dict[PerFormatQualityFormat, Union[int, Literal["auto"]]],
        fallback: Union[int, Literal["auto"], None] = None,
    ) -> str:
        """Per-format quality in one URL: a dict of format to value (1
        to 100 or ``"auto"``), plus an optional fallback for formats
        with no entry. Resolved by the server after format
        negotiation; entries spell in canonical (format-name) order
        with the fallback first."""
        if not isinstance(entries, dict):
            _fail("quality", entries, "a dict of format names to qualities")
        if not entries and fallback is None:
            _fail("quality", entries, "at least one entry or a fallback")
        parts: List[str] = []
        if fallback is not None:
            parts.append("auto" if fallback == "auto" else str(_int_in("quality", fallback, 1, 100)))
        for name in entries:
            if not isinstance(name, str):
                _fail("quality", name, f"one of {', '.join(QUALITY_FORMATS)}")
        for name in sorted(entries):
            if name not in QUALITY_FORMATS:
                _fail("quality", name, f"one of {', '.join(QUALITY_FORMATS)}")
            value = entries[name]
            parts.append(f"{name}:{'auto' if value == 'auto' else _int_in('quality', value, 1, 100)}")
        return f"quality({','.join(parts)})"

    @staticmethod
    def palette(colors: int) -> str:
        """Indexed PNG output with at most this many colors, 2 to 256;
        other formats ignore it."""
        return f"palette({_int_in('palette colors', colors, 2, 256)})"

    @staticmethod
    def format(
        fmt: ImageFormat,
        *,
        progressive: Optional[bool] = None,
        subsampling: Optional[JpegSubsampling] = None,
        interlace: Optional[bool] = None,
        method: Optional[int] = None,
        sns: Optional[int] = None,
    ) -> str:
        """Output format, with its encoder options (issue #319):
        ``progressive`` and ``subsampling`` (444, 422, or 420) for
        JPEG, ``interlace`` for PNG, ``method`` (0 to 6) and ``sns``
        (smart subsampling, 0 to 100) for WebP. An option its format
        cannot honour raises here, like the server's 400."""
        if fmt not in IMAGE_FORMATS:
            _fail("format", fmt, f"one of {', '.join(IMAGE_FORMATS)}")
        options = {
            name: value
            for name, value in (
                ("progressive", progressive),
                ("subsampling", subsampling),
                ("interlace", interlace),
                ("method", method),
                ("sns", sns),
            )
            if value is not None
        }
        # The output segment is the canonical spelling: the `jpg`
        # input alias spells as `jpeg`, like the server's canonical
        # form.
        args: List[str] = ["jpeg" if fmt == "jpg" else fmt]
        if fmt in ("jpeg", "jpg"):
            if options.keys() & {"interlace", "method", "sns"}:
                _fail("format", options, "jpeg takes progressive and subsampling only")
            if progressive:
                args.append("progressive")
            if subsampling is not None:
                if (
                    isinstance(subsampling, bool)
                    or not isinstance(subsampling, int)
                    or subsampling not in JPEG_SUBSAMPLINGS
                ):
                    _fail("subsampling", subsampling, f"one of {', '.join(map(str, JPEG_SUBSAMPLINGS))}")
                args.append(f"subsampling:{subsampling}")
        elif fmt == "png":
            if options.keys() & {"progressive", "subsampling", "method", "sns"}:
                _fail("format", options, "png takes interlace only")
            if interlace:
                args.append("interlace")
        elif fmt == "webp":
            if options.keys() & {"progressive", "subsampling", "interlace"}:
                _fail("format", options, "webp takes method and sns only")
            if method is not None:
                args.append(f"method:{_int_in('method', method, 0, 6)}")
            if sns is not None:
                args.append(f"sns:{_int_in('sns', sns, 0, 100)}")
        elif options:
            _fail("format", options, f"{fmt} takes no encoder options")
        return f"format({','.join(args)})"

    @staticmethod
    def fill(color: str) -> str:
        """The background color other operations pad with (RRGGBB)."""
        return f"fill({_hex6('fill color', color)})"

    @staticmethod
    def face_pad(amount: float) -> str:
        """How much a face crop expands around the face: 0 to 3."""
        return f"facepad({_num(_float_in('facepad', amount, 0, 3))})"

    @staticmethod
    def studio(
        mode: StudioMode,
        *,
        bg: Optional[str] = None,
        shadow: Optional[StudioShadow] = None,
        pad: Optional[int] = None,
    ) -> str:
        """Studio mode: ``studio(product)`` composites the subject the
        server's segmentation model finds onto a backdrop with a
        synthesized contact shadow, scaled and centered to the pad
        margin; ``studio(scene)`` keeps the photograph's scene,
        reframing it on the detected subject and correcting light and
        color. See `StudioOptions` for the named arguments. Arguments
        spelled at their defaults (shadow ``soft``, pad 10) are
        omitted, matching the server's canonical spelling, so
        ``studio("product", pad=10)`` and ``studio("product")`` are
        the same string."""
        if mode not in STUDIO_MODES:
            _fail("studio mode", mode, f"one of {', '.join(STUDIO_MODES)}")
        if mode == "scene" and (bg is not None or shadow is not None):
            options = {name: value for name, value in (("bg", bg), ("shadow", shadow), ("pad", pad)) if value is not None}
            _fail("studio scene options", options, "pad only: bg and shadow are product arguments")
        parts: List[str] = [mode]
        if mode == "product":
            if bg is not None:
                backdrop = bg if bg in STUDIO_BACKDROPS else _hex6("studio bg", bg)
                parts.append(f"bg:{backdrop}")
            if shadow is not None:
                if shadow not in STUDIO_SHADOWS:
                    _fail("studio shadow", shadow, f"one of {', '.join(STUDIO_SHADOWS)}")
                if shadow != "soft":
                    parts.append(f"shadow:{shadow}")
        if pad is not None:
            _int_in("studio pad", pad, 0, 40)
            if pad != 10:
                parts.append(f"pad:{pad}")
        return f"studio({','.join(parts)})"

    @staticmethod
    def upscale(factor: UpscaleFactor) -> str:
        """AI upscale by the given factor."""
        if isinstance(factor, bool) or not isinstance(factor, int) or factor not in UPSCALE_FACTORS:
            _fail("upscale factor", factor, _either_of(UPSCALE_FACTORS))
        return f"upscale({factor})"

    @staticmethod
    def dpr(factor: float) -> str:
        """Device pixel ratio the sizes are multiplied by: 1, 1.5, 2,
        or 3."""
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor not in DPR_FACTORS:
            _fail("dpr", factor, _either_of(DPR_FACTORS))
        return f"dpr({_num(factor)})"

    @staticmethod
    def max_bytes(budget: int) -> str:
        """Cap the output bytes, lowering quality to fit: at least 1,
        and at most the server's unsigned 64-bit cap."""
        if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 2**64 - 1:
            _fail("max_bytes", budget, "a positive integer byte budget")
        return f"max_bytes({budget})"

    @staticmethod
    def radius(value: Union[int, Sequence[int]]) -> str:
        """Round the corners: one radius for all, or four in the
        order top-left, top-right, bottom-right, bottom-left."""
        if isinstance(value, (list, tuple)):
            if len(value) != 4:
                _fail("radius", value, "one corner radius or four")
            corners = list(value)
        else:
            corners = [value] * 4
        for corner in corners:
            _int_in("radius", corner, 0, MAX_DIMENSION)
        return f"radius({','.join(map(str, corners))})"

    @staticmethod
    def border(width: int, color: str) -> str:
        """Paint an opaque border inside the frame (RRGGBB); a width of
        0 is a no-op."""
        return f"border({_int_in('border width', width, 0, MAX_DIMENSION)},{_hex6('border color', color)})"

    @staticmethod
    def mask(shape: MaskShape) -> str:
        """Cut the image to a shape, transparent outside it."""
        if shape not in MASK_SHAPES:
            _fail("mask shape", shape, _either_of(MASK_SHAPES, True))
        return f"mask({shape})"

    @staticmethod
    def duotone(shadow: str, highlight: str) -> str:
        """Map the shadows and highlights to two RRGGBB colors."""
        return f"duotone({_hex6('duotone shadow', shadow)},{_hex6('duotone highlight', highlight)})"

    @staticmethod
    def monochrome(color: str) -> str:
        """Tint the image in one RRGGBB color."""
        return f"monochrome({_hex6('monochrome color', color)})"

    @staticmethod
    def pixelate(block_size: int) -> str:
        """Mosaic block size in pixels: 1 to `MAX_DIMENSION`."""
        return f"pixelate({_int_in('pixelate size', block_size, 1, MAX_DIMENSION)})"

    @staticmethod
    def flip(axis: FlipAxis) -> str:
        """Mirror across the horizontal (``h``) or vertical (``v``)
        axis."""
        if axis not in FLIP_AXES:
            _fail("flip axis", axis, _either_of(FLIP_AXES, True))
        return f"flip({axis})"

    @staticmethod
    def watermark(url: str, x: int, y: int, alpha: int) -> str:
        """Overlay a watermark image at (x, y), opacity 0 to 100."""
        _int_in("watermark x", x, *_INT32)
        _int_in("watermark y", y, *_INT32)
        _int_in("watermark alpha", alpha, 0, 100)
        if not isinstance(url, str) or url == "":
            _fail("watermark url", url, "a non-empty URL")
        return f"watermark({_encode_arg('watermark url', url)},{x},{y},{alpha})"

    @staticmethod
    def overlay(
        url: str,
        x: int,
        y: int,
        alpha: int,
        blend: BlendMode,
        width_pct: Optional[int] = None,
    ) -> str:
        """Composite another image over the result at (x, y) with
        opacity 0 to 100 and a blend mode; ``width_pct`` optionally
        scales it to that percent of the base width (1 to 400)."""
        _int_in("overlay x", x, *_INT32)
        _int_in("overlay y", y, *_INT32)
        _int_in("overlay alpha", alpha, 0, 100)
        if blend not in BLEND_MODES:
            _fail("overlay blend", blend, f"one of {', '.join(BLEND_MODES)}")
        if not isinstance(url, str) or url == "":
            _fail("overlay url", url, "a non-empty URL")
        width = "" if width_pct is None else f",{_int_in('overlay width', width_pct, 1, 400)}"
        return f"overlay({_encode_arg('overlay url', url)},{x},{y},{alpha},{blend}{width})"

    @staticmethod
    def text(content: str, size: int, color: str, anchor: Anchor) -> str:
        """Draw a text overlay. Color is RRGGBB or RRGGBBAA; the
        canonical spelling is always eight lowercase hex digits. The
        label holds at most 1000 characters, and characters times
        size is at most 100000 (the server's caps)."""
        if anchor not in TEXT_ANCHORS:
            _fail("text anchor", anchor, f"one of {', '.join(TEXT_ANCHORS)}")
        if not isinstance(content, str) or content == "":
            _fail("text", content, "a non-empty label")
        _int_in("text size", size, 1, 1000)
        if not text_label_fits(content, size):
            _fail(
                "text",
                content,
                f"at most {MAX_TEXT_GLYPHS} characters, and characters times size at most {MAX_TEXT_EXTENT}",
            )
        return f"text({_encode_arg('text', content)},{size},{_hex6or8('text color', color)},{anchor})"

    @staticmethod
    def blur_class(name: str) -> str:
        """Blur every detected object of the named class."""
        return f"blurclass({_class_name('blurclass name', name)})"

    @staticmethod
    def pixelate_class(name: str) -> str:
        """Pixelate every detected object of the named class."""
        return f"pixelateclass({_class_name('pixelateclass name', name)})"

    @staticmethod
    def recolor_class(name: str, color: str) -> str:
        """Recolor every detected object of the named class to the
        given RRGGBB color."""
        return f"recolorclass({_class_name('recolorclass name', name)},{_hex6('recolorclass color', color)})"

    @staticmethod
    def blur_region(x: int, y: int, w: int, h: int) -> str:
        """Blur the rectangle (x, y, w, h) in pixels."""
        return f"blur_region({_region(x, y, w, h)})"

    @staticmethod
    def pixelate_region(x: int, y: int, w: int, h: int) -> str:
        """Pixelate the rectangle (x, y, w, h) in pixels."""
        return f"pixelate_region({_region(x, y, w, h)})"

    @staticmethod
    def remove_region(x: int, y: int, w: int, h: int) -> str:
        """Erase the rectangle (x, y, w, h) in pixels and inpaint it
        (AI)."""
        return f"remove_region({_region(x, y, w, h)})"


#: Operation segments: ``ops.size(300, 200)`` is ``300x200``.
ops = _Ops()

#: Filter calls for `ops.filters`: ``filter.blur(2)`` is ``blur(2)``.
#: Imported by name it shadows the builtin ``filter`` in the
#: importing module; ``import crabcrop_url`` and
#: ``crabcrop_url.filter`` keep both.
filter = _Filter()
