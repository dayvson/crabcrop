"""The parts of the URL grammar the typed builders check against:
the server's caps, the closed sets of names and numbers, and the
class-name and text-label rules (issue #716). They are the values
the TypeScript SDK's ``explain.ts`` and ``limits.ts`` give its
builders, and the test suite reads them back from those files, so
the two SDKs cannot drift apart unnoticed. Each closed set is the
``Literal`` type the builders' signatures use, read back with
``get_args``, so the type and the check are one copy here.
"""

from __future__ import annotations

import unicodedata
from typing import Literal, Union, get_args

#: The server's per-axis pixel cap (``MAX_DIMENSION`` in the
#: server's ``url.rs``): the largest value a size side, a crop or
#: pad box, a region, a border, a radius, or a pixelate block can
#: name. A larger value is a parse error on the server, so the
#: builders and ``build_srcset`` check against this one constant.
MAX_DIMENSION = 16383

#: Output formats the ``format(...)`` filter accepts; ``jpg`` is an
#: input alias of ``jpeg``.
ImageFormat = Literal["jpeg", "jpg", "png", "webp", "jxl", "avif", "gif", "mp4", "webm"]

#: Formats a per-format quality entry can name, in canonical (name)
#: order: the still formats, since the video formats take no quality.
PerFormatQualityFormat = Literal["avif", "gif", "jpeg", "jxl", "png", "webp"]

#: Crop alignments; ``center`` is the bare ``crop:WxH`` spelling.
CropAlign = Literal[
    "center",
    "smart",
    "face",
    "top",
    "bottom",
    "left",
    "right",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
]

#: Text-overlay anchor points.
Anchor = Literal[
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
]

#: Overlay blend modes.
BlendMode = Literal["normal", "multiply", "screen", "overlay"]

#: Mask shapes for the ``mask(...)`` filter.
MaskShape = Literal["circle", "ellipse"]

#: Channels the ``curve(...)`` filter remaps.
CurveChannel = Literal["rgb", "r", "g", "b"]

#: Modes the ``studio(...)`` filter runs.
StudioMode = Literal["product", "scene"]

#: Contact shadow strengths for ``studio("product", ...)``.
StudioShadow = Literal["soft", "strong", "none"]

#: Axes ``flip(...)`` mirrors across.
FlipAxis = Literal["h", "v"]

#: JPEG chroma subsamplings ``format("jpeg", subsampling=...)`` takes.
JpegSubsampling = Literal[444, 422, 420]

#: Factors ``upscale(...)`` takes.
UpscaleFactor = Literal[2, 4]

#: One side of a size: a pixel count (1 to `MAX_DIMENSION`), 0 for
#: proportional, a negative count to flip that axis, or ``"orig"``
#: for the source's own dimension.
Dimension = Union[int, Literal["orig"]]

IMAGE_FORMATS = get_args(ImageFormat)
QUALITY_FORMATS = get_args(PerFormatQualityFormat)
CROP_ALIGNS = get_args(CropAlign)
TEXT_ANCHORS = get_args(Anchor)
BLEND_MODES = get_args(BlendMode)
MASK_SHAPES = get_args(MaskShape)
CURVE_CHANNELS = get_args(CurveChannel)
STUDIO_MODES = get_args(StudioMode)
STUDIO_SHADOWS = get_args(StudioShadow)
FLIP_AXES = get_args(FlipAxis)
JPEG_SUBSAMPLINGS = get_args(JpegSubsampling)
UPSCALE_FACTORS = get_args(UpscaleFactor)

#: Named backdrops for ``studio("product", bg=...)``; any RRGGBB
#: color also works.
STUDIO_BACKDROPS = ("white", "brand", "none")

#: Device pixel ratios ``dpr(...)`` takes. A float has no
#: ``Literal`` type, so this set is a plain tuple.
DPR_FACTORS = (1, 1.5, 2, 3)

#: The objects crop's cap on distinct class names (url.rs
#: MAX_OBJECTS_CLASSES).
MAX_OBJECTS_CLASSES = 32

#: The most characters a ``text`` label may hold (url.rs
#: MAX_TEXT_GLYPHS, issue #606).
MAX_TEXT_GLYPHS = 1000

#: The most a ``text`` label's character count times its size may
#: come to (url.rs MAX_TEXT_EXTENT).
MAX_TEXT_EXTENT = 100000

# The Unicode White_Space set, the set Rust's `str::trim` strips.
# Python's `str.strip()` disagrees: it also strips U+001C to
# U+001F, which the server keeps (issue #537).
_SERVER_WHITE_SPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0 "
    "           "
    "    　"
)


def trim_like_server(text: str) -> str:
    """Trim like the server's Rust ``str::trim``: exactly the Unicode
    White_Space set, from both ends. U+FEFF is not in it, so a class
    name wrapped in it is no class name, and U+0085 is."""
    start, end = 0, len(text)
    while start < end and text[start] in _SERVER_WHITE_SPACE:
        start += 1
    while end > start and text[end - 1] in _SERVER_WHITE_SPACE:
        end -= 1
    return text[start:end]


def is_class_name(name: str) -> bool:
    """A detection class name as the TypeScript SDK checks it,
    already trimmed: 1 to 64 UTF-8 bytes (bytes, not characters, as
    on the server) of letters, digits, space, ``_``, ``.``, ``-``,
    ``/``. A letter or digit is a character of a Unicode general
    category ``L`` or ``N``, the ``\\p{L}`` and ``\\p{N}`` of the
    TypeScript check; the bytes are counted last, so a lone
    surrogate fails the character test before any encoding."""
    return (
        len(name) >= 1
        and all(c in " _.-/" or unicodedata.category(c)[0] in "LN" for c in name)
        and len(name.encode("utf-8")) <= 64
    )


def class_fold_key(name: str) -> str:
    """The key a class name folds to when the objects crop drops
    repeats. The server compares names with ``eq_ignore_ascii_case``
    (url.rs parse_crop_align), so only A to Z fold: ``E`` and ``e``
    are one class, ``É`` and ``é`` are two, and the Kelvin sign is
    not ``k`` (issue #482). ``str.lower`` would fold all three."""
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in name)


def text_label_fits(label: str, size: int) -> bool:
    """Whether a ``text`` label at ``size`` is within the server's
    caps. Characters are Unicode code points, as the server counts
    them, not bytes."""
    return len(label) <= MAX_TEXT_GLYPHS and len(label) * size <= MAX_TEXT_EXTENT
