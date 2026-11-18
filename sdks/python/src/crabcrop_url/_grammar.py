"""The parts of the URL grammar the typed builders and the explainer
check against: the server's caps, the closed sets of names and
numbers, the class-name and text-label rules, and the reserved
segment (issue #716). They are the values the TypeScript SDK's
``explain.ts`` and ``limits.ts`` give its builders and its explainer,
and the test suite reads them back from those files, so the two SDKs
cannot drift apart unnoticed. Each closed set is the ``Literal`` type
the builders' signatures use, read back with ``get_args``, so the
type and the check are one copy here.
"""

from __future__ import annotations

import bisect
import unicodedata
from typing import List, Literal, Tuple, Union, get_args

#: The server's per-axis pixel cap (``MAX_DIMENSION`` in the
#: server's ``url.rs``): the largest value a size side, a crop or
#: pad box, a region, a border, a radius, or a pixelate block can
#: name. A larger value is a parse error on the server, so the
#: builders and ``build_srcset`` check against this one constant.
MAX_DIMENSION = 16383

#: The first remainder segment reserved for key derivation.
#: Every derived subkey (purge, purge-group, upload, seal, and
#: the server's cache entry key) is the HMAC of a label such as
#: ``crabcrop/purge-key`` under the secret, which is the image
#: signature of that label. No image URL may begin with this
#: segment (the server refuses one with ``400
#: reserved_segment``), and `sign` and every builder refuse a
#: remainder that does, so no image signature is ever a subkey.
#: The explainer reports such a remainder as the server answers
#: it.
RESERVED_SEGMENT = "crabcrop"


def is_reserved_remainder(remainder: str) -> bool:
    """Whether ``remainder`` begins with `RESERVED_SEGMENT`: it
    is that segment alone or starts with ``crabcrop/``. The
    test is on the text as written, as the HMAC is:
    ``exp:1/crabcrop/x`` and ``CRABCROP/x`` are not reserved."""
    return remainder.split("/", 1)[0] == RESERVED_SEGMENT


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
    "\t\n\x0b\x0c\r \x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
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


# The characters Rust's `char::is_alphanumeric` takes beyond the
# Unicode general categories L and N: the rest of the Alphabetic
# property, such as the vowel signs of Indic scripts (the `ि` and `ं`
# of `हिंदी`), the points of Hebrew and Arabic, combining marks such
# as U+0345, and circled letters such as `Ⓐ`. The server's class-name
# test is `is_alphanumeric` (url.rs parse_class_name), and Python's
# `unicodedata` has no Alphabetic property, so these are written out:
# Alphabetic minus L and N in Unicode 17.0, as hex code points and
# inclusive ranges. L and N themselves come from `unicodedata`.
_MORE_ALPHANUMERIC = """
0345 0363-036F 05B0-05BD 05BF 05C1-05C2 05C4-05C5 05C7 0610-061A
064B-0657 0659-065F 0670 06D6-06DC 06E1-06E4 06E7-06E8 06ED 0711
0730-073F 07A6-07B0 0816-0817 081B-0823 0825-0827 0829-082C 0897
08D4-08DF 08E3-08E9 08F0-0903 093A-093B 093E-094C 094E-094F 0955-0957
0962-0963 0981-0983 09BE-09C4 09C7-09C8 09CB-09CC 09D7 09E2-09E3
0A01-0A03 0A3E-0A42 0A47-0A48 0A4B-0A4C 0A51 0A70-0A71 0A75 0A81-0A83
0ABE-0AC5 0AC7-0AC9 0ACB-0ACC 0AE2-0AE3 0AFA-0AFC 0B01-0B03 0B3E-0B44
0B47-0B48 0B4B-0B4C 0B56-0B57 0B62-0B63 0B82 0BBE-0BC2 0BC6-0BC8
0BCA-0BCC 0BD7 0C00-0C04 0C3E-0C44 0C46-0C48 0C4A-0C4C 0C55-0C56
0C62-0C63 0C81-0C83 0CBE-0CC4 0CC6-0CC8 0CCA-0CCC 0CD5-0CD6 0CE2-0CE3
0CF3 0D00-0D03 0D3E-0D44 0D46-0D48 0D4A-0D4C 0D57 0D62-0D63 0D81-0D83
0DCF-0DD4 0DD6 0DD8-0DDF 0DF2-0DF3 0E31 0E34-0E3A 0E4D 0EB1 0EB4-0EB9
0EBB-0EBC 0ECD 0F71-0F83 0F8D-0F97 0F99-0FBC 102B-1036 1038 103B-103E
1056-1059 105E-1060 1062-1064 1067-106D 1071-1074 1082-108D 108F
109A-109D 1712-1713 1732-1733 1752-1753 1772-1773 17B6-17C8 1885-1886
18A9 1920-192B 1930-1938 1A17-1A1B 1A55-1A5E 1A61-1A74 1ABF-1AC0
1ACC-1ACE 1B00-1B04 1B35-1B43 1B80-1B82 1BA1-1BA9 1BAC-1BAD 1BE7-1BF1
1C24-1C36 1DD3-1DF4 24B6-24E9 2DE0-2DFF A674-A67B A69E-A69F A802 A80B
A823-A827 A880-A881 A8B4-A8C3 A8C5 A8FF A926-A92A A947-A952 A980-A983
A9B4-A9BF A9E5 AA29-AA36 AA43 AA4C-AA4D AA7B-AA7D AAB0 AAB2-AAB4
AAB7-AAB8 AABE AAEB-AAEF AAF5 ABE3-ABEA FB1E 10376-1037A 10A01-10A03
10A05-10A06 10A0C-10A0F 10D24-10D27 10D69 10EAB-10EAC 10EFA-10EFC
11000-11002 11038-11045 11073-11074 11080-11082 110B0-110B8 110C2
11100-11102 11127-11132 11145-11146 11180-11182 111B3-111BF 111CE-111CF
1122C-11234 11237 1123E 11241 112DF-112E8 11300-11303 1133E-11344
11347-11348 1134B-1134C 11357 11362-11363 113B8-113C0 113C2 113C5
113C7-113CA 113CC-113CD 11435-11441 11443-11445 114B0-114C1 115AF-115B5
115B8-115BE 115DC-115DD 11630-1163E 11640 116AB-116B5 1171D-1172A
1182C-11838 11930-11935 11937-11938 1193B-1193C 11940 11942 119D1-119D7
119DA-119DF 119E4 11A01-11A0A 11A35-11A39 11A3B-11A3E 11A51-11A5B
11A8A-11A97 11B60-11B67 11C2F-11C36 11C38-11C3E 11C92-11CA7 11CA9-11CB6
11D31-11D36 11D3A 11D3C-11D3D 11D3F-11D41 11D43 11D47 11D8A-11D8E
11D90-11D91 11D93-11D96 11EF3-11EF6 11F00-11F01 11F03 11F34-11F3A
11F3E-11F40 1611E-1612E 16F4F 16F51-16F87 16F8F-16F92 16FF0-16FF1 1BC9E
1E000-1E006 1E008-1E018 1E01B-1E021 1E023-1E024 1E026-1E02A 1E08F 1E6E3
1E6E6 1E6EE-1E6EF 1E6F5 1E947 1F130-1F149 1F150-1F169 1F170-1F189
"""


def _code_point_ranges(text: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    starts: List[int] = []
    ends: List[int] = []
    for token in text.split():
        first, _, last = token.partition("-")
        starts.append(int(first, 16))
        ends.append(int(last or first, 16))
    return tuple(starts), tuple(ends)


_MORE_STARTS, _MORE_ENDS = _code_point_ranges(_MORE_ALPHANUMERIC)


def is_alphanumeric(char: str) -> bool:
    """Whether one character is a letter or digit as the server's
    Rust ``char::is_alphanumeric`` reads it: a character of a Unicode
    general category ``L`` or ``N``, or one of the other Alphabetic
    characters."""
    if unicodedata.category(char)[0] in "LN":
        return True
    index = bisect.bisect_right(_MORE_STARTS, ord(char)) - 1
    return index >= 0 and ord(char) <= _MORE_ENDS[index]


def is_class_name(name: str) -> bool:
    """A detection class name as the server's ``parse_class_name``
    takes it, already trimmed: 1 to 64 UTF-8 bytes (bytes, not
    characters, as on the server) of letters, digits, space, ``_``,
    ``.``, ``-``, ``/``. A letter or digit is what `is_alphanumeric`
    calls one, so ``हिंदी`` is a class name, and a decomposed ``é``
    (``e`` and U+0301, a mark that is not Alphabetic) is not. The
    bytes are counted last, so a lone surrogate fails the character
    test before any encoding."""
    return (
        len(name) >= 1
        and all(c in " _.-/" or is_alphanumeric(c) for c in name)
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
