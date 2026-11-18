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


# The letters and digits of a class name: what the server's Rust
# `char::is_alphanumeric` takes (url.rs parse_class_name), the Unicode
# Alphabetic property and the number categories, in Unicode 17.0. The
# set holds more than the letter categories: the vowel signs of Indic
# scripts (the `\u093f` and `\u0902` of `\u0939\u093f\u0902\u0926\u0940`), the points of Hebrew and
# Arabic, combining marks such as U+0345, and circled letters such as
# `\u24b6`. Rust 1.97's `char::is_alphanumeric` and Node 22.23's
# `[\p{Alphabetic}\p{N}]`, the TypeScript explainer's test, give this
# same set. It is written out whole, as hex code points and inclusive
# ranges: Python's `unicodedata` has no Alphabetic property, and its
# general categories are those of the interpreter's own Unicode
# version (13.0 on Python 3.9, 16.0 on 3.14), which miss the letters
# and digits added since.
_ALPHANUMERIC_UNICODE = (17, 0)
_ALPHANUMERIC = """
0030-0039 0041-005A 0061-007A 00AA 00B2-00B3 00B5 00B9-00BA 00BC-00BE
00C0-00D6 00D8-00F6 00F8-02C1 02C6-02D1 02E0-02E4 02EC 02EE 0345
0363-0374 0376-0377 037A-037D 037F 0386 0388-038A 038C 038E-03A1
03A3-03F5 03F7-0481 048A-052F 0531-0556 0559 0560-0588 05B0-05BD 05BF
05C1-05C2 05C4-05C5 05C7 05D0-05EA 05EF-05F2 0610-061A 0620-0657
0659-0669 066E-06D3 06D5-06DC 06E1-06E8 06ED-06FC 06FF 0710-073F
074D-07B1 07C0-07EA 07F4-07F5 07FA 0800-0817 081A-082C 0840-0858
0860-086A 0870-0887 0889-088F 0897 08A0-08C9 08D4-08DF 08E3-08E9
08F0-093B 093D-094C 094E-0950 0955-0963 0966-096F 0971-0983 0985-098C
098F-0990 0993-09A8 09AA-09B0 09B2 09B6-09B9 09BD-09C4 09C7-09C8
09CB-09CC 09CE 09D7 09DC-09DD 09DF-09E3 09E6-09F1 09F4-09F9 09FC
0A01-0A03 0A05-0A0A 0A0F-0A10 0A13-0A28 0A2A-0A30 0A32-0A33 0A35-0A36
0A38-0A39 0A3E-0A42 0A47-0A48 0A4B-0A4C 0A51 0A59-0A5C 0A5E 0A66-0A75
0A81-0A83 0A85-0A8D 0A8F-0A91 0A93-0AA8 0AAA-0AB0 0AB2-0AB3 0AB5-0AB9
0ABD-0AC5 0AC7-0AC9 0ACB-0ACC 0AD0 0AE0-0AE3 0AE6-0AEF 0AF9-0AFC
0B01-0B03 0B05-0B0C 0B0F-0B10 0B13-0B28 0B2A-0B30 0B32-0B33 0B35-0B39
0B3D-0B44 0B47-0B48 0B4B-0B4C 0B56-0B57 0B5C-0B5D 0B5F-0B63 0B66-0B6F
0B71-0B77 0B82-0B83 0B85-0B8A 0B8E-0B90 0B92-0B95 0B99-0B9A 0B9C
0B9E-0B9F 0BA3-0BA4 0BA8-0BAA 0BAE-0BB9 0BBE-0BC2 0BC6-0BC8 0BCA-0BCC
0BD0 0BD7 0BE6-0BF2 0C00-0C0C 0C0E-0C10 0C12-0C28 0C2A-0C39 0C3D-0C44
0C46-0C48 0C4A-0C4C 0C55-0C56 0C58-0C5A 0C5C-0C5D 0C60-0C63 0C66-0C6F
0C78-0C7E 0C80-0C83 0C85-0C8C 0C8E-0C90 0C92-0CA8 0CAA-0CB3 0CB5-0CB9
0CBD-0CC4 0CC6-0CC8 0CCA-0CCC 0CD5-0CD6 0CDC-0CDE 0CE0-0CE3 0CE6-0CEF
0CF1-0CF3 0D00-0D0C 0D0E-0D10 0D12-0D3A 0D3D-0D44 0D46-0D48 0D4A-0D4C
0D4E 0D54-0D63 0D66-0D78 0D7A-0D7F 0D81-0D83 0D85-0D96 0D9A-0DB1
0DB3-0DBB 0DBD 0DC0-0DC6 0DCF-0DD4 0DD6 0DD8-0DDF 0DE6-0DEF 0DF2-0DF3
0E01-0E3A 0E40-0E46 0E4D 0E50-0E59 0E81-0E82 0E84 0E86-0E8A 0E8C-0EA3
0EA5 0EA7-0EB9 0EBB-0EBD 0EC0-0EC4 0EC6 0ECD 0ED0-0ED9 0EDC-0EDF 0F00
0F20-0F33 0F40-0F47 0F49-0F6C 0F71-0F83 0F88-0F97 0F99-0FBC 1000-1036
1038 103B-1049 1050-109D 10A0-10C5 10C7 10CD 10D0-10FA 10FC-1248
124A-124D 1250-1256 1258 125A-125D 1260-1288 128A-128D 1290-12B0
12B2-12B5 12B8-12BE 12C0 12C2-12C5 12C8-12D6 12D8-1310 1312-1315
1318-135A 1369-137C 1380-138F 13A0-13F5 13F8-13FD 1401-166C 166F-167F
1681-169A 16A0-16EA 16EE-16F8 1700-1713 171F-1733 1740-1753 1760-176C
176E-1770 1772-1773 1780-17B3 17B6-17C8 17D7 17DC 17E0-17E9 17F0-17F9
1810-1819 1820-1878 1880-18AA 18B0-18F5 1900-191E 1920-192B 1930-1938
1946-196D 1970-1974 1980-19AB 19B0-19C9 19D0-19DA 1A00-1A1B 1A20-1A5E
1A61-1A74 1A80-1A89 1A90-1A99 1AA7 1ABF-1AC0 1ACC-1ACE 1B00-1B33
1B35-1B43 1B45-1B4C 1B50-1B59 1B80-1BA9 1BAC-1BE5 1BE7-1BF1 1C00-1C36
1C40-1C49 1C4D-1C7D 1C80-1C8A 1C90-1CBA 1CBD-1CBF 1CE9-1CEC 1CEE-1CF3
1CF5-1CF6 1CFA 1D00-1DBF 1DD3-1DF4 1E00-1F15 1F18-1F1D 1F20-1F45
1F48-1F4D 1F50-1F57 1F59 1F5B 1F5D 1F5F-1F7D 1F80-1FB4 1FB6-1FBC 1FBE
1FC2-1FC4 1FC6-1FCC 1FD0-1FD3 1FD6-1FDB 1FE0-1FEC 1FF2-1FF4 1FF6-1FFC
2070-2071 2074-2079 207F-2089 2090-209C 2102 2107 210A-2113 2115
2119-211D 2124 2126 2128 212A-212D 212F-2139 213C-213F 2145-2149 214E
2150-2189 2460-249B 24B6-24FF 2776-2793 2C00-2CE4 2CEB-2CEE 2CF2-2CF3
2CFD 2D00-2D25 2D27 2D2D 2D30-2D67 2D6F 2D80-2D96 2DA0-2DA6 2DA8-2DAE
2DB0-2DB6 2DB8-2DBE 2DC0-2DC6 2DC8-2DCE 2DD0-2DD6 2DD8-2DDE 2DE0-2DFF
2E2F 3005-3007 3021-3029 3031-3035 3038-303C 3041-3096 309D-309F
30A1-30FA 30FC-30FF 3105-312F 3131-318E 3192-3195 31A0-31BF 31F0-31FF
3220-3229 3248-324F 3251-325F 3280-3289 32B1-32BF 3400-4DBF 4E00-A48C
A4D0-A4FD A500-A60C A610-A62B A640-A66E A674-A67B A67F-A6EF A717-A71F
A722-A788 A78B-A7DC A7F1-A805 A807-A827 A830-A835 A840-A873 A880-A8C3
A8C5 A8D0-A8D9 A8F2-A8F7 A8FB A8FD-A92A A930-A952 A960-A97C A980-A9B2
A9B4-A9BF A9CF-A9D9 A9E0-A9FE AA00-AA36 AA40-AA4D AA50-AA59 AA60-AA76
AA7A-AABE AAC0 AAC2 AADB-AADD AAE0-AAEF AAF2-AAF5 AB01-AB06 AB09-AB0E
AB11-AB16 AB20-AB26 AB28-AB2E AB30-AB5A AB5C-AB69 AB70-ABEA ABF0-ABF9
AC00-D7A3 D7B0-D7C6 D7CB-D7FB F900-FA6D FA70-FAD9 FB00-FB06 FB13-FB17
FB1D-FB28 FB2A-FB36 FB38-FB3C FB3E FB40-FB41 FB43-FB44 FB46-FBB1
FBD3-FD3D FD50-FD8F FD92-FDC7 FDF0-FDFB FE70-FE74 FE76-FEFC FF10-FF19
FF21-FF3A FF41-FF5A FF66-FFBE FFC2-FFC7 FFCA-FFCF FFD2-FFD7 FFDA-FFDC
10000-1000B 1000D-10026 10028-1003A 1003C-1003D 1003F-1004D
10050-1005D 10080-100FA 10107-10133 10140-10178 1018A-1018B
10280-1029C 102A0-102D0 102E1-102FB 10300-10323 1032D-1034A
10350-1037A 10380-1039D 103A0-103C3 103C8-103CF 103D1-103D5
10400-1049D 104A0-104A9 104B0-104D3 104D8-104FB 10500-10527
10530-10563 10570-1057A 1057C-1058A 1058C-10592 10594-10595
10597-105A1 105A3-105B1 105B3-105B9 105BB-105BC 105C0-105F3
10600-10736 10740-10755 10760-10767 10780-10785 10787-107B0
107B2-107BA 10800-10805 10808 1080A-10835 10837-10838 1083C
1083F-10855 10858-10876 10879-1089E 108A7-108AF 108E0-108F2
108F4-108F5 108FB-1091B 10920-10939 10940-10959 10980-109B7
109BC-109CF 109D2-10A03 10A05-10A06 10A0C-10A13 10A15-10A17
10A19-10A35 10A40-10A48 10A60-10A7E 10A80-10A9F 10AC0-10AC7
10AC9-10AE4 10AEB-10AEF 10B00-10B35 10B40-10B55 10B58-10B72
10B78-10B91 10BA9-10BAF 10C00-10C48 10C80-10CB2 10CC0-10CF2
10CFA-10D27 10D30-10D39 10D40-10D65 10D69 10D6F-10D85 10E60-10E7E
10E80-10EA9 10EAB-10EAC 10EB0-10EB1 10EC2-10EC7 10EFA-10EFC
10F00-10F27 10F30-10F45 10F51-10F54 10F70-10F81 10FB0-10FCB
10FE0-10FF6 11000-11045 11052-1106F 11071-11075 11080-110B8 110C2
110D0-110E8 110F0-110F9 11100-11132 11136-1113F 11144-11147
11150-11172 11176 11180-111BF 111C1-111C4 111CE-111DA 111DC
111E1-111F4 11200-11211 11213-11234 11237 1123E-11241 11280-11286
11288 1128A-1128D 1128F-1129D 1129F-112A8 112B0-112E8 112F0-112F9
11300-11303 11305-1130C 1130F-11310 11313-11328 1132A-11330
11332-11333 11335-11339 1133D-11344 11347-11348 1134B-1134C 11350
11357 1135D-11363 11380-11389 1138B 1138E 11390-113B5 113B7-113C0
113C2 113C5 113C7-113CA 113CC-113CD 113D1 113D3 11400-11441
11443-11445 11447-1144A 11450-11459 1145F-11461 11480-114C1
114C4-114C5 114C7 114D0-114D9 11580-115B5 115B8-115BE 115D8-115DD
11600-1163E 11640 11644 11650-11659 11680-116B5 116B8 116C0-116C9
116D0-116E3 11700-1171A 1171D-1172A 11730-1173B 11740-11746
11800-11838 118A0-118F2 118FF-11906 11909 1190C-11913 11915-11916
11918-11935 11937-11938 1193B-1193C 1193F-11942 11950-11959
119A0-119A7 119AA-119D7 119DA-119DF 119E1 119E3-119E4 11A00-11A32
11A35-11A3E 11A50-11A97 11A9D 11AB0-11AF8 11B60-11B67 11BC0-11BE0
11BF0-11BF9 11C00-11C08 11C0A-11C36 11C38-11C3E 11C40 11C50-11C6C
11C72-11C8F 11C92-11CA7 11CA9-11CB6 11D00-11D06 11D08-11D09
11D0B-11D36 11D3A 11D3C-11D3D 11D3F-11D41 11D43 11D46-11D47
11D50-11D59 11D60-11D65 11D67-11D68 11D6A-11D8E 11D90-11D91
11D93-11D96 11D98 11DA0-11DA9 11DB0-11DDB 11DE0-11DE9 11EE0-11EF6
11F00-11F10 11F12-11F3A 11F3E-11F40 11F50-11F59 11FB0 11FC0-11FD4
12000-12399 12400-1246E 12480-12543 12F90-12FF0 13000-1342F
13441-13446 13460-143FA 14400-14646 16100-1612E 16130-16139
16800-16A38 16A40-16A5E 16A60-16A69 16A70-16ABE 16AC0-16AC9
16AD0-16AED 16B00-16B2F 16B40-16B43 16B50-16B59 16B5B-16B61
16B63-16B77 16B7D-16B8F 16D40-16D6C 16D70-16D79 16E40-16E96
16EA0-16EB8 16EBB-16ED3 16F00-16F4A 16F4F-16F87 16F8F-16F9F
16FE0-16FE1 16FE3 16FF0-16FF6 17000-18CD5 18CFF-18D1E 18D80-18DF2
1AFF0-1AFF3 1AFF5-1AFFB 1AFFD-1AFFE 1B000-1B122 1B132 1B150-1B152
1B155 1B164-1B167 1B170-1B2FB 1BC00-1BC6A 1BC70-1BC7C 1BC80-1BC88
1BC90-1BC99 1BC9E 1CCF0-1CCF9 1D2C0-1D2D3 1D2E0-1D2F3 1D360-1D378
1D400-1D454 1D456-1D49C 1D49E-1D49F 1D4A2 1D4A5-1D4A6 1D4A9-1D4AC
1D4AE-1D4B9 1D4BB 1D4BD-1D4C3 1D4C5-1D505 1D507-1D50A 1D50D-1D514
1D516-1D51C 1D51E-1D539 1D53B-1D53E 1D540-1D544 1D546 1D54A-1D550
1D552-1D6A5 1D6A8-1D6C0 1D6C2-1D6DA 1D6DC-1D6FA 1D6FC-1D714
1D716-1D734 1D736-1D74E 1D750-1D76E 1D770-1D788 1D78A-1D7A8
1D7AA-1D7C2 1D7C4-1D7CB 1D7CE-1D7FF 1DF00-1DF1E 1DF25-1DF2A
1E000-1E006 1E008-1E018 1E01B-1E021 1E023-1E024 1E026-1E02A
1E030-1E06D 1E08F 1E100-1E12C 1E137-1E13D 1E140-1E149 1E14E
1E290-1E2AD 1E2C0-1E2EB 1E2F0-1E2F9 1E4D0-1E4EB 1E4F0-1E4F9
1E5D0-1E5ED 1E5F0-1E5FA 1E6C0-1E6DE 1E6E0-1E6F5 1E6FE-1E6FF
1E7E0-1E7E6 1E7E8-1E7EB 1E7ED-1E7EE 1E7F0-1E7FE 1E800-1E8C4
1E8C7-1E8CF 1E900-1E943 1E947 1E94B 1E950-1E959 1EC71-1ECAB
1ECAD-1ECAF 1ECB1-1ECB4 1ED01-1ED2D 1ED2F-1ED3D 1EE00-1EE03
1EE05-1EE1F 1EE21-1EE22 1EE24 1EE27 1EE29-1EE32 1EE34-1EE37 1EE39
1EE3B 1EE42 1EE47 1EE49 1EE4B 1EE4D-1EE4F 1EE51-1EE52 1EE54 1EE57
1EE59 1EE5B 1EE5D 1EE5F 1EE61-1EE62 1EE64 1EE67-1EE6A 1EE6C-1EE72
1EE74-1EE77 1EE79-1EE7C 1EE7E 1EE80-1EE89 1EE8B-1EE9B 1EEA1-1EEA3
1EEA5-1EEA9 1EEAB-1EEBB 1F100-1F10C 1F130-1F149 1F150-1F169
1F170-1F189 1FBF0-1FBF9 20000-2A6DF 2A700-2B81D 2B820-2CEAD
2CEB0-2EBE0 2EBF0-2EE5D 2F800-2FA1D 30000-3134A 31350-33479
"""


def _code_point_ranges(text: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    starts: List[int] = []
    ends: List[int] = []
    for token in text.split():
        first, _, last = token.partition("-")
        starts.append(int(first, 16))
        ends.append(int(last or first, 16))
    return tuple(starts), tuple(ends)


_ALPHANUMERIC_STARTS, _ALPHANUMERIC_ENDS = _code_point_ranges(_ALPHANUMERIC)


def is_alphanumeric(char: str) -> bool:
    """Whether one character is a letter or digit as the server's
    Rust ``char::is_alphanumeric`` reads it, in Unicode 17.0 on every
    Python, whatever the version of its ``unicodedata``."""
    code = ord(char)
    index = bisect.bisect_right(_ALPHANUMERIC_STARTS, code) - 1
    return index >= 0 and code <= _ALPHANUMERIC_ENDS[index]


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
