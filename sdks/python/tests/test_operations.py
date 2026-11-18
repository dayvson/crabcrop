"""The typed operation builders (issue #716): the builder cases of
spec/operation-vectors.json, which the TypeScript SDK runs too, the
TypeScript suite's own expectations, and the grammar's sets read
back from the TypeScript SDK's source."""

import ast
import json
import re
import struct
import sys
import unicodedata
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crabcrop_url import MAX_DIMENSION, build_url, filter, join_ops, ops  # noqa: E402
from crabcrop_url import _grammar  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
CORPUS = json.loads((REPO / "spec" / "operation-vectors.json").read_text(encoding="utf-8"))
TYPESCRIPT_SRC = REPO / "sdks" / "typescript" / "src"

# The corpus passes an options object last to these two calls; the
# Python builders take those options as keyword arguments.
KEYWORD_OPTIONS = {"filter.format", "filter.studio"}


def python_builder(call):
    """The Python builder for a corpus call such as
    ``ops.cropObjects``: the same namespace, the name in snake case."""
    namespace, name = call.split(".")
    table = ops if namespace == "ops" else filter
    return getattr(table, re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower())


def run_case(case):
    args = [run_case(a) if isinstance(a, dict) and "call" in a else a for a in case["args"]]
    kwargs = args.pop() if case["call"] in KEYWORD_OPTIONS and args and isinstance(args[-1], dict) else {}
    return python_builder(case["call"])(*args, **kwargs)


class TestBuilderCorpus(unittest.TestCase):
    def test_builders_match_the_generated_builder_corpus(self):
        # Each entry is one builder call and the segment it must
        # produce, or a call it must refuse: the same file the
        # server crate checks against its parser and the TypeScript
        # SDK checks against its builders.
        cases = CORPUS["builder_cases"]
        self.assertGreaterEqual(len(cases), 100)
        for case in cases:
            with self.subTest(call=case["call"], args=case["args"]):
                if case.get("throws"):
                    with self.assertRaises(ValueError):
                        run_case(case)
                else:
                    self.assertEqual(run_case(case), case["segment"])

    def test_the_corpus_names_python_builders_only(self):
        # A builder the TypeScript SDK gains shows up here first.
        names = {c["call"] for c in CORPUS["builder_cases"]}
        for case in CORPUS["builder_cases"]:
            names.update(a["call"] for a in case["args"] if isinstance(a, dict) and "call" in a)
        for call in sorted(names):
            with self.subTest(call=call):
                self.assertTrue(callable(python_builder(call)))


class TestTypedSpellings(unittest.TestCase):
    """The TypeScript suite's hand-written spellings
    (sdks/typescript/test/operations.test.ts), in Python."""

    def test_segments_match_the_canonical_spellings(self):
        cases = [
            (ops.size(300, 200), "300x200"),
            (ops.size(300, 0), "300x0"),
            (ops.size("orig", 200), "origx200"),
            (ops.size(-300, 200), "-300x200"),
            (ops.fit_in(800, 600), "fit-in:800x600"),
            (ops.fit_in(800, 600, True), "fit-in:800x600:adaptive"),
            (ops.crop(1200, 630), "crop:1200x630"),
            (ops.crop(1200, 630, "smart"), "crop:1200x630:smart"),
            (ops.crop(100, 100, "face"), "crop:100x100:face"),
            (ops.crop(100, 100, "bottom-right"), "crop:100x100:bottom-right"),
            (ops.crop_objects(400, 300, ["dog"]), "crop:400x300:objects(dog):1"),
            (ops.crop_objects(400, 300, ["person", "car"], 1.5), "crop:400x300:objects(person,car):1.5"),
            (ops.crop_objects(400, 300, ("dog", "Dog")), "crop:400x300:objects(dog):1"),
            (ops.crop_objects(400, 300, ["traffic light"], 2), "crop:400x300:objects(traffic%20light):2"),
            # The cap counts distinct classes, as the server's does
            # (issue #408): 33 entries naming 32 classes build.
            (
                ops.crop_objects(400, 300, [f"c{i}" for i in range(32)] + ["C0"]),
                f"crop:400x300:objects({','.join(f'c{i}' for i in range(32))}):1",
            ),
            (ops.crop_rect(10, 20, 110, 220), "crop-rect:10x20:110x220"),
            (ops.focal(10, 20, 110, 220), "focal:10x20:110x220"),
            (ops.focal_point(0.75, 0.4, 2), "focalpoint:0.75x0.4:2"),
            (ops.focal_point(0.75, 0.4), "focalpoint:0.75x0.4:1"),
            (ops.pad(100, 100), "pad:100x100"),
            (ops.pad(100, 100, "FF8800"), "pad:100x100:ff8800"),
            (ops.pad(100, 100, "blur"), "pad:100x100:blur"),
            (ops.preset("thumb-2_large"), "preset:thumb-2_large"),
            (ops.maxage(0), "maxage:0"),
            (ops.maxage(31536000), "maxage:31536000"),
            (ops.debug_focal(), "debug:focal"),
            (ops.filters(filter.studio("product", bg="white")), "filters:studio(product,bg:white)"),
            (ops.filters(filter.studio("scene", pad=25)), "filters:studio(scene,pad:25)"),
        ]
        for got, want in cases:
            self.assertEqual(got, want)

    def test_filters_match_the_canonical_spellings(self):
        cases = [
            (filter.grayscale(), "grayscale"),
            (filter.invert(), "invert"),
            (filter.no_upscale(), "no_upscale"),
            (filter.equalize(), "equalize"),
            (filter.enhance(), "enhance"),
            (filter.remove_bg(), "removebg"),
            (filter.keep_profile(), "keep_profile"),
            (filter.sepia(), "sepia"),
            (filter.blur_faces(), "blurfaces"),
            (filter.pixelate_faces(), "pixelatefaces"),
            (filter.blur(2.5), "blur(2.5)"),
            (filter.sharpen(0.5), "sharpen(0.5)"),
            (filter.brighten(-20), "brighten(-20)"),
            (filter.contrast(-100), "contrast(-100)"),
            (filter.gamma(1), "gamma(1)"),
            (filter.exposure(1.5), "exposure(1.5)"),
            (filter.exposure(0), "exposure(0)"),
            (filter.highlights_shadows(30, -20), "highlights_shadows(30,-20)"),
            (filter.vibrance(-100), "vibrance(-100)"),
            (filter.hue_rotate(90), "huerotate(90)"),
            (filter.saturation(0.5), "saturation(0.5)"),
            (filter.rgb(1, -2, 3), "rgb(1,-2,3)"),
            (filter.curve("r", [(0, 64), (255, 255)]), "curve(r,0,64,255,255)"),
            (filter.curve("g", [[0, 0], [128, 255], [255, 255]]), "curve(g,0,0,128,255,255,255)"),
            (filter.convolution([0, -1, 0, -1, 5, -1, 0, -1, 0]), "convolution(0,-1,0,-1,5,-1,0,-1,0,0)"),
            (filter.convolution([0, -1, 0, -1, 5, -1, 0, -1, 0], True), "convolution(0,-1,0,-1,5,-1,0,-1,0,1)"),
            (
                filter.convolution([0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625], 1),
                "convolution(0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625,1)",
            ),
            (
                filter.convolution([1 if i == 12 else 0 for i in range(25)], 0),
                "convolution(0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0)",
            ),
            (filter.noise(40), "noise(40)"),
            (filter.rotate(-30.25), "rotate(-30.25)"),
            (filter.trim(), "trim(10)"),
            (filter.trim_sides(20, 5, 0, 5, 0), "trim(20,5,0,5,0)"),
            (filter.quality(1), "quality(1)"),
            (filter.quality("auto"), "quality(auto)"),
            (filter.palette(256), "palette(256)"),
            (filter.format("webm"), "format(webm)"),
            (filter.format("jpeg", progressive=True, subsampling=444), "format(jpeg,progressive,subsampling:444)"),
            (filter.format("jpeg", progressive=False), "format(jpeg)"),
            (filter.format("png", interlace=True), "format(png,interlace)"),
            (filter.format("webp", method=0, sns=100), "format(webp,method:0,sns:100)"),
            (filter.quality_per_format({"webp": 75, "jpeg": 80}), "quality(jpeg:80,webp:75)"),
            (filter.quality_per_format({"jpeg": "auto", "avif": 60}, 70), "quality(70,avif:60,jpeg:auto)"),
            (filter.quality_per_format({}, "auto"), "quality(auto)"),
            (filter.fill("FF8800"), "fill(ff8800)"),
            (filter.face_pad(0.25), "facepad(0.25)"),
            (filter.studio("product", bg="F5F0E8"), "studio(product,bg:f5f0e8)"),
            (filter.studio("product", bg="f5f0e8", shadow="strong", pad=25), "studio(product,bg:f5f0e8,shadow:strong,pad:25)"),
            # Grammar order (bg, shadow, pad), whatever the call's order.
            (filter.studio("product", pad=0, shadow="none"), "studio(product,shadow:none,pad:0)"),
            # Arguments at their defaults collapse into the bare form.
            (filter.studio("product", shadow="soft", pad=10), "studio(product)"),
            (filter.studio("scene", pad=10), "studio(scene)"),
            (filter.upscale(4), "upscale(4)"),
            (filter.dpr(2), "dpr(2)"),
            (filter.dpr(2.0), "dpr(2)"),
            (filter.max_bytes(100000), "max_bytes(100000)"),
            (filter.max_bytes(2**64 - 1), f"max_bytes({2**64 - 1})"),
            (filter.radius(24), "radius(24,24,24,24)"),
            (filter.radius((24, 24, 0, 0)), "radius(24,24,0,0)"),
            (filter.border(4, "000000"), "border(4,000000)"),
            (filter.mask("ellipse"), "mask(ellipse)"),
            (filter.duotone("1a1a40", "FFD9A0"), "duotone(1a1a40,ffd9a0)"),
            (filter.monochrome("ff8000"), "monochrome(ff8000)"),
            (filter.pixelate(MAX_DIMENSION), f"pixelate({MAX_DIMENSION})"),
            (filter.flip("v"), "flip(v)"),
            (
                filter.watermark("https://example.com/wm.png", 10, 10, 80),
                "watermark(https%3A%2F%2Fexample.com%2Fwm.png,10,10,80)",
            ),
            (
                filter.overlay("https://example.com/ov.png", 5, 5, 50, "multiply", 50),
                "overlay(https%3A%2F%2Fexample.com%2Fov.png,5,5,50,multiply,50)",
            ),
            (filter.text("Hello, World!", 32, "FF8800", "center"), "text(Hello%2C%20World%21,32,ff8800ff,center)"),
            # The encoding covers everything outside the unreserved
            # set, the characters encodeURIComponent leaves included.
            (filter.text("(a)!*'~", 32, "ff880080", "top-left"), "text(%28a%29%21%2A%27~,32,ff880080,top-left)"),
            (filter.text("é" * 1000, 100, "ffffff", "top-left"), f"text({'%C3%A9' * 1000},100,ffffffff,top-left)"),
            (filter.text("🦀" * 1000, 100, "ffffff", "top-left"), f"text({'%F0%9F%A6%80' * 1000},100,ffffffff,top-left)"),
            (filter.blur_class("é" * 32), f"blurclass({'%C3%A9' * 32})"),
            (filter.pixelate_class("license plate"), "pixelateclass(license%20plate)"),
            (filter.recolor_class("person ", "FF0000"), "recolorclass(person,ff0000)"),
            (filter.blur_region(120, 40, 180, 60), "blur_region(120,40,180,60)"),
            (filter.pixelate_region(120, 40, 180, 60), "pixelate_region(120,40,180,60)"),
            (filter.remove_region(0, 0, 1, 1), "remove_region(0,0,1,1)"),
        ]
        for got, want in cases:
            self.assertEqual(got, want)

    def test_a_typed_pipeline_signs_like_the_string_pipeline(self):
        typed = join_ops(
            ops.size(300, 200),
            ops.filters(
                filter.hue_rotate(90),
                filter.sharpen(0.5),
                filter.saturation(2),
                filter.rgb(1, -2, 3),
                filter.equalize(),
            ),
        )
        written = "300x200/filters:huerotate(90):sharpen(0.5):saturation(2):rgb(1,-2,3):equalize"
        self.assertEqual(typed, written)
        source = "https://example.com/photo.jpg"
        self.assertEqual(
            build_url("https://img.example.com", "test-secret", typed, source),
            build_url("https://img.example.com", "test-secret", written, source),
        )

    def test_decimal_arguments_spell_the_f32_the_server_prints(self):
        # Issue #472: the server reads these arguments as 32-bit
        # floats and prints the shortest decimal that reads back as
        # the same one. Each expected spelling is what the
        # TypeScript SDK's builders write for the same value.
        def f32(bits):
            return struct.unpack("<f", struct.pack("<I", bits))[0]

        cases = [
            (ops.focal_point(0.1 + 0.2, 0.4, 2), "focalpoint:0.3x0.4:2"),
            (ops.crop_objects(100, 100, ["dog"], 0.1 + 0.2), "crop:100x100:objects(dog):0.3"),
            (filter.blur(1 / 3), "blur(0.33333334)"),
            (filter.face_pad(3.141592653589793 / 2), "facepad(1.5707964)"),
            (filter.rotate(45.1 + 1e-12), "rotate(45.1)"),
            (filter.rotate(-35999.99999), "rotate(-36000)"),
            (filter.saturation(254.99999999), "saturation(255)"),
            (filter.exposure(-0.1), "exposure(-0.1)"),
            (filter.contrast(-0.0), "contrast(0)"),
            (filter.sharpen(1e-7), "sharpen(0.0000001)"),
            (filter.sharpen(2 / 3), "sharpen(0.6666667)"),
            (filter.sharpen(16777217), "sharpen(16777216)"),
            (filter.sharpen(123456789), "sharpen(123456790)"),
            (filter.sharpen(9.999999e-5), "sharpen(0.00009999999)"),
            (filter.sharpen(f32(0x3F800001)), "sharpen(1.0000001)"),
            (filter.sharpen(f32(0x4B7FFFFF)), "sharpen(16777215)"),
            (filter.sharpen(3e38), f"sharpen(3{'0' * 38})"),
            (filter.sharpen(f32(0x7F7FFFFF)), "sharpen(340282350000000000000000000000000000000)"),
            (filter.sharpen(2.0**-149), f"sharpen(0.{'0' * 44}1)"),
            (filter.sharpen(f32(0x007FFFFF)), f"sharpen(0.{'0' * 37}11754942)"),
            (filter.sharpen(f32(0x00800001)), f"sharpen(0.{'0' * 37}11754945)"),
            (filter.blur(2.0**-126), f"blur(0.{'0' * 37}11754944)"),
            # Kernel values are 64-bit: JavaScript's spelling, with an
            # exponent spelling (below 1e-6) written out.
            (
                filter.convolution([0.1 + 0.2, 0, 0, 0, 1, 0, 0, 0, 0]),
                "convolution(0.30000000000000004,0,0,0,1,0,0,0,0,0)",
            ),
            (filter.convolution([1e-7, 0, 0, 0, 1, 0, 0, 0, 0]), "convolution(0.0000001,0,0,0,1,0,0,0,0,0)"),
            (filter.convolution([1.5e-9, 0, 0, 0, 1, 0, 0, 0, 0]), "convolution(0.0000000015,0,0,0,1,0,0,0,0,0)"),
            (
                filter.convolution([1.2345678901234567e-06, 0, 0, 0, 1, 0, 0, 0, 0]),
                "convolution(0.0000012345678901234567,0,0,0,1,0,0,0,0,0)",
            ),
            (
                filter.convolution([1023.9999999999999, 0, 0, 0, 1, 0, 0, 0, 0]),
                "convolution(1023.9999999999999,0,0,0,1,0,0,0,0,0)",
            ),
        ]
        for got, want in cases:
            self.assertEqual(got, want)
        for refused in (
            lambda: filter.sharpen(1e39),
            lambda: filter.sharpen(1e-50),
            lambda: filter.blur(1e-40),
            lambda: filter.gamma(1e-50),
            lambda: ops.focal_point(0.5, 0.5, 1e-50),
            lambda: ops.crop_objects(100, 100, ["dog"], 1e-50),
        ):
            with self.assertRaises(ValueError):
                refused()

    def test_class_names_spell_trimmed_as_the_server_keys_them(self):
        cases = [
            (ops.crop_objects(100, 100, [" person", "PERSON\t"]), "crop:100x100:objects(person):1"),
            (ops.crop_objects(100, 100, [" traffic light "]), "crop:100x100:objects(traffic%20light):1"),
            (filter.blur_class("\u0085person\u3000"), "blurclass(person)"),
            (filter.pixelate_class("\nperson"), "pixelateclass(person)"),
        ]
        for got, want in cases:
            self.assertEqual(got, want)
        # The server keeps U+FEFF and the information separators,
        # which Python's str.strip() would remove.
        for kept in ("\ufeffperson", "person\x1c", "\x1fperson"):
            with self.assertRaises(ValueError, msg=repr(kept)):
                filter.blur_class(kept)
        # Only A to Z fold: the Kelvin sign is its own class.
        self.assertEqual(ops.crop_objects(1, 1, ["K", "k", "K"]), "crop:1x1:objects(%E2%84%AA,k):1")

    def test_class_names_take_the_letters_and_digits_the_server_takes(self):
        # The server's test is Rust's char::is_alphanumeric, so a
        # letter is any Alphabetic character: beside the categories L
        # and N, the vowel signs of Indic scripts, the points of
        # Hebrew, combining marks such as U+0345, and circled letters.
        hindi = "\u0939\u093f\u0902\u0926\u0940"  # हिंदी: two vowel signs and a nasal sign
        shalom = "\u05e9\u05b8\u05c1\u05dc\u05d5\u05b9\u05dd"  # שָׁלוֹם, pointed
        cases = [
            (filter.blur_class(hindi), "blurclass(%E0%A4%B9%E0%A4%BF%E0%A4%82%E0%A4%A6%E0%A5%80)"),
            (ops.crop_objects(100, 100, ["Ⓐ"]), "crop:100x100:objects(%E2%92%B6):1"),
            (filter.pixelate_class("\u03b1\u0345"), "pixelateclass(%CE%B1%CD%85)"),
            (filter.recolor_class(shalom, "FF0000"), "recolorclass(%D7%A9%D6%B8%D7%81%D7%9C%D7%95%D6%B9%D7%9D,ff0000)"),
        ]
        for got, want in cases:
            self.assertEqual(got, want)
        # A mark that is not Alphabetic is no letter: the server
        # refuses a decomposed é (e, then U+0301) and a lone mark.
        for refused in ("e\u0301", "\u0301", "a\u20dd"):
            with self.assertRaises(ValueError, msg=repr(refused)):
                filter.blur_class(refused)

    def test_the_alphabetic_table_holds_only_what_the_categories_miss(self):
        # The written-out ranges are sorted, apart, and hold no
        # character this Python already reads as L or N.
        starts, ends = _grammar._MORE_STARTS, _grammar._MORE_ENDS
        self.assertGreater(len(starts), 200)
        for index, (start, end) in enumerate(zip(starts, ends)):
            self.assertLessEqual(start, end)
            if index:
                self.assertLess(ends[index - 1] + 1, start)
            for code in range(start, end + 1):
                self.assertNotIn(unicodedata.category(chr(code))[0], "LN", hex(code))
                self.assertTrue(_grammar.is_alphanumeric(chr(code)), hex(code))
        for char in "aZ\u0663\u2160\u00bd":
            self.assertTrue(_grammar.is_alphanumeric(char), repr(char))
        for char in " _;\u0301\u20dd\u200b\U0001f600":
            self.assertFalse(_grammar.is_alphanumeric(char), repr(char))

    def test_builders_refuse_what_the_server_refuses(self):
        bad = [
            lambda: filter.quality(0),
            lambda: filter.quality(101),
            lambda: filter.quality_per_format({}),
            lambda: filter.quality_per_format({"jpeg": 0}),
            lambda: filter.quality_per_format({"jpg": 80}),
            lambda: filter.format("jpeg", interlace=True),
            lambda: filter.format("png", progressive=False),
            lambda: filter.format("webp", method=7),
            lambda: filter.format("webp", sns=101),
            lambda: filter.format("avif", subsampling=420),
            lambda: filter.format("jpeg", subsampling=421),
            lambda: filter.format("bmp"),
            lambda: filter.palette(257),
            lambda: filter.brighten(256),
            lambda: filter.contrast(101),
            lambda: filter.gamma(11),
            lambda: filter.gamma(float("nan")),
            lambda: filter.exposure(float("inf")),
            lambda: filter.highlights_shadows(0, -101),
            lambda: filter.vibrance(float("nan")),
            lambda: filter.trim_sides(20, -1, 0, 0, 0),
            lambda: filter.saturation(256),
            lambda: filter.blur(0),
            lambda: filter.blur(513),
            lambda: filter.curve("x", [(0, 0), (255, 255)]),
            lambda: filter.curve("rgb", [(0, 0)]),
            lambda: filter.curve("rgb", [(i * 15, i * 15) for i in range(17)]),
            lambda: filter.curve("rgb", [(128, 0), (64, 255)]),
            lambda: filter.curve("rgb", [(0, 0), (255, 255.5)]),
            lambda: filter.curve("rgb", [(0, 0, 0), (255, 255)]),
            lambda: filter.convolution([0, 0, 0, 0, 1, 0, 0, 0]),
            lambda: filter.convolution([0, -1, 0, -1, 1025, -1, 0, -1, 0]),
            lambda: filter.convolution([0, -1, 0, -1, float("nan"), -1, 0, -1, 0]),
            lambda: filter.convolution([0, -1, 0, -1, 5, -1, 0, -1, 0], 2),
            lambda: filter.noise(1.5),
            lambda: filter.rotate(36001),
            lambda: ops.size(16384, 10),
            lambda: ops.size(1.5, 10),
            lambda: ops.fit_in(-300, 200),
            lambda: ops.crop(100, 100, "middle"),
            lambda: ops.crop_objects(0, 100, ["dog"]),
            lambda: ops.crop_objects(100, 100, []),
            lambda: ops.crop_objects(100, 100, ["bad;name"]),
            lambda: ops.crop_objects(100, 100, ["dog"], 101),
            lambda: ops.crop_objects(100, 100, [f"c{i}" for i in range(33)]),
            lambda: ops.pad(100, 100, "xyz"),
            lambda: ops.crop_rect(10, 10, 10, 20),
            lambda: ops.focal(-1, 0, 10, 20),
            lambda: ops.focal_point(0.75, -0.1),
            lambda: ops.focal_point(0.75, 0.4, 0),
            lambda: ops.maxage(31536001),
            lambda: filter.text("", 32, "ff8800", "center"),
            lambda: filter.text("hi", 32, "ff88", "center"),
            lambda: filter.text("hi", 32, "ff8800", "middle"),
            lambda: filter.text("a" * 1001, 12, "ffffff", "top-left"),
            lambda: filter.text("W" * 101, 1000, "ffffff", "top-left"),
            lambda: filter.watermark("", 0, 0, 50),
            lambda: filter.watermark("https://x/y.png", 0, 0, 101),
            lambda: filter.overlay("https://x/y.png", 0, 0, 50, "darken"),
            lambda: filter.overlay("https://x/y.png", 0, 0, 50, "normal", 401),
            lambda: filter.blur_class("é" * 33),
            lambda: ops.preset("thumb!"),
            lambda: ops.preset(""),
            lambda: ops.preset("a/b"),
            lambda: filter.max_bytes(0),
            lambda: filter.max_bytes(2**64),
            lambda: filter.studio("portrait"),
            lambda: filter.studio("product", bg="blur"),
            lambda: filter.studio("product", bg="White"),
            lambda: filter.studio("product", shadow="hard"),
            lambda: filter.studio("product", pad=41),
            lambda: filter.studio("scene", bg="white"),
            lambda: filter.studio("scene", shadow="soft"),
            lambda: filter.upscale(3),
            lambda: filter.dpr(2.5),
            lambda: filter.radius(-1),
            lambda: filter.border(-1, "000000"),
            lambda: filter.mask("square"),
            lambda: filter.flip("x"),
            lambda: ops.filters(),
        ]
        for build in bad:
            with self.assertRaises(ValueError):
                build()


class TestHostileArguments(unittest.TestCase):
    """Arguments the type hints forbid still raise ValueError, the
    one error every refusal is, and never mint a segment."""

    def test_every_refusal_is_a_value_error(self):
        bad = [
            # A bool is an int in Python: True must not read as 1.
            lambda: ops.size(True, 200),
            lambda: filter.brighten(False),
            lambda: filter.sharpen(True),
            lambda: filter.dpr(True),
            lambda: filter.upscale(True),
            lambda: filter.format("jpeg", subsampling=True),
            lambda: filter.max_bytes(True),
            # Numbers spelled as strings, and integral floats for
            # integer parameters, as build_srcset refuses them.
            lambda: filter.blur("2"),
            lambda: filter.sharpen("2"),
            lambda: ops.maxage(60.0),
            lambda: filter.upscale(2.0),
            lambda: filter.format("jpeg", subsampling=444.0),
            lambda: ops.crop(300.0, 200),
            lambda: ops.size(None, 200),
            # Values past every float range.
            lambda: filter.sharpen(10**400),
            lambda: filter.blur(10**400),
            lambda: filter.contrast(10**400),
            lambda: filter.rotate(-(10**400)),
            lambda: ops.size(10**400, 1),
            # Containers of the wrong shape.
            lambda: ops.crop_objects(100, 100, "person"),
            lambda: ops.crop_objects(100, 100, [42]),
            lambda: filter.radius([1, 2]),
            lambda: filter.radius([]),
            lambda: filter.curve("rgb", "0,0,255,255"),
            lambda: filter.convolution("010101010"),
            lambda: filter.quality_per_format([("jpeg", 80)]),
            lambda: filter.quality_per_format({1: 80}),
            lambda: ops.filters(filter.grayscale(), 7),
            lambda: join_ops("300x200", None),
            # Text that cannot be UTF-8 encoded.
            lambda: filter.text("\ud800", 12, "ffffff", "center"),
            lambda: filter.watermark("https://x/\udfff.png", 0, 0, 50),
            lambda: filter.blur_class("\ud800"),
            # A hex color with a trailing newline, which a `$`
            # pattern would let through.
            lambda: filter.fill("ffffff\n"),
            lambda: ops.preset("thumb\n"),
        ]
        for build in bad:
            with self.assertRaises(ValueError):
                build()


def typescript_constant(path, name):
    """The value of ``export const <name> = ...;`` in a TypeScript
    source: a number, or an ``[...] as const`` array of literals."""
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^export const {name} = (.*?)(?: as const)?;$", text, re.M | re.S)
    if match is None:
        raise AssertionError(f"{path.name} has no `export const {name}`")
    return ast.literal_eval(match.group(1))


class TestGrammarSets(unittest.TestCase):
    def test_the_sets_and_caps_are_the_typescript_sdks(self):
        # The builders' closed sets and caps are the values the
        # TypeScript SDK's builders import, so the two SDKs refuse
        # the same names and numbers.
        explain = TYPESCRIPT_SRC / "explain.ts"
        for name in (
            "CROP_ALIGNS",
            "TEXT_ANCHORS",
            "IMAGE_FORMATS",
            "QUALITY_FORMATS",
            "JPEG_SUBSAMPLINGS",
            "BLEND_MODES",
            "CURVE_CHANNELS",
            "STUDIO_MODES",
            "STUDIO_BACKDROPS",
            "STUDIO_SHADOWS",
            "UPSCALE_FACTORS",
            "DPR_FACTORS",
            "MASK_SHAPES",
            "FLIP_AXES",
            "MAX_OBJECTS_CLASSES",
            "MAX_TEXT_GLYPHS",
            "MAX_TEXT_EXTENT",
        ):
            with self.subTest(name=name):
                want = typescript_constant(explain, name)
                got = getattr(_grammar, name)
                self.assertEqual(list(got) if isinstance(got, tuple) else got, want)
        self.assertEqual(MAX_DIMENSION, typescript_constant(TYPESCRIPT_SRC / "limits.ts", "MAX_DIMENSION"))
        self.assertEqual(MAX_DIMENSION, 16383)


if __name__ == "__main__":
    unittest.main()
