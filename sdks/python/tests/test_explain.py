"""The explainer (issue #716): ``explain_url``, the port of the
TypeScript SDK's ``explainUrl``. The two grammar corpora run here as
sdks/typescript/test/vectors.test.ts runs them, with the same URL
around each remainder, and the TypeScript suites' explainer cases
that the corpora do not cover (numbers.test.ts, explain-limits.test.ts,
grammar-sets.test.ts, and the explainer parts of vectors.test.ts,
operations.test.ts, exports.test.ts and trim.test.ts) are ported
below. test_explain_differential.py runs the two explainers against
each other."""

import ast
import datetime
import json
import math
import random
import re
import sys
import time
import unittest
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import crabcrop_url as cc  # noqa: E402
from crabcrop_url import _operations  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
EXPLAIN_TS = REPO / "sdks" / "typescript" / "src" / "explain.ts"
VERDICTS = json.loads((REPO / "spec" / "explain-verdicts.json").read_text(encoding="utf-8"))
VECTORS = json.loads((REPO / "spec" / "operation-vectors.json").read_text(encoding="utf-8"))
SIGNING_VECTORS = json.loads((REPO / "spec" / "test-vectors.json").read_text(encoding="utf-8"))

TAG = "a" * 64
SRC = "https://example.com/a.jpg"


def errors(ops, src="https://example.com/a.jpg"):
    return cc.explain_url(f"https://x/{TAG}/{ops}/{src}").errors


def same_float(a, b):
    """Equal, and the same zero: ``Object.is`` for the values here."""
    return a == b and math.copysign(1, a) == math.copysign(1, b)


def f32(value):
    """JavaScript's ``Math.fround``: the double rounded to 32 bits."""
    return _operations._f32(value)


def timed(run, *args):
    started = time.perf_counter()
    out = run(*args)
    return out, time.perf_counter() - started


class TestCorpora(unittest.TestCase):
    """The shared corpora, with the URLs vectors.test.ts puts around
    them."""

    def test_verdicts_match_the_server_grammar(self):
        # spec/explain-verdicts.json (audit P-3 / AP-23): the same
        # entries the playground suite and the server crate's
        # differential tests run.
        cases = VERDICTS["cases"]
        self.assertGreaterEqual(len(cases), 300)
        for case in cases:
            out = cc.explain_url(f"https://img.example.com/{TAG}/{case['remainder']}")
            with self.subTest(remainder=case["remainder"]):
                self.assertEqual(out.source, case["source"])
                self.assertEqual(len(out.errors) > 0, case["error"], out.errors)

    def test_verdicts_match_the_generated_combination_corpus(self):
        # spec/operation-vectors.json (issue #353), at the reference
        # time vectors.test.ts gives.
        cases = VECTORS["cases"]
        self.assertGreaterEqual(len(cases), 700)
        for case in cases:
            out = cc.explain_url(
                f"https://img.example.com/{VECTORS['signature']}/{case['remainder']}", 1700000000
            )
            with self.subTest(remainder=case["remainder"]):
                self.assertEqual(len(out.errors) == 0, case["ok"], out.errors)
                self.assertEqual(out.source, case["source"])

    def test_the_explainer_accepts_what_the_builders_emit_and_rejects_what_they_refuse(self):
        # operations.test.ts: every segment a builder case emits
        # explains cleanly, and every one it refuses (the spelling it
        # would have produced) is one the explainer rejects.
        checked = 0
        for case in VECTORS["builder_cases"]:
            if "segment" not in case:
                continue
            segment = f"filters:{case['segment']}" if case["call"].startswith("filter.") else case["segment"]
            out = cc.explain_url(f"/unsafe/{segment}/https://example.com/a.jpg")
            with self.subTest(call=case["call"], segment=segment):
                self.assertEqual(len(out.errors) > 0, case.get("throws") is True, out.errors)
            checked += 1
        self.assertGreater(checked, 100)


class TestExplanation(unittest.TestCase):
    """The shape of an explanation, and the cases vectors.test.ts and
    exports.test.ts pin."""

    def test_an_explanation_itemizes_the_signature_segments_and_source(self):
        out = cc.explain_url(
            f"https://img.example.com/k2:{TAG}/exp:1893456000/crop:1200x630:smart/{SRC}", 1700000000
        )
        self.assertEqual(
            out.signature,
            cc.ExplainedSignature(
                "hmac",
                TAG,
                'HMAC-SHA256 of the remainder, signed with key "k2"; the server verifies it against '
                "that key only (secret rotation)",
                "k2",
            ),
        )
        self.assertEqual(
            out.segments,
            [
                cc.ExplainedSegment(
                    "exp:1893456000",
                    "expiry",
                    "Expiry",
                    detail="valid strictly before 2030-01-01 00:00:00 UTC (1893456000)",
                    expired=False,
                ),
                cc.ExplainedSegment(
                    "crop:1200x630:smart",
                    "operation",
                    "Crop",
                    detail="crop to the aspect of 1200x630 and resize to it; alignment: smart",
                ),
            ],
        )
        self.assertEqual(out.source, SRC)
        self.assertEqual(out.remainder, f"exp:1893456000/crop:1200x630:smart/{SRC}")
        self.assertEqual(out.errors, [])
        self.assertIsInstance(out, cc.Explanation)

    def test_the_reference_time_decides_expiry(self):
        url = f"/{TAG}/exp:1893456000/{SRC}"
        self.assertFalse(cc.explain_url(url, 1893455999).segments[0].expired)
        expired = cc.explain_url(url, 1893456000).segments[0]
        self.assertTrue(expired.expired)
        self.assertTrue(expired.detail.endswith(" — EXPIRED at the reference time"))
        # The default reference time is now.
        self.assertTrue(cc.explain_url(f"/{TAG}/exp:1/{SRC}").segments[0].expired)
        self.assertFalse(cc.explain_url(f"/{TAG}/exp:{int(time.time()) + 3600}/{SRC}").segments[0].expired)

    def test_expiry_dates_spell_as_javascript_dates_do(self):
        # Date.toISOString, the `T` a space and `Z` UTC: a year past
        # 9999 is the six-digit +YYYYYY, and past JavaScript's last
        # date the explainer says so.
        cases = [
            ("0", "1970-01-01 00:00:00 UTC"),
            ("99999999999", "5138-11-16 09:46:39 UTC"),
            ("253402300799", "9999-12-31 23:59:59 UTC"),
            ("253402300800", "+010000-01-01 00:00:00 UTC"),
            ("8640000000000", "+275760-09-13 00:00:00 UTC"),
            ("8640000000001", "a date beyond representation"),
            ("18446744073709551615", "a date beyond representation"),
        ]
        for value, when in cases:
            detail = cc.explain_url(f"/{TAG}/exp:{value}/{SRC}", 0).segments[0].detail
            self.assertTrue(detail.startswith(f"valid strictly before {when} ({value})"), detail)
        overflow = cc.explain_url(f"/{TAG}/exp:18446744073709551616/{SRC}").segments[0]
        self.assertEqual(
            overflow.error,
            "expiry overflows the server's u64 parser: exp:18446744073709551616 (the server answers 400)",
        )
        self.assertIsNone(overflow.expired)
        # The calendar arithmetic agrees with datetime wherever
        # datetime reaches.
        rng = random.Random(716)
        for seconds in [rng.randrange(253402300800) for _ in range(2000)] + [951782400, 4107542400]:
            want = datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            self.assertEqual(cc._explain._utc_text(seconds), want, seconds)

    def test_a_sealed_token_is_sealed_not_malformed(self):
        # Issue #388, with the spec's sealed tokens in place of
        # sealWithNonce: the explainer reads the token's shape only.
        sealed = [v for v in SIGNING_VECTORS if v.get("sealed")]
        plain = next(v["token"] for v in sealed if "key_id" not in v)
        keyed = next(v for v in sealed if "key_id" in v)
        out = cc.explain_url(f"https://img.example.com/{plain}")
        self.assertEqual(out.signature.kind, "sealed")
        self.assertEqual(out.signature.value, plain)
        self.assertIsNone(out.signature.key_id)
        self.assertEqual((out.errors, out.segments, out.source), ([], [], ""))
        keyed_out = cc.explain_url(f"https://img.example.com/{keyed['token']}")
        self.assertEqual(keyed_out.signature.kind, "sealed")
        self.assertEqual(keyed_out.signature.key_id, keyed["key_id"])
        self.assertEqual(keyed_out.signature.value, keyed["token"].split(":", 1)[1])
        self.assertEqual(keyed_out.errors, [])
        # A trailing path behind a token is not part of the URL the
        # server serves; the token still classifies.
        trailed = cc.explain_url(f"https://img.example.com/{plain}/300x200/{SRC}")
        self.assertEqual(trailed.signature.kind, "sealed")
        self.assertEqual(trailed.segments, [])
        self.assertEqual(trailed.remainder, f"300x200/{SRC}")
        # An unknown version marker is not a malformed signature
        # either: the server refuses it uniformly.
        imitation = cc.explain_url(f"https://img.example.com/s2.AAAA/300x200/{SRC}")
        self.assertEqual(imitation.signature.kind, "invalid")
        self.assertIn("uniform 403 refusal", imitation.signature.detail)

    def test_signature_shapes(self):
        cases = [
            ("unsafe", "unsafe", []),
            (TAG, "hmac", []),
            ("abc123", "invalid", ["malformed signature segment"]),
            (TAG.upper(), "invalid", ["malformed signature segment"]),
            (f"{'k' * 33}:{TAG}", "invalid", ["malformed signature segment"]),
        ]
        for raw, kind, errs in cases:
            out = cc.explain_url(f"https://img.example.com/{raw}/300x200/{SRC}")
            with self.subTest(signature=raw):
                self.assertEqual(out.signature.kind, kind)
                self.assertEqual(out.signature.value, raw)
                self.assertEqual(out.errors, errs)
        single = cc.explain_url("https://img.example.com/photo.jpg")
        self.assertIsNone(single.signature)
        self.assertEqual(single.errors, ["no signature segment: the path is a single segment"])

    def test_the_path_is_cut_by_hand_not_by_a_url_parser(self):
        # Dot segments stay literal (audit P-3), a query and a
        # fragment are not part of the signed path, and a bare path
        # reads like a URL's.
        for given in (
            f"https://img.example.com/{TAG}/300x200/https://e.com/%2e%2e/../a.jpg?w=1#top",
            f"/{TAG}/300x200/https://e.com/%2e%2e/../a.jpg#top",
            f"{TAG}/300x200/https://e.com/%2e%2e/../a.jpg?w=1",
        ):
            out = cc.explain_url(given)
            self.assertEqual(out.source, "https://e.com/%2e%2e/../a.jpg", given)
            self.assertEqual(out.remainder, "300x200/https://e.com/%2e%2e/../a.jpg", given)

    def test_studio_filters_explain_instead_of_counting_as_unknown(self):
        out = cc.explain_url(f"https://img.example.com/{TAG}/filters:studio(product,bg:white)/{SRC}")
        self.assertEqual(out.errors, [])
        self.assertEqual(out.source, SRC)
        self.assertRegex(out.segments[0].detail, r"studio\(product,bg:white\): studio mode")
        # A chain neighbour of a studio filter still parses.
        self.assertEqual(errors("filters:grayscale:studio(scene,pad:25)"), [])
        # And a product-only argument under scene is an error, not a
        # silent no-op.
        self.assertTrue(errors("filters:studio(scene,bg:white)"))

    def test_objects_crops_explain_and_their_malformed_forms_are_refused(self):
        good = cc.explain_url(f"https://img.example.com/{TAG}/crop:400x300:objects(person,car):1.5/{SRC}")
        self.assertEqual(good.errors, [])
        self.assertEqual(good.source, SRC)
        self.assertIn("alignment: objects (person, car)", good.segments[0].detail)
        self.assertIn("scaled by 1.5", good.segments[0].detail)
        # The scale may be omitted (it defaults to 1), a
        # percent-encoded class name decodes in the explanation, and
        # a scale shows as JavaScript's Number spells it.
        tight = cc.explain_url(f"https://img.example.com/{TAG}/crop:400x300:objects(traffic%20light)/{SRC}")
        self.assertEqual(tight.errors, [])
        self.assertIn("objects (traffic light)", tight.segments[0].detail)
        self.assertIn("scaled by 1 about", tight.segments[0].detail)
        self.assertIn(
            "scaled by 1e-7 about",
            cc.explain_url(f"/{TAG}/crop:4x3:objects(dog):1e-7/{SRC}").segments[0].detail,
        )
        self.assertIn(
            "scaled by 2 about",
            cc.explain_url(f"/{TAG}/crop:4x3:objects(dog):+2.00/{SRC}").segments[0].detail,
        )
        for bad in (
            "crop:400x300:objects()",
            "crop:400x300:objects(person,)",
            "crop:400x300:objects(person",
            "crop:400x300:objects(person)junk",
            "crop:400x300:objects(person):0",
            "crop:400x300:objects(person):101",
            "crop:400x300:objects(bad;name)",
            "crop:400x300:objects",
            "crop:-400x300:objects(person)",
        ):
            self.assertTrue(errors(bad), bad)
        # The cap counts distinct names, folded ASCII-only.
        names = [f"c{i}" for i in range(32)]
        self.assertEqual(errors(f"crop:9x9:objects({','.join(names + ['C0'])})"), [])
        self.assertTrue(errors(f"crop:9x9:objects({','.join(names + ['c32'])})"))

    def test_encoder_options_and_per_format_quality_explain(self):
        # Issue #319.
        self.assertEqual(
            errors("filters:format(jpeg,progressive,subsampling:444):quality(80,webp:75,jpeg:90)"), []
        )
        self.assertEqual(errors("filters:format(webp,method:6,sns:20)"), [])
        self.assertTrue(errors("filters:format(avif,subsampling:420)"))
        self.assertTrue(errors("filters:quality(jpeg:80,jpeg:90)"))
        self.assertTrue(errors("filters:quality(mp4:80)"))

    def test_max_dimension_is_the_explainers_size_cap(self):
        # exports.test.ts.
        self.assertIsNone(cc.explain_url(f"/unsafe/{cc.MAX_DIMENSION}x0/{SRC}").segments[0].error)
        self.assertIn(
            "pixelate",
            cc.explain_url(f"/unsafe/filters:pixelate({cc.MAX_DIMENSION + 1})/{SRC}").segments[0].error,
        )
        self.assertEqual(
            cc.explain_url(f"/unsafe/{cc.MAX_DIMENSION + 1}x0/{SRC}").errors,
            [
                "malformed size segment: 16384x0 (each side is a pixel count up to 16383, 0, or orig; the server answers 400)"
            ],
        )


class TestNumbers(unittest.TestCase):
    """numbers.test.ts (issue #712): the explainer reads every number
    as the server's Rust types read it."""

    def test_read_decimal_rounds_to_the_nearest_32_bit_float_exactly(self):
        # Each decimal sits next to the midpoint between two 32-bit
        # floats. Rounding to a double first lands on the midpoint,
        # and the tie then goes to the even float, the wrong one.
        misread = [
            ("-967.109100341796874999999999999999990000000000000000000000000000", -967.1090698242188),
            ("-134.516242980957031250000000100000000000000000000000000000000000", -134.51625061035156),
            ("566.857391357421874999999999999999900000000000000000000000000000", 566.8573608398438),
            ("966.323883056640625000000000000001000000000000000000000000000000", 966.3239135742188),
            ("701.345001220703125000000000100000000000000000000000000000000000", 701.3450317382812),
            # Just above the midpoint between 512 and the next float:
            # the server reads 512.00006, past the blur bound.
            ("512.0000305175781250000000000000001", 512.00006103515625),
            # Just below the midpoint past the largest float: the
            # server reads f32::MAX, a sharpen amount, not infinity.
            ("3.4028235677973366e38", 3.4028234663852886e38),
        ]
        for text, want in misread:
            self.assertNotEqual(f32(float(text)), want, f"{text} needs no exact reading")
            self.assertEqual(cc._explain.read_decimal(text, 32), want, text)
        # The other side of the same midpoints.
        self.assertEqual(cc._explain.read_decimal("512.0000305175781249999999999999999", 32), 512)
        self.assertEqual(cc._explain.read_decimal("3.4028235677973367e38", 32), math.inf)

    def test_read_decimal_reads_rusts_float_syntax_and_nothing_else(self):
        read = [
            ("2", 2),
            ("+2", 2),
            ("2.", 2),
            (".5", 0.5),
            ("+.5e1", 5),
            ("2e0", 2),
            ("1E+2", 100),
            ("007", 7),
            ("-0", -0.0),
            ("-1e-50", -0.0),
            ("1e-46", 0.0),
            ("1e-45", 1.401298464324817e-45),
            ("1e39", math.inf),
            ("-1e39", -math.inf),
            ("1e999999999999999999999", math.inf),
            ("1e-999999999999999999999", 0.0),
        ]
        for text, want in read:
            got = cc._explain.read_decimal(text, 32)
            self.assertTrue(same_float(got, want), f"{text}: {got}")
        refused = [
            "",
            ".",
            "+",
            "e5",
            "1e",
            "1e+",
            "1.5.5",
            "+-1",
            " 1",
            "1 ",
            "0x10",
            "1_0",
            "inf",
            "-infinity",
            "NaN",
            "1\n",
            "\u0661",
            "1\u0660",
        ]
        for text in refused:
            self.assertIsNone(cc._explain.read_decimal(text, 32), repr(text))
        # The 64-bit reading is the double Python's own parser gives,
        # which is correctly rounded, as JavaScript's is.
        for text in (
            "0.1",
            "86400.000000000000000001",
            "1024.0000000000001",
            "2.2250738585072011e-308",
            "1e309",
        ):
            self.assertEqual(cc._explain.read_decimal(text, 64), float(text), text)

    def test_read_decimal_reads_a_decimal_as_long_as_the_url_in_linear_time(self):
        # The digits after the thousandth only decide a sticky digit,
        # and no digit run reaches int() whole: Python refuses to
        # convert one past 4300 digits.
        long = [
            f"1.{'1' * 64000}",
            f"0.{'0' * 32000}1{'0' * 32000}e32010",
            f"{'9' * 64000}e-63990",
        ]
        for text in long:
            value, seconds = timed(cc._explain.read_decimal, text, 32)
            self.assertLess(seconds, 1, f"{len(text)} characters took {seconds:.3f} s")
            self.assertTrue(value is not None and math.isfinite(value), value)
        self.assertEqual(cc._explain.read_decimal(f"1.{'1' * 64000}", 32), f32(1.1111111111111112))

    def test_the_explainer_holds_a_numbers_bounds_to_its_rounded_value(self):
        for ok in (
            "filters:blur(512.00001)",
            "filters:dpr(1.50000001)",
            "focalpoint:1.00000001x0.5",
            "filters:rotate(36000.001)",
            "filters:sharpen(3.4028235e38)",
            "filters:format(webp,method:+6)",
        ):
            self.assertEqual(errors(ok), [], ok)
        for bad in (
            "filters:blur(512.0001)",
            "filters:blur(1e-40)",
            "filters:gamma(1e-46)",
            "filters:sharpen(1e39)",
            "filters:trim(-0)",
            "crop-rect:-0x0:10x10",
        ):
            self.assertEqual(len(errors(bad)), 1, bad)

    def test_integers_take_a_sign_and_leading_zeros_as_rust_reads_them(self):
        # A digit run longer than Python converts is still a number:
        # one past every range.
        for ok in (
            "filters:trim(+020)",
            "filters:brighten(-0)",
            "filters:huerotate(-2147483648)",
            f"filters:pixelate({'0' * 5000}16383)",
            "filters:max_bytes(18446744073709551615)",
            "page:007",
            f"maxage:{'0' * 5000}60",
            "crop-rect:+0x0:+10x10",
        ):
            self.assertEqual(errors(ok), [], ok)
        for bad in (
            "filters:huerotate(2147483648)",
            "filters:max_bytes(18446744073709551616)",
            f"filters:pixelate({'1' * 5000})",
            f"exp:{'9' * 5000}",
            f"maxage:{'1' * 5000}",
            "filters:noise(\u0661)",
            "page:0",
            "page:1000000",
        ):
            self.assertEqual(len(errors(bad)), 1, bad)

    def test_a_class_name_is_letters_or_digits_as_rusts_is_alphanumeric_reads_them(self):
        # The Unicode Alphabetic property holds more than the letter
        # categories: Devanagari vowel signs, U+0345, circled letters.
        for name in ("हिंदी", "a\u0345", "Ⓐ", "Ⅰ", "traffic light", "e-bike_2.0/x"):
            self.assertEqual(errors(f"filters:blurclass({quote(name, safe='')})"), [], name)
        for name in ("✓", "a\u200db", "a*b", "", "é" * 33):
            self.assertEqual(len(errors(f"filters:blurclass({quote(name, safe='')})")), 1, name)

    def test_the_explainer_refuses_what_the_server_refuses_before_the_parse_and_after_it(self):
        def reserved(remainder):
            return cc.explain_url(f"https://x/{TAG}/{remainder}").errors

        self.assertRegex(
            reserved("crabcrop/purge-key")[0], r"reserved segment crabcrop.*400 \(reserved_segment\)"
        )
        self.assertRegex(
            reserved("purge/300x200/a.jpg")[0], r"purge or upload segment.*400 \(reserved_segment\)"
        )
        self.assertEqual(reserved("crabcrop-assets/a.jpg"), [])

        def inverts(n):
            return "filters:" + ":".join(["invert"] * n)

        self.assertEqual(errors(inverts(100)), [])
        self.assertEqual(
            errors(inverts(101)),
            [
                "the request has 101 operations, over the 100 operation limit: the server answers 400 (unsupported_operation)"
            ],
        )
        # A preset adds at least one operation the explainer cannot see.
        self.assertRegex(errors(f"preset:thumb/{inverts(100)}")[0], "at least 101 operations")

        def watermarks(n):
            return "filters:" + ":".join(f"watermark(w{i}.png,0,0,50)" for i in range(n))

        self.assertEqual(errors(watermarks(16)), [])
        self.assertEqual(errors(f"{watermarks(16)}:watermark(w%30.png,9,9,50)"), [])
        self.assertEqual(
            errors(watermarks(17)),
            [
                "the request names 17 distinct watermark and overlay URLs, over the 16 URL limit: the "
                "server answers 400 (unsupported_operation)"
            ],
        )
        # The caps apply to a request that parses, as on the server.
        self.assertEqual(errors(f"{inverts(101)}/filters:bogus"), ["unknown filter: bogus"])

    def test_escapes_decode_as_decode_uri_component_decodes_them(self):
        # `%` and two hex digits, the bytes UTF-8 (issue #712):
        # urllib's unquote would pass every one of these refusals.
        for ok in ("%41", "%c3%A9", "a%20b", "%E2%92%B6", "%F0%9F%98%80"):
            self.assertIsNotNone(cc._explain._percent_decode(ok), ok)
        self.assertEqual(cc._explain._percent_decode("traffic%20light%e2%92%b6"), "traffic lightⒶ")
        for bad in (
            "%",
            "%4",
            "%zz",
            "%+1",
            "%C3",
            "%C3A",
            "%C0%80",
            "%ED%A0%80",
            "%F4%90%80%80",
            "%80",
            "%FF",
        ):
            self.assertIsNone(cc._explain._percent_decode(bad), bad)
            self.assertEqual(len(errors(f"filters:watermark({bad},0,0,50)")), 1, bad)


class TestLimits(unittest.TestCase):
    """explain-limits.test.ts (issue #594): a float argument costs time
    linear in its length, and the explainer reads at most 64 KiB."""

    FLOAT_SITES = (
        "filters:blur(#)",
        "filters:sharpen(#)",
        "filters:contrast(#)",
        "filters:gamma(#)",
        "filters:exposure(#)",
        "filters:highlights_shadows(#,0)",
        "filters:vibrance(#)",
        "filters:saturation(#)",
        "filters:facepad(#)",
        "filters:dpr(#)",
        "focalpoint:#x0.5",
        "focalpoint:0.5x#",
        "focalpoint:0.5x0.5:#",
        "crop:100x100:objects(person):#",
    )

    def test_a_long_digit_run_in_a_float_argument_costs_linear_time(self):
        digits = "1" * 65000
        for site in self.FLOAT_SITES:
            for tail in ("z", "e", ".z"):
                url = f"https://x/{TAG}/{site.replace('#', digits + tail)}/https://e.com/a.jpg"
                self.assertLessEqual(len(url), 64 * 1024)
                out, seconds = timed(cc.explain_url, url)
                self.assertLess(seconds, 1, f"{site} with {tail}: explain_url took {seconds:.3f} s")
                self.assertEqual(len(out.errors), 1, f"{site} with {tail}")
                self.assertNotIn("longer than 64 KiB", out.errors[0])
                verdict, seconds = timed(cc.verify_url_signature, "secret", url)
                self.assertLess(seconds, 1, f"{site} with {tail}: verify_url_signature took {seconds:.3f} s")
                self.assertIs(verdict, False)

    def test_the_float_grammar_accepts_and_rejects_what_it_did_before(self):
        # The pattern the TypeScript explainer replaced, kept as the
        # reference on short strings. sharpen reads a float above 0
        # and contrast one in -100..100, so the two together show the
        # verdict on every string whose value is above -100; both read
        # the decimal as the 32-bit float the server reads.
        old = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", re.ASCII)

        def accepts(s):
            return not errors(f"filters:sharpen({s})", "https://e.com/a.jpg") or not errors(
                f"filters:contrast({s})", "https://e.com/a.jpg"
            )

        def reference(s):
            if old.fullmatch(s) is None:
                return False
            f = f32(float(s))
            return math.isfinite(f) and f >= -100

        strings = [""]
        checked = 0
        for _ in range(5):
            strings = [s + c for s in strings for c in "10.eE+-z"]
            for s in strings:
                self.assertEqual(accepts(s), reference(s), repr(s))
                checked += 1
        self.assertEqual(checked, 37448)
        for s in ("1", "1.", ".5", "1.5", "1e5", "1.e5", ".5e-3", "+1", "-1", "007", "1E+2"):
            self.assertTrue(accepts(s), s)
        for s in ("", ".", "e5", "1e", "1.5.5", "+-1", " 1", "1 ", "0x10", "inf", "NaN", "1e999", "1_0"):
            self.assertFalse(accepts(s), repr(s))

    def test_explain_url_reads_at_most_64_kib(self):
        secret = "secret"
        cap = 64 * 1024
        head = f"https://x/{'0' * 64}/300x200/https://e.com/"

        def fill(size):
            return f"{head}{'a' * (size - len(head) - 4)}.jpg"

        def signed(url):
            remainder = url[len("https://x/") + 65 :]
            return url.replace("0" * 64, cc.sign(secret, remainder))

        # A signed URL of exactly the cap explains and verifies.
        at_cap = signed(fill(cap))
        self.assertEqual(len(at_cap), cap)
        ok = cc.explain_url(at_cap)
        self.assertEqual(ok.errors, [])
        self.assertEqual(ok.signature.kind, "hmac")
        self.assertIs(cc.verify_url_signature(secret, at_cap), True)

        # One byte over: one error, nothing else read, and the
        # verifier answers None, as for any malformed URL, even
        # though the tag is right for its remainder.
        over = signed(fill(cap + 1))
        out = cc.explain_url(over)
        self.assertEqual(
            out.errors,
            ["the URL is longer than 64 KiB (65536 bytes): the explainer does not read it"],
        )
        self.assertEqual((out.signature, out.segments, out.source, out.remainder), (None, [], "", ""))
        self.assertIsNone(cc.verify_url_signature(secret, over))
        self.assertIsNone(cc.verify_url_signature({"k": secret}, over))

        # The cap counts UTF-8 bytes, as the server reads them, not
        # characters: one `é` is two bytes, and a lone surrogate the
        # three TextEncoder writes for it.
        wide = f"{fill(cap - 1)[:-4]}é.jpg"
        self.assertEqual(len(wide), cap)
        self.assertIn("longer than 64 KiB", cc.explain_url(wide).errors[0])
        wide_ok = f"{fill(cap - 1)[:-5]}é.jpg"
        self.assertEqual(len(wide_ok.encode("utf-8")), cap)
        self.assertNotIn("longer than 64 KiB", "\n".join(cc.explain_url(wide_ok).errors))
        lone = f"{fill(cap - 2)[:-5]}\ud800.jpg"
        self.assertEqual(len(lone.encode("utf-8", "surrogatepass")), cap)
        self.assertNotIn("longer than 64 KiB", "\n".join(cc.explain_url(lone).errors))
        self.assertIn("longer than 64 KiB", cc.explain_url(lone + "x").errors[0])

        # The issue's own input, 100,000 digits: refused at once.
        long = f"https://x/{TAG}/filters:blur({'1' * 100_000}z)/{SRC}"
        refused, seconds = timed(cc.explain_url, long)
        self.assertLess(seconds, 1)
        self.assertEqual(len(refused.errors), 1)
        self.assertIn("longer than 64 KiB", refused.errors[0])
        self.assertIsNone(cc.verify_url_signature(secret, long))


class TestTrim(unittest.TestCase):
    """trim.test.ts, through the explainer: it trims its input and each
    class name as Rust's ``str::trim`` does, the Unicode White_Space
    set and nothing else."""

    WHITE_SPACE = (
        "\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007"
        "\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
    )
    # Python's str.strip removes the first four; the rest are not
    # White_Space either (U+180E has not been since Unicode 6.3).
    KEPT = "\x1c\x1d\x1e\x1f\ufeff\u200b\u180e\u2060"

    def test_the_input_is_trimmed_of_white_space_and_nothing_else(self):
        url = f"https://img.example.com/{TAG}/300x200/{SRC}"
        bare = cc.explain_url(url)
        self.assertEqual(bare.errors, [])
        for c in self.WHITE_SPACE:
            self.assertEqual(cc.explain_url(f"{c}{c}{url}{c}"), bare, f"U+{ord(c):04X}")
        for c in self.KEPT:
            self.assertNotEqual(cc.explain_url(f"{c}{url}{c}"), bare, f"U+{ord(c):04X}")

    def test_a_class_name_is_trimmed_of_white_space_and_nothing_else(self):
        for c in self.WHITE_SPACE:
            self.assertEqual(
                errors(f"filters:blurclass({quote(c + 'person' + c, safe='')})"), [], f"U+{ord(c):04X}"
            )
        for c in self.KEPT:
            self.assertEqual(
                len(errors(f"filters:blurclass({quote(c + 'person', safe='')})")), 1, f"U+{ord(c):04X}"
            )

    def test_a_long_run_of_white_space_costs_linear_time(self):
        # Two runs of 30,000 fit under the 64 KiB the explainer reads.
        run = " " * 30000
        url = f"https://img.example.com/{TAG}/300x200/{SRC}"
        for given in (f"{run}{url}", f"{url}{run}", f"{run}{url}{run}"):
            out, seconds = timed(cc.explain_url, given)
            self.assertEqual(out.errors, [])
            self.assertLess(seconds, 1)


class TestGrammarSets(unittest.TestCase):
    def test_every_member_of_every_shared_set_builds_a_segment_the_explainer_accepts(self):
        # grammar-sets.test.ts: the builders and the explainer read
        # the closed sets from one place, _grammar.py.
        g = cc._grammar
        f = cc.filter
        calls = (
            [f.text("hi", 12, "ffffff", anchor) for anchor in g.TEXT_ANCHORS]
            + [f.format(fmt) for fmt in g.IMAGE_FORMATS]
            + [f.quality_per_format({fmt: 70}) for fmt in g.QUALITY_FORMATS]
            + [f.format("jpeg", subsampling=s) for s in g.JPEG_SUBSAMPLINGS]
            + [f.overlay("https://example.com/o.png", 0, 0, 50, blend) for blend in g.BLEND_MODES]
            + [f.curve(channel, [(0, 0), (255, 255)]) for channel in g.CURVE_CHANNELS]
            + [f.studio(mode) for mode in g.STUDIO_MODES]
            + [f.studio("product", bg=bg) for bg in g.STUDIO_BACKDROPS]
            + [f.studio("product", shadow=shadow) for shadow in g.STUDIO_SHADOWS]
            + [f.upscale(factor) for factor in g.UPSCALE_FACTORS]
            + [f.dpr(factor) for factor in g.DPR_FACTORS]
            + [f.mask(shape) for shape in g.MASK_SHAPES]
            + [f.flip(axis) for axis in g.FLIP_AXES]
        )
        segments = [cc.ops.crop(100, 100, align) for align in g.CROP_ALIGNS] + [f"filters:{c}" for c in calls]
        for segment in segments:
            self.assertEqual(cc.explain_url(f"/unsafe/{segment}/{SRC}").errors, [], segment)


def typescript_constant(name):
    match = re.search(rf"^export const {name} = (\d+);$", EXPLAIN_TS.read_text(encoding="utf-8"), re.M)
    if match is None:
        raise AssertionError(f"explain.ts has no `export const {name}`")
    return int(match.group(1))


class TestTables(unittest.TestCase):
    """What only the explainer holds, read back from explain.ts."""

    def test_the_filter_descriptions_are_the_typescript_sdks(self):
        text = EXPLAIN_TS.read_text(encoding="utf-8")
        body = re.search(
            r"^export const FILTER_DESCRIPTIONS: Record<string, string> = \{\n(.*?)^\};", text, re.M | re.S
        ).group(1)
        entries = re.findall(r'^\s*([a-z_0-9]+):\s*("(?:[^"\\\n]|\\.)*"),?\s*$', body, re.M)
        self.assertEqual(
            len(entries), len([line for line in body.splitlines() if line.strip().endswith(",")])
        )
        want = {name: ast.literal_eval(value) for name, value in entries}
        self.assertGreaterEqual(len(want), 50)
        self.assertEqual(dict(cc.FILTER_DESCRIPTIONS), want)
        # The table drives the unknown-filter verdict, so it is read
        # only.
        with self.assertRaises(TypeError):
            cc.FILTER_DESCRIPTIONS["bogus"] = "a filter of my own"  # type: ignore[index]

    def test_every_described_filter_has_the_servers_argument_check(self):
        explain = cc._explain
        described = set(cc.FILTER_DESCRIPTIONS)
        self.assertEqual(described, explain._NO_ARG_FILTERS | set(explain._FILTER_ARGUMENTS))
        self.assertEqual(explain._NO_ARG_FILTERS & set(explain._FILTER_ARGUMENTS), set())

    def test_the_caps_are_the_typescript_sdks(self):
        self.assertEqual(cc._explain.MAX_OPERATIONS, typescript_constant("MAX_OPERATIONS"))
        self.assertEqual(cc._explain.MAX_WATERMARK_URLS, typescript_constant("MAX_WATERMARK_URLS"))
        text = EXPLAIN_TS.read_text(encoding="utf-8")
        self.assertIn("const MAX_EXPLAIN_BYTES = 64 * 1024;", text)
        self.assertEqual(cc._explain.MAX_EXPLAIN_BYTES, 64 * 1024)
        self.assertIn("const KEPT_DIGITS = 1000;", text)
        self.assertEqual(cc._explain._KEPT_DIGITS, 1000)


class TestHostileInput(unittest.TestCase):
    """The explainer reads untrusted URLs: no input makes it raise or
    stall."""

    def test_random_input_never_raises(self):
        rng = random.Random(594)
        alphabet = list("0123456789abcdefx:/,()%+-.eE?#_ ") + [
            "\x00",
            "\ud800",
            "\udfff",
            "é",
            "😀",
            "\u2028",
            "\ufeff",
        ]
        pieces = [
            "filters:",
            "crop:",
            "exp:",
            "s1.",
            "k1:",
            TAG,
            "unsafe/",
            "https://",
            "(" * 50,
            ")" * 50,
            "9" * 400,
        ]
        for _ in range(3000):
            text = "".join(
                rng.choice(alphabet if rng.random() < 0.9 else pieces) for _ in range(rng.randrange(80))
            )
            out = cc.explain_url(text)
            self.assertIsInstance(out, cc.Explanation)
            self.assertIn(cc.verify_url_signature("secret", text), (None, True, False))
            self.assertIn(cc.verify_url_signature({"k1": "secret"}, text), (None, True, False))

    def test_a_url_that_is_not_text_is_refused(self):
        for given in (b"https://x/unsafe/300x200/a.jpg", None, 7):
            with self.assertRaises(TypeError):
                cc.explain_url(given)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
