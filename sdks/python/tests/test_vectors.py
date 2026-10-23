"""The package's signing suite is the canonical vector file
from the spec directory, shared by every implementation."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crabcrop_url import build_srcset, build_url, sign, verify  # noqa: E402

VECTORS = json.loads(
    (Path(__file__).resolve().parents[3] / "spec" / "test-vectors.json").read_text()
)


class TestVectors(unittest.TestCase):
    def test_all_vectors(self):
        self.assertGreaterEqual(len(VECTORS), 5)
        for v in VECTORS:
            with self.subTest(description=v["description"]):
                self.assertEqual(sign(v["secret"], v["remainder"]), v["signature"])

    def test_verify_accepts_the_vectors_and_rejects_tampering(self):
        for v in VECTORS:
            with self.subTest(description=v["description"]):
                self.assertTrue(verify(v["secret"], v["signature"], v["remainder"]))
                self.assertFalse(verify(v["secret"], v["signature"], v["remainder"] + "x"))
                self.assertFalse(verify("wrong-secret", v["signature"], v["remainder"]))
                # A non-canonical tag fails: the server
                # accepts only the lowercase spelling.
                self.assertFalse(
                    verify(v["secret"], v["signature"].upper(), v["remainder"])
                )
        self.assertFalse(verify("s", "not-a-signature", "300x200/https://e.com/a.jpg"))


class TestBuilders(unittest.TestCase):
    def test_spec_worked_example(self):
        self.assertEqual(
            build_url(
                "https://images.example.com/",
                "your-secret",
                "300x200",
                "https://example.com/photo.jpg",
            ),
            "https://images.example.com/"
            "3c1f6c180116661877bd4c85fd201a178087d74c1cc188cc584e819d901308c6"
            "/300x200/https://example.com/photo.jpg",
        )

    def test_expiry_and_null_secret(self):
        self.assertEqual(
            build_url(
                "https://img.example.com",
                None,
                "preset:thumb",
                "s3://media/a.jpg",
                expires=1893456000,
            ),
            "https://img.example.com/unsafe/exp:1893456000/preset:thumb/s3://media/a.jpg",
        )

    def test_srcset(self):
        self.assertEqual(
            build_srcset(
                "https://img.example.com", None, "https://example.com/a.jpg", [320, 640]
            ),
            "https://img.example.com/unsafe/320x0/https://example.com/a.jpg 320w, "
            "https://img.example.com/unsafe/640x0/https://example.com/a.jpg 640w",
        )

    def test_srcset_rejects_widths_the_grammar_cannot_express(self):
        # 320.5 is not an integer, -300 is the flip spelling,
        # 0 is not a size, and 16384 is past the server cap:
        # each would mint a URL the server rejects.
        for bad in (320.5, -300, 0, 16384, True):
            with self.assertRaises(ValueError):
                build_srcset(
                    "https://img.example.com", None, "https://example.com/a.jpg", [bad]
                )
        build_srcset(
            "https://img.example.com", None, "https://example.com/a.jpg", [1, 16383]
        )


    def test_builders_reject_unbuildable_urls(self):
        # Audit P-4 and P-5: these can never verify, so the
        # builders raise instead of minting dead URLs.
        for source in (
            "https://example.com/a.jpg?w=1",
            "https://example.com/a.jpg#frag",
            "https://example.com/my photo.jpg",
            "https://example.com/café.jpg",
        ):
            with self.assertRaises(ValueError, msg=source):
                build_url("https://img.example.com", b"k", "300x200", source)
        for bad_expires in (-1, 1.5, True, 2**64):
            with self.assertRaises(ValueError, msg=repr(bad_expires)):
                build_url(
                    "https://img.example.com",
                    b"k",
                    "300x200",
                    "https://example.com/a.jpg",
                    expires=bad_expires,
                )


if __name__ == "__main__":
    unittest.main()
