"""The package's signing suite is the canonical vector file
from the spec directory, shared by every implementation."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crabcrop_url import build_srcset, build_url, sign  # noqa: E402

VECTORS = json.loads(
    (Path(__file__).resolve().parents[3] / "spec" / "test-vectors.json").read_text()
)


class TestVectors(unittest.TestCase):
    def test_all_vectors(self):
        self.assertGreaterEqual(len(VECTORS), 5)
        for v in VECTORS:
            with self.subTest(description=v["description"]):
                self.assertEqual(sign(v["secret"], v["remainder"]), v["signature"])


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


if __name__ == "__main__":
    unittest.main()
