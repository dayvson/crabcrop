"""The package's signing suite is the canonical vector file
from the spec directory, shared by every implementation."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crabcrop_url import (  # noqa: E402
    SealError,
    build_sealed_url,
    build_srcset,
    build_url,
    purge_group_signature,
    purge_signature,
    seal_url,
    seal_with_nonce,
    sign,
    unseal,
    upload_signature,
    sign_with_key_id,
    verify,
)

VECTORS = json.loads(
    (Path(__file__).resolve().parents[3] / "spec" / "test-vectors.json").read_text()
)

SIGNED_VECTORS = [v for v in VECTORS if "signature" in v]
SEALED_VECTORS = [v for v in VECTORS if v.get("sealed")]
HAS_CRYPTOGRAPHY = __import__("importlib").util.find_spec("cryptography") is not None


class TestVectors(unittest.TestCase):
    def test_all_vectors(self):
        self.assertGreaterEqual(len(SIGNED_VECTORS), 5)
        for v in SIGNED_VECTORS:
            with self.subTest(description=v["description"]):
                self.assertEqual(sign(v["secret"], v["remainder"]), v["signature"])

    def test_verify_accepts_the_vectors_and_rejects_tampering(self):
        for v in SIGNED_VECTORS:
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


class TestPurgeSignatures(unittest.TestCase):
    def test_published_vectors(self):
        # The same pinned literals the Rust crate and the
        # TypeScript SDK assert.
        self.assertEqual(
            purge_signature("test-secret", "300x200/https://example.com/a.jpg"),
            "9b58a43a984e47cbde3ec08333290acad06c48a5e9b85b99124ae725218e90e8",
        )
        self.assertEqual(
            purge_group_signature("test-secret", "thumb"),
            "256634813bd6237179d251ba4526685f05db8a42ae056e4a04566f00042f0439",
        )
        # Purge-group signs under its own subkey: a
        # per-target purge signature never transplants to it.
        self.assertNotEqual(
            purge_group_signature("test-secret", "thumb"),
            purge_signature("test-secret", "purge-group/thumb"),
        )
        self.assertNotEqual(
            purge_group_signature("test-secret", "thumb"),
            purge_signature("test-secret", "thumb"),
        )


class TestUploadSignatures(unittest.TestCase):
    def test_published_vectors(self):
        # The same pinned literals the Rust crate and the
        # TypeScript SDK assert.
        self.assertEqual(
            upload_signature("test-secret", "heroes/cat.jpg"),
            "81d87b588203fdad32f5431c0a893bf7ddad8188f5cca123a1b057a439b34b84",
        )
        self.assertNotEqual(
            upload_signature("test-secret", "heroes/cat.jpg"),
            purge_signature("test-secret", "heroes/cat.jpg"),
        )

    def test_expiring_form_vector(self):
        # The expiring form signs ``exp:<expires>/<name>``:
        # the same pinned value the Rust crate asserts,
        # bound to both the expiry and the name.
        self.assertEqual(
            upload_signature("test-secret", "heroes/cat.jpg", 1893456000),
            "391061e9bd262811f9bac647b5ff535f987e2478394675407f895823f45e59d2",
        )
        self.assertNotEqual(
            upload_signature("test-secret", "heroes/cat.jpg", 1893456000),
            upload_signature("test-secret", "heroes/cat.jpg"),
        )
        self.assertNotEqual(
            upload_signature("test-secret", "heroes/cat.jpg", 1893456000),
            upload_signature("test-secret", "heroes/dog.jpg", 1893456000),
        )


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

    def test_root_source_builds_signed_as_written(self):
        # A short source (issue #248) is signed exactly as
        # written; the signature is the spec's canonical
        # vector for this remainder.
        self.assertEqual(
            build_url(
                "https://img.example.com",
                "s3cr3t",
                "300x200",
                "root:photos/cat.jpg",
            ),
            "https://img.example.com/"
            "47e002f609cfbfc252eb1357bbee780bbbaf4c336847610caf44617b06453824"
            "/300x200/root:photos/cat.jpg",
        )
        # The character refusals still bite a short source.
        with self.assertRaises(ValueError):
            build_url(
                "https://img.example.com", b"k", "300x200", "root:photos/cat.jpg?w=1"
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


class TestKeyIds(unittest.TestCase):
    def test_sign_with_key_id_stamps_the_segment(self):
        remainder = "300x200/example.com/i.jpg"
        self.assertEqual(
            sign_with_key_id("test-secret", "k2", remainder),
            f"k2:{sign('test-secret', remainder)}",
        )
        self.assertEqual(
            build_url(
                "https://img.example.com",
                "test-secret",
                "300x200",
                "example.com/i.jpg",
                key_id="k2",
            ),
            f"https://img.example.com/k2:{sign('test-secret', remainder)}/300x200/example.com/i.jpg",
        )
        with self.assertRaises(ValueError):
            sign_with_key_id("test-secret", "bad id", remainder)
        with self.assertRaises(ValueError):
            build_url(
                "https://img.example.com",
                None,
                "300x200",
                "example.com/i.jpg",
                key_id="k2",
            )

    def test_verify_accepts_a_keyed_segment_for_its_secret(self):
        remainder = "300x200/example.com/i.jpg"
        segment = sign_with_key_id("old-secret", "k-old", remainder)
        self.assertTrue(verify("old-secret", segment, remainder))
        self.assertFalse(verify("new-secret", segment, remainder))
        self.assertFalse(verify("old-secret", "bad id:" + sign("old-secret", remainder), remainder))


@unittest.skipUnless(HAS_CRYPTOGRAPHY, "sealed vectors need the cryptography extra")
class TestSealedVectors(unittest.TestCase):
    """The sealed vectors from the shared file (issue #255):
    AES-GCM is a deterministic function of (key, nonce,
    plaintext, AAD), so an injected nonce pins the token
    byte for byte, and every token opens back."""

    def test_sealed_vectors(self):
        self.assertGreaterEqual(len(SEALED_VECTORS), 3)
        for v in SEALED_VECTORS:
            with self.subTest(description=v["description"]):
                nonce = bytes.fromhex(v["nonce"])
                self.assertEqual(
                    seal_with_nonce(v["secret"], v["remainder"], nonce, v.get("key_id")),
                    v["token"],
                )
                self.assertEqual(unseal(v["secret"], v["token"]), v["remainder"])
                with self.assertRaises(SealError):
                    unseal("wrong-secret", v["token"])

    def test_round_trip_and_builders(self):
        remainder = "300x200/https://example.com/photo.jpg"
        token = seal_url("test-secret", remainder)
        self.assertTrue(token.startswith("s1."))
        self.assertEqual(unseal("test-secret", token), remainder)
        keyed = seal_url("test-secret", remainder, key_id="k2")
        self.assertTrue(keyed.startswith("k2:s1."))
        self.assertEqual(unseal("test-secret", keyed), remainder)
        self.assertNotEqual(seal_url("test-secret", remainder), token)
        damaged = token[:-1] + ("B" if token.endswith("A") else "A")
        with self.assertRaises(SealError):
            unseal("test-secret", damaged)
        with self.assertRaises(SealError):
            unseal("test-secret", "s2.AAAA")
        url = build_sealed_url(
            "https://img.example.com/",
            "test-secret",
            "300x200",
            "https://example.com/photo.jpg",
            name="photo.jpg",
        )
        path = url[len("https://img.example.com/"):]
        segment, _, name = path.partition("/")
        self.assertEqual(name, "photo.jpg")
        self.assertEqual(unseal("test-secret", segment), remainder)
        with self.assertRaises(SealError):
            build_sealed_url(
                "https://img.example.com", None, "300x200", "https://example.com/photo.jpg"
            )
        with self.assertRaises(SealError):
            seal_url("test-secret", "a" * 2049)


class TestSealedWithoutExtra(unittest.TestCase):
    """The rest of the package stays zero-dependency: with
    the extra absent, the seal functions raise the error
    that names the install, and nothing else changes."""

    @unittest.skipIf(HAS_CRYPTOGRAPHY, "only meaningful without the extra installed")
    def test_missing_extra_error_is_clear(self):
        with self.assertRaises(SealError) as ctx:
            seal_url("test-secret", "300x200/https://example.com/photo.jpg")
        self.assertIn("crabcrop-url[seal]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
