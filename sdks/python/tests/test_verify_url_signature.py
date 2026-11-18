"""``verify_url_signature`` (issue #716), the port of the TypeScript
SDK's ``verifyUrlSignature``: the spec's signing vectors as URLs, the
keyring rules of sdks/typescript/test/vectors.test.ts, the reserved
remainders, and the answers for what is not a signed URL."""

import hashlib
import hmac
import json
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import crabcrop_url as cc  # noqa: E402

VECTORS = json.loads(
    (Path(__file__).resolve().parents[3] / "spec" / "test-vectors.json").read_text(encoding="utf-8")
)
BASE = "https://img.example.com"
# The signed vectors whose remainder is a URL path as written: the
# first vector's remainder ends in `?`, which a URL reads as the start
# of its query (test_a_query_is_not_signed covers it).
URL_VECTORS = [
    v for v in VECTORS if "signature" in v and "?" not in v["remainder"] and "#" not in v["remainder"]
]
RESERVED_VECTORS = [v for v in VECTORS if v.get("reserved")]


def raw_tag(secret, remainder):
    """HMAC-SHA256 of a remainder, which `sign` refuses to compute for
    a reserved one."""
    return hmac.new(secret.encode("utf-8"), remainder.encode("utf-8"), hashlib.sha256).hexdigest()


class TestVectors(unittest.TestCase):
    def test_the_spec_vectors_verify_as_urls(self):
        self.assertGreaterEqual(len(URL_VECTORS), 20)
        for v in URL_VECTORS:
            url = f"{BASE}/{v['signature']}/{v['remainder']}"
            with self.subTest(description=v["description"]):
                self.assertIs(cc.verify_url_signature(v["secret"], url), True)
                self.assertIs(cc.verify_url_signature(v["secret"].encode("utf-8"), url), True)
                self.assertIs(cc.verify_url_signature("wrong-secret", url), False)
                # A bare path reads as the URL does.
                self.assertIs(
                    cc.verify_url_signature(v["secret"], f"/{v['signature']}/{v['remainder']}"), True
                )
                # A tampered remainder fails; a query string is not
                # signed, so adding one changes nothing.
                self.assertIs(cc.verify_url_signature(v["secret"], url + "x"), False)
                self.assertIs(cc.verify_url_signature(v["secret"], url + "?w=1#top"), True)

    def test_a_query_is_not_signed(self):
        # The first vector signs `what do ye want for nothing?`; as a
        # URL, the `?` starts the query, so the signed path is shorter
        # and the tag no longer matches it.
        jefe = VECTORS[0]
        self.assertTrue(jefe["remainder"].endswith("?"))
        url = f"{BASE}/{jefe['signature']}/{jefe['remainder']}"
        self.assertEqual(cc.explain_url(url).remainder, jefe["remainder"][:-1])
        self.assertIs(cc.verify_url_signature(jefe["secret"], url), False)

    def test_one_secret_ignores_the_key_id(self):
        for v in URL_VECTORS:
            url = f"{BASE}/k2:{v['signature']}/{v['remainder']}"
            with self.subTest(description=v["description"]):
                self.assertIs(cc.verify_url_signature(v["secret"], url), True)
                self.assertIs(cc.verify_url_signature("wrong-secret", url), False)


class TestKeyring(unittest.TestCase):
    def test_a_keyed_url_consults_its_named_key_only(self):
        for v in URL_VECTORS:
            url = f"{BASE}/k2:{v['signature']}/{v['remainder']}"
            with self.subTest(description=v["description"]):
                self.assertIs(cc.verify_url_signature({"k2": v["secret"]}, url), True)
                self.assertIs(cc.verify_url_signature({"k1": "wrong", "k2": v["secret"]}, url), True)
                # The right secret under another ID does not count, and
                # an ID the ring does not hold is False.
                self.assertIs(cc.verify_url_signature({"k1": v["secret"], "k2": "wrong"}, url), False)
                self.assertIs(cc.verify_url_signature({"k1": v["secret"]}, url), False)
                self.assertIs(cc.verify_url_signature({}, url), False)

    def test_an_unkeyed_url_consults_every_key(self):
        for v in URL_VECTORS:
            url = f"{BASE}/{v['signature']}/{v['remainder']}"
            with self.subTest(description=v["description"]):
                self.assertIs(cc.verify_url_signature({"old": v["secret"], "new": "other"}, url), True)
                self.assertIs(cc.verify_url_signature({"old": "other", "new": v["secret"]}, url), True)
                self.assertIs(cc.verify_url_signature({"old": "other"}, url), False)
                self.assertIs(cc.verify_url_signature({}, url), False)

    def test_the_typescript_suites_rotation_cases(self):
        # vectors.test.ts, "verifyUrlSignature routes by key ID with a
        # keyring".
        remainder = "300x200/example.com/i.jpg"
        keyed = f"{BASE}/{cc.sign_with_key_id('old-secret', 'k-old', remainder)}/{remainder}"
        unkeyed = f"{BASE}/{cc.sign('old-secret', remainder)}/{remainder}"
        ring = {"k-old": "old-secret", "k-new": "new-secret"}
        self.assertIs(cc.verify_url_signature(ring, keyed), True)
        self.assertIs(cc.verify_url_signature({"k-new": "new-secret"}, keyed), False)
        self.assertIs(cc.verify_url_signature({"k-old": "new-secret"}, keyed), False)
        self.assertIs(cc.verify_url_signature(ring, unkeyed), True)
        self.assertIs(cc.verify_url_signature({"k-new": "other"}, unkeyed), False)
        self.assertIs(cc.verify_url_signature("old-secret", keyed), True)
        self.assertIs(cc.verify_url_signature("new-secret", keyed), False)

    def test_key_ids_named_like_object_properties_are_ordinary_ids(self):
        # vectors.test.ts: the TypeScript keyring is an object, so it
        # must read own properties only. A mapping has no inherited
        # keys, and these IDs route like any other.
        remainder = "300x200/example.com/i.jpg"
        for key_id in ("constructor", "toString", "__proto__", "hasOwnProperty", "__class__", "keys", "get"):
            url = f"{BASE}/{cc.sign_with_key_id('old-secret', key_id, remainder)}/{remainder}"
            with self.subTest(key_id=key_id):
                self.assertIs(cc.verify_url_signature({"k-old": "old-secret"}, url), False)
                self.assertIs(cc.verify_url_signature({key_id: "old-secret"}, url), True)

    def test_a_keyring_may_hold_bytes(self):
        remainder = "300x200/example.com/i.jpg"
        url = f"{BASE}/{cc.sign_with_key_id(b'old-secret', 'k1', remainder)}/{remainder}"
        self.assertIs(cc.verify_url_signature({"k1": b"old-secret"}, url), True)
        self.assertIs(cc.verify_url_signature({"k1": b"other"}, url), False)


class TestNotSigned(unittest.TestCase):
    def test_unsigned_sealed_and_malformed_urls_answer_none(self):
        tag = cc.sign("s3cr3t", "300x200/https://example.com/a.jpg")
        sealed = [v["token"] for v in VECTORS if v.get("sealed")]
        self.assertEqual(len(sealed), 3)
        for url in (
            [
                f"{BASE}/unsafe/300x200/https://example.com/a.jpg",
                f"{BASE}/photo.jpg",
                f"{BASE}/{tag[:40]}/300x200/https://example.com/a.jpg",
                f"{BASE}/{tag.upper()}/300x200/https://example.com/a.jpg",
                f"{BASE}/bad id:{tag}/300x200/https://example.com/a.jpg",
                f"{BASE}/s2.AAAA/300x200/https://example.com/a.jpg",
                "",
                f"{BASE}/",
            ]
            + [f"{BASE}/{token}" for token in sealed]
            + [f"{BASE}/{token}/photo.jpg" for token in sealed]
        ):
            with self.subTest(url=url):
                self.assertIsNone(cc.verify_url_signature("s3cr3t", url))
                self.assertIsNone(cc.verify_url_signature({"k2": "s3cr3t"}, url))

    def test_a_url_longer_than_the_explainer_reads_answers_none(self):
        remainder = "300x200/https://e.com/" + "a" * (64 * 1024) + ".jpg"
        url = f"https://x/{cc.sign('secret', remainder)}/{remainder}"
        self.assertIsNone(cc.verify_url_signature("secret", url))
        self.assertIsNone(cc.verify_url_signature({"k": "secret"}, url))


class TestReserved(unittest.TestCase):
    def test_a_reserved_remainder_is_false_whatever_its_tag(self):
        # The tag is the right HMAC of the remainder, which is a
        # derived subkey or could be one: no verifier confirms it.
        self.assertGreaterEqual(len(RESERVED_VECTORS), 9)
        for v in RESERVED_VECTORS:
            tag = raw_tag("s3cr3t", v["remainder"])
            with self.subTest(description=v["description"]):
                self.assertIs(cc.verify_url_signature("s3cr3t", f"{BASE}/{tag}/{v['remainder']}"), False)
                self.assertIs(
                    cc.verify_url_signature({"k": "s3cr3t"}, f"{BASE}/{tag}/{v['remainder']}"), False
                )
                self.assertIs(
                    cc.verify_url_signature({"k": "s3cr3t"}, f"{BASE}/k:{tag}/{v['remainder']}"), False
                )

    def test_no_url_carrying_a_subkey_verifies(self):
        # vectors.test.ts, "no signer returns a derived subkey".
        secret = "label-test-secret-0123456789"
        for label in (
            "crabcrop/purge-key",
            "crabcrop/purge-group-key",
            "crabcrop/upload-key",
            "crabcrop/seal-key",
            "crabcrop/entry-key",
        ):
            subkey = raw_tag(secret, label)
            with self.subTest(label=label):
                self.assertIs(cc.verify_url_signature(secret, f"{BASE}/{subkey}/{label}"), False)
                self.assertIs(cc.verify_url_signature({"k": secret}, f"{BASE}/k:{subkey}/{label}"), False)


class TestSurrogates(unittest.TestCase):
    """A Python string can hold surrogate code points, which UTF-8
    cannot encode (``surrogateescape`` makes one of each byte that is
    not UTF-8). The TypeScript SDK's HMAC hashes a lone surrogate as
    U+FFFD, as TextEncoder and node:crypto write it, and reads a pair
    as the character it encodes (crypto.test.ts pins both), so the
    verifier answers as it does, and raises for no text."""

    REMAINDER = "300x200/https://example.com/a.jpg"

    def test_a_lone_surrogate_in_the_url_hashes_as_the_replacement_character(self):
        lone = chr(0xDCFF)
        remainder = f"300x200/https://example.com/{lone}.jpg"
        tag = raw_tag("s3cr3t", remainder.replace(lone, "�"))
        self.assertIs(cc.verify_url_signature("s3cr3t", f"{BASE}/{tag}/{remainder}"), True)
        self.assertIs(cc.verify_url_signature({"k": "s3cr3t"}, f"{BASE}/{tag}/{remainder}"), True)
        self.assertIs(cc.verify_url_signature({"k": "s3cr3t"}, f"{BASE}/k:{tag}/{remainder}"), True)
        self.assertIs(cc.verify_url_signature("wrong", f"{BASE}/{tag}/{remainder}"), False)
        self.assertIs(cc.verify_url_signature("s3cr3t", f"{BASE}/{'a' * 64}/{remainder}"), False)

    def test_a_lone_surrogate_in_a_secret_hashes_the_same_way(self):
        secret = "s3cr3t" + chr(0xD800)
        tag = raw_tag("s3cr3t�", self.REMAINDER)
        url = f"{BASE}/{tag}/{self.REMAINDER}"
        self.assertIs(cc.verify_url_signature(secret, url), True)
        self.assertIs(cc.verify_url_signature({"a": "other", "b": secret}, url), True)
        self.assertIs(cc.verify_url_signature({"k": secret}, f"{BASE}/k:{tag}/{self.REMAINDER}"), True)
        self.assertIs(cc.verify_url_signature({"k": "s3cr3t"}, f"{BASE}/k:{tag}/{self.REMAINDER}"), False)

    def test_a_surrogate_pair_is_the_character_it_encodes(self):
        pair = chr(0xD83D) + chr(0xDE00)  # U+1F600 as two UTF-16 code units
        tag = raw_tag("s3cr3t", "300x200/https://example.com/\U0001f600.jpg")
        url = f"{BASE}/{tag}/300x200/https://example.com/{pair}.jpg"
        self.assertIs(cc.verify_url_signature("s3cr3t", url), True)
        self.assertIs(cc.verify_url_signature("s3cr3t" + pair, url), False)
        keyed = raw_tag("s3cr3t\U0001f600", self.REMAINDER)
        self.assertIs(cc.verify_url_signature("s3cr3t" + pair, f"{BASE}/{keyed}/{self.REMAINDER}"), True)

    def test_no_text_makes_the_verifier_raise(self):
        rng = random.Random(716)
        alphabet = ["a", "/", ":", "%", "x", "1", "é", "\U0001f600", chr(0xD800), chr(0xDBFF), chr(0xDC00), chr(0xDFFF), "\x00"]
        for _ in range(500):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(20)))
            remainder = f"300x200/{text}"
            secret = "".join(rng.choice(alphabet) for _ in range(rng.randrange(8)))
            tag = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
            for url in (f"{BASE}/{tag}/{remainder}", f"{BASE}/k:{tag}/{remainder}", f"{BASE}/{tag}/{text}"):
                for key in (secret, {"k": secret}, {"k": "s3cr3t", "j": secret}, {}):
                    with self.subTest(url=ascii(url), key=ascii(key)):
                        self.assertIn(cc.verify_url_signature(key, url), (None, True, False))


class TestComparison(unittest.TestCase):
    def test_tags_compare_in_constant_time(self):
        # hmac.compare_digest, as `verify` compares: a plain == leaks
        # how much of the tag was right.
        v = URL_VECTORS[0]
        url = f"{BASE}/{v['signature']}/{v['remainder']}"
        with mock.patch("hmac.compare_digest", wraps=hmac.compare_digest) as compare:
            self.assertIs(cc.verify_url_signature(v["secret"], url), True)
            self.assertIs(cc.verify_url_signature({"a": "x", "b": "y"}, url), False)
        self.assertEqual(compare.call_count, 3)
        for call in compare.call_args_list:
            self.assertEqual(call.args[1], v["signature"])

    def test_a_secret_that_is_not_text_or_bytes_raises(self):
        v = URL_VECTORS[0]
        url = f"{BASE}/{v['signature']}/{v['remainder']}"
        for secret in (None, 7, ["s3cr3t"]):
            with self.subTest(secret=secret), self.assertRaises(TypeError):
                cc.verify_url_signature(secret, url)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
