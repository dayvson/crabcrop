"""The package's names (issue #716): every export of the TypeScript
SDK under its Python spelling, the earlier Python names kept
working, and an ``__all__`` that lists the public API."""

import inspect
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import crabcrop_url  # noqa: E402
from crabcrop_url import build_srcset, build_url, sign_with_key_id  # noqa: E402

TYPESCRIPT_INDEX = Path(__file__).resolve().parents[2] / "typescript" / "src" / "index.ts"

# The TypeScript SDK's exports and their Python names. ``srcset`` is
# one word in HTML, and the Rust crate spells the function
# ``build_srcset`` too.
PYTHON_NAMES = {
    "RESERVED_SEGMENT": "RESERVED_SEGMENT",
    "isReservedRemainder": "is_reserved_remainder",
    "sign": "sign",
    "purgeSignature": "purge_signature",
    "purgeGroupSignature": "purge_group_signature",
    "uploadSignature": "upload_signature",
    "isValidKeyId": "is_valid_key_id",
    "signWithKeyId": "sign_with_key_id",
    "MAX_SEALED_REMAINDER": "MAX_SEALED_REMAINDER",
    "sealWithNonce": "seal_with_nonce",
    "seal": "seal",
    "unseal": "unseal",
    "buildSealedURL": "build_sealed_url",
    "buildURL": "build_url",
    "buildSrcSet": "build_srcset",
    "MAX_DIMENSION": "MAX_DIMENSION",
    "ops": "ops",
    "filter": "filter",
    "joinOps": "join_ops",
    "Anchor": "Anchor",
    "BlendMode": "BlendMode",
    "CropAlign": "CropAlign",
    "CurveChannel": "CurveChannel",
    "Dimension": "Dimension",
    "FormatOptions": "FormatOptions",
    "ImageFormat": "ImageFormat",
    "MaskShape": "MaskShape",
    "PerFormatQualityFormat": "PerFormatQualityFormat",
    "StudioMode": "StudioMode",
    "StudioOptions": "StudioOptions",
    "StudioShadow": "StudioShadow",
}

# The explainer and what reads URLs through it: the Python port
# follows issue #712, which changes the explainer's number rules.
EXPLAINER = {
    "explainUrl",
    "FILTER_DESCRIPTIONS",
    "verifyUrlSignature",
    "ExplainedSegment",
    "Explanation",
    "SegmentKind",
    "SignatureKind",
}

# Option objects the Python functions take as keyword arguments.
KEYWORD_ARGUMENTS = {"BuildOptions", "SealOptions"}


def typescript_exports():
    text = TYPESCRIPT_INDEX.read_text(encoding="utf-8")
    names = set(re.findall(r"^export (?:const|function|interface|type) (\w+)", text, re.M))
    for group in re.findall(r"^export (?:type )?\{([^}]*)\} from", text, re.M):
        names.update(name.strip() for name in group.split(",") if name.strip())
    return names


class TestNames(unittest.TestCase):
    def test_every_typescript_export_has_its_python_name(self):
        exports = typescript_exports()
        self.assertIn("buildURL", exports)
        for name in sorted(exports):
            with self.subTest(export=name):
                if name in EXPLAINER or name in KEYWORD_ARGUMENTS:
                    continue
                self.assertIn(name, PYTHON_NAMES, f"the TypeScript SDK exports {name}; give it a Python name")
                python = PYTHON_NAMES[name]
                self.assertTrue(hasattr(crabcrop_url, python), python)
                self.assertIn(python, crabcrop_url.__all__)

    def test_seal_is_the_typescript_and_rust_name_and_seal_url_still_works(self):
        self.assertIs(crabcrop_url.seal_url, crabcrop_url.seal)
        self.assertIn("seal_url", crabcrop_url.__all__)

    def test_build_srcset_takes_a_key_id_as_build_srcset_does_in_typescript(self):
        base, source = "https://img.example.com", "https://example.com/a.jpg"
        keyed = build_srcset(base, "test-secret", source, [320, 640], key_id="k2")
        self.assertEqual(
            keyed,
            ", ".join(f"{build_url(base, 'test-secret', f'{w}x0', source, key_id='k2')} {w}w" for w in (320, 640)),
        )
        self.assertIn(f"/{sign_with_key_id('test-secret', 'k2', f'320x0/{source}')}/", keyed)
        with self.assertRaises(ValueError):
            build_srcset(base, None, source, [320], key_id="k2")

    def test_all_lists_every_public_name(self):
        public = set()
        for name, value in vars(crabcrop_url).items():
            if name.startswith("_") or inspect.ismodule(value):
                continue
            if inspect.isfunction(value) or inspect.isclass(value):
                if value.__module__.startswith("crabcrop_url"):
                    public.add(name)
            elif re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                public.add(name)
        public |= {"ops", "filter", "Secret"}
        self.assertLessEqual(public, set(crabcrop_url.__all__))
        for name in crabcrop_url.__all__:
            self.assertTrue(hasattr(crabcrop_url, name), name)
        self.assertEqual(len(crabcrop_url.__all__), len(set(crabcrop_url.__all__)))

    def test_a_star_import_brings_the_api_and_not_the_implementation(self):
        namespace = {}
        exec("from crabcrop_url import *", namespace)
        names = set(namespace) - {"__builtins__"}
        self.assertLessEqual({"ops", "filter", "join_ops", "build_url", "seal"}, names)
        self.assertEqual(names & {"base64", "hashlib", "hmac", "os", "Optional", "annotations"}, set())


if __name__ == "__main__":
    unittest.main()
