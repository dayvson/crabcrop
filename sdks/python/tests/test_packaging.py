"""The package metadata (issue #716): an SPDX license expression,
Python-version classifiers that match requires-python, the typing
marker in the package data, and one version."""

import re
import sys
import unittest
from pathlib import Path

PYTHON_SDK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_SDK / "src"))

import crabcrop_url  # noqa: E402

try:
    import tomllib
except ImportError:  # Python 3.10 and older
    tomllib = None

PYPROJECT = (PYTHON_SDK / "pyproject.toml").read_text(encoding="utf-8")


class TestVersion(unittest.TestCase):
    def test_the_package_version_is_the_project_version(self):
        # scripts/bump_version.py moves both, and
        # scripts/check_versions.py reads both.
        project = re.search(r'^\[project\][^\[]*?^version = "([^"]+)"$', PYPROJECT, re.M | re.S)
        self.assertIsNotNone(project)
        self.assertEqual(crabcrop_url.__version__, project.group(1))


@unittest.skipIf(tomllib is None, "reading pyproject.toml needs tomllib (Python 3.11 or newer)")
class TestMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pyproject = tomllib.loads(PYPROJECT)
        cls.project = cls.pyproject["project"]
        cls.classifiers = cls.project["classifiers"]

    def test_the_license_is_an_spdx_expression(self):
        # PEP 639: the expression replaces the legacy table, and
        # setuptools refuses a license classifier beside it.
        self.assertEqual(self.project["license"], "MIT")
        self.assertEqual([c for c in self.classifiers if c.startswith("License ::")], [])
        # Expressions need setuptools 77, and the release builds
        # with the hashed pin in build-requirements.txt.
        self.assertIn("setuptools>=77", self.pyproject["build-system"]["requires"])
        pin = re.search(r"^setuptools==(\d+)\.", (PYTHON_SDK / "build-requirements.txt").read_text(), re.M)
        self.assertIsNotNone(pin)
        self.assertGreaterEqual(int(pin.group(1)), 77)

    def test_the_classifiers_list_each_python_from_requires_python_on(self):
        floor = re.fullmatch(r">=3\.(\d+)", self.project["requires-python"])
        self.assertIsNotNone(floor, self.project["requires-python"])
        minors = sorted(
            int(m.group(1))
            for m in (re.fullmatch(r"Programming Language :: Python :: 3\.(\d+)", c) for c in self.classifiers)
            if m
        )
        self.assertTrue(minors, "no Python version classifiers")
        self.assertEqual(minors[0], int(floor.group(1)))
        self.assertEqual(minors, list(range(minors[0], minors[-1] + 1)), "a gap in the Python versions")
        self.assertIn("Programming Language :: Python :: 3 :: Only", self.classifiers)

    def test_the_package_data_ships_the_typing_marker(self):
        # PEP 561: py.typed tells type checkers to read the inline
        # annotations.
        self.assertTrue((PYTHON_SDK / "src" / "crabcrop_url" / "py.typed").is_file())
        self.assertIn("py.typed", self.pyproject["tool"]["setuptools"]["package-data"]["crabcrop_url"])
        self.assertIn("Typing :: Typed", self.classifiers)


if __name__ == "__main__":
    unittest.main()
