#!/usr/bin/env python3
"""Tests for scripts/bump_version.py and scripts/check_versions.py.

Each case copies the part of the working tree a bump can touch into a
temporary directory: every tracked file that mentions the current version
(a bump cannot leave the old version in a file that never had it), the
sites in scripts/version_sites.py, and the version tools themselves. It
then bumps the copy with its own bump_version.py and reads it back with
its own check_versions.py, as a release does (RELEASING.md, step 2).
Run directly: python3 scripts/test_version_tools.py
"""
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import version_sites  # noqa: E402

TOOLS = ("bump_version.py", "check_versions.py", "version_sites.py")

# Files that spell versions on purpose, as test inputs or examples: a
# bump must not move them, and the release version appearing in one is
# no pin. Keep this to such files; a file that ships or documents the
# release version belongs in scripts/version_sites.py.
EXAMPLES = {
    "scripts/test_all.sh": "the npm_dist_tag.sh cases",
    "sdks/mcp/test/pack.test.mjs": "a consumer package's own version",
    ".github/workflows/release.yml": "cargo-dist's comment on tag formats",
    "docs/release-workflow.yml": "the mirror of release.yml",
}


def carriers(version):
    """The tracked files of the working tree that mention `version` as
    a word, as `git grep -lwF` lists them."""
    found = subprocess.run(
        ["git", "-C", str(ROOT), "grep", "-lzwF", "-e", version],
        capture_output=True,
        text=True,
    )
    if found.returncode not in (0, 1):
        raise RuntimeError(f"git grep failed: {found.stderr.strip()}")
    return [path for path in found.stdout.split("\0") if path]


def third_party(rel, text, pos):
    """Whether offset `pos` of a lockfile lies in a third-party package's
    entry, where a version equal to ours is that package's own."""
    name = rel.rsplit("/", 1)[-1]
    if name == "Cargo.lock":
        start = max(text.rfind("[[package]]", 0, pos), 0)
        end = text.find("[[package]]", pos)
        return "\nsource = " in text[start : end if end != -1 else len(text)]
    if name == "package-lock.json":
        keys = re.findall(r'(?m)^    "((?:[^"\\]|\\.)*)": \{$', text[:pos])
        return bool(keys) and keys[-1].startswith("node_modules/")
    return False


def bumped(version, step):
    major, minor, patch = (int(part) for part in version.split("."))
    return f"{major}.{minor}.{patch + 1}" if step == "patch" else f"{major}.{minor + 1}.0"


class VersionToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.version, _, cls.problems = version_sites.check(ROOT)

    def setUp(self):
        self.assertEqual(self.problems, [], "the working tree's own version sites disagree")
        self.tmp = tempfile.TemporaryDirectory(prefix="crabcrop-version-tools-")
        self.tree = Path(self.tmp.name)
        files = set(carriers(self.version)) | {site.path for site in version_sites.SITES}
        files |= {f"scripts/{tool}" for tool in TOOLS}
        # Release notes name every version, and moving them is the
        # changelog's job (RELEASING.md, step 1).
        files.discard("CHANGELOG.md")
        for rel in files:
            target = self.tree / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / rel).read_bytes())
        self.files = sorted(files)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, name, *args):
        done = subprocess.run(
            [sys.executable, str(self.tree / "scripts" / name), *args],
            capture_output=True,
            text=True,
        )
        return done.returncode, done.stdout, done.stderr

    def leftovers(self, old):
        """`git grep -nwF old` over the copy: each line a bump left the
        old version on. Third-party lockfile entries and EXAMPLES do
        not count, nor does a prerelease or build of the old version
        (0.10.0-rc.1 in an example), which is another version."""
        word = re.compile(rf"(?<![0-9A-Za-z_]){re.escape(old)}(?![0-9A-Za-z_])(?![-+][0-9A-Za-z])")
        hits = []
        for rel in self.files:
            if rel in EXAMPLES:
                continue
            text = (self.tree / rel).read_text(encoding="utf-8", errors="replace")
            for match in word.finditer(text):
                if third_party(rel, text, match.start()):
                    continue
                line = text.count("\n", 0, match.start())
                hits.append(f"{rel}:{line + 1}: {text.splitlines()[line].strip()}")
        return hits

    def snapshot(self):
        return {rel: (self.tree / rel).read_bytes() for rel in self.files}

    def bump(self, version):
        code, out, err = self.run_tool("bump_version.py", version)
        self.assertEqual(code, 0, out + err)
        self.assertIn(f"all artifacts at {version}", out)

    def check_passes(self, version):
        code, out, err = self.run_tool("check_versions.py")
        self.assertEqual(code, 0, out + err)
        self.assertIn(f"all artifacts at {version}", out)

    def assert_reported(self, code, out, err, problems):
        """The tool failed cleanly, naming exactly these problems, one
        per indented stderr line."""
        self.assertEqual(code, 1, out + err)
        self.assertNotIn("Traceback", err)
        self.assertEqual([line[2:] for line in err.splitlines() if line.startswith("  ")], problems)

    def assert_nothing_left(self):
        self.assertEqual(
            self.leftovers(self.version),
            [],
            "the bump left the old version behind: add each site to scripts/version_sites.py; "
            "a mention that is not a pin of this release (a test input, another project's "
            "version, generated text) is reworded, or its file joins EXAMPLES in this test",
        )

    def requirement(self):
        text = (self.tree / "Cargo.toml").read_text(encoding="utf-8")
        return re.search(r'^crabcrop-url = \{[^}\n]*\bversion = "([^"]+)"', text, re.M).group(1)

    def test_a_patch_bump_leaves_nothing_behind(self):
        new = bumped(self.version, "patch")
        requirement = self.requirement()
        self.bump(new)
        self.assert_nothing_left()
        self.check_passes(new)
        self.assertEqual(self.requirement(), requirement, "a patch bump keeps the crabcrop-url requirement")

    def test_the_scan_counts_every_plain_mention(self):
        old = self.version
        lines = (
            f"docker pull ghcr.io/dayvson/crabcrop:{old}",
            f'"version": "{old}"',
            f"the {old}-rc.1 candidate",
            f"built as {old}+sha.5",
            f"tag v{old}",
        )
        for rel in ("docs/example.md", *EXAMPLES):
            (self.tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.tree / rel).write_text("\n".join(lines) + "\n", encoding="utf-8")
            if rel not in self.files:
                self.files.append(rel)
        found = [hit for hit in self.leftovers(old) if hit.startswith(("docs/example.md:", *EXAMPLES))]
        self.assertEqual(found, [f"docs/example.md:1: {lines[0]}", f"docs/example.md:2: {lines[1]}"])

    def test_a_minor_bump_moves_the_crabcrop_url_requirement(self):
        new = bumped(self.version, "minor")
        self.bump(new)
        self.assert_nothing_left()
        self.assertEqual(self.requirement(), new.rsplit(".", 1)[0])
        self.check_passes(new)

    def test_a_bump_that_changes_the_version_length(self):
        # Every pin grows, so a replacement that shifts the offset of a
        # pin still to be replaced corrupts its file (server.json holds
        # its two pins in the opposite order to the site's fields).
        major, minor, patch = self.version.split(".")
        new = f"{int(major) + 1}.{minor}.{patch}0"
        self.assertGreater(len(new), len(self.version))
        self.bump(new)
        self.assert_nothing_left()
        self.assertEqual(self.requirement(), new.rsplit(".", 1)[0])
        self.check_passes(new)

    def test_the_check_fails_on_a_stale_requirement(self):
        new = bumped(self.version, "minor")
        self.bump(new)
        cargo = self.tree / "Cargo.toml"
        old_minor, new_minor = self.version.rsplit(".", 1)[0], new.rsplit(".", 1)[0]
        cargo.write_text(
            cargo.read_text(encoding="utf-8").replace(f'version = "{new_minor}"', f'version = "{old_minor}"'),
            encoding="utf-8",
        )
        self.assert_reported(
            *self.run_tool("check_versions.py"),
            [f"Cargo.toml: crabcrop-url requirement (major.minor) is {old_minor}, expected {new_minor}"],
        )

    def test_the_check_fails_on_a_stale_lockfile(self):
        new = bumped(self.version, "patch")
        self.bump(new)
        linked = 'packages["../../sdks/typescript"].version'
        for rel, entries in (
            ("Cargo.lock", ["crabcrop package entry", "crabcrop-url package entry"]),
            ("fuzz/Cargo.lock", ["crabcrop package entry", "crabcrop-url package entry"]),
            ("sdks/typescript/package-lock.json", ['version, packages[""].version']),
            ("sdks/mcp/package-lock.json", ['version, packages[""].version, packages["../typescript"].version']),
            ("frameworks/nuxt/package-lock.json", [f'version, packages[""].version, {linked}']),
            ("tests/browser-bundles/package-lock.json", [linked]),
        ):
            with self.subTest(lockfile=rel):
                bumped_text = (self.tree / rel).read_bytes()
                # The file as the tree has it: every pin at the old version.
                (self.tree / rel).write_bytes((ROOT / rel).read_bytes())
                try:
                    self.assert_reported(
                        *self.run_tool("check_versions.py"),
                        [f"{rel}: {entry} is {self.version}, expected {new}" for entry in entries],
                    )
                finally:
                    (self.tree / rel).write_bytes(bumped_text)
        self.check_passes(new)

    def test_a_wrong_helm_tag_count_is_a_mismatch(self):
        values = "deploy/helm/crabcrop/values.yaml"
        production = "deploy/helm/crabcrop/values-production.yaml"
        both = 'image tags (`tag: "..."` lines: image.tag and ai.image.tag)'
        third = f'\nsidecar:\n  image:\n    tag: "{self.version}"\n'
        cases = (
            (values, lambda text: text + third, f"expected 2 {both}, found 3"),
            (values, lambda text: re.sub(r'(?m)^    tag: "[^"]+"\n', "", text, count=1), f"expected 2 {both}, found 1"),
            (
                production,
                lambda text: re.sub(r'(?m)^  tag: "[^"]+"\n', "", text),
                'expected 1 image tag (`tag: "..."` line: image.tag), found 0',
            ),
        )
        for rel, change, problem in cases:
            with self.subTest(file=rel, problem=problem):
                path = self.tree / rel
                before = path.read_text(encoding="utf-8")
                path.write_text(change(before), encoding="utf-8")
                self.assertNotEqual(path.read_text(encoding="utf-8"), before, "the case must change the file")
                try:
                    self.assert_reported(*self.run_tool("check_versions.py"), [f"{rel}: {problem}"])
                    snapshot = self.snapshot()
                    code, out, err = self.run_tool("bump_version.py", bumped(self.version, "patch"))
                    self.assert_reported(code, out, err, [f"{rel}: {problem}"])
                    self.assertIn("nothing was written", err)
                    self.assertEqual(self.snapshot(), snapshot, "a refused bump wrote files")
                finally:
                    path.write_text(before, encoding="utf-8")


class LeftoverScanTest(unittest.TestCase):
    def test_only_third_party_lockfile_entries_are_skipped(self):
        # Below any release version the tree will carry, so this file
        # never holds a pin of it.
        version = "0.0.1"
        cargo = (
            f'[[package]]\nname = "crabcrop"\nversion = "{version}"\n\n'
            f'[[package]]\nname = "other"\nversion = "{version}"\n'
            'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        )
        npm = json.dumps(
            {"packages": {"../../sdks/typescript": {"version": version}, "node_modules/other": {"version": version}}},
            indent=2,
        )
        for rel, text in (("fuzz/Cargo.lock", cargo), ("frameworks/nuxt/package-lock.json", npm)):
            with self.subTest(lockfile=rel):
                skipped = [third_party(rel, text, m.start()) for m in re.finditer(re.escape(version), text)]
                self.assertEqual(skipped, [False, True])
        self.assertFalse(third_party("README.md", cargo, cargo.rindex(version)))


if __name__ == "__main__":
    unittest.main()
