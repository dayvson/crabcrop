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
    "scripts/test_check_locks.py": "made-up lockfile entries, third-party crates included",
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


# RENDERING_VERSION's doc comment in src/main.rs keeps one entry per
# value, "N: the first release after X.Y.Z" (RELEASING.md, step 4).
# That names the release before the change, so it is history a bump
# keeps, and each raise adds one naming the version of its day. Only a
# version that ends this phrase inside that comment is skipped.
HISTORY_FILE = "src/main.rs"
RENDERING_CONST = re.compile(r"(?m)^[ \t]*(?:pub(?:\([^)]*\))?\s+)?const RENDERING_VERSION\b")
_GAP = r"\s+(?:///\s*)?"  # a space, or a line break inside the comment
HISTORY_ENTRY = re.compile(r"\bthe" + "".join(_GAP + word for word in ("first", "release", "after")) + _GAP + r"\Z")


def rendering_history(rel, text, pos):
    """Whether offset `pos` of a file is the release an entry of
    RENDERING_VERSION's doc comment counts from."""
    if rel != HISTORY_FILE:
        return False
    const = RENDERING_CONST.search(text)
    if const is None or pos >= const.start():
        return False
    start = const.start()
    for line in reversed(text[:start].splitlines(keepends=True)):
        if not line.lstrip().startswith("///"):
            break
        start -= len(line)
    return pos >= start and HISTORY_ENTRY.search(text[start:pos]) is not None


def dav1d_pin(rel, text, pos):
    """Whether offset `pos` of a file is the dav1d pin, or a doc's mention
    of it (version_sites.DAV1D_MENTIONS): dav1d's own version, which a
    crabcrop release that happens to share the number leaves alone."""
    if rel == version_sites.DAV1D_PIN:
        start = text.rfind("\n", 0, pos) + 1
        return text.startswith("DAV1D_VERSION=", start)
    return any(
        site.path == rel and any(start <= pos < end for start, end in site.find(text))
        for site in version_sites.DAV1D_MENTIONS
    )


# The Helm README's own-build snippet builds and pushes an image, then
# installs it with `--set image.tag=`. A release that moves the first
# two and not the third leaves the chart pulling a tag the snippet
# never pushed (issue #565).
HELM_README = "deploy/helm/crabcrop/README.md"


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
        # The dav1d pin and the docs that name it: the check reads them.
        files |= {version_sites.DAV1D_PIN} | {site.path for site in version_sites.DAV1D_MENTIONS}
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
        old version on. Third-party lockfile entries, EXAMPLES and the
        rendering version's history entries do not count, nor does a
        prerelease or build of the old version (0.10.0-rc.1 in an
        example), which is another version."""
        word = re.compile(rf"(?<![0-9A-Za-z_]){re.escape(old)}(?![0-9A-Za-z_])(?![-+][0-9A-Za-z])")
        hits = []
        for rel in self.files:
            if rel in EXAMPLES:
                continue
            text = (self.tree / rel).read_text(encoding="utf-8", errors="replace")
            for match in word.finditer(text):
                if (
                    third_party(rel, text, match.start())
                    or rendering_history(rel, text, match.start())
                    or dav1d_pin(rel, text, match.start())
                ):
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
            "version, generated text) is reworded, or its file joins EXAMPLES in this test; "
            "a RENDERING_VERSION entry reads 'N: the first release after X.Y.Z' (RELEASING.md, step 4)",
        )

    def requirement(self):
        text = (self.tree / "Cargo.toml").read_text(encoding="utf-8")
        return re.search(r'^crabcrop-url = \{[^}\n]*\bversion = "([^"]+)"', text, re.M).group(1)

    def own_build_pins(self):
        """Every X.Y.Z in the Helm README's own-build snippet, in order."""
        text = (self.tree / HELM_README).read_text(encoding="utf-8")
        blocks = [block for block in re.findall(r"(?ms)^```bash\n(.*?)^```$", text) if "docker build" in block]
        self.assertEqual(len(blocks), 1, f"{HELM_README}: expected one bash block with `docker build`")
        return re.findall(r"\d+\.\d+\.\d+", blocks[0])

    def test_a_bump_moves_the_helm_own_build_snippet(self):
        self.assertEqual(self.own_build_pins(), [self.version] * 3)
        new = bumped(self.version, "patch")
        self.bump(new)
        self.assertEqual(self.own_build_pins(), [new] * 3)

    def test_the_check_fails_on_a_stale_helm_image_tag(self):
        # A pin that was stale before a bump is not the old version
        # after it, so the leftover scan cannot see it: the check must.
        new = bumped(self.version, "patch")
        self.bump(new)
        path = self.tree / HELM_README
        text = path.read_text(encoding="utf-8")
        path.write_text(re.sub(r"(?<=image\.tag=)\d+\.\d+\.\d+", self.version, text), encoding="utf-8")
        self.assert_reported(
            *self.run_tool("check_versions.py"),
            [f"{HELM_README}: image pins is {self.version}, expected {new}"],
        )

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

    def test_the_scan_skips_only_the_rendering_history(self):
        old, rel = self.version, HISTORY_FILE
        path = self.tree / rel
        if rel not in self.files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / rel).read_bytes())
            self.files.append(rel)
        text = path.read_text(encoding="utf-8")
        const = RENDERING_CONST.search(text)
        self.assertIsNotNone(const, f"RENDERING_VERSION left {rel}: move HISTORY_FILE with it")
        # The comment starts after the last blank line above the const.
        comment = text.rindex("\n\n", 0, const.start()) + 2
        above = f"// the first release after {old}, above the comment\n"
        entries = (
            f"/// 8: the first release after {old}, which changes\n"
            f"/// 9: the first release after\n/// {old}, wrapped.\n"
            f"/// Built by crabcrop {old}.\n"
        )
        elsewhere = f'\n// the first release after {old}\nfn later() {{}}\n\npub const PINNED: &str = "{old}";\n'
        path.write_text(
            text[:comment] + above + text[comment : const.start()] + entries + text[const.start() :] + elsewhere,
            encoding="utf-8",
        )
        found = [hit.split(": ", 1)[1] for hit in self.leftovers(old) if hit.startswith(f"{rel}:")]
        # Entries in the comment are skipped, the file's own history
        # included; any other mention, in the comment or not, before
        # the const or after it, counts.
        self.assertEqual(
            found,
            [
                f"// the first release after {old}, above the comment",
                f"/// Built by crabcrop {old}.",
                f"// the first release after {old}",
                f'pub const PINNED: &str = "{old}";',
            ],
        )

    def test_a_bump_leaves_the_dav1d_pin(self):
        dav1d, _ = version_sites.dav1d_pin(self.tree)
        before = (self.tree / version_sites.DAV1D_PIN).read_bytes()
        self.bump(bumped(self.version, "minor"))
        self.assertEqual((self.tree / version_sites.DAV1D_PIN).read_bytes(), before)
        self.assertEqual(version_sites.dav1d_pin(self.tree)[0], dav1d)

    def test_the_scan_skips_only_the_dav1d_pin_and_its_mentions(self):
        # A crabcrop release that shares dav1d's number leaves the pin
        # and the docs that name it, and nothing else.
        dav1d, _ = version_sites.dav1d_pin(self.tree)
        pin = self.tree / version_sites.DAV1D_PIN
        pin.write_text(pin.read_text(encoding="utf-8") + f"# built as {dav1d}\n", encoding="utf-8")
        files = (version_sites.DAV1D_PIN, *(site.path for site in version_sites.DAV1D_MENTIONS))
        found = [hit for hit in self.leftovers(dav1d) if hit.startswith(files)]
        self.assertEqual(found, [f"{version_sites.DAV1D_PIN}:{len(pin.read_text().splitlines())}: # built as {dav1d}"])

    def test_the_check_fails_when_a_doc_names_another_dav1d(self):
        dav1d, _ = version_sites.dav1d_pin(self.tree)
        doc = self.tree / "docs/deployment.md"
        doc.write_text(
            doc.read_text(encoding="utf-8").replace(f"link dav1d {dav1d}", "link dav1d 1.3.0"),
            encoding="utf-8",
        )
        self.assert_reported(
            *self.run_tool("check_versions.py"),
            [
                "docs/deployment.md: dav1d version the release binaries link is 1.3.0, "
                f"but {version_sites.DAV1D_PIN} pins {dav1d}"
            ],
        )

    def test_the_check_fails_on_a_malformed_dav1d_pin(self):
        pin = self.tree / version_sites.DAV1D_PIN
        text = pin.read_text(encoding="utf-8")
        for broken, problem in (
            (re.sub(r"(?m)^DAV1D_SHA256=.*$", "DAV1D_SHA256=abc", text), "DAV1D_SHA256 is not 64 lowercase hex digits"),
            (re.sub(r"(?m)^DAV1D_VERSION=.*$", "DAV1D_VERSION=latest", text), "DAV1D_VERSION 'latest' is not X.Y.Z"),
            (re.sub(r"(?m)^DAV1D_VERSION=.*\n", "", text), "expected one DAV1D_VERSION= line and one DAV1D_SHA256= line"),
        ):
            with self.subTest(problem=problem):
                pin.write_text(broken, encoding="utf-8")
                self.assert_reported(*self.run_tool("check_versions.py"), [f"{version_sites.DAV1D_PIN}: {problem}"])

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
