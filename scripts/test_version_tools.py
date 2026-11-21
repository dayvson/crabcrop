#!/usr/bin/env python3
"""Tests for scripts/bump_version.py and scripts/check_versions.py.

Each case copies the part of the working tree a bump can touch into a
temporary directory: every tracked file that mentions the current version
(a bump cannot leave the old version in a file that never had it), the
sites in scripts/version_sites.py, and the version tools themselves. It
then bumps the copy with its own bump_version.py and reads it back with
its own check_versions.py, as a release does (RELEASING.md, step 2). The
copy also carries CHANGELOG.md and an empty changelog.d/, since the bump
assembles the release's CHANGELOG section (RELEASING.md, step 1); the
release notes, fragments included, stay out of the leftover scan.
Run directly: python3 scripts/test_version_tools.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import version_sites  # noqa: E402

TOOLS = ("bump_version.py", "check_versions.py", "version_sites.py", "assemble_changelog.py")

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
    "scripts/test_publish_crates.py": "made-up crates.io index entries",
    "scripts/test_publish_mcp_registry.py": "made-up MCP registry listings",
    # Generated from Cargo.lock, the workspace's own crates left out, so
    # every version in it is a third-party crate's (issue #682).
    "THIRD-PARTY-NOTICES.md": "third-party crates' own versions",
    # The recipe imports the Bunny SDK at the SDK's own version, which
    # is no pin of this release either.
    "deploy/bunnycdn/cache-tag-middleware.ts": "the Bunny SDK's own version",
    # Dated benchmark records: the version in them is the release the
    # run measured, a historical fact a bump must not rewrite. The
    # harness pins that do move with a release are sites instead
    # (scripts/version_sites.py).
    "benchmark/results/2026-11-15.md": "a dated benchmark record",
    "benchmark/results/2026-11-15.json": "a dated benchmark record",
    "benchmark/results/2026-11-15-cache.json": "a dated benchmark record",
    "benchmark/results/2026-11-15-imgproxy4.md": "a dated benchmark record",
    "benchmark/results/2026-11-15-imgproxy4.json": "a dated benchmark record",
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


# A Rust #[deprecated] attribute names the release its deprecation began
# in, `since = "X.Y.Z"` (crabcrop-url's build_sealed_url, issue #712).
# That is history a bump keeps, as the rendering version's entries are:
# every later release still says when the deprecation began. Only the
# value of `since` inside such an attribute is skipped; the attribute's
# note, and any other mention, counts.
DEPRECATED_OPEN = re.compile(r"#\[\s*deprecated\s*\(")
SINCE_VALUE = re.compile(r'\bsince\s*=\s*"\Z')


def deprecation_since(rel, text, pos):
    """Whether offset `pos` of a Rust file is the `since` value of a
    `#[deprecated(...)]` attribute that is still open there."""
    if not rel.endswith(".rs"):
        return False
    opened = None
    for opened in DEPRECATED_OPEN.finditer(text, 0, pos):
        pass
    if opened is None or ")]" in text[opened.end() : pos]:
        return False
    return SINCE_VALUE.search(text, opened.end(), pos) is not None


# The Helm README's own-build snippet builds and pushes an image, then
# installs it with `--set image.tag=`. A release that moves the first
# two and not the third leaves the chart pulling a tag the snippet
# never pushed (issue #565).
HELM_README = "deploy/helm/crabcrop/README.md"

# The benchmark harness, which builds the release its pins name.
BENCHMARK_SETUP = "benchmark/setup.sh"


def run_setup(tree, *args):
    """Run `tree`'s benchmark/setup.sh. Everything it would install
    with is a stub that fails, so a run that gets past the crabcrop pin
    installs nothing, and git never finds a repository above `tree`."""
    stubs = Path(tree) / ".stubs"
    stubs.mkdir(exist_ok=True)
    for tool in ("sudo", "apt-get", "curl", "go", "cargo", "pip"):
        stub = stubs / tool
        stub.write_text(f"#!/bin/sh\necho called >> '{stubs}/called'\nexit 97\n", encoding="utf-8")
        stub.chmod(0o755)
    git = shutil.which("git")
    env = {
        "PATH": os.pathsep.join([str(stubs), *([os.path.dirname(git)] if git else []), "/usr/bin", "/bin"]),
        "HOME": str(tree),
        "GIT_CEILING_DIRECTORIES": str(Path(tree).parent),
        "LC_ALL": "C",
    }
    return subprocess.run(
        ["bash", str(Path(tree) / BENCHMARK_SETUP), *args], capture_output=True, text=True, env=env, timeout=60
    )


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
        files |= {site.path for site in version_sites.RELEASE_COMMITS}
        files |= {f"scripts/{tool}" for tool in TOOLS}
        # The dav1d pin and the docs that name it: the check reads them.
        files |= {version_sites.DAV1D_PIN} | {site.path for site in version_sites.DAV1D_MENTIONS}
        # Release notes name every version, and moving them is the
        # changelog's job (RELEASING.md, step 1): the copy carries the
        # CHANGELOG for the bump to assemble, outside the leftover scan,
        # and no fragments but the ones a case writes.
        files = {rel for rel in files if rel != "CHANGELOG.md" and not rel.startswith("changelog.d/")}
        for rel in files | {"CHANGELOG.md"}:
            target = self.tree / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / rel).read_bytes())
        (self.tree / "changelog.d").mkdir()
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
        old version on. Third-party lockfile entries, EXAMPLES, the
        rendering version's history entries and a deprecation's `since`
        do not count, nor does a prerelease or build of the old version
        (0.10.0-rc.1 in an example), which is another version."""
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
                    or deprecation_since(rel, text, match.start())
                ):
                    continue
                line = text.count("\n", 0, match.start())
                hits.append(f"{rel}:{line + 1}: {text.splitlines()[line].strip()}")
        return hits

    def snapshot(self):
        tracked = [*self.files, "CHANGELOG.md"]
        tracked += [f"changelog.d/{p.name}" for p in sorted((self.tree / "changelog.d").iterdir())]
        return {rel: (self.tree / rel).read_bytes() for rel in tracked}

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

    def test_a_bump_assembles_the_changelog(self):
        new = bumped(self.version, "patch")
        (self.tree / "changelog.d/README.md").write_text("# Changelog fragments\n", encoding="utf-8")
        (self.tree / "changelog.d/1.fixed.md").write_text("A bump fragment (issue #1).\n", encoding="utf-8")
        before = (self.tree / "CHANGELOG.md").read_text(encoding="utf-8")
        code, out, err = self.run_tool("bump_version.py", new, "--date", "2026-02-03")
        self.assertEqual(code, 0, out + err)
        self.assertIn(f"all artifacts at {new}", out)
        self.assertIn("removed changelog.d/1.fixed.md", out)
        text = (self.tree / "CHANGELOG.md").read_text(encoding="utf-8")
        head = f"## [Unreleased]\n\n## [{new}] - 2026-02-03\n\n"
        self.assertIn(head, text)
        released = text[text.index(head) : text.index(f"## [{self.version}]")]
        self.assertIn("- A bump fragment (issue #1).\n", released)
        # The hand-written entries move with it, as written.
        unreleased = before[before.index("## [Unreleased]") : before.index(f"## [{self.version}]")]
        for line in unreleased.splitlines():
            if line.startswith("- "):
                self.assertIn(line, released)
        self.assertEqual(sorted(p.name for p in (self.tree / "changelog.d").iterdir()), ["README.md"])

    def test_a_fragment_that_names_the_version_is_no_leftover(self):
        # `git grep` lists a tracked changelog.d/ file that names the
        # current version (a fragment saying "a regression from X.Y.Z",
        # or the README) with the carriers. The copy leaves it out, as it
        # does CHANGELOG.md, since the bump assembles and deletes a
        # fragment, and the release notes are no leftover.
        found = [*carriers(self.version), "changelog.d/README.md"]
        self.tmp.cleanup()
        with mock.patch.object(sys.modules[__name__], "carriers", lambda version: found):
            self.setUp()
        self.assertFalse([rel for rel in self.files if rel.startswith("changelog.d/")], self.files)
        fragment = f"Fixed a regression from {self.version} (issue #720).\n"
        (self.tree / "changelog.d/720.fixed.md").write_text(fragment, encoding="utf-8")
        new = bumped(self.version, "patch")
        self.bump(new)
        self.assert_nothing_left()
        self.assertIn(f"- {fragment}", (self.tree / "CHANGELOG.md").read_text(encoding="utf-8"))
        self.assertEqual(list((self.tree / "changelog.d").iterdir()), [])

    def test_a_dry_run_bump_writes_no_changelog(self):
        (self.tree / "changelog.d/1.fixed.md").write_text("A bump fragment (issue #1).\n", encoding="utf-8")
        snapshot = self.snapshot()
        code, out, err = self.run_tool("bump_version.py", bumped(self.version, "patch"), "--dry-run")
        self.assertEqual(code, 0, out + err)
        self.assertIn("would update CHANGELOG.md", out)
        self.assertIn("would remove changelog.d/1.fixed.md", out)
        self.assertEqual(self.snapshot(), snapshot, "a dry run wrote files")

    def test_a_malformed_fragment_stops_the_bump(self):
        (self.tree / "changelog.d/1.feature.md").write_text("A thing (issue #1).\n", encoding="utf-8")
        snapshot = self.snapshot()
        code, out, err = self.run_tool("bump_version.py", bumped(self.version, "patch"))
        self.assert_reported(code, out, err, ["changelog.d/1.feature.md: unknown section 'feature': use one of added, changed, deprecated, removed, fixed, security, or summary"])
        self.assertIn("nothing was written", err)
        self.assertEqual(self.snapshot(), snapshot, "a refused bump wrote files")

    def test_a_bump_writes_the_summary_first(self):
        # A summary fragment's lines open the release's section, above its
        # first heading (RELEASING.md step 1, issue #740).
        new = bumped(self.version, "patch")
        (self.tree / "changelog.d/1.fixed.md").write_text("A bump fragment (issue #1).\n", encoding="utf-8")
        (self.tree / "changelog.d/1.summary.md").write_text("- Upgrade for the fix (issue #1).\n", encoding="utf-8")
        code, out, err = self.run_tool("bump_version.py", new, "--date", "2026-02-03")
        self.assertEqual(code, 0, out + err)
        self.assertIn("opens with a summary of 1 line(s)", out)
        self.assertIn("removed changelog.d/1.summary.md", out)
        text = (self.tree / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn(f"## [{new}] - 2026-02-03\n\n- Upgrade for the fix (issue #1).\n\n### ", text)
        self.assertEqual(list((self.tree / "changelog.d").iterdir()), [])

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

    def test_the_scan_skips_only_a_deprecations_since(self):
        # The release a deprecation began in stays when the version moves
        # on (issue #712's build_sealed_url). Its note, an attribute that
        # is closed, another key spelled since, and a file that is not
        # Rust all count.
        old = self.version
        rel, doc = "crates/example/src/lib.rs", "docs/example.md"
        lines = (
            f'#[deprecated(since = "{old}", note = "use `b`")]',
            "pub fn a() {}",
            "#[deprecated(",
            f'    since = "{old}",',
            f'    note = "gone in the release after {old}"',
            ")]",
            "pub fn c() {}",
            f'#[cfg_attr(docsrs, doc = "since = \\"{old}\\"")]',
            f'pub const PINNED: &str = "{old}";',
        )
        for name, body in ((rel, lines), (doc, (f'#[deprecated(since = "{old}")]',))):
            (self.tree / name).parent.mkdir(parents=True, exist_ok=True)
            (self.tree / name).write_text("\n".join(body) + "\n", encoding="utf-8")
            self.files.append(name)
        found = [hit for hit in self.leftovers(old) if hit.startswith((f"{rel}:", f"{doc}:"))]
        self.assertEqual(
            found,
            [
                f'{rel}:5: note = "gone in the release after {old}"',
                f'{rel}:8: #[cfg_attr(docsrs, doc = "since = \\"{old}\\"")]',
                f'{rel}:9: pub const PINNED: &str = "{old}";',
                f'{doc}:1: #[deprecated(since = "{old}")]',
            ],
        )

    def test_a_bump_leaves_the_dav1d_pin(self):
        dav1d, _ = version_sites.dav1d_pin(self.tree)
        before = (self.tree / version_sites.DAV1D_PIN).read_bytes()
        self.bump(bumped(self.version, "minor"))
        self.assertEqual((self.tree / version_sites.DAV1D_PIN).read_bytes(), before)
        self.assertEqual(version_sites.dav1d_pin(self.tree)[0], dav1d)

    def benchmark_commit(self):
        text = (self.tree / BENCHMARK_SETUP).read_text(encoding="utf-8")
        (value,) = re.findall(r"(?m)^CRABCROP_COMMIT=(.*)$", text)
        return value

    def set_benchmark_commit(self, commit):
        path = self.tree / BENCHMARK_SETUP
        text, count = re.subn(r"(?m)^CRABCROP_COMMIT=.*$", f"CRABCROP_COMMIT={commit}", path.read_text(encoding="utf-8"))
        self.assertEqual(count, 1)
        path.write_text(text, encoding="utf-8")

    def test_the_check_fails_on_a_commit_pin_of_another_release(self):
        # Where git can read the pinned commit, it must be one of the
        # release the tree carries. A release commit made with the bump
        # from before issue #711 moved the version and kept the previous
        # release's commit, which setup.sh refuses; so does the check.
        def git(*args):
            done = subprocess.run(
                ["git", "-C", str(self.tree), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            return done.stdout.strip()

        git("init", "-q")
        git("add", "-A")
        git("commit", "-q", "-m", f"Release v{self.version}")
        released = git("rev-parse", "HEAD")
        self.set_benchmark_commit(released)
        self.check_passes(self.version)
        new = bumped(self.version, "patch")
        self.bump(new)
        self.set_benchmark_commit(released)
        what = "CRABCROP_COMMIT (the commit CRABCROP_TAG names)"
        self.assert_reported(
            *self.run_tool("check_versions.py"),
            [
                f"{BENCHMARK_SETUP}: {what} is {released}, a commit of {self.version}, not {new}; empty it, as a "
                f"bump does, or pin the commit v{new} names"
            ],
        )
        # A commit this clone does not have is left to setup.sh, which
        # refuses it; a shallow checkout lacks the commit it pins.
        self.set_benchmark_commit("0" * 40)
        self.check_passes(new)

    def test_a_bump_empties_the_benchmark_commit_pin(self):
        # The tree pins the commit of the release it measures. A bump
        # cannot know the commit the new tag will name, so it empties
        # the pin, and setup.sh, which used to build the previous
        # release's commit under the new tag's name, refuses to run
        # until that tag exists (issue #711). Start from a pinned tree,
        # as the refresh after a tag leaves it: the tree itself is empty
        # between a bump and its tag.
        self.set_benchmark_commit("0123456789abcdef0123456789abcdef01234567")
        self.assertRegex(self.benchmark_commit(), r"^[0-9a-f]{40}$")
        new = bumped(self.version, "minor")
        code, out, err = self.run_tool("bump_version.py", new)
        self.assertEqual(code, 0, out + err)
        self.assertIn(f"emptied CRABCROP_COMMIT (the commit CRABCROP_TAG names) in {BENCHMARK_SETUP}", out)
        self.assertEqual(self.benchmark_commit(), "")
        self.check_passes(new)
        done = run_setup(self.tree, "--check")
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn(f"this clone has no v{new} tag, and CRABCROP_COMMIT is empty", done.stderr)

    def test_a_bump_to_the_same_version_keeps_the_benchmark_commit_pin(self):
        # The refresh after the tag pins its commit; a second bump to the
        # version the tree already carries moves nothing, so it keeps
        # that pin. A later bump that does move the version empties it,
        # and one that finds it empty says nothing about it.
        new = bumped(self.version, "minor")
        self.bump(new)
        self.assertEqual(self.benchmark_commit(), "")
        path = self.tree / BENCHMARK_SETUP
        pinned = "0123456789abcdef0123456789abcdef01234567"
        path.write_text(
            re.sub(r"(?m)^CRABCROP_COMMIT=$", f"CRABCROP_COMMIT={pinned}", path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        code, out, err = self.run_tool("bump_version.py", new)
        self.assertEqual(code, 0, out + err)
        self.assertNotIn("emptied", out)
        self.assertEqual(self.benchmark_commit(), pinned)
        code, out, err = self.run_tool("bump_version.py", bumped(new, "patch"))
        self.assertEqual(code, 0, out + err)
        self.assertIn("emptied CRABCROP_COMMIT", out)
        self.assertEqual(self.benchmark_commit(), "")
        code, out, err = self.run_tool("bump_version.py", bumped(bumped(new, "patch"), "patch"))
        self.assertEqual(code, 0, out + err)
        self.assertNotIn("emptied", out)
        self.assertEqual(self.benchmark_commit(), "")

    def test_the_check_fails_on_a_malformed_commit_pin(self):
        path = self.tree / BENCHMARK_SETUP
        text = path.read_text(encoding="utf-8")
        what = "CRABCROP_COMMIT (the commit CRABCROP_TAG names)"
        for broken, problem in (
            (re.sub(r"(?m)^CRABCROP_COMMIT=.*$", "CRABCROP_COMMIT=abc1234", text), f"{what} is 'abc1234', expected a full commit SHA or nothing"),
            (
                re.sub(r"(?m)^CRABCROP_COMMIT=.*$", f"CRABCROP_COMMIT={'A' * 40}", text),
                f"{what} is '{'A' * 40}', expected a full commit SHA or nothing",
            ),
            (re.sub(r"(?m)^CRABCROP_COMMIT=.*\n", "", text), f"expected 1 {what}, found 0"),
        ):
            with self.subTest(problem=problem):
                path.write_text(broken, encoding="utf-8")
                self.assert_reported(*self.run_tool("check_versions.py"), [f"{BENCHMARK_SETUP}: {problem}"])
        # A pin the bump cannot find stops it before anything is written.
        snapshot = self.snapshot()
        code, out, err = self.run_tool("bump_version.py", bumped(self.version, "patch"))
        self.assert_reported(code, out, err, [f"{BENCHMARK_SETUP}: expected 1 {what}, found 0"])
        self.assertEqual(self.snapshot(), snapshot, "a refused bump wrote files")

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


class BenchmarkPinTest(unittest.TestCase):
    """benchmark/setup.sh builds the release its pins name, or nothing
    (issue #711). Each case runs the tree's setup.sh with --check in a
    repository of two releases: the one the tree carries, tagged, and
    the next minor release, untagged until a case tags it, as between a
    bump and its tag."""

    @classmethod
    def setUpClass(cls):
        cls.old_version = version_sites.check(ROOT)[0]
        cls.new_version = bumped(cls.old_version, "minor")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="crabcrop-benchmark-pin-")
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.old = self.release(self.old_version)
        self.git("tag", f"v{self.old_version}")
        self.new = self.release(self.new_version)
        setup = self.repo / BENCHMARK_SETUP
        setup.parent.mkdir()
        setup.write_bytes((ROOT / BENCHMARK_SETUP).read_bytes())

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        done = subprocess.run(
            ["git", "-C", str(self.repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            capture_output=True,
            text=True,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout.strip()

    def release(self, version):
        """Commit a Cargo.toml of `version`; return the commit."""
        (self.repo / "Cargo.toml").write_text(
            f'[package]\nname = "crabcrop"\nversion = "{version}"\n\n[dependencies]\nhex = {{ version = "0.4" }}\n',
            encoding="utf-8",
        )
        self.git("add", "Cargo.toml")
        self.git("commit", "-q", "-m", f"Release v{version}")
        return self.git("rev-parse", "HEAD")

    def check(self, version, commit):
        """setup.sh --check with the harness pinned at `version`, its
        tag, and `commit`."""
        path = self.repo / BENCHMARK_SETUP
        text = path.read_text(encoding="utf-8")
        for name, value in (("VERSION", version), ("TAG", f"v{version}"), ("COMMIT", commit)):
            text, count = re.subn(rf"(?m)^CRABCROP_{name}=.*$", f"CRABCROP_{name}={value}", text)
            self.assertEqual(count, 1, f"{BENCHMARK_SETUP} has no CRABCROP_{name}= line")
        path.write_text(text, encoding="utf-8")
        done = run_setup(self.repo, "--check")
        self.assertFalse((self.repo / ".stubs" / "called").exists(), "--check ran an install step")
        self.assertFalse((self.repo / "benchmark" / ".work").exists(), "--check made benchmark/.work")
        return done

    def assert_builds(self, done, commit, source):
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f": commit {commit}, from {source}", done.stdout)

    def assert_refused(self, done, reason):
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn(reason, done.stderr)
        self.assertNotIn(": commit ", done.stdout)

    def test_the_tag_names_the_commit(self):
        old = self.old_version
        self.assert_builds(self.check(old, self.old), self.old, f"the v{old} tag")
        self.assert_builds(self.check(old, ""), self.old, f"the v{old} tag")

    def test_a_pin_the_tag_disagrees_with_is_refused(self):
        old = self.old_version
        self.assert_refused(self.check(old, self.new), f"v{old} is commit {self.old}, but CRABCROP_COMMIT pins {self.new}")

    def test_the_previous_release_is_never_built_under_the_new_tag(self):
        # The pins a bump left before issue #711: the new version and
        # tag, and the previous release's commit. setup.sh fell back to
        # that commit while the new tag was missing.
        old, new = self.old_version, self.new_version
        self.assert_refused(self.check(new, self.old), f"commit {self.old} is crabcrop {old}, not {new}")

    def test_an_empty_pin_waits_for_the_tag(self):
        new = self.new_version
        self.assert_refused(self.check(new, ""), f"this clone has no v{new} tag, and CRABCROP_COMMIT is empty")
        self.git("tag", "-a", "-m", f"v{new}", f"v{new}", self.new)
        self.assert_builds(self.check(new, ""), self.new, f"the v{new} tag")

    def test_a_pinned_commit_of_the_version_builds_without_the_tag(self):
        new = self.new_version
        self.assert_builds(self.check(new, self.new), self.new, f"CRABCROP_COMMIT (this clone has no v{new} tag)")
        missing = "0" * 40
        self.assert_refused(self.check(new, missing), f"neither the v{new} tag nor the pinned commit {missing}")

    def test_a_commit_without_a_readable_version_is_refused(self):
        # A pinned commit whose tree has no Cargo.toml is refused with
        # the reason, not cut short by `set -e`.
        new = self.new_version
        self.git("rm", "-q", "Cargo.toml")
        self.git("commit", "-q", "-m", "No manifest")
        bare = self.git("rev-parse", "HEAD")
        self.assert_refused(self.check(new, bare), f"commit {bare} is crabcrop of no readable version, not {new}")

    def test_a_malformed_pin_is_refused(self):
        old = self.old_version
        self.assert_refused(self.check(old, self.old[:12]), f"CRABCROP_COMMIT '{self.old[:12]}' is not a full commit SHA")
        self.assert_refused(self.check(old, self.old.upper()), f"CRABCROP_COMMIT '{self.old.upper()}' is not a full commit SHA")

    def test_a_worktree_left_at_another_commit_is_refused(self):
        tree = self.repo / "benchmark" / ".work" / f"crabcrop-v{self.old_version}"
        self.git("worktree", "add", "-q", "--detach", str(tree), self.new)
        path = self.repo / BENCHMARK_SETUP
        path.write_text(re.sub(r"(?m)^CRABCROP_COMMIT=.*$", f"CRABCROP_COMMIT={self.old}", path.read_text()))
        done = run_setup(self.repo, "--check")
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn(f"{tree} holds {self.new}, not {self.old}", done.stderr)
        self.git("-C", str(tree), "checkout", "-q", "--detach", self.old)
        done = run_setup(self.repo, "--check")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_an_unknown_argument_is_refused(self):
        done = run_setup(self.repo, "--fast")
        self.assertEqual(done.returncode, 2)
        self.assertIn("usage: bash benchmark/setup.sh [--check]", done.stderr)


if __name__ == "__main__":
    unittest.main()
