#!/usr/bin/env python3
"""Tests for scripts/check_rendering_version.py.

Each case builds a throwaway git repository with a tagged release, changes
the corpus or the constant after the tag, and runs the check against it as
the gate does. Run directly: python3 scripts/test_check_rendering_version.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "check_rendering_version.py"


def without_repository_variables(environ):
    """The environment minus the variables that pick a repository.

    A git hook exports GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE and the rest
    of git's repository-local variables so the commands it runs find the
    hook's repository. Inherited here (scripts/test_all.sh run from a
    pre-push hook, say), they would point every command at that repository
    instead of the throwaway one: `git add -A` and `git commit` would commit
    the caller's uncommitted work and `git tag` would tag it. Git lists the
    variables itself, so the set follows the installed version.
    """
    listed = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True, check=True
    ).stdout.split()
    return {k: v for k, v in environ.items() if k not in listed}


# Isolated from the host's git configuration (signing, hooks, default
# branch) and from any repository the caller's environment names, so the
# cases behave the same on every machine.
GIT_ENV = {
    **without_repository_variables(os.environ),
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}


def entry(entry_id, sha="aa", path=None):
    return {
        "id": entry_id,
        "kind": "render",
        "path": path or f"/sig/300x200/{entry_id}.jpg",
        "status": 200,
        "content_type": "image/jpeg",
        "bytes": 100,
        "sha256": sha,
        "note": "",
    }


def main_rs(version):
    return f"// the server\nconst RENDERING_VERSION: u32 = {version};\nfn main() {{}}\n"


class RenderingVersionCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        self.git("init", "-q", "-b", "main")

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args, cwd=None):
        subprocess.run(
            ["git", *args], cwd=cwd or self.repo, env=GIT_ENV, check=True, capture_output=True
        )

    def write(self, entries=None, version=None, rel="src/main.rs"):
        if entries is not None:
            corpus = self.repo / "tests/upgrade/corpus.json"
            corpus.parent.mkdir(parents=True, exist_ok=True)
            corpus.write_text(json.dumps({"baseline": {}, "entries": entries}, indent=1))
        if version is not None:
            source = self.repo / rel
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(main_rs(version))

    def commit(self, message="change"):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def release(self, entries, version, tag="v1.0.0"):
        self.write(entries, version)
        self.commit(f"Release {tag}")
        self.git("tag", tag)

    def run_check(self, root=None, *flags, **env):
        done = subprocess.run(
            [sys.executable, "-I", str(SCRIPT), "--root", str(root or self.repo), *flags],
            env={**GIT_ENV, **env},
            capture_output=True,
            text=True,
        )
        return done.returncode, done.stdout + done.stderr

    def test_unchanged_corpus_passes(self):
        self.release([entry("a"), entry("b")], 1)
        self.write(version=1)
        (self.repo / "README").write_text("later work\n")
        self.commit()
        code, out = self.run_check()
        self.assertEqual(code, 0, out)
        self.assertIn("no corpus entry re-baselined since v1.0.0", out)

    def test_rebaseline_without_bump_fails_naming_the_entries(self):
        self.release([entry("a"), entry("b"), entry("c")], 1)
        self.write([entry("a"), entry("b", sha="bb"), entry("c", sha="cc")])
        self.commit("Re-baseline b and c")
        code, out = self.run_check()
        self.assertEqual(code, 1, out)
        self.assertIn("2 upgrade corpus entries were re-baselined since v1.0.0 (b, c)", out)
        self.assertIn("RENDERING_VERSION is still 1", out)
        # The epoch note can be a CHANGELOG line or a fragment (issue #539).
        self.assertIn("'This release starts a new cache epoch'", out)
        self.assertIn("changelog.d/<issue>.changed.md", out)

    def test_each_golden_field_counts_as_a_rebaseline(self):
        self.release([entry("a")], 1)
        for field, value in (("status", 500), ("content_type", "image/png"), ("bytes", 99)):
            changed = entry("a")
            changed[field] = value
            self.write([changed])
            code, out = self.run_check()
            self.assertEqual(code, 1, f"{field}: {out}")

    def test_rebaseline_with_bump_passes(self):
        self.release([entry("a"), entry("b")], 1)
        self.write([entry("a"), entry("b", sha="bb")], 2)
        self.commit("Re-baseline b and bump")
        code, out = self.run_check()
        self.assertEqual(code, 0, out)
        self.assertIn("covered by RENDERING_VERSION 1 -> 2", out)

    def test_uncommitted_rebaseline_is_checked(self):
        # The local gate runs before the commit: the working tree counts.
        self.release([entry("a")], 1)
        self.write([entry("a", sha="bb")])
        code, out = self.run_check()
        self.assertEqual(code, 1, out)

    def test_added_removed_and_reminted_entries_are_not_rebaselines(self):
        self.release([entry("a"), entry("gone"), entry("sealed", path="/s1.nonce-one")], 1)
        self.write([entry("a"), entry("new", sha="ff"), entry("sealed", path="/s1.nonce-two")])
        self.commit("Freeze again")
        code, out = self.run_check()
        self.assertEqual(code, 0, out)

    def test_later_release_resets_the_comparison(self):
        self.release([entry("a")], 1)
        self.release([entry("a", sha="bb")], 2, tag="v1.1.0")
        code, out = self.run_check()
        self.assertEqual(code, 0, out)
        self.assertIn("since v1.1.0 (RENDERING_VERSION 2)", out)

    def test_version_going_backwards_fails(self):
        self.release([entry("a")], 3)
        self.write(version=2)
        self.commit()
        code, out = self.run_check()
        self.assertEqual(code, 1, out)
        self.assertIn("went backwards", out)

    def test_tag_before_the_constant_counts_as_zero(self):
        self.write([entry("a")])
        (self.repo / "src").mkdir()
        (self.repo / "src/main.rs").write_text("fn main() {}\n")
        self.commit("Release v0.1.0")
        self.git("tag", "v0.1.0")
        self.write([entry("a", sha="bb")], 1)
        self.commit()
        code, out = self.run_check()
        self.assertEqual(code, 0, out)
        self.assertIn("RENDERING_VERSION 0 -> 1", out)

    def test_constant_found_after_moving_files(self):
        self.release([entry("a")], 1)
        (self.repo / "src/main.rs").write_text("fn main() {}\n")
        self.write([entry("a", sha="bb")], 2, rel="src/config/fingerprint.rs")
        self.commit("Move the constant")
        code, out = self.run_check()
        self.assertEqual(code, 0, out)

    def test_missing_constant_fails(self):
        self.release([entry("a")], 1)
        (self.repo / "src/main.rs").write_text("fn main() {}\n")
        code, out = self.run_check()
        self.assertEqual(code, 1, out)
        self.assertIn("RENDERING_VERSION was not found", out)

    def test_unreadable_corpus_fails_with_a_message(self):
        self.release([entry("a")], 1)
        for broken in (
            b"{not json",
            b"[]",
            b'{"entries": [{"sha256": "aa"}]}',
            b'{"entries": [{"id": "a", "note": "\xff"}]}',
            b"[" * 100_000,
        ):
            (self.repo / "tests/upgrade/corpus.json").write_bytes(broken)
            code, out = self.run_check()
            self.assertEqual(code, 1, out)
            self.assertIn("is not a readable corpus", out)
            self.assertNotIn("Traceback", out)

    def test_repeated_entry_id_fails(self):
        self.release([entry("a")], 1)
        self.write([entry("a"), entry("a", sha="bb")])
        code, out = self.run_check()
        self.assertEqual(code, 1, out)
        self.assertIn("repeats an entry id", out)

    def test_no_tag_skips(self):
        self.write([entry("a")], 1)
        self.commit()
        code, out = self.run_check()
        self.assertEqual(code, 0, out)
        self.assertIn("skipped: no release tag", out)

    def test_shallow_clone_without_tags_skips_with_the_fix(self):
        self.release([entry("a")], 1)
        self.write([entry("a", sha="bb")])
        self.commit("Re-baseline after the tag")
        clone = Path(self.tmp.name) / "clone"
        self.git(
            "clone", "-q", "--depth", "1", "--no-tags", self.repo.as_uri(), str(clone),
            cwd=self.tmp.name,
        )
        code, out = self.run_check(clone)
        self.assertEqual(code, 0, out)
        self.assertIn("skipped", out)
        self.assertIn("shallow clone", out)
        self.assertIn("git fetch --tags --unshallow", out)

    def test_not_a_checkout_skips(self):
        plain = Path(self.tmp.name) / "plain"
        plain.mkdir()
        # The ceiling keeps git from finding a repository above the temp dir.
        code, out = self.run_check(plain, GIT_CEILING_DIRECTORIES=self.tmp.name)
        self.assertEqual(code, 0, out)
        self.assertIn("skipped: not a git checkout", out)

    # --strict is how CI runs it: there a skip means the checkout lost the
    # history or the tags, so every way of skipping fails instead.
    def test_strict_fails_where_the_check_would_skip(self):
        self.write([entry("a")], 1)
        self.commit()
        code, out = self.run_check(None, "--strict")
        self.assertEqual(code, 1, out)
        self.assertIn("could not run (--strict): no release tag", out)
        self.assertNotIn("skipped", out)

        plain = Path(self.tmp.name) / "plain"
        plain.mkdir()
        code, out = self.run_check(plain, "--strict", GIT_CEILING_DIRECTORIES=self.tmp.name)
        self.assertEqual(code, 1, out)
        self.assertIn("could not run (--strict): not a git checkout", out)

    def test_strict_fails_in_a_shallow_clone_with_the_fix(self):
        self.release([entry("a")], 1)
        (self.repo / "README").write_text("later work\n")
        self.commit()
        clone = Path(self.tmp.name) / "clone"
        self.git(
            "clone", "-q", "--depth", "1", "--no-tags", self.repo.as_uri(), str(clone),
            cwd=self.tmp.name,
        )
        code, out = self.run_check(clone, "--strict")
        self.assertEqual(code, 1, out)
        self.assertIn("shallow clone", out)
        self.assertIn("git fetch --tags --unshallow", out)

    def test_strict_judges_normally_once_a_tag_is_reachable(self):
        self.release([entry("a"), entry("b")], 1)
        self.write([entry("a"), entry("b", sha="bb")])
        self.commit("Re-baseline after the tag")
        code, out = self.run_check(None, "--strict")
        self.assertEqual(code, 1, out)
        self.assertIn("re-baselined since v1.0.0 (b)", out)
        self.write(version=2)
        self.commit("Bump")
        code, out = self.run_check(None, "--strict")
        self.assertEqual(code, 0, out)
        self.assertIn("covered by RENDERING_VERSION 1 -> 2", out)

    def test_a_hook_environment_leaves_the_callers_repository_alone(self):
        # Run one case in a child that inherits GIT_DIR and GIT_WORK_TREE
        # naming a caller repository with uncommitted work, as a hook would
        # export them. The case must pass and the caller must be untouched.
        caller = Path(self.tmp.name) / "caller"
        caller.mkdir()
        self.git("init", "-q", "-b", "main", cwd=caller)
        (caller / "wip.txt").write_text("uncommitted work\n")
        done = subprocess.run(
            [
                sys.executable, "-I", str(Path(__file__).resolve()),
                "RenderingVersionCheck.test_rebaseline_with_bump_passes",
            ],
            env={**os.environ, "GIT_DIR": str(caller / ".git"), "GIT_WORK_TREE": str(caller)},
            capture_output=True,
            text=True,
        )
        self.assertEqual(done.returncode, 0, done.stderr)

        def caller_git(*args):
            return subprocess.run(
                ["git", *args], cwd=caller, env=GIT_ENV, capture_output=True, text=True
            )

        self.assertNotEqual(caller_git("rev-parse", "--verify", "-q", "HEAD").returncode, 0)
        self.assertEqual(caller_git("tag").stdout, "")
        self.assertEqual(caller_git("status", "--porcelain").stdout, "?? wip.txt\n")


if __name__ == "__main__":
    unittest.main()
