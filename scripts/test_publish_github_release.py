#!/usr/bin/env python3
"""Tests for scripts/publish_github_release.sh.

A stub `gh` on PATH records every call and plays a repository with no
release for the tag, a published one, a draft, or a failing API, so each
case asserts the exact commands the release job would run, and that a
published release is refused unless the override is set (issue #552).
Run directly:
python3 scripts/test_publish_github_release.py
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "publish_github_release.sh"
BASH = shutil.which("bash")
COMMIT = "3334fa3407917686510a722c93901adcac6763f4"

STUB = """#!{python}
import json, os, sys
args = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
state = os.environ["STUB_RELEASE"]
if args[:2] == ["release", "view"]:
    if state == "absent":
        sys.exit("release not found")
    if state == "error":
        sys.exit("HTTP 502: Bad Gateway")
    # The script asks for `--jq .isDraft`, which prints true or false.
    print("null" if state == "garbled" else str(state == "draft").lower())
    sys.exit(0)
if args[:2] == ["release", os.environ.get("STUB_FAIL", "")]:
    sys.exit(args[1] + " failed")
if args[:2] == ["release", "create"] and state in ("published", "draft"):
    tag = args[args.index("--") + 1] if "--" in args else args[2]
    sys.exit("a release with the same tag name already exists: " + tag)
"""


class PublishGithubRelease(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.bin = root / "bin"
        self.bin.mkdir()
        gh = self.bin / "gh"
        gh.write_text(STUB.format(python=sys.executable))
        gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
        self.log = root / "gh.log"
        self.notes = root / "notes.txt"
        self.notes.write_text("## Release Notes\n\n- a change\n")
        self.assets = []
        for name in ("crabcrop-x86_64-unknown-linux-gnu.tar.xz", "sha256.sum"):
            asset = root / name
            asset.write_text(name)
            self.assets.append(str(asset))

    def tearDown(self):
        self.tmp.cleanup()

    def publish(self, release, *args, fail="", adopt=None):
        """Run the script against a stub repository; returns (exit code, gh calls).

        `adopt` is the ADOPT_PUBLISHED_RELEASE value, or None to leave it
        unset whatever the caller's environment holds."""
        if not args:
            args = ("v1.2.3", COMMIT, "1.2.3 - 2026-11-07", str(self.notes), "false", *self.assets)
        env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "STUB_LOG": str(self.log),
            "STUB_RELEASE": release,
            "STUB_FAIL": fail,
        }
        env.pop("ADOPT_PUBLISHED_RELEASE", None)
        if adopt is not None:
            env["ADOPT_PUBLISHED_RELEASE"] = adopt
        done = subprocess.run([BASH, str(SCRIPT), *args], env=env, capture_output=True, text=True)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        self.output = done.stdout + done.stderr
        return done.returncode, calls

    def view(self, tag="v1.2.3"):
        return ["release", "view", "--json", "isDraft", "--jq", ".isDraft", "--", tag]

    def create(self, tag="v1.2.3", title="1.2.3 - 2026-11-07", prerelease="false", assets=None):
        return [
            "release", "create", "--target", COMMIT, f"--prerelease={prerelease}",
            "--title", title, "--notes-file", str(self.notes), "--", tag, *(assets or self.assets),
        ]

    def upload(self, tag="v1.2.3", assets=None):
        return ["release", "upload", "--clobber", "--", tag, *(assets or self.assets)]

    def edit(self, tag="v1.2.3", title="1.2.3 - 2026-11-07", prerelease="false"):
        return [
            "release", "edit", "--target", COMMIT, "--title", title, "--notes-file", str(self.notes),
            f"--prerelease={prerelease}", "--draft=false", "--", tag,
        ]

    def test_creates_the_release_when_none_exists(self):
        code, calls = self.publish("absent")
        self.assertEqual(code, 0, self.output)
        self.assertEqual(calls, [self.view(), self.create()])

    def test_creates_a_prerelease(self):
        args = ("v0.10.0-rc.1", COMMIT, "0.10.0-rc.1", str(self.notes), "true", *self.assets)
        code, calls = self.publish("absent", *args)
        self.assertEqual(code, 0, self.output)
        self.assertEqual(calls, [self.view("v0.10.0-rc.1"), self.create("v0.10.0-rc.1", "0.10.0-rc.1", "true")])

    def test_refuses_a_published_release(self):
        # v0.10.0: the release was published by hand, which created the
        # tag, 37 minutes before the gated run attached its assets. The
        # script adopted it silently; now it stops before any change and
        # names the release and the fix.
        code, calls = self.publish("published")
        self.assertEqual(code, 1, self.output)
        self.assertEqual(calls, [self.view()])
        self.assertIn("::error title=Release v1.2.3 is already published::", self.output)
        self.assertIn("the release for v1.2.3 is already published; refusing to adopt it", self.output)
        self.assertIn("gh release edit v1.2.3 --draft=true", self.output)
        self.assertIn("re-run only the failed jobs", self.output)
        self.assertIn("ADOPT_PUBLISHED_RELEASE", self.output)
        self.assertIn("An immutable release takes no new assets", self.output)

    def test_the_override_adopts_a_published_release_with_a_warning(self):
        code, calls = self.publish("published", adopt="1")
        self.assertEqual(code, 0, self.output)
        self.assertEqual(calls, [self.view(), self.upload(), self.edit()])
        self.assertIn("::warning title=Adopting published release v1.2.3::", self.output)

    def test_only_1_is_the_override(self):
        # An empty repository variable reaches the script as an empty
        # string; anything but 1 refuses.
        for value in ("", "0", "true", "yes", " 1"):
            with self.subTest(value=value):
                self.log.unlink(missing_ok=True)
                code, calls = self.publish("published", adopt=value)
                self.assertEqual(code, 1, self.output)
                self.assertEqual(calls, [self.view()])

    def test_a_draft_needs_no_override(self):
        code, calls = self.publish("draft", adopt="")
        self.assertEqual(code, 0, self.output)
        self.assertEqual(calls, [self.view(), self.upload(), self.edit()])
        self.assertNotIn("::warning", self.output)

    def test_an_unreadable_draft_flag_stops(self):
        code, calls = self.publish("garbled")
        self.assertEqual(code, 1, self.output)
        self.assertEqual(calls, [self.view()])
        self.assertIn("cannot tell whether the release for v1.2.3 is a draft", self.output)

    def test_publishes_an_existing_draft(self):
        code, calls = self.publish("draft")
        self.assertEqual(code, 0, self.output)
        self.assertEqual(calls, [self.view(), self.upload(), self.edit()])

    def test_update_keeps_the_prerelease_flag_explicit(self):
        args = ("v0.10.0-rc.1", COMMIT, "0.10.0-rc.1", str(self.notes), "true", *self.assets)
        code, calls = self.publish("draft", *args)
        self.assertEqual(code, 0, self.output)
        self.assertEqual(calls[-1], self.edit("v0.10.0-rc.1", "0.10.0-rc.1", "true"))

    def test_a_failed_lookup_falls_back_to_create(self):
        # Not the not-found answer: the create runs as before and reports
        # whatever is really wrong.
        code, calls = self.publish("error")
        self.assertEqual(code, 0, self.output)
        self.assertEqual(calls, [self.view(), self.create()])

    def test_a_failed_upload_leaves_the_release_unpublished(self):
        code, calls = self.publish("draft", fail="upload")
        self.assertNotEqual(code, 0)
        self.assertEqual(calls, [self.view(), self.upload()])

    def test_a_failed_create_fails_the_step(self):
        code, calls = self.publish("absent", fail="create")
        self.assertNotEqual(code, 0)
        self.assertEqual(calls, [self.view(), self.create()])

    def test_dash_led_values_stay_positional(self):
        root = Path(self.tmp.name)
        asset = root / "-x.tar.xz"
        asset.write_text("x")
        tag, title = "-v1.2.3", "--draft"
        for release in ("absent", "draft", "published"):
            with self.subTest(release=release):
                self.log.unlink(missing_ok=True)
                code, calls = self.publish(
                    release, tag, COMMIT, title, str(self.notes), "false", str(asset), adopt="1"
                )
                self.assertEqual(code, 0, self.output)
                self.assertEqual(calls[0], self.view(tag))
                for call in calls[1:]:
                    # Everything after `--` is the tag and the assets.
                    self.assertEqual(call[call.index("--") + 1], tag)
                    if "--title" in call:
                        self.assertEqual(call[call.index("--title") + 1], title)

    def test_values_are_passed_verbatim(self):
        title = "1.2.3 $(touch pwned) `id` ; rm -rf / \"quoted\"\nsecond line"
        code, calls = self.publish("absent", "v1.2.3", COMMIT, title, str(self.notes), "false", *self.assets)
        self.assertEqual(code, 0, self.output)
        self.assertEqual(calls[-1], self.create(title=title))
        self.assertFalse((Path.cwd() / "pwned").exists())

    def test_invalid_arguments_make_no_gh_call(self):
        missing = str(Path(self.tmp.name) / "missing")
        good = ["v1.2.3", COMMIT, "title", str(self.notes), "false", *self.assets]
        cases = {
            "too few arguments": good[:5],
            "no assets": good[:5],
            "empty tag": ["", *good[1:]],
            "empty commit": [good[0], "", *good[2:]],
            "empty title": [*good[:2], "", *good[3:]],
            "missing notes": [*good[:3], missing, *good[4:]],
            "prerelease yes": [*good[:4], "yes", *good[5:]],
            "prerelease empty": [*good[:4], "", *good[5:]],
            "prerelease flag": [*good[:4], "--prerelease", *good[5:]],
            "missing asset": [*good, missing],
            "directory asset": [*good, self.tmp.name],
        }
        for name, args in cases.items():
            with self.subTest(case=name):
                self.log.unlink(missing_ok=True)
                code, calls = self.publish("absent", *args)
                self.assertEqual(code, 2, self.output)
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
