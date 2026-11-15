#!/usr/bin/env python3
"""Tests for scripts/check_notices.py.

Each case builds a throwaway checkout (dist's config, the generator's
config and template, a fake registry of crates) and stands a fake in for
cargo and cargo-about: it answers `cargo tree` and `cargo-about generate`
from the crates the case declares, as the real tools would, so the cases
need no network, no cargo-about and no compiler. One case checks this
repository's own file. Run directly: python3 scripts/test_check_notices.py
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "check_notices.py"
sys.path.insert(0, str(HERE))

import check_notices  # noqa: E402

TARGETS = ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"]
MIT = "MIT License\n\nCopyright (c) {who}\n\nPermission is hereby granted, free of charge.\n"
APACHE = "Apache License\nVersion 2.0, January 2004\n"
VENDORED = "Copyright (c) 2010, Vendor Inc. All rights reserved.\n\nRedistribution in binary form keeps this notice.\n"
DOWNLOADED = "Copyright 2018, Downloaded Library authors\n\nKeep this notice.\n"
PIN = "ab" * 32


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


TEMPLATE = f"""# Third-party notices

<!-- stamp -->

The binary also carries other people's code.

## Native libraries

<!-- include: crate=vendored-sys path=vendor/COPYING sha256={digest(VENDORED)} -->

<!-- copyrights: crate=vendored-sys glob=vendor/**/*.c -->

<!-- pinned: file=scripts/build_lib.sh sha256={PIN} -->

<!-- include: file=scripts/notices/lib/COPYING label="`COPYING` in lib 1.0" sha256={digest(DOWNLOADED)} -->

## Rust crates

<!-- crates -->

<!-- source-offers -->

<!-- notice-files -->

## License texts

<!-- licenses -->
"""


class Fake:
    """cargo and cargo-about, answering from the crates a case declares.

    A crate uses the licenses `uses` maps to the file holding each text,
    or None for a license whose text the crate does not ship (cargo-about
    then reproduces the license's standard text, with no source path)."""

    def __init__(self, registry):
        self.registry = registry
        self.crates = {}
        self.version = check_notices.CARGO_ABOUT_VERSION
        self.failure = None
        self.warnings = ""
        self.calls = []

    def add(self, name, version, license, uses, files=None, targets=None, source=check_notices.CRATES_IO):
        root = self.registry / f"{name}-{version}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "Cargo.toml").write_text(f'[package]\nname = "{name}"\nversion = "{version}"\n')
        for rel, text in (files or {}).items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        self.crates[(name, version)] = {
            "license": license,
            "uses": uses,
            "targets": targets,
            "source": source,
            "root": root,
        }

    def package(self, name, version, source, root):
        return {
            "name": name,
            "version": version,
            "source": source,
            "manifest_path": str(root / "Cargo.toml"),
            "repository": f"https://example.invalid/{name}",
        }

    def findings(self):
        """The JSON cargo-about prints: one entry per distinct text."""
        workspace = self.registry.parent / "workspace"
        first_party = self.package("crabcrop", "0.0.0", None, workspace)
        crates = [{"package": first_party, "license": "MIT"}]
        texts = {("MIT", MIT.format(who="crabcrop")): {"source_path": str(workspace / "LICENSE"), "used_by": [first_party]}}
        for (name, version), crate in sorted(self.crates.items()):
            package = self.package(name, version, crate["source"], crate["root"])
            crates.append({"package": package, "license": crate["license"]})
            for license_id, rel in crate["uses"].items():
                if rel is None:
                    text, path = f"The standard {license_id} text.\n", None
                else:
                    text, path = (crate["root"] / rel).read_text(), str(crate["root"] / rel)
                entry = texts.setdefault((license_id, text), {"source_path": path, "used_by": []})
                entry["used_by"].append(package)
        names = {"MIT": "MIT License", "Apache-2.0": "Apache License 2.0", "MPL-2.0": "Mozilla Public License 2.0"}
        licenses = [
            {
                "name": names.get(license_id, license_id),
                "id": license_id,
                "text": text,
                "source_path": entry["source_path"],
                "used_by": [{"crate": package, "path": None} for package in entry["used_by"]],
            }
            for (license_id, text), entry in sorted(texts.items())
        ]
        return {"overview": [], "licenses": licenses, "crates": crates}

    def __call__(self, cmd, cwd):
        self.calls.append(cmd)
        if cmd[0] == "fake-cargo-about":
            if cmd[1:] == ["--version"]:
                return 0, f"cargo-about {self.version}\n", ""
            if self.failure:
                return 1, "", self.failure
            return 0, json.dumps(self.findings()), self.warnings
        if cmd[0] == "cargo" and cmd[1] == "fetch":
            return 0, "", ""
        if cmd[0] == "cargo" and cmd[1] == "tree":
            target = cmd[cmd.index("--target") + 1]
            lines = [f"crabcrop v0.0.0 ({self.registry.parent / 'workspace'})"]
            for (name, version), crate in sorted(self.crates.items()):
                if crate["targets"] is None or target in crate["targets"]:
                    lines += [f"{name} v{version}", f"{name} v{version} (*)"]
            return 0, "\n".join(lines) + "\n", ""
        raise AssertionError(f"unexpected command: {cmd}")


class CheckNotices(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "checkout"
        self.fake = Fake(base / "registry")
        (base / "registry").mkdir()
        self.write("dist-workspace.toml", self.dist())
        self.write("scripts/release_formula.py", f"    {check_notices.FORMULA_INSTALL}\n")
        self.write("scripts/notices/about.toml", 'accepted = [\n    "MIT",\n    "Apache-2.0",\n    "MPL-2.0",\n]\n')
        self.write("scripts/notices/template.md", TEMPLATE)
        self.write("scripts/notices/lib/COPYING", DOWNLOADED)
        self.write("scripts/build_lib.sh", f"LIB_SHA256={PIN}\n")
        self.fake.add("alpha", "1.0.0", "MIT", {"MIT": "LICENSE"}, {"LICENSE": MIT.format(who="alpha")})
        self.fake.add("beta", "2.0.0", "MIT OR Apache-2.0", {"MIT": "LICENSE-MIT"}, {"LICENSE-MIT": MIT.format(who="alpha")})
        self.fake.add(
            "vendored-sys",
            "0.1.0",
            "MIT",
            {"MIT": None},
            {
                "vendor/COPYING": VENDORED,
                "vendor/src/a.c": "/*\n * Copyright (c) 2010, Vendor Inc.\n * Copyright 2012 Someone Else\n */\nint a;\n",
                "vendor/src/b.h": "/* Copyright 1999 Not In The Glob */\n",
                "NOTICE": "This crate includes code from the Vendor project.\n",
            },
        )
        self.fake.add("mp4parse", "0.17.0", "MPL-2.0", {"MPL-2.0": "LICENSE"}, {"LICENSE": "Mozilla Public License\n"})
        self.fake.add("darwin-only", "0.3.0", "Apache-2.0", {"Apache-2.0": "LICENSE"}, {"LICENSE": APACHE}, targets=["aarch64-apple-darwin"])
        self.env = mock.patch.dict(os.environ, {"CARGO_ABOUT": "fake-cargo-about"})
        self.env.start()
        os.environ.pop("CARGO", None)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def dist(self, include='include = ["THIRD-PARTY-NOTICES.md"]\n'):
        return f'[dist]\ntargets = {json.dumps(TARGETS)}\nfeatures = ["avif-decode"]\n{include}'

    def write(self, rel, text):
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.root / rel).write_text(text)

    def read(self, rel=check_notices.NOTICES):
        return (self.root / rel).read_text()

    def main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = check_notices.main([*args, "--root", str(self.root)], runner=self.fake)
        return code, out.getvalue() + err.getvalue()

    def generate(self):
        code, output = self.main("--write")
        self.assertEqual(code, 0, output)
        return self.read()

    def check(self, strict=True):
        return check_notices.check(self.root, strict=strict, runner=self.fake)

    def assert_fails(self, expected, strict=True):
        problems, _ = self.check(strict)
        text = "\n".join(problems)
        self.assertTrue(problems, "the check passed")
        for line in expected:
            self.assertIn(line, text)
        return text

    def test_this_repository_passes(self):
        # The crate table against cargo tree, the stamp, the pins and the
        # shipping config. Regenerating needs cargo-about, which CI's
        # notices job installs.
        if not check_notices.shutil.which(check_notices.cargo()):
            self.skipTest("no cargo to list the release build's crates with")
        problems, _ = check_notices.check(check_notices.ROOT, regenerate=False)
        self.assertEqual(problems, [])

    def test_a_generated_file_passes(self):
        text = self.generate()
        problems, notes = self.check()
        self.assertEqual(problems, [])
        self.assertIn(f"regenerated with cargo-about {check_notices.CARGO_ABOUT_VERSION}: unchanged", notes)
        self.assertTrue(text.startswith("# Third-party notices\n\n<!-- Generated by scripts/check_notices.py"))
        code, output = self.main()
        self.assertEqual(code, 0, output)
        self.assertIn("lists the 5 crates the release build compiles (2 targets, features: avif-decode)", output)

    def test_the_rendering(self):
        text = self.generate()
        # The workspace's own crates are not third-party notices.
        self.assertNotIn("crabcrop 0.0.0", text)
        self.assertNotIn("Copyright (c) crabcrop", text)
        # A crate one target alone builds is listed; the table names the
        # declared license and the one whose text is reproduced.
        self.assertIn("| darwin-only | 0.3.0 | Apache-2.0 | Apache-2.0 |", text)
        self.assertIn("| beta | 2.0.0 | MIT OR Apache-2.0 | MIT |", text)
        self.assertIn("That build compiles these 5 crates", text)
        # One text two crates share appears once, crediting both.
        self.assertEqual(text.count("Copyright (c) alpha"), 1)
        self.assertIn("Used by alpha 1.0.0 and beta 2.0.0. From `LICENSE` in alpha 1.0.0:", text)
        self.assertIn("Used by vendored-sys 0.1.0, which ship no copy of the license; this is its standard text:", text)
        # The template's copies: a vendored file, pinned by digest...
        self.assertIn(f"From `vendor/COPYING` in vendored-sys 0.1.0:\n\n```text\n{VENDORED}```", text)
        # ...the copyright lines of the files a glob names...
        self.assertIn(
            "From the `vendor/**/*.c` files of vendored-sys 0.1.0:\n\n"
            "```text\nCopyright (c) 2010, Vendor Inc.\nCopyright 2012 Someone Else\n```",
            text,
        )
        self.assertNotIn("Not In The Glob", text)
        # ...and a text copied from a download the build pins.
        self.assertIn(f"From `COPYING` in lib 1.0:\n\n```text\n{DOWNLOADED}```", text)
        self.assertNotIn("<!-- pinned", text)
        self.assertNotIn("<!-- include", text)
        # The MPL crate's source code, and every NOTICE file.
        self.assertIn("- mp4parse 0.17.0 (MPL-2.0): https://crates.io/api/v1/crates/mp4parse/0.17.0/download", text)
        self.assertIn("From `NOTICE` in vendored-sys 0.1.0:\n\n```text\nThis crate includes code", text)
        self.assertNotIn("\n\n\n", text)

    def test_a_fence_outlasts_the_backticks_in_a_text(self):
        self.fake.add("ticks", "1.0.0", "MIT", {"MIT": "LICENSE"}, {"LICENSE": "Copyright ticks\n````\nnot a fence\n"})
        text = self.generate()
        self.assertIn("`````text\nCopyright ticks\n````\nnot a fence\n`````", text)

    def test_a_new_crate_fails(self):
        self.generate()
        self.fake.add("gamma", "0.3.0", "MIT", {"MIT": "LICENSE"}, {"LICENSE": MIT.format(who="gamma")})
        self.assert_fails(
            [
                "the release build compiles gamma 0.3.0, which THIRD-PARTY-NOTICES.md does not list",
                "THIRD-PARTY-NOTICES.md is not what cargo-about",
                "+| gamma | 0.3.0 | MIT | MIT |",
                "regenerate it with python3 scripts/check_notices.py --write",
            ]
        )

    def test_a_new_crate_fails_without_cargo_about(self):
        # The release preflight installs no cargo-about: cargo tree alone
        # finds a crate the file does not list.
        self.generate()
        self.fake.add("gamma", "0.3.0", "MIT", {"MIT": "LICENSE"}, {"LICENSE": MIT.format(who="gamma")})
        with mock.patch.dict(os.environ, {"CARGO_ABOUT": ""}), mock.patch.object(check_notices.shutil, "which", return_value=None):
            text = self.assert_fails(["the release build compiles gamma 0.3.0"], strict=False)
        self.assertNotIn("not what cargo-about", text)

    def test_a_moved_version_fails(self):
        self.generate()
        crate = self.fake.crates.pop(("beta", "2.0.0"))
        self.fake.add("beta", "2.1.0", crate["license"], crate["uses"], {"LICENSE-MIT": MIT.format(who="alpha")})
        self.assert_fails(["the release build compiles beta 2.1.0, and THIRD-PARTY-NOTICES.md lists beta 2.0.0"])

    def test_a_dropped_crate_fails(self):
        self.generate()
        del self.fake.crates[("darwin-only", "0.3.0")]
        self.assert_fails(["THIRD-PARTY-NOTICES.md lists darwin-only 0.3.0, which the release build no longer compiles"])

    def test_a_hand_edit_fails(self):
        text = self.generate()
        self.write(check_notices.NOTICES, text.replace("other people's code", "other people's fine code"))
        self.assert_fails(["THIRD-PARTY-NOTICES.md was edited by hand"])

    def test_a_changed_input_fails_before_regenerating(self):
        self.generate()
        self.write("scripts/notices/template.md", TEMPLATE.replace("other people's code", "code by others"))
        with mock.patch.dict(os.environ, {"CARGO_ABOUT": ""}), mock.patch.object(check_notices.shutil, "which", return_value=None):
            self.assert_fails([f"{check_notices.TEMPLATE}, a file it copies, or dist's targets or features changed"], strict=False)
        self.write("dist-workspace.toml", self.dist().replace('["avif-decode"]', "[]"))
        self.write("scripts/notices/template.md", TEMPLATE)
        self.assert_fails(["dist's targets or features changed since THIRD-PARTY-NOTICES.md was generated"])

    def test_without_cargo_about_the_check_says_what_it_skipped(self):
        self.generate()
        with mock.patch.dict(os.environ, {"CARGO_ABOUT": ""}), mock.patch.object(check_notices.shutil, "which", return_value=None):
            problems, notes = self.check(strict=False)
            self.assertEqual(problems, [])
            self.assertEqual(notes, [f"not regenerated: cargo-about {check_notices.CARGO_ABOUT_VERSION} is not installed ({check_notices.INSTALL})"])
            self.assert_fails([f"cargo-about {check_notices.CARGO_ABOUT_VERSION} is not installed ({check_notices.INSTALL})"])
            code, output = self.main("--write")
            self.assertEqual(code, 1)
            self.assertIn(check_notices.INSTALL, output)

    def test_another_cargo_about_version_fails(self):
        self.generate()
        self.fake.version = "0.9.2"
        self.assert_fails(["fake-cargo-about is cargo-about 0.9.2, and the notices are generated with"])

    def test_an_unknown_license_fails(self):
        self.generate()
        self.fake.failure = (
            "error: failed to satisfy license requirements\n"
            "  ┌─ registry/gpl-crate-1.0.0/Cargo.toml:5:12\n"
            '5 │ license = "GPL-3.0-only"\n'
            "\x1b[31mERROR\x1b[0m encountered 1 errors resolving licenses, unable to generate output\n"
        )
        text = self.assert_fails(["cargo-about failed:", 'license = "GPL-3.0-only"', "accept it there with its reason"])
        self.assertNotIn("\x1b[", text)

    def test_a_cargo_about_warning_fails(self):
        self.generate()
        self.fake.warnings = "2026-11-15T16:57:10Z \x1b[33mWARN\x1b[0m config found for a crate not present in the graph\n"
        self.fake.warnings = self.fake.warnings.replace("\x1b[33mWARN\x1b[0m", "[\x1b[33mWARN\x1b[0m]")
        self.assert_fails(["cargo-about warned", "config found for a crate not present in the graph"])

    def test_an_accepted_license_no_crate_needs_fails(self):
        self.write("scripts/notices/about.toml", 'accepted = ["MIT", "Apache-2.0", "MPL-2.0", "ISC"]\n')
        code, output = self.main("--write")
        self.assertEqual(code, 1)
        self.assertIn("accepts ISC, which no crate in the release build needs: remove it", output)

    def test_a_changed_vendored_text_stops_the_generator(self):
        (self.fake.crates[("vendored-sys", "0.1.0")]["root"] / "vendor/COPYING").write_text(VENDORED + "New term.\n")
        code, output = self.main("--write")
        self.assertEqual(code, 1)
        self.assertIn(
            f"the text of `vendor/COPYING` in vendored-sys 0.1.0 has changed (sha256 {digest(VENDORED + 'New term.' + chr(10))}",
            output,
        )
        self.assertFalse((self.root / check_notices.NOTICES).exists())

    def test_a_moved_download_pin_fails(self):
        self.generate()
        self.write("scripts/build_lib.sh", "LIB_SHA256=" + "cd" * 32 + "\n")
        self.assert_fails([f"scripts/build_lib.sh no longer pins the download with sha256 {PIN}"])

    def test_an_artifact_that_drops_the_file_fails(self):
        self.generate()
        self.write("dist-workspace.toml", self.dist(include=""))
        self.write("scripts/release_formula.py", '    bin.install "crabcrop"\n')
        self.assert_fails(
            [
                "dist-workspace.toml: `include` does not name THIRD-PARTY-NOTICES.md",
                f"scripts/release_formula.py: the formula does not `{check_notices.FORMULA_INSTALL}`",
            ]
        )

    def test_a_missing_file_fails(self):
        done = subprocess.run([sys.executable, str(SCRIPT), "--root", str(self.root)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("THIRD-PARTY-NOTICES.md: cannot read it", done.stderr)

    def test_template_mistakes_are_refused(self):
        cases = {
            "<!-- include: crate=a path=b sha265=c -->": "the include directive takes no sha265=",
            "<!-- include: crate=a file=b sha256=c -->": "an include names either crate= and path=, or file= and label=",
            "<!-- copyright: crate=a -->": "unknown directive",
            "<!-- pinned: file=a stray -->": "`stray` is not key=value",
        }
        for line, message in cases.items():
            with self.subTest(line=line):
                with self.assertRaises(check_notices.NoticeError) as caught:
                    check_notices.parse_template(TEMPLATE + line + "\n")
                self.assertIn(message, str(caught.exception))
        with self.assertRaises(check_notices.NoticeError) as caught:
            check_notices.parse_template(TEMPLATE.replace("<!-- licenses -->", ""))
        self.assertIn("needs exactly one `<!-- licenses -->` line, found 0", str(caught.exception))

    def test_the_formula_installs_the_file(self):
        sums = self.root / "sha256.sum"
        sums.write_text(
            "".join(
                f"{'0' * 64}  crabcrop-{target}.tar.xz\n"
                for target in ["aarch64-apple-darwin", "x86_64-apple-darwin", "aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"]
            )
        )
        done = subprocess.run(
            [sys.executable, str(HERE / "release_formula.py"), "1.2.3", str(sums)], capture_output=True, text=True
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(f'    bin.install "crabcrop"\n    {check_notices.FORMULA_INSTALL}\n', done.stdout)


if __name__ == "__main__":
    unittest.main()
