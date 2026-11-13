#!/usr/bin/env python3
"""Tests for scripts/build_dav1d.sh.

A tarball that does not match the pinned hash stops the script before
anything reads it: no source tree, no Python tools, no meson. The linkage
check fails on a binary that links libdav1d and passes one that does not.
`curl`, `python3`, `meson`, `ninja`, `uname`, `ldd` and `otool` are stubs
on PATH, so nothing here downloads or compiles. A real build needs the
network and a compiler: CI's macOS job, the nightly fuzz jobs and every
release run one. Run directly: python3 scripts/test_build_dav1d.py
"""
import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "build_dav1d.sh"


def pin(name):
    found = re.findall(rf"^{name}=(\S+)$", SCRIPT.read_text(encoding="utf-8"), re.M)
    if len(found) != 1:
        raise AssertionError(f"{SCRIPT.name} has {len(found)} {name}= lines, not 1")
    return found[0]


class BuildDav1d(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="crabcrop-dav1d-")
        self.root = Path(self.tmp.name)
        self.stubs = self.root / "stubs"
        self.stubs.mkdir()
        self.calls = self.root / "calls.log"
        # Anything past the checksum check must not run.
        for tool in ("python3", "meson", "ninja", "tar", "pkg-config"):
            self.stub(tool, f'echo {tool} "$@" >> "{self.calls}"\nexit 0\n')
        # A download that brings the wrong bytes.
        self.stub(
            "curl",
            f'echo curl "$@" >> "{self.calls}"\n'
            'while [ "$#" -gt 0 ]; do\n'
            '  if [ "$1" = "-o" ]; then printf "not the dav1d release" > "$2"; fi\n'
            "  shift\n"
            "done\n",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def stub(self, name, body):
        path = self.stubs / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def run_script(self, *args, **env):
        environment = {**os.environ, "PATH": f"{self.stubs}:{os.environ['PATH']}", **env}
        environment.pop("GITHUB_ENV", None)
        return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=environment)

    def called(self):
        return self.calls.read_text().splitlines() if self.calls.exists() else []

    def test_the_pin_is_well_formed(self):
        self.assertRegex(pin("DAV1D_VERSION"), r"^\d+\.\d+\.\d+$")
        self.assertRegex(pin("DAV1D_SHA256"), r"^[0-9a-f]{64}$")
        self.assertEqual(
            pin("DAV1D_URL"),
            '"https://downloads.videolan.org/pub/videolan/dav1d/$DAV1D_VERSION/dav1d-$DAV1D_VERSION.tar.xz"',
        )

    def test_a_wrong_local_tarball_stops_before_the_build(self):
        bad = self.root / "dav1d.tar.xz"
        bad.write_bytes(b"\xfd7zXZ\x00 not the dav1d release")
        out = self.root / "out"
        done = self.run_script(str(out), DAV1D_TARBALL=str(bad))
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("does not match the pinned sha256", done.stderr)
        self.assertEqual(done.stdout, "")
        self.assertEqual(self.called(), [])
        self.assertEqual(sorted(p.name for p in out.iterdir()), [])

    def test_a_wrong_download_stops_before_the_build(self):
        out = self.root / "out"
        done = self.run_script(str(out))
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("does not match the pinned sha256", done.stderr)
        calls = self.called()
        self.assertEqual(len(calls), 1, calls)
        self.assertTrue(calls[0].startswith("curl --proto =https --tlsv1.2 "), calls)
        self.assertIn(f"/dav1d-{pin('DAV1D_VERSION')}.tar.xz", calls[0])
        self.assertEqual(sorted(p.name for p in out.iterdir()), [])

    def test_usage(self):
        for args in ((), ("a", "b")):
            with self.subTest(args=args):
                done = self.run_script(*args)
                self.assertEqual(done.returncode, 1)
                self.assertIn("usage", done.stderr)

    def check_binary(self, system, listing, *files):
        self.stub("uname", f'echo {system}\n')
        tool = "otool" if system == "Darwin" else "ldd"
        listing_file = self.root / "listing.txt"
        listing_file.write_text(listing)
        self.stub(tool, f'cat "{listing_file}"\n')
        return self.run_script("--check-binary", *files)

    def test_the_linkage_check(self):
        binary = self.root / "crabcrop"
        binary.write_bytes(b"\x7fELF")
        static = {
            "Linux": "\tlinux-vdso.so.1\n\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6\n",
            "Darwin": f"{binary}:\n\t/usr/lib/libSystem.B.dylib\n\t/usr/lib/libiconv.2.dylib\n",
        }
        dynamic = {
            "Linux": static["Linux"] + "\tlibdav1d.so.7 => /lib/x86_64-linux-gnu/libdav1d.so.7\n",
            "Darwin": static["Darwin"] + "\t/opt/homebrew/opt/dav1d/lib/libdav1d.7.dylib\n",
        }
        for system in ("Linux", "Darwin"):
            with self.subTest(system=system):
                done = self.check_binary(system, static[system], str(binary))
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertIn("dav1d is linked statically", done.stdout)
                done = self.check_binary(system, dynamic[system], str(binary))
                self.assertEqual(done.returncode, 1)
                self.assertIn("links libdav1d dynamically", done.stderr)

    def test_the_linkage_check_needs_a_binary(self):
        done = self.check_binary("Linux", "")
        self.assertEqual(done.returncode, 1)
        self.assertIn("needs at least one binary", done.stderr)
        done = self.check_binary("Linux", "", str(self.root / "missing"))
        self.assertEqual(done.returncode, 1)
        self.assertIn("no binary at", done.stderr)


if __name__ == "__main__":
    unittest.main()
