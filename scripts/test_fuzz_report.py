#!/usr/bin/env python3
"""Tests for scripts/fuzz_report.py (issue #710).

A cargo-fuzz run that fails prints the failing input several ways, and
the nightly workflow's logs are public. Each case writes a log the way
libFuzzer and cargo-fuzz print one, with an input planted in every place
they print it, and checks that the report keeps what a maintainer needs
(where it failed, the run's statistics, the input's hash and size) and
none of the input. Run directly: python3 scripts/test_fuzz_report.py
"""
import base64
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "fuzz_report.py"
sys.path.insert(0, str(HERE))

import fuzz_report  # noqa: E402

# A failing input that looks like libFuzzer's own lines, so a filter that
# matched it line by line would keep it.
INPUT = b"INFO: secret-input-7f3a/unsafe/filters:rotate(91)/example.com/x.jpg"
SHA1 = hashlib.sha1(INPUT).hexdigest()
HEX = ",".join(f"0x{byte:x}" for byte in INPUT) + ","
DEBUG = "\n".join(["\t["] + [f"\t    {byte}," for byte in INPUT] + ["\t]"])

FAILED_RUN = f"""\
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 1977046021
INFO: Loaded 1 modules   (226151 inline 8-bit counters): 226151 [0x5613d2c9bd28, 0x5613d2cd318f),
INFO:       78 files found in /home/runner/work/_temp/corpus/parse_request
INFO: seed corpus: files: 78 min: 1b max: 1009b total: 11000b rss: 48Mb
#79\tINITED cov: 5241 ft: 9932 corp: 64/8711b exec/s: 0 rss: 61Mb
#1024\tpulse  cov: 5302 ft: 10144 corp: 70/9012b lim: 1009 exec/s: 512 rss: 70Mb
#1093\tNEW    cov: 5303 ft: 10145 corp: 71/9030b lim: 1009 exec/s: 546 rss: 70Mb L: 18/18 MS: 2 ChangeByte-CMP- DE: "secret-input-7f3a"-
thread '<unnamed>' (4321) panicked at fuzz/src/lib.rs:123:9:
assertion `left == right` failed: canonical form is not a fixed point
  left: "secret-input-7f3a"
 right: "secret-input-7f3b"
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
==4321== ERROR: libFuzzer: deadly signal
    #0 0x5613d1f2c3a1 in __sanitizer_print_stack_trace /rustc/llvm/compiler-rt/lib/asan/asan_stack.cpp:87:3
    #1 0x5613d2a1b2c3 in crabcrop_fuzz::check_round_trip /home/runner/work/crabcrop/crabcrop/fuzz/src/lib.rs:123:9
NOTE: libFuzzer has rudimentary signal handlers.
      Combine libFuzzer with AddressSanitizer or similar for better crash reports.
SUMMARY: libFuzzer: deadly signal
MS: 1 InsertByte- DE: "secret-input-7f3a"-; base unit: 0a1b2c3d4e5f60718293a4b5c6d7e8f901234567
{HEX}
{INPUT.decode()}
artifact_prefix='fuzz/artifacts/parse_request/'; Test unit written to fuzz/artifacts/parse_request/crash-{SHA1}
Base64: {base64.b64encode(INPUT).decode()}

Failing input:

\tfuzz/artifacts/parse_request/crash-{SHA1}

Output of `std::fmt::Debug`:

{DEBUG}

Reproduce with:

\tcargo fuzz run parse_request fuzz/artifacts/parse_request/crash-{SHA1}

Error: Fuzz target exited with exit status: 77
"""

CLEAN_RUN = """\
INFO: Seed: 42
#79\tINITED cov: 5241 ft: 9932 corp: 64/8711b exec/s: 0 rss: 61Mb
#2097152\tpulse  cov: 5310 ft: 10200 corp: 75/9500b lim: 1009 exec/s: 2330 rss: 300Mb
#2100000\tDONE   cov: 5310 ft: 10201 corp: 75/9501b lim: 1009 exec/s: 2333 rss: 300Mb
Done 2100000 runs in 900 second(s)
stat::number_of_executed_units: 2100000
stat::peak_rss_mb:              300
"""


class Report(unittest.TestCase):
    def run_report(self, log, artifacts=None):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "fuzz.log"
            log_path.write_text(log, encoding="utf-8")
            directory = Path(tmp) / "artifacts"
            if artifacts is not None:
                directory.mkdir()
                for name, data in artifacts.items():
                    (directory / name).write_bytes(data)
            done = subprocess.run(
                [sys.executable, str(SCRIPT), str(log_path), str(directory)],
                capture_output=True,
                text=True,
            )
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout

    def test_a_failed_run_is_reported_without_its_input(self):
        out = self.run_report(FAILED_RUN, {f"crash-{SHA1}": INPUT})
        for planted in (
            "secret-input-7f3a",
            "secret-input-7f3b",
            HEX,
            base64.b64encode(INPUT).decode(),
            str(INPUT[0]) + ",",
        ):
            self.assertNotIn(planted, out)
        self.assertFalse([line for line in out.splitlines() if line.startswith("\t")], out)

    def test_a_failed_run_keeps_where_and_how_it_failed(self):
        out = self.run_report(FAILED_RUN, {f"crash-{SHA1}": INPUT})
        for kept in (
            "INFO: Seed: 1977046021",
            "#79\tINITED cov: 5241",
            "#1024\tpulse  cov: 5302",
            "thread '<unnamed>' (4321) panicked at fuzz/src/lib.rs:123:9:",
            "==4321== ERROR: libFuzzer: deadly signal",
            "    #1 0x5613d2a1b2c3 in crabcrop_fuzz::check_round_trip",
            "SUMMARY: libFuzzer: deadly signal",
            f"Test unit written to fuzz/artifacts/parse_request/crash-{SHA1}",
            f"failing input (crash): sha256 {hashlib.sha256(INPUT).hexdigest()}, {len(INPUT)} bytes",
        ):
            self.assertIn(kept, out)
        self.assertNotIn("NEW    cov", out)
        self.assertRegex(out, r"\(\d+ lines of the run's output left out")

    def test_every_kind_of_artifact_is_named(self):
        artifacts = {"leak-aa": b"x", "oom-bb": b"yy", "slow-unit-cc": b"zzz", "timeout-dd": b"w"}
        out = self.run_report(CLEAN_RUN, artifacts)
        for kind, data in (("leak", b"x"), ("oom", b"yy"), ("slow-unit", b"zzz"), ("timeout", b"w")):
            self.assertIn(f"failing input ({kind}): sha256 {hashlib.sha256(data).hexdigest()}, {len(data)} bytes", out)

    def test_a_clean_run_keeps_its_statistics(self):
        out = self.run_report(CLEAN_RUN)
        for kept in ("#2100000\tDONE", "Done 2100000 runs in 900 second(s)", "stat::number_of_executed_units: 2100000"):
            self.assertIn(kept, out)
        self.assertIn("no failing input", out)

    def test_a_rust_backtrace_keeps_its_frames(self):
        log = (
            "thread '<unnamed>' panicked at src/engine.rs:10:5:\n"
            "index out of bounds: the len is 0 but the index is 7f3a\n"
            "stack backtrace:\n"
            "   0: rust_begin_unwind\n"
            "   1: crabcrop::engine::decode_image\n"
            "             at /home/runner/work/crabcrop/crabcrop/src/engine.rs:10:5\n"
            "==12== ERROR: libFuzzer: deadly signal\n"
        )
        kept, _ = fuzz_report.report_lines(log)
        self.assertEqual(
            kept,
            [
                "thread '<unnamed>' panicked at src/engine.rs:10:5:",
                "stack backtrace:",
                "   0: rust_begin_unwind",
                "   1: crabcrop::engine::decode_image",
                "             at /home/runner/work/crabcrop/crabcrop/src/engine.rs:10:5",
                "==12== ERROR: libFuzzer: deadly signal",
            ],
        )

    def test_an_unreadable_log_fails(self):
        done = subprocess.run([sys.executable, str(SCRIPT), "/nonexistent/fuzz.log", "/nonexistent"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 1)
        self.assertIn("cannot read the run's log", done.stderr)


if __name__ == "__main__":
    unittest.main()
