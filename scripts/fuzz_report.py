#!/usr/bin/env python3
"""Report a cargo-fuzz run without the input it failed on (issue #710).

The nightly fuzz workflow runs in a public repository, where every job
log and artifact is public, and a crashing input is a reproducer: until
its fix ships, it is a way to crash or stall the server. The run's own
output carries the input whenever it fails: libFuzzer prints an input of
up to 256 bytes in hex, as text, and in base64; cargo-fuzz prints its
`Debug` form; a harness assertion can quote it. So the workflow sends
that output to a file, uploads nothing, and runs this script, which
prints only what cannot carry the input:

- libFuzzer's own lines: its INFO lines, its progress at start-up, on
  each power-of-two pulse and at the end (not the lines for each new or
  reduced input, whose dictionary entries quote input bytes), its final
  statistics, and its ERROR and SUMMARY lines, as the sanitizers'
  are;
- where a panic happened (the `panicked at <file>:<line>:<column>:`
  line, never the message after it) and the stack frames;
- cargo-fuzz's own `Error:` line, which says how the run ended;
- each input the run left in the artifacts directory, by its kind
  (crash, leak, timeout, oom, slow-unit), SHA-256 and size.

Everything else is left out and counted. fuzz/README.md ("Real fuzzing")
says how to work from the report.

Usage: python3 scripts/fuzz_report.py LOG ARTIFACTS_DIR
"""
import hashlib
import re
import sys
from pathlib import Path

# Lines that can carry no input bytes. Each pattern is anchored, and a
# line libFuzzer prints the input on is dropped before these are tried.
KEEP = [
    # Progress without the mutation details: `#1024 pulse cov: ...`.
    re.compile(r"#\d+\s+(?:INITED|pulse|DONE|RELOAD)\b[\w :/.-]*"),
    re.compile(r"INFO: .*"),
    re.compile(r"stat::\w+: *\d+"),
    re.compile(r"Done \d+ runs in \d+ second\(s\)"),
    # libFuzzer's and the sanitizers' error and summary lines.
    re.compile(r"==\d+== ?(?:ERROR|WARNING): .*"),
    re.compile(r"SUMMARY: .*"),
    re.compile(r"(?:READ|WRITE) of size \d+ at 0x[0-9a-f]+ thread \w+"),
    # A sanitizer stack frame: `    #3 0x55d0c1 in crabcrop::url::parse src/url.rs:12:5`.
    re.compile(r"\s*#\d+ 0x[0-9a-f]+ .*"),
    # Where a panic happened; its message is on the lines after it.
    re.compile(r"thread '[^']*'(?: \(\d+\))? panicked at [^\s'\"]+:\d+:\d+:"),
    re.compile(r"artifact_prefix='[^']*'; Test unit written to \S+"),
    # cargo-fuzz's own error, such as the exit status it reports.
    re.compile(r"Error: .*"),
]
# libFuzzer's hex dump of an input: `0x2f,0x75,...,`. The line after it
# prints the same bytes as text.
HEX_DUMP = re.compile(r"(?:0x[0-9a-f]{1,2},)+")
# The end of a panic's message: the abort libFuzzer reports, or the
# backtrace that follows it.
PANIC_MESSAGE_END = re.compile(r"==\d+==|stack backtrace:|note: run with `RUST_BACKTRACE")
RUST_FRAME = [re.compile(r"\s+\d+: [\w<&*:\[\]\(\)]+.*"), re.compile(r"\s+at \S+:\d+:\d+")]


def report_lines(log_text):
    """The lines of `log_text` that carry no input, and how many were left out."""
    kept, dropped = [], 0
    skip_next = False
    in_panic_message = False
    in_backtrace = False
    for line in log_text.splitlines():
        if skip_next:
            skip_next = False
            dropped += 1
            continue
        if in_panic_message:
            if not PANIC_MESSAGE_END.match(line):
                dropped += 1
                continue
            in_panic_message = False
        if line.startswith("stack backtrace:"):
            in_backtrace = True
            kept.append(line)
            continue
        if in_backtrace:
            if any(pattern.fullmatch(line) for pattern in RUST_FRAME):
                kept.append(line)
                continue
            in_backtrace = False
        if HEX_DUMP.fullmatch(line):
            # The hex dump, and on the next line the same bytes as text.
            skip_next = True
            dropped += 1
            continue
        if line.startswith("\t") or line.startswith("Base64:"):
            # cargo-fuzz's indented Debug form and libFuzzer's base64.
            dropped += 1
            continue
        if any(pattern.fullmatch(line) for pattern in KEEP):
            kept.append(line)
            if "panicked at" in line:
                in_panic_message = True
            continue
        dropped += 1
    return kept, dropped


def artifact_lines(directory):
    """One line per input in `directory`: its kind, SHA-256 and size."""
    path = Path(directory)
    if not path.is_dir():
        return [f"no failing input: {directory} does not exist"]
    found = sorted(entry for entry in path.iterdir() if entry.is_file())
    if not found:
        return [f"no failing input in {directory}"]
    lines = []
    for entry in found:
        data = entry.read_bytes()
        kind = re.match(r"(slow-unit|[a-z]+)-", entry.name)
        lines.append(
            f"failing input ({kind.group(1) if kind else 'unknown'}): sha256 {hashlib.sha256(data).hexdigest()}, "
            f"{len(data)} bytes; not uploaded, see fuzz/README.md (Real fuzzing)"
        )
    return lines


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: fuzz_report.py LOG ARTIFACTS_DIR", file=sys.stderr)
        return 2
    log, artifacts = args
    try:
        text = Path(log).read_bytes().decode("utf-8", errors="replace")
    except OSError as err:
        print(f"fuzz_report: cannot read the run's log: {err}", file=sys.stderr)
        return 1
    kept, dropped = report_lines(text)
    for line in kept:
        print(line)
    print(f"({dropped} lines of the run's output left out: they can carry the input)")
    for line in artifact_lines(artifacts):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
