#!/usr/bin/env python3
"""Tests for the tier choice in .github/workflows/functional.yml.

A pull request gets the full functional tier when its diff touches a path
the "Decide the tier" step lists, and the smoke tier otherwise. Each case
runs that step's own shell script, read from the workflow, with git stubbed
to print the case's diff, and reads the tier it writes to GITHUB_OUTPUT.
Run directly: python3 scripts/test_functional_tier.py
"""
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent / ".github/workflows/functional.yml"

def tier_step_script(text):
    """The `run: |` block of the step with `id: tier`, dedented."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "id: tier")
    run = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    key_indent = len(lines[run]) - len(lines[run].lstrip())
    block = []
    for line in lines[run + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= key_indent:
            break
        block.append(line)
    indent = min(len(line) - len(line.lstrip()) for line in block if line.strip())
    return "\n".join(line[indent:] for line in block) + "\n"


SCRIPT = tier_step_script(WORKFLOW.read_text(encoding="utf-8"))


def choose_tier(files, event="pull_request", input_tier=""):
    # The script as the runner gets it: GitHub substitutes the expressions
    # before the shell starts, and runs it with bash -eo pipefail.
    script = SCRIPT
    for expression, value in (
        ("${{ github.event_name }}", event),
        ("${{ github.base_ref }}", "main"),
        ("${{ inputs.tier }}", input_tier),
    ):
        script = script.replace(expression, value)
    unknown = re.findall(r"\$\{\{[^}]*\}\}", script)
    assert not unknown, f"teach this test the expressions {unknown}"
    # The only git command the step runs is the diff against the base.
    stub = (
        'git() {\n'
        '  [ "$*" = "diff --name-only origin/main...HEAD" ] || { echo "unexpected: git $*" >&2; return 2; }\n'
        '  printf "%s" "$DIFF"\n'
        '}\n'
    )
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "output"
        output.touch()
        done = subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", stub + script],
            env={**os.environ, "DIFF": "".join(f + "\n" for f in files), "GITHUB_OUTPUT": str(output)},
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        found = re.findall(r"^tier=(\w+)$", output.read_text(), re.M)
    assert len(found) == 1, found
    return found[0]


class FunctionalTier(unittest.TestCase):
    def assertTier(self, files, expected):
        self.assertEqual(choose_tier(files), expected, files)

    def test_a_server_only_change_runs_the_full_tier(self):
        self.assertTier(["src/server.rs"], "full")

    def test_every_module_under_src_runs_the_full_tier(self):
        # The modules issue #458 found outside the old list, the ones that
        # were on it, and a module that does not exist yet.
        for path in (
            "src/server/process.rs", "src/server/meta.rs", "src/server/tests.rs",
            "src/cache.rs", "src/storage.rs", "src/s3store.rs", "src/upload.rs",
            "src/main.rs", "src/lib.rs", "src/toolrun.rs",
            "src/engine.rs", "src/engine/encode.rs", "src/url.rs", "src/loader.rs",
            "src/a_module_added_later.rs", "src/new_dir/deeper/mod.rs",
        ):
            self.assertTier([path], "full")

    def test_dependencies_and_the_harness_run_the_full_tier(self):
        for path in (
            "Cargo.lock", "Cargo.toml", "crates/crabcrop-url/Cargo.toml",
            "tests/functional/run.py", "tests/functional/cases_core.json",
        ):
            self.assertTier([path], "full")

    def test_one_matching_path_anywhere_in_the_diff_is_enough(self):
        self.assertTier(["README.md", "docs/url-grammar.md", "src/upload.rs", "sdks/python/x.py"], "full")

    def test_changes_outside_the_listed_paths_run_smoke(self):
        for files in (
            ["README.md"],
            ["docs/configuration.md", "docs/rust-workflow.yml"],
            [".github/workflows/rust.yml"],
            ["sdks/typescript/src/index.ts"],
            ["crates/crabcrop-url/src/lib.rs"],
            ["tests/library.rs"],
            ["srcx/main.rs", "a/src/main.rs", "src"],
            ["docs/a file with spaces.md"],
            [],
        ):
            self.assertTier(files, "smoke")

    def test_events_other_than_pull_requests_ignore_the_diff(self):
        self.assertEqual(choose_tier(["README.md"], event="schedule"), "full")
        self.assertEqual(choose_tier(["README.md"], event="push"), "full")
        self.assertEqual(choose_tier([], event="workflow_dispatch", input_tier="smoke"), "smoke")
        self.assertEqual(choose_tier(["src/server.rs"], event="workflow_dispatch", input_tier="smoke"), "smoke")


if __name__ == "__main__":
    unittest.main()
