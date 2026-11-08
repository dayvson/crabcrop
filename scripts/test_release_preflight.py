#!/usr/bin/env python3
"""Tests for scripts/release_preflight.py and its wiring in release.yml.

The checks run against canned API responses, the documented shapes and
hostile ones; `gh api` itself is replaced by a stub on PATH. Run
directly: python3 scripts/test_release_preflight.py
"""
import contextlib
import importlib.util
import io
import os
import re
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("release_preflight", HERE / "release_preflight.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)

SHA = "3334fa3407917686510a722c93901adcac6763f4"
OTHER_SHA = "f09710bc4a18601c1bbd58b38ccc977897cc1297"

# The documented response of GET /repos/{owner}/{repo}/environments/{name},
# trimmed to the fields the check reads.
GUARDED = {
    "name": "release",
    "protection_rules": [
        {"id": 3736, "type": "wait_timer", "wait_timer": 30},
        {
            "id": 3755,
            "type": "required_reviewers",
            "prevent_self_review": False,
            "reviewers": [{"type": "User", "reviewer": {"login": "dayvson", "id": 59447}}],
        },
        {"id": 3756, "type": "branch_policy"},
    ],
    "deployment_branch_policy": {"protected_branches": False, "custom_branch_policies": True},
}


def run(sha=SHA, event="push", branch="main", status="completed", conclusion="success", created="2026-11-07T23:00:40Z", run_id=1):
    return {
        "id": run_id,
        "head_sha": sha,
        "event": event,
        "head_branch": branch,
        "status": status,
        "conclusion": conclusion,
        "created_at": created,
        "path": ".github/workflows/rust.yml",
    }


def runs(*entries):
    return {"total_count": len(entries), "workflow_runs": list(entries)}


def ci(response):
    return preflight.ci_problems(response, SHA, "rust.yml", "main")


class EnvironmentCheck(unittest.TestCase):
    def test_required_reviewers_rule_passes(self):
        self.assertEqual(preflight.environment_problems(GUARDED, "release"), [])

    def test_rule_without_a_reviewers_field_counts(self):
        env = {"protection_rules": [{"type": "required_reviewers"}]}
        self.assertEqual(preflight.environment_problems(env, "release"), [])

    def test_unprotected_environments_fail(self):
        for env in (
            {"name": "release"},
            {"name": "release", "protection_rules": []},
            {"protection_rules": [{"type": "wait_timer", "wait_timer": 30}, {"type": "branch_policy"}]},
            {"protection_rules": [{"type": "required_reviewers", "reviewers": []}]},
        ):
            with self.subTest(env=env):
                problems = preflight.environment_problems(env, "release")
                self.assertEqual(len(problems), 1)
                self.assertIn("release", problems[0])

    def test_missing_rule_names_the_setting(self):
        (problem,) = preflight.environment_problems({"protection_rules": []}, "release")
        self.assertIn("Required reviewers", problem)

    def test_hostile_shapes_fail_without_raising(self):
        for env in (
            None,
            "release",
            42,
            [GUARDED],
            {"protection_rules": "required_reviewers"},
            {"protection_rules": {"type": "required_reviewers"}},
            {"protection_rules": [None, "required_reviewers", ["required_reviewers"], 7]},
            {"protection_rules": [{"type": ["required_reviewers"]}]},
            {"protection_rules": [{"type": "Required_Reviewers", "reviewers": [{}]}]},
            {"protection_rules": [{"type": "required_reviewers", "reviewers": "dayvson"}]},
            {"protection_rules": [{"type": "required_reviewers", "reviewers": {"login": "dayvson"}}]},
            {"protection_rules": [{"type": "required_reviewers", "reviewers": None}]},
        ):
            with self.subTest(env=env):
                self.assertTrue(preflight.environment_problems(env, "release"))


class CiCheck(unittest.TestCase):
    def test_green_push_run_on_main_passes(self):
        self.assertEqual(ci(runs(run())), [])

    def test_no_run_fails(self):
        (problem,) = ci(runs())
        self.assertIn("no rust.yml push run on main", problem)

    def test_runs_that_do_not_test_this_commit_on_main_fail(self):
        for entry in (
            # A pull request run tests the merge with main, not the commit.
            run(event="pull_request"),
            run(branch="v0.9.11"),
            run(branch="claude/feature"),
            # In case the API ignored the head_sha filter.
            run(sha=OTHER_SHA),
            run(sha=SHA.upper()),
        ):
            with self.subTest(entry=entry):
                self.assertTrue(ci(runs(entry)))

    def test_unfinished_run_fails(self):
        for status in ("queued", "in_progress", "waiting", "requested", None):
            with self.subTest(status=status):
                (problem,) = ci(runs(run(status=status, conclusion=None)))
                self.assertIn("not finished", problem)

    def test_failed_run_fails(self):
        for conclusion in ("failure", "cancelled", "timed_out", "skipped", "neutral", "action_required", None, True):
            with self.subTest(conclusion=conclusion):
                (problem,) = ci(runs(run(conclusion=conclusion)))
                self.assertIn("not success", problem)

    def test_latest_run_decides(self):
        earlier, later = "2026-11-07T23:00:40Z", "2026-11-07T23:40:00Z"
        self.assertTrue(ci(runs(run(created=earlier, run_id=1), run(created=later, run_id=2, conclusion="failure"))))
        self.assertEqual(ci(runs(run(created=later, run_id=2), run(created=earlier, run_id=1, conclusion="failure"))), [])
        # A re-run still in progress is the latest state of that run.
        self.assertTrue(ci(runs(run(created=later, run_id=2, status="in_progress", conclusion=None), run(created=earlier, run_id=1))))

    def test_hostile_shapes_fail_without_raising(self):
        for response in (
            None,
            [],
            "workflow_runs",
            {},
            {"workflow_runs": None},
            {"workflow_runs": {"0": run()}},
            {"workflow_runs": [None, 1, "run", [run()]]},
            {"workflow_runs": [{**run(), "head_sha": None}]},
        ):
            with self.subTest(response=response):
                self.assertTrue(ci(response))

    def test_odd_ordering_fields_do_not_raise(self):
        entries = [
            {**run(), "created_at": None, "id": True},
            {**run(), "created_at": 17, "id": "2"},
            {**run(conclusion="failure"), "created_at": ["x"], "id": None},
        ]
        # All three sort as equals; whichever max() keeps, the check returns a verdict.
        self.assertIsInstance(ci(runs(*entries)), list)

    def test_api_values_cannot_inject_workflow_commands(self):
        hostile = "failure\n::set-output name=x::y\r\n::error::forged"
        for entry in (run(conclusion=hostile), run(status=hostile, conclusion=None)):
            with self.subTest(entry=entry):
                (problem,) = ci(runs(entry))
                self.assertNotIn("\n", problem)
                self.assertNotIn("\r", problem)
                self.assertNotIn("::", problem)

    def test_long_api_values_are_cut(self):
        (problem,) = ci(runs(run(conclusion="x" * 10_000)))
        self.assertLess(len(problem), 400)


class Check(unittest.TestCase):
    def fetcher(self, responses):
        self.paths = []

        def fetch(path):
            self.paths.append(path)
            for prefix, response in responses.items():
                if path.startswith(prefix):
                    if isinstance(response, Exception):
                        raise response
                    return response
            raise AssertionError(f"unexpected path {path}")

        return fetch

    def test_requests_the_environment_and_the_commits_push_runs(self):
        fetch = self.fetcher({"repos/o/r/environments/": GUARDED, "repos/o/r/actions/": runs(run())})
        self.assertEqual(preflight.check("o/r", SHA, "release", "rust.yml", "main", fetch), [])
        self.assertEqual(
            self.paths,
            [
                "repos/o/r/environments/release",
                f"repos/o/r/actions/workflows/rust.yml/runs?head_sha={SHA}&event=push&branch=main&per_page=100",
            ],
        )

    def test_reports_both_problems(self):
        fetch = self.fetcher({"repos/o/r/environments/": {"protection_rules": []}, "repos/o/r/actions/": runs()})
        self.assertEqual(len(preflight.check("o/r", SHA, "release", "rust.yml", "main", fetch)), 2)

    def test_unreadable_environment_fails_closed(self):
        fetch = self.fetcher(
            {
                "repos/o/r/environments/": preflight.ApiError("gh api failed: HTTP 404"),
                "repos/o/r/actions/": runs(run()),
            }
        )
        (problem,) = preflight.check("o/r", SHA, "release", "rust.yml", "main", fetch)
        self.assertIn("could not read the `release` environment", problem)
        self.assertIn("actions: read", problem)

    def test_unreadable_runs_fail_closed(self):
        fetch = self.fetcher(
            {"repos/o/r/environments/": GUARDED, "repos/o/r/actions/": preflight.ApiError("gh api failed: HTTP 500")}
        )
        (problem,) = preflight.check("o/r", SHA, "release", "rust.yml", "main", fetch)
        self.assertIn("could not read the rust.yml runs", problem)


class Main(unittest.TestCase):
    def call(self, argv, fetch=None, actions=False):
        def refuse(path):
            raise AssertionError(f"fetched {path} for invalid arguments")

        out, err = io.StringIO(), io.StringIO()
        env = {"GITHUB_ACTIONS": "true"} if actions else {"GITHUB_ACTIONS": ""}
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = preflight.main(argv, fetch or refuse)
            except SystemExit as exit_:
                code = exit_.code
        return code, out.getvalue(), err.getvalue()

    def test_passing_release(self):
        fetch = lambda path: GUARDED if "/environments/" in path else runs(run())
        code, out, _ = self.call(["--repo", "dayvson/crabcrop", "--sha", SHA], fetch)
        self.assertEqual(code, 0)
        self.assertIn("requires a reviewer", out)

    def test_failing_release_annotates_each_problem_on_one_line(self):
        hostile = runs(run(conclusion="failure\n::warning::forged"))
        fetch = lambda path: {"protection_rules": []} if "/environments/" in path else hostile
        code, out, _ = self.call(["--repo", "dayvson/crabcrop", "--sha", SHA], fetch, actions=True)
        self.assertEqual(code, 1)
        lines = out.splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertTrue(line.startswith("::error title=Release preflight::"), line)
            self.assertEqual(line.count("::"), 2, line)

    def test_invalid_arguments_are_refused_before_any_request(self):
        for argv in (
            ["--repo", "dayvson/crabcrop", "--sha", "HEAD"],
            ["--repo", "dayvson/crabcrop", "--sha", SHA.upper()],
            ["--repo", "dayvson/crabcrop", "--sha", SHA[:-1]],
            ["--repo", "dayvson/crabcrop", "--sha", SHA + "0"],
            ["--repo", "dayvson/crabcrop", "--sha", SHA[:-2] + "&x"],
            ["--repo", "dayvson/crabcrop", "--sha", SHA + "\n"],
            ["--repo", "dayvson", "--sha", SHA],
            ["--repo", "dayvson/crabcrop/../x", "--sha", SHA],
            ["--repo", "dayvson/crab crop", "--sha", SHA],
            ["--repo", "dayvson/..", "--sha", SHA],
            ["--repo", "../crabcrop", "--sha", SHA],
            ["--repo", "dayvson/crabcrop", "--sha", SHA, "--workflow", ".."],
            ["--repo", "dayvson/crabcrop", "--sha", SHA, "--environment", "."],
            ["--repo", "dayvson/crabcrop", "--sha", SHA, "--branch", "-main"],
            ["--repo", "dayvson/crabcrop", "--sha", SHA, "--environment", "release?x=1"],
            ["--repo", "dayvson/crabcrop", "--sha", SHA, "--workflow", "../rust.yml"],
            ["--repo", "dayvson/crabcrop", "--sha", SHA, "--branch", "main&event=pull_request"],
            ["--repo", "dayvson/crabcrop", "--sha", SHA, "--branch", ""],
        ):
            with self.subTest(argv=argv):
                code, out, err = self.call(argv)
                self.assertEqual(code, 2)
                self.assertNotIn("\n::", "\n" + out + err)

    def test_dot_led_repository_names_are_accepted(self):
        fetch = lambda path: GUARDED if "/environments/" in path else runs(run())
        code, _, _ = self.call(["--repo", "dayvson/.github", "--sha", SHA], fetch)
        self.assertEqual(code, 0)

    def test_missing_arguments_are_a_usage_error(self):
        code, _, _ = self.call(["--repo", "dayvson/crabcrop"])
        self.assertEqual(code, 2)


class GhApi(unittest.TestCase):
    """`gh api` replaced by a stub that prints what the case needs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bin = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def stub(self, stdout, code=0, stderr=""):
        # PATH holds only the stub's directory, so the stub names its
        # interpreter by absolute path and needs no other command.
        payload = self.bin / "payload"
        payload.write_text(stdout)
        gh = self.bin / "gh"
        gh.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            f"sys.stdout.write(open({str(payload)!r}).read())\n"
            f"sys.stderr.write({stderr!r})\n"
            f"sys.exit({code})\n"
        )
        gh.chmod(gh.stat().st_mode | stat.S_IXUSR)

    def api(self, path="repos/o/r/environments/release"):
        with mock.patch.dict(os.environ, {"PATH": str(self.bin)}):
            return preflight.gh_api(path)

    def test_parses_json(self):
        self.stub('{"protection_rules": []}')
        self.assertEqual(self.api(), {"protection_rules": []})

    def test_failures_raise_api_error(self):
        cases = [
            ("not json", 0, ""),
            ("", 0, ""),
            ("[" * 100_000, 0, ""),
            ('{"message": "Not Found"}', 1, "gh: Not Found (HTTP 404)"),
        ]
        for stdout, code, stderr in cases:
            with self.subTest(stdout=stdout[:20], code=code):
                self.stub(stdout, code, stderr)
                with self.assertRaises(preflight.ApiError) as raised:
                    self.api()
                self.assertNotIn("\n", str(raised.exception))
                self.assertNotIn("::", str(raised.exception))

    def test_missing_gh_raises_api_error(self):
        with self.assertRaises(preflight.ApiError):
            self.api()


class Workflow(unittest.TestCase):
    """release.yml runs the preflight first and publishes through the script."""

    @classmethod
    def setUpClass(cls):
        text = (ROOT / ".github/workflows/release.yml").read_text()
        cls.jobs = {}
        name = None
        for line in text.split("\n")[text.split("\n").index("jobs:") + 1 :]:
            header = re.fullmatch(r"  ([A-Za-z0-9_-]+):\s*", line)
            if header:
                name = header.group(1)
                cls.jobs[name] = []
            elif name:
                cls.jobs[name].append(line)

    def job(self, name):
        if name not in self.jobs:
            self.fail(f"release.yml has no {name} job (jobs: {', '.join(self.jobs)})")
        return "\n".join(line for line in self.jobs[name] if not line.lstrip().startswith("#"))

    def test_preflight_job_runs_the_check_with_actions_read(self):
        job = self.job("preflight")
        self.assertIn("python3 scripts/release_preflight.py", job)
        self.assertRegex(job, r'"?actions"?: "?read"?')

    def test_plan_waits_for_preflight(self):
        plan = self.job("plan")
        self.assertTrue(re.search(r"needs:\s*\n\s*- preflight\n", plan), "plan does not need preflight")
        self.assertTrue("needs.preflight.result == 'success'" in plan, "plan does not wait for a passed preflight")

    def test_host_creates_or_updates_through_the_script(self):
        host = self.job("host")
        self.assertTrue("bash scripts/publish_github_release.sh" in host, "host does not run the script")
        for name in self.jobs:
            with self.subTest(job=name):
                self.assertFalse("gh release create" in self.job(name), f"{name} runs gh release create")


if __name__ == "__main__":
    unittest.main()
