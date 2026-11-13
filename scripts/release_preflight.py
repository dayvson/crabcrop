#!/usr/bin/env python3
"""Release preflight: stop a tag run before it builds unless the approval gate and CI hold.

The release workflow's first job runs this (issue #457). It reads these
things from the GitHub REST API through `gh api` and fails the job,
which stops every later job, when one is wrong:

1. The `release` environment must carry a required-reviewers protection
   rule. The release approval and the registry jobs wait on that
   environment (issue #599), so the rule is what makes a human approve
   every publish; without it they would run unattended, and nothing else
   would notice the setting had gone.
2. The tagged commit must have a successful push run of rust.yml on main:
   the hard rule in AGENTS.md, never release a head whose main CI is not
   green. The latest such run counts, and one still running is not green.
3. Releases must be immutable (issue #599), so a published release's
   assets cannot be replaced. Reading the setting takes the Administration
   permission, which the workflow's token does not have: when it cannot
   be read, the preflight warns and goes on, and RELEASING.md lists the
   setting among the ones the owner keeps.

Both answers come from an API, so the parsing assumes nothing about their
shape: a response that is not what the check needs fails the check; it
never raises and never passes.

Usage: release_preflight.py --repo OWNER/NAME --sha COMMIT
       [--environment release] [--workflow rust.yml] [--branch main]
"""
import argparse
import json
import os
import re
import subprocess
import sys

# Each value lands in an API path, so none may be `.` or `..` or carry
# a separator or a query character.
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*/(?!\.\.?$)[A-Za-z0-9_.-]+")
SHA = re.compile(r"[0-9a-f]{40}")


def shown(value, limit=60):
    """An API-supplied value made safe to print.

    The job's log turns a line that starts with `::` into a workflow
    command, so a value read from the API never reaches the output as is:
    anything outside a conservative character set becomes `?`, and long
    values are cut.
    """
    text = value if isinstance(value, str) else repr(value)
    text = "".join(c if c.isascii() and (c.isalnum() or c in " _.,-/()") else "?" for c in text)
    return text[:limit] + ("..." if len(text) > limit else "")


def environment_problems(environment, name):
    """Why the environment does not enforce an approval; empty when it does."""
    if not isinstance(environment, dict):
        return [f"the `{name}` environment response is not a JSON object"]
    rules = environment.get("protection_rules")
    if not isinstance(rules, list):
        return [
            f"the `{name}` environment has no protection rules, so its jobs run "
            "without an approval; add a Required reviewers rule"
        ]
    reviewers_rules = [
        rule for rule in rules if isinstance(rule, dict) and rule.get("type") == "required_reviewers"
    ]
    if not reviewers_rules:
        return [
            f"the `{name}` environment has no Required reviewers rule, so the release "
            "would publish without an approval; add one under Settings, "
            f"Environments, {name}, Deployment protection rules"
        ]
    for rule in reviewers_rules:
        # The rule is what GitHub enforces, and it cannot be saved with
        # nobody in it, so a rule without a reviewers field still counts;
        # one that lists nobody, or lists them as something other than a
        # list, does not.
        if "reviewers" not in rule:
            return []
        reviewers = rule["reviewers"]
        if isinstance(reviewers, list) and reviewers:
            return []
    return [f"the `{name}` environment's Required reviewers rule names no reviewer"]


def ci_problems(runs, sha, workflow, branch):
    """Why the commit's CI is not green; empty when its latest run passed."""
    entries = runs.get("workflow_runs") if isinstance(runs, dict) else None
    if not isinstance(entries, list):
        return [f"the {workflow} runs response has no workflow_runs list"]
    matching = [
        run
        for run in entries
        if isinstance(run, dict)
        and run.get("head_sha") == sha
        and run.get("event") == "push"
        and run.get("head_branch") == branch
    ]
    if not matching:
        return [
            f"{sha} has no {workflow} push run on {branch}; tag only a commit "
            f"whose {branch} CI has passed"
        ]

    def order(run):
        created = run.get("created_at")
        number = run.get("id")
        return (
            created if isinstance(created, str) else "",
            number if isinstance(number, int) and not isinstance(number, bool) else -1,
        )

    latest = max(matching, key=order)
    status, conclusion = latest.get("status"), latest.get("conclusion")
    if status != "completed":
        return [
            f"the latest {workflow} run on {branch} for {sha} is not finished "
            f"(status {shown(status)}); re-run this job once it passes"
        ]
    if conclusion != "success":
        return [
            f"the latest {workflow} run on {branch} for {sha} concluded "
            f"{shown(conclusion)}, not success; fix {branch} and tag a green commit"
        ]
    return []


def immutable_problems(setting):
    """Why releases are not immutable; empty when they are."""
    if not isinstance(setting, dict) or not isinstance(setting.get("enabled"), bool):
        return ["the immutable-releases response has no true or false `enabled` field"]
    if not setting["enabled"]:
        return [
            "immutable releases are off, so a published release's assets can still be replaced; turn "
            "them on under Settings, General, Releases (RELEASING.md, Repository settings)"
        ]
    return []


def immutable_check(repo, fetch):
    """(problems, warnings) for the immutable-releases setting."""
    try:
        setting = fetch(f"repos/{repo}/immutable-releases")
    except ApiError as err:
        return [], [
            f"could not read whether releases are immutable ({err}); the job's token cannot read that "
            "setting, so check it by hand (RELEASING.md, Repository settings)"
        ]
    return immutable_problems(setting), []


class ApiError(Exception):
    pass


def gh_api(path):
    """GET a REST path with `gh api` and parse the JSON body."""
    try:
        done = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as err:
        raise ApiError(f"gh api {path} did not run: {shown(str(err), limit=200)}") from err
    if done.returncode != 0:
        detail = shown(done.stderr.strip() or done.stdout.strip(), limit=200)
        raise ApiError(f"gh api {path} failed: {detail}")
    try:
        return json.loads(done.stdout)
    except (ValueError, RecursionError) as err:
        raise ApiError(f"gh api {path} returned no JSON document") from err


def check(repo, sha, environment, workflow, branch, fetch=gh_api):
    """Every problem found; empty when the release may proceed."""
    problems = []
    try:
        found = fetch(f"repos/{repo}/environments/{environment}")
    except ApiError as err:
        problems.append(
            f"could not read the `{environment}` environment ({err}); the job needs "
            "`actions: read`, and the environment must exist"
        )
    else:
        problems += environment_problems(found, environment)
    try:
        runs = fetch(
            f"repos/{repo}/actions/workflows/{workflow}/runs"
            f"?head_sha={sha}&event=push&branch={branch}&per_page=100"
        )
    except ApiError as err:
        problems.append(f"could not read the {workflow} runs for {sha} ({err})")
    else:
        problems += ci_problems(runs, sha, workflow, branch)
    return problems


def report(problems, stream=None, level="error"):
    """Print each problem, as an annotation of `level` when running in Actions."""
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
    for problem in problems:
        if annotate:
            line = f"::{level} title=Release preflight::{problem}"
        else:
            line = f"release preflight: {problem}" if level == "error" else f"release preflight: {level}: {problem}"
        print(line, file=stream or sys.stdout)


def main(argv=None, fetch=gh_api):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="OWNER/NAME")
    parser.add_argument("--sha", required=True, help="the tagged commit, 40 hex digits")
    parser.add_argument("--environment", default="release")
    parser.add_argument("--workflow", default="rust.yml")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args(argv)

    invalid = [
        f"--{flag} {shown(value)!r} is not {what}"
        for flag, value, pattern, what in (
            ("repo", args.repo, REPO, "OWNER/NAME"),
            ("sha", args.sha, SHA, "a full lowercase commit SHA"),
            ("environment", args.environment, NAME, "an environment name"),
            ("workflow", args.workflow, NAME, "a workflow file name"),
            ("branch", args.branch, NAME, "a branch name"),
        )
        if not pattern.fullmatch(value)
    ]
    if invalid:
        report(invalid, sys.stderr)
        return 2

    problems = check(args.repo, args.sha, args.environment, args.workflow, args.branch, fetch)
    immutable, warnings = immutable_check(args.repo, fetch)
    problems += immutable
    report(warnings, level="warning")
    if problems:
        report(problems)
        return 1
    print(
        f"release preflight: the `{args.environment}` environment requires a reviewer, and "
        f"{args.sha} passed {args.workflow} on {args.branch}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
