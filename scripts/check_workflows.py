#!/usr/bin/env python3
"""Check the GitHub Actions workflows against the release hardening rules.

The release pipeline runs third-party code next to write tokens and
registry secrets. Every workflow file, and its docs/ mirror, must follow
these rules:

  pinned     Every action a step or a job uses is pinned to a full commit
             SHA. A tag can move; a local ./ reusable workflow is fine
             (issue #593).
  pipe       No run step pipes a download into a shell (`| sh`, `| bash`),
             and none runs dist's generated `matrix.install_dist` command,
             which does (issue #597).
  installer  A step that fetches cargo-dist-installer.sh checks it against
             a pinned SHA-256 before it runs it (issue #597).
  pip        Every pip install reads a requirements file with
             --require-hashes and names no package on the command line, and
             `python -m build` runs with --no-isolation, so the build
             backend is a hashed one too (issue #598).
  secrets    A step that can read a registry secret installs and builds no
             Python package (issue #598).
  dav1d      No workflow sets SYSTEM_DEPS_DAV1D_BUILD_INTERNAL, which makes
             dav1d-sys clone dav1d by ref name at build time with no hash
             check; scripts/build_dav1d.sh builds it from a pinned,
             checksummed tarball instead (issue #596).
  gate       A job that can write contents or packages, mint an OIDC token,
             or read a secret other than GITHUB_TOKEN waits for the approval
             of the `release` environment. It names the environment, or it
             needs a job that does and its `if:` cannot run it after that
             job failed or was rejected (issue #599).

The files are read with a small YAML reader for the part of YAML that
workflows use: block mappings and sequences, literal block scalars,
quoted and plain scalars, and flow sequences. It needs only the standard
library. A construct outside that part is reported, never guessed at.

Run from anywhere: python3 scripts/check_workflows.py [--root DIR] [FILE...]
With no FILE it checks .github/workflows/*.yml and docs/*-workflow.yml.
scripts/test_all.sh runs it with its tests (scripts/test_check_workflows.py).
"""
import argparse
import json
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE_ENVIRONMENT = "release"


# ---------------------------------------------------------------- YAML


class YamlError(Exception):
    pass


# A mapping key at the start of a line: quoted, or plain up to the first
# colon that a space or the end of the line follows.
KEY = re.compile(
    r"""(?:"((?:[^"\\]|\\.)*)"|'((?:[^']|'')*)'|([^\s#'"\[\]{}&*!|>%@`,-][^#]*?|-[^\s#][^#]*?))[ \t]*:(?=[ \t]|$)"""
)
BLOCK_HEADER = re.compile(r"([|>])(?:([-+])([1-9])?|([1-9])([-+])?)?[ \t]*(?:#.*)?")


class _Reader:
    """Parses the YAML subset described in the module docstring.

    Scalars stay strings, as PyYAML's BaseLoader leaves them: `on`,
    `true` and `30` are the strings "on", "true" and "30".
    """

    def __init__(self, text, name):
        self.lines = text.split("\n")
        self.name = name

    def fail(self, index, message):
        raise YamlError(f"{self.name}:{index + 1}: {message}")

    def indent(self, index):
        line = self.lines[index]
        return len(line) - len(line.lstrip(" "))

    def significant(self, index):
        """The next line at or after `index` that holds more than a comment."""
        while index < len(self.lines):
            stripped = self.lines[index].strip()
            if stripped and not stripped.startswith("#"):
                if self.lines[index].lstrip(" ").startswith("\t"):
                    self.fail(index, "a tab in the indentation")
                return index
            index += 1
        return index

    def is_item(self, index):
        content = self.lines[index][self.indent(index) :]
        return content == "-" or content.startswith(("- ", "-\t"))

    def document(self):
        start = self.significant(0)
        if start < len(self.lines) and self.lines[start].strip() in ("---", "..."):
            self.fail(start, "document markers are not supported")
        value, index = self.node(0, 0)
        index = self.significant(index)
        if index < len(self.lines):
            self.fail(index, "unexpected indentation or content")
        return value

    def node(self, index, min_indent):
        index = self.significant(index)
        if index >= len(self.lines) or self.indent(index) < min_indent:
            return "", index
        indent = self.indent(index)
        content = self.lines[index][indent:]
        if self.is_item(index):
            return self.sequence(index, indent)
        if KEY.match(content):
            return self.mapping(index, indent)
        if content.startswith(("|", ">")):
            self.fail(index, "a block scalar must follow a key or a dash")
        return self.scalar(content, index), index + 1

    def mapping(self, index, indent):
        result = {}
        while True:
            index = self.significant(index)
            if index >= len(self.lines) or self.indent(index) < indent:
                return result, index
            if self.indent(index) > indent:
                self.fail(index, "unexpected indentation (multi-line plain or quoted scalars are not supported)")
            if self.is_item(index):
                return result, index
            content = self.lines[index][indent:]
            match = KEY.match(content)
            if not match:
                self.fail(index, "expected a `key:` line")
            if match.group(1) is not None:
                key = self._double(f'"{match.group(1)}"', index)
            elif match.group(2) is not None:
                key = match.group(2).replace("''", "'")
            else:
                key = match.group(3)
            if key in result:
                self.fail(index, f"duplicate key {key!r}")
            result[key], index = self.value(content[match.end() :], index, indent)

    def value(self, rest, index, indent):
        """The value of a key at `indent` whose line continues with `rest`."""
        rest = rest.strip(" \t")
        if not rest or rest.startswith("#"):
            following = self.significant(index + 1)
            if following < len(self.lines):
                if self.indent(following) == indent and self.is_item(following):
                    return self.sequence(following, indent)
                if self.indent(following) > indent:
                    return self.node(following, indent + 1)
            return "", index + 1
        if rest.startswith(("|", ">")):
            return self.block(rest, index, indent)
        return self.scalar(rest, index), index + 1

    def sequence(self, index, indent):
        items = []
        while True:
            index = self.significant(index)
            if index >= len(self.lines) or self.indent(index) != indent or not self.is_item(index):
                if index < len(self.lines) and self.indent(index) > indent:
                    self.fail(index, "unexpected indentation")
                return items, index
            after = self.lines[index][indent + 1 :]
            stripped = after.lstrip(" ")
            if not stripped or stripped.startswith("#"):
                following = self.significant(index + 1)
                if following < len(self.lines) and self.indent(following) > indent:
                    item, index = self.node(following, indent + 1)
                else:
                    item, index = "", index + 1
            else:
                column = indent + 1 + len(after) - len(stripped)
                # Read the rest of the line as if the dash were a space:
                # the item is then a node that starts at `column`.
                self.lines[index] = " " * column + stripped
                if self.is_item(index) or KEY.match(stripped):
                    item, index = self.node(index, column)
                elif stripped.startswith(("|", ">")):
                    item, index = self.block(stripped, index, indent)
                else:
                    item, index = self.scalar(stripped, index), index + 1
            items.append(item)

    def block(self, header, index, indent):
        """A literal block scalar whose lines are indented past `indent`."""
        match = BLOCK_HEADER.fullmatch(header)
        if not match:
            self.fail(index, "malformed block scalar header")
        if match.group(1) == ">":
            self.fail(index, "folded block scalars are not supported")
        chomp = match.group(2) or match.group(5) or ""
        explicit = match.group(3) or match.group(4)
        body, width, end = [], None, index + 1
        while end < len(self.lines):
            line = self.lines[end]
            if line.strip(" ") == "":
                body.append(line)
                end += 1
                continue
            current = self.indent(end)
            if current <= indent:
                break
            if width is None:
                width = indent + int(explicit) if explicit else current
            if current < width:
                break
            body.append(line)
            end += 1
        if width is None:
            width = indent + int(explicit) if explicit else indent + 1
        text = "".join((line[width:] if len(line) > width else "") + "\n" for line in body)
        if chomp == "-":
            text = text.rstrip("\n")
        elif chomp != "+":
            text = text.rstrip("\n") + "\n" if text.strip("\n") else ""
        return text, end

    def scalar(self, text, index):
        """A one-line scalar: quoted, a flow sequence, or plain."""
        if text.startswith('"'):
            end = self._quoted_end(text, index)
            self._only_comment(text[end:], index)
            return self._double(text[:end], index)
        if text.startswith("'"):
            end = self._quoted_end(text, index)
            self._only_comment(text[end:], index)
            return text[1 : end - 1].replace("''", "'")
        if text.startswith("["):
            return self._flow(text, index)
        if text.startswith("{"):
            body, _, rest = text[1:].partition("}")
            if body.strip():
                self.fail(index, "flow mappings are not supported")
            self._only_comment(rest, index)
            return {}
        if text[0] in "&*!%@`":
            self.fail(index, f"{text[0]!r} (anchors, aliases, tags, directives) is not supported")
        cut = re.search(r"[ \t]#", text)
        return (text[: cut.start()] if cut else text).rstrip(" \t")

    def _quoted_end(self, text, index):
        """The offset just past the quoted scalar `text` starts with."""
        quote, at = text[0], 1
        while at < len(text):
            if quote == '"' and text[at] == "\\":
                at += 2
                continue
            if text[at] == quote:
                if quote == "'" and text[at + 1 : at + 2] == "'":
                    at += 2
                    continue
                return at + 1
            at += 1
        self.fail(index, "unterminated quoted scalar (multi-line quoted scalars are not supported)")

    def _double(self, quoted, index):
        try:
            return json.loads(quoted)
        except ValueError:
            self.fail(index, f"unsupported escape in {quoted}")

    def _only_comment(self, rest, index):
        rest = rest.strip(" \t")
        if rest and not rest.startswith("#"):
            self.fail(index, f"unexpected text after a scalar: {rest!r}")

    def _flow(self, text, index):
        items, current, at = [], "", 1
        while at < len(text):
            ch = text[at]
            if ch in "\"'":
                end = at + self._quoted_end(text[at:], index)
                current += text[at:end]
                at = end
                continue
            if ch in "[{":
                self.fail(index, "nested flow collections are not supported")
            if ch in ",]":
                item = current.strip(" \t")
                if item:
                    items.append(self.scalar(item, index))
                elif ch == "," or items:
                    self.fail(index, "empty item in a flow sequence")
                current = ""
                if ch == "]":
                    self._only_comment(text[at + 1 :], index)
                    return items
            else:
                current += ch
            at += 1
        self.fail(index, "unterminated flow sequence")


def load(text, name="<workflow>"):
    """The workflow `text` as dicts, lists and strings; raises YamlError."""
    return _Reader(text, name).document()


# --------------------------------------------------------------- rules

SHA_PIN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}")
DOCKER_PIN = re.compile(r"docker://[^@\s]+@sha256:[0-9a-f]{64}")
SECRET = re.compile(r"\bsecrets\.([A-Za-z_][A-Za-z0-9_]*)")
SHELLS = r"(?:ba|da|z|k)?sh|pwsh|powershell|iex|Invoke-Expression"
PIPE_INTO_SHELL = [
    # `curl ... | sh`, `| sudo bash -s --` (a single pipe, not `||`).
    re.compile(rf"(?<!\|)\|&?\s*(?:sudo\s+(?:-\S+\s+)*)?(?:env\s+)?(?:/usr)?(?:/bin/)?(?:{SHELLS})(?![\w.-])"),
    # `bash <(curl ...)`, `source <(wget ...)`.
    re.compile(rf"(?:(?:{SHELLS})|source|\.)\s+<\(\s*(?:curl|wget)\b"),
    # `sh -c "$(curl ...)"`.
    re.compile(rf"(?:{SHELLS})\s+-c\s+[\"']?\$\(\s*(?:curl|wget)\b"),
]
PIP_INSTALL = re.compile(r"\bpip3?(?:\.exe)?[\"']?\s+install\b")
BUILD_RUN = re.compile(r"\bpython(?:3(?:\.\d+)?)?\s+-m\s+build\b")
PYTHON_PACKAGE_WORK = re.compile(r"\bpip3?(?:\.exe)?[\"']?\s+(?:install|wheel|download)\b|\bpython(?:3(?:\.\d+)?)?\s+-m\s+build\b|\bsetup\.py\b")
CHECKSUM = re.compile(r"\bsha256sum\b[^\n]*\s(?:-c|--check)\b|\bshasum\b[^\n]*-a\s*256\b[^\n]*\s(?:-c|--check)\b")
RUNS_A_SCRIPT = re.compile(r"(?:^|[;&|(]\s*|\b(?:then|do|else)\s+)(?:sudo\s+)?(?:(?:ba|da|z)?sh\s+(?!-c\b)\S|\./\S)")
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
STATUS_FUNCTIONS = re.compile(r"\b(?:always|failure|cancelled)\s*\(")

# pip options a hashed install may use. Anything else (an index, a find
# link, an editable path, a target) changes what gets installed or where.
PIP_FLAGS = {
    "--require-hashes",
    "--user",
    "--break-system-packages",
    "-q",
    "-qq",
    "--quiet",
    "--no-deps",
    "--no-input",
    "--disable-pip-version-check",
    "--no-cache-dir",
    "--no-warn-script-location",
}
PIP_FLAGS_WITH_VALUE = {"--only-binary", "--progress-bar"}
PIP_REQUIREMENT = {"-r", "--requirement"}


def shell_lines(script):
    """The script's lines, continuations joined and comment lines dropped."""
    joined = re.sub(r"\\\n", " ", script)
    return [line for line in joined.split("\n") if not line.lstrip().startswith("#")]


def pip_problems(command):
    """Why one `pip install ...` command is not a hashed install."""
    tail = PIP_INSTALL.split(command, maxsplit=1)[1]
    tail = re.split(r"\s(?:&&|\|\||[;|&])\s|[;)]|\s#", tail)[0]
    try:
        tokens = shlex.split(tail)
    except ValueError:
        tokens = tail.split()
    problems, files, at = [], [], 0
    while at < len(tokens):
        token = tokens[at]
        if token in PIP_REQUIREMENT:
            if at + 1 < len(tokens):
                files.append(tokens[at + 1])
            at += 2
            continue
        if token.startswith("--requirement="):
            files.append(token.split("=", 1)[1])
        elif token.startswith("-r") and len(token) > 2:
            files.append(token[2:])
        elif token in PIP_FLAGS_WITH_VALUE:
            at += 1
        elif token in PIP_FLAGS or token.split("=", 1)[0] in PIP_FLAGS_WITH_VALUE:
            pass
        elif token.startswith("-"):
            problems.append(f"uses the pip option {token}, which a hashed install does not need")
        else:
            problems.append(f"names {token!r} on the command line")
        at += 1
    if "--require-hashes" not in tokens:
        problems.append("has no --require-hashes")
    if not files:
        problems.append("reads no requirements file (-r FILE)")
    return problems


def run_problems(script):
    """The pipe, installer and pip rules for one run script."""
    problems = []
    lines = shell_lines(script)
    for line in lines:
        if "install_dist" in line:
            problems.append(
                "runs dist's matrix.install_dist command, which pipes the cargo-dist installer into sh; "
                "download it, check its pinned sha256, then run it"
            )
        for pattern in PIPE_INTO_SHELL:
            if pattern.search(line):
                problems.append(f"pipes a download into a shell: {line.strip()}")
                break
        for match in PIP_INSTALL.finditer(line):
            for problem in pip_problems(line[match.start() :]):
                problems.append(f"`pip install` {problem}: {line.strip()}")
        for match in BUILD_RUN.finditer(line):
            rest = line[match.end() :]
            if not re.search(r"(?:^|\s)(?:--no-isolation|-n)(?=\s|$)", rest):
                problems.append(
                    "runs `python -m build` without --no-isolation, which installs an unhashed build backend"
                )
    return problems


def installer_problems(script, env):
    """A step that fetches cargo-dist-installer.sh checks a pinned hash first."""
    lines = shell_lines(script)
    if not any("cargo-dist-installer" in line for line in lines):
        return []
    check = next((at for at, line in enumerate(lines) if CHECKSUM.search(line)), None)
    runs = next((at for at, line in enumerate(lines) if RUNS_A_SCRIPT.search(line.strip())), None)
    if check is None:
        return ["fetches cargo-dist-installer.sh but never checks its sha256 (sha256sum -c or shasum -a 256 -c)"]
    if runs is not None and runs < check:
        return ["runs cargo-dist-installer.sh before it checks its sha256"]
    line = lines[check]
    if HEX64.search(line):
        return []
    for name in re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)", line):
        if HEX64.fullmatch(str(env.get(name, ""))):
            return []
    return ["checks cargo-dist-installer.sh against no pinned sha256 (a literal, or an env value of 64 hex digits)"]


def strings(value):
    """Every string inside a parsed value, keys included."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def secret_names(value):
    """The secrets other than GITHUB_TOKEN that a parsed value reads."""
    return sorted({name for text in strings(value) for name in SECRET.findall(text)} - {"GITHUB_TOKEN"})


def as_list(value):
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, str) and value else []


def mapping(value):
    return value if isinstance(value, dict) else {}


def environment_name(job):
    value = job.get("environment")
    if isinstance(value, dict):
        value = value.get("name")
    return value if isinstance(value, str) else None


def write_scopes(job, workflow):
    """What the job's token may write that the gate rule covers."""
    permissions = job["permissions"] if "permissions" in job else workflow.get("permissions")
    if permissions is None:
        return ["everything the repository's default token allows (no permissions block)"]
    if isinstance(permissions, str):
        return ["everything (write-all)"] if permissions.strip() == "write-all" else []
    return [
        f"{scope}: write"
        for scope in ("contents", "packages", "id-token")
        if isinstance(permissions, dict) and str(permissions.get(scope, "")).strip() == "write"
    ]


def conjuncts(condition):
    """The top-level `&&` terms of an `if:` expression."""
    text = condition.strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2]
    terms, depth, quote, current = [], 0, None, ""
    at = 0
    while at < len(text):
        ch = text[at]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and text.startswith("&&", at):
            terms.append(current)
            current = ""
            at += 2
            continue
        current += ch
        at += 1
    terms.append(current)
    return [re.sub(r"\s+", " ", term.strip().replace('"', "'")) for term in terms]


def waits_for(job, need):
    """Whether `job` runs only after `need` succeeded."""
    condition = job.get("if")
    if not isinstance(condition, str) or not STATUS_FUNCTIONS.search(condition):
        return True
    wanted = {f"needs.{need}.result == 'success'", f"'success' == needs.{need}.result"}
    return any(term in wanted for term in conjuncts(condition))


def gate_problems(workflow, name):
    jobs = mapping(workflow.get("jobs"))
    gated = {}

    def is_gated(job_name, seen=()):
        if job_name in gated:
            return gated[job_name]
        job = mapping(jobs.get(job_name))
        result = environment_name(job) == RELEASE_ENVIRONMENT
        if not result:
            for need in as_list(job.get("needs")):
                if need in jobs and need not in seen and waits_for(job, need) and is_gated(need, seen + (job_name,)):
                    result = True
                    break
        gated[job_name] = result
        return result

    problems = []
    for job_name, job in jobs.items():
        job = mapping(job)
        reasons = write_scopes(job, workflow)
        secrets = secret_names(job)
        if job.get("secrets") == "inherit":
            secrets.append("every secret (secrets: inherit)")
        reasons += [f"reads {secret}" for secret in secrets]
        if reasons and not is_gated(job_name):
            problems.append(
                f"{name}: job {job_name} can write or read secrets ({'; '.join(reasons)}) but does not wait for "
                f"the `{RELEASE_ENVIRONMENT}` environment's approval: name the environment, or need a job that "
                "does with an `if:` that runs only after it succeeded"
            )
    return problems


def check_workflow(text, name):
    """Every problem in one workflow file."""
    try:
        workflow = load(text, name)
    except YamlError as err:
        return [f"{err} (scripts/check_workflows.py reads a subset of YAML; keep the workflow inside it)"]
    if not isinstance(workflow, dict):
        return [f"{name}: not a YAML mapping"]
    problems = []
    workflow_env = mapping(workflow.get("env"))
    for key in workflow_env:
        if key == "SYSTEM_DEPS_DAV1D_BUILD_INTERNAL":
            problems.append(f"{name}: the workflow env sets {key}; build dav1d with scripts/build_dav1d.sh")
    for job_name, job in mapping(workflow.get("jobs")).items():
        job = mapping(job)
        where = f"{name}: job {job_name}"
        uses = job.get("uses")
        if isinstance(uses, str) and not (uses.startswith("./") or SHA_PIN.fullmatch(uses)):
            problems.append(f"{where}: uses {uses}, which is not pinned to a full commit SHA")
        job_env = {**workflow_env, **mapping(job.get("env"))}
        if "SYSTEM_DEPS_DAV1D_BUILD_INTERNAL" in mapping(job.get("env")):
            problems.append(f"{where}: the job env sets SYSTEM_DEPS_DAV1D_BUILD_INTERNAL; build dav1d with scripts/build_dav1d.sh")
        steps = job.get("steps")
        for number, step in enumerate(steps if isinstance(steps, list) else [], 1):
            step = mapping(step)
            label = step.get("name") or step.get("uses") or (str(step.get("run", "")).strip().split("\n")[0][:50])
            at = f"{where}, step {number} ({label})"
            uses = step.get("uses")
            if isinstance(uses, str) and not (
                uses.startswith("./") or SHA_PIN.fullmatch(uses) or DOCKER_PIN.fullmatch(uses)
            ):
                problems.append(f"{at}: uses {uses}, which is not pinned to a full commit SHA")
            step_env = mapping(step.get("env"))
            env = {**job_env, **step_env}
            if "SYSTEM_DEPS_DAV1D_BUILD_INTERNAL" in step_env:
                problems.append(f"{at}: sets SYSTEM_DEPS_DAV1D_BUILD_INTERNAL; build dav1d with scripts/build_dav1d.sh")
            script = step.get("run")
            if not isinstance(script, str):
                continue
            if any("SYSTEM_DEPS_DAV1D_BUILD_INTERNAL" in line for line in shell_lines(script)):
                problems.append(f"{at}: sets SYSTEM_DEPS_DAV1D_BUILD_INTERNAL; build dav1d with scripts/build_dav1d.sh")
            problems += [f"{at}: {problem}" for problem in run_problems(script)]
            problems += [f"{at}: {problem}" for problem in installer_problems(script, env)]
            secrets = secret_names([env, script])
            if secrets and PYTHON_PACKAGE_WORK.search("\n".join(shell_lines(script))):
                problems.append(
                    f"{at}: installs or builds Python packages while it can read {', '.join(secrets)}; "
                    "do that in a step or job without the secret"
                )
    problems += gate_problems(workflow, name)
    return problems


def default_files(root):
    return sorted((root / ".github" / "workflows").glob("*.yml")) + sorted((root / "docs").glob("*-workflow.yml"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    files = args.files or default_files(args.root)
    if not files:
        print("check_workflows: no workflow files found", file=sys.stderr)
        return 1
    problems = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as err:
            problems.append(f"{path}: cannot read it ({err})")
            continue
        try:
            shown = path.resolve().relative_to(args.root.resolve())
        except ValueError:
            shown = path
        problems += check_workflow(text, str(shown))
    if problems:
        print("workflow rules broken (see scripts/check_workflows.py):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"{len(files)} workflow files: actions pinned, no download piped into a shell, hashed pip installs, "
          "dav1d from a checked tarball, and every publishing job behind the release approval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
