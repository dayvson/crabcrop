#!/usr/bin/env python3
"""Tests for scripts/check_workflows.py.

The YAML reader is compared with PyYAML's BaseLoader on every workflow in
the repository when PyYAML is installed, and on small cases always. Each
rule gets the workflow lines that broke it before its issue was fixed,
which must fail, and the lines that replaced them, which must pass. The
last cases run the check on this repository's docs/ mirrors, which hold
the workflows as they are to be landed. Run directly:
python3 scripts/test_check_workflows.py
"""
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPT = HERE / "check_workflows.py"
sys.path.insert(0, str(HERE))

import check_workflows as cw  # noqa: E402

try:
    import yaml
except ImportError:  # the check itself needs only the standard library
    yaml = None

SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"
HASH = "a7b3b0812d624a09a534ba272aa8e04c801e56a83d553327a4a8632bc03abb94"


def problems(text):
    return cw.check_workflow(textwrap.dedent(text), "test.yml")


def step_workflow(step, env=""):
    """A workflow whose one job runs `step` (YAML for one step, unindented)."""
    body = textwrap.indent(textwrap.dedent(step).strip("\n"), " " * 6)
    return f"""\
name: T
on: push
permissions:
  contents: read
{env}jobs:
  build:
    runs-on: ubuntu-22.04
    steps:
{body}
"""


def step_problems(step, env=""):
    return cw.check_workflow(step_workflow(step, env), "test.yml")


def workflow_files():
    return sorted((ROOT / ".github" / "workflows").glob("*.yml")) + sorted((ROOT / "docs").glob("*-workflow.yml"))


class Reader(unittest.TestCase):
    @unittest.skipIf(yaml is None, "PyYAML is not installed")
    def test_reads_every_workflow_as_pyyaml_does(self):
        files = workflow_files()
        self.assertTrue(files)
        for path in files:
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertEqual(cw.load(text, path.name), yaml.load(text, Loader=yaml.BaseLoader))

    def test_scalars_stay_strings(self):
        doc = cw.load('on: push\nn: 30\nb: true\nq: "x: y"\ns: \'it\'\'s\'\nempty:\nf: [ "a", b ]\nm: {}\n')
        self.assertEqual(doc, {"on": "push", "n": "30", "b": "true", "q": "x: y", "s": "it's", "empty": "", "f": ["a", "b"], "m": {}})

    def test_comments_are_dropped_outside_scalars(self):
        doc = cw.load('a: x # note\nb: "y # kept"\n# a line\nc: z#kept\n')
        self.assertEqual(doc, {"a": "x", "b": "y # kept", "c": "z#kept"})

    def test_sequences_in_both_indentations(self):
        doc = cw.load("a:\n- x\n- y: 1\n  z: 2\nb:\n  - - p\n    - q\n  -\n    k: v\n")
        self.assertEqual(doc, {"a": ["x", {"y": "1", "z": "2"}], "b": [["p", "q"], {"k": "v"}]})

    def test_literal_block_scalars_and_chomping(self):
        text = "a: |\n  one\n\n  # not a comment\n    two\n\nb: |-\n  x\nc: |+\n  y\n\nd: x\n"
        self.assertEqual(
            cw.load(text),
            {"a": "one\n\n# not a comment\n  two\n", "b": "x", "c": "y\n\n", "d": "x"},
        )

    def test_quoted_keys(self):
        self.assertEqual(cw.load('"contents": "write"\n\'id-token\': read\n'), {"contents": "write", "id-token": "read"})

    def test_constructs_outside_the_subset_are_errors(self):
        for text in (
            "a: >\n  folded\n",
            "a: &anchor x\nb: *anchor\n",
            "a: plain\n  continued\n",
            'a: "quoted\n  continued"\n',
            "a: {b: c}\n",
            "a:\n\t- x\n",
            "a: 1\na: 2\n",
            "---\na: 1\n",
        ):
            with self.subTest(text=text):
                with self.assertRaises(cw.YamlError):
                    cw.load(text)

    def test_the_check_reports_an_unreadable_file(self):
        (problem,) = problems("on: push\njobs:\n  a: >\n    folded\n")
        self.assertIn("test.yml:3", problem)
        self.assertIn("folded", problem)


class Pinned(unittest.TestCase):
    """Issue #593: every action pinned by SHA."""

    def test_a_tag_fails(self):
        (problem,) = step_problems("- name: Cache\n  uses: actions/cache@v6\n")
        self.assertIn("uses actions/cache@v6, which is not pinned", problem)

    def test_a_full_sha_passes(self):
        self.assertEqual(step_problems("- uses: actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9 # v6.1.0\n"), [])
        self.assertEqual(step_problems(f"- uses: actions/cache/restore@{SHA}\n"), [])

    def test_short_shas_branches_and_docker_tags_fail(self):
        for ref in (f"actions/checkout@{SHA[:12]}", "dtolnay/rust-toolchain@master", "docker://alpine:3.20", "actions/checkout"):
            with self.subTest(ref=ref):
                self.assertEqual(len(step_problems(f"- uses: {ref}\n")), 1)

    def test_docker_digests_pass(self):
        self.assertEqual(step_problems(f"- uses: docker://alpine@sha256:{HASH}\n"), [])

    def test_reusable_workflows(self):
        local = "on: push\npermissions: {}\njobs:\n  gate:\n    uses: ./.github/workflows/ai-models.yml\n"
        self.assertEqual(problems(local), [])
        remote = local.replace("./.github/workflows/ai-models.yml", "octo/flows/.github/workflows/x.yml@v1")
        (problem,) = problems(remote)
        self.assertIn("job gate: uses octo/flows/.github/workflows/x.yml@v1", problem)


class Pipe(unittest.TestCase):
    """Issue #597: no download piped into a shell."""

    def test_the_generated_install_dist_step_fails(self):
        (problem,) = step_problems("- name: Install dist\n  run: ${{ matrix.install_dist.run }}\n")
        self.assertIn("matrix.install_dist", problem)

    def test_pipes_into_shells_fail(self):
        for line in (
            "curl --proto '=https' -LsSf https://example.com/install.sh | sh",
            "curl -fsSL https://example.com/x | bash -s -- --yes",
            "wget -qO- https://example.com/x |sudo sh",
            "curl -fsSL https://example.com/x | /bin/bash",
            "bash <(curl -fsSL https://example.com/x)",
            'sh -c "$(curl -fsSL https://example.com/x)"',
            "irm https://example.com/x | iex",
        ):
            with self.subTest(line=line):
                found = step_problems(f"- run: |\n    set -e\n    {line}\n")
                self.assertEqual(len(found), 1, found)
                self.assertIn("pipes a download into a shell", found[0])

    def test_other_pipes_pass(self):
        for line in (
            'echo "$SUM  file" | shasum -a 256 -c -',
            'echo "$SUM  file" | sha256sum -c -',
            "make || bash scripts/fallback.sh",
            "# never: curl https://example.com/x | sh",
            "printf '%s\\n' \"$x\" | grep -c .",
        ):
            with self.subTest(line=line):
                self.assertEqual(step_problems(f"- run: |\n    {line}\n"), [])


class Installer(unittest.TestCase):
    """Issue #597: the cargo-dist installer runs only after its hash check."""

    ENV = (
        "env:\n"
        '  CARGO_DIST_INSTALLER_URL: "https://github.com/axodotdev/cargo-dist/releases/download/v0.33.1/cargo-dist-installer.sh"\n'
        f'  CARGO_DIST_INSTALLER_SHA256: "{HASH}"\n'
    )

    def test_plans_step_before_the_fix_passes(self):
        step = f"""\
            - name: Install dist
              shell: bash
              run: |
                curl --proto '=https' --tlsv1.2 -LsSf https://github.com/axodotdev/cargo-dist/releases/download/v0.33.1/cargo-dist-installer.sh -o cargo-dist-installer.sh
                echo "{HASH}  cargo-dist-installer.sh" | sha256sum -c -
                sh cargo-dist-installer.sh
            """
        self.assertEqual(step_problems(step), [])

    def test_the_shared_env_step_passes(self):
        step = """\
            - name: Install dist
              shell: bash
              run: |
                installer="$RUNNER_TEMP/cargo-dist-installer.sh"
                curl --proto '=https' --tlsv1.2 -LsSf "$CARGO_DIST_INSTALLER_URL" -o "$installer"
                echo "$CARGO_DIST_INSTALLER_SHA256  $installer" | shasum -a 256 -c -
                sh "$installer"
            """
        self.assertEqual(step_problems(step, self.ENV), [])
        unpinned = self.ENV.replace(HASH, "${{ vars.DIST_SHA }}")
        (problem,) = step_problems(step, unpinned)
        self.assertIn("no pinned sha256", problem)
        (problem,) = step_problems(step)
        self.assertIn("no pinned sha256", problem)

    def test_no_check_or_a_late_check_fails(self):
        unchecked = """\
            - run: |
                curl -LsSf "$URL" -o cargo-dist-installer.sh
                sh cargo-dist-installer.sh
            """
        (problem,) = step_problems(unchecked)
        self.assertIn("never checks its sha256", problem)
        late = f"""\
            - run: |
                curl -LsSf "$URL" -o cargo-dist-installer.sh
                sh cargo-dist-installer.sh
                echo "{HASH}  cargo-dist-installer.sh" | sha256sum -c -
            """
        (problem,) = step_problems(late)
        self.assertIn("before it checks", problem)

    def test_a_fetch_through_the_env_url_is_checked_too(self):
        for url in ('"$CARGO_DIST_INSTALLER_URL"', '"${CARGO_DIST_INSTALLER_URL}"', "${{ env.CARGO_DIST_INSTALLER_URL }}"):
            with self.subTest(url=url):
                step = f"""\
                    - run: |
                        curl -LsSf {url} -o "$RUNNER_TEMP/dist.sh"; sh "$RUNNER_TEMP/dist.sh"
                    """
                (problem,) = step_problems(step, self.ENV)
                self.assertIn("never checks its sha256", problem)

    def test_a_run_before_the_check_on_one_line_fails(self):
        step = """\
            - run: |
                curl -LsSf "$CARGO_DIST_INSTALLER_URL" -o i.sh
                sh i.sh && echo "$CARGO_DIST_INSTALLER_SHA256  i.sh" | shasum -a 256 -c -
            """
        (problem,) = step_problems(step, self.ENV)
        self.assertIn("before it checks", problem)
        checked_first = """\
            - run: |
                curl -LsSf "$CARGO_DIST_INSTALLER_URL" -o i.sh
                echo "$CARGO_DIST_INSTALLER_SHA256  i.sh" | shasum -a 256 -c - && sh i.sh
            """
        self.assertEqual(step_problems(checked_first, self.ENV), [])

    def test_a_check_that_cannot_fail_fails(self):
        for tail in ("|| true", "|| :", '|| echo "checksum mismatch"'):
            with self.subTest(tail=tail):
                step = f"""\
                    - run: |
                        curl -LsSf "$CARGO_DIST_INSTALLER_URL" -o i.sh
                        echo "$CARGO_DIST_INSTALLER_SHA256  i.sh" | shasum -a 256 -c - {tail}
                        sh i.sh
                    """
                (problem,) = step_problems(step, self.ENV)
                self.assertIn("must stop the step", problem)
        for tail in ("|| exit 1", '|| { echo "checksum mismatch" >&2; exit 1; }', "|| false"):
            with self.subTest(tail=tail):
                step = f"""\
                    - run: |
                        curl -LsSf "$CARGO_DIST_INSTALLER_URL" -o i.sh
                        echo "$CARGO_DIST_INSTALLER_SHA256  i.sh" | shasum -a 256 -c - {tail}
                        sh i.sh
                    """
                self.assertEqual(step_problems(step, self.ENV), [])


class Pip(unittest.TestCase):
    """Issue #598: every pip install is hashed, the build backend included."""

    def test_the_lines_before_the_fix_fail(self):
        for line in (
            'python3 -m pip install $pip_flags "ninja==1.13.2"',
            '"$RUNNER_TEMP/meson-venv/bin/pip" install "ninja==1.13.2"',
            '"$RUNNER_TEMP/ai-venv/bin/pip" install --quiet onnx==1.23.2',
            'python -m pip install "build==1.6.1" "twine==7.0.0"',
            "python3 -m pip install --quiet pillow",
            "python3 -m pip install $pip_flags --require-hashes -r meson-requirements.txt",
        ):
            with self.subTest(line=line):
                found = step_problems(f"- run: |\n    {line}\n")
                self.assertTrue(found)
                self.assertTrue(all("`pip install`" in problem for problem in found), found)

    def test_hashed_installs_pass(self):
        for line in (
            "python3 -m pip install --user --break-system-packages --require-hashes -r scripts/requirements/dav1d-build.txt",
            '"$RUNNER_TEMP/ai-venv/bin/pip" install --quiet --require-hashes -r scripts/requirements/ai-fixtures.txt',
            "pip3 install --require-hashes --requirement=build-requirements.txt && echo done",
            "python -m pip install --require-hashes -r a.txt -r b.txt",
        ):
            with self.subTest(line=line):
                self.assertEqual(step_problems(f"- run: |\n    {line}\n"), [])

    def test_continued_lines_are_one_command(self):
        self.assertEqual(step_problems("- run: |\n    pip install \\\n      --require-hashes \\\n      -r x.txt\n"), [])
        (problem,) = step_problems("- run: |\n    pip install --require-hashes -r x.txt \\\n      extra\n")
        self.assertIn("names 'extra'", problem)

    def test_what_a_hashed_install_must_not_do(self):
        cases = {
            "pip install -r x.txt": "has no --require-hashes",
            "pip install --require-hashes": "reads no requirements file",
            "pip install --require-hashes -r x.txt --index-url https://example.com/simple": "--index-url",
            "pip install --require-hashes -r x.txt -e .": "-e",
        }
        for line, expected in cases.items():
            with self.subTest(line=line):
                found = step_problems(f"- run: {line}\n")
                self.assertTrue(any(expected in problem for problem in found), found)

    def test_python_build_needs_no_isolation(self):
        (problem,) = step_problems("- run: python -m build\n")
        self.assertIn("--no-isolation", problem)
        self.assertEqual(step_problems("- run: python -m build --no-isolation\n"), [])
        self.assertEqual(step_problems("- run: python3 -m build -n --sdist\n"), [])


class Dav1d(unittest.TestCase):
    """Issue #596: no workflow builds dav1d from a ref fetched at build time."""

    def test_the_variable_fails_at_every_level(self):
        wf = "on: push\npermissions: {}\nenv:\n  SYSTEM_DEPS_DAV1D_BUILD_INTERNAL: \"always\"\njobs: {}\n"
        (problem,) = problems(wf)
        self.assertIn("workflow env sets SYSTEM_DEPS_DAV1D_BUILD_INTERNAL", problem)
        job = textwrap.dedent(
            """\
            on: push
            permissions: {}
            jobs:
              macos:
                runs-on: macos-latest
                env:
                  SYSTEM_DEPS_DAV1D_BUILD_INTERNAL: "always"
                steps:
                  - run: cargo check
            """
        )
        (problem,) = problems(job)
        self.assertIn("job env sets", problem)
        found = step_problems('- env:\n    SYSTEM_DEPS_DAV1D_BUILD_INTERNAL: "always"\n  run: dist build\n')
        self.assertEqual(len(found), 1, found)
        found = step_problems("- run: |\n    export SYSTEM_DEPS_DAV1D_BUILD_INTERNAL=always\n    cargo build\n")
        self.assertEqual(len(found), 1, found)

    def test_the_build_script_passes(self):
        self.assertEqual(step_problems('- run: bash scripts/build_dav1d.sh "$RUNNER_TEMP/dav1d"\n'), [])


class Secrets(unittest.TestCase):
    """Issue #598: nothing installs or builds Python packages next to a registry secret."""

    def test_the_old_publish_step_fails(self):
        step = """\
            - name: Build and upload
              working-directory: sdks/python
              run: |
                python -m pip install --require-hashes -r build-requirements.txt
                python -m build --no-isolation
                python -m twine upload dist/*
              env:
                TWINE_USERNAME: __token__
                TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
            """
        found = [p for p in step_problems(step) if "can read PYPI_TOKEN" in p]
        self.assertEqual(len(found), 1, step_problems(step))

    def test_the_upload_step_alone_passes(self):
        step = """\
            - name: Upload
              run: python -m twine upload --non-interactive dist/*
              env:
                TWINE_USERNAME: __token__
                TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
            """
        self.assertEqual([p for p in step_problems(step) if "can read" in p], [])

    def test_a_job_level_secret_reaches_every_step(self):
        wf = textwrap.dedent(
            """\
            on: push
            permissions: {}
            jobs:
              pypi:
                runs-on: ubuntu-22.04
                environment: release
                env:
                  TOKEN: ${{ secrets.PYPI_TOKEN }}
                steps:
                  - run: python -m pip install --require-hashes -r r.txt
            """
        )
        (problem,) = problems(wf)
        self.assertIn("can read PYPI_TOKEN", problem)

    def test_github_token_is_not_a_registry_secret(self):
        step = "- run: python -m pip install --require-hashes -r r.txt\n  env:\n    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n"
        self.assertEqual(step_problems(step), [])


# The job graph of release.yml before issue #599: the release and the
# images publish before anyone approves.
OLD_GATE = """\
on: push
permissions:
  "contents": "read"
jobs:
  plan:
    runs-on: ubuntu-22.04
    steps:
      - run: dist plan
  ai-models:
    needs: plan
    permissions:
      contents: read
    uses: ./.github/workflows/ai-models.yml
  host:
    needs:
      - plan
      - ai-models
    if: ${{ always() && needs.plan.result == 'success' && needs.ai-models.result == 'success' }}
    runs-on: ubuntu-22.04
    permissions:
      "contents": "write"
    steps:
      - run: bash scripts/publish_github_release.sh
  build-images:
    needs: host
    if: ${{ !github.event.pull_request }}
    runs-on: ubuntu-22.04
    permissions:
      contents: read
      packages: write
    steps:
      - run: docker push
  publish-crate:
    needs:
      - plan
      - host
    runs-on: ubuntu-22.04
    environment: release
    permissions:
      contents: read
    steps:
      - run: cargo publish
        env:
          CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_REGISTRY_TOKEN }}
"""

# The same jobs after the fix: one approval job ahead of everything that
# publishes.
NEW_GATE = OLD_GATE.replace(
    """  host:
    needs:
      - plan
      - ai-models
    if: ${{ always() && needs.plan.result == 'success' && needs.ai-models.result == 'success' }}
""",
    """  release-gate:
    needs:
      - plan
      - ai-models
    if: ${{ always() && needs.plan.result == 'success' && needs.ai-models.result == 'success' }}
    runs-on: ubuntu-22.04
    environment: release
    permissions: {}
    steps:
      - run: echo approved
  host:
    needs:
      - plan
      - ai-models
      - release-gate
    if: ${{ always() && needs.release-gate.result == 'success' && needs.plan.result == 'success' }}
""",
)


class Gate(unittest.TestCase):
    """Issue #599: everything that publishes waits for the release approval."""

    def test_the_old_layout_fails_on_host_and_build_images(self):
        found = problems(OLD_GATE)
        self.assertEqual([p.split(" can ")[0] for p in found], ["test.yml: job host", "test.yml: job build-images"])
        self.assertIn("contents: write", found[0])
        self.assertIn("packages: write", found[1])

    def test_the_gated_layout_passes(self):
        self.assertEqual(problems(NEW_GATE), [])

    def test_an_if_that_ignores_the_gate_result_fails(self):
        for condition in (
            "${{ always() && needs.plan.result == 'success' }}",
            "${{ !cancelled() && needs.plan.result == 'success' }}",
            "${{ always() && (needs.release-gate.result == 'success' || github.event_name == 'workflow_dispatch') }}",
            "${{ failure() || needs.release-gate.result == 'success' }}",
        ):
            with self.subTest(condition=condition):
                text = NEW_GATE.replace(
                    "if: ${{ always() && needs.release-gate.result == 'success' && needs.plan.result == 'success' }}",
                    f"if: {condition}",
                )
                found = problems(text)
                self.assertEqual([p.split(" can ")[0] for p in found], ["test.yml: job host", "test.yml: job build-images"])

    def test_a_plain_if_waits_for_success(self):
        text = NEW_GATE.replace(
            "if: ${{ always() && needs.release-gate.result == 'success' && needs.plan.result == 'success' }}",
            "if: ${{ !github.event.pull_request }}",
        )
        self.assertEqual(problems(text), [])

    def test_secrets_ids_and_missing_permissions_need_the_gate(self):
        base = "on: push\npermissions:\n  contents: read\njobs:\n  j:\n    runs-on: ubuntu-22.04\n"
        cases = {
            "    steps:\n      - run: npm publish\n        env:\n          T: ${{ secrets.NPM_TOKEN }}\n": "reads NPM_TOKEN",
            "    permissions:\n      id-token: write\n    steps:\n      - run: true\n": "id-token: write",
            "    uses: ./.github/workflows/x.yml\n    secrets: inherit\n": "secrets: inherit",
        }
        for job, reason in cases.items():
            with self.subTest(reason=reason):
                text = base + job
                if "uses:" in job:
                    text = text.replace("    runs-on: ubuntu-22.04\n", "")
                (problem,) = problems(text)
                self.assertIn(reason, problem)
                self.assertEqual(problems(text.replace("  j:\n", "  j:\n    environment: release\n")), [])
        (problem,) = problems("on: push\njobs:\n  j:\n    runs-on: ubuntu-22.04\n    steps:\n      - run: true\n")
        self.assertIn("no permissions block", problem)

    def test_the_environment_can_be_a_mapping(self):
        text = OLD_GATE.replace(
            "    permissions:\n      \"contents\": \"write\"\n",
            "    environment:\n      name: release\n      url: https://example.com\n    permissions:\n      \"contents\": \"write\"\n",
        )
        self.assertEqual(problems(text), [])


class ReleaseLayout(unittest.TestCase):
    """The release workflow as docs/release-workflow.yml lands it."""

    @classmethod
    def setUpClass(cls):
        cls.jobs = cw.load((ROOT / "docs" / "release-workflow.yml").read_text(encoding="utf-8"))["jobs"]

    def step_index(self, job, name):
        names = [step.get("name") for step in self.jobs[job]["steps"]]
        self.assertIn(name, names, f"{job} has no step {name!r}")
        return names.index(name)

    def test_the_approval_comes_before_everything_that_publishes(self):
        gate = self.jobs["release-gate"]
        self.assertEqual(gate["environment"], "release")
        self.assertEqual(gate["permissions"], {})
        for build in ("build-local-artifacts", "build-global-artifacts", "build-pypi", "ai-models"):
            self.assertIn(build, gate["needs"])
        self.assertIn("release-gate", self.jobs["host"]["needs"])
        self.assertIn("needs.release-gate.result == 'success'", cw.conjuncts(self.jobs["host"]["if"]))
        self.assertEqual(self.jobs["build-images"]["needs"], "host")
        for job in ("publish-crate", "publish-npm", "publish-mcp", "publish-pypi"):
            self.assertIn("host", self.jobs[job]["needs"])
            self.assertEqual(self.jobs[job]["environment"], "release")

    def test_pull_requests_skip_the_approval(self):
        self.assertIn("needs.plan.outputs.publishing == 'true'", cw.conjuncts(self.jobs["release-gate"]["if"]))

    def test_the_formula_is_uploaded_before_the_release_is_published(self):
        self.assertLess(
            self.step_index("host", "Generate the Homebrew formula"),
            self.step_index("host", "Create or update GitHub Release"),
        )
        self.assertIn("> artifacts/crabcrop.rb", self.jobs["host"]["steps"][self.step_index("host", "Generate the Homebrew formula")]["run"])

    def test_pypi_builds_without_secrets_and_uploads_with_the_token_on_one_step(self):
        build = self.jobs["build-pypi"]
        self.assertEqual(cw.secret_names(build), [])
        upload = [step for step in build["steps"] if str(step.get("uses", "")).startswith("actions/upload-artifact@")]
        self.assertEqual([step["with"]["name"] for step in upload], ["pypi-dist"])
        publish = self.jobs["publish-pypi"]
        self.assertIn("build-pypi", publish["needs"])
        with_token = [step["name"] for step in publish["steps"] if cw.secret_names(step)]
        self.assertEqual(with_token, ["Upload to PyPI"])
        self.assertNotIn("build", " ".join(str(step.get("run", "")) for step in publish["steps"]).replace("pypi-dist", ""))

    def test_every_dav1d_build_uses_the_script(self):
        for path, job in (
            ("release-workflow.yml", "build-local-artifacts"),
            ("rust-workflow.yml", "macos"),
            ("fuzz-workflow.yml", "fuzz"),
            ("fuzz-workflow.yml", "tests"),
        ):
            with self.subTest(job=job):
                steps = cw.load((ROOT / "docs" / path).read_text(encoding="utf-8"))["jobs"][job]["steps"]
                runs = [str(step.get("run", "")) for step in steps]
                built = next(at for at, run in enumerate(runs) if "scripts/build_dav1d.sh" in run and "--check-binary" not in run)
                cargo = next(
                    at for at, run in enumerate(runs) if re.search(r"\bcargo (?:\+\S+ )?(?:build|check|test|fuzz)\b|\bdist build\b", run)
                )
                self.assertLess(built, cargo, f"{path} {job} builds before it builds dav1d")
        checked = self.step_index("build-local-artifacts", "Check that dav1d is linked statically")
        self.assertGreater(checked, self.step_index("build-local-artifacts", "Build artifacts"))


class Repository(unittest.TestCase):
    """The workflows this repository lands pass (their docs/ mirrors)."""

    def test_the_mirrors_pass(self):
        mirrors = sorted((ROOT / "docs").glob("*-workflow.yml"))
        self.assertTrue(mirrors)
        done = subprocess.run([sys.executable, str(SCRIPT), *map(str, mirrors)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_the_command_reports_each_problem_and_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.yml"
            bad.write_text(step_workflow("- uses: actions/cache@v6\n- run: curl -fsSL https://x | sh\n"))
            done = subprocess.run([sys.executable, str(SCRIPT), str(bad)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 1)
        lines = [line for line in done.stderr.splitlines() if line.startswith("  ")]
        self.assertEqual(len(lines), 2, done.stderr)
        self.assertNotIn("Traceback", done.stderr)


if __name__ == "__main__":
    unittest.main()
