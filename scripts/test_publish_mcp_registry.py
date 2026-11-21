#!/usr/bin/env python3
"""Tests for scripts/publish_mcp_registry.py.

A fake MCP registry answers the version lookups from a table and lists a
version once the publisher has published it; the publisher is a stub
that records its arguments, as a function and, in one case, as an
executable on disk. Each case asserts the publisher calls the release
job would make (issue #711), and that a prerelease, a version already
listed, a manifest that does not describe the release, and a namespace
no login can reach publish nothing. Run directly:
python3 scripts/test_publish_mcp_registry.py
"""
import contextlib
import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("publish_mcp_registry", HERE / "publish_mcp_registry.py")
pm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pm)

# The script turns a failure into a GitHub annotation under Actions; the
# cases read the plain message instead, and CI shows no error for them.
os.environ.pop("GITHUB_ACTIONS", None)

REGISTRY = "https://registry.example"
NAME = "io.github.crabcrop/crabcrop"


def manifests(version="0.14.0", **changes):
    """server.json and package.json for `version`, as sdks/mcp has them, with `changes` applied."""
    server = {
        "name": NAME,
        "version": version,
        "packages": [{"registryType": "npm", "identifier": "@crabcrop/mcp", "version": version, "transport": {"type": "stdio"}}],
    }
    package = {"name": "@crabcrop/mcp", "version": version, "mcpName": NAME}
    for key, value in changes.items():
        where, field = key.split("__")
        if where == "npm":
            server["packages"][0][field] = value
        else:
            (server if where == "server" else package)[field] = value
    return server, package


class Registry:
    """The registry's version lookups, and a publisher that adds to them."""

    def __init__(self, listed=(), status=None, login=0, publish=0, lists_publish=True):
        self.versions = set(listed)
        self.status, self.login_code, self.publish_code = status, login, publish
        self.lists_publish = lists_publish
        self.reads, self.calls = [], []

    def fetch(self, url):
        self.reads.append(url)
        if self.status is not None:
            return self.status, ""
        prefix = f"{REGISTRY}/v0.1/servers/io.github.crabcrop%2Fcrabcrop/versions/"
        if not url.startswith(prefix) or not url.endswith("?include_deleted=true"):
            return 404, ""
        version = url[len(prefix) : -len("?include_deleted=true")]
        if version not in self.versions:
            return 404, ""
        return 200, json.dumps({"server": {"name": NAME, "version": version}, "_meta": {}})

    def run(self, args):
        self.calls.append(args[1:])
        if args[1] == "login":
            return mock.Mock(returncode=self.login_code)
        if args[1] == "publish":
            if self.publish_code == 0 and self.lists_publish:
                server = json.loads(Path(args[2]).read_text(encoding="utf-8"))
                self.versions.add(server["version"])
            return mock.Mock(returncode=self.publish_code)
        raise AssertionError(f"unexpected publisher call {args}")


class PublishMcpRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="crabcrop-mcp-registry-")
        self.dir = Path(self.tmp.name)
        self.write(*manifests())

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, server, package):
        (self.dir / "server.json").write_text(json.dumps(server), encoding="utf-8")
        (self.dir / "package.json").write_text(json.dumps(package), encoding="utf-8")

    def main(self, registry, tag="v0.14.0", owner="crabcrop", token=None, wait="0"):
        """(exit code, progress lines, stderr) of one run against `registry`."""
        lines, err = [], io.StringIO()
        env = {"MCP_GITHUB_TOKEN": token} if token is not None else {}
        args = [
            "--tag", tag,
            "--publisher", "/opt/mcp-publisher",
            "--server-json", str(self.dir / "server.json"),
            "--package-json", str(self.dir / "package.json"),
            "--registry", REGISTRY,
            "--owner", owner,
            "--wait", wait,
        ]
        with mock.patch.dict(os.environ, env), contextlib.redirect_stderr(err):
            if token is None:
                os.environ.pop("MCP_GITHUB_TOKEN", None)
            code = pm.main(args, fetch=registry.fetch, run=registry.run, sleep=lambda seconds: None, out=lines.append)
        return code, lines, err.getvalue()

    def test_the_repository_owners_namespace_publishes_with_oidc(self):
        registry = Registry(listed={"0.13.0"})
        code, lines, err = self.main(registry, owner="crabcrop")
        self.assertEqual(code, 0, err)
        self.assertEqual(
            registry.calls,
            [["login", "github-oidc", "--registry", REGISTRY], ["publish", str(self.dir / "server.json")]],
        )
        self.assertIn(f"the MCP registry lists {NAME} 0.14.0", lines)

    def test_another_accounts_namespace_logs_in_with_the_token(self):
        # The entry is the crabcrop organization's, and the repository is
        # a user's: OIDC would grant only io.github.<that user>/.
        registry = Registry()
        code, lines, err = self.main(registry, owner="a-user", token="ghp_owner_token")
        self.assertEqual(code, 0, err)
        self.assertEqual(
            registry.calls,
            [["login", "github", "--token", "ghp_owner_token", "--registry", REGISTRY], ["publish", str(self.dir / "server.json")]],
        )
        self.assertFalse(any("ghp_owner_token" in line for line in lines), "the token was printed")

    def test_the_namespace_owner_is_matched_without_case(self):
        registry = Registry()
        code, _, err = self.main(registry, owner="CrabCrop", token="ghp_owner_token")
        self.assertEqual(code, 0, err)
        self.assertEqual(registry.calls[0][:2], ["login", "github-oidc"])

    def test_another_accounts_namespace_without_a_token_publishes_nothing(self):
        registry = Registry()
        for token in (None, ""):
            with self.subTest(token=token):
                code, _, err = self.main(registry, owner="a-user", token=token)
                self.assertEqual(code, 1)
                self.assertIn("grants a GitHub OIDC token only its repository owner's (io.github.a-user/)", err)
                self.assertIn("add MCP_GITHUB_TOKEN", err)
                self.assertEqual(registry.calls, [])

    def test_a_listed_version_is_not_published_again(self):
        registry = Registry(listed={"0.14.0"})
        code, lines, err = self.main(registry)
        self.assertEqual(code, 0, err)
        self.assertEqual(registry.calls, [])
        self.assertIn(f"the MCP registry already lists {NAME} 0.14.0; nothing to publish", lines)
        # Deleted versions count: the lookup asks for them too.
        self.assertTrue(all(url.endswith("?include_deleted=true") for url in registry.reads), registry.reads)

    def test_a_prerelease_is_not_listed(self):
        self.write(*manifests("0.14.0-rc.1"))
        registry = Registry()
        code, lines, err = self.main(registry, tag="v0.14.0-rc.1")
        self.assertEqual(code, 0, err)
        self.assertEqual((registry.reads, registry.calls), ([], []))
        self.assertIn("0.14.0-rc.1 is a prerelease: not listing it in the MCP registry, where it would become the latest version", lines)

    def test_prereleases_follow_the_npm_dist_tag_rule(self):
        for version, prerelease in (("0.14.0", False), ("0.14.0-rc.1", True), ("1.2.3-beta.2+sha.5", True), ("1.2.3+build-7", False)):
            with self.subTest(version=version):
                self.assertEqual(pm.is_prerelease(version), prerelease)

    def test_a_manifest_that_does_not_describe_the_release_publishes_nothing(self):
        cases = {
            "server.json's version is '0.13.0', not '0.14.0'": {"server__version": "0.13.0"},
            "server.json's npm package version is '0.13.0', not '0.14.0'": {"npm__version": "0.13.0"},
            "server.json's npm package is '@crabcrop/url', not '@crabcrop/mcp'": {"npm__identifier": "@crabcrop/url"},
            "package.json's version is '0.13.0', not '0.14.0'": {"package__version": "0.13.0"},
            "package.json's mcpName is 'io.github.other/crabcrop', not the entry's name": {"package__mcpName": "io.github.other/crabcrop"},
            "server.json lists 0 npm packages, not one": {"npm__registryType": "pypi"},
        }
        for problem, changes in cases.items():
            with self.subTest(problem=problem):
                self.write(*manifests(**changes))
                registry = Registry()
                code, _, err = self.main(registry)
                self.assertEqual(code, 1)
                self.assertIn(problem, err)
                self.assertEqual((registry.reads, registry.calls), ([], []))

    def test_a_name_no_github_login_can_publish_is_refused(self):
        server, package = manifests(server__name="com.example/crabcrop", package__mcpName="com.example/crabcrop")
        self.write(server, package)
        registry = Registry()
        registry.fetch = lambda url: (404, "")
        code, _, err = self.main(registry)
        self.assertEqual(code, 1)
        self.assertIn("is not io.github.<account>/<server>", err)
        self.assertEqual(registry.calls, [])

    def test_an_unexpected_registry_answer_publishes_nothing(self):
        for status in (401, 403, 500, 502):
            with self.subTest(status=status):
                registry = Registry(status=status)
                code, _, err = self.main(registry)
                self.assertEqual(code, 1)
                self.assertIn(f"the MCP registry answered {status}", err)
                self.assertEqual(registry.calls, [])

    def test_an_answer_for_another_version_is_refused(self):
        registry = Registry()
        registry.fetch = lambda url: (200, json.dumps({"server": {"version": "0.13.0"}}))
        code, _, err = self.main(registry)
        self.assertEqual(code, 1)
        self.assertIn("answered version '0.13.0' when asked for '0.14.0'", err)
        self.assertEqual(registry.calls, [])

    def test_a_failed_login_or_publish_fails_the_job(self):
        registry = Registry(login=1)
        code, _, err = self.main(registry)
        self.assertEqual(code, 1)
        self.assertIn("mcp-publisher login failed (exit 1)", err)
        self.assertEqual([call[0] for call in registry.calls], ["login"])
        registry = Registry(publish=1)
        code, _, err = self.main(registry)
        self.assertEqual(code, 1)
        self.assertIn("mcp-publisher publish failed (exit 1)", err)

    def test_a_publish_the_registry_does_not_list_fails(self):
        registry = Registry(lists_publish=False)
        code, _, err = self.main(registry)
        self.assertEqual(code, 1)
        self.assertIn(f"mcp-publisher published {NAME} 0.14.0, but the registry has not listed it within 0 s", err)

    def test_bad_arguments_publish_nothing(self):
        for tag in ("0.14.0", "v0.14", "latest"):
            with self.subTest(tag=tag):
                registry = Registry()
                code, _, _ = self.main(registry, tag=tag)
                self.assertEqual(code, 2)
                self.assertEqual((registry.reads, registry.calls), ([], []))

    def test_the_publisher_runs_as_a_program(self):
        stub = self.dir / "mcp-publisher"
        log = self.dir / "calls.log"
        stub.write_text(
            f"#!{sys.executable}\nimport json, sys\nwith open({str(log)!r}, 'a') as f:\n    f.write(json.dumps(sys.argv[1:]) + '\\n')\n",
            encoding="utf-8",
        )
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)

        def calls():
            return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []

        def fetch(url):
            # Listed once the stub has been asked to publish.
            if any(call[0] == "publish" for call in calls()):
                return 200, json.dumps({"server": {"version": "0.14.0"}})
            return 404, ""

        done = pm.publish(
            "0.14.0",
            str(stub),
            self.dir / "server.json",
            self.dir / "package.json",
            registry=REGISTRY,
            owner="crabcrop",
            fetch=fetch,
            wait=0,
            sleep=lambda seconds: None,
            out=lambda line: None,
        )
        self.assertTrue(done)
        self.assertEqual(calls(), [["login", "github-oidc", "--registry", REGISTRY], ["publish", str(self.dir / "server.json")]])

    def test_this_repositorys_manifest_describes_its_release(self):
        server = json.loads((ROOT / "sdks/mcp/server.json").read_text(encoding="utf-8"))
        package = json.loads((ROOT / "sdks/mcp/package.json").read_text(encoding="utf-8"))
        self.assertEqual(pm.manifest_problems(server, package, package["version"]), [])
        self.assertRegex(server["name"], pm.GITHUB_NAME)
        # The schema's own spelling, which every consumer of the file reads.
        self.assertEqual(server["packages"][0].get("runtimeHint"), "npx")

    def test_the_lookup_url_encodes_the_name(self):
        self.assertEqual(
            pm.version_url("https://registry.modelcontextprotocol.io/", NAME, "0.14.0"),
            "https://registry.modelcontextprotocol.io/v0.1/servers/io.github.crabcrop%2Fcrabcrop/versions/0.14.0?include_deleted=true",
        )


if __name__ == "__main__":
    unittest.main()
