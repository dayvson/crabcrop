#!/usr/bin/env python3
"""Tests for scripts/publish_crates.py.

A fake sparse index and a fake cargo stand in for crates.io: the index
answers from a table of versions, and `cargo publish` adds the version to
it, after as many index reads as a case asks, as crates.io does once its
index catches up. `cargo package` writes the server crate's package with
the Cargo.lock cargo would give it: crabcrop-url at the newest version
that the server's `0.y` requirement allows among those the index cargo
reads lists, which can lag the index the script reads. Each case asserts
the exact cargo commands the release job would run and what it says when
it stops (issue #711). The last case reads this workspace with the real
cargo. Run directly:
python3 scripts/test_publish_crates.py
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("publish_crates", HERE / "publish_crates.py")
pc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pc)

# The script turns a failure into a GitHub annotation under Actions; the
# cases read the plain message instead, and CI shows no error for them.
os.environ.pop("GITHUB_ACTIONS", None)

INDEX = "https://index.example"
TMP = None


def setUpModule():
    global TMP
    TMP = tempfile.TemporaryDirectory(prefix="crabcrop-publish-crates-")


def tearDownModule():
    TMP.cleanup()


def write_crate(path, root, lock):
    """A .crate holding only `root`/Cargo.lock, with `lock` as its text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = lock.encode("utf-8")
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(f"{root}/Cargo.lock")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))


def numeric(version):
    return tuple(int(part) for part in version.split("."))


class Done:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class Registry:
    """A workspace, its cargo, and the index cargo publishes to."""

    def __init__(self, versions=None, have=None, lag=0, fail=(), status=None, yanked=(), cargo_lag=0, package_fail=False):
        # The workspace: crabcrop depends on crabcrop-url by path, as the
        # root Cargo.toml has it, and on a dev-only path crate.
        self.have = have or {"crabcrop-url": "0.14.0", "crabcrop": "0.14.0"}
        self.index = {name: list(found) for name, found in (versions or {}).items()}
        self.lag, self.fail, self.status, self.yanked = lag, set(fail), status, set(yanked)
        # How many more `cargo package` runs read an index that does not
        # list this release's crabcrop-url yet.
        self.cargo_lag, self.package_fail = cargo_lag, package_fail
        self.commands, self.reads, self.pending = [], [], {}
        self.target = tempfile.mkdtemp(dir=TMP.name)

    def metadata(self):
        packages = [
            {"name": "crabcrop-url", "version": self.have["crabcrop-url"], "dependencies": [{"name": "hex", "kind": None}]},
            {
                "name": "crabcrop",
                "version": self.have["crabcrop"],
                "dependencies": [
                    {"name": "crabcrop-url", "kind": None, "path": "/w/crates/crabcrop-url"},
                    {"name": "fixtures", "kind": "dev", "path": "/w/tests/fixtures"},
                ],
            },
        ]
        return json.dumps({"packages": packages, "workspace_members": [], "target_directory": self.target})

    def package(self, name):
        """`cargo package -p name --no-verify`: the .crate, with the Cargo.lock cargo gives it."""
        if self.package_fail:
            return Done(returncode=101, stderr="error: failed to prepare local package for uploading")
        lock = [(name, self.have[name])]
        if name == "crabcrop":
            lagging = self.cargo_lag > 0
            self.cargo_lag -= lagging
            wanted = self.have["crabcrop-url"].split(".")[:2]
            seen = [
                vers
                for vers in self.index.get("crabcrop-url", [])
                if "-" not in vers and vers.split(".")[:2] == wanted and not (lagging and vers == self.have["crabcrop-url"])
            ]
            if not seen:
                return Done(returncode=101, stderr=f'error: failed to select a version for the requirement `crabcrop-url = "^{".".join(wanted)}"`')
            lock.append(("crabcrop-url", max(seen, key=numeric)))
        text = "version = 4\n\n" + "".join(f'[[package]]\nname = "{n}"\nversion = "{v}"\n\n' for n, v in sorted(lock))
        root = f"{name}-{self.have[name]}"
        write_crate(Path(self.target) / "package" / f"{root}.crate", root, text)
        return Done()

    def run(self, args, capture=False):
        self.commands.append(args)
        if args[:2] == ["cargo", "metadata"]:
            return Done(stdout=self.metadata())
        if args[:2] == ["cargo", "package"]:
            self.assertion(args == ["cargo", "package", "-p", args[3], "--locked", "--no-verify"], args)
            return self.package(args[3])
        if args[:2] == ["cargo", "publish"]:
            name = args[args.index("-p") + 1]
            if name in self.fail:
                return Done(returncode=101)
            self.pending[name] = self.lag
            return Done()
        raise AssertionError(f"unexpected command {args}")

    def fetch(self, url):
        self.reads.append(url)
        if self.status is not None:
            return self.status, ""
        name = url.rsplit("/", 1)[1]
        if name in self.pending:
            if self.pending[name] == 0:
                self.index.setdefault(name, []).append(self.have[name])
                del self.pending[name]
            else:
                self.pending[name] -= 1
        if name not in self.index:
            return 404, ""
        lines = [json.dumps({"name": name, "vers": vers, "yanked": (name, vers) in self.yanked}) for vers in self.index[name]]
        return 200, "\n".join(lines) + "\n"

    @staticmethod
    def assertion(holds, args):
        if not holds:
            raise AssertionError(f"unexpected command {args}")

    def published(self):
        return [args[args.index("-p") + 1] for args in self.commands if args[:2] == ["cargo", "publish"]]

    def cargo(self):
        """The cargo commands after `cargo metadata`, as `verb name`."""
        return [f"{args[1]} {args[args.index('-p') + 1]}" for args in self.commands if args[1] != "metadata"]

    def main(self, *args, tag="v0.14.0", fetch=None):
        """(exit code, lines printed as progress, stderr) of one run."""
        lines, err = [], io.StringIO()
        with contextlib.redirect_stderr(err):
            code = pc.main(
                ["--tag", tag, "--index", INDEX, "--poll", "1", *args],
                fetch=fetch or self.fetch,
                run=self.run,
                sleep=lambda seconds: None,
                out=lines.append,
            )
        return code, lines, err.getvalue()


class PublishCrates(unittest.TestCase):
    def test_publishes_each_crate_once_the_one_before_it_is_listed(self):
        registry = Registry(versions={"crabcrop-url": ["0.13.0"], "crabcrop": ["0.13.0"]}, lag=2)
        code, lines, err = registry.main("crabcrop-url", "crabcrop")
        self.assertEqual(code, 0, err)
        self.assertEqual(
            [args for args in registry.commands if args[:2] == ["cargo", "publish"]],
            [["cargo", "publish", "-p", "crabcrop-url", "--locked"], ["cargo", "publish", "-p", "crabcrop", "--locked"]],
        )
        # One read before the publish, then three while the index caught
        # up: crabcrop's publish waited for crabcrop-url to be listed.
        reads = [url for url in registry.reads if url.endswith("/cr/ab/crabcrop-url")]
        self.assertEqual(len(reads), 4, registry.reads)
        self.assertIn("crabcrop-url 0.14.0 is on crates.io", lines)
        self.assertEqual(registry.index["crabcrop-url"], ["0.13.0", "0.14.0"])
        # crabcrop's package was made, and its lockfile read, between the
        # two publishes.
        self.assertEqual(registry.cargo(), ["publish crabcrop-url", "package crabcrop", "publish crabcrop"])
        self.assertIn("crabcrop's package locks crabcrop-url at 0.14.0", lines)

    def test_a_rerun_publishes_only_what_the_earlier_attempt_did_not(self):
        registry = Registry(versions={"crabcrop-url": ["0.13.0", "0.14.0"], "crabcrop": ["0.13.0"]})
        code, lines, err = registry.main("crabcrop-url", "crabcrop")
        self.assertEqual(code, 0, err)
        self.assertEqual(registry.published(), ["crabcrop"])
        self.assertIn("crabcrop-url 0.14.0 is already on crates.io; skipping it", lines)

    def test_nothing_is_published_when_both_are_listed(self):
        registry = Registry(versions={"crabcrop-url": ["0.14.0"], "crabcrop": ["0.14.0-rc.1", "0.14.0"]})
        code, _, err = registry.main("crabcrop-url", "crabcrop")
        self.assertEqual(code, 0, err)
        self.assertEqual(registry.published(), [])

    def test_a_yanked_version_is_published_already(self):
        # crates.io never takes a version twice, yanked or not.
        registry = Registry(versions={"crabcrop-url": ["0.14.0"], "crabcrop": ["0.14.0"]}, yanked={("crabcrop", "0.14.0")})
        code, _, err = registry.main("crabcrop-url", "crabcrop")
        self.assertEqual(code, 0, err)
        self.assertEqual(registry.published(), [])

    def test_a_prerelease_is_not_the_release(self):
        registry = Registry(versions={"crabcrop-url": ["0.14.0-rc.1"], "crabcrop": ["0.14.0-rc.1"]})
        code, _, err = registry.main("crabcrop-url", "crabcrop")
        self.assertEqual(code, 0, err)
        self.assertEqual(registry.published(), ["crabcrop-url", "crabcrop"])

    def test_a_crate_the_index_has_never_seen_is_published(self):
        registry = Registry(versions={})
        code, _, err = registry.main("crabcrop-url", "crabcrop")
        self.assertEqual(code, 0, err)
        self.assertEqual(registry.published(), ["crabcrop-url", "crabcrop"])

    def test_a_crate_at_another_version_stops_everything_first(self):
        registry = Registry(have={"crabcrop-url": "0.14.0", "crabcrop": "0.13.0"})
        code, _, err = registry.main("crabcrop-url", "crabcrop")
        self.assertEqual(code, 1)
        self.assertIn("crabcrop is at 0.13.0, not 0.14.0, the version the tag names", err)
        self.assertEqual(registry.published(), [])
        self.assertEqual(registry.reads, [], "the index was read before the versions were checked")

    def test_a_crate_before_its_dependency_is_refused(self):
        registry = Registry()
        code, _, err = registry.main("crabcrop", "crabcrop-url")
        self.assertEqual(code, 1)
        self.assertIn("crabcrop depends on crabcrop-url, so crabcrop-url must come before it", err)
        self.assertEqual(registry.published(), [])

    def test_a_dev_dependency_sets_no_order(self):
        registry = Registry()
        packages, target = pc.workspace(registry.run)
        self.assertEqual(packages["crabcrop"], ("0.14.0", ["crabcrop-url"]))
        self.assertEqual(target, registry.target)
        packages = {"crabcrop": ("0.14.0", ["crabcrop-url"]), "fixtures": ("0.14.0", [])}
        self.assertEqual(pc.plan_problems(["crabcrop", "fixtures"], "0.14.0", packages), [])

    def test_an_unknown_or_repeated_crate_is_refused(self):
        registry = Registry()
        code, _, err = registry.main("crabcrop-url", "crabcrop-cli")
        self.assertEqual(code, 1)
        self.assertIn("crabcrop-cli is not a package of this workspace", err)
        code, _, err = registry.main("crabcrop-url", "crabcrop-url")
        self.assertEqual(code, 1)
        self.assertIn("a crate is named twice", err)
        self.assertEqual(registry.published(), [])

    def test_an_index_that_cannot_answer_stops_before_publishing(self):
        for status in (403, 500, 503):
            with self.subTest(status=status):
                registry = Registry(status=status)
                code, _, err = registry.main("crabcrop-url", "crabcrop")
                self.assertEqual(code, 1)
                self.assertIn(f"the crates.io index answered {status}", err)
                self.assertEqual(registry.published(), [])

    def test_an_unreachable_index_stops_before_publishing(self):
        registry = Registry()

        def unreachable(url):
            raise pc.Refusal(f"could not read {url}: timed out")

        code, _, err = registry.main("crabcrop-url", fetch=unreachable)
        self.assertEqual(code, 1)
        self.assertIn("could not read https://index.example/cr/ab/crabcrop-url", err)
        self.assertEqual(registry.published(), [])

    def test_an_index_line_that_is_not_json_stops(self):
        registry = Registry()
        code, _, err = registry.main("crabcrop-url", fetch=lambda url: (200, "{not json\n"))
        self.assertEqual(code, 1)
        self.assertIn("has a line that is not JSON", err)
        self.assertEqual(registry.published(), [])

    def test_a_failed_publish_stops_the_crates_after_it(self):
        registry = Registry(fail={"crabcrop-url"})
        code, _, err = registry.main("crabcrop-url", "crabcrop")
        self.assertEqual(code, 1)
        self.assertIn("cargo publish -p crabcrop-url failed (exit 101); crabcrop not published", err)
        self.assertEqual(registry.published(), ["crabcrop-url"])

    def test_a_publish_the_index_never_lists_fails_after_the_wait(self):
        registry = Registry(lag=10**6)
        code, _, err = registry.main("--wait", "0", "crabcrop-url", "crabcrop")
        self.assertEqual(code, 1)
        self.assertIn("crabcrop-url 0.14.0 was published, but the crates.io index has not listed it within 0 s", err)
        self.assertEqual(registry.published(), ["crabcrop-url"])

    def test_a_failure_under_actions_is_an_annotation(self):
        registry = Registry(status=500)
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}), contextlib.redirect_stdout(out):
            code, _, err = registry.main("crabcrop-url")
        self.assertEqual(code, 1)
        self.assertEqual(err, "")
        self.assertTrue(out.getvalue().startswith("::error title=crates.io publish::the crates.io index answered 500"), out.getvalue())

    def test_bad_arguments_publish_nothing(self):
        for args in (
            ["--tag", "0.14.0", "crabcrop"],
            ["--tag", "v0.14", "crabcrop"],
            ["--tag", "v0.14.0", "../crabcrop"],
            ["--tag", "v0.14.0", "--poll", "0", "crabcrop"],
        ):
            with self.subTest(args=args):
                registry = Registry()
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(pc.main(args, fetch=registry.fetch, run=registry.run, out=lambda line: None), 2)
                self.assertEqual(registry.commands, [])

    def test_the_server_crate_waits_until_its_package_locks_the_new_crabcrop_url(self):
        # A patch release: the server's "0.14" requirement also allows
        # 0.14.0, and cargo locks it while the index cargo reads does not
        # list 0.14.1 yet, though the one the script reads does. Published
        # then, crabcrop 0.14.1 would carry a Cargo.lock naming
        # crabcrop-url 0.14.0, and `cargo install --locked` would build it.
        have = {"crabcrop-url": "0.14.1", "crabcrop": "0.14.1"}
        registry = Registry(versions={"crabcrop-url": ["0.14.0"], "crabcrop": ["0.14.0"]}, have=have, cargo_lag=2)
        code, lines, err = registry.main("crabcrop-url", "crabcrop", tag="v0.14.1")
        self.assertEqual(code, 0, err)
        self.assertEqual(
            registry.cargo(),
            ["publish crabcrop-url", "package crabcrop", "package crabcrop", "package crabcrop", "publish crabcrop"],
        )
        self.assertIn("crabcrop's package locks crabcrop-url at 0.14.1", lines)

    def test_a_package_that_still_locks_the_previous_release_is_not_published(self):
        have = {"crabcrop-url": "0.14.1", "crabcrop": "0.14.1"}
        registry = Registry(versions={"crabcrop-url": ["0.14.0"], "crabcrop": ["0.14.0"]}, have=have, cargo_lag=10**6)
        code, _, err = registry.main("--wait", "0", "crabcrop-url", "crabcrop", tag="v0.14.1")
        self.assertEqual(code, 1)
        self.assertIn("crabcrop's package locks crabcrop-url 0.14.0, not 0.14.1; crabcrop is not published", err)
        self.assertEqual(registry.published(), ["crabcrop-url"])

    def test_a_package_that_cannot_resolve_yet_is_packaged_again(self):
        # A minor release: the "0.14" requirement allows no published
        # version until 0.14.0 is listed, so packaging fails meanwhile.
        registry = Registry(versions={"crabcrop-url": ["0.13.0"], "crabcrop": ["0.13.0"]}, cargo_lag=1)
        code, _, err = registry.main("crabcrop-url", "crabcrop")
        self.assertEqual(code, 0, err)
        self.assertEqual(registry.cargo(), ["publish crabcrop-url", "package crabcrop", "package crabcrop", "publish crabcrop"])
        registry = Registry(versions={"crabcrop-url": ["0.13.0"], "crabcrop": ["0.13.0"]}, cargo_lag=10**6)
        code, _, err = registry.main("--wait", "0", "crabcrop-url", "crabcrop")
        self.assertEqual(code, 1)
        self.assertIn("cargo package -p crabcrop failed (exit 101): error: failed to select a version", err)
        self.assertEqual(registry.published(), ["crabcrop-url"])

    def test_a_rerun_checks_the_lock_before_it_publishes_the_server_crate(self):
        registry = Registry(versions={"crabcrop-url": ["0.13.0", "0.14.0"], "crabcrop": ["0.13.0"]}, cargo_lag=1)
        code, _, err = registry.main("crabcrop-url", "crabcrop")
        self.assertEqual(code, 0, err)
        self.assertEqual(registry.cargo(), ["package crabcrop", "package crabcrop", "publish crabcrop"])

    def test_a_package_that_fails_to_build_its_lock_is_not_published(self):
        registry = Registry(versions={"crabcrop-url": ["0.14.0"], "crabcrop": ["0.13.0"]}, package_fail=True)
        code, _, err = registry.main("--wait", "0", "crabcrop-url", "crabcrop")
        self.assertEqual(code, 1)
        self.assertIn("cargo package -p crabcrop failed (exit 101)", err)
        self.assertEqual(registry.published(), [])

    def test_no_verify_publishes_without_building(self):
        registry = Registry(versions={"crabcrop-url": ["0.13.0"], "crabcrop": ["0.13.0"]})
        code, _, err = registry.main("--no-verify", "crabcrop-url", "crabcrop")
        self.assertEqual(code, 0, err)
        self.assertEqual(
            [args for args in registry.commands if args[:2] == ["cargo", "publish"]],
            [
                ["cargo", "publish", "-p", "crabcrop-url", "--locked", "--no-verify"],
                ["cargo", "publish", "-p", "crabcrop", "--locked", "--no-verify"],
            ],
        )

    def test_the_lock_check_reads_the_lockfile_cargo_writes(self):
        lock = (
            "# This file is automatically @generated by Cargo.\n# It is not intended for manual editing.\nversion = 4\n\n"
            '[[package]]\nname = "crabcrop"\nversion = "0.14.1"\ndependencies = [\n "crabcrop-url",\n "hex",\n]\n\n'
            '[[package]]\nname = "crabcrop-url"\nversion = "0.14.1"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n'
            'checksum = "8814850c136d4dd6e473413897d2233174f4216d199fd809f99b717644754c1a"\n\n'
            '[[package]]\nname = "hex"\nversion = "0.4.3"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n'
        )
        target = Path(tempfile.mkdtemp(dir=TMP.name))
        write_crate(target / "package" / "crabcrop-0.14.1.crate", "crabcrop-0.14.1", lock)
        locked = pc.packaged_lock("crabcrop", "0.14.1", target, lambda args, capture=False: Done())
        self.assertEqual(locked, {"crabcrop": {"0.14.1"}, "crabcrop-url": {"0.14.1"}, "hex": {"0.4.3"}})
        # A package cargo did not write is a refusal, never a traceback.
        with self.assertRaises(pc.Refusal):
            pc.packaged_lock("crabcrop", "0.14.2", target, lambda args, capture=False: Done())

    def test_index_paths_follow_cargo(self):
        for name, path in (("a", "1/a"), ("ab", "2/ab"), ("abc", "3/a/abc"), ("crabcrop", "cr/ab/crabcrop"), ("Crabcrop-URL", "cr/ab/crabcrop-url")):
            with self.subTest(name=name):
                self.assertEqual(pc.index_path(name), path)

    @unittest.skipIf(shutil.which("cargo") is None, "cargo is not installed")
    def test_this_workspace_publishes_crabcrop_url_first(self):
        def run(args, capture=False):
            return subprocess.run(args, capture_output=capture, text=True, cwd=ROOT)

        packages, target = pc.workspace(run)
        self.assertTrue(Path(target).is_absolute(), target)
        version = packages["crabcrop"][0]
        self.assertEqual(packages["crabcrop-url"][0], version)
        self.assertIn("crabcrop-url", packages["crabcrop"][1])
        self.assertEqual(pc.plan_problems(["crabcrop-url", "crabcrop"], version, packages), [])
        self.assertTrue(pc.plan_problems(["crabcrop", "crabcrop-url"], version, packages))


if __name__ == "__main__":
    unittest.main()
