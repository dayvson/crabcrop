#!/usr/bin/env python3
"""Tests for scripts/check_docs.py.

Each case writes a small repository whose code and docs agree, breaks one
side, and expects the matching check to name the break; the pull-request
check runs against a throwaway git history. Run directly:
python3 scripts/test_check_docs.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_docs  # noqa: E402

SCRIPT = HERE / "check_docs.py"

MAIN_RS = r"""
use std::env;

const DEFAULT_WIDTH_STEP: u32 = 100;

fn run() -> Result<(), String> {
    let port: u16 = parse_env_value("PORT", env::var("PORT").ok(), 3000)?;
    let cache_mb: usize = parse_env("CRABCROP_CACHE_MAX_MB", 256)?;
    let unsafe_ok = parse_bool("CRABCROP_ALLOW_UNSAFE", false)?;
    let block = parse_bool_value(
        "CRABCROP_BLOCK_PRIVATE_SOURCES",
        get("CRABCROP_BLOCK_PRIVATE_SOURCES"),
        true,
    )?;
    let step: u32 = parse_env("CRABCROP_WIDTH_STEP", DEFAULT_WIDTH_STEP)?;
    let speed: u8 = parse_env("CRABCROP_AVIF_SPEED", crate::engine::DEFAULT_AVIF_SPEED)?;
    let budget: u64 = parse_env_value(
        "CRABCROP_SOURCE_BUFFER_MAX_MB",
        get("CRABCROP_SOURCE_BUFFER_MAX_MB"),
        default_mb,
    )?;
    let key = env::var("AWS_ACCESS_KEY_ID");
    let model = env::var("CRABCROP_TEST_ORT_DYLIB");
    Ok(())
}

#[cfg(test)]
mod tests {
    // A closing brace in a comment: }
    const QUOTE: &str = "}";
    const RAW: &str = r#"{"error":{"code":"}"}"#;
    const BRACE: char = '}';

    fn reads<'a>(name: &'a str) -> &'a str {
        let _ = ("CRABCROP_TEST_FACE_MODEL", "CRABCROP_ONLY_IN_A_TEST");
        name
    }
}

fn after_the_tests() -> bool {
    std::env::var_os("CRABCROP_LATE").is_some()
}
"""

ENGINE_RS = "pub const DEFAULT_AVIF_SPEED: u8 = 9;\n"

CLI_RS = 'fn base() -> Option<String> { std::env::var("CRABCROP_BASE").ok() }\n'

SERVER_RS = r"""
fn forbidden() -> Response {
    error_response(StatusCode::FORBIDDEN, "bad_signature", msg, accept)
}

fn parts(e: EngineError) -> (StatusCode, &'static str, String) {
    match e {
        EngineError::Decode(_) => (
            StatusCode::BAD_GATEWAY,
            "decode_failed",
            "source image could not be decoded".to_string(),
        ),
        EngineError::Busy => (StatusCode::SERVICE_UNAVAILABLE, "server_busy", busy),
    }
}

#[cfg(test)]
mod tests;
"""

SERVER_TESTS_RS = r"""
#[test]
fn codes() {
    assert_eq!(code_of(bad()), "bad_signature");
    assert_eq!(body, r#"{"error":{"code":"decode_failed"}}"#);
    assert!(text.contains("\"server_busy\""));
    let not_found = 1;
}
"""

URL_RS = r"""
impl ParseError {
    pub fn code(&self) -> &'static str {
        match self {
            ParseError::EmptyPath => "empty_path",
            ParseError::InvalidSize(_) => "invalid_size",
        }
    }
}

fn parse_filters(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    for part in split(rest) {
        let filter = match (name, params) {
            ("grayscale", None) => Filter::Grayscale,
            ("blur", Some(p)) => {
                if p.is_empty() {
                    return Err(invalid());
                }
                Filter::Blur(p.parse()?)
            }
            ("blur_region", Some(p))
            | ("pixelate_region", Some(p)) => region(p)?,
            ("watermark", Some(p)) => watermark(p)?,
            ("quality", Some(p)) => quality(p)?,
            _ => return Err(invalid()),
        };
    }
}
"""

CONFIGURATION_MD = """# Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `CRABCROP_CACHE_MAX_MB` | `256` | Result cache budget. |
| `PORT` | `3000` | Listen port. |
| `CRABCROP_ALLOW_UNSAFE` | off | Honors `unsafe` URLs. |
| `CRABCROP_BLOCK_PRIVATE_SOURCES` | on | Refuses private sources. |
| `CRABCROP_WIDTH_STEP` | `100` | Width hint step. |
| `CRABCROP_AVIF_SPEED` | `9` | AVIF speed. |
| `CRABCROP_SOURCE_BUFFER_MAX_MB` | 40 per core | Source buffer budget. |

Credentials come from `AWS_ACCESS_KEY_ID`. `CRABCROP_LATE` is read late.
"""

RESPONSES_MD = """# Responses

Grammar failures (a `400` below) carry the parser's stable names: `empty_path`, `invalid_size`.

| Code | Status | Failure |
| --- | --- | --- |
| `bad_signature` | `403` | The signature does not verify. |
| `decode_failed` | `502` | The bytes could not be decoded. |
| `server_busy` | `503` | A gate waited past its timeout. |
| `uploads_not_configured` | `404` | Reserved; not sent today. |

The statuses you can get:

| Status | Meaning |
| --- | --- |
| `400` | Parse failures. |
| `403` | Bad signature. |
| `404` | Reserved upload code. |
| `502` | Decode failures. |
| `503` | Shed. |
"""

FILTERS_MD = """# Filters

| Filter | Parameters | Description | Example |
| --- | --- | --- | --- |
| `grayscale` | none | Gray. | `filters:grayscale` |
| `blur` | `blur(2)` | Blur. | `filters:blur(2)` |
| `blur_region` | `blur_region(x,y,w,h)` | Blur a box. | `filters:blur_region(1,2,3,4)` |
| `pixelate_region` | `pixelate_region(x,y,w,h)` | Pixelate a box. | `filters:pixelate_region(1,2,3,4)` |

| Directive | Parameters | Description | Example |
| --- | --- | --- | --- |
| `quality` | `quality(80)` | Quality. | `filters:quality(80)` |

| Parameter | Description |
| --- | --- |
| `url` | Not a filter: an argument table. |

```
filters:watermark(<url>,<x>,<y>,<alpha>)
```
"""

ENDPOINTS_MD = "# Endpoints\n\nThe CLI reads `CRABCROP_BASE`.\n"
README_MD = "# crabcrop\n\nSet `CRABCROP_CACHE_MAX_MB` to size the cache.\n"
CHANGELOG_MD = "# Changelog\n\n## [Unreleased]\n\n## [0.1.0] - 2026-01-01\n\n- First.\n"
FUNCTIONAL_PY = 'EXPECT = {"code": "empty_path"}\n'
VECTORS_JSON = json.dumps({"cases": [{"ok": False, "error": "invalid_size"}]})

TREE = {
    "src/main.rs": MAIN_RS,
    "src/engine.rs": ENGINE_RS,
    "src/cli.rs": CLI_RS,
    "src/server.rs": SERVER_RS,
    "src/server/tests.rs": SERVER_TESTS_RS,
    "src/url.rs": URL_RS,
    "docs/configuration.md": CONFIGURATION_MD,
    "docs/responses.md": RESPONSES_MD,
    "docs/filters.md": FILTERS_MD,
    "docs/endpoints.md": ENDPOINTS_MD,
    "README.md": README_MD,
    "CHANGELOG.md": CHANGELOG_MD,
    "tests/functional/cases.py": FUNCTIONAL_PY,
    "spec/operation-vectors.json": VECTORS_JSON,
}


def without_repository_variables(environ):
    """The environment minus git's repository-local variables, which a hook
    would export and which would point git at the caller's repository."""
    listed = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True, check=True
    ).stdout.split()
    return {k: v for k, v in environ.items() if k not in listed}


GIT_ENV = {
    **without_repository_variables(os.environ),
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}


class TreeChecks(unittest.TestCase):
    """The tree checks, against a fixture whose code and docs agree."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for rel, text in TREE.items():
            self.write(rel, text)
        # The fixture's own allowlists: the real ones name the real docs' codes.
        for name, value in (
            ("RESERVED", {"uploads_not_configured"}),
            ("UNTESTED", {"uploads_not_configured"}),
        ):
            patcher = mock.patch.object(check_docs, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def edit(self, rel, old, new):
        text = (self.root / rel).read_text(encoding="utf-8")
        self.assertIn(old, text, f"fixture {rel} lacks {old!r}")
        self.write(rel, text.replace(old, new, 1))

    def problems(self):
        return check_docs.run_checks(self.root)

    def assertOnly(self, check, *fragments):
        """Exactly the named check fails, with each fragment in its output."""
        problems = self.problems()
        self.assertTrue(problems, "the broken input passed every check")
        self.assertEqual({name for name, _ in problems}, {check}, problems)
        text = "\n".join(problem for _, problem in problems)
        for fragment in fragments:
            self.assertIn(fragment, text)

    def test_fixture_passes(self):
        self.assertEqual(self.problems(), [])

    def test_script_exits_nonzero_and_names_the_check(self):
        self.edit("CHANGELOG.md", "## [Unreleased]\n\n", "")
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root)], capture_output=True, text=True
        )
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("check_docs: changelog:", done.stderr)

    # env

    def test_env_variable_read_but_undocumented(self):
        self.edit(
            "docs/configuration.md", "| `CRABCROP_WIDTH_STEP` | `100` | Width hint step. |\n", ""
        )
        self.assertOnly("env", "CRABCROP_WIDTH_STEP is read by the server but not documented")

    def test_env_variable_documented_but_never_read(self):
        self.edit(
            "docs/configuration.md", "| `PORT`", "| `CRABCROP_GONE` | unset | Removed. |\n| `PORT`"
        )
        self.assertOnly("env", "documents CRABCROP_GONE, which no server code reads")

    def test_env_variable_read_only_by_a_test_module_is_not_read(self):
        self.edit(
            "docs/configuration.md",
            "| `PORT`",
            "| `CRABCROP_ONLY_IN_A_TEST` | unset | No. |\n| `PORT`",
        )
        self.assertOnly("env", "documents CRABCROP_ONLY_IN_A_TEST, which no server code reads")

    def test_env_code_after_a_test_module_is_server_code(self):
        # The braces inside the module's strings, comment and char literal
        # must not end it early or late: CRABCROP_LATE, read after it, counts.
        self.edit("docs/configuration.md", "`CRABCROP_LATE` is read late.", "")
        self.assertOnly("env", "CRABCROP_LATE is read by the server but not documented")

    def test_env_test_variable_in_the_docs(self):
        self.edit(
            "README.md", "to size the cache.", "to size the cache; see `CRABCROP_TEST_FACE_MODEL`."
        )
        self.assertOnly("env", "README.md names the test-only variable CRABCROP_TEST_FACE_MODEL")

    def test_env_page_names_a_variable_nothing_reads(self):
        self.edit("README.md", "to size the cache.", "to size the cache, and `CRABCROP_CACHE_TTL`.")
        self.assertOnly(
            "env", "README.md names CRABCROP_CACHE_TTL, which no server or CLI code reads"
        )

    def test_env_cli_variable_missing_from_its_page(self):
        self.write("docs/endpoints.md", "# Endpoints\n")
        self.assertOnly(
            "env", "CRABCROP_BASE (read by the CLI) is not documented in docs/endpoints.md"
        )

    def test_env_cli_allowlist_entry_the_cli_no_longer_reads(self):
        self.write("src/cli.rs", "fn base() {}\n")
        self.assertOnly("env", "CLI_ENV lists CRABCROP_BASE, which src/cli.rs does not read")

    def test_env_other_variable_named_nowhere(self):
        self.edit("docs/configuration.md", "Credentials come from `AWS_ACCESS_KEY_ID`. ", "")
        self.assertOnly("env", "AWS_ACCESS_KEY_ID is read by the server but named nowhere")

    # defaults

    def test_defaults_literal_disagrees(self):
        self.edit("docs/configuration.md", "| `256` |", "| `512` |")
        self.assertOnly("defaults", "CRABCROP_CACHE_MAX_MB defaults to 256 in src/main.rs")

    def test_defaults_boolean_disagrees(self):
        self.edit(
            "docs/configuration.md",
            "| `CRABCROP_ALLOW_UNSAFE` | off |",
            "| `CRABCROP_ALLOW_UNSAFE` | on |",
        )
        self.assertOnly("defaults", "CRABCROP_ALLOW_UNSAFE defaults to off")

    def test_defaults_multiline_call_disagrees(self):
        self.edit(
            "docs/configuration.md",
            "| `CRABCROP_BLOCK_PRIVATE_SOURCES` | on |",
            "| `CRABCROP_BLOCK_PRIVATE_SOURCES` | off |",
        )
        self.assertOnly("defaults", "CRABCROP_BLOCK_PRIVATE_SOURCES defaults to on")

    def test_defaults_constant_in_the_same_file_disagrees(self):
        self.edit(
            "src/main.rs",
            "const DEFAULT_WIDTH_STEP: u32 = 100;",
            "const DEFAULT_WIDTH_STEP: u32 = 50;",
        )
        self.assertOnly("defaults", "CRABCROP_WIDTH_STEP defaults to 50")

    def test_defaults_constant_in_another_module_disagrees(self):
        self.edit("src/engine.rs", "= 9;", "= 8;")
        self.assertOnly("defaults", "CRABCROP_AVIF_SPEED defaults to 8")

    def test_defaults_computed_default_is_not_judged(self):
        self.edit("docs/configuration.md", "| 40 per core |", "| anything at all |")
        self.assertEqual(self.problems(), [])

    def test_defaults_finding_no_call_fails_rather_than_passing_empty(self):
        text = (self.root / "src/main.rs").read_text()
        self.write(
            "src/main.rs", text.replace("parse_env", "read_env").replace("parse_bool", "read_bool")
        )
        self.assertOnly("defaults", "found no parse_env or parse_bool call")

    # codes

    def test_codes_built_but_undocumented(self):
        self.edit(
            "src/server.rs",
            '"server_busy", busy)',
            '"server_busy", busy),\n        _ => (StatusCode::CONFLICT, "upload_exists", m)',
        )
        self.assertOnly("codes", "upload_exists is built by the server but has no row")

    def test_codes_documented_status_disagrees(self):
        self.edit("docs/responses.md", "| `decode_failed` | `502` |", "| `decode_failed` | `500` |")
        problems = self.problems()
        self.assertIn(("codes", "decode_failed is documented as 500 but built with 502"), problems)

    def test_codes_documented_but_never_built(self):
        self.edit(
            "src/server.rs", '(StatusCode::SERVICE_UNAVAILABLE, "server_busy", busy)', "busy()"
        )
        self.assertOnly("codes", "documents server_busy, which the server never builds")

    def test_codes_built_only_in_a_test_does_not_count(self):
        self.edit(
            "src/server.rs", '(StatusCode::SERVICE_UNAVAILABLE, "server_busy", busy)', "busy()"
        )
        self.write(
            "src/server/tests.rs",
            SERVER_TESTS_RS
            + 'const X: (StatusCode, &str) = (StatusCode::SERVICE_UNAVAILABLE, "server_busy");\n',
        )
        self.assertOnly("codes", "documents server_busy, which the server never builds")

    def test_codes_reserved_code_need_not_be_built(self):
        self.assertNotIn("uploads_not_configured", (self.root / "src/server.rs").read_text())
        self.assertEqual(self.problems(), [])

    def test_codes_reserved_entry_must_be_documented(self):
        with mock.patch.object(check_docs, "RESERVED", {"uploads_not_configured", "retired_code"}):
            self.assertOnly(
                "codes", "RESERVED lists retired_code, which docs/responses.md does not document"
            )

    def test_codes_built_with_two_statuses(self):
        self.edit(
            "src/server.rs",
            '(StatusCode::SERVICE_UNAVAILABLE, "server_busy", busy)',
            '(StatusCode::SERVICE_UNAVAILABLE, "server_busy", busy),\n        _ => (StatusCode::GATEWAY_TIMEOUT, "server_busy", m)',
        )
        self.assertOnly("codes", "server_busy is built with more than one status (503, 504)")

    def test_codes_unknown_status_constant(self):
        self.edit(
            "src/server.rs",
            'StatusCode::FORBIDDEN, "bad_signature"',
            'StatusCode::IM_A_TEAPOT, "bad_signature"',
        )
        problems = self.problems()
        self.assertIn(
            ("codes", "StatusCode::IM_A_TEAPOT is not in STATUS_NUMBERS: add its number"), problems
        )

    def test_codes_parser_code_missing_from_the_grammar_list(self):
        self.edit(
            "src/url.rs",
            '=> "invalid_size",',
            '=> "invalid_size",\n            ParseError::InvalidPad(_) => "invalid_pad",',
        )
        problems = self.problems()
        self.assertIn(
            (
                "codes",
                "parser code invalid_pad is missing from the grammar list in docs/responses.md",
            ),
            problems,
        )

    def test_codes_grammar_list_names_a_code_the_parser_never_returns(self):
        self.edit(
            "docs/responses.md",
            "`empty_path`, `invalid_size`",
            "`empty_path`, `invalid_size`, `invalid_gone`",
        )
        problems = self.problems()
        self.assertIn(
            (
                "codes",
                "docs/responses.md lists invalid_gone as a grammar failure, which the parser never returns",
            ),
            problems,
        )

    def test_codes_listed_twice(self):
        self.edit(
            "docs/responses.md",
            "| `server_busy` | `503` | A gate waited past its timeout. |\n",
            "| `server_busy` | `503` | A gate waited past its timeout. |\n| `server_busy` | `503` | Again. |\n",
        )
        self.assertOnly("codes", "documents server_busy twice")

    # statuses

    def test_statuses_table_misses_a_used_status(self):
        self.edit("docs/responses.md", "| `503` | Shed. |\n", "")
        self.assertOnly(
            "statuses", "status 503 is used by a code but missing from the statuses table"
        )

    def test_statuses_table_lists_an_unused_status(self):
        self.edit(
            "docs/responses.md", "| `503` | Shed. |\n", "| `503` | Shed. |\n| `418` | Teapot. |\n"
        )
        self.assertOnly("statuses", "the statuses table lists 418, which no documented code uses")

    # filters

    def test_filters_accepted_but_undocumented(self):
        self.edit(
            "src/url.rs",
            '("grayscale", None) => Filter::Grayscale,',
            '("grayscale", None) => Filter::Grayscale,\n            ("sepia", None) => Filter::Sepia,',
        )
        self.assertOnly("filters", "filter sepia is accepted by parse_filters but has no row")

    def test_filters_alternation_arm_counts(self):
        self.edit(
            "docs/filters.md",
            "| `pixelate_region` | `pixelate_region(x,y,w,h)` | Pixelate a box. | `filters:pixelate_region(1,2,3,4)` |\n",
            "",
        )
        self.assertOnly("filters", "filter pixelate_region is accepted by parse_filters")

    def test_filters_signature_line_documents_a_filter(self):
        self.edit("docs/filters.md", "filters:watermark(<url>", "watermark(<url>")
        self.assertOnly("filters", "filter watermark is accepted by parse_filters")

    def test_filters_row_for_a_filter_the_parser_rejects(self):
        self.edit(
            "docs/filters.md",
            "| `blur` | `blur(2)` | Blur. | `filters:blur(2)` |\n",
            "| `blur` | `blur(2)` | Blur. | `filters:blur(2)` |\n| `glow` | none | Glow. | `filters:glow` |\n",
        )
        self.assertOnly("filters", "documents filter glow, which parse_filters does not accept")

    def test_filters_directive_rows_count(self):
        self.edit(
            "docs/filters.md",
            "| `quality` | `quality(80)` | Quality. | `filters:quality(80)` |\n",
            "",
        )
        self.assertOnly("filters", "filter quality is accepted by parse_filters")

    # changelog

    def test_changelog_opens_with_a_release(self):
        self.edit("CHANGELOG.md", "## [Unreleased]\n\n", "")
        self.assertOnly("changelog", "opens with '## [0.1.0] - 2026-01-01'")

    def test_changelog_without_sections(self):
        self.write("CHANGELOG.md", "# Changelog\n")
        self.assertOnly("changelog", "has no ## [Unreleased] section")

    # tested

    def test_tested_documented_code_no_test_names(self):
        # The fixture names it only inside an escaped string, \"server_busy\".
        self.edit("src/server/tests.rs", "server_busy", "other")
        self.assertOnly("tested", "server_busy is documented but no test names it")

    def test_tested_an_identifier_is_not_a_name(self):
        # `let not_found = ...` in a test names a variable, not the code.
        self.edit(
            "docs/responses.md",
            "| `server_busy` |",
            "| `not_found` | `404` | Nothing here. |\n| `server_busy` |",
        )
        self.edit(
            "src/server.rs",
            "error_response(StatusCode::FORBIDDEN",
            'error_response(StatusCode::NOT_FOUND, "not_found", m, a);\n    error_response(StatusCode::FORBIDDEN',
        )
        self.assertOnly("tested", "not_found is documented but no test names it")

    def test_tested_functional_suite_counts(self):
        # Only the functional suite names empty_path.
        self.write("tests/functional/cases.py", "EXPECT = {}\n")
        self.assertOnly("tested", "empty_path is documented but no test names it")

    def test_tested_parser_corpus_counts(self):
        # Only the parser corpus a server test replays names invalid_size.
        self.write("spec/operation-vectors.json", "{}")
        self.assertOnly("tested", "invalid_size is documented but no test names it")

    def test_tested_allowlisted_code_a_test_now_names(self):
        self.write(
            "tests/functional/cases.py", FUNCTIONAL_PY + 'MORE = ["uploads_not_configured"]\n'
        )
        self.assertOnly(
            "tested", "uploads_not_configured is named in a test now: remove it from UNTESTED"
        )

    def test_tested_allowlist_entry_for_an_undocumented_code(self):
        with mock.patch.object(
            check_docs, "UNTESTED", {"uploads_not_configured", "never_documented"}
        ):
            self.assertOnly(
                "tested",
                "UNTESTED lists never_documented, which docs/responses.md does not document",
            )


class ChangelogEntry(unittest.TestCase):
    """--changelog-entry, against a throwaway history."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.git("init", "-q", "-b", "main")
        self.write(
            "CHANGELOG.md",
            "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- An old entry.\n\n## [0.1.0] - 2026-01-01\n\n- First.\n",
        )
        self.write("src/lib.rs", "pub fn f() {}\n")
        self.write("docs/guide.md", "# Guide\n")
        self.commit("base")
        self.base = self.git("rev-parse", "HEAD").strip()

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.run(
            ["git", *args], cwd=self.repo, env=GIT_ENV, check=True, capture_output=True, text=True
        ).stdout

    def write(self, rel, text):
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def entry(self, *labels):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(self.repo),
                "--changelog-entry",
                self.base,
                "--labels",
                json.dumps(list(labels)),
            ],
            env=GIT_ENV,
            capture_output=True,
            text=True,
        )

    def add_line(self, after, line):
        text = (self.repo / "CHANGELOG.md").read_text()
        self.assertIn(after, text)
        self.write("CHANGELOG.md", text.replace(after, after + line, 1))

    def test_code_change_without_an_entry_fails(self):
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.commit("change")
        done = self.entry()
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("touches src/lib.rs but adds no line under ## [Unreleased]", done.stderr)

    def test_the_no_changelog_label_clears_it(self):
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.commit("change")
        done = self.entry("dependencies", "no-changelog")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("no-changelog label is set", done.stdout)

    def test_another_label_does_not(self):
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.commit("change")
        self.assertEqual(self.entry("documentation").returncode, 1)

    def test_an_entry_under_unreleased_passes(self):
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.add_line("- An old entry.\n", "- The change (issue #1).\n")
        self.commit("change")
        done = self.entry()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("gains 1 line(s)", done.stdout)

    def test_an_entry_under_a_release_does_not_count(self):
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.add_line("- First.\n", "- Misfiled.\n")
        self.commit("change")
        self.assertEqual(self.entry().returncode, 1)

    def test_reordering_or_a_new_subheading_adds_nothing(self):
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.add_line("## [Unreleased]\n\n", "### Added\n\n")
        self.commit("change")
        self.assertEqual(self.entry().returncode, 1)

    def test_every_watched_tree_needs_an_entry(self):
        for rel in ("sdks/python/x.py", "frameworks/nuxt/x.ts", "plugins/wordpress/x.php"):
            with self.subTest(rel=rel):
                self.write(rel, f"// {rel}\n")
                self.commit(rel)
                done = self.entry()
                self.assertEqual(done.returncode, 1, done.stdout)
                self.assertIn(rel, done.stderr)
                self.git("reset", "-q", "--hard", self.base)

    def test_a_change_elsewhere_needs_no_entry(self):
        self.write("docs/guide.md", "# Guide\n\nMore.\n")
        self.write("scripts/tool.py", "print()\n")
        self.commit("docs")
        done = self.entry()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("touches none of", done.stdout)

    def test_commits_the_base_gained_since_do_not_count(self):
        # main moves on (a src/ change with its entry) after the branch
        # point; the branch itself only edits docs.
        self.git("checkout", "-q", "-b", "topic")
        self.write("docs/guide.md", "# Guide\n\nMore.\n")
        self.commit("docs")
        self.git("checkout", "-q", "main")
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.add_line("- An old entry.\n", "- Main's change.\n")
        self.commit("main moves")
        self.base = self.git("rev-parse", "HEAD").strip()
        self.git("checkout", "-q", "topic")
        done = self.entry()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("touches none of", done.stdout)

    def test_labels_must_be_a_json_list(self):
        done = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(self.repo),
                "--changelog-entry",
                self.base,
                "--labels",
                "no-changelog",
            ],
            env=GIT_ENV,
            capture_output=True,
            text=True,
        )
        self.assertEqual(done.returncode, 1)
        self.assertIn("--labels takes a JSON list", done.stderr)

    def test_an_unknown_base_fails(self):
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.repo), "--changelog-entry", "0" * 40],
            env=GIT_ENV,
            capture_output=True,
            text=True,
        )
        self.assertEqual(done.returncode, 1)
        self.assertIn("git merge-base", done.stderr)


if __name__ == "__main__":
    unittest.main()
