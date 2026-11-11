#!/usr/bin/env python3
"""Tests for scripts/check_docs.py.

Each case writes a small repository whose code and docs agree, breaks one
side, and expects the matching check to name the break; the pull-request
check runs against a throwaway git history. The changelog fragment tools
(scripts/assemble_changelog.py, issue #539) are tested here too, since CI
runs this file: the assembler against small trees, and fragments from two
branches merged in either order. Run directly:
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
ASSEMBLE = HERE / "assemble_changelog.py"

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
    let filters = split(rest).map(|(name, params)| filters::read_filter(name, params, seg));
    filters.collect::<Result<_, _>>().map(Operation::Filters)
}
"""

FILTERS_RS = r"""
//! - `"glow" => Glow;` in a doc comment is not a row.

macro_rules! filter_specs {
    ($($rows:tt)*) => {
        filter_specs!(@rows [] [] [] [] $($rows)*);
    };
}

filter_specs! {
    "grayscale" => Grayscale;
    "blur"(sigma: BLUR_SIGMA) => Blur(sigma);
    "blur_region"(x: REGION_ORIGIN, y: REGION_ORIGIN, w: REGION_EXTENT, h: REGION_EXTENT)
        => BlurRegion { x, y, w, h };
    "pixelate_region"(
        x: REGION_ORIGIN,
        y: REGION_ORIGIN,
        w: REGION_EXTENT,
        h: REGION_EXTENT
    ) => PixelateRegion { x, y, w, h };
    "watermark"(custom read_watermark)
        => Watermark { url, x, y, alpha } spelled watermark_args(url, *x, *y, *alpha);
    "quality"(custom |p| parse_quality(p).map(Filter::Quality))
        => Quality(quality) spelled quality_args(quality);
}

fn read_watermark(p: &str) -> Option<Filter> {
    let url = match p {
        "glow" => return None,
        url => url,
    };
    Some(Filter::Watermark { url: url.into() })
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
    "src/url/filters.rs": FILTERS_RS,
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

    def test_env_variable_read_only_by_a_cfg_all_test_module_is_not_read(self):
        # src/toolrun.rs gates its tests on #[cfg(all(test, unix))].
        self.write(
            "src/toolrun.rs",
            "#[cfg(all(test, unix))]\nmod tests {\n"
            '    const X: &str = "CRABCROP_ONLY_IN_A_UNIX_TEST";\n}\n',
        )
        self.edit(
            "docs/configuration.md",
            "| `PORT`",
            "| `CRABCROP_ONLY_IN_A_UNIX_TEST` | unset | No. |\n| `PORT`",
        )
        self.assertOnly("env", "documents CRABCROP_ONLY_IN_A_UNIX_TEST, which no server code reads")

    def test_env_variable_read_only_by_a_path_test_module_is_not_read(self):
        # main.rs is a crate root, so it loads its tests from src/main/tests.rs
        # through #[path]; a plain `mod tests;` there would be src/tests.rs.
        text = (self.root / "src/main.rs").read_text(encoding="utf-8")
        head, rest = text.split("#[cfg(test)]\nmod tests {\n", 1)
        body, tail = rest.split("\n}\n", 1)
        declaration = '#[cfg(test)]\n#[path = "main/tests.rs"]\nmod tests;\n'
        self.write("src/main.rs", head + declaration + tail)
        self.write("src/main/tests.rs", body + "\n")
        self.assertEqual(self.problems(), [])
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

    def test_env_variable_read_through_the_injected_lookup_is_read(self):
        # Issue #477: the boot configuration reads PORT through its `get`.
        self.edit("src/main.rs", 'env::var("PORT").ok()', 'get("PORT")')
        self.assertEqual(self.problems(), [])

    def test_env_map_or_header_lookup_is_not_a_variable(self):
        # `.get("PORT")` on a map is no read of the variable PORT.
        self.edit("src/main.rs", 'env::var("PORT").ok()', 'headers.get("PORT")')
        self.assertOnly("env", "documents PORT, which no server code reads")

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

    def test_codes_reserved_code_the_server_builds(self):
        self.edit(
            "src/server.rs",
            '(StatusCode::SERVICE_UNAVAILABLE, "server_busy", busy),',
            '(StatusCode::SERVICE_UNAVAILABLE, "server_busy", busy),\n'
            '        EngineError::Off => (StatusCode::NOT_FOUND, "uploads_not_configured", m),',
        )
        self.assertOnly("codes", "RESERVED lists uploads_not_configured, which the server builds")

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

    def test_codes_parser_code_the_table_documents_as_a_400(self):
        # A parser code the server also builds, as a 400, is documented by
        # its table row; the grammar list may name it too.
        self.edit("src/server.rs", "StatusCode::SERVICE_UNAVAILABLE", "StatusCode::BAD_REQUEST")
        self.edit("docs/responses.md", "| `server_busy` | `503` |", "| `server_busy` | `400` |")
        self.edit("docs/responses.md", "| `503` | Shed. |\n", "")
        self.edit(
            "src/url.rs",
            '=> "invalid_size",',
            '=> "invalid_size",\n            ParseError::TooMany(_) => "server_busy",',
        )
        self.assertEqual(self.problems(), [])
        self.edit(
            "docs/responses.md",
            "`empty_path`, `invalid_size`",
            "`empty_path`, `invalid_size`, `server_busy`",
        )
        self.assertEqual(self.problems(), [])

    def test_codes_parser_code_the_table_documents_with_another_status(self):
        self.edit(
            "src/url.rs",
            '=> "invalid_size",',
            '=> "invalid_size",\n            ParseError::TooMany(_) => "server_busy",',
        )
        self.edit(
            "docs/responses.md",
            "`empty_path`, `invalid_size`",
            "`empty_path`, `invalid_size`, `server_busy`",
        )
        self.assertOnly(
            "codes",
            "parser code server_busy answers 400, but docs/responses.md documents it as 503",
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
            "src/url/filters.rs",
            '"grayscale" => Grayscale;',
            '"grayscale" => Grayscale;\n    "sepia" => Sepia;',
        )
        self.assertOnly("filters", "filter sepia is accepted by parse_filters but has no row")

    def test_filters_shorthand_row_counts(self):
        self.edit(
            "src/url/filters.rs",
            '"grayscale" => Grayscale;',
            '"grayscale" => Grayscale;\n    shorthand "trim" => Trim(DEFAULT_TRIM_TOLERANCE);',
        )
        self.assertOnly("filters", "filter trim is accepted by parse_filters but has no row")

    def test_filters_row_written_over_several_lines_counts(self):
        self.edit(
            "docs/filters.md",
            "| `pixelate_region` | `pixelate_region(x,y,w,h)` | Pixelate a box. | `filters:pixelate_region(1,2,3,4)` |\n",
            "",
        )
        self.assertOnly("filters", "filter pixelate_region is accepted by parse_filters")

    def test_filters_signature_line_documents_a_filter(self):
        self.edit("docs/filters.md", "filters:watermark(<url>", "watermark(<url>")
        self.assertOnly("filters", "filter watermark is accepted by parse_filters")

    def test_filters_names_outside_the_table_do_not_count(self):
        # read_watermark's match arm and the module doc hold a quoted name
        # and an arrow, like a row, but sit outside the table.
        text = (self.root / "src/url/filters.rs").read_text(encoding="utf-8")
        self.assertIn('        "glow" => return None,', text)
        self.assertEqual(self.problems(), [])
        # The same line inside the table is a row.
        self.edit("src/url/filters.rs", '    "grayscale" =>', '    "glow" => Glow;\n    "grayscale" =>')
        self.assertOnly("filters", "filter glow is accepted by parse_filters but has no row")

    def test_filters_without_the_table(self):
        self.edit("src/url/filters.rs", "filter_specs! {", "filter_rows! {")
        self.assertOnly("filters", "src/url/filters.rs has no filter_specs! table")

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

    def test_changelog_fragments_that_assemble_pass(self):
        self.write("changelog.d/README.md", "# Changelog fragments\n")
        self.write("changelog.d/7.fixed.md", "A fix (issue #7).\n")
        self.write("changelog.d/7-again.fixed.md", "Another fix (issue #7).\n")
        self.assertEqual(self.problems(), [])

    def test_changelog_malformed_fragment(self):
        self.write("changelog.d/7.feature.md", "A thing (issue #7).\n")
        self.assertOnly("changelog", "changelog.d/7.feature.md", "unknown section 'feature'")

    def test_changelog_second_cache_epoch_note(self):
        self.write("changelog.d/7.changed.md", "This release starts a new cache epoch (issue #7).\n")
        self.write("changelog.d/9.changed.md", "This release starts a new cache epoch (issue #9).\n")
        self.assertOnly("changelog", "changelog.d/9.changed.md", "second cache-epoch note")

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

    # Fragments (issue #539): changelog.d/<issue>[-<slug>].<section>.md.

    def test_a_fragment_is_an_entry(self):
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.write("changelog.d/1.fixed.md", "The change (issue #1).\n")
        self.commit("change")
        done = self.entry()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("adds the fragment changelog.d/1.fixed.md", done.stdout)

    def test_a_fragment_with_a_slug_is_an_entry(self):
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.write("changelog.d/1-second-part.security.md", "The change (issues #1 and #2).\n")
        self.commit("change")
        done = self.entry()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("changelog.d/1-second-part.security.md", done.stdout)

    def test_a_wrapped_fragment_with_its_own_bullet_is_an_entry(self):
        # The assembler joins the lines and drops the "- ", so the gate
        # accepts what the assembler can write.
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.write("changelog.d/1.changed.md", "- The change, wrapped\n  over two lines (issues\n#2 and #1).\n\n\n")
        self.commit("change")
        done = self.entry()
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_a_byte_order_mark_and_a_bullet_line_are_dropped(self):
        # Some editors save a byte-order mark; some write the bullet on a
        # line of its own. Neither reaches the entry (see AssembleChangelog).
        for text in ("\ufeff- The change (issue #1).\n", "- \n  The change (issue #1).\n"):
            with self.subTest(text=text):
                self.write("src/lib.rs", "pub fn f() { g() }\n")
                self.write("changelog.d/1.fixed.md", text)
                self.commit("change")
                done = self.entry()
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertIn("adds the fragment changelog.d/1.fixed.md", done.stdout)
                self.git("reset", "-q", "--hard", self.base)

    def test_an_edited_fragment_is_an_entry(self):
        self.write("changelog.d/1.changed.md", "The change (issue #1).\n")
        self.commit("fragment on main")
        self.base = self.git("rev-parse", "HEAD").strip()
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.write("changelog.d/1.changed.md", "The change, and its follow-up (issue #1).\n")
        self.commit("follow-up")
        done = self.entry()
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_malformed_fragments_fail(self):
        cases = (
            ("changelog.d/fix-1.md", "The change (issue #1).\n", "is not named <issue>.<section>.md"),
            ("changelog.d/1.md", "The change (issue #1).\n", "is not named <issue>.<section>.md"),
            ("changelog.d/01.fixed.md", "The change (issue #1).\n", "is not named <issue>.<section>.md"),
            ("changelog.d/1_x.fixed.md", "The change (issue #1).\n", "is not named <issue>.<section>.md"),
            ("changelog.d/1-Big.fixed.md", "The change (issue #1).\n", "is not named <issue>.<section>.md"),
            ("changelog.d/1.fixed.txt", "The change (issue #1).\n", "is not named <issue>.<section>.md"),
            ("changelog.d/1.fixed.2.md", "The change (issue #1).\n", "is not named <issue>.<section>.md"),
            ("changelog.d/1.feature.md", "The change (issue #1).\n", "unknown section 'feature'"),
            ("changelog.d/1.Fixed.md", "The change (issue #1).\n", "unknown section 'Fixed'"),
            ("changelog.d/1.fixed.md", "The change.\n", "names no issue"),
            ("changelog.d/1.fixed.md", "The change (see #1).\n", "names no issue"),
            ("changelog.d/1.fixed.md", "The change (issue #2).\n", "names issue #2, not #1"),
            ("changelog.d/1.fixed.md", "The change (issues #2 and #3).\n", "names issues #2 and #3, not #1"),
            ("changelog.d/1.fixed.md", "\n \n", "is empty"),
            ("changelog.d/1.fixed.md", "", "is empty"),
            ("changelog.d/1.fixed.md", "One (issue #1).\n\nTwo.\n", "one paragraph"),
            ("changelog.d/1.fixed.md", "- One (issue #1).\n- Two (issue #1).\n", "starts another entry"),
            ("changelog.d/1.fixed.md", "-\n- One (issue #1).\n", "starts another entry"),
            ("changelog.d/1.fixed.md", "- - One (issue #1).\n", "starts another entry"),
            ("changelog.d/1.fixed.md", "Scaled by (1\n- Y)^2 (issue #1).\n", "rewrap it"),
            ("changelog.d/1.fixed.md", "- \n\n", "is empty"),
            ("changelog.d/1.fixed.md", "### Fixed\nOne (issue #1).\n", "a heading"),
            ("changelog.d/1.fixed.md", "One (issue #1),\n# wrapped.\n", "a heading"),
            ("changelog.d/1.fixed.md", b"\xff\xfe (issue #1)\n", "not UTF-8"),
            (
                "changelog.d/1.fixed.md",
                "This release starts a new cache epoch (issue #1).\n",
                "a cache-epoch note belongs under Changed",
            ),
            (
                "changelog.d/1.fixed.md",
                "\ufeffThis release starts a new cache epoch (issue #1).\n",
                "a cache-epoch note belongs under Changed",
            ),
            (
                "changelog.d/1.fixed.md",
                "this release starts a new cache epoch (issue #1).\n",
                "a cache-epoch note belongs under Changed",
            ),
            ("changelog.d/sub/1.fixed.md", "The change (issue #1).\n", "directly in changelog.d/"),
            ("1.fixed.md", "The change (issue #1).\n", "directly in changelog.d/"),
            ("docs/changelog.d/1.fixed.md", "The change (issue #1).\n", "directly in changelog.d/"),
        )
        for rel, text, message in cases:
            with self.subTest(rel=rel, text=text):
                self.write("src/lib.rs", "pub fn f() { g() }\n")
                path = self.repo / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(text if isinstance(text, bytes) else text.encode())
                self.commit("change")
                done = self.entry()
                self.assertEqual(done.returncode, 1, done.stdout)
                self.assertIn(rel, done.stderr)
                self.assertIn(message, done.stderr)
                self.git("reset", "-q", "--hard", self.base)
                self.git("clean", "-q", "-fdx")

    def test_a_malformed_fragment_fails_beside_a_changelog_line(self):
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.add_line("- An old entry.\n", "- The change (issue #1).\n")
        self.write("changelog.d/1.feature.md", "The change (issue #1).\n")
        self.commit("change")
        done = self.entry()
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("unknown section 'feature'", done.stderr)

    def test_a_malformed_fragment_fails_under_the_label_and_outside_the_watched_trees(self):
        # The label waives the entry, not the release: a fragment that is
        # there must still assemble.
        self.write("docs/guide.md", "# Guide\n\nMore.\n")
        self.write("changelog.d/1.fixed.md", "The change.\n")
        self.commit("docs")
        for labels in ((), ("no-changelog",)):
            with self.subTest(labels=labels):
                done = self.entry(*labels)
                self.assertEqual(done.returncode, 1, done.stdout)
                self.assertIn("changelog.d/1.fixed.md: names no issue", done.stderr)

    def test_the_fragment_readme_is_no_entry(self):
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        self.write("changelog.d/README.md", "# Changelog fragments\n\nSee CONTRIBUTING.md.\n")
        self.commit("change")
        done = self.entry()
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("adds no line under ## [Unreleased]", done.stderr)
        self.assertNotIn("README.md:", done.stderr)

    def test_a_deleted_fragment_is_no_entry(self):
        # A release deletes the fragments it assembled; that adds nothing.
        self.write("changelog.d/1.fixed.md", "The change (issue #1).\n")
        self.commit("fragment on main")
        self.base = self.git("rev-parse", "HEAD").strip()
        self.write("src/lib.rs", "pub fn f() { g() }\n")
        (self.repo / "changelog.d/1.fixed.md").unlink()
        self.commit("change")
        done = self.entry()
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("no changelog.d/ fragment", done.stderr)

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


class FragmentMerge(unittest.TestCase):
    """Two pull requests that each add a fragment merge in either order
    without a conflict, and the gate passes on each and on the merge."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.git("init", "-q", "-b", "main")
        self.write(
            "CHANGELOG.md",
            "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- A hand-written fix (issue #3).\n\n"
            "## [0.1.0] - 2026-01-01\n\n- First.\n",
        )
        self.write("changelog.d/README.md", "# Changelog fragments\n")
        self.write("src/a.rs", "pub fn a() {}\n")
        self.write("src/b.rs", "pub fn b() {}\n")
        self.commit("base")
        for branch, rs, fragment, text in (
            ("one", "src/a.rs", "12.fixed.md", "Fix twelve (issue #12)."),
            ("two", "src/b.rs", "9.fixed.md", "Fix nine (issue #9)."),
        ):
            self.git("checkout", "-q", "-b", branch, "main")
            self.write(rs, f"// {branch}\n")
            self.write(f"changelog.d/{fragment}", text + "\n")
            self.commit(branch)
        self.git("checkout", "-q", "main")

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args, check=True):
        return subprocess.run(
            ["git", *args], cwd=self.repo, env=GIT_ENV, check=check, capture_output=True, text=True
        )

    def write(self, rel, text):
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def gate(self, base):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.repo), "--changelog-entry", base],
            env=GIT_ENV,
            capture_output=True,
            text=True,
        )

    def test_either_merge_order(self):
        for branch in ("one", "two"):
            self.git("checkout", "-q", branch)
            done = self.gate("main")
            self.assertEqual(done.returncode, 0, done.stderr)
        trees = []
        for first, second in (("one", "two"), ("two", "one")):
            with self.subTest(order=(first, second)):
                self.git("checkout", "-q", "-B", f"after-{first}", "main")
                for branch in (first, second):
                    # The pull request's merge commit, as the workflow
                    # checks it out: HEAD^1 is the base it merges into.
                    merged = self.git("merge", "-q", "--no-ff", "-m", f"merge {branch}", branch, check=False)
                    self.assertEqual(merged.returncode, 0, merged.stdout + merged.stderr)
                    done = self.gate("HEAD^1")
                    self.assertEqual(done.returncode, 0, done.stderr)
                    self.assertIn("adds the fragment", done.stdout)
                trees.append(self.git("rev-parse", "HEAD^{tree}").stdout.strip())
                preview = subprocess.run(
                    [sys.executable, str(ASSEMBLE), "--root", str(self.repo), "--check"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(preview.returncode, 0, preview.stderr)
                self.assertEqual(
                    preview.stdout,
                    "## [Unreleased]\n\n### Fixed\n\n- A hand-written fix (issue #3).\n\n"
                    "- Fix nine (issue #9).\n\n- Fix twelve (issue #12).\n",
                )
        self.assertEqual(trees[0], trees[1], "the merge order changed the merged tree")


ASSEMBLY_CHANGELOG = """# Changelog

Notes on the format.

## [Unreleased]

### Added

- A hand-written addition (issue #20).

### Fixed

- A hand-written fix (issue #30).

- Another hand-written fix (issue #3).

## [0.1.0] - 2026-01-01

### Fixed

- First.
"""

ASSEMBLY_FRAGMENTS = {
    "40.fixed.md": "A later fix (issue #40).\n",
    "5.fixed.md": "An earlier fix (issue\n#5).\n",
    "5-again.fixed.md": "- A second fix for the same issue (issue #5).\n\n",
    "7.added.md": "A new thing (issues #7 and #8).\n",
    "9.changed.md": "A change (issue #9).",
    "11.security.md": "A security fix (issue #11).\r\n",
    "README.md": "# Changelog fragments\n\nNot an entry (issue #1).\n",
}

ASSEMBLED = """### Added

- A hand-written addition (issue #20).

- A new thing (issues #7 and #8).

### Changed

- A change (issue #9).

### Fixed

- A hand-written fix (issue #30).

- Another hand-written fix (issue #3).

- An earlier fix (issue #5).

- A second fix for the same issue (issue #5).

- A later fix (issue #40).

### Security

- A security fix (issue #11).
"""


class AssembleChangelog(unittest.TestCase):
    """scripts/assemble_changelog.py against a small tree."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.write("CHANGELOG.md", ASSEMBLY_CHANGELOG)
        for name, text in ASSEMBLY_FRAGMENTS.items():
            self.write(f"changelog.d/{name}", text)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rel, text, root=None):
        path = (root or self.root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)

    def run_script(self, *args, root=None):
        return subprocess.run(
            [sys.executable, str(ASSEMBLE), "--root", str(root or self.root), *args],
            capture_output=True,
            text=True,
        )

    def snapshot(self, root=None):
        base = root or self.root
        return {str(p.relative_to(base)): p.read_bytes() for p in sorted(base.rglob("*")) if p.is_file()}

    def changelog(self):
        return (self.root / "CHANGELOG.md").read_text(encoding="utf-8")

    def test_check_previews_unreleased_and_writes_nothing(self):
        before = self.snapshot()
        done = self.run_script("--check")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout, "## [Unreleased]\n\n" + ASSEMBLED)
        self.assertIn("6 fragment(s)", done.stderr)
        self.assertEqual(self.snapshot(), before)

    def test_a_release_writes_the_section_and_deletes_the_fragments(self):
        done = self.run_script("0.2.0", "--date", "2026-02-03")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(
            self.changelog(),
            ASSEMBLY_CHANGELOG.replace(
                ASSEMBLY_CHANGELOG[ASSEMBLY_CHANGELOG.index("## [Unreleased]") : ASSEMBLY_CHANGELOG.index("## [0.1.0]")],
                "## [Unreleased]\n\n## [0.2.0] - 2026-02-03\n\n" + ASSEMBLED + "\n",
            ),
        )
        self.assertEqual(sorted(p.name for p in (self.root / "changelog.d").iterdir()), ["README.md"])
        for name in ASSEMBLY_FRAGMENTS:
            if name != "README.md":
                self.assertIn(f"changelog.d/{name}", done.stdout)

    def test_the_same_tree_assembles_to_the_same_bytes(self):
        copy = tempfile.TemporaryDirectory()
        self.addCleanup(copy.cleanup)
        other = Path(copy.name)
        for rel, data in self.snapshot().items():
            (other / rel).parent.mkdir(parents=True, exist_ok=True)
            (other / rel).write_bytes(data)
        previews = [self.run_script("--check").stdout, self.run_script("--check", root=other).stdout]
        self.assertEqual(previews[0], previews[1])
        for root in (self.root, other):
            done = self.run_script("0.2.0", "--date", "2026-02-03", root=root)
            self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self.snapshot(), self.snapshot(other))
        # A second run of the same release finds nothing left to move.
        released = self.snapshot()
        done = self.run_script("0.2.0", "--date", "2026-02-03")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("already has ## [0.2.0]", done.stdout)
        self.assertEqual(self.snapshot(), released)

    def test_without_fragments_the_section_is_moved_as_written(self):
        for name in ASSEMBLY_FRAGMENTS:
            if name != "README.md":
                (self.root / "changelog.d" / name).unlink()
        done = self.run_script("--check")
        self.assertEqual(done.returncode, 0, done.stderr)
        start, end = ASSEMBLY_CHANGELOG.index("## [Unreleased]"), ASSEMBLY_CHANGELOG.index("## [0.1.0]")
        self.assertEqual(done.stdout, ASSEMBLY_CHANGELOG[start:end].rstrip("\n") + "\n")
        done = self.run_script("0.2.0", "--date", "2026-02-03")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(
            self.changelog(),
            ASSEMBLY_CHANGELOG.replace("## [Unreleased]\n", "## [Unreleased]\n\n## [0.2.0] - 2026-02-03\n", 1),
        )

    def test_a_missing_heading_is_created_in_changelog_order(self):
        self.write("CHANGELOG.md", "# Changelog\n\n## [Unreleased]\n\n## [0.1.0] - 2026-01-01\n\n- First.\n")
        for name in ASSEMBLY_FRAGMENTS:
            if name != "README.md":
                (self.root / "changelog.d" / name).unlink()
        self.write("changelog.d/4.removed.md", "A removal (issue #4).\n")
        self.write("changelog.d/2.deprecated.md", "A deprecation (issue #2).\n")
        done = self.run_script("--check")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(
            done.stdout,
            "## [Unreleased]\n\n### Deprecated\n\n- A deprecation (issue #2).\n\n"
            "### Removed\n\n- A removal (issue #4).\n",
        )
        done = self.run_script("0.2.0", "--date", "2026-02-03")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(
            self.changelog(),
            "# Changelog\n\n## [Unreleased]\n\n## [0.2.0] - 2026-02-03\n\n### Deprecated\n\n"
            "- A deprecation (issue #2).\n\n### Removed\n\n- A removal (issue #4).\n\n"
            "## [0.1.0] - 2026-01-01\n\n- First.\n",
        )

    def test_unreleased_as_the_last_section(self):
        self.write("CHANGELOG.md", "# Changelog\n\n## [Unreleased]\n")
        done = self.run_script("0.1.0", "--date", "2026-01-01")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(
            self.changelog(),
            "# Changelog\n\n## [Unreleased]\n\n## [0.1.0] - 2026-01-01\n\n" + ASSEMBLED.replace(
                "### Added\n\n- A hand-written addition (issue #20).\n\n", "### Added\n\n"
            ).replace("- A hand-written fix (issue #30).\n\n- Another hand-written fix (issue #3).\n\n", ""),
        )

    def test_a_cache_epoch_fragment_leads_changed(self):
        self.write(
            "CHANGELOG.md",
            ASSEMBLY_CHANGELOG.replace("### Fixed\n", "### Changed\n\n- A hand-written change (issue #2).\n\n### Fixed\n", 1),
        )
        self.write(
            "changelog.d/6.changed.md",
            "This release starts a new cache epoch. Changes below alter the bytes (issue #6).\n",
        )
        done = self.run_script("--check")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(
            "### Changed\n\n- This release starts a new cache epoch. Changes below alter the bytes (issue #6).\n\n"
            "- A hand-written change (issue #2).\n\n- A change (issue #9).\n\n### Fixed",
            done.stdout,
        )

    def test_a_second_cache_epoch_note_is_refused(self):
        # The note is known by its opening words in any letter case, and a
        # byte-order mark in front of them hides nothing.
        for hand, fragment in (
            ("This release starts a new cache epoch", "This release starts a new cache epoch"),
            ("this release starts a new cache epoch", "This release starts a new cache epoch"),
            ("This release starts a new cache epoch", "\ufeffThis release starts a new cache epoch"),
            ("This release starts a new cache epoch", "\ufeff- this Release Starts a New Cache Epoch"),
        ):
            self.write(
                "CHANGELOG.md",
                ASSEMBLY_CHANGELOG.replace(
                    "### Fixed\n", f"### Changed\n\n- {hand} (issue #2).\n\n### Fixed\n", 1
                ),
            )
            self.write("changelog.d/6.changed.md", f"{fragment} (issue #6).\n")
            before = self.snapshot()
            for args in (("--check",), ("0.2.0", "--date", "2026-02-03")):
                with self.subTest(hand=hand, fragment=fragment, args=args):
                    done = self.run_script(*args)
                    self.assertEqual(done.returncode, 1, done.stdout)
                    self.assertIn("changelog.d/6.changed.md: a second cache-epoch note", done.stderr)
                    self.assertEqual(self.snapshot(), before)

    def test_a_byte_order_mark_and_a_bullet_line_are_dropped(self):
        for name in ASSEMBLY_FRAGMENTS:
            if name != "README.md":
                (self.root / "changelog.d" / name).unlink()
        self.write("changelog.d/12.fixed.md", "\ufeff- Fix twelve (issue #12).\n")
        self.write("changelog.d/13.fixed.md", "-\nFix thirteen (issue #13).\n")
        self.write("changelog.d/14.fixed.md", "\ufeff* \r\n  Fix fourteen,\r\n  wrapped (issue #14).\r\n")
        self.write("changelog.d/15.changed.md", "\ufeffthis release starts a new cache epoch (issue #15).\n")
        done = self.run_script("--check")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("\ufeff", done.stdout)
        self.assertIn(
            "### Changed\n\n- this release starts a new cache epoch (issue #15).\n\n### Fixed\n\n"
            "- A hand-written fix (issue #30).\n\n- Another hand-written fix (issue #3).\n\n"
            "- Fix twelve (issue #12).\n\n- Fix thirteen (issue #13).\n\n- Fix fourteen, wrapped (issue #14).\n",
            done.stdout,
        )

    def test_a_malformed_fragment_stops_both_modes_and_writes_nothing(self):
        self.write("changelog.d/8.feature.md", "A thing (issue #8).\n")
        self.write("changelog.d/9.fixed.md", "A thing (issue #10).\n")
        before = self.snapshot()
        for args in (("--check",), ("0.2.0", "--date", "2026-02-03")):
            with self.subTest(args=args):
                done = self.run_script(*args)
                self.assertEqual(done.returncode, 1, done.stdout)
                self.assertIn("changelog.d/8.feature.md: unknown section 'feature'", done.stderr)
                self.assertIn("changelog.d/9.fixed.md: names issue #10, not #9", done.stderr)
                self.assertEqual(self.snapshot(), before)

    def test_a_release_refuses_bad_input(self):
        before = self.snapshot()
        for args, message in (
            (("0.2",), "X.Y.Z"),
            (("v0.2.0",), "X.Y.Z"),
            (("0.2.0", "--date", "2026-02-30"), "YYYY-MM-DD"),
            (("0.1.0", "--date", "2026-02-03"), "already has ## [0.1.0]"),
            ((), "--check or a version"),
        ):
            with self.subTest(args=args):
                done = self.run_script(*args)
                self.assertNotEqual(done.returncode, 0, done.stdout)
                self.assertIn(message, done.stderr)
                self.assertEqual(self.snapshot(), before)

    def test_a_changelog_without_unreleased_is_refused(self):
        self.write("CHANGELOG.md", "# Changelog\n\n## [0.1.0] - 2026-01-01\n\n- First.\n")
        done = self.run_script("--check")
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("no ## [Unreleased] section", done.stderr)


if __name__ == "__main__":
    unittest.main()
