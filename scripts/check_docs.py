#!/usr/bin/env python3
"""Fail when the reference docs or the CHANGELOG fall behind the code.

Nothing else ties docs/ to src/, so each evaluation used to find new drift and
the CHANGELOG was caught up at release time (issue #473). The checks read the
code and the docs as text, with the standard library only. Server code means
src/ outside its #[cfg(test)] modules; test code is those modules plus tests/.

  env        Every CRABCROP_* variable server code reads is in
             docs/configuration.md, and every one documented there is read.
             Test variables (CRABCROP_TEST_*) stay out of the docs, and the
             operator CLI's variables (CLI_ENV) are documented on their own
             page. The other variables server code reads by name (PORT, AWS_*,
             ORT_DYLIB_PATH) are named somewhere in README.md or docs/, and
             every CRABCROP_* name those pages mention is read.
  defaults   Where parse_env, parse_env_value, parse_bool or parse_bool_value
             takes a literal default, or a constant defined as one, the
             configuration table's Default column says the same.
  codes      Every (StatusCode, "code") pair server code builds has a row in
             docs/responses.md with that status, and every row's code is
             built unless it is RESERVED. The parser's codes
             (ParseError::code in src/url.rs) are the documented grammar list,
             except one the code table already documents as a 400.
  statuses   The statuses table lists exactly the statuses the codes use.
  filters    Every filter parse_filters accepts has a row in a Filter or
             Directive table of docs/filters.md, or a `filters:<name>(`
             signature line, and every such row or line names one it accepts.
  changelog  CHANGELOG.md's first section is ## [Unreleased].
  tested     Every documented code is named, quoted, in test code or test
             data, except the codes in UNTESTED. That allowlist only shrinks: a
             listed code that a test names fails until its entry is removed.

Usage:
  scripts/check_docs.py [--root DIR]
  scripts/check_docs.py --changelog-entry BASE [--labels JSON] [--root DIR]

The second form checks one pull request instead, from git: when the diff from
the merge base of BASE and HEAD to HEAD touches src/, sdks/, frameworks/ or
plugins/, CHANGELOG.md must
gain a line under ## [Unreleased], unless the labels (a JSON list of names,
as .github/workflows/changelog.yml passes them) include no-changelog.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGURATION = "docs/configuration.md"
RESPONSES = "docs/responses.md"
FILTERS = "docs/filters.md"
CHANGELOG = "CHANGELOG.md"
UNRELEASED = "## [Unreleased]"

# Variables only the test suites read: they point the ignored real-model tests
# at their files and never configure a server.
TEST_ENV_PREFIX = "CRABCROP_TEST_"

# Variables the operator CLI reads (src/cli.rs), with the page that documents
# them. CRABCROP_SECRET is shared with the server, so it is in the table.
CLI_ENV = {"CRABCROP_BASE": "docs/endpoints.md"}

# Documented codes the server need not build. Codes are frozen as published,
# so a retired code keeps its row, marked reserved, instead of disappearing.
RESERVED = set()

# Documented codes no test names yet. This list only shrinks: write the test
# rather than add an entry. A listed code that a test names fails the check
# until its entry is removed.
UNTESTED = {
    "invalid_path",
    "invalid_purge_group",
    "invalid_upload_name",
    "meta_not_available",
    "processing_failed",
    "reserved_segment",
    "upload_media_mismatch",
    "upload_store_failed",
    "upload_store_unavailable",
    "validation_failed",
}

# Test code and data besides the #[cfg(test)] modules of src/: the integration
# and functional suites, and the parser corpus a server test replays.
TEST_TREES = ("tests",)
TEST_DATA = ("spec/operation-vectors.json",)
TEST_SUFFIXES = {".rs", ".py", ".mjs", ".js", ".ts", ".json", ".sh"}

# A change under these paths needs a CHANGELOG line (--changelog-entry).
ENTRY_PATHS = ("src/", "sdks/", "frameworks/", "plugins/")
NO_CHANGELOG_LABEL = "no-changelog"

STATUS_NUMBERS = {
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "PAYMENT_REQUIRED": 402,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "METHOD_NOT_ALLOWED": 405,
    "NOT_ACCEPTABLE": 406,
    "REQUEST_TIMEOUT": 408,
    "CONFLICT": 409,
    "GONE": 410,
    "LENGTH_REQUIRED": 411,
    "PRECONDITION_FAILED": 412,
    "PAYLOAD_TOO_LARGE": 413,
    "URI_TOO_LONG": 414,
    "UNSUPPORTED_MEDIA_TYPE": 415,
    "RANGE_NOT_SATISFIABLE": 416,
    "UNPROCESSABLE_ENTITY": 422,
    "TOO_MANY_REQUESTS": 429,
    "REQUEST_HEADER_FIELDS_TOO_LARGE": 431,
    "INTERNAL_SERVER_ERROR": 500,
    "NOT_IMPLEMENTED": 501,
    "BAD_GATEWAY": 502,
    "SERVICE_UNAVAILABLE": 503,
    "GATEWAY_TIMEOUT": 504,
    "INSUFFICIENT_STORAGE": 507,
}

ENV_LITERAL = re.compile(r'"(CRABCROP_[A-Z0-9_]+)"')
AWS_LITERAL = re.compile(r'"(AWS_[A-Z0-9_]+)"')
ENV_CALL = re.compile(r'env::var(?:_os)?\(\s*"([A-Z][A-Z0-9_]*)"\s*\)')
ENV_TOKEN = re.compile(r"\bCRABCROP_[A-Z0-9_]*[A-Z0-9]\b")
STATUS_PAIR = re.compile(r'StatusCode::([A-Z_]+)\s*,\s*"([a-z][a-z0-9_]*)"')
TICKED = re.compile(r"`([^`]+)`")
# `#[cfg(test)]`, or `#[cfg(all(test, ...))]` as src/toolrun.rs gates its tests.
CFG_TEST_MOD = re.compile(
    r"#\[cfg\((?:test|all\(\s*test\s*,[^\]]*\))\)\]\s*(?:#\[[^\]]*\]\s*)*"
    r"(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*([{;])"
)

# A default the check can judge is a literal or a constant's path; anything
# else (a computed budget, a variable) is left alone.
DEFAULT_ARG = r"([A-Za-z0-9_:.]+)"
VALUE_ARG = r"(?:[^,()]|\([^()]*\))+"
DEFAULT_CALLS = tuple(
    re.compile(pattern)
    for pattern in (
        r'\bparse_(?:env|bool)\(\s*"([A-Z0-9_]+)"\s*,\s*' + DEFAULT_ARG + r"\s*,?\s*\)",
        r'\bparse_(?:env|bool)_value\(\s*"([A-Z0-9_]+)"\s*,'
        + VALUE_ARG
        + r",\s*"
        + DEFAULT_ARG
        + r"\s*,?\s*\)",
    )
)


class CheckError(Exception):
    """A state the check cannot judge: the gate fails with this message."""


# ---------------------------------------------------------------- reading


def read(root, rel):
    try:
        return (Path(root) / rel).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CheckError(f"{rel} is missing") from None


def matching_brace(text, start):
    """Index of the brace closing the one at `start`, skipping Rust literals.

    Strings, raw strings, character literals and comments may hold braces,
    so a plain count would end a module or function early or late.
    """
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end + 1
            continue
        if text.startswith("/*", i):
            level = 0
            while i < n:
                if text.startswith("/*", i):
                    level += 1
                    i += 2
                elif text.startswith("*/", i):
                    level -= 1
                    i += 2
                    if level == 0:
                        break
                else:
                    i += 1
            continue
        if c in "rbc" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            raw = re.match(r'(?:br|cr|r)(#*)"', text[i:])
            if raw:
                close = '"' + raw.group(1)
                end = text.find(close, i + raw.end())
                i = n if end < 0 else end + len(close)
                continue
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
            continue
        if c == "'":
            if i + 1 < n and text[i + 1] == "\\":
                end = text.find("'", i + 3)
                i = n if end < 0 else end + 1
            elif i + 2 < n and text[i + 2] == "'":
                i += 3
            else:
                i += 1  # a lifetime or a loop label
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise CheckError("unbalanced braces after offset %d" % start)


def split_tests(text):
    """(code, tests, external): the text outside #[cfg(test)] modules, the
    inline modules' text, and the names of `#[cfg(test)] mod name;` files."""
    code, tests, external = [], [], []
    pos = 0
    for m in CFG_TEST_MOD.finditer(text):
        if m.start() < pos:
            continue
        if m.group(2) == ";":
            external.append(m.group(1))
            continue
        end = matching_brace(text, m.end() - 1)
        code.append(text[pos : m.start()])
        tests.append(text[m.start() : end + 1])
        pos = end + 1
    code.append(text[pos:])
    return "".join(code), "".join(tests), external


def rust_sources(root):
    """{relative path: (server code, test code)} for every .rs file in src/.

    A file that a `#[cfg(test)] mod name;` declaration loads is all test.
    """
    split = {}
    test_files = set()
    for path in sorted((Path(root) / "src").rglob("*.rs")):
        rel = path.relative_to(root).as_posix()
        try:
            code, tests, external = split_tests(path.read_text(encoding="utf-8"))
        except CheckError as e:
            raise CheckError(f"{rel}: {e}") from None
        split[path] = (code, tests)
        home = path.parent if path.stem in ("main", "lib", "mod") else path.with_suffix("")
        for name in external:
            test_files.add(home / f"{name}.rs")
            test_files.update((home / name).rglob("*.rs"))
    if not split:
        raise CheckError("src/ holds no Rust sources")
    return {
        path.relative_to(root).as_posix(): (
            ("", code + tests) if path in test_files else (code, tests)
        )
        for path, (code, tests) in split.items()
    }


def server_code(sources):
    return "\n".join(code for code, _ in sources.values())


def test_text(root, sources):
    parts = [tests for _, tests in sources.values()]
    for tree in TEST_TREES:
        for path in sorted((Path(root) / tree).rglob("*")):
            if path.is_file() and path.suffix in TEST_SUFFIXES and "node_modules" not in path.parts:
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
    for rel in TEST_DATA:
        if (Path(root) / rel).is_file():
            parts.append(read(root, rel))
    return "\n".join(parts)


def split_row(line):
    line = line.strip()
    if not line.startswith("|"):
        return None
    return [cell.strip() for cell in line.strip("|").split("|")]


def table_rows(text, first_headers):
    """The data rows of every Markdown table whose first header cell is one
    of `first_headers`, each a list of stripped cells."""
    rows = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        cells = split_row(lines[i])
        if (
            cells
            and cells[0] in first_headers
            and i + 1 < len(lines)
            and re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", lines[i + 1].strip())
        ):
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            continue
        i += 1
    return rows


def reference_pages(root):
    """{relative path: text} for README.md and every page in docs/."""
    paths = [Path(root) / "README.md", *sorted((Path(root) / "docs").glob("*.md"))]
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in paths if p.is_file()
    }


# ---------------------------------------------------------------- env


def documented_env(root):
    """{name: Default cell} for the configuration table's variables, plus
    every other CRABCROP_* name the page mentions (cell None)."""
    text = read(root, CONFIGURATION)
    names = {}
    for cells in table_rows(text, {"Variable"}):
        for name in TICKED.findall(cells[0]):
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                names[name] = cells[1] if len(cells) > 1 else ""
    if not names:
        raise CheckError(f"{CONFIGURATION} has no Variable table to read")
    for name in ENV_TOKEN.findall(text):
        names.setdefault(name, None)
    return names


def read_env(sources):
    """(CRABCROP_* names, other names) that server code reads."""
    code = server_code(sources)
    own = set(ENV_LITERAL.findall(code))
    if not own:
        raise CheckError("found no CRABCROP_* name in server code")
    other = set(AWS_LITERAL.findall(code))
    other.update(n for n in ENV_CALL.findall(code) if not n.startswith("CRABCROP_"))
    return own, other


def check_env(root, sources):
    problems = []
    documented = documented_env(root)
    own, other = read_env(sources)
    pages = reference_pages(root)
    cli_code = sources.get("src/cli.rs", ("", ""))[0]
    for name in sorted(own):
        if name.startswith(TEST_ENV_PREFIX) or name in documented:
            continue
        if name in CLI_ENV:
            if not re.search(rf"\b{name}\b", pages.get(CLI_ENV[name], "")):
                problems.append(f"{name} (read by the CLI) is not documented in {CLI_ENV[name]}")
            continue
        problems.append(f"{name} is read by the server but not documented in {CONFIGURATION}")
    for name in sorted(CLI_ENV):
        if f'"{name}"' not in cli_code:
            problems.append(f"CLI_ENV lists {name}, which src/cli.rs does not read: remove it")
    everything = "\n".join(pages.values())
    for name in sorted(other - set(documented)):
        if not re.search(rf"(?<![A-Z0-9_]){name}(?![A-Z0-9_])", everything):
            problems.append(f"{name} is read by the server but named nowhere in README.md or docs/")
    for name in sorted(documented):
        if name not in own and name not in other and not name.startswith(TEST_ENV_PREFIX):
            problems.append(f"{CONFIGURATION} documents {name}, which no server code reads")
    for page, text in pages.items():
        for name in sorted(set(ENV_TOKEN.findall(text))):
            if name.startswith(TEST_ENV_PREFIX):
                problems.append(f"{page} names the test-only variable {name}")
            elif name not in own and page != CONFIGURATION:
                problems.append(f"{page} names {name}, which no server or CLI code reads")
    return problems


# ---------------------------------------------------------------- defaults


def literal_value(token):
    """A Rust literal's value (bool, int or float), or None."""
    if token in ("true", "false"):
        return token == "true"
    plain = token.replace("_", "")
    m = re.fullmatch(r"(\d+)(?:[ui](?:8|16|32|64|128|size))?", plain)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(\d+\.\d*)(?:f32|f64)?|(\d+)(?:f32|f64)", plain)
    if m:
        return float(m.group(1) or m.group(2))
    return None


def resolve_default(sources, rel, token):
    """The value of a default argument, or None when it is computed.

    A path names a constant in its module's files (src/a/b.rs and the files
    under src/a/b/, which it may re-export); a bare name, one in `rel`.
    """
    value = literal_value(token)
    if value is not None:
        return value
    m = re.fullmatch(r"(?:crate::)?((?:[a-z_][a-z0-9_]*::)*)([A-Z][A-Z0-9_]*)", token)
    if not m:
        return None
    modules = [part for part in m.group(1).split("::") if part]
    if modules:
        base = "src/" + "/".join(modules)
        files = [f for f in sources if f == base + ".rs" or f.startswith(base + "/")]
    else:
        files = [rel]
    const = re.compile(rf"\bconst\s+{m.group(2)}\s*:\s*\w+\s*=\s*([\w.]+)\s*;")
    values = {literal_value(found) for f in files for found in const.findall(sources[f][0])}
    return values.pop() if len(values) == 1 else None


def defaults_in_code(sources):
    """{name: {(value, file)}} for every parse call with a judgeable default."""
    found = {}
    for rel, (code, _) in sources.items():
        for pattern in DEFAULT_CALLS:
            for name, token in pattern.findall(code):
                value = resolve_default(sources, rel, token)
                if value is not None:
                    found.setdefault(name, set()).add((value, rel))
    if not found:
        raise CheckError("found no parse_env or parse_bool call with a literal default")
    return found


def doc_default_matches(cell, value):
    if isinstance(value, bool):
        words = cell.replace("`", "").split()
        return bool(words) and words[0].lower() == ("on" if value else "off")
    for token in TICKED.findall(cell):
        try:
            if float(token) == float(value):
                return True
        except ValueError:
            continue
    return False


def describe(value):
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def check_defaults(root, sources):
    problems = []
    documented = documented_env(root)
    for name, found in sorted(defaults_in_code(sources).items()):
        cell = documented.get(name)
        if cell is None:
            continue  # undocumented: the env check reports it
        for value, rel in sorted(found, key=str):
            if not doc_default_matches(cell, value):
                problems.append(
                    f"{name} defaults to {describe(value)} in {rel}, "
                    f"but {CONFIGURATION} gives {cell!r}"
                )
    return problems


# ---------------------------------------------------------------- codes


def built_codes(sources):
    """({code: {status}}, unknown status constants) for server code."""
    pairs, unknown = {}, set()
    for m in STATUS_PAIR.finditer(server_code(sources)):
        status = STATUS_NUMBERS.get(m.group(1))
        if status is None:
            unknown.add(m.group(1))
        else:
            pairs.setdefault(m.group(2), set()).add(status)
    if not pairs:
        raise CheckError('found no (StatusCode::..., "code") pair in server code')
    return pairs, unknown


def parser_codes(root):
    text = read(root, "src/url.rs")
    m = re.search(r"\bfn code\(&self\)\s*->\s*&'static str\s*\{", text)
    if not m:
        raise CheckError("src/url.rs has no ParseError::code to read the parser codes from")
    codes = set(
        re.findall(r'=>\s*"([a-z][a-z0-9_]*)"', text[m.end() : matching_brace(text, m.end() - 1)])
    )
    if not codes:
        raise CheckError("ParseError::code in src/url.rs returns no code this check can read")
    return codes


def documented_codes(root):
    """(code table {code: status}, codes listed twice, grammar list, statuses)."""
    text = read(root, RESPONSES)
    rows, twice = {}, set()
    for cells in table_rows(text, {"Code"}):
        code = TICKED.findall(cells[0])
        status = TICKED.findall(cells[1]) if len(cells) > 1 else []
        if len(code) != 1 or len(status) != 1 or not status[0].isdigit():
            raise CheckError(f"{RESPONSES}: cannot read the code row {' | '.join(cells)!r}")
        if code[0] in rows:
            twice.add(code[0])
        rows[code[0]] = int(status[0])
    if not rows:
        raise CheckError(f"{RESPONSES} has no Code table to read")
    grammar = None
    for paragraph in text.split("\n\n"):
        if paragraph.startswith("Grammar failures"):
            grammar = {t for t in TICKED.findall(paragraph) if re.fullmatch(r"[a-z][a-z0-9_]*", t)}
    if not grammar:
        raise CheckError(f"{RESPONSES} has no 'Grammar failures' paragraph naming the parser codes")
    statuses = set()
    for cells in table_rows(text, {"Status"}):
        ticked = TICKED.findall(cells[0])
        if len(ticked) != 1 or not ticked[0].isdigit():
            raise CheckError(f"{RESPONSES}: cannot read the status row {' | '.join(cells)!r}")
        statuses.add(int(ticked[0]))
    if not statuses:
        raise CheckError(f"{RESPONSES} has no Status table to read")
    return rows, twice, grammar, statuses


def check_codes(root, sources):
    problems = []
    built, unknown = built_codes(sources)
    rows, twice, grammar, _ = documented_codes(root)
    parser = parser_codes(root)
    for name in sorted(unknown):
        problems.append(f"StatusCode::{name} is not in STATUS_NUMBERS: add its number")
    for code in sorted(twice):
        problems.append(f"{RESPONSES} documents {code} twice")
    for code, statuses in sorted(built.items()):
        got = ", ".join(str(s) for s in sorted(statuses))
        if len(statuses) > 1:
            problems.append(f"{code} is built with more than one status ({got})")
        if code not in rows:
            problems.append(f"{code} is built by the server but has no row in {RESPONSES}")
        elif rows[code] not in statuses:
            problems.append(f"{code} is documented as {rows[code]} but built with {got}")
    for code in sorted(set(rows) - set(built) - RESERVED):
        problems.append(
            f"{RESPONSES} documents {code}, which the server never builds "
            "(a retired code keeps its row, marked reserved, and goes in RESERVED)"
        )
    for code in sorted(RESERVED - set(rows)):
        problems.append(f"RESERVED lists {code}, which {RESPONSES} does not document: remove it")
    # A parser code that the server also builds (unsupported_operation once the
    # parser caps the operation count) is documented by its table row, which
    # must then say 400, the status of every grammar failure.
    for code in sorted(parser - grammar - set(rows)):
        problems.append(f"parser code {code} is missing from the grammar list in {RESPONSES}")
    for code in sorted(parser & set(rows)):
        if rows[code] != 400:
            problems.append(
                f"parser code {code} answers 400, but {RESPONSES} documents it as {rows[code]}"
            )
    for code in sorted(grammar - parser):
        problems.append(
            f"{RESPONSES} lists {code} as a grammar failure, which the parser never returns"
        )
    return problems


def check_statuses(root, sources):
    rows, _, _, statuses = documented_codes(root)
    used = set(rows.values()) | {400}  # every grammar failure is a 400
    problems = []
    for status in sorted(used - statuses):
        problems.append(f"status {status} is used by a code but missing from the statuses table")
    for status in sorted(statuses - used):
        problems.append(f"the statuses table lists {status}, which no documented code uses")
    return problems


# ---------------------------------------------------------------- filters


def parsed_filters(root):
    text = read(root, "src/url.rs")
    m = re.search(r"\bfn parse_filters\([^)]*\)[^{]*\{", text)
    if not m:
        raise CheckError("src/url.rs has no parse_filters to read the filter names from")
    body = text[m.end() : matching_brace(text, m.end() - 1)]
    names = set(re.findall(r'^\s*\|?\s*\(\s*"([a-z][a-z0-9_]*)"\s*,', body, re.M))
    if not names:
        raise CheckError("parse_filters in src/url.rs has no match arm this check can read")
    return names


def documented_filters(root):
    text = read(root, FILTERS)
    names = set()
    for cells in table_rows(text, {"Filter", "Directive"}):
        ticked = TICKED.findall(cells[0])
        if len(ticked) != 1 or not re.fullmatch(r"[a-z][a-z0-9_]*", ticked[0]):
            raise CheckError(f"{FILTERS}: cannot read the filter row {' | '.join(cells)!r}")
        names.add(ticked[0])
    names.update(re.findall(r"^filters:([a-z][a-z0-9_]*)\(", text, re.M))
    if not names:
        raise CheckError(f"{FILTERS} has no Filter table to read")
    return names


def check_filters(root, sources):
    parsed = parsed_filters(root)
    documented = documented_filters(root)
    problems = []
    for name in sorted(parsed - documented):
        problems.append(
            f"filter {name} is accepted by parse_filters but has no row or "
            f"filters:{name}( line in {FILTERS}"
        )
    for name in sorted(documented - parsed):
        problems.append(f"{FILTERS} documents filter {name}, which parse_filters does not accept")
    return problems


# ---------------------------------------------------------------- changelog


def check_changelog(root, sources):
    for line in read(root, CHANGELOG).splitlines():
        if line.startswith("## "):
            if line.rstrip() == UNRELEASED:
                return []
            return [
                f"{CHANGELOG} opens with {line.rstrip()!r}: keep an {UNRELEASED} heading "
                "above the latest release, empty or not (RELEASING.md step 1)"
            ]
    return [f"{CHANGELOG} has no {UNRELEASED} section"]


# ---------------------------------------------------------------- tested


def named_in(text, code):
    """Whether `code` appears as a quoted string, escaped quotes included."""
    return re.search(rf"[\"']{code}(?=\\?[\"'])", text) is not None


def check_tested(root, sources):
    rows, _, grammar, _ = documented_codes(root)
    tests = test_text(root, sources)
    documented = set(rows) | grammar
    problems = []
    for code in sorted(documented):
        named = named_in(tests, code)
        if named and code in UNTESTED:
            problems.append(f"{code} is named in a test now: remove it from UNTESTED")
        elif not named and code not in UNTESTED:
            problems.append(f"{code} is documented but no test names it: add a test that sends it")
    for code in sorted(UNTESTED - documented):
        problems.append(f"UNTESTED lists {code}, which {RESPONSES} does not document: remove it")
    return problems


CHECKS = (
    ("env", check_env),
    ("defaults", check_defaults),
    ("codes", check_codes),
    ("statuses", check_statuses),
    ("filters", check_filters),
    ("changelog", check_changelog),
    ("tested", check_tested),
)


def run_checks(root):
    """[(check, problem)] for the tree at `root`; empty when it is in sync."""
    try:
        sources = rust_sources(root)
    except CheckError as e:
        return [("sources", str(e))]
    problems = []
    for name, check in CHECKS:
        try:
            found = check(root, sources)
        except CheckError as e:
            found = [str(e)]
        problems.extend((name, problem) for problem in found)
    return problems


def summary(root):
    sources = rust_sources(root)
    own, _ = read_env(sources)
    rows, _, grammar, statuses = documented_codes(root)
    return (
        f"{len(own)} CRABCROP_* names read, "
        f"{len(defaults_in_code(sources))} judged defaults, "
        f"{len(rows)} codes and {len(grammar)} parser codes, "
        f"{len(statuses)} statuses, {len(parsed_filters(root))} filters, "
        f"{len(UNTESTED)} codes still untested"
    )


# ---------------------------------------------------------------- one PR


def git(root, *args):
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise CheckError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def unreleased_lines(text):
    """The non-blank lines of the Unreleased section, subheadings left out."""
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            if inside:
                break
            inside = line.rstrip() == UNRELEASED
        elif inside and line.strip() and not line.startswith("#"):
            lines.append(line.rstrip())
    return lines


def check_entry(root, base, labels):
    """(problems, note) for the pull request whose diff runs from `base` to HEAD."""
    if NO_CHANGELOG_LABEL in labels:
        return [], f"the {NO_CHANGELOG_LABEL} label is set, so no entry is needed"
    # From where HEAD left BASE, so commits BASE gained since do not count.
    base = git(root, "merge-base", base, "HEAD").strip()
    changed = git(root, "diff", "--no-renames", "--name-only", base, "HEAD").splitlines()
    touched = sorted(p for p in changed if p.startswith(ENTRY_PATHS))
    if not touched:
        return [], "the diff touches none of " + ", ".join(ENTRY_PATHS) + ", so no entry is needed"
    added = []
    if CHANGELOG in changed:
        listed = git(root, "ls-tree", "--name-only", base, "--", CHANGELOG).strip()
        before = git(root, "show", f"{base}:{CHANGELOG}") if listed else ""
        added = unreleased_lines(git(root, "show", f"HEAD:{CHANGELOG}"))
        for line in unreleased_lines(before):
            if line in added:
                added.remove(line)
    if added:
        return [], f"{CHANGELOG} gains {len(added)} line(s) under {UNRELEASED}"
    shown = ", ".join(touched[:3]) + (f" and {len(touched) - 3} more" if len(touched) > 3 else "")
    return [
        f"this change touches {shown} but adds no line under {UNRELEASED} in {CHANGELOG}: "
        f"add one, or, when nothing user-facing changed, ask a maintainer for the "
        f"{NO_CHANGELOG_LABEL} label"
    ], None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="repository root (default: this checkout)"
    )
    parser.add_argument(
        "--changelog-entry", metavar="BASE", help="check one pull request's diff from BASE to HEAD"
    )
    parser.add_argument("--labels", default="[]", help="the pull request's labels, as a JSON list")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    if args.changelog_entry:
        try:
            labels = json.loads(args.labels)
        except ValueError:
            labels = None
        if not isinstance(labels, list):
            print(f"check_docs: --labels takes a JSON list, got {args.labels!r}", file=sys.stderr)
            return 1
        try:
            problems, note = check_entry(root, args.changelog_entry, labels)
        except CheckError as e:
            problems, note = [str(e)], None
        for problem in problems:
            print(f"check_docs: changelog-entry: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(f"check_docs: changelog-entry ok: {note}")
        return 0

    problems = run_checks(root)
    for name, problem in problems:
        print(f"check_docs: {name}: {problem}", file=sys.stderr)
    if problems:
        print(
            f"check_docs: {len(problems)} problem(s); see scripts/check_docs.py for each check",
            file=sys.stderr,
        )
        return 1
    print(f"check_docs: ok ({summary(root)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
