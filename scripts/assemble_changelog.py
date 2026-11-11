#!/usr/bin/env python3
"""Assemble the changelog.d/ fragments into CHANGELOG.md (issue #539).

Every pull request used to add its CHANGELOG line at the same place, so each
merge put every other open pull request with an entry into conflict. A pull
request now adds one file instead, changelog.d/<issue>.<section>.md, holding
its entry's paragraph; two pull requests never touch the same file, and the
release gathers them.

  Name     <issue>.<section>.md, or <issue>-<slug>.<section>.md when that
           name is taken (a second entry for the same issue and section).
           The issue is a number without leading zeros, the slug lowercase
           letters and digits in hyphen-separated words, the section one of
           added, changed, deprecated, removed, fixed, security.
  Text     One paragraph in CHANGELOG.md's house style, naming its issue the
           way the entries there do, "(issue #N)" or "(issues #N and #M)".
           Wrapped lines are joined with one space, a byte-order mark,
           surrounding blank lines and a leading "- " are dropped (the
           assembler writes the bullet), and a blank line, a second bullet
           or a heading line is refused.
  Order    Sections in CHANGELOG.md's own order, Added, Changed, Deprecated,
           Removed, Fixed, Security, a missing heading created in its place;
           within one, the hand-written entries already under
           ## [Unreleased] first, as written, then the fragments by issue
           number, the plain name before its slugged ones, slugs
           alphabetically. A fragment that opens "This release starts a new
           cache epoch" (RELEASING.md step 4, in any letter case) leads
           Changed instead, since it speaks of the changes below it;
           [Unreleased] holds one such note at most.
  Ignored  changelog.d/README.md and dotfiles.

Usage:
  scripts/assemble_changelog.py --check [--root DIR]
  scripts/assemble_changelog.py X.Y.Z [--date YYYY-MM-DD] [--root DIR]

--check prints ## [Unreleased] as the release will write it, the fragments
in place, and writes nothing; it exits 1 only when a fragment is malformed
(or CHANGELOG.md has no single ## [Unreleased] to assemble into). Given a version,
the script moves ## [Unreleased]'s entries and every fragment into a new
## [X.Y.Z] - YYYY-MM-DD section (today by default), leaves an empty
## [Unreleased] above it, and deletes the fragments it wrote; headings left
with no entry are dropped. scripts/bump_version.py runs that step, so the
release commit carries the assembled section that cargo-dist reads the
release notes from (RELEASING.md step 1). Running it again for a version
CHANGELOG.md already has is a no-op when nothing is left to assemble, and
refused otherwise. scripts/check_docs.py checks the fragments of a tree and,
with --changelog-entry, the ones a pull request adds, with this module's
rules. Standard library only.
"""

import argparse
import datetime
import re
import sys
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHANGELOG = "CHANGELOG.md"
FRAGMENTS = "changelog.d"
UNRELEASED = "## [Unreleased]"
SECTIONS = ("added", "changed", "deprecated", "removed", "fixed", "security")
HEADINGS = {section: f"### {section.capitalize()}" for section in SECTIONS}
README = "README.md"
EPOCH = "This release starts a new cache epoch"
SECOND_EPOCH = (
    "a second cache-epoch note: ## [Unreleased] already has one, "
    "so add this change to that note instead"
)

STEM = re.compile(r"(?P<issue>[1-9][0-9]*)(?:-(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*))?")
ISSUE_REF = re.compile(r"\bissues?\s+#\d+(?:\s*(?:,\s*(?:and\s+)?|and\s+)#\d+)*", re.I)
# A list item's marker: a line opening with one starts another entry.
BULLET = re.compile(r"[-*+](?:\s+|$)")
# A Markdown heading line; "#612 and #626)" wrapped onto a line of its own is none.
HEADING = re.compile(r"#{1,6}(?:\s|$)")
VERSION = re.compile(r"\d+\.\d+\.\d+")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

BAD_NAME = (
    "is not named <issue>.<section>.md or <issue>-<slug>.<section>.md "
    "(CONTRIBUTING.md, Changelog)"
)
NOT_DIRECT = "fragments sit directly in changelog.d/, not in a subdirectory"

Fragment = namedtuple("Fragment", "name issue slug section text")
Plan = namedtuple("Plan", "text remove note")


class FragmentError(Exception):
    """One fragment's problem, without its path."""


class AssembleError(Exception):
    """Every problem that stops an assembly, each a full message."""

    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def ignored(name):
    return name == README or name.startswith(".")


def parse_name(name):
    """(issue, slug, section) from a fragment's file name."""
    parts = name.split(".")
    stem = STEM.fullmatch(parts[0])
    if len(parts) != 3 or parts[2] != "md" or stem is None:
        raise FragmentError(BAD_NAME)
    if parts[1] not in SECTIONS:
        raise FragmentError(f"unknown section {parts[1]!r}: use one of {', '.join(SECTIONS)}")
    return int(stem.group("issue")), stem.group("slug") or "", parts[1]


def looks_like_fragment(name):
    try:
        parse_name(name)
    except FragmentError:
        return False
    return True


def issues_named(text):
    """The issue numbers the text names as `issue #N` or `issues #N and #M`."""
    found = []
    for ref in ISSUE_REF.finditer(text):
        for number in re.findall(r"#(\d+)", ref.group(0)):
            if int(number) not in found:
                found.append(int(number))
    return found


def describe(issues):
    names = [f"#{n}" for n in issues]
    if len(names) == 1:
        return f"issue {names[0]}"
    return f"issues {', '.join(names[:-1])} and {names[-1]}"


def opens_epoch(text):
    """Whether an entry's text is the cache-epoch note (RELEASING.md step 4)."""
    return text[: len(EPOCH)].lower() == EPOCH.lower()


def entry_text(raw, issue, section):
    """The entry a fragment's bytes hold, normalized as the module says."""
    try:
        # An editor's byte-order mark is no text: kept, it would hide a
        # leading "- " or a cache-epoch note.
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise FragmentError("is not UTF-8 text") from None
    lines = [line.strip() for line in text.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    bullet = BULLET.match(lines[0]) if lines else None
    if bullet:
        lines[0] = lines[0][bullet.end() :]
        if not lines[0]:  # the bullet on a line of its own
            lines.pop(0)
    if not lines:
        raise FragmentError("is empty: write the entry's paragraph")
    if "" in lines:
        raise FragmentError(
            "holds more than one paragraph: an entry is one paragraph "
            "(a second entry is a second fragment)"
        )
    if any(BULLET.match(line) for line in lines):
        raise FragmentError(
            "has a line opening with '- ', '* ' or '+ ', which starts another entry: "
            "a second entry is a second fragment; if the paragraph wrapped there, rewrap it"
        )
    if any(HEADING.match(line) for line in lines):
        raise FragmentError(
            "has a line opening with '# ', a heading, which the assembler writes: "
            "if the paragraph wrapped there, rewrap it"
        )
    text = " ".join(lines)
    named = issues_named(text)
    if not named:
        raise FragmentError(f"names no issue: say (issue #{issue}) as the entries in {CHANGELOG} do")
    if issue not in named:
        raise FragmentError(f"names {describe(named)}, not #{issue} from its file name")
    if opens_epoch(text) and section != "changed":
        raise FragmentError(
            f"a cache-epoch note belongs under Changed: name it {issue}.changed.md"
        )
    return text


def parse(name, raw):
    """The Fragment a file named `name` holding `raw` bytes is, or FragmentError."""
    issue, slug, section = parse_name(name)
    return Fragment(name, issue, slug, section, entry_text(raw, issue, section))


def check_changed_file(rel, read):
    """(the Fragment it is or None, its problems as "rel: problem" messages)
    for one file, by repository path, that a change adds or edits. A file
    that neither is nor looks like a fragment gives (None, []). `read`
    returns the file's bytes; only a file in changelog.d/ is read."""
    folder, _, name = rel.partition("/")
    if folder == FRAGMENTS:
        if "/" in name:
            return None, [f"{rel}: {NOT_DIRECT}"]
        if ignored(name):
            return None, []
        try:
            return parse(name, read()), []
        except FragmentError as e:
            return None, [f"{rel}: {e}"]
    if looks_like_fragment(rel.rsplit("/", 1)[-1]):
        return None, [f"{rel}: a changelog fragment belongs directly in {FRAGMENTS}/"]
    return None, []


def order(fragment):
    return (fragment.issue, fragment.slug)


def read_fragments(root):
    """(fragments by issue order, problems) for the tree at `root`."""
    folder = Path(root) / FRAGMENTS
    if not folder.is_dir():
        return [], []
    found, problems = [], []
    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        if ignored(path.name):
            continue
        rel = f"{FRAGMENTS}/{path.name}"
        if not path.is_file():
            problems.append(f"{rel}: {NOT_DIRECT}")
            continue
        try:
            found.append(parse(path.name, path.read_bytes()))
        except FragmentError as e:
            problems.append(f"{rel}: {e}")
    return sorted(found, key=order), problems


def locate(lines):
    """(start, end): the ## [Unreleased] heading's index and the next ## heading's."""
    found = [i for i, line in enumerate(lines) if line.rstrip() == UNRELEASED]
    if not found:
        raise AssembleError([f"{CHANGELOG} has no {UNRELEASED} section to assemble into"])
    if len(found) > 1:
        # A release would take the first and leave the second's entries
        # below it, where no later release looks.
        raise AssembleError(
            [
                f"{CHANGELOG} has {len(found)} {UNRELEASED} sections, at lines "
                f"{', '.join(str(i + 1) for i in found)}: merge them into the first"
            ]
        )
    start = found[0]
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return start, end


def is_epoch(entry_line):
    return entry_line.startswith("- ") and opens_epoch(entry_line[2:])


def load(root):
    """(CHANGELOG.md's lines, start, end, fragments); AssembleError with every problem."""
    path = Path(root) / CHANGELOG
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise AssembleError([f"{CHANGELOG} is missing"]) from None
    # Lines end at "\n" alone (reading turned CRLF into it); splitlines()
    # would also break at a form feed or U+2028 inside an entry.
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    start, end = locate(lines)
    fragments, problems = read_fragments(root)
    notes = sum(1 for line in lines[start + 1 : end] if is_epoch(line))
    for fragment in fragments:
        if opens_epoch(fragment.text):
            if notes:
                problems.append(f"{FRAGMENTS}/{fragment.name}: {SECOND_EPOCH}")
            notes += 1
    if problems:
        raise AssembleError(problems)
    return lines, start, end, fragments


def trimmed(lines):
    lines = list(lines)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def place(content, lead, tail):
    """A subsection's lines after its heading, with entries before and after
    the hand-written ones, one blank line between blocks."""
    written = trimmed(content)
    blocks = [[f"- {text}"] for text in lead]
    blocks += [written] if written else []
    blocks += [[f"- {text}"] for text in tail]
    out = []
    for block in blocks:
        out += [""] + block
    return out + [""]


def assemble_body(body, fragments):
    """[Unreleased]'s lines after its heading with the fragments in place and
    headings that hold nothing dropped. Untouched text keeps its bytes."""
    preamble, subsections = [], []
    for line in body:
        if line.startswith("### "):
            subsections.append([line, []])
        elif subsections:
            subsections[-1][1].append(line)
        else:
            preamble.append(line)
    rank = {HEADINGS[s]: i for i, s in enumerate(SECTIONS)}
    for i, section in enumerate(SECTIONS):
        mine = [f for f in fragments if f.section == section]
        if not mine:
            continue
        heading = HEADINGS[section]
        at = next((j for j, sub in enumerate(subsections) if sub[0].rstrip() == heading), None)
        if at is None:
            later = (j for j, sub in enumerate(subsections) if rank.get(sub[0].rstrip(), -1) > i)
            at = next(later, len(subsections))
            subsections.insert(at, [heading, []])
        lead = [f.text for f in mine if opens_epoch(f.text)]
        tail = [f.text for f in mine if not opens_epoch(f.text)]
        subsections[at][1] = place(subsections[at][1], lead, tail)
    out = list(preamble)
    for heading, content in subsections:
        if not trimmed(content):
            continue
        if out and out[-1].strip():
            out.append("")
        if not out:
            out.append("")
        out += [heading] + content
    return out


def entries(lines):
    return sum(1 for line in lines if line.startswith("- "))


def preview(root):
    """(## [Unreleased] as the release will write it, its note); AssembleError."""
    lines, start, end, fragments = load(root)
    body = assemble_body(lines[start + 1 : end], fragments)
    section = trimmed([lines[start]] + body)
    hand = entries(lines[start + 1 : end])
    note = f"{len(fragments)} fragment(s) after {hand} hand-written entr{'y' if hand == 1 else 'ies'}"
    return "\n".join(section) + "\n", note


def release_date(date):
    if date is None:
        return datetime.date.today().isoformat()
    try:
        if not DATE.fullmatch(date):
            raise ValueError(date)
        datetime.date.fromisoformat(date)
    except ValueError:
        raise AssembleError([f"the release date is YYYY-MM-DD, a real day; got {date!r}"]) from None
    return date


def plan(root, version, date=None):
    """The release's Plan: CHANGELOG.md's new text (None when nothing is to
    be written), the fragments to remove, and a note. AssembleError lists
    every problem; nothing is written either way."""
    if not VERSION.fullmatch(version or ""):
        raise AssembleError([f"the release takes a version X.Y.Z; got {version!r}"])
    date = release_date(date)
    lines, start, end, fragments = load(root)
    body = lines[start + 1 : end]
    released = re.compile(rf"## \[{re.escape(version)}\](?:\s|$)")
    if any(released.match(line) for line in lines):
        if not fragments and not any(line.strip() for line in body):
            return Plan(None, [], f"{CHANGELOG} already has ## [{version}] and nothing is left to assemble")
        raise AssembleError(
            [
                f"{CHANGELOG} already has ## [{version}], and {UNRELEASED} or {FRAGMENTS}/ "
                "still holds entries: release them under a new version"
            ]
        )
    section = trimmed(assemble_body(body, fragments))
    head = [lines[start].rstrip(), "", f"## [{version}] - {date}"]
    new = lines[:start] + head + ([""] + section if section else [])
    if end < len(lines):
        new += [""] + lines[end:]
    hand = entries(body)
    note = (
        f"## [{version}] - {date} holds {hand} hand-written entr{'y' if hand == 1 else 'ies'} "
        f"and {len(fragments)} fragment(s)"
    )
    if not section:
        note += f"; it is empty, since {UNRELEASED} and {FRAGMENTS}/ held nothing to release"
    return Plan("\n".join(new) + "\n", [f"{FRAGMENTS}/{f.name}" for f in fragments], note)


def apply(root, plan, dry_run=False):
    """Write a Plan (unless dry_run); the lines that say what was done."""
    if plan.text is None:
        return [plan.note]
    if dry_run:
        return [f"would update {CHANGELOG}: {plan.note}"] + [f"would remove {rel}" for rel in plan.remove]
    with open(Path(root) / CHANGELOG, "w", encoding="utf-8", newline="\n") as f:
        f.write(plan.text)
    for rel in plan.remove:
        (Path(root) / rel).unlink()
    return [f"updated {CHANGELOG}: {plan.note}"] + [f"removed {rel}" for rel in plan.remove]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("version", nargs="?", help="the release version, X.Y.Z")
    parser.add_argument("--check", action="store_true", help="print the assembled [Unreleased]; write nothing")
    parser.add_argument("--date", help="the release date, YYYY-MM-DD (default: today)")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (default: this checkout)")
    args = parser.parse_args(argv)
    if args.check == bool(args.version):
        parser.error("give either --check or a version X.Y.Z")
    if args.check and args.date:
        parser.error("--date goes with a version")
    root = args.root.resolve()
    try:
        if args.check:
            text, note = preview(root)
            sys.stdout.write(text)
            print(f"assemble_changelog: {note} under {UNRELEASED}; nothing written", file=sys.stderr)
            return 0
        for line in apply(root, plan(root, args.version, args.date)):
            print(line)
        return 0
    except AssembleError as e:
        for problem in e.problems:
            print(f"assemble_changelog: {problem}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
