"""The release version sites, shared by the version tools.

crabcrop releases one version number across all artifacts (RELEASING.md).
SITES lists every place in the tree that pins it: scripts/bump_version.py
writes them and scripts/check_versions.py reads them, from this one list,
so the two cannot disagree about what a release moves. A file that
carries the release version and is missing here gets edited by hand at
every release, or left behind (issue #486); scripts/test_version_tools.py
bumps a copy of the tree and fails on any old version left outside
CHANGELOG.md.

A site finds its pins as the offsets of each version string, so the bump
rewrites those characters and nothing else, and the check reads exactly
what the bump writes. Each site also says how many pins its file carries:
a file with another count (a third `tag:` line in the Helm values, a
lockfile entry gone) is reported as a problem with that file, never a
traceback.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class SiteError(Exception):
    """A file whose pins are not where its site says they are."""


class Site:
    """The release version pins in one file.

    `find(text)` returns the (start, end) offsets of each pin's version
    string. `count` is how many pins the file carries: a number, or None
    for one or more (a guide may show the image several times). A `minor`
    site pins major.minor only, as a dependency requirement does: it moves
    on a minor bump and stays put on a patch one.
    """

    def __init__(self, path, what, find, count=1, minor=False):
        self.path = path
        self.what = what
        self.find = find
        self.count = count
        self.minor = minor

    def expected(self, version):
        return ".".join(version.split(".")[:2]) if self.minor else version

    def locate(self, text):
        spans = self.find(text)
        if self.count is None and not spans:
            raise SiteError(f"no {self.what} found")
        if self.count is not None and len(spans) != self.count:
            raise SiteError(f"expected {self.count} {self.what}, found {len(spans)}")
        return spans


def pattern(path, what, regex, flags=0, count=1, minor=False):
    """A site found by a regex whose first group is the version."""
    compiled = re.compile(regex, flags)
    return Site(path, what, lambda text: [m.span(1) for m in compiled.finditer(text)], count, minor)


def lock_entry(path, name):
    """A Cargo.lock entry of a package in this repository. Path packages
    carry no `source` line, and cargo writes the version under the name,
    so the bump moves the entry exactly as `cargo update -w` would."""
    return pattern(
        path,
        f"{name} package entry",
        rf'^\[\[package\]\]\nname = "{re.escape(name)}"\nversion = "([^"]+)"$(?!\nsource = )',
        re.M,
    )


_WS = re.compile(r"[ \t\n\r]*")
_DECODE = json.JSONDecoder().raw_decode


def _field_name(field):
    out = ""
    for key in field:
        if isinstance(key, int):
            out += f"[{key}]"
        elif re.fullmatch(r"[A-Za-z_]\w*", key):
            out += f".{key}" if out else key
        else:
            out += f"[{json.dumps(key)}]"
    return out


def _value_offset(text, field):
    """The offset of the value at `field` (object keys and list indices)
    in the valid JSON document `text`, or None when it has none."""
    i = _WS.match(text).end()
    for key in field:
        in_list = isinstance(key, int)
        if text[i : i + 1] != ("[" if in_list else "{"):
            return None
        i = _WS.match(text, i + 1).end()
        index = 0
        while text[i : i + 1] not in ("]", "}"):
            if in_list:
                hit = index == key
            else:
                name, i = json.decoder.scanstring(text, i + 1)
                i = _WS.match(text, _WS.match(text, i).end() + 1).end()
                hit = name == key
            if hit:
                break
            _, i = _DECODE(text, i)
            i = _WS.match(text, i).end()
            if text[i : i + 1] == ",":
                i = _WS.match(text, i + 1).end()
            index += 1
        else:
            return None
    return i


def json_fields(path, *fields):
    """A site found by JSON field paths whose values are the version.

    The offsets come from the document's own text, so a bump keeps the
    file's formatting (a package.json is laid out by hand; a
    package-lock.json by npm) and changes the version strings alone.
    """

    def find(text):
        try:
            json.loads(text)
        except ValueError as err:
            raise SiteError(f"not valid JSON ({err})") from None
        spans = []
        for field in fields:
            start = _value_offset(text, field)
            if start is None:
                raise SiteError(f"no {_field_name(field)} field")
            value, end = _DECODE(text, start)
            if not isinstance(value, str) or text[start + 1 : end - 1] != value:
                raise SiteError(f"{_field_name(field)} is not a plain string")
            spans.append((start + 1, end - 1))
        return spans

    return Site(path, ", ".join(_field_name(f) for f in fields), find, count=len(fields))


VERSION = ("version",)
ROOT_PACKAGE = ("packages", "", "version")


def linked(relative):
    """A package-lock.json entry for @crabcrop/url linked from the tree:
    npm records the linked package's own version, which a bump moves."""
    return ("packages", relative, "version")


FRAMEWORKS = ("nextjs", "nuxt", "hydrogen", "astro", "sveltekit")
# A guide pins the release as an image (`crabcrop:X.Y.Z`,
# `crabcrop-ai:X.Y.Z`) or as a chart override (`--set image.tag=X.Y.Z`
# in the Helm README's own-build snippet). The override was missed for
# three releases, so that snippet installed a tag it never pushed
# (issue #565).
IMAGE_PIN = r"(?:crabcrop(?:-ai)?:|\bimage\.tag=)(\d+\.\d+\.\d+)"

# The first site is the reference: the server's own version is the
# release version, and every other site must agree with it.
SITES = [
    pattern("Cargo.toml", "[package] version", r'\[package\][^\[]*?^version\s*=\s*"([^"]+)"', re.S | re.M),
    # The server's requirement on the crate it ships with. A minor
    # bump must move it ("0.9" to "0.10"), or the crates.io package of
    # the server would ask for a crabcrop-url the release does not publish.
    pattern(
        "Cargo.toml",
        "crabcrop-url requirement (major.minor)",
        r'^crabcrop-url\s*=\s*\{[^}\n]*\bversion\s*=\s*"([^"]+)"',
        re.M,
        minor=True,
    ),
    pattern(
        "crates/crabcrop-url/Cargo.toml",
        "[package] version",
        r'\[package\][^\[]*?^version\s*=\s*"([^"]+)"',
        re.S | re.M,
    ),
    lock_entry("Cargo.lock", "crabcrop"),
    lock_entry("Cargo.lock", "crabcrop-url"),
    # The fuzz package builds the server library and the crate, so its
    # own lockfile records both (fuzz/README.md).
    lock_entry("fuzz/Cargo.lock", "crabcrop"),
    lock_entry("fuzz/Cargo.lock", "crabcrop-url"),
    json_fields("sdks/typescript/package.json", VERSION),
    json_fields("sdks/typescript/package-lock.json", VERSION, ROOT_PACKAGE),
    json_fields("sdks/mcp/package.json", VERSION),
    json_fields("sdks/mcp/package-lock.json", VERSION, ROOT_PACKAGE, linked("../typescript")),
    *(
        site
        for name in FRAMEWORKS
        for site in (
            json_fields(f"frameworks/{name}/package.json", VERSION),
            json_fields(
                f"frameworks/{name}/package-lock.json",
                VERSION,
                ROOT_PACKAGE,
                linked("../../sdks/typescript"),
            ),
        )
    ),
    # The browser-bundles harness has no version of its own, but its
    # lockfile records the SDK it links.
    json_fields("tests/browser-bundles/package-lock.json", linked("../../sdks/typescript")),
    json_fields("playground/package.json", VERSION),
    pattern(
        "sdks/python/pyproject.toml",
        "[project] version",
        r'\[project\][^\[]*?^version\s*=\s*"([^"]+)"',
        re.S | re.M,
    ),
    pattern(
        "sdks/mcp/src/index.ts",
        "SERVER.version",
        r'const SERVER = \{ name: "crabcrop", version: "([^"]+)" \}',
    ),
    # The MCP registry manifest carries the version twice: the
    # manifest itself and its npm package entry.
    json_fields("sdks/mcp/server.json", VERSION, ("packages", 0, "version")),
    pattern("plugins/wordpress/crabcrop.php", "plugin header Version", r"^\s*\* Version:\s+(\S+)", re.M),
    pattern("deploy/helm/crabcrop/Chart.yaml", "version", r"^version:\s*(\S+)", re.M),
    pattern("deploy/helm/crabcrop/Chart.yaml", "appVersion", r'^appVersion:\s*"([^"]+)"', re.M),
    # The deploy pins (issue #383): the tags a reader copies out of the
    # chart values, the kustomize base and AI component, the README's
    # install lines and the platform guides. A release that leaves one
    # behind ships a guide pointing at an old image.
    pattern(
        "deploy/helm/crabcrop/values.yaml",
        'image tags (`tag: "..."` lines: image.tag and ai.image.tag)',
        r'^[ \t]+tag:\s*"([^"]+)"',
        re.M,
        count=2,
    ),
    pattern(
        "deploy/helm/crabcrop/values-production.yaml",
        'image tag (`tag: "..."` line: image.tag)',
        r'^[ \t]+tag:\s*"([^"]+)"',
        re.M,
    ),
    pattern("deploy/kustomize/base/deployment.yaml", "crabcrop image", r"image: ghcr\.io/dayvson/crabcrop:(\S+)"),
    pattern(
        "deploy/kustomize/components/ai/deployment-patch.yaml",
        "crabcrop-ai image",
        r"image: ghcr\.io/dayvson/crabcrop-ai:(\S+)",
    ),
    *(
        pattern(guide, "image pins", IMAGE_PIN, count=None)
        for guide in (
            "README.md",
            "deploy/cloud-run/README.md",
            "deploy/container-apps/README.md",
            "deploy/ecs-fargate/README.md",
            "deploy/helm/crabcrop/README.md",
        )
    ),
]


def _read(root, path):
    try:
        return (root / path).read_text(encoding="utf-8")
    except OSError as err:
        raise SiteError(f"cannot read it ({err.strerror or err})") from None
    except UnicodeDecodeError:
        raise SiteError("not UTF-8 text") from None


def check(root=ROOT):
    """Read every site.

    Returns (version, rows, problems): the release version (the first
    site's, or None when it cannot be read), one (values, site) row per
    site with the pins' values in file order (None when the site cannot
    be read), and one message per site that is unreadable or disagrees.
    """
    version, rows, problems = None, [], []
    for index, site in enumerate(SITES):
        try:
            text = _read(root, site.path)
            values = [text[start:end] for start, end in site.locate(text)]
        except SiteError as err:
            rows.append((None, site))
            problems.append(f"{site.path}: {err}")
            continue
        rows.append((values, site))
        if index == 0:
            version = values[0]
        if version is None:
            continue
        want = site.expected(version)
        stale = sorted({value for value in values if value != want})
        if stale:
            problems.append(f"{site.path}: {site.what} is {', '.join(stale)}, expected {want}")
    if version is None:
        problems.append(f"no release version to compare against: {SITES[0].path} {SITES[0].what} is unreadable")
    return version, rows, problems


def bump(version, root=ROOT):
    """Move every site to `version`.

    Returns (changed, problems): the new text of each file the bump
    changes, and one message per site that stops it. Nothing is written
    here; with any problem, the caller writes nothing at all.
    """
    original, texts, problems = {}, {}, []
    for site in SITES:
        try:
            if site.path not in texts:
                original[site.path] = texts[site.path] = _read(root, site.path)
            text = texts[site.path]
            new = site.expected(version)
            # Last pin first, so each replacement leaves the offsets
            # before it valid; fields come in list order, not file order.
            for start, end in sorted(site.locate(text), reverse=True):
                text = text[:start] + new + text[end:]
        except SiteError as err:
            problems.append(f"{site.path}: {err}")
            continue
        texts[site.path] = text
    changed = {path: text for path, text in texts.items() if text != original[path]}
    return changed, problems
