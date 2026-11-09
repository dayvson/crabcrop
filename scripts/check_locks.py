#!/usr/bin/env python3
"""Check that the Cargo lockfiles give every decoder one version.

Two rules (issue #552), both read from Cargo.lock and fuzz/Cargo.lock:

1. No decoder in two versions. A crate in DECODERS, which parse request
   or source-image bytes in the shipped binary, appears once per lockfile:
   one version from one source.
   A second version means some input reaches a decoder that the tests,
   the fuzzing and the pre-checks do not run: #494's lock refresh left
   oxideav-heif decoding HEVC with oxideav-h265 0.0.15, while the SPS
   pre-check in src/heic.rs and the fuzz lock used 0.0.16. A split cargo
   cannot unify, because each dependent requires a version the other's
   requirement excludes, is allowed only when SPLITS names it, says why
   no request bytes reach the second copy, and lists the same
   semver-compatible lines the lockfile carries.

2. The fuzz lock matches the root lock. Every package both lockfiles
   carry has the same versions and sources in each, so the fuzz harness
   builds the decoders the server ships. A package only one lockfile
   carries (libfuzzer-sys and its tree, the root's dev-dependencies) is
   not compared. fuzz/README.md has the refresh recipe.

DECODERS covers the image decoders and the parsers crabcrop's own
pre-checks share with them (oxideav-h265 for HEIC, roxmltree and
simplecss for SVG). It leaves out the C2PA manifest reader (c2pa and
its ASN.1, CBOR and XML tree), which reads a source's provenance but
decodes no pixels and is not fuzzed; c2pa itself requires der 0.7
(through pkcs8 0.10) and der 0.8, a split no lock change can remove.

Run from anywhere: python3 scripts/check_locks.py [--root DIR]. CI runs
it in rust.yml's build job and scripts/test_all.sh runs it with its
tests (scripts/test_check_locks.py).
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCKS = ("Cargo.lock", "fuzz/Cargo.lock")

# Crates that parse bytes a request or its source image supplies, by what
# they decode. A name ending in "-" or "_" covers every crate with that
# prefix, so a new oxideav codec is covered the day it enters the lock.
DECODERS = {
    # HEIC: the container and every codec it can dispatch to.
    "oxideav-": "HEIC and HEVC",
    # Raster formats.
    "image": "the raster decoders' front end",
    "png": "PNG",
    "fdeflate": "PNG's inflate",
    "flate2": "zlib inflate",
    "miniz_oxide": "zlib inflate behind flate2",
    "gif": "GIF",
    "weezl": "LZW for GIF and TIFF",
    "zune-jpeg": "JPEG",
    "zune-core": "JPEG",
    "image-webp": "WebP",
    "webp": "WebP, through libwebp",
    "libwebp-sys": "WebP, through libwebp",
    "tiff": "TIFF",
    "jxl": "JPEG XL",
    "jxl_": "JPEG XL",
    "dav1d": "AVIF, with avif-decode",
    "dav1d-sys": "AVIF, with avif-decode",
    "mp4parse": "the AVIF container, with avif-decode",
    # SVG.
    "resvg": "SVG rendering",
    "usvg": "SVG parsing",
    "roxmltree": "SVG XML",
    # src/svg.rs's expansion pre-check reads a document's CSS with
    # simplecss before usvg reads it again: both must be one version.
    "simplecss": "SVG CSS",
    "svgtypes": "SVG attribute values",
    "data-url": "SVG data: URLs",
    "imagesize": "the size of an image an SVG embeds",
    # Metadata and color.
    "kamadak-exif": "EXIF",
    "moxcms": "ICC profiles",
}

# Decoders that legitimately appear twice: name -> (the semver-compatible
# lines the lockfile carries, why no request bytes reach the extra copy).
# A new line, or a split that went away, fails the check until this entry
# is reviewed and updated.
SPLITS = {
    "moxcms": (
        ("0.8", "0.9"),
        "image 0.25 requires 0.8 and calls it only to build profiles from "
        "CICP codes; source ICC bytes are parsed by src/color.rs with 0.9",
    ),
    "miniz_oxide": (
        ("0.8", "0.9"),
        "png 0.18 lists 0.8 but inflates with fdeflate and flate2 and never "
        "calls it; flate2 1.1 inflates with 0.9",
    ),
    "roxmltree": (
        ("0.20", "0.21"),
        "fontdb's fontconfig-parser requires 0.20 and reads only the host's "
        "fontconfig files; SVG is parsed by usvg and src/svg.rs with 0.21",
    ),
}

FIELD = re.compile(r'^(name|version|source) = "([^"]*)"$')


def parse_lock(text):
    """Every [[package]] entry of a Cargo.lock as (name, version, source).

    A path package has no source; it is reported as "path"."""
    packages = []
    for block in re.split(r"^\[\[package\]\]$", text, flags=re.M)[1:]:
        fields = {}
        for line in block.splitlines():
            if line.startswith("["):
                break
            found = FIELD.match(line.strip())
            if found:
                fields[found.group(1)] = found.group(2)
        if "name" not in fields or "version" not in fields:
            raise ValueError(f"a [[package]] entry without a name and version: {block.strip()[:80]!r}")
        packages.append((fields["name"], fields["version"], fields.get("source", "path")))
    if not packages:
        raise ValueError("no [[package]] entries")
    return packages


def versions_by_name(packages):
    found = {}
    for name, version, source in packages:
        found.setdefault(name, set()).add((version, source))
    return found


def version_key(version):
    """Numeric order for a version string: 0.9.10 sorts after 0.9.9."""
    release = re.split(r"[-+]", version, maxsplit=1)[0]
    return [int(part) if part.isdigit() else 0 for part in release.split(".")]


def compatible_line(version):
    """The semver-compatible line Cargo unifies within: 1.4.2 -> 1,
    0.9.2 -> 0.9, 0.0.16 -> 0.0.16 (every 0.0.x is its own line)."""
    major, minor, patch = (version_key(version) + [0, 0, 0])[:3]
    if major:
        return str(major)
    if minor:
        return f"0.{minor}"
    return f"0.0.{patch}"


def decoder_label(name):
    for key, label in DECODERS.items():
        if name == key or (key.endswith(("-", "_")) and name.startswith(key)):
            return label
    return None


def check_decoders(lock, found):
    problems = []
    for name in sorted(found):
        label = decoder_label(name)
        if label is None:
            continue
        versions = sorted({version for version, _ in found[name]}, key=version_key)
        lines = sorted({compatible_line(v) for v in versions}, key=version_key)
        listed = ", ".join(versions)
        if len(found[name]) > len(versions):
            # One version from two sources, such as crates.io and a git
            # fork: two copies of the decoder all the same.
            problems.append(
                f"{lock}: {name} ({label}) is locked from {len(found[name])} sources "
                f"({', '.join(f'{v} from {s}' for v, s in sorted(found[name]))}), "
                "and only one may decode input: drop the patch or fork that adds the second"
            )
            continue
        if name not in SPLITS:
            if len(versions) > 1:
                problems.append(
                    f"{lock}: {name} ({label}) is locked at {len(versions)} versions ({listed}), "
                    f"and only one may decode input: run cargo update -p {name}@{versions[0]} "
                    f"--precise {versions[-1]} if every dependent allows it, or align the requirements "
                    f"(cargo tree -i {name}@{versions[0]} shows who needs it)"
                )
            continue
        expected, _ = SPLITS[name]
        if len(lines) < len(versions):
            problems.append(
                f"{lock}: {name} ({label}) has more than one version in a compatible line ({listed}); "
                f"cargo update -p {name}@{versions[0]} can unify them"
            )
        elif len(versions) == 1:
            problems.append(
                f"{lock}: {name} ({label}) is down to one version ({listed}); "
                "remove its entry from SPLITS in scripts/check_locks.py"
            )
        elif tuple(lines) != expected:
            problems.append(
                f"{lock}: {name} ({label}) is locked at {listed} (lines {', '.join(lines)}), but SPLITS "
                f"allows lines {', '.join(expected)}; check that no request bytes reach the extra copy, "
                "then update SPLITS in scripts/check_locks.py"
            )
    return problems


def describe(entries):
    return ", ".join(version if source != "path" else f"{version} (path)" for version, source in sorted(entries))


def check_agreement(root_found, fuzz_found):
    problems = []
    for name in sorted(set(root_found) & set(fuzz_found)):
        if root_found[name] != fuzz_found[name]:
            problems.append(
                f"fuzz/Cargo.lock has {name} {describe(fuzz_found[name])}, "
                f"Cargo.lock has {describe(root_found[name])}"
            )
    if problems:
        problems.append(
            "refresh the fuzz lock from the root lock (fuzz/README.md): "
            "cp Cargo.lock fuzz/Cargo.lock && cargo update -w --manifest-path fuzz/Cargo.toml"
        )
    return problems


def check(root):
    problems = []
    found = {}
    for lock in LOCKS:
        path = Path(root) / lock
        try:
            found[lock] = versions_by_name(parse_lock(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as error:
            problems.append(f"{lock}: cannot read it: {error}")
            continue
        problems += check_decoders(lock, found[lock])
    if len(found) == len(LOCKS):
        problems += check_agreement(found["Cargo.lock"], found["fuzz/Cargo.lock"])
    return found, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="repository checkout to check")
    args = parser.parse_args(argv)
    found, problems = check(args.root)
    if problems:
        print("lockfile check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    root_found = found["Cargo.lock"]
    shared = set(root_found) & set(found["fuzz/Cargo.lock"])
    decoders = sorted(n for n in root_found if decoder_label(n))
    print(
        f"{len(decoders)} decoder crates at one version each (splits allowed: {', '.join(sorted(SPLITS))}); "
        f"fuzz/Cargo.lock agrees with Cargo.lock on all {len(shared)} shared packages"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
