"""The Python explainer and URL verifier against the TypeScript ones
(issue #716), on inputs generated from the two grammar corpora: their
remainders with segments dropped, repeated and swapped, filters
dropped and repeated, characters changed, and numbers set to the edges
of the ranges the grammar checks, under every signature shape and URL
form the explainer reads. Every field of every explanation must agree:
the signature, each segment's kind, title, detail, error and expiry
flag, the source, the remainder and the errors, message for message.
So must every answer of the two verifiers, asked about each input as
generated and re-signed with a known secret, under one secret and
under keyrings.

The TypeScript SDK runs on Node from its sources, sdks/typescript/src,
so the run needs Node 22.18 or newer (or 23.6 or newer), where type
stripping is on by default, and no build or npm install: the harness
resolves the `#crypto` import, which package.json maps to the build
output, to the sources' crypto/node.ts. Without such a Node the tests
are skipped, unless CRABCROP_DIFF_REQUIRE_NODE=1, which
scripts/test_all.sh sets, since its TypeScript steps need that Node
anyway: then they fail.

The inputs are a function of the seed, so a failure repeats. To run
it alone, or on a larger or different set:

    cd sdks/python
    python3 -m unittest tests.test_explain_differential -v
    CRABCROP_DIFF_CASES=100000 CRABCROP_DIFF_SEED=7 \\
        python3 -m unittest tests.test_explain_differential -v
"""

import hashlib
import hmac
import json
import os
import random
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import crabcrop_url as cc  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
INDEX_TS = REPO / "sdks" / "typescript" / "src" / "index.ts"
VERDICTS = json.loads((REPO / "spec" / "explain-verdicts.json").read_text(encoding="utf-8"))
VECTORS = json.loads((REPO / "spec" / "operation-vectors.json").read_text(encoding="utf-8"))

DEFAULT_CASES = 6000
CASES = int(os.environ.get("CRABCROP_DIFF_CASES", str(DEFAULT_CASES)))
SEED = int(os.environ.get("CRABCROP_DIFF_SEED", "716"))
REQUIRE_NODE = os.environ.get("CRABCROP_DIFF_REQUIRE_NODE") == "1"
NOW = 1700000000
SECRET = "s3cr3t"

# Reads {"now", "inputs", "verify"} on stdin and writes the TypeScript
# explanations and verifier answers, in order, on stdout. The SDK's
# index.ts imports `#crypto`, which package.json maps to the build
# output; from the sources it is crypto/node.ts.
HARNESS = """
import { register } from "node:module";
const hook = `export async function resolve(specifier, context, next) {
  const target = specifier === "#crypto" ? new URL("./crypto/node.ts", context.parentURL).href : specifier;
  return next(target, context);
}`;
register("data:text/javascript," + encodeURIComponent(hook));
const { explainUrl, verifyUrlSignature } = await import(process.argv[1]);
let text = "";
process.stdin.setEncoding("utf8");
for await (const chunk of process.stdin) text += chunk;
const { now, inputs, verify } = JSON.parse(text);
process.stdout.write(JSON.stringify({
  explanations: inputs.map((input) => explainUrl(input, now)),
  verdicts: verify.map(([secret, input]) => verifyUrlSignature(secret, input)),
}));
"""

# The tag shapes the explainer tells apart, and the URL forms around
# them.
TAG = "a" * 64
SIGNATURES = [
    TAG,
    TAG,
    TAG,
    "0123456789abcdef" * 4,
    f"k1:{TAG}",
    f"key_ID-2:{TAG}",
    f"{'k' * 32}:{TAG}",
    f"{'k' * 33}:{TAG}",
    f":{TAG}",
    "unsafe",
    "UNSAFE",
    "abc123",
    TAG.upper(),
    TAG + "0",
    "s1.AAECAwQFBgcICQoL4rtOMQmUdNPYyz3r",
    "k2:s1._-7dzLuqmYh3ZlVE6KzevG04Kld",
    "s1.",
    "s2.AAAA",
    "k:s9.x",
    "",
]
PREFIXES = [
    "https://img.example.com/",
    "https://img.example.com/",
    "http://127.0.0.1:3000/",
    "/",
    "",
    "HTTPS://Img.Example.com/",
    "1http://x/",
    "https://x",
]
SUFFIXES = ["", "", "", "", "?w=1", "#top", "?a#b", "/", "//"]
PADDING = ["", "", "", "", " ", "\t\n", "\u3000", "\u0085", "\xa0", "\ufeff", "\u200b"]

INTEGERS = [
    "0", "-0", "+0", "00", "1", "-1", "+1", "2", "3", "4", "6", "7", "9", "10", "40", "41",
    "64", "99", "100", "101", "255", "256", "-255", "-256", "400", "401", "999", "1000",
    "1001", "16383", "16384", "-16383", "65535", "999999", "1000000", "0999999", "31536000",
    "31536001", "86400", "86401", "2147483647", "2147483648", "-2147483648", "-2147483649",
    "4294967295", "4294967296", "18446744073709551615", "18446744073709551616",
    "1" + "0" * 25, "0" * 30 + "5", "+" + "0" * 40 + "16383",
]
DECIMALS = [
    "0.5", ".5", "5.", "1e2", "1E+2", "1e-2", "+.5e1", "2e0", "0.0", "-0.0", "1.5", "2.2",
    "512", "512.00001", "512.0001", "512.0000305175781250000000000000001",
    "512.0000305175781249999999999999999", "1e-46", "1e-45", "1.1754942e-38", "1.1754944e-38",
    "3.4028235e38", "3.4028236e38", "3.4028235677973366e38", "3.4028235677973367e38", "1e39",
    "-1e39", "1e999", "1e-999", "36000", "36000.001", "36000.01", "-36000.001",
    "86400.000000000000000001", "86400.0001", "1024", "1024.0000000000001", "1024.0001",
    "-1024", "100.000001", "100.00001", "99.99999", "1.00000001", "1.0000001", "1.50000001",
    "0.30000000000000004", "1e-7", "0.000001", "33.333333333333336", "1" + "0" * 400,
    "0." + "0" * 300 + "1", "1." + "1" * 1500, "5" * 1100 + "e-1098", "0." + "9" * 1200,
    "1" + "0" * 1100 + "1e-1062", "0x10", "inf", "NaN", "infinity", "1_0", "", ".", "e5", "1e",
    "-", "+", "1.5.5", "+-1",
]
EXPIRIES = [
    "0", str(NOW - 1), str(NOW), str(NOW + 1), "1893456000", "253402300799", "253402300800",
    "8640000000000", "8640000000001", "18446744073709551615", "18446744073709551616",
    "000001893456000", "+1893456000", "-1", "1.5", "", "99999999999999999999999999",
]
# Characters and escapes a change puts in, beside the remainder's own.
INSERTS = list("0123456789abcdefxX:,()/+-.%eE_ ;*!~?#") + [
    "é", "ß", "\u093f", "Ⓐ", "\u0345", "\u0301", "\u200d", "\ufeff", "\u3000", "\u0085",
    "\xa0", "\u2028", "\r", "\n", "\t", "\x00", "😀", "中", "%41", "%C3%A9", "%ED%A0%80",
    "%C0%80", "%F4%90%80%80", "%E2%82", "%zz", "%4", "%2B1", "%2C", "%3A", "%28", "%20",
    "crabcrop/", "purge/", "upload/", "preset:thumb/", "exp:1/", "maxage:0/", "debug:focal/",
]
NUMBER = re.compile(r"[+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?")


def corpus_remainders():
    remainders = [c["remainder"] for c in VERDICTS["cases"]]
    remainders += [c["remainder"] for c in VECTORS["cases"]]
    for case in VECTORS["builder_cases"]:
        segment = case.get("segment")
        if segment is not None:
            if case["call"].startswith("filter."):
                segment = f"filters:{segment}"
            remainders.append(f"{segment}/https://example.com/a.jpg")
    return remainders


def split_filters(chain):
    """A filters: list cut where the server cuts it, outside
    parentheses."""
    parts, depth, start = [], 0, 0
    for i, c in enumerate(chain):
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == ":" and depth == 0:
            parts.append(chain[start:i])
            start = i + 1
    parts.append(chain[start:])
    return parts


def mutate(rng, remainder):
    """One change to a remainder, of a kind picked at random."""
    segments = remainder.split("/")
    kind = rng.randrange(11)
    if kind == 0 and len(segments) > 1:
        del segments[rng.randrange(len(segments))]
    elif kind == 1:
        i = rng.randrange(len(segments))
        segments[i:i] = [segments[i]] * rng.choice([1, 1, 2, 16, 99, 100])
    elif kind == 2 and len(segments) > 1:
        i, j = rng.randrange(len(segments)), rng.randrange(len(segments))
        segments[i], segments[j] = segments[j], segments[i]
    elif kind == 3:
        filters = [i for i, s in enumerate(segments) if s.startswith("filters:")]
        if filters:
            i = rng.choice(filters)
            calls = split_filters(segments[i][8:])
            j = rng.randrange(len(calls))
            how = rng.randrange(3)
            if how == 0 and len(calls) > 1:
                del calls[j]
            elif how == 1:
                calls[j:j] = [calls[j]] * rng.choice([1, 15, 16, 99, 100])
            else:
                k = rng.randrange(len(calls))
                calls[j], calls[k] = calls[k], calls[j]
            segments[i] = "filters:" + ":".join(calls)
    elif kind in (4, 5, 6):
        numbers = list(NUMBER.finditer(remainder))
        if numbers:
            hit = rng.choice(numbers)
            pool = rng.choice([INTEGERS, DECIMALS]) if kind != 6 else INTEGERS + DECIMALS
            return remainder[: hit.start()] + rng.choice(pool) + remainder[hit.end() :]
    elif kind == 7:
        if remainder:
            i = rng.randrange(len(remainder))
            return remainder[:i] + rng.choice(INSERTS) + remainder[i + 1 :]
    elif kind == 8:
        i = rng.randrange(len(remainder) + 1)
        return remainder[:i] + rng.choice(INSERTS) + remainder[i:]
    elif kind == 9:
        if remainder:
            i = rng.randrange(len(remainder))
            return remainder[:i] + remainder[i + 1 :]
    else:
        segments.insert(rng.randrange(len(segments) + 1), f"exp:{rng.choice(EXPIRIES)}")
    return "/".join(segments)


def generate_inputs(count, seed):
    rng = random.Random(seed)
    bases = corpus_remainders()
    inputs = [f"https://img.example.com/{TAG}/{r}" for r in bases]
    # Inputs at the 64 KiB cap, a byte under and over it, ASCII and
    # not, and the shapes with no remainder at all.
    head = f"https://x/{TAG}/300x200/https://e.com/"
    for size in (65535, 65536, 65537):
        inputs.append(head + "a" * (size - len(head) - 4) + ".jpg")
        inputs.append(head + "a" * (size - len(head) - 6) + "é.jpg")
    inputs.append(f"https://x/{TAG}/filters:blur({'1' * 64000}z)/https://e.com/a.jpg")
    inputs += ["", " ", "/", "//", "https://x", "https://x/", f"/{TAG}", f"/{TAG}/", "unsafe/a.jpg"]
    inputs += ["s1.AAAA", "k1:s1.AAAA", "/s1.AAAA/300x200/a.jpg", "s1.AAAA?x", "s1.AA/AA"]
    # Every filter bare, with empty parentheses, and with arguments of
    # each shape, and the forms the corpora spell one way only.
    for name in [*cc.FILTER_DESCRIPTIONS, "bogus", "constructor", "__proto__"]:
        for args in ("", "()", "(1)", "(-1)", "(auto)", "(ff0000)", "(a,b)", "(%zz)", "(1,2,3,4)"):
            inputs.append(f"https://img.example.com/{TAG}/filters:{name}{args}/https://example.com/a.jpg")
    for remainder in (
        "crop:400x300:objects/a.jpg",
        "crop:400x300:objectsx/a.jpg",
        "crop:400x300:objects(person/a.jpg",
        "crop:400x300:objects(person):/a.jpg",
        "crop:400x300:objects(person)x/a.jpg",
        "preset:/a.jpg",
        "preset:thumb",
        "exp:1/a.jpg",
    ):
        inputs.append(f"https://img.example.com/{TAG}/{remainder}")
    while len(inputs) < count:
        remainder = rng.choice(bases)
        for _ in range(rng.choice([1, 1, 1, 2, 3, 5])):
            remainder = mutate(rng, remainder)
        url = rng.choice(PREFIXES) + rng.choice(SIGNATURES) + "/" + remainder + rng.choice(SUFFIXES)
        pad = rng.choice(PADDING)
        inputs.append(pad + url + (pad if rng.random() < 0.5 else ""))
    return inputs[:count]


def typescript_shape(explanation):
    """The JSON the TypeScript explainer writes for the same answer:
    absent fields left out, ``key_id`` as ``keyId``."""
    signature = explanation.signature
    if signature is not None:
        signature = {"kind": signature.kind, "value": signature.value, "detail": signature.detail}
        if explanation.signature.key_id is not None:
            signature["keyId"] = explanation.signature.key_id
    segments = []
    for segment in explanation.segments:
        shape = {"raw": segment.raw, "kind": segment.kind, "title": segment.title}
        for name in ("detail", "error", "expired"):
            if getattr(segment, name) is not None:
                shape[name] = getattr(segment, name)
        segments.append(shape)
    return {
        "signature": signature,
        "segments": segments,
        "source": explanation.source,
        "remainder": explanation.remainder,
        "errors": explanation.errors,
    }


def node_with_type_stripping():
    """The node on PATH when it runs TypeScript sources as they are
    (22.18 or newer, or 23.6 or newer), else None."""
    node = shutil.which("node")
    if node is None:
        return None
    try:
        version = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.match(r"v(\d+)\.(\d+)\.", version)
    if match is None:
        return None
    major, minor = int(match.group(1)), int(match.group(2))
    if major >= 24 or (major == 23 and minor >= 6) or (major == 22 and minor >= 18):
        return node
    return None


NODE = node_with_type_stripping()


def verifier_questions(inputs):
    """What the two verifiers are asked about the inputs: each input
    that carries an HMAC tag, as generated and re-signed under SECRET,
    under one secret, right and wrong, and under keyrings that hold the
    right key under the URL's ID, the right secret under another ID,
    or neither; any other input under one secret."""
    questions = []
    for given in inputs:
        explained = cc.explain_url(given, NOW)
        signature = explained.signature
        if signature is None or signature.kind != "hmac":
            questions.append([SECRET, given])
            continue
        remainder = explained.remainder.encode("utf-8", "surrogatepass")
        tag = hmac.new(SECRET.encode("utf-8"), remainder, hashlib.sha256).hexdigest()
        named = signature.key_id or "k1"
        for url in (given, given.replace(signature.value, tag, 1)):
            questions += [
                [SECRET, url],
                ["other", url],
                [{named: SECRET}, url],
                [{named: "other", "k-other": SECRET}, url],
                [{"k-old": "other", "k-new": SECRET}, url],
                [{}, url],
            ]
    return questions


@unittest.skipIf(
    NODE is None and not REQUIRE_NODE, "needs Node 22.18+ (or 23.6+) on PATH to run the TypeScript SDK"
)
class TestDifferential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if NODE is None:
            raise AssertionError(
                "CRABCROP_DIFF_REQUIRE_NODE=1, and no Node 22.18+ (or 23.6+) is on PATH to run the TypeScript SDK"
            )

    def test_python_and_typescript_explainers_and_verifiers_agree(self):
        inputs = generate_inputs(CASES, SEED)
        self.assertEqual(len(inputs), CASES)
        self.assertGreater(len(set(inputs)), CASES * 0.9, "the generator repeats itself")
        questions = verifier_questions(inputs)
        run = subprocess.run(
            [NODE, "--input-type=module", "-e", HARNESS, INDEX_TS.as_uri()],
            input=json.dumps({"now": NOW, "inputs": inputs, "verify": questions}),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        typescript = json.loads(run.stdout)
        self.assertEqual(len(typescript["explanations"]), len(inputs))
        self.assertEqual(len(typescript["verdicts"]), len(questions))
        mismatches = []
        for given, want in zip(inputs, typescript["explanations"]):
            got = typescript_shape(cc.explain_url(given, NOW))
            if got != want:
                mismatches.append((given, got, want))
        shown = "\n".join(
            f"input {given!r}\n  python     {json.dumps(got)}\n  typescript {json.dumps(want)}"
            for given, got, want in mismatches[:5]
        )
        self.assertEqual(
            len(mismatches),
            0,
            f"{len(mismatches)} of {len(inputs)} inputs explain differently (seed {SEED}):\n{shown}",
        )
        answers = {None: 0, True: 0, False: 0}
        disagreements = []
        for (key, given), want in zip(questions, typescript["verdicts"]):
            got = cc.verify_url_signature(key, given)
            answers[got] += 1
            if got is not want:
                disagreements.append((key, given, got, want))
        shown = "\n".join(
            f"verify_url_signature({key!r}, {given!r})\n  python {got}, typescript {want}"
            for key, given, got, want in disagreements[:5]
        )
        self.assertEqual(
            len(disagreements),
            0,
            f"{len(disagreements)} of {len(questions)} verifier answers differ (seed {SEED}):\n{shown}",
        )
        # The questions reach every answer, True included.
        self.assertGreater(min(answers.values()), len(inputs) // 20, answers)

    def test_class_names_take_the_letters_and_digits_the_typescript_explainer_takes(self):
        # The class-name table is Unicode 17.0's, written out whole; on
        # a Node with the same Unicode tables, the TypeScript explainer's
        # [\p{Alphabetic}\p{N}] is exactly that set.
        version = ".".join(map(str, cc._grammar._ALPHANUMERIC_UNICODE))
        run = subprocess.run([NODE, "-p", "process.versions.unicode"], capture_output=True, text=True, timeout=60)
        if run.stdout.strip() != version:
            self.skipTest(f"Node reads Unicode {run.stdout.strip()}, the table {version}")
        script = (
            "const re = /^[\\p{Alphabetic}\\p{N}]$/u; const out = [];"
            "for (let c = 0; c <= 0x10ffff; c++) {"
            " if ((c < 0xd800 || c > 0xdfff) && re.test(String.fromCodePoint(c))) out.push(c); }"
            "process.stdout.write(JSON.stringify(out));"
        )
        run = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr)
        typescript = json.loads(run.stdout)
        python = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF and cc._grammar.is_alphanumeric(chr(c))]
        self.assertGreater(len(python), 140000)
        self.assertEqual(python, typescript)

    def test_the_inputs_reach_every_segment_kind_and_verdict(self):
        # A generator that drifted into one corner would pass the
        # comparison above while testing little. About a fifth of the
        # inputs parse, more of them at the default size, where the
        # corpora's own remainders are a larger share.
        count = max(CASES, DEFAULT_CASES)
        inputs = generate_inputs(count, SEED)
        kinds, titles, errors, ok = set(), set(), 0, 0
        for given in inputs:
            out = cc.explain_url(given, NOW)
            kinds.update(s.kind for s in out.segments)
            titles.update(s.title for s in out.segments)
            errors += bool(out.errors)
            ok += not out.errors
        self.assertEqual(kinds, {"expiry", "maxage", "debug", "preset", "operation"})
        self.assertGreaterEqual(len(titles), 15, titles)
        self.assertGreater(errors, count // 10)
        self.assertGreater(ok, count // 10)


if __name__ == "__main__":
    unittest.main()
