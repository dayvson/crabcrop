//! Issue #741 (step 4) splits the filter enum by who
//! applies a filter. The split must not change what a URL
//! means: these tests hold the current parser, canonical
//! form, cache key form, directive analysis and default
//! merge to main's, frozen in [`frozen`], over every
//! remainder the corpora hold and over chains generated
//! from the filter table's own samples. Each remainder must
//! parse to the same verdict (the same error, or the same
//! source and metadata), and then canonicalize, key,
//! analyze, and merge under each of [`DEFAULTS`] to the
//! same bytes and the same directives.

mod frozen;

use super::FILTER_SPECS;
use super::tests::CUSTOM_SAMPLES;
use crate::url::{Operation, Presets, RequestDirectives};

/// Server-wide default operations the merge runs under: the
/// singleton directives a request's own replace (format,
/// quality, fill, dpr, max_bytes), the ones both keep
/// (no_upscale, keep_profile, palette), sizes and plain
/// filters, and the upgrade corpus's presets.
const DEFAULTS: &[&str] = &[
    "filters:format(webp):quality(80)",
    "filters:fill(ff8000):dpr(2):max_bytes(50000)",
    "300x200/filters:grayscale:format(avif):quality(auto):no_upscale",
    "fit-in:800x600/filters:keep_profile:palette(64):sharpen(1)",
    "crop:400x225/filters:quality(75)",
    "300x200/filters:format(webp)",
];

/// The source the builder cases' segments and many
/// generated chains end in.
const SOURCE: &str = "https://example.com/a.jpg";

/// What was compared, for the counts the tests print.
#[derive(Default, Debug)]
struct Tally {
    remainders: usize,
    parsed: usize,
    refused: usize,
    merges: usize,
}

/// The defaults of [`DEFAULTS`], parsed by both grammars.
fn defaults() -> Vec<(Vec<Operation>, Vec<frozen::Operation>)> {
    DEFAULTS
        .iter()
        .map(|text| {
            let terminated = format!("{text}/example.com/defaults");
            let (now, source, _, _) = crate::url::parse_operations_with_expiry(&terminated)
                .unwrap_or_else(|e| panic!("{text}: {e}"));
            assert_eq!(source, "example.com/defaults", "{text}");
            let (then, _, _, _) = frozen::parse_operations_with_expiry(&terminated)
                .unwrap_or_else(|e| panic!("{text}: {e}"));
            (now, then)
        })
        .collect()
}

/// Hold one remainder to main's grammar, adding what
/// differs to `bad`.
fn check(
    remainder: &str,
    defaults: &[(Vec<Operation>, Vec<frozen::Operation>)],
    tally: &mut Tally,
    bad: &mut Vec<String>,
) {
    tally.remainders += 1;
    let key = crate::url::canonical_key_form(remainder);
    let main_key = frozen::canonical_key_form(remainder);
    if key != main_key {
        bad.push(format!(
            "{remainder:?}: key {key:?}, main keyed {main_key:?}"
        ));
    }
    let parsed = crate::url::parse_operations_with_expiry(remainder);
    let main_parsed = frozen::parse_operations_with_expiry(remainder);
    let ((ops, source, expiry, max_age), (main_ops, main_source, main_expiry, main_max_age)) =
        match (parsed, main_parsed) {
            (Ok(now), Ok(then)) => (now, then),
            (Err(e), Err(main_e)) => {
                tally.refused += 1;
                if e != main_e {
                    bad.push(format!(
                        "{remainder:?}: refused with {e:?}, main with {main_e:?}"
                    ));
                }
                return;
            }
            (now, then) => {
                bad.push(format!(
                    "{remainder:?}: parsed as {:?}, main as {:?}",
                    now.map(|p| p.1),
                    then.map(|p| p.1)
                ));
                return;
            }
        };
    tally.parsed += 1;
    if (source, expiry, max_age) != (main_source, main_expiry, main_max_age) {
        bad.push(format!(
            "{remainder:?}: source and metadata {:?}, main {:?}",
            (source, expiry, max_age),
            (main_source, main_expiry, main_max_age)
        ));
    }
    let form = crate::url::canonical_form(&ops, source);
    let main_form = frozen::canonical_form(&main_ops, source);
    if form != main_form {
        bad.push(format!("{remainder:?}: form {form:?}, main {main_form:?}"));
    }
    let directives = RequestDirectives::analyze(&ops);
    let main_directives = frozen::analyze(&main_ops);
    if directives != main_directives {
        bad.push(format!(
            "{remainder:?}: directives {directives:?}, main {main_directives:?}"
        ));
    }
    for (index, (now, then)) in defaults.iter().enumerate() {
        tally.merges += 1;
        let merged = crate::url::merge_default_ops(now, ops.clone());
        let main_merged = frozen::merge_default_ops(then, main_ops.clone());
        let form = crate::url::canonical_form(&merged, source);
        let main_form = frozen::canonical_form(&main_merged, source);
        if form != main_form {
            bad.push(format!(
                "{remainder:?} under {}: form {form:?}, main {main_form:?}",
                DEFAULTS[index]
            ));
        }
        if RequestDirectives::analyze(&merged) != frozen::analyze(&main_merged) {
            bad.push(format!(
                "{remainder:?} under {}: the merged directives differ",
                DEFAULTS[index]
            ));
        }
    }
}

/// Fail with every difference, or print what was compared.
fn conclude(what: &str, tally: &Tally, bad: &[String]) {
    assert!(
        bad.is_empty(),
        "{what}: {} differences from main in {} remainders:\n{}",
        bad.len(),
        tally.remainders,
        bad.join("\n")
    );
    eprintln!("{what}: {tally:?}");
}

/// The upgrade corpus's presets, as the server loads them.
fn upgrade_presets() -> Presets {
    include_str!("../../../tests/upgrade/presets.txt")
        .lines()
        .filter(|line| !line.trim_start().starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let (name, text) = line.split_once('=').expect("name = definition");
            let preset = text.trim().parse().expect("the definition parses");
            (name.trim().to_string(), preset)
        })
        .collect()
}

/// Every remainder the corpora hold, by corpus: the signing
/// vectors (with their operations joined to their sources),
/// the generated operation vectors (with the builder cases'
/// segments put in front of a source), the explainer
/// verdicts, and the upgrade corpus's paths without their
/// signatures (with presets expanded as well as without).
fn corpora() -> Vec<(&'static str, Vec<String>)> {
    let json = |text: &str| -> serde_json::Value { serde_json::from_str(text).unwrap() };
    let text = |value: &serde_json::Value| value.as_str().map(str::to_string);

    let mut signing = Vec::new();
    for case in json(include_str!("../../../spec/test-vectors.json"))
        .as_array()
        .unwrap()
    {
        signing.extend(text(&case["remainder"]));
        if let (Some(ops), Some(source)) = (text(&case["operations"]), text(&case["source"])) {
            signing.push(if ops.is_empty() {
                source
            } else {
                format!("{ops}/{source}")
            });
        }
    }

    let vectors = json(include_str!("../../../spec/operation-vectors.json"));
    let mut operations: Vec<String> = vectors["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|case| text(&case["remainder"]))
        .collect();
    for case in vectors["builder_cases"].as_array().unwrap() {
        for field in ["segment", "canonical"] {
            if let Some(segment) = text(&case[field]) {
                operations.push(format!("{segment}/{SOURCE}"));
            }
        }
    }

    let verdicts: Vec<String> = json(include_str!("../../../spec/explain-verdicts.json"))["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|case| text(&case["remainder"]))
        .collect();

    let presets = upgrade_presets();
    let mut upgrade = Vec::new();
    for entry in json(include_str!("../../../tests/upgrade/corpus.json"))["entries"]
        .as_array()
        .unwrap()
    {
        let path = text(&entry["path"]).expect("every entry has a path");
        let remainder = path
            .trim_start_matches('/')
            .split_once('/')
            .map_or(String::new(), |(_, rest)| rest.to_string());
        if let Ok(expanded) = crate::url::expand_presets(&remainder, &presets, false)
            && expanded != remainder
        {
            upgrade.push(expanded);
        }
        upgrade.push(remainder);
        upgrade.extend(text(&entry["remainder"]));
    }

    vec![
        ("spec/test-vectors.json", signing),
        ("spec/operation-vectors.json", operations),
        ("spec/explain-verdicts.json", verdicts),
        ("tests/upgrade/corpus.json", upgrade),
    ]
}

#[test]
fn the_corpora_parse_and_canonicalize_as_main_did() {
    let defaults = defaults();
    let mut total = Tally::default();
    for (corpus, remainders) in corpora() {
        assert!(!remainders.is_empty(), "{corpus} holds no remainder");
        let mut tally = Tally::default();
        let mut bad = Vec::new();
        for remainder in &remainders {
            check(remainder, &defaults, &mut tally, &mut bad);
        }
        conclude(corpus, &tally, &bad);
        total.remainders += tally.remainders;
        total.parsed += tally.parsed;
        total.refused += tally.refused;
        total.merges += tally.merges;
    }
    // The corpora as they stand: 66 signing remainders, 750
    // generated cases with 208 builder segments, 320
    // verdicts, and 39 upgrade remainders.
    assert!(total.remainders >= 1_383, "{total:?}");
    assert!(total.parsed > 900 && total.refused > 450, "{total:?}");
    eprintln!("all corpora: {total:?}");
}

/// A deterministic stream of numbers (xorshift64*), so a
/// difference names a remainder that reproduces it.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a>(&mut self, items: &'a [String]) -> &'a str {
        &items[self.below(items.len())]
    }
}

/// The filter spellings the table's own tests read and
/// refuse: each row's arguments at every accepted and
/// refused sample, the others at their defaults; a bare
/// row with and without empty parentheses, and with an
/// argument; and the custom rows' samples. Returns the
/// accepted spellings, then the rest.
fn filter_spellings() -> (Vec<String>, Vec<String>) {
    let mut accepted = Vec::new();
    let mut refused = Vec::new();
    for spec in FILTER_SPECS {
        let name = spec.name;
        let Some(kinds) = spec.args else {
            let (_, ok, bad) = CUSTOM_SAMPLES
                .iter()
                .find(|(n, _, _)| *n == name)
                .unwrap_or_else(|| panic!("{name} has no custom samples"));
            accepted.extend(ok.iter().map(|t| t.to_string()));
            refused.extend(bad.iter().map(|t| t.to_string()));
            continue;
        };
        if kinds.is_empty() {
            accepted.extend([name.to_string(), format!("{name}()")]);
            refused.push(format!("{name}(1)"));
            continue;
        }
        let defaults: Vec<String> = kinds.iter().map(|k| k.accepted()[0].clone()).collect();
        for (i, kind) in kinds.iter().enumerate() {
            let with = |value: &str| {
                let mut args = defaults.clone();
                args[i] = value.to_string();
                format!("{name}({})", args.join(","))
            };
            accepted.extend(kind.accepted().iter().map(|v| with(v)));
            refused.extend(kind.refused().iter().map(|v| with(v)));
        }
    }
    (accepted, refused)
}

/// `filters:` segments no row reads, one way or another.
const MALFORMED: &[&str] = &[
    "filters:",
    "filters::grayscale",
    "filters:grayscale:",
    "filters:grayscale::invert",
    "filters:Grayscale",
    "filters:blur(2",
    "filters:blur(2))",
    "filters:blur((2))",
    "filters:blur()",
    "filters:(2)",
    "filters:blur(2)x",
    "filters:studio(product,pad:10:x)",
    "filters:format(webp):",
];

/// The operations every generated chain draws from besides
/// its `filters:` segments: each segment of a corpus
/// remainder that the parser reads as an operation or as
/// metadata. Returns the ones it accepts, then the ones it
/// refuses.
fn other_segments(corpora: &[(&'static str, Vec<String>)]) -> (Vec<String>, Vec<String>) {
    let mut segments: Vec<String> = corpora
        .iter()
        .flat_map(|(_, remainders)| remainders.iter())
        .flat_map(|remainder| remainder.split('/'))
        .filter(|seg| {
            !seg.starts_with("filters:")
                && (seg.starts_with("exp:")
                    || seg.starts_with("maxage:")
                    || crate::url::parse_operation(seg).is_some())
        })
        .map(str::to_string)
        .collect();
    segments.sort();
    segments.dedup();
    segments
        .into_iter()
        .partition(|seg| crate::url::parse_operations(&format!("{seg}/{SOURCE}")).is_ok())
}

/// The sources generated chains end in.
const SOURCES: &[&str] = &[
    SOURCE,
    "example.com/a.png",
    "https://example.com/path/to/b.webp",
    "root:photos/crab-photo.jpg",
    "https://example.com/filters:grayscale/a.jpg",
];

/// How many chains [`generated_chains_parse_and_canonicalize_as_main_did`] builds.
const GENERATED: usize = 50_000;

#[test]
fn generated_chains_parse_and_canonicalize_as_main_did() {
    let (accepted, refused) = filter_spellings();
    let malformed: Vec<String> = MALFORMED.iter().map(|s| s.to_string()).collect();
    let (others, refused_others) = other_segments(&corpora());
    let sources: Vec<String> = SOURCES.iter().map(|s| s.to_string()).collect();
    let pools = (
        accepted.len(),
        refused.len(),
        others.len(),
        refused_others.len(),
    );
    eprintln!("filters accepted and refused, other segments accepted and refused: {pools:?}");
    assert!(
        pools.0 > 150 && pools.1 > 150 && pools.2 > 100 && pools.3 > 50,
        "{pools:?}"
    );
    let defaults = defaults();
    let mut rng = Rng(0x0741_0004_5eed_cafe);
    let mut tally = Tally::default();
    let mut bad = Vec::new();
    for _ in 0..GENERATED {
        let mut segments: Vec<String> = Vec::new();
        // One to five segments: mostly filters and the
        // operations the parser accepts, now and then one
        // it refuses.
        for _ in 0..=rng.below(5) {
            match rng.below(40) {
                0 => segments.push(rng.pick(&malformed).to_string()),
                1 => segments.push(rng.pick(&refused_others).to_string()),
                2..=17 => segments.push(rng.pick(&others).to_string()),
                _ => {
                    let filters: Vec<&str> = (0..=rng.below(4))
                        .map(|_| {
                            if rng.below(40) == 0 {
                                rng.pick(&refused)
                            } else {
                                rng.pick(&accepted)
                            }
                        })
                        .collect();
                    segments.push(format!("filters:{}", filters.join(":")));
                }
            }
        }
        segments.push(rng.pick(&sources).to_string());
        check(&segments.join("/"), &defaults, &mut tally, &mut bad);
    }
    conclude("generated chains", &tally, &bad);
    assert_eq!(tally.remainders, GENERATED);
    assert!(
        tally.parsed > GENERATED / 2 && tally.refused > GENERATED / 10,
        "{tally:?}"
    );
}
