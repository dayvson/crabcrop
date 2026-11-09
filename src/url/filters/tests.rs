//! The table's own tests. They check that the table agrees
//! with itself: every row round-trips through the parser and
//! the canonical spelling, and every kind refuses the
//! spellings one step past its bounds with the code its row
//! declares. The bounds and codes themselves are pinned
//! outside the table, by spec/operation-vectors.json and the
//! url tests; the own-code filters are also pinned here, by
//! literal errors. The table also names exactly the filters
//! docs/filters.md documents.

use std::collections::BTreeSet;
use std::fmt::Display;
use std::str::FromStr;

use super::*;
use crate::url::{Operation, parse_operations};

/// A row of the table, as the tests walk it.
pub(super) struct Spec {
    pub(super) name: &'static str,
    /// The argument kinds, in order (none for a bare name);
    /// None for a row with a sub-grammar of its own.
    pub(super) args: Option<&'static [&'static dyn Samples]>,
    /// Read, never written: another row spells the filter.
    pub(super) shorthand: bool,
}

/// Spellings a kind accepts and refuses.
pub(super) trait Samples {
    /// Spellings it reads, the first one usable as a default
    /// while another argument varies.
    fn accepted(&self) -> Vec<String>;
    /// Spellings it refuses.
    fn refused(&self) -> Vec<String>;
}

/// The spellings one step past a numeric bound: the next
/// integer, or the next representable float.
trait Neighbors: Copy + Display {
    fn below(self) -> String;
    fn beyond(self) -> String;
}

macro_rules! integer_neighbors {
    ($($t:ty),*) => {$(
        impl Neighbors for $t {
            fn below(self) -> String {
                (i128::from(self) - 1).to_string()
            }
            fn beyond(self) -> String {
                (i128::from(self) + 1).to_string()
            }
        }
    )*};
}

integer_neighbors!(u8, u16, u32, u64, i32);

impl Neighbors for f32 {
    fn below(self) -> String {
        self.next_down().to_string()
    }
    fn beyond(self) -> String {
        // Display spells the shortest text that reads back
        // to the same float, and f32::MAX's next is infinity.
        self.next_up().to_string()
    }
}

impl<T: FromStr + PartialOrd + Neighbors> Samples for Number<T> {
    fn accepted(&self) -> Vec<String> {
        let mut out = vec![self.spell(&self.max)];
        if self.open {
            out.push(self.min.beyond());
        } else {
            out.insert(0, self.spell(&self.min));
        }
        out
    }
    fn refused(&self) -> Vec<String> {
        let mut out = vec![self.min.below(), self.max.beyond()];
        if self.open {
            out.push(self.spell(&self.min));
        }
        out
    }
}

impl<T: FromStr + PartialEq + Neighbors> Samples for OneOf<T> {
    fn accepted(&self) -> Vec<String> {
        self.0.iter().map(|v| self.spell(v)).collect()
    }
    fn refused(&self) -> Vec<String> {
        self.0
            .iter()
            .flat_map(|v| [v.below(), v.beyond()])
            .filter(|text| self.read(text).is_none())
            .collect()
    }
}

impl<T: Copy> Samples for Keyword<T> {
    fn accepted(&self) -> Vec<String> {
        self.values.iter().map(|v| self.spell(v)).collect()
    }
    fn refused(&self) -> Vec<String> {
        self.accepted()
            .iter()
            .map(|name| name.to_uppercase())
            .collect()
    }
}

impl Samples for Hex {
    fn accepted(&self) -> Vec<String> {
        vec!["00ff7f".to_string(), "ABCDEF".to_string()]
    }
    fn refused(&self) -> Vec<String> {
        ["00ff7", "00ff7f0", "00ff7g", "+0ff7f"]
            .map(String::from)
            .to_vec()
    }
}

impl Samples for ClassName {
    fn accepted(&self) -> Vec<String> {
        vec!["person".to_string(), "traffic%20light".to_string()]
    }
    fn refused(&self) -> Vec<String> {
        ["%ZZ", "a*b", "%20"].map(String::from).to_vec()
    }
}

impl Samples for Angle {
    fn accepted(&self) -> Vec<String> {
        ["90", "-36000", "36000", "45.5"].map(String::from).to_vec()
    }
    fn refused(&self) -> Vec<String> {
        ["36000.5", "1e2", ".5", "+5", "-"]
            .map(String::from)
            .to_vec()
    }
}

/// Spellings for the rows with a sub-grammar of their own,
/// which no kind can generate: (name, accepted, refused).
const CUSTOM_SAMPLES: &[(&str, &[&str], &[&str])] = &[
    (
        "curve",
        &[
            "curve(rgb,0,0,255,255)",
            "curve(r,0,64,128,100,255,255)",
            "curve(b,+0,000,255,255)",
        ],
        &[
            "curve(rgb,0,0)",
            "curve(x,0,0,255,255)",
            "curve(g,9,0,9,255)",
        ],
    ),
    (
        "convolution",
        &[
            "convolution(0,-1,0,-1,5,-1,0,-1,0,0)",
            "convolution(0.50,1,1,1,1,1,1,1,-1.0,1)",
            "convolution(1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1,1)",
        ],
        &["convolution(1,1,1)", "convolution(1,1,1,1,1,1,1,1,1,2)"],
    ),
    (
        "format",
        &[
            "format(webp)",
            "format(jpg)",
            "format(jpeg,subsampling:444,progressive)",
            "format(png,interlace)",
            "format(webp,sns:50,method:6)",
        ],
        &["format(bmp)", "format(avif,progressive)"],
    ),
    (
        "quality",
        &[
            "quality(80)",
            "quality(auto)",
            "quality(webp:75,80,jpeg:auto)",
        ],
        &["quality(0)", "quality(mp4:80)", "quality(80,90)"],
    ),
    (
        "watermark",
        &[
            "watermark(https%3A%2F%2Fexample.com%2Flogo.png,-10,-10,50)",
            "watermark(a,b,0,0,100)",
        ],
        &[
            "watermark(a,0,0,101)",
            "watermark(,0,0,50)",
            "watermark(a,0,0)",
        ],
    ),
    (
        "overlay",
        &[
            "overlay(https%3A%2F%2Fexample.com%2Fo.png,0,0,80,multiply)",
            "overlay(o.png,-5,5,100,screen,400)",
        ],
        &[
            "overlay(o.png,0,0,80,darken)",
            "overlay(o.png,0,0,80,normal,0)",
        ],
    ),
    (
        "text",
        &[
            "text(Hello%2C%20world,24,ffffff,bottom-right)",
            "text(a,b,1000,ffffff80,center)",
        ],
        &[
            "text(a,0,ffffff,center)",
            "text(a,24,ffffff,middle)",
            "text(,24,ffffff,center)",
        ],
    ),
    (
        "studio",
        &[
            "studio(product)",
            "studio(product,pad:10)",
            "studio(product,bg:FF8000,shadow:strong,pad:0)",
            "studio(scene,pad:40)",
        ],
        &[
            "studio(product,pad:41)",
            "studio(scene,bg:white)",
            "studio(lifestyle)",
        ],
    ),
];

const SOURCE: &str = "https://example.com/a.jpg";

/// Parse `filters:<text>`, which must hold one filter.
fn parse_one(text: &str) -> Result<Filter, ParseError> {
    let input = format!("filters:{text}/{SOURCE}");
    let (ops, source) = parse_operations(&input)?;
    assert_eq!(source, SOURCE, "{text}");
    match &ops[..] {
        [Operation::Filters(filters)] if filters.len() == 1 => Ok(filters[0].clone()),
        other => panic!("{text} parsed as {other:?}"),
    }
}

/// A filter with the entries of a per-format quality list in
/// format-name order: the list is a set, kept in the order it
/// was written, and only its spelling sorts it.
fn sorted(filter: Filter) -> Filter {
    match filter {
        Filter::Quality(Quality::PerFormat(mut entries)) => {
            entries.sort_by_key(|(format, _)| format.map(format_text));
            Filter::Quality(Quality::PerFormat(entries))
        }
        other => other,
    }
}

/// Parse `text`, spell the filter, and parse the spelling:
/// the same filter both times, and the spelling its own
/// canonical form. Returns the spelling.
fn round_trip(text: &str) -> String {
    let filter = parse_one(text).unwrap_or_else(|e| panic!("{text} should parse: {e}"));
    let canonical = filter_text(&filter);
    let again = parse_one(&canonical)
        .unwrap_or_else(|e| panic!("{text} spells as {canonical}, which does not parse: {e}"));
    assert_eq!(
        sorted(again.clone()),
        sorted(filter),
        "{text} spells as {canonical}, which reads back differently"
    );
    assert_eq!(
        filter_text(&again),
        canonical,
        "{canonical} is not its own canonical form"
    );
    canonical
}

/// `text` is refused with the error its row declares.
fn assert_refused(name: &str, text: &str) {
    let expected = argument_error(name)(format!("filters:{text}"));
    assert_eq!(parse_one(text), Err(expected), "{text} should be refused");
}

#[test]
fn every_row_round_trips_and_refuses_past_its_bounds() {
    for spec in FILTER_SPECS {
        let name = spec.name;
        let Some(kinds) = spec.args else {
            let (_, accepted, refused) = CUSTOM_SAMPLES
                .iter()
                .find(|(n, _, _)| *n == name)
                .unwrap_or_else(|| {
                    panic!("{name} has its own sub-grammar: add it to CUSTOM_SAMPLES")
                });
            for text in *accepted {
                assert!(round_trip(text).starts_with(&format!("{name}(")), "{text}");
            }
            for text in *refused {
                assert_refused(name, text);
            }
            continue;
        };
        if kinds.is_empty() {
            for text in [name.to_string(), format!("{name}()")] {
                let canonical = round_trip(&text);
                if !spec.shorthand {
                    assert_eq!(canonical, name);
                }
            }
            continue;
        }
        let defaults: Vec<String> = kinds.iter().map(|k| k.accepted()[0].clone()).collect();
        for (i, kind) in kinds.iter().enumerate() {
            let with = |value: &str| {
                let mut args = defaults.clone();
                args[i] = value.to_string();
                format!("{name}({})", args.join(","))
            };
            for value in kind.accepted() {
                let text = with(&value);
                let canonical = round_trip(&text);
                assert!(
                    canonical.starts_with(&format!("{name}(")),
                    "{text} spells as {canonical}"
                );
                if !spec.shorthand {
                    assert_eq!(
                        canonical.matches(',').count(),
                        kinds.len() - 1,
                        "{canonical}"
                    );
                }
            }
            for value in kind.refused() {
                assert_refused(name, &with(&value));
            }
        }
    }
}

#[test]
fn names_answer_invalid_filters_in_the_wrong_shape() {
    let invalid = |text: &str| Err(ParseError::InvalidFilters(format!("filters:{text}")));
    let names: BTreeSet<&str> = FILTER_SPECS.iter().map(|s| s.name).collect();
    for name in names {
        let rows = || FILTER_SPECS.iter().filter(|s| s.name == name);
        // A filter that needs arguments refuses its bare name
        // with invalid_filters, whatever its own code.
        if !rows().any(|s| s.args.is_some_and(|a| a.is_empty())) {
            assert_eq!(parse_one(name), invalid(name));
            assert_eq!(
                parse_one(&format!("{name}()")),
                invalid(&format!("{name}()"))
            );
        }
        // A filter that takes none refuses an argument list.
        if rows().all(|s| s.args.is_some_and(|a| a.is_empty())) {
            let text = format!("{name}(1)");
            assert_eq!(parse_one(&text), invalid(&text));
        }
    }
    for text in ["frobnicate", "frobnicate(1)", "Grayscale", "(1)"] {
        assert_eq!(parse_one(text), invalid(text));
    }
}

/// The rows with an error code of their own answer it for a
/// bad argument list, pinned here by literal messages rather
/// than through the table's own `argument_error`.
#[test]
fn own_code_rows_answer_their_own_code() {
    let cases: [(&str, &str, &[&str]); 4] = [
        (
            "invalid_noise",
            "invalid noise filter in segment",
            &["noise(101)", "noise(-1)", "noise(5,5)"],
        ),
        (
            "invalid_curve",
            "invalid curve filter in segment",
            &["curve(rgb,0,0)", "curve(rgb,0,0,256,255)"],
        ),
        (
            "invalid_convolution",
            "invalid convolution filter in segment",
            &["convolution(1,1,1)", "convolution(1,1,1,1,1,1,1,1,1,2)"],
        ),
        (
            "invalid_filters",
            "invalid filters segment",
            &["contrast(101)", "noise", "curve()"],
        ),
    ];
    for (code, message, texts) in cases {
        for text in texts {
            let error = parse_one(text).expect_err(text);
            assert_eq!(error.code(), code, "{text}");
            assert_eq!(error.to_string(), format!("{message}: filters:{text}"));
        }
    }
}

/// Whether `s` reads as a filter name: a lowercase letter,
/// then lowercase letters, digits, and underscores.
fn is_name(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// The cells of a Markdown table row, trimmed.
fn cells(line: &str) -> Vec<&str> {
    line.trim()
        .strip_prefix('|')
        .and_then(|l| l.strip_suffix('|'))
        .map(|l| l.split('|').map(str::trim).collect())
        .unwrap_or_default()
}

/// The filters a page documents, read the way
/// scripts/check_docs.py reads them: the first cell of every
/// row of a table headed `Filter` or `Directive`, and every
/// `filters:<name>(` line (the watermark, overlay, and text
/// sections).
fn documented_filters(doc: &str) -> BTreeSet<&str> {
    let mut names = BTreeSet::new();
    let mut lines = doc.lines().peekable();
    while let Some(line) = lines.next() {
        if let Some((name, _)) = line
            .strip_prefix("filters:")
            .and_then(|r| r.split_once('('))
            && is_name(name)
        {
            names.insert(name);
        }
        let headed = matches!(cells(line).first(), Some(&("Filter" | "Directive")));
        if headed
            && lines
                .next_if(|l| l.trim_start().starts_with("| ---"))
                .is_some()
        {
            while let Some(row) = lines.next_if(|l| l.trim_start().starts_with('|')) {
                let first = cells(row)[0];
                let name = first
                    .strip_prefix('`')
                    .and_then(|n| n.strip_suffix('`'))
                    .filter(|n| is_name(n))
                    .unwrap_or_else(|| panic!("cannot read the filter row {row}"));
                names.insert(name);
            }
        }
    }
    names
}

#[test]
fn table_names_the_filters_the_docs_document() {
    let table: BTreeSet<&str> = FILTER_SPECS.iter().map(|s| s.name).collect();
    let documented = documented_filters(include_str!("../../../docs/filters.md"));
    let undocumented: Vec<_> = table.difference(&documented).collect();
    let unknown: Vec<_> = documented.difference(&table).collect();
    assert!(
        undocumented.is_empty() && unknown.is_empty(),
        "in the table but not docs/filters.md: {undocumented:?}; \
         in docs/filters.md but not the table: {unknown:?}"
    );
}
