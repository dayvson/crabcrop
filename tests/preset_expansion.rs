//! Preset expansion is counted before it is built, from
//! outside the crate and under a recording allocator. A
//! request that repeats a preset reference is refused for its
//! operation count, or for the length its references would
//! expand to, without the expanded text being allocated: the
//! cost of the refusal follows the request's own length, not
//! the definitions it names.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use crabcrop::url::{
    MAX_EXPANDED_LEN, MAX_OPERATIONS, ParseError, Presets, expand_presets, parse_operations,
};

/// The global allocator of this test binary: the system
/// allocator, plus a running total of the bytes requested on
/// a thread that armed it.
struct Recorder;

thread_local! {
    /// `None` while disarmed, else the bytes requested so far.
    /// A const-initialized Cell needs no lazy setup and no
    /// destructor, so the allocator can touch it.
    static REQUESTED: Cell<Option<usize>> = const { Cell::new(None) };
}

impl Recorder {
    fn record(size: usize) {
        let _ = REQUESTED.try_with(|total| {
            if let Some(seen) = total.get() {
                total.set(Some(seen.saturating_add(size)));
            }
        });
    }
}

// SAFETY: every method forwards to `System` unchanged.
unsafe impl GlobalAlloc for Recorder {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Recorder = Recorder;

/// Run `f` with the recorder armed on this thread; return its
/// result and the bytes it requested in total.
fn requested<T>(f: impl FnOnce() -> T) -> (T, usize) {
    REQUESTED.with(|total| total.set(Some(0)));
    let out = f();
    let total = REQUESTED.with(|total| total.replace(None)).unwrap_or(0);
    (out, total)
}

/// What a refusal may allocate: room for an error that
/// carries a short segment, nothing in proportion to the
/// request or the definitions it names.
const SMALL: usize = 1024;

fn presets(entries: &[(&str, String)]) -> Presets {
    entries
        .iter()
        .map(|(name, text)| (name.to_string(), text.parse().expect("a valid preset")))
        .collect()
}

/// A 100-operation preset of 5x5 convolutions, about 11.5 KB.
fn convolutions() -> String {
    let kernel = format!("convolution({},1)", vec!["0.5"; 25].join(","));
    format!("filters:{}", vec![kernel; MAX_OPERATIONS].join(":"))
}

/// A 100-operation preset of inverts, 707 bytes.
fn inverts() -> String {
    format!("filters:{}", vec!["invert"; MAX_OPERATIONS].join(":"))
}

/// `refs` references to `name`, then a source: 7,000
/// one-letter references make a 63 KB remainder, about as many
/// as one request target can carry.
fn repeated(name: &str, refs: usize) -> String {
    format!(
        "{}example.com/a.jpg",
        format!("preset:{name}/").repeat(refs)
    )
}

#[test]
fn repeated_references_are_refused_before_the_expansion_is_built() {
    let table = presets(&[("c", convolutions()), ("i", inverts())]);
    for name in ["c", "i"] {
        let remainder = repeated(name, 7_000);
        for only_presets in [true, false] {
            let (result, bytes) = requested(|| expand_presets(&remainder, &table, only_presets));
            // The refusal the parsed list got before, with the
            // same count and text.
            let err = result.unwrap_err();
            assert_eq!(err, ParseError::TooManyOperations(700_000), "{name}");
            assert_eq!(err.code(), "unsupported_operation");
            assert_eq!(
                err.to_string(),
                "request has 700000 operations, over the 100 operation limit"
            );
            // Nothing in proportion to the expansion (5 MB for
            // the inverts, 80 MB for the convolutions) was
            // allocated on the way.
            assert!(bytes < SMALL, "{name}: {bytes} bytes requested");
        }
    }
}

#[test]
fn an_expansion_past_the_byte_cap_is_refused_unbuilt() {
    // One operation, but a long one: a watermark whose source
    // URL runs to 4 KB. A hundred references stay within the
    // operation cap and would expand to about 400 KB.
    let long = format!(
        "filters:watermark(https%3A%2F%2Fexample.com%2F{}.png,0,0,100)",
        "w".repeat(4_000)
    );
    let table = presets(&[("w", long.clone())]);
    let remainder = repeated("w", MAX_OPERATIONS);
    let (result, bytes) = requested(|| expand_presets(&remainder, &table, true));
    let expanded_len = (long.len() + 1) * MAX_OPERATIONS + "example.com/a.jpg".len();
    assert!(expanded_len > MAX_EXPANDED_LEN);
    let err = result.unwrap_err();
    assert_eq!(err, ParseError::ExpansionTooLong(expanded_len));
    assert_eq!(err.code(), "unsupported_operation");
    assert_eq!(
        err.to_string(),
        format!("request expands to {expanded_len} bytes, over the 262144 byte limit")
    );
    assert!(bytes < SMALL, "{bytes} bytes requested");
    // Fewer references fit, and expand in full.
    let refs = MAX_EXPANDED_LEN / (long.len() + 1) - 1;
    let expanded = expand_presets(&repeated("w", refs), &table, true).unwrap();
    assert_eq!(
        expanded,
        format!("{}example.com/a.jpg", format!("{long}/").repeat(refs))
    );
}

#[test]
fn expansions_within_the_caps_are_unchanged() {
    let card = "crop:1200x630:smart/filters:format(webp):quality(80):sharpen(0.5)";
    let table = presets(&[
        ("card", card.to_string()),
        ("thumb", "300x200".to_string()),
        ("i", inverts()),
    ]);
    for (remainder, expanded) in [
        (
            "preset:card/example.com/a.jpg".to_string(),
            format!("{card}/example.com/a.jpg"),
        ),
        (
            "exp:99/preset:thumb/maxage:60/filters:blur(2)/example.com/preset:card/a.jpg"
                .to_string(),
            "exp:99/300x200/maxage:60/filters:blur(2)/example.com/preset:card/a.jpg".to_string(),
        ),
        // Operations only, and an empty segment where the
        // source would start: both copy through as before.
        ("preset:thumb".to_string(), "300x200".to_string()),
        ("preset:thumb/".to_string(), "300x200".to_string()),
        (
            "preset:thumb//example.com/a.jpg".to_string(),
            "300x200//example.com/a.jpg".to_string(),
        ),
        (String::new(), String::new()),
        // A malformed operation starts the copied tail; the
        // real parse reports it.
        (
            "preset:thumb/filters:bogus/preset:card/x".to_string(),
            "300x200/filters:bogus/preset:card/x".to_string(),
        ),
    ] {
        assert_eq!(
            expand_presets(&remainder, &table, false).unwrap(),
            expanded,
            "{remainder}"
        );
    }
    // A realistic preset repeated up to the cap expands and
    // parses to exactly the cap: 25 references of four
    // operations each.
    let expanded = expand_presets(&repeated("card", 25), &table, true).unwrap();
    let (ops, source) = parse_operations(&expanded).unwrap();
    assert_eq!(crabcrop::url::operation_count(&ops), MAX_OPERATIONS);
    assert_eq!(source, "example.com/a.jpg");
    // One more reference, or one more operation after a
    // 100-operation preset, is refused with the count the
    // parsed list would have.
    assert_eq!(
        expand_presets(&repeated("card", 26), &table, true),
        Err(ParseError::TooManyOperations(104))
    );
    assert!(expand_presets("preset:i/example.com/a.jpg", &table, false).is_ok());
    assert_eq!(
        expand_presets("preset:i/300x200/example.com/a.jpg", &table, false),
        Err(ParseError::TooManyOperations(MAX_OPERATIONS + 1))
    );
    // Unknown names and free-form operations keep their own
    // refusals, wherever they sit among too many operations.
    let unknown = format!("{}preset:nope/example.com/a.jpg", "preset:i/".repeat(10));
    assert_eq!(
        expand_presets(&unknown, &table, true),
        Err(ParseError::UnknownPreset("nope".to_string()))
    );
    let free_form = format!("{}300x200/example.com/a.jpg", "preset:i/".repeat(10));
    assert_eq!(
        expand_presets(&free_form, &table, true),
        Err(ParseError::PresetsOnly("300x200".to_string()))
    );
}
