//! The pickle machine against the single opcode loop it
//! had before its opcodes got handlers (issue #741). The
//! loop is frozen in `handler_proofs/frozen.rs`, and every
//! input here must come out of both machines the same way,
//! as the same value or the same error message, but for
//! the two changes issue #760 made after the loop:
//!
//! - An open MARK fences the stack. Where the loop pops or
//!   reads a value from below the latest open MARK, the
//!   machine refuses that opcode with the fence error. A
//!   model of the stack's height and its MARKs
//!   ([`first_breach`]) finds the first such opcode, and
//!   the fence error is expected there when the loop reads
//!   every opcode before it without an error; the loop's
//!   outcome is expected everywhere else. The loop panics
//!   only after such an opcode, at a TUPLE or SETITEMS whose
//!   MARK stands past the top of the stack, so the machine
//!   must never panic.
//! - NEWTRUE pushes true and NEWFALSE false, where the loop
//!   pushed the reverse, so a bool in the machine's value
//!   matches its opposite in the loop's.
//!
//! The inputs are the checkpoint fixtures the unit tests
//! build, the fence tests' pickles, and one more: the
//! scalar opcodes at their edges, which none of those
//! holds. The small ones run whole, cut
//! at every length, with every byte deleted or overwritten,
//! and with every span of up to eight bytes doubled: the
//! head pickle's bytes take every value, the others' every
//! opcode the loop reads and a few other bytes. The large
//! ones, which run until a limit stops them, run whole and
//! cut at a sample of lengths. As property tests, random
//! opcode streams run, and the fixtures mutated, truncated
//! and with spans repeated. An ignored test adds the LAION
//! heads the AI workflow downloads.
//!
//! The cases per property default to [`DEFAULT_CASES`];
//! `PROPTEST_CASES` raises it (the nightly job does). A
//! failure is saved next to this file in
//! `handler_proofs.proptest-regressions`, which is checked
//! in so the seed replays first on every later run.

mod frozen;

use std::sync::OnceLock;

use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::{FileFailurePersistence, TestCaseError};

use super::super::tests::{head_pickle, overflowing_size_pickle};
use super::tests::{
    ABOVE_A_MARK, BELOW_A_MARK, STALE_MARKS, deep_dicts, deep_tuple_chain, memo_doubling,
    memoized_string, nested, os_system_reduce, tuple_read_back, tuple_stored_100_times,
};
use super::{MAX_DEPTH, PickleValue, fence_error, unpickle};

/// What a machine made of an input. A panic is an outcome
/// too: the loop panics when a TUPLE or SETITEMS finds its
/// mark past the top of the stack. No input may panic the
/// machine (issue #760).
#[derive(Debug)]
enum Outcome {
    Value(PickleValue),
    Error(String),
    /// The panic's message, which only the reports print
    /// (through `Debug`): a panic matches nothing.
    #[allow(dead_code)]
    Panicked(String),
}

fn outcome(run: impl FnOnce() -> Result<PickleValue, String>) -> Outcome {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(value)) => Outcome::Value(value),
        Ok(Err(message)) => Outcome::Error(message),
        Err(payload) => Outcome::Panicked(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default(),
        ),
    }
}

/// The machine's value against the loop's: as `PartialEq`
/// has them, except floats, which compare by their bits, so
/// a NaN matches itself, and bools, which compare swapped:
/// the loop pushed false for NEWTRUE and true for NEWFALSE,
/// and the machine pushes what they name (issue #760).
fn same_value(a: &PickleValue, b: &PickleValue) -> bool {
    match (a, b) {
        (PickleValue::Float(a), PickleValue::Float(b)) => a.to_bits() == b.to_bits(),
        (PickleValue::Bool(a), PickleValue::Bool(b)) => a != b,
        (PickleValue::Tuple(a), PickleValue::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_value(a, b))
        }
        (PickleValue::Map(a), PickleValue::Map(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|((ka, va), (kb, vb))| ka == kb && same_value(va, vb))
        }
        _ => a == b,
    }
}

/// The machine's outcome against the loop's. A panic
/// matches nothing: the machine must not panic, and where
/// the loop panics, the fence error is expected instead.
fn same(a: &Outcome, b: &Outcome) -> bool {
    match (a, b) {
        (Outcome::Value(a), Outcome::Value(b)) => same_value(a, b),
        (Outcome::Error(a), Outcome::Error(b)) => a == b,
        _ => false,
    }
}

/// What each machine made of `pkl`, and the breach the loop
/// reaches in it, when the machine's outcome is not the one
/// expected: the fence error at that breach, and the loop's
/// outcome where there is none.
fn disagreement(pkl: &[u8]) -> Option<String> {
    let handlers = outcome(|| unpickle(pkl));
    let frozen = outcome(|| frozen::unpickle(pkl));
    let breach = reached_breach(pkl);
    let expected = match breach {
        Some((_, name)) => matches!(&handlers, Outcome::Error(e) if *e == fence_error(name)),
        None => same(&handlers, &frozen),
    };
    if expected {
        return None;
    }
    Some(format!(
        "{} bytes, starting b\"{}\":\n  the handlers: {handlers:?}\n  the loop: {frozen:?}\n  \
         the breach the loop reaches (offset, opcode): {breach:?}",
        pkl.len(),
        pkl[..pkl.len().min(160)].escape_ascii()
    ))
}

/// The loop refuses a pickle longer than this before it
/// reads an opcode (`frozen::unpickle`).
const LOOP_MAX_BYTES: usize = 1024 * 1024;

/// An opcode that pops or reads a value from below an open
/// MARK: its offset in the pickle and its name.
type Breach = (usize, &'static str);

/// The breach the loop reaches in `pkl`: the first one
/// [`first_breach`] finds, when the loop reads every opcode
/// before it without an error.
fn reached_breach(pkl: &[u8]) -> Option<Breach> {
    if pkl.len() > LOOP_MAX_BYTES {
        return None;
    }
    let breach = first_breach(pkl)?;
    let before = &pkl[..breach.0];
    match outcome(|| frozen::unpickle(before)) {
        Outcome::Error(e) if e == "the pickle ends without a STOP" => Some(breach),
        _ => None,
    }
}

/// The first opcode in `pkl` that pops or reads a value from
/// below an open MARK, found by a model of the loop's stack:
/// its height, and the height each open MARK recorded. The
/// model walks the opcodes as the loop parses them and takes
/// each to succeed; [`reached_breach`] checks that the loop
/// gets that far. It stops where the loop stops for certain:
/// at STOP, at a byte the loop does not read, at an operand
/// that runs past the end, and at a TUPLE or SETITEMS with no
/// MARK open.
///
/// What an opcode uses is what CPython's unpickler takes from
/// its stack (`Modules/_pickle.c`): BUILD and REDUCE the top
/// two values, SETITEM the top three (the dict too),
/// BINPERSID, BINPUT, LONG_BINPUT and STOP the top one, TUPLE1
/// to TUPLE3 the top one to three, SETITEMS the dict under the
/// MARK it closes, and TUPLE only what stands above its own
/// MARK. An opcode breaches when the lowest value it uses
/// stands below the latest open MARK, unless that MARK is at
/// the bottom of the stack, where it fences nothing.
fn first_breach(pkl: &[u8]) -> Option<Breach> {
    let mut at = 0;
    let mut height = 0usize;
    let mut marks: Vec<usize> = Vec::new();
    loop {
        // No breach without a MARK, so none is left once no
        // MARK is open and no '(' byte follows.
        if marks.is_empty() && !pkl.get(at..)?.contains(&b'(') {
            return None;
        }
        let start = at;
        let op = *pkl.get(start)?;
        at = opcode_end(pkl, start)?;
        let h = height as isize;
        // The opcode's name, the lowest stack position it uses
        // (none: `isize::MAX`), and the height after it.
        let (name, lowest, after) = match op {
            b'(' => {
                marks.push(height);
                continue;
            }
            b't' => {
                let mark = marks.pop()?;
                ("TUPLE", isize::MAX, mark + 1)
            }
            b'u' => {
                let mark = marks.pop()?;
                ("SETITEMS", mark as isize - 1, mark)
            }
            b'b' => ("BUILD", h - 2, height.saturating_sub(1)),
            b'R' => ("REDUCE", h - 2, height.saturating_sub(1)),
            b's' => ("SETITEM", h - 3, height.saturating_sub(2)),
            b'Q' => ("BINPERSID", h - 1, height),
            b'q' => ("BINPUT", h - 1, height),
            b'r' => ("LONG_BINPUT", h - 1, height),
            b'.' => ("STOP", h - 1, height),
            0x85..=0x87 => {
                let n = usize::from(op - 0x84);
                let name = ["TUPLE1", "TUPLE2", "TUPLE3"][n - 1];
                (name, h - n as isize, height.saturating_sub(n) + 1)
            }
            0x80 => ("PROTO", isize::MAX, height),
            // Every other opcode the loop reads pushes one value.
            _ => ("", isize::MAX, height + 1),
        };
        let fence = marks.last().map_or(0, |&mark| mark as isize);
        if fence > 0 && lowest < fence {
            return Some((start, name));
        }
        if op == b'.' {
            return None;
        }
        height = after;
    }
}

/// Where the opcode at `at` in `pkl` ends, past its operand,
/// as the loop parses it: `None` for a byte the loop does not
/// read, or an operand that runs past the end.
fn opcode_end(pkl: &[u8], at: usize) -> Option<usize> {
    let length = |i: usize| pkl.get(i).map(|&b| usize::from(b));
    let word = |i: usize| {
        pkl.get(i..i + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    let line = |i: usize| Some(i + pkl.get(i..)?.iter().position(|&b| b == b'\n')? + 1);
    let end = match *pkl.get(at)? {
        b'N'
        | 0x88
        | 0x89
        | b'('
        | b')'
        | b't'
        | 0x85..=0x87
        | b'}'
        | b's'
        | b'u'
        | b'b'
        | b'R'
        | b'Q'
        | b'.' => at + 1,
        0x80 | b'K' | b'q' | b'h' => at + 2,
        b'M' => at + 3,
        b'J' | b'r' | b'j' => at + 5,
        b'G' => at + 9,
        0x8c | b'U' | 0x8a => at + 2 + length(at + 1)?,
        b'X' | 0x8b => at + 5 + word(at + 1)?,
        b'L' => line(at + 1)?,
        b'c' => line(line(at + 1)?)?,
        _ => return None,
    };
    (end <= pkl.len()).then_some(end)
}

/// One way to vary a fixture.
#[derive(Debug, Clone, Copy)]
enum Variant {
    Whole,
    Cut(usize),
    Set(usize, u8),
    Delete(usize),
    /// The span at the first position, as long as the
    /// second, written twice.
    Double(usize, usize),
}

impl Variant {
    fn apply(self, pkl: &[u8]) -> Vec<u8> {
        match self {
            Variant::Whole => pkl.to_vec(),
            Variant::Cut(len) => pkl[..len].to_vec(),
            Variant::Set(at, byte) => {
                let mut out = pkl.to_vec();
                out[at] = byte;
                out
            }
            Variant::Delete(at) => [&pkl[..at], &pkl[at + 1..]].concat(),
            Variant::Double(at, len) => [&pkl[..at + len], &pkl[at..]].concat(),
        }
    }
}

/// Every variant of a fixture `len` bytes long: whole, cut
/// at every length, with every byte set to each of `bytes`
/// or deleted, and with every span of up to eight bytes
/// doubled.
fn every_variant(len: usize, bytes: &[u8]) -> impl Iterator<Item = Variant> + '_ {
    let cuts = (0..len).map(Variant::Cut);
    let sets = (0..len).flat_map(move |at| bytes.iter().map(move |&byte| Variant::Set(at, byte)));
    let deletes = (0..len).map(Variant::Delete);
    let doubles =
        (0..len).flat_map(move |at| (1..=(len - at).min(8)).map(move |n| Variant::Double(at, n)));
    std::iter::once(Variant::Whole)
        .chain(cuts)
        .chain(sets)
        .chain(deletes)
        .chain(doubles)
}

/// The variants of a large fixture: whole, and cut at the
/// first 64 lengths, the last four and 16 between.
fn sampled_cuts(len: usize) -> impl Iterator<Item = Variant> {
    let step = (len / 16).max(1);
    let cuts = (0..len)
        .filter(move |&n| n < 64 || n + 4 >= len || n % step == 0)
        .map(Variant::Cut);
    std::iter::once(Variant::Whole).chain(cuts)
}

/// The opcodes the loop reads, in the order of its match
/// arms.
const OPCODES: &[u8] = b"\x80cX\x8cUKMJL\x8a\x8bGN\x88\x89)(t\x85\x86\x87QR}subqrhj.";

/// The bytes the sweeps write over the smaller fixtures:
/// every opcode the loop reads, the small numbers lengths
/// and memo indices take, and bytes it refuses: 0x7f, 0xff,
/// FRAME (protocol 4, past the protocol 2 a head save
/// uses), and APPEND and EMPTY_LIST (a list, which a head
/// save does not hold).
fn sweep_bytes() -> Vec<u8> {
    let mut bytes = OPCODES.to_vec();
    bytes.extend(0..8);
    bytes.extend([0x7f, 0xff, 0x95, b'a', b']']);
    bytes
}

/// Run both machines over each variant of `pkl`, and fail
/// with the first few where the machine's outcome is not
/// the one expected (see `disagreement`).
fn check(name: &str, pkl: &[u8], variants: impl Iterator<Item = Variant>) {
    let (mut runs, mut differ, mut reports) = (0usize, 0usize, Vec::new());
    for variant in variants {
        runs += 1;
        if let Some(report) = disagreement(&variant.apply(pkl)) {
            differ += 1;
            if reports.len() < 3 {
                reports.push(format!("{variant:?}: {report}"));
            }
        }
    }
    assert_eq!(
        differ,
        0,
        "{name}: {differ} of {runs} variants come out differently, among them\n{}",
        reports.join("\n")
    );
}

/// A fixture: its name and its bytes.
type Fixture = (String, Vec<u8>);

/// The scalar opcodes at their edges, in one tuple: BININT1
/// and BININT2 at their largest, BININT at -1 and at both
/// ends of the i32 range, a LONG line at each end of the
/// i64 range (one with the L suffix, one without), LONG1
/// and LONG4 at every length up to the eight bytes an i64
/// holds with the top bit set (sign-extended below eight
/// bytes, not at eight), a negative zero and a NaN, a
/// SHORT_BINSTRING that is not UTF-8 (read lossily) and a
/// BINUNICODE that is. A head save writes none of these,
/// and the sweeps vary each.
fn scalar_edges() -> Vec<u8> {
    let mut pkl = b"\x80\x02(K\xffM\xff\xff".to_vec();
    pkl.extend_from_slice(b"J\xff\xff\xff\xffJ\x00\x00\x00\x80J\xff\xff\xff\x7f");
    pkl.extend_from_slice(b"L-9223372036854775808L\nL9223372036854775807\n");
    for len in 0..=8u8 {
        let body: Vec<u8> = (1..=len)
            .map(|i| if i == len { 0x80 | i } else { i })
            .collect();
        pkl.extend_from_slice(&[0x8a, len]);
        pkl.extend_from_slice(&body);
        pkl.push(0x8b);
        pkl.extend_from_slice(&u32::from(len).to_le_bytes());
        pkl.extend_from_slice(&body);
    }
    pkl.extend_from_slice(b"G\x80\x00\x00\x00\x00\x00\x00\x00");
    pkl.extend_from_slice(b"G\x7f\xf8\x00\x00\x00\x00\x00\x01");
    pkl.extend_from_slice(b"U\x02\xff\xfeX\x02\x00\x00\x00\xc3\xa9");
    pkl.extend_from_slice(b"t.");
    pkl
}

/// The other fixtures that run in microseconds: the head
/// pickle with two bias values and with a size that
/// overflows, the hostile reduce, a short memo chain, the
/// empty pickle and the scalar edges.
fn small_fixtures() -> Vec<Fixture> {
    vec![
        ("head_pickle(3, 2)".to_string(), head_pickle(3, 2)),
        (
            "overflowing_size_pickle()".to_string(),
            overflowing_size_pickle(),
        ),
        ("os_system_reduce()".to_string(), os_system_reduce()),
        ("memo_doubling(8)".to_string(), memo_doubling(8)),
        ("the empty pickle".to_string(), Vec::new()),
        ("scalar_edges()".to_string(), scalar_edges()),
    ]
}

/// Every nesting shape, at `MAX_DEPTH` and one past it.
fn nesting_fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    for opcode in [b't', 0x85, 0x86, 0x87, b's', b'u'] {
        for levels in [MAX_DEPTH, MAX_DEPTH + 1] {
            out.push((
                format!("nested(0x{opcode:02x}, {levels})"),
                nested(opcode, levels),
            ));
        }
    }
    out
}

/// The fence tests' pickles (issue #760): the ones that
/// left a MARK past the top of the stack, and for each
/// opcode that pops or reads the stack, one where it uses a
/// value just below an open MARK and one where it uses
/// values just above one.
fn fence_fixtures() -> Vec<Fixture> {
    let named = |kind: &str, (name, pkl): &(&str, &[u8])| (format!("{kind}: {name}"), pkl.to_vec());
    let stale = STALE_MARKS.iter().map(|case| named("a stale MARK", case));
    let below = BELOW_A_MARK.iter().map(|case| named("below a MARK", case));
    let above = ABOVE_A_MARK.iter().map(|case| named("above a MARK", case));
    stale.chain(below).chain(above).collect()
}

/// The fixtures that copy or nest until the budget or the
/// depth bound stops them.
fn large_fixtures() -> Vec<Fixture> {
    vec![
        ("memo_doubling(40)".to_string(), memo_doubling(40)),
        ("memoized_string()".to_string(), memoized_string()),
        ("tuple_read_back()".to_string(), tuple_read_back()),
        (
            "tuple_stored_100_times()".to_string(),
            tuple_stored_100_times(),
        ),
        ("deep_tuple_chain()".to_string(), deep_tuple_chain()),
        ("deep_dicts()".to_string(), deep_dicts()),
    ]
}

#[test]
fn the_machines_agree_on_every_variant_of_the_head_pickle() {
    let every_byte: Vec<u8> = (0..=u8::MAX).collect();
    let pkl = head_pickle(3, 1);
    check(
        "head_pickle(3, 1)",
        &pkl,
        every_variant(pkl.len(), &every_byte),
    );
}

#[test]
fn the_machines_agree_on_the_variants_of_the_small_fixtures() {
    let bytes = sweep_bytes();
    for (name, pkl) in small_fixtures() {
        check(&name, &pkl, every_variant(pkl.len(), &bytes));
    }
}

#[test]
fn the_machines_agree_on_the_variants_of_the_nesting_shapes() {
    let bytes = sweep_bytes();
    for (name, pkl) in nesting_fixtures() {
        check(&name, &pkl, every_variant(pkl.len(), &bytes));
    }
}

#[test]
fn the_machines_agree_on_the_variants_of_the_fence_cases() {
    let bytes = sweep_bytes();
    for (name, pkl) in fence_fixtures() {
        check(&name, &pkl, every_variant(pkl.len(), &bytes));
    }
}

#[test]
fn the_machines_agree_on_the_large_fixtures_whole_and_cut() {
    for (name, pkl) in large_fixtures() {
        check(&name, &pkl, sampled_cuts(pkl.len()));
    }
}

/// The LAION heads the AI workflow downloads, run like the
/// head pickle:
/// CRABCROP_TEST_AESTHETIC_HEAD=... CRABCROP_TEST_AESTHETIC_HEAD_L14=...
/// cargo test -- --ignored the_machines_agree_on_the_real_heads
#[test]
#[ignore = "needs the LAION head files (env-configured)"]
fn the_machines_agree_on_the_real_heads() {
    for var in [
        "CRABCROP_TEST_AESTHETIC_HEAD",
        "CRABCROP_TEST_AESTHETIC_HEAD_L14",
    ] {
        let Some(path) = crate::ai::env_or_skip(var) else {
            continue;
        };
        let bytes = std::fs::read(&path).unwrap();
        let entries = super::super::zip::read_zip_entries(&bytes).unwrap();
        let pkl = &entries
            .iter()
            .find(|e| e.name == "archive/data.pkl")
            .unwrap_or_else(|| panic!("{path} holds no archive/data.pkl"))
            .data;
        assert!(
            matches!(unpickle(pkl), Ok(PickleValue::Map(_))),
            "{path} does not unpickle to a state dict"
        );
        let every_byte: Vec<u8> = (0..=u8::MAX).collect();
        check(&path, pkl, every_variant(pkl.len(), &every_byte));
    }
}

/// Cases per property when `PROPTEST_CASES` is unset. A
/// case runs two machines over a few hundred bytes, so the
/// default is well above the URL and crop suites'.
const DEFAULT_CASES: u32 = 1024;

fn config() -> ProptestConfig {
    let mut config = ProptestConfig {
        failure_persistence: Some(Box::new(FileFailurePersistence::WithSource(
            "proptest-regressions",
        ))),
        ..ProptestConfig::default()
    };
    if std::env::var_os("PROPTEST_CASES").is_none() {
        config.cases = DEFAULT_CASES;
    }
    config
}

/// Every fixture, built once.
fn fixtures() -> &'static [Fixture] {
    static ALL: OnceLock<Vec<Fixture>> = OnceLock::new();
    ALL.get_or_init(|| {
        let head = ("head_pickle(3, 1)".to_string(), head_pickle(3, 1));
        std::iter::once(head)
            .chain(small_fixtures())
            .chain(nesting_fixtures())
            .chain(large_fixtures())
            .collect()
    })
}

fn pick(fixture: Index) -> &'static [u8] {
    let all = fixtures();
    &all[fixture.index(all.len())].1
}

/// A byte for an edit: an opcode the loop reads, a small
/// number (the lengths and memo indices), or any byte.
fn byte() -> impl Strategy<Value = u8> {
    prop_oneof![prop::sample::select(OPCODES), 0u8..8, any::<u8>()]
}

/// Where an edit lands: anywhere, or in the first 64
/// bytes, where the large fixtures do their work before a
/// limit stops them.
#[derive(Debug, Clone)]
enum Spot {
    Anywhere(Index),
    Early(usize),
}

impl Spot {
    /// A position below `len`, which is not zero.
    fn below(&self, len: usize) -> usize {
        match self {
            Spot::Anywhere(index) => index.index(len),
            Spot::Early(at) => (*at).min(len - 1),
        }
    }
}

fn spot() -> impl Strategy<Value = Spot> {
    prop_oneof![
        any::<Index>().prop_map(Spot::Anywhere),
        (0usize..64).prop_map(Spot::Early),
    ]
}

#[derive(Debug, Clone)]
enum Edit {
    Set(Spot, u8),
    Insert(Spot, u8),
    Delete(Spot),
}

impl Edit {
    fn apply(&self, pkl: &mut Vec<u8>) {
        match self {
            Edit::Set(spot, byte) if !pkl.is_empty() => {
                let at = spot.below(pkl.len());
                pkl[at] = *byte;
            }
            Edit::Insert(spot, byte) => {
                let at = spot.below(pkl.len() + 1);
                pkl.insert(at, *byte);
            }
            Edit::Delete(spot) if !pkl.is_empty() => {
                let at = spot.below(pkl.len());
                pkl.remove(at);
            }
            _ => {}
        }
    }
}

fn edit() -> impl Strategy<Value = Edit> {
    prop_oneof![
        3 => (spot(), byte()).prop_map(|(spot, byte)| Edit::Set(spot, byte)),
        1 => (spot(), byte()).prop_map(|(spot, byte)| Edit::Insert(spot, byte)),
        1 => spot().prop_map(Edit::Delete),
    ]
}

/// The head pickle, for streams to splice runs of.
fn head() -> &'static [u8] {
    static HEAD: OnceLock<Vec<u8>> = OnceLock::new();
    HEAD.get_or_init(|| head_pickle(3, 1))
}

/// A run of the head pickle: real opcode sequences (a
/// persistent id, a rebuild, a memo get) for a stream.
fn head_run() -> impl Strategy<Value = Vec<u8>> {
    (any::<Index>(), 1usize..=64).prop_map(|(at, len)| {
        let head = head();
        let start = at.index(head.len());
        head[start..(start + len).min(head.len())].to_vec()
    })
}

/// LONG: a decimal line, with or without the L suffix
/// Python 2 wrote, or a line that is not a number.
fn long_line_op() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        any::<i64>().prop_map(|v| v.to_string()),
        "[-+]?[0-9]{0,21}L?",
        "[ -~]{0,6}",
    ]
    .prop_map(|digits| [&b"L"[..], digits.as_bytes(), &b"\n"[..]].concat())
}

/// LONG1 or LONG4: a length, at times past the eight bytes
/// an i64 holds, and up to ten bytes that may not match it.
fn long_bytes_op() -> impl Strategy<Value = Vec<u8>> {
    (
        any::<bool>(),
        0u8..=10,
        prop::collection::vec(any::<u8>(), 0..=10),
    )
        .prop_map(|(short, len, body)| {
            let mut out = if short {
                vec![0x8a, len]
            } else {
                [&[0x8b][..], &u32::from(len).to_le_bytes()[..]].concat()
            };
            out.extend_from_slice(&body);
            out
        })
}

/// BINUNICODE, SHORT_BINUNICODE or SHORT_BINSTRING over a
/// word a head save holds or any bytes, its length now and
/// then off by up to two.
fn string_op() -> impl Strategy<Value = Vec<u8>> {
    let words = vec![
        "storage",
        "weight",
        "bias",
        "cpu",
        "0",
        "1",
        "_metadata",
        "version",
        "",
    ];
    let body = prop_oneof![
        2 => prop::sample::select(words).prop_map(|w| w.as_bytes().to_vec()),
        1 => prop::collection::vec(any::<u8>(), 0..8),
    ];
    let skew = prop_oneof![6 => Just(0i64), 1 => -2i64..=2];
    (prop::sample::select(&b"X\x8cU"[..]), body, skew).prop_map(|(op, body, skew)| {
        let len = (body.len() as i64 + skew).max(0) as u32;
        let mut out = vec![op];
        if op == b'X' {
            out.extend_from_slice(&len.to_le_bytes());
        } else {
            out.push(len as u8);
        }
        out.extend_from_slice(&body);
        out
    })
}

/// GLOBAL: a module and a name the machine meets in head
/// saves or refuses by name, or bytes that may not form a
/// line at all.
fn global_op() -> impl Strategy<Value = Vec<u8>> {
    let lines = vec![
        "collections\nOrderedDict\n",
        "torch._utils\n_rebuild_tensor_v2\n",
        "torch._utils\n_rebuild_tensor\n",
        "torch\nFloatStorage\n",
        "torch\nHalfStorage\n",
        "os\nsystem\n",
    ];
    prop_oneof![
        3 => prop::sample::select(lines).prop_map(|l| l.as_bytes().to_vec()),
        1 => prop::collection::vec(any::<u8>(), 0..12),
    ]
    .prop_map(|line| [&b"c"[..], &line[..]].concat())
}

/// One opcode with an operand near the ones the machine
/// reads, any single byte, or a run of the head pickle.
fn token() -> impl Strategy<Value = Vec<u8>> {
    // Memo indices: mostly the few a stream stores, at
    // times far past them or past the 4096 bound.
    let index = prop_oneof![3 => 0u8..4, 1 => any::<u8>()];
    let wide_index = prop_oneof![
        3 => 0u32..4,
        1 => prop::sample::select(&[4095u32, 4096, 4097, u32::MAX][..]),
        1 => any::<u32>(),
    ];
    prop_oneof![
        8 => prop::sample::select(&b"N\x88\x89)(t\x85\x86\x87QR}sub."[..]).prop_map(|op| vec![op]),
        3 => (prop::sample::select(&b"\x80Kqh"[..]), index).prop_map(|(op, v)| vec![op, v]),
        2 => (prop::sample::select(&b"rj"[..]), wide_index)
            .prop_map(|(op, v)| [&[op][..], &v.to_le_bytes()[..]].concat()),
        1 => any::<u16>().prop_map(|v| [&b"M"[..], &v.to_le_bytes()[..]].concat()),
        1 => any::<u32>().prop_map(|v| [&b"J"[..], &v.to_le_bytes()[..]].concat()),
        1 => any::<u64>().prop_map(|v| [&b"G"[..], &v.to_be_bytes()[..]].concat()),
        1 => long_line_op(),
        1 => long_bytes_op(),
        3 => string_op(),
        2 => global_op(),
        1 => any::<u8>().prop_map(|b| vec![b]),
        2 => head_run(),
    ]
}

/// Tokens after PROTO 2 or not, ending in STOP or not.
fn stream() -> impl Strategy<Value = Vec<u8>> {
    (
        any::<bool>(),
        prop::collection::vec(token(), 0..40),
        prop::bool::weighted(0.75),
    )
        .prop_map(|(proto, tokens, stop)| {
            let mut pkl = if proto {
                b"\x80\x02".to_vec()
            } else {
                Vec::new()
            };
            pkl.extend(tokens.concat());
            if stop {
                pkl.push(b'.');
            }
            pkl
        })
}

/// Fail the property unless both machines make the same of
/// `pkl`.
fn agree(pkl: &[u8]) -> Result<(), TestCaseError> {
    match disagreement(pkl) {
        None => Ok(()),
        Some(report) => Err(TestCaseError::fail(report)),
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn the_machines_agree_on_random_opcode_streams(pkl in stream()) {
        agree(&pkl)?;
    }

    #[test]
    fn the_machines_agree_on_mutated_fixtures(
        fixture in any::<Index>(),
        edits in prop::collection::vec(edit(), 1..6),
    ) {
        let mut pkl = pick(fixture).to_vec();
        for edit in &edits {
            edit.apply(&mut pkl);
        }
        agree(&pkl)?;
    }

    #[test]
    fn the_machines_agree_on_truncated_fixtures(fixture in any::<Index>(), cut in spot()) {
        let pkl = pick(fixture);
        agree(&pkl[..cut.below(pkl.len() + 1)])?;
    }

    #[test]
    fn the_machines_agree_on_fixtures_with_repeated_spans(
        fixture in any::<Index>(),
        at in spot(),
        len in 1usize..=48,
        copies in 1usize..=8,
    ) {
        let pkl = pick(fixture);
        if !pkl.is_empty() {
            let start = at.below(pkl.len());
            let end = (start + len).min(pkl.len());
            let mut repeated = pkl[..end].to_vec();
            for _ in 0..copies {
                repeated.extend_from_slice(&pkl[start..end]);
            }
            repeated.extend_from_slice(&pkl[end..]);
            agree(&repeated)?;
        }
    }
}
