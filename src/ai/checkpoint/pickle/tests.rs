use super::*;

/// A reduce of os.system.
pub(super) fn os_system_reduce() -> Vec<u8> {
    b"\x80\x02cos\nsystem\n)R.".to_vec()
}

#[test]
fn unpickle_refuses_an_unknown_callable_by_name() {
    // A reduce of os.system is the canonical
    // hostile pickle; the machine refuses it
    // before anything runs (nothing here can run:
    // the machine only builds data).
    let err = unpickle(&os_system_reduce()).unwrap_err();
    assert!(
        err.contains("unsupported pickle callable os.system"),
        "{err}"
    );
}

/// The memo shape from issue #609. Each step reads the
/// memoized value twice, pairs the two copies, and
/// stores the pair back, so the tree doubles with every
/// 7 bytes.
pub(super) fn memo_doubling(steps: usize) -> Vec<u8> {
    let mut pkl = b"(\x80\x02)q\x00".to_vec();
    for _ in 0..steps {
        pkl.extend_from_slice(b"h\x00h\x00\x86q\x00");
    }
    pkl.extend_from_slice(b"t.");
    pkl
}

/// Run `unpickle` on a thread with a 256 KiB stack and
/// drop what it built there. A deep recursion in the
/// machine, or in the drop of its value, overflows that
/// stack.
fn unpickle_on_a_small_stack(pkl: Vec<u8>) -> Result<(), String> {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || unpickle(&pkl).map(drop))
        .unwrap()
        .join()
        .unwrap()
}

fn budget_error() -> String {
    format!("over {MAX_BUILT} values and string bytes")
}

fn depth_error() -> String {
    format!("more than {MAX_DEPTH} levels deep")
}

#[test]
fn unpickle_refuses_memo_copies_past_the_budget() {
    // 40 steps are 290 bytes and would build about 2^40
    // values. The budget stops the machine early.
    let err = unpickle(&memo_doubling(40)).unwrap_err();
    assert!(err.contains(&budget_error()), "{err}");
    // A few small copies stay well inside it.
    assert!(unpickle(&memo_doubling(8)).is_ok());
}

/// One 32 KiB string, stored in the memo and read back
/// twice.
pub(super) fn memoized_string() -> Vec<u8> {
    let mut pkl = b"\x80\x02(X".to_vec();
    pkl.extend_from_slice(&(32 * 1024u32).to_le_bytes());
    pkl.resize(pkl.len() + 32 * 1024, b'a');
    pkl.extend_from_slice(b"q\x00h\x00h\x00t.");
    pkl
}

#[test]
fn unpickle_counts_memo_copies_of_a_string_by_their_bytes() {
    // One 32 KiB string, stored and read back twice: three
    // values, but 128 KiB of copied bytes. The count
    // passes the budget at the store already. Without the
    // bytes in the count, an 896 KiB string read back from
    // the memo 65,000 times would copy about 55 GiB.
    let err = unpickle(&memoized_string()).unwrap_err();
    assert!(err.contains(&budget_error()), "{err}");
}

/// A 1,000-item tuple, stored in the memo once and read
/// back 100 times.
pub(super) fn tuple_read_back() -> Vec<u8> {
    let mut pkl = b"\x80\x02((".to_vec();
    pkl.resize(pkl.len() + 1000, b'N');
    pkl.extend_from_slice(b"tq\x00");
    for _ in 0..100 {
        pkl.extend_from_slice(b"h\x00");
    }
    pkl.extend_from_slice(b"t.");
    pkl
}

#[test]
fn unpickle_counts_every_memo_read() {
    // A 1,000-item tuple, stored once and read back 100
    // times. Each read counts at the tuple's size.
    let err = unpickle(&tuple_read_back()).unwrap_err();
    assert!(err.contains(&budget_error()), "{err}");
}

/// A 1,000-item tuple stored in 100 memo slots.
pub(super) fn tuple_stored_100_times() -> Vec<u8> {
    let mut pkl = b"\x80\x02(".to_vec();
    pkl.resize(pkl.len() + 1000, b'N');
    pkl.push(b't');
    for i in 0..100u8 {
        pkl.extend_from_slice(&[b'q', i]);
    }
    pkl.push(b'.');
    pkl
}

#[test]
fn unpickle_counts_every_memo_write() {
    // The same tuple stored in 100 memo slots. Each store
    // counts at the tuple's size.
    let err = unpickle(&tuple_stored_100_times()).unwrap_err();
    assert!(err.contains(&budget_error()), "{err}");
}

/// An empty tuple, then TUPLE1 up to the 1 MiB cap.
pub(super) fn deep_tuple_chain() -> Vec<u8> {
    let mut pkl = b"\x80\x02)".to_vec();
    pkl.resize(1024 * 1024 - 1, 0x85);
    pkl.push(b'.');
    pkl
}

#[test]
fn unpickle_refuses_a_deep_tuple_chain_on_a_small_stack() {
    // An empty tuple, then TUPLE1 up to the 1 MiB cap:
    // about a million levels, whose recursive drop
    // overflowed the stack and aborted the process.
    let err = unpickle_on_a_small_stack(deep_tuple_chain()).unwrap_err();
    assert!(err.contains(&depth_error()), "{err}");
}

/// 20,000 dicts on the stack, each with a key, then one
/// SETITEM per level.
pub(super) fn deep_dicts() -> Vec<u8> {
    let levels = 20_000;
    let mut pkl = b"\x80\x02".to_vec();
    for _ in 0..levels {
        pkl.extend_from_slice(b"}\x8c\x01k");
    }
    pkl.push(b'}');
    pkl.resize(pkl.len() + levels, b's');
    pkl.push(b'.');
    pkl
}

#[test]
fn unpickle_refuses_deeply_nested_dicts_on_a_small_stack() {
    // 20,000 dicts on the stack, each with a key, then
    // one SETITEM per level, each storing the dict above
    // into the one below it. They fit in the budget, so
    // the depth bound is what refuses them.
    let err = unpickle_on_a_small_stack(deep_dicts()).unwrap_err();
    assert!(err.contains(&depth_error()), "{err}");
}

/// A pickle whose root nests `levels` tuples or dicts,
/// each level built by `opcode`.
pub(super) fn nested(opcode: u8, levels: usize) -> Vec<u8> {
    let n = levels - 1;
    let mut pkl = b"\x80\x02".to_vec();
    let repeat = |pkl: &mut Vec<u8>, bytes: &[u8], count: usize| {
        for _ in 0..count {
            pkl.extend_from_slice(bytes);
        }
    };
    match opcode {
        // TUPLE: a mark per level, then the innermost
        // tuple, then one TUPLE per level.
        b't' => {
            repeat(&mut pkl, b"(", n);
            pkl.push(b')');
            repeat(&mut pkl, b"t", n);
        }
        0x85 => {
            pkl.push(b')');
            repeat(&mut pkl, b"\x85", n);
        }
        0x86 => {
            pkl.push(b')');
            repeat(&mut pkl, b"N\x86", n);
        }
        0x87 => {
            pkl.push(b')');
            repeat(&mut pkl, b"NN\x87", n);
        }
        // SETITEM and SETITEMS: a dict and a key per
        // level, then the innermost dict, then one store
        // per level.
        b's' => {
            repeat(&mut pkl, b"}\x8c\x01k", n);
            pkl.push(b'}');
            repeat(&mut pkl, b"s", n);
        }
        b'u' => {
            repeat(&mut pkl, b"}(\x8c\x01k", n);
            pkl.push(b'}');
            repeat(&mut pkl, b"u", n);
        }
        _ => unreachable!(),
    }
    pkl.push(b'.');
    pkl
}

/// How many tuples or dicts deep a value nests.
fn depth_of(value: &PickleValue) -> usize {
    match value {
        PickleValue::Tuple(items) => 1 + items.iter().map(depth_of).max().unwrap_or(0),
        PickleValue::Map(pairs) => 1 + pairs.iter().map(|(_, v)| depth_of(v)).max().unwrap_or(0),
        _ => 0,
    }
}

/// The pickles of issue #760: an opcode takes a value from
/// below a MARK, then the TUPLE or SETITEMS that closes the
/// MARK finds it past the top of the stack. The loop before
/// the issue panicked there. Each comes with the opcode
/// that is now refused.
pub(super) const STALE_MARKS: &[(&str, &[u8])] = &[
    ("BUILD", b"N}(bt."),
    ("TUPLE2", b"))(\x86t."),
    ("BUILD", b"N}(bu."),
];

/// For each opcode that pops or reads the stack, a pickle
/// where it uses a value from below an open MARK. CPython's
/// unpickler refuses each one ("unexpected MARK found");
/// the loop before issue #760 accepted each one.
pub(super) const BELOW_A_MARK: &[(&str, &[u8])] = &[
    // The object under the state.
    ("BUILD", b"}(Nb."),
    // The callable under the arguments.
    ("REDUCE", b"ccollections\nOrderedDict\n()R."),
    // The persistent id.
    (
        "BINPERSID",
        b"(\x8c\x07storagectorch\nFloatStorage\n\x8c\x010\x8c\x03cpuK\x01t(Q.",
    ),
    // The dict under the key and value.
    ("SETITEM", b"}(\x8c\x01k\x8c\x01vs."),
    // The dict under the MARK that SETITEMS closes, below
    // the MARK before it.
    ("SETITEMS", b"}((\x8c\x01k\x8c\x01vu."),
    // One value below the MARK.
    ("TUPLE1", b")(\x85."),
    ("TUPLE2", b")(N\x86."),
    ("TUPLE3", b")(NN\x87."),
    // The top the memo would store.
    ("BINPUT", b"N(q\x00."),
    ("LONG_BINPUT", b"N(r\x00\x00\x00\x00."),
    // The pickle's value.
    ("STOP", b"N(."),
];

/// The same opcodes using only values above a MARK that
/// has a value under it, closed by TUPLE before the end.
/// CPython reads each one, and so does the machine. STOP's
/// value stands over a MARK at the bottom, which fences
/// nothing: over any other MARK, a value stands under the
/// MARK too, and STOP refuses a stack of two.
pub(super) const ABOVE_A_MARK: &[(&str, &[u8])] = &[
    ("BUILD", b"N(}Nbt\x86."),
    ("REDUCE", b"N(ccollections\nOrderedDict\n)Rt\x86."),
    (
        "BINPERSID",
        b"N((\x8c\x07storagectorch\nFloatStorage\n\x8c\x010\x8c\x03cpuK\x01tQt\x86.",
    ),
    ("SETITEM", b"N(}\x8c\x01k\x8c\x01vst\x86."),
    // The dict under the MARK that SETITEMS closes, above
    // the MARK before it: CPython reads it there.
    ("SETITEMS", b"N(}(\x8c\x01k\x8c\x01vut\x86."),
    ("TUPLE1", b"N(N\x85t\x86."),
    ("TUPLE2", b"N(NN\x86t\x86."),
    ("TUPLE3", b"N(NNN\x87t\x86."),
    ("BINPUT", b"N(Nq\x00t\x86."),
    ("LONG_BINPUT", b"N(Nr\x00\x00\x00\x00t\x86."),
    ("STOP", b"(N."),
];

/// The refusal of `name` at a MARK.
fn below_a_mark(name: &str) -> String {
    format!("a pickle {name} uses a value from below an open MARK")
}

#[test]
fn unpickle_refuses_the_pickles_that_left_a_stale_mark_without_a_panic() {
    for (name, pkl) in STALE_MARKS {
        let result = std::panic::catch_unwind(|| unpickle(pkl))
            .unwrap_or_else(|_| panic!("b\"{}\" panicked", pkl.escape_ascii()));
        assert_eq!(
            result,
            Err(below_a_mark(name)),
            "b\"{}\"",
            pkl.escape_ascii()
        );
    }
}

#[test]
fn a_mark_fences_every_opcode_that_pops_or_reads_the_stack() {
    for (name, pkl) in BELOW_A_MARK {
        assert_eq!(
            unpickle(pkl),
            Err(below_a_mark(name)),
            "b\"{}\"",
            pkl.escape_ascii()
        );
    }
    for (name, pkl) in ABOVE_A_MARK {
        let result = unpickle(pkl);
        assert!(
            result.is_ok(),
            "{name}: b\"{}\": {result:?}",
            pkl.escape_ascii()
        );
    }
}

#[test]
fn the_latest_open_mark_is_the_fence() {
    // Two MARKs open, at heights 1 and 2: BUILD's state and
    // object stand above the first but not the second, and
    // CPython refuses BUILD at the second.
    assert_eq!(unpickle(b"N(N(Nb."), Err(below_a_mark("BUILD")));
    // Once TUPLE closes the second, the first is the fence
    // again, and BUILD's values stand above it: CPython reads
    // this one, and so does the machine.
    assert_eq!(
        unpickle(b"N(}(NtNbt\x86."),
        Ok(PickleValue::Tuple(vec![
            PickleValue::None,
            PickleValue::Tuple(vec![
                PickleValue::Map(Vec::new()),
                PickleValue::Tuple(vec![PickleValue::None]),
            ]),
        ]))
    );
}

#[test]
fn unpickle_reads_newtrue_as_true_and_newfalse_as_false() {
    assert_eq!(unpickle(b"\x80\x02\x88."), Ok(PickleValue::Bool(true)));
    assert_eq!(unpickle(b"\x80\x02\x89."), Ok(PickleValue::Bool(false)));
}

#[test]
fn unpickle_bounds_nesting_for_every_opcode_that_nests() {
    for opcode in [b't', 0x85, 0x86, 0x87, b's', b'u'] {
        let value = unpickle(&nested(opcode, MAX_DEPTH))
            .unwrap_or_else(|e| panic!("opcode 0x{opcode:02x} at the limit: {e}"));
        assert_eq!(depth_of(&value), MAX_DEPTH, "opcode 0x{opcode:02x}");
        let err = match unpickle(&nested(opcode, MAX_DEPTH + 1)) {
            Ok(_) => panic!("opcode 0x{opcode:02x} one past the limit was accepted"),
            Err(e) => e,
        };
        assert!(err.contains(&depth_error()), "opcode 0x{opcode:02x}: {err}");
    }
}
