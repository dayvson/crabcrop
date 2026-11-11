use super::*;

#[test]
fn unpickle_refuses_an_unknown_callable_by_name() {
    // A reduce of os.system is the canonical
    // hostile pickle; the machine refuses it
    // before anything runs (nothing here can run:
    // the machine only builds data).
    let pkl = b"\x80\x02cos\nsystem\n)R.".to_vec();
    let err = unpickle(&pkl).unwrap_err();
    assert!(
        err.contains("unsupported pickle callable os.system"),
        "{err}"
    );
}

/// The memo shape from issue #609. Each step reads the
/// memoized value twice, pairs the two copies, and
/// stores the pair back, so the tree doubles with every
/// 7 bytes.
fn memo_doubling(steps: usize) -> Vec<u8> {
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

#[test]
fn unpickle_counts_memo_copies_of_a_string_by_their_bytes() {
    // One 32 KiB string, stored and read back twice: three
    // values, but 128 KiB of copied bytes. Without the
    // bytes in the count, a 512 KiB string read back from
    // the memo 250,000 times would copy 128 GiB.
    let mut pkl = b"\x80\x02(X".to_vec();
    pkl.extend_from_slice(&(32 * 1024u32).to_le_bytes());
    pkl.resize(pkl.len() + 32 * 1024, b'a');
    pkl.extend_from_slice(b"q\x00h\x00h\x00t.");
    let err = unpickle(&pkl).unwrap_err();
    assert!(err.contains(&budget_error()), "{err}");
}

#[test]
fn unpickle_refuses_a_deep_tuple_chain_on_a_small_stack() {
    // An empty tuple, then TUPLE1 up to the 1 MiB cap:
    // about a million levels, whose recursive drop
    // overflowed the stack and aborted the process.
    let mut pkl = b"\x80\x02)".to_vec();
    pkl.resize(1024 * 1024 - 1, 0x85);
    pkl.push(b'.');
    let err = unpickle_on_a_small_stack(pkl).unwrap_err();
    assert!(err.contains(&depth_error()), "{err}");
}

#[test]
fn unpickle_refuses_deeply_nested_dicts_on_a_small_stack() {
    // 20,000 dicts on the stack, each with a key, then
    // one SETITEM per level, each storing the dict above
    // into the one below it. They fit in the budget, so
    // the depth bound is what refuses them.
    let levels = 20_000;
    let mut pkl = b"\x80\x02".to_vec();
    for _ in 0..levels {
        pkl.extend_from_slice(b"}\x8c\x01k");
    }
    pkl.push(b'}');
    pkl.resize(pkl.len() + levels, b's');
    pkl.push(b'.');
    let err = unpickle_on_a_small_stack(pkl).unwrap_err();
    assert!(err.contains(&depth_error()), "{err}");
}

/// A pickle whose root nests `levels` tuples or dicts,
/// each level built by `opcode`.
fn nested(opcode: u8, levels: usize) -> Vec<u8> {
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
