use super::*;
use std::sync::Arc;

/// A key for the tests of the entries: any 16 bytes will do.
fn key(n: u32) -> MemoKey {
    let mut bytes = [0u8; 16];
    bytes[..4].copy_from_slice(&n.to_le_bytes());
    MemoKey(bytes)
}

fn keys_in_order<B>(memo: &DetectionMemo<B>) -> Vec<MemoKey> {
    memo.entries.lock().unwrap().iter().map(|e| e.0).collect()
}

#[test]
fn a_key_is_the_digest_of_its_parts_and_their_lengths() {
    // The same parts give the same key; any part that
    // differs, and the same bytes cut into other parts,
    // give another.
    let memo = DetectionMemo::<u32>::new();
    let a = memo.key(&[b"pixels", &640u32.to_le_bytes(), &480u32.to_le_bytes()]);
    assert_eq!(
        a,
        memo.key(&[b"pixels", &640u32.to_le_bytes(), &480u32.to_le_bytes()])
    );
    assert_ne!(
        a,
        memo.key(&[b"pixelz", &640u32.to_le_bytes(), &480u32.to_le_bytes()])
    );
    assert_ne!(
        a,
        memo.key(&[b"pixels", &641u32.to_le_bytes(), &480u32.to_le_bytes()])
    );
    assert_ne!(
        a,
        memo.key(&[b"pixels", &480u32.to_le_bytes(), &640u32.to_le_bytes()])
    );
    assert_ne!(memo.key(&[b"ab", b"c"]), memo.key(&[b"a", b"bc"]));
    assert_ne!(memo.key(&[b"abc"]), memo.key(&[b"abc", b""]));
    assert_ne!(memo.key(&[]), memo.key(&[b""]));
}

#[test]
fn the_digest_is_siphash_2_4_with_a_128_bit_output_under_the_secret() {
    // With the secret set to the reference key 00..0f, the
    // key of no parts is the digest of the empty message:
    // the first SipHash-2-4-128 test vector of the
    // algorithm's reference implementation. SipHash-1-3, a
    // 64-bit output, or a digest that leaves the secret out
    // gives another value.
    let mut memo = DetectionMemo::<u32>::new();
    memo.secret = (0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908);
    let vector = [
        0xa3, 0x81, 0x7f, 0x04, 0xba, 0x25, 0xa8, 0xe6, 0x6d, 0xf6, 0x72, 0x14, 0xc7, 0x55, 0x02,
        0x93,
    ];
    assert_eq!(memo.key(&[]), MemoKey(vector));
    // A part is its eight-byte little-endian length, then
    // its bytes.
    use siphasher::sip128::SipHasher24;
    let mut message = 3u64.to_le_bytes().to_vec();
    message.extend_from_slice(b"abc");
    let want = SipHasher24::new_with_keys(memo.secret.0, memo.secret.1)
        .hash(&message)
        .as_bytes();
    assert_eq!(memo.key(&[b"abc"]), MemoKey(want));
}

#[test]
fn every_memo_keys_with_its_own_secret() {
    // Two memos give the same input different keys: each
    // drew its own secret, so a key made by one means
    // nothing to another, and nobody outside a memo can
    // make its keys.
    let parts: [&[u8]; 3] = [b"pixels", &640u32.to_le_bytes(), &480u32.to_le_bytes()];
    let first = DetectionMemo::<u32>::new();
    let second = DetectionMemo::<u32>::new();
    assert_ne!(first.secret, second.secret);
    assert_ne!(first.key(&parts), second.key(&parts));
    assert_ne!(first.key(&[]), second.key(&[]));
    // Neither half of a secret is a constant, and the two
    // halves are drawn apart.
    let third = DetectionMemo::<u32>::new();
    assert_ne!(first.secret.0, third.secret.0);
    assert_ne!(first.secret.1, third.secret.1);
    assert_ne!(first.secret.0, first.secret.1);
    // A result one memo keeps is a miss in another.
    first.put(first.key(&parts), &[1]);
    assert_eq!(first.get(&first.key(&parts)), Some(vec![1]));
    assert_eq!(second.get(&second.key(&parts)), None);
}

#[test]
fn a_memo_gives_the_same_input_the_same_key_every_time() {
    // The secret is drawn once, when the memo is made: the
    // same parts get the same key on every call and from
    // every thread, so a repeat finds its entry.
    let memo = Arc::new(DetectionMemo::<u32>::new());
    let parts: [&[u8]; 3] = [b"pixels", &640u32.to_le_bytes(), &480u32.to_le_bytes()];
    let first = memo.key(&parts);
    assert_eq!(memo.key(&parts), first);
    let from_another_thread = {
        let memo = memo.clone();
        std::thread::spawn(move || {
            memo.key(&[b"pixels", &640u32.to_le_bytes(), &480u32.to_le_bytes()])
        })
        .join()
        .unwrap()
    };
    assert_eq!(from_another_thread, first);
    memo.put(first, &[7]);
    assert_eq!(memo.get(&memo.key(&parts)), Some(vec![7]));
}

#[test]
fn a_hit_returns_what_was_put_and_a_miss_nothing() {
    let memo = DetectionMemo::<u32>::new();
    assert_eq!(memo.get(&key(1)), None);
    memo.put(key(1), &[10, 11]);
    memo.put(key(2), &[]);
    assert_eq!(memo.get(&key(1)), Some(vec![10, 11]));
    // An empty result is a result: no faces is an answer.
    assert_eq!(memo.get(&key(2)), Some(Vec::new()));
    assert_eq!(memo.get(&key(3)), None);
    // A second answer under a key replaces the first.
    memo.put(key(1), &[12]);
    assert_eq!(memo.get(&key(1)), Some(vec![12]));
    assert_eq!(memo.entries.lock().unwrap().len(), 2);
}

#[test]
fn the_memo_never_holds_more_than_its_capacity_and_drops_the_least_recently_used() {
    let memo = DetectionMemo::<u32>::with_capacity(3);
    for n in 0..3 {
        memo.put(key(n), &[n]);
    }
    assert_eq!(keys_in_order(&memo), [key(2), key(1), key(0)]);
    // A hit makes its entry the most recently used.
    assert_eq!(memo.get(&key(0)), Some(vec![0]));
    assert_eq!(keys_in_order(&memo), [key(0), key(2), key(1)]);
    // A fourth result pushes out the least recently used,
    // which is now 1, not 0.
    memo.put(key(3), &[3]);
    assert_eq!(keys_in_order(&memo), [key(3), key(0), key(2)]);
    assert_eq!(memo.get(&key(1)), None);
    assert_eq!(memo.get(&key(0)), Some(vec![0]));
    // However many results pass through, it stays at its
    // capacity, and the survivors are the last three.
    for n in 10..1000 {
        memo.put(key(n), &[n]);
        assert!(memo.entries.lock().unwrap().len() <= 3);
    }
    assert_eq!(keys_in_order(&memo), [key(999), key(998), key(997)]);
    // A provider's memo holds 256 results.
    let memo = DetectionMemo::<u32>::new();
    for n in 0..300 {
        memo.put(key(n), &[n]);
    }
    assert_eq!(memo.entries.lock().unwrap().len(), MEMO_ENTRIES);
    assert_eq!(MEMO_ENTRIES, 256);
    assert_eq!(memo.get(&key(43)), None);
    assert_eq!(memo.get(&key(44)), Some(vec![44]));
    // A memo of no entries remembers nothing.
    let none = DetectionMemo::<u32>::with_capacity(0);
    none.put(key(1), &[1]);
    assert_eq!(none.get(&key(1)), None);
}

#[test]
fn a_result_with_too_many_boxes_is_not_remembered() {
    // The bound on entries is not a bound on memory unless
    // an entry is bounded too.
    let memo = DetectionMemo::<u32>::new();
    let most = vec![7u32; MEMO_BOXES];
    memo.put(key(1), &most);
    assert_eq!(memo.get(&key(1)), Some(most));
    memo.put(key(2), &vec![7u32; MEMO_BOXES + 1]);
    assert_eq!(memo.get(&key(2)), None);
    // And it does not push a remembered result out.
    assert_eq!(memo.entries.lock().unwrap().len(), 1);
}

#[test]
fn a_poisoned_lock_is_a_miss_and_never_an_error() {
    let memo = Arc::new(DetectionMemo::<u32>::new());
    memo.put(key(1), &[1]);
    let poisoner = memo.clone();
    let panicked = std::thread::spawn(move || {
        let _held = poisoner.entries.lock().unwrap();
        panic!("poison the memo's lock");
    })
    .join();
    assert!(panicked.is_err());
    assert!(memo.entries.is_poisoned());
    // Every lookup misses, so the caller runs the model;
    // nothing is remembered; nothing panics.
    assert_eq!(memo.get(&key(1)), None);
    memo.put(key(2), &[2]);
    assert_eq!(memo.get(&key(2)), None);
}

#[test]
fn concurrent_lookups_and_results_keep_the_memo_whole() {
    // Eight threads look up and remember 64 keys in a memo
    // of 16: every hit is the value its key was given, and
    // the memo ends within its capacity with no key twice.
    let memo = Arc::new(DetectionMemo::<u32>::with_capacity(16));
    let threads: Vec<_> = (0..8u32)
        .map(|t| {
            let memo = memo.clone();
            std::thread::spawn(move || {
                let mut hits = 0u32;
                for round in 0..2000u32 {
                    let n = (round * 7 + t * 13) % 64;
                    match memo.get(&key(n)) {
                        Some(found) => {
                            assert_eq!(found, [n, n + 1]);
                            hits += 1;
                        }
                        None => memo.put(key(n), &[n, n + 1]),
                    }
                }
                hits
            })
        })
        .collect();
    let hits: u32 = threads.into_iter().map(|t| t.join().unwrap()).sum();
    assert!(hits > 0);
    let keys = keys_in_order(&memo);
    assert!(keys.len() <= 16);
    for (i, k) in keys.iter().enumerate() {
        assert!(!keys[..i].contains(k), "a key is in the memo twice");
    }
}

#[test]
fn what_the_memo_holds_goes_when_the_memo_goes() {
    // The memo is a field of its provider: there is no
    // process-wide table for a result to outlive it in.
    let boxes = Arc::new(());
    let memo = DetectionMemo::<Arc<()>>::new();
    memo.put(key(1), &[boxes.clone(), boxes.clone()]);
    assert_eq!(Arc::strong_count(&boxes), 3);
    assert_eq!(memo.get(&key(1)).map(|found| found.len()), Some(2));
    assert_eq!(Arc::strong_count(&boxes), 3);
    drop(memo);
    assert_eq!(Arc::strong_count(&boxes), 1);
}

#[test]
fn an_entry_is_a_key_and_boxes_and_nothing_else() {
    // No pixels: an entry of a face memo is its 16-byte key
    // and its boxes, 20 bytes each.
    assert_eq!(std::mem::size_of::<MemoKey>(), 16);
    assert_eq!(std::mem::size_of::<crate::ai::FaceBox>(), 20);
    assert_eq!(
        std::mem::size_of::<(MemoKey, Vec<crate::ai::FaceBox>)>(),
        16 + std::mem::size_of::<Vec<crate::ai::FaceBox>>()
    );
}
