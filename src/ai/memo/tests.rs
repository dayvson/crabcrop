use super::*;
use std::sync::Arc;

fn key(n: u32) -> MemoKey {
    MemoKey::of(&[&n.to_le_bytes()])
}

fn keys_in_order<B>(memo: &DetectionMemo<B>) -> Vec<MemoKey> {
    memo.entries.lock().unwrap().iter().map(|e| e.0).collect()
}

#[test]
fn a_key_is_the_digest_of_its_parts_and_their_lengths() {
    // The same parts give the same key; any part that
    // differs, and the same bytes cut into other parts,
    // give another.
    let a = MemoKey::of(&[b"pixels", &640u32.to_le_bytes(), &480u32.to_le_bytes()]);
    assert_eq!(
        a,
        MemoKey::of(&[b"pixels", &640u32.to_le_bytes(), &480u32.to_le_bytes()])
    );
    assert_ne!(
        a,
        MemoKey::of(&[b"pixelz", &640u32.to_le_bytes(), &480u32.to_le_bytes()])
    );
    assert_ne!(
        a,
        MemoKey::of(&[b"pixels", &641u32.to_le_bytes(), &480u32.to_le_bytes()])
    );
    assert_ne!(
        a,
        MemoKey::of(&[b"pixels", &480u32.to_le_bytes(), &640u32.to_le_bytes()])
    );
    assert_ne!(MemoKey::of(&[b"ab", b"c"]), MemoKey::of(&[b"a", b"bc"]));
    assert_ne!(MemoKey::of(&[b"abc"]), MemoKey::of(&[b"abc", b""]));
    // It is SHA-256: one part of no bytes hashes its
    // eight-byte length, a known digest.
    use sha2::{Digest, Sha256};
    let want: [u8; 32] = Sha256::digest(0u64.to_le_bytes()).into();
    assert_eq!(MemoKey::of(&[b""]).0, want);
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
    // No pixels: an entry of a face memo is its 32-byte key
    // and its boxes, 20 bytes each.
    assert_eq!(std::mem::size_of::<MemoKey>(), 32);
    assert_eq!(std::mem::size_of::<crate::ai::FaceBox>(), 20);
    assert_eq!(
        std::mem::size_of::<(MemoKey, Vec<crate::ai::FaceBox>)>(),
        32 + std::mem::size_of::<Vec<crate::ai::FaceBox>>()
    );
}
