//! The deterministic fuzz smoke (issue #146): every seed in
//! the corpus, every truncation of the representative seeds,
//! and single-byte mutations of a grammar-rich seed, all run
//! through the same invariant harness as the cargo-fuzz
//! target. Bounded and fast, so it belongs on the local
//! checklist (scripts/test_all.sh); the open-ended fuzzing
//! itself runs under cargo-fuzz on a nightly toolchain (see
//! fuzz/README.md).

use std::path::PathBuf;

fn corpus_inputs() -> Vec<String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/parse_request");
    let mut inputs: Vec<String> = std::fs::read_dir(&dir)
        .expect("the seed corpus directory exists")
        .filter_map(|e| e.ok())
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .collect();
    inputs.sort();
    assert!(inputs.len() >= 25, "the corpus should stay seeded");
    inputs
}

#[test]
fn seeds_satisfy_the_invariants() {
    for input in corpus_inputs() {
        crabcrop_fuzz::check(&input);
    }
}

#[test]
fn truncations_and_mutations_satisfy_the_invariants() {
    let rich = "/unsafe/filters:rotate(45.5):quality(80):format(webp)/300x200/example.com/img.jpg";
    // Every truncation.
    for end in 0..=rich.len() {
        if rich.is_char_boundary(end) {
            crabcrop_fuzz::check(&rich[..end]);
        }
    }
    // Single-byte substitutions with grammar-significant
    // characters.
    let mut bytes = rich.as_bytes().to_vec();
    for i in 0..bytes.len() {
        for replacement in [b'/', b':', b'(', b')', b'%', b'0', b'x', b'-', b'.'] {
            let original = bytes[i];
            bytes[i] = replacement;
            if let Ok(text) = std::str::from_utf8(&bytes) {
                crabcrop_fuzz::check(text);
            }
            bytes[i] = original;
        }
    }
}
