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

fn byte_corpus_inputs(name: &str) -> Vec<Vec<u8>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("corpus/{name}"));
    let mut inputs: Vec<Vec<u8>> = std::fs::read_dir(&dir)
        .expect("the byte seed corpus directory exists")
        .filter_map(|e| e.ok())
        .filter_map(|e| std::fs::read(e.path()).ok())
        .collect();
    inputs.sort();
    assert!(inputs.len() >= 4, "the {name} corpus should stay seeded");
    inputs
}

#[test]
fn decode_seeds_satisfy_the_invariants() {
    for input in byte_corpus_inputs("decode") {
        crabcrop_fuzz::check_decode(&input);
    }
}

#[test]
fn decode_truncations_and_mutations_satisfy_the_invariants() {
    // The smallest rich seed: a real PNG with header,
    // chunks, and pixel data, so truncations land inside
    // every structure the decoder walks.
    let seed = byte_corpus_inputs("decode")
        .into_iter()
        .find(|s| s.starts_with(b"\x89PNG"))
        .expect("a PNG seed");
    for end in 0..=seed.len() {
        crabcrop_fuzz::check_decode(&seed[..end]);
    }
    let mut bytes = seed.clone();
    for i in 0..bytes.len() {
        for replacement in [0x00u8, 0xff, 0x89, b'P', b'I', b't'] {
            let original = bytes[i];
            bytes[i] = replacement;
            crabcrop_fuzz::check_decode(&bytes);
            bytes[i] = original;
        }
    }
}

#[test]
fn source_meta_seeds_satisfy_the_invariants() {
    for input in byte_corpus_inputs("source_meta") {
        crabcrop_fuzz::check_meta(&input);
    }
}

#[test]
fn source_meta_truncations_and_mutations_satisfy_the_invariants() {
    // The EXIF seed's front: the metadata segments live
    // there, so truncations land inside the JPEG segments
    // and the TIFF stream the patcher walks. The sweep
    // works on the front 8 KB so it stays fast; the full
    // seeds run whole in the seeds test above.
    let seed: Vec<u8> = byte_corpus_inputs("source_meta")
        .into_iter()
        .find(|s| s.starts_with(b"\xff\xd8"))
        .expect("a JPEG seed")
        .into_iter()
        .take(8192)
        .collect();
    for end in 0..=seed.len() {
        crabcrop_fuzz::check_meta(&seed[..end]);
    }
    let mut bytes = seed.clone();
    for i in 0..bytes.len().min(4096) {
        for replacement in [0x00u8, 0xff, 0xe1, b'E', b'x'] {
            let original = bytes[i];
            bytes[i] = replacement;
            crabcrop_fuzz::check_meta(&bytes);
            bytes[i] = original;
        }
    }
}

#[test]
fn op_chain_seeds_satisfy_the_invariants() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/op_chain");
    let mut inputs: Vec<String> = std::fs::read_dir(&dir)
        .expect("the chain seed corpus directory exists")
        .filter_map(|e| e.ok())
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .collect();
    inputs.sort();
    assert!(inputs.len() >= 5, "the chain corpus should stay seeded");
    for input in &inputs {
        crabcrop_fuzz::check_chain(input);
    }
    // Mutations of a chain seed, with grammar-significant
    // replacements, like the parser sweep.
    let rich = inputs
        .iter()
        .max_by_key(|s| s.len())
        .expect("a chain seed")
        .clone();
    let mut bytes = rich.into_bytes();
    for i in 0..bytes.len() {
        for replacement in [b'/', b':', b'(', b')', b',', b'0', b'-'] {
            let original = bytes[i];
            bytes[i] = replacement;
            if let Ok(text) = std::str::from_utf8(&bytes) {
                crabcrop_fuzz::check_chain(text);
            }
            bytes[i] = original;
        }
    }
}

fn seal_corpus_inputs() -> Vec<String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/unseal");
    let mut inputs: Vec<String> = std::fs::read_dir(&dir)
        .expect("the seal seed corpus directory exists")
        .filter_map(|e| e.ok())
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .collect();
    inputs.sort();
    assert!(inputs.len() >= 8, "the seal corpus should stay seeded");
    inputs
}

#[test]
fn seal_seeds_satisfy_the_invariants() {
    for input in seal_corpus_inputs() {
        crabcrop_fuzz::check_seal(input.as_bytes());
    }
}

#[test]
fn seal_truncations_and_mutations_satisfy_the_invariants() {
    let remainder = "filters:rotate(45.5):quality(80):format(webp)/300x200/example.com/img.jpg";
    crabcrop_fuzz::check_seal(remainder.as_bytes());
    let token = crabcrop_fuzz::sample_token(remainder);
    for end in 0..=token.len() {
        if token.is_char_boundary(end) {
            crabcrop_fuzz::check_seal(token[..end].as_bytes());
        }
    }
    let mut bytes = token.into_bytes();
    for i in 0..bytes.len() {
        for replacement in [b'.', b':', b'A', b'0', b'_', b'-', b'/', b'%'] {
            let original = bytes[i];
            bytes[i] = replacement;
            if let Ok(text) = std::str::from_utf8(&bytes) {
                crabcrop_fuzz::check_seal(text.as_bytes());
            }
            bytes[i] = original;
        }
    }
}
