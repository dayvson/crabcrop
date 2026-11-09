use super::*;
use crate::ai::moderation::VIT_STATS;

#[test]
fn crop224_resize_dims_clamps_the_long_side() {
    // Issue #359: the short side scales to 224 and the
    // long side clamps at the cap, so an extreme aspect
    // no longer produces an unbounded intermediate.
    assert_eq!(crop224_resize_dims(16384, 1), (4096, 224));
    assert_eq!(crop224_resize_dims(1, 16384), (224, 4096));
    // Ordinary aspects are untouched.
    assert_eq!(crop224_resize_dims(100, 50), (448, 224));
    assert_eq!(crop224_resize_dims(50, 100), (224, 448));
    assert_eq!(crop224_resize_dims(224, 224), (224, 224));
}

#[test]
fn preprocess_clip_survives_an_extreme_aspect() {
    // The eval case: a 16384x1 source previously drove
    // a 2.47 GB intermediate allocation and aborted.
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        16384,
        1,
        image::Rgb([255, 0, 0]),
    ));
    let input = crop224_preprocess(&img, CLIP_STATS);
    assert_eq!(input.len(), 3 * 224 * 224);
    assert!(input.iter().all(|v| v.is_finite()));
    let input = crop224_preprocess(&img, VIT_STATS);
    assert_eq!(input.len(), 3 * 224 * 224);
}

#[test]
#[allow(clippy::excessive_precision)] // the model's published constants
fn preprocess_clip_normalizes_a_solid_image() {
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        100,
        50,
        image::Rgb([255, 0, 0]),
    ));
    let input = crop224_preprocess(&img, CLIP_STATS);
    assert_eq!(input.len(), 3 * 224 * 224);
    let r = (1.0f32 - 0.48145466) / 0.26862954;
    let g = (0.0f32 - 0.4578275) / 0.26130258;
    assert!((input[0] - r).abs() < 1e-4, "red plane {}", input[0]);
    assert!((input[224 * 224] - g).abs() < 1e-4, "green plane");
    assert!(input.iter().all(|v| v.is_finite()));
}

#[test]
fn crop224_preprocess_gives_clip_and_vit_the_same_crop() {
    // One body serves both models (issue #484): undoing
    // each model's normalization recovers the same
    // 224x224 crop, so the statistics are all that differ.
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(300, 200, |x, y| {
        image::Rgb([x as u8, y as u8, (x ^ y) as u8])
    }));
    let clip = crop224_preprocess(&img, CLIP_STATS);
    let vit = crop224_preprocess(&img, VIT_STATS);
    assert_eq!(clip.len(), 3 * 224 * 224);
    assert_eq!(vit.len(), clip.len());
    let plane = 224 * 224;
    for (i, (a, b)) in clip.iter().zip(&vit).enumerate() {
        let c = i / plane;
        let from_clip = a * CLIP_STATS.1[c] + CLIP_STATS.0[c];
        let from_vit = b * VIT_STATS.1[c] + VIT_STATS.0[c];
        assert!((from_clip - from_vit).abs() < 1e-5, "value {i}");
    }
}

#[test]
fn read_labels_skips_comments_and_blanks() {
    let dir = crate::test_support::scratch_dir("clip-labels");
    let path = dir.join("labels.txt");
    std::fs::write(&path, "# heading\n\nocean\n  sky  \n#kite\n").unwrap();
    let got = read_labels(path.to_str().unwrap()).unwrap();
    assert_eq!(got, ["ocean", "sky"]);
    let empty = dir.join("empty.txt");
    std::fs::write(&empty, "# only comments\n\n").unwrap();
    assert!(read_labels(empty.to_str().unwrap()).is_err());
}

#[test]
fn pick_embedding_prefers_the_named_projection_outputs() {
    let hidden: ModelOutput = (
        "last_hidden_state".to_string(),
        vec![1, 2, 3],
        (0..6).map(|v| v as f32).collect(),
    );
    let pooled: ModelOutput = ("pooled_output".to_string(), vec![1, 3], vec![7.0, 8.0, 9.0]);
    let image: ModelOutput = ("image_embeds".to_string(), vec![1, 3], vec![1.0, 2.0, 3.0]);
    // image_embeds wins over pooled_output and the rest.
    let outputs = vec![hidden.clone(), pooled.clone(), image];
    assert_eq!(pick_embedding(&outputs, None).unwrap(), [1.0, 2.0, 3.0]);
    // text_embeds is the projection pick on the text side.
    let text: ModelOutput = ("text_embeds".to_string(), vec![1, 3], vec![4.0, 5.0, 6.0]);
    let outputs = vec![hidden.clone(), text];
    assert_eq!(pick_embedding(&outputs, Some(1)).unwrap(), [4.0, 5.0, 6.0]);
    // With no named projection, pooled_output is next.
    let outputs = vec![hidden, pooled];
    assert_eq!(pick_embedding(&outputs, None).unwrap(), [7.0, 8.0, 9.0]);
}

#[test]
fn pick_embedding_reads_the_requested_row_of_a_hidden_state() {
    // A [1, 3, 2] hidden state with no projection or
    // pooled output: the fallback reads one row of the
    // first output.
    let outputs: Vec<ModelOutput> = vec![(
        "last_hidden_state".to_string(),
        vec![1, 3, 2],
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    )];
    // The text path asks for the EOT row explicitly.
    assert_eq!(pick_embedding(&outputs, Some(2)).unwrap(), [4.0, 5.0]);
    // The image path takes row 0.
    assert_eq!(pick_embedding(&outputs, None).unwrap(), [0.0, 1.0]);
    // A rank-2 first output falls back to its first row.
    let outputs: Vec<ModelOutput> = vec![(
        "logits".to_string(),
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )];
    assert_eq!(pick_embedding(&outputs, None).unwrap(), [1.0, 2.0, 3.0]);
}

#[test]
fn pick_embedding_rejects_missing_and_undersized_outputs() {
    // No outputs at all is a BadOutput.
    let err = pick_embedding(&[], None).unwrap_err();
    assert!(matches!(err, AiError::BadOutput(_)), "{err}");
    assert!(err.to_string().contains("no outputs"), "{err}");
    // An EOT row past the end of the declared shape is
    // a BadOutput, not a short or padded embedding.
    let outputs: Vec<ModelOutput> = vec![(
        "last_hidden_state".to_string(),
        vec![1, 2, 2],
        vec![0.0, 1.0, 2.0, 3.0],
    )];
    let err = pick_embedding(&outputs, Some(5)).unwrap_err();
    assert!(matches!(err, AiError::BadOutput(_)), "{err}");
    assert!(
        err.to_string().contains("smaller than its declared shape"),
        "{err}"
    );
}

#[test]
fn first_row_takes_the_last_dimension_and_falls_back_to_all_data() {
    // A [2, 3] output: the first row is the first 3.
    assert_eq!(
        first_row(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [1.0, 2.0, 3.0]
    );
    // A rank-1 output has no row structure: everything.
    assert_eq!(first_row(&[3], &[1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
    // A declared row wider than the data: everything,
    // rather than a panic or a truncated row.
    assert_eq!(first_row(&[1, 10], &[1.0, 2.0]), [1.0, 2.0]);
}

#[test]
fn normalize_scales_to_unit_length_and_keeps_the_zero_vector() {
    let out = normalize(&[3.0, 4.0]);
    assert!(
        (out[0] - 0.6).abs() < 1e-6 && (out[1] - 0.8).abs() < 1e-6,
        "{out:?}"
    );
    let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
    // The zero vector has no direction: it is returned
    // as-is instead of dividing by zero.
    assert_eq!(normalize(&[0.0, 0.0]), [0.0, 0.0]);
}

/// Reference token ids for label-shaped text under the
/// CLIP ViT-B/32 tokenizer (issue #414). Not derived
/// from this crate: OpenAI's reference tokenizer over
/// its `bpe_simple_vocab_16e6` merges and Hugging Face
/// `tokenizers` over a CLIP `tokenizer.json` both produce
/// every row. Whitespace separates pieces and is
/// never a token itself, so a phrase is exactly its
/// words' ids between SOT (49406) and EOT (49407); the
/// multi-word rows are what a stray space token (#404)
/// would shift. The last rows cover the other piece
/// kinds: a contraction, single digits, punctuation
/// runs, and whitespace and case normalization.
const CLIP_REFERENCE_IDS: &[(&str, &[i64])] = &[
    (
        "a photo of a dog",
        &[49406, 320, 1125, 539, 320, 1929, 49407],
    ),
    ("kite surfing", &[49406, 19867, 15762, 49407]),
    ("palm tree", &[49406, 8612, 2677, 49407]),
    ("city skyline", &[49406, 1305, 14606, 49407]),
    ("aerial view", &[49406, 12440, 1093, 49407]),
    ("night scene", &[49406, 930, 3562, 49407]),
    ("dog", &[49406, 1929, 49407]),
    ("it's me", &[49406, 585, 568, 614, 49407]),
    ("top 10", &[49406, 1253, 272, 271, 49407]),
    ("sun-set!", &[49406, 2176, 268, 1167, 256, 49407]),
    ("  Kite   SURFING ", &[49406, 19867, 15762, 49407]),
];

/// Encode every [`CLIP_REFERENCE_IDS`] row with the real
/// tokenizer file and compare the unpadded ids (mask 1),
/// reporting every mismatch at once.
fn assert_clip_reference_ids(tokenizer_path: &str) {
    let tokenizer = crate::tokenizer::ClipTokenizer::load(tokenizer_path).unwrap();
    let mismatches: Vec<String> = CLIP_REFERENCE_IDS
        .iter()
        .filter_map(|(text, expected)| {
            let (ids, mask) = tokenizer.encode(text).unwrap();
            let got = &ids[..mask.iter().filter(|&&m| m == 1).count()];
            (got != *expected).then(|| format!("{text:?}: got {got:?}, reference {expected:?}"))
        })
        .collect();
    assert!(
        mismatches.is_empty(),
        "the CLIP tokenizer differs from the reference ids:\n{}",
        mismatches.join("\n")
    );
}

/// The tokenizer half of the live CLIP check, runnable
/// without the runtime or the models. Ignored: needs
/// the export's tokenizer file:
/// CRABCROP_TEST_CLIP_TOKENIZER=...
#[test]
#[ignore = "needs the CLIP tokenizer.json (env-configured)"]
fn clip_tokenizer_matches_the_reference_ids() {
    assert_clip_reference_ids(&std::env::var("CRABCROP_TEST_CLIP_TOKENIZER").unwrap());
}

/// The labels of the ai-models workflow's set that
/// describe its fixture photo (Ultralytics' bus.jpg:
/// people on a city street in front of a bus). Which of
/// them wins depends on the export, so the live check
/// is membership: a broken preprocess, shifted label
/// embeddings, or a mismatched model pair puts an
/// unrelated label (a landscape, an animal, food) on top.
const CLIP_FIXTURE_TOP_LABELS: &[&str] = &[
    "people",
    "street",
    "buildings",
    "city skyline",
    "portrait",
    "car",
];

/// Live check with a real CLIP export: the tokenizer
/// matches the reference ids, and tagging the fixture
/// photo puts a label that describes it on top. Ignored:
/// needs the runtime, both models, the tokenizer, a
/// label file, and a photo (the expectations are the
/// ai-models workflow's label set and bus.jpg):
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_CLIP_IMAGE_MODEL=...
/// CRABCROP_TEST_CLIP_TEXT_MODEL=... CRABCROP_TEST_CLIP_TOKENIZER=...
/// CRABCROP_TEST_CLIP_LABELS=... CRABCROP_TEST_IMAGE=...
#[test]
#[ignore = "needs the ONNX runtime and CLIP model files (env-configured)"]
fn clip_live_tags_a_photo() {
    let var = |name: &str| std::env::var(name).unwrap();
    // The labels are embedded from these ids at load, so
    // a tokenizer regression fails here by name instead of
    // as quietly shifted scores.
    assert_clip_reference_ids(&var("CRABCROP_TEST_CLIP_TOKENIZER"));
    let provider = OnnxClipProvider::load(
        Some(&var("CRABCROP_TEST_ORT_DYLIB")),
        &var("CRABCROP_TEST_CLIP_IMAGE_MODEL"),
        &var("CRABCROP_TEST_CLIP_TEXT_MODEL"),
        &var("CRABCROP_TEST_CLIP_TOKENIZER"),
        &var("CRABCROP_TEST_CLIP_LABELS"),
    )
    .unwrap();
    let img = image::open(var("CRABCROP_TEST_IMAGE")).unwrap();
    let scores = provider.scores(&img).unwrap();
    assert_eq!(scores.len(), provider.label_count());
    assert!(scores.iter().all(|s| s.score.is_finite()), "{scores:?}");
    let mut ranked: Vec<&TagScore> = scores.iter().collect();
    ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
    println!("ranked: {ranked:?}");
    assert!(
        CLIP_FIXTURE_TOP_LABELS.contains(&ranked[0].label.as_str()),
        "top label {:?} does not describe the fixture photo; ranked: {ranked:?}",
        ranked[0].label
    );
}

/// Write bytes to a fresh temp file and return the
/// scratch directory it lives in and its path: the
/// load() failure tests need real files (a label list,
/// a garbage tokenizer) without committing fixtures
/// (issue #526).
fn temp_file(tag: &str, contents: &[u8]) -> (crate::test_support::ScratchDir, String) {
    let dir = crate::test_support::scratch_dir(&format!("ai-load-{tag}"));
    let path = dir
        .join(format!("{tag}.bin"))
        .to_string_lossy()
        .into_owned();
    std::fs::write(&path, contents).unwrap();
    (dir, path)
}

/// A path that does not exist: a name inside a scratch
/// directory that is removed again before this returns.
fn missing_path(tag: &str) -> String {
    crate::test_support::scratch_dir(&format!("ai-load-missing-{tag}"))
        .join(format!("{tag}.onnx"))
        .to_string_lossy()
        .into_owned()
}

/// Assert a load() result failed as AiError::Load and
/// return the message for wording checks.
fn expect_load<T>(result: Result<T, AiError>) -> String {
    match result {
        Err(AiError::Load(message)) => message,
        Err(other) => panic!("expected a load error, got {other}"),
        Ok(_) => panic!("expected a load error, but the load succeeded"),
    }
}

/// Load-failure coverage (#179, item 2). The label and
/// tokenizer files are read before any session is
/// built, so these run in the plain gate with no
/// runtime at all: a missing label file is Load.
#[test]
fn clip_load_missing_labels_file_is_a_load_error() {
    let missing = missing_path("clip-labels");
    let message = expect_load(OnnxClipProvider::load(
        None, &missing, &missing, &missing, &missing,
    ));
    assert!(message.contains("cannot read the label file"), "{message}");
}

/// A label file with only comments and blank lines has
/// no labels: Load, still before the runtime is touched.
#[test]
fn clip_load_empty_labels_file_is_a_load_error() {
    let (_labels, labels) = temp_file("clip-labels-empty", b"# only a comment\n\n");
    let missing = missing_path("clip-empty");
    let message = expect_load(OnnxClipProvider::load(
        None, &missing, &missing, &missing, &labels,
    ));
    assert!(message.contains("contains no labels"), "{message}");
}

/// With valid labels, a missing tokenizer file is the
/// next pre-runtime failure: Load.
#[test]
fn clip_load_missing_tokenizer_file_is_a_load_error() {
    let (_labels, labels) = temp_file("clip-labels-ok", b"cat\ndog\n");
    let missing = missing_path("clip-tokenizer");
    let message = expect_load(OnnxClipProvider::load(
        None, &missing, &missing, &missing, &labels,
    ));
    assert!(
        message.contains("cannot read the tokenizer file"),
        "{message}"
    );
}

/// A tokenizer file that is not BPE JSON fails the
/// same pre-runtime way.
#[test]
fn clip_load_garbage_tokenizer_file_is_a_load_error() {
    let (_labels, labels) = temp_file("clip-labels-ok2", b"cat\ndog\n");
    let (_tokenizer, tokenizer) = temp_file("clip-tokenizer-garbage", b"{ this is not json");
    let missing = missing_path("clip-garbage-tok");
    let message = expect_load(OnnxClipProvider::load(
        None, &missing, &missing, &tokenizer, &labels,
    ));
    assert!(message.contains("tokenizer"), "{message}");
}

/// With the files in place, a missing text model fails
/// in the session builder: Load.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_CLIP_IMAGE_MODEL=...
/// CRABCROP_TEST_CLIP_TOKENIZER=... CRABCROP_TEST_CLIP_LABELS=...
/// cargo test -- --ignored clip_load_missing_text_model
#[test]
#[ignore = "needs the ONNX runtime and CLIP model files (env-configured)"]
fn clip_load_missing_text_model_is_a_load_error() {
    let var = |name: &str| std::env::var(name).unwrap();
    let message = expect_load(OnnxClipProvider::load(
        Some(&var("CRABCROP_TEST_ORT_DYLIB")),
        &var("CRABCROP_TEST_CLIP_IMAGE_MODEL"),
        &missing_path("clip-text"),
        &var("CRABCROP_TEST_CLIP_TOKENIZER"),
        &var("CRABCROP_TEST_CLIP_LABELS"),
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// A garbage text model file fails the same way.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_CLIP_IMAGE_MODEL=...
/// CRABCROP_TEST_CLIP_TOKENIZER=... CRABCROP_TEST_CLIP_LABELS=...
/// cargo test -- --ignored clip_load_garbage_text_model
#[test]
#[ignore = "needs the ONNX runtime and CLIP model files (env-configured)"]
fn clip_load_garbage_text_model_is_a_load_error() {
    let var = |name: &str| std::env::var(name).unwrap();
    let (_model, garbage) = temp_file("clip-text-garbage", b"this is not an onnx model at all");
    let message = expect_load(OnnxClipProvider::load(
        Some(&var("CRABCROP_TEST_ORT_DYLIB")),
        &var("CRABCROP_TEST_CLIP_IMAGE_MODEL"),
        &garbage,
        &var("CRABCROP_TEST_CLIP_TOKENIZER"),
        &var("CRABCROP_TEST_CLIP_LABELS"),
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// A missing image model is reached only after the
/// text side has embedded every label; still Load.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_CLIP_TEXT_MODEL=...
/// CRABCROP_TEST_CLIP_TOKENIZER=... CRABCROP_TEST_CLIP_LABELS=...
/// cargo test -- --ignored clip_load_missing_image_model
#[test]
#[ignore = "needs the ONNX runtime and CLIP model files (env-configured)"]
fn clip_load_missing_image_model_is_a_load_error() {
    let var = |name: &str| std::env::var(name).unwrap();
    let message = expect_load(OnnxClipProvider::load(
        Some(&var("CRABCROP_TEST_ORT_DYLIB")),
        &missing_path("clip-image"),
        &var("CRABCROP_TEST_CLIP_TEXT_MODEL"),
        &var("CRABCROP_TEST_CLIP_TOKENIZER"),
        &var("CRABCROP_TEST_CLIP_LABELS"),
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// A swapped pair (the vision model as the text model)
/// cannot embed the labels: embedding is part of
/// loading, so the failure is Load, not a request-time
/// Inference.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_CLIP_IMAGE_MODEL=...
/// CRABCROP_TEST_CLIP_TEXT_MODEL=... CRABCROP_TEST_CLIP_TOKENIZER=...
/// CRABCROP_TEST_CLIP_LABELS=...
/// cargo test -- --ignored clip_load_swapped_model_pair
#[test]
#[ignore = "needs the ONNX runtime and CLIP model files (env-configured)"]
fn clip_load_swapped_model_pair_is_a_load_error() {
    let var = |name: &str| std::env::var(name).unwrap();
    expect_load(OnnxClipProvider::load(
        Some(&var("CRABCROP_TEST_ORT_DYLIB")),
        &var("CRABCROP_TEST_CLIP_TEXT_MODEL"),
        &var("CRABCROP_TEST_CLIP_IMAGE_MODEL"),
        &var("CRABCROP_TEST_CLIP_TOKENIZER"),
        &var("CRABCROP_TEST_CLIP_LABELS"),
    ));
}
