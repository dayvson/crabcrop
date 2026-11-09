use super::*;
use crate::ai::env_or_skip;

#[test]
#[allow(clippy::excessive_precision)] // the model's published constants
fn preprocess_vit_normalizes_a_solid_image() {
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        100,
        50,
        image::Rgb([255, 0, 0]),
    ));
    let input = crop224_preprocess(&img, VIT_STATS);
    assert_eq!(input.len(), 3 * 224 * 224);
    // (1 - 0.485) / 0.229 = 2.2489 in red, (0 - 0.456)
    // / 0.224 = -2.0357 in green.
    assert!((input[0] - 2.2489).abs() < 0.001, "red {}", input[0]);
    assert!(
        (input[224 * 224] - -2.0357).abs() < 0.001,
        "green {}",
        input[224 * 224]
    );
    assert!(input.iter().all(|v| v.is_finite()));
}

#[test]
fn softmax_turns_logits_into_probabilities() {
    let probs = softmax(&[1.0, 2.0, 3.0]);
    assert_eq!(probs.len(), 3);
    assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    // Equal logits split evenly.
    let flat = softmax(&[0.5, 0.5]);
    assert!((flat[0] - 0.5).abs() < 1e-6);
    // Large logits cannot overflow: the max is
    // subtracted before the exponentials.
    let big = softmax(&[1000.0, 1001.0]);
    assert!(big.iter().all(|v| v.is_finite()));
    assert!((big.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    // An empty row stays empty rather than dividing
    // by a zero total.
    assert!(softmax(&[]).is_empty());
}

#[test]
fn pick_logits_prefers_the_logits_output_and_takes_its_first_row() {
    let outputs = vec![
        ("hidden".to_string(), vec![1, 3], vec![9.0, 9.0, 9.0]),
        ("logits".to_string(), vec![1, 2], vec![1.5, -0.5]),
    ];
    assert_eq!(pick_logits(&outputs).unwrap(), vec![1.5, -0.5]);
    // Without a named output, the first one is used.
    let outputs = vec![("out".to_string(), vec![2], vec![0.25, 0.75])];
    assert_eq!(pick_logits(&outputs).unwrap(), vec![0.25, 0.75]);
}

#[test]
fn pick_logits_rejects_missing_and_empty_outputs() {
    let err = pick_logits(&[]).unwrap_err();
    assert!(matches!(err, AiError::BadOutput(_)), "{err}");
    let outputs = vec![("logits".to_string(), vec![1, 0], Vec::new())];
    let err = pick_logits(&outputs).unwrap_err();
    assert!(matches!(err, AiError::BadOutput(_)), "{err}");
}

#[test]
fn moderation_labels_default_to_the_falconsai_pair_for_two_outputs() {
    let labels = resolve_moderation_labels(None, 2, "model.onnx").unwrap();
    assert_eq!(labels, vec!["normal".to_string(), "nsfw".to_string()]);
}

#[test]
fn moderation_labels_require_a_file_for_other_output_counts() {
    let err = resolve_moderation_labels(None, 5, "model.onnx").unwrap_err();
    assert!(matches!(err, AiError::Load(_)), "{err}");
    assert!(
        err.to_string().contains("CRABCROP_MODERATION_LABELS"),
        "{err}"
    );
}

#[test]
fn moderation_labels_must_match_the_output_count() {
    let named = vec!["safe".to_string(), "explicit".to_string()];
    let labels = resolve_moderation_labels(Some(named.clone()), 2, "model.onnx").unwrap();
    assert_eq!(labels, named);
    let err =
        resolve_moderation_labels(Some(vec!["only".to_string()]), 2, "model.onnx").unwrap_err();
    assert!(matches!(err, AiError::Load(_)), "{err}");
    assert!(
        err.to_string().contains("name one label per output"),
        "{err}"
    );
}

#[derive(Debug)]
struct StubTagger(Result<Vec<TagScore>, AiError>);

impl TagProvider for StubTagger {
    fn scores(&self, _img: &DynamicImage) -> Result<Vec<TagScore>, AiError> {
        self.0.clone()
    }
}

fn solid() -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        16,
        16,
        image::Rgb([120, 120, 120]),
    ))
}

#[test]
fn clip_moderation_reports_signals_and_no_verdicts() {
    let tagger = StubTagger(Ok(vec![TagScore {
        label: "a photo of a weapon".to_string(),
        score: 0.24,
    }]));
    let provider =
        ClipModerationProvider::new(std::sync::Arc::new(tagger), "clip.onnx".to_string());
    let report = provider.moderate(&solid()).unwrap();
    assert_eq!(report.model, "clip.onnx");
    assert!(report.scores.is_empty());
    assert_eq!(report.signals.len(), 1);
    assert_eq!(report.signals[0].label, "a photo of a weapon");
}

#[test]
fn clip_moderation_propagates_tagger_failures() {
    let tagger = StubTagger(Err(AiError::Inference("boom".to_string())));
    let provider =
        ClipModerationProvider::new(std::sync::Arc::new(tagger), "clip.onnx".to_string());
    let err = provider.moderate(&solid()).unwrap_err();
    assert!(matches!(err, AiError::Inference(_)), "{err}");
}

#[test]
fn model_identities_report_file_names_not_paths() {
    // Meta names a model by its file name: the
    // configured path is the operator's filesystem
    // layout and never reaches a response. Every
    // provider that reports a model identity follows
    // the rule the OCR provider set (issue #268).
    assert_eq!(
        model_display_name("/srv/models/clip_vision.onnx"),
        "clip_vision.onnx"
    );
    assert_eq!(model_display_name("clip_vision.onnx"), "clip_vision.onnx");
    let tagger = StubTagger(Ok(vec![TagScore {
        label: "x".to_string(),
        score: 0.5,
    }]));
    let provider = ClipModerationProvider::new(
        std::sync::Arc::new(tagger),
        "/srv/models/clip_vision.onnx".to_string(),
    );
    let report = provider.moderate(&solid()).unwrap();
    assert_eq!(report.model, "clip_vision.onnx");
}

/// Write bytes to a fresh temp file and return the
/// scratch directory it lives in and its path: the
/// load() failure tests need real files (a garbage
/// model) without committing fixtures (issue #526).
fn temp_file(tag: &str, contents: &[u8]) -> (crate::test_support::ScratchDir, String) {
    let dir = crate::test_support::scratch_dir(&format!("moderation-load-{tag}"));
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
    crate::test_support::scratch_dir(&format!("moderation-load-missing-{tag}"))
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

/// Real-model verification, run manually with the
/// converted Falconsai artifact in place:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_MODERATION_MODEL=...
/// cargo test -- --ignored moderation_with_the_real_model
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn moderation_with_the_real_model() {
    let Some(dylib) = env_or_skip("CRABCROP_TEST_ORT_DYLIB") else {
        return;
    };
    let Some(model) = env_or_skip("CRABCROP_TEST_MODERATION_MODEL") else {
        return;
    };
    let provider = OnnxModerationProvider::load(Some(&dylib), &model, None).unwrap();
    // No labels configured: the Falconsai pair applies.
    assert_eq!(provider.labels(), ["normal", "nsfw"]);
    let bytes = include_bytes!("../../../tests/fixtures/portrait.jpg");
    let img = image::load_from_memory(bytes).unwrap();
    let report = provider.moderate(&img).unwrap();
    // Meta names a model by its file name, not the operator's path.
    assert_eq!(report.model, model_display_name(&model));
    assert!(report.signals.is_empty());
    assert_eq!(report.scores.len(), 2);
    assert_eq!(report.scores[0].label, "normal");
    assert_eq!(report.scores[1].label, "nsfw");
    // The spike measured this fixture at normal 0.9994+
    // through the reference pipeline; the bound leaves
    // room for build-to-build wobble.
    assert!(report.scores[0].score > 0.99, "scores: {:?}", report.scores);
    assert!(report.scores[1].score < 0.01, "scores: {:?}", report.scores);
    let total: f32 = report.scores.iter().map(|s| s.score).sum();
    assert!((total - 1.0).abs() < 1e-3, "total {total}");
    // A solid image is classified too, with the same
    // probability contract.
    let report = provider.moderate(&solid()).unwrap();
    assert!(
        report.scores.iter().all(|s| (0.0..=1.0).contains(&s.score)),
        "scores: {:?}",
        report.scores
    );
}

/// Operator labels rename the model's outputs when
/// their count matches:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_MODERATION_MODEL=...
/// cargo test -- --ignored moderation_labels_rename
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn moderation_labels_rename_the_outputs() {
    let Some(dylib) = env_or_skip("CRABCROP_TEST_ORT_DYLIB") else {
        return;
    };
    let Some(model) = env_or_skip("CRABCROP_TEST_MODERATION_MODEL") else {
        return;
    };
    let named = vec!["safe".to_string(), "explicit".to_string()];
    let provider = OnnxModerationProvider::load(Some(&dylib), &model, Some(named)).unwrap();
    assert_eq!(provider.labels(), ["safe", "explicit"]);
    let report = provider.moderate(&solid()).unwrap();
    assert_eq!(report.scores[0].label, "safe");
    assert_eq!(report.scores[1].label, "explicit");
    // A count that disagrees with the model is a load
    // error, not a mislabeled response.
    let err = expect_load(OnnxModerationProvider::load(
        Some(&dylib),
        &model,
        Some(vec!["only".to_string()]),
    ));
    assert!(err.contains("name one label per output"), "{err}");
}

/// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored moderation_load_missing
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn moderation_load_missing_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let message = expect_load(OnnxModerationProvider::load(
        Some(&dylib),
        &missing_path("moderation"),
        None,
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored moderation_load_garbage
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn moderation_load_garbage_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let (_model, garbage) = temp_file("moderation-garbage", b"this is not an onnx model at all");
    let message = expect_load(OnnxModerationProvider::load(Some(&dylib), &garbage, None));
    assert!(message.contains("cannot load the model"), "{message}");
}
