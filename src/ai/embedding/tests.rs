use super::*;
use crate::ai::{OnnxClipProvider, TagScore, env_or_skip};

fn solid() -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        16,
        16,
        image::Rgb([120, 120, 120]),
    ))
}

#[derive(Debug)]
struct EmbeddingTagger(Result<Option<Vec<f32>>, AiError>);

impl TagProvider for EmbeddingTagger {
    fn scores(&self, _img: &DynamicImage) -> Result<Vec<TagScore>, AiError> {
        Ok(Vec::new())
    }

    fn embedding(&self, _img: &DynamicImage) -> Result<Option<Vec<f32>>, AiError> {
        self.0.clone()
    }
}

#[derive(Debug)]
struct ScoresOnlyTagger;

impl TagProvider for ScoresOnlyTagger {
    fn scores(&self, _img: &DynamicImage) -> Result<Vec<TagScore>, AiError> {
        Ok(Vec::new())
    }
}

#[test]
fn tag_provider_embedding_defaults_to_none() {
    // A provider that is not an encoder has no
    // embedding to give; the default says so
    // instead of failing.
    assert_eq!(ScoresOnlyTagger.embedding(&solid()).unwrap(), None);
}

#[test]
fn clip_embedding_provider_surfaces_the_tagger_vector() {
    let tagger = EmbeddingTagger(Ok(Some(vec![0.6, 0.8])));
    let provider = ClipEmbeddingProvider::new(std::sync::Arc::new(tagger), "clip.onnx".to_string());
    let embedding = provider.embed(&solid()).unwrap();
    assert_eq!(embedding.model, "clip.onnx");
    assert_eq!(embedding.vector, vec![0.6, 0.8]);
}

#[test]
fn clip_embedding_provider_reports_a_tagger_without_embeddings() {
    let tagger = EmbeddingTagger(Ok(None));
    let provider = ClipEmbeddingProvider::new(std::sync::Arc::new(tagger), "clip.onnx".to_string());
    let err = provider.embed(&solid()).unwrap_err();
    assert!(matches!(err, AiError::Inference(_)), "{err}");
    assert!(
        err.to_string().contains("does not produce embeddings"),
        "{err}"
    );
}

#[test]
fn clip_embedding_provider_propagates_tagger_failures() {
    let tagger = EmbeddingTagger(Err(AiError::Inference("boom".to_string())));
    let provider = ClipEmbeddingProvider::new(std::sync::Arc::new(tagger), "clip.onnx".to_string());
    let err = provider.embed(&solid()).unwrap_err();
    assert!(matches!(err, AiError::Inference(_)), "{err}");
    assert!(err.to_string().contains("boom"), "{err}");
}

#[test]
fn aesthetic_head_scores_a_matching_embedding() {
    let head =
        AestheticHead::from_weights(vec![0.5, -0.25, 2.0], 0.75, "head.pth".to_string()).unwrap();
    assert_eq!(head.expected_dim(), 3);
    assert_eq!(head.identity(), "head.pth");
    // 0.5*2 + (-0.25)*4 + 2*0.5 + 0.75 = 1.75.
    let score = head.score(&[2.0, 4.0, 0.5]).unwrap();
    assert!((score - 1.75).abs() < 1e-6, "score {score}");
}

#[test]
fn aesthetic_head_dimension_mismatch_scores_none() {
    let head =
        AestheticHead::from_weights(vec![0.5, -0.25, 2.0], 0.75, "head.pth".to_string()).unwrap();
    assert_eq!(head.score(&[1.0, 2.0]), None);
    assert_eq!(head.score(&[1.0, 2.0, 3.0, 4.0]), None);
}

/// Real-model verification, run manually with the
/// CLIP pair and the LAION B/32 head the spike used:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_CLIP_IMAGE_MODEL=...
/// CRABCROP_TEST_CLIP_TEXT_MODEL=... CRABCROP_TEST_CLIP_TOKENIZER=...
/// CRABCROP_TEST_CLIP_LABELS=... CRABCROP_TEST_AESTHETIC_HEAD=...
/// cargo test -- --ignored embedding_and_aesthetic_with_the_real_models
/// Set CRABCROP_TEST_AESTHETIC_IMAGE to also score
/// that image (the spike's kite beach photo, 6.84).
#[test]
#[ignore = "needs the ONNX runtime, CLIP model files, and a head file (env-configured)"]
fn embedding_and_aesthetic_with_the_real_models() {
    let Some(dylib) = env_or_skip("CRABCROP_TEST_ORT_DYLIB") else {
        return;
    };
    let Some(image_model) = env_or_skip("CRABCROP_TEST_CLIP_IMAGE_MODEL") else {
        return;
    };
    let Some(text_model) = env_or_skip("CRABCROP_TEST_CLIP_TEXT_MODEL") else {
        return;
    };
    let Some(tokenizer) = env_or_skip("CRABCROP_TEST_CLIP_TOKENIZER") else {
        return;
    };
    let Some(labels) = env_or_skip("CRABCROP_TEST_CLIP_LABELS") else {
        return;
    };
    let Some(head_path) = env_or_skip("CRABCROP_TEST_AESTHETIC_HEAD") else {
        return;
    };
    let tagger =
        OnnxClipProvider::load(Some(&dylib), &image_model, &text_model, &tokenizer, &labels)
            .unwrap();
    let provider = ClipEmbeddingProvider::new(std::sync::Arc::new(tagger), image_model.clone());
    let head = AestheticHead::load(&head_path).unwrap();
    assert_eq!(head.expected_dim(), 512);
    let bytes = include_bytes!("../../../tests/fixtures/portrait.jpg");
    let img = image::load_from_memory(bytes).unwrap();
    let embedding = provider.embed(&img).unwrap();
    // Meta names a model by its file name, not the operator's path.
    assert_eq!(embedding.model, model_display_name(&image_model));
    assert_eq!(embedding.vector.len(), 512);
    let norm: f32 = embedding.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "norm {norm}");
    // The spike scored this fixture 7.03 through
    // the reference pipeline on the B/32 head; the
    // band leaves room for preprocessing wobble
    // between the two pipelines.
    let score = head.score(&embedding.vector).unwrap();
    assert!((6.5..=7.6).contains(&score), "portrait score {score}");
    if let Ok(path) = std::env::var("CRABCROP_TEST_AESTHETIC_IMAGE") {
        let img = image::open(&path).unwrap();
        let embedding = provider.embed(&img).unwrap();
        let score = head.score(&embedding.vector).unwrap();
        // The spike's kite beach score: 6.84.
        assert!((6.3..=7.4).contains(&score), "image score {score}");
    }
}

/// A head written for a different encoder loads
/// fine: the dimension mismatch is a request-time
/// soft failure, not a load error.
/// CRABCROP_TEST_AESTHETIC_HEAD_L14=...
/// cargo test -- --ignored aesthetic_head_for_another_encoder
#[test]
#[ignore = "needs a head file for another encoder (env-configured)"]
fn aesthetic_head_for_another_encoder_loads_and_scores_none() {
    let Some(head_path) = env_or_skip("CRABCROP_TEST_AESTHETIC_HEAD_L14") else {
        return;
    };
    let head = AestheticHead::load(&head_path).unwrap();
    assert_eq!(head.expected_dim(), 768);
    // A 512-dim embedding against the 768 head:
    // no score, and no panic.
    assert_eq!(head.score(&vec![0.03; 512]), None);
    assert!(head.score(&vec![0.03; 768]).is_some());
}
