use super::*;
use crate::ai::test_support::{expect_load, missing_path, solid, temp_file};

/// Real-model verification, run manually with the deployment
/// artifacts in place:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_UPSCALE_MODEL=...
/// cargo test -- --ignored upscale_with_the_real_model
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn upscale_with_the_real_model() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_UPSCALE_MODEL").unwrap();
    let provider = OnnxUpscaleProvider::load(Some(&dylib), &model).unwrap();
    assert!(provider.factor() >= 2, "factor {}", provider.factor());
    let out = provider.upscale(&solid(120, 30, 200, 24, 18)).unwrap();
    assert_eq!(out.width(), 24 * provider.factor());
    assert_eq!(out.height(), 18 * provider.factor());
}

#[test]
fn upscale_padded_dims_round_up_to_a_multiple_of_eight() {
    // Issue #326: inputs are padded to a multiple of 8
    // before inference; sizes already on the grid pass
    // through unchanged.
    assert_eq!(upscale_padded_dims(101, 53), (104, 56));
    assert_eq!(upscale_padded_dims(202, 106), (208, 112));
    assert_eq!(upscale_padded_dims(16, 16), (16, 16));
    assert_eq!(upscale_padded_dims(8, 8), (8, 8));
    assert_eq!(upscale_padded_dims(1, 1), (8, 8));
    assert_eq!(upscale_padded_dims(9, 7), (16, 8));
}

/// Real-model verification for issue #326, run manually
/// with the deployment artifacts in place: odd dimensions
/// used to fail inside the model. The provider pads to a
/// multiple of 8 and crops back, so a 101x53 input yields
/// exactly 202x106 at the model's 2x factor, and a second
/// pass (the engine's route to 4x) exactly 404x212.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_UPSCALE_MODEL=...
/// cargo test -- --ignored upscale_odd_dimensions_with_the_real_model
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn upscale_odd_dimensions_with_the_real_model() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_UPSCALE_MODEL").unwrap();
    let provider = OnnxUpscaleProvider::load(Some(&dylib), &model).unwrap();
    let f = provider.factor();
    let out = provider.upscale(&solid(120, 30, 200, 101, 53)).unwrap();
    assert_eq!((out.width(), out.height()), (101 * f, 53 * f));
    let out = provider.upscale(&out).unwrap();
    assert_eq!((out.width(), out.height()), (101 * f * f, 53 * f * f));
}

/// The upscale provider's missing-file failure is Load,
/// before its probe ever runs.
/// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored upscale_load_missing
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn upscale_load_missing_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let message = expect_load(OnnxUpscaleProvider::load(
        Some(&dylib),
        &missing_path("upscale"),
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored upscale_load_garbage
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn upscale_load_garbage_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let garbage = temp_file("upscale-garbage", b"this is not an onnx model at all");
    let message = expect_load(OnnxUpscaleProvider::load(Some(&dylib), &garbage));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// A face model cannot run the load probe (its fixed
/// 640x640 input rejects the 16x16 probe): the probe is
/// part of loading, so the failure is Load, not a
/// request-time Inference.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_FACE_MODEL=...
/// cargo test -- --ignored upscale_load_face_model
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn upscale_load_face_model_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_FACE_MODEL").unwrap();
    expect_load(OnnxUpscaleProvider::load(Some(&dylib), &model));
}

/// Same probe classification for a detection model.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_DETECT_MODEL=...
/// cargo test -- --ignored upscale_load_detection_model
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn upscale_load_detection_model_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_DETECT_MODEL").unwrap();
    expect_load(OnnxUpscaleProvider::load(Some(&dylib), &model));
}

/// A model that runs the probe but does not enlarge
/// (an identity export) is rejected by the factor
/// check, as Load with the reason spelled out.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_IDENTITY_MODEL=...
/// cargo test -- --ignored upscale_load_non_upscaling
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn upscale_load_non_upscaling_model_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_IDENTITY_MODEL").unwrap();
    let message = expect_load(OnnxUpscaleProvider::load(Some(&dylib), &model));
    assert!(
        message.contains("not an integer-factor super-resolution model"),
        "{message}"
    );
}
