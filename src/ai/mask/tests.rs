use super::*;
use crate::ai::test_support::{expect_load, missing_path, solid, temp_file};

#[test]
fn apply_mask_scales_alpha_by_the_mask() {
    // Left half background (mask 0), right half subject (255).
    let mask = GrayImage::from_fn(4, 2, |x, _| image::Luma([if x < 2 { 0 } else { 255 }]));
    let out = apply_mask(&solid(200, 30, 30, 4, 2), &mask).to_rgba8();
    assert_eq!(out.get_pixel(0, 0)[3], 0);
    assert_eq!(out.get_pixel(3, 0)[3], 255);
    // Color is untouched.
    assert_eq!(out.get_pixel(3, 0)[0], 200);
}

#[test]
fn apply_mask_respects_existing_alpha() {
    let mut img = DynamicImage::new_rgba8(2, 1);
    img.as_mut_rgba8()
        .unwrap()
        .put_pixel(0, 0, image::Rgba([10, 20, 30, 128]));
    let mask = GrayImage::from_pixel(2, 1, image::Luma([128]));
    let out = apply_mask(&img, &mask).to_rgba8();
    // 128 * 128 / 255 = 64.
    assert_eq!(out.get_pixel(0, 0)[3], 64);
}

#[test]
fn preprocess_produces_a_normalized_chw_tensor() {
    let input = preprocess(&solid(255, 255, 255, 8, 8));
    assert_eq!(input.len(), 3 * 320 * 320);
    // White scales to 1.0 by the max, then normalizes:
    // (1 - 0.485) / 0.229 = 2.2489... in the red channel.
    assert!((input[0] - 2.2489).abs() < 0.001, "got {}", input[0]);
    // A black image must not divide by zero; its red channel
    // is (0 - 0.485) / 0.229 = -2.1179...
    let input = preprocess(&solid(0, 0, 0, 8, 8));
    assert!((input[0] - -2.1179).abs() < 0.001, "got {}", input[0]);
}

/// Real-model verification, run manually with the deployment
/// artifacts in place:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
/// cargo test -- --ignored removebg_with_the_real_model
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn removebg_with_the_real_model() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_REMOVEBG_MODEL").unwrap();
    let provider = OnnxMaskProvider::load(Some(&dylib), &model).unwrap();
    // A dark square on a light background: the model should
    // call the square the subject.
    let mut img = DynamicImage::new_rgb8(200, 200);
    for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        *p = if (60..140).contains(&x) && (60..140).contains(&y) {
            image::Rgb([30, 30, 30])
        } else {
            image::Rgb([240, 240, 240])
        };
    }
    let mask = provider.mask(&img).unwrap();
    assert_eq!((mask.width(), mask.height()), (200, 200));
    assert!(
        mask.get_pixel(100, 100)[0] > 200,
        "center (subject) mask = {}",
        mask.get_pixel(100, 100)[0]
    );
    assert!(
        mask.get_pixel(5, 5)[0] < 55,
        "corner (background) mask = {}",
        mask.get_pixel(5, 5)[0]
    );
}

/// The mask provider shares the session builder with
/// the face provider; its missing-file failure is the
/// same Load.
/// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored mask_load_missing
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn mask_load_missing_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let message = expect_load(OnnxMaskProvider::load(Some(&dylib), &missing_path("mask")));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored mask_load_garbage
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn mask_load_garbage_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let garbage = temp_file("mask-garbage", b"this is not an onnx model at all");
    let message = expect_load(OnnxMaskProvider::load(Some(&dylib), &garbage));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_NO_INPUT_MODEL=...
/// cargo test -- --ignored mask_load_model_without_inputs
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn mask_load_model_without_inputs_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_NO_INPUT_MODEL").unwrap();
    let message = expect_load(OnnxMaskProvider::load(Some(&dylib), &model));
    assert!(message.contains("has no inputs"), "{message}");
}

#[test]
fn postprocess_maps_min_and_max_to_black_and_white() {
    // A 320x320 mask field: -1 at the origin, 3 at its
    // neighbour, 0 elsewhere. The range maps onto
    // 0..255, so the interior value 0 lands at
    // (0 - -1) / 4 * 255 = 63.75, rounding to 64.
    let mut data = vec![0f32; 320 * 320];
    data[0] = -1.0;
    data[1] = 3.0;
    let mask = postprocess(&data, 320, 320).unwrap();
    assert_eq!((mask.width(), mask.height()), (320, 320));
    assert_eq!(mask.get_pixel(0, 0)[0], 0);
    assert_eq!(mask.get_pixel(1, 0)[0], 255);
    assert_eq!(mask.get_pixel(2, 0)[0], 64);
}

#[test]
fn postprocess_flat_mask_normalizes_to_black_not_nan_or_gray() {
    // A constant field has no range: the epsilon floor
    // makes every (v - min) / range exactly 0, so the
    // whole mask is black.
    let data = vec![2.5f32; 320 * 320];
    let mask = postprocess(&data, 320, 320).unwrap();
    assert!(mask.pixels().all(|p| p[0] == 0));
}

#[test]
fn postprocess_resizes_the_mask_back_to_the_source_size() {
    // Top half at the range minimum, bottom half at the
    // maximum: after the resize to 160x160 the extremes
    // still read as pure black and white.
    let mut data = vec![0f32; 320 * 320];
    for v in &mut data[320 * 160..] {
        *v = 1.0;
    }
    let mask = postprocess(&data, 160, 160).unwrap();
    assert_eq!((mask.width(), mask.height()), (160, 160));
    assert_eq!(mask.get_pixel(0, 0)[0], 0);
    assert_eq!(mask.get_pixel(0, 159)[0], 255);
}

#[test]
fn postprocess_rejects_an_output_shorter_than_the_field() {
    // An export whose first output is smaller than the
    // 320x320 field (a 160x160 output, say) is a
    // BadOutput, not an indexing panic under the session
    // lock (audit R2-28).
    let data = vec![0.5f32; 160 * 160];
    let err = postprocess(&data, 320, 320).unwrap_err();
    assert!(matches!(err, AiError::BadOutput(_)), "{err}");
    assert!(err.to_string().contains("320x320"), "{err}");
}
