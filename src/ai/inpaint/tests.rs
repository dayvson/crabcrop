use super::*;
use crate::ai::env_or_skip;
use crate::ai::test_support::{expect_load, missing_path, solid};

#[test]
fn region_mask_marks_only_the_rectangle() {
    let mask = region_mask(10, 8, 2, 3, 4, 2);
    assert_eq!(mask.get_pixel(2, 3)[0], 255);
    assert_eq!(mask.get_pixel(5, 4)[0], 255);
    assert_eq!(mask.get_pixel(1, 3)[0], 0);
    assert_eq!(mask.get_pixel(6, 3)[0], 0);
    assert_eq!(mask.get_pixel(2, 5)[0], 0);
    // A rectangle hanging off the edge clamps instead of
    // panicking: the engine bounds-checks first, but the
    // mask builder does not trust it to.
    let mask = region_mask(10, 8, 8, 6, 40, 40);
    assert_eq!(mask.get_pixel(9, 7)[0], 255);
}

#[test]
fn mask_bbox_finds_the_marked_box() {
    assert_eq!(mask_bbox(&GrayImage::new(10, 10)), None);
    let mask = region_mask(40, 30, 7, 9, 5, 3);
    assert_eq!(mask_bbox(&mask), Some((7, 9, 5, 3)));
}

#[test]
fn inpaint_window_centers_on_the_box_with_context() {
    // Twice the box's larger axis of context, centered.
    assert_eq!(inpaint_window(100, 80, (40, 30, 20, 20)), (30, 20, 40, 40));
    // A corner box: the window clamps to the edges.
    assert_eq!(inpaint_window(100, 80, (0, 0, 10, 10)), (0, 0, 20, 20));
    // A box against the far edges: the window slides
    // back inside.
    assert_eq!(inpaint_window(100, 80, (90, 70, 10, 10)), (80, 60, 20, 20));
    // A box covering the whole image: the window is the
    // image.
    assert_eq!(inpaint_window(100, 80, (0, 0, 100, 80)), (0, 0, 100, 80));
    // An image smaller than the square on one axis: the
    // window takes that whole axis.
    assert_eq!(inpaint_window(30, 20, (10, 5, 10, 10)), (5, 0, 20, 20));
}

#[test]
fn composite_inpaint_takes_only_the_hole_from_the_fill() {
    let img = solid(200, 30, 30, 40, 30);
    let mask = region_mask(40, 30, 10, 10, 10, 10);
    let window = inpaint_window(40, 30, (10, 10, 10, 10));
    let filled = image::RgbImage::from_pixel(window.2, window.3, image::Rgb([30, 30, 200]));
    let out = composite_inpaint(&img, &mask, window, &filled).to_rgb8();
    // Inside the hole: the fill.
    assert_eq!(*out.get_pixel(15, 15), image::Rgb([30, 30, 200]));
    // Inside the window but outside the hole: the
    // original, bit for bit (the model regenerates the
    // whole window; only the hole may be believed).
    assert_eq!(*out.get_pixel(9, 9), image::Rgb([200, 30, 30]));
    // Outside the window: untouched.
    assert_eq!(*out.get_pixel(35, 25), image::Rgb([200, 30, 30]));
}

#[test]
fn composite_inpaint_keeps_the_source_alpha() {
    let mut rgba = image::RgbaImage::new(20, 20);
    for p in rgba.pixels_mut() {
        *p = image::Rgba([10, 20, 30, 200]);
    }
    let img = DynamicImage::ImageRgba8(rgba);
    let mask = region_mask(20, 20, 5, 5, 10, 10);
    let window = inpaint_window(20, 20, (5, 5, 10, 10));
    let filled = image::RgbImage::from_pixel(window.2, window.3, image::Rgb([99, 99, 99]));
    let out = composite_inpaint(&img, &mask, window, &filled).to_rgba8();
    assert_eq!(*out.get_pixel(10, 10), image::Rgba([99, 99, 99, 200]));
    assert_eq!(*out.get_pixel(0, 0), image::Rgba([10, 20, 30, 200]));
}

/// Real-model verification, run manually with the deployment
/// artifacts in place:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_INPAINT_MODEL=...
/// cargo test -- --ignored inpaint_with_the_real_model
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn inpaint_with_the_real_model() {
    let Some(dylib) = env_or_skip("CRABCROP_TEST_ORT_DYLIB") else {
        return;
    };
    let Some(model) = env_or_skip("CRABCROP_TEST_INPAINT_MODEL") else {
        return;
    };
    let provider = OnnxInpaintProvider::load(Some(&dylib), &model).unwrap();
    // A smooth gradient with a solid red square pasted
    // in the middle: the model's one plausible answer
    // is the gradient, continued.
    let mut rgb = image::RgbImage::new(256, 256);
    for (x, y, p) in rgb.enumerate_pixels_mut() {
        *p = if (88..168).contains(&x) && (88..168).contains(&y) {
            image::Rgb([255, 0, 0])
        } else {
            image::Rgb([x as u8, y as u8, 102])
        };
    }
    let img = DynamicImage::ImageRgb8(rgb);
    let mask = region_mask(256, 256, 88, 88, 80, 80);
    let out = provider.inpaint(&img, &mask).unwrap().to_rgb8();
    assert_eq!((out.width(), out.height()), (256, 256));
    let center = out.get_pixel(128, 128);
    // The fill on this geometry (the hole is half the
    // crop window, upscaled to the model's 512) measures
    // (97, 86, 79) against the true gradient's
    // (128, 128, 102): murkier than the ideal, but the
    // square is unambiguously gone and the fill is in
    // the gradient's family. The strict half of this
    // test is the bit-identity outside, below.
    assert!(
        (60..=180).contains(&center[0])
            && (60..=180).contains(&center[1])
            && (50..=150).contains(&center[2])
            && center[0] - center[1] < 45,
        "the red square should be gone, filled with the gradient: {center:?}"
    );
    // Outside the removal: the original, bit for bit.
    assert_eq!(*out.get_pixel(4, 4), image::Rgb([4, 4, 102]));
    assert_eq!(*out.get_pixel(250, 200), image::Rgb([250, 200, 102]));
    // An empty mask removes nothing and runs no fill.
    let out = provider
        .inpaint(&img, &GrayImage::new(256, 256))
        .unwrap()
        .to_rgb8();
    assert_eq!(*out.get_pixel(128, 128), image::Rgb([255, 0, 0]));
}

/// A one-input model is not an inpainting model: the
/// load rejects it as Load, at boot, before any request
/// can discover it. Ignored like the other load tests:
/// load() initializes the runtime first.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_IDENTITY_MODEL=...
/// cargo test -- --ignored inpaint_load_rejects
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn inpaint_load_rejects_a_single_input_model() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_IDENTITY_MODEL").unwrap();
    let message = expect_load(OnnxInpaintProvider::load(Some(&dylib), &model));
    assert!(message.contains("takes an image and a mask"), "{message}");
}

/// A missing inpainting model file fails as Load, the
/// same boot story as the other providers.
/// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored inpaint_load_missing
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn inpaint_load_missing_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let message = expect_load(OnnxInpaintProvider::load(
        Some(&dylib),
        &missing_path("inpaint"),
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}
