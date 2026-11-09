use super::*;
use crate::ai::env_or_skip;
use crate::ai::test_support::{OPAQUE16, expect_load, missing_path, rgba_halves, solid};

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
    let filled = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        window.2,
        window.3,
        image::Rgb([30, 30, 200]),
    ));
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
    let filled = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        window.2,
        window.3,
        image::Rgb([99, 99, 99]),
    ));
    let out = composite_inpaint(&img, &mask, window, &filled).to_rgba8();
    assert_eq!(*out.get_pixel(10, 10), image::Rgba([99, 99, 99, 200]));
    assert_eq!(*out.get_pixel(0, 0), image::Rgba([10, 20, 30, 200]));
}

/// A fake inpainting model standing in for the session: it
/// answers with its own input image, scaled to the export's
/// 0..255 output range, so the fill is the crop window's own
/// color carried through the model's 512x512 working size and
/// back.
fn fake_lama(image: Vec<f32>, mask: Vec<f32>) -> Result<(Vec<i64>, Vec<f32>), AiError> {
    let size = INPAINT_INPUT_SIZE as usize;
    assert_eq!(image.len(), 3 * size * size, "the image input is 3x512x512");
    assert_eq!(mask.len(), size * size, "the mask input is 512x512");
    assert!(mask.iter().any(|&m| m > 0.0), "the hole reaches the model");
    let out = image.into_iter().map(|v| v * 255.0).collect();
    Ok((vec![1, 3, size as i64, size as i64], out))
}

/// Issue #551: an RGBA source at `depth` comes back from a
/// removal with its alpha unchanged, sample for sample (inside
/// the hole too: inpainting fills color, not transparency),
/// every pixel outside the hole bit for bit, the fill at the
/// source's depth, and the source's layout. The hole straddles
/// the seam between the transparent and opaque halves.
fn assert_remove_region_keeps_alpha_and_depth(depth: SampleDepth) {
    let src = rgba_halves(depth, 40, 24);
    let mask = region_mask(40, 24, 14, 8, 12, 8);
    let out = inpaint_with(&src, &mask, fake_lama).unwrap();
    assert_removal_keeps_alpha_and_depth(&src, &mask, &out);
    // Inside the hole, away from the seam, the identity
    // model's fill is the source's own color at the
    // source's depth.
    let (src16, out16) = (src.to_rgba16(), out.to_rgba16());
    assert_eq!(out16.get_pixel(25, 12), src16.get_pixel(25, 12));
}

/// What any model's removal must keep, whatever color it
/// fills the hole with: the source's size, layout and
/// depth, its alpha sample for sample (inside the hole
/// too: inpainting fills color, not transparency), and
/// every pixel outside the hole bit for bit.
fn assert_removal_keeps_alpha_and_depth(src: &DynamicImage, mask: &GrayImage, out: &DynamicImage) {
    assert_eq!((out.width(), out.height()), (src.width(), src.height()));
    assert_eq!(
        out.color(),
        src.color(),
        "the result keeps the source's layout and sample depth"
    );
    let (src16, out16) = (src.to_rgba16(), out.to_rgba16());
    for (x, y, px) in out16.enumerate_pixels() {
        let s = src16.get_pixel(x, y);
        assert_eq!(px[3], s[3], "alpha is the source's at ({x}, {y})");
        if mask.get_pixel(x, y)[0] == 0 {
            assert_eq!(
                px, s,
                "outside the hole, the source bit for bit at ({x}, {y})"
            );
        }
    }
}

#[test]
fn remove_region_keeps_alpha_for_an_8_bit_rgba_source() {
    assert_remove_region_keeps_alpha_and_depth(SampleDepth::Eight);
}

#[test]
fn remove_region_keeps_alpha_and_16_bits_for_a_16_bit_rgba_source() {
    assert_remove_region_keeps_alpha_and_depth(SampleDepth::Sixteen);
    let src = rgba_halves(SampleDepth::Sixteen, 40, 24);
    let out = inpaint_with(&src, &region_mask(40, 24, 14, 8, 12, 8), fake_lama).unwrap();
    // Samples off the 8-bit grid survive exactly, in the
    // fill as well as around it.
    let out = out.as_rgba16().unwrap();
    for (x, y) in [(25, 12), (35, 2)] {
        let px = out.get_pixel(x, y);
        assert_eq!(px.0, [OPAQUE16[0], OPAQUE16[1], OPAQUE16[2], u16::MAX]);
    }
}

#[test]
fn remove_region_keeps_alpha_and_floats_for_a_float_rgba_source() {
    assert_remove_region_keeps_alpha_and_depth(SampleDepth::Float);
    let src = rgba_halves(SampleDepth::Float, 40, 24);
    let out = inpaint_with(&src, &region_mask(40, 24, 14, 8, 12, 8), fake_lama).unwrap();
    let (src, out) = (src.as_rgba32f().unwrap(), out.as_rgba32f().unwrap());
    // Outside the hole the floats are the source's own.
    assert_eq!(out.get_pixel(35, 2), src.get_pixel(35, 2));
    // The fill went through two float resizes and the
    // model's 0..255 range: equal to well under a 16-bit
    // step.
    let (fill, want) = (out.get_pixel(25, 12), src.get_pixel(25, 12));
    for c in 0..3 {
        assert!(
            (fill[c] - want[c]).abs() < 1e-5,
            "{fill:?} against {want:?}"
        );
    }
    assert_eq!(fill[3], want[3]);
}

/// The 8-bit path is the arithmetic the provider always had:
/// the export's 0..255 values round to 8 bits, clamped, a NaN
/// read as 0, and an opaque source stays RGB.
#[test]
fn remove_region_of_an_8_bit_source_rounds_the_fill_to_8_bits() {
    let src = solid(200, 30, 30, 16, 16);
    let mask = region_mask(16, 16, 4, 4, 8, 8);
    let size = INPAINT_INPUT_SIZE as usize;
    let out = inpaint_with(&src, &mask, |_image, _mask| {
        let plane = size * size;
        let mut data = vec![0f32; 3 * plane];
        data[..plane].fill(300.0);
        data[plane..2 * plane].fill(f32::NAN);
        data[2 * plane..].fill(99.5);
        Ok((vec![1, 3, size as i64, size as i64], data))
    })
    .unwrap();
    assert_eq!(out.color(), image::ColorType::Rgb8);
    let rgb = out.as_rgb8().unwrap();
    assert_eq!(rgb.get_pixel(8, 8).0, [255, 0, 100]);
    assert_eq!(rgb.get_pixel(0, 0).0, [200, 30, 30]);
}

/// A removal over an empty mask runs no model at all and
/// returns the source as it is, at any depth.
#[test]
fn remove_region_with_an_empty_mask_runs_no_model() {
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen, SampleDepth::Float] {
        let src = rgba_halves(depth, 20, 10);
        let out = inpaint_with(
            &src,
            &GrayImage::new(20, 10),
            |_, _| -> Result<(Vec<i64>, Vec<f32>), AiError> {
                panic!("the model must not run for an empty mask")
            },
        )
        .unwrap();
        assert_eq!(out, src);
    }
}

/// An 8-bit fill composited over a 16-bit source is written
/// at the source's depth: the composite converts the fill,
/// never the source.
#[test]
fn composite_inpaint_writes_at_the_source_depth() {
    let src = rgba_halves(SampleDepth::Sixteen, 20, 20);
    let mask = region_mask(20, 20, 12, 5, 6, 6);
    let window = inpaint_window(20, 20, (12, 5, 6, 6));
    let filled = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        window.2,
        window.3,
        image::Rgb([0x80, 0x40, 0x20]),
    ));
    let out = composite_inpaint(&src, &mask, window, &filled);
    assert_eq!(out.color(), image::ColorType::Rgba16);
    let out = out.as_rgba16().unwrap();
    assert_eq!(out.get_pixel(14, 7).0, [0x8080, 0x4040, 0x2020, u16::MAX]);
    assert_eq!(
        out.get_pixel(19, 19).0,
        [OPAQUE16[0], OPAQUE16[1], OPAQUE16[2], u16::MAX]
    );
    assert_eq!(out.get_pixel(0, 0).0, [u16::MAX, 0, 0, 0]);
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

/// Issue #551 through the real session: a removal on a
/// 16-bit or float RGBA source keeps its alpha and its
/// depth. The fill's color is not checked, so any
/// inpainting model will do.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_INPAINT_MODEL=...
/// cargo test -- --ignored inpaint_keeps_alpha_and_depth_with_the_real_model
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn inpaint_keeps_alpha_and_depth_with_the_real_model() {
    let Some(dylib) = env_or_skip("CRABCROP_TEST_ORT_DYLIB") else {
        return;
    };
    let Some(model) = env_or_skip("CRABCROP_TEST_INPAINT_MODEL") else {
        return;
    };
    let provider = OnnxInpaintProvider::load(Some(&dylib), &model).unwrap();
    for depth in [SampleDepth::Sixteen, SampleDepth::Float] {
        let src = rgba_halves(depth, 40, 24);
        let mask = region_mask(40, 24, 14, 8, 12, 8);
        let out = provider.inpaint(&src, &mask).unwrap();
        assert_removal_keeps_alpha_and_depth(&src, &mask, &out);
    }
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
