use super::*;
use crate::ai::test_support::{
    OPAQUE16, expect_load, layouts, missing_path, noise_rgb, rgba_halves, snapshot, solid,
    temp_file,
};
use image::Pixel;

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

/// A fake 2x super-resolution model standing in for the
/// session: a nearest-neighbor enlargement of the planar
/// input tensor, the same graph as a one-node ONNX `Resize`
/// (mode nearest, scales 1, 1, 2, 2). It answers in the
/// model's 0..1 range.
fn fake_sr_x2(input: Vec<f32>, dims: [usize; 4]) -> Result<(Vec<i64>, Vec<f32>), AiError> {
    let [n, c, h, w] = dims;
    assert_eq!(
        input.len(),
        n * c * h * w,
        "the input tensor matches its dims"
    );
    let (oh, ow) = (h * 2, w * 2);
    let mut out = vec![0f32; c * oh * ow];
    for ch in 0..c {
        for y in 0..oh {
            for x in 0..ow {
                out[(ch * oh + y) * ow + x] = input[(ch * h + y / 2) * w + x / 2];
            }
        }
    }
    Ok((vec![1, c as i64, oh as i64, ow as i64], out))
}

/// Issue #551: an RGBA source at `depth` comes back from a 2x
/// upscale with its transparent half still transparent, its
/// opaque half opaque in the source's own color, and the
/// source's layout and depth. 20x12 is off the 8-pixel grid,
/// so the pad-and-crop path (issue #326) runs too.
fn assert_upscale_keeps_alpha_and_depth(depth: SampleDepth) {
    let src = rgba_halves(depth, 20, 12);
    let out = upscale_with(&src, fake_sr_x2).unwrap();
    assert_upscaled_halves_keep_alpha_and_depth(&src, &out, 2);
    let (src16, out16) = (src.to_rgba16(), out.to_rgba16());
    for (x, y, px) in out16.enumerate_pixels() {
        if x >= 22 {
            assert_eq!(
                *px,
                *src16.get_pixel(x / 2, y / 2),
                "the opaque half keeps its color at ({x}, {y})"
            );
        }
    }
}

/// What any model's upscale of an [`rgba_halves`] source
/// must keep, whatever color the model paints: the factor
/// times the source size, the source's layout and depth,
/// and each half's alpha. The alpha plane is enlarged with
/// a Triangle resize, which blends the output columns
/// within one source pixel of the seam; every other column
/// keeps its half's alpha exactly.
fn assert_upscaled_halves_keep_alpha_and_depth(src: &DynamicImage, out: &DynamicImage, f: u32) {
    assert_eq!(
        (out.width(), out.height()),
        (src.width() * f, src.height() * f)
    );
    assert_eq!(
        out.color(),
        src.color(),
        "the result keeps the source's layout and sample depth"
    );
    let seam = src.width() / 2 * f;
    for (x, y, px) in out.to_rgba16().enumerate_pixels() {
        if x + f < seam {
            assert_eq!(px[3], 0, "transparent stays transparent at ({x}, {y})");
        }
        if x >= seam + f {
            assert_eq!(px[3], u16::MAX, "opaque stays opaque at ({x}, {y})");
        }
    }
}

#[test]
fn upscale_keeps_alpha_for_an_8_bit_rgba_source() {
    assert_upscale_keeps_alpha_and_depth(SampleDepth::Eight);
}

#[test]
fn upscale_keeps_alpha_and_16_bits_for_a_16_bit_rgba_source() {
    assert_upscale_keeps_alpha_and_depth(SampleDepth::Sixteen);
    // Samples off the 8-bit grid survive exactly.
    let out = upscale_with(&rgba_halves(SampleDepth::Sixteen, 20, 12), fake_sr_x2).unwrap();
    let px = out.as_rgba16().unwrap().get_pixel(30, 6);
    assert_eq!(px.0, [OPAQUE16[0], OPAQUE16[1], OPAQUE16[2], u16::MAX]);
}

#[test]
fn upscale_keeps_alpha_and_floats_for_a_float_rgba_source() {
    assert_upscale_keeps_alpha_and_depth(SampleDepth::Float);
    let src = rgba_halves(SampleDepth::Float, 20, 12);
    let out = upscale_with(&src, fake_sr_x2).unwrap();
    let want = *src.as_rgba32f().unwrap().get_pixel(15, 3);
    assert_eq!(*out.as_rgba32f().unwrap().get_pixel(30, 6), want);
}

/// A source without alpha gets an RGB result at its own
/// depth: carrying alpha is decided by the source's color
/// type, not assumed.
#[test]
fn upscale_of_an_opaque_16_bit_source_is_16_bit_rgb() {
    let src = DynamicImage::ImageRgb16(rgba_halves(SampleDepth::Sixteen, 20, 12).to_rgb16());
    let out = upscale_with(&src, fake_sr_x2).unwrap();
    assert_eq!(out.color(), image::ColorType::Rgb16);
    assert_eq!(
        out.as_rgb16().unwrap().get_pixel(30, 6).0,
        OPAQUE16,
        "16-bit samples survive exactly"
    );
}

/// The 8-bit path is the arithmetic the provider always had:
/// an opaque 8-bit source comes back as 8-bit RGB with the
/// model's values rounded to 8 bits, out-of-range values
/// clamped and a NaN read as 0.
#[test]
fn upscale_of_an_8_bit_source_rounds_the_model_output_to_8_bits() {
    let src = solid(10, 20, 30, 8, 8);
    let out = upscale_with(&src, |_input, [_, c, h, w]| {
        let plane = 4 * h * w;
        let mut data = vec![0.5f32; c * plane];
        data[0] = 1.7;
        data[plane] = -0.3;
        data[2 * plane] = f32::NAN;
        Ok((vec![1, c as i64, 2 * h as i64, 2 * w as i64], data))
    })
    .unwrap();
    assert_eq!(out.color(), image::ColorType::Rgb8);
    let rgb = out.as_rgb8().unwrap();
    assert_eq!(rgb.get_pixel(0, 0).0, [255, 0, 0]);
    assert_eq!(rgb.get_pixel(1, 0).0, [128, 128, 128]);
}

/// A model output that is short of the values its shape
/// claims is refused before anything indexes by it (issue
/// #267), at every depth.
#[test]
fn upscale_refuses_a_short_model_output() {
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen, SampleDepth::Float] {
        let err = upscale_with(&rgba_halves(depth, 8, 8), |_input, _dims| {
            Ok((vec![1, 3, 16, 16], vec![0.0f32; 10]))
        })
        .unwrap_err();
        assert!(matches!(err, AiError::BadOutput(_)), "{err:?}");
    }
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

/// Issue #551 through the real session: a 16-bit or float
/// RGBA source keeps its alpha and its depth. Only alpha,
/// layout and size are checked, never color, so any
/// super-resolution model will do (the authored CI fixture
/// has random weights).
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_UPSCALE_MODEL=...
/// cargo test -- --ignored upscale_keeps_alpha_and_depth_with_the_real_model
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn upscale_keeps_alpha_and_depth_with_the_real_model() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_UPSCALE_MODEL").unwrap();
    let provider = OnnxUpscaleProvider::load(Some(&dylib), &model).unwrap();
    for depth in [SampleDepth::Sixteen, SampleDepth::Float] {
        let src = rgba_halves(depth, 20, 12);
        let out = provider.upscale(&src).unwrap();
        assert_upscaled_halves_keep_alpha_and_depth(&src, &out, provider.factor());
    }
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
    let (_model, garbage) = temp_file("upscale-garbage", b"this is not an onnx model at all");
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

/// `upscale_at` as it was before issue #675: the input
/// filled through an index computation with two clamps for
/// every sample, and the output converted one sample at a
/// time through `from_unit`, which is passed in because the
/// 8-bit one rounded with `round` then.
fn upscale_by_samples<S: Sample, O: TensorOutput>(
    img: &DynamicImage,
    run: impl FnOnce(Vec<f32>, [usize; 4]) -> Result<O, AiError>,
    from_unit: impl Fn(f32) -> S,
) -> Result<DynamicImage, AiError> {
    let (w, h) = (img.width(), img.height());
    let (pw, ph) = upscale_padded_dims(w, h);
    let rgb = S::rgb(img);
    let raw = rgb.as_raw();
    let in_plane = pw as usize * ph as usize;
    let mut input = vec![0f32; 3 * in_plane];
    for y in 0..ph as usize {
        let sy = y.min(h.saturating_sub(1) as usize);
        for x in 0..pw as usize {
            let sx = x.min(w.saturating_sub(1) as usize);
            let src = (sy * w as usize + sx) * 3;
            let dst = y * pw as usize + x;
            for c in 0..3 {
                input[c * in_plane + dst] = raw[src + c].to_unit();
            }
        }
    }
    let output = run(input, [1, 3, ph as usize, pw as usize])?;
    let (shape, data) = output.f32_tensor()?;
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
        return Err(AiError::BadOutput(format!(
            "unexpected model output shape: {shape:?}"
        )));
    }
    let (ow, oh) = (shape[3] as u32, shape[2] as u32);
    let plane = (ow as usize)
        .checked_mul(oh as usize)
        .ok_or_else(|| AiError::BadOutput(format!("model output shape overflows: {shape:?}")))?;
    if data.len() < plane * 3 {
        return Err(AiError::BadOutput(format!(
            "model output has {} values, short of the {} its shape claims",
            data.len(),
            plane * 3
        )));
    }
    let mut out = ImageBuffer::<S::Rgb, Vec<S>>::new(ow, oh);
    for (i, px) in out.pixels_mut().enumerate() {
        for (c, v) in px.channels_mut().iter_mut().enumerate() {
            *v = from_unit(data[c * plane + i]);
        }
    }
    if (pw, ph) != (w, h) {
        let f = if pw > 0 && ph > 0 && ow % pw == 0 && oh % ph == 0 && ow / pw == oh / ph {
            ow / pw
        } else {
            0
        };
        let (Some(crop_w), Some(crop_h)) = (w.checked_mul(f), h.checked_mul(f)) else {
            return Err(AiError::BadOutput(format!(
                "unexpected model output shape for a padded input: {shape:?}"
            )));
        };
        if f < 1 || crop_w > ow || crop_h > oh {
            return Err(AiError::BadOutput(format!(
                "unexpected model output shape for a padded input: {shape:?}"
            )));
        }
        out = image::imageops::crop_imm(&out, 0, 0, crop_w, crop_h).to_image();
    }
    Ok(with_source_alpha(img, out))
}

/// [`upscale_by_samples`] at the image's depth, as
/// `upscale_with` dispatches.
fn upscale_with_by_samples<O: TensorOutput>(
    img: &DynamicImage,
    run: impl FnOnce(Vec<f32>, [usize; 4]) -> Result<O, AiError>,
) -> Result<DynamicImage, AiError> {
    match SampleDepth::of(img) {
        SampleDepth::Eight => {
            upscale_by_samples::<u8, O>(img, run, |v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        }
        SampleDepth::Sixteen => upscale_by_samples::<u16, O>(img, run, u16::from_unit),
        SampleDepth::Float => upscale_by_samples::<f32, O>(img, run, f32::from_unit),
    }
}

/// A fake 2x model whose output depends on its whole input
/// and leaves the 0..1 range: the nearest-neighbor
/// enlargement, stretched and shifted per channel, with a
/// NaN, an infinity and exact half levels planted.
fn fake_sr_x2_wild(input: Vec<f32>, dims: [usize; 4]) -> Result<(Vec<i64>, Vec<f32>), AiError> {
    let (shape, mut out) = fake_sr_x2(input, dims)?;
    let plane = out.len() / 3;
    for (i, v) in out.iter_mut().enumerate() {
        let c = i / plane.max(1);
        *v = *v * (1.3 - 0.2 * c as f32) - 0.1 + ((i % 7) as f32) * 0.011;
        match i % 211 {
            0 => *v = f32::NAN,
            1 => *v = f32::INFINITY,
            2 => *v = f32::NEG_INFINITY,
            3 => *v = ((i % 255) as f32 + 0.5) / 255.0,
            _ => {}
        }
    }
    Ok((shape, out))
}

#[test]
fn upscale_matches_the_per_sample_form() {
    // Issue #675: the input is filled a row at a time and
    // the output converted a block of a plane at a time.
    // The result is the per-sample form's, for every layout
    // and depth, at sizes on the 8-pixel grid and off it
    // (padded, then cropped), down to one pixel.
    for (w, h) in [(16, 16), (8, 24), (13, 7), (20, 12), (1, 1), (9, 300)] {
        let rgb = noise_rgb(w, h, w * 19 + h);
        for (layout, img) in layouts(&rgb) {
            for model in [fake_sr_x2, fake_sr_x2_wild] {
                let got = upscale_with(&img, model).unwrap();
                let want = upscale_with_by_samples(&img, model).unwrap();
                assert_eq!(snapshot(&got), snapshot(&want), "{layout} {w}x{h}");
            }
        }
    }
}

#[test]
fn the_model_input_is_the_per_sample_one_bit_for_bit() {
    // The tensor the model is handed, padding included, for
    // every depth: the fake model hands its input back.
    for (w, h) in [(16, 8), (13, 7), (1, 1), (3, 21)] {
        let rgb = noise_rgb(w, h, w * 23 + h);
        for (layout, img) in layouts(&rgb) {
            let seen = std::cell::RefCell::new((Vec::new(), [0usize; 4]));
            let record = |input: Vec<f32>, dims: [usize; 4]| {
                *seen.borrow_mut() = (input.clone(), dims);
                fake_sr_x2(input, dims)
            };
            upscale_with(&img, record).unwrap();
            let got = seen.replace((Vec::new(), [0; 4]));
            upscale_with_by_samples(&img, record).unwrap();
            let want = seen.replace((Vec::new(), [0; 4]));
            assert_eq!(got.1, want.1, "{layout} {w}x{h}");
            assert_eq!(got.0.len(), want.0.len(), "{layout} {w}x{h}");
            assert!(
                got.0
                    .iter()
                    .zip(&want.0)
                    .all(|(g, w)| g.to_bits() == w.to_bits()),
                "{layout} {w}x{h}"
            );
        }
    }
}
