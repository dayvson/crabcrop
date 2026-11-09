use super::*;
use crate::ai::env_or_skip;
use crate::ai::test_support::{expect_load, missing_path};
use image::imageops::FilterType;

#[test]
fn segment_preprocess_pads_and_normalizes() {
    // 200x100 red: the longest side scales to 1024, so
    // the image occupies 1024x512 and the rest of the
    // square is normalized-space zero padding.
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        200,
        100,
        image::Rgb([255, 0, 0]),
    ));
    let input = segment_preprocess(&img);
    let plane = 1024 * 1024;
    assert_eq!(input.len(), 3 * plane);
    let at = |c: usize, x: usize, y: usize| input[c * plane + y * 1024 + x];
    // The SAM statistics, in 0..255 scale.
    assert!((at(0, 0, 0) - (255.0 - 123.675) / 58.395).abs() < 0.01);
    assert!((at(1, 0, 0) - (0.0 - 116.28) / 57.12).abs() < 0.01);
    assert!((at(2, 1023, 511) - (0.0 - 103.53) / 57.375).abs() < 0.01);
    // Below the image: padding, exactly zero.
    assert_eq!(at(0, 0, 512), 0.0);
    assert_eq!(at(1, 1023, 1023), 0.0);
}

#[test]
fn segment_box_coords_scale_into_encoder_space() {
    // 800x600 image: the scale is 1024/800. The box
    // corners land where the encoder sees them.
    let coords = segment_box_coords(800, 600, (260, 180, 300, 290));
    let want = [332.8, 230.4, 716.8, 601.6];
    for (got, want) in coords.iter().zip(want) {
        assert!((got - want).abs() < 0.01, "{got} vs {want}");
    }
    // A square image at the encoder's own size is the
    // identity.
    assert_eq!(
        segment_box_coords(1024, 1024, (10, 20, 30, 40)),
        [10.0, 20.0, 40.0, 60.0]
    );
}

#[test]
fn sigmoid_strengths_map_logits_to_a_soft_mask() {
    let strengths: Vec<u8> = [-100.0, 0.0, 100.0].map(sigmoid_strength).to_vec();
    assert_eq!(strengths, vec![0, 128, 255]);
}

#[test]
fn mask_geometry_measures_a_soft_mask() {
    // Left half at full strength, one dim pixel at
    // (7, 7): the tight box is the half, the dim pixel
    // still weighs into coverage and the centroid.
    let mask = GrayImage::from_fn(10, 10, |x, y| {
        if x < 5 {
            image::Luma([255])
        } else if (x, y) == (7, 7) {
            image::Luma([100])
        } else {
            image::Luma([0])
        }
    });
    let g = mask_geometry(&mask).expect("a geometry");
    assert_eq!(g.tight, (0, 0, 5, 10));
    assert!((g.coverage - 0.5039).abs() < 0.001, "{}", g.coverage);
    assert!((g.centroid.0 - 0.2039).abs() < 0.001, "{}", g.centroid.0);
    assert!((g.centroid.1 - 0.4519).abs() < 0.001, "{}", g.centroid.1);
    // Nothing over half strength: no object, no
    // geometry, even though the mask is not empty.
    let faint = GrayImage::from_pixel(10, 10, image::Luma([100]));
    assert_eq!(mask_geometry(&faint), None);
    assert_eq!(mask_geometry(&GrayImage::new(10, 10)), None);
    assert_eq!(mask_geometry(&GrayImage::new(0, 0)), None);
}

/// The real MobileSAM pair, end to end: a red disc on
/// a textured ground, prompted with the disc's box.
/// The mask must be the disc's silhouette: strong at
/// the center, absent at the box's corners, covering
/// about pi/4 of the box.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_SEGMENT_ENCODER_MODEL=...
/// CRABCROP_TEST_SEGMENT_DECODER_MODEL=...
/// cargo test -- --ignored segment_with_the_real_model
#[test]
#[ignore = "needs the ONNX runtime and model files (env-configured)"]
fn segment_with_the_real_model() {
    let Some(dylib) = env_or_skip("CRABCROP_TEST_ORT_DYLIB") else {
        return;
    };
    let Some(encoder) = env_or_skip("CRABCROP_TEST_SEGMENT_ENCODER_MODEL") else {
        return;
    };
    let Some(decoder) = env_or_skip("CRABCROP_TEST_SEGMENT_DECODER_MODEL") else {
        return;
    };
    let provider = OnnxSegmentProvider::load(Some(&dylib), &encoder, &decoder).unwrap();
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(200, 150, |x, y| {
        let dx = x as i32 - 100;
        let dy = y as i32 - 75;
        if dx * dx + dy * dy <= 40 * 40 {
            image::Rgb([220, 40, 30])
        } else {
            let v = 90 + ((x * 7 + y * 13) % 40) as u8;
            image::Rgb([v, v, v])
        }
    }));
    let mask = provider.segment(&img, (60, 35, 80, 80)).unwrap();
    assert_eq!((mask.width(), mask.height()), (80, 80));
    assert!(
        mask.get_pixel(40, 40)[0] > 200,
        "center {}",
        mask.get_pixel(40, 40)[0]
    );
    assert!(
        mask.get_pixel(2, 2)[0] < 30,
        "corner {}",
        mask.get_pixel(2, 2)[0]
    );
    let g = mask_geometry(&mask).expect("the disc has geometry");
    assert!(
        (0.6..0.92).contains(&g.coverage),
        "coverage {} (pi/4 is 0.785)",
        g.coverage
    );
    // Encoding is reusable: two boxes against one
    // encoding decode independently.
    let embedding = provider.encode(&img).unwrap();
    let again = provider
        .decode(&embedding, 200, 150, (60, 35, 80, 80))
        .unwrap();
    assert_eq!(again, mask);
    // The same scene at 10x, past the encoder's 1024:
    // the decoder answers at the encoder's scale and
    // the box is resampled to its own size (issue #410).
    let big = image::imageops::resize(&img.to_rgb8(), 2000, 1500, FilterType::Nearest);
    let mask = provider
        .segment(&DynamicImage::ImageRgb8(big), (600, 350, 800, 800))
        .unwrap();
    assert_eq!((mask.width(), mask.height()), (800, 800));
    assert!(mask.get_pixel(400, 400)[0] > 200, "center");
    assert!(mask.get_pixel(20, 20)[0] < 30, "corner");
    let g = mask_geometry(&mask).expect("the disc has geometry");
    assert!((0.6..0.92).contains(&g.coverage), "coverage {}", g.coverage);
}

/// A missing file in the pair fails as Load, the same
/// boot story as the other providers.
/// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored segment_load_missing
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn segment_load_missing_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let message = expect_load(OnnxSegmentProvider::load(
        Some(&dylib),
        &missing_path("segment-encoder"),
        &missing_path("segment-decoder"),
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// A one-input model is not a segment decoder: the
/// load rejects the pair as Load. The encoder file
/// stands in for the decoder here (it has one input).
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_SEGMENT_ENCODER_MODEL=...
/// cargo test -- --ignored segment_load_rejects
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn segment_load_rejects_a_decoder_without_the_prompt_inputs() {
    let Some(dylib) = env_or_skip("CRABCROP_TEST_ORT_DYLIB") else {
        return;
    };
    let Some(encoder) = env_or_skip("CRABCROP_TEST_SEGMENT_ENCODER_MODEL") else {
        return;
    };
    let message = expect_load(OnnxSegmentProvider::load(Some(&dylib), &encoder, &encoder));
    assert!(message.contains("a segment decoder takes"), "{message}");
}

#[test]
fn mask_planes_are_charged_against_the_pixel_budget() {
    // Issue #359: a source-sized mask plane passes; a
    // claimed shape past the shared output budget is a
    // bad output before any copy materializes it.
    assert!(check_mask_plane(8000, 8000).is_ok());
    let err = check_mask_plane(20000, 20000).unwrap_err();
    assert!(matches!(err, AiError::BadOutput(_)), "{err}");
}

#[test]
fn segment_mask_dims_stay_at_the_encoder_scale() {
    // Issue #410: frames that fit the encoder ask for
    // themselves; larger ones ask for the size the
    // encoder saw, never more than 1024 a side.
    let cases = [
        ((64, 48), (64, 48)),
        ((1024, 1024), (1024, 1024)),
        ((1025, 10), (1024, 10)),
        ((8192, 8192), (1024, 1024)),
        ((16384, 900), (1024, 56)),
        ((3000, 100_000), (31, 1024)),
        ((100_000, 3), (1024, 1)),
        ((0, 0), (1, 1)),
    ];
    for ((w, h), want) in cases {
        let got = segment_mask_dims(w, h);
        assert_eq!(got, want, "{w}x{h}");
        assert!(got.0 <= SEGMENT_INPUT && got.1 <= SEGMENT_INPUT);
    }
    // Above the bound, the plane is exactly what
    // segment_preprocess resized the frame to.
    assert_eq!(
        segment_mask_dims(4000, 3000),
        segment_input_dims(4000, 3000)
    );
}

/// A fake SAM decoder for `decode_box`: like the
/// `SamOnnxModel` export it answers exactly the plane it
/// is asked for, here four candidates. Candidate 2 has
/// the best predicted IoU and draws the object, an inner
/// rectangle a quarter of the box's area (logit +8
/// inside, -8 outside), at each plane pixel's center in
/// frame coordinates; the others claim the whole frame.
/// Records every plane it is asked for.
fn fake_decode(
    img_w: u32,
    img_h: u32,
    bbox: (u32, u32, u32, u32),
    asked: &mut Vec<(u32, u32)>,
) -> Result<GrayImage, AiError> {
    let (x, y, w, h) = bbox;
    let object = (
        x as f32 + w as f32 / 4.0,
        y as f32 + h as f32 / 4.0,
        x as f32 + w as f32 * 0.75,
        y as f32 + h as f32 * 0.75,
    );
    decode_box(img_w, img_h, bbox, |prompt, finish| {
        let (mw, mh) = (prompt.mask_w, prompt.mask_h);
        asked.push((mw, mh));
        let plane = (mw * mh) as usize;
        let mut logits = vec![8.0f32; 4 * plane];
        let (kx, ky) = (img_w as f32 / mw as f32, img_h as f32 / mh as f32);
        for py in 0..mh {
            for px in 0..mw {
                let (fx, fy) = ((px as f32 + 0.5) * kx, (py as f32 + 0.5) * ky);
                let inside =
                    (object.0..object.2).contains(&fx) && (object.1..object.3).contains(&fy);
                logits[2 * plane + (py * mw + px) as usize] = if inside { 8.0 } else { -8.0 };
            }
        }
        finish(MaskAnswer {
            dims: &[1, 4, mh as i64, mw as i64],
            logits: &logits,
            scores: &[0.2, 0.5, 0.97, 0.1],
        })
    })
}

#[test]
fn decode_box_never_asks_for_more_than_the_encoder_scale() {
    // Issue #410: the decoder used to be asked for the
    // whole frame, candidates x H x W floats per box. The
    // box mask must still come back at box size and
    // agree with a full-resolution answer away from the
    // object's edge.
    let cases = [
        (200, 150, (60, 35, 80, 80)),
        (8192, 8192, (2000, 3000, 1600, 1200)),
        (12_000, 9_000, (11_000, 100, 1_000, 500)),
        // Past the right edge: the strip beyond the
        // frame stays background.
        (16_384, 900, (16_000, 800, 600, 200)),
    ];
    for (img_w, img_h, bbox) in cases {
        let mut asked = Vec::new();
        let mask = fake_decode(img_w, img_h, bbox, &mut asked).unwrap();
        assert_eq!(asked, [segment_mask_dims(img_w, img_h)], "{img_w}x{img_h}");
        assert!(asked[0].0 <= SEGMENT_INPUT && asked[0].1 <= SEGMENT_INPUT);
        let (x, y, w, h) = bbox;
        assert_eq!((mask.width(), mask.height()), (w, h), "{img_w}x{img_h}");
        // One plane pixel in frame units, the blur the
        // bounded plane adds at the object's edge.
        let cell = (img_w as f32 / asked[0].0 as f32).max(img_h as f32 / asked[0].1 as f32);
        let (ox0, oy0) = (x as f32 + w as f32 / 4.0, y as f32 + h as f32 / 4.0);
        let (ox1, oy1) = (x as f32 + w as f32 * 0.75, y as f32 + h as f32 * 0.75);
        for (bx, by, p) in mask.enumerate_pixels() {
            let (fx, fy) = ((x + bx) as f32 + 0.5, (y + by) as f32 + 0.5);
            if x + bx >= img_w || y + by >= img_h {
                assert_eq!(p[0], 0, "off-frame ({bx}, {by})");
                continue;
            }
            let edge = (fx - ox0)
                .abs()
                .min((fx - ox1).abs())
                .min((fy - oy0).abs())
                .min((fy - oy1).abs());
            if edge <= 2.0 * cell {
                continue;
            }
            let inside = (ox0..ox1).contains(&fx) && (oy0..oy1).contains(&fy);
            let want = if inside { 255 } else { 0 };
            assert_eq!(p[0], want, "{img_w}x{img_h} box pixel ({bx}, {by})");
        }
    }
}

#[test]
fn decode_box_samples_a_smaller_plane_over_the_frame() {
    // A decoder answering a coarser plane than asked (a
    // low-resolution export) still yields a box-sized
    // mask, the plane stretched over the frame.
    let mask = decode_box(4000, 2000, (1000, 500, 400, 300), |_, finish| {
        let mut logits = vec![-8.0f32; 64 * 32];
        // Plane pixel (16, 8) covers frame x 1000..1062,
        // y 500..562: the box's top-left corner.
        logits[8 * 64 + 16] = 8.0;
        finish(MaskAnswer {
            dims: &[1, 1, 32, 64],
            logits: &logits,
            scores: &[0.9],
        })
    })
    .unwrap();
    assert_eq!((mask.width(), mask.height()), (400, 300));
    assert_eq!(mask.get_pixel(31, 31)[0], 255);
    assert_eq!(mask.get_pixel(399, 299)[0], 0);
}

#[test]
fn decode_box_rejects_answers_that_do_not_fit_the_request() {
    let run = |dims: [i64; 4], values: usize, scores: Vec<f32>| {
        decode_box(8192, 8192, (100, 100, 50, 50), |_, finish| {
            finish(MaskAnswer {
                dims: &dims,
                logits: &vec![0.0; values],
                scores: &scores,
            })
        })
    };
    // A plane bigger than the 1024x1024 asked for: the
    // decoder ignored orig_im_size.
    let err = run([1, 1, 8192, 8192], 16, vec![0.9]).unwrap_err();
    assert!(err.to_string().contains("for a 1024x1024 request"), "{err}");
    // Fewer values than the claimed shape.
    let err = run([1, 2, 1024, 1024], 1024 * 1024, vec![0.1, 0.9]).unwrap_err();
    assert!(err.to_string().contains("short of"), "{err}");
    // More candidates than scores, a negative dim, a
    // wrong rank, an empty plane.
    for (dims, scores) in [
        ([1, 3, 1024, 1024], vec![0.9]),
        ([1, 1, -1, 1024], vec![0.9]),
        ([2, 1, 1024, 1024], vec![0.9]),
        ([1, 1, 0, 1024], vec![0.9]),
        ([1, 0, 1024, 1024], vec![]),
    ] {
        let err = run(dims, 1024 * 1024, scores).unwrap_err();
        assert!(matches!(err, AiError::BadOutput(_)), "{dims:?}: {err}");
    }
    // An empty box never reaches the decoder.
    let mask = decode_box(8192, 8192, (10, 10, 0, 5), |_, _| {
        panic!("an empty box must not run the decoder")
    })
    .unwrap();
    assert_eq!((mask.width(), mask.height()), (0, 5));
}

#[test]
fn decode_box_reads_an_exact_plane_without_resampling() {
    // A frame that fits the encoder gets its own size
    // back, and each box pixel is the logistic of its
    // own logit, as before issue #410.
    let logits: Vec<f32> = (0..20 * 10).map(|i| (i % 7) as f32 - 3.0).collect();
    let mask = decode_box(20, 10, (3, 2, 5, 4), |prompt, finish| {
        assert_eq!((prompt.mask_w, prompt.mask_h), (20, 10));
        finish(MaskAnswer {
            dims: &[1, 1, 10, 20],
            logits: &logits,
            scores: &[0.9],
        })
    })
    .unwrap();
    for (bx, by, p) in mask.enumerate_pixels() {
        let v = logits[((2 + by) * 20 + 3 + bx) as usize];
        assert_eq!(p[0], sigmoid_strength(v), "({bx}, {by})");
    }
}
