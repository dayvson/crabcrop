use super::*;
use crate::ai::test_support::{noise_rgb, opaque_rgba, same_bits};

fn classes() -> Vec<String> {
    ["person", "car"].iter().map(|s| s.to_string()).collect()
}

/// A channels-first output with two candidates on a
/// 100x100 image (no letterbox padding, scale 6.4: the
/// image was scaled to 640). Candidate 0: a person at
/// center (320,320), 128x256 in letterbox space, score
/// 0.9. Candidate 1: a car overlapping the person class
/// region but a different class, score 0.8.
fn synthetic(channels_first: bool) -> (Vec<f32>, Vec<i64>) {
    let classes = classes();
    let rows = 4 + classes.len();
    let n = 2usize;
    let mut data = vec![0f32; rows * n];
    let mut set = |cand: usize, ch: usize, v: f32| {
        if channels_first {
            data[ch * n + cand] = v;
        } else {
            data[cand * rows + ch] = v;
        }
    };
    // candidate 0: person
    set(0, 0, 320.0);
    set(0, 1, 320.0);
    set(0, 2, 128.0);
    set(0, 3, 256.0);
    set(0, 4, 0.9);
    // candidate 1: car, same box
    set(1, 0, 320.0);
    set(1, 1, 320.0);
    set(1, 2, 128.0);
    set(1, 3, 256.0);
    set(1, 5, 0.8);
    let dims = if channels_first {
        vec![1, rows as i64, n as i64]
    } else {
        vec![1, n as i64, rows as i64]
    };
    (data, dims)
}

#[test]
fn postprocess_reads_both_layouts() {
    for channels_first in [true, false] {
        let (data, dims) = synthetic(channels_first);
        let found =
            postprocess_detections(&data, &dims, &classes(), 0.5, 100, 100, 6.4, 0, 0).unwrap();
        assert_eq!(found.len(), 2, "channels_first={channels_first}");
        assert_eq!(found[0].label, "person");
        assert_eq!(found[0].score, 0.9);
        // (320-64)/6.4 = 40, (320-128)/6.4 = 30, 128/6.4 = 20, 256/6.4 = 40
        assert_eq!(
            (found[0].x, found[0].y, found[0].w, found[0].h),
            (40, 30, 20, 40)
        );
        assert_eq!(found[1].label, "car");
    }
}

#[test]
fn postprocess_threshold_and_same_class_nms() {
    let classes = classes();
    // One strong person, one weaker person at the same
    // box (suppressed), one weak car (below threshold).
    let rows = 4 + classes.len();
    let n = 3usize;
    let mut data = vec![0f32; rows * n];
    let mut set = |cand: usize, ch: usize, v: f32| data[ch * n + cand] = v;
    for (cand, score) in [(0usize, 0.9f32), (1, 0.7), (2, 0.3)] {
        set(cand, 0, 100.0);
        set(cand, 1, 100.0);
        set(cand, 2, 50.0);
        set(cand, 3, 50.0);
        set(cand, 4, score);
    }
    set(2, 4, 0.0);
    set(2, 5, 0.3);
    let found = postprocess_detections(
        &data,
        &[1, rows as i64, n as i64],
        &classes,
        0.5,
        640,
        640,
        1.0,
        0,
        0,
    )
    .unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].score, 0.9);
}

#[test]
fn postprocess_rejects_a_wrong_family_layout() {
    // A v5-style [1, N, 85] output for an 80-class
    // vocabulary does not fit 4+classes either way.
    let classes: Vec<String> = (0..80).map(|i| format!("class{i}")).collect();
    let err = postprocess_detections(
        &[0.0; 16],
        &[1, 25200, 85],
        &classes,
        0.5,
        10,
        10,
        1.0,
        0,
        0,
    )
    .unwrap_err();
    assert!(
        matches!(err, AiError::BadOutput(_)) && err.to_string().contains("4+classes"),
        "{err}"
    );
}

#[test]
fn names_metadata_parses_both_quote_styles() {
    let single = "{0: 'person', 1: 'bicycle', 2: 'car'}";
    assert_eq!(
        parse_names_metadata(single).unwrap(),
        ["person", "bicycle", "car"]
    );
    let double = "{\"0\": \"person\", \"1\": \"car\"}";
    assert_eq!(parse_names_metadata(double).unwrap(), ["person", "car"]);
    assert!(parse_names_metadata("no names here").is_none());
}

#[test]
fn letterbox_of_a_square_image_is_unpadded_at_scale_one() {
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        DETECT_INPUT,
        DETECT_INPUT,
        image::Rgb([10, 20, 30]),
    ));
    let (input, scale, pad_x, pad_y) = letterbox(&img);
    assert_eq!(scale, 1.0);
    assert_eq!((pad_x, pad_y), (0, 0));
    assert_eq!(input.len(), 3 * 640 * 640);
    // Planar CHW at 0..1: red plane, then green, then blue.
    let plane = 640 * 640;
    assert!((input[0] - 10.0 / 255.0).abs() < 1e-6, "red {}", input[0]);
    assert!((input[plane] - 20.0 / 255.0).abs() < 1e-6, "green");
    assert!((input[2 * plane] - 30.0 / 255.0).abs() < 1e-6, "blue");
}

#[test]
fn letterbox_pads_a_wide_image_top_and_bottom_with_the_standard_gray() {
    // 1280x640 fits at scale 0.5 as 640x320, centered:
    // 160 rows of pad above and below the content.
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        1280,
        640,
        image::Rgb([255, 0, 0]),
    ));
    let (input, scale, pad_x, pad_y) = letterbox(&img);
    assert_eq!(scale, 0.5);
    assert_eq!((pad_x, pad_y), (0, 160));
    let plane = 640 * 640;
    let gray = 114.0 / 255.0;
    // The padding reads as the standard gray in every
    // plane, including the last pad row.
    assert!((input[0] - gray).abs() < 1e-6);
    assert!((input[plane] - gray).abs() < 1e-6);
    assert!((input[159 * 640] - gray).abs() < 1e-6);
    // The first content row is the image itself: red at
    // full strength, green at zero.
    let first_content = 160 * 640;
    assert!((input[first_content] - 1.0).abs() < 1e-6);
    assert!(input[plane + first_content].abs() < 1e-6);
}

#[test]
fn letterbox_pads_a_tall_image_on_the_sides_and_upscales_a_small_one() {
    // 320x1280 fits at scale 0.5 as 160x640: the pad is
    // horizontal instead.
    let tall = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        320,
        1280,
        image::Rgb([0, 0, 0]),
    ));
    let (_, scale, pad_x, pad_y) = letterbox(&tall);
    assert_eq!(scale, 0.5);
    assert_eq!((pad_x, pad_y), (240, 0));
    // 100x50 upscales by 6.4 to 640x320: vertical pad.
    let small =
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(100, 50, image::Rgb([0, 0, 0])));
    let (_, scale, pad_x, pad_y) = letterbox(&small);
    assert_eq!(scale, 6.4);
    assert_eq!((pad_x, pad_y), (0, 160));
}

/// Live check with a real YOLOv8 export: a street
/// fixture must yield a person and a vehicle. Ignored:
/// needs the runtime, a detection model, and a photo:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_DETECT_MODEL=...
/// CRABCROP_TEST_DETECT_LABELS=... (optional)
/// CRABCROP_TEST_IMAGE=...
#[test]
#[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
fn detect_live_finds_people_and_vehicles() {
    let var = |name: &str| std::env::var(name).unwrap();
    let provider = OnnxDetectionProvider::load(
        Some(&var("CRABCROP_TEST_ORT_DYLIB")),
        &var("CRABCROP_TEST_DETECT_MODEL"),
        std::env::var("CRABCROP_TEST_DETECT_LABELS").ok().as_deref(),
        0.35,
    )
    .unwrap();
    println!("classes: {} known", provider.describe().len());
    let img = image::open(var("CRABCROP_TEST_IMAGE")).unwrap();
    let found = provider.detect(&img).unwrap();
    for d in &found {
        println!(
            "  {} {:.3} at {},{} {}x{}",
            d.label, d.score, d.x, d.y, d.w, d.h
        );
    }
    assert!(
        found.iter().any(|d| d.label == "person"),
        "no person in {found:?}"
    );
    assert!(
        found
            .iter()
            .any(|d| ["car", "truck", "bus"].contains(&d.label.as_str())),
        "no vehicle in {found:?}"
    );
}

/// Write bytes to a fresh temp file and return the
/// scratch directory it lives in and its path: the
/// load() failure tests need a real label file without
/// committing fixtures (issue #526).
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

/// Load-failure coverage (#179, item 2): a missing
/// detection model file fails as AiError::Load.
/// CRABCROP_TEST_ORT_DYLIB=...
/// cargo test -- --ignored detection_load_missing
#[test]
#[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
fn detection_load_missing_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let message = expect_load(OnnxDetectionProvider::load(
        Some(&dylib),
        &missing_path("detect"),
        None,
        0.5,
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// A garbage detection model file fails the same way.
/// CRABCROP_TEST_ORT_DYLIB=...
/// cargo test -- --ignored detection_load_garbage
#[test]
#[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
fn detection_load_garbage_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let (_model, garbage) = temp_file("detect-garbage", b"this is not an onnx model at all");
    let message = expect_load(OnnxDetectionProvider::load(
        Some(&dylib),
        &garbage,
        None,
        0.5,
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// A model with no `names` metadata (the removebg
/// model stands in) and no labels file configured has
/// no class vocabulary: Load, naming the remedy.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
/// cargo test -- --ignored detection_load_model_without_class_names
#[test]
#[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
fn detection_load_model_without_class_names_or_labels_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_REMOVEBG_MODEL").unwrap();
    let message = expect_load(OnnxDetectionProvider::load(Some(&dylib), &model, None, 0.5));
    assert!(
        message.contains("carries no class names metadata"),
        "{message}"
    );
}

/// The same model with a labels path that does not
/// exist fails reading the labels: Load.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
/// cargo test -- --ignored detection_load_missing_labels
#[test]
#[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
fn detection_load_missing_labels_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_REMOVEBG_MODEL").unwrap();
    let message = expect_load(OnnxDetectionProvider::load(
        Some(&dylib),
        &model,
        Some(&missing_path("detect-labels")),
        0.5,
    ));
    assert!(
        message.contains("cannot read the detection labels"),
        "{message}"
    );
}

/// With labels supplied, a non-YOLO model reaches the
/// load probe and fails it: Load, classified like the
/// other providers' probe failures.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
/// cargo test -- --ignored detection_load_non_yolo
#[test]
#[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
fn detection_load_non_yolo_model_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_REMOVEBG_MODEL").unwrap();
    let (_labels, labels) = temp_file("detect-labels", b"person\ncar\n");
    let message = expect_load(OnnxDetectionProvider::load(
        Some(&dylib),
        &model,
        Some(&labels),
        0.5,
    ));
    assert!(message.contains("not a YOLOv8-family export"), "{message}");
}

/// Eighty class names in model order with index 0
/// renamed: enough for the real YOLO export's probe,
/// and proof of which vocabulary won.
fn renamed_vocabulary() -> Vec<String> {
    (0..80)
        .map(|i| {
            if i == 0 {
                "human".to_string()
            } else {
                format!("class{i}")
            }
        })
        .collect()
}

/// Declared classes win over the model's own `names`
/// metadata (issue #166): the same YOLO export, loaded
/// with a declaration vocabulary, reports and returns
/// the declaration's names.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_DETECT_MODEL=...
/// CRABCROP_TEST_IMAGE=...
/// cargo test -- --ignored declared_classes_win
#[test]
#[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
fn declared_classes_win_over_model_metadata() {
    let var = |name: &str| std::env::var(name).unwrap();
    let provider = OnnxDetectionProvider::load_with_classes(
        Some(&var("CRABCROP_TEST_ORT_DYLIB")),
        &var("CRABCROP_TEST_DETECT_MODEL"),
        Some(renamed_vocabulary()),
        None,
        0.35,
    )
    .unwrap();
    assert_eq!(provider.describe()[0], "human");
    assert!(provider.knows_class("human"));
    assert!(!provider.knows_class("person"));
    let img = image::open(var("CRABCROP_TEST_IMAGE")).unwrap();
    let found = provider.detect(&img).unwrap();
    assert!(
        found.iter().any(|d| d.label == "human"),
        "no human in {found:?}"
    );
    assert!(
        found.iter().all(|d| d.label != "person"),
        "metadata vocabulary leaked into {found:?}"
    );
}

/// A declared detector whose model file is missing
/// fails as AiError::Load.
/// CRABCROP_TEST_ORT_DYLIB=...
/// cargo test -- --ignored declared_load_missing
#[test]
#[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
fn declared_load_missing_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let message = expect_load(OnnxDetectionProvider::load_with_classes(
        Some(&dylib),
        &missing_path("declared"),
        Some(renamed_vocabulary()),
        None,
        0.5,
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// A garbage declared model file fails the same way.
/// CRABCROP_TEST_ORT_DYLIB=...
/// cargo test -- --ignored declared_load_garbage
#[test]
#[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
fn declared_load_garbage_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let (_model, garbage) = temp_file("declared-garbage", b"this is not an onnx model at all");
    let message = expect_load(OnnxDetectionProvider::load_with_classes(
        Some(&dylib),
        &garbage,
        Some(renamed_vocabulary()),
        None,
        0.5,
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// Declared classes do not save a non-YOLO model: the
/// probe still refuses it as Load.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
/// cargo test -- --ignored declared_load_non_yolo
#[test]
#[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
fn declared_load_non_yolo_model_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_REMOVEBG_MODEL").unwrap();
    let message = expect_load(OnnxDetectionProvider::load_with_classes(
        Some(&dylib),
        &model,
        Some(vec!["a".to_string(), "b".to_string()]),
        None,
        0.5,
    ));
    assert!(message.contains("not a YOLOv8-family export"), "{message}");
}

#[test]
fn letterbox_reads_an_rgb8_frame_in_place_to_the_same_tensor() {
    // Issue #672: an RGB8 frame is resized from where it
    // is. The tensor, scale and padding are the ones the
    // same colors give in a layout that is converted
    // first: wide, tall, upscaled and one pixel.
    for (w, h) in [(810, 540), (301, 517), (5, 3), (1, 1)] {
        let rgb = noise_rgb(w, h, w * 5 + h);
        let (borrowed, scale, pad_x, pad_y) = letterbox(&DynamicImage::ImageRgb8(rgb.clone()));
        let (converted, scale2, pad_x2, pad_y2) = letterbox(&opaque_rgba(&rgb));
        assert!(same_bits(&borrowed, &converted), "{w}x{h}");
        assert_eq!((scale, pad_x, pad_y), (scale2, pad_x2, pad_y2), "{w}x{h}");
    }
}

/// `letterbox` as it filled its tensor before issue #676: a
/// division and a remainder place every pixel.
fn letterbox_by_pixels(img: &DynamicImage) -> (Vec<f32>, f32, u32, u32) {
    let (w, h) = (img.width().max(1), img.height().max(1));
    let scale = (DETECT_INPUT as f32 / w as f32).min(DETECT_INPUT as f32 / h as f32);
    let nw = ((w as f32 * scale).round() as u32).clamp(1, DETECT_INPUT);
    let nh = ((h as f32 * scale).round() as u32).clamp(1, DETECT_INPUT);
    let resized = fir_resize(&img.to_rgb8(), nw, nh);
    let pad_x = (DETECT_INPUT - nw) / 2;
    let pad_y = (DETECT_INPUT - nh) / 2;
    let plane = (DETECT_INPUT * DETECT_INPUT) as usize;
    let mut out = vec![114f32 / 255.0; 3 * plane];
    for (i, px) in resized.pixels().enumerate() {
        let x = i as u32 % nw;
        let y = i as u32 / nw;
        let at = ((y + pad_y) * DETECT_INPUT + (x + pad_x)) as usize;
        for c in 0..3 {
            out[c * plane + at] = px[c] as f32 / 255.0;
        }
    }
    (out, scale, pad_x, pad_y)
}

#[test]
fn letterbox_fills_by_rows_to_the_per_pixel_tensor() {
    // Issue #676: the tensor, the scale and the padding are
    // the ones the per-pixel fill gave: wide, tall, square,
    // upscaled, a sliver either way, and one pixel.
    for (w, h) in [
        (810, 540),
        (301, 517),
        (640, 640),
        (64, 64),
        (5, 3),
        (2000, 3),
        (3, 2000),
        (1, 1),
    ] {
        let img = DynamicImage::ImageRgb8(noise_rgb(w, h, w * 29 + h));
        let (got, scale, pad_x, pad_y) = letterbox(&img);
        let (want, want_scale, want_pad_x, want_pad_y) = letterbox_by_pixels(&img);
        assert!(same_bits(&got, &want), "{w}x{h}");
        assert_eq!(scale.to_bits(), want_scale.to_bits(), "{w}x{h}");
        assert_eq!((pad_x, pad_y), (want_pad_x, want_pad_y), "{w}x{h}");
    }
}

/// `postprocess_detections` as it was before issue #676:
/// the best class of a candidate found by reading one value
/// from each class row, for both layouts.
#[allow(clippy::too_many_arguments)]
fn postprocess_by_candidates(
    data: &[f32],
    dims: &[i64],
    classes: &[String],
    threshold: f32,
    img_w: u32,
    img_h: u32,
    scale: f32,
    pad_x: u32,
    pad_y: u32,
) -> Result<Vec<Detection>, AiError> {
    if dims.len() != 3 || dims[0] != 1 {
        return Err(AiError::BadOutput(format!(
            "expected a [1, 4+classes, N] output, got dims {dims:?}"
        )));
    }
    let rows = (4 + classes.len()) as i64;
    let channels_first = dims[1] == rows;
    let transposed = dims[2] == rows;
    if !channels_first && !transposed {
        return Err(AiError::BadOutput(format!(
            "expected a [1, 4+classes, N] output for {} classes, got dims {dims:?}",
            classes.len()
        )));
    }
    let n = if channels_first { dims[2] } else { dims[1] } as usize;
    let stride = rows as usize;
    let at = |candidate: usize, channel: usize| -> f32 {
        if channels_first {
            data[channel * n + candidate]
        } else {
            data[candidate * stride + channel]
        }
    };
    let mut found: Vec<Detection> = Vec::new();
    for i in 0..n {
        let (mut best_class, mut best_score) = (0usize, f32::MIN);
        for (c, _) in classes.iter().enumerate() {
            let score = at(i, 4 + c);
            if score > best_score {
                best_score = score;
                best_class = c;
            }
        }
        if best_score < threshold {
            continue;
        }
        let (cx, cy, bw, bh) = (at(i, 0), at(i, 1), at(i, 2), at(i, 3));
        let x1 = ((cx - bw / 2.0) - pad_x as f32) / scale;
        let y1 = ((cy - bh / 2.0) - pad_y as f32) / scale;
        let x2 = ((cx + bw / 2.0) - pad_x as f32) / scale;
        let y2 = ((cy + bh / 2.0) - pad_y as f32) / scale;
        let x = x1.round().clamp(0.0, img_w as f32) as u32;
        let y = y1.round().clamp(0.0, img_h as f32) as u32;
        let x2 = x2.round().clamp(0.0, img_w as f32) as u32;
        let y2 = y2.round().clamp(0.0, img_h as f32) as u32;
        if x2 <= x || y2 <= y {
            continue;
        }
        found.push(Detection {
            label: classes[best_class].clone(),
            score: (best_score * 1000.0).round() / 1000.0,
            x,
            y,
            w: x2 - x,
            h: y2 - y,
        });
    }
    Ok(nms_detections(found))
}

/// A channels-first output of `n` candidates over seven
/// classes: noisy scores and disjoint boxes, with the cases
/// a scan order could settle differently planted in it.
fn tricky_output(n: usize) -> (Vec<f32>, Vec<i64>, Vec<String>) {
    let classes: Vec<String> = (0..7).map(|c| format!("class{c}")).collect();
    let rows = 4 + classes.len();
    let mut state = 0x9e37_79b9u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    let mut data = vec![0f32; rows * n];
    for i in 0..n {
        // Boxes on a grid, 20 pixels a side, so that
        // suppression does not hide a wrong class.
        data[i] = 20.0 + (i % 20) as f32 * 30.0;
        data[n + i] = 20.0 + (i / 20) as f32 * 30.0;
        data[2 * n + i] = 20.0;
        data[3 * n + i] = 20.0;
        for c in 0..classes.len() {
            data[(4 + c) * n + i] = next();
        }
    }
    let mut plant = |i: usize, scores: [f32; 7]| {
        for (c, score) in scores.into_iter().enumerate() {
            data[(4 + c) * n + i] = score;
        }
    };
    let nan = f32::NAN;
    // Ties: the first of the equal classes wins.
    plant(0, [0.9, 0.9, 0.1, 0.9, 0.2, 0.3, 0.9]);
    plant(1, [0.1, 0.7, 0.7, 0.2, 0.7, 0.3, 0.1]);
    plant(2, [0.8; 7]);
    // A NaN never wins and never ends the scan.
    plant(3, [nan, 0.6, nan, 0.95, nan, 0.1, 0.2]);
    plant(4, [0.6, nan, 0.99, nan, 0.1, 0.2, nan]);
    // Nothing but NaN: no class, no detection.
    plant(5, [nan; 7]);
    // The lowest scores there are: none passes the start.
    plant(6, [f32::MIN; 7]);
    plant(7, [f32::NEG_INFINITY; 7]);
    // Infinity wins once, at its first class.
    plant(8, [0.5, f32::INFINITY, 0.7, f32::INFINITY, 0.1, 0.2, 0.3]);
    // The last class and the first.
    plant(9, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.97]);
    plant(10, [0.97, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]);
    // Signed zeros compare equal: the first stays.
    plant(11, [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0]);
    (data, vec![1, rows as i64, n as i64], classes)
}

/// The same output with candidates as rows.
fn transpose(data: &[f32], dims: &[i64]) -> (Vec<f32>, Vec<i64>) {
    let (rows, n) = (dims[1] as usize, dims[2] as usize);
    let mut out = vec![0f32; rows * n];
    for r in 0..rows {
        for i in 0..n {
            out[i * rows + r] = data[r * n + i];
        }
    }
    (out, vec![1, n as i64, rows as i64])
}

#[test]
fn the_class_scan_by_rows_picks_what_the_scan_by_candidates_picked() {
    // Issue #676: a channels-first output is scanned class
    // row by class row. Every candidate still gets the
    // class and score the per-candidate scan gave it, ties
    // and NaN included, at every threshold; and a
    // transposed output, which keeps the per-candidate
    // scan, gives the same detections again.
    for n in [12usize, 400] {
        let (data, dims, classes) = tricky_output(n);
        let (flipped, flipped_dims) = transpose(&data, &dims);
        for threshold in [0.0, 0.25, 0.5, 0.9, 1.0, f32::MIN] {
            let want =
                postprocess_by_candidates(&data, &dims, &classes, threshold, 640, 640, 1.0, 0, 0)
                    .unwrap();
            let got =
                postprocess_detections(&data, &dims, &classes, threshold, 640, 640, 1.0, 0, 0)
                    .unwrap();
            let label = format!("{n} candidates at {threshold}");
            assert_eq!(format!("{got:?}"), format!("{want:?}"), "{label}");
            let got = postprocess_detections(
                &flipped,
                &flipped_dims,
                &classes,
                threshold,
                640,
                640,
                1.0,
                0,
                0,
            )
            .unwrap();
            assert_eq!(
                format!("{got:?}"),
                format!("{want:?}"),
                "transposed, {label}"
            );
        }
        // The planted cases decided what they should have.
        let found =
            postprocess_detections(&data, &dims, &classes, 0.5, 640, 640, 1.0, 0, 0).unwrap();
        let label_at = |i: usize| {
            let (x, y) = (10 + (i % 20) as u32 * 30, 10 + (i / 20) as u32 * 30);
            found
                .iter()
                .find(|d| (d.x, d.y) == (x, y))
                .map(|d| d.label.as_str())
        };
        assert_eq!(label_at(0), Some("class0"));
        assert_eq!(label_at(1), Some("class1"));
        assert_eq!(label_at(2), Some("class0"));
        assert_eq!(label_at(3), Some("class3"));
        assert_eq!(label_at(4), Some("class2"));
        assert_eq!(label_at(5), None);
        assert_eq!(label_at(6), None);
        assert_eq!(label_at(7), None);
        assert_eq!(label_at(8), Some("class1"));
        assert_eq!(label_at(9), Some("class6"));
        assert_eq!(label_at(10), Some("class0"));
        assert_eq!(label_at(11), None);
    }
    // No classes at all, and no candidates.
    let none: Vec<String> = Vec::new();
    for (data, dims) in [
        (vec![1.0f32; 4 * 3], vec![1, 4, 3]),
        (Vec::new(), vec![1, 4, 0]),
    ] {
        let want = postprocess_by_candidates(&data, &dims, &none, 0.5, 64, 64, 1.0, 0, 0).unwrap();
        let got = postprocess_detections(&data, &dims, &none, 0.5, 64, 64, 1.0, 0, 0).unwrap();
        assert_eq!(format!("{got:?}"), format!("{want:?}"));
    }
}
