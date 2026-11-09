use super::*;
use crate::ai::env_or_skip;
use crate::test_support::{ScratchDir, scratch_dir};

/// Write bytes to a fresh temp file and return the
/// scratch directory it lives in and its path (issue
/// #526).
fn temp_file(tag: &str, contents: &[u8]) -> (ScratchDir, String) {
    let dir = scratch_dir(&format!("ocr-{tag}"));
    let path = dir.join(tag).to_string_lossy().into_owned();
    std::fs::write(&path, contents).unwrap();
    (dir, path)
}

fn expect_load<T>(result: Result<T, AiError>) -> String {
    match result {
        Ok(_) => panic!("expected a load error"),
        Err(AiError::Load(message)) => message,
        Err(other) => panic!("expected a load error, got {other:?}"),
    }
}

#[test]
fn read_ocr_dict_keeps_every_line_verbatim() {
    // Only line terminators come off: a line that is
    // itself a space is a class, not padding, and a
    // trailing newline does not add a phantom class.
    let (_dict, path) = temp_file("dict2", b"a\nb\n \nx\r\nlast\n");
    let dict = read_ocr_dict(&path).unwrap();
    assert_eq!(dict, vec!["a", "b", " ", "x", "last"]);
    // No trailing newline: the last line still counts.
    let (_dict, path) = temp_file("dict3", b"a\nb");
    assert_eq!(read_ocr_dict(&path).unwrap(), vec!["a", "b"]);
    // An empty dictionary is a load error.
    let (_dict, path) = temp_file("dict4", b"");
    let message = expect_load(read_ocr_dict(&path));
    assert!(message.contains("contains no characters"), "{message}");
    // A missing dictionary is a load error naming it.
    let message = expect_load(read_ocr_dict("/nonexistent/ppocr-dict.txt"));
    assert!(
        message.contains("cannot read the OCR dictionary"),
        "{message}"
    );
}

#[test]
fn ocr_det_size_limits_and_rounds_to_32() {
    // Under the limit: only the stride rounding moves.
    assert_eq!(ocr_det_size(900, 360), (928, 384));
    assert_eq!(ocr_det_size(640, 480), (640, 480));
    // Over the limit: the longer side lands on it.
    assert_eq!(ocr_det_size(2000, 1000), (960, 480));
    // A tiny image still gets a runnable tensor.
    assert_eq!(ocr_det_size(10, 10), (32, 32));
}

#[test]
fn ocr_det_preprocess_normalizes_with_imagenet_stats() {
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(32, 32, image::Rgb([255, 0, 0])));
    let (input, rw, rh) = ocr_det_preprocess(&img);
    assert_eq!((rw, rh), (32, 32));
    let plane = 32 * 32;
    assert_eq!(input.len(), 3 * plane);
    assert!((input[0] - (1.0 - 0.485) / 0.229).abs() < 0.001);
    assert!((input[plane] - (0.0 - 0.456) / 0.224).abs() < 0.001);
    assert!((input[2 * plane] - (0.0 - 0.406) / 0.225).abs() < 0.001);
}

#[test]
fn convex_hull_drops_interior_and_collinear_points() {
    let mut points = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
    points.push((5.0, 5.0)); // interior
    points.push((5.0, 0.0)); // collinear on the top edge
    let hull = convex_hull(&points);
    assert_eq!(hull.len(), 4);
    for corner in [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)] {
        assert!(hull.contains(&corner), "missing {corner:?} in {hull:?}");
    }
}

#[test]
fn min_area_rect_fits_axis_aligned_and_rotated_squares() {
    let hull = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
    let rect = min_area_rect(&hull).unwrap();
    assert!((quad_area(&rect) - 100.0).abs() < 0.01);
    // A diamond (a square rotated 45 degrees, vertices
    // on the axes): its minimum rectangle is itself,
    // area half the bounding square.
    let hull = vec![(5.0, 0.0), (10.0, 5.0), (5.0, 10.0), (0.0, 5.0)];
    let rect = min_area_rect(&hull).unwrap();
    assert!(
        (quad_area(&rect) - 50.0).abs() < 0.01,
        "area {}",
        quad_area(&rect)
    );
    // Degenerate hulls have no rectangle.
    assert!(min_area_rect(&[(1.0, 1.0), (2.0, 2.0)]).is_none());
}

#[test]
fn unclip_grows_a_box_by_the_db_distance() {
    // 100x20 box: area 2000, perimeter 240, distance
    // 2000 * 1.5 / 240 = 12.5 on every side.
    let quad = [(0.0, 0.0), (100.0, 0.0), (100.0, 20.0), (0.0, 20.0)];
    let out = unclip_quad(&quad, 1.5);
    let want = [(-12.5, -12.5), (112.5, -12.5), (112.5, 32.5), (-12.5, 32.5)];
    for (got, want) in out.iter().zip(want) {
        assert!(
            (got.0 - want.0).abs() < 0.01 && (got.1 - want.1).abs() < 0.01,
            "got {got:?}, want {want:?}"
        );
    }
}

#[test]
fn order_quad_starts_top_left_and_runs_clockwise() {
    // The same square in a scrambled cyclic order.
    let mut quad = [(100.0, 20.0), (100.0, 0.0), (0.0, 0.0), (0.0, 20.0)];
    order_quad(&mut quad);
    assert_eq!(quad[0], (0.0, 0.0));
    assert_eq!(quad[1], (100.0, 0.0));
    assert_eq!(quad[2], (100.0, 20.0));
    assert_eq!(quad[3], (0.0, 20.0));
}

#[test]
fn ocr_det_quads_finds_boxes_and_drops_weak_and_tiny_ones() {
    // A 200x100 map at image scale. One strong text
    // blob, one weak blob (mean under the box
    // threshold), one specks blob (too few pixels).
    let (mw, mh) = (200usize, 100usize);
    let mut prob = vec![0.0f32; mw * mh];
    let fill = |prob: &mut Vec<f32>, x0: usize, x1: usize, y0: usize, y1: usize, v: f32| {
        for y in y0..=y1 {
            for x in x0..=x1 {
                prob[y * mw + x] = v;
            }
        }
    };
    fill(&mut prob, 50, 150, 40, 60, 0.9);
    fill(&mut prob, 20, 60, 75, 90, 0.4);
    fill(&mut prob, 180, 181, 10, 11, 0.95);
    let quads = ocr_det_quads(&prob, 200, 100, 200, 100);
    assert_eq!(quads.len(), 1, "quads: {quads:?}");
    let q = quads[0];
    // The box grew by the unclip distance 12.5 and the
    // first corner is its top-left.
    assert!(
        (q[0].0 - 37.5).abs() < 1.5 && (q[0].1 - 27.5).abs() < 1.5,
        "{q:?}"
    );
    assert!(
        (q[2].0 - 162.5).abs() < 1.5 && (q[2].1 - 72.5).abs() < 1.5,
        "{q:?}"
    );
}

#[test]
fn ocr_det_quads_drops_a_whole_frame_box() {
    // A uniformly hot map (the detector firing on a
    // smooth gradient background) yields one component
    // covering the frame: dropped, not a text line.
    let probs = vec![0.9f32; 64 * 64];
    assert!(ocr_det_quads(&probs, 64, 64, 64, 64).is_empty());
}

#[test]
fn ocr_det_quads_sorts_same_line_boxes_left_to_right() {
    // Two blobs: the right one starts 4 px higher, the
    // left one 4 px lower; they are one reading line
    // (within the 10 px tolerance), so x decides.
    let (mw, mh) = (200usize, 120usize);
    let mut prob = vec![0.0f32; mw * mh];
    for y in 40..=60 {
        for x in 120..=180 {
            prob[y * mw + x] = 0.9;
        }
    }
    for y in 44..=64 {
        for x in 20..=80 {
            prob[y * mw + x] = 0.9;
        }
    }
    let quads = ocr_det_quads(&prob, 200, 120, 200, 120);
    assert_eq!(quads.len(), 2);
    assert!(quads[0][0].0 < quads[1][0].0, "left box first: {quads:?}");
}

#[test]
fn warp_quad_identity_and_downscale_sample_the_source() {
    let mut img = image::RgbImage::new(100, 50);
    for (x, y, p) in img.enumerate_pixels_mut() {
        *p = image::Rgb([x as u8, y as u8, 7]);
    }
    let quad = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)];
    let same = warp_quad(&img, &quad, 100, 50).unwrap();
    assert_eq!(*same.get_pixel(37, 21), image::Rgb([37, 21, 7]));
    // Half size: output (10, 5) samples source (20, 10).
    let half = warp_quad(&img, &quad, 50, 25).unwrap();
    assert_eq!(*half.get_pixel(10, 5), image::Rgb([20, 10, 7]));
}

#[test]
fn rotate_ccw_turns_a_row_into_a_column() {
    let mut img = image::RgbImage::new(2, 1);
    img.put_pixel(0, 0, image::Rgb([10, 0, 0]));
    img.put_pixel(1, 0, image::Rgb([20, 0, 0]));
    let out = rotate_ccw(&img);
    assert_eq!((out.width(), out.height()), (1, 2));
    // Counterclockwise: the right end goes on top.
    assert_eq!(*out.get_pixel(0, 0), image::Rgb([20, 0, 0]));
    assert_eq!(*out.get_pixel(0, 1), image::Rgb([10, 0, 0]));
}

#[test]
fn ocr_rec_preprocess_shapes_and_scales() {
    let crop = image::RgbImage::from_pixel(100, 50, image::Rgb([255, 255, 255]));
    let (input, shape) = ocr_rec_preprocess(&crop);
    assert_eq!(shape, [1, 3, 48, 96]);
    assert_eq!(input.len(), 3 * 48 * 96);
    // White normalizes to 1 in the (x/255 - 0.5) / 0.5
    // scaling.
    assert!(input.iter().all(|&v| (v - 1.0).abs() < 0.001));
    // An absurd sliver hits the width cap instead of
    // asking for a hundred-thousand-pixel tensor.
    let sliver = image::RgbImage::from_pixel(4000, 3, image::Rgb([0, 0, 0]));
    let (_, shape) = ocr_rec_preprocess(&sliver);
    assert_eq!(shape[3], OCR_REC_MAX_WIDTH as usize);
}

#[test]
fn ocr_ctc_decode_collapses_repeats_and_scores_kept_steps() {
    // Classes: 0 blank, 1..=dict a/b, last space.
    let dict = vec!["a".to_string(), "b".to_string()];
    let classes = 4;
    let step = |idx: usize, conf: f32| {
        let mut row = vec![0.0f32; classes];
        row[idx] = conf;
        row
    };
    let mut probs = Vec::new();
    for (idx, conf) in [(1, 0.9), (1, 0.8), (0, 0.7), (2, 0.6), (3, 0.5), (1, 0.4)] {
        probs.extend(step(idx, conf));
    }
    let (text, score) = ocr_ctc_decode(&probs, 6, classes, &dict).unwrap();
    assert_eq!(text, "ab a");
    // The kept steps are the first a, the b, the
    // space, and the final a.
    assert!(
        (score - (0.9 + 0.6 + 0.5 + 0.4) / 4.0).abs() < 0.001,
        "{score}"
    );
    // All blank reads as no line at all.
    let blank: Vec<f32> = step(0, 0.99).repeat(3);
    assert!(ocr_ctc_decode(&blank, 3, classes, &dict).is_none());
}

/// Real-model verification, run manually with the
/// deployment artifacts in place:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_OCR_DETECT_MODEL=...
/// CRABCROP_TEST_OCR_RECOGNIZE_MODEL=... CRABCROP_TEST_OCR_DICT=...
/// cargo test -- --ignored ocr_with_the_real_models
#[test]
#[ignore = "needs the ONNX runtime and model files (env-configured)"]
fn ocr_with_the_real_models() {
    let Some(dylib) = env_or_skip("CRABCROP_TEST_ORT_DYLIB") else {
        return;
    };
    let Some(detect) = env_or_skip("CRABCROP_TEST_OCR_DETECT_MODEL") else {
        return;
    };
    let Some(recognize) = env_or_skip("CRABCROP_TEST_OCR_RECOGNIZE_MODEL") else {
        return;
    };
    let Some(dict) = env_or_skip("CRABCROP_TEST_OCR_DICT") else {
        return;
    };
    let provider = OnnxOcrProvider::load(Some(&dylib), &detect, &recognize, &dict, "en").unwrap();
    // The shipped dictionary is the PP-OCRv6 table.
    assert_eq!(provider.dict_len(), 18708);
    // A blank image has no text: no lines, no error.
    let blank = DynamicImage::new_rgb8(200, 100);
    let report = provider.recognize(&blank).unwrap();
    assert!(report.lines.is_empty(), "lines: {:?}", report.lines);
    assert_eq!(report.language, "en");
    // The functional corpus screenshot is the ground
    // truth: sharp UI text with known strings.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/functional/corpus/screenshot.png"
    );
    let img = image::open(path).unwrap();
    let report = provider.recognize(&img).unwrap();
    let content = report
        .lines
        .iter()
        .map(|l| l.text.clone())
        .collect::<Vec<_>>()
        .join("\n");
    for want in [
        "General",
        "Image pipeline",
        "# sharp edges stress text rendering",
        "jumps over the lazy dog.",
        "Save changes",
    ] {
        assert!(content.contains(want), "missing {want:?} in:\n{content}");
    }
    // Every reported box sits inside the image and
    // every score is a probability.
    for line in &report.lines {
        assert!(line.x + line.w <= img.width(), "{line:?}");
        assert!(line.y + line.h <= img.height(), "{line:?}");
        assert!((0.0..=1.0).contains(&line.score), "{line:?}");
    }
}

/// A dictionary that does not match the recognition
/// model fails the load probe: the class count the
/// probe reads cannot line up. Ignored like the other
/// load tests: load() initializes the runtime first.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_OCR_DETECT_MODEL=...
/// CRABCROP_TEST_OCR_RECOGNIZE_MODEL=...
/// cargo test -- --ignored ocr_load_rejects
#[test]
#[ignore = "needs the ONNX runtime and model files (env-configured)"]
fn ocr_load_rejects_a_mismatched_dictionary() {
    let Some(dylib) = env_or_skip("CRABCROP_TEST_ORT_DYLIB") else {
        return;
    };
    let Some(detect) = env_or_skip("CRABCROP_TEST_OCR_DETECT_MODEL") else {
        return;
    };
    let Some(recognize) = env_or_skip("CRABCROP_TEST_OCR_RECOGNIZE_MODEL") else {
        return;
    };
    let (_tiny, tiny) = temp_file("tiny-dict", b"a\nb\nc\n");
    let message = expect_load(OnnxOcrProvider::load(
        Some(&dylib),
        &detect,
        &recognize,
        &tiny,
        "en",
    ));
    assert!(message.contains("failed their load probe"), "{message}");
}

/// A missing model file fails as Load, the same boot
/// story as the other providers.
/// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored ocr_load_missing
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn ocr_load_missing_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let (_dict, dict) = temp_file("some-dict", b"a\nb\n");
    let message = expect_load(OnnxOcrProvider::load(
        Some(&dylib),
        "/nonexistent/ocr-det.onnx",
        "/nonexistent/ocr-rec.onnx",
        &dict,
        "en",
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}
