use super::*;
use crate::ai::test_support::{
    expect_load, missing_path, noise_rgb, opaque_rgba, same_bits, solid, temp_file,
};

/// Real-model verification, run manually with the deployment
/// artifacts in place:
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_FACE_MODEL=...
/// cargo test -- --ignored face_detection_with_the_real_model
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn face_detection_with_the_real_model() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_FACE_MODEL").unwrap();
    let provider = OnnxFaceProvider::load(Some(&dylib), &model).unwrap();
    let bytes = include_bytes!("../../../tests/fixtures/portrait.jpg");
    let img = image::load_from_memory(bytes).unwrap();
    let faces = provider.detect(&img).unwrap();
    assert_eq!(faces.len(), 1, "faces: {faces:?}");
    let f = faces[0];
    assert!(f.score > 0.8, "score {}", f.score);
    // The fixture portrait's face, measured: the box lands
    // on it with room for build-to-build wobble.
    assert!((130..=175).contains(&f.x), "x {}", f.x);
    assert!((15..=50).contains(&f.y), "y {}", f.y);
    assert!((70..=110).contains(&f.w), "w {}", f.w);
    assert!((115..=165).contains(&f.h), "h {}", f.h);
    // A flat image has no faces.
    assert!(
        provider
            .detect(&solid(10, 20, 30, 120, 90))
            .unwrap()
            .is_empty()
    );
}

/// Wrong-model rejection for the face load probe (issue
/// #380), run manually with the deployment artifacts in
/// place: the removebg model is a valid ONNX file but not
/// a YuNet export, so loading it as the face model must
/// fail at load, not on the first face request.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
/// cargo test -- --ignored face_load_rejects_a_non_yunet_model
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn face_load_rejects_a_non_yunet_model() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_REMOVEBG_MODEL").unwrap();
    let err = OnnxFaceProvider::load(Some(&dylib), &model)
        .expect_err("a U2-Net model is not a YuNet export");
    assert!(
        matches!(&err, AiError::Load(m) if m.contains("face probe")),
        "load error: {err:?}"
    );
}

/// Load-failure coverage (#179, item 2): a missing model
/// file fails as AiError::Load. Ignored like the other
/// real-model tests: load() initializes the runtime
/// before it reaches the file, so a configured dylib is
/// required to exercise the file path at all.
/// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored face_load_missing
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn face_load_missing_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let message = expect_load(OnnxFaceProvider::load(Some(&dylib), &missing_path("face")));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// A file that exists but is not a model fails the same
/// way: the session builder rejects it as Load.
/// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored face_load_garbage
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn face_load_garbage_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let (_model, garbage) = temp_file("face-garbage", b"this is not an onnx model at all");
    let message = expect_load(OnnxFaceProvider::load(Some(&dylib), &garbage));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// A valid model with no inputs cannot serve as a face
/// detector: load() reports it as Load, not a session
/// with nothing to run.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_NO_INPUT_MODEL=...
/// cargo test -- --ignored face_load_model_without_inputs
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn face_load_model_without_inputs_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_NO_INPUT_MODEL").unwrap();
    let message = expect_load(OnnxFaceProvider::load(Some(&dylib), &model));
    assert!(message.contains("has no inputs"), "{message}");
}

/// Build a full YuNet output set with every head silent
/// (score 0 everywhere), ready for peaks to be planted.
fn silent_face_outputs() -> std::collections::HashMap<String, Vec<f32>> {
    let mut outputs = std::collections::HashMap::new();
    for stride in [8u32, 16, 32] {
        let grid = (FACE_INPUT / stride) as usize;
        outputs.insert(format!("cls_{stride}"), vec![0f32; grid * grid]);
        outputs.insert(format!("obj_{stride}"), vec![0f32; grid * grid]);
        outputs.insert(format!("bbox_{stride}"), vec![0f32; grid * grid * 4]);
    }
    outputs
}

/// Plant a detection peak: class/objectness scores and
/// the raw bbox offsets/log-sizes at one grid cell.
fn plant_peak(
    outputs: &mut std::collections::HashMap<String, Vec<f32>>,
    stride: u32,
    row: u32,
    col: u32,
    cls: f32,
    obj: f32,
    bbox: [f32; 4],
) {
    let grid = FACE_INPUT / stride;
    let idx = (row * grid + col) as usize;
    outputs.get_mut(&format!("cls_{stride}")).unwrap()[idx] = cls;
    outputs.get_mut(&format!("obj_{stride}")).unwrap()[idx] = obj;
    let head = outputs.get_mut(&format!("bbox_{stride}")).unwrap();
    head[idx * 4..idx * 4 + 4].copy_from_slice(&bbox);
}

#[test]
fn face_grid_peak_decodes_to_the_expected_box() {
    // A 640x640 source maps 1:1 from model space. A peak
    // at stride-8 cell (row 10, col 20) with zero offsets
    // and zero log-sizes: center (160, 80), size 8x8.
    let mut outputs = silent_face_outputs();
    plant_peak(&mut outputs, 8, 10, 20, 0.81, 1.0, [0.0, 0.0, 0.0, 0.0]);
    let found = decode_face_candidates(&outputs, 640, 640, FACE_SCORE_THRESHOLD).unwrap();
    assert_eq!(found.len(), 1);
    let f = found[0];
    assert_eq!((f.x, f.y, f.w, f.h), (156, 76, 8, 8));
    // The score is the geometric mean: sqrt(0.81 * 1.0).
    assert!((f.score - 0.9).abs() < 1e-6, "score {}", f.score);
}

#[test]
fn face_grid_offsets_and_log_sizes_decode() {
    // Offsets shift the center within the cell and sizes
    // are exponentiated: center ((20+0.5)*8, (10+0.25)*8)
    // = (164, 82), size (e^ln2 * 8, e^0 * 8) = (16, 8).
    let mut outputs = silent_face_outputs();
    plant_peak(
        &mut outputs,
        8,
        10,
        20,
        1.0,
        1.0,
        [0.5, 0.25, 2f32.ln(), 0.0],
    );
    let found = decode_face_candidates(&outputs, 640, 640, FACE_SCORE_THRESHOLD).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(
        (found[0].x, found[0].y, found[0].w, found[0].h),
        (156, 78, 16, 8)
    );
}

#[test]
fn face_grid_drops_peaks_below_the_threshold() {
    // sqrt(0.25 * 1.0) = 0.5 < 0.6: no candidate.
    let mut outputs = silent_face_outputs();
    plant_peak(&mut outputs, 8, 0, 0, 0.25, 1.0, [0.0; 4]);
    let found = decode_face_candidates(&outputs, 640, 640, FACE_SCORE_THRESHOLD).unwrap();
    assert!(found.is_empty());
}

#[test]
fn face_grid_scores_are_clamped_before_the_geometric_mean() {
    // Out-of-range head values clamp to 0..1 first:
    // sqrt(1.0 * 0.64) = 0.8, not sqrt(4.0 * 0.64) = 1.6.
    let mut outputs = silent_face_outputs();
    plant_peak(&mut outputs, 8, 3, 3, 4.0, 0.64, [0.0; 4]);
    let found = decode_face_candidates(&outputs, 640, 640, FACE_SCORE_THRESHOLD).unwrap();
    assert_eq!(found.len(), 1);
    assert!(
        (found[0].score - 0.8).abs() < 1e-6,
        "score {}",
        found[0].score
    );
    // The threshold is the caller's: at 0.85 the same
    // peak is dropped.
    let found = decode_face_candidates(&outputs, 640, 640, 0.85).unwrap();
    assert!(found.is_empty());
}

#[test]
fn face_grid_combines_all_three_strides_in_stride_order() {
    let mut outputs = silent_face_outputs();
    plant_peak(&mut outputs, 32, 5, 5, 1.0, 1.0, [0.0; 4]);
    plant_peak(&mut outputs, 8, 10, 20, 0.81, 1.0, [0.0; 4]);
    plant_peak(&mut outputs, 16, 2, 3, 0.64, 1.0, [0.0; 4]);
    let found = decode_face_candidates(&outputs, 640, 640, FACE_SCORE_THRESHOLD).unwrap();
    assert_eq!(found.len(), 3);
    // Strides decode in 8, 16, 32 order regardless of
    // planting order; unit sizes are one stride each.
    assert_eq!(
        (found[0].x, found[0].y, found[0].w, found[0].h),
        (156, 76, 8, 8)
    );
    assert_eq!(
        (found[1].x, found[1].y, found[1].w, found[1].h),
        (40, 24, 16, 16)
    );
    assert_eq!(
        (found[2].x, found[2].y, found[2].w, found[2].h),
        (144, 144, 32, 32)
    );
}

#[test]
fn face_grid_maps_back_with_independent_axis_scales_and_clamps() {
    // A 1280x640 source: model x coordinates double, y
    // coordinates do not. A peak in the corner cell hangs
    // off the top-left edge and clamps to it, keeping its
    // full scaled size.
    let mut outputs = silent_face_outputs();
    plant_peak(&mut outputs, 8, 0, 0, 1.0, 1.0, [0.0; 4]);
    let found = decode_face_candidates(&outputs, 1280, 640, FACE_SCORE_THRESHOLD).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(
        (found[0].x, found[0].y, found[0].w, found[0].h),
        (0, 0, 16, 8)
    );
}

#[test]
fn face_grid_missing_head_is_a_bad_output() {
    let mut outputs = silent_face_outputs();
    outputs.remove("bbox_16");
    let err = decode_face_candidates(&outputs, 640, 640, FACE_SCORE_THRESHOLD).unwrap_err();
    assert!(matches!(err, AiError::BadOutput(_)), "{err}");
    assert!(err.to_string().contains("bbox_16"), "{err}");
}

fn face(x: u32, y: u32, w: u32, h: u32, score: f32) -> FaceBox {
    FaceBox { x, y, w, h, score }
}

#[test]
fn iou_measures_overlap_containment_and_disjoint_boxes() {
    let a = face(0, 0, 10, 10, 0.9);
    // A box overlaps itself completely.
    assert_eq!(iou(&a, &a), 1.0);
    // Disjoint boxes do not overlap at all.
    assert_eq!(iou(&a, &face(20, 20, 10, 10, 0.5)), 0.0);
    // Edge-touching boxes share a border, not an area.
    assert_eq!(iou(&a, &face(10, 0, 10, 10, 0.5)), 0.0);
    // Containment: the intersection is the inner box,
    // the union is the outer one: 16 / 100.
    assert!((iou(&a, &face(2, 2, 4, 4, 0.5)) - 0.16).abs() < 1e-6);
    // Half-shifted: 50 shared of a 150 union = 1/3.
    assert!((iou(&a, &face(5, 0, 10, 10, 0.5)) - 1.0 / 3.0).abs() < 1e-6);
    // Two zero-area boxes have no union: defined as 0.
    assert_eq!(iou(&face(3, 3, 0, 0, 0.5), &face(3, 3, 0, 0, 0.5)), 0.0);
}

#[test]
fn nms_keeps_the_best_box_of_an_overlapping_cluster() {
    // IoU 80/120 = 0.67 > 0.3: the weaker box is dropped
    // even though it arrives first.
    let kept = nms(vec![face(2, 0, 10, 10, 0.8), face(0, 0, 10, 10, 0.9)]);
    assert_eq!(kept, vec![face(0, 0, 10, 10, 0.9)]);
}

#[test]
fn nms_keeps_disjoint_boxes_best_first() {
    let kept = nms(vec![
        face(100, 100, 10, 10, 0.7),
        face(0, 0, 10, 10, 0.95),
        face(50, 0, 8, 8, 0.8),
    ]);
    assert_eq!(
        kept,
        vec![
            face(0, 0, 10, 10, 0.95),
            face(50, 0, 8, 8, 0.8),
            face(100, 100, 10, 10, 0.7),
        ]
    );
}

#[test]
fn nms_keeps_a_box_at_exactly_the_iou_threshold() {
    // 60 shared of a 200 union is exactly 0.3, and the
    // rule is "at most the threshold survives".
    let kept = nms(vec![face(0, 0, 13, 10, 0.9), face(7, 0, 13, 10, 0.8)]);
    assert_eq!(kept.len(), 2);
}

#[test]
fn nms_of_nothing_is_nothing() {
    assert!(nms(Vec::new()).is_empty());
}

#[test]
fn face_tensor_is_bit_identical_to_the_indexed_fill() {
    // Issue #672: the planes are written front to back. The
    // tensor is the one the per-pixel indexed stores built:
    // BGR planar, each sample the pixel's own value.
    for (w, h) in [(FACE_INPUT, FACE_INPUT), (7, 5), (1, 1)] {
        let small = noise_rgb(w, h, w + h);
        let plane = (w * h) as usize;
        let mut want = vec![0f32; 3 * plane];
        for (i, px) in small.pixels().enumerate() {
            want[i] = px[2] as f32;
            want[plane + i] = px[1] as f32;
            want[2 * plane + i] = px[0] as f32;
        }
        assert!(same_bits(&face_tensor(&small), &want), "{w}x{h}");
    }
}

/// A fake YuNet that counts its runs. Its one face sits in
/// a cell picked from the input, so two inputs get two
/// answers and an answer handed to the wrong input shows.
fn fake_yunet(
    runs: &std::cell::Cell<u32>,
) -> impl Fn(Vec<f32>) -> Result<std::collections::HashMap<String, Vec<f32>>, AiError> + '_ {
    move |input| {
        runs.set(runs.get() + 1);
        assert_eq!(input.len(), 3 * (FACE_INPUT * FACE_INPUT) as usize);
        let sum: f32 = input.iter().step_by(1009).sum();
        let cell = sum as u32 % 6400;
        let mut outputs = silent_face_outputs();
        plant_peak(&mut outputs, 8, cell / 80, cell % 80, 1.0, 1.0, [0.0; 4]);
        Ok(outputs)
    }
}

/// What the model answers for a frame with no memo in the
/// way.
fn fresh_faces(img: &DynamicImage) -> Vec<FaceBox> {
    let runs = std::cell::Cell::new(0);
    let found = detect_faces(None, img, fake_yunet(&runs)).unwrap();
    assert_eq!(runs.get(), 1);
    found
}

#[test]
fn a_repeated_frame_is_answered_without_running_the_face_model() {
    // Issue #677: the second detection of a frame finds its
    // boxes in the memo. They are the boxes a fresh run
    // gives, and what is remembered is the model's input,
    // not the frame: a copy of the frame, and the same
    // pixels in a layout that is converted first, hit too.
    let memo = DetectionMemo::new();
    let runs = std::cell::Cell::new(0);
    let rgb = noise_rgb(300, 200, 61);
    let img = DynamicImage::ImageRgb8(rgb.clone());
    let first = detect_faces(Some(&memo), &img, fake_yunet(&runs)).unwrap();
    assert_eq!(runs.get(), 1);
    assert_eq!(first.len(), 1);
    assert_eq!(first, fresh_faces(&img));
    for again in [
        img.clone(),
        DynamicImage::ImageRgb8(rgb.clone()),
        opaque_rgba(&rgb),
    ] {
        assert_eq!(
            detect_faces(Some(&memo), &again, fake_yunet(&runs)).unwrap(),
            first
        );
        assert_eq!(runs.get(), 1, "a repeat ran the model");
    }
    // Another frame is another input: it runs the model
    // and gets its own boxes, and the first is still there.
    let other = DynamicImage::ImageRgb8(noise_rgb(300, 200, 62));
    let second = detect_faces(Some(&memo), &other, fake_yunet(&runs)).unwrap();
    assert_eq!(runs.get(), 2);
    assert_eq!(second, fresh_faces(&other));
    assert_ne!(second, first);
    assert_eq!(
        detect_faces(Some(&memo), &img, fake_yunet(&runs)).unwrap(),
        first
    );
    assert_eq!(
        detect_faces(Some(&memo), &other, fake_yunet(&runs)).unwrap(),
        second
    );
    assert_eq!(runs.get(), 2);
}

#[test]
fn one_differing_pixel_of_the_face_input_is_a_miss() {
    // A frame at the model's own size is its input as it
    // stands. One sample of one pixel, one level apart,
    // is another input: the model runs.
    let memo = DetectionMemo::new();
    let runs = std::cell::Cell::new(0);
    let rgb = noise_rgb(FACE_INPUT, FACE_INPUT, 63);
    let img = DynamicImage::ImageRgb8(rgb.clone());
    assert_eq!(
        fir_resize(&rgb, FACE_INPUT, FACE_INPUT).as_raw(),
        rgb.as_raw(),
        "a frame at the input size is the input"
    );
    let first = detect_faces(Some(&memo), &img, fake_yunet(&runs)).unwrap();
    for (x, y, c) in [(0, 0, 0), (639, 639, 2), (317, 201, 1)] {
        let mut changed = rgb.clone();
        let sample = &mut changed.get_pixel_mut(x, y).0[c];
        *sample = sample.wrapping_add(1);
        let changed = DynamicImage::ImageRgb8(changed);
        let before = runs.get();
        let found = detect_faces(Some(&memo), &changed, fake_yunet(&runs)).unwrap();
        assert_eq!(runs.get(), before + 1, "pixel {x},{y} sample {c} hit");
        assert_eq!(found, fresh_faces(&changed));
    }
    assert_eq!(
        detect_faces(Some(&memo), &img, fake_yunet(&runs)).unwrap(),
        first
    );
    assert_eq!(runs.get(), 4);
}

#[test]
fn the_same_face_input_at_another_source_size_is_a_miss() {
    // Two flat frames of one color resize to the same
    // input, and the model answers the same cell for both.
    // The boxes are mapped back to the source, so they
    // differ, and one frame's must never answer the other.
    let memo = DetectionMemo::new();
    let runs = std::cell::Cell::new(0);
    let narrow = solid(90, 120, 150, 100, 80);
    let wide = solid(90, 120, 150, 200, 80);
    let tall = solid(90, 120, 150, 100, 160);
    assert_eq!(
        fir_resize(&narrow.to_rgb8(), FACE_INPUT, FACE_INPUT).as_raw(),
        fir_resize(&wide.to_rgb8(), FACE_INPUT, FACE_INPUT).as_raw()
    );
    let in_narrow = detect_faces(Some(&memo), &narrow, fake_yunet(&runs)).unwrap();
    let in_wide = detect_faces(Some(&memo), &wide, fake_yunet(&runs)).unwrap();
    let in_tall = detect_faces(Some(&memo), &tall, fake_yunet(&runs)).unwrap();
    assert_eq!(runs.get(), 3);
    assert_eq!(in_narrow, fresh_faces(&narrow));
    assert_eq!(in_wide, fresh_faces(&wide));
    assert_eq!(in_tall, fresh_faces(&tall));
    assert_ne!(in_narrow, in_wide);
    assert_ne!(in_narrow, in_tall);
    // Each is remembered under its own size.
    assert_eq!(
        detect_faces(Some(&memo), &wide, fake_yunet(&runs)).unwrap(),
        in_wide
    );
    assert_eq!(
        detect_faces(Some(&memo), &narrow, fake_yunet(&runs)).unwrap(),
        in_narrow
    );
    assert_eq!(runs.get(), 3);
}

#[test]
fn a_failed_face_inference_is_not_remembered() {
    let memo = DetectionMemo::new();
    let img = DynamicImage::ImageRgb8(noise_rgb(120, 90, 64));
    // The model fails: the error goes out, nothing stays.
    let failed = detect_faces(Some(&memo), &img, |_input| {
        Err(AiError::Inference("inference failed: boom".to_string()))
    });
    assert_eq!(
        failed,
        Err(AiError::Inference("inference failed: boom".to_string()))
    );
    // The model answers something that does not decode.
    let undecodable = detect_faces(Some(&memo), &img, |_input| {
        let mut outputs = silent_face_outputs();
        outputs.remove("bbox_16");
        Ok(outputs)
    });
    assert!(matches!(undecodable, Err(AiError::BadOutput(_))));
    // The next detection runs the model and gets the
    // answer; only that is remembered.
    let runs = std::cell::Cell::new(0);
    let found = detect_faces(Some(&memo), &img, fake_yunet(&runs)).unwrap();
    assert_eq!(runs.get(), 1);
    assert_eq!(found, fresh_faces(&img));
    assert_eq!(
        detect_faces(Some(&memo), &img, fake_yunet(&runs)).unwrap(),
        found
    );
    assert_eq!(runs.get(), 1);
    // An empty frame never reaches the model or the memo.
    let empty = DynamicImage::new_rgb8(0, 7);
    assert_eq!(
        detect_faces(Some(&memo), &empty, fake_yunet(&runs)).unwrap(),
        []
    );
    assert_eq!(runs.get(), 1);
}

#[test]
fn concurrent_face_detections_share_the_memo_safely() {
    // Four threads detect over the same three frames
    // against one memo: every answer is the fresh one, and
    // the model runs at most once for a call and at least
    // once for a frame.
    let memo = DetectionMemo::new();
    let frames: Vec<DynamicImage> = (0..3)
        .map(|i| DynamicImage::ImageRgb8(noise_rgb(64, 48, 70 + i)))
        .collect();
    let fresh: Vec<Vec<FaceBox>> = frames.iter().map(fresh_faces).collect();
    let total = std::sync::atomic::AtomicU32::new(0);
    std::thread::scope(|scope| {
        for t in 0..4 {
            let (memo, frames, fresh, total) = (&memo, &frames, &fresh, &total);
            scope.spawn(move || {
                let runs = std::cell::Cell::new(0);
                for round in 0..4 {
                    let i = (round + t) % frames.len();
                    let found = detect_faces(Some(memo), &frames[i], fake_yunet(&runs)).unwrap();
                    assert_eq!(found, fresh[i]);
                }
                total.fetch_add(runs.get(), std::sync::atomic::Ordering::SeqCst);
            });
        }
    });
    let total = total.into_inner();
    assert!((3..=16).contains(&total), "{total} runs");
    // Everything is remembered now.
    let runs = std::cell::Cell::new(0);
    for (frame, want) in frames.iter().zip(&fresh) {
        assert_eq!(
            &detect_faces(Some(&memo), frame, fake_yunet(&runs)).unwrap(),
            want
        );
    }
    assert_eq!(runs.get(), 0);
}

#[test]
fn without_a_memo_every_face_detection_runs_the_model() {
    // A provider whose session lock is poisoned leaves its
    // memo out, so that every detection reaches the lock and
    // fails there as it always did. Without a memo nothing
    // is looked up and nothing is remembered: a frame the
    // memo knows runs the model again.
    let memo = DetectionMemo::new();
    let runs = std::cell::Cell::new(0);
    let img = DynamicImage::ImageRgb8(noise_rgb(120, 90, 65));
    let first = detect_faces(Some(&memo), &img, fake_yunet(&runs)).unwrap();
    assert_eq!(runs.get(), 1);
    assert_eq!(detect_faces(None, &img, fake_yunet(&runs)).unwrap(), first);
    assert_eq!(runs.get(), 2, "a detection without the memo did not run");
    // The model's failure goes out, although the memo could
    // have answered.
    let failed = detect_faces(None, &img, |_input| Err(AiError::LockPoisoned));
    assert_eq!(failed, Err(AiError::LockPoisoned));
    // And a frame detected without the memo is not in it.
    let other = DynamicImage::ImageRgb8(noise_rgb(120, 90, 66));
    detect_faces(None, &other, fake_yunet(&runs)).unwrap();
    assert_eq!(runs.get(), 3);
    detect_faces(Some(&memo), &other, fake_yunet(&runs)).unwrap();
    assert_eq!(runs.get(), 4);
}

/// The memo with the real model: a repeat gets the boxes
/// the model gave, and another frame gets its own.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_FACE_MODEL=...
/// cargo test -- --ignored face_detection_repeats_with_the_real_model
#[test]
#[ignore = "needs the ONNX runtime and a model file (env-configured)"]
fn face_detection_repeats_with_the_real_model() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let model = std::env::var("CRABCROP_TEST_FACE_MODEL").unwrap();
    let provider = OnnxFaceProvider::load(Some(&dylib), &model).unwrap();
    let bytes = include_bytes!("../../../tests/fixtures/portrait.jpg");
    let img = image::load_from_memory(bytes).unwrap();
    let first = provider.detect(&img).unwrap();
    assert_eq!(first.len(), 1, "faces: {first:?}");
    assert_eq!(provider.detect(&img).unwrap(), first);
    // The same portrait mirrored is another input, with
    // its face on the other side.
    let mirrored = img.fliph();
    let other = provider.detect(&mirrored).unwrap();
    assert_eq!(other.len(), 1, "faces: {other:?}");
    assert_ne!(other, first);
    assert_eq!(provider.detect(&mirrored).unwrap(), other);
    assert_eq!(provider.detect(&img).unwrap(), first);
    // A provider that has never seen the portrait gives
    // the same boxes: the memo changes no answer.
    let fresh = OnnxFaceProvider::load(Some(&dylib), &model).unwrap();
    assert_eq!(fresh.detect(&img).unwrap(), first);
}
