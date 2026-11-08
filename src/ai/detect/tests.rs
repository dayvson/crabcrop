use super::*;

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

/// Write bytes to a fresh temp file and return its
/// path: the load() failure tests need a real label
/// file without committing fixtures.
fn temp_file(tag: &str, contents: &[u8]) -> String {
    let path =
        std::env::temp_dir().join(format!("crabcrop-ai-load-{tag}-{}.bin", std::process::id()));
    std::fs::write(&path, contents).unwrap();
    path.to_string_lossy().into_owned()
}

/// A path inside the temp dir that is never created.
fn missing_path(tag: &str) -> String {
    std::env::temp_dir()
        .join(format!(
            "crabcrop-ai-load-missing-{tag}-{}.onnx",
            std::process::id()
        ))
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
    let garbage = temp_file("detect-garbage", b"this is not an onnx model at all");
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
    let labels = temp_file("detect-labels", b"person\ncar\n");
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
    let garbage = temp_file("declared-garbage", b"this is not an onnx model at all");
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
