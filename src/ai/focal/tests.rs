use super::*;
use crate::ai::env_or_skip;

fn grid(h: usize, w: usize, max_at: usize) -> Vec<f32> {
    let mut data = vec![0f32; h * w];
    data[max_at] = 9.0;
    data
}

#[test]
fn heatmap_argmax_reads_every_accepted_layout() {
    // 4 rows x 5 cols, max at row 2, col 3: the cell
    // center is (3.5/5, 2.5/4).
    let data = grid(4, 5, 2 * 5 + 3);
    for dims in [&[4, 5][..], &[1, 4, 5][..], &[1, 1, 4, 5][..]] {
        let (x, y) = heatmap_argmax(dims, &data).unwrap();
        assert!((x - 0.7).abs() < 1e-6, "x {x} for {dims:?}");
        assert!((y - 0.625).abs() < 1e-6, "y {y} for {dims:?}");
    }
}

#[test]
fn heatmap_argmax_takes_the_first_maximum_in_row_major_order() {
    let mut data = grid(3, 3, 8);
    data[2] = 9.0; // an equal maximum, earlier in row-major order
    let (x, y) = heatmap_argmax(&[3, 3], &data).unwrap();
    assert!((x - 2.5 / 3.0).abs() < 1e-6, "x {x}");
    assert!((y - 0.5 / 3.0).abs() < 1e-6, "y {y}");
}

#[test]
fn heatmap_argmax_of_a_single_cell_is_the_center() {
    let (x, y) = heatmap_argmax(&[1, 1], &[0.25]).unwrap();
    assert_eq!((x, y), (0.5, 0.5));
}

#[test]
fn heatmap_argmax_rejects_shapes_outside_the_contract() {
    let cases: Vec<(Vec<i64>, usize)> = vec![
        (vec![9], 9),             // a vector, not a grid
        (vec![2, 3, 3], 18),      // a batch of two grids
        (vec![1, 2, 3, 3], 18),   // two channels
        (vec![1, 1, 1, 3, 3], 9), // rank five
        (vec![3, 3], 8),          // fewer values than cells
        (vec![3, 3], 10),         // more values than cells
        (vec![0, 3], 0),          // an empty grid
    ];
    for (dims, len) in cases {
        let data = vec![0f32; len];
        let err = heatmap_argmax(&dims, &data).unwrap_err();
        assert!(matches!(err, AiError::BadOutput(_)), "{dims:?}: {err}");
    }
}

/// Assert a focal load() result failed as AiError::Load
/// and return the message for wording checks.
fn expect_load(result: Result<OnnxFocalProvider, AiError>) -> String {
    match result {
        Err(AiError::Load(message)) => message,
        Err(other) => panic!("expected a load error, got {other}"),
        Ok(_) => panic!("expected a load error, but the load succeeded"),
    }
}

/// Live check with the authored luma-heatmap model
/// (spikes/166-focal): the brightest block of the
/// fixture is the argmax. Ignored: needs the runtime
/// and the model: CRABCROP_TEST_ORT_DYLIB=...
/// CRABCROP_TEST_FOCAL_MODEL=...
/// cargo test -- --ignored focal_live
#[test]
#[ignore = "needs the ONNX runtime and a focal model (env-configured)"]
fn focal_live_point_tracks_the_brightest_block() {
    let Some(dylib) = env_or_skip("CRABCROP_TEST_ORT_DYLIB") else {
        return;
    };
    let Some(model) = env_or_skip("CRABCROP_TEST_FOCAL_MODEL") else {
        return;
    };
    let provider = OnnxFocalProvider::load(Some(&dylib), &model).unwrap();
    // 64x64 black with a white block at x 40..48,
    // y 8..16; the model sees it at 32x32, so the
    // block lands at x 20..24, y 4..8 and the first
    // row-major maximum is its top-left cell.
    let mut img = image::RgbImage::new(64, 64);
    for (x, y, p) in img.enumerate_pixels_mut() {
        if (40..48).contains(&x) && (8..16).contains(&y) {
            *p = image::Rgb([255, 255, 255]);
        }
    }
    let (x, y) = provider.focal_point(&DynamicImage::ImageRgb8(img)).unwrap();
    assert!((0.60..0.70).contains(&x), "x {x}");
    assert!((0.10..0.20).contains(&y), "y {y}");
}

/// A missing focal model file fails as AiError::Load.
/// CRABCROP_TEST_ORT_DYLIB=...
/// cargo test -- --ignored focal_load_missing
#[test]
#[ignore = "needs the ONNX runtime and a focal model (env-configured)"]
fn focal_load_missing_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let missing = std::env::temp_dir()
        .join(format!(
            "crabcrop-focal-missing-{}.onnx",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned();
    let message = expect_load(OnnxFocalProvider::load(Some(&dylib), &missing));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// A garbage focal model file fails the same way.
/// CRABCROP_TEST_ORT_DYLIB=...
/// cargo test -- --ignored focal_load_garbage
#[test]
#[ignore = "needs the ONNX runtime and a focal model (env-configured)"]
fn focal_load_garbage_model_file_is_a_load_error() {
    let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
    let path =
        std::env::temp_dir().join(format!("crabcrop-focal-garbage-{}.bin", std::process::id()));
    std::fs::write(&path, b"this is not an onnx model at all").unwrap();
    let message = expect_load(OnnxFocalProvider::load(
        Some(&dylib),
        &path.to_string_lossy(),
    ));
    assert!(message.contains("cannot load the model"), "{message}");
}

/// A multi-output model (the face detector stands in)
/// fails the probe: the contract takes exactly one
/// heatmap.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_FACE_MODEL=...
/// cargo test -- --ignored focal_load_multi_output
#[test]
#[ignore = "needs the ONNX runtime and a focal model (env-configured)"]
fn focal_load_multi_output_model_is_a_load_error() {
    let var = |name: &str| std::env::var(name).unwrap();
    let message = expect_load(OnnxFocalProvider::load(
        Some(&var("CRABCROP_TEST_ORT_DYLIB")),
        &var("CRABCROP_TEST_FACE_MODEL"),
    ));
    assert!(
        message.contains("does not produce a heatmap output"),
        "{message}"
    );
    assert!(message.contains("exactly one heatmap"), "{message}");
}

/// The segmentation model fails the same probe with
/// its seven outputs.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
/// cargo test -- --ignored focal_load_seven_output
#[test]
#[ignore = "needs the ONNX runtime and a focal model (env-configured)"]
fn focal_load_seven_output_model_is_a_load_error() {
    let var = |name: &str| std::env::var(name).unwrap();
    let message = expect_load(OnnxFocalProvider::load(
        Some(&var("CRABCROP_TEST_ORT_DYLIB")),
        &var("CRABCROP_TEST_REMOVEBG_MODEL"),
    ));
    assert!(message.contains("exactly one heatmap"), "{message}");
}

/// A model whose input shape is dynamic (the identity
/// fixture) never reaches the probe: the input contract
/// refuses it at load.
/// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_IDENTITY_MODEL=...
/// cargo test -- --ignored focal_load_dynamic_input
#[test]
#[ignore = "needs the ONNX runtime and a focal model (env-configured)"]
fn focal_load_dynamic_input_model_is_a_load_error() {
    let var = |name: &str| std::env::var(name).unwrap();
    let message = expect_load(OnnxFocalProvider::load(
        Some(&var("CRABCROP_TEST_ORT_DYLIB")),
        &var("CRABCROP_TEST_IDENTITY_MODEL"),
    ));
    assert!(
        message.contains("must declare a fixed [1, 1 or 3, H, W] input"),
        "{message}"
    );
}

#[test]
fn focal_dims_accept_shapes_within_the_cap() {
    // The detection provider's fixed frame size is the
    // cap itself; ordinary saliency shapes sit far
    // below it.
    assert_eq!(
        focal_dims(&[1, 3, 640, 640], "m.onnx").unwrap(),
        (3, 640, 640)
    );
    assert_eq!(
        focal_dims(&[1, 1, 256, 256], "m.onnx").unwrap(),
        (1, 256, 256)
    );
    assert_eq!(focal_dims(&[1, 3, 1, 1], "m.onnx").unwrap(), (3, 1, 1));
}

#[test]
fn focal_dims_reject_shapes_over_the_pixel_cap() {
    // The audit's shape: a small declared file hiding a
    // 16384x16384 frame would materialize gigabytes at
    // the boot probe (audit R2-26).
    let err = focal_dims(&[1, 3, 16384, 16384], "m.onnx").unwrap_err();
    assert!(err.to_string().contains("pixel cap"), "{err}");
    // A 4096x4096 frame boots but would repeat a roughly
    // 200 MB tensor allocation per smart crop: refused.
    let err = focal_dims(&[1, 1, 4096, 4096], "m.onnx").unwrap_err();
    assert!(err.to_string().contains("pixel cap"), "{err}");
    // One axis over the cap with the other at 1 pixel:
    // the product is what is bounded.
    assert!(focal_dims(&[1, 3, 640, 641], "m.onnx").is_err());
    // Absurd dims refuse cleanly instead of overflowing
    // the pixel count.
    assert!(focal_dims(&[1, 3, i64::MAX, i64::MAX], "m.onnx").is_err());
}

#[test]
fn focal_dims_keep_the_shape_contract() {
    // Rank, batch, channels, and zero axes refuse with
    // the shape message, as before the cap existed.
    for dims in [
        vec![3, 224, 224],
        vec![2, 3, 224, 224],
        vec![1, 2, 224, 224],
        vec![1, 3, 0, 224],
        vec![1, 3, 224, 0],
        vec![1, 3, -1, 224],
    ] {
        let err = focal_dims(&dims, "m.onnx").unwrap_err();
        assert!(
            err.to_string()
                .contains("must declare a fixed [1, 1 or 3, H, W] input"),
            "{dims:?}: {err}"
        );
    }
}
