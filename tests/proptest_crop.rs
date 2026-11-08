//! Property tests for crop geometry (issue #478): for
//! arbitrary image and target sizes, alignments, focal
//! rectangles, focal points, face boxes, and object unions,
//! the window every crop entry point cuts is
//!
//! - non-empty;
//! - inside the image;
//! - never larger than the fill window, the largest window
//!   of the target aspect the image holds (whole pixels,
//!   rounded down, at least 1);
//! - at the target aspect within a pixel of rounding: some
//!   exact `target_w:target_h` rectangle lies within one
//!   pixel of it on each axis.
//!
//! The traced entry points report the window as (left, top,
//! width, height), so no pixel is inspected. The properties
//! hold however the engine maps a focal form onto the frame
//! before the crop runs (issue #465): rectangles past the
//! frame, reversed corners, and non-finite points and zooms
//! are all generated.
//!
//! The cases per property default to [`DEFAULT_CASES`];
//! `PROPTEST_CASES` raises it (the nightly job does). A
//! failure is saved next to this file in
//! `proptest_crop.proptest-regressions`, which is checked in
//! so the seed replays first on every later run.

use crabcrop::ai::{AiError, FaceBox, FaceProvider, FocalProvider};
use crabcrop::crop;
use crabcrop::url::{Align, MAX_DIMENSION};
use image::DynamicImage;
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

/// Cases per property when `PROPTEST_CASES` is unset.
const DEFAULT_CASES: u32 = 64;

fn config() -> ProptestConfig {
    let mut config = ProptestConfig {
        failure_persistence: Some(Box::new(FileFailurePersistence::WithSource(
            "proptest-regressions",
        ))),
        ..ProptestConfig::default()
    };
    if std::env::var_os("PROPTEST_CASES").is_none() {
        config.cases = DEFAULT_CASES;
    }
    config
}

/// A face provider that reports fixed boxes.
#[derive(Debug)]
struct Faces(Vec<FaceBox>);

impl FaceProvider for Faces {
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<FaceBox>, AiError> {
        Ok(self.0.clone())
    }
}

/// A declared focal provider that reports a fixed point,
/// finite or not (the chain clamps what it gets).
#[derive(Debug)]
struct Point(f32, f32);

impl FocalProvider for Point {
    fn focal_point(&self, _img: &DynamicImage) -> Result<(f32, f32), AiError> {
        Ok((self.0, self.1))
    }
}

/// Image dimensions: square-ish, extreme strips both ways,
/// and tiny frames, with the pixel count kept small enough
/// that a debug build copies the cut window quickly.
fn image_size() -> impl Strategy<Value = (u32, u32)> {
    prop_oneof![
        (1u32..=64, 1u32..=64),
        (1u32..=400, 1u32..=400),
        (1u32..=4096, 1u32..=8),
        (1u32..=8, 1u32..=4096),
    ]
}

/// A target size within the grammar: every crop and pad box
/// is 1 to `MAX_DIMENSION` pixels per axis.
fn target() -> impl Strategy<Value = (u32, u32)> {
    let side = prop_oneof![
        1u32..=64,
        1u32..=MAX_DIMENSION,
        Just(1),
        Just(MAX_DIMENSION)
    ];
    (side.clone(), side)
}

/// A fraction or zoom, mostly in range, sometimes anything
/// an `f32` can hold (NaN and the infinities included).
fn loose_f32(lo: f32, hi: f32) -> impl Strategy<Value = f32> {
    prop_oneof![4 => lo..=hi, 1 => any::<f32>()]
}

/// Face boxes inside a `w` by `h` image, as providers report
/// them.
fn faces_in(w: u32, h: u32) -> impl Strategy<Value = Vec<FaceBox>> {
    let face = (0..w, 0..h).prop_flat_map(move |(x, y)| {
        (Just(x), Just(y), 0..=w - x, 0..=h - y, 0.0f32..=1.0)
            .prop_map(|(x, y, w, h, score)| FaceBox { x, y, w, h, score })
    });
    prop::collection::vec(face, 0..=4)
}

/// The largest `tw:th` window a `w` by `h` image holds, in
/// whole pixels: the full extent on one axis, the other
/// rounded down (at least 1).
fn fill_window(w: u32, h: u32, tw: u32, th: u32) -> (u32, u32) {
    let (w64, h64, tw64, th64) = (w as u64, h as u64, tw as u64, th as u64);
    if w64 * th64 > tw64 * h64 {
        ((h64 * tw64 / th64).max(1) as u32, h)
    } else if w64 * th64 < tw64 * h64 {
        (w, (w64 * th64 / tw64).max(1) as u32)
    } else {
        (w, h)
    }
}

/// The four window invariants, for a window cut from a `w`
/// by `h` image toward a `tw` by `th` target.
fn check_window(
    (w, h): (u32, u32),
    (tw, th): (u32, u32),
    window: (u32, u32, u32, u32),
) -> Result<(), TestCaseError> {
    let (x, y, cw, ch) = window;
    prop_assert!(cw >= 1 && ch >= 1, "empty window {:?}", window);
    prop_assert!(
        x as u64 + cw as u64 <= w as u64 && y as u64 + ch as u64 <= h as u64,
        "window {:?} leaves the {}x{} image",
        window,
        w,
        h
    );
    let (fill_w, fill_h) = fill_window(w, h, tw, th);
    prop_assert!(
        cw <= fill_w && ch <= fill_h,
        "window {:?} is larger than the {}x{} fill window",
        window,
        fill_w,
        fill_h
    );
    // Some exact tw:th rectangle (s*tw, s*th) lies within a
    // pixel of (cw, ch) on both axes exactly when
    // |cw*th - ch*tw| <= tw + th.
    let skew = (cw as i128 * th as i128 - ch as i128 * tw as i128).abs();
    prop_assert!(
        skew <= tw as i128 + th as i128,
        "window {:?} is off the {}:{} aspect by more than a pixel",
        window,
        tw,
        th
    );
    Ok(())
}

fn check_output(out: &DynamicImage, window: (u32, u32, u32, u32)) -> Result<(), TestCaseError> {
    prop_assert_eq!((out.width(), out.height()), (window.2, window.3));
    Ok(())
}

fn align() -> impl Strategy<Value = Align> {
    prop::sample::select(vec![
        Align::Center,
        Align::Smart,
        Align::Face,
        Align::Top,
        Align::Bottom,
        Align::Left,
        Align::Right,
        Align::TopLeft,
        Align::TopRight,
        Align::BottomLeft,
        Align::BottomRight,
        Align::Objects {
            classes: vec!["person".to_string()],
            scale: 1.0,
        },
    ])
}

/// An image size with the faces a provider might find in it.
fn image_with_faces() -> impl Strategy<Value = ((u32, u32), Option<Vec<FaceBox>>)> {
    image_size().prop_flat_map(|(w, h)| (Just((w, h)), proptest::option::of(faces_in(w, h))))
}

proptest! {
    #![proptest_config(config())]

    /// The plain aspect crop, every alignment: directional
    /// placement, the center, and smart placement on a
    /// declared point, with or without faces to keep whole.
    #[test]
    fn aspect_crop_windows(
        ((w, h), faces) in image_with_faces(),
        (tw, th) in target(),
        align in align(),
        px in loose_f32(0.0, 1.0),
        py in loose_f32(0.0, 1.0),
    ) {
        let img = DynamicImage::new_luma8(w, h);
        let faces = faces.map(Faces);
        let declared = Point(px, py);
        let (out, trace) = crop::crop_to_aspect_traced(
            img,
            tw,
            th,
            align,
            faces.as_ref().map(|f| f as &dyn FaceProvider),
            Some(&declared),
        );
        let trace = trace.expect("a non-zero target always crops");
        check_window((w, h), (tw, th), trace.window)?;
        check_output(&out, trace.window)?;
        prop_assert!(trace.faces_kept <= trace.faces_detected);
    }

    /// The `focal:` crop, for any rectangle: inside the
    /// frame, past it, reversed. A rectangle that fits the
    /// window (once clamped to the frame) stays inside it.
    #[test]
    fn focal_rectangle_crop_windows(
        (w, h) in image_size(),
        (tw, th) in target(),
        rect in prop_oneof![
            (0u32..=4200, 0u32..=4200, 0u32..=4200, 0u32..=4200),
            any::<(u32, u32, u32, u32)>(),
        ],
    ) {
        let img = DynamicImage::new_luma8(w, h);
        let (out, window) = crop::crop_to_aspect_focal_traced(img, tw, th, rect);
        let window = window.expect("a non-zero target always crops");
        check_window((w, h), (tw, th), window)?;
        check_output(&out, window)?;
        let (x, y, cw, ch) = window;
        let (left, top, right, bottom) = rect;
        for (start, end, at, len, extent) in [(left, right, x, cw, w), (top, bottom, y, ch, h)] {
            let start = start.min(extent);
            let end = end.min(extent).max(start);
            if end - start <= len {
                prop_assert!(
                    at <= start && end <= at + len,
                    "rectangle {:?} fits the window {:?} but is cut",
                    rect,
                    window
                );
            }
        }
    }

    /// The `focalpoint:` crop, for any point and zoom.
    #[test]
    fn focal_point_crop_windows(
        (w, h) in image_size(),
        (tw, th) in target(),
        x in loose_f32(0.0, 1.0),
        y in loose_f32(0.0, 1.0),
        zoom in loose_f32(0.0, 100.0),
    ) {
        let img = DynamicImage::new_luma8(w, h);
        let (out, window) = crop::crop_to_aspect_point_traced(img, tw, th, (x, y, zoom));
        let window = window.expect("a non-zero target always crops");
        check_window((w, h), (tw, th), window)?;
        check_output(&out, window)?;
    }

    /// The `face` crop around a detected face, at any padding.
    #[test]
    fn face_crop_windows(
        ((w, h), faces) in image_size()
            .prop_flat_map(|(w, h)| (Just((w, h)), faces_in(w, h).prop_filter("a face", |f| !f.is_empty()))),
        (tw, th) in target(),
        padding in loose_f32(0.0, 3.0),
    ) {
        let img = DynamicImage::new_luma8(w, h);
        let (out, window) = crop::crop_around_face_traced(img, tw, th, &faces[0], padding);
        let window = window.expect("a non-empty image always crops");
        check_window((w, h), (tw, th), window)?;
        check_output(&out, window)?;
    }

    /// The `objects` crop around a detection union, at any
    /// scale, the union anywhere (it comes from detector
    /// boxes, which a crop must survive even past the frame).
    #[test]
    fn objects_crop_windows(
        (w, h) in image_size(),
        (tw, th) in target(),
        union in prop_oneof![
            (0u32..=4200, 0u32..=4200, 0u32..=4200, 0u32..=4200),
            any::<(u32, u32, u32, u32)>(),
        ],
        scale in loose_f32(0.0, 100.0),
    ) {
        let img = DynamicImage::new_luma8(w, h);
        let (out, window) = crop::crop_around_objects_traced(img, tw, th, union, scale);
        let window = window.expect("a non-empty image always crops");
        check_window((w, h), (tw, th), window)?;
        check_output(&out, window)?;
    }

    /// A zero target dimension crops nothing: the image comes
    /// back whole and no window is reported.
    #[test]
    fn zero_targets_cut_nothing(
        (w, h) in image_size(),
        (tw, th) in prop_oneof![
            (Just(0u32), 0u32..=MAX_DIMENSION),
            (0u32..=MAX_DIMENSION, Just(0u32)),
        ],
    ) {
        let (out, window) =
            crop::crop_to_aspect_focal_traced(DynamicImage::new_luma8(w, h), tw, th, (0, 0, w, h));
        prop_assert_eq!((window, out.width(), out.height()), (None, w, h));
        let (out, window) =
            crop::crop_to_aspect_point_traced(DynamicImage::new_luma8(w, h), tw, th, (0.5, 0.5, 1.0));
        prop_assert_eq!((window, out.width(), out.height()), (None, w, h));
        let (out, trace) = crop::crop_to_aspect_traced(
            DynamicImage::new_luma8(w, h),
            tw,
            th,
            Align::Center,
            None,
            None,
        );
        prop_assert_eq!((trace.is_none(), out.width(), out.height()), (true, w, h));
    }
}

/// What these properties found (issue #478), kept as plain
/// cases next to their seeds in
/// `proptest_crop.proptest-regressions`. A subject region the
/// frame clamps to a fractional fill size used to round up
/// one pixel past the fill window (19 wide where the fill
/// window is 18, 113 where it is 112).
#[test]
fn subject_regions_never_round_past_the_fill_window() {
    let (_, window) = crop::crop_around_objects_traced(
        DynamicImage::new_luma8(40, 41),
        7393,
        16383,
        (0, 0, 0, 4_918_054),
        0.0,
    );
    assert_eq!(window, Some((0, 0, 18, 41)));
    let face = FaceBox {
        x: 0,
        y: 1,
        w: 37,
        h: 57,
        score: 0.417291,
    };
    let (_, window) = crop::crop_around_face_traced(
        DynamicImage::new_luma8(374, 339),
        4271,
        12855,
        &face,
        1.260_244_6,
    );
    assert_eq!(window, Some((0, 0, 112, 339)));
}

/// A zoomed point crop used to divide the whole-pixel fill
/// window (2 wide, rounded down from 2.7) by the zoom and
/// round again, cutting 1x821 where the 40:16383 aspect asks
/// for 2 pixels across 821.
#[test]
fn zoomed_point_windows_keep_the_target_aspect() {
    let (_, window) = crop::crop_to_aspect_point_traced(
        DynamicImage::new_luma8(3, 1106),
        40,
        16383,
        (0.0, 0.0, 1.347_743),
    );
    assert_eq!(window, Some((0, 0, 2, 821)));
}
