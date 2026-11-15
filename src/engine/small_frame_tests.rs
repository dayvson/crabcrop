//! Small frames with an alpha channel through every still
//! encoder (issue #670). JPEG XL output of a thumbnail cut
//! from a transparent image answered 500: the encoder
//! panicked on some frames between 8x8 and 25x25 and on thin
//! strips. Each frame here is encoded as a request would
//! encode it and decoded back, so an encoder that fails on a
//! small frame, or writes one that does not decode to the
//! picture it was given, fails a test instead of a request.
//!
//! The frames are in `tests/fixtures/small-alpha/`: the
//! functional corpus's cut-out (`crab-cutout.png`, a subject
//! on a transparent background, its edge feathered) as the
//! server renders it for `/unsafe/WxH/filters:format(png)/`,
//! one file per size. They are the pixels a request for that
//! size hands its encoder, kept as files because decoding
//! and resizing the 1000x667 cut-out for every size takes
//! seconds in a debug build.

use super::*;
use crate::url::Quality;

/// The cut-out at `width` x `height`, from the fixtures.
fn cutout_frame(width: u32, height: u32) -> DynamicImage {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/fixtures/small-alpha/{width}x{height}.png"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let frame = decode_image(&bytes).expect("the frame decodes");
    assert_eq!((frame.width(), frame.height()), (width, height));
    assert!(
        frame.color().has_alpha(),
        "{width}x{height} has an alpha channel"
    );
    frame
}

/// The sizes of the sweep over every format: squares from one
/// pixel up, around the sizes an encoder works in blocks of,
/// and the strips a thumbnail or an icon comes in.
fn sweep_sizes() -> Vec<(u32, u32)> {
    let mut sizes: Vec<(u32, u32)> = (1..=9).map(|side| (side, side)).collect();
    sizes.extend([
        (12, 12),
        (15, 15),
        (16, 16),
        (17, 17),
        (24, 24),
        (25, 25),
        (32, 32),
        (40, 40),
        (64, 64),
        (9, 17),
        (17, 9),
        (8, 16),
        (16, 8),
        (1, 33),
        (33, 1),
        (2, 64),
        (64, 2),
        (3, 5),
        (100, 7),
        (7, 100),
    ]);
    sizes
}

/// Encode `frame` as a request for `format` at `quality`
/// would.
fn encoded(
    frame: &DynamicImage,
    format: StillFormat,
    quality: Option<u8>,
) -> Result<Vec<u8>, EngineError> {
    let opts = EncodeOptions {
        quality: quality.map(Quality::Value),
        ..EncodeOptions::default()
    };
    Engine::encode_still(frame, format, &opts, &ProcessContext::default()).map(|out| out.bytes)
}

/// How far a decoded output is from the frame it was encoded
/// from. On the 0 to 255 scale: the mean and the largest
/// difference in alpha, and the mean difference in color as
/// a viewer composites it (each channel times its alpha, so
/// the color stored under clear pixels does not count). And
/// in pixels: how many of the frame's solid pixels (alpha
/// 128 and up) decoded solid, and how many of its clear ones
/// (alpha under 16) decoded under 128, each with its total.
#[derive(Debug)]
struct Drift {
    alpha_mean: f64,
    alpha_max: u8,
    color_mean: f64,
    solid_kept: (u32, u32),
    clear_kept: (u32, u32),
}

/// The drift of `decoded` from `frame`. `alpha_of` maps a
/// frame's alpha to the alpha its format can hold.
fn drift(frame: &DynamicImage, decoded: &DynamicImage, alpha_of: impl Fn(u8) -> u8) -> Drift {
    let (frame, decoded) = (frame.to_rgba8(), decoded.to_rgba8());
    assert_eq!(frame.dimensions(), decoded.dimensions());
    let (mut alpha_sum, mut alpha_max, mut color_sum) = (0.0, 0, 0.0);
    let (mut solid_kept, mut clear_kept) = ((0, 0), (0, 0));
    for (want, got) in frame.pixels().zip(decoded.pixels()) {
        let want_alpha = alpha_of(want[3]);
        let alpha = want_alpha.abs_diff(got[3]);
        alpha_sum += f64::from(alpha);
        alpha_max = alpha_max.max(alpha);
        if want_alpha >= 128 {
            solid_kept.1 += 1;
            solid_kept.0 += u32::from(got[3] >= 128);
        }
        if want_alpha < 16 {
            clear_kept.1 += 1;
            clear_kept.0 += u32::from(got[3] < 128);
        }
        for channel in 0..3 {
            let want = f64::from(want[channel]) * f64::from(want_alpha) / 255.0;
            let got = f64::from(got[channel]) * f64::from(got[3]) / 255.0;
            color_sum += (want - got).abs();
        }
    }
    let pixels = f64::from(frame.width() * frame.height());
    Drift {
        alpha_mean: alpha_sum / pixels,
        alpha_max,
        color_mean: color_sum / (3.0 * pixels),
        solid_kept,
        clear_kept,
    }
}

/// The alpha a format can hold for a frame's alpha `a`.
fn held_alpha(format: StillFormat, a: u8) -> u8 {
    match format {
        // No alpha channel: the stored color is served opaque.
        StillFormat::Jpeg => 255,
        // One bit: anything but clear is opaque.
        StillFormat::Gif => {
            if a == 0 {
                0
            } else {
                255
            }
        }
        _ => a,
    }
}

/// The lowest JPEG XL quality at which every frame of the
/// fixtures keeps its subject. The encoder codes alpha as
/// lossily as the quality asks, and below this a frame as
/// small as these can lose its alpha altogether and decode
/// fully transparent: 8x16 and 16x8 do at quality 10, and
/// the squares from 12x12 to 16x16 at quality 5. Release
/// 0.5.5 of the encoder did the same on the frames it could
/// encode. Under this quality a frame is held to encoding
/// and decoding at its size, and to nothing about its
/// pixels.
const LOWEST_QUALITY_THAT_KEEPS_THE_SUBJECT: u8 = 15;

/// How far an output may sit from its frame: the largest
/// mean alpha difference and the largest mean color
/// difference, on the 0 to 255 scale. Each bound is about
/// half as much again as the worst frame of the fixtures
/// measures. The lossy encoders code alpha lossily too, so
/// the JPEG XL bounds follow the quality. `None` under
/// [`LOWEST_QUALITY_THAT_KEEPS_THE_SUBJECT`].
fn tolerance(format: StillFormat, quality: Option<u8>) -> Option<(f64, f64)> {
    let quality = quality.unwrap_or(80);
    Some(match format {
        StillFormat::Png => (0.0, 0.0),
        StillFormat::Gif => (0.0, 2.0),
        StillFormat::Jpeg => (0.0, 26.0),
        StillFormat::WebP if quality >= 100 => (0.0, 0.0),
        StillFormat::WebP => (0.0, 5.0),
        StillFormat::Avif => (10.0, 9.0),
        StillFormat::Jxl => match quality {
            100.. => (1.0, 2.0),
            95..=99 => (4.0, 6.0),
            80..=94 => (16.0, 12.0),
            50..=79 => (34.0, 21.0),
            LOWEST_QUALITY_THAT_KEEPS_THE_SUBJECT..=49 => (45.0, 25.0),
            _ => return None,
        },
    })
}

/// Hold `bytes`, the output of `frame` in `format` at
/// `quality`, to the frame: its size, an alpha channel where
/// the format has one, the format's [`tolerance`], and its
/// subject, of which at least half the solid pixels decode
/// solid and nine in ten of the clear ones stay clear (the
/// worst frame of the fixtures keeps seven in ten and
/// ninety-seven in a hundred).
fn assert_decodes_to(frame: &DynamicImage, bytes: &[u8], format: StillFormat, quality: Option<u8>) {
    let (width, height) = (frame.width(), frame.height());
    let what = format!("{format:?} {width}x{height} at quality {quality:?}");
    // AVIF sources decode only in builds with the
    // avif-decode feature; without it the container's own
    // size is what there is to check.
    #[cfg(not(feature = "avif-decode"))]
    if format == StillFormat::Avif {
        assert_eq!(
            crate::avif::declared_size(bytes),
            Some((width, height)),
            "{what}"
        );
        return;
    }
    let decoded = decode_image(bytes).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(
        (decoded.width(), decoded.height()),
        (width, height),
        "{what}"
    );
    assert_eq!(
        decoded.color().has_alpha(),
        format != StillFormat::Jpeg,
        "{what}: decoded as {:?}",
        decoded.color()
    );
    let Some((alpha, color)) = tolerance(format, quality) else {
        return;
    };
    let drift = drift(frame, &decoded, |alpha| held_alpha(format, alpha));
    assert!(
        drift.alpha_mean <= alpha && drift.color_mean <= color,
        "{what}: {drift:?} against a tolerance of {alpha} in alpha and {color} in color"
    );
    if (alpha, color) == (0.0, 0.0) {
        assert_eq!(drift.alpha_max, 0, "{what}");
    }
    let (solid, solid_total) = drift.solid_kept;
    let (clear, clear_total) = drift.clear_kept;
    assert!(
        2 * solid >= solid_total && 10 * clear >= 9 * clear_total,
        "{what}: the subject is lost, {drift:?}"
    );
}

/// Encode `frame` to `format` at `quality` and hold the
/// output to the frame (see [`assert_decodes_to`]).
fn assert_round_trip(frame: &DynamicImage, format: StillFormat, quality: Option<u8>) {
    let bytes = encoded(frame, format, quality).unwrap_or_else(|e| {
        panic!(
            "{format:?} {}x{} at quality {quality:?}: {e}",
            frame.width(),
            frame.height()
        )
    });
    assert_decodes_to(frame, &bytes, format, quality);
}

#[test]
fn jpeg_xl_encodes_the_small_frames_with_alpha_that_failed() {
    // Every request of the report's replay that answered
    // 500: the cut-out at 56 sizes and five qualities (the
    // default, 10, 50, 95 and 100), of which these 69 failed.
    let failed: [(u32, u32, &[Option<u8>]); 22] = [
        (1, 33, &[None, Some(50), Some(95), Some(100)]),
        (2, 64, &[None, Some(50), Some(95), Some(100)]),
        (8, 8, &[None, Some(95), Some(100)]),
        (8, 16, &[None, Some(10), Some(50), Some(95), Some(100)]),
        (9, 9, &[None, Some(50), Some(95)]),
        (9, 17, &[None, Some(95)]),
        (10, 10, &[None, Some(10), Some(50), Some(95)]),
        (11, 11, &[None, Some(10), Some(50), Some(95)]),
        (12, 12, &[None, Some(50), Some(95)]),
        (13, 13, &[None, Some(10), Some(50), Some(95)]),
        (14, 14, &[None, Some(10), Some(50), Some(95), Some(100)]),
        (15, 15, &[None, Some(10), Some(50), Some(95), Some(100)]),
        (16, 8, &[None, Some(10), Some(50), Some(95)]),
        (16, 16, &[None, Some(10), Some(50), Some(95)]),
        (17, 9, &[None, Some(10), Some(50), Some(95)]),
        (17, 17, &[Some(95)]),
        (18, 18, &[None]),
        (21, 21, &[Some(50), Some(95)]),
        (24, 24, &[None]),
        (25, 25, &[Some(95)]),
        (33, 1, &[Some(95)]),
        (64, 2, &[None, Some(50), Some(95), Some(100)]),
    ];
    let mut requests = 0;
    for (width, height, qualities) in failed {
        let frame = cutout_frame(width, height);
        for &quality in qualities {
            assert_round_trip(&frame, StillFormat::Jxl, quality);
            requests += 1;
        }
    }
    assert_eq!(requests, 69);
}

#[test]
fn jpeg_xl_encodes_small_frames_with_alpha_at_every_quality_step() {
    // Sizes among the ones that failed and around them,
    // each from the lowest quality to the highest, and
    // through the `quality(auto)` search, which encodes and
    // decodes several candidates of its own.
    for (width, height) in [(7, 7), (8, 8), (16, 16), (8, 16), (2, 64)] {
        let frame = cutout_frame(width, height);
        for quality in [1, 10, 15, 35, 50, 65, 80, 95, 100] {
            assert_round_trip(&frame, StillFormat::Jxl, Some(quality));
        }
        let auto = EncodeOptions {
            quality: Some(Quality::Auto),
            ..EncodeOptions::default()
        };
        let out = Engine::encode_still(&frame, StillFormat::Jxl, &auto, &ProcessContext::default())
            .unwrap_or_else(|e| panic!("{width}x{height} at quality(auto): {e}"));
        let chosen = out.chosen_quality.expect("the search reports its quality");
        assert!((50..=95).contains(&chosen), "{width}x{height}: {chosen}");
        assert_decodes_to(&frame, &out.bytes, StillFormat::Jxl, Some(chosen));
    }
}

#[test]
fn every_format_encodes_small_frames_with_alpha() {
    // One pass over every still encoder at its default
    // quality: an encoder that fails on a small frame with
    // transparency, or writes one that does not decode back
    // to it, fails here, whichever format it is.
    let frames: Vec<DynamicImage> = sweep_sizes()
        .into_iter()
        .map(|(width, height)| cutout_frame(width, height))
        .collect();
    for format in [
        StillFormat::Jpeg,
        StillFormat::Png,
        StillFormat::WebP,
        StillFormat::Avif,
        StillFormat::Gif,
        StillFormat::Jxl,
    ] {
        for frame in &frames {
            assert_round_trip(frame, format, None);
        }
    }
    // WebP's lossless encode, which quality 100 selects.
    for frame in &frames {
        assert_round_trip(frame, StillFormat::WebP, Some(100));
    }
}
