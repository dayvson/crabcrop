//! `quality(auto)` verifies its pick for outputs of up to
//! [`crate::quality::MAX_VERIFIED_PIXELS`]; a larger output is
//! encoded once at the pick.
//!
//! Most tests run the encode stage with a bound of 64x64
//! pixels in place of the real one, so the frames on both
//! sides of it stay small enough to encode, decode and score
//! in a debug build. The tests of the real bound use flat
//! frames and never score one.

use super::*;
use crate::quality::MAX_VERIFIED_PIXELS;
use crate::url::FormatOptions;
use std::cell::Cell;

/// The bound the tests hand the encode stage: a 64x64 frame.
const BOUND: u64 = 64 * 64;

/// A frame with soft detail, the kind of content the search
/// moves the quality for.
fn frame(width: u32, height: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_fn(width, height, |x, y| {
        let wave = |v: u32, period: f64| ((f64::from(v) / period).sin() * 60.0 + 128.0) as u32;
        let noise = (x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77)) >> 28;
        image::Rgb([
            (wave(x, 5.0) + noise) as u8,
            (wave(y, 7.0) + noise) as u8,
            (wave(x + y, 11.0) + noise) as u8,
        ])
    }))
}

/// The encode options the filters of `filters` collect.
fn options(filters: &str) -> EncodeOptions {
    let remainder = format!("filters:{filters}/example.com/a.png");
    let (ops, _) = crate::url::parse_operations(&remainder).expect("the filters parse");
    let directives = crate::url::RequestDirectives::analyze(&ops);
    let ctx = ProcessContext::default();
    apply_operations(frame(4, 4), &ops, None, 1.0, &ctx, &directives, None)
        .expect("the filters apply")
        .encode_opts
}

/// What the pipeline scores a candidate with.
fn scored(img: &DynamicImage, candidate: &[u8]) -> Option<f64> {
    decode_image(candidate)
        .ok()
        .and_then(|decoded| crate::quality::ssimulacra2(img, &decoded))
}

/// The encode stage under [`BOUND`], counting the candidates
/// it scores.
fn encode_counting(
    img: &DynamicImage,
    format: StillFormat,
    opts: &EncodeOptions,
    ctx: &ProcessContext<'_>,
) -> (StillEncode, usize) {
    let calls = Cell::new(0);
    let out = Engine::encode_still_within(img, format, opts, ctx, BOUND, |candidate| {
        calls.set(calls.get() + 1);
        scored(img, candidate)
    })
    .expect("the frame encodes");
    (out, calls.get())
}

/// The encode stage under [`BOUND`] for a frame past it: a
/// candidate that reaches the scorer, and so a decoder,
/// fails the test.
fn encode_unscored(
    img: &DynamicImage,
    format: StillFormat,
    opts: &EncodeOptions,
    ctx: &ProcessContext<'_>,
) -> StillEncode {
    Engine::encode_still_within(img, format, opts, ctx, BOUND, |_| {
        panic!("an output past the bound is neither decoded nor scored")
    })
    .expect("the frame encodes")
}

/// One encode of `img` at `quality`, with nothing else in
/// force.
fn plain(img: &DynamicImage, format: StillFormat, quality: u8) -> Vec<u8> {
    let ctx = ProcessContext::default();
    encode(
        &EncodeBuffers::new(img),
        format,
        Some(quality),
        None,
        None,
        FormatOptions::default(),
        ctx.avif_speed,
        ctx.jxl_speed,
    )
    .expect("the frame encodes")
}

#[test]
fn the_bound_is_four_megapixels() {
    assert_eq!(MAX_VERIFIED_PIXELS, 4_194_304);
    assert_eq!(MAX_VERIFIED_PIXELS, 2048 * 2048);
}

#[test]
fn the_pick_is_verified_at_the_bound_and_under_it_and_not_past_it() {
    let ctx = ProcessContext::default();
    let auto = options("quality(auto)");
    for format in [StillFormat::Jpeg, StillFormat::Jxl] {
        // Exactly at the bound, in two shapes, one pixel
        // under it, and well under it: the search runs and
        // reports its score.
        for (width, height) in [(64, 64), (128, 32), (63, 65), (16, 16)] {
            let img = frame(width, height);
            assert!(u64::from(width * height) <= BOUND);
            let (out, scored) = encode_counting(&img, format, &auto, &ctx);
            let what = format!("{format:?} {width}x{height}");
            assert!(scored >= 1, "{what}: {scored} candidates scored");
            assert!(out.ssim.is_some(), "{what}");
            assert!(
                out.chosen_quality.is_some_and(|q| (50..=95).contains(&q)),
                "{what}: {:?}",
                out.chosen_quality
            );
        }
        // One pixel past the bound, in two shapes, and well
        // past it: one encode at the heuristic's pick, which
        // is reported, and no score.
        for (width, height) in [(17, 241), (241, 17), (65, 64), (96, 80)] {
            let img = frame(width, height);
            assert!(u64::from(width * height) > BOUND);
            let pick = crate::quality::auto_quality(&img);
            let out = encode_unscored(&img, format, &auto, &ctx);
            let what = format!("{format:?} {width}x{height}");
            assert_eq!(out.chosen_quality, Some(pick), "{what}");
            assert_eq!(out.ssim, None, "{what}");
            assert!(out.bytes == plain(&img, format, pick), "{what}");
        }
    }
    // 17 x 241 is the test bound and one pixel, 63 x 65 the
    // bound less one.
    assert_eq!((17 * 241, 63 * 65), (BOUND + 1, BOUND - 1));
}

#[test]
fn an_output_past_the_bound_is_reported_like_a_format_with_no_search() {
    // WebP and AVIF have never searched: `quality(auto)`
    // encodes them once at the pick and reports the pick and
    // no score. A JPEG or JPEG XL output past the bound is
    // reported the same way.
    let ctx = ProcessContext::default();
    let auto = options("quality(auto)");
    let img = frame(96, 80);
    let pick = crate::quality::auto_quality(&img);
    for format in [StillFormat::WebP, StillFormat::Jpeg, StillFormat::Jxl] {
        let out = encode_unscored(&img, format, &auto, &ctx);
        assert_eq!(
            (out.chosen_quality, out.ssim),
            (Some(pick), None),
            "{format:?}"
        );
    }
}

#[test]
fn the_bound_holds_after_a_max_bytes_ladder() {
    let ctx = ProcessContext::default();
    for format in [StillFormat::Jpeg, StillFormat::Jxl] {
        // Past the bound: the ladder starts from the pick and
        // steps down as it does from a quality the URL names,
        // and its result is not scored either. A budget one
        // byte under the first encode is met at the first
        // step, with encodes to spare for a score.
        let img = frame(96, 80);
        let pick = crate::quality::auto_quality(&img);
        let budget = plain(&img, format, pick).len() - 1;
        let auto = options(&format!("quality(auto):max_bytes({budget})"));
        let named = options(&format!("quality({pick}):max_bytes({budget})"));
        let out = encode_unscored(&img, format, &auto, &ctx);
        let reference = encode_unscored(&img, format, &named, &ctx);
        assert!(out.bytes == reference.bytes, "{format:?}");
        assert_eq!(out.chosen_quality, reference.chosen_quality, "{format:?}");
        assert!(
            out.chosen_quality.is_some_and(|q| q < pick),
            "{format:?}: the ladder ran from {pick} to {:?}",
            out.chosen_quality
        );
        assert_eq!(out.ssim, None, "{format:?}");
        // At the bound the search runs first, and the bytes
        // the ladder ends on are scored once more: the score
        // reported is theirs.
        let img = frame(64, 64);
        let budget = plain(&img, format, 20).len();
        let auto = options(&format!("quality(auto):max_bytes({budget})"));
        let (out, candidates) = encode_counting(&img, format, &auto, &ctx);
        assert!(
            candidates >= 2,
            "{format:?}: {candidates} candidates scored"
        );
        assert!(
            out.chosen_quality.is_some_and(|q| q < 50) && out.bytes.len() <= budget,
            "{format:?}: {:?} in {} bytes for a budget of {budget}",
            out.chosen_quality,
            out.bytes.len()
        );
        assert!(out.ssim.is_some(), "{format:?}");
        assert_eq!(out.ssim, scored(&img, &out.bytes), "{format:?}");
    }
}

#[test]
fn save_data_and_per_format_qualities_meet_the_same_bound() {
    let img = frame(96, 80);
    let small = frame(64, 64);
    let pick = crate::quality::auto_quality(&img);
    let saving = ProcessContext {
        save_data: true,
        ..ProcessContext::default()
    };
    let ctx = ProcessContext::default();
    for format in [StillFormat::Jpeg, StillFormat::Jxl] {
        // Save-Data leaves a quality the URL names alone,
        // `auto` included, so the bound decides as without it.
        let auto = options("quality(auto)");
        let out = encode_unscored(&img, format, &auto, &saving);
        assert_eq!((out.chosen_quality, out.ssim), (Some(pick), None));
        assert!(out.bytes == plain(&img, format, pick), "{format:?}");
        let (out, scored) = encode_counting(&small, format, &auto, &saving);
        assert!(scored >= 1 && out.ssim.is_some(), "{format:?}");
        // With no quality named, Save-Data steps the default
        // down and nothing is scored on either side.
        let unnamed = EncodeOptions::default();
        for frame in [&img, &small] {
            let out = encode_unscored(frame, format, &unnamed, &saving);
            assert_eq!((out.chosen_quality, out.ssim), (Some(70), None));
        }
    }
    // A per-format list resolves once the format is settled:
    // `auto` for this format meets the bound, a number for
    // this format never searched.
    let listed = options("quality(jpeg:auto,jxl:60)");
    let out = encode_unscored(&img, StillFormat::Jpeg, &listed, &ctx);
    assert_eq!((out.chosen_quality, out.ssim), (Some(pick), None));
    let (out, scored) = encode_counting(&small, StillFormat::Jpeg, &listed, &ctx);
    assert!(scored >= 1 && out.ssim.is_some());
    for frame in [&img, &small] {
        let out = encode_unscored(frame, StillFormat::Jxl, &listed, &ctx);
        assert_eq!((out.chosen_quality, out.ssim), (None, None));
        assert!(out.bytes == plain(frame, StillFormat::Jxl, 60));
    }
    // The list's fallback is `auto` for the formats it does
    // not name.
    let fallback = options("quality(auto,jpeg:60)");
    let out = encode_unscored(&img, StillFormat::Jxl, &fallback, &ctx);
    assert_eq!((out.chosen_quality, out.ssim), (Some(pick), None));
    let (out, scored) = encode_counting(&small, StillFormat::Jxl, &fallback, &ctx);
    assert!(scored >= 1 && out.ssim.is_some());
}

#[test]
fn the_request_path_verifies_under_the_real_bound() {
    // `encode_still` is the stage a request runs: the real
    // bound, the real decode and score.
    let ctx = ProcessContext::default();
    let auto = options("quality(auto)");
    let img = frame(64, 64);
    for format in [StillFormat::Jpeg, StillFormat::Jxl] {
        let asked = crate::quality::counters::asked();
        let out = Engine::encode_still(&img, format, &auto, &ctx).unwrap();
        assert!(out.ssim.is_some(), "{format:?}");
        assert!(crate::quality::counters::asked() > asked, "{format:?}");
        let same =
            Engine::encode_still_within(&img, format, &auto, &ctx, MAX_VERIFIED_PIXELS, |bytes| {
                scored(&img, bytes)
            })
            .unwrap();
        assert!(out.bytes == same.bytes, "{format:?}");
        assert_eq!(out.chosen_quality, same.chosen_quality, "{format:?}");
        assert_eq!(out.ssim, same.ssim, "{format:?}");
    }
}

#[test]
fn the_real_bound_is_met_exactly() {
    // Flat frames at the real bound, one pixel under it and
    // one pixel past it, with a stand-in for the score: the
    // comparison is the one the request path makes, and
    // nothing this size is decoded or scored in the test.
    let ctx = ProcessContext::default();
    let auto = options("quality(auto)");
    for (width, height, verified) in [(2048, 2048, true), (2047, 2049, true), (1985, 2113, false)] {
        let img = DynamicImage::new_rgb8(width, height);
        let pixels = u64::from(width) * u64::from(height);
        assert_eq!(pixels <= MAX_VERIFIED_PIXELS, verified);
        let calls = Cell::new(0);
        let out = Engine::encode_still_within(
            &img,
            StillFormat::Jpeg,
            &auto,
            &ctx,
            MAX_VERIFIED_PIXELS,
            |_| {
                calls.set(calls.get() + 1);
                Some(80.0)
            },
        )
        .unwrap();
        let what = format!("{width}x{height}");
        // A flat frame's pick is the lowest quality the
        // heuristic gives, and a score of 80 accepts it.
        assert_eq!(out.chosen_quality, Some(50), "{what}");
        assert_eq!(calls.get(), usize::from(verified), "{what}");
        assert_eq!(out.ssim, verified.then_some(80.0), "{what}");
    }
    // 2047 x 2049 is the bound less one pixel, 1985 x 2113
    // the bound and one.
    assert_eq!(2047 * 2049, MAX_VERIFIED_PIXELS - 1);
    assert_eq!(1985 * 2113, MAX_VERIFIED_PIXELS + 1);
}

#[test]
fn the_request_path_asks_for_no_score_past_the_real_bound() {
    // The stage a request runs, on a frame one pixel past
    // the real bound: it reports the pick and no score, and
    // the scorer is never asked, so no candidate was decoded
    // for it.
    let ctx = ProcessContext::default();
    let auto = options("quality(auto)");
    let img = DynamicImage::new_rgb8(1985, 2113);
    let asked = crate::quality::counters::asked();
    let out = Engine::encode_still(&img, StillFormat::Jpeg, &auto, &ctx).unwrap();
    assert_eq!((out.chosen_quality, out.ssim), (Some(50), None));
    assert_eq!(crate::quality::counters::asked(), asked);
    assert!(out.bytes == plain(&img, StillFormat::Jpeg, 50));
}

#[test]
fn nothing_past_the_bound_is_scored() {
    // The metric's own entries refuse a pair past the bound
    // before they read a pixel, whoever calls them: one pixel
    // past, in two shapes, and far past.
    let references = crate::quality::counters::snapshot;
    let before = references();
    for (width, height) in [(1985, 2113), (2049, 2048), (4096, 1025), (4000, 4000)] {
        assert!(u64::from(width) * u64::from(height) > MAX_VERIFIED_PIXELS);
        let image = DynamicImage::new_rgb8(width, height);
        let what = format!("{width}x{height}");
        assert_eq!(crate::quality::ssimulacra2(&image, &image), None, "{what}");
        let reference = crate::quality::SsimReference::new(&image);
        assert_eq!(reference.score(&image), None, "{what}");
    }
    // No reference was prepared for any of them.
    assert_eq!(references(), before);
}
