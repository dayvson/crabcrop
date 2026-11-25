//! The scaled JPEG decode (issue #653): which requests take
//! it and at which scale; that every size a request works out
//! is the one the full-size path works out; that the decode
//! limits stand ahead of the decode; and that a file
//! libjpeg-turbo refuses, or warns about, gets the answer it
//! always got.
//!
//! "The full-size path" is the same engine with the plan
//! switched off on the test's thread ([`at_full_size`]): every
//! request as it rendered before the scaled decode existed.

use super::super::{
    ChainState, Decoded, Engine, EngineError, ProcessContext, ProcessedImage, decode_image,
    decode_watermarks, dpr_factor, probe_source,
};
use super::*;
use crate::url::{ImageFormat, RequestDirectives};

/// A picture with smooth and busy parts and no symmetry, the
/// same on every run: a turned or mirrored copy of it is a
/// different picture.
fn picture(width: u32, height: u32) -> image::RgbImage {
    image::RgbImage::from_fn(width, height, |x, y| {
        let mut h = x
            .wrapping_mul(0x9E37_79B1)
            .wrapping_add(y.wrapping_mul(0x85EB_CA77));
        h ^= h >> 15;
        h = h.wrapping_mul(0x2C1B_3C6D);
        h ^= h >> 12;
        // Texture on the left third only, so the picture has
        // detail a reduction must average and areas it must
        // leave alone.
        let grain = if x * 3 < width { h % 48 } else { 0 };
        let band = if (x / 24 + y / 16).is_multiple_of(2) {
            40
        } else {
            0
        };
        image::Rgb([
            (x * 200 / width.max(1) + grain) as u8,
            (y * 200 / height.max(1) + band) as u8,
            (220 - (x + y) * 180 / (width + height).max(1)) as u8,
        ])
    })
}

/// `img` as a JPEG from libjpeg-turbo, quality 90 and 4:2:0
/// unless `set` says otherwise.
fn jpeg_with(img: &image::RgbImage, set: impl FnOnce(&mut turbojpeg::Compressor)) -> Vec<u8> {
    let mut compressor = turbojpeg::Compressor::new().unwrap();
    compressor.set_quality(90).unwrap();
    compressor.set_subsamp(turbojpeg::Subsamp::Sub2x2).unwrap();
    set(&mut compressor);
    compressor
        .compress_to_vec(turbojpeg::Image {
            pixels: img.as_raw().as_slice(),
            width: img.width() as usize,
            pitch: img.width() as usize * 3,
            height: img.height() as usize,
            format: turbojpeg::PixelFormat::RGB,
        })
        .unwrap()
}

fn jpeg(img: &image::RgbImage) -> Vec<u8> {
    jpeg_with(img, |_| {})
}

fn progressive_jpeg(img: &image::RgbImage) -> Vec<u8> {
    jpeg_with(img, |c| c.set_progressive(true).unwrap())
}

/// A four-component JPEG: CMYK as Photoshop writes it, or
/// YCCK, its colour-transformed form.
fn four_component_jpeg(width: u32, height: u32, colorspace: turbojpeg::Colorspace) -> Vec<u8> {
    let pixels: Vec<u8> = picture(width, height)
        .pixels()
        .flat_map(|p| [p[0], p[1], p[2], 255 - p[0] / 2])
        .collect();
    let mut compressor = turbojpeg::Compressor::new().unwrap();
    compressor.set_quality(90).unwrap();
    compressor.set_subsamp(turbojpeg::Subsamp::None).unwrap();
    compressor.set_colorspace(colorspace).unwrap();
    compressor
        .compress_to_vec(turbojpeg::Image {
            pixels: pixels.as_slice(),
            width: width as usize,
            pitch: width as usize * 4,
            height: height as usize,
            format: turbojpeg::PixelFormat::CMYK,
        })
        .unwrap()
}

/// The segments of a JPEG up to its first scan, as (marker,
/// offset of the marker, offset past the segment).
fn header_segments(bytes: &[u8]) -> Vec<(u8, usize, usize)> {
    assert_eq!(&bytes[..2], [0xFF, 0xD8], "no SOI");
    let mut segments = Vec::new();
    let mut at = 2;
    loop {
        assert_eq!(bytes[at], 0xFF, "no marker at offset {at}");
        let marker = bytes[at + 1];
        let len = usize::from(u16::from_be_bytes([bytes[at + 2], bytes[at + 3]]));
        segments.push((marker, at, at + 2 + len));
        if marker == 0xDA {
            return segments;
        }
        at += 2 + len;
    }
}

/// Where the frame header's payload starts: the sample
/// precision, then the height and the width, two bytes each.
fn frame_header(bytes: &[u8]) -> usize {
    header_segments(bytes)
        .into_iter()
        .find(|(marker, ..)| matches!(marker, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB))
        .expect("a frame header")
        .1
        + 4
}

/// `bytes` with another size written in its frame header: a
/// file that declares one size and carries another.
fn declaring(bytes: &[u8], width: u16, height: u16) -> Vec<u8> {
    let at = frame_header(bytes);
    let mut out = bytes.to_vec();
    out[at + 1..at + 3].copy_from_slice(&height.to_be_bytes());
    out[at + 3..at + 5].copy_from_slice(&width.to_be_bytes());
    out
}

/// `bytes` with `segment` (a whole marker segment) placed
/// right after SOI.
fn with_segment(bytes: &[u8], marker: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = bytes[..2].to_vec();
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&bytes[2..]);
    out
}

/// `bytes` with an EXIF segment that carries `orientation`,
/// as a phone camera writes it.
fn with_orientation(bytes: &[u8], orientation: u16) -> Vec<u8> {
    let mut exif = b"Exif\0\0II*\0".to_vec();
    exif.extend_from_slice(&8u32.to_le_bytes());
    exif.extend_from_slice(&1u16.to_le_bytes());
    exif.extend_from_slice(&0x0112u16.to_le_bytes());
    exif.extend_from_slice(&3u16.to_le_bytes());
    exif.extend_from_slice(&1u32.to_le_bytes());
    exif.extend_from_slice(&orientation.to_le_bytes());
    exif.extend_from_slice(&[0, 0]);
    exif.extend_from_slice(&0u32.to_le_bytes());
    with_segment(bytes, 0xE1, &exif)
}

/// `bytes` with a big-endian (Motorola order) EXIF segment
/// whose first directory holds an orientation tag.
fn with_big_endian_orientation(bytes: &[u8], orientation: u16) -> Vec<u8> {
    let mut exif = b"Exif\0\0MM\0*".to_vec();
    exif.extend_from_slice(&8u32.to_be_bytes());
    exif.extend_from_slice(&1u16.to_be_bytes());
    exif.extend_from_slice(&0x0112u16.to_be_bytes());
    exif.extend_from_slice(&3u16.to_be_bytes());
    exif.extend_from_slice(&1u32.to_be_bytes());
    exif.extend_from_slice(&orientation.to_be_bytes());
    exif.extend_from_slice(&[0, 0]);
    exif.extend_from_slice(&0u32.to_be_bytes());
    with_segment(bytes, 0xE1, &exif)
}

/// The declared size of `bytes` where the first look and
/// both decoders take the file, as [`plan`] asks.
fn eligible(bytes: &[u8]) -> Option<(u32, u32)> {
    glance(bytes).and_then(|glance| eligible_size(bytes, &glance))
}

/// `bytes` with the Display P3 profile embedded.
fn with_display_p3(bytes: &[u8]) -> Vec<u8> {
    let icc = moxcms::ColorProfile::new_display_p3().encode().unwrap();
    crate::color::inject_jpeg(
        bytes,
        Some(&icc),
        &crate::color::SourceMeta::default(),
        false,
    )
}

fn ops_of(remainder: &str) -> Vec<Operation> {
    if remainder.is_empty() {
        return Vec::new();
    }
    crate::url::parse_operations(&format!("{remainder}/https://example.com/a.jpg"))
        .unwrap_or_else(|e| panic!("{remainder}: {e:?}"))
        .0
}

/// The plan for `ops` on `bytes` at a DPR factor, as the
/// decode stage asks for it.
fn plan_of(bytes: &[u8], ops: &[Operation], dpr: f32) -> Option<ScaledDecode> {
    let directives = RequestDirectives::analyze(ops);
    plan(bytes, ops, &directives, dpr)
}

/// The plan for `remainder` on `bytes`, at DPR factor 1.
fn plan_for(bytes: &[u8], remainder: &str) -> Option<ScaledDecode> {
    plan_of(bytes, &ops_of(remainder), 1.0)
}

/// The scale `remainder` gets on `bytes`: 2, 4 or 8, or
/// `None` for a full-size decode.
fn scale_for(bytes: &[u8], remainder: &str) -> Option<u32> {
    plan_for(bytes, remainder).map(|scaled| scaled.denominator)
}

/// Run `render` with the scaled decode switched off: the
/// full-size path every request took before it existed.
fn at_full_size<T>(render: impl FnOnce() -> T) -> T {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            FULL_SIZE_ONLY.set(false);
        }
    }
    FULL_SIZE_ONLY.set(true);
    let _restore = Restore;
    render()
}

fn render(bytes: &[u8], remainder: &str) -> Result<ProcessedImage, EngineError> {
    Engine.process(bytes, &ops_of(remainder))
}

/// What a render answered, in a form two renders compare by:
/// the output's sizes, the source's, the format and the
/// bytes, or the error.
type Answer = Result<((u32, u32), (u32, u32), ImageFormat, Vec<u8>), String>;

fn answer(result: Result<ProcessedImage, EngineError>) -> Answer {
    result
        .map(|out| {
            (
                (out.width, out.height),
                (out.source_width, out.source_height),
                out.format,
                out.bytes,
            )
        })
        .map_err(|e| format!("{e:?}"))
}

/// A render's output size and the source size it reports.
type Sizes = ((u32, u32), (u32, u32));

/// The sizes of an answer and its format, or its error: what
/// must be the same on both paths even where the pixels are
/// not.
fn sizes(answer: &Answer) -> Result<(Sizes, ImageFormat), String> {
    answer
        .as_ref()
        .map(|(output, source, format, _)| ((*output, *source), *format))
        .map_err(Clone::clone)
}

/// The decode stage as a render runs it: the frame the chain
/// starts from, and the full size it stands for when the
/// frame is a reduced one.
fn decode_stage(
    bytes: &[u8],
    ops: &[Operation],
    ctx: &ProcessContext<'_>,
) -> Result<(DynamicImage, Option<Reduced>), EngineError> {
    let directives = RequestDirectives::analyze(ops);
    let source = Engine::sniff_source(bytes)?;
    match Engine::decode_source(&source, ops, ctx, &directives, false)? {
        Decoded::Still(img, _, reduced) => Ok((img, reduced)),
        Decoded::Rendered(_) => panic!("a still source rendered as an animation"),
    }
}

/// A render up to its encode: the pixels the encoder would
/// get, and the source size the render reports.
fn chain(bytes: &[u8], ops: &[Operation]) -> Result<(DynamicImage, (u32, u32)), EngineError> {
    let ctx = ProcessContext::default();
    let (img, reduced) = decode_stage(bytes, ops, &ctx)?;
    let directives = RequestDirectives::analyze(ops);
    let (outcome, source) = Engine::still_chain(img, reduced, ops, &ctx, &directives)?;
    Ok((outcome.img, source))
}

/// A source with its full-size decode made once: the frame
/// every request starts from on the full-size path, whatever
/// its operations. Many requests can then be held against
/// that path for the cost of their chains alone.
struct Source {
    bytes: Vec<u8>,
    full: DynamicImage,
}

impl Source {
    fn new(bytes: Vec<u8>) -> Source {
        let ctx = ProcessContext::default();
        let (full, reduced) = at_full_size(|| decode_stage(&bytes, &[], &ctx)).unwrap();
        assert_eq!(reduced, None);
        Source { bytes, full }
    }

    /// The output size and the source size of `ops` on the
    /// path the request takes, and on the full-size path.
    fn sizes(&self, ops: &[Operation]) -> [Result<Sizes, String>; 2] {
        let ctx = ProcessContext::default();
        let directives = RequestDirectives::analyze(ops);
        let sizes = |result: Result<(DynamicImage, (u32, u32)), EngineError>| {
            result
                .map(|(img, source)| ((img.width(), img.height()), source))
                .map_err(|e| format!("{e:?}"))
        };
        [
            sizes(chain(&self.bytes, ops)),
            sizes(
                Engine::still_chain(self.full.clone(), None, ops, &ctx, &directives)
                    .map(|(outcome, source)| (outcome.img, source)),
            ),
        ]
    }
}

/// A render's pixels on the path the request takes, and at
/// full size.
fn both_pixels(bytes: &[u8], remainder: &str) -> (DynamicImage, DynamicImage) {
    let ops = ops_of(remainder);
    let ours = chain(bytes, &ops).unwrap_or_else(|e| panic!("{remainder}: {e:?}"));
    let theirs =
        at_full_size(|| chain(bytes, &ops)).unwrap_or_else(|e| panic!("{remainder}: {e:?}"));
    (ours.0, theirs.0)
}

/// Peak signal-to-noise ratio between two images of one
/// size, over their RGB samples, in decibels.
fn psnr(a: &DynamicImage, b: &DynamicImage) -> f64 {
    assert_eq!((a.width(), a.height()), (b.width(), b.height()));
    let (a, b) = (a.to_rgb8(), b.to_rgb8());
    let squares: f64 = a
        .as_raw()
        .iter()
        .zip(b.as_raw())
        .map(|(x, y)| f64::from(x.abs_diff(*y)).powi(2))
        .sum();
    if squares == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0 * 255.0 * a.as_raw().len() as f64 / squares).log10()
}

// The rule.

#[test]
fn a_side_rounds_up_as_libjpeg_rounds_it() {
    for (side, half, quarter, eighth) in [
        (1, 1, 1, 1),
        (2, 1, 1, 1),
        (3, 2, 1, 1),
        (7, 4, 2, 1),
        (8, 4, 2, 1),
        (9, 5, 3, 2),
        (17, 9, 5, 3),
        (2400, 1200, 600, 300),
        (2401, 1201, 601, 301),
        (16_384, 8192, 4096, 2048),
    ] {
        assert_eq!(scaled_side(side, 2), half, "{side} at 1/2");
        assert_eq!(scaled_side(side, 4), quarter, "{side} at 1/4");
        assert_eq!(scaled_side(side, 8), eighth, "{side} at 1/8");
    }
}

/// The scale a resize to `width` by `height` (0 for the
/// proportional side) gets on a stored image, by the rule
/// alone.
fn rule(stored: (u32, u32), orientation: Option<u32>, width: u32, height: u32) -> Option<u32> {
    rule_at(stored, orientation, width, height, 1.0, false)
}

fn rule_at(
    stored: (u32, u32),
    orientation: Option<u32>,
    width: u32,
    height: u32,
    dpr: f32,
    no_upscale: bool,
) -> Option<u32> {
    let side = |n| match n {
        0 => crate::url::Dimension::Proportional,
        n => crate::url::Dimension::Pixels(n),
    };
    let size = ImageSize {
        width: side(width),
        height: side(height),
        flip_horizontal: false,
        flip_vertical: false,
    };
    denominator_for(stored, orientation, &size, dpr, no_upscale)
}

#[test]
fn the_frame_is_the_smallest_still_one_and_a_half_times_the_target() {
    let stored = (2400, 1600);
    // The benchmark's request: the half-size frame is exactly
    // one and a half times 800, and 800 rows cover 533.
    assert_eq!(rule(stored, None, 800, 0), Some(2));
    assert_eq!(rule(stored, None, 800, 533), Some(2));
    // One pixel more on either axis and the half-size frame
    // is too small: full size.
    assert_eq!(rule(stored, None, 801, 533), None);
    assert_eq!(rule(stored, None, 800, 534), None);
    assert_eq!(rule(stored, None, 801, 0), None);
    // The same boundary at a quarter (600x400) and at an
    // eighth (300x200): past it, the next frame up.
    assert_eq!(rule(stored, None, 400, 266), Some(4));
    assert_eq!(rule(stored, None, 401, 266), Some(2));
    assert_eq!(rule(stored, None, 400, 267), Some(2));
    assert_eq!(rule(stored, None, 200, 133), Some(8));
    assert_eq!(rule(stored, None, 201, 133), Some(4));
    assert_eq!(rule(stored, None, 200, 134), Some(4));
    // Each axis is held on its own: one that would fit an
    // eighth does not carry one that needs the full size.
    assert_eq!(rule(stored, None, 100, 1600), None);
    assert_eq!(rule(stored, None, 2400, 10), None);
    assert_eq!(rule(stored, None, 100, 534), None);
    assert_eq!(rule(stored, None, 100, 533), Some(2));
    assert_eq!(rule(stored, None, 801, 10), None);
    assert_eq!(rule(stored, None, 800, 10), Some(2));
    // Far below every frame, the smallest one.
    assert_eq!(rule(stored, None, 1, 1), Some(8));
    assert_eq!(rule(stored, None, 16, 0), Some(8));
    // A target at or past the source never scales.
    for (width, height) in [(2400, 1600), (2400, 0), (0, 1600), (4800, 0), (1200, 800)] {
        assert_eq!(rule(stored, None, width, height), None, "{width}x{height}");
    }
}

#[test]
fn the_rule_is_one_and_a_half_times_on_both_axes_for_every_size() {
    // The rule stated another way, in floating point, over
    // sizes on and off the block grid and targets around
    // every boundary: the largest reduction whose frame,
    // rounded up, is still 1.5 times the target on each axis.
    let mut scaled = 0;
    let sources: [(u32, u32); 10] = [
        (2400, 1600),
        (2401, 1601),
        (1000, 667),
        (999, 1),
        (1, 999),
        (17, 9),
        (8, 8),
        (3, 3),
        (1, 1),
        (16_384, 4096),
    ];
    for stored in sources {
        for width in (1..=stored.0).step_by(usize::max(stored.0 as usize / 97, 1)) {
            for height in (1..=stored.1).step_by(usize::max(stored.1 as usize / 89, 1)) {
                let expected = [8u32, 4, 2].into_iter().find(|d| {
                    f64::from(stored.0.div_ceil(*d)) >= 1.5 * f64::from(width)
                        && f64::from(stored.1.div_ceil(*d)) >= 1.5 * f64::from(height)
                });
                assert_eq!(
                    rule(stored, None, width, height),
                    expected,
                    "{stored:?} to {width}x{height}"
                );
                scaled += usize::from(expected.is_some());
            }
        }
    }
    assert!(scaled > 4_000, "only {scaled} targets scaled");
}

#[test]
fn a_proportional_side_is_resolved_from_the_declared_size() {
    // 0x533 is 799x533 on a 2400x1600 source and 0x534 is
    // 801x534: the side the URL leaves out is the one that
    // decides.
    assert_eq!(rule((2400, 1600), None, 0, 533), Some(2));
    assert_eq!(rule((2400, 1600), None, 0, 534), None);
    // A tall source: 800 wide is 2400 high, three times the
    // half-size frame's rows short.
    assert_eq!(rule((1600, 4800), None, 800, 0), None);
    assert_eq!(rule((1600, 4800), None, 0, 1600), Some(2));
    assert_eq!(rule((1600, 4800), None, 0, 1601), None);
    // A source that is not a multiple of the scale: the frame
    // rounds up to 1201x801, which carries one pixel more.
    assert_eq!(rule((2401, 1601), None, 800, 0), Some(2));
    assert_eq!(rule((2401, 1601), None, 801, 0), None);
    assert_eq!(rule((2401, 1601), None, 0, 534), Some(2));
    assert_eq!(rule((2401, 1601), None, 0, 535), None);
}

#[test]
fn the_target_is_the_one_after_dpr() {
    let stored = (2400, 1600);
    assert_eq!(rule_at(stored, None, 400, 0, 1.0, false), Some(4));
    // 400 wide at DPR 2 is the 800-wide render.
    assert_eq!(rule_at(stored, None, 400, 0, 2.0, false), Some(2));
    assert_eq!(rule_at(stored, None, 401, 0, 2.0, false), None);
    assert_eq!(rule_at(stored, None, 400, 0, 3.0, false), None);
    // 1.5 times 533 is 800 wide, to the pixel; 534 is 801.
    assert_eq!(rule_at(stored, None, 533, 0, 1.5, false), Some(2));
    assert_eq!(rule_at(stored, None, 534, 0, 1.5, false), None);
    assert_eq!(rule_at(stored, None, 100, 0, 2.0, false), Some(8));
    assert_eq!(rule_at(stored, None, 101, 0, 2.0, false), Some(4));
    assert_eq!(rule_at(stored, None, 100, 0, 3.0, false), Some(4));
    assert_eq!(rule_at(stored, None, 66, 0, 3.0, false), Some(8));
}

#[test]
fn no_upscale_never_changes_the_scale() {
    // The cap holds a side at the source's own size, and a
    // target with such a side is past every frame with or
    // without it.
    for stored in [(2400, 1600), (999, 333), (17, 9)] {
        for width in [0, 1, 7, 100, 400, 800, 801, 1200, 2400, 5000] {
            for height in [0, 1, 7, 100, 266, 533, 534, 1600, 5000] {
                for orientation in [None, Some(6)] {
                    assert_eq!(
                        rule_at(stored, orientation, width, height, 1.0, true),
                        rule_at(stored, orientation, width, height, 1.0, false),
                        "{stored:?} to {width}x{height}, orientation {orientation:?}"
                    );
                }
            }
        }
    }
    assert_eq!(rule_at((2400, 1600), None, 5000, 100, 1.0, true), None);
}

#[test]
fn a_quarter_turn_holds_the_target_against_the_stored_axes() {
    let stored = (2400, 1600);
    // Orientations 1 to 4 keep the axes: 800 wide is 800x533.
    for orientation in [None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(9)] {
        assert_eq!(
            rule(stored, orientation, 800, 0),
            Some(2),
            "{orientation:?}"
        );
        assert_eq!(rule(stored, orientation, 801, 0), None, "{orientation:?}");
    }
    // Orientations 5 to 8 show the image 1600x2400. 800 wide
    // is then 800x1200, which is the half-size frame itself
    // lying on its side: full size. 533 wide is 533x799.
    for orientation in [5, 6, 7, 8].map(Some) {
        assert_eq!(rule(stored, orientation, 800, 0), None, "{orientation:?}");
        assert_eq!(
            rule(stored, orientation, 533, 0),
            Some(2),
            "{orientation:?}"
        );
        assert_eq!(rule(stored, orientation, 534, 0), None, "{orientation:?}");
        assert_eq!(
            rule(stored, orientation, 0, 800),
            Some(2),
            "{orientation:?}"
        );
        assert_eq!(rule(stored, orientation, 0, 801), None, "{orientation:?}");
        assert_eq!(
            rule(stored, orientation, 266, 0),
            Some(4),
            "{orientation:?}"
        );
    }
    // A box that fits upright and not turned, and the other
    // way round: a wide strip, 2400x400.
    let strip = (2400, 400);
    assert_eq!(rule(strip, None, 800, 100), Some(2));
    assert_eq!(rule(strip, Some(6), 800, 100), None);
    assert_eq!(rule(strip, Some(6), 100, 800), Some(2));
    assert_eq!(rule(strip, None, 100, 800), None);
}

/// The source the request tests share: 240x160, a tenth of
/// the benchmark's 2400x1600 on each side.
fn source() -> Vec<u8> {
    jpeg(&picture(240, 160))
}

#[test]
fn a_resize_or_a_fixed_crop_first_makes_a_request_eligible() {
    let source = source();
    // The benchmark's request at a tenth of its size, and the
    // quarter and the eighth below it.
    assert_eq!(scale_for(&source, "80x0"), Some(2));
    assert_eq!(scale_for(&source, "81x0"), None);
    assert_eq!(scale_for(&source, "40x0"), Some(4));
    assert_eq!(scale_for(&source, "20x0"), Some(8));
    // A crop to a fixed alignment is eligible too (issue #785),
    // at the scale its window leaves room for, whatever the
    // alignment. The 240x160 source cropped to 20x20 keeps a
    // 160x160 window, a quarter of which still covers 20x20;
    // to 40x40 only a half does.
    for remainder in [
        "crop:20x20",
        "crop:20x20:top",
        "crop:20x20:bottom-right",
        "crop:30x20",
    ] {
        assert_eq!(scale_for(&source, remainder), Some(4), "{remainder}");
    }
    for remainder in ["crop:40x40", "crop:40x40:left"] {
        assert_eq!(scale_for(&source, remainder), Some(2), "{remainder}");
    }
    // Whatever follows the resize, the first operation
    // decides, flips included.
    for remainder in [
        "80x0/filters:grayscale",
        "80x0/crop:25x25:smart",
        "80x0/fit-in:50x50",
        "80x0/50x0",
        "80x0/crop-rect:0x0:25x25",
        "80x0/pad:100x100:ffffff",
        "80x0/filters:format(webp):quality(70)",
        "80x0/debug:focal/crop:25x25",
        "80x0/focalpoint:0.5x0.5/crop:25x25",
        "80x0/filters:no_upscale",
        "-80x0",
        "80x-53",
        "-80x-53",
        "0x53",
        "80x53",
    ] {
        assert_eq!(scale_for(&source, remainder), Some(2), "{remainder}");
    }
    // Any other first operation, and no operation at all,
    // decodes at full size, however small its own target. A
    // smart, face or object crop reads the pixels to place its
    // window, so it is not a fixed fill crop and decodes whole.
    for remainder in [
        "",
        "crop:20x20:smart",
        "crop:20x20:face",
        "crop:20x20:objects(person)",
        "fit-in:20x20",
        "fit-in:20x0",
        "fit-in:0x20",
        "fit-in:20x20:adaptive",
        "pad:20x20:ffffff",
        "crop-rect:0x0:50x50/20x0",
        "focal:0x0:50x50/20x0",
        "focalpoint:0.5x0.5/20x0",
        "debug:focal/20x0",
        "filters:grayscale/20x0",
        "filters:format(png)/20x0",
        "filters:no_upscale/20x0",
        "filters:dpr(1)/20x0",
    ] {
        assert_eq!(scale_for(&source, remainder), None, "{remainder:?}");
    }
    // A resize or a fixed crop that shrinks nothing, or not
    // enough to leave a one-and-a-half-times resample.
    for remainder in [
        "0x0",
        "origxorig",
        "origx20",
        "20xorig",
        "240x160",
        "120x80",
        "crop:80x80",
        "crop:120x80",
        "crop:240x160:top",
    ] {
        assert_eq!(scale_for(&source, remainder), None, "{remainder}");
    }
    // A zero pixel side is the resize's to refuse.
    for remainder in ["00x20", "20x00"] {
        assert_eq!(scale_for(&source, remainder), None, "{remainder}");
    }
    // A request that selects a video frame or a PDF page
    // decodes what its tool wrote.
    for remainder in ["20x0/frame:poster", "20x0/page:2"] {
        assert_eq!(scale_for(&source, remainder), None, "{remainder}");
    }
}

#[test]
fn a_dpr_directive_and_a_dpr_hint_both_scale_the_target() {
    let source = source();
    // 40 wide at DPR 2 is the 80-wide render: a half, where
    // 40 wide alone takes a quarter.
    let ops = ops_of("40x0/filters:dpr(2)");
    let directives = RequestDirectives::analyze(&ops);
    let dpr = dpr_factor(directives.dpr, &ProcessContext::default());
    assert_eq!(plan_of(&source, &ops, dpr).map(|s| s.denominator), Some(2));
    assert_eq!(scale_for(&source, "40x0"), Some(4));
    // The same factor from the client hint, which the context
    // carries.
    let ops = ops_of("40x0");
    let directives = RequestDirectives::analyze(&ops);
    let hinted = ProcessContext {
        dpr: Some(2.0),
        ..ProcessContext::default()
    };
    let dpr = dpr_factor(directives.dpr, &hinted);
    assert_eq!(plan_of(&source, &ops, dpr).map(|s| s.denominator), Some(2));
    // And through a render: the frame the chain starts from
    // is the half-size one, and the output is the hinted size.
    let (frame, reduced) = decode_stage(&source, &ops, &hinted).unwrap();
    assert_eq!((frame.width(), frame.height()), (120, 80));
    assert_eq!(reduced.map(|r| r.full), Some((240, 160)));
    let out = Engine
        .process_preferred(&source, &ops, hinted.clone())
        .unwrap();
    assert_eq!((out.width, out.height), (80, 53));
    let at_full = at_full_size(|| Engine.process_preferred(&source, &ops, hinted));
    assert_eq!(sizes(&answer(Ok(out))), sizes(&answer(at_full)));
}

#[test]
fn only_a_ycbcr_jpeg_is_eligible() {
    let img = picture(240, 160);
    // Every chroma layout libjpeg-turbo writes, baseline and
    // progressive.
    for subsamp in [
        turbojpeg::Subsamp::None,
        turbojpeg::Subsamp::Sub2x1,
        turbojpeg::Subsamp::Sub2x2,
        turbojpeg::Subsamp::Sub1x2,
        turbojpeg::Subsamp::Sub4x1,
    ] {
        for progressive in [false, true] {
            let bytes = jpeg_with(&img, |c| {
                c.set_subsamp(subsamp).unwrap();
                c.set_progressive(progressive).unwrap();
            });
            assert_eq!(
                scale_for(&bytes, "80x0"),
                Some(2),
                "{subsamp:?}, progressive {progressive}"
            );
        }
    }
    // One component: grayscale.
    let gray = jpeg_with(&img, |c| c.set_subsamp(turbojpeg::Subsamp::Gray).unwrap());
    assert_eq!(scale_for(&gray, "80x0"), None, "grayscale");
    // Four: CMYK and YCCK.
    for colorspace in [turbojpeg::Colorspace::CMYK, turbojpeg::Colorspace::YCCK] {
        let bytes = four_component_jpeg(240, 160, colorspace);
        assert_eq!(scale_for(&bytes, "80x0"), None, "{colorspace:?}");
    }
    // Three components stored as RGB, with no colour
    // transform.
    let rgb = jpeg_with(&img, |c| {
        c.set_subsamp(turbojpeg::Subsamp::None).unwrap();
        c.set_colorspace(turbojpeg::Colorspace::RGB).unwrap();
    });
    assert_eq!(scale_for(&rgb, "80x0"), None, "RGB-coded");
    // Codings zune-jpeg does not read, which libjpeg-turbo
    // would: arithmetic and lossless.
    let arithmetic = jpeg_with(&img, |c| c.set_arithmetic(true).unwrap());
    assert!(turbojpeg::read_header(&arithmetic).unwrap().is_arithmetic);
    assert_eq!(scale_for(&arithmetic, "80x0"), None, "arithmetic");
    let lossless = jpeg_with(&img, |c| c.set_lossless(true).unwrap());
    assert!(turbojpeg::read_header(&lossless).unwrap().is_lossless);
    assert_eq!(scale_for(&lossless, "80x0"), None, "lossless");
    // A frame header that declares 12-bit samples.
    let mut twelve = jpeg(&img);
    let at = frame_header(&twelve);
    assert_eq!(twelve[at], 8);
    twelve[at] = 12;
    assert_eq!(scale_for(&twelve, "80x0"), None, "12-bit");
    // Not a JPEG at all.
    let png = {
        let mut out = std::io::Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(img.clone())
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    };
    for (what, bytes) in [
        ("a PNG", png.as_slice()),
        ("nothing", b"".as_slice()),
        ("a JPEG signature alone", b"\xff\xd8\xff".as_slice()),
        ("text", b"not an image".as_slice()),
    ] {
        assert_eq!(scale_for(bytes, "80x0"), None, "{what}");
    }
}

#[test]
fn the_two_decoders_must_agree_on_the_colour_space() {
    // A file whose components are named R, G and B but which
    // also carries a JFIF marker: zune-jpeg goes by the names
    // and reads RGB, libjpeg goes by the marker and reads
    // YCbCr. Which is right does not matter here; the two
    // would colour it differently, so it is never scaled.
    let mut bytes = jpeg_with(&picture(240, 160), |c| {
        c.set_subsamp(turbojpeg::Subsamp::None).unwrap();
    });
    assert_eq!(scale_for(&bytes, "80x0"), Some(2));
    let at = frame_header(&bytes);
    assert_eq!(bytes[at + 5], 3, "three components");
    let names = at + 6;
    assert_eq!(
        [bytes[names], bytes[names + 3], bytes[names + 6]],
        [1, 2, 3]
    );
    let scan = header_segments(&bytes).last().unwrap().1 + 5;
    assert_eq!([bytes[scan], bytes[scan + 2], bytes[scan + 4]], [1, 2, 3]);
    for (i, name) in b"RGB".iter().enumerate() {
        bytes[names + 3 * i] = *name;
        bytes[scan + 2 * i] = *name;
    }
    assert_eq!(
        turbojpeg::read_header(&bytes).unwrap().colorspace,
        turbojpeg::Colorspace::YCbCr,
        "libjpeg-turbo's reading"
    );
    assert_eq!(scale_for(&bytes, "80x0"), None);
    // The render is the full-size path's, to the byte.
    assert_eq!(
        answer(render(&bytes, "80x0")),
        answer(at_full_size(|| render(&bytes, "80x0")))
    );
}

// The frame, and what it stands for.

#[test]
fn the_decode_stage_hands_on_the_reduced_frame_and_the_declared_size() {
    let source = source();
    let ctx = ProcessContext::default();
    let stage = |bytes: &[u8], remainder: &str| {
        decode_stage(bytes, &ops_of(remainder), &ctx)
            .unwrap_or_else(|e| panic!("{remainder}: {e:?}"))
    };
    for (remainder, frame) in [("80x0", (120, 80)), ("40x0", (60, 40)), ("20x0", (30, 20))] {
        let (img, reduced) = stage(&source, remainder);
        assert_eq!((img.width(), img.height()), frame, "{remainder}");
        assert!(matches!(img, DynamicImage::ImageRgb8(_)), "{remainder}");
        // The frame stands for the 240x160 image, which fills
        // it: 240 and 160 are multiples of every scale.
        let filled = (0.0, 0.0, f64::from(frame.0), f64::from(frame.1));
        assert_eq!(
            reduced,
            Some(Reduced {
                full: (240, 160),
                window: filled
            }),
            "{remainder}"
        );
    }
    // A full-size decode says nothing: the frame is the image.
    for remainder in ["81x0", "crop:120x80", ""] {
        let (img, reduced) = stage(&source, remainder);
        assert_eq!((img.width(), img.height()), (240, 160), "{remainder:?}");
        assert_eq!(reduced, None, "{remainder:?}");
    }
    // A fixed-alignment crop reduces too (issue #785): crop:20x20
    // on the 240x160 source decodes at a quarter, and the frame
    // stands for the full 240x160 image.
    let (img, reduced) = stage(&source, "crop:20x20");
    assert_eq!((img.width(), img.height()), (60, 40));
    assert_eq!(reduced.map(|r| r.full), Some((240, 160)));
    // Every orientation: the frame comes upright, and the
    // size it stands for is the upright one.
    for orientation in 1..=8u16 {
        let bytes = with_orientation(&source, orientation);
        let (img, reduced) = stage(&bytes, "50x0");
        let (frame, full) = if orientation >= 5 {
            ((80, 120), (160, 240))
        } else {
            ((120, 80), (240, 160))
        };
        assert_eq!(
            (img.width(), img.height()),
            frame,
            "orientation {orientation}"
        );
        assert_eq!(
            reduced.map(|r| r.full),
            Some(full),
            "orientation {orientation}"
        );
        // The header's own size is the stored one, as before.
        assert_eq!(probe_source(&bytes).dimensions, Some((240, 160)));
    }
}

#[test]
fn the_image_fills_a_frame_short_of_its_last_column_and_row() {
    // 245x165 at a quarter is a 62x42 frame, and the image is
    // 61.25 by 41.25 of it: the last column and row hold a
    // quarter of a pixel of image and the padding of the
    // file's blocks. As stored the image sits at the frame's
    // top left; each orientation carries that corner along.
    let scaled = |orientation| ScaledDecode {
        denominator: 4,
        stored: (245, 165),
        orientation,
    };
    assert_eq!(scaled(None).frame(), (62, 42));
    for (orientation, full, window) in [
        (None, (245, 165), (0.0, 0.0, 61.25, 41.25)),
        (Some(1), (245, 165), (0.0, 0.0, 61.25, 41.25)),
        (Some(2), (245, 165), (0.75, 0.0, 61.25, 41.25)),
        (Some(3), (245, 165), (0.75, 0.75, 61.25, 41.25)),
        (Some(4), (245, 165), (0.0, 0.75, 61.25, 41.25)),
        (Some(5), (165, 245), (0.0, 0.0, 41.25, 61.25)),
        (Some(6), (165, 245), (0.75, 0.0, 41.25, 61.25)),
        (Some(7), (165, 245), (0.75, 0.75, 41.25, 61.25)),
        (Some(8), (165, 245), (0.0, 0.75, 41.25, 61.25)),
        (Some(9), (245, 165), (0.0, 0.0, 61.25, 41.25)),
    ] {
        assert_eq!(
            scaled(orientation).reduced(),
            Reduced { full, window },
            "orientation {orientation:?}"
        );
    }
    // The decode stage hands the same on.
    let bytes = with_orientation(&jpeg(&picture(245, 165)), 6);
    let ops = ops_of("25x0");
    let (frame, reduced) = decode_stage(&bytes, &ops, &ProcessContext::default()).unwrap();
    assert_eq!((frame.width(), frame.height()), (42, 62));
    assert_eq!(
        reduced,
        Some(Reduced {
            full: (165, 245),
            window: (0.75, 0.0, 41.25, 61.25)
        })
    );
}

#[test]
fn a_frame_is_the_picture_at_reduced_scale_for_every_size_and_layout() {
    // Sizes on and off the 8-pixel block grid and off every
    // scale, down to one pixel; each chroma layout; baseline
    // and progressive. The frame is libjpeg's size for the
    // scale, and it is the picture, to its last row and
    // column: its brightness is the full-size decode's,
    // averaged down by the same factor. (Brightness, because
    // colour detail a file stores at half resolution is at
    // half the frame's resolution too.)
    let luma = |p: &image::Rgb<u8>| {
        (u32::from(p[0]) * 299 + u32::from(p[1]) * 587 + u32::from(p[2]) * 114) / 1000
    };
    let mut frames = 0;
    let mut worst = 0.0f64;
    for (width, height) in [
        (1, 1),
        (2, 3),
        (7, 5),
        (8, 8),
        (9, 17),
        (16, 16),
        (33, 31),
        (97, 61),
        (160, 120),
        (241, 163),
    ] {
        let img = picture(width, height);
        for subsamp in [
            turbojpeg::Subsamp::None,
            turbojpeg::Subsamp::Sub2x1,
            turbojpeg::Subsamp::Sub2x2,
            turbojpeg::Subsamp::Sub1x2,
            turbojpeg::Subsamp::Sub4x1,
        ] {
            for progressive in [false, true] {
                let bytes = jpeg_with(&img, |c| {
                    c.set_quality(95).unwrap();
                    c.set_subsamp(subsamp).unwrap();
                    c.set_progressive(progressive).unwrap();
                });
                let full = decode_image(&bytes).unwrap().to_rgb8();
                for denominator in DENOMINATORS {
                    let what = format!(
                        "{width}x{height} {subsamp:?} progressive {progressive} at 1/{denominator}"
                    );
                    let frame = decode_at(&bytes, denominator)
                        .unwrap_or_else(|| panic!("{what}: no frame"))
                        .to_rgb8();
                    assert_eq!(
                        (frame.width(), frame.height()),
                        (width.div_ceil(denominator), height.div_ceil(denominator)),
                        "{what}"
                    );
                    // Each frame pixel against the mean of the
                    // pixels it stands for. A smaller image is
                    // mostly the padding that fills out its
                    // blocks, which the frame's pixels average
                    // in and the full-size decode cuts off.
                    if width.min(height) >= 16 {
                        let mut squares = 0.0;
                        for (x, y, pixel) in frame.enumerate_pixels() {
                            let (mut sum, mut count) = (0, 0);
                            for sy in y * denominator..((y + 1) * denominator).min(height) {
                                for sx in x * denominator..((x + 1) * denominator).min(width) {
                                    sum += luma(full.get_pixel(sx, sy));
                                    count += 1;
                                }
                            }
                            squares += f64::from((sum / count).abs_diff(luma(pixel))).powi(2);
                        }
                        let rms = (squares / f64::from(frame.width() * frame.height())).sqrt();
                        worst = worst.max(rms);
                    }
                    frames += 1;
                }
            }
        }
    }
    assert_eq!(frames, 10 * 5 * 2 * 3);
    assert!(worst < 4.0, "{worst:.1} levels from the averaged picture");
}

#[test]
fn every_size_a_render_works_out_is_the_full_size_path_s() {
    // Sources on and off the scales, upright and turned, and
    // requests whose sizes lean on the source's: a
    // proportional side, `orig`, `no_upscale` on a later
    // enlargement, a flip, a crop and a fit after the resize,
    // a DPR factor.
    let mut scaled = 0;
    for ((width, height), orientations) in [
        ((120, 80), [1u16, 3, 6, 8].as_slice()),
        ((121, 83), [6].as_slice()),
        ((83, 125), [8].as_slice()),
        ((97, 61), [1].as_slice()),
    ] {
        let plain = jpeg(&picture(width, height));
        for orientation in orientations {
            let source = Source::new(with_orientation(&plain, *orientation));
            let upright = if *orientation >= 5 {
                (height, width)
            } else {
                (width, height)
            };
            for remainder in [
                "40x0",
                "0x25",
                "25x0",
                "12x17",
                "7x0",
                "0x7",
                "4x3",
                "1x1",
                "-22x0",
                "22x-10",
                "25x0/50x0",
                "25x0/500x0/filters:no_upscale",
                "25x0/0x500/filters:no_upscale",
                "12x0/filters:no_upscale/origx0",
                "15x0/filters:no_upscale/pad:100x100:ffffff",
                "30x0/crop:12x10",
                "30x0/crop:125x100/filters:no_upscale",
                "30x0/fit-in:16x16",
                "30x0/fit-in:225x225/filters:no_upscale",
                "11x0/filters:dpr(2)",
                "11x0/filters:dpr(3):no_upscale",
                "17x0/crop-rect:1x1:10x7/5x0",
                "17x0/focalpoint:0.2x0.8/crop:7x7",
                "22x0/filters:rotate(90)",
                "1250x0/filters:no_upscale",
                "origx5",
            ] {
                let what = format!("{width}x{height}, orientation {orientation}, {remainder}");
                let [ours, theirs] = source.sizes(&ops_of(remainder));
                assert_eq!(ours, theirs, "{what}");
                let (_, reported) = ours.unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(reported, upright, "{what}: the reported source size");
                scaled += usize::from(scale_for(&source.bytes, remainder).is_some());
            }
        }
    }
    assert!(scaled > 120, "only {scaled} renders took the scaled decode");
}

#[test]
fn a_proportional_side_and_the_cap_come_from_the_declared_size() {
    // 101x67 at a half is 51x34. 30 wide is 19 high by the
    // declared size (67 * 30 / 101) and would be 20 by the
    // frame's (34 * 30 / 51).
    let bytes = jpeg(&picture(101, 67));
    assert_eq!(scale_for(&bytes, "30x0"), Some(2));
    let out = render(&bytes, "30x0").unwrap();
    assert_eq!((out.width, out.height), (30, 19));
    assert_eq!((out.source_width, out.source_height), (101, 67));
    // `no_upscale` caps a later enlargement at the source's
    // 101x67, not at the 51x34 the chain started from.
    let out = render(&bytes, "30x0/90x0/filters:no_upscale").unwrap();
    assert_eq!((out.width, out.height), (90, 57));
    let out = render(&bytes, "30x0/200x0/filters:no_upscale").unwrap();
    assert_eq!((out.width, out.height), (101, 67));
    // A width hint is a width-only resize under `no_upscale`,
    // put ahead of the URL's operations: below the source it
    // scales the decode, and past the source it returns the
    // source's own size.
    let hinted = |width: u32| {
        let mut ops = vec![
            Operation::Size(ImageSize {
                width: crate::url::Dimension::Pixels(width),
                height: crate::url::Dimension::Proportional,
                flip_horizontal: false,
                flip_vertical: false,
            }),
            Operation::Filters(vec![crate::url::Filter::Plain(
                crate::url::PlainFilter::NoUpscale,
            )]),
        ];
        ops.extend(ops_of("filters:grayscale"));
        ops
    };
    assert_eq!(
        plan_of(&bytes, &hinted(33), 1.0).map(|s| s.denominator),
        Some(2)
    );
    let out = Engine.process(&bytes, &hinted(33)).unwrap();
    assert_eq!((out.width, out.height), (33, 21));
    assert_eq!((out.source_width, out.source_height), (101, 67));
    assert_eq!(plan_of(&bytes, &hinted(400), 1.0), None);
    let out = Engine.process(&bytes, &hinted(400)).unwrap();
    assert_eq!((out.width, out.height), (101, 67));
    assert_eq!((out.source_width, out.source_height), (101, 67));
}

#[test]
fn the_url_test_vectors_render_at_the_sizes_they_did() {
    // Every remainder of the shared grammar corpus and of the
    // signing vectors that parses, on a JPEG its 300x200 and
    // 100x100 targets shrink far enough: where the scaled
    // decode is taken, the output size and the reported
    // source size are the full-size path's, or the refusal
    // is. The rest never reach the scaled decode.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("spec");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(root.join(name)).unwrap()).unwrap()
    };
    let mut remainders: Vec<String> = Vec::new();
    for case in read("operation-vectors.json")["cases"].as_array().unwrap() {
        remainders.push(case["remainder"].as_str().unwrap().to_string());
    }
    for case in read("test-vectors.json").as_array().unwrap() {
        if let Some(remainder) = case["remainder"].as_str() {
            remainders.push(remainder.to_string());
        }
    }
    assert!(remainders.len() > 700, "{} remainders", remainders.len());
    let source = Source::new(jpeg(&picture(907, 613)));
    let ctx = ProcessContext::default();
    let (mut parsed, mut scaled, mut refused) = (0, 0, 0);
    // Many vectors spell the same operations, over another
    // source or with another signature: each list of
    // operations renders once.
    let mut rendered: Vec<Vec<Operation>> = Vec::new();
    // In a debug build most of a full-size render is its
    // first step, the resize of the whole frame, and many
    // vectors open with the same one. The image it leaves is
    // made once for each, and a chain goes on from a copy.
    type FirstStep = ((Operation, bool), Result<DynamicImage, String>);
    let mut first_steps: Vec<FirstStep> = Vec::new();
    for remainder in &remainders {
        let Ok((ops, _)) = crate::url::parse_operations(remainder) else {
            continue;
        };
        parsed += 1;
        let directives = RequestDirectives::analyze(&ops);
        let dpr = dpr_factor(directives.dpr, &ctx);
        if plan_of(&source.bytes, &ops, dpr).is_none() {
            continue;
        }
        scaled += 1;
        if rendered.contains(&ops) {
            continue;
        }
        rendered.push(ops.clone());
        assert_eq!(dpr, 1.0, "{remainder}");
        let ours = chain(&source.bytes, &ops)
            .map(|(img, reported)| ((img.width(), img.height()), reported))
            .map_err(|e| format!("{e:?}"));
        // The full-size path: the chain's own steps, the
        // first of them from the store above.
        let cap = directives.no_upscale.then_some((907, 613));
        let key = (ops[0].clone(), directives.no_upscale);
        let first = match first_steps.iter().find(|(known, _)| *known == key) {
            Some((_, first)) => first.clone(),
            None => {
                let first = ChainState::new(&ops, cap, dpr, &ctx, &directives, None)
                    .step(source.full.clone(), 0, &ops[0])
                    .map_err(|e| format!("{e:?}"));
                first_steps.push((key, first.clone()));
                first
            }
        };
        let theirs = first.and_then(|mut img| {
            let mut steps = ChainState::new(&ops, cap, dpr, &ctx, &directives, None);
            for (index, op) in ops.iter().enumerate().skip(1) {
                img = steps.step(img, index, op).map_err(|e| format!("{e:?}"))?;
            }
            let outcome = steps.finish(img).map_err(|e| format!("{e:?}"))?;
            Ok(((outcome.img.width(), outcome.img.height()), (907, 613)))
        });
        assert_eq!(ours, theirs, "{remainder}");
        if ours.is_err() {
            refused += 1;
        }
    }
    assert!(parsed > 400, "{parsed} remainders parsed");
    assert!(scaled >= 30, "only {scaled} vectors took the scaled decode");
    assert!(refused * 2 < rendered.len(), "{refused} refused");
}

// The picture.

#[test]
fn a_scaled_render_is_the_full_size_render_s_picture() {
    // The same picture, to well within what the eye sees, at
    // every scale and orientation, baseline and progressive,
    // for a size every scale divides and for one a pixel past
    // it, whose frames end in a column and a row that are
    // mostly padding. A frame turned the wrong way, mirrored,
    // cut short, or resampled a fraction of a pixel out of
    // place would be many decibels away.
    let mut worst = f64::INFINITY;
    for (width, height) in [(240, 160), (241, 161)] {
        let img = picture(width, height);
        for bytes in [jpeg(&img), progressive_jpeg(&img)] {
            for orientation in 1..=8u16 {
                let bytes = with_orientation(&bytes, orientation);
                let remainders = if orientation >= 5 {
                    [("50x0", 2), ("25x0", 4), ("12x0", 8)]
                } else {
                    [("80x0", 2), ("40x0", 4), ("20x0", 8)]
                };
                for (remainder, scale) in remainders {
                    let what = format!("{width}x{height}, orientation {orientation}, {remainder}");
                    assert_eq!(scale_for(&bytes, remainder), Some(scale), "{what}");
                    let (ours, theirs) = both_pixels(&bytes, remainder);
                    let apart = psnr(&ours, &theirs);
                    assert!(apart > 42.0, "{what}: {apart:.1} dB");
                    worst = worst.min(apart);
                }
            }
        }
    }
    // And it is another render: the pixels did change.
    assert!(
        worst < 60.0,
        "the scaled renders are {worst:.1} dB apart at worst"
    );
}

#[test]
fn a_scaled_crop_is_the_full_size_crop_s_picture() {
    // Issue #785: a fixed-alignment crop decoded at reduced scale
    // is the same picture as the full-size crop, to well within
    // what the eye sees, at every alignment and orientation,
    // baseline and progressive, for a size the scale divides and
    // for one a pixel past it. A window cut in the wrong place,
    // turned the wrong way, or resampled out of place would be
    // many decibels away.
    use crate::url::Align;
    let aligns = [
        Align::Center,
        Align::Top,
        Align::Bottom,
        Align::Left,
        Align::Right,
        Align::TopLeft,
        Align::TopRight,
        Align::BottomLeft,
        Align::BottomRight,
    ];
    let mut worst = f64::INFINITY;
    for (width, height) in [(240, 160), (241, 161)] {
        let img = picture(width, height);
        for bytes in [jpeg(&img), progressive_jpeg(&img)] {
            for orientation in 1..=8u16 {
                let bytes = with_orientation(&bytes, orientation);
                for align in &aligns {
                    let remainder = format!("crop:40x40:{}", align_slug(align));
                    assert_eq!(
                        scale_for(&bytes, &remainder),
                        Some(2),
                        "{width}x{height} orientation {orientation} {remainder}"
                    );
                    let (ours, theirs) = both_pixels(&bytes, &remainder);
                    assert_eq!((ours.width(), ours.height()), (40, 40));
                    assert_eq!(
                        (ours.width(), ours.height()),
                        (theirs.width(), theirs.height())
                    );
                    let apart = psnr(&ours, &theirs);
                    assert!(
                        apart > 40.0,
                        "{width}x{height} orientation {orientation} {remainder}: {apart:.1} dB"
                    );
                    worst = worst.min(apart);
                }
            }
        }
    }
    // The pixels did change: the scaled crop is not byte-identical
    // to the full-size crop.
    assert!(
        worst < 70.0,
        "scaled crops are {worst:.1} dB apart at worst"
    );
}

/// The URL alignment token for a fixed alignment.
fn align_slug(align: &crate::url::Align) -> &'static str {
    use crate::url::Align;
    match align {
        Align::Center => "center",
        Align::Top => "top",
        Align::Bottom => "bottom",
        Align::Left => "left",
        Align::Right => "right",
        Align::TopLeft => "top-left",
        Align::TopRight => "top-right",
        Align::BottomLeft => "bottom-left",
        Align::BottomRight => "bottom-right",
        _ => unreachable!("only fixed alignments are slugged here"),
    }
}

#[test]
fn a_scaled_crop_resolves_the_sizes_the_full_size_crop_does() {
    // Issue #785: every size a crop works out — the output size,
    // the source size it reports, under DPR, no_upscale, odd
    // sizes and every orientation — is the one the full-size
    // crop works out. Only the pixels differ.
    for (width, height) in [(240, 160), (241, 161)] {
        let plain = picture(width, height);
        for base in [jpeg(&plain), progressive_jpeg(&plain)] {
            for orientation in 1..=8u16 {
                let src = Source::new(with_orientation(&base, orientation));
                for remainder in [
                    "crop:40x40",
                    "crop:40x40:top",
                    "crop:50x30:bottom-right",
                    "crop:40x40/filters:dpr(2)",
                    "crop:40x40/filters:no_upscale",
                    "crop:30x40:left/filters:dpr(2):no_upscale",
                ] {
                    let [ours, theirs] = src.sizes(&ops_of(remainder));
                    assert_eq!(
                        ours, theirs,
                        "{width}x{height} orientation {orientation} {remainder}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_crop_of_a_file_the_scaled_decode_does_not_take_falls_back_to_full_size() {
    // Issue #785: the scaled crop stands down for any file the
    // scaled decode does not take — a four-component JPEG, CMYK or
    // YCCK, is not the 8-bit YCbCr frame `eligible_size` requires —
    // exactly as a scaled resize does, and the full-size crop
    // answers with the pixels or the error it always gave.
    for colorspace in [turbojpeg::Colorspace::CMYK, turbojpeg::Colorspace::YCCK] {
        let bytes = four_component_jpeg(240, 160, colorspace);
        assert_eq!(scale_for(&bytes, "crop:40x40"), None, "{colorspace:?}");
        assert_eq!(
            answer(render(&bytes, "crop:40x40")),
            answer(at_full_size(|| render(&bytes, "crop:40x40"))),
            "{colorspace:?}"
        );
    }
}

/// Issue #785 measurement: the crop-fill scaled decode against
/// the full-size crop on the 25 benchmark sources (2400x1600).
/// Run with `--ignored --nocapture`. Reports, per crop, the
/// SSIMULACRA2 of the scaled crop against the full-size crop and
/// the PSNR between them, and the aggregate floor the keep
/// decision is made on.
#[test]
#[ignore = "measurement for issue #785; run with --ignored --nocapture"]
fn measure_crop_fill_scaled_vs_full() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/benchmark/fixtures");
    let mut worst_psnr = f64::INFINITY;
    let mut worst_ssim = f64::INFINITY;
    let mut psnrs = Vec::new();
    for crop in [
        "crop:400x300",
        "crop:400x300:top",
        "crop:300x400:bottom-right",
    ] {
        for i in 0..25 {
            let path = format!("{dir}/src{i:02}.jpg");
            let Ok(bytes) = std::fs::read(&path) else {
                eprintln!("skip {path}: unreadable");
                continue;
            };
            let scale = scale_for(&bytes, crop);
            let (scaled, full) = both_pixels(&bytes, crop);
            let p = psnr(&scaled, &full);
            let ss = crate::quality::ssimulacra2(&full, &scaled).unwrap_or(f64::NAN);
            worst_psnr = worst_psnr.min(p);
            if ss.is_finite() {
                worst_ssim = worst_ssim.min(ss);
            }
            psnrs.push(p);
            eprintln!("{crop} src{i:02} scale={scale:?} psnr={p:.2}dB ssimulacra2={ss:.2}");
        }
    }
    psnrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = psnrs[psnrs.len() / 2];
    eprintln!(
        "crop-fill: {} crops, PSNR floor {:.2} dB, median {:.2} dB, SSIMULACRA2 floor {:.2}",
        psnrs.len(),
        worst_psnr,
        median,
        worst_ssim
    );
}

#[test]
fn a_phone_photo_is_turned_and_colour_managed_as_at_full_size() {
    // Orientation 6 and a Display P3 profile, as a phone
    // writes its photos: the reduced frame is turned upright
    // and converted to sRGB like the full-size one.
    let plain = jpeg(&picture(240, 160));
    let phone = with_orientation(&with_display_p3(&plain), 6);
    assert_eq!(scale_for(&phone, "50x0"), Some(2));
    let out = render(&phone, "50x0").unwrap();
    assert_eq!((out.width, out.height), (50, 75));
    assert_eq!((out.source_width, out.source_height), (160, 240));
    let (ours, theirs) = both_pixels(&phone, "50x0");
    let apart = psnr(&ours, &theirs);
    assert!(apart > 38.0, "{apart:.1} dB from the full-size render");
    // The conversion ran: without the profile the same pixels
    // render as other colours, further off than the two paths
    // are from each other.
    let (untagged, _) = both_pixels(&with_orientation(&plain, 6), "50x0");
    let unconverted = psnr(&ours, &untagged);
    assert!(
        unconverted < apart - 6.0,
        "{unconverted:.1} dB from the untagged render, {apart:.1} dB from the full-size one"
    );
    // `keep_profile` keeps the pixels as they are and carries
    // the profile, on both paths: the file is the untagged
    // render's with the profile added.
    let kept = render(&phone, "50x0/filters:keep_profile").unwrap();
    let kept_full = at_full_size(|| render(&phone, "50x0/filters:keep_profile")).unwrap();
    let profile = |bytes: &[u8]| crate::color::extract_source_meta(bytes, false).icc;
    assert!(profile(&kept.bytes).is_some());
    assert_eq!(profile(&kept.bytes), profile(&kept_full.bytes));
    let plain_render = render(&with_orientation(&plain, 6), "50x0").unwrap();
    assert_eq!(
        decode_image(&kept.bytes).unwrap().as_bytes(),
        decode_image(&plain_render.bytes).unwrap().as_bytes()
    );
    let apart = psnr(
        &decode_image(&kept.bytes).unwrap(),
        &decode_image(&kept_full.bytes).unwrap(),
    );
    assert!(apart > 32.0, "keep_profile: {apart:.1} dB");
}

// The other callers of the JPEG decode.

#[test]
fn watermarks_and_the_quality_search_decode_at_full_size() {
    let source = source();
    // A watermark is decoded whole, whatever the request's
    // first operation: it is drawn at its own size.
    let watermarks = decode_watermarks(&[("wm.jpg".to_string(), source.clone())], None).unwrap();
    let mark = watermarks.get("wm.jpg").unwrap().image();
    assert_eq!((mark.width(), mark.height()), (240, 160));
    let ops = ops_of("80x0/filters:watermark(wm.jpg,0,0,50)");
    let ctx = ProcessContext {
        watermarks: Some(&watermarks),
        ..ProcessContext::default()
    };
    let out = Engine.process_preferred(&source, &ops, ctx).unwrap();
    assert_eq!((out.width, out.height), (80, 53));
    assert_eq!((out.source_width, out.source_height), (240, 160));
    // `quality(auto)` decodes its own candidates to score
    // them: a JPEG of the output's size, with no operations.
    let out = render(&source, "80x0/filters:quality(auto)").unwrap();
    assert_eq!((out.width, out.height), (80, 53));
    assert!(out.ssimulacra2.is_some());
    // The decode every other caller uses takes no operations
    // and has no scale: an upload is validated by it, and a
    // meta request reads its own output through it.
    let whole = decode_image(&source).unwrap();
    assert_eq!((whole.width(), whole.height()), (240, 160));
    assert!(super::super::validate_source_image(&source).is_ok());
}

// The limits.

#[test]
fn a_source_past_the_decode_limits_is_refused_before_any_decode() {
    // One pixel past the 16384 px side limit, a real file: a
    // quarter of it would be a small frame, and it is refused
    // by its declared size all the same, as at full size.
    let wide = jpeg(&image::RgbImage::from_pixel(
        16_385,
        16,
        image::Rgb([90, 140, 200]),
    ));
    assert_eq!(rule((16_385, 16), None, 800, 2), Some(4));
    assert_eq!(plan_for(&wide, "800x2"), None);
    let refused = answer(render(&wide, "800x2"));
    assert_eq!(refused, answer(at_full_size(|| render(&wide, "800x2"))));
    assert!(
        refused
            .as_ref()
            .is_err_and(|e| e.contains("16385x16 exceed the 16384 px decode limit")),
        "{refused:?}"
    );
    // At the limit it decodes, at scale.
    let at_limit = jpeg(&image::RgbImage::from_pixel(
        16_384,
        16,
        image::Rgb([90, 140, 200]),
    ));
    assert_eq!(scale_for(&at_limit, "800x2"), Some(4));
    let out = render(&at_limit, "800x2").unwrap();
    assert_eq!((out.width, out.height), (800, 2));
    assert_eq!((out.source_width, out.source_height), (16_384, 16));
    // Past the 256 MiB decode allocation limit, by what the
    // header declares: 16384x4097 is one row too many.
    let declared = declaring(&jpeg(&picture(64, 48)), 16_384, 4097);
    assert_eq!(rule((16_384, 4097), None, 100, 0), Some(8));
    assert_eq!(plan_for(&declared, "100x0"), None);
    let refused = answer(render(&declared, "100x0"));
    assert_eq!(refused, answer(at_full_size(|| render(&declared, "100x0"))));
    assert!(
        refused
            .as_ref()
            .is_err_and(|e| e.contains("16384x4097 exceed the decode allocation limit")),
        "{refused:?}"
    );
    // 16384x4096 is the largest declared size the limit
    // admits, and so the largest a scaled decode may start on.
    let declared = declaring(&jpeg(&picture(64, 48)), 16_384, 4096);
    assert_eq!(scale_for(&declared, "100x0"), Some(8));
}

/// A progressive CMYK JPEG, as [`four_component_jpeg`] writes
/// a CMYK one.
fn progressive_cmyk_jpeg(width: u32, height: u32) -> Vec<u8> {
    let pixels: Vec<u8> = picture(width, height)
        .pixels()
        .flat_map(|p| [p[0], p[1], p[2], 255 - p[0] / 2])
        .collect();
    let mut compressor = turbojpeg::Compressor::new().unwrap();
    compressor.set_quality(90).unwrap();
    compressor.set_subsamp(turbojpeg::Subsamp::None).unwrap();
    compressor
        .set_colorspace(turbojpeg::Colorspace::CMYK)
        .unwrap();
    compressor.set_progressive(true).unwrap();
    compressor
        .compress_to_vec(turbojpeg::Image {
            pixels: pixels.as_slice(),
            width: width as usize,
            pitch: width as usize * 4,
            height: height as usize,
            format: turbojpeg::PixelFormat::CMYK,
        })
        .unwrap()
}

#[test]
fn a_progressive_jpeg_is_charged_its_coefficients_before_any_decode() {
    // A progressive file cannot be decoded a row at a time:
    // either decoder keeps every component's coefficients, two
    // bytes a sample, until the last scan, beside the frame.
    // So the frame, at four bytes a pixel, and the coefficients
    // are charged together against the 256 MiB decode budget,
    // from the frame header, before anything decodes. At 4:2:0
    // a 6200x6200 frame is 154 MB and its coefficients 116 MB:
    // past the budget, though the frame alone is inside it.
    let declared = declaring(&progressive_jpeg(&picture(64, 48)), 6200, 6200);
    let refused = decode_image(&declared);
    assert!(
        matches!(&refused, Err(EngineError::Decode(why)) if why.contains("coefficients")),
        "{:?}",
        refused.map(|img| (img.width(), img.height()))
    );
    // A request that shrinks it far enough for a scaled decode
    // is refused the same way: the scaled decode holds the same
    // coefficients, and a file the full-size path refuses is
    // not eligible for it.
    let answered = answer(render(&declared, "100x0"));
    assert_eq!(
        answered,
        answer(at_full_size(|| render(&declared, "100x0")))
    );
    assert!(
        answered.as_ref().is_err_and(|e| e.contains("coefficients")),
        "{answered:?}"
    );
    // A CMYK file keeps four components at full resolution and
    // never takes the scaled decode: a 4800x4800 one holds a
    // 92 MB frame and 184 MB of coefficients.
    let cmyk = declaring(&progressive_cmyk_jpeg(64, 48), 4800, 4800);
    let refused = decode_image(&cmyk);
    assert!(
        matches!(&refused, Err(EngineError::Decode(why)) if why.contains("coefficients")),
        "{:?}",
        refused.map(|img| (img.width(), img.height()))
    );
}

/// The coefficients [`jpeg_coefficient_bytes`] charges
/// `bytes`, from zune-jpeg's reading of its header.
fn coefficients(bytes: &[u8]) -> u64 {
    let mut zune = zune_jpeg::JpegDecoder::new_with_options(
        zune_core::bytestream::ZCursor::new(bytes),
        zune_jpeg_options(),
    );
    zune.decode_headers().unwrap();
    jpeg_coefficient_bytes(bytes, &zune)
}

#[test]
fn the_coefficients_charged_are_the_blocks_the_decoders_keep() {
    // 100x60 is 13x8 blocks at full resolution, and 7 MCUs of
    // 16 px across, 4 of 16 px down: each component keeps its
    // blocks over the image padded out to whole MCUs, 64
    // coefficients of two bytes a block. A baseline file keeps
    // none.
    let img = picture(100, 60);
    let layouts = [
        // Every component at full resolution.
        (turbojpeg::Subsamp::None, 3 * 13 * 8),
        // Luma at 2x1 in MCUs of 16x8, chroma at half width.
        (turbojpeg::Subsamp::Sub2x1, 14 * 8 + 2 * 7 * 8),
        (turbojpeg::Subsamp::Sub2x2, 14 * 8 + 2 * 7 * 4),
        (turbojpeg::Subsamp::Sub1x2, 13 * 8 + 2 * 13 * 4),
        // MCUs of 32x8: four across.
        (turbojpeg::Subsamp::Sub4x1, 16 * 8 + 2 * 4 * 8),
        (turbojpeg::Subsamp::Gray, 13 * 8),
    ];
    for (subsamp, blocks) in layouts {
        let baseline = jpeg_with(&img, |c| c.set_subsamp(subsamp).unwrap());
        let progressive = jpeg_with(&img, |c| {
            c.set_subsamp(subsamp).unwrap();
            c.set_progressive(true).unwrap();
        });
        assert_eq!(coefficients(&baseline), 0, "{subsamp:?} baseline");
        assert_eq!(
            coefficients(&progressive),
            blocks * 128,
            "{subsamp:?} progressive"
        );
    }
    // CMYK keeps four components at full resolution.
    assert_eq!(
        coefficients(&progressive_cmyk_jpeg(100, 60)),
        4 * 13 * 8 * 128
    );
    // The charge is the frame and the coefficients together:
    // at 4:2:0, 6100x6100 holds 261 MB and fits the budget,
    // and 6200x6200 holds 269 MB and does not.
    let small = progressive_jpeg(&picture(64, 48));
    for (side, fits) in [(6100, true), (6200, false)] {
        let declared = declaring(&small, side, side);
        let held = u64::from(side) * u64::from(side) * 4 + coefficients(&declared);
        assert_eq!(held <= MAX_DECODE_ALLOC, fits, "{side}: {held}");
    }
}

#[test]
fn the_sampling_factors_are_read_from_the_frame_zune_jpeg_reads() {
    let img = picture(100, 60);
    let progressive = progressive_jpeg(&img);
    // 4:2:0: luma at 2x2 and each chroma component at 1x1.
    let exact = (14 * 8 + 2 * 7 * 4) * 128;
    // Every component at 2x2.
    let most = 3 * 14 * 8 * 128;
    assert_eq!(coefficients(&progressive), exact);
    // A segment that carries another JPEG, as an EXIF thumbnail
    // does, is stepped over by its length, frame header and all.
    let thumbnail = jpeg_with(&picture(30, 20), |c| {
        c.set_subsamp(turbojpeg::Subsamp::None).unwrap();
        c.set_progressive(true).unwrap();
    });
    let with_thumbnail = with_segment(
        &progressive,
        0xE1,
        &[&b"Exif\0\0"[..], &thumbnail[..]].concat(),
    );
    assert_eq!(coefficients(&with_thumbnail), exact);
    // zune-jpeg reads five bytes of an APP0 segment of length 6,
    // which holds four: the byte after it is the segment's to
    // zune-jpeg, and the walk steps over it too.
    let app0 = [
        &progressive[..2],
        &[0xFF, 0xE0, 0, 6, b'J', b'F', b'I', b'F', 0][..],
        &progressive[2..],
    ]
    .concat();
    assert_eq!(coefficients(&app0), exact);
    // A stray byte between two segments, which zune-jpeg passes
    // over and the walk does not follow: every component is
    // charged at the largest factors, the most the frame holds.
    let (_, _, first_end) = header_segments(&progressive)[0];
    let stray = [
        &progressive[..first_end],
        &[0u8][..],
        &progressive[first_end..],
    ]
    .concat();
    assert_eq!(coefficients(&stray), most);
    // zune-jpeg decodes all three as the file itself.
    let want = decode_image(&progressive).unwrap();
    for (what, bytes) in [
        ("app0", &app0),
        ("stray", &stray),
        ("thumbnail", &with_thumbnail),
    ] {
        assert!(
            decode_image(bytes).unwrap().as_bytes() == want.as_bytes(),
            "{what}"
        );
    }
}

/// A sequential JPEG of `components` components at 1x1, each
/// coded in a scan of its own, of a flat picture: one Huffman
/// table of one two-bit code for each class, so every block is
/// a DC difference of zero and an end of block, four zero bits.
fn a_component_a_scan_jpeg(width: u16, height: u16, components: u8) -> Vec<u8> {
    let mut out = vec![0xFF, 0xD8];
    // A quantization table of ones.
    out.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
    out.extend_from_slice(&[1u8; 64]);
    out.extend_from_slice(&[0xFF, 0xC0]);
    out.extend_from_slice(&(8 + 3 * u16::from(components)).to_be_bytes());
    out.push(8);
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&width.to_be_bytes());
    out.push(components);
    for component in 1..=components {
        out.extend_from_slice(&[component, 0x11, 0]);
    }
    for class in [0x00, 0x10] {
        out.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, class]);
        let mut counts = [0u8; 16];
        counts[1] = 1;
        out.extend_from_slice(&counts);
        out.push(0);
    }
    let blocks = usize::from(width).div_ceil(8) * usize::from(height).div_ceil(8);
    for component in 1..=components {
        out.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 1, component, 0x00, 0, 63, 0]);
        let mut data = vec![0u8; blocks.div_ceil(2)];
        if blocks % 2 == 1 {
            // The last block's four bits, then ones to the byte.
            *data.last_mut().unwrap() = 0x0F;
        }
        out.extend_from_slice(&data);
    }
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

#[test]
fn a_sequential_jpeg_coded_a_component_a_scan_is_charged_its_coefficients() {
    // A sequential file whose first scan leaves components to
    // later scans cannot be decoded a row at a time either:
    // zune-jpeg keeps every component's coefficients until the
    // scans that carry the rest have come, as for a progressive
    // file, so it is charged them the same way. 8192x8192 in
    // three components at full resolution is a 256 MiB frame and
    // 384 MiB of coefficients.
    let small = a_component_a_scan_jpeg(64, 48, 3);
    let declared = declaring(&small, 8192, 8192);
    assert_eq!(coefficients(&declared), 3 * 1024 * 1024 * 128);
    let refused = decode_image(&declared);
    assert!(
        matches!(&refused, Err(EngineError::Decode(why))
            if why.contains("coefficients") && !why.contains("progressive")),
        "{:?}",
        refused.map(|img| (img.width(), img.height()))
    );
    // A shrinking request is refused the same way: such a file
    // never takes the scaled decode.
    assert_eq!(plan_for(&declared, "100x0"), None);
    let answered = answer(render(&declared, "100x0"));
    assert!(
        answered.as_ref().is_err_and(|e| e.contains("coefficients")),
        "{answered:?}"
    );
    // Four components, as a CMYK file has them.
    let four = declaring(&a_component_a_scan_jpeg(64, 48, 4), 6000, 6000);
    assert_eq!(coefficients(&four), 4 * 750 * 750 * 128);
    assert!(decode_image(&four).is_err());
    // Within the budget such a file decodes as before.
    assert_eq!(coefficients(&small), 3 * 8 * 6 * 128);
    let img = decode_image(&small).unwrap();
    assert_eq!((img.width(), img.height()), (64, 48));
    // A sequential file whose first scan carries every component
    // is decoded a row of MCUs at a time and is charged its frame
    // alone, as is a sequential file of one component.
    let one_scan = declaring(&jpeg(&picture(64, 48)), 8192, 8192);
    assert_eq!(coefficients(&one_scan), 0);
    let gray = jpeg_with(&picture(64, 48), |c| {
        c.set_subsamp(turbojpeg::Subsamp::Gray).unwrap();
    });
    assert_eq!(coefficients(&declaring(&gray, 8192, 8192)), 0);
}

#[test]
fn a_passed_deadline_stops_the_request_before_a_second_decode() {
    let past = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
    let ctx = ProcessContext {
        deadline: past,
        ..ProcessContext::default()
    };
    let source = source();
    // A request past its deadline is stopped before it
    // decodes anything, on either path.
    let ops = ops_of("80x0");
    assert!(matches!(
        Engine.process_preferred(&source, &ops, ctx.clone()),
        Err(EngineError::Timeout)
    ));
    // The decode stage alone, as a request that reached it in
    // time runs it. A file the scaled decode refuses is not
    // decoded a second time once the deadline has passed.
    let truncated = &source[..source.len() / 2];
    let directives = RequestDirectives::analyze(&ops);
    assert!(plan_of(truncated, &ops, 1.0).is_some());
    let sniffed = Engine::sniff_source(truncated).unwrap();
    assert!(matches!(
        Engine::decode_source(&sniffed, &ops, &ctx, &directives, false),
        Err(EngineError::Timeout)
    ));
    // With time left the full-size decode answers.
    assert!(
        decode_stage(truncated, &ops, &ProcessContext::default()).is_ok_and(|(img, reduced)| (
            img.width(),
            img.height()
        ) == (
            240, 160
        )
            && reduced.is_none())
    );
    // A file it accepts has one decode, and nothing to check.
    let sniffed = Engine::sniff_source(&source).unwrap();
    assert!(matches!(
        Engine::decode_source(&sniffed, &ops, &ctx, &directives, false),
        Ok(Decoded::Still(_, _, Some(reduced))) if reduced.full == (240, 160)
    ));
}

// Hostile input.

/// Render `remainder` on a file the scaled decode is planned
/// for and must refuse: the answer is the full-size path's,
/// to the byte or to the error.
fn assert_refused_and_answered_as_before(what: &str, bytes: &[u8], remainder: &str) {
    let scaled = plan_for(bytes, remainder).unwrap_or_else(|| panic!("{what}: not planned"));
    assert!(decode(bytes, &scaled).is_none(), "{what}: decoded at scale");
    assert_eq!(
        answer(render(bytes, remainder)),
        answer(at_full_size(|| render(bytes, remainder))),
        "{what}"
    );
}

/// Render `remainder` on a hostile file, whatever the scaled
/// decode makes of it: it returns, and the request gets the
/// full-size path's answer (where the scaled decode stood
/// down), or the same sizes, the same kind of answer and,
/// where both render, the same picture. Returns whether the
/// scaled decode produced a frame.
fn assert_survives(what: &str, bytes: &[u8], remainder: &str) -> bool {
    let frame = plan_for(bytes, remainder).and_then(|scaled| decode(bytes, &scaled));
    let ours = answer(render(bytes, remainder));
    let theirs = answer(at_full_size(|| render(bytes, remainder)));
    if frame.is_none() {
        assert_eq!(ours, theirs, "{what}");
    } else {
        assert_eq!(sizes(&ours), sizes(&theirs), "{what}");
        if let (Ok(ours), Ok(theirs)) = (&ours, &theirs) {
            let apart = psnr(
                &decode_image(&ours.3).unwrap(),
                &decode_image(&theirs.3).unwrap(),
            );
            assert!(apart > 20.0, "{what}: {apart:.1} dB between the two paths");
        }
    }
    frame.is_some()
}

#[test]
fn a_truncated_file_gets_the_full_size_path_s_answer() {
    let img = picture(120, 80);
    for (kind, bytes) in [
        ("baseline", jpeg(&img)),
        ("progressive", progressive_jpeg(&img)),
    ] {
        assert_eq!(scale_for(&bytes, "40x0"), Some(2), "{kind}");
        let scan = header_segments(&bytes).last().unwrap().1;
        // Cut inside the header, at the scan's start, and
        // through the entropy-coded data to the last byte.
        let cuts = (0..scan)
            .step_by(61)
            .chain((scan..bytes.len()).step_by((bytes.len() - scan) / 17));
        let (mut cut_count, mut planned) = (0, 0);
        for cut in cuts.chain([bytes.len() - 2, bytes.len() - 1]) {
            let what = format!("{kind} cut at {cut} of {}", bytes.len());
            let cut = &bytes[..cut];
            cut_count += 1;
            if plan_for(cut, "40x0").is_some() {
                planned += 1;
                // libjpeg-turbo reads a short file to its end
                // and says so with a warning.
                assert_refused_and_answered_as_before(&what, cut, "40x0");
            } else {
                assert_survives(&what, cut, "40x0");
            }
        }
        assert!(
            cut_count > 20 && planned > 15,
            "{kind}: {planned} of {cut_count}"
        );
    }
}

/// The seed of the decode target's corpus (fuzz/corpus/decode)
/// named `name`.
fn fuzz_seed(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz/corpus/decode")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn a_damaged_progressive_file_gets_one_answer_at_every_size() {
    // Issue #713: one of the progressive files a differential
    // run found libjpeg-turbo reading without a warning where
    // zune-jpeg stops, at the end of its first scan. It
    // rendered at the sizes that took the scaled decode and
    // failed at full size. The scaled decode now leaves it to
    // the full-size path, so every size gets zune-jpeg's
    // answer: the error.
    let bytes = fuzz_seed("jpeg-progressive-damaged-scan");
    let look = glance(&bytes).unwrap();
    assert!(look.progressive);
    for denominator in DENOMINATORS {
        let scaled = ScaledDecode {
            denominator,
            stored: look.size,
            orientation: None,
        };
        assert!(decode(&bytes, &scaled).is_some(), "1/{denominator}");
    }
    assert_eq!(first_scan::marker_in_first_scan(&bytes), Some(false));
    let full = answer(at_full_size(|| render(&bytes, "97x0")));
    assert!(
        full.as_ref()
            .is_err_and(|e| e.starts_with("Decode(") && e.contains("Marker missing")),
        "{full:?}"
    );
    for remainder in ["12x8", "16x0", "32x0", "49x0", "97x0", "crop:20x20", "0x0"] {
        assert_eq!(plan_for(&bytes, remainder), None, "{remainder}");
        assert_eq!(answer(render(&bytes, remainder)), full, "{remainder}");
    }
}

/// The data of a progressive file's first scan: from the end
/// of its header to the marker that ends it.
fn first_scan_data(bytes: &[u8]) -> std::ops::Range<usize> {
    let start = header_segments(bytes).last().unwrap().2;
    let end = (start..bytes.len() - 1)
        .find(|&at| bytes[at] == 0xFF && bytes[at + 1] != 0x00)
        .expect("a marker after the first scan");
    start..end
}

/// Whether zune-jpeg refuses `bytes` because its reader did
/// not come to the marker that ends the first scan.
fn zune_misses_the_marker(bytes: &[u8]) -> bool {
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
        zune_core::bytestream::ZCursor::new(bytes),
        zune_jpeg_options(),
    );
    decoder
        .decode()
        .is_err_and(|e| format!("{e:?}").contains("Marker missing where expected"))
}

#[test]
fn the_first_scan_is_read_as_zune_jpeg_reads_it() {
    // Issue #713. zune-jpeg refuses a progressive file whose
    // first scan its reader is done with before it comes to
    // the marker that ends the scan, and the scaled decode
    // leaves those to the full-size path (see `first_scan`).
    // The files here are damaged where that turns: bytes
    // added at the end of the first scan's data, which a
    // reader that fills four bytes at a time may or may not
    // come past, and each bit of the scan's data flipped in
    // turn, which makes codes read shorter or longer. Wherever
    // the reading follows the scan, it gives zune-jpeg's own
    // answer; and a request gets one answer at every size,
    // the scaled decode's or the full-size path's.
    let (mut comes, mut misses, mut unread, mut rendered) = (0, 0, 0, 0);
    for (width, height) in [(16, 16), (40, 24), (61, 37)] {
        let img = picture(width, height);
        let remainder = format!("{}x0", width / 3);
        for subsamp in [
            turbojpeg::Subsamp::None,
            turbojpeg::Subsamp::Sub2x1,
            turbojpeg::Subsamp::Sub2x2,
            turbojpeg::Subsamp::Sub1x2,
            turbojpeg::Subsamp::Sub4x1,
        ] {
            let bytes = jpeg_with(&img, |c| {
                c.set_subsamp(subsamp).unwrap();
                c.set_progressive(true).unwrap();
            });
            let kind = format!("{width}x{height} {subsamp:?}");
            assert_eq!(
                first_scan::marker_in_first_scan(&bytes),
                Some(true),
                "{kind}"
            );
            assert!(plan_for(&bytes, &remainder).is_some(), "{kind}");
            let data = first_scan_data(&bytes);
            let mut damaged = Vec::new();
            for extra in 1..=9 {
                for fill in [&[0x00][..], &[0x5A], &[0xFF, 0x00]] {
                    let mut longer = bytes[..data.end].to_vec();
                    for _ in 0..extra {
                        longer.extend_from_slice(fill);
                    }
                    longer.extend_from_slice(&bytes[data.end..]);
                    damaged.push((format!("{kind}, {extra} x {fill:02X?} added"), longer));
                }
            }
            for at in data.clone() {
                for bit in 0..8 {
                    let mut flipped = bytes.clone();
                    flipped[at] ^= 1 << bit;
                    damaged.push((format!("{kind}, byte {at} bit {bit} flipped"), flipped));
                }
            }
            for (n, (what, file)) in damaged.iter().enumerate() {
                match first_scan::marker_in_first_scan(file) {
                    Some(reached) => {
                        assert_eq!(reached, !zune_misses_the_marker(file), "{what}");
                        if reached {
                            comes += 1;
                        } else {
                            misses += 1;
                        }
                    }
                    None => unread += 1,
                }
                // Each added run, and every seventh flip:
                // rendered on both paths.
                if n < 27 || n % 7 == 0 {
                    assert_survives(what, file, &remainder);
                    rendered += 1;
                }
            }
        }
    }
    assert!(
        comes > 2000 && misses > 150 && unread * 4 < comes + misses && rendered > 600,
        "{comes} come to the marker, {misses} miss it, {unread} unread, {rendered} rendered"
    );
}

/// Entropy-coded bits, the first in the highest bit of each
/// byte, with a zero stuffed after each 0xFF.
#[derive(Default)]
struct BitWriter {
    out: Vec<u8>,
    held: u32,
    count: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, length: u32) {
        for bit in (0..length).rev() {
            self.held = self.held << 1 | (value >> bit & 1);
            self.count += 1;
            if self.count == 8 {
                self.out.push(self.held as u8);
                if self.held == 0xFF {
                    self.out.push(0x00);
                }
                (self.held, self.count) = (0, 0);
            }
        }
    }

    /// The bytes, the last one padded with ones.
    fn finish(mut self) -> Vec<u8> {
        while self.count != 0 {
            self.put(1, 1);
        }
        self.out
    }
}

/// A progressive JPEG of DC scans only, `width` by `height`,
/// with luma sampled `luma` (chroma at 1x1), and the DC table
/// of the JPEG standard (Annex K.3). Its scans carry the
/// components `scans` names in turn (1 is luma, 2 and 3
/// chroma): a scan of one component walks that component's
/// own blocks, and one of more walks the frame's MCUs. The DC
/// values vary from block to block, so each code is 2 to 9
/// bits with up to 11 more after it.
fn dc_only_progressive(width: u16, height: u16, luma: (u8, u8), scans: &[&[u8]]) -> Vec<u8> {
    const BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    // The code, and its length, of each size of difference.
    let mut codes = [(0u32, 0u32); 12];
    let (mut code, mut size) = (0u32, 0usize);
    for (length, count) in (1..=16u32).zip(BITS) {
        for _ in 0..count {
            codes[size] = (code, length);
            code += 1;
            size += 1;
        }
        code <<= 1;
    }
    let sampling = [luma, (1, 1), (1, 1)];
    let (h_max, v_max) = (u32::from(luma.0), u32::from(luma.1));
    let (w, h) = (u32::from(width), u32::from(height));
    let mut out = vec![0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x43, 0x00];
    out.extend_from_slice(&[1; 64]);
    out.extend_from_slice(&[0xFF, 0xC2, 0x00, 0x11, 8]);
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&width.to_be_bytes());
    out.push(3);
    for (id, (across, down)) in (1u8..).zip(sampling) {
        out.extend_from_slice(&[id, across << 4 | down, 0]);
    }
    out.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x1F, 0x00]);
    out.extend_from_slice(&BITS);
    out.extend(0..12u8);
    for scan in scans {
        out.extend_from_slice(&[0xFF, 0xDA, 0, 6 + 2 * scan.len() as u8, scan.len() as u8]);
        for id in *scan {
            out.extend_from_slice(&[*id, 0x00]);
        }
        out.extend_from_slice(&[0, 0, 0]);
        let mut blocks = Vec::new();
        if let [id] = scan {
            let (across, down) = sampling[usize::from(id - 1)];
            for row in 0..(h * u32::from(down)).div_ceil(8 * v_max) {
                for col in 0..(w * u32::from(across)).div_ceil(8 * h_max) {
                    blocks.push((*id, row, col));
                }
            }
        } else {
            for mcu_row in 0..h.div_ceil(8 * v_max) {
                for mcu_col in 0..w.div_ceil(8 * h_max) {
                    for id in *scan {
                        let (across, down) = sampling[usize::from(id - 1)];
                        let (across, down) = (u32::from(across), u32::from(down));
                        for v in 0..down {
                            for u in 0..across {
                                blocks.push((*id, mcu_row * down + v, mcu_col * across + u));
                            }
                        }
                    }
                }
            }
        }
        let mut bits = BitWriter::default();
        let mut previous = [0i32; 3];
        for (id, row, col) in blocks {
            let hash = (u32::from(id) * 7919 + row * 104_729 + col * 1_299_709)
                .wrapping_mul(2_654_435_761);
            let value = (hash >> 16) as i32 % 2001 - 1000;
            let diff = value - previous[usize::from(id - 1)];
            previous[usize::from(id - 1)] = value;
            let size = 32 - diff.unsigned_abs().leading_zeros();
            let (code, length) = codes[size as usize];
            bits.put(code, length);
            if size > 0 {
                let extra = if diff >= 0 { diff } else { diff - 1 };
                bits.put(extra as u32 & ((1 << size) - 1), size);
            }
        }
        out.extend(bits.finish());
    }
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

#[test]
fn a_first_scan_of_one_or_two_components_is_read_as_zune_jpeg_reads_it() {
    // Issue #713. libjpeg-turbo writes a progressive file's
    // first scan with all three components; mozjpeg, which
    // writes many of the progressive JPEGs served on the web,
    // gives each component a DC scan of its own by default, so
    // its first scan is luma alone, which walks the luma
    // blocks' own grid instead of the frame's MCUs. Such files
    // are written here in every chroma layout, at sizes off
    // the block grid: one DC scan per component, two
    // components interleaved and the third alone, and a chroma
    // component first. Each is read to its marker, decodes,
    // and takes the scaled decode; damaged after its first
    // scan's data or in a bit of it, the reading gives
    // zune-jpeg's own answer wherever it follows the scan.
    let (mut comes, mut misses, mut unread) = (0, 0, 0);
    let plans: [&[&[u8]]; 3] = [&[&[1], &[2], &[3]], &[&[1, 2], &[3]], &[&[2], &[1], &[3]]];
    for luma in [(1, 1), (2, 1), (2, 2), (1, 2), (4, 1)] {
        for (width, height) in [(16, 16), (40, 24), (61, 37), (97, 61)] {
            for scans in plans {
                let bytes = dc_only_progressive(width, height, luma, scans);
                let kind = format!("{width}x{height}, luma {luma:?}, scans {scans:?}");
                assert_eq!(
                    first_scan::marker_in_first_scan(&bytes),
                    Some(true),
                    "{kind}"
                );
                assert!(decode_image(&bytes).is_ok(), "{kind}");
                assert!(
                    plan_for(&bytes, &format!("{}x0", width / 3)).is_some(),
                    "{kind}"
                );
                let data = first_scan_data(&bytes);
                let mut damaged = Vec::new();
                for extra in 1..=9 {
                    for fill in [&[0x00][..], &[0x5A], &[0xFF, 0x00]] {
                        let mut longer = bytes[..data.end].to_vec();
                        for _ in 0..extra {
                            longer.extend_from_slice(fill);
                        }
                        longer.extend_from_slice(&bytes[data.end..]);
                        damaged.push((format!("{kind}, {extra} x {fill:02X?} added"), longer));
                    }
                }
                for at in data.clone() {
                    for bit in 0..8 {
                        let mut flipped = bytes.clone();
                        flipped[at] ^= 1 << bit;
                        damaged.push((format!("{kind}, byte {at} bit {bit} flipped"), flipped));
                    }
                }
                for (what, file) in &damaged {
                    match first_scan::marker_in_first_scan(file) {
                        Some(reached) => {
                            assert_eq!(reached, !zune_misses_the_marker(file), "{what}");
                            if reached {
                                comes += 1;
                            } else {
                                misses += 1;
                            }
                        }
                        None => unread += 1,
                    }
                }
            }
        }
    }
    assert!(
        comes > 2000 && misses > 150 && unread * 4 < comes + misses,
        "{comes} come to the marker, {misses} miss it, {unread} unread"
    );
}

#[test]
fn a_file_that_declares_another_size_than_it_carries_is_answered_as_before() {
    let img = picture(120, 80);
    for (kind, bytes) in [
        ("baseline", jpeg(&img)),
        ("progressive", progressive_jpeg(&img)),
    ] {
        for (width, height) in [
            (120, 160),
            (120, 81),
            (120, 40),
            (240, 80),
            (128, 80),
            (60, 80),
            (480, 320),
            (8, 8),
            (3, 3),
        ] {
            let what = format!("{kind} declaring {width}x{height}");
            let lying = declaring(&bytes, width, height);
            // A progressive file that declares fewer blocks
            // (16 by 16 pixels each, in 4:2:0) than it holds
            // leaves its reader short of the marker that ends
            // its first scan, and zune-jpeg refuses it: it is
            // not planned (issue #713), and every size gets
            // that refusal.
            let fewer = width.div_ceil(16) * height.div_ceil(16) < 8 * 5;
            if kind == "progressive" && fewer {
                assert_eq!(plan_for(&lying, "1x1"), None, "{what}");
                let refused = answer(at_full_size(|| render(&lying, "1x1")));
                assert!(
                    refused
                        .as_ref()
                        .is_err_and(|e| e.contains("Marker missing")),
                    "{what}: {refused:?}"
                );
            } else {
                assert!(plan_for(&lying, "1x1").is_some(), "{what}");
            }
            assert_survives(&what, &lying, "1x1");
        }
    }
    // A size the header cannot state: zero.
    for (width, height) in [(0, 80), (120, 0), (0, 0)] {
        let lying = declaring(&jpeg(&img), width, height);
        assert_eq!(plan_for(&lying, "1x1"), None, "{width}x{height}");
        assert_eq!(
            answer(render(&lying, "1x1")),
            answer(at_full_size(|| render(&lying, "1x1")))
        );
    }
}

/// A flat gray JPEG written by hand, so that its structure is
/// free: 4:4:4, every block a zero DC difference and nothing
/// else, which one-symbol Huffman tables code in one bit
/// each.
struct Flat {
    bytes: Vec<u8>,
}

impl Flat {
    /// The file up to its frame header: `frame` is 0xC0 for
    /// baseline and 0xC2 for progressive.
    fn start(frame: u8, width: u16, height: u16) -> Flat {
        let mut bytes = vec![0xFF, 0xD8];
        bytes.extend_from_slice(b"\xff\xe0\x00\x10JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
        bytes.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
        bytes.extend_from_slice(&[1; 64]);
        bytes.extend_from_slice(&[0xFF, frame, 0x00, 0x11, 8]);
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&[3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0]);
        Flat { bytes }
    }

    /// A Huffman table of one symbol, coded as the bit 0.
    /// `class` is 0 for a DC table and 1 for an AC table.
    fn table(mut self, class: u8, symbol: u8) -> Flat {
        self.bytes
            .extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, class << 4, 1]);
        self.bytes.extend_from_slice(&[0; 15]);
        self.bytes.push(symbol);
        self
    }

    /// A scan header over `components`, with its spectral
    /// band.
    fn scan(mut self, components: &[u8], band: (u8, u8)) -> Flat {
        self.bytes
            .extend_from_slice(&[0xFF, 0xDA, 0, 6 + 2 * components.len() as u8]);
        self.bytes.push(components.len() as u8);
        for component in components {
            self.bytes.extend_from_slice(&[*component, 0x00]);
        }
        self.bytes.extend_from_slice(&[band.0, band.1, 0x00]);
        self
    }

    /// `zeros` zero bits of entropy-coded data, padded to a
    /// whole byte with ones.
    fn zero_bits(mut self, zeros: usize) -> Flat {
        self.bytes.extend(std::iter::repeat_n(0x00, zeros / 8));
        if !zeros.is_multiple_of(8) {
            self.bytes.push(0xFFu8 >> (zeros % 8));
        }
        self
    }

    fn raw(mut self, bytes: &[u8]) -> Flat {
        self.bytes.extend_from_slice(bytes);
        self
    }

    fn end(mut self) -> Vec<u8> {
        self.bytes.extend_from_slice(&[0xFF, 0xD9]);
        self.bytes
    }
}

/// A flat gray baseline JPEG, `side` pixels square, with
/// `restart` blocks between restart markers when it is set.
/// `markers` writes the marker after each interval: the
/// number the standard asks for is its argument.
fn flat_baseline(side: u16, restart: Option<u16>, markers: impl Fn(u8) -> Vec<u8>) -> Vec<u8> {
    let blocks = usize::from(side / 8) * usize::from(side / 8);
    let mut flat = Flat::start(0xC0, side, side).table(0, 0x00).table(1, 0x00);
    if let Some(interval) = restart {
        flat = flat
            .raw(&[0xFF, 0xDD, 0x00, 0x04])
            .raw(&interval.to_be_bytes());
    }
    flat = flat.scan(&[1, 2, 3], (0, 63));
    // Each block of each component is two zero bits: no DC
    // difference, then the end of the block.
    match restart {
        None => flat = flat.zero_bits(blocks * 6),
        Some(interval) => {
            let interval = usize::from(interval);
            let mut done = 0;
            let mut number = 0u8;
            while done < blocks {
                let run = interval.min(blocks - done);
                flat = flat.zero_bits(run * 6);
                done += run;
                if done < blocks {
                    flat = flat.raw(&markers(number));
                    number = (number + 1) % 8;
                }
            }
        }
    }
    flat.end()
}

/// A flat gray progressive JPEG, 16x16, of `scans` scans: the
/// DC scan, then one scan for every AC coefficient of every
/// component (189 of them, each coefficient sent once, so
/// the progression is a legal one), then the same again for
/// as long as `scans` asks.
fn flat_progressive(scans: usize) -> Vec<u8> {
    assert!(scans >= 1);
    // One block's DC difference of zero is the symbol 0. An
    // AC scan over four blocks with nothing in them is one
    // end-of-band run of four: the symbol 0x20 and two zero
    // bits.
    let mut flat = Flat::start(0xC2, 16, 16)
        .table(0, 0x00)
        .table(1, 0x20)
        .scan(&[1, 2, 3], (0, 0))
        .zero_bits(12);
    for scan in 0..scans - 1 {
        let component = 1 + (scan / 63 % 3) as u8;
        let coefficient = 1 + (scan % 63) as u8;
        flat = flat
            .scan(&[component], (coefficient, coefficient))
            .zero_bits(3);
    }
    flat.end()
}

#[test]
fn the_files_written_by_hand_are_files_both_decoders_read() {
    let gray = image::Rgb([128, 128, 128]);
    for (what, bytes, side) in [
        ("baseline", flat_baseline(64, None, |_| Vec::new()), 64),
        (
            "baseline with restart markers",
            flat_baseline(64, Some(3), |n| vec![0xFF, 0xD0 + n]),
            64,
        ),
        ("progressive", flat_progressive(100), 16),
    ] {
        let full = decode_image(&bytes)
            .unwrap_or_else(|e| panic!("{what}: {e:?}"))
            .to_rgb8();
        assert_eq!((full.width(), full.height()), (side, side), "{what}");
        assert!(full.pixels().all(|p| *p == gray), "{what}");
        let theirs = turbojpeg::decompress(&bytes, turbojpeg::PixelFormat::RGB)
            .unwrap_or_else(|e| panic!("{what}: libjpeg-turbo: {e}"));
        assert!(theirs.pixels.iter().all(|v| *v == 128), "{what}");
        for denominator in DENOMINATORS {
            let frame = decode_at(&bytes, denominator)
                .unwrap_or_else(|| panic!("{what} at 1/{denominator}"))
                .to_rgb8();
            assert_eq!(frame.width(), side.div_ceil(denominator), "{what}");
            assert!(frame.pixels().all(|p| *p == gray), "{what}");
        }
    }
}

#[test]
fn a_progressive_file_past_the_scan_limit_is_refused_as_zune_jpeg_refuses_it() {
    // At the limit both decoders read the file, and it takes
    // the scaled decode.
    let at_limit = flat_progressive(100);
    assert_eq!(scale_for(&at_limit, "1x1"), Some(8));
    let frame = decode(&at_limit, &plan_for(&at_limit, "1x1").unwrap());
    assert!(frame.is_some(), "100 scans");
    assert!(decode_image(&at_limit).is_ok(), "100 scans at full size");
    let out = render(&at_limit, "1x1").unwrap();
    assert_eq!((out.width, out.height), (1, 1));
    // One scan more and both refuse it. The scaled decode
    // stops at the scan past the limit, and the request gets
    // the full-size path's refusal: zune-jpeg's.
    for scans in [101, 190, 191, 3000] {
        let bytes = flat_progressive(scans);
        let what = format!("{scans} scans");
        assert_refused_and_answered_as_before(&what, &bytes, "1x1");
        let refused = answer(render(&bytes, "1x1"));
        assert!(
            refused
                .as_ref()
                .is_err_and(|e| e.starts_with("Decode(") && e.contains("Too many scans")),
            "{what}: {refused:?}"
        );
    }
    // Without its limit libjpeg-turbo reads all of them: the
    // limit is what refuses the file, not its structure.
    assert!(turbojpeg::decompress(&flat_progressive(191), turbojpeg::PixelFormat::RGB).is_ok());
}

#[test]
fn restart_markers_out_of_turn_are_answered_as_before() {
    let clean = flat_baseline(64, Some(3), |n| vec![0xFF, 0xD0 + n]);
    assert_eq!(scale_for(&clean, "8x8"), Some(4));
    let mut frames = 0;
    for (what, bytes) in [
        (
            "every marker numbered 0",
            flat_baseline(64, Some(3), |_| vec![0xFF, 0xD0]),
        ),
        (
            "markers counting down",
            flat_baseline(64, Some(3), |n| vec![0xFF, 0xD7 - n]),
        ),
        (
            "an interval declared and no markers",
            flat_baseline(64, Some(3), |_| Vec::new()),
        ),
        (
            "markers twice over",
            flat_baseline(64, Some(3), |n| vec![0xFF, 0xD0 + n, 0xFF, 0xD0 + n]),
        ),
        (
            "a thousand markers after each interval",
            flat_baseline(64, Some(3), |n| [0xFF, 0xD0 + n].repeat(1000)),
        ),
        (
            "fill bytes ahead of each marker",
            flat_baseline(64, Some(3), |n| {
                let mut fill = vec![0xFF; 500];
                fill.push(0xD0 + n);
                fill
            }),
        ),
        (
            "a marker after every block",
            flat_baseline(64, Some(1), |n| vec![0xFF, 0xD0 + n]),
        ),
        (
            "an interval longer than the image",
            flat_baseline(64, Some(u16::MAX), |n| vec![0xFF, 0xD0 + n]),
        ),
        ("an interval of zero with markers in the data", {
            // The interval rewritten to zero after the file
            // was laid out with markers every three blocks.
            let mut bytes = clean.clone();
            let at = header_segments(&bytes)
                .into_iter()
                .find(|(marker, ..)| *marker == 0xDD)
                .unwrap()
                .1;
            bytes[at + 4..at + 6].copy_from_slice(&[0, 0]);
            bytes
        }),
        ("an end marker where a restart belongs", {
            flat_baseline(64, Some(3), |n| {
                if n == 2 {
                    vec![0xFF, 0xD9]
                } else {
                    vec![0xFF, 0xD0 + n]
                }
            })
        }),
    ] {
        assert!(plan_for(&bytes, "8x8").is_some(), "{what}");
        frames += usize::from(assert_survives(what, &bytes, "8x8"));
    }
    // Most of these libjpeg-turbo warns about; whichever it
    // reads without a word, the full-size path reads too.
    assert!(frames <= 4, "{frames} files decoded at scale");
}

#[test]
fn damaged_tables_and_damaged_data_are_answered_as_before() {
    let img = picture(120, 80);
    let baseline = jpeg(&img);
    let progressive = progressive_jpeg(&img);
    let (mut files, mut frames) = (0, 0);
    let mut check = |what: String, bytes: &[u8]| {
        files += 1;
        frames += usize::from(assert_survives(&what, bytes, "40x0"));
    };
    for (kind, bytes) in [("baseline", &baseline), ("progressive", &progressive)] {
        // Every Huffman table, in turn, declared with more
        // codes than its lengths can hold, with no codes at
        // all, with too many one-bit codes, and with symbols
        // out of range.
        let tables: Vec<usize> = bytes
            .windows(2)
            .enumerate()
            .filter(|(_, pair)| *pair == [0xFF, 0xC4])
            .map(|(at, _)| at)
            .collect();
        assert!(!tables.is_empty(), "{kind}: no Huffman tables");
        for (n, table) in tables.iter().enumerate() {
            let counts = table + 5;
            let mut overfull = bytes.to_vec();
            overfull[counts..counts + 16].fill(0xFF);
            check(format!("{kind} table {n} overfull"), &overfull);
            let mut empty = bytes.to_vec();
            empty[counts..counts + 16].fill(0);
            check(format!("{kind} table {n} empty"), &empty);
            let mut skewed = bytes.to_vec();
            skewed[counts] = 3;
            check(
                format!("{kind} table {n} with three one-bit codes"),
                &skewed,
            );
            let mut wild = bytes.to_vec();
            wild[counts + 16..counts + 18].fill(0xFF);
            check(format!("{kind} table {n} with wild symbols"), &wild);
        }
        // The quantization table zeroed, and declared 16-bit.
        let quant = header_segments(bytes)
            .into_iter()
            .find(|(marker, ..)| *marker == 0xDB)
            .unwrap()
            .1;
        let mut zeroed = bytes.to_vec();
        zeroed[quant + 5..quant + 69].fill(0);
        check(format!("{kind} zero quantizers"), &zeroed);
        let mut wide = bytes.to_vec();
        wide[quant + 4] |= 0x10;
        check(format!("{kind} 16-bit quantizers"), &wide);
        // The entropy-coded data: bytes flipped, zeroed, and
        // a marker dropped into the middle of a scan.
        let scan = header_segments(bytes).last().unwrap().2;
        for at in (scan..bytes.len() - 2).step_by((bytes.len() - scan) / 11) {
            let mut flipped = bytes.to_vec();
            flipped[at] ^= 0x5A;
            check(format!("{kind} byte {at} flipped"), &flipped);
            let mut zero_run = bytes.to_vec();
            zero_run[at..(at + 40).min(bytes.len() - 2)].fill(0);
            check(format!("{kind} zeros from {at}"), &zero_run);
            let mut marker = bytes.to_vec();
            marker[at..at + 2].copy_from_slice(&[0xFF, 0xC4]);
            check(format!("{kind} a table marker at {at}"), &marker);
        }
        // Bytes between the last scan and the end marker.
        let mut trailing = bytes[..bytes.len() - 2].to_vec();
        trailing.extend_from_slice(&[0x12; 300]);
        trailing.extend_from_slice(&[0xFF, 0xD9]);
        check(
            format!("{kind} stray bytes before the end marker"),
            &trailing,
        );
    }
    // Some damage reads cleanly, as damage does: such a frame
    // holds the picture the full-size path holds. Most does
    // not, and takes the full-size path.
    assert!(files > 80, "{files} damaged files");
    assert!(frames * 2 < files, "{frames} of {files} decoded at scale");
}

/// `bytes` without the header segments `drop` picks out, by
/// marker and by position among the segments.
fn without_segments(bytes: &[u8], drop: impl Fn(u8, usize) -> bool) -> Vec<u8> {
    let segments = header_segments(bytes);
    let mut out = bytes[..2].to_vec();
    for (index, (marker, start, end)) in segments.iter().enumerate() {
        if !drop(*marker, index) {
            out.extend_from_slice(&bytes[*start..*end]);
        }
    }
    out.extend_from_slice(&bytes[segments.last().unwrap().2..]);
    out
}

#[test]
fn a_file_that_leaves_its_huffman_tables_out_is_refused_as_before() {
    let whole = jpeg(&picture(120, 80));
    assert_eq!(scale_for(&whole, "40x0"), Some(2));
    let read = |bytes: &[u8]| turbojpeg::decompress(bytes, turbojpeg::PixelFormat::RGB);
    let tables: Vec<usize> = header_segments(&whole)
        .iter()
        .enumerate()
        .filter(|(_, (marker, ..))| *marker == 0xC4)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(tables.len(), 4, "two DC and two AC tables, a segment each");
    // No tables at all, as a frame of motion JPEG is written,
    // and each table left out in turn. libjpeg fills in the
    // standard tables, which are the ones this file was
    // written with, and reads the same picture. zune-jpeg
    // refuses the file, so it is a bad source at every size.
    let mut files = vec![(
        "no tables".to_string(),
        without_segments(&whole, |m, _| m == 0xC4),
    )];
    for (n, table) in tables.iter().enumerate() {
        files.push((
            format!("table {n} left out"),
            without_segments(&whole, |_, index| index == *table),
        ));
    }
    // A table defined in a slot the scan does not name.
    let (_, first_table, _) = header_segments(&whole)[tables[0]];
    let mut misplaced = whole.clone();
    assert_eq!(misplaced[first_table + 4], 0x00);
    misplaced[first_table + 4] = 0x02;
    files.push(("the first table in slot 2".to_string(), misplaced));
    for (what, bytes) in &files {
        let theirs = read(bytes).unwrap_or_else(|e| panic!("{what}: libjpeg-turbo: {e}"));
        assert_eq!(theirs.pixels, read(&whole).unwrap().pixels, "{what}");
        assert_eq!(eligible(bytes), None, "{what}");
        for remainder in ["40x0", "crop:20x20"] {
            let refused = answer(render(bytes, remainder));
            assert!(
                refused.as_ref().is_err_and(|e| e.starts_with("Decode(")),
                "{what}, {remainder}: {refused:?}"
            );
            assert_eq!(
                refused,
                answer(at_full_size(|| render(bytes, remainder))),
                "{what}, {remainder}"
            );
        }
    }
    // A progressive file gets no tables filled in: one that
    // leaves them out is refused by libjpeg-turbo too. The
    // reading of its first scan (issue #713) finds no table
    // to read it with, so it is not planned at all, and the
    // full-size path answers.
    let progressive = progressive_jpeg(&picture(120, 80));
    let bare = without_segments(&progressive, |m, _| m == 0xC4);
    assert!(read(&bare).is_err());
    assert_eq!(first_scan::marker_in_first_scan(&bare), None);
    assert_eq!(plan_for(&bare, "40x0"), None);
    assert_eq!(
        answer(render(&bare, "40x0")),
        answer(at_full_size(|| render(&bare, "40x0"))),
        "a progressive file with no tables"
    );
    // A sequential file whose first scan holds one component
    // has other scans behind it, which the header does not
    // show: full size.
    let blocks = 64;
    let scans = Flat::start(0xC0, 64, 64)
        .table(0, 0x00)
        .table(1, 0x00)
        .scan(&[1], (0, 63))
        .zero_bits(blocks * 2)
        .scan(&[2], (0, 63))
        .zero_bits(blocks * 2)
        .scan(&[3], (0, 63))
        .zero_bits(blocks * 2)
        .end();
    assert!(read(&scans).is_ok(), "libjpeg-turbo reads the three scans");
    assert_eq!(eligible(&scans), None);
    assert_eq!(
        answer(render(&scans, "8x8")),
        answer(at_full_size(|| render(&scans, "8x8")))
    );
}

#[test]
fn the_header_walk_stands_down_for_what_it_does_not_expect() {
    let whole = jpeg(&picture(40, 24));
    assert_eq!(glance(&whole).map(|g| g.names_its_tables), Some(true));
    // Fill bytes ahead of a marker are part of the format.
    let (_, quantizers, _) = header_segments(&whole)[1];
    let mut filled = whole[..quantizers].to_vec();
    filled.extend_from_slice(&[0xFF; 7]);
    filled.extend_from_slice(&whole[quantizers..]);
    assert_eq!(glance(&filled).map(|g| g.names_its_tables), Some(true));
    // No scan, a header cut anywhere, and bytes that are not
    // a marker where one belongs.
    let scan = header_segments(&whole).last().unwrap().1;
    for end in 0..scan + 6 {
        assert_eq!(
            glance(&whole[..end]).map(|g| g.names_its_tables),
            None,
            "cut at {end}"
        );
    }
    let mut stray = whole.clone();
    stray[quantizers] = 0x00;
    assert_eq!(glance(&stray).map(|g| g.names_its_tables), None);
    // A segment that states less than its own length field,
    // and one that runs past the file.
    let mut short = whole.clone();
    short[quantizers + 2..quantizers + 4].copy_from_slice(&[0, 1]);
    assert_eq!(glance(&short).map(|g| g.names_its_tables), None);
    let mut long = whole.clone();
    long[quantizers + 2..quantizers + 4].copy_from_slice(&[0xFF, 0xFF]);
    assert_eq!(glance(&long).map(|g| g.names_its_tables), None);
    // A table in a class or a slot the format does not have,
    // and one cut short by its segment.
    let table = header_segments(&whole)
        .into_iter()
        .find(|(marker, ..)| *marker == 0xC4)
        .unwrap()
        .1;
    for id in [0x20, 0x04, 0xFF] {
        let mut wild = whole.clone();
        wild[table + 4] = id;
        assert_eq!(
            glance(&wild).map(|g| g.names_its_tables),
            None,
            "table id {id:#x}"
        );
    }
    let mut cut = whole.clone();
    cut[table + 5] = 0xFF;
    assert_eq!(glance(&cut).map(|g| g.names_its_tables), None);
    // A marker that stands alone, ahead of any scan.
    let mut lone = whole[..2].to_vec();
    lone.extend_from_slice(&[0xFF, 0xD0]);
    lone.extend_from_slice(&whole[2..]);
    assert_eq!(glance(&lone).map(|g| g.names_its_tables), None);
    // A scan that names a slot nothing defined.
    let mut unnamed = whole.clone();
    unnamed[scan + 6] = 0x33;
    assert_eq!(glance(&unnamed).map(|g| g.names_its_tables), Some(false));
    assert_eq!(glance(b"\xff\xd8"), None);
}

#[test]
fn the_first_look_reads_the_frame_and_the_orientation_as_the_decoders_do() {
    // The walk ahead of the decoders reads the size, the
    // coding and the EXIF orientation that the two decoders
    // and the EXIF reader of the decode stage read, for every
    // orientation, in both byte orders, baseline and
    // progressive.
    let img = picture(97, 61);
    for (bytes, progressive) in [(jpeg(&img), false), (progressive_jpeg(&img), true)] {
        let upright = glance(&bytes).unwrap();
        assert_eq!(
            (upright.size, upright.progressive, upright.orientation),
            ((97, 61), progressive, None)
        );
        assert_eq!(eligible(&bytes), Some((97, 61)));
        for orientation in 1..=8u16 {
            for turned in [
                with_orientation(&bytes, orientation),
                with_big_endian_orientation(&bytes, orientation),
            ] {
                let expected = Some(u32::from(orientation));
                assert_eq!(glance(&turned).unwrap().orientation, expected);
                assert_eq!(super::super::exif_orientation(&turned), expected);
            }
        }
        // An EXIF segment with no directory to read leaves the
        // look without an orientation; a request on such a
        // file is then judged upright by the look, and decided
        // by the decoders and the EXIF reader.
        let unreadable = with_segment(&bytes, 0xE1, b"Exif\0\0II*\0\xff\xff\xff\x7f");
        assert_eq!(glance(&unreadable).unwrap().orientation, None);
    }
    // Every file the look turns down, with both decoders
    // turning it down after it.
    let mut twelve = jpeg(&img);
    let at = frame_header(&twelve);
    twelve[at] = 12;
    for (what, bytes) in [
        (
            "grayscale",
            jpeg_with(&img, |c| c.set_subsamp(turbojpeg::Subsamp::Gray).unwrap()),
        ),
        (
            "CMYK",
            four_component_jpeg(97, 61, turbojpeg::Colorspace::CMYK),
        ),
        (
            "YCCK",
            four_component_jpeg(97, 61, turbojpeg::Colorspace::YCCK),
        ),
        ("12-bit", twelve),
        (
            "arithmetic",
            jpeg_with(&img, |c| c.set_arithmetic(true).unwrap()),
        ),
        (
            "lossless",
            jpeg_with(&img, |c| c.set_lossless(true).unwrap()),
        ),
        ("a zero height", declaring(&jpeg(&img), 97, 0)),
    ] {
        assert_eq!(glance(&bytes), None, "{what}");
    }
}

#[test]
fn a_request_the_first_look_turns_down_leaves_the_header_to_the_full_size_decode() {
    // A phone photo stored 240x160 and shown 160x240: 80 wide
    // is half its width, so the look turns it down from the
    // orientation it reads, and neither decoder reads the
    // header ahead of the full-size decode (see `Glance`).
    // Read upright, 80 wide would have been a third.
    let plain = jpeg(&picture(240, 160));
    let phone = with_orientation(&plain, 6);
    let header_reads = || HEADER_READS.get();
    let before = header_reads();
    assert_eq!(scale_for(&phone, "80x0"), None);
    assert_eq!(scale_for(&phone, "100x0"), None);
    assert_eq!(scale_for(&plain, "100x0"), None);
    assert_eq!(header_reads(), before, "a request too large for any scale");
    // Nor does any request the look turns down by the file.
    let gray = jpeg_with(&picture(240, 160), |c| {
        c.set_subsamp(turbojpeg::Subsamp::Gray).unwrap()
    });
    assert_eq!(scale_for(&gray, "20x0"), None);
    assert_eq!(header_reads(), before, "a grayscale file");
    // 50 wide is a third of the phone photo's width and 80 a
    // third of the plain one's: the decoders read the header,
    // and both requests decode at reduced scale.
    assert_eq!(scale_for(&phone, "50x0"), Some(2));
    assert_eq!(header_reads(), before + 1);
    assert_eq!(scale_for(&plain, "80x0"), Some(2));
    assert_eq!(header_reads(), before + 2);
}

#[test]
fn a_warning_is_a_refusal_and_the_full_size_decode_answers() {
    // libjpeg-turbo reads these three to the end, hands back
    // every pixel it could make, and warns. Here that is a
    // refusal: the full-size path renders the file exactly as
    // it did, and that render is the response.
    let bytes = jpeg(&picture(120, 80));
    let stray = {
        let mut stray = bytes[..bytes.len() - 2].to_vec();
        stray.extend_from_slice(&[0x12; 64]);
        stray.extend_from_slice(&[0xFF, 0xD9]);
        stray
    };
    let short = &bytes[..bytes.len() - 200];
    let no_end = &bytes[..bytes.len() - 2];
    for (what, file) in [
        ("stray bytes before the end marker", stray.as_slice()),
        ("a scan cut short", short),
        ("no end marker", no_end),
    ] {
        let warned = turbojpeg::decompress(file, turbojpeg::PixelFormat::RGB);
        assert!(warned.is_err(), "{what}: libjpeg-turbo said nothing");
        assert_refused_and_answered_as_before(what, file, "40x0");
        // And the full-size path does render it.
        let out = render(file, "40x0").unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert_eq!((out.width, out.height), (40, 26), "{what}");
    }
}

#[test]
fn codings_only_libjpeg_turbo_reads_are_refused_as_before() {
    // libjpeg-turbo decodes arithmetic-coded and lossless
    // files. zune-jpeg does not, so such a file was always a
    // bad source; it stays one whatever the request asks for.
    let img = picture(120, 80);
    for (what, bytes) in [
        (
            "arithmetic",
            jpeg_with(&img, |c| c.set_arithmetic(true).unwrap()),
        ),
        (
            "progressive arithmetic",
            jpeg_with(&img, |c| {
                c.set_arithmetic(true).unwrap();
                c.set_progressive(true).unwrap();
            }),
        ),
        (
            "lossless",
            jpeg_with(&img, |c| c.set_lossless(true).unwrap()),
        ),
    ] {
        assert!(
            turbojpeg::decompress(&bytes, turbojpeg::PixelFormat::RGB).is_ok(),
            "{what}: libjpeg-turbo reads it"
        );
        for remainder in ["40x0", "crop:20x20", ""] {
            let refused = answer(render(&bytes, remainder));
            assert!(
                refused.as_ref().is_err_and(|e| e.starts_with("Decode(")),
                "{what}, {remainder:?}: {refused:?}"
            );
            assert_eq!(refused, answer(at_full_size(|| render(&bytes, remainder))));
        }
    }
    // A header that declares 12-bit samples.
    let mut twelve = jpeg(&img);
    let at = frame_header(&twelve);
    twelve[at] = 12;
    let refused = answer(render(&twelve, "40x0"));
    assert!(refused.as_ref().is_err_and(|e| e.starts_with("Decode(")));
    assert_eq!(refused, answer(at_full_size(|| render(&twelve, "40x0"))));
}

#[test]
fn grayscale_and_four_component_sources_render_as_before() {
    let img = picture(120, 80);
    let gray = jpeg_with(&img, |c| c.set_subsamp(turbojpeg::Subsamp::Gray).unwrap());
    let cmyk = four_component_jpeg(120, 80, turbojpeg::Colorspace::CMYK);
    let ycck = four_component_jpeg(120, 80, turbojpeg::Colorspace::YCCK);
    for (what, bytes) in [("grayscale", &gray), ("CMYK", &cmyk), ("YCCK", &ycck)] {
        for remainder in ["40x0", "20x0/filters:format(png)", "10x0"] {
            let ours = answer(render(bytes, remainder));
            assert!(ours.is_ok(), "{what}, {remainder}: {ours:?}");
            assert_eq!(
                ours,
                answer(at_full_size(|| render(bytes, remainder))),
                "{what}, {remainder}"
            );
        }
    }
}

#[test]
fn tiny_sources_render_at_the_sizes_they_did() {
    for (width, height) in [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (3, 3),
        (4, 3),
        (7, 5),
        (8, 8),
        (9, 9),
        (16, 1),
    ] {
        let bytes = jpeg(&picture(width, height));
        for remainder in ["1x1", "1x0", "0x1", "2x0", "2x2", "3x3"] {
            let what = format!("{width}x{height} to {remainder}");
            let ours = answer(render(&bytes, remainder));
            let theirs = answer(at_full_size(|| render(&bytes, remainder)));
            assert!(ours.is_ok(), "{what}: {ours:?}");
            assert_eq!(sizes(&ours), sizes(&theirs), "{what}");
        }
    }
    // A one-pixel source has no smaller frame; three pixels
    // have a two-pixel one, which a one-pixel target takes.
    assert_eq!(scale_for(&jpeg(&picture(1, 1)), "1x1"), None);
    assert_eq!(scale_for(&jpeg(&picture(2, 2)), "1x1"), None);
    assert_eq!(scale_for(&jpeg(&picture(3, 3)), "1x1"), Some(2));
    assert_eq!(scale_for(&jpeg(&picture(9, 9)), "1x1"), Some(8));
}

// The fuzz corpus.

#[test]
fn the_fuzz_seeds_are_the_shapes_they_are_named_for() {
    // The decode target's JPEG seeds (fuzz/corpus/decode): the
    // files the scaled decode takes, and one for each way it
    // stands down. A seed that stopped being its shape would
    // leave that way unseeded.
    let seed = |name: &str| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fuzz/corpus/decode")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    };
    let frames = |bytes: &[u8]| {
        DENOMINATORS
            .into_iter()
            .filter(|d| decode_at(bytes, *d).is_some())
            .count()
    };
    // Decoded at every scale: baseline and progressive, the
    // last with a DC scan per component, as mozjpeg writes
    // them, so its first scan holds luma alone (issue #713).
    for name in [
        "jpeg-small",
        "jpeg-blue",
        "jpeg-progressive",
        "jpeg-progressive-scan-per-component",
    ] {
        assert_eq!(frames(&seed(name)), 3, "{name}");
    }
    assert_eq!(
        seed("jpeg-progressive-scan-per-component"),
        dc_only_progressive(61, 37, (2, 2), &[&[1], &[2], &[3]])
    );
    assert!(
        turbojpeg::read_header(&seed("jpeg-progressive"))
            .unwrap()
            .is_progressive
    );
    // Turned away by their headers, before any decode.
    for name in [
        "jpeg-gray",
        "jpeg-cmyk",
        "jpeg-arithmetic",
        "jpeg-lossless",
        "jpeg-12-bit",
        "jpeg-size-past-the-limit",
    ] {
        let bytes = seed(name);
        assert_eq!(eligible(&bytes), None, "{name}");
        assert_eq!(frames(&bytes), 0, "{name}");
    }
    let header = |name: &str| turbojpeg::read_header(&seed(name)).unwrap();
    assert_eq!(header("jpeg-gray").colorspace, turbojpeg::Colorspace::Gray);
    assert_eq!(header("jpeg-cmyk").colorspace, turbojpeg::Colorspace::CMYK);
    assert!(header("jpeg-arithmetic").is_arithmetic);
    assert!(header("jpeg-lossless").is_lossless);
    assert_eq!(seed("jpeg-12-bit")[frame_header(&seed("jpeg-12-bit"))], 12);
    let past = header("jpeg-size-past-the-limit");
    assert_eq!((past.width, past.height), (16_384, 4097));
    // Eligible by their headers, and refused by libjpeg-turbo:
    // a scan cut short, a scan past the limit, and restart
    // markers that count the wrong way.
    for name in [
        "jpeg-truncated",
        "jpeg-scans-past-the-limit",
        "jpeg-restart-out-of-turn",
    ] {
        let bytes = seed(name);
        assert!(eligible(&bytes).is_some(), "{name}");
        assert_eq!(frames(&bytes), 0, "{name}");
    }
    // A progressive file whose first scan zune-jpeg reads
    // short of its marker, and libjpeg-turbo reads without a
    // warning (issue #713): turned away before the decode, as
    // the full-size decode refuses it.
    let damaged = seed("jpeg-progressive-damaged-scan");
    assert!(glance(&damaged).unwrap().progressive);
    assert_eq!(first_scan::marker_in_first_scan(&damaged), Some(false));
    assert_eq!(eligible(&damaged), None);
    assert_eq!(frames(&damaged), 0);
    assert!(decode_image(&damaged).is_err());
    // The two written by hand are the files these tests write.
    assert_eq!(seed("jpeg-scans-past-the-limit"), flat_progressive(191));
    assert_eq!(
        seed("jpeg-restart-out-of-turn"),
        flat_baseline(64, Some(3), |n| vec![0xFF, 0xD7 - n])
    );
}

// The same bytes on every host.

/// One line per frame, its size and SHA-256, for a set of
/// scaled decodes that walks libjpeg-turbo's vector code:
/// widths on both sides of the 8, 16 and 32 pixel steps its
/// loops take, so every loop's tail runs; each chroma layout,
/// which decides the block sizes its inverse DCT runs at and
/// whether chroma is upsampled; baseline and progressive; and
/// the three scales.
fn frame_digest_lines() -> String {
    use sha2::Digest;
    let mut lines = String::new();
    let mut add = |img: &image::RgbImage, quality: i32| {
        for subsamp in [
            turbojpeg::Subsamp::None,
            turbojpeg::Subsamp::Sub2x1,
            turbojpeg::Subsamp::Sub2x2,
            turbojpeg::Subsamp::Sub1x2,
            turbojpeg::Subsamp::Sub4x1,
        ] {
            for progressive in [false, true] {
                let bytes = jpeg_with(img, |c| {
                    c.set_quality(quality).unwrap();
                    c.set_subsamp(subsamp).unwrap();
                    c.set_progressive(progressive).unwrap();
                });
                for denominator in DENOMINATORS {
                    let frame = decode_at(&bytes, denominator).expect("a frame");
                    lines.push_str(&format!(
                        "{}x{} quality {quality} {subsamp:?} progressive {progressive} at 1/{denominator}: {}x{} {}\n",
                        img.width(),
                        img.height(),
                        frame.width(),
                        frame.height(),
                        hex::encode(sha2::Sha256::digest(frame.as_bytes()))
                    ));
                }
            }
        }
    };
    for width in [
        1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 97, 129, 255,
    ] {
        add(&picture(width, 37), 85);
    }
    for quality in [1, 40, 100] {
        add(&picture(131, 93), quality);
    }
    lines
}

/// Not a test on its own: the process
/// `frames_are_the_same_with_and_without_vector_code` starts,
/// by re-running this test binary with one of libjpeg-turbo's
/// switches set, writes its digests where that test reads
/// them.
#[test]
#[ignore = "a helper the vector-code test runs as its child"]
fn frame_digests_helper() {
    let Ok(path) = std::env::var("CRABCROP_TEST_FRAME_DIGESTS") else {
        return;
    };
    std::fs::write(path, frame_digest_lines()).unwrap();
}

/// A scaled decode is the same bytes whatever the CPU offers,
/// so that two replicas on different hosts render one URL to
/// the same response. libjpeg-turbo picks its inverse DCT,
/// chroma upsampling and colour conversion per host: AVX2 or
/// SSE2 on x86-64, Neon on aarch64, plain C where it finds
/// neither. The set of decodes above is repeated in child
/// processes with the library's own switches set:
/// `JSIMD_FORCENONE` leaves it the plain C paths and
/// `JSIMD_FORCESSE2` withholds AVX2. (A switch a platform
/// does not know changes nothing there.) The digest of the
/// whole set is pinned as well: CI runs this test on x86-64
/// and on aarch64, under Linux and macOS, so it is also the
/// proof that those decode the same bytes.
#[test]
fn frames_are_the_same_with_and_without_vector_code() {
    use sha2::Digest;
    use std::process::{Command, Stdio};
    let here = frame_digest_lines();
    assert_eq!(here.lines().count(), 22 * 5 * 2 * 3);
    let dir = crate::test_support::scratch_dir("jpeg-frame-digests");
    for switch in ["JSIMD_FORCENONE", "JSIMD_FORCESSE2"] {
        let path = dir.join(switch);
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "engine::jpeg_scaled::tests::frame_digests_helper",
                "--ignored",
                "--test-threads=1",
                "--quiet",
            ])
            .env(switch, "1")
            .env("CRABCROP_TEST_FRAME_DIGESTS", &path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "{switch}: {status}");
        let there = std::fs::read_to_string(&path).expect("the child wrote its digests");
        assert_eq!(there, here, "the frames differ with {switch}=1");
    }
    assert_eq!(
        hex::encode(sha2::Sha256::digest(here.as_bytes())),
        "d37684112d316b4e2640e84ac28422ddca859d5c99044d0664929a4f7a7cffe3",
        "the scaled decode's frames changed:\n{here}"
    );
}
