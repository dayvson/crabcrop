//! The libwebp walk over animated WebP against the image
//! crate's frame iterator (issue #646): the same frames for
//! every animation, and the iterator's own answer for every
//! file the walk does not handle.
//!
//! The fixtures are in `tests/fixtures/animated-webp/`, written
//! by `make_fixtures.py` there; the two animations the
//! repository already carries join them, and two tests build a
//! couple of hundred more from frames libwebp encodes on the
//! spot.

use super::super::{
    EngineError, Frames, MAX_ANIMATED_FRAMES, MAX_DECODE_ALLOC, collect_frames, decode_limits,
    decode_webp_frames, open_webp_animation, webp_animation_params, webp_frames_through_libwebp,
};
use super::*;
use std::io::Cursor;
use std::path::PathBuf;

/// The frames the image crate's iterator yields: every decode
/// before issue #646, and the reference here.
fn by_image_crate(bytes: &[u8]) -> Result<Frames, EngineError> {
    use image::AnimationDecoder;
    let (decoder, _) = open_webp_animation(bytes)?;
    collect_frames(decoder.into_frames(), "WebP", None)
}

/// The frames of the walk alone. `None` when it hands the file
/// back to the image crate's iterator.
fn through_libwebp(bytes: &[u8]) -> Option<Result<Frames, EngineError>> {
    let (_, background) = open_webp_animation(bytes).expect("the image crate opens the file");
    webp_frames_through_libwebp(bytes, background, None)
}

/// Two decodes hold the same frames: as many, each the same
/// size, offset, delay and bytes. Returns how many.
fn assert_same_frames(what: &str, got: &Frames, want: &Frames) -> usize {
    assert_eq!(got.len(), want.len(), "{what}: frame count");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.buffer().dimensions(),
            w.buffer().dimensions(),
            "{what}: frame {i} size"
        );
        assert_eq!(
            (g.left(), g.top()),
            (w.left(), w.top()),
            "{what}: frame {i} offset"
        );
        assert_eq!(
            g.delay().numer_denom_ms(),
            w.delay().numer_denom_ms(),
            "{what}: frame {i} delay"
        );
        let (g, w) = (g.buffer().as_raw(), w.buffer().as_raw());
        if g != w {
            let differing = g.iter().zip(w).filter(|(a, b)| a != b).count();
            let most = g.iter().zip(w).map(|(a, b)| a.abs_diff(*b)).max();
            panic!("{what}: frame {i}: {differing} bytes differ, by up to {most:?}");
        }
    }
    got.len()
}

fn repo_file(path: &str) -> Vec<u8> {
    let full = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path);
    std::fs::read(&full).unwrap_or_else(|e| panic!("{}: {e}", full.display()))
}

fn fixture(name: &str) -> Vec<u8> {
    repo_file(&format!("tests/fixtures/animated-webp/{name}"))
}

/// Every checked-in animated WebP, by name: this module's
/// fixtures, then the two the repository already carries.
fn animations() -> Vec<(String, Vec<u8>)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/animated-webp");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".webp"))
        .collect();
    names.sort();
    let mut all: Vec<(String, Vec<u8>)> = names
        .into_iter()
        .map(|name| {
            let bytes = fixture(&name);
            (name, bytes)
        })
        .collect();
    all.push((
        "anim.webp".to_string(),
        repo_file("tests/functional/corpus/anim.webp"),
    ));
    all.push((
        "crabcrop-mascot.webp".to_string(),
        repo_file("assets/crabcrop-mascot.webp"),
    ));
    all
}

/// The checked-in animations: name, frames, canvas. What each
/// `w` file is for is written next to it in `make_fixtures.py`.
const ANIMATIONS: [(&str, usize, (u32, u32)); 22] = [
    ("p01_pillow_lossy.webp", 6, (96, 64)),
    ("p02_pillow_lossless.webp", 3, (96, 64)),
    ("p03_pillow_mixed_minsize.webp", 6, (96, 64)),
    ("p04_pillow_sprite_alpha.webp", 6, (80, 60)),
    ("p05_pillow_sprite_alpha_lossless.webp", 6, (80, 60)),
    ("w01_lossy_full_blend.webp", 3, (72, 54)),
    ("w02_lossy_alpha_sub_blend.webp", 4, (72, 54)),
    ("w03_lossy_sub_dispose_bg_red.webp", 5, (72, 54)),
    ("w04_lossless_alpha_sub_dispose_bg_semi.webp", 5, (72, 54)),
    ("w05_noblend.webp", 5, (72, 54)),
    ("w06_no_alpha_flag.webp", 3, (72, 54)),
    ("w07_lossless_rgb.webp", 4, (72, 54)),
    ("w08_mixed.webp", 8, (72, 54)),
    ("w09_tiny.webp", 4, (3, 3)),
    ("w10_first_frame_small_bg.webp", 3, (72, 54)),
    ("w11_full_dispose_then_frames.webp", 5, (72, 54)),
    ("w12_odd_canvas.webp", 7, (73, 55)),
    ("w13_lossless_alpha_without_the_hint.webp", 3, (72, 54)),
    ("w14_durations_and_loops.webp", 4, (16, 12)),
    ("w15_chunks_around_the_frames.webp", 3, (72, 54)),
    ("anim.webp", 12, (320, 213)),
    ("crabcrop-mascot.webp", 40, (400, 400)),
];

/// The first `limit` frames of an animation by each path: the
/// walk's, then the image crate's. The walk's are fewer where
/// it stops short of them.
fn leading_frames(bytes: &[u8], limit: usize) -> (Frames, Frames) {
    use image::AnimationDecoder;
    let (decoder, background) = open_webp_animation(bytes).expect("the image crate opens the file");
    let want = collect_frames(decoder.into_frames().take(limit), "WebP", None).unwrap();
    let mut animation = Animation::open(bytes, background).expect("a plain animation");
    let walked = std::iter::from_fn(|| animation.next_frame().ok().map(Ok)).take(limit);
    (collect_frames(walked, "WebP", None).unwrap(), want)
}

#[test]
fn every_checked_in_animation_decodes_to_the_image_crates_frames() {
    let all = animations();
    let names: Vec<&str> = all.iter().map(|(name, _)| name.as_str()).collect();
    let listed: Vec<&str> = ANIMATIONS.iter().map(|(name, ..)| *name).collect();
    assert_eq!(names, listed, "the fixtures on disk and the list here");
    let mut frames = 0;
    for ((name, bytes), (_, count, canvas)) in all.iter().zip(ANIMATIONS) {
        let opened = Animation::open(bytes, None).unwrap_or_else(|| panic!("{name}: not walked"));
        assert_eq!(
            (opened.frames as usize, (opened.width, opened.height)),
            (count, canvas),
            "{name}"
        );
        if name == "crabcrop-mascot.webp" {
            // Forty frames of 400x400 are seconds of decoding
            // in a debug build: the first ten.
            let (walked, want) = leading_frames(bytes, 10);
            assert_eq!(want.len(), 10, "{name}");
            frames += assert_same_frames(name, &walked, &want);
            continue;
        }
        let want = by_image_crate(bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((want.len(), want.dimensions()), (count, canvas), "{name}");
        // The walk handles the file to its last frame, so the
        // frames compared are its own, not the iterator's.
        let walked = through_libwebp(bytes)
            .unwrap_or_else(|| panic!("{name}: the walk handed the file back"))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        frames += assert_same_frames(name, &walked, &want);
        let decoded = decode_webp_frames(bytes, None).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_same_frames(name, &decoded, &want);
    }
    let listed: usize = ANIMATIONS.iter().map(|(_, n, _)| n).sum();
    assert_eq!(listed, 145);
    assert_eq!(frames, listed - 30);
}

#[test]
fn durations_and_loop_counts_come_through() {
    // Durations are the 24 bits the container has for them,
    // unclamped, and the loop count is read from the ANIM
    // chunk as before.
    let bytes = fixture("w14_durations_and_loops.webp");
    let frames = through_libwebp(&bytes).unwrap().unwrap();
    let delays: Vec<(u32, u32)> = frames.iter().map(|f| f.delay().numer_denom_ms()).collect();
    assert_eq!(delays, [(0, 1), (1, 1), (0xFF_FFFF, 1), (65_536, 1)]);
    assert_eq!(
        webp_animation_params(&bytes).map(|(loops, _)| loops),
        Some(65_535)
    );
    let bytes = fixture("w08_mixed.webp");
    assert_eq!(
        webp_animation_params(&bytes),
        Some((3, [255, 255, 255, 255]))
    );
    let delays: Vec<u32> = decode_webp_frames(&bytes, None)
        .unwrap()
        .iter()
        .map(|f| f.delay().numer_denom_ms().0)
        .collect();
    assert_eq!(delays, [30, 40, 50, 60, 70, 80, 90, 100]);
}

// ----------------------------------------------------------------
// Animations assembled here, from stills libwebp encodes.
// ----------------------------------------------------------------

/// A small deterministic generator (Knuth's MMIX constants).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }

    fn chance(&mut self, percent: u32) -> bool {
        self.below(100) < percent
    }
}

fn riff_chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    push_chunk(&mut out, fourcc, payload);
    out
}

/// The bitstream chunks of a still WebP file (`ALPH`, `VP8 `,
/// `VP8L`), as they sit in it: what an ANMF chunk carries.
fn bitstream_chunks(still: &[u8]) -> Vec<u8> {
    assert_eq!((&still[..4], &still[8..12]), (&b"RIFF"[..], &b"WEBP"[..]));
    let mut out = Vec::new();
    let mut at = 12;
    while let Some((fourcc, size, rounded)) = chunk_header(still, at) {
        if matches!(&fourcc, b"ALPH" | b"VP8 " | b"VP8L") {
            out.extend_from_slice(&riff_chunk(&fourcc, &still[at + 8..at + 8 + size]));
        }
        at += 8 + rounded;
    }
    assert!(!out.is_empty(), "a still WebP carries a bitstream");
    out
}

/// How a test frame is coded.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Coding {
    /// `VP8 `: lossy, no alpha.
    Lossy,
    /// `ALPH` and `VP8 `: lossy with an alpha plane (libwebp
    /// writes plain `VP8 ` when every pixel is opaque).
    LossyAlpha,
    /// `VP8L` from RGBA pixels.
    Lossless,
    /// `VP8L` from RGB pixels: the alpha hint unset.
    LosslessRgb,
}

/// The bitstream chunks of `rgba` (`w` by `h`) coded as a still
/// with libwebp's settings as `config` has them (`lossless` is
/// set here, from `coding`).
fn encode_still(
    rgba: &[u8],
    w: u32,
    h: u32,
    coding: Coding,
    mut config: webp::WebPConfig,
) -> Vec<u8> {
    let rgb: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    config.lossless = i32::from(matches!(coding, Coding::Lossless | Coding::LosslessRgb));
    let encoder = match coding {
        Coding::Lossy | Coding::LosslessRgb => webp::Encoder::from_rgb(&rgb, w, h),
        Coding::LossyAlpha | Coding::Lossless => webp::Encoder::from_rgba(rgba, w, h),
    };
    let still = encoder
        .encode_advanced(&config)
        .unwrap_or_else(|e| panic!("libwebp refused the test frame: {e:?}"));
    bitstream_chunks(&still)
}

/// libwebp's default settings at `quality`.
fn settings(quality: f32) -> webp::WebPConfig {
    let mut config = webp::WebPConfig::new().expect("libwebp's default settings");
    config.quality = quality;
    config
}

/// Encoder settings drawn so that the frames meet the decoders
/// widely: every method, one to four segments, both loop
/// filters at any strength and sharpness, one to eight token
/// partitions, each alpha filter with and without alpha
/// compression, sharp chroma, near-lossless and exact modes.
fn random_settings(rng: &mut Rng) -> webp::WebPConfig {
    let mut config = settings(rng.below(101) as f32);
    config.method = rng.below(7) as i32;
    config.segments = 1 + rng.below(4) as i32;
    config.sns_strength = rng.below(101) as i32;
    config.filter_strength = rng.below(101) as i32;
    config.filter_sharpness = rng.below(8) as i32;
    config.filter_type = rng.below(2) as i32;
    config.autofilter = rng.below(2) as i32;
    config.alpha_compression = rng.below(2) as i32;
    config.alpha_filtering = rng.below(3) as i32;
    config.alpha_quality = rng.below(101) as i32;
    config.partitions = rng.below(4) as i32;
    config.use_sharp_yuv = rng.below(2) as i32;
    config.exact = rng.below(2) as i32;
    if rng.chance(25) {
        config.near_lossless = rng.below(101) as i32;
    }
    config
}

fn u24_le(value: u32) -> [u8; 3] {
    let bytes = value.to_le_bytes();
    assert_eq!(bytes[3], 0, "{value} does not fit 24 bits");
    [bytes[0], bytes[1], bytes[2]]
}

/// One frame of a test animation.
#[derive(Clone)]
struct TestFrame {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    duration: u32,
    blend: bool,
    dispose: bool,
    /// The bitstream chunks (see [`bitstream_chunks`]).
    bitstream: Vec<u8>,
}

impl TestFrame {
    fn anmf(&self) -> Vec<u8> {
        assert!(
            self.x.is_multiple_of(2) && self.y.is_multiple_of(2),
            "frame offsets are even"
        );
        let mut payload = Vec::new();
        payload.extend_from_slice(&u24_le(self.x / 2));
        payload.extend_from_slice(&u24_le(self.y / 2));
        payload.extend_from_slice(&u24_le(self.width - 1));
        payload.extend_from_slice(&u24_le(self.height - 1));
        payload.extend_from_slice(&u24_le(self.duration));
        payload.push(u8::from(!self.blend) << 1 | u8::from(self.dispose));
        payload.extend_from_slice(&self.bitstream);
        riff_chunk(b"ANMF", &payload)
    }
}

/// An animated WebP file around ready-made chunks: `VP8X` (with
/// the animation flag, and the alpha flag when `alpha`), `ANIM`
/// with `background` (R, G, B, A) and `loops`, then `chunks`.
fn animation_of(
    canvas: (u32, u32),
    alpha: bool,
    background: [u8; 4],
    loops: u16,
    chunks: &[Vec<u8>],
) -> Vec<u8> {
    let mut vp8x = vec![0x02 | if alpha { 0x10 } else { 0 }, 0, 0, 0];
    vp8x.extend_from_slice(&u24_le(canvas.0 - 1));
    vp8x.extend_from_slice(&u24_le(canvas.1 - 1));
    let [r, g, b, a] = background;
    let mut anim = vec![b, g, r, a];
    anim.extend_from_slice(&loops.to_le_bytes());
    let mut file = b"RIFF\0\0\0\0WEBP".to_vec();
    file.extend_from_slice(&riff_chunk(b"VP8X", &vp8x));
    file.extend_from_slice(&riff_chunk(b"ANIM", &anim));
    for chunk in chunks {
        file.extend_from_slice(chunk);
    }
    let riff_size = (file.len() - 8) as u32;
    file[4..8].copy_from_slice(&riff_size.to_le_bytes());
    file
}

fn animation(
    canvas: (u32, u32),
    alpha: bool,
    background: [u8; 4],
    loops: u16,
    frames: &[TestFrame],
) -> Vec<u8> {
    let chunks: Vec<Vec<u8>> = frames.iter().map(TestFrame::anmf).collect();
    animation_of(canvas, alpha, background, loops, &chunks)
}

/// `w` by `h` RGBA pixels: a gradient under noise, with one of
/// five alpha shapes (opaque, noise over every value, a
/// transparent block, opaque with a few translucent pixels,
/// fully transparent).
fn noise_pixels(rng: &mut Rng, w: u32, h: u32, alpha: u32) -> Vec<u8> {
    let (dx, dy, base) = (rng.below(9), rng.below(9), rng.below(256));
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let shade = (base + x * dx + y * dy) as u8;
            let a = match alpha {
                0 => 255,
                1 => rng.below(256) as u8,
                2 => {
                    if x < w / 2 && y >= h / 3 {
                        0
                    } else {
                        255
                    }
                }
                3 => {
                    if rng.chance(8) {
                        rng.below(256) as u8
                    } else {
                        255
                    }
                }
                _ => 0,
            };
            out.extend_from_slice(&[
                shade ^ (rng.below(32) as u8),
                shade.wrapping_mul(3) ^ (rng.below(32) as u8),
                255 - shade,
                a,
            ]);
        }
    }
    out
}

/// A random animation: a canvas up to 72 px a side, one to six
/// frames anywhere on it, each coded one of the four ways with
/// settings of its own, with or without the blend and dispose
/// flags, any duration, any background, with or without the
/// canvas's alpha flag.
fn random_animation(rng: &mut Rng) -> (Vec<u8>, usize) {
    let canvas = (1 + rng.below(72), 1 + rng.below(72));
    let count = 1 + rng.below(6) as usize;
    let frames: Vec<TestFrame> = (0..count)
        .map(|_| {
            let (width, height) = if rng.chance(25) {
                canvas
            } else {
                (1 + rng.below(canvas.0), 1 + rng.below(canvas.1))
            };
            let x = 2 * rng.below((canvas.0 - width) / 2 + 1);
            let y = 2 * rng.below((canvas.1 - height) / 2 + 1);
            let coding = [
                Coding::Lossy,
                Coding::LossyAlpha,
                Coding::Lossless,
                Coding::LosslessRgb,
            ][rng.below(4) as usize];
            let alpha = rng.below(5);
            let pixels = noise_pixels(rng, width, height, alpha);
            TestFrame {
                x,
                y,
                width,
                height,
                duration: rng.below(1 << 24),
                blend: rng.chance(60),
                dispose: rng.chance(40),
                bitstream: encode_still(&pixels, width, height, coding, random_settings(rng)),
            }
        })
        .collect();
    let background = match rng.below(4) {
        0 => [0, 0, 0, 0],
        1 => [255, 255, 255, 255],
        _ => rng.next().to_le_bytes(),
    };
    let file = animation(
        canvas,
        rng.chance(80),
        background,
        rng.below(1 << 16) as u16,
        &frames,
    );
    (file, count)
}

#[test]
fn random_animations_decode_to_the_image_crates_frames() {
    let mut rng = Rng(0x646);
    let (mut files, mut frames) = (0, 0);
    for round in 0..150 {
        let (bytes, count) = random_animation(&mut rng);
        let what = format!("animation {round}");
        let want = by_image_crate(&bytes).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(want.len(), count, "{what}");
        let walked = through_libwebp(&bytes)
            .unwrap_or_else(|| panic!("{what}: the walk handed the file back"))
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        frames += assert_same_frames(&what, &walked, &want);
        files += 1;
    }
    assert_eq!(files, 150);
    assert_eq!(frames, 529);
}

#[test]
fn without_a_background_colour_the_canvas_is_left_as_the_image_crate_leaves_it() {
    // `decode_webp_frames` sets the ANIM colour whenever the
    // container has one. Should the container probe ever read
    // none, neither decoder fills or clears with a colour.
    use image::{AnimationDecoder, ImageDecoder};
    let mut rng = Rng(0x339);
    let mut frames = 0;
    for round in 0..60 {
        let (bytes, _) = random_animation(&mut rng);
        let what = format!("animation {round}");
        let mut decoder = image::codecs::webp::WebPDecoder::new(Cursor::new(&bytes[..])).unwrap();
        decoder.set_limits(decode_limits()).unwrap();
        let want = collect_frames(decoder.into_frames(), "WebP", None).unwrap();
        let walked = webp_frames_through_libwebp(&bytes, None, None)
            .unwrap_or_else(|| panic!("{what}: the walk handed the file back"))
            .unwrap();
        frames += assert_same_frames(&what, &walked, &want);
    }
    assert_eq!(frames, 207);
}

// ----------------------------------------------------------------
// The compositing, held to image-webp's.
// ----------------------------------------------------------------

/// image-webp 0.2.4's `blend_pixel_nonpremult`, as written
/// there.
fn image_webp_blend(buffer: [u8; 4], canvas: [u8; 4]) -> [u8; 4] {
    fn channel(src: u32, src_a: u8, dst: u32, dst_a: u8, scale: u32, shift: u32) -> u8 {
        let src_channel = ((src >> shift) & 0xff) as u8;
        let dst_channel = ((dst >> shift) & 0xff) as u8;
        let blend_unscaled = (u32::from(src_channel) * u32::from(src_a))
            + (u32::from(dst_channel) * u32::from(dst_a));
        ((blend_unscaled * scale) >> 24) as u8
    }
    let (src, dst) = (u32::from_le_bytes(buffer), u32::from_le_bytes(canvas));
    let src_a = ((src >> 24) & 0xff) as u8;
    let blended = if src_a == 0 {
        dst
    } else {
        let dst_a = ((dst >> 24) & 0xff) as u8;
        let dst_factor_a = div_by_255(u32::from(dst_a) * (255 - u32::from(src_a)));
        let blend_a = u32::from(src_a) + dst_factor_a;
        let scale = (1u32 << 24) / blend_a;
        let r = channel(src, src_a, dst, dst_factor_a as u8, scale, 0);
        let g = channel(src, src_a, dst, dst_factor_a as u8, scale, 8);
        let b = channel(src, src_a, dst, dst_factor_a as u8, scale, 16);
        u32::from(r) | u32::from(g) << 8 | u32::from(b) << 16 | blend_a << 24
    };
    blended.to_le_bytes()
}

#[test]
fn the_blend_is_image_webps() {
    // Every source alpha and every canvas alpha, over channel
    // values that include both ends.
    let values = |step: usize| (0..=255u8).step_by(step).chain([1, 254, 255]);
    for src_a in 0..=255u8 {
        for dst_a in 0..=255u8 {
            for s in values(51) {
                for d in values(85) {
                    let src = [s, s.wrapping_add(31), 255 - s, src_a];
                    let mut dst = [d, 255 - d, d.wrapping_mul(3), dst_a];
                    let want = image_webp_blend(src, dst);
                    blend(src, &mut dst);
                    assert_eq!(dst, want, "{src:?} over alpha {dst_a}, channel {d}");
                }
            }
        }
    }
    // And every channel pair at the alphas the shortcuts and
    // the table take.
    for src_a in [0, 1, 2, 127, 128, 254, 255] {
        for dst_a in [0, 1, 128, 254, 255] {
            for s in 0..=255u8 {
                for d in 0..=255u8 {
                    let src = [s, 255 - s, s ^ 0x55, src_a];
                    let mut dst = [d, d ^ 0xaa, 255 - d, dst_a];
                    let want = image_webp_blend(src, dst);
                    blend(src, &mut dst);
                    assert_eq!(dst, want, "{src:?} over alpha {dst_a}, channel {d}");
                }
            }
        }
    }
    // The first kept behaviour: an opaque pixel of a blended
    // frame lands one under its own value, zero excepted.
    let mut canvas = [1, 2, 3, 4];
    blend([10, 0, 255, 255], &mut canvas);
    assert_eq!(canvas, [9, 0, 254, 255]);
}

#[test]
fn a_disposed_rectangle_is_cleared_as_image_webp_clears_it() {
    // A 4x2 canvas of zeros, the colour (1, 2, 3, 4), and the
    // rectangle x 1, y 0, 2x2 for the two partial fills. The
    // expected bytes are image-webp's loops run by hand.
    let color = [1, 2, 3, 4];
    let rect = (1, 0, 2, 2);
    let filled = |full_size, frame_has_alpha| {
        let mut canvas = vec![0u8; 4 * 2 * 4];
        clear(&mut canvas, 4, color, full_size, frame_has_alpha, rect);
        canvas
    };
    // Before a full-size frame with alpha: every pixel.
    assert_eq!(filled(true, true), [1, 2, 3, 4].repeat(8));
    // Before a full-size frame without alpha: three bytes at a
    // time, so the colour's alpha never lands and the channels
    // slide (the third kept behaviour).
    let slid: Vec<u8> = [1, 2, 3]
        .iter()
        .copied()
        .cycle()
        .take(30)
        .chain([0, 0])
        .collect();
    assert_eq!(filled(true, false), slid);
    // Before a smaller frame with alpha: the rectangle.
    let mut rows = vec![0u8; 32];
    for row in [0, 16] {
        rows[row + 4..row + 12].copy_from_slice(&[1, 2, 3, 4, 1, 2, 3, 4]);
    }
    assert_eq!(filled(false, true), rows);
    // Before a smaller frame without alpha: the rectangle's
    // pixel indices, times three instead of four.
    let mut slid = vec![0u8; 32];
    for start in [3, 15] {
        slid[start..start + 6].copy_from_slice(&[1, 2, 3, 1, 2, 3]);
    }
    assert_eq!(filled(false, false), slid);
}

#[test]
fn the_background_colour_shows_where_no_frame_has_drawn() {
    // The second kept behaviour, in a fixture: the first frame
    // covers part of the canvas, and the rest is the ANIM
    // colour (libwebp's own animation decoder leaves it
    // transparent black).
    let bytes = fixture("w10_first_frame_small_bg.webp");
    assert_eq!(
        webp_animation_params(&bytes).map(|(_, color)| color),
        Some([40, 80, 160, 255])
    );
    let frames = through_libwebp(&bytes).unwrap().unwrap();
    let first = frames.iter().next().unwrap().buffer();
    assert_eq!(first.get_pixel(0, 0).0, [40, 80, 160, 255]);
    assert_eq!(first.get_pixel(71, 53).0, [40, 80, 160, 255]);
}

// ----------------------------------------------------------------
// Files the walk does not handle: the image crate's answer.
// ----------------------------------------------------------------

/// What a decode came to: frames, an error, or a panic (which
/// the server contains per request).
#[derive(Debug)]
enum Outcome {
    Frames(Frames),
    Error(EngineError),
    Panic,
}

fn outcome(decode: impl FnOnce() -> Result<Frames, EngineError>) -> Outcome {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(decode)) {
        Ok(Ok(frames)) => Outcome::Frames(frames),
        Ok(Err(error)) => Outcome::Error(error),
        Err(_) => Outcome::Panic,
    }
}

impl std::fmt::Debug for Frames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} frames of {:?}", self.len(), self.dimensions())
    }
}

/// `decode_webp_frames` answers `bytes` exactly as the image
/// crate's iterator does: the same frames, the same error, or
/// the same panic. Returns that answer.
fn assert_same_answer(what: &str, bytes: &[u8]) -> Outcome {
    let want = outcome(|| by_image_crate(bytes));
    let got = outcome(|| decode_webp_frames(bytes, None));
    match (&got, &want) {
        (Outcome::Frames(got), Outcome::Frames(want)) => {
            assert_same_frames(what, got, want);
        }
        (Outcome::Error(got), Outcome::Error(want)) => assert_eq!(got, want, "{what}"),
        (Outcome::Panic, Outcome::Panic) => {}
        _ => panic!("{what}: {got:?} where the image crate answers {want:?}"),
    }
    want
}

/// Whether the walk hands `bytes` back to the image crate's
/// iterator.
fn falls_back(bytes: &[u8]) -> bool {
    let (_, background) = open_webp_animation(bytes).expect("the image crate opens the file");
    webp_frames_through_libwebp(bytes, background, None).is_none()
}

/// Three frames on a 40x30 canvas, one coded each way, so a
/// test can damage any of them: a full-canvas lossy frame, a
/// lossy frame with alpha, a lossless frame.
fn three_frames() -> Vec<TestFrame> {
    let mut rng = Rng(7);
    [
        (0, 0, 40, 30, Coding::Lossy, 0),
        (4, 2, 21, 17, Coding::LossyAlpha, 1),
        (10, 8, 16, 12, Coding::Lossless, 1),
    ]
    .into_iter()
    .map(|(x, y, width, height, coding, alpha)| {
        let pixels = noise_pixels(&mut rng, width, height, alpha);
        TestFrame {
            x,
            y,
            width,
            height,
            duration: 40,
            blend: true,
            dispose: false,
            bitstream: encode_still(&pixels, width, height, coding, settings(75.0)),
        }
    })
    .collect()
}

const CANVAS: (u32, u32) = (40, 30);
const BACKGROUND: [u8; 4] = [20, 40, 60, 255];

fn three_frame_file(frames: &[TestFrame]) -> Vec<u8> {
    animation(CANVAS, true, BACKGROUND, 0, frames)
}

#[test]
fn the_three_frame_animation_is_walked_whole() {
    // The base of the damaged files below is itself handled by
    // the walk, so each of them falls back for its damage.
    let frames = three_frames();
    assert_eq!(&frames[0].bitstream[..4], b"VP8 ");
    assert_eq!(&frames[1].bitstream[..4], b"ALPH");
    assert_eq!(&frames[2].bitstream[..4], b"VP8L");
    let bytes = three_frame_file(&frames);
    assert!(!falls_back(&bytes));
    assert!(matches!(
        assert_same_answer("the base file", &bytes),
        Outcome::Frames(frames) if frames.len() == 3
    ));
}

#[test]
fn a_truncated_file_gets_the_image_crates_answer() {
    // Cut anywhere: inside the last frame's bitstream, inside
    // its ANMF header, between chunks, down to the container's
    // first chunks. The RIFF size still names the whole file.
    let whole = three_frame_file(&three_frames());
    let mut refused = 0;
    for cut in (1..whole.len()).rev().step_by(7).chain(1..40) {
        let bytes = &whole[..whole.len() - cut];
        let what = format!("{cut} bytes cut");
        match assert_same_answer(&what, bytes) {
            Outcome::Frames(_) => {}
            Outcome::Error(_) | Outcome::Panic => refused += 1,
        }
    }
    assert!(refused > 100, "{refused} of the cuts are refused");
    // A cut inside the last frame: the walk hands the file
    // back at that frame, after decoding the two before it.
    let bytes = &whole[..whole.len() - 9];
    assert!(falls_back(bytes));
    assert!(matches!(
        assert_same_answer("9 bytes cut", bytes),
        Outcome::Error(EngineError::Decode(_))
    ));
}

#[test]
fn a_frame_outside_the_canvas_gets_the_image_crates_error() {
    for (x, y) in [(22, 2), (4, 16), (40, 30), (1 << 24, 0)] {
        let mut frames = three_frames();
        (frames[1].x, frames[1].y) = (x, y);
        let bytes = three_frame_file(&frames);
        let what = format!("the second frame at {x},{y}");
        assert!(falls_back(&bytes), "{what}");
        let Outcome::Error(EngineError::Decode(why)) = assert_same_answer(&what, &bytes) else {
            panic!("{what}: decoded");
        };
        assert!(why.contains("outside"), "{what}: {why}");
    }
    // A frame larger than the format allows, on a canvas that
    // is not: refused before its bitstream is read.
    let mut frames = three_frames();
    frames[2].width = 16_385;
    let bytes = animation((16_384, 30), true, BACKGROUND, 0, &frames);
    assert!(falls_back(&bytes));
    assert!(matches!(
        assert_same_answer("a frame 16385 px wide", &bytes),
        Outcome::Error(EngineError::Decode(_))
    ));
}

#[test]
fn a_bitstream_of_another_size_than_its_frame_gets_the_image_crates_answer() {
    // The ANMF header states one size and the bitstream codes
    // another, larger or smaller, for each kind of frame. The
    // walk never composites such a frame.
    for index in 0..3 {
        for (dw, dh) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1), (-3, -2)] {
            let mut frames = three_frames();
            let frame = &mut frames[index];
            frame.width = (frame.width as i32 + dw) as u32;
            frame.height = (frame.height as i32 + dh) as u32;
            if frame.x + frame.width > CANVAS.0 || frame.y + frame.height > CANVAS.1 {
                (frame.x, frame.y) = (0, 0);
            }
            let canvas = (CANVAS.0 + 2, CANVAS.1 + 2);
            let bytes = animation(canvas, true, BACKGROUND, 0, &frames);
            let what = format!("frame {index} stated {dw:+},{dh:+} off its bitstream");
            assert!(falls_back(&bytes), "{what}");
            assert!(
                !matches!(assert_same_answer(&what, &bytes), Outcome::Frames(_)),
                "{what}: decoded"
            );
        }
    }
}

/// A lossless stream of one colour, transparent black, with
/// `width` by `height` in its header: after the header come
/// three flag bits that turn the transforms, the colour cache
/// and the meta codes off, and five prefix codes of one symbol
/// each, so a pixel takes no bits and the same eight bytes are
/// a whole image at any size the header states.
fn one_colour_stream(width: u32, height: u32) -> Vec<u8> {
    let header = (width - 1) | (height - 1) << 14 | 1 << 28;
    let mut stream = vec![0x2f];
    stream.extend_from_slice(&header.to_le_bytes());
    stream.extend_from_slice(&[0x88, 0x88, 0x08]);
    stream
}

#[test]
fn a_stream_of_another_size_than_its_frame_is_refused_on_its_header() {
    // libwebp decodes a still at the size its bitstream
    // states, whatever the ANMF header says, and a stream of
    // eight bytes can state 16384x16384: a gigabyte of pixels.
    // So the walk reads the stream's header first and hands
    // libwebp nothing that is not the frame's size, as
    // image-webp refuses a lossless stream of another size
    // before it decodes it.
    let frame = |stream| FrameRef {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
        duration: 0,
        blend: true,
        dispose: false,
        bitstream: Bitstream::Lossless(stream),
    };
    // At the frame's size the stream is a frame like any other.
    let fitting = one_colour_stream(4, 4);
    assert!(still_of(&frame(&fitting)).is_some());
    let (pixels, has_alpha) = decode_bitstream(&frame(&fitting)).unwrap();
    assert!(has_alpha);
    assert_eq!(pixels[..], [0u8; 4 * 4 * 4]);
    // The same bytes with 2048x2048 in their header: libwebp
    // decodes them, four million pixels, when it is asked to.
    // The walk does not ask.
    let larger = one_colour_stream(2048, 2048);
    let still = lossless_still_file(&larger).unwrap();
    let decoded = webp::Decoder::new(&still)
        .decode()
        .expect("a whole image to libwebp");
    assert_eq!((decoded.width(), decoded.height()), (2048, 2048));
    assert!(still_of(&frame(&larger)).is_none());
    assert!(decode_bitstream(&frame(&larger)).is_none());

    // In an animation, behind three good frames: the file is
    // handed back before the stream decodes, and the error is
    // the image crate's. The largest size a header can state
    // is refused the same way, at no cost.
    let with_stream = |width, height| {
        let mut frames = three_frames();
        frames.push(TestFrame {
            x: 2,
            y: 2,
            width: 4,
            height: 4,
            duration: 40,
            blend: true,
            dispose: false,
            bitstream: riff_chunk(b"VP8L", &one_colour_stream(width, height)),
        });
        three_frame_file(&frames)
    };
    let bytes = with_stream(4, 4);
    assert!(!falls_back(&bytes));
    assert!(matches!(
        assert_same_answer("a one-colour frame", &bytes),
        Outcome::Frames(frames) if frames.len() == 4
    ));
    for (width, height) in [(2048, 2048), (16_384, 16_384), (4, 5), (3, 4)] {
        let bytes = with_stream(width, height);
        let what = format!("a {width}x{height} stream in a 4x4 frame");
        assert!(falls_back(&bytes), "{what}");
        let Outcome::Error(EngineError::Decode(why)) = assert_same_answer(&what, &bytes) else {
            panic!("{what}: decoded");
        };
        assert!(why.contains("image sizes"), "{what}: {why}");
    }
}

#[test]
fn garbage_after_a_bitstream_header_gets_the_image_crates_error() {
    // A lossless frame keeps its five header bytes and a lossy
    // one its ten; the rest is replaced. Neither decoder takes
    // the result for a frame, and the error is the image
    // crate's.
    for (index, header, fill) in [(2, 8 + 5, 0xFF), (2, 8 + 5, 0x00), (0, 8 + 10, 0xFF)] {
        let mut frames = three_frames();
        for byte in &mut frames[index].bitstream[header..] {
            *byte = fill;
        }
        let bytes = three_frame_file(&frames);
        let what = format!("frame {index} filled with {fill:#04x}");
        assert!(falls_back(&bytes), "{what}");
        assert!(
            matches!(
                assert_same_answer(&what, &bytes),
                Outcome::Error(EngineError::Decode(_))
            ),
            "{what}"
        );
    }
    // A chunk that is no bitstream at all, and an alpha plane
    // followed by a chunk named as a lossless stream.
    let mut frames = three_frames();
    frames[0].bitstream[..4].copy_from_slice(b"JUNK");
    let bytes = three_frame_file(&frames);
    assert!(falls_back(&bytes));
    assert!(matches!(
        assert_same_answer("an unknown chunk in a frame", &bytes),
        Outcome::Error(EngineError::Decode(_))
    ));
    let mut frames = three_frames();
    let alph_size = u32::from_le_bytes(frames[1].bitstream[4..8].try_into().unwrap()) as usize;
    let lossy_at = 8 + alph_size + (alph_size & 1);
    assert_eq!(&frames[1].bitstream[lossy_at..lossy_at + 4], b"VP8 ");
    frames[1].bitstream[lossy_at..lossy_at + 4].copy_from_slice(b"VP8L");
    // image-webp does not look at that chunk's name and decodes
    // it as the lossy stream it is. The walk does look, hands
    // the file back, and the frames are image-webp's.
    let bytes = three_frame_file(&frames);
    assert!(falls_back(&bytes));
    assert!(matches!(
        assert_same_answer("ALPH before a chunk named VP8L", &bytes),
        Outcome::Frames(frames) if frames.len() == 3
    ));
}

#[test]
fn a_frame_image_webp_refuses_by_its_header_is_not_decoded_by_libwebp() {
    // The colour space bit of a lossy frame's header: libwebp
    // ignores it and would decode the frame, image-webp
    // refuses it, so the walk leaves the frame to image-webp.
    for index in [0, 1] {
        let mut frames = three_frames();
        let vp8_at = frames[index]
            .bitstream
            .windows(4)
            .position(|w| w == b"VP8 ")
            .unwrap();
        frames[index].bitstream[vp8_at + 8 + 10] |= 0x80;
        let bytes = three_frame_file(&frames);
        assert!(falls_back(&bytes), "frame {index}");
        let Outcome::Error(EngineError::Decode(why)) =
            assert_same_answer("the colour space bit", &bytes)
        else {
            panic!("frame {index}: decoded");
        };
        assert!(why.contains("color space"), "frame {index}: {why}");
    }
}

#[test]
fn chunks_the_frame_walk_does_not_expect_get_the_image_crates_answer() {
    let frames = three_frames();
    let chunks: Vec<Vec<u8>> = frames.iter().map(TestFrame::anmf).collect();
    // Another chunk between two frames: image-webp steps from
    // one ANMF chunk to where the next should start.
    let mut between = chunks.clone();
    between.insert(1, riff_chunk(b"EXIF", b"between"));
    let bytes = animation_of(CANVAS, true, BACKGROUND, 0, &between);
    assert!(falls_back(&bytes));
    assert!(matches!(
        assert_same_answer("a chunk between frames", &bytes),
        Outcome::Error(EngineError::Decode(_))
    ));
    // An ANMF chunk of odd size: image-webp steps by the size
    // without its padding byte, and misses the next chunk.
    let mut odd = frames.clone();
    odd[0].bitstream.push(0);
    let anmf_size = u32::from_le_bytes(odd[0].anmf()[4..8].try_into().unwrap());
    assert_eq!(anmf_size % 2, 1, "the first ANMF chunk has an odd size");
    let bytes = three_frame_file(&odd);
    assert!(falls_back(&bytes));
    assert!(matches!(
        assert_same_answer("an ANMF chunk of odd size", &bytes),
        Outcome::Error(EngineError::Decode(_))
    ));
    // An ANMF chunk too short to hold a frame, behind good
    // ones.
    let mut short = chunks.clone();
    short.push(riff_chunk(b"ANMF", &[0; 24]));
    let bytes = animation_of(CANVAS, true, BACKGROUND, 0, &short);
    assert!(falls_back(&bytes));
    assert_same_answer("a short ANMF chunk", &bytes);
    // A RIFF size that ends before the last frame: both stop
    // counting frames there.
    let mut bytes = three_frame_file(&frames);
    let shorter = (bytes.len() - 8 - chunks[2].len() - 10) as u32;
    bytes[4..8].copy_from_slice(&shorter.to_le_bytes());
    assert!(!falls_back(&bytes));
    assert!(matches!(
        assert_same_answer("a RIFF size before the last frame", &bytes),
        Outcome::Frames(frames) if frames.len() == 2
    ));
}

/// The bitstream chunks inside the ANMF chunks of `bytes`, in
/// file order: where each chunk's header starts, its fourcc,
/// and its payload size.
fn frame_chunks(bytes: &[u8]) -> Vec<(usize, [u8; 4], usize)> {
    let mut found = Vec::new();
    let mut at = 12;
    while let Some((fourcc, size, rounded)) = chunk_header(bytes, at) {
        if &fourcc == b"ANMF" {
            let end = (at + 8 + size).min(bytes.len());
            let mut inner = at + 8 + 16;
            while let Some((fourcc, size, rounded)) = chunk_header(bytes, inner) {
                if inner + 8 + size > end {
                    break;
                }
                found.push((inner, fourcc, size));
                inner += 8 + rounded;
            }
        }
        at += 8 + rounded;
    }
    found
}

#[test]
fn a_lossless_stream_reaches_libwebp_without_the_padding_byte() {
    // libwebp's still decoder reads a bitstream on to the end
    // of the bytes it is given. In a file, a chunk of odd size
    // is followed by a padding byte, so a lossless stream cut
    // short can borrow eight zero bits and decode. image-webp
    // reads the chunk and no further, and refuses that stream.
    // So the file the walk hands libwebp ends with the stream's
    // last byte.
    let whole = fixture("w14_durations_and_loops.webp");
    let &(at, fourcc, size) = frame_chunks(&whole).last().unwrap();
    assert_eq!((&fourcc, size), (b"VP8L", 167));
    let frame = |stream| FrameRef {
        x: 0,
        y: 0,
        width: 16,
        height: 12,
        duration: 0,
        blend: true,
        dispose: false,
        bitstream: Bitstream::Lossless(stream),
    };
    let stream = &whole[at + 8..at + 8 + size];
    assert!(decode_bitstream(&frame(stream)).is_some());
    // Two bytes short: libwebp decodes it in a file with the
    // padding byte its odd size brings, and not in the file
    // the walk makes.
    let short = &stream[..size - 2];
    let mut hinted = short.to_vec();
    hinted[4] |= 0x10;
    let file = still_file(&[(b"VP8L", hinted.as_slice())]).unwrap();
    assert_eq!(file.len() % 2, 0, "the file ends on the padding byte");
    assert!(
        webp::Decoder::new(&file).decode().is_some(),
        "libwebp decodes the cut stream when a padding byte follows it"
    );
    assert!(decode_bitstream(&frame(short)).is_none());
    // The two files differ in that byte alone, and in the
    // RIFF size that counts it.
    let unpadded = lossless_still_file(&hinted).unwrap();
    assert_eq!(unpadded.len() + 1, file.len());
    assert_eq!(unpadded[8..], file[8..unpadded.len()]);
    assert!(webp::Decoder::new(&unpadded).decode().is_none());
    // In the animation: the last frame's chunk two bytes
    // shorter by its size field. The image crate refuses the
    // file, and so does `decode_webp_frames`.
    let mut bytes = whole.clone();
    bytes[at + 4..at + 8].copy_from_slice(&((size - 2) as u32).to_le_bytes());
    assert!(falls_back(&bytes));
    assert!(matches!(
        assert_same_answer("the last frame, two bytes short", &bytes),
        Outcome::Error(EngineError::Decode(why)) if why.contains("Corrupt bitstream")
    ));
}

#[test]
fn a_lossless_stream_of_a_few_bytes_is_walked() {
    // A frame of one colour codes to a lossless stream of
    // eight to ten bytes (libwebp's encoder writes them that
    // short too), fewer than the twelve of a RIFF header,
    // which is the least libwebp takes for a file: handed over
    // bare, such a stream is refused unread. In its chunk it
    // decodes. A fully transparent frame is what an encoder
    // writes where nothing changed between two frames.
    let (width, height) = (24, 16);
    let stream = one_colour_stream(width, height);
    assert_eq!(stream.len(), 8);
    assert!(
        webp::Decoder::new(&stream).decode().is_none(),
        "libwebp takes no file this short"
    );
    let mut frames = three_frames();
    frames.push(TestFrame {
        x: 2,
        y: 4,
        width,
        height,
        duration: 40,
        blend: false,
        dispose: true,
        bitstream: riff_chunk(b"VP8L", &stream),
    });
    let bytes = three_frame_file(&frames);
    assert!(!falls_back(&bytes));
    let Outcome::Frames(decoded) = assert_same_answer("a lossless stream of a few bytes", &bytes)
    else {
        panic!("the file was refused");
    };
    assert_eq!(decoded.len(), 4);
    let last = decoded.iter().last().unwrap().buffer();
    assert_eq!(last.get_pixel(2, 4).0, [0, 0, 0, 0]);
    assert_eq!(last.get_pixel(25, 19).0, [0, 0, 0, 0]);
}

#[test]
fn a_bitstream_chunk_cut_short_gets_the_image_crates_answer() {
    // Every bitstream chunk of a file with each kind of frame,
    // cut one to four bytes short by its size field: whatever
    // the image crate makes of the file, so does the walk.
    let whole = fixture("w14_durations_and_loops.webp");
    let chunks = frame_chunks(&whole);
    let kinds: Vec<&[u8; 4]> = chunks.iter().map(|(_, fourcc, _)| fourcc).collect();
    assert_eq!(kinds, [b"VP8 ", b"VP8 ", b"ALPH", b"VP8 ", b"VP8L"]);
    let (mut frames, mut refused) = (0, 0);
    for (at, fourcc, size) in chunks {
        for cut in 1..=4 {
            let mut bytes = whole.clone();
            bytes[at + 4..at + 8].copy_from_slice(&((size - cut) as u32).to_le_bytes());
            let what = format!("{} at {at}, {cut} short", String::from_utf8_lossy(&fourcc));
            match assert_same_answer(&what, &bytes) {
                Outcome::Frames(_) => frames += 1,
                Outcome::Error(_) => refused += 1,
                Outcome::Panic => panic!("{what}: the image crate panics"),
            }
        }
    }
    assert_eq!(frames + refused, 5 * 4);
    assert!(refused > 0, "the image crate refuses some of the cut files");
}

#[test]
fn damage_around_the_bitstreams_gets_the_image_crates_answer() {
    // The bytes outside the frames' compressed data, damaged
    // one bit at a time: the RIFF and VP8X headers, the ANIM
    // chunk, each ANMF chunk's fields, every chunk name and
    // size, and the chunks between the frames. Whatever the
    // image crate's decoder makes of the damaged file (the
    // frames, other frames, an error, in a few cases a panic)
    // the walk makes the same of it. Two files of a few
    // hundred bytes that hold every kind of frame between
    // them get every bit of every such byte flipped. The
    // third, with chunks the walk steps over before and after
    // its frames, is a longer decode: one bit of each byte,
    // a different one from byte to byte.
    let (mut files, mut frames, mut errors, mut panics) = (0, 0, 0, 0);
    for (name, every_bit) in [
        ("w09_tiny.webp", true),
        ("w14_durations_and_loops.webp", true),
        ("w15_chunks_around_the_frames.webp", false),
    ] {
        let whole = fixture(name);
        let mut in_a_bitstream = vec![false; whole.len()];
        for (at, _, size) in frame_chunks(&whole) {
            in_a_bitstream[at + 8..at + 8 + size].fill(true);
        }
        for at in (0..whole.len()).filter(|at| !in_a_bitstream[*at]) {
            let bits = if every_bit { 0..8 } else { at % 8..at % 8 + 1 };
            for bit in bits {
                let mut bytes = whole.clone();
                bytes[at] ^= 1 << bit;
                match assert_same_answer(&format!("{name}, bit {bit} of byte {at}"), &bytes) {
                    Outcome::Frames(_) => frames += 1,
                    Outcome::Error(_) => errors += 1,
                    Outcome::Panic => panics += 1,
                }
                files += 1;
            }
        }
    }
    assert_eq!(files, 1448 + 1456 + 197);
    assert_eq!(files, frames + errors + panics);
    assert!(frames > 0 && errors > 0, "{frames} frames, {errors} errors");
}

#[test]
fn files_that_are_no_plain_animation_are_not_walked() {
    let frames = three_frames();
    let whole = three_frame_file(&frames);
    assert!(Animation::open(&whole, None).is_some());
    // Not RIFF, not WebP, too short for a header.
    assert!(Animation::open(b"", None).is_none());
    assert!(Animation::open(&whole[..29], None).is_none());
    let mut other = whole.clone();
    other[..4].copy_from_slice(b"RIFX");
    assert!(Animation::open(&other, None).is_none());
    let mut other = whole.clone();
    other[8..12].copy_from_slice(b"WEBM");
    assert!(Animation::open(&other, None).is_none());
    // A still file.
    let still = webp::Encoder::from_rgb(&[9; 12], 2, 2).encode(80.0);
    assert!(Animation::open(&still, None).is_none());
    // The animation flag unset, with the frames still there:
    // the image crate's iterator answers (it panics, which the
    // server contains), and so does `decode_webp_frames`.
    let mut unflagged = whole.clone();
    unflagged[20] &= !0x02;
    assert!(Animation::open(&unflagged, None).is_none());
    assert_same_answer("the animation flag unset", &unflagged);
    // A VP8X chunk shorter than its ten bytes.
    let mut short = whole.clone();
    short[16..20].copy_from_slice(&8u32.to_le_bytes());
    assert!(Animation::open(&short, None).is_none());
    // A canvas past the decode allocation budget is the image
    // crate's to refuse, as it always has been: the walk
    // allocates no canvas for it.
    assert_eq!(MAX_DECODE_ALLOC, 256 << 20);
    let mut vast = whole.clone();
    vast[24..27].copy_from_slice(&u24_le(8192));
    vast[27..30].copy_from_slice(&u24_le(8191));
    assert!(Animation::open(&vast, None).is_none());
    let mut fits = whole;
    fits[24..27].copy_from_slice(&u24_le(8191));
    fits[27..30].copy_from_slice(&u24_le(8191));
    assert!(Animation::open(&fits, None).is_some());
}

// ----------------------------------------------------------------
// The limits, through the walk.
// ----------------------------------------------------------------

#[test]
fn the_frame_cap_and_the_deadline_hold_through_the_walk() {
    // One 1x1 frame, 201 times: the collector refuses the
    // 201st, and the refusal is the walk's own answer, not a
    // fallback's.
    let mut rng = Rng(1);
    let pixels = noise_pixels(&mut rng, 1, 1, 0);
    let frame = TestFrame {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
        duration: 10,
        blend: true,
        dispose: false,
        bitstream: encode_still(&pixels, 1, 1, Coding::Lossless, settings(75.0)),
    };
    let at_the_cap = animation(
        (1, 1),
        true,
        [0; 4],
        0,
        &vec![frame.clone(); MAX_ANIMATED_FRAMES],
    );
    let walked = through_libwebp(&at_the_cap).unwrap().unwrap();
    assert_eq!(walked.len(), MAX_ANIMATED_FRAMES);
    let past_the_cap = animation(
        (1, 1),
        true,
        [0; 4],
        0,
        &vec![frame; MAX_ANIMATED_FRAMES + 1],
    );
    let refusal = EngineError::Decode("WebP has more than 200 frames".to_string());
    assert_eq!(
        through_libwebp(&past_the_cap).unwrap().unwrap_err(),
        refusal
    );
    assert_eq!(by_image_crate(&past_the_cap).unwrap_err(), refusal);
    assert_eq!(
        decode_webp_frames(&past_the_cap, None).unwrap_err(),
        refusal
    );

    // Past the deadline no frame decodes, and the walk says so
    // itself.
    let bytes = three_frame_file(&three_frames());
    let past = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
    let (_, background) = open_webp_animation(&bytes).unwrap();
    assert_eq!(
        webp_frames_through_libwebp(&bytes, background, past)
            .unwrap()
            .unwrap_err(),
        EngineError::Timeout
    );
    assert_eq!(
        decode_webp_frames(&bytes, past).unwrap_err(),
        EngineError::Timeout
    );
    let ahead = Some(std::time::Instant::now() + std::time::Duration::from_secs(60));
    assert_eq!(decode_webp_frames(&bytes, ahead).unwrap().len(), 3);
}
