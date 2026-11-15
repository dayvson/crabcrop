//! Byte-identity proofs for the engine's fast paths: each is
//! compared with the code it replaced, which stays here as it
//! was.

use super::*;

/// Smooth content with a little noise on top.
fn smooth(x: u32, y: u32, c: u32) -> u8 {
    let ramp = (x * 3 + y * 5 + c * 40) % 256;
    let mut h = x
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(y.wrapping_mul(0x85EB_CA77))
        .wrapping_add(c.wrapping_mul(0xC2B2_AE3D));
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    ((ramp * 3 + (h % 256)) / 4) as u8
}

/// Hard edges between the extremes: a Lanczos resample
/// overshoots there, which is where a clamp or an alpha
/// rounding would part two implementations.
fn harsh(x: u32, y: u32, c: u32) -> u8 {
    match (x / 3 + y / 2 + c) % 5 {
        0 => 0,
        1 => 255,
        2 => 1,
        3 => 254,
        _ => smooth(x, y, c),
    }
}

type Content = fn(u32, u32, u32) -> u8;

/// An alpha plane with opaque, transparent and partial runs.
fn mixed_alpha(content: Content) -> impl Fn(u32, u32) -> u8 {
    move |x, y| match (x / 7 + y / 5) % 4 {
        0 => 255,
        1 => 0,
        _ => content(y, x, 3),
    }
}

/// A 16-bit sample that uses the low byte too.
fn deep(content: Content, x: u32, y: u32, c: u32) -> u16 {
    u16::from(content(x, y, c)) << 8 | u16::from(content(y, x, c + 5))
}

/// Every layout a decode can hand the chain, filled from
/// `content`, with `alpha` as the alpha plane of the layouts
/// that have one. Sixteen-bit and float alpha stretch the
/// same eight-bit plane, so a transparent or opaque pixel is
/// one in every layout.
fn layouts(
    w: u32,
    h: u32,
    content: Content,
    alpha: impl Fn(u32, u32) -> u8,
) -> Vec<(&'static str, DynamicImage)> {
    let c8 = |x, y, c| content(x, y, c);
    let c16 = |x, y, c| deep(content, x, y, c);
    let cf = |x, y, c| f32::from(deep(content, x, y, c)) / 65535.0;
    let a16 = |x, y| u16::from(alpha(x, y)) * 257;
    let af = |x, y| f32::from(alpha(x, y)) / 255.0;
    vec![
        (
            "rgb8",
            DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::Rgb([c8(x, y, 0), c8(x, y, 1), c8(x, y, 2)])
            })),
        ),
        (
            "rgba8",
            DynamicImage::ImageRgba8(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::Rgba([c8(x, y, 0), c8(x, y, 1), c8(x, y, 2), alpha(x, y)])
            })),
        ),
        (
            "luma8",
            DynamicImage::ImageLuma8(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::Luma([c8(x, y, 0)])
            })),
        ),
        (
            "lumaa8",
            DynamicImage::ImageLumaA8(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::LumaA([c8(x, y, 0), alpha(x, y)])
            })),
        ),
        (
            "rgb16",
            DynamicImage::ImageRgb16(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::Rgb([c16(x, y, 0), c16(x, y, 1), c16(x, y, 2)])
            })),
        ),
        (
            "rgba16",
            DynamicImage::ImageRgba16(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::Rgba([c16(x, y, 0), c16(x, y, 1), c16(x, y, 2), a16(x, y)])
            })),
        ),
        (
            "luma16",
            DynamicImage::ImageLuma16(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::Luma([c16(x, y, 0)])
            })),
        ),
        (
            "lumaa16",
            DynamicImage::ImageLumaA16(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::LumaA([c16(x, y, 0), a16(x, y)])
            })),
        ),
        (
            "rgb32f",
            DynamicImage::ImageRgb32F(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::Rgb([cf(x, y, 0), cf(x, y, 1), cf(x, y, 2)])
            })),
        ),
        (
            "rgba32f",
            DynamicImage::ImageRgba32F(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::Rgba([cf(x, y, 0), cf(x, y, 1), cf(x, y, 2), af(x, y)])
            })),
        ),
    ]
}

/// Both contents in every layout, each under its mixed alpha.
fn frames_at(w: u32, h: u32) -> Vec<(String, DynamicImage)> {
    let mut frames = Vec::new();
    for (kind, content) in [("smooth", smooth as Content), ("harsh", harsh)] {
        for (layout, img) in layouts(w, h, content, mixed_alpha(content)) {
            frames.push((format!("{kind} {layout} {w}x{h}"), img));
        }
    }
    frames
}

/// The layout, the size and every byte.
fn same(a: &DynamicImage, b: &DynamicImage) -> bool {
    a.color() == b.color()
        && (a.width(), a.height()) == (b.width(), b.height())
        && a.as_bytes() == b.as_bytes()
}

/// Makes the backend's convolution refuse on this thread
/// until dropped.
struct RefusedConvolve;

impl RefusedConvolve {
    fn new() -> Self {
        CONVOLVE_REFUSES.set(true);
        RefusedConvolve
    }
}

impl Drop for RefusedConvolve {
    fn drop(&mut self) {
        CONVOLVE_REFUSES.set(false);
    }
}

#[test]
fn owned_resize_matches_the_borrowed_resize() {
    // Issue #657: a frame with alpha is premultiplied where
    // it sits. Every layout, down and up on each axis alone
    // and on both, to one pixel, and at the size it already
    // has, where the backend copies and no premultiply may
    // run.
    let mut cases = 0;
    for (w, h) in [(1, 1), (3, 2), (9, 17), (64, 48), (101, 53)] {
        for (name, img) in frames_at(w, h) {
            for (tw, th) in [
                (w.div_ceil(2), h.div_ceil(2)),
                (w.div_ceil(3), h),
                (w, h.div_ceil(3)),
                (w * 2, h * 2),
                (w + 1, h + 3),
                (1, 1),
                (40, 27),
                (33, 77),
                (w, h),
            ] {
                let borrowed = resize_lanczos3(&img, tw, th);
                let owned = resize_lanczos3_owned(img.clone(), tw, th);
                assert!(same(&borrowed, &owned), "{name} to {tw}x{th}");
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 900);
}

/// Alpha levels whose reciprocal is a whole number, in eight
/// bits and (times 257) in sixteen: dividing by one undoes a
/// premultiply by it exactly on every instruction set, while
/// a frame premultiplied twice still shows.
fn whole_reciprocal_alpha(x: u32, y: u32) -> u8 {
    [255, 51, 0, 85, 17][((x / 3 + y / 2) % 5) as usize]
}

/// The layouts with alpha, on hard edges, under alpha levels
/// a divide undoes exactly: [`whole_reciprocal_alpha`] for
/// the integer layouts, powers of two for the float one.
fn exactly_divisible_frames(w: u32, h: u32) -> Vec<(&'static str, DynamicImage)> {
    let mut frames: Vec<_> = layouts(w, h, harsh, whole_reciprocal_alpha)
        .into_iter()
        .filter(|(name, img)| img.color().has_alpha() && *name != "rgba32f")
        .collect();
    frames.push((
        "rgba32f",
        DynamicImage::ImageRgba32F(image::ImageBuffer::from_fn(w, h, |x, y| {
            let color = |c| f32::from(deep(harsh, x, y, c)) / 65535.0;
            let alpha = [1.0, 0.25, 0.0, 0.5][((x / 3 + y / 2) % 4) as usize];
            image::Rgba([color(0), color(1), color(2), alpha])
        })),
    ));
    frames
}

#[test]
fn a_refused_resample_never_hands_on_a_premultiplied_frame() {
    // Issue #657: the frame is premultiplied before the
    // backend is asked to resample it. No input makes the
    // backend refuse at that point, so the refusal is forced
    // here: the frame must be divided back before the plain
    // path takes it, or the plain path would premultiply it
    // a second time.
    let mut cases = 0;
    for (w, h) in [(9, 17), (64, 48)] {
        for (name, img) in exactly_divisible_frames(w, h) {
            for (tw, th) in [(w.div_ceil(2), h.div_ceil(2)), (w * 2, h + 3)] {
                let plain = resize_lanczos3(&img, tw, th);
                let refused = {
                    let _refusal = RefusedConvolve::new();
                    resize_lanczos3_owned(img.clone(), tw, th)
                };
                assert!(same(&plain, &refused), "{name} {w}x{h} to {tw}x{th}");
                // With the backend answering, the same call
                // takes the in-place path, and agrees.
                let in_place = resize_lanczos3_owned(img.clone(), tw, th);
                assert!(same(&plain, &in_place), "{name} {w}x{h} to {tw}x{th}");
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 2 * 5 * 2);
}

/// The cut every aspect crop made before issue #654: the
/// image crate's, one pixel at a time through the frame's
/// RGBA view.
fn crate_cut(img: &DynamicImage, (x, y, w, h): Window) -> DynamicImage {
    DynamicImage::ImageRgba8(image::imageops::crop_imm(img, x, y, w, h).to_image())
}

/// The two-step chain a crop was: the cut, then the resize of
/// the cut.
fn cut_then_resize(img: &DynamicImage, window: Window, w: u32, h: u32) -> DynamicImage {
    resize_lanczos3(&crate_cut(img, window), w, h)
}

/// Windows of a `w` by `h` frame: the whole of it, halves on
/// each side, an interior one, the last pixel, and one-pixel
/// strips.
fn windows(w: u32, h: u32) -> Vec<Window> {
    let mut windows = vec![
        (0, 0, w, h),
        (0, 0, w.div_ceil(2), h),
        (w / 2, 0, w - w / 2, h),
        (0, h / 2, w, h - h / 2),
        (w / 3, h / 3, (w / 3).max(1), (h / 3).max(1)),
        (w - 1, h - 1, 1, 1),
        (0, 0, 1, h),
        (0, 0, w, 1),
    ];
    windows.dedup();
    windows
}

#[test]
fn cut_by_rows_matches_the_crate_cut() {
    // Issue #654: RGB8 and RGBA8 windows are copied by rows.
    // Every layout, windows on every edge, and windows that
    // reach past the frame, which both cuts clamp.
    let mut cases = 0;
    for (w, h) in [(1, 1), (2, 3), (7, 5), (64, 48)] {
        for (name, img) in frames_at(w, h) {
            let mut windows = windows(w, h);
            windows.extend([(w / 2, h / 2, w, h), (w, h, 1, 1), (0, 0, w + 5, h + 5)]);
            for window in windows {
                let ours = DynamicImage::ImageRgba8(crate::crop::cut_rgba(&img, window));
                assert!(same(&crate_cut(&img, window), &ours), "{name} {window:?}");
                cases += 1;
            }
        }
    }
    assert!(cases >= 4 * 20 * 6, "{cases}");
}

/// Targets for a `cw` by `ch` window: down and up on both
/// axes, one axis each way, one axis left alone, one pixel,
/// two fixed shapes, and the window's own size.
fn targets(cw: u32, ch: u32) -> Vec<(u32, u32)> {
    vec![
        (cw.div_ceil(2), ch.div_ceil(2)),
        (cw.div_ceil(3), ch.div_ceil(5)),
        (cw * 2, ch * 2),
        (cw * 3 + 1, ch + 1),
        (cw + 1, ch.div_ceil(2)),
        (1, 1),
        (cw, ch.div_ceil(2)),
        (cw * 2, ch),
        (40, 30),
        (17, 51),
        (cw, ch),
    ]
}

#[test]
fn window_resample_matches_cut_then_resize() {
    // Issue #654: the crop's closing resize reads its window
    // through a view of the frame where that holds no more
    // memory than the cut did, and cuts first where it would.
    // RGB8 and RGBA8 go both ways, on every window and
    // target; the other layouts are always cut first, as
    // before, so two of each stand for them.
    let (mut cases, mut from_frame, mut cut_first) = (0, 0, 0);
    let mut check = |name: &str, img: &DynamicImage, window: Window, (tw, th): (u32, u32)| {
        let ours = resize_lanczos3_cut(img.clone(), window, tw, th);
        assert!(
            same(&cut_then_resize(img, window, tw, th), &ours),
            "{name} {window:?} to {tw}x{th}"
        );
        cases += 1;
        let (frame, dims) = (img.as_bytes().len() as u64, (window.2, window.3));
        match img {
            _ if dims == (tw, th) => {}
            DynamicImage::ImageRgb8(_) if view_holds_no_more(frame, 3, dims, (tw, th)) => {
                from_frame += 1;
            }
            DynamicImage::ImageRgba8(_) if view_holds_no_more(frame, 4, dims, (tw, th)) => {
                from_frame += 1;
            }
            _ => cut_first += 1,
        }
    };
    for (w, h) in [(1, 1), (2, 3), (7, 5), (33, 21)] {
        for (name, img) in frames_at(w, h) {
            let viewed = matches!(
                img,
                DynamicImage::ImageRgb8(_) | DynamicImage::ImageRgba8(_)
            );
            let inner = (w / 3, h / 3, (w / 3).max(1), (h / 3).max(1));
            let windows = if viewed {
                windows(w, h)
            } else {
                vec![(0, 0, w, h), inner]
            };
            for window in windows {
                let mut targets = targets(window.2, window.3);
                targets.truncate(if viewed { targets.len() } else { 2 });
                for target in targets {
                    check(&name, &img, window, target);
                }
            }
        }
    }
    // Wider frames, for whole vector blocks and kernels many
    // rows deep, in the shapes a crop makes.
    for (w, h, windows, targets) in [
        (
            101,
            53,
            [(15, 0, 71, 53), (0, 7, 101, 40)],
            [(40, 30), (30, 40), (142, 106)],
        ),
        (
            333,
            217,
            [(22, 0, 289, 217), (0, 3, 333, 187)],
            [(40, 30), (30, 40), (400, 300)],
        ),
    ] {
        for (name, img) in layouts(w, h, harsh, mixed_alpha(harsh)).into_iter().take(2) {
            for window in windows {
                for target in targets {
                    check(name, &img, window, target);
                }
            }
        }
    }
    assert!(cases > 1000, "{cases}");
    assert!(
        from_frame > 300 && cut_first > 300,
        "{from_frame} {cut_first}"
    );
}

#[test]
fn rgb_rows_widen_to_rgba_in_place() {
    // Issue #654: the RGB8 resample lands at the front of
    // the RGBA8 result's own buffer and is widened where it
    // sits. Every length around the run size and its
    // multiples, where whole runs give way to single pixels;
    // the bytes past the pixels are left alone.
    for pixels in (0..=1100).chain([2047, 2048, 2049, 5000]) {
        let packed: Vec<u8> = (0..pixels * 3).map(|i| (i * 7 + i / 5) as u8).collect();
        let mut buf = packed.clone();
        buf.resize(pixels * 4 + 3, 0xAB);
        spread_rgb_to_rgba(&mut buf, pixels);
        let (spread, past) = buf.split_at(pixels * 4);
        let (wide, _) = spread.as_chunks::<4>();
        let (narrow, _) = packed.as_chunks::<3>();
        for (i, (wide, px)) in wide.iter().zip(narrow).enumerate() {
            assert_eq!(*wide, [px[0], px[1], px[2], 255], "pixel {i} of {pixels}");
        }
        assert_eq!(past, [0xAB; 3], "{pixels}");
    }
}

#[test]
fn a_window_is_read_from_the_frame_only_where_that_holds_no_more() {
    // Issue #654: reading the window from the frame keeps
    // the frame until the output is written, where the
    // cut-first chain released it after the cut. The rule
    // that picks between them, on the shapes it is for: an
    // aspect crop of a photo, reduced or enlarged; a frame
    // of exactly twice the cut and one byte more; a small
    // window of a large frame, reduced well, barely reduced
    // and enlarged; and the edge of "the working rows and
    // the output fit in the cut" for both layouts.
    let photo = 2400 * 1600 * 3;
    assert!(view_holds_no_more(photo, 3, (2133, 1600), (400, 300)));
    assert!(view_holds_no_more(photo, 3, (2133, 1600), (4000, 3000)));
    assert!(view_holds_no_more(80_000, 3, (100, 100), (150, 150)));
    assert!(!view_holds_no_more(80_001, 3, (100, 100), (150, 150)));
    let large = 6000 * 4000 * 3;
    assert!(view_holds_no_more(large, 3, (400, 400), (100, 100)));
    assert!(!view_holds_no_more(large, 3, (400, 400), (380, 380)));
    assert!(!view_holds_no_more(large, 3, (400, 400), (4000, 4000)));
    assert!(view_holds_no_more(large, 3, (400, 400), (500, 200)));
    assert!(!view_holds_no_more(large, 3, (400, 400), (501, 200)));
    assert!(view_holds_no_more(large, 4, (400, 400), (400, 200)));
    assert!(!view_holds_no_more(large, 4, (400, 400), (401, 200)));
    // What the rule promises, as arithmetic: wherever it
    // says yes, the frame, the working rows and the output
    // together are no more than the larger moment of the
    // cut-first chain as it was: the frame beside the cut,
    // or the cut, its premultiplied copy, that chain's
    // working rows and the output.
    let (mut yes, mut no) = (0, 0);
    for channels in [3u64, 4] {
        for (fw, fh) in [(64u64, 48u64), (400, 300), (2400, 1600), (6000, 4000)] {
            for (ww, wh) in [(1u32, 1u32), (16, 12), (48, 48), (300, 300), (2133, 1600)] {
                if u64::from(ww) > fw || u64::from(wh) > fh {
                    continue;
                }
                for (w, h) in [
                    (1u32, 1u32),
                    (8, 6),
                    (40, 30),
                    (299, 299),
                    (400, 300),
                    (4000, 3000),
                ] {
                    let frame = fw * fh * channels;
                    let cut = 4 * u64::from(ww) * u64::from(wh);
                    let output = 4 * u64::from(w) * u64::from(h);
                    let working = |bytes: u64| bytes * u64::from(ww) * u64::from(h);
                    let before = (frame + cut).max(2 * cut + working(4) + output);
                    let from_frame = frame + working(channels) + output;
                    if view_holds_no_more(frame, channels, (ww, wh), (w, h)) {
                        assert!(from_frame <= before, "{fw}x{fh} {ww}x{wh} to {w}x{h}");
                        yes += 1;
                    } else {
                        no += 1;
                    }
                }
            }
        }
    }
    assert!(yes > 50 && no > 50, "{yes} {no}");
}

#[test]
fn the_view_ends_at_the_window_and_the_crop_box_does_not() {
    // Issue #654 leans on this difference. Two frames that
    // agree inside the window and nowhere else resample to
    // the same bytes through the cropped view, which is why
    // the view matches a copy of the window. Through the
    // backend's crop box (the `crop-rect` fusion of issue
    // #287) they do not: its kernel reads the frame around
    // the box.
    let window @ (x, y, ww, wh) = (8, 4, 48, 40);
    for (name, img) in layouts(64, 48, harsh, mixed_alpha(harsh))
        .into_iter()
        .take(2)
    {
        let mut elsewhere = img.clone();
        let channels = usize::from(img.color().channel_count());
        let stride = 64 * channels;
        for (i, v) in image_bytes_mut(&mut elsewhere).iter_mut().enumerate() {
            let (px, py) = ((i % stride / channels) as u32, (i / stride) as u32);
            let inside = (x..x + ww).contains(&px) && (y..y + wh).contains(&py);
            if !inside {
                *v = !*v;
            }
        }
        assert!(!same(&img, &elsewhere));
        for (tw, th) in [(20, 17), (72, 60)] {
            // Both targets read the window from the frame.
            assert!(view_holds_no_more(
                img.as_bytes().len() as u64,
                channels as u64,
                (ww, wh),
                (tw, th)
            ));
            let viewed = resize_lanczos3_cut(img.clone(), window, tw, th);
            assert!(
                same(
                    &viewed,
                    &resize_lanczos3_cut(elsewhere.clone(), window, tw, th)
                ),
                "{name} to {tw}x{th}: the view read outside its window"
            );
            assert!(same(&viewed, &cut_then_resize(&img, window, tw, th)));
            let boxed = resize_lanczos3_cropped(&img, x, y, ww, wh, tw, th);
            assert!(
                !same(
                    &boxed,
                    &resize_lanczos3_cropped(&elsewhere, x, y, ww, wh, tw, th)
                ),
                "{name} to {tw}x{th}: the crop box stayed inside its box"
            );
        }
    }
}

/// The bytes of an 8-bit color frame, to edit in place.
fn image_bytes_mut(img: &mut DynamicImage) -> &mut [u8] {
    match img {
        DynamicImage::ImageRgb8(i) => i,
        DynamicImage::ImageRgba8(i) => i,
        _ => unreachable!("an 8-bit color frame"),
    }
}

#[test]
fn an_opaque_window_stays_opaque_within_the_reduction_bound() {
    // Issue #654: an RGB8 window is resampled without its
    // alpha, which is only the RGBA8 resample's bytes while
    // that alpha comes out 255. Up to the bound it does, by
    // arithmetic; a strip at the bound and one past it both
    // match the two-step chain, the second by taking it.
    assert!(opaque_stays_opaque((1024, 2048), (1, 2)));
    assert!(!opaque_stays_opaque((1025, 1), (1, 1)));
    assert!(!opaque_stays_opaque((1, 2049), (1, 2)));
    for (ww, wh, tw, th) in [
        (1024, 2, 1, 1),
        (2, 1024, 1, 1),
        (1500, 2, 1, 1),
        (2, 1500, 1, 1),
        (2, 1500, 1, 2),
    ] {
        let img = DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(ww + 3, wh + 1, |x, y| {
            image::Rgb([harsh(x, y, 0), harsh(x, y, 1), harsh(x, y, 2)])
        }));
        let window = (2, 1, ww, wh);
        let ours = resize_lanczos3_cut(img.clone(), window, tw, th);
        assert!(
            same(&cut_then_resize(&img, window, tw, th), &ours),
            "{ww}x{wh} to {tw}x{th}"
        );
        let (pixels, _) = ours.as_bytes().as_chunks::<4>();
        assert!(pixels.iter().all(|px| px[3] == 255));
    }
}

#[test]
fn a_refused_window_resample_takes_the_two_step_chain() {
    // Issue #654: the window of an RGBA8 frame is
    // premultiplied where it sits. A refusal after that, which
    // is forced here, must leave the window straight for the
    // cut that follows; an RGB8 frame is never written to.
    // Both targets of each layout are read from the frame or
    // cut first as the rule has it; either way the refusal
    // ends in the plain resize of the cut.
    let window = (5, 3, 40, 30);
    let rgba = exactly_divisible_frames(64, 48).remove(0).1;
    assert!(matches!(rgba, DynamicImage::ImageRgba8(_)));
    let rgb = DynamicImage::ImageRgb8(rgba.to_rgb8());
    for img in [rgba, rgb] {
        for (tw, th) in [(20, 15), (60, 31)] {
            let two_step = cut_then_resize(&img, window, tw, th);
            let refused = {
                let _refusal = RefusedConvolve::new();
                resize_lanczos3_cut(img.clone(), window, tw, th)
            };
            assert!(same(&two_step, &refused), "{:?} to {tw}x{th}", img.color());
            assert!(same(
                &two_step,
                &resize_lanczos3_cut(img.clone(), window, tw, th)
            ));
        }
    }
}

fn crop_op(w: u32, h: u32, align: Align) -> Operation {
    Operation::Crop {
        size: ImageSize {
            width: Dimension::Pixels(w),
            height: Dimension::Pixels(h),
            flip_horizontal: false,
            flip_vertical: false,
        },
        align,
    }
}

/// One still through the chain.
fn run(img: &DynamicImage, ops: &[Operation], ctx: &ProcessContext<'_>) -> DynamicImage {
    let directives = crate::url::RequestDirectives::analyze(ops);
    apply_operations(img.clone(), ops, None, 1.0, ctx, &directives, None)
        .expect("the chain runs")
        .img
}

/// What a crop to `tw` by `th` was before issue #654, given
/// the window it decided on (`None` when it made no cut):
/// the crate's cut, then the resize unless the cut already
/// has the target's size.
fn old_crop(img: &DynamicImage, window: Option<Window>, tw: u32, th: u32) -> DynamicImage {
    let cut = match window {
        Some(window) => crate_cut(img, window),
        None => img.clone(),
    };
    if (cut.width(), cut.height()) == (tw, th) {
        cut
    } else {
        resize_lanczos3(&cut, tw, th)
    }
}

/// Whether a frame already has the target's aspect: an
/// aspect, focal or point crop then makes no cut and the
/// frame keeps its layout.
fn aspect_matches(img: &DynamicImage, tw: u32, th: u32) -> bool {
    img.width() as u64 * th as u64 == tw as u64 * img.height() as u64
}

#[test]
fn crop_chain_matches_cut_then_resize() {
    // Issue #654, through the operation itself: frames down
    // to one pixel in every layout, landscape and portrait
    // frames in the layouts that stand for the rest, every
    // alignment that positions by rule and the one that runs
    // the focal analysis.
    let ctx = ProcessContext::default();
    let mut cases = 0;
    for (w, h, every_layout, targets) in [
        (1, 1, true, &[(40, 30), (30, 40), (1, 1), (80, 7)][..]),
        (3, 2, true, &[(40, 30), (30, 40), (1, 1), (80, 7)]),
        (101, 53, false, &[(40, 30), (30, 40)]),
        (53, 101, false, &[(40, 30), (30, 40)]),
    ] {
        for (name, img) in layouts(w, h, harsh, mixed_alpha(harsh)) {
            if !every_layout && !["rgb8", "rgba8", "luma8", "rgba16", "rgb32f"].contains(&name) {
                continue;
            }
            for &(tw, th) in targets {
                for align in [
                    Align::Center,
                    Align::Top,
                    Align::Bottom,
                    Align::Left,
                    Align::Right,
                    Align::TopLeft,
                    Align::TopRight,
                    Align::BottomLeft,
                    Align::BottomRight,
                    Align::Smart,
                ] {
                    let (cut, trace) = crate::crop::crop_to_aspect_traced(
                        img.clone(),
                        tw,
                        th,
                        align.clone(),
                        None,
                        None,
                    );
                    let window = (!aspect_matches(&img, tw, th))
                        .then(|| trace.expect("a crop traces its window").window);
                    // The traced crop still hands back the cut.
                    let expected_cut = old_crop(&img, window, cut.width(), cut.height());
                    assert!(same(&expected_cut, &cut), "cut {name} {w}x{h} {align:?}");
                    let ours = run(&img, &[crop_op(tw, th, align.clone())], &ctx);
                    assert!(
                        same(&old_crop(&img, window, tw, th), &ours),
                        "{name} {w}x{h} to {tw}x{th} {align:?}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, (2 * 10 * 4 + 2 * 5 * 2) * 10);
}

/// One face, off center.
#[derive(Debug)]
struct OneFace;

const FACE: crate::ai::FaceBox = crate::ai::FaceBox {
    x: 60,
    y: 8,
    w: 20,
    h: 24,
    score: 0.9,
};

impl crate::ai::FaceProvider for OneFace {
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
        Ok(vec![FACE])
    }
}

/// Two objects of one class.
#[derive(Debug)]
struct TwoCars;

fn cars() -> Vec<crate::ai::Detection> {
    [(10, 20, 30, 20), (50, 5, 25, 30)]
        .into_iter()
        .map(|(x, y, w, h)| crate::ai::Detection {
            label: "car".to_string(),
            score: 0.8,
            x,
            y,
            w,
            h,
        })
        .collect()
}

impl crate::ai::DetectionProvider for TwoCars {
    fn knows_class(&self, name: &str) -> bool {
        name.eq_ignore_ascii_case("car")
    }
    fn describe(&self) -> Vec<String> {
        vec!["car".to_string()]
    }
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
        Ok(cars())
    }
}

#[test]
fn subject_and_focal_crops_match_cut_then_resize() {
    // Issue #654: the other four ways a crop finds its
    // window (a face, detected objects, a focal rectangle, a
    // focal point) hand it on uncut as well.
    let (face, objects) = (OneFace, TwoCars);
    let ctx = ProcessContext {
        face_provider: Some(&face),
        detect_provider: Some(&objects),
        ..Default::default()
    };
    let dims = (101, 53);
    let named_rect = (55, 10, 90, 40);
    let named_point = (0.8, 0.3, 2.0);
    // The rectangle and the point as the crop receives them.
    let Focal::Rect(rect) = PendingFocal::rect(named_rect, dims).resolve(dims) else {
        unreachable!("a rectangle resolves to a rectangle");
    };
    let Focal::Point(point) = PendingFocal::point(named_point, dims).resolve(dims) else {
        unreachable!("a point resolves to a point");
    };
    let union = crate::crop::union_box(&cars()).expect("two boxes");
    let padding = crate::url::RequestDirectives::analyze(&[]).face_padding;
    let mut cases = 0;
    for (name, img) in layouts(dims.0, dims.1, harsh, mixed_alpha(harsh)) {
        for (tw, th) in [(40, 30), (30, 40), dims] {
            // An aspect that already matches: the focal
            // crops report the whole frame and cut nothing.
            let cuts = !aspect_matches(&img, tw, th);
            let decided: [(&str, Vec<Operation>, Option<Window>); 4] = [
                (
                    "face",
                    vec![crop_op(tw, th, Align::Face)],
                    crate::crop::crop_around_face_traced(img.clone(), tw, th, &FACE, padding).1,
                ),
                (
                    "objects",
                    vec![crop_op(
                        tw,
                        th,
                        Align::Objects {
                            classes: vec!["car".to_string()],
                            scale: 1.5,
                        },
                    )],
                    crate::crop::crop_around_objects_traced(img.clone(), tw, th, union, 1.5).1,
                ),
                (
                    "focal rectangle",
                    vec![
                        Operation::FocalRect {
                            left: named_rect.0,
                            top: named_rect.1,
                            right: named_rect.2,
                            bottom: named_rect.3,
                        },
                        crop_op(tw, th, Align::Center),
                    ],
                    crate::crop::crop_to_aspect_focal_traced(img.clone(), tw, th, rect)
                        .1
                        .filter(|_| cuts),
                ),
                (
                    "focal point",
                    vec![
                        Operation::FocalPoint {
                            x: named_point.0,
                            y: named_point.1,
                            zoom: named_point.2,
                        },
                        crop_op(tw, th, Align::Smart),
                    ],
                    crate::crop::crop_to_aspect_point_traced(img.clone(), tw, th, point)
                        .1
                        .filter(|_| cuts),
                ),
            ];
            for (how, ops, window) in decided {
                let ours = run(&img, &ops, &ctx);
                assert!(
                    same(&old_crop(&img, window, tw, th), &ours),
                    "{how} {name} to {tw}x{th} {window:?}"
                );
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 10 * 3 * 4);
}

#[test]
fn animated_frames_replay_one_window_from_their_own_frames() {
    // Issue #654 and issue #376: the first frame of an
    // animated smart crop decides the window and the later
    // frames replay it, now uncut. Each RGBA8 frame has that
    // window premultiplied where it sits and is then
    // dropped; the next frame must come out as the two-step
    // chain made it.
    let ctx = ProcessContext::default();
    let (tw, th) = (15, 20);
    let ops = [crop_op(tw, th, Align::Smart)];
    let directives = crate::url::RequestDirectives::analyze(&ops);
    let frame = |content: Content| layouts(101, 53, content, mixed_alpha(content)).remove(1).1;
    let frames = [frame(harsh), frame(smooth), frame(harsh)];
    let mut share = FrameShare {
        smart_windows: Some(SmartWindows::default()),
        ..FrameShare::default()
    };
    let (_, trace) =
        crate::crop::crop_to_aspect_traced(frames[0].clone(), tw, th, Align::Smart, None, None);
    let window = trace.expect("the first frame decides").window;
    // The replayed frames take the path that writes to them.
    assert!(view_holds_no_more(
        frames[1].as_bytes().len() as u64,
        4,
        (window.2, window.3),
        (tw, th)
    ));
    for (index, img) in frames.iter().enumerate() {
        let ours = apply_operations(
            img.clone(),
            &ops,
            None,
            1.0,
            &ctx,
            &directives,
            Some(&mut share),
        )
        .expect("the chain runs")
        .img;
        assert!(
            same(&old_crop(img, Some(window), tw, th), &ours),
            "frame {index}"
        );
        let shared = share.smart_windows.as_ref().expect("shared windows");
        assert_eq!(shared.get((0, 101, 53)), Some(window), "frame {index}");
    }
}

#[test]
fn the_debug_view_draws_on_the_frame_as_it_came() {
    // Issue #654: the debug view annotates the frame the
    // crop ran on. It keeps its own copy, taken before the
    // crop, so no in-place step of the crop's resize shows
    // on it: the view is the one drawn on the untouched
    // frame.
    let ctx = ProcessContext::default();
    let img = layouts(101, 53, harsh, mixed_alpha(harsh)).remove(1).1;
    for align in [Align::Center, Align::Smart] {
        let ops = [Operation::DebugFocal, crop_op(30, 40, align.clone())];
        let (_, trace) =
            crate::crop::crop_to_aspect_traced(img.clone(), 30, 40, align.clone(), None, None);
        let capture = capture_from_trace(img.clone(), trace.expect("a crop traces"));
        let expected = crate::debugview::render(&capture);
        assert!(same(&expected, &run(&img, &ops, &ctx)), "{align:?}");
    }
}

#[test]
fn crop_size_checks_see_the_window_the_cut_had() {
    // Issue #654: the crop no longer materializes its window
    // before the closing resize, so that resize checks its
    // target against the window's dimensions instead of a
    // cut frame's. They are the same numbers: a target that
    // keeps the window as it is passes and comes back as the
    // cut, a target past the per-axis limit is refused where
    // it was, and a window that reaches past the frame is
    // sized as the clamped cut.
    let img = layouts(64, 48, harsh, mixed_alpha(harsh)).remove(0).1;
    let size = |w, h| ImageSize {
        width: Dimension::Pixels(w),
        height: Dimension::Pixels(h),
        flip_horizontal: false,
        flip_vertical: false,
    };
    let window = (8, 0, 48, 48);
    let cut = apply_size_windowed(
        img.clone(),
        &size(48, 48),
        None,
        1.0,
        SourceWindow::Cut(window),
    )
    .expect("the window's own size");
    assert!(same(&crate_cut(&img, window), &cut));
    let over = crate::url::MAX_DIMENSION + 1;
    let windowed = apply_size_windowed(
        img.clone(),
        &size(over, 48),
        None,
        1.0,
        SourceWindow::Cut(window),
    )
    .expect_err("past the limit");
    let two_step = apply_size(crate_cut(&img, window), &size(over, 48), None, 1.0)
        .expect_err("past the limit");
    assert_eq!(windowed.to_string(), two_step.to_string());
    // Proportional and capped sizes resolve against the window.
    let proportional = ImageSize {
        width: Dimension::Pixels(24),
        height: Dimension::Proportional,
        flip_horizontal: true,
        flip_vertical: false,
    };
    for cap in [
        None,
        Some(SizeCap::Fit(20, 20)),
        Some(SizeCap::Axes(30, 10)),
    ] {
        let windowed = apply_size_windowed(
            img.clone(),
            &proportional,
            cap,
            1.0,
            SourceWindow::Cut(window),
        )
        .expect("a size inside the limits");
        let two_step = apply_size(crate_cut(&img, window), &proportional, cap, 1.0)
            .expect("a size inside the limits");
        assert!(same(&two_step, &windowed));
    }
    // A window past the frame: both paths clamp it first. A
    // target of the size the window names is then a resize
    // of the clamped cut, not the cut as it stands.
    let past = (40, 30, 64, 48);
    assert_eq!(crate::crop::clamp_window(past, (64, 48)), (40, 30, 24, 18));
    for (tw, th) in [(12, 9), (64, 48), (24, 18)] {
        let windowed = apply_size_windowed(
            img.clone(),
            &size(tw, th),
            None,
            1.0,
            SourceWindow::Cut(past),
        )
        .expect("a clamped window");
        let two_step =
            apply_size(crate_cut(&img, past), &size(tw, th), None, 1.0).expect("a clamped window");
        assert!(same(&two_step, &windowed), "{tw}x{th}");
        assert_eq!((windowed.width(), windowed.height()), (tw, th));
    }
}

/// `apply_pad` as it stood before issue #656: the frame
/// cloned for the fit whatever the fill, the canvas built as
/// a `DynamicImage` and copied to RGBA8, and the fitted image
/// blended over it one pixel at a time by the crate.
fn old_apply_pad(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    fill: PadFill,
    directive_fill: Option<[u8; 3]>,
    cap: Option<(u32, u32)>,
    deadline: Option<std::time::Instant>,
) -> Result<(DynamicImage, Window), EngineError> {
    if target_w == 0 || target_h == 0 {
        let whole = (0, 0, img.width(), img.height());
        return Ok((img, whole));
    }
    check_output_size(target_w, target_h, (img.width(), img.height()))?;
    let fitted = apply_fit_in(
        img.clone(),
        &ImageSize {
            width: Dimension::Pixels(target_w),
            height: Dimension::Pixels(target_h),
            flip_horizontal: false,
            flip_vertical: false,
        },
        false,
        cap,
        1.0,
    )?;
    if fitted.width() == target_w && fitted.height() == target_h {
        return Ok((fitted, (0, 0, target_w, target_h)));
    }
    let color = match fill {
        PadFill::Color(c) => Some(c),
        PadFill::Directive => Some(directive_fill.unwrap_or([0, 0, 0])),
        PadFill::Blur => None,
    };
    let canvas = match color {
        Some(c) => DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            target_w,
            target_h,
            image::Rgba([c[0], c[1], c[2], 255]),
        )),
        None => {
            let sigma = ((target_w.max(target_h) as f32) / 16.0).clamp(2.0, 80.0);
            crate::filters::blur_fast_within(
                &img.resize_to_fill(target_w, target_h, image::imageops::FilterType::Triangle),
                sigma,
                deadline,
            )?
        }
    };
    let mut base = canvas.to_rgba8();
    let x = (target_w - fitted.width()) / 2;
    let y = (target_h - fitted.height()) / 2;
    image::imageops::overlay(&mut base, &fitted.to_rgba8(), x as i64, y as i64);
    let placed = (x, y, fitted.width(), fitted.height());
    Ok((DynamicImage::ImageRgba8(base), placed))
}

#[test]
fn pad_matches_the_clone_and_overlay_it_replaced() {
    // Issue #656: every layout (the ones without alpha are
    // written over the canvas by rows, the ones with alpha
    // still blend), frames from one pixel up, canvases that
    // bar the sides, bar the top, shrink, enlarge and match,
    // the four fills, with and without the `no_upscale` cap.
    // The image, and where it reports the fitted frame sits.
    let far = Some(std::time::Instant::now() + std::time::Duration::from_secs(3600));
    let mut cases = 0;
    for (w, h) in [(1, 1), (3, 2), (37, 20), (20, 37)] {
        for (name, img) in layouts(w, h, harsh, mixed_alpha(harsh)) {
            for (tw, th) in [(50, 50), (30, 50), (50, 30), (7, 3), (37, 20), (1, 1)] {
                for fill in [
                    PadFill::Color([255, 255, 255]),
                    PadFill::Color([17, 34, 51]),
                    PadFill::Directive,
                    PadFill::Blur,
                ] {
                    for cap in [None, Some((w, h))] {
                        // The blurred fill is slow in a debug
                        // build: it runs where its path
                        // differs, not on every combination.
                        let blurs = ["rgb8", "rgba8", "luma16"].contains(&name)
                            && cap.is_none()
                            && [(30, 50), (50, 30), (1, 1)].contains(&(tw, th));
                        if fill == PadFill::Blur && !blurs {
                            continue;
                        }
                        let directive = (tw > th).then_some([9, 8, 7]);
                        let old = old_apply_pad(img.clone(), tw, th, fill, directive, cap, far)
                            .expect("a pad inside the limits");
                        let new = apply_pad(img.clone(), tw, th, fill, directive, cap, far)
                            .expect("a pad inside the limits");
                        assert!(
                            same(&old.0, &new.0),
                            "{name} {w}x{h} to {tw}x{th} {fill:?} {cap:?}"
                        );
                        assert_eq!(old.1, new.1, "{name} {w}x{h} to {tw}x{th} {fill:?} {cap:?}");
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 4 * 10 * 6 * 3 * 2 + 4 * 3 * 3);
    // A zero side pads nothing, and a canvas past the limit
    // is refused before the fit, as it was.
    let img = layouts(37, 20, harsh, mixed_alpha(harsh)).remove(0).1;
    let (same_img, placed) =
        apply_pad(img.clone(), 0, 50, PadFill::Blur, None, None, None).expect("no canvas");
    assert!(same(&img, &same_img));
    assert_eq!(placed, (0, 0, 37, 20));
    let over = crate::url::MAX_DIMENSION + 1;
    let refusal = |pad: Result<(DynamicImage, Window), EngineError>| {
        pad.expect_err("a canvas past the limit").to_string()
    };
    assert_eq!(
        refusal(old_apply_pad(
            img.clone(),
            over,
            9,
            PadFill::Directive,
            None,
            None,
            None
        )),
        refusal(apply_pad(
            img.clone(),
            over,
            9,
            PadFill::Directive,
            None,
            None,
            None
        )),
    );
}

#[test]
fn placing_falls_back_to_the_blend_wherever_a_row_copy_is_not_it() {
    // Issue #656: the row copy stands for the crate's overlay
    // only for a top image without alpha that lies inside
    // the canvas. One with alpha blends, and one the canvas
    // clips is clipped, both by the crate; all three match
    // the crate's overlay of the RGBA8 form.
    let canvas = || {
        image::RgbaImage::from_fn(40, 30, |x, y| {
            image::Rgba([
                smooth(x, y, 0),
                smooth(x, y, 1),
                smooth(x, y, 2),
                smooth(y, x, 3),
            ])
        })
    };
    let mut cases = 0;
    for (name, top) in layouts(20, 12, harsh, mixed_alpha(harsh)) {
        for (x, y) in [(0, 0), (20, 18), (10, 9), (21, 3), (5, 19), (40, 30)] {
            let mut expected = canvas();
            image::imageops::overlay(&mut expected, &top.to_rgba8(), x as i64, y as i64);
            let mut ours = canvas();
            place_on(&mut ours, top.clone(), x, y);
            assert_eq!(expected.as_raw(), ours.as_raw(), "{name} at {x},{y}");
            cases += 1;
        }
    }
    assert_eq!(cases, 60);
}
