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
