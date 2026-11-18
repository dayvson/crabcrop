//! Issue #739: the filters run their row bands on one pool, and
//! every banded filter writes the same bytes at every pool
//! size. Each case runs on one thread (one band, on the calling
//! thread) and on pools of 2, 3 and 8 threads, each pool
//! splitting a pass into as many bands; eight is past the four
//! bands any host splits a pass into, so the frame gets more
//! band edges than a server ever cuts. Every frame is large
//! enough for the filter to band at each size, and each case
//! checks that the pool's threads ran its bands.

use super::bands::on_test_pool;
use super::*;
use crate::url::FlipDirection;

/// The pool sizes past one thread the filters run at.
const POOL_SIZES: [usize; 3] = [2, 3, 8];

/// Grain in 0..64 for a pixel's channel: a hash of where it
/// is, so no two neighbouring rows repeat and a band edge in
/// the wrong place changes bytes.
fn grain(x: u32, y: u32, channel: u64) -> u32 {
    let mut v = ((u64::from(x) << 32) | u64::from(y)) ^ channel.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    v ^= v >> 30;
    v = v.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v ^= v >> 27;
    v = v.wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^= v >> 31;
    (v & 63) as u32
}

/// A photo-like RGB frame: smooth gradients under the grain.
fn photo(w: u32, h: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([
            (x * 180 / w + grain(x, y, 0)) as u8,
            (y * 170 / h + grain(x, y, 1) + 20) as u8,
            ((x + y) * 120 / (w + h) + grain(x, y, 2) + 40) as u8,
        ])
    }))
}

/// `img` with an alpha channel of its own grain.
fn with_alpha(img: &DynamicImage) -> DynamicImage {
    let mut rgba = img.to_rgba8();
    for (x, y, p) in rgba.enumerate_pixels_mut() {
        p[3] = (128 + 2 * grain(x, y, 3)) as u8;
    }
    DynamicImage::ImageRgba8(rgba)
}

/// The image's size and layout, then its samples' bytes.
fn bytes_of(img: &DynamicImage) -> Vec<u8> {
    let mut bytes = format!("{}x{} {:?} ", img.width(), img.height(), img.color()).into_bytes();
    bytes.extend_from_slice(img.as_bytes());
    bytes
}

/// `filters` applied to a copy of `img`, as bytes.
fn render(img: &DynamicImage, filters: &[Filter]) -> Vec<u8> {
    bytes_of(&apply_filters(img.clone(), filters).0)
}

/// Run `render` on one thread and on each pool size: every
/// size must hand at least as many bands to its pool's
/// threads as it has threads, and write the one-thread bytes.
fn at_every_pool_size(name: &str, render: impl Fn() -> Vec<u8>) {
    let (one, pooled) = on_test_pool(1, &render);
    assert_eq!(pooled, 0, "{name}: one thread handed bands to a pool");
    for threads in POOL_SIZES {
        let (bytes, pooled) = on_test_pool(threads, &render);
        assert!(
            pooled >= threads,
            "{name}: the pool of {threads} threads ran {pooled} bands"
        );
        assert_eq!(
            one.len(),
            bytes.len(),
            "{name}: {threads} threads wrote another length"
        );
        if let Some(at) = one.iter().zip(&bytes).position(|(a, b)| a != b) {
            panic!(
                "{name}: {threads} threads wrote byte {at} as {}, one thread as {}",
                bytes[at], one[at]
            );
        }
    }
}

#[test]
fn every_blur_writes_the_same_bytes_at_every_pool_size() {
    // A blur bands any frame of two rows or more. Here 61 rows
    // make bands of 5 to 31 rows, against kernels that reach 7
    // rows (sigma 2.5, the ring pass) and 58 rows (sigma 18,
    // the transient's two passes), in every sample type.
    let rgb = photo(97, 61);
    let rgba = with_alpha(&rgb);
    let layouts = [
        DynamicImage::ImageLuma8(rgb.to_luma8()),
        DynamicImage::ImageLumaA8(rgba.to_luma_alpha8()),
        rgb.clone(),
        rgba.clone(),
        DynamicImage::ImageRgb16(rgb.to_rgb16()),
        DynamicImage::ImageRgba16(rgba.to_rgba16()),
        DynamicImage::ImageRgb32F(rgb.to_rgb32f()),
        DynamicImage::ImageRgba32F(rgba.to_rgba32f()),
    ];
    for img in &layouts {
        for sigma in [2.5f32, 18.0] {
            at_every_pool_size(&format!("blur({sigma}) {:?}", img.color()), || {
                render(img, &[Filter::Blur(sigma)])
            });
        }
    }
    // The studio's alpha blur runs the float passes on gray.
    let gray = rgb.to_luma8();
    for sigma in [2.5f32, 18.0] {
        at_every_pool_size(&format!("blur_gray_within({sigma})"), || {
            blur_gray_within(&gray, sigma, None)
                .expect("no deadline")
                .into_raw()
        });
    }
}

#[test]
fn every_per_pixel_filter_writes_the_same_bytes_at_every_pool_size() {
    // These band a buffer of 3,000,000 subpixels and 256 rows
    // or more. At 1024x1000 the RGB copy that contrast,
    // saturation, vibrance and enhance work on is past that
    // too when the frame has alpha.
    let rgb = photo(1024, 1000);
    let rgba = with_alpha(&rgb);
    let filters = [
        Filter::Grayscale,
        Filter::Brighten(-37),
        Filter::Contrast(35.0),
        Filter::Gamma(2.2),
        Filter::Exposure(0.8),
        Filter::HighlightsShadows(40.0, -30.0),
        Filter::HueRotate(77),
        Filter::Saturation(1.6),
        Filter::Vibrance(45.0),
        Filter::Rgb(12, -30, 45),
        Filter::Curve(CurveChannel::Rgb, vec![(0, 10), (90, 140), (255, 230)]),
        Filter::Enhance,
        Filter::Sepia,
        Filter::Duotone([26, 26, 64], [255, 217, 160]),
        Filter::Monochrome([255, 128, 0]),
    ];
    for img in [&rgb, &rgba] {
        for filter in &filters {
            at_every_pool_size(&format!("{filter:?} {:?}", img.color()), || {
                render(img, std::slice::from_ref(filter))
            });
        }
    }
}

#[test]
fn every_turn_and_flip_writes_the_same_bytes_at_every_pool_size() {
    // An arbitrary angle bands a canvas of 128 rows or more.
    let small = photo(300, 200);
    for (degrees, img, fill) in [
        (30.0f32, small.clone(), None),
        (-77.5, with_alpha(&small), None),
        (200.0, small.clone(), Some([9, 99, 199])),
    ] {
        at_every_pool_size(&format!("rotate({degrees}) {:?}", img.color()), || {
            bytes_of(&apply_filters_filled(img.clone(), &[Filter::Rotate(degrees)], fill).0)
        });
    }
    // The half and three-quarter turns and the flips band a
    // frame of 4,000,000 subpixels or more, where they leave
    // the image crate's loops. The quarter turn runs on one
    // thread at every size (issue #648).
    for img in [photo(1160, 1160), with_alpha(&photo(1010, 1000))] {
        for filter in [
            Filter::Rotate(180.0),
            Filter::Rotate(270.0),
            Filter::Flip(FlipDirection::Horizontal),
            Filter::Flip(FlipDirection::Vertical),
        ] {
            at_every_pool_size(&format!("{filter:?} {:?}", img.color()), || {
                render(&img, std::slice::from_ref(&filter))
            });
        }
    }
}
