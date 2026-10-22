//! Content-aware quality selection.
//!
//! [`auto_quality`] picks an encoding quality from the image itself:
//! flat images compress well at low quality, while detailed or noisy
//! images need more. The heuristic is deliberately simple and
//! deterministic (no machine learning): the mean absolute difference
//! between neighboring luma samples, measured on a thumbnail so the
//! cost is bounded. It lives in this one function so a perceptual
//! metric can replace it later without touching callers.

use image::DynamicImage;

/// Lowest quality automatic selection will pick.
pub const MIN_AUTO_QUALITY: u8 = 50;
/// Highest quality automatic selection will pick.
pub const MAX_AUTO_QUALITY: u8 = 95;

/// Mean neighbor difference (0..=1) at which selection saturates.
const SATURATION_POINT: f64 = 0.20;

/// Choose an encoding quality for `img`, in
/// `MIN_AUTO_QUALITY..=MAX_AUTO_QUALITY`.
pub fn auto_quality(img: &DynamicImage) -> u8 {
    // Analyze a thumbnail: bounded cost, and detail density survives.
    // Never upscale: `thumbnail` would pad small images with
    // nearest-neighbor blocks and dilute the neighbor differences.
    let thumb = if img.width() > 256 || img.height() > 256 {
        img.thumbnail(256, 256).to_luma8()
    } else {
        img.to_luma8()
    };
    let (w, h) = (thumb.width(), thumb.height());
    if w < 2 && h < 2 {
        return MIN_AUTO_QUALITY;
    }
    let mut sum: u64 = 0;
    let mut count: u64 = 0;
    for y in 0..h {
        for x in 0..w {
            let p = thumb.get_pixel(x, y)[0] as i32;
            if x + 1 < w {
                sum += (p - thumb.get_pixel(x + 1, y)[0] as i32).unsigned_abs() as u64;
                count += 1;
            }
            if y + 1 < h {
                sum += (p - thumb.get_pixel(x, y + 1)[0] as i32).unsigned_abs() as u64;
                count += 1;
            }
        }
    }
    let mean_diff = (sum as f64 / count as f64) / 255.0;
    let t = (mean_diff / SATURATION_POINT).clamp(0.0, 1.0);
    MIN_AUTO_QUALITY + ((MAX_AUTO_QUALITY - MIN_AUTO_QUALITY) as f64 * t).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn solid(v: u8) -> DynamicImage {
        DynamicImage::ImageLuma8(image::GrayImage::from_pixel(64, 64, image::Luma([v])))
    }

    /// Per-pixel hash noise: neighboring samples are decorrelated, so
    /// neighbor differences behave like independent uniforms.
    fn noise() -> DynamicImage {
        fn hash(x: u32, y: u32) -> u8 {
            let mut n = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
            n ^= n >> 13;
            n = n.wrapping_mul(0xC2B2_AE35);
            (n >> 24) as u8
        }
        let mut img = DynamicImage::new_rgb8(64, 64);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = Rgb([hash(x, y), hash(x + 101, y + 57), hash(x + 7, y + 201)]);
        }
        img
    }

    /// Two-pixel-wide vertical stripes: edges on half the neighbor
    /// pairs, silence on the rest. Moderate detail.
    fn stripes() -> DynamicImage {
        let mut img = DynamicImage::new_luma8(64, 64);
        for (x, _y, p) in img.as_mut_luma8().unwrap().enumerate_pixels_mut() {
            p[0] = if (x / 2) % 2 == 0 { 100 } else { 156 };
        }
        img
    }

    #[test]
    fn flat_image_gets_the_minimum() {
        assert_eq!(auto_quality(&solid(128)), MIN_AUTO_QUALITY);
    }

    #[test]
    fn noise_gets_high_quality() {
        let q = auto_quality(&noise());
        assert!(q >= 90, "noise quality = {q}");
    }

    #[test]
    fn moderate_detail_lands_between() {
        let q = auto_quality(&stripes());
        assert!(q > MIN_AUTO_QUALITY + 5 && q < 90, "stripes quality = {q}");
    }

    #[test]
    fn detail_beats_flat() {
        assert!(auto_quality(&noise()) > auto_quality(&solid(200)));
        assert!(auto_quality(&stripes()) > auto_quality(&solid(200)));
    }
}
