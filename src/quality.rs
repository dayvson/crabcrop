//! Content-aware quality selection.
//!
//! [`auto_quality`] picks an encoding quality from the image itself:
//! flat images compress well at low quality, while detailed or noisy
//! images need more. The heuristic is deliberately simple and
//! deterministic (no machine learning): the mean absolute difference
//! between neighboring luma samples, measured on a thumbnail so the
//! cost is bounded.
//!
//! The heuristic is only the first guess. For the quality-bearing
//! formats with a decoder to verify against, [`encode_to_target`]
//! closes the loop: encode, decode, score with SSIMULACRA2
//! ([`ssimulacra2`]), and bisect on quality until the score lands
//! in the target band.

use image::DynamicImage;

/// Lowest quality automatic selection will pick.
pub const MIN_AUTO_QUALITY: u8 = 50;
/// Highest quality automatic selection will pick.
pub const MAX_AUTO_QUALITY: u8 = 95;

/// Default SSIMULACRA2 target for verified `quality(auto)` encodes.
/// On the metric's scale 100 is identical and 75 is the "high
/// quality, small artifacts" point the format's authors calibrate
/// against; deployments can move it with `CRABCROP_SSIM_TARGET`.
pub const DEFAULT_SSIM_TARGET: f64 = 75.0;

/// How far above the target a score may land and still be
/// accepted without another encode. The tolerance is asymmetric
/// on purpose: below target is a quality defect, above target
/// only wastes bytes.
pub const SSIM_OVER_TOLERANCE: f64 = 10.0;

/// Total encodes the perceptual loop may spend on one image,
/// first guess included.
pub const MAX_PERCEPTUAL_ATTEMPTS: usize = 4;

/// The result of a verified encode: the bytes to serve, the
/// quality that produced them, and the SSIMULACRA2 score they
/// measured at (`None` when the candidate could not be scored
/// and the loop served it unverified).
pub struct ScoredEncode {
    pub bytes: Vec<u8>,
    pub quality: u8,
    pub score: Option<f64>,
}

/// Score `distorted` against `reference` with SSIMULACRA2.
/// Returns `None` when the pair cannot be scored (size mismatch,
/// empty, beyond the metric's pixel limit, or a metric error):
/// scoring is verification, never a reason to fail a request.
pub fn ssimulacra2(reference: &DynamicImage, distorted: &DynamicImage) -> Option<f64> {
    if reference.width() != distorted.width() || reference.height() != distorted.height() {
        return None;
    }
    let (w, h) = (reference.width(), reference.height());
    if w == 0 || h == 0 || (w as u64 * h as u64) > fast_ssim2::MAX_IMAGE_PIXELS as u64 {
        return None;
    }
    let reference = reference.to_rgb8();
    let distorted = distorted.to_rgb8();
    let stride = w as usize * 3;
    let source = fast_ssim2::PixelSlice::new(
        reference.as_raw(),
        w,
        h,
        stride,
        fast_ssim2::PixelDescriptor::RGB8_SRGB,
    )
    .ok()?;
    let distorted = fast_ssim2::PixelSlice::new(
        distorted.as_raw(),
        w,
        h,
        stride,
        fast_ssim2::PixelDescriptor::RGB8_SRGB,
    )
    .ok()?;
    fast_ssim2::compute_ssimulacra2(&source, &distorted).ok()
}

/// Encode toward a SSIMULACRA2 `target` by bisection on quality,
/// starting from `first_guess`. `attempt` encodes at a quality
/// and returns the bytes plus their measured score (`None` when
/// unscored, which ends the search: that encode is served
/// unverified). At most [`MAX_PERCEPTUAL_ATTEMPTS`] encodes run.
/// The search stays inside the automatic range
/// ([`MIN_AUTO_QUALITY`]..=[`MAX_AUTO_QUALITY`]): the band yields
/// to the floor when even the lowest allowed quality scores above
/// it. A candidate is accepted when its score lands in
/// `[target, target + SSIM_OVER_TOLERANCE]`; if none does, the
/// smallest candidate that met the target wins, else the
/// best-scoring candidate.
pub fn encode_to_target<E, F>(
    first_guess: u8,
    target: f64,
    mut attempt: F,
) -> Result<ScoredEncode, E>
where
    F: FnMut(u8) -> Result<(Vec<u8>, Option<f64>), E>,
{
    let mut candidates: Vec<ScoredEncode> = Vec::new();
    // Bisection bounds: the highest quality known to score below
    // target, and the lowest known to score at or above it. The
    // sentinels sit one step outside the searchable range.
    let (mut lo, mut hi) = (MIN_AUTO_QUALITY as u16 - 1, MAX_AUTO_QUALITY as u16 + 1);
    let mut q = first_guess.clamp(MIN_AUTO_QUALITY, MAX_AUTO_QUALITY);
    for _ in 0..MAX_PERCEPTUAL_ATTEMPTS {
        let (bytes, score) = attempt(q)?;
        let Some(score) = score else {
            return Ok(ScoredEncode {
                bytes,
                quality: q,
                score: None,
            });
        };
        if (target..=target + SSIM_OVER_TOLERANCE).contains(&score) {
            return Ok(ScoredEncode {
                bytes,
                quality: q,
                score: Some(score),
            });
        }
        if score < target {
            lo = lo.max(q as u16);
        } else {
            hi = hi.min(q as u16);
        }
        candidates.push(ScoredEncode {
            bytes,
            quality: q,
            score: Some(score),
        });
        let next = ((lo + hi) / 2).clamp(MIN_AUTO_QUALITY as u16, MAX_AUTO_QUALITY as u16) as u8;
        if next == q {
            break;
        }
        q = next;
    }
    // No candidate landed in the band. Prefer the smallest bytes
    // among those that met the target; failing that, the highest
    // score achieved.
    let meeting = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.score.is_some_and(|s| s >= target))
        .min_by_key(|(_, c)| c.bytes.len())
        .map(|(i, _)| i);
    if let Some(pos) = meeting {
        return Ok(candidates.swap_remove(pos));
    }
    let best_pos = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.score
                .unwrap_or(f64::MIN)
                .partial_cmp(&b.score.unwrap_or(f64::MIN))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .expect("at least one scored candidate exists");
    Ok(candidates.swap_remove(best_pos))
}

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

    #[test]
    fn identical_images_score_100() {
        let img = noise();
        let score = ssimulacra2(&img, &img.clone()).expect("same-size images score");
        assert!(score > 99.9, "identity score = {score}");
    }

    #[test]
    fn degraded_images_score_lower() {
        let img = noise();
        let blurred = img.blur(2.0);
        let score = ssimulacra2(&img, &blurred).expect("same-size images score");
        // Severe distortion can push SSIMULACRA2 below zero; the
        // point is only that it lands far under high quality.
        assert!(score < 90.0, "blur score = {score}");
    }

    #[test]
    fn mismatched_sizes_do_not_score() {
        assert!(ssimulacra2(&solid(10), &DynamicImage::new_rgb8(10, 10)).is_none());
    }

    /// Run the loop against a fake `score = f(quality)` encoder and
    /// report (attempted qualities, outcome).
    fn run_loop(
        score_of: impl Fn(u8) -> f64,
        first_guess: u8,
        target: f64,
    ) -> (Vec<u8>, ScoredEncode) {
        let tried = std::cell::RefCell::new(Vec::new());
        let out = encode_to_target(first_guess, target, |q| {
            tried.borrow_mut().push(q);
            // Bytes shrink as quality drops, so size stands in for
            // the quality that produced them.
            Ok::<_, ()>((vec![0u8; q as usize * 10], Some(score_of(q))))
        })
        .unwrap();
        (tried.into_inner(), out)
    }

    #[test]
    fn loop_accepts_a_first_guess_in_band() {
        let (tried, out) = run_loop(|q| q as f64, 80, 75.0);
        assert_eq!(tried, vec![80]);
        assert_eq!(out.quality, 80);
        assert_eq!(out.score, Some(80.0));
    }

    #[test]
    fn loop_bisects_up_from_a_low_guess() {
        let (tried, out) = run_loop(|q| q as f64, 50, 75.0);
        assert_eq!(tried, vec![50, 73, 84]);
        assert_eq!(out.quality, 84);
        assert_eq!(out.score, Some(84.0));
    }

    #[test]
    fn loop_bisects_down_from_a_high_guess() {
        let (tried, out) = run_loop(|q| q as f64, 95, 75.0);
        // 95 overshoots the band (75..=85), 72 undershoots, 83
        // lands inside it.
        assert_eq!(tried, vec![95, 72, 83]);
        assert_eq!(out.quality, 83);
        assert_eq!(out.score, Some(83.0));
    }

    #[test]
    fn loop_respects_the_quality_floor() {
        // Every allowed quality scores above the band: the floor
        // binds and the first (smallest) encode is served.
        let (tried, out) = run_loop(|_| 99.0, 50, 75.0);
        assert_eq!(tried, vec![50]);
        assert_eq!(out.quality, 50);
        assert_eq!(out.score, Some(99.0));
    }

    #[test]
    fn loop_never_exceeds_the_attempt_bound() {
        // A target this encoder can never reach (max score 90).
        let (tried, out) = run_loop(|q| q as f64 * 0.9, 60, 99.0);
        assert_eq!(tried.len(), MAX_PERCEPTUAL_ATTEMPTS);
        // Best effort: the highest score achieved wins.
        assert!(out.score.is_some_and(|s| s > 80.0));
    }

    #[test]
    fn loop_serves_an_unscored_candidate_unverified() {
        let out =
            encode_to_target(70, 75.0, |_q| Ok::<_, ()>((vec![0u8; 7], None::<f64>))).unwrap();
        assert_eq!(out.quality, 70);
        assert_eq!(out.score, None);
        assert_eq!(out.bytes.len(), 7);
    }
}
