//! Studio mode (issue #197): the `studio(product,...)`
//! filter's composite. One filter names an outcome, a
//! listing-ready product shot, and expands deterministically
//! into steps the engine already trusts: the segmentation
//! model's mask, a defringe pass that decontaminates edge
//! colors, a backdrop with a synthesized contact shadow, and
//! a pad that scales and centers the subject on the canvas
//! in effect at the filter's position in the chain.
//!
//! Everything here is pure pixel work over the decoded image
//! and the provider's mask; the engine owns the provider call
//! and the gates. The outcome carries a [`StudioRecord`] of
//! what ran (resolved parameters, steps, fallbacks), which
//! meta responses surface so the operation stays accountable
//! instead of magical.

use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage, imageops::FilterType};

use crate::url::{StudioBg, StudioShadow, StudioSpec};

/// Whether studio stage timing is printed (issue #220): on
/// when `CRABCROP_STUDIO_TIMING` is set to any value.
/// Diagnostics go to stderr, never to the response path,
/// and the flag is read once per process.
pub(crate) fn timing_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CRABCROP_STUDIO_TIMING").is_some())
}

/// Print one stage's elapsed time when timing is on.
pub(crate) fn timing_mark(stage: &str, start: std::time::Instant) {
    if timing_on() {
        eprintln!(
            "[studio-timing] {stage}: {:.1} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// The mask value at or above which a pixel counts as
/// subject: the geometry (bounding box, subject presence)
/// is settled on this threshold, while compositing uses the
/// full alpha ramp.
const SUBJECT_THRESHOLD: u8 = 128;

/// What studio resolved and ran, reported in meta (issue
/// #197). The server serializes it verbatim; every value is
/// a resolved fact about this render, never a request
/// echo that could disagree with the output.
#[derive(Debug, Clone, PartialEq)]
pub struct StudioRecord {
    /// The resolved backdrop color. `None` is `bg:none`:
    /// the subject kept its transparency.
    pub bg: Option<[u8; 3]>,
    /// Where the backdrop came from: `"argument"` (named in
    /// the URL), `"brand"` (the operator brand color), or
    /// `"default"` (plain white, because neither applied).
    pub bg_from: &'static str,
    /// The shadow the URL asked for, whether or not one was
    /// synthesized (a `bg:none` composite skips it).
    pub shadow: StudioShadow,
    /// The pad margin applied, percent per side.
    pub pad: u8,
    /// The expansion steps that ran, in pipeline order:
    /// drawn from `mask`, `defringe`, `backdrop`, `shadow`,
    /// `pad`, and `fallback` (the no-subject path).
    pub steps: Vec<&'static str>,
    /// The fallbacks taken, in the order they were
    /// decided: drawn from `no_subject` and
    /// `bg_brand_unset`.
    pub fallbacks: Vec<&'static str>,
    /// The mask provider's confidence in its mask, when the
    /// provider exposes one (see
    /// [`crate::ai::MaskProvider::confidence`]).
    pub confidence: Option<f32>,
}

/// The composite's output: the rendered image and its
/// record.
pub struct StudioOutcome {
    pub image: DynamicImage,
    pub record: StudioRecord,
}

/// A backdrop resolution before pixels: the color to fill
/// (or `None` for transparency), its provenance, and
/// whether the brand resolution fell back to white.
struct ResolvedBg {
    color: Option<[u8; 3]>,
    from: &'static str,
    brand_fallback: bool,
}

fn resolve_bg(spec: &StudioSpec, brand_color: Option<[u8; 3]>) -> ResolvedBg {
    match spec.bg {
        StudioBg::White => ResolvedBg {
            color: Some([255, 255, 255]),
            from: "argument",
            brand_fallback: false,
        },
        StudioBg::Color(c) => ResolvedBg {
            color: Some(c),
            from: "argument",
            brand_fallback: false,
        },
        StudioBg::None => ResolvedBg {
            // Transparency has no provenance to report
            // beyond the argument itself.
            color: None,
            from: "argument",
            brand_fallback: false,
        },
        StudioBg::Brand => match brand_color {
            Some(c) => ResolvedBg {
                color: Some(c),
                from: "brand",
                brand_fallback: false,
            },
            None => ResolvedBg {
                color: Some([255, 255, 255]),
                from: "default",
                brand_fallback: true,
            },
        },
        StudioBg::Default => match brand_color {
            Some(c) => ResolvedBg {
                color: Some(c),
                from: "brand",
                brand_fallback: false,
            },
            None => ResolvedBg {
                color: Some([255, 255, 255]),
                from: "default",
                brand_fallback: false,
            },
        },
    }
}

/// The inclusive bounding box of the mask pixels at or
/// above `threshold`, as (x, y, width, height). `None`
/// when no pixel qualifies.
fn mask_bbox(mask: &GrayImage, threshold: u8) -> Option<(u32, u32, u32, u32)> {
    let (mut min_x, mut min_y) = (u32::MAX, u32::MAX);
    let (mut max_x, mut max_y) = (0u32, 0u32);
    let mut any = false;
    for (x, y, px) in mask.enumerate_pixels() {
        if px[0] >= threshold {
            any = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    any.then(|| (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

/// Decontaminate edge colors (pipeline step 2). A photo's
/// edge pixels are blends of subject and background; once
/// the mask cuts the subject out, those blends would ride
/// along as a halo of the old background. For every
/// partially covered pixel, estimate the local background
/// color and un-mix it: `fg = (color - (1 - a) * bg) / a`,
/// with the divisor floored so faint pixels do not amplify
/// noise into color spikes.
///
/// The background estimate is local: the mean color of the
/// fully uncovered pixels in the pixel's grid cell and its
/// neighbors (the grid is at most 64 by 64 cells), falling
/// back to the whole image's uncovered mean, and to no
/// change at all when the mask covers everything.
fn defringe(img: &DynamicImage, mask: &GrayImage) -> DynamicImage {
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    // Global uncovered mean, the last-resort estimate.
    let (mut total, mut count) = ([0u64; 3], 0u64);
    for (px, m) in rgba.pixels().zip(mask.pixels()) {
        if m[0] == 0 {
            for c in 0..3 {
                total[c] += px[c] as u64;
            }
            count += 1;
        }
    }
    if count == 0 {
        return img.clone();
    }
    let global = [
        (total[0] / count) as f32,
        (total[1] / count) as f32,
        (total[2] / count) as f32,
    ];
    // Grid of uncovered sums: cell size chosen so neither
    // axis exceeds 64 cells.
    let cell = (w.max(h) / 64).max(1);
    let (gw, gh) = (w.div_ceil(cell), h.div_ceil(cell));
    let mut sums = vec![[0u64; 3]; (gw * gh) as usize];
    let mut counts = vec![0u64; (gw * gh) as usize];
    for (x, y, px) in rgba.enumerate_pixels() {
        if mask.get_pixel(x, y)[0] == 0 {
            let i = (y / cell * gw + x / cell) as usize;
            for c in 0..3 {
                sums[i][c] += px[c] as u64;
            }
            counts[i] += 1;
        }
    }
    let local_bg = |x: u32, y: u32| -> [f32; 3] {
        let (cx, cy) = (x / cell, y / cell);
        let (mut s, mut n) = ([0u64; 3], 0u64);
        for gy in cy.saturating_sub(1)..=(cy + 1).min(gh - 1) {
            for gx in cx.saturating_sub(1)..=(cx + 1).min(gw - 1) {
                let i = (gy * gw + gx) as usize;
                for c in 0..3 {
                    s[c] += sums[i][c];
                }
                n += counts[i];
            }
        }
        if n == 0 {
            global
        } else {
            [
                s[0] as f32 / n as f32,
                s[1] as f32 / n as f32,
                s[2] as f32 / n as f32,
            ]
        }
    };
    let mut out = rgba;
    for (x, y, px) in out.enumerate_pixels_mut() {
        let m = mask.get_pixel(x, y)[0];
        if m == 0 || m == 255 {
            continue;
        }
        let a = m as f32 / 255.0;
        let divisor = a.max(0.2);
        let bg = local_bg(x, y);
        for c in 0..3 {
            let fg = (px[c] as f32 - (1.0 - a) * bg[c]) / divisor;
            px[c] = fg.round().clamp(0.0, 255.0) as u8;
        }
    }
    DynamicImage::ImageRgba8(out)
}

/// Scale `piece` (a crop of the working image) so its
/// subject box lands centered inside the canvas's pad
/// margins, and return the scaled piece plus the canvas
/// offset its top-left corner goes to. The subject box is
/// (x, y, w, h) in the piece's own coordinates.
fn place(
    piece: &RgbaImage,
    subject: (u32, u32, u32, u32),
    canvas: (u32, u32),
    pad: u8,
) -> (RgbaImage, i64, i64) {
    let (w, h) = canvas;
    let (sx, sy, sw, sh) = subject;
    // The inner box the subject must fit: the canvas less
    // the pad margin on every side. pad tops out at 40,
    // so this never inverts.
    let keep = (100.0 - 2.0 * pad as f32) / 100.0;
    let (iw, ih) = (w as f32 * keep, h as f32 * keep);
    let scale = (iw / sw as f32).min(ih / sh as f32);
    let scaled = image::imageops::resize(
        piece,
        ((piece.width() as f32 * scale).round() as u32).max(1),
        ((piece.height() as f32 * scale).round() as u32).max(1),
        FilterType::Triangle,
    );
    // Put the subject box's center on the canvas center:
    // the offset is the canvas center less the scaled
    // distance from the piece's origin to the box center.
    let bcx = (sx as f32 + sw as f32 / 2.0) * scale;
    let bcy = (sy as f32 + sh as f32 / 2.0) * scale;
    let dx = (w as f32 / 2.0 - bcx).round() as i64;
    let dy = (h as f32 / 2.0 - bcy).round() as i64;
    (scaled, dx, dy)
}

/// Synthesize the contact shadow (part of pipeline step
/// 3): the placed subject's alpha, blurred, dimmed by the
/// strength's peak factor, and offset down. Returned as an
/// RGBA layer the size of the canvas.
fn shadow_layer(
    alpha: &GrayImage,
    at: (i64, i64),
    canvas: (u32, u32),
    shadow: StudioShadow,
) -> RgbaImage {
    let (w, h) = canvas;
    let min_dim = w.min(h) as f32;
    let (sigma, peak) = match shadow {
        StudioShadow::Soft => ((min_dim * 0.02).max(1.0), 0.35),
        StudioShadow::Strong => ((min_dim * 0.006).max(1.0), 0.55),
        StudioShadow::None => unreachable!("shadow layers are only built for soft or strong"),
    };
    let blurred = image::imageops::blur(alpha, sigma);
    let drop = ((h as f32 * 0.01).round() as i64).max(1);
    let mut layer = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
    for (x, y, px) in blurred.enumerate_pixels() {
        let a = (px[0] as f32 / 255.0 * peak * 255.0).round() as u8;
        if a == 0 {
            continue;
        }
        let (tx, ty) = (x as i64 + at.0, y as i64 + at.1 + drop);
        if tx >= 0 && ty >= 0 && tx < w as i64 && ty < h as i64 {
            layer.put_pixel(tx as u32, ty as u32, Rgba([0, 0, 0, a]));
        }
    }
    layer
}

/// Run the studio expansion over `img` with the provider's
/// `mask`: resolve the backdrop, cut and defringe the
/// subject (or take the no-subject fallback), synthesize
/// the shadow, and place everything on the canvas the
/// image already occupies.
pub fn composite(
    img: &DynamicImage,
    mask: &GrayImage,
    confidence: Option<f32>,
    spec: &StudioSpec,
    brand_color: Option<[u8; 3]>,
) -> StudioOutcome {
    let (w, h) = (img.width(), img.height());
    let resolved = resolve_bg(spec, brand_color);
    let mut fallbacks: Vec<&'static str> = Vec::new();
    if resolved.brand_fallback {
        fallbacks.push("bg_brand_unset");
    }
    let mut steps: Vec<&'static str> = vec!["mask"];

    // Fit the mask to the image, exactly as `apply_mask`
    // does: the provider answers at its own resolution.
    let t_stage = std::time::Instant::now();
    let fitted;
    let mask = if mask.width() == w && mask.height() == h {
        mask
    } else {
        fitted = image::imageops::resize(mask, w, h, FilterType::Triangle);
        &fitted
    };

    let subject = mask_bbox(mask, SUBJECT_THRESHOLD);
    timing_mark("mask_prep", t_stage);
    let (piece, subject_in_piece, shadow_ok) = match subject {
        Some(bbox) => {
            steps.push("defringe");
            let t_stage = std::time::Instant::now();
            let masked = crate::ai::apply_mask(img, mask);
            let clean = defringe(&masked, mask).to_rgba8();
            timing_mark("defringe", t_stage);
            // Crop to the covered region: nothing outside
            // it contributes pixels, and the crop keeps the
            // pad scaling cheap. The subject box is
            // re-based into the crop's coordinates.
            let region = mask_bbox(mask, 1).expect("a subject implies covered pixels");
            let (rx, ry, rw, rh) = region;
            let crop = image::imageops::crop_imm(&clean, rx, ry, rw, rh).to_image();
            (
                crop,
                (bbox.0 - rx, bbox.1 - ry, bbox.2, bbox.3),
                spec.shadow != StudioShadow::None,
            )
        }
        None => {
            // No subject: serve the original image, padded
            // onto the backdrop, and say so in the record.
            steps.push("fallback");
            fallbacks.push("no_subject");
            (img.to_rgba8(), (0, 0, w, h), false)
        }
    };
    steps.push("backdrop");

    let t_stage = std::time::Instant::now();
    let (scaled, dx, dy) = place(&piece, subject_in_piece, (w, h), spec.pad);
    timing_mark("place", t_stage);
    let mut canvas = match resolved.color {
        Some(c) => RgbaImage::from_pixel(w, h, Rgba([c[0], c[1], c[2], 255])),
        None => RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0])),
    };
    // The shadow rides under the subject, and only over a
    // backdrop: over transparency there is nothing for it
    // to darken.
    if shadow_ok && resolved.color.is_some() {
        let t_stage = std::time::Instant::now();
        let alpha = GrayImage::from_fn(scaled.width(), scaled.height(), |x, y| {
            Luma([scaled.get_pixel(x, y)[3]])
        });
        let layer = shadow_layer(&alpha, (dx, dy), (w, h), spec.shadow);
        image::imageops::overlay(&mut canvas, &layer, 0, 0);
        timing_mark("shadow", t_stage);
        steps.push("shadow");
    }
    let t_stage = std::time::Instant::now();
    image::imageops::overlay(&mut canvas, &scaled, dx, dy);
    timing_mark("overlay", t_stage);
    steps.push("pad");

    StudioOutcome {
        image: DynamicImage::ImageRgba8(canvas),
        record: StudioRecord {
            bg: resolved.color,
            bg_from: resolved.from,
            shadow: spec.shadow,
            pad: spec.pad,
            steps,
            fallbacks,
            confidence,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageBuffer;

    const GREEN: [u8; 3] = [30, 160, 30];
    const RED: [u8; 3] = [200, 30, 30];
    const BLEND: [u8; 3] = [115, 95, 30];

    /// A product photo on a 100x80 canvas: a red square
    /// (the product) at 30..70 x 20..60 on a green backdrop,
    /// with a one-pixel blended ring where the photo itself
    /// mixed the two colors, and the mask to match: 255
    /// inside the square, 128 on the ring, 0 outside.
    fn product_photo() -> (DynamicImage, GrayImage) {
        let img = ImageBuffer::from_fn(100, 80, |x, y| {
            let in_sq = (30..70).contains(&x) && (20..60).contains(&y);
            let in_core = (31..69).contains(&x) && (21..59).contains(&y);
            let c = if in_core {
                RED
            } else if in_sq {
                BLEND
            } else {
                GREEN
            };
            Rgba([c[0], c[1], c[2], 255])
        });
        let mask = GrayImage::from_fn(100, 80, |x, y| {
            let in_sq = (30..70).contains(&x) && (20..60).contains(&y);
            let in_core = (31..69).contains(&x) && (21..59).contains(&y);
            Luma([if in_core {
                255
            } else if in_sq {
                128
            } else {
                0
            }])
        });
        (DynamicImage::ImageRgba8(img), mask)
    }

    fn spec(bg: StudioBg, shadow: StudioShadow, pad: u8) -> StudioSpec {
        StudioSpec { bg, shadow, pad }
    }

    fn px(img: &DynamicImage, x: u32, y: u32) -> [u8; 4] {
        img.to_rgba8().get_pixel(x, y).0
    }

    #[test]
    fn white_backdrop_is_pure_and_the_edge_is_defringed() {
        let (img, mask) = product_photo();
        let out = composite(
            &img,
            &mask,
            None,
            &spec(StudioBg::White, StudioShadow::None, 0),
            None,
        );
        // The corners are exactly the backdrop color: no
        // trace of the old green anywhere the subject is not.
        for (x, y) in [(0, 0), (99, 0), (0, 79), (99, 79), (5, 40), (94, 40)] {
            assert_eq!(px(&out.image, x, y), [255, 255, 255, 255], "corner {x},{y}");
        }
        // Pad 0 scales the 40x40 subject (at 30..70, 20..60)
        // to fill the frame's short axis: it lands at
        // x 10..90, y 0..80. The center is full-strength red.
        assert_eq!(px(&out.image, 50, 40), [RED[0], RED[1], RED[2], 255]);
        // A pixel on the scaled blended ring (the ring maps
        // to the subject's rim): the defringe un-mixed the
        // green out of it, so it reads as red over white,
        // not as the muddy blend over white. Without the
        // defringe the green channel would sit near 175;
        // decontaminated it lands near 142.
        let rim = (0..100)
            .map(|x| px(&out.image, x, 40))
            .filter(|p| p[0] > 150 && p[0] < 250 && p[1] > 100)
            .collect::<Vec<_>>();
        assert!(!rim.is_empty(), "expected a rim of half-strength red");
        assert!(
            rim.iter().all(|p| p[1] <= 155),
            "rim pixels carry a green halo: {rim:?}"
        );
        assert_eq!(out.record.steps, ["mask", "defringe", "backdrop", "pad"]);
        assert!(out.record.fallbacks.is_empty());
        assert_eq!(out.record.bg, Some([255, 255, 255]));
        assert_eq!(out.record.bg_from, "argument");
    }

    #[test]
    fn shadow_darkens_below_the_subject_and_none_does_not() {
        let (img, mask) = product_photo();
        let soft = composite(
            &img,
            &mask,
            None,
            &spec(StudioBg::White, StudioShadow::Soft, 10),
            None,
        );
        assert!(soft.record.steps.contains(&"shadow"));
        // The subject lands at y 8..72 (pad 10 on 100x80).
        // The row just under its bottom edge carries the
        // blurred, dropped shadow: darker than the backdrop.
        // Rows above the subject and far below it stay pure.
        let under = px(&soft.image, 50, 72);
        assert!(
            under[0] < 250 && under[1] < 250 && under[2] < 250,
            "soft shadow missing under the subject: {under:?}"
        );
        assert_eq!(px(&soft.image, 50, 4), [255, 255, 255, 255]);
        assert_eq!(px(&soft.image, 50, 79), [255, 255, 255, 255]);
        let strong = composite(
            &img,
            &mask,
            None,
            &spec(StudioBg::White, StudioShadow::Strong, 10),
            None,
        );
        let under_strong = px(&strong.image, 50, 72);
        assert!(
            under_strong[0] < 250,
            "strong shadow missing under the subject: {under_strong:?}"
        );
        let none = composite(
            &img,
            &mask,
            None,
            &spec(StudioBg::White, StudioShadow::None, 10),
            None,
        );
        assert!(!none.record.steps.contains(&"shadow"));
        assert_eq!(px(&none.image, 50, 72), [255, 255, 255, 255]);
    }

    #[test]
    fn bg_none_keeps_transparency_and_skips_the_shadow() {
        let (img, mask) = product_photo();
        let out = composite(
            &img,
            &mask,
            None,
            &spec(StudioBg::None, StudioShadow::Soft, 10),
            None,
        );
        assert_eq!(out.record.bg, None);
        assert!(!out.record.steps.contains(&"shadow"));
        // The corner is fully transparent, the subject is
        // opaque red, and nothing darkened the transparent
        // field below the subject.
        assert_eq!(px(&out.image, 0, 0)[3], 0);
        assert_eq!(px(&out.image, 50, 40), [RED[0], RED[1], RED[2], 255]);
        assert_eq!(px(&out.image, 50, 75)[3], 0);
    }

    #[test]
    fn pad_geometry_holds_at_0_10_and_40() {
        let (img, mask) = product_photo();
        // Probe the middle row: the leftmost and rightmost
        // non-backdrop pixels are the placed subject's rim.
        let extent = |pad: u8| {
            let out = composite(
                &img,
                &mask,
                None,
                &spec(StudioBg::White, StudioShadow::None, pad),
                None,
            );
            let row: Vec<[u8; 4]> = (0..100).map(|x| px(&out.image, x, 40)).collect();
            let first = row.iter().position(|p| *p != [255, 255, 255, 255]).unwrap();
            let last = row
                .iter()
                .rposition(|p| *p != [255, 255, 255, 255])
                .unwrap();
            (first, last)
        };
        // The subject box is 40 wide on a 100-wide canvas.
        // Pad 0: it fills the short axis and lands 10..90.
        let (l, r) = extent(0);
        assert!(
            (9..=11).contains(&l) && (88..=90).contains(&r),
            "pad 0: {l}..{r}"
        );
        // Pad 10: the subject takes 80 percent, landing
        // inside 10..90 on this row's axis and 8..72 scaled
        // by the short axis: 18..82.
        let (l, r) = extent(10);
        assert!(
            (17..=19).contains(&l) && (80..=82).contains(&r),
            "pad 10: {l}..{r}"
        );
        // Pad 40: the subject takes 20 percent, centered:
        // 42..58.
        let (l, r) = extent(40);
        assert!(
            (41..=43).contains(&l) && (56..=58).contains(&r),
            "pad 40: {l}..{r}"
        );
    }

    #[test]
    fn brand_resolution_and_its_fallback_are_recorded() {
        let (img, mask) = product_photo();
        let brand = [245, 240, 232];
        // Explicit brand, configured: the brand color, from
        // the brand setting, no fallback.
        let out = composite(
            &img,
            &mask,
            None,
            &spec(StudioBg::Brand, StudioShadow::None, 10),
            Some(brand),
        );
        assert_eq!(px(&out.image, 0, 0), [brand[0], brand[1], brand[2], 255]);
        assert_eq!(out.record.bg_from, "brand");
        assert!(out.record.fallbacks.is_empty());
        // Explicit brand, not configured: white, and the
        // resolution is recorded as a fallback.
        let out = composite(
            &img,
            &mask,
            None,
            &spec(StudioBg::Brand, StudioShadow::None, 10),
            None,
        );
        assert_eq!(px(&out.image, 0, 0), [255, 255, 255, 255]);
        assert_eq!(out.record.bg_from, "default");
        assert_eq!(out.record.fallbacks, ["bg_brand_unset"]);
        // Unnamed backdrop: the brand color when configured
        // (no fallback: nothing was promised), plain
        // default white when not.
        let out = composite(
            &img,
            &mask,
            None,
            &spec(StudioBg::Default, StudioShadow::None, 10),
            Some(brand),
        );
        assert_eq!(px(&out.image, 0, 0), [brand[0], brand[1], brand[2], 255]);
        assert_eq!(out.record.bg_from, "brand");
        let out = composite(
            &img,
            &mask,
            None,
            &spec(StudioBg::Default, StudioShadow::None, 10),
            None,
        );
        assert_eq!(px(&out.image, 0, 0), [255, 255, 255, 255]);
        assert_eq!(out.record.bg_from, "default");
        assert!(out.record.fallbacks.is_empty());
        // A named color is its own provenance.
        let out = composite(
            &img,
            &mask,
            None,
            &spec(StudioBg::Color([245, 240, 232]), StudioShadow::None, 10),
            None,
        );
        assert_eq!(out.record.bg_from, "argument");
        assert_eq!(out.record.bg, Some([245, 240, 232]));
    }

    #[test]
    fn no_subject_serves_the_padded_original_and_records_it() {
        let (img, _) = product_photo();
        let empty = GrayImage::from_pixel(100, 80, Luma([0]));
        let out = composite(
            &img,
            &empty,
            Some(0.25),
            &spec(StudioBg::White, StudioShadow::Soft, 10),
            None,
        );
        assert_eq!(out.record.fallbacks, ["no_subject"]);
        assert_eq!(
            out.record.steps,
            ["mask", "fallback", "backdrop", "pad"],
            "the fallback path skips defringe and shadow"
        );
        // The padded original: white margins, and the photo
        // itself in the middle (its red square at center).
        assert_eq!(px(&out.image, 0, 0), [255, 255, 255, 255]);
        assert_eq!(px(&out.image, 50, 40), [RED[0], RED[1], RED[2], 255]);
        // The provider's confidence rides the record.
        assert_eq!(out.record.confidence, Some(0.25));
    }

    #[test]
    fn a_faint_mask_is_no_subject_too() {
        let (img, _) = product_photo();
        // A mask that never reaches the subject threshold
        // is degenerate: the fallback serves the original.
        let faint = GrayImage::from_pixel(100, 80, Luma([64]));
        let out = composite(
            &img,
            &faint,
            None,
            &spec(StudioBg::White, StudioShadow::None, 10),
            None,
        );
        assert_eq!(out.record.fallbacks, ["no_subject"]);
    }

    #[test]
    fn a_small_mask_is_fitted_to_the_image() {
        let (img, _) = product_photo();
        // Providers answer at their own resolution (the
        // ONNX mask model at 320x320); a 4x4 stand-in proves
        // the fit: left half background, right half subject.
        let small = GrayImage::from_fn(4, 4, |x, _| Luma([if x < 2 { 0 } else { 255 }]));
        let out = composite(
            &img,
            &small,
            None,
            &spec(StudioBg::White, StudioShadow::None, 10),
            None,
        );
        // The right half of the photo (green field and all)
        // is the subject here; the composite is opaque and
        // the left margin is pure backdrop.
        assert_eq!(px(&out.image, 0, 0), [255, 255, 255, 255]);
        assert_eq!(px(&out.image, 99, 79)[3], 255);
        assert!(out.record.fallbacks.is_empty());
    }
}
