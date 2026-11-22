//! Focal point detection: where the detail in an image is.
//!
//! [`focal_point_with`] is the entry point the cropper uses:
//! faces first when a face provider is configured (see
//! [`face_focal_point`]), then a declared focal provider
//! when one is configured, the saliency estimate of
//! [`focal_point`] otherwise. That arrangement is deliberate:
//! crop code never learns where the point came from.
//!
//! When faces fed the point, [`face_safe_start`] adjusts
//! the crop window's placement so the boxes stay whole
//! whenever the geometry allows it (issue #199).

use image::DynamicImage;

/// A point of interest, as fractions of the image width and
/// height, each in 0.0..=1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocalPoint {
    pub x: f32,
    pub y: f32,
}

/// The middle of the image: the answer for images with no
/// detectable detail, and the fallback everywhere else.
pub const CENTER: FocalPoint = FocalPoint { x: 0.5, y: 0.5 };

/// Thumbnail bound for the analysis. Saliency structure survives
/// downscaling, and the bound keeps detection proportional to a
/// constant rather than to the source size.
const MAX_DIM: u32 = 128;

/// Estimate the focal point of `img`, preferring detected
/// faces when a face provider is configured: on people photos
/// the crop should land on faces, and saliency is the fallback
/// for everything else (no provider, no faces, or a detection
/// failure, which never fails the request it enhances).
/// The composed form is kept for tests and callers that do
/// not need the trace; the engine consumes
/// [`focal_point_traced`].
pub fn focal_point_with(
    img: &DynamicImage,
    faces: Option<&dyn crate::ai::FaceProvider>,
) -> FocalPoint {
    focal_point_traced(img, faces).0
}

/// [`focal_point_with`], also returning the face boxes that
/// produced the point (empty when the point came from the
/// saliency estimate): the `debug:focal` view draws what fed
/// the decision, so the traced form is the one it consumes.
/// Detection itself happens exactly as in [`focal_point_with`]:
/// one provider call, faces first, saliency otherwise.
pub fn focal_point_traced(
    img: &DynamicImage,
    faces: Option<&dyn crate::ai::FaceProvider>,
) -> (FocalPoint, Vec<crate::ai::FaceBox>) {
    focal_point_chained(img, faces, None)
}

/// The full smart-crop focal chain (issue #166): a caller
/// `focal:` rectangle outranks everything and is consumed
/// by the engine before this runs; here, a face provider's
/// point wins when one is configured, then a declared
/// focal provider's, then the saliency estimate. Providers
/// share the face rule: a failure or an empty answer falls
/// through to the next source and never fails the request
/// the point enhances.
pub fn focal_point_chained(
    img: &DynamicImage,
    faces: Option<&dyn crate::ai::FaceProvider>,
    declared: Option<&dyn crate::ai::FocalProvider>,
) -> (FocalPoint, Vec<crate::ai::FaceBox>) {
    if let Some(provider) = faces
        && let Ok(found) = provider.detect(img)
        && let Some(point) = face_focal_point(img, &found)
    {
        return (point, found);
    }
    if let Some(provider) = declared
        && let Ok((x, y)) = provider.focal_point(img)
    {
        return (
            FocalPoint {
                x: x.clamp(0.0, 1.0),
                y: y.clamp(0.0, 1.0),
            },
            Vec::new(),
        );
    }
    (focal_point(img), Vec::new())
}

/// The area-weighted centroid of the detected faces' centers:
/// bigger faces pull harder.
fn face_focal_point(img: &DynamicImage, faces: &[crate::ai::FaceBox]) -> Option<FocalPoint> {
    if faces.is_empty() || img.width() == 0 || img.height() == 0 {
        return None;
    }
    let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
    for f in faces {
        let area = f.w as f64 * f.h as f64;
        sx += (f.x as f64 + f.w as f64 / 2.0) * area;
        sy += (f.y as f64 + f.h as f64 / 2.0) * area;
        sw += area;
    }
    Some(FocalPoint {
        x: (sx / sw / img.width() as f64) as f32,
        y: (sy / sw / img.height() as f64) as f32,
    })
}

/// One detected face projected onto the axis a crop window
/// slides along (issue #199): the face's span on that axis,
/// and its full area, the weight keeping it whole
/// contributes when window positions are scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceSpan {
    pub start: u32,
    pub end: u32,
    pub area: u64,
}

/// The faces' spans along the axis a crop window slides
/// on: a wider source crops its width, so the window
/// slides along x (`vertical == false`); a taller source
/// crops its height and the window slides along y.
/// Providers clamp their boxes to the image, so a span
/// never leaves the axis it is projected from.
pub fn face_spans(faces: &[crate::ai::FaceBox], vertical: bool) -> Vec<FaceSpan> {
    faces
        .iter()
        .map(|f| {
            let (start, len) = if vertical { (f.y, f.h) } else { (f.x, f.w) };
            FaceSpan {
                start,
                end: start.saturating_add(len),
                area: f.w as u64 * f.h as u64,
            }
        })
        .collect()
}

/// Face-safe window placement (issue #199): choose the
/// start of a `window`-long crop window sliding over an
/// `extent`-long axis so detected faces stay whole
/// whenever the geometry allows it. `preferred` is the
/// start the focal-point rule picks today; the answer
/// keeps the most face area whole and otherwise moves as
/// little as possible:
///
/// - a start keeping every face whole wins, at the point
///   of that range nearest `preferred`, so behavior
///   changes only when the focal window would cut a face;
/// - otherwise the start keeping the greatest total face
///   area whole wins (a face counts only kept whole,
///   never partially), nearest `preferred` among ties and
///   then the smallest start, so the choice is
///   deterministic;
/// - when no start keeps any face whole (every face is
///   larger than the window on this axis), the answer is
///   `preferred` itself: today's centered window.
///
/// Returns the start and how many faces it keeps whole.
pub fn face_safe_start(
    spans: &[FaceSpan],
    preferred: u32,
    window: u32,
    extent: u32,
) -> (u32, usize) {
    let max_start = extent.saturating_sub(window);
    let preferred = preferred.min(max_start);
    // The starts keeping a face whole form one interval:
    // late enough that the window's far edge reaches the
    // face's end, early enough that its near edge is not
    // past the face's start. The interval is empty when
    // the face is larger than the window on this axis.
    let intervals: Vec<(u32, u32, u64)> = spans
        .iter()
        .filter_map(|s| {
            let lo = s.end.saturating_sub(window);
            let hi = s.start.min(max_start);
            (lo <= hi).then_some((lo, hi, s.area))
        })
        .collect();
    if intervals.is_empty() {
        return (preferred, 0);
    }
    let kept_at = |start: u32| -> (u64, usize) {
        intervals
            .iter()
            .filter(|&&(lo, hi, _)| lo <= start && start <= hi)
            .fold((0u64, 0usize), |(area, n), &(_, _, a)| (area + a, n + 1))
    };
    // Kept area only changes at interval edges, so the
    // best start is an edge or `preferred` itself (the
    // point of a best plateau nearest `preferred`).
    let mut best = preferred;
    let (mut best_area, mut best_kept) = kept_at(preferred);
    for &(lo, hi, _) in &intervals {
        for cand in [lo, hi] {
            let (area, kept) = kept_at(cand);
            let better = area > best_area
                || (area == best_area
                    && (cand.abs_diff(preferred) < best.abs_diff(preferred)
                        || (cand.abs_diff(preferred) == best.abs_diff(preferred) && cand < best)));
            if better {
                best = cand;
                best_area = area;
                best_kept = kept;
            }
        }
    }
    (best, best_kept)
}

/// Estimate the focal point of `img`: the centroid of its
/// gradient energy. Detail (edges, texture) pulls the point;
/// flat regions contribute nothing. A uniform image has no
/// gradient anywhere and reports [`CENTER`].
pub fn focal_point(img: &DynamicImage) -> FocalPoint {
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return CENTER;
    }
    let thumb = if w.max(h) > MAX_DIM {
        thumbnail_fast(img, MAX_DIM, MAX_DIM)
    } else {
        img.clone()
    };
    let gray = thumb.to_luma8();
    let (tw, th) = (gray.width(), gray.height());
    if tw < 3 || th < 3 {
        return CENTER;
    }
    let mut total = 0f64;
    let mut sum_x = 0f64;
    let mut sum_y = 0f64;
    // Interior pixels only: the frame itself is not content, and
    // border gradients from resizing would pull the centroid.
    for y in 1..th - 1 {
        for x in 1..tw - 1 {
            let gx = gray.get_pixel(x + 1, y)[0] as i32 - gray.get_pixel(x - 1, y)[0] as i32;
            let gy = gray.get_pixel(x, y + 1)[0] as i32 - gray.get_pixel(x, y - 1)[0] as i32;
            let energy = (gx.abs() + gy.abs()) as f64;
            total += energy;
            sum_x += energy * x as f64;
            sum_y += energy * y as f64;
        }
    }
    if total <= 0.0 {
        return CENTER;
    }
    FocalPoint {
        x: (sum_x / total) as f32 / (tw - 1) as f32,
        y: (sum_y / total) as f32 / (th - 1) as f32,
    }
}

/// `DynamicImage::thumbnail`, byte for byte, for the two
/// layouts nearly every source decodes to (issue #655). The
/// crate's thumbnail is a block mean: every output pixel is
/// the rounded integer mean of the block of source pixels it
/// covers, read one pixel at a time. This sums whole source
/// rows into per-column sums and then sums each block's
/// columns, which adds the same integers in another order.
/// Every other layout, and a thumbnail [`thumbnail_plan`]
/// has no blocks for, takes the crate's path.
pub(crate) fn thumbnail_fast(img: &DynamicImage, nwidth: u32, nheight: u32) -> DynamicImage {
    let plan = match img {
        DynamicImage::ImageRgb8(_) | DynamicImage::ImageRgba8(_) => {
            thumbnail_plan(img.width(), img.height(), nwidth, nheight)
        }
        _ => None,
    };
    let Some(plan) = plan else {
        return img.thumbnail(nwidth, nheight);
    };
    let (w, h) = plan.size();
    let mut thumb = match img {
        DynamicImage::ImageRgb8(src) => {
            let out = thumbnail_blocks::<3>(src.as_raw(), src.width(), &plan);
            DynamicImage::ImageRgb8(image::RgbImage::from_raw(w, h, out).expect("thumbnail size"))
        }
        DynamicImage::ImageRgba8(src) => {
            let out = thumbnail_blocks::<4>(src.as_raw(), src.width(), &plan);
            DynamicImage::ImageRgba8(image::RgbaImage::from_raw(w, h, out).expect("thumbnail size"))
        }
        _ => return img.thumbnail(nwidth, nheight),
    };
    // The crate's thumbnail keeps the color space.
    let _ = thumb.set_color_space(img.color_space());
    thumb
}

/// The blocks of a block-mean thumbnail: for each output
/// column and each output row, the half-open run of source
/// columns or rows it averages.
struct ThumbnailPlan {
    columns: Vec<(u32, u32)>,
    rows: Vec<(u32, u32)>,
}

impl ThumbnailPlan {
    fn size(&self) -> (u32, u32) {
        (self.columns.len() as u32, self.rows.len() as u32)
    }
}

/// The largest block [`thumbnail_plan`] plans: 8-bit samples
/// summed over this many pixels, plus the rounding term,
/// still fit the `u32` the crate sums in.
const THUMBNAIL_BLOCK_MAX: u64 = 1 << 24;

/// The blocks the crate's thumbnail averages when it fits a
/// `width` by `height` image inside `nwidth` by `nheight`,
/// from the crate's own formulas. `None` when the crate does
/// something other than average whole blocks: a block that
/// holds no whole source pixel on an axis (it interpolates
/// there, as it does for every upscale), an empty image, or
/// a block past [`THUMBNAIL_BLOCK_MAX`].
fn thumbnail_plan(width: u32, height: u32, nwidth: u32, nheight: u32) -> Option<ThumbnailPlan> {
    if width == 0 || height == 0 {
        return None;
    }
    // The crate's `resize_dimensions`, fitting inside.
    let wratio = f64::from(nwidth) / f64::from(width);
    let hratio = f64::from(nheight) / f64::from(height);
    let ratio = f64::min(wratio, hratio);
    let new_w = ((f64::from(width) * ratio).round() as u64).max(1);
    let new_h = ((f64::from(height) * ratio).round() as u64).max(1);
    let plan = ThumbnailPlan {
        columns: thumbnail_bounds(width, u32::try_from(new_w).ok()?)?,
        rows: thumbnail_bounds(height, u32::try_from(new_h).ok()?)?,
    };
    let longest = |blocks: &[(u32, u32)]| blocks.iter().map(|&(low, high)| high - low).max();
    let block = u64::from(longest(&plan.columns)?) * u64::from(longest(&plan.rows)?);
    (block <= THUMBNAIL_BLOCK_MAX).then_some(plan)
}

/// The block bounds the crate's thumbnail computes along one
/// axis of `len` source pixels for `new_len` output pixels,
/// in its own `f32` arithmetic. `None` when a block holds no
/// whole source pixel.
fn thumbnail_bounds(len: u32, new_len: u32) -> Option<Vec<(u32, u32)>> {
    let ratio = len as f32 / new_len as f32;
    let mut bounds = Vec::new();
    for out in 0..new_len {
        let lowf = out as f32 * ratio;
        let highf = lowf + ratio;
        let low = (lowf.ceil() as u32).clamp(0, len - 1);
        let high = (highf.ceil() as u32).clamp(low, len);
        if low == high {
            return None;
        }
        bounds.push((low, high));
    }
    Some(bounds)
}

/// The block means of a raw 8-bit buffer of `CH` channels,
/// `width` pixels wide, over the blocks of `plan`.
fn thumbnail_blocks<const CH: usize>(raw: &[u8], width: u32, plan: &ThumbnailPlan) -> Vec<u8> {
    let stride = width as usize * CH;
    let mut sums = vec![0u32; stride];
    let mut out = vec![0u8; plan.columns.len() * plan.rows.len() * CH];
    let out_rows = out.chunks_exact_mut(plan.columns.len() * CH);
    for (&(top, bottom), out_row) in plan.rows.iter().zip(out_rows) {
        sums.fill(0);
        for line in raw[top as usize * stride..bottom as usize * stride].chunks_exact(stride) {
            for (sum, &sample) in sums.iter_mut().zip(line) {
                *sum += u32::from(sample);
            }
        }
        let pixels = out_row.as_chunks_mut::<CH>().0;
        for (&(left, right), pixel) in plan.columns.iter().zip(pixels) {
            let n = (right - left) * (bottom - top);
            let mut block = [n / 2; CH];
            for column in sums[left as usize * CH..right as usize * CH]
                .as_chunks::<CH>()
                .0
            {
                for (total, &sum) in block.iter_mut().zip(column) {
                    *total += sum;
                }
            }
            for (sample, total) in pixel.iter_mut().zip(block) {
                *sample = (total / n).min(255) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    /// 100x100 mid-gray with a high-contrast checker block at
    /// `x0..x0+20, y0..y0+20`.
    fn block_at(x0: u32, y0: u32) -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(100, 100);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let in_block = (x0..x0 + 20).contains(&x) && (y0..y0 + 20).contains(&y);
            *p = if in_block {
                if (x + y) % 2 == 0 {
                    Rgb([200, 200, 200])
                } else {
                    Rgb([50, 50, 50])
                }
            } else {
                Rgb([128, 128, 128])
            };
        }
        img
    }

    #[test]
    fn focal_lands_near_an_off_center_block() {
        // Block center is at (0.80, 0.50) of the image.
        let f = focal_point(&block_at(70, 40));
        assert!(f.x > 0.65 && f.x < 0.95, "x {}", f.x);
        assert!(f.y > 0.35 && f.y < 0.65, "y {}", f.y);
    }

    #[test]
    fn focal_follows_the_block_to_a_corner() {
        // Block center is at (0.10, 0.10).
        let f = focal_point(&block_at(0, 0));
        assert!(f.x < 0.35, "x {}", f.x);
        assert!(f.y < 0.35, "y {}", f.y);
    }

    #[test]
    fn uniform_image_is_center() {
        let img = DynamicImage::new_rgb8(50, 50);
        assert_eq!(focal_point(&img), CENTER);
    }

    /// Hard edges and noise, different in every channel.
    fn busy(x: u32, y: u32, c: u32) -> u8 {
        let mut h = x
            .wrapping_mul(0x9E37_79B1)
            .wrapping_add(y.wrapping_mul(0x85EB_CA77))
            .wrapping_add(c.wrapping_mul(0xC2B2_AE3D));
        h ^= h >> 15;
        h = h.wrapping_mul(0x2C1B_3C6D);
        match (x / 3 + y / 2 + c) % 4 {
            0 => 0,
            1 => 255,
            _ => (h >> 12) as u8,
        }
    }

    fn busy_rgb(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(w, h, |x, y| {
            Rgb([busy(x, y, 0), busy(x, y, 1), busy(x, y, 2)])
        }))
    }

    fn busy_rgba(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgba8(image::ImageBuffer::from_fn(w, h, |x, y| {
            image::Rgba([busy(x, y, 0), busy(x, y, 1), busy(x, y, 2), busy(y, x, 3)])
        }))
    }

    /// The layout, the size, the color space and every byte.
    fn whole(img: &DynamicImage) -> (image::ColorType, (u32, u32), image::metadata::Cicp, Vec<u8>) {
        (
            img.color(),
            (img.width(), img.height()),
            img.color_space(),
            img.as_bytes().to_vec(),
        )
    }

    #[test]
    fn thumbnail_fast_matches_the_crate_thumbnail() {
        // Issue #655: the saliency thumbnail is summed by
        // rows. Against the crate on both layouts it takes:
        // one pixel on a side, sizes just over the bound,
        // strips, the longest side a source may have, and
        // the shape of a typical photo; targets that shrink
        // to the analysis size and to a pixel, that leave an
        // axis with blocks of one, and that would enlarge,
        // where the crate interpolates and so does this.
        let small = [
            (128, 128),
            (100, 100),
            (16, 16),
            (1, 1),
            (128, 7),
            (160, 160),
        ];
        let shrink = [(128, 128), (1, 1)];
        let mut cases = 0;
        for (w, h, targets) in [
            (1, 1, &small[..]),
            (3, 2, &small),
            (1, 129, &small),
            (129, 1, &small),
            (129, 129, &small),
            (130, 97, &small),
            (255, 256, &small),
            (257, 255, &small),
            (1000, 3, &small),
            (3, 1000, &small),
            (320, 213, &shrink),
            (16383, 9, &shrink),
            (9, 16383, &shrink),
            (4000, 37, &shrink),
            (1200, 800, &shrink[..1]),
        ] {
            for img in [busy_rgb(w, h), busy_rgba(w, h)] {
                for &(nw, nh) in targets {
                    assert_eq!(
                        whole(&img.thumbnail(nw, nh)),
                        whole(&thumbnail_fast(&img, nw, nh)),
                        "{:?} {w}x{h} to fit {nw}x{nh}",
                        img.color()
                    );
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 2 * (10 * 6 + 4 * 2 + 1));
        // The color space travels with the thumbnail, as it
        // does through the crate's.
        let mut p3 = busy_rgb(257, 255);
        p3.set_color_space(image::metadata::Cicp::DISPLAY_P3)
            .unwrap();
        let ours = thumbnail_fast(&p3, 128, 128);
        assert_eq!(ours.color_space(), image::metadata::Cicp::DISPLAY_P3);
        assert_eq!(whole(&p3.thumbnail(128, 128)), whole(&ours));
    }

    #[test]
    fn thumbnail_fast_leaves_the_rest_to_the_crate() {
        // Issue #655: only RGB8 and RGBA8 block means are
        // summed here. Every other layout comes back as the
        // crate makes it.
        let rgba = busy_rgba(255, 256);
        for img in [
            DynamicImage::ImageLuma8(rgba.to_luma8()),
            DynamicImage::ImageLumaA8(rgba.to_luma_alpha8()),
            DynamicImage::ImageRgb16(rgba.to_rgb16()),
            DynamicImage::ImageRgba16(rgba.to_rgba16()),
            DynamicImage::ImageLuma16(rgba.to_luma16()),
            DynamicImage::ImageLumaA16(rgba.to_luma_alpha16()),
            DynamicImage::ImageRgb32F(rgba.to_rgb32f()),
            DynamicImage::ImageRgba32F(rgba.to_rgba32f()),
        ] {
            assert_eq!(
                whole(&img.thumbnail(128, 128)),
                whole(&thumbnail_fast(&img, 128, 128)),
                "{:?}",
                img.color()
            );
        }
        // No plan where the crate does not average whole
        // blocks: an enlargement, whose blocks run empty, an
        // empty image, and a block whose sum could pass the
        // `u32` the crate adds in (4096 squared is the last
        // that cannot).
        assert!(thumbnail_plan(100, 100, 128, 128).is_none());
        assert!(thumbnail_plan(0, 5, 128, 128).is_none());
        assert!(thumbnail_plan(4097, 4096, 1, 1).is_none());
        let plan = thumbnail_plan(4096, 4096, 1, 1).expect("a block of 2^24 pixels");
        assert_eq!(
            (plan.columns, plan.rows),
            (vec![(0, 4096)], vec![(0, 4096)])
        );
        // A block is never empty and never leaves the image.
        let plan = thumbnail_plan(2400, 1600, 128, 128).expect("a photo shrinks in blocks");
        assert_eq!(plan.size(), (128, 85));
        for (blocks, len) in [(&plan.columns, 2400), (&plan.rows, 1600)] {
            assert!(blocks.iter().all(|&(low, high)| low < high && high <= len));
        }
    }

    /// A provider that reports one face at a fixed place.
    #[derive(Debug)]
    struct FixedFace;

    impl crate::ai::FaceProvider for FixedFace {
        fn detect(
            &self,
            _img: &DynamicImage,
        ) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
            Ok(vec![crate::ai::FaceBox {
                x: 0,
                y: 0,
                w: 20,
                h: 20,
                score: 0.99,
            }])
        }
    }

    /// A provider that finds nothing.
    #[derive(Debug)]
    struct NoFaces;

    impl crate::ai::FaceProvider for NoFaces {
        fn detect(
            &self,
            _img: &DynamicImage,
        ) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn a_detected_face_wins_over_saliency() {
        // Saliency detail sits at the right (block center 0.80);
        // the face sits at the left (center 0.10). Faces win.
        let img = block_at(70, 40);
        let p = focal_point_with(&img, Some(&FixedFace));
        assert!(p.x < 0.2 && p.y < 0.2, "got {p:?}");
    }

    #[test]
    fn traced_reports_the_faces_that_fed_the_point() {
        let img = block_at(70, 40);
        let (point, boxes) = focal_point_traced(&img, Some(&FixedFace));
        assert_eq!(point, focal_point_with(&img, Some(&FixedFace)));
        assert_eq!(boxes.len(), 1);
        assert_eq!(
            (boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h),
            (0, 0, 20, 20)
        );
        // Saliency-fed points trace no boxes.
        let (point, boxes) = focal_point_traced(&img, Some(&NoFaces));
        assert_eq!(point, focal_point(&img));
        assert!(boxes.is_empty());
    }

    #[test]
    fn no_faces_falls_back_to_saliency() {
        let img = block_at(70, 40);
        assert_eq!(focal_point_with(&img, Some(&NoFaces)), focal_point(&img));
        assert_eq!(focal_point_with(&img, None), focal_point(&img));
    }

    /// A declared focal provider with a fixed answer at
    /// the right edge, against the saliency block's left.
    #[derive(Debug)]
    struct FixedFocal;

    impl crate::ai::FocalProvider for FixedFocal {
        fn focal_point(&self, _img: &DynamicImage) -> Result<(f32, f32), crate::ai::AiError> {
            Ok((0.9, 0.5))
        }
    }

    /// A declared focal provider that always fails.
    #[derive(Debug)]
    struct FailingFocal;

    impl crate::ai::FocalProvider for FailingFocal {
        fn focal_point(&self, _img: &DynamicImage) -> Result<(f32, f32), crate::ai::AiError> {
            Err(crate::ai::AiError::Inference(
                "heatmap exploded".to_string(),
            ))
        }
    }

    #[test]
    fn a_declared_focal_point_wins_over_saliency() {
        // Saliency detail sits at the left (block center
        // 0.10); the declared provider says 0.90. It wins.
        let img = block_at(0, 40);
        let (point, boxes) = focal_point_chained(&img, Some(&NoFaces), Some(&FixedFocal));
        assert_eq!(point, FocalPoint { x: 0.9, y: 0.5 });
        assert!(boxes.is_empty());
    }

    #[test]
    fn a_face_wins_over_a_declared_focal_point() {
        // Face center 0.10 against the declared 0.90: faces
        // outrank declared providers in the chain.
        let img = block_at(70, 40);
        let (point, boxes) = focal_point_chained(&img, Some(&FixedFace), Some(&FixedFocal));
        assert!(point.x < 0.2 && point.y < 0.2, "got {point:?}");
        assert_eq!(boxes.len(), 1);
    }

    #[test]
    fn a_failing_declared_focal_falls_back_to_saliency() {
        let img = block_at(70, 40);
        let (point, boxes) = focal_point_chained(&img, None, Some(&FailingFocal));
        assert_eq!(point, focal_point(&img));
        assert!(boxes.is_empty());
    }

    fn span(start: u32, end: u32, area: u64) -> FaceSpan {
        FaceSpan { start, end, area }
    }

    #[test]
    fn face_spans_project_onto_the_sliding_axis() {
        let faces = vec![crate::ai::FaceBox {
            x: 30,
            y: 40,
            w: 50,
            h: 60,
            score: 0.9,
        }];
        assert_eq!(
            face_spans(&faces, false),
            vec![FaceSpan {
                start: 30,
                end: 80,
                area: 3000
            }]
        );
        assert_eq!(
            face_spans(&faces, true),
            vec![FaceSpan {
                start: 40,
                end: 100,
                area: 3000
            }]
        );
    }

    #[test]
    fn face_safe_start_keeps_every_face_when_a_position_exists() {
        // Window 200 over extent 400. Only start 30 keeps
        // both spans whole; the preferred start 0 keeps
        // just the first.
        let spans = vec![span(30, 70, 1600), span(210, 230, 400)];
        assert_eq!(face_safe_start(&spans, 0, 200, 400), (30, 2));
        // A preferred start that already keeps both is
        // returned untouched: behavior changes only when
        // it must.
        assert_eq!(face_safe_start(&spans, 30, 200, 400), (30, 2));
    }

    #[test]
    fn face_safe_start_prefers_the_larger_face() {
        // Both spans cannot fit together (0..400 of spans
        // in a 200 window). The 3600-area face wins over
        // the 1600-area one, even though the preferred
        // start 38 keeps neither.
        let spans = vec![span(0, 60, 3600), span(360, 400, 1600)];
        assert_eq!(face_safe_start(&spans, 38, 200, 400), (0, 1));
    }

    #[test]
    fn face_safe_start_tie_breaks_are_deterministic() {
        // Equal areas, both intervals single points (0 and
        // 200... the second face's interval is 40..=200,
        // entered at 40): the start nearest the preferred
        // one wins; at equal distance, the smaller start.
        let spans = vec![span(0, 40, 1600), span(200, 240, 1600)];
        assert_eq!(face_safe_start(&spans, 25, 200, 400), (40, 1));
        assert_eq!(face_safe_start(&spans, 20, 200, 400), (0, 1));
    }

    #[test]
    fn face_safe_start_centers_an_oversized_face_via_preferred() {
        // The span (120 long) exceeds the window (100) on
        // this axis: no start keeps it whole, so the focal
        // start stands and nothing is counted kept.
        let spans = vec![span(40, 160, 14400)];
        assert_eq!(face_safe_start(&spans, 50, 100, 200), (50, 0));
    }

    #[test]
    fn face_safe_start_without_faces_is_the_preferred_start() {
        assert_eq!(face_safe_start(&[], 42, 100, 400), (42, 0));
        // A preferred start past the axis clamps into range.
        assert_eq!(face_safe_start(&[], 999, 100, 400), (300, 0));
    }

    #[test]
    fn face_safe_start_handles_a_window_sized_face() {
        // The span is exactly the window: one start keeps
        // it, and the placement moves there.
        let spans = vec![span(50, 250, 5000)];
        assert_eq!(face_safe_start(&spans, 0, 200, 400), (50, 1));
    }
}
