//! Cropping images to a target aspect ratio.
//!
//! [`crop_to_aspect`] is a pure function: source image plus target
//! dimensions plus [`Align`]ment in, cropped image out. It keeps as many
//! source pixels as possible and never crops outside the image.

use image::DynamicImage;

use crate::url::Align;

/// Crop `img` to the `target_w:target_h` aspect ratio, keeping the
/// `align` portion.
///
/// Uses the full source height (or width); returns `img` unchanged when
/// the ratios already match or the target dimensions are zero.
///
/// The engine calls [`crop_to_aspect_with`] (it may have a face
/// provider); this composed form is kept for tests and future
/// library use.
#[allow(dead_code)]
pub fn crop_to_aspect(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    align: Align,
) -> DynamicImage {
    crop_to_aspect_with(img, target_w, target_h, align, None, None)
}

/// What a crop decided, reported by the `*_traced` crop
/// entry points for the `debug:focal` view: the window the
/// crop cut, and the focal point it positioned on (with the
/// face boxes that produced that point) when the alignment
/// has one. Directional alignments position by rule and
/// report no focal point.
#[derive(Debug, Clone, PartialEq)]
pub struct CropTrace {
    /// The cut window: (left, top, width, height) in source
    /// pixels. A crop whose aspect already matches reports
    /// the full frame.
    pub window: (u32, u32, u32, u32),
    /// The focal point the window was positioned on, as
    /// fractions of the image dimensions.
    pub focal: Option<crate::focus::FocalPoint>,
    /// The detected faces that produced the focal point
    /// (smart alignment with a face provider); empty
    /// otherwise.
    pub faces: Vec<crate::ai::FaceBox>,
    /// How many faces the decision detected: the length
    /// of `faces`, carried as a count so the record
    /// states the guarantee's denominator directly.
    pub faces_detected: usize,
    /// How many of the detected faces the final window
    /// keeps fully inside (issue #199): smart placement
    /// keeps faces whole whenever the geometry allows
    /// it, and this count is that promise made
    /// inspectable. Equals `faces_detected` whenever
    /// every face could be kept.
    pub faces_kept: usize,
}

/// [`crop_to_aspect`] with an optional face provider: when one
/// is configured, smart alignment prefers detected faces over
/// the saliency estimate. A declared focal provider (issue
/// #166) slots between them in the same chain.
pub fn crop_to_aspect_with(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    align: Align,
    faces: Option<&dyn crate::ai::FaceProvider>,
    declared: Option<&dyn crate::ai::FocalProvider>,
) -> DynamicImage {
    crop_to_aspect_impl(img, target_w, target_h, align, faces, declared, false).0
}

/// [`crop_to_aspect_with`], also reporting what the crop
/// decided (see [`CropTrace`]). The window is positioned on
/// the very focal value the trace reports: the trace is a
/// by-product of the real decision, never a recomputation.
/// The trace is `None` only when no crop ran at all (a zero
/// target dimension).
pub fn crop_to_aspect_traced(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    align: Align,
    faces: Option<&dyn crate::ai::FaceProvider>,
    declared: Option<&dyn crate::ai::FocalProvider>,
) -> (DynamicImage, Option<CropTrace>) {
    crop_to_aspect_impl(img, target_w, target_h, align, faces, declared, true)
}

#[allow(clippy::too_many_arguments)]
fn crop_to_aspect_impl(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    align: Align,
    faces: Option<&dyn crate::ai::FaceProvider>,
    declared: Option<&dyn crate::ai::FocalProvider>,
    trace: bool,
) -> (DynamicImage, Option<CropTrace>) {
    if target_w == 0 || target_h == 0 {
        return (img, None);
    }
    let (w, h) = (img.width(), img.height());
    // Compare w/h with target_w/target_h via cross-multiplication.
    let src_ratio = w as u64 * target_h as u64;
    let dst_ratio = target_w as u64 * h as u64;
    if src_ratio == dst_ratio {
        // No crop runs, so the plain path decides nothing
        // (and runs no detection, exactly as before). The
        // traced path still reports the focal analysis: a
        // full-frame window positioned on the point.
        let traced = trace.then(|| focal_for(align, &img, faces, declared));
        return (
            img,
            traced.map(|(point, face_boxes)| CropTrace {
                faces_detected: face_boxes.len(),
                faces_kept: faces_inside(&face_boxes, (0, 0, w, h)),
                window: (0, 0, w, h),
                focal: point,
                faces: face_boxes,
            }),
        );
    }
    let (mut focal, mut face_boxes) = (None, Vec::new());
    let (x, y, cw, ch) = if src_ratio > dst_ratio {
        // Source is wider: crop the width.
        let cw = ((h as u64 * target_w as u64) / target_h as u64).max(1) as u32;
        let x = match align {
            Align::Left | Align::TopLeft | Align::BottomLeft => 0,
            Align::Right | Align::TopRight | Align::BottomRight => w.saturating_sub(cw),
            Align::Smart => {
                let (point, boxes) = crate::focus::focal_point_chained(&img, faces, declared);
                focal = Some(point);
                face_boxes = boxes;
                smart_start(&face_boxes, false, point_start(point.x, cw, w), cw, w)
            }
            _ => (w.saturating_sub(cw)) / 2,
        };
        (x, 0, cw, h)
    } else {
        // Source is taller: crop the height.
        let ch = ((w as u64 * target_h as u64) / target_w as u64).max(1) as u32;
        let y = match align {
            Align::Top | Align::TopLeft | Align::TopRight => 0,
            Align::Bottom | Align::BottomLeft | Align::BottomRight => h.saturating_sub(ch),
            Align::Smart => {
                let (point, boxes) = crate::focus::focal_point_chained(&img, faces, declared);
                focal = Some(point);
                face_boxes = boxes;
                smart_start(&face_boxes, true, point_start(point.y, ch, h), ch, h)
            }
            _ => (h.saturating_sub(ch)) / 2,
        };
        (0, y, w, ch)
    };
    let faces_detected = face_boxes.len();
    let faces_kept = faces_inside(&face_boxes, (x, y, cw, ch));
    let cropped = image::imageops::crop_imm(&img, x, y, cw, ch).to_image();
    (
        DynamicImage::ImageRgba8(cropped),
        trace.then_some(CropTrace {
            window: (x, y, cw, ch),
            focal: if align == Align::Center {
                Some(crate::focus::CENTER)
            } else {
                focal
            },
            faces: face_boxes,
            faces_detected,
            faces_kept,
        }),
    )
}

/// The window start for smart alignment: the focal-point
/// start, adjusted by face-safe placement (issue #199)
/// when faces fed the decision. With no faces the focal
/// start stands, exactly as before face-safety.
fn smart_start(
    faces: &[crate::ai::FaceBox],
    vertical: bool,
    preferred: u32,
    window: u32,
    extent: u32,
) -> u32 {
    if faces.is_empty() {
        return preferred;
    }
    let spans = crate::focus::face_spans(faces, vertical);
    crate::focus::face_safe_start(&spans, preferred, window, extent).0
}

/// How many of `faces` lie fully inside `window`, a
/// (left, top, width, height) rectangle in source pixels:
/// the whole box, edges included, on both axes.
fn faces_inside(faces: &[crate::ai::FaceBox], window: (u32, u32, u32, u32)) -> usize {
    let (x, y) = (window.0 as u64, window.1 as u64);
    let (right, bottom) = (x + window.2 as u64, y + window.3 as u64);
    faces
        .iter()
        .filter(|f| {
            let (fx, fy) = (f.x as u64, f.y as u64);
            fx >= x && fy >= y && fx + f.w as u64 <= right && fy + f.h as u64 <= bottom
        })
        .count()
}

/// The focal point an alignment positions on, for the
/// aspect-matched trace: `Smart` runs the face-plus-saliency
/// estimate, `Center` is the image center by definition, and
/// directional alignments have no point.
fn focal_for(
    align: Align,
    img: &DynamicImage,
    faces: Option<&dyn crate::ai::FaceProvider>,
    declared: Option<&dyn crate::ai::FocalProvider>,
) -> (Option<crate::focus::FocalPoint>, Vec<crate::ai::FaceBox>) {
    match align {
        Align::Smart => {
            let (point, boxes) = crate::focus::focal_point_chained(img, faces, declared);
            (Some(point), boxes)
        }
        Align::Center => (Some(crate::focus::CENTER), Vec::new()),
        _ => (None, Vec::new()),
    }
}

/// Window start along one axis for a window positioned on a
/// focal coordinate: center the `window` span on the point,
/// clamped so the window stays inside the image.
fn point_start(frac: f32, window: u32, extent: u32) -> u32 {
    let center = (frac * extent as f32).round() as i32;
    let max_start = extent.saturating_sub(window) as i32;
    (center - window as i32 / 2).clamp(0, max_start) as u32
}

/// Crop `img` to the `target_w:target_h` aspect ratio, positioned
/// on the caller-supplied focal rectangle `(left, top, right,
/// bottom)` (a `focal:` segment): the window is centered on the
/// rectangle's center, then clamped so a rectangle that fits
/// inside the window stays fully inside it. Like
/// [`crop_to_aspect`], it keeps as many source pixels as possible
/// and returns `img` unchanged when the ratios already match or
/// the target dimensions are zero.
pub fn crop_to_aspect_focal(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    rect: (u32, u32, u32, u32),
) -> DynamicImage {
    crop_to_aspect_focal_traced(img, target_w, target_h, rect).0
}

/// [`crop_to_aspect_focal`], also reporting the window it
/// cut, as (left, top, width, height): `None` when no crop
/// ran (a zero target), the full frame when the aspect
/// already matched.
pub fn crop_to_aspect_focal_traced(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    rect: (u32, u32, u32, u32),
) -> (DynamicImage, Option<(u32, u32, u32, u32)>) {
    if target_w == 0 || target_h == 0 {
        return (img, None);
    }
    let (w, h) = (img.width(), img.height());
    // Compare w/h with target_w/target_h via cross-multiplication.
    let src_ratio = w as u64 * target_h as u64;
    let dst_ratio = target_w as u64 * h as u64;
    if src_ratio == dst_ratio {
        return (img, Some((0, 0, w, h)));
    }
    let (left, top, right, bottom) = rect;
    let (x, y, cw, ch) = if src_ratio > dst_ratio {
        // Source is wider: crop the width.
        let cw = ((h as u64 * target_w as u64) / target_h as u64).max(1) as u32;
        (focal_start(left, right, cw, w), 0, cw, h)
    } else {
        // Source is taller: crop the height.
        let ch = ((w as u64 * target_h as u64) / target_w as u64).max(1) as u32;
        (0, focal_start(top, bottom, ch, h), w, ch)
    };
    let cropped = image::imageops::crop_imm(&img, x, y, cw, ch).to_image();
    (DynamicImage::ImageRgba8(cropped), Some((x, y, cw, ch)))
}

/// Crop `img` to the `target_w:target_h` aspect around `face`:
/// the smallest region with that aspect which contains the face
/// expanded by `padding` (a fraction of the face's size, added
/// on every side), centered on the face's center and clamped
/// into the image. When the padded region cannot fit (the
/// image is too small on one axis) it shrinks to fit and the
/// face fills more of the frame. The engine resizes the result
/// to the exact target, like every other crop.
pub fn crop_around_face(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    face: &crate::ai::FaceBox,
    padding: f32,
) -> DynamicImage {
    crop_around_face_traced(img, target_w, target_h, face, padding).0
}

/// [`crop_around_face`], also reporting the window it cut,
/// as (left, top, width, height): `None` when no crop ran
/// (a zero target or an empty image).
pub fn crop_around_face_traced(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    face: &crate::ai::FaceBox,
    padding: f32,
) -> (DynamicImage, Option<(u32, u32, u32, u32)>) {
    if target_w == 0 || target_h == 0 || img.width() == 0 || img.height() == 0 {
        return (img, None);
    }
    let (w, h) = (img.width() as f64, img.height() as f64);
    let (tw, th) = (target_w as f64, target_h as f64);
    let pad = padding.max(0.0) as f64;
    let pw = face.w as f64 * (1.0 + 2.0 * pad);
    let ph = face.h as f64 * (1.0 + 2.0 * pad);
    // The smallest target-aspect region holding the padded face.
    let scale = (pw / tw).max(ph / th);
    let (mut rw, mut rh) = (tw * scale, th * scale);
    // Shrink to fit the image, keeping the aspect.
    if rw > w {
        rw = w;
        rh = rw * th / tw;
    }
    if rh > h {
        rh = h;
        rw = rh * tw / th;
    }
    let cx = face.x as f64 + face.w as f64 / 2.0;
    let cy = face.y as f64 + face.h as f64 / 2.0;
    let ox = (cx - rw / 2.0).clamp(0.0, (w - rw).max(0.0));
    let oy = (cy - rh / 2.0).clamp(0.0, (h - rh).max(0.0));
    let cw = (rw.round() as u32).clamp(1, img.width());
    let ch = (rh.round() as u32).clamp(1, img.height());
    let x = (ox.round() as u32).min(img.width() - cw);
    let y = (oy.round() as u32).min(img.height() - ch);
    let cropped = image::imageops::crop_imm(&img, x, y, cw, ch).to_image();
    (DynamicImage::ImageRgba8(cropped), Some((x, y, cw, ch)))
}

/// Window start along one axis for a focal rectangle spanning
/// `rect_start..rect_end`: center the `window` span on the
/// rectangle's center, clamped so a rectangle that fits inside
/// the window stays fully inside it, and so the window stays
/// inside the image either way.
fn focal_start(rect_start: u32, rect_end: u32, window: u32, extent: u32) -> u32 {
    let max_start = extent.saturating_sub(window) as i64;
    let preferred = (rect_start as i64 + rect_end as i64) / 2 - window as i64 / 2;
    let start = if rect_end - rect_start <= window {
        // The rectangle fits: keep every pixel of it.
        let lo = (rect_end as i64 - window as i64).max(0);
        let hi = (rect_start as i64).min(max_start);
        preferred.clamp(lo, hi)
    } else {
        preferred.clamp(0, max_start)
    };
    start as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, Rgb};

    /// 100x60 with vertical stripes: red | green | blue.
    fn vstripes() -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(100, 60);
        for (x, _, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if x < 20 {
                Rgb([255, 0, 0])
            } else if x < 80 {
                Rgb([0, 255, 0])
            } else {
                Rgb([0, 0, 255])
            };
        }
        img
    }

    /// 60x100 with horizontal stripes: red / green / blue.
    fn hstripes() -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(60, 100);
        for (_, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if y < 20 {
                Rgb([255, 0, 0])
            } else if y < 80 {
                Rgb([0, 255, 0])
            } else {
                Rgb([0, 0, 255])
            };
        }
        img
    }

    fn rgb(img: &DynamicImage, x: u32, y: u32) -> [u8; 3] {
        let p = img.get_pixel(x, y);
        [p[0], p[1], p[2]]
    }

    #[test]
    fn crops_wider_source_to_square_centered() {
        let out = crop_to_aspect(vstripes(), 1, 1, Align::Center);
        assert_eq!((out.width(), out.height()), (60, 60));
        // x offset 20: keeps the green middle.
        assert_eq!(rgb(&out, 0, 0), [0, 255, 0]);
        assert_eq!(rgb(&out, 59, 59), [0, 255, 0]);
    }

    #[test]
    fn crops_taller_source_to_square_centered() {
        let out = crop_to_aspect(hstripes(), 1, 1, Align::Center);
        assert_eq!((out.width(), out.height()), (60, 60));
        assert_eq!(rgb(&out, 0, 0), [0, 255, 0]);
        assert_eq!(rgb(&out, 59, 59), [0, 255, 0]);
    }

    #[test]
    fn alignment_left_keeps_left() {
        let out = crop_to_aspect(vstripes(), 1, 1, Align::Left);
        assert_eq!((out.width(), out.height()), (60, 60));
        assert_eq!(rgb(&out, 0, 0), [255, 0, 0]);
    }

    #[test]
    fn alignment_right_keeps_right() {
        let out = crop_to_aspect(vstripes(), 1, 1, Align::Right);
        assert_eq!((out.width(), out.height()), (60, 60));
        assert_eq!(rgb(&out, 59, 0), [0, 0, 255]);
    }

    #[test]
    fn alignment_top_keeps_top() {
        let out = crop_to_aspect(hstripes(), 1, 1, Align::Top);
        assert_eq!((out.width(), out.height()), (60, 60));
        assert_eq!(rgb(&out, 0, 0), [255, 0, 0]);
    }

    #[test]
    fn alignment_bottom_keeps_bottom() {
        let out = crop_to_aspect(hstripes(), 1, 1, Align::Bottom);
        assert_eq!((out.width(), out.height()), (60, 60));
        assert_eq!(rgb(&out, 0, 59), [0, 0, 255]);
    }

    #[test]
    fn corner_top_left_matches_left_on_a_wider_source() {
        // The width is cropped, so the corner's horizontal
        // component rules: top-left is exactly left.
        let corner = crop_to_aspect(vstripes(), 1, 1, Align::TopLeft);
        let plain = crop_to_aspect(vstripes(), 1, 1, Align::Left);
        assert_eq!(corner.to_rgb8(), plain.to_rgb8());
        assert_eq!(rgb(&corner, 0, 0), [255, 0, 0]);
    }

    #[test]
    fn corner_top_left_matches_top_on_a_taller_source() {
        // The height is cropped, so the vertical component
        // rules: top-left is exactly top.
        let corner = crop_to_aspect(hstripes(), 1, 1, Align::TopLeft);
        let plain = crop_to_aspect(hstripes(), 1, 1, Align::Top);
        assert_eq!(corner.to_rgb8(), plain.to_rgb8());
        assert_eq!(rgb(&corner, 0, 0), [255, 0, 0]);
    }

    #[test]
    fn corner_bottom_right_matches_right_on_a_wider_source() {
        let corner = crop_to_aspect(vstripes(), 1, 1, Align::BottomRight);
        let plain = crop_to_aspect(vstripes(), 1, 1, Align::Right);
        assert_eq!(corner.to_rgb8(), plain.to_rgb8());
        assert_eq!(rgb(&corner, 59, 0), [0, 0, 255]);
    }

    #[test]
    fn corner_bottom_right_matches_bottom_on_a_taller_source() {
        let corner = crop_to_aspect(hstripes(), 1, 1, Align::BottomRight);
        let plain = crop_to_aspect(hstripes(), 1, 1, Align::Bottom);
        assert_eq!(corner.to_rgb8(), plain.to_rgb8());
        assert_eq!(rgb(&corner, 0, 59), [0, 0, 255]);
    }

    #[test]
    fn no_op_when_ratios_match() {
        let out = crop_to_aspect(vstripes(), 5, 3, Align::Center);
        assert_eq!((out.width(), out.height()), (100, 60));
        assert_eq!(rgb(&out, 0, 0), [255, 0, 0]);
        assert_eq!(rgb(&out, 99, 0), [0, 0, 255]);
    }

    /// 200x100 mid-gray, with a noisy red block at the far right
    /// (x 170..196, y 30..70).
    fn right_block() -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(200, 100);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if (170..196).contains(&x) && (30..70).contains(&y) {
                if (x + y) % 2 == 0 {
                    Rgb([255, 0, 0])
                } else {
                    Rgb([128, 0, 0])
                }
            } else {
                Rgb([128, 128, 128])
            };
        }
        img
    }

    fn has_pure_red(img: &DynamicImage) -> bool {
        img.to_rgb8()
            .pixels()
            .any(|p| p[0] == 255 && p[1] == 0 && p[2] == 0)
    }

    #[test]
    fn smart_crop_keeps_the_off_center_subject() {
        // A square crop of the 200x100 source is 100 wide: centered
        // it spans x 50..150 and misses the block entirely, while
        // smart alignment centers the window on the block.
        let centered = crop_to_aspect(right_block(), 1, 1, Align::Center);
        assert!(!has_pure_red(&centered));
        let smart = crop_to_aspect(right_block(), 1, 1, Align::Smart);
        assert_eq!((smart.width(), smart.height()), (100, 100));
        assert!(has_pure_red(&smart));
    }

    #[test]
    fn traced_crop_reports_the_window_it_cut() {
        // The trace is a by-product of the real crop: same
        // pixels out, and the window matches the crop.
        let (out, trace) = crop_to_aspect_traced(vstripes(), 1, 1, Align::Center, None, None);
        let plain = crop_to_aspect(vstripes(), 1, 1, Align::Center);
        assert_eq!(out.to_rgb8(), plain.to_rgb8());
        let trace = trace.unwrap();
        assert_eq!(trace.window, (20, 0, 60, 60));
        assert_eq!(trace.focal, Some(crate::focus::CENTER));
        assert!(trace.faces.is_empty());
        // A directional alignment reports no focal point.
        let (_, trace) = crop_to_aspect_traced(vstripes(), 1, 1, Align::Left, None, None);
        let trace = trace.unwrap();
        assert_eq!(trace.window, (0, 0, 60, 60));
        assert_eq!(trace.focal, None);
        // A zero target runs no crop and traces nothing.
        let (_, trace) = crop_to_aspect_traced(vstripes(), 0, 0, Align::Center, None, None);
        assert!(trace.is_none());
    }

    #[test]
    fn traced_smart_crop_window_covers_the_reported_point() {
        let (_, trace) = crop_to_aspect_traced(right_block(), 1, 1, Align::Smart, None, None);
        let trace = trace.unwrap();
        let point = trace.focal.unwrap();
        let px = (point.x * 200.0).round() as u32;
        let (x, _, cw, _) = trace.window;
        assert!(
            x <= px && px < x + cw,
            "window {x}..{} vs point {px}",
            x + cw
        );
    }

    /// A declared focal provider pinned to the left edge
    /// (issue #166): against the right block's saliency, the
    /// declared point steers both the window and the trace.
    #[derive(Debug)]
    struct LeftFocal;

    impl crate::ai::FocalProvider for LeftFocal {
        fn focal_point(&self, _img: &DynamicImage) -> Result<(f32, f32), crate::ai::AiError> {
            Ok((0.05, 0.5))
        }
    }

    #[test]
    fn traced_smart_crop_reports_the_declared_focal_point() {
        let (_, trace) =
            crop_to_aspect_traced(right_block(), 1, 1, Align::Smart, None, Some(&LeftFocal));
        let trace = trace.unwrap();
        assert_eq!(
            trace.focal,
            Some(crate::focus::FocalPoint { x: 0.05, y: 0.5 })
        );
        assert_eq!(trace.window.0, 0);
        assert!(trace.faces.is_empty());
    }

    #[test]
    fn traced_focal_and_face_crops_report_their_windows() {
        let (_, window) = crop_to_aspect_focal_traced(vstripes(), 1, 1, (80, 10, 100, 50));
        assert_eq!(window, Some((40, 0, 60, 60)));
        let (_, window) = crop_around_face_traced(face_canvas(), 1, 1, &face(300, 20, 60, 60), 0.5);
        assert_eq!(window, Some((270, 0, 120, 120)));
    }

    #[test]
    fn smart_crop_on_a_uniform_image_matches_center() {
        let img = DynamicImage::new_rgb8(200, 100);
        let smart = crop_to_aspect(img.clone(), 1, 1, Align::Smart);
        let centered = crop_to_aspect(img, 1, 1, Align::Center);
        assert_eq!(smart.to_rgb8(), centered.to_rgb8());
    }

    fn face(x: u32, y: u32, w: u32, h: u32) -> crate::ai::FaceBox {
        crate::ai::FaceBox {
            x,
            y,
            w,
            h,
            score: 0.95,
        }
    }

    /// 400x300 gray, with a pure-red block exactly over the
    /// face box used below: x 300..360, y 20..80.
    fn face_canvas() -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(400, 300);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if (300..360).contains(&x) && (20..80).contains(&y) {
                Rgb([255, 0, 0])
            } else {
                Rgb([128, 128, 128])
            };
        }
        img
    }

    #[test]
    fn face_crop_centers_the_padded_face() {
        // Face 60x60 at center (330, 50); padding 0.5 makes the
        // region 120x120, clamped to y=0: origin (270, 0).
        let out = crop_around_face(face_canvas(), 1, 1, &face(300, 20, 60, 60), 0.5);
        assert_eq!((out.width(), out.height()), (120, 120));
        // The face lands at (30..90, 20..80) of the crop.
        assert_eq!(rgb(&out, 60, 50), [255, 0, 0]);
        assert_eq!(rgb(&out, 0, 0), [128, 128, 128]);
        assert_eq!(rgb(&out, 29, 50), [128, 128, 128]);
    }

    #[test]
    fn face_crop_without_padding_fits_the_target_aspect() {
        // Face 60x60, target 2:1, no padding: the region must
        // contain the face at aspect 2, so 120x60.
        let out = crop_around_face(face_canvas(), 2, 1, &face(300, 20, 60, 60), 0.0);
        assert_eq!((out.width(), out.height()), (120, 60));
        // Face center still centered: face at (30..90, 0..60).
        assert_eq!(rgb(&out, 60, 30), [255, 0, 0]);
        assert_eq!(rgb(&out, 0, 30), [128, 128, 128]);
    }

    #[test]
    fn face_crop_clamps_into_the_corner() {
        // Face at the very corner; heavy padding would push the
        // region outside on both axes.
        let mut img = DynamicImage::new_rgb8(200, 200);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if x < 40 && y < 40 {
                Rgb([255, 0, 0])
            } else {
                Rgb([128, 128, 128])
            };
        }
        let out = crop_around_face(img, 1, 1, &face(0, 0, 40, 40), 1.0);
        assert_eq!((out.width(), out.height()), (120, 120));
        assert_eq!(rgb(&out, 0, 0), [255, 0, 0]);
        assert_eq!(rgb(&out, 119, 119), [128, 128, 128]);
    }

    #[test]
    fn focal_crop_keeps_a_fitting_rectangle_fully_inside() {
        // A square crop of the 100x60 stripes is 60 wide. The
        // rectangle x 70..95 fits, so the window slides right
        // until the rectangle's right edge is inside: start 40,
        // window x 40..100 (a centered window would end at 80
        // and cut the rectangle).
        let out = crop_to_aspect_focal(vstripes(), 1, 1, (70, 0, 95, 60));
        assert_eq!((out.width(), out.height()), (60, 60));
        assert_eq!(rgb(&out, 0, 0), [0, 255, 0]);
        // The rectangle spans x 30..55 of the crop...
        assert_eq!(rgb(&out, 30, 0), [0, 255, 0]);
        assert_eq!(rgb(&out, 54, 0), [0, 0, 255]);
        // ...and the window's last column is the image's edge.
        assert_eq!(rgb(&out, 59, 59), [0, 0, 255]);
    }

    #[test]
    fn focal_crop_centers_on_an_oversized_rectangle() {
        // The rectangle x 10..95 is wider than the 60px window:
        // the window centers on the rectangle's center (52),
        // start 22, instead of pinning either edge.
        let out = crop_to_aspect_focal(vstripes(), 1, 1, (10, 0, 95, 60));
        assert_eq!((out.width(), out.height()), (60, 60));
        assert_eq!(rgb(&out, 0, 0), [0, 255, 0]);
        // x 81 in the source: blue. A centered crop (start 20)
        // would end at x 79, still green.
        assert_eq!(rgb(&out, 59, 0), [0, 0, 255]);
    }

    #[test]
    fn focal_crop_hugs_the_edge_the_rectangle_hugs() {
        // A rectangle against the left edge keeps the window
        // against the left edge too: start 0.
        let out = crop_to_aspect_focal(vstripes(), 1, 1, (0, 0, 25, 60));
        assert_eq!((out.width(), out.height()), (60, 60));
        assert_eq!(rgb(&out, 0, 0), [255, 0, 0]);
        assert_eq!(rgb(&out, 19, 0), [255, 0, 0]);
        assert_eq!(rgb(&out, 59, 0), [0, 255, 0]);
    }

    #[test]
    fn focal_crop_positions_on_the_vertical_axis() {
        // A square crop of the 60x100 stripes is 60 tall. The
        // rectangle y 70..95 fits, so the window slides down:
        // start 40, window y 40..100.
        let out = crop_to_aspect_focal(hstripes(), 1, 1, (0, 70, 60, 95));
        assert_eq!((out.width(), out.height()), (60, 60));
        assert_eq!(rgb(&out, 0, 0), [0, 255, 0]);
        assert_eq!(rgb(&out, 0, 59), [0, 0, 255]);
    }

    #[test]
    fn focal_crop_is_a_no_op_when_ratios_match() {
        let out = crop_to_aspect_focal(vstripes(), 5, 3, (70, 0, 95, 60));
        assert_eq!((out.width(), out.height()), (100, 60));
        assert_eq!(rgb(&out, 0, 0), [255, 0, 0]);
        assert_eq!(rgb(&out, 99, 0), [0, 0, 255]);
    }

    /// A face provider reporting fixed boxes, the stub
    /// pattern the focus tests use.
    #[derive(Debug)]
    struct StubFaces(Vec<crate::ai::FaceBox>);

    impl crate::ai::FaceProvider for StubFaces {
        fn detect(
            &self,
            _img: &DynamicImage,
        ) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
            Ok(self.0.clone())
        }
    }

    /// A face provider that always fails: detection errors
    /// never fail the crop they enhance.
    #[derive(Debug)]
    struct FailingFaces;

    impl crate::ai::FaceProvider for FailingFaces {
        fn detect(
            &self,
            _img: &DynamicImage,
        ) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
            Err(crate::ai::AiError::Inference(
                "detector exploded".to_string(),
            ))
        }
    }

    /// The window the pre-#199 rule cut for a traced smart
    /// crop: centered on the traced focal point, clamped.
    /// Recomputed from the trace so a test can show what
    /// face-safety changed.
    fn focal_window(trace: &CropTrace, img_w: u32, img_h: u32) -> (u32, u32, u32, u32) {
        let point = trace.focal.unwrap();
        let (x, y, cw, ch) = trace.window;
        if cw < img_w {
            (point_start(point.x, cw, img_w), y, cw, ch)
        } else {
            (x, point_start(point.y, ch, img_h), cw, ch)
        }
    }

    #[test]
    fn face_safe_smart_crop_keeps_an_edge_face_whole() {
        // 400x200, red block over the face box at the right
        // edge (x 360..400). A square crop is 200 wide: the
        // centered window (100..300) misses the face
        // entirely, while the face-safe smart window shifts
        // to the edge and keeps it whole.
        let mut img = DynamicImage::new_rgb8(400, 200);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if (360..400).contains(&x) && (80..120).contains(&y) {
                Rgb([255, 0, 0])
            } else {
                Rgb([128, 128, 128])
            };
        }
        let stub = StubFaces(vec![face(360, 80, 40, 40)]);
        let (out, trace) =
            crop_to_aspect_traced(img.clone(), 1, 1, Align::Smart, Some(&stub), None);
        let trace = trace.unwrap();
        assert_eq!(trace.window, (200, 0, 200, 200));
        assert_eq!(trace.faces_detected, 1);
        assert_eq!(trace.faces_kept, 1);
        assert!(has_pure_red(&out));
        let centered = crop_to_aspect(img, 1, 1, Align::Center);
        assert!(!has_pure_red(&centered));
    }

    #[test]
    fn face_safe_smart_crop_shifts_to_keep_two_faces_whole() {
        // Two faces whose spans together exactly fill the
        // 200-wide window: only start 30 keeps both. The
        // area-weighted centroid sits at x 84, so the focal
        // window (start 0) would cut the smaller face.
        let img = DynamicImage::new_rgb8(400, 200);
        let boxes = vec![face(30, 80, 40, 40), face(210, 90, 20, 20)];
        let stub = StubFaces(boxes.clone());
        let (_, trace) = crop_to_aspect_traced(img, 1, 1, Align::Smart, Some(&stub), None);
        let trace = trace.unwrap();
        assert_eq!(trace.window, (30, 0, 200, 200));
        assert_eq!(trace.faces_detected, 2);
        assert_eq!(trace.faces_kept, 2);
        let old = focal_window(&trace, 400, 200);
        assert_eq!(old, (0, 0, 200, 200));
        assert_eq!(faces_inside(&boxes, old), 1);
    }

    #[test]
    fn face_safe_smart_crop_keeps_the_larger_face_when_both_cannot_fit() {
        // The spans cover the whole width, so no window
        // keeps both. The centroid window (start 38) keeps
        // neither whole; face-safety moves to the larger
        // face and the record shows 1 of 2 kept.
        let img = DynamicImage::new_rgb8(400, 200);
        let boxes = vec![face(0, 80, 60, 60), face(360, 80, 40, 40)];
        let stub = StubFaces(boxes.clone());
        let (_, trace) = crop_to_aspect_traced(img, 1, 1, Align::Smart, Some(&stub), None);
        let trace = trace.unwrap();
        assert_eq!(trace.window, (0, 0, 200, 200));
        assert_eq!(trace.faces_detected, 2);
        assert_eq!(trace.faces_kept, 1);
        assert_eq!(faces_inside(&boxes, trace.window), 1);
        assert_eq!(trace.faces[0], boxes[0]);
        let old = focal_window(&trace, 400, 200);
        assert_eq!(old, (38, 0, 200, 200));
        assert_eq!(faces_inside(&boxes, old), 0);
    }

    #[test]
    fn face_safe_smart_crop_centers_a_face_larger_than_the_window() {
        // A 4:1 crop of the 400x200 source is 400x100,
        // sliding on y. The face is 120 tall: it can never
        // be whole, so the window centers on it as before,
        // and the record honestly keeps 0.
        let img = DynamicImage::new_rgb8(400, 200);
        let stub = StubFaces(vec![face(150, 40, 100, 120)]);
        let (_, trace) = crop_to_aspect_traced(img, 4, 1, Align::Smart, Some(&stub), None);
        let trace = trace.unwrap();
        assert_eq!(trace.window, (0, 50, 400, 100));
        assert_eq!(trace.faces_detected, 1);
        assert_eq!(trace.faces_kept, 0);
        // Window center == face center on the sliding axis.
        assert_eq!(trace.window.1 + trace.window.3 / 2, 40 + 120 / 2);
    }

    #[test]
    fn smart_crop_without_faces_matches_todays_window() {
        // Regression pin: no provider, an empty provider,
        // and a failing provider all produce exactly the
        // window the saliency rule has always cut on this
        // fixture, and record no faces.
        let expected = (100, 0, 100, 100);
        let (_, trace) = crop_to_aspect_traced(right_block(), 1, 1, Align::Smart, None, None);
        let trace = trace.unwrap();
        assert_eq!(trace.window, expected);
        assert_eq!(trace.faces_detected, 0);
        assert_eq!(trace.faces_kept, 0);
        let empty = StubFaces(Vec::new());
        let (_, trace) =
            crop_to_aspect_traced(right_block(), 1, 1, Align::Smart, Some(&empty), None);
        assert_eq!(trace.unwrap().window, expected);
        let (_, trace) =
            crop_to_aspect_traced(right_block(), 1, 1, Align::Smart, Some(&FailingFaces), None);
        assert_eq!(trace.unwrap().window, expected);
    }

    #[test]
    fn face_safe_smart_crop_keeps_the_portrait_fixture_face_whole() {
        // The checked-in portrait (400x499) with a stub
        // reporting its measured face box (the real-model
        // test in ai.rs brackets x 130..175, y 15..50,
        // w 70..110, h 115..165; the stub uses the middle).
        let bytes = include_bytes!("../tests/fixtures/portrait.jpg");
        let img = image::load_from_memory(bytes).unwrap();
        assert_eq!((img.width(), img.height()), (400, 499));
        let stub = StubFaces(vec![face(150, 30, 90, 140)]);
        for (tw, th, window) in [(1, 1, (0, 0, 400, 400)), (2, 1, (0, 0, 400, 200))] {
            let (_, trace) =
                crop_to_aspect_traced(img.clone(), tw, th, Align::Smart, Some(&stub), None);
            let trace = trace.unwrap();
            assert_eq!(trace.window, window, "target {tw}:{th}");
            assert_eq!(trace.faces_detected, 1);
            assert_eq!(trace.faces_kept, 1);
        }
    }
}
