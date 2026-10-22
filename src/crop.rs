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
pub fn crop_to_aspect(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    align: Align,
) -> DynamicImage {
    if target_w == 0 || target_h == 0 {
        return img;
    }
    let (w, h) = (img.width(), img.height());
    // Compare w/h with target_w/target_h via cross-multiplication.
    let src_ratio = w as u64 * target_h as u64;
    let dst_ratio = target_w as u64 * h as u64;
    if src_ratio == dst_ratio {
        return img;
    }
    let (x, y, cw, ch) = if src_ratio > dst_ratio {
        // Source is wider: crop the width.
        let cw = ((h as u64 * target_w as u64) / target_h as u64).max(1) as u32;
        let x = match align {
            Align::Left => 0,
            Align::Right => w.saturating_sub(cw),
            Align::Smart => smart_start(&img, cw, w, true),
            _ => (w.saturating_sub(cw)) / 2,
        };
        (x, 0, cw, h)
    } else {
        // Source is taller: crop the height.
        let ch = ((w as u64 * target_h as u64) / target_w as u64).max(1) as u32;
        let y = match align {
            Align::Top => 0,
            Align::Bottom => h.saturating_sub(ch),
            Align::Smart => smart_start(&img, ch, h, false),
            _ => (h.saturating_sub(ch)) / 2,
        };
        (0, y, w, ch)
    };
    let cropped = image::imageops::crop_imm(&img, x, y, cw, ch).to_image();
    DynamicImage::ImageRgba8(cropped)
}

/// Window start along one axis for smart alignment: center the
/// `window` span on the focal point's coordinate along that
/// axis, clamped so the window stays inside the image.
fn smart_start(img: &DynamicImage, window: u32, extent: u32, horizontal: bool) -> u32 {
    let f = crate::focus::focal_point(img);
    let frac = if horizontal { f.x } else { f.y };
    let center = (frac * extent as f32).round() as i32;
    let max_start = extent.saturating_sub(window) as i32;
    (center - window as i32 / 2).clamp(0, max_start) as u32
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
    fn smart_crop_on_a_uniform_image_matches_center() {
        let img = DynamicImage::new_rgb8(200, 100);
        let smart = crop_to_aspect(img.clone(), 1, 1, Align::Smart);
        let centered = crop_to_aspect(img, 1, 1, Align::Center);
        assert_eq!(smart.to_rgb8(), centered.to_rgb8());
    }
}
