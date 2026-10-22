//! Focal point detection: where the detail in an image is.
//!
//! [`focal_point_with`] is the entry point the cropper uses:
//! faces first when a face provider is configured (see
//! [`face_focal_point`]), the saliency estimate of
//! [`focal_point`] otherwise. That arrangement is deliberate:
//! crop code never learns where the point came from.

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
pub fn focal_point_with(
    img: &DynamicImage,
    faces: Option<&dyn crate::ai::FaceProvider>,
) -> FocalPoint {
    if let Some(provider) = faces
        && let Ok(found) = provider.detect(img)
        && let Some(point) = face_focal_point(img, &found)
    {
        return point;
    }
    focal_point(img)
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
        img.thumbnail(MAX_DIM, MAX_DIM)
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

    /// A provider that reports one face at a fixed place.
    #[derive(Debug)]
    struct FixedFace;

    impl crate::ai::FaceProvider for FixedFace {
        fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::FaceBox>, String> {
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
        fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::FaceBox>, String> {
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
    fn no_faces_falls_back_to_saliency() {
        let img = block_at(70, 40);
        assert_eq!(focal_point_with(&img, Some(&NoFaces)), focal_point(&img));
        assert_eq!(focal_point_with(&img, None), focal_point(&img));
    }
}
