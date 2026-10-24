//! The `debug:focal` view: draw the focal decision a crop
//! made (or would make) on the canvas it operated on.
//!
//! The engine runs the chain normally and hands this module
//! the values the real crop code path used — the focal
//! point, the focal rectangle when one was in play, the crop
//! window, and any detected face boxes that fed the decision
//! — captured in a [`FocalCapture`] at the moment of the
//! crop. Nothing here recomputes a decision; it only draws.
//!
//! Marker scheme (fixed, so tests and docs can name it):
//!
//! - the focal crosshair and the focal rectangle outline are
//!   magenta, [`FOCAL_COLOR`];
//! - the crop window outline is cyan-blue, [`WINDOW_COLOR`];
//! - face boxes that fed the decision are amber,
//!   [`FACE_COLOR`].
//!
//! Rectangles are drawn as border bands of `thickness`
//! pixels just inside their edges, so the corner pixels of a
//! rectangle carry its color exactly. Line thickness and the
//! crosshair arm length scale with the smaller image
//! dimension, so the markers stay visible at any size.

use image::{DynamicImage, Rgba};

/// Crosshair and focal-rectangle color: opaque magenta.
pub const FOCAL_COLOR: Rgba<u8> = Rgba([255, 0, 128, 255]);
/// Crop-window outline color: opaque cyan-blue.
pub const WINDOW_COLOR: Rgba<u8> = Rgba([0, 180, 255, 255]);
/// Feeding face-box color: opaque amber.
pub const FACE_COLOR: Rgba<u8> = Rgba([255, 170, 0, 255]);

/// Everything the debug view draws, in the pixel coordinates
/// of `canvas`: the image as it stood when the crop ran (or
/// the final image, when no crop ran).
#[derive(Debug, Clone)]
pub struct FocalCapture {
    /// The image the markers are drawn on.
    pub canvas: DynamicImage,
    /// The chosen focal point, in canvas pixels.
    pub focal: Option<(u32, u32)>,
    /// The focal rectangle in play, as edges (left, top,
    /// right-exclusive, bottom-exclusive): a caller `focal:`
    /// rectangle, or the face box a `face` crop centered on.
    pub focal_rect: Option<(u32, u32, u32, u32)>,
    /// The crop window the crop cut, as (left, top, width,
    /// height); `None` when no crop ran.
    pub window: Option<(u32, u32, u32, u32)>,
    /// Detected face boxes that fed the focal decision.
    pub faces: Vec<crate::ai::FaceBox>,
}

/// Draw the capture's markers on its canvas and return the
/// diagnostic image. Draw order is faces, focal rectangle,
/// crop window, crosshair last, so the point itself is never
/// hidden under an outline that passes through it.
pub fn render(capture: &FocalCapture) -> DynamicImage {
    let mut img = capture.canvas.to_rgba8();
    let (w, h) = (img.width(), img.height());
    let min_dim = w.min(h);
    let thickness = (min_dim / 128).clamp(1, 6);
    for face in &capture.faces {
        draw_rect(
            &mut img,
            (
                face.x,
                face.y,
                face.x.saturating_add(face.w),
                face.y.saturating_add(face.h),
            ),
            FACE_COLOR,
            thickness,
        );
    }
    if let Some(rect) = capture.focal_rect {
        draw_rect(&mut img, rect, FOCAL_COLOR, thickness);
    }
    if let Some((x, y, cw, ch)) = capture.window {
        draw_rect(
            &mut img,
            (x, y, x.saturating_add(cw), y.saturating_add(ch)),
            WINDOW_COLOR,
            thickness,
        );
    }
    if let Some((fx, fy)) = capture.focal {
        let arm = (min_dim / 12).max(6) as i64;
        let half = (thickness / 2) as i64;
        for d in -arm..=arm {
            for off in -half..=(thickness as i64 - 1 - half) {
                put(&mut img, fx as i64 + d, fy as i64 + off, FOCAL_COLOR);
                put(&mut img, fx as i64 + off, fy as i64 + d, FOCAL_COLOR);
            }
        }
    }
    DynamicImage::ImageRgba8(img)
}

/// Draw the border bands of the rectangle with edges
/// (left, top, right-exclusive, bottom-exclusive): `t`
/// pixels inward from each edge, clipped to the image.
fn draw_rect(img: &mut image::RgbaImage, edges: (u32, u32, u32, u32), color: Rgba<u8>, t: u32) {
    let (l, top, r, b) = (
        edges.0 as i64,
        edges.1 as i64,
        edges.2 as i64,
        edges.3 as i64,
    );
    let t = t as i64;
    for y in top..b {
        for x in l..r {
            if x - l < t || r - 1 - x < t || y - top < t || b - 1 - y < t {
                put(img, x, y, color);
            }
        }
    }
}

fn put(img: &mut image::RgbaImage, x: i64, y: i64, color: Rgba<u8>) {
    if x >= 0 && y >= 0 && x < img.width() as i64 && y < img.height() as i64 {
        img.put_pixel(x as u32, y as u32, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    fn capture(w: u32, h: u32) -> FocalCapture {
        FocalCapture {
            canvas: DynamicImage::new_rgba8(w, h),
            focal: None,
            focal_rect: None,
            window: None,
            faces: Vec::new(),
        }
    }

    fn px(img: &DynamicImage, x: u32, y: u32) -> Rgba<u8> {
        img.get_pixel(x, y)
    }

    #[test]
    fn window_corners_carry_the_window_color() {
        let mut cap = capture(40, 30);
        cap.window = Some((5, 4, 20, 10));
        let out = render(&cap);
        // All four corners of the window outline.
        assert_eq!(px(&out, 5, 4), WINDOW_COLOR);
        assert_eq!(px(&out, 24, 4), WINDOW_COLOR);
        assert_eq!(px(&out, 5, 13), WINDOW_COLOR);
        assert_eq!(px(&out, 24, 13), WINDOW_COLOR);
        // The interior and the outside stay untouched.
        assert_eq!(px(&out, 15, 9), Rgba([0, 0, 0, 0]));
        assert_eq!(px(&out, 4, 4), Rgba([0, 0, 0, 0]));
    }

    #[test]
    fn crosshair_center_and_arms_carry_the_focal_color() {
        let mut cap = capture(100, 100);
        cap.focal = Some((50, 50));
        let out = render(&cap);
        assert_eq!(px(&out, 50, 50), FOCAL_COLOR);
        // Arm length is min_dim/12 = 8 on a 100px canvas.
        assert_eq!(px(&out, 58, 50), FOCAL_COLOR);
        assert_eq!(px(&out, 50, 42), FOCAL_COLOR);
        assert_eq!(px(&out, 59, 50), Rgba([0, 0, 0, 0]));
        // Off-axis pixels are not part of the cross.
        assert_eq!(px(&out, 55, 55), Rgba([0, 0, 0, 0]));
    }

    #[test]
    fn focal_rect_and_face_boxes_use_their_own_colors() {
        let mut cap = capture(60, 60);
        cap.focal_rect = Some((10, 10, 30, 30));
        cap.faces = vec![crate::ai::FaceBox {
            x: 35,
            y: 35,
            w: 20,
            h: 20,
            score: 0.9,
        }];
        let out = render(&cap);
        assert_eq!(px(&out, 10, 10), FOCAL_COLOR);
        assert_eq!(px(&out, 29, 29), FOCAL_COLOR);
        assert_eq!(px(&out, 35, 35), FACE_COLOR);
        assert_eq!(px(&out, 54, 54), FACE_COLOR);
        // A face interior is not an outline.
        assert_eq!(px(&out, 45, 45), Rgba([0, 0, 0, 0]));
    }

    #[test]
    fn thickness_scales_with_the_image() {
        // 256px canvas: thickness 2, so the second row of the
        // window's top band is drawn too.
        let mut cap = capture(256, 256);
        cap.window = Some((10, 10, 100, 100));
        let out = render(&cap);
        assert_eq!(px(&out, 50, 10), WINDOW_COLOR);
        assert_eq!(px(&out, 50, 11), WINDOW_COLOR);
        assert_eq!(px(&out, 50, 12), Rgba([0, 0, 0, 0]));
    }
}
