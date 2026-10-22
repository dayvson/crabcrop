//! Applying filter operations to decoded images.
//!
//! [`apply_filters`] runs pixel filters in URL order and collects
//! encoding directives. `Format`, `Quality`, `Fill`, and
//! `MaxBytes` do not touch pixels; they are recorded for the
//! encoder.

use image::DynamicImage;

use crate::url::{Filter, ImageFormat, Quality};

/// Encoding directives collected from filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeOptions {
    /// Output container override. `None` keeps the detected source format.
    pub format: Option<ImageFormat>,
    /// Encoding quality directive. `None` means the encoder default.
    pub quality: Option<Quality>,
    /// Cap every resize at the source dimensions.
    pub no_upscale: bool,
    /// Background color for flattening transparency at encode time.
    pub fill: Option<[u8; 3]>,
    /// Byte budget for the encoded output, quality permitting.
    pub max_bytes: Option<u64>,
}

impl EncodeOptions {
    /// Merge `other` into `self`: directives set in `other` win,
    /// unset ones leave `self` alone. `no_upscale` is sticky:
    /// once requested anywhere, it stays requested.
    pub fn absorb(&mut self, other: EncodeOptions) {
        if other.format.is_some() {
            self.format = other.format;
        }
        if other.quality.is_some() {
            self.quality = other.quality;
        }
        if other.fill.is_some() {
            self.fill = other.fill;
        }
        if other.max_bytes.is_some() {
            self.max_bytes = other.max_bytes;
        }
        self.no_upscale |= other.no_upscale;
    }
}

/// Watermark images resolved by the server, keyed by the URL
/// the request referenced. The engine only ever sees pixels;
/// fetching happens outside it.
pub type WatermarkImages = std::collections::HashMap<String, DynamicImage>;

/// Apply `filters` in order.
///
/// Returns the transformed image plus the encoding directives.
/// Later directives win when a filter repeats.
pub fn apply_filters(img: DynamicImage, filters: &[Filter]) -> (DynamicImage, EncodeOptions) {
    let mut img = img;
    let mut options = EncodeOptions::default();
    for filter in filters {
        match filter {
            // Normalize to RGB8: encoders universally accept it, and the
            // gray values are preserved (r == g == b).
            Filter::Grayscale => img = DynamicImage::ImageRgb8(img.grayscale().to_rgb8()),
            Filter::Invert => img.invert(),
            Filter::Blur(sigma) => img = img.blur(*sigma),
            Filter::Brighten(amount) => img = img.brighten(*amount),
            Filter::Contrast(amount) => {
                let rgb = image::imageops::contrast(&img.to_rgb8(), *amount);
                img = DynamicImage::ImageRgb8(rgb);
            }
            Filter::HueRotate(degrees) => img = img.huerotate(*degrees),
            Filter::Sharpen(amount) => {
                // Unsharp mask as a 3x3 convolution: the center is
                // boosted, the cross neighbors subtracted.
                let a = *amount;
                let kernel: [f32; 9] = [0.0, -a, 0.0, -a, 1.0 + 4.0 * a, -a, 0.0, -a, 0.0];
                let rgb = image::imageops::filter3x3(&img.to_rgb8(), &kernel);
                img = DynamicImage::ImageRgb8(rgb);
            }
            Filter::Saturation(amount) => {
                img = DynamicImage::ImageRgb8(saturate(&img.to_rgb8(), *amount));
            }
            Filter::Rgb(r, g, b) => {
                let mut rgb = img.to_rgb8();
                for p in rgb.pixels_mut() {
                    p[0] = (p[0] as i32 + r).clamp(0, 255) as u8;
                    p[1] = (p[1] as i32 + g).clamp(0, 255) as u8;
                    p[2] = (p[2] as i32 + b).clamp(0, 255) as u8;
                }
                img = DynamicImage::ImageRgb8(rgb);
            }
            Filter::Equalize => {
                img = DynamicImage::ImageRgb8(equalize(&img.to_rgb8()));
            }
            Filter::Rotate(degrees) => {
                img = match ((degrees % 360) + 360) % 360 {
                    90 => img.rotate90(),
                    180 => img.rotate180(),
                    270 => img.rotate270(),
                    _ => img,
                };
            }
            Filter::Trim(tolerance) => img = trim(img, *tolerance),
            // Watermarks are applied by the engine, which owns
            // the resolved images; one reaching this function is
            // unresolved by construction and leaves pixels alone.
            // (The engine never routes them here.)
            Filter::Watermark { .. } => {}
            // Background removal is applied by the engine, which
            // owns the mask provider. (Same routing as above.)
            Filter::RemoveBg => {}
            // Upscaling is applied by the engine, which owns the
            // upscale provider. (Same routing.)
            Filter::Upscale(_) => {}
            Filter::Fill(color) => options.fill = Some(*color),
            Filter::MaxBytes(n) => options.max_bytes = Some(*n),
            Filter::Quality(q) => options.quality = Some(*q),
            Filter::NoUpscale => options.no_upscale = true,
            Filter::Format(f) => options.format = Some(*f),
        }
    }
    (img, options)
}

/// Crop away the uniform border: keep the bounding box of
/// pixels that differ from the top-left corner color by more
/// than `tolerance` in any channel (alpha included). A fully
/// uniform image is returned unchanged, never emptied.
fn trim(img: DynamicImage, tolerance: u8) -> DynamicImage {
    let rgba = img.to_rgba8();
    let corner = *rgba.get_pixel(0, 0);
    let tol = tolerance as i32;
    let differs =
        |p: &image::Rgba<u8>| (0..4).any(|c| (p[c] as i32 - corner[c] as i32).abs() > tol);
    let (mut min_x, mut min_y) = (rgba.width(), rgba.height());
    let (mut max_x, mut max_y) = (0u32, 0u32);
    let mut found = false;
    for (x, y, p) in rgba.enumerate_pixels() {
        if differs(p) {
            found = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !found {
        return img;
    }
    DynamicImage::ImageRgba8(
        image::imageops::crop_imm(&rgba, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
            .to_image(),
    )
}

/// Scale each pixel's distance from its luma by `amount`:
/// 0 collapses to gray, 1 is the identity.
fn saturate(img: &image::RgbImage, amount: f32) -> image::RgbImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let luma = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        for c in 0..3 {
            p[c] = (luma + (p[c] as f32 - luma) * amount).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Histogram equalization, independently per channel.
fn equalize(img: &image::RgbImage) -> image::RgbImage {
    let total = (img.width() * img.height()) as u64;
    let mut maps: [[u8; 256]; 3] = [[0; 256]; 3];
    for c in 0..3 {
        let mut hist = [0u64; 256];
        for p in img.pixels() {
            hist[p[c] as usize] += 1;
        }
        let mut cdf = [0u64; 256];
        let mut acc = 0u64;
        for i in 0..256 {
            acc += hist[i];
            cdf[i] = acc;
        }
        let cdf_min = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0);
        for i in 0..256 {
            maps[c][i] = if total == cdf_min {
                i as u8
            } else {
                // Values below the minimum present value have a
                // CDF of 0 and map to 0.
                ((cdf[i].saturating_sub(cdf_min) as f64 * 255.0 / (total - cdf_min) as f64).round())
                    as u8
            };
        }
    }
    let mut out = img.clone();
    for p in out.pixels_mut() {
        for c in 0..3 {
            p[c] = maps[c][p[c] as usize];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn striped() -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(64, 64);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let v = ((x + y) % 256) as u8;
            *p = Rgb([v, 255 - v, (x * y % 256) as u8]);
        }
        img
    }

    fn solid(r: u8, g: u8, b: u8) -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(4, 4);
        for p in img.as_mut_rgb8().unwrap().pixels_mut() {
            *p = Rgb([r, g, b]);
        }
        img
    }

    fn first_pixel(img: &DynamicImage) -> [u8; 3] {
        let rgb = img.to_rgb8();
        let p = rgb.pixels().next().unwrap();
        [p[0], p[1], p[2]]
    }

    #[test]
    fn later_directives_win() {
        let (img, opts) = apply_filters(
            striped(),
            &[
                Filter::Quality(Quality::Value(10)),
                Filter::Quality(Quality::Value(90)),
                Filter::Format(ImageFormat::Png),
                Filter::Format(ImageFormat::Jpeg),
            ],
        );
        assert_eq!((img.width(), img.height()), (64, 64));
        assert_eq!(
            opts,
            EncodeOptions {
                format: Some(ImageFormat::Jpeg),
                quality: Some(Quality::Value(90)),
                no_upscale: false,
                fill: None,
                max_bytes: None,
            }
        );
    }

    #[test]
    fn no_upscale_is_collected_as_a_directive() {
        let img = DynamicImage::new_rgb8(8, 8);
        let (_, opts) = apply_filters(img, &[Filter::NoUpscale]);
        assert!(opts.no_upscale);
    }

    #[test]
    fn sharpen_increases_edge_contrast() {
        // Vertical step: left half 100, right half 200.
        let mut img = DynamicImage::new_rgb8(16, 16);
        for (x, _, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let v = if x < 8 { 100 } else { 200 };
            *p = Rgb([v, v, v]);
        }
        let (img, _) = apply_filters(img, &[Filter::Sharpen(1.0)]);
        let rgb = img.to_rgb8();
        let left = rgb.get_pixel(7, 8)[0] as i32;
        let right = rgb.get_pixel(8, 8)[0] as i32;
        assert!(right - left > 100, "left {left} right {right}");
    }

    #[test]
    fn saturation_scales_color() {
        let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::Saturation(0.0)]);
        let p = first_pixel(&img);
        assert_eq!(p[0], p[1]);
        assert_eq!(p[1], p[2]);
        let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::Saturation(1.0)]);
        assert_eq!(first_pixel(&img), [200, 100, 50]);
        let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::Saturation(2.0)]);
        let p = first_pixel(&img);
        assert!(p[0] as i32 - p[2] as i32 > 150, "pixel {p:?}");
    }

    #[test]
    fn rgb_offsets_channels() {
        let (img, _) = apply_filters(solid(100, 100, 100), &[Filter::Rgb(50, -30, 0)]);
        assert_eq!(first_pixel(&img), [150, 70, 100]);
    }

    #[test]
    fn equalize_spreads_the_range() {
        // All pixels in a narrow band, 100..=107.
        let mut img = DynamicImage::new_rgb8(16, 16);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let v = 100 + ((x + y) % 8) as u8;
            *p = Rgb([v, v, v]);
        }
        let (img, _) = apply_filters(img, &[Filter::Equalize]);
        let rgb = img.to_rgb8();
        let min = rgb.pixels().map(|p| p[0]).min().unwrap();
        let max = rgb.pixels().map(|p| p[0]).max().unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, 255);
    }

    #[test]
    fn rotate_swaps_dimensions() {
        let (img, _) = apply_filters(DynamicImage::new_rgb8(4, 2), &[Filter::Rotate(90)]);
        assert_eq!((img.width(), img.height()), (2, 4));
        let (img, _) = apply_filters(DynamicImage::new_rgb8(4, 2), &[Filter::Rotate(-90)]);
        assert_eq!((img.width(), img.height()), (2, 4));
        let (img, _) = apply_filters(DynamicImage::new_rgb8(4, 2), &[Filter::Rotate(180)]);
        assert_eq!((img.width(), img.height()), (4, 2));
    }

    #[test]
    fn fill_and_max_bytes_are_collected() {
        let (_, opts) = apply_filters(
            solid(1, 2, 3),
            &[Filter::Fill([255, 136, 0]), Filter::MaxBytes(5000)],
        );
        assert_eq!(opts.fill, Some([255, 136, 0]));
        assert_eq!(opts.max_bytes, Some(5000));
    }

    #[test]
    fn absorb_merges_directives() {
        let mut base = EncodeOptions::default();
        base.absorb(EncodeOptions {
            format: Some(ImageFormat::Png),
            quality: None,
            no_upscale: true,
            fill: None,
            max_bytes: Some(100),
        });
        base.absorb(EncodeOptions {
            format: None,
            quality: Some(Quality::Value(50)),
            no_upscale: false,
            fill: None,
            max_bytes: None,
        });
        assert_eq!(
            base,
            EncodeOptions {
                format: Some(ImageFormat::Png),
                quality: Some(Quality::Value(50)),
                no_upscale: true,
                fill: None,
                max_bytes: Some(100),
            }
        );
    }

    #[test]
    fn empty_filters_are_noop() {
        let (img, opts) = apply_filters(striped(), &[]);
        assert_eq!((img.width(), img.height()), (64, 64));
        assert_eq!(opts, EncodeOptions::default());
    }

    #[test]
    fn grayscale_desaturates() {
        let (img, _) = apply_filters(striped(), &[Filter::Grayscale]);
        for p in img.to_rgb8().pixels() {
            assert_eq!(p[0], p[1]);
            assert_eq!(p[1], p[2]);
        }
    }

    #[test]
    fn invert_flips_channels() {
        let (img, _) = apply_filters(solid(10, 128, 250), &[Filter::Invert]);
        assert_eq!(first_pixel(&img), [245, 127, 5]);
    }

    #[test]
    fn brighten_shifts_and_clamps() {
        let (img, _) = apply_filters(solid(100, 150, 200), &[Filter::Brighten(10)]);
        assert_eq!(first_pixel(&img), [110, 160, 210]);

        let (img, _) = apply_filters(solid(200, 200, 200), &[Filter::Brighten(100)]);
        assert_eq!(first_pixel(&img), [255, 255, 255]);

        let (img, _) = apply_filters(solid(100, 150, 200), &[Filter::Brighten(-300)]);
        assert_eq!(first_pixel(&img), [0, 0, 0]);
    }

    #[test]
    fn blur_smooths_a_hard_edge() {
        let mut img = DynamicImage::new_rgb8(32, 32);
        for (x, _y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let v = if x < 16 { 0 } else { 255 };
            *p = Rgb([v, v, v]);
        }
        let (img, _) = apply_filters(img, &[Filter::Blur(2.0)]);
        let rgb = img.to_rgb8();
        // Pixels hugging the edge are now a blend of black and white.
        for (x, y, p) in rgb.enumerate_pixels() {
            if x == 15 || x == 16 {
                assert!(p[0] > 0 && p[0] < 255, "pixel ({x}, {y}) = {p:?}");
            }
        }
    }

    #[test]
    fn huerotate_rotates_and_comes_back() {
        let (img, _) = apply_filters(solid(255, 0, 0), &[Filter::HueRotate(90)]);
        let px = first_pixel(&img);
        // The hue moved: pure red is gone.
        assert_ne!(px, [255, 0, 0]);

        // A full turn is (approximately) the identity.
        let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::HueRotate(360)]);
        let px = first_pixel(&img);
        for (got, want) in px.iter().zip([200, 100, 50]) {
            assert!((*got as i16 - want as i16).abs() <= 1, "pixel = {px:?}");
        }
    }

    #[test]
    fn contrast_pushes_values_from_mid_gray() {
        let (img, _) = apply_filters(solid(100, 100, 100), &[Filter::Contrast(2.0)]);
        let px = first_pixel(&img);
        assert!(px[0] < 100, "pixel = {px:?}");

        let (img, _) = apply_filters(solid(200, 200, 200), &[Filter::Contrast(2.0)]);
        let px = first_pixel(&img);
        assert!(px[0] > 200, "pixel = {px:?}");
    }

    #[test]
    fn new_filters_compose_in_order() {
        // invert then brighten(10): (10 -> 245 -> 255).
        let (img, _) = apply_filters(solid(10, 10, 10), &[Filter::Invert, Filter::Brighten(10)]);
        assert_eq!(first_pixel(&img), [255, 255, 255]);
    }

    /// 40x30 `bg` with an `fg` rectangle at (10..30, 8..22).
    fn bordered(bg: [u8; 3], fg: [u8; 3]) -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(40, 30);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if (10..30).contains(&x) && (8..22).contains(&y) {
                Rgb(fg)
            } else {
                Rgb(bg)
            };
        }
        img
    }

    #[test]
    fn trim_crops_to_the_inner_box() {
        let (img, _) = apply_filters(bordered([255, 255, 255], [255, 0, 0]), &[Filter::Trim(0)]);
        assert_eq!((img.width(), img.height()), (20, 14));
        assert_eq!(first_pixel(&img), [255, 0, 0]);
    }

    #[test]
    fn trim_tolerance_decides_what_counts_as_border() {
        // Corner 255, an inner ring of 245, black center 10..20.
        let build = || {
            let mut img = DynamicImage::new_rgb8(30, 30);
            for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
                *p = if (10..20).contains(&x) && (10..20).contains(&y) {
                    Rgb([0, 0, 0])
                } else if (5..25).contains(&x) && (5..25).contains(&y) {
                    Rgb([245, 245, 245])
                } else {
                    Rgb([255, 255, 255])
                };
            }
            img
        };
        // Tolerance 10: the ring (10 away) is still border.
        let (img, _) = apply_filters(build(), &[Filter::Trim(10)]);
        assert_eq!((img.width(), img.height()), (10, 10));
        // Tolerance 5: the ring is content.
        let (img, _) = apply_filters(build(), &[Filter::Trim(5)]);
        assert_eq!((img.width(), img.height()), (20, 20));
    }

    #[test]
    fn trim_uniform_image_is_a_noop() {
        let (img, _) = apply_filters(solid(77, 77, 77), &[Filter::Trim(10)]);
        assert_eq!((img.width(), img.height()), (4, 4));
        assert_eq!(first_pixel(&img), [77, 77, 77]);
    }
}
