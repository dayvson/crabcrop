//! Applying filter operations to decoded images.
//!
//! [`apply_filters`] runs pixel filters in URL order and collects
//! encoding directives. `Format` and `Quality` do not touch pixels;
//! they are recorded for the encoder.

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
}

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
            Filter::Quality(q) => options.quality = Some(*q),
            Filter::NoUpscale => options.no_upscale = true,
            Filter::Format(f) => options.format = Some(*f),
        }
    }
    (img, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, Rgb};

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
}
