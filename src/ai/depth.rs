//! Sample depth for the image-to-image providers (upscale and
//! region removal). Their models read RGB as floats and know
//! nothing of alpha or bit depth, so the providers read the
//! source at its own depth, write the result back at that
//! depth, and carry the source's alpha around the model
//! (issue #551). They used to read and write 8-bit RGB and
//! carry alpha only for 8-bit sources, so a 16-bit or float
//! RGBA source came back opaque, showing the color stored
//! under its transparent pixels, and lost its precision.

use image::imageops::FilterType;
use image::{DynamicImage, ImageBuffer, Luma, Pixel, Rgb, Rgba};

/// The three sample depths a decoded image comes in, read
/// from its color type rather than by matching variants, so
/// every layout of a depth (gray or color, with alpha or
/// without) takes the same path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SampleDepth {
    Eight,
    Sixteen,
    Float,
}

impl SampleDepth {
    pub(super) fn of(img: &DynamicImage) -> Self {
        let color = img.color();
        match color.bytes_per_pixel() / color.channel_count().max(1) {
            2 => SampleDepth::Sixteen,
            4 => SampleDepth::Float,
            _ => SampleDepth::Eight,
        }
    }
}

/// A sample type the providers read a source and write its
/// result at. The `u8` implementation is the arithmetic the
/// providers always had, so an 8-bit source renders exactly
/// as before.
pub(super) trait Sample: image::Primitive + 'static {
    /// `Rgb<Self>`, `Rgba<Self>` and `Luma<Self>`, named
    /// through the trait: the bound the image crate puts on
    /// its pixel types' samples cannot be named outside the
    /// crate, so generic code learns these are pixels here.
    type Rgb: Pixel<Subpixel = Self> + 'static;
    type Rgba: Pixel<Subpixel = Self> + 'static;
    type Luma: Pixel<Subpixel = Self> + 'static;
    /// The image's color, alpha dropped, at this depth.
    fn rgb(img: &DynamicImage) -> ImageBuffer<Self::Rgb, Vec<Self>>;
    /// The image with alpha at this depth (opaque where the
    /// image has none).
    fn rgba(img: &DynamicImage) -> ImageBuffer<Self::Rgba, Vec<Self>>;
    /// A sample as a model input value in 0..1. A float sample
    /// outside that range (a high-dynamic-range source) is
    /// clamped, as the 8-bit conversion clamps it.
    fn to_unit(self) -> f32;
    /// A model output value in 0..1 as a sample, clamped; a
    /// NaN reads as 0.
    fn from_unit(v: f32) -> Self;
    /// A model output value in 0..255 (the inpainting export's
    /// output range) as a sample, clamped; a NaN reads as 0.
    fn from_255(v: f32) -> Self;
    fn rgb_image(buf: ImageBuffer<Self::Rgb, Vec<Self>>) -> DynamicImage;
    fn rgba_image(buf: ImageBuffer<Self::Rgba, Vec<Self>>) -> DynamicImage;
}

impl Sample for u8 {
    type Rgb = Rgb<u8>;
    type Rgba = Rgba<u8>;
    type Luma = Luma<u8>;

    fn rgb(img: &DynamicImage) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        img.to_rgb8()
    }
    fn rgba(img: &DynamicImage) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        img.to_rgba8()
    }
    fn to_unit(self) -> f32 {
        self as f32 / 255.0
    }
    fn from_unit(v: f32) -> u8 {
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    }
    fn from_255(v: f32) -> u8 {
        v.clamp(0.0, 255.0).round() as u8
    }
    fn rgb_image(buf: ImageBuffer<Rgb<u8>, Vec<u8>>) -> DynamicImage {
        DynamicImage::ImageRgb8(buf)
    }
    fn rgba_image(buf: ImageBuffer<Rgba<u8>, Vec<u8>>) -> DynamicImage {
        DynamicImage::ImageRgba8(buf)
    }
}

impl Sample for u16 {
    type Rgb = Rgb<u16>;
    type Rgba = Rgba<u16>;
    type Luma = Luma<u16>;

    fn rgb(img: &DynamicImage) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
        img.to_rgb16()
    }
    fn rgba(img: &DynamicImage) -> ImageBuffer<Rgba<u16>, Vec<u16>> {
        img.to_rgba16()
    }
    fn to_unit(self) -> f32 {
        self as f32 / 65535.0
    }
    fn from_unit(v: f32) -> u16 {
        (v.clamp(0.0, 1.0) * 65535.0).round() as u16
    }
    fn from_255(v: f32) -> u16 {
        Self::from_unit(v / 255.0)
    }
    fn rgb_image(buf: ImageBuffer<Rgb<u16>, Vec<u16>>) -> DynamicImage {
        DynamicImage::ImageRgb16(buf)
    }
    fn rgba_image(buf: ImageBuffer<Rgba<u16>, Vec<u16>>) -> DynamicImage {
        DynamicImage::ImageRgba16(buf)
    }
}

impl Sample for f32 {
    type Rgb = Rgb<f32>;
    type Rgba = Rgba<f32>;
    type Luma = Luma<f32>;

    fn rgb(img: &DynamicImage) -> ImageBuffer<Rgb<f32>, Vec<f32>> {
        img.to_rgb32f()
    }
    fn rgba(img: &DynamicImage) -> ImageBuffer<Rgba<f32>, Vec<f32>> {
        img.to_rgba32f()
    }
    fn to_unit(self) -> f32 {
        Self::from_unit(self)
    }
    fn from_unit(v: f32) -> f32 {
        if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
    }
    fn from_255(v: f32) -> f32 {
        Self::from_unit(v / 255.0)
    }
    fn rgb_image(buf: ImageBuffer<Rgb<f32>, Vec<f32>>) -> DynamicImage {
        DynamicImage::ImageRgb32F(buf)
    }
    fn rgba_image(buf: ImageBuffer<Rgba<f32>, Vec<f32>>) -> DynamicImage {
        DynamicImage::ImageRgba32F(buf)
    }
}

/// A provider's RGB result with the source's alpha carried
/// over, at the result's depth. Whether there is alpha to
/// carry is the source's color type's answer, so every source
/// with an alpha channel keeps it, whatever its depth. When
/// the result is larger than the source (upscale), the alpha
/// plane is enlarged with a plain Triangle resize; at the same
/// size (region removal) it is the source's, sample for
/// sample. A source without alpha gets an RGB result.
pub(super) fn with_source_alpha<S: Sample>(
    src: &DynamicImage,
    rgb: ImageBuffer<S::Rgb, Vec<S>>,
) -> DynamicImage {
    if !src.color().has_alpha() {
        return S::rgb_image(rgb);
    }
    let (w, h) = rgb.dimensions();
    let rgba = S::rgba(src);
    let alpha = ImageBuffer::<S::Luma, Vec<S>>::from_fn(src.width(), src.height(), |x, y| {
        *S::Luma::from_slice(&[rgba.get_pixel(x, y).channels()[3]])
    });
    let alpha = if alpha.dimensions() == (w, h) {
        alpha
    } else {
        image::imageops::resize(&alpha, w, h, FilterType::Triangle)
    };
    let out = ImageBuffer::<S::Rgba, Vec<S>>::from_fn(w, h, |x, y| {
        let c = rgb.get_pixel(x, y).channels();
        *S::Rgba::from_slice(&[c[0], c[1], c[2], alpha.get_pixel(x, y).channels()[0]])
    });
    S::rgba_image(out)
}

#[cfg(test)]
mod tests;
