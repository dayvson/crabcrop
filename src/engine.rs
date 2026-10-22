//! The image engine: decode, transform, and encode.
//!
//! [`Engine::process`] takes source bytes and the parsed [`Operation`]s,
//! and returns the encoded output plus its format.

use std::sync::Once;

use image::{DynamicImage, ExtendedColorType, ImageEncoder};

use crate::crop::crop_to_aspect;
use crate::filters::apply_filters;
use crate::quality::auto_quality;
use crate::url::{Dimension, ImageFormat, ImageSize, Operation, Quality};

/// Register the JPEG XL decoder with the image crate, once.
///
/// The image crate has no JPEG XL support of its own; the jxl-rs
/// integration hooks its format detection and decoder in globally.
fn ensure_jxl_decoder() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        jxl_image_rs_integration::register_image_decoding_hook();
    });
}

/// The encoded output plus its container format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessedImage {
    pub bytes: Vec<u8>,
    pub format: ImageFormat,
    /// The quality chosen for this output, set only when the request
    /// asked for `quality(auto)` and content analysis picked one.
    pub quality: Option<u8>,
}

/// What went wrong while processing an image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// The source bytes were not a decodable image. Carries the cause.
    Decode(String),
    /// Encoding the output failed. Carries the cause.
    Encode(String),
    /// The operation is not supported yet. Carries what was asked for.
    Unsupported(String),
}

/// Decodes, resizes, and encodes images.
#[derive(Debug, Clone, Copy, Default)]
pub struct Engine;

impl Engine {
    /// Run the pipeline: decode, apply operations in order, encode.
    ///
    /// The output format is the detected source format unless a
    /// `format(...)` filter overrides it (JPEG when the source format
    /// cannot be detected and no filter overrides it). A `crop:W×H`
    /// crops to the W:H aspect ratio and then resizes to exactly W×H.
    pub fn process(&self, bytes: &[u8], ops: &[Operation]) -> Result<ProcessedImage, EngineError> {
        ensure_jxl_decoder();
        let mut img =
            image::load_from_memory(bytes).map_err(|e| EngineError::Decode(e.to_string()))?;
        let mut format = detect_format(bytes);
        let mut quality = None;
        for op in ops {
            match op {
                Operation::Size(size) => img = apply_size(img, size),
                Operation::Crop { size, align } => {
                    let (tw, th) = match (size.width, size.height) {
                        (Dimension::Pixels(w), Dimension::Pixels(h)) => (w, h),
                        _ => {
                            return Err(EngineError::Unsupported(
                                "crop with non-pixel dimensions".to_string(),
                            ));
                        }
                    };
                    img = crop_to_aspect(img, tw, th, *align);
                    img = apply_size(
                        img,
                        &ImageSize {
                            width: Dimension::Pixels(tw),
                            height: Dimension::Pixels(th),
                            flip_horizontal: false,
                            flip_vertical: false,
                        },
                    );
                }
                Operation::Filters(filters) => {
                    let (new_img, options) = apply_filters(img, filters);
                    img = new_img;
                    if let Some(f) = options.format {
                        format = f;
                    }
                    if let Some(q) = options.quality {
                        quality = Some(q);
                    }
                }
            }
        }
        // Resolve the quality directive against the final image.
        // `quality(auto)` is the only directive reported back: it is
        // the one the caller did not choose themselves.
        let (encode_quality, chosen_quality) = match quality {
            Some(Quality::Value(q)) => (Some(q), None),
            Some(Quality::Auto) => {
                let q = auto_quality(&img);
                (Some(q), Some(q))
            }
            None => (None, None),
        };
        let bytes = encode(&img, format, encode_quality)?;
        Ok(ProcessedImage {
            bytes,
            format,
            quality: chosen_quality,
        })
    }
}

/// The output container: detected source format, or JPEG when the
/// source format is unknown or not one we emit.
fn detect_format(bytes: &[u8]) -> ImageFormat {
    // The image crate's guess_format does not know JPEG XL.
    if is_jxl(bytes) {
        return ImageFormat::Jxl;
    }
    match image::guess_format(bytes) {
        Ok(image::ImageFormat::Jpeg) => ImageFormat::Jpeg,
        Ok(image::ImageFormat::Png) => ImageFormat::Png,
        Ok(image::ImageFormat::WebP) => ImageFormat::WebP,
        _ => ImageFormat::Jpeg,
    }
}

/// JPEG XL detection: the bare codestream signature, or the first
/// bytes of the ISO BMFF container signature.
fn is_jxl(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xFF, 0x0A])
        || bytes.starts_with(&[
            0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
        ])
}

/// Encode `img` to `format`. `quality` applies to JPEG and JPEG XL.
fn encode(
    img: &DynamicImage,
    format: ImageFormat,
    quality: Option<u8>,
) -> Result<Vec<u8>, EngineError> {
    let (w, h) = (img.width(), img.height());
    if format == ImageFormat::Jxl {
        // Pure-Rust encoder (jixel). Its quality is a libjpeg-like
        // 0-100 scale, matching the quality filter's semantics.
        let config = jixel::EncodeConfig::default().with_quality(quality.unwrap_or(80) as f32);
        let encoded = if img.color().has_alpha() {
            let rgba = img.to_rgba8();
            jixel::encode_image_with_alpha(rgba.as_raw(), w as usize, h as usize, &config)
        } else {
            let rgb = img.to_rgb8();
            jixel::encode_image(rgb.as_raw(), w as usize, h as usize, &config)
        };
        return encoded.map_err(|e| EngineError::Encode(e.to_string()));
    }
    let mut buf = Vec::new();
    let (pixels, color) = (img.as_bytes(), ExtendedColorType::from(img.color()));
    match format {
        ImageFormat::Jpeg => {
            // JPEG has no alpha channel.
            let rgb = img.to_rgb8();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality.unwrap_or(80))
                .write_image(rgb.as_raw(), w, h, ExtendedColorType::Rgb8)
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        ImageFormat::Png => {
            image::codecs::png::PngEncoder::new(&mut buf)
                .write_image(pixels, w, h, color)
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        ImageFormat::WebP => {
            // The image crate's WebP encoder is lossless-only.
            image::codecs::webp::WebPEncoder::new_lossless(&mut buf)
                .write_image(pixels, w, h, color)
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        ImageFormat::Jxl => unreachable!("JPEG XL returns early in encode"),
    }
    Ok(buf)
}

/// Resize `img` per `size`, honoring proportional/original dimensions
/// and flips.
fn apply_size(img: DynamicImage, size: &ImageSize) -> DynamicImage {
    let (sw, sh) = (img.width(), img.height());
    let w = match size.width {
        Dimension::Pixels(n) => n,
        Dimension::Original => sw,
        Dimension::Proportional => match size.height {
            Dimension::Pixels(h) => ((sw as u64 * h as u64) / sh as u64).max(1) as u32,
            _ => sw,
        },
    };
    let h = match size.height {
        Dimension::Pixels(n) => n,
        Dimension::Original => sh,
        Dimension::Proportional => match size.width {
            Dimension::Pixels(w) => ((sh as u64 * w as u64) / sw as u64).max(1) as u32,
            _ => sh,
        },
    };
    let mut img = if w == sw && h == sh {
        img
    } else {
        let resized = image::imageops::resize(&img, w, h, image::imageops::FilterType::Lanczos3);
        DynamicImage::ImageRgba8(resized)
    };
    if size.flip_horizontal {
        img = img.fliph();
    }
    if size.flip_vertical {
        img = img.flipv();
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::url::{Align, Filter, ImageFormat, Quality};
    use image::GenericImageView;

    fn png_bytes(img: &DynamicImage) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    fn solid_png(width: u32, height: u32, color: [u8; 3]) -> Vec<u8> {
        let mut img = DynamicImage::new_rgb8(width, height);
        for p in img.as_mut_rgb8().unwrap().pixels_mut() {
            *p = image::Rgb(color);
        }
        png_bytes(&img)
    }

    fn size_op(width: Dimension, height: Dimension) -> Operation {
        Operation::Size(ImageSize {
            width,
            height,
            flip_horizontal: false,
            flip_vertical: false,
        })
    }

    fn decode_dims(bytes: &[u8]) -> (u32, u32) {
        let img = image::load_from_memory(bytes).unwrap();
        (img.width(), img.height())
    }

    #[test]
    fn round_trips_png() {
        let engine = Engine;
        let out = engine
            .process(&solid_png(100, 60, [10, 20, 30]), &[])
            .unwrap()
            .bytes;
        assert_eq!(&out[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(decode_dims(&out), (100, 60));
    }

    #[test]
    fn resizes_to_exact_size() {
        let engine = Engine;
        let out = engine
            .process(
                &solid_png(100, 60, [10, 20, 30]),
                &[size_op(Dimension::Pixels(50), Dimension::Pixels(30))],
            )
            .unwrap()
            .bytes;
        assert_eq!(decode_dims(&out), (50, 30));
    }

    #[test]
    fn resizes_proportionally() {
        let engine = Engine;
        let bytes = solid_png(100, 60, [10, 20, 30]);
        let out = engine
            .process(
                &bytes,
                &[size_op(Dimension::Pixels(50), Dimension::Proportional)],
            )
            .unwrap()
            .bytes;
        assert_eq!(decode_dims(&out), (50, 30));
        let out = engine
            .process(
                &bytes,
                &[size_op(Dimension::Proportional, Dimension::Pixels(30))],
            )
            .unwrap()
            .bytes;
        assert_eq!(decode_dims(&out), (50, 30));
    }

    #[test]
    fn keeps_original_dimensions() {
        let engine = Engine;
        let out = engine
            .process(
                &solid_png(100, 60, [10, 20, 30]),
                &[size_op(Dimension::Original, Dimension::Original)],
            )
            .unwrap()
            .bytes;
        assert_eq!(decode_dims(&out), (100, 60));
    }

    #[test]
    fn flips_horizontally() {
        let mut img = DynamicImage::new_rgb8(2, 1);
        let rgb = img.as_mut_rgb8().unwrap();
        rgb.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        rgb.put_pixel(1, 0, image::Rgb([0, 0, 255]));
        let engine = Engine;
        let out = engine
            .process(
                &png_bytes(&img),
                &[Operation::Size(ImageSize {
                    width: Dimension::Pixels(2),
                    height: Dimension::Pixels(1),
                    flip_horizontal: true,
                    flip_vertical: false,
                })],
            )
            .unwrap()
            .bytes;
        let decoded = image::load_from_memory(&out).unwrap();
        let left = decoded.get_pixel(0, 0);
        let right = decoded.get_pixel(1, 0);
        assert_eq!([left[0], left[1], left[2]], [0, 0, 255]);
        assert_eq!([right[0], right[1], right[2]], [255, 0, 0]);
    }

    #[test]
    fn preserves_jpeg_format() {
        let mut buf = std::io::Cursor::new(Vec::new());
        DynamicImage::new_rgb8(100, 60)
            .write_to(&mut buf, image::ImageFormat::Jpeg)
            .unwrap();
        let buf = buf.into_inner();
        let engine = Engine;
        let out = engine.process(&buf, &[]).unwrap().bytes;
        assert_eq!(&out[..2], b"\xff\xd8");
        assert_eq!(decode_dims(&out), (100, 60));
    }

    #[test]
    fn rejects_corrupt_input() {
        let engine = Engine;
        assert!(matches!(
            engine.process(b"not an image", &[]),
            Err(EngineError::Decode(_))
        ));
    }

    #[test]
    fn crops_then_resizes_to_exact_size() {
        let engine = Engine;
        let out = engine
            .process(
                &solid_png(100, 60, [10, 20, 30]),
                &[Operation::Crop {
                    size: ImageSize {
                        width: Dimension::Pixels(50),
                        height: Dimension::Pixels(50),
                        flip_horizontal: false,
                        flip_vertical: false,
                    },
                    align: Align::Center,
                }],
            )
            .unwrap()
            .bytes;
        assert_eq!(decode_dims(&out), (50, 50));
    }

    fn gradient_png() -> Vec<u8> {
        let mut img = DynamicImage::new_rgb8(64, 64);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let v = ((x + y) % 256) as u8;
            *p = image::Rgb([v, 255 - v, (x * y % 256) as u8]);
        }
        png_bytes(&img)
    }

    #[test]
    fn grayscale_filter_desaturates() {
        let engine = Engine;
        let out = engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![Filter::Grayscale])],
            )
            .unwrap()
            .bytes;
        let decoded = image::load_from_memory(&out).unwrap();
        for p in decoded.to_rgb8().pixels() {
            assert_eq!(p[0], p[1]);
            assert_eq!(p[1], p[2]);
        }
    }

    #[test]
    fn quality_filter_changes_jpeg_size() {
        let engine = Engine;
        let filters = |q| {
            vec![Operation::Filters(vec![
                Filter::Format(ImageFormat::Jpeg),
                Filter::Quality(Quality::Value(q)),
            ])]
        };
        let hi = engine.process(&gradient_png(), &filters(95)).unwrap();
        let lo = engine.process(&gradient_png(), &filters(10)).unwrap();
        assert_eq!(hi.format, ImageFormat::Jpeg);
        assert!(hi.bytes.len() > lo.bytes.len());
        assert_eq!(&hi.bytes[..2], b"\xff\xd8");
    }

    #[test]
    fn format_filter_selects_container() {
        let engine = Engine;
        // PNG in, JPEG out.
        let out = engine
            .process(
                &solid_png(100, 60, [10, 20, 30]),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jpeg)])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jpeg);
        assert_eq!(&out.bytes[..2], b"\xff\xd8");
        // JPEG in, PNG out.
        let mut jpeg = std::io::Cursor::new(Vec::new());
        DynamicImage::new_rgb8(100, 60)
            .write_to(&mut jpeg, image::ImageFormat::Jpeg)
            .unwrap();
        let out = engine
            .process(
                &jpeg.into_inner(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Png)])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Png);
        assert_eq!(&out.bytes[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn filters_combine_with_resize() {
        let engine = Engine;
        let out = engine
            .process(
                &gradient_png(),
                &[
                    size_op(Dimension::Pixels(32), Dimension::Pixels(32)),
                    Operation::Filters(vec![Filter::Grayscale, Filter::Format(ImageFormat::Jpeg)]),
                ],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jpeg);
        let decoded = image::load_from_memory(&out.bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (32, 32));
        let p = decoded.get_pixel(0, 0);
        assert_eq!(p[0], p[1]);
    }

    #[test]
    fn new_filters_flow_through_engine() {
        let engine = Engine;
        let out = engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![
                    Filter::Blur(1.5),
                    Filter::Invert,
                    Filter::Format(ImageFormat::Png),
                ])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Png);
        assert!(out.bytes.starts_with(&[0x89, b'P', b'N', b'G']));
        image::load_from_memory(&out.bytes).unwrap();
    }

    #[test]
    fn jxl_format_encodes() {
        let engine = Engine;
        let out = engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jxl)])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jxl);
        // Bare codestream signature.
        assert!(out.bytes.starts_with(&[0xFF, 0x0A]));
    }

    #[test]
    fn jxl_source_decodes_and_round_trips() {
        let engine = Engine;
        let jxl = engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jxl)])],
            )
            .unwrap();
        // The JXL output becomes the source: resize with no format
        // filter. Detection must keep JPEG XL and decoding must work.
        let out = engine
            .process(
                &jxl.bytes,
                &[size_op(Dimension::Pixels(16), Dimension::Pixels(16))],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jxl);
        let decoded = image::load_from_memory(&out.bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (16, 16));
    }

    #[test]
    fn jxl_quality_changes_size() {
        let engine = Engine;
        let mk = |q| {
            engine
                .process(
                    &gradient_png(),
                    &[Operation::Filters(vec![
                        Filter::Quality(Quality::Value(q)),
                        Filter::Format(ImageFormat::Jxl),
                    ])],
                )
                .unwrap()
        };
        assert!(mk(95).bytes.len() > mk(10).bytes.len());
    }

    fn flat_png() -> Vec<u8> {
        png_bytes(&DynamicImage::ImageLuma8(image::GrayImage::from_pixel(
            120,
            120,
            image::Luma([128]),
        )))
    }

    fn noise_png() -> Vec<u8> {
        let mut state: u32 = 0x2545_F491;
        let mut img = DynamicImage::new_rgb8(120, 120);
        for p in img.as_mut_rgb8().unwrap().pixels_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *p = image::Rgb([(state >> 24) as u8, (state >> 16) as u8, state as u8]);
        }
        png_bytes(&img)
    }

    #[test]
    fn auto_quality_resolves_and_reports() {
        let engine = Engine;
        let ops = &[Operation::Filters(vec![
            Filter::Quality(Quality::Auto),
            Filter::Format(ImageFormat::Jpeg),
        ])];
        let flat = engine.process(&flat_png(), ops).unwrap();
        assert_eq!(flat.quality, Some(50));
        let noisy = engine.process(&noise_png(), ops).unwrap();
        let q = noisy.quality.expect("auto quality is reported");
        assert!((50..=95).contains(&q), "chosen quality {q}");
        assert!(q > flat.quality.unwrap());
        // The detailed image ends up larger than the flat one.
        assert!(noisy.bytes.len() > flat.bytes.len());
    }

    #[test]
    fn explicit_quality_reports_nothing() {
        let engine = Engine;
        let out = engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![Filter::Quality(Quality::Value(
                    70,
                ))])],
            )
            .unwrap();
        assert_eq!(out.quality, None);
    }
}
