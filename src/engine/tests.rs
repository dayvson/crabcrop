use super::*;
use crate::filters::WatermarkImages;
use crate::url::FormatSpec;
use crate::url::Subsampling;
use crate::url::{Align, Filter, ImageFormat, Quality};
use image::GenericImageView;
use image::ImageEncoder;

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
fn validate_source_image_accepts_the_servable_set() {
    // The upload acceptance test (issue #168) walks the
    // pipeline's own decode branches.
    assert!(validate_source_image(&solid_png(20, 10, [1, 2, 3])).is_ok());
    let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10" fill="red"/></svg>"#;
    assert!(validate_source_image(svg).is_ok());
    // Non-images, truncated images, and empty bodies
    // are refused.
    assert!(validate_source_image(b"not an image at all").is_err());
    assert!(validate_source_image(b"").is_err());
    let truncated = &solid_png(20, 10, [1, 2, 3])[..20];
    assert!(validate_source_image(truncated).is_err());
    // A script-carrying SVG is still an SVG the
    // sanitizer renders safely, so it validates like
    // any other SVG source would.
    let hostile = br#"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><script>alert(1)</script><rect width="4" height="4"/></svg>"#;
    assert!(validate_source_image(hostile).is_ok());
}

#[test]
fn prepare_upload_reports_media_and_stores_the_sanitized_svg() {
    // Raster uploads store the caller's bytes verbatim
    // and report their kind.
    let png = solid_png(20, 10, [1, 2, 3]);
    let prepared = prepare_upload(&png).unwrap();
    assert_eq!(prepared.media, UploadMedia::Png);
    assert_eq!(prepared.stored, png);
    // An SVG upload validates, and what would be
    // stored is the sanitized document: the script is
    // gone, the drawing survives (audit R2-22).
    let hostile = br#"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><script>alert(1)</script><rect width="4" height="4"/></svg>"#;
    let prepared = prepare_upload(hostile).unwrap();
    assert_eq!(prepared.media, UploadMedia::Svg);
    let stored = String::from_utf8(prepared.stored).unwrap();
    assert!(!stored.contains("script"), "{stored}");
    assert!(stored.contains("<rect"), "{stored}");
    // The stored form is itself a valid upload.
    assert_eq!(
        prepare_upload(stored.as_bytes()).unwrap().media,
        UploadMedia::Svg
    );
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

/// A photo surrogate for perceptual tests: smooth
/// multi-frequency content (sums of sinusoids) at a size the
/// metric's scales can work with. Unlike the sawtooth
/// gradient, lossy codecs can encode it well.
fn photo_like_png() -> Vec<u8> {
    let mut img = DynamicImage::new_rgb8(240, 180);
    for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        let (fx, fy) = (x as f64, y as f64);
        let wave = |phase: f64, a: f64, b: f64| {
            128.0
                + 62.0 * (fx * a + phase).sin() * (fy * b + phase * 0.7).cos()
                + 24.0 * ((fx + fy) * 0.021 + phase).sin()
        };
        *p = image::Rgb([
            wave(0.0, 0.11, 0.07).clamp(0.0, 255.0) as u8,
            wave(2.1, 0.09, 0.13).clamp(0.0, 255.0) as u8,
            wave(4.2, 0.05, 0.05).clamp(0.0, 255.0) as u8,
        ]);
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
            Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
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
fn region_filters_apply_and_check_bounds() {
    let engine = Engine;
    let src = solid_png(100, 80, [200, 30, 30]);
    // Multiple regions in one chain apply.
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::BlurRegion {
                    x: 0,
                    y: 0,
                    w: 10,
                    h: 10,
                },
                Filter::PixelateRegion {
                    x: 90,
                    y: 70,
                    w: 10,
                    h: 10,
                },
            ])],
        )
        .unwrap();
    assert_eq!((out.width, out.height), (100, 80));
    // Overflow is an error naming the region and the bounds.
    let err = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::BlurRegion {
                x: 90,
                y: 70,
                w: 20,
                h: 20,
            }])],
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("blur_region(90,70,20,20) exceeds image bounds 100x80"),
        "got: {err}"
    );
    // Near-MAX coordinates must error, not wrap the
    // bounds check (or panic on it in debug builds).
    let err = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::BlurRegion {
                x: u32::MAX - 10,
                y: 0,
                w: 20,
                h: 10,
            }])],
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("exceeds image bounds 100x80"),
        "got: {err}"
    );
    // Bounds are the size at the point the filter runs:
    // after a resize to 50x40, a 60-wide region no longer fits.
    let err = engine
        .process(
            &src,
            &[
                size_op(Dimension::Pixels(50), Dimension::Pixels(40)),
                Operation::Filters(vec![Filter::PixelateRegion {
                    x: 0,
                    y: 0,
                    w: 60,
                    h: 10,
                }]),
            ],
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("pixelate_region(0,0,60,10) exceeds image bounds 50x40"),
        "got: {err}"
    );
}

#[test]
fn shape_filters_need_an_alpha_capable_format() {
    let engine = Engine;
    let src = solid_png(40, 40, [200, 30, 30]);
    let err = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Radius([10, 10, 10, 10]),
                Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
            ])],
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("radius needs an alpha-capable output format"),
        "got: {err}"
    );
    let err = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Mask(crate::url::MaskShape::Circle),
                Filter::Format(FormatSpec::plain(ImageFormat::Gif)),
            ])],
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("mask needs an alpha-capable output format"),
        "got: {err}"
    );
    // PNG output carries the transparency through.
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Radius([10, 10, 10, 10]),
                Filter::Format(FormatSpec::plain(ImageFormat::Png)),
            ])],
        )
        .unwrap();
    let decoded = image::load_from_memory(&out.bytes).unwrap();
    assert_eq!(decoded.to_rgba8().get_pixel(0, 0)[3], 0);
    assert_eq!(decoded.to_rgba8().get_pixel(20, 20)[3], 255);
    // Border is opaque paint: JPEG is fine.
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Border(3, [0, 255, 0]),
                Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
            ])],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Jpeg);
}

#[test]
fn format_filter_selects_container() {
    let engine = Engine;
    // PNG in, JPEG out.
    let out = engine
        .process(
            &solid_png(100, 60, [10, 20, 30]),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Jpeg,
            ))])],
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
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Png,
            ))])],
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
                Operation::Filters(vec![
                    Filter::Grayscale,
                    Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
                ]),
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
                Filter::Format(FormatSpec::plain(ImageFormat::Png)),
            ])],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Png);
    assert!(out.bytes.starts_with(&[0x89, b'P', b'N', b'G']));
    image::load_from_memory(&out.bytes).unwrap();
}

#[test]
fn noise_flows_through_engine_and_is_deterministic() {
    let engine = Engine;
    let run = |filters: Vec<Filter>| {
        engine
            .process(
                &gradient_png(),
                &[
                    size_op(Dimension::Pixels(32), Dimension::Pixels(32)),
                    Operation::Filters(filters),
                ],
            )
            .unwrap()
    };
    // The filter composes with resize and a format
    // directive like any other pixel filter.
    let noisy = run(vec![
        Filter::Noise(60),
        Filter::Format(FormatSpec::plain(ImageFormat::Png)),
    ]);
    assert_eq!(noisy.format, ImageFormat::Png);
    let decoded = image::load_from_memory(&noisy.bytes).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (32, 32));
    // The same request renders the same bytes on a
    // second run: the field is seeded from content.
    let again = run(vec![
        Filter::Noise(60),
        Filter::Format(FormatSpec::plain(ImageFormat::Png)),
    ]);
    assert_eq!(noisy.bytes, again.bytes);
    // And it is not the clean image.
    let clean = run(vec![Filter::Format(FormatSpec::plain(ImageFormat::Png))]);
    assert_ne!(noisy.bytes, clean.bytes);
    // Amount 0 is a byte-identical no-op end to end.
    let zero = run(vec![
        Filter::Noise(0),
        Filter::Format(FormatSpec::plain(ImageFormat::Png)),
    ]);
    assert_eq!(zero.bytes, clean.bytes);
}

#[test]
fn jxl_format_encodes() {
    let engine = Engine;
    let out = engine
        .process(
            &gradient_png(),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Jxl,
            ))])],
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
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Jxl,
            ))])],
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
                    Filter::Format(FormatSpec::plain(ImageFormat::Jxl)),
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
        Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
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

/// A PNG carrying a Display P3 profile and three known
/// pixels, built with the production injector.
fn p3_png() -> Vec<u8> {
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(3, 1, |x, _| {
        [
            image::Rgb([200, 80, 40]),
            image::Rgb([40, 120, 200]),
            image::Rgb([90, 160, 90]),
        ][x as usize]
    }));
    let icc = moxcms::ColorProfile::new_display_p3().encode().unwrap();
    crate::color::inject_png(
        &png_bytes(&img),
        Some(&icc),
        &crate::color::SourceMeta::default(),
        false,
    )
}

#[test]
fn a_wide_gamut_source_converts_to_srgb_by_default() {
    let engine = Engine;
    let out = engine
        .process(
            &p3_png(),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Png,
            ))])],
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    let expected: [[u8; 3]; 3] = [[216, 69, 17], [0, 122, 206], [61, 162, 81]];
    for (i, want) in expected.iter().enumerate() {
        let got = img.get_pixel(i as u32, 0).0;
        for (g, w) in got.iter().zip(want.iter()) {
            assert!(
                (i32::from(*g) - i32::from(*w)).abs() <= 2,
                "pixel {i}: {got:?} vs {want:?}"
            );
        }
    }
    // And the converted output carries no profile: sRGB
    // is the assumed space now.
    assert!(
        crate::color::extract_source_meta(&out.bytes, true)
            .icc
            .is_none()
    );
}

#[test]
fn keep_profile_keeps_pixels_and_embeds_the_profile() {
    let engine = Engine;
    let icc = moxcms::ColorProfile::new_display_p3().encode().unwrap();
    let out = engine
        .process(
            &p3_png(),
            &[Operation::Filters(vec![
                Filter::KeepProfile,
                Filter::Format(FormatSpec::plain(ImageFormat::Png)),
            ])],
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    assert_eq!(img.get_pixel(0, 0).0, [200, 80, 40]);
    assert_eq!(
        crate::color::extract_source_meta(&out.bytes, true).icc,
        Some(icc)
    );
}

#[test]
fn keep_profile_converts_when_the_output_cannot_tag() {
    // Issue #374: WebP output cannot embed the profile
    // in this pipeline, so keep_profile must convert
    // to sRGB instead of shipping P3 pixels labeled as
    // nothing. The kept and unkept WebP renders of the
    // P3 source must carry the same (converted) pixel.
    let engine = Engine;
    let kept = engine
        .process(
            &p3_png(),
            &[Operation::Filters(vec![
                Filter::KeepProfile,
                Filter::Format(FormatSpec::plain(ImageFormat::WebP)),
            ])],
        )
        .unwrap();
    let plain = engine
        .process(
            &p3_png(),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::WebP,
            ))])],
        )
        .unwrap();
    let kept_img = image::load_from_memory(&kept.bytes).unwrap().to_rgb8();
    let plain_img = image::load_from_memory(&plain.bytes).unwrap().to_rgb8();
    assert_eq!(kept_img.get_pixel(0, 0), plain_img.get_pixel(0, 0));
    // And the converted pixel is not the raw P3 value:
    // the conversion really ran.
    assert_ne!(plain_img.get_pixel(0, 0).0, [200, 80, 40]);
}

#[test]
fn a_cmyk_jpeg_converts_sanely() {
    // Saturated red in CMYK (Adobe storage), encoded
    // with the same encoder family the engine uses.
    let mut data = Vec::new();
    for _ in 0..64 {
        data.extend_from_slice(&[0u8, 255, 255, 0]);
    }
    let mut jpg = Vec::new();
    jpeg_encoder::Encoder::new(&mut jpg, 95)
        .encode(&data, 8, 8, jpeg_encoder::ColorType::Cmyk)
        .unwrap();
    let engine = Engine;
    let out = engine
        .process(
            &jpg,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Png,
            ))])],
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    let px = img.get_pixel(4, 4).0;
    assert!(
        px[0] > 150 && px[1] < 100 && px[2] < 100,
        "CMYK red came out as {px:?}"
    );
}

/// A JPEG with EXIF (orientation 6) and XMP, built with
/// the production injector over an engine-encoded photo.
fn meta_jpeg() -> Vec<u8> {
    let mut base = Vec::new();
    image::load_from_memory(&gradient_png())
        .unwrap()
        .write_to(
            &mut std::io::Cursor::new(&mut base),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
    let mut tiff = Vec::new();
    tiff.extend_from_slice(b"II");
    tiff.extend_from_slice(&42u16.to_le_bytes());
    tiff.extend_from_slice(&8u32.to_le_bytes());
    tiff.extend_from_slice(&1u16.to_le_bytes());
    tiff.extend_from_slice(&0x0112u16.to_le_bytes());
    tiff.extend_from_slice(&3u16.to_le_bytes());
    tiff.extend_from_slice(&1u32.to_le_bytes());
    tiff.extend_from_slice(&6u16.to_le_bytes());
    tiff.extend_from_slice(&0u16.to_le_bytes());
    tiff.extend_from_slice(&0u32.to_le_bytes());
    let meta = crate::color::SourceMeta {
        icc: None,
        exif_tiff: Some(tiff),
        xmp: Some(b"<x:xmpmeta>fixture</x:xmpmeta>".to_vec()),
        xmp_compressed: false,
        photoshop: None,
    };
    crate::color::inject_jpeg(&base, None, &meta, true)
}

#[test]
fn metadata_is_stripped_by_default_and_preserved_on_policy() {
    let engine = Engine;
    let src = meta_jpeg();
    // Default: gone.
    let out = engine.process(&src, &[]).unwrap();
    let meta = crate::color::extract_source_meta(&out.bytes, true);
    assert!(meta.exif_tiff.is_none() && meta.xmp.is_none());
    // Preserve: both copied, orientation patched to 1
    // (the fixture's 6 already rotated the pixels).
    let out = engine
        .process_preferred(
            &src,
            &[],
            ProcessContext {
                preserve_metadata: true,
                ..Default::default()
            },
        )
        .unwrap();
    let meta = crate::color::extract_source_meta(&out.bytes, true);
    assert_eq!(
        meta.xmp.as_deref(),
        Some(b"<x:xmpmeta>fixture</x:xmpmeta>".as_slice())
    );
    let tiff = meta.exif_tiff.expect("exif preserved");
    assert_eq!(
        u16::from_le_bytes(tiff[8 + 2 + 8..8 + 2 + 10].try_into().unwrap()),
        1
    );
}

/// A zlib stream that inflates to `len` zero bytes, at
/// about a thousandth of that size.
fn zlib_bomb(len: usize) -> Vec<u8> {
    use std::io::Write as _;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
    encoder.write_all(&vec![0u8; len]).unwrap();
    encoder.finish().unwrap()
}

/// `png` with `chunks` inserted right after its IHDR.
fn with_chunks_after_ihdr(png: &[u8], chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let mut out = png[..33].to_vec();
    for (kind, data) in chunks {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(*kind);
        out.extend_from_slice(data);
        let mut crc = flate2::Crc::new();
        crc.update(*kind);
        crc.update(data);
        out.extend_from_slice(&crc.sum().to_be_bytes());
    }
    out.extend_from_slice(&png[33..]);
    out
}

#[test]
fn png_metadata_bombs_render_like_the_plain_png() {
    // Issue #399: 40 iCCP and 40 XMP iTXt chunks, each
    // a ~16 KB stream that inflates one byte past its
    // cap, in a file well under the fetch cap. Every
    // chunk used to be inflated, on every request and
    // under either policy; only the first of each kind
    // is now (the XMP only when the policy keeps it),
    // both are refused past their caps, and the render
    // is byte-identical to the same PNG without them.
    let plain = solid_png(8, 8, [30, 120, 200]);
    let mut iccp = b"ICC Profile\0\0".to_vec();
    iccp.extend_from_slice(&zlib_bomb(crate::color::MAX_ICC_BYTES + 1));
    let mut itxt = b"XML:com.adobe.xmp\0\x01\0\0\0".to_vec();
    itxt.extend_from_slice(&zlib_bomb(crate::color::MAX_XMP_BYTES + 1));
    let mut chunks: Vec<(&[u8; 4], &[u8])> = Vec::new();
    for _ in 0..40 {
        chunks.push((b"iCCP", &iccp));
        chunks.push((b"iTXt", &itxt));
    }
    let bombed = with_chunks_after_ihdr(&plain, &chunks);
    assert!(bombed.len() < 1024 * 1024, "{} bytes", bombed.len());
    let meta = crate::color::extract_source_meta(&bombed, true);
    assert!(meta.icc.is_none() && meta.xmp.is_none());
    let engine = Engine;
    for preserve_metadata in [false, true] {
        let ctx = || ProcessContext {
            preserve_metadata,
            ..Default::default()
        };
        let want = engine.process_preferred(&plain, &[], ctx()).unwrap();
        let got = engine.process_preferred(&bombed, &[], ctx()).unwrap();
        assert_eq!(got.bytes, want.bytes, "preserve {preserve_metadata}");
    }
}

/// An `iCCP` chunk payload carrying `profile`, compressed.
fn iccp_chunk(profile: &[u8]) -> Vec<u8> {
    use std::io::Write as _;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
    encoder.write_all(profile).unwrap();
    let mut data = b"ICC Profile\0\0".to_vec();
    data.extend_from_slice(&encoder.finish().unwrap());
    data
}

/// `len` zero bytes behind an ICC header that describes
/// them: the size field and the `acsp` signature in place,
/// no tags. Passes the embed check up to the embed cap.
fn padded_profile(len: usize) -> Vec<u8> {
    let mut icc = vec![0u8; len];
    icc[0..4].copy_from_slice(&(len as u32).to_be_bytes());
    icc[36..40].copy_from_slice(b"acsp");
    icc
}

/// `32x32/filters:[keep_profile:]format(<format>)`.
fn small_render(format: ImageFormat, keep_profile: bool) -> Vec<Operation> {
    let mut filters = vec![Filter::Format(FormatSpec::plain(format))];
    if keep_profile {
        filters.insert(0, Filter::KeepProfile);
    }
    vec![
        size_op(Dimension::Pixels(32), Dimension::Pixels(32)),
        Operation::Filters(filters),
    ]
}

#[test]
fn compressed_png_metadata_does_not_inflate_the_output() {
    // Issue #445: metadata inflated from compressed PNG
    // chunks was written back raw. Under preserve, a
    // 64x64 PNG carrying a 16 MiB XMP packet in a 16 KB
    // stream rendered to a 16 MiB PNG at 32x32; under the
    // default policy, keep_profile with JPEG output turned
    // a 4 MiB iCCP of zeros into a 4 MiB JPEG. Past their
    // caps, neither is copied now, into either format: the
    // render is the one the source without the chunk gets.
    let plain = gradient_png();
    let xmp_chunk = |len: usize| {
        let mut data = b"XML:com.adobe.xmp\0\x01\0\0\0".to_vec();
        data.extend_from_slice(&zlib_bomb(len));
        data
    };
    let xmp_issue = xmp_chunk(crate::color::MAX_META_BYTES);
    let iccp_zeros = iccp_chunk(&vec![0u8; crate::color::MAX_ICC_BYTES]);
    let iccp_headed = iccp_chunk(&padded_profile(crate::color::MAX_ICC_BYTES));
    let with = |kind: &[u8; 4], chunk: &[u8]| with_chunks_after_ihdr(&plain, &[(kind, chunk)]);
    let engine = Engine;
    let render = |src: &[u8], format, preserve_metadata, keep_profile| {
        let ctx = ProcessContext {
            preserve_metadata,
            ..Default::default()
        };
        let out = engine
            .process_preferred(src, &small_render(format, keep_profile), ctx)
            .unwrap();
        assert_eq!(out.format, format);
        out.bytes
    };
    let cases = [
        ("16 MiB XMP", with(b"iTXt", &xmp_issue), true, false),
        (
            "16 MiB XMP, keep_profile",
            with(b"iTXt", &xmp_issue),
            true,
            true,
        ),
        (
            "4 MiB iCCP of zeros",
            with(b"iCCP", &iccp_zeros),
            false,
            true,
        ),
        (
            "4 MiB iCCP, preserve",
            with(b"iCCP", &iccp_zeros),
            true,
            true,
        ),
        (
            "4 MiB iCCP behind a header",
            with(b"iCCP", &iccp_headed),
            false,
            true,
        ),
    ];
    for (name, bombed, preserve, keep) in cases {
        assert!(bombed.len() < 32 * 1024, "{name}: {} bytes", bombed.len());
        for format in [ImageFormat::Png, ImageFormat::Jpeg] {
            let want = render(&plain, format, preserve, keep);
            let got = render(&bombed, format, preserve, keep);
            assert!(
                got == want,
                "{name} as {format:?}: {} bytes against {} without the chunk",
                got.len(),
                want.len()
            );
        }
    }
    // A packet at its cap is copied into PNG, compressed as
    // the source stored it (a JPEG segment holds 64 KiB, so
    // not into JPEG): the render grows by no more than the
    // chunk weighed in the source.
    let bombed = with(b"iTXt", &xmp_chunk(crate::color::MAX_XMP_BYTES));
    let carried = bombed.len() - plain.len();
    let want = render(&plain, ImageFormat::Png, true, false);
    let got = render(&bombed, ImageFormat::Png, true, false);
    assert!(
        got.len() <= want.len() + carried,
        "{} bytes against {} plus the {carried}-byte chunk",
        got.len(),
        want.len()
    );
    let xmp = crate::color::extract_source_meta(&got, true).xmp;
    assert_eq!(xmp.map(|xmp| xmp.len()), Some(crate::color::MAX_XMP_BYTES));
    let want = render(&plain, ImageFormat::Jpeg, true, false);
    assert!(render(&bombed, ImageFormat::Jpeg, true, false) == want);
    // The ceiling on what a profile can still add: one at
    // the embed cap is kept and goes into a JPEG raw, in
    // APP2 segments of at most 65519 bytes plus 18 bytes
    // of marker, length, and chunk header.
    let at_cap = padded_profile(crate::color::MAX_EMBED_ICC_BYTES);
    let bombed = with(b"iCCP", &iccp_chunk(&at_cap));
    let want = render(&plain, ImageFormat::Jpeg, false, true);
    let got = render(&bombed, ImageFormat::Jpeg, false, true);
    let segments = crate::color::MAX_EMBED_ICC_BYTES.div_ceil(65519);
    assert_eq!(
        got.len(),
        want.len() + crate::color::MAX_EMBED_ICC_BYTES + 18 * segments
    );
    assert_eq!(
        crate::color::extract_source_meta(&got, false).icc,
        Some(at_cap)
    );
}

#[test]
fn keep_profile_converts_a_profile_it_cannot_embed() {
    // Issue #445: keep_profile is convert-or-tag (issue
    // #374), and a profile that fails the embed check is
    // not tagged, so its pixels must be converted like
    // the default's. Two Display P3 profiles that still
    // parse: one with a byte past its declared size, one
    // grown past the embed cap with its size field
    // rewritten to match.
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(3, 1, |x, _| {
        [
            image::Rgb([200, 80, 40]),
            image::Rgb([40, 120, 200]),
            image::Rgb([90, 160, 90]),
        ][x as usize]
    }));
    let p3 = moxcms::ColorProfile::new_display_p3().encode().unwrap();
    let mut trailing = p3.clone();
    trailing.push(0);
    let mut oversized = p3.clone();
    oversized.resize(crate::color::MAX_EMBED_ICC_BYTES + 4, 0);
    let len = oversized.len() as u32;
    oversized[0..4].copy_from_slice(&len.to_be_bytes());
    let engine = Engine;
    for (name, icc) in [("trailing byte", trailing), ("past the cap", oversized)] {
        assert!(!crate::color::icc_embeddable(&icc), "{name}");
        assert!(moxcms::ColorProfile::new_from_slice(&icc).is_ok(), "{name}");
        let src = with_chunks_after_ihdr(&png_bytes(&img), &[(b"iCCP", &iccp_chunk(&icc))]);
        for format in [ImageFormat::Png, ImageFormat::Jpeg] {
            let ops = |keep: bool| {
                let mut filters = vec![Filter::Format(FormatSpec::plain(format))];
                if keep {
                    filters.insert(0, Filter::KeepProfile);
                }
                [Operation::Filters(filters)]
            };
            let kept = engine.process(&src, &ops(true)).unwrap();
            let converted = engine.process(&src, &ops(false)).unwrap();
            assert_eq!(kept.bytes, converted.bytes, "{name} as {format:?}");
            assert!(
                crate::color::extract_source_meta(&kept.bytes, false)
                    .icc
                    .is_none(),
                "{name} as {format:?}"
            );
            // And the conversion really ran: the lossless
            // pixel is not the raw P3 value.
            if format == ImageFormat::Png {
                let px = image::load_from_memory(&kept.bytes).unwrap().to_rgb8();
                assert_ne!(px.get_pixel(0, 0).0, [200, 80, 40], "{name}");
            }
        }
    }
}

#[test]
fn a_passed_processing_deadline_aborts_the_request() {
    // Issue #260: with the deadline already gone, the
    // pipeline refuses before doing the work; with
    // time on the clock the same request runs.
    let engine = Engine;
    let src = gradient_png();
    let past = ProcessContext {
        deadline: Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
        ..Default::default()
    };
    let err = engine
        .process_preferred(&src, &[], past)
        .expect_err("a passed deadline aborts");
    assert_eq!(err, EngineError::Timeout);
    let future = ProcessContext {
        deadline: Some(std::time::Instant::now() + std::time::Duration::from_secs(60)),
        ..Default::default()
    };
    engine.process_preferred(&src, &[], future).unwrap();
    // No deadline at all stays unbounded (tests and
    // library callers rely on it).
    engine.process(&src, &[]).unwrap();
}

#[test]
fn save_data_steps_the_default_quality_and_reports_it() {
    let engine = Engine;
    let ctx = ProcessContext {
        save_data: true,
        ..Default::default()
    };
    // A JPEG with no quality directive encodes at the
    // stepped quality and reports it.
    let out = engine
        .process_preferred(
            &gradient_png(),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Jpeg,
            ))])],
            ctx,
        )
        .unwrap();
    assert_eq!(out.quality, Some(SAVE_DATA_QUALITY));
    // An explicit quality stands: nothing is reported,
    // exactly as without the hint.
    let out = engine
        .process_preferred(
            &gradient_png(),
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
                Filter::Quality(Quality::Value(90)),
            ])],
            ProcessContext {
                save_data: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.quality, None);
    // PNG has no quality to step.
    let out = engine
        .process_preferred(
            &gradient_png(),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Png,
            ))])],
            ProcessContext {
                save_data: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.quality, None);
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
    assert_eq!(out.ssimulacra2, None);
}

#[test]
fn auto_quality_jpeg_is_verified_against_the_target() {
    let engine = Engine;
    let ops = &[Operation::Filters(vec![
        Filter::Quality(Quality::Auto),
        Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
    ])];
    // The photo surrogate is ordinary content for the metric:
    // the loop must land it at or above the target.
    let out = engine.process(&photo_like_png(), ops).unwrap();
    let score = out.ssimulacra2.expect("auto JPEG is scored");
    assert!(
        (crate::quality::DEFAULT_SSIM_TARGET..=100.0).contains(&score),
        "score {score}"
    );
    assert!(out.quality.is_some_and(|q| (50..=95).contains(&q)));
    // The noisy fixture cannot reach the target at any allowed
    // quality; the loop still scores it and serves its best.
    let noisy = engine.process(&noise_png(), ops).unwrap();
    assert!(noisy.ssimulacra2.is_some());
}

#[test]
fn auto_quality_jxl_is_verified_too() {
    let engine = Engine;
    let out = engine
        .process(
            &photo_like_png(),
            &[Operation::Filters(vec![
                Filter::Quality(Quality::Auto),
                Filter::Format(FormatSpec::plain(ImageFormat::Jxl)),
            ])],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Jxl);
    let score = out.ssimulacra2.expect("auto JXL is scored");
    assert!(
        (crate::quality::DEFAULT_SSIM_TARGET..=100.0).contains(&score),
        "score {score}"
    );
}

#[test]
fn auto_quality_webp_keeps_the_one_shot_pick() {
    let engine = Engine;
    let out = engine
        .process(
            &gradient_png(),
            &[Operation::Filters(vec![
                Filter::Quality(Quality::Auto),
                Filter::Format(FormatSpec::plain(ImageFormat::WebP)),
            ])],
        )
        .unwrap();
    // WebP is out of the verified scope for now: the heuristic
    // quality is reported, and nothing was scored.
    assert!(out.quality.is_some());
    assert_eq!(out.ssimulacra2, None);
}

/// 8x4 pattern with only the top-left quadrant white.
fn quadrant_pattern() -> DynamicImage {
    let mut img = DynamicImage::new_rgb8(8, 4);
    for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        let v = if x < 4 && y < 2 { 255 } else { 0 };
        *p = image::Rgb([v, v, v]);
    }
    img
}

/// Encode as JPEG and splice in an EXIF APP1 segment carrying
/// `orientation`, the way a phone camera writes it.
fn jpeg_with_orientation(img: &DynamicImage, orientation: u16) -> Vec<u8> {
    let mut buf = std::io::Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 100)
        .write_image(
            img.as_bytes(),
            img.width(),
            img.height(),
            image::ExtendedColorType::from(img.color()),
        )
        .unwrap();
    let jpeg = buf.into_inner();
    assert_eq!(&jpeg[..2], b"\xff\xd8", "JPEG starts with SOI");

    let mut tiff = Vec::new();
    tiff.extend_from_slice(b"II"); // little-endian TIFF
    tiff.extend_from_slice(&42u16.to_le_bytes());
    tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 offset
    tiff.extend_from_slice(&1u16.to_le_bytes()); // one entry
    tiff.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
    tiff.extend_from_slice(&3u16.to_le_bytes()); // SHORT
    tiff.extend_from_slice(&1u32.to_le_bytes()); // count
    tiff.extend_from_slice(&orientation.to_le_bytes());
    tiff.extend_from_slice(&[0, 0]); // value padding
    tiff.extend_from_slice(&0u32.to_le_bytes()); // next IFD

    let mut app1 = Vec::new();
    app1.extend_from_slice(b"Exif\0\0");
    app1.extend_from_slice(&tiff);

    let mut out = Vec::new();
    out.extend_from_slice(&jpeg[..2]);
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&app1);
    out.extend_from_slice(&jpeg[2..]);
    out
}

/// (width, height, [TL, TR, BL, BR] brightness) of an image.
fn shape_and_quadrants(bytes: &[u8]) -> (u32, u32, [bool; 4]) {
    let img = image::load_from_memory(bytes).unwrap().to_luma8();
    let (w, h) = (img.width(), img.height());
    let bright = |x: u32, y: u32| img.get_pixel(x, y)[0] > 128;
    (
        w,
        h,
        [
            bright(w / 4, h / 4),
            bright(3 * w / 4, h / 4),
            bright(w / 4, 3 * h / 4),
            bright(3 * w / 4, 3 * h / 4),
        ],
    )
}

#[test]
fn exif_orientations_come_out_upright() {
    let engine = Engine;
    // The stored pattern has its white quadrant top-left. Each
    // orientation value states where the stored pixels must move
    // for the image to display upright.
    type OrientationCase = (u16, (u32, u32, [bool; 4]));
    let cases: [OrientationCase; 8] = [
        (1, (8, 4, [true, false, false, false])),
        (2, (8, 4, [false, true, false, false])),
        (3, (8, 4, [false, false, false, true])),
        (4, (8, 4, [false, false, true, false])),
        (5, (4, 8, [true, false, false, false])),
        (6, (4, 8, [false, true, false, false])),
        (7, (4, 8, [false, false, false, true])),
        (8, (4, 8, [false, false, true, false])),
    ];
    for (orientation, expected) in cases {
        let src = jpeg_with_orientation(&quadrant_pattern(), orientation);
        let out = engine.process(&src, &[]).unwrap();
        assert_eq!(
            shape_and_quadrants(&out.bytes),
            expected,
            "orientation {orientation}"
        );
    }
}

#[test]
fn jpeg_without_exif_is_untouched() {
    let engine = Engine;
    let img = quadrant_pattern();
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
    let out = engine.process(&buf.into_inner(), &[]).unwrap();
    assert_eq!(
        shape_and_quadrants(&out.bytes),
        (8, 4, [true, false, false, false])
    );
}

#[test]
fn invalid_exif_orientation_is_ignored() {
    let engine = Engine;
    let src = jpeg_with_orientation(&quadrant_pattern(), 42);
    let out = engine.process(&src, &[]).unwrap();
    assert_eq!(
        shape_and_quadrants(&out.bytes),
        (8, 4, [true, false, false, false])
    );
}

#[test]
fn avif_format_encodes() {
    let engine = Engine;
    let out = engine
        .process(
            &gradient_png(),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Avif,
            ))])],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Avif);
    // ISOBMFF layout: box size, then "ftyp", then the brand.
    assert_eq!(&out.bytes[4..8], b"ftyp");
    assert_eq!(&out.bytes[8..12], b"avif");
}

#[test]
fn avif_sources_decode_or_fail_clearly() {
    // Encode a small AVIF with the always-available
    // encoder, then run it through the bounded decode.
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        24,
        16,
        image::Rgb([200, 40, 90]),
    ));
    let mut bytes = Vec::new();
    image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut bytes, 10, 100)
        .write_image(img.as_bytes(), 24, 16, image::ExtendedColorType::Rgb8)
        .unwrap();
    assert_eq!(&bytes[8..12], b"avif");
    #[cfg(feature = "avif-decode")]
    {
        let decoded = decode_image(&bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (24, 16));
        let px = decoded.to_rgb8().get_pixel(12, 8).0;
        assert!(
            px[0] > 150 && px[1] < 110 && px[2] > 50 && px[2] < 140,
            "decoded pixel {px:?} drifted from the source color"
        );
    }
    #[cfg(not(feature = "avif-decode"))]
    {
        let err = decode_image(&bytes).unwrap_err();
        let EngineError::Decode(message) = err else {
            panic!("expected a decode error, got {err:?}");
        };
        assert!(message.contains("avif-decode"), "{message}");
    }
}

#[test]
fn avif_quality_changes_size() {
    let engine = Engine;
    let mk = |q| {
        engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![
                    Filter::Quality(Quality::Value(q)),
                    Filter::Format(FormatSpec::plain(ImageFormat::Avif)),
                ])],
            )
            .unwrap()
    };
    assert!(mk(95).bytes.len() > mk(10).bytes.len());
}

#[test]
fn preferred_format_applies_only_without_explicit_format() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &gradient_png(),
            &[],
            ProcessContext {
                preferred: Some(ImageFormat::WebP),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::WebP);
    let out = engine
        .process_preferred(
            &gradient_png(),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Png,
            ))])],
            ProcessContext {
                preferred: Some(ImageFormat::WebP),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Png);
}

fn fit_in_op(w: Dimension, h: Dimension) -> Operation {
    Operation::FitIn {
        size: ImageSize {
            width: w,
            height: h,
            flip_horizontal: false,
            flip_vertical: false,
        },
        adaptive: false,
    }
}

fn fit_in_adaptive_op(w: Dimension, h: Dimension) -> Operation {
    Operation::FitIn {
        size: ImageSize {
            width: w,
            height: h,
            flip_horizontal: false,
            flip_vertical: false,
        },
        adaptive: true,
    }
}

fn dims_of(out: &ProcessedImage) -> (u32, u32) {
    let img = image::load_from_memory(&out.bytes).unwrap();
    (img.width(), img.height())
}

#[test]
fn fit_in_fits_the_box_and_keeps_aspect() {
    let engine = Engine;
    let src = png_bytes(&DynamicImage::new_rgb8(400, 200));
    // Both bounds: the tighter one rules.
    let out = engine
        .process(
            &src,
            &[fit_in_op(Dimension::Pixels(100), Dimension::Pixels(100))],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (100, 50));
    // Single-axis bound.
    let out = engine
        .process(
            &src,
            &[fit_in_op(Dimension::Pixels(300), Dimension::Proportional)],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (300, 150));
    // Upscaling is allowed without no_upscale.
    let out = engine
        .process(
            &src,
            &[fit_in_op(Dimension::Pixels(1000), Dimension::Pixels(1000))],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (1000, 500));
}

#[test]
fn fit_in_adaptive_swaps_the_box_when_orientations_disagree() {
    let engine = Engine;
    // Portrait 200x400 into a landscape 300x100 box: plain
    // fit-in is ruled by the height (50x100); adaptive fits
    // the swapped 100x300 box instead (100x200).
    let portrait = png_bytes(&DynamicImage::new_rgb8(200, 400));
    let out = engine
        .process(
            &portrait,
            &[fit_in_op(Dimension::Pixels(300), Dimension::Pixels(100))],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (50, 100));
    let out = engine
        .process(
            &portrait,
            &[fit_in_adaptive_op(
                Dimension::Pixels(300),
                Dimension::Pixels(100),
            )],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (100, 200));
    // Landscape 400x200 into a portrait 100x300 box: plain
    // is ruled by the width (100x50); adaptive fits the
    // swapped 300x100 box instead (200x100).
    let landscape = png_bytes(&DynamicImage::new_rgb8(400, 200));
    let out = engine
        .process(
            &landscape,
            &[fit_in_op(Dimension::Pixels(100), Dimension::Pixels(300))],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (100, 50));
    let out = engine
        .process(
            &landscape,
            &[fit_in_adaptive_op(
                Dimension::Pixels(100),
                Dimension::Pixels(300),
            )],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (200, 100));
}

#[test]
fn fit_in_adaptive_matches_plain_fit_in_when_orientations_agree() {
    let engine = Engine;
    // Landscape source, landscape box: no swap, byte-for-byte
    // the plain behavior.
    let landscape = png_bytes(&DynamicImage::new_rgb8(400, 200));
    let plain = engine
        .process(
            &landscape,
            &[fit_in_op(Dimension::Pixels(300), Dimension::Pixels(100))],
        )
        .unwrap();
    let adaptive = engine
        .process(
            &landscape,
            &[fit_in_adaptive_op(
                Dimension::Pixels(300),
                Dimension::Pixels(100),
            )],
        )
        .unwrap();
    assert_eq!(dims_of(&adaptive), (200, 100));
    assert_eq!(adaptive.bytes, plain.bytes);
    // Portrait source, portrait box: no swap either.
    let portrait = png_bytes(&DynamicImage::new_rgb8(200, 400));
    let plain = engine
        .process(
            &portrait,
            &[fit_in_op(Dimension::Pixels(100), Dimension::Pixels(300))],
        )
        .unwrap();
    let adaptive = engine
        .process(
            &portrait,
            &[fit_in_adaptive_op(
                Dimension::Pixels(100),
                Dimension::Pixels(300),
            )],
        )
        .unwrap();
    assert_eq!(dims_of(&adaptive), (100, 200));
    assert_eq!(adaptive.bytes, plain.bytes);
}

#[test]
fn fit_in_adaptive_never_swaps_square_or_open_boxes() {
    let engine = Engine;
    // Square source: no orientation, so the output is the
    // plain fit-in output.
    let square = png_bytes(&DynamicImage::new_rgb8(200, 200));
    let plain = engine
        .process(
            &square,
            &[fit_in_op(Dimension::Pixels(300), Dimension::Pixels(100))],
        )
        .unwrap();
    let adaptive = engine
        .process(
            &square,
            &[fit_in_adaptive_op(
                Dimension::Pixels(300),
                Dimension::Pixels(100),
            )],
        )
        .unwrap();
    assert_eq!(dims_of(&adaptive), (100, 100));
    assert_eq!(adaptive.bytes, plain.bytes);
    // Square box: swapping it would change nothing, and the
    // code does not swap it.
    let portrait = png_bytes(&DynamicImage::new_rgb8(200, 400));
    let plain = engine
        .process(
            &portrait,
            &[fit_in_op(Dimension::Pixels(100), Dimension::Pixels(100))],
        )
        .unwrap();
    let adaptive = engine
        .process(
            &portrait,
            &[fit_in_adaptive_op(
                Dimension::Pixels(100),
                Dimension::Pixels(100),
            )],
        )
        .unwrap();
    assert_eq!(dims_of(&adaptive), (50, 100));
    assert_eq!(adaptive.bytes, plain.bytes);
    // A box with a proportional axis has no fixed
    // orientation: no swap, the single bound rules.
    let plain = engine
        .process(
            &portrait,
            &[fit_in_op(Dimension::Pixels(300), Dimension::Proportional)],
        )
        .unwrap();
    let adaptive = engine
        .process(
            &portrait,
            &[fit_in_adaptive_op(
                Dimension::Pixels(300),
                Dimension::Proportional,
            )],
        )
        .unwrap();
    assert_eq!(dims_of(&adaptive), (300, 600));
    assert_eq!(adaptive.bytes, plain.bytes);
}

#[test]
fn fit_in_box_swap_decision() {
    let px = |w: u32, h: u32| ImageSize {
        width: Dimension::Pixels(w),
        height: Dimension::Pixels(h),
        flip_horizontal: false,
        flip_vertical: false,
    };
    // Not adaptive: never swaps, whatever the orientations.
    assert_eq!(fit_in_box(&px(300, 100), 200, 400, false), px(300, 100));
    // Disagreement swaps in both directions.
    assert_eq!(fit_in_box(&px(300, 100), 200, 400, true), px(100, 300));
    assert_eq!(fit_in_box(&px(100, 300), 400, 200, true), px(300, 100));
    // Agreement keeps the box.
    assert_eq!(fit_in_box(&px(300, 100), 400, 200, true), px(300, 100));
    assert_eq!(fit_in_box(&px(100, 300), 200, 400, true), px(100, 300));
    // Equality on either side never swaps.
    assert_eq!(fit_in_box(&px(300, 100), 200, 200, true), px(300, 100));
    assert_eq!(fit_in_box(&px(100, 100), 200, 400, true), px(100, 100));
    // No two pixel axes, no swap.
    let open = ImageSize {
        width: Dimension::Pixels(300),
        height: Dimension::Proportional,
        flip_horizontal: false,
        flip_vertical: false,
    };
    assert_eq!(fit_in_box(&open, 200, 400, true), open);
    let orig = ImageSize {
        width: Dimension::Original,
        height: Dimension::Pixels(100),
        flip_horizontal: false,
        flip_vertical: false,
    };
    assert_eq!(fit_in_box(&orig, 200, 400, true), orig);
}

/// A size side written as a multi-digit zero (`00x100`) parses
/// as an exact zero: the spec vectors pin the spelling, so the
/// parser keeps it. The engine refuses it with
/// `unsupported_operation` (issue #464) on every route a size
/// reaches it by: before, the resize handed the encoder a
/// 0-wide image (a 500), and DPR scaling or fit-in rounded the
/// zero up to a 1-pixel render nobody asked for.
#[test]
fn a_zero_pixel_side_is_refused_on_every_route() {
    let engine = Engine;
    let still = png_bytes(&gradient_image());
    let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" width="48" height="32"><rect width="48" height="32" fill="red"/></svg>"#;
    let sources: [(&str, &[u8]); 3] = [
        ("png", &still),
        ("animated gif", &animated_gif_bytes()),
        ("svg", svg),
    ];
    let refused = [
        "00x100",
        "100x00",
        "00x00",
        "00xorig",
        "origx00",
        "-00x-100",
        "000000x100",
        "00x100/filters:dpr(2)",
        "100x00/filters:dpr(3)",
        "fit-in:00x100",
        "fit-in:100x00",
        "fit-in:00x00",
        "fit-in:00x100:adaptive",
        "fit-in:00x100/filters:dpr(2)",
        // A later valid size does not rescue the request.
        "00x100/50x50",
    ];
    for (kind, src) in sources {
        // The crop+resize fusion resolves the size against
        // the window, through the same check. (An SVG source
        // rasterizes at the first size's target, too small
        // for the rectangle, which is its own refusal.)
        let fused = (kind != "svg").then_some("crop-rect:0x0:24x16/00x10");
        for remainder in refused.into_iter().chain(fused) {
            let (ops, _) =
                crate::url::parse_operations(&format!("{remainder}/example.com/a.png")).unwrap();
            let err = engine
                .process(src, &ops)
                .expect_err(&format!("{kind} {remainder}"));
            assert!(
                matches!(err, EngineError::Unsupported(_)),
                "{kind} {remainder}: {err:?}"
            );
            assert!(
                err.to_string().contains("zero pixel side"),
                "{kind} {remainder}: {err}"
            );
        }
    }
    // A lone 0 keeps its meaning: proportional for a resize,
    // unbounded for fit-in.
    for (remainder, dims) in [
        ("0x16", (24, 16)),
        ("24x0", (24, 16)),
        ("0x0", (48, 32)),
        ("fit-in:0x16", (24, 16)),
        ("fit-in:24x0", (24, 16)),
    ] {
        let (ops, _) =
            crate::url::parse_operations(&format!("{remainder}/example.com/a.png")).unwrap();
        let out = engine.process(&still, &ops).unwrap();
        assert_eq!(decode_dims(&out.bytes), dims, "{remainder}");
    }
}

#[test]
fn crop_rect_extracts_the_exact_region() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    let out = engine
        .process(
            &src,
            &[Operation::CropRect {
                left: 0,
                top: 0,
                right: 4,
                bottom: 2,
            }],
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert_eq!((img.width(), img.height()), (4, 2));
    assert!(img.pixels().all(|p| p[0] == 255));
    let out = engine
        .process(
            &src,
            &[Operation::CropRect {
                left: 4,
                top: 2,
                right: 8,
                bottom: 4,
            }],
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(img.pixels().all(|p| p[0] == 0));
}

#[test]
fn crop_rect_out_of_bounds_is_an_error() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    let err = engine
        .process(
            &src,
            &[Operation::CropRect {
                left: 0,
                top: 0,
                right: 9,
                bottom: 4,
            }],
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
}

fn rect_op(left: u32, top: u32, right: u32, bottom: u32) -> Operation {
    Operation::CropRect {
        left,
        top,
        right,
        bottom,
    }
}

#[test]
fn fusion_plan_accepts_the_canonical_chain() {
    // Issue #287: one crop-rect feeding one resize, and
    // nothing else, fuses; the plan carries the window.
    let ops = [
        rect_op(600, 400, 1800, 1200),
        size_op(Dimension::Pixels(800), Dimension::Pixels(533)),
    ];
    let plan = fusion_plan(&ops).expect("the canonical chain fuses");
    assert_eq!(plan.crop_index, 0);
    assert_eq!(plan.size_index, 1);
    assert_eq!(plan.window, (600, 400, 1200, 800));
    // Work ahead of the crop runs identically on either
    // path, and decode-time no-ops may sit between the
    // crop and the resize and after it.
    let ops = [
        Operation::Filters(vec![Filter::Grayscale]),
        rect_op(600, 400, 1800, 1200),
        Operation::Page(2),
        size_op(Dimension::Pixels(800), Dimension::Pixels(533)),
        Operation::Frame(crate::url::FrameSelection::Poster),
    ];
    assert!(fusion_plan(&ops).is_some());
    // A focal rectangle ahead of the crop validates
    // against the pre-crop frame on both paths.
    let ops = [
        focal_op(0, 0, 100, 100),
        rect_op(600, 400, 1800, 1200),
        size_op(Dimension::Pixels(800), Dimension::Pixels(533)),
    ];
    assert!(fusion_plan(&ops).is_some());
}

#[test]
fn fusion_plan_g1_requires_one_crop_rect() {
    // An aspect crop's window is a run-time decision, not
    // a static rectangle.
    let ops = [
        crop_op(800, 533, Align::Center),
        size_op(Dimension::Pixels(800), Dimension::Pixels(533)),
    ];
    assert!(fusion_plan(&ops).is_none());
    // Two crop-rects: the second window is named in the
    // first window's coordinates, not the frame's.
    let ops = [
        rect_op(0, 0, 100, 100),
        rect_op(0, 0, 50, 50),
        size_op(Dimension::Pixels(80), Dimension::Pixels(80)),
    ];
    assert!(fusion_plan(&ops).is_none());
    // No crop at all.
    let ops = [size_op(Dimension::Pixels(800), Dimension::Pixels(533))];
    assert!(fusion_plan(&ops).is_none());
}

#[test]
fn fusion_plan_g2_requires_one_resize() {
    // No resize: nothing for the crop to fuse into.
    let ops = [rect_op(0, 0, 100, 100)];
    assert!(fusion_plan(&ops).is_none());
    // Two resizes: the plan would have to pick one.
    let ops = [
        rect_op(0, 0, 100, 100),
        size_op(Dimension::Pixels(80), Dimension::Pixels(80)),
        size_op(Dimension::Pixels(40), Dimension::Pixels(40)),
    ];
    assert!(fusion_plan(&ops).is_none());
}

#[test]
fn fusion_plan_g3_requires_the_crop_to_feed_the_resize() {
    // A pixel operation between the crop and the resize
    // runs on the window in the two-step path.
    let ops = [
        rect_op(0, 0, 100, 100),
        Operation::Filters(vec![Filter::Grayscale]),
        size_op(Dimension::Pixels(80), Dimension::Pixels(80)),
    ];
    assert!(fusion_plan(&ops).is_none());
    // A resize ahead of the crop resamples the frame, not
    // the window.
    let ops = [
        size_op(Dimension::Pixels(80), Dimension::Pixels(80)),
        rect_op(0, 0, 100, 100),
    ];
    assert!(fusion_plan(&ops).is_none());
}

#[test]
fn fusion_plan_g4_requires_the_resize_to_be_the_final_pixel_op() {
    let ops = [
        rect_op(0, 0, 100, 100),
        size_op(Dimension::Pixels(80), Dimension::Pixels(80)),
        Operation::Filters(vec![Filter::Grayscale]),
    ];
    assert!(fusion_plan(&ops).is_none());
}

#[test]
fn fusion_plan_g5_refuses_geometry_consumers_after_the_crop() {
    // A focal rectangle after the crop validates against
    // the frame in effect where it sits: the window in
    // the two-step path, the whole frame under fusion.
    let ops = [
        rect_op(0, 0, 100, 100),
        focal_op(0, 0, 50, 50),
        size_op(Dimension::Pixels(80), Dimension::Pixels(80)),
    ];
    assert!(fusion_plan(&ops).is_none());
    let ops = [
        rect_op(0, 0, 100, 100),
        size_op(Dimension::Pixels(80), Dimension::Pixels(80)),
        focal_op(0, 0, 50, 50),
    ];
    assert!(fusion_plan(&ops).is_none());
    // A pad after the crop builds its canvas from the
    // frame in effect where it sits.
    let ops = [
        rect_op(0, 0, 100, 100),
        Operation::Pad {
            size: ImageSize {
                width: Dimension::Pixels(200),
                height: Dimension::Pixels(200),
                flip_horizontal: false,
                flip_vertical: false,
            },
            fill: PadFill::Color([0, 0, 0]),
        },
        size_op(Dimension::Pixels(80), Dimension::Pixels(80)),
    ];
    assert!(fusion_plan(&ops).is_none());
}

#[test]
fn fusion_plan_g6_refuses_debug_chains_and_empty_rectangles() {
    // The debug view renders a diagnostic of the exact
    // pixels, so its chains stay on the two-step path.
    let ops = [
        Operation::DebugFocal,
        rect_op(0, 0, 100, 100),
        size_op(Dimension::Pixels(80), Dimension::Pixels(80)),
    ];
    assert!(fusion_plan(&ops).is_none());
    // Zero-area rectangles keep today's behavior, odd as
    // it may be, on the two-step path.
    let ops = [
        rect_op(10, 10, 10, 20),
        size_op(Dimension::Pixels(80), Dimension::Pixels(80)),
    ];
    assert!(fusion_plan(&ops).is_none());
    let ops = [
        rect_op(10, 10, 20, 10),
        size_op(Dimension::Pixels(80), Dimension::Pixels(80)),
    ];
    assert!(fusion_plan(&ops).is_none());
}

/// The pixel delta between two same-sized images, in
/// the wave-3 lane's measures: (percent of pixels
/// differing, max subpixel delta, mean subpixel delta).
fn pixel_delta(a: &DynamicImage, b: &DynamicImage) -> (f64, u32, f64) {
    let (a, b) = (a.to_rgb8(), b.to_rgb8());
    assert_eq!(a.dimensions(), b.dimensions());
    let mut diff_px = 0u64;
    let mut max = 0u32;
    let mut sum = 0u64;
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        if pa != pb {
            diff_px += 1;
        }
        for i in 0..3 {
            let d = pa.0[i].abs_diff(pb.0[i]) as u32;
            max = max.max(d);
            sum += d as u64;
        }
    }
    let total = (a.width() * a.height()) as u64;
    (
        diff_px as f64 * 100.0 / total as f64,
        max,
        sum as f64 / (total * 3) as f64,
    )
}

/// The sinusoid photo surrogate at any size (the
/// generator behind `photo_like_png`, unencoded).
fn photo_like_rgb(w: u32, h: u32) -> DynamicImage {
    let mut img = DynamicImage::new_rgb8(w, h);
    for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        let (fx, fy) = (x as f64, y as f64);
        let wave = |phase: f64, a: f64, b: f64| {
            128.0
                + 62.0 * (fx * a + phase).sin() * (fy * b + phase * 0.7).cos()
                + 24.0 * ((fx + fy) * 0.021 + phase).sin()
        };
        *p = image::Rgb([
            wave(0.0, 0.11, 0.07).clamp(0.0, 255.0) as u8,
            wave(2.1, 0.09, 0.13).clamp(0.0, 255.0) as u8,
            wave(4.2, 0.05, 0.05).clamp(0.0, 255.0) as u8,
        ]);
    }
    img
}

/// The wave-3 lane's probe photo (a gradient with
/// LCG noise), the fixture the accepted fusion
/// envelope was measured on, at the lane's
/// own shapes.
fn lane_photo(w: u32, h: u32) -> DynamicImage {
    let mut buf = vec![0u8; (w * h * 3) as usize];
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u8
    };
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            buf[i] = ((x * 255) / w.max(1)) as u8 ^ (next() >> 3);
            buf[i + 1] = ((y * 255) / h.max(1)) as u8 ^ (next() >> 3);
            buf[i + 2] = (((x + y) * 255) / (w + h).max(1)) as u8 ^ (next() >> 3);
        }
    }
    DynamicImage::ImageRgb8(image::RgbImage::from_raw(w, h, buf).unwrap())
}

fn run_chain(img: &DynamicImage, ops: &[Operation]) -> DynamicImage {
    let directives = crate::url::RequestDirectives::analyze(ops);
    let ctx = ProcessContext::default();
    apply_operations(img.clone(), ops, None, 1.0, &ctx, &directives, None)
        .unwrap()
        .0
}

#[test]
fn fused_crop_resize_downscale_stays_inside_the_accepted_delta() {
    // Issue #287: the fused output differs from the
    // two-step chain by rounding-class deltas only. On
    // this fixture at this shape the wave-3 lane
    // measured the envelope the change was accepted
    // with: 1.0 percent of pixels differ, max delta 4,
    // mean 0.007 (reproduced exactly by this test's
    // inputs). The guards keep the delta in that class
    // with headroom for the rounding jitter between the
    // backend's CPU kernel tiers; a wrong window or
    // geometry would land orders of magnitude outside
    // them.
    let src = lane_photo(2400, 1600);
    let ops = [
        rect_op(600, 400, 1800, 1200),
        size_op(Dimension::Pixels(800), Dimension::Pixels(533)),
    ];
    let fused = run_chain(&src, &ops);
    let two_step = resize_lanczos3(&src.crop_imm(600, 400, 1200, 800), 800, 533);
    assert_eq!(fused.dimensions(), (800, 533));
    let (pct, max, mean) = pixel_delta(&fused, &two_step);
    assert!(pct > 0.0, "the fused path must actually engage");
    assert!(pct <= 3.5, "{pct}% of pixels differ");
    assert!(max <= 6, "max delta {max}");
    assert!(mean <= 0.02, "mean delta {mean}");
}

#[test]
fn fused_crop_resize_upscale_stays_inside_the_accepted_delta() {
    // The upscale shape from the same probe: 3.5
    // percent of pixels differ, max delta 10, mean
    // 0.027 measured; the guards carry the same
    // headroom as the downscale's.
    let src = lane_photo(2400, 1600);
    let ops = [
        rect_op(100, 100, 700, 500),
        size_op(Dimension::Pixels(1200), Dimension::Pixels(800)),
    ];
    let fused = run_chain(&src, &ops);
    let two_step = resize_lanczos3(&src.crop_imm(100, 100, 600, 400), 1200, 800);
    assert_eq!(fused.dimensions(), (1200, 800));
    let (pct, max, mean) = pixel_delta(&fused, &two_step);
    assert!(pct > 0.0, "the fused path must actually engage");
    assert!(pct <= 5.0, "{pct}% of pixels differ");
    assert!(max <= 14, "max delta {max}");
    assert!(mean <= 0.04, "mean delta {mean}");
}

#[test]
fn fused_chain_out_of_bounds_crop_rect_errors_like_the_two_step_path() {
    // The fused crop validates before it defers: the
    // error is the two-step path's own.
    let engine = Engine;
    let src = solid_png(100, 60, [10, 20, 30]);
    let plain = engine.process(&src, &[rect_op(0, 0, 101, 60)]).unwrap_err();
    let fused = engine
        .process(
            &src,
            &[
                rect_op(0, 0, 101, 60),
                size_op(Dimension::Pixels(50), Dimension::Pixels(30)),
            ],
        )
        .unwrap_err();
    assert_eq!(fused.to_string(), plain.to_string());
}

#[test]
fn fused_chain_without_a_resample_returns_the_window_itself() {
    // When the resize target is the window's own size no
    // resample runs on either path, and the fused path
    // materializes the window exactly.
    let src = photo_like_rgb(120, 80);
    let ops = [
        rect_op(10, 10, 50, 40),
        size_op(Dimension::Pixels(40), Dimension::Pixels(30)),
    ];
    let out = run_chain(&src, &ops);
    let window = src.crop_imm(10, 10, 40, 30);
    assert_eq!(out.to_rgb8().as_raw(), window.to_rgb8().as_raw());
}

fn focal_op(left: u32, top: u32, right: u32, bottom: u32) -> Operation {
    Operation::FocalRect {
        left,
        top,
        right,
        bottom,
    }
}

fn square_crop(align: Align) -> Operation {
    Operation::Crop {
        size: ImageSize {
            width: Dimension::Pixels(2),
            height: Dimension::Pixels(2),
            flip_horizontal: false,
            flip_vertical: false,
        },
        align,
    }
}

#[test]
fn focal_smart_crop_keeps_the_named_region() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    // The focal rectangle names the black bottom-right of
    // the 8x4 pattern. A square crop window is 4 wide; the
    // rectangle forces it to x 4..8, all black.
    let out = engine
        .process(&src, &[focal_op(4, 2, 8, 4), square_crop(Align::Smart)])
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert_eq!((img.width(), img.height()), (2, 2));
    assert!(img.pixels().all(|p| p[0] == 0));
    // Control: the same smart crop without the rectangle
    // keeps part of the white quadrant.
    let out = engine.process(&src, &[square_crop(Align::Smart)]).unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(img.pixels().any(|p| p[0] > 0));
}

#[test]
fn focal_center_crop_centers_on_the_rectangle() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    let out = engine
        .process(&src, &[focal_op(4, 2, 8, 4), square_crop(Align::Center)])
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert_eq!((img.width(), img.height()), (2, 2));
    assert!(img.pixels().all(|p| p[0] == 0));
    // Control: a plain centered crop of the pattern keeps
    // part of the white quadrant.
    let out = engine.process(&src, &[square_crop(Align::Center)]).unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(img.pixels().any(|p| p[0] > 0));
}

#[test]
fn focal_is_ignored_by_a_directional_align() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    // Left alignment is an explicit instruction: the crop
    // takes the left half (white quadrant included) even
    // though the focal rectangle names the right side.
    let out = engine
        .process(&src, &[focal_op(4, 2, 8, 4), square_crop(Align::Left)])
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert_eq!((img.width(), img.height()), (2, 2));
    assert!(img.pixels().any(|p| p[0] > 0));
}

#[test]
fn focal_out_of_bounds_is_an_error() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    let err = engine
        .process(&src, &[focal_op(0, 0, 9, 4), square_crop(Align::Smart)])
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    let err = engine
        .process(&src, &[focal_op(0, 0, 8, 5), square_crop(Align::Smart)])
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
}

#[test]
fn focal_without_a_crop_is_a_no_op() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    let plain = engine.process(&src, &[]).unwrap();
    let with_focal = engine.process(&src, &[focal_op(1, 1, 7, 3)]).unwrap();
    assert_eq!(with_focal.bytes, plain.bytes);
    assert_eq!(dims_of(&with_focal), dims_of(&plain));
}

fn point_op(x: f32, y: f32, zoom: f32) -> Operation {
    Operation::FocalPoint { x, y, zoom }
}

#[test]
fn focal_point_centers_the_crop_on_the_point() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    // The bottom-right corner point: the 4x4 fill window
    // anchors at (4, 0), all black. The plain center crop
    // (window at (2, 0)) keeps part of the white quadrant.
    let out = engine
        .process(&src, &[point_op(1.0, 1.0, 1.0), square_crop(Align::Center)])
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(img.pixels().all(|p| p[0] == 0));
    let out = engine.process(&src, &[square_crop(Align::Center)]).unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(img.pixels().any(|p| p[0] > 0));
    // The top-left corner point keeps the white quadrant
    // in frame (window at (0, 0), mixed), where the
    // zoomed version below is uniformly white. The
    // downscale blends the rows, so the checks are on
    // bands, not exact values.
    let out = engine
        .process(&src, &[point_op(0.0, 0.0, 1.0), square_crop(Align::Center)])
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(img.pixels().any(|p| p[0] > 200));
    assert!(img.pixels().any(|p| p[0] < 55));
}

#[test]
fn focal_point_zoom_tightens_the_window() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    // Zoom 2 at the top-left corner: the window is the
    // 2x2 white block itself.
    let out = engine
        .process(&src, &[point_op(0.0, 0.0, 2.0), square_crop(Align::Center)])
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(img.pixels().all(|p| p[0] == 255));
    // Zoom 2 at the bottom-right corner: all black.
    let out = engine
        .process(&src, &[point_op(1.0, 1.0, 2.0), square_crop(Align::Center)])
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(img.pixels().all(|p| p[0] == 0));
}

#[test]
fn focal_point_and_rect_last_one_wins() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    // Rect after point: the rectangle (the white block)
    // steers, so white survives.
    let out = engine
        .process(
            &src,
            &[
                point_op(1.0, 1.0, 1.0),
                focal_op(0, 0, 4, 2),
                square_crop(Align::Center),
            ],
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(img.pixels().any(|p| p[0] > 0));
    // Point after rect: the corner point steers, all black.
    let out = engine
        .process(
            &src,
            &[
                focal_op(0, 0, 4, 2),
                point_op(1.0, 1.0, 1.0),
                square_crop(Align::Center),
            ],
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(img.pixels().all(|p| p[0] == 0));
}

#[test]
fn focal_point_is_ignored_by_a_directional_align() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    let out = engine
        .process(&src, &[point_op(1.0, 1.0, 1.0), square_crop(Align::Left)])
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(img.pixels().any(|p| p[0] > 0));
}

#[test]
fn focal_point_without_a_crop_is_a_no_op() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    let plain = engine.process(&src, &[]).unwrap();
    let with_point = engine.process(&src, &[point_op(0.25, 0.75, 3.0)]).unwrap();
    assert_eq!(with_point.bytes, plain.bytes);
}

/// `w` x `h` with `n` equal vertical stripes, stripe `i` the
/// gray `20 + 10 * i`: no stripe is black (the pad fill) or
/// white (a trim border), and every stripe tells itself
/// apart from its neighbors.
fn stripes(w: u32, h: u32, n: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, _| {
        let v = (20 + 10 * (x * n / w)) as u8;
        image::Rgb([v, v, v])
    }))
}

/// Which stripe of [`stripes`] the output pixel shows.
fn stripe_at(img: &DynamicImage, x: u32, y: u32) -> i32 {
    let v = img.to_rgb8().get_pixel(x, y)[0] as i32;
    ((v - 20) as f32 / 10.0).round() as i32
}

#[test]
fn focal_rect_follows_a_resize_to_its_crop() {
    // Issue #465: the rectangle names stripe 6 of a
    // 2000x1000 source of 100 px stripes. A resize between
    // the segment and the crop left it in source
    // coordinates, which the crop re-clamped onto the
    // 500 px frame's right edge: the window centered on the
    // stripe 14/15 boundary. The rectangle now scales with
    // the image, so all three spellings agree on stripe 6.
    let src = stripes(2000, 1000, 20);
    let width_500 = size_op(Dimension::Pixels(500), Dimension::Proportional);
    let spellings: [Vec<Operation>; 3] = [
        vec![
            focal_op(600, 0, 700, 100),
            width_500.clone(),
            crop_op(100, 100, Align::Center),
        ],
        vec![focal_op(600, 0, 700, 100), crop_op(100, 100, Align::Center)],
        vec![
            width_500,
            focal_op(150, 0, 175, 25),
            crop_op(100, 100, Align::Center),
        ],
    ];
    for ops in &spellings {
        let out = run_chain(&src, ops);
        assert_eq!((out.width(), out.height()), (100, 100));
        assert_eq!(stripe_at(&out, 50, 50), 6, "{ops:?}");
    }
}

#[test]
fn focal_point_follows_a_crop_rect_to_its_crop() {
    // Issue #465: the point names stripe 12 (x 1200 of
    // 2000). The crop-rect keeps x 1000..2000, where the
    // stale fraction 0.6 lands on x 1600 instead, so the
    // window centered near stripe 16. The point now moves
    // with the cut: x 200 of the 1000 px frame, a window
    // clamped to the frame's left edge, and stripe 12 at the
    // output's center.
    let src = stripes(2000, 1000, 20);
    let ops = [
        point_op(0.6, 0.5, 1.0),
        rect_op(1000, 0, 2000, 500),
        crop_op(100, 100, Align::Center),
    ];
    let out = run_chain(&src, &ops);
    assert_eq!(stripe_at(&out, 50, 50), 12);
    // The window starts at the cut's left edge (stripe 10),
    // where the stale fraction's window started at stripe 13.
    assert_eq!(stripe_at(&out, 5, 50), 10);
}

#[test]
fn focal_forms_follow_a_flip_a_pad_and_a_trim() {
    // 200x100, 10 px stripes: the rectangle names stripe 6.
    let src = stripes(200, 100, 20);
    let rect = || focal_op(60, 0, 70, 10);
    let crop = || crop_op(100, 100, Align::Center);
    // A resize flip mirrors the frame: stripe 6 now sits at
    // x 130..140, and the rectangle goes with it.
    let flipped = Operation::Size(ImageSize {
        width: Dimension::Pixels(200),
        height: Dimension::Pixels(100),
        flip_horizontal: true,
        flip_vertical: false,
    });
    let out = run_chain(&src, &[rect(), flipped, crop()]);
    assert_eq!(stripe_at(&out, 50, 50), 6);
    // A pad sets the frame 100 px into a 400 px canvas; the
    // stale rectangle centered the window on black fill, the
    // stale point on stripe 3.
    let out = run_chain(
        &src,
        &[rect(), pad_op(400, 100, PadFill::Directive), crop()],
    );
    assert_eq!(stripe_at(&out, 50, 50), 6);
    let out = run_chain(
        &src,
        &[
            point_op(0.325, 0.5, 1.0),
            pad_op(400, 100, PadFill::Directive),
            crop(),
        ],
    );
    assert_eq!(stripe_at(&out, 50, 50), 6);
    // A trim cuts a 10 px white border away: the rectangle,
    // named on the bordered frame, moves with the cut (the
    // stale one landed on stripe 7).
    let mut bordered = image::RgbImage::from_pixel(220, 120, image::Rgb([255, 255, 255]));
    image::imageops::overlay(&mut bordered, &src.to_rgb8(), 10, 10);
    let bordered = DynamicImage::ImageRgb8(bordered);
    let out = run_chain(
        &bordered,
        &[
            focal_op(70, 10, 80, 20),
            Operation::Filters(vec![Filter::Trim(10)]),
            crop(),
        ],
    );
    assert_eq!((out.width(), out.height()), (100, 100));
    assert_eq!(stripe_at(&out, 50, 50), 6);
}

#[test]
fn focal_forms_survive_filters_that_paint_in_place_and_drop_on_ones_that_move() {
    let src = stripes(200, 100, 20);
    let crop = || crop_op(100, 100, Align::Center);
    // A border paints over the frame without moving it: the
    // rectangle still steers the crop.
    let out = run_chain(
        &src,
        &[
            focal_op(60, 0, 70, 10),
            Operation::Filters(vec![Filter::Border(2, [255, 0, 0])]),
            crop(),
        ],
    );
    assert_eq!(stripe_at(&out, 50, 50), 6);
    // A rotation moves every pixel in a way the chain does
    // not track: the form drops, and the crop is the plain
    // center crop of the rotated frame.
    let rotate = || Operation::Filters(vec![Filter::Rotate(180.0)]);
    let plain = run_chain(&src, &[rotate(), crop()]);
    for focal in [focal_op(60, 0, 70, 10), point_op(0.3, 0.5, 1.0)] {
        let out = run_chain(&src, &[focal, rotate(), crop()]);
        assert_eq!(out.to_rgb8().as_raw(), plain.to_rgb8().as_raw());
    }
}

#[test]
fn focal_forms_survive_hostile_geometry() {
    // Rectangles and points cut away by a crop-rect, pushed
    // through one-pixel frames, flips and pads, never panic
    // and never leave the frame: the crop always cuts a
    // window of the requested size.
    let src = stripes(200, 100, 20);
    let focals = [
        focal_op(0, 0, 1, 1),
        focal_op(190, 90, 200, 100),
        focal_op(0, 0, 200, 100),
        point_op(0.0, 0.0, 1.0),
        point_op(1.0, 1.0, 100.0),
        point_op(0.5, 0.5, 0.01),
    ];
    let steps = [
        rect_op(100, 50, 101, 51),
        rect_op(0, 0, 50, 100),
        rect_op(150, 0, 200, 100),
        size_op(Dimension::Pixels(1), Dimension::Pixels(1)),
        size_op(Dimension::Pixels(3), Dimension::Pixels(700)),
        Operation::Size(ImageSize {
            width: Dimension::Pixels(17),
            height: Dimension::Pixels(5),
            flip_horizontal: true,
            flip_vertical: true,
        }),
        Operation::FitIn {
            size: ImageSize {
                width: Dimension::Pixels(1),
                height: Dimension::Proportional,
                flip_horizontal: false,
                flip_vertical: false,
            },
            adaptive: false,
        },
        pad_op(1, 900, PadFill::Directive),
        Operation::Filters(vec![Filter::TrimSides(0, [u32::MAX; 4])]),
    ];
    let ctx = ProcessContext::default();
    for focal in &focals {
        for step in &steps {
            for second in &steps {
                let ops = [
                    focal.clone(),
                    step.clone(),
                    second.clone(),
                    crop_op(7, 3, Align::Smart),
                ];
                let directives = crate::url::RequestDirectives::analyze(&ops);
                // A crop-rect past a frame an earlier step
                // shrank is the usual 400, never a panic.
                match apply_operations(src.clone(), &ops, None, 1.0, &ctx, &directives, None) {
                    Ok((out, ..)) => assert_eq!((out.width(), out.height()), (7, 3), "{ops:?}"),
                    Err(err) => assert!(
                        matches!(&err, EngineError::Unsupported(msg) if msg.contains("crop-rect")),
                        "{ops:?}: {err:?}"
                    ),
                }
            }
        }
    }
}

#[test]
fn pending_focal_maps_through_each_step() {
    // The geometry behind the crop tests, exactly.
    let rect = PendingFocal::rect((60, 10, 70, 30), (200, 100));
    assert_eq!(
        rect.scaled((100, 50), false, false).resolve((100, 50)),
        Focal::Rect((30, 5, 35, 15))
    );
    assert_eq!(
        rect.scaled((200, 100), true, true).resolve((200, 100)),
        Focal::Rect((130, 70, 140, 90))
    );
    // A cut keeps the part inside; a rectangle cut away
    // entirely lands on the nearest edge.
    assert_eq!(
        rect.cropped((65, 0, 100, 100)).resolve((100, 100)),
        Focal::Rect((0, 10, 5, 30))
    );
    assert_eq!(
        rect.cropped((100, 0, 100, 100)).resolve((100, 100)),
        Focal::Rect((0, 10, 0, 30))
    );
    assert_eq!(
        rect.placed((100, 0, 200, 100), (400, 100))
            .resolve((400, 100)),
        Focal::Rect((160, 10, 170, 30))
    );
    // A point comes back as fractions of the frame in hand,
    // exactly as named when nothing moved it.
    let point = PendingFocal::point((0.6, 0.5, 2.0), (2000, 1000));
    assert_eq!(point.resolve((2000, 1000)), Focal::Point((0.6, 0.5, 2.0)));
    // 0.6 is not exact in f32, so the moved point is
    // checked to within a rounding step.
    let Focal::Point((x, y, zoom)) = point.cropped((1000, 0, 1000, 500)).resolve((1000, 500))
    else {
        panic!("a point resolves to a point");
    };
    assert!((x - 0.2).abs() < 1e-6, "{x}");
    assert_eq!((y, zoom), (1.0, 2.0));
}

fn crop_op(w: u32, h: u32, align: Align) -> Operation {
    Operation::Crop {
        size: ImageSize {
            width: Dimension::Pixels(w),
            height: Dimension::Pixels(h),
            flip_horizontal: false,
            flip_vertical: false,
        },
        align,
    }
}

fn debug_px(out: &ProcessedImage, x: u32, y: u32) -> image::Rgba<u8> {
    *image::load_from_memory(&out.bytes)
        .unwrap()
        .to_rgba8()
        .get_pixel(x, y)
}

/// 100x100 gray with a 20x20 checker block at (70, 40):
/// the saliency fixture the focus tests use, so the focal
/// point is deterministic (near the block's center).
fn debug_block_image() -> DynamicImage {
    let mut img = DynamicImage::new_rgb8(100, 100);
    for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        *p = if (70..90).contains(&x) && (40..60).contains(&y) {
            if (x + y) % 2 == 0 {
                image::Rgb([200, 200, 200])
            } else {
                image::Rgb([50, 50, 50])
            }
        } else {
            image::Rgb([128, 128, 128])
        };
    }
    img
}

#[test]
fn debug_focal_draws_the_smart_crop_decision() {
    let engine = Engine;
    let img = debug_block_image();
    let src = png_bytes(&img);
    let point = crate::focus::focal_point(&img);
    let (fx, fy) = (
        (point.x * 100.0).round() as u32,
        (point.y * 100.0).round() as u32,
    );
    let out = engine
        .process(
            &src,
            &[Operation::DebugFocal, crop_op(50, 100, Align::Smart)],
        )
        .unwrap();
    // The response is the 100x100 canvas the crop ran
    // on, not the 50x100 crop result.
    assert_eq!(dims_of(&out), (100, 100));
    // The crosshair sits on the focal point the crop used.
    assert_eq!(debug_px(&out, fx, fy), crate::debugview::FOCAL_COLOR);
    // The window outline matches the window the same URL
    // without the debug segment cuts: 50 wide, positioned
    // on the focal point, full height.
    let start = ((point.x * 100.0).round() as i32 - 25).clamp(0, 50) as u32;
    assert_eq!(debug_px(&out, start, 0), crate::debugview::WINDOW_COLOR);
    assert_eq!(
        debug_px(&out, start + 49, 99),
        crate::debugview::WINDOW_COLOR
    );
    let plain = engine
        .process(&src, &[crop_op(50, 100, Align::Smart)])
        .unwrap();
    assert_eq!(dims_of(&plain), (50, 100));
    // An unmarked pixel keeps the canvas value.
    assert_eq!(debug_px(&out, 10, 10), image::Rgba([128, 128, 128, 255]));
}

#[test]
fn debug_focal_draws_the_caller_focal_rectangle() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    let out = engine
        .process(
            &src,
            &[
                Operation::DebugFocal,
                focal_op(5, 2, 7, 4),
                square_crop(Align::Smart),
            ],
        )
        .unwrap();
    // The canvas is the 8x4 pattern, not the 2x2 result.
    assert_eq!(dims_of(&out), (8, 4));
    // The focal rectangle's top edge, clear of the window.
    assert_eq!(debug_px(&out, 6, 2), crate::debugview::FOCAL_COLOR);
    // The crosshair at the rectangle's center (6, 3) is
    // drawn last, over the window's bottom band.
    assert_eq!(debug_px(&out, 6, 3), crate::debugview::FOCAL_COLOR);
    // The crop window is x 4..8, full height (the same
    // window the plain focal test crops to all black).
    assert_eq!(debug_px(&out, 4, 0), crate::debugview::WINDOW_COLOR);
    assert_eq!(debug_px(&out, 7, 0), crate::debugview::WINDOW_COLOR);
    // The same chain without the debug segment still
    // produces the plain 2x2 crop.
    let plain = engine
        .process(&src, &[focal_op(5, 2, 7, 4), square_crop(Align::Smart)])
        .unwrap();
    assert_eq!(dims_of(&plain), (2, 2));
}

#[test]
fn debug_focal_without_a_crop_marks_the_final_image() {
    let engine = Engine;
    let img = debug_block_image();
    let src = png_bytes(&img);
    let point = crate::focus::focal_point(&img);
    let (fx, fy) = (
        (point.x * 100.0).round() as u32,
        (point.y * 100.0).round() as u32,
    );
    let out = engine.process(&src, &[Operation::DebugFocal]).unwrap();
    assert_eq!(dims_of(&out), (100, 100));
    assert_eq!(debug_px(&out, fx, fy), crate::debugview::FOCAL_COLOR);
    // No crop ran, so no window outline exists anywhere.
    let rgba = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
    assert!(rgba.pixels().all(|p| *p != crate::debugview::WINDOW_COLOR));
    // And the diagnostic is not the plain result.
    let plain = engine.process(&src, &[]).unwrap();
    assert_ne!(out.bytes, plain.bytes);
}

#[test]
fn debug_focal_on_an_animated_source_is_a_specific_error() {
    let engine = Engine;
    for src in [animated_gif_bytes(), animated_webp_bytes()] {
        let err = engine
            .process_preferred(&src, &[Operation::DebugFocal], ProcessContext::default())
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("debug:focal"), "{err}");
        assert!(err.to_string().contains("animated"), "{err}");
    }
}

#[test]
fn ai_filters_refuse_animated_sources() {
    // Issue #261: an inference filter in an animated
    // chain would run once per frame (up to the 200
    // frame cap) under one AI gate permit. The refusal
    // names the class; a plain filter chain on the
    // same source still runs.
    let engine = Engine;
    let src = animated_gif_bytes();
    let ops = [Operation::Filters(vec![Filter::RemoveRegion {
        x: 0,
        y: 0,
        w: 5,
        h: 5,
    }])];
    // The provider is configured, so the configuration
    // preflight passes and the animated refusal is
    // what answers.
    let ctx = ProcessContext {
        inpaint_provider: Some(&FlatInpaint),
        ..Default::default()
    };
    let err = engine.process_preferred(&src, &ops, ctx).unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("animated"), "{err}");
    let plain = [Operation::Filters(vec![Filter::Grayscale])];
    engine
        .process_preferred(&src, &plain, ProcessContext::default())
        .unwrap();
}

#[test]
fn debug_focal_with_a_video_format_is_a_specific_error() {
    let engine = Engine;
    let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
    let src = png_bytes(&DynamicImage::new_rgb8(20, 12));
    let err = engine
        .process_preferred(
            &src,
            &[
                Operation::DebugFocal,
                Operation::Filters(vec![Filter::Format(FormatSpec::plain(ImageFormat::Mp4))]),
            ],
            ProcessContext {
                video_encoder: Some(&encoder),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("debug:focal"), "{err}");
    assert!(err.to_string().contains("mp4"), "{err}");
}

#[test]
fn no_upscale_caps_every_resize_mode() {
    let engine = Engine;
    let src = png_bytes(&DynamicImage::new_rgb8(200, 200));
    let directive = Operation::Filters(vec![Filter::NoUpscale]);
    // Exact size, with the directive after the size on purpose:
    // the cap is request-wide, not positional. Each axis
    // clamps at the source dimension (issue #230).
    let out = engine
        .process(
            &src,
            &[
                size_op(Dimension::Pixels(1000), Dimension::Pixels(500)),
                directive.clone(),
            ],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (200, 200));
    // The README's example: a 300x200 resize of a 200x150
    // source returns the source at its own dimensions, not
    // a proportional shrink to 200x133.
    let wide = png_bytes(&DynamicImage::new_rgb8(200, 150));
    let out = engine
        .process(
            &wide,
            &[
                size_op(Dimension::Pixels(300), Dimension::Pixels(200)),
                directive.clone(),
            ],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (200, 150));
    // A mixed request clamps only the axis that would
    // enlarge; the shrinking axis is honored as written.
    let out = engine
        .process(
            &wide,
            &[
                size_op(Dimension::Pixels(300), Dimension::Pixels(100)),
                directive.clone(),
            ],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (200, 100));
    // Fit-in.
    let out = engine
        .process(
            &src,
            &[
                directive.clone(),
                fit_in_op(Dimension::Pixels(1000), Dimension::Pixels(1000)),
            ],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (200, 200));
    // Crop (aspect crop plus exact resize).
    let out = engine
        .process(
            &src,
            &[
                directive,
                Operation::Crop {
                    size: ImageSize {
                        width: Dimension::Pixels(500),
                        height: Dimension::Pixels(500),
                        flip_horizontal: false,
                        flip_vertical: false,
                    },
                    align: Align::Center,
                },
            ],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (200, 200));
    // Sanity: without the directive the same size upscales freely.
    let out = engine
        .process(
            &src,
            &[size_op(Dimension::Pixels(1000), Dimension::Pixels(500))],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (1000, 500));
}

#[test]
fn max_bytes_fits_the_budget() {
    let engine = Engine;
    let src = noise_png();
    let control = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Jpeg,
            ))])],
        )
        .unwrap();
    assert!(
        control.bytes.len() > 5000,
        "control {} bytes",
        control.bytes.len()
    );
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
                Filter::MaxBytes(5000),
            ])],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Jpeg);
    assert!(out.bytes.len() <= 5000, "{} bytes", out.bytes.len());
    // The reduction is reported as the chosen quality.
    let q = out.quality.expect("reduced quality is reported");
    assert!(q < 80, "quality {q}");
}

fn webp_chunk(bytes: &[u8]) -> &'static str {
    if bytes.windows(4).any(|c| c == b"VP8L") {
        "VP8L"
    } else if bytes.windows(4).any(|c| c == b"VP8 ") {
        "VP8 "
    } else {
        "?"
    }
}

#[test]
fn webp_encodes_lossy_by_default() {
    let engine = Engine;
    let src = noise_png();
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::WebP,
            ))])],
        )
        .unwrap();
    assert_eq!(webp_chunk(&out.bytes), "VP8 ");
    assert_eq!(decode_dims(&out.bytes), (120, 120));
    // Lossy on noise is far smaller than the lossless encode.
    let lossless = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::WebP)),
                Filter::Quality(Quality::Value(100)),
            ])],
        )
        .unwrap();
    assert_eq!(webp_chunk(&lossless.bytes), "VP8L");
    assert!(
        out.bytes.len() * 2 < lossless.bytes.len(),
        "lossy {} vs lossless {}",
        out.bytes.len(),
        lossless.bytes.len()
    );
}

#[test]
fn webp_quality_100_is_lossless_and_exact() {
    let engine = Engine;
    let src = noise_png();
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::WebP)),
                Filter::Quality(Quality::Value(100)),
            ])],
        )
        .unwrap();
    let decoded = image::load_from_memory(&out.bytes).unwrap();
    let original = image::load_from_memory(&src).unwrap();
    assert_eq!(decoded.to_rgb8(), original.to_rgb8());
}

#[test]
fn webp_lossy_size_tracks_quality() {
    let engine = Engine;
    let src = noise_png();
    let len = |q: u8| {
        engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Format(FormatSpec::plain(ImageFormat::WebP)),
                    Filter::Quality(Quality::Value(q)),
                ])],
            )
            .unwrap()
            .bytes
            .len()
    };
    assert!(len(50) < len(95), "q50 {} vs q95 {}", len(50), len(95));
}

#[test]
fn webp_lossy_preserves_alpha() {
    let engine = Engine;
    let mut rgba = image::RgbaImage::from_pixel(16, 16, image::Rgba([200, 30, 30, 255]));
    for x in 0..8 {
        for y in 0..16 {
            rgba.put_pixel(x, y, image::Rgba([30, 30, 200, 0]));
        }
    }
    let src = png_bytes(&DynamicImage::ImageRgba8(rgba));
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::WebP)),
                Filter::Quality(Quality::Value(80)),
            ])],
        )
        .unwrap();
    let decoded = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
    assert_eq!(decoded.get_pixel(12, 8)[3], 255);
    assert_eq!(decoded.get_pixel(4, 8)[3], 0);
}

#[test]
fn webp_lossy_opaque_rgba_keeps_alpha() {
    // An RGBA source whose alpha is fully opaque still
    // takes the RGBA encode branch; the decoded output
    // must keep alpha 255 everywhere and stay lossy.
    let engine = Engine;
    let rgba = image::RgbaImage::from_pixel(24, 18, image::Rgba([90, 140, 60, 255]));
    let src = png_bytes(&DynamicImage::ImageRgba8(rgba));
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::WebP)),
                Filter::Quality(Quality::Value(80)),
            ])],
        )
        .unwrap();
    assert_eq!(webp_chunk(&out.bytes), "VP8 ");
    assert_eq!(decode_dims(&out.bytes), (24, 18));
    let decoded = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
    assert!(decoded.pixels().all(|p| p[3] == 255));
}

#[test]
fn webp_lossy_keeps_channel_order() {
    // Three vertical bands, each dominated by one channel:
    // a swapped-channel encode would flip the per-band
    // dominant channel in the decoded output.
    let engine = Engine;
    let mut rgb = image::RgbImage::new(48, 16);
    for (x, _y, px) in rgb.enumerate_pixels_mut() {
        *px = match x {
            0..16 => image::Rgb([210, 40, 40]),
            16..32 => image::Rgb([40, 200, 50]),
            _ => image::Rgb([50, 60, 215]),
        };
    }
    let src = png_bytes(&DynamicImage::ImageRgb8(rgb));
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::WebP)),
                Filter::Quality(Quality::Value(90)),
            ])],
        )
        .unwrap();
    let decoded = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    let band = |x: u32| {
        let p = decoded.get_pixel(x, 8);
        (p[0], p[1], p[2])
    };
    let (r, g, b) = band(8);
    assert!(r > g + 60 && r > b + 60, "red band decoded {r},{g},{b}");
    let (r, g, b) = band(24);
    assert!(g > r + 60 && g > b + 60, "green band decoded {r},{g},{b}");
    let (r, g, b) = band(40);
    assert!(b > r + 60 && b > g + 60, "blue band decoded {r},{g},{b}");
}

#[test]
fn webp_max_bytes_uses_the_quality_ladder() {
    let engine = Engine;
    let src = noise_png();
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::WebP)),
                Filter::MaxBytes(4000),
            ])],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::WebP);
    assert!(out.bytes.len() <= 4000, "{} bytes", out.bytes.len());
}

/// The sampling factors from a JPEG's SOF0 segment, per
/// component: (id, sampling byte). 4:2:0 shows Y as 0x22.
fn jpeg_sof_sampling(bytes: &[u8]) -> Vec<(u8, u8)> {
    let mut i = 2;
    while i + 9 < bytes.len() {
        if bytes[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = bytes[i + 1];
        if marker == 0xC0 {
            let n = bytes[i + 9] as usize;
            return (0..n)
                .map(|c| (bytes[i + 10 + c * 3], bytes[i + 11 + c * 3]))
                .collect();
        }
        if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
            i += 2;
            continue;
        }
        let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        i += 2 + len;
    }
    panic!("no SOF0 segment");
}

#[test]
fn jpeg_uses_420_subsampling_and_stays_compact() {
    let engine = Engine;
    let src = noise_png();
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Jpeg,
            ))])],
        )
        .unwrap();
    let sampling = jpeg_sof_sampling(&out.bytes);
    assert_eq!(sampling.len(), 3);
    assert_eq!(sampling[0].1, 0x22, "Y sampling (4:2:0)");
    assert_eq!(sampling[1].1, 0x11, "Cb sampling");
    // Regression bound from the #38 encoder swap: this
    // fixture encoded to 20,746 B with the previous encoder
    // and 12,069 B with jpeg-encoder at 4:2:0.
    assert!(out.bytes.len() < 14000, "{} bytes", out.bytes.len());
    assert_eq!(decode_dims(&out.bytes), (120, 120));
}

#[test]
fn resize_backend_matches_lanczos3_within_rounding() {
    // Same filter (Lanczos3) in both backends, so outputs
    // track closely: on photo content the spike measured a
    // max per-channel diff of 1 (RMS 0.14). This pure-noise
    // fixture is the adversarial case for kernel precision:
    // measured RMS 0.48 with a tail to 9 (0.3% of channels
    // over 2). The bounds below sit well above that and far
    // below any real breakage (a wrong filter or stride
    // pushes RMS into the tens).
    let src = image::load_from_memory(&noise_png()).unwrap();
    let ours = resize_lanczos3(&src, 60, 60).to_rgb8();
    let reference = image::imageops::resize(&src, 60, 60, image::imageops::FilterType::Lanczos3);
    let (mut maxd, mut sq) = (0u8, 0f64);
    for (a, b) in ours.pixels().zip(reference.pixels()) {
        for c in 0..3 {
            let d = a[c].abs_diff(b[c]);
            maxd = maxd.max(d);
            sq += (d as f64) * (d as f64);
        }
    }
    let rms = (sq / (60.0 * 60.0 * 3.0)).sqrt();
    assert!(rms < 1.0, "rms diff {rms}");
    assert!(maxd <= 10, "max diff {maxd}");
    // The source pixel type is preserved, not forced to RGBA.
    assert!(matches!(
        resize_lanczos3(&src, 30, 30),
        DynamicImage::ImageRgb8(_)
    ));
}

#[test]
fn resize_backend_does_not_bleed_transparent_colors() {
    // Left half transparent red, right half opaque blue. With
    // premultiplied-alpha resampling, the nearly transparent
    // boundary pixel takes its color from the opaque side;
    // straight-alpha resampling would tint it red.
    let mut img = image::RgbaImage::from_pixel(64, 64, image::Rgba([255, 0, 0, 0]));
    for x in 32..64 {
        for y in 0..64 {
            img.put_pixel(x, y, image::Rgba([0, 0, 255, 255]));
        }
    }
    let out = resize_lanczos3(&DynamicImage::ImageRgba8(img), 16, 16).to_rgba8();
    let edge = out.get_pixel(7, 8);
    assert_eq!(edge[0], 0, "red bled into the boundary pixel: {edge:?}");
    assert!(edge[2] > 200, "blue contribution lost: {edge:?}");
    assert!(edge[3] < 64, "edge alpha should be small: {edge:?}");
}

#[test]
fn fill_flattens_transparency_for_jpeg() {
    let engine = Engine;
    // Fully transparent green: RGB stored under zero alpha.
    let rgba = image::RgbaImage::from_pixel(16, 16, image::Rgba([0, 255, 0, 0]));
    let src = png_bytes(&DynamicImage::ImageRgba8(rgba));
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
                Filter::Fill([0, 0, 255]),
            ])],
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    let p = img.get_pixel(8, 8);
    assert!(p[2] > 200 && p[1] < 60, "pixel {p:?}");
    // Without fill, the stored green shows through.
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Jpeg,
            ))])],
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    let p = img.get_pixel(8, 8);
    assert!(p[1] > 200, "pixel {p:?}");
}

fn gif_bytes(img: &DynamicImage) -> Vec<u8> {
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Gif)
        .unwrap();
    out
}

/// Two 40x20 frames: red, then blue.
fn animated_gif_bytes() -> Vec<u8> {
    let red = image::RgbaImage::from_pixel(40, 20, image::Rgba([255, 0, 0, 255]));
    let blue = image::RgbaImage::from_pixel(40, 20, image::Rgba([0, 0, 255, 255]));
    let delay = image::Delay::from_numer_denom_ms(100, 1);
    let frames = vec![
        image::Frame::from_parts(red, 0, 0, delay),
        image::Frame::from_parts(blue, 0, 0, delay),
    ];
    let mut out = Vec::new();
    image::codecs::gif::GifEncoder::new(&mut out)
        .encode_frames(frames)
        .unwrap();
    out
}

#[test]
fn decode_image_rejects_oversized_dimensions() {
    let big = DynamicImage::ImageRgb8(image::RgbImage::new(17_000, 2));
    let bytes = png_bytes(&big);
    let err = decode_image(&bytes).unwrap_err();
    assert!(matches!(err, EngineError::Decode(_)), "{err:?}");
}

#[test]
fn gif_frame_cap_rejects_frame_bombs() {
    let make = |n: usize| {
        let delay = image::Delay::from_numer_denom_ms(100, 1);
        let frames: Vec<image::Frame> = (0..n)
            .map(|_| image::Frame::from_parts(image::RgbaImage::new(4, 4), 0, 0, delay))
            .collect();
        let mut out = Vec::new();
        image::codecs::gif::GifEncoder::new(&mut out)
            .encode_frames(frames)
            .unwrap();
        out
    };
    assert!(decode_gif_frames(&make(201)).is_err());
    assert_eq!(decode_gif_frames(&make(3)).unwrap().len(), 3);
}

#[test]
fn animated_frames_charge_bytes_against_the_decode_budget() {
    // A 1000x500 frame is 500k pixels, 2 MiB retained as
    // RGBA: 200 of them sit exactly at the pixel budget
    // (100M) but reach 400 MiB of frames, past the
    // 256 MiB decode allocation budget. Collection must
    // refuse on the bytes, not sail past on the pixels.
    let delay = image::Delay::from_numer_denom_ms(100, 1);
    let frames = (0..200).map(|_| {
        Ok::<image::Frame, image::ImageError>(image::Frame::from_parts(
            image::RgbaImage::new(1000, 500),
            0,
            0,
            delay,
        ))
    });
    let err = match collect_frames(frames, "GIF") {
        Ok(_) => panic!("collection should refuse on the byte budget"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("decode allocation"), "{err}");
    // A shorter run of the same frames is inside every
    // budget and collects fully.
    let frames = (0..100).map(|_| {
        Ok::<image::Frame, image::ImageError>(image::Frame::from_parts(
            image::RgbaImage::new(1000, 500),
            0,
            0,
            delay,
        ))
    });
    assert_eq!(collect_frames(frames, "GIF").unwrap().len(), 100);
}

#[test]
fn output_pixel_budget_rejects_huge_requests() {
    let img = DynamicImage::ImageRgb8(image::RgbImage::new(64, 64));
    let size = ImageSize {
        width: Dimension::Pixels(16_383),
        height: Dimension::Pixels(16_383),
        flip_horizontal: false,
        flip_vertical: false,
    };
    let err = apply_size(img.clone(), &size, None, 1.0).unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    let err = apply_pad(img, 16_383, 16_383, PadFill::Directive, None, None).unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
}

#[test]
fn output_budget_charges_frames_against_one_total() {
    let mut budget = OutputBudget::with_limit(250);
    assert!(budget.charge(10, 10).is_ok());
    assert!(budget.charge(10, 10).is_ok());
    // Each frame is inside the limit on its own; the
    // running total is what passes it (issue #182).
    let err = budget.charge(10, 10).unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    // A single frame can also blow the budget by itself.
    let mut budget = OutputBudget::with_limit(250);
    assert!(budget.charge(20, 20).is_err());
}

/// Two 100x100 frames, tiny on every input cap. Padded
/// (with upscale capped, so the pad is a canvas fill,
/// not a resample) to 10000x6000, each frame is 60M
/// output pixels: individually inside the per-frame
/// budget, 120M in sum, past the one shared budget.
fn aggregate_overflow_gif() -> Vec<u8> {
    let delay = image::Delay::from_numer_denom_ms(100, 1);
    let frames: Vec<image::Frame> = (0..2)
        .map(|_| image::Frame::from_parts(image::RgbaImage::new(100, 100), 0, 0, delay))
        .collect();
    let mut src = Vec::new();
    image::codecs::gif::GifEncoder::new(&mut src)
        .encode_frames(frames)
        .unwrap();
    src
}

fn aggregate_overflow_ops() -> Vec<Operation> {
    vec![
        Operation::Filters(vec![Filter::NoUpscale]),
        Operation::Pad {
            size: ImageSize {
                width: Dimension::Pixels(10_000),
                height: Dimension::Pixels(6_000),
                flip_horizontal: false,
                flip_vertical: false,
            },
            fill: PadFill::Color([0, 0, 0]),
        },
    ]
}

#[test]
fn animated_output_budget_rejects_aggregate_overflow() {
    // Issue #182, audit R2-01: the frames pass every
    // per-frame check; only the aggregate charge refuses.
    let engine = Engine;
    let err = engine
        .process(&aggregate_overflow_gif(), &aggregate_overflow_ops())
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("across frames"), "{err}");
}

#[test]
fn video_output_budget_rejects_aggregate_overflow() {
    // The same aggregate budget guards the video path:
    // the request fails before the encoder ever runs.
    let engine = Engine;
    let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
    let mut ops = aggregate_overflow_ops();
    ops.push(Operation::Filters(vec![Filter::Format(FormatSpec::plain(
        ImageFormat::Mp4,
    ))]));
    let err = engine
        .process_preferred(
            &aggregate_overflow_gif(),
            &ops,
            ProcessContext {
                video_encoder: Some(&encoder),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("across frames"), "{err}");
}

#[test]
fn rotate_canvas_is_checked_before_it_allocates() {
    // An 8000x8000 image is a legal size; a 45 degree
    // rotation would land on a 11314x11314 canvas (128M
    // px), past the output budget. The chain is rejected
    // by the canvas check, before the rotation runs
    // (issue #182, audit R2-12).
    let img = DynamicImage::new_rgba8(8000, 8000);
    let ops = [Operation::Filters(vec![Filter::Rotate(45.0)])];
    let directives = crate::url::RequestDirectives::analyze(&ops);
    let ctx = ProcessContext::default();
    let err = apply_operations(img.clone(), &ops, None, 1.0, &ctx, &directives, None).unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    // A quarter turn keeps the pixel count and passes.
    let ops = [Operation::Filters(vec![Filter::Rotate(90.0)])];
    let directives = crate::url::RequestDirectives::analyze(&ops);
    let (out, _, _, _, _, _) =
        apply_operations(img, &ops, None, 1.0, &ctx, &directives, None).unwrap();
    assert_eq!((out.width(), out.height()), (8000, 8000));
}

#[test]
fn rotate_chains_inside_the_budget_still_work() {
    let engine = Engine;
    let out = engine
        .process(
            &solid_png(100, 100, [200, 30, 30]),
            &[Operation::Filters(vec![
                Filter::Rotate(45.0),
                Filter::Rotate(45.0),
            ])],
        )
        .unwrap();
    assert_eq!(decode_dims(&out.bytes), (201, 201));
}

#[test]
fn operation_count_cap_rejects_overlong_chains() {
    let engine = Engine;
    let src = solid_png(4, 4, [1, 2, 3]);
    // One segment carrying MAX_OPERATIONS + 1 cheap
    // filters is over: filters count individually.
    let one_segment = vec![Operation::Filters(vec![Filter::Invert; MAX_OPERATIONS + 1])];
    let err = engine.process(&src, &one_segment).unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    // So are MAX_OPERATIONS + 1 single-filter segments.
    let many_segments: Vec<Operation> = (0..=MAX_OPERATIONS)
        .map(|_| Operation::Filters(vec![Filter::Invert]))
        .collect();
    assert!(engine.process(&src, &many_segments).is_err());
    // Exactly at the cap still processes.
    let at_cap: Vec<Operation> = (0..MAX_OPERATIONS)
        .map(|_| Operation::Filters(vec![Filter::Invert]))
        .collect();
    assert!(engine.process(&src, &at_cap).is_ok());
}

/// A varied 48x32 RGB image: flat images make encoder
/// options invisible, gradients and edges do not.
fn gradient_image() -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_fn(48, 32, |x, y| {
        image::Rgb([
            ((x * 5 + y * 3) % 256) as u8,
            ((x * x + y * 7) % 256) as u8,
            ((x * 11 + y * y) % 256) as u8,
        ])
    }))
}

fn gradient_png_small() -> Vec<u8> {
    let mut buf = Vec::new();
    image::codecs::png::PngEncoder::new(&mut buf)
        .write_image(
            gradient_image().as_bytes(),
            48,
            32,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    buf
}

#[test]
fn default_format_options_reproduce_the_standing_encoders() {
    // Byte identity for existing URLs, proven against the
    // encoders configured exactly as the pre-options code
    // configured them (issue #319's defaults rule).
    let img = gradient_image();
    let src = EncodeBuffers::new(&img);

    // JPEG: jpeg-encoder at quality 80, 4:2:0, baseline.
    let rgb = img.to_rgb8();
    let mut legacy = Vec::new();
    let mut enc = jpeg_encoder::Encoder::new(&mut legacy, 80);
    enc.set_sampling_factor(jpeg_encoder::SamplingFactor::F_2_2);
    enc.encode(rgb.as_raw(), 48, 32, jpeg_encoder::ColorType::Rgb)
        .unwrap();
    let now = encode(
        &src,
        ImageFormat::Jpeg,
        Some(80),
        None,
        None,
        FormatOptions::default(),
        DEFAULT_AVIF_SPEED,
        DEFAULT_JXL_SPEED,
    )
    .unwrap();
    assert_eq!(now, legacy, "jpeg default encode changed");

    // WebP lossy: libwebp, quality 80, method 4, no SNS
    // override.
    let mut config = webp::WebPConfig::new().unwrap();
    config.quality = 80.0;
    config.method = 4;
    let legacy = webp::Encoder::from_rgb(rgb.as_raw(), 48, 32)
        .encode_advanced(&config)
        .unwrap()
        .to_vec();
    let now = encode(
        &src,
        ImageFormat::WebP,
        Some(80),
        None,
        None,
        FormatOptions::default(),
        DEFAULT_AVIF_SPEED,
        DEFAULT_JXL_SPEED,
    )
    .unwrap();
    assert_eq!(now, legacy, "webp default encode changed");

    // PNG: the image crate's encoder, untouched.
    let mut legacy = Vec::new();
    image::codecs::png::PngEncoder::new(&mut legacy)
        .write_image(img.as_bytes(), 48, 32, image::ExtendedColorType::Rgb8)
        .unwrap();
    let now = encode(
        &src,
        ImageFormat::Png,
        None,
        None,
        None,
        FormatOptions::default(),
        DEFAULT_AVIF_SPEED,
        DEFAULT_JXL_SPEED,
    )
    .unwrap();
    assert_eq!(now, legacy, "png default encode changed");
}

#[test]
fn jpeg_options_change_the_encode() {
    let img = gradient_image();
    let src = EncodeBuffers::new(&img);
    let encode_with = |options: FormatOptions| {
        encode(
            &src,
            ImageFormat::Jpeg,
            Some(80),
            None,
            None,
            options,
            DEFAULT_AVIF_SPEED,
            DEFAULT_JXL_SPEED,
        )
        .unwrap()
    };
    let baseline = encode_with(FormatOptions::default());
    let progressive = encode_with(FormatOptions {
        jpeg_progressive: true,
        ..FormatOptions::default()
    });
    assert_ne!(progressive, baseline);
    // Progressive output is still a JPEG of the same size
    // class and dimensions.
    assert_eq!(decode_dims(&progressive), (48, 32));
    let s444 = encode_with(FormatOptions {
        jpeg_subsampling: Some(Subsampling::S444),
        ..FormatOptions::default()
    });
    assert_ne!(s444, baseline);
    assert_eq!(decode_dims(&s444), (48, 32));
    let s422 = encode_with(FormatOptions {
        jpeg_subsampling: Some(Subsampling::S422),
        ..FormatOptions::default()
    });
    assert_ne!(s422, baseline);
    assert_ne!(s422, s444);
    // Spelling the standing behavior changes nothing.
    let s420 = encode_with(FormatOptions {
        jpeg_subsampling: Some(Subsampling::S420),
        ..FormatOptions::default()
    });
    assert_eq!(s420, baseline);
}

#[test]
fn png_interlace_marks_the_file_and_round_trips() {
    let img = gradient_image();
    let src = EncodeBuffers::new(&img);
    let plain = encode(
        &src,
        ImageFormat::Png,
        None,
        None,
        None,
        FormatOptions::default(),
        DEFAULT_AVIF_SPEED,
        DEFAULT_JXL_SPEED,
    )
    .unwrap();
    // IHDR: 8 signature + 4 length + 4 type + 4 width +
    // 4 height + 1 depth + 1 color type, then interlace.
    assert_eq!(plain[28], 0);
    let interlaced = encode(
        &src,
        ImageFormat::Png,
        None,
        None,
        None,
        FormatOptions {
            png_interlace: true,
            ..FormatOptions::default()
        },
        DEFAULT_AVIF_SPEED,
        DEFAULT_JXL_SPEED,
    )
    .unwrap();
    assert_eq!(interlaced[28], 1, "IHDR interlace byte must be Adam7");
    let decoded = image::load_from_memory(&interlaced).unwrap();
    assert_eq!(decoded.to_rgb8(), img.to_rgb8());
    // The indexed path interlaces too.
    let indexed = encode_png_indexed(&src, 64, true).unwrap();
    assert_eq!(indexed[28], 1);
    let decoded = image::load_from_memory(&indexed).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (48, 32));
}

/// 16-bit samples in an interlaced PNG are written big-endian,
/// as the format requires (issue #464): the hand-assembled
/// Adam7 stream used to copy the image's native-order bytes,
/// so a little-endian host swapped every sample of a 16-bit
/// source's interlaced render.
#[test]
fn png_interlace_writes_16_bit_samples_big_endian() {
    let deep = DynamicImage::ImageRgba16(
        DynamicImage::ImageRgba32F(image::Rgba32FImage::from_fn(48, 32, |x, y| {
            image::Rgba([x as f32 / 47.0, y as f32 / 31.0, 0.3, 0.6])
        }))
        .to_rgba16(),
    );
    let src = EncodeBuffers::new(&deep);
    let interlaced = encode(
        &src,
        ImageFormat::Png,
        None,
        None,
        None,
        FormatOptions {
            png_interlace: true,
            ..FormatOptions::default()
        },
        DEFAULT_AVIF_SPEED,
        DEFAULT_JXL_SPEED,
    )
    .unwrap();
    assert_eq!(interlaced[28], 1, "IHDR interlace byte must be Adam7");
    let decoded = image::load_from_memory(&interlaced).unwrap();
    assert_eq!(decoded.color(), image::ColorType::Rgba16);
    assert_eq!(decoded.to_rgba16(), deep.to_rgba16());
}

/// A float TIFF built in memory, which decodes to 32-bit
/// float samples (Rgb32F, or Rgba32F with `alpha`).
fn float_tiff(alpha: bool, sample: impl Fn(u32, u32) -> [f32; 4]) -> Vec<u8> {
    let rgba = image::Rgba32FImage::from_fn(48, 32, |x, y| image::Rgba(sample(x, y)));
    let img = if alpha {
        DynamicImage::ImageRgba32F(rgba)
    } else {
        DynamicImage::ImageRgb32F(DynamicImage::ImageRgba32F(rgba).to_rgb32f())
    };
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Tiff)
        .unwrap();
    out
}

fn float_gradient(x: u32, y: u32) -> [f32; 4] {
    [x as f32 / 47.0, y as f32 / 31.0, 0.25, 0.5]
}

/// The integer render a float source should get: its samples
/// clamped to 0..1 at 16 bits.
fn float_expected(bytes: &[u8]) -> DynamicImage {
    let decoded = image::load_from_memory(bytes).unwrap();
    if decoded.color().has_alpha() {
        DynamicImage::ImageRgba16(decoded.to_rgba16())
    } else {
        DynamicImage::ImageRgb16(decoded.to_rgb16())
    }
}

fn process_spelled(src: &[u8], remainder: &str, preferred: Option<ImageFormat>) -> ProcessedImage {
    let path = if remainder.is_empty() {
        "example.com/a.tif".to_string()
    } else {
        format!("{remainder}/example.com/a.tif")
    };
    let (ops, _) = crate::url::parse_operations(&path).unwrap();
    Engine
        .process_preferred(
            src,
            &ops,
            ProcessContext {
                preferred,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{remainder} ({preferred:?}): {e:?}"))
}

/// Float (HDR) sources decode to 32-bit float samples, which
/// the PNG and AVIF encoders do not take: every PNG and AVIF
/// render of one failed with a 500, including the request
/// that names no format and negotiates AVIF from `Accept`
/// (issue #464). They now encode from a 16-bit conversion; the
/// lossless WebP encode, which takes only 8-bit samples, from
/// the 8-bit layout the lossy encode already used.
#[test]
fn float_sources_encode_to_every_still_format() {
    for alpha in [false, true] {
        let tiff = float_tiff(alpha, float_gradient);
        let source = image::load_from_memory(&tiff).unwrap();
        assert_eq!(
            source.color(),
            if alpha {
                image::ColorType::Rgba32F
            } else {
                image::ColorType::Rgb32F
            }
        );
        let expected = float_expected(&tiff);
        let png = process_spelled(&tiff, "filters:format(png)", None);
        assert_eq!(png.format, ImageFormat::Png);
        let decoded = image::load_from_memory(&png.bytes).unwrap();
        assert_eq!(decoded.color(), expected.color(), "alpha {alpha}");
        assert_eq!(decoded, expected, "alpha {alpha}");
        // Interlaced: the same samples, Adam7-ordered.
        let interlaced = process_spelled(&tiff, "filters:format(png,interlace)", None);
        assert_eq!(interlaced.bytes[28], 1);
        assert_eq!(
            image::load_from_memory(&interlaced.bytes).unwrap(),
            expected
        );
        // Indexed PNG already read the 8-bit layout.
        let indexed = process_spelled(&tiff, "filters:format(png):palette(64)", None);
        assert_eq!(decode_dims(&indexed.bytes), (48, 32));
        // A resize keeps float samples, and still encodes.
        let resized = process_spelled(&tiff, "24x0/filters:format(png)", None);
        assert_eq!(decode_dims(&resized.bytes), (24, 16));
        // AVIF, named and negotiated.
        for (remainder, preferred) in [
            ("filters:format(avif)", None),
            ("filters:format(avif):quality(50)", None),
            ("24x0", Some(ImageFormat::Avif)),
        ] {
            let avif = process_spelled(&tiff, remainder, preferred);
            assert_eq!(avif.format, ImageFormat::Avif, "{remainder}");
            assert_eq!(&avif.bytes[4..8], b"ftyp", "{remainder}");
        }
        // Lossless WebP keeps the 8-bit pixels exactly.
        let webp = process_spelled(&tiff, "filters:format(webp):quality(100)", None);
        let decoded = image::load_from_memory(&webp.bytes).unwrap();
        assert_eq!(decoded.to_rgba8(), source.to_rgba8(), "alpha {alpha}");
        // The formats that already worked still do.
        for format in ["jpeg", "webp", "gif", "jxl"] {
            let out = process_spelled(&tiff, &format!("filters:format({format})"), None);
            assert!(!out.bytes.is_empty(), "{format}");
        }
    }
}

/// A Radiance HDR source takes the same route as a float TIFF.
#[test]
fn radiance_hdr_sources_encode_to_png_and_avif() {
    let hdr_img = DynamicImage::ImageRgb32F(image::Rgb32FImage::from_fn(48, 32, |x, y| {
        image::Rgb([x as f32 / 47.0, y as f32 / 31.0, 0.25])
    }));
    let mut hdr = Vec::new();
    hdr_img
        .write_to(&mut Cursor::new(&mut hdr), image::ImageFormat::Hdr)
        .unwrap();
    let png = process_spelled(&hdr, "filters:format(png)", None);
    assert_eq!(
        image::load_from_memory(&png.bytes).unwrap(),
        float_expected(&hdr)
    );
    let interlaced = process_spelled(&hdr, "filters:format(png,interlace)", None);
    assert_eq!(
        image::load_from_memory(&interlaced.bytes).unwrap(),
        float_expected(&hdr)
    );
    let avif = process_spelled(&hdr, "", Some(ImageFormat::Avif));
    assert_eq!(avif.format, ImageFormat::Avif);
}

/// Float samples are source content: NaN, the infinities,
/// negatives, values past 1 and subnormals all encode without
/// a panic, clamped to the displayable range (NaN to the top
/// of it, the image crate's rule for every integer layout).
#[test]
fn hostile_float_samples_clamp_instead_of_failing() {
    let hostile = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        2.0,
        f32::MAX,
        f32::MIN_POSITIVE / 2.0,
        -0.0,
    ];
    for alpha in [false, true] {
        let tiff = float_tiff(alpha, |x, y| {
            let at = |i: u32| hostile[(i as usize) % hostile.len()];
            [at(x), at(y), at(x + y), at(x * 3 + y)]
        });
        let expected = float_expected(&tiff);
        for remainder in [
            "filters:format(png)",
            "filters:format(png,interlace)",
            "filters:format(avif)",
            "filters:format(webp):quality(100)",
            "filters:format(webp)",
            "filters:format(jpeg)",
            "24x0/filters:format(png)",
            "fit-in:20x20/filters:format(png,interlace)",
        ] {
            let out = process_spelled(&tiff, remainder, None);
            assert!(!out.bytes.is_empty(), "{remainder}");
        }
        let png = process_spelled(&tiff, "filters:format(png)", None);
        let decoded = image::load_from_memory(&png.bytes).unwrap();
        assert_eq!(decoded, expected, "alpha {alpha}");
        // Red reads the hostile list along x: NaN, then
        // infinity, then negative infinity.
        let red = |x: u32| decoded.to_rgba16().get_pixel(x, 0).0[0];
        assert_eq!(red(0), u16::MAX, "NaN");
        assert_eq!(red(1), u16::MAX, "infinity");
        assert_eq!(red(2), 0, "negative infinity");
    }
}

#[test]
fn webp_options_change_the_encode() {
    let img = gradient_image();
    let src = EncodeBuffers::new(&img);
    let encode_with = |options: FormatOptions| {
        encode(
            &src,
            ImageFormat::WebP,
            Some(80),
            None,
            None,
            options,
            DEFAULT_AVIF_SPEED,
            DEFAULT_JXL_SPEED,
        )
        .unwrap()
    };
    let baseline = encode_with(FormatOptions::default());
    let method0 = encode_with(FormatOptions {
        webp_method: Some(0),
        ..FormatOptions::default()
    });
    assert_ne!(method0, baseline);
    let method6 = encode_with(FormatOptions {
        webp_method: Some(6),
        ..FormatOptions::default()
    });
    assert_ne!(method6, baseline);
    assert_ne!(method6, method0);
    let sns0 = encode_with(FormatOptions {
        webp_sns: Some(0),
        ..FormatOptions::default()
    });
    let sns100 = encode_with(FormatOptions {
        webp_sns: Some(100),
        ..FormatOptions::default()
    });
    assert_ne!(sns0, sns100);
    // Lossless with a method option goes through libwebp
    // and still decodes to the same pixels.
    let lossless = encode(
        &src,
        ImageFormat::WebP,
        Some(100),
        None,
        None,
        FormatOptions {
            webp_method: Some(6),
            ..FormatOptions::default()
        },
        DEFAULT_AVIF_SPEED,
        DEFAULT_JXL_SPEED,
    )
    .unwrap();
    let decoded = image::load_from_memory(&lossless).unwrap();
    assert_eq!(decoded.to_rgb8(), img.to_rgb8());
}

#[test]
fn format_options_do_not_leak_across_formats() {
    let img = gradient_image();
    let src = EncodeBuffers::new(&img);
    // WebP and PNG fields set while encoding a JPEG: the
    // JPEG encode must be the default one.
    let jpeg = encode(
        &src,
        ImageFormat::Jpeg,
        Some(80),
        None,
        None,
        FormatOptions {
            webp_method: Some(0),
            webp_sns: Some(0),
            png_interlace: true,
            ..FormatOptions::default()
        },
        DEFAULT_AVIF_SPEED,
        DEFAULT_JXL_SPEED,
    )
    .unwrap();
    let jpeg_default = encode(
        &src,
        ImageFormat::Jpeg,
        Some(80),
        None,
        None,
        FormatOptions::default(),
        DEFAULT_AVIF_SPEED,
        DEFAULT_JXL_SPEED,
    )
    .unwrap();
    assert_eq!(jpeg, jpeg_default);
    // JPEG fields set while encoding a PNG.
    let png = encode(
        &src,
        ImageFormat::Png,
        None,
        None,
        None,
        FormatOptions {
            jpeg_progressive: true,
            jpeg_subsampling: Some(Subsampling::S444),
            ..FormatOptions::default()
        },
        DEFAULT_AVIF_SPEED,
        DEFAULT_JXL_SPEED,
    )
    .unwrap();
    let png_default = encode(
        &src,
        ImageFormat::Png,
        None,
        None,
        None,
        FormatOptions::default(),
        DEFAULT_AVIF_SPEED,
        DEFAULT_JXL_SPEED,
    )
    .unwrap();
    assert_eq!(png, png_default);
}

#[test]
fn per_format_quality_applies_after_format_negotiation() {
    let engine = Engine;
    let source = gradient_png_small();
    let quality = Filter::Quality(Quality::PerFormat(vec![
        (Some(ImageFormat::Jpeg), crate::url::QualityValue::Value(30)),
        (Some(ImageFormat::WebP), crate::url::QualityValue::Value(95)),
    ]));
    let as_webp = engine
        .process(
            &source,
            &[
                Operation::Filters(vec![Filter::Format(FormatSpec::plain(ImageFormat::WebP))]),
                Operation::Filters(vec![quality.clone()]),
            ],
        )
        .unwrap()
        .bytes;
    let webp_95 = engine
        .process(
            &source,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::WebP)),
                Filter::Quality(Quality::Value(95)),
            ])],
        )
        .unwrap()
        .bytes;
    assert_eq!(as_webp, webp_95, "webp entry must win for webp output");
    let as_jpeg = engine
        .process(
            &source,
            &[
                Operation::Filters(vec![Filter::Format(FormatSpec::plain(ImageFormat::Jpeg))]),
                Operation::Filters(vec![quality]),
            ],
        )
        .unwrap()
        .bytes;
    let jpeg_30 = engine
        .process(
            &source,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
                Filter::Quality(Quality::Value(30)),
            ])],
        )
        .unwrap()
        .bytes;
    assert_eq!(as_jpeg, jpeg_30, "jpeg entry must win for jpeg output");
    assert_ne!(as_jpeg, as_webp);
}

#[test]
fn jxl_dimension_probe_accepts_small_and_rejects_huge() {
    let img = DynamicImage::ImageRgb8(image::RgbImage::new(24, 16));
    let bytes = encode(
        &EncodeBuffers::new(&img),
        ImageFormat::Jxl,
        Some(80),
        None,
        None,
        FormatOptions::default(),
        DEFAULT_AVIF_SPEED,
        DEFAULT_JXL_SPEED,
    )
    .unwrap();
    assert!(check_jxl_dimensions(&bytes).is_ok());
    assert!(check_decoded_dimensions(20_000, 10).is_err());
    assert!(check_decoded_dimensions(16_384, 16_384).is_ok());
}

/// The (type, data length) of every PNG chunk, in order.
fn png_chunks(bytes: &[u8]) -> Vec<(String, usize)> {
    assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]));
    let mut chunks = Vec::new();
    let mut at = 8;
    while at + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let kind = String::from_utf8_lossy(&bytes[at + 4..at + 8]).into_owned();
        chunks.push((kind, len));
        at += 12 + len;
    }
    chunks
}

#[test]
fn palette_writes_an_indexed_png_within_budget() {
    let engine = Engine;
    let src = png_bytes(&photo_like_rgb(96, 96));
    let ops = [Operation::Filters(vec![
        Filter::Format(FormatSpec::plain(ImageFormat::Png)),
        Filter::Palette(16),
    ])];
    let out = engine.process(&src, &ops).unwrap();
    // IHDR color type 3 (indexed), a PLTE of at most 16
    // entries, and no tRNS for an opaque source.
    assert_eq!(out.bytes[25], 3);
    let chunks = png_chunks(&out.bytes);
    let plte = chunks
        .iter()
        .find(|(k, _)| k == "PLTE")
        .expect("PLTE chunk");
    assert!(
        plte.1 <= 16 * 3 && plte.1.is_multiple_of(3),
        "PLTE {}",
        plte.1
    );
    assert!(chunks.iter().all(|(k, _)| k != "tRNS"));
    // Decoded, the image holds at most 16 distinct colors.
    let decoded = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
    let mut colors = std::collections::HashSet::new();
    for p in decoded.pixels() {
        colors.insert(p.0);
    }
    assert!(colors.len() <= 16, "{} colors", colors.len());
    assert_eq!((decoded.width(), decoded.height()), (96, 96));
    // The directive also composes with the detected
    // format: a PNG source needs no format() to go indexed.
    let out = engine
        .process(&src, &[Operation::Filters(vec![Filter::Palette(16)])])
        .unwrap();
    assert_eq!(out.bytes[25], 3);
    // The full-color control is not indexed, and bigger.
    let full = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Png,
            ))])],
        )
        .unwrap();
    assert_ne!(full.bytes[25], 3);
    assert!(out.bytes.len() < full.bytes.len());
}

#[test]
fn palette_keeps_alpha_in_trns() {
    let engine = Engine;
    let mut img = DynamicImage::new_rgba8(3, 1);
    let rgba = img.as_mut_rgba8().unwrap();
    rgba.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
    rgba.put_pixel(1, 0, image::Rgba([0, 255, 0, 128]));
    rgba.put_pixel(2, 0, image::Rgba([0, 0, 255, 0]));
    let src = png_bytes(&img);
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::Png)),
                Filter::Palette(16),
            ])],
        )
        .unwrap();
    let chunks = png_chunks(&out.bytes);
    assert!(
        chunks.iter().any(|(k, _)| k == "tRNS"),
        "chunks: {chunks:?}"
    );
    // Three source colors in a budget of 16 survive
    // exactly, alpha included (no GIF-style binarizing).
    let decoded = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
    assert_eq!(decoded.get_pixel(0, 0).0, [255, 0, 0, 255]);
    assert_eq!(decoded.get_pixel(1, 0).0, [0, 255, 0, 128]);
    assert_eq!(decoded.get_pixel(2, 0).0[3], 0);
}

#[test]
fn palette_is_ignored_by_other_formats() {
    let engine = Engine;
    let src = png_bytes(&photo_like_rgb(32, 32));
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
                Filter::Palette(16),
            ])],
        )
        .unwrap();
    assert!(out.bytes.starts_with(&[0xFF, 0xD8]), "not a JPEG");
}

fn decoded_gif_frames(bytes: &[u8]) -> Vec<image::Frame> {
    use image::AnimationDecoder;
    let decoder = image::codecs::gif::GifDecoder::new(Cursor::new(bytes)).unwrap();
    decoder.into_frames().collect_frames().unwrap()
}

#[test]
fn static_gif_round_trips_as_gif() {
    let engine = Engine;
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        24,
        16,
        image::Rgb([10, 200, 30]),
    ));
    let src = gif_bytes(&img);
    let out = engine.process(&src, &[]).unwrap();
    assert_eq!(out.format, ImageFormat::Gif);
    assert_eq!(&out.bytes[..6], b"GIF89a");
    assert_eq!(dims_of(&out), (24, 16));
}

#[test]
fn format_gif_converts_a_png_source() {
    let engine = Engine;
    let src = png_bytes(&DynamicImage::new_rgb8(20, 12));
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Gif,
            ))])],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Gif);
    assert_eq!(&out.bytes[..6], b"GIF89a");
}

/// A still with more colors than a GIF palette holds, so
/// the quantizer really runs, plus a fully transparent
/// block whose RGB must not leak into the output.
fn colorful_transparent_png() -> Vec<u8> {
    let mut img = image::RgbaImage::new(48, 32);
    for (x, y, px) in img.enumerate_pixels_mut() {
        *px = image::Rgba([
            (x * 5 % 256) as u8,
            (y * 8 % 256) as u8,
            ((x + y) * 3 % 256) as u8,
            0xFF,
        ]);
    }
    for y in 0..8 {
        for x in 0..8 {
            img.put_pixel(x, y, image::Rgba([200, 30, 30, 0]));
        }
    }
    let mut out = Vec::new();
    DynamicImage::ImageRgba8(img)
        .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
        .unwrap();
    out
}

#[test]
fn gif_still_quantizer_keeps_transparency() {
    // Issue #294: the quantizr path must carry the frame's
    // transparent index, not just its colors.
    let engine = Engine;
    let out = engine
        .process(
            &colorful_transparent_png(),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Gif,
            ))])],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Gif);
    assert_eq!(&out.bytes[..6], b"GIF89a");
    let frames = decoded_gif_frames(&out.bytes);
    assert_eq!(frames.len(), 1);
    let buf = frames[0].buffer();
    assert_eq!((buf.width(), buf.height()), (48, 32));
    // The transparent block decodes back to alpha 0, the
    // opaque gradient to alpha 255.
    assert_eq!(buf.get_pixel(4, 4)[3], 0);
    assert_eq!(buf.get_pixel(40, 24)[3], 0xFF);
}

/// Two 40x20 frames with a transparent corner: frame one
/// opaque red with the corner cut out, frame two blue.
fn animated_transparent_gif_bytes() -> Vec<u8> {
    let mut red = image::RgbaImage::from_pixel(40, 20, image::Rgba([255, 0, 0, 255]));
    for y in 0..5 {
        for x in 0..5 {
            red.put_pixel(x, y, image::Rgba([255, 0, 0, 0]));
        }
    }
    let mut blue = image::RgbaImage::from_pixel(40, 20, image::Rgba([0, 0, 255, 255]));
    for y in 0..5 {
        for x in 0..5 {
            blue.put_pixel(x, y, image::Rgba([0, 0, 255, 0]));
        }
    }
    let delay = image::Delay::from_numer_denom_ms(100, 1);
    let frames = vec![
        image::Frame::from_parts(red, 0, 0, delay),
        image::Frame::from_parts(blue, 0, 0, delay),
    ];
    let mut out = Vec::new();
    image::codecs::gif::GifEncoder::new(&mut out)
        .encode_frames(frames)
        .unwrap();
    out
}

#[test]
fn animated_gif_quantizer_keeps_frames_delays_and_transparency() {
    // Issue #294: the per-frame swap must keep every frame,
    // its delay, and its transparent pixels.
    let engine = Engine;
    let src = animated_transparent_gif_bytes();
    let out = engine.process(&src, &[]).unwrap();
    assert_eq!(out.format, ImageFormat::Gif);
    let frames = decoded_gif_frames(&out.bytes);
    assert_eq!(frames.len(), 2);
    for frame in &frames {
        let (numer, denom) = frame.delay().numer_denom_ms();
        assert_eq!(numer / denom, 100, "frame delay must survive");
        let buf = frame.buffer();
        assert_eq!((buf.width(), buf.height()), (40, 20));
        assert_eq!(
            buf.get_pixel(2, 2)[3],
            0,
            "transparent corner stays transparent"
        );
        assert_eq!(buf.get_pixel(20, 10)[3], 0xFF);
    }
    // Frame bodies keep their colors: red first, blue second.
    assert!(frames[0].buffer().get_pixel(20, 10)[0] > 200);
    assert!(frames[1].buffer().get_pixel(20, 10)[2] > 200);
}

#[test]
fn animated_gif_keeps_its_frames_through_resize() {
    let engine = Engine;
    let src = animated_gif_bytes();
    let out = engine
        .process(
            &src,
            &[size_op(Dimension::Pixels(20), Dimension::Pixels(10))],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Gif);
    let frames = decoded_gif_frames(&out.bytes);
    assert_eq!(frames.len(), 2);
    assert!(
        frames
            .iter()
            .all(|f| f.buffer().width() == 20 && f.buffer().height() == 10)
    );
}

/// A focal provider that answers with the centroid of
/// the bright pixels and counts its runs: the two-frame
/// fixture below mirrors its bright half between
/// frames, so a per-frame analysis would move the
/// smart crop window, and the count shows whether it
/// ran per frame or once.
#[derive(Debug, Default)]
struct BrightCentroid {
    calls: std::sync::atomic::AtomicUsize,
}

impl crate::ai::FocalProvider for BrightCentroid {
    fn focal_point(&self, img: &DynamicImage) -> Result<(f32, f32), crate::ai::AiError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let rgb = img.to_rgb8();
        let (mut sx, mut n) = (0u64, 0u64);
        for (x, _y, px) in rgb.enumerate_pixels() {
            if px.0[0] > 200 {
                sx += x as u64;
                n += 1;
            }
        }
        let cx = if n == 0 {
            0.5
        } else {
            (sx as f32 / n as f32) / rgb.width() as f32
        };
        Ok((cx, 0.5))
    }
}

/// Two 40x20 frames whose bright half mirrors: white
/// on the left first, then on the right.
fn mirrored_halves_gif() -> Vec<u8> {
    let frame = |bright_left: bool| {
        let img = image::RgbaImage::from_fn(40, 20, |x, _y| {
            let bright = (x < 20) == bright_left;
            let v = if bright { 255 } else { 0 };
            image::Rgba([v, v, v, 255])
        });
        image::Frame::from_parts(img, 0, 0, image::Delay::from_numer_denom_ms(100, 1))
    };
    let mut out = Vec::new();
    image::codecs::gif::GifEncoder::new(&mut out)
        .encode_frames(vec![frame(true), frame(false)])
        .unwrap();
    out
}

#[test]
fn animated_smart_crop_uses_one_window_for_every_frame() {
    // Issue #376: the smart crop once decided per
    // frame, so the window jumped as the content moved
    // (and the focal analysis, face detection included,
    // ran per frame). The first frame now decides and
    // the rest replay its window: the mirrored second
    // frame is cropped on the same (left) side, so its
    // output is the dark half, and the provider ran
    // once.
    let engine = Engine;
    let provider = BrightCentroid::default();
    let out = engine
        .process_preferred(
            &mirrored_halves_gif(),
            &[crop_op(20, 20, Align::Smart)],
            ProcessContext {
                focal_provider: Some(&provider),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Gif);
    let frames = decoded_gif_frames(&out.bytes);
    assert_eq!(frames.len(), 2);
    let center = |f: &image::Frame| f.buffer().get_pixel(10, 10).0[0];
    assert!(center(&frames[0]) > 200, "frame one keeps its bright half");
    assert!(center(&frames[1]) < 55, "frame two replays the same window");
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

/// 64x64 white frames, each holding one black square given
/// as (left, top, side); a side of 0 leaves the frame white.
fn square_frames(squares: &[(u32, u32, u32)]) -> Vec<image::Frame> {
    squares
        .iter()
        .map(|&(x, y, side)| {
            let img = image::RgbaImage::from_fn(64, 64, |px, py| {
                if (x..x + side).contains(&px) && (y..y + side).contains(&py) {
                    image::Rgba([0, 0, 0, 255])
                } else {
                    image::Rgba([255, 255, 255, 255])
                }
            });
            image::Frame::from_parts(img, 0, 0, image::Delay::from_numer_denom_ms(100, 1))
        })
        .collect()
}

fn square_gif(squares: &[(u32, u32, u32)]) -> Vec<u8> {
    let mut out = Vec::new();
    image::codecs::gif::GifEncoder::new(&mut out)
        .encode_frames(square_frames(squares))
        .unwrap();
    out
}

fn square_webp(squares: &[(u32, u32, u32)]) -> Vec<u8> {
    let frames: Vec<_> = square_frames(squares)
        .into_iter()
        .map(|f| {
            let buffer = f.into_buffer();
            let image = oxideav_webp::WebpImage::from_rgba8(64, 64, buffer.into_raw()).unwrap();
            oxideav_webp::Frame::new(image, Some(std::time::Duration::from_millis(100)))
        })
        .collect();
    let opts = oxideav_webp::EncodeOptions::default()
        .with_frame_mode(oxideav_webp::AnimFrameMode::Lossless);
    oxideav_webp::encode_animation(&frames, &opts).unwrap()
}

/// A GIF frame's rectangle as (left, top, width, height).
type GifRect = (u16, u16, u16, u16);

/// A GIF's logical screen and each frame's rectangle, read
/// from the file itself: the image crate's decoder
/// composites every frame onto the screen and hides both.
fn gif_layout(bytes: &[u8]) -> ((u16, u16), Vec<GifRect>) {
    let mut options = gif::DecodeOptions::new();
    options.skip_frame_decoding(true);
    let mut decoder = options.read_info(bytes).unwrap();
    let screen = (decoder.width(), decoder.height());
    let mut frames = Vec::new();
    while let Some(frame) = decoder.next_frame_info().unwrap() {
        frames.push((frame.left, frame.top, frame.width, frame.height));
    }
    (screen, frames)
}

fn trim_to(format: ImageFormat, mut filters: Vec<Filter>) -> Vec<Operation> {
    filters.push(Filter::Format(FormatSpec::plain(format)));
    vec![Operation::Filters(filters)]
}

fn luma_at(frame: &image::Frame, x: u32, y: u32) -> u8 {
    frame.buffer().get_pixel(x, y).0[0]
}

#[test]
fn animated_trim_cuts_one_window_for_every_frame() {
    // Issue #465: trim decided per frame, so frames whose
    // content shrank came out at 60, 44, and 24 px, all
    // pushed at 0,0 on a 60x60 screen taken from the first.
    // The frames now share one cut, the union of their
    // boxes: every frame fills the screen, and each square
    // stays where it was relative to the others.
    let engine = Engine;
    let shrinking = [(2, 2, 60), (10, 10, 44), (20, 20, 24)];
    let out = engine
        .process(
            &square_gif(&shrinking),
            &trim_to(ImageFormat::Gif, vec![Filter::Trim(10)]),
        )
        .unwrap();
    assert_eq!((out.width, out.height), (60, 60));
    let (screen, rects) = gif_layout(&out.bytes);
    assert_eq!(screen, (60, 60));
    assert_eq!(rects, vec![(0, 0, 60, 60); 3]);
    let frames = decoded_gif_frames(&out.bytes);
    assert_eq!(luma_at(&frames[0], 0, 0), 0);
    // The 44 px square sat at 10 of 64; the cut starts at 2.
    assert_eq!(luma_at(&frames[1], 8, 8), 0);
    assert_eq!(luma_at(&frames[1], 7, 7), 255);
    assert_eq!(luma_at(&frames[1], 51, 51), 0);
    assert_eq!(luma_at(&frames[1], 52, 52), 255);
    assert_eq!(luma_at(&frames[2], 18, 18), 0);
    assert_eq!(luma_at(&frames[2], 17, 17), 255);
    // Later frames that trim larger set the canvas too: the
    // screen once came from the first frame's 40 px, and
    // the 60 and 63 px frames spilled past it.
    let growing = [(12, 12, 40), (2, 2, 60), (1, 1, 63)];
    let out = engine
        .process(
            &square_gif(&growing),
            &trim_to(ImageFormat::Gif, vec![Filter::Trim(10)]),
        )
        .unwrap();
    let (screen, rects) = gif_layout(&out.bytes);
    assert_eq!(screen, (63, 63));
    assert_eq!(rects, vec![(0, 0, 63, 63); 3]);
    let frames = decoded_gif_frames(&out.bytes);
    assert_eq!(luma_at(&frames[0], 11, 11), 0);
    assert_eq!(luma_at(&frames[0], 10, 10), 255);
}

#[test]
fn animated_trim_to_webp_and_video_shares_one_canvas() {
    // Issue #465: the animated WebP encoder refused frames
    // of different sizes, a 500 for any trimmed animation
    // whose content moved; the video plan refused them with
    // a 400. One cut gives every frame the same canvas.
    let engine = Engine;
    let shrinking = [(2, 2, 60), (10, 10, 44), (20, 20, 24)];
    for src in [square_gif(&shrinking), square_webp(&shrinking)] {
        let out = engine
            .process(&src, &trim_to(ImageFormat::WebP, vec![Filter::Trim(10)]))
            .unwrap();
        assert_eq!(out.format, ImageFormat::WebP);
        assert_eq!((out.width, out.height), (60, 60));
        let frames = decoded_webp_frames(&out.bytes);
        assert_eq!(frames.len(), 3);
        for frame in &frames {
            assert_eq!(frame.buffer().dimensions(), (60, 60));
        }
        // Lossy, so the squares are checked by band.
        assert!(luma_at(&frames[1], 20, 20) < 60);
        assert!(luma_at(&frames[1], 3, 3) > 200);
        let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
        let out = engine
            .process_preferred(
                &src,
                &trim_to(ImageFormat::Mp4, vec![Filter::Trim(10)]),
                ProcessContext {
                    video_encoder: Some(&encoder),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.bytes, b"ENGINE-VIDEO");
        assert_eq!((out.width, out.height), (60, 60));
    }
}

#[test]
fn animated_trim_survives_hostile_frames() {
    // Whatever the frames hold and however the chain trims
    // them, every frame lands on the one screen.
    let engine = Engine;
    let one_canvas = |squares: &[(u32, u32, u32)], ops: &[Operation]| {
        let out = engine.process(&square_gif(squares), ops).unwrap();
        let (screen, rects) = gif_layout(&out.bytes);
        assert_eq!(rects.len(), squares.len(), "{ops:?}");
        for rect in &rects {
            assert_eq!(*rect, (0, 0, screen.0, screen.1), "{ops:?}");
        }
        (screen.0 as u32, screen.1 as u32)
    };
    let trim = |filters: Vec<Filter>| trim_to(ImageFormat::Gif, filters);
    // A blank frame has no box of its own: it takes the
    // others' cut instead of keeping its whole 64x64.
    let with_blank = [(2, 2, 60), (0, 0, 0), (20, 20, 24)];
    assert_eq!(
        one_canvas(&with_blank, &trim(vec![Filter::Trim(10)])),
        (60, 60)
    );
    // Nothing to cut in any frame: nothing is cut.
    let blank = [(0, 0, 0), (0, 0, 0)];
    assert_eq!(one_canvas(&blank, &trim(vec![Filter::Trim(10)])), (64, 64));
    // Sides past the box clamp to one pixel, in every frame.
    let shrinking = [(2, 2, 60), (10, 10, 44), (20, 20, 24)];
    assert_eq!(
        one_canvas(
            &shrinking,
            &trim(vec![Filter::TrimSides(10, [u32::MAX; 4])])
        ),
        (1, 1)
    );
    // Trims chained around other geometry, a later trim
    // surveyed on frames the earlier one already cut, and a
    // shared smart crop after them.
    let chained = [
        rect_op(0, 0, 64, 40),
        Operation::Filters(vec![Filter::Trim(10), Filter::Rotate(90.0)]),
        size_op(Dimension::Pixels(30), Dimension::Proportional),
        Operation::Filters(vec![Filter::TrimSides(0, [1, 0, 0, 2]), Filter::Trim(250)]),
        crop_op(9, 7, Align::Smart),
        Operation::Filters(vec![Filter::Format(FormatSpec::plain(ImageFormat::Gif))]),
    ];
    assert_eq!(one_canvas(&shrinking, &chained), (9, 7));
    // The render's deadline bounds the survey too.
    let err = engine
        .process_preferred(
            &square_gif(&shrinking),
            &trim(vec![Filter::Trim(10)]),
            ProcessContext {
                deadline: Some(std::time::Instant::now()),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Timeout), "{err:?}");
}

#[test]
fn animated_gif_to_a_still_format_keeps_the_first_frame() {
    let engine = Engine;
    let src = animated_gif_bytes();
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Png,
            ))])],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Png);
    // The first frame is solid red.
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    assert_eq!(img.get_pixel(0, 0), &image::Rgb([255, 0, 0]));
}

#[test]
fn animated_gif_stays_out_of_jxl_negotiation() {
    let engine = Engine;
    let src = animated_gif_bytes();
    // A negotiated (preferred) JXL falls back to GIF so the
    // frames survive.
    let out = engine
        .process_preferred(
            &src,
            &[],
            ProcessContext {
                preferred: Some(ImageFormat::Jxl),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Gif);
    assert_eq!(decoded_gif_frames(&out.bytes).len(), 2);
    // An explicit format(jxl) still converts, first frame only.
    let out = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Jxl,
            ))])],
            ProcessContext {
                preferred: Some(ImageFormat::Jxl),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Jxl);
}

#[test]
fn animated_sources_keep_their_frames_under_negotiation() {
    let engine = Engine;
    // Issue #370: negotiation once guarded only JXL, so
    // an Accept naming AVIF flattened an animation to
    // its first frame, and a negotiated WebP did the
    // same to a GIF source. Negotiated preferences now
    // keep the frames: AVIF falls back to the source
    // format, and GIF and WebP cross over as animations.
    let gif = animated_gif_bytes();
    let out = engine
        .process_preferred(
            &gif,
            &[],
            ProcessContext {
                preferred: Some(ImageFormat::Avif),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Gif);
    assert_eq!(decoded_gif_frames(&out.bytes).len(), 2);

    let out = engine
        .process_preferred(
            &gif,
            &[],
            ProcessContext {
                preferred: Some(ImageFormat::WebP),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::WebP);
    assert_eq!(decoded_webp_frames(&out.bytes).len(), 2);

    let webp = animated_webp_bytes();
    let out = engine
        .process_preferred(
            &webp,
            &[],
            ProcessContext {
                preferred: Some(ImageFormat::Avif),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::WebP);
    assert_eq!(decoded_webp_frames(&out.bytes).len(), 3);

    let out = engine
        .process_preferred(
            &webp,
            &[],
            ProcessContext {
                preferred: Some(ImageFormat::Gif),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Gif);
    assert_eq!(decoded_gif_frames(&out.bytes).len(), 3);
}

#[test]
fn animated_output_format_keeps_frames_under_negotiation_only() {
    // Issue #370's resolution, shared with the server's
    // animation gate (issue #405): explicit beats
    // negotiated, a negotiated GIF or WebP crosses over,
    // and anything else negotiated falls back to the
    // source format.
    use ImageFormat::*;
    let cases = [
        (Gif, None, None, Gif),
        (Gif, None, Some(WebP), WebP),
        (Gif, None, Some(Avif), Gif),
        (Gif, None, Some(Jxl), Gif),
        (Gif, None, Some(Mp4), Gif),
        (Gif, Some(WebP), None, WebP),
        (Gif, Some(Png), Some(WebP), Png),
        (Gif, Some(Mp4), Some(WebP), Mp4),
        (WebP, None, None, WebP),
        (WebP, None, Some(Gif), Gif),
        (WebP, None, Some(Avif), WebP),
        (WebP, Some(Gif), Some(WebP), Gif),
        (WebP, Some(Jxl), None, Jxl),
    ];
    for (source, explicit, preferred, want) in cases {
        assert_eq!(
            animated_output_format(source, explicit, preferred),
            want,
            "{source:?} {explicit:?} {preferred:?}"
        );
    }
}

/// A hand-built GIF of `frames` 1x1 frames over a
/// two-entry global palette, each frame the smallest
/// valid LZW stream (clear, pixel 0, end), then the
/// trailer: the base the probe's hostile cases cut up.
fn raw_gif(frames: usize) -> Vec<u8> {
    let mut out = b"GIF89a".to_vec();
    out.extend_from_slice(&[1, 0, 1, 0, 0x80, 0, 0]);
    out.extend_from_slice(&[0, 0, 0, 255, 255, 255]);
    for _ in 0..frames {
        out.extend_from_slice(&[0x2C, 0, 0, 0, 0, 1, 0, 1, 0, 0]);
        out.extend_from_slice(&[2, 2, 0x44, 0x01, 0]);
    }
    out.push(0x3B);
    out
}

#[test]
fn animated_gif_probe_counts_frames_like_the_decoder() {
    // Issue #405: the server routes GIF-to-WebP
    // renders to the animation gate on this probe, so
    // it must call a source animated exactly when the
    // decoder hands the engine more than one frame.
    let still = gif_bytes(&DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
        40,
        20,
        image::Rgba([255, 0, 0, 255]),
    )));
    for (name, bytes) in [
        ("raw 1", raw_gif(1)),
        ("raw 2", raw_gif(2)),
        ("raw 3", raw_gif(3)),
        ("animated", animated_gif_bytes()),
        ("still", still),
    ] {
        let frames = decode_gif_frames(&bytes).unwrap().len();
        assert_eq!(
            is_animated_gif(&bytes),
            frames > 1,
            "{name}: {frames} frames"
        );
    }
    assert!(!is_animated_gif(&animated_webp_bytes()));
    assert!(!is_animated_gif(&png_bytes(&DynamicImage::new_rgb8(4, 4))));
}

#[test]
fn animated_gif_probe_survives_hostile_input() {
    // The probe runs on untrusted source bytes before
    // any gate, on the async worker: malformed input
    // must read as not animated without panicking,
    // allocating pixels, or walking past the second
    // frame.
    for bytes in [
        &b""[..],
        b"GIF",
        b"GIF89",
        b"GIF89a",
        b"GIF89a\x01\x00",
        b"GIFxxx\0\0\0\0",
    ] {
        assert!(!is_animated_gif(bytes), "{bytes:?}");
    }
    // Every truncation of a two-frame file: no panic,
    // and not animated until the second frame's header
    // has arrived (after the file header, screen
    // descriptor, palette, and first frame: the second
    // frame's descriptor, code size, and first
    // sub-block length).
    let full = raw_gif(2);
    let second_frame_ready = 6 + 7 + 6 + 15 + 12;
    for len in 0..full.len() {
        assert_eq!(
            is_animated_gif(&full[..len]),
            len >= second_frame_ready,
            "prefix of {len} bytes"
        );
    }
    // A first-frame sub-block length that runs past the
    // end of the file swallows the second frame: the
    // walk hits the end, and the decoder refuses it.
    let mut overlong = raw_gif(2);
    overlong[6 + 7 + 6 + 11] = 0xFF;
    assert!(!is_animated_gif(&overlong));
    assert!(decode_gif_frames(&overlong).is_err());
    // No color table anywhere: the decoder refuses the
    // frame, and so does the probe.
    let mut no_palette = b"GIF89a".to_vec();
    no_palette.extend_from_slice(&[1, 0, 1, 0, 0, 0, 0]);
    for _ in 0..2 {
        no_palette.extend_from_slice(&[0x2C, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 2, 0x44, 0x01, 0]);
    }
    no_palette.push(0x3B);
    assert!(!is_animated_gif(&no_palette));
    assert!(decode_gif_frames(&no_palette).is_err());
    // An LZW code size past the format's maximum.
    let mut bad_code_size = raw_gif(2);
    bad_code_size[6 + 7 + 6 + 10] = 13;
    assert!(!is_animated_gif(&bad_code_size));
    // An unknown block type where a block should start.
    let mut unknown_block = raw_gif(2);
    unknown_block[6 + 7 + 6] = 0x99;
    assert!(!is_animated_gif(&unknown_block));
    assert!(decode_gif_frames(&unknown_block).is_err());
    // Megabytes of comment sub-blocks ahead of the
    // frames: the walk skips them without buffering and
    // still finds the frame count.
    let comments = |frames: usize| {
        let mut bytes = raw_gif(frames);
        let mut filler = vec![0x21, 0xFE];
        for _ in 0..16_384 {
            filler.push(255);
            filler.extend_from_slice(&[b'x'; 255]);
        }
        filler.push(0);
        bytes.splice(6 + 7 + 6..6 + 7 + 6, filler);
        bytes
    };
    assert!(!is_animated_gif(&comments(1)));
    assert!(is_animated_gif(&comments(2)));
    // A frame bomb is still only probed to its second
    // frame; the decoder's own cap refuses it later.
    assert!(is_animated_gif(&raw_gif(5_000)));
}

/// A fake video encoder for the video tests: drains
/// stdin and answers with the script's canned behavior.
fn video_test_encoder(script: &str) -> crate::video::VideoEncoder {
    use std::io::Write as _;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir =
        std::env::temp_dir().join(format!("crabcrop-engine-video-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("encoder.sh");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(script.as_bytes()).unwrap();
    drop(file);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    crate::video::VideoEncoder::new(path, std::time::Duration::from_secs(30), 64 * 1024 * 1024)
        .unwrap()
}

const FAKE_VIDEO_SCRIPT: &str = "#!/bin/sh\ncat > /dev/null\nprintf 'ENGINE-VIDEO'\n";

/// A fake frame extractor (issue #214): its probe
/// prints `report` on stderr, with the probe's
/// `-dump_separator` after each stream's codec name as
/// the reference tool prints it (issue #402), and its
/// extraction prints a real 8x6 PNG. Returns the
/// extractor (the script lives in a temp dir keyed by
/// `name`).
fn frame_test_extractor(name: &str, report: &str) -> crate::frames::VideoFrameExtractor {
    use std::io::Write as _;
    let dir = std::env::temp_dir().join(format!(
        "crabcrop-engine-frame-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("frame.png"),
        png_bytes(&DynamicImage::new_rgb8(8, 6)),
    )
    .unwrap();
    let frame_path = dir.join("frame.png").display().to_string();
    let script = format!(
        "#!/bin/sh\ncat > /dev/null\nsep=', '\nprev=\nfor a in \"$@\"; do\n  if [ \"$a\" = \"image2pipe\" ]; then cat \"{frame_path}\"; exit 0; fi\n  if [ \"$prev\" = -dump_separator ]; then sep=\"$a\"; fi\n  prev=\"$a\"\ndone\nprintf '%s\\n' {report} | sed \"/^  Stream #/s/, /$sep/\" >&2\nexit 1\n"
    );
    let path = dir.join("decoder.sh");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(script.as_bytes()).unwrap();
    drop(file);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    crate::frames::VideoFrameExtractor::new(path, std::time::Duration::from_secs(30)).unwrap()
}

/// Bytes that sniff as no image: an ISO-BMFF header
/// fragment is enough (the extractor is fake; the
/// container is never really parsed in these tests).
const FAKE_VIDEO_BYTES: &[u8] = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2avc1";

const FAKE_PROBE_REPORT: &str =
    "'  Duration: 00:00:03.00, start: 0.000000' '  Stream #0:0: Video: h264, yuv420p, 8x6, 10 fps'";

#[test]
fn frame_segment_without_a_decoder_is_not_configured() {
    let engine = Engine;
    let err = engine
        .process_preferred(
            FAKE_VIDEO_BYTES,
            &[Operation::Frame(crate::url::FrameSelection::Poster)],
            ProcessContext::default(),
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
    assert!(err.to_string().contains("CRABCROP_VIDEO_DECODER"), "{err}");
}

#[test]
fn frame_segment_on_an_image_source_is_a_decode_failure() {
    let engine = Engine;
    let extractor = frame_test_extractor("image-source", FAKE_PROBE_REPORT);
    let ctx = ProcessContext {
        frame_extractor: Some(&extractor),
        ..Default::default()
    };
    // A still PNG and an animated GIF: frame selection
    // is video-only, and animated image sources keep
    // their own handling, so both refuse as bad
    // sources rather than yielding a frame.
    let sources = [
        png_bytes(&DynamicImage::new_rgb8(20, 12)),
        animated_gif_bytes(),
    ];
    for src in sources {
        let err = engine
            .process_preferred(
                &src,
                &[Operation::Frame(crate::url::FrameSelection::Poster)],
                ctx.clone(),
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Decode(_)), "{err:?}");
        assert!(err.to_string().contains("needs a video source"), "{err}");
    }
}

#[test]
fn frame_segment_extracts_and_processes_the_frame() {
    let engine = Engine;
    let extractor = frame_test_extractor("extract", FAKE_PROBE_REPORT);
    let out = engine
        .process_preferred(
            FAKE_VIDEO_BYTES,
            &[
                Operation::Frame(crate::url::FrameSelection::Seconds(1.5)),
                size_op(Dimension::Pixels(4), Dimension::Pixels(3)),
            ],
            ProcessContext {
                frame_extractor: Some(&extractor),
                ..Default::default()
            },
        )
        .unwrap();
    // The operation chain ran on the extracted frame.
    assert_eq!((out.width, out.height), (4, 3));
    assert_eq!((out.source_width, out.source_height), (8, 6));
    // And the video record rides along for meta.
    let video = out.video.expect("a frame render carries the video record");
    assert_eq!(video.duration_seconds, Some(3.0));
    assert_eq!((video.width, video.height), (8, 6));
    assert_eq!(video.frame_seconds, 1.5);
}

#[test]
fn frame_segment_on_a_streamless_source_is_a_decode_failure() {
    let engine = Engine;
    let extractor = frame_test_extractor("streamless", "'no streams in here at all'");
    let err = engine
        .process_preferred(
            FAKE_VIDEO_BYTES,
            &[Operation::Frame(crate::url::FrameSelection::Poster)],
            ProcessContext {
                frame_extractor: Some(&extractor),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Decode(_)), "{err:?}");
    assert!(err.to_string().contains("no video stream"), "{err}");
}

#[test]
fn probe_source_labels_videos_and_never_mislabels_avif() {
    // The explain probe (issue #214): a video
    // container is kind "video" with unknown
    // dimensions; an AVIF still shares the ISO-BMFF
    // box and must keep its own kind.
    let probe = probe_source(FAKE_VIDEO_BYTES);
    assert_eq!(probe.kind, "video");
    assert_eq!(probe.dimensions, None);
    let webm = [0x1A, 0x45, 0xDF, 0xA3, 1, 2, 3, 4];
    assert_eq!(probe_source(&webm).kind, "video");
    let mut avif = Vec::new();
    avif.extend_from_slice(&24u32.to_be_bytes());
    avif.extend_from_slice(b"ftypavif");
    avif.extend_from_slice(&[0, 0, 0, 0]);
    avif.extend_from_slice(b"avifmif1");
    assert_eq!(probe_source(&avif).kind, "avif");
}

// -----------------------------------------------------
// TIFF sources (issue #313). The fixtures are written
// byte by byte: one strip, 8-bit channels, chunky
// planar layout, no compression, so the tests owe
// nothing to any encoder's habits.
// -----------------------------------------------------

fn tiff_put16(out: &mut Vec<u8>, v: u16, be: bool) {
    out.extend_from_slice(&if be { v.to_be_bytes() } else { v.to_le_bytes() });
}

fn tiff_put32(out: &mut Vec<u8>, v: u32, be: bool) {
    out.extend_from_slice(&if be { v.to_be_bytes() } else { v.to_le_bytes() });
}

/// A TIFF header pointing at a first IFD at offset 8.
fn tiff_header(be: bool) -> Vec<u8> {
    let mut out = Vec::new();
    if be {
        out.extend_from_slice(b"MM\0*");
    } else {
        out.extend_from_slice(b"II*\0");
    }
    tiff_put32(&mut out, 8, be);
    out
}

/// One page's IFD block (the IFD itself, the out-of-line
/// BitsPerSample array for RGB, then the strip pixels),
/// laid out to sit at absolute offset `ifd_at` in the
/// file, chaining to `next` (0 ends the chain). `samples`
/// is 3 (RGB, photometric 2) or 1 (gray, photometric 1).
fn tiff_page(
    w: u32,
    h: u32,
    samples: u16,
    pixels: &[u8],
    be: bool,
    ifd_at: u32,
    next: u32,
) -> Vec<u8> {
    // A SHORT value sits in the first two bytes of the
    // four-byte value field, in file byte order.
    let short_val = |v: u16| -> u32 { if be { u32::from(v) << 16 } else { u32::from(v) } };
    let bits_at = ifd_at + 2 + 12 * 10 + 4;
    let pixels_at = bits_at + if samples == 3 { 6 } else { 0 };
    let mut out = Vec::new();
    tiff_put16(&mut out, 10, be);
    let entry = |out: &mut Vec<u8>, tag: u16, kind: u16, count: u32, value: u32| {
        tiff_put16(out, tag, be);
        tiff_put16(out, kind, be);
        tiff_put32(out, count, be);
        tiff_put32(out, value, be);
    };
    entry(&mut out, 256, 4, 1, w);
    entry(&mut out, 257, 4, 1, h);
    if samples == 3 {
        entry(&mut out, 258, 3, 3, bits_at);
    } else {
        entry(&mut out, 258, 3, 1, short_val(8));
    }
    entry(&mut out, 259, 3, 1, short_val(1));
    entry(
        &mut out,
        262,
        3,
        1,
        short_val(if samples == 3 { 2 } else { 1 }),
    );
    entry(&mut out, 273, 4, 1, pixels_at);
    entry(&mut out, 277, 3, 1, short_val(samples));
    entry(&mut out, 278, 4, 1, h);
    entry(&mut out, 279, 4, 1, pixels.len() as u32);
    entry(&mut out, 284, 3, 1, short_val(1));
    tiff_put32(&mut out, next, be);
    if samples == 3 {
        for _ in 0..3 {
            tiff_put16(&mut out, 8, be);
        }
    }
    out.extend_from_slice(pixels);
    out
}

fn tiff_file(w: u32, h: u32, samples: u16, pixels: &[u8], be: bool) -> Vec<u8> {
    let mut out = tiff_header(be);
    out.extend_from_slice(&tiff_page(w, h, samples, pixels, be, 8, 0));
    out
}

/// Rewrite the value field of `tag` in the IFD at
/// `ifd_at`, and read fields back, for the hostile
/// variants below.
fn tiff_patch_tag(bytes: &mut [u8], ifd_at: usize, tag: u16, value: u32) {
    let be = &bytes[0..2] == b"MM";
    let put32 = |bytes: &mut [u8], at: usize, v: u32| {
        bytes[at..at + 4].copy_from_slice(&if be { v.to_be_bytes() } else { v.to_le_bytes() });
    };
    let count = {
        let raw: [u8; 2] = bytes[ifd_at..ifd_at + 2].try_into().unwrap();
        if be {
            u16::from_be_bytes(raw)
        } else {
            u16::from_le_bytes(raw)
        }
    };
    for i in 0..count as usize {
        let at = ifd_at + 2 + 12 * i;
        let raw: [u8; 2] = bytes[at..at + 2].try_into().unwrap();
        let found = if be {
            u16::from_be_bytes(raw)
        } else {
            u16::from_le_bytes(raw)
        };
        if found == tag {
            put32(bytes, at + 8, value);
            return;
        }
    }
    panic!("tag {tag} not in the fixture IFD");
}

#[test]
fn tiff_sources_decode_in_both_byte_orders() {
    // Little-endian RGB, four distinct pixels.
    let pixels: [u8; 12] = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
    let img = decode_image(&tiff_file(2, 2, 3, &pixels, false)).unwrap();
    assert_eq!((img.width(), img.height()), (2, 2));
    let rgb = img.to_rgb8();
    assert_eq!(rgb.get_pixel(0, 0).0, [255, 0, 0]);
    assert_eq!(rgb.get_pixel(1, 0).0, [0, 255, 0]);
    assert_eq!(rgb.get_pixel(0, 1).0, [0, 0, 255]);
    assert_eq!(rgb.get_pixel(1, 1).0, [255, 255, 0]);
    // The same file big-endian decodes identically.
    let img_be = decode_image(&tiff_file(2, 2, 3, &pixels, true)).unwrap();
    assert_eq!(img_be.to_rgb8(), rgb);
    // Grayscale too.
    let img = decode_image(&tiff_file(3, 1, 1, &[10, 128, 250], false)).unwrap();
    let rgb = img.to_rgb8();
    assert_eq!(rgb.get_pixel(0, 0).0, [10, 10, 10]);
    assert_eq!(rgb.get_pixel(2, 0).0, [250, 250, 250]);
}

#[test]
fn tiff_flows_through_the_pipeline() {
    let engine = Engine;
    let src = tiff_file(4, 2, 3, &[200, 30, 30].repeat(8), false);
    assert_eq!(probe_source(&src).kind, "tiff");
    let prepared = prepare_upload(&src).unwrap();
    assert_eq!(prepared.media, UploadMedia::Tiff);
    let out = engine
        .process(
            &src,
            &[
                Operation::Size(ImageSize {
                    width: Dimension::Pixels(2),
                    height: Dimension::Pixels(1),
                    flip_horizontal: false,
                    flip_vertical: false,
                }),
                Operation::Filters(vec![Filter::Format(FormatSpec::plain(ImageFormat::Png))]),
            ],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (2, 1));
    let decoded = image::load_from_memory(&out.bytes).unwrap();
    assert_eq!(decoded.to_rgb8().get_pixel(0, 0).0, [200, 30, 30]);
}

#[test]
fn tiff_multipage_renders_the_first_page() {
    let page1 = tiff_page(2, 1, 3, &[255, 0, 0, 0, 255, 0], false, 8, 0);
    let page2_at = 8 + page1.len() as u32;
    let mut src = tiff_header(false);
    src.extend_from_slice(&tiff_page(
        2,
        1,
        3,
        &[255, 0, 0, 0, 255, 0],
        false,
        8,
        page2_at,
    ));
    src.extend_from_slice(&tiff_page(1, 1, 3, &[0, 0, 255], false, page2_at, 0));
    let img = decode_image(&src).unwrap();
    assert_eq!((img.width(), img.height()), (2, 1));
    assert_eq!(img.to_rgb8().get_pixel(1, 0).0, [0, 255, 0]);
}

#[test]
fn tiff_hostile_inputs_fail_bounded_and_clean() {
    let good = tiff_file(2, 2, 3, &[1, 2, 3].repeat(4), false);
    // Announced dimensions past the decode bounds are
    // refused by the header probe, before any decode
    // allocation is sized by them.
    let mut huge = good.clone();
    tiff_patch_tag(&mut huge, 8, 256, 20_000);
    tiff_patch_tag(&mut huge, 8, 257, 20_000);
    assert!(decode_image(&huge).is_err());
    // A cyclic next-IFD chain (the first page points at
    // itself) is refused by the decoder itself, which
    // reports the cycle: a clean error, never a hang.
    let mut cyclic = good.clone();
    let next_at = 8 + 2 + 12 * 10;
    cyclic[next_at..next_at + 4].copy_from_slice(&8u32.to_le_bytes());
    let err = decode_image(&cyclic).unwrap_err();
    assert!(err.to_string().contains("cycle"), "{err}");
    // A strip that ends early is an error, not a panic
    // or a padded read.
    let truncated = &good[..good.len() - 5];
    assert!(decode_image(truncated).is_err());
    // A header with no IFD behind it, and random bytes
    // with the magic, fail the same way.
    assert!(decode_image(&tiff_header(false)).is_err());
    assert!(decode_image(b"II*\0").is_err());
    assert!(decode_image(&[]).is_err());
}

#[test]
fn frame_percentage_without_a_duration_is_unsupported() {
    let engine = Engine;
    let extractor = frame_test_extractor(
        "no-duration",
        "'  Duration: N/A, start: 0.000000' '  Stream #0:0: Video: h264, yuv420p, 8x6, 10 fps'",
    );
    let err = engine
        .process_preferred(
            FAKE_VIDEO_BYTES,
            &[Operation::Frame(crate::url::FrameSelection::Percent(50.0))],
            ProcessContext {
                frame_extractor: Some(&extractor),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("duration"), "{err}");
}

/// A fake PDF renderer pair (issue #215): the probe
/// prints a report naming `pages` pages, the renderer
/// prints a real 8x6 PNG. Returns the renderer (the
/// scripts live in a temp dir keyed by `name`).
fn pdf_test_renderer(name: &str, pages: u32) -> crate::pdf::PdfRenderer {
    use std::io::Write as _;
    let dir =
        std::env::temp_dir().join(format!("crabcrop-engine-pdf-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("page.png"),
        png_bytes(&DynamicImage::new_rgb8(8, 6)),
    )
    .unwrap();
    let page_path = dir.join("page.png").display().to_string();
    let write = |file: &str, script: String| {
        let path = dir.join(file);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(script.as_bytes()).unwrap();
        drop(f);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    };
    write(
        "pdfinfo",
        format!(
            "#!/bin/sh\ncat > /dev/null\nprintf '%s\\n' 'Pages:           {pages}' 'Encrypted:       no' 'Page size:       612 x 792 pts (letter)'\nexit 0\n"
        ),
    );
    let renderer = write(
        "pdftocairo",
        format!("#!/bin/sh\ncat > /dev/null\ncat \"{page_path}\"\nexit 0\n"),
    );
    crate::pdf::PdfRenderer::new(renderer, std::time::Duration::from_secs(30)).unwrap()
}

/// Bytes that carry the PDF header and sniff as no
/// image (the tools are fake; the document is never
/// really parsed in these tests).
const FAKE_PDF_BYTES: &[u8] = b"%PDF-1.4\nfake document bytes for the engine tests";

#[test]
fn page_segment_without_a_renderer_is_not_configured() {
    let engine = Engine;
    let err = engine
        .process_preferred(
            FAKE_PDF_BYTES,
            &[Operation::Page(1)],
            ProcessContext::default(),
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
    assert!(err.to_string().contains("CRABCROP_PDF_RENDERER"), "{err}");
}

#[test]
fn page_segment_on_a_non_pdf_source_is_a_decode_failure() {
    let engine = Engine;
    let renderer = pdf_test_renderer("non-pdf", 3);
    let ctx = ProcessContext {
        pdf_renderer: Some(&renderer),
        ..Default::default()
    };
    // A still PNG and a video fragment: page selection
    // is PDF-only, and both refuse as bad sources
    // before any tool spawns.
    let sources: [&[u8]; 2] = [
        &png_bytes(&DynamicImage::new_rgb8(20, 12)),
        FAKE_VIDEO_BYTES,
    ];
    for src in sources {
        let err = engine
            .process_preferred(src, &[Operation::Page(1)], ctx.clone())
            .unwrap_err();
        assert!(matches!(err, EngineError::Decode(_)), "{err:?}");
        assert!(err.to_string().contains("needs a PDF source"), "{err}");
    }
}

#[test]
fn page_segment_renders_and_processes_the_page() {
    let engine = Engine;
    let renderer = pdf_test_renderer("render", 3);
    let out = engine
        .process_preferred(
            FAKE_PDF_BYTES,
            &[
                Operation::Page(2),
                size_op(Dimension::Pixels(4), Dimension::Pixels(3)),
            ],
            ProcessContext {
                pdf_renderer: Some(&renderer),
                ..Default::default()
            },
        )
        .unwrap();
    // The operation chain ran on the rendered page.
    assert_eq!((out.width, out.height), (4, 3));
    assert_eq!((out.source_width, out.source_height), (8, 6));
    // And the PDF record rides along for meta.
    let pdf = out.pdf.expect("a page render carries the PDF record");
    assert_eq!(pdf.pages, 3);
    assert_eq!(pdf.page, 2);
    assert_eq!((pdf.page_width, pdf.page_height), (612.0, 792.0));
    // No video record on a PDF render.
    assert!(out.video.is_none());
}

#[test]
fn probe_source_labels_pdfs() {
    // The explain probe (issue #215): a PDF document
    // is kind "pdf" with unknown dimensions; the
    // header sniff keeps its window, so junk with a
    // late header is not a PDF.
    let probe = probe_source(FAKE_PDF_BYTES);
    assert_eq!(probe.kind, "pdf");
    assert_eq!(probe.dimensions, None);
    let mut late = vec![b' '; 2000];
    late.extend_from_slice(b"%PDF-1.4");
    assert_eq!(probe_source(&late).kind, "unknown");
}

#[test]
fn page_segment_out_of_range_is_unsupported() {
    let engine = Engine;
    let renderer = pdf_test_renderer("range", 3);
    let err = engine
        .process_preferred(
            FAKE_PDF_BYTES,
            &[Operation::Page(9)],
            ProcessContext {
                pdf_renderer: Some(&renderer),
                ..Default::default()
            },
        )
        .unwrap_err();
    // Out of range refuses; it never clamps to the
    // last page the way a frame timestamp clamps.
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("page 9"), "{err}");
    assert!(err.to_string().contains("3 pages"), "{err}");
}

#[test]
fn animated_gif_to_mp4_uses_the_configured_encoder() {
    let engine = Engine;
    let src = animated_gif_bytes();
    let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
    let out = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Mp4,
            ))])],
            ProcessContext {
                video_encoder: Some(&encoder),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Mp4);
    assert_eq!(out.bytes, b"ENGINE-VIDEO");
    assert_eq!((out.width, out.height), (40, 20));
    assert_eq!((out.source_width, out.source_height), (40, 20));
}

#[test]
fn animated_webp_to_webm_uses_the_configured_encoder() {
    let engine = Engine;
    let src = animated_webp_bytes();
    let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
    let out = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::WebM,
            ))])],
            ProcessContext {
                video_encoder: Some(&encoder),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::WebM);
    assert_eq!(out.bytes, b"ENGINE-VIDEO");
}

#[test]
fn video_output_runs_the_operations_per_frame() {
    let engine = Engine;
    let src = animated_gif_bytes();
    let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
    let out = engine
        .process_preferred(
            &src,
            &[
                size_op(Dimension::Pixels(20), Dimension::Pixels(10)),
                Operation::Filters(vec![Filter::Format(FormatSpec::plain(ImageFormat::Mp4))]),
            ],
            ProcessContext {
                video_encoder: Some(&encoder),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Mp4);
    assert_eq!((out.width, out.height), (20, 10));
    assert_eq!((out.source_width, out.source_height), (40, 20));
}

#[test]
fn video_format_on_a_still_source_is_a_specific_error() {
    let engine = Engine;
    let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
    let ctx = ProcessContext {
        video_encoder: Some(&encoder),
        ..Default::default()
    };
    // A still PNG, and a single-frame GIF: both stills,
    // both refused with the animated-source error, and
    // neither falls back to a first-frame image.
    let stills = [
        png_bytes(&DynamicImage::new_rgb8(20, 12)),
        gif_bytes(&DynamicImage::new_rgb8(20, 12)),
    ];
    for src in stills {
        let err = engine
            .process_preferred(
                &src,
                &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                    ImageFormat::Mp4,
                ))])],
                ctx.clone(),
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("animated source"), "{err}");
        assert!(err.to_string().contains("still image"), "{err}");
    }
}

#[test]
fn video_format_without_an_encoder_is_not_configured() {
    let engine = Engine;
    // Animated source: the deployment cannot do it.
    let err = engine
        .process(
            &animated_gif_bytes(),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Mp4,
            ))])],
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
    assert!(err.to_string().contains("CRABCROP_VIDEO_ENCODER"), "{err}");
    // Still source, same answer: the not-configured
    // preflight runs before the still check, so an
    // unconfigured server never blames the source.
    let err = engine
        .process(
            &png_bytes(&DynamicImage::new_rgb8(20, 12)),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::WebM,
            ))])],
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
}

#[test]
fn a_failing_encoder_is_a_video_error() {
    let engine = Engine;
    let encoder = video_test_encoder("#!/bin/sh\ncat > /dev/null\necho 'no codec' >&2\nexit 3\n");
    let err = engine
        .process_preferred(
            &animated_gif_bytes(),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Mp4,
            ))])],
            ProcessContext {
                video_encoder: Some(&encoder),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Video(_)), "{err:?}");
    assert!(err.to_string().contains("video encoder failed"), "{err}");
    assert!(err.to_string().contains("no codec"), "{err}");
}

/// Three 40x20 frames: red, green, blue, with delays of
/// 100, 200, and 300 ms. Built with the oxideav encoder
/// (the image crate's WebP encoder writes stills only);
/// frames are coded lossless so colors round-trip exactly.
fn animated_webp_bytes() -> Vec<u8> {
    let frame = |rgb: [u8; 3], delay_ms: u64| {
        let img = image::RgbaImage::from_pixel(40, 20, image::Rgba([rgb[0], rgb[1], rgb[2], 255]));
        let image = oxideav_webp::WebpImage::from_rgba8(40, 20, img.as_raw().clone()).unwrap();
        oxideav_webp::Frame::new(image, Some(std::time::Duration::from_millis(delay_ms)))
    };
    let frames = vec![
        frame([255, 0, 0], 100),
        frame([0, 200, 0], 200),
        frame([0, 0, 255], 300),
    ];
    let opts = oxideav_webp::EncodeOptions::default()
        .with_frame_mode(oxideav_webp::AnimFrameMode::Lossless);
    oxideav_webp::encode_animation(&frames, &opts).unwrap()
}

fn decoded_webp_frames(bytes: &[u8]) -> Vec<image::Frame> {
    use image::AnimationDecoder;
    let decoder = image::codecs::webp::WebPDecoder::new(Cursor::new(bytes)).unwrap();
    decoder.into_frames().collect_frames().unwrap()
}

fn frame_delay_ms(frame: &image::Frame) -> u64 {
    let duration: std::time::Duration = frame.delay().into();
    duration.as_millis() as u64
}

#[test]
fn animated_webp_fixture_decodes_to_its_frames() {
    let frames = decode_webp_frames(&animated_webp_bytes()).unwrap();
    assert_eq!(frames.len(), 3);
    assert!(
        frames
            .iter()
            .all(|f| f.buffer().width() == 40 && f.buffer().height() == 20)
    );
    let delays: Vec<u64> = frames.iter().map(frame_delay_ms).collect();
    assert_eq!(delays, [100, 200, 300]);
    assert_eq!(
        frames[0].buffer().get_pixel(0, 0),
        &image::Rgba([255, 0, 0, 255])
    );
    assert_eq!(
        frames[2].buffer().get_pixel(0, 0),
        &image::Rgba([0, 0, 255, 255])
    );
}

#[test]
fn animated_webp_keeps_its_frames_through_resize() {
    let engine = Engine;
    let src = animated_webp_bytes();
    let out = engine
        .process(
            &src,
            &[size_op(Dimension::Pixels(20), Dimension::Pixels(10))],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::WebP);
    let frames = decoded_webp_frames(&out.bytes);
    assert_eq!(frames.len(), 3);
    assert!(
        frames
            .iter()
            .all(|f| f.buffer().width() == 20 && f.buffer().height() == 10)
    );
    let delays: Vec<u64> = frames.iter().map(frame_delay_ms).collect();
    assert_eq!(delays, [100, 200, 300]);
    // Lossy frames (issue #323): the colors survive the
    // resize close to exact, within lossy tolerance.
    assert_pixel_near(frames[1].buffer().get_pixel(0, 0), [0, 200, 0, 255], 16);
}

#[test]
fn animated_webp_keeps_its_frames_through_grayscale() {
    let engine = Engine;
    let src = animated_webp_bytes();
    let out = engine
        .process(&src, &[Operation::Filters(vec![Filter::Grayscale])])
        .unwrap();
    assert_eq!(out.format, ImageFormat::WebP);
    let frames = decoded_webp_frames(&out.bytes);
    assert_eq!(frames.len(), 3);
    let delays: Vec<u64> = frames.iter().map(frame_delay_ms).collect();
    assert_eq!(delays, [100, 200, 300]);
    for frame in &frames {
        // Lossy encode (issue #323): the channels of a
        // grayscale pixel stay equal within lossy tolerance
        // rather than exactly.
        let p = frame.buffer().get_pixel(0, 0);
        assert!(p[0].abs_diff(p[1]) <= 8, "pixel {p:?}");
        assert!(p[1].abs_diff(p[2]) <= 8, "pixel {p:?}");
    }
}

/// Three 40x20 frames with distinct colors and alpha
/// values, delays of 100, 200, and 300 ms, and a loop
/// count of 3. Built with the oxideav encoder, like
/// [`animated_webp_bytes`], so the fixture's pixels and
/// timing are known exactly before the libwebp output
/// path under test ever runs.
fn animated_webp_alpha_loop_bytes() -> Vec<u8> {
    let frame = |rgba: [u8; 4], delay_ms: u64| {
        let img = image::RgbaImage::from_pixel(40, 20, image::Rgba(rgba));
        let image = oxideav_webp::WebpImage::from_rgba8(40, 20, img.as_raw().clone()).unwrap();
        oxideav_webp::Frame::new(image, Some(std::time::Duration::from_millis(delay_ms)))
    };
    let frames = vec![
        frame([255, 0, 0, 128], 100),
        frame([0, 200, 0, 64], 200),
        frame([0, 0, 255, 255], 300),
    ];
    let opts = oxideav_webp::EncodeOptions::default()
        .with_frame_mode(oxideav_webp::AnimFrameMode::Lossless)
        .with_loop_count(3);
    oxideav_webp::encode_animation(&frames, &opts).unwrap()
}

fn assert_pixel_near(actual: &image::Rgba<u8>, expected: [u8; 4], tol: u8) {
    for (channel, (a, e)) in actual.0.iter().zip(expected).enumerate() {
        let diff = a.abs_diff(e);
        assert!(
            diff <= tol,
            "channel {channel}: pixel {actual:?}, expected near {expected:?} (tol {tol})"
        );
    }
}

#[test]
fn animated_webp_output_preserves_frames_durations_alpha_and_loop_count() {
    let engine = Engine;
    let src = animated_webp_alpha_loop_bytes();
    assert_eq!(webp_animation_params(&src).map(|p| p.0), Some(3));
    let out = engine.process(&src, &[]).unwrap();
    assert_eq!(out.format, ImageFormat::WebP);
    // The loop count survives the libwebp mux exactly.
    assert_eq!(webp_animation_params(&out.bytes).map(|p| p.0), Some(3));
    let frames = decoded_webp_frames(&out.bytes);
    assert_eq!(frames.len(), 3);
    assert!(
        frames
            .iter()
            .all(|f| f.buffer().width() == 40 && f.buffer().height() == 20)
    );
    // Per-frame durations survive exactly, including the
    // last frame's 300 ms (not the 150 ms average of the
    // two before it).
    let delays: Vec<u64> = frames.iter().map(frame_delay_ms).collect();
    assert_eq!(delays, [100, 200, 300]);
    // Lossy at the default quality 80: colors and alpha
    // land close to the source frames, not exact.
    assert_pixel_near(frames[0].buffer().get_pixel(0, 0), [255, 0, 0, 128], 32);
    assert_pixel_near(frames[1].buffer().get_pixel(0, 0), [0, 200, 0, 64], 32);
    assert_pixel_near(frames[2].buffer().get_pixel(0, 0), [0, 0, 255, 255], 32);
}

#[test]
fn animated_webp_encode_honors_a_passed_deadline() {
    // Issue #322: the per-frame deadline check inside
    // the animated WebP encode hands back Timeout
    // instead of feeding every frame to libwebp when
    // the deadline has already passed.
    let delay = image::Delay::from_numer_denom_ms(100, 1);
    let frames = vec![
        image::Frame::from_parts(
            image::RgbaImage::from_pixel(40, 20, image::Rgba([255, 0, 0, 255])),
            0,
            0,
            delay,
        ),
        image::Frame::from_parts(
            image::RgbaImage::from_pixel(40, 20, image::Rgba([0, 0, 255, 255])),
            0,
            0,
            delay,
        ),
    ];
    let err = encode_animated_webp(
        frames,
        0,
        None,
        FormatOptions::default(),
        Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
    )
    .unwrap_err();
    assert_eq!(err, EngineError::Timeout);
}

#[test]
fn animated_webp_quality_100_is_lossless_and_default_is_quality_80() {
    let engine = Engine;
    let src = animated_webp_alpha_loop_bytes();
    let lossless = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Quality(Quality::Value(
                100,
            ))])],
        )
        .unwrap();
    let frames = decoded_webp_frames(&lossless.bytes);
    assert_eq!(frames.len(), 3);
    let delays: Vec<u64> = frames.iter().map(frame_delay_ms).collect();
    assert_eq!(delays, [100, 200, 300]);
    assert_eq!(webp_animation_params(&lossless.bytes).map(|p| p.0), Some(3));
    // Lossless means exact pixels, alpha included. The
    // check runs through the libwebp decoder: the image
    // crate's decoder composites a blended frame with a
    // rounding that can land one unit off (a [0, 0, 255,
    // 255] frame over a translucent one reads back as
    // 254), which is a decode-side artifact the spike
    // probe on issue #323 isolated from the encode.
    let anim = webp::AnimDecoder::new(&lossless.bytes).decode().unwrap();
    assert_eq!(anim.loop_count, 3);
    let decoded: Vec<image::RgbaImage> = anim
        .into_iter()
        .map(|f| {
            let img: DynamicImage = (&f).into();
            img.to_rgba8()
        })
        .collect();
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[0].get_pixel(0, 0), &image::Rgba([255, 0, 0, 128]));
    assert_eq!(decoded[1].get_pixel(0, 0), &image::Rgba([0, 200, 0, 64]));
    assert_eq!(decoded[2].get_pixel(0, 0), &image::Rgba([0, 0, 255, 255]));
    // An unspecified quality encodes exactly like an
    // explicit quality(80): the default is threaded
    // through, not dropped.
    let default = engine.process(&src, &[]).unwrap();
    let explicit = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Quality(Quality::Value(
                80,
            ))])],
        )
        .unwrap();
    assert_eq!(default.bytes, explicit.bytes);
    // The lossy default and the lossless encode of the
    // same frames are different bytes.
    assert_ne!(default.bytes, lossless.bytes);
}

#[test]
fn animated_webp_to_gif_keeps_its_frames() {
    let engine = Engine;
    let src = animated_webp_bytes();
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Gif,
            ))])],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Gif);
    let frames = decoded_gif_frames(&out.bytes);
    assert_eq!(frames.len(), 3);
    let delays: Vec<u64> = frames.iter().map(frame_delay_ms).collect();
    assert_eq!(delays, [100, 200, 300]);
}

#[test]
fn animated_webp_to_a_still_format_keeps_the_first_frame() {
    let engine = Engine;
    let src = animated_webp_bytes();
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Jpeg,
            ))])],
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Jpeg);
    // The first frame is solid red (JPEG is lossy: near,
    // not exact).
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    let p = img.get_pixel(0, 0);
    assert!(p[0] > 200 && p[1] < 60 && p[2] < 60, "pixel {p:?}");
}

#[test]
fn animated_webp_stays_out_of_jxl_negotiation() {
    let engine = Engine;
    let src = animated_webp_bytes();
    // A negotiated (preferred) JXL falls back to WebP so the
    // frames survive.
    let out = engine
        .process_preferred(
            &src,
            &[],
            ProcessContext {
                preferred: Some(ImageFormat::Jxl),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::WebP);
    assert_eq!(decoded_webp_frames(&out.bytes).len(), 3);
    // An explicit format(jxl) still converts, first frame only.
    let out = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Jxl,
            ))])],
            ProcessContext {
                preferred: Some(ImageFormat::Jxl),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Jxl);
}

#[test]
fn webp_frame_cap_rejects_frame_bombs() {
    // Build a one-frame animation, then restamp its ANMF
    // chunk 201 times (the RIFF size is the only other field
    // that changes). Encoding 201 frames with the VP8L
    // encoder just for this test would take minutes; the
    // decoder cannot tell the difference.
    let one = {
        let img = image::RgbaImage::from_pixel(4, 4, image::Rgba([9, 9, 9, 255]));
        let image = oxideav_webp::WebpImage::from_rgba8(4, 4, img.as_raw().clone()).unwrap();
        let frame = oxideav_webp::Frame::new(image, Some(std::time::Duration::from_millis(50)));
        let opts = oxideav_webp::EncodeOptions::default()
            .with_frame_mode(oxideav_webp::AnimFrameMode::Lossless);
        oxideav_webp::encode_animation(&[frame], &opts).unwrap()
    };
    assert_eq!(decode_webp_frames(&one).unwrap().len(), 1);
    let anmf_at = one
        .windows(4)
        .position(|w| w == b"ANMF")
        .expect("a one-frame animation has an ANMF chunk");
    let chunk_len = u32::from_le_bytes(one[anmf_at + 4..anmf_at + 8].try_into().unwrap()) as usize;
    let chunk_span = 8 + chunk_len + (chunk_len % 2);
    let mut bomb = one[..anmf_at].to_vec();
    for _ in 0..201 {
        bomb.extend_from_slice(&one[anmf_at..anmf_at + chunk_span]);
    }
    let riff_size = (bomb.len() - 8) as u32;
    bomb[4..8].copy_from_slice(&riff_size.to_le_bytes());
    assert!(decode_webp_frames(&bomb).is_err());
}

#[test]
fn placeholders_are_computed_only_on_request() {
    let engine = Engine;
    let src = png_bytes(&DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        100,
        60,
        image::Rgb([255, 0, 0]),
    )));
    let plain = engine.process(&src, &[]).unwrap();
    assert!(plain.placeholder.is_none());
    let out = engine
        .process_preferred(
            &src,
            &[size_op(Dimension::Pixels(50), Dimension::Pixels(30))],
            ProcessContext {
                placeholders: true,
                ..Default::default()
            },
        )
        .unwrap();
    let ph = out.placeholder.unwrap();
    assert_eq!(ph.dominant_color, "#ff0000");
    assert!(ph.data_uri.starts_with("data:image/jpeg;base64,"));
}

#[test]
fn heic_detection_reads_ftyp_brands() {
    assert!(is_heic(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1"));
    assert!(is_heic(b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00mif1heic"));
    // AVIF shares the BMFF container but not the brands.
    assert!(!is_heic(
        b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
    ));
    assert!(!is_heic(&[0u8; 4]));
    assert!(!is_heic(b"not an image at all"));
}

#[test]
fn heic_source_without_an_image_fails_with_a_specific_error() {
    // The ftyp brands alone are not an image: the HEIC decode
    // path fails, by name, instead of falling through to a
    // generic format error. The full decode behaviour lives
    // in the heic module's tests.
    let engine = Engine;
    let heic: &[u8] = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1....";
    let err = engine.process(heic, &[]).unwrap_err();
    match err {
        EngineError::Decode(msg) => assert!(msg.contains("HEIC"), "{msg}"),
        other => panic!("expected Decode, got {other:?}"),
    }
}

fn watermark_map() -> WatermarkImages {
    let mut map = WatermarkImages::new();
    map.insert(
        "wm".to_string(),
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 255]))),
    );
    map
}

fn red_source() -> Vec<u8> {
    png_bytes(&DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        20,
        20,
        image::Rgb([255, 0, 0]),
    )))
}

#[test]
fn watermark_overlays_and_blends() {
    let engine = Engine;
    let src = red_source();
    let map = watermark_map();
    let wm_op = |x: i32, y: i32, alpha: u8| {
        Operation::Filters(vec![Filter::Watermark {
            url: "wm".to_string(),
            x,
            y,
            alpha,
        }])
    };
    // Full opacity at the top-left corner.
    let out = engine
        .process_preferred(
            &src,
            &[wm_op(0, 0, 100)],
            ProcessContext {
                watermarks: Some(&map),
                ..Default::default()
            },
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    assert_eq!(img.get_pixel(0, 0), &image::Rgb([0, 0, 255]));
    assert_eq!(img.get_pixel(10, 10), &image::Rgb([255, 0, 0]));
    // Half opacity blends evenly.
    let out = engine
        .process_preferred(
            &src,
            &[wm_op(0, 0, 50)],
            ProcessContext {
                watermarks: Some(&map),
                ..Default::default()
            },
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    assert_eq!(img.get_pixel(0, 0), &image::Rgb([128, 0, 128]));
}

#[test]
fn watermark_negative_coordinates_anchor_from_the_edges() {
    let engine = Engine;
    let src = red_source();
    let map = watermark_map();
    let out = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![Filter::Watermark {
                url: "wm".to_string(),
                x: -2,
                y: -2,
                alpha: 100,
            }])],
            ProcessContext {
                watermarks: Some(&map),
                ..Default::default()
            },
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    // The 4x4 watermark lands at (14, 14): 20 - 4 - 2.
    assert_eq!(img.get_pixel(14, 14), &image::Rgb([0, 0, 255]));
    assert_eq!(img.get_pixel(17, 17), &image::Rgb([0, 0, 255]));
    assert_eq!(img.get_pixel(19, 19), &image::Rgb([255, 0, 0]));
    assert_eq!(img.get_pixel(13, 13), &image::Rgb([255, 0, 0]));
}

fn overlay_map() -> WatermarkImages {
    let mut map = WatermarkImages::new();
    map.insert(
        "ov".to_string(),
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            2,
            2,
            image::Rgb([200, 100, 50]),
        )),
    );
    map
}

fn overlay_base() -> Vec<u8> {
    png_bytes(&DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        4,
        4,
        image::Rgb([100, 150, 200]),
    )))
}

#[test]
fn overlay_blend_modes_match_hand_computed_values() {
    let engine = Engine;
    let map = overlay_map();
    // Base (100,150,200), overlay (200,100,50), full alpha.
    let cases = [
        (crate::url::BlendMode::Multiply, [78, 59, 39]),
        (crate::url::BlendMode::Screen, [222, 191, 211]),
        (crate::url::BlendMode::Overlay, [157, 127, 167]),
        (crate::url::BlendMode::Normal, [200, 100, 50]),
    ];
    for (mode, want) in cases {
        let out = engine
            .process_preferred(
                &overlay_base(),
                &[Operation::Filters(vec![Filter::Overlay {
                    url: "ov".to_string(),
                    x: 0,
                    y: 0,
                    alpha: 100,
                    blend: mode,
                    width: None,
                }])],
                ProcessContext {
                    watermarks: Some(&map),
                    ..Default::default()
                },
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        assert_eq!(img.get_pixel(0, 0), &image::Rgb(want), "mode {mode:?}");
        assert_eq!(img.get_pixel(3, 3), &image::Rgb([100, 150, 200]));
    }
    // Half opacity, normal mode: the plain average.
    let out = engine
        .process_preferred(
            &overlay_base(),
            &[Operation::Filters(vec![Filter::Overlay {
                url: "ov".to_string(),
                x: 0,
                y: 0,
                alpha: 50,
                blend: crate::url::BlendMode::Normal,
                width: None,
            }])],
            ProcessContext {
                watermarks: Some(&map),
                ..Default::default()
            },
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    assert_eq!(img.get_pixel(0, 0), &image::Rgb([150, 125, 125]));
}

#[test]
fn overlay_normal_mode_reproduces_watermark() {
    let engine = Engine;
    let map = overlay_map();
    let make_ctx = || ProcessContext {
        watermarks: Some(&map),
        ..Default::default()
    };
    let via_watermark = engine
        .process_preferred(
            &overlay_base(),
            &[Operation::Filters(vec![Filter::Watermark {
                url: "ov".to_string(),
                x: 1,
                y: 1,
                alpha: 80,
            }])],
            make_ctx(),
        )
        .unwrap();
    let via_overlay = engine
        .process_preferred(
            &overlay_base(),
            &[Operation::Filters(vec![Filter::Overlay {
                url: "ov".to_string(),
                x: 1,
                y: 1,
                alpha: 80,
                blend: crate::url::BlendMode::Normal,
                width: None,
            }])],
            make_ctx(),
        )
        .unwrap();
    assert_eq!(via_watermark.bytes, via_overlay.bytes);
}

#[test]
fn overlay_width_scales_relative_to_the_base() {
    let engine = Engine;
    let mut map = WatermarkImages::new();
    map.insert(
        "blue".to_string(),
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 255]))),
    );
    let out = engine
        .process_preferred(
            &red_source(),
            &[Operation::Filters(vec![Filter::Overlay {
                url: "blue".to_string(),
                x: -2,
                y: -2,
                alpha: 100,
                blend: crate::url::BlendMode::Normal,
                width: Some(50),
            }])],
            ProcessContext {
                watermarks: Some(&map),
                ..Default::default()
            },
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    // Scaled to 50% of 20 = 10 px, anchored at (8, 8).
    assert_eq!(img.get_pixel(8, 8), &image::Rgb([0, 0, 255]));
    assert_eq!(img.get_pixel(17, 17), &image::Rgb([0, 0, 255]));
    assert_eq!(img.get_pixel(18, 18), &image::Rgb([255, 0, 0]));
    assert_eq!(img.get_pixel(7, 7), &image::Rgb([255, 0, 0]));
}

#[test]
fn overlay_scaling_is_charged_against_the_pixel_budget() {
    // Issue #358: a 1x16384 overlay at 400 percent of a
    // 1000 px base scales to 4000x65,536,001, far past
    // the output pixel budget. The blend must refuse
    // before the resize allocates it (the unbounded
    // form aborted the process on allocation failure).
    let base = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        1000,
        10,
        image::Rgb([255, 0, 0]),
    ));
    let ov = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        1,
        16384,
        image::Rgb([0, 0, 255]),
    ));
    let ctx = ProcessContext::default();
    let err = blend_overlay(
        &ctx,
        &base,
        &ov,
        0,
        0,
        100,
        crate::url::BlendMode::Normal,
        Some(400),
    )
    .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err}");
    assert!(err.to_string().contains("pixel limit"), "{err}");
    // A scale inside the budget still blends.
    let small = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 255])));
    let out = blend_overlay(
        &ctx,
        &base,
        &small,
        0,
        0,
        100,
        crate::url::BlendMode::Normal,
        Some(10),
    )
    .unwrap();
    assert_eq!((out.width(), out.height()), (1000, 10));
}

#[test]
fn overlay_blend_honors_the_deadline() {
    // Issue #358: the per-pixel blend loop checks the
    // render deadline per row instead of running to
    // completion past it.
    let base =
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(64, 64, image::Rgb([255, 0, 0])));
    let ov = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(64, 64, image::Rgb([0, 0, 255])));
    let ctx = ProcessContext {
        deadline: Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
        ..Default::default()
    };
    let err = blend_overlay(
        &ctx,
        &base,
        &ov,
        0,
        0,
        100,
        crate::url::BlendMode::Normal,
        None,
    )
    .unwrap_err();
    assert!(matches!(err, EngineError::Timeout), "{err}");
}

#[test]
fn unresolved_overlay_is_an_error() {
    let engine = Engine;
    let err = engine
        .process(
            &overlay_base(),
            &[Operation::Filters(vec![Filter::Overlay {
                url: "missing".to_string(),
                x: 0,
                y: 0,
                alpha: 100,
                blend: crate::url::BlendMode::Normal,
                width: None,
            }])],
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("overlay was not resolved"),
        "got: {err}"
    );
}

#[test]
fn unresolved_watermark_is_an_error() {
    let engine = Engine;
    let src = red_source();
    let err = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Watermark {
                url: "missing".to_string(),
                x: 0,
                y: 0,
                alpha: 100,
            }])],
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
}

/// A mask provider with a fixed answer: the left half is
/// background, the right half is subject.
#[derive(Debug)]
struct HalfMask;

impl crate::ai::MaskProvider for HalfMask {
    fn mask(&self, img: &DynamicImage) -> Result<image::GrayImage, crate::ai::AiError> {
        Ok(image::GrayImage::from_fn(
            img.width(),
            img.height(),
            |x, _| image::Luma([if x < img.width() / 2 { 0 } else { 255 }]),
        ))
    }
}

#[test]
fn removebg_applies_the_provider_mask() {
    let engine = Engine;
    let src = red_source();
    let out = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![Filter::RemoveBg])],
            ProcessContext {
                mask_provider: Some(&HalfMask),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Png);
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
    assert_eq!(img.get_pixel(2, 10)[3], 0);
    assert_eq!(img.get_pixel(17, 10)[3], 255);
    // The subject's color survives untouched.
    assert_eq!(img.get_pixel(17, 10)[0], 255);
}

#[test]
fn removebg_rejects_alphaless_output() {
    let engine = Engine;
    let ctx = || ProcessContext {
        mask_provider: Some(&HalfMask),
        ..Default::default()
    };
    // An explicit JPEG format conflicts.
    let err = engine
        .process_preferred(
            &red_source(),
            &[Operation::Filters(vec![
                Filter::RemoveBg,
                Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
            ])],
            ctx(),
        )
        .unwrap_err();
    match &err {
        EngineError::Unsupported(msg) => assert!(msg.contains("alpha"), "{msg}"),
        other => panic!("expected Unsupported, got {other:?}"),
    }
    // So does a JPEG source with no format named: the
    // resolved output is JPEG.
    let mut jpeg = Vec::new();
    DynamicImage::ImageRgb8(image::RgbImage::from_pixel(20, 20, image::Rgb([255, 0, 0])))
        .write_to(
            &mut std::io::Cursor::new(&mut jpeg),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
    let err = engine
        .process_preferred(&jpeg, &[Operation::Filters(vec![Filter::RemoveBg])], ctx())
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
}

#[test]
fn removebg_without_a_provider_is_a_clean_error() {
    let engine = Engine;
    // PNG source: the format check passes, the provider
    // check is the one that fires.
    let err = engine
        .process(&red_source(), &[Operation::Filters(vec![Filter::RemoveBg])])
        .unwrap_err();
    match &err {
        EngineError::NotConfigured(msg) => {
            assert!(msg.contains("not configured"), "{msg}")
        }
        other => panic!("expected NotConfigured, got {other:?}"),
    }
}

/// A mask provider for studio tests: the central square
/// is the subject, and it exposes a confidence, so the
/// record's plumbing for both is exercised.
#[derive(Debug)]
struct SquareMask;

impl crate::ai::MaskProvider for SquareMask {
    fn mask(&self, img: &DynamicImage) -> Result<image::GrayImage, crate::ai::AiError> {
        let (w, h) = (img.width(), img.height());
        Ok(image::GrayImage::from_fn(w, h, |x, y| {
            let inside = (w / 4..3 * w / 4).contains(&x) && (h / 4..3 * h / 4).contains(&y);
            image::Luma([if inside { 255 } else { 0 }])
        }))
    }

    fn confidence(&self, _img: &DynamicImage, _mask: &image::GrayImage) -> Option<f32> {
        Some(0.93)
    }
}

/// A 40x40 "product photo": a red square at 10..30 on
/// green, as PNG bytes.
fn product_source() -> Vec<u8> {
    let img = DynamicImage::ImageRgba8(image::RgbaImage::from_fn(40, 40, |x, y| {
        if (10..30).contains(&x) && (10..30).contains(&y) {
            image::Rgba([200, 30, 30, 255])
        } else {
            image::Rgba([30, 160, 30, 255])
        }
    }));
    let mut bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Png,
    )
    .unwrap();
    bytes
}

fn studio_spec() -> crate::url::StudioSpec {
    crate::url::StudioSpec {
        mode: crate::url::StudioMode::Product,
        bg: crate::url::StudioBg::White,
        shadow: crate::url::StudioShadow::None,
        pad: 10,
    }
}

#[test]
fn studio_without_a_provider_is_a_clean_error() {
    let engine = Engine;
    let err = engine
        .process(
            &product_source(),
            &[Operation::Filters(vec![Filter::Studio(studio_spec())])],
        )
        .unwrap_err();
    match &err {
        EngineError::NotConfigured(msg) => {
            assert!(msg.contains("studio is not configured"), "{msg}")
        }
        other => panic!("expected NotConfigured, got {other:?}"),
    }
}

#[test]
fn studio_composites_and_records_the_run() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &product_source(),
            &[Operation::Filters(vec![
                Filter::Studio(studio_spec()),
                Filter::Format(FormatSpec::plain(ImageFormat::Png)),
            ])],
            ProcessContext {
                mask_provider: Some(&SquareMask),
                ..Default::default()
            },
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
    // White backdrop at the corners, the red product in
    // the middle: the composite, not the green photo.
    assert_eq!(img.get_pixel(0, 0).0, [255, 255, 255, 255]);
    assert_eq!(img.get_pixel(20, 20).0, [200, 30, 30, 255]);
    // The record describes the run for meta.
    let record = out.studio.expect("a studio run records itself");
    assert_eq!(record.bg, Some([255, 255, 255]));
    assert_eq!(record.bg_from, "argument");
    assert_eq!(record.shadow, crate::url::StudioShadow::None);
    assert_eq!(record.pad, 10);
    assert_eq!(record.steps, ["mask", "defringe", "backdrop", "pad"]);
    assert!(record.fallbacks.is_empty());
    assert_eq!(record.confidence, Some(0.93));
}

#[test]
fn studio_scene_without_a_provider_is_a_clean_error() {
    // Scene runs the same mask model as product
    // (issue #198), so the preflight refusal is the
    // same one.
    let engine = Engine;
    let err = engine
        .process(
            &product_source(),
            &[Operation::Filters(vec![Filter::Studio(
                crate::url::StudioSpec {
                    mode: crate::url::StudioMode::Scene,
                    ..crate::url::StudioSpec::default()
                },
            )])],
        )
        .unwrap_err();
    match &err {
        EngineError::NotConfigured(msg) => {
            assert!(msg.contains("studio is not configured"), "{msg}")
        }
        other => panic!("expected NotConfigured, got {other:?}"),
    }
}

#[test]
fn studio_scene_reframes_and_records_the_run() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &product_source(),
            &[Operation::Filters(vec![
                Filter::Studio(crate::url::StudioSpec {
                    mode: crate::url::StudioMode::Scene,
                    ..crate::url::StudioSpec::default()
                }),
                Filter::Format(FormatSpec::plain(ImageFormat::Png)),
            ])],
            ProcessContext {
                mask_provider: Some(&SquareMask),
                ..Default::default()
            },
        )
        .unwrap();
    // The canvas in effect never changes: the 40x40
    // photo comes back 40x40, reframed on the mask's
    // central square and corrected.
    let img = image::load_from_memory(&out.bytes).unwrap();
    assert_eq!((img.width(), img.height()), (40, 40));
    let record = out.studio.expect("a studio run records itself");
    assert_eq!(record.mode, crate::url::StudioMode::Scene);
    assert_eq!(record.steps, ["mask", "crop", "correct"]);
    assert!(record.fallbacks.is_empty());
    assert_eq!(record.confidence, Some(0.93));
    let srec = record.scene.expect("a scene run records its facts");
    assert_eq!(srec.subject, Some((10, 10, 20, 20)));
    assert_eq!(srec.subjects, 1);
    assert_eq!(srec.crop, (8, 8, 25, 25));
}

#[test]
fn studio_output_is_a_normal_image_in_any_format() {
    // Unlike removebg, whose output is transparency, a
    // studio composite flattens onto its backdrop, so a
    // JPEG output is fine.
    let engine = Engine;
    let out = engine
        .process_preferred(
            &product_source(),
            &[Operation::Filters(vec![
                Filter::Studio(studio_spec()),
                Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
            ])],
            ProcessContext {
                mask_provider: Some(&SquareMask),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.format, ImageFormat::Jpeg);
}

#[test]
fn studio_resolves_the_brand_color_from_context() {
    let engine = Engine;
    let brand = [245, 240, 232];
    let out = engine
        .process_preferred(
            &product_source(),
            &[Operation::Filters(vec![Filter::Studio(
                crate::url::StudioSpec::default(),
            )])],
            ProcessContext {
                mask_provider: Some(&SquareMask),
                brand_color: Some(brand),
                ..Default::default()
            },
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
    assert_eq!(img.get_pixel(0, 0).0, [brand[0], brand[1], brand[2], 255]);
    let record = out.studio.unwrap();
    assert_eq!(record.bg, Some(brand));
    assert_eq!(record.bg_from, "brand");
    // The default shadow (soft) ran over the backdrop.
    assert!(record.steps.contains(&"shadow"));
}

#[test]
fn studio_counts_once_against_the_operation_cap() {
    let engine = Engine;
    let ctx = || ProcessContext {
        mask_provider: Some(&SquareMask),
        ..Default::default()
    };
    // Studio plus MAX_OPERATIONS - 1 cheap filters is
    // exactly at the cap and processes.
    let mut at_cap = vec![Filter::Studio(studio_spec())];
    at_cap.extend((0..MAX_OPERATIONS - 1).map(|_| Filter::Invert));
    let out = engine.process_preferred(&product_source(), &[Operation::Filters(at_cap)], ctx());
    assert!(out.is_ok(), "{out:?}");
    // One more filter tips it over.
    let mut over_cap = vec![Filter::Studio(studio_spec())];
    over_cap.extend((0..MAX_OPERATIONS).map(|_| Filter::Invert));
    let err = engine
        .process_preferred(&product_source(), &[Operation::Filters(over_cap)], ctx())
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("operation limit"), "{err}");
}

/// A fake inpainting provider: fills the masked pixels
/// with one flat color, the deterministic stand-in for
/// the model's fill.
#[derive(Debug)]
struct FlatInpaint;

impl crate::ai::InpaintProvider for FlatInpaint {
    fn inpaint(
        &self,
        img: &DynamicImage,
        mask: &image::GrayImage,
    ) -> Result<DynamicImage, crate::ai::AiError> {
        let mut rgb = img.to_rgb8();
        for (x, y, m) in mask.enumerate_pixels() {
            if m[0] > 0 {
                *rgb.get_pixel_mut(x, y) = image::Rgb([30, 30, 200]);
            }
        }
        Ok(DynamicImage::ImageRgb8(rgb))
    }
}

/// A fake inpainting provider whose inference fails.
#[derive(Debug)]
struct FailingInpaint;

impl crate::ai::InpaintProvider for FailingInpaint {
    fn inpaint(
        &self,
        _img: &DynamicImage,
        _mask: &image::GrayImage,
    ) -> Result<DynamicImage, crate::ai::AiError> {
        Err(crate::ai::AiError::Inference(
            "the model fell over".to_string(),
        ))
    }
}

#[test]
fn remove_region_fills_the_rectangle_and_records_it() {
    let engine = Engine;
    let src = solid_png(100, 80, [200, 30, 30]);
    let out = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![Filter::RemoveRegion {
                x: 10,
                y: 10,
                w: 20,
                h: 20,
            }])],
            ProcessContext {
                inpaint_provider: Some(&FlatInpaint),
                ..Default::default()
            },
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    // Inside the rectangle: the fill. Outside: original.
    assert_eq!(*img.get_pixel(20, 20), image::Rgb([30, 30, 200]));
    assert_eq!(*img.get_pixel(9, 9), image::Rgb([200, 30, 30]));
    assert_eq!(*img.get_pixel(30, 30), image::Rgb([200, 30, 30]));
    // The meta record carries the rectangle.
    let record = out.inpaint.unwrap();
    assert_eq!(record.regions, vec![(10, 10, 20, 20)]);
}

#[test]
fn remove_region_records_every_removal_in_chain_order() {
    let engine = Engine;
    let src = solid_png(100, 80, [200, 30, 30]);
    let out = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![
                Filter::RemoveRegion {
                    x: 0,
                    y: 0,
                    w: 10,
                    h: 10,
                },
                Filter::RemoveRegion {
                    x: 50,
                    y: 50,
                    w: 10,
                    h: 10,
                },
            ])],
            ProcessContext {
                inpaint_provider: Some(&FlatInpaint),
                ..Default::default()
            },
        )
        .unwrap();
    let record = out.inpaint.unwrap();
    assert_eq!(record.regions, vec![(0, 0, 10, 10), (50, 50, 10, 10)]);
}

#[test]
fn remove_region_without_a_provider_is_not_configured() {
    let engine = Engine;
    let src = solid_png(100, 80, [200, 30, 30]);
    let err = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![Filter::RemoveRegion {
                x: 0,
                y: 0,
                w: 10,
                h: 10,
            }])],
            ProcessContext::default(),
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
    assert!(
        err.to_string()
            .contains("remove_region is not configured on this server"),
        "{err}"
    );
}

#[test]
fn remove_region_beyond_bounds_fails_like_the_other_region_filters() {
    let engine = Engine;
    let src = solid_png(100, 80, [200, 30, 30]);
    let err = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![Filter::RemoveRegion {
                x: 90,
                y: 70,
                w: 20,
                h: 20,
            }])],
            ProcessContext {
                inpaint_provider: Some(&FlatInpaint),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    assert!(
        err.to_string()
            .contains("remove_region(90,70,20,20) exceeds image bounds 100x80"),
        "{err}"
    );
}

#[test]
fn remove_region_addresses_the_image_at_its_point_in_the_chain() {
    // After a resize to 50x40, a rectangle legal on the
    // 100x80 source overflows the image in effect, and
    // one fitted to the resized image succeeds: the
    // region filters' shared coordinate rule.
    let engine = Engine;
    let src = solid_png(100, 80, [200, 30, 30]);
    let ctx = || ProcessContext {
        inpaint_provider: Some(&FlatInpaint),
        ..Default::default()
    };
    let err = engine
        .process_preferred(
            &src,
            &[
                size_op(Dimension::Pixels(50), Dimension::Pixels(40)),
                Operation::Filters(vec![Filter::RemoveRegion {
                    x: 40,
                    y: 30,
                    w: 20,
                    h: 20,
                }]),
            ],
            ctx(),
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("remove_region(40,30,20,20) exceeds image bounds 50x40"),
        "{err}"
    );
    let out = engine
        .process_preferred(
            &src,
            &[
                size_op(Dimension::Pixels(50), Dimension::Pixels(40)),
                Operation::Filters(vec![Filter::RemoveRegion {
                    x: 0,
                    y: 0,
                    w: 50,
                    h: 40,
                }]),
            ],
            ctx(),
        )
        .unwrap();
    assert_eq!((out.width, out.height), (50, 40));
    assert_eq!(out.inpaint.unwrap().regions, vec![(0, 0, 50, 40)]);
}

#[test]
fn remove_region_provider_failure_is_an_inference_error() {
    let engine = Engine;
    let src = solid_png(100, 80, [200, 30, 30]);
    let err = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![Filter::RemoveRegion {
                x: 0,
                y: 0,
                w: 10,
                h: 10,
            }])],
            ProcessContext {
                inpaint_provider: Some(&FailingInpaint),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Inference(_)), "{err:?}");
    assert!(err.to_string().contains("the model fell over"), "{err}");
}

/// A fake upscale provider: a deterministic 2x
/// nearest-neighbor enlargement.
#[derive(Debug)]
struct DoubleUpscale;

impl crate::ai::UpscaleProvider for DoubleUpscale {
    fn factor(&self) -> u32 {
        2
    }

    fn upscale(&self, img: &DynamicImage) -> Result<DynamicImage, crate::ai::AiError> {
        Ok(DynamicImage::ImageRgba8(image::imageops::resize(
            &img.to_rgba8(),
            img.width() * 2,
            img.height() * 2,
            image::imageops::FilterType::Nearest,
        )))
    }
}

/// A fake upscale provider that takes its time and
/// counts its runs: the deadline tests use it to make
/// the clock pass mid-segment and to observe how far
/// the segment got.
#[derive(Debug, Default)]
struct SlowCountingUpscale {
    calls: std::sync::atomic::AtomicUsize,
}

impl crate::ai::UpscaleProvider for SlowCountingUpscale {
    fn factor(&self) -> u32 {
        2
    }

    fn upscale(&self, img: &DynamicImage) -> Result<DynamicImage, crate::ai::AiError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(300));
        Ok(DynamicImage::ImageRgba8(image::imageops::resize(
            &img.to_rgba8(),
            img.width() * 2,
            img.height() * 2,
            image::imageops::FilterType::Nearest,
        )))
    }
}

#[test]
fn a_filters_segment_checks_the_deadline_between_filters() {
    // The deadline passes while the segment's first
    // (slow) filter runs. The per-operation check alone
    // cannot stop the segment: it is one operation, so
    // without a check inside it the second filter runs
    // too, and the request only fails later, at an
    // encode check if there is one. With the check, the
    // segment stops before the second filter: one
    // provider run, and a Timeout error.
    let engine = Engine;
    let src = red_source();
    let provider = SlowCountingUpscale::default();
    let ctx = ProcessContext {
        upscale_provider: Some(&provider),
        deadline: Some(std::time::Instant::now() + std::time::Duration::from_millis(100)),
        ..Default::default()
    };
    let err = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![
                Filter::Upscale(2),
                Filter::Upscale(2),
            ])],
            ctx,
        )
        .expect_err("the segment must stop at the deadline");
    assert_eq!(err, EngineError::Timeout);
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn upscale_doubles_and_quadruples_dimensions() {
    let engine = Engine;
    let src = red_source(); // 20x20
    let ctx = || ProcessContext {
        upscale_provider: Some(&DoubleUpscale),
        ..Default::default()
    };
    let out = engine
        .process_preferred(&src, &[Operation::Filters(vec![Filter::Upscale(2)])], ctx())
        .unwrap();
    assert_eq!((out.width, out.height), (40, 40));
    // 4x from a native-2x model is two runs.
    let out = engine
        .process_preferred(&src, &[Operation::Filters(vec![Filter::Upscale(4)])], ctx())
        .unwrap();
    assert_eq!((out.width, out.height), (80, 80));
}

#[test]
fn upscale_respects_the_output_cap() {
    let engine = Engine;
    let src = png_bytes(&DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        2100,
        10,
        image::Rgb([9, 9, 9]),
    )));
    let err = engine
        .process_preferred(
            &src,
            &[Operation::Filters(vec![Filter::Upscale(2)])],
            ProcessContext {
                upscale_provider: Some(&DoubleUpscale),
                ..Default::default()
            },
        )
        .unwrap_err();
    match &err {
        EngineError::Unsupported(msg) => assert!(msg.contains("4096"), "{msg}"),
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

/// A 2x provider that counts its runs, so the budget
/// tests can prove a rejection lands before any
/// inference happens.
#[derive(Debug, Default)]
struct CountingUpscale {
    runs: std::sync::atomic::AtomicUsize,
}

impl CountingUpscale {
    fn runs(&self) -> usize {
        self.runs.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl crate::ai::UpscaleProvider for CountingUpscale {
    fn factor(&self) -> u32 {
        2
    }

    fn upscale(&self, img: &DynamicImage) -> Result<DynamicImage, crate::ai::AiError> {
        self.runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(DynamicImage::ImageRgba8(image::imageops::resize(
            &img.to_rgba8(),
            img.width() * 2,
            img.height() * 2,
            image::imageops::FilterType::Nearest,
        )))
    }
}

fn upscale_attempt(
    provider: &CountingUpscale,
    w: u32,
    h: u32,
    factor: u32,
) -> Result<ProcessedImage, EngineError> {
    let src = png_bytes(&DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        w,
        h,
        image::Rgb([9, 9, 9]),
    )));
    Engine.process_preferred(
        &src,
        &[Operation::Filters(vec![Filter::Upscale(factor)])],
        ProcessContext {
            upscale_provider: Some(provider),
            ..Default::default()
        },
    )
}

#[test]
fn upscale_rejects_an_input_over_the_budget_before_any_inference() {
    // The issue #244 reproduction's shape: 900x560 in,
    // 504,000 pixels, which OOM-killed the server when
    // nothing stood between the request and the model.
    let provider = CountingUpscale::default();
    let err = upscale_attempt(&provider, 900, 560, 2).unwrap_err();
    match &err {
        EngineError::Unsupported(msg) => {
            assert!(msg.contains("504000"), "{msg}");
            assert!(msg.contains("262144"), "{msg}");
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
    assert_eq!(provider.runs(), 0, "no model run may start");
}

#[test]
fn upscale_budget_is_inclusive_at_the_cap() {
    // Exactly at the budget (512x512 = 262,144 pixels)
    // runs, like every cap in the engine; one row over
    // it does not. The campaign's working 480x360 run
    // (172,800 pixels) sits comfortably under it.
    let provider = CountingUpscale::default();
    let out = upscale_attempt(&provider, 512, 512, 2).unwrap();
    assert_eq!((out.width, out.height), (1024, 1024));
    assert_eq!(provider.runs(), 1);
    let provider = CountingUpscale::default();
    let out = upscale_attempt(&provider, 480, 360, 2).unwrap();
    assert_eq!((out.width, out.height), (960, 720));
    let provider = CountingUpscale::default();
    assert!(upscale_attempt(&provider, 512, 513, 2).is_err());
    assert_eq!(provider.runs(), 0);
}

#[test]
fn upscale_budgets_every_composed_run_upfront() {
    // The campaign's composed upscale(4): 256x192 in,
    // runs of 49,152 and 196,608 pixels, both under
    // the budget, landing exactly 1024x768.
    let provider = CountingUpscale::default();
    let out = upscale_attempt(&provider, 256, 192, 4).unwrap();
    assert_eq!((out.width, out.height), (1024, 768));
    assert_eq!(provider.runs(), 2);
    // A 300x300 upscale(4) passes a lone x2 budget
    // check (90,000 pixels) but its second run would
    // take 360,000: the projection refuses it before
    // the first run starts.
    let provider = CountingUpscale::default();
    let err = upscale_attempt(&provider, 300, 300, 4).unwrap_err();
    match &err {
        EngineError::Unsupported(msg) => assert!(msg.contains("360000"), "{msg}"),
        other => panic!("expected Unsupported, got {other:?}"),
    }
    assert_eq!(provider.runs(), 0);
}

#[test]
fn upscale_without_a_provider_is_a_clean_error() {
    let engine = Engine;
    let err = engine
        .process(
            &red_source(),
            &[Operation::Filters(vec![Filter::Upscale(2)])],
        )
        .unwrap_err();
    match &err {
        EngineError::NotConfigured(msg) => {
            assert!(msg.contains("not configured"), "{msg}")
        }
        other => panic!("expected NotConfigured, got {other:?}"),
    }
}

/// A face provider reporting a face over the right half.
#[derive(Debug)]
struct RightFace;

impl crate::ai::FaceProvider for RightFace {
    fn detect(&self, img: &DynamicImage) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
        Ok(vec![crate::ai::FaceBox {
            x: img.width() / 2,
            y: 0,
            w: img.width() / 2,
            h: img.height(),
            score: 0.95,
        }])
    }
}

#[test]
fn smart_crop_lands_on_a_detected_face() {
    let engine = Engine;
    // Left half red, right half blue; the face is on the
    // right. A 1:2 smart crop (50x100 out of 100x100) must
    // take the right half wholesale.
    let mut img = image::RgbImage::new(100, 100);
    for (x, _y, p) in img.enumerate_pixels_mut() {
        *p = if x < 50 {
            image::Rgb([255, 0, 0])
        } else {
            image::Rgb([0, 0, 255])
        };
    }
    let src = png_bytes(&DynamicImage::ImageRgb8(img));
    let out = engine
        .process_preferred(
            &src,
            &[Operation::Crop {
                size: ImageSize {
                    width: Dimension::Pixels(50),
                    height: Dimension::Pixels(100),
                    flip_horizontal: false,
                    flip_vertical: false,
                },
                align: Align::Smart,
            }],
            ProcessContext {
                face_provider: Some(&RightFace),
                detect_provider: None,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!((out.width, out.height), (50, 100));
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    assert_eq!(result.get_pixel(25, 50), &image::Rgb([0, 0, 255]));
}

/// The known face of `face_scene`: box (140, 30, 40, 40).
fn known_face() -> crate::ai::FaceBox {
    crate::ai::FaceBox {
        x: 140,
        y: 30,
        w: 40,
        h: 40,
        score: 0.95,
    }
}

#[derive(Debug)]
struct KnownFace;

impl crate::ai::FaceProvider for KnownFace {
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
        Ok(vec![known_face()])
    }
}

#[derive(Debug, Default)]
struct CountingFace {
    calls: std::sync::atomic::AtomicUsize,
}

impl crate::ai::FaceProvider for CountingFace {
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(vec![known_face()])
    }
}

/// Reports several faces under 20 px inside the checkerboard
/// of `face_scene`, like a detector on a group photo or after
/// a downscale (issue #401).
#[derive(Debug)]
struct SmallFaces;

const SMALL_FACES: [(u32, u32, u32, u32); 3] =
    [(141, 31, 8, 6), (152, 40, 12, 12), (160, 52, 19, 15)];

impl crate::ai::FaceProvider for SmallFaces {
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
        Ok(SMALL_FACES
            .iter()
            .map(|&(x, y, w, h)| crate::ai::FaceBox {
                x,
                y,
                w,
                h,
                score: 0.9,
            })
            .collect())
    }
}

#[derive(Debug)]
struct FailingFace;

impl crate::ai::FaceProvider for FailingFace {
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
        Err(crate::ai::AiError::Inference(
            "detector exploded".to_string(),
        ))
    }
}

/// The class-redaction scene's vocabulary: person on
/// the left, car on the right.
#[derive(Debug)]
struct KnownObjects;

fn known_detections() -> Vec<crate::ai::Detection> {
    vec![
        crate::ai::Detection {
            label: "person".to_string(),
            score: 0.9,
            x: 10,
            y: 10,
            w: 60,
            h: 80,
        },
        crate::ai::Detection {
            label: "car".to_string(),
            score: 0.8,
            x: 120,
            y: 10,
            w: 60,
            h: 80,
        },
    ]
}

impl crate::ai::DetectionProvider for KnownObjects {
    fn knows_class(&self, name: &str) -> bool {
        ["person", "car"]
            .iter()
            .any(|c| name.eq_ignore_ascii_case(c))
    }
    fn describe(&self) -> Vec<String> {
        vec!["person".to_string(), "car".to_string()]
    }
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
        Ok(known_detections())
    }
}

#[derive(Debug)]
struct FailingObjects;

impl crate::ai::DetectionProvider for FailingObjects {
    fn knows_class(&self, name: &str) -> bool {
        name.eq_ignore_ascii_case("person")
    }
    fn describe(&self) -> Vec<String> {
        vec!["person".to_string()]
    }
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
        Err(crate::ai::AiError::Inference(
            "detector exploded".to_string(),
        ))
    }
}

#[derive(Debug, Default)]
struct CountingObjects {
    calls: std::sync::atomic::AtomicUsize,
}

impl crate::ai::DetectionProvider for CountingObjects {
    fn knows_class(&self, name: &str) -> bool {
        ["person", "car"]
            .iter()
            .any(|c| name.eq_ignore_ascii_case(c))
    }
    fn describe(&self) -> Vec<String> {
        vec!["person".to_string(), "car".to_string()]
    }
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(known_detections())
    }
}

/// 200x100, a black/white checkerboard everywhere: any
/// redaction (blur or pixelate) visibly changes its box.
fn class_scene() -> Vec<u8> {
    let img = image::RgbImage::from_fn(200, 100, |x, y| {
        if (x + y) % 2 == 0 {
            image::Rgb([0, 0, 0])
        } else {
            image::Rgb([255, 255, 255])
        }
    });
    png_bytes(&DynamicImage::ImageRgb8(img))
}

fn detect_ctx(provider: &dyn crate::ai::DetectionProvider) -> ProcessContext<'_> {
    ProcessContext {
        detect_provider: Some(provider),
        ..Default::default()
    }
}

#[test]
fn pixelateclass_redacts_only_the_named_class() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::PixelateClass(
                "Person".to_string(),
            )])],
            detect_ctx(&KnownObjects),
        )
        .unwrap();
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    // Inside the person box the checkerboard is gone
    // (pixelate averages blocks: neighbors now agree).
    assert_eq!(result.get_pixel(11, 11), result.get_pixel(12, 11));
    assert_eq!(result.get_pixel(11, 11), result.get_pixel(11, 12));
    // Inside the car box the checkerboard survives.
    assert_ne!(result.get_pixel(121, 11), result.get_pixel(122, 11));
}

#[test]
fn chained_class_filters_share_one_detection() {
    let engine = Engine;
    let counter = CountingObjects::default();
    engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![
                Filter::BlurClass("person".to_string()),
                Filter::PixelateClass("car".to_string()),
            ])],
            detect_ctx(&counter),
        )
        .unwrap();
    assert_eq!(counter.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn a_flip_between_class_filters_redetects() {
    // Issue #373: a flip keeps the dimensions, so the
    // dimensions-keyed memo once served the pre-flip
    // boxes to the second filter and the redaction
    // landed on the wrong side. A geometric step
    // invalidates the memo: two detections.
    let engine = Engine;
    let counter = CountingObjects::default();
    engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![
                Filter::BlurClass("person".to_string()),
                Filter::Flip(crate::url::FlipDirection::Horizontal),
                Filter::PixelateClass("car".to_string()),
            ])],
            detect_ctx(&counter),
        )
        .unwrap();
    assert_eq!(counter.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

/// A detector that knows person and bicycle but only
/// ever finds the person: the no-match witness for the
/// objects crop fallback (issue #318).
#[derive(Debug)]
struct SparseObjects;

impl crate::ai::DetectionProvider for SparseObjects {
    fn knows_class(&self, name: &str) -> bool {
        ["person", "bicycle"]
            .iter()
            .any(|c| name.eq_ignore_ascii_case(c))
    }
    fn describe(&self) -> Vec<String> {
        vec!["person".to_string(), "bicycle".to_string()]
    }
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
        Ok(vec![crate::ai::Detection {
            label: "person".to_string(),
            score: 0.9,
            x: 10,
            y: 10,
            w: 60,
            h: 80,
        }])
    }
}

fn objects_crop_op(w: u32, h: u32, classes: &[&str], scale: f32) -> Operation {
    Operation::Crop {
        size: ImageSize {
            width: Dimension::Pixels(w),
            height: Dimension::Pixels(h),
            flip_horizontal: false,
            flip_vertical: false,
        },
        align: Align::Objects {
            classes: classes.iter().map(|c| c.to_string()).collect(),
            scale,
        },
    }
}

/// 200x100 where pixel (x, y) reads back as [x, y, 0],
/// so a crop's window is visible in its pixels (the
/// crop module's coords pattern).
fn coords_scene() -> Vec<u8> {
    let img = image::RgbImage::from_fn(200, 100, |x, y| image::Rgb([x as u8, y as u8, 0]));
    png_bytes(&DynamicImage::ImageRgb8(img))
}

#[test]
fn objects_crop_frames_the_named_class() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &coords_scene(),
            &[objects_crop_op(60, 80, &["person"], 1.0)],
            detect_ctx(&KnownObjects),
        )
        .unwrap();
    assert_eq!((out.width, out.height), (60, 80));
    // The record names the path, the detections used,
    // and the union box framed.
    let record = out.objects_crop.expect("an objects crop record");
    assert_eq!(record.classes, ["person"]);
    assert!(!record.fallback);
    assert_eq!(record.scale, 1.0);
    assert_eq!(record.union, Some((10, 10, 60, 80)));
    assert_eq!(record.objects.len(), 1);
    assert_eq!(record.objects[0].label, "person");
    // The window is the person box itself: the corner
    // pixels are the box's own corners (the resize to
    // the same size leaves them in place).
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    assert_eq!(result.get_pixel(0, 0), &image::Rgb([10, 10, 0]));
    assert_eq!(result.get_pixel(59, 79), &image::Rgb([69, 89, 0]));
}

#[test]
fn objects_crop_unions_several_classes_and_shares_one_detection() {
    let engine = Engine;
    let counter = CountingObjects::default();
    let out = engine
        .process_preferred(
            &coords_scene(),
            &[objects_crop_op(170, 80, &["person", "car"], 1.0)],
            detect_ctx(&counter),
        )
        .unwrap();
    assert_eq!((out.width, out.height), (170, 80));
    let record = out.objects_crop.expect("an objects crop record");
    assert!(!record.fallback);
    // The union spans both boxes: x 10..180, y 10..90.
    assert_eq!(record.union, Some((10, 10, 170, 80)));
    assert_eq!(record.objects.len(), 2);
    // Both classes resolve to the one provider, whose
    // memoized detection serves the second class too.
    assert_eq!(counter.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn objects_crop_with_no_match_falls_back_to_the_plain_crop() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &coords_scene(),
            &[objects_crop_op(100, 100, &["bicycle"], 1.5)],
            detect_ctx(&SparseObjects),
        )
        .unwrap();
    // The record states the fallback plainly.
    let record = out.objects_crop.expect("an objects crop record");
    assert!(record.fallback);
    assert_eq!(record.classes, ["bicycle"]);
    assert_eq!(record.scale, 1.5);
    assert_eq!(record.union, None);
    assert!(record.objects.is_empty());
    // And the render is the plain center crop of the
    // same size, byte for byte.
    let plain = engine
        .process_preferred(
            &coords_scene(),
            &[crop_op(100, 100, Align::Center)],
            detect_ctx(&SparseObjects),
        )
        .unwrap();
    assert_eq!(out.bytes, plain.bytes);
}

#[test]
fn objects_crop_with_a_failing_detector_falls_back_to_the_plain_crop() {
    // The face crop's downgrade rule: a detection
    // failure degrades the mode, never the request.
    let engine = Engine;
    let out = engine
        .process_preferred(
            &coords_scene(),
            &[objects_crop_op(100, 100, &["person"], 1.0)],
            detect_ctx(&FailingObjects),
        )
        .unwrap();
    let record = out.objects_crop.expect("an objects crop record");
    assert!(record.fallback);
    let plain = engine
        .process_preferred(
            &coords_scene(),
            &[crop_op(100, 100, Align::Center)],
            detect_ctx(&FailingObjects),
        )
        .unwrap();
    assert_eq!(out.bytes, plain.bytes);
}

#[test]
fn objects_crop_without_a_model_is_not_configured() {
    // Never a silent plain crop when detections were
    // requested and the feature is off (issue #318).
    let engine = Engine;
    let err = engine
        .process_preferred(
            &coords_scene(),
            &[objects_crop_op(100, 100, &["person"], 1.0)],
            ProcessContext::default(),
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
    assert!(
        err.to_string().contains("object crop is not configured"),
        "{err}"
    );
}

#[test]
fn objects_crop_with_an_unknown_class_is_refused() {
    let engine = Engine;
    let err = engine
        .process_preferred(
            &coords_scene(),
            &[objects_crop_op(100, 100, &["hovercraft"], 1.0)],
            detect_ctx(&KnownObjects),
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("unknown detection class"), "{err}");
}

#[test]
fn debug_focal_objects_crop_marks_the_objects_and_the_union() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &coords_scene(),
            &[
                Operation::DebugFocal,
                objects_crop_op(100, 100, &["person", "car"], 1.0),
            ],
            detect_ctx(&KnownObjects),
        )
        .unwrap();
    // The canvas is the 200x100 scene, not the crop.
    assert_eq!(dims_of(&out), (200, 100));
    // A used detection box edge carries the amber
    // detection color where nothing overdraws it: the
    // person box's right edge at (69, 50).
    assert_eq!(debug_px(&out, 69, 50), crate::debugview::FACE_COLOR);
    // The union rectangle is the focal rectangle: its
    // corner (10, 10) is magenta, over the person box's
    // own corner.
    assert_eq!(debug_px(&out, 10, 10), crate::debugview::FOCAL_COLOR);
    // The crosshair sits at the union's center (95, 50).
    assert_eq!(debug_px(&out, 95, 50), crate::debugview::FOCAL_COLOR);
    // The window the crop cut is (45, 0, 100, 100): its
    // corner carries the window color.
    assert_eq!(debug_px(&out, 45, 0), crate::debugview::WINDOW_COLOR);
}

#[test]
fn objects_crop_with_a_real_model_frames_the_dog() {
    // Env-gated like the ai.rs live checks: without the
    // ONNX runtime and a detection model the test skips
    // (the CI ai-models workflow provisions both).
    use crate::ai::DetectionProvider as _;
    let (Some(dylib), Some(model)) = (
        crate::ai::env_or_skip("CRABCROP_TEST_ORT_DYLIB"),
        crate::ai::env_or_skip("CRABCROP_TEST_DETECT_MODEL"),
    ) else {
        return;
    };
    let labels = std::env::var("CRABCROP_TEST_DETECT_LABELS").ok();
    let provider =
        crate::ai::OnnxDetectionProvider::load(Some(&dylib), &model, labels.as_deref(), 0.35)
            .unwrap();
    assert!(provider.knows_class("dog"));
    let bytes = include_bytes!("../../tests/functional/corpus/dog.jpg");
    let engine = Engine;
    let out = engine
        .process_preferred(
            bytes,
            &[objects_crop_op(200, 200, &["dog"], 1.5)],
            detect_ctx(&provider),
        )
        .unwrap();
    assert_eq!((out.width, out.height), (200, 200));
    let record = out.objects_crop.expect("an objects crop record");
    assert!(!record.fallback, "the dog must be found: {record:?}");
    assert!(!record.objects.is_empty());
    assert!(
        record
            .objects
            .iter()
            .all(|d| d.label.eq_ignore_ascii_case("dog"))
    );
    let (x, y, w, h) = record.union.expect("a union box");
    assert!(x + w <= out.source_width && y + h <= out.source_height);
}

/// A declared detector (issue #166) whose vocabulary is
/// its own: zebra, over the car box of the class scene.
#[derive(Debug)]
struct DeclaredObjects;

impl crate::ai::DetectionProvider for DeclaredObjects {
    fn knows_class(&self, name: &str) -> bool {
        name.eq_ignore_ascii_case("zebra")
    }
    fn describe(&self) -> Vec<String> {
        vec!["zebra".to_string()]
    }
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
        Ok(vec![crate::ai::Detection {
            label: "zebra".to_string(),
            score: 0.7,
            x: 120,
            y: 10,
            w: 60,
            h: 80,
        }])
    }
}

/// A declared detector that knows person (shared with
/// the built-in) and zebra (its own), counting its runs:
/// the resolution-order witness.
#[derive(Debug, Default)]
struct CountingDeclared {
    calls: std::sync::atomic::AtomicUsize,
}

impl crate::ai::DetectionProvider for CountingDeclared {
    fn knows_class(&self, name: &str) -> bool {
        ["person", "zebra"]
            .iter()
            .any(|c| name.eq_ignore_ascii_case(c))
    }
    fn describe(&self) -> Vec<String> {
        vec!["person".to_string(), "zebra".to_string()]
    }
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(vec![
            crate::ai::Detection {
                label: "person".to_string(),
                score: 0.9,
                x: 10,
                y: 10,
                w: 60,
                h: 80,
            },
            crate::ai::Detection {
                label: "zebra".to_string(),
                score: 0.7,
                x: 120,
                y: 10,
                w: 60,
                h: 80,
            },
        ])
    }
}

fn declared_ctx<'a>(
    builtin: Option<&'a dyn crate::ai::DetectionProvider>,
    declared: &'a [&'a dyn crate::ai::DetectionProvider],
) -> ProcessContext<'a> {
    ProcessContext {
        detect_provider: builtin,
        declared_detect_providers: declared,
        ..Default::default()
    }
}

#[test]
fn declared_detector_drives_redaction_for_its_own_class() {
    let engine = Engine;
    let declared: [&dyn crate::ai::DetectionProvider; 1] = [&DeclaredObjects];
    let out = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::PixelateClass(
                "zebra".to_string(),
            )])],
            declared_ctx(Some(&KnownObjects), &declared),
        )
        .unwrap();
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    // The zebra box (the declared provider's) is
    // pixelated: neighbors inside it agree.
    assert_eq!(result.get_pixel(121, 11), result.get_pixel(122, 11));
    // The person box is untouched: only the declared
    // provider ran, and it returns no person boxes.
    assert_ne!(result.get_pixel(11, 11), result.get_pixel(12, 11));
}

#[test]
fn class_resolution_prefers_the_builtin_detector() {
    let engine = Engine;
    let builtin = CountingObjects::default();
    let declared_counter = CountingDeclared::default();
    let declared: [&dyn crate::ai::DetectionProvider; 1] = [&declared_counter];
    engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::BlurClass(
                "person".to_string(),
            )])],
            declared_ctx(Some(&builtin), &declared),
        )
        .unwrap();
    // Both providers know "person": the built-in runs,
    // the declared one never does.
    assert_eq!(builtin.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        declared_counter
            .calls
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
}

#[test]
fn declared_detectors_resolve_in_order_and_share_one_run_each() {
    let engine = Engine;
    let counting = CountingDeclared::default();
    let declared: [&dyn crate::ai::DetectionProvider; 2] = [&DeclaredObjects, &counting];
    // Zebra resolves to the first declared provider
    // (DeclaredObjects); person only exists on the
    // second. Each runs at most once for the request.
    engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![
                Filter::PixelateClass("zebra".to_string()),
                Filter::BlurClass("person".to_string()),
                Filter::PixelateClass("person".to_string()),
            ])],
            declared_ctx(None, &declared),
        )
        .unwrap();
    assert_eq!(counting.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    // A class no provider knows is still a clean
    // refusal with declared providers configured.
    let err = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::BlurClass(
                "giraffe".to_string(),
            )])],
            declared_ctx(None, &declared),
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
}

/// A declared focal provider pinned to the right edge
/// (issue #166).
#[derive(Debug)]
struct RightFocal;

impl crate::ai::FocalProvider for RightFocal {
    fn focal_point(&self, _img: &DynamicImage) -> Result<(f32, f32), crate::ai::AiError> {
        Ok((0.95, 0.5))
    }
}

/// Left half red, right half blue: the scene the face
/// smart-crop test uses, reused for focal precedence.
fn halves_scene() -> Vec<u8> {
    let mut img = image::RgbImage::new(100, 100);
    for (x, _y, p) in img.enumerate_pixels_mut() {
        *p = if x < 50 {
            image::Rgb([255, 0, 0])
        } else {
            image::Rgb([0, 0, 255])
        };
    }
    png_bytes(&DynamicImage::ImageRgb8(img))
}

fn smart_half_crop() -> Operation {
    Operation::Crop {
        size: ImageSize {
            width: Dimension::Pixels(50),
            height: Dimension::Pixels(100),
            flip_horizontal: false,
            flip_vertical: false,
        },
        align: Align::Smart,
    }
}

#[test]
fn declared_focal_point_steers_a_smart_crop() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &halves_scene(),
            &[smart_half_crop()],
            ProcessContext {
                focal_provider: Some(&RightFocal),
                ..Default::default()
            },
        )
        .unwrap();
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    // The declared point at x 0.95 takes the right
    // (blue) half wholesale.
    assert_eq!(result.get_pixel(25, 50), &image::Rgb([0, 0, 255]));
}

/// A face provider reporting a face over the left half:
/// the mirror of RightFace, for precedence against the
/// declared focal point at the right.
#[derive(Debug)]
struct LeftFace;

impl crate::ai::FaceProvider for LeftFace {
    fn detect(&self, img: &DynamicImage) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
        Ok(vec![crate::ai::FaceBox {
            x: 0,
            y: 0,
            w: img.width() / 2,
            h: img.height(),
            score: 0.95,
        }])
    }
}

#[test]
fn face_provider_beats_the_declared_focal_point() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &halves_scene(),
            &[smart_half_crop()],
            ProcessContext {
                face_provider: Some(&LeftFace),
                focal_provider: Some(&RightFocal),
                ..Default::default()
            },
        )
        .unwrap();
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    assert_eq!(result.get_pixel(25, 50), &image::Rgb([255, 0, 0]));
}

#[test]
fn caller_focal_rect_beats_the_declared_focal_point() {
    let engine = Engine;
    let src = png_bytes(&quadrant_pattern());
    // The focal rectangle names the black bottom-right;
    // the declared provider would pull a smart crop to
    // the pattern's own saliency (which keeps some
    // white, as the control in
    // focal_smart_crop_keeps_the_named_region shows).
    let out = engine
        .process_preferred(
            &src,
            &[focal_op(4, 2, 8, 4), square_crop(Align::Smart)],
            ProcessContext {
                focal_provider: Some(&RightFocal),
                ..Default::default()
            },
        )
        .unwrap();
    let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(img.pixels().all(|p| p[0] == 0));
}

#[test]
fn blurclass_unknown_class_and_missing_model_fail_cleanly() {
    let engine = Engine;
    let err = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::BlurClass(
                "zebra".to_string(),
            )])],
            detect_ctx(&KnownObjects),
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    let err = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::BlurClass(
                "person".to_string(),
            )])],
            ProcessContext::default(),
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
}

#[test]
fn blurclass_detection_failure_is_an_inference_error() {
    // The #42 rule: never serve an unredacted image
    // because detection failed.
    let engine = Engine;
    let err = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::BlurClass(
                "person".to_string(),
            )])],
            detect_ctx(&FailingObjects),
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Inference(_)), "{err:?}");
}

/// A fake segment provider (issue #175): the
/// silhouette of any box is its left half, at full
/// strength; the right half is background. Counts
/// encodings and decodings so the memoization is
/// observable.
#[derive(Debug, Default)]
struct LeftHalfSegment {
    encodes: std::sync::atomic::AtomicUsize,
    decodes: std::sync::atomic::AtomicUsize,
}

impl crate::ai::SegmentProvider for LeftHalfSegment {
    fn encode(
        &self,
        _img: &DynamicImage,
    ) -> Result<crate::ai::SegmentEmbedding, crate::ai::AiError> {
        self.encodes
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(crate::ai::SegmentEmbedding { data: vec![0.0] })
    }

    fn decode(
        &self,
        _embedding: &crate::ai::SegmentEmbedding,
        _img_w: u32,
        _img_h: u32,
        bbox: (u32, u32, u32, u32),
    ) -> Result<image::GrayImage, crate::ai::AiError> {
        self.decodes
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let (_, _, w, h) = bbox;
        Ok(image::GrayImage::from_fn(w, h, |lx, _| {
            if lx < w / 2 {
                image::Luma([255])
            } else {
                image::Luma([0])
            }
        }))
    }
}

fn segment_ctx<'a>(
    detect: &'a dyn crate::ai::DetectionProvider,
    segment: &'a dyn crate::ai::SegmentProvider,
) -> ProcessContext<'a> {
    ProcessContext {
        detect_provider: Some(detect),
        segment_provider: Some(segment),
        ..Default::default()
    }
}

#[test]
fn blurclass_with_a_segment_provider_blurs_only_the_silhouette() {
    let engine = Engine;
    let segment = LeftHalfSegment::default();
    let out = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::BlurClass(
                "person".to_string(),
            )])],
            segment_ctx(&KnownObjects, &segment),
        )
        .unwrap();
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    // The person box is (10, 10, 60, 80); its left half
    // is the silhouette. (11, 50) is inside it: the
    // checkerboard there is blurred away from both
    // source extremes.
    let blurred = result.get_pixel(11, 50);
    assert!(blurred[0] > 40 && blurred[0] < 215, "{blurred:?}");
    // (55, 50) is inside the box but outside the
    // silhouette: (55 + 50) is odd, so the source is
    // white, and it must survive exactly — the
    // rectangle form would have blurred it.
    assert_eq!(result.get_pixel(55, 50), &image::Rgb([255, 255, 255]));
    // The car box is untouched by a person filter:
    // (121 + 11) is even, so the source pixel is black.
    assert_eq!(result.get_pixel(121, 11), &image::Rgb([0, 0, 0]));
    // The record carries the mask geometry: the tight
    // box is the box's left half, coverage a half.
    let record = out.segments.expect("a segments record");
    assert_eq!(record.objects.len(), 1);
    let object = &record.objects[0];
    assert_eq!(object.label, "person");
    assert_eq!(object.detection, (10, 10, 60, 80));
    assert_eq!(object.tight_box, (10, 10, 30, 80));
    assert!((object.coverage - 0.5).abs() < 0.01, "{}", object.coverage);
    assert_eq!(object.centroid, (25, 50));
}

#[test]
fn pixelateclass_with_a_segment_provider_pixelates_only_the_silhouette() {
    let engine = Engine;
    let segment = LeftHalfSegment::default();
    let out = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::PixelateClass(
                "person".to_string(),
            )])],
            segment_ctx(&KnownObjects, &segment),
        )
        .unwrap();
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    // Inside the silhouette the checkerboard is gone.
    assert_eq!(result.get_pixel(11, 11), result.get_pixel(12, 11));
    // Inside the box but outside the silhouette it
    // survives.
    assert_ne!(result.get_pixel(41, 11), result.get_pixel(42, 11));
}

#[test]
fn chained_class_operations_share_one_encoding() {
    let engine = Engine;
    let segment = LeftHalfSegment::default();
    engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![
                Filter::BlurClass("person".to_string()),
                Filter::PixelateClass("car".to_string()),
                Filter::RecolorClass("person".to_string(), [255, 0, 0]),
            ])],
            segment_ctx(&KnownObjects, &segment),
        )
        .unwrap();
    // Three class operations, three boxes decoded, but
    // the image was encoded once: the encoder is the
    // expensive half and the memo serves the chain.
    assert_eq!(segment.encodes.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(segment.decodes.load(std::sync::atomic::Ordering::SeqCst), 3);
}

#[test]
fn recolorclass_paints_the_box_without_a_segment_provider() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::RecolorClass(
                "person".to_string(),
                [255, 0, 0],
            )])],
            detect_ctx(&KnownObjects),
        )
        .unwrap();
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    // The whole person box is painted, corners
    // included: without a segment provider the box is
    // the shape.
    assert_eq!(result.get_pixel(20, 50), &image::Rgb([255, 0, 0]));
    assert_eq!(result.get_pixel(69, 89), &image::Rgb([255, 0, 0]));
    // The car box keeps the checkerboard, and no
    // segments record exists: nothing was segmented.
    assert_ne!(result.get_pixel(121, 11), result.get_pixel(122, 11));
    assert!(out.segments.is_none());
}

#[test]
fn recolorclass_with_a_segment_provider_paints_only_the_silhouette() {
    let engine = Engine;
    let segment = LeftHalfSegment::default();
    let out = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::RecolorClass(
                "person".to_string(),
                [255, 0, 0],
            )])],
            segment_ctx(&KnownObjects, &segment),
        )
        .unwrap();
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    assert_eq!(result.get_pixel(20, 50), &image::Rgb([255, 0, 0]));
    // In the box but outside the silhouette: (55 + 50)
    // is odd, so white survives, unpainted.
    assert_eq!(result.get_pixel(55, 50), &image::Rgb([255, 255, 255]));
    assert!(out.segments.is_some());
}

#[test]
fn recolorclass_unknown_class_and_missing_model_fail_cleanly() {
    let engine = Engine;
    let err = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::RecolorClass(
                "zebra".to_string(),
                [255, 0, 0],
            )])],
            detect_ctx(&KnownObjects),
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    let err = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::RecolorClass(
                "person".to_string(),
                [255, 0, 0],
            )])],
            ProcessContext::default(),
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
}

/// A segment provider whose encoding fails.
#[derive(Debug)]
struct FailingSegment;

impl crate::ai::SegmentProvider for FailingSegment {
    fn encode(
        &self,
        _img: &DynamicImage,
    ) -> Result<crate::ai::SegmentEmbedding, crate::ai::AiError> {
        Err(crate::ai::AiError::Inference(
            "encoder exploded".to_string(),
        ))
    }

    fn decode(
        &self,
        _embedding: &crate::ai::SegmentEmbedding,
        _img_w: u32,
        _img_h: u32,
        _bbox: (u32, u32, u32, u32),
    ) -> Result<image::GrayImage, crate::ai::AiError> {
        unreachable!("decode is not reached when encode fails")
    }
}

#[test]
fn segment_failure_fails_the_request() {
    // The class rule, extended: a segmentation failure
    // fails the request rather than serve a silently
    // rectangle-redacted image.
    let engine = Engine;
    let err = engine
        .process_preferred(
            &class_scene(),
            &[Operation::Filters(vec![Filter::BlurClass(
                "person".to_string(),
            )])],
            segment_ctx(&KnownObjects, &FailingSegment),
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Inference(_)), "{err:?}");
}

/// 200x160 flat gray, except the known face box, which is a
/// black/white checkerboard.
fn face_scene() -> Vec<u8> {
    let mut img = image::RgbImage::from_pixel(200, 160, image::Rgb([90, 90, 90]));
    for (x, y, p) in img.enumerate_pixels_mut() {
        if (140..180).contains(&x) && (30..70).contains(&y) {
            *p = if (x + y) % 2 == 0 {
                image::Rgb([0, 0, 0])
            } else {
                image::Rgb([255, 255, 255])
            };
        }
    }
    png_bytes(&DynamicImage::ImageRgb8(img))
}

fn face_crop_op() -> Operation {
    Operation::Crop {
        size: ImageSize {
            width: Dimension::Pixels(100),
            height: Dimension::Pixels(100),
            flip_horizontal: false,
            flip_vertical: false,
        },
        align: Align::Face,
    }
}

#[test]
fn face_crop_targets_the_detected_face() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &face_scene(),
            &[face_crop_op()],
            ProcessContext {
                face_provider: Some(&KnownFace),
                detect_provider: None,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!((out.width, out.height), (100, 100));
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    // Default padding 0.5: the crop is the 80x80 region at
    // (120, 10), so the output center is the face center:
    // checkerboard blended by the resize (dark or light,
    // never the flat background gray), corner is background.
    let center = result.get_pixel(50, 50);
    assert!(center[0] < 40 || center[0] > 200, "center {center:?}");
    assert_eq!(result.get_pixel(2, 2), &image::Rgb([90, 90, 90]));
}

#[test]
fn debug_focal_face_crop_marks_the_face_it_centered_on() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &face_scene(),
            &[Operation::DebugFocal, face_crop_op()],
            ProcessContext {
                face_provider: Some(&KnownFace),
                detect_provider: None,
                ..Default::default()
            },
        )
        .unwrap();
    // The canvas is the 200x160 scene, not the 100x100 crop.
    assert_eq!(dims_of(&out), (200, 160));
    // The chosen face is the focal rectangle: its corner
    // carries the focal color, and the crosshair sits at
    // its center (160, 50).
    assert_eq!(debug_px(&out, 140, 30), crate::debugview::FOCAL_COLOR);
    assert_eq!(debug_px(&out, 160, 50), crate::debugview::FOCAL_COLOR);
    // The window is the padded-face window the plain test
    // documents: 80x80 at (120, 10).
    assert_eq!(debug_px(&out, 120, 10), crate::debugview::WINDOW_COLOR);
    assert_eq!(debug_px(&out, 199, 89), crate::debugview::WINDOW_COLOR);
}

#[test]
fn debug_focal_smart_crop_draws_the_faces_that_fed_it() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &face_scene(),
            &[Operation::DebugFocal, crop_op(100, 100, Align::Smart)],
            ProcessContext {
                face_provider: Some(&KnownFace),
                detect_provider: None,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(dims_of(&out), (200, 160));
    // The detected face box is amber; the crosshair is on
    // its center, which is where the smart window went.
    assert_eq!(debug_px(&out, 140, 30), crate::debugview::FACE_COLOR);
    assert_eq!(debug_px(&out, 160, 50), crate::debugview::FOCAL_COLOR);
    // Window: 160 wide (the full height at 1:1), starting
    // at x 40 (centered on the face, clamped).
    assert_eq!(debug_px(&out, 40, 0), crate::debugview::WINDOW_COLOR);
}

#[test]
fn face_crop_without_a_model_is_a_clean_error() {
    let engine = Engine;
    let err = engine
        .process(&face_scene(), &[face_crop_op()])
        .unwrap_err();
    match &err {
        EngineError::NotConfigured(msg) => assert!(msg.contains("face"), "{msg}"),
        other => panic!("expected NotConfigured, got {other:?}"),
    }
}

#[test]
fn face_crop_falls_back_when_detection_fails() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &face_scene(),
            &[face_crop_op()],
            ProcessContext {
                face_provider: Some(&FailingFace),
                detect_provider: None,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!((out.width, out.height), (100, 100));
}

#[test]
fn blurfaces_redacts_only_the_face_region() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &face_scene(),
            &[Operation::Filters(vec![Filter::BlurFaces])],
            ProcessContext {
                face_provider: Some(&KnownFace),
                detect_provider: None,
                ..Default::default()
            },
        )
        .unwrap();
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    // Outside the box: untouched.
    assert_eq!(result.get_pixel(10, 10), &image::Rgb([90, 90, 90]));
    assert_eq!(result.get_pixel(139, 50), &image::Rgb([90, 90, 90]));
    // Inside: the checkerboard extremes are blurred away.
    let mut extremes = 0;
    for y in 30..70 {
        for x in 140..180 {
            let v = result.get_pixel(x, y)[0];
            if v == 0 || v == 255 {
                extremes += 1;
            }
        }
    }
    assert!(extremes < 80, "{extremes} of 1600 pixels still extreme");
}

#[test]
fn pixelatefaces_blocks_the_face_region() {
    let engine = Engine;
    let out = engine
        .process_preferred(
            &face_scene(),
            &[Operation::Filters(vec![Filter::PixelateFaces])],
            ProcessContext {
                face_provider: Some(&KnownFace),
                detect_provider: None,
                ..Default::default()
            },
        )
        .unwrap();
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    assert_eq!(result.get_pixel(10, 10), &image::Rgb([90, 90, 90]));
    // Neighboring pixels inside one 5 px cell (40 px / 8)
    // agree.
    assert_eq!(result.get_pixel(141, 31), result.get_pixel(142, 32));
    // The checkerboard is gone: each 5x5 cell averages its
    // black and white pixels into a gray (a sampled block
    // would keep one of them), and with 13 of one and 12 of
    // the other there are only two such grays.
    let mut distinct = std::collections::HashSet::new();
    for y in 30..70 {
        for x in 140..180 {
            let p = result.get_pixel(x, y).0;
            assert!(p[0] != 0 && p[0] != 255, "face pixel ({x}, {y}) survived");
            distinct.insert(p);
        }
    }
    assert!(distinct.len() <= 2, "{} distinct colors", distinct.len());
}

#[test]
fn pixelatefaces_redacts_faces_under_20_px() {
    // Issue #401: faces narrower than 20 px got a 1 px block
    // and came back bit-identical with a success.
    let engine = Engine;
    let out = engine
        .process_preferred(
            &face_scene(),
            &[Operation::Filters(vec![Filter::PixelateFaces])],
            ProcessContext {
                face_provider: Some(&SmallFaces),
                detect_provider: None,
                ..Default::default()
            },
        )
        .unwrap();
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
    let source = image::load_from_memory(&face_scene()).unwrap().to_rgb8();
    let in_a_face = |x: u32, y: u32| {
        SMALL_FACES
            .iter()
            .any(|&(fx, fy, fw, fh)| (fx..fx + fw).contains(&x) && (fy..fy + fh).contains(&y))
    };
    for (x, y, p) in result.enumerate_pixels() {
        if in_a_face(x, y) {
            // The checkerboard's black and white are averaged
            // into grays.
            assert!(p[0] != 0 && p[0] != 255, "face pixel ({x}, {y}) survived");
        } else {
            assert_eq!(p, source.get_pixel(x, y), "pixel ({x}, {y}) changed");
        }
    }
}

#[test]
fn blurfaces_fails_when_detection_fails() {
    let engine = Engine;
    let err = engine
        .process_preferred(
            &face_scene(),
            &[Operation::Filters(vec![Filter::BlurFaces])],
            ProcessContext {
                face_provider: Some(&FailingFace),
                detect_provider: None,
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Inference(_)), "{err:?}");
}

#[test]
fn blurfaces_without_a_model_is_a_clean_error() {
    let engine = Engine;
    let err = engine
        .process(
            &face_scene(),
            &[Operation::Filters(vec![Filter::BlurFaces])],
        )
        .unwrap_err();
    match &err {
        EngineError::NotConfigured(msg) => assert!(msg.contains("face"), "{msg}"),
        other => panic!("expected NotConfigured, got {other:?}"),
    }
}

/// The boxes `alpha_redaction_scene` hides: the known face,
/// the known person, and a free region.
const ALPHA_HIDDEN_BOXES: [(u32, u32, u32, u32); 3] =
    [(140, 30, 40, 40), (10, 10, 60, 80), (80, 100, 40, 40)];

/// Issue #444: a black and white checkerboard stored as RGBA,
/// opaque except for `ALPHA_HIDDEN_BOXES`, which are at
/// `alpha`. The fake detectors report the boxes whatever the
/// alpha, as YuNet does (it reads RGB only).
fn alpha_redaction_scene(alpha: u8) -> Vec<u8> {
    let img = image::RgbaImage::from_fn(200, 160, |x, y| {
        let v = if (x + y) % 2 == 0 { 0 } else { 255 };
        let hidden = ALPHA_HIDDEN_BOXES
            .iter()
            .any(|&(bx, by, bw, bh)| (bx..bx + bw).contains(&x) && (by..by + bh).contains(&y));
        image::Rgba([v, v, v, if hidden { alpha } else { 255 }])
    });
    png_bytes(&DynamicImage::ImageRgba8(img))
}

#[test]
fn redaction_leaves_no_original_color_under_alpha() {
    // Issue #444, end to end: a face, an object, or a region
    // stored at alpha 0 or 128 came back with all or a third
    // of its original color, and JPEG output (which drops
    // alpha) served it. The measure is the issue's: the slope
    // of the output's red against the source's inside the
    // redacted area, about 1 for a pass-through and 0 for
    // none.
    let engine = Engine;
    let segment = LeftHalfSegment::default();
    let ctx = |models: &str| match models {
        "face" => ProcessContext {
            face_provider: Some(&KnownFace),
            ..Default::default()
        },
        "objects" => detect_ctx(&KnownObjects),
        "masks" => segment_ctx(&KnownObjects, &segment),
        _ => ProcessContext::default(),
    };
    let person = || "person".to_string();
    let face = (140, 30, 40, 40);
    let (rx, ry, rw, rh) = (80, 100, 40, 40);
    let cases = [
        ("blurfaces", Filter::BlurFaces, "face", face),
        ("pixelatefaces", Filter::PixelateFaces, "face", face),
        (
            "blurclass",
            Filter::BlurClass(person()),
            "objects",
            (10, 10, 60, 80),
        ),
        (
            "pixelateclass",
            Filter::PixelateClass(person()),
            "objects",
            (10, 10, 60, 80),
        ),
        // The silhouette is the box's left half.
        (
            "blurclass masked",
            Filter::BlurClass(person()),
            "masks",
            (10, 10, 30, 80),
        ),
        (
            "pixelateclass masked",
            Filter::PixelateClass(person()),
            "masks",
            (10, 10, 30, 80),
        ),
        (
            "blur_region",
            Filter::BlurRegion {
                x: rx,
                y: ry,
                w: rw,
                h: rh,
            },
            "none",
            (rx, ry, rw, rh),
        ),
        (
            "pixelate_region",
            Filter::PixelateRegion {
                x: rx,
                y: ry,
                w: rw,
                h: rh,
            },
            "none",
            (rx, ry, rw, rh),
        ),
    ];
    for alpha in [0u8, 128] {
        let source = alpha_redaction_scene(alpha);
        let src = image::load_from_memory(&source).unwrap().to_rgba8();
        for (name, filter, models, (bx, by, bw, bh)) in &cases {
            for format in [ImageFormat::Jpeg, ImageFormat::Png] {
                let out = engine
                    .process_preferred(
                        &source,
                        &[Operation::Filters(vec![
                            filter.clone(),
                            Filter::Format(FormatSpec::plain(format)),
                            Filter::Quality(Quality::Value(100)),
                        ])],
                        ctx(models),
                    )
                    .unwrap();
                let result = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
                let pairs: Vec<(f64, f64)> = (*by..by + bh)
                    .flat_map(|y| (*bx..bx + bw).map(move |x| (x, y)))
                    .map(|(x, y)| {
                        (
                            src.get_pixel(x, y)[0] as f64,
                            result.get_pixel(x, y)[0] as f64,
                        )
                    })
                    .collect();
                let n = pairs.len() as f64;
                let ms = pairs.iter().map(|p| p.0).sum::<f64>() / n;
                let mo = pairs.iter().map(|p| p.1).sum::<f64>() / n;
                let cov: f64 = pairs.iter().map(|(s, o)| (s - ms) * (o - mo)).sum();
                let var: f64 = pairs.iter().map(|(s, _)| (s - ms).powi(2)).sum();
                let slope = cov / var;
                assert!(
                    slope.abs() < 0.1,
                    "{name} at alpha {alpha} to {format:?}: slope {slope:.3}"
                );
                if format == ImageFormat::Png {
                    // The redacted area keeps the source's alpha.
                    assert_eq!(result.get_pixel(*bx + 1, *by + 1)[3], alpha, "{name}");
                }
            }
        }
    }
}

#[test]
fn chained_face_ops_detect_once() {
    let engine = Engine;
    let provider = CountingFace::default();
    engine
        .process_preferred(
            &face_scene(),
            &[Operation::Filters(vec![
                Filter::BlurFaces,
                Filter::PixelateFaces,
            ])],
            ProcessContext {
                face_provider: Some(&provider),
                detect_provider: None,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn a_flip_between_face_ops_redetects() {
    // Issue #373, the face memo's half of the class
    // test above: same dimensions after the flip, so
    // only the invalidation keeps the second op from
    // reusing the pre-flip face box.
    let engine = Engine;
    let provider = CountingFace::default();
    engine
        .process_preferred(
            &face_scene(),
            &[Operation::Filters(vec![
                Filter::BlurFaces,
                Filter::Flip(crate::url::FlipDirection::Horizontal),
                Filter::PixelateFaces,
            ])],
            ProcessContext {
                face_provider: Some(&provider),
                detect_provider: None,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

fn pad_op(w: u32, h: u32, fill: PadFill) -> Operation {
    Operation::Pad {
        size: ImageSize {
            width: Dimension::Pixels(w),
            height: Dimension::Pixels(h),
            flip_horizontal: false,
            flip_vertical: false,
        },
        fill,
    }
}

fn decoded(out: &ProcessedImage) -> image::RgbImage {
    image::load_from_memory(&out.bytes).unwrap().to_rgb8()
}

#[test]
fn pad_portrait_into_landscape_defaults_to_black() {
    let engine = Engine;
    // 100x200 red into 200x100: fits at 50x100, centered,
    // so x 75..125 is red and the rest is black.
    let out = engine
        .process(
            &solid_png(100, 200, [255, 0, 0]),
            &[pad_op(200, 100, PadFill::Directive)],
        )
        .unwrap();
    assert_eq!((out.width, out.height), (200, 100));
    let img = decoded(&out);
    assert_eq!(img.get_pixel(100, 50), &image::Rgb([255, 0, 0]));
    assert_eq!(img.get_pixel(124, 50), &image::Rgb([255, 0, 0]));
    assert_eq!(img.get_pixel(74, 50), &image::Rgb([0, 0, 0]));
    assert_eq!(img.get_pixel(125, 50), &image::Rgb([0, 0, 0]));
    assert_eq!(img.get_pixel(0, 0), &image::Rgb([0, 0, 0]));
}

#[test]
fn pad_landscape_into_portrait_with_a_color() {
    let engine = Engine;
    // 200x100 blue into 100x200 with orange bars: fits at
    // 100x50, centered vertically at y 75..125.
    let out = engine
        .process(
            &solid_png(200, 100, [0, 0, 255]),
            &[pad_op(100, 200, PadFill::Color([255, 136, 0]))],
        )
        .unwrap();
    assert_eq!((out.width, out.height), (100, 200));
    let img = decoded(&out);
    assert_eq!(img.get_pixel(50, 100), &image::Rgb([0, 0, 255]));
    assert_eq!(img.get_pixel(50, 74), &image::Rgb([255, 136, 0]));
    assert_eq!(img.get_pixel(50, 125), &image::Rgb([255, 136, 0]));
}

#[test]
fn pad_never_adapts_its_canvas() {
    // `adaptive` is a fit-in modifier; pad's canvas is
    // explicit, so an orientation mismatch never swaps it:
    // the output is exactly the requested canvas in both
    // mismatch directions.
    let engine = Engine;
    let out = engine
        .process(
            &solid_png(100, 200, [255, 0, 0]),
            &[pad_op(200, 100, PadFill::Directive)],
        )
        .unwrap();
    assert_eq!((out.width, out.height), (200, 100));
    let out = engine
        .process(
            &solid_png(200, 100, [255, 0, 0]),
            &[pad_op(100, 200, PadFill::Directive)],
        )
        .unwrap();
    assert_eq!((out.width, out.height), (100, 200));
}

#[test]
fn pad_uses_the_fill_directive_in_either_order() {
    let engine = Engine;
    for ops in [
        vec![
            Operation::Filters(vec![Filter::Fill([0, 0, 255])]),
            pad_op(200, 100, PadFill::Directive),
        ],
        vec![
            pad_op(200, 100, PadFill::Directive),
            Operation::Filters(vec![Filter::Fill([0, 0, 255])]),
        ],
    ] {
        let out = engine
            .process(&solid_png(100, 200, [0, 200, 0]), &ops)
            .unwrap();
        let img = decoded(&out);
        assert_eq!(img.get_pixel(0, 0), &image::Rgb([0, 0, 255]));
        assert_eq!(img.get_pixel(100, 50), &image::Rgb([0, 200, 0]));
    }
}

#[test]
fn pad_blur_bars_are_ambient() {
    let engine = Engine;
    // Left half red, right half blue, 100x100 into 200x100:
    // the image sits at x 50..150 unscaled; the bars are the
    // blurred cover copy, red-dominant on the left.
    let mut src = image::RgbImage::new(100, 100);
    for (x, _y, p) in src.enumerate_pixels_mut() {
        *p = if x < 50 {
            image::Rgb([255, 0, 0])
        } else {
            image::Rgb([0, 0, 255])
        };
    }
    let out = engine
        .process(
            &png_bytes(&DynamicImage::ImageRgb8(src)),
            &[pad_op(200, 100, PadFill::Blur)],
        )
        .unwrap();
    assert_eq!((out.width, out.height), (200, 100));
    let img = decoded(&out);
    // The fitted image itself is exact in the middle.
    assert_eq!(img.get_pixel(60, 50), &image::Rgb([255, 0, 0]));
    assert_eq!(img.get_pixel(140, 50), &image::Rgb([0, 0, 255]));
    // The left bar is a blurred wash: red leads, but the
    // pure source red is softened by the blur.
    let bar = img.get_pixel(10, 50);
    assert!(bar[0] > 140 && bar[2] < 110, "bar {bar:?}");
    let right_bar = img.get_pixel(190, 50);
    assert!(
        right_bar[2] > 140 && right_bar[0] < 110,
        "bar {right_bar:?}"
    );
}

#[test]
fn pad_with_a_matching_aspect_changes_nothing() {
    let engine = Engine;
    let out = engine
        .process(
            &solid_png(200, 100, [10, 20, 30]),
            &[pad_op(200, 100, PadFill::Color([255, 255, 255]))],
        )
        .unwrap();
    assert_eq!((out.width, out.height), (200, 100));
    let img = decoded(&out);
    assert_eq!(img.get_pixel(0, 0), &image::Rgb([10, 20, 30]));
    assert_eq!(img.get_pixel(199, 99), &image::Rgb([10, 20, 30]));
}

/// A renderer that stamps one white pixel at the origin:
/// proof of routing and ordering, nothing more.
#[derive(Debug)]
struct MarkerRenderer;

impl crate::text::TextRenderer for MarkerRenderer {
    fn draw(
        &self,
        img: &DynamicImage,
        _text: &str,
        _size: u32,
        _color: [u8; 4],
        _anchor: crate::url::Anchor,
    ) -> Result<DynamicImage, String> {
        let mut out = img.to_rgba8();
        out.put_pixel(0, 0, image::Rgba([255, 255, 255, 255]));
        Ok(DynamicImage::ImageRgba8(out))
    }
}

fn text_op() -> Operation {
    Operation::Filters(vec![Filter::Text {
        text: "HI".to_string(),
        size: 20,
        color: [255, 255, 255, 255],
        anchor: crate::url::Anchor::BottomRight,
    }])
}

#[test]
fn svg_sources_rasterize_through_the_pipeline() {
    let engine = Engine;
    let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><rect width="100" height="100" fill="#00ff00"/><circle cx="50" cy="50" r="20" fill="#ff0000"/></svg>"##;
    // No size asked: intrinsic size, PNG output.
    let out = engine.process(svg, &[]).unwrap();
    assert_eq!((out.width, out.height), (100, 100));
    assert_eq!(out.format, ImageFormat::Png);
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
    assert_eq!(img.get_pixel(50, 50), &image::Rgba([255, 0, 0, 255]));
    // A size op renders the vector at that size, crisply:
    // the circle scales with it.
    let out = engine
        .process(
            svg,
            &[size_op(Dimension::Pixels(50), Dimension::Pixels(50))],
        )
        .unwrap();
    assert_eq!((out.width, out.height), (50, 50));
    let img = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
    assert_eq!(img.get_pixel(25, 25), &image::Rgba([255, 0, 0, 255]));
    assert_eq!(img.get_pixel(2, 2), &image::Rgba([0, 255, 0, 255]));
    // Non-SVG XML is a bad source, not a render.
    assert!(matches!(
        engine.process(b"<note><to>x</to></note>", &[]),
        Err(EngineError::Decode(_))
    ));
}

#[test]
fn dpr_scales_size_and_crop_targets() {
    let engine = Engine;
    let src = solid_png(100, 100, [9, 9, 9]);
    let dims = |ops: &[Operation]| {
        let out = engine.process(&src, ops).unwrap();
        (out.width, out.height)
    };
    // dpr(2) doubles the size target.
    assert_eq!(
        dims(&[
            size_op(Dimension::Pixels(25), Dimension::Pixels(25)),
            Operation::Filters(vec![Filter::Dpr(2.0)])
        ]),
        (50, 50)
    );
    // A fractional factor rounds to whole pixels: 33 * 1.5.
    assert_eq!(
        dims(&[
            size_op(Dimension::Pixels(33), Dimension::Pixels(33)),
            Operation::Filters(vec![Filter::Dpr(1.5)])
        ]),
        (50, 50)
    );
    // dpr(1) is a no-op.
    assert_eq!(
        dims(&[
            size_op(Dimension::Pixels(25), Dimension::Pixels(25)),
            Operation::Filters(vec![Filter::Dpr(1.0)])
        ]),
        (25, 25)
    );
    // Crop targets scale too.
    assert_eq!(
        dims(&[
            Operation::Crop {
                size: ImageSize {
                    width: Dimension::Pixels(25),
                    height: Dimension::Pixels(30),
                    flip_horizontal: false,
                    flip_vertical: false,
                },
                align: Align::Center,
            },
            Operation::Filters(vec![Filter::Dpr(2.0)]),
        ]),
        (50, 60)
    );
}

#[test]
fn dpr_runs_into_the_no_upscale_cap_not_past_it() {
    let engine = Engine;
    let out = engine
        .process(
            &solid_png(60, 60, [9, 9, 9]),
            &[
                size_op(Dimension::Pixels(50), Dimension::Pixels(50)),
                Operation::Filters(vec![Filter::Dpr(2.0), Filter::NoUpscale]),
            ],
        )
        .unwrap();
    assert_eq!((out.width, out.height), (60, 60));
}

#[test]
fn dpr_hint_applies_without_a_directive_and_loses_to_one() {
    let engine = Engine;
    let src = solid_png(100, 100, [9, 9, 9]);
    // The hint alone doubles.
    let out = engine
        .process_preferred(
            &src,
            &[size_op(Dimension::Pixels(25), Dimension::Pixels(25))],
            ProcessContext {
                dpr: Some(2.0),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!((out.width, out.height), (50, 50));
    // An explicit directive beats the hint: 40 * 1.5, not * 2.
    let out = engine
        .process_preferred(
            &src,
            &[
                size_op(Dimension::Pixels(40), Dimension::Pixels(40)),
                Operation::Filters(vec![Filter::Dpr(1.5)]),
            ],
            ProcessContext {
                dpr: Some(2.0),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!((out.width, out.height), (60, 60));
}

#[test]
fn text_routes_to_the_renderer_after_the_resize() {
    let engine = Engine;
    let ops = vec![
        Operation::Size(ImageSize {
            width: Dimension::Pixels(50),
            height: Dimension::Pixels(50),
            flip_horizontal: false,
            flip_vertical: false,
        }),
        text_op(),
    ];
    let out = engine
        .process_preferred(
            &solid_png(100, 100, [0, 0, 0]),
            &ops,
            ProcessContext {
                text_renderer: Some(&MarkerRenderer),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!((out.width, out.height), (50, 50));
    let result = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
    assert_eq!(result.get_pixel(0, 0), &image::Rgba([255, 255, 255, 255]));
}

#[test]
fn text_without_a_renderer_is_a_clean_error() {
    let engine = Engine;
    assert!(matches!(
        engine.process(&solid_png(10, 10, [0, 0, 0]), &[text_op()]),
        Err(EngineError::NotConfigured(msg)) if msg.contains("text is not configured")
    ));
}

#[test]
fn trim_composes_with_a_following_resize() {
    let engine = Engine;
    // 60x40 white with a centered black box, 20x20 at (20, 10).
    let mut img = DynamicImage::new_rgb8(60, 40);
    for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        *p = if (20..40).contains(&x) && (10..30).contains(&y) {
            image::Rgb([0, 0, 0])
        } else {
            image::Rgb([255, 255, 255])
        };
    }
    let src = png_bytes(&img);
    let out = engine
        .process(
            &src,
            &[
                Operation::Filters(vec![Filter::Trim(10)]),
                size_op(Dimension::Pixels(10), Dimension::Pixels(10)),
            ],
        )
        .unwrap();
    assert_eq!(dims_of(&out), (10, 10));
    // Every output pixel is black: the resize saw only the
    // trimmed box, not the white frame.
    let decoded = image::load_from_memory(&out.bytes).unwrap().to_luma8();
    assert!(decoded.pixels().all(|p| p[0] == 0));
}

#[test]
fn processed_image_reports_dimensions() {
    let engine = Engine;
    let src = png_bytes(&DynamicImage::new_rgb8(400, 200));
    let out = engine
        .process(
            &src,
            &[size_op(Dimension::Pixels(100), Dimension::Pixels(50))],
        )
        .unwrap();
    assert_eq!((out.width, out.height), (100, 50));
    assert_eq!((out.source_width, out.source_height), (400, 200));
}

#[test]
fn quality_ladder_steps_by_format() {
    // The descent is planned against the steps that
    // remain: geometric from the current quality, landing
    // on the floor of 1 at the final step (issue #229).
    assert_eq!(next_quality(80, 5), 33);
    assert_eq!(next_quality(80, 1), 1);
    assert_eq!(next_quality(2, 5), 1);
    let mut q = 95;
    for steps_left in (1..=5).rev() {
        q = next_quality(q, steps_left);
    }
    assert_eq!(q, 1, "five planned steps from 95 reach the floor");
    let mut q = 95;
    for steps_left in (1..=4).rev() {
        q = next_quality(q, steps_left);
    }
    assert_eq!(q, 1, "four planned steps from 95 reach the floor");
    assert_eq!(max_quality_attempts(ImageFormat::Avif), 4);
    assert!(max_quality_attempts(ImageFormat::Jpeg) > 4);
}

#[test]
fn max_bytes_unreachable_budget_returns_the_floor_encode() {
    // Issue #229: with a budget nothing can meet, the
    // ladder must still walk to quality 1 and return that
    // smallest encode, not stop wherever the encode pool
    // used to run out.
    let engine = Engine;
    let src = noise_png();
    let floor = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
                Filter::Quality(Quality::Value(1)),
            ])],
        )
        .unwrap();
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
                Filter::Quality(Quality::Value(95)),
                Filter::MaxBytes(1),
            ])],
        )
        .unwrap();
    assert_eq!(out.quality, Some(1));
    assert_eq!(out.bytes, floor.bytes);
}

#[test]
fn max_bytes_still_stops_at_the_first_fitting_quality() {
    // A budget the ladder reaches midway behaves exactly
    // as before: stop at the first quality that fits.
    let engine = Engine;
    let src = noise_png();
    let at = |q: u8| {
        engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
                    Filter::Quality(Quality::Value(q)),
                ])],
            )
            .unwrap()
            .bytes
            .len() as u64
    };
    let budget = at(15);
    assert!(at(38) > budget, "the step above 15 must not fit");
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![
                Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
                Filter::Quality(Quality::Value(95)),
                Filter::MaxBytes(budget),
            ])],
        )
        .unwrap();
    assert!(out.bytes.len() as u64 <= budget);
    assert_eq!(out.quality, Some(15));
}

#[test]
fn errors_render() {
    assert_eq!(
        EngineError::Unsupported("crop".to_string()).to_string(),
        "unsupported operation: crop"
    );
}

/// Timing reference for the issue thread; run explicitly with
/// `cargo test avif_encode_bench -- --ignored --nocapture`.
#[test]
#[ignore = "timing benchmark"]
fn avif_encode_bench() {
    let mut img = DynamicImage::new_rgb8(512, 384);
    for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        *p = image::Rgb([(x / 2) as u8, (y / 2) as u8, ((x + y) / 4) as u8]);
    }
    let src = png_bytes(&img);
    let engine = Engine;
    let start = std::time::Instant::now();
    let out = engine
        .process(
            &src,
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Avif,
            ))])],
        )
        .unwrap();
    eprintln!(
        "avif 512x384 q80 speed8: {:?}, {} bytes",
        start.elapsed(),
        out.bytes.len()
    );
}
