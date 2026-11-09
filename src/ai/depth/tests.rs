use super::*;

#[test]
fn sample_depth_reads_the_color_type() {
    let (w, h) = (2, 2);
    for (img, want) in [
        (DynamicImage::new_luma8(w, h), SampleDepth::Eight),
        (DynamicImage::new_luma_a8(w, h), SampleDepth::Eight),
        (DynamicImage::new_rgb8(w, h), SampleDepth::Eight),
        (DynamicImage::new_rgba8(w, h), SampleDepth::Eight),
        (DynamicImage::new_luma16(w, h), SampleDepth::Sixteen),
        (DynamicImage::new_luma_a16(w, h), SampleDepth::Sixteen),
        (DynamicImage::new_rgb16(w, h), SampleDepth::Sixteen),
        (DynamicImage::new_rgba16(w, h), SampleDepth::Sixteen),
        (DynamicImage::new_rgb32f(w, h), SampleDepth::Float),
        (DynamicImage::new_rgba32f(w, h), SampleDepth::Float),
    ] {
        assert_eq!(SampleDepth::of(&img), want, "{:?}", img.color());
    }
}

#[test]
fn model_values_clamp_into_each_depth() {
    assert_eq!(u8::from_unit(1.5), 255);
    assert_eq!(u8::from_unit(-1.0), 0);
    assert_eq!(u8::from_unit(f32::NAN), 0);
    assert_eq!(u8::from_255(254.6), 255);
    assert_eq!(u16::from_unit(0x1234 as f32 / 65535.0), 0x1234);
    assert_eq!(u16::from_unit(f32::NAN), 0);
    assert_eq!(u16::from_255(255.0), u16::MAX);
    assert_eq!(f32::from_unit(2.0), 1.0);
    assert_eq!(f32::from_unit(f32::NAN), 0.0);
    assert_eq!(f32::from_255(51.0), 0.2);
    // A high-dynamic-range float sample is clamped on its
    // way into the model, as the 8-bit conversion clamps it.
    assert_eq!(3.0f32.to_unit(), 1.0);
    assert_eq!(0x8000u16.to_unit(), 0x8000 as f32 / 65535.0);
}

/// Alpha is carried for every layout with an alpha channel,
/// at that layout's depth, and resized with the result; a
/// gray-alpha source gets an RGBA result.
#[test]
fn with_source_alpha_carries_alpha_at_the_source_depth() {
    let src = DynamicImage::ImageLumaA16(image::ImageBuffer::from_fn(4, 2, |x, _| {
        image::LumaA([0x4321, if x < 2 { 0 } else { u16::MAX }])
    }));
    let rgb = ImageBuffer::from_pixel(8, 4, Rgb([1u16, 2, 3]));
    let out = with_source_alpha(&src, rgb);
    assert_eq!(out.color(), image::ColorType::Rgba16);
    let out = out.as_rgba16().unwrap();
    assert_eq!(out.get_pixel(0, 0).0, [1, 2, 3, 0]);
    assert_eq!(out.get_pixel(7, 3).0, [1, 2, 3, u16::MAX]);
    // At the same size the alpha is the source's, sample
    // for sample.
    let src = DynamicImage::ImageRgba32F(ImageBuffer::from_fn(3, 1, |x, _| {
        Rgba([0.0, 0.0, 0.0, x as f32 * 0.25])
    }));
    let out = with_source_alpha(&src, ImageBuffer::from_pixel(3, 1, Rgb([0.5f32, 0.5, 0.5])));
    let alpha: Vec<f32> = out.as_rgba32f().unwrap().pixels().map(|p| p[3]).collect();
    assert_eq!(alpha, [0.0, 0.25, 0.5]);
    // A source without alpha gets an RGB result.
    let src = DynamicImage::new_luma16(3, 1);
    let out = with_source_alpha(&src, ImageBuffer::from_pixel(3, 1, Rgb([7u16, 8, 9])));
    assert_eq!(out.color(), image::ColorType::Rgb16);
}
