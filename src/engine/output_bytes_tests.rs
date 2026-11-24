//! The byte budget an output step is held to (issue #784). A
//! step allocates its output in the layout of the frame in
//! hand, so a 16-bit or float frame takes two to four times
//! the bytes of an 8-bit one at the same size. A step whose
//! frame would pass the bytes an RGBA8 frame at the pixel
//! limit holds is refused before it allocates; one at the
//! budget renders as it did.

use super::*;

/// A 2x2 frame in `layout`, one color.
fn frame(layout: image::ColorType, width: u32, height: u32) -> DynamicImage {
    let rgba = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
        width,
        height,
        image::Rgba([40, 120, 200, 255]),
    ));
    match layout {
        image::ColorType::Rgba16 => DynamicImage::ImageRgba16(rgba.to_rgba16()),
        image::ColorType::Rgb16 => DynamicImage::ImageRgb16(rgba.to_rgb16()),
        image::ColorType::Rgba32F => DynamicImage::ImageRgba32F(rgba.to_rgba32f()),
        image::ColorType::Rgb32F => DynamicImage::ImageRgb32F(rgba.to_rgb32f()),
        other => panic!("no frame for {other:?}"),
    }
}

/// The chain's image for `remainder` (operations only) over
/// `img`, or its refusal.
fn chain(img: DynamicImage, remainder: &str) -> Result<DynamicImage, EngineError> {
    let (ops, _) = crate::url::parse_operations(&format!("{remainder}/example.com/a.png")).unwrap();
    let directives = crate::url::RequestDirectives::analyze(&ops);
    let ctx = ProcessContext::default();
    apply_operations(img, &ops, None, 1.0, &ctx, &directives, None).map(|outcome| outcome.img)
}

/// The refusal `remainder` gets over `img`. A match, not
/// `unwrap_err`: on a regression the image is hundreds of
/// megabytes, and its `Debug` form would flood the log.
fn refusal(img: DynamicImage, remainder: &str) -> String {
    let layout = img.color();
    match chain(img, remainder) {
        Err(err) => {
            assert!(
                matches!(err, EngineError::Unsupported(_)),
                "{layout:?} {remainder}: {err:?}"
            );
            err.to_string()
        }
        Ok(out) => panic!(
            "{layout:?} {remainder} rendered {}x{} as {:?}",
            out.width(),
            out.height(),
            out.color()
        ),
    }
}

/// The refusal of a `width` x `height` frame at `bytes` a
/// pixel.
fn past_budget(width: u32, height: u32, bytes: u32) -> String {
    format!(
        "unsupported operation: requested output {width}x{height} at {bytes} bytes a pixel \
         exceeds the 400000000 byte limit"
    )
}

#[test]
fn a_float_and_a_sixteen_bit_resize_at_the_byte_limit_render_and_past_it_are_refused() {
    // 5000x5000 at 16 bytes a pixel and 10000x5000 at 8 hold
    // exactly the bytes of an RGBA8 frame at the 100 million
    // pixel limit: both render, in their own layout. One more
    // row passes the budget, inside the pixel limit.
    for (layout, at, bytes) in [
        (image::ColorType::Rgba32F, (5000, 5000), 16),
        (image::ColorType::Rgba16, (10000, 5000), 8),
    ] {
        let (w, h) = at;
        assert_eq!(
            refusal(frame(layout, 2, 2), &format!("{w}x{}", h + 1)),
            past_budget(w, h + 1, bytes),
            "{layout:?}"
        );
        let out = chain(frame(layout, 2, 2), &format!("{w}x{h}"))
            .unwrap_or_else(|err| panic!("{layout:?} {w}x{h}: {err}"));
        assert_eq!(
            (out.width(), out.height(), out.color()),
            (w, h, layout),
            "{layout:?}"
        );
    }
}

#[test]
fn every_step_that_grows_a_deep_frame_is_held_to_the_byte_limit() {
    // Each refusal names the frame the step would allocate
    // in the frame's layout, before it allocates it: the
    // resize closing a crop, a fit-in, the image a pad fits
    // into its canvas, and the cover a blurred pad scales the
    // image to before it crops and blurs it.
    let float = image::ColorType::Rgba32F;
    for (remainder, refused) in [
        ("5000x5001", past_budget(5000, 5001, 16)),
        ("fit-in:5001x5001", past_budget(5001, 5001, 16)),
        ("crop:5000x5001", past_budget(5000, 5001, 16)),
        ("pad:5001x5001", past_budget(5001, 5001, 16)),
    ] {
        assert_eq!(
            refusal(frame(float, 2, 2), remainder),
            refused,
            "{remainder}"
        );
    }
    // A 2:1 frame padded into a square: the image fits at
    // 4000x2000 and the canvas is 4000x4000, both inside the
    // budget, but the blurred backdrop first scales the frame
    // to cover the canvas, 8000x4000 in its own layout.
    assert_eq!(
        refusal(frame(float, 2, 1), "pad:4000x4000:blur"),
        past_budget(8000, 4000, 16)
    );
    // The same pad with a solid fill renders: its canvas is
    // RGBA8.
    let out = chain(frame(float, 2, 1), "pad:4000x4000").unwrap();
    assert_eq!(
        (out.width(), out.height(), out.color()),
        (4000, 4000, image::ColorType::Rgba8)
    );
}

#[test]
fn a_hinted_factor_steps_down_from_a_frame_past_the_byte_limit() {
    // A DPR client hint is advisory (issue #529): when it
    // alone takes a step past the byte budget, the render
    // steps the factor down as it does at the pixel limit.
    // 2000x2000 at factor 3 is 6000x6000, inside the pixel
    // limit but 576 MB as float; at factor 2 it is 256 MB.
    let mut tiff = Vec::new();
    frame(image::ColorType::Rgba32F, 2, 2)
        .write_to(&mut Cursor::new(&mut tiff), image::ImageFormat::Tiff)
        .unwrap();
    let (ops, _) =
        crate::url::parse_operations("2000x2000/filters:format(jpeg)/example.com/a.tif").unwrap();
    let out = Engine
        .process_preferred(
            &tiff,
            &ops,
            ProcessContext {
                dpr: Some(3.0),
                ..Default::default()
            },
        )
        .unwrap_or_else(|err| panic!("{err}"));
    assert_eq!((out.width, out.height), (4000, 4000));
}

#[test]
fn the_byte_limit_binds_only_past_four_bytes_a_pixel() {
    use super::limits::{MAX_OUTPUT_BYTES, check_output_bytes};
    use image::ColorType;
    assert_eq!(MAX_OUTPUT_BYTES, 4 * MAX_OUTPUT_PIXELS);
    // Every 8-bit layout, and 16-bit gray with or without
    // alpha, fits at the pixel limit: the pixel limit is what
    // holds them.
    for layout in [
        ColorType::L8,
        ColorType::La8,
        ColorType::Rgb8,
        ColorType::Rgba8,
        ColorType::L16,
        ColorType::La16,
    ] {
        assert!(
            check_output_bytes(10_000, 10_000, layout).is_ok(),
            "{layout:?}"
        );
    }
    // The wider layouts reach the byte limit first: they fit
    // at it, and one more row passes it.
    for (layout, (w, h)) in [
        (ColorType::Rgb16, (10_000, 6_666)),
        (ColorType::Rgba16, (10_000, 5_000)),
        (ColorType::Rgb32F, (10_000, 3_333)),
        (ColorType::Rgba32F, (5_000, 5_000)),
    ] {
        assert!(check_output_bytes(w, h, layout).is_ok(), "{layout:?}");
        let err = check_output_bytes(w, h + 1, layout).unwrap_err();
        assert_eq!(
            err.to_string(),
            past_budget(w, h + 1, u32::from(layout.bytes_per_pixel())),
            "{layout:?}"
        );
    }
    let err = check_output_bytes(u32::MAX, u32::MAX, ColorType::Rgba32F).unwrap_err();
    assert!(err.to_string().ends_with("byte limit"), "{err}");
    // It reads as a refusal of an output size, so a hinted
    // DPR factor steps down from it (issue #529).
    let (ops, _) = crate::url::parse_operations("2000x2000/example.com/a.tif").unwrap();
    let directives = crate::url::RequestDirectives::analyze(&ops);
    let err = check_output_bytes(6_000, 6_000, ColorType::Rgba32F).unwrap_err();
    assert_eq!(
        hint_step_down(&err, &ops, &directives, Some(3.0)),
        Some(2.0)
    );
}

#[test]
fn a_scaled_overlay_part_past_the_byte_limit_is_refused() {
    // `overlay(...,width)` scales the part that lands in the
    // overlay's own layout before it becomes RGBA8: a 1x1 float
    // overlay at the full width of a 5001x5001 base would scale
    // a 5001x5001 part at 16 bytes a pixel. It is refused before
    // the part is scaled, as a step's image is.
    let base = DynamicImage::ImageLuma8(image::GrayImage::new(5001, 5001));
    let overlay = crate::filters::WatermarkImage::new(DynamicImage::ImageRgba32F(
        image::Rgba32FImage::from_pixel(1, 1, image::Rgba([0.8, 0.2, 0.2, 1.0])),
    ));
    let ctx = ProcessContext::default();
    let err = match blend_overlay(
        &ctx,
        &base,
        &overlay,
        0,
        0,
        100,
        crate::url::BlendMode::Normal,
        Some(100),
    ) {
        Err(err) => err,
        Ok(out) => panic!("composited {}x{}", out.width(), out.height()),
    };
    assert_eq!(err.to_string(), past_budget(5001, 5001, 16));
}

/// A 5x1000 RGBA8 strip, as wide as it is tall times 1/200.
fn strip() -> DynamicImage {
    DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
        5,
        1000,
        image::Rgba([40, 120, 200, 255]),
    ))
}

#[test]
fn a_blurred_pad_whose_cover_passes_the_byte_limit_is_refused_at_any_depth() {
    // The image a blurred pad scales to cover its canvas follows
    // the ratio of the canvas's aspect to the image's, not the
    // size of the output: a 1000x1000 canvas over a 1:200 strip
    // needs a 1000x200000 cover, 800 MB as RGBA8, for a
    // one-megapixel output. It is refused at 8 bits as at 16.
    assert_eq!(
        refusal(strip(), "pad:1000x1000:blur"),
        past_budget(1000, 200_000, 4)
    );
}

#[test]
fn an_eight_bit_blurred_pad_whose_cover_fits_renders() {
    // The same strip on a 100x100 canvas: a 100x20000 cover, 8
    // MB, within the budget, renders as it always did.
    let out = chain(strip(), "pad:100x100:blur").unwrap();
    assert_eq!(
        (out.width(), out.height(), out.color()),
        (100, 100, image::ColorType::Rgba8)
    );
}
