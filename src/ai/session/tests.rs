use super::*;
use crate::ai::test_support::{layouts, noise_rgb};
use image::GrayImage;

#[test]
fn ort_search_path_prefers_the_ort_dylib_variable() {
    // #180: the fallback resolution mirrors ort's own, so
    // the preflight loads exactly what ort would have
    // loaded lazily (and panicked on).
    assert_eq!(
        ort_search_path(Some("/opt/ort/libonnxruntime.so")),
        std::path::PathBuf::from("/opt/ort/libonnxruntime.so")
    );
    let default = std::path::PathBuf::from(format!(
        "{}onnxruntime{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ));
    assert_eq!(ort_search_path(None), default);
    assert_eq!(ort_search_path(Some("")), default);
}

#[test]
fn fir_resize_runs_rgb_and_gray_through_one_kernel() {
    // One body serves both layouts (issue #484): a gray
    // image resampled as RGB and as Luma lands on the same
    // value in every channel, at the requested size, up,
    // down, and at its own size.
    let gray = GrayImage::from_fn(37, 23, |x, y| image::Luma([((x * 7 + y * 11) % 256) as u8]));
    let rgb = image::DynamicImage::ImageLuma8(gray.clone()).to_rgb8();
    for (w, h) in [(64, 40), (13, 7), (37, 23)] {
        let resized_gray = fir_resize(&gray, w, h);
        let resized_rgb = fir_resize(&rgb, w, h);
        assert_eq!(resized_gray.dimensions(), (w, h));
        assert_eq!(resized_rgb.dimensions(), (w, h));
        for (g, c) in resized_gray.pixels().zip(resized_rgb.pixels()) {
            assert_eq!(c.0, [g.0[0]; 3], "{w}x{h}");
        }
    }
}

/// `fir_resize` as it was before issue #672: the resizer is
/// handed a copy of the frame it owns.
fn fir_resize_from_a_copy<P: FirPixel>(
    img: &ImageBuffer<P, Vec<u8>>,
    w: u32,
    h: u32,
) -> ImageBuffer<P, Vec<u8>> {
    use fast_image_resize::images::Image as FirImage;
    use fast_image_resize::{ResizeAlg, ResizeOptions, Resizer};
    let src =
        FirImage::from_vec_u8(img.width(), img.height(), img.as_raw().clone(), P::LAYOUT).unwrap();
    let mut dst = FirImage::new(w, h, P::LAYOUT);
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(
        fast_image_resize::FilterType::Bilinear,
    ));
    Resizer::new().resize(&src, &mut dst, &opts).unwrap();
    ImageBuffer::from_raw(w, h, dst.into_vec()).unwrap()
}

/// Source and target sizes the providers reach: one pixel
/// either way, upscales, downscales, an unchanged size, and
/// the models' own edges.
const RESIZES: [((u32, u32), (u32, u32)); 9] = [
    ((1, 1), (1, 1)),
    ((1, 1), (640, 640)),
    ((5, 3), (320, 320)),
    ((37, 23), (64, 40)),
    ((37, 23), (13, 7)),
    ((37, 23), (37, 23)),
    ((700, 500), (640, 640)),
    ((301, 517), (373, 640)),
    ((640, 480), (1, 1)),
];

#[test]
fn fir_resize_on_a_borrowed_frame_matches_the_copying_form() {
    // Issue #672: the resizer reads the frame in place.
    // The bytes it returns are the ones it returned when it
    // was handed a copy, for RGB and for gray buffers.
    for ((sw, sh), (dw, dh)) in RESIZES {
        let rgb = noise_rgb(sw, sh, sw * 31 + sh);
        assert_eq!(
            fir_resize(&rgb, dw, dh).into_raw(),
            fir_resize_from_a_copy(&rgb, dw, dh).into_raw(),
            "RGB {sw}x{sh} to {dw}x{dh}"
        );
        let gray = image::DynamicImage::ImageRgb8(rgb).to_luma8();
        assert_eq!(
            fir_resize(&gray, dw, dh).into_raw(),
            fir_resize_from_a_copy(&gray, dw, dh).into_raw(),
            "gray {sw}x{sh} to {dw}x{dh}"
        );
    }
}

#[test]
fn rgb8_view_borrows_an_rgb8_frame_and_converts_every_other_layout() {
    // Issue #672: an RGB8 frame is read where it is, never
    // copied; any other layout is converted exactly as
    // `to_rgb8` converts it.
    let rgb = noise_rgb(19, 11, 7);
    for (layout, img) in layouts(&rgb) {
        let view = rgb8_view(&img);
        assert_eq!(view.as_raw(), img.to_rgb8().as_raw(), "{layout}");
        match img.as_rgb8() {
            Some(own) => {
                assert!(matches!(view, std::borrow::Cow::Borrowed(_)), "{layout}");
                assert!(std::ptr::eq(view.as_raw().as_ptr(), own.as_raw().as_ptr()));
            }
            None => assert!(matches!(view, std::borrow::Cow::Owned(_)), "{layout}"),
        }
    }
}

#[test]
fn the_view_feeds_the_resizer_the_bytes_the_copies_did() {
    // The two steps together, as every preprocessor runs
    // them, against the pre-#672 form (a `to_rgb8` copy,
    // then a copy for the resizer), at one pixel, upscaled
    // and downscaled: every layout where the target is
    // small, and where it is a model's whole input, which
    // is most of a tenth of a second in a debug build, a
    // frame that is borrowed and one that is converted.
    for ((sw, sh), (dw, dh)) in RESIZES {
        let rgb = noise_rgb(sw, sh, sw * 17 + sh);
        let layouts = layouts(&rgb);
        let count = if dw * dh > 100_000 { 2 } else { layouts.len() };
        for (layout, img) in layouts.into_iter().take(count) {
            assert_eq!(
                fir_resize(&*rgb8_view(&img), dw, dh).into_raw(),
                fir_resize_from_a_copy(&img.to_rgb8(), dw, dh).into_raw(),
                "{layout} {sw}x{sh} to {dw}x{dh}"
            );
        }
    }
}
