use super::*;
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
