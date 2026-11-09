//! Helpers the AI providers' unit tests share.

use crate::ai::AiError;
use crate::ai::depth::SampleDepth;
use crate::test_support::{ScratchDir, scratch_dir};
use image::DynamicImage;

pub(super) fn solid(r: u8, g: u8, b: u8, w: u32, h: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([r, g, b])))
}

/// Write bytes to a fresh temp file and return the
/// scratch directory it lives in and its path: the
/// load() failure tests need real files (a garbage
/// model) without committing fixtures. The file lives
/// as long as the directory does, and no test sees a
/// file an earlier test process left (issue #526).
pub(super) fn temp_file(tag: &str, contents: &[u8]) -> (ScratchDir, String) {
    let dir = scratch_dir(&format!("ai-load-{tag}"));
    let path = dir
        .join(format!("{tag}.bin"))
        .to_string_lossy()
        .into_owned();
    std::fs::write(&path, contents).unwrap();
    (dir, path)
}

/// A path that does not exist: a name inside a scratch
/// directory that is removed again before this returns.
pub(super) fn missing_path(tag: &str) -> String {
    scratch_dir(&format!("ai-load-missing-{tag}"))
        .join(format!("{tag}.onnx"))
        .to_string_lossy()
        .into_owned()
}

/// Assert a load() result failed as AiError::Load, the
/// class boot turns into a startup diagnostic, and
/// return the message for wording checks.
pub(super) fn expect_load<T>(result: Result<T, AiError>) -> String {
    match result {
        Err(AiError::Load(message)) => message,
        Err(other) => panic!("expected a load error, got {other}"),
        Ok(_) => panic!("expected a load error, but the load succeeded"),
    }
}

/// The color the RGBA fixtures below store in their opaque
/// half: 16-bit samples off the 8-bit grid, so a result that
/// passed through 8 bits on the way shows it (issue #551).
pub(super) const OPAQUE16: [u16; 3] = [0x1234, 0x5678, 0x9abc];

/// An RGBA source at the given depth: the left half fully
/// transparent with saturated red stored under it (what a
/// design tool's export can leave there), the right half
/// opaque in [`OPAQUE16`], rounded to 8 bits for an 8-bit
/// source and exact as floats for a float one.
pub(super) fn rgba_halves(depth: SampleDepth, w: u32, h: u32) -> DynamicImage {
    let sixteen = image::ImageBuffer::from_fn(w, h, |x, _| {
        if x < w / 2 {
            image::Rgba([u16::MAX, 0, 0, 0])
        } else {
            image::Rgba([OPAQUE16[0], OPAQUE16[1], OPAQUE16[2], u16::MAX])
        }
    });
    let img = DynamicImage::ImageRgba16(sixteen);
    match depth {
        SampleDepth::Eight => DynamicImage::ImageRgba8(img.to_rgba8()),
        SampleDepth::Sixteen => img,
        SampleDepth::Float => DynamicImage::ImageRgba32F(img.to_rgba32f()),
    }
}
