//! Helpers the AI providers' unit tests share.

use crate::ai::AiError;
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
