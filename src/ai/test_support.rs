//! Helpers the AI providers' unit tests share.

use crate::ai::AiError;
use image::DynamicImage;

pub(super) fn solid(r: u8, g: u8, b: u8, w: u32, h: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([r, g, b])))
}

/// Write bytes to a fresh temp file and return its
/// path: the load() failure tests need real files (a
/// garbage model) without committing fixtures. The tag
/// keeps parallel tests on distinct files.
pub(super) fn temp_file(tag: &str, contents: &[u8]) -> String {
    let path =
        std::env::temp_dir().join(format!("crabcrop-ai-load-{tag}-{}.bin", std::process::id()));
    std::fs::write(&path, contents).unwrap();
    path.to_string_lossy().into_owned()
}

/// A path inside the temp dir that is never created.
pub(super) fn missing_path(tag: &str) -> String {
    std::env::temp_dir()
        .join(format!(
            "crabcrop-ai-load-missing-{tag}-{}.onnx",
            std::process::id()
        ))
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
