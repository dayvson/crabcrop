//! The shared ONNX plumbing: the process-wide runtime
//! environment, session building, output collection, the
//! model display name, and the resampling the providers'
//! preprocessing shares.

use image::imageops::FilterType;
use image::{ImageBuffer, Luma, Pixel, Rgb};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::sync::OnceLock;

use super::AiError;

/// The process-wide ONNX environment, created on first use:
/// in practice that is at boot, when the server preloads its
/// configured models.
static ENVIRONMENT: OnceLock<ort::environment::Environment> = OnceLock::new();

/// The dylib path `ort` searches when no explicit path is
/// configured, resolved the way ort resolves it: the
/// `ORT_DYLIB_PATH` variable wins when set and non-empty;
/// otherwise the platform's library name (`libonnxruntime.so`
/// on Linux), which the dynamic loader searches normally.
/// Pure, so the resolution is testable without the runtime.
fn ort_search_path(ort_dylib_path: Option<&str>) -> std::path::PathBuf {
    match ort_dylib_path {
        Some(path) if !path.is_empty() => std::path::PathBuf::from(path),
        _ => std::path::PathBuf::from(format!(
            "{}onnxruntime{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        )),
    }
}

fn environment(dylib: Option<&str>) -> Result<&'static ort::environment::Environment, AiError> {
    if let Some(env) = ENVIRONMENT.get() {
        return Ok(env);
    }
    // Load the runtime explicitly, from the configured path
    // or from the path ort itself would search. `ort::init()`
    // defers that load to the first API call, where a missing
    // runtime panics inside ort; `init_from` returns the
    // failure as an error, so a configured model with no
    // loadable runtime fails boot with the diagnostic below
    // instead of a panic (#180).
    let path = match dylib {
        Some(path) => std::path::PathBuf::from(path),
        None => ort_search_path(std::env::var("ORT_DYLIB_PATH").ok().as_deref()),
    };
    let builder = ort::init_from(&path).map_err(|e| {
        AiError::Load(format!(
            "cannot load the ONNX runtime from {}: {e}",
            path.display()
        ))
    })?;
    let env = builder
        .build()
        .map_err(|e| AiError::Load(format!("cannot initialize the ONNX runtime: {e}")))?;
    let _ = ENVIRONMENT.set(env);
    ENVIRONMENT
        .get()
        .ok_or_else(|| AiError::Load("ONNX environment unavailable".to_string()))
}

/// Build an inference session for a model file and report its
/// first input's name. Every failure is a plain message: at
/// boot it becomes the startup diagnostic.
pub(super) fn build_session(
    dylib: Option<&str>,
    model_path: &str,
) -> Result<(Session, String), AiError> {
    let env = environment(dylib)?;
    let session = Session::builder(env)
        .map_err(|e| AiError::Load(format!("cannot create a session: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| AiError::Load(format!("cannot configure the session: {e}")))?
        .commit_from_file(model_path)
        .map_err(|e| AiError::Load(format!("cannot load the model {model_path}: {e}")))?;
    let input_name = session
        .inputs()
        .first()
        .map(|i| i.name().to_string())
        .ok_or_else(|| AiError::Load(format!("the model {model_path} has no inputs")))?;
    Ok((session, input_name))
}

/// The pixel layouts `fir_resize` takes, each paired with
/// its fast_image_resize layout: RGB model inputs and
/// grayscale mask fields.
pub(super) trait FirPixel: Pixel<Subpixel = u8> + 'static {
    const LAYOUT: fast_image_resize::PixelType;
}

impl FirPixel for Rgb<u8> {
    const LAYOUT: fast_image_resize::PixelType = fast_image_resize::PixelType::U8x3;
}

impl FirPixel for Luma<u8> {
    const LAYOUT: fast_image_resize::PixelType = fast_image_resize::PixelType::U8;
}

/// Resample with fast_image_resize's Bilinear kernel instead
/// of imageops Triangle (issue #283). The AI providers'
/// resampling cost more than the small models' inference
/// (YuNet: 57 ms of resize against 44 ms of inference), and
/// the SIMD kernel does the same resizes 7 to 12 times
/// faster. The kernel change moves model inputs and mask
/// edges by a few levels of 255 at most, measured on the
/// functional corpus and confirmed visually; the issue
/// carries the bounds. Falls back to the Triangle path if
/// the SIMD resizer rejects the geometry. One body serves
/// RGB model inputs and grayscale mask fields.
pub(super) fn fir_resize<P: FirPixel>(
    img: &ImageBuffer<P, Vec<u8>>,
    w: u32,
    h: u32,
) -> ImageBuffer<P, Vec<u8>> {
    use fast_image_resize::images::Image as FirImage;
    use fast_image_resize::{ResizeAlg, ResizeOptions, Resizer};
    let fallback = || image::imageops::resize(img, w, h, FilterType::Triangle);
    let Ok(src) = FirImage::from_vec_u8(img.width(), img.height(), img.as_raw().clone(), P::LAYOUT)
    else {
        return fallback();
    };
    let mut dst = FirImage::new(w, h, P::LAYOUT);
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(
        fast_image_resize::FilterType::Bilinear,
    ));
    match Resizer::new().resize(&src, &mut dst, &opts) {
        Ok(()) => ImageBuffer::from_raw(w, h, dst.into_vec()).unwrap_or_else(fallback),
        Err(_) => fallback(),
    }
}

/// One model output: name, dims, data.
pub(super) type ModelOutput = (String, Vec<i64>, Vec<f32>);

/// Collect a session's outputs as (name, dims, data).
pub(super) fn collect_outputs(
    outputs: ort::session::SessionOutputs<'_>,
) -> Result<Vec<ModelOutput>, AiError> {
    let mut out = Vec::new();
    for (name, value) in outputs.into_iter() {
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| AiError::BadOutput(format!("cannot read the model output: {e}")))?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        out.push((name.to_string(), dims, data.to_vec()));
    }
    Ok(out)
}

/// The name meta reports for a configured model file:
/// the file name, never the operator's filesystem
/// layout. The OCR providers have always reported it
/// this way (issue #268); every model identity in a
/// meta response follows the same rule.
pub(super) fn model_display_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

#[cfg(test)]
mod tests;
