//! The shared ONNX plumbing: the process-wide runtime
//! environment, session building, output collection, the
//! model display name, and the resampling the providers'
//! preprocessing shares.

use image::imageops::FilterType;
use image::{ImageBuffer, Luma, Pixel, Rgb};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use super::AiError;

/// The locks [`lock_model`] has taken back from a panic since
/// boot, which its log line counts.
static RECOVERED: AtomicU64 = AtomicU64::new(0);

/// Lock one of a provider's mutexes: its model's session, or
/// the memo beside it. `what` names it for the log.
///
/// A request that panics while it holds such a lock poisons
/// it. A lock that failed from then on would leave the model
/// answering no request for the life of the process (issue
/// #715), and what these locks guard comes through a panic
/// whole: a session is the model's weights and its runtime
/// handle, which a run reads and never changes (the lock is
/// there because `Session::run` takes `&mut self`), and a
/// memo's value is replaced whole. So the guard is taken
/// back from the poison and the poison cleared, and the
/// recovery is counted and logged on stderr, once for each
/// panic.
pub(super) fn lock_model<'a, T>(lock: &'a Mutex<T>, what: &str) -> MutexGuard<'a, T> {
    lock_recovering(lock, what, &RECOVERED, |line| {
        // Not `eprintln!`, which panics when stderr is
        // closed: that panic would poison the lock again.
        let _ = writeln!(std::io::stderr(), "{line}");
    })
}

/// [`lock_model`], with the count and the log it reports a
/// recovery to named.
fn lock_recovering<'a, T>(
    lock: &'a Mutex<T>,
    what: &str,
    recovered: &AtomicU64,
    log: impl FnOnce(&str),
) -> MutexGuard<'a, T> {
    lock.lock().unwrap_or_else(|poisoned| {
        lock.clear_poison();
        let count = recovered.fetch_add(1, Ordering::Relaxed) + 1;
        log(&format!(
            "ai: a request panicked while it held {what}; the lock is recovered and the model \
             keeps serving (recoveries since boot: {count})"
        ));
        poisoned.into_inner()
    })
}

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

/// The one thread pool every model runs on (issue #693). ONNX
/// Runtime gives each session a pool of its own by default: a
/// thread per core, its workers pinned to cores 1 and up, and
/// spinning for a while after each operator before they sleep.
/// Two inferences at once, which the AI gate allows, then put
/// two pools' workers on the same cores, each pool's spinning
/// in the other's way and core 0 left to the callers: on four
/// cores, four AI requests at once took 1.28 times as long as
/// with one pool. The shared pool is the size and the spinning
/// of a session's own (a thread per core, spinning on), so one
/// inference alone runs as it did, and ort runs every session
/// built in this environment on it. Its threads are not pinned.
fn shared_thread_pool() -> ort::environment::GlobalThreadPoolOptions {
    ort::environment::GlobalThreadPoolOptions::default()
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
        .with_global_thread_pool(shared_thread_pool())
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
/// RGB model inputs and grayscale mask fields. The resizer
/// only reads its source, so it borrows the pixels (issue
/// #672): a copy of a 2400x1600 frame is 11.5 MB in front
/// of a model that answers in 5 ms.
pub(super) fn fir_resize<P: FirPixel>(
    img: &ImageBuffer<P, Vec<u8>>,
    w: u32,
    h: u32,
) -> ImageBuffer<P, Vec<u8>> {
    use fast_image_resize::images::{Image as FirImage, ImageRef as FirImageRef};
    use fast_image_resize::{ResizeAlg, ResizeOptions, Resizer};
    let fallback = || image::imageops::resize(img, w, h, FilterType::Triangle);
    let Ok(src) = FirImageRef::new(img.width(), img.height(), img.as_raw(), P::LAYOUT) else {
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

/// The frame's RGB8 pixels for a model's preprocessing:
/// borrowed when the frame already is RGB8, as every decoded
/// JPEG is, and converted otherwise. `to_rgb8` copies the
/// whole frame even when there is nothing to convert (issue
/// #672).
pub(super) fn rgb8_view(img: &image::DynamicImage) -> std::borrow::Cow<'_, image::RgbImage> {
    match img.as_rgb8() {
        Some(rgb) => std::borrow::Cow::Borrowed(rgb),
        None => std::borrow::Cow::Owned(img.to_rgb8()),
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

/// A session's first output, owned: the single-output models
/// (upscale, inpainting) and the detector read their result
/// from it.
pub(super) fn first_output(
    outputs: ort::session::SessionOutputs<'_>,
) -> Result<ort::value::DynValue, AiError> {
    outputs
        .into_iter()
        .next()
        .map(|(_, value)| value)
        .ok_or_else(|| AiError::BadOutput("the model produced no outputs".to_string()))
}

/// A model run's result as the providers read it: one f32
/// tensor, shape and data, borrowed in place. The session's
/// output implements it; so does a test's fake model, which
/// lets the pre- and post-processing around a run be tested
/// without the runtime.
pub(super) trait TensorOutput {
    fn f32_tensor(&self) -> Result<(&[i64], &[f32]), AiError>;
}

impl TensorOutput for ort::value::DynValue {
    fn f32_tensor(&self) -> Result<(&[i64], &[f32]), AiError> {
        let (shape, data) = self
            .try_extract_tensor::<f32>()
            .map_err(|e| AiError::BadOutput(format!("cannot read the model output: {e}")))?;
        Ok((shape, data))
    }
}

/// A fake model's output: dims and data.
#[cfg(test)]
impl TensorOutput for (Vec<i64>, Vec<f32>) {
    fn f32_tensor(&self) -> Result<(&[i64], &[f32]), AiError> {
        Ok((&self.0, &self.1))
    }
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
