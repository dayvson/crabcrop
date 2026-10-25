//! In-process AI models (ONNX). The engine stays free of the
//! runtime: it depends only on the provider traits defined
//! here. The server builds the real providers at boot from
//! operator configuration (a model is preloaded when one is
//! configured, and a broken configuration fails startup).
//!
//! The plumbing is shared with AI upscale: a model is loaded
//! once against a process-wide ONNX environment, and inference
//! runs behind the server-wide concurrency gate.

use image::{DynamicImage, GrayImage, imageops::FilterType};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::sync::{Mutex, OnceLock};

/// How an AI operation failed, typed so callers can map the
/// failure class to a response instead of collapsing every
/// cause into one 500 (AP-16). The messages keep the wording
/// the plain strings carried; the variant is the diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AiError {
    /// A model, the runtime, or a model-adjacent file (labels,
    /// tokenizer data) could not be loaded. Raised at boot,
    /// where it fails startup, or while preparing a provider.
    Load(String),
    /// The runtime failed while running a model: building an
    /// input tensor, executing the session, a transient ORT
    /// error.
    Inference(String),
    /// A provider's session lock was poisoned: an earlier
    /// inference panicked while holding it, so the session's
    /// state can no longer be trusted.
    LockPoisoned,
    /// The model ran but its output is not what the provider
    /// contract requires: no outputs, an unreadable tensor,
    /// an unexpected shape, an embedding of the wrong size.
    BadOutput(String),
}

impl std::fmt::Display for AiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AiError::Load(cause) => write!(f, "{cause}"),
            AiError::Inference(cause) => write!(f, "{cause}"),
            AiError::LockPoisoned => write!(f, "a model session lock was poisoned"),
            AiError::BadOutput(cause) => write!(f, "{cause}"),
        }
    }
}

impl std::error::Error for AiError {}

/// Produces a foreground mask for an image: bright where the
/// subject is, dark where the background is. Implemented by the
/// ONNX provider in production and by fakes in tests.
pub trait MaskProvider: Send + Sync + std::fmt::Debug {
    fn mask(&self, img: &DynamicImage) -> Result<GrayImage, AiError>;

    /// The provider's confidence in the mask it produced for
    /// this image, as a probability-like score in 0..=1, when
    /// the provider's output supports one (issue #197: the
    /// studio record in meta reports it). The default is
    /// `None`, and the ONNX provider keeps it: its mask field
    /// is min-max normalized before it is returned, which is
    /// a contrast stretch, not a probability, so reporting a
    /// number would overstate what the model said.
    fn confidence(&self, _img: &DynamicImage, _mask: &GrayImage) -> Option<f32> {
        None
    }
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
fn build_session(dylib: Option<&str>, model_path: &str) -> Result<(Session, String), AiError> {
    let env = environment(dylib)?;
    let session = Session::builder(env)
        .map_err(|e| AiError::Load(format!("cannot create a session: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
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

/// Background removal backed by a U²-Net-family ONNX model
/// (single 320x320 input), e.g. the u2netp model the `ai`
/// container image ships.
pub struct OnnxMaskProvider {
    session: Mutex<Session>,
    input_name: String,
}

impl OnnxMaskProvider {
    /// Load a model file against the runtime at `dylib` (or the
    /// runtime's own search when `None`). A probe inference on a
    /// blank image runs at load, like the other providers: a
    /// model whose output does not fit the mask contract (a
    /// 320x320 field) is rejected here, at boot, with a Load
    /// error, not on the first removebg request (audit R2-28).
    pub fn load(dylib: Option<&str>, model_path: &str) -> Result<Self, AiError> {
        let (session, input_name) = build_session(dylib, model_path)?;
        let provider = Self {
            session: Mutex::new(session),
            input_name,
        };
        provider.mask(&DynamicImage::new_rgb8(8, 8)).map_err(|e| {
            AiError::Load(format!("the model {model_path} failed its mask probe: {e}"))
        })?;
        Ok(provider)
    }
}

impl std::fmt::Debug for OnnxMaskProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxMaskProvider")
            .field("input_name", &self.input_name)
            .finish_non_exhaustive()
    }
}

impl MaskProvider for OnnxMaskProvider {
    fn mask(&self, img: &DynamicImage) -> Result<GrayImage, AiError> {
        let input = preprocess(img);
        let tensor =
            ort::value::Tensor::from_array(([1, 3, 320, 320], input.into_boxed_slice()))
                .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
        let mut session = self.session.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
        let (_, first) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| AiError::BadOutput("the model produced no outputs".to_string()))?;
        let (_shape, data) = first
            .try_extract_tensor::<f32>()
            .map_err(|e| AiError::BadOutput(format!("cannot read the model output: {e}")))?;
        postprocess(data, img.width(), img.height())
    }
}

/// U²-Net preprocessing, matching the reference pipeline:
/// resize to the model's 320x320, scale by the image maximum,
/// then normalize per channel with ImageNet statistics. Output
/// is planar CHW.
fn preprocess(img: &DynamicImage) -> Vec<f32> {
    let small = img.resize_exact(320, 320, FilterType::Triangle).to_rgb8();
    let max = small.pixels().flat_map(|p| p.0).max().unwrap_or(0);
    let scale = if max == 0 { 1.0 } else { max as f32 };
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let mut out = vec![0f32; 3 * 320 * 320];
    for (i, px) in small.pixels().enumerate() {
        for (c, (mean, std)) in MEAN.iter().zip(STD.iter()).enumerate() {
            out[c * 320 * 320 + i] = ((px[c] as f32 / scale) - mean) / std;
        }
    }
    out
}

/// Turn the model's first output into a mask: min-max
/// normalize to 0..255 (the reference pipeline's behavior),
/// then resize back to the source dimensions. The output is
/// indexed as a 320x320 field, so its length is checked
/// first: an export producing fewer values is a BadOutput,
/// never an indexing panic under the session lock (audit
/// R2-28).
fn postprocess(data: &[f32], w: u32, h: u32) -> Result<GrayImage, AiError> {
    if data.len() < 320 * 320 {
        return Err(AiError::BadOutput(format!(
            "the mask model produced {} output values; the contract takes a 320x320 field ({})",
            data.len(),
            320 * 320
        )));
    }
    let (min, max) = data
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let range = (max - min).max(f32::EPSILON);
    let mask = GrayImage::from_fn(320, 320, |x, y| {
        let v = data[(y * 320 + x) as usize];
        image::Luma([((v - min) / range * 255.0).round() as u8])
    });
    Ok(if (w, h) == (320, 320) {
        mask
    } else {
        image::imageops::resize(&mask, w, h, FilterType::Triangle)
    })
}

/// Apply a foreground mask: the output alpha is the source
/// alpha scaled by the mask, so the subject stays and the
/// background becomes transparent.
pub fn apply_mask(img: &DynamicImage, mask: &GrayImage) -> DynamicImage {
    let mut rgba = img.to_rgba8();
    let fitted;
    let mask = if mask.width() == rgba.width() && mask.height() == rgba.height() {
        mask
    } else {
        fitted = image::imageops::resize(mask, rgba.width(), rgba.height(), FilterType::Triangle);
        &fitted
    };
    for (px, m) in rgba.pixels_mut().zip(mask.pixels()) {
        px[3] = ((px[3] as u16 * m[0] as u16) / 255) as u8;
    }
    DynamicImage::ImageRgba8(rgba)
}

/// Enlarges an image with a super-resolution model. The model
/// has a native factor; the engine composes runs and trims to
/// the exact requested size.
pub trait UpscaleProvider: Send + Sync + std::fmt::Debug {
    /// The model's native upscale factor (2, 3, 4...).
    fn factor(&self) -> u32;
    fn upscale(&self, img: &DynamicImage) -> Result<DynamicImage, AiError>;
}

/// Super-resolution backed by an ONNX model with the usual
/// contract: dynamic-size RGB in ([1, 3, H, W], values 0..1),
/// the same image at an integer factor out. Real-ESRGAN-family
/// ONNX exports follow it.
pub struct OnnxUpscaleProvider {
    session: Mutex<Session>,
    input_name: String,
    factor: u32,
}

impl OnnxUpscaleProvider {
    /// Load a model file. The native factor is not declared
    /// anywhere in the format, so it is probed at load with a
    /// 16x16 image: a model that does not enlarge by an integer
    /// factor is rejected here, at boot, not per request.
    pub fn load(dylib: Option<&str>, model_path: &str) -> Result<Self, AiError> {
        let (session, input_name) = build_session(dylib, model_path)?;
        let probe = Self {
            session: Mutex::new(session),
            input_name,
            factor: 0,
        };
        // The probe runs as part of loading, so a model that
        // cannot run it at all has failed to load, not failed
        // an inference: classify it Load, as the detection
        // provider's load probe does.
        let out = probe
            .upscale(&DynamicImage::new_rgb8(16, 16))
            .map_err(|e| {
                AiError::Load(format!(
                    "the model {model_path} failed its super-resolution probe: {e}"
                ))
            })?;
        if out.width() % 16 != 0 || out.width() / 16 < 2 || out.height() != out.width() {
            return Err(AiError::Load(format!(
                "the model {model_path} is not an integer-factor super-resolution model \
                 (a 16x16 probe produced {}x{})",
                out.width(),
                out.height()
            )));
        }
        Ok(Self {
            factor: out.width() / 16,
            ..probe
        })
    }
}

impl std::fmt::Debug for OnnxUpscaleProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxUpscaleProvider")
            .field("input_name", &self.input_name)
            .field("factor", &self.factor)
            .finish_non_exhaustive()
    }
}

impl UpscaleProvider for OnnxUpscaleProvider {
    fn factor(&self) -> u32 {
        self.factor
    }

    fn upscale(&self, img: &DynamicImage) -> Result<DynamicImage, AiError> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = img.to_rgb8();
        let mut input = vec![0f32; 3 * w * h];
        for (i, px) in rgb.pixels().enumerate() {
            for c in 0..3 {
                input[c * w * h + i] = px[c] as f32 / 255.0;
            }
        }
        let tensor = ort::value::Tensor::from_array(([1, 3, h, w], input.into_boxed_slice()))
            .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
        let mut session = self.session.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
        let (_, first) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| AiError::BadOutput("the model produced no outputs".to_string()))?;
        let (shape, data) = first
            .try_extract_tensor::<f32>()
            .map_err(|e| AiError::BadOutput(format!("cannot read the model output: {e}")))?;
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
            return Err(AiError::BadOutput(format!(
                "unexpected model output shape: {shape:?}"
            )));
        }
        let (ow, oh) = (shape[3] as u32, shape[2] as u32);
        let mut out = image::RgbImage::new(ow, oh);
        for (i, px) in out.pixels_mut().enumerate() {
            for c in 0..3 {
                let v = data[c * (ow * oh) as usize + i];
                px[c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        let out = DynamicImage::ImageRgb8(out);
        // The model sees RGB only; carry the source alpha over
        // with a plain resize so transparency survives.
        if matches!(
            img,
            DynamicImage::ImageRgba8(_) | DynamicImage::ImageLumaA8(_)
        ) {
            let src = img.to_rgba8();
            let alpha = GrayImage::from_fn(img.width(), img.height(), |x, y| {
                image::Luma([src.get_pixel(x, y)[3]])
            });
            let alpha = image::imageops::resize(&alpha, ow, oh, FilterType::Triangle);
            let mut rgba = out.to_rgba8();
            for (px, a) in rgba.pixels_mut().zip(alpha.pixels()) {
                px[3] = a[0];
            }
            return Ok(DynamicImage::ImageRgba8(rgba));
        }
        Ok(out)
    }
}

/// A detected face, in source-image pixels, with the
/// detector's confidence in 0.0..=1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub score: f32,
}

/// Detects faces in an image. Implemented by the ONNX provider
/// in production and by fakes in tests.
pub trait FaceProvider: Send + Sync + std::fmt::Debug {
    fn detect(&self, img: &DynamicImage) -> Result<Vec<FaceBox>, AiError>;
}

/// Face detection backed by YuNet (the 2023mar ONNX export):
/// a fixed 640x640 input, three stride heads (8/16/32), and
/// boxes regressed directly from cell offsets, decoded exactly
/// as the reference implementation does. Detections below a
/// 0.6 confidence are dropped, then non-maximum suppression at
/// 0.3 IoU keeps the best box of each cluster.
pub struct OnnxFaceProvider {
    session: Mutex<Session>,
    input_name: String,
}

impl OnnxFaceProvider {
    pub fn load(dylib: Option<&str>, model_path: &str) -> Result<Self, AiError> {
        let (session, input_name) = build_session(dylib, model_path)?;
        Ok(Self {
            session: Mutex::new(session),
            input_name,
        })
    }
}

impl std::fmt::Debug for OnnxFaceProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxFaceProvider")
            .field("input_name", &self.input_name)
            .finish_non_exhaustive()
    }
}

/// The export's fixed input edge.
const FACE_INPUT: u32 = 640;
/// Confidence floor, matching the reference defaults.
const FACE_SCORE_THRESHOLD: f32 = 0.6;
/// NMS IoU ceiling, matching the reference defaults.
const FACE_NMS_THRESHOLD: f32 = 0.3;

impl FaceProvider for OnnxFaceProvider {
    fn detect(&self, img: &DynamicImage) -> Result<Vec<FaceBox>, AiError> {
        let (ow, oh) = (img.width(), img.height());
        if ow == 0 || oh == 0 {
            return Ok(Vec::new());
        }
        // The export takes exactly 640x640. The image is scaled
        // to fit (the reference wrapper resizes the same way);
        // boxes map back with independent axis scales.
        let small = img
            .resize_exact(FACE_INPUT, FACE_INPUT, FilterType::Triangle)
            .to_rgb8();
        // BGR planar floats at 0..255: the channel order and
        // scale the model was trained with.
        let plane = (FACE_INPUT * FACE_INPUT) as usize;
        let mut input = vec![0f32; 3 * plane];
        for (i, px) in small.pixels().enumerate() {
            input[i] = px[2] as f32;
            input[plane + i] = px[1] as f32;
            input[2 * plane + i] = px[0] as f32;
        }
        let tensor = ort::value::Tensor::from_array((
            [1, 3, FACE_INPUT as usize, FACE_INPUT as usize],
            input.into_boxed_slice(),
        ))
        .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
        let mut session = self.session.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
        let mut by_name: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::new();
        for (name, value) in outputs.into_iter() {
            let (_, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| AiError::BadOutput(format!("cannot read the model output: {e}")))?;
            by_name.insert(name.to_string(), data.to_vec());
        }
        drop(session);
        let candidates = decode_face_candidates(&by_name, ow, oh, FACE_SCORE_THRESHOLD)?;
        Ok(nms(candidates))
    }
}

/// Decode YuNet's three stride heads into candidate boxes
/// in source-image coordinates, before NMS: for each cell,
/// the geometric mean of the clamped class and objectness
/// scores must clear `threshold`; box centers are cell
/// offsets within the 640x640 model space, sizes are
/// exponentiated log sizes, and both map back with
/// independent axis scales, clamped to the image. A
/// missing head is a BadOutput, as it was when this ran
/// inline in `detect`.
fn decode_face_candidates(
    outputs: &std::collections::HashMap<String, Vec<f32>>,
    ow: u32,
    oh: u32,
    threshold: f32,
) -> Result<Vec<FaceBox>, AiError> {
    let get = |name: &str| -> Result<&Vec<f32>, AiError> {
        outputs
            .get(name)
            .ok_or_else(|| AiError::BadOutput(format!("the model produced no {name} output")))
    };
    let mut candidates: Vec<FaceBox> = Vec::new();
    for stride in [8u32, 16, 32] {
        let cls = get(&format!("cls_{stride}"))?;
        let obj = get(&format!("obj_{stride}"))?;
        let bbox = get(&format!("bbox_{stride}"))?;
        let grid = FACE_INPUT / stride;
        for r in 0..grid {
            for c in 0..grid {
                let idx = (r * grid + c) as usize;
                let score = (cls[idx].clamp(0.0, 1.0) * obj[idx].clamp(0.0, 1.0)).sqrt();
                if score < threshold {
                    continue;
                }
                let cx = (c as f32 + bbox[idx * 4]) * stride as f32;
                let cy = (r as f32 + bbox[idx * 4 + 1]) * stride as f32;
                let w = bbox[idx * 4 + 2].exp() * stride as f32;
                let h = bbox[idx * 4 + 3].exp() * stride as f32;
                // Map back to source coordinates, clamped
                // to the image: boxes may hang off an edge.
                let fx = (cx - w / 2.0) * ow as f32 / FACE_INPUT as f32;
                let fy = (cy - h / 2.0) * oh as f32 / FACE_INPUT as f32;
                let fw = w * ow as f32 / FACE_INPUT as f32;
                let fh = h * oh as f32 / FACE_INPUT as f32;
                let x = fx.round().clamp(0.0, ow as f32) as u32;
                let y = fy.round().clamp(0.0, oh as f32) as u32;
                candidates.push(FaceBox {
                    x,
                    y,
                    w: (fw.round() as u32).min(ow - x),
                    h: (fh.round() as u32).min(oh - y),
                    score,
                });
            }
        }
    }
    Ok(candidates)
}

/// Greedy non-maximum suppression: best score first, a
/// candidate survives only if it overlaps every kept box by
/// at most the IoU threshold.
fn nms(mut boxes: Vec<FaceBox>) -> Vec<FaceBox> {
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<FaceBox> = Vec::new();
    for b in boxes {
        if kept.iter().all(|k| iou(k, &b) <= FACE_NMS_THRESHOLD) {
            kept.push(b);
        }
    }
    kept
}

fn iou(a: &FaceBox, b: &FaceBox) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let inter = x2.saturating_sub(x1) as f32 * y2.saturating_sub(y1) as f32;
    let union = (a.w * a.h + b.w * b.h) as f32 - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(r: u8, g: u8, b: u8, w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([r, g, b])))
    }

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
    fn apply_mask_scales_alpha_by_the_mask() {
        // Left half background (mask 0), right half subject (255).
        let mask = GrayImage::from_fn(4, 2, |x, _| image::Luma([if x < 2 { 0 } else { 255 }]));
        let out = apply_mask(&solid(200, 30, 30, 4, 2), &mask).to_rgba8();
        assert_eq!(out.get_pixel(0, 0)[3], 0);
        assert_eq!(out.get_pixel(3, 0)[3], 255);
        // Color is untouched.
        assert_eq!(out.get_pixel(3, 0)[0], 200);
    }

    #[test]
    fn apply_mask_respects_existing_alpha() {
        let mut img = DynamicImage::new_rgba8(2, 1);
        img.as_mut_rgba8()
            .unwrap()
            .put_pixel(0, 0, image::Rgba([10, 20, 30, 128]));
        let mask = GrayImage::from_pixel(2, 1, image::Luma([128]));
        let out = apply_mask(&img, &mask).to_rgba8();
        // 128 * 128 / 255 = 64.
        assert_eq!(out.get_pixel(0, 0)[3], 64);
    }

    #[test]
    fn preprocess_produces_a_normalized_chw_tensor() {
        let input = preprocess(&solid(255, 255, 255, 8, 8));
        assert_eq!(input.len(), 3 * 320 * 320);
        // White scales to 1.0 by the max, then normalizes:
        // (1 - 0.485) / 0.229 = 2.2489... in the red channel.
        assert!((input[0] - 2.2489).abs() < 0.001, "got {}", input[0]);
        // A black image must not divide by zero; its red channel
        // is (0 - 0.485) / 0.229 = -2.1179...
        let input = preprocess(&solid(0, 0, 0, 8, 8));
        assert!((input[0] - -2.1179).abs() < 0.001, "got {}", input[0]);
    }

    /// Real-model verification, run manually with the deployment
    /// artifacts in place:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_FACE_MODEL=...
    /// cargo test -- --ignored face_detection_with_the_real_model
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn face_detection_with_the_real_model() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_FACE_MODEL").unwrap();
        let provider = OnnxFaceProvider::load(Some(&dylib), &model).unwrap();
        let bytes = include_bytes!("../tests/fixtures/portrait.jpg");
        let img = image::load_from_memory(bytes).unwrap();
        let faces = provider.detect(&img).unwrap();
        assert_eq!(faces.len(), 1, "faces: {faces:?}");
        let f = faces[0];
        assert!(f.score > 0.8, "score {}", f.score);
        // The fixture portrait's face, measured: the box lands
        // on it with room for build-to-build wobble.
        assert!((130..=175).contains(&f.x), "x {}", f.x);
        assert!((15..=50).contains(&f.y), "y {}", f.y);
        assert!((70..=110).contains(&f.w), "w {}", f.w);
        assert!((115..=165).contains(&f.h), "h {}", f.h);
        // A flat image has no faces.
        assert!(
            provider
                .detect(&solid(10, 20, 30, 120, 90))
                .unwrap()
                .is_empty()
        );
    }

    /// Real-model verification, run manually with the deployment
    /// artifacts in place:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_UPSCALE_MODEL=...
    /// cargo test -- --ignored upscale_with_the_real_model
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn upscale_with_the_real_model() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_UPSCALE_MODEL").unwrap();
        let provider = OnnxUpscaleProvider::load(Some(&dylib), &model).unwrap();
        assert!(provider.factor() >= 2, "factor {}", provider.factor());
        let out = provider.upscale(&solid(120, 30, 200, 24, 18)).unwrap();
        assert_eq!(out.width(), 24 * provider.factor());
        assert_eq!(out.height(), 18 * provider.factor());
    }

    /// Real-model verification, run manually with the deployment
    /// artifacts in place:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
    /// cargo test -- --ignored removebg_with_the_real_model
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn removebg_with_the_real_model() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_REMOVEBG_MODEL").unwrap();
        let provider = OnnxMaskProvider::load(Some(&dylib), &model).unwrap();
        // A dark square on a light background: the model should
        // call the square the subject.
        let mut img = DynamicImage::new_rgb8(200, 200);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if (60..140).contains(&x) && (60..140).contains(&y) {
                image::Rgb([30, 30, 30])
            } else {
                image::Rgb([240, 240, 240])
            };
        }
        let mask = provider.mask(&img).unwrap();
        assert_eq!((mask.width(), mask.height()), (200, 200));
        assert!(
            mask.get_pixel(100, 100)[0] > 200,
            "center (subject) mask = {}",
            mask.get_pixel(100, 100)[0]
        );
        assert!(
            mask.get_pixel(5, 5)[0] < 55,
            "corner (background) mask = {}",
            mask.get_pixel(5, 5)[0]
        );
    }

    /// Write bytes to a fresh temp file and return its
    /// path: the load() failure tests need real files (a
    /// garbage model) without committing fixtures. The tag
    /// keeps parallel tests on distinct files.
    fn temp_file(tag: &str, contents: &[u8]) -> String {
        let path =
            std::env::temp_dir().join(format!("crabcrop-ai-load-{tag}-{}.bin", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// A path inside the temp dir that is never created.
    fn missing_path(tag: &str) -> String {
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
    fn expect_load<T>(result: Result<T, AiError>) -> String {
        match result {
            Err(AiError::Load(message)) => message,
            Err(other) => panic!("expected a load error, got {other}"),
            Ok(_) => panic!("expected a load error, but the load succeeded"),
        }
    }

    /// Load-failure coverage (#179, item 2): a missing model
    /// file fails as AiError::Load. Ignored like the other
    /// real-model tests: load() initializes the runtime
    /// before it reaches the file, so a configured dylib is
    /// required to exercise the file path at all.
    /// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored face_load_missing
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn face_load_missing_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let message = expect_load(OnnxFaceProvider::load(Some(&dylib), &missing_path("face")));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// A file that exists but is not a model fails the same
    /// way: the session builder rejects it as Load.
    /// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored face_load_garbage
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn face_load_garbage_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let garbage = temp_file("face-garbage", b"this is not an onnx model at all");
        let message = expect_load(OnnxFaceProvider::load(Some(&dylib), &garbage));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// A valid model with no inputs cannot serve as a face
    /// detector: load() reports it as Load, not a session
    /// with nothing to run.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_NO_INPUT_MODEL=...
    /// cargo test -- --ignored face_load_model_without_inputs
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn face_load_model_without_inputs_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_NO_INPUT_MODEL").unwrap();
        let message = expect_load(OnnxFaceProvider::load(Some(&dylib), &model));
        assert!(message.contains("has no inputs"), "{message}");
    }

    /// The mask provider shares the session builder with
    /// the face provider; its missing-file failure is the
    /// same Load.
    /// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored mask_load_missing
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn mask_load_missing_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let message = expect_load(OnnxMaskProvider::load(Some(&dylib), &missing_path("mask")));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored mask_load_garbage
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn mask_load_garbage_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let garbage = temp_file("mask-garbage", b"this is not an onnx model at all");
        let message = expect_load(OnnxMaskProvider::load(Some(&dylib), &garbage));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_NO_INPUT_MODEL=...
    /// cargo test -- --ignored mask_load_model_without_inputs
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn mask_load_model_without_inputs_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_NO_INPUT_MODEL").unwrap();
        let message = expect_load(OnnxMaskProvider::load(Some(&dylib), &model));
        assert!(message.contains("has no inputs"), "{message}");
    }

    /// The upscale provider's missing-file failure is Load,
    /// before its probe ever runs.
    /// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored upscale_load_missing
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn upscale_load_missing_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let message = expect_load(OnnxUpscaleProvider::load(
            Some(&dylib),
            &missing_path("upscale"),
        ));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored upscale_load_garbage
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn upscale_load_garbage_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let garbage = temp_file("upscale-garbage", b"this is not an onnx model at all");
        let message = expect_load(OnnxUpscaleProvider::load(Some(&dylib), &garbage));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// A face model cannot run the load probe (its fixed
    /// 640x640 input rejects the 16x16 probe): the probe is
    /// part of loading, so the failure is Load, not a
    /// request-time Inference.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_FACE_MODEL=...
    /// cargo test -- --ignored upscale_load_face_model
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn upscale_load_face_model_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_FACE_MODEL").unwrap();
        expect_load(OnnxUpscaleProvider::load(Some(&dylib), &model));
    }

    /// Same probe classification for a detection model.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_DETECT_MODEL=...
    /// cargo test -- --ignored upscale_load_detection_model
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn upscale_load_detection_model_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_DETECT_MODEL").unwrap();
        expect_load(OnnxUpscaleProvider::load(Some(&dylib), &model));
    }

    /// A model that runs the probe but does not enlarge
    /// (an identity export) is rejected by the factor
    /// check, as Load with the reason spelled out.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_IDENTITY_MODEL=...
    /// cargo test -- --ignored upscale_load_non_upscaling
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn upscale_load_non_upscaling_model_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_IDENTITY_MODEL").unwrap();
        let message = expect_load(OnnxUpscaleProvider::load(Some(&dylib), &model));
        assert!(
            message.contains("not an integer-factor super-resolution model"),
            "{message}"
        );
    }

    /// Build a full YuNet output set with every head silent
    /// (score 0 everywhere), ready for peaks to be planted.
    fn silent_face_outputs() -> std::collections::HashMap<String, Vec<f32>> {
        let mut outputs = std::collections::HashMap::new();
        for stride in [8u32, 16, 32] {
            let grid = (FACE_INPUT / stride) as usize;
            outputs.insert(format!("cls_{stride}"), vec![0f32; grid * grid]);
            outputs.insert(format!("obj_{stride}"), vec![0f32; grid * grid]);
            outputs.insert(format!("bbox_{stride}"), vec![0f32; grid * grid * 4]);
        }
        outputs
    }

    /// Plant a detection peak: class/objectness scores and
    /// the raw bbox offsets/log-sizes at one grid cell.
    fn plant_peak(
        outputs: &mut std::collections::HashMap<String, Vec<f32>>,
        stride: u32,
        row: u32,
        col: u32,
        cls: f32,
        obj: f32,
        bbox: [f32; 4],
    ) {
        let grid = FACE_INPUT / stride;
        let idx = (row * grid + col) as usize;
        outputs.get_mut(&format!("cls_{stride}")).unwrap()[idx] = cls;
        outputs.get_mut(&format!("obj_{stride}")).unwrap()[idx] = obj;
        let head = outputs.get_mut(&format!("bbox_{stride}")).unwrap();
        head[idx * 4..idx * 4 + 4].copy_from_slice(&bbox);
    }

    #[test]
    fn face_grid_peak_decodes_to_the_expected_box() {
        // A 640x640 source maps 1:1 from model space. A peak
        // at stride-8 cell (row 10, col 20) with zero offsets
        // and zero log-sizes: center (160, 80), size 8x8.
        let mut outputs = silent_face_outputs();
        plant_peak(&mut outputs, 8, 10, 20, 0.81, 1.0, [0.0, 0.0, 0.0, 0.0]);
        let found = decode_face_candidates(&outputs, 640, 640, FACE_SCORE_THRESHOLD).unwrap();
        assert_eq!(found.len(), 1);
        let f = found[0];
        assert_eq!((f.x, f.y, f.w, f.h), (156, 76, 8, 8));
        // The score is the geometric mean: sqrt(0.81 * 1.0).
        assert!((f.score - 0.9).abs() < 1e-6, "score {}", f.score);
    }

    #[test]
    fn face_grid_offsets_and_log_sizes_decode() {
        // Offsets shift the center within the cell and sizes
        // are exponentiated: center ((20+0.5)*8, (10+0.25)*8)
        // = (164, 82), size (e^ln2 * 8, e^0 * 8) = (16, 8).
        let mut outputs = silent_face_outputs();
        plant_peak(
            &mut outputs,
            8,
            10,
            20,
            1.0,
            1.0,
            [0.5, 0.25, 2f32.ln(), 0.0],
        );
        let found = decode_face_candidates(&outputs, 640, 640, FACE_SCORE_THRESHOLD).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(
            (found[0].x, found[0].y, found[0].w, found[0].h),
            (156, 78, 16, 8)
        );
    }

    #[test]
    fn face_grid_drops_peaks_below_the_threshold() {
        // sqrt(0.25 * 1.0) = 0.5 < 0.6: no candidate.
        let mut outputs = silent_face_outputs();
        plant_peak(&mut outputs, 8, 0, 0, 0.25, 1.0, [0.0; 4]);
        let found = decode_face_candidates(&outputs, 640, 640, FACE_SCORE_THRESHOLD).unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn face_grid_scores_are_clamped_before_the_geometric_mean() {
        // Out-of-range head values clamp to 0..1 first:
        // sqrt(1.0 * 0.64) = 0.8, not sqrt(4.0 * 0.64) = 1.6.
        let mut outputs = silent_face_outputs();
        plant_peak(&mut outputs, 8, 3, 3, 4.0, 0.64, [0.0; 4]);
        let found = decode_face_candidates(&outputs, 640, 640, FACE_SCORE_THRESHOLD).unwrap();
        assert_eq!(found.len(), 1);
        assert!(
            (found[0].score - 0.8).abs() < 1e-6,
            "score {}",
            found[0].score
        );
        // The threshold is the caller's: at 0.85 the same
        // peak is dropped.
        let found = decode_face_candidates(&outputs, 640, 640, 0.85).unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn face_grid_combines_all_three_strides_in_stride_order() {
        let mut outputs = silent_face_outputs();
        plant_peak(&mut outputs, 32, 5, 5, 1.0, 1.0, [0.0; 4]);
        plant_peak(&mut outputs, 8, 10, 20, 0.81, 1.0, [0.0; 4]);
        plant_peak(&mut outputs, 16, 2, 3, 0.64, 1.0, [0.0; 4]);
        let found = decode_face_candidates(&outputs, 640, 640, FACE_SCORE_THRESHOLD).unwrap();
        assert_eq!(found.len(), 3);
        // Strides decode in 8, 16, 32 order regardless of
        // planting order; unit sizes are one stride each.
        assert_eq!(
            (found[0].x, found[0].y, found[0].w, found[0].h),
            (156, 76, 8, 8)
        );
        assert_eq!(
            (found[1].x, found[1].y, found[1].w, found[1].h),
            (40, 24, 16, 16)
        );
        assert_eq!(
            (found[2].x, found[2].y, found[2].w, found[2].h),
            (144, 144, 32, 32)
        );
    }

    #[test]
    fn face_grid_maps_back_with_independent_axis_scales_and_clamps() {
        // A 1280x640 source: model x coordinates double, y
        // coordinates do not. A peak in the corner cell hangs
        // off the top-left edge and clamps to it, keeping its
        // full scaled size.
        let mut outputs = silent_face_outputs();
        plant_peak(&mut outputs, 8, 0, 0, 1.0, 1.0, [0.0; 4]);
        let found = decode_face_candidates(&outputs, 1280, 640, FACE_SCORE_THRESHOLD).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(
            (found[0].x, found[0].y, found[0].w, found[0].h),
            (0, 0, 16, 8)
        );
    }

    #[test]
    fn face_grid_missing_head_is_a_bad_output() {
        let mut outputs = silent_face_outputs();
        outputs.remove("bbox_16");
        let err = decode_face_candidates(&outputs, 640, 640, FACE_SCORE_THRESHOLD).unwrap_err();
        assert!(matches!(err, AiError::BadOutput(_)), "{err}");
        assert!(err.to_string().contains("bbox_16"), "{err}");
    }

    fn face(x: u32, y: u32, w: u32, h: u32, score: f32) -> FaceBox {
        FaceBox { x, y, w, h, score }
    }

    #[test]
    fn iou_measures_overlap_containment_and_disjoint_boxes() {
        let a = face(0, 0, 10, 10, 0.9);
        // A box overlaps itself completely.
        assert_eq!(iou(&a, &a), 1.0);
        // Disjoint boxes do not overlap at all.
        assert_eq!(iou(&a, &face(20, 20, 10, 10, 0.5)), 0.0);
        // Edge-touching boxes share a border, not an area.
        assert_eq!(iou(&a, &face(10, 0, 10, 10, 0.5)), 0.0);
        // Containment: the intersection is the inner box,
        // the union is the outer one: 16 / 100.
        assert!((iou(&a, &face(2, 2, 4, 4, 0.5)) - 0.16).abs() < 1e-6);
        // Half-shifted: 50 shared of a 150 union = 1/3.
        assert!((iou(&a, &face(5, 0, 10, 10, 0.5)) - 1.0 / 3.0).abs() < 1e-6);
        // Two zero-area boxes have no union: defined as 0.
        assert_eq!(iou(&face(3, 3, 0, 0, 0.5), &face(3, 3, 0, 0, 0.5)), 0.0);
    }

    #[test]
    fn nms_keeps_the_best_box_of_an_overlapping_cluster() {
        // IoU 80/120 = 0.67 > 0.3: the weaker box is dropped
        // even though it arrives first.
        let kept = nms(vec![face(2, 0, 10, 10, 0.8), face(0, 0, 10, 10, 0.9)]);
        assert_eq!(kept, vec![face(0, 0, 10, 10, 0.9)]);
    }

    #[test]
    fn nms_keeps_disjoint_boxes_best_first() {
        let kept = nms(vec![
            face(100, 100, 10, 10, 0.7),
            face(0, 0, 10, 10, 0.95),
            face(50, 0, 8, 8, 0.8),
        ]);
        assert_eq!(
            kept,
            vec![
                face(0, 0, 10, 10, 0.95),
                face(50, 0, 8, 8, 0.8),
                face(100, 100, 10, 10, 0.7),
            ]
        );
    }

    #[test]
    fn nms_keeps_a_box_at_exactly_the_iou_threshold() {
        // 60 shared of a 200 union is exactly 0.3, and the
        // rule is "at most the threshold survives".
        let kept = nms(vec![face(0, 0, 13, 10, 0.9), face(7, 0, 13, 10, 0.8)]);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn nms_of_nothing_is_nothing() {
        assert!(nms(Vec::new()).is_empty());
    }

    #[test]
    fn postprocess_maps_min_and_max_to_black_and_white() {
        // A 320x320 mask field: -1 at the origin, 3 at its
        // neighbour, 0 elsewhere. The range maps onto
        // 0..255, so the interior value 0 lands at
        // (0 - -1) / 4 * 255 = 63.75, rounding to 64.
        let mut data = vec![0f32; 320 * 320];
        data[0] = -1.0;
        data[1] = 3.0;
        let mask = postprocess(&data, 320, 320).unwrap();
        assert_eq!((mask.width(), mask.height()), (320, 320));
        assert_eq!(mask.get_pixel(0, 0)[0], 0);
        assert_eq!(mask.get_pixel(1, 0)[0], 255);
        assert_eq!(mask.get_pixel(2, 0)[0], 64);
    }

    #[test]
    fn postprocess_flat_mask_normalizes_to_black_not_nan_or_gray() {
        // A constant field has no range: the epsilon floor
        // makes every (v - min) / range exactly 0, so the
        // whole mask is black.
        let data = vec![2.5f32; 320 * 320];
        let mask = postprocess(&data, 320, 320).unwrap();
        assert!(mask.pixels().all(|p| p[0] == 0));
    }

    #[test]
    fn postprocess_resizes_the_mask_back_to_the_source_size() {
        // Top half at the range minimum, bottom half at the
        // maximum: after the resize to 160x160 the extremes
        // still read as pure black and white.
        let mut data = vec![0f32; 320 * 320];
        for v in &mut data[320 * 160..] {
            *v = 1.0;
        }
        let mask = postprocess(&data, 160, 160).unwrap();
        assert_eq!((mask.width(), mask.height()), (160, 160));
        assert_eq!(mask.get_pixel(0, 0)[0], 0);
        assert_eq!(mask.get_pixel(0, 159)[0], 255);
    }

    #[test]
    fn postprocess_rejects_an_output_shorter_than_the_field() {
        // An export whose first output is smaller than the
        // 320x320 field (a 160x160 output, say) is a
        // BadOutput, not an indexing panic under the session
        // lock (audit R2-28).
        let data = vec![0.5f32; 160 * 160];
        let err = postprocess(&data, 320, 320).unwrap_err();
        assert!(matches!(err, AiError::BadOutput(_)), "{err}");
        assert!(err.to_string().contains("320x320"), "{err}");
    }
}

/// One label and its raw score from a tag provider: for
/// CLIP, the cosine similarity between the image embedding
/// and the label embedding. Presentation (softmax,
/// thresholding, capping, rounding) is the caller's job
/// (issue #152).
#[derive(Debug, Clone, PartialEq)]
pub struct TagScore {
    pub label: String,
    pub score: f32,
}

/// Scores images against a fixed label set.
pub trait TagProvider: Send + Sync + std::fmt::Debug {
    /// Raw scores for every configured label, in label
    /// order: no thresholding, capping, or rounding, so
    /// callers can tune presentation and debug borderline
    /// labels from the same numbers.
    fn scores(&self, img: &DynamicImage) -> Result<Vec<TagScore>, AiError>;
}

/// Zero-shot tagging backed by a CLIP image/text ONNX model
/// pair. The label set is operator configuration: labels
/// are embedded once at load, so a request runs only the
/// image encoder and a cosine comparison. The exports are
/// expected to include the projection head (outputs named
/// `image_embeds`/`text_embeds` or `pooled_output`), which
/// is validated at load: mismatched pairs fail startup.
pub struct OnnxClipProvider {
    vision: Mutex<Session>,
    vision_input: String,
    labels: Vec<String>,
    /// Label embeddings, flattened `labels.len()` x `dim`,
    /// each row L2-normalized.
    label_embeddings: Vec<f32>,
    dim: usize,
}

impl OnnxClipProvider {
    /// Load the model pair, the tokenizer file, and the
    /// label file, embedding every label before returning.
    pub fn load(
        dylib: Option<&str>,
        image_model: &str,
        text_model: &str,
        tokenizer_path: &str,
        labels_path: &str,
    ) -> Result<Self, AiError> {
        let labels = read_labels(labels_path)?;
        let tokenizer =
            crate::tokenizer::ClipTokenizer::load(tokenizer_path).map_err(AiError::Load)?;
        let (text, _) = build_session(dylib, text_model)?;
        let text_inputs: Vec<String> = text.inputs().iter().map(|i| i.name().to_string()).collect();
        let mut text = text;
        let mut label_embeddings: Vec<f32> = Vec::new();
        let mut dim = 0usize;
        for label in &labels {
            let (ids, mask) = tokenizer.encode(label).map_err(AiError::Load)?;
            // Embedding the labels is part of loading: a text
            // model that cannot run here is a load failure,
            // not a request-time inference failure.
            let emb = run_text(&mut text, &text_inputs, &ids, &mask).map_err(|e| {
                AiError::Load(format!("the text model could not embed the labels: {e}"))
            })?;
            if dim == 0 {
                dim = emb.len();
            } else if emb.len() != dim {
                return Err(AiError::Load(
                    "the text model returned inconsistent embedding sizes".to_string(),
                ));
            }
            label_embeddings.extend_from_slice(&normalize(&emb));
        }
        let (vision, vision_input) = build_session(dylib, image_model)?;
        let provider = Self {
            vision: Mutex::new(vision),
            vision_input,
            labels,
            label_embeddings,
            dim,
        };
        // Fail fast on a mismatched pair: embed a black
        // image once and check the projection size agrees
        // with the text side.
        let probe = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
        let emb = provider
            .embed_image(&probe)
            .map_err(|e| AiError::Load(format!("the image model failed its load probe: {e}")))?;
        if emb.len() != provider.dim {
            return Err(AiError::Load(format!(
                "the image model produced a {}-dimensional embedding but the text model produced {}: the pair does not share a projection space",
                emb.len(),
                provider.dim
            )));
        }
        Ok(provider)
    }

    /// The number of configured labels.
    pub fn label_count(&self) -> usize {
        self.labels.len()
    }

    fn embed_image(&self, img: &DynamicImage) -> Result<Vec<f32>, AiError> {
        let input = preprocess_clip(img);
        let tensor =
            ort::value::Tensor::from_array(([1, 3, 224, 224], input.into_boxed_slice()))
                .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
        let mut session = self.vision.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![self.vision_input.as_str() => tensor])
            .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
        let outputs = collect_outputs(outputs)?;
        pick_embedding(&outputs, None)
    }
}

impl std::fmt::Debug for OnnxClipProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxClipProvider")
            .field("labels", &self.labels.len())
            .field("dim", &self.dim)
            .finish_non_exhaustive()
    }
}

impl TagProvider for OnnxClipProvider {
    fn scores(&self, img: &DynamicImage) -> Result<Vec<TagScore>, AiError> {
        let emb = normalize(&self.embed_image(img)?);
        Ok(self
            .labels
            .iter()
            .enumerate()
            .map(|(i, label)| TagScore {
                label: label.clone(),
                score: emb
                    .iter()
                    .zip(&self.label_embeddings[i * self.dim..(i + 1) * self.dim])
                    .map(|(a, b)| a * b)
                    .sum(),
            })
            .collect())
    }
}

/// Read the label file: one label per line, blank lines
/// and `#` comments skipped. No labels is an error.
/// Public for the declared-detector builder (issue #166),
/// which resolves a declaration's labels path to a
/// vocabulary before loading, so declared labels win over
/// model metadata exactly like inline ones.
pub fn read_labels(path: &str) -> Result<Vec<String>, AiError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| AiError::Load(format!("cannot read the label file {path}: {e}")))?;
    let labels: Vec<String> = raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    if labels.is_empty() {
        return Err(AiError::Load(format!(
            "the label file {path} contains no labels"
        )));
    }
    Ok(labels)
}

/// Run the text model for one tokenized label and pick its
/// embedding. Inputs are fed by name, so exports with or
/// without an attention mask (or token type ids) all work.
fn run_text(
    session: &mut Session,
    input_names: &[String],
    ids: &[i64],
    mask: &[i64],
) -> Result<Vec<f32>, AiError> {
    let n = ids.len();
    let ids_tensor = ort::value::Tensor::from_array(([1, n], ids.to_vec().into_boxed_slice()))
        .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
    let mask_tensor = ort::value::Tensor::from_array(([1, n], mask.to_vec().into_boxed_slice()))
        .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
    let zeros_tensor =
        ort::value::Tensor::from_array(([1, n], vec![0i64; n].into_boxed_slice()))
            .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
    let ids_name = input_names
        .iter()
        .find(|name| name.contains("input_ids"))
        .or_else(|| input_names.first())
        .ok_or_else(|| AiError::Load("the text model has no inputs".to_string()))?;
    let mask_name = input_names.iter().find(|name| name.contains("attention"));
    let types_name = input_names.iter().find(|name| name.contains("token_type"));
    let outputs = match (mask_name, types_name) {
        (Some(m), Some(t)) => session.run(ort::inputs![
            ids_name.as_str() => ids_tensor,
            m.as_str() => mask_tensor,
            t.as_str() => zeros_tensor,
        ]),
        (Some(m), None) => session.run(ort::inputs![
            ids_name.as_str() => ids_tensor,
            m.as_str() => mask_tensor,
        ]),
        (None, Some(t)) => session.run(ort::inputs![
            ids_name.as_str() => ids_tensor,
            t.as_str() => zeros_tensor,
        ]),
        (None, None) => session.run(ort::inputs![ids_name.as_str() => ids_tensor]),
    }
    .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
    let outputs = collect_outputs(outputs)?;
    // The text embedding is the pooled row at the EOT
    // position: the last non-padding token.
    let eot_row = mask.iter().filter(|&&m| m == 1).count().saturating_sub(1);
    pick_embedding(&outputs, Some(eot_row))
}

/// One model output: name, dims, data.
type ModelOutput = (String, Vec<i64>, Vec<f32>);

/// Collect a session's outputs as (name, dims, data).
fn collect_outputs(outputs: ort::session::SessionOutputs<'_>) -> Result<Vec<ModelOutput>, AiError> {
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

/// Choose the embedding from a model's outputs:
/// `image_embeds` / `text_embeds` first, then
/// `pooled_output`, then the first output (row 0, or the
/// EOT row for text hidden states).
fn pick_embedding(outputs: &[ModelOutput], eot_row: Option<usize>) -> Result<Vec<f32>, AiError> {
    for wanted in ["image_embeds", "text_embeds", "pooled_output"] {
        if let Some((_, dims, data)) = outputs.iter().find(|(name, _, _)| name == wanted) {
            return Ok(first_row(dims, data));
        }
    }
    let (_, dims, data) = outputs
        .first()
        .ok_or_else(|| AiError::BadOutput("the model produced no outputs".to_string()))?;
    if dims.len() == 3 {
        let dim = dims[2] as usize;
        let row = eot_row.unwrap_or(0);
        let start = row * dim;
        if start + dim > data.len() {
            return Err(AiError::BadOutput(
                "the model output is smaller than its declared shape".to_string(),
            ));
        }
        return Ok(data[start..start + dim].to_vec());
    }
    Ok(first_row(dims, data))
}

fn first_row(dims: &[i64], data: &[f32]) -> Vec<f32> {
    if dims.len() >= 2 {
        let dim = dims[dims.len() - 1] as usize;
        if dim > 0 && data.len() >= dim {
            return data[..dim].to_vec();
        }
    }
    data.to_vec()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// CLIP image preprocessing, matching the export's
/// declared pipeline: resize the short side to 224
/// (bicubic), center crop 224x224, normalize with the
/// CLIP mean/std. Output is planar CHW.
#[allow(clippy::excessive_precision)] // the model's published constants
fn preprocess_clip(img: &DynamicImage) -> Vec<f32> {
    let (w, h) = (img.width().max(1), img.height().max(1));
    let scale = 224.0 / w.min(h) as f32;
    let nw = ((w as f32 * scale).round() as u32).max(224);
    let nh = ((h as f32 * scale).round() as u32).max(224);
    let resized = img.resize_exact(nw, nh, FilterType::CatmullRom);
    let cropped = resized
        .crop_imm((nw - 224) / 2, (nh - 224) / 2, 224, 224)
        .to_rgb8();
    const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
    let plane = 224 * 224;
    let mut out = vec![0f32; 3 * plane];
    for (i, px) in cropped.pixels().enumerate() {
        for (c, (mean, std)) in MEAN.iter().zip(STD.iter()).enumerate() {
            out[c * plane + i] = ((px[c] as f32 / 255.0) - mean) / std;
        }
    }
    out
}

#[cfg(test)]
mod clip_tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)] // the model's published constants
    fn preprocess_clip_normalizes_a_solid_image() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            100,
            50,
            image::Rgb([255, 0, 0]),
        ));
        let input = preprocess_clip(&img);
        assert_eq!(input.len(), 3 * 224 * 224);
        let r = (1.0f32 - 0.48145466) / 0.26862954;
        let g = (0.0f32 - 0.4578275) / 0.26130258;
        assert!((input[0] - r).abs() < 1e-4, "red plane {}", input[0]);
        assert!((input[224 * 224] - g).abs() < 1e-4, "green plane");
        assert!(input.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn read_labels_skips_comments_and_blanks() {
        let path = std::env::temp_dir().join(format!("crabcrop-labels-{}.txt", std::process::id()));
        std::fs::write(&path, "# heading\n\nocean\n  sky  \n#kite\n").unwrap();
        let got = read_labels(path.to_str().unwrap()).unwrap();
        assert_eq!(got, ["ocean", "sky"]);
        std::fs::remove_file(&path).ok();
        let empty =
            std::env::temp_dir().join(format!("crabcrop-labels-empty-{}.txt", std::process::id()));
        std::fs::write(&empty, "# only comments\n\n").unwrap();
        assert!(read_labels(empty.to_str().unwrap()).is_err());
        std::fs::remove_file(&empty).ok();
    }

    #[test]
    fn pick_embedding_prefers_the_named_projection_outputs() {
        let hidden: ModelOutput = (
            "last_hidden_state".to_string(),
            vec![1, 2, 3],
            (0..6).map(|v| v as f32).collect(),
        );
        let pooled: ModelOutput = ("pooled_output".to_string(), vec![1, 3], vec![7.0, 8.0, 9.0]);
        let image: ModelOutput = ("image_embeds".to_string(), vec![1, 3], vec![1.0, 2.0, 3.0]);
        // image_embeds wins over pooled_output and the rest.
        let outputs = vec![hidden.clone(), pooled.clone(), image];
        assert_eq!(pick_embedding(&outputs, None).unwrap(), [1.0, 2.0, 3.0]);
        // text_embeds is the projection pick on the text side.
        let text: ModelOutput = ("text_embeds".to_string(), vec![1, 3], vec![4.0, 5.0, 6.0]);
        let outputs = vec![hidden.clone(), text];
        assert_eq!(pick_embedding(&outputs, Some(1)).unwrap(), [4.0, 5.0, 6.0]);
        // With no named projection, pooled_output is next.
        let outputs = vec![hidden, pooled];
        assert_eq!(pick_embedding(&outputs, None).unwrap(), [7.0, 8.0, 9.0]);
    }

    #[test]
    fn pick_embedding_reads_the_requested_row_of_a_hidden_state() {
        // A [1, 3, 2] hidden state with no projection or
        // pooled output: the fallback reads one row of the
        // first output.
        let outputs: Vec<ModelOutput> = vec![(
            "last_hidden_state".to_string(),
            vec![1, 3, 2],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )];
        // The text path asks for the EOT row explicitly.
        assert_eq!(pick_embedding(&outputs, Some(2)).unwrap(), [4.0, 5.0]);
        // The image path takes row 0.
        assert_eq!(pick_embedding(&outputs, None).unwrap(), [0.0, 1.0]);
        // A rank-2 first output falls back to its first row.
        let outputs: Vec<ModelOutput> = vec![(
            "logits".to_string(),
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )];
        assert_eq!(pick_embedding(&outputs, None).unwrap(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn pick_embedding_rejects_missing_and_undersized_outputs() {
        // No outputs at all is a BadOutput.
        let err = pick_embedding(&[], None).unwrap_err();
        assert!(matches!(err, AiError::BadOutput(_)), "{err}");
        assert!(err.to_string().contains("no outputs"), "{err}");
        // An EOT row past the end of the declared shape is
        // a BadOutput, not a short or padded embedding.
        let outputs: Vec<ModelOutput> = vec![(
            "last_hidden_state".to_string(),
            vec![1, 2, 2],
            vec![0.0, 1.0, 2.0, 3.0],
        )];
        let err = pick_embedding(&outputs, Some(5)).unwrap_err();
        assert!(matches!(err, AiError::BadOutput(_)), "{err}");
        assert!(
            err.to_string().contains("smaller than its declared shape"),
            "{err}"
        );
    }

    #[test]
    fn first_row_takes_the_last_dimension_and_falls_back_to_all_data() {
        // A [2, 3] output: the first row is the first 3.
        assert_eq!(
            first_row(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [1.0, 2.0, 3.0]
        );
        // A rank-1 output has no row structure: everything.
        assert_eq!(first_row(&[3], &[1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
        // A declared row wider than the data: everything,
        // rather than a panic or a truncated row.
        assert_eq!(first_row(&[1, 10], &[1.0, 2.0]), [1.0, 2.0]);
    }

    #[test]
    fn normalize_scales_to_unit_length_and_keeps_the_zero_vector() {
        let out = normalize(&[3.0, 4.0]);
        assert!(
            (out[0] - 0.6).abs() < 1e-6 && (out[1] - 0.8).abs() < 1e-6,
            "{out:?}"
        );
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        // The zero vector has no direction: it is returned
        // as-is instead of dividing by zero.
        assert_eq!(normalize(&[0.0, 0.0]), [0.0, 0.0]);
    }

    /// Live check with a real CLIP export: tags a fixture
    /// photo and asserts the annotation is sane. Ignored:
    /// needs the runtime, both models, the tokenizer, a
    /// label file, and a photo:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_CLIP_IMAGE_MODEL=...
    /// CRABCROP_TEST_CLIP_TEXT_MODEL=... CRABCROP_TEST_CLIP_TOKENIZER=...
    /// CRABCROP_TEST_CLIP_LABELS=... CRABCROP_TEST_IMAGE=...
    #[test]
    #[ignore = "needs the ONNX runtime and CLIP model files (env-configured)"]
    fn clip_live_tags_a_photo() {
        let var = |name: &str| std::env::var(name).unwrap();
        let provider = OnnxClipProvider::load(
            Some(&var("CRABCROP_TEST_ORT_DYLIB")),
            &var("CRABCROP_TEST_CLIP_IMAGE_MODEL"),
            &var("CRABCROP_TEST_CLIP_TEXT_MODEL"),
            &var("CRABCROP_TEST_CLIP_TOKENIZER"),
            &var("CRABCROP_TEST_CLIP_LABELS"),
        )
        .unwrap();
        let img = image::open(var("CRABCROP_TEST_IMAGE")).unwrap();
        let scores = provider.scores(&img).unwrap();
        println!("scores: {scores:?}");
        assert_eq!(scores.len(), provider.label_count());
        assert!(scores.iter().all(|s| s.score.is_finite()));
    }

    /// Write bytes to a fresh temp file and return its
    /// path: the load() failure tests need real files
    /// (a label list, a garbage tokenizer) without
    /// committing fixtures.
    fn temp_file(tag: &str, contents: &[u8]) -> String {
        let path =
            std::env::temp_dir().join(format!("crabcrop-ai-load-{tag}-{}.bin", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// A path inside the temp dir that is never created.
    fn missing_path(tag: &str) -> String {
        std::env::temp_dir()
            .join(format!(
                "crabcrop-ai-load-missing-{tag}-{}.onnx",
                std::process::id()
            ))
            .to_string_lossy()
            .into_owned()
    }

    /// Assert a load() result failed as AiError::Load and
    /// return the message for wording checks.
    fn expect_load<T>(result: Result<T, AiError>) -> String {
        match result {
            Err(AiError::Load(message)) => message,
            Err(other) => panic!("expected a load error, got {other}"),
            Ok(_) => panic!("expected a load error, but the load succeeded"),
        }
    }

    /// Load-failure coverage (#179, item 2). The label and
    /// tokenizer files are read before any session is
    /// built, so these run in the plain gate with no
    /// runtime at all: a missing label file is Load.
    #[test]
    fn clip_load_missing_labels_file_is_a_load_error() {
        let missing = missing_path("clip-labels");
        let message = expect_load(OnnxClipProvider::load(
            None, &missing, &missing, &missing, &missing,
        ));
        assert!(message.contains("cannot read the label file"), "{message}");
    }

    /// A label file with only comments and blank lines has
    /// no labels: Load, still before the runtime is touched.
    #[test]
    fn clip_load_empty_labels_file_is_a_load_error() {
        let labels = temp_file("clip-labels-empty", b"# only a comment\n\n");
        let missing = missing_path("clip-empty");
        let message = expect_load(OnnxClipProvider::load(
            None, &missing, &missing, &missing, &labels,
        ));
        assert!(message.contains("contains no labels"), "{message}");
    }

    /// With valid labels, a missing tokenizer file is the
    /// next pre-runtime failure: Load.
    #[test]
    fn clip_load_missing_tokenizer_file_is_a_load_error() {
        let labels = temp_file("clip-labels-ok", b"cat\ndog\n");
        let missing = missing_path("clip-tokenizer");
        let message = expect_load(OnnxClipProvider::load(
            None, &missing, &missing, &missing, &labels,
        ));
        assert!(
            message.contains("cannot read the tokenizer file"),
            "{message}"
        );
    }

    /// A tokenizer file that is not BPE JSON fails the
    /// same pre-runtime way.
    #[test]
    fn clip_load_garbage_tokenizer_file_is_a_load_error() {
        let labels = temp_file("clip-labels-ok2", b"cat\ndog\n");
        let tokenizer = temp_file("clip-tokenizer-garbage", b"{ this is not json");
        let missing = missing_path("clip-garbage-tok");
        let message = expect_load(OnnxClipProvider::load(
            None, &missing, &missing, &tokenizer, &labels,
        ));
        assert!(message.contains("tokenizer"), "{message}");
    }

    /// With the files in place, a missing text model fails
    /// in the session builder: Load.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_CLIP_IMAGE_MODEL=...
    /// CRABCROP_TEST_CLIP_TOKENIZER=... CRABCROP_TEST_CLIP_LABELS=...
    /// cargo test -- --ignored clip_load_missing_text_model
    #[test]
    #[ignore = "needs the ONNX runtime and CLIP model files (env-configured)"]
    fn clip_load_missing_text_model_is_a_load_error() {
        let var = |name: &str| std::env::var(name).unwrap();
        let message = expect_load(OnnxClipProvider::load(
            Some(&var("CRABCROP_TEST_ORT_DYLIB")),
            &var("CRABCROP_TEST_CLIP_IMAGE_MODEL"),
            &missing_path("clip-text"),
            &var("CRABCROP_TEST_CLIP_TOKENIZER"),
            &var("CRABCROP_TEST_CLIP_LABELS"),
        ));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// A garbage text model file fails the same way.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_CLIP_IMAGE_MODEL=...
    /// CRABCROP_TEST_CLIP_TOKENIZER=... CRABCROP_TEST_CLIP_LABELS=...
    /// cargo test -- --ignored clip_load_garbage_text_model
    #[test]
    #[ignore = "needs the ONNX runtime and CLIP model files (env-configured)"]
    fn clip_load_garbage_text_model_is_a_load_error() {
        let var = |name: &str| std::env::var(name).unwrap();
        let garbage = temp_file("clip-text-garbage", b"this is not an onnx model at all");
        let message = expect_load(OnnxClipProvider::load(
            Some(&var("CRABCROP_TEST_ORT_DYLIB")),
            &var("CRABCROP_TEST_CLIP_IMAGE_MODEL"),
            &garbage,
            &var("CRABCROP_TEST_CLIP_TOKENIZER"),
            &var("CRABCROP_TEST_CLIP_LABELS"),
        ));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// A missing image model is reached only after the
    /// text side has embedded every label; still Load.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_CLIP_TEXT_MODEL=...
    /// CRABCROP_TEST_CLIP_TOKENIZER=... CRABCROP_TEST_CLIP_LABELS=...
    /// cargo test -- --ignored clip_load_missing_image_model
    #[test]
    #[ignore = "needs the ONNX runtime and CLIP model files (env-configured)"]
    fn clip_load_missing_image_model_is_a_load_error() {
        let var = |name: &str| std::env::var(name).unwrap();
        let message = expect_load(OnnxClipProvider::load(
            Some(&var("CRABCROP_TEST_ORT_DYLIB")),
            &missing_path("clip-image"),
            &var("CRABCROP_TEST_CLIP_TEXT_MODEL"),
            &var("CRABCROP_TEST_CLIP_TOKENIZER"),
            &var("CRABCROP_TEST_CLIP_LABELS"),
        ));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// A swapped pair (the vision model as the text model)
    /// cannot embed the labels: embedding is part of
    /// loading, so the failure is Load, not a request-time
    /// Inference.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_CLIP_IMAGE_MODEL=...
    /// CRABCROP_TEST_CLIP_TEXT_MODEL=... CRABCROP_TEST_CLIP_TOKENIZER=...
    /// CRABCROP_TEST_CLIP_LABELS=...
    /// cargo test -- --ignored clip_load_swapped_model_pair
    #[test]
    #[ignore = "needs the ONNX runtime and CLIP model files (env-configured)"]
    fn clip_load_swapped_model_pair_is_a_load_error() {
        let var = |name: &str| std::env::var(name).unwrap();
        expect_load(OnnxClipProvider::load(
            Some(&var("CRABCROP_TEST_ORT_DYLIB")),
            &var("CRABCROP_TEST_CLIP_TEXT_MODEL"),
            &var("CRABCROP_TEST_CLIP_IMAGE_MODEL"),
            &var("CRABCROP_TEST_CLIP_TOKENIZER"),
            &var("CRABCROP_TEST_CLIP_LABELS"),
        ));
    }
}

/// The outcome of one moderation run: the model that
/// produced it, its per-category verdict scores, and any
/// zero-shot review signals. A given provider fills
/// exactly one of the two lists: a classifier reports
/// verdicts, the CLIP fallback reports signals.
/// Presentation (rounding, ordering) is the caller's job,
/// like tag composition (issue #152).
#[derive(Debug, Clone, PartialEq)]
pub struct ModerationScores {
    /// The configured model identity reported in meta:
    /// the verdict model path, or the CLIP image model
    /// path when the fallback produced the signals.
    pub model: String,
    /// Verdict scores: per-category probabilities from
    /// the configured classifier, in the model's output
    /// order. Empty when the CLIP fallback ran.
    pub scores: Vec<TagScore>,
    /// Zero-shot review signals: cosine scores against
    /// operator prompt labels. Signals are leads for a
    /// human or a downstream review queue, never
    /// verdicts; the meta response marks them
    /// review-only. Empty when a classifier produced
    /// the verdicts.
    pub signals: Vec<TagScore>,
}

/// Scores images for content moderation (issue #171).
pub trait ModerationProvider: Send + Sync + std::fmt::Debug {
    fn moderate(&self, img: &DynamicImage) -> Result<ModerationScores, AiError>;
}

/// Content moderation backed by a ViT classifier ONNX
/// export of the Falconsai nsfw_image_detection shape
/// (one [1, 3, 224, 224] image input, one logits row
/// out). Category names are operator configuration;
/// without them, a two-output model gets the Falconsai
/// pair (`normal`, `nsfw`), and any other output count
/// is a load error: unnamed categories would be
/// meaningless in the meta response.
pub struct OnnxModerationProvider {
    session: Mutex<Session>,
    input_name: String,
    labels: Vec<String>,
    model: String,
}

impl OnnxModerationProvider {
    /// Load the model and probe it on a blank image: the
    /// probe fixes the output count the configured
    /// labels are resolved against, so a mismatch fails
    /// startup instead of the first moderated request.
    pub fn load(
        dylib: Option<&str>,
        model_path: &str,
        labels: Option<Vec<String>>,
    ) -> Result<Self, AiError> {
        let (session, input_name) = build_session(dylib, model_path)?;
        let mut provider = Self {
            session: Mutex::new(session),
            input_name,
            labels: Vec::new(),
            model: model_path.to_string(),
        };
        let probe = provider
            .logits(&DynamicImage::new_rgb8(8, 8))
            .map_err(|e| {
                AiError::Load(format!(
                    "the moderation model {model_path} failed its load probe: {e}"
                ))
            })?;
        provider.labels = resolve_moderation_labels(labels, probe.len(), model_path)?;
        Ok(provider)
    }

    /// The category names, in the model's output order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    fn logits(&self, img: &DynamicImage) -> Result<Vec<f32>, AiError> {
        let input = preprocess_vit(img);
        let tensor =
            ort::value::Tensor::from_array(([1, 3, 224, 224], input.into_boxed_slice()))
                .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
        let mut session = self.session.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
        let outputs = collect_outputs(outputs)?;
        pick_logits(&outputs)
    }
}

impl std::fmt::Debug for OnnxModerationProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxModerationProvider")
            .field("labels", &self.labels)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl ModerationProvider for OnnxModerationProvider {
    fn moderate(&self, img: &DynamicImage) -> Result<ModerationScores, AiError> {
        let logits = self.logits(img)?;
        if logits.len() != self.labels.len() {
            return Err(AiError::BadOutput(format!(
                "the moderation model produced {} outputs for {} configured labels",
                logits.len(),
                self.labels.len()
            )));
        }
        let probs = softmax(&logits);
        Ok(ModerationScores {
            model: self.model.clone(),
            scores: self
                .labels
                .iter()
                .cloned()
                .zip(probs)
                .map(|(label, score)| TagScore { label, score })
                .collect(),
            signals: Vec::new(),
        })
    }
}

/// Moderation signals from a CLIP tag provider: the
/// zero-shot fallback for deployments without a verdict
/// classifier. The wrapped provider scores operator
/// prompt labels by cosine; this provider reports them
/// as review signals and nothing else. Wrapping any
/// `TagProvider` keeps the fallback on the tagger's
/// loading path.
pub struct ClipModerationProvider {
    tagger: std::sync::Arc<dyn TagProvider>,
    model: String,
}

impl ClipModerationProvider {
    /// Wrap a tag provider; `model` is the identity the
    /// meta response reports (the CLIP image model path).
    pub fn new(tagger: std::sync::Arc<dyn TagProvider>, model: String) -> Self {
        Self { tagger, model }
    }
}

impl std::fmt::Debug for ClipModerationProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClipModerationProvider")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl ModerationProvider for ClipModerationProvider {
    fn moderate(&self, img: &DynamicImage) -> Result<ModerationScores, AiError> {
        Ok(ModerationScores {
            model: self.model.clone(),
            scores: Vec::new(),
            signals: self.tagger.scores(img)?,
        })
    }
}

/// Resolve the category names for a moderation model
/// against its probed output count. Configured labels
/// win when their count matches; without labels, a
/// two-output model gets the Falconsai pair (`normal`,
/// `nsfw`) and any other count is a load error naming
/// the fix.
fn resolve_moderation_labels(
    configured: Option<Vec<String>>,
    outputs: usize,
    model_path: &str,
) -> Result<Vec<String>, AiError> {
    match configured {
        Some(labels) => {
            if labels.len() == outputs {
                Ok(labels)
            } else {
                Err(AiError::Load(format!(
                    "the configured moderation labels name {} categories but the model {model_path} produces {outputs}: name one label per output, in output order",
                    labels.len()
                )))
            }
        }
        None if outputs == 2 => Ok(vec!["normal".to_string(), "nsfw".to_string()]),
        None => Err(AiError::Load(format!(
            "the moderation model {model_path} produces {outputs} categories and no labels are configured: set CRABCROP_MODERATION_LABELS to name them"
        ))),
    }
}

/// Choose the logits row from a classifier's outputs:
/// the output named `logits` first, then the first
/// output. A multi-row output contributes its first
/// row.
fn pick_logits(outputs: &[ModelOutput]) -> Result<Vec<f32>, AiError> {
    let (_, dims, data) = outputs
        .iter()
        .find(|(name, _, _)| name == "logits")
        .or_else(|| outputs.first())
        .ok_or_else(|| AiError::BadOutput("the model produced no outputs".to_string()))?;
    if data.is_empty() {
        return Err(AiError::BadOutput(
            "the model produced an empty output".to_string(),
        ));
    }
    Ok(first_row(dims, data))
}

/// Softmax over one logits row, max-subtracted so large
/// logits cannot overflow the exponentials. An empty
/// row stays empty.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    if total == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.iter().map(|e| e / total).collect()
}

/// The standard ViT preprocessing the moderation
/// classifier was published with: resize the short side
/// to 224 (bicubic), center crop 224x224, normalize with
/// the ImageNet mean/std. Output is planar CHW. The
/// geometry mirrors `preprocess_clip`; the statistics
/// are the ones this model family declares.
#[allow(clippy::excessive_precision)] // the model's published constants
fn preprocess_vit(img: &DynamicImage) -> Vec<f32> {
    let (w, h) = (img.width().max(1), img.height().max(1));
    let scale = 224.0 / w.min(h) as f32;
    let nw = ((w as f32 * scale).round() as u32).max(224);
    let nh = ((h as f32 * scale).round() as u32).max(224);
    let resized = img.resize_exact(nw, nh, FilterType::CatmullRom);
    let cropped = resized
        .crop_imm((nw - 224) / 2, (nh - 224) / 2, 224, 224)
        .to_rgb8();
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let plane = 224 * 224;
    let mut out = vec![0f32; 3 * plane];
    for (i, px) in cropped.pixels().enumerate() {
        for (c, (mean, std)) in MEAN.iter().zip(STD.iter()).enumerate() {
            out[c * plane + i] = ((px[c] as f32 / 255.0) - mean) / std;
        }
    }
    out
}

#[cfg(test)]
mod moderation_tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)] // the model's published constants
    fn preprocess_vit_normalizes_a_solid_image() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            100,
            50,
            image::Rgb([255, 0, 0]),
        ));
        let input = preprocess_vit(&img);
        assert_eq!(input.len(), 3 * 224 * 224);
        // (1 - 0.485) / 0.229 = 2.2489 in red, (0 - 0.456)
        // / 0.224 = -2.0357 in green.
        assert!((input[0] - 2.2489).abs() < 0.001, "red {}", input[0]);
        assert!(
            (input[224 * 224] - -2.0357).abs() < 0.001,
            "green {}",
            input[224 * 224]
        );
        assert!(input.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn softmax_turns_logits_into_probabilities() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        assert_eq!(probs.len(), 3);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
        // Equal logits split evenly.
        let flat = softmax(&[0.5, 0.5]);
        assert!((flat[0] - 0.5).abs() < 1e-6);
        // Large logits cannot overflow: the max is
        // subtracted before the exponentials.
        let big = softmax(&[1000.0, 1001.0]);
        assert!(big.iter().all(|v| v.is_finite()));
        assert!((big.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // An empty row stays empty rather than dividing
        // by a zero total.
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn pick_logits_prefers_the_logits_output_and_takes_its_first_row() {
        let outputs = vec![
            (
                "hidden".to_string(),
                vec![1, 3],
                vec![9.0, 9.0, 9.0],
            ),
            (
                "logits".to_string(),
                vec![1, 2],
                vec![1.5, -0.5],
            ),
        ];
        assert_eq!(pick_logits(&outputs).unwrap(), vec![1.5, -0.5]);
        // Without a named output, the first one is used.
        let outputs = vec![("out".to_string(), vec![2], vec![0.25, 0.75])];
        assert_eq!(pick_logits(&outputs).unwrap(), vec![0.25, 0.75]);
    }

    #[test]
    fn pick_logits_rejects_missing_and_empty_outputs() {
        let err = pick_logits(&[]).unwrap_err();
        assert!(matches!(err, AiError::BadOutput(_)), "{err}");
        let outputs = vec![("logits".to_string(), vec![1, 0], Vec::new())];
        let err = pick_logits(&outputs).unwrap_err();
        assert!(matches!(err, AiError::BadOutput(_)), "{err}");
    }

    #[test]
    fn moderation_labels_default_to_the_falconsai_pair_for_two_outputs() {
        let labels = resolve_moderation_labels(None, 2, "model.onnx").unwrap();
        assert_eq!(labels, vec!["normal".to_string(), "nsfw".to_string()]);
    }

    #[test]
    fn moderation_labels_require_a_file_for_other_output_counts() {
        let err = resolve_moderation_labels(None, 5, "model.onnx").unwrap_err();
        assert!(matches!(err, AiError::Load(_)), "{err}");
        assert!(err.to_string().contains("CRABCROP_MODERATION_LABELS"), "{err}");
    }

    #[test]
    fn moderation_labels_must_match_the_output_count() {
        let named = vec!["safe".to_string(), "explicit".to_string()];
        let labels =
            resolve_moderation_labels(Some(named.clone()), 2, "model.onnx").unwrap();
        assert_eq!(labels, named);
        let err =
            resolve_moderation_labels(Some(vec!["only".to_string()]), 2, "model.onnx")
                .unwrap_err();
        assert!(matches!(err, AiError::Load(_)), "{err}");
        assert!(err.to_string().contains("name one label per output"), "{err}");
    }

    #[derive(Debug)]
    struct StubTagger(Result<Vec<TagScore>, AiError>);

    impl TagProvider for StubTagger {
        fn scores(&self, _img: &DynamicImage) -> Result<Vec<TagScore>, AiError> {
            self.0.clone()
        }
    }

    fn solid() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            16,
            16,
            image::Rgb([120, 120, 120]),
        ))
    }

    #[test]
    fn clip_moderation_reports_signals_and_no_verdicts() {
        let tagger = StubTagger(Ok(vec![TagScore {
            label: "a photo of a weapon".to_string(),
            score: 0.24,
        }]));
        let provider =
            ClipModerationProvider::new(std::sync::Arc::new(tagger), "clip.onnx".to_string());
        let report = provider.moderate(&solid()).unwrap();
        assert_eq!(report.model, "clip.onnx");
        assert!(report.scores.is_empty());
        assert_eq!(report.signals.len(), 1);
        assert_eq!(report.signals[0].label, "a photo of a weapon");
    }

    #[test]
    fn clip_moderation_propagates_tagger_failures() {
        let tagger = StubTagger(Err(AiError::Inference("boom".to_string())));
        let provider =
            ClipModerationProvider::new(std::sync::Arc::new(tagger), "clip.onnx".to_string());
        let err = provider.moderate(&solid()).unwrap_err();
        assert!(matches!(err, AiError::Inference(_)), "{err}");
    }

    /// Write bytes to a fresh temp file and return its
    /// path: the load() failure tests need real files
    /// (a garbage model) without committing fixtures.
    fn temp_file(tag: &str, contents: &[u8]) -> String {
        let path = std::env::temp_dir().join(format!(
            "crabcrop-moderation-load-{tag}-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// A path inside the temp dir that is never created.
    fn missing_path(tag: &str) -> String {
        std::env::temp_dir()
            .join(format!(
                "crabcrop-moderation-load-missing-{tag}-{}.onnx",
                std::process::id()
            ))
            .to_string_lossy()
            .into_owned()
    }

    /// Assert a load() result failed as AiError::Load and
    /// return the message for wording checks.
    fn expect_load<T>(result: Result<T, AiError>) -> String {
        match result {
            Err(AiError::Load(message)) => message,
            Err(other) => panic!("expected a load error, got {other}"),
            Ok(_) => panic!("expected a load error, but the load succeeded"),
        }
    }

    /// Real-model verification, run manually with the
    /// converted Falconsai artifact in place:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_MODERATION_MODEL=...
    /// cargo test -- --ignored moderation_with_the_real_model
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn moderation_with_the_real_model() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_MODERATION_MODEL").unwrap();
        let provider = OnnxModerationProvider::load(Some(&dylib), &model, None).unwrap();
        // No labels configured: the Falconsai pair applies.
        assert_eq!(provider.labels(), ["normal", "nsfw"]);
        let bytes = include_bytes!("../tests/fixtures/portrait.jpg");
        let img = image::load_from_memory(bytes).unwrap();
        let report = provider.moderate(&img).unwrap();
        assert_eq!(report.model, model);
        assert!(report.signals.is_empty());
        assert_eq!(report.scores.len(), 2);
        assert_eq!(report.scores[0].label, "normal");
        assert_eq!(report.scores[1].label, "nsfw");
        // The spike measured this fixture at normal 0.9994+
        // through the reference pipeline; the bound leaves
        // room for build-to-build wobble.
        assert!(report.scores[0].score > 0.99, "scores: {:?}", report.scores);
        assert!(report.scores[1].score < 0.01, "scores: {:?}", report.scores);
        let total: f32 = report.scores.iter().map(|s| s.score).sum();
        assert!((total - 1.0).abs() < 1e-3, "total {total}");
        // A solid image is classified too, with the same
        // probability contract.
        let report = provider.moderate(&solid()).unwrap();
        assert!(
            report
                .scores
                .iter()
                .all(|s| (0.0..=1.0).contains(&s.score)),
            "scores: {:?}",
            report.scores
        );
    }

    /// Operator labels rename the model's outputs when
    /// their count matches:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_MODERATION_MODEL=...
    /// cargo test -- --ignored moderation_labels_rename
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn moderation_labels_rename_the_outputs() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_MODERATION_MODEL").unwrap();
        let named = vec!["safe".to_string(), "explicit".to_string()];
        let provider =
            OnnxModerationProvider::load(Some(&dylib), &model, Some(named)).unwrap();
        assert_eq!(provider.labels(), ["safe", "explicit"]);
        let report = provider.moderate(&solid()).unwrap();
        assert_eq!(report.scores[0].label, "safe");
        assert_eq!(report.scores[1].label, "explicit");
        // A count that disagrees with the model is a load
        // error, not a mislabeled response.
        let err = expect_load(OnnxModerationProvider::load(
            Some(&dylib),
            &model,
            Some(vec!["only".to_string()]),
        ));
        assert!(err.contains("name one label per output"), "{err}");
    }

    /// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored moderation_load_missing
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn moderation_load_missing_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let message = expect_load(OnnxModerationProvider::load(
            Some(&dylib),
            &missing_path("moderation"),
            None,
        ));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored moderation_load_garbage
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn moderation_load_garbage_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let garbage = temp_file("moderation-garbage", b"this is not an onnx model at all");
        let message =
            expect_load(OnnxModerationProvider::load(Some(&dylib), &garbage, None));
        assert!(message.contains("cannot load the model"), "{message}");
    }
}

/// One detected object: a class label, its confidence,
/// and its box in image pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub label: String,
    pub score: f32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Detects objects in an image, in the shape of the face
/// provider: the classes the model knows, and the boxes it
/// finds.
pub trait DetectionProvider: Send + Sync + std::fmt::Debug {
    /// Whether the model's vocabulary contains `name`,
    /// case-insensitively, matching how class names are
    /// written in URLs.
    fn knows_class(&self, name: &str) -> bool;
    /// The model's class vocabulary, in model order, as an
    /// owned list: for boot-time reporting. Request paths
    /// use `knows_class`; no caller needs the slice itself
    /// (issue #152).
    fn describe(&self) -> Vec<String>;
    fn detect(&self, img: &DynamicImage) -> Result<Vec<Detection>, AiError>;
}

/// The detector's fixed input edge (YOLO letterbox).
const DETECT_INPUT: u32 = 640;
/// NMS IoU ceiling for detection post-processing.
const DETECT_NMS_THRESHOLD: f32 = 0.45;

/// Object detection backed by a YOLOv8-family ONNX export:
/// 640x640 letterbox in, `[1, 4+classes, N]` out (the
/// transposed layout is accepted too), per-class NMS. A
/// v5-style objectness layout is rejected at load.
pub struct OnnxDetectionProvider {
    session: Mutex<Session>,
    input_name: String,
    classes: Vec<String>,
    threshold: f32,
}

impl OnnxDetectionProvider {
    /// Load a detection model. The class vocabulary comes
    /// from the model's own `names` metadata when the
    /// export carries it, else from `labels_path` (one
    /// class per line, in index order); neither is a load
    /// error. The output layout is probed with a dummy run
    /// before returning, so a wrong-family model fails at
    /// boot like every other provider.
    pub fn load(
        dylib: Option<&str>,
        model_path: &str,
        labels_path: Option<&str>,
        threshold: f32,
    ) -> Result<Self, AiError> {
        Self::load_with_classes(dylib, model_path, None, labels_path, threshold)
    }

    /// Load a detection model with an optional class
    /// vocabulary declared by the operator (issue #166, the
    /// `boxes` contract). Declared classes win over the
    /// model's own `names` metadata when both exist: the
    /// declaration is the operator's statement of what the
    /// boxes mean. With no declared classes the vocabulary
    /// resolves exactly as [`Self::load`] documents.
    pub fn load_with_classes(
        dylib: Option<&str>,
        model_path: &str,
        declared_classes: Option<Vec<String>>,
        labels_path: Option<&str>,
        threshold: f32,
    ) -> Result<Self, AiError> {
        let (session, input_name) = build_session(dylib, model_path)?;
        let classes = match declared_classes {
            Some(classes) => classes,
            None => match classes_from_metadata(&session) {
                Some(classes) => classes,
                None => match labels_path {
                    Some(path) => read_labels(path).map_err(|e| {
                        AiError::Load(format!("cannot read the detection labels: {e}"))
                    })?,
                    None => {
                        return Err(AiError::Load(format!(
                            "the model {model_path} carries no class names metadata and no CRABCROP_DETECT_LABELS file was configured"
                        )));
                    }
                },
            },
        };
        let provider = Self {
            session: Mutex::new(session),
            input_name,
            classes,
            threshold,
        };
        // Probe the output layout with a blank image: any
        // detections found are discarded, but a layout the
        // post-processor cannot read fails here.
        let probe = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
        provider.detect(&probe).map_err(|e| {
            AiError::Load(format!(
                "the detection model is not a YOLOv8-family export: {e}"
            ))
        })?;
        Ok(provider)
    }
}

impl std::fmt::Debug for OnnxDetectionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxDetectionProvider")
            .field("classes", &self.classes.len())
            .finish_non_exhaustive()
    }
}

impl DetectionProvider for OnnxDetectionProvider {
    fn knows_class(&self, name: &str) -> bool {
        self.classes.iter().any(|c| c.eq_ignore_ascii_case(name))
    }

    fn describe(&self) -> Vec<String> {
        self.classes.clone()
    }

    fn detect(&self, img: &DynamicImage) -> Result<Vec<Detection>, AiError> {
        let (input, scale, pad_x, pad_y) = letterbox(img);
        let tensor = ort::value::Tensor::from_array((
            [1, 3, DETECT_INPUT as usize, DETECT_INPUT as usize],
            input.into_boxed_slice(),
        ))
        .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
        let mut session = self.session.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
        let outputs = collect_outputs(outputs)?;
        let (_, dims, data) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| AiError::BadOutput("the model produced no outputs".to_string()))?;
        postprocess_detections(
            &data,
            &dims,
            &self.classes,
            self.threshold,
            img.width(),
            img.height(),
            scale,
            pad_x,
            pad_y,
        )
    }
}

/// The class vocabulary from a model's `names` metadata
/// (Ultralytics exports carry it as a Python-dict-style
/// string: `{0: 'person', 1: 'bicycle', ...}`).
fn classes_from_metadata(session: &Session) -> Option<Vec<String>> {
    let raw = session.metadata().ok()?.custom("names")?;
    parse_names_metadata(&raw)
}

/// Extract the quoted names, in order, from a `names`
/// metadata value. Single or double quotes both work.
fn parse_names_metadata(raw: &str) -> Option<Vec<String>> {
    for quote in ['\'', '"'] {
        let mut names = Vec::new();
        let mut rest = raw;
        while let Some(start) = rest.find(quote) {
            let tail = &rest[start + 1..];
            let Some(end) = tail.find(quote) else { break };
            let token = &tail[..end];
            // JSON-style metadata quotes its numeric keys
            // too (`{"0": "person"}`): a bare number is a
            // key, never a class name.
            if !token.chars().all(|c| c.is_ascii_digit()) || token.is_empty() {
                names.push(token.to_string());
            }
            rest = &tail[end + 1..];
        }
        if names.len() >= 2 {
            return Some(names);
        }
    }
    None
}

/// Letterbox an image into the detector's square input:
/// scale to fit, pad with the standard gray, planar CHW
/// at 0..1. Returns the tensor, the scale, and the padding
/// offset, which the post-processor needs to map boxes
/// back to image coordinates.
fn letterbox(img: &DynamicImage) -> (Vec<f32>, f32, u32, u32) {
    let (w, h) = (img.width().max(1), img.height().max(1));
    let scale = (DETECT_INPUT as f32 / w as f32).min(DETECT_INPUT as f32 / h as f32);
    let nw = ((w as f32 * scale).round() as u32).clamp(1, DETECT_INPUT);
    let nh = ((h as f32 * scale).round() as u32).clamp(1, DETECT_INPUT);
    let resized = img.resize_exact(nw, nh, FilterType::Triangle).to_rgb8();
    let pad_x = (DETECT_INPUT - nw) / 2;
    let pad_y = (DETECT_INPUT - nh) / 2;
    let plane = (DETECT_INPUT * DETECT_INPUT) as usize;
    let mut out = vec![114f32 / 255.0; 3 * plane];
    for (i, px) in resized.pixels().enumerate() {
        let x = i as u32 % nw;
        let y = i as u32 / nw;
        let at = ((y + pad_y) * DETECT_INPUT + (x + pad_x)) as usize;
        for c in 0..3 {
            out[c * plane + at] = px[c] as f32 / 255.0;
        }
    }
    (out, scale, pad_x, pad_y)
}

/// Turn a YOLOv8-family output tensor into detections in
/// image coordinates: best class per candidate, score
/// threshold, letterbox inversion, per-class NMS, best
/// first. `dims`/`data` describe the raw first output in
/// either the channels-first or transposed layout.
#[allow(clippy::too_many_arguments)]
fn postprocess_detections(
    data: &[f32],
    dims: &[i64],
    classes: &[String],
    threshold: f32,
    img_w: u32,
    img_h: u32,
    scale: f32,
    pad_x: u32,
    pad_y: u32,
) -> Result<Vec<Detection>, AiError> {
    if dims.len() != 3 || dims[0] != 1 {
        return Err(AiError::BadOutput(format!(
            "expected a [1, 4+classes, N] output, got dims {dims:?}"
        )));
    }
    let rows = (4 + classes.len()) as i64;
    let channels_first = dims[1] == rows;
    let transposed = dims[2] == rows;
    if !channels_first && !transposed {
        return Err(AiError::BadOutput(format!(
            "expected a [1, 4+classes, N] output for {} classes, got dims {dims:?}",
            classes.len()
        )));
    }
    let n = if channels_first { dims[2] } else { dims[1] } as usize;
    let stride = rows as usize;
    let at = |candidate: usize, channel: usize| -> f32 {
        if channels_first {
            data[channel * n + candidate]
        } else {
            data[candidate * stride + channel]
        }
    };
    let mut found: Vec<Detection> = Vec::new();
    for i in 0..n {
        let (mut best_class, mut best_score) = (0usize, f32::MIN);
        for (c, _) in classes.iter().enumerate() {
            let score = at(i, 4 + c);
            if score > best_score {
                best_score = score;
                best_class = c;
            }
        }
        if best_score < threshold {
            continue;
        }
        let (cx, cy, bw, bh) = (at(i, 0), at(i, 1), at(i, 2), at(i, 3));
        let x1 = ((cx - bw / 2.0) - pad_x as f32) / scale;
        let y1 = ((cy - bh / 2.0) - pad_y as f32) / scale;
        let x2 = ((cx + bw / 2.0) - pad_x as f32) / scale;
        let y2 = ((cy + bh / 2.0) - pad_y as f32) / scale;
        let x = x1.round().clamp(0.0, img_w as f32) as u32;
        let y = y1.round().clamp(0.0, img_h as f32) as u32;
        let x2 = x2.round().clamp(0.0, img_w as f32) as u32;
        let y2 = y2.round().clamp(0.0, img_h as f32) as u32;
        if x2 <= x || y2 <= y {
            continue;
        }
        found.push(Detection {
            label: classes[best_class].clone(),
            score: (best_score * 1000.0).round() / 1000.0,
            x,
            y,
            w: x2 - x,
            h: y2 - y,
        });
    }
    Ok(nms_detections(found))
}

/// Per-class non-maximum suppression, best first: a
/// candidate survives only if it overlaps every kept box
/// of the SAME class by at most the IoU ceiling.
fn nms_detections(mut found: Vec<Detection>) -> Vec<Detection> {
    found.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<Detection> = Vec::new();
    for cand in found {
        let suppressed = kept
            .iter()
            .any(|k| k.label == cand.label && detection_iou(k, &cand) > DETECT_NMS_THRESHOLD);
        if !suppressed {
            kept.push(cand);
        }
    }
    kept
}

fn detection_iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }
    let inter = ((x2 - x1) * (y2 - y1)) as f32;
    let union = (a.w * a.h + b.w * b.h) as f32 - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Supplies a focal point for smart crops, in the shape of
/// the other providers: implemented by the ONNX provider in
/// production and by fakes in tests. The point is a pair of
/// fractions of the image width and height, each in
/// 0.0..=1.0: the shape `crate::focus::FocalPoint` carries
/// through the cropper, without this module depending on it.
pub trait FocalProvider: Send + Sync + std::fmt::Debug {
    fn focal_point(&self, img: &DynamicImage) -> Result<(f32, f32), AiError>;
}

/// Focal point backed by a declared heatmap model (issue
/// #166, the `focal` contract). The model declares one
/// fixed `[1, C, H, W]` float input, C 1 or 3, and produces
/// exactly one output: a heatmap in `[H, W]`, `[1, H, W]`,
/// or `[1, 1, H, W]` layout. The image is resampled to the
/// model's own input size (RGB planes for a 3-channel
/// input, luma for 1-channel, values at 0..1); the point
/// is the heatmap's argmax cell center. Both the input
/// shape and the output contract are probed at load, so a
/// model that does not fit refuses at boot.
pub struct OnnxFocalProvider {
    session: Mutex<Session>,
    input_name: String,
    channels: usize,
    input_w: u32,
    input_h: u32,
}

impl OnnxFocalProvider {
    /// Load a focal model against the runtime at `dylib`
    /// (or the runtime's own search when `None`).
    pub fn load(dylib: Option<&str>, model_path: &str) -> Result<Self, AiError> {
        let (session, input_name) = build_session(dylib, model_path)?;
        let (channels, input_w, input_h) = focal_input_shape(&session, model_path)?;
        let provider = Self {
            session: Mutex::new(session),
            input_name,
            channels,
            input_w,
            input_h,
        };
        // Probe the output contract with a blank image, like
        // the detection probe: a wrong-shape output fails
        // here, not on the first smart-crop request.
        let probe = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
        provider.run_heatmap(&probe).map_err(|e| {
            AiError::Load(format!(
                "the focal model {model_path} does not produce a heatmap output: {e}"
            ))
        })?;
        Ok(provider)
    }

    /// Run the model and return the heatmap's dims and
    /// values, the output contract validated.
    fn run_heatmap(&self, img: &DynamicImage) -> Result<(Vec<i64>, Vec<f32>), AiError> {
        let resized = img.resize_exact(self.input_w, self.input_h, FilterType::Triangle);
        let plane = (self.input_w * self.input_h) as usize;
        let mut input = vec![0f32; self.channels * plane];
        if self.channels == 3 {
            let rgb = resized.to_rgb8();
            for (i, px) in rgb.pixels().enumerate() {
                for c in 0..3 {
                    input[c * plane + i] = px[c] as f32 / 255.0;
                }
            }
        } else {
            let gray = resized.to_luma8();
            for (i, px) in gray.pixels().enumerate() {
                input[i] = px[0] as f32 / 255.0;
            }
        }
        let tensor = ort::value::Tensor::from_array((
            [
                1usize,
                self.channels,
                self.input_h as usize,
                self.input_w as usize,
            ],
            input.into_boxed_slice(),
        ))
        .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
        let mut session = self.session.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
        let outputs = collect_outputs(outputs)?;
        if outputs.len() != 1 {
            return Err(AiError::BadOutput(format!(
                "the focal model produced {} outputs; the contract takes exactly one heatmap",
                outputs.len()
            )));
        }
        let (_, dims, data) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| AiError::BadOutput("the model produced no outputs".to_string()))?;
        heatmap_dims(&dims, data.len())?;
        Ok((dims, data))
    }
}

impl std::fmt::Debug for OnnxFocalProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxFocalProvider")
            .field("input_name", &self.input_name)
            .field("input", &(self.channels, self.input_w, self.input_h))
            .finish_non_exhaustive()
    }
}

impl FocalProvider for OnnxFocalProvider {
    fn focal_point(&self, img: &DynamicImage) -> Result<(f32, f32), AiError> {
        let (dims, data) = self.run_heatmap(img)?;
        heatmap_argmax(&dims, &data)
    }
}

/// The largest declared focal input, in pixels: the
/// detection provider's fixed 640x640 input is the biggest
/// of the built-in contracts, so a declared focal frame may
/// not exceed it. The bound also caps the input tensor's
/// bytes (at most 3 channels of f32 over the frame, about
/// 5 MB) and the boot probe's resized copy, which are
/// allocated from the declared shape (audit R2-26).
const MAX_FOCAL_INPUT_PIXELS: i64 = 640 * 640;

/// Validate a focal model's declared input dims against the
/// contract: a fixed `[1, C, H, W]` shape with C 1 (luma) or
/// 3 (RGB) and at most [`MAX_FOCAL_INPUT_PIXELS`] pixels.
/// Pure, so the contract is testable without a model.
fn focal_dims(dims: &[i64], model_path: &str) -> Result<(usize, u32, u32), AiError> {
    match dims {
        [1, c, h, w] if (*c == 1 || *c == 3) && *h > 0 && *w > 0 => {
            let pixels = h.checked_mul(*w).ok_or_else(|| {
                AiError::Load(format!(
                    "the focal model {model_path} declares an input shape {dims:?} \
                     whose pixel count overflows"
                ))
            })?;
            if pixels > MAX_FOCAL_INPUT_PIXELS {
                return Err(AiError::Load(format!(
                    "the focal model {model_path} declares a {w}x{h} input ({pixels} pixels), \
                     over the {MAX_FOCAL_INPUT_PIXELS} pixel cap for declared focal models"
                )));
            }
            Ok((*c as usize, *w as u32, *h as u32))
        }
        _ => Err(AiError::Load(format!(
            "the focal model {model_path} must declare a fixed [1, 1 or 3, H, W] input, got {dims:?}"
        ))),
    }
}

/// The focal model's declared input contract: a fixed
/// `[1, C, H, W]` tensor shape with C 1 (luma) or 3 (RGB),
/// bounded by [`MAX_FOCAL_INPUT_PIXELS`]. A dynamic
/// dimension anywhere, another rank, another channel count,
/// or a frame over the cap is a load error naming the
/// actual shape.
fn focal_input_shape(session: &Session, model_path: &str) -> Result<(usize, u32, u32), AiError> {
    let input = session
        .inputs()
        .first()
        .ok_or_else(|| AiError::Load(format!("the focal model {model_path} has no inputs")))?;
    let dims: Vec<i64> = match input.dtype() {
        ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
        _ => {
            return Err(AiError::Load(format!(
                "the focal model {model_path} input is not a tensor"
            )));
        }
    };
    focal_dims(&dims, model_path)
}

/// The heatmap behind an output: `[H, W]`, `[1, H, W]`, or
/// `[1, 1, H, W]`, with exactly H * W values. Returns the
/// grid size.
fn heatmap_dims(dims: &[i64], values: usize) -> Result<(usize, usize), AiError> {
    let (h, w) = match dims {
        [h, w] => (*h, *w),
        [1, h, w] => (*h, *w),
        [1, 1, h, w] => (*h, *w),
        _ => {
            return Err(AiError::BadOutput(format!(
                "expected a heatmap output ([H, W], [1, H, W], or [1, 1, H, W]), got dims {dims:?}"
            )));
        }
    };
    if h <= 0 || w <= 0 || h as usize * w as usize != values {
        return Err(AiError::BadOutput(format!(
            "heatmap dims {dims:?} do not match the {values} output values"
        )));
    }
    Ok((h as usize, w as usize))
}

/// The focal point of a heatmap: the argmax cell's center,
/// as fractions of the width and height. Ties resolve to
/// the first maximum in row-major order (a strict-greater
/// scan), so repeated runs always agree.
fn heatmap_argmax(dims: &[i64], data: &[f32]) -> Result<(f32, f32), AiError> {
    let (h, w) = heatmap_dims(dims, data.len())?;
    let mut best = 0usize;
    for (i, v) in data.iter().enumerate() {
        if *v > data[best] {
            best = i;
        }
    }
    Ok((
        (best % w) as f32 / w as f32 + 0.5 / w as f32,
        (best / w) as f32 / h as f32 + 0.5 / h as f32,
    ))
}

#[cfg(test)]
mod detect_tests {
    use super::*;

    fn classes() -> Vec<String> {
        ["person", "car"].iter().map(|s| s.to_string()).collect()
    }

    /// A channels-first output with two candidates on a
    /// 100x100 image (no letterbox padding, scale 6.4: the
    /// image was scaled to 640). Candidate 0: a person at
    /// center (320,320), 128x256 in letterbox space, score
    /// 0.9. Candidate 1: a car overlapping the person class
    /// region but a different class, score 0.8.
    fn synthetic(channels_first: bool) -> (Vec<f32>, Vec<i64>) {
        let classes = classes();
        let rows = 4 + classes.len();
        let n = 2usize;
        let mut data = vec![0f32; rows * n];
        let mut set = |cand: usize, ch: usize, v: f32| {
            if channels_first {
                data[ch * n + cand] = v;
            } else {
                data[cand * rows + ch] = v;
            }
        };
        // candidate 0: person
        set(0, 0, 320.0);
        set(0, 1, 320.0);
        set(0, 2, 128.0);
        set(0, 3, 256.0);
        set(0, 4, 0.9);
        // candidate 1: car, same box
        set(1, 0, 320.0);
        set(1, 1, 320.0);
        set(1, 2, 128.0);
        set(1, 3, 256.0);
        set(1, 5, 0.8);
        let dims = if channels_first {
            vec![1, rows as i64, n as i64]
        } else {
            vec![1, n as i64, rows as i64]
        };
        (data, dims)
    }

    #[test]
    fn postprocess_reads_both_layouts() {
        for channels_first in [true, false] {
            let (data, dims) = synthetic(channels_first);
            let found =
                postprocess_detections(&data, &dims, &classes(), 0.5, 100, 100, 6.4, 0, 0).unwrap();
            assert_eq!(found.len(), 2, "channels_first={channels_first}");
            assert_eq!(found[0].label, "person");
            assert_eq!(found[0].score, 0.9);
            // (320-64)/6.4 = 40, (320-128)/6.4 = 30, 128/6.4 = 20, 256/6.4 = 40
            assert_eq!(
                (found[0].x, found[0].y, found[0].w, found[0].h),
                (40, 30, 20, 40)
            );
            assert_eq!(found[1].label, "car");
        }
    }

    #[test]
    fn postprocess_threshold_and_same_class_nms() {
        let classes = classes();
        // One strong person, one weaker person at the same
        // box (suppressed), one weak car (below threshold).
        let rows = 4 + classes.len();
        let n = 3usize;
        let mut data = vec![0f32; rows * n];
        let mut set = |cand: usize, ch: usize, v: f32| data[ch * n + cand] = v;
        for (cand, score) in [(0usize, 0.9f32), (1, 0.7), (2, 0.3)] {
            set(cand, 0, 100.0);
            set(cand, 1, 100.0);
            set(cand, 2, 50.0);
            set(cand, 3, 50.0);
            set(cand, 4, score);
        }
        set(2, 4, 0.0);
        set(2, 5, 0.3);
        let found = postprocess_detections(
            &data,
            &[1, rows as i64, n as i64],
            &classes,
            0.5,
            640,
            640,
            1.0,
            0,
            0,
        )
        .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].score, 0.9);
    }

    #[test]
    fn postprocess_rejects_a_wrong_family_layout() {
        // A v5-style [1, N, 85] output for an 80-class
        // vocabulary does not fit 4+classes either way.
        let classes: Vec<String> = (0..80).map(|i| format!("class{i}")).collect();
        let err = postprocess_detections(
            &[0.0; 16],
            &[1, 25200, 85],
            &classes,
            0.5,
            10,
            10,
            1.0,
            0,
            0,
        )
        .unwrap_err();
        assert!(
            matches!(err, AiError::BadOutput(_)) && err.to_string().contains("4+classes"),
            "{err}"
        );
    }

    #[test]
    fn ai_error_display_keeps_the_cause() {
        // The typed variants carry the same messages the
        // plain strings carried, so boot diagnostics and log
        // lines read as they did before AP-16.
        assert_eq!(
            AiError::Load("cannot load the model x: boom".to_string()).to_string(),
            "cannot load the model x: boom"
        );
        assert_eq!(
            AiError::Inference("inference failed: boom".to_string()).to_string(),
            "inference failed: boom"
        );
        assert_eq!(
            AiError::BadOutput("unexpected model output shape: [1]".to_string()).to_string(),
            "unexpected model output shape: [1]"
        );
        assert!(AiError::LockPoisoned.to_string().contains("poisoned"));
    }

    #[test]
    fn names_metadata_parses_both_quote_styles() {
        let single = "{0: 'person', 1: 'bicycle', 2: 'car'}";
        assert_eq!(
            parse_names_metadata(single).unwrap(),
            ["person", "bicycle", "car"]
        );
        let double = "{\"0\": \"person\", \"1\": \"car\"}";
        assert_eq!(parse_names_metadata(double).unwrap(), ["person", "car"]);
        assert!(parse_names_metadata("no names here").is_none());
    }

    #[test]
    fn letterbox_of_a_square_image_is_unpadded_at_scale_one() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            DETECT_INPUT,
            DETECT_INPUT,
            image::Rgb([10, 20, 30]),
        ));
        let (input, scale, pad_x, pad_y) = letterbox(&img);
        assert_eq!(scale, 1.0);
        assert_eq!((pad_x, pad_y), (0, 0));
        assert_eq!(input.len(), 3 * 640 * 640);
        // Planar CHW at 0..1: red plane, then green, then blue.
        let plane = 640 * 640;
        assert!((input[0] - 10.0 / 255.0).abs() < 1e-6, "red {}", input[0]);
        assert!((input[plane] - 20.0 / 255.0).abs() < 1e-6, "green");
        assert!((input[2 * plane] - 30.0 / 255.0).abs() < 1e-6, "blue");
    }

    #[test]
    fn letterbox_pads_a_wide_image_top_and_bottom_with_the_standard_gray() {
        // 1280x640 fits at scale 0.5 as 640x320, centered:
        // 160 rows of pad above and below the content.
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            1280,
            640,
            image::Rgb([255, 0, 0]),
        ));
        let (input, scale, pad_x, pad_y) = letterbox(&img);
        assert_eq!(scale, 0.5);
        assert_eq!((pad_x, pad_y), (0, 160));
        let plane = 640 * 640;
        let gray = 114.0 / 255.0;
        // The padding reads as the standard gray in every
        // plane, including the last pad row.
        assert!((input[0] - gray).abs() < 1e-6);
        assert!((input[plane] - gray).abs() < 1e-6);
        assert!((input[159 * 640] - gray).abs() < 1e-6);
        // The first content row is the image itself: red at
        // full strength, green at zero.
        let first_content = 160 * 640;
        assert!((input[first_content] - 1.0).abs() < 1e-6);
        assert!(input[plane + first_content].abs() < 1e-6);
    }

    #[test]
    fn letterbox_pads_a_tall_image_on_the_sides_and_upscales_a_small_one() {
        // 320x1280 fits at scale 0.5 as 160x640: the pad is
        // horizontal instead.
        let tall = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            320,
            1280,
            image::Rgb([0, 0, 0]),
        ));
        let (_, scale, pad_x, pad_y) = letterbox(&tall);
        assert_eq!(scale, 0.5);
        assert_eq!((pad_x, pad_y), (240, 0));
        // 100x50 upscales by 6.4 to 640x320: vertical pad.
        let small =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(100, 50, image::Rgb([0, 0, 0])));
        let (_, scale, pad_x, pad_y) = letterbox(&small);
        assert_eq!(scale, 6.4);
        assert_eq!((pad_x, pad_y), (0, 160));
    }

    /// Live check with a real YOLOv8 export: a street
    /// fixture must yield a person and a vehicle. Ignored:
    /// needs the runtime, a detection model, and a photo:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_DETECT_MODEL=...
    /// CRABCROP_TEST_DETECT_LABELS=... (optional)
    /// CRABCROP_TEST_IMAGE=...
    #[test]
    #[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
    fn detect_live_finds_people_and_vehicles() {
        let var = |name: &str| std::env::var(name).unwrap();
        let provider = OnnxDetectionProvider::load(
            Some(&var("CRABCROP_TEST_ORT_DYLIB")),
            &var("CRABCROP_TEST_DETECT_MODEL"),
            std::env::var("CRABCROP_TEST_DETECT_LABELS").ok().as_deref(),
            0.35,
        )
        .unwrap();
        println!("classes: {} known", provider.describe().len());
        let img = image::open(var("CRABCROP_TEST_IMAGE")).unwrap();
        let found = provider.detect(&img).unwrap();
        for d in &found {
            println!(
                "  {} {:.3} at {},{} {}x{}",
                d.label, d.score, d.x, d.y, d.w, d.h
            );
        }
        assert!(
            found.iter().any(|d| d.label == "person"),
            "no person in {found:?}"
        );
        assert!(
            found
                .iter()
                .any(|d| ["car", "truck", "bus"].contains(&d.label.as_str())),
            "no vehicle in {found:?}"
        );
    }

    /// Write bytes to a fresh temp file and return its
    /// path: the load() failure tests need a real label
    /// file without committing fixtures.
    fn temp_file(tag: &str, contents: &[u8]) -> String {
        let path =
            std::env::temp_dir().join(format!("crabcrop-ai-load-{tag}-{}.bin", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// A path inside the temp dir that is never created.
    fn missing_path(tag: &str) -> String {
        std::env::temp_dir()
            .join(format!(
                "crabcrop-ai-load-missing-{tag}-{}.onnx",
                std::process::id()
            ))
            .to_string_lossy()
            .into_owned()
    }

    /// Assert a load() result failed as AiError::Load and
    /// return the message for wording checks.
    fn expect_load<T>(result: Result<T, AiError>) -> String {
        match result {
            Err(AiError::Load(message)) => message,
            Err(other) => panic!("expected a load error, got {other}"),
            Ok(_) => panic!("expected a load error, but the load succeeded"),
        }
    }

    /// Load-failure coverage (#179, item 2): a missing
    /// detection model file fails as AiError::Load.
    /// CRABCROP_TEST_ORT_DYLIB=...
    /// cargo test -- --ignored detection_load_missing
    #[test]
    #[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
    fn detection_load_missing_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let message = expect_load(OnnxDetectionProvider::load(
            Some(&dylib),
            &missing_path("detect"),
            None,
            0.5,
        ));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// A garbage detection model file fails the same way.
    /// CRABCROP_TEST_ORT_DYLIB=...
    /// cargo test -- --ignored detection_load_garbage
    #[test]
    #[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
    fn detection_load_garbage_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let garbage = temp_file("detect-garbage", b"this is not an onnx model at all");
        let message = expect_load(OnnxDetectionProvider::load(
            Some(&dylib),
            &garbage,
            None,
            0.5,
        ));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// A model with no `names` metadata (the removebg
    /// model stands in) and no labels file configured has
    /// no class vocabulary: Load, naming the remedy.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
    /// cargo test -- --ignored detection_load_model_without_class_names
    #[test]
    #[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
    fn detection_load_model_without_class_names_or_labels_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_REMOVEBG_MODEL").unwrap();
        let message = expect_load(OnnxDetectionProvider::load(Some(&dylib), &model, None, 0.5));
        assert!(
            message.contains("carries no class names metadata"),
            "{message}"
        );
    }

    /// The same model with a labels path that does not
    /// exist fails reading the labels: Load.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
    /// cargo test -- --ignored detection_load_missing_labels
    #[test]
    #[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
    fn detection_load_missing_labels_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_REMOVEBG_MODEL").unwrap();
        let message = expect_load(OnnxDetectionProvider::load(
            Some(&dylib),
            &model,
            Some(&missing_path("detect-labels")),
            0.5,
        ));
        assert!(
            message.contains("cannot read the detection labels"),
            "{message}"
        );
    }

    /// With labels supplied, a non-YOLO model reaches the
    /// load probe and fails it: Load, classified like the
    /// other providers' probe failures.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
    /// cargo test -- --ignored detection_load_non_yolo
    #[test]
    #[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
    fn detection_load_non_yolo_model_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_REMOVEBG_MODEL").unwrap();
        let labels = temp_file("detect-labels", b"person\ncar\n");
        let message = expect_load(OnnxDetectionProvider::load(
            Some(&dylib),
            &model,
            Some(&labels),
            0.5,
        ));
        assert!(message.contains("not a YOLOv8-family export"), "{message}");
    }

    /// Eighty class names in model order with index 0
    /// renamed: enough for the real YOLO export's probe,
    /// and proof of which vocabulary won.
    fn renamed_vocabulary() -> Vec<String> {
        (0..80)
            .map(|i| {
                if i == 0 {
                    "human".to_string()
                } else {
                    format!("class{i}")
                }
            })
            .collect()
    }

    /// Declared classes win over the model's own `names`
    /// metadata (issue #166): the same YOLO export, loaded
    /// with a declaration vocabulary, reports and returns
    /// the declaration's names.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_DETECT_MODEL=...
    /// CRABCROP_TEST_IMAGE=...
    /// cargo test -- --ignored declared_classes_win
    #[test]
    #[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
    fn declared_classes_win_over_model_metadata() {
        let var = |name: &str| std::env::var(name).unwrap();
        let provider = OnnxDetectionProvider::load_with_classes(
            Some(&var("CRABCROP_TEST_ORT_DYLIB")),
            &var("CRABCROP_TEST_DETECT_MODEL"),
            Some(renamed_vocabulary()),
            None,
            0.35,
        )
        .unwrap();
        assert_eq!(provider.describe()[0], "human");
        assert!(provider.knows_class("human"));
        assert!(!provider.knows_class("person"));
        let img = image::open(var("CRABCROP_TEST_IMAGE")).unwrap();
        let found = provider.detect(&img).unwrap();
        assert!(
            found.iter().any(|d| d.label == "human"),
            "no human in {found:?}"
        );
        assert!(
            found.iter().all(|d| d.label != "person"),
            "metadata vocabulary leaked into {found:?}"
        );
    }

    /// A declared detector whose model file is missing
    /// fails as AiError::Load.
    /// CRABCROP_TEST_ORT_DYLIB=...
    /// cargo test -- --ignored declared_load_missing
    #[test]
    #[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
    fn declared_load_missing_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let message = expect_load(OnnxDetectionProvider::load_with_classes(
            Some(&dylib),
            &missing_path("declared"),
            Some(renamed_vocabulary()),
            None,
            0.5,
        ));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// A garbage declared model file fails the same way.
    /// CRABCROP_TEST_ORT_DYLIB=...
    /// cargo test -- --ignored declared_load_garbage
    #[test]
    #[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
    fn declared_load_garbage_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let garbage = temp_file("declared-garbage", b"this is not an onnx model at all");
        let message = expect_load(OnnxDetectionProvider::load_with_classes(
            Some(&dylib),
            &garbage,
            Some(renamed_vocabulary()),
            None,
            0.5,
        ));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// Declared classes do not save a non-YOLO model: the
    /// probe still refuses it as Load.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
    /// cargo test -- --ignored declared_load_non_yolo
    #[test]
    #[ignore = "needs the ONNX runtime and a detection model (env-configured)"]
    fn declared_load_non_yolo_model_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_REMOVEBG_MODEL").unwrap();
        let message = expect_load(OnnxDetectionProvider::load_with_classes(
            Some(&dylib),
            &model,
            Some(vec!["a".to_string(), "b".to_string()]),
            None,
            0.5,
        ));
        assert!(message.contains("not a YOLOv8-family export"), "{message}");
    }
}

#[cfg(test)]
mod focal_tests {
    use super::*;

    fn grid(h: usize, w: usize, max_at: usize) -> Vec<f32> {
        let mut data = vec![0f32; h * w];
        data[max_at] = 9.0;
        data
    }

    #[test]
    fn heatmap_argmax_reads_every_accepted_layout() {
        // 4 rows x 5 cols, max at row 2, col 3: the cell
        // center is (3.5/5, 2.5/4).
        let data = grid(4, 5, 2 * 5 + 3);
        for dims in [&[4, 5][..], &[1, 4, 5][..], &[1, 1, 4, 5][..]] {
            let (x, y) = heatmap_argmax(dims, &data).unwrap();
            assert!((x - 0.7).abs() < 1e-6, "x {x} for {dims:?}");
            assert!((y - 0.625).abs() < 1e-6, "y {y} for {dims:?}");
        }
    }

    #[test]
    fn heatmap_argmax_takes_the_first_maximum_in_row_major_order() {
        let mut data = grid(3, 3, 8);
        data[2] = 9.0; // an equal maximum, earlier in row-major order
        let (x, y) = heatmap_argmax(&[3, 3], &data).unwrap();
        assert!((x - 2.5 / 3.0).abs() < 1e-6, "x {x}");
        assert!((y - 0.5 / 3.0).abs() < 1e-6, "y {y}");
    }

    #[test]
    fn heatmap_argmax_of_a_single_cell_is_the_center() {
        let (x, y) = heatmap_argmax(&[1, 1], &[0.25]).unwrap();
        assert_eq!((x, y), (0.5, 0.5));
    }

    #[test]
    fn heatmap_argmax_rejects_shapes_outside_the_contract() {
        let cases: Vec<(Vec<i64>, usize)> = vec![
            (vec![9], 9),             // a vector, not a grid
            (vec![2, 3, 3], 18),      // a batch of two grids
            (vec![1, 2, 3, 3], 18),   // two channels
            (vec![1, 1, 1, 3, 3], 9), // rank five
            (vec![3, 3], 8),          // fewer values than cells
            (vec![3, 3], 10),         // more values than cells
            (vec![0, 3], 0),          // an empty grid
        ];
        for (dims, len) in cases {
            let data = vec![0f32; len];
            let err = heatmap_argmax(&dims, &data).unwrap_err();
            assert!(matches!(err, AiError::BadOutput(_)), "{dims:?}: {err}");
        }
    }

    /// Assert a focal load() result failed as AiError::Load
    /// and return the message for wording checks.
    fn expect_load(result: Result<OnnxFocalProvider, AiError>) -> String {
        match result {
            Err(AiError::Load(message)) => message,
            Err(other) => panic!("expected a load error, got {other}"),
            Ok(_) => panic!("expected a load error, but the load succeeded"),
        }
    }

    /// Live check with the authored luma-heatmap model
    /// (spikes/166-focal): the brightest block of the
    /// fixture is the argmax. Ignored: needs the runtime
    /// and the model: CRABCROP_TEST_ORT_DYLIB=...
    /// CRABCROP_TEST_FOCAL_MODEL=...
    /// cargo test -- --ignored focal_live
    #[test]
    #[ignore = "needs the ONNX runtime and a focal model (env-configured)"]
    fn focal_live_point_tracks_the_brightest_block() {
        let var = |name: &str| std::env::var(name).unwrap();
        let provider = OnnxFocalProvider::load(
            Some(&var("CRABCROP_TEST_ORT_DYLIB")),
            &var("CRABCROP_TEST_FOCAL_MODEL"),
        )
        .unwrap();
        // 64x64 black with a white block at x 40..48,
        // y 8..16; the model sees it at 32x32, so the
        // block lands at x 20..24, y 4..8 and the first
        // row-major maximum is its top-left cell.
        let mut img = image::RgbImage::new(64, 64);
        for (x, y, p) in img.enumerate_pixels_mut() {
            if (40..48).contains(&x) && (8..16).contains(&y) {
                *p = image::Rgb([255, 255, 255]);
            }
        }
        let (x, y) = provider.focal_point(&DynamicImage::ImageRgb8(img)).unwrap();
        assert!((0.60..0.70).contains(&x), "x {x}");
        assert!((0.10..0.20).contains(&y), "y {y}");
    }

    /// A missing focal model file fails as AiError::Load.
    /// CRABCROP_TEST_ORT_DYLIB=...
    /// cargo test -- --ignored focal_load_missing
    #[test]
    #[ignore = "needs the ONNX runtime and a focal model (env-configured)"]
    fn focal_load_missing_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let missing = std::env::temp_dir()
            .join(format!(
                "crabcrop-focal-missing-{}.onnx",
                std::process::id()
            ))
            .to_string_lossy()
            .into_owned();
        let message = expect_load(OnnxFocalProvider::load(Some(&dylib), &missing));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// A garbage focal model file fails the same way.
    /// CRABCROP_TEST_ORT_DYLIB=...
    /// cargo test -- --ignored focal_load_garbage
    #[test]
    #[ignore = "needs the ONNX runtime and a focal model (env-configured)"]
    fn focal_load_garbage_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let path =
            std::env::temp_dir().join(format!("crabcrop-focal-garbage-{}.bin", std::process::id()));
        std::fs::write(&path, b"this is not an onnx model at all").unwrap();
        let message = expect_load(OnnxFocalProvider::load(
            Some(&dylib),
            &path.to_string_lossy(),
        ));
        assert!(message.contains("cannot load the model"), "{message}");
    }

    /// A multi-output model (the face detector stands in)
    /// fails the probe: the contract takes exactly one
    /// heatmap.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_FACE_MODEL=...
    /// cargo test -- --ignored focal_load_multi_output
    #[test]
    #[ignore = "needs the ONNX runtime and a focal model (env-configured)"]
    fn focal_load_multi_output_model_is_a_load_error() {
        let var = |name: &str| std::env::var(name).unwrap();
        let message = expect_load(OnnxFocalProvider::load(
            Some(&var("CRABCROP_TEST_ORT_DYLIB")),
            &var("CRABCROP_TEST_FACE_MODEL"),
        ));
        assert!(
            message.contains("does not produce a heatmap output"),
            "{message}"
        );
        assert!(message.contains("exactly one heatmap"), "{message}");
    }

    /// The segmentation model fails the same probe with
    /// its seven outputs.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
    /// cargo test -- --ignored focal_load_seven_output
    #[test]
    #[ignore = "needs the ONNX runtime and a focal model (env-configured)"]
    fn focal_load_seven_output_model_is_a_load_error() {
        let var = |name: &str| std::env::var(name).unwrap();
        let message = expect_load(OnnxFocalProvider::load(
            Some(&var("CRABCROP_TEST_ORT_DYLIB")),
            &var("CRABCROP_TEST_REMOVEBG_MODEL"),
        ));
        assert!(message.contains("exactly one heatmap"), "{message}");
    }

    /// A model whose input shape is dynamic (the identity
    /// fixture) never reaches the probe: the input contract
    /// refuses it at load.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_IDENTITY_MODEL=...
    /// cargo test -- --ignored focal_load_dynamic_input
    #[test]
    #[ignore = "needs the ONNX runtime and a focal model (env-configured)"]
    fn focal_load_dynamic_input_model_is_a_load_error() {
        let var = |name: &str| std::env::var(name).unwrap();
        let message = expect_load(OnnxFocalProvider::load(
            Some(&var("CRABCROP_TEST_ORT_DYLIB")),
            &var("CRABCROP_TEST_IDENTITY_MODEL"),
        ));
        assert!(
            message.contains("must declare a fixed [1, 1 or 3, H, W] input"),
            "{message}"
        );
    }

    #[test]
    fn focal_dims_accept_shapes_within_the_cap() {
        // The detection provider's fixed frame size is the
        // cap itself; ordinary saliency shapes sit far
        // below it.
        assert_eq!(
            focal_dims(&[1, 3, 640, 640], "m.onnx").unwrap(),
            (3, 640, 640)
        );
        assert_eq!(
            focal_dims(&[1, 1, 256, 256], "m.onnx").unwrap(),
            (1, 256, 256)
        );
        assert_eq!(focal_dims(&[1, 3, 1, 1], "m.onnx").unwrap(), (3, 1, 1));
    }

    #[test]
    fn focal_dims_reject_shapes_over_the_pixel_cap() {
        // The audit's shape: a small declared file hiding a
        // 16384x16384 frame would materialize gigabytes at
        // the boot probe (audit R2-26).
        let err = focal_dims(&[1, 3, 16384, 16384], "m.onnx").unwrap_err();
        assert!(err.to_string().contains("pixel cap"), "{err}");
        // A 4096x4096 frame boots but would repeat a roughly
        // 200 MB tensor allocation per smart crop: refused.
        let err = focal_dims(&[1, 1, 4096, 4096], "m.onnx").unwrap_err();
        assert!(err.to_string().contains("pixel cap"), "{err}");
        // One axis over the cap with the other at 1 pixel:
        // the product is what is bounded.
        assert!(focal_dims(&[1, 3, 640, 641], "m.onnx").is_err());
        // Absurd dims refuse cleanly instead of overflowing
        // the pixel count.
        assert!(focal_dims(&[1, 3, i64::MAX, i64::MAX], "m.onnx").is_err());
    }

    #[test]
    fn focal_dims_keep_the_shape_contract() {
        // Rank, batch, channels, and zero axes refuse with
        // the shape message, as before the cap existed.
        for dims in [
            vec![3, 224, 224],
            vec![2, 3, 224, 224],
            vec![1, 2, 224, 224],
            vec![1, 3, 0, 224],
            vec![1, 3, 224, 0],
            vec![1, 3, -1, 224],
        ] {
            let err = focal_dims(&dims, "m.onnx").unwrap_err();
            assert!(
                err.to_string()
                    .contains("must declare a fixed [1, 1 or 3, H, W] input"),
                "{dims:?}: {err}"
            );
        }
    }
}
