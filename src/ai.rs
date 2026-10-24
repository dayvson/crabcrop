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
}

/// The process-wide ONNX environment, created on first use:
/// in practice that is at boot, when the server preloads its
/// configured models.
static ENVIRONMENT: OnceLock<ort::environment::Environment> = OnceLock::new();

fn environment(dylib: Option<&str>) -> Result<&'static ort::environment::Environment, AiError> {
    if let Some(env) = ENVIRONMENT.get() {
        return Ok(env);
    }
    let builder = match dylib {
        Some(path) => ort::init_from(path)
            .map_err(|e| AiError::Load(format!("cannot load the ONNX runtime from {path}: {e}")))?,
        None => ort::init(),
    };
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
    /// runtime's own search when `None`).
    pub fn load(dylib: Option<&str>, model_path: &str) -> Result<Self, AiError> {
        let (session, input_name) = build_session(dylib, model_path)?;
        Ok(Self {
            session: Mutex::new(session),
            input_name,
        })
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
        Ok(postprocess(data, img.width(), img.height()))
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
/// then resize back to the source dimensions.
fn postprocess(data: &[f32], w: u32, h: u32) -> GrayImage {
    let (min, max) = data
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let range = (max - min).max(f32::EPSILON);
    let mask = GrayImage::from_fn(320, 320, |x, y| {
        let v = data[(y * 320 + x) as usize];
        image::Luma([((v - min) / range * 255.0).round() as u8])
    });
    if (w, h) == (320, 320) {
        mask
    } else {
        image::imageops::resize(&mask, w, h, FilterType::Triangle)
    }
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
        let out = probe.upscale(&DynamicImage::new_rgb8(16, 16))?;
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
        let mask = postprocess(&data, 320, 320);
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
        let mask = postprocess(&data, 320, 320);
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
        let mask = postprocess(&data, 160, 160);
        assert_eq!((mask.width(), mask.height()), (160, 160));
        assert_eq!(mask.get_pixel(0, 0)[0], 0);
        assert_eq!(mask.get_pixel(0, 159)[0], 255);
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
            let emb = run_text(&mut text, &text_inputs, &ids, &mask)?;
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
        let emb = provider.embed_image(&probe)?;
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
fn read_labels(path: &str) -> Result<Vec<String>, AiError> {
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
        let (session, input_name) = build_session(dylib, model_path)?;
        let classes = match classes_from_metadata(&session) {
            Some(classes) => classes,
            None => match labels_path {
                Some(path) => read_labels(path)
                    .map_err(|e| AiError::Load(format!("cannot read the detection labels: {e}")))?,
                None => {
                    return Err(AiError::Load(format!(
                        "the model {model_path} carries no class names metadata and no CRABCROP_DETECT_LABELS file was configured"
                    )));
                }
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
}
