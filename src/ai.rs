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

/// The LaMa export's fixed input size: image and mask are
/// both fed at this square resolution, whatever the crop
/// window's shape (issue #176).
pub const INPAINT_INPUT_SIZE: u32 = 512;

/// A white-on-black mask of one rectangle, at the image's
/// own resolution: the mask `remove_region` hands to the
/// inpainting provider (white is the hole to fill).
pub fn region_mask(width: u32, height: u32, x: u32, y: u32, w: u32, h: u32) -> GrayImage {
    let mut mask = GrayImage::new(width, height);
    for py in y..y.saturating_add(h).min(height) {
        for px in x..x.saturating_add(w).min(width) {
            mask.put_pixel(px, py, image::Luma([255]));
        }
    }
    mask
}

/// The bounding box of the mask's nonzero pixels, as
/// (x, y, w, h); `None` when the mask removes nothing.
fn mask_bbox(mask: &GrayImage) -> Option<(u32, u32, u32, u32)> {
    let (mut min_x, mut min_y) = (u32::MAX, u32::MAX);
    let (mut max_x, mut max_y) = (0u32, 0u32);
    for (x, y, p) in mask.enumerate_pixels() {
        if p[0] > 0 {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if min_x == u32::MAX {
        return None;
    }
    Some((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

/// The crop window an inpainting run works on: a square
/// centered on the mask's bounding box, with a side of
/// twice the box's larger axis so the model sees as much
/// surrounding context as hole, clamped into the image (a
/// window axis stops at the image edge, so an image
/// smaller than the square on one axis makes the window
/// non-square there; the resize to the model's square
/// input absorbs the difference). Pure, so the geometry
/// is testable without the runtime.
fn inpaint_window(img_w: u32, img_h: u32, bbox: (u32, u32, u32, u32)) -> (u32, u32, u32, u32) {
    let (bx, by, bw, bh) = bbox;
    let side = bw
        .max(bh)
        .saturating_mul(2)
        .min(img_w.max(img_h))
        .max(bw.max(bh));
    let (w, h) = (side.min(img_w), side.min(img_h));
    let (cx, cy) = (bx + bw / 2, by + bh / 2);
    let x = cx.saturating_sub(w / 2).min(img_w - w);
    let y = cy.saturating_sub(h / 2).min(img_h - h);
    (x, y, w, h)
}

/// Composite a filled crop window back into the image:
/// pixels under the mask (at the image's own resolution)
/// come from the filled window, every other pixel is the
/// original, bit for bit. The model's output is a
/// regeneration of the whole window, plausible everywhere
/// but faithful nowhere, so only the hole may be taken
/// from it. The source's alpha, when it has one, is kept:
/// the model sees RGB only, and inpainting fills color,
/// not transparency.
pub fn composite_inpaint(
    img: &DynamicImage,
    mask: &GrayImage,
    window: (u32, u32, u32, u32),
    filled: &image::RgbImage,
) -> DynamicImage {
    let (wx, wy, ww, wh) = window;
    let mut rgb = img.to_rgb8();
    let fitted;
    let filled = if filled.width() == ww && filled.height() == wh {
        filled
    } else {
        fitted = image::imageops::resize(filled, ww, wh, FilterType::Triangle);
        &fitted
    };
    for ly in 0..wh {
        for lx in 0..ww {
            let (gx, gy) = (wx + lx, wy + ly);
            if mask.get_pixel(gx, gy)[0] > 0 {
                *rgb.get_pixel_mut(gx, gy) = *filled.get_pixel(lx, ly);
            }
        }
    }
    let out = DynamicImage::ImageRgb8(rgb);
    if matches!(
        img,
        DynamicImage::ImageRgba8(_) | DynamicImage::ImageLumaA8(_)
    ) {
        let src = img.to_rgba8();
        let mut rgba = out.to_rgba8();
        for (px, s) in rgba.pixels_mut().zip(src.pixels()) {
            px[3] = s[3];
        }
        return DynamicImage::ImageRgba8(rgba);
    }
    out
}

/// Erases the masked part of an image and inpaints it from
/// the surrounding pixels. Implemented by the ONNX provider
/// in production and by fakes in tests.
pub trait InpaintProvider: Send + Sync + std::fmt::Debug {
    /// Fill the mask's nonzero region of `img`. The mask is
    /// the image's size; nonzero is the hole. Pixels outside
    /// the hole come back unchanged.
    fn inpaint(&self, img: &DynamicImage, mask: &GrayImage) -> Result<DynamicImage, AiError>;
}

/// Region removal backed by a LaMa-family ONNX model
/// (issue #176): a two-input session (image plus mask) at
/// a fixed 512x512 working resolution, e.g. the OpenCV zoo
/// export of big-lama. The export's contract, verified
/// against the artifact: the image goes in as RGB in
/// 0..1, the mask as 0..1 with 1 the hole, and the output
/// comes back in 0..255 (the asymmetry is the export's,
/// not a typo).
pub struct OnnxInpaintProvider {
    session: Mutex<Session>,
    image_input: String,
    mask_input: String,
}

impl OnnxInpaintProvider {
    /// Load a model file. The two inputs are found by
    /// their conventional names (`image`, `mask`), falling
    /// back to declaration order; a model without two
    /// inputs is rejected here. The load probes the model
    /// with a small removal, like the upscale load probes
    /// its factor: a model that cannot run it at all has
    /// failed to load, not failed an inference.
    pub fn load(dylib: Option<&str>, model_path: &str) -> Result<Self, AiError> {
        let (session, _) = build_session(dylib, model_path)?;
        let names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        if names.len() < 2 {
            return Err(AiError::Load(format!(
                "the model {model_path} has {} input(s); an inpainting model \
                 takes an image and a mask",
                names.len()
            )));
        }
        let pick = |want: &str, fallback: usize| -> String {
            names
                .iter()
                .find(|n| n.eq_ignore_ascii_case(want))
                .cloned()
                .unwrap_or_else(|| names[fallback].clone())
        };
        let probe = Self {
            session: Mutex::new(session),
            image_input: pick("image", 0),
            mask_input: pick("mask", 1),
        };
        let img = DynamicImage::new_rgb8(64, 64);
        let mask = region_mask(64, 64, 24, 24, 16, 16);
        probe.inpaint(&img, &mask).map_err(|e| {
            AiError::Load(format!(
                "the model {model_path} failed its inpainting probe: {e}"
            ))
        })?;
        Ok(probe)
    }
}

impl std::fmt::Debug for OnnxInpaintProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxInpaintProvider")
            .field("image_input", &self.image_input)
            .field("mask_input", &self.mask_input)
            .finish_non_exhaustive()
    }
}

impl InpaintProvider for OnnxInpaintProvider {
    fn inpaint(&self, img: &DynamicImage, mask: &GrayImage) -> Result<DynamicImage, AiError> {
        let fitted;
        let mask = if mask.width() == img.width() && mask.height() == img.height() {
            mask
        } else {
            fitted = image::imageops::resize(mask, img.width(), img.height(), FilterType::Triangle);
            &fitted
        };
        let Some(bbox) = mask_bbox(mask) else {
            return Ok(img.clone());
        };
        let window = inpaint_window(img.width(), img.height(), bbox);
        let (wx, wy, ww, wh) = window;
        let size = INPAINT_INPUT_SIZE as usize;
        let crop = img.crop_imm(wx, wy, ww, wh).to_rgb8();
        let crop = image::imageops::resize(
            &crop,
            INPAINT_INPUT_SIZE,
            INPAINT_INPUT_SIZE,
            FilterType::Triangle,
        );
        let mask_crop = image::imageops::crop_imm(mask, wx, wy, ww, wh).to_image();
        let mask_crop = image::imageops::resize(
            &mask_crop,
            INPAINT_INPUT_SIZE,
            INPAINT_INPUT_SIZE,
            FilterType::Triangle,
        );
        let mut image_input = vec![0f32; 3 * size * size];
        for (i, px) in crop.pixels().enumerate() {
            for c in 0..3 {
                image_input[c * size * size + i] = px[c] as f32 / 255.0;
            }
        }
        let mut mask_input = vec![0f32; size * size];
        for (i, px) in mask_crop.pixels().enumerate() {
            mask_input[i] = px[0] as f32 / 255.0;
        }
        let image_tensor =
            ort::value::Tensor::from_array(([1, 3, size, size], image_input.into_boxed_slice()))
                .map_err(|e| AiError::Inference(format!("cannot build the image tensor: {e}")))?;
        let mask_tensor =
            ort::value::Tensor::from_array(([1, 1, size, size], mask_input.into_boxed_slice()))
                .map_err(|e| AiError::Inference(format!("cannot build the mask tensor: {e}")))?;
        let mut session = self.session.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![
                self.image_input.as_str() => image_tensor,
                self.mask_input.as_str() => mask_tensor
            ])
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
        // The export answers in 0..255 (see the type's
        // contract note), so the values clamp there, not
        // at 1.
        let mut filled = image::RgbImage::new(ow, oh);
        for (i, px) in filled.pixels_mut().enumerate() {
            for c in 0..3 {
                let v = data[c * (ow * oh) as usize + i];
                px[c] = v.clamp(0.0, 255.0).round() as u8;
            }
        }
        Ok(composite_inpaint(img, mask, window, &filled))
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

    #[test]
    fn region_mask_marks_only_the_rectangle() {
        let mask = region_mask(10, 8, 2, 3, 4, 2);
        assert_eq!(mask.get_pixel(2, 3)[0], 255);
        assert_eq!(mask.get_pixel(5, 4)[0], 255);
        assert_eq!(mask.get_pixel(1, 3)[0], 0);
        assert_eq!(mask.get_pixel(6, 3)[0], 0);
        assert_eq!(mask.get_pixel(2, 5)[0], 0);
        // A rectangle hanging off the edge clamps instead of
        // panicking: the engine bounds-checks first, but the
        // mask builder does not trust it to.
        let mask = region_mask(10, 8, 8, 6, 40, 40);
        assert_eq!(mask.get_pixel(9, 7)[0], 255);
    }

    #[test]
    fn mask_bbox_finds_the_marked_box() {
        assert_eq!(mask_bbox(&GrayImage::new(10, 10)), None);
        let mask = region_mask(40, 30, 7, 9, 5, 3);
        assert_eq!(mask_bbox(&mask), Some((7, 9, 5, 3)));
    }

    #[test]
    fn inpaint_window_centers_on_the_box_with_context() {
        // Twice the box's larger axis of context, centered.
        assert_eq!(inpaint_window(100, 80, (40, 30, 20, 20)), (30, 20, 40, 40));
        // A corner box: the window clamps to the edges.
        assert_eq!(inpaint_window(100, 80, (0, 0, 10, 10)), (0, 0, 20, 20));
        // A box against the far edges: the window slides
        // back inside.
        assert_eq!(inpaint_window(100, 80, (90, 70, 10, 10)), (80, 60, 20, 20));
        // A box covering the whole image: the window is the
        // image.
        assert_eq!(inpaint_window(100, 80, (0, 0, 100, 80)), (0, 0, 100, 80));
        // An image smaller than the square on one axis: the
        // window takes that whole axis.
        assert_eq!(inpaint_window(30, 20, (10, 5, 10, 10)), (5, 0, 20, 20));
    }

    #[test]
    fn composite_inpaint_takes_only_the_hole_from_the_fill() {
        let img = solid(200, 30, 30, 40, 30);
        let mask = region_mask(40, 30, 10, 10, 10, 10);
        let window = inpaint_window(40, 30, (10, 10, 10, 10));
        let filled = image::RgbImage::from_pixel(window.2, window.3, image::Rgb([30, 30, 200]));
        let out = composite_inpaint(&img, &mask, window, &filled).to_rgb8();
        // Inside the hole: the fill.
        assert_eq!(*out.get_pixel(15, 15), image::Rgb([30, 30, 200]));
        // Inside the window but outside the hole: the
        // original, bit for bit (the model regenerates the
        // whole window; only the hole may be believed).
        assert_eq!(*out.get_pixel(9, 9), image::Rgb([200, 30, 30]));
        // Outside the window: untouched.
        assert_eq!(*out.get_pixel(35, 25), image::Rgb([200, 30, 30]));
    }

    #[test]
    fn composite_inpaint_keeps_the_source_alpha() {
        let mut rgba = image::RgbaImage::new(20, 20);
        for p in rgba.pixels_mut() {
            *p = image::Rgba([10, 20, 30, 200]);
        }
        let img = DynamicImage::ImageRgba8(rgba);
        let mask = region_mask(20, 20, 5, 5, 10, 10);
        let window = inpaint_window(20, 20, (5, 5, 10, 10));
        let filled = image::RgbImage::from_pixel(window.2, window.3, image::Rgb([99, 99, 99]));
        let out = composite_inpaint(&img, &mask, window, &filled).to_rgba8();
        assert_eq!(*out.get_pixel(10, 10), image::Rgba([99, 99, 99, 200]));
        assert_eq!(*out.get_pixel(0, 0), image::Rgba([10, 20, 30, 200]));
    }

    /// Real-model verification, run manually with the deployment
    /// artifacts in place:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_INPAINT_MODEL=...
    /// cargo test -- --ignored inpaint_with_the_real_model
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn inpaint_with_the_real_model() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_INPAINT_MODEL").unwrap();
        let provider = OnnxInpaintProvider::load(Some(&dylib), &model).unwrap();
        // A smooth gradient with a solid red square pasted
        // in the middle: the model's one plausible answer
        // is the gradient, continued.
        let mut rgb = image::RgbImage::new(256, 256);
        for (x, y, p) in rgb.enumerate_pixels_mut() {
            *p = if (88..168).contains(&x) && (88..168).contains(&y) {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([x as u8, y as u8, 102])
            };
        }
        let img = DynamicImage::ImageRgb8(rgb);
        let mask = region_mask(256, 256, 88, 88, 80, 80);
        let out = provider.inpaint(&img, &mask).unwrap().to_rgb8();
        assert_eq!((out.width(), out.height()), (256, 256));
        let center = out.get_pixel(128, 128);
        // The fill on this geometry (the hole is half the
        // crop window, upscaled to the model's 512) measures
        // (97, 86, 79) against the true gradient's
        // (128, 128, 102): murkier than the ideal, but the
        // square is unambiguously gone and the fill is in
        // the gradient's family. The strict half of this
        // test is the bit-identity outside, below.
        assert!(
            (60..=180).contains(&center[0])
                && (60..=180).contains(&center[1])
                && (50..=150).contains(&center[2])
                && center[0] - center[1] < 45,
            "the red square should be gone, filled with the gradient: {center:?}"
        );
        // Outside the removal: the original, bit for bit.
        assert_eq!(*out.get_pixel(4, 4), image::Rgb([4, 4, 102]));
        assert_eq!(*out.get_pixel(250, 200), image::Rgb([250, 200, 102]));
        // An empty mask removes nothing and runs no fill.
        let out = provider
            .inpaint(&img, &GrayImage::new(256, 256))
            .unwrap()
            .to_rgb8();
        assert_eq!(*out.get_pixel(128, 128), image::Rgb([255, 0, 0]));
    }

    /// A one-input model is not an inpainting model: the
    /// load rejects it as Load, at boot, before any request
    /// can discover it. Ignored like the other load tests:
    /// load() initializes the runtime first.
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_IDENTITY_MODEL=...
    /// cargo test -- --ignored inpaint_load_rejects
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn inpaint_load_rejects_a_single_input_model() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_IDENTITY_MODEL").unwrap();
        let message = expect_load(OnnxInpaintProvider::load(Some(&dylib), &model));
        assert!(message.contains("takes an image and a mask"), "{message}");
    }

    /// A missing inpainting model file fails as Load, the
    /// same boot story as the other providers.
    /// CRABCROP_TEST_ORT_DYLIB=... cargo test -- --ignored inpaint_load_missing
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn inpaint_load_missing_model_file_is_a_load_error() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let message = expect_load(OnnxInpaintProvider::load(
            Some(&dylib),
            &missing_path("inpaint"),
        ));
        assert!(message.contains("cannot load the model"), "{message}");
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

    /// The image embedding the scores are computed
    /// from, L2-normalized, when the provider is an
    /// encoder (issue #172): meta surfaces it to
    /// clients that opt in with `embeddings=1`. The
    /// default is `None`: a provider that is not an
    /// encoder has no embedding to give.
    fn embedding(&self, img: &DynamicImage) -> Result<Option<Vec<f32>>, AiError> {
        let _ = img;
        Ok(None)
    }
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
    fn embedding(&self, img: &DynamicImage) -> Result<Option<Vec<f32>>, AiError> {
        Ok(Some(normalize(&self.embed_image(img)?)))
    }

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

/// The outcome of one embedding run (issue #172): the
/// configured encoder identity and its L2-normalized
/// image embedding. Presentation (the f16 packing on
/// the wire) is the caller's job, like score
/// composition (issue #152).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageEmbedding {
    /// The configured encoder identity reported in
    /// meta: the CLIP image model path. A vector is
    /// meaningless without the encoder that made it,
    /// so the identity is part of the value, not
    /// metadata about it.
    pub model: String,
    /// The embedding, L2-normalized (unit length), in
    /// the encoder's projection space.
    pub vector: Vec<f32>,
}

/// Produces image embeddings for meta responses that
/// opt in with `embeddings=1` (issue #172).
pub trait EmbeddingProvider: Send + Sync + std::fmt::Debug {
    fn embed(&self, img: &DynamicImage) -> Result<ImageEmbedding, AiError>;
}

/// Image embeddings from a CLIP tag provider: the
/// tagger's image encoder computes this vector on
/// every annotation (the cosine scores come from it)
/// and the annotation path discards it; this provider
/// surfaces it, wrapped with the encoder identity the
/// meta response reports. Wraps any `TagProvider`,
/// like the CLIP moderation fallback (issue #171).
pub struct ClipEmbeddingProvider {
    tagger: std::sync::Arc<dyn TagProvider>,
    model: String,
}

impl ClipEmbeddingProvider {
    /// Wrap a tag provider; `model` is the identity
    /// the meta response reports (the CLIP image model
    /// path).
    pub fn new(tagger: std::sync::Arc<dyn TagProvider>, model: String) -> Self {
        Self { tagger, model }
    }
}

impl std::fmt::Debug for ClipEmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClipEmbeddingProvider")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl EmbeddingProvider for ClipEmbeddingProvider {
    fn embed(&self, img: &DynamicImage) -> Result<ImageEmbedding, AiError> {
        match self.tagger.embedding(img)? {
            Some(vector) => Ok(ImageEmbedding {
                model: self.model.clone(),
                vector,
            }),
            None => Err(AiError::Inference(
                "the configured tag provider does not produce embeddings".to_string(),
            )),
        }
    }
}

/// A LAION-style linear aesthetic head (issue #172):
/// one weight per embedding dimension plus a bias,
/// scored as a dot product against the normalized
/// embedding. The weights file is the PyTorch save of
/// the head's state dict (the `sa_0_4_vit_*_linear.pth`
/// files LAION publishes): a zip archive holding a
/// small pickle and the raw float storages. Loading
/// parses exactly that shape and nothing else, so a
/// configured head fails at boot when malformed,
/// never at the first scored request. The score is a
/// descriptive signal only: it is not a quality gate,
/// and nobody should build one on it.
pub struct AestheticHead {
    weights: Vec<f32>,
    bias: f32,
    /// The configured path, reported in meta as the
    /// head's identity.
    identity: String,
}

impl AestheticHead {
    /// Load and validate a head file: unreadable,
    /// unparseable, or wrongly shaped files are load
    /// errors naming the file.
    pub fn load(path: &str) -> Result<Self, AiError> {
        let bytes = std::fs::read(path)
            .map_err(|e| AiError::Load(format!("cannot read the aesthetic head {path}: {e}")))?;
        // A linear head is a few kilobytes; the cap
        // keeps a wrong file (a full model save, say)
        // from being buffered and parsed as one.
        if bytes.len() > 16 * 1024 * 1024 {
            return Err(AiError::Load(format!(
                "the aesthetic head {path} is {} bytes: a linear head is a few kilobytes, not a model file",
                bytes.len()
            )));
        }
        let (weights, bias) = parse_aesthetic_head(&bytes)
            .map_err(|e| AiError::Load(format!("the aesthetic head {path} is not usable: {e}")))?;
        Self::from_weights(weights, bias, path.to_string())
    }

    /// Assemble a head from parsed weights: at least
    /// one weight, exactly one bias (the bias is a
    /// scalar by the time parsing is done).
    pub(crate) fn from_weights(
        weights: Vec<f32>,
        bias: f32,
        identity: String,
    ) -> Result<Self, AiError> {
        if weights.is_empty() {
            return Err(AiError::Load(
                "the aesthetic head has no weights".to_string(),
            ));
        }
        Ok(Self {
            weights,
            bias,
            identity,
        })
    }

    /// The head's identity, reported in meta: the
    /// configured path it was loaded from.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The embedding dimension the head expects.
    pub fn expected_dim(&self) -> usize {
        self.weights.len()
    }

    /// Score an embedding, or `None` when its
    /// dimension does not match the head's: a mismatch
    /// is a soft failure the caller reports (null
    /// score, a reason, expected vs got), never a
    /// wrong number computed from a truncated dot
    /// product.
    pub fn score(&self, embedding: &[f32]) -> Option<f32> {
        if embedding.len() != self.weights.len() {
            return None;
        }
        Some(
            self.weights
                .iter()
                .zip(embedding)
                .map(|(w, e)| w * e)
                .sum::<f32>()
                + self.bias,
        )
    }
}

impl std::fmt::Debug for AestheticHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AestheticHead")
            .field("dim", &self.weights.len())
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

/// Extract the (weights, bias) pair from a head
/// file's bytes: the zip entries, the pickle's state
/// dict, and the float storages it names. Every
/// deviation from the expected shape is an error
/// saying what was found instead.
fn parse_aesthetic_head(bytes: &[u8]) -> Result<(Vec<f32>, f32), String> {
    let entries = read_zip_entries(bytes)?;
    let entry = |name: &str| {
        entries
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.data.as_slice())
    };
    let pkl = entry("archive/data.pkl")
        .ok_or("the archive holds no archive/data.pkl: not a PyTorch save")?;
    let root = unpickle(pkl)?;
    let PickleValue::Map(pairs) = root else {
        return Err("the save's root is not a state dict".to_string());
    };
    let mut weight: Option<PickleTensor> = None;
    let mut bias: Option<PickleTensor> = None;
    for (key, value) in &pairs {
        let PickleValue::Tensor(tensor) = value else {
            return Err(format!(
                "the state dict entry {key:?} is not a tensor: only a single linear head (weight and bias tensors) is supported"
            ));
        };
        match key.as_str() {
            "weight" => weight = Some(tensor.clone()),
            "bias" => bias = Some(tensor.clone()),
            _ => {
                return Err(format!(
                    "the state dict holds an unexpected entry {key:?}: only a single linear head (weight and bias tensors) is supported"
                ));
            }
        }
    }
    let weight = weight.ok_or("the state dict holds no weight tensor")?;
    let bias = bias.ok_or("the state dict holds no bias tensor")?;
    let storage_floats = |tensor: &PickleTensor| -> Result<Vec<f32>, String> {
        let name = format!("archive/data/{}", tensor.storage);
        let data = entry(&name).ok_or_else(|| {
            format!("the archive holds no {name}: the tensor's storage is missing")
        })?;
        if data.len() % 4 != 0 {
            return Err(format!(
                "the storage {name} is {} bytes, not a whole number of float32 values",
                data.len()
            ));
        }
        let floats: Vec<f32> = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let count = tensor.size.iter().product::<usize>();
        let end = tensor
            .offset
            .checked_add(count)
            .ok_or("the tensor size overflows")?;
        if end > floats.len() {
            return Err(format!(
                "the tensor overruns its storage: offset {} plus {count} values against {} stored",
                tensor.offset,
                floats.len()
            ));
        }
        Ok(floats[tensor.offset..end].to_vec())
    };
    let weights = storage_floats(&weight)?;
    if weights.is_empty() {
        return Err("the weight tensor is empty".to_string());
    }
    let bias = storage_floats(&bias)?;
    if bias.len() != 1 {
        return Err(format!(
            "the bias tensor holds {} values: a linear head's bias is a single value",
            bias.len()
        ));
    }
    Ok((weights, bias[0]))
}

/// One entry of a zip archive, extracted.
struct ZipEntry {
    name: String,
    data: Vec<u8>,
}

/// Read every entry of a zip archive held in memory.
/// Deliberately minimal, sized for the few-kilobyte
/// saves `torch.save` writes for a linear head:
/// stored and deflated entries, no encryption, no
/// zip64, no multi-disk archives; anything else is
/// an error naming the limitation. Entry payloads are
/// capped as they inflate, so a hostile archive
/// cannot expand without bound.
fn read_zip_entries(bytes: &[u8]) -> Result<Vec<ZipEntry>, String> {
    let u16_at = |at: usize| -> Result<u16, String> {
        bytes
            .get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .ok_or_else(|| "the archive is truncated".to_string())
    };
    let u32_at = |at: usize| -> Result<u32, String> {
        bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| "the archive is truncated".to_string())
    };
    // The end-of-central-directory record closes the
    // file; scan the largest window a comment allows,
    // from the end.
    let window = bytes.len().saturating_sub(22 + u16::MAX as usize);
    let mut eocd = None;
    let mut at = bytes.len();
    while at > window {
        at -= 1;
        if bytes[at..].starts_with(b"PK\x05\x06") {
            eocd = Some(at);
            break;
        }
    }
    let eocd = eocd.ok_or("no end-of-central-directory record: not a zip archive")?;
    if u16_at(eocd + 4)? != 0 || u16_at(eocd + 6)? != 0 {
        return Err("multi-disk archives are not supported".to_string());
    }
    let count = u16_at(eocd + 10)? as usize;
    if count > 64 {
        return Err(format!(
            "the archive holds {count} entries: a head save holds a handful"
        ));
    }
    let mut dir = u32_at(eocd + 16)? as usize;
    let mut entries = Vec::new();
    for _ in 0..count {
        if !bytes
            .get(dir..)
            .is_some_and(|b| b.starts_with(b"PK\x01\x02"))
        {
            return Err("the central directory is malformed".to_string());
        }
        let method = u16_at(dir + 10)?;
        let compressed_len = u32_at(dir + 20)? as usize;
        let plain_len = u32_at(dir + 24)? as usize;
        let name_len = u16_at(dir + 28)? as usize;
        let extra_len = u16_at(dir + 30)? as usize;
        let comment_len = u16_at(dir + 32)? as usize;
        let local_at = u32_at(dir + 42)? as usize;
        if compressed_len == u32::MAX as usize || plain_len == u32::MAX as usize {
            return Err("zip64 archives are not supported".to_string());
        }
        let name_bytes = bytes
            .get(dir + 46..dir + 46 + name_len)
            .ok_or("the central directory is truncated")?;
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        if !bytes
            .get(local_at..)
            .is_some_and(|b| b.starts_with(b"PK\x03\x04"))
        {
            return Err(format!("the local header for {name} is malformed"));
        }
        let data_at =
            local_at + 30 + u16_at(local_at + 26)? as usize + u16_at(local_at + 28)? as usize;
        let raw = bytes
            .get(data_at..data_at + compressed_len)
            .ok_or("the archive is truncated")?;
        let data = match method {
            0 => raw.to_vec(),
            8 => {
                use std::io::Read;
                let mut out = Vec::new();
                flate2::read::DeflateDecoder::new(raw)
                    .take((plain_len + 1) as u64)
                    .read_to_end(&mut out)
                    .map_err(|e| format!("the entry {name} does not inflate: {e}"))?;
                out
            }
            _ => {
                return Err(format!(
                    "the entry {name} uses unsupported compression method {method}"
                ));
            }
        };
        if data.len() != plain_len {
            return Err(format!(
                "the entry {name} holds {} bytes against the directory's {plain_len}",
                data.len()
            ));
        }
        entries.push(ZipEntry { name, data });
        dir += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

/// A tensor reference out of the pickle machine: the
/// storage it reads from, the element offset into it,
/// and its shape. Only the flat element sequence is
/// consumed (in storage order), so the stride never
/// matters here.
#[derive(Debug, Clone, PartialEq)]
struct PickleTensor {
    storage: String,
    offset: usize,
    size: Vec<usize>,
}

/// A value on the restricted pickle machine's stack.
#[derive(Debug, Clone, PartialEq)]
enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// A global that has not been called: (module,
    /// name). Only a small allowlist is callable, and
    /// only through REDUCE.
    Global(String, String),
    Tuple(Vec<PickleValue>),
    /// A tensor storage, by the archive member key its
    /// bytes live under (`archive/data/<key>`).
    Storage(String),
    Tensor(PickleTensor),
    /// An insertion-ordered string-keyed map: the
    /// state dict itself, and plain dicts (the BUILD
    /// state) share the shape.
    Map(Vec<(String, PickleValue)>),
}

/// A restricted pickle interpreter, sized for the
/// state dicts `torch.save` writes for a linear head:
/// an ordered dict of tensor rebuilds over float
/// storages, with memo references. The opcode set is
/// an allowlist; anything else (a code-carrying
/// global, a reduce of an unknown callable, an
/// unexpected structure) is an error, so a hostile or
/// merely different save fails at load instead of
/// being half-parsed. This is a data reader, not a
/// general unpickler: it cannot construct arbitrary
/// objects and never touches the filesystem.
struct PickleMachine<'a> {
    bytes: &'a [u8],
    at: usize,
    stack: Vec<PickleValue>,
    marks: Vec<usize>,
    memo: Vec<Option<PickleValue>>,
}

impl<'a> PickleMachine<'a> {
    fn run(mut self) -> Result<PickleValue, String> {
        loop {
            let op = self.byte().ok_or("the pickle ends without a STOP")?;
            match op {
                // PROTO: the version byte is advisory;
                // the opcode allowlist below is the
                // real gate.
                0x80 => {
                    self.byte().ok_or("the pickle ends inside PROTO")?;
                }
                // GLOBAL: 'c' module '\n' name '\n'
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(PickleValue::Global(module, name));
                }
                // BINUNICODE / SHORT_BINUNICODE /
                // SHORT_BINSTRING: length-prefixed
                // strings (the state dict's keys and the
                // storage arguments).
                b'X' => {
                    let len = self.u32()? as usize;
                    let raw = self.take(len)?;
                    self.stack
                        .push(PickleValue::Str(String::from_utf8(raw.to_vec()).map_err(
                            |_| "a pickle string is not valid UTF-8".to_string(),
                        )?));
                }
                0x8c => {
                    let len = self.byte().ok_or("the pickle ends inside a short string")? as usize;
                    let raw = self.take(len)?;
                    self.stack
                        .push(PickleValue::Str(String::from_utf8(raw.to_vec()).map_err(
                            |_| "a pickle string is not valid UTF-8".to_string(),
                        )?));
                }
                b'U' => {
                    let len = self.byte().ok_or("the pickle ends inside a short string")? as usize;
                    let raw = self.take(len)?;
                    self.stack
                        .push(PickleValue::Str(String::from_utf8_lossy(raw).into_owned()));
                }
                // Integers: BININT1, BININT2, BININT,
                // LONG (a decimal line), LONG1/LONG4
                // (little-endian two's complement).
                b'K' => {
                    let v = self.byte().ok_or("the pickle ends inside BININT1")?;
                    self.stack.push(PickleValue::Int(v as i64));
                }
                b'M' => {
                    let v = self.u16()?;
                    self.stack.push(PickleValue::Int(v as i64));
                }
                b'J' => {
                    let v = self.u32()? as i32;
                    self.stack.push(PickleValue::Int(v as i64));
                }
                b'L' => {
                    let raw = self.line()?;
                    let digits = raw.strip_suffix('L').unwrap_or(&raw);
                    let v: i64 = digits
                        .parse()
                        .map_err(|_| format!("a pickle LONG is not a number: {raw:?}"))?;
                    self.stack.push(PickleValue::Int(v));
                }
                0x8a | 0x8b => {
                    let len = if op == 0x8a {
                        self.byte().ok_or("the pickle ends inside LONG1")? as usize
                    } else {
                        self.u32()? as usize
                    };
                    if len > 8 {
                        return Err("a pickle long is too large for a head file".to_string());
                    }
                    let raw = self.take(len)?;
                    let mut v: i64 = 0;
                    for (i, b) in raw.iter().enumerate() {
                        v |= (*b as i64) << (8 * i);
                    }
                    if len > 0 && len < 8 && raw[len - 1] & 0x80 != 0 {
                        v -= 1i64 << (8 * len);
                    }
                    self.stack.push(PickleValue::Int(v));
                }
                // BINFLOAT: big-endian f64.
                b'G' => {
                    let raw = self.take(8)?;
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(raw);
                    self.stack.push(PickleValue::Float(f64::from_be_bytes(buf)));
                }
                b'N' => self.stack.push(PickleValue::None),
                0x88 => self.stack.push(PickleValue::Bool(false)),
                0x89 => self.stack.push(PickleValue::Bool(true)),
                b')' => self.stack.push(PickleValue::Tuple(Vec::new())),
                b'(' => self.marks.push(self.stack.len()),
                b't' => {
                    let from = self.mark()?;
                    let items: Vec<PickleValue> = self.stack.drain(from..).collect();
                    self.stack.push(PickleValue::Tuple(items));
                }
                0x85..=0x87 => {
                    let n = (op - 0x84) as usize;
                    if self.stack.len() < n {
                        return Err("the pickle's stack underflows at a tuple".to_string());
                    }
                    let items: Vec<PickleValue> =
                        self.stack.drain(self.stack.len() - n..).collect();
                    self.stack.push(PickleValue::Tuple(items));
                }
                // BINPERSID: torch's persistent id for a
                // storage, a tuple of (the word
                // "storage", the storage class, the
                // archive key, the location, the element
                // count). Only the key is used.
                b'Q' => {
                    let value = self.pop()?;
                    let PickleValue::Tuple(parts) = &value else {
                        return Err("a pickle persistent id is not a tuple".to_string());
                    };
                    let [
                        PickleValue::Str(kind),
                        PickleValue::Global(_, class),
                        PickleValue::Str(key),
                        ..,
                    ] = parts.as_slice()
                    else {
                        return Err("a pickle persistent id does not name a storage".to_string());
                    };
                    if kind.as_str() != "storage" || !class.contains("Storage") {
                        return Err(format!(
                            "a pickle persistent id names {kind} {class}, not a storage"
                        ));
                    }
                    self.stack.push(PickleValue::Storage(key.clone()));
                }
                // REDUCE: call an allowlisted global.
                b'R' => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let PickleValue::Global(module, name) = callable else {
                        return Err("a pickle REDUCE target is not a global".to_string());
                    };
                    let PickleValue::Tuple(args) = args else {
                        return Err("a pickle REDUCE argument is not a tuple".to_string());
                    };
                    let value = reduce(&module, &name, args)?;
                    self.stack.push(value);
                }
                b'}' => self.stack.push(PickleValue::Map(Vec::new())),
                // SETITEM: value, then key, onto the map
                // below them.
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    let PickleValue::Str(key) = key else {
                        return Err("a pickle dict key is not a string".to_string());
                    };
                    let Some(PickleValue::Map(pairs)) = self.stack.last_mut() else {
                        return Err("a pickle SETITEM has no dict under it".to_string());
                    };
                    set_map_entry(pairs, key, value);
                }
                // SETITEMS: key/value pairs since the
                // mark, onto the map below it.
                b'u' => {
                    let from = self.mark()?;
                    let items: Vec<PickleValue> = self.stack.drain(from..).collect();
                    if !items.len().is_multiple_of(2) {
                        return Err("a pickle SETITEMS holds an odd item count".to_string());
                    }
                    let Some(PickleValue::Map(pairs)) = self.stack.last_mut() else {
                        return Err("a pickle SETITEMS has no dict under it".to_string());
                    };
                    let mut iter = items.into_iter();
                    while let (Some(k), Some(v)) = (iter.next(), iter.next()) {
                        let PickleValue::Str(key) = k else {
                            return Err("a pickle dict key is not a string".to_string());
                        };
                        set_map_entry(pairs, key, v);
                    }
                }
                // BUILD: the state (a metadata dict in
                // torch saves) is validated as data and
                // discarded; the built object stays.
                b'b' => {
                    let state = self.pop()?;
                    if !matches!(state, PickleValue::Map(_) | PickleValue::None) {
                        return Err("a pickle BUILD state is not a dict".to_string());
                    }
                    if self.stack.is_empty() {
                        return Err("a pickle BUILD has no object under it".to_string());
                    }
                }
                // BINPUT / LONG_BINPUT: memoize the
                // stack top.
                b'q' => {
                    let at = self.byte().ok_or("the pickle ends inside BINPUT")? as usize;
                    self.put_memo(at)?;
                }
                b'r' => {
                    let at = self.u32()? as usize;
                    self.put_memo(at)?;
                }
                // BINGET / LONG_BINGET.
                b'h' => {
                    let at = self.byte().ok_or("the pickle ends inside BINGET")? as usize;
                    self.get_memo(at)?;
                }
                b'j' => {
                    let at = self.u32()? as usize;
                    self.get_memo(at)?;
                }
                b'.' => {
                    if self.stack.len() != 1 {
                        return Err(format!(
                            "the pickle ends with {} values on the stack, not one",
                            self.stack.len()
                        ));
                    }
                    return Ok(self.stack.pop().unwrap());
                }
                _ => {
                    return Err(format!(
                        "unsupported pickle opcode 0x{op:02x}: not a head state dict shape this loader reads"
                    ));
                }
            }
        }
    }

    fn byte(&mut self) -> Option<u8> {
        let b = *self.bytes.get(self.at)?;
        self.at += 1;
        Some(b)
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let raw = self
            .bytes
            .get(self.at..self.at + len)
            .ok_or("the pickle is truncated")?;
        self.at += len;
        Ok(raw)
    }

    fn u16(&mut self) -> Result<u16, String> {
        let raw = self.take(2)?;
        Ok(u16::from_le_bytes([raw[0], raw[1]]))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let raw = self.take(4)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn line(&mut self) -> Result<String, String> {
        let start = self.at;
        let end = self
            .bytes
            .iter()
            .skip(start)
            .position(|b| *b == b'\n')
            .map(|p| start + p)
            .ok_or("the pickle ends inside a line")?;
        let raw = &self.bytes[start..end];
        self.at = end + 1;
        String::from_utf8(raw.to_vec()).map_err(|_| "a pickle line is not valid UTF-8".to_string())
    }

    fn pop(&mut self) -> Result<PickleValue, String> {
        self.stack
            .pop()
            .ok_or_else(|| "the pickle's stack underflows".to_string())
    }

    fn mark(&mut self) -> Result<usize, String> {
        self.marks
            .pop()
            .ok_or_else(|| "the pickle has a tuple or items opcode with no mark".to_string())
    }

    fn put_memo(&mut self, at: usize) -> Result<(), String> {
        if at > 4096 {
            return Err("a pickle memo index is implausibly large".to_string());
        }
        let value = self
            .stack
            .last()
            .cloned()
            .ok_or("a pickle memo stores an empty stack")?;
        if self.memo.len() <= at {
            self.memo.resize(at + 1, None);
        }
        self.memo[at] = Some(value);
        Ok(())
    }

    fn get_memo(&mut self, at: usize) -> Result<(), String> {
        let value = self
            .memo
            .get(at)
            .cloned()
            .flatten()
            .ok_or_else(|| format!("a pickle memo reference to {at} was never stored"))?;
        self.stack.push(value);
        Ok(())
    }
}

/// The REDUCE allowlist: build an ordered dict, or
/// rebuild a tensor over a storage. Anything else is
/// refused by name, so the failure says what the save
/// tried to do.
fn reduce(module: &str, name: &str, args: Vec<PickleValue>) -> Result<PickleValue, String> {
    match (module, name) {
        ("collections", "OrderedDict") => {
            if args.is_empty() {
                Ok(PickleValue::Map(Vec::new()))
            } else {
                Err("an OrderedDict reduce with arguments is not a state dict".to_string())
            }
        }
        ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
            let mut args = args.into_iter();
            let storage = match args.next() {
                Some(PickleValue::Storage(key)) => key,
                _ => return Err("a tensor rebuild does not start from a storage".to_string()),
            };
            let offset = match args.next() {
                Some(PickleValue::Int(v)) if v >= 0 => v as usize,
                _ => return Err("a tensor rebuild has no valid offset".to_string()),
            };
            let size = match args.next() {
                Some(PickleValue::Tuple(items)) => {
                    let mut size = Vec::with_capacity(items.len());
                    for item in items {
                        match item {
                            PickleValue::Int(v) if v >= 0 => size.push(v as usize),
                            _ => return Err("a tensor rebuild has an invalid size".to_string()),
                        }
                    }
                    size
                }
                _ => return Err("a tensor rebuild has no size".to_string()),
            };
            // The stride, the requires-grad flag, and
            // any later arguments are validated only
            // as data: the weights are read flat, in
            // storage order.
            Ok(PickleValue::Tensor(PickleTensor {
                storage,
                offset,
                size,
            }))
        }
        _ => Err(format!(
            "unsupported pickle callable {module}.{name}: not a head state dict shape this loader reads"
        )),
    }
}

/// Insert or replace one entry of a pickle map.
fn set_map_entry(pairs: &mut Vec<(String, PickleValue)>, key: String, value: PickleValue) {
    if let Some(slot) = pairs.iter_mut().find(|(k, _)| *k == key) {
        slot.1 = value;
    } else {
        pairs.push((key, value));
    }
}

/// Run the restricted pickle machine over a
/// `data.pkl` payload.
fn unpickle(bytes: &[u8]) -> Result<PickleValue, String> {
    if bytes.len() > 1024 * 1024 {
        return Err("the data.pkl is implausibly large for a head state dict".to_string());
    }
    PickleMachine {
        bytes,
        at: 0,
        stack: Vec::new(),
        marks: Vec::new(),
        memo: Vec::new(),
    }
    .run()
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
            ("hidden".to_string(), vec![1, 3], vec![9.0, 9.0, 9.0]),
            ("logits".to_string(), vec![1, 2], vec![1.5, -0.5]),
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
        assert!(
            err.to_string().contains("CRABCROP_MODERATION_LABELS"),
            "{err}"
        );
    }

    #[test]
    fn moderation_labels_must_match_the_output_count() {
        let named = vec!["safe".to_string(), "explicit".to_string()];
        let labels = resolve_moderation_labels(Some(named.clone()), 2, "model.onnx").unwrap();
        assert_eq!(labels, named);
        let err =
            resolve_moderation_labels(Some(vec!["only".to_string()]), 2, "model.onnx").unwrap_err();
        assert!(matches!(err, AiError::Load(_)), "{err}");
        assert!(
            err.to_string().contains("name one label per output"),
            "{err}"
        );
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
            report.scores.iter().all(|s| (0.0..=1.0).contains(&s.score)),
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
        let provider = OnnxModerationProvider::load(Some(&dylib), &model, Some(named)).unwrap();
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
        let message = expect_load(OnnxModerationProvider::load(Some(&dylib), &garbage, None));
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

#[cfg(test)]
mod embedding_tests {
    use super::*;

    fn solid() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            16,
            16,
            image::Rgb([120, 120, 120]),
        ))
    }

    fn temp_file(tag: &str, contents: &[u8]) -> String {
        let path = std::env::temp_dir().join(format!(
            "crabcrop-embedding-{tag}-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    fn missing_path(tag: &str) -> String {
        std::env::temp_dir()
            .join(format!(
                "crabcrop-embedding-missing-{tag}-{}.pth",
                std::process::id()
            ))
            .to_string_lossy()
            .into_owned()
    }

    fn expect_load<T>(result: Result<T, AiError>) -> String {
        match result {
            Err(AiError::Load(message)) => message,
            Err(other) => panic!("expected a load error, got {other}"),
            Ok(_) => panic!("expected a load error, but the load succeeded"),
        }
    }

    #[derive(Debug)]
    struct EmbeddingTagger(Result<Option<Vec<f32>>, AiError>);

    impl TagProvider for EmbeddingTagger {
        fn scores(&self, _img: &DynamicImage) -> Result<Vec<TagScore>, AiError> {
            Ok(Vec::new())
        }

        fn embedding(&self, _img: &DynamicImage) -> Result<Option<Vec<f32>>, AiError> {
            self.0.clone()
        }
    }

    #[derive(Debug)]
    struct ScoresOnlyTagger;

    impl TagProvider for ScoresOnlyTagger {
        fn scores(&self, _img: &DynamicImage) -> Result<Vec<TagScore>, AiError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn tag_provider_embedding_defaults_to_none() {
        // A provider that is not an encoder has no
        // embedding to give; the default says so
        // instead of failing.
        assert_eq!(ScoresOnlyTagger.embedding(&solid()).unwrap(), None);
    }

    #[test]
    fn clip_embedding_provider_surfaces_the_tagger_vector() {
        let tagger = EmbeddingTagger(Ok(Some(vec![0.6, 0.8])));
        let provider =
            ClipEmbeddingProvider::new(std::sync::Arc::new(tagger), "clip.onnx".to_string());
        let embedding = provider.embed(&solid()).unwrap();
        assert_eq!(embedding.model, "clip.onnx");
        assert_eq!(embedding.vector, vec![0.6, 0.8]);
    }

    #[test]
    fn clip_embedding_provider_reports_a_tagger_without_embeddings() {
        let tagger = EmbeddingTagger(Ok(None));
        let provider =
            ClipEmbeddingProvider::new(std::sync::Arc::new(tagger), "clip.onnx".to_string());
        let err = provider.embed(&solid()).unwrap_err();
        assert!(matches!(err, AiError::Inference(_)), "{err}");
        assert!(
            err.to_string().contains("does not produce embeddings"),
            "{err}"
        );
    }

    #[test]
    fn clip_embedding_provider_propagates_tagger_failures() {
        let tagger = EmbeddingTagger(Err(AiError::Inference("boom".to_string())));
        let provider =
            ClipEmbeddingProvider::new(std::sync::Arc::new(tagger), "clip.onnx".to_string());
        let err = provider.embed(&solid()).unwrap_err();
        assert!(matches!(err, AiError::Inference(_)), "{err}");
        assert!(err.to_string().contains("boom"), "{err}");
    }

    #[test]
    fn aesthetic_head_scores_a_matching_embedding() {
        let head = AestheticHead::from_weights(vec![0.5, -0.25, 2.0], 0.75, "head.pth".to_string())
            .unwrap();
        assert_eq!(head.expected_dim(), 3);
        assert_eq!(head.identity(), "head.pth");
        // 0.5*2 + (-0.25)*4 + 2*0.5 + 0.75 = 1.75.
        let score = head.score(&[2.0, 4.0, 0.5]).unwrap();
        assert!((score - 1.75).abs() < 1e-6, "score {score}");
    }

    #[test]
    fn aesthetic_head_dimension_mismatch_scores_none() {
        let head = AestheticHead::from_weights(vec![0.5, -0.25, 2.0], 0.75, "head.pth".to_string())
            .unwrap();
        assert_eq!(head.score(&[1.0, 2.0]), None);
        assert_eq!(head.score(&[1.0, 2.0, 3.0, 4.0]), None);
    }

    /// A pickle string (BINUNICODE) for the synthetic
    /// head saves below.
    fn pkl_str(out: &mut Vec<u8>, s: &str) {
        out.push(b'X');
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    /// The pickle `torch.save` writes for a linear
    /// head's state dict, reduced to the shape the
    /// loader reads: an ordered dict holding a weight
    /// tensor over storage "0" and a bias tensor over
    /// storage "1", closed by the metadata BUILD tail.
    /// Memo indices follow the real files.
    fn head_pickle(dim: usize, bias_count: usize) -> Vec<u8> {
        let mut p = Vec::new();
        let binint2 = |p: &mut Vec<u8>, v: usize| {
            p.push(b'M');
            p.extend_from_slice(&(v as u16).to_le_bytes());
        };
        // PROTO 2, then the root OrderedDict.
        p.extend_from_slice(b"\x80\x02");
        p.extend_from_slice(b"ccollections\nOrderedDict\n");
        p.extend_from_slice(b"q\x00)Rq\x01");
        p.push(b'(');
        // The weight entry.
        pkl_str(&mut p, "weight");
        p.extend_from_slice(b"q\x02");
        p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
        p.extend_from_slice(b"q\x03");
        p.push(b'(');
        p.push(b'(');
        pkl_str(&mut p, "storage");
        p.extend_from_slice(b"q\x04");
        p.extend_from_slice(b"ctorch\nFloatStorage\n");
        p.extend_from_slice(b"q\x05");
        pkl_str(&mut p, "0");
        p.extend_from_slice(b"q\x06");
        pkl_str(&mut p, "cpu");
        p.extend_from_slice(b"q\x07");
        binint2(&mut p, dim);
        p.extend_from_slice(b"tq\x08");
        p.push(b'Q'); // BINPERSID: the storage tuple
        p.extend_from_slice(b"K\x00"); // offset 0
        p.extend_from_slice(b"K\x01"); // size (1, dim)
        binint2(&mut p, dim);
        p.extend_from_slice(b"\x86q\x09");
        binint2(&mut p, dim); // stride (dim, 1)
        p.extend_from_slice(b"K\x01\x86q\x0a");
        p.extend_from_slice(b"\x89"); // requires_grad: false
        p.extend_from_slice(b"h\x00)"); // hooks: an empty OrderedDict, reduced next
        p.extend_from_slice(b"Rq\x0btq\x0cRq\x0d");
        // The bias entry, reusing the memoized globals.
        pkl_str(&mut p, "bias");
        p.extend_from_slice(b"q\x0e");
        p.extend_from_slice(b"h\x03");
        p.push(b'(');
        p.push(b'(');
        p.extend_from_slice(b"h\x04h\x05");
        pkl_str(&mut p, "1");
        p.extend_from_slice(b"q\x0f");
        p.extend_from_slice(b"h\x07");
        p.push(b'K');
        p.push(bias_count as u8);
        p.extend_from_slice(b"tq\x10");
        p.push(b'Q');
        p.extend_from_slice(b"K\x00"); // offset 0
        p.push(b'K');
        p.push(bias_count as u8); // size (bias_count,)
        p.extend_from_slice(b"\x85q\x11");
        p.extend_from_slice(b"K\x01"); // stride (1,)
        p.extend_from_slice(b"\x85q\x12");
        p.extend_from_slice(b"\x89");
        p.extend_from_slice(b"h\x00)");
        p.extend_from_slice(b"Rq\x13tq\x14Rq\x15");
        p.push(b'u'); // SETITEMS into the root dict
        // The BUILD tail: a state dict whose only
        // entry is an empty _metadata dict.
        p.push(b'}');
        p.extend_from_slice(b"q\x16");
        pkl_str(&mut p, "_metadata");
        p.extend_from_slice(b"q\x17");
        p.extend_from_slice(b"h\x00)Rq\x18");
        p.push(b's');
        p.push(b'b');
        p.push(b'.');
        p
    }

    /// Wrap a pickle and its storages in a stored-entry
    /// zip archive of the shape `torch.save` writes.
    fn head_archive(pkl: &[u8], storages: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let mut central: Vec<u8> = Vec::new();
        let mut entries: Vec<(&str, &[u8])> = vec![("archive/data.pkl", pkl)];
        entries.extend_from_slice(storages);
        entries.push(("archive/version", b"3\n"));
        for (name, data) in &entries {
            let local_at = out.len() as u32;
            out.extend_from_slice(b"PK\x03\x04");
            out.extend_from_slice(&20u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // flags
            out.extend_from_slice(&0u16.to_le_bytes()); // method: stored
            out.extend_from_slice(&0u32.to_le_bytes()); // time and date
            out.extend_from_slice(&0u32.to_le_bytes()); // crc (the reader ignores it)
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // extra
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);
            central.extend_from_slice(b"PK\x01\x02");
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u32.to_le_bytes());
            central.extend_from_slice(&0u32.to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u32.to_le_bytes());
            central.extend_from_slice(&local_at.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        let cd_at = out.len() as u32;
        let cd_len = central.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(b"PK\x05\x06");
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&cd_len.to_le_bytes());
        out.extend_from_slice(&cd_at.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    fn floats_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn aesthetic_head_loads_a_synthetic_pth() {
        let weights = [0.5f32, -0.25, 2.0];
        let archive = head_archive(
            &head_pickle(3, 1),
            &[
                ("archive/data/0", &floats_bytes(&weights)),
                ("archive/data/1", &floats_bytes(&[0.75])),
            ],
        );
        let path = temp_file("head-ok", &archive);
        let head = AestheticHead::load(&path).unwrap();
        assert_eq!(head.expected_dim(), 3);
        let score = head.score(&[2.0, 4.0, 0.5]).unwrap();
        assert!((score - 1.75).abs() < 1e-6, "score {score}");
    }

    #[test]
    fn aesthetic_head_load_rejects_garbage_and_missing_files() {
        let path = temp_file("head-garbage", b"this is not a zip archive at all");
        let err = expect_load(AestheticHead::load(&path));
        assert!(err.contains("not usable"), "{err}");
        let err = expect_load(AestheticHead::load(&missing_path("head")));
        assert!(err.contains("cannot read the aesthetic head"), "{err}");
    }

    #[test]
    fn aesthetic_head_rejects_an_archive_without_a_pickle() {
        let archive = head_archive(b"", &[]);
        // An empty pickle payload is not a state dict;
        // the failure must be a load error either way.
        let path = temp_file("head-empty-pkl", &archive);
        let err = expect_load(AestheticHead::load(&path));
        assert!(err.contains("not usable"), "{err}");
    }

    #[test]
    fn aesthetic_head_rejects_a_multi_value_bias() {
        let archive = head_archive(
            &head_pickle(3, 2),
            &[
                ("archive/data/0", &floats_bytes(&[0.5, -0.25, 2.0])),
                ("archive/data/1", &floats_bytes(&[0.75, 0.5])),
            ],
        );
        let path = temp_file("head-wide-bias", &archive);
        let err = expect_load(AestheticHead::load(&path));
        assert!(err.contains("bias is a single value"), "{err}");
    }

    #[test]
    fn aesthetic_head_rejects_a_storage_shorter_than_the_tensor() {
        let archive = head_archive(
            &head_pickle(3, 1),
            &[
                ("archive/data/0", &floats_bytes(&[0.5, -0.25])),
                ("archive/data/1", &floats_bytes(&[0.75])),
            ],
        );
        let path = temp_file("head-short-storage", &archive);
        let err = expect_load(AestheticHead::load(&path));
        assert!(err.contains("overruns its storage"), "{err}");
    }

    #[test]
    fn unpickle_refuses_an_unknown_callable_by_name() {
        // A reduce of os.system is the canonical
        // hostile pickle; the machine refuses it
        // before anything runs (nothing here can run:
        // the machine only builds data).
        let pkl = b"\x80\x02cos\nsystem\n)R.".to_vec();
        let err = unpickle(&pkl).unwrap_err();
        assert!(
            err.contains("unsupported pickle callable os.system"),
            "{err}"
        );
    }

    /// Real-model verification, run manually with the
    /// CLIP pair and the LAION B/32 head the spike used:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_CLIP_IMAGE_MODEL=...
    /// CRABCROP_TEST_CLIP_TEXT_MODEL=... CRABCROP_TEST_CLIP_TOKENIZER=...
    /// CRABCROP_TEST_CLIP_LABELS=... CRABCROP_TEST_AESTHETIC_HEAD=...
    /// cargo test -- --ignored embedding_and_aesthetic_with_the_real_models
    /// Set CRABCROP_TEST_AESTHETIC_IMAGE to also score
    /// that image (the spike's kite beach photo, 6.84).
    #[test]
    #[ignore = "needs the ONNX runtime, CLIP model files, and a head file (env-configured)"]
    fn embedding_and_aesthetic_with_the_real_models() {
        let var = |name: &str| std::env::var(name).unwrap();
        let image_model = var("CRABCROP_TEST_CLIP_IMAGE_MODEL");
        let tagger = OnnxClipProvider::load(
            Some(&var("CRABCROP_TEST_ORT_DYLIB")),
            &image_model,
            &var("CRABCROP_TEST_CLIP_TEXT_MODEL"),
            &var("CRABCROP_TEST_CLIP_TOKENIZER"),
            &var("CRABCROP_TEST_CLIP_LABELS"),
        )
        .unwrap();
        let provider = ClipEmbeddingProvider::new(std::sync::Arc::new(tagger), image_model.clone());
        let head = AestheticHead::load(&var("CRABCROP_TEST_AESTHETIC_HEAD")).unwrap();
        assert_eq!(head.expected_dim(), 512);
        let bytes = include_bytes!("../tests/fixtures/portrait.jpg");
        let img = image::load_from_memory(bytes).unwrap();
        let embedding = provider.embed(&img).unwrap();
        assert_eq!(embedding.model, image_model);
        assert_eq!(embedding.vector.len(), 512);
        let norm: f32 = embedding.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "norm {norm}");
        // The spike scored this fixture 7.03 through
        // the reference pipeline on the B/32 head; the
        // band leaves room for preprocessing wobble
        // between the two pipelines.
        let score = head.score(&embedding.vector).unwrap();
        assert!((6.5..=7.6).contains(&score), "portrait score {score}");
        if let Ok(path) = std::env::var("CRABCROP_TEST_AESTHETIC_IMAGE") {
            let img = image::open(&path).unwrap();
            let embedding = provider.embed(&img).unwrap();
            let score = head.score(&embedding.vector).unwrap();
            // The spike's kite beach score: 6.84.
            assert!((6.3..=7.4).contains(&score), "image score {score}");
        }
    }

    /// A head written for a different encoder loads
    /// fine: the dimension mismatch is a request-time
    /// soft failure, not a load error.
    /// CRABCROP_TEST_AESTHETIC_HEAD_L14=...
    /// cargo test -- --ignored aesthetic_head_for_another_encoder
    #[test]
    #[ignore = "needs a head file for another encoder (env-configured)"]
    fn aesthetic_head_for_another_encoder_loads_and_scores_none() {
        let head = AestheticHead::load(&std::env::var("CRABCROP_TEST_AESTHETIC_HEAD_L14").unwrap())
            .unwrap();
        assert_eq!(head.expected_dim(), 768);
        // A 512-dim embedding against the 768 head:
        // no score, and no panic.
        assert_eq!(head.score(&vec![0.03; 512]), None);
        assert!(head.score(&vec![0.03; 768]).is_some());
    }
}
