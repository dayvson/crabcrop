//! Background removal: the U²-Net-family mask provider and
//! the mask application.

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage};
use ort::session::Session;
use std::sync::Mutex;

use super::session::{build_session, fir_resize};
use super::{AiError, MaskProvider};

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
    let small = fir_resize(&img.to_rgb8(), 320, 320);
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
        fir_resize(&mask, w, h)
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

#[cfg(test)]
mod tests;
