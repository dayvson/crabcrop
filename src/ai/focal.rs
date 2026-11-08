//! Focal points: the heatmap provider and its shape
//! contract.

use image::DynamicImage;
use image::imageops::FilterType;
use ort::session::Session;
use std::sync::Mutex;

use super::session::{build_session, collect_outputs};
use super::{AiError, FocalProvider};

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
mod tests;
