//! AI upscale: the super-resolution provider.

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage};
use ort::session::Session;
use std::sync::Mutex;

use super::session::build_session;
use super::{AiError, UpscaleProvider};

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

/// Super-resolution exports in the Swin family run their
/// window attention on a grid of 8x8 windows and pad the
/// input up to a whole number of windows inside the graph.
/// In the exports this service loads, that internal padding
/// disagrees with a later reshape whenever a side is not
/// already a multiple of 8, and the run fails (issue #326).
const UPSCALE_PAD_MULTIPLE: u32 = 8;

/// The size an upscale input is padded to before inference:
/// both sides rounded up to a multiple of
/// [`UPSCALE_PAD_MULTIPLE`].
fn upscale_padded_dims(w: u32, h: u32) -> (u32, u32) {
    let round = |v: u32| v.div_ceil(UPSCALE_PAD_MULTIPLE) * UPSCALE_PAD_MULTIPLE;
    (round(w), round(h))
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
        let (w, h) = (img.width(), img.height());
        // Pad to a multiple of 8 with edge pixels replicated,
        // so the model's own window padding is a no-op and
        // its reshape agrees with its output (issue #326);
        // the result is cropped back below.
        let (pw, ph) = upscale_padded_dims(w, h);
        let rgb = img.to_rgb8();
        let raw = rgb.as_raw();
        let in_plane = pw as usize * ph as usize;
        let mut input = vec![0f32; 3 * in_plane];
        for y in 0..ph as usize {
            let sy = y.min(h.saturating_sub(1) as usize);
            for x in 0..pw as usize {
                let sx = x.min(w.saturating_sub(1) as usize);
                let src = (sy * w as usize + sx) * 3;
                let dst = y * pw as usize + x;
                for c in 0..3 {
                    input[c * in_plane + dst] = raw[src + c] as f32 / 255.0;
                }
            }
        }
        let tensor = ort::value::Tensor::from_array((
            [1, 3, ph as usize, pw as usize],
            input.into_boxed_slice(),
        ))
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
        // The claimed shape must fit the values actually
        // returned before anything indexes by it (issue
        // #267): a corrupt-but-loadable model can claim
        // dimensions its tensor does not carry.
        let plane = (ow as usize).checked_mul(oh as usize).ok_or_else(|| {
            AiError::BadOutput(format!("model output shape overflows: {shape:?}"))
        })?;
        if data.len() < plane * 3 {
            return Err(AiError::BadOutput(format!(
                "model output has {} values, short of the {} its shape claims",
                data.len(),
                plane * 3
            )));
        }
        let mut out = image::RgbImage::new(ow, oh);
        for (i, px) in out.pixels_mut().enumerate() {
            for c in 0..3 {
                let v = data[c * plane + i];
                px[c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        let mut out = DynamicImage::ImageRgb8(out);
        let (ow, oh) = if (pw, ph) != (w, h) {
            // The model enlarged the padded input; crop the
            // replicated-edge margin back off so the result
            // is exactly the factor times the source size
            // (issue #326).
            let f = if pw > 0 && ph > 0 && ow % pw == 0 && oh % ph == 0 && ow / pw == oh / ph {
                ow / pw
            } else {
                0
            };
            let (Some(crop_w), Some(crop_h)) = (w.checked_mul(f), h.checked_mul(f)) else {
                return Err(AiError::BadOutput(format!(
                    "unexpected model output shape for a padded input: {shape:?}"
                )));
            };
            if f < 1 || crop_w > ow || crop_h > oh {
                return Err(AiError::BadOutput(format!(
                    "unexpected model output shape for a padded input: {shape:?}"
                )));
            }
            out = out.crop_imm(0, 0, crop_w, crop_h);
            (crop_w, crop_h)
        } else {
            (ow, oh)
        };
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

#[cfg(test)]
mod tests;
