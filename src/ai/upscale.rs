//! AI upscale: the super-resolution provider.

use image::{DynamicImage, ImageBuffer};
use ort::session::Session;
use std::sync::Mutex;

use super::depth::{Sample, SampleDepth, with_source_alpha};
use super::session::{TensorOutput, build_session, first_output};
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
        upscale_with(img, |input, dims| {
            let tensor = ort::value::Tensor::from_array((dims, input.into_boxed_slice()))
                .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
            let mut session = self.session.lock().map_err(|_| AiError::LockPoisoned)?;
            let outputs = session
                .run(ort::inputs![self.input_name.as_str() => tensor])
                .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
            first_output(outputs)
        })
    }
}

/// One upscale around a model run: `run` takes the planar
/// input tensor and its dims and returns the model's output.
/// The provider passes its session; the tests pass a fake
/// model. The source is read and the result written at the
/// source's own sample depth, with its alpha carried over
/// (issue #551).
fn upscale_with<O: TensorOutput>(
    img: &DynamicImage,
    run: impl FnOnce(Vec<f32>, [usize; 4]) -> Result<O, AiError>,
) -> Result<DynamicImage, AiError> {
    match SampleDepth::of(img) {
        SampleDepth::Eight => upscale_at::<u8, O>(img, run),
        SampleDepth::Sixteen => upscale_at::<u16, O>(img, run),
        SampleDepth::Float => upscale_at::<f32, O>(img, run),
    }
}

fn upscale_at<S: Sample, O: TensorOutput>(
    img: &DynamicImage,
    run: impl FnOnce(Vec<f32>, [usize; 4]) -> Result<O, AiError>,
) -> Result<DynamicImage, AiError> {
    let (w, h) = (img.width(), img.height());
    // Pad to a multiple of 8 with edge pixels replicated,
    // so the model's own window padding is a no-op and
    // its reshape agrees with its output (issue #326);
    // the result is cropped back below.
    let (pw, ph) = upscale_padded_dims(w, h);
    let input = padded_input::<S>(img, (pw, ph));
    let output = run(input, [1, 3, ph as usize, pw as usize])?;
    let (shape, data) = output.f32_tensor()?;
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
    let plane = (ow as usize)
        .checked_mul(oh as usize)
        .ok_or_else(|| AiError::BadOutput(format!("model output shape overflows: {shape:?}")))?;
    if data.len() < plane * 3 {
        return Err(AiError::BadOutput(format!(
            "model output has {} values, short of the {} its shape claims",
            data.len(),
            plane * 3
        )));
    }
    let mut out = ImageBuffer::<S::Rgb, Vec<S>>::new(ow, oh);
    samples_from_planes::<S>(&data[..plane * 3], plane, &mut out);
    if (pw, ph) != (w, h) {
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
        out = image::imageops::crop_imm(&out, 0, 0, crop_w, crop_h).to_image();
    }
    // The model sees color only; the source's alpha, at
    // any depth, is carried over with a plain resize so
    // transparency survives.
    Ok(with_source_alpha(img, out))
}

/// The model's planar input for a frame: its RGB samples as
/// values in 0..1, on a canvas of `(pw, ph)` whose columns
/// and rows past the frame repeat its last column and row.
/// Filled a row at a time.
fn padded_input<S: Sample>(img: &DynamicImage, (pw, ph): (u32, u32)) -> Vec<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let (pw, ph) = (pw as usize, ph as usize);
    let plane = pw * ph;
    let mut input = vec![0f32; 3 * plane];
    if w == 0 || h == 0 {
        return input;
    }
    let rgb = S::rgb(img);
    let raw = rgb.as_raw();
    let (red, rest) = input.split_at_mut(plane);
    let (green, blue) = rest.split_at_mut(plane);
    let rows = red
        .chunks_exact_mut(pw)
        .zip(green.chunks_exact_mut(pw))
        .zip(blue.chunks_exact_mut(pw));
    for (y, ((r, g), b)) in rows.enumerate() {
        let sy = y.min(h - 1);
        let src = &raw[sy * w * 3..(sy + 1) * w * 3];
        let (pixels, _) = src.as_chunks::<3>();
        for (((px, r), g), b) in pixels
            .iter()
            .zip(r.iter_mut())
            .zip(g.iter_mut())
            .zip(b.iter_mut())
        {
            *r = px[0].to_unit();
            *g = px[1].to_unit();
            *b = px[2].to_unit();
        }
        for plane_row in [r, g, b] {
            let last = plane_row[w - 1];
            plane_row[w..].fill(last);
        }
    }
    input
}

/// The model's three output planes as interleaved samples:
/// sample `c` of pixel `i` is `data[c * plane + i]`, clamped
/// and rounded by [`Sample::from_unit`]. A block of each
/// plane is converted at a time, so the conversion is a
/// loop over one slice, and the block is then interleaved.
fn samples_from_planes<S: Sample>(data: &[f32], plane: usize, out: &mut [S]) {
    const BLOCK: usize = 256;
    let (red, rest) = data.split_at(plane);
    let (green, blue) = rest.split_at(plane);
    let mut block = [[S::DEFAULT_MIN_VALUE; BLOCK]; 3];
    let blocks = out
        .chunks_mut(3 * BLOCK)
        .zip(red.chunks(BLOCK))
        .zip(green.chunks(BLOCK))
        .zip(blue.chunks(BLOCK));
    for (((pixels, r), g), b) in blocks {
        for (samples, values) in block.iter_mut().zip([r, g, b]) {
            for (sample, &value) in samples.iter_mut().zip(values) {
                *sample = S::from_unit(value);
            }
        }
        let [r, g, b] = &block;
        let (pixels, _) = pixels.as_chunks_mut::<3>();
        for (((px, &r), &g), &b) in pixels.iter_mut().zip(r).zip(g).zip(b) {
            *px = [r, g, b];
        }
    }
}

#[cfg(test)]
mod tests;
