//! Face detection: the YuNet provider, its grid decode, and
//! NMS.

use image::DynamicImage;
use ort::session::Session;
use std::sync::Mutex;

use super::session::{build_session, fir_resize_rgb};
use super::{AiError, FaceBox, FaceProvider};

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
    /// Load a model file against the runtime at `dylib` (or the
    /// runtime's own search when `None`). A probe detection on a
    /// blank image runs at load, like the other providers: a
    /// model that is not a YuNet export (a CLIP vision model is
    /// the realistic mix-up) parses as a session but has the
    /// wrong input shape and none of the YuNet heads, so it is
    /// rejected here, at boot, with a Load error, not as a 500
    /// on the first face request (issue #380).
    pub fn load(dylib: Option<&str>, model_path: &str) -> Result<Self, AiError> {
        let (session, input_name) = build_session(dylib, model_path)?;
        let provider = Self {
            session: Mutex::new(session),
            input_name,
        };
        provider
            .detect(&DynamicImage::new_rgb8(8, 8))
            .map_err(|e| {
                AiError::Load(format!("the model {model_path} failed its face probe: {e}"))
            })?;
        Ok(provider)
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
        let small = fir_resize_rgb(&img.to_rgb8(), FACE_INPUT, FACE_INPUT);
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
mod tests;
