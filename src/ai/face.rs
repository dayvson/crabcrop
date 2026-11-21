//! Face detection: the YuNet provider, its grid decode, and
//! NMS.

use image::DynamicImage;
use ort::session::Session;
use std::sync::Mutex;

use super::memo::DetectionMemo;
use super::session::{build_session, fir_resize, lock_model, rgb8_view};
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
    /// The boxes of recent inputs (issue #677): a smart crop
    /// of the same source at another size asks for the same
    /// faces.
    memo: DetectionMemo<FaceBox>,
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
            memo: DetectionMemo::new(),
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
        let memo = Some(&self.memo);
        detect_faces(memo, img, |input| {
            let tensor = ort::value::Tensor::from_array((
                [1, 3, FACE_INPUT as usize, FACE_INPUT as usize],
                input.into_boxed_slice(),
            ))
            .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
            let mut session = lock_model(&self.session, "the face detection model's session");
            let outputs = session
                .run(ort::inputs![self.input_name.as_str() => tensor])
                .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
            let mut by_name: FaceOutputs = std::collections::HashMap::new();
            for (name, value) in outputs.into_iter() {
                let (_, data) = value.try_extract_tensor::<f32>().map_err(|e| {
                    AiError::BadOutput(format!("cannot read the model output: {e}"))
                })?;
                by_name.insert(name.to_string(), data.to_vec());
            }
            Ok(by_name)
        })
    }
}

/// YuNet's outputs, by name.
type FaceOutputs = std::collections::HashMap<String, Vec<f32>>;

/// One face detection around a model run: `run` takes
/// YuNet's input tensor and returns its outputs by name.
/// The provider passes its session; the tests pass a fake
/// model. With a memo, an input it knows is answered from
/// it and `run` is not called, and a result is remembered
/// only once the model has run and its output has decoded:
/// a failure is never remembered.
fn detect_faces(
    memo: Option<&DetectionMemo<FaceBox>>,
    img: &DynamicImage,
    run: impl FnOnce(Vec<f32>) -> Result<FaceOutputs, AiError>,
) -> Result<Vec<FaceBox>, AiError> {
    let (ow, oh) = (img.width(), img.height());
    if ow == 0 || oh == 0 {
        return Ok(Vec::new());
    }
    // The export takes exactly 640x640. The image is scaled
    // to fit (the reference wrapper resizes the same way);
    // boxes map back with independent axis scales.
    let small = fir_resize(&*rgb8_view(img), FACE_INPUT, FACE_INPUT);
    // Two things decide the boxes: the model's input, which
    // is these pixels in a fixed arrangement, and the source
    // size the boxes are mapped back to.
    let remembered = memo.map(|memo| {
        let key = memo.key(&[
            small.as_raw().as_slice(),
            &ow.to_le_bytes(),
            &oh.to_le_bytes(),
        ]);
        (memo, key)
    });
    if let Some((memo, key)) = &remembered
        && let Some(found) = memo.get(key)
    {
        return Ok(found);
    }
    let outputs = run(face_tensor(&small))?;
    let found = nms(decode_face_candidates(
        &outputs,
        ow,
        oh,
        FACE_SCORE_THRESHOLD,
    )?);
    if let Some((memo, key)) = remembered {
        memo.put(key, &found);
    }
    Ok(found)
}

/// YuNet's input for a frame at the model's size: BGR planar
/// floats at 0..255, the channel order and scale the model
/// was trained with. Each plane is written front to back.
fn face_tensor(small: &image::RgbImage) -> Vec<f32> {
    let plane = small.width() as usize * small.height() as usize;
    let mut input = vec![0f32; 3 * plane];
    let (blue, rest) = input.split_at_mut(plane);
    let (green, red) = rest.split_at_mut(plane);
    let (pixels, _) = small.as_raw().as_chunks::<3>();
    for (((px, b), g), r) in pixels.iter().zip(blue).zip(green).zip(red) {
        *b = f32::from(px[2]);
        *g = f32::from(px[1]);
        *r = f32::from(px[0]);
    }
    input
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
