//! Object detection: the YOLO provider, letterboxing, and
//! its postprocessing.

use image::DynamicImage;
use ort::session::Session;
use std::sync::Mutex;

use super::clip::read_labels;
use super::session::{build_session, collect_outputs, fir_resize_rgb};
use super::{AiError, Detection, DetectionProvider};

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
    let resized = fir_resize_rgb(&img.to_rgb8(), nw, nh);
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
mod tests;
