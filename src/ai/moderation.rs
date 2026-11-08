//! Moderation signals: the classifier provider and the
//! CLIP label-signal provider.

use image::DynamicImage;
use image::imageops::FilterType;
use ort::session::Session;
use std::sync::Mutex;

use super::clip::{crop224_resize_dims, first_row};
use super::session::{ModelOutput, build_session, collect_outputs, model_display_name};
use super::{AiError, ModerationProvider, ModerationScores, TagProvider, TagScore};

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
            model: model_display_name(model_path),
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
    /// meta response reports: the CLIP image model's
    /// file name (callers pass the configured path).
    pub fn new(tagger: std::sync::Arc<dyn TagProvider>, model: String) -> Self {
        Self {
            tagger,
            model: model_display_name(&model),
        }
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
pub(super) fn preprocess_vit(img: &DynamicImage) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let (nw, nh) = crop224_resize_dims(w, h);
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
mod tests;
