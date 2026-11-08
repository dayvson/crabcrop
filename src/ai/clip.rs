//! CLIP tagging: the image and text encoder pair, label
//! scoring, and the CLIP preprocessing.

use image::DynamicImage;
use image::imageops::FilterType;
use ort::session::Session;
use std::sync::Mutex;

use super::session::{ModelOutput, build_session, collect_outputs};
use super::{AiError, TagProvider, TagScore};

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
    /// The last image embedding computed, keyed by the exact
    /// preprocessed tensor that produced it: one meta request
    /// asks this provider for tag scores and for the embedding
    /// itself (and an aesthetic score off the same vector) as
    /// separate calls on the same image, and the vision run is
    /// the expensive half of both. The comparison is exact
    /// slice equality, so a hit returns precisely the vector
    /// a fresh run would: the model is a pure function of the
    /// tensor. A miss on any byte runs the model as before.
    last_embedding: Mutex<Option<(Vec<f32>, Vec<f32>)>>,
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
            last_embedding: Mutex::new(None),
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
        if let Some((tensor, embedding)) = self
            .last_embedding
            .lock()
            .map_err(|_| AiError::LockPoisoned)?
            .as_ref()
            && *tensor == input
        {
            return Ok(embedding.clone());
        }
        let tensor =
            ort::value::Tensor::from_array(([1, 3, 224, 224], input.clone().into_boxed_slice()))
                .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
        let mut session = self.vision.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![self.vision_input.as_str() => tensor])
            .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
        let outputs = collect_outputs(outputs)?;
        let embedding = pick_embedding(&outputs, None)?;
        *self
            .last_embedding
            .lock()
            .map_err(|_| AiError::LockPoisoned)? = Some((input, embedding.clone()));
        Ok(embedding)
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

pub(super) fn first_row(dims: &[i64], data: &[f32]) -> Vec<f32> {
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

/// The longest side the 224-crop preprocessors
/// (CLIP, ViT) may resize to (issue #359). Scaling the
/// short side to 224 leaves the long side unbounded, so
/// an extreme aspect input (16384x1) produced an
/// intermediate of hundreds of millions of pixels on the
/// way to a 224x224 crop. The cap bounds the intermediate
/// at 4096x224 (under a million pixels); aspects beyond
/// about 18:1 clamp the long side, the OCR recognition
/// clamp's shape, and only the center crop is consumed
/// either way.
const PREPROCESS_MAX_SIDE: u32 = 4096;

/// The resize dimensions for the 224-crop preprocessors:
/// the short side scales to 224, the long side follows
/// the aspect and clamps at [`PREPROCESS_MAX_SIDE`].
pub(super) fn crop224_resize_dims(w: u32, h: u32) -> (u32, u32) {
    let (w, h) = (w.max(1), h.max(1));
    let scale = 224.0 / w.min(h) as f32;
    let nw = ((w as f32 * scale).round() as u32).clamp(224, PREPROCESS_MAX_SIDE);
    let nh = ((h as f32 * scale).round() as u32).clamp(224, PREPROCESS_MAX_SIDE);
    (nw, nh)
}

/// CLIP image preprocessing, matching the export's
/// declared pipeline: resize the short side to 224
/// (bicubic), center crop 224x224, normalize with the
/// CLIP mean/std. Output is planar CHW.
#[allow(clippy::excessive_precision)] // the model's published constants
fn preprocess_clip(img: &DynamicImage) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let (nw, nh) = crop224_resize_dims(w, h);
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
mod tests;
