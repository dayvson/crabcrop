//! Image embeddings (issue #172): the CLIP embedding
//! provider and the LAION aesthetic head scored on it.

use image::DynamicImage;

use super::checkpoint::{HEAD_BUDGET, parse_aesthetic_head};
use super::session::model_display_name;
use super::{AiError, EmbeddingProvider, ImageEmbedding, TagProvider};

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
    /// the meta response reports: the CLIP image
    /// model's file name (callers pass the configured
    /// path).
    pub fn new(tagger: std::sync::Arc<dyn TagProvider>, model: String) -> Self {
        Self {
            tagger,
            model: model_display_name(&model),
        }
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
    /// The head file's name, reported in meta as the
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
        if bytes.len() > HEAD_BUDGET {
            return Err(AiError::Load(format!(
                "the aesthetic head {path} is {} bytes: a linear head is a few kilobytes, not a model file",
                bytes.len()
            )));
        }
        let (weights, bias) = parse_aesthetic_head(&bytes)
            .map_err(|e| AiError::Load(format!("the aesthetic head {path} is not usable: {e}")))?;
        Self::from_weights(weights, bias, model_display_name(path))
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
    /// file name it was loaded from.
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

#[cfg(test)]
mod tests;
