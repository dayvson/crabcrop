//! In-process AI models (ONNX). The engine stays free of the
//! runtime: it depends only on the provider traits defined
//! here. The server builds the real providers at boot from
//! operator configuration (a model is preloaded when one is
//! configured, and a broken configuration fails startup).
//!
//! The plumbing is shared with AI upscale: a model is loaded
//! once against a process-wide ONNX environment, and inference
//! runs behind the server-wide concurrency gate.
//!
//! The provider traits and the types they exchange live
//! here; each provider family lives in its own submodule,
//! with the runtime plumbing they share in `session`. The
//! submodules' public items are re-exported, so callers
//! name everything as `crate::ai::<name>`.

use image::{DynamicImage, GrayImage};

mod checkpoint;
mod clip;
mod detect;
mod embedding;
mod face;
mod focal;
mod inpaint;
mod mask;
mod moderation;
mod ocr;
mod sam;
mod session;
mod upscale;

pub use clip::{OnnxClipProvider, read_labels};
pub use detect::OnnxDetectionProvider;
pub use embedding::{AestheticHead, ClipEmbeddingProvider};
pub use face::OnnxFaceProvider;
pub use focal::OnnxFocalProvider;
pub use inpaint::{INPAINT_INPUT_SIZE, OnnxInpaintProvider, composite_inpaint, region_mask};
pub use mask::{OnnxMaskProvider, apply_mask};
pub use moderation::{ClipModerationProvider, OnnxModerationProvider};
pub use ocr::{OCR_MAX_LINES, OnnxOcrProvider, read_ocr_dict};
pub use sam::{
    MaskGeometry, OnnxSegmentProvider, mask_geometry, segment_box_coords, segment_preprocess,
};
pub use upscale::OnnxUpscaleProvider;

/// How an AI operation failed, typed so callers can map the
/// failure class to a response instead of collapsing every
/// cause into one 500 (AP-16). The messages keep the wording
/// the plain strings carried; the variant is the diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AiError {
    /// A model, the runtime, or a model-adjacent file (labels,
    /// tokenizer data) could not be loaded. Raised at boot,
    /// where it fails startup, or while preparing a provider.
    Load(String),
    /// The runtime failed while running a model: building an
    /// input tensor, executing the session, a transient ORT
    /// error.
    Inference(String),
    /// A provider's session lock was poisoned: an earlier
    /// inference panicked while holding it, so the session's
    /// state can no longer be trusted.
    LockPoisoned,
    /// The model ran but its output is not what the provider
    /// contract requires: no outputs, an unreadable tensor,
    /// an unexpected shape, an embedding of the wrong size.
    BadOutput(String),
}

impl std::fmt::Display for AiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AiError::Load(cause) => write!(f, "{cause}"),
            AiError::Inference(cause) => write!(f, "{cause}"),
            AiError::LockPoisoned => write!(f, "a model session lock was poisoned"),
            AiError::BadOutput(cause) => write!(f, "{cause}"),
        }
    }
}

impl std::error::Error for AiError {}

/// Produces a foreground mask for an image: bright where the
/// subject is, dark where the background is. Implemented by the
/// ONNX provider in production and by fakes in tests.
pub trait MaskProvider: Send + Sync + std::fmt::Debug {
    fn mask(&self, img: &DynamicImage) -> Result<GrayImage, AiError>;

    /// The provider's confidence in the mask it produced for
    /// this image, as a probability-like score in 0..=1, when
    /// the provider's output supports one (issue #197: the
    /// studio record in meta reports it). The default is
    /// `None`, and the ONNX provider keeps it: its mask field
    /// is min-max normalized before it is returned, which is
    /// a contrast stretch, not a probability, so reporting a
    /// number would overstate what the model said.
    fn confidence(&self, _img: &DynamicImage, _mask: &GrayImage) -> Option<f32> {
        None
    }
}

/// Enlarges an image with a super-resolution model. The model
/// has a native factor; the engine composes runs and trims to
/// the exact requested size.
pub trait UpscaleProvider: Send + Sync + std::fmt::Debug {
    /// The model's native upscale factor (2, 3, 4...).
    fn factor(&self) -> u32;
    fn upscale(&self, img: &DynamicImage) -> Result<DynamicImage, AiError>;
}

/// Erases the masked part of an image and inpaints it from
/// the surrounding pixels. Implemented by the ONNX provider
/// in production and by fakes in tests.
pub trait InpaintProvider: Send + Sync + std::fmt::Debug {
    /// Fill the mask's nonzero region of `img`. The mask is
    /// the image's size; nonzero is the hole. Pixels outside
    /// the hole come back unchanged.
    fn inpaint(&self, img: &DynamicImage, mask: &GrayImage) -> Result<DynamicImage, AiError>;
}

/// A SAM-family image embedding: the encoder's output for
/// one image, opaque to callers. Segmentation decoders
/// consume it once per box prompt, so a request encodes
/// once and decodes per object (issue #175).
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentEmbedding {
    /// The embedding field in the SAM layout: 256 channels
    /// on a 64x64 grid, row-major.
    pub data: Vec<f32>,
}

/// Produces a per-object mask from a box prompt: the
/// silhouette of the object inside a detection's rectangle,
/// upgrading class operations from rectangles to the
/// object's own shape (issue #175). Implemented by the
/// ONNX provider in production and by fakes in tests.
pub trait SegmentProvider: Send + Sync + std::fmt::Debug {
    /// Encode an image once; the result feeds every
    /// `decode` for that image.
    fn encode(&self, img: &DynamicImage) -> Result<SegmentEmbedding, AiError>;
    /// Decode one box prompt against a previous `encode`
    /// of the same image. `bbox` is (x, y, w, h) in image
    /// pixels; the mask comes back at the box's own
    /// resolution, soft (strength 0..255), zero where the
    /// object is not.
    fn decode(
        &self,
        embedding: &SegmentEmbedding,
        img_w: u32,
        img_h: u32,
        bbox: (u32, u32, u32, u32),
    ) -> Result<GrayImage, AiError>;
    /// Encode and decode in one call, for callers with a
    /// single box to segment.
    fn segment(
        &self,
        img: &DynamicImage,
        bbox: (u32, u32, u32, u32),
    ) -> Result<GrayImage, AiError> {
        let embedding = self.encode(img)?;
        self.decode(&embedding, img.width(), img.height(), bbox)
    }
}

/// A detected face, in source-image pixels, with the
/// detector's confidence in 0.0..=1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub score: f32,
}

/// Detects faces in an image. Implemented by the ONNX provider
/// in production and by fakes in tests.
pub trait FaceProvider: Send + Sync + std::fmt::Debug {
    fn detect(&self, img: &DynamicImage) -> Result<Vec<FaceBox>, AiError>;
}

/// One label and its raw score from a tag provider: for
/// CLIP, the cosine similarity between the image embedding
/// and the label embedding. Presentation (softmax,
/// thresholding, capping, rounding) is the caller's job
/// (issue #152).
#[derive(Debug, Clone, PartialEq)]
pub struct TagScore {
    pub label: String,
    pub score: f32,
}

/// Scores images against a fixed label set.
pub trait TagProvider: Send + Sync + std::fmt::Debug {
    /// Raw scores for every configured label, in label
    /// order: no thresholding, capping, or rounding, so
    /// callers can tune presentation and debug borderline
    /// labels from the same numbers.
    fn scores(&self, img: &DynamicImage) -> Result<Vec<TagScore>, AiError>;

    /// The image embedding the scores are computed
    /// from, L2-normalized, when the provider is an
    /// encoder (issue #172): meta surfaces it to
    /// clients that opt in with `embeddings=1`. The
    /// default is `None`: a provider that is not an
    /// encoder has no embedding to give.
    fn embedding(&self, img: &DynamicImage) -> Result<Option<Vec<f32>>, AiError> {
        let _ = img;
        Ok(None)
    }
}

/// The outcome of one moderation run: the model that
/// produced it, its per-category verdict scores, and any
/// zero-shot review signals. A given provider fills
/// exactly one of the two lists: a classifier reports
/// verdicts, the CLIP fallback reports signals.
/// Presentation (rounding, ordering) is the caller's job,
/// like tag composition (issue #152).
#[derive(Debug, Clone, PartialEq)]
pub struct ModerationScores {
    /// The configured model identity reported in meta:
    /// the verdict model path, or the CLIP image model
    /// path when the fallback produced the signals.
    pub model: String,
    /// Verdict scores: per-category probabilities from
    /// the configured classifier, in the model's output
    /// order. Empty when the CLIP fallback ran.
    pub scores: Vec<TagScore>,
    /// Zero-shot review signals: cosine scores against
    /// operator prompt labels. Signals are leads for a
    /// human or a downstream review queue, never
    /// verdicts; the meta response marks them
    /// review-only. Empty when a classifier produced
    /// the verdicts.
    pub signals: Vec<TagScore>,
}

/// Scores images for content moderation (issue #171).
pub trait ModerationProvider: Send + Sync + std::fmt::Debug {
    fn moderate(&self, img: &DynamicImage) -> Result<ModerationScores, AiError>;
}

/// The outcome of one embedding run (issue #172): the
/// configured encoder identity and its L2-normalized
/// image embedding. Presentation (the f16 packing on
/// the wire) is the caller's job, like score
/// composition (issue #152).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageEmbedding {
    /// The configured encoder identity reported in
    /// meta: the CLIP image model's file name. A vector is
    /// meaningless without the encoder that made it,
    /// so the identity is part of the value, not
    /// metadata about it.
    pub model: String,
    /// The embedding, L2-normalized (unit length), in
    /// the encoder's projection space.
    pub vector: Vec<f32>,
}

/// Produces image embeddings for meta responses that
/// opt in with `embeddings=1` (issue #172).
pub trait EmbeddingProvider: Send + Sync + std::fmt::Debug {
    fn embed(&self, img: &DynamicImage) -> Result<ImageEmbedding, AiError>;
}

/// One detected object: a class label, its confidence,
/// and its box in image pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub label: String,
    pub score: f32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Detects objects in an image, in the shape of the face
/// provider: the classes the model knows, and the boxes it
/// finds.
pub trait DetectionProvider: Send + Sync + std::fmt::Debug {
    /// Whether the model's vocabulary contains `name`,
    /// case-insensitively, matching how class names are
    /// written in URLs.
    fn knows_class(&self, name: &str) -> bool;
    /// The model's class vocabulary, in model order, as an
    /// owned list: for boot-time reporting. Request paths
    /// use `knows_class`; no caller needs the slice itself
    /// (issue #152).
    fn describe(&self) -> Vec<String>;
    fn detect(&self, img: &DynamicImage) -> Result<Vec<Detection>, AiError>;
}

/// Supplies a focal point for smart crops, in the shape of
/// the other providers: implemented by the ONNX provider in
/// production and by fakes in tests. The point is a pair of
/// fractions of the image width and height, each in
/// 0.0..=1.0: the shape `crate::focus::FocalPoint` carries
/// through the cropper, without this module depending on it.
pub trait FocalProvider: Send + Sync + std::fmt::Debug {
    fn focal_point(&self, img: &DynamicImage) -> Result<(f32, f32), AiError>;
}

/// One line of text read by OCR (issue #177): the decoded
/// string, the mean confidence of the characters that
/// produced it, and its axis-aligned box in the image's
/// own pixels (the quadrilateral the detector found,
/// bounded).
#[derive(Debug, Clone, PartialEq)]
pub struct OcrLine {
    pub text: String,
    pub score: f32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// The outcome of one OCR run (issue #177): the model
/// identity and the lines, in reading order. The model
/// named is the recognition model: it is the half that
/// fixes the script the text is read in. The language is
/// the operator's declaration of the language that model
/// serves, echoed so the response describes itself; the
/// pipeline never guesses it. Presentation (score
/// rounding, the joined text content) is the caller's
/// job, like the other meta compositions (issue #152).
#[derive(Debug, Clone, PartialEq)]
pub struct OcrReport {
    pub model: String,
    pub language: String,
    pub lines: Vec<OcrLine>,
}

/// Extracts the text of an image: lines with boxes and
/// scores (issue #177). Implemented by the ONNX provider
/// in production and by fakes in tests.
pub trait OcrProvider: Send + Sync + std::fmt::Debug {
    fn recognize(&self, img: &DynamicImage) -> Result<OcrReport, AiError>;

    /// Recognize, stopping at `deadline`: the report
    /// carries the lines gathered before the deadline
    /// instead of running to completion. A page can hold
    /// hundreds of detected lines, each with its own
    /// recognition run, so the aggregate is the cost, not
    /// any single line. The default runs unbounded (the
    /// plain `recognize` contract); the ONNX provider
    /// checks the deadline between lines.
    fn recognize_deadline(
        &self,
        img: &DynamicImage,
        deadline: std::time::Instant,
    ) -> Result<OcrReport, AiError> {
        let _ = deadline;
        self.recognize(img)
    }
}

/// Reads one of the CRABCROP_TEST_* variables the real-model
/// tests take their model paths from. An unset variable means
/// that model is not provisioned in this environment, so the
/// caller early-returns and the test skips instead of
/// panicking: provisioning is a deliberate choice, and the
/// ai-models workflow sets only some of the variables. A set
/// variable is used as given: a non-UTF-8 value panics, and
/// a missing or invalid file it points at fails the test.
///
/// Not `cfg(test)`: the binary's own tests (`src/main.rs`, a
/// separate crate from this library) call it too.
#[doc(hidden)]
pub fn env_or_skip(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => {
            eprintln!("skipping: {name} is not set");
            None
        }
        Err(err) => panic!("{name}: {err}"),
    }
}

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
