//! Pipeline state (issue #345): the processed image
//! the pipeline produces and the records it carries
//! (video, PDF, studio-adjacent inpaint, segments,
//! objects crop), plus the process context threaded
//! through the operations.
//!
//! engine.rs re-exports these types.

use super::EngineError;
use crate::filters::WatermarkImages;
use crate::url::ImageFormat;

/// The encoded output plus its container format.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedImage {
    pub bytes: Vec<u8>,
    pub format: ImageFormat,
    /// The quality chosen for this output, set when the service
    /// picked it: `quality(auto)` content analysis, or a
    /// `max_bytes` reduction. `None` when the request named the
    /// quality (or the default stood).
    pub quality: Option<u8>,
    /// Output dimensions in pixels.
    pub width: u32,
    pub height: u32,
    /// Source dimensions in pixels, after EXIF orientation (the
    /// dimensions the pipeline treats as the source).
    pub source_width: u32,
    pub source_height: u32,
    /// Placeholder data (dominant color + tiny preview) for meta
    /// responses, computed from the final pixels when the caller
    /// asked for it. `None` otherwise.
    pub placeholder: Option<crate::placeholder::Placeholder>,
    /// The studio record (issue #197) when a `studio(...)`
    /// filter ran on the still pipeline: the resolved
    /// parameters, steps, and fallbacks, for meta responses.
    /// `None` when no studio filter ran (or on the animated
    /// paths, whose per-frame records are not surfaced).
    pub studio: Option<crate::studio::StudioRecord>,
    /// The region-removal record (issue #176) when a
    /// `remove_region(...)` filter ran on the still
    /// pipeline: the rectangles the inpainting model
    /// erased, for meta responses. `None` when no removal
    /// ran (or on the animated paths).
    pub inpaint: Option<InpaintRecord>,
    /// The segmentation record (issue #175) when a class
    /// operation ran with a segment provider configured on
    /// the still pipeline: the objects it segmented and
    /// their mask geometry, for meta responses. `None`
    /// when no class operation segmented anything (or on
    /// the animated paths).
    pub segments: Option<SegmentRecord>,
    /// The objects-crop record (issue #318) when an
    /// `objects` crop ran on the still pipeline: the
    /// classes it named, the detections it used, and the
    /// union box it framed (or the fallback it took), for
    /// meta responses. `None` when no objects crop ran
    /// (or on the animated paths).
    pub objects_crop: Option<ObjectsCropRecord>,
    /// The SSIMULACRA2 score of the returned bytes against the
    /// processed source, set when `quality(auto)` verified its
    /// encode (JPEG and JXL). `None` when nothing was scored.
    pub ssimulacra2: Option<f64>,
    /// The video record (issue #214) when the source was a
    /// video and a `frame:` segment selected the still this
    /// response was rendered from: the probed facts and the
    /// resolved timestamp, for meta responses. `None` for
    /// every image source.
    pub video: Option<VideoSourceMeta>,
    /// The PDF record (issue #215) when the source was a
    /// PDF and a `page:` segment selected the page this
    /// response was rendered from: the probed page count,
    /// the selected page, and the page size, for meta
    /// responses. `None` for every non-PDF source.
    pub pdf: Option<PdfSourceMeta>,
    /// The C2PA provenance facts (issue #173) for meta
    /// responses: what the source carried and what the
    /// output carries. The engine never sets this; the
    /// server fills it in around the signing step, which
    /// lives outside the engine (the manifest is embedded
    /// in the encoded bytes, after encoding). `None` unless
    /// the caller asked for the facts.
    pub provenance: Option<crate::provenance::ProvenanceFacts>,
}

/// Facts about a video source, probed at frame
/// extraction (issue #214), reported in meta responses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoSourceMeta {
    /// Container duration in seconds, when the container
    /// reports one.
    pub duration_seconds: Option<f64>,
    /// The video stream's coded width.
    pub width: u32,
    /// The video stream's coded height.
    pub height: u32,
    /// The resolved timestamp the frame was selected at,
    /// in seconds (after clamping to the last frame).
    pub frame_seconds: f64,
}

/// The region-removal record (issue #176) when one or
/// more `remove_region(...)` filters ran on the still
/// pipeline: the rectangles erased, in chain order, in
/// the pixel coordinates in effect where each filter ran,
/// for meta responses. `None` when no removal ran (or on
/// the animated paths, whose per-frame records are not
/// surfaced, like the studio record).
#[derive(Debug, Clone, PartialEq)]
pub struct InpaintRecord {
    /// The erased rectangles, each (x, y, w, h).
    pub regions: Vec<(u32, u32, u32, u32)>,
}

/// The segmentation record (issue #175): every object a
/// class operation segmented, in chain order, in the
/// pixel coordinates in effect where its filter ran, for
/// meta responses.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentRecord {
    /// The segmented objects.
    pub objects: Vec<SegmentObject>,
}

/// One segmented object in the segmentation record: the
/// detection it came from and the geometry of its mask,
/// the tighter statement of the object that crop
/// anchoring consumes (issue #175).
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentObject {
    /// The detection's class label, in the model's own
    /// spelling.
    pub label: String,
    /// The detection's score.
    pub score: f32,
    /// The detection's box, (x, y, w, h).
    pub detection: (u32, u32, u32, u32),
    /// The silhouette's tight box, (x, y, w, h) in image
    /// coordinates: where the object actually is, inside
    /// its detection box.
    pub tight_box: (u32, u32, u32, u32),
    /// The silhouette's soft area share of the detection
    /// box, 0..=1.
    pub coverage: f32,
    /// The silhouette's center of mass, in image
    /// coordinates.
    pub centroid: (u32, u32),
}

/// The objects-crop record (issue #318): what an `objects`
/// crop decided, for meta responses. Recorded wherever the
/// crop ran on the still pipeline, whichever path it took:
/// the classes it named, the detections it used, and the
/// union box it framed, or the fallback it took instead.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectsCropRecord {
    /// The classes the crop named, in request order, in
    /// the URL's own spellings.
    pub classes: Vec<String>,
    /// The detections the crop used (the named classes'
    /// boxes), in the pixel coordinates in effect where
    /// the crop ran. Empty on the fallback path.
    pub objects: Vec<crate::ai::Detection>,
    /// The union box the crop framed, (x, y, w, h) in the
    /// same coordinates; `None` on the fallback path.
    pub union: Option<(u32, u32, u32, u32)>,
    /// The scale factor the crop applied around the union
    /// box.
    pub scale: f32,
    /// Whether the crop fell back to the plain crop of the
    /// same size (no detection matched, or detection
    /// failed): the record states the path, never just the
    /// outcome's shape.
    pub fallback: bool,
}

/// Facts about a PDF source, probed at page rendering
/// (issue #215), reported in meta responses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PdfSourceMeta {
    /// The document's total page count.
    pub pages: u32,
    /// The selected page, 1-based (after the grammar's
    /// canonicalization).
    pub page: u32,
    /// The page's width in points.
    pub page_width: f64,
    /// The page's height in points.
    pub page_height: f64,
}

/// Per-request context the caller hands to the engine: the
/// things a pipeline run needs that are not in the URL.
#[derive(Clone)]
pub struct ProcessContext<'a> {
    /// Negotiated format preference (from `Accept`); applies
    /// only when the operations name no format themselves.
    pub preferred: Option<ImageFormat>,
    /// Watermark images resolved by the server, keyed by URL.
    pub watermarks: Option<&'a WatermarkImages>,
    /// Compute placeholder data from the final pixels.
    pub placeholders: bool,
    /// Background-removal provider, when the deployment
    /// configured a model.
    pub mask_provider: Option<&'a dyn crate::ai::MaskProvider>,
    /// Upscale provider, when the deployment configured a model.
    pub upscale_provider: Option<&'a dyn crate::ai::UpscaleProvider>,
    /// Inpainting provider for `remove_region`, when the
    /// deployment configured a model.
    pub inpaint_provider: Option<&'a dyn crate::ai::InpaintProvider>,
    /// Segment provider for mask-tight class operations
    /// (issue #175), when the deployment configured a
    /// model pair.
    pub segment_provider: Option<&'a dyn crate::ai::SegmentProvider>,
    /// Face provider for smart crops, when configured.
    pub face_provider: Option<&'a dyn crate::ai::FaceProvider>,
    /// Detection provider for class redaction, when
    /// configured.
    pub detect_provider: Option<&'a dyn crate::ai::DetectionProvider>,
    /// Declared detection providers (issue #166), in
    /// declaration order: class resolution walks the
    /// built-in provider first, then these.
    pub declared_detect_providers: &'a [&'a dyn crate::ai::DetectionProvider],
    /// Declared focal provider for smart crops (issue
    /// #166): the first `focal` declaration, consulted
    /// between the face provider and saliency.
    pub focal_provider: Option<&'a dyn crate::ai::FocalProvider>,
    /// Text renderer, when a font is configured.
    pub text_renderer: Option<&'a dyn crate::text::TextRenderer>,
    /// Video encoder, when the deployment configured an
    /// external encoder program (`CRABCROP_VIDEO_ENCODER`).
    pub video_encoder: Option<&'a crate::video::VideoEncoder>,
    /// Frame extractor, when the deployment configured an
    /// external decoder program (`CRABCROP_VIDEO_DECODER`):
    /// a `frame:` segment on a video source pulls its
    /// still through this (issue #214).
    pub frame_extractor: Option<&'a crate::frames::VideoFrameExtractor>,
    /// PDF renderer, when the deployment configured an
    /// external renderer program (`CRABCROP_PDF_RENDERER`):
    /// a `page:` segment on a PDF source renders its page
    /// through this (issue #215).
    pub pdf_renderer: Option<&'a crate::pdf::PdfRenderer>,
    /// DPR fallback from the request's client hint, used when
    /// the URL carries no explicit `dpr` directive. Advisory:
    /// a factor that pushes a size past the output limits
    /// steps down to the largest one that fits (issue #529).
    pub dpr: Option<f32>,
    /// SSIMULACRA2 target for verified `quality(auto)` encodes,
    /// from `CRABCROP_SSIM_TARGET`.
    pub ssim_target: f64,
    /// The client asked to save data (`Save-Data: on`):
    /// step the default encode quality down one notch.
    pub save_data: bool,
    /// The operator's metadata policy (`CRABCROP_METADATA`):
    /// copy the source's EXIF/XMP (and IPTC for JPEG) into
    /// JPEG and PNG outputs.
    pub preserve_metadata: bool,
    /// Operator brand color for `studio(...)` composites
    /// (issue #197), from `CRABCROP_BRAND_COLOR`. `None`
    /// means `bg:brand` and an unnamed backdrop resolve to
    /// white.
    pub brand_color: Option<[u8; 3]>,
    /// ravif speed tier for AVIF encodes (1 slowest to 10
    /// fastest), from `CRABCROP_AVIF_SPEED`.
    pub avif_speed: u8,
    /// jixel speed tier for JPEG XL encodes, from
    /// `CRABCROP_JXL_SPEED`.
    pub jxl_speed: jixel::Speed,
    /// Processing deadline for this request (issue #260),
    /// from `CRABCROP_PROCESS_TIMEOUT_SECS`: pipeline
    /// stages check it between operations and frames and
    /// abort with `EngineError::Timeout` once it passes,
    /// and the chunked encode loops (per frame for
    /// animated GIF and WebP, per attempt for the quality
    /// ladders) check it inside the encode as well (issue
    /// #322). A single opaque encoder call (one still
    /// encode, libwebp's animated `try_encode`, one AI
    /// inference, the external video encoder) has no
    /// checkpoint inside it and is never interrupted
    /// mid-call. `None` means no deadline (tests, library
    /// use).
    pub deadline: Option<std::time::Instant>,
}

impl ProcessContext<'_> {
    /// Fail with `EngineError::Timeout` when the request's
    /// processing deadline has passed.
    pub fn check_deadline(&self) -> Result<(), EngineError> {
        if let Some(deadline) = self.deadline
            && std::time::Instant::now() >= deadline
        {
            return Err(EngineError::Timeout);
        }
        Ok(())
    }
}
