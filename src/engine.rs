//! The image engine: decode, transform, and encode.
//!
//! [`Engine::process`] takes source bytes and the parsed [`Operation`]s,
//! and returns the encoded output plus its format.

use std::io::Cursor;
use std::sync::Once;

use image::{DynamicImage, ExtendedColorType, ImageEncoder};

use crate::filters::{EncodeOptions, WatermarkImages};
use crate::quality::auto_quality;
use crate::url::{Align, Dimension, Filter, ImageFormat, ImageSize, Operation, PadFill, Quality};

/// Register the JPEG XL decoder with the image crate, once.
///
/// The image crate has no JPEG XL support of its own; the jxl-rs
/// integration hooks its format detection and decoder in globally.
fn ensure_jxl_decoder() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        jxl_image_rs_integration::register_image_decoding_hook();
    });
}

/// Shared resource limits for decoding and output sizing
/// (security audit E-1/E-2/E-3/E-4/E-9): one policy, applied at
/// every entry point, so a hostile source or an oversized
/// request fails cheaply instead of allocating gigabytes.
pub(crate) const MAX_DECODE_DIMENSION: u32 = 16_384;
pub(crate) const MAX_DECODE_ALLOC: u64 = 256 * 1024 * 1024;
const MAX_ANIMATED_FRAMES: usize = 200;
const MAX_ANIMATED_PIXELS: u64 = 100_000_000;
pub(crate) const MAX_OUTPUT_PIXELS: u64 = 100_000_000;

/// The most operations one request may execute, after preset
/// expansion (issue #182, audit R2-12): each operation counts
/// once, and each filter inside a `filters:` segment counts
/// once, because each is a pass over the pixels. Real chains
/// (a size, a crop, a dozen filters) sit far below this; the
/// cap exists so thousands of repeated cheap operations
/// cannot burn a process-gate slot on CPU alone.
const MAX_OPERATIONS: usize = 100;

/// The per-request encode budget (audit E-6): the
/// quality(auto) verification loop and the max_bytes ladder
/// draw from one pool, so their multipliers cannot stack
/// into a dozen-plus full-size encodes per cache miss.
const MAX_REQUEST_ENCODES: usize = 6;

/// The quality a `Save-Data` request steps down to: one
/// notch (10 points) below the default 80 the encoders and
/// the `max_bytes` ladder assume.
pub const SAVE_DATA_QUALITY: u8 = 70;

pub(crate) fn decode_limits() -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DECODE_DIMENSION);
    limits.max_image_height = Some(MAX_DECODE_DIMENSION);
    limits.max_alloc = Some(MAX_DECODE_ALLOC);
    limits
}

/// Decode an image from memory under the shared decode limits.
/// Every decode of untrusted bytes routes through here (the
/// main pipeline, the quality loop's re-decodes, watermark
/// images in the server), so no entry point can be bombed past
/// the budget. The declared dimensions are probed first and
/// checked in our own code, because decoder adherence to
/// `max_alloc` varies by format; a probe failure is not fatal
/// (some valid files have headers the probe cannot read), the
/// limited decode behind it is the second line.
pub(crate) fn decode_image(bytes: &[u8]) -> Result<DynamicImage, EngineError> {
    // AVIF decoding is compiled in only with the avif-decode
    // feature (issue #141): dav1d is a C dependency the
    // default build does not carry. Name the way out rather
    // than surfacing the image crate's bare unsupported
    // error. Feature builds fall through to the generic
    // bounded decode below, dav1d included.
    #[cfg(not(feature = "avif-decode"))]
    if matches!(image::guess_format(bytes), Ok(image::ImageFormat::Avif)) {
        return Err(EngineError::Decode(
            "AVIF sources cannot be decoded in this build (built without AVIF decoding: the Docker image and release binaries include it; source builds enable it with --features avif-decode)".to_string(),
        ));
    }
    let probe = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format();
    if let Ok(probe) = probe
        && let Ok((width, height)) = probe.into_dimensions()
    {
        check_decoded_dimensions(width as u64, height as u64)?;
        if width as u64 * height as u64 * 4 > MAX_DECODE_ALLOC {
            return Err(EngineError::Decode(format!(
                "image dimensions {width}x{height} exceed the decode allocation limit"
            )));
        }
    }
    let mut reader = image::ImageReader::new(Cursor::new(bytes));
    reader.limits(decode_limits());
    let reader = reader
        .with_guessed_format()
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    reader
        .decode()
        .map_err(|e| EngineError::Decode(e.to_string()))
}

/// The media kind an upload validated as (issue #192):
/// what the endpoint binds the object name's extension
/// against, so a validated image cannot be stored under
/// a name that presents it as something else. `Other` is
/// a format the generic decoder accepts but the binding
/// table does not enumerate (the PNM/TGA/QOI long tail);
/// those carry no extension binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UploadMedia {
    Svg,
    Heic,
    Jpeg,
    Png,
    Gif,
    WebP,
    Avif,
    Jxl,
    Bmp,
    Tiff,
    Ico,
    Other,
}

impl UploadMedia {
    /// The name extensions this media kind may be stored
    /// under (lowercase, no dot), or `None` when the kind
    /// carries no binding.
    pub(crate) fn allowed_extensions(self) -> Option<&'static [&'static str]> {
        match self {
            UploadMedia::Svg => Some(&["svg"]),
            UploadMedia::Heic => Some(&["heic", "heif"]),
            UploadMedia::Jpeg => Some(&["jpg", "jpeg"]),
            UploadMedia::Png => Some(&["png"]),
            UploadMedia::Gif => Some(&["gif"]),
            UploadMedia::WebP => Some(&["webp"]),
            UploadMedia::Avif => Some(&["avif"]),
            UploadMedia::Jxl => Some(&["jxl"]),
            UploadMedia::Bmp => Some(&["bmp"]),
            UploadMedia::Tiff => Some(&["tif", "tiff"]),
            UploadMedia::Ico => Some(&["ico"]),
            UploadMedia::Other => None,
        }
    }

    /// A short lowercase label for messages.
    pub(crate) fn label(self) -> &'static str {
        match self {
            UploadMedia::Svg => "svg",
            UploadMedia::Heic => "heic",
            UploadMedia::Jpeg => "jpeg",
            UploadMedia::Png => "png",
            UploadMedia::Gif => "gif",
            UploadMedia::WebP => "webp",
            UploadMedia::Avif => "avif",
            UploadMedia::Jxl => "jpeg xl",
            UploadMedia::Bmp => "bmp",
            UploadMedia::Tiff => "tiff",
            UploadMedia::Ico => "ico",
            UploadMedia::Other => "image",
        }
    }
}

/// An upload that passed validation (issue #192): the
/// bytes to store and the media kind they validated as.
/// For SVG the stored bytes are the sanitized document,
/// never the caller's original (audit R2-22); for every
/// other kind they are the caller's bytes verbatim, an
/// original being the point of the endpoint.
#[derive(Debug)]
pub(crate) struct PreparedUpload {
    pub stored: Vec<u8>,
    pub media: UploadMedia,
}

/// Validate that `bytes` are an image this server can
/// actually serve (issue #168): the upload endpoint's
/// acceptance test. This walks the same decode branches
/// `process_preferred` walks (SVG through the sanitizer,
/// HEIC through the oxideav stack, animated GIF and WebP
/// frame by frame under the animation caps, JPEG XL
/// through its header probe, everything else through the
/// bounded generic decode), so bytes accepted here are
/// bytes a transform request can process, and bytes the
/// pipeline would reject are rejected here, before they
/// are stored. The decoded pixels are discarded; only
/// decodability is the question. The upload handler
/// calls [`prepare_upload`] (it also needs the prepared
/// form); this predicate stays as the acceptance test's
/// own spelling.
#[allow(dead_code)]
pub(crate) fn validate_source_image(bytes: &[u8]) -> Result<(), EngineError> {
    prepare_upload(bytes).map(|_| ())
}

/// [`validate_source_image`] that also reports what to
/// store and what the bytes are (issue #192): the upload
/// handler stores the prepared bytes and binds the
/// object name's extension to the media kind.
pub(crate) fn prepare_upload(bytes: &[u8]) -> Result<PreparedUpload, EngineError> {
    ensure_jxl_decoder();
    if crate::svg::is_svg(bytes) {
        crate::svg::prepare(bytes).map_err(EngineError::Decode)?;
        let stored = crate::svg::sanitized_bytes(bytes).map_err(EngineError::Decode)?;
        return Ok(PreparedUpload {
            stored,
            media: UploadMedia::Svg,
        });
    }
    if is_heic(bytes) {
        crate::heic::decode_heic(bytes)?;
        return Ok(PreparedUpload {
            stored: bytes.to_vec(),
            media: UploadMedia::Heic,
        });
    }
    let detected = detect_format(bytes);
    if detected == ImageFormat::Gif {
        decode_gif_frames(bytes)?;
        return Ok(PreparedUpload {
            stored: bytes.to_vec(),
            media: UploadMedia::Gif,
        });
    }
    if detected == ImageFormat::WebP && webp_animation_params(bytes).is_some() {
        decode_webp_frames(bytes)?;
        return Ok(PreparedUpload {
            stored: bytes.to_vec(),
            media: UploadMedia::WebP,
        });
    }
    if detected == ImageFormat::Jxl {
        check_jxl_dimensions(bytes)?;
    }
    decode_image(bytes)?;
    // The generic decode accepted the bytes; name what it
    // accepted. The guess mirrors the decoder's own, with
    // JPEG XL (which the image crate does not know) taken
    // from the branch detection above.
    let media = if detected == ImageFormat::Jxl {
        UploadMedia::Jxl
    } else {
        match image::guess_format(bytes) {
            Ok(image::ImageFormat::Jpeg) => UploadMedia::Jpeg,
            Ok(image::ImageFormat::Png) => UploadMedia::Png,
            Ok(image::ImageFormat::Gif) => UploadMedia::Gif,
            Ok(image::ImageFormat::WebP) => UploadMedia::WebP,
            Ok(image::ImageFormat::Avif) => UploadMedia::Avif,
            Ok(image::ImageFormat::Bmp) => UploadMedia::Bmp,
            Ok(image::ImageFormat::Tiff) => UploadMedia::Tiff,
            Ok(image::ImageFormat::Ico) => UploadMedia::Ico,
            _ => UploadMedia::Other,
        }
    };
    Ok(PreparedUpload {
        stored: bytes.to_vec(),
        media,
    })
}

pub(crate) fn check_decoded_dimensions(width: u64, height: u64) -> Result<(), EngineError> {
    if width > MAX_DECODE_DIMENSION as u64 || height > MAX_DECODE_DIMENSION as u64 {
        return Err(EngineError::Decode(format!(
            "image dimensions {width}x{height} exceed the {MAX_DECODE_DIMENSION} px decode limit"
        )));
    }
    Ok(())
}

/// JPEG XL sources decode through the integration hook, whose
/// decoder ignores image-crate limits, so the declared size is
/// probed from the header and rejected over budget before any
/// full decode happens.
fn check_jxl_dimensions(bytes: &[u8]) -> Result<(), EngineError> {
    let (width, height) = jxl_dimensions(bytes)
        .ok_or_else(|| EngineError::Decode("JPEG XL header carried no size".to_string()))?;
    check_decoded_dimensions(width as u64, height as u64)
}

/// The dimensions in a JPEG XL header, from the decoder's
/// basic-info event alone: no frame is decoded. `None` when
/// the header does not parse or announces no size.
fn jxl_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    use jxl::api::{Event, JxlDecoder, JxlDecoderOptions};
    let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());
    let mut input = std::io::BufReader::new(Cursor::new(bytes));
    let event = decoder.process(&mut input, None, None).ok()?;
    if event != Event::BasicInfo {
        return None;
    }
    decoder
        .basic_info()
        .map(|info| (info.size.0 as u32, info.size.1 as u32))
}

/// The facts the `/explain` report (issue #196) may state
/// about a source: its media kind and its pixel dimensions,
/// both read from headers alone. Probing never decodes a
/// frame, never rasterizes an SVG, and never runs the HEIC
/// decoder: dimensions come from the image crate's header
/// probe, the SVG document's intrinsic size, the HEIC item
/// graph's announced size, or the JPEG XL basic info.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SourceProbe {
    /// A short lowercase media label: `jpeg`, `png`, `gif`,
    /// `webp`, `avif`, `jxl`, `heic`, `svg`, one of the image
    /// crate's long-tail kinds (`bmp`, `tiff`, `ico`), or
    /// `unknown` when the bytes match nothing the pipeline
    /// reads.
    pub kind: &'static str,
    /// Pixel dimensions when the header announces them.
    pub dimensions: Option<(u32, u32)>,
}

/// Probe source bytes for the explain report: kind and
/// dimensions, headers only (see [`SourceProbe`]).
pub(crate) fn probe_source(bytes: &[u8]) -> SourceProbe {
    if crate::svg::is_svg(bytes) {
        let dimensions = crate::svg::prepare(bytes).ok().and_then(|prepared| {
            let (w, h) = prepared.intrinsic();
            let (w, h) = (w.round() as u32, h.round() as u32);
            if w >= 1 && h >= 1 { Some((w, h)) } else { None }
        });
        return SourceProbe {
            kind: "svg",
            dimensions,
        };
    }
    if is_heic(bytes) {
        return SourceProbe {
            kind: "heic",
            dimensions: crate::heic::probe_dimensions(bytes),
        };
    }
    if is_jxl(bytes) {
        return SourceProbe {
            kind: "jxl",
            dimensions: jxl_dimensions(bytes),
        };
    }
    let kind = match image::guess_format(bytes) {
        Ok(image::ImageFormat::Jpeg) => "jpeg",
        Ok(image::ImageFormat::Png) => "png",
        Ok(image::ImageFormat::Gif) => "gif",
        Ok(image::ImageFormat::WebP) => "webp",
        Ok(image::ImageFormat::Avif) => "avif",
        Ok(image::ImageFormat::Bmp) => "bmp",
        Ok(image::ImageFormat::Tiff) => "tiff",
        Ok(image::ImageFormat::Ico) => "ico",
        // Not an image: a video container (issue #214)
        // gets its own kind. An AVIF still shares the
        // ISO-BMFF box but matched above, so it never
        // lands here. A video's dimensions live behind
        // the decoder's probe, which explain never runs,
        // so they stay unknown (the reader below cannot
        // read them either).
        _ if is_video_container(bytes) => "video",
        // Not an image either: a PDF document (issue
        // #215) gets its own kind. A page's dimensions
        // live behind the renderer's probe, which
        // explain never runs, so they stay unknown.
        _ if crate::pdf::sniff_pdf(bytes) => "pdf",
        _ => "unknown",
    };
    let dimensions = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()
        .and_then(|probe| probe.into_dimensions().ok());
    SourceProbe { kind, dimensions }
}

/// Reject an output size past the shared pixel budget, before
/// anything allocates it. u64 math, so DPR-scaled dimensions
/// cannot wrap past the check.
fn check_output_pixels(width: u32, height: u32) -> Result<(), EngineError> {
    if width as u64 * height as u64 > MAX_OUTPUT_PIXELS {
        return Err(EngineError::Unsupported(format!(
            "requested output {width}x{height} exceeds the {MAX_OUTPUT_PIXELS} pixel limit"
        )));
    }
    Ok(())
}

/// A running total of output pixels across the frames of one
/// animated or video output, charged against the same budget
/// a single still output gets (issue #182, audit R2-01). The
/// per-frame check alone lets 200 individually legal frames
/// retain hundreds of millions of pixels before any encoder
/// runs; charging each frame as it is produced rejects the
/// request once the total passes the budget, so what is
/// retained stays within the budget plus one frame.
struct OutputBudget {
    limit: u64,
    total: u64,
}

impl OutputBudget {
    fn new() -> Self {
        Self {
            limit: MAX_OUTPUT_PIXELS,
            total: 0,
        }
    }

    #[cfg(test)]
    fn with_limit(limit: u64) -> Self {
        Self { limit, total: 0 }
    }

    /// Charge one frame's output pixels against the budget.
    fn charge(&mut self, width: u32, height: u32) -> Result<(), EngineError> {
        self.total = self
            .total
            .saturating_add(u64::from(width) * u64::from(height));
        if self.total > self.limit {
            return Err(EngineError::Unsupported(format!(
                "animated output exceeds the {} pixel budget across frames",
                self.limit
            )));
        }
        Ok(())
    }
}

/// Reject a request whose expanded operation list runs past
/// [`MAX_OPERATIONS`]: the count the engine executes, not the
/// count the URL spells, so preset expansion cannot multiply
/// work past the cap (issue #182, audit R2-12).
fn check_operation_count(ops: &[Operation]) -> Result<(), EngineError> {
    let count: usize = ops
        .iter()
        .map(|op| match op {
            Operation::Filters(filters) => filters.len(),
            _ => 1,
        })
        .sum();
    if count > MAX_OPERATIONS {
        return Err(EngineError::Unsupported(format!(
            "request has {count} operations, over the {MAX_OPERATIONS} operation limit"
        )));
    }
    Ok(())
}

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

/// What went wrong while processing an image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// The source bytes were not a decodable image. Carries the cause.
    Decode(String),
    /// Encoding the output failed. Carries the cause.
    Encode(String),
    /// The operation is not supported yet. Carries what was asked for.
    Unsupported(String),
    /// The operation needs a model or resource this server
    /// was not configured with. Distinct from Unsupported:
    /// the URL is fine, the deployment cannot do it, so the
    /// server answers 501 rather than blaming the request.
    NotConfigured(String),
    /// An AI operation failed. Carries the typed cause
    /// (issue #148): the server maps BadOutput to 502 and
    /// the rest to 500.
    Inference(crate::ai::AiError),
    /// The external video encoder failed (issue #162):
    /// spawn failure, non-zero exit, timeout, output past
    /// the cap, or no output. The server maps it to 502:
    /// the URL was fine, the operator's encoder was not.
    Video(String),
    /// The external video decoder failed while extracting
    /// a frame (issue #214): spawn failure, non-zero exit,
    /// timeout, output past the cap, or no frame produced.
    /// The server maps it to 502, like the encode side:
    /// the URL was fine, the operator's decoder was not.
    Frame(String),
    /// The external PDF renderer failed while rendering
    /// a page (issue #215): spawn failure, non-zero exit,
    /// timeout, output past the cap, or output that is
    /// not a PNG. The server maps it to 502, exactly like
    /// the frame side: the URL was fine, the operator's
    /// renderer was not.
    Pdf(String),
    /// A text overlay could not be rendered. Carries the cause.
    Render(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Decode(cause) => {
                write!(f, "cannot decode source image: {cause}")
            }
            EngineError::Encode(cause) => {
                write!(f, "cannot encode output image: {cause}")
            }
            EngineError::Unsupported(what) => {
                write!(f, "unsupported operation: {what}")
            }
            EngineError::NotConfigured(what) => {
                write!(f, "not configured: {what}")
            }
            EngineError::Inference(cause) => {
                write!(f, "AI inference failed: {cause}")
            }
            EngineError::Video(cause) => {
                write!(f, "video encoder failed: {cause}")
            }
            EngineError::Frame(cause) => {
                write!(f, "video frame extraction failed: {cause}")
            }
            EngineError::Pdf(cause) => {
                write!(f, "PDF rendering failed: {cause}")
            }
            EngineError::Render(cause) => {
                write!(f, "cannot render text overlay: {cause}")
            }
        }
    }
}

impl std::error::Error for EngineError {}

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
    /// the URL carries no explicit `dpr` directive.
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
}

/// Default AVIF speed tier: the value the encoder ran at
/// before it became configurable.
pub const DEFAULT_AVIF_SPEED: u8 = 8;

/// Default JPEG XL speed tier: jixel's own default.
pub const DEFAULT_JXL_SPEED: jixel::Speed = jixel::Speed::Fast;

impl Default for ProcessContext<'_> {
    fn default() -> Self {
        ProcessContext {
            preferred: None,
            watermarks: None,
            placeholders: false,
            mask_provider: None,
            upscale_provider: None,
            face_provider: None,
            detect_provider: None,
            declared_detect_providers: &[],
            focal_provider: None,
            text_renderer: None,
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            dpr: None,
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            save_data: false,
            preserve_metadata: false,
            brand_color: None,
            avif_speed: DEFAULT_AVIF_SPEED,
            jxl_speed: DEFAULT_JXL_SPEED,
        }
    }
}

/// Decodes, resizes, and encodes images.
#[derive(Debug, Clone, Copy, Default)]
pub struct Engine;

impl Engine {
    /// Run the pipeline: decode, apply operations in order, encode.
    ///
    /// The output format is the detected source format unless a
    /// `format(...)` filter overrides it (JPEG when the source format
    /// cannot be detected and no filter overrides it). A `crop:W×H`
    /// crops to the W:H aspect ratio and then resizes to exactly W×H.
    ///
    /// The server calls [`Engine::process_preferred`] instead,
    /// because it negotiates format from `Accept`; this composed
    /// form is kept for tests and future library use.
    #[allow(dead_code)]
    pub fn process(&self, bytes: &[u8], ops: &[Operation]) -> Result<ProcessedImage, EngineError> {
        self.process_preferred(bytes, ops, ProcessContext::default())
    }

    /// Like [`Engine::process`], with the caller's context: a
    /// negotiated format preference, resolved watermark images,
    /// the placeholder switch, and the AI providers.
    pub fn process_preferred(
        &self,
        bytes: &[u8],
        ops: &[Operation],
        ctx: ProcessContext<'_>,
    ) -> Result<ProcessedImage, EngineError> {
        ensure_jxl_decoder();
        // One pass over the ops list settles every directive
        // the pipeline reads (issue #136); nothing below
        // re-walks the list looking for them.
        let directives = crate::url::RequestDirectives::analyze(ops);
        // Frame selection (issue #214): a `frame:` segment
        // makes the source a video whose selected frame
        // becomes the source image for everything below.
        // The extractor is required (a deployment without
        // one answers NotConfigured, like video output
        // without an encoder); an image source is refused
        // as a bad source, because frame selection is
        // video-only and animated GIF and WebP sources keep
        // their own handling; and the extracted PNG
        // replaces the bytes, so the format detection, the
        // validation preflights, the source metadata, and
        // the decode chain below all see the frame.
        let frame_png;
        let mut video_meta: Option<VideoSourceMeta> = None;
        let bytes = if let Some(selection) = directives.frame {
            let extractor = ctx.frame_extractor.ok_or_else(|| {
                EngineError::NotConfigured(
                    "frame extraction is not configured on this server (no CRABCROP_VIDEO_DECODER)"
                        .to_string(),
                )
            })?;
            if image::guess_format(bytes).is_ok()
                || is_jxl(bytes)
                || is_heic(bytes)
                || crate::svg::is_svg(bytes)
            {
                return Err(EngineError::Decode(
                    "frame selection needs a video source, but this source is an image".to_string(),
                ));
            }
            let frame = extractor
                .extract_frame(bytes, selection)
                .map_err(|err| match err {
                    crate::frames::FrameError::NotVideo(why) => EngineError::Decode(why),
                    crate::frames::FrameError::Unsupported(why) => EngineError::Unsupported(why),
                    crate::frames::FrameError::Failed(why) => EngineError::Frame(why),
                })?;
            video_meta = Some(VideoSourceMeta {
                duration_seconds: frame.duration,
                width: frame.width,
                height: frame.height,
                frame_seconds: frame.timestamp,
            });
            frame_png = frame.png;
            &frame_png
        } else {
            bytes
        };
        // Page selection (issue #215): a `page:` segment
        // makes the source a PDF whose selected page
        // becomes the source image for everything below,
        // the frame branch's shape exactly. The renderer
        // is required (a deployment without one answers
        // NotConfigured, like frame extraction without a
        // decoder); the engine sniffs before the renderer
        // spawns anything, so image bytes and non-PDF
        // bytes are refused as a bad source here and never
        // reach the tool; and the rendered PNG replaces
        // the bytes, so the format detection, the
        // validation preflights, and the decode chain
        // below all see the page.
        let page_png;
        let mut pdf_meta: Option<PdfSourceMeta> = None;
        let bytes = if let Some(page) = directives.page {
            let renderer = ctx.pdf_renderer.ok_or_else(|| {
                EngineError::NotConfigured(
                    "page rendering is not configured on this server (no CRABCROP_PDF_RENDERER)"
                        .to_string(),
                )
            })?;
            if image::guess_format(bytes).is_ok()
                || is_jxl(bytes)
                || is_heic(bytes)
                || crate::svg::is_svg(bytes)
                || !crate::pdf::sniff_pdf(bytes)
            {
                return Err(EngineError::Decode(
                    "page selection needs a PDF source, but this source is not a PDF".to_string(),
                ));
            }
            let rendered = renderer.render_page(bytes, page).map_err(|err| match err {
                crate::pdf::PdfError::NotPdf(why) => EngineError::Decode(why),
                crate::pdf::PdfError::Unsupported(why) => EngineError::Unsupported(why),
                crate::pdf::PdfError::Failed(why) => EngineError::Pdf(why),
            })?;
            pdf_meta = Some(PdfSourceMeta {
                pages: rendered.pages,
                page: rendered.page,
                page_width: rendered.page_width,
                page_height: rendered.page_height,
            });
            page_png = rendered.png;
            &page_png
        } else {
            bytes
        };
        // HEIC/HEIF sources are detected explicitly: they decode
        // through the pure-Rust oxideav stack (issue #163), in the
        // decode chain below, so the format gets its own metadata
        // path instead of the generic one.
        let heic_source = is_heic(bytes);
        // SVG sources are sniffed by content (the declared type
        // can lie), sanitized, and parsed up front: the raster
        // becomes the decoded image below, and the output
        // format defaults to PNG (SVG has no raster format).
        let prepared_svg = if crate::svg::is_svg(bytes) {
            Some(crate::svg::prepare(bytes).map_err(EngineError::Decode)?)
        } else {
            None
        };
        let detected = if prepared_svg.is_some() {
            ImageFormat::Png
        } else {
            detect_format(bytes)
        };
        // Background removal is validated before any real work:
        // its output is transparency, so the resolved format
        // must carry alpha, and a provider must be configured.
        if directives.removebg {
            let resolved = directives
                .explicit_format
                .or(ctx.preferred)
                .unwrap_or(detected);
            if matches!(resolved, ImageFormat::Jpeg | ImageFormat::Gif) {
                return Err(EngineError::Unsupported(
                    "removebg needs an alpha-capable output format (png, webp, jxl, avif)"
                        .to_string(),
                ));
            }
            if ctx.mask_provider.is_none() {
                return Err(EngineError::NotConfigured(
                    "removebg is not configured on this server".to_string(),
                ));
            }
        }
        // Studio mode is validated the same way (issue
        // #197): it runs the same mask model, so without a
        // provider it fails like removebg does, before any
        // real work. Its output needs no format check: the
        // composite is a normal image in the chain, valid
        // in every output format.
        if directives.studio && ctx.mask_provider.is_none() {
            return Err(EngineError::NotConfigured(
                "studio is not configured on this server (no mask model)".to_string(),
            ));
        }
        // Shape filters are validated the same way: their
        // output is transparency, so the resolved format must
        // carry alpha.
        if let Some(name) = directives.shape_filter {
            let resolved = directives
                .explicit_format
                .or(ctx.preferred)
                .unwrap_or(detected);
            if matches!(resolved, ImageFormat::Jpeg | ImageFormat::Gif) {
                return Err(EngineError::Unsupported(format!(
                    "{name} needs an alpha-capable output format (png, webp, jxl, avif)"
                )));
            }
        }
        if directives.upscale && ctx.upscale_provider.is_none() {
            return Err(EngineError::NotConfigured(
                "upscale is not configured on this server".to_string(),
            ));
        }
        // Video output is validated like the AI features:
        // before any real work, and distinctly. The URL is
        // fine but the deployment cannot do it (no encoder
        // configured), so this is NotConfigured (501), never
        // a silent fallback to an animated image format.
        if matches!(
            directives.explicit_format.or(ctx.preferred),
            Some(ImageFormat::Mp4 | ImageFormat::WebM)
        ) && ctx.video_encoder.is_none()
        {
            return Err(EngineError::NotConfigured(
                "video output is not configured on this server (no CRABCROP_VIDEO_ENCODER)"
                    .to_string(),
            ));
        }
        if directives.text && ctx.text_renderer.is_none() {
            return Err(EngineError::NotConfigured(
                "text is not configured on this server (no font)".to_string(),
            ));
        }
        if directives.needs_face_model && ctx.face_provider.is_none() {
            return Err(EngineError::NotConfigured(
                "face operations are not configured on this server (no face model)".to_string(),
            ));
        }
        if directives.has_class_redaction() {
            let providers = detection_providers(&ctx);
            if providers.is_empty() {
                return Err(EngineError::NotConfigured(
                    "class redaction is not configured on this server (no detection model)"
                        .to_string(),
                ));
            }
            for class in &directives.classes {
                if resolve_detector(&providers, class).is_none() {
                    return Err(EngineError::Unsupported(format!(
                        "unknown detection class {class:?}: not in the detection model's vocabulary"
                    )));
                }
            }
        }
        // Source metadata, read from the raw bytes once:
        // the ICC profile drives color conversion below and
        // (policy permitting) the metadata injection at the
        // end.
        // Replaced wholesale by the HEIC branch of the decode
        // chain, whose metadata comes from the decoder rather
        // than the container bytes.
        let mut source_meta = crate::color::extract_source_meta(bytes);
        // An animated WebP source (one carrying an ANIM chunk)
        // gets the GIF treatment below; the probe also yields
        // the loop count the animated path carries over.
        let webp_anim = if detected == ImageFormat::WebP {
            webp_animation_params(bytes)
        } else {
            None
        };
        let img = if detected == ImageFormat::Gif {
            // GIF sources decode frame by frame: an animated source
            // whose output stays GIF keeps its frames (each runs the
            // operations, delays preserved); anything else continues
            // with the first frame, which is also what converting an
            // animated GIF to a still format means.
            let frames = decode_gif_frames(bytes)?;
            if frames.len() > 1 && directives.debug_focal {
                return Err(EngineError::Unsupported(
                    "debug:focal renders a diagnostic image and cannot annotate an animated source"
                        .to_string(),
                ));
            }
            let mut resolved = directives
                .explicit_format
                .or(ctx.preferred)
                .unwrap_or(detected);
            if frames.len() > 1
                && resolved == ImageFormat::Jxl
                && directives.explicit_format.is_none()
            {
                // Animated sources stay out of JXL negotiation:
                // a negotiated JXL would silently drop the frames,
                // so the preference falls back to GIF. An explicit
                // format(jxl) still converts (first frame), as
                // documented for still conversions.
                resolved = ImageFormat::Gif;
            }
            if frames.len() > 1 && resolved == ImageFormat::Gif {
                return self.process_animated(frames, ops, &ctx, ImageFormat::Gif, None);
            }
            if frames.len() > 1 && resolved.is_video() {
                return self.process_video(frames, ops, &ctx, resolved);
            }
            DynamicImage::ImageRgba8(
                frames
                    .into_iter()
                    .next()
                    .expect("decode_gif_frames returns at least one frame")
                    .into_buffer(),
            )
        } else if let Some((loop_count, _)) = webp_anim {
            // Animated WebP sources decode frame by frame, like
            // GIFs: an animated output (WebP or GIF) keeps its
            // frames (each runs the operations, delays preserved);
            // anything else continues with the first frame, which
            // is also what converting an animated WebP to a still
            // format means.
            let frames = decode_webp_frames(bytes)?;
            if frames.len() > 1 && directives.debug_focal {
                return Err(EngineError::Unsupported(
                    "debug:focal renders a diagnostic image and cannot annotate an animated source"
                        .to_string(),
                ));
            }
            let mut resolved = directives
                .explicit_format
                .or(ctx.preferred)
                .unwrap_or(detected);
            if frames.len() > 1
                && resolved == ImageFormat::Jxl
                && directives.explicit_format.is_none()
            {
                // Animated sources stay out of JXL negotiation:
                // a negotiated JXL would silently drop the frames,
                // so the preference falls back to WebP. An explicit
                // format(jxl) still converts (first frame), as
                // documented for still conversions.
                resolved = ImageFormat::WebP;
            }
            if frames.len() > 1 && matches!(resolved, ImageFormat::WebP | ImageFormat::Gif) {
                return self.process_animated(frames, ops, &ctx, resolved, Some(loop_count));
            }
            if frames.len() > 1 && resolved.is_video() {
                return self.process_video(frames, ops, &ctx, resolved);
            }
            DynamicImage::ImageRgba8(
                frames
                    .into_iter()
                    .next()
                    .expect("decode_webp_frames returns at least one frame")
                    .into_buffer(),
            )
        } else if let Some(prep) = &prepared_svg {
            // Rasterize at the size the request is heading for
            // (or the intrinsic size when it asks for none), so
            // vectors stay crisp; the normal pipeline runs on
            // the raster from here.
            let (tw, th) =
                svg_raster_target(ops, prep.intrinsic(), dpr_factor(directives.dpr, &ctx));
            prep.render(tw, th).map_err(EngineError::Decode)?
        } else if heic_source {
            // HEIC/HEIF: the oxideav decode applies the EXIF
            // orientation from the decoder's own metadata (the
            // container bytes carry no orientation the generic
            // reader can see) and hands back the source metadata
            // the rest of the pipeline expects. Color management
            // then runs exactly as for the other formats.
            let (mut img, meta) = crate::heic::decode_heic(bytes)?;
            source_meta = meta;
            if !directives.keep_profile
                && let Some(icc) = &source_meta.icc
            {
                img = crate::color::convert_to_srgb(img, icc);
            }
            img
        } else {
            if detected == ImageFormat::Jxl {
                check_jxl_dimensions(bytes)?;
            }
            let mut img = decode_image(bytes)?;
            // Phone photos store rotation in the EXIF orientation
            // tag instead of the pixels. Honor it before any
            // operation, so sizes and crops see the image as the
            // viewer would. (GIFs carry no EXIF, hence the split.)
            if let Some(orientation) = exif_orientation(bytes) {
                img = apply_orientation(img, orientation);
            }
            // Color management: pixels arrive in the source
            // profile's space; convert to sRGB so every later
            // stage works in one known space, unless the
            // request asked to keep the profile (then the
            // profile is embedded at encode time instead).
            if !directives.keep_profile
                && let Some(icc) = &source_meta.icc
            {
                img = crate::color::convert_to_srgb(img, icc);
            }
            img
        };
        let mut format = detected;
        let (source_width, source_height) = (img.width(), img.height());
        // `no_upscale` is a request-wide directive, like the encoding
        // directives: wherever it appears, it caps every resize in
        // the request at the (oriented) source dimensions.
        let cap = if directives.no_upscale {
            Some((img.width(), img.height()))
        } else {
            None
        };
        let (img, encode_opts, studio) = apply_operations(
            img,
            ops,
            cap,
            dpr_factor(directives.dpr, &ctx),
            &ctx,
            &directives,
        )?;
        // Placeholders describe the final pixels, so they are
        // computed here, before encoding.
        let placeholder = if ctx.placeholders {
            Some(crate::placeholder::compute(&img))
        } else {
            None
        };
        if let Some(f) = encode_opts.format {
            format = f;
        } else if let Some(p) = ctx.preferred {
            format = p;
        }
        // A video format that reaches the still pipeline is
        // a still source asking for video: animated sources
        // returned through `process_video` above. There is
        // no first-frame fallback: the request named a video
        // and this source cannot become one.
        if format.is_video() {
            if directives.debug_focal {
                return Err(EngineError::Unsupported(format!(
                    "debug:focal renders a diagnostic image and cannot be served as \
                     format({}), which encodes video",
                    crate::url::format_text(format)
                )));
            }
            return Err(EngineError::Unsupported(format!(
                "format({}) encodes video and needs an animated source \
                 (an animated GIF or WebP); this source is a still image",
                crate::url::format_text(format)
            )));
        }
        // Resolve the quality directive against the final image.
        // `quality(auto)` is reported back: it is the one the caller
        // did not choose themselves.
        let (encode_quality, mut chosen_quality) = match encode_opts.quality {
            Some(Quality::Value(q)) => (Some(q), None),
            Some(Quality::Auto) => {
                let q = auto_quality(&img);
                (Some(q), Some(q))
            }
            // Save-Data steps the default quality down one
            // notch (80 to 70) on the formats that carry a
            // quality, and reports the step like an auto pick.
            // An explicit quality(...) of either kind stands.
            None if ctx.save_data
                && matches!(
                    format,
                    ImageFormat::Jpeg | ImageFormat::Jxl | ImageFormat::Avif | ImageFormat::WebP
                ) =>
            {
                (Some(SAVE_DATA_QUALITY), Some(SAVE_DATA_QUALITY))
            }
            None => (None, None),
        };
        // `quality(auto)` on a format with a decoder to verify
        // against (JPEG, JXL) closes the loop: encode, decode,
        // score with SSIMULACRA2, and bisect on quality until the
        // score lands in the target band. The heuristic pick is
        // the first guess. Other formats keep the one-shot pick.
        let perceptual = matches!(encode_opts.quality, Some(Quality::Auto))
            && matches!(format, ImageFormat::Jpeg | ImageFormat::Jxl);
        let mut ssim: Option<f64> = None;
        let enc_src = EncodeBuffers::new(&img);
        let mut encode_budget = MAX_REQUEST_ENCODES;
        let mut bytes = if perceptual {
            let first_guess = encode_quality.expect("auto quality resolves a value");
            let scored = crate::quality::encode_to_target(
                first_guess,
                ctx.ssim_target,
                &mut encode_budget,
                |q| {
                    let candidate = encode(
                        &enc_src,
                        format,
                        Some(q),
                        encode_opts.fill,
                        ctx.avif_speed,
                        ctx.jxl_speed,
                    )?;
                    let score = decode_image(&candidate)
                        .ok()
                        .and_then(|decoded| crate::quality::ssimulacra2(&img, &decoded));
                    Ok::<_, EngineError>((candidate, score))
                },
            )?;
            chosen_quality = Some(scored.quality);
            ssim = scored.score;
            scored.bytes
        } else {
            let t_timing = std::time::Instant::now();
            let bytes = encode(
                &enc_src,
                format,
                encode_quality,
                encode_opts.fill,
                ctx.avif_speed,
                ctx.jxl_speed,
            )?;
            crate::studio::timing_mark("encode", t_timing);
            encode_budget -= 1;
            bytes
        };
        // `max_bytes` is a budget, pursued by re-encoding at
        // decreasing quality. Only the quality-bearing formats have
        // a lever; for the others the directive cannot be honored
        // and the single encode stands. If quality bottoms out over
        // budget, the smallest attempt is returned: an image that
        // misses its budget beats an error. The ladder is bounded
        // per format so a budget cannot multiply an expensive
        // encode without limit, and the descent is planned against
        // the attempts that remain (see `next_quality`), so the
        // last allowed attempt always lands on the quality floor:
        // the walk used to stop wherever the encode pool ran out
        // (quality 15 from a 95 start), leaving tighter budgets
        // silently unchased (issue #229).
        let quality_bearing = matches!(
            format,
            ImageFormat::Jpeg | ImageFormat::Jxl | ImageFormat::Avif | ImageFormat::WebP
        );
        let mut ladder_ran = false;
        if let Some(budget) = encode_opts.max_bytes.filter(|_| quality_bearing) {
            let mut current = chosen_quality.or(encode_quality).unwrap_or(80);
            let mut attempts = 0;
            while bytes.len() as u64 > budget
                && current > 1
                && attempts < max_quality_attempts(format)
                && encode_budget > 0
            {
                let steps_remaining =
                    (max_quality_attempts(format) - attempts).min(encode_budget as u32);
                current = next_quality(current, steps_remaining);
                bytes = encode(
                    &enc_src,
                    format,
                    Some(current),
                    encode_opts.fill,
                    ctx.avif_speed,
                    ctx.jxl_speed,
                )?;
                encode_budget -= 1;
                chosen_quality = Some(current);
                attempts += 1;
                ladder_ran = true;
            }
        }
        if ladder_ran && ssim.is_some() && encode_budget > 0 {
            // The budget overruled the verified quality, so the
            // reported score must describe the bytes actually
            // served: re-score the final encode once.
            ssim = decode_image(&bytes)
                .ok()
                .and_then(|decoded| crate::quality::ssimulacra2(&img, &decoded));
        }
        // Metadata and profile embedding happen on the final
        // bytes: the ICC goes in when the request kept the
        // profile (it was not converted away), preserved
        // metadata when the operator policy allows it. Only
        // JPEG and PNG outputs carry either in this pass.
        let keep_profile = encode_opts.keep_profile;
        let bytes = if (keep_profile && source_meta.icc.is_some()) || ctx.preserve_metadata {
            let embed_icc = if keep_profile {
                source_meta.icc.as_deref()
            } else {
                None
            };
            match format {
                ImageFormat::Jpeg => crate::color::inject_jpeg(
                    &bytes,
                    embed_icc,
                    &source_meta,
                    ctx.preserve_metadata,
                ),
                ImageFormat::Png => {
                    crate::color::inject_png(&bytes, embed_icc, &source_meta, ctx.preserve_metadata)
                }
                _ => bytes,
            }
        } else {
            bytes
        };
        Ok(ProcessedImage {
            bytes,
            format,
            quality: chosen_quality,
            width: img.width(),
            height: img.height(),
            source_width,
            source_height,
            placeholder,
            studio,
            ssimulacra2: ssim,
            video: video_meta,
            pdf: pdf_meta,
        })
    }

    /// Process an animated source (GIF or WebP) frame by frame:
    /// every frame runs the same operations and keeps its delay,
    /// so the output stays animated in `format`. Encoding
    /// directives do not apply (neither animated encoder here
    /// has a quality lever). A WebP output carries the source's
    /// loop count over; GIF output keeps the GIF encoder's
    /// default, as it always has.
    fn process_animated(
        &self,
        frames: Vec<image::Frame>,
        ops: &[Operation],
        ctx: &ProcessContext<'_>,
        format: ImageFormat,
        loop_count: Option<u16>,
    ) -> Result<ProcessedImage, EngineError> {
        let directives = crate::url::RequestDirectives::analyze(ops);
        let (source_width, source_height) = {
            let b = frames[0].buffer();
            (b.width(), b.height())
        };
        let first = frames[0].buffer();
        let cap = if directives.no_upscale {
            Some((first.width(), first.height()))
        } else {
            None
        };
        let dpr = dpr_factor(directives.dpr, ctx);
        let mut budget = OutputBudget::new();
        let mut out_frames = Vec::with_capacity(frames.len());
        for frame in frames {
            let delay = frame.delay();
            let img = DynamicImage::ImageRgba8(frame.into_buffer());
            let (img, _, _) = apply_operations(img, ops, cap, dpr, ctx, &directives)?;
            budget.charge(img.width(), img.height())?;
            out_frames.push(image::Frame::from_parts(img.to_rgba8(), 0, 0, delay));
        }
        let (width, height) = out_frames
            .first()
            .map(|f| (f.buffer().width(), f.buffer().height()))
            .unwrap_or((source_width, source_height));
        // An animated output's placeholder comes from its first
        // processed frame.
        let placeholder = if ctx.placeholders {
            out_frames
                .first()
                .map(|f| crate::placeholder::compute(&DynamicImage::ImageRgba8(f.buffer().clone())))
        } else {
            None
        };
        let out = match format {
            ImageFormat::Gif => {
                let mut out = Vec::new();
                image::codecs::gif::GifEncoder::new(&mut out)
                    .encode_frames(out_frames)
                    .map_err(|e| EngineError::Encode(e.to_string()))?;
                out
            }
            ImageFormat::WebP => encode_animated_webp(out_frames, loop_count.unwrap_or(0))?,
            _ => unreachable!("process_animated is only called for GIF and WebP"),
        };
        Ok(ProcessedImage {
            bytes: out,
            format,
            quality: None,
            width,
            height,
            source_width,
            source_height,
            placeholder,
            studio: None,
            ssimulacra2: None,
            video: None,
            pdf: None,
        })
    }

    /// Process an animated source (GIF or WebP) into a
    /// video (issue #162): every frame runs the same
    /// operations as the animated image paths, then the
    /// operator-configured external encoder turns the frames
    /// into MP4 or WebM under the pipe contract in
    /// [`crate::video`]. Frame delays travel with the
    /// frames; the encoder side turns them into a constant
    /// frame rate (see `frame_schedule`). The reported
    /// dimensions are the encoder's: odd dimensions pad up
    /// to even for 4:2:0.
    fn process_video(
        &self,
        frames: Vec<image::Frame>,
        ops: &[Operation],
        ctx: &ProcessContext<'_>,
        format: ImageFormat,
    ) -> Result<ProcessedImage, EngineError> {
        let encoder = ctx.video_encoder.ok_or_else(|| {
            EngineError::NotConfigured(
                "video output is not configured on this server (no CRABCROP_VIDEO_ENCODER)"
                    .to_string(),
            )
        })?;
        let directives = crate::url::RequestDirectives::analyze(ops);
        let (source_width, source_height) = {
            let b = frames[0].buffer();
            (b.width(), b.height())
        };
        let first = frames[0].buffer();
        let cap = if directives.no_upscale {
            Some((first.width(), first.height()))
        } else {
            None
        };
        let dpr = dpr_factor(directives.dpr, ctx);
        let mut budget = OutputBudget::new();
        let mut out_frames = Vec::with_capacity(frames.len());
        for frame in frames {
            let delay_ms = std::time::Duration::from(frame.delay()).as_millis() as u64;
            let img = DynamicImage::ImageRgba8(frame.into_buffer());
            let (img, _, _) = apply_operations(img, ops, cap, dpr, ctx, &directives)?;
            budget.charge(img.width(), img.height())?;
            out_frames.push(crate::video::VideoFrame {
                pixels: img.to_rgba8(),
                delay_ms,
            });
        }
        // The placeholder (a meta-only surface; meta itself
        // refuses video URLs in the server) describes the
        // first processed frame, as the animated paths do.
        let placeholder = if ctx.placeholders {
            out_frames
                .first()
                .map(|f| crate::placeholder::compute(&DynamicImage::ImageRgba8(f.pixels.clone())))
        } else {
            None
        };
        let bytes = encoder.encode(&out_frames, format).map_err(|e| match e {
            crate::video::VideoError::Unsupported(what) => EngineError::Unsupported(what),
            crate::video::VideoError::Failed(what) => EngineError::Video(what),
        })?;
        let (width, height) = out_frames
            .first()
            .map(|f| (f.pixels.width(), f.pixels.height()))
            .unwrap_or((source_width, source_height));
        Ok(ProcessedImage {
            bytes,
            format,
            quality: None,
            width: crate::video::padded_dimension(width),
            height: crate::video::padded_dimension(height),
            source_width,
            source_height,
            placeholder,
            studio: None,
            ssimulacra2: None,
            video: None,
            pdf: None,
        })
    }
}

/// Collect an animation decoder's frames under the shared
/// limits: a frame cap and a total-decoded-pixels budget, so a
/// small file cannot expand into gigabytes of frames. Errors
/// when the source is malformed, over budget, or carries no
/// frames. `format` names the source format in error messages.
fn collect_frames(
    frames: impl Iterator<Item = Result<image::Frame, image::ImageError>>,
    format: &str,
) -> Result<Vec<image::Frame>, EngineError> {
    let mut collected = Vec::new();
    let mut total_pixels: u64 = 0;
    for frame in frames {
        let frame = frame.map_err(|e| EngineError::Decode(e.to_string()))?;
        if collected.len() >= MAX_ANIMATED_FRAMES {
            return Err(EngineError::Decode(format!(
                "{format} has more than {MAX_ANIMATED_FRAMES} frames"
            )));
        }
        total_pixels += frame.buffer().width() as u64 * frame.buffer().height() as u64;
        if total_pixels > MAX_ANIMATED_PIXELS {
            return Err(EngineError::Decode(format!(
                "{format} frames exceed the {MAX_ANIMATED_PIXELS} decoded-pixel limit"
            )));
        }
        collected.push(frame);
    }
    if collected.is_empty() {
        return Err(EngineError::Decode(format!("{format} has no frames")));
    }
    Ok(collected)
}

/// Decode all frames of a GIF source, under the shared limits:
/// the decoder gets explicit dimension/allocation limits (its
/// defaults are none) and collection enforces the shared
/// frame and pixel budgets (see [`collect_frames`]).
fn decode_gif_frames(bytes: &[u8]) -> Result<Vec<image::Frame>, EngineError> {
    use image::{AnimationDecoder, ImageDecoder};
    let mut decoder = image::codecs::gif::GifDecoder::new(Cursor::new(bytes))
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    decoder
        .set_limits(decode_limits())
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    collect_frames(decoder.into_frames(), "GIF")
}

/// The ANIM parameters of a WebP source: `(loop count,
/// background color as [R, G, B, A])`, or `None` when the
/// source carries no ANIM chunk (a still) or its container
/// does not parse; genuinely malformed files fail at decode
/// either way.
fn webp_animation_params(bytes: &[u8]) -> Option<(u16, [u8; 4])> {
    oxideav_webp::animation_params(bytes).ok().flatten()
}

/// Whether the source bytes are an animated WebP (they
/// carry an ANIM chunk). A container probe, not a decode:
/// the server uses it to route animated WebP renders to
/// their own gate (issue #231) before any engine work runs.
pub(crate) fn is_animated_webp(bytes: &[u8]) -> bool {
    webp_animation_params(bytes).is_some()
}

/// Decode all frames of an animated WebP source, under the
/// same limits as GIF. The in-tree decoder (image-webp)
/// composites every frame onto the full canvas, honoring
/// sub-rectangle frames and the blending flag; disposal of
/// a frame to the background color is honored only when the
/// file's ANIM background color has been set on the decoder
/// first (the decoder otherwise leaves the disposed rectangle
/// in place), so the color is read from the container and set
/// before iterating.
fn decode_webp_frames(bytes: &[u8]) -> Result<Vec<image::Frame>, EngineError> {
    use image::{AnimationDecoder, ImageDecoder};
    let mut decoder = image::codecs::webp::WebPDecoder::new(Cursor::new(bytes))
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    decoder
        .set_limits(decode_limits())
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    if let Some((_, background)) = webp_animation_params(bytes) {
        // Losing the fill on a decoder refusal only costs the
        // disposal behavior; the frames still decode.
        let _ = decoder.set_background_color(image::Rgba(background));
    }
    collect_frames(decoder.into_frames(), "WebP")
}

/// Encode frames as an animated WebP. The pure-Rust encoder
/// (oxideav-webp, the crate the lossy still path uses) writes
/// lossless VP8L frames and has no lossy animation mode, so
/// there is no quality lever here; `loop_count` is carried
/// over from the source (0 loops forever). The frames are
/// consumed so each pixel buffer moves into the encoder
/// instead of being cloned a second time (issue #182).
fn encode_animated_webp(
    frames: Vec<image::Frame>,
    loop_count: u16,
) -> Result<Vec<u8>, EngineError> {
    let frames: Vec<oxideav_webp::Frame> = frames
        .into_iter()
        .map(|frame| -> Result<oxideav_webp::Frame, EngineError> {
            let delay: std::time::Duration = frame.delay().into();
            let buffer = frame.into_buffer();
            let image = oxideav_webp::WebpImage::from_rgba8(
                buffer.width(),
                buffer.height(),
                buffer.into_raw(),
            )
            .map_err(|e| EngineError::Encode(e.to_string()))?;
            Ok(oxideav_webp::Frame::new(image, Some(delay)))
        })
        .collect::<Result<_, _>>()?;
    let opts = oxideav_webp::EncodeOptions::default().with_loop_count(loop_count);
    oxideav_webp::encode_animation(&frames, &opts).map_err(|e| EngineError::Encode(e.to_string()))
}

/// The largest width or height an AI upscale may produce.
/// Super-resolution multiplies activations by the factor
/// squared; this keeps the result inside a sane memory envelope.
const MAX_UPSCALE_DIM: u64 = 4096;

/// The largest input, in pixels, one AI upscale run may
/// take. A dynamic-size model's activation memory grows
/// with its input pixels, which the output cap above does
/// not bind at all: the campaign's Swin2SR x2 export ran a
/// 196,608-pixel composed run (the second half of an
/// `upscale(4)`) and a 172,800-pixel run fine, and
/// OOM-killed the whole process on a 504,000-pixel input
/// (issue #244). The budget sits at about half that kill
/// size, above every measured working run.
const MAX_UPSCALE_INPUT_PIXELS: u64 = 262_144;

/// AI-upscale `img` to `factor`x its current dimensions. The
/// provider's model runs at its native factor as many times as
/// needed; a final Lanczos pass lands the exact requested size
/// when the native factor does not divide it.
fn upscale_image(
    img: DynamicImage,
    factor: u32,
    provider: &dyn crate::ai::UpscaleProvider,
) -> Result<DynamicImage, EngineError> {
    let (tw, th) = (
        img.width() as u64 * factor as u64,
        img.height() as u64 * factor as u64,
    );
    if tw > MAX_UPSCALE_DIM || th > MAX_UPSCALE_DIM {
        return Err(EngineError::Unsupported(format!(
            "upscale({factor}) would produce {tw}x{th}, over the {MAX_UPSCALE_DIM}px limit; \
             resize the source first"
        )));
    }
    // Budget every run's input before any inference
    // happens (issue #244): a composed factor feeds each
    // run the previous run's output, so the projection
    // walks the same arithmetic the loop below will. A
    // run over the budget is a clean error here, never an
    // out-of-memory kill inside the model.
    let native = provider.factor().max(2) as u64;
    let mut run_pixels = img.width() as u64 * img.height() as u64;
    let mut projected = 1u64;
    while projected < factor as u64 {
        if run_pixels > MAX_UPSCALE_INPUT_PIXELS {
            return Err(EngineError::Unsupported(format!(
                "upscale({factor}) on a {}x{} image would feed a model run {run_pixels} \
                 pixels, over the {MAX_UPSCALE_INPUT_PIXELS} pixel input budget; \
                 resize the source first",
                img.width(),
                img.height()
            )));
        }
        run_pixels *= native * native;
        projected *= native;
    }
    let (target_w, target_h) = (tw as u32, th as u32);
    let mut img = img;
    let mut scale = 1u32;
    while scale < factor {
        let before = (img.width(), img.height());
        img = provider.upscale(&img).map_err(EngineError::Inference)?;
        if img.width() <= before.0 || img.height() <= before.1 {
            return Err(EngineError::Inference(crate::ai::AiError::BadOutput(
                "the upscale model did not enlarge the image".to_string(),
            )));
        }
        scale *= provider.factor().max(2);
    }
    if img.width() != target_w || img.height() != target_h {
        img = resize_lanczos3(&img, target_w, target_h);
    }
    Ok(img)
}

/// Apply the operations to one image, collecting the encoding
/// directives along the way. Shared by the still pipeline and
/// the animated paths (GIF and WebP), which run it once per
/// frame.
/// The effective DPR factor: an explicit `dpr(...)` directive
/// wins, then the client hint carried on the context, else 1.
fn dpr_factor(explicit: Option<f32>, ctx: &ProcessContext<'_>) -> f32 {
    explicit.or(ctx.dpr).unwrap_or(1.0)
}

/// Memoized face detection for one pipeline run: the image
/// dimensions the detections belong to, and the provider's
/// result. Several face operations on the same image state
/// share one inference; an operation that changes the size
/// re-detects against the new state.
type FaceMemo = Option<(
    u32,
    u32,
    Result<Vec<crate::ai::FaceBox>, crate::ai::AiError>,
)>;

/// Detect faces in `img` through the context's provider,
/// serving the memoized result when the image state matches.
/// The provider's failure surfaces as an inference error;
/// callers with a fallback (the face crop) may downgrade it.
fn detect_faces(
    memo: &mut FaceMemo,
    img: &DynamicImage,
    ctx: &ProcessContext<'_>,
) -> Result<Vec<crate::ai::FaceBox>, EngineError> {
    let provider = ctx.face_provider.ok_or_else(|| {
        EngineError::NotConfigured(
            "face operations are not configured on this server (no face model)".to_string(),
        )
    })?;
    if let Some((w, h, cached)) = &memo
        && *w == img.width()
        && *h == img.height()
    {
        return cached.clone().map_err(EngineError::Inference);
    }
    let found = provider.detect(img);
    *memo = Some((img.width(), img.height(), found.clone()));
    found.map_err(EngineError::Inference)
}

/// Memoized object detection for one pipeline run, the
/// face memo's twin: chained class filters on the same
/// image state share one inference.
type DetectionMemo = Option<(
    u32,
    u32,
    Result<Vec<crate::ai::Detection>, crate::ai::AiError>,
)>;

/// Every detection provider that can answer for this
/// request, in resolution order (issue #166): the built-in
/// detector first, then the declared detectors in
/// declaration order.
fn detection_providers<'a>(ctx: &ProcessContext<'a>) -> Vec<&'a dyn crate::ai::DetectionProvider> {
    let mut providers: Vec<&dyn crate::ai::DetectionProvider> = Vec::new();
    if let Some(provider) = ctx.detect_provider {
        providers.push(provider);
    }
    providers.extend(ctx.declared_detect_providers.iter().copied());
    providers
}

/// The slot of the first provider (in resolution order)
/// whose vocabulary knows `class`: the one provider a
/// class filter runs.
fn resolve_detector(providers: &[&dyn crate::ai::DetectionProvider], class: &str) -> Option<usize> {
    providers.iter().position(|p| p.knows_class(class))
}

/// Detect objects in `img` through one provider, memoized
/// like face detection. A provider failure is an inference
/// error: on a redaction request that fails the request
/// (never an unredacted response).
fn detect_objects(
    memo: &mut DetectionMemo,
    img: &DynamicImage,
    provider: &dyn crate::ai::DetectionProvider,
) -> Result<Vec<crate::ai::Detection>, EngineError> {
    if let Some((w, h, cached)) = &memo
        && *w == img.width()
        && *h == img.height()
    {
        return cached.clone().map_err(EngineError::Inference);
    }
    let found = provider.detect(img);
    *memo = Some((img.width(), img.height(), found.clone()));
    found.map_err(EngineError::Inference)
}

/// The detections of one class, through the provider that
/// resolution picks for it (issue #166): each provider has
/// its own memo slot, so chained class filters still share
/// one run per provider per image state.
fn detect_for_class(
    memos: &mut [DetectionMemo],
    providers: &[&dyn crate::ai::DetectionProvider],
    img: &DynamicImage,
    class: &str,
) -> Result<Vec<crate::ai::Detection>, EngineError> {
    if providers.is_empty() {
        return Err(EngineError::NotConfigured(
            "class redaction is not configured on this server (no detection model)".to_string(),
        ));
    }
    let slot = resolve_detector(providers, class).ok_or_else(|| {
        EngineError::Unsupported(format!(
            "unknown detection class {class:?}: not in the detection model's vocabulary"
        ))
    })?;
    let detections = detect_objects(&mut memos[slot], img, providers[slot])?;
    Ok(detections_of_class(&detections, class))
}

/// The detections of one class (case-insensitive, the
/// vocabulary's spelling is the model's own).
fn detections_of_class(
    detections: &[crate::ai::Detection],
    class: &str,
) -> Vec<crate::ai::Detection> {
    detections
        .iter()
        .filter(|d| d.label.eq_ignore_ascii_case(class))
        .cloned()
        .collect()
}

/// The crop-rect stance for explicit regions: a rectangle
/// that overflows the image in effect at this point in the
/// chain fails the request, rather than redacting less than
/// the caller specified.
fn check_region(
    name: &str,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    img: &DynamicImage,
) -> Result<(), EngineError> {
    // The coordinates are parser-supplied u32s: plain
    // addition can wrap (release) or panic (debug) before
    // the comparison ever runs, so compare the far edge
    // with checked arithmetic.
    let in_bounds = x.checked_add(w).is_some_and(|right| right <= img.width())
        && y.checked_add(h)
            .is_some_and(|bottom| bottom <= img.height());
    if !in_bounds {
        return Err(EngineError::Unsupported(format!(
            "{name}({x},{y},{w},{h}) exceeds image bounds {}x{}",
            img.width(),
            img.height()
        )));
    }
    Ok(())
}

fn apply_operations(
    mut img: DynamicImage,
    ops: &[Operation],
    cap: Option<(u32, u32)>,
    dpr: f32,
    ctx: &ProcessContext<'_>,
    directives: &crate::url::RequestDirectives,
) -> Result<
    (
        DynamicImage,
        EncodeOptions,
        Option<crate::studio::StudioRecord>,
    ),
    EngineError,
> {
    check_operation_count(ops)?;
    let mut encode_opts = EncodeOptions::default();
    let mut face_memo: FaceMemo = None;
    // One detection memo per provider, in resolution order
    // (issue #166).
    let detect_providers = detection_providers(ctx);
    let mut detection_memos: Vec<DetectionMemo> = vec![None; detect_providers.len()];
    // The pending focal rectangle from a `focal:` segment,
    // consumed by the next crop.
    let mut focal: Option<(u32, u32, u32, u32)> = None;
    // The `debug:focal` state: armed by the segment, captured
    // by the first crop that follows it. The capture holds
    // the values that crop actually used (threaded out of the
    // crop functions, never recomputed), and the response is
    // the rendered capture instead of the transform result.
    let mut debug_armed = false;
    let mut debug_capture: Option<crate::debugview::FocalCapture> = None;
    // The studio record (issue #197): the last `studio(...)`
    // filter in the chain wins, since its composite is the
    // look the response carries.
    let mut studio_record: Option<crate::studio::StudioRecord> = None;
    for op in ops {
        match op {
            Operation::DebugFocal => debug_armed = true,
            // Frame selection already happened at decode
            // (issue #214): the frame is the source image
            // this chain runs on, so there is nothing left
            // to apply here.
            Operation::Frame(_) => {}
            // Page selection likewise (issue #215): the
            // rendered page is the source image this chain
            // runs on.
            Operation::Page(_) => {}
            Operation::Size(size) => {
                img = apply_size(img, size, cap.map(|(cw, ch)| SizeCap::Axes(cw, ch)), dpr)?
            }
            Operation::FitIn { size, adaptive } => {
                img = apply_fit_in(img, size, *adaptive, cap, dpr)?
            }
            Operation::Pad { size, fill } => {
                let (tw, th) = match (size.width, size.height) {
                    (Dimension::Pixels(w), Dimension::Pixels(h)) => {
                        (scale_px(w, dpr), scale_px(h, dpr))
                    }
                    _ => {
                        return Err(EngineError::Unsupported(
                            "pad with non-pixel dimensions".to_string(),
                        ));
                    }
                };
                img = apply_pad(img, tw, th, *fill, directives.fill, cap)?;
            }
            Operation::FocalRect {
                left,
                top,
                right,
                bottom,
            } => {
                // Same stance as crop-rect: a rectangle that
                // overflows the image in effect at this point
                // fails the request.
                if *right > img.width() || *bottom > img.height() {
                    return Err(EngineError::Unsupported(format!(
                        "focal {left}x{top}:{right}x{bottom} exceeds image bounds {}x{}",
                        img.width(),
                        img.height()
                    )));
                }
                focal = Some((*left, *top, *right, *bottom));
            }
            Operation::CropRect {
                left,
                top,
                right,
                bottom,
            } => {
                if *right > img.width() || *bottom > img.height() {
                    return Err(EngineError::Unsupported(format!(
                        "crop-rect {left}x{top}:{right}x{bottom} exceeds image bounds {}x{}",
                        img.width(),
                        img.height()
                    )));
                }
                img = img.crop_imm(*left, *top, right - left, bottom - top);
            }
            Operation::Crop { size, align } => {
                let (tw, th) = match (size.width, size.height) {
                    (Dimension::Pixels(w), Dimension::Pixels(h)) => {
                        (scale_px(w, dpr), scale_px(h, dpr))
                    }
                    _ => {
                        return Err(EngineError::Unsupported(
                            "crop with non-pixel dimensions".to_string(),
                        ));
                    }
                };
                // The crop consumes any pending focal rectangle.
                // Center and smart crops position on it (smart
                // skips face and saliency detection entirely);
                // face and directional alignments were explicit
                // caller instructions and ignore it.
                let focal_rect = focal
                    .take()
                    .filter(|_| *align == Align::Center || *align == Align::Smart);
                // With the debug view armed, this crop is the
                // one the view annotates: clone the canvas it
                // operates on and run the traced crop variants,
                // which report the decision they actually made.
                let want_trace = debug_armed && debug_capture.is_none();
                let canvas = want_trace.then(|| img.clone());
                img = if let Some(rect) = focal_rect {
                    if want_trace {
                        let (cropped, window) =
                            crate::crop::crop_to_aspect_focal_traced(img, tw, th, rect);
                        let canvas = canvas.expect("want_trace clones the canvas");
                        let (cw, ch) = (canvas.width(), canvas.height());
                        debug_capture = Some(crate::debugview::FocalCapture {
                            canvas,
                            focal: Some((
                                ((rect.0 + rect.2) / 2).min(cw.saturating_sub(1)),
                                ((rect.1 + rect.3) / 2).min(ch.saturating_sub(1)),
                            )),
                            focal_rect: Some(rect),
                            window,
                            faces: Vec::new(),
                        });
                        cropped
                    } else {
                        crate::crop::crop_to_aspect_focal(img, tw, th, rect)
                    }
                } else if *align == Align::Face {
                    // Crop around the largest detected face. A
                    // detection failure or an empty result falls
                    // back to the saliency smart crop: a crop
                    // always exists, so it never fails the
                    // request.
                    match detect_faces(&mut face_memo, &img, ctx) {
                        Ok(faces) if !faces.is_empty() => {
                            let face = *faces
                                .iter()
                                .max_by_key(|f| f.w as u64 * f.h as u64)
                                .expect("faces is non-empty");
                            if want_trace {
                                let (cropped, window) = crate::crop::crop_around_face_traced(
                                    img,
                                    tw,
                                    th,
                                    &face,
                                    directives.face_padding,
                                );
                                debug_capture = Some(crate::debugview::FocalCapture {
                                    canvas: canvas.expect("want_trace clones the canvas"),
                                    focal: Some((face.x + face.w / 2, face.y + face.h / 2)),
                                    focal_rect: Some((
                                        face.x,
                                        face.y,
                                        face.x + face.w,
                                        face.y + face.h,
                                    )),
                                    window,
                                    faces,
                                });
                                cropped
                            } else {
                                crate::crop::crop_around_face(
                                    img,
                                    tw,
                                    th,
                                    &face,
                                    directives.face_padding,
                                )
                            }
                        }
                        Ok(_) | Err(EngineError::Inference(_)) => {
                            if want_trace {
                                let (cropped, trace) = crate::crop::crop_to_aspect_traced(
                                    img,
                                    tw,
                                    th,
                                    Align::Smart,
                                    None,
                                    ctx.focal_provider,
                                );
                                if let Some(trace) = trace {
                                    debug_capture = Some(capture_from_trace(
                                        canvas.expect("want_trace clones the canvas"),
                                        trace,
                                    ));
                                }
                                cropped
                            } else {
                                crate::crop::crop_to_aspect_with(
                                    img,
                                    tw,
                                    th,
                                    Align::Smart,
                                    None,
                                    ctx.focal_provider,
                                )
                            }
                        }
                        Err(e) => return Err(e),
                    }
                } else if want_trace {
                    let (cropped, trace) = crate::crop::crop_to_aspect_traced(
                        img,
                        tw,
                        th,
                        *align,
                        ctx.face_provider,
                        ctx.focal_provider,
                    );
                    if let Some(trace) = trace {
                        debug_capture = Some(capture_from_trace(
                            canvas.expect("want_trace clones the canvas"),
                            trace,
                        ));
                    }
                    cropped
                } else {
                    crate::crop::crop_to_aspect_with(
                        img,
                        tw,
                        th,
                        *align,
                        ctx.face_provider,
                        ctx.focal_provider,
                    )
                };
                img = apply_size(
                    img,
                    &ImageSize {
                        width: Dimension::Pixels(tw),
                        height: Dimension::Pixels(th),
                        flip_horizontal: false,
                        flip_vertical: false,
                    },
                    cap.map(|(cw, ch)| SizeCap::Fit(cw, ch)),
                    1.0,
                )?;
            }
            Operation::Filters(filters) => {
                // One filter at a time, in order: watermark
                // filters resolve against the server-fetched
                // images, everything else goes through
                // `apply_filters` as before.
                for filter in filters {
                    match filter {
                        Filter::Watermark { url, x, y, alpha } => {
                            let wm = ctx.watermarks.and_then(|m| m.get(url)).ok_or_else(|| {
                                EngineError::Unsupported(format!(
                                    "watermark was not resolved: {url}"
                                ))
                            })?;
                            img = overlay_watermark(&img, wm, *x, *y, *alpha);
                        }
                        Filter::Overlay {
                            url,
                            x,
                            y,
                            alpha,
                            blend,
                            width,
                        } => {
                            let ov = ctx.watermarks.and_then(|m| m.get(url)).ok_or_else(|| {
                                EngineError::Unsupported(format!("overlay was not resolved: {url}"))
                            })?;
                            img = blend_overlay(&img, ov, *x, *y, *alpha, *blend, *width);
                        }
                        Filter::RemoveBg => {
                            let provider = ctx.mask_provider.ok_or_else(|| {
                                EngineError::NotConfigured(
                                    "removebg is not configured on this server".to_string(),
                                )
                            })?;
                            let mask = provider.mask(&img).map_err(EngineError::Inference)?;
                            img = crate::ai::apply_mask(&img, &mask);
                        }
                        Filter::Studio(spec) => {
                            // Studio runs the removebg mask
                            // model, then composites in
                            // `crate::studio`: mask, defringe,
                            // backdrop and shadow, pad (issue
                            // #197). The record rides out with
                            // the result for meta.
                            let provider = ctx.mask_provider.ok_or_else(|| {
                                EngineError::NotConfigured(
                                    "studio is not configured on this server (no mask model)"
                                        .to_string(),
                                )
                            })?;
                            let t_timing = std::time::Instant::now();
                            let mask = provider.mask(&img).map_err(EngineError::Inference)?;
                            let confidence = provider.confidence(&img, &mask);
                            crate::studio::timing_mark("inference", t_timing);
                            let outcome = crate::studio::composite(
                                &img,
                                &mask,
                                confidence,
                                spec,
                                ctx.brand_color,
                            );
                            img = outcome.image;
                            studio_record = Some(outcome.record);
                        }
                        Filter::BlurFaces => {
                            // A detection failure fails the
                            // request: serving the image
                            // unredacted would be the real
                            // failure.
                            let faces = detect_faces(&mut face_memo, &img, ctx)?;
                            img = crate::filters::blur_faces(&img, &faces);
                        }
                        Filter::PixelateFaces => {
                            let faces = detect_faces(&mut face_memo, &img, ctx)?;
                            img = crate::filters::pixelate_faces(&img, &faces);
                        }
                        Filter::BlurClass(class) => {
                            // The #42 rule, inherited: a
                            // detection failure fails the
                            // request rather than serve the
                            // image unredacted.
                            let boxes = detect_for_class(
                                &mut detection_memos,
                                &detect_providers,
                                &img,
                                class,
                            )?;
                            img = crate::filters::blur_detections(&img, &boxes);
                        }
                        Filter::PixelateClass(class) => {
                            let boxes = detect_for_class(
                                &mut detection_memos,
                                &detect_providers,
                                &img,
                                class,
                            )?;
                            img = crate::filters::pixelate_detections(&img, &boxes);
                        }
                        // Padding is read from the request as a
                        // whole by the face crop; no pixels here.
                        Filter::FacePad(_) => {}
                        Filter::Upscale(factor) => {
                            let provider = ctx.upscale_provider.ok_or_else(|| {
                                EngineError::NotConfigured(
                                    "upscale is not configured on this server".to_string(),
                                )
                            })?;
                            img = upscale_image(img, *factor, provider)?;
                        }
                        Filter::BlurRegion { x, y, w, h } => {
                            check_region("blur_region", *x, *y, *w, *h, &img)?;
                            img = crate::filters::blur_region(&img, *x, *y, *w, *h);
                        }
                        Filter::PixelateRegion { x, y, w, h } => {
                            check_region("pixelate_region", *x, *y, *w, *h, &img)?;
                            img = crate::filters::pixelate_region(&img, *x, *y, *w, *h);
                        }
                        Filter::Text {
                            text,
                            size,
                            color,
                            anchor,
                        } => {
                            let renderer = ctx.text_renderer.ok_or_else(|| {
                                EngineError::NotConfigured(
                                    "text is not configured on this server (no font)".to_string(),
                                )
                            })?;
                            img = renderer
                                .draw(&img, text, *size, *color, *anchor)
                                .map_err(EngineError::Render)?;
                        }
                        other => {
                            // Arbitrary-angle rotation is
                            // the one plain filter that grows
                            // the canvas: check the canvas it
                            // is about to allocate before it
                            // allocates (issue #182, R2-12).
                            if let Filter::Rotate(degrees) = other {
                                let (cw, ch) = crate::filters::rotated_canvas(
                                    img.width(),
                                    img.height(),
                                    *degrees,
                                );
                                check_output_pixels(cw, ch)?;
                            }
                            let (new_img, options) = crate::filters::apply_filters_filled(
                                img,
                                std::slice::from_ref(other),
                                directives.fill,
                            );
                            img = new_img;
                            encode_opts.absorb(options);
                        }
                    }
                    // Every filter leaves the image inside
                    // the output budget: growth is checked
                    // before it compounds across a chain,
                    // not just at the size operations.
                    check_output_pixels(img.width(), img.height())?;
                }
            }
        }
        // The running image stays inside the output budget
        // after every operation, not only after the ones
        // that size it (issue #182, audit R2-12).
        check_output_pixels(img.width(), img.height())?;
    }
    if debug_armed {
        // The debug view replaces the transform result. With
        // no capture, no crop followed the segment: the view
        // shows the focal analysis of the final image. With
        // one, it shows the decision that crop actually made,
        // on the canvas it operated on.
        let capture = match debug_capture {
            Some(capture) => capture,
            None => {
                let (point, faces) =
                    crate::focus::focal_point_chained(&img, ctx.face_provider, ctx.focal_provider);
                let (w, h) = (img.width(), img.height());
                crate::debugview::FocalCapture {
                    canvas: img.clone(),
                    focal: Some((
                        ((point.x * w as f32).round() as u32).min(w.saturating_sub(1)),
                        ((point.y * h as f32).round() as u32).min(h.saturating_sub(1)),
                    )),
                    focal_rect: None,
                    window: None,
                    faces,
                }
            }
        };
        img = crate::debugview::render(&capture);
    }
    // The last word on size: whatever the chain did, the
    // image that goes to the encoder is inside the output
    // budget (issue #182, audit R2-12).
    check_output_pixels(img.width(), img.height())?;
    Ok((img, encode_opts, studio_record))
}

/// Build the debug capture for a crop that reported a
/// [`crate::crop::CropTrace`]: the traced focal fraction
/// becomes a pixel point on the canvas the crop ran on.
fn capture_from_trace(
    canvas: DynamicImage,
    trace: crate::crop::CropTrace,
) -> crate::debugview::FocalCapture {
    let (w, h) = (canvas.width(), canvas.height());
    let focal = trace.focal.map(|p| {
        (
            ((p.x * w as f32).round() as u32).min(w.saturating_sub(1)),
            ((p.y * h as f32).round() as u32).min(h.saturating_sub(1)),
        )
    });
    crate::debugview::FocalCapture {
        canvas,
        focal,
        focal_rect: None,
        window: Some(trace.window),
        faces: trace.faces,
    }
}

/// The EXIF orientation stored in the source bytes (1-8), if the
/// container carries a readable one. Missing or malformed metadata
/// yields `None`; re-encoding drops EXIF from outputs, so this read
/// is the only place orientation exists.
pub(crate) fn exif_orientation(bytes: &[u8]) -> Option<u32> {
    let mut cursor = std::io::Cursor::new(bytes);
    let exif = exif::Reader::new().read_from_container(&mut cursor).ok()?;
    let field = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?;
    field.value.get_uint(0)
}

/// Apply the transform an EXIF orientation value prescribes.
/// Values outside 1-8 leave the image untouched.
pub(crate) fn apply_orientation(img: DynamicImage, orientation: u32) -> DynamicImage {
    match orientation {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        // Transpose and anti-transpose: a rotation plus a mirror.
        5 => img.rotate90().fliph(),
        6 => img.rotate90(),
        7 => img.rotate270().fliph(),
        8 => img.rotate270(),
        _ => img,
    }
}

/// The largest raster the SVG path will produce per axis:
/// an intrinsic size is source-controlled, so it is capped
/// like AI upscaling output is.
const MAX_SVG_RASTER: u32 = 4096;

fn clamp_raster(w: u32, h: u32) -> (u32, u32) {
    let m = w.max(h);
    if m <= MAX_SVG_RASTER {
        (w.max(1), h.max(1))
    } else {
        let scale = MAX_SVG_RASTER as f64 / m as f64;
        (
            ((w as f64 * scale).round() as u32).max(1),
            ((h as f64 * scale).round() as u32).max(1),
        )
    }
}

/// Where an SVG raster should be rendered: the first size-like
/// operation's target (DPR-scaled, resolved against the
/// intrinsic size exactly as the pipeline will resolve it), or
/// the intrinsic size when the request asks for none.
fn svg_raster_target(ops: &[Operation], intrinsic: (f32, f32), dpr: f32) -> (u32, u32) {
    let (iw, ih) = (
        (intrinsic.0.round() as u32).max(1),
        (intrinsic.1.round() as u32).max(1),
    );
    for op in ops {
        match op {
            Operation::Size(size) => {
                let w = match size.width {
                    Dimension::Pixels(n) => scale_px(n, dpr),
                    Dimension::Original => iw,
                    Dimension::Proportional => match size.height {
                        Dimension::Pixels(h) => {
                            ((iw as u64 * scale_px(h, dpr) as u64) / ih as u64).max(1) as u32
                        }
                        _ => iw,
                    },
                };
                let h = match size.height {
                    Dimension::Pixels(n) => scale_px(n, dpr),
                    Dimension::Original => ih,
                    Dimension::Proportional => match size.width {
                        Dimension::Pixels(w) => {
                            ((ih as u64 * scale_px(w, dpr) as u64) / iw as u64).max(1) as u32
                        }
                        _ => ih,
                    },
                };
                return clamp_raster(w, h);
            }
            Operation::FitIn { size, adaptive } => {
                // Predict with the same box the pipeline will
                // fit into: DPR-scale the pixel axes, then let
                // adaptive swap them against the intrinsic
                // orientation.
                let resolved = ImageSize {
                    width: match size.width {
                        Dimension::Pixels(n) => Dimension::Pixels(scale_px(n, dpr)),
                        d => d,
                    },
                    height: match size.height {
                        Dimension::Pixels(n) => Dimension::Pixels(scale_px(n, dpr)),
                        d => d,
                    },
                    flip_horizontal: size.flip_horizontal,
                    flip_vertical: size.flip_vertical,
                };
                let boxed = fit_in_box(&resolved, iw, ih, *adaptive);
                let mut scale = f64::INFINITY;
                match boxed.width {
                    Dimension::Pixels(n) => scale = scale.min(n as f64 / iw as f64),
                    Dimension::Original => scale = scale.min(1.0),
                    Dimension::Proportional => {}
                }
                match boxed.height {
                    Dimension::Pixels(n) => scale = scale.min(n as f64 / ih as f64),
                    Dimension::Original => scale = scale.min(1.0),
                    Dimension::Proportional => {}
                }
                if scale.is_finite() {
                    return clamp_raster(
                        ((iw as f64 * scale).round() as u32).max(1),
                        ((ih as f64 * scale).round() as u32).max(1),
                    );
                }
            }
            Operation::Crop { size, .. } => {
                if let (Dimension::Pixels(w), Dimension::Pixels(h)) = (size.width, size.height) {
                    return clamp_raster(scale_px(w, dpr), scale_px(h, dpr));
                }
            }
            _ => {}
        }
    }
    clamp_raster(iw, ih)
}

/// The output container: detected source format, or JPEG when the
/// source format is unknown or not one we emit.
fn detect_format(bytes: &[u8]) -> ImageFormat {
    // The image crate's guess_format does not know JPEG XL.
    if is_jxl(bytes) {
        return ImageFormat::Jxl;
    }
    match image::guess_format(bytes) {
        Ok(image::ImageFormat::Jpeg) => ImageFormat::Jpeg,
        Ok(image::ImageFormat::Png) => ImageFormat::Png,
        Ok(image::ImageFormat::WebP) => ImageFormat::WebP,
        Ok(image::ImageFormat::Avif) => ImageFormat::Avif,
        Ok(image::ImageFormat::Gif) => ImageFormat::Gif,
        _ => ImageFormat::Jpeg,
    }
}

/// JPEG XL detection: the bare codestream signature, or the first
/// bytes of the ISO BMFF container signature.
fn is_jxl(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xFF, 0x0A])
        || bytes.starts_with(&[
            0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
        ])
}

/// Whether `bytes` carry a HEIC/HEIF file type box: an ISO-BMFF
/// `ftyp` box whose major brand is one of the HEVC image brands
/// (or the generic HEIF brands). AVIF shares the container but
/// not the brands, and is not matched.
fn is_heic(bytes: &[u8]) -> bool {
    if bytes.len() < 12 || &bytes[4..8] != b"ftyp" {
        return false;
    }
    matches!(
        &bytes[8..12],
        b"heic"
            | b"heix"
            | b"hevc"
            | b"hevx"
            | b"heim"
            | b"heis"
            | b"hevm"
            | b"hevs"
            | b"mif1"
            | b"mif2"
            | b"msf1"
    )
}

/// Whether `bytes` open as a video container: an
/// ISO-BMFF `ftyp` box whose major brand is not one of
/// the still-image brands `is_heic` claims (MP4, MOV,
/// and their kin), or a Matroska/WebM EBML header. The
/// one caller (`probe_source`) consults this only after
/// the image formats have had their claim, so an AVIF
/// still, which shares the box, is never mislabeled. A
/// container sniff only: whether it holds a readable
/// video stream is the decoder probe's call (issue
/// #214).
fn is_video_container(bytes: &[u8]) -> bool {
    if bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return true;
    }
    bytes.len() >= 12 && &bytes[4..8] == b"ftyp" && !is_heic(bytes)
}

/// One step down the `max_bytes` quality ladder, planned
/// against the steps that remain (the caller's smaller of the
/// per-format attempt cap and the shared encode pool): the
/// descent from the current quality to the floor is geometric,
/// spaced so the final allowed step lands exactly on quality
/// 1, the grammar's floor. A fixed-ratio walk cannot promise
/// that (from 95 it spent the whole pool reaching 15, and
/// budgets below that encode were never chased; issue #229).
/// The attempt caps still bound the cost per format, AVIF's
/// most tightly, since its encodes are the expensive ones.
fn next_quality(current: u8, steps_remaining: u32) -> u8 {
    if steps_remaining <= 1 || current <= 2 {
        return 1;
    }
    let frac = f64::from(steps_remaining - 1) / f64::from(steps_remaining);
    let next = (f64::from(current).powf(frac)) as u8;
    next.clamp(1, current - 1)
}

/// How many re-encodes `max_bytes` may attempt per format before
/// returning the smallest result it has.
fn max_quality_attempts(format: ImageFormat) -> u32 {
    match format {
        ImageFormat::Avif => 4,
        _ => 12,
    }
}

/// Pixel layouts for one encode phase, converted at most once
/// each and shared by every encode of the same image (issue
/// #138). The quality ladders (perceptual bisection, max_bytes)
/// re-encode one image up to a dozen times; converting the
/// full frame inside each attempt copied it every time. The
/// buffers compute lazily: a JPEG-only request never pays for
/// the RGBA layout.
struct EncodeBuffers<'a> {
    img: &'a DynamicImage,
    rgb: std::cell::OnceCell<image::RgbImage>,
    rgba: std::cell::OnceCell<image::RgbaImage>,
}

impl<'a> EncodeBuffers<'a> {
    fn new(img: &'a DynamicImage) -> Self {
        Self {
            img,
            rgb: std::cell::OnceCell::new(),
            rgba: std::cell::OnceCell::new(),
        }
    }

    fn rgb(&self) -> &image::RgbImage {
        self.rgb.get_or_init(|| self.img.to_rgb8())
    }

    fn rgba(&self) -> &image::RgbaImage {
        self.rgba.get_or_init(|| self.img.to_rgba8())
    }
}

/// Encode `src.img` to `format`. `quality` applies to JPEG,
/// JPEG XL, AVIF, and WebP (where 100 selects the lossless
/// encode). `fill` composites transparency for formats without
/// an alpha channel (only JPEG needs it here).
fn encode(
    src: &EncodeBuffers<'_>,
    format: ImageFormat,
    quality: Option<u8>,
    fill: Option<[u8; 3]>,
    avif_speed: u8,
    jxl_speed: jixel::Speed,
) -> Result<Vec<u8>, EngineError> {
    let img = src.img;
    let (w, h) = (img.width(), img.height());
    if format == ImageFormat::Jxl {
        // Pure-Rust encoder (jixel). Its quality is a libjpeg-like
        // 0-100 scale, matching the quality filter's semantics.
        let config = jixel::EncodeConfig::default()
            .with_quality(quality.unwrap_or(80) as f32)
            .with_speed(jxl_speed);
        let encoded = if img.color().has_alpha() {
            jixel::encode_image_with_alpha(src.rgba().as_raw(), w as usize, h as usize, &config)
        } else {
            jixel::encode_image(src.rgb().as_raw(), w as usize, h as usize, &config)
        };
        return encoded.map_err(|e| EngineError::Encode(e.to_string()));
    }
    if format == ImageFormat::WebP {
        return encode_webp(src, quality);
    }
    let mut buf = Vec::new();
    let (pixels, color) = (img.as_bytes(), ExtendedColorType::from(img.color()));
    match format {
        ImageFormat::Jpeg => {
            // JPEG has no alpha channel. A fill color composites
            // transparency over that background; without one, the
            // stored RGB values are used as-is. Encoding uses the
            // jpeg-encoder crate with 4:2:0 chroma subsampling
            // (the image crate's vendored encoder is fixed at
            // 4:2:2, and its quality scale runs hot compared to
            // libjpeg). Optimized Huffman tables stay OFF:
            // jpeg-encoder's optimized tables for degenerate
            // (tiny, flat) inputs produce files zune-jpeg decodes
            // incorrectly, an interop hazard measured in #38.
            let flattened;
            let rgb: &image::RgbImage = match fill {
                Some(bg) if img.color().has_alpha() => {
                    flattened = flatten(img, bg);
                    &flattened
                }
                _ => src.rgb(),
            };
            if w > u16::MAX as u32 || h > u16::MAX as u32 {
                return Err(EngineError::Encode(
                    "jpeg: dimension exceeds the 65535 format limit".into(),
                ));
            }
            let mut enc = jpeg_encoder::Encoder::new(&mut buf, quality.unwrap_or(80));
            enc.set_sampling_factor(jpeg_encoder::SamplingFactor::F_2_2);
            enc.encode(
                rgb.as_raw(),
                w as u16,
                h as u16,
                jpeg_encoder::ColorType::Rgb,
            )
            .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        ImageFormat::Png => {
            image::codecs::png::PngEncoder::new(&mut buf)
                .write_image(pixels, w, h, color)
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        ImageFormat::WebP => unreachable!("WebP returns early in encode"),
        ImageFormat::Avif => {
            // Pure-Rust ravif encoder via the image crate. The
            // speed tier is operator configuration; AVIF encoding
            // is the expensive step in the pipeline by a wide
            // margin, so the tier trades real latency for bytes.
            image::codecs::avif::AvifEncoder::new_with_speed_quality(
                &mut buf,
                avif_speed,
                quality.unwrap_or(80),
            )
            .write_image(pixels, w, h, color)
            .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        ImageFormat::Gif => {
            image::codecs::gif::GifEncoder::new(&mut buf)
                .encode(src.rgba().as_raw(), w, h, ExtendedColorType::Rgba8)
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        ImageFormat::Jxl => unreachable!("JPEG XL returns early in encode"),
        ImageFormat::Mp4 | ImageFormat::WebM => {
            unreachable!("video formats are encoded by process_video, never here")
        }
    }
    Ok(buf)
}

/// Encode WebP. Quality below 100 is a lossy VP8 encode via
/// oxideav-webp (pure Rust); quality 100 (or an encode the ladder
/// starts there) keeps the image crate's lossless VP8L encode.
/// The default quality is 80, like the other lossy formats.
fn encode_webp(src: &EncodeBuffers<'_>, quality: Option<u8>) -> Result<Vec<u8>, EngineError> {
    let img = src.img;
    let q = quality.unwrap_or(80);
    if q >= 100 {
        let mut buf = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut buf)
            .write_image(
                img.as_bytes(),
                img.width(),
                img.height(),
                ExtendedColorType::from(img.color()),
            )
            .map_err(|e| EngineError::Encode(e.to_string()))?;
        return Ok(buf);
    }
    // VP8 frames carry 14-bit dimensions.
    if img.width() > 16383 || img.height() > 16383 {
        return Err(EngineError::Encode(
            "webp: dimension exceeds the 16383 format limit".into(),
        ));
    }
    let opts = oxideav_webp::EncodeOptions::default().with_quality(q as f32);
    let encoded = if img.color().has_alpha() {
        oxideav_webp::encode_rgba8(img.width(), img.height(), src.rgba().as_raw(), &opts)
    } else {
        oxideav_webp::encode_rgb8(img.width(), img.height(), src.rgb().as_raw(), &opts)
    };
    encoded.map_err(|e| EngineError::Encode(e.to_string()))
}

/// Overlay `wm` onto `base` at (x, y) with `alpha` percent
/// opacity, honoring the watermark's own alpha channel.
/// Negative coordinates anchor from the right/bottom edges
/// (x = -1 puts the watermark's right edge one pixel from the
/// base's right edge). Parts falling outside the base are
/// clipped, never a panic.
/// One channel of a blend mode: the CSS formulas on the
/// 0-255 scale (`b` is the base channel, `o` the overlay's).
fn blend_channel(mode: crate::url::BlendMode, b: f32, o: f32) -> f32 {
    use crate::url::BlendMode;
    match mode {
        BlendMode::Normal => o,
        BlendMode::Multiply => b * o / 255.0,
        BlendMode::Screen => 255.0 - (255.0 - b) * (255.0 - o) / 255.0,
        BlendMode::Overlay => {
            if b < 127.5 {
                2.0 * b * o / 255.0
            } else {
                255.0 - 2.0 * (255.0 - b) * (255.0 - o) / 255.0
            }
        }
    }
}

/// Composite `ov` over `base` with a blend mode, at (x, y)
/// with an opacity (0-100), optionally after scaling `ov`
/// to `width_pct` percent of the base width (aspect
/// preserved). Anchoring and clipping follow
/// `overlay_watermark`, as does the alpha convention: the
/// effective alpha is the overlay pixel's alpha times the
/// opacity, and the base alpha channel is preserved, so
/// `Normal` reproduces a watermark exactly.
fn blend_overlay(
    base: &DynamicImage,
    ov: &DynamicImage,
    x: i32,
    y: i32,
    alpha: u8,
    mode: crate::url::BlendMode,
    width_pct: Option<u32>,
) -> DynamicImage {
    let scaled;
    let ov = match width_pct {
        Some(pct) => {
            let target = ((base.width() as u64 * pct as u64) / 100).max(1) as u32;
            scaled = ov.resize(target, u32::MAX, image::imageops::FilterType::Lanczos3);
            &scaled
        }
        None => ov,
    };
    let mut out = base.to_rgba8();
    let ov = ov.to_rgba8();
    let left = if x >= 0 {
        x as i64
    } else {
        out.width() as i64 - ov.width() as i64 + x as i64
    };
    let top = if y >= 0 {
        y as i64
    } else {
        out.height() as i64 - ov.height() as i64 + y as i64
    };
    let opacity = alpha as f32 / 100.0;
    let (bw, bh) = (out.width() as i64, out.height() as i64);
    for oy in 0..ov.height() as i64 {
        for ox in 0..ov.width() as i64 {
            let (bx, by) = (left + ox, top + oy);
            if bx < 0 || by < 0 || bx >= bw || by >= bh {
                continue;
            }
            let top_px = ov.get_pixel(ox as u32, oy as u32);
            let a = opacity * (top_px[3] as f32 / 255.0);
            if a <= 0.0 {
                continue;
            }
            let bot = out.get_pixel_mut(bx as u32, by as u32);
            for c in 0..3 {
                let blended = blend_channel(mode, bot[c] as f32, top_px[c] as f32);
                bot[c] = (bot[c] as f32 * (1.0 - a) + blended * a).round() as u8;
            }
        }
    }
    DynamicImage::ImageRgba8(out)
}

fn overlay_watermark(
    base: &DynamicImage,
    wm: &DynamicImage,
    x: i32,
    y: i32,
    alpha: u8,
) -> DynamicImage {
    let mut out = base.to_rgba8();
    let wm = wm.to_rgba8();
    let left = if x >= 0 {
        x as i64
    } else {
        out.width() as i64 - wm.width() as i64 + x as i64
    };
    let top = if y >= 0 {
        y as i64
    } else {
        out.height() as i64 - wm.height() as i64 + y as i64
    };
    let opacity = alpha as f32 / 100.0;
    let (bw, bh) = (out.width() as i64, out.height() as i64);
    for wy in 0..wm.height() as i64 {
        for wx in 0..wm.width() as i64 {
            let (bx, by) = (left + wx, top + wy);
            if bx < 0 || by < 0 || bx >= bw || by >= bh {
                continue;
            }
            let top_px = wm.get_pixel(wx as u32, wy as u32);
            let a = opacity * (top_px[3] as f32 / 255.0);
            if a <= 0.0 {
                continue;
            }
            let bot = out.get_pixel_mut(bx as u32, by as u32);
            for c in 0..3 {
                bot[c] = (bot[c] as f32 * (1.0 - a) + top_px[c] as f32 * a).round() as u8;
            }
        }
    }
    DynamicImage::ImageRgba8(out)
}

/// Composite an image over a solid background color, dropping the
/// alpha channel.
fn flatten(img: &DynamicImage, bg: [u8; 3]) -> image::RgbImage {
    let mut base = image::RgbaImage::from_pixel(
        img.width(),
        img.height(),
        image::Rgba([bg[0], bg[1], bg[2], 255]),
    );
    image::imageops::overlay(&mut base, &img.to_rgba8(), 0, 0);
    DynamicImage::ImageRgba8(base).to_rgb8()
}

/// Shrink `(w, h)` proportionally to fit inside `cap`. Never
/// grows the size.
fn cap_dimensions(w: u32, h: u32, (cw, ch): (u32, u32)) -> (u32, u32) {
    if w <= cw && h <= ch {
        return (w, h);
    }
    let scale = (cw as f64 / w as f64).min(ch as f64 / h as f64);
    (
        ((w as f64 * scale).round() as u32).max(1),
        ((h as f64 * scale).round() as u32).max(1),
    )
}

/// How the `no_upscale` cap (the oriented source dimensions)
/// bounds an [`apply_size`] target.
#[derive(Clone, Copy)]
enum SizeCap {
    /// Clamp each axis independently at the source dimension.
    /// A plain size names an exact box and distorts to it by
    /// design, so the cap clamps the axes it names: a 300x200
    /// request on a 200x150 source returns the source at
    /// 200x150, as the README promises (issue #230). Shrinking
    /// the box proportionally instead silently traded the
    /// blocked enlargement for a resample to a third aspect
    /// (200x133) that nobody asked for.
    Axes(u32, u32),
    /// Shrink the target proportionally until it fits. The
    /// crop's resize uses this: its target carries the crop
    /// window's own proportions, so shrinking preserves them
    /// and the window is never enlarged or distorted.
    Fit(u32, u32),
}

/// One pixel target under DPR: whole pixels, never zero.
/// A factor of 1 returns the input unchanged.
fn scale_px(n: u32, dpr: f32) -> u32 {
    if dpr == 1.0 {
        n
    } else {
        ((n as f32 * dpr).round() as u32).max(1)
    }
}

/// DPR applies to pixel targets only: `orig` already means the
/// source dimension and proportional dimensions derive from
/// the dimension they follow (which is scaled itself).
fn scale_dimension(d: Dimension, dpr: f32) -> Dimension {
    match d {
        Dimension::Pixels(n) => Dimension::Pixels(scale_px(n, dpr)),
        other => other,
    }
}

/// Resize `img` per `size`, honoring proportional/original dimensions
/// and flips. Pixel targets are DPR-scaled first; `cap` (from
/// `no_upscale`) bounds the result afterwards, so DPR can never
/// push past the source when the cap is set.
fn apply_size(
    img: DynamicImage,
    size: &ImageSize,
    cap: Option<SizeCap>,
    dpr: f32,
) -> Result<DynamicImage, EngineError> {
    let size = &ImageSize {
        width: scale_dimension(size.width, dpr),
        height: scale_dimension(size.height, dpr),
        flip_horizontal: size.flip_horizontal,
        flip_vertical: size.flip_vertical,
    };
    let (sw, sh) = (img.width(), img.height());
    let w = match size.width {
        Dimension::Pixels(n) => n,
        Dimension::Original => sw,
        Dimension::Proportional => match size.height {
            Dimension::Pixels(h) => ((sw as u64 * h as u64) / sh as u64).max(1) as u32,
            _ => sw,
        },
    };
    let h = match size.height {
        Dimension::Pixels(n) => n,
        Dimension::Original => sh,
        Dimension::Proportional => match size.width {
            Dimension::Pixels(w) => ((sh as u64 * w as u64) / sw as u64).max(1) as u32,
            _ => sh,
        },
    };
    let (w, h) = match cap {
        Some(SizeCap::Axes(cw, ch)) => (w.min(cw), h.min(ch)),
        Some(SizeCap::Fit(cw, ch)) => cap_dimensions(w, h, (cw, ch)),
        None => (w, h),
    };
    check_output_pixels(w, h)?;
    let mut img = if w == sw && h == sh {
        img
    } else {
        resize_lanczos3(&img, w, h)
    };
    if size.flip_horizontal {
        img = img.fliph();
    }
    if size.flip_vertical {
        img = img.flipv();
    }
    Ok(img)
}

/// Resize to (w, h) with the SIMD backend (fast_image_resize),
/// Lanczos3. Same filter as the image crate's Lanczos3 up to
/// rounding (max diff 1 per channel on photo content), except
/// alpha: the backend resamples premultiplied alpha, so colors
/// stored under transparent pixels no longer bleed into visible
/// ones. Pixel types the backend does not cover fall back to
/// the image crate.
pub(crate) fn resize_lanczos3(img: &DynamicImage, w: u32, h: u32) -> DynamicImage {
    let mut dst = match img {
        DynamicImage::ImageLuma8(_) => DynamicImage::new_luma8(w, h),
        DynamicImage::ImageLumaA8(_) => DynamicImage::new_luma_a8(w, h),
        DynamicImage::ImageRgb8(_) => DynamicImage::new_rgb8(w, h),
        DynamicImage::ImageRgba8(_) => DynamicImage::new_rgba8(w, h),
        DynamicImage::ImageLuma16(_) => DynamicImage::new_luma16(w, h),
        DynamicImage::ImageLumaA16(_) => DynamicImage::new_luma_a16(w, h),
        DynamicImage::ImageRgb16(_) => DynamicImage::new_rgb16(w, h),
        DynamicImage::ImageRgba16(_) => DynamicImage::new_rgba16(w, h),
        DynamicImage::ImageRgb32F(_) => DynamicImage::new_rgb32f(w, h),
        DynamicImage::ImageRgba32F(_) => DynamicImage::new_rgba32f(w, h),
        _ => {
            return DynamicImage::ImageRgba8(image::imageops::resize(
                img,
                w,
                h,
                image::imageops::FilterType::Lanczos3,
            ));
        }
    };
    let opts = fast_image_resize::ResizeOptions::new().resize_alg(
        fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::Lanczos3),
    );
    match fast_image_resize::Resizer::new().resize(img, &mut dst, &opts) {
        Ok(()) => dst,
        Err(_) => DynamicImage::ImageRgba8(image::imageops::resize(
            img,
            w,
            h,
            image::imageops::FilterType::Lanczos3,
        )),
    }
}

/// Scale proportionally to fit inside the `fit-in` box, preserving
/// aspect and never distorting. A 0 dimension bounds nothing;
/// `orig` bounds at the source dimension (a scale of 1).
/// Pad `img` to exactly `target_w` x `target_h`: fit the image
/// inside the box (the fit-in rules, `no_upscale` cap included),
/// center it, and fill the rest of the canvas: a solid color
/// (the pad segment's own, or the request's `fill(...)`
/// directive, or black), or a blurred cover-scaled copy of the
/// image itself.
fn apply_pad(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    fill: PadFill,
    directive_fill: Option<[u8; 3]>,
    cap: Option<(u32, u32)>,
) -> Result<DynamicImage, EngineError> {
    if target_w == 0 || target_h == 0 {
        return Ok(img);
    }
    check_output_pixels(target_w, target_h)?;
    let fitted = apply_fit_in(
        img.clone(),
        &ImageSize {
            width: Dimension::Pixels(target_w),
            height: Dimension::Pixels(target_h),
            flip_horizontal: false,
            flip_vertical: false,
        },
        // Pad never adapts: the canvas is the explicit box,
        // so the fit inside it uses the box as written.
        false,
        cap,
        1.0,
    )?;
    if fitted.width() == target_w && fitted.height() == target_h {
        return Ok(fitted);
    }
    let color = match fill {
        PadFill::Color(c) => Some(c),
        PadFill::Directive => Some(directive_fill.unwrap_or([0, 0, 0])),
        PadFill::Blur => None,
    };
    let canvas = match color {
        Some(c) => DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            target_w,
            target_h,
            image::Rgba([c[0], c[1], c[2], 255]),
        )),
        None => {
            // Ambient bars: the image scaled to cover the whole
            // canvas, then blurred hard enough that only its
            // broad colors survive.
            let sigma = ((target_w.max(target_h) as f32) / 16.0).clamp(2.0, 80.0);
            img.resize_to_fill(target_w, target_h, image::imageops::FilterType::Triangle)
                .blur(sigma)
        }
    };
    let mut base = canvas.to_rgba8();
    let x = (target_w - fitted.width()) / 2;
    let y = (target_h - fitted.height()) / 2;
    image::imageops::overlay(&mut base, &fitted.to_rgba8(), x as i64, y as i64);
    Ok(DynamicImage::ImageRgba8(base))
}

/// The box an adaptive fit-in actually fits into: the declared
/// box with its axes swapped when the source orientation and
/// the box orientation disagree (portrait source into a
/// landscape box, or landscape source into a portrait box).
/// Orientations compare by strict inequality, so equality on
/// either side never swaps: a square source and a square box
/// have no orientation to disagree with. A box without two
/// pixel axes (a proportional or `orig` axis) likewise has no
/// fixed orientation and never swaps.
fn fit_in_box(size: &ImageSize, sw: u32, sh: u32, adaptive: bool) -> ImageSize {
    if !adaptive {
        return *size;
    }
    let (bw, bh) = match (size.width, size.height) {
        (Dimension::Pixels(w), Dimension::Pixels(h)) => (w, h),
        _ => return *size,
    };
    let disagree = (sh > sw && bw > bh) || (sw > sh && bh > bw);
    if disagree {
        ImageSize {
            width: size.height,
            height: size.width,
            flip_horizontal: size.flip_horizontal,
            flip_vertical: size.flip_vertical,
        }
    } else {
        *size
    }
}

fn apply_fit_in(
    img: DynamicImage,
    size: &ImageSize,
    adaptive: bool,
    cap: Option<(u32, u32)>,
    dpr: f32,
) -> Result<DynamicImage, EngineError> {
    let scaled = ImageSize {
        width: scale_dimension(size.width, dpr),
        height: scale_dimension(size.height, dpr),
        flip_horizontal: size.flip_horizontal,
        flip_vertical: size.flip_vertical,
    };
    let (sw, sh) = (img.width(), img.height());
    let size = &fit_in_box(&scaled, sw, sh, adaptive);
    let mut scale = f64::INFINITY;
    match size.width {
        Dimension::Pixels(n) => scale = scale.min(n as f64 / sw as f64),
        Dimension::Original => scale = scale.min(1.0),
        Dimension::Proportional => {}
    }
    match size.height {
        Dimension::Pixels(n) => scale = scale.min(n as f64 / sh as f64),
        Dimension::Original => scale = scale.min(1.0),
        Dimension::Proportional => {}
    }
    if !scale.is_finite() {
        return Ok(img);
    }
    let w = ((sw as f64 * scale).round() as u32).max(1);
    let h = ((sh as f64 * scale).round() as u32).max(1);
    let (w, h) = match cap {
        Some(c) => cap_dimensions(w, h, c),
        None => (w, h),
    };
    check_output_pixels(w, h)?;
    if w == sw && h == sh {
        Ok(img)
    } else {
        Ok(resize_lanczos3(&img, w, h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::url::{Align, Filter, ImageFormat, Quality};
    use image::GenericImageView;

    fn png_bytes(img: &DynamicImage) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    fn solid_png(width: u32, height: u32, color: [u8; 3]) -> Vec<u8> {
        let mut img = DynamicImage::new_rgb8(width, height);
        for p in img.as_mut_rgb8().unwrap().pixels_mut() {
            *p = image::Rgb(color);
        }
        png_bytes(&img)
    }

    fn size_op(width: Dimension, height: Dimension) -> Operation {
        Operation::Size(ImageSize {
            width,
            height,
            flip_horizontal: false,
            flip_vertical: false,
        })
    }

    fn decode_dims(bytes: &[u8]) -> (u32, u32) {
        let img = image::load_from_memory(bytes).unwrap();
        (img.width(), img.height())
    }

    #[test]
    fn validate_source_image_accepts_the_servable_set() {
        // The upload acceptance test (issue #168) walks the
        // pipeline's own decode branches.
        assert!(validate_source_image(&solid_png(20, 10, [1, 2, 3])).is_ok());
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10" fill="red"/></svg>"#;
        assert!(validate_source_image(svg).is_ok());
        // Non-images, truncated images, and empty bodies
        // are refused.
        assert!(validate_source_image(b"not an image at all").is_err());
        assert!(validate_source_image(b"").is_err());
        let truncated = &solid_png(20, 10, [1, 2, 3])[..20];
        assert!(validate_source_image(truncated).is_err());
        // A script-carrying SVG is still an SVG the
        // sanitizer renders safely, so it validates like
        // any other SVG source would.
        let hostile = br#"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><script>alert(1)</script><rect width="4" height="4"/></svg>"#;
        assert!(validate_source_image(hostile).is_ok());
    }

    #[test]
    fn prepare_upload_reports_media_and_stores_the_sanitized_svg() {
        // Raster uploads store the caller's bytes verbatim
        // and report their kind.
        let png = solid_png(20, 10, [1, 2, 3]);
        let prepared = prepare_upload(&png).unwrap();
        assert_eq!(prepared.media, UploadMedia::Png);
        assert_eq!(prepared.stored, png);
        // An SVG upload validates, and what would be
        // stored is the sanitized document: the script is
        // gone, the drawing survives (audit R2-22).
        let hostile = br#"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><script>alert(1)</script><rect width="4" height="4"/></svg>"#;
        let prepared = prepare_upload(hostile).unwrap();
        assert_eq!(prepared.media, UploadMedia::Svg);
        let stored = String::from_utf8(prepared.stored).unwrap();
        assert!(!stored.contains("script"), "{stored}");
        assert!(stored.contains("<rect"), "{stored}");
        // The stored form is itself a valid upload.
        assert_eq!(
            prepare_upload(stored.as_bytes()).unwrap().media,
            UploadMedia::Svg
        );
    }

    #[test]
    fn round_trips_png() {
        let engine = Engine;
        let out = engine
            .process(&solid_png(100, 60, [10, 20, 30]), &[])
            .unwrap()
            .bytes;
        assert_eq!(&out[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(decode_dims(&out), (100, 60));
    }

    #[test]
    fn resizes_to_exact_size() {
        let engine = Engine;
        let out = engine
            .process(
                &solid_png(100, 60, [10, 20, 30]),
                &[size_op(Dimension::Pixels(50), Dimension::Pixels(30))],
            )
            .unwrap()
            .bytes;
        assert_eq!(decode_dims(&out), (50, 30));
    }

    #[test]
    fn resizes_proportionally() {
        let engine = Engine;
        let bytes = solid_png(100, 60, [10, 20, 30]);
        let out = engine
            .process(
                &bytes,
                &[size_op(Dimension::Pixels(50), Dimension::Proportional)],
            )
            .unwrap()
            .bytes;
        assert_eq!(decode_dims(&out), (50, 30));
        let out = engine
            .process(
                &bytes,
                &[size_op(Dimension::Proportional, Dimension::Pixels(30))],
            )
            .unwrap()
            .bytes;
        assert_eq!(decode_dims(&out), (50, 30));
    }

    #[test]
    fn keeps_original_dimensions() {
        let engine = Engine;
        let out = engine
            .process(
                &solid_png(100, 60, [10, 20, 30]),
                &[size_op(Dimension::Original, Dimension::Original)],
            )
            .unwrap()
            .bytes;
        assert_eq!(decode_dims(&out), (100, 60));
    }

    #[test]
    fn flips_horizontally() {
        let mut img = DynamicImage::new_rgb8(2, 1);
        let rgb = img.as_mut_rgb8().unwrap();
        rgb.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        rgb.put_pixel(1, 0, image::Rgb([0, 0, 255]));
        let engine = Engine;
        let out = engine
            .process(
                &png_bytes(&img),
                &[Operation::Size(ImageSize {
                    width: Dimension::Pixels(2),
                    height: Dimension::Pixels(1),
                    flip_horizontal: true,
                    flip_vertical: false,
                })],
            )
            .unwrap()
            .bytes;
        let decoded = image::load_from_memory(&out).unwrap();
        let left = decoded.get_pixel(0, 0);
        let right = decoded.get_pixel(1, 0);
        assert_eq!([left[0], left[1], left[2]], [0, 0, 255]);
        assert_eq!([right[0], right[1], right[2]], [255, 0, 0]);
    }

    #[test]
    fn preserves_jpeg_format() {
        let mut buf = std::io::Cursor::new(Vec::new());
        DynamicImage::new_rgb8(100, 60)
            .write_to(&mut buf, image::ImageFormat::Jpeg)
            .unwrap();
        let buf = buf.into_inner();
        let engine = Engine;
        let out = engine.process(&buf, &[]).unwrap().bytes;
        assert_eq!(&out[..2], b"\xff\xd8");
        assert_eq!(decode_dims(&out), (100, 60));
    }

    #[test]
    fn rejects_corrupt_input() {
        let engine = Engine;
        assert!(matches!(
            engine.process(b"not an image", &[]),
            Err(EngineError::Decode(_))
        ));
    }

    #[test]
    fn crops_then_resizes_to_exact_size() {
        let engine = Engine;
        let out = engine
            .process(
                &solid_png(100, 60, [10, 20, 30]),
                &[Operation::Crop {
                    size: ImageSize {
                        width: Dimension::Pixels(50),
                        height: Dimension::Pixels(50),
                        flip_horizontal: false,
                        flip_vertical: false,
                    },
                    align: Align::Center,
                }],
            )
            .unwrap()
            .bytes;
        assert_eq!(decode_dims(&out), (50, 50));
    }

    fn gradient_png() -> Vec<u8> {
        let mut img = DynamicImage::new_rgb8(64, 64);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let v = ((x + y) % 256) as u8;
            *p = image::Rgb([v, 255 - v, (x * y % 256) as u8]);
        }
        png_bytes(&img)
    }

    /// A photo surrogate for perceptual tests: smooth
    /// multi-frequency content (sums of sinusoids) at a size the
    /// metric's scales can work with. Unlike the sawtooth
    /// gradient, lossy codecs can encode it well.
    fn photo_like_png() -> Vec<u8> {
        let mut img = DynamicImage::new_rgb8(240, 180);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let (fx, fy) = (x as f64, y as f64);
            let wave = |phase: f64, a: f64, b: f64| {
                128.0
                    + 62.0 * (fx * a + phase).sin() * (fy * b + phase * 0.7).cos()
                    + 24.0 * ((fx + fy) * 0.021 + phase).sin()
            };
            *p = image::Rgb([
                wave(0.0, 0.11, 0.07).clamp(0.0, 255.0) as u8,
                wave(2.1, 0.09, 0.13).clamp(0.0, 255.0) as u8,
                wave(4.2, 0.05, 0.05).clamp(0.0, 255.0) as u8,
            ]);
        }
        png_bytes(&img)
    }

    #[test]
    fn grayscale_filter_desaturates() {
        let engine = Engine;
        let out = engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![Filter::Grayscale])],
            )
            .unwrap()
            .bytes;
        let decoded = image::load_from_memory(&out).unwrap();
        for p in decoded.to_rgb8().pixels() {
            assert_eq!(p[0], p[1]);
            assert_eq!(p[1], p[2]);
        }
    }

    #[test]
    fn quality_filter_changes_jpeg_size() {
        let engine = Engine;
        let filters = |q| {
            vec![Operation::Filters(vec![
                Filter::Format(ImageFormat::Jpeg),
                Filter::Quality(Quality::Value(q)),
            ])]
        };
        let hi = engine.process(&gradient_png(), &filters(95)).unwrap();
        let lo = engine.process(&gradient_png(), &filters(10)).unwrap();
        assert_eq!(hi.format, ImageFormat::Jpeg);
        assert!(hi.bytes.len() > lo.bytes.len());
        assert_eq!(&hi.bytes[..2], b"\xff\xd8");
    }

    #[test]
    fn region_filters_apply_and_check_bounds() {
        let engine = Engine;
        let src = solid_png(100, 80, [200, 30, 30]);
        // Multiple regions in one chain apply.
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::BlurRegion {
                        x: 0,
                        y: 0,
                        w: 10,
                        h: 10,
                    },
                    Filter::PixelateRegion {
                        x: 90,
                        y: 70,
                        w: 10,
                        h: 10,
                    },
                ])],
            )
            .unwrap();
        assert_eq!((out.width, out.height), (100, 80));
        // Overflow is an error naming the region and the bounds.
        let err = engine
            .process(
                &src,
                &[Operation::Filters(vec![Filter::BlurRegion {
                    x: 90,
                    y: 70,
                    w: 20,
                    h: 20,
                }])],
            )
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("blur_region(90,70,20,20) exceeds image bounds 100x80"),
            "got: {err}"
        );
        // Near-MAX coordinates must error, not wrap the
        // bounds check (or panic on it in debug builds).
        let err = engine
            .process(
                &src,
                &[Operation::Filters(vec![Filter::BlurRegion {
                    x: u32::MAX - 10,
                    y: 0,
                    w: 20,
                    h: 10,
                }])],
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("exceeds image bounds 100x80"),
            "got: {err}"
        );
        // Bounds are the size at the point the filter runs:
        // after a resize to 50x40, a 60-wide region no longer fits.
        let err = engine
            .process(
                &src,
                &[
                    size_op(Dimension::Pixels(50), Dimension::Pixels(40)),
                    Operation::Filters(vec![Filter::PixelateRegion {
                        x: 0,
                        y: 0,
                        w: 60,
                        h: 10,
                    }]),
                ],
            )
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("pixelate_region(0,0,60,10) exceeds image bounds 50x40"),
            "got: {err}"
        );
    }

    #[test]
    fn shape_filters_need_an_alpha_capable_format() {
        let engine = Engine;
        let src = solid_png(40, 40, [200, 30, 30]);
        let err = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Radius([10, 10, 10, 10]),
                    Filter::Format(ImageFormat::Jpeg),
                ])],
            )
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("radius needs an alpha-capable output format"),
            "got: {err}"
        );
        let err = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Mask(crate::url::MaskShape::Circle),
                    Filter::Format(ImageFormat::Gif),
                ])],
            )
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("mask needs an alpha-capable output format"),
            "got: {err}"
        );
        // PNG output carries the transparency through.
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Radius([10, 10, 10, 10]),
                    Filter::Format(ImageFormat::Png),
                ])],
            )
            .unwrap();
        let decoded = image::load_from_memory(&out.bytes).unwrap();
        assert_eq!(decoded.to_rgba8().get_pixel(0, 0)[3], 0);
        assert_eq!(decoded.to_rgba8().get_pixel(20, 20)[3], 255);
        // Border is opaque paint: JPEG is fine.
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Border(3, [0, 255, 0]),
                    Filter::Format(ImageFormat::Jpeg),
                ])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jpeg);
    }

    #[test]
    fn format_filter_selects_container() {
        let engine = Engine;
        // PNG in, JPEG out.
        let out = engine
            .process(
                &solid_png(100, 60, [10, 20, 30]),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jpeg)])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jpeg);
        assert_eq!(&out.bytes[..2], b"\xff\xd8");
        // JPEG in, PNG out.
        let mut jpeg = std::io::Cursor::new(Vec::new());
        DynamicImage::new_rgb8(100, 60)
            .write_to(&mut jpeg, image::ImageFormat::Jpeg)
            .unwrap();
        let out = engine
            .process(
                &jpeg.into_inner(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Png)])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Png);
        assert_eq!(&out.bytes[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn filters_combine_with_resize() {
        let engine = Engine;
        let out = engine
            .process(
                &gradient_png(),
                &[
                    size_op(Dimension::Pixels(32), Dimension::Pixels(32)),
                    Operation::Filters(vec![Filter::Grayscale, Filter::Format(ImageFormat::Jpeg)]),
                ],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jpeg);
        let decoded = image::load_from_memory(&out.bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (32, 32));
        let p = decoded.get_pixel(0, 0);
        assert_eq!(p[0], p[1]);
    }

    #[test]
    fn new_filters_flow_through_engine() {
        let engine = Engine;
        let out = engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![
                    Filter::Blur(1.5),
                    Filter::Invert,
                    Filter::Format(ImageFormat::Png),
                ])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Png);
        assert!(out.bytes.starts_with(&[0x89, b'P', b'N', b'G']));
        image::load_from_memory(&out.bytes).unwrap();
    }

    #[test]
    fn noise_flows_through_engine_and_is_deterministic() {
        let engine = Engine;
        let run = |filters: Vec<Filter>| {
            engine
                .process(
                    &gradient_png(),
                    &[
                        size_op(Dimension::Pixels(32), Dimension::Pixels(32)),
                        Operation::Filters(filters),
                    ],
                )
                .unwrap()
        };
        // The filter composes with resize and a format
        // directive like any other pixel filter.
        let noisy = run(vec![Filter::Noise(60), Filter::Format(ImageFormat::Png)]);
        assert_eq!(noisy.format, ImageFormat::Png);
        let decoded = image::load_from_memory(&noisy.bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (32, 32));
        // The same request renders the same bytes on a
        // second run: the field is seeded from content.
        let again = run(vec![Filter::Noise(60), Filter::Format(ImageFormat::Png)]);
        assert_eq!(noisy.bytes, again.bytes);
        // And it is not the clean image.
        let clean = run(vec![Filter::Format(ImageFormat::Png)]);
        assert_ne!(noisy.bytes, clean.bytes);
        // Amount 0 is a byte-identical no-op end to end.
        let zero = run(vec![Filter::Noise(0), Filter::Format(ImageFormat::Png)]);
        assert_eq!(zero.bytes, clean.bytes);
    }

    #[test]
    fn jxl_format_encodes() {
        let engine = Engine;
        let out = engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jxl)])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jxl);
        // Bare codestream signature.
        assert!(out.bytes.starts_with(&[0xFF, 0x0A]));
    }

    #[test]
    fn jxl_source_decodes_and_round_trips() {
        let engine = Engine;
        let jxl = engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jxl)])],
            )
            .unwrap();
        // The JXL output becomes the source: resize with no format
        // filter. Detection must keep JPEG XL and decoding must work.
        let out = engine
            .process(
                &jxl.bytes,
                &[size_op(Dimension::Pixels(16), Dimension::Pixels(16))],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jxl);
        let decoded = image::load_from_memory(&out.bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (16, 16));
    }

    #[test]
    fn jxl_quality_changes_size() {
        let engine = Engine;
        let mk = |q| {
            engine
                .process(
                    &gradient_png(),
                    &[Operation::Filters(vec![
                        Filter::Quality(Quality::Value(q)),
                        Filter::Format(ImageFormat::Jxl),
                    ])],
                )
                .unwrap()
        };
        assert!(mk(95).bytes.len() > mk(10).bytes.len());
    }

    fn flat_png() -> Vec<u8> {
        png_bytes(&DynamicImage::ImageLuma8(image::GrayImage::from_pixel(
            120,
            120,
            image::Luma([128]),
        )))
    }

    fn noise_png() -> Vec<u8> {
        let mut state: u32 = 0x2545_F491;
        let mut img = DynamicImage::new_rgb8(120, 120);
        for p in img.as_mut_rgb8().unwrap().pixels_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *p = image::Rgb([(state >> 24) as u8, (state >> 16) as u8, state as u8]);
        }
        png_bytes(&img)
    }

    #[test]
    fn auto_quality_resolves_and_reports() {
        let engine = Engine;
        let ops = &[Operation::Filters(vec![
            Filter::Quality(Quality::Auto),
            Filter::Format(ImageFormat::Jpeg),
        ])];
        let flat = engine.process(&flat_png(), ops).unwrap();
        assert_eq!(flat.quality, Some(50));
        let noisy = engine.process(&noise_png(), ops).unwrap();
        let q = noisy.quality.expect("auto quality is reported");
        assert!((50..=95).contains(&q), "chosen quality {q}");
        assert!(q > flat.quality.unwrap());
        // The detailed image ends up larger than the flat one.
        assert!(noisy.bytes.len() > flat.bytes.len());
    }

    /// A PNG carrying a Display P3 profile and three known
    /// pixels, built with the production injector.
    fn p3_png() -> Vec<u8> {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(3, 1, |x, _| {
            [
                image::Rgb([200, 80, 40]),
                image::Rgb([40, 120, 200]),
                image::Rgb([90, 160, 90]),
            ][x as usize]
        }));
        let icc = moxcms::ColorProfile::new_display_p3().encode().unwrap();
        crate::color::inject_png(
            &png_bytes(&img),
            Some(&icc),
            &crate::color::SourceMeta::default(),
            false,
        )
    }

    #[test]
    fn a_wide_gamut_source_converts_to_srgb_by_default() {
        let engine = Engine;
        let out = engine
            .process(
                &p3_png(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Png)])],
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        let expected: [[u8; 3]; 3] = [[216, 69, 17], [0, 122, 206], [61, 162, 81]];
        for (i, want) in expected.iter().enumerate() {
            let got = img.get_pixel(i as u32, 0).0;
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(
                    (i32::from(*g) - i32::from(*w)).abs() <= 2,
                    "pixel {i}: {got:?} vs {want:?}"
                );
            }
        }
        // And the converted output carries no profile: sRGB
        // is the assumed space now.
        assert!(crate::color::extract_source_meta(&out.bytes).icc.is_none());
    }

    #[test]
    fn keep_profile_keeps_pixels_and_embeds_the_profile() {
        let engine = Engine;
        let icc = moxcms::ColorProfile::new_display_p3().encode().unwrap();
        let out = engine
            .process(
                &p3_png(),
                &[Operation::Filters(vec![
                    Filter::KeepProfile,
                    Filter::Format(ImageFormat::Png),
                ])],
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        assert_eq!(img.get_pixel(0, 0).0, [200, 80, 40]);
        assert_eq!(crate::color::extract_source_meta(&out.bytes).icc, Some(icc));
    }

    #[test]
    fn a_cmyk_jpeg_converts_sanely() {
        // Saturated red in CMYK (Adobe storage), encoded
        // with the same encoder family the engine uses.
        let mut data = Vec::new();
        for _ in 0..64 {
            data.extend_from_slice(&[0u8, 255, 255, 0]);
        }
        let mut jpg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpg, 95)
            .encode(&data, 8, 8, jpeg_encoder::ColorType::Cmyk)
            .unwrap();
        let engine = Engine;
        let out = engine
            .process(
                &jpg,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Png)])],
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        let px = img.get_pixel(4, 4).0;
        assert!(
            px[0] > 150 && px[1] < 100 && px[2] < 100,
            "CMYK red came out as {px:?}"
        );
    }

    /// A JPEG with EXIF (orientation 6) and XMP, built with
    /// the production injector over an engine-encoded photo.
    fn meta_jpeg() -> Vec<u8> {
        let mut base = Vec::new();
        image::load_from_memory(&gradient_png())
            .unwrap()
            .write_to(
                &mut std::io::Cursor::new(&mut base),
                image::ImageFormat::Jpeg,
            )
            .unwrap();
        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"II");
        tiff.extend_from_slice(&42u16.to_le_bytes());
        tiff.extend_from_slice(&8u32.to_le_bytes());
        tiff.extend_from_slice(&1u16.to_le_bytes());
        tiff.extend_from_slice(&0x0112u16.to_le_bytes());
        tiff.extend_from_slice(&3u16.to_le_bytes());
        tiff.extend_from_slice(&1u32.to_le_bytes());
        tiff.extend_from_slice(&6u16.to_le_bytes());
        tiff.extend_from_slice(&0u16.to_le_bytes());
        tiff.extend_from_slice(&0u32.to_le_bytes());
        let meta = crate::color::SourceMeta {
            icc: None,
            exif_tiff: Some(tiff),
            xmp: Some(b"<x:xmpmeta>fixture</x:xmpmeta>".to_vec()),
            photoshop: None,
        };
        crate::color::inject_jpeg(&base, None, &meta, true)
    }

    #[test]
    fn metadata_is_stripped_by_default_and_preserved_on_policy() {
        let engine = Engine;
        let src = meta_jpeg();
        // Default: gone.
        let out = engine.process(&src, &[]).unwrap();
        let meta = crate::color::extract_source_meta(&out.bytes);
        assert!(meta.exif_tiff.is_none() && meta.xmp.is_none());
        // Preserve: both copied, orientation patched to 1
        // (the fixture's 6 already rotated the pixels).
        let out = engine
            .process_preferred(
                &src,
                &[],
                ProcessContext {
                    preserve_metadata: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let meta = crate::color::extract_source_meta(&out.bytes);
        assert_eq!(
            meta.xmp.as_deref(),
            Some(b"<x:xmpmeta>fixture</x:xmpmeta>".as_slice())
        );
        let tiff = meta.exif_tiff.expect("exif preserved");
        assert_eq!(
            u16::from_le_bytes(tiff[8 + 2 + 8..8 + 2 + 10].try_into().unwrap()),
            1
        );
    }

    #[test]
    fn save_data_steps_the_default_quality_and_reports_it() {
        let engine = Engine;
        let ctx = ProcessContext {
            save_data: true,
            ..Default::default()
        };
        // A JPEG with no quality directive encodes at the
        // stepped quality and reports it.
        let out = engine
            .process_preferred(
                &gradient_png(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jpeg)])],
                ctx,
            )
            .unwrap();
        assert_eq!(out.quality, Some(SAVE_DATA_QUALITY));
        // An explicit quality stands: nothing is reported,
        // exactly as without the hint.
        let out = engine
            .process_preferred(
                &gradient_png(),
                &[Operation::Filters(vec![
                    Filter::Format(ImageFormat::Jpeg),
                    Filter::Quality(Quality::Value(90)),
                ])],
                ProcessContext {
                    save_data: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.quality, None);
        // PNG has no quality to step.
        let out = engine
            .process_preferred(
                &gradient_png(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Png)])],
                ProcessContext {
                    save_data: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.quality, None);
    }

    #[test]
    fn explicit_quality_reports_nothing() {
        let engine = Engine;
        let out = engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![Filter::Quality(Quality::Value(
                    70,
                ))])],
            )
            .unwrap();
        assert_eq!(out.quality, None);
        assert_eq!(out.ssimulacra2, None);
    }

    #[test]
    fn auto_quality_jpeg_is_verified_against_the_target() {
        let engine = Engine;
        let ops = &[Operation::Filters(vec![
            Filter::Quality(Quality::Auto),
            Filter::Format(ImageFormat::Jpeg),
        ])];
        // The photo surrogate is ordinary content for the metric:
        // the loop must land it at or above the target.
        let out = engine.process(&photo_like_png(), ops).unwrap();
        let score = out.ssimulacra2.expect("auto JPEG is scored");
        assert!(
            (crate::quality::DEFAULT_SSIM_TARGET..=100.0).contains(&score),
            "score {score}"
        );
        assert!(out.quality.is_some_and(|q| (50..=95).contains(&q)));
        // The noisy fixture cannot reach the target at any allowed
        // quality; the loop still scores it and serves its best.
        let noisy = engine.process(&noise_png(), ops).unwrap();
        assert!(noisy.ssimulacra2.is_some());
    }

    #[test]
    fn auto_quality_jxl_is_verified_too() {
        let engine = Engine;
        let out = engine
            .process(
                &photo_like_png(),
                &[Operation::Filters(vec![
                    Filter::Quality(Quality::Auto),
                    Filter::Format(ImageFormat::Jxl),
                ])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jxl);
        let score = out.ssimulacra2.expect("auto JXL is scored");
        assert!(
            (crate::quality::DEFAULT_SSIM_TARGET..=100.0).contains(&score),
            "score {score}"
        );
    }

    #[test]
    fn auto_quality_webp_keeps_the_one_shot_pick() {
        let engine = Engine;
        let out = engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![
                    Filter::Quality(Quality::Auto),
                    Filter::Format(ImageFormat::WebP),
                ])],
            )
            .unwrap();
        // WebP is out of the verified scope for now: the heuristic
        // quality is reported, and nothing was scored.
        assert!(out.quality.is_some());
        assert_eq!(out.ssimulacra2, None);
    }

    /// 8x4 pattern with only the top-left quadrant white.
    fn quadrant_pattern() -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(8, 4);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let v = if x < 4 && y < 2 { 255 } else { 0 };
            *p = image::Rgb([v, v, v]);
        }
        img
    }

    /// Encode as JPEG and splice in an EXIF APP1 segment carrying
    /// `orientation`, the way a phone camera writes it.
    fn jpeg_with_orientation(img: &DynamicImage, orientation: u16) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 100)
            .write_image(
                img.as_bytes(),
                img.width(),
                img.height(),
                image::ExtendedColorType::from(img.color()),
            )
            .unwrap();
        let jpeg = buf.into_inner();
        assert_eq!(&jpeg[..2], b"\xff\xd8", "JPEG starts with SOI");

        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"II"); // little-endian TIFF
        tiff.extend_from_slice(&42u16.to_le_bytes());
        tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 offset
        tiff.extend_from_slice(&1u16.to_le_bytes()); // one entry
        tiff.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
        tiff.extend_from_slice(&3u16.to_le_bytes()); // SHORT
        tiff.extend_from_slice(&1u32.to_le_bytes()); // count
        tiff.extend_from_slice(&orientation.to_le_bytes());
        tiff.extend_from_slice(&[0, 0]); // value padding
        tiff.extend_from_slice(&0u32.to_le_bytes()); // next IFD

        let mut app1 = Vec::new();
        app1.extend_from_slice(b"Exif\0\0");
        app1.extend_from_slice(&tiff);

        let mut out = Vec::new();
        out.extend_from_slice(&jpeg[..2]);
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&app1);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    /// (width, height, [TL, TR, BL, BR] brightness) of an image.
    fn shape_and_quadrants(bytes: &[u8]) -> (u32, u32, [bool; 4]) {
        let img = image::load_from_memory(bytes).unwrap().to_luma8();
        let (w, h) = (img.width(), img.height());
        let bright = |x: u32, y: u32| img.get_pixel(x, y)[0] > 128;
        (
            w,
            h,
            [
                bright(w / 4, h / 4),
                bright(3 * w / 4, h / 4),
                bright(w / 4, 3 * h / 4),
                bright(3 * w / 4, 3 * h / 4),
            ],
        )
    }

    #[test]
    fn exif_orientations_come_out_upright() {
        let engine = Engine;
        // The stored pattern has its white quadrant top-left. Each
        // orientation value states where the stored pixels must move
        // for the image to display upright.
        type OrientationCase = (u16, (u32, u32, [bool; 4]));
        let cases: [OrientationCase; 8] = [
            (1, (8, 4, [true, false, false, false])),
            (2, (8, 4, [false, true, false, false])),
            (3, (8, 4, [false, false, false, true])),
            (4, (8, 4, [false, false, true, false])),
            (5, (4, 8, [true, false, false, false])),
            (6, (4, 8, [false, true, false, false])),
            (7, (4, 8, [false, false, false, true])),
            (8, (4, 8, [false, false, true, false])),
        ];
        for (orientation, expected) in cases {
            let src = jpeg_with_orientation(&quadrant_pattern(), orientation);
            let out = engine.process(&src, &[]).unwrap();
            assert_eq!(
                shape_and_quadrants(&out.bytes),
                expected,
                "orientation {orientation}"
            );
        }
    }

    #[test]
    fn jpeg_without_exif_is_untouched() {
        let engine = Engine;
        let img = quadrant_pattern();
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
        let out = engine.process(&buf.into_inner(), &[]).unwrap();
        assert_eq!(
            shape_and_quadrants(&out.bytes),
            (8, 4, [true, false, false, false])
        );
    }

    #[test]
    fn invalid_exif_orientation_is_ignored() {
        let engine = Engine;
        let src = jpeg_with_orientation(&quadrant_pattern(), 42);
        let out = engine.process(&src, &[]).unwrap();
        assert_eq!(
            shape_and_quadrants(&out.bytes),
            (8, 4, [true, false, false, false])
        );
    }

    #[test]
    fn avif_format_encodes() {
        let engine = Engine;
        let out = engine
            .process(
                &gradient_png(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Avif)])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Avif);
        // ISOBMFF layout: box size, then "ftyp", then the brand.
        assert_eq!(&out.bytes[4..8], b"ftyp");
        assert_eq!(&out.bytes[8..12], b"avif");
    }

    #[test]
    fn avif_sources_decode_or_fail_clearly() {
        // Encode a small AVIF with the always-available
        // encoder, then run it through the bounded decode.
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            24,
            16,
            image::Rgb([200, 40, 90]),
        ));
        let mut bytes = Vec::new();
        image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut bytes, 10, 100)
            .write_image(img.as_bytes(), 24, 16, image::ExtendedColorType::Rgb8)
            .unwrap();
        assert_eq!(&bytes[8..12], b"avif");
        #[cfg(feature = "avif-decode")]
        {
            let decoded = decode_image(&bytes).unwrap();
            assert_eq!((decoded.width(), decoded.height()), (24, 16));
            let px = decoded.to_rgb8().get_pixel(12, 8).0;
            assert!(
                px[0] > 150 && px[1] < 110 && px[2] > 50 && px[2] < 140,
                "decoded pixel {px:?} drifted from the source color"
            );
        }
        #[cfg(not(feature = "avif-decode"))]
        {
            let err = decode_image(&bytes).unwrap_err();
            let EngineError::Decode(message) = err else {
                panic!("expected a decode error, got {err:?}");
            };
            assert!(message.contains("avif-decode"), "{message}");
        }
    }

    #[test]
    fn avif_quality_changes_size() {
        let engine = Engine;
        let mk = |q| {
            engine
                .process(
                    &gradient_png(),
                    &[Operation::Filters(vec![
                        Filter::Quality(Quality::Value(q)),
                        Filter::Format(ImageFormat::Avif),
                    ])],
                )
                .unwrap()
        };
        assert!(mk(95).bytes.len() > mk(10).bytes.len());
    }

    #[test]
    fn preferred_format_applies_only_without_explicit_format() {
        let engine = Engine;
        let out = engine
            .process_preferred(
                &gradient_png(),
                &[],
                ProcessContext {
                    preferred: Some(ImageFormat::WebP),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::WebP);
        let out = engine
            .process_preferred(
                &gradient_png(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Png)])],
                ProcessContext {
                    preferred: Some(ImageFormat::WebP),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Png);
    }

    fn fit_in_op(w: Dimension, h: Dimension) -> Operation {
        Operation::FitIn {
            size: ImageSize {
                width: w,
                height: h,
                flip_horizontal: false,
                flip_vertical: false,
            },
            adaptive: false,
        }
    }

    fn fit_in_adaptive_op(w: Dimension, h: Dimension) -> Operation {
        Operation::FitIn {
            size: ImageSize {
                width: w,
                height: h,
                flip_horizontal: false,
                flip_vertical: false,
            },
            adaptive: true,
        }
    }

    fn dims_of(out: &ProcessedImage) -> (u32, u32) {
        let img = image::load_from_memory(&out.bytes).unwrap();
        (img.width(), img.height())
    }

    #[test]
    fn fit_in_fits_the_box_and_keeps_aspect() {
        let engine = Engine;
        let src = png_bytes(&DynamicImage::new_rgb8(400, 200));
        // Both bounds: the tighter one rules.
        let out = engine
            .process(
                &src,
                &[fit_in_op(Dimension::Pixels(100), Dimension::Pixels(100))],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (100, 50));
        // Single-axis bound.
        let out = engine
            .process(
                &src,
                &[fit_in_op(Dimension::Pixels(300), Dimension::Proportional)],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (300, 150));
        // Upscaling is allowed without no_upscale.
        let out = engine
            .process(
                &src,
                &[fit_in_op(Dimension::Pixels(1000), Dimension::Pixels(1000))],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (1000, 500));
    }

    #[test]
    fn fit_in_adaptive_swaps_the_box_when_orientations_disagree() {
        let engine = Engine;
        // Portrait 200x400 into a landscape 300x100 box: plain
        // fit-in is ruled by the height (50x100); adaptive fits
        // the swapped 100x300 box instead (100x200).
        let portrait = png_bytes(&DynamicImage::new_rgb8(200, 400));
        let out = engine
            .process(
                &portrait,
                &[fit_in_op(Dimension::Pixels(300), Dimension::Pixels(100))],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (50, 100));
        let out = engine
            .process(
                &portrait,
                &[fit_in_adaptive_op(
                    Dimension::Pixels(300),
                    Dimension::Pixels(100),
                )],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (100, 200));
        // Landscape 400x200 into a portrait 100x300 box: plain
        // is ruled by the width (100x50); adaptive fits the
        // swapped 300x100 box instead (200x100).
        let landscape = png_bytes(&DynamicImage::new_rgb8(400, 200));
        let out = engine
            .process(
                &landscape,
                &[fit_in_op(Dimension::Pixels(100), Dimension::Pixels(300))],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (100, 50));
        let out = engine
            .process(
                &landscape,
                &[fit_in_adaptive_op(
                    Dimension::Pixels(100),
                    Dimension::Pixels(300),
                )],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (200, 100));
    }

    #[test]
    fn fit_in_adaptive_matches_plain_fit_in_when_orientations_agree() {
        let engine = Engine;
        // Landscape source, landscape box: no swap, byte-for-byte
        // the plain behavior.
        let landscape = png_bytes(&DynamicImage::new_rgb8(400, 200));
        let plain = engine
            .process(
                &landscape,
                &[fit_in_op(Dimension::Pixels(300), Dimension::Pixels(100))],
            )
            .unwrap();
        let adaptive = engine
            .process(
                &landscape,
                &[fit_in_adaptive_op(
                    Dimension::Pixels(300),
                    Dimension::Pixels(100),
                )],
            )
            .unwrap();
        assert_eq!(dims_of(&adaptive), (200, 100));
        assert_eq!(adaptive.bytes, plain.bytes);
        // Portrait source, portrait box: no swap either.
        let portrait = png_bytes(&DynamicImage::new_rgb8(200, 400));
        let plain = engine
            .process(
                &portrait,
                &[fit_in_op(Dimension::Pixels(100), Dimension::Pixels(300))],
            )
            .unwrap();
        let adaptive = engine
            .process(
                &portrait,
                &[fit_in_adaptive_op(
                    Dimension::Pixels(100),
                    Dimension::Pixels(300),
                )],
            )
            .unwrap();
        assert_eq!(dims_of(&adaptive), (100, 200));
        assert_eq!(adaptive.bytes, plain.bytes);
    }

    #[test]
    fn fit_in_adaptive_never_swaps_square_or_open_boxes() {
        let engine = Engine;
        // Square source: no orientation, so the output is the
        // plain fit-in output.
        let square = png_bytes(&DynamicImage::new_rgb8(200, 200));
        let plain = engine
            .process(
                &square,
                &[fit_in_op(Dimension::Pixels(300), Dimension::Pixels(100))],
            )
            .unwrap();
        let adaptive = engine
            .process(
                &square,
                &[fit_in_adaptive_op(
                    Dimension::Pixels(300),
                    Dimension::Pixels(100),
                )],
            )
            .unwrap();
        assert_eq!(dims_of(&adaptive), (100, 100));
        assert_eq!(adaptive.bytes, plain.bytes);
        // Square box: swapping it would change nothing, and the
        // code does not swap it.
        let portrait = png_bytes(&DynamicImage::new_rgb8(200, 400));
        let plain = engine
            .process(
                &portrait,
                &[fit_in_op(Dimension::Pixels(100), Dimension::Pixels(100))],
            )
            .unwrap();
        let adaptive = engine
            .process(
                &portrait,
                &[fit_in_adaptive_op(
                    Dimension::Pixels(100),
                    Dimension::Pixels(100),
                )],
            )
            .unwrap();
        assert_eq!(dims_of(&adaptive), (50, 100));
        assert_eq!(adaptive.bytes, plain.bytes);
        // A box with a proportional axis has no fixed
        // orientation: no swap, the single bound rules.
        let plain = engine
            .process(
                &portrait,
                &[fit_in_op(Dimension::Pixels(300), Dimension::Proportional)],
            )
            .unwrap();
        let adaptive = engine
            .process(
                &portrait,
                &[fit_in_adaptive_op(
                    Dimension::Pixels(300),
                    Dimension::Proportional,
                )],
            )
            .unwrap();
        assert_eq!(dims_of(&adaptive), (300, 600));
        assert_eq!(adaptive.bytes, plain.bytes);
    }

    #[test]
    fn fit_in_box_swap_decision() {
        let px = |w: u32, h: u32| ImageSize {
            width: Dimension::Pixels(w),
            height: Dimension::Pixels(h),
            flip_horizontal: false,
            flip_vertical: false,
        };
        // Not adaptive: never swaps, whatever the orientations.
        assert_eq!(fit_in_box(&px(300, 100), 200, 400, false), px(300, 100));
        // Disagreement swaps in both directions.
        assert_eq!(fit_in_box(&px(300, 100), 200, 400, true), px(100, 300));
        assert_eq!(fit_in_box(&px(100, 300), 400, 200, true), px(300, 100));
        // Agreement keeps the box.
        assert_eq!(fit_in_box(&px(300, 100), 400, 200, true), px(300, 100));
        assert_eq!(fit_in_box(&px(100, 300), 200, 400, true), px(100, 300));
        // Equality on either side never swaps.
        assert_eq!(fit_in_box(&px(300, 100), 200, 200, true), px(300, 100));
        assert_eq!(fit_in_box(&px(100, 100), 200, 400, true), px(100, 100));
        // No two pixel axes, no swap.
        let open = ImageSize {
            width: Dimension::Pixels(300),
            height: Dimension::Proportional,
            flip_horizontal: false,
            flip_vertical: false,
        };
        assert_eq!(fit_in_box(&open, 200, 400, true), open);
        let orig = ImageSize {
            width: Dimension::Original,
            height: Dimension::Pixels(100),
            flip_horizontal: false,
            flip_vertical: false,
        };
        assert_eq!(fit_in_box(&orig, 200, 400, true), orig);
    }

    #[test]
    fn crop_rect_extracts_the_exact_region() {
        let engine = Engine;
        let src = png_bytes(&quadrant_pattern());
        let out = engine
            .process(
                &src,
                &[Operation::CropRect {
                    left: 0,
                    top: 0,
                    right: 4,
                    bottom: 2,
                }],
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
        assert_eq!((img.width(), img.height()), (4, 2));
        assert!(img.pixels().all(|p| p[0] == 255));
        let out = engine
            .process(
                &src,
                &[Operation::CropRect {
                    left: 4,
                    top: 2,
                    right: 8,
                    bottom: 4,
                }],
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
        assert!(img.pixels().all(|p| p[0] == 0));
    }

    #[test]
    fn crop_rect_out_of_bounds_is_an_error() {
        let engine = Engine;
        let src = png_bytes(&quadrant_pattern());
        let err = engine
            .process(
                &src,
                &[Operation::CropRect {
                    left: 0,
                    top: 0,
                    right: 9,
                    bottom: 4,
                }],
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    }

    fn focal_op(left: u32, top: u32, right: u32, bottom: u32) -> Operation {
        Operation::FocalRect {
            left,
            top,
            right,
            bottom,
        }
    }

    fn square_crop(align: Align) -> Operation {
        Operation::Crop {
            size: ImageSize {
                width: Dimension::Pixels(2),
                height: Dimension::Pixels(2),
                flip_horizontal: false,
                flip_vertical: false,
            },
            align,
        }
    }

    #[test]
    fn focal_smart_crop_keeps_the_named_region() {
        let engine = Engine;
        let src = png_bytes(&quadrant_pattern());
        // The focal rectangle names the black bottom-right of
        // the 8x4 pattern. A square crop window is 4 wide; the
        // rectangle forces it to x 4..8, all black.
        let out = engine
            .process(&src, &[focal_op(4, 2, 8, 4), square_crop(Align::Smart)])
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
        assert_eq!((img.width(), img.height()), (2, 2));
        assert!(img.pixels().all(|p| p[0] == 0));
        // Control: the same smart crop without the rectangle
        // keeps part of the white quadrant.
        let out = engine.process(&src, &[square_crop(Align::Smart)]).unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
        assert!(img.pixels().any(|p| p[0] > 0));
    }

    #[test]
    fn focal_center_crop_centers_on_the_rectangle() {
        let engine = Engine;
        let src = png_bytes(&quadrant_pattern());
        let out = engine
            .process(&src, &[focal_op(4, 2, 8, 4), square_crop(Align::Center)])
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
        assert_eq!((img.width(), img.height()), (2, 2));
        assert!(img.pixels().all(|p| p[0] == 0));
        // Control: a plain centered crop of the pattern keeps
        // part of the white quadrant.
        let out = engine.process(&src, &[square_crop(Align::Center)]).unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
        assert!(img.pixels().any(|p| p[0] > 0));
    }

    #[test]
    fn focal_is_ignored_by_a_directional_align() {
        let engine = Engine;
        let src = png_bytes(&quadrant_pattern());
        // Left alignment is an explicit instruction: the crop
        // takes the left half (white quadrant included) even
        // though the focal rectangle names the right side.
        let out = engine
            .process(&src, &[focal_op(4, 2, 8, 4), square_crop(Align::Left)])
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
        assert_eq!((img.width(), img.height()), (2, 2));
        assert!(img.pixels().any(|p| p[0] > 0));
    }

    #[test]
    fn focal_out_of_bounds_is_an_error() {
        let engine = Engine;
        let src = png_bytes(&quadrant_pattern());
        let err = engine
            .process(&src, &[focal_op(0, 0, 9, 4), square_crop(Align::Smart)])
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        let err = engine
            .process(&src, &[focal_op(0, 0, 8, 5), square_crop(Align::Smart)])
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    }

    #[test]
    fn focal_without_a_crop_is_a_no_op() {
        let engine = Engine;
        let src = png_bytes(&quadrant_pattern());
        let plain = engine.process(&src, &[]).unwrap();
        let with_focal = engine.process(&src, &[focal_op(1, 1, 7, 3)]).unwrap();
        assert_eq!(with_focal.bytes, plain.bytes);
        assert_eq!(dims_of(&with_focal), dims_of(&plain));
    }

    fn crop_op(w: u32, h: u32, align: Align) -> Operation {
        Operation::Crop {
            size: ImageSize {
                width: Dimension::Pixels(w),
                height: Dimension::Pixels(h),
                flip_horizontal: false,
                flip_vertical: false,
            },
            align,
        }
    }

    fn debug_px(out: &ProcessedImage, x: u32, y: u32) -> image::Rgba<u8> {
        *image::load_from_memory(&out.bytes)
            .unwrap()
            .to_rgba8()
            .get_pixel(x, y)
    }

    /// 100x100 gray with a 20x20 checker block at (70, 40):
    /// the saliency fixture the focus tests use, so the focal
    /// point is deterministic (near the block's center).
    fn debug_block_image() -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(100, 100);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if (70..90).contains(&x) && (40..60).contains(&y) {
                if (x + y) % 2 == 0 {
                    image::Rgb([200, 200, 200])
                } else {
                    image::Rgb([50, 50, 50])
                }
            } else {
                image::Rgb([128, 128, 128])
            };
        }
        img
    }

    #[test]
    fn debug_focal_draws_the_smart_crop_decision() {
        let engine = Engine;
        let img = debug_block_image();
        let src = png_bytes(&img);
        let point = crate::focus::focal_point(&img);
        let (fx, fy) = (
            (point.x * 100.0).round() as u32,
            (point.y * 100.0).round() as u32,
        );
        let out = engine
            .process(
                &src,
                &[Operation::DebugFocal, crop_op(50, 100, Align::Smart)],
            )
            .unwrap();
        // The response is the 100x100 canvas the crop ran
        // on, not the 50x100 crop result.
        assert_eq!(dims_of(&out), (100, 100));
        // The crosshair sits on the focal point the crop used.
        assert_eq!(debug_px(&out, fx, fy), crate::debugview::FOCAL_COLOR);
        // The window outline matches the window the same URL
        // without the debug segment cuts: 50 wide, positioned
        // on the focal point, full height.
        let start = ((point.x * 100.0).round() as i32 - 25).clamp(0, 50) as u32;
        assert_eq!(debug_px(&out, start, 0), crate::debugview::WINDOW_COLOR);
        assert_eq!(
            debug_px(&out, start + 49, 99),
            crate::debugview::WINDOW_COLOR
        );
        let plain = engine
            .process(&src, &[crop_op(50, 100, Align::Smart)])
            .unwrap();
        assert_eq!(dims_of(&plain), (50, 100));
        // An unmarked pixel keeps the canvas value.
        assert_eq!(debug_px(&out, 10, 10), image::Rgba([128, 128, 128, 255]));
    }

    #[test]
    fn debug_focal_draws_the_caller_focal_rectangle() {
        let engine = Engine;
        let src = png_bytes(&quadrant_pattern());
        let out = engine
            .process(
                &src,
                &[
                    Operation::DebugFocal,
                    focal_op(5, 2, 7, 4),
                    square_crop(Align::Smart),
                ],
            )
            .unwrap();
        // The canvas is the 8x4 pattern, not the 2x2 result.
        assert_eq!(dims_of(&out), (8, 4));
        // The focal rectangle's top edge, clear of the window.
        assert_eq!(debug_px(&out, 6, 2), crate::debugview::FOCAL_COLOR);
        // The crosshair at the rectangle's center (6, 3) is
        // drawn last, over the window's bottom band.
        assert_eq!(debug_px(&out, 6, 3), crate::debugview::FOCAL_COLOR);
        // The crop window is x 4..8, full height (the same
        // window the plain focal test crops to all black).
        assert_eq!(debug_px(&out, 4, 0), crate::debugview::WINDOW_COLOR);
        assert_eq!(debug_px(&out, 7, 0), crate::debugview::WINDOW_COLOR);
        // The same chain without the debug segment still
        // produces the plain 2x2 crop.
        let plain = engine
            .process(&src, &[focal_op(5, 2, 7, 4), square_crop(Align::Smart)])
            .unwrap();
        assert_eq!(dims_of(&plain), (2, 2));
    }

    #[test]
    fn debug_focal_without_a_crop_marks_the_final_image() {
        let engine = Engine;
        let img = debug_block_image();
        let src = png_bytes(&img);
        let point = crate::focus::focal_point(&img);
        let (fx, fy) = (
            (point.x * 100.0).round() as u32,
            (point.y * 100.0).round() as u32,
        );
        let out = engine.process(&src, &[Operation::DebugFocal]).unwrap();
        assert_eq!(dims_of(&out), (100, 100));
        assert_eq!(debug_px(&out, fx, fy), crate::debugview::FOCAL_COLOR);
        // No crop ran, so no window outline exists anywhere.
        let rgba = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
        assert!(rgba.pixels().all(|p| *p != crate::debugview::WINDOW_COLOR));
        // And the diagnostic is not the plain result.
        let plain = engine.process(&src, &[]).unwrap();
        assert_ne!(out.bytes, plain.bytes);
    }

    #[test]
    fn debug_focal_on_an_animated_source_is_a_specific_error() {
        let engine = Engine;
        for src in [animated_gif_bytes(), animated_webp_bytes()] {
            let err = engine
                .process_preferred(&src, &[Operation::DebugFocal], ProcessContext::default())
                .unwrap_err();
            assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
            assert!(err.to_string().contains("debug:focal"), "{err}");
            assert!(err.to_string().contains("animated"), "{err}");
        }
    }

    #[test]
    fn debug_focal_with_a_video_format_is_a_specific_error() {
        let engine = Engine;
        let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
        let src = png_bytes(&DynamicImage::new_rgb8(20, 12));
        let err = engine
            .process_preferred(
                &src,
                &[
                    Operation::DebugFocal,
                    Operation::Filters(vec![Filter::Format(ImageFormat::Mp4)]),
                ],
                ProcessContext {
                    video_encoder: Some(&encoder),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("debug:focal"), "{err}");
        assert!(err.to_string().contains("mp4"), "{err}");
    }

    #[test]
    fn no_upscale_caps_every_resize_mode() {
        let engine = Engine;
        let src = png_bytes(&DynamicImage::new_rgb8(200, 200));
        let directive = Operation::Filters(vec![Filter::NoUpscale]);
        // Exact size, with the directive after the size on purpose:
        // the cap is request-wide, not positional. Each axis
        // clamps at the source dimension (issue #230).
        let out = engine
            .process(
                &src,
                &[
                    size_op(Dimension::Pixels(1000), Dimension::Pixels(500)),
                    directive.clone(),
                ],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (200, 200));
        // The README's example: a 300x200 resize of a 200x150
        // source returns the source at its own dimensions, not
        // a proportional shrink to 200x133.
        let wide = png_bytes(&DynamicImage::new_rgb8(200, 150));
        let out = engine
            .process(
                &wide,
                &[
                    size_op(Dimension::Pixels(300), Dimension::Pixels(200)),
                    directive.clone(),
                ],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (200, 150));
        // A mixed request clamps only the axis that would
        // enlarge; the shrinking axis is honored as written.
        let out = engine
            .process(
                &wide,
                &[
                    size_op(Dimension::Pixels(300), Dimension::Pixels(100)),
                    directive.clone(),
                ],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (200, 100));
        // Fit-in.
        let out = engine
            .process(
                &src,
                &[
                    directive.clone(),
                    fit_in_op(Dimension::Pixels(1000), Dimension::Pixels(1000)),
                ],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (200, 200));
        // Crop (aspect crop plus exact resize).
        let out = engine
            .process(
                &src,
                &[
                    directive,
                    Operation::Crop {
                        size: ImageSize {
                            width: Dimension::Pixels(500),
                            height: Dimension::Pixels(500),
                            flip_horizontal: false,
                            flip_vertical: false,
                        },
                        align: Align::Center,
                    },
                ],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (200, 200));
        // Sanity: without the directive the same size upscales freely.
        let out = engine
            .process(
                &src,
                &[size_op(Dimension::Pixels(1000), Dimension::Pixels(500))],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (1000, 500));
    }

    #[test]
    fn max_bytes_fits_the_budget() {
        let engine = Engine;
        let src = noise_png();
        let control = engine
            .process(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jpeg)])],
            )
            .unwrap();
        assert!(
            control.bytes.len() > 5000,
            "control {} bytes",
            control.bytes.len()
        );
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Format(ImageFormat::Jpeg),
                    Filter::MaxBytes(5000),
                ])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jpeg);
        assert!(out.bytes.len() <= 5000, "{} bytes", out.bytes.len());
        // The reduction is reported as the chosen quality.
        let q = out.quality.expect("reduced quality is reported");
        assert!(q < 80, "quality {q}");
    }

    fn webp_chunk(bytes: &[u8]) -> &'static str {
        if bytes.windows(4).any(|c| c == b"VP8L") {
            "VP8L"
        } else if bytes.windows(4).any(|c| c == b"VP8 ") {
            "VP8 "
        } else {
            "?"
        }
    }

    #[test]
    fn webp_encodes_lossy_by_default() {
        let engine = Engine;
        let src = noise_png();
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::WebP)])],
            )
            .unwrap();
        assert_eq!(webp_chunk(&out.bytes), "VP8 ");
        assert_eq!(decode_dims(&out.bytes), (120, 120));
        // Lossy on noise is far smaller than the lossless encode.
        let lossless = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Format(ImageFormat::WebP),
                    Filter::Quality(Quality::Value(100)),
                ])],
            )
            .unwrap();
        assert_eq!(webp_chunk(&lossless.bytes), "VP8L");
        assert!(
            out.bytes.len() * 2 < lossless.bytes.len(),
            "lossy {} vs lossless {}",
            out.bytes.len(),
            lossless.bytes.len()
        );
    }

    #[test]
    fn webp_quality_100_is_lossless_and_exact() {
        let engine = Engine;
        let src = noise_png();
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Format(ImageFormat::WebP),
                    Filter::Quality(Quality::Value(100)),
                ])],
            )
            .unwrap();
        let decoded = image::load_from_memory(&out.bytes).unwrap();
        let original = image::load_from_memory(&src).unwrap();
        assert_eq!(decoded.to_rgb8(), original.to_rgb8());
    }

    #[test]
    fn webp_lossy_size_tracks_quality() {
        let engine = Engine;
        let src = noise_png();
        let len = |q: u8| {
            engine
                .process(
                    &src,
                    &[Operation::Filters(vec![
                        Filter::Format(ImageFormat::WebP),
                        Filter::Quality(Quality::Value(q)),
                    ])],
                )
                .unwrap()
                .bytes
                .len()
        };
        assert!(len(50) < len(95), "q50 {} vs q95 {}", len(50), len(95));
    }

    #[test]
    fn webp_lossy_preserves_alpha() {
        let engine = Engine;
        let mut rgba = image::RgbaImage::from_pixel(16, 16, image::Rgba([200, 30, 30, 255]));
        for x in 0..8 {
            for y in 0..16 {
                rgba.put_pixel(x, y, image::Rgba([30, 30, 200, 0]));
            }
        }
        let src = png_bytes(&DynamicImage::ImageRgba8(rgba));
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Format(ImageFormat::WebP),
                    Filter::Quality(Quality::Value(80)),
                ])],
            )
            .unwrap();
        let decoded = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
        assert_eq!(decoded.get_pixel(12, 8)[3], 255);
        assert_eq!(decoded.get_pixel(4, 8)[3], 0);
    }

    #[test]
    fn webp_max_bytes_uses_the_quality_ladder() {
        let engine = Engine;
        let src = noise_png();
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Format(ImageFormat::WebP),
                    Filter::MaxBytes(4000),
                ])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::WebP);
        assert!(out.bytes.len() <= 4000, "{} bytes", out.bytes.len());
    }

    /// The sampling factors from a JPEG's SOF0 segment, per
    /// component: (id, sampling byte). 4:2:0 shows Y as 0x22.
    fn jpeg_sof_sampling(bytes: &[u8]) -> Vec<(u8, u8)> {
        let mut i = 2;
        while i + 9 < bytes.len() {
            if bytes[i] != 0xFF {
                i += 1;
                continue;
            }
            let marker = bytes[i + 1];
            if marker == 0xC0 {
                let n = bytes[i + 9] as usize;
                return (0..n)
                    .map(|c| (bytes[i + 10 + c * 3], bytes[i + 11 + c * 3]))
                    .collect();
            }
            if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
                i += 2;
                continue;
            }
            let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            i += 2 + len;
        }
        panic!("no SOF0 segment");
    }

    #[test]
    fn jpeg_uses_420_subsampling_and_stays_compact() {
        let engine = Engine;
        let src = noise_png();
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jpeg)])],
            )
            .unwrap();
        let sampling = jpeg_sof_sampling(&out.bytes);
        assert_eq!(sampling.len(), 3);
        assert_eq!(sampling[0].1, 0x22, "Y sampling (4:2:0)");
        assert_eq!(sampling[1].1, 0x11, "Cb sampling");
        // Regression bound from the #38 encoder swap: this
        // fixture encoded to 20,746 B with the previous encoder
        // and 12,069 B with jpeg-encoder at 4:2:0.
        assert!(out.bytes.len() < 14000, "{} bytes", out.bytes.len());
        assert_eq!(decode_dims(&out.bytes), (120, 120));
    }

    #[test]
    fn resize_backend_matches_lanczos3_within_rounding() {
        // Same filter (Lanczos3) in both backends, so outputs
        // track closely: on photo content the spike measured a
        // max per-channel diff of 1 (RMS 0.14). This pure-noise
        // fixture is the adversarial case for kernel precision:
        // measured RMS 0.48 with a tail to 9 (0.3% of channels
        // over 2). The bounds below sit well above that and far
        // below any real breakage (a wrong filter or stride
        // pushes RMS into the tens).
        let src = image::load_from_memory(&noise_png()).unwrap();
        let ours = resize_lanczos3(&src, 60, 60).to_rgb8();
        let reference =
            image::imageops::resize(&src, 60, 60, image::imageops::FilterType::Lanczos3);
        let (mut maxd, mut sq) = (0u8, 0f64);
        for (a, b) in ours.pixels().zip(reference.pixels()) {
            for c in 0..3 {
                let d = a[c].abs_diff(b[c]);
                maxd = maxd.max(d);
                sq += (d as f64) * (d as f64);
            }
        }
        let rms = (sq / (60.0 * 60.0 * 3.0)).sqrt();
        assert!(rms < 1.0, "rms diff {rms}");
        assert!(maxd <= 10, "max diff {maxd}");
        // The source pixel type is preserved, not forced to RGBA.
        assert!(matches!(
            resize_lanczos3(&src, 30, 30),
            DynamicImage::ImageRgb8(_)
        ));
    }

    #[test]
    fn resize_backend_does_not_bleed_transparent_colors() {
        // Left half transparent red, right half opaque blue. With
        // premultiplied-alpha resampling, the nearly transparent
        // boundary pixel takes its color from the opaque side;
        // straight-alpha resampling would tint it red.
        let mut img = image::RgbaImage::from_pixel(64, 64, image::Rgba([255, 0, 0, 0]));
        for x in 32..64 {
            for y in 0..64 {
                img.put_pixel(x, y, image::Rgba([0, 0, 255, 255]));
            }
        }
        let out = resize_lanczos3(&DynamicImage::ImageRgba8(img), 16, 16).to_rgba8();
        let edge = out.get_pixel(7, 8);
        assert_eq!(edge[0], 0, "red bled into the boundary pixel: {edge:?}");
        assert!(edge[2] > 200, "blue contribution lost: {edge:?}");
        assert!(edge[3] < 64, "edge alpha should be small: {edge:?}");
    }

    #[test]
    fn fill_flattens_transparency_for_jpeg() {
        let engine = Engine;
        // Fully transparent green: RGB stored under zero alpha.
        let rgba = image::RgbaImage::from_pixel(16, 16, image::Rgba([0, 255, 0, 0]));
        let src = png_bytes(&DynamicImage::ImageRgba8(rgba));
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Format(ImageFormat::Jpeg),
                    Filter::Fill([0, 0, 255]),
                ])],
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        let p = img.get_pixel(8, 8);
        assert!(p[2] > 200 && p[1] < 60, "pixel {p:?}");
        // Without fill, the stored green shows through.
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jpeg)])],
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        let p = img.get_pixel(8, 8);
        assert!(p[1] > 200, "pixel {p:?}");
    }

    fn gif_bytes(img: &DynamicImage) -> Vec<u8> {
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Gif)
            .unwrap();
        out
    }

    /// Two 40x20 frames: red, then blue.
    fn animated_gif_bytes() -> Vec<u8> {
        let red = image::RgbaImage::from_pixel(40, 20, image::Rgba([255, 0, 0, 255]));
        let blue = image::RgbaImage::from_pixel(40, 20, image::Rgba([0, 0, 255, 255]));
        let delay = image::Delay::from_numer_denom_ms(100, 1);
        let frames = vec![
            image::Frame::from_parts(red, 0, 0, delay),
            image::Frame::from_parts(blue, 0, 0, delay),
        ];
        let mut out = Vec::new();
        image::codecs::gif::GifEncoder::new(&mut out)
            .encode_frames(frames)
            .unwrap();
        out
    }

    #[test]
    fn decode_image_rejects_oversized_dimensions() {
        let big = DynamicImage::ImageRgb8(image::RgbImage::new(17_000, 2));
        let bytes = png_bytes(&big);
        let err = decode_image(&bytes).unwrap_err();
        assert!(matches!(err, EngineError::Decode(_)), "{err:?}");
    }

    #[test]
    fn gif_frame_cap_rejects_frame_bombs() {
        let make = |n: usize| {
            let delay = image::Delay::from_numer_denom_ms(100, 1);
            let frames: Vec<image::Frame> = (0..n)
                .map(|_| image::Frame::from_parts(image::RgbaImage::new(4, 4), 0, 0, delay))
                .collect();
            let mut out = Vec::new();
            image::codecs::gif::GifEncoder::new(&mut out)
                .encode_frames(frames)
                .unwrap();
            out
        };
        assert!(decode_gif_frames(&make(201)).is_err());
        assert_eq!(decode_gif_frames(&make(3)).unwrap().len(), 3);
    }

    #[test]
    fn output_pixel_budget_rejects_huge_requests() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(64, 64));
        let size = ImageSize {
            width: Dimension::Pixels(16_383),
            height: Dimension::Pixels(16_383),
            flip_horizontal: false,
            flip_vertical: false,
        };
        let err = apply_size(img.clone(), &size, None, 1.0).unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        let err = apply_pad(img, 16_383, 16_383, PadFill::Directive, None, None).unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    }

    #[test]
    fn output_budget_charges_frames_against_one_total() {
        let mut budget = OutputBudget::with_limit(250);
        assert!(budget.charge(10, 10).is_ok());
        assert!(budget.charge(10, 10).is_ok());
        // Each frame is inside the limit on its own; the
        // running total is what passes it (issue #182).
        let err = budget.charge(10, 10).unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        // A single frame can also blow the budget by itself.
        let mut budget = OutputBudget::with_limit(250);
        assert!(budget.charge(20, 20).is_err());
    }

    /// Two 100x100 frames, tiny on every input cap. Padded
    /// (with upscale capped, so the pad is a canvas fill,
    /// not a resample) to 10000x6000, each frame is 60M
    /// output pixels: individually inside the per-frame
    /// budget, 120M in sum, past the one shared budget.
    fn aggregate_overflow_gif() -> Vec<u8> {
        let delay = image::Delay::from_numer_denom_ms(100, 1);
        let frames: Vec<image::Frame> = (0..2)
            .map(|_| image::Frame::from_parts(image::RgbaImage::new(100, 100), 0, 0, delay))
            .collect();
        let mut src = Vec::new();
        image::codecs::gif::GifEncoder::new(&mut src)
            .encode_frames(frames)
            .unwrap();
        src
    }

    fn aggregate_overflow_ops() -> Vec<Operation> {
        vec![
            Operation::Filters(vec![Filter::NoUpscale]),
            Operation::Pad {
                size: ImageSize {
                    width: Dimension::Pixels(10_000),
                    height: Dimension::Pixels(6_000),
                    flip_horizontal: false,
                    flip_vertical: false,
                },
                fill: PadFill::Color([0, 0, 0]),
            },
        ]
    }

    #[test]
    fn animated_output_budget_rejects_aggregate_overflow() {
        // Issue #182, audit R2-01: the frames pass every
        // per-frame check; only the aggregate charge refuses.
        let engine = Engine;
        let err = engine
            .process(&aggregate_overflow_gif(), &aggregate_overflow_ops())
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("across frames"), "{err}");
    }

    #[test]
    fn video_output_budget_rejects_aggregate_overflow() {
        // The same aggregate budget guards the video path:
        // the request fails before the encoder ever runs.
        let engine = Engine;
        let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
        let mut ops = aggregate_overflow_ops();
        ops.push(Operation::Filters(vec![Filter::Format(ImageFormat::Mp4)]));
        let err = engine
            .process_preferred(
                &aggregate_overflow_gif(),
                &ops,
                ProcessContext {
                    video_encoder: Some(&encoder),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("across frames"), "{err}");
    }

    #[test]
    fn rotate_canvas_is_checked_before_it_allocates() {
        // An 8000x8000 image is a legal size; a 45 degree
        // rotation would land on a 11314x11314 canvas (128M
        // px), past the output budget. The chain is rejected
        // by the canvas check, before the rotation runs
        // (issue #182, audit R2-12).
        let img = DynamicImage::new_rgba8(8000, 8000);
        let ops = [Operation::Filters(vec![Filter::Rotate(45.0)])];
        let directives = crate::url::RequestDirectives::analyze(&ops);
        let ctx = ProcessContext::default();
        let err = apply_operations(img.clone(), &ops, None, 1.0, &ctx, &directives).unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        // A quarter turn keeps the pixel count and passes.
        let ops = [Operation::Filters(vec![Filter::Rotate(90.0)])];
        let directives = crate::url::RequestDirectives::analyze(&ops);
        let (out, _, _) = apply_operations(img, &ops, None, 1.0, &ctx, &directives).unwrap();
        assert_eq!((out.width(), out.height()), (8000, 8000));
    }

    #[test]
    fn rotate_chains_inside_the_budget_still_work() {
        let engine = Engine;
        let out = engine
            .process(
                &solid_png(100, 100, [200, 30, 30]),
                &[Operation::Filters(vec![
                    Filter::Rotate(45.0),
                    Filter::Rotate(45.0),
                ])],
            )
            .unwrap();
        assert_eq!(decode_dims(&out.bytes), (201, 201));
    }

    #[test]
    fn operation_count_cap_rejects_overlong_chains() {
        let engine = Engine;
        let src = solid_png(4, 4, [1, 2, 3]);
        // One segment carrying MAX_OPERATIONS + 1 cheap
        // filters is over: filters count individually.
        let one_segment = vec![Operation::Filters(vec![Filter::Invert; MAX_OPERATIONS + 1])];
        let err = engine.process(&src, &one_segment).unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        // So are MAX_OPERATIONS + 1 single-filter segments.
        let many_segments: Vec<Operation> = (0..=MAX_OPERATIONS)
            .map(|_| Operation::Filters(vec![Filter::Invert]))
            .collect();
        assert!(engine.process(&src, &many_segments).is_err());
        // Exactly at the cap still processes.
        let at_cap: Vec<Operation> = (0..MAX_OPERATIONS)
            .map(|_| Operation::Filters(vec![Filter::Invert]))
            .collect();
        assert!(engine.process(&src, &at_cap).is_ok());
    }

    #[test]
    fn jxl_dimension_probe_accepts_small_and_rejects_huge() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(24, 16));
        let bytes = encode(
            &EncodeBuffers::new(&img),
            ImageFormat::Jxl,
            Some(80),
            None,
            DEFAULT_AVIF_SPEED,
            DEFAULT_JXL_SPEED,
        )
        .unwrap();
        assert!(check_jxl_dimensions(&bytes).is_ok());
        assert!(check_decoded_dimensions(20_000, 10).is_err());
        assert!(check_decoded_dimensions(16_384, 16_384).is_ok());
    }

    fn decoded_gif_frames(bytes: &[u8]) -> Vec<image::Frame> {
        use image::AnimationDecoder;
        let decoder = image::codecs::gif::GifDecoder::new(Cursor::new(bytes)).unwrap();
        decoder.into_frames().collect_frames().unwrap()
    }

    #[test]
    fn static_gif_round_trips_as_gif() {
        let engine = Engine;
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            24,
            16,
            image::Rgb([10, 200, 30]),
        ));
        let src = gif_bytes(&img);
        let out = engine.process(&src, &[]).unwrap();
        assert_eq!(out.format, ImageFormat::Gif);
        assert_eq!(&out.bytes[..6], b"GIF89a");
        assert_eq!(dims_of(&out), (24, 16));
    }

    #[test]
    fn format_gif_converts_a_png_source() {
        let engine = Engine;
        let src = png_bytes(&DynamicImage::new_rgb8(20, 12));
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Gif)])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Gif);
        assert_eq!(&out.bytes[..6], b"GIF89a");
    }

    #[test]
    fn animated_gif_keeps_its_frames_through_resize() {
        let engine = Engine;
        let src = animated_gif_bytes();
        let out = engine
            .process(
                &src,
                &[size_op(Dimension::Pixels(20), Dimension::Pixels(10))],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Gif);
        let frames = decoded_gif_frames(&out.bytes);
        assert_eq!(frames.len(), 2);
        assert!(
            frames
                .iter()
                .all(|f| f.buffer().width() == 20 && f.buffer().height() == 10)
        );
    }

    #[test]
    fn animated_gif_to_a_still_format_keeps_the_first_frame() {
        let engine = Engine;
        let src = animated_gif_bytes();
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Png)])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Png);
        // The first frame is solid red.
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        assert_eq!(img.get_pixel(0, 0), &image::Rgb([255, 0, 0]));
    }

    #[test]
    fn animated_gif_stays_out_of_jxl_negotiation() {
        let engine = Engine;
        let src = animated_gif_bytes();
        // A negotiated (preferred) JXL falls back to GIF so the
        // frames survive.
        let out = engine
            .process_preferred(
                &src,
                &[],
                ProcessContext {
                    preferred: Some(ImageFormat::Jxl),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Gif);
        assert_eq!(decoded_gif_frames(&out.bytes).len(), 2);
        // An explicit format(jxl) still converts, first frame only.
        let out = engine
            .process_preferred(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jxl)])],
                ProcessContext {
                    preferred: Some(ImageFormat::Jxl),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jxl);
    }

    /// A fake video encoder for the video tests: drains
    /// stdin and answers with the script's canned behavior.
    fn video_test_encoder(script: &str) -> crate::video::VideoEncoder {
        use std::io::Write as _;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("crabcrop-engine-video-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("encoder.sh");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(script.as_bytes()).unwrap();
        drop(file);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        crate::video::VideoEncoder::new(path, std::time::Duration::from_secs(30), 64 * 1024 * 1024)
            .unwrap()
    }

    const FAKE_VIDEO_SCRIPT: &str = "#!/bin/sh\ncat > /dev/null\nprintf 'ENGINE-VIDEO'\n";

    /// A fake frame extractor (issue #214): its probe
    /// prints `report` on stderr, its extraction prints a
    /// real 8x6 PNG. Returns the extractor (the script
    /// lives in a temp dir keyed by `name`).
    fn frame_test_extractor(name: &str, report: &str) -> crate::frames::VideoFrameExtractor {
        use std::io::Write as _;
        let dir = std::env::temp_dir().join(format!(
            "crabcrop-engine-frame-{}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("frame.png"),
            png_bytes(&DynamicImage::new_rgb8(8, 6)),
        )
        .unwrap();
        let frame_path = dir.join("frame.png").display().to_string();
        let script = format!(
            "#!/bin/sh\ncat > /dev/null\nfor a in \"$@\"; do\n  if [ \"$a\" = \"image2pipe\" ]; then cat \"{frame_path}\"; exit 0; fi\ndone\nprintf '%s\\n' {report} >&2\nexit 1\n"
        );
        let path = dir.join("decoder.sh");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(script.as_bytes()).unwrap();
        drop(file);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        crate::frames::VideoFrameExtractor::new(path, std::time::Duration::from_secs(30)).unwrap()
    }

    /// Bytes that sniff as no image: an ISO-BMFF header
    /// fragment is enough (the extractor is fake; the
    /// container is never really parsed in these tests).
    const FAKE_VIDEO_BYTES: &[u8] = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2avc1";

    const FAKE_PROBE_REPORT: &str = "'  Duration: 00:00:03.00, start: 0.000000' '  Stream #0:0: Video: h264, yuv420p, 8x6, 10 fps'";

    #[test]
    fn frame_segment_without_a_decoder_is_not_configured() {
        let engine = Engine;
        let err = engine
            .process_preferred(
                FAKE_VIDEO_BYTES,
                &[Operation::Frame(crate::url::FrameSelection::Poster)],
                ProcessContext::default(),
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
        assert!(err.to_string().contains("CRABCROP_VIDEO_DECODER"), "{err}");
    }

    #[test]
    fn frame_segment_on_an_image_source_is_a_decode_failure() {
        let engine = Engine;
        let extractor = frame_test_extractor("image-source", FAKE_PROBE_REPORT);
        let ctx = ProcessContext {
            frame_extractor: Some(&extractor),
            ..Default::default()
        };
        // A still PNG and an animated GIF: frame selection
        // is video-only, and animated image sources keep
        // their own handling, so both refuse as bad
        // sources rather than yielding a frame.
        let sources = [
            png_bytes(&DynamicImage::new_rgb8(20, 12)),
            animated_gif_bytes(),
        ];
        for src in sources {
            let err = engine
                .process_preferred(
                    &src,
                    &[Operation::Frame(crate::url::FrameSelection::Poster)],
                    ctx.clone(),
                )
                .unwrap_err();
            assert!(matches!(err, EngineError::Decode(_)), "{err:?}");
            assert!(err.to_string().contains("needs a video source"), "{err}");
        }
    }

    #[test]
    fn frame_segment_extracts_and_processes_the_frame() {
        let engine = Engine;
        let extractor = frame_test_extractor("extract", FAKE_PROBE_REPORT);
        let out = engine
            .process_preferred(
                FAKE_VIDEO_BYTES,
                &[
                    Operation::Frame(crate::url::FrameSelection::Seconds(1.5)),
                    size_op(Dimension::Pixels(4), Dimension::Pixels(3)),
                ],
                ProcessContext {
                    frame_extractor: Some(&extractor),
                    ..Default::default()
                },
            )
            .unwrap();
        // The operation chain ran on the extracted frame.
        assert_eq!((out.width, out.height), (4, 3));
        assert_eq!((out.source_width, out.source_height), (8, 6));
        // And the video record rides along for meta.
        let video = out.video.expect("a frame render carries the video record");
        assert_eq!(video.duration_seconds, Some(3.0));
        assert_eq!((video.width, video.height), (8, 6));
        assert_eq!(video.frame_seconds, 1.5);
    }

    #[test]
    fn frame_segment_on_a_streamless_source_is_a_decode_failure() {
        let engine = Engine;
        let extractor = frame_test_extractor("streamless", "'no streams in here at all'");
        let err = engine
            .process_preferred(
                FAKE_VIDEO_BYTES,
                &[Operation::Frame(crate::url::FrameSelection::Poster)],
                ProcessContext {
                    frame_extractor: Some(&extractor),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Decode(_)), "{err:?}");
        assert!(err.to_string().contains("no video stream"), "{err}");
    }

    #[test]
    fn probe_source_labels_videos_and_never_mislabels_avif() {
        // The explain probe (issue #214): a video
        // container is kind "video" with unknown
        // dimensions; an AVIF still shares the ISO-BMFF
        // box and must keep its own kind.
        let probe = probe_source(FAKE_VIDEO_BYTES);
        assert_eq!(probe.kind, "video");
        assert_eq!(probe.dimensions, None);
        let webm = [0x1A, 0x45, 0xDF, 0xA3, 1, 2, 3, 4];
        assert_eq!(probe_source(&webm).kind, "video");
        let mut avif = Vec::new();
        avif.extend_from_slice(&24u32.to_be_bytes());
        avif.extend_from_slice(b"ftypavif");
        avif.extend_from_slice(&[0, 0, 0, 0]);
        avif.extend_from_slice(b"avifmif1");
        assert_eq!(probe_source(&avif).kind, "avif");
    }

    #[test]
    fn frame_percentage_without_a_duration_is_unsupported() {
        let engine = Engine;
        let extractor = frame_test_extractor(
            "no-duration",
            "'  Duration: N/A, start: 0.000000' '  Stream #0:0: Video: h264, yuv420p, 8x6, 10 fps'",
        );
        let err = engine
            .process_preferred(
                FAKE_VIDEO_BYTES,
                &[Operation::Frame(crate::url::FrameSelection::Percent(50.0))],
                ProcessContext {
                    frame_extractor: Some(&extractor),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("duration"), "{err}");
    }

    /// A fake PDF renderer pair (issue #215): the probe
    /// prints a report naming `pages` pages, the renderer
    /// prints a real 8x6 PNG. Returns the renderer (the
    /// scripts live in a temp dir keyed by `name`).
    fn pdf_test_renderer(name: &str, pages: u32) -> crate::pdf::PdfRenderer {
        use std::io::Write as _;
        let dir =
            std::env::temp_dir().join(format!("crabcrop-engine-pdf-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("page.png"),
            png_bytes(&DynamicImage::new_rgb8(8, 6)),
        )
        .unwrap();
        let page_path = dir.join("page.png").display().to_string();
        let write = |file: &str, script: String| {
            let path = dir.join(file);
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(script.as_bytes()).unwrap();
            drop(f);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            path
        };
        write(
            "pdfinfo",
            format!(
                "#!/bin/sh\ncat > /dev/null\nprintf '%s\\n' 'Pages:           {pages}' 'Encrypted:       no' 'Page size:       612 x 792 pts (letter)'\nexit 0\n"
            ),
        );
        let renderer = write(
            "pdftocairo",
            format!("#!/bin/sh\ncat > /dev/null\ncat \"{page_path}\"\nexit 0\n"),
        );
        crate::pdf::PdfRenderer::new(renderer, std::time::Duration::from_secs(30)).unwrap()
    }

    /// Bytes that carry the PDF header and sniff as no
    /// image (the tools are fake; the document is never
    /// really parsed in these tests).
    const FAKE_PDF_BYTES: &[u8] = b"%PDF-1.4\nfake document bytes for the engine tests";

    #[test]
    fn page_segment_without_a_renderer_is_not_configured() {
        let engine = Engine;
        let err = engine
            .process_preferred(
                FAKE_PDF_BYTES,
                &[Operation::Page(1)],
                ProcessContext::default(),
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
        assert!(err.to_string().contains("CRABCROP_PDF_RENDERER"), "{err}");
    }

    #[test]
    fn page_segment_on_a_non_pdf_source_is_a_decode_failure() {
        let engine = Engine;
        let renderer = pdf_test_renderer("non-pdf", 3);
        let ctx = ProcessContext {
            pdf_renderer: Some(&renderer),
            ..Default::default()
        };
        // A still PNG and a video fragment: page selection
        // is PDF-only, and both refuse as bad sources
        // before any tool spawns.
        let sources: [&[u8]; 2] = [
            &png_bytes(&DynamicImage::new_rgb8(20, 12)),
            FAKE_VIDEO_BYTES,
        ];
        for src in sources {
            let err = engine
                .process_preferred(src, &[Operation::Page(1)], ctx.clone())
                .unwrap_err();
            assert!(matches!(err, EngineError::Decode(_)), "{err:?}");
            assert!(err.to_string().contains("needs a PDF source"), "{err}");
        }
    }

    #[test]
    fn page_segment_renders_and_processes_the_page() {
        let engine = Engine;
        let renderer = pdf_test_renderer("render", 3);
        let out = engine
            .process_preferred(
                FAKE_PDF_BYTES,
                &[
                    Operation::Page(2),
                    size_op(Dimension::Pixels(4), Dimension::Pixels(3)),
                ],
                ProcessContext {
                    pdf_renderer: Some(&renderer),
                    ..Default::default()
                },
            )
            .unwrap();
        // The operation chain ran on the rendered page.
        assert_eq!((out.width, out.height), (4, 3));
        assert_eq!((out.source_width, out.source_height), (8, 6));
        // And the PDF record rides along for meta.
        let pdf = out.pdf.expect("a page render carries the PDF record");
        assert_eq!(pdf.pages, 3);
        assert_eq!(pdf.page, 2);
        assert_eq!((pdf.page_width, pdf.page_height), (612.0, 792.0));
        // No video record on a PDF render.
        assert!(out.video.is_none());
    }

    #[test]
    fn probe_source_labels_pdfs() {
        // The explain probe (issue #215): a PDF document
        // is kind "pdf" with unknown dimensions; the
        // header sniff keeps its window, so junk with a
        // late header is not a PDF.
        let probe = probe_source(FAKE_PDF_BYTES);
        assert_eq!(probe.kind, "pdf");
        assert_eq!(probe.dimensions, None);
        let mut late = vec![b' '; 2000];
        late.extend_from_slice(b"%PDF-1.4");
        assert_eq!(probe_source(&late).kind, "unknown");
    }

    #[test]
    fn page_segment_out_of_range_is_unsupported() {
        let engine = Engine;
        let renderer = pdf_test_renderer("range", 3);
        let err = engine
            .process_preferred(
                FAKE_PDF_BYTES,
                &[Operation::Page(9)],
                ProcessContext {
                    pdf_renderer: Some(&renderer),
                    ..Default::default()
                },
            )
            .unwrap_err();
        // Out of range refuses; it never clamps to the
        // last page the way a frame timestamp clamps.
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("page 9"), "{err}");
        assert!(err.to_string().contains("3 pages"), "{err}");
    }

    #[test]
    fn animated_gif_to_mp4_uses_the_configured_encoder() {
        let engine = Engine;
        let src = animated_gif_bytes();
        let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
        let out = engine
            .process_preferred(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Mp4)])],
                ProcessContext {
                    video_encoder: Some(&encoder),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Mp4);
        assert_eq!(out.bytes, b"ENGINE-VIDEO");
        assert_eq!((out.width, out.height), (40, 20));
        assert_eq!((out.source_width, out.source_height), (40, 20));
    }

    #[test]
    fn animated_webp_to_webm_uses_the_configured_encoder() {
        let engine = Engine;
        let src = animated_webp_bytes();
        let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
        let out = engine
            .process_preferred(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::WebM)])],
                ProcessContext {
                    video_encoder: Some(&encoder),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::WebM);
        assert_eq!(out.bytes, b"ENGINE-VIDEO");
    }

    #[test]
    fn video_output_runs_the_operations_per_frame() {
        let engine = Engine;
        let src = animated_gif_bytes();
        let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
        let out = engine
            .process_preferred(
                &src,
                &[
                    size_op(Dimension::Pixels(20), Dimension::Pixels(10)),
                    Operation::Filters(vec![Filter::Format(ImageFormat::Mp4)]),
                ],
                ProcessContext {
                    video_encoder: Some(&encoder),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Mp4);
        assert_eq!((out.width, out.height), (20, 10));
        assert_eq!((out.source_width, out.source_height), (40, 20));
    }

    #[test]
    fn video_format_on_a_still_source_is_a_specific_error() {
        let engine = Engine;
        let encoder = video_test_encoder(FAKE_VIDEO_SCRIPT);
        let ctx = ProcessContext {
            video_encoder: Some(&encoder),
            ..Default::default()
        };
        // A still PNG, and a single-frame GIF: both stills,
        // both refused with the animated-source error, and
        // neither falls back to a first-frame image.
        let stills = [
            png_bytes(&DynamicImage::new_rgb8(20, 12)),
            gif_bytes(&DynamicImage::new_rgb8(20, 12)),
        ];
        for src in stills {
            let err = engine
                .process_preferred(
                    &src,
                    &[Operation::Filters(vec![Filter::Format(ImageFormat::Mp4)])],
                    ctx.clone(),
                )
                .unwrap_err();
            assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
            assert!(err.to_string().contains("animated source"), "{err}");
            assert!(err.to_string().contains("still image"), "{err}");
        }
    }

    #[test]
    fn video_format_without_an_encoder_is_not_configured() {
        let engine = Engine;
        // Animated source: the deployment cannot do it.
        let err = engine
            .process(
                &animated_gif_bytes(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Mp4)])],
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
        assert!(err.to_string().contains("CRABCROP_VIDEO_ENCODER"), "{err}");
        // Still source, same answer: the not-configured
        // preflight runs before the still check, so an
        // unconfigured server never blames the source.
        let err = engine
            .process(
                &png_bytes(&DynamicImage::new_rgb8(20, 12)),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::WebM)])],
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
    }

    #[test]
    fn a_failing_encoder_is_a_video_error() {
        let engine = Engine;
        let encoder =
            video_test_encoder("#!/bin/sh\ncat > /dev/null\necho 'no codec' >&2\nexit 3\n");
        let err = engine
            .process_preferred(
                &animated_gif_bytes(),
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Mp4)])],
                ProcessContext {
                    video_encoder: Some(&encoder),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Video(_)), "{err:?}");
        assert!(err.to_string().contains("video encoder failed"), "{err}");
        assert!(err.to_string().contains("no codec"), "{err}");
    }

    /// Three 40x20 frames: red, green, blue, with delays of
    /// 100, 200, and 300 ms. Built with the oxideav encoder
    /// (the image crate's WebP encoder writes stills only);
    /// frames are coded lossless so colors round-trip exactly.
    fn animated_webp_bytes() -> Vec<u8> {
        let frame = |rgb: [u8; 3], delay_ms: u64| {
            let img =
                image::RgbaImage::from_pixel(40, 20, image::Rgba([rgb[0], rgb[1], rgb[2], 255]));
            let image = oxideav_webp::WebpImage::from_rgba8(40, 20, img.as_raw().clone()).unwrap();
            oxideav_webp::Frame::new(image, Some(std::time::Duration::from_millis(delay_ms)))
        };
        let frames = vec![
            frame([255, 0, 0], 100),
            frame([0, 200, 0], 200),
            frame([0, 0, 255], 300),
        ];
        let opts = oxideav_webp::EncodeOptions::default()
            .with_frame_mode(oxideav_webp::AnimFrameMode::Lossless);
        oxideav_webp::encode_animation(&frames, &opts).unwrap()
    }

    fn decoded_webp_frames(bytes: &[u8]) -> Vec<image::Frame> {
        use image::AnimationDecoder;
        let decoder = image::codecs::webp::WebPDecoder::new(Cursor::new(bytes)).unwrap();
        decoder.into_frames().collect_frames().unwrap()
    }

    fn frame_delay_ms(frame: &image::Frame) -> u64 {
        let duration: std::time::Duration = frame.delay().into();
        duration.as_millis() as u64
    }

    #[test]
    fn animated_webp_fixture_decodes_to_its_frames() {
        let frames = decode_webp_frames(&animated_webp_bytes()).unwrap();
        assert_eq!(frames.len(), 3);
        assert!(
            frames
                .iter()
                .all(|f| f.buffer().width() == 40 && f.buffer().height() == 20)
        );
        let delays: Vec<u64> = frames.iter().map(frame_delay_ms).collect();
        assert_eq!(delays, [100, 200, 300]);
        assert_eq!(
            frames[0].buffer().get_pixel(0, 0),
            &image::Rgba([255, 0, 0, 255])
        );
        assert_eq!(
            frames[2].buffer().get_pixel(0, 0),
            &image::Rgba([0, 0, 255, 255])
        );
    }

    #[test]
    fn animated_webp_keeps_its_frames_through_resize() {
        let engine = Engine;
        let src = animated_webp_bytes();
        let out = engine
            .process(
                &src,
                &[size_op(Dimension::Pixels(20), Dimension::Pixels(10))],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::WebP);
        let frames = decoded_webp_frames(&out.bytes);
        assert_eq!(frames.len(), 3);
        assert!(
            frames
                .iter()
                .all(|f| f.buffer().width() == 20 && f.buffer().height() == 10)
        );
        let delays: Vec<u64> = frames.iter().map(frame_delay_ms).collect();
        assert_eq!(delays, [100, 200, 300]);
        // Lossless frames: the colors survive the resize exactly.
        assert_eq!(
            frames[1].buffer().get_pixel(0, 0),
            &image::Rgba([0, 200, 0, 255])
        );
    }

    #[test]
    fn animated_webp_keeps_its_frames_through_grayscale() {
        let engine = Engine;
        let src = animated_webp_bytes();
        let out = engine
            .process(&src, &[Operation::Filters(vec![Filter::Grayscale])])
            .unwrap();
        assert_eq!(out.format, ImageFormat::WebP);
        let frames = decoded_webp_frames(&out.bytes);
        assert_eq!(frames.len(), 3);
        let delays: Vec<u64> = frames.iter().map(frame_delay_ms).collect();
        assert_eq!(delays, [100, 200, 300]);
        for frame in &frames {
            let p = frame.buffer().get_pixel(0, 0);
            assert_eq!(p[0], p[1], "pixel {p:?}");
            assert_eq!(p[1], p[2], "pixel {p:?}");
        }
    }

    #[test]
    fn animated_webp_to_gif_keeps_its_frames() {
        let engine = Engine;
        let src = animated_webp_bytes();
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Gif)])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Gif);
        let frames = decoded_gif_frames(&out.bytes);
        assert_eq!(frames.len(), 3);
        let delays: Vec<u64> = frames.iter().map(frame_delay_ms).collect();
        assert_eq!(delays, [100, 200, 300]);
    }

    #[test]
    fn animated_webp_to_a_still_format_keeps_the_first_frame() {
        let engine = Engine;
        let src = animated_webp_bytes();
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jpeg)])],
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jpeg);
        // The first frame is solid red (JPEG is lossy: near,
        // not exact).
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        let p = img.get_pixel(0, 0);
        assert!(p[0] > 200 && p[1] < 60 && p[2] < 60, "pixel {p:?}");
    }

    #[test]
    fn animated_webp_stays_out_of_jxl_negotiation() {
        let engine = Engine;
        let src = animated_webp_bytes();
        // A negotiated (preferred) JXL falls back to WebP so the
        // frames survive.
        let out = engine
            .process_preferred(
                &src,
                &[],
                ProcessContext {
                    preferred: Some(ImageFormat::Jxl),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::WebP);
        assert_eq!(decoded_webp_frames(&out.bytes).len(), 3);
        // An explicit format(jxl) still converts, first frame only.
        let out = engine
            .process_preferred(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Jxl)])],
                ProcessContext {
                    preferred: Some(ImageFormat::Jxl),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jxl);
    }

    #[test]
    fn webp_frame_cap_rejects_frame_bombs() {
        // Build a one-frame animation, then restamp its ANMF
        // chunk 201 times (the RIFF size is the only other field
        // that changes). Encoding 201 frames with the VP8L
        // encoder just for this test would take minutes; the
        // decoder cannot tell the difference.
        let one = {
            let img = image::RgbaImage::from_pixel(4, 4, image::Rgba([9, 9, 9, 255]));
            let image = oxideav_webp::WebpImage::from_rgba8(4, 4, img.as_raw().clone()).unwrap();
            let frame = oxideav_webp::Frame::new(image, Some(std::time::Duration::from_millis(50)));
            let opts = oxideav_webp::EncodeOptions::default()
                .with_frame_mode(oxideav_webp::AnimFrameMode::Lossless);
            oxideav_webp::encode_animation(&[frame], &opts).unwrap()
        };
        assert_eq!(decode_webp_frames(&one).unwrap().len(), 1);
        let anmf_at = one
            .windows(4)
            .position(|w| w == b"ANMF")
            .expect("a one-frame animation has an ANMF chunk");
        let chunk_len =
            u32::from_le_bytes(one[anmf_at + 4..anmf_at + 8].try_into().unwrap()) as usize;
        let chunk_span = 8 + chunk_len + (chunk_len % 2);
        let mut bomb = one[..anmf_at].to_vec();
        for _ in 0..201 {
            bomb.extend_from_slice(&one[anmf_at..anmf_at + chunk_span]);
        }
        let riff_size = (bomb.len() - 8) as u32;
        bomb[4..8].copy_from_slice(&riff_size.to_le_bytes());
        assert!(decode_webp_frames(&bomb).is_err());
    }

    #[test]
    fn placeholders_are_computed_only_on_request() {
        let engine = Engine;
        let src = png_bytes(&DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            100,
            60,
            image::Rgb([255, 0, 0]),
        )));
        let plain = engine.process(&src, &[]).unwrap();
        assert!(plain.placeholder.is_none());
        let out = engine
            .process_preferred(
                &src,
                &[size_op(Dimension::Pixels(50), Dimension::Pixels(30))],
                ProcessContext {
                    placeholders: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let ph = out.placeholder.unwrap();
        assert_eq!(ph.dominant_color, "#ff0000");
        assert!(ph.data_uri.starts_with("data:image/jpeg;base64,"));
    }

    #[test]
    fn heic_detection_reads_ftyp_brands() {
        assert!(is_heic(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1"));
        assert!(is_heic(b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00mif1heic"));
        // AVIF shares the BMFF container but not the brands.
        assert!(!is_heic(
            b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
        ));
        assert!(!is_heic(&[0u8; 4]));
        assert!(!is_heic(b"not an image at all"));
    }

    #[test]
    fn heic_source_without_an_image_fails_with_a_specific_error() {
        // The ftyp brands alone are not an image: the HEIC decode
        // path fails, by name, instead of falling through to a
        // generic format error. The full decode behaviour lives
        // in the heic module's tests.
        let engine = Engine;
        let heic: &[u8] = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1....";
        let err = engine.process(heic, &[]).unwrap_err();
        match err {
            EngineError::Decode(msg) => assert!(msg.contains("HEIC"), "{msg}"),
            other => panic!("expected Decode, got {other:?}"),
        }
    }

    fn watermark_map() -> WatermarkImages {
        let mut map = WatermarkImages::new();
        map.insert(
            "wm".to_string(),
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 255]))),
        );
        map
    }

    fn red_source() -> Vec<u8> {
        png_bytes(&DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            20,
            20,
            image::Rgb([255, 0, 0]),
        )))
    }

    #[test]
    fn watermark_overlays_and_blends() {
        let engine = Engine;
        let src = red_source();
        let map = watermark_map();
        let wm_op = |x: i32, y: i32, alpha: u8| {
            Operation::Filters(vec![Filter::Watermark {
                url: "wm".to_string(),
                x,
                y,
                alpha,
            }])
        };
        // Full opacity at the top-left corner.
        let out = engine
            .process_preferred(
                &src,
                &[wm_op(0, 0, 100)],
                ProcessContext {
                    watermarks: Some(&map),
                    ..Default::default()
                },
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        assert_eq!(img.get_pixel(0, 0), &image::Rgb([0, 0, 255]));
        assert_eq!(img.get_pixel(10, 10), &image::Rgb([255, 0, 0]));
        // Half opacity blends evenly.
        let out = engine
            .process_preferred(
                &src,
                &[wm_op(0, 0, 50)],
                ProcessContext {
                    watermarks: Some(&map),
                    ..Default::default()
                },
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        assert_eq!(img.get_pixel(0, 0), &image::Rgb([128, 0, 128]));
    }

    #[test]
    fn watermark_negative_coordinates_anchor_from_the_edges() {
        let engine = Engine;
        let src = red_source();
        let map = watermark_map();
        let out = engine
            .process_preferred(
                &src,
                &[Operation::Filters(vec![Filter::Watermark {
                    url: "wm".to_string(),
                    x: -2,
                    y: -2,
                    alpha: 100,
                }])],
                ProcessContext {
                    watermarks: Some(&map),
                    ..Default::default()
                },
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        // The 4x4 watermark lands at (14, 14): 20 - 4 - 2.
        assert_eq!(img.get_pixel(14, 14), &image::Rgb([0, 0, 255]));
        assert_eq!(img.get_pixel(17, 17), &image::Rgb([0, 0, 255]));
        assert_eq!(img.get_pixel(19, 19), &image::Rgb([255, 0, 0]));
        assert_eq!(img.get_pixel(13, 13), &image::Rgb([255, 0, 0]));
    }

    fn overlay_map() -> WatermarkImages {
        let mut map = WatermarkImages::new();
        map.insert(
            "ov".to_string(),
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                2,
                2,
                image::Rgb([200, 100, 50]),
            )),
        );
        map
    }

    fn overlay_base() -> Vec<u8> {
        png_bytes(&DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            4,
            4,
            image::Rgb([100, 150, 200]),
        )))
    }

    #[test]
    fn overlay_blend_modes_match_hand_computed_values() {
        let engine = Engine;
        let map = overlay_map();
        // Base (100,150,200), overlay (200,100,50), full alpha.
        let cases = [
            (crate::url::BlendMode::Multiply, [78, 59, 39]),
            (crate::url::BlendMode::Screen, [222, 191, 211]),
            (crate::url::BlendMode::Overlay, [157, 127, 167]),
            (crate::url::BlendMode::Normal, [200, 100, 50]),
        ];
        for (mode, want) in cases {
            let out = engine
                .process_preferred(
                    &overlay_base(),
                    &[Operation::Filters(vec![Filter::Overlay {
                        url: "ov".to_string(),
                        x: 0,
                        y: 0,
                        alpha: 100,
                        blend: mode,
                        width: None,
                    }])],
                    ProcessContext {
                        watermarks: Some(&map),
                        ..Default::default()
                    },
                )
                .unwrap();
            let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
            assert_eq!(img.get_pixel(0, 0), &image::Rgb(want), "mode {mode:?}");
            assert_eq!(img.get_pixel(3, 3), &image::Rgb([100, 150, 200]));
        }
        // Half opacity, normal mode: the plain average.
        let out = engine
            .process_preferred(
                &overlay_base(),
                &[Operation::Filters(vec![Filter::Overlay {
                    url: "ov".to_string(),
                    x: 0,
                    y: 0,
                    alpha: 50,
                    blend: crate::url::BlendMode::Normal,
                    width: None,
                }])],
                ProcessContext {
                    watermarks: Some(&map),
                    ..Default::default()
                },
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        assert_eq!(img.get_pixel(0, 0), &image::Rgb([150, 125, 125]));
    }

    #[test]
    fn overlay_normal_mode_reproduces_watermark() {
        let engine = Engine;
        let map = overlay_map();
        let make_ctx = || ProcessContext {
            watermarks: Some(&map),
            ..Default::default()
        };
        let via_watermark = engine
            .process_preferred(
                &overlay_base(),
                &[Operation::Filters(vec![Filter::Watermark {
                    url: "ov".to_string(),
                    x: 1,
                    y: 1,
                    alpha: 80,
                }])],
                make_ctx(),
            )
            .unwrap();
        let via_overlay = engine
            .process_preferred(
                &overlay_base(),
                &[Operation::Filters(vec![Filter::Overlay {
                    url: "ov".to_string(),
                    x: 1,
                    y: 1,
                    alpha: 80,
                    blend: crate::url::BlendMode::Normal,
                    width: None,
                }])],
                make_ctx(),
            )
            .unwrap();
        assert_eq!(via_watermark.bytes, via_overlay.bytes);
    }

    #[test]
    fn overlay_width_scales_relative_to_the_base() {
        let engine = Engine;
        let mut map = WatermarkImages::new();
        map.insert(
            "blue".to_string(),
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 255]))),
        );
        let out = engine
            .process_preferred(
                &red_source(),
                &[Operation::Filters(vec![Filter::Overlay {
                    url: "blue".to_string(),
                    x: -2,
                    y: -2,
                    alpha: 100,
                    blend: crate::url::BlendMode::Normal,
                    width: Some(50),
                }])],
                ProcessContext {
                    watermarks: Some(&map),
                    ..Default::default()
                },
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        // Scaled to 50% of 20 = 10 px, anchored at (8, 8).
        assert_eq!(img.get_pixel(8, 8), &image::Rgb([0, 0, 255]));
        assert_eq!(img.get_pixel(17, 17), &image::Rgb([0, 0, 255]));
        assert_eq!(img.get_pixel(18, 18), &image::Rgb([255, 0, 0]));
        assert_eq!(img.get_pixel(7, 7), &image::Rgb([255, 0, 0]));
    }

    #[test]
    fn unresolved_overlay_is_an_error() {
        let engine = Engine;
        let err = engine
            .process(
                &overlay_base(),
                &[Operation::Filters(vec![Filter::Overlay {
                    url: "missing".to_string(),
                    x: 0,
                    y: 0,
                    alpha: 100,
                    blend: crate::url::BlendMode::Normal,
                    width: None,
                }])],
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("overlay was not resolved"),
            "got: {err}"
        );
    }

    #[test]
    fn unresolved_watermark_is_an_error() {
        let engine = Engine;
        let src = red_source();
        let err = engine
            .process(
                &src,
                &[Operation::Filters(vec![Filter::Watermark {
                    url: "missing".to_string(),
                    x: 0,
                    y: 0,
                    alpha: 100,
                }])],
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    }

    /// A mask provider with a fixed answer: the left half is
    /// background, the right half is subject.
    #[derive(Debug)]
    struct HalfMask;

    impl crate::ai::MaskProvider for HalfMask {
        fn mask(&self, img: &DynamicImage) -> Result<image::GrayImage, crate::ai::AiError> {
            Ok(image::GrayImage::from_fn(
                img.width(),
                img.height(),
                |x, _| image::Luma([if x < img.width() / 2 { 0 } else { 255 }]),
            ))
        }
    }

    #[test]
    fn removebg_applies_the_provider_mask() {
        let engine = Engine;
        let src = red_source();
        let out = engine
            .process_preferred(
                &src,
                &[Operation::Filters(vec![Filter::RemoveBg])],
                ProcessContext {
                    mask_provider: Some(&HalfMask),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Png);
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
        assert_eq!(img.get_pixel(2, 10)[3], 0);
        assert_eq!(img.get_pixel(17, 10)[3], 255);
        // The subject's color survives untouched.
        assert_eq!(img.get_pixel(17, 10)[0], 255);
    }

    #[test]
    fn removebg_rejects_alphaless_output() {
        let engine = Engine;
        let ctx = || ProcessContext {
            mask_provider: Some(&HalfMask),
            ..Default::default()
        };
        // An explicit JPEG format conflicts.
        let err = engine
            .process_preferred(
                &red_source(),
                &[Operation::Filters(vec![
                    Filter::RemoveBg,
                    Filter::Format(ImageFormat::Jpeg),
                ])],
                ctx(),
            )
            .unwrap_err();
        match &err {
            EngineError::Unsupported(msg) => assert!(msg.contains("alpha"), "{msg}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
        // So does a JPEG source with no format named: the
        // resolved output is JPEG.
        let mut jpeg = Vec::new();
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(20, 20, image::Rgb([255, 0, 0])))
            .write_to(
                &mut std::io::Cursor::new(&mut jpeg),
                image::ImageFormat::Jpeg,
            )
            .unwrap();
        let err = engine
            .process_preferred(&jpeg, &[Operation::Filters(vec![Filter::RemoveBg])], ctx())
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    }

    #[test]
    fn removebg_without_a_provider_is_a_clean_error() {
        let engine = Engine;
        // PNG source: the format check passes, the provider
        // check is the one that fires.
        let err = engine
            .process(&red_source(), &[Operation::Filters(vec![Filter::RemoveBg])])
            .unwrap_err();
        match &err {
            EngineError::NotConfigured(msg) => {
                assert!(msg.contains("not configured"), "{msg}")
            }
            other => panic!("expected NotConfigured, got {other:?}"),
        }
    }

    /// A mask provider for studio tests: the central square
    /// is the subject, and it exposes a confidence, so the
    /// record's plumbing for both is exercised.
    #[derive(Debug)]
    struct SquareMask;

    impl crate::ai::MaskProvider for SquareMask {
        fn mask(&self, img: &DynamicImage) -> Result<image::GrayImage, crate::ai::AiError> {
            let (w, h) = (img.width(), img.height());
            Ok(image::GrayImage::from_fn(w, h, |x, y| {
                let inside = (w / 4..3 * w / 4).contains(&x) && (h / 4..3 * h / 4).contains(&y);
                image::Luma([if inside { 255 } else { 0 }])
            }))
        }

        fn confidence(&self, _img: &DynamicImage, _mask: &image::GrayImage) -> Option<f32> {
            Some(0.93)
        }
    }

    /// A 40x40 "product photo": a red square at 10..30 on
    /// green, as PNG bytes.
    fn product_source() -> Vec<u8> {
        let img = DynamicImage::ImageRgba8(image::RgbaImage::from_fn(40, 40, |x, y| {
            if (10..30).contains(&x) && (10..30).contains(&y) {
                image::Rgba([200, 30, 30, 255])
            } else {
                image::Rgba([30, 160, 30, 255])
            }
        }));
        let mut bytes = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
        bytes
    }

    fn studio_spec() -> crate::url::StudioSpec {
        crate::url::StudioSpec {
            bg: crate::url::StudioBg::White,
            shadow: crate::url::StudioShadow::None,
            pad: 10,
        }
    }

    #[test]
    fn studio_without_a_provider_is_a_clean_error() {
        let engine = Engine;
        let err = engine
            .process(
                &product_source(),
                &[Operation::Filters(vec![Filter::Studio(studio_spec())])],
            )
            .unwrap_err();
        match &err {
            EngineError::NotConfigured(msg) => {
                assert!(msg.contains("studio is not configured"), "{msg}")
            }
            other => panic!("expected NotConfigured, got {other:?}"),
        }
    }

    #[test]
    fn studio_composites_and_records_the_run() {
        let engine = Engine;
        let out = engine
            .process_preferred(
                &product_source(),
                &[Operation::Filters(vec![
                    Filter::Studio(studio_spec()),
                    Filter::Format(ImageFormat::Png),
                ])],
                ProcessContext {
                    mask_provider: Some(&SquareMask),
                    ..Default::default()
                },
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
        // White backdrop at the corners, the red product in
        // the middle: the composite, not the green photo.
        assert_eq!(img.get_pixel(0, 0).0, [255, 255, 255, 255]);
        assert_eq!(img.get_pixel(20, 20).0, [200, 30, 30, 255]);
        // The record describes the run for meta.
        let record = out.studio.expect("a studio run records itself");
        assert_eq!(record.bg, Some([255, 255, 255]));
        assert_eq!(record.bg_from, "argument");
        assert_eq!(record.shadow, crate::url::StudioShadow::None);
        assert_eq!(record.pad, 10);
        assert_eq!(record.steps, ["mask", "defringe", "backdrop", "pad"]);
        assert!(record.fallbacks.is_empty());
        assert_eq!(record.confidence, Some(0.93));
    }

    #[test]
    fn studio_output_is_a_normal_image_in_any_format() {
        // Unlike removebg, whose output is transparency, a
        // studio composite flattens onto its backdrop, so a
        // JPEG output is fine.
        let engine = Engine;
        let out = engine
            .process_preferred(
                &product_source(),
                &[Operation::Filters(vec![
                    Filter::Studio(studio_spec()),
                    Filter::Format(ImageFormat::Jpeg),
                ])],
                ProcessContext {
                    mask_provider: Some(&SquareMask),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.format, ImageFormat::Jpeg);
    }

    #[test]
    fn studio_resolves_the_brand_color_from_context() {
        let engine = Engine;
        let brand = [245, 240, 232];
        let out = engine
            .process_preferred(
                &product_source(),
                &[Operation::Filters(vec![Filter::Studio(
                    crate::url::StudioSpec::default(),
                )])],
                ProcessContext {
                    mask_provider: Some(&SquareMask),
                    brand_color: Some(brand),
                    ..Default::default()
                },
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
        assert_eq!(img.get_pixel(0, 0).0, [brand[0], brand[1], brand[2], 255]);
        let record = out.studio.unwrap();
        assert_eq!(record.bg, Some(brand));
        assert_eq!(record.bg_from, "brand");
        // The default shadow (soft) ran over the backdrop.
        assert!(record.steps.contains(&"shadow"));
    }

    #[test]
    fn studio_counts_once_against_the_operation_cap() {
        let engine = Engine;
        let ctx = || ProcessContext {
            mask_provider: Some(&SquareMask),
            ..Default::default()
        };
        // Studio plus MAX_OPERATIONS - 1 cheap filters is
        // exactly at the cap and processes.
        let mut at_cap = vec![Filter::Studio(studio_spec())];
        at_cap.extend((0..MAX_OPERATIONS - 1).map(|_| Filter::Invert));
        let out = engine.process_preferred(&product_source(), &[Operation::Filters(at_cap)], ctx());
        assert!(out.is_ok(), "{out:?}");
        // One more filter tips it over.
        let mut over_cap = vec![Filter::Studio(studio_spec())];
        over_cap.extend((0..MAX_OPERATIONS).map(|_| Filter::Invert));
        let err = engine
            .process_preferred(&product_source(), &[Operation::Filters(over_cap)], ctx())
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("operation limit"), "{err}");
    }

    /// A fake upscale provider: a deterministic 2x
    /// nearest-neighbor enlargement.
    #[derive(Debug)]
    struct DoubleUpscale;

    impl crate::ai::UpscaleProvider for DoubleUpscale {
        fn factor(&self) -> u32 {
            2
        }

        fn upscale(&self, img: &DynamicImage) -> Result<DynamicImage, crate::ai::AiError> {
            Ok(DynamicImage::ImageRgba8(image::imageops::resize(
                &img.to_rgba8(),
                img.width() * 2,
                img.height() * 2,
                image::imageops::FilterType::Nearest,
            )))
        }
    }

    #[test]
    fn upscale_doubles_and_quadruples_dimensions() {
        let engine = Engine;
        let src = red_source(); // 20x20
        let ctx = || ProcessContext {
            upscale_provider: Some(&DoubleUpscale),
            ..Default::default()
        };
        let out = engine
            .process_preferred(&src, &[Operation::Filters(vec![Filter::Upscale(2)])], ctx())
            .unwrap();
        assert_eq!((out.width, out.height), (40, 40));
        // 4x from a native-2x model is two runs.
        let out = engine
            .process_preferred(&src, &[Operation::Filters(vec![Filter::Upscale(4)])], ctx())
            .unwrap();
        assert_eq!((out.width, out.height), (80, 80));
    }

    #[test]
    fn upscale_respects_the_output_cap() {
        let engine = Engine;
        let src = png_bytes(&DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            2100,
            10,
            image::Rgb([9, 9, 9]),
        )));
        let err = engine
            .process_preferred(
                &src,
                &[Operation::Filters(vec![Filter::Upscale(2)])],
                ProcessContext {
                    upscale_provider: Some(&DoubleUpscale),
                    ..Default::default()
                },
            )
            .unwrap_err();
        match &err {
            EngineError::Unsupported(msg) => assert!(msg.contains("4096"), "{msg}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    /// A 2x provider that counts its runs, so the budget
    /// tests can prove a rejection lands before any
    /// inference happens.
    #[derive(Debug, Default)]
    struct CountingUpscale {
        runs: std::sync::atomic::AtomicUsize,
    }

    impl CountingUpscale {
        fn runs(&self) -> usize {
            self.runs.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl crate::ai::UpscaleProvider for CountingUpscale {
        fn factor(&self) -> u32 {
            2
        }

        fn upscale(&self, img: &DynamicImage) -> Result<DynamicImage, crate::ai::AiError> {
            self.runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(DynamicImage::ImageRgba8(image::imageops::resize(
                &img.to_rgba8(),
                img.width() * 2,
                img.height() * 2,
                image::imageops::FilterType::Nearest,
            )))
        }
    }

    fn upscale_attempt(
        provider: &CountingUpscale,
        w: u32,
        h: u32,
        factor: u32,
    ) -> Result<ProcessedImage, EngineError> {
        let src = png_bytes(&DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            w,
            h,
            image::Rgb([9, 9, 9]),
        )));
        Engine.process_preferred(
            &src,
            &[Operation::Filters(vec![Filter::Upscale(factor)])],
            ProcessContext {
                upscale_provider: Some(provider),
                ..Default::default()
            },
        )
    }

    #[test]
    fn upscale_rejects_an_input_over_the_budget_before_any_inference() {
        // The issue #244 reproduction's shape: 900x560 in,
        // 504,000 pixels, which OOM-killed the server when
        // nothing stood between the request and the model.
        let provider = CountingUpscale::default();
        let err = upscale_attempt(&provider, 900, 560, 2).unwrap_err();
        match &err {
            EngineError::Unsupported(msg) => {
                assert!(msg.contains("504000"), "{msg}");
                assert!(msg.contains("262144"), "{msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        assert_eq!(provider.runs(), 0, "no model run may start");
    }

    #[test]
    fn upscale_budget_is_inclusive_at_the_cap() {
        // Exactly at the budget (512x512 = 262,144 pixels)
        // runs, like every cap in the engine; one row over
        // it does not. The campaign's working 480x360 run
        // (172,800 pixels) sits comfortably under it.
        let provider = CountingUpscale::default();
        let out = upscale_attempt(&provider, 512, 512, 2).unwrap();
        assert_eq!((out.width, out.height), (1024, 1024));
        assert_eq!(provider.runs(), 1);
        let provider = CountingUpscale::default();
        let out = upscale_attempt(&provider, 480, 360, 2).unwrap();
        assert_eq!((out.width, out.height), (960, 720));
        let provider = CountingUpscale::default();
        assert!(upscale_attempt(&provider, 512, 513, 2).is_err());
        assert_eq!(provider.runs(), 0);
    }

    #[test]
    fn upscale_budgets_every_composed_run_upfront() {
        // The campaign's composed upscale(4): 256x192 in,
        // runs of 49,152 and 196,608 pixels, both under
        // the budget, landing exactly 1024x768.
        let provider = CountingUpscale::default();
        let out = upscale_attempt(&provider, 256, 192, 4).unwrap();
        assert_eq!((out.width, out.height), (1024, 768));
        assert_eq!(provider.runs(), 2);
        // A 300x300 upscale(4) passes a lone x2 budget
        // check (90,000 pixels) but its second run would
        // take 360,000: the projection refuses it before
        // the first run starts.
        let provider = CountingUpscale::default();
        let err = upscale_attempt(&provider, 300, 300, 4).unwrap_err();
        match &err {
            EngineError::Unsupported(msg) => assert!(msg.contains("360000"), "{msg}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
        assert_eq!(provider.runs(), 0);
    }

    #[test]
    fn upscale_without_a_provider_is_a_clean_error() {
        let engine = Engine;
        let err = engine
            .process(
                &red_source(),
                &[Operation::Filters(vec![Filter::Upscale(2)])],
            )
            .unwrap_err();
        match &err {
            EngineError::NotConfigured(msg) => {
                assert!(msg.contains("not configured"), "{msg}")
            }
            other => panic!("expected NotConfigured, got {other:?}"),
        }
    }

    /// A face provider reporting a face over the right half.
    #[derive(Debug)]
    struct RightFace;

    impl crate::ai::FaceProvider for RightFace {
        fn detect(
            &self,
            img: &DynamicImage,
        ) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
            Ok(vec![crate::ai::FaceBox {
                x: img.width() / 2,
                y: 0,
                w: img.width() / 2,
                h: img.height(),
                score: 0.95,
            }])
        }
    }

    #[test]
    fn smart_crop_lands_on_a_detected_face() {
        let engine = Engine;
        // Left half red, right half blue; the face is on the
        // right. A 1:2 smart crop (50x100 out of 100x100) must
        // take the right half wholesale.
        let mut img = image::RgbImage::new(100, 100);
        for (x, _y, p) in img.enumerate_pixels_mut() {
            *p = if x < 50 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 0, 255])
            };
        }
        let src = png_bytes(&DynamicImage::ImageRgb8(img));
        let out = engine
            .process_preferred(
                &src,
                &[Operation::Crop {
                    size: ImageSize {
                        width: Dimension::Pixels(50),
                        height: Dimension::Pixels(100),
                        flip_horizontal: false,
                        flip_vertical: false,
                    },
                    align: Align::Smart,
                }],
                ProcessContext {
                    face_provider: Some(&RightFace),
                    detect_provider: None,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!((out.width, out.height), (50, 100));
        let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        assert_eq!(result.get_pixel(25, 50), &image::Rgb([0, 0, 255]));
    }

    /// The known face of `face_scene`: box (140, 30, 40, 40).
    fn known_face() -> crate::ai::FaceBox {
        crate::ai::FaceBox {
            x: 140,
            y: 30,
            w: 40,
            h: 40,
            score: 0.95,
        }
    }

    #[derive(Debug)]
    struct KnownFace;

    impl crate::ai::FaceProvider for KnownFace {
        fn detect(
            &self,
            _img: &DynamicImage,
        ) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
            Ok(vec![known_face()])
        }
    }

    #[derive(Debug, Default)]
    struct CountingFace {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl crate::ai::FaceProvider for CountingFace {
        fn detect(
            &self,
            _img: &DynamicImage,
        ) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(vec![known_face()])
        }
    }

    #[derive(Debug)]
    struct FailingFace;

    impl crate::ai::FaceProvider for FailingFace {
        fn detect(
            &self,
            _img: &DynamicImage,
        ) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
            Err(crate::ai::AiError::Inference(
                "detector exploded".to_string(),
            ))
        }
    }

    /// The class-redaction scene's vocabulary: person on
    /// the left, car on the right.
    #[derive(Debug)]
    struct KnownObjects;

    fn known_detections() -> Vec<crate::ai::Detection> {
        vec![
            crate::ai::Detection {
                label: "person".to_string(),
                score: 0.9,
                x: 10,
                y: 10,
                w: 60,
                h: 80,
            },
            crate::ai::Detection {
                label: "car".to_string(),
                score: 0.8,
                x: 120,
                y: 10,
                w: 60,
                h: 80,
            },
        ]
    }

    impl crate::ai::DetectionProvider for KnownObjects {
        fn knows_class(&self, name: &str) -> bool {
            ["person", "car"]
                .iter()
                .any(|c| name.eq_ignore_ascii_case(c))
        }
        fn describe(&self) -> Vec<String> {
            vec!["person".to_string(), "car".to_string()]
        }
        fn detect(
            &self,
            _img: &DynamicImage,
        ) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
            Ok(known_detections())
        }
    }

    #[derive(Debug)]
    struct FailingObjects;

    impl crate::ai::DetectionProvider for FailingObjects {
        fn knows_class(&self, name: &str) -> bool {
            name.eq_ignore_ascii_case("person")
        }
        fn describe(&self) -> Vec<String> {
            vec!["person".to_string()]
        }
        fn detect(
            &self,
            _img: &DynamicImage,
        ) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
            Err(crate::ai::AiError::Inference(
                "detector exploded".to_string(),
            ))
        }
    }

    #[derive(Debug, Default)]
    struct CountingObjects {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl crate::ai::DetectionProvider for CountingObjects {
        fn knows_class(&self, name: &str) -> bool {
            ["person", "car"]
                .iter()
                .any(|c| name.eq_ignore_ascii_case(c))
        }
        fn describe(&self) -> Vec<String> {
            vec!["person".to_string(), "car".to_string()]
        }
        fn detect(
            &self,
            _img: &DynamicImage,
        ) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(known_detections())
        }
    }

    /// 200x100, a black/white checkerboard everywhere: any
    /// redaction (blur or pixelate) visibly changes its box.
    fn class_scene() -> Vec<u8> {
        let img = image::RgbImage::from_fn(200, 100, |x, y| {
            if (x + y) % 2 == 0 {
                image::Rgb([0, 0, 0])
            } else {
                image::Rgb([255, 255, 255])
            }
        });
        png_bytes(&DynamicImage::ImageRgb8(img))
    }

    fn detect_ctx(provider: &dyn crate::ai::DetectionProvider) -> ProcessContext<'_> {
        ProcessContext {
            detect_provider: Some(provider),
            ..Default::default()
        }
    }

    #[test]
    fn pixelateclass_redacts_only_the_named_class() {
        let engine = Engine;
        let out = engine
            .process_preferred(
                &class_scene(),
                &[Operation::Filters(vec![Filter::PixelateClass(
                    "Person".to_string(),
                )])],
                detect_ctx(&KnownObjects),
            )
            .unwrap();
        let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        // Inside the person box the checkerboard is gone
        // (pixelate averages blocks: neighbors now agree).
        assert_eq!(result.get_pixel(11, 11), result.get_pixel(12, 11));
        assert_eq!(result.get_pixel(11, 11), result.get_pixel(11, 12));
        // Inside the car box the checkerboard survives.
        assert_ne!(result.get_pixel(121, 11), result.get_pixel(122, 11));
    }

    #[test]
    fn chained_class_filters_share_one_detection() {
        let engine = Engine;
        let counter = CountingObjects::default();
        engine
            .process_preferred(
                &class_scene(),
                &[Operation::Filters(vec![
                    Filter::BlurClass("person".to_string()),
                    Filter::PixelateClass("car".to_string()),
                ])],
                detect_ctx(&counter),
            )
            .unwrap();
        assert_eq!(counter.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    /// A declared detector (issue #166) whose vocabulary is
    /// its own: zebra, over the car box of the class scene.
    #[derive(Debug)]
    struct DeclaredObjects;

    impl crate::ai::DetectionProvider for DeclaredObjects {
        fn knows_class(&self, name: &str) -> bool {
            name.eq_ignore_ascii_case("zebra")
        }
        fn describe(&self) -> Vec<String> {
            vec!["zebra".to_string()]
        }
        fn detect(
            &self,
            _img: &DynamicImage,
        ) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
            Ok(vec![crate::ai::Detection {
                label: "zebra".to_string(),
                score: 0.7,
                x: 120,
                y: 10,
                w: 60,
                h: 80,
            }])
        }
    }

    /// A declared detector that knows person (shared with
    /// the built-in) and zebra (its own), counting its runs:
    /// the resolution-order witness.
    #[derive(Debug, Default)]
    struct CountingDeclared {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl crate::ai::DetectionProvider for CountingDeclared {
        fn knows_class(&self, name: &str) -> bool {
            ["person", "zebra"]
                .iter()
                .any(|c| name.eq_ignore_ascii_case(c))
        }
        fn describe(&self) -> Vec<String> {
            vec!["person".to_string(), "zebra".to_string()]
        }
        fn detect(
            &self,
            _img: &DynamicImage,
        ) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(vec![
                crate::ai::Detection {
                    label: "person".to_string(),
                    score: 0.9,
                    x: 10,
                    y: 10,
                    w: 60,
                    h: 80,
                },
                crate::ai::Detection {
                    label: "zebra".to_string(),
                    score: 0.7,
                    x: 120,
                    y: 10,
                    w: 60,
                    h: 80,
                },
            ])
        }
    }

    fn declared_ctx<'a>(
        builtin: Option<&'a dyn crate::ai::DetectionProvider>,
        declared: &'a [&'a dyn crate::ai::DetectionProvider],
    ) -> ProcessContext<'a> {
        ProcessContext {
            detect_provider: builtin,
            declared_detect_providers: declared,
            ..Default::default()
        }
    }

    #[test]
    fn declared_detector_drives_redaction_for_its_own_class() {
        let engine = Engine;
        let declared: [&dyn crate::ai::DetectionProvider; 1] = [&DeclaredObjects];
        let out = engine
            .process_preferred(
                &class_scene(),
                &[Operation::Filters(vec![Filter::PixelateClass(
                    "zebra".to_string(),
                )])],
                declared_ctx(Some(&KnownObjects), &declared),
            )
            .unwrap();
        let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        // The zebra box (the declared provider's) is
        // pixelated: neighbors inside it agree.
        assert_eq!(result.get_pixel(121, 11), result.get_pixel(122, 11));
        // The person box is untouched: only the declared
        // provider ran, and it returns no person boxes.
        assert_ne!(result.get_pixel(11, 11), result.get_pixel(12, 11));
    }

    #[test]
    fn class_resolution_prefers_the_builtin_detector() {
        let engine = Engine;
        let builtin = CountingObjects::default();
        let declared_counter = CountingDeclared::default();
        let declared: [&dyn crate::ai::DetectionProvider; 1] = [&declared_counter];
        engine
            .process_preferred(
                &class_scene(),
                &[Operation::Filters(vec![Filter::BlurClass(
                    "person".to_string(),
                )])],
                declared_ctx(Some(&builtin), &declared),
            )
            .unwrap();
        // Both providers know "person": the built-in runs,
        // the declared one never does.
        assert_eq!(builtin.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(
            declared_counter
                .calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[test]
    fn declared_detectors_resolve_in_order_and_share_one_run_each() {
        let engine = Engine;
        let counting = CountingDeclared::default();
        let declared: [&dyn crate::ai::DetectionProvider; 2] = [&DeclaredObjects, &counting];
        // Zebra resolves to the first declared provider
        // (DeclaredObjects); person only exists on the
        // second. Each runs at most once for the request.
        engine
            .process_preferred(
                &class_scene(),
                &[Operation::Filters(vec![
                    Filter::PixelateClass("zebra".to_string()),
                    Filter::BlurClass("person".to_string()),
                    Filter::PixelateClass("person".to_string()),
                ])],
                declared_ctx(None, &declared),
            )
            .unwrap();
        assert_eq!(counting.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        // A class no provider knows is still a clean
        // refusal with declared providers configured.
        let err = engine
            .process_preferred(
                &class_scene(),
                &[Operation::Filters(vec![Filter::BlurClass(
                    "giraffe".to_string(),
                )])],
                declared_ctx(None, &declared),
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
    }

    /// A declared focal provider pinned to the right edge
    /// (issue #166).
    #[derive(Debug)]
    struct RightFocal;

    impl crate::ai::FocalProvider for RightFocal {
        fn focal_point(&self, _img: &DynamicImage) -> Result<(f32, f32), crate::ai::AiError> {
            Ok((0.95, 0.5))
        }
    }

    /// Left half red, right half blue: the scene the face
    /// smart-crop test uses, reused for focal precedence.
    fn halves_scene() -> Vec<u8> {
        let mut img = image::RgbImage::new(100, 100);
        for (x, _y, p) in img.enumerate_pixels_mut() {
            *p = if x < 50 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 0, 255])
            };
        }
        png_bytes(&DynamicImage::ImageRgb8(img))
    }

    fn smart_half_crop() -> Operation {
        Operation::Crop {
            size: ImageSize {
                width: Dimension::Pixels(50),
                height: Dimension::Pixels(100),
                flip_horizontal: false,
                flip_vertical: false,
            },
            align: Align::Smart,
        }
    }

    #[test]
    fn declared_focal_point_steers_a_smart_crop() {
        let engine = Engine;
        let out = engine
            .process_preferred(
                &halves_scene(),
                &[smart_half_crop()],
                ProcessContext {
                    focal_provider: Some(&RightFocal),
                    ..Default::default()
                },
            )
            .unwrap();
        let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        // The declared point at x 0.95 takes the right
        // (blue) half wholesale.
        assert_eq!(result.get_pixel(25, 50), &image::Rgb([0, 0, 255]));
    }

    /// A face provider reporting a face over the left half:
    /// the mirror of RightFace, for precedence against the
    /// declared focal point at the right.
    #[derive(Debug)]
    struct LeftFace;

    impl crate::ai::FaceProvider for LeftFace {
        fn detect(
            &self,
            img: &DynamicImage,
        ) -> Result<Vec<crate::ai::FaceBox>, crate::ai::AiError> {
            Ok(vec![crate::ai::FaceBox {
                x: 0,
                y: 0,
                w: img.width() / 2,
                h: img.height(),
                score: 0.95,
            }])
        }
    }

    #[test]
    fn face_provider_beats_the_declared_focal_point() {
        let engine = Engine;
        let out = engine
            .process_preferred(
                &halves_scene(),
                &[smart_half_crop()],
                ProcessContext {
                    face_provider: Some(&LeftFace),
                    focal_provider: Some(&RightFocal),
                    ..Default::default()
                },
            )
            .unwrap();
        let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        assert_eq!(result.get_pixel(25, 50), &image::Rgb([255, 0, 0]));
    }

    #[test]
    fn caller_focal_rect_beats_the_declared_focal_point() {
        let engine = Engine;
        let src = png_bytes(&quadrant_pattern());
        // The focal rectangle names the black bottom-right;
        // the declared provider would pull a smart crop to
        // the pattern's own saliency (which keeps some
        // white, as the control in
        // focal_smart_crop_keeps_the_named_region shows).
        let out = engine
            .process_preferred(
                &src,
                &[focal_op(4, 2, 8, 4), square_crop(Align::Smart)],
                ProcessContext {
                    focal_provider: Some(&RightFocal),
                    ..Default::default()
                },
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap().to_luma8();
        assert!(img.pixels().all(|p| p[0] == 0));
    }

    #[test]
    fn blurclass_unknown_class_and_missing_model_fail_cleanly() {
        let engine = Engine;
        let err = engine
            .process_preferred(
                &class_scene(),
                &[Operation::Filters(vec![Filter::BlurClass(
                    "zebra".to_string(),
                )])],
                detect_ctx(&KnownObjects),
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "{err:?}");
        let err = engine
            .process_preferred(
                &class_scene(),
                &[Operation::Filters(vec![Filter::BlurClass(
                    "person".to_string(),
                )])],
                ProcessContext::default(),
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::NotConfigured(_)), "{err:?}");
    }

    #[test]
    fn blurclass_detection_failure_is_an_inference_error() {
        // The #42 rule: never serve an unredacted image
        // because detection failed.
        let engine = Engine;
        let err = engine
            .process_preferred(
                &class_scene(),
                &[Operation::Filters(vec![Filter::BlurClass(
                    "person".to_string(),
                )])],
                detect_ctx(&FailingObjects),
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Inference(_)), "{err:?}");
    }

    /// 200x160 flat gray, except the known face box, which is a
    /// black/white checkerboard.
    fn face_scene() -> Vec<u8> {
        let mut img = image::RgbImage::from_pixel(200, 160, image::Rgb([90, 90, 90]));
        for (x, y, p) in img.enumerate_pixels_mut() {
            if (140..180).contains(&x) && (30..70).contains(&y) {
                *p = if (x + y) % 2 == 0 {
                    image::Rgb([0, 0, 0])
                } else {
                    image::Rgb([255, 255, 255])
                };
            }
        }
        png_bytes(&DynamicImage::ImageRgb8(img))
    }

    fn face_crop_op() -> Operation {
        Operation::Crop {
            size: ImageSize {
                width: Dimension::Pixels(100),
                height: Dimension::Pixels(100),
                flip_horizontal: false,
                flip_vertical: false,
            },
            align: Align::Face,
        }
    }

    #[test]
    fn face_crop_targets_the_detected_face() {
        let engine = Engine;
        let out = engine
            .process_preferred(
                &face_scene(),
                &[face_crop_op()],
                ProcessContext {
                    face_provider: Some(&KnownFace),
                    detect_provider: None,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!((out.width, out.height), (100, 100));
        let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        // Default padding 0.5: the crop is the 80x80 region at
        // (120, 10), so the output center is the face center:
        // checkerboard blended by the resize (dark or light,
        // never the flat background gray), corner is background.
        let center = result.get_pixel(50, 50);
        assert!(center[0] < 40 || center[0] > 200, "center {center:?}");
        assert_eq!(result.get_pixel(2, 2), &image::Rgb([90, 90, 90]));
    }

    #[test]
    fn debug_focal_face_crop_marks_the_face_it_centered_on() {
        let engine = Engine;
        let out = engine
            .process_preferred(
                &face_scene(),
                &[Operation::DebugFocal, face_crop_op()],
                ProcessContext {
                    face_provider: Some(&KnownFace),
                    detect_provider: None,
                    ..Default::default()
                },
            )
            .unwrap();
        // The canvas is the 200x160 scene, not the 100x100 crop.
        assert_eq!(dims_of(&out), (200, 160));
        // The chosen face is the focal rectangle: its corner
        // carries the focal color, and the crosshair sits at
        // its center (160, 50).
        assert_eq!(debug_px(&out, 140, 30), crate::debugview::FOCAL_COLOR);
        assert_eq!(debug_px(&out, 160, 50), crate::debugview::FOCAL_COLOR);
        // The window is the padded-face window the plain test
        // documents: 80x80 at (120, 10).
        assert_eq!(debug_px(&out, 120, 10), crate::debugview::WINDOW_COLOR);
        assert_eq!(debug_px(&out, 199, 89), crate::debugview::WINDOW_COLOR);
    }

    #[test]
    fn debug_focal_smart_crop_draws_the_faces_that_fed_it() {
        let engine = Engine;
        let out = engine
            .process_preferred(
                &face_scene(),
                &[Operation::DebugFocal, crop_op(100, 100, Align::Smart)],
                ProcessContext {
                    face_provider: Some(&KnownFace),
                    detect_provider: None,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(dims_of(&out), (200, 160));
        // The detected face box is amber; the crosshair is on
        // its center, which is where the smart window went.
        assert_eq!(debug_px(&out, 140, 30), crate::debugview::FACE_COLOR);
        assert_eq!(debug_px(&out, 160, 50), crate::debugview::FOCAL_COLOR);
        // Window: 160 wide (the full height at 1:1), starting
        // at x 40 (centered on the face, clamped).
        assert_eq!(debug_px(&out, 40, 0), crate::debugview::WINDOW_COLOR);
    }

    #[test]
    fn face_crop_without_a_model_is_a_clean_error() {
        let engine = Engine;
        let err = engine
            .process(&face_scene(), &[face_crop_op()])
            .unwrap_err();
        match &err {
            EngineError::NotConfigured(msg) => assert!(msg.contains("face"), "{msg}"),
            other => panic!("expected NotConfigured, got {other:?}"),
        }
    }

    #[test]
    fn face_crop_falls_back_when_detection_fails() {
        let engine = Engine;
        let out = engine
            .process_preferred(
                &face_scene(),
                &[face_crop_op()],
                ProcessContext {
                    face_provider: Some(&FailingFace),
                    detect_provider: None,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!((out.width, out.height), (100, 100));
    }

    #[test]
    fn blurfaces_redacts_only_the_face_region() {
        let engine = Engine;
        let out = engine
            .process_preferred(
                &face_scene(),
                &[Operation::Filters(vec![Filter::BlurFaces])],
                ProcessContext {
                    face_provider: Some(&KnownFace),
                    detect_provider: None,
                    ..Default::default()
                },
            )
            .unwrap();
        let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        // Outside the box: untouched.
        assert_eq!(result.get_pixel(10, 10), &image::Rgb([90, 90, 90]));
        assert_eq!(result.get_pixel(139, 50), &image::Rgb([90, 90, 90]));
        // Inside: the checkerboard extremes are blurred away.
        let mut extremes = 0;
        for y in 30..70 {
            for x in 140..180 {
                let v = result.get_pixel(x, y)[0];
                if v == 0 || v == 255 {
                    extremes += 1;
                }
            }
        }
        assert!(extremes < 80, "{extremes} of 1600 pixels still extreme");
    }

    #[test]
    fn pixelatefaces_blocks_the_face_region() {
        let engine = Engine;
        let out = engine
            .process_preferred(
                &face_scene(),
                &[Operation::Filters(vec![Filter::PixelateFaces])],
                ProcessContext {
                    face_provider: Some(&KnownFace),
                    detect_provider: None,
                    ..Default::default()
                },
            )
            .unwrap();
        let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        assert_eq!(result.get_pixel(10, 10), &image::Rgb([90, 90, 90]));
        // Neighboring pixels inside one 4px block agree.
        assert_eq!(result.get_pixel(141, 31), result.get_pixel(142, 32));
        // The checkerboard is gone: blocks are mid-gray averages
        // or single extremes, but adjacent blocks differ from a
        // per-pixel checker pattern.
        let mut distinct = std::collections::HashSet::new();
        for y in 30..70 {
            for x in 140..180 {
                distinct.insert(result.get_pixel(x, y).0);
            }
        }
        assert!(distinct.len() <= 100, "{} distinct colors", distinct.len());
    }

    #[test]
    fn blurfaces_fails_when_detection_fails() {
        let engine = Engine;
        let err = engine
            .process_preferred(
                &face_scene(),
                &[Operation::Filters(vec![Filter::BlurFaces])],
                ProcessContext {
                    face_provider: Some(&FailingFace),
                    detect_provider: None,
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::Inference(_)), "{err:?}");
    }

    #[test]
    fn blurfaces_without_a_model_is_a_clean_error() {
        let engine = Engine;
        let err = engine
            .process(
                &face_scene(),
                &[Operation::Filters(vec![Filter::BlurFaces])],
            )
            .unwrap_err();
        match &err {
            EngineError::NotConfigured(msg) => assert!(msg.contains("face"), "{msg}"),
            other => panic!("expected NotConfigured, got {other:?}"),
        }
    }

    #[test]
    fn chained_face_ops_detect_once() {
        let engine = Engine;
        let provider = CountingFace::default();
        engine
            .process_preferred(
                &face_scene(),
                &[Operation::Filters(vec![
                    Filter::BlurFaces,
                    Filter::PixelateFaces,
                ])],
                ProcessContext {
                    face_provider: Some(&provider),
                    detect_provider: None,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    fn pad_op(w: u32, h: u32, fill: PadFill) -> Operation {
        Operation::Pad {
            size: ImageSize {
                width: Dimension::Pixels(w),
                height: Dimension::Pixels(h),
                flip_horizontal: false,
                flip_vertical: false,
            },
            fill,
        }
    }

    fn decoded(out: &ProcessedImage) -> image::RgbImage {
        image::load_from_memory(&out.bytes).unwrap().to_rgb8()
    }

    #[test]
    fn pad_portrait_into_landscape_defaults_to_black() {
        let engine = Engine;
        // 100x200 red into 200x100: fits at 50x100, centered,
        // so x 75..125 is red and the rest is black.
        let out = engine
            .process(
                &solid_png(100, 200, [255, 0, 0]),
                &[pad_op(200, 100, PadFill::Directive)],
            )
            .unwrap();
        assert_eq!((out.width, out.height), (200, 100));
        let img = decoded(&out);
        assert_eq!(img.get_pixel(100, 50), &image::Rgb([255, 0, 0]));
        assert_eq!(img.get_pixel(124, 50), &image::Rgb([255, 0, 0]));
        assert_eq!(img.get_pixel(74, 50), &image::Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(125, 50), &image::Rgb([0, 0, 0]));
        assert_eq!(img.get_pixel(0, 0), &image::Rgb([0, 0, 0]));
    }

    #[test]
    fn pad_landscape_into_portrait_with_a_color() {
        let engine = Engine;
        // 200x100 blue into 100x200 with orange bars: fits at
        // 100x50, centered vertically at y 75..125.
        let out = engine
            .process(
                &solid_png(200, 100, [0, 0, 255]),
                &[pad_op(100, 200, PadFill::Color([255, 136, 0]))],
            )
            .unwrap();
        assert_eq!((out.width, out.height), (100, 200));
        let img = decoded(&out);
        assert_eq!(img.get_pixel(50, 100), &image::Rgb([0, 0, 255]));
        assert_eq!(img.get_pixel(50, 74), &image::Rgb([255, 136, 0]));
        assert_eq!(img.get_pixel(50, 125), &image::Rgb([255, 136, 0]));
    }

    #[test]
    fn pad_never_adapts_its_canvas() {
        // `adaptive` is a fit-in modifier; pad's canvas is
        // explicit, so an orientation mismatch never swaps it:
        // the output is exactly the requested canvas in both
        // mismatch directions.
        let engine = Engine;
        let out = engine
            .process(
                &solid_png(100, 200, [255, 0, 0]),
                &[pad_op(200, 100, PadFill::Directive)],
            )
            .unwrap();
        assert_eq!((out.width, out.height), (200, 100));
        let out = engine
            .process(
                &solid_png(200, 100, [255, 0, 0]),
                &[pad_op(100, 200, PadFill::Directive)],
            )
            .unwrap();
        assert_eq!((out.width, out.height), (100, 200));
    }

    #[test]
    fn pad_uses_the_fill_directive_in_either_order() {
        let engine = Engine;
        for ops in [
            vec![
                Operation::Filters(vec![Filter::Fill([0, 0, 255])]),
                pad_op(200, 100, PadFill::Directive),
            ],
            vec![
                pad_op(200, 100, PadFill::Directive),
                Operation::Filters(vec![Filter::Fill([0, 0, 255])]),
            ],
        ] {
            let out = engine
                .process(&solid_png(100, 200, [0, 200, 0]), &ops)
                .unwrap();
            let img = decoded(&out);
            assert_eq!(img.get_pixel(0, 0), &image::Rgb([0, 0, 255]));
            assert_eq!(img.get_pixel(100, 50), &image::Rgb([0, 200, 0]));
        }
    }

    #[test]
    fn pad_blur_bars_are_ambient() {
        let engine = Engine;
        // Left half red, right half blue, 100x100 into 200x100:
        // the image sits at x 50..150 unscaled; the bars are the
        // blurred cover copy, red-dominant on the left.
        let mut src = image::RgbImage::new(100, 100);
        for (x, _y, p) in src.enumerate_pixels_mut() {
            *p = if x < 50 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 0, 255])
            };
        }
        let out = engine
            .process(
                &png_bytes(&DynamicImage::ImageRgb8(src)),
                &[pad_op(200, 100, PadFill::Blur)],
            )
            .unwrap();
        assert_eq!((out.width, out.height), (200, 100));
        let img = decoded(&out);
        // The fitted image itself is exact in the middle.
        assert_eq!(img.get_pixel(60, 50), &image::Rgb([255, 0, 0]));
        assert_eq!(img.get_pixel(140, 50), &image::Rgb([0, 0, 255]));
        // The left bar is a blurred wash: red leads, but the
        // pure source red is softened by the blur.
        let bar = img.get_pixel(10, 50);
        assert!(bar[0] > 140 && bar[2] < 110, "bar {bar:?}");
        let right_bar = img.get_pixel(190, 50);
        assert!(
            right_bar[2] > 140 && right_bar[0] < 110,
            "bar {right_bar:?}"
        );
    }

    #[test]
    fn pad_with_a_matching_aspect_changes_nothing() {
        let engine = Engine;
        let out = engine
            .process(
                &solid_png(200, 100, [10, 20, 30]),
                &[pad_op(200, 100, PadFill::Color([255, 255, 255]))],
            )
            .unwrap();
        assert_eq!((out.width, out.height), (200, 100));
        let img = decoded(&out);
        assert_eq!(img.get_pixel(0, 0), &image::Rgb([10, 20, 30]));
        assert_eq!(img.get_pixel(199, 99), &image::Rgb([10, 20, 30]));
    }

    /// A renderer that stamps one white pixel at the origin:
    /// proof of routing and ordering, nothing more.
    #[derive(Debug)]
    struct MarkerRenderer;

    impl crate::text::TextRenderer for MarkerRenderer {
        fn draw(
            &self,
            img: &DynamicImage,
            _text: &str,
            _size: u32,
            _color: [u8; 4],
            _anchor: crate::url::Anchor,
        ) -> Result<DynamicImage, String> {
            let mut out = img.to_rgba8();
            out.put_pixel(0, 0, image::Rgba([255, 255, 255, 255]));
            Ok(DynamicImage::ImageRgba8(out))
        }
    }

    fn text_op() -> Operation {
        Operation::Filters(vec![Filter::Text {
            text: "HI".to_string(),
            size: 20,
            color: [255, 255, 255, 255],
            anchor: crate::url::Anchor::BottomRight,
        }])
    }

    #[test]
    fn svg_sources_rasterize_through_the_pipeline() {
        let engine = Engine;
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><rect width="100" height="100" fill="#00ff00"/><circle cx="50" cy="50" r="20" fill="#ff0000"/></svg>"##;
        // No size asked: intrinsic size, PNG output.
        let out = engine.process(svg, &[]).unwrap();
        assert_eq!((out.width, out.height), (100, 100));
        assert_eq!(out.format, ImageFormat::Png);
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
        assert_eq!(img.get_pixel(50, 50), &image::Rgba([255, 0, 0, 255]));
        // A size op renders the vector at that size, crisply:
        // the circle scales with it.
        let out = engine
            .process(
                svg,
                &[size_op(Dimension::Pixels(50), Dimension::Pixels(50))],
            )
            .unwrap();
        assert_eq!((out.width, out.height), (50, 50));
        let img = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
        assert_eq!(img.get_pixel(25, 25), &image::Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(2, 2), &image::Rgba([0, 255, 0, 255]));
        // Non-SVG XML is a bad source, not a render.
        assert!(matches!(
            engine.process(b"<note><to>x</to></note>", &[]),
            Err(EngineError::Decode(_))
        ));
    }

    #[test]
    fn dpr_scales_size_and_crop_targets() {
        let engine = Engine;
        let src = solid_png(100, 100, [9, 9, 9]);
        let dims = |ops: &[Operation]| {
            let out = engine.process(&src, ops).unwrap();
            (out.width, out.height)
        };
        // dpr(2) doubles the size target.
        assert_eq!(
            dims(&[
                size_op(Dimension::Pixels(25), Dimension::Pixels(25)),
                Operation::Filters(vec![Filter::Dpr(2.0)])
            ]),
            (50, 50)
        );
        // A fractional factor rounds to whole pixels: 33 * 1.5.
        assert_eq!(
            dims(&[
                size_op(Dimension::Pixels(33), Dimension::Pixels(33)),
                Operation::Filters(vec![Filter::Dpr(1.5)])
            ]),
            (50, 50)
        );
        // dpr(1) is a no-op.
        assert_eq!(
            dims(&[
                size_op(Dimension::Pixels(25), Dimension::Pixels(25)),
                Operation::Filters(vec![Filter::Dpr(1.0)])
            ]),
            (25, 25)
        );
        // Crop targets scale too.
        assert_eq!(
            dims(&[
                Operation::Crop {
                    size: ImageSize {
                        width: Dimension::Pixels(25),
                        height: Dimension::Pixels(30),
                        flip_horizontal: false,
                        flip_vertical: false,
                    },
                    align: Align::Center,
                },
                Operation::Filters(vec![Filter::Dpr(2.0)]),
            ]),
            (50, 60)
        );
    }

    #[test]
    fn dpr_runs_into_the_no_upscale_cap_not_past_it() {
        let engine = Engine;
        let out = engine
            .process(
                &solid_png(60, 60, [9, 9, 9]),
                &[
                    size_op(Dimension::Pixels(50), Dimension::Pixels(50)),
                    Operation::Filters(vec![Filter::Dpr(2.0), Filter::NoUpscale]),
                ],
            )
            .unwrap();
        assert_eq!((out.width, out.height), (60, 60));
    }

    #[test]
    fn dpr_hint_applies_without_a_directive_and_loses_to_one() {
        let engine = Engine;
        let src = solid_png(100, 100, [9, 9, 9]);
        // The hint alone doubles.
        let out = engine
            .process_preferred(
                &src,
                &[size_op(Dimension::Pixels(25), Dimension::Pixels(25))],
                ProcessContext {
                    dpr: Some(2.0),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!((out.width, out.height), (50, 50));
        // An explicit directive beats the hint: 40 * 1.5, not * 2.
        let out = engine
            .process_preferred(
                &src,
                &[
                    size_op(Dimension::Pixels(40), Dimension::Pixels(40)),
                    Operation::Filters(vec![Filter::Dpr(1.5)]),
                ],
                ProcessContext {
                    dpr: Some(2.0),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!((out.width, out.height), (60, 60));
    }

    #[test]
    fn text_routes_to_the_renderer_after_the_resize() {
        let engine = Engine;
        let ops = vec![
            Operation::Size(ImageSize {
                width: Dimension::Pixels(50),
                height: Dimension::Pixels(50),
                flip_horizontal: false,
                flip_vertical: false,
            }),
            text_op(),
        ];
        let out = engine
            .process_preferred(
                &solid_png(100, 100, [0, 0, 0]),
                &ops,
                ProcessContext {
                    text_renderer: Some(&MarkerRenderer),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!((out.width, out.height), (50, 50));
        let result = image::load_from_memory(&out.bytes).unwrap().to_rgba8();
        assert_eq!(result.get_pixel(0, 0), &image::Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn text_without_a_renderer_is_a_clean_error() {
        let engine = Engine;
        assert!(matches!(
            engine.process(&solid_png(10, 10, [0, 0, 0]), &[text_op()]),
            Err(EngineError::NotConfigured(msg)) if msg.contains("text is not configured")
        ));
    }

    #[test]
    fn trim_composes_with_a_following_resize() {
        let engine = Engine;
        // 60x40 white with a centered black box, 20x20 at (20, 10).
        let mut img = DynamicImage::new_rgb8(60, 40);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if (20..40).contains(&x) && (10..30).contains(&y) {
                image::Rgb([0, 0, 0])
            } else {
                image::Rgb([255, 255, 255])
            };
        }
        let src = png_bytes(&img);
        let out = engine
            .process(
                &src,
                &[
                    Operation::Filters(vec![Filter::Trim(10)]),
                    size_op(Dimension::Pixels(10), Dimension::Pixels(10)),
                ],
            )
            .unwrap();
        assert_eq!(dims_of(&out), (10, 10));
        // Every output pixel is black: the resize saw only the
        // trimmed box, not the white frame.
        let decoded = image::load_from_memory(&out.bytes).unwrap().to_luma8();
        assert!(decoded.pixels().all(|p| p[0] == 0));
    }

    #[test]
    fn processed_image_reports_dimensions() {
        let engine = Engine;
        let src = png_bytes(&DynamicImage::new_rgb8(400, 200));
        let out = engine
            .process(
                &src,
                &[size_op(Dimension::Pixels(100), Dimension::Pixels(50))],
            )
            .unwrap();
        assert_eq!((out.width, out.height), (100, 50));
        assert_eq!((out.source_width, out.source_height), (400, 200));
    }

    #[test]
    fn quality_ladder_steps_by_format() {
        // The descent is planned against the steps that
        // remain: geometric from the current quality, landing
        // on the floor of 1 at the final step (issue #229).
        assert_eq!(next_quality(80, 5), 33);
        assert_eq!(next_quality(80, 1), 1);
        assert_eq!(next_quality(2, 5), 1);
        let mut q = 95;
        for steps_left in (1..=5).rev() {
            q = next_quality(q, steps_left);
        }
        assert_eq!(q, 1, "five planned steps from 95 reach the floor");
        let mut q = 95;
        for steps_left in (1..=4).rev() {
            q = next_quality(q, steps_left);
        }
        assert_eq!(q, 1, "four planned steps from 95 reach the floor");
        assert_eq!(max_quality_attempts(ImageFormat::Avif), 4);
        assert!(max_quality_attempts(ImageFormat::Jpeg) > 4);
    }

    #[test]
    fn max_bytes_unreachable_budget_returns_the_floor_encode() {
        // Issue #229: with a budget nothing can meet, the
        // ladder must still walk to quality 1 and return that
        // smallest encode, not stop wherever the encode pool
        // used to run out.
        let engine = Engine;
        let src = noise_png();
        let floor = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Format(ImageFormat::Jpeg),
                    Filter::Quality(Quality::Value(1)),
                ])],
            )
            .unwrap();
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Format(ImageFormat::Jpeg),
                    Filter::Quality(Quality::Value(95)),
                    Filter::MaxBytes(1),
                ])],
            )
            .unwrap();
        assert_eq!(out.quality, Some(1));
        assert_eq!(out.bytes, floor.bytes);
    }

    #[test]
    fn max_bytes_still_stops_at_the_first_fitting_quality() {
        // A budget the ladder reaches midway behaves exactly
        // as before: stop at the first quality that fits.
        let engine = Engine;
        let src = noise_png();
        let at = |q: u8| {
            engine
                .process(
                    &src,
                    &[Operation::Filters(vec![
                        Filter::Format(ImageFormat::Jpeg),
                        Filter::Quality(Quality::Value(q)),
                    ])],
                )
                .unwrap()
                .bytes
                .len() as u64
        };
        let budget = at(15);
        assert!(at(38) > budget, "the step above 15 must not fit");
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![
                    Filter::Format(ImageFormat::Jpeg),
                    Filter::Quality(Quality::Value(95)),
                    Filter::MaxBytes(budget),
                ])],
            )
            .unwrap();
        assert!(out.bytes.len() as u64 <= budget);
        assert_eq!(out.quality, Some(15));
    }

    #[test]
    fn errors_render() {
        assert_eq!(
            EngineError::Unsupported("crop".to_string()).to_string(),
            "unsupported operation: crop"
        );
    }

    /// Timing reference for the issue thread; run explicitly with
    /// `cargo test avif_encode_bench -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing benchmark"]
    fn avif_encode_bench() {
        let mut img = DynamicImage::new_rgb8(512, 384);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = image::Rgb([(x / 2) as u8, (y / 2) as u8, ((x + y) / 4) as u8]);
        }
        let src = png_bytes(&img);
        let engine = Engine;
        let start = std::time::Instant::now();
        let out = engine
            .process(
                &src,
                &[Operation::Filters(vec![Filter::Format(ImageFormat::Avif)])],
            )
            .unwrap();
        eprintln!(
            "avif 512x384 q80 speed8: {:?}, {} bytes",
            start.elapsed(),
            out.bytes.len()
        );
    }
}
