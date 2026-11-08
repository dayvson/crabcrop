//! The image engine: decode, transform, and encode.
//!
//! [`Engine::process`] takes source bytes and the parsed [`Operation`]s,
//! and returns the encoded output plus its format.

use std::io::Cursor;
use std::sync::Once;

use image::DynamicImage;

use crate::filters::EncodeOptions;
use crate::quality::auto_quality;
use crate::url::{
    Align, Dimension, Filter, FormatOptions, ImageFormat, ImageSize, Operation, PadFill, Quality,
};

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

// The operation cap lives with the grammar (issue #469);
// the engine's tests reach it from here through `super::*`.
#[cfg(test)]
use crate::url::MAX_OPERATIONS;

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
/// limited decode behind it is the second line. An AVIF
/// container is screened before either (issue #522): its
/// parser reserves what the boxes declare, ahead of every
/// limit here. Public for the fuzz harness's decode target
/// (issue #417).
pub fn decode_image(bytes: &[u8]) -> Result<DynamicImage, EngineError> {
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
    let screened = crate::avif::screen(bytes)?;
    let bytes: &[u8] = &screened;
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
    // Fast paths for the two hottest source formats, taken only
    // where they are output-identical to the generic decode
    // below (the probe above has already enforced the
    // dimension and allocation budgets). Any fast-path failure
    // falls through to the generic decode, so corrupt or exotic
    // files get the generic decode's error behavior.
    match image::guess_format(bytes) {
        Ok(image::ImageFormat::Jpeg) => {
            if let Some(img) = decode_jpeg_direct(bytes) {
                return Ok(img);
            }
        }
        Ok(image::ImageFormat::WebP) => {
            if let Some(img) = decode_webp_direct(bytes) {
                return Ok(img);
            }
        }
        _ => {}
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

/// JPEG decode through zune-jpeg directly: the same decoder
/// the image crate wraps, with the same options the wrapper
/// uses, minus the wrapper's input copy, header pre-pass, and
/// decode-into-buffer plumbing (measured about 28 percent
/// faster at 2400x1600, byte-identical output). Restricted to
/// the YCbCr and RGB inputs the generic path also surfaces as
/// Rgb8; grayscale, CMYK, and anything else returns None and
/// keeps the generic path, as does any zune error (for
/// example arithmetic-coded files zune cannot read).
fn decode_jpeg_direct(bytes: &[u8]) -> Option<DynamicImage> {
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_strict_mode(false)
        .set_max_width(MAX_DECODE_DIMENSION as usize)
        .set_max_height(MAX_DECODE_DIMENSION as usize);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(Cursor::new(bytes), options);
    decoder.decode_headers().ok()?;
    match decoder.input_colorspace()? {
        ColorSpace::YCbCr | ColorSpace::RGB => {}
        _ => return None,
    }
    let (width, height) = decoder.dimensions()?;
    let pixels = decoder.decode().ok()?;
    let buf =
        image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width as u32, height as u32, pixels)?;
    Some(DynamicImage::ImageRgb8(buf))
}

/// WebP still decode through libwebp (the `webp` crate):
/// measured about 3.5x faster than image-webp at 2400x1600
/// with bit-identical pixels, for lossy and lossless files
/// alike. The crate refuses animated files (returns None),
/// which keeps animated sources on the generic path, and its
/// DynamicImage variant matches the generic path (Rgba8 only
/// when the file has alpha).
fn decode_webp_direct(bytes: &[u8]) -> Option<DynamicImage> {
    let decoded = webp::Decoder::new(bytes).decode()?;
    Some(decoded.to_image())
}

/// The media kind an upload validated as (issue #192):
/// what the endpoint binds the object name's extension
/// against, so a validated image cannot be stored under
/// a name that presents it as something else. Every kind
/// carries a binding (issue #466): a body the generic
/// decoder accepts but this table does not name is
/// refused, never stored unbound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadMedia {
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
}

impl UploadMedia {
    /// The name extensions this media kind may be stored
    /// under (lowercase, no dot).
    pub(crate) fn allowed_extensions(self) -> &'static [&'static str] {
        match self {
            UploadMedia::Svg => &["svg"],
            UploadMedia::Heic => &["heic", "heif"],
            UploadMedia::Jpeg => &["jpg", "jpeg"],
            UploadMedia::Png => &["png"],
            UploadMedia::Gif => &["gif"],
            UploadMedia::WebP => &["webp"],
            UploadMedia::Avif => &["avif"],
            UploadMedia::Jxl => &["jxl"],
            UploadMedia::Bmp => &["bmp"],
            UploadMedia::Tiff => &["tif", "tiff"],
            UploadMedia::Ico => &["ico"],
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
/// own spelling, and the fuzz harness's decode target
/// drives every branch through it (issue #417).
pub fn validate_source_image(bytes: &[u8]) -> Result<(), EngineError> {
    prepare_upload(bytes).map(|_| ())
}

/// [`validate_source_image`] that also reports what to
/// store and what the bytes are (issue #192): the upload
/// handler stores the prepared bytes and binds the
/// object name's extension to the media kind.
pub(crate) fn prepare_upload(bytes: &[u8]) -> Result<PreparedUpload, EngineError> {
    ensure_jxl_decoder();
    if crate::svg::is_svg(bytes) {
        // The parse runs in the SVG worker (issue #447); what
        // is stored is the sanitized document it parsed.
        let screened = crate::svg::screen(bytes).map_err(EngineError::Decode)?;
        screened
            .intrinsic_size(None)
            .map_err(|e| EngineError::Decode(e.to_string()))?;
        return Ok(PreparedUpload {
            stored: screened.into_bytes(),
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
    // from the branch detection above. The image crate is
    // built with these formats only (issue #466), so the
    // last arm is unreachable today; it refuses rather
    // than storing a kind no extension is bound to, should
    // a feature change ever compile another decoder in.
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
            other => {
                let format = other.map_or_else(|_| "unknown".to_string(), |f| format!("{f:?}"));
                return Err(EngineError::Decode(format!(
                    "the {format} format is not an upload format this server accepts"
                )));
            }
        }
    };
    Ok(PreparedUpload {
        stored: bytes.to_vec(),
        media,
    })
}

/// The Content-Type the sniff half of upload validation
/// assigns to `bytes` (issue #366): the same classification
/// [`prepare_upload`] walks, spelled as a media type. The
/// upload store records it on the object it writes, because
/// the loader admits store responses by content type and a
/// store that defaults headerless objects to a binary type
/// would otherwise make every upload unloadable. `None`
/// when the bytes match nothing the pipeline reads.
pub(crate) fn sniff_content_type(bytes: &[u8]) -> Option<&'static str> {
    if crate::svg::is_svg(bytes) {
        return Some("image/svg+xml");
    }
    if is_heic(bytes) {
        return Some("image/heic");
    }
    if is_jxl(bytes) {
        return Some("image/jxl");
    }
    image::guess_format(bytes)
        .ok()
        .map(|format| format.to_mime_type())
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
pub struct SourceProbe {
    /// A short lowercase media label: `jpeg`, `png`, `gif`,
    /// `webp`, `avif`, `jxl`, `heic`, `svg`, one of the image
    /// crate's long-tail kinds (`bmp`, `tiff`, `ico`), or
    /// `unknown` when the bytes match nothing the pipeline
    /// reads.
    pub kind: &'static str,
    /// Pixel dimensions when the header announces them.
    pub dimensions: Option<(u32, u32)>,
}

/// How long the explain probe waits for an SVG's intrinsic
/// size: a parse with no render, which takes milliseconds
/// for any document within the bounds.
const SVG_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Probe source bytes for the explain report: kind and
/// dimensions, headers only (see [`SourceProbe`]).
pub(crate) fn probe_source(bytes: &[u8]) -> SourceProbe {
    if crate::svg::is_svg(bytes) {
        // The intrinsic size takes a parse, which runs in the
        // SVG worker (issue #447) under the probe's own short
        // deadline: the explain handler waits on it.
        let deadline = std::time::Instant::now() + SVG_PROBE_TIMEOUT;
        let dimensions = crate::svg::screen(bytes)
            .ok()
            .and_then(|screened| screened.intrinsic_size(Some(deadline)).ok())
            .and_then(|(w, h)| {
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
    // The probe parses an AVIF container too, so it is
    // screened first (issue #522); a refused one states no
    // dimensions.
    let dimensions = crate::avif::screen(bytes).ok().and_then(|bytes| {
        image::ImageReader::new(Cursor::new(&*bytes))
            .with_guessed_format()
            .ok()
            .and_then(|probe| probe.into_dimensions().ok())
    });
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

/// Reject a size the engine works out for a step's output
/// past the per-axis limit a size written in the URL is held
/// to ([`crate::url::MAX_DIMENSION`]), then past the pixel
/// budget (issue #521). Every step that resolves its own
/// size checks it here before allocating it: a resize with a
/// proportional or `orig` side, a fit-in scale, a crop or pad
/// target, a DPR factor on any of them, a rotated canvas.
/// The parser bounds only what the URL spells, and a chain
/// can grow a proportional side past it inside the budget:
/// `16383x1/0x2` keeps the aspect ratio at height 2 and asks
/// for 32766x2, which WebP cannot encode. The step
/// is refused, as a written side past the limit is, rather
/// than clamped to a size nobody asked for.
///
/// `input` is the size the step starts from. A step that
/// ends at that same size works nothing out (`0x0`,
/// `fit-in:0x0`, `rotate(180)`, a width hint or `no_upscale`
/// held at the source), so it is held to the budget only: a
/// source at the 16384 px decode ceiling passes through it
/// as decoded, as it passes through a chain with no sizing
/// step at all.
fn check_output_size(width: u32, height: u32, input: (u32, u32)) -> Result<(), EngineError> {
    let max = crate::url::MAX_DIMENSION;
    if (width, height) != input && (width > max || height > max) {
        return Err(EngineError::Unsupported(format!(
            "requested output {width}x{height} exceeds the {max} pixel limit per axis"
        )));
    }
    check_output_pixels(width, height)
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

/// Reject an operation list past
/// [`crate::url::MAX_OPERATIONS`] (issue #182, audit
/// R2-12). The server refuses such a request at parse time,
/// before any fetch (issue #469); this is the backstop for
/// every other caller of the engine, with the same message
/// and the same `unsupported_operation` class.
fn check_operation_count(ops: &[Operation]) -> Result<(), EngineError> {
    crate::url::check_operation_count(ops).map_err(|e| EngineError::Unsupported(e.to_string()))
}

mod state;
pub use state::{
    InpaintRecord, ObjectsCropRecord, PdfSourceMeta, ProcessContext, ProcessedImage, SegmentObject,
    SegmentRecord, VideoSourceMeta,
};

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
    /// The request ran past its processing deadline
    /// (issue #260): the pipeline checks the deadline
    /// between operations and frames and aborts instead
    /// of running unbounded. The server maps it to 504.
    Timeout,
    /// Signing the output failed because the C2PA timestamp
    /// authority did not answer in time, refused, or sent
    /// something that is not a timestamp (issue #467). A
    /// dependency outage a retry may clear, so the server
    /// maps it to 503 with `Retry-After`. Carries the cause.
    ProvenanceUnavailable(String),
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
            EngineError::Timeout => {
                write!(f, "processing exceeded the time budget")
            }
            EngineError::ProvenanceUnavailable(cause) => {
                write!(f, "C2PA timestamp authority failed: {cause}")
            }
        }
    }
}

impl std::error::Error for EngineError {}

/// Default AVIF speed tier (issue #281): speed 9 encodes
/// about 1.8x faster than 8 for about 15 percent more bytes.
pub const DEFAULT_AVIF_SPEED: u8 = 9;

/// Default JPEG XL speed tier (issue #282): Fastest encodes
/// about 2x faster than jixel's Fast for about 18 percent
/// more bytes.
pub const DEFAULT_JXL_SPEED: jixel::Speed = jixel::Speed::Fastest;

impl Default for ProcessContext<'_> {
    fn default() -> Self {
        ProcessContext {
            preferred: None,
            watermarks: None,
            placeholders: false,
            mask_provider: None,
            upscale_provider: None,
            inpaint_provider: None,
            segment_provider: None,
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
            deadline: None,
        }
    }
}

/// Inference-class operations do not run on animated
/// sources (issue #261): the animated paths apply the
/// chain per frame, so one request would run inference
/// up to 200 times and hold an AI gate permit for
/// minutes to hours. The set is the AI gate's own
/// (`RequestDirectives::uses_ai_beyond_smart_crop`): a
/// smart crop stays available on animated sources under
/// the processing deadline. Its focal analysis is the
/// built-in saliency statistic, or, with a face or focal
/// provider configured, an inference that takes the AI
/// gate like any other (issue #379). An animated image
/// render runs that analysis once, on the first frame,
/// and replays the window (issue #376); a video render
/// still runs it on every frame. A still spelling of the
/// same source (a `frame:` selection renders one frame)
/// runs these operations normally.
fn refuse_animated_ai(directives: &crate::url::RequestDirectives) -> Result<(), EngineError> {
    let inference_bound = directives.removebg
        || directives.studio
        || directives.remove_region
        || directives.upscale
        || directives.needs_face_model
        || directives.has_class_redaction()
        || directives.objects_crop
        || directives.debug_focal;
    if inference_bound {
        return Err(EngineError::Unsupported(
            "AI filters do not run on animated sources".to_string(),
        ));
    }
    Ok(())
}

/// The source bytes the still pipeline decodes (after any
/// frame or page selection) and what sniffing them settled:
/// the format their content declares, and whether they take
/// the HEIC or the SVG decode path (an SVG arrives sanitized
/// and screened; the worker parses it at decode).
struct SniffedSource<'b> {
    bytes: &'b [u8],
    detected: ImageFormat,
    heic: bool,
    svg: Option<crate::svg::Screened>,
}

/// What the decode stage of [`Engine::process_preferred`]
/// hands on.
enum Decoded {
    /// The still image the operation chain runs on (an
    /// animated source's first frame when the output is a
    /// still format), with the source metadata.
    Still(DynamicImage, crate::color::SourceMeta),
    /// An animated source the animated or video path
    /// rendered whole: the request's result.
    Rendered(Box<ProcessedImage>),
}

/// The quality directive resolved for the still encode.
struct StillQuality {
    /// The quality the first encode runs at; `None` keeps
    /// the encoder default.
    encode_quality: Option<u8>,
    /// The quality reported back, set when the service
    /// picked it (`quality(auto)`, the Save-Data step).
    chosen_quality: Option<u8>,
    /// Whether `quality(auto)` verifies its pick through
    /// SSIMULACRA2 (JPEG and JPEG XL outputs).
    perceptual: bool,
}

/// The still encode's result: the bytes, the quality the
/// service chose for them (reported back), and their
/// SSIMULACRA2 score when the perceptual loop verified them.
struct StillEncode {
    bytes: Vec<u8>,
    chosen_quality: Option<u8>,
    ssim: Option<f64>,
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
    ///
    /// The stages run in order: frame and page selection,
    /// sniffing, preflight, decode, the operation chain
    /// (`apply_operations`), the still encode, and metadata
    /// embedding. The functions after this one hold every
    /// stage but the chain.
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
        // A selected video frame or PDF page replaces the
        // source bytes for every stage below.
        let frame = Self::select_frame(bytes, &directives, &ctx)?;
        let bytes = frame.as_ref().map_or(bytes, |(png, _)| png.as_slice());
        let page = Self::select_page(bytes, &directives, &ctx)?;
        let bytes = page.as_ref().map_or(bytes, |(png, _)| png.as_slice());
        let source = Self::sniff_source(bytes)?;
        let detected = source.detected;
        // keep_profile is convert-or-tag (issue #374): the
        // kept profile is embedded at encode time, which
        // only JPEG and PNG outputs carry in this pipeline.
        // When the request resolves to any other output
        // format, keeping the profile would leave the
        // pixels in the source space with nothing labeling
        // them (viewers assume sRGB and show wrong colors),
        // so the conversion to sRGB runs as if the
        // directive were absent, exactly as the filter
        // docs state.
        let resolved_output = directives
            .explicit_format
            .or(ctx.preferred)
            .unwrap_or(detected);
        let keep_profile = directives.keep_profile
            && matches!(resolved_output, ImageFormat::Jpeg | ImageFormat::Png);
        Self::preflight(&directives, &ctx, detected)?;
        let (img, source_meta) =
            match self.decode_source(&source, ops, &ctx, &directives, keep_profile)? {
                Decoded::Still(img, source_meta) => (img, source_meta),
                Decoded::Rendered(processed) => return Ok(*processed),
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
        ctx.check_deadline()?;
        let (img, encode_opts, studio, inpaint, segments, objects_crop) = apply_operations(
            img,
            ops,
            cap,
            dpr_factor(directives.dpr, &ctx),
            &ctx,
            &directives,
            None,
        )?;
        ctx.check_deadline()?;
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
        // returned through `process_video` at decode. There is
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
        let encoded = Self::encode_still(&img, format, &encode_opts, &ctx)?;
        let bytes = Self::embed_source_meta(
            encoded.bytes,
            format,
            encode_opts.keep_profile,
            &source_meta,
            &ctx,
        );
        Ok(ProcessedImage {
            bytes,
            format,
            quality: encoded.chosen_quality,
            width: img.width(),
            height: img.height(),
            source_width,
            source_height,
            placeholder,
            studio,
            inpaint,
            segments,
            objects_crop,
            ssimulacra2: encoded.ssim,
            video: frame.as_ref().map(|(_, meta)| *meta),
            pdf: page.as_ref().map(|(_, meta)| *meta),
            provenance: None,
        })
    }

    /// Frame selection (issue #214), the first stage of
    /// [`Engine::process_preferred`]: a `frame:` segment
    /// makes the source a video whose selected frame
    /// becomes the source image for the rest of the
    /// pipeline. The extractor is required (a deployment
    /// without one answers NotConfigured, like video output
    /// without an encoder); an image source is refused
    /// as a bad source, because frame selection is
    /// video-only and animated GIF and WebP sources keep
    /// their own handling; and the extracted PNG
    /// replaces the bytes, so the format detection, the
    /// validation preflights, the source metadata, and
    /// the decode chain all see the frame. `None` when
    /// the request selects no frame.
    fn select_frame(
        bytes: &[u8],
        directives: &crate::url::RequestDirectives,
        ctx: &ProcessContext<'_>,
    ) -> Result<Option<(Vec<u8>, VideoSourceMeta)>, EngineError> {
        let Some(selection) = directives.frame else {
            return Ok(None);
        };
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
        let video_meta = VideoSourceMeta {
            duration_seconds: frame.duration,
            width: frame.width,
            height: frame.height,
            frame_seconds: frame.timestamp,
        };
        Ok(Some((frame.png, video_meta)))
    }

    /// Page selection (issue #215), after frame selection:
    /// a `page:` segment makes the source a PDF whose
    /// selected page becomes the source image for the rest
    /// of the pipeline, the frame stage's shape exactly. The
    /// renderer is required (a deployment without one
    /// answers NotConfigured, like frame extraction without
    /// a decoder); the engine sniffs before the renderer
    /// spawns anything, so image bytes and non-PDF bytes
    /// are refused as a bad source here and never reach
    /// the tool; and the rendered PNG replaces the bytes,
    /// so the format detection, the validation preflights,
    /// and the decode chain all see the page. `None` when
    /// the request selects no page.
    fn select_page(
        bytes: &[u8],
        directives: &crate::url::RequestDirectives,
        ctx: &ProcessContext<'_>,
    ) -> Result<Option<(Vec<u8>, PdfSourceMeta)>, EngineError> {
        let Some(page) = directives.page else {
            return Ok(None);
        };
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
        let pdf_meta = PdfSourceMeta {
            pages: rendered.pages,
            page: rendered.page,
            page_width: rendered.page_width,
            page_height: rendered.page_height,
        };
        Ok(Some((rendered.png, pdf_meta)))
    }

    /// Sniff the source bytes by content: the format they
    /// declare, and whether they take the HEIC or the SVG
    /// decode path.
    fn sniff_source(bytes: &[u8]) -> Result<SniffedSource<'_>, EngineError> {
        // HEIC/HEIF sources are detected explicitly: they decode
        // through the pure-Rust oxideav stack (issue #163), in the
        // decode chain below, so the format gets its own metadata
        // path instead of the generic one.
        let heic = is_heic(bytes);
        // SVG sources are sniffed by content (the declared type
        // can lie), sanitized, and screened up front: the raster
        // becomes the decoded image below, and the output
        // format defaults to PNG (SVG has no raster format).
        let svg = if crate::svg::is_svg(bytes) {
            Some(crate::svg::screen(bytes).map_err(EngineError::Decode)?)
        } else {
            None
        };
        let detected = if svg.is_some() {
            ImageFormat::Png
        } else {
            detect_format(bytes)
        };
        Ok(SniffedSource {
            bytes,
            detected,
            heic,
            svg,
        })
    }

    /// Preflight: refuse, before any real work, a request
    /// this deployment cannot serve (a feature whose model,
    /// encoder, or font is not configured) or whose
    /// resolved output format cannot carry what it asks
    /// for. The checks run in a fixed order, so a request
    /// that fails several gets the first one's error.
    fn preflight(
        directives: &crate::url::RequestDirectives,
        ctx: &ProcessContext<'_>,
        detected: ImageFormat,
    ) -> Result<(), EngineError> {
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
        // Region removal is validated the same way (issue
        // #176): the URL is fine, the deployment has no
        // inpainting model, so it fails before any work.
        if directives.remove_region && ctx.inpaint_provider.is_none() {
            return Err(EngineError::NotConfigured(
                "remove_region is not configured on this server".to_string(),
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
            let providers = detection_providers(ctx);
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
        // The objects crop is validated the same way (issue
        // #318): the URL is fine but the deployment has no
        // detection model, or the model does not know a
        // named class. Both refuse before any work: the
        // mode is never a silent plain crop when detections
        // were requested and the feature is off. The plain
        // crop fallback covers "the model ran and found
        // none of the classes", decided in the crop itself.
        if directives.objects_crop {
            let providers = detection_providers(ctx);
            if providers.is_empty() {
                return Err(EngineError::NotConfigured(
                    "object crop is not configured on this server (no detection model)".to_string(),
                ));
            }
            for class in &directives.object_classes {
                if resolve_detector(&providers, class).is_none() {
                    return Err(EngineError::Unsupported(format!(
                        "unknown detection class {class:?}: not in the detection model's vocabulary"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Decode the sniffed source into the still image the
    /// operation chain runs on, with the source metadata the
    /// embedding stage reads. An animated GIF or WebP source
    /// whose output stays animated (or becomes video) is
    /// rendered whole by the animated or video path instead,
    /// and comes back as [`Decoded::Rendered`].
    fn decode_source(
        &self,
        source: &SniffedSource<'_>,
        ops: &[Operation],
        ctx: &ProcessContext<'_>,
        directives: &crate::url::RequestDirectives,
        keep_profile: bool,
    ) -> Result<Decoded, EngineError> {
        let bytes = source.bytes;
        let detected = source.detected;
        // Source metadata, read from the raw bytes once:
        // the ICC profile drives color conversion below and
        // (policy permitting) the metadata injection at the
        // end. EXIF and XMP are read only when the policy
        // will keep them (issue #399): an XMP packet can be
        // compressed, and inflating it for nothing is work a
        // source chooses.
        // Replaced wholesale by the HEIC branch of the decode
        // chain, whose metadata comes from the decoder rather
        // than the container bytes.
        let mut source_meta = crate::color::extract_source_meta(bytes, ctx.preserve_metadata);
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
            if frames.len() > 1 {
                let resolved = animated_output_format(
                    ImageFormat::Gif,
                    directives.explicit_format,
                    ctx.preferred,
                );
                if matches!(resolved, ImageFormat::Gif | ImageFormat::WebP) {
                    let processed = self.process_animated(frames, ops, ctx, resolved, None)?;
                    return Ok(Decoded::Rendered(Box::new(processed)));
                }
                if resolved.is_video() {
                    let processed = self.process_video(frames, ops, ctx, resolved)?;
                    return Ok(Decoded::Rendered(Box::new(processed)));
                }
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
            if frames.len() > 1 {
                let resolved = animated_output_format(
                    ImageFormat::WebP,
                    directives.explicit_format,
                    ctx.preferred,
                );
                if matches!(resolved, ImageFormat::WebP | ImageFormat::Gif) {
                    let processed =
                        self.process_animated(frames, ops, ctx, resolved, Some(loop_count))?;
                    return Ok(Decoded::Rendered(Box::new(processed)));
                }
                if resolved.is_video() {
                    let processed = self.process_video(frames, ops, ctx, resolved)?;
                    return Ok(Decoded::Rendered(Box::new(processed)));
                }
            }
            DynamicImage::ImageRgba8(
                frames
                    .into_iter()
                    .next()
                    .expect("decode_webp_frames returns at least one frame")
                    .into_buffer(),
            )
        } else if let Some(screened) = &source.svg {
            // Rasterize at the size the request is heading for
            // (or the intrinsic size when it asks for none), so
            // vectors stay crisp; the normal pipeline runs on
            // the raster from here. The parse and the render
            // run in the SVG worker (issue #447), killed at the
            // request's deadline.
            let dpr = dpr_factor(directives.dpr, ctx);
            screened
                .rasterize(ctx.deadline, |intrinsic| {
                    svg_raster_target(ops, intrinsic, dpr)
                })
                .map_err(|e| match e {
                    crate::svg::RenderError::TimedOut(_) if ctx.deadline.is_some() => {
                        EngineError::Timeout
                    }
                    e => EngineError::Decode(e.to_string()),
                })?
        } else if source.heic {
            // HEIC/HEIF: the oxideav decode applies the EXIF
            // orientation from the decoder's own metadata (the
            // container bytes carry no orientation the generic
            // reader can see) and hands back the source metadata
            // the rest of the pipeline expects. Color management
            // then runs exactly as for the other formats.
            let (mut img, meta) = crate::heic::decode_heic(bytes)?;
            source_meta = meta;
            if let Some(icc) = &source_meta.icc
                && !(keep_profile && crate::color::icc_embeddable(icc))
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
            // A profile the output will not carry (malformed,
            // or past the embed cap, issue #445) converts
            // anyway: untagged pixels would show wrong colors.
            if let Some(icc) = &source_meta.icc
                && !(keep_profile && crate::color::icc_embeddable(icc))
            {
                img = crate::color::convert_to_srgb(img, icc);
            }
            img
        };
        Ok(Decoded::Still(img, source_meta))
    }

    /// Resolve the quality directive against the final image
    /// and the settled output format, for the still encode.
    fn resolve_still_quality(
        img: &DynamicImage,
        format: ImageFormat,
        encode_opts: &EncodeOptions,
        ctx: &ProcessContext<'_>,
    ) -> StillQuality {
        // Resolve the quality directive against the final image.
        // A per-format quality (issue #319) resolves here, now
        // that format negotiation has settled: the entry for
        // this format, else the list's fallback, else no
        // directive in force for this format.
        // `quality(auto)` is reported back: it is the one the caller
        // did not choose themselves.
        let resolved_quality = encode_opts
            .quality
            .as_ref()
            .and_then(|q| q.resolved_for(format));
        let (encode_quality, chosen_quality) = match resolved_quality {
            Some(Quality::Value(q)) => (Some(q), None),
            Some(Quality::Auto) => {
                let q = auto_quality(img);
                (Some(q), Some(q))
            }
            // resolved_for never returns the list itself.
            Some(Quality::PerFormat(_)) => {
                unreachable!("resolved_for resolves per-format quality to a single value")
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
        let perceptual = matches!(resolved_quality, Some(Quality::Auto))
            && matches!(format, ImageFormat::Jpeg | ImageFormat::Jxl);
        StillQuality {
            encode_quality,
            chosen_quality,
            perceptual,
        }
    }

    /// Encode the final still image: one encode at the
    /// resolved quality (or the SSIMULACRA2 bisection for a
    /// perceptual `quality(auto)`), then the `max_bytes`
    /// ladder, all drawing on one per-request encode budget.
    fn encode_still(
        img: &DynamicImage,
        format: ImageFormat,
        encode_opts: &EncodeOptions,
        ctx: &ProcessContext<'_>,
    ) -> Result<StillEncode, EngineError> {
        let StillQuality {
            encode_quality,
            mut chosen_quality,
            perceptual,
        } = Self::resolve_still_quality(img, format, encode_opts, ctx);
        let mut ssim: Option<f64> = None;
        let enc_src = EncodeBuffers::new(img);
        let mut encode_budget = MAX_REQUEST_ENCODES;
        let mut bytes = if perceptual {
            let first_guess = encode_quality.expect("auto quality resolves a value");
            let scored = crate::quality::encode_to_target(
                first_guess,
                ctx.ssim_target,
                &mut encode_budget,
                |q| {
                    // Between attempts, not just between
                    // stages (issue #322): a bisection that
                    // ignores the deadline would keep
                    // encoding long past it.
                    ctx.check_deadline()?;
                    let candidate = encode(
                        &enc_src,
                        format,
                        Some(q),
                        encode_opts.fill,
                        encode_opts.palette,
                        encode_opts.format_options,
                        ctx.avif_speed,
                        ctx.jxl_speed,
                    )?;
                    let score = decode_image(&candidate)
                        .ok()
                        .and_then(|decoded| crate::quality::ssimulacra2(img, &decoded));
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
                encode_opts.palette,
                encode_opts.format_options,
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
        // last allowed attempt always lands on the quality floor
        // instead of wherever the encode pool runs out, and a
        // tight budget is never left silently unchased (issue
        // #229).
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
                // Between ladder attempts, not just between
                // stages (issue #322): each re-encode is a
                // fresh chance to honor the deadline.
                ctx.check_deadline()?;
                let steps_remaining =
                    (max_quality_attempts(format) - attempts).min(encode_budget as u32);
                current = next_quality(current, steps_remaining);
                bytes = encode(
                    &enc_src,
                    format,
                    Some(current),
                    encode_opts.fill,
                    encode_opts.palette,
                    encode_opts.format_options,
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
                .and_then(|decoded| crate::quality::ssimulacra2(img, &decoded));
        }
        Ok(StillEncode {
            bytes,
            chosen_quality,
            ssim,
        })
    }

    /// Metadata and profile embedding happen on the final
    /// bytes: the ICC goes in when the request kept the
    /// profile (it was not converted away), preserved
    /// metadata when the operator policy allows it. Only
    /// JPEG and PNG outputs carry either in this pass.
    fn embed_source_meta(
        bytes: Vec<u8>,
        format: ImageFormat,
        keep_profile: bool,
        source_meta: &crate::color::SourceMeta,
        ctx: &ProcessContext<'_>,
    ) -> Vec<u8> {
        if (keep_profile && source_meta.icc.is_some()) || ctx.preserve_metadata {
            let embed_icc = if keep_profile {
                source_meta.icc.as_deref()
            } else {
                None
            };
            match format {
                ImageFormat::Jpeg => {
                    crate::color::inject_jpeg(&bytes, embed_icc, source_meta, ctx.preserve_metadata)
                }
                ImageFormat::Png => {
                    crate::color::inject_png(&bytes, embed_icc, source_meta, ctx.preserve_metadata)
                }
                _ => bytes,
            }
        } else {
            bytes
        }
    }

    /// Process an animated source (GIF or WebP) frame by frame:
    /// every frame runs the same operations and keeps its delay,
    /// so the output stays animated in `format`. A WebP output
    /// carries the source's loop count over and honors the
    /// request quality like the still WebP path does (issue
    /// #323); GIF output keeps the GIF encoder's default, as it
    /// always has, and has no quality lever.
    fn process_animated(
        &self,
        frames: Vec<image::Frame>,
        ops: &[Operation],
        ctx: &ProcessContext<'_>,
        format: ImageFormat,
        loop_count: Option<u16>,
    ) -> Result<ProcessedImage, EngineError> {
        let directives = crate::url::RequestDirectives::analyze(ops);
        refuse_animated_ai(&directives)?;
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
        // The encoding directives are a property of the ops,
        // not of a frame, so every frame reports the same ones;
        // the first frame's copy is the request's.
        let mut encode_opts = EncodeOptions::default();
        // One smart crop window per crop for the whole
        // animation (issue #376): the first frame decides,
        // the rest replay. One trim window per trim (issue
        // #465): the union over every frame, settled before
        // any frame renders, so the frames share one canvas.
        let mut share = FrameShare {
            smart_windows: Some(SmartWindows::default()),
            ..FrameShare::default()
        };
        survey_trims(&frames, ops, cap, dpr, ctx, &directives, &mut share)?;
        for frame in frames {
            ctx.check_deadline()?;
            let delay = frame.delay();
            let img = DynamicImage::ImageRgba8(frame.into_buffer());
            let (img, opts, _, _, _, _) =
                apply_operations(img, ops, cap, dpr, ctx, &directives, Some(&mut share))?;
            if out_frames.is_empty() {
                encode_opts = opts;
            }
            budget.charge(img.width(), img.height())?;
            out_frames.push(image::Frame::from_parts(img.to_rgba8(), 0, 0, delay));
        }
        // Resolve the quality directive for a WebP output the
        // same way the still path does: an explicit value is
        // used as given, `quality(auto)` is picked from the
        // first processed frame (the frame the placeholder also
        // describes) and reported back, and Save-Data steps an
        // unspecified quality down one notch. GIF output has no
        // quality lever, so nothing resolves for it.
        let (encode_quality, chosen_quality) = if format == ImageFormat::WebP {
            match encode_opts
                .quality
                .as_ref()
                .and_then(|q| q.resolved_for(format))
            {
                Some(Quality::Value(q)) => (Some(q), None),
                Some(Quality::Auto) => {
                    let first = out_frames
                        .first()
                        .expect("process_animated is called with at least one frame");
                    let q = auto_quality(&DynamicImage::ImageRgba8(first.buffer().clone()));
                    (Some(q), Some(q))
                }
                // resolved_for never returns the list itself.
                Some(Quality::PerFormat(_)) => {
                    unreachable!("resolved_for resolves per-format quality to a single value")
                }
                None if ctx.save_data => (Some(SAVE_DATA_QUALITY), Some(SAVE_DATA_QUALITY)),
                None => (None, None),
            }
        } else {
            (None, None)
        };
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
                let first = out_frames
                    .first()
                    .expect("process_animated is called with at least one frame");
                let (canvas_w, canvas_h) = (first.buffer().width(), first.buffer().height());
                let mut enc = gif::Encoder::new(
                    &mut out,
                    u16::try_from(canvas_w).map_err(|_| {
                        EngineError::Encode("gif: width exceeds the 65535 format limit".to_string())
                    })?,
                    u16::try_from(canvas_h).map_err(|_| {
                        EngineError::Encode(
                            "gif: height exceeds the 65535 format limit".to_string(),
                        )
                    })?,
                    &[],
                )
                .map_err(|e| EngineError::Encode(e.to_string()))?;
                for f in &out_frames {
                    // The deadline is checked per frame inside
                    // the encode loop itself (issue #322): a
                    // many-frame GIF encode is chunked work, so
                    // it aborts at the deadline instead of
                    // running every frame's quantization first
                    // and being noticed only afterwards.
                    ctx.check_deadline()?;
                    // GIF delays are hundredths of a second; the
                    // image crate saturated this conversion at
                    // u16::MAX, and so does this.
                    let (numer, denom) = f.delay().numer_denom_ms();
                    let delay = u16::try_from(numer / denom / 10).unwrap_or(u16::MAX);
                    let buffer = f.buffer();
                    let frame = gif_frame_quantized(
                        buffer.as_raw(),
                        buffer.width(),
                        buffer.height(),
                        delay,
                    )?;
                    enc.write_frame(&frame)
                        .map_err(|e| EngineError::Encode(e.to_string()))?;
                }
                drop(enc);
                out
            }
            ImageFormat::WebP => encode_animated_webp(
                out_frames,
                loop_count.unwrap_or(0),
                encode_quality,
                encode_opts.format_options,
                ctx.deadline,
            )?,
            _ => unreachable!("process_animated is only called for GIF and WebP"),
        };
        Ok(ProcessedImage {
            bytes: out,
            format,
            quality: chosen_quality,
            width,
            height,
            source_width,
            source_height,
            placeholder,
            studio: None,
            inpaint: None,
            segments: None,
            objects_crop: None,
            ssimulacra2: None,
            video: None,
            pdf: None,
            provenance: None,
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
        refuse_animated_ai(&directives)?;
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
        // The frames share their trim windows (issue #465),
        // as on the animated image paths: the encoder takes
        // one frame size.
        let mut share = FrameShare::default();
        survey_trims(&frames, ops, cap, dpr, ctx, &directives, &mut share)?;
        for frame in frames {
            ctx.check_deadline()?;
            let delay_ms = std::time::Duration::from(frame.delay()).as_millis() as u64;
            let img = DynamicImage::ImageRgba8(frame.into_buffer());
            let (img, _, _, _, _, _) =
                apply_operations(img, ops, cap, dpr, ctx, &directives, Some(&mut share))?;
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
            inpaint: None,
            segments: None,
            objects_crop: None,
            ssimulacra2: None,
            video: None,
            pdf: None,
            provenance: None,
        })
    }
}

/// Collect an animation decoder's frames under the shared
/// limits: a frame cap, a total-decoded-pixels budget, and
/// the decode allocation budget charged on the retained
/// RGBA bytes, so a small file cannot expand into gigabytes
/// of frames. The pixel budget alone is not a memory bound:
/// at its 100M-pixel cap the retained frames are 400 MiB,
/// well past the 256 MiB allocation budget every still
/// decode honors. Errors when the source is malformed, over
/// budget, or carries no frames. `format` names the source
/// format in error messages.
fn collect_frames(
    frames: impl Iterator<Item = Result<image::Frame, image::ImageError>>,
    format: &str,
) -> Result<Vec<image::Frame>, EngineError> {
    let mut collected = Vec::new();
    let mut total_pixels: u64 = 0;
    let mut total_bytes: u64 = 0;
    for frame in frames {
        let frame = frame.map_err(|e| EngineError::Decode(e.to_string()))?;
        if collected.len() >= MAX_ANIMATED_FRAMES {
            return Err(EngineError::Decode(format!(
                "{format} has more than {MAX_ANIMATED_FRAMES} frames"
            )));
        }
        let frame_pixels = frame.buffer().width() as u64 * frame.buffer().height() as u64;
        total_pixels += frame_pixels;
        if total_pixels > MAX_ANIMATED_PIXELS {
            return Err(EngineError::Decode(format!(
                "{format} frames exceed the {MAX_ANIMATED_PIXELS} decoded-pixel limit"
            )));
        }
        total_bytes += frame_pixels * 4;
        if total_bytes > MAX_DECODE_ALLOC {
            return Err(EngineError::Decode(format!(
                "{format} frames exceed the {MAX_DECODE_ALLOC} byte decode allocation limit"
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

/// Whether the source bytes are an animated GIF: a GIF with
/// a second frame, which is what sends [`decode_gif_frames`]
/// output down the animated paths. A block walk, not a
/// decode: the gif crate (the parser the decoder itself
/// uses) reads frame descriptors with LZW decoding skipped,
/// and the walk stops at the second frame, so it costs one
/// pass over the first frame's compressed bytes and
/// allocates no pixels. The server uses it, beside
/// [`is_animated_webp`], to route animated WebP encodes from
/// GIF sources to their gate (issue #405) before any engine
/// work runs. A file that does not parse that far reads as
/// not animated; the decode then refuses it anyway, or
/// renders a still.
pub(crate) fn is_animated_gif(bytes: &[u8]) -> bool {
    if !bytes.starts_with(b"GIF") {
        return false;
    }
    let mut options = gif::DecodeOptions::new();
    options.skip_frame_decoding(true);
    let Ok(mut decoder) = options.read_info(bytes) else {
        return false;
    };
    for _ in 0..2 {
        if !matches!(decoder.next_frame_info(), Ok(Some(_))) {
            return false;
        }
    }
    true
}

/// The format an animated source renders to while it keeps
/// its frames (or goes to video): an explicit `format(...)`
/// first, then the negotiated preference, then the source's
/// own format. Negotiation never flattens an animation
/// (issue #370): a negotiated GIF or WebP keeps the frames
/// in that format, and any other negotiated format (AVIF and
/// JPEG XL cannot carry them) falls back to the source
/// format. An explicit format always wins; a still format
/// then converts the first frame, as documented for still
/// conversions. The engine's animated branches and the
/// server's animation gate predicate (issue #405) both
/// resolve through this one function, so the gate cannot
/// drift from the encoder it guards again.
pub(crate) fn animated_output_format(
    source: ImageFormat,
    explicit: Option<ImageFormat>,
    preferred: Option<ImageFormat>,
) -> ImageFormat {
    match (explicit, preferred) {
        (Some(format), _) => format,
        (None, Some(format @ (ImageFormat::Gif | ImageFormat::WebP))) => format,
        (None, _) => source,
    }
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

/// Encode frames as an animated WebP through libwebp's
/// animated encoder (the `webp` crate's `AnimEncoder`, method
/// 4; issue #323). The pure-Rust alternative (oxideav-webp)
/// writes lossless VP8L frames only and has no lossy
/// animation mode, so animated output through it has no
/// quality lever, and a cold encode of a real animation ran past the
/// 600 s processing deadline on the production soak host
/// (issue #310). The quality mapping matches the still path
/// in `encode::encode_webp`: the default quality is 80, quality
/// below 100 is a lossy encode at that quality, and quality
/// 100 is a lossless encode through the same libwebp encoder.
/// The lossless corner stays on libwebp because its animated
/// lossless encode is faster than the oxideav lossless
/// encode, not slower: on the soak site files the oxideav
/// encode did not finish inside the 600 s deadline at all,
/// while libwebp lossless took 23.6 s for the 40 frame mascot
/// and 148.5 s for the 120 frame snap-toy (measured in the
/// bench recorded on issue #323). That is the opposite of the
/// roughly 92x gap the still lossless path measured in issue
/// #292, so no second encoder stays wired in for one corner.
/// The swap follows the #292 pattern and is a temporary
/// measure: the animated encode returns to oxideav-webp once
/// a release closes the gap to roughly 1.3x of libwebp method
/// 4 at equal or better size and quality.
/// `loop_count` is carried over from the source (0 loops
/// forever) and set on the muxer, which is where the ANIM
/// loop count lives. Frames arrive already composited onto
/// the full canvas by the decode side, so each one is handed
/// to the encoder as a full-canvas RGBA frame and alpha is
/// preserved.
/// Frame timestamps are the accumulated start times of each
/// frame, with one adjustment: the timeline starts at minus
/// the total duration instead of at zero. The `webp` crate
/// finishes an animation by adding a null frame at timestamp
/// 0, and libwebp takes the last frame's duration from that
/// final timestamp; starting below zero makes the final call
/// land exactly on the end of the last frame through the
/// uint32 wrap-around libwebp's timestamp arithmetic allows,
/// so the last frame keeps its own duration instead of being
/// assigned the average of the frames before it. A total
/// duration beyond the i32 timestamp range (an animation
/// longer than about 24 days) falls back to a zero-based
/// timeline and that average.
fn encode_animated_webp(
    frames: Vec<image::Frame>,
    loop_count: u16,
    quality: Option<u8>,
    format_options: FormatOptions,
    deadline: Option<std::time::Instant>,
) -> Result<Vec<u8>, EngineError> {
    let first = frames
        .first()
        .ok_or_else(|| EngineError::Encode("webp: animation has no frames".to_string()))?;
    let (width, height) = (first.buffer().width(), first.buffer().height());
    // VP8 frames carry 14-bit dimensions, as on the still path.
    if width > 16383 || height > 16383 {
        return Err(EngineError::Encode(
            "webp: dimension exceeds the 16383 format limit".into(),
        ));
    }
    let q = quality.unwrap_or(80);
    let mut config = webp::WebPConfig::new().map_err(|_| {
        EngineError::Encode("webp: libwebp rejected its default configuration".into())
    })?;
    config.quality = q as f32;
    config.method = format_options.webp_method.map(i32::from).unwrap_or(4);
    if let Some(sns) = format_options.webp_sns {
        config.sns_strength = i32::from(sns);
    }
    if q >= 100 {
        config.lossless = 1;
    }
    let delays_ms: Vec<i64> = frames
        .iter()
        .map(|frame| {
            let delay: std::time::Duration = frame.delay().into();
            i64::try_from(delay.as_millis()).unwrap_or(i64::MAX)
        })
        .collect();
    let total_ms: i64 = delays_ms.iter().fold(0, |acc, d| acc.saturating_add(*d));
    let mut timestamp_ms: i64 = if total_ms <= i64::from(i32::MAX) {
        -total_ms
    } else {
        0
    };
    let mut encoder = webp::AnimEncoder::new(width, height, &config);
    encoder.set_loop_count(i32::from(loop_count));
    for (frame, delay_ms) in frames.iter().zip(&delays_ms) {
        // The deadline is checked per frame while the
        // frames are handed over (issue #322). The final
        // `try_encode` below is a single libwebp call with
        // no cancellation hook, so it remains one opaque
        // stage: it cannot be interrupted mid-call, and
        // the server side bounds what it can hold (see
        // `run_process` in server/process.rs).
        if let Some(deadline) = deadline
            && std::time::Instant::now() >= deadline
        {
            return Err(EngineError::Timeout);
        }
        let buffer = frame.buffer();
        if buffer.width() != width || buffer.height() != height {
            return Err(EngineError::Encode(
                "webp: animated frames do not share one canvas".into(),
            ));
        }
        encoder.add_frame(webp::AnimFrame::from_rgba(
            buffer.as_raw(),
            width,
            height,
            i32::try_from(timestamp_ms).unwrap_or(i32::MAX),
        ));
        timestamp_ms = timestamp_ms.saturating_add(*delay_ms);
    }
    let encoded = encoder
        .try_encode()
        .map_err(|e| EngineError::Encode(format!("webp: {e:?}")))?;
    Ok(encoded.to_vec())
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

/// The effective DPR factor: an explicit `dpr(...)` directive
/// wins, then the client hint carried on the context, else 1.
fn dpr_factor(explicit: Option<f32>, ctx: &ProcessContext<'_>) -> f32 {
    explicit.or(ctx.dpr).unwrap_or(1.0)
}

/// Whether a filter leaves every existing pixel where it
/// was (issue #373): the detection memos key on the image
/// dimensions, so a chain step that moves content (a
/// flip, a rotation, a trim, an upscale, a border, an
/// overlay a face could hide behind) must invalidate
/// them, or the next detection consumer is served boxes
/// measured on a state the pixels have left, and a
/// redaction lands where the object no longer is.
/// Photometric filters, the detection consumers
/// themselves, region filters (their coordinates are the
/// request's own), and encode-time directives preserve
/// positions. Anything not listed here invalidates: a
/// new filter variant errs toward a fresh inference,
/// never toward stale boxes.
fn filter_preserves_positions(filter: &Filter) -> bool {
    matches!(
        filter,
        Filter::Grayscale
            | Filter::Invert
            | Filter::Blur(_)
            | Filter::Brighten(_)
            | Filter::Contrast(_)
            | Filter::Gamma(_)
            | Filter::Exposure(_)
            | Filter::HighlightsShadows(..)
            | Filter::Vibrance(_)
            | Filter::HueRotate(_)
            | Filter::Sharpen(_)
            | Filter::Saturation(_)
            | Filter::Rgb(..)
            | Filter::Curve(..)
            | Filter::Convolution(..)
            | Filter::Noise(_)
            | Filter::Equalize
            | Filter::Enhance
            | Filter::Sepia
            | Filter::Duotone(..)
            | Filter::Monochrome(_)
            | Filter::Pixelate(_)
            | Filter::Fill(_)
            | Filter::Radius(_)
            | Filter::Mask(_)
            | Filter::BlurFaces
            | Filter::PixelateFaces
            | Filter::BlurClass(_)
            | Filter::PixelateClass(_)
            | Filter::RecolorClass(..)
            | Filter::FacePad(_)
            | Filter::BlurRegion { .. }
            | Filter::PixelateRegion { .. }
            | Filter::RemoveRegion { .. }
            | Filter::KeepProfile
            | Filter::Dpr(_)
            | Filter::MaxBytes(_)
            | Filter::Quality(_)
            | Filter::Palette(_)
            | Filter::Format(_)
            | Filter::NoUpscale
    )
}

/// Whether a filter leaves every pixel where it was, so a
/// pending focal form stays put (issue #465): the
/// position-preserving filters of #373, plus the ones that
/// paint over the frame in place (watermark, overlay, text,
/// border, background removal). Those add content a detector
/// could find, which is why #373 re-detects after them, but
/// they move nothing. Anything else drops the form: a new
/// filter errs toward the crop's own alignment, never toward
/// a stale position.
fn filter_keeps_focal(filter: &Filter) -> bool {
    filter_preserves_positions(filter)
        || matches!(
            filter,
            Filter::Watermark { .. }
                | Filter::Overlay { .. }
                | Filter::Text { .. }
                | Filter::Border(..)
                | Filter::RemoveBg
        )
}

/// Drop every detection memo: the chain step that just
/// ran moved the content the memos' boxes describe.
fn invalidate_detection_memos(
    face_memo: &mut FaceMemo,
    detection_memos: &mut [DetectionMemo],
    segment_memo: &mut SegmentMemo,
) {
    *face_memo = None;
    detection_memos.iter_mut().for_each(|memo| *memo = None);
    *segment_memo = None;
}

/// Memoized face detection for one pipeline run: the image
/// dimensions the detections belong to, and the provider's
/// result. Several face operations on the same image state
/// share one inference; an operation that changes the size
/// re-detects against the new state, and a step that moves
/// content invalidates the memo outright (see
/// `filter_preserves_positions`).
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

/// Memoized image encoding for one pipeline run (issue
/// #175): the image dimensions the embedding belongs to,
/// and the segment provider's result. The encoder is the
/// expensive half of segmentation (the decoder runs per
/// box), so chained class operations on the same image
/// state share one encoding, like detection memoization.
type SegmentMemo = Option<(
    u32,
    u32,
    Result<crate::ai::SegmentEmbedding, crate::ai::AiError>,
)>;

/// Segment one detection's box against the memoized
/// encoding of `img`. A provider failure is an inference
/// error: on a redaction request that fails the request,
/// the same rule detection follows (never an unredacted,
/// or silently rectangle-redacted, response).
fn segment_detection(
    memo: &mut SegmentMemo,
    provider: &dyn crate::ai::SegmentProvider,
    img: &DynamicImage,
    det: &crate::ai::Detection,
) -> Result<image::GrayImage, EngineError> {
    let dims = (img.width(), img.height());
    let stale = !matches!(memo, Some((w, h, _)) if (*w, *h) == dims);
    if stale {
        *memo = Some((dims.0, dims.1, provider.encode(img)));
    }
    let (_, _, cached) = memo.as_ref().expect("memo just populated");
    let embedding = cached
        .as_ref()
        .map_err(|e| EngineError::Inference(e.clone()))?;
    provider
        .decode(embedding, dims.0, dims.1, (det.x, det.y, det.w, det.h))
        .map_err(EngineError::Inference)
}

/// The detections of one class operation with their
/// segmentation masks, plus the record entries the masks
/// produced (issue #175). The geometry is measured on
/// each mask and stated in image coordinates: the tight
/// box and centroid are the mask geometry crop anchoring
/// consumes. A mask in which the model found no object
/// (nothing over half strength) still redacts its box's
/// soft shape, it just contributes no geometry, so it
/// leaves no record entry.
fn segment_for_class(
    memo: &mut SegmentMemo,
    provider: &dyn crate::ai::SegmentProvider,
    img: &DynamicImage,
    detections: &[crate::ai::Detection],
) -> Result<(Vec<crate::filters::MaskedDetection>, Vec<SegmentObject>), EngineError> {
    let mut masked = Vec::with_capacity(detections.len());
    let mut objects = Vec::new();
    for det in detections {
        let mask = segment_detection(memo, provider, img, det)?;
        if let Some(geometry) = crate::ai::mask_geometry(&mask) {
            let (tx, ty, tw, th) = geometry.tight;
            objects.push(SegmentObject {
                label: det.label.clone(),
                score: det.score,
                detection: (det.x, det.y, det.w, det.h),
                tight_box: (det.x + tx, det.y + ty, tw, th),
                coverage: geometry.coverage,
                centroid: (
                    det.x + (geometry.centroid.0 * det.w as f32).round() as u32,
                    det.y + (geometry.centroid.1 * det.h as f32).round() as u32,
                ),
            });
        }
        masked.push(crate::filters::MaskedDetection {
            detection: det.clone(),
            mask,
        });
    }
    Ok((masked, objects))
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

/// What [`apply_operations`] hands back: the rendered
/// image, its encode options, and the records the
/// operations produced (studio, inpaint, segments) for
/// the meta response.
type ApplyOutcome = (
    DynamicImage,
    EncodeOptions,
    Option<crate::studio::StudioRecord>,
    Option<InpaintRecord>,
    Option<SegmentRecord>,
    Option<ObjectsCropRecord>,
);

/// A crop+resize fusion the chain qualifies for (issue
/// #287): the rectangle a `crop-rect` cuts, resampled to the
/// chain's resize target in a single fast_image_resize call
/// through its crop box, instead of materializing the window
/// and resampling that. The backend computes its convolution
/// coefficients against the full-source geometry when a crop
/// box is set, so the fused result differs from the two-step
/// chain by rounding-class deltas, measured at 1.0 to 3.5
/// percent of pixels differing, a max
/// delta of 4 on downscales and 10 on upscales, and a mean
/// delta of at most 0.027, for a 1.5 to 3.6x speedup on the
/// fused shapes. Any chain the gates in [`fusion_plan`] do
/// not accept runs the two-step path untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FusionPlan {
    /// Index of the `crop-rect` operation in the chain.
    crop_index: usize,
    /// Index of the resize operation in the chain.
    size_index: usize,
    /// The crop window: (left, top, width, height).
    window: (u32, u32, u32, u32),
}

/// Decide whether the chain's crop and resize fuse (issue
/// #287). The gates, one per condition:
///
/// - G1: exactly one crop in the chain, and it is a
///   `crop-rect`. Its window is static in the operation; an
///   aspect crop's window depends on focal and detection
///   state decided at run time, so it never fuses.
/// - G2: exactly one resize (`Operation::Size`) in the
///   chain.
/// - G3: the crop precedes the resize, and only decode-time
///   no-ops (`frame`, `page`) sit between them: no pixel
///   operation may run on the window before the resample.
/// - G4: the resize is the chain's final pixel operation:
///   only decode-time no-ops follow it.
/// - G5: no operation after the crop consumes the full
///   frame's geometry. A focal rectangle validates against
///   the frame in effect where it sits, a pad builds a
///   canvas from it, filters read regions and detections
///   from it; under fusion the frame at those points is the
///   un-cropped one, so any of them after the crop refuses
///   the fusion. G3 and G4 are this gate's two halves; it is
///   checked as its own scan so the condition stands on its
///   own.
/// - G6: the chain is not a debug chain (`debug:focal`
///   renders a diagnostic of the exact pixels, so it stays
///   on the two-step path), and the rectangle has positive
///   area.
fn fusion_plan(ops: &[Operation]) -> Option<FusionPlan> {
    // G1: one crop, and it is a crop-rect.
    let mut crop: Option<(usize, (u32, u32, u32, u32))> = None;
    for (index, op) in ops.iter().enumerate() {
        match op {
            Operation::CropRect {
                left,
                top,
                right,
                bottom,
            } => {
                if crop.is_some() {
                    return None;
                }
                crop = Some((index, (*left, *top, *right, *bottom)));
            }
            Operation::Crop { .. } => return None,
            _ => {}
        }
    }
    let (crop_index, (left, top, right, bottom)) = crop?;
    // G2: one resize.
    let mut size_index: Option<usize> = None;
    for (index, op) in ops.iter().enumerate() {
        if matches!(op, Operation::Size(_)) {
            if size_index.is_some() {
                return None;
            }
            size_index = Some(index);
        }
    }
    let size_index = size_index?;
    // G3: the crop precedes the resize, with only decode-time
    // no-ops between them.
    if crop_index > size_index {
        return None;
    }
    let decode_noop = |op: &Operation| matches!(op, Operation::Frame(_) | Operation::Page(_));
    if !ops[crop_index + 1..size_index].iter().all(decode_noop) {
        return None;
    }
    // G4: the resize is the final pixel operation.
    if !ops[size_index + 1..].iter().all(decode_noop) {
        return None;
    }
    // G5: nothing after the crop consumes the full frame.
    for (index, op) in ops.iter().enumerate().skip(crop_index + 1) {
        if index != size_index && !decode_noop(op) {
            return None;
        }
    }
    // G6: no debug chain, and the rectangle has positive
    // area.
    if ops.iter().any(|op| matches!(op, Operation::DebugFocal)) {
        return None;
    }
    if right <= left || bottom <= top {
        return None;
    }
    Some(FusionPlan {
        crop_index,
        size_index,
        window: (left, top, right - left, bottom - top),
    })
}

/// One shared smart crop window: the chain position and
/// frame dimensions it was decided on, and the window
/// itself as (x, y, width, height).
#[derive(Debug)]
struct SmartWindow {
    op_index: usize,
    width: u32,
    height: u32,
    window: (u32, u32, u32, u32),
}

/// Smart crop windows shared across the frames of one
/// animated render (issue #376): the first frame's smart
/// crop decides its window, and every later frame replays
/// that window instead of running the focal analysis
/// again, so the crop cannot jitter frame to frame and
/// the analysis behind it (saliency, plus face detection
/// when a provider feeds it) runs once per render, not
/// once per frame. Entries key on the crop's position in
/// the chain and the frame dimensions it decided on, so
/// a frame of a different size decides for itself.
#[derive(Debug, Default)]
struct SmartWindows {
    entries: Vec<SmartWindow>,
}

impl SmartWindows {
    fn get(&self, key: (usize, u32, u32)) -> Option<(u32, u32, u32, u32)> {
        self.entries
            .iter()
            .find(|e| (e.op_index, e.width, e.height) == key)
            .map(|e| e.window)
    }

    fn set(&mut self, key: (usize, u32, u32), window: (u32, u32, u32, u32)) {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|e| (e.op_index, e.width, e.height) == key)
        {
            entry.window = window;
        } else {
            self.entries.push(SmartWindow {
                op_index: key.0,
                width: key.1,
                height: key.2,
                window,
            });
        }
    }
}

/// A rectangle of an image as (left, top, width, height).
type Window = (u32, u32, u32, u32);

/// One trim's shared window: the trim's position in the
/// chain (operation and filter indexes), the frame
/// dimensions it was surveyed at, and the union of the boxes
/// the frames' own tolerance passes keep; `None` when every
/// frame was uniform.
#[derive(Debug)]
struct TrimWindow {
    key: (usize, usize),
    dims: (u32, u32),
    found: Option<Window>,
}

/// Trim windows shared across the frames of one animated
/// render (issue #465). Trim cuts a box that depends on the
/// content, so frames trimmed one at a time came out at
/// different sizes: the GIF screen took the first frame's
/// size and misplaced or clipped the rest, and the animated
/// WebP and video encoders refused the frames. Each window
/// is settled by [`survey_trims`] before any frame renders,
/// as the union of every frame's own box, so one cut holds
/// every frame's content and the frames stay registered on
/// one canvas. Entries key on the trim's chain position and
/// the frame dimensions, so a frame of a different size
/// trims for itself.
#[derive(Debug, Default)]
struct TrimWindows {
    entries: Vec<TrimWindow>,
}

impl TrimWindows {
    /// The settled box for the trim at `key` on a frame of
    /// `dims`, or `None` when no window is settled for it.
    fn get(&self, key: (usize, usize), dims: (u32, u32)) -> Option<Option<Window>> {
        self.entries
            .iter()
            .find(|e| e.key == key && e.dims == dims)
            .map(|e| e.found)
    }

    fn settle(&mut self, key: (usize, usize), dims: (u32, u32), found: Option<Window>) {
        self.entries.retain(|e| e.key != key);
        self.entries.push(TrimWindow { key, dims, found });
    }
}

/// The geometry the frames of one animated render share, so
/// the frames come out on one canvas, cut from the same part
/// of the scene.
#[derive(Debug, Default)]
struct FrameShare {
    /// One smart crop window per crop (issue #376). `None`
    /// on the video path, whose frames still decide their
    /// smart crops one at a time.
    smart_windows: Option<SmartWindows>,
    /// One trim window per trim filter (issue #465).
    trims: TrimWindows,
}

/// The smallest box holding both boxes; `None` stands for no
/// box at all.
fn union_box(a: Option<Window>, b: Option<Window>) -> Option<Window> {
    match (a, b) {
        (Some((ax, ay, aw, ah)), Some((bx, by, bw, bh))) => {
            let (left, top) = (ax.min(bx), ay.min(by));
            let right = (ax + aw).max(bx + bw);
            let bottom = (ay + ah).max(by + bh);
            Some((left, top, right - left, bottom - top))
        }
        (one, None) | (None, one) => one,
    }
}

/// Settle the trim windows of an animated render (issue
/// #465) before its frames render: for each trim filter in
/// chain order, run every frame up to that filter (the trims
/// before it replaying their settled windows, so the survey
/// sees each frame as the render will) and record the union
/// of the boxes the frames' own tolerance passes keep. The
/// cost is one extra chain prefix per trim per frame, under
/// the render's own deadline. A lone frame has nothing to
/// share and trims for itself.
fn survey_trims(
    frames: &[image::Frame],
    ops: &[Operation],
    cap: Option<(u32, u32)>,
    dpr: f32,
    ctx: &ProcessContext<'_>,
    directives: &crate::url::RequestDirectives,
    share: &mut FrameShare,
) -> Result<(), EngineError> {
    if frames.len() < 2 {
        return Ok(());
    }
    check_operation_count(ops)?;
    for (op_index, op) in ops.iter().enumerate() {
        let Operation::Filters(filters) = op else {
            continue;
        };
        for (filter_index, filter) in filters.iter().enumerate() {
            let tolerance = match filter {
                Filter::Trim(tolerance) | Filter::TrimSides(tolerance, _) => *tolerance,
                _ => continue,
            };
            let key = (op_index, filter_index);
            // The first frame's dimensions and the union so far.
            let mut surveyed: Option<((u32, u32), Option<Window>)> = None;
            for frame in frames {
                ctx.check_deadline()?;
                let img = DynamicImage::ImageRgba8(frame.buffer().clone());
                let mut chain = ChainState::new(ops, cap, dpr, ctx, directives, Some(&mut *share));
                let img = chain.run_to(img, ops, key)?;
                let dims = (img.width(), img.height());
                let found = crate::filters::trim_box(&img, tolerance);
                match &mut surveyed {
                    None => surveyed = Some((dims, found)),
                    Some((first, union)) if *first == dims => *union = union_box(*union, found),
                    // Dimensions at a trim follow from the
                    // chain alone, so frames agree on them;
                    // one that does not trims for itself.
                    Some(_) => {}
                }
            }
            if let Some((dims, found)) = surveyed {
                share.trims.settle(key, dims, found);
            }
        }
    }
    Ok(())
}

/// Apply the operations to one image, collecting the encoding
/// directives along the way. Shared by the still pipeline and
/// the animated paths (GIF and WebP), which run it once per
/// frame.
fn apply_operations(
    mut img: DynamicImage,
    ops: &[Operation],
    cap: Option<(u32, u32)>,
    dpr: f32,
    ctx: &ProcessContext<'_>,
    directives: &crate::url::RequestDirectives,
    share: Option<&mut FrameShare>,
) -> Result<ApplyOutcome, EngineError> {
    check_operation_count(ops)?;
    let mut chain = ChainState::new(ops, cap, dpr, ctx, directives, share);
    for (op_index, op) in ops.iter().enumerate() {
        img = chain.step(img, op_index, op)?;
    }
    chain.finish(img)
}

/// The running state of one [`apply_operations`] pass: the
/// chain's inputs, the encode options the filters collect,
/// the focal and debug state a crop consumes, the inference
/// memos for the image state in hand, and the records the
/// meta response carries.
struct ChainState<'r, 'a> {
    /// The `no_upscale` cap: the (oriented) source
    /// dimensions, when the request asked for it.
    cap: Option<(u32, u32)>,
    /// The effective DPR factor (see `dpr_factor`).
    dpr: f32,
    ctx: &'r ProcessContext<'a>,
    directives: &'r crate::url::RequestDirectives,
    /// The geometry an animated render shares across its
    /// frames (issues #376, #465); `None` on a still.
    share: Option<&'r mut FrameShare>,
    encode_opts: EncodeOptions,
    face_memo: FaceMemo,
    /// The detection providers in resolution order, with
    /// one detection memo per provider (issue #166).
    detect_providers: Vec<&'a dyn crate::ai::DetectionProvider>,
    detection_memos: Vec<DetectionMemo>,
    /// The pending focal form from a `focal:` or
    /// `focalpoint:` segment (issue #313), consumed by the
    /// next crop. One slot holds both forms, so the last one
    /// in the chain is the one the crop obeys.
    focal: Option<PendingFocal>,
    /// The `debug:focal` state: armed by the segment, captured
    /// by the first crop that follows it. The capture holds
    /// the values that crop actually used (threaded out of the
    /// crop functions, never recomputed), and the response is
    /// the rendered capture instead of the transform result.
    debug_armed: bool,
    debug_capture: Option<crate::debugview::FocalCapture>,
    /// The studio record (issue #197): the last `studio(...)`
    /// filter in the chain wins, since its composite is the
    /// look the response carries.
    studio_record: Option<crate::studio::StudioRecord>,
    /// The region-removal record (issue #176): every
    /// `remove_region(...)` in the chain, in order.
    inpaint_regions: Vec<(u32, u32, u32, u32)>,
    /// The segmentation state (issue #175): the encoder
    /// memo (one encoding per image state, however many
    /// class operations decode against it) and the record
    /// of what was segmented, in chain order.
    segment_memo: SegmentMemo,
    segment_objects: Vec<SegmentObject>,
    /// The objects-crop record (issue #318): the last
    /// `objects` crop in the chain wins, like the studio
    /// record, since its framing is the one the response
    /// carries.
    objects_crop_record: Option<ObjectsCropRecord>,
    /// The crop+resize fusion plan (issue #287): when the
    /// chain qualifies, the crop-rect arm records its window
    /// instead of materializing it, and the resize arm
    /// resamples the window straight from the frame in hand.
    fusion: Option<FusionPlan>,
    fused_window: Option<(u32, u32, u32, u32)>,
}

/// A pending focal form (issues #313, #465): the rectangle
/// a `focal:` segment named or the point a `focalpoint:`
/// segment named, carried in the coordinates of the image in
/// hand. The segment names it against the image at its own
/// position in the chain; each later step that moves pixels
/// maps it along (a resize or fit-in scales it, a resize flip
/// mirrors it, a crop-rect or trim cuts it, a pad places
/// it), and a filter that moves pixels any other way drops
/// it, so the consuming crop never reads coordinates of an
/// image that is gone.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingFocal {
    /// `None` for a rectangle, the zoom for a point.
    zoom: Option<f32>,
    /// (left, top, right, bottom) in the coordinates of a
    /// `basis.0` x `basis.1` image. A point is a rectangle
    /// of no size.
    edges: [f64; 4],
    basis: (u32, u32),
}

/// A pending focal form resolved against the image a crop
/// cuts: a rectangle in its pixels, or a point as fractions
/// of it with the zoom.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Focal {
    Rect((u32, u32, u32, u32)),
    Point((f32, f32, f32)),
}

impl PendingFocal {
    fn rect((left, top, right, bottom): (u32, u32, u32, u32), basis: (u32, u32)) -> Self {
        PendingFocal {
            zoom: None,
            edges: [left as f64, top as f64, right as f64, bottom as f64],
            basis,
        }
    }

    fn point((x, y, zoom): (f32, f32, f32), basis: (u32, u32)) -> Self {
        let (px, py) = (x as f64 * basis.0 as f64, y as f64 * basis.1 as f64);
        PendingFocal {
            zoom: Some(zoom),
            edges: [px, py, px, py],
            basis,
        }
    }

    /// Through a resize of the whole image to `to`, mirrored
    /// afterwards on the flipped axes.
    fn scaled(self, to: (u32, u32), flip_h: bool, flip_v: bool) -> Self {
        let ratio = |to: u32, from: u32| {
            if from == 0 {
                0.0
            } else {
                to as f64 / from as f64
            }
        };
        let (sx, sy) = (ratio(to.0, self.basis.0), ratio(to.1, self.basis.1));
        let [mut left, mut top, mut right, mut bottom] = self.edges;
        (left, right) = (left * sx, right * sx);
        (top, bottom) = (top * sy, bottom * sy);
        if flip_h {
            (left, right) = (to.0 as f64 - right, to.0 as f64 - left);
        }
        if flip_v {
            (top, bottom) = (to.1 as f64 - bottom, to.1 as f64 - top);
        }
        PendingFocal {
            zoom: self.zoom,
            edges: [left, top, right, bottom],
            basis: to,
        }
    }

    /// Through a cut to `window`.
    /// Whatever falls outside the window lands on its nearest
    /// edge, so a rectangle the cut removed entirely still
    /// pulls the crop toward the side it was on.
    fn cropped(self, (x, y, w, h): Window) -> Self {
        let [left, top, right, bottom] = self.edges;
        let along =
            |v: f64, origin: u32, extent: u32| (v - origin as f64).clamp(0.0, extent as f64);
        PendingFocal {
            zoom: self.zoom,
            edges: [
                along(left, x, w),
                along(top, y, h),
                along(right, x, w),
                along(bottom, y, h),
            ],
            basis: (w, h),
        }
    }

    /// Through a fit onto a larger canvas: the image scaled
    /// to `placed`'s size and set at its offset on a
    /// `canvas`-sized image.
    fn placed(self, (x, y, w, h): Window, canvas: (u32, u32)) -> Self {
        let fitted = self.scaled((w, h), false, false);
        let [left, top, right, bottom] = fitted.edges;
        let (dx, dy) = (x as f64, y as f64);
        PendingFocal {
            zoom: self.zoom,
            edges: [left + dx, top + dy, right + dx, bottom + dy],
            basis: canvas,
        }
    }

    /// Resolve against the `dims` image the crop cuts. The
    /// steps keep the basis in step with the image; a size
    /// change no step mapped scales over.
    fn resolve(self, dims: (u32, u32)) -> Focal {
        let at = if self.basis == dims {
            self
        } else {
            self.scaled(dims, false, false)
        };
        let [left, top, right, bottom] = at.edges;
        match at.zoom {
            None => {
                let px = |v: f64, extent: u32| v.round().clamp(0.0, extent as f64) as u32;
                let (left, top) = (px(left, dims.0), px(top, dims.1));
                let (right, bottom) = (px(right, dims.0).max(left), px(bottom, dims.1).max(top));
                Focal::Rect((left, top, right, bottom))
            }
            Some(zoom) => {
                let frac = |v: f64, extent: u32| {
                    if extent == 0 {
                        0.5
                    } else {
                        (v / extent as f64).clamp(0.0, 1.0) as f32
                    }
                };
                Focal::Point((frac(left, dims.0), frac(top, dims.1), zoom))
            }
        }
    }
}

/// What a class filter paints (see
/// [`ChainState::class_targets`]): the segmentation masks
/// when a segment provider is configured, the detection
/// boxes when not.
enum ClassTargets {
    Masked(Vec<crate::filters::MaskedDetection>),
    Boxes(Vec<crate::ai::Detection>),
}

impl<'r, 'a> ChainState<'r, 'a> {
    fn new(
        ops: &[Operation],
        cap: Option<(u32, u32)>,
        dpr: f32,
        ctx: &'r ProcessContext<'a>,
        directives: &'r crate::url::RequestDirectives,
        share: Option<&'r mut FrameShare>,
    ) -> Self {
        let detect_providers = detection_providers(ctx);
        let detection_memos = vec![None; detect_providers.len()];
        ChainState {
            cap,
            dpr,
            ctx,
            directives,
            share,
            encode_opts: EncodeOptions::default(),
            face_memo: None,
            detect_providers,
            detection_memos,
            focal: None,
            debug_armed: false,
            debug_capture: None,
            studio_record: None,
            inpaint_regions: Vec::new(),
            segment_memo: None,
            segment_objects: Vec::new(),
            objects_crop_record: None,
            fusion: fusion_plan(ops),
            fused_window: None,
        }
    }

    /// Run one operation of the chain, with what every
    /// operation owes the chain around it: the deadline, the
    /// detection memos, and the output budget.
    fn step(
        &mut self,
        mut img: DynamicImage,
        op_index: usize,
        op: &Operation,
    ) -> Result<DynamicImage, EngineError> {
        self.ctx.check_deadline()?;
        img = self.apply_op(img, op_index, op)?;
        // Every operation other than the filters segment
        // sizes, crops, or pads, so the detection memos never
        // survive one (issue #373); the filters arm
        // invalidates per filter instead, and the no-op
        // arms (frame, page, debug) simply pay a fresh
        // inference in a chain that mixes them with
        // detection, which no real chain does.
        if !matches!(op, Operation::Filters(_)) {
            invalidate_detection_memos(
                &mut self.face_memo,
                &mut self.detection_memos,
                &mut self.segment_memo,
            );
        }
        // The running image stays inside the output budget
        // after every operation, not only after the ones
        // that size it (issue #182, audit R2-12). While a
        // fused window is pending (issue #287) the image in
        // hand is still the pre-crop frame: the two-step
        // path checks the window at this point, and the
        // fused crop ran that same check on the window when
        // it recorded it.
        if self.fused_window.is_none() {
            check_output_pixels(img.width(), img.height())?;
        }
        Ok(img)
    }

    /// Run the chain on `img` up to the filter at `key` (the
    /// operation and filter indexes) and return the image
    /// that filter receives: the trim survey's view of a
    /// frame (see [`survey_trims`]).
    fn run_to(
        &mut self,
        mut img: DynamicImage,
        ops: &[Operation],
        (op_index, filter_index): (usize, usize),
    ) -> Result<DynamicImage, EngineError> {
        for (index, op) in ops.iter().enumerate().take(op_index) {
            img = self.step(img, index, op)?;
        }
        if let Some(Operation::Filters(filters)) = ops.get(op_index) {
            for (index, filter) in filters.iter().enumerate().take(filter_index) {
                img = self.filter_step(img, (op_index, index), filter)?;
            }
        }
        Ok(img)
    }

    /// Apply one operation of the chain to the image in hand.
    fn apply_op(
        &mut self,
        mut img: DynamicImage,
        op_index: usize,
        op: &Operation,
    ) -> Result<DynamicImage, EngineError> {
        match op {
            Operation::DebugFocal => self.debug_armed = true,
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
                img = if self.fusion.is_some_and(|plan| plan.size_index == op_index) {
                    let window = self
                        .fused_window
                        .take()
                        .expect("the fusion plan pairs this resize with its crop");
                    apply_size_cropped(
                        img,
                        window,
                        size,
                        self.cap.map(|(cw, ch)| SizeCap::Axes(cw, ch)),
                        self.dpr,
                    )?
                } else {
                    apply_size(
                        img,
                        size,
                        self.cap.map(|(cw, ch)| SizeCap::Axes(cw, ch)),
                        self.dpr,
                    )?
                };
                let dims = (img.width(), img.height());
                self.focal = self
                    .focal
                    .map(|f| f.scaled(dims, size.flip_horizontal, size.flip_vertical));
            }
            Operation::FitIn { size, adaptive } => {
                img = apply_fit_in(img, size, *adaptive, self.cap, self.dpr)?;
                let dims = (img.width(), img.height());
                self.focal = self.focal.map(|f| f.scaled(dims, false, false));
            }
            Operation::Pad { size, fill } => {
                let (tw, th) = match (size.width, size.height) {
                    (Dimension::Pixels(w), Dimension::Pixels(h)) => {
                        (scale_px(w, self.dpr), scale_px(h, self.dpr))
                    }
                    _ => {
                        return Err(EngineError::Unsupported(
                            "pad with non-pixel dimensions".to_string(),
                        ));
                    }
                };
                let (padded, placed) =
                    apply_pad(img, tw, th, *fill, self.directives.fill, self.cap)?;
                img = padded;
                let dims = (img.width(), img.height());
                self.focal = self.focal.map(|f| f.placed(placed, dims));
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
                self.focal = Some(PendingFocal::rect(
                    (*left, *top, *right, *bottom),
                    (img.width(), img.height()),
                ));
            }
            Operation::FocalPoint { x, y, zoom } => {
                // Fractions of the image in effect at this
                // point, so there is no bounds failure to
                // make: any image can carry the point.
                self.focal = Some(PendingFocal::point(
                    (*x, *y, *zoom),
                    (img.width(), img.height()),
                ));
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
                if self.fusion.is_some_and(|plan| plan.crop_index == op_index) {
                    // Issue #287: the bounds validation above
                    // already ran, so an out-of-bounds
                    // rectangle errors exactly as the two-step
                    // path does. The window itself is cut by
                    // the resize, not here; the budget check
                    // the two-step path runs on the cropped
                    // result at the end of this step runs on
                    // the window now, so an oversized window
                    // refuses the same way.
                    check_output_pixels(right - left, bottom - top)?;
                    self.fused_window = Some((*left, *top, right - left, bottom - top));
                } else {
                    img = img.crop_imm(*left, *top, right - left, bottom - top);
                }
                let window = (*left, *top, right - left, bottom - top);
                self.focal = self.focal.map(|f| f.cropped(window));
            }
            Operation::Crop { size, align } => img = self.crop(img, op_index, size, align)?,
            Operation::Filters(filters) => {
                for (filter_index, filter) in filters.iter().enumerate() {
                    img = self.filter_step(img, (op_index, filter_index), filter)?;
                }
            }
        }
        Ok(img)
    }

    /// One filter of a `filters:` segment, at `key` (the
    /// operation and filter indexes), with what every
    /// filter owes the chain around it: the deadline, the
    /// pending focal form, the detection memos, and the
    /// output budget. Trims run here, since they need their
    /// chain position to find the window an animated render
    /// shares (issue #465); every other filter goes through
    /// [`ChainState::apply_filter`].
    fn filter_step(
        &mut self,
        img: DynamicImage,
        key: (usize, usize),
        filter: &Filter,
    ) -> Result<DynamicImage, EngineError> {
        // The deadline is checked between filters, not only
        // between operations: a segment is one operation to
        // the per-operation check, so without this a long
        // segment (a hundred blurs in one `filters:`) runs
        // far past the deadline the runner already reported
        // to the client.
        self.ctx.check_deadline()?;
        let before = (img.width(), img.height());
        let img = match filter {
            Filter::Trim(tolerance) => self.trim(img, key, *tolerance, [0; 4]),
            Filter::TrimSides(tolerance, sides) => self.trim(img, key, *tolerance, *sides),
            _ => {
                let img = self.apply_filter(img, filter)?;
                // A filter that moves pixels in a way the
                // chain does not track (a rotation, a flip,
                // an upscale, studio) leaves the pending
                // focal form pointing at content that is no
                // longer there: drop it, and the crop falls
                // back to its alignment (issue #465).
                if !(filter_keeps_focal(filter) && (img.width(), img.height()) == before) {
                    self.focal = None;
                }
                img
            }
        };
        // A filter that moves content leaves the detection
        // memos describing a state the pixels have left
        // (issue #373): the next detection consumer must
        // re-detect.
        if !filter_preserves_positions(filter) {
            invalidate_detection_memos(
                &mut self.face_memo,
                &mut self.detection_memos,
                &mut self.segment_memo,
            );
        }
        // Every filter leaves the image inside the output
        // budget: growth is checked before it compounds
        // across a chain, not just at the size operations.
        check_output_pixels(img.width(), img.height())?;
        Ok(img)
    }

    /// A `trim` filter: the window the frame's own tolerance
    /// pass keeps, or on an animated render the window every
    /// frame shares (issue #465), less the named sides. A
    /// pending focal form moves with the cut.
    fn trim(
        &mut self,
        img: DynamicImage,
        key: (usize, usize),
        tolerance: u8,
        sides: [u32; 4],
    ) -> DynamicImage {
        let dims = (img.width(), img.height());
        let found = match self.share.as_deref().and_then(|s| s.trims.get(key, dims)) {
            Some(shared) => shared,
            None => crate::filters::trim_box(&img, tolerance),
        };
        let window = crate::filters::trim_window(dims, found, sides);
        self.focal = self.focal.map(|f| f.cropped(window));
        crate::filters::trim_to(img, found, sides)
    }

    /// The `crop:` operation: crop to the target's aspect
    /// ratio, positioned by a pending focal rectangle or
    /// point, or else by the alignment, then resize to
    /// exactly the target.
    fn crop(
        &mut self,
        img: DynamicImage,
        op_index: usize,
        size: &ImageSize,
        align: &Align,
    ) -> Result<DynamicImage, EngineError> {
        let (tw, th) = match (size.width, size.height) {
            (Dimension::Pixels(w), Dimension::Pixels(h)) => {
                (scale_px(w, self.dpr), scale_px(h, self.dpr))
            }
            _ => {
                return Err(EngineError::Unsupported(
                    "crop with non-pixel dimensions".to_string(),
                ));
            }
        };
        // The resize that closes the crop lands on this size
        // (the same `no_upscale` fit as below): refuse one past
        // the limit here, before any detection or saliency
        // pass runs on a crop that cannot be served (issue
        // #521). A target that keeps the image in hand as it
        // is passes, like any step that keeps its size.
        let target = match self.cap {
            Some(cap) => cap_dimensions(tw, th, cap),
            None => (tw, th),
        };
        check_output_size(target.0, target.1, (img.width(), img.height()))?;
        // The crop consumes any pending focal form, mapped
        // onto the image in hand by the steps since its
        // segment (issue #465). Center and smart crops
        // position on it (smart skips face and saliency
        // detection entirely); face and directional
        // alignments were explicit caller instructions and
        // ignore it.
        let focal = self
            .focal
            .take()
            .filter(|_| *align == Align::Center || *align == Align::Smart)
            .map(|f| f.resolve((img.width(), img.height())));
        // With the debug view armed, this crop is the
        // one the view annotates: clone the canvas it
        // operates on and run the traced crop variants,
        // which report the decision they actually made.
        let want_trace = self.debug_armed && self.debug_capture.is_none();
        let canvas = want_trace.then(|| img.clone());
        let img = if let Some(Focal::Rect(rect)) = focal {
            self.focal_rect_crop(img, (tw, th), rect, canvas)
        } else if let Some(Focal::Point(pt)) = focal {
            self.focal_point_crop(img, (tw, th), pt, canvas)
        } else if *align == Align::Face {
            self.face_crop(img, (tw, th), canvas)?
        } else if let Align::Objects { classes, scale } = align {
            self.objects_crop(img, (tw, th), classes, *scale, canvas)?
        } else if let Some(canvas) = canvas {
            let (cropped, trace) = crate::crop::crop_to_aspect_traced(
                img,
                tw,
                th,
                align.clone(),
                self.ctx.face_provider,
                self.ctx.focal_provider,
            );
            if let Some(trace) = trace {
                self.debug_capture = Some(capture_from_trace(canvas, trace));
            }
            cropped
        } else if *align == Align::Smart
            && self
                .share
                .as_ref()
                .is_some_and(|s| s.smart_windows.is_some())
        {
            self.shared_smart_crop(img, (tw, th), op_index, align)
        } else {
            crate::crop::crop_to_aspect_with(
                img,
                tw,
                th,
                align.clone(),
                self.ctx.face_provider,
                self.ctx.focal_provider,
            )
        };
        apply_size(
            img,
            &ImageSize {
                width: Dimension::Pixels(tw),
                height: Dimension::Pixels(th),
                flip_horizontal: false,
                flip_vertical: false,
            },
            self.cap.map(|(cw, ch)| SizeCap::Fit(cw, ch)),
            1.0,
        )
    }

    /// A center or smart crop positioned on a pending focal
    /// rectangle. `canvas` is set when the debug view
    /// annotates this crop.
    fn focal_rect_crop(
        &mut self,
        img: DynamicImage,
        (tw, th): (u32, u32),
        rect: (u32, u32, u32, u32),
        canvas: Option<DynamicImage>,
    ) -> DynamicImage {
        if let Some(canvas) = canvas {
            let (cropped, window) = crate::crop::crop_to_aspect_focal_traced(img, tw, th, rect);
            let (cw, ch) = (canvas.width(), canvas.height());
            self.debug_capture = Some(crate::debugview::FocalCapture {
                canvas,
                focal: Some((
                    ((rect.0 + rect.2) / 2).min(cw.saturating_sub(1)),
                    ((rect.1 + rect.3) / 2).min(ch.saturating_sub(1)),
                )),
                focal_rect: Some(rect),
                window,
                faces: Vec::new(),
                objects: Vec::new(),
            });
            cropped
        } else {
            crate::crop::crop_to_aspect_focal(img, tw, th, rect)
        }
    }

    /// A center or smart crop centered on a pending focal
    /// point (issue #313). `canvas` is set when the debug
    /// view annotates this crop.
    fn focal_point_crop(
        &mut self,
        img: DynamicImage,
        (tw, th): (u32, u32),
        pt: (f32, f32, f32),
        canvas: Option<DynamicImage>,
    ) -> DynamicImage {
        if let Some(canvas) = canvas {
            let (cropped, window) = crate::crop::crop_to_aspect_point_traced(img, tw, th, pt);
            let (cw, ch) = (canvas.width(), canvas.height());
            self.debug_capture = Some(crate::debugview::FocalCapture {
                canvas,
                focal: Some((
                    ((pt.0 * cw as f32).round() as u32).min(cw.saturating_sub(1)),
                    ((pt.1 * ch as f32).round() as u32).min(ch.saturating_sub(1)),
                )),
                focal_rect: None,
                window,
                faces: Vec::new(),
                objects: Vec::new(),
            });
            cropped
        } else {
            crate::crop::crop_to_aspect_point(img, tw, th, pt)
        }
    }

    /// Crop around the largest detected face. A detection
    /// failure or an empty result falls back to the saliency
    /// smart crop: a crop always exists, so it never fails
    /// the request. `canvas` is set when the debug view
    /// annotates this crop.
    fn face_crop(
        &mut self,
        img: DynamicImage,
        (tw, th): (u32, u32),
        canvas: Option<DynamicImage>,
    ) -> Result<DynamicImage, EngineError> {
        let cropped = match detect_faces(&mut self.face_memo, &img, self.ctx) {
            Ok(faces) if !faces.is_empty() => {
                let face = *faces
                    .iter()
                    .max_by_key(|f| f.w as u64 * f.h as u64)
                    .expect("faces is non-empty");
                if let Some(canvas) = canvas {
                    let (cropped, window) = crate::crop::crop_around_face_traced(
                        img,
                        tw,
                        th,
                        &face,
                        self.directives.face_padding,
                    );
                    self.debug_capture = Some(crate::debugview::FocalCapture {
                        canvas,
                        focal: Some((face.x + face.w / 2, face.y + face.h / 2)),
                        focal_rect: Some((face.x, face.y, face.x + face.w, face.y + face.h)),
                        window,
                        faces,
                        objects: Vec::new(),
                    });
                    cropped
                } else {
                    crate::crop::crop_around_face(img, tw, th, &face, self.directives.face_padding)
                }
            }
            Ok(_) | Err(EngineError::Inference(_)) => {
                if let Some(canvas) = canvas {
                    let (cropped, trace) = crate::crop::crop_to_aspect_traced(
                        img,
                        tw,
                        th,
                        Align::Smart,
                        None,
                        self.ctx.focal_provider,
                    );
                    if let Some(trace) = trace {
                        self.debug_capture = Some(capture_from_trace(canvas, trace));
                    }
                    cropped
                } else {
                    crate::crop::crop_to_aspect_with(
                        img,
                        tw,
                        th,
                        Align::Smart,
                        None,
                        self.ctx.focal_provider,
                    )
                }
            }
            Err(e) => return Err(e),
        };
        Ok(cropped)
    }

    /// Crop to the union of the detected objects of the named
    /// classes (issue #318). Preflight already refused the
    /// request when no detection provider is configured or a
    /// named class is unknown, so what remains is the model's
    /// answer: a union to frame, or none. None (and a
    /// detection failure, downgraded exactly like the face
    /// crop's) falls back to the plain crop of the same size;
    /// the record states which path was taken either way.
    /// `canvas` is set when the debug view annotates this
    /// crop.
    fn objects_crop(
        &mut self,
        img: DynamicImage,
        (tw, th): (u32, u32),
        classes: &[String],
        scale: f32,
        canvas: Option<DynamicImage>,
    ) -> Result<DynamicImage, EngineError> {
        let mut used: Vec<crate::ai::Detection> = Vec::new();
        let mut answered = true;
        for class in classes {
            match detect_for_class(
                &mut self.detection_memos,
                &self.detect_providers,
                &img,
                class,
            ) {
                Ok(found) => used.extend(found),
                Err(EngineError::Inference(_)) => {
                    answered = false;
                    break;
                }
                Err(e) => return Err(e),
            }
        }
        let union = if answered {
            crate::crop::union_box(&used)
        } else {
            None
        };
        let cropped = match union {
            Some(union) => {
                self.objects_crop_record = Some(ObjectsCropRecord {
                    classes: classes.to_vec(),
                    objects: used.clone(),
                    union: Some((union.0, union.1, union.2 - union.0, union.3 - union.1)),
                    scale,
                    fallback: false,
                });
                if let Some(canvas) = canvas {
                    let (cropped, window) =
                        crate::crop::crop_around_objects_traced(img, tw, th, union, scale);
                    self.debug_capture = Some(crate::debugview::FocalCapture {
                        canvas,
                        focal: Some(((union.0 + union.2) / 2, (union.1 + union.3) / 2)),
                        focal_rect: Some(union),
                        window,
                        faces: Vec::new(),
                        objects: used,
                    });
                    cropped
                } else {
                    crate::crop::crop_around_objects(img, tw, th, union, scale)
                }
            }
            None => {
                self.objects_crop_record = Some(ObjectsCropRecord {
                    classes: classes.to_vec(),
                    objects: Vec::new(),
                    union: None,
                    scale,
                    fallback: true,
                });
                if let Some(canvas) = canvas {
                    let (cropped, trace) =
                        crate::crop::crop_to_aspect_traced(img, tw, th, Align::Center, None, None);
                    if let Some(trace) = trace {
                        self.debug_capture = Some(capture_from_trace(canvas, trace));
                    }
                    cropped
                } else {
                    crate::crop::crop_to_aspect_with(img, tw, th, Align::Center, None, None)
                }
            }
        };
        Ok(cropped)
    }

    /// A smart crop in an animated render sharing windows
    /// across frames (issue #376): the first frame decides
    /// through the traced crop (the trace reports the very
    /// window the crop used) and later frames replay that
    /// window verbatim, so the analysis runs once and the
    /// framing cannot jitter.
    fn shared_smart_crop(
        &mut self,
        img: DynamicImage,
        (tw, th): (u32, u32),
        op_index: usize,
        align: &Align,
    ) -> DynamicImage {
        let windows = self
            .share
            .as_deref_mut()
            .and_then(|s| s.smart_windows.as_mut())
            .expect("the crop checked the option");
        let key = (op_index, img.width(), img.height());
        if let Some((x, y, cw, ch)) = windows.get(key) {
            DynamicImage::ImageRgba8(image::imageops::crop_imm(&img, x, y, cw, ch).to_image())
        } else {
            let (cropped, trace) = crate::crop::crop_to_aspect_traced(
                img,
                tw,
                th,
                align.clone(),
                self.ctx.face_provider,
                self.ctx.focal_provider,
            );
            if let Some(trace) = trace {
                windows.set(key, trace.window);
            }
            cropped
        }
    }

    /// Apply one filter of a `filters:` segment: watermark
    /// and overlay filters resolve against the
    /// server-fetched images, the model and font filters run
    /// through their providers, and everything else is a
    /// plain filter (see [`ChainState::plain_filter`]).
    fn apply_filter(
        &mut self,
        mut img: DynamicImage,
        filter: &Filter,
    ) -> Result<DynamicImage, EngineError> {
        match filter {
            Filter::Watermark { url, x, y, alpha } => {
                let wm = self
                    .ctx
                    .watermarks
                    .and_then(|m| m.get(url))
                    .ok_or_else(|| {
                        EngineError::Unsupported(format!("watermark was not resolved: {url}"))
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
                let ov = self
                    .ctx
                    .watermarks
                    .and_then(|m| m.get(url))
                    .ok_or_else(|| {
                        EngineError::Unsupported(format!("overlay was not resolved: {url}"))
                    })?;
                img = blend_overlay(self.ctx, &img, ov, *x, *y, *alpha, *blend, *width)?;
            }
            Filter::RemoveBg => {
                let provider = self.ctx.mask_provider.ok_or_else(|| {
                    EngineError::NotConfigured(
                        "removebg is not configured on this server".to_string(),
                    )
                })?;
                let mask = provider.mask(&img).map_err(EngineError::Inference)?;
                img = crate::ai::apply_mask(&img, &mask);
            }
            Filter::Studio(spec) => img = self.studio(&img, spec)?,
            Filter::BlurFaces => {
                // A detection failure fails the
                // request: serving the image
                // unredacted would be the real
                // failure.
                let faces = detect_faces(&mut self.face_memo, &img, self.ctx)?;
                img = crate::filters::blur_faces(&img, &faces);
            }
            Filter::PixelateFaces => {
                let faces = detect_faces(&mut self.face_memo, &img, self.ctx)?;
                img = crate::filters::pixelate_faces(&img, &faces);
            }
            Filter::BlurClass(class) => {
                img = match self.class_targets(&img, class)? {
                    ClassTargets::Masked(masked) => {
                        crate::filters::blur_detections_masked(&img, &masked)
                    }
                    ClassTargets::Boxes(boxes) => crate::filters::blur_detections(&img, &boxes),
                };
            }
            Filter::PixelateClass(class) => {
                img = match self.class_targets(&img, class)? {
                    ClassTargets::Masked(masked) => {
                        crate::filters::pixelate_detections_masked(&img, &masked)
                    }
                    ClassTargets::Boxes(boxes) => crate::filters::pixelate_detections(&img, &boxes),
                };
            }
            Filter::RecolorClass(class, color) => {
                // The class rules are the
                // redaction filters': detection
                // resolves the class, and a
                // failure fails the request.
                // The paint follows the
                // silhouette when a segment
                // provider is configured, the
                // box when not (issue #175).
                img = match self.class_targets(&img, class)? {
                    ClassTargets::Masked(masked) => {
                        crate::filters::recolor_detections_masked(&img, &masked, *color)
                    }
                    ClassTargets::Boxes(boxes) => {
                        crate::filters::recolor_detections(&img, &boxes, *color)
                    }
                };
            }
            // Padding is read from the request as a
            // whole by the face crop; no pixels here.
            Filter::FacePad(_) => {}
            Filter::Upscale(factor) => {
                let provider = self.ctx.upscale_provider.ok_or_else(|| {
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
            Filter::RemoveRegion { x, y, w, h } => {
                // Same bounds discipline as the
                // other region filters: a rectangle
                // that overflows the image in
                // effect here fails the request
                // rather than erasing less than
                // the caller named (issue #176).
                check_region("remove_region", *x, *y, *w, *h, &img)?;
                let provider = self.ctx.inpaint_provider.ok_or_else(|| {
                    EngineError::NotConfigured(
                        "remove_region is not configured on this server".to_string(),
                    )
                })?;
                let mask = crate::ai::region_mask(img.width(), img.height(), *x, *y, *w, *h);
                img = provider
                    .inpaint(&img, &mask)
                    .map_err(EngineError::Inference)?;
                self.inpaint_regions.push((*x, *y, *w, *h));
            }
            Filter::Text {
                text,
                size,
                color,
                anchor,
            } => {
                let renderer = self.ctx.text_renderer.ok_or_else(|| {
                    EngineError::NotConfigured(
                        "text is not configured on this server (no font)".to_string(),
                    )
                })?;
                img = renderer
                    .draw(&img, text, *size, *color, *anchor)
                    .map_err(EngineError::Render)?;
            }
            other => img = self.plain_filter(img, other)?,
        }
        Ok(img)
    }

    /// Studio runs the removebg mask model, then expands in
    /// `crate::studio`: product mode composites (mask,
    /// defringe, backdrop and shadow, pad; issue #197), scene
    /// mode reframes and corrects (mask, crop, correct; issue
    /// #198). The record rides out with the result for meta.
    fn studio(
        &mut self,
        img: &DynamicImage,
        spec: &crate::url::StudioSpec,
    ) -> Result<DynamicImage, EngineError> {
        let provider = self.ctx.mask_provider.ok_or_else(|| {
            EngineError::NotConfigured(
                "studio is not configured on this server (no mask model)".to_string(),
            )
        })?;
        let t_timing = std::time::Instant::now();
        let mask = provider.mask(img).map_err(EngineError::Inference)?;
        let confidence = provider.confidence(img, &mask);
        crate::studio::timing_mark("inference", t_timing);
        let outcome = match spec.mode {
            crate::url::StudioMode::Product => {
                crate::studio::composite(img, &mask, confidence, spec, self.ctx.brand_color)
            }
            crate::url::StudioMode::Scene => crate::studio::scene(img, &mask, confidence, spec),
        };
        self.studio_record = Some(outcome.record);
        Ok(outcome.image)
    }

    /// What a class filter paints: the detections of `class`
    /// in `img`, segmented into masks when a segment provider
    /// is configured (issue #175: mask-tight, the object's
    /// silhouette rather than its box), with the segmented
    /// objects added to the record. The #42 rule, inherited:
    /// a detection or segmentation failure fails the request
    /// rather than serve the image unredacted.
    fn class_targets(
        &mut self,
        img: &DynamicImage,
        class: &str,
    ) -> Result<ClassTargets, EngineError> {
        let boxes = detect_for_class(
            &mut self.detection_memos,
            &self.detect_providers,
            img,
            class,
        )?;
        Ok(match self.ctx.segment_provider {
            Some(provider) => {
                let (masked, mut objects) =
                    segment_for_class(&mut self.segment_memo, provider, img, &boxes)?;
                self.segment_objects.append(&mut objects);
                ClassTargets::Masked(masked)
            }
            None => ClassTargets::Boxes(boxes),
        })
    }

    /// A plain filter: pixels in, pixels out through
    /// `apply_filters_filled`, with any encoding directive it
    /// carries absorbed into the chain's encode options.
    fn plain_filter(
        &mut self,
        img: DynamicImage,
        filter: &Filter,
    ) -> Result<DynamicImage, EngineError> {
        // Arbitrary-angle rotation is
        // the one plain filter that grows
        // the canvas: check the canvas it
        // is about to allocate before it
        // allocates (issue #182, R2-12),
        // per axis too (issue #521).
        if let Filter::Rotate(degrees) = filter {
            let (cw, ch) = crate::filters::rotated_canvas(img.width(), img.height(), *degrees);
            check_output_size(cw, ch, (img.width(), img.height()))?;
        }
        let (new_img, options) = crate::filters::apply_filters_filled(
            img,
            std::slice::from_ref(filter),
            self.directives.fill,
        );
        self.encode_opts.absorb(options);
        Ok(new_img)
    }

    /// Close the chain: render the debug view when
    /// `debug:focal` armed it, check the final image against
    /// the output budget, and hand back the image with the
    /// encode options and the records.
    fn finish(self, mut img: DynamicImage) -> Result<ApplyOutcome, EngineError> {
        if self.debug_armed {
            // The debug view replaces the transform result. With
            // no capture, no crop followed the segment: the view
            // shows the focal analysis of the final image. With
            // one, it shows the decision that crop actually made,
            // on the canvas it operated on.
            let capture = match self.debug_capture {
                Some(capture) => capture,
                None => {
                    let (point, faces) = crate::focus::focal_point_chained(
                        &img,
                        self.ctx.face_provider,
                        self.ctx.focal_provider,
                    );
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
                        objects: Vec::new(),
                    }
                }
            };
            img = crate::debugview::render(&capture);
        }
        // The last word on size: whatever the chain did, the
        // image that goes to the encoder is inside the output
        // budget (issue #182, audit R2-12).
        check_output_pixels(img.width(), img.height())?;
        let inpaint_record = if self.inpaint_regions.is_empty() {
            None
        } else {
            Some(InpaintRecord {
                regions: self.inpaint_regions,
            })
        };
        let segment_record = if self.segment_objects.is_empty() {
            None
        } else {
            Some(SegmentRecord {
                objects: self.segment_objects,
            })
        };
        Ok((
            img,
            self.encode_opts,
            self.studio_record,
            inpaint_record,
            segment_record,
            self.objects_crop_record,
        ))
    }
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
        objects: Vec::new(),
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
        2 => crate::filters::fliph_fast(&img),
        3 => crate::filters::rotate180_fast(&img),
        4 => crate::filters::flipv_fast(&img),
        // Transpose and anti-transpose: a rotation plus a mirror.
        5 => crate::filters::fliph_fast(&img.rotate90()),
        6 => img.rotate90(),
        7 => crate::filters::fliph_fast(&crate::filters::rotate270_fast(&img)),
        8 => crate::filters::rotate270_fast(&img),
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
pub(crate) fn svg_raster_target(ops: &[Operation], intrinsic: (f32, f32), dpr: f32) -> (u32, u32) {
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
                        Dimension::Pixels(h) => proportional_side(iw, scale_px(h, dpr), ih),
                        _ => iw,
                    },
                };
                let h = match size.height {
                    Dimension::Pixels(n) => scale_px(n, dpr),
                    Dimension::Original => ih,
                    Dimension::Proportional => match size.width {
                        Dimension::Pixels(w) => proportional_side(ih, scale_px(w, dpr), iw),
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

mod encode;
/// Names the test module uses from encode: re-exported
/// only for test builds, where they are used.
#[cfg(test)]
pub(in crate::engine) use encode::encode_png_indexed;
pub(in crate::engine) use encode::{
    EncodeBuffers, encode, gif_frame_quantized, max_quality_attempts, next_quality,
};

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
#[allow(clippy::too_many_arguments)]
fn blend_overlay(
    ctx: &ProcessContext<'_>,
    base: &DynamicImage,
    ov: &DynamicImage,
    x: i32,
    y: i32,
    alpha: u8,
    mode: crate::url::BlendMode,
    width_pct: Option<u32>,
) -> Result<DynamicImage, EngineError> {
    let scaled;
    let ov = match width_pct {
        Some(pct) => {
            let target = ((base.width() as u64 * pct as u64) / 100).max(1);
            // The scaled height follows the overlay's aspect
            // with no bound of its own, so an extreme aspect
            // (or a very wide base) grows the intermediate
            // far past the pixel budget that guards the
            // other resize paths, up to aborting the process
            // on the allocation (issue #358). Charge an upper
            // bound of the scaled pair against the shared
            // budget before the resize allocates it.
            let scaled_h = (ov.height() as u64 * target / ov.width().max(1) as u64) + 1;
            if target * scaled_h > MAX_OUTPUT_PIXELS {
                return Err(EngineError::Unsupported(format!(
                    "scaled overlay {target}x{scaled_h} exceeds the {MAX_OUTPUT_PIXELS} pixel limit"
                )));
            }
            scaled = ov.resize(
                target as u32,
                u32::MAX,
                image::imageops::FilterType::Lanczos3,
            );
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
        // The blend walks every overlay pixel; a large
        // legal pair still runs long, so the deadline is
        // honored per row (issue #358).
        ctx.check_deadline()?;
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
    Ok(DynamicImage::ImageRgba8(out))
}

/// Overlay `wm` onto `base` at (x, y) with `alpha` percent
/// opacity, honoring the watermark's own alpha channel.
/// Negative coordinates anchor from the right/bottom edges
/// (x = -1 puts the watermark's right edge one pixel from the
/// base's right edge). Parts falling outside the base are
/// clipped, never a panic.
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
pub(crate) fn cap_dimensions(w: u32, h: u32, (cw, ch): (u32, u32)) -> (u32, u32) {
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
    apply_size_windowed(img, size, cap, dpr, None)
}

/// [`apply_size`] for the crop+resize fusion (issue #287):
/// `img` is still the whole frame and `window` is the (left,
/// top, width, height) rectangle the chain's crop-rect named.
/// The size resolves against the window's dimensions, exactly
/// as it resolves against the materialized crop in the
/// two-step path, and the resample reads the window straight
/// from the frame.
fn apply_size_cropped(
    img: DynamicImage,
    window: (u32, u32, u32, u32),
    size: &ImageSize,
    cap: Option<SizeCap>,
    dpr: f32,
) -> Result<DynamicImage, EngineError> {
    apply_size_windowed(img, size, cap, dpr, Some(window))
}

/// Whether a resize or fit-in box asks for a zero pixel side
/// (issue #464). A single `0` is proportional (or unbounded),
/// but a zero written with more digits (`00x100`) parses as an
/// exact zero: the spelling is shared grammar that the spec
/// vectors pin, so the parser keeps it and the engine refuses
/// it. Read on the size as requested, before DPR scaling
/// rounds a zero up to one pixel. The `/explain` forecast asks
/// the same question, so the two cannot disagree.
pub(crate) fn has_zero_side(size: &ImageSize) -> bool {
    size.width == Dimension::Pixels(0) || size.height == Dimension::Pixels(0)
}

/// Refuse a [`has_zero_side`] box with `unsupported_operation`,
/// the answer an output past the pixel budget gets.
fn check_nonzero_size(size: &ImageSize) -> Result<(), EngineError> {
    if has_zero_side(size) {
        return Err(EngineError::Unsupported(
            "requested size has a zero pixel side; write 0 alone for a \
             proportional or unbounded axis"
                .to_string(),
        ));
    }
    Ok(())
}

/// `side * to / from`: the proportional side of a resize
/// following the other side from `from` to `to`, floored and
/// never below 1. A quotient past `u32` saturates rather than
/// wraps, so the per-axis check always sees it (issue #521).
fn proportional_side(side: u32, to: u32, from: u32) -> u32 {
    let q = (side as u64 * to as u64) / from.max(1) as u64;
    u32::try_from(q).unwrap_or(u32::MAX).max(1)
}

fn apply_size_windowed(
    img: DynamicImage,
    size: &ImageSize,
    cap: Option<SizeCap>,
    dpr: f32,
    window: Option<(u32, u32, u32, u32)>,
) -> Result<DynamicImage, EngineError> {
    check_nonzero_size(size)?;
    let size = &ImageSize {
        width: scale_dimension(size.width, dpr),
        height: scale_dimension(size.height, dpr),
        flip_horizontal: size.flip_horizontal,
        flip_vertical: size.flip_vertical,
    };
    let (sw, sh) = match window {
        Some((_, _, w, h)) => (w, h),
        None => (img.width(), img.height()),
    };
    let w = match size.width {
        Dimension::Pixels(n) => n,
        Dimension::Original => sw,
        Dimension::Proportional => match size.height {
            Dimension::Pixels(h) => proportional_side(sw, h, sh),
            _ => sw,
        },
    };
    let h = match size.height {
        Dimension::Pixels(n) => n,
        Dimension::Original => sh,
        Dimension::Proportional => match size.width {
            Dimension::Pixels(w) => proportional_side(sh, w, sw),
            _ => sh,
        },
    };
    let (w, h) = match cap {
        Some(SizeCap::Axes(cw, ch)) => (w.min(cw), h.min(ch)),
        Some(SizeCap::Fit(cw, ch)) => cap_dimensions(w, h, (cw, ch)),
        None => (w, h),
    };
    check_output_size(w, h, (sw, sh))?;
    let mut img = if w == sw && h == sh {
        match window {
            // No resample runs: the two-step path returns
            // the materialized window itself, so the fused
            // path materializes it here.
            Some((x, y, ww, wh)) => img.crop_imm(x, y, ww, wh),
            None => img,
        }
    } else {
        match window {
            Some((x, y, ww, wh)) => resize_lanczos3_cropped(&img, x, y, ww, wh, w, h),
            None => resize_lanczos3(&img, w, h),
        }
    };
    if size.flip_horizontal {
        img = crate::filters::fliph_fast(&img);
    }
    if size.flip_vertical {
        img = crate::filters::flipv_fast(&img);
    }
    Ok(img)
}

/// Resize to (w, h) with the SIMD backend (fast_image_resize),
/// Lanczos3. Same filter as the image crate's Lanczos3 up to
/// rounding (max diff 1 per channel on photo content), except
/// alpha: the backend resamples premultiplied alpha, so colors
/// stored under transparent pixels do not bleed into visible
/// ones. Pixel types the backend does not cover fall back to
/// the image crate.
pub(crate) fn resize_lanczos3(img: &DynamicImage, w: u32, h: u32) -> DynamicImage {
    let mut dst = match blank_like(img, w, h) {
        Some(dst) => dst,
        None => {
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

/// A blank destination image of `img`'s own variant at
/// (w, h): the resizers' shared starting point. `None` for
/// the variants outside the table, which the resizers hand
/// to the image crate instead.
fn blank_like(img: &DynamicImage, w: u32, h: u32) -> Option<DynamicImage> {
    match img {
        DynamicImage::ImageLuma8(_) => Some(DynamicImage::new_luma8(w, h)),
        DynamicImage::ImageLumaA8(_) => Some(DynamicImage::new_luma_a8(w, h)),
        DynamicImage::ImageRgb8(_) => Some(DynamicImage::new_rgb8(w, h)),
        DynamicImage::ImageRgba8(_) => Some(DynamicImage::new_rgba8(w, h)),
        DynamicImage::ImageLuma16(_) => Some(DynamicImage::new_luma16(w, h)),
        DynamicImage::ImageLumaA16(_) => Some(DynamicImage::new_luma_a16(w, h)),
        DynamicImage::ImageRgb16(_) => Some(DynamicImage::new_rgb16(w, h)),
        DynamicImage::ImageRgba16(_) => Some(DynamicImage::new_rgba16(w, h)),
        DynamicImage::ImageRgb32F(_) => Some(DynamicImage::new_rgb32f(w, h)),
        DynamicImage::ImageRgba32F(_) => Some(DynamicImage::new_rgba32f(w, h)),
        _ => None,
    }
}

/// [`resize_lanczos3`] fused with the crop that precedes it
/// (issue #287): the resample reads the (x, y, w, h) window
/// straight from `img` through the backend's crop box,
/// instead of resampling a materialized copy of the window.
/// The backend computes its coefficients against the
/// full-source geometry when a crop box is set, so the
/// result differs from the two-step chain by the
/// rounding-class deltas [`fusion_plan`] documents. A
/// backend failure, or a variant outside the table, falls
/// back to the exact two-step chain.
pub(crate) fn resize_lanczos3_cropped(
    img: &DynamicImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    target_w: u32,
    target_h: u32,
) -> DynamicImage {
    let two_step = || resize_lanczos3(&img.crop_imm(x, y, w, h), target_w, target_h);
    let mut dst = match blank_like(img, target_w, target_h) {
        Some(dst) => dst,
        None => return two_step(),
    };
    let opts = fast_image_resize::ResizeOptions::new()
        .crop(x as f64, y as f64, w as f64, h as f64)
        .resize_alg(fast_image_resize::ResizeAlg::Convolution(
            fast_image_resize::FilterType::Lanczos3,
        ));
    match fast_image_resize::Resizer::new().resize(img, &mut dst, &opts) {
        Ok(()) => dst,
        Err(_) => two_step(),
    }
}

/// Pad `img` to exactly `target_w` x `target_h`: fit the image
/// inside the box (the fit-in rules, `no_upscale` cap included),
/// center it, and fill the rest of the canvas: a solid color
/// (the pad segment's own, or the request's `fill(...)`
/// directive, or black), or a blurred cover-scaled copy of the
/// image itself. Also returns where the fitted image sits on
/// the canvas, which the pending focal form follows (issue
/// #465).
fn apply_pad(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    fill: PadFill,
    directive_fill: Option<[u8; 3]>,
    cap: Option<(u32, u32)>,
) -> Result<(DynamicImage, Window), EngineError> {
    if target_w == 0 || target_h == 0 {
        let whole = (0, 0, img.width(), img.height());
        return Ok((img, whole));
    }
    check_output_size(target_w, target_h, (img.width(), img.height()))?;
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
        return Ok((fitted, (0, 0, target_w, target_h)));
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
            crate::filters::blur_fast(
                &img.resize_to_fill(target_w, target_h, image::imageops::FilterType::Triangle),
                sigma,
            )
        }
    };
    let mut base = canvas.to_rgba8();
    let x = (target_w - fitted.width()) / 2;
    let y = (target_h - fitted.height()) / 2;
    image::imageops::overlay(&mut base, &fitted.to_rgba8(), x as i64, y as i64);
    let placed = (x, y, fitted.width(), fitted.height());
    Ok((DynamicImage::ImageRgba8(base), placed))
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

/// Scale proportionally to fit inside the `fit-in` box, preserving
/// aspect and never distorting. A 0 dimension bounds nothing;
/// `orig` bounds at the source dimension (a scale of 1). The box
/// scales by `dpr` first, and with `adaptive` its axes swap to
/// match the source's orientation (see [`fit_in_box`]); `cap`
/// then shrinks the result to fit inside it (`no_upscale`).
fn apply_fit_in(
    img: DynamicImage,
    size: &ImageSize,
    adaptive: bool,
    cap: Option<(u32, u32)>,
    dpr: f32,
) -> Result<DynamicImage, EngineError> {
    check_nonzero_size(size)?;
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
    check_output_size(w, h, (sw, sh))?;
    if w == sw && h == sh {
        Ok(img)
    } else {
        Ok(resize_lanczos3(&img, w, h))
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod test_formats;
