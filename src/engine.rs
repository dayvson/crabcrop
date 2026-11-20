//! The image engine: decode, transform, and encode.
//!
//! [`Engine::process`] takes source bytes and the parsed [`Operation`]s,
//! and returns the encoded output plus its format.

use std::io::Cursor;

use image::DynamicImage;

use crate::filters::EncodeOptions;
use crate::quality::auto_quality;
#[cfg(test)]
use crate::url::PadFill;
use crate::url::{
    Align, Dimension, Filter, FormatOptions, ImageFormat, ImageSize, Operation, PlainFilter,
    ResolvedQuality,
};

// The operation cap lives with the grammar (issue #469);
// the engine's tests reach it from here through `super::*`.
#[cfg(test)]
use crate::url::MAX_OPERATIONS;

mod limits;
#[cfg(test)]
pub(in crate::engine) use limits::hint_step_down;
pub(in crate::engine) use limits::{
    MAX_ANIMATED_FRAMES, MAX_ANIMATED_PIXELS, OutputBudget, check_final_output,
    check_hinted_output, check_operation_count, check_output_pixels, check_output_size, dpr_factor,
    step_down_hint,
};
pub(crate) use limits::{
    MAX_DECODE_ALLOC, MAX_DECODE_DIMENSION, MAX_OUTPUT_PIXELS, dpr_step_below, scales_with_dpr,
};
mod decode;
#[cfg(test)]
pub(in crate::engine) use decode::{
    MAX_WATERMARK_BYTES, decode_jxl, decode_watermarks_within, decode_webp_direct, jxl_dimensions,
    jxl_sample_limit,
};
pub(crate) use decode::{
    PreparedUpload, apply_orientation, check_decoded_dimensions, decode_watermarks,
    exif_orientation, is_jxl_codestream, prepare_upload, probe_is_heavy, probe_kind, probe_source,
    sniff_content_type, svg_raster_target,
};
pub use decode::{
    SourceProbe, UploadMedia, decode_image, decode_jpeg_scaled, validate_source_image,
};
pub(in crate::engine) use decode::{
    check_jxl_dimensions, decode_limits, decode_source_image, detect_format, ensure_jxl_decoder,
    is_heic, is_jxl, zune_jpeg_options,
};
mod resize;
#[cfg(test)]
pub(in crate::engine) use resize::{
    CONVOLVE_REFUSES, ChangingRows, fit_in_box, place_on, resize_lanczos3_cropped,
    resize_lanczos3_owned, view_holds_no_more,
};
pub(in crate::engine) use resize::{
    SizeCap, SourceWindow, apply_fit_in, apply_fit_in_lent, apply_pad, apply_size,
    apply_size_cropped, apply_size_lent, apply_size_windowed, resize_lanczos3, resize_target,
    scale_px,
};
pub(crate) use resize::{cap_dimensions, has_zero_side};
#[cfg(any(test, doc))]
pub(in crate::engine) use resize::{resize_lanczos3_cut, resize_lanczos3_lent};
mod overlay;
#[cfg(test)]
pub(in crate::engine) use overlay::{blend_channel, blend_overlay, composite, landing, lands_on};
pub(in crate::engine) use overlay::{
    blend_overlay_keeping, flatten, flattened_for, overlay_watermark,
};
mod detect;
pub(in crate::engine) use detect::{
    DetectionMemo, FaceMemo, MAX_SEGMENT_DECODES, SegmentMemo, detect_faces, detect_for_class,
    detection_providers, filter_preserves_positions, invalidate_detection_memos, resolve_detector,
    segment_for_class,
};

/// The per-request encode budget (audit E-6): the
/// quality(auto) verification loop and the max_bytes ladder
/// draw from one pool, so their multipliers cannot stack
/// into a dozen-plus full-size encodes per cache miss.
const MAX_REQUEST_ENCODES: usize = 6;

/// The quality a `Save-Data` request steps down to: one
/// notch (10 points) below the default 80 the encoders and
/// the `max_bytes` ladder assume.
pub const SAVE_DATA_QUALITY: u8 = 70;

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

/// A filter that stopped at the processing deadline (issues
/// #606, #614) fails the render as any other deadline check
/// does.
impl From<crate::filters::DeadlinePassed> for EngineError {
    fn from(_: crate::filters::DeadlinePassed) -> Self {
        EngineError::Timeout
    }
}

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
            avif_tiles: AvifTiles::default(),
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
/// gate like any other (issue #379). An animated render,
/// image or video, runs that analysis once, on the first
/// frame, and replays the window (issues #376, #713). A
/// still spelling of the same source (a `frame:` selection
/// renders one frame) runs these operations normally.
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
    /// still format), with the source metadata. The last
    /// field is set for a JPEG decoded at reduced scale
    /// (issue #653): the image the frame is a reduction of,
    /// whose size is the source's wherever one is reported
    /// or resolved against.
    Still(
        DynamicImage,
        crate::color::SourceMeta,
        Option<jpeg_scaled::Reduced>,
    ),
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
    /// The first guess of the SSIMULACRA2 loop, set when
    /// `quality(auto)` verifies its pick through it (JPEG and
    /// JPEG XL outputs): the heuristic pick, which is also
    /// `encode_quality`.
    perceptual: Option<u8>,
}

/// The still encode's result: the bytes, the quality the
/// service chose for them (reported back), and their
/// SSIMULACRA2 score when the perceptual loop verified them.
struct StillEncode {
    bytes: Vec<u8>,
    chosen_quality: Option<u8>,
    ssim: Option<f64>,
}

/// A resolved quality directive as an encode runs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct QualityInForce {
    /// The quality the encode uses; `None` keeps the encoder
    /// default.
    encode: Option<u8>,
    /// The quality reported back, set when the service chose
    /// it rather than the caller.
    chosen: Option<u8>,
}

/// Resolve a quality directive for an encode. `quality(auto)`
/// is chosen by `pick` and reported. With no directive in
/// force, `save_data` steps the default down one notch and
/// reports the step like an auto pick; an explicit
/// `quality(...)` of either kind stands. The still and the
/// animated WebP encodes both resolve through here.
fn quality_in_force(
    resolved: Option<ResolvedQuality>,
    save_data: bool,
    pick: impl FnOnce() -> u8,
) -> QualityInForce {
    let chosen = |q| QualityInForce {
        encode: Some(q),
        chosen: Some(q),
    };
    match resolved {
        Some(ResolvedQuality::Value(q)) => QualityInForce {
            encode: Some(q),
            chosen: None,
        },
        Some(ResolvedQuality::Auto) => chosen(pick()),
        None if save_data => chosen(SAVE_DATA_QUALITY),
        None => QualityInForce::default(),
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
    ///
    /// The stages run in order: frame and page selection,
    /// sniffing, preflight, decode, the operation chain
    /// (`apply_operations`), the still encode, and metadata
    /// embedding. The functions after this one hold every
    /// stage but the chain.
    ///
    /// A DPR factor from the client hint (`ctx.dpr`) is
    /// advisory (issue #529): the render starts at the
    /// largest factor up to the hint whose sizes the source's
    /// header shows to fit the output limits, and when the
    /// render still refuses its output size there, the stages
    /// after selection run again at the next factor down
    /// (`step_down_hint`), so the hint never turns a URL that
    /// renders without it into an error.
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
        let video = frame.as_ref().map(|(_, meta)| *meta);
        let pdf = page.as_ref().map(|(_, meta)| *meta);
        // Selection runs once, so a frame or page tool never
        // runs again for a lower factor. A retry does run the
        // stages after it again (the decode, and for an SVG
        // the rasterizing worker), but only for a refusal the
        // header walk could not foresee; like the first run,
        // it starts only before the deadline.
        step_down_hint(bytes, ops, &directives, ctx.dpr, |dpr| {
            let ctx = ProcessContext { dpr, ..ctx.clone() };
            self.process_selected(bytes, ops, ctx, &directives, video, pdf)
        })
    }

    /// The stages of [`Engine::process_preferred`] after frame
    /// and page selection, at the DPR factor `ctx` carries:
    /// sniffing, preflight, decode, the operation chain, the
    /// still encode, and metadata embedding. `video` and `pdf`
    /// are the selection's records, carried onto the result.
    /// `directives` are the request's one analysis of `ops`,
    /// which every stage below reads (issue #480).
    fn process_selected(
        &self,
        bytes: &[u8],
        ops: &[Operation],
        ctx: ProcessContext<'_>,
        directives: &crate::url::RequestDirectives,
        video: Option<VideoSourceMeta>,
        pdf: Option<PdfSourceMeta>,
    ) -> Result<ProcessedImage, EngineError> {
        // Sniffing screens an SVG in this process, and the
        // stages before it (frame or page selection, a run at
        // a higher factor) may have used the whole budget:
        // nothing here starts past the deadline.
        ctx.check_deadline()?;
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
        Self::preflight(directives, &ctx, detected)?;
        let (img, source_meta, reduced) =
            match Self::decode_source(&source, ops, &ctx, directives, keep_profile)? {
                Decoded::Still(img, source_meta, reduced) => (img, source_meta, reduced),
                Decoded::Rendered(processed) => return Ok(*processed),
            };
        let mut format = detected;
        ctx.check_deadline()?;
        let (outcome, (source_width, source_height)) =
            Self::still_chain(img, reduced, ops, &ctx, directives)?;
        let ApplyOutcome {
            img,
            encode_opts,
            studio,
            inpaint,
            segments,
            objects_crop,
        } = outcome;
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
        let Some(still) = StillFormat::of(format) else {
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
        };
        let encoded = Self::encode_still(&img, still, &encode_opts, &ctx)?;
        let bytes = Self::embed_source_meta(
            encoded.bytes,
            still,
            encode_opts.keep_profile,
            source_meta,
            (img.width(), img.height()),
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
            video,
            pdf,
            provenance: None,
        })
    }

    /// The operation chain of a still render, on the frame
    /// the decode stage handed on: what the chain made of it,
    /// and the source size the render reports.
    ///
    /// A frame the JPEG decoder reduced (issue #653) stands for
    /// the image its header declares, `reduced`. That image's
    /// size is the source size and the `no_upscale` cap, and
    /// the chain's first operation, the plain resize that made
    /// the decode eligible (see [`jpeg_scaled::plan`]),
    /// resolves its target against it and resamples the
    /// image's part of the frame to that target. Every
    /// operation after it starts from an image of the size it
    /// always started from.
    fn still_chain(
        img: DynamicImage,
        reduced: Option<jpeg_scaled::Reduced>,
        ops: &[Operation],
        ctx: &ProcessContext<'_>,
        directives: &crate::url::RequestDirectives,
    ) -> Result<(ApplyOutcome, (u32, u32)), EngineError> {
        let source_size = match reduced {
            Some(reduced) => reduced.full,
            None => (img.width(), img.height()),
        };
        // `no_upscale` is a request-wide directive, like the encoding
        // directives: wherever it appears, it caps every resize in
        // the request at the (oriented) source dimensions.
        let cap = if directives.no_upscale {
            Some(source_size)
        } else {
            None
        };
        check_operation_count(ops)?;
        let dpr = dpr_factor(directives.dpr, ctx);
        let mut chain = ChainState::new(ops, cap, dpr, ctx, directives, None);
        chain.reduced = reduced;
        Ok((chain.run(img, ops)?, source_size))
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
        // GIF sources, and WebP sources carrying an ANIM chunk
        // (whose probe also yields the loop count the animated
        // path carries over), decode frame by frame.
        let animated = match detected {
            ImageFormat::Gif => Some((AnimatedFormat::Gif, None)),
            ImageFormat::WebP => webp_animation_params(bytes)
                .map(|(loop_count, _)| (AnimatedFormat::WebP, Some(loop_count))),
            _ => None,
        };
        if let Some((format, loop_count)) = animated {
            return Self::decode_animated(
                bytes,
                format,
                loop_count,
                ops,
                ctx,
                directives,
                source_meta,
            );
        }
        // Set when the decode hands on a reduced frame (see
        // [`Decoded::Still`]).
        let mut reduced = None;
        let img = if let Some(screened) = &source.svg {
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
            // A JPEG whose first operation shrinks it far enough
            // decodes at reduced scale (issue #653), and the
            // frame comes with the EXIF orientation the plan
            // read. Every other still decodes as it always did:
            // an AVIF source in the capped worker, killed at the
            // request deadline (issue #522), and every other
            // still in process.
            let dpr = dpr_factor(directives.dpr, ctx);
            let (mut img, orientation) =
                match jpeg_scaled::decode_planned(bytes, ops, directives, dpr, ctx)? {
                    Some((frame, scaled)) => {
                        reduced = Some(scaled.reduced());
                        (frame, scaled.orientation)
                    }
                    None => (
                        decode_source_image(bytes, ctx.deadline)?,
                        exif_orientation(bytes),
                    ),
                };
            // Phone photos store rotation in the EXIF orientation
            // tag instead of the pixels. Honor it before any
            // operation, so sizes and crops see the image as the
            // viewer would. (GIFs carry no EXIF, hence the split.)
            if let Some(orientation) = orientation {
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
        Ok(Decoded::Still(img, source_meta, reduced))
    }

    /// Decode an animated GIF or WebP source frame by frame. An
    /// animated source whose output stays animated (GIF or
    /// WebP) keeps its frames, each running the operations with
    /// its delay preserved, and one whose output is a video
    /// encodes them: either render is the request's result.
    /// Anything else continues with the first frame, which is
    /// also what converting an animation to a still format
    /// means, as does a source with one frame.
    fn decode_animated(
        bytes: &[u8],
        source: AnimatedFormat,
        loop_count: Option<u16>,
        ops: &[Operation],
        ctx: &ProcessContext<'_>,
        directives: &crate::url::RequestDirectives,
        source_meta: crate::color::SourceMeta,
    ) -> Result<Decoded, EngineError> {
        let frames = match source {
            AnimatedFormat::Gif => decode_gif_frames(bytes, ctx.deadline)?,
            AnimatedFormat::WebP => decode_webp_frames(bytes, ctx.deadline)?,
        };
        if frames.len() > 1 {
            if directives.debug_focal {
                return Err(EngineError::Unsupported(
                    "debug:focal renders a diagnostic image and cannot annotate an animated source"
                        .to_string(),
                ));
            }
            let resolved =
                animated_output_format(source.into(), directives.explicit_format, ctx.preferred);
            if let Some(format) = AnimatedFormat::of(resolved) {
                let processed =
                    Self::process_animated(frames, ops, ctx, directives, format, loop_count)?;
                return Ok(Decoded::Rendered(Box::new(processed)));
            }
            if let Some(format) = crate::video::VideoFormat::of(resolved) {
                let processed = Self::process_video(frames, ops, ctx, directives, format)?;
                return Ok(Decoded::Rendered(Box::new(processed)));
            }
        }
        let first = DynamicImage::ImageRgba8(frames.first.into_buffer());
        Ok(Decoded::Still(first, source_meta, None))
    }

    /// Resolve the quality directive against the final image
    /// and the settled output format, for the still encode.
    fn resolve_still_quality(
        img: &DynamicImage,
        format: StillFormat,
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
            .and_then(|q| q.resolved_for(format.into()));
        // Save-Data steps the default quality down one notch
        // (80 to 70) on the formats that carry a quality.
        let save_data = ctx.save_data && format.carries_quality();
        let QualityInForce {
            encode: encode_quality,
            chosen: chosen_quality,
        } = quality_in_force(resolved_quality, save_data, || auto_quality(img));
        // `quality(auto)` on a format with a decoder to verify
        // against (JPEG, JXL) closes the loop: encode, decode,
        // score with SSIMULACRA2, and bisect on quality until the
        // score lands in the target band. The heuristic pick is
        // the first guess. Other formats keep the one-shot pick,
        // and so does an output too large to verify (see
        // `encode_still_within`).
        let perceptual = encode_quality.filter(|_| {
            matches!(resolved_quality, Some(ResolvedQuality::Auto))
                && matches!(format, StillFormat::Jpeg | StillFormat::Jxl)
        });
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
        format: StillFormat,
        encode_opts: &EncodeOptions,
        ctx: &ProcessContext<'_>,
    ) -> Result<StillEncode, EngineError> {
        // A JPEG with a fill encodes the image flattened over
        // it, so that is the image the encodes, the quality
        // pick and the scores read (issue #713): scoring the
        // unflattened one measured the colors under its
        // transparent pixels, which the file never shows. It
        // is flattened once here rather than at every encode.
        let flattened = flattened_for(img, format, encode_opts.fill);
        let img = flattened.as_ref().unwrap_or(img);
        // Every candidate of a verified search is scored
        // against this image: the metric's work on it is done
        // once, at the first score, and dropped with this call
        // (issue #664). An encode that scores nothing prepares
        // nothing.
        let reference = crate::quality::SsimReference::new(img);
        Self::encode_still_within(
            img,
            format,
            encode_opts,
            ctx,
            crate::quality::MAX_VERIFIED_PIXELS,
            |candidate| {
                decode_image(candidate)
                    .ok()
                    .and_then(|decoded| reference.score(&decoded))
            },
        )
    }

    /// [`Engine::encode_still`] with the two things a test
    /// stands in for: `max_verified_pixels`, the largest
    /// output whose `quality(auto)` pick is verified (always
    /// [`crate::quality::MAX_VERIFIED_PIXELS`]), and `score`,
    /// which decodes a candidate's bytes and scores them
    /// against `img`.
    fn encode_still_within(
        img: &DynamicImage,
        format: StillFormat,
        encode_opts: &EncodeOptions,
        ctx: &ProcessContext<'_>,
        max_verified_pixels: u64,
        score: impl Fn(&[u8]) -> Option<f64>,
    ) -> Result<StillEncode, EngineError> {
        let StillQuality {
            encode_quality,
            mut chosen_quality,
            perceptual,
        } = Self::resolve_still_quality(img, format, encode_opts, ctx);
        // `quality(auto)` verifies its pick for an output of
        // up to `max_verified_pixels`. A larger output is
        // encoded once at the pick and reports it, as the
        // formats with no search do. That is settled here,
        // before any candidate is encoded.
        let perceptual = perceptual
            .filter(|_| u64::from(img.width()) * u64::from(img.height()) <= max_verified_pixels);
        let mut ssim: Option<f64> = None;
        let enc_src = EncodeBuffers::new(img);
        // Every encode of this image differs only in its quality.
        let encode_at = |quality: Option<u8>| {
            encode(
                &enc_src,
                format,
                quality,
                encode_opts.fill,
                encode_opts.palette,
                encode_opts.format_options,
                ctx.avif_speed,
                ctx.avif_tiles,
                ctx.jxl_speed,
            )
        };
        let mut encode_budget = MAX_REQUEST_ENCODES;
        let mut bytes = if let Some(first_guess) = perceptual {
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
                    let candidate = encode_at(Some(q))?;
                    let score = score(&candidate);
                    Ok::<_, EngineError>((candidate, score))
                },
            )?;
            chosen_quality = Some(scored.quality);
            ssim = scored.score;
            scored.bytes
        } else {
            let t_timing = std::time::Instant::now();
            let bytes = encode_at(encode_quality)?;
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
        let mut ladder_ran = false;
        if let Some(budget) = encode_opts.max_bytes.filter(|_| format.carries_quality()) {
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
                bytes = encode_at(Some(current))?;
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
            ssim = score(&bytes);
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
    /// JPEG and PNG outputs carry either in this pass. A
    /// preserved EXIF gives the output's size, `size`, not
    /// the source's (issue #611).
    fn embed_source_meta(
        bytes: Vec<u8>,
        format: StillFormat,
        keep_profile: bool,
        mut source_meta: crate::color::SourceMeta,
        size: (u32, u32),
        ctx: &ProcessContext<'_>,
    ) -> Vec<u8> {
        if ctx.preserve_metadata
            && let Some(tiff) = &mut source_meta.exif_tiff
        {
            crate::color::patch_tiff_dimensions(tiff, size.0, size.1);
        }
        let source_meta = &source_meta;
        if (keep_profile && source_meta.icc.is_some()) || ctx.preserve_metadata {
            let embed_icc = if keep_profile {
                source_meta.icc.as_deref()
            } else {
                None
            };
            match format {
                StillFormat::Jpeg => {
                    crate::color::inject_jpeg(&bytes, embed_icc, source_meta, ctx.preserve_metadata)
                }
                StillFormat::Png => {
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
        frames: Frames,
        ops: &[Operation],
        ctx: &ProcessContext<'_>,
        directives: &crate::url::RequestDirectives,
        format: AnimatedFormat,
        loop_count: Option<u16>,
    ) -> Result<ProcessedImage, EngineError> {
        let source_size = frames.dimensions();
        // One smart crop window per crop for the whole
        // animation (issue #376): the first frame decides,
        // the rest replay. One trim window per trim (issue
        // #465): the union over every frame, settled before
        // any frame renders, so the frames share one canvas.
        let (mut frames, encode_opts) =
            render_frames(frames, ops, ctx, directives, FrameShare::default())?;
        // Resolve the quality directive for a WebP output the
        // same way the still path does: an explicit value is
        // used as given, `quality(auto)` is picked from the
        // first processed frame (the frame the placeholder also
        // describes) and reported back, and Save-Data steps an
        // unspecified quality down one notch. GIF output has no
        // quality lever, so nothing resolves for it.
        let quality = match format {
            AnimatedFormat::WebP => {
                let resolved = encode_opts
                    .quality
                    .as_ref()
                    .and_then(|q| q.resolved_for(ImageFormat::WebP));
                quality_in_force(resolved, ctx.save_data, || frames.with_first(auto_quality))
            }
            AnimatedFormat::Gif => QualityInForce::default(),
        };
        let placeholder = first_frame_placeholder(&mut frames, ctx);
        let out = match format {
            AnimatedFormat::Gif => encode_animated_gif(&frames, ctx.deadline)?,
            AnimatedFormat::WebP => encode_animated_webp(
                &frames,
                loop_count.unwrap_or(0),
                quality.encode,
                encode_opts.format_options,
                ctx.deadline,
            )?,
        };
        Ok(ProcessedImage::from_frames(
            out,
            format.into(),
            quality.chosen,
            frames.dimensions(),
            source_size,
            placeholder,
        ))
    }

    /// Process an animated source (GIF or WebP) into a
    /// video (issue #162): every frame runs the same
    /// operations as the animated image paths, then the
    /// operator-configured external encoder turns the frames
    /// into MP4 or WebM under the pipe contract in
    /// [`crate::video`]. Frame delays travel with the
    /// frames; the encoder side turns them into a constant
    /// frame rate (see `frame_schedule`). The reported
    /// dimensions are the encoder's: an odd side pads up to
    /// even for 4:2:0, or crops down at the per-axis limit
    /// (see [`crate::video::even_dimension`]).
    fn process_video(
        frames: Frames,
        ops: &[Operation],
        ctx: &ProcessContext<'_>,
        directives: &crate::url::RequestDirectives,
        format: crate::video::VideoFormat,
    ) -> Result<ProcessedImage, EngineError> {
        let encoder = ctx.video_encoder.ok_or_else(|| {
            EngineError::NotConfigured(
                "video output is not configured on this server (no CRABCROP_VIDEO_ENCODER)"
                    .to_string(),
            )
        })?;
        let source_size = frames.dimensions();
        // The frames share their smart crop windows and their
        // trim windows (issues #376, #465, #713), as on the
        // animated image paths: the first frame decides each
        // window, so a smart crop holds still from frame to
        // frame, and the encoder takes one frame size.
        let (mut frames, _) = render_frames(frames, ops, ctx, directives, FrameShare::default())?;
        // The placeholder is a meta-only surface (meta itself
        // refuses video URLs in the server). It is taken here,
        // before the frames move into the encoder's.
        let placeholder = first_frame_placeholder(&mut frames, ctx);
        let (width, height) = frames.dimensions();
        let out_frames: Vec<crate::video::VideoFrame> = frames
            .into_iter()
            .map(|frame| crate::video::VideoFrame {
                delay_ms: std::time::Duration::from(frame.delay()).as_millis() as u64,
                pixels: frame.into_buffer(),
            })
            .collect();
        let bytes = encoder.encode(out_frames, format).map_err(|e| match e {
            crate::video::VideoError::Unsupported(what) => EngineError::Unsupported(what),
            crate::video::VideoError::Failed(what) => EngineError::Video(what),
        })?;
        Ok(ProcessedImage::from_frames(
            bytes,
            format.into(),
            None,
            (
                crate::video::even_dimension(width),
                crate::video::even_dimension(height),
            ),
            source_size,
            placeholder,
        ))
    }
}

/// The placeholder of an animated or video render, when the
/// caller asked for one: it describes the first processed
/// frame.
fn first_frame_placeholder(
    frames: &mut Frames,
    ctx: &ProcessContext<'_>,
) -> Option<crate::placeholder::Placeholder> {
    ctx.placeholders
        .then(|| frames.with_first(crate::placeholder::compute))
}

/// Run the operations over every frame of an animated render,
/// for the animated image and the video paths alike: the
/// inference-class operations refused first (see
/// [`refuse_animated_ai`]), the frames' shared geometry in
/// `share` settled before any frame renders (see
/// [`survey_trims`]), and the run's pixels charged against one
/// output budget, which the first frame's size settles up front
/// (see [`OutputBudget::check_frames`]). The rendered frames
/// keep their delays. The encoding directives are a property of
/// the ops, not of a frame, so every frame reports the same
/// ones; the first frame's copy is the request's and comes back
/// with the frames.
fn render_frames(
    mut frames: Frames,
    ops: &[Operation],
    ctx: &ProcessContext<'_>,
    directives: &crate::url::RequestDirectives,
    mut share: FrameShare,
) -> Result<(Frames, EncodeOptions), EngineError> {
    refuse_animated_ai(directives)?;
    let cap = if directives.no_upscale {
        Some(frames.dimensions())
    } else {
        None
    };
    let dpr = dpr_factor(directives.dpr, ctx);
    survey_trims(&mut frames, ops, cap, dpr, ctx, directives, &mut share)?;
    let frame_count = frames.len();
    let mut budget = OutputBudget::new();
    let mut render = |frame: image::Frame| {
        ctx.check_deadline()?;
        let delay = frame.delay();
        let img = DynamicImage::ImageRgba8(frame.into_buffer());
        let outcome = apply_operations(img, ops, cap, dpr, ctx, directives, Some(&mut share))?;
        Ok::<_, EngineError>((outcome.img, outcome.encode_opts, delay))
    };
    let (img, encode_opts, delay) = render(frames.first)?;
    budget.check_frames(img.width(), img.height(), frame_count)?;
    budget.charge(img.width(), img.height())?;
    // A rendered frame is the chain's own image, so an RGBA8
    // one becomes the frame without a copy (issue #713); any
    // other layout converts as it always did.
    let first = image::Frame::from_parts(img.into_rgba8(), 0, 0, delay);
    let mut rest = Vec::with_capacity(frames.rest.len());
    for frame in frames.rest {
        let (img, _, delay) = render(frame)?;
        budget.charge(img.width(), img.height())?;
        rest.push(image::Frame::from_parts(img.into_rgba8(), 0, 0, delay));
    }
    Ok((Frames { first, rest }, encode_opts))
}

/// Encode rendered frames as an animated GIF: one screen the
/// first frame's size, and each frame quantized onto a palette
/// of its own (see [`gif_frame_quantized`]) with its delay in
/// hundredths of a second. GIF output has no quality lever and
/// keeps the GIF encoder's defaults otherwise, as it always has.
/// The frames are prepared on up to two threads (see
/// [`gif_frame_workers`]) and written in frame order.
fn encode_animated_gif(
    frames: &Frames,
    deadline: Option<std::time::Instant>,
) -> Result<Vec<u8>, EngineError> {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let (width, height) = frames.dimensions();
    let workers = gif_frame_workers(cores, frames.len(), width, height);
    encode_animated_gif_on(frames, deadline, workers, gif_frame_quantized)
}

/// The most threads that prepare the frames of one animated
/// GIF (issue #695). An animated GIF render holds one permit of
/// the stills lane, like any still, so its threads share the
/// cores with the still renders beside it. With four, a still
/// render beside one GIF encode took 1.13 to 1.29 times as long
/// as beside the single-thread encoder, and its 90th percentile
/// about 1.6 times; with two it took 0.98 times as long, and a
/// GIF render still got 1.76 times the throughput of one thread
/// (four gave 2.34). The bytes are the same at any count.
const GIF_FRAME_THREADS: usize = 2;

/// How many threads prepare the frames of an animated GIF on
/// a host with `cores` cores (issue #667): the cores, at most
/// [`GIF_FRAME_THREADS`], never more than the frames, and one
/// for a single frame or for an animation under 64x64 pixels,
/// too small to repay a thread.
fn gif_frame_workers(cores: usize, frames: usize, width: u32, height: u32) -> usize {
    if frames < 2 || u64::from(width) * u64::from(height) < 64 * 64 {
        return 1;
    }
    cores.clamp(1, GIF_FRAME_THREADS).min(frames)
}

/// [`encode_animated_gif`] on `workers` threads. `quantize`
/// turns one frame's RGBA pixels, width, height and delay
/// into its GIF frame: always [`gif_frame_quantized`], and a
/// parameter only so a test can see which frames are
/// prepared.
///
/// Each frame is quantized onto its own palette and compressed
/// by itself, with nothing carried from one frame to the next,
/// so the frames can be prepared in any order: one thread
/// writes each as it goes, and several prepare them side by
/// side (see [`in_index_order`]) and then write them in frame
/// order, the LZW pass `write_frame` runs having run on the
/// thread that quantized the frame. The file is the same
/// bytes either way.
///
/// The prepared frames wait for the write as palette and
/// LZW data, the file's own image data: at most about a byte
/// and a half per pixel, where the frames they come from hold
/// four, so the frame cap and the pixel budgets that bound the
/// frames bound them too. Each thread works on one frame at a
/// time.
fn encode_animated_gif_on(
    frames: &Frames,
    deadline: Option<std::time::Instant>,
    workers: usize,
    quantize: impl Fn(&[u8], u32, u32, u16) -> Result<gif::Frame<'static>, EngineError> + Sync,
) -> Result<Vec<u8>, EngineError> {
    let mut out = Vec::new();
    let (canvas_w, canvas_h) = frames.dimensions();
    let mut enc = gif::Encoder::new(
        &mut out,
        u16::try_from(canvas_w).map_err(|_| {
            EngineError::Encode("gif: width exceeds the 65535 format limit".to_string())
        })?,
        u16::try_from(canvas_h).map_err(|_| {
            EngineError::Encode("gif: height exceeds the 65535 format limit".to_string())
        })?,
        &[],
    )
    .map_err(|e| EngineError::Encode(e.to_string()))?;
    let prepare = |f: &image::Frame| {
        // The deadline is checked per frame inside the encode
        // loop itself (issue #322), on whichever thread takes
        // the frame: a many-frame GIF encode is chunked work,
        // so it aborts at the deadline instead of running
        // every frame's quantization first and being noticed
        // only afterwards.
        state::check_deadline(deadline)?;
        // GIF delays are hundredths of a second; the image
        // crate saturated this conversion at u16::MAX, and so
        // does this.
        let (numer, denom) = f.delay().numer_denom_ms();
        let delay = u16::try_from(numer / denom / 10).unwrap_or(u16::MAX);
        let buffer = f.buffer();
        quantize(buffer.as_raw(), buffer.width(), buffer.height(), delay)
    };
    if workers <= 1 {
        for f in frames.iter() {
            let frame = prepare(f)?;
            enc.write_frame(&frame)
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
    } else {
        let all: Vec<&image::Frame> = frames.iter().collect();
        let prepared = in_index_order(all.len(), workers, |index| {
            let mut frame = prepare(all[index])?;
            frame.make_lzw_pre_encoded();
            Ok(frame)
        })?;
        for frame in &prepared {
            enc.write_lzw_pre_encoded_frame(frame)
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
    }
    drop(enc);
    Ok(out)
}

/// Run `work` for every index below `count` on up to `workers`
/// threads, the calling thread among them, and return the
/// results in index order.
///
/// The threads take the indexes in order from a shared
/// counter. A failure stops every thread before its next
/// index, so the work behind it is never started, and the
/// failure returned is the first in index order: the one a
/// loop over the indexes would have met. (An index is left
/// untaken only behind one that failed, and every index taken
/// is finished, so no index below the first failure is
/// missing.) A panic in `work` stops the other threads the
/// same way and is raised again on the calling thread, where
/// it unwinds as it would have from a loop.
///
/// The threads are scoped: all of them have ended when this
/// returns, however it returns. A thread the host cannot
/// start is one helper fewer, not a failure, since the
/// calling thread takes indexes too.
fn in_index_order<T: Send>(
    count: usize,
    workers: usize,
    work: impl Fn(usize) -> Result<T, EngineError> + Sync,
) -> Result<Vec<T>, EngineError> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    /// Sets the flag when dropped by a panic's unwinding.
    struct StopOnPanic<'a>(&'a AtomicBool);
    impl Drop for StopOnPanic<'_> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                self.0.store(true, Ordering::Relaxed);
            }
        }
    }
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let take = || {
        let mut done: Vec<(usize, Result<T, EngineError>)> = Vec::new();
        // Declared after `done`, so a panic's unwinding sets
        // the flag before it drops what this thread finished.
        let _stop_on_panic = StopOnPanic(&stop);
        while !stop.load(Ordering::Relaxed) {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= count {
                break;
            }
            let result = work(index);
            if result.is_err() {
                stop.store(true, Ordering::Relaxed);
            }
            done.push((index, result));
        }
        done
    };
    let mut results: Vec<Option<Result<T, EngineError>>> = (0..count).map(|_| None).collect();
    std::thread::scope(|scope| {
        let helpers: Vec<_> = (1..workers.min(count))
            .filter_map(|_| std::thread::Builder::new().spawn_scoped(scope, take).ok())
            .collect();
        let mut done = take();
        for helper in helpers {
            match helper.join() {
                Ok(more) => done.extend(more),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        for (index, result) in done {
            results[index] = Some(result);
        }
    });
    results
        .into_iter()
        .map(|result| result.expect("an index is left untaken only behind a failed one"))
        .collect()
}

/// Collect an animation decoder's frames under the shared
/// limits: a frame cap, a total-decoded-pixels budget, and
/// the decode allocation budget charged on the retained
/// RGBA bytes, so a small file cannot expand into gigabytes
/// of frames. The pixel budget alone is not a memory bound:
/// at its 100M-pixel cap the retained frames are 400 MiB,
/// well past the 256 MiB allocation budget every still
/// decode honors. The request's `deadline` is checked before
/// each frame decodes, so a long animation stops with
/// [`EngineError::Timeout`] at the deadline instead of
/// decoding on to its last frame. Errors when the source is
/// malformed, over budget, or carries no frames. `format`
/// names the source format in error messages.
fn collect_frames(
    mut frames: impl Iterator<Item = Result<image::Frame, image::ImageError>>,
    format: &str,
    deadline: Option<std::time::Instant>,
) -> Result<Frames, EngineError> {
    let mut collected: Option<Frames> = None;
    let mut total_pixels: u64 = 0;
    let mut total_bytes: u64 = 0;
    loop {
        state::check_deadline(deadline)?;
        let Some(frame) = frames.next() else {
            break;
        };
        let frame = frame.map_err(|e| EngineError::Decode(e.to_string()))?;
        if collected.as_ref().map_or(0, Frames::len) >= MAX_ANIMATED_FRAMES {
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
        match &mut collected {
            None => {
                collected = Some(Frames {
                    first: frame,
                    rest: Vec::new(),
                })
            }
            Some(frames) => frames.rest.push(frame),
        }
    }
    collected.ok_or_else(|| EngineError::Decode(format!("{format} has no frames")))
}

/// The frames of an animated source (issue #480): never empty,
/// so the first frame is always there to read, the one a still
/// conversion keeps and whose size an animated render starts
/// from.
struct Frames {
    first: image::Frame,
    rest: Vec<image::Frame>,
}

impl Frames {
    fn len(&self) -> usize {
        1 + self.rest.len()
    }

    fn iter(&self) -> impl Iterator<Item = &image::Frame> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut image::Frame> {
        std::iter::once(&mut self.first).chain(&mut self.rest)
    }

    /// The first frame's width and height.
    fn dimensions(&self) -> (u32, u32) {
        self.first.buffer().dimensions()
    }

    /// `f` of the first frame as a `DynamicImage`, lent for the
    /// call (see [`lend_frame`]), where it was copied once for
    /// each look (issue #713): the quality pick and the
    /// placeholder each read the whole frame.
    fn with_first<T>(&mut self, f: impl FnOnce(&DynamicImage) -> T) -> T {
        lend_frame(&mut self.first, |img| f(img))
    }
}

/// `f` of `frame` as a `DynamicImage`: the frame's buffer is
/// lent for the call and put back, with the frame's position
/// and delay, instead of copied (issues #713, #737). `f` may
/// work on the pixels where they sit, as the trim survey's
/// resize does (see [`resize_lanczos3_lent`]), as long as it
/// leaves them as they came.
fn lend_frame<T>(frame: &mut image::Frame, f: impl FnOnce(&mut DynamicImage) -> T) -> T {
    let (left, top, delay) = (frame.left(), frame.top(), frame.delay());
    let empty = image::Frame::new(image::RgbaImage::new(0, 0));
    let mut img = DynamicImage::ImageRgba8(std::mem::replace(frame, empty).into_buffer());
    let out = f(&mut img);
    *frame = image::Frame::from_parts(img.into_rgba8(), left, top, delay);
    out
}

/// A copy of a lent frame, for a chain whose first step cannot
/// read the frame where it is (see [`ChainState::run_to`]).
fn copy_of_lent(frame: &DynamicImage) -> DynamicImage {
    #[cfg(test)]
    LENT_FRAME_COPIES.set(LENT_FRAME_COPIES.get() + 1);
    frame.clone()
}

#[cfg(test)]
thread_local! {
    /// How many lent frames [`copy_of_lent`] copied on this
    /// thread, so a test can see which chains read their frames
    /// where they are.
    static LENT_FRAME_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl IntoIterator for Frames {
    type Item = image::Frame;
    type IntoIter =
        std::iter::Chain<std::iter::Once<image::Frame>, std::vec::IntoIter<image::Frame>>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self.first).chain(self.rest)
    }
}

/// Decode all frames of a GIF source, under the shared limits:
/// the frames' declared sizes are charged first (see
/// [`check_gif_frame_budget`]), the decoder gets explicit
/// dimension/allocation limits (its defaults are none), and
/// collection enforces the shared frame and pixel budgets and
/// the request's deadline (see [`collect_frames`]).
fn decode_gif_frames(
    bytes: &[u8],
    deadline: Option<std::time::Instant>,
) -> Result<Frames, EngineError> {
    use image::{AnimationDecoder, ImageDecoder};
    check_gif_frame_budget(bytes)?;
    let mut decoder = image::codecs::gif::GifDecoder::new(Cursor::new(bytes))
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    decoder
        .set_limits(decode_limits())
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    collect_frames(decoder.into_frames(), "GIF", deadline)
}

/// Charge a GIF's frames to the animation budgets from their
/// descriptors, before any frame decodes. The decoder hands
/// back every frame composited onto the logical screen, so
/// [`collect_frames`] sees each at the screen's size. Each
/// frame is decoded at its own declared size first, though,
/// and a descriptor may declare one far larger than the
/// screen (the gif crate leaves frames unchecked against it):
/// a 1x1 screen can carry 64-megapixel frames, each decoded in
/// full and then cut down to one pixel. So each frame counts
/// here at the larger of the screen and its own declared size,
/// against the same frame cap and decoded-pixel budget. A frame
/// that fits the screen counts as it always has. The walk reads
/// block structure only, with LZW decoding skipped, as
/// [`is_animated_gif`] does. A file the walk cannot read to its
/// end is left to the decoder, which stops at the same point
/// with its own error, past only frames counted here.
fn check_gif_frame_budget(bytes: &[u8]) -> Result<(), EngineError> {
    let mut options = gif::DecodeOptions::new();
    options.skip_frame_decoding(true);
    let Ok(mut decoder) = options.read_info(bytes) else {
        return Ok(());
    };
    let screen = u64::from(decoder.width()) * u64::from(decoder.height());
    let mut frames: usize = 0;
    let mut total_pixels: u64 = 0;
    let mut kept_bytes: u64 = 0;
    while let Ok(Some(frame)) = decoder.next_frame_info() {
        frames += 1;
        if frames > MAX_ANIMATED_FRAMES {
            return Err(EngineError::Decode(format!(
                "GIF has more than {MAX_ANIMATED_FRAMES} frames"
            )));
        }
        let declared = u64::from(frame.width) * u64::from(frame.height);
        total_pixels += screen.max(declared);
        if total_pixels > MAX_ANIMATED_PIXELS {
            return Err(EngineError::Decode(format!(
                "GIF frames exceed the {MAX_ANIMATED_PIXELS} decoded-pixel limit"
            )));
        }
        // Each frame comes back composited onto the screen, and the
        // collector charges the bytes it keeps against
        // MAX_DECODE_ALLOC, so the screen's bytes per frame are
        // charged here too, before any frame decodes.
        kept_bytes += screen * 4;
        if kept_bytes > MAX_DECODE_ALLOC {
            return Err(EngineError::Decode(format!(
                "GIF frames exceed the {MAX_DECODE_ALLOC} byte decode allocation limit"
            )));
        }
    }
    Ok(())
}

/// Charge an animated WebP's frames to the animation budgets
/// from its VP8X canvas and its ANMF headers, before any frame
/// decodes, as [`check_gif_frame_budget`] charges a GIF from its
/// descriptors. Every frame composites onto the full canvas, so
/// the frame count times the canvas pixels is the animation's
/// decoded-pixel total, charged against [`MAX_ANIMATED_PIXELS`]
/// as the collector charges the frames it retains; the walk and
/// the image crate's decoder each hold a canvas-sized working
/// buffer and a per-frame copy of it, so the canvas is charged
/// against [`MAX_DECODE_ALLOC`] at four bytes a pixel. The image
/// crate's WebP decoder enforces only the per-axis dimension
/// limit, not the allocation budget, so without this charge an
/// over-budget canvas would be allocated before the collector
/// could refuse it. The canvas and the frame count are read by
/// [`webp_anim::header`], the way image-webp reads them, so the
/// pixels charged here are the pixels it allocates. A container
/// that does not parse this far states no canvas; the decoder
/// then opens it and fails before it allocates a canvas.
fn check_webp_frame_budget(bytes: &[u8]) -> Result<(), EngineError> {
    let Some(header) = webp_anim::header(bytes) else {
        return Ok(());
    };
    // A canvas past the per-axis dimension limit is refused by the
    // image crate's decoder (`set_limits` in `open_webp_animation`)
    // before it allocates, so it is left that error. This charge is
    // for a canvas within the dimension limit, which that decoder
    // admits and would then allocate past the allocation budget it
    // does not enforce.
    if header.width > MAX_DECODE_DIMENSION || header.height > MAX_DECODE_DIMENSION {
        return Ok(());
    }
    let canvas = u64::from(header.width) * u64::from(header.height);
    if canvas.saturating_mul(4) > MAX_DECODE_ALLOC {
        return Err(EngineError::Decode(format!(
            "WebP canvas {}x{} exceeds the {MAX_DECODE_ALLOC} byte decode allocation limit",
            header.width, header.height
        )));
    }
    let total_pixels = u64::from(header.frames).saturating_mul(canvas);
    if total_pixels > MAX_ANIMATED_PIXELS {
        return Err(EngineError::Decode(format!(
            "WebP frames exceed the {MAX_ANIMATED_PIXELS} decoded-pixel limit"
        )));
    }
    // The collector keeps every frame at the canvas's size and
    // charges the bytes it keeps against MAX_DECODE_ALLOC, frame by
    // frame. Charging that total here refuses, before any frame
    // decodes, exactly the animations the collector would refuse
    // partway through.
    if total_pixels.saturating_mul(4) > MAX_DECODE_ALLOC {
        return Err(EngineError::Decode(format!(
            "WebP frames exceed the {MAX_DECODE_ALLOC} byte decode allocation limit"
        )));
    }
    Ok(())
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

/// The formats an animation keeps its frames in (issue #480):
/// an animated source's own format, and the format its frames
/// render to on the animated path. The video formats
/// ([`crate::video::VideoFormat`]) take the frames too, on the
/// video path, and every other format takes the first frame as
/// a still.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnimatedFormat {
    Gif,
    WebP,
}

impl AnimatedFormat {
    /// The animated format `format` names, or `None` for one
    /// that cannot keep frames. Every format is listed, so a
    /// new one has to say which it is.
    fn of(format: ImageFormat) -> Option<Self> {
        match format {
            ImageFormat::Gif => Some(AnimatedFormat::Gif),
            ImageFormat::WebP => Some(AnimatedFormat::WebP),
            ImageFormat::Jpeg
            | ImageFormat::Png
            | ImageFormat::Jxl
            | ImageFormat::Avif
            | ImageFormat::Mp4
            | ImageFormat::WebM => None,
        }
    }
}

impl From<AnimatedFormat> for ImageFormat {
    fn from(format: AnimatedFormat) -> Self {
        match format {
            AnimatedFormat::Gif => ImageFormat::Gif,
            AnimatedFormat::WebP => ImageFormat::WebP,
        }
    }
}

/// Decode all frames of an animated WebP source, under the
/// same limits as GIF. Every frame comes back composited onto
/// the full canvas the way the image crate's decoder
/// (image-webp) composites it, honoring sub-rectangle frames
/// and the blending flag; disposal of a frame to the
/// background color is honored only when the file's ANIM
/// background color has been set on that decoder first (it
/// otherwise leaves the disposed rectangle in place), so the
/// color is read from the container and set before any frame
/// decodes (see [`open_webp_animation`]).
///
/// The frames' bitstreams go through libwebp, which decodes
/// them to the same pixels about twice as fast (issue #646,
/// see [`webp_anim`]). The image crate's iterator still takes,
/// from the start, any file that walk does not handle, so an
/// unusual or broken file gets the frames or the error it
/// always got.
fn decode_webp_frames(
    bytes: &[u8],
    deadline: Option<std::time::Instant>,
) -> Result<Frames, EngineError> {
    use image::AnimationDecoder;
    // Charge the canvas and the frame count from the headers
    // before either decoder opens the file, as the GIF path does
    // (see [`check_webp_frame_budget`]). An over-budget canvas is
    // refused here, never handed to the image crate's iterator.
    check_webp_frame_budget(bytes)?;
    let (decoder, background) = open_webp_animation(bytes)?;
    if let Some(frames) = webp_frames_through_libwebp(bytes, background, deadline) {
        return frames;
    }
    collect_frames(decoder.into_frames(), "WebP", deadline)
}

/// An animated WebP the image crate's decoder has opened, and
/// the ANIM background color set on that decoder, if any.
type OpenedWebp<'a> = (
    image::codecs::webp::WebPDecoder<Cursor<&'a [u8]>>,
    Option<[u8; 4]>,
);

/// Open an animated WebP with the image crate's decoder: the
/// container checks and their errors, the decode limits, and
/// the ANIM background color. Returns the decoder and the
/// color it took, if any.
fn open_webp_animation(bytes: &[u8]) -> Result<OpenedWebp<'_>, EngineError> {
    use image::ImageDecoder;
    let mut decoder = image::codecs::webp::WebPDecoder::new(Cursor::new(bytes))
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    decoder
        .set_limits(decode_limits())
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    let mut background = None;
    if let Some((_, color)) = webp_animation_params(bytes) {
        // Losing the fill on a decoder refusal only costs the
        // disposal behavior; the frames still decode.
        if decoder.set_background_color(image::Rgba(color)).is_ok() {
            background = Some(color);
        }
    }
    Ok((decoder, background))
}

/// The frames of an animated WebP that [`open_webp_animation`]
/// has opened, each bitstream decoded by libwebp and composited
/// by [`webp_anim`], under the limits and the deadline
/// [`collect_frames`] holds every animation to. `None` when the
/// walk met something it does not handle, at any frame: the
/// frames it had are dropped and the image crate's iterator
/// decides.
fn webp_frames_through_libwebp(
    bytes: &[u8],
    background: Option<[u8; 4]>,
    deadline: Option<std::time::Instant>,
) -> Option<Result<Frames, EngineError>> {
    let mut animation = webp_anim::Animation::open(bytes, background)?;
    let mut unsupported = false;
    let frames = std::iter::from_fn(|| match animation.next_frame() {
        Ok(frame) => Some(Ok(frame)),
        Err(webp_anim::Stop::Done) => None,
        Err(webp_anim::Stop::Unsupported) => {
            unsupported = true;
            None
        }
    });
    let collected = collect_frames(frames, "WebP", deadline);
    (!unsupported).then_some(collected)
}

/// Encode frames as an animated WebP through libwebp's
/// animated encoder (the `webp` crate's `AnimEncoder`, method
/// 4 by default; issue #323). The pure-Rust alternative
/// (oxideav-webp) writes lossless VP8L frames only and has no lossy
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
    frames: &Frames,
    loop_count: u16,
    quality: Option<u8>,
    format_options: FormatOptions,
    deadline: Option<std::time::Instant>,
) -> Result<Vec<u8>, EngineError> {
    let (width, height) = frames.dimensions();
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
    // libwebp's animation options (`minimize_size`,
    // `allow_mixed`, `kmin`/`kmax`) stay at their defaults
    // (issue #340). The `webp` crate fills them in inside
    // `try_encode` and exposes none of them, and reaching
    // them needs unsafe FFI, which `#![forbid(unsafe_code)]`
    // rules out. Measured through libwebp-sys,
    // `minimize_size` is the one that matters: 13.0% fewer bytes (75,826
    // against 87,164) and +0.21 dB, for 46% more encode time
    // (1,031 ms against 708 ms). `allow_mixed` left photo
    // content byte-identical, kmax 17 with kmin 9 saved 1.2%,
    // and both flags together took 10.2x the time of
    // `minimize_size` alone for the same bytes. That saving
    // is not worth a new unsafe FFI surface plus a fork of
    // the crate's assemble path, so the defaults stay.
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
        state::check_deadline(deadline)?;
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
                | Filter::Plain(PlainFilter::Border(..))
                | Filter::RemoveBg
        )
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
/// operations produced for the meta response.
#[derive(Debug)]
struct ApplyOutcome {
    img: DynamicImage,
    encode_opts: EncodeOptions,
    /// The studio record (issue #197).
    studio: Option<crate::studio::StudioRecord>,
    /// The region-removal record (issue #176).
    inpaint: Option<InpaintRecord>,
    /// The segmentation record (issue #175).
    segments: Option<SegmentRecord>,
    /// The objects-crop record (issue #318).
    objects_crop: Option<ObjectsCropRecord>,
}

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
///   state decided at run time, so it never fuses. (An
///   aspect crop's own resize reads its window from the
///   frame too, by a route that keeps the two-step bytes:
///   see [`resize_lanczos3_cut`].)
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

    /// A smart crop in an animated render sharing windows
    /// across frames (issue #376): the first frame decides
    /// through the traced crop (the trace reports the very
    /// window the crop used) and later frames replay that
    /// window verbatim, so the analysis runs once and the
    /// framing cannot jitter. `op_index` is the crop's
    /// position in the chain. A replayed window is handed
    /// back uncut, for the resize that closes the crop to
    /// read from the frame (issue #654).
    fn crop(
        &mut self,
        img: DynamicImage,
        (tw, th): (u32, u32),
        op_index: usize,
        align: &Align,
        ctx: &ProcessContext<'_>,
    ) -> crate::crop::Uncut {
        let key = (op_index, img.width(), img.height());
        if let Some(window) = self.get(key) {
            crate::crop::Uncut {
                img,
                window: Some(window),
            }
        } else {
            let (cropped, trace) = crate::crop::crop_to_aspect_traced(
                img,
                tw,
                th,
                align.clone(),
                ctx.face_provider,
                ctx.focal_provider,
            );
            if let Some(trace) = trace {
                self.set(key, trace.window);
            }
            crate::crop::Uncut::whole(cropped)
        }
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
/// of the scene, and the scaled overlay parts they all draw.
#[derive(Debug, Default)]
struct FrameShare {
    /// One smart crop window per crop (issue #376), on the
    /// animated image and video paths alike: the video path
    /// let each frame decide its own until issue #713.
    smart_windows: SmartWindows,
    /// One trim window per trim filter (issue #465).
    trims: TrimWindows,
    /// The parts `overlay(...,width)` scaled, kept for the
    /// frames after the one that scaled them.
    overlays: ScaledOverlays,
}

/// The scaled overlay parts the frames of one animated render
/// share. `overlay(...,width)` scales the part of the overlay
/// that lands on the image, and those pixels depend on the
/// overlay, the size it is scaled to and the part that lands,
/// nothing else. The frames of a render share one canvas, so
/// they all ask for the same part: the first frame scales it,
/// and the frames after it draw the kept pixels, where each
/// scaled them again. A part is kept under all three, so a
/// frame of another size, or another overlay, asks for a part
/// of its own and scales it. The overlay is named by the URL
/// it was fetched from, which names one image for the whole
/// render.
///
/// The frames render in turn, each chain borrowing the render's
/// one share for its run (see [`render_frames`] and
/// [`survey_trims`]), so the parts are never read and written at
/// once. They hold at most as many pixels as the image they land
/// on: a part past that is scaled for each frame, as before, so
/// a chain of many scaled overlays keeps no more than one
/// frame's worth of them.
#[derive(Debug, Default)]
struct ScaledOverlays {
    entries: Vec<ScaledOverlay>,
}

/// One kept part: what its pixels depend on, and the pixels.
#[derive(Debug)]
struct ScaledOverlay {
    url: String,
    size: (u32, u32),
    part: crate::filters::Rect,
    pixels: std::sync::Arc<image::RgbaImage>,
}

impl ScaledOverlays {
    /// The pixels of `part` of the overlay at `url` scaled to
    /// `size`: the kept ones when an earlier frame scaled them,
    /// or else what `scale` makes, kept while the kept parts
    /// still hold no more than `room` pixels with them.
    fn part<E>(
        &mut self,
        url: &str,
        size: (u32, u32),
        part: crate::filters::Rect,
        room: u64,
        scale: impl FnOnce() -> Result<image::RgbaImage, E>,
    ) -> Result<std::sync::Arc<image::RgbaImage>, E> {
        let pixels_of = |(_, _, w, h): crate::filters::Rect| u64::from(w) * u64::from(h);
        if let Some(kept) = self
            .entries
            .iter()
            .find(|e| e.url == url && e.size == size && e.part == part)
        {
            return Ok(std::sync::Arc::clone(&kept.pixels));
        }
        let pixels = std::sync::Arc::new(scale()?);
        let held: u64 = self.entries.iter().map(|e| pixels_of(e.part)).sum();
        if held + pixels_of(part) <= room {
            self.entries.push(ScaledOverlay {
                url: url.to_owned(),
                size,
                part,
                pixels: std::sync::Arc::clone(&pixels),
            });
        }
        Ok(pixels)
    }
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
///
/// The render needs the frames as they came, and a prefix's
/// steps work on the image they are handed (a resize
/// premultiplies its own frame where it sits), so each frame
/// is lent to its prefix and comes back unchanged (issue
/// #737). A prefix that starts with a resize or a fit-in that
/// resamples the whole frame reads the lent frame where it is
/// (see [`ChainState::run_to`]); any other prefix runs on a
/// copy of the frame. A trim nothing runs before reads the
/// frames where they are (issue #713): its survey is the
/// tolerance pass alone.
fn survey_trims(
    frames: &mut Frames,
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
                Filter::Plain(
                    PlainFilter::Trim(tolerance) | PlainFilter::TrimSides(tolerance, _),
                ) => *tolerance,
                _ => continue,
            };
            let key = (op_index, filter_index);
            let first_step = op_index == 0 && filter_index == 0;
            // The first frame's dimensions and the union so far.
            let mut surveyed: Option<((u32, u32), Option<Window>)> = None;
            for frame in frames.iter_mut() {
                ctx.check_deadline()?;
                let (dims, found) = if first_step {
                    let rgba = frame.buffer();
                    (
                        rgba.dimensions(),
                        crate::filters::trim_box_rgba(rgba, tolerance),
                    )
                } else {
                    lend_frame(frame, |lent| {
                        let mut chain =
                            ChainState::new(ops, cap, dpr, ctx, directives, Some(&mut *share));
                        let img = chain.run_to(lent, ops, key)?;
                        Ok::<_, EngineError>((
                            (img.width(), img.height()),
                            crate::filters::trim_box(&img, tolerance),
                        ))
                    })?
                };
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
    img: DynamicImage,
    ops: &[Operation],
    cap: Option<(u32, u32)>,
    dpr: f32,
    ctx: &ProcessContext<'_>,
    directives: &crate::url::RequestDirectives,
    share: Option<&mut FrameShare>,
) -> Result<ApplyOutcome, EngineError> {
    check_operation_count(ops)?;
    ChainState::new(ops, cap, dpr, ctx, directives, share).run(img, ops)
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
    /// What an animated render shares across its frames: its
    /// geometry (issues #376, #465) and its scaled overlay
    /// parts; `None` on a still.
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
    /// How many box-prompted decodes the chain has run, held
    /// to [`MAX_SEGMENT_DECODES`] (issue #713).
    segment_decodes: usize,
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
    /// The image the frame stands for when the chain starts
    /// from a reduced decode of it (issue #653), until the
    /// chain's first operation, a plain resize, has
    /// resampled the frame to its target (see
    /// [`Engine::still_chain`]).
    reduced: Option<jpeg_scaled::Reduced>,
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
/// [`ChainState::class_targets`]): the detections with their
/// segmentation masks, and the detections it paints by their
/// boxes. Without a segment provider every detection is a
/// box; with one, every detection is masked until the
/// request's decodes run out (see [`MAX_SEGMENT_DECODES`]).
struct ClassTargets {
    masked: Vec<crate::filters::MaskedDetection>,
    boxes: Vec<crate::ai::Detection>,
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
            segment_decodes: 0,
            objects_crop_record: None,
            fusion: fusion_plan(ops),
            fused_window: None,
            reduced: None,
        }
    }

    /// Run the whole chain on `img`: every operation in
    /// order, then what the finished chain owes its caller.
    fn run(
        mut self,
        mut img: DynamicImage,
        ops: &[Operation],
    ) -> Result<ApplyOutcome, EngineError> {
        for (op_index, op) in ops.iter().enumerate() {
            img = self.step(img, op_index, op)?;
        }
        let hinted = self.directives.dpr.is_none() && self.dpr > 1.0 && scales_with_dpr(ops);
        let outcome = self.finish(img)?;
        if hinted {
            check_hinted_output(outcome.img.width(), outcome.img.height())?;
        }
        Ok(outcome)
    }

    /// Run one operation of the chain, with what every
    /// operation owes the chain around it: the deadline, the
    /// detection memos, and the output budget.
    fn step(
        &mut self,
        img: DynamicImage,
        op_index: usize,
        op: &Operation,
    ) -> Result<DynamicImage, EngineError> {
        self.ctx.check_deadline()?;
        let img = self.apply_op(img, op_index, op)?;
        self.stepped(img, op)
    }

    /// What an operation owes the chain once it has run (see
    /// [`ChainState::step`]): the detection memos and the
    /// output budget.
    fn stepped(&mut self, img: DynamicImage, op: &Operation) -> Result<DynamicImage, EngineError> {
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

    /// Run the chain up to the filter at `key` (the operation
    /// and filter indexes) on a frame lent for the run, and
    /// return the image that filter receives: the trim
    /// survey's view of a frame (see [`survey_trims`]). The
    /// frame comes back as it was. A first operation that
    /// resamples the whole frame reads it where it is (see
    /// [`ChainState::lent_step`]); after any other, the chain
    /// runs on a copy of the frame (issue #737).
    fn run_to(
        &mut self,
        frame: &mut DynamicImage,
        ops: &[Operation],
        (op_index, filter_index): (usize, usize),
    ) -> Result<DynamicImage, EngineError> {
        let lent = match ops.first() {
            Some(first) if op_index > 0 => self.lent_step(frame, first)?,
            _ => None,
        };
        let (mut img, next) = match lent {
            Some(img) => (img, 1),
            None => (copy_of_lent(frame), 0),
        };
        for (index, op) in ops.iter().enumerate().take(op_index).skip(next) {
            img = self.step(img, index, op)?;
        }
        if let Some(Operation::Filters(filters)) = ops.get(op_index) {
            for (index, filter) in filters.iter().enumerate().take(filter_index) {
                img = self.filter_step(img, (op_index, index), filter)?;
            }
        }
        Ok(img)
    }

    /// The chain's first operation, `op`, on a frame lent for
    /// it (issue #737): a resize or a fit-in that resamples the
    /// whole frame reads it where it is (see
    /// [`resize_lanczos3_lent`]) and returns the image it
    /// makes, with what [`ChainState::step`] owes the chain
    /// around it. `None` for any other operation, and for a
    /// resize that leaves the frame at its size, which hands
    /// on the frame itself: those run on a copy.
    fn lent_step(
        &mut self,
        frame: &mut DynamicImage,
        op: &Operation,
    ) -> Result<Option<DynamicImage>, EngineError> {
        self.ctx.check_deadline()?;
        let img = match op {
            Operation::Size(size) if self.fused_window.is_none() && self.reduced.is_none() => {
                let cap = self.cap.map(|(cw, ch)| SizeCap::Axes(cw, ch));
                let Some(img) = apply_size_lent(frame, size, cap, self.dpr)? else {
                    return Ok(None);
                };
                self.focal_resized(&img, size.flip_horizontal, size.flip_vertical);
                img
            }
            Operation::FitIn { size, adaptive } => {
                let Some(img) = apply_fit_in_lent(frame, size, *adaptive, self.cap, self.dpr)?
                else {
                    return Ok(None);
                };
                self.focal_resized(&img, false, false);
                img
            }
            _ => return Ok(None),
        };
        self.stepped(img, op).map(Some)
    }

    /// Carry the pending focal form through a resize of the
    /// whole image to `img`, mirrored on the axes it flipped.
    fn focal_resized(&mut self, img: &DynamicImage, flip_h: bool, flip_v: bool) {
        let dims = (img.width(), img.height());
        self.focal = self.focal.map(|f| f.scaled(dims, flip_h, flip_v));
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
                // A pending fused window (issue #287) is the
                // plan's crop, waiting for this resize: the plan
                // admits one resize per chain, after the crop.
                img = if let Some(window) = self.fused_window.take() {
                    apply_size_cropped(
                        img,
                        window,
                        size,
                        self.cap.map(|(cw, ch)| SizeCap::Axes(cw, ch)),
                        self.dpr,
                    )?
                } else if let Some(reduced) = self.reduced.take() {
                    // The frame is a reduced decode (issue
                    // #653): the size resolves against the
                    // full image, as it would have on it.
                    apply_size_windowed(
                        img,
                        size,
                        self.cap.map(|(cw, ch)| SizeCap::Axes(cw, ch)),
                        self.dpr,
                        SourceWindow::Reduced(reduced),
                    )?
                } else {
                    apply_size(
                        img,
                        size,
                        self.cap.map(|(cw, ch)| SizeCap::Axes(cw, ch)),
                        self.dpr,
                    )?
                };
                self.focal_resized(&img, size.flip_horizontal, size.flip_vertical);
            }
            Operation::FitIn { size, adaptive } => {
                img = apply_fit_in(img, size, *adaptive, self.cap, self.dpr)?;
                self.focal_resized(&img, false, false);
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
                let (padded, placed) = apply_pad(
                    img,
                    tw,
                    th,
                    *fill,
                    self.directives.fill,
                    self.cap,
                    self.ctx.deadline,
                )?;
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
            Filter::Plain(PlainFilter::Trim(tolerance)) => self.trim(img, key, *tolerance, [0; 4]),
            Filter::Plain(PlainFilter::TrimSides(tolerance, sides)) => {
                self.trim(img, key, *tolerance, *sides)
            }
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
        // Every arm hands back the frame and the window it
        // decided on. The traced arms, which the debug view
        // and an animation's first frame take, hand back the
        // window already cut.
        let uncut = if let Some(Focal::Rect(rect)) = focal {
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
            crate::crop::Uncut::whole(cropped)
        } else if *align == Align::Smart
            && let Some(windows) = self.share.as_deref_mut().map(|s| &mut s.smart_windows)
        {
            windows.crop(img, (tw, th), op_index, align, self.ctx)
        } else {
            crate::crop::crop_to_aspect_uncut(
                img,
                tw,
                th,
                align.clone(),
                self.ctx.face_provider,
                self.ctx.focal_provider,
            )
        };
        // The resize that closes the crop reads the window
        // straight from the frame (issue #654). It sizes
        // and checks against the window's dimensions, as it
        // did against the cut window's.
        apply_size_windowed(
            uncut.img,
            &ImageSize {
                width: Dimension::Pixels(tw),
                height: Dimension::Pixels(th),
                flip_horizontal: false,
                flip_vertical: false,
            },
            self.cap.map(|(cw, ch)| SizeCap::Fit(cw, ch)),
            1.0,
            match uncut.window {
                Some(window) => SourceWindow::Cut(window),
                None => SourceWindow::Whole,
            },
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
    ) -> crate::crop::Uncut {
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
            crate::crop::Uncut::whole(cropped)
        } else {
            crate::crop::crop_to_aspect_focal_uncut(img, tw, th, rect).0
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
    ) -> crate::crop::Uncut {
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
            crate::crop::Uncut::whole(cropped)
        } else {
            crate::crop::crop_to_aspect_point_uncut(img, tw, th, pt).0
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
    ) -> Result<crate::crop::Uncut, EngineError> {
        let faces = match detect_faces(&mut self.face_memo, &img, self.ctx) {
            Ok(faces) => faces,
            Err(EngineError::Inference(_)) => Vec::new(),
            Err(e) => return Err(e),
        };
        let largest = faces
            .iter()
            .max_by_key(|f| f.w as u64 * f.h as u64)
            .copied();
        let cropped = match largest {
            Some(face) => {
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
                    crate::crop::Uncut::whole(cropped)
                } else {
                    crate::crop::crop_around_face_uncut(
                        img,
                        tw,
                        th,
                        &face,
                        self.directives.face_padding,
                    )
                    .0
                }
            }
            None => {
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
                    crate::crop::Uncut::whole(cropped)
                } else {
                    crate::crop::crop_to_aspect_uncut(
                        img,
                        tw,
                        th,
                        Align::Smart,
                        None,
                        self.ctx.focal_provider,
                    )
                }
            }
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
    ) -> Result<crate::crop::Uncut, EngineError> {
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
                    crate::crop::Uncut::whole(cropped)
                } else {
                    crate::crop::crop_around_objects_uncut(img, tw, th, union, scale).0
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
                    crate::crop::Uncut::whole(cropped)
                } else {
                    crate::crop::crop_to_aspect_uncut(img, tw, th, Align::Center, None, None)
                }
            }
        };
        Ok(cropped)
    }

    /// Apply one filter of a `filters:` segment: a plain
    /// filter runs through `crate::filters` (see
    /// [`ChainState::plain_filter`]), watermark and overlay
    /// filters resolve against the server-fetched images, the
    /// model and font filters run through their providers,
    /// and the region filters check their rectangle against
    /// the image in effect. The match names every variant
    /// (issue #741), so a new filter does not compile until
    /// it is applied here or is a [`PlainFilter`].
    fn apply_filter(
        &mut self,
        mut img: DynamicImage,
        filter: &Filter,
    ) -> Result<DynamicImage, EngineError> {
        match filter {
            Filter::Plain(plain) => img = self.plain_filter(img, plain)?,
            Filter::Watermark { url, x, y, alpha } => {
                let wm = self
                    .ctx
                    .watermarks
                    .and_then(|m| m.get(url))
                    .ok_or_else(|| {
                        EngineError::Unsupported(format!("watermark was not resolved: {url}"))
                    })?;
                img = overlay_watermark(self.ctx, &img, wm, *x, *y, *alpha)?;
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
                // The frames of an animated render share the
                // scaled part (see `ScaledOverlays`).
                let kept = self
                    .share
                    .as_deref_mut()
                    .map(|share| (url.as_str(), &mut share.overlays));
                img = blend_overlay_keeping(
                    self.ctx,
                    &img,
                    ov,
                    (*x, *y),
                    *alpha,
                    *blend,
                    *width,
                    kept,
                )?;
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
                img = crate::filters::blur_faces(&img, &faces, self.ctx.deadline)?;
            }
            Filter::PixelateFaces => {
                let faces = detect_faces(&mut self.face_memo, &img, self.ctx)?;
                img = crate::filters::pixelate_faces(&img, &faces);
            }
            Filter::BlurClass(class) => {
                let deadline = self.ctx.deadline;
                let ClassTargets { masked, boxes } = self.class_targets(&img, class)?;
                if !masked.is_empty() {
                    img = crate::filters::blur_detections_masked(&img, &masked, deadline)?;
                }
                if !boxes.is_empty() {
                    img = crate::filters::blur_detections(&img, &boxes, deadline)?;
                }
            }
            Filter::PixelateClass(class) => {
                let ClassTargets { masked, boxes } = self.class_targets(&img, class)?;
                if !masked.is_empty() {
                    img = crate::filters::pixelate_detections_masked(&img, &masked);
                }
                if !boxes.is_empty() {
                    img = crate::filters::pixelate_detections(&img, &boxes);
                }
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
                let ClassTargets { masked, boxes } = self.class_targets(&img, class)?;
                if !masked.is_empty() {
                    img = crate::filters::recolor_detections_masked(&img, &masked, *color);
                }
                if !boxes.is_empty() {
                    img = crate::filters::recolor_detections(&img, &boxes, *color);
                }
            }
            // Request-wide directives, read from the request
            // as a whole: the face crop reads the padding, and
            // the DPR factor scales the size targets before
            // the chain runs. No pixels here.
            Filter::FacePad(_) | Filter::Dpr(_) => {}
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
                img = crate::filters::blur_region(&img, *x, *y, *w, *h, self.ctx.deadline)?;
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
                // The renderer checks the deadline between
                // glyphs (issue #606).
                img = renderer
                    .draw(&img, text, *size, *color, *anchor, self.ctx.deadline)
                    .map_err(|e| match e {
                        crate::text::DrawError::Timeout => EngineError::Timeout,
                        crate::text::DrawError::Failed(message) => EngineError::Render(message),
                    })?;
            }
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
            // The shadow's blur checks the deadline before
            // each row (issue #614).
            crate::url::StudioMode::Product => crate::studio::composite_within(
                img,
                &mask,
                confidence,
                spec,
                self.ctx.brand_color,
                self.ctx.deadline,
            )?,
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
    ///
    /// A request runs at most [`MAX_SEGMENT_DECODES`] decodes
    /// across its class filters (issue #713), to the
    /// detections in the order the detector gives them, best
    /// first. Past that, the rest are painted by their boxes,
    /// as without a segment provider: a box holds the whole
    /// object, so nothing is left unredacted. The deadline is
    /// checked before each decode.
    fn class_targets(
        &mut self,
        img: &DynamicImage,
        class: &str,
    ) -> Result<ClassTargets, EngineError> {
        let mut boxes = detect_for_class(
            &mut self.detection_memos,
            &self.detect_providers,
            img,
            class,
        )?;
        let Some(provider) = self.ctx.segment_provider else {
            return Ok(ClassTargets {
                masked: Vec::new(),
                boxes,
            });
        };
        let room = MAX_SEGMENT_DECODES.saturating_sub(self.segment_decodes);
        let rest = boxes.split_off(room.min(boxes.len()));
        let ctx = self.ctx;
        let (masked, mut objects) =
            segment_for_class(&mut self.segment_memo, provider, img, &boxes, || {
                ctx.check_deadline()
            })?;
        self.segment_decodes += masked.len();
        self.segment_objects.append(&mut objects);
        Ok(ClassTargets {
            masked,
            boxes: rest,
        })
    }

    /// A plain filter: pixels in, pixels out through
    /// `apply_filters_within`, with any encoding directive it
    /// carries absorbed into the chain's encode options.
    fn plain_filter(
        &mut self,
        img: DynamicImage,
        filter: &PlainFilter,
    ) -> Result<DynamicImage, EngineError> {
        // Arbitrary-angle rotation is
        // the one plain filter that grows
        // the canvas: check the canvas it
        // is about to allocate before it
        // allocates (issue #182, R2-12),
        // per axis too (issue #521).
        if let PlainFilter::Rotate(degrees) = filter {
            let (cw, ch) = crate::filters::rotated_canvas(img.width(), img.height(), *degrees);
            check_output_size(cw, ch, (img.width(), img.height()))?;
        }
        // A blur checks the deadline before each row (issue
        // #614).
        let (new_img, options) = crate::filters::apply_filters_within(
            img,
            std::slice::from_ref(filter),
            self.directives.fill,
            self.ctx.deadline,
        )?;
        self.encode_opts.absorb(options);
        Ok(new_img)
    }

    /// Close the chain: render the debug view when
    /// `debug:focal` armed it, check the final image against
    /// the per-axis limit and the output budget, and hand
    /// back the image with the encode options and the
    /// records.
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
        // budget (issue #182, audit R2-12) and the per-axis
        // limit (issue #528). The still, animated and video
        // paths all end here, once per frame.
        check_final_output(img.width(), img.height())?;
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
        Ok(ApplyOutcome {
            img,
            encode_opts: self.encode_opts,
            studio: self.studio_record,
            inpaint: inpaint_record,
            segments: segment_record,
            objects_crop: self.objects_crop_record,
        })
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

mod encode;
mod jpeg_scaled;
mod overlay_scale;
mod webp_anim;
pub use encode::AvifTiles;
pub(in crate::engine) use encode::{
    EncodeBuffers, StillFormat, encode, gif_frame_quantized, max_quality_attempts, next_quality,
};
/// Names the test module uses from encode: re-exported
/// only for test builds, where they are used.
#[cfg(test)]
pub(in crate::engine) use encode::{encode_png_indexed, jxl_config};

#[cfg(test)]
mod tests;

// Fast paths against the code they replaced, byte for byte.
#[cfg(test)]
mod identity_proofs;

// Small frames with an alpha channel through every encoder.
#[cfg(test)]
mod small_frame_tests;

// Animated GIF frames prepared on several threads.
#[cfg(test)]
mod gif_frame_tests;

// The largest output whose `quality(auto)` pick is verified.
#[cfg(test)]
mod quality_bound_tests;

// Layout and depth through crops and composites, the quality
// reference, the filter bounds, shared video windows and the
// segmentation cap (issue #713).
#[cfg(test)]
mod fidelity_tests;

#[cfg(test)]
pub(crate) mod test_formats;
