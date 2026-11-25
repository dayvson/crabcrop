//! The image engine: decode, transform, and encode.
//!
//! [`Engine::process`] takes source bytes and the parsed [`Operation`]s,
//! and returns the encoded output plus its format.

#[cfg(test)]
use std::io::Cursor;

use image::DynamicImage;

use crate::filters::EncodeOptions;
use crate::quality::auto_quality;
#[cfg(test)]
use crate::url::{Align, Dimension, Filter, FormatOptions, PadFill, PlainFilter};
use crate::url::{ImageFormat, ImageSize, Operation, ResolvedQuality};

// The operation cap lives with the grammar (issue #469);
// the engine's tests reach it from here through `super::*`.
#[cfg(test)]
use crate::url::MAX_OPERATIONS;

mod limits;
#[cfg(test)]
pub(in crate::engine) use limits::{
    MAX_ANIMATED_FRAMES, OutputBudget, check_final_output, check_hinted_output,
    check_output_pixels, check_output_size, hint_step_down,
};
pub(crate) use limits::{
    MAX_DECODE_ALLOC, MAX_DECODE_DIMENSION, MAX_OUTPUT_PIXELS, dpr_step_below, scales_with_dpr,
};
pub(in crate::engine) use limits::{check_operation_count, dpr_factor, step_down_hint};
mod decode;
#[cfg(test)]
pub(in crate::engine) use decode::{
    MAX_WATERMARK_BYTES, decode_jxl, decode_limits, decode_watermarks_within, decode_webp_direct,
    jxl_dimensions, jxl_sample_limit,
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
    check_jxl_dimensions, decode_source_image, detect_format, ensure_jxl_decoder, is_heic, is_jxl,
    jpeg_coefficient_bytes, zune_jpeg_options,
};
mod resize;
#[cfg(test)]
pub(in crate::engine) use resize::{
    CONVOLVE_REFUSES, ChangingRows, PREMULTIPLIED, SourceWindow, apply_fit_in, apply_pad,
    apply_size, apply_size_windowed, fit_in_box, kernel_span, place_on, resize_lanczos3,
    resize_lanczos3_cropped, resize_lanczos3_cropped_owned, resize_lanczos3_cut,
    resize_lanczos3_lent, resize_lanczos3_owned, view_holds_no_more,
};
pub(in crate::engine) use resize::{SizeCap, resize_target, scale_px};
pub(crate) use resize::{cap_dimensions, has_zero_side};
mod overlay;
#[cfg(test)]
pub(in crate::engine) use overlay::{
    blend_channel, blend_over, blend_overlay, composite, landing, lands_on, overlay_watermark,
};
pub(in crate::engine) use overlay::{flatten, flattened_for};
mod detect;
#[cfg(test)]
pub(in crate::engine) use detect::MAX_SEGMENT_DECODES;
pub(in crate::engine) use detect::{detection_providers, resolve_detector};
mod focal;
#[cfg(test)]
pub(in crate::engine) use focal::{Focal, PendingFocal, capture_from_trace};
mod animated;
#[cfg(any(test, doc))]
pub(in crate::engine) use animated::check_webp_frame_budget;
pub(in crate::engine) use animated::{
    AnimatedFormat, FrameShare, Frames, decode_gif_frames, decode_webp_frames, encode_animated_gif,
    encode_animated_webp, first_frame_placeholder, render_frames, webp_animation_params,
};
#[cfg(test)]
pub(in crate::engine) use animated::{
    GIF_FRAME_THREADS, LENT_FRAME_COPIES, check_gif_frame_budget, collect_frames,
    encode_animated_gif_on, gif_frame_workers, in_index_order, open_webp_animation, survey_trims,
    webp_frames_through_libwebp,
};
pub(crate) use animated::{animated_output_format, is_animated_gif, is_animated_webp};
mod chain;
pub(in crate::engine) use chain::{ApplyOutcome, ChainState};
#[cfg(test)]
pub(in crate::engine) use chain::{apply_operations, fusion_plan};

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
            jxl_threads: None,
            deadline: None,
        }
    }
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
        let mut source = Self::sniff_source(bytes)?;
        if pdf.is_some() {
            // A rendered PDF page re-enters the pipeline as a
            // raster whose source format is PNG, whatever
            // container the renderer wrote (issue #786): the
            // stock renderer writes TIFF now, but the page's
            // default output format and its source-format
            // record stay PNG, the contract from when every
            // renderer wrote PNG. The decode below still sniffs
            // the real bytes, so the TIFF decodes as a TIFF.
            //
            // The frame is byte-identical to the PNG's because
            // the stock pdftocairo attaches no ICC profile or
            // metadata to either raster, and
            // `color::extract_source_meta` reads ICC/EXIF/XMP
            // only from JPEG and PNG, so a TIFF page gets no
            // source metadata and its pixels are taken as sRGB.
            // Were a future renderer to attach an ICC profile to
            // the PNG, the PNG path would color-convert while
            // the TIFF path (no ICC read) would not; see the
            // note on `crate::pdf::render_format_for`.
            source.detected = ImageFormat::Png;
        }
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
        // A 16-bit source resampled at 16 bits costs four to five
        // times an 8-bit resample (issue #814). When every output
        // this request can produce is 8-bit, the extra bits never
        // reach the response, so the frame is narrowed to 8 bits
        // before the chain resamples it. A resized render's pixels
        // then round at 8 bits instead of 16; a 16-bit output
        // (PNG) keeps its bits, and a JPEG reduced decode, already
        // 8-bit, is untouched.
        let img = narrow_for_eight_bit_output(img, resolved_output);
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
    /// the tool; and the rendered raster replaces the bytes,
    /// so the validation preflights and the decode chain all
    /// see the page (its source format is pinned to PNG in
    /// [`Self::process_selected`], issue #786). `None` when
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
        Ok(Some((rendered.raster, pdf_meta)))
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
                ctx.jxl_threads,
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

/// Narrow a 16-bit frame to 8 bits before the chain resamples
/// it, when `output` is a format this pipeline only ever writes
/// at 8 bits (issue #814). The frame keeps its channels: a
/// 16-bit gray frame narrows to 8-bit gray, a 16-bit frame with
/// alpha keeps its alpha. A float frame is left as it is (it is
/// not the 16-bit integer case the slow kernels cost most on),
/// and so is a frame for a 16-bit output. The narrowing is the
/// same high-byte conversion the encoder would run on the
/// resampled frame, so the only change is that the resample now
/// rounds at 8 bits: a resized render's pixels move by at most
/// one level, and a render that resamples nothing is byte for
/// byte as it was.
fn narrow_for_eight_bit_output(img: DynamicImage, output: ImageFormat) -> DynamicImage {
    if !output_is_eight_bit(output) {
        return img;
    }
    match img {
        DynamicImage::ImageLuma16(_) => DynamicImage::ImageLuma8(img.to_luma8()),
        DynamicImage::ImageLumaA16(_) => DynamicImage::ImageLumaA8(img.to_luma_alpha8()),
        DynamicImage::ImageRgb16(_) => DynamicImage::ImageRgb8(img.to_rgb8()),
        DynamicImage::ImageRgba16(_) => DynamicImage::ImageRgba8(img.to_rgba8()),
        other => other,
    }
}

/// Whether `output` is a format this pipeline writes only at 8
/// bits per channel, so narrowing a 16-bit source to 8 bits
/// before resampling loses no bit the response would carry
/// (issue #814). PNG is the one still output that keeps 16-bit
/// samples (see `encode::png_color_params`); JPEG, WebP, GIF,
/// AVIF and JPEG XL are 8-bit here. Any other format (a video
/// format, say) is treated as not 8-bit, so nothing is narrowed
/// for it.
fn output_is_eight_bit(output: ImageFormat) -> bool {
    matches!(
        output,
        ImageFormat::Jpeg
            | ImageFormat::WebP
            | ImageFormat::Gif
            | ImageFormat::Avif
            | ImageFormat::Jxl
    )
}

/// A rectangle of an image as (left, top, width, height).
type Window = (u32, u32, u32, u32);

mod encode;
mod jpeg_scaled;
mod overlay_scale;
pub(crate) use overlay_scale::scaled_part;
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

// The byte budget a step's output frame is held to, in the
// layout the step allocates (issue #784).
#[cfg(test)]
mod output_bytes_tests;

#[cfg(test)]
pub(crate) mod test_formats;
