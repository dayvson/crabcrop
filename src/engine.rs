//! The image engine: decode, transform, and encode.
//!
//! [`Engine::process`] takes source bytes and the parsed [`Operation`]s,
//! and returns the encoded output plus its format.

use std::io::Cursor;
use std::sync::Once;

use image::{DynamicImage, ExtendedColorType, ImageEncoder};

use crate::filters::{EncodeOptions, WatermarkImages, apply_filters};
use crate::quality::auto_quality;
use crate::url::{Dimension, Filter, ImageFormat, ImageSize, Operation, Quality};

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

/// The encoded output plus its container format.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// An AI model failed at inference time. Carries the cause.
    Inference(String),
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
            EngineError::Inference(cause) => {
                write!(f, "AI inference failed: {cause}")
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
#[derive(Default, Clone)]
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
    /// Text renderer, when a font is configured.
    pub text_renderer: Option<&'a dyn crate::text::TextRenderer>,
    /// DPR fallback from the request's client hint, used when
    /// the URL carries no explicit `dpr` directive.
    pub dpr: Option<f32>,
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
        // HEIC/HEIF sources are detected explicitly so the failure
        // names the real problem instead of a generic decode error.
        // The decoder evaluation (issue #24) found no option that
        // is both license-compatible with this MIT crate and
        // buildable in the project's environments today; this
        // check is the seam a future decoder plugs into.
        if is_heic(bytes) {
            return Err(EngineError::Decode(
                "HEIC/HEIF source: no HEIC decoder in this build".to_string(),
            ));
        }
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
        if crate::url::has_removebg(ops) {
            let resolved = crate::url::explicit_format(ops)
                .or(ctx.preferred)
                .unwrap_or(detected);
            if matches!(resolved, ImageFormat::Jpeg | ImageFormat::Gif) {
                return Err(EngineError::Unsupported(
                    "removebg needs an alpha-capable output format (png, webp, jxl, avif)"
                        .to_string(),
                ));
            }
            if ctx.mask_provider.is_none() {
                return Err(EngineError::Unsupported(
                    "removebg is not configured on this server".to_string(),
                ));
            }
        }
        if crate::url::has_upscale(ops) && ctx.upscale_provider.is_none() {
            return Err(EngineError::Unsupported(
                "upscale is not configured on this server".to_string(),
            ));
        }
        if crate::url::has_text(ops) && ctx.text_renderer.is_none() {
            return Err(EngineError::Unsupported(
                "text is not configured on this server (no font)".to_string(),
            ));
        }
        let img = if detected == ImageFormat::Gif {
            // GIF sources decode frame by frame: an animated source
            // whose output stays GIF keeps its frames (each runs the
            // operations, delays preserved); anything else continues
            // with the first frame, which is also what converting an
            // animated GIF to a still format means.
            let frames = decode_gif_frames(bytes)?;
            if frames.len() > 1
                && crate::url::explicit_format(ops)
                    .or(ctx.preferred)
                    .unwrap_or(detected)
                    == ImageFormat::Gif
            {
                return self.process_animated_gif(frames, ops, &ctx);
            }
            DynamicImage::ImageRgba8(
                frames
                    .into_iter()
                    .next()
                    .expect("decode_gif_frames returns at least one frame")
                    .into_buffer(),
            )
        } else if let Some(prep) = &prepared_svg {
            // Rasterize at the size the request is heading for
            // (or the intrinsic size when it asks for none), so
            // vectors stay crisp; the normal pipeline runs on
            // the raster from here.
            let (tw, th) = svg_raster_target(ops, prep.intrinsic(), dpr_factor(ops, &ctx));
            prep.render(tw, th).map_err(EngineError::Decode)?
        } else {
            let mut img =
                image::load_from_memory(bytes).map_err(|e| EngineError::Decode(e.to_string()))?;
            // Phone photos store rotation in the EXIF orientation
            // tag instead of the pixels. Honor it before any
            // operation, so sizes and crops see the image as the
            // viewer would. (GIFs carry no EXIF, hence the split.)
            if let Some(orientation) = exif_orientation(bytes) {
                img = apply_orientation(img, orientation);
            }
            img
        };
        let mut format = detected;
        let (source_width, source_height) = (img.width(), img.height());
        // `no_upscale` is a request-wide directive, like the encoding
        // directives: wherever it appears, it caps every resize in
        // the request at the (oriented) source dimensions.
        let cap = if crate::url::has_no_upscale(ops) {
            Some((img.width(), img.height()))
        } else {
            None
        };
        let (img, directives) = apply_operations(img, ops, cap, dpr_factor(ops, &ctx), &ctx)?;
        // Placeholders describe the final pixels, so they are
        // computed here, before encoding.
        let placeholder = if ctx.placeholders {
            Some(crate::placeholder::compute(&img))
        } else {
            None
        };
        if let Some(f) = directives.format {
            format = f;
        } else if let Some(p) = ctx.preferred {
            format = p;
        }
        // Resolve the quality directive against the final image.
        // `quality(auto)` is reported back: it is the one the caller
        // did not choose themselves.
        let (encode_quality, mut chosen_quality) = match directives.quality {
            Some(Quality::Value(q)) => (Some(q), None),
            Some(Quality::Auto) => {
                let q = auto_quality(&img);
                (Some(q), Some(q))
            }
            None => (None, None),
        };
        let mut bytes = encode(&img, format, encode_quality, directives.fill)?;
        // `max_bytes` is a budget, pursued by re-encoding at
        // decreasing quality. Only the quality-bearing formats have
        // a lever; for the others the directive cannot be honored
        // and the single encode stands. If quality bottoms out over
        // budget, the smallest attempt is returned: an image that
        // misses its budget beats an error. The ladder is bounded
        // per format so a budget cannot multiply an expensive
        // encode without limit.
        let quality_bearing = matches!(
            format,
            ImageFormat::Jpeg | ImageFormat::Jxl | ImageFormat::Avif | ImageFormat::WebP
        );
        if let Some(budget) = directives.max_bytes.filter(|_| quality_bearing) {
            let mut current = encode_quality.unwrap_or(80);
            let mut attempts = 0;
            while bytes.len() as u64 > budget
                && current > 1
                && attempts < max_quality_attempts(format)
            {
                current = next_quality(format, current);
                bytes = encode(&img, format, Some(current), directives.fill)?;
                chosen_quality = Some(current);
                attempts += 1;
            }
        }
        Ok(ProcessedImage {
            bytes,
            format,
            quality: chosen_quality,
            width: img.width(),
            height: img.height(),
            source_width,
            source_height,
            placeholder,
        })
    }

    /// Process an animated GIF frame by frame: every frame runs
    /// the same operations and keeps its delay, so the output
    /// stays animated. Encoding directives do not apply (GIF has
    /// no quality lever and keeps its alpha).
    fn process_animated_gif(
        &self,
        frames: Vec<image::Frame>,
        ops: &[Operation],
        ctx: &ProcessContext<'_>,
    ) -> Result<ProcessedImage, EngineError> {
        let (source_width, source_height) = {
            let b = frames[0].buffer();
            (b.width(), b.height())
        };
        let first = frames[0].buffer();
        let cap = if crate::url::has_no_upscale(ops) {
            Some((first.width(), first.height()))
        } else {
            None
        };
        let dpr = dpr_factor(ops, ctx);
        let mut out_frames = Vec::with_capacity(frames.len());
        for frame in frames {
            let delay = frame.delay();
            let img = DynamicImage::ImageRgba8(frame.into_buffer());
            let (img, _) = apply_operations(img, ops, cap, dpr, ctx)?;
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
        let mut out = Vec::new();
        image::codecs::gif::GifEncoder::new(&mut out)
            .encode_frames(out_frames)
            .map_err(|e| EngineError::Encode(e.to_string()))?;
        Ok(ProcessedImage {
            bytes: out,
            format: ImageFormat::Gif,
            quality: None,
            width,
            height,
            source_width,
            source_height,
            placeholder,
        })
    }
}

/// Decode all frames of a GIF source. Errors when the GIF is
/// malformed or carries no frames.
fn decode_gif_frames(bytes: &[u8]) -> Result<Vec<image::Frame>, EngineError> {
    use image::AnimationDecoder;
    let decoder = image::codecs::gif::GifDecoder::new(Cursor::new(bytes))
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    let frames = decoder
        .into_frames()
        .collect_frames()
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    if frames.is_empty() {
        return Err(EngineError::Decode("GIF has no frames".to_string()));
    }
    Ok(frames)
}

/// The largest width or height an AI upscale may produce.
/// Super-resolution multiplies activations by the factor
/// squared; this keeps the result inside a sane memory envelope.
const MAX_UPSCALE_DIM: u64 = 4096;

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
    let (target_w, target_h) = (tw as u32, th as u32);
    let mut img = img;
    let mut scale = 1u32;
    while scale < factor {
        let before = (img.width(), img.height());
        img = provider.upscale(&img).map_err(EngineError::Inference)?;
        if img.width() <= before.0 || img.height() <= before.1 {
            return Err(EngineError::Inference(
                "the upscale model did not enlarge the image".to_string(),
            ));
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
/// the animated-GIF path, which runs it once per frame.
/// The effective DPR factor: an explicit `dpr(...)` directive
/// wins, then the client hint carried on the context, else 1.
fn dpr_factor(ops: &[Operation], ctx: &ProcessContext<'_>) -> f32 {
    crate::url::explicit_dpr(ops).or(ctx.dpr).unwrap_or(1.0)
}

fn apply_operations(
    mut img: DynamicImage,
    ops: &[Operation],
    cap: Option<(u32, u32)>,
    dpr: f32,
    ctx: &ProcessContext<'_>,
) -> Result<(DynamicImage, EncodeOptions), EngineError> {
    let mut directives = EncodeOptions::default();
    for op in ops {
        match op {
            Operation::Size(size) => img = apply_size(img, size, cap, dpr),
            Operation::FitIn(size) => img = apply_fit_in(img, size, cap, dpr),
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
                img = crate::crop::crop_to_aspect_with(img, tw, th, *align, ctx.face_provider);
                img = apply_size(
                    img,
                    &ImageSize {
                        width: Dimension::Pixels(tw),
                        height: Dimension::Pixels(th),
                        flip_horizontal: false,
                        flip_vertical: false,
                    },
                    cap,
                    1.0,
                );
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
                        Filter::RemoveBg => {
                            let provider = ctx.mask_provider.ok_or_else(|| {
                                EngineError::Unsupported(
                                    "removebg is not configured on this server".to_string(),
                                )
                            })?;
                            let mask = provider.mask(&img).map_err(EngineError::Inference)?;
                            img = crate::ai::apply_mask(&img, &mask);
                        }
                        Filter::Upscale(factor) => {
                            let provider = ctx.upscale_provider.ok_or_else(|| {
                                EngineError::Unsupported(
                                    "upscale is not configured on this server".to_string(),
                                )
                            })?;
                            img = upscale_image(img, *factor, provider)?;
                        }
                        Filter::Text {
                            text,
                            size,
                            color,
                            anchor,
                        } => {
                            let renderer = ctx.text_renderer.ok_or_else(|| {
                                EngineError::Unsupported(
                                    "text is not configured on this server (no font)".to_string(),
                                )
                            })?;
                            img = renderer
                                .draw(&img, text, *size, *color, *anchor)
                                .map_err(EngineError::Render)?;
                        }
                        other => {
                            let (new_img, options) =
                                apply_filters(img, std::slice::from_ref(other));
                            img = new_img;
                            directives.absorb(options);
                        }
                    }
                }
            }
        }
    }
    Ok((img, directives))
}

/// The EXIF orientation stored in the source bytes (1-8), if the
/// container carries a readable one. Missing or malformed metadata
/// yields `None`; re-encoding drops EXIF from outputs, so this read
/// is the only place orientation exists.
fn exif_orientation(bytes: &[u8]) -> Option<u32> {
    let mut cursor = std::io::Cursor::new(bytes);
    let exif = exif::Reader::new().read_from_container(&mut cursor).ok()?;
    let field = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?;
    field.value.get_uint(0)
}

/// Apply the transform an EXIF orientation value prescribes.
/// Values outside 1-8 leave the image untouched.
fn apply_orientation(img: DynamicImage, orientation: u32) -> DynamicImage {
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
            Operation::FitIn(size) => {
                let mut scale = f64::INFINITY;
                match size.width {
                    Dimension::Pixels(n) => scale = scale.min(scale_px(n, dpr) as f64 / iw as f64),
                    Dimension::Original => scale = scale.min(1.0),
                    Dimension::Proportional => {}
                }
                match size.height {
                    Dimension::Pixels(n) => scale = scale.min(scale_px(n, dpr) as f64 / ih as f64),
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

/// One step down the `max_bytes` quality ladder. AVIF encodes
/// are the pipeline's most expensive step by far, so they halve;
/// JPEG and JPEG XL shave 30% per step.
fn next_quality(format: ImageFormat, current: u8) -> u8 {
    match format {
        ImageFormat::Avif => (current / 2).max(1),
        _ => ((current as u32 * 7 / 10) as u8).max(1),
    }
}

/// How many re-encodes `max_bytes` may attempt per format before
/// returning the smallest result it has.
fn max_quality_attempts(format: ImageFormat) -> u32 {
    match format {
        ImageFormat::Avif => 4,
        _ => 12,
    }
}

/// Encode `img` to `format`. `quality` applies to JPEG, JPEG XL,
/// AVIF, and WebP (where 100 selects the lossless encode).
/// `fill` composites transparency for formats without an
/// alpha channel (only JPEG needs it here).
fn encode(
    img: &DynamicImage,
    format: ImageFormat,
    quality: Option<u8>,
    fill: Option<[u8; 3]>,
) -> Result<Vec<u8>, EngineError> {
    let (w, h) = (img.width(), img.height());
    if format == ImageFormat::Jxl {
        // Pure-Rust encoder (jixel). Its quality is a libjpeg-like
        // 0-100 scale, matching the quality filter's semantics.
        let config = jixel::EncodeConfig::default().with_quality(quality.unwrap_or(80) as f32);
        let encoded = if img.color().has_alpha() {
            let rgba = img.to_rgba8();
            jixel::encode_image_with_alpha(rgba.as_raw(), w as usize, h as usize, &config)
        } else {
            let rgb = img.to_rgb8();
            jixel::encode_image(rgb.as_raw(), w as usize, h as usize, &config)
        };
        return encoded.map_err(|e| EngineError::Encode(e.to_string()));
    }
    if format == ImageFormat::WebP {
        return encode_webp(img, quality);
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
            let rgb = match fill {
                Some(bg) if img.color().has_alpha() => flatten(img, bg),
                _ => img.to_rgb8(),
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
            // Pure-Rust ravif encoder via the image crate. Speed 8
            // keeps on-demand encoding responsive; AVIF encoding is
            // the expensive step in the pipeline by a wide margin.
            image::codecs::avif::AvifEncoder::new_with_speed_quality(
                &mut buf,
                8,
                quality.unwrap_or(80),
            )
            .write_image(pixels, w, h, color)
            .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        ImageFormat::Gif => {
            image::codecs::gif::GifEncoder::new(&mut buf)
                .encode(img.to_rgba8().as_raw(), w, h, ExtendedColorType::Rgba8)
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        ImageFormat::Jxl => unreachable!("JPEG XL returns early in encode"),
    }
    Ok(buf)
}

/// Encode WebP. Quality below 100 is a lossy VP8 encode via
/// oxideav-webp (pure Rust); quality 100 (or an encode the ladder
/// starts there) keeps the image crate's lossless VP8L encode.
/// The default quality is 80, like the other lossy formats.
fn encode_webp(img: &DynamicImage, quality: Option<u8>) -> Result<Vec<u8>, EngineError> {
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
        let rgba = img.to_rgba8();
        oxideav_webp::encode_rgba8(img.width(), img.height(), rgba.as_raw(), &opts)
    } else {
        let rgb = img.to_rgb8();
        oxideav_webp::encode_rgb8(img.width(), img.height(), rgb.as_raw(), &opts)
    };
    encoded.map_err(|e| EngineError::Encode(e.to_string()))
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

/// Shrink `(w, h)` proportionally to fit inside `cap`, if a cap is
/// set and the size exceeds it. Never grows the size.
fn cap_dimensions(w: u32, h: u32, cap: Option<(u32, u32)>) -> (u32, u32) {
    let Some((cw, ch)) = cap else {
        return (w, h);
    };
    if w <= cw && h <= ch {
        return (w, h);
    }
    let scale = (cw as f64 / w as f64).min(ch as f64 / h as f64);
    (
        ((w as f64 * scale).round() as u32).max(1),
        ((h as f64 * scale).round() as u32).max(1),
    )
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
    cap: Option<(u32, u32)>,
    dpr: f32,
) -> DynamicImage {
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
    let (w, h) = cap_dimensions(w, h, cap);
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
    img
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
fn apply_fit_in(
    img: DynamicImage,
    size: &ImageSize,
    cap: Option<(u32, u32)>,
    dpr: f32,
) -> DynamicImage {
    let size = &ImageSize {
        width: scale_dimension(size.width, dpr),
        height: scale_dimension(size.height, dpr),
        flip_horizontal: size.flip_horizontal,
        flip_vertical: size.flip_vertical,
    };
    let (sw, sh) = (img.width(), img.height());
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
        return img;
    }
    let w = ((sw as f64 * scale).round() as u32).max(1);
    let h = ((sh as f64 * scale).round() as u32).max(1);
    let (w, h) = cap_dimensions(w, h, cap);
    if w == sw && h == sh {
        img
    } else {
        resize_lanczos3(&img, w, h)
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
        Operation::FitIn(ImageSize {
            width: w,
            height: h,
            flip_horizontal: false,
            flip_vertical: false,
        })
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

    #[test]
    fn no_upscale_caps_every_resize_mode() {
        let engine = Engine;
        let src = png_bytes(&DynamicImage::new_rgb8(200, 200));
        let directive = Operation::Filters(vec![Filter::NoUpscale]);
        // Exact size, with the directive after the size on purpose:
        // the cap is request-wide, not positional.
        let out = engine
            .process(
                &src,
                &[
                    size_op(Dimension::Pixels(1000), Dimension::Pixels(500)),
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
    fn heic_source_fails_with_a_specific_error() {
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
        fn mask(&self, img: &DynamicImage) -> Result<image::GrayImage, String> {
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
            EngineError::Unsupported(msg) => assert!(msg.contains("not configured"), "{msg}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    /// A fake upscale provider: a deterministic 2x
    /// nearest-neighbor enlargement.
    #[derive(Debug)]
    struct DoubleUpscale;

    impl crate::ai::UpscaleProvider for DoubleUpscale {
        fn factor(&self) -> u32 {
            2
        }

        fn upscale(&self, img: &DynamicImage) -> Result<DynamicImage, String> {
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
            EngineError::Unsupported(msg) => assert!(msg.contains("not configured"), "{msg}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    /// A face provider reporting a face over the right half.
    #[derive(Debug)]
    struct RightFace;

    impl crate::ai::FaceProvider for RightFace {
        fn detect(&self, img: &DynamicImage) -> Result<Vec<crate::ai::FaceBox>, String> {
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
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!((out.width, out.height), (50, 100));
        let result = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        assert_eq!(result.get_pixel(25, 50), &image::Rgb([0, 0, 255]));
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
            Err(EngineError::Unsupported(msg)) if msg.contains("text is not configured")
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
        assert_eq!(next_quality(ImageFormat::Jpeg, 80), 56);
        assert_eq!(next_quality(ImageFormat::Jxl, 80), 56);
        assert_eq!(next_quality(ImageFormat::Avif, 80), 40);
        assert_eq!(next_quality(ImageFormat::Avif, 1), 1);
        assert_eq!(max_quality_attempts(ImageFormat::Avif), 4);
        assert!(max_quality_attempts(ImageFormat::Jpeg) > 4);
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
