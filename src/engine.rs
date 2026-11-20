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
    Align, Dimension, Filter, FormatOptions, ImageFormat, ImageSize, Operation, PadFill,
    PlainFilter, ResolvedQuality,
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
pub(in crate::engine) const MAX_ANIMATED_FRAMES: usize = 200;
pub(in crate::engine) const MAX_ANIMATED_PIXELS: u64 = 100_000_000;
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
/// images in the server), or, for an AVIF source in the server,
/// through [`decode_source_image`] and the worker's decode under
/// the same limits, so no entry point can be bombed past the
/// budget. The declared dimensions are probed first and
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
    // An AVIF's declared size was checked against the decode
    // limits in the screen, from its `ispe` property, so the
    // dimension probe here would only re-run the decoder it
    // bounds (issue #522). Every other format is probed first
    // and checked in our own code, because decoder adherence to
    // `max_alloc` varies by format; a probe failure is not fatal
    // (some valid files have headers the probe cannot read), the
    // limited decode behind it is the second line.
    if !matches!(image::guess_format(bytes), Ok(image::ImageFormat::Avif)) {
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
    }
    // JPEG XL decodes with its frames held to the budget too,
    // which the image crate's hook cannot do (see decode_jxl).
    if is_jxl(bytes) {
        return decode_jxl(bytes);
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

/// Decode an untrusted still source, routing an AVIF through
/// the capped worker process with a deadline (issue #522): the
/// AV1 decode is a large native decoder fed untrusted bytes, so
/// it runs in the worker SVG rendering already uses (issue
/// #447), and a decode still running at `deadline` is killed
/// with the worker, not left on a render thread. The container is
/// screened in this process first (the cheap box and item
/// checks), then the screened bytes are handed to the worker,
/// which runs the same bounded image-crate decode the
/// in-process path runs, so a valid file decodes to the same
/// pixels either way. Non-AVIF sources, and builds or contexts
/// with no worker (the library, the fuzz harness, the CLI),
/// decode in process through [`decode_image`]. The server's own
/// freshly encoded output is re-decoded in process through
/// [`decode_image`], which keeps the quality search off the
/// per-decode worker spawn.
pub(crate) fn decode_source_image(
    bytes: &[u8],
    deadline: Option<std::time::Instant>,
) -> Result<DynamicImage, EngineError> {
    #[cfg(feature = "avif-decode")]
    if matches!(image::guess_format(bytes), Ok(image::ImageFormat::Avif)) {
        let screened = crate::avif::screen(bytes)?;
        let worker_deadline =
            deadline.unwrap_or_else(|| std::time::Instant::now() + crate::svg::WORKER_TIMEOUT);
        if let Some(result) = crate::svg::decode_avif_in_worker(
            &screened,
            MAX_DECODE_DIMENSION,
            MAX_DECODE_ALLOC,
            worker_deadline,
        ) {
            return result.map_err(|e| match e {
                crate::svg::RenderError::TimedOut(_) if deadline.is_some() => EngineError::Timeout,
                e => EngineError::Decode(e.to_string()),
            });
        }
    }
    let _ = deadline;
    decode_image(bytes)
}

/// The most memory the decoded watermark and overlay images
/// of one request may hold together (issue #613): twice the
/// decode allocation limit, so the largest image the limits
/// admit fits with its RGBA copy (see
/// [`crate::filters::WatermarkImage::held_bytes`]).
pub(crate) const MAX_WATERMARK_BYTES: u64 = 2 * MAX_DECODE_ALLOC;

/// Decode a request's watermark and overlay images, keyed by
/// URL, within [`MAX_WATERMARK_BYTES`] (issue #613). Each
/// image used to be bounded alone, so a request naming many
/// large ones held all of them at once.
pub(crate) fn decode_watermarks(
    sources: &[(String, Vec<u8>)],
    deadline: Option<std::time::Instant>,
) -> Result<crate::filters::WatermarkImages, EngineError> {
    decode_watermarks_within(sources, deadline, MAX_WATERMARK_BYTES)
}

/// [`decode_watermarks`] against any budget. Each image is
/// charged before it decodes, at four bytes a pixel of the
/// size its header declares, and again after, at what it
/// holds. The deadline is checked before each decode.
fn decode_watermarks_within(
    sources: &[(String, Vec<u8>)],
    deadline: Option<std::time::Instant>,
    budget: u64,
) -> Result<crate::filters::WatermarkImages, EngineError> {
    let over = |need: u64| {
        EngineError::Unsupported(format!(
            "watermark and overlay images would hold {need} bytes decoded, over the {budget} byte limit"
        ))
    };
    let mut images = crate::filters::WatermarkImages::new();
    let mut held: u64 = 0;
    for (url, bytes) in sources {
        state::check_deadline(deadline)?;
        if let Some((width, height)) = declared_size(bytes) {
            let need = held.saturating_add(width as u64 * height as u64 * 4);
            if need > budget {
                return Err(over(need));
            }
        }
        // An AVIF watermark decodes in the capped worker under
        // the request deadline, like the source (issue #522).
        let image = crate::filters::WatermarkImage::new(decode_source_image(bytes, deadline)?);
        held = held.saturating_add(image.held_bytes());
        if held > budget {
            return Err(over(held));
        }
        images.insert(url.clone(), image);
    }
    Ok(images)
}

/// The size an image's header declares, read without a
/// decode, for the formats a watermark can be. `None` when
/// the header states none; the decode limits still bound
/// such an image.
fn declared_size(bytes: &[u8]) -> Option<(u32, u32)> {
    match probe_kind(bytes) {
        "jxl" => jxl_dimensions(bytes),
        "avif" => crate::avif::declared_size(bytes),
        "svg" | "heic" | "video" | "pdf" | "unknown" => None,
        _ => image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .ok()
            .and_then(|probe| probe.into_dimensions().ok()),
    }
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
    let mut decoder =
        zune_jpeg::JpegDecoder::new_with_options(Cursor::new(bytes), zune_jpeg_options());
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

/// The options [`decode_jpeg_direct`] decodes under. The
/// scaled decode reads a file's header with the same ones
/// ([`jpeg_scaled`]), so the size and the colour space it
/// goes by are the ones a full-size decode would.
fn zune_jpeg_options() -> zune_core::options::DecoderOptions {
    use zune_core::colorspace::ColorSpace;
    zune_core::options::DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_strict_mode(false)
        .set_max_width(MAX_DECODE_DIMENSION as usize)
        .set_max_height(MAX_DECODE_DIMENSION as usize)
}

/// The scaled JPEG decode on its own (issue #653): `bytes`
/// at one over `denominator` (2, 4 or 8), decoded exactly as
/// a render decodes a source its first resize shrinks that
/// far. `None` for a file the scaled decode does not take
/// and for one libjpeg-turbo refuses or warns about, which a
/// render decodes at full size instead. Public for the fuzz
/// harness's decode target, as [`decode_image`] is.
pub fn decode_jpeg_scaled(bytes: &[u8], denominator: u32) -> Option<DynamicImage> {
    jpeg_scaled::decode_at(bytes, denominator)
}

/// WebP still decode through libwebp (the `webp` crate):
/// measured about 3.5x faster than image-webp at 2400x1600
/// with bit-identical pixels, for lossy and lossless files
/// alike. The crate refuses animated files (returns None),
/// which keeps animated sources on the generic path, and its
/// DynamicImage variant matches the generic path (Rgba8 only
/// when the file has alpha).
///
/// libwebp ignores the image crate's decode limits, and the
/// dimension probe ahead of this fast path is not fatal (some
/// valid files have headers the probe cannot read), so the size
/// libwebp will decode is read from the bitstream header and
/// held to the same dimension and allocation budgets here,
/// before any decode. Over either, the fast path stands down to
/// the bounded generic decode, which refuses the file: image-webp
/// fails on the same header the probe could not read, or the
/// probe already charged the budget on one it could.
fn decode_webp_direct(bytes: &[u8]) -> Option<DynamicImage> {
    let features = webp::BitstreamFeatures::new(bytes)?;
    let (width, height) = (u64::from(features.width()), u64::from(features.height()));
    if width > MAX_DECODE_DIMENSION as u64
        || height > MAX_DECODE_DIMENSION as u64
        || width * height * 4 > MAX_DECODE_ALLOC
    {
        return None;
    }
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
        decode_gif_frames(bytes, None)?;
        return Ok(PreparedUpload {
            stored: bytes.to_vec(),
            media: UploadMedia::Gif,
        });
    }
    if detected == ImageFormat::WebP && webp_animation_params(bytes).is_some() {
        decode_webp_frames(bytes, None)?;
        return Ok(PreparedUpload {
            stored: bytes.to_vec(),
            media: UploadMedia::WebP,
        });
    }
    if detected == ImageFormat::Jxl {
        check_jxl_dimensions(bytes)?;
    }
    // Upload validation has no request deadline, so an AVIF
    // upload decodes in the worker under its default timeout
    // (issue #522); every other still decodes in process.
    decode_source_image(bytes, None)?;
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
    jxl_header(bytes).map(|(width, height, _)| (width, height))
}

/// A JPEG XL header's size and its number of extra channels,
/// from the basic-info event alone, as [`jxl_dimensions`].
fn jxl_header(bytes: &[u8]) -> Option<(u32, u32, usize)> {
    use jxl::api::{Event, JxlDecoder, JxlDecoderOptions};
    let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());
    let mut input = std::io::BufReader::new(Cursor::new(bytes));
    let event = decoder.process(&mut input, None, None).ok()?;
    if event != Event::BasicInfo {
        return None;
    }
    decoder.basic_info().map(|info| {
        (
            info.size.0 as u32,
            info.size.1 as u32,
            info.extra_channels.len(),
        )
    })
}

/// The most samples a frame of a JPEG XL file with
/// `extra_channels` may hold: as many pixels as the decode
/// budget gives a whole image ([`MAX_DECODE_ALLOC`] as RGBA,
/// the bound every still is held to), each with its three
/// colour samples and the extra ones. jxl-rs refuses a frame
/// whose count reaches the limit, hence the one more.
fn jxl_sample_limit(extra_channels: usize) -> usize {
    let pixels = usize::try_from(MAX_DECODE_ALLOC / 4).unwrap_or(usize::MAX);
    pixels.saturating_mul(3 + extra_channels).saturating_add(1)
}

/// Decode a JPEG XL source with jxl-rs held to the decode
/// budget.
///
/// The image crate's JPEG XL hook (`jxl-image-rs-integration`)
/// runs jxl-rs with its default options, which set no sample
/// limit, so every frame header is taken at its word. A frame
/// may declare a crop size of its own, unrelated to the image
/// size the header states (the one [`check_jxl_dimensions`]
/// checks), and the decoder allocates and renders what the
/// frame declares. Here the decoder carries the limit
/// ([`jxl_sample_limit`]), so a frame holding more pixels
/// than a whole image may is refused at its header, before
/// anything is allocated for it.
///
/// Everything else is what the hook does, so a file decodes
/// to the same pixels: the output format for each bit depth,
/// colour, and alpha, the oriented size, the first displayed
/// frame, and the image crate's allocation checks on the way
/// in, with the same errors. The hook still detects the format
/// and reads the header for the size probe in [`decode_image`],
/// which decodes no frame.
fn decode_jxl(bytes: &[u8]) -> Result<DynamicImage, EngineError> {
    use image::error::{DecodingError, ImageFormatHint, LimitError, LimitErrorKind};
    use image::{ColorType, ImageBuffer, ImageError};
    use jxl::api::{
        Endianness, Event, ExtraChannel, JxlBitDepth, JxlColorType, JxlDataFormat, JxlDecoder,
        JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
    };
    let fail = |e: ImageError| EngineError::Decode(e.to_string());
    let decoding = |e: Box<dyn std::error::Error + Send + Sync>| {
        fail(ImageError::Decoding(DecodingError::new(
            ImageFormatHint::Name("JPEG XL".to_owned()),
            e,
        )))
    };
    let truncated = || decoding("truncated JPEG XL image".into());
    let limit = |kind| fail(ImageError::Limits(LimitError::from_kind(kind)));

    let extra_channels = jxl_header(bytes).map_or(0, |(_, _, n)| n);
    let mut options = JxlDecoderOptions::default();
    options.sample_limit = Some(jxl_sample_limit(extra_channels));
    let mut decoder = JxlDecoder::new(options);
    let mut input = std::io::BufReader::new(Cursor::new(bytes));
    if decoder
        .process(&mut input, None, None)
        .map_err(|e| decoding(e.into()))?
        != Event::BasicInfo
    {
        return Err(truncated());
    }
    let info = decoder.basic_info().ok_or_else(truncated)?.clone();
    let width = u32::try_from(info.size.0).map_err(|_| limit(LimitErrorKind::DimensionError))?;
    let height = u32::try_from(info.size.1).map_err(|_| limit(LimitErrorKind::DimensionError))?;
    let grayscale = decoder
        .current_pixel_format()
        .ok_or_else(truncated)?
        .color_type
        .is_grayscale();
    let alpha = info
        .extra_channels
        .iter()
        .any(|channel| channel.ec_type == ExtraChannel::Alpha);
    let jxl_color = match (grayscale, alpha) {
        (true, false) => JxlColorType::Grayscale,
        (true, true) => JxlColorType::GrayscaleAlpha,
        (false, false) => JxlColorType::Rgb,
        (false, true) => JxlColorType::Rgba,
    };
    let (color, data_format) = match info.bit_depth {
        JxlBitDepth::Float { .. } => (
            if alpha {
                ColorType::Rgba32F
            } else {
                ColorType::Rgb32F
            },
            JxlDataFormat::F32 {
                endianness: Endianness::native(),
            },
        ),
        JxlBitDepth::Int { bits_per_sample } if bits_per_sample <= 8 => (
            match (grayscale, alpha) {
                (true, false) => ColorType::L8,
                (true, true) => ColorType::La8,
                (false, false) => ColorType::Rgb8,
                (false, true) => ColorType::Rgba8,
            },
            JxlDataFormat::U8 { bit_depth: 8 },
        ),
        JxlBitDepth::Int { .. } => (
            match (grayscale, alpha) {
                (true, false) => ColorType::L16,
                (true, true) => ColorType::La16,
                (false, false) => ColorType::Rgb16,
                (false, true) => ColorType::Rgba16,
            },
            JxlDataFormat::U16 {
                endianness: Endianness::native(),
                bit_depth: 16,
            },
        ),
    };
    // A float output is colour only: the hook asks for RGB or
    // RGBA whatever the colour space, as here.
    let jxl_color = match data_format {
        JxlDataFormat::F32 { .. } if alpha => JxlColorType::Rgba,
        JxlDataFormat::F32 { .. } => JxlColorType::Rgb,
        _ => jxl_color,
    };
    decoder
        .set_pixel_format(JxlPixelFormat {
            color_type: jxl_color,
            color_data_format: Some(data_format),
            extra_channel_format: vec![None; info.extra_channels.len()],
        })
        .map_err(|e| decoding(e.into()))?;
    // The image crate's checks before it reads the pixels:
    // the reader sets the output buffer aside from the budget,
    // and the hook holds what is left to the output again,
    // with a conversion buffer for samples wider than a byte.
    let output = u64::from(width) * u64::from(height) * color.bytes_per_pixel() as u64;
    let remaining = MAX_DECODE_ALLOC
        .checked_sub(output)
        .ok_or_else(|| limit(LimitErrorKind::InsufficientMemory))?;
    check_decoded_dimensions(u64::from(width), u64::from(height))?;
    let wide = color.bytes_per_pixel() / color.channel_count() > 1;
    let scratch = if wide { output } else { 0 };
    if output.saturating_add(scratch) > remaining {
        return Err(limit(LimitErrorKind::InsufficientMemory));
    }
    // The decoder writes wider samples in native byte order and
    // wants each row aligned to the sample size, so the bytes
    // start at the first aligned offset of a slightly larger
    // buffer, then become samples.
    let len = usize::try_from(output).map_err(|_| limit(LimitErrorKind::InsufficientMemory))?;
    let sample = color.bytes_per_pixel() as usize / color.channel_count() as usize;
    let mut raw = vec![0u8; len + sample - 1];
    let start = (sample - raw.as_ptr() as usize % sample) % sample;
    let bytes = &mut raw[start..start + len];
    let row = len / (height as usize).max(1);
    let mut frame = JxlOutputBuffer::new(bytes, height as usize, row);
    loop {
        match decoder
            .process(&mut input, Some(std::slice::from_mut(&mut frame)), None)
            .map_err(|e| decoding(e.into()))?
        {
            Event::FrameHeader => {}
            Event::FrameComplete { .. } => break,
            _ => return Err(truncated()),
        }
    }
    let bytes = &raw[start..start + len];
    let u16s = || -> Vec<u16> {
        let (pairs, _) = bytes.as_chunks::<2>();
        pairs.iter().map(|p| u16::from_ne_bytes(*p)).collect()
    };
    let f32s = || -> Vec<f32> {
        let (quads, _) = bytes.as_chunks::<4>();
        quads.iter().map(|q| f32::from_ne_bytes(*q)).collect()
    };
    let image = match color {
        ColorType::L8 => ImageBuffer::from_raw(width, height, raw).map(DynamicImage::ImageLuma8),
        ColorType::La8 => ImageBuffer::from_raw(width, height, raw).map(DynamicImage::ImageLumaA8),
        ColorType::Rgb8 => ImageBuffer::from_raw(width, height, raw).map(DynamicImage::ImageRgb8),
        ColorType::Rgba8 => ImageBuffer::from_raw(width, height, raw).map(DynamicImage::ImageRgba8),
        ColorType::L16 => {
            ImageBuffer::from_raw(width, height, u16s()).map(DynamicImage::ImageLuma16)
        }
        ColorType::La16 => {
            ImageBuffer::from_raw(width, height, u16s()).map(DynamicImage::ImageLumaA16)
        }
        ColorType::Rgb16 => {
            ImageBuffer::from_raw(width, height, u16s()).map(DynamicImage::ImageRgb16)
        }
        ColorType::Rgba16 => {
            ImageBuffer::from_raw(width, height, u16s()).map(DynamicImage::ImageRgba16)
        }
        ColorType::Rgb32F => {
            ImageBuffer::from_raw(width, height, f32s()).map(DynamicImage::ImageRgb32F)
        }
        ColorType::Rgba32F => {
            ImageBuffer::from_raw(width, height, f32s()).map(DynamicImage::ImageRgba32F)
        }
        _ => None,
    };
    image.ok_or_else(|| {
        fail(ImageError::Parameter(
            image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::DimensionMismatch,
            ),
        ))
    })
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

/// The media kind [`probe_source`] reports for `bytes`, from
/// their signatures alone: no header is parsed for it.
pub(crate) fn probe_kind(bytes: &[u8]) -> &'static str {
    if crate::svg::is_svg(bytes) {
        return "svg";
    }
    if is_heic(bytes) {
        return "heic";
    }
    if is_jxl(bytes) {
        return "jxl";
    }
    match image::guess_format(bytes) {
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
    }
}

/// Whether sizing a source of `kind` is more than a header
/// read: an SVG's intrinsic size takes a parse of the whole
/// document in the SVG worker (up to [`SVG_PROBE_TIMEOUT`]),
/// an AVIF's goes through the decoder, and a HEIC's takes a
/// walk of its item graph, which the file's references drive
/// (bounded, but up to tens of milliseconds). A caller on an
/// async runtime runs [`probe_source`] for these on a
/// blocking thread.
pub(crate) fn probe_is_heavy(kind: &str) -> bool {
    matches!(kind, "svg" | "avif" | "heic")
}

/// A test hook standing in for an SVG whose size probe is
/// slow (a document whose usvg parse runs near the probe's
/// deadline), without relying on usvg's timing or crafting
/// such a document: an SVG source whose bytes contain this
/// marker makes [`probe_source`] sleep [`SVG_PROBE_TEST_DELAY`]
/// while it sizes it. The sleep runs wherever `probe_source`
/// runs, so it blocks a request thread when the probe is on
/// one and a blocking thread when it is not.
#[cfg(test)]
pub(crate) const SVG_PROBE_SLOW_MARKER: &[u8] = b"crabcrop-slow-probe";

/// How long the [`SVG_PROBE_SLOW_MARKER`] hook sleeps.
#[cfg(test)]
pub(crate) const SVG_PROBE_TEST_DELAY: std::time::Duration = std::time::Duration::from_millis(1500);

/// Probe source bytes for the explain report: kind and
/// dimensions, headers only (see [`SourceProbe`]). For the
/// kinds [`probe_is_heavy`] names, this blocks for as long as
/// the parse or the decode takes.
pub(crate) fn probe_source(bytes: &[u8]) -> SourceProbe {
    let kind = probe_kind(bytes);
    #[cfg(test)]
    if kind == "svg"
        && bytes
            .windows(SVG_PROBE_SLOW_MARKER.len())
            .any(|w| w == SVG_PROBE_SLOW_MARKER)
    {
        std::thread::sleep(SVG_PROBE_TEST_DELAY);
    }
    let dimensions = match kind {
        // The intrinsic size takes a parse, which runs in the
        // SVG worker (issue #447) under the probe's own short
        // deadline: the explain handler waits on it.
        "svg" => {
            let deadline = std::time::Instant::now() + SVG_PROBE_TIMEOUT;
            crate::svg::screen(bytes)
                .ok()
                .and_then(|screened| screened.intrinsic_size(Some(deadline)).ok())
                .and_then(|(w, h)| {
                    let (w, h) = (w.round() as u32, h.round() as u32);
                    if w >= 1 && h >= 1 { Some((w, h)) } else { None }
                })
        }
        "heic" => crate::heic::probe_dimensions(bytes),
        "jxl" => jxl_dimensions(bytes),
        // An AVIF's size is read from its container's `ispe`
        // property, not by decoding it: sizing an AVIF no longer
        // runs the decoder. A container the walk refuses, or one
        // that names no `ispe`, states no dimensions.
        "avif" => crate::avif::declared_size(bytes),
        // Every other format reads its size from the image
        // crate's header probe, after the same container screen
        // (issue #522), so a refused one states no dimensions.
        _ => crate::avif::screen(bytes).ok().and_then(|bytes| {
            image::ImageReader::new(Cursor::new(&*bytes))
                .with_guessed_format()
                .ok()
                .and_then(|probe| probe.into_dimensions().ok())
        }),
    };
    SourceProbe { kind, dimensions }
}

/// Reject an output size past the shared pixel budget, before
/// anything allocates it. u64 math, so DPR-scaled dimensions
/// cannot wrap past the check.
pub(in crate::engine) fn check_output_pixels(width: u32, height: u32) -> Result<(), EngineError> {
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
/// as decoded, and a later step can still bring it within
/// the limit. Whatever the chain leaves at the end is held
/// to the limit by [`check_final_output`].
pub(in crate::engine) fn check_output_size(
    width: u32,
    height: u32,
    input: (u32, u32),
) -> Result<(), EngineError> {
    let max = crate::url::MAX_DIMENSION;
    if (width, height) != input && (width > max || height > max) {
        return Err(EngineError::Unsupported(format!(
            "requested output {width}x{height} exceeds the {max} pixel limit per axis"
        )));
    }
    check_output_pixels(width, height)
}

/// The last word on size, for the image a chain hands its
/// encoder: within [`crate::url::MAX_DIMENSION`] per axis,
/// then within the pixel budget (issue #528). The steps
/// let a size they did not change pass (see
/// [`check_output_size`]), so a source at the 16384 px
/// decode ceiling reached the encoder as decoded when no
/// step resized it: WebP refused the side with a `500`, and
/// every other format served an output past the documented
/// limit. There is no later step to bring the size within
/// the limit, so this check has no such exemption.
///
/// The refusal starts `requested output WxH exceeds`, as
/// the steps' refusals do: it is the same refusal of an
/// output size, made later, and whatever tells output-size
/// refusals from others by that wording must count it too.
pub(in crate::engine) fn check_final_output(width: u32, height: u32) -> Result<(), EngineError> {
    let max = crate::url::MAX_DIMENSION;
    if width > max || height > max {
        return Err(EngineError::Unsupported(format!(
            "requested output {width}x{height} exceeds the {max} pixel limit per axis; \
             resize or crop it to fit"
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
pub(in crate::engine) struct OutputBudget {
    limit: u64,
    total: u64,
}

impl OutputBudget {
    pub(in crate::engine) fn new() -> Self {
        Self {
            limit: MAX_OUTPUT_PIXELS,
            total: 0,
        }
    }

    #[cfg(test)]
    pub(in crate::engine) fn with_limit(limit: u64) -> Self {
        Self { limit, total: 0 }
    }

    /// Charge one frame's output pixels against the budget.
    pub(in crate::engine) fn charge(&mut self, width: u32, height: u32) -> Result<(), EngineError> {
        self.total = self
            .total
            .saturating_add(u64::from(width) * u64::from(height));
        if self.total > self.limit {
            return Err(self.refusal());
        }
        Ok(())
    }

    /// Refuse up front when `frames` frames of `width` x
    /// `height` would pass the budget. Every frame of one
    /// animation renders at one size (each starts at the
    /// canvas size, and the trim and smart-crop windows are
    /// shared), so the first frame settles what the run will
    /// charge: the refusal comes after one frame instead of a
    /// budget's worth, with the message the charge gives. A
    /// hinted DPR factor tried and refused for it costs one
    /// frame this way (issue #529).
    pub(in crate::engine) fn check_frames(
        &self,
        width: u32,
        height: u32,
        frames: usize,
    ) -> Result<(), EngineError> {
        let run = (u64::from(width) * u64::from(height)).saturating_mul(frames as u64);
        if self.total.saturating_add(run) > self.limit {
            return Err(self.refusal());
        }
        Ok(())
    }

    /// The refusal of a run past the budget.
    fn refusal(&self) -> EngineError {
        EngineError::Unsupported(format!(
            "animated output exceeds the {} pixel budget across frames",
            self.limit
        ))
    }
}

/// Reject an operation list past
/// [`crate::url::MAX_OPERATIONS`] (issue #182, audit
/// R2-12). The server refuses such a request at parse time,
/// before any fetch (issue #469); this is the backstop for
/// every other caller of the engine, with the same message
/// and the same `unsupported_operation` class.
pub(in crate::engine) fn check_operation_count(ops: &[Operation]) -> Result<(), EngineError> {
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

/// The effective DPR factor: an explicit `dpr(...)` directive
/// wins, then the client hint carried on the context, else 1.
pub(in crate::engine) fn dpr_factor(explicit: Option<f32>, ctx: &ProcessContext<'_>) -> f32 {
    explicit.or(ctx.dpr).unwrap_or(1.0)
}

/// Run `render` for a request whose source is `bytes`, at
/// the DPR factor its hint `hint` settles on (issue #529):
/// first at the factor [`hint_start`] picks from the source's
/// header, then at each lower factor [`hint_step_down`] names
/// while the render still refuses its output size. The
/// result at the largest factor that fits, or the refusal
/// that stands. A refused try stops at the step that asks
/// for the oversized image, before that step allocates it.
pub(in crate::engine) fn step_down_hint<T>(
    bytes: &[u8],
    ops: &[Operation],
    directives: &crate::url::RequestDirectives,
    hint: Option<f32>,
    mut render: impl FnMut(Option<f32>) -> Result<T, EngineError>,
) -> Result<T, EngineError> {
    let mut hint = hint_start(bytes, ops, directives, hint);
    loop {
        match render(hint) {
            Err(err) => match hint_step_down(&err, ops, directives, hint) {
                Some(lower) => hint = Some(lower),
                None => return Err(err),
            },
            done => return done,
        }
    }
}

/// The factor a hinted render starts at (issue #529): the
/// largest of [`crate::url::DPR_FACTORS`] up to `hint` whose
/// sizes the walk of [`crate::explain::settle_hint`] (the
/// `/explain` forecast's) finds within the output limits, so
/// a factor the limits plainly refuse costs no decode. The
/// walk starts from the size the decode hands the chain: the
/// header's, turned by the EXIF orientation where the decode
/// turns it. Where the header states no size, and for an SVG
/// (whose intrinsic size takes a worker of its own) or a
/// HEIC (whose decoder turns it by metadata of its own), the
/// render starts at the hint and steps on its refusals alone.
fn hint_start(
    bytes: &[u8],
    ops: &[Operation],
    directives: &crate::url::RequestDirectives,
    hint: Option<f32>,
) -> Option<f32> {
    let factor = hint?;
    if directives.dpr.is_some()
        || factor <= 1.0
        || !scales_with_dpr(ops)
        || crate::svg::is_svg(bytes)
        || is_heic(bytes)
    {
        return hint;
    }
    let probe = probe_source(bytes);
    let Some((width, height)) = probe.dimensions else {
        return hint;
    };
    // `decode_source` turns a still by its EXIF orientation;
    // GIF and animated WebP frames keep their stored size.
    let turned = probe.kind != "gif"
        && webp_animation_params(bytes).is_none()
        && matches!(exif_orientation(bytes), Some(5..=8));
    let dimensions = Some(if turned {
        (height, width)
    } else {
        (width, height)
    });
    Some(crate::explain::settle_hint(
        ops,
        directives,
        &SourceProbe {
            dimensions,
            ..probe
        },
        factor,
    ))
}

/// The factor a render refused at the hinted DPR factor
/// `hint` tries next, or `None` when the refusal stands
/// (issue #529). A client hint is advisory, so when it alone
/// pushes a size past the per-axis limit or the pixel budget,
/// the render steps down [`crate::url::DPR_FACTORS`] to the
/// largest factor whose sizes fit both; at factor 1 the URL
/// itself is over, and its refusal stands. Nothing else steps:
/// a factor from `dpr(...)` (in the URL, a preset, or the
/// operator's default operations) is the request's own and
/// keeps its `400`, a refusal of any other kind is not the
/// factor's doing, and a chain with no pixel size for the
/// factor to scale renders the same at every factor.
pub(in crate::engine) fn hint_step_down(
    err: &EngineError,
    ops: &[Operation],
    directives: &crate::url::RequestDirectives,
    hint: Option<f32>,
) -> Option<f32> {
    if directives.dpr.is_some() || !is_output_size_refusal(err) || !scales_with_dpr(ops) {
        return None;
    }
    dpr_step_below(hint?)
}

/// The largest of [`crate::url::DPR_FACTORS`] below `factor`,
/// the next one a hinted factor steps down to (issue #529);
/// `None` at factor 1.
pub(crate) fn dpr_step_below(factor: f32) -> Option<f32> {
    crate::url::DPR_FACTORS
        .iter()
        .rev()
        .copied()
        .find(|&lower| lower < factor)
}

/// How every refusal of an output size opens: a step past
/// the per-axis limit or the pixel budget
/// ([`check_output_size`], [`check_output_pixels`]), a hinted
/// output past the per-axis limit ([`check_hinted_output`]),
/// a scaled overlay past the budget, and an animated output
/// past the budget across its frames ([`OutputBudget`]). A
/// check that refuses an output size later in the pipeline
/// words its refusal `requested output WxH exceeds ...` too,
/// so the step-down counts it.
const OUTPUT_SIZE_REFUSALS: [&str; 3] = [
    "requested output ",
    "scaled overlay ",
    "animated output exceeds ",
];

/// Whether `err` is a refusal of an output size (see
/// [`OUTPUT_SIZE_REFUSALS`]): the refusals a DPR factor can
/// cause, since it scales the sizes they measure.
fn is_output_size_refusal(err: &EngineError) -> bool {
    match err {
        EngineError::Unsupported(what) => OUTPUT_SIZE_REFUSALS
            .iter()
            .any(|opening| what.starts_with(opening)),
        _ => false,
    }
}

/// Hold the image a render at a hinted DPR factor past 1
/// ends with to [`crate::url::MAX_DIMENSION`] per axis, a
/// size no step changed included (issue #529). The steps let
/// a size they did not change pass ([`check_output_size`]),
/// so a hinted factor could take a resize to a 16384 px
/// source's own size and serve that side (or, as WebP, fail
/// to encode it), past the limit a stepped factor promises to
/// fit. The refusal is an output-size refusal, so the factor
/// steps down. At factor 1 the render is the one the URL gets
/// without the hint, under its own rules.
pub(in crate::engine) fn check_hinted_output(width: u32, height: u32) -> Result<(), EngineError> {
    let max = crate::url::MAX_DIMENSION;
    if width > max || height > max {
        return Err(EngineError::Unsupported(format!(
            "requested output {width}x{height} exceeds the {max} pixel limit per axis; \
             resize or crop it to fit"
        )));
    }
    Ok(())
}

/// Whether any step of `ops` names a pixel size, the one
/// thing a DPR factor scales: a resize, `fit-in`, `crop`, or
/// `pad` with a pixel axis. An SVG source's raster size
/// follows the first of these too.
pub(crate) fn scales_with_dpr(ops: &[Operation]) -> bool {
    ops.iter().any(|op| match op {
        Operation::Size(size)
        | Operation::Crop { size, .. }
        | Operation::Pad { size, .. }
        | Operation::FitIn { size, .. } => {
            matches!(size.width, Dimension::Pixels(_))
                || matches!(size.height, Dimension::Pixels(_))
        }
        _ => false,
    })
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
        Filter::Plain(
            PlainFilter::Grayscale
                | PlainFilter::Invert
                | PlainFilter::Blur(_)
                | PlainFilter::Brighten(_)
                | PlainFilter::Contrast(_)
                | PlainFilter::Gamma(_)
                | PlainFilter::Exposure(_)
                | PlainFilter::HighlightsShadows(..)
                | PlainFilter::Vibrance(_)
                | PlainFilter::HueRotate(_)
                | PlainFilter::Sharpen(_)
                | PlainFilter::Saturation(_)
                | PlainFilter::Rgb(..)
                | PlainFilter::Curve(..)
                | PlainFilter::Convolution(..)
                | PlainFilter::Noise(_)
                | PlainFilter::Equalize
                | PlainFilter::Enhance
                | PlainFilter::Sepia
                | PlainFilter::Duotone(..)
                | PlainFilter::Monochrome(_)
                | PlainFilter::Pixelate(_)
                | PlainFilter::Fill(_)
                | PlainFilter::Radius(_)
                | PlainFilter::Mask(_)
                | PlainFilter::KeepProfile
                | PlainFilter::MaxBytes(_)
                | PlainFilter::Quality(_)
                | PlainFilter::Palette(_)
                | PlainFilter::Format(_)
                | PlainFilter::NoUpscale
        ) | Filter::BlurFaces
            | Filter::PixelateFaces
            | Filter::BlurClass(_)
            | Filter::PixelateClass(_)
            | Filter::RecolorClass(..)
            | Filter::FacePad(_)
            | Filter::BlurRegion { .. }
            | Filter::PixelateRegion { .. }
            | Filter::RemoveRegion { .. }
            | Filter::Dpr(_)
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
                | Filter::Plain(PlainFilter::Border(..))
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
    if memo.as_ref().is_some_and(|(w, h, _)| (*w, *h) != dims) {
        *memo = None;
    }
    let (_, _, cached) = memo.get_or_insert_with(|| (dims.0, dims.1, provider.encode(img)));
    let embedding = cached
        .as_ref()
        .map_err(|e| EngineError::Inference(e.clone()))?;
    provider
        .decode(embedding, dims.0, dims.1, (det.x, det.y, det.w, det.h))
        .map_err(EngineError::Inference)
}

/// The most box-prompted segmentation decodes one request
/// runs (issue #713). Each detection of a class filter is one
/// decode, and a crowd can hold hundreds of detections: an
/// evaluation measured about 1.2 s more for 35 more people in
/// a debug build, all of it holding one of the AI permits. 64
/// covers a group photograph, about two seconds at that rate;
/// the detections past it are redacted by their boxes (see
/// [`ChainState::class_targets`]).
pub(crate) const MAX_SEGMENT_DECODES: usize = 64;

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
    check_deadline: impl Fn() -> Result<(), EngineError>,
) -> Result<(Vec<crate::filters::MaskedDetection>, Vec<SegmentObject>), EngineError> {
    let mut masked = Vec::with_capacity(detections.len());
    let mut objects = Vec::new();
    for det in detections {
        // Between decodes, not only between filters (issue
        // #713): a request with many objects stops at its
        // deadline instead of decoding on.
        check_deadline()?;
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
        5 => crate::filters::fliph_fast(&crate::filters::rotate90_fast(&img)),
        6 => crate::filters::rotate90_fast(&img),
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
    is_jxl_codestream(bytes)
        || bytes.starts_with(&[
            0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
        ])
}

/// Whether `bytes` are a bare JPEG XL codestream (the form the
/// encoder writes) rather than the box container. Only the
/// container has boxes for metadata, a C2PA manifest among them.
pub(crate) fn is_jxl_codestream(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xFF, 0x0A])
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
/// `Normal` reproduces a watermark exactly. A scaled overlay
/// is scaled only where it lands, a row at a time, with the
/// deadline checked before each row and within a row that
/// reads many source samples (see `overlay_scale`): the same
/// pixels the whole scale gave there.
///
/// On a frame of an animated render, `kept` holds the URL the
/// overlay was fetched from and the scaled parts the render's
/// frames share (see [`ScaledOverlays`]): a part an earlier
/// frame scaled is drawn from there, with the same pixels, and
/// one this frame scales is kept there for the frames after
/// it. A still passes `None`.
#[allow(clippy::too_many_arguments)]
fn blend_overlay_keeping(
    ctx: &ProcessContext<'_>,
    base: &DynamicImage,
    ov: &crate::filters::WatermarkImage,
    (x, y): (i32, i32),
    alpha: u8,
    mode: crate::url::BlendMode,
    width_pct: Option<u32>,
    kept: Option<(&str, &mut ScaledOverlays)>,
) -> Result<DynamicImage, EngineError> {
    // The deadline is checked before anything is copied or
    // scaled, as well as per row of the blend (issue #604).
    ctx.check_deadline()?;
    let target = match width_pct {
        Some(pct) => {
            let ov = ov.image();
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
            Some(target as u32)
        }
        None => None,
    };
    // An overlay that lands off the canvas is neither scaled
    // nor copied (issue #604), and leaves the base as it was,
    // in its own layout (issue #713).
    if !lands_on((base.width(), base.height()), (x, y)) {
        return Ok(base.clone());
    }
    // The scaled size is known before anything is scaled, so
    // only the part that lands is: a scale the pixel budget
    // admits can still take long whole, and a sliver of it on
    // the canvas no longer pays for the rest.
    let source_size = (ov.image().width(), ov.image().height());
    let top_size = match target {
        Some(target) => overlay_scale::scaled_size(source_size, target),
        None => source_size,
    };
    let Some(part) = landing((base.width(), base.height()), top_size, (x, y)) else {
        return Ok(base.clone());
    };
    let top = if top_size == source_size {
        // Scaled to its own size, the overlay is itself.
        ov.rgba_part(part.0)
    } else {
        let scale =
            || overlay_scale::scaled_part(ov.image(), top_size, part.0, || ctx.check_deadline());
        let origin = (part.0.0, part.0.1);
        match kept {
            Some((url, kept)) => {
                let room = u64::from(base.width()) * u64::from(base.height());
                let pixels = kept.part(url, top_size, part.0, room, scale)?;
                crate::filters::RgbaPart::shared(pixels, origin, top_size)
            }
            None => crate::filters::RgbaPart::copied(scale()?, origin, top_size),
        }
    };
    // The blend walks every overlay pixel; a large legal pair
    // still runs long, so the deadline is honored per row
    // (issue #358).
    composite(
        base,
        &top,
        (x, y),
        alpha,
        || ctx.check_deadline(),
        |b, o| blend_channel(mode, b, o),
    )
}

/// [`blend_overlay_keeping`] on a still, with nothing kept:
/// the form the tests call.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn blend_overlay(
    ctx: &ProcessContext<'_>,
    base: &DynamicImage,
    ov: &crate::filters::WatermarkImage,
    x: i32,
    y: i32,
    alpha: u8,
    mode: crate::url::BlendMode,
    width_pct: Option<u32>,
) -> Result<DynamicImage, EngineError> {
    blend_overlay_keeping(ctx, base, ov, (x, y), alpha, mode, width_pct, None)
}

/// Overlay `wm` onto `base` at (x, y) with `alpha` percent
/// opacity, honoring the watermark's own alpha channel.
/// Negative coordinates anchor from the right/bottom edges
/// (x = -1 puts the watermark's right edge one pixel from the
/// base's right edge). Parts falling outside the base are
/// clipped, never a panic. The deadline is checked before
/// anything is copied and per row of the composite, as the
/// overlay checks it, and only the part of the watermark that
/// lands on the base is read or copied (issue #604). A
/// watermark that lands nowhere leaves the base as it was, in
/// its own layout (issue #713).
fn overlay_watermark(
    ctx: &ProcessContext<'_>,
    base: &DynamicImage,
    wm: &crate::filters::WatermarkImage,
    x: i32,
    y: i32,
    alpha: u8,
) -> Result<DynamicImage, EngineError> {
    ctx.check_deadline()?;
    let top_size = (wm.image().width(), wm.image().height());
    let Some(part) = landing((base.width(), base.height()), top_size, (x, y)) else {
        return Ok(base.clone());
    };
    composite(
        base,
        &wm.rgba_part(part.0),
        (x, y),
        alpha,
        || ctx.check_deadline(),
        |_, o| o,
    )
}

/// Whether an image placed at (x, y) on a base of
/// `(width, height)` covers any base pixel, by the anchoring
/// rule [`composite`] follows. The answer does not depend on
/// the placed image's size, as long as it has a pixel: a
/// coordinate of 0 or more must fall inside the base, and a
/// negative one, which anchors the far edge that many pixels
/// in from the base's far edge, must be less than the base's
/// extent. So it can be asked before an overlay is scaled.
fn lands_on((width, height): (u32, u32), (x, y): (i32, i32)) -> bool {
    x.unsigned_abs() < width && y.unsigned_abs() < height
}

/// The part of a top image of size `top` placed at (x, y) on
/// a base of size `base` that lands on the base, and the base
/// pixel its top-left corner lands on (issue #604). A
/// negative coordinate anchors the top's right or bottom edge
/// that many pixels in from the base's. `None` when no pixel
/// lands.
fn landing(
    (bw, bh): (u32, u32),
    (tw, th): (u32, u32),
    (x, y): (i32, i32),
) -> Option<(crate::filters::Rect, (u32, u32))> {
    let (bw, bh, tw, th) = (bw as i64, bh as i64, tw as i64, th as i64);
    let left = if x >= 0 { x as i64 } else { bw - tw + x as i64 };
    let upper = if y >= 0 { y as i64 } else { bh - th + y as i64 };
    let columns = (-left).max(0)..(bw - left).min(tw);
    let rows = (-upper).max(0)..(bh - upper).min(th);
    if columns.is_empty() || rows.is_empty() {
        return None;
    }
    let part = (
        columns.start as u32,
        rows.start as u32,
        (columns.end - columns.start) as u32,
        (rows.end - rows.start) as u32,
    );
    Some((
        part,
        ((left + columns.start) as u32, (upper + rows.start) as u32),
    ))
}

/// The anchor, clip, and alpha loop the overlay and the
/// watermark share (issue #480). `top` sits on `base` at
/// `(x, y)`, a negative coordinate anchoring from the right or
/// bottom edge, and whatever falls outside `base` is clipped.
/// Each covered pixel's effective alpha is the top pixel's own
/// alpha times `alpha` percent, and each color channel moves
/// that far toward `blend(base, top)`, the per-pixel blend of
/// the base channel and the top one; the base alpha channel is
/// kept. Only the part of `top` that lands on `base` is read
/// (issue #604, see [`landing`]), and `each_row` runs before
/// each of its rows, where the callers check the deadline.
/// The blend works in 8-bit RGBA, so a composite that lands
/// is RGBA8; one that lands nowhere is the base as it was, in
/// its own layout (issue #713).
fn composite<E, T>(
    base: &DynamicImage,
    top: &T,
    (x, y): (i32, i32),
    alpha: u8,
    mut each_row: impl FnMut() -> Result<(), E>,
    blend: impl Fn(f32, f32) -> f32,
) -> Result<DynamicImage, E>
where
    T: image::GenericImageView<Pixel = image::Rgba<u8>>,
{
    let Some(((tx0, ty0, w, h), (bx0, by0))) =
        landing((base.width(), base.height()), top.dimensions(), (x, y))
    else {
        return Ok(base.clone());
    };
    let mut out = base.to_rgba8();
    let opacity = alpha as f32 / 100.0;
    for row in 0..h {
        each_row()?;
        for column in 0..w {
            let top_px = top.get_pixel(tx0 + column, ty0 + row);
            let a = opacity * (top_px[3] as f32 / 255.0);
            if a <= 0.0 {
                continue;
            }
            let bot = out.get_pixel_mut(bx0 + column, by0 + row);
            for c in 0..3 {
                let blended = blend(bot[c] as f32, top_px[c] as f32);
                bot[c] = (bot[c] as f32 * (1.0 - a) + blended * a).round() as u8;
            }
        }
    }
    Ok(DynamicImage::ImageRgba8(out))
}

/// The image a JPEG with a `fill` encodes: `img` flattened over
/// the fill, as the encoder flattens it, when `img` has alpha;
/// `None` for every other format, a JPEG without a fill, and
/// an image with no alpha to flatten, which encode as they
/// are.
fn flattened_for(
    img: &DynamicImage,
    format: StillFormat,
    fill: Option<[u8; 3]>,
) -> Option<DynamicImage> {
    match (format, fill) {
        (StillFormat::Jpeg, Some(bg)) if img.color().has_alpha() => {
            Some(DynamicImage::ImageRgb8(flatten(img, bg)))
        }
        _ => None,
    }
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
    apply_size_windowed(img, size, cap, dpr, SourceWindow::Whole)
}

/// What part of the frame in hand a resize reads.
#[derive(Clone, Copy)]
enum SourceWindow {
    /// All of it.
    Whole,
    /// The rectangle a `crop-rect` named, resampled through
    /// the backend's crop box (issue #287).
    Fused(Window),
    /// The window an aspect crop decided on (issue #654).
    /// The result is in the frame's layout and byte for byte
    /// the two-step chain's: the window cut, then resized
    /// (see [`resize_lanczos3_cut`]).
    Cut(Window),
    /// A frame the JPEG decoder reduced (issue #653): the
    /// picture of a larger image, at a fraction of its size.
    /// The size resolves and is checked against that image's
    /// dimensions, so the target is the one the image would
    /// get, and the part of the frame the image fills is
    /// resampled to it (see [`resize_lanczos3_reduced`]).
    Reduced(jpeg_scaled::Reduced),
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
    apply_size_windowed(img, size, cap, dpr, SourceWindow::Fused(window))
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

/// The size a resize to `size` leaves a `sw` by `sh` image
/// at: a pixel side as written, scaled by `dpr`; an `orig`
/// side and a proportional side from the image; then `cap`,
/// from `no_upscale`. [`apply_size_windowed`] resamples to
/// it, and the scaled JPEG decode reads it ahead of the
/// decode (issue #653), so the two cannot disagree on where
/// a resize is heading.
fn resize_target(
    size: &ImageSize,
    dpr: f32,
    (sw, sh): (u32, u32),
    cap: Option<SizeCap>,
) -> (u32, u32) {
    let width = scale_dimension(size.width, dpr);
    let height = scale_dimension(size.height, dpr);
    let w = match width {
        Dimension::Pixels(n) => n,
        Dimension::Original => sw,
        Dimension::Proportional => match height {
            Dimension::Pixels(h) => proportional_side(sw, h, sh),
            _ => sw,
        },
    };
    let h = match height {
        Dimension::Pixels(n) => n,
        Dimension::Original => sh,
        Dimension::Proportional => match width {
            Dimension::Pixels(w) => proportional_side(sh, w, sw),
            _ => sh,
        },
    };
    match cap {
        Some(SizeCap::Axes(cw, ch)) => (w.min(cw), h.min(ch)),
        Some(SizeCap::Fit(cw, ch)) => cap_dimensions(w, h, (cw, ch)),
        None => (w, h),
    }
}

fn apply_size_windowed(
    img: DynamicImage,
    size: &ImageSize,
    cap: Option<SizeCap>,
    dpr: f32,
    window: SourceWindow,
) -> Result<DynamicImage, EngineError> {
    check_nonzero_size(size)?;
    // A cut window is clamped into the frame first, as the
    // cut itself clamps it, so the size resolves and is
    // checked against the dimensions a materialized cut
    // would have.
    let window = match window {
        SourceWindow::Cut(window) => SourceWindow::Cut(crate::crop::clamp_window(
            window,
            (img.width(), img.height()),
        )),
        other => other,
    };
    let (sw, sh) = match window {
        SourceWindow::Fused((_, _, w, h)) | SourceWindow::Cut((_, _, w, h)) => (w, h),
        SourceWindow::Reduced(reduced) => reduced.full,
        SourceWindow::Whole => (img.width(), img.height()),
    };
    let (w, h) = resize_target(size, dpr, (sw, sh), cap);
    check_output_size(w, h, (sw, sh))?;
    let unchanged = w == sw && h == sh;
    let img = match window {
        // A reduced frame is not the size the target was
        // resolved against, so it is resampled whatever the
        // target is.
        SourceWindow::Reduced(reduced) => resize_lanczos3_reduced(img, reduced.window, w, h),
        // No resample runs: the two-step path returns
        // the materialized window itself, so the fused
        // path materializes it here.
        SourceWindow::Fused((x, y, ww, wh)) if unchanged => img.crop_imm(x, y, ww, wh),
        SourceWindow::Cut(window) if unchanged => crate::crop::cut(&img, window),
        SourceWindow::Whole if unchanged => img,
        SourceWindow::Fused((x, y, ww, wh)) => resize_lanczos3_cropped(&img, x, y, ww, wh, w, h),
        SourceWindow::Cut(window) => resize_lanczos3_cut(img, window, w, h),
        SourceWindow::Whole => resize_lanczos3_owned(img, w, h),
    };
    Ok(flipped(img, size))
}

/// [`apply_size`] of a frame lent for the resize (issue #737):
/// the image it makes, with the frame back as it came (see
/// [`resize_lanczos3_lent`]). `None` when the size is the
/// frame's own, where [`apply_size`] hands on the frame itself,
/// which a lent frame cannot be.
fn apply_size_lent(
    frame: &mut DynamicImage,
    size: &ImageSize,
    cap: Option<SizeCap>,
    dpr: f32,
) -> Result<Option<DynamicImage>, EngineError> {
    check_nonzero_size(size)?;
    let dims = (frame.width(), frame.height());
    let (w, h) = resize_target(size, dpr, dims, cap);
    check_output_size(w, h, dims)?;
    if (w, h) == dims {
        return Ok(None);
    }
    Ok(Some(flipped(resize_lanczos3_lent(frame, w, h), size)))
}

/// `img` mirrored on the axes `size` flips.
fn flipped(mut img: DynamicImage, size: &ImageSize) -> DynamicImage {
    if size.flip_horizontal {
        img = crate::filters::fliph_fast(&img);
    }
    if size.flip_vertical {
        img = crate::filters::flipv_fast(&img);
    }
    img
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
    match fast_image_resize::Resizer::new().resize(img, &mut dst, &lanczos3_options()) {
        Ok(()) => dst,
        Err(_) => DynamicImage::ImageRgba8(image::imageops::resize(
            img,
            w,
            h,
            image::imageops::FilterType::Lanczos3,
        )),
    }
}

/// The backend's options for the one resample every resize
/// here runs: the Lanczos3 convolution.
fn lanczos3_options() -> fast_image_resize::ResizeOptions {
    fast_image_resize::ResizeOptions::new().resize_alg(fast_image_resize::ResizeAlg::Convolution(
        fast_image_resize::FilterType::Lanczos3,
    ))
}

/// [`resize_lanczos3`] of a frame the caller is done with
/// (issue #657). The backend resamples a layout with alpha
/// premultiplied, and it premultiplies a borrowed frame into
/// a second buffer the size of the frame. An owned frame is
/// premultiplied where it sits instead (see
/// [`resample_premultiplying`]): the same three steps on the
/// same values, so the same bytes, without that buffer. A
/// layout without alpha takes [`resize_lanczos3`] as it is,
/// and so does a size that needs no resample: the backend
/// copies those pixels untouched, which a premultiply and a
/// divide around the copy would not.
fn resize_lanczos3_owned(mut img: DynamicImage, w: u32, h: u32) -> DynamicImage {
    use fast_image_resize::IntoImageView;
    let has_alpha_step = img
        .pixel_type()
        .is_some_and(|layout| fast_image_resize::MulDiv::new().is_supported(layout));
    let (sw, sh) = (img.width(), img.height());
    if !has_alpha_step || (w, h) == (sw, sh) || w == 0 || h == 0 || sw == 0 || sh == 0 {
        return resize_lanczos3(&img, w, h);
    }
    let Some(mut dst) = blank_like(&img, w, h) else {
        return resize_lanczos3(&img, w, h);
    };
    if resample_premultiplying(&mut img, &mut dst) {
        dst
    } else {
        resize_lanczos3(&img, w, h)
    }
}

/// [`resize_lanczos3_owned`] of a frame lent for the resize
/// (issue #737): the same bytes, and the frame back as it came.
/// Read through [`resize_lanczos3`], a frame with alpha would be
/// premultiplied into a second buffer the size of the frame. An
/// RGBA8 frame is premultiplied where it sits instead, as an
/// owned frame is, with each row the premultiply changes copied
/// out before it and written back after the resample. The
/// premultiply rounds `c * alpha / 255`, which keeps a pixel at
/// full alpha and a black one as they are, so the rows to copy
/// are those holding any other pixel: none for an opaque frame,
/// and none for a GIF frame, whose pixels are opaque or
/// transparent black. Any other layout, and a size that needs
/// no resample, take [`resize_lanczos3`].
fn resize_lanczos3_lent(img: &mut DynamicImage, w: u32, h: u32) -> DynamicImage {
    let (sw, sh) = (img.width(), img.height());
    let resamples = (w, h) != (sw, sh) && w > 0 && h > 0 && sw > 0 && sh > 0;
    let changing = match &*img {
        DynamicImage::ImageRgba8(rgba) if resamples => ChangingRows::of(rgba),
        _ => return resize_lanczos3(img, w, h),
    };
    let mut dst = DynamicImage::new_rgba8(w, h);
    let done = resample_premultiplying(img, &mut dst);
    if let DynamicImage::ImageRgba8(rgba) = img {
        changing.put_back(rgba);
    }
    if done {
        dst
    } else {
        resize_lanczos3(img, w, h)
    }
}

/// The rows of an RGBA8 frame that a premultiply changes,
/// copied out before it runs (see [`resize_lanczos3_lent`]).
struct ChangingRows {
    /// The bytes in a row of the frame.
    stride: usize,
    /// The row indexes, in order.
    rows: Vec<usize>,
    /// Their bytes, one row after another.
    bytes: Vec<u8>,
}

impl ChangingRows {
    /// Copy out the rows of `frame` holding a pixel that is
    /// neither opaque nor black.
    fn of(frame: &image::RgbaImage) -> Self {
        let mut changing = ChangingRows {
            stride: frame.width() as usize * 4,
            rows: Vec::new(),
            bytes: Vec::new(),
        };
        if changing.stride == 0 {
            return changing;
        }
        let rows = frame.as_raw().chunks_exact(changing.stride);
        for (y, row) in rows.take(frame.height() as usize).enumerate() {
            let changes = row
                .as_chunks::<4>()
                .0
                .iter()
                .any(|&[r, g, b, a]| a != u8::MAX && (r | g | b) != 0);
            if changes {
                changing.rows.push(y);
                changing.bytes.extend_from_slice(row);
            }
        }
        changing
    }

    /// Write the copied rows back into `frame`, the frame they
    /// were copied from.
    fn put_back(&self, frame: &mut image::RgbaImage) {
        if self.rows.is_empty() {
            return;
        }
        let pixels: &mut [u8] = frame;
        for (&y, row) in self.rows.iter().zip(self.bytes.chunks_exact(self.stride)) {
            pixels[y * self.stride..][..self.stride].copy_from_slice(row);
        }
    }
}

/// Resample `src` into `dst` the way the backend resamples a
/// layout with alpha, but premultiplying `src` where it sits
/// instead of into a copy: the backend's own premultiply,
/// its convolution with the alpha step off, and its own
/// divide of the result. `true` when `dst` holds the result.
///
/// On `false` `dst` holds nothing usable and `src` is
/// straight alpha again, so a premultiplied frame never
/// leaves this function. The backend refuses layouts, not
/// pixels, and it refuses the premultiply before it writes
/// a pixel; the callers hand over only layouts it takes.
/// Should the convolution or the divide refuse all the same,
/// `src` is divided back before the caller takes its plain
/// path with it (a test forces this branch). The divide is
/// the premultiply's inverse to within its rounding, so that
/// path, which nothing reaches, can sit a rounding step away
/// from the plain one.
fn resample_premultiplying<S, D>(src: &mut S, dst: &mut D) -> bool
where
    S: fast_image_resize::IntoImageViewMut,
    D: fast_image_resize::IntoImageViewMut,
{
    let mul_div = fast_image_resize::MulDiv::new();
    if mul_div.multiply_alpha_inplace(src).is_err() {
        return false;
    }
    if convolve_lanczos3(src, dst) && mul_div.divide_alpha_inplace(dst).is_ok() {
        return true;
    }
    let _ = mul_div.divide_alpha_inplace(src);
    false
}

/// The backend's Lanczos3 convolution alone, its alpha step
/// off: what it runs between its own premultiply and divide.
/// `false` when it refuses.
fn convolve_lanczos3<S, D>(src: &S, dst: &mut D) -> bool
where
    S: fast_image_resize::IntoImageView,
    D: fast_image_resize::IntoImageViewMut,
{
    #[cfg(test)]
    if CONVOLVE_REFUSES.get() {
        return false;
    }
    fast_image_resize::Resizer::new()
        .resize(src, dst, &lanczos3_options().use_alpha(false))
        .is_ok()
}

#[cfg(test)]
thread_local! {
    /// Makes [`convolve_lanczos3`] refuse on this thread, so
    /// a test can walk the path behind a backend refusal,
    /// which no input reaches.
    static CONVOLVE_REFUSES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
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

/// An aspect crop's window resized to (w, h), in the frame's
/// layout (issues #654, #713): byte for byte
/// [`resize_lanczos3`] of the window cut
/// ([`crate::crop::cut`]).
///
/// Where it can, the resample reads the window straight from
/// the frame, without making that cut (see
/// [`resample_window`]). Everywhere else the window is cut
/// first, the frame is released, and the cut is resized: a
/// layout other than RGB8 and RGBA8, a window that needs no
/// resample, a case [`view_holds_no_more`] turns away, and a
/// backend refusal.
fn resize_lanczos3_cut(img: DynamicImage, window: Window, w: u32, h: u32) -> DynamicImage {
    let (x, y, ww, wh) = window;
    let inside =
        x as u64 + ww as u64 <= img.width() as u64 && y as u64 + wh as u64 <= img.height() as u64;
    let resamples = inside && ww > 0 && wh > 0 && w > 0 && h > 0 && (ww, wh) != (w, h);
    let img = if resamples {
        match resample_window(img, window, w, h) {
            Ok(resized) => return resized,
            Err(img) => img,
        }
    } else {
        img
    };
    let cut = crate::crop::cut(&img, window);
    drop(img);
    resize_lanczos3_owned(cut, w, h)
}

/// The window of an RGB8 or RGBA8 frame resampled to (w, h)
/// through a view of the frame, in the frame's layout; or the
/// frame handed back, for the caller to cut the window first.
///
/// A view ends at the window's edges, so the resample sees
/// exactly the pixels a cut would hold. The backend's crop
/// box is not that: its kernel reads past the box into the
/// frame around it, which is why
/// [`resize_lanczos3_cropped`] (issue #287) differs from the
/// two-step chain and this does not. A test pins both.
///
/// An RGB8 frame is resampled as RGB8, into the result's own
/// buffer. An RGBA8 frame has its window premultiplied where
/// it sits (see [`resample_premultiplying`]): the crop owns
/// the frame and nothing reads it afterwards.
///
/// `Err` hands the frame back as it came: another layout, a
/// case [`view_holds_no_more`] turns away, and a backend
/// refusal, which no input reaches and which leaves an RGBA8
/// window straight again.
fn resample_window(
    img: DynamicImage,
    (x, y, ww, wh): Window,
    w: u32,
    h: u32,
) -> Result<DynamicImage, DynamicImage> {
    let frame_bytes = img.as_bytes().len() as u64;
    match img {
        DynamicImage::ImageRgb8(rgb) if view_holds_no_more(frame_bytes, 3, (ww, wh), (w, h)) => {
            let mut resized = image::RgbImage::new(w, h);
            let done = fast_image_resize::images::CroppedImage::new(&rgb, x, y, ww, wh)
                .is_ok_and(|view| convolve_lanczos3(&view, &mut resized));
            if done {
                Ok(DynamicImage::ImageRgb8(resized))
            } else {
                Err(DynamicImage::ImageRgb8(rgb))
            }
        }
        DynamicImage::ImageRgba8(mut rgba)
            if view_holds_no_more(frame_bytes, 4, (ww, wh), (w, h)) =>
        {
            let mut resized = image::RgbaImage::new(w, h);
            let done = fast_image_resize::images::CroppedImageMut::new(&mut rgba, x, y, ww, wh)
                .is_ok_and(|mut view| resample_premultiplying(&mut view, &mut resized));
            if done {
                Ok(DynamicImage::ImageRgba8(resized))
            } else {
                Err(DynamicImage::ImageRgba8(rgba))
            }
        }
        other => Err(other),
    }
}

/// Whether resampling a window straight from the frame holds
/// no more memory at its peak than the cut-first chain did
/// before issue #654, which cut every layout to RGBA8. That
/// chain peaked at the larger of two moments: the frame
/// beside the RGBA8 cut; and, with the frame released, the
/// cut, the backend's premultiplied copy of it, the backend's
/// working rows and the output. Read from the frame, a crop
/// holds the frame, the working rows and the output at once,
/// and an RGB8 crop's output now has three bytes a pixel
/// where that chain's had four (issue #713), so the bound
/// holds for it with room to spare. That is no more than the second
/// moment when the frame is no larger than the cut and its
/// copy, and no more than the first when the working rows
/// and the output fit in the size of the cut. So an aspect
/// crop, whose window is most of the frame, reads from the
/// frame, and so does a small window reduced well; a small
/// window enlarged, or barely reduced, is cut first and the
/// frame released, as before.
///
/// `frame_bytes` is the frame's buffer and `channels` its
/// bytes per pixel; the bound counts the cut and the output
/// at the four that chain's had. The backend convolves down
/// the columns first, into working rows no wider than the
/// window and as tall as the output.
fn view_holds_no_more(
    frame_bytes: u64,
    channels: u64,
    (ww, wh): (u32, u32),
    (w, h): (u32, u32),
) -> bool {
    let cut = 4 * ww as u64 * wh as u64;
    let output = 4 * w as u64 * h as u64;
    let working = channels * ww as u64 * h as u64;
    frame_bytes <= 2 * cut || working + output <= cut
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
    let opts = lanczos3_options().crop(x as f64, y as f64, w as f64, h as f64);
    match fast_image_resize::Resizer::new().resize(img, &mut dst, &opts) {
        Ok(()) => dst,
        Err(_) => two_step(),
    }
}

/// [`resize_lanczos3`] of a frame the JPEG decoder reduced
/// (issue #653), read through the backend's crop box: the
/// (left, top, width, height) part of the frame that the image
/// fills, which ends a fraction of a pixel short of the frame
/// along the edges where the image's size is not a multiple
/// of the scale. The backend places its kernel by that box,
/// so the picture is resampled at its own scale and keeps its
/// place; resampling the whole frame instead stretches it by
/// up to a pixel and moves the far edge by as much. A backend
/// refusal, which no reduced frame meets, resamples the whole
/// frame.
fn resize_lanczos3_reduced(
    img: DynamicImage,
    (left, top, width, height): (f64, f64, f64, f64),
    w: u32,
    h: u32,
) -> DynamicImage {
    let Some(mut dst) = blank_like(&img, w, h) else {
        return resize_lanczos3_owned(img, w, h);
    };
    let opts = lanczos3_options().crop(left, top, width, height);
    match fast_image_resize::Resizer::new().resize(&img, &mut dst, &opts) {
        Ok(()) => dst,
        Err(_) => resize_lanczos3_owned(img, w, h),
    }
}

/// Pad `img` to exactly `target_w` x `target_h`: fit the image
/// inside the box (the fit-in rules, `no_upscale` cap included),
/// center it, and fill the rest of the canvas: a solid color
/// (the pad segment's own, or the request's `fill(...)`
/// directive, or black), or a blurred cover-scaled copy of the
/// image itself. Also returns where the fitted image sits on
/// the canvas, which the pending focal form follows (issue
/// #465). The blurred fill checks `deadline` before each row
/// (issue #614).
fn apply_pad(
    img: DynamicImage,
    target_w: u32,
    target_h: u32,
    fill: PadFill,
    directive_fill: Option<[u8; 3]>,
    cap: Option<(u32, u32)>,
    deadline: Option<std::time::Instant>,
) -> Result<(DynamicImage, Window), EngineError> {
    /// What fills the canvas around the fitted image.
    enum Backdrop {
        Solid([u8; 3]),
        /// The frame as it came, to scale and blur.
        Blurred(DynamicImage),
    }
    if target_w == 0 || target_h == 0 {
        let whole = (0, 0, img.width(), img.height());
        return Ok((img, whole));
    }
    check_output_size(target_w, target_h, (img.width(), img.height()))?;
    let backdrop = match fill {
        PadFill::Color(c) => Backdrop::Solid(c),
        PadFill::Directive => Backdrop::Solid(directive_fill.unwrap_or([0, 0, 0])),
        // Only the blurred fill reads the frame again after
        // the fit, so only it keeps a copy (issue #656).
        PadFill::Blur => Backdrop::Blurred(img.clone()),
    };
    let fitted = apply_fit_in(
        img,
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
    let mut base = match backdrop {
        Backdrop::Solid(c) => image::RgbaImage::from_raw(
            target_w,
            target_h,
            [c[0], c[1], c[2], 255].repeat(target_w as usize * target_h as usize),
        )
        .expect("canvas size"),
        Backdrop::Blurred(source) => {
            // Ambient bars: the image scaled to cover the whole
            // canvas, then blurred hard enough that only its
            // broad colors survive.
            let sigma = ((target_w.max(target_h) as f32) / 16.0).clamp(2.0, 80.0);
            crate::filters::blur_fast_within(
                &source.resize_to_fill(target_w, target_h, image::imageops::FilterType::Triangle),
                sigma,
                deadline,
            )?
            .into_rgba8()
        }
    };
    let x = (target_w - fitted.width()) / 2;
    let y = (target_h - fitted.height()) / 2;
    let placed = (x, y, fitted.width(), fitted.height());
    place_on(&mut base, fitted, x, y);
    Ok((DynamicImage::ImageRgba8(base), placed))
}

/// Composite `top` over `base` at (x, y): byte for byte
/// `image::imageops::overlay` of `top` as RGBA8 (issue #656).
/// A source-over blend with an opaque top pixel is that
/// pixel, and a layout without an alpha channel converts to
/// opaque pixels only. So a `top` without alpha that lies
/// inside `base` is written over it by rows, with no blend
/// per pixel; one with alpha, or one that would be clipped,
/// takes the crate's overlay.
fn place_on(base: &mut image::RgbaImage, top: DynamicImage, x: u32, y: u32) {
    let (tw, th) = (top.width() as usize, top.height() as usize);
    let inside =
        x as u64 + tw as u64 <= base.width() as u64 && y as u64 + th as u64 <= base.height() as u64;
    if top.color().has_alpha() || !inside || tw == 0 || th == 0 {
        image::imageops::overlay(base, &top.into_rgba8(), x as i64, y as i64);
        return;
    }
    let stride = base.width() as usize * 4;
    let span = x as usize * 4..(x as usize + tw) * 4;
    let rows = base.chunks_exact_mut(stride).skip(y as usize).take(th);
    match top {
        DynamicImage::ImageRgb8(rgb) => {
            for (dst, src) in rows.zip(rgb.as_raw().chunks_exact(tw * 3)) {
                let dst = dst[span.clone()].as_chunks_mut::<4>().0;
                for (d, s) in dst.iter_mut().zip(src.as_chunks::<3>().0) {
                    *d = [s[0], s[1], s[2], 255];
                }
            }
        }
        // Gray, 16-bit and float: the crate's conversion,
        // which is what the overlay read.
        other => {
            let rgba = other.into_rgba8();
            for (dst, src) in rows.zip(rgba.as_raw().chunks_exact(tw * 4)) {
                dst[span.clone()].copy_from_slice(src);
            }
        }
    }
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
    let dims = (img.width(), img.height());
    match fit_in_target(size, adaptive, cap, dpr, dims)? {
        Some((w, h)) if (w, h) != dims => Ok(resize_lanczos3_owned(img, w, h)),
        _ => Ok(img),
    }
}

/// [`apply_fit_in`] of a frame lent for the resize (issue
/// #737), as [`apply_size_lent`] is of a resize: `None` when
/// the fit-in leaves the frame as it is.
fn apply_fit_in_lent(
    frame: &mut DynamicImage,
    size: &ImageSize,
    adaptive: bool,
    cap: Option<(u32, u32)>,
    dpr: f32,
) -> Result<Option<DynamicImage>, EngineError> {
    let dims = (frame.width(), frame.height());
    Ok(match fit_in_target(size, adaptive, cap, dpr, dims)? {
        Some((w, h)) if (w, h) != dims => Some(resize_lanczos3_lent(frame, w, h)),
        _ => None,
    })
}

/// The size a fit-in to `size` leaves a `sw` by `sh` image at,
/// checked against the output bounds; `None` when the box
/// bounds neither axis, so the image passes as it is.
fn fit_in_target(
    size: &ImageSize,
    adaptive: bool,
    cap: Option<(u32, u32)>,
    dpr: f32,
    (sw, sh): (u32, u32),
) -> Result<Option<(u32, u32)>, EngineError> {
    check_nonzero_size(size)?;
    let scaled = ImageSize {
        width: scale_dimension(size.width, dpr),
        height: scale_dimension(size.height, dpr),
        flip_horizontal: size.flip_horizontal,
        flip_vertical: size.flip_vertical,
    };
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
        return Ok(None);
    }
    let w = ((sw as f64 * scale).round() as u32).max(1);
    let h = ((sh as f64 * scale).round() as u32).max(1);
    let (w, h) = match cap {
        Some(c) => cap_dimensions(w, h, c),
        None => (w, h),
    };
    check_output_size(w, h, (sw, sh))?;
    Ok(Some((w, h)))
}

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
