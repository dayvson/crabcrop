use std::io::Cursor;
use std::sync::Once;

use image::DynamicImage;

use super::animated::{decode_gif_frames, decode_webp_frames, webp_animation_params};
use super::limits::{MAX_DECODE_ALLOC, MAX_DECODE_DIMENSION};
use super::resize::{fit_in_box, proportional_side, scale_px};
use super::{EngineError, jpeg_scaled, state};
use crate::url::{Dimension, ImageFormat, ImageSize, Operation};

/// Register the JPEG XL decoder with the image crate, once.
///
/// The image crate has no JPEG XL support of its own; the jxl-rs
/// integration hooks its format detection and decoder in globally.
pub(in crate::engine) fn ensure_jxl_decoder() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        jxl_image_rs_integration::register_image_decoding_hook();
    });
}

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
pub(in crate::engine) fn decode_watermarks_within(
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
pub(in crate::engine) fn zune_jpeg_options() -> zune_core::options::DecoderOptions {
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
pub(in crate::engine) fn decode_webp_direct(bytes: &[u8]) -> Option<DynamicImage> {
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
pub(in crate::engine) fn check_jxl_dimensions(bytes: &[u8]) -> Result<(), EngineError> {
    let (width, height) = jxl_dimensions(bytes)
        .ok_or_else(|| EngineError::Decode("JPEG XL header carried no size".to_string()))?;
    check_decoded_dimensions(width as u64, height as u64)
}

/// The dimensions in a JPEG XL header, from the decoder's
/// basic-info event alone: no frame is decoded. `None` when
/// the header does not parse or announces no size.
pub(in crate::engine) fn jxl_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
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
pub(in crate::engine) fn jxl_sample_limit(extra_channels: usize) -> usize {
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
pub(in crate::engine) fn decode_jxl(bytes: &[u8]) -> Result<DynamicImage, EngineError> {
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
pub(in crate::engine) fn detect_format(bytes: &[u8]) -> ImageFormat {
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
pub(in crate::engine) fn is_jxl(bytes: &[u8]) -> bool {
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
pub(in crate::engine) fn is_heic(bytes: &[u8]) -> bool {
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
