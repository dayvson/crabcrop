//! Encoding (issue #345): the encode entry point and
//! the quality-ladder step helpers (the auto-quality
//! walk itself runs in the pipeline), turning the
//! processed image into output bytes per format, and
//! the hand-rolled PNG paths (indexed and Adam7
//! interlaced assembly).
//!
//! engine.rs re-exports the entry point the pipeline uses.

use image::{DynamicImage, ExtendedColorType, ImageEncoder};

use super::{EngineError, flatten};
use crate::url::{FormatOptions, ImageFormat, Subsampling};

/// One step down the `max_bytes` quality ladder, planned
/// against the steps that remain (the caller's smaller of the
/// per-format attempt cap and the shared encode pool): the
/// descent from the current quality to the floor is geometric,
/// spaced so the final allowed step lands exactly on quality
/// 1, the grammar's floor. A fixed-ratio walk cannot promise
/// that (from 95 it spends the whole pool reaching 15, and
/// budgets below that encode go unchased; issue #229).
/// The attempt caps still bound the cost per format, AVIF's
/// most tightly, since its encodes are the expensive ones.
pub(in crate::engine) fn next_quality(current: u8, steps_remaining: u32) -> u8 {
    if steps_remaining <= 1 || current <= 2 {
        return 1;
    }
    let frac = f64::from(steps_remaining - 1) / f64::from(steps_remaining);
    let next = (f64::from(current).powf(frac)) as u8;
    next.clamp(1, current - 1)
}

/// How many re-encodes `max_bytes` may attempt per format before
/// returning the smallest result it has.
pub(in crate::engine) fn max_quality_attempts(format: StillFormat) -> u32 {
    match format {
        StillFormat::Avif => 4,
        _ => 12,
    }
}

/// The formats a still image encodes to (issue #480): every
/// output format but the video ones, which only the video path
/// encodes. [`encode`] takes this, so its match has one arm per
/// format and none it cannot reach; the still pipeline turns a
/// video format away before it encodes anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::engine) enum StillFormat {
    Jpeg,
    Png,
    WebP,
    Avif,
    Gif,
    Jxl,
}

impl StillFormat {
    /// The still format `format` names, or `None` for a video
    /// format.
    pub(in crate::engine) fn of(format: ImageFormat) -> Option<Self> {
        match format {
            ImageFormat::Jpeg => Some(StillFormat::Jpeg),
            ImageFormat::Png => Some(StillFormat::Png),
            ImageFormat::WebP => Some(StillFormat::WebP),
            ImageFormat::Avif => Some(StillFormat::Avif),
            ImageFormat::Gif => Some(StillFormat::Gif),
            ImageFormat::Jxl => Some(StillFormat::Jxl),
            ImageFormat::Mp4 | ImageFormat::WebM => None,
        }
    }

    /// Whether the encoder takes a quality: the formats the
    /// Save-Data step and the `max_bytes` ladder can move.
    pub(in crate::engine) fn carries_quality(self) -> bool {
        matches!(
            self,
            StillFormat::Jpeg | StillFormat::Jxl | StillFormat::Avif | StillFormat::WebP
        )
    }
}

impl From<StillFormat> for ImageFormat {
    fn from(format: StillFormat) -> Self {
        match format {
            StillFormat::Jpeg => ImageFormat::Jpeg,
            StillFormat::Png => ImageFormat::Png,
            StillFormat::WebP => ImageFormat::WebP,
            StillFormat::Avif => ImageFormat::Avif,
            StillFormat::Gif => ImageFormat::Gif,
            StillFormat::Jxl => ImageFormat::Jxl,
        }
    }
}

/// Pixel layouts for one encode phase, converted at most once
/// each and shared by every encode of the same image (issue
/// #138). The quality ladders (perceptual bisection, max_bytes)
/// re-encode one image up to a dozen times; converting the
/// full frame inside each attempt copied it every time. The
/// buffers compute lazily: a JPEG-only request never pays for
/// the RGBA layout.
pub(in crate::engine) struct EncodeBuffers<'a> {
    img: &'a DynamicImage,
    rgb: std::cell::OnceCell<image::RgbImage>,
    rgba: std::cell::OnceCell<image::RgbaImage>,
    sixteen: std::cell::OnceCell<DynamicImage>,
}

impl<'a> EncodeBuffers<'a> {
    pub(in crate::engine) fn new(img: &'a DynamicImage) -> Self {
        Self {
            img,
            rgb: std::cell::OnceCell::new(),
            rgba: std::cell::OnceCell::new(),
            sixteen: std::cell::OnceCell::new(),
        }
    }

    fn rgb(&self) -> &image::RgbImage {
        self.rgb.get_or_init(|| self.img.to_rgb8())
    }

    fn rgba(&self) -> &image::RgbaImage {
        self.rgba.get_or_init(|| self.img.to_rgba8())
    }

    /// The image with integer samples, for the encoders that
    /// write the native layout (PNG, plain and interlaced, and
    /// AVIF), none of which takes 32-bit float (issue #464). A
    /// float source (a float TIFF, a Radiance HDR file) converts
    /// to 16 bits per channel, clamped to the 0 to 1 range as the
    /// 8-bit layouts above clamp it; every other image is itself.
    fn integer(&self) -> &DynamicImage {
        let img = self.img;
        match img {
            DynamicImage::ImageRgb32F(_) => self
                .sixteen
                .get_or_init(|| DynamicImage::ImageRgb16(img.to_rgb16())),
            DynamicImage::ImageRgba32F(_) => self
                .sixteen
                .get_or_init(|| DynamicImage::ImageRgba16(img.to_rgba16())),
            _ => img,
        }
    }
}

/// Quantize one RGBA frame with quantizr and wrap it as a
/// `gif::Frame` carrying its own palette (issue #294). This is
/// the GIF write path for both still and animated output; it
/// replaces the image crate's NeuQuant conversion
/// (`Frame::from_rgba_speed`) and mirrors its alpha semantics:
/// alpha is binarized before quantizing (any non-zero alpha is
/// opaque; fully transparent pixels are unified to one color),
/// because GIF carries only 1-bit transparency. The palette
/// entry those pixels land on has alpha 0, and it becomes the
/// frame's transparent index.
pub(in crate::engine) fn gif_frame_quantized(
    rgba: &[u8],
    width: u32,
    height: u32,
    delay: u16,
) -> Result<gif::Frame<'static>, EngineError> {
    let encode_err = |msg: &str| EngineError::Encode(format!("gif: {msg}"));
    if width == 0 || height == 0 {
        return Err(encode_err("zero dimension"));
    }
    let w = u16::try_from(width).map_err(|_| encode_err("width exceeds the 65535 format limit"))?;
    let h =
        u16::try_from(height).map_err(|_| encode_err("height exceeds the 65535 format limit"))?;
    // Binarize alpha on a copy: quantizr reads the buffer as
    // given, and the caller's frame is shared.
    let mut pixels = rgba.to_vec();
    let mut transparent_color: Option<[u8; 3]> = None;
    let (chunks, _) = pixels.as_chunks_mut::<4>();
    for pix in chunks {
        if pix[3] == 0 {
            match transparent_color {
                Some([r, g, b]) => {
                    pix[0] = r;
                    pix[1] = g;
                    pix[2] = b;
                }
                None => transparent_color = Some([pix[0], pix[1], pix[2]]),
            }
        } else {
            pix[3] = 0xFF;
        }
    }
    let image = quantizr::Image::new(&pixels, width as usize, height as usize)
        .map_err(|e| encode_err(&e.to_string()))?;
    let mut opts = quantizr::Options::default();
    opts.set_max_colors(256)
        .map_err(|e| encode_err(&e.to_string()))?;
    let mut result = quantizr::QuantizeResult::quantize(&image, &opts);
    result
        .set_dithering_level(1.0)
        .map_err(|e| encode_err(&e.to_string()))?;
    let palette = result.get_palette();
    let entries = &palette.entries[..palette.count as usize];
    let mut palette_bytes = Vec::with_capacity(entries.len() * 3);
    for entry in entries {
        palette_bytes.extend_from_slice(&[entry.r, entry.g, entry.b]);
    }
    // The transparent pixels quantized to the palette's
    // alpha-0 entry (fully transparent input is one color to
    // quantizr, whatever its RGB was); if clustering left no
    // pure alpha-0 entry, the lowest-alpha entry is the one
    // they remapped to, and it takes the role instead.
    let transparent = if transparent_color.is_some() {
        entries
            .iter()
            .position(|entry| entry.a == 0)
            .or_else(|| {
                entries
                    .iter()
                    .enumerate()
                    .filter(|(_, entry)| entry.a < 0xFF)
                    .min_by_key(|(_, entry)| entry.a)
                    .map(|(i, _)| i)
            })
            .map(|i| i as u8)
    } else {
        None
    };
    let mut indices = vec![0u8; width as usize * height as usize];
    result
        .remap_image(&image, &mut indices)
        .map_err(|e| encode_err(&e.to_string()))?;
    let mut frame = gif::Frame::from_palette_pixels(w, h, indices, palette_bytes, transparent);
    frame.delay = delay;
    // The image crate's encoder disposed every frame to the
    // background; keep that behavior.
    frame.dispose = gif::DisposalMethod::Background;
    Ok(frame)
}

/// Encode `src.img` to `format`. `quality` applies to JPEG,
/// JPEG XL, AVIF, and WebP (where 100 selects the lossless
/// encode). `fill` composites transparency for formats without
/// an alpha channel (only JPEG needs it here). `palette` is
/// the indexed-color budget for PNG output (issue #313); the
/// other formats ignore it.
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn encode(
    src: &EncodeBuffers<'_>,
    format: StillFormat,
    quality: Option<u8>,
    fill: Option<[u8; 3]>,
    palette: Option<u16>,
    format_options: FormatOptions,
    avif_speed: u8,
    jxl_speed: jixel::Speed,
) -> Result<Vec<u8>, EngineError> {
    let img = src.img;
    let (w, h) = (img.width(), img.height());
    let mut buf = Vec::new();
    match format {
        StillFormat::Jpeg => {
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
            enc.set_sampling_factor(match format_options.jpeg_subsampling {
                Some(Subsampling::S444) => jpeg_encoder::SamplingFactor::F_1_1,
                Some(Subsampling::S422) => jpeg_encoder::SamplingFactor::F_2_1,
                // The standing behavior: 4:2:0, also what an
                // explicit subsampling:420 spells.
                Some(Subsampling::S420) | None => jpeg_encoder::SamplingFactor::F_2_2,
            });
            if format_options.jpeg_progressive {
                enc.set_progressive(true);
            }
            enc.encode(
                rgb.as_raw(),
                w as u16,
                h as u16,
                jpeg_encoder::ColorType::Rgb,
            )
            .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        StillFormat::Png => {
            if let Some(max_colors) = palette {
                return encode_png_indexed(src, max_colors, format_options.png_interlace);
            }
            if format_options.png_interlace {
                return encode_png_interlaced(src);
            }
            let img = src.integer();
            image::codecs::png::PngEncoder::new(&mut buf)
                .write_image(img.as_bytes(), w, h, img.color().into())
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        StillFormat::WebP => return encode_webp(src, quality, format_options),
        StillFormat::Avif => {
            // Pure-Rust ravif encoder via the image crate. The
            // speed tier is operator configuration; AVIF encoding
            // is the expensive step in the pipeline by a wide
            // margin, so the tier trades real latency for bytes.
            // Opaque 8-bit RGB skips the image crate and takes
            // the same ravif call with its planes already
            // converted (issue #663).
            let img = src.integer();
            if let Some(rgb) = avif_direct(img) {
                return encode_avif_rgb8(rgb, avif_speed, quality.unwrap_or(80));
            }
            image::codecs::avif::AvifEncoder::new_with_speed_quality(
                &mut buf,
                avif_speed,
                quality.unwrap_or(80),
            )
            .write_image(img.as_bytes(), w, h, img.color().into())
            .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        StillFormat::Gif => {
            let frame = gif_frame_quantized(src.rgba().as_raw(), w, h, 0)?;
            let mut enc = gif::Encoder::new(&mut buf, frame.width, frame.height, &[])
                .map_err(|e| EngineError::Encode(e.to_string()))?;
            enc.write_frame(&frame)
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        StillFormat::Jxl => {
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
    }
    Ok(buf)
}

/// The BT.601 luma weights as the AVIF encoder (ravif)
/// holds them: `f32` constants.
const AVIF_BT601: [f32; 3] = [0.2990, 0.5870, 0.1140];

/// How close to a rounding boundary, in 1/65536 of a sample,
/// a fixed-point sum may come before [`avif_ycbcr`] takes the
/// exact path. The sum is within 1.5 units of the real value
/// (three table entries, each rounded to the nearest unit).
/// ravif's `f32` evaluation is within 2 units of it: three
/// roundings of at most half a unit go into Y, a fourth into
/// the difference from it, the chroma factor (below 0.72)
/// scales those four, and the last rounding adds half a unit.
/// So a sum more than 3.5 units from a boundary rounds the
/// same way on both sides. Over all 16,777,216 colors the
/// farthest sum that rounds the other way is one unit out; 8
/// leaves room, and sends about one color in 200 down the
/// exact path.
const AVIF_FIXED_MARGIN: i32 = 8;

/// Each channel value's share of Y, Cb and Cr in 16.16 fixed
/// point, for [`avif_ycbcr`]. The blue and red tables carry
/// the +128 of their own chroma channel.
struct AvifYcbcrTables {
    red: [[i32; 3]; 256],
    green: [[i32; 3]; 256],
    blue: [[i32; 3]; 256],
}

static AVIF_YCBCR: AvifYcbcrTables = AvifYcbcrTables::new();

impl AvifYcbcrTables {
    /// The tables for `y = kr r + kg g + kb b`,
    /// `cb = (b - y) cb_scale + 128` and
    /// `cr = (r - y) cr_scale + 128`, from the constants as
    /// ravif's `f32` arithmetic holds them.
    const fn new() -> Self {
        let [kr, kg, kb] = AVIF_BT601;
        let cb_scale = (0.5 / (1.0 - kb)) as f64;
        let cr_scale = (0.5 / (1.0 - kr)) as f64;
        let (kr, kg, kb) = (kr as f64, kg as f64, kb as f64);
        let mut tables = AvifYcbcrTables {
            red: [[0; 3]; 256],
            green: [[0; 3]; 256],
            blue: [[0; 3]; 256],
        };
        let mut value = 0;
        while value < 256 {
            let v = value as f64;
            tables.red[value] = [
                avif_fixed(kr * v),
                avif_fixed(-cb_scale * kr * v),
                avif_fixed(cr_scale * (1.0 - kr) * v + 128.0),
            ];
            tables.green[value] = [
                avif_fixed(kg * v),
                avif_fixed(-cb_scale * kg * v),
                avif_fixed(-cr_scale * kg * v),
            ];
            tables.blue[value] = [
                avif_fixed(kb * v),
                avif_fixed(cb_scale * (1.0 - kb) * v + 128.0),
                avif_fixed(-cr_scale * kb * v),
            ];
            value += 1;
        }
        tables
    }
}

/// `value` in 16.16 fixed point, rounded to the nearest unit.
const fn avif_fixed(value: f64) -> i32 {
    let units = value * 65536.0;
    if units < 0.0 {
        (units - 0.5) as i32
    } else {
        (units + 0.5) as i32
    }
}

/// Y, Cb and Cr of one RGB pixel as sums of table entries, in
/// 16.16 fixed point.
#[inline]
fn avif_fixed_sums(r: u8, g: u8, b: u8) -> [i32; 3] {
    let (red, green, blue) = (
        &AVIF_YCBCR.red[r as usize],
        &AVIF_YCBCR.green[g as usize],
        &AVIF_YCBCR.blue[b as usize],
    );
    [
        red[0] + green[0] + blue[0],
        red[1] + green[1] + blue[1],
        red[2] + green[2] + blue[2],
    ]
}

/// A fixed-point sum as an 8-bit sample: rounded to nearest,
/// a half going up as ravif's `round` takes it.
#[inline]
fn avif_fixed_sample(sum: i32) -> u8 {
    ((sum + 0x8000) >> 16).clamp(0, 255) as u8
}

/// One RGB pixel as the full-range 8-bit BT.601 YCbCr triple
/// that ravif's own conversion (`rgb_to_8_bit_ycbcr`) yields,
/// value for value. ravif evaluates it in `f32` with fused
/// multiply-adds and `round`; a build without the FMA target
/// feature (the x86-64 baseline) turns each of those into a
/// libm call, nine a pixel.
///
/// Here the three values are sums of table entries in fixed
/// point. A sum that is not near a rounding boundary rounds
/// the way ravif's value does (see [`AVIF_FIXED_MARGIN`]); one
/// that is near takes [`avif_ycbcr_exact`].
#[inline]
fn avif_ycbcr(r: u8, g: u8, b: u8) -> [u8; 3] {
    let sums = avif_fixed_sums(r, g, b);
    // The fraction's distance from one half, as one unsigned
    // comparison.
    let near_boundary = |sum: i32| {
        (((sum & 0xFFFF) - (0x8000 - AVIF_FIXED_MARGIN)) as u32) <= (2 * AVIF_FIXED_MARGIN) as u32
    };
    if near_boundary(sums[0]) | near_boundary(sums[1]) | near_boundary(sums[2]) {
        return avif_ycbcr_exact(r, g, b);
    }
    sums.map(avif_fixed_sample)
}

/// [`avif_ycbcr`] by ravif's own steps:
/// `y = fma(kb, b, fma(kr, r, kg * g))`, then
/// `cb = fma(b - y, 0.5 / (1 - kb), 128)` and the same for
/// `cr` from `r`, each rounded to the nearest integer. A
/// fused multiply-add is one multiply and one add in `f64`,
/// which holds the product of two `f32` values exactly,
/// rounded once to `f32`. The sums are exact in `f64` too,
/// except that the last one can round there for the darkest
/// colors (a Y below 4), where a second rounding could in
/// principle land elsewhere; it does not, for any color:
/// `avif_ycbcr_matches_ravif` compares every such color with
/// the fused form. The final rounding is `+ 0.5` and a
/// truncating cast, which is `round` for these values.
#[cold]
#[inline(never)]
fn avif_ycbcr_exact(r: u8, g: u8, b: u8) -> [u8; 3] {
    let [kr, kg, kb] = AVIF_BT601;
    let (r, g, b) = (f32::from(r), f32::from(g), f32::from(b));
    let fused = |x: f32, y: f32, z: f32| (f64::from(x) * f64::from(y) + f64::from(z)) as f32;
    let y = fused(kb, b, fused(kr, r, kg * g));
    let cb = fused(b - y, 0.5 / (1.0 - kb), 128.0);
    let cr = fused(r - y, 0.5 / (1.0 - kr), 128.0);
    [y, cb, cr].map(|value| (f64::from(value) + 0.5) as u8)
}

/// The image [`encode_avif_rgb8`] takes: opaque 8-bit RGB
/// with at least one pixel. Alpha, gray, 16-bit and (through
/// [`EncodeBuffers::integer`]) float images keep the image
/// crate's encoder, and so does an empty image, which it
/// refuses in its own words.
fn avif_direct(img: &DynamicImage) -> Option<&image::RgbImage> {
    match img {
        DynamicImage::ImageRgb8(rgb) if rgb.width() > 0 && rgb.height() > 0 => Some(rgb),
        _ => None,
    }
}

/// Encode an opaque 8-bit RGB image to AVIF through ravif
/// directly (issue #663). The image crate's encoder makes
/// this ravif call itself (`encode_rgb`, with the quality
/// for color and alpha, the speed and 8-bit depth set and
/// everything else at ravif's defaults, the thread count
/// among them), and ravif converts each pixel to YCbCr and
/// passes the planes to `encode_raw_planes_8_bit`. The
/// planes handed over here hold those same values (see
/// [`avif_ycbcr`]), so the file is the same, byte for byte:
/// `avif_rgb8_bytes_match_the_image_crate` holds the two
/// together.
fn encode_avif_rgb8(rgb: &image::RgbImage, speed: u8, quality: u8) -> Result<Vec<u8>, EngineError> {
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    // The clamps of `AvifEncoder::new_with_speed_quality`.
    let quality = f32::from(quality.min(100));
    let encoder = ravif::Encoder::new()
        .with_quality(quality)
        .with_alpha_quality(quality)
        .with_speed(speed.min(10))
        .with_bit_depth(ravif::BitDepth::Eight);
    let pixels = rgb.as_raw()[..w * h * 3].as_chunks::<3>().0;
    let planes = pixels.iter().map(|&[r, g, b]| avif_ycbcr(r, g, b));
    let encoded = encoder
        .encode_raw_planes_8_bit(
            w,
            h,
            planes,
            None::<[u8; 0]>,
            ravif::PixelRange::Full,
            ravif::MatrixCoefficients::BT601,
        )
        .map_err(|e| {
            // Wrapped as the image crate wraps it, so a
            // failure reads as it always has.
            let wrapped = image::ImageError::Encoding(image::error::EncodingError::new(
                image::ImageFormat::Avif.into(),
                e,
            ));
            EngineError::Encode(wrapped.to_string())
        })?;
    Ok(encoded.avif_file)
}

/// Encode an indexed-color (palette) PNG (issue #313): the
/// RGBA pixels are quantized to at most `max_colors` palette
/// entries with quantizr (the GIF path's quantizer, dithering
/// at full strength), and the palette becomes the file's PLTE
/// chunk, with the entries' alphas carried in tRNS. Unlike
/// the GIF path, alpha is not binarized: palette entries keep
/// their own alpha, so translucent pixels map to translucent
/// entries. The `png` crate writes the file directly because
/// the image crate's PNG encoder has no indexed color type;
/// the crate is the same one the image crate itself wraps.
pub(in crate::engine) fn encode_png_indexed(
    src: &EncodeBuffers<'_>,
    max_colors: u16,
    interlace: bool,
) -> Result<Vec<u8>, EngineError> {
    let encode_err = |msg: &str| EngineError::Encode(format!("png: {msg}"));
    let img = src.img;
    let (w, h) = (img.width(), img.height());
    let rgba = src.rgba();
    let image = quantizr::Image::new(rgba.as_raw(), w as usize, h as usize)
        .map_err(|e| encode_err(&e.to_string()))?;
    let mut opts = quantizr::Options::default();
    opts.set_max_colors(i32::from(max_colors))
        .map_err(|e| encode_err(&e.to_string()))?;
    let mut result = quantizr::QuantizeResult::quantize(&image, &opts);
    result
        .set_dithering_level(1.0)
        .map_err(|e| encode_err(&e.to_string()))?;
    let palette = result.get_palette();
    let entries = &palette.entries[..palette.count as usize];
    let mut indices = vec![0u8; w as usize * h as usize];
    result
        .remap_image(&image, &mut indices)
        .map_err(|e| encode_err(&e.to_string()))?;
    let mut palette_bytes = Vec::with_capacity(entries.len() * 3);
    let mut trns_bytes = Vec::with_capacity(entries.len());
    for entry in entries {
        palette_bytes.extend_from_slice(&[entry.r, entry.g, entry.b]);
        trns_bytes.push(entry.a);
    }
    // tRNS is omitted when every entry is opaque: the chunk
    // would carry no information.
    let trns = if trns_bytes.iter().any(|a| *a != 0xFF) {
        Some(trns_bytes)
    } else {
        None
    };
    if interlace {
        return assemble_interlaced_png(
            w,
            h,
            png::ColorType::Indexed,
            png::BitDepth::Eight,
            Some(&palette_bytes),
            trns.as_deref(),
            1,
            |y| &indices[y as usize * w as usize..(y as usize + 1) * w as usize],
        );
    }
    let mut buf = Vec::new();
    let mut encoder = png::Encoder::new(&mut buf, w, h);
    encoder.set_color(png::ColorType::Indexed);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_palette(palette_bytes);
    if let Some(trns) = trns {
        encoder.set_trns(trns);
    }
    let mut writer = encoder
        .write_header()
        .map_err(|e| encode_err(&e.to_string()))?;
    writer
        .write_image_data(&indices)
        .map_err(|e| encode_err(&e.to_string()))?;
    writer.finish().map_err(|e| encode_err(&e.to_string()))?;
    Ok(buf)
}

/// The seven Adam7 passes as (first x, first y, x step,
/// y step): the order interlaced PNG data is stored in.
const ADAM7_PASSES: [(u32, u32, u32, u32); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

/// The filtered data stream of an interlaced PNG: Adam7
/// pass rows in storage order, each prefixed with its
/// filter byte. Rows use the Sub filter (each byte stored
/// as the difference from the byte one pixel back), the
/// simple choice that keeps interlaced output compact
/// without a per-row search. `row` returns the full row
/// `y` of the source at `bytes_per_pixel` per pixel.
fn adam7_filtered_stream<'a>(
    w: u32,
    h: u32,
    bytes_per_pixel: usize,
    row: impl Fn(u32) -> &'a [u8],
) -> Vec<u8> {
    let mut stream = Vec::new();
    let mut pass_row: Vec<u8> = Vec::new();
    for (x0, y0, dx, dy) in ADAM7_PASSES {
        if w <= x0 || h <= y0 {
            continue;
        }
        let mut y = y0;
        while y < h {
            let full = row(y);
            pass_row.clear();
            let mut x = x0;
            while x < w {
                let start = x as usize * bytes_per_pixel;
                pass_row.extend_from_slice(&full[start..start + bytes_per_pixel]);
                x += dx;
            }
            stream.push(1u8); // Sub
            for (i, byte) in pass_row.iter().enumerate() {
                let left = if i >= bytes_per_pixel {
                    pass_row[i - bytes_per_pixel]
                } else {
                    0
                };
                stream.push(byte.wrapping_sub(left));
            }
            y += dy;
        }
    }
    stream
}

/// The PNG chunk CRC (CRC-32 over the chunk type and data),
/// table-driven.
const fn png_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

fn png_crc32(kind: &[u8; 4], data: &[u8]) -> u32 {
    const TABLE: [u32; 256] = png_crc_table();
    let mut crc = 0xFFFF_FFFFu32;
    for byte in kind.iter().chain(data.iter()) {
        crc = TABLE[((crc ^ u32::from(*byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn write_png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    out.extend_from_slice(&png_crc32(kind, data).to_be_bytes());
}

/// Assemble an interlaced PNG by hand (issue #319). The
/// png crate's encoder cannot write one: it stamps the
/// IHDR interlace byte but lays filter bytes out at
/// full-width row boundaries, which corrupts the pass
/// stream (proven by the round-trip test). Here the file
/// is signature, IHDR (interlace method 1), the optional
/// palette chunks, one IDAT with the deflated pass stream
/// from [`adam7_filtered_stream`], and IEND.
#[allow(clippy::too_many_arguments)]
fn assemble_interlaced_png<'a>(
    w: u32,
    h: u32,
    color: png::ColorType,
    depth: png::BitDepth,
    palette: Option<&[u8]>,
    trns: Option<&[u8]>,
    bytes_per_pixel: usize,
    row: impl Fn(u32) -> &'a [u8],
) -> Result<Vec<u8>, EngineError> {
    let stream = adam7_filtered_stream(w, h, bytes_per_pixel, row);
    // fdeflate, the same compressor the png crate's Fast
    // setting ships: the previous flate2 level-6 pass cost
    // about 90% of this encode (issue #338).
    let idat = fdeflate::compress_to_vec(&stream);
    let color_byte = match color {
        png::ColorType::Grayscale => 0,
        png::ColorType::Rgb => 2,
        png::ColorType::Indexed => 3,
        png::ColorType::GrayscaleAlpha => 4,
        png::ColorType::Rgba => 6,
    };
    let depth_byte = match depth {
        png::BitDepth::One => 1,
        png::BitDepth::Two => 2,
        png::BitDepth::Four => 4,
        png::BitDepth::Eight => 8,
        png::BitDepth::Sixteen => 16,
    };
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[depth_byte, color_byte, 0, 0, 1]);
    let mut out = Vec::with_capacity(idat.len() + 64);
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    write_png_chunk(&mut out, b"IHDR", &ihdr);
    if let Some(palette) = palette {
        write_png_chunk(&mut out, b"PLTE", palette);
    }
    if let Some(trns) = trns {
        write_png_chunk(&mut out, b"tRNS", trns);
    }
    write_png_chunk(&mut out, b"IDAT", &idat);
    write_png_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

/// Encode an interlaced (Adam7) PNG (issue #319), assembled
/// by hand through [`assemble_interlaced_png`]: neither the
/// image crate's PNG encoder nor the png crate's writer can
/// lay out an interlaced data stream.
fn encode_png_interlaced(src: &EncodeBuffers<'_>) -> Result<Vec<u8>, EngineError> {
    let encode_err = |msg: &str| EngineError::Encode(format!("png: {msg}"));
    let img = src.integer();
    let (w, h) = (img.width(), img.height());
    let color = ExtendedColorType::from(img.color());
    let (png_color, depth, bytes_per_pixel) = match color {
        ExtendedColorType::L8 => (png::ColorType::Grayscale, png::BitDepth::Eight, 1),
        ExtendedColorType::La8 => (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight, 2),
        ExtendedColorType::Rgb8 => (png::ColorType::Rgb, png::BitDepth::Eight, 3),
        ExtendedColorType::Rgba8 => (png::ColorType::Rgba, png::BitDepth::Eight, 4),
        ExtendedColorType::L16 => (png::ColorType::Grayscale, png::BitDepth::Sixteen, 2),
        ExtendedColorType::La16 => (png::ColorType::GrayscaleAlpha, png::BitDepth::Sixteen, 4),
        ExtendedColorType::Rgb16 => (png::ColorType::Rgb, png::BitDepth::Sixteen, 6),
        ExtendedColorType::Rgba16 => (png::ColorType::Rgba, png::BitDepth::Sixteen, 8),
        other => {
            return Err(encode_err(&format!(
                "interlace is not available for color type {other:?}"
            )));
        }
    };
    // PNG stores 16-bit samples big-endian, and the image
    // holds them in native order (issue #464: written as
    // they lay, a little-endian host swapped every sample's
    // bytes). The image crate's own PNG encoder reorders the
    // same way.
    let reordered: Vec<u8>;
    let pixels = if depth == png::BitDepth::Sixteen {
        reordered = img
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|sample| u16::from_ne_bytes(*sample).to_be_bytes())
            .collect();
        &reordered[..]
    } else {
        img.as_bytes()
    };
    let row_len = w as usize * bytes_per_pixel;
    assemble_interlaced_png(w, h, png_color, depth, None, None, bytes_per_pixel, |y| {
        &pixels[y as usize * row_len..(y as usize + 1) * row_len]
    })
}

/// Encode WebP. Quality below 100 is a lossy VP8 encode via
/// libwebp (the `webp` crate, method 4; issue #292):
/// measured about 2.7x faster than the oxideav-webp encode,
/// at slightly higher PSNR; the output bytes differ, which
/// is the accepted cost.
/// The swap is a temporary measure: the daily oxideav watch
/// re-benches this path, and the lossy encode returns to
/// oxideav-webp once a release closes the gap to roughly 1.3x
/// of libwebp method 4 at equal or better size and quality.
/// Quality 100 (or an encode the ladder starts there) keeps
/// the image crate's lossless VP8L encode (libwebp lossless
/// measured about 92x slower, so it stays). The default
/// quality is 80, like the other lossy formats.
fn encode_webp(
    src: &EncodeBuffers<'_>,
    quality: Option<u8>,
    format_options: FormatOptions,
) -> Result<Vec<u8>, EngineError> {
    let img = src.img;
    let q = quality.unwrap_or(80);
    if q >= 100 && !format_options.is_empty() {
        // Options were named, so the lossless encode goes
        // through libwebp too: its method is the effort
        // control the image crate's lossless encoder does not
        // expose. Without options the standing image-crate
        // lossless encode stands (the comment above has the
        // measurement behind that split).
        if img.width() > 16383 || img.height() > 16383 {
            return Err(EngineError::Encode(
                "webp: dimension exceeds the 16383 format limit".into(),
            ));
        }
        let encoder = if img.color().has_alpha() {
            webp::Encoder::from_rgba(src.rgba().as_raw(), img.width(), img.height())
        } else {
            webp::Encoder::from_rgb(src.rgb().as_raw(), img.width(), img.height())
        };
        let mut config = webp::WebPConfig::new().map_err(|_| {
            EngineError::Encode("webp: libwebp rejected its default configuration".into())
        })?;
        config.lossless = 1;
        config.quality = 100.0;
        config.method = format_options.webp_method.map(i32::from).unwrap_or(4);
        let encoded = encoder
            .encode_advanced(&config)
            .map_err(|e| EngineError::Encode(format!("webp: {e:?}")))?;
        return Ok(encoded.to_vec());
    }
    if q >= 100 {
        // The image crate's lossless encoder takes 8-bit samples
        // only: a 16-bit or float image encodes from the shared
        // 8-bit layout the lossy path reads (issue #464).
        let (pixels, color) = match img.color() {
            image::ColorType::L8
            | image::ColorType::La8
            | image::ColorType::Rgb8
            | image::ColorType::Rgba8 => (img.as_bytes(), ExtendedColorType::from(img.color())),
            color if color.has_alpha() => {
                (src.rgba().as_raw().as_slice(), ExtendedColorType::Rgba8)
            }
            _ => (src.rgb().as_raw().as_slice(), ExtendedColorType::Rgb8),
        };
        let mut buf = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut buf)
            .write_image(pixels, img.width(), img.height(), color)
            .map_err(|e| EngineError::Encode(e.to_string()))?;
        return Ok(buf);
    }
    // VP8 frames carry 14-bit dimensions.
    if img.width() > 16383 || img.height() > 16383 {
        return Err(EngineError::Encode(
            "webp: dimension exceeds the 16383 format limit".into(),
        ));
    }
    let encoder = if img.color().has_alpha() {
        webp::Encoder::from_rgba(src.rgba().as_raw(), img.width(), img.height())
    } else {
        webp::Encoder::from_rgb(src.rgb().as_raw(), img.width(), img.height())
    };
    // The lane prototype's configuration: libwebp defaults
    // with the quality set and the method pinned to 4.
    let mut config = webp::WebPConfig::new().map_err(|_| {
        EngineError::Encode("webp: libwebp rejected its default configuration".into())
    })?;
    config.quality = q as f32;
    config.method = format_options.webp_method.map(i32::from).unwrap_or(4);
    if let Some(sns) = format_options.webp_sns {
        config.sns_strength = i32::from(sns);
    }
    let encoded = encoder
        .encode_advanced(&config)
        .map_err(|e| EngineError::Encode(format!("webp: {e:?}")))?;
    Ok(encoded.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{DEFAULT_AVIF_SPEED, DEFAULT_JXL_SPEED};

    /// Noise and a ramp in a checkerboard: flat steps,
    /// gradients, and in the noise colors that take the exact
    /// path.
    fn test_field(w: u32, h: u32) -> image::RgbImage {
        image::RgbImage::from_fn(w, h, |x, y| {
            let mut n = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
            n ^= n >> 13;
            n = n.wrapping_mul(0xC2B2_AE35);
            if (x / 16 + y / 16) % 2 == 0 {
                image::Rgb([(n >> 24) as u8, (n >> 16) as u8, (n >> 8) as u8])
            } else {
                image::Rgb([(x * 3) as u8, (y * 5) as u8, (x + y) as u8])
            }
        })
    }

    /// ravif's conversion as its source spells it (ravif 0.13,
    /// `rgb_to_ycbcr` at 8 bits). `mul_add` is a fused
    /// multiply-add on every target, through the CPU's
    /// instruction or through libm, so this is the value ravif
    /// computes wherever the test runs.
    fn ravif_ycbcr(r: u8, g: u8, b: u8) -> [u8; 3] {
        let matrix = AVIF_BT601;
        let depth = 8;
        let max_value = ((1 << depth) - 1) as f32;
        let scale = max_value / 255.;
        let shift = (max_value * 0.5).round();
        let y = (scale * matrix[2]).mul_add(
            f32::from(b),
            (scale * matrix[0]).mul_add(f32::from(r), scale * matrix[1] * f32::from(g)),
        );
        let cb = f32::from(b)
            .mul_add(scale, -y)
            .mul_add(0.5 / (1. - matrix[2]), shift);
        let cr = f32::from(r)
            .mul_add(scale, -y)
            .mul_add(0.5 / (1. - matrix[0]), shift);
        [y.round() as u8, cb.round() as u8, cr.round() as u8]
    }

    /// Colors whose table sums round the wrong way where they
    /// stand: each lands within one unit of a rounding
    /// boundary, on the other side of it from ravif's value.
    /// Y, Cb and Cr each go wrong for three of them.
    const NEEDS_THE_EXACT_PATH: [[u8; 3]; 9] = [
        [255, 203, 71],
        [255, 127, 129],
        [255, 37, 224],
        [253, 253, 254],
        [253, 253, 244],
        [253, 253, 232],
        [255, 250, 250],
        [255, 240, 240],
        [255, 220, 220],
    ];

    #[test]
    fn avif_ycbcr_matches_ravif() {
        // The whole cube, all 16,777,216 colors, was compared
        // once on both paths and agrees: 17,531 colors need the
        // exact path, every one within one unit of a boundary.
        // That loop takes ten seconds in a debug build, so the
        // suite holds the cube's faces, the dark corner and a
        // sample through the rest, which between them meet
        // hundreds of the colors the tables alone get wrong.
        let check = |r: u8, g: u8, b: u8| {
            let want = ravif_ycbcr(r, g, b);
            assert_eq!(avif_ycbcr(r, g, b), want, "rgb({r}, {g}, {b})");
            assert_eq!(
                avif_ycbcr_exact(r, g, b),
                want,
                "exact path, rgb({r}, {g}, {b})"
            );
        };
        // The faces of the color cube: every color with a
        // channel at either end, where the samples clamp.
        for u in 0..=255 {
            for v in 0..=255 {
                for end in [0, 255] {
                    check(end, u, v);
                    check(u, end, v);
                    check(u, v, end);
                }
            }
        }
        // The darkest colors: every one with a Y below 4 lies
        // in this corner, and for those the exact path's last
        // sum can round in `f64`.
        for r in 0..16 {
            for g in 0..8 {
                for b in 0..40 {
                    check(r, g, b);
                }
            }
        }
        // One color in 31 through the whole cube, which meets
        // every value of every channel.
        for color in (0..1u32 << 24).step_by(31) {
            check((color >> 16) as u8, (color >> 8) as u8, color as u8);
        }
        for [r, g, b] in NEEDS_THE_EXACT_PATH {
            assert_ne!(
                avif_fixed_sums(r, g, b).map(avif_fixed_sample),
                ravif_ycbcr(r, g, b),
                "rgb({r}, {g}, {b}) no longer needs the exact path"
            );
            check(r, g, b);
        }
    }

    /// The image crate's AVIF encode of `img`: the path every
    /// AVIF output took before the direct call.
    fn image_crate_avif(img: &DynamicImage, speed: u8, quality: u8) -> image::ImageResult<Vec<u8>> {
        let mut out = Vec::new();
        image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut out, speed, quality)
            .write_image(
                img.as_bytes(),
                img.width(),
                img.height(),
                img.color().into(),
            )?;
        Ok(out)
    }

    fn engine_avif(img: &DynamicImage, speed: u8, quality: u8) -> Result<Vec<u8>, EngineError> {
        encode(
            &EncodeBuffers::new(img),
            StillFormat::Avif,
            Some(quality),
            None,
            None,
            FormatOptions::default(),
            speed,
            DEFAULT_JXL_SPEED,
        )
    }

    #[test]
    fn avif_rgb8_bytes_match_the_image_crate() {
        let same = |img: &DynamicImage, speed: u8, quality: u8| {
            assert!(avif_direct(img).is_some());
            let got = engine_avif(img, speed, quality).unwrap();
            let want = image_crate_avif(img, speed, quality).unwrap();
            assert!(
                got == want,
                "{}x{} speed {speed} quality {quality}: {} and {} bytes",
                img.width(),
                img.height(),
                got.len(),
                want.len()
            );
        };
        let field = |w: u32, h: u32| DynamicImage::ImageRgb8(test_field(w, h));
        // Every speed the server can be configured with, on a
        // frame small enough for the slow ones: its bytes
        // change at six of the nine steps from one speed to the
        // next, the step from 9 to 10 among them.
        for speed in 1..=10 {
            same(&field(16, 8), speed, 80);
        }
        // The quality range, with both sides of the steps ravif
        // takes at 25, 55, 80 and 82.
        for quality in [1, 25, 26, 55, 56, 79, 80, 81, 82, 100] {
            same(&field(17, 9), DEFAULT_AVIF_SPEED, quality);
        }
        // Sizes from one pixel up, odd ones included, and one
        // past a 64-pixel block on both sides.
        for (w, h) in [(1, 1), (1, 9), (7, 5)] {
            same(&field(w, h), DEFAULT_AVIF_SPEED, 80);
            same(&field(w, h), 10, 60);
        }
        same(&field(65, 65), 10, 60);
        // A frame large enough for two tiles. ravif takes the
        // tile count from the size of the thread pool, so the
        // bytes hold only while the direct call leaves the
        // thread count unset, as the image crate does. A smooth
        // frame keeps the encode short.
        let smooth = image::RgbImage::from_fn(256, 128, |x, y| {
            image::Rgb([x as u8, (y * 2) as u8, ((x + y) / 2) as u8])
        });
        same(&DynamicImage::ImageRgb8(smooth), 10, 80);
    }

    #[test]
    fn avif_goes_direct_for_opaque_rgb8_only() {
        let rgb = test_field(9, 5);
        let rgba = image::RgbaImage::from_fn(9, 5, |x, y| {
            let [r, g, b] = rgb.get_pixel(x, y).0;
            image::Rgba([r, g, b, (x * 28) as u8])
        });
        let opaque_rgba = DynamicImage::ImageRgb8(rgb.clone()).to_rgba8();
        let others = [
            DynamicImage::ImageRgba8(rgba.clone()),
            DynamicImage::ImageRgba8(opaque_rgba),
            DynamicImage::ImageLuma8(DynamicImage::ImageRgb8(rgb.clone()).to_luma8()),
            DynamicImage::ImageLumaA8(DynamicImage::ImageRgba8(rgba.clone()).to_luma_alpha8()),
            DynamicImage::ImageRgb16(DynamicImage::ImageRgb8(rgb.clone()).to_rgb16()),
            DynamicImage::ImageRgba16(DynamicImage::ImageRgba8(rgba.clone()).to_rgba16()),
            DynamicImage::ImageLuma16(DynamicImage::ImageRgb8(rgb.clone()).to_luma16()),
            DynamicImage::ImageRgb32F(DynamicImage::ImageRgb8(rgb.clone()).to_rgb32f()),
            DynamicImage::ImageRgba32F(DynamicImage::ImageRgba8(rgba).to_rgba32f()),
        ];
        for other in &others {
            // What the AVIF arm hands on: a float image as 16
            // bits, every other image as it is.
            let src = EncodeBuffers::new(other);
            let handed = src.integer();
            assert!(avif_direct(handed).is_none(), "{:?}", other.color());
            // The bytes are the image crate's, as they were.
            assert!(
                engine_avif(other, 10, 80).unwrap() == image_crate_avif(handed, 10, 80).unwrap(),
                "{:?}",
                other.color()
            );
        }
        assert!(avif_direct(&DynamicImage::ImageRgb8(rgb)).is_some());
        // An image with no pixels is left to the image crate.
        for (w, h) in [(0, 0), (0, 5), (5, 0)] {
            assert!(avif_direct(&DynamicImage::new_rgb8(w, h)).is_none());
        }
    }

    #[test]
    fn avif_failures_read_as_the_image_crates() {
        let message = |e: image::ImageError| EngineError::Encode(e.to_string()).to_string();
        // An image with no pixels: the image crate refuses it.
        let empty = DynamicImage::new_rgb8(0, 0);
        assert_eq!(
            engine_avif(&empty, DEFAULT_AVIF_SPEED, 80)
                .unwrap_err()
                .to_string(),
            message(image_crate_avif(&empty, DEFAULT_AVIF_SPEED, 80).unwrap_err())
        );
        // A frame the AV1 encoder refuses (one past its widest):
        // the direct call reports it in the image crate's words.
        let wide = DynamicImage::new_rgb8(65_536, 1);
        assert!(avif_direct(&wide).is_some());
        let direct = engine_avif(&wide, DEFAULT_AVIF_SPEED, 80).unwrap_err();
        assert!(matches!(direct, EngineError::Encode(_)), "{direct:?}");
        assert_eq!(
            direct.to_string(),
            message(image_crate_avif(&wide, DEFAULT_AVIF_SPEED, 80).unwrap_err())
        );
    }
}
