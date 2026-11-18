//! Encoding (issue #345): the encode entry point and
//! the quality-ladder step helpers (the auto-quality
//! walk itself runs in the pipeline), turning the
//! processed image into output bytes per format, and
//! the hand-rolled PNG paths (indexed and Adam7
//! interlaced assembly).
//!
//! engine.rs re-exports the entry point the pipeline uses.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

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
/// the RGBA layout. An image that already has the layout an
/// encoder asks for is lent as it is (issue #665): converting
/// it would only copy the frame.
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
        match self.img {
            DynamicImage::ImageRgb8(rgb) => rgb,
            img => self.rgb.get_or_init(|| img.to_rgb8()),
        }
    }

    fn rgba(&self) -> &image::RgbaImage {
        match self.img {
            DynamicImage::ImageRgba8(rgba) => rgba,
            img => self.rgba.get_or_init(|| img.to_rgba8()),
        }
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
/// other formats ignore it. `avif_speed` and `avif_tiles`
/// apply to AVIF only, and `jxl_speed` to JPEG XL only.
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn encode(
    src: &EncodeBuffers<'_>,
    format: StillFormat,
    quality: Option<u8>,
    fill: Option<[u8; 3]>,
    palette: Option<u16>,
    format_options: FormatOptions,
    avif_speed: u8,
    avif_tiles: AvifTiles,
    jxl_speed: jixel::Speed,
) -> Result<Vec<u8>, EngineError> {
    let img = src.img;
    let (w, h) = (img.width(), img.height());
    let mut buf = Vec::new();
    match format {
        StillFormat::Jpeg => {
            // JPEG has no alpha channel. A fill color composites
            // transparency over that background; without one, the
            // stored RGB values are used as-is. libjpeg-turbo
            // writes the file (see `encode_jpeg`).
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
            return encode_jpeg(rgb, quality.unwrap_or(80), format_options);
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
            // Either way the encode runs in a pool of as many
            // threads as `avif_tiles` asks for, which is what
            // fixes its tile count (issues #671, #688).
            let img = src.integer();
            let quality = quality.unwrap_or(80);
            let pools = AvifPools::get(avif_tiles)?;
            if let Some(rgb) = avif_direct(img) {
                return pools.run(|| encode_avif_rgb8(rgb, avif_speed, quality));
            }
            let out = &mut buf;
            pools.run(move || {
                image::codecs::avif::AvifEncoder::new_with_speed_quality(out, avif_speed, quality)
                    .write_image(img.as_bytes(), w, h, img.color().into())
                    .map_err(|e| EngineError::Encode(e.to_string()))
            })?;
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
            let alpha = img.color().has_alpha();
            let config = jxl_config(quality.unwrap_or(80), jxl_speed, w, h, alpha);
            let encoded = if alpha {
                jixel::encode_image_with_alpha(src.rgba().as_raw(), w as usize, h as usize, &config)
            } else {
                jixel::encode_image(src.rgb().as_raw(), w as usize, h as usize, &config)
            };
            return encoded.map_err(|e| EngineError::Encode(e.to_string()));
        }
    }
    Ok(buf)
}

/// The quality under which the alpha of a small JPEG XL frame
/// is coded as at this quality (issue #696). jixel codes alpha
/// as lossily as the color unless told otherwise, in a squeeze
/// pyramid that halves the frame until it fits 8x8: the step
/// of its coarsest level grows with the distance and halves
/// with every level. Under this quality a frame with few
/// levels can get a coarsest step larger than any alpha it
/// holds. Every sample rounds to 0, the encoder keeps that
/// over the exact alpha because it is smaller, and the frame
/// decodes fully transparent, as the corpus cut-out did at
/// 16x16, at 32x8, and at 16x8 at every quality from 1 to 12.
/// The color still follows the quality asked for; the alpha
/// takes this quality's distance, at which those frames keep
/// it. From this quality up, and in a larger frame, the
/// encoder gets the settings it always had.
const JXL_ALPHA_FLOOR_QUALITY: u8 = 15;

/// The largest frame, in pixels, whose JPEG XL alpha is held
/// to [`JXL_ALPHA_FLOOR_QUALITY`]: 64x64. The pyramid of a
/// larger frame has at least seven levels, so its coarsest
/// step stays under 20 of 255 even at quality 1. The largest
/// frame found losing its alpha under quality 15, a one-pixel
/// ring at half alpha, was 48x32. Larger images keep the
/// bytes they had, where holding their alpha too would cost
/// them 4 to 8% under quality 15.
const JXL_ALPHA_FLOOR_PIXELS: u64 = 64 * 64;

/// The JPEG XL encoder's settings at `quality` and `speed`
/// for a `width` x `height` frame, with an alpha channel when
/// `alpha` is set: the quality's distance, and in a frame of
/// up to [`JXL_ALPHA_FLOOR_PIXELS`] with alpha, under
/// [`JXL_ALPHA_FLOOR_QUALITY`], that quality's distance for
/// the alpha.
pub(in crate::engine) fn jxl_config(
    quality: u8,
    speed: jixel::Speed,
    width: u32,
    height: u32,
    alpha: bool,
) -> jixel::EncodeConfig {
    let config = jixel::EncodeConfig::default()
        .with_quality(f32::from(quality))
        .with_speed(speed);
    let small = u64::from(width) * u64::from(height) <= JXL_ALPHA_FLOOR_PIXELS;
    if alpha && small && quality < JXL_ALPHA_FLOOR_QUALITY {
        let floor = jixel::distance_from_quality(f32::from(JXL_ALPHA_FLOOR_QUALITY));
        return config.with_alpha_distance(floor);
    }
    config
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

/// How many tiles an AVIF is written in, where the frame is
/// large enough for that many: `CRABCROP_AVIF_TILES`, one of
/// 1, 2, 4, 8 and 16 (issues #671, #688). The encoder (ravif)
/// is given no tile count: it takes the number of threads of
/// the rayon pool it runs in, held to what the frame's size
/// allows. Left to rayon's global pool that is the host's
/// core count, so replicas with different core counts
/// rendered one URL to different files; the engine runs it
/// in pools of this many threads instead, whatever the host.
/// More tiles encode faster on a host with the cores for
/// them and compress worse: an 800x533 photograph comes out
/// 1% larger in four tiles than in one and 4% larger in
/// sixteen, a flatter frame 3% and 11%. The count is part of
/// what a URL renders to, so it is in the configuration
/// fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AvifTiles(u8);

impl AvifTiles {
    /// `count` tiles, if it is one of 1, 2, 4, 8 and 16.
    pub fn new(count: u8) -> Option<AvifTiles> {
        matches!(count, 1 | 2 | 4 | 8 | 16).then_some(AvifTiles(count))
    }

    /// The tile count: the threads of each pool an encode
    /// runs in.
    pub fn count(self) -> usize {
        usize::from(self.0)
    }
}

impl Default for AvifTiles {
    /// Four: what a four-core host has always written, and
    /// every host since issue #671.
    fn default() -> Self {
        AvifTiles(4)
    }
}

/// The thread pools AVIF encodes run in, for one tile count:
/// each of exactly that many threads, which is what holds the
/// tile count, and one pool for every that many cores
/// (rounded up), so a host with more cores still runs that
/// many encodes side by side. A count above the host's core
/// count is one pool, of more threads than cores. Built at
/// the first AVIF encode and kept: asking the encoder for a
/// thread count instead (`with_num_threads`) builds and drops
/// a pool inside every encode, which measured 9% slower at
/// 800x533.
struct AvifPools {
    pools: Vec<rayon::ThreadPool>,
    /// The encodes running in each pool.
    running: Vec<AtomicUsize>,
}

/// The pools of each tile count, at the count's exponent: 1,
/// 2, 4, 8 and 16 tiles are slots 0 to 4. A server encodes in
/// one count, the one it is configured with, so it builds one
/// set.
static AVIF_POOLS: [OnceLock<AvifPools>; 5] = [const { OnceLock::new() }; 5];

impl AvifPools {
    fn get(tiles: AvifTiles) -> Result<&'static AvifPools, EngineError> {
        let slot = &AVIF_POOLS[tiles.0.trailing_zeros() as usize];
        if let Some(pools) = slot.get() {
            return Ok(pools);
        }
        // A failure to start the threads fails this encode and
        // is not remembered: the next one tries again.
        let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        let built = Self::build(tiles, cores.div_ceil(tiles.count()))?;
        Ok(slot.get_or_init(|| built))
    }

    /// `count` pools of `tiles` threads each.
    fn build(tiles: AvifTiles, count: usize) -> Result<AvifPools, EngineError> {
        let pools = (0..count)
            .map(|pool| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(tiles.count())
                    .thread_name(move |thread| format!("avif-{pool}-{thread}"))
                    .build()
                    .map_err(|e| EngineError::Encode(format!("avif: no encoder threads: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AvifPools {
            running: pools.iter().map(|_| AtomicUsize::new(0)).collect(),
            pools,
        })
    }

    /// Run `encode` in the pool with the fewest encodes
    /// running. Encodes that share a pool share its threads.
    fn run<R: Send>(&self, encode: impl FnOnce() -> R + Send) -> R {
        struct Running<'a>(&'a AtomicUsize);
        impl Drop for Running<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }
        let pool = (0..self.pools.len())
            .min_by_key(|&pool| self.running[pool].load(Ordering::Relaxed))
            .unwrap_or(0);
        self.running[pool].fetch_add(1, Ordering::Relaxed);
        let _running = Running(&self.running[pool]);
        self.pools[pool].install(encode)
    }
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
/// among them: the pool the caller runs this in decides
/// it), and ravif converts each pixel to YCbCr and
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

/// The PNG chunk CRC: the CRC-32 of the chunk type and data,
/// from crc32fast (issue #737).
fn png_crc32(kind: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc = crc32fast::Hasher::new();
    crc.update(kind);
    crc.update(data);
    crc.finalize()
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

/// Encode JPEG with libjpeg-turbo, through the `turbojpeg`
/// crate (issue #668): three components (Y, Cb, Cr) from
/// 8-bit RGB, behind a JFIF header, as every JPEG this
/// engine has written.
///
/// - `quality` is libjpeg's 1 to 100 scale: the Annex K
///   tables scaled as `jpeg_set_quality` scales them and held
///   to 8 bits, which are the tables the encoder before this
///   one wrote for the same number. A value outside the range
///   is brought into it, as that encoder did.
/// - Chroma is subsampled 4:2:0 unless the request names
///   4:2:2 or 4:4:4. Where it is subsampled, each chroma
///   sample is the average of the pixels it stands for. The
///   encoder before this one kept one of them and dropped the
///   rest, which aliased color edges.
/// - Baseline output carries the standard Huffman tables.
///   Progressive output cannot: libjpeg builds tables from the
///   image for every progressive file. The encoder before
///   this one could build such tables too, and for tiny and
///   flat images it wrote files that zune-jpeg, the decoder
///   this engine reads JPEG with, decoded to the wrong colors
///   (issue #38), which is why they stayed off. libjpeg's own
///   are read back correctly: the tests decode tiny and flat
///   images, baseline and progressive, at every subsampling.
/// - The DCT is libjpeg's accurate integer one, the library's
///   default, which the `turbojpeg` crate has no switch to
///   change. The color conversion, the chroma averaging and
///   the quantization are integer arithmetic too, and the
///   library's C code and its SSE2, AVX2 and Neon code
///   compute all of them to the same values. So the bytes do
///   not depend on the CPU or the architecture: the tests pin
///   them, and compare the vector code with the plain C code.
///
/// libjpeg takes no side past 65,500 pixels, a little under
/// the format's 65,535 that the caller holds the image to. No
/// output comes near either (see `check_final_output`); a
/// side between them is the library's error, returned here.
fn encode_jpeg(
    rgb: &image::RgbImage,
    quality: u8,
    format_options: FormatOptions,
) -> Result<Vec<u8>, EngineError> {
    let failed = |e: turbojpeg::Error| EngineError::Encode(format!("jpeg: {e}"));
    let mut compressor = turbojpeg::Compressor::new().map_err(failed)?;
    compressor
        .set_quality(i32::from(quality.clamp(1, 100)))
        .map_err(failed)?;
    compressor
        .set_subsamp(match format_options.jpeg_subsampling {
            Some(Subsampling::S444) => turbojpeg::Subsamp::None,
            Some(Subsampling::S422) => turbojpeg::Subsamp::Sub2x1,
            // The standing behavior: 4:2:0, also what an
            // explicit subsampling:420 spells.
            Some(Subsampling::S420) | None => turbojpeg::Subsamp::Sub2x2,
        })
        .map_err(failed)?;
    compressor
        .set_progressive(format_options.jpeg_progressive)
        .map_err(failed)?;
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let image = turbojpeg::Image {
        pixels: rgb.as_raw().as_slice(),
        width,
        pitch: width * 3,
        height,
        format: turbojpeg::PixelFormat::RGB,
    };
    // The library grows its own output buffer, from 4 KiB by
    // doubling, and the file is copied out of it at the end:
    // for a moment the encoder holds the file twice, and
    // never a buffer sized for the worst case, which the
    // library puts at 3 to 6 bytes a pixel, as much as the
    // frame or twice it. Handing in a buffer to start from
    // would skip the first doublings, but libjpeg-turbo
    // 3.1.0 leaks such a buffer once the file outgrows it.
    compressor.compress_to_vec(image).map_err(failed)
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

    #[test]
    fn a_frame_in_the_encoders_layout_is_lent_not_copied() {
        // Issue #665: an image that already has the layout an
        // encoder reads is that encoder's buffer.
        let rgb = DynamicImage::ImageRgb8(test_field(9, 5));
        let src = EncodeBuffers::new(&rgb);
        assert!(std::ptr::eq(src.rgb(), rgb.as_rgb8().unwrap()));
        let with_alpha = DynamicImage::ImageRgba8(image::RgbaImage::from_fn(9, 5, |x, y| {
            image::Rgba([(x * 20) as u8, (y * 40) as u8, 7, (x * 28 + y) as u8])
        }));
        let src_alpha = EncodeBuffers::new(&with_alpha);
        assert!(std::ptr::eq(
            src_alpha.rgba(),
            with_alpha.as_rgba8().unwrap()
        ));

        // The layout an image does not have is converted once,
        // to what the conversion always gave: the RGBA layout of
        // an opaque image, the RGB layout of one with alpha (a
        // JPEG without a fill), and both layouts of gray, 16-bit
        // and float images.
        let gray = DynamicImage::ImageLuma8(rgb.to_luma8());
        let others = [
            rgb.clone(),
            with_alpha.clone(),
            gray.clone(),
            DynamicImage::ImageLumaA8(with_alpha.to_luma_alpha8()),
            DynamicImage::ImageRgb16(rgb.to_rgb16()),
            DynamicImage::ImageRgba16(with_alpha.to_rgba16()),
            DynamicImage::ImageLuma16(gray.to_luma16()),
            DynamicImage::ImageRgb32F(rgb.to_rgb32f()),
            DynamicImage::ImageRgba32F(with_alpha.to_rgba32f()),
        ];
        for img in &others {
            let src = EncodeBuffers::new(img);
            let (as_rgb, as_rgba) = (src.rgb(), src.rgba());
            assert!(*as_rgb == img.to_rgb8(), "{:?}", img.color());
            assert!(*as_rgba == img.to_rgba8(), "{:?}", img.color());
            assert!(std::ptr::eq(as_rgb, src.rgb()), "{:?}", img.color());
            assert!(std::ptr::eq(as_rgba, src.rgba()), "{:?}", img.color());
            let own = img.as_bytes().as_ptr();
            let lent_rgb = matches!(img, DynamicImage::ImageRgb8(_));
            let lent_rgba = matches!(img, DynamicImage::ImageRgba8(_));
            assert_eq!(
                as_rgb.as_raw().as_ptr() == own,
                lent_rgb,
                "{:?}",
                img.color()
            );
            assert_eq!(
                as_rgba.as_raw().as_ptr() == own,
                lent_rgba,
                "{:?}",
                img.color()
            );
        }

        // Converting an image to the layout it has copies its
        // buffer as it stands, bytes past the pixels included,
        // so the lent frame is what the copy held even then.
        let padded = image::RgbImage::from_raw(2, 2, vec![7; 20]).unwrap();
        let padded = DynamicImage::ImageRgb8(padded);
        let src = EncodeBuffers::new(&padded);
        assert!(std::ptr::eq(src.rgb(), padded.as_rgb8().unwrap()));
        assert_eq!(src.rgb().as_raw(), padded.to_rgb8().as_raw());
    }

    #[test]
    fn a_jpeg_of_a_flattened_image_does_not_read_the_lent_frame() {
        // Issue #665: an image with alpha flattened onto a fill
        // encodes from the flattened pixels, not from its own
        // buffer, and no fill reads its color channels as they
        // are.
        let with_alpha = DynamicImage::ImageRgba8(image::RgbaImage::from_fn(24, 16, |x, y| {
            image::Rgba([(x * 9) as u8, (y * 13) as u8, 90, (x * 11) as u8])
        }));
        let jpeg = |img: &DynamicImage, fill: Option<[u8; 3]>| {
            encode(
                &EncodeBuffers::new(img),
                StillFormat::Jpeg,
                Some(80),
                fill,
                None,
                FormatOptions::default(),
                DEFAULT_AVIF_SPEED,
                AvifTiles::default(),
                DEFAULT_JXL_SPEED,
            )
            .unwrap()
        };
        let white = [255, 255, 255];
        let flattened = DynamicImage::ImageRgb8(flatten(&with_alpha, white));
        assert!(jpeg(&with_alpha, Some(white)) == jpeg(&flattened, None));
        let channels = DynamicImage::ImageRgb8(with_alpha.to_rgb8());
        assert!(jpeg(&with_alpha, None) == jpeg(&channels, None));
        assert!(jpeg(&with_alpha, None) != jpeg(&with_alpha, Some(white)));
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
    /// AVIF output took before the direct call, asked for the
    /// engine's default tile count.
    fn image_crate_avif(img: &DynamicImage, speed: u8, quality: u8) -> image::ImageResult<Vec<u8>> {
        image_crate_avif_in_tiles(img, speed, quality, AvifTiles::default().count())
    }

    /// The image crate's AVIF encode of `img` in `tiles` tiles,
    /// or as many as the frame's size allows. The crate passes
    /// the thread count it is asked for to ravif, which takes
    /// it for its tile count as well (issue #671).
    fn image_crate_avif_in_tiles(
        img: &DynamicImage,
        speed: u8,
        quality: u8,
        tiles: usize,
    ) -> image::ImageResult<Vec<u8>> {
        let mut out = Vec::new();
        image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut out, speed, quality)
            .with_num_threads(Some(tiles))
            .write_image(
                img.as_bytes(),
                img.width(),
                img.height(),
                img.color().into(),
            )?;
        Ok(out)
    }

    fn engine_avif(img: &DynamicImage, speed: u8, quality: u8) -> Result<Vec<u8>, EngineError> {
        engine_avif_in_tiles(img, speed, quality, AvifTiles::default())
    }

    /// The engine's AVIF encode of `img`, asked for `tiles`.
    fn engine_avif_in_tiles(
        img: &DynamicImage,
        speed: u8,
        quality: u8,
        tiles: AvifTiles,
    ) -> Result<Vec<u8>, EngineError> {
        encode(
            &EncodeBuffers::new(img),
            StillFormat::Avif,
            Some(quality),
            None,
            None,
            FormatOptions::default(),
            speed,
            tiles,
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
        // tile count from the size of the thread pool it runs
        // in, so the bytes hold only while the direct call
        // runs in the pool the image crate's does (issue #671).
        // A smooth frame keeps the encode short.
        let smooth = image::RgbImage::from_fn(256, 128, |x, y| {
            image::Rgb([x as u8, (y * 2) as u8, ((x + y) / 2) as u8])
        });
        same(&DynamicImage::ImageRgb8(smooth), 10, 80);
    }

    /// Smooth frames of 256x256, opaque and with alpha: large
    /// enough for four tiles and no more at speed 10 and
    /// quality 80, where ravif wants 128 pixels a side for a
    /// tile. (Under quality 80 it wants 256, and this frame
    /// is one tile whatever the pool.)
    fn four_tile_frames() -> [DynamicImage; 2] {
        let rgb = image::RgbImage::from_fn(256, 256, |x, y| {
            image::Rgb([x as u8, y as u8, ((x + y) / 2) as u8])
        });
        let rgba = image::RgbaImage::from_fn(256, 256, |x, y| {
            image::Rgba([x as u8, y as u8, ((x + y) / 2) as u8, (255 - x / 2) as u8])
        });
        [DynamicImage::ImageRgb8(rgb), DynamicImage::ImageRgba8(rgba)]
    }

    #[test]
    fn avif_bytes_do_not_follow_the_callers_thread_pool() {
        // Issue #671: ravif writes as many tiles as the rayon
        // pool it runs in has threads. Left to the global pool
        // that was the host's core count, so replicas with
        // different core counts wrote different files. A pool
        // around the call stands in for such a host here: with
        // one thread, with two and with eight the bytes are
        // those of a call from outside any pool, for the direct
        // path and for the image crate's.
        for img in four_tile_frames() {
            let plain = engine_avif(&img, 10, 80).unwrap();
            for threads in [1, 2, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let inside = pool.install(|| engine_avif(&img, 10, 80)).unwrap();
                assert!(
                    inside == plain,
                    "{:?} from a pool of {threads}: {} and {} bytes",
                    img.color(),
                    inside.len(),
                    plain.len()
                );
            }
        }
    }

    /// A smooth opaque frame of 512x288: room for nine tiles at
    /// speed 10 and quality 80, so eight threads write eight
    /// tiles and sixteen write more.
    fn more_than_eight_tile_frame() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_fn(512, 288, |x, y| {
            image::Rgb([(x / 2) as u8, (y * 2 / 3) as u8, ((x + y) / 4) as u8])
        }))
    }

    /// Every tile count the setting takes.
    fn every_tile_count() -> [AvifTiles; 5] {
        [1, 2, 4, 8, 16].map(|count| AvifTiles::new(count).unwrap())
    }

    #[test]
    fn avif_is_written_in_the_tiles_it_is_asked_for() {
        // Issue #688: `CRABCROP_AVIF_TILES` names the count,
        // four unless it says otherwise (issue #671). The count
        // is part of what a URL renders to: a change to the
        // default changes the bytes of every AVIF large enough
        // for two tiles, and needs a new rendering version.
        assert_eq!(AvifTiles::default().count(), 4);
        // Each count on a frame with room for more tiles than
        // the count below it: one, two and four on a frame with
        // room for four, eight and sixteen on one with room for
        // nine. Smooth frames keep the encodes short.
        let [four_tiles, with_alpha] = four_tile_frames();
        let more_than_eight = more_than_eight_tile_frame();
        let frames = [
            &four_tiles,
            &four_tiles,
            &four_tiles,
            &more_than_eight,
            &more_than_eight,
        ];
        let counts = every_tile_count();
        let files: [Vec<u8>; 5] = std::array::from_fn(|i| {
            let (tiles, img) = (counts[i], frames[i]);
            let want = image_crate_avif_in_tiles(img, 10, 80, tiles.count()).unwrap();
            let got = engine_avif_in_tiles(img, 10, 80, tiles).unwrap();
            assert!(
                got == want,
                "{tiles:?}: {} and {} bytes",
                got.len(),
                want.len()
            );
            want
        });
        // What the comparison rests on: ravif takes the thread
        // count it is asked for as its tile count, held to what
        // the frame allows, and on its frame each count is
        // another file than the counts below it (the second
        // frame has room for more than eight tiles, so four are
        // another file there too). A count the engine did not
        // follow would show as another file, and an update that
        // changes how ravif picks its tiles fails here.
        let [one, two, four, eight, sixteen] = &files;
        assert!(one != two && two != four && one != four);
        assert!(eight != sixteen);
        // An image with alpha takes the image crate's encoder,
        // in the same pools.
        let two_tiles = AvifTiles::new(2).unwrap();
        assert!(
            engine_avif_in_tiles(&with_alpha, 10, 80, two_tiles).unwrap()
                == image_crate_avif_in_tiles(&with_alpha, 10, 80, 2).unwrap()
        );
    }

    #[test]
    fn avif_tile_counts_are_the_five_the_setting_takes() {
        for count in 0..=u8::MAX {
            let tiles = AvifTiles::new(count);
            assert_eq!(
                tiles.is_some(),
                [1, 2, 4, 8, 16].contains(&count),
                "{count}"
            );
            if let Some(tiles) = tiles {
                assert_eq!(tiles.count(), usize::from(count));
            }
        }
    }

    #[test]
    fn avif_pools_are_built_for_the_tile_count_and_the_cores() {
        // A server's pools (issue #688): as many threads each as
        // the count, one pool for every that many cores, rounded
        // up, so a count above the cores is one pool. Each count
        // has its own, built once and kept.
        let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        for tiles in every_tile_count() {
            let pools = AvifPools::get(tiles).unwrap();
            assert_eq!(
                pools.pools.len(),
                cores.div_ceil(tiles.count()),
                "{tiles:?}"
            );
            for pool in &pools.pools {
                assert_eq!(pool.current_num_threads(), tiles.count(), "{tiles:?}");
            }
            assert!(std::ptr::eq(pools, AvifPools::get(tiles).unwrap()));
        }
    }

    #[test]
    fn avif_pools_hold_their_tile_count_and_spread_the_encodes() {
        for tiles in every_tile_count() {
            let pools = AvifPools::build(tiles, 3).unwrap();
            assert_eq!(pools.pools.len(), 3);
            for pool in &pools.pools {
                assert_eq!(pool.current_num_threads(), tiles.count());
            }
            // An encode runs on a thread of its pool, where ravif
            // reads the pool's size.
            assert_eq!(pools.run(rayon::current_num_threads), tiles.count());
        }
        let pools = AvifPools::build(AvifTiles::default(), 3).unwrap();
        // Three encodes at once take the three pools, a fourth
        // shares the first, and a pool is free again once its
        // encode returns, an encode that panics included.
        let running = |pools: &AvifPools| {
            pools
                .running
                .iter()
                .map(|n| n.load(Ordering::Relaxed))
                .collect::<Vec<_>>()
        };
        let (entered_tx, entered) = std::sync::mpsc::channel();
        let (release, held) = std::sync::mpsc::channel::<()>();
        let held = std::sync::Mutex::new(held);
        std::thread::scope(|scope| {
            for expected in [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 1]] {
                let entered_tx = entered_tx.clone();
                let (shared, gate) = (&pools, &held);
                scope.spawn(move || {
                    shared.run(|| {
                        entered_tx.send(()).unwrap();
                        let _ = gate.lock().unwrap().recv();
                    });
                });
                entered.recv().unwrap();
                assert_eq!(running(&pools), expected);
            }
            for _ in 0..4 {
                release.send(()).unwrap();
            }
        });
        assert_eq!(running(&pools), [0, 0, 0]);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pools.run(|| panic!("an encode that panics"));
        }));
        assert!(panicked.is_err());
        assert_eq!(running(&pools), [0, 0, 0]);
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

    /// The table [`png_crc32_by_table`] reads: the CRC this
    /// module computed by hand before issue #737, kept as the
    /// oracle for the crate that replaced it.
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

    fn png_crc32_by_table(kind: &[u8; 4], data: &[u8]) -> u32 {
        const TABLE: [u32; 256] = png_crc_table();
        let mut crc = 0xFFFF_FFFFu32;
        for byte in kind.iter().chain(data.iter()) {
            crc = TABLE[((crc ^ u32::from(*byte)) & 0xFF) as usize] ^ (crc >> 8);
        }
        crc ^ 0xFFFF_FFFF
    }

    /// Every chunk type crabcrop writes into a PNG itself: the
    /// interlaced file's (issue #319) and the metadata
    /// `color::inject_png` adds.
    const PNG_CHUNK_TYPES: [&[u8; 4]; 8] = [
        b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND", b"iCCP", b"eXIf", b"iTXt",
    ];

    #[test]
    fn the_png_chunk_crc_matches_the_hand_written_table() {
        // Issue #737: crc32fast replaced the table loop, and the
        // CRC it writes must be the one the table gave: each
        // chunk type crabcrop writes, over random bytes, zeros
        // and ones at every length crc_lengths names.
        let lengths = crate::test_support::crc_lengths();
        let longest = lengths.iter().max().unwrap();
        let random = crate::test_support::Rng::new(737).bytes(longest + 8);
        let mut cases = 0;
        for &len in &lengths {
            let kind = PNG_CHUNK_TYPES[len % PNG_CHUNK_TYPES.len()];
            // The random bytes start at a different alignment
            // for each length.
            let shifted = &random[len % 8..][..len];
            for data in [shifted, &vec![0; len], &vec![0xFF; len]] {
                assert_eq!(
                    png_crc32(kind, data),
                    png_crc32_by_table(kind, data),
                    "{} with {len} bytes",
                    String::from_utf8_lossy(kind)
                );
                cases += 1;
            }
        }
        // Each type on its own, with no data and with a little.
        for kind in PNG_CHUNK_TYPES {
            for data in [&[][..], b"x", &random[..13]] {
                assert_eq!(png_crc32(kind, data), png_crc32_by_table(kind, data));
                cases += 1;
            }
        }
        assert_eq!(cases, 3 * lengths.len() + 3 * PNG_CHUNK_TYPES.len());
        // The value every PNG ends with, from the specification.
        assert_eq!(png_crc32(b"IEND", &[]), 0xAE42_6082);
    }
}
