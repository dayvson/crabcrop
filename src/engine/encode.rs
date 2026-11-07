//! Encoding (issue #345): the encode entry point and
//! the quality-ladder step helpers (the auto-quality
//! walk itself runs in the pipeline), turning the
//! processed image into output bytes per format, and
//! the hand-rolled PNG paths (indexed and Adam7
//! interlaced assembly).
//!
//! Extracted from engine.rs as a pure code move: every
//! item keeps its name and behavior, and engine.rs
//! re-exports the entry point the pipeline uses, so
//! existing paths do not change.

use image::{DynamicImage, ExtendedColorType, ImageEncoder};

use super::{EngineError, flatten};
use crate::url::{FormatOptions, ImageFormat, Subsampling};

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
pub(in crate::engine) fn max_quality_attempts(format: ImageFormat) -> u32 {
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
pub(in crate::engine) struct EncodeBuffers<'a> {
    img: &'a DynamicImage,
    rgb: std::cell::OnceCell<image::RgbImage>,
    rgba: std::cell::OnceCell<image::RgbaImage>,
}

impl<'a> EncodeBuffers<'a> {
    pub(in crate::engine) fn new(img: &'a DynamicImage) -> Self {
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
    format: ImageFormat,
    quality: Option<u8>,
    fill: Option<[u8; 3]>,
    palette: Option<u16>,
    format_options: FormatOptions,
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
        return encode_webp(src, quality, format_options);
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
        ImageFormat::Png => {
            if let Some(max_colors) = palette {
                return encode_png_indexed(src, max_colors, format_options.png_interlace);
            }
            if format_options.png_interlace {
                return encode_png_interlaced(src);
            }
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
            let frame = gif_frame_quantized(src.rgba().as_raw(), w, h, 0)?;
            let mut enc = gif::Encoder::new(&mut buf, frame.width, frame.height, &[])
                .map_err(|e| EngineError::Encode(e.to_string()))?;
            enc.write_frame(&frame)
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
        ImageFormat::Jxl => unreachable!("JPEG XL returns early in encode"),
        ImageFormat::Mp4 | ImageFormat::WebM => {
            unreachable!("video formats are encoded by process_video, never here")
        }
    }
    Ok(buf)
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
    let img = src.img;
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
    let pixels = img.as_bytes();
    let row_len = w as usize * bytes_per_pixel;
    assemble_interlaced_png(w, h, png_color, depth, None, None, bytes_per_pixel, |y| {
        &pixels[y as usize * row_len..(y as usize + 1) * row_len]
    })
}

/// Encode WebP. Quality below 100 is a lossy VP8 encode via
/// libwebp (the `webp` crate, method 4) since issue #292:
/// measured about 2.7x faster than the oxideav-webp encode it
/// replaced, at slightly higher PSNR; the output bytes
/// differ, which is the accepted cost.
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
