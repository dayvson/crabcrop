//! Applying filter operations to decoded images.
//!
//! [`apply_filters`] runs pixel filters in URL order and collects
//! encoding directives. `Format`, `Quality`, `Fill`, and
//! `MaxBytes` do not touch pixels; they are recorded for the
//! encoder.

use image::DynamicImage;

use crate::url::{CurveChannel, Filter, ImageFormat, MAX_BLUR_SIGMA, Quality};

/// Encoding directives collected from filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeOptions {
    /// Output container override. `None` keeps the detected source format.
    pub format: Option<ImageFormat>,
    /// Encoding quality directive. `None` means the encoder default.
    pub quality: Option<Quality>,
    /// Cap every resize at the source dimensions.
    pub no_upscale: bool,
    /// Background color for flattening transparency at encode time.
    pub fill: Option<[u8; 3]>,
    /// Byte budget for the encoded output, quality permitting.
    pub max_bytes: Option<u64>,
    /// Keep the source color profile: skip the conversion to
    /// sRGB and embed the profile in the output instead.
    pub keep_profile: bool,
}

impl EncodeOptions {
    /// Merge `other` into `self`: directives set in `other` win,
    /// unset ones leave `self` alone. `no_upscale` is sticky:
    /// once requested anywhere, it stays requested.
    pub fn absorb(&mut self, other: EncodeOptions) {
        if other.format.is_some() {
            self.format = other.format;
        }
        if other.quality.is_some() {
            self.quality = other.quality;
        }
        if other.fill.is_some() {
            self.fill = other.fill;
        }
        if other.max_bytes.is_some() {
            self.max_bytes = other.max_bytes;
        }
        self.no_upscale |= other.no_upscale;
        self.keep_profile |= other.keep_profile;
    }
}

/// Watermark images resolved by the server, keyed by the URL
/// the request referenced. The engine only ever sees pixels;
/// fetching happens outside it.
pub type WatermarkImages = std::collections::HashMap<String, DynamicImage>;

/// The sigma a `Filter::Blur` applies with. The parser caps
/// sigma at [`MAX_BLUR_SIGMA`] and rejects non-normal values,
/// but a filter built any other way must not reach the image
/// crate's Gaussian setup with a sigma that panics it (a
/// subnormal trips an `is_normal` assert) or sizes a kernel
/// allocation by an unbounded value. Out-of-contract values
/// clamp to the cap; a non-positive or non-normal value falls
/// back to the smallest normal sigma, an effective no-op.
fn blur_sigma(sigma: f32) -> f32 {
    if sigma.is_normal() && sigma > 0.0 {
        sigma.min(MAX_BLUR_SIGMA)
    } else {
        f32::MIN_POSITIVE
    }
}

/// Apply `filters` in order.
///
/// Returns the transformed image plus the encoding directives.
/// Later directives win when a filter repeats.
#[cfg(test)]
pub fn apply_filters(img: DynamicImage, filters: &[Filter]) -> (DynamicImage, EncodeOptions) {
    apply_filters_filled(img, filters, None)
}

/// [`apply_filters`] with the request's resolved fill color,
/// which arbitrary-angle rotation uses for the corners the
/// turned image leaves uncovered (issue #144).
pub fn apply_filters_filled(
    img: DynamicImage,
    filters: &[Filter],
    fill: Option<[u8; 3]>,
) -> (DynamicImage, EncodeOptions) {
    let mut img = img;
    let mut options = EncodeOptions::default();
    for filter in filters {
        match filter {
            // Normalize to RGB8: encoders universally accept it, and the
            // gray values are preserved (r == g == b).
            Filter::Grayscale => img = DynamicImage::ImageRgb8(img.grayscale().to_rgb8()),
            Filter::Invert => img.invert(),
            Filter::Blur(sigma) => img = img.blur(blur_sigma(*sigma)),
            Filter::Brighten(amount) => img = img.brighten(*amount),
            Filter::Contrast(amount) => {
                let rgb = image::imageops::contrast(&img.into_rgb8(), *amount);
                img = DynamicImage::ImageRgb8(rgb);
            }
            Filter::HueRotate(degrees) => img = img.huerotate(*degrees),
            Filter::Sharpen(amount) => {
                // Unsharp mask as a 3x3 convolution: the center is
                // boosted, the cross neighbors subtracted.
                let a = *amount;
                let kernel: [f32; 9] = [0.0, -a, 0.0, -a, 1.0 + 4.0 * a, -a, 0.0, -a, 0.0];
                let rgb = image::imageops::filter3x3(&img.into_rgb8(), &kernel);
                img = DynamicImage::ImageRgb8(rgb);
            }
            Filter::Saturation(amount) => {
                img = DynamicImage::ImageRgb8(saturate(&img.into_rgb8(), *amount));
            }
            Filter::Rgb(r, g, b) => {
                let mut rgb = img.into_rgb8();
                for p in rgb.pixels_mut() {
                    p[0] = (p[0] as i32 + r).clamp(0, 255) as u8;
                    p[1] = (p[1] as i32 + g).clamp(0, 255) as u8;
                    p[2] = (p[2] as i32 + b).clamp(0, 255) as u8;
                }
                img = DynamicImage::ImageRgb8(rgb);
            }
            Filter::Curve(channel, points) => {
                img = tone_curve(&img, *channel, points);
            }
            Filter::Convolution(values, normalize) => {
                img = convolve(&img, values, *normalize);
            }
            Filter::Noise(amount) => {
                img = add_noise(&img, *amount);
            }
            Filter::Equalize => {
                img = DynamicImage::ImageRgb8(equalize(&img.into_rgb8()));
            }
            Filter::Rotate(degrees) => {
                let normalized = ((degrees % 360.0) + 360.0) % 360.0;
                img = if normalized == 90.0 {
                    img.rotate90()
                } else if normalized == 180.0 {
                    img.rotate180()
                } else if normalized == 270.0 {
                    img.rotate270()
                } else if normalized == 0.0 {
                    img
                } else {
                    rotate_arbitrary(img, normalized, fill)
                };
            }
            Filter::Trim(tolerance) => img = trim(img, *tolerance),
            // Watermarks are applied by the engine, which owns
            // the resolved images; one reaching this function is
            // unresolved by construction and leaves pixels alone.
            // (The engine never routes them here.)
            // Region redaction runs in the engine, which
            // validates the rectangle against the image size
            // in effect at that point. (Same routing.)
            Filter::BlurRegion { .. } | Filter::PixelateRegion { .. } => {}
            Filter::Watermark { .. } => {}
            // Overlays are applied by the engine, which owns
            // the resolved images. (Same routing as above.)
            Filter::Overlay { .. } => {}
            // Background removal is applied by the engine, which
            // owns the mask provider. (Same routing as above.)
            Filter::RemoveBg => {}
            // Studio mode is applied by the engine, which owns
            // the mask provider and the brand color. (Same
            // routing as above.)
            Filter::Studio(_) => {}
            // Upscaling is applied by the engine, which owns the
            // upscale provider. (Same routing.)
            Filter::Upscale(_) => {}
            // Text is drawn by the engine, which owns the
            // configured font renderer. (Same routing.)
            Filter::Text { .. } => {}
            // Face operations run in the engine, which owns the
            // face provider and the shared detection memo. Face
            // padding is a directive the face crop reads from
            // the request as a whole. (Same routing.)
            Filter::BlurFaces | Filter::PixelateFaces | Filter::FacePad(_) => {}
            // Class redaction is engine-routed (it needs the
            // detection provider), like the face filters above.
            Filter::BlurClass(_) | Filter::PixelateClass(_) => {}
            // DPR scales size targets in the engine, which
            // resolves it against the request as a whole (with
            // the client-hint fallback). (Same routing.)
            Filter::Dpr(_) => {}
            Filter::Sepia => img = sepia(&img),
            Filter::Duotone(shadow, highlight) => img = duotone(&img, *shadow, *highlight),
            Filter::Monochrome(color) => img = monochrome(&img, *color),
            Filter::Pixelate(block) => img = pixelate(&img, *block),
            Filter::Flip(direction) => {
                img = match direction {
                    crate::url::FlipDirection::Horizontal => img.fliph(),
                    crate::url::FlipDirection::Vertical => img.flipv(),
                }
            }
            Filter::Radius(radii) => img = rounded_corners(&img, *radii),
            Filter::Border(width, color) => img = draw_border(&img, *width, *color),
            Filter::Mask(shape) => img = mask_shape(&img, *shape),
            Filter::Fill(color) => options.fill = Some(*color),
            Filter::MaxBytes(n) => options.max_bytes = Some(*n),
            Filter::KeepProfile => options.keep_profile = true,
            Filter::Quality(q) => options.quality = Some(*q),
            Filter::NoUpscale => options.no_upscale = true,
            Filter::Format(f) => options.format = Some(*f),
        }
    }
    (img, options)
}

/// The canvas a rotation of a `w` by `h` image lands on:
/// quarter turns swap the axes, an arbitrary angle gets the
/// rotated bounding box. Shared with the engine, which
/// checks the canvas against the output pixel budget before
/// the rotation allocates it (issue #182, audit R2-12).
pub(crate) fn rotated_canvas(w: u32, h: u32, degrees: f32) -> (u32, u32) {
    let normalized = ((degrees % 360.0) + 360.0) % 360.0;
    if normalized == 90.0 || normalized == 270.0 {
        return (h, w);
    }
    if normalized == 0.0 || normalized == 180.0 {
        return (w, h);
    }
    let theta = normalized.to_radians();
    let (sin, cos) = theta.sin_cos();
    let (w, h) = (w as f32, h as f32);
    (
        (w * cos.abs() + h * sin.abs()).ceil().max(1.0) as u32,
        (w * sin.abs() + h * cos.abs()).ceil().max(1.0) as u32,
    )
}

/// Rotate by a non-quarter angle (issue #144): the canvas
/// expands to the rotated bounding box, destination pixels
/// inverse-map to the source with bilinear sampling, and
/// uncovered corners take the background: transparent when
/// the image carries alpha, otherwise the request's fill
/// color, defaulting to white like `pad`. Positive angles
/// turn clockwise, matching the quarter-turn filters.
fn rotate_arbitrary(img: DynamicImage, degrees: f32, fill: Option<[u8; 3]>) -> DynamicImage {
    let has_alpha = img.color().has_alpha();
    let background: [u8; 4] = if has_alpha {
        [0, 0, 0, 0]
    } else {
        let [r, g, b] = fill.unwrap_or([255, 255, 255]);
        [r, g, b, 255]
    };
    let src = img.into_rgba8();
    let (w, h) = (src.width() as f32, src.height() as f32);
    let theta = degrees.to_radians();
    let (sin, cos) = theta.sin_cos();
    let (new_w, new_h) = rotated_canvas(src.width(), src.height(), degrees);
    let (scx, scy) = ((w - 1.0) / 2.0, (h - 1.0) / 2.0);
    let (dcx, dcy) = ((new_w as f32 - 1.0) / 2.0, (new_h as f32 - 1.0) / 2.0);
    let mut out = image::RgbaImage::from_pixel(new_w, new_h, image::Rgba(background));
    let sample = |x: i32, y: i32| -> [f32; 4] {
        if x < 0 || y < 0 || x >= src.width() as i32 || y >= src.height() as i32 {
            [
                background[0] as f32,
                background[1] as f32,
                background[2] as f32,
                background[3] as f32,
            ]
        } else {
            let p = src.get_pixel(x as u32, y as u32).0;
            [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
        }
    };
    for y in 0..new_h {
        for x in 0..new_w {
            let dx = x as f32 - dcx;
            let dy = y as f32 - dcy;
            let sx = cos * dx + sin * dy + scx;
            let sy = -sin * dx + cos * dy + scy;
            let x0 = sx.floor();
            let y0 = sy.floor();
            let fx = sx - x0;
            let fy = sy - y0;
            let (x0, y0) = (x0 as i32, y0 as i32);
            let mut px = [0f32; 4];
            for (wx, xx) in [(1.0 - fx, x0), (fx, x0 + 1)] {
                for (wy, yy) in [(1.0 - fy, y0), (fy, y0 + 1)] {
                    let tap = sample(xx, yy);
                    let weight = wx * wy;
                    for (i, channel) in px.iter_mut().enumerate() {
                        *channel += tap[i] * weight;
                    }
                }
            }
            out.put_pixel(
                x,
                y,
                image::Rgba([
                    px[0].round() as u8,
                    px[1].round() as u8,
                    px[2].round() as u8,
                    px[3].round() as u8,
                ]),
            );
        }
    }
    if has_alpha {
        DynamicImage::ImageRgba8(out)
    } else {
        DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(out).to_rgb8())
    }
}

/// Rec. 601 luma, the same weights the saturation filter
/// and the image crate's grayscale use.
fn luma(r: u8, g: u8, b: u8) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

/// Map the RGB channels of every pixel through `f`, which
/// returns the new channels. Alpha survives untouched when
/// the image carries it; other channel layouts normalize to
/// RGB first, like the sibling color filters.
fn map_rgb(img: &DynamicImage, f: impl Fn(u8, u8, u8) -> [u8; 3]) -> DynamicImage {
    if img.color().has_alpha() {
        let mut rgba = img.to_rgba8();
        for p in rgba.pixels_mut() {
            let [r, g, b] = f(p[0], p[1], p[2]);
            p[0] = r;
            p[1] = g;
            p[2] = b;
        }
        DynamicImage::ImageRgba8(rgba)
    } else {
        let mut rgb = img.to_rgb8();
        for p in rgb.pixels_mut() {
            let [r, g, b] = f(p[0], p[1], p[2]);
            p[0] = r;
            p[1] = g;
            p[2] = b;
        }
        DynamicImage::ImageRgb8(rgb)
    }
}

/// The 256-entry lookup table for one tone curve: inputs at
/// or below the first point's x take its y, inputs at or
/// above the last point's x take its y, and everything
/// between is linearly interpolated between the surrounding
/// points. Interpolation rounds to the nearest value, halves
/// away from zero, in exact integer arithmetic, so the table
/// is the same on every platform.
fn curve_lut(points: &[(u8, u8)]) -> [u8; 256] {
    let mut lut = [0u8; 256];
    let first = points[0];
    let last = points[points.len() - 1];
    for (i, entry) in lut.iter_mut().enumerate() {
        let v = i as i32;
        *entry = if v <= i32::from(first.0) {
            first.1
        } else if v >= i32::from(last.0) {
            last.1
        } else {
            // The segment whose right end is the first point
            // at or past v (x values strictly increase).
            let seg = points.windows(2).find(|w| v <= i32::from(w[1].0)).unwrap();
            let (x0, y0) = (i32::from(seg[0].0), i32::from(seg[0].1));
            let (x1, y1) = (i32::from(seg[1].0), i32::from(seg[1].1));
            let num = (y1 - y0) * (v - x0);
            let den = x1 - x0;
            let delta = if num >= 0 {
                (2 * num + den) / (2 * den)
            } else {
                -((2 * -num + den) / (2 * den))
            };
            (y0 + delta) as u8
        };
    }
    lut
}

/// Remap one channel (or all three, for `rgb`) through the
/// curve's lookup table. Untouched channels pass through the
/// identity table, and alpha is never touched.
pub fn tone_curve(img: &DynamicImage, channel: CurveChannel, points: &[(u8, u8)]) -> DynamicImage {
    let identity: [u8; 256] = std::array::from_fn(|i| i as u8);
    let curved = curve_lut(points);
    let (r_lut, g_lut, b_lut) = match channel {
        CurveChannel::Rgb => (&curved, &curved, &curved),
        CurveChannel::R => (&curved, &identity, &identity),
        CurveChannel::G => (&identity, &curved, &identity),
        CurveChannel::B => (&identity, &identity, &curved),
    };
    map_rgb(img, |r, g, b| {
        [
            r_lut[usize::from(r)],
            g_lut[usize::from(g)],
            b_lut[usize::from(b)],
        ]
    })
}

/// Round to the nearest integer, halves away from zero
/// (the `f64::round` rule), and clamp to the byte range:
/// the finish for the float pixel math (the tone curve's
/// table applies the same rule in exact integer form).
fn round_clamp(value: f64) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

/// Apply a caller-supplied convolution kernel to the RGB
/// channels. `kernel` is row-major; its length picks the
/// size, 9 for 3x3 or 25 for 5x5 (the parser guarantees
/// nothing else reaches here). The edges replicate: a tap
/// outside the frame reads the nearest edge pixel, mirroring
/// the image edges like the blur and sharpen filters built
/// on the image crate's own convolution. With `normalize`
/// set, the kernel is first scaled by the sum of its
/// entries, so an all-ones kernel averages its window; a
/// zero sum has no scale to apply, and the kernel applies
/// as written. Accumulation is `f64`, rounded to nearest
/// with halves away from zero and clamped to 0-255. Alpha
/// is never touched.
pub fn convolve(img: &DynamicImage, kernel: &[f64], normalize: bool) -> DynamicImage {
    let size = if kernel.len() == 9 { 3 } else { 5 };
    let radius = (size / 2) as i32;
    let sum: f64 = kernel.iter().sum();
    let scale = if normalize && sum != 0.0 {
        1.0 / sum
    } else {
        1.0
    };
    let has_alpha = img.color().has_alpha();
    let rgba = img.to_rgba8();
    let (width, height) = (rgba.width() as i32, rgba.height() as i32);
    let mut out = rgba.clone();
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0f64; 3];
            for ky in 0..size as i32 {
                let sy = (y + ky - radius).clamp(0, height - 1) as u32;
                for kx in 0..size as i32 {
                    let k = kernel[(ky as usize) * size + kx as usize] * scale;
                    if k == 0.0 {
                        continue;
                    }
                    let sx = (x + kx - radius).clamp(0, width - 1) as u32;
                    let p = rgba.get_pixel(sx, sy);
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a += k * f64::from(p[c]);
                    }
                }
            }
            let o = out.get_pixel_mut(x as u32, y as u32);
            for (c, a) in acc.iter().enumerate() {
                o[c] = round_clamp(*a);
            }
        }
    }
    if has_alpha {
        DynamicImage::ImageRgba8(out)
    } else {
        DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(out).to_rgb8())
    }
}

/// SplitMix64: a small deterministic 64-bit generator,
/// used here only to turn a content seed into a noise
/// field. Same seed, same stream, on every platform: the
/// arithmetic is all wrapping integer math.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The noise field's seed: an FNV-1a (64-bit) hash of the
/// image's RGBA pixel buffer, with the amount folded in,
/// so the field depends on both what the image is and how
/// much grain was asked for. Same input and amount, same
/// seed, same field, same bytes out: result caching stays
/// sound and no RNG state or clock is involved.
fn noise_seed(bytes: &[u8], amount: u8) -> u64 {
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash ^ u64::from(amount).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Add monochrome film grain: one noise value per pixel,
/// drawn in row-major order and applied equally to the
/// red, green, and blue channels (alpha untouched), so
/// the result reads as grain rather than colour speckle.
/// `amount` is a percent of full scale: the value is
/// uniform over plus or minus `amount * 255 / 100` (the
/// amplitude, rounded to the nearest step), so 100 swings
/// a channel by up to 255 in either direction before the
/// clamp to 0-255. The field is deterministic (see
/// [`noise_seed`]), and amount 0 returns the image
/// untouched, byte for byte.
pub fn add_noise(img: &DynamicImage, amount: u8) -> DynamicImage {
    if amount == 0 {
        return img.clone();
    }
    let has_alpha = img.color().has_alpha();
    let mut rgba = img.to_rgba8();
    let seed = noise_seed(rgba.as_raw(), amount);
    let mut rng = SplitMix64(seed);
    let amplitude = (i32::from(amount) * 255 + 50) / 100;
    let span = (2 * amplitude + 1) as u64;
    for p in rgba.pixels_mut() {
        let delta = (rng.next_u64() % span) as i32 - amplitude;
        for c in 0..3 {
            p[c] = (i32::from(p[c]) + delta).clamp(0, 255) as u8;
        }
    }
    if has_alpha {
        DynamicImage::ImageRgba8(rgba)
    } else {
        DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(rgba).to_rgb8())
    }
}

/// The classic warm sepia tone (the standard sepia matrix).
pub fn sepia(img: &DynamicImage) -> DynamicImage {
    map_rgb(img, |r, g, b| {
        let (r, g, b) = (r as f32, g as f32, b as f32);
        [
            (0.393 * r + 0.769 * g + 0.189 * b)
                .round()
                .clamp(0.0, 255.0) as u8,
            (0.349 * r + 0.686 * g + 0.168 * b)
                .round()
                .clamp(0.0, 255.0) as u8,
            (0.272 * r + 0.534 * g + 0.131 * b)
                .round()
                .clamp(0.0, 255.0) as u8,
        ]
    })
}

/// Map shadows to `shadow` and highlights to `highlight` by
/// luminance: black renders exactly `shadow`, white exactly
/// `highlight`, and everything between is the straight
/// interpolation at the pixel's luminance.
pub fn duotone(img: &DynamicImage, shadow: [u8; 3], highlight: [u8; 3]) -> DynamicImage {
    map_rgb(img, |r, g, b| {
        let t = luma(r, g, b) / 255.0;
        let mut out = [0u8; 3];
        for (c, o) in out.iter_mut().enumerate() {
            *o = (shadow[c] as f32 + (highlight[c] as f32 - shadow[c] as f32) * t)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        out
    })
}

/// Single-hue rendering in `color`: each pixel's luminance
/// scaled by the color, so full luminance renders exactly
/// `color` and black stays black.
pub fn monochrome(img: &DynamicImage, color: [u8; 3]) -> DynamicImage {
    map_rgb(img, |r, g, b| {
        let l = luma(r, g, b);
        let mut out = [0u8; 3];
        for (c, o) in out.iter_mut().enumerate() {
            *o = (l * color[c] as f32 / 255.0).round().clamp(0.0, 255.0) as u8;
        }
        out
    })
}

/// Mosaic the image into `block`-pixel squares anchored at
/// the top-left, each painted with the average of its pixels
/// (alpha included when the image carries it). A block of 1
/// is the identity; blocks at the right and bottom edges are
/// as large as the frame allows.
pub fn pixelate(img: &DynamicImage, block: u32) -> DynamicImage {
    if block <= 1 {
        return img.clone();
    }
    if img.color().has_alpha() {
        let mut rgba = img.to_rgba8();
        average_blocks(
            &mut rgba,
            block,
            |p| [p[0], p[1], p[2], p[3]],
            |p, v| {
                p[0] = v[0];
                p[1] = v[1];
                p[2] = v[2];
                p[3] = v[3];
            },
        );
        DynamicImage::ImageRgba8(rgba)
    } else {
        let mut rgb = img.to_rgb8();
        average_blocks(
            &mut rgb,
            block,
            |p| [p[0], p[1], p[2], 0],
            |p, v| {
                p[0] = v[0];
                p[1] = v[1];
                p[2] = v[2];
            },
        );
        DynamicImage::ImageRgb8(rgb)
    }
}

/// Shared block averaging for `pixelate`, over either pixel
/// buffer: `get` reads the channels that take part (the
/// fourth is an alpha slot, 0 for RGB), `set` writes the
/// averaged pixel back.
fn average_blocks<P: image::Pixel>(
    img: &mut image::ImageBuffer<P, Vec<P::Subpixel>>,
    block: u32,
    get: impl Fn(&P) -> [u8; 4],
    set: impl Fn(&mut P, [u8; 4]),
) {
    let (w, h) = (img.width(), img.height());
    for by in (0..h).step_by(block as usize) {
        for bx in (0..w).step_by(block as usize) {
            let (bw, bh) = (block.min(w - bx), block.min(h - by));
            let mut sums = [0u64; 4];
            for y in by..by + bh {
                for x in bx..bx + bw {
                    let v = get(&img.get_pixel(x, y).clone());
                    for c in 0..4 {
                        sums[c] += v[c] as u64;
                    }
                }
            }
            let n = (bw * bh) as f32;
            let avg = [
                (sums[0] as f32 / n).round() as u8,
                (sums[1] as f32 / n).round() as u8,
                (sums[2] as f32 / n).round() as u8,
                (sums[3] as f32 / n).round() as u8,
            ];
            for y in by..by + bh {
                for x in bx..bx + bw {
                    set(img.get_pixel_mut(x, y), avg);
                }
            }
        }
    }
}

/// Apply `f` to the rectangle (x, y, w, h) of `img` and
/// composite the result back over an RGBA copy; pixels
/// outside the rectangle are untouched, and alpha survives.
/// The caller guarantees the rectangle is in bounds.
fn apply_to_region(
    img: &DynamicImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    f: impl Fn(&DynamicImage) -> DynamicImage,
) -> DynamicImage {
    let mut base = img.to_rgba8();
    let region = img.crop_imm(x, y, w, h);
    let done = f(&region).to_rgba8();
    image::imageops::overlay(&mut base, &done, x as i64, y as i64);
    DynamicImage::ImageRgba8(base)
}

/// Gaussian-blur one rectangle of `img` (explicit region
/// redaction: plates, screens, documents). The sigma scales
/// with the region, min(w, h) / 8 clamped to [1, 30], so a
/// tall thin region blurs to its own scale.
pub fn blur_region(img: &DynamicImage, x: u32, y: u32, w: u32, h: u32) -> DynamicImage {
    let sigma = (w.min(h) as f32 / 8.0).clamp(1.0, 30.0);
    apply_to_region(img, x, y, w, h, |region| region.blur(sigma))
}

/// Pixelate one rectangle of `img`, with the block size
/// derived from the region: min(w, h) / 10, at least 2 px,
/// using the same block averaging as the `pixelate` filter.
pub fn pixelate_region(img: &DynamicImage, x: u32, y: u32, w: u32, h: u32) -> DynamicImage {
    let block = (w.min(h) / 10).max(2);
    apply_to_region(img, x, y, w, h, |region| pixelate(region, block))
}

/// Round the corners of `img` by the given radii (CSS order:
/// top-left, top-right, bottom-right, bottom-left). Radii
/// larger than the frame shrink proportionally so adjacent
/// corners never overlap (the CSS rule). Only the alpha
/// channel changes; coverage is antialiased over one pixel
/// from the signed distance to the corner arc, multiplying
/// the pixel's existing alpha.
pub fn rounded_corners(img: &DynamicImage, radii: [u32; 4]) -> DynamicImage {
    if radii.iter().all(|&r| r == 0) {
        return img.clone();
    }
    let (w, h) = (img.width(), img.height());
    let (wf, hf) = (w as f32, h as f32);
    let mut rs = [
        radii[0] as f32,
        radii[1] as f32,
        radii[2] as f32,
        radii[3] as f32,
    ];
    let mut f = 1.0f32;
    for (edge, sum) in [
        (wf, rs[0] + rs[1]),
        (hf, rs[1] + rs[2]),
        (wf, rs[2] + rs[3]),
        (hf, rs[3] + rs[0]),
    ] {
        if sum > edge {
            f = f.min(edge / sum);
        }
    }
    for r in &mut rs {
        *r *= f;
    }
    // (radius, arc center x, arc center y, square x range, square y range)
    let corners = [
        (
            rs[0],
            rs[0],
            rs[0],
            0..(rs[0].ceil() as u32).min(w),
            0..(rs[0].ceil() as u32).min(h),
        ),
        (
            rs[1],
            wf - rs[1],
            rs[1],
            w - (rs[1].ceil() as u32).min(w)..w,
            0..(rs[1].ceil() as u32).min(h),
        ),
        (
            rs[2],
            wf - rs[2],
            hf - rs[2],
            w - (rs[2].ceil() as u32).min(w)..w,
            h - (rs[2].ceil() as u32).min(h)..h,
        ),
        (
            rs[3],
            rs[3],
            hf - rs[3],
            0..(rs[3].ceil() as u32).min(w),
            h - (rs[3].ceil() as u32).min(h)..h,
        ),
    ];
    let mut rgba = img.to_rgba8();
    for (r, ax, ay, xs, ys) in corners {
        if r <= 0.0 {
            continue;
        }
        for y in ys {
            for x in xs.clone() {
                let d = ((x as f32 + 0.5 - ax).powi(2) + (y as f32 + 0.5 - ay).powi(2)).sqrt() - r;
                let coverage = (0.5 - d).clamp(0.0, 1.0);
                if coverage < 1.0 {
                    let px = rgba.get_pixel_mut(x, y);
                    px[3] = (px[3] as f32 * coverage).round() as u8;
                }
            }
        }
    }
    DynamicImage::ImageRgba8(rgba)
}

/// Keep only the inscribed shape of `img`; everything outside
/// becomes transparent. The edge is antialiased over one pixel
/// from the signed distance to the boundary (exact for
/// circles, a close approximation for ellipses), multiplying
/// the pixel's existing alpha.
pub fn mask_shape(img: &DynamicImage, shape: crate::url::MaskShape) -> DynamicImage {
    let (w, h) = (img.width(), img.height());
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let (rx, ry) = match shape {
        crate::url::MaskShape::Circle => {
            let r = cx.min(cy);
            (r, r)
        }
        crate::url::MaskShape::Ellipse => (cx, cy),
    };
    let feather_scale = rx.min(ry);
    let mut rgba = img.to_rgba8();
    for y in 0..h {
        for x in 0..w {
            let nx = (x as f32 + 0.5 - cx) / rx;
            let ny = (y as f32 + 0.5 - cy) / ry;
            let d = ((nx * nx + ny * ny).sqrt() - 1.0) * feather_scale;
            let coverage = (0.5 - d).clamp(0.0, 1.0);
            if coverage <= 0.0 {
                rgba.get_pixel_mut(x, y)[3] = 0;
            } else if coverage < 1.0 {
                let px = rgba.get_pixel_mut(x, y);
                px[3] = (px[3] as f32 * coverage).round() as u8;
            }
        }
    }
    DynamicImage::ImageRgba8(rgba)
}

/// Paint an opaque band of `color`, `width` pixels wide,
/// inside the frame. Dimensions do not change; a width of
/// half the smaller dimension or more paints the whole image.
pub fn draw_border(img: &DynamicImage, width: u32, color: [u8; 3]) -> DynamicImage {
    let (w, h) = (img.width(), img.height());
    let mut rgba = img.to_rgba8();
    let band = image::Rgba([color[0], color[1], color[2], 255]);
    let bh = width.min(h);
    let bw = width.min(w);
    for y in 0..bh {
        for x in 0..w {
            rgba.put_pixel(x, y, band);
        }
    }
    for y in h - bh..h {
        for x in 0..w {
            rgba.put_pixel(x, y, band);
        }
    }
    for y in bh..h - bh {
        for x in 0..bw {
            rgba.put_pixel(x, y, band);
        }
        for x in w - bw..w {
            rgba.put_pixel(x, y, band);
        }
    }
    DynamicImage::ImageRgba8(rgba)
}

/// Crop away the uniform border: keep the bounding box of
/// pixels that differ from the top-left corner color by more
/// than `tolerance` in any channel (alpha included). A fully
/// uniform image is returned unchanged, never emptied.
/// Blur every detected face region in `img`. The blur sigma
/// scales with each face's size, so the redaction holds up at
/// any resolution. Pixels outside the face boxes are untouched.
pub fn blur_faces(img: &DynamicImage, faces: &[crate::ai::FaceBox]) -> DynamicImage {
    redact_faces(img, faces, |region, face| {
        let sigma = (face.w.min(face.h) as f32 / 6.0).max(1.0);
        region.blur(sigma)
    })
}

/// Pixelate every detected face region in `img`: each face is
/// resampled down to a handful of blocks and back with nearest
/// neighbor. Pixels outside the face boxes are untouched.
pub fn pixelate_faces(img: &DynamicImage, faces: &[crate::ai::FaceBox]) -> DynamicImage {
    redact_faces(img, faces, |region, face| {
        let block = (face.w / 10).max(1);
        let (sw, sh) = (
            (region.width() / block).max(1),
            (region.height() / block).max(1),
        );
        region
            .resize_exact(sw, sh, image::imageops::FilterType::Nearest)
            .resize_exact(
                region.width(),
                region.height(),
                image::imageops::FilterType::Nearest,
            )
    })
}

/// Blur every detection's box in `img`, through the
/// region redaction mechanics (strength derived from the
/// box size). Boxes are clamped to the image; empty or
/// fully out-of-frame boxes are skipped.
pub fn blur_detections(img: &DynamicImage, detections: &[crate::ai::Detection]) -> DynamicImage {
    redact_detections(img, detections, blur_region)
}

/// Pixelate every detection's box in `img`, through the
/// region redaction mechanics.
pub fn pixelate_detections(
    img: &DynamicImage,
    detections: &[crate::ai::Detection],
) -> DynamicImage {
    redact_detections(img, detections, pixelate_region)
}

fn redact_detections(
    img: &DynamicImage,
    detections: &[crate::ai::Detection],
    redact: fn(&DynamicImage, u32, u32, u32, u32) -> DynamicImage,
) -> DynamicImage {
    let mut out = img.clone();
    for det in detections {
        let x = det.x.min(img.width());
        let y = det.y.min(img.height());
        let w = det.w.min(img.width() - x);
        let h = det.h.min(img.height() - y);
        if w == 0 || h == 0 {
            continue;
        }
        out = redact(&out, x, y, w, h);
    }
    out
}

/// Apply `redact` to each face region of `img` and composite
/// the results back. Face boxes are clamped to the image;
/// empty or fully out-of-frame boxes are skipped.
fn redact_faces(
    img: &DynamicImage,
    faces: &[crate::ai::FaceBox],
    redact: impl Fn(&DynamicImage, &crate::ai::FaceBox) -> DynamicImage,
) -> DynamicImage {
    if faces.is_empty() {
        return img.clone();
    }
    let mut base = img.to_rgba8();
    for face in faces {
        let x = face.x.min(img.width());
        let y = face.y.min(img.height());
        let w = face.w.min(img.width() - x);
        let h = face.h.min(img.height() - y);
        if w == 0 || h == 0 {
            continue;
        }
        let region = img.crop_imm(x, y, w, h);
        let done = redact(&region, face).to_rgba8();
        image::imageops::overlay(&mut base, &done, x as i64, y as i64);
    }
    DynamicImage::ImageRgba8(base)
}

fn trim(img: DynamicImage, tolerance: u8) -> DynamicImage {
    let rgba = img.to_rgba8();
    let corner = *rgba.get_pixel(0, 0);
    let tol = tolerance as i32;
    let differs =
        |p: &image::Rgba<u8>| (0..4).any(|c| (p[c] as i32 - corner[c] as i32).abs() > tol);
    let (mut min_x, mut min_y) = (rgba.width(), rgba.height());
    let (mut max_x, mut max_y) = (0u32, 0u32);
    let mut found = false;
    for (x, y, p) in rgba.enumerate_pixels() {
        if differs(p) {
            found = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !found {
        return img;
    }
    DynamicImage::ImageRgba8(
        image::imageops::crop_imm(&rgba, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
            .to_image(),
    )
}

/// Scale each pixel's distance from its luma by `amount`:
/// 0 collapses to gray, 1 is the identity.
fn saturate(img: &image::RgbImage, amount: f32) -> image::RgbImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let luma = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        for c in 0..3 {
            p[c] = (luma + (p[c] as f32 - luma) * amount).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Histogram equalization, independently per channel.
fn equalize(img: &image::RgbImage) -> image::RgbImage {
    let total = (img.width() * img.height()) as u64;
    let mut maps: [[u8; 256]; 3] = [[0; 256]; 3];
    for c in 0..3 {
        let mut hist = [0u64; 256];
        for p in img.pixels() {
            hist[p[c] as usize] += 1;
        }
        let mut cdf = [0u64; 256];
        let mut acc = 0u64;
        for i in 0..256 {
            acc += hist[i];
            cdf[i] = acc;
        }
        let cdf_min = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0);
        for i in 0..256 {
            maps[c][i] = if total == cdf_min {
                i as u8
            } else {
                // Values below the minimum present value have a
                // CDF of 0 and map to 0.
                ((cdf[i].saturating_sub(cdf_min) as f64 * 255.0 / (total - cdf_min) as f64).round())
                    as u8
            };
        }
    }
    let mut out = img.clone();
    for p in out.pixels_mut() {
        for c in 0..3 {
            p[c] = maps[c][p[c] as usize];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn alpha_at(img: &DynamicImage, x: u32, y: u32) -> u8 {
        img.to_rgba8().get_pixel(x, y)[3]
    }

    fn opaque(w: u32, h: u32, color: [u8; 3]) -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(w, h);
        for p in img.as_mut_rgb8().unwrap().pixels_mut() {
            *p = Rgb(color);
        }
        img
    }

    #[test]
    fn blur_at_the_sigma_cap_still_renders() {
        let img = opaque(16, 16, [120, 30, 200]);
        let (out, _) = apply_filters(img, &[Filter::Blur(MAX_BLUR_SIGMA)]);
        assert_eq!((out.width(), out.height()), (16, 16));
    }

    #[test]
    fn blur_sigma_guard_tames_out_of_contract_values() {
        assert_eq!(blur_sigma(2.5), 2.5);
        assert_eq!(blur_sigma(1e9), MAX_BLUR_SIGMA);
        assert_eq!(blur_sigma(0.0), f32::MIN_POSITIVE);
        assert_eq!(blur_sigma(f32::NAN), f32::MIN_POSITIVE);
        // A Filter built without the parser must not panic the
        // image crate (subnormal sigma) or size a kernel by an
        // unbounded one.
        let img = opaque(8, 8, [10, 20, 30]);
        let (out, _) = apply_filters(img.clone(), &[Filter::Blur(1e-45)]);
        assert_eq!((out.width(), out.height()), (8, 8));
        let (out, _) = apply_filters(img, &[Filter::Blur(1e9)]);
        assert_eq!((out.width(), out.height()), (8, 8));
    }

    #[test]
    fn radius_cuts_corners_and_antialiases() {
        let img = rounded_corners(&opaque(100, 100, [255, 0, 0]), [20, 20, 20, 20]);
        assert_eq!(alpha_at(&img, 0, 0), 0);
        assert_eq!(alpha_at(&img, 50, 50), 255);
        let edge = alpha_at(&img, 5, 6);
        assert!(edge > 0 && edge < 255, "antialiased edge, got {edge}");
        // Color channels are untouched.
        assert_eq!(img.into_rgba8().get_pixel(50, 50).0, [255, 0, 0, 255]);
    }

    #[test]
    fn radius_multiplies_existing_alpha() {
        let mut img = DynamicImage::new_rgba8(40, 40);
        for p in img.as_mut_rgba8().unwrap().pixels_mut() {
            *p = image::Rgba([10, 20, 30, 100]);
        }
        let img = rounded_corners(&img, [10, 10, 10, 10]);
        assert_eq!(alpha_at(&img, 20, 20), 100);
        assert_eq!(alpha_at(&img, 0, 0), 0);
    }

    #[test]
    fn radius_is_per_corner() {
        let img = rounded_corners(&opaque(100, 100, [0, 0, 255]), [20, 0, 0, 0]);
        assert_eq!(alpha_at(&img, 0, 0), 0);
        assert_eq!(alpha_at(&img, 99, 0), 255);
        assert_eq!(alpha_at(&img, 99, 99), 255);
        assert_eq!(alpha_at(&img, 0, 99), 255);
    }

    #[test]
    fn radius_scales_to_the_frame() {
        // 80px radii on a 100px frame shrink to 50 (CSS rule).
        let img = rounded_corners(&opaque(100, 100, [9, 9, 9]), [80, 80, 80, 80]);
        assert_eq!(alpha_at(&img, 0, 0), 0);
        // (0, 50) sits on the scaled arc itself: antialiased
        // to nearly full, not exactly full.
        assert!(alpha_at(&img, 0, 50) >= 250);
        assert_eq!(alpha_at(&img, 50, 50), 255);
    }

    #[test]
    fn mask_circle_and_ellipse_geometry() {
        let circle = mask_shape(&opaque(100, 60, [255, 0, 0]), crate::url::MaskShape::Circle);
        assert_eq!(alpha_at(&circle, 50, 30), 255);
        assert_eq!(alpha_at(&circle, 0, 0), 0);
        assert_eq!(alpha_at(&circle, 99, 30), 0);
        // The circle's diameter is the height, so it touches
        // the top and bottom edges at the horizontal center.
        assert!(alpha_at(&circle, 50, 59) >= 250);
        assert!(alpha_at(&circle, 50, 0) >= 250);
        let ellipse = mask_shape(
            &opaque(100, 60, [255, 0, 0]),
            crate::url::MaskShape::Ellipse,
        );
        assert_eq!(alpha_at(&ellipse, 50, 30), 255);
        assert!(alpha_at(&ellipse, 50, 0) >= 250);
        // The ellipse touches the left edge at the vertical
        // center: inside the feather there, fully outside
        // higher up.
        assert!(alpha_at(&ellipse, 0, 30) >= 150);
        assert_eq!(alpha_at(&ellipse, 0, 15), 0);
        assert_eq!(alpha_at(&ellipse, 0, 0), 0);
    }

    #[test]
    fn border_paints_inside_the_frame() {
        let img = draw_border(&opaque(10, 10, [255, 0, 0]), 2, [0, 255, 0]);
        assert_eq!((img.width(), img.height()), (10, 10));
        let rgba = img.into_rgba8();
        for (x, y) in [(0, 0), (1, 5), (9, 9), (5, 0), (8, 8)] {
            assert_eq!(
                rgba.get_pixel(x, y).0,
                [0, 255, 0, 255],
                "band pixel {x},{y}"
            );
        }
        for (x, y) in [(2, 2), (5, 5), (7, 7)] {
            assert_eq!(
                rgba.get_pixel(x, y).0,
                [255, 0, 0, 255],
                "interior pixel {x},{y}"
            );
        }
    }

    #[test]
    fn border_wider_than_half_paints_everything() {
        let img = draw_border(&opaque(10, 10, [255, 0, 0]), 5, [0, 0, 255]);
        assert_eq!(img.into_rgba8().get_pixel(5, 5).0, [0, 0, 255, 255]);
    }

    #[test]
    fn sepia_tones_with_the_standard_matrix() {
        let (img, _) = apply_filters(solid(128, 128, 128), &[Filter::Sepia]);
        assert_eq!(first_pixel(&img), [173, 154, 120]);
        let (img, _) = apply_filters(solid(0, 0, 0), &[Filter::Sepia]);
        assert_eq!(first_pixel(&img), [0, 0, 0]);
    }

    #[test]
    fn sepia_keeps_alpha() {
        let mut img = DynamicImage::new_rgba8(2, 2);
        for p in img.as_mut_rgba8().unwrap().pixels_mut() {
            *p = image::Rgba([100, 150, 200, 77]);
        }
        let out = sepia(&img);
        assert_eq!(out.to_rgba8().get_pixel(0, 0)[3], 77);
    }

    fn pixel_row(pixels: &[[u8; 3]]) -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(pixels.len() as u32, 1);
        for (x, _, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = Rgb(pixels[x as usize]);
        }
        img
    }

    fn row_rgb(img: &DynamicImage) -> Vec<[u8; 3]> {
        img.to_rgb8().pixels().map(|p| [p[0], p[1], p[2]]).collect()
    }

    #[test]
    fn curve_identity_is_a_no_op() {
        let img = striped();
        let before = img.to_rgb8().into_raw();
        for channel in [
            CurveChannel::Rgb,
            CurveChannel::R,
            CurveChannel::G,
            CurveChannel::B,
        ] {
            let (out, _) = apply_filters(
                img.clone(),
                &[Filter::Curve(channel, vec![(0, 0), (255, 255)])],
            );
            assert_eq!(out.to_rgb8().into_raw(), before, "channel {channel:?}");
        }
    }

    #[test]
    fn curve_two_point_lift_matches_hand_computed_values() {
        // curve(r,0,64,255,255): black lifts to 64, white
        // stays, and 128 lands at 64 + round(191 * 128 / 255)
        // = 64 + 96 = 160.
        let img = pixel_row(&[[0, 10, 20], [128, 10, 20], [255, 10, 20]]);
        let (out, _) = apply_filters(
            img,
            &[Filter::Curve(CurveChannel::R, vec![(0, 64), (255, 255)])],
        );
        assert_eq!(
            row_rgb(&out),
            vec![[64, 10, 20], [160, 10, 20], [255, 10, 20]]
        );
    }

    #[test]
    fn curve_interpolation_rounds_half_away_from_zero() {
        // Points (0,0) and (2,1): input 1 sits exactly on the
        // midpoint 0.5 and rounds away from zero, to 1. The
        // descending mirror (0,1),(2,0) sends 1 to 0 by the
        // same rule.
        let img = pixel_row(&[[0, 0, 0], [1, 0, 0], [2, 0, 0]]);
        let (out, _) = apply_filters(
            img.clone(),
            &[Filter::Curve(CurveChannel::R, vec![(0, 0), (2, 1)])],
        );
        assert_eq!(row_rgb(&out), vec![[0, 0, 0], [1, 0, 0], [1, 0, 0]]);
        let (out, _) = apply_filters(img, &[Filter::Curve(CurveChannel::R, vec![(0, 1), (2, 0)])]);
        assert_eq!(row_rgb(&out), vec![[1, 0, 0], [0, 0, 0], [0, 0, 0]]);
    }

    #[test]
    fn curve_channels_are_independent_and_alpha_survives() {
        let mut img = DynamicImage::new_rgba8(1, 1);
        img.as_mut_rgba8().unwrap().get_pixel_mut(0, 0).0 = [100, 150, 200, 77];
        let (out, _) = apply_filters(
            img,
            &[Filter::Curve(CurveChannel::R, vec![(0, 64), (255, 255)])],
        );
        // Red: 64 + round(191 * 100 / 255) = 64 + 75 = 139.
        // Green, blue, and alpha are not the curve's business.
        assert_eq!(out.to_rgba8().get_pixel(0, 0).0, [139, 150, 200, 77]);
    }

    #[test]
    fn curve_clamps_outside_the_point_range() {
        // curve(rgb,32,10,224,245): everything at or below 32
        // is 10, everything at or above 224 is 245, and the
        // middle interpolates (100 -> 93, and 128 is the
        // segment midpoint: 10 + round(235 * 96 / 192) = 128).
        let img = pixel_row(&[
            [0, 0, 0],
            [31, 31, 31],
            [32, 32, 32],
            [100, 100, 100],
            [128, 128, 128],
            [224, 224, 224],
            [255, 255, 255],
        ]);
        let (out, _) = apply_filters(
            img,
            &[Filter::Curve(CurveChannel::Rgb, vec![(32, 10), (224, 245)])],
        );
        let mapped: Vec<u8> = row_rgb(&out).iter().map(|p| p[0]).collect();
        assert_eq!(mapped, vec![10, 10, 10, 93, 128, 245, 245]);
    }

    #[test]
    fn curve_uses_the_segment_around_the_input() {
        // Three points: a steep rise to white at 128, then
        // flat. 64 is the first segment's exact midpoint
        // (255 * 64 / 128 = 127.5, rounding to 128); 200 sits
        // in the flat segment and stays 255.
        let img = pixel_row(&[[64, 64, 64], [128, 128, 128], [200, 200, 200]]);
        let (out, _) = apply_filters(
            img,
            &[Filter::Curve(
                CurveChannel::Rgb,
                vec![(0, 0), (128, 255), (255, 255)],
            )],
        );
        assert_eq!(
            row_rgb(&out),
            vec![[128, 128, 128], [255, 255, 255], [255, 255, 255]]
        );
    }

    #[test]
    fn curves_compose_in_chain_order() {
        // An rgb curve followed by an r curve: the second
        // sees the first's output (in the other order the red
        // would end at 128, not 255).
        let img = pixel_row(&[[255, 255, 255]]);
        let (out, _) = apply_filters(
            img,
            &[
                Filter::Curve(CurveChannel::Rgb, vec![(0, 0), (255, 128)]),
                Filter::Curve(CurveChannel::R, vec![(0, 0), (128, 255)]),
            ],
        );
        assert_eq!(row_rgb(&out), vec![[255, 128, 128]]);
    }

    fn gray_grid(rows: &[&[u8]]) -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(rows[0].len() as u32, rows.len() as u32);
        for (y, row) in rows.iter().enumerate() {
            for (x, v) in row.iter().enumerate() {
                img.as_mut_rgb8()
                    .unwrap()
                    .get_pixel_mut(x as u32, y as u32)
                    .0 = [*v, *v, *v];
            }
        }
        img
    }

    fn gray_values(img: &DynamicImage) -> Vec<Vec<u8>> {
        let rgb = img.to_rgb8();
        (0..rgb.height())
            .map(|y| (0..rgb.width()).map(|x| rgb.get_pixel(x, y)[0]).collect())
            .collect()
    }

    /// The gradient fixture several convolution tests share.
    fn gradient() -> DynamicImage {
        gray_grid(&[&[0, 10, 20], &[30, 40, 50], &[60, 70, 80]])
    }

    #[test]
    fn convolution_identity_is_a_no_op() {
        let img = striped();
        let before = img.to_rgb8().into_raw();
        let mut identity5 = vec![0.0; 25];
        identity5[12] = 1.0;
        for kernel in [
            (vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], false),
            (vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], true),
            (identity5, false),
        ] {
            let (out, _) = apply_filters(img.clone(), &[Filter::Convolution(kernel.0, kernel.1)]);
            assert_eq!(out.to_rgb8().into_raw(), before);
        }
    }

    #[test]
    fn convolution_sharpen_matches_hand_computed_pixels() {
        // The classic sharpen kernel on a lone bright pixel:
        // the center is 5 * 50 - (10 + 10 + 10 + 10) = 210,
        // and every other window's sum goes negative (the
        // bright pixel is at most one cross tap) and clamps
        // to 0.
        let img = gray_grid(&[&[0, 10, 0], &[10, 50, 10], &[0, 10, 0]]);
        let (out, _) = apply_filters(
            img,
            &[Filter::Convolution(
                vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
                false,
            )],
        );
        assert_eq!(
            gray_values(&out),
            vec![vec![0, 0, 0], vec![0, 210, 0], vec![0, 0, 0]]
        );
    }

    #[test]
    fn convolution_normalized_box_averages_the_window() {
        // An all-ones kernel with normalize on is the box
        // average over the edge-replicated window: the center
        // is 360 / 9 = 40, the corner counts its replicated
        // taps (120 / 9, rounding to 13), and so on.
        let (out, _) = apply_filters(gradient(), &[Filter::Convolution(vec![1.0; 9], true)]);
        assert_eq!(
            gray_values(&out),
            vec![vec![13, 20, 27], vec![33, 40, 47], vec![53, 60, 67]]
        );
        // Rounding is to nearest: a lone 5 in a 3x3 of zeros
        // averages to 5 / 9 = 0.55..., which rounds up to 1
        // in every window (each contains the 5 exactly once).
        let lone = gray_grid(&[&[0, 0, 0], &[0, 5, 0], &[0, 0, 0]]);
        let (out, _) = apply_filters(lone, &[Filter::Convolution(vec![1.0; 9], true)]);
        assert_eq!(
            gray_values(&out),
            vec![vec![1, 1, 1], vec![1, 1, 1], vec![1, 1, 1]]
        );
    }

    #[test]
    fn convolution_unnormalized_box_sums_and_clamps() {
        // The same kernel without normalize sums the window:
        // the top row stays under the ceiling, everything
        // from the middle row down clamps at 255.
        let (out, _) = apply_filters(gradient(), &[Filter::Convolution(vec![1.0; 9], false)]);
        assert_eq!(
            gray_values(&out),
            vec![
                vec![120, 180, 240],
                vec![255, 255, 255],
                vec![255, 255, 255]
            ]
        );
    }

    #[test]
    fn convolution_zero_sum_kernel_ignores_normalize() {
        // An edge kernel sums to zero, so normalize has no
        // scale to apply: both spellings must produce the
        // kernel exactly as written (center: the cross taps
        // sum to 10 + 30 + 50 + 70 = 160, minus 4 * 40, is 0).
        let kernel = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
        let (plain, _) = apply_filters(gradient(), &[Filter::Convolution(kernel.clone(), false)]);
        let (normalized, _) = apply_filters(gradient(), &[Filter::Convolution(kernel, true)]);
        assert_eq!(plain.to_rgb8().into_raw(), normalized.to_rgb8().into_raw());
        assert_eq!(gray_values(&normalized)[1][1], 0);
    }

    #[test]
    fn convolution_five_by_five_applies_its_taps() {
        // A single tap in the kernel's top-left corner: every
        // output reads the source two pixels up and left,
        // clamped to the frame (the edge replication rule).
        let img = gray_grid(&[
            &[0, 1, 2, 3, 4],
            &[5, 6, 7, 8, 9],
            &[10, 11, 12, 13, 14],
            &[15, 16, 17, 18, 19],
            &[20, 21, 22, 23, 24],
        ]);
        let mut kernel = vec![0.0; 25];
        kernel[0] = 1.0;
        let (out, _) = apply_filters(img, &[Filter::Convolution(kernel, false)]);
        assert_eq!(
            gray_values(&out),
            vec![
                vec![0, 0, 0, 1, 2],
                vec![0, 0, 0, 1, 2],
                vec![0, 0, 0, 1, 2],
                vec![5, 5, 5, 6, 7],
                vec![10, 10, 10, 11, 12],
            ]
        );
    }

    #[test]
    fn convolution_edges_replicate_at_the_corners() {
        // A kernel with only its right tap set: each output
        // is the pixel to its right, and the right column
        // reads itself (the edge, replicated).
        let img = gray_grid(&[&[10, 20], &[30, 40]]);
        let (out, _) = apply_filters(
            img,
            &[Filter::Convolution(
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                false,
            )],
        );
        assert_eq!(gray_values(&out), vec![vec![20, 20], vec![40, 40]]);
    }

    #[test]
    fn convolution_decimal_kernel_matches_hand_computed_values() {
        // The 3x3 gaussian (weights sum to 1) on the linear
        // gradient: the center is the corners at 1/16
        // (160 / 16 = 10), the edges at 1/8 (160 / 8 = 20),
        // and the center at 1/4 (40 / 4 = 10), totaling 40.
        let gaussian = vec![
            0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625,
        ];
        let (out, _) = apply_filters(gradient(), &[Filter::Convolution(gaussian.clone(), false)]);
        assert_eq!(gray_values(&out)[1][1], 40);
        // On a flat field the weights summing to 1 leave the
        // value alone, exactly (every weight is dyadic).
        let flat = gray_grid(&[&[80, 80, 80], &[80, 80, 80], &[80, 80, 80]]);
        let (out, _) = apply_filters(flat, &[Filter::Convolution(gaussian, false)]);
        assert_eq!(
            gray_values(&out),
            vec![vec![80, 80, 80], vec![80, 80, 80], vec![80, 80, 80]]
        );
    }

    #[test]
    fn convolution_never_touches_alpha() {
        let mut img = DynamicImage::new_rgba8(2, 2);
        let px = [
            [10, 20, 30, 77],
            [40, 50, 60, 77],
            [70, 80, 90, 77],
            [100, 110, 120, 77],
        ];
        for (i, p) in img.as_mut_rgba8().unwrap().pixels_mut().enumerate() {
            *p = image::Rgba(px[i]);
        }
        let (out, _) = apply_filters(img, &[Filter::Convolution(vec![1.0; 9], true)]);
        let rgba = out.to_rgba8();
        // The box average at (0, 0) weights the corner pixel
        // four times, its edge neighbors twice, and the far
        // corner once: (360, 450, 540) / 9 = (40, 50, 60).
        assert_eq!(rgba.get_pixel(0, 0).0, [40, 50, 60, 77]);
        assert!(rgba.pixels().all(|p| p[3] == 77));
    }

    #[test]
    fn noise_zero_is_a_byte_identical_no_op() {
        // On an RGBA image (variant and alpha included) and
        // an RGB one: amount 0 returns the image untouched.
        let mut img = DynamicImage::new_rgba8(3, 2);
        for (i, p) in img.as_mut_rgba8().unwrap().pixels_mut().enumerate() {
            *p = image::Rgba([10 + 30 * i as u8, 20 + 20 * i as u8, 30 + 10 * i as u8, 77]);
        }
        let before = img.to_rgba8().into_raw();
        let (out, _) = apply_filters(img, &[Filter::Noise(0)]);
        assert!(matches!(out, DynamicImage::ImageRgba8(_)));
        assert_eq!(out.to_rgba8().into_raw(), before);
        let img = striped();
        let before = img.to_rgb8().into_raw();
        let (out, _) = apply_filters(img, &[Filter::Noise(0)]);
        assert_eq!(out.to_rgb8().into_raw(), before);
    }

    #[test]
    fn noise_matches_the_documented_generator() {
        // The generator, pinned end to end: the seed is the
        // FNV-1a hash of the RGBA buffer folded with the
        // amount, the stream is SplitMix64, one draw per
        // pixel in row-major order, and the delta is the
        // draw modulo 2A+1 minus A, with A the rounded
        // amplitude (255 at amount 100, 102 at 40). For
        // this row the amount-100 draws give +220, +131,
        // and -238 (the last pixel clamps at 0), and the
        // amount-40 draws give -61, +20, and +22.
        let img = pixel_row(&[[10, 20, 30], [100, 110, 120], [200, 210, 220]]);
        let (out, _) = apply_filters(img.clone(), &[Filter::Noise(100)]);
        assert_eq!(
            row_rgb(&out),
            vec![[230, 240, 250], [231, 241, 251], [0, 0, 0]]
        );
        let (out, _) = apply_filters(img, &[Filter::Noise(40)]);
        assert_eq!(
            row_rgb(&out),
            vec![[0, 0, 0], [120, 130, 140], [222, 232, 242]]
        );
    }

    #[test]
    fn noise_is_deterministic_and_depends_on_the_amount() {
        // The same input and amount render the same bytes
        // every run; a different amount is a different
        // field (and a different image).
        let img = striped();
        let (a, _) = apply_filters(img.clone(), &[Filter::Noise(60)]);
        let (b, _) = apply_filters(img.clone(), &[Filter::Noise(60)]);
        assert_eq!(a.to_rgb8().into_raw(), b.to_rgb8().into_raw());
        let (c, _) = apply_filters(img, &[Filter::Noise(61)]);
        assert_ne!(a.to_rgb8().into_raw(), c.to_rgb8().into_raw());
    }

    #[test]
    fn noise_full_strength_moves_a_flat_field_both_ways() {
        // On a flat mid-gray field at 100, some pixels land
        // above 128 and some below: the field swings both
        // ways and stays inside the byte range by clamping.
        let (out, _) = apply_filters(opaque(64, 64, [128, 128, 128]), &[Filter::Noise(100)]);
        let raw = out.to_rgb8().into_raw();
        assert!(raw.iter().any(|&v| v > 128));
        assert!(raw.iter().any(|&v| v < 128));
        // No channel of any pixel may differ from another:
        // even on the way through the clamp the value is
        // one per pixel, shared by all three channels.
        for px in out.to_rgb8().pixels() {
            assert_eq!(px[0], px[1]);
            assert_eq!(px[1], px[2]);
        }
    }

    #[test]
    fn noise_is_monochrome_and_preserves_alpha() {
        // Mid-range pixels and a modest amount (amplitude
        // 51 at 20), so nothing clamps and the per-pixel
        // delta must be identical on all three channels.
        let mut img = DynamicImage::new_rgba8(4, 1);
        let src = [
            [110, 120, 130, 77],
            [125, 135, 145, 77],
            [100, 140, 120, 77],
            [130, 115, 105, 77],
        ];
        for (i, p) in img.as_mut_rgba8().unwrap().pixels_mut().enumerate() {
            *p = image::Rgba(src[i]);
        }
        let (out, _) = apply_filters(img, &[Filter::Noise(20)]);
        let rgba = out.to_rgba8();
        for (x, s) in src.iter().enumerate() {
            let o = rgba.get_pixel(x as u32, 0).0;
            let delta = i32::from(o[0]) - i32::from(s[0]);
            assert_eq!(i32::from(o[1]) - i32::from(s[1]), delta, "pixel {x}");
            assert_eq!(i32::from(o[2]) - i32::from(s[2]), delta, "pixel {x}");
            assert_eq!(o[3], 77, "alpha survives at pixel {x}");
        }
    }

    #[test]
    fn duotone_maps_shadows_and_highlights() {
        let bw = |px: [u8; 3]| {
            let mut img = DynamicImage::new_rgb8(1, 1);
            img.as_mut_rgb8().unwrap().get_pixel_mut(0, 0).0 = px;
            duotone(&img, [255, 0, 0], [0, 0, 255])
                .to_rgb8()
                .get_pixel(0, 0)
                .0
        };
        assert_eq!(bw([0, 0, 0]), [255, 0, 0]);
        assert_eq!(bw([255, 255, 255]), [0, 0, 255]);
        // Pure red's luminance is 0.299 of the way from red to blue.
        assert_eq!(bw([255, 0, 0]), [179, 0, 76]);
    }

    #[test]
    fn monochrome_scales_luminance_by_the_color() {
        let one = |px: [u8; 3]| {
            let mut img = DynamicImage::new_rgb8(1, 1);
            img.as_mut_rgb8().unwrap().get_pixel_mut(0, 0).0 = px;
            monochrome(&img, [255, 128, 0]).to_rgb8().get_pixel(0, 0).0
        };
        assert_eq!(one([255, 255, 255]), [255, 128, 0]);
        assert_eq!(one([0, 0, 0]), [0, 0, 0]);
        assert_eq!(one([128, 128, 128]), [128, 64, 0]);
    }

    #[test]
    fn pixelate_averages_blocks() {
        // 4x4, four solid quadrants: 2px blocks stay pure.
        let mut img = DynamicImage::new_rgb8(4, 4);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = match (x < 2, y < 2) {
                (true, true) => Rgb([255, 0, 0]),
                (false, true) => Rgb([0, 0, 255]),
                (true, false) => Rgb([0, 255, 0]),
                (false, false) => Rgb([255, 255, 255]),
            };
        }
        let out = pixelate(&img, 2);
        let rgb = out.to_rgb8();
        assert_eq!(rgb.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(rgb.get_pixel(3, 0).0, [0, 0, 255]);
        assert_eq!(rgb.get_pixel(0, 3).0, [0, 255, 0]);
        assert_eq!(rgb.get_pixel(3, 3).0, [255, 255, 255]);
        // One 4x4 block over a half-red half-blue image averages.
        let mut img = DynamicImage::new_rgb8(4, 2);
        for (x, _y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if x < 2 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            };
        }
        let out = pixelate(&img, 4);
        assert_eq!(out.to_rgb8().get_pixel(0, 0).0, [128, 0, 128]);
        assert_eq!(out.to_rgb8().get_pixel(3, 1).0, [128, 0, 128]);
        // Block 1 is the identity.
        assert_eq!(pixelate(&img, 1).to_rgb8(), img.into_rgb8());
    }

    #[test]
    fn flip_mirrors_both_ways() {
        let mut img = DynamicImage::new_rgb8(2, 1);
        img.as_mut_rgb8().unwrap().get_pixel_mut(0, 0).0 = [255, 0, 0];
        img.as_mut_rgb8().unwrap().get_pixel_mut(1, 0).0 = [0, 0, 255];
        let (out, _) = apply_filters(
            img.clone(),
            &[Filter::Flip(crate::url::FlipDirection::Horizontal)],
        );
        assert_eq!(out.to_rgb8().get_pixel(0, 0).0, [0, 0, 255]);
        assert_eq!(out.to_rgb8().get_pixel(1, 0).0, [255, 0, 0]);
        let mut img = DynamicImage::new_rgb8(1, 2);
        img.as_mut_rgb8().unwrap().get_pixel_mut(0, 0).0 = [255, 0, 0];
        img.as_mut_rgb8().unwrap().get_pixel_mut(0, 1).0 = [0, 0, 255];
        let (out, _) = apply_filters(img, &[Filter::Flip(crate::url::FlipDirection::Vertical)]);
        assert_eq!(out.to_rgb8().get_pixel(0, 0).0, [0, 0, 255]);
        assert_eq!(out.to_rgb8().get_pixel(0, 1).0, [255, 0, 0]);
    }

    #[test]
    fn blur_region_smooths_only_the_region() {
        // 1px checkerboard: maximal local contrast.
        let mut img = DynamicImage::new_rgb8(40, 40);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let v = if (x + y) % 2 == 0 { 0 } else { 255 };
            *p = Rgb([v, v, v]);
        }
        let out = blur_region(&img, 10, 10, 20, 20);
        let rgb = out.to_rgb8();
        // Outside the region: pixel-exact.
        assert_eq!(rgb.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(rgb.get_pixel(39, 39).0, [0, 0, 0]);
        assert_eq!(rgb.get_pixel(8, 15).0, [255, 255, 255]);
        // Inside: the checkerboard is smoothed toward mid gray,
        // so neighbors differ by far less than the original 255.
        let a = rgb.get_pixel(20, 20)[0] as i32;
        let b = rgb.get_pixel(21, 20)[0] as i32;
        assert!((a - b).abs() < 60, "neighbors {a} vs {b}");
        assert!(a > 60 && a < 195, "smoothed toward gray, got {a}");
    }

    #[test]
    fn pixelate_region_averages_inside_only() {
        // Left half red, right half blue, edge at x = 20.
        let mut img = DynamicImage::new_rgb8(40, 40);
        for (x, _y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if x < 20 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            };
        }
        // Region starts at x = 11, so its 2px blocks straddle
        // the color edge at block [19, 20].
        let out = pixelate_region(&img, 11, 10, 20, 20);
        let rgb = out.to_rgb8();
        assert_eq!(rgb.get_pixel(19, 15).0, [128, 0, 128]);
        assert_eq!(rgb.get_pixel(20, 15).0, [128, 0, 128]);
        // Pure blocks inside stay pure.
        assert_eq!(rgb.get_pixel(12, 15).0, [255, 0, 0]);
        assert_eq!(rgb.get_pixel(29, 15).0, [0, 0, 255]);
        // Outside the region nothing changes.
        assert_eq!(rgb.get_pixel(5, 15).0, [255, 0, 0]);
        assert_eq!(rgb.get_pixel(35, 15).0, [0, 0, 255]);
        assert_eq!(rgb.get_pixel(19, 5).0, [255, 0, 0]);
    }

    fn striped() -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(64, 64);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let v = ((x + y) % 256) as u8;
            *p = Rgb([v, 255 - v, (x * y % 256) as u8]);
        }
        img
    }

    fn solid(r: u8, g: u8, b: u8) -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(4, 4);
        for p in img.as_mut_rgb8().unwrap().pixels_mut() {
            *p = Rgb([r, g, b]);
        }
        img
    }

    fn first_pixel(img: &DynamicImage) -> [u8; 3] {
        let rgb = img.to_rgb8();
        let p = rgb.pixels().next().unwrap();
        [p[0], p[1], p[2]]
    }

    #[test]
    fn later_directives_win() {
        let (img, opts) = apply_filters(
            striped(),
            &[
                Filter::Quality(Quality::Value(10)),
                Filter::Quality(Quality::Value(90)),
                Filter::Format(ImageFormat::Png),
                Filter::Format(ImageFormat::Jpeg),
            ],
        );
        assert_eq!((img.width(), img.height()), (64, 64));
        assert_eq!(
            opts,
            EncodeOptions {
                format: Some(ImageFormat::Jpeg),
                quality: Some(Quality::Value(90)),
                no_upscale: false,
                fill: None,
                max_bytes: None,
                keep_profile: false,
            }
        );
    }

    #[test]
    fn no_upscale_is_collected_as_a_directive() {
        let img = DynamicImage::new_rgb8(8, 8);
        let (_, opts) = apply_filters(img, &[Filter::NoUpscale]);
        assert!(opts.no_upscale);
    }

    #[test]
    fn sharpen_increases_edge_contrast() {
        // Vertical step: left half 100, right half 200.
        let mut img = DynamicImage::new_rgb8(16, 16);
        for (x, _, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let v = if x < 8 { 100 } else { 200 };
            *p = Rgb([v, v, v]);
        }
        let (img, _) = apply_filters(img, &[Filter::Sharpen(1.0)]);
        let rgb = img.into_rgb8();
        let left = rgb.get_pixel(7, 8)[0] as i32;
        let right = rgb.get_pixel(8, 8)[0] as i32;
        assert!(right - left > 100, "left {left} right {right}");
    }

    #[test]
    fn saturation_scales_color() {
        let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::Saturation(0.0)]);
        let p = first_pixel(&img);
        assert_eq!(p[0], p[1]);
        assert_eq!(p[1], p[2]);
        let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::Saturation(1.0)]);
        assert_eq!(first_pixel(&img), [200, 100, 50]);
        let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::Saturation(2.0)]);
        let p = first_pixel(&img);
        assert!(p[0] as i32 - p[2] as i32 > 150, "pixel {p:?}");
    }

    #[test]
    fn rgb_offsets_channels() {
        let (img, _) = apply_filters(solid(100, 100, 100), &[Filter::Rgb(50, -30, 0)]);
        assert_eq!(first_pixel(&img), [150, 70, 100]);
    }

    #[test]
    fn equalize_spreads_the_range() {
        // All pixels in a narrow band, 100..=107.
        let mut img = DynamicImage::new_rgb8(16, 16);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let v = 100 + ((x + y) % 8) as u8;
            *p = Rgb([v, v, v]);
        }
        let (img, _) = apply_filters(img, &[Filter::Equalize]);
        let rgb = img.into_rgb8();
        let min = rgb.pixels().map(|p| p[0]).min().unwrap();
        let max = rgb.pixels().map(|p| p[0]).max().unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, 255);
    }

    #[test]
    fn rotate_45_expands_the_canvas_and_fills_corners() {
        let img = DynamicImage::ImageRgb8(solid(255, 0, 0).to_rgb8());
        let img = img.resize_exact(4, 2, image::imageops::FilterType::Nearest);
        let (out, _) = apply_filters(img, &[Filter::Rotate(45.0)]);
        // ceil(4*cos45 + 2*sin45) = ceil(4.243) = 5 on both axes.
        assert_eq!((out.width(), out.height()), (5, 5));
        let rgb = out.to_rgb8();
        // The uncovered corner takes the default background.
        assert_eq!(rgb.get_pixel(0, 0).0, [255, 255, 255]);
        // The middle is still the source color.
        assert_eq!(rgb.get_pixel(2, 2).0, [255, 0, 0]);
    }

    #[test]
    fn rotate_45_uses_the_request_fill_for_corners() {
        let img = DynamicImage::ImageRgb8(solid(255, 0, 0).to_rgb8()).resize_exact(
            4,
            2,
            image::imageops::FilterType::Nearest,
        );
        let (out, _) = apply_filters_filled(img, &[Filter::Rotate(45.0)], Some([10, 20, 30]));
        assert_eq!(out.to_rgb8().get_pixel(0, 0).0, [10, 20, 30]);
    }

    #[test]
    fn rotate_45_keeps_alpha_corners_transparent() {
        let mut rgba = image::RgbaImage::new(4, 2);
        for p in rgba.pixels_mut() {
            *p = image::Rgba([255, 0, 0, 255]);
        }
        let (out, _) = apply_filters(DynamicImage::ImageRgba8(rgba), &[Filter::Rotate(45.0)]);
        let rgba = out.to_rgba8();
        assert_eq!(rgba.get_pixel(0, 0).0[3], 0);
        assert_eq!(rgba.get_pixel(2, 2).0, [255, 0, 0, 255]);
    }

    #[test]
    fn rotate_positive_angles_turn_clockwise() {
        // A marker above center moves to the upper right under
        // a positive (clockwise) rotation, matching rotate90.
        let mut rgb = image::RgbImage::new(20, 20);
        rgb.put_pixel(10, 2, image::Rgb([255, 0, 0]));
        let (out, _) = apply_filters(DynamicImage::ImageRgb8(rgb), &[Filter::Rotate(45.0)]);
        let rgb = out.to_rgb8();
        // Track the reddest pixel (the white corners have no
        // red-over-green excess).
        let (mut bx, mut by, mut best) = (0u32, 0u32, 0i16);
        for (x, y, p) in rgb.enumerate_pixels() {
            let excess = p.0[0] as i16 - p.0[1] as i16;
            if excess > best {
                best = excess;
                bx = x;
                by = y;
            }
        }
        assert!(bx > out.width() / 2, "marker x {bx} of {}", out.width());
        assert!(by < out.height() / 2, "marker y {by} of {}", out.height());
    }

    #[test]
    fn rotate_full_turns_are_identity() {
        let (out, _) = apply_filters(DynamicImage::new_rgb8(4, 2), &[Filter::Rotate(360.0)]);
        assert_eq!((out.width(), out.height()), (4, 2));
    }

    #[test]
    fn rotate_swaps_dimensions() {
        let (img, _) = apply_filters(DynamicImage::new_rgb8(4, 2), &[Filter::Rotate(90.0)]);
        assert_eq!((img.width(), img.height()), (2, 4));
        let (img, _) = apply_filters(DynamicImage::new_rgb8(4, 2), &[Filter::Rotate(-90.0)]);
        assert_eq!((img.width(), img.height()), (2, 4));
        let (img, _) = apply_filters(DynamicImage::new_rgb8(4, 2), &[Filter::Rotate(180.0)]);
        assert_eq!((img.width(), img.height()), (4, 2));
    }

    #[test]
    fn fill_and_max_bytes_are_collected() {
        let (_, opts) = apply_filters(
            solid(1, 2, 3),
            &[Filter::Fill([255, 136, 0]), Filter::MaxBytes(5000)],
        );
        assert_eq!(opts.fill, Some([255, 136, 0]));
        assert_eq!(opts.max_bytes, Some(5000));
    }

    #[test]
    fn absorb_merges_directives() {
        let mut base = EncodeOptions::default();
        base.absorb(EncodeOptions {
            format: Some(ImageFormat::Png),
            quality: None,
            no_upscale: true,
            fill: None,
            max_bytes: Some(100),
            keep_profile: true,
        });
        base.absorb(EncodeOptions {
            format: None,
            quality: Some(Quality::Value(50)),
            no_upscale: false,
            fill: None,
            max_bytes: None,
            keep_profile: false,
        });
        assert_eq!(
            base,
            EncodeOptions {
                keep_profile: true,
                format: Some(ImageFormat::Png),
                quality: Some(Quality::Value(50)),
                no_upscale: true,
                fill: None,
                max_bytes: Some(100),
            }
        );
    }

    #[test]
    fn empty_filters_are_noop() {
        let (img, opts) = apply_filters(striped(), &[]);
        assert_eq!((img.width(), img.height()), (64, 64));
        assert_eq!(opts, EncodeOptions::default());
    }

    #[test]
    fn grayscale_desaturates() {
        let (img, _) = apply_filters(striped(), &[Filter::Grayscale]);
        for p in img.into_rgb8().pixels() {
            assert_eq!(p[0], p[1]);
            assert_eq!(p[1], p[2]);
        }
    }

    #[test]
    fn invert_flips_channels() {
        let (img, _) = apply_filters(solid(10, 128, 250), &[Filter::Invert]);
        assert_eq!(first_pixel(&img), [245, 127, 5]);
    }

    #[test]
    fn brighten_shifts_and_clamps() {
        let (img, _) = apply_filters(solid(100, 150, 200), &[Filter::Brighten(10)]);
        assert_eq!(first_pixel(&img), [110, 160, 210]);

        let (img, _) = apply_filters(solid(200, 200, 200), &[Filter::Brighten(100)]);
        assert_eq!(first_pixel(&img), [255, 255, 255]);

        let (img, _) = apply_filters(solid(100, 150, 200), &[Filter::Brighten(-300)]);
        assert_eq!(first_pixel(&img), [0, 0, 0]);
    }

    #[test]
    fn blur_smooths_a_hard_edge() {
        let mut img = DynamicImage::new_rgb8(32, 32);
        for (x, _y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            let v = if x < 16 { 0 } else { 255 };
            *p = Rgb([v, v, v]);
        }
        let (img, _) = apply_filters(img, &[Filter::Blur(2.0)]);
        let rgb = img.into_rgb8();
        // Pixels hugging the edge are now a blend of black and white.
        for (x, y, p) in rgb.enumerate_pixels() {
            if x == 15 || x == 16 {
                assert!(p[0] > 0 && p[0] < 255, "pixel ({x}, {y}) = {p:?}");
            }
        }
    }

    #[test]
    fn huerotate_rotates_and_comes_back() {
        let (img, _) = apply_filters(solid(255, 0, 0), &[Filter::HueRotate(90)]);
        let px = first_pixel(&img);
        // The hue moved: pure red is gone.
        assert_ne!(px, [255, 0, 0]);

        // A full turn is (approximately) the identity.
        let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::HueRotate(360)]);
        let px = first_pixel(&img);
        for (got, want) in px.iter().zip([200, 100, 50]) {
            assert!((*got as i16 - want as i16).abs() <= 1, "pixel = {px:?}");
        }
    }

    #[test]
    fn contrast_pushes_values_from_mid_gray() {
        let (img, _) = apply_filters(solid(100, 100, 100), &[Filter::Contrast(2.0)]);
        let px = first_pixel(&img);
        assert!(px[0] < 100, "pixel = {px:?}");

        let (img, _) = apply_filters(solid(200, 200, 200), &[Filter::Contrast(2.0)]);
        let px = first_pixel(&img);
        assert!(px[0] > 200, "pixel = {px:?}");
    }

    #[test]
    fn new_filters_compose_in_order() {
        // invert then brighten(10): (10 -> 245 -> 255).
        let (img, _) = apply_filters(solid(10, 10, 10), &[Filter::Invert, Filter::Brighten(10)]);
        assert_eq!(first_pixel(&img), [255, 255, 255]);
    }

    /// 40x30 `bg` with an `fg` rectangle at (10..30, 8..22).
    fn bordered(bg: [u8; 3], fg: [u8; 3]) -> DynamicImage {
        let mut img = DynamicImage::new_rgb8(40, 30);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if (10..30).contains(&x) && (8..22).contains(&y) {
                Rgb(fg)
            } else {
                Rgb(bg)
            };
        }
        img
    }

    #[test]
    fn trim_crops_to_the_inner_box() {
        let (img, _) = apply_filters(bordered([255, 255, 255], [255, 0, 0]), &[Filter::Trim(0)]);
        assert_eq!((img.width(), img.height()), (20, 14));
        assert_eq!(first_pixel(&img), [255, 0, 0]);
    }

    #[test]
    fn trim_tolerance_decides_what_counts_as_border() {
        // Corner 255, an inner ring of 245, black center 10..20.
        let build = || {
            let mut img = DynamicImage::new_rgb8(30, 30);
            for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
                *p = if (10..20).contains(&x) && (10..20).contains(&y) {
                    Rgb([0, 0, 0])
                } else if (5..25).contains(&x) && (5..25).contains(&y) {
                    Rgb([245, 245, 245])
                } else {
                    Rgb([255, 255, 255])
                };
            }
            img
        };
        // Tolerance 10: the ring (10 away) is still border.
        let (img, _) = apply_filters(build(), &[Filter::Trim(10)]);
        assert_eq!((img.width(), img.height()), (10, 10));
        // Tolerance 5: the ring is content.
        let (img, _) = apply_filters(build(), &[Filter::Trim(5)]);
        assert_eq!((img.width(), img.height()), (20, 20));
    }

    #[test]
    fn trim_uniform_image_is_a_noop() {
        let (img, _) = apply_filters(solid(77, 77, 77), &[Filter::Trim(10)]);
        assert_eq!((img.width(), img.height()), (4, 4));
        assert_eq!(first_pixel(&img), [77, 77, 77]);
    }
}
