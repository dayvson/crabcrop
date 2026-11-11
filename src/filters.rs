//! Applying filter operations to decoded images.
//!
//! [`apply_filters_filled`] runs pixel filters in URL order and collects
//! encoding directives. `Format`, `Quality`, `Fill`, and
//! `MaxBytes` do not touch pixels; they are recorded for the
//! encoder.

use image::DynamicImage;

use crate::url::{
    CurveChannel, Filter, FormatOptions, ImageFormat, MAX_BLUR_SIGMA, MAX_EXPOSURE, MAX_GAMMA,
    Quality,
};

/// Encoding directives collected from filters.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EncodeOptions {
    /// Output container override. `None` keeps the detected source format.
    pub format: Option<ImageFormat>,
    /// Encoder options from the format filter (issue #319).
    /// Meaningful only when `format` is set by the same
    /// filter; absorb keeps the two together.
    pub format_options: FormatOptions,
    /// Encoding quality directive. `None` means the encoder default.
    pub quality: Option<Quality>,
    /// Color budget for indexed PNG output (issue #313).
    /// `None` writes the usual full-color PNG.
    pub palette: Option<u16>,
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
            self.format_options = other.format_options;
        }
        if other.quality.is_some() {
            self.quality = other.quality;
        }
        if other.palette.is_some() {
            self.palette = other.palette;
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
pub type WatermarkImages = std::collections::HashMap<String, WatermarkImage>;

/// One decoded watermark or overlay image (issue #604). The
/// composite reads RGBA pixels, and only those of the part
/// that lands on the image below. An image that decoded as
/// RGBA is read in place. Any other has that part copied to
/// RGBA the first time a composite draws it, and the copy is
/// kept for the rest of the request, so an animated source,
/// whose frames all place it the same way, copies it once. A
/// later use that needs a part the kept copy does not cover
/// copies its own part. The decoded image stays as it is,
/// because `overlay(...,width)` resizes that one.
#[derive(Debug)]
pub struct WatermarkImage {
    image: DynamicImage,
    /// The kept RGBA copy, and the part of the image it covers.
    rgba: std::sync::OnceLock<(Rect, image::RgbaImage)>,
    /// How many RGBA copies were made, kept or not.
    #[cfg(test)]
    conversions: std::sync::atomic::AtomicUsize,
}

/// A part of an image: x, y, width, height.
pub type Rect = (u32, u32, u32, u32);

impl WatermarkImage {
    pub fn new(image: DynamicImage) -> Self {
        WatermarkImage {
            image,
            rgba: std::sync::OnceLock::new(),
            #[cfg(test)]
            conversions: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// The image as it decoded.
    pub fn image(&self) -> &DynamicImage {
        &self.image
    }

    /// The RGBA pixels of the part `rect` of the image: read
    /// in place when the image decoded as RGBA, from the kept
    /// copy when it covers `rect`, and otherwise from a copy
    /// of `rect` alone, which is kept when it is the first.
    pub fn rgba_part(&self, rect: Rect) -> RgbaPart<'_> {
        if self.image.as_rgba8().is_some() {
            return RgbaPart::of(&self.image, rect);
        }
        let (kept, pixels) = self.rgba.get_or_init(|| (rect, self.copy_part(rect)));
        let (kx, ky, kw, kh) = *kept;
        let (x, y, w, h) = rect;
        if x >= kx && y >= ky && x + w <= kx + kw && y + h <= ky + kh {
            return RgbaPart {
                pixels: std::borrow::Cow::Borrowed(pixels),
                origin: (kx, ky),
                size: (self.image.width(), self.image.height()),
            };
        }
        RgbaPart {
            pixels: std::borrow::Cow::Owned(self.copy_part(rect)),
            origin: (x, y),
            size: (self.image.width(), self.image.height()),
        }
    }

    fn copy_part(&self, rect: Rect) -> image::RgbaImage {
        #[cfg(test)]
        self.conversions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        copy_part(&self.image, rect)
    }

    /// The most memory this image holds (issue #613): its
    /// decoded pixels, plus an RGBA copy of all of it when it
    /// did not decode as RGBA. The kept copy covers one part
    /// at most, and is counted at the whole image's size
    /// whether or not it has been made.
    pub fn held_bytes(&self) -> u64 {
        let decoded = self.image.as_bytes().len() as u64;
        if self.image.as_rgba8().is_some() {
            decoded
        } else {
            decoded + self.image.width() as u64 * self.image.height() as u64 * 4
        }
    }

    /// How many RGBA copies were made, kept or not.
    #[cfg(test)]
    pub(crate) fn conversions(&self) -> usize {
        self.conversions.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The part of the image the kept RGBA copy covers.
    #[cfg(test)]
    pub(crate) fn kept_part(&self) -> Option<Rect> {
        self.rgba.get().map(|(rect, _)| *rect)
    }
}

/// The part `rect` of `image` in RGBA. Each pixel converts on
/// its own, so the part converts to the same pixels as the
/// same part of the whole image does.
fn copy_part(image: &DynamicImage, (x, y, w, h): Rect) -> image::RgbaImage {
    image.crop_imm(x, y, w, h).into_rgba8()
}

/// The RGBA pixels of a part of an image (issue #604), read
/// in the whole image's coordinates. A composite reads only
/// the part it asked for; reading a pixel outside it panics.
pub struct RgbaPart<'a> {
    pixels: std::borrow::Cow<'a, image::RgbaImage>,
    /// Where `pixels` starts in the whole image.
    origin: (u32, u32),
    /// The whole image's size.
    size: (u32, u32),
}

impl<'a> RgbaPart<'a> {
    /// The part `rect` of `image`: all of it read in place
    /// when it is RGBA, otherwise a copy of `rect` alone.
    pub fn of(image: &'a DynamicImage, rect: Rect) -> Self {
        let size = (image.width(), image.height());
        match image.as_rgba8() {
            Some(pixels) => RgbaPart {
                pixels: std::borrow::Cow::Borrowed(pixels),
                origin: (0, 0),
                size,
            },
            None => RgbaPart {
                pixels: std::borrow::Cow::Owned(copy_part(image, rect)),
                origin: (rect.0, rect.1),
                size,
            },
        }
    }
}

impl image::GenericImageView for RgbaPart<'_> {
    type Pixel = image::Rgba<u8>;

    fn dimensions(&self) -> (u32, u32) {
        self.size
    }

    fn get_pixel(&self, x: u32, y: u32) -> image::Rgba<u8> {
        *self.pixels.get_pixel(x - self.origin.0, y - self.origin.1)
    }
}

impl From<DynamicImage> for WatermarkImage {
    fn from(image: DynamicImage) -> Self {
        WatermarkImage::new(image)
    }
}

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

/// `apply_filters` (test builds only) with the request's resolved fill color,
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
            Filter::Grayscale => {
                img = map_rgb(img, |r, g, b| {
                    let y = luma(r, g, b).round().clamp(0.0, 255.0) as u8;
                    [y, y, y]
                });
            }
            Filter::Invert => img.invert(),
            Filter::Blur(sigma) => img = blur_fast(&img, blur_sigma(*sigma)),
            Filter::Brighten(amount) => img = brighten_fast(img, *amount),
            Filter::Contrast(amount) => {
                img = map_rgb_image(img, |rgb| contrast_in_place(rgb, *amount));
            }
            Filter::Gamma(gamma) => {
                img = gamma_correct(img, *gamma);
            }
            Filter::Exposure(stops) => {
                img = exposure_correct(img, *stops);
            }
            Filter::HighlightsShadows(highlights, shadows) => {
                img = highlights_shadows(&img, *highlights, *shadows);
            }
            Filter::HueRotate(degrees) => img = huerotate_fast(img, *degrees),
            Filter::Sharpen(amount) => {
                // Unsharp mask as a 3x3 convolution: the center is
                // boosted, the cross neighbors subtracted.
                let a = *amount;
                let kernel: [f32; 9] = [0.0, -a, 0.0, -a, 1.0 + 4.0 * a, -a, 0.0, -a, 0.0];
                img = map_rgb_image(img, |rgb| filter3x3_rgb(&rgb, &kernel));
            }
            Filter::Saturation(amount) => {
                img = map_rgb_image(img, |rgb| saturate(rgb, *amount));
            }
            Filter::Vibrance(amount) => {
                img = map_rgb_image(img, |rgb| vibrance(rgb, *amount));
            }
            Filter::Rgb(r, g, b) => {
                img = map_rgb(img, |pr, pg, pb| {
                    [
                        (pr as i32 + r).clamp(0, 255) as u8,
                        (pg as i32 + g).clamp(0, 255) as u8,
                        (pb as i32 + b).clamp(0, 255) as u8,
                    ]
                });
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
                img = map_rgb_image(img, equalize);
            }
            Filter::Enhance => {
                img = map_rgb_image(img, auto_levels);
            }
            Filter::Rotate(degrees) => {
                let normalized = fold_degrees(*degrees);
                img = if normalized == 90.0 {
                    img.rotate90()
                } else if normalized == 180.0 {
                    rotate180_fast(&img)
                } else if normalized == 270.0 {
                    rotate270_fast(&img)
                } else if normalized == 0.0 {
                    img
                } else {
                    rotate_arbitrary(img, normalized, fill)
                };
            }
            Filter::Trim(tolerance) => img = trim(img, *tolerance),
            Filter::TrimSides(tolerance, sides) => img = trim_sides(img, *tolerance, *sides),
            // Watermarks are applied by the engine, which owns
            // the resolved images; one reaching this function is
            // unresolved by construction and leaves pixels alone.
            // (The engine never routes them here.)
            // Region redaction runs in the engine, which
            // validates the rectangle against the image size
            // in effect at that point. (Same routing.)
            Filter::BlurRegion { .. } | Filter::PixelateRegion { .. } => {}
            // Region removal runs in the engine, which owns
            // the inpainting provider. (Same routing as the
            // other region filters.)
            Filter::RemoveRegion { .. } => {}
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
            // Class operations are engine-routed (they need
            // the detection provider), like the face filters
            // above.
            Filter::BlurClass(_) | Filter::PixelateClass(_) | Filter::RecolorClass(..) => {}
            // DPR scales size targets in the engine, which
            // resolves it against the request as a whole (with
            // the client-hint fallback). (Same routing.)
            Filter::Dpr(_) => {}
            Filter::Sepia => img = sepia(img),
            Filter::Duotone(shadow, highlight) => img = duotone(img, *shadow, *highlight),
            Filter::Monochrome(color) => img = monochrome(img, *color),
            Filter::Pixelate(block) => img = pixelate(&img, *block),
            Filter::Flip(direction) => {
                img = match direction {
                    crate::url::FlipDirection::Horizontal => fliph_fast(&img),
                    crate::url::FlipDirection::Vertical => flipv_fast(&img),
                }
            }
            Filter::Radius(radii) => img = rounded_corners(&img, *radii),
            Filter::Border(width, color) => img = draw_border(&img, *width, *color),
            Filter::Mask(shape) => img = mask_shape(&img, *shape),
            Filter::Fill(color) => options.fill = Some(*color),
            Filter::MaxBytes(n) => options.max_bytes = Some(*n),
            Filter::KeepProfile => options.keep_profile = true,
            Filter::Quality(q) => options.quality = Some(q.clone()),
            Filter::Palette(n) => options.palette = Some(*n),
            Filter::NoUpscale => options.no_upscale = true,
            Filter::Format(spec) => {
                options.format = Some(spec.format);
                options.format_options = spec.options;
            }
        }
    }
    (img, options)
}

/// The canvas a rotation of a `w` by `h` image lands on:
/// quarter turns swap the axes, an arbitrary angle gets the
/// rotated bounding box. Shared with the engine, which
/// checks the canvas against the output pixel budget before
/// the rotation allocates it (issue #182, audit R2-12), and
/// with the fuzz harness, which asserts every rotation lands
/// on it (issue #415).
pub fn rotated_canvas(w: u32, h: u32, degrees: f32) -> (u32, u32) {
    folded_canvas(w, h, fold_degrees(degrees))
}

/// An angle in degrees folded into `[0, 360)`, the form
/// every rotation computes with.
fn fold_degrees(degrees: f32) -> f32 {
    ((degrees % 360.0) + 360.0) % 360.0
}

/// [`rotated_canvas`] for an angle [`fold_degrees`] has
/// already folded. Folding is not idempotent in `f32`:
/// folding a folded negative angle again can move its last
/// bits (`rotate(-113.0128)` folds to 246.98721, which
/// folds again to 246.98718), and on a 640x480 image the
/// two angles' bounding boxes differ by a pixel. So the
/// rotation sizes its canvas from the exact angle it
/// samples with, and the prediction above folds once too
/// (issue #415).
fn folded_canvas(w: u32, h: u32, normalized: f32) -> (u32, u32) {
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
/// `degrees` arrives folded by [`fold_degrees`].
fn rotate_arbitrary(img: DynamicImage, degrees: f32, fill: Option<[u8; 3]>) -> DynamicImage {
    let has_alpha = img.color().has_alpha();
    let background: [u8; 4] = if has_alpha {
        [0, 0, 0, 0]
    } else {
        let [r, g, b] = fill.unwrap_or([255, 255, 255]);
        [r, g, b, 255]
    };
    let src = img.into_rgba8();
    let (sw, sh) = (src.width(), src.height());
    let (w, h) = (sw as f32, sh as f32);
    let theta = degrees.to_radians();
    let (sin, cos) = theta.sin_cos();
    let (new_w, new_h) = folded_canvas(sw, sh, degrees);
    let (scx, scy) = ((w - 1.0) / 2.0, (h - 1.0) / 2.0);
    let (dcx, dcy) = ((new_w as f32 - 1.0) / 2.0, (new_h as f32 - 1.0) / 2.0);
    let src_raw = src.as_raw();
    let sample = |x: i32, y: i32| -> [f32; 4] {
        if x < 0 || y < 0 || x >= sw as i32 || y >= sh as i32 {
            [
                background[0] as f32,
                background[1] as f32,
                background[2] as f32,
                background[3] as f32,
            ]
        } else {
            let i = ((y as u32 * sw + x as u32) * 4) as usize;
            [
                src_raw[i] as f32,
                src_raw[i + 1] as f32,
                src_raw[i + 2] as f32,
                src_raw[i + 3] as f32,
            ]
        }
    };
    let mut out = image::RgbaImage::from_pixel(new_w, new_h, image::Rgba(background));
    // Every output pixel is a pure function of the source,
    // so the canvas splits into row bands across scoped
    // threads without changing a byte; small
    // canvases run sequentially on the calling thread.
    let row_len = (new_w * 4) as usize;
    let threads = band_threads();
    let band_rows = if threads > 1 && new_h >= 128 {
        (new_h as usize).div_ceil(threads)
    } else {
        new_h as usize
    };
    let out_raw = out.as_mut();
    std::thread::scope(|s| {
        for (bi, band) in out_raw.chunks_mut(band_rows * row_len).enumerate() {
            let y_start = (bi * band_rows) as u32;
            let sample = &sample;
            s.spawn(move || {
                let rows = band.len() / row_len;
                for ly in 0..rows {
                    let y = y_start + ly as u32;
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
                        let o = ly * row_len + (x * 4) as usize;
                        band[o] = px[0].round() as u8;
                        band[o + 1] = px[1].round() as u8;
                        band[o + 2] = px[2].round() as u8;
                        band[o + 3] = px[3].round() as u8;
                    }
                }
            });
        }
    });
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
fn map_rgb(img: DynamicImage, f: impl Fn(u8, u8, u8) -> [u8; 3] + Sync) -> DynamicImage {
    if img.color().has_alpha() {
        let mut rgba = img.into_rgba8();
        let w = rgba.width() as usize;
        map_rgb_bands::<4>(rgba.as_mut(), w, &f);
        DynamicImage::ImageRgba8(rgba)
    } else {
        let mut rgb = img.into_rgb8();
        let w = rgb.width() as usize;
        map_rgb_bands::<3>(rgb.as_mut(), w, &f);
        DynamicImage::ImageRgb8(rgb)
    }
}

/// Apply a per-pixel RGB mapping over a raw channel buffer,
/// in row bands across scoped threads when the frame is
/// large enough for the split to pay for itself. Each output
/// pixel depends only on the same input pixel, so banding
/// cannot change the result; small frames run sequentially
/// on the calling thread.
fn map_rgb_bands<const CH: usize>(
    buf: &mut [u8],
    width: usize,
    f: &(impl Fn(u8, u8, u8) -> [u8; 3] + Sync),
) {
    let row = width * CH;
    let rows = buf.len() / row.max(1);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4);
    if threads < 2 || rows < 256 || buf.len() < 3_000_000 {
        for p in buf.as_chunks_mut::<CH>().0 {
            let [r, g, b] = f(p[0], p[1], p[2]);
            p[0] = r;
            p[1] = g;
            p[2] = b;
        }
        return;
    }
    let band_rows = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for band in buf.chunks_mut(band_rows * row) {
            s.spawn(move || {
                for p in band.as_chunks_mut::<CH>().0 {
                    let [r, g, b] = f(p[0], p[1], p[2]);
                    p[0] = r;
                    p[1] = g;
                    p[2] = b;
                }
            });
        }
    });
}

fn map_rgb_image(
    img: DynamicImage,
    f: impl FnOnce(image::RgbImage) -> image::RgbImage,
) -> DynamicImage {
    if img.color().has_alpha() {
        let mapped = f(img.to_rgb8());
        let mut out = img.into_rgba8();
        for (o, m) in out.pixels_mut().zip(mapped.pixels()) {
            o[0] = m[0];
            o[1] = m[1];
            o[2] = m[2];
        }
        DynamicImage::ImageRgba8(out)
    } else {
        DynamicImage::ImageRgb8(f(img.into_rgb8()))
    }
}

// ----------------------------------------------------------------
// Fast paths. Every function in this section is a
// byte-identical stand-in for the image-crate operation it
// replaces: the identity is pinned by tests in this module
// against the crate over fixture sweeps, and hue was also
// checked exhaustively over the RGB cube, outside the suite.
// ----------------------------------------------------------------

/// Below this frame size (in subpixels) the crate's
/// permutation loops are cache-resident and beat a thread
/// spawn plus a fresh destination buffer; the fast paths
/// gate on it and fall back to the crate.
const PERM_FAST_MIN_SUBPIXELS: usize = 4_000_000;

fn blur_kernel_size(sigma: f32) -> u32 {
    let possible_size = (((((sigma - 0.8) / 0.3) + 1.) * 2.) + 1.).max(3.) as u32;
    if possible_size.is_multiple_of(2) {
        return possible_size + 1;
    }
    possible_size
}

fn blur_gaussian_kernel(width: usize, sigma: f32) -> Vec<f32> {
    let mut sum_norm: f32 = 0f32;
    let mut kernel = vec![0f32; width];
    let scale = 1f32 / (f32::sqrt(2f32 * std::f32::consts::PI) * sigma);
    let mean = (width / 2) as f32;
    for (x, weight) in kernel.iter_mut().enumerate() {
        let new_weight = f32::exp(-0.5f32 * f32::powf((x as f32 - mean) / sigma, 2.0f32)) * scale;
        *weight = new_weight;
        sum_norm += new_weight;
    }
    if sum_norm != 0f32 {
        let sum_scale = 1f32 / sum_norm;
        for weight in &mut kernel {
            *weight *= sum_scale;
        }
    }
    kernel
}

fn blur_prepare_kernel<T: Copy + PartialEq + Default>(kernel: &[T]) -> Vec<T> {
    let zero = T::default();
    let mut new_kernel = kernel.to_vec();
    while new_kernel.len() > 2
        && new_kernel.last() == Some(&zero)
        && new_kernel.first() == Some(&zero)
    {
        new_kernel.remove(0);
        new_kernel.remove(new_kernel.len() - 1);
    }
    new_kernel
}

// The min/max chain is the crate's own expression; clamp
// would differ on NaN, and the port stays bit-faithful.
#[allow(clippy::manual_clamp)]
fn blur_q15(w: f32) -> u32 {
    (w * 32768.0).min(32767.0).max(0.0) as u32
}

fn blur_q15_out(acc: u32) -> u8 {
    ((acc + (1 << 14)) >> 15).min(255) as u8
}

fn blur_par_bands<T, F>(dst: &mut [T], row_len: usize, rows: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    if rows == 0 || row_len == 0 {
        return;
    }
    let band_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4)
        .min(rows);
    if band_threads <= 1 {
        f(0, dst);
        return;
    }
    let band = rows.div_ceil(band_threads);
    let mut chunks: Vec<(usize, &mut [T])> = Vec::with_capacity(band_threads);
    let mut rest = dst;
    let mut start = 0usize;
    while !rest.is_empty() {
        let take = (band * row_len).min(rest.len());
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(take);
        chunks.push((start, head));
        start += take / row_len;
        rest = tail;
    }
    std::thread::scope(|s| {
        for (start_row, chunk) in chunks {
            let f = &f;
            s.spawn(move || f(start_row, chunk));
        }
    });
}

fn blur_clamp_idx(i: isize, len: usize) -> usize {
    i.clamp(0, len as isize - 1) as usize
}

fn blur_arena_row<const N: usize>(src_row: &[u8], pad: usize, out: &mut Vec<u8>) {
    let w = src_row.len();
    out.clear();
    out.reserve(w + 2 * pad * N);
    let first = &src_row[..N];
    let last = &src_row[w - N..];
    for _ in 0..pad {
        out.extend_from_slice(first);
    }
    out.extend_from_slice(src_row);
    for _ in 0..pad {
        out.extend_from_slice(last);
    }
}

fn blur_sym_row<const N: usize>(arena: &[u8], dst_row: &mut [u8], kernel: &[u32]) {
    let len = kernel.len();
    let half = len / 2;
    let hc = kernel[half];
    let (chunks, rem) = dst_row.as_chunks_mut::<4>();
    for (x, dst) in chunks.iter_mut().enumerate() {
        let src = &arena[x * 4..];
        let c = &src[half * N..half * N + 4];
        let mut k0 = c[0] as u32 * hc;
        let mut k1 = c[1] as u32 * hc;
        let mut k2 = c[2] as u32 * hc;
        let mut k3 = c[3] as u32 * hc;
        for i in 0..half {
            let other = len - 1 - i;
            let fw = &src[i * N..i * N + 4];
            let bw = &src[other * N..other * N + 4];
            k0 += (fw[0] as u32 + bw[0] as u32) * kernel[i];
            k1 += (fw[1] as u32 + bw[1] as u32) * kernel[i];
            k2 += (fw[2] as u32 + bw[2] as u32) * kernel[i];
            k3 += (fw[3] as u32 + bw[3] as u32) * kernel[i];
        }
        dst[0] = blur_q15_out(k0);
        dst[1] = blur_q15_out(k1);
        dst[2] = blur_q15_out(k2);
        dst[3] = blur_q15_out(k3);
    }
    let dz = chunks.len() * 4;
    for (x, dst) in rem.iter_mut().enumerate() {
        let src = &arena[dz + x..];
        let mut acc = src[half * N] as u32 * hc;
        for i in 0..half {
            let other = len - 1 - i;
            acc += (src[i * N] as u32 + src[other * N] as u32) * kernel[i];
        }
        *dst = blur_q15_out(acc);
    }
}

fn blur_sym_col(brows: &[&[u8]], dst_row: &mut [u8], kernel: &[u32]) {
    let len = kernel.len();
    let half = len / 2;
    let hc = kernel[half];
    let center = brows[half];
    let total = dst_row.len();
    let n32 = total / 32;
    let n16 = (total - n32 * 32) / 16;
    let (head, tail) = dst_row.split_at_mut(n32 * 32);
    for (x, chunk) in head.chunks_mut(32).enumerate() {
        let cx = x * 32;
        let mut store0 = [0u32; 16];
        let mut store1 = [0u32; 16];
        for (dst, src) in store0.iter_mut().zip(&center[cx..cx + 16]) {
            *dst = *src as u32 * hc;
        }
        for (dst, src) in store1.iter_mut().zip(&center[cx + 16..cx + 32]) {
            *dst = *src as u32 * hc;
        }
        for i in 0..half {
            let other = len - 1 - i;
            let fw0 = &brows[i][cx..cx + 16];
            let bw0 = &brows[other][cx..cx + 16];
            let fw1 = &brows[i][cx + 16..cx + 32];
            let bw1 = &brows[other][cx + 16..cx + 32];
            for ((dst, f), b) in store0.iter_mut().zip(fw0).zip(bw0) {
                *dst += (*f as u32 + *b as u32) * kernel[i];
            }
            for ((dst, f), b) in store1.iter_mut().zip(fw1).zip(bw1) {
                *dst += (*f as u32 + *b as u32) * kernel[i];
            }
        }
        let (lo, hi) = chunk.split_at_mut(16);
        for (src, dst) in store0.iter().zip(lo) {
            *dst = blur_q15_out(*src);
        }
        for (src, dst) in store1.iter().zip(hi) {
            *dst = blur_q15_out(*src);
        }
    }
    let (head16, scalar) = tail.split_at_mut(n16 * 16);
    for (x, chunk) in head16.chunks_mut(16).enumerate() {
        let cx = n32 * 32 + x * 16;
        let mut store = [0u32; 16];
        for (dst, src) in store.iter_mut().zip(&center[cx..cx + 16]) {
            *dst = *src as u32 * hc;
        }
        for i in 0..half {
            let other = len - 1 - i;
            let fw = &brows[i][cx..cx + 16];
            let bw = &brows[other][cx..cx + 16];
            for ((dst, f), b) in store.iter_mut().zip(fw).zip(bw) {
                *dst += (*f as u32 + *b as u32) * kernel[i];
            }
        }
        for (src, dst) in store.iter().zip(chunk.iter_mut()) {
            *dst = blur_q15_out(*src);
        }
    }
    let scalar_start = n32 * 32 + n16 * 16;
    for (j, dst) in scalar.iter_mut().enumerate() {
        let cx = scalar_start + j;
        let mut acc = center[cx] as u32 * hc;
        for i in 0..half {
            let other = len - 1 - i;
            acc += (brows[i][cx] as u32 + brows[other][cx] as u32) * kernel[i];
        }
        *dst = blur_q15_out(acc);
    }
}

pub fn blur_ring<const N: usize>(src: &[u8], w: usize, h: usize, sigma: f32) -> Vec<u8> {
    let sigma = if sigma == 0.0 { 0.8 } else { sigma };
    if w == 0 || h == 0 {
        return src.to_vec();
    }
    let f_kernel = blur_gaussian_kernel(blur_kernel_size(sigma) as usize, sigma);
    let kernel: Vec<u32> =
        blur_prepare_kernel(&f_kernel.iter().map(|&v| blur_q15(v)).collect::<Vec<_>>());
    let half = kernel.len() / 2;
    let pad = half.max(1);
    let row_len = w * N;
    let mut out = vec![0u8; src.len()];
    blur_par_bands(&mut out, row_len, h, |y0, band| {
        let perm_band_rows = band.len() / row_len;
        let klen = kernel.len();
        let mut ring = vec![0u8; klen * row_len];
        let mut arena: Vec<u8> = Vec::new();
        let mut next_row = blur_clamp_idx(y0 as isize - half as isize, h);
        for ly in 0..perm_band_rows {
            let y = y0 + ly;
            let need_hi = blur_clamp_idx(y as isize + half as isize, h);
            while next_row <= need_hi {
                let slot = next_row % klen;
                blur_arena_row::<N>(
                    &src[next_row * row_len..(next_row + 1) * row_len],
                    pad,
                    &mut arena,
                );
                blur_sym_row::<N>(
                    &arena,
                    &mut ring[slot * row_len..(slot + 1) * row_len],
                    &kernel,
                );
                next_row += 1;
            }
            let mut brows: Vec<&[u8]> = Vec::with_capacity(klen);
            for k in 0..klen {
                let sy = blur_clamp_idx(y as isize + k as isize - half as isize, h);
                brows.push(&ring[(sy % klen) * row_len..(sy % klen + 1) * row_len]);
            }
            blur_sym_col(&brows, &mut band[ly * row_len..(ly + 1) * row_len], &kernel);
        }
    });
    out
}

/// Banded-transient variant for large kernels: horizontal pass
/// into a full transient buffer (rows in parallel), then the
/// column pass in output-row bands over the shared transient.
/// Same inner functions, so the same per-pixel arithmetic.
pub fn blur_transient<const N: usize>(src: &[u8], w: usize, h: usize, sigma: f32) -> Vec<u8> {
    let sigma = if sigma == 0.0 { 0.8 } else { sigma };
    if w == 0 || h == 0 {
        return src.to_vec();
    }
    let f_kernel = blur_gaussian_kernel(blur_kernel_size(sigma) as usize, sigma);
    let kernel: Vec<u32> =
        blur_prepare_kernel(&f_kernel.iter().map(|&v| blur_q15(v)).collect::<Vec<_>>());
    let half = kernel.len() / 2;
    let pad = half.max(1);
    let row_len = w * N;
    let mut transient = vec![0u8; src.len()];
    blur_par_bands(&mut transient, row_len, h, |y0, band| {
        let mut arena: Vec<u8> = Vec::new();
        for (ly, dst_row) in band.chunks_mut(row_len).enumerate() {
            let y = y0 + ly;
            blur_arena_row::<N>(&src[y * row_len..(y + 1) * row_len], pad, &mut arena);
            blur_sym_row::<N>(&arena, dst_row, &kernel);
        }
    });
    let mut out = vec![0u8; src.len()];
    blur_par_bands(&mut out, row_len, h, |y0, band| {
        for (ly, dst_row) in band.chunks_mut(row_len).enumerate() {
            let y = y0 + ly;
            let mut brows: Vec<&[u8]> = Vec::with_capacity(kernel.len());
            for k in 0..kernel.len() {
                let sy = blur_clamp_idx(y as isize + k as isize - half as isize, h);
                brows.push(&transient[sy * row_len..(sy + 1) * row_len]);
            }
            blur_sym_col(&brows, dst_row, &kernel);
        }
    });
    out
}

pub(crate) fn blur_fast(img: &DynamicImage, sigma: f32) -> DynamicImage {
    // Ring for kernel < 55 (the crate's own cutoff), transient
    // above it: same split as the crate, both banded.
    let sigma = if sigma == 0.0 { 0.8 } else { sigma };
    let klen = blur_kernel_size(sigma) as usize;
    macro_rules! arm {
        ($i:expr, $ch:expr, $ctor:expr) => {{
            let (w, h) = ($i.width() as usize, $i.height() as usize);
            let out = if klen < 55 {
                blur_ring::<$ch>($i.as_raw(), w, h, sigma)
            } else {
                blur_transient::<$ch>($i.as_raw(), w, h, sigma)
            };
            $ctor(out)
        }};
    }
    match img {
        DynamicImage::ImageLuma8(i) => arm!(i, 1, |out| DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(i.width(), i.height(), out).expect("blur size")
        )),
        DynamicImage::ImageLumaA8(i) => arm!(i, 2, |out| DynamicImage::ImageLumaA8(
            image::GrayAlphaImage::from_raw(i.width(), i.height(), out).expect("blur size")
        )),
        DynamicImage::ImageRgb8(i) => arm!(i, 3, |out| DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(i.width(), i.height(), out).expect("blur size")
        )),
        DynamicImage::ImageRgba8(i) => arm!(i, 4, |out| DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(i.width(), i.height(), out).expect("blur size")
        )),
        other => other.blur(sigma),
    }
}

fn band_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4)
}

fn perm_band_rows(rows: usize, total_subpx: usize) -> usize {
    if band_threads() > 1 && total_subpx >= 3_000_000 {
        rows.div_ceil(band_threads())
    } else {
        rows
    }
}

fn map_pixels_bands<const CH: usize>(
    buf: &mut [u8],
    width: usize,
    f: &(impl Fn(&mut [u8; CH]) + Sync),
) {
    let total = buf.len();
    let px = buf.as_chunks_mut::<CH>().0;
    let rows = px.len() / width.max(1);
    let br = perm_band_rows(rows, total);
    if br >= rows {
        for p in px {
            f(p);
        }
        return;
    }
    std::thread::scope(|s| {
        for band in px.chunks_mut(br * width) {
            s.spawn(move || {
                for p in band {
                    f(p);
                }
            });
        }
    });
}

fn rot270_px<const CH: usize>(src: &[[u8; CH]], w: usize, h: usize, dst: &mut [[u8; CH]]) {
    let br = perm_band_rows(w, w * h * CH);
    std::thread::scope(|s| {
        for (bi, band) in dst.chunks_mut(br * h).enumerate() {
            let y0 = bi * br;
            s.spawn(move || {
                let tile = 64usize;
                let mut ty = 0usize;
                while ty < h {
                    let t_end = (ty + tile).min(h);
                    for (ly, orow) in band.chunks_mut(h).enumerate() {
                        let c = w - 1 - (y0 + ly);
                        for y in ty..t_end {
                            orow[y] = src[y * w + c];
                        }
                    }
                    ty = t_end;
                }
            });
        }
    });
}

fn rot180_px<const CH: usize>(src: &[[u8; CH]], _w: usize, _h: usize, dst: &mut [[u8; CH]]) {
    let n = src.len();
    let br = perm_band_rows(n, n * CH);
    std::thread::scope(|s| {
        let mut rest = dst;
        let mut start = 0usize;
        while !rest.is_empty() {
            let take = br.min(rest.len());
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(take);
            let st = start;
            start += take;
            rest = tail;
            s.spawn(move || {
                for (i, px) in head.iter_mut().enumerate() {
                    *px = src[n - 1 - (st + i)];
                }
            });
        }
    });
}

fn fliph_px<const CH: usize>(src: &[[u8; CH]], w: usize, h: usize, dst: &mut [[u8; CH]]) {
    let br = perm_band_rows(h, w * h * CH);
    std::thread::scope(|s| {
        for (sband, dband) in src.chunks(br * w).zip(dst.chunks_mut(br * w)) {
            s.spawn(move || {
                for (srow, drow) in sband.chunks(w).zip(dband.chunks_mut(w)) {
                    for (i, px) in drow.iter_mut().enumerate() {
                        *px = srow[w - 1 - i];
                    }
                }
            });
        }
    });
}

fn flipv_px<const CH: usize>(src: &[[u8; CH]], w: usize, h: usize, dst: &mut [[u8; CH]]) {
    let br = perm_band_rows(h, w * h * CH);
    std::thread::scope(|s| {
        let mut rest = dst;
        let mut y0 = 0usize;
        while !rest.is_empty() {
            let take = (br * w).min(rest.len());
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(take);
            let rows = take / w;
            let src_rows = &src[(h - y0 - rows) * w..(h - y0) * w];
            s.spawn(move || {
                for r in 0..rows {
                    head[r * w..(r + 1) * w]
                        .copy_from_slice(&src_rows[(rows - 1 - r) * w..(rows - r) * w]);
                }
            });
            y0 += rows;
            rest = tail;
        }
    });
}

fn run_perm<const CH: usize>(
    raw: &[u8],
    w: usize,
    h: usize,
    op: impl Fn(&[[u8; CH]], usize, usize, &mut [[u8; CH]]),
) -> Vec<u8> {
    let src = raw.as_chunks::<CH>().0;
    let mut dst = vec![0u8; raw.len()];
    let dstp = dst.as_chunks_mut::<CH>().0;
    op(src, w, h, dstp);
    dst
}

macro_rules! perm_fn {
    ($name:ident, $op:expr, $swap:expr, $fallback:ident) => {
        pub(crate) fn $name(img: &DynamicImage) -> DynamicImage {
            if (img.width() as usize)
                * (img.height() as usize)
                * (img.color().bytes_per_pixel() as usize)
                < PERM_FAST_MIN_SUBPIXELS
            {
                return img.$fallback();
            }
            match img {
                DynamicImage::ImageLuma8(i) => {
                    let (w, h) = (i.width() as usize, i.height() as usize);
                    let out = run_perm::<1>(i.as_raw(), w, h, $op);
                    let (ow, oh) = if $swap { (h, w) } else { (w, h) };
                    DynamicImage::ImageLuma8(
                        image::GrayImage::from_raw(ow as u32, oh as u32, out).expect("perm size"),
                    )
                }
                DynamicImage::ImageLumaA8(i) => {
                    let (w, h) = (i.width() as usize, i.height() as usize);
                    let out = run_perm::<2>(i.as_raw(), w, h, $op);
                    let (ow, oh) = if $swap { (h, w) } else { (w, h) };
                    DynamicImage::ImageLumaA8(
                        image::GrayAlphaImage::from_raw(ow as u32, oh as u32, out)
                            .expect("perm size"),
                    )
                }
                DynamicImage::ImageRgb8(i) => {
                    let (w, h) = (i.width() as usize, i.height() as usize);
                    let out = run_perm::<3>(i.as_raw(), w, h, $op);
                    let (ow, oh) = if $swap { (h, w) } else { (w, h) };
                    DynamicImage::ImageRgb8(
                        image::RgbImage::from_raw(ow as u32, oh as u32, out).expect("perm size"),
                    )
                }
                DynamicImage::ImageRgba8(i) => {
                    let (w, h) = (i.width() as usize, i.height() as usize);
                    let out = run_perm::<4>(i.as_raw(), w, h, $op);
                    let (ow, oh) = if $swap { (h, w) } else { (w, h) };
                    DynamicImage::ImageRgba8(
                        image::RgbaImage::from_raw(ow as u32, oh as u32, out).expect("perm size"),
                    )
                }
                other => other.$fallback(),
            }
        }
    };
}

perm_fn!(rotate270_fast, rot270_px, true, rotate270);
perm_fn!(rotate180_fast, rot180_px, false, rotate180);
perm_fn!(fliph_fast, fliph_px, false, fliph);
perm_fn!(flipv_fast, flipv_px, false, flipv);

// ---------------- huerotate ----------------

fn hue_matrix(value: i32) -> [f64; 9] {
    let angle: f64 = value as f64;
    let cosv = angle.to_radians().cos();
    let sinv = angle.to_radians().sin();
    [
        0.213 + cosv * 0.787 - sinv * 0.213,
        0.715 - cosv * 0.715 - sinv * 0.715,
        0.072 - cosv * 0.072 + sinv * 0.928,
        0.213 - cosv * 0.213 + sinv * 0.143,
        0.715 + cosv * 0.285 + sinv * 0.140,
        0.072 - cosv * 0.072 - sinv * 0.283,
        0.213 - cosv * 0.213 - sinv * 0.787,
        0.715 - cosv * 0.715 + sinv * 0.715,
        0.072 + cosv * 0.928 + sinv * 0.072,
    ]
}

fn hue_apply<const CH: usize>(buf: &mut [u8], width: usize, m: &[f64; 9]) {
    // The crate reads pixels through channels4(), which pads
    // missing channels with 255 (it does not replicate luma):
    // Luma inputs are (v, 255, 255); LumaA inputs are
    // (v, alpha, 255) and its output alpha is clamp(new_g),
    // because from_channels keeps the first CHANNELS values.
    map_pixels_bands::<CH>(buf, width, &|p: &mut [u8; CH]| {
        let r = p[0] as f64;
        let (g, b) = match CH {
            1 => (255.0, 255.0),
            2 => (p[1] as f64, 255.0),
            _ => (p[1] as f64, p[2] as f64),
        };
        let nr = m[0] * r + m[1] * g + m[2] * b;
        if CH == 2 {
            let ng = m[3] * r + m[4] * g + m[5] * b;
            p[0] = nr.clamp(0.0, 255.0) as u8;
            p[1] = ng.clamp(0.0, 255.0) as u8;
            return;
        }
        p[0] = nr.clamp(0.0, 255.0) as u8;
        if CH >= 3 {
            let ng = m[3] * r + m[4] * g + m[5] * b;
            let nb = m[6] * r + m[7] * g + m[8] * b;
            p[1] = ng.clamp(0.0, 255.0) as u8;
            p[2] = nb.clamp(0.0, 255.0) as u8;
        }
    });
}

pub(crate) fn huerotate_fast(img: DynamicImage, value: i32) -> DynamicImage {
    let m = hue_matrix(value);
    match img {
        DynamicImage::ImageLuma8(mut i) => {
            let w = i.width() as usize;
            hue_apply::<1>(i.as_mut(), w, &m);
            DynamicImage::ImageLuma8(i)
        }
        DynamicImage::ImageLumaA8(mut i) => {
            let w = i.width() as usize;
            hue_apply::<2>(i.as_mut(), w, &m);
            DynamicImage::ImageLumaA8(i)
        }
        DynamicImage::ImageRgb8(mut i) => {
            let w = i.width() as usize;
            hue_apply::<3>(i.as_mut(), w, &m);
            DynamicImage::ImageRgb8(i)
        }
        DynamicImage::ImageRgba8(mut i) => {
            let w = i.width() as usize;
            hue_apply::<4>(i.as_mut(), w, &m);
            DynamicImage::ImageRgba8(i)
        }
        other => other.huerotate(value),
    }
}

// ---------------- brighten ----------------

pub(crate) fn brighten_fast(img: DynamicImage, value: i32) -> DynamicImage {
    let lut: [u8; 256] = std::array::from_fn(|b| (b as i32 + value).clamp(0, 255) as u8);
    fn apply<const CH: usize, const ALPHA: bool>(buf: &mut [u8], width: usize, lut: &[u8; 256]) {
        map_pixels_bands::<CH>(buf, width, &|p: &mut [u8; CH]| {
            let n = if ALPHA { CH - 1 } else { CH };
            for c in 0..n {
                p[c] = lut[p[c] as usize];
            }
        });
    }
    match img {
        DynamicImage::ImageLuma8(mut i) => {
            let w = i.width() as usize;
            apply::<1, false>(i.as_mut(), w, &lut);
            DynamicImage::ImageLuma8(i)
        }
        DynamicImage::ImageLumaA8(mut i) => {
            let w = i.width() as usize;
            apply::<2, true>(i.as_mut(), w, &lut);
            DynamicImage::ImageLumaA8(i)
        }
        DynamicImage::ImageRgb8(mut i) => {
            let w = i.width() as usize;
            apply::<3, false>(i.as_mut(), w, &lut);
            DynamicImage::ImageRgb8(i)
        }
        DynamicImage::ImageRgba8(mut i) => {
            let w = i.width() as usize;
            apply::<4, true>(i.as_mut(), w, &lut);
            DynamicImage::ImageRgba8(i)
        }
        other => other.brighten(value),
    }
}

/// 3x3 convolution of an RGB image with the exact arithmetic
/// of the image crate's `filter3x3` (same tap order, same
/// inverse-sum scaling, same clamp and truncation), but over
/// raw row slices instead of per-pixel trait calls, and with
/// one deliberate divergence: the crate writes only interior
/// pixels and leaves a zeroed border, so sharpening framed
/// every output in black and turned an image under 3px on
/// an axis entirely black (issue #378). Here the border
/// pixels are computed with their taps clamped to the
/// nearest edge pixel (edge replication, the same rule the
/// `convolve` filter documents), and small images are
/// computed whole. Zero-weight taps are skipped: every
/// input channel is a non-negative f32, so a zero tap only
/// ever added +0.0, which cannot change the running sum.
fn filter3x3_rgb(img: &image::RgbImage, kernel: &[f32; 9]) -> image::RgbImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut out = image::RgbImage::new(w as u32, h as u32);
    if w == 0 || h == 0 {
        return out;
    }
    let src = img.as_raw();
    let dst: &mut [u8] = out.as_mut();
    let stride = w * 3;
    let sum: f32 = kernel.iter().sum();
    let inverse_sum = if sum == 0.0 { 1.0 } else { 1.0 / sum };
    // (row delta, byte offset of the tap column within its
    // row relative to the pixel's own first byte, weight)
    let taps: Vec<(usize, usize, f32)> = kernel
        .iter()
        .enumerate()
        .filter(|(_, k)| **k != 0.0)
        .map(|(ki, k)| (ki / 3, (ki % 3) * 3, *k))
        .collect();
    if w >= 3 && h >= 3 {
        for y in 1..h - 1 {
            let rows = [
                &src[(y - 1) * stride..y * stride],
                &src[y * stride..(y + 1) * stride],
                &src[(y + 1) * stride..(y + 2) * stride],
            ];
            for x in 1..w - 1 {
                let xb = x * 3;
                let mut t = [0f32; 3];
                for &(rd, coff, k) in &taps {
                    let b = xb + coff - 3;
                    let row = rows[rd];
                    t[0] += row[b] as f32 * k;
                    t[1] += row[b + 1] as f32 * k;
                    t[2] += row[b + 2] as f32 * k;
                }
                let base = y * stride + xb;
                for c in 0..3 {
                    dst[base + c] = (t[c] * inverse_sum).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    // The border ring, and every pixel of an image under
    // 3px on either axis: the same sum, with each tap's
    // coordinates clamped into the frame.
    for y in 0..h {
        for x in 0..w {
            if w >= 3 && h >= 3 && (1..w - 1).contains(&x) && (1..h - 1).contains(&y) {
                continue;
            }
            let mut t = [0f32; 3];
            for &(rd, coff, k) in &taps {
                let sy = (y + rd).saturating_sub(1).min(h - 1);
                let sx = (x + coff / 3).saturating_sub(1).min(w - 1);
                let b = sy * stride + sx * 3;
                t[0] += src[b] as f32 * k;
                t[1] += src[b + 1] as f32 * k;
                t[2] += src[b + 2] as f32 * k;
            }
            let base = y * stride + x * 3;
            for c in 0..3 {
                dst[base + c] = (t[c] * inverse_sum).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Contrast through a 256-entry lookup table: the image
/// crate's per-channel formula (scale about the midpoint by
/// `((100 + amount) / 100)^2`, clamp, truncate) depends only
/// on the input byte, so evaluating it once per byte value
/// with the identical f32 arithmetic and then mapping bytes
/// is byte-identical to evaluating it per pixel.
fn contrast_in_place(mut img: image::RgbImage, amount: f32) -> image::RgbImage {
    let percent = ((100.0 + amount) / 100.0).powi(2);
    let mut lut = [0u8; 256];
    for (i, e) in lut.iter_mut().enumerate() {
        let c = i as f32;
        let d = ((c / 255.0 - 0.5) * percent + 0.5) * 255.0;
        *e = d.clamp(0.0, 255.0) as u8;
    }
    let w = img.width() as usize;
    map_rgb_bands::<3>(img.as_mut(), w, &|r, g, b| {
        [lut[r as usize], lut[g as usize], lut[b as usize]]
    });
    img
}

/// The 256-entry lookup table for one gamma divisor: entry
/// `v` is `255 * (v / 255)^(1 / g)`, rounded to the nearest
/// value, so 0 and 255 are fixed points at every setting
/// and the table is monotone. A divisor outside the
/// grammar's contract (non-finite, non-positive, or past
/// [`MAX_GAMMA`]) is sanitized like `blur_sigma` sanitizes
/// its input: it clamps into the accepted range instead of
/// reaching the power function with a value that would
/// invert the curve or produce NaNs.
fn gamma_lut(gamma: f32) -> [u8; 256] {
    let g = if gamma.is_finite() && gamma > 0.0 {
        gamma.min(MAX_GAMMA)
    } else {
        1.0
    };
    let exponent = 1.0 / g;
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        *entry = (255.0 * (i as f32 / 255.0).powf(exponent))
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Gamma correction (issue #313): remap every channel
/// through the divisor's lookup table. Alpha is never
/// touched (the `map_rgb` contract, like the other color
/// filters).
fn gamma_correct(img: DynamicImage, gamma: f32) -> DynamicImage {
    let lut = gamma_lut(gamma);
    map_rgb(img, |r, g, b| {
        [
            lut[usize::from(r)],
            lut[usize::from(g)],
            lut[usize::from(b)],
        ]
    })
}

/// The sRGB electro-optical transfer function and its
/// inverse (the standard piecewise forms): encoded value
/// to linear light, and back.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// The 256-entry lookup table for one exposure value in
/// stops: entry `v` is the sRGB encoding of `v`'s linear
/// value multiplied by `2^stops`, rounded to the nearest
/// value. The table is monotone, 0 is a fixed point at
/// every setting (zero light times any factor is zero),
/// and 255 is a fixed point for non-negative stops. A
/// stops value outside the grammar's contract is
/// sanitized like the gamma table sanitizes its divisor:
/// a non-finite value becomes 0 stops (the identity) and
/// the magnitude clamps to [`MAX_EXPOSURE`]. Zero stops
/// short-circuits to the identity table, so the no-op is
/// exact by construction rather than by round-trip luck.
fn exposure_lut(stops: f32) -> [u8; 256] {
    let stops = if stops.is_finite() {
        stops.clamp(-MAX_EXPOSURE, MAX_EXPOSURE)
    } else {
        0.0
    };
    if stops == 0.0 {
        return std::array::from_fn(|i| i as u8);
    }
    let factor = 2f32.powf(stops);
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let linear = srgb_to_linear(i as f32 / 255.0) * factor;
        *entry = (255.0 * linear_to_srgb(linear)).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Exposure (issue #317): remap every channel through the
/// stops lookup table, the EV-style linear-light multiply.
/// Alpha is never touched (the `map_rgb` contract).
fn exposure_correct(img: DynamicImage, stops: f32) -> DynamicImage {
    let lut = exposure_lut(stops);
    map_rgb(img, |r, g, b| {
        [
            lut[usize::from(r)],
            lut[usize::from(g)],
            lut[usize::from(b)],
        ]
    })
}

/// Clamp a tonal percent into the grammar's -100..=100
/// range: the sanitization the tonal filters apply to
/// values built away from the parser, mirroring how
/// `gamma_lut` sanitizes its divisor. A non-finite value
/// becomes 0, the identity.
fn percent_sanitized(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-100.0, 100.0)
    } else {
        0.0
    }
}

/// Highlights and shadows (issue #317): one tonal filter
/// with a percent argument per end of the scale. Each
/// pixel moves by a single offset shared by its three
/// channels (so hue survives; alpha is untouched, the
/// `map_rgb` contract), sized by where the pixel's
/// luminance sits: the shadows weight is `(1 - Y)^2` and
/// the highlights weight `Y^2`, with `Y` the Rec. 601
/// luma in 0..1. Either argument at its full 100 moves a
/// pixel at its own end of the scale by a quarter of full
/// scale (63.75 levels), reached only by pure black under
/// shadows or pure white under highlights; midtones move
/// by the blend of the two weights. The two ranges are
/// independent by construction: the shadows argument
/// cannot move white, nor highlights black.
fn highlights_shadows(img: &DynamicImage, highlights: f32, shadows: f32) -> DynamicImage {
    let highlights = percent_sanitized(highlights);
    let shadows = percent_sanitized(shadows);
    map_rgb(img.clone(), |r, g, b| {
        let y = luma(r, g, b) / 255.0;
        let shadow_weight = (1.0 - y) * (1.0 - y);
        let highlight_weight = y * y;
        let offset =
            ((shadows / 100.0) * shadow_weight + (highlights / 100.0) * highlight_weight) * 63.75;
        let shift = offset.round() as i32;
        [
            (i32::from(r) + shift).clamp(0, 255) as u8,
            (i32::from(g) + shift).clamp(0, 255) as u8,
            (i32::from(b) + shift).clamp(0, 255) as u8,
        ]
    })
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
    map_rgb(img.clone(), |r, g, b| {
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
    // Fast path for opaque RGB sources: the RGBA path below
    // only ever reads and writes their three color channels
    // (the synthesized alpha is dropped again on the way
    // back), so working on the RGB buffer directly is
    // byte-identical while skipping both conversions.
    if let DynamicImage::ImageRgb8(rgb) = img {
        let out = convolve_raw::<3>(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            kernel,
            size,
            radius,
            scale,
        );
        return DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(rgb.width(), rgb.height(), out).unwrap(),
        );
    }
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let out = convolve_raw::<4>(rgba.as_raw(), w, h, kernel, size, radius, scale);
    let out = image::RgbaImage::from_raw(w, h, out).unwrap();
    if has_alpha {
        DynamicImage::ImageRgba8(out)
    } else {
        DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(out).to_rgb8())
    }
}

/// Raw-slice convolution core for [`convolve`]: the same
/// clamped sampling and the same f64 accumulation order (row
/// of taps, then column, zero weights skipped) as the
/// per-pixel original, without per-tap trait calls. `CH` is
/// the source's channel count; only the first three channels
/// are convolved or written, the rest pass through from the
/// copied source buffer.
fn convolve_raw<const CH: usize>(
    src: &[u8],
    width: u32,
    height: u32,
    kernel: &[f64],
    size: usize,
    radius: i32,
    scale: f64,
) -> Vec<u8> {
    let (w, h) = (width as i32, height as i32);
    let stride = width as usize * CH;
    let mut out = src.to_vec();
    // One output pixel: taps in kernel order (row, then
    // column), zero weights skipped, f64 throughout.
    let acc_at = |x: i32, y: i32, clamped: bool| -> [f64; 3] {
        let mut acc = [0.0f64; 3];
        for ky in 0..size as i32 {
            let sy = if clamped {
                (y + ky - radius).clamp(0, h - 1)
            } else {
                y + ky - radius
            } as usize;
            for kx in 0..size as i32 {
                let k = kernel[(ky as usize) * size + kx as usize] * scale;
                if k == 0.0 {
                    continue;
                }
                let sx = if clamped {
                    (x + kx - radius).clamp(0, w - 1)
                } else {
                    x + kx - radius
                } as usize;
                let base = sy * stride + sx * CH;
                for (c, a) in acc.iter_mut().enumerate() {
                    *a += k * f64::from(src[base + c]);
                }
            }
        }
        acc
    };
    let r = radius;
    let interior = w > 2 * r && h > 2 * r;
    for y in 0..h {
        for x in 0..w {
            let clamped = !(interior && x >= r && x < w - r && y >= r && y < h - r);
            let acc = acc_at(x, y, clamped);
            let base = y as usize * stride + x as usize * CH;
            for (c, a) in acc.iter().enumerate() {
                out[base + c] = round_clamp(*a);
            }
        }
    }
    out
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
pub fn sepia(img: DynamicImage) -> DynamicImage {
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
pub fn duotone(img: DynamicImage, shadow: [u8; 3], highlight: [u8; 3]) -> DynamicImage {
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
pub fn monochrome(img: DynamicImage, color: [u8; 3]) -> DynamicImage {
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

/// Apply `f` to the rectangle (x, y, w, h) of `img` and write
/// the result into an RGBA copy in place of the rectangle's
/// pixels (see [`put_redacted`]); pixels outside the
/// rectangle are untouched. The caller guarantees the
/// rectangle is in bounds.
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
    put_redacted(&mut base, f(&region).to_rgba8(), x, y);
    DynamicImage::ImageRgba8(base)
}

/// Write the redacted pixels `done` over the rectangle of
/// `base` at (x, y), replacing every channel (issue #444).
/// Compositing them (`imageops::overlay`) would let the
/// original show through wherever the redacted alpha is
/// below 255: all of it at alpha 0, a third at alpha 128.
/// The face detector reads RGB only and a JPEG output writes
/// the stored RGB, so a face stored at alpha 0 would be
/// detected and then served intact.
///
/// A rectangle of uniform alpha keeps that alpha exactly: it
/// carries no detail to redact, and the blur's fixed-point
/// kernel sums to slightly under one from a sigma of about
/// 18, which would leave an opaque rectangle's alpha just
/// under 255 (253 at sigma 30). Like `overlay`, it clips to
/// `base`, so an empty or out-of-frame rectangle is a no-op.
fn put_redacted(base: &mut image::RgbaImage, mut done: image::RgbaImage, x: u32, y: u32) {
    let w = done.width().min(base.width().saturating_sub(x));
    let h = done.height().min(base.height().saturating_sub(y));
    if w == 0 || h == 0 {
        return;
    }
    let alpha = base.get_pixel(x, y)[3];
    let uniform = (y..y + h).all(|py| (x..x + w).all(|px| base.get_pixel(px, py)[3] == alpha));
    if uniform {
        for p in done.pixels_mut() {
            p[3] = alpha;
        }
    }
    image::imageops::replace(base, &done, x as i64, y as i64);
}

/// Gaussian-blur one rectangle of `img` (explicit region
/// redaction: plates, screens, documents). The sigma scales
/// with the region, min(w, h) / 8 clamped to [1, 30], so a
/// tall thin region blurs to its own scale.
pub fn blur_region(img: &DynamicImage, x: u32, y: u32, w: u32, h: u32) -> DynamicImage {
    let sigma = (w.min(h) as f32 / 8.0).clamp(1.0, 30.0);
    apply_to_region(img, x, y, w, h, |region| blur_fast(region, sigma))
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

/// Blur every detected face region in `img`. The blur sigma
/// scales with each face's size, so the redaction holds up at
/// any resolution. Pixels outside the face boxes are untouched.
pub fn blur_faces(img: &DynamicImage, faces: &[crate::ai::FaceBox]) -> DynamicImage {
    redact_faces(img, faces, |region, face| {
        let sigma = (face.w.min(face.h) as f32 / 6.0).max(1.0);
        blur_fast(region, sigma)
    })
}

/// The most mosaic cells a pixelated face spans along its
/// longer side.
const FACE_MOSAIC_CELLS: u32 = 8;

/// Pixelate every detected face region in `img` into a coarse
/// mosaic of averaged cells. The cell size comes from the face
/// box: its longer side spans at most `FACE_MOSAIC_CELLS`
/// cells, and a cell is never under 2 px, so a face only a few
/// pixels wide is averaged away rather than passed through
/// (issue #401: with a 1 px block a nearest-neighbor round
/// trip is an identity for faces under 20 px, and sampling
/// keeps a quarter of the pixels of slightly larger ones).
/// Pixels outside the face boxes are untouched.
pub fn pixelate_faces(img: &DynamicImage, faces: &[crate::ai::FaceBox]) -> DynamicImage {
    redact_faces(img, faces, |region, face| {
        let cell = face.w.max(face.h).div_ceil(FACE_MOSAIC_CELLS).max(2);
        mosaic(region, cell)
    })
}

/// Paint `img` as a grid of cells, each with the average of its
/// pixels (alpha included). The grid has `side / cell` cells
/// along each axis (at least one), spread evenly, so every cell
/// is at least `cell` px on a side, or the whole axis when the
/// image is shorter: unlike `pixelate`'s anchored blocks, no
/// thin strip at the right or bottom edge keeps its detail.
fn mosaic(img: &DynamicImage, cell: u32) -> DynamicImage {
    let mut rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let cell = cell.max(1);
    let (nx, ny) = ((w / cell).max(1), (h / cell).max(1));
    // u64: with small cells on a long side, i * side passes
    // u32::MAX.
    let edge = |i: u32, side: u32, n: u32| (i as u64 * side as u64 / n as u64) as u32;
    for j in 0..ny {
        let (y0, y1) = (edge(j, h, ny), edge(j + 1, h, ny));
        for i in 0..nx {
            let (x0, x1) = (edge(i, w, nx), edge(i + 1, w, nx));
            let mut sums = [0u64; 4];
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = rgba.get_pixel(x, y);
                    for c in 0..4 {
                        sums[c] += p[c] as u64;
                    }
                }
            }
            let n = (x1 - x0) as u64 * (y1 - y0) as u64;
            if n == 0 {
                continue;
            }
            let avg = image::Rgba(sums.map(|s| ((s + n / 2) / n) as u8));
            for y in y0..y1 {
                for x in x0..x1 {
                    rgba.put_pixel(x, y, avg);
                }
            }
        }
    }
    DynamicImage::ImageRgba8(rgba)
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

/// Clamp a detection's box to the image, the redaction
/// helpers' shared rule; `None` when nothing of the box
/// is in frame.
fn clamped_box(img: &DynamicImage, det: &crate::ai::Detection) -> Option<(u32, u32, u32, u32)> {
    let x = det.x.min(img.width());
    let y = det.y.min(img.height());
    let w = det.w.min(img.width() - x);
    let h = det.h.min(img.height() - y);
    if w == 0 || h == 0 {
        None
    } else {
        Some((x, y, w, h))
    }
}

/// Blend `over` into `base` inside one box, weighted per
/// pixel by a soft mask at the box's own resolution (issue
/// #175): strength 255 takes the pixel from `over`, 0
/// keeps `base`, and the values between antialias the
/// silhouette's edge. All four channels blend, so the
/// result matches what the unmasked region operation
/// would have produced inside the shape. The mask is
/// resized to the box when an export answered at another
/// resolution.
fn blend_masked(
    base: &DynamicImage,
    over: &DynamicImage,
    mask: &image::GrayImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> DynamicImage {
    let fitted;
    let mask = if (mask.width(), mask.height()) == (w, h) {
        mask
    } else {
        fitted = image::imageops::resize(mask, w, h, image::imageops::FilterType::Triangle);
        &fitted
    };
    let mut out = base.to_rgba8();
    let over = over.to_rgba8();
    for ly in 0..h {
        for lx in 0..w {
            let m = mask.get_pixel(lx, ly)[0] as u32;
            if m == 0 {
                continue;
            }
            let (gx, gy) = (x + lx, y + ly);
            let dst = out.get_pixel_mut(gx, gy);
            if m == 255 {
                *dst = *over.get_pixel(gx, gy);
            } else {
                let src = over.get_pixel(gx, gy);
                for c in 0..4 {
                    dst[c] = ((dst[c] as u32 * (255 - m) + src[c] as u32 * m) / 255) as u8;
                }
            }
        }
    }
    DynamicImage::ImageRgba8(out)
}

/// Blur every detection's silhouette in `img` (issue
/// #175): the mask-tight form of [`blur_detections`], used
/// when a segment provider is configured. Each mask is at
/// its box's resolution; the blur itself is the region
/// blur, so strength matches the rectangle form exactly
/// and only the shape changes. Pixels outside every
/// silhouette are untouched, including the parts of a
/// detection's rectangle the object does not fill.
pub fn blur_detections_masked(img: &DynamicImage, masked: &[MaskedDetection]) -> DynamicImage {
    redact_masked(img, masked, blur_region)
}

/// Pixelate every detection's silhouette in `img`: the
/// mask-tight form of [`pixelate_detections`].
pub fn pixelate_detections_masked(img: &DynamicImage, masked: &[MaskedDetection]) -> DynamicImage {
    redact_masked(img, masked, pixelate_region)
}

fn redact_masked(
    img: &DynamicImage,
    masked: &[MaskedDetection],
    redact: fn(&DynamicImage, u32, u32, u32, u32) -> DynamicImage,
) -> DynamicImage {
    let mut out = img.clone();
    for item in masked {
        let Some((x, y, w, h)) = clamped_box(&out, &item.detection) else {
            continue;
        };
        let done = redact(&out, x, y, w, h);
        out = blend_masked(&out, &done, &item.mask, x, y, w, h);
    }
    out
}

/// One detection with its segmentation mask, the unit the
/// mask-tight class operations work on (issue #175). The
/// mask is at the detection box's resolution.
pub struct MaskedDetection {
    pub detection: crate::ai::Detection,
    pub mask: image::GrayImage,
}

/// Recolor every detection's box in `img`: the object's
/// pixels take `color`, their alpha untouched (the paint
/// is a tint of the object, not a cutout). Without a
/// segment provider this is the rectangle form.
pub fn recolor_detections(
    img: &DynamicImage,
    detections: &[crate::ai::Detection],
    color: [u8; 3],
) -> DynamicImage {
    let mut out = img.clone();
    for det in detections {
        let Some((x, y, w, h)) = clamped_box(&out, det) else {
            continue;
        };
        out = paint_box(&out, x, y, w, h, None, color);
    }
    out
}

/// Recolor every detection's silhouette in `img` (issue
/// #175): the mask-tight form of [`recolor_detections`],
/// the silhouette's edge antialiased by the soft mask.
pub fn recolor_detections_masked(
    img: &DynamicImage,
    masked: &[MaskedDetection],
    color: [u8; 3],
) -> DynamicImage {
    let mut out = img.clone();
    for item in masked {
        let Some((x, y, w, h)) = clamped_box(&out, &item.detection) else {
            continue;
        };
        out = paint_box(&out, x, y, w, h, Some(&item.mask), color);
    }
    out
}

/// Paint `color` over one box of `img`, optionally through
/// a soft mask at the box's resolution; alpha is preserved.
fn paint_box(
    img: &DynamicImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    mask: Option<&image::GrayImage>,
    color: [u8; 3],
) -> DynamicImage {
    let fitted;
    let mask = match mask {
        Some(m) if (m.width(), m.height()) != (w, h) => {
            fitted = image::imageops::resize(m, w, h, image::imageops::FilterType::Triangle);
            Some(&fitted)
        }
        other => other,
    };
    let mut out = img.to_rgba8();
    for ly in 0..h {
        for lx in 0..w {
            let m = mask.map_or(255, |mk| mk.get_pixel(lx, ly)[0] as u32);
            if m == 0 {
                continue;
            }
            let px = out.get_pixel_mut(x + lx, y + ly);
            if m == 255 {
                px[0] = color[0];
                px[1] = color[1];
                px[2] = color[2];
            } else {
                for c in 0..3 {
                    px[c] = ((px[c] as u32 * (255 - m) + color[c] as u32 * m) / 255) as u8;
                }
            }
        }
    }
    DynamicImage::ImageRgba8(out)
}

/// Apply `redact` to each face region of `img` and write the
/// results back in place of the face pixels (see
/// [`put_redacted`]). Face boxes are clamped to the image;
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
        put_redacted(&mut base, redact(&region, face).to_rgba8(), x, y);
    }
    DynamicImage::ImageRgba8(base)
}

/// Crop away the uniform border: keep the bounding box of
/// pixels that differ from the top-left corner color by more
/// than `tolerance` in any channel (alpha included). A fully
/// uniform image is returned unchanged, never emptied.
fn trim(img: DynamicImage, tolerance: u8) -> DynamicImage {
    let found = trim_box(&img, tolerance);
    trim_to(img, found, [0; 4])
}

/// Per-side trim (issue #313): the tolerance pass runs
/// first, exactly as `trim` alone, then the named pixel
/// amounts come off the left, top, right, and bottom of
/// what remains. The amounts clamp so one pixel always
/// survives on each axis: the near side takes from the
/// dimension first and the far side takes from what is
/// left, so an oversized pair can never underflow or
/// produce an empty image.
fn trim_sides(img: DynamicImage, tolerance: u8, sides: [u32; 4]) -> DynamicImage {
    let found = trim_box(&img, tolerance);
    trim_to(img, found, sides)
}

/// The box [`trim`]'s tolerance pass keeps, as (left, top,
/// width, height): the bounding box of the pixels that differ
/// from the top-left corner color by more than `tolerance` in
/// any channel (alpha included), or `None` when no pixel
/// does. The engine unions these boxes across the frames of
/// an animation, so every frame takes one cut (issue #465).
pub(crate) fn trim_box(img: &DynamicImage, tolerance: u8) -> Option<(u32, u32, u32, u32)> {
    let converted;
    let rgba = match img.as_rgba8() {
        Some(rgba) => rgba,
        None => {
            converted = img.to_rgba8();
            &converted
        }
    };
    if rgba.width() == 0 || rgba.height() == 0 {
        return None;
    }
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
    found.then(|| (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

/// The window a trim keeps of a `w` x `h` frame, as (left,
/// top, width, height): the tolerance box (`None` is the
/// whole frame) less the `sides` pixels off its left, top,
/// right, and bottom, clamped the way [`trim_sides`]
/// describes.
pub(crate) fn trim_window(
    (w, h): (u32, u32),
    found: Option<(u32, u32, u32, u32)>,
    sides: [u32; 4],
) -> (u32, u32, u32, u32) {
    let (x, y, bw, bh) = found.unwrap_or((0, 0, w, h));
    let [left, top, right, bottom] = sides;
    let left = left.min(bw.saturating_sub(1));
    let right = right.min(bw.saturating_sub(1) - left);
    let top = top.min(bh.saturating_sub(1));
    let bottom = bottom.min(bh.saturating_sub(1) - top);
    (x + left, y + top, bw - left - right, bh - top - bottom)
}

/// Cut `img` to its [`trim_window`]. A cut through a found
/// box comes back as RGBA; with no box and no sides to shave
/// the image comes back untouched, pixel type and all.
pub(crate) fn trim_to(
    img: DynamicImage,
    found: Option<(u32, u32, u32, u32)>,
    sides: [u32; 4],
) -> DynamicImage {
    let (x, y, w, h) = trim_window((img.width(), img.height()), found, sides);
    if found.is_some() {
        DynamicImage::ImageRgba8(img.crop_imm(x, y, w, h).to_rgba8())
    } else if (x, y, w, h) == (0, 0, img.width(), img.height()) {
        img
    } else {
        img.crop_imm(x, y, w, h)
    }
}

/// Scale each pixel's distance from its luma by `amount`:
/// 0 collapses to gray, 1 is the identity.
fn saturate(mut out: image::RgbImage, amount: f32) -> image::RgbImage {
    let w = out.width() as usize;
    map_rgb_bands::<3>(out.as_mut(), w, &|r, g, b| {
        let luma = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
        [
            (luma + (r as f32 - luma) * amount).clamp(0.0, 255.0) as u8,
            (luma + (g as f32 - luma) * amount).clamp(0.0, 255.0) as u8,
            (luma + (b as f32 - luma) * amount).clamp(0.0, 255.0) as u8,
        ]
    });
    out
}

/// Vibrance (issue #317): like [`saturate`], but the
/// factor is per pixel: `1 + (amount / 100) * (1 - s)`,
/// where `s` is the pixel's HSV saturation in 0..1. A
/// muted pixel (s near 0) takes nearly the full percent
/// as its factor change; a fully saturated pixel does
/// not move at all, which is what protects vivid colors
/// and skin tones relative to the flat `saturation`
/// scale. Negative percents run the same weighting in
/// reverse, draining the muted colors first. The finish
/// matches `saturate`: scale the distance from luma,
/// clamp, truncate.
fn vibrance(mut out: image::RgbImage, amount: f32) -> image::RgbImage {
    let amount = percent_sanitized(amount);
    let w = out.width() as usize;
    map_rgb_bands::<3>(out.as_mut(), w, &|r, g, b| {
        let max = r.max(g).max(b) as f32;
        let min = r.min(g).min(b) as f32;
        let s = if max > 0.0 { (max - min) / max } else { 0.0 };
        let factor = 1.0 + (amount / 100.0) * (1.0 - s);
        let luma = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
        [
            (luma + (r as f32 - luma) * factor).clamp(0.0, 255.0) as u8,
            (luma + (g as f32 - luma) * factor).clamp(0.0, 255.0) as u8,
            (luma + (b as f32 - luma) * factor).clamp(0.0, 255.0) as u8,
        ]
    });
    out
}

/// The smallest percentile span [`auto_levels`] will
/// stretch: a channel whose 0.5th and 99.5th percentiles
/// sit closer than this is left unmapped, because
/// stretching a near-flat channel mostly amplifies
/// noise. The studio scene correction uses the same
/// floor for the same reason.
const AUTO_LEVELS_MIN_SPAN: u8 = 16;

/// The value at percentile `num / den` of a channel
/// histogram: the smallest value whose cumulative count
/// reaches that share of the total (the studio scene
/// correction's percentile, same definition).
fn channel_percentile(hist: &[u64; 256], total: u64, num: u64, den: u64) -> u8 {
    let target = (total * num).div_ceil(den).max(1);
    let mut acc = 0u64;
    for (v, n) in hist.iter().enumerate() {
        acc += n;
        if acc >= target {
            return v as u8;
        }
    }
    255
}

/// Auto-enhance (issue #317): a gentle per-channel
/// auto-levels. Each channel's 0.5th percentile maps to
/// 0 and its 99.5th percentile to 255, with the values
/// between stretched linearly: a compressed range opens
/// toward the ends, a color cast fades (each channel
/// stretches on its own percentiles), and an image that
/// already spans the range is left nearly alone. The
/// clipping percentiles are deliberately conservative,
/// which is what keeps the filter gentle next to
/// [`equalize`] and its full redistribution. A channel
/// whose percentile span is under
/// [`AUTO_LEVELS_MIN_SPAN`] is left unmapped.
fn auto_levels(img: image::RgbImage) -> image::RgbImage {
    let total = (img.width() * img.height()) as u64;
    // Banded histogram: per-band local counts merged by integer
    // addition, which is order-independent, so the merged counts
    // are exactly the sequential pass's counts. Small frames count
    // sequentially over the raw buffer.
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.as_raw();
    let row = w * 3;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4);
    let hists: [[u64; 256]; 3] = if threads < 2 || h < 256 || raw.len() < 3_000_000 {
        let mut hists = [[0u64; 256]; 3];
        for p in raw.as_chunks::<3>().0 {
            hists[0][p[0] as usize] += 1;
            hists[1][p[1] as usize] += 1;
            hists[2][p[2] as usize] += 1;
        }
        hists
    } else {
        let band_rows = h.div_ceil(threads);
        let locals: Vec<[[u64; 256]; 3]> = std::thread::scope(|s| {
            let handles: Vec<_> = raw
                .chunks(band_rows * row)
                .map(|band| {
                    s.spawn(move || {
                        let mut hists = [[0u64; 256]; 3];
                        for p in band.as_chunks::<3>().0 {
                            hists[0][p[0] as usize] += 1;
                            hists[1][p[1] as usize] += 1;
                            hists[2][p[2] as usize] += 1;
                        }
                        hists
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut hists = [[0u64; 256]; 3];
        for local in locals {
            for c in 0..3 {
                for v in 0..256 {
                    hists[c][v] += local[c][v];
                }
            }
        }
        hists
    };
    let mut maps: [[u8; 256]; 3] = [[0; 256]; 3];
    for c in 0..3 {
        let lo = channel_percentile(&hists[c], total, 1, 200);
        let hi = channel_percentile(&hists[c], total, 199, 200);
        let (lo, hi) = if hi.saturating_sub(lo) < AUTO_LEVELS_MIN_SPAN {
            (0usize, 255usize)
        } else {
            (usize::from(lo), usize::from(hi))
        };
        for (v, slot) in maps[c].iter_mut().enumerate() {
            *slot = if v <= lo {
                0
            } else if v >= hi {
                255
            } else {
                ((v - lo) * 255 + (hi - lo) / 2) / (hi - lo)
            } as u8;
        }
    }
    let mut out = img;
    let w = out.width() as usize;
    map_rgb_bands::<3>(out.as_mut(), w, &|r, g, b| {
        [
            maps[0][usize::from(r)],
            maps[1][usize::from(g)],
            maps[2][usize::from(b)],
        ]
    });
    out
}

/// Histogram equalization, independently per channel.
fn equalize(img: image::RgbImage) -> image::RgbImage {
    let total = (img.width() * img.height()) as u64;
    let mut maps: [[u8; 256]; 3] = [[0; 256]; 3];
    // One pass for all three channel histograms (the counts
    // are the same three passes would produce).
    let mut hists = [[0u64; 256]; 3];
    for p in img.pixels() {
        hists[0][p[0] as usize] += 1;
        hists[1][p[1] as usize] += 1;
        hists[2][p[2] as usize] += 1;
    }
    for c in 0..3 {
        let hist = hists[c];
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
    let mut out = img;
    for p in out.pixels_mut() {
        for c in 0..3 {
            p[c] = maps[c][p[c] as usize];
        }
    }
    out
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod identity_proofs;

// ===== Wave-3 identity proofs: the fast paths in this
// module against the image-crate operations they replace.
#[cfg(test)]
mod w3proofs;

// ----------------------------------------------------------------
// Wave-4 proofs. The tonal winners change only plumbing: exposure
// and gamma take the frame by value instead of cloning it into
// `map_rgb`, and `auto_levels` counts its histogram in row bands
// (integer counts merge exactly) and applies its maps through
// `map_rgb_bands`. Each is pinned here against a frozen copy of
// the pre-change implementation.
// ----------------------------------------------------------------

#[cfg(test)]
mod w4proofs;
