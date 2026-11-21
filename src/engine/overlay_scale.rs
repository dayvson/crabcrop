//! The scale behind `overlay(...,width)`: only the part of the
//! scaled overlay that lands on the image below is computed,
//! a row at a time, with the deadline checked before each row
//! and, within a row that reads many source samples, every
//! [`CHECK_EVERY`] samples.
//!
//! The overlay used to be scaled whole by the image crate's
//! Lanczos3 resize, one call that nothing could interrupt,
//! and only then cut to the part that lands. The pixel budget
//! bounded that call's size but not its time, so a scale that
//! lands in full could run well past the processing deadline,
//! and one that lands only a sliver still paid for the whole
//! overlay. This module computes the same pixels as that
//! call, for the part alone: the crate's two passes (a
//! vertical one into `f32`, then a horizontal one rounded
//! into the image's own samples), with the same weights,
//! summed in the same order, and the same rounding. Each
//! output pixel of either pass depends only on its own row's
//! or column's taps, so a row of the part needs the vertical
//! pass for that row over the source columns the part's
//! horizontal taps read, and nothing else. The identity
//! proofs (`identity_proofs.rs`) pin it to the crate's resize
//! in every layout.

use image::{DynamicImage, ImageBuffer, Pixel, RgbaImage};

/// The most source samples the vertical pass reads between
/// two deadline checks within one row. A row of an overlay
/// scaled far down reads many source rows, up to the whole
/// overlay for a scale to a pixel or two, so a check before
/// each row alone would leave that row's work unbounded by
/// the deadline. About a million samples is a millisecond or
/// two of work, and a check costs a clock read.
pub(super) const CHECK_EVERY: usize = 1 << 20;

#[cfg(test)]
thread_local! {
    /// How many parts [`scaled_part`] has scaled on this thread,
    /// so a test can see how often a render scales its overlay.
    pub(super) static SCALED_PARTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The size the image crate's
/// `resize(width, u32::MAX, Lanczos3)` gives an image of
/// `(w, h)`: its rule for fitting inside a box with the
/// aspect kept (`resize_dimensions` with `fill` off), which
/// the crate does not export.
pub(super) fn scaled_size((w, h): (u32, u32), width: u32) -> (u32, u32) {
    let height = u32::MAX;
    let wratio = f64::from(width) / f64::from(w);
    let hratio = f64::from(height) / f64::from(h);
    let ratio = f64::min(wratio, hratio);
    let nw = ((f64::from(w) * ratio).round() as u64).max(1);
    let nh = ((f64::from(h) * ratio).round() as u64).max(1);
    if nw > u64::from(u32::MAX) {
        let ratio = f64::from(u32::MAX) / f64::from(w);
        (u32::MAX, ((f64::from(h) * ratio).round() as u32).max(1))
    } else if nh > u64::from(u32::MAX) {
        let ratio = f64::from(u32::MAX) / f64::from(h);
        (((f64::from(w) * ratio).round() as u32).max(1), u32::MAX)
    } else {
        (nw as u32, nh as u32)
    }
}

/// The part `(x, y, width, height)` of `image` scaled to
/// `size` with the image crate's Lanczos3 resize, in RGBA:
/// the pixels of `image.resize_exact(size.0, size.1,
/// Lanczos3)` in that part, converted the way `into_rgba8`
/// converts them. `check` runs before each row of the part,
/// and within a row every [`CHECK_EVERY`] source samples its
/// vertical pass reads, where the caller checks the deadline;
/// its error stops the scale.
pub(super) fn scaled_part<E>(
    image: &DynamicImage,
    size: (u32, u32),
    part: crate::filters::Rect,
    check: impl FnMut() -> Result<(), E>,
) -> Result<RgbaImage, E> {
    #[cfg(test)]
    SCALED_PARTS.set(SCALED_PARTS.get() + 1);
    let (x, y, w, h) = part;
    if size == (image.width(), image.height()) {
        // The crate copies an image already at the size.
        return Ok(image.crop_imm(x, y, w, h).into_rgba8());
    }
    if image.width() == 0 || image.height() == 0 {
        // And fills an empty one's scale with zeros.
        let whole = image.resize_exact(size.0, size.1, image::imageops::FilterType::Lanczos3);
        return Ok(whole.crop_imm(x, y, w, h).into_rgba8());
    }
    Ok(match image {
        DynamicImage::ImageRgba8(src) => scale(src, size, part, check)?,
        DynamicImage::ImageRgb8(src) => {
            DynamicImage::ImageRgb8(scale(src, size, part, check)?).into_rgba8()
        }
        DynamicImage::ImageLuma8(src) => {
            DynamicImage::ImageLuma8(scale(src, size, part, check)?).into_rgba8()
        }
        DynamicImage::ImageLumaA8(src) => {
            DynamicImage::ImageLumaA8(scale(src, size, part, check)?).into_rgba8()
        }
        DynamicImage::ImageRgb16(src) => {
            DynamicImage::ImageRgb16(scale(src, size, part, check)?).into_rgba8()
        }
        DynamicImage::ImageRgba16(src) => {
            DynamicImage::ImageRgba16(scale(src, size, part, check)?).into_rgba8()
        }
        DynamicImage::ImageLuma16(src) => {
            DynamicImage::ImageLuma16(scale(src, size, part, check)?).into_rgba8()
        }
        DynamicImage::ImageLumaA16(src) => {
            DynamicImage::ImageLumaA16(scale(src, size, part, check)?).into_rgba8()
        }
        DynamicImage::ImageRgb32F(src) => {
            DynamicImage::ImageRgb32F(scale(src, size, part, check)?).into_rgba8()
        }
        DynamicImage::ImageRgba32F(src) => {
            DynamicImage::ImageRgba32F(scale(src, size, part, check)?).into_rgba8()
        }
        // A layout this module does not know: the crate's own
        // resize, whole, as before.
        other => {
            let whole = other.resize_exact(size.0, size.1, image::imageops::FilterType::Lanczos3);
            whole.crop_imm(x, y, w, h).into_rgba8()
        }
    })
}

/// A sample type the resize reads into `f32` and rounds back
/// into, as the crate does: an integer sample converts
/// exactly, and comes back rounded to the nearest within its
/// range; a float sample comes back clamped to 0 to 1.
trait Sample: Copy + 'static {
    fn read(self) -> f32;
    fn write(value: f32) -> Self;
}

/// The crate's clamp: a value below `min` is `min`, one above
/// `max` is `max`, anything else (NaN included) is itself.
fn clamp(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

impl Sample for u8 {
    fn read(self) -> f32 {
        f32::from(self)
    }
    fn write(value: f32) -> Self {
        clamp(value, 0.0, 255.0).round() as u8
    }
}

impl Sample for u16 {
    fn read(self) -> f32 {
        f32::from(self)
    }
    fn write(value: f32) -> Self {
        clamp(value, 0.0, 65535.0).round() as u16
    }
}

impl Sample for f32 {
    fn read(self) -> f32 {
        self
    }
    fn write(value: f32) -> Self {
        clamp(value, 0.0, 1.0)
    }
}

/// The crate's Lanczos kernel with a window of 3.
fn lanczos3(x: f32) -> f32 {
    fn sinc(t: f32) -> f32 {
        let a = t * std::f32::consts::PI;
        if t == 0.0 { 1.0 } else { a.sin() / a }
    }
    if x.abs() < 3.0 {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// The source samples one output row or column reads along an
/// axis of `len` scaled to `new_len`, from `first` on, and
/// their weights: the crate's taps, computed as it computes
/// them.
struct Taps {
    first: u32,
    weights: Vec<f32>,
}

impl Taps {
    fn of(out: u32, len: u32, new_len: u32) -> Self {
        let ratio = len as f32 / new_len as f32;
        let sratio = if ratio < 1.0 { 1.0 } else { ratio };
        let support = 3.0 * sratio;
        let input = (out as f32 + 0.5) * ratio;
        let first = ((input - support).floor() as i64).clamp(0, i64::from(len) - 1) as u32;
        let end =
            ((input + support).ceil() as i64).clamp(i64::from(first) + 1, i64::from(len)) as u32;
        let input = input - 0.5;
        let mut weights = Vec::with_capacity((end - first) as usize);
        let mut sum = 0.0;
        for i in first..end {
            let w = lanczos3((i as f32 - input) / sratio);
            weights.push(w);
            sum += w;
        }
        for w in &mut weights {
            *w /= sum;
        }
        Taps { first, weights }
    }

    fn end(&self) -> u32 {
        self.first + self.weights.len() as u32
    }
}

/// [`scaled_part`] for one layout, in its own samples.
fn scale<P, E>(
    src: &ImageBuffer<P, Vec<P::Subpixel>>,
    (new_w, new_h): (u32, u32),
    (x0, y0, w, h): crate::filters::Rect,
    mut check: impl FnMut() -> Result<(), E>,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, E>
where
    P: Pixel + 'static,
    P::Subpixel: Sample,
{
    let channels = usize::from(P::CHANNEL_COUNT);
    let (src_w, src_h) = src.dimensions();
    let samples = src.as_raw();
    let columns: Vec<Taps> = (x0..x0 + w).map(|x| Taps::of(x, src_w, new_w)).collect();
    // The source columns the part's columns read: their taps
    // only move right from one column to the next.
    let (left, right) = match (columns.first(), columns.last()) {
        (Some(first), Some(last)) => (first.first, last.end()),
        _ => (0, 0),
    };
    let span = (right - left) as usize * channels;
    let mut row = vec![0f32; span];
    let mut out = Vec::with_capacity(w as usize * h as usize * channels);
    for y in y0..y0 + h {
        check()?;
        // The vertical pass for this row, over those columns,
        // one source row at a time: each sum still takes its
        // taps in the crate's order, first to last, so the
        // sums are the crate's.
        let taps = Taps::of(y, src_h, new_h);
        row.fill(0.0);
        let mut unchecked = 0;
        for (i, &weight) in taps.weights.iter().enumerate() {
            if unchecked >= CHECK_EVERY {
                check()?;
                unchecked = 0;
            }
            let at = ((taps.first as usize + i) * src_w as usize + left as usize) * channels;
            for (sum, &sample) in row.iter_mut().zip(&samples[at..at + span]) {
                *sum += sample.read() * weight;
            }
            unchecked += span;
        }
        // The horizontal pass for the part's columns.
        for taps in &columns {
            let mut sums = [0f32; 4];
            for (i, &weight) in taps.weights.iter().enumerate() {
                let at = ((taps.first - left) as usize + i) * channels;
                for (sum, &value) in sums.iter_mut().zip(&row[at..at + channels]) {
                    *sum += value * weight;
                }
            }
            out.extend(sums[..channels].iter().map(|&sum| P::Subpixel::write(sum)));
        }
    }
    Ok(ImageBuffer::from_raw(w, h, out).expect("the part's samples, row by row"))
}
