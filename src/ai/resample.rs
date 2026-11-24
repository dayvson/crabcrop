//! `image::imageops::resize` for 8-bit buffers, byte for
//! byte, over rows (issue #673).
//!
//! A few frames are still resampled with the image crate's
//! own resize: the 224-crop of the CLIP and ViT preprocessing,
//! the studio composite's subject (issue #674), the blurred
//! backdrop of a pad (issue #806) and the previews a meta
//! response's placeholder is computed from (issue #811). What
//! they produce feeds a model or is the response, so the
//! kernel cannot change without changing bytes. The crate
//! reads one pixel at a time through `get_pixel`, column by
//! column, into a four-channel float image. This module does
//! the same arithmetic on rows: the same weights from the
//! same expressions, the same accumulation order for every
//! sample, the same clamp and rounding. The tests compare it
//! with the crate.
//!
//! It can compute a window of the output alone, for a
//! caller that crops right after it resizes.

use image::DynamicImage;

/// The sampling filters the callers use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kernel {
    Triangle,
    CatmullRom,
}

impl Kernel {
    /// The crate's filter of the same name.
    fn filter(self) -> image::imageops::FilterType {
        match self {
            Kernel::Triangle => image::imageops::FilterType::Triangle,
            Kernel::CatmullRom => image::imageops::FilterType::CatmullRom,
        }
    }

    /// The filter's support, as the crate's `resize` sets it.
    fn support(self) -> f32 {
        match self {
            Kernel::Triangle => 1.0,
            Kernel::CatmullRom => 2.0,
        }
    }

    /// The crate's kernel function, verbatim.
    fn weight(self, x: f32) -> f32 {
        match self {
            Kernel::Triangle => {
                if x.abs() < 1.0 {
                    1.0 - x.abs()
                } else {
                    0.0
                }
            }
            Kernel::CatmullRom => {
                let (b, c) = (0.0f32, 0.5f32);
                let a = x.abs();
                let k = if a < 1.0 {
                    (12.0 - 9.0 * b - 6.0 * c) * a.powi(3)
                        + (-18.0 + 12.0 * b + 6.0 * c) * a.powi(2)
                        + (6.0 - 2.0 * b)
                } else if a < 2.0 {
                    (-b - 6.0 * c) * a.powi(3)
                        + (6.0 * b + 30.0 * c) * a.powi(2)
                        + (-12.0 * b - 48.0 * c) * a
                        + (8.0 * b + 24.0 * c)
                } else {
                    0.0
                };
                k / 6.0
            }
        }
    }
}

/// The largest weight this module computes with. The
/// normalized weights of these kernels stay under 3; with
/// every weight at most this, no sum of 8-bit samples can
/// overflow or come out NaN, at any size, so the crate's own
/// conversion cannot fail and the two always agree. A set of
/// weights past it is left to the crate.
const WEIGHT_LIMIT: f32 = 8.0;

/// One axis of a resize: for each output sample, the first
/// source index it reads and its normalized weights, which
/// apply to that index and the ones after it.
struct Taps {
    /// (first source index, offset of the weights, count).
    runs: Vec<(usize, usize, usize)>,
    weights: Vec<f32>,
}

impl Taps {
    fn run(&self, i: usize) -> (usize, &[f32]) {
        let (first, at, count) = self.runs[i];
        (first, &self.weights[at..at + count])
    }

    /// The source range the runs read, first to one past last.
    fn source_range(&self) -> (usize, usize) {
        let from = self.runs.iter().map(|r| r.0).min().unwrap_or(0);
        let to = self.runs.iter().map(|r| r.0 + r.2).max().unwrap_or(0);
        (from, to)
    }
}

/// The taps of output samples `from..to` along an axis that
/// goes from `src_len` to `dst_len` samples, computed as the
/// crate's `vertical_sample` and `horizontal_sample` compute
/// them. `None` when a weight is not finite or passes
/// [`WEIGHT_LIMIT`].
fn axis_taps(src_len: u32, dst_len: u32, kernel: Kernel, from: u32, to: u32) -> Option<Taps> {
    let ratio = src_len as f32 / dst_len as f32;
    let sratio = if ratio < 1.0 { 1.0 } else { ratio };
    let src_support = kernel.support() * sratio;
    let mut taps = Taps {
        runs: Vec::with_capacity((to - from) as usize),
        weights: Vec::new(),
    };
    for out in from..to {
        let input = (out as f32 + 0.5) * ratio;
        let left = (input - src_support).floor() as i64;
        let left = left.clamp(0, i64::from(src_len) - 1) as u32;
        let right = (input + src_support).ceil() as i64;
        let right = right.clamp(i64::from(left) + 1, i64::from(src_len)) as u32;
        let input = input - 0.5;
        let at = taps.weights.len();
        let mut sum = 0.0;
        for i in left..right {
            let w = kernel.weight((i as f32 - input) / sratio);
            taps.weights.push(w);
            sum += w;
        }
        for w in &mut taps.weights[at..] {
            *w /= sum;
            if !w.is_finite() || w.abs() > WEIGHT_LIMIT {
                return None;
            }
        }
        taps.runs.push((left as usize, at, (right - left) as usize));
    }
    Some(taps)
}

/// The crate's conversion of a sum to a sample: clamp to
/// 0..=255, then round half away from zero. The clamped
/// value's integer part is its truncating cast and the
/// difference to it is exact, so the rounding is one
/// comparison and no call to `round`.
#[inline]
fn to_u8(t: f32) -> u8 {
    let x = t.clamp(0.0, 255.0);
    let whole = x as u8;
    whole + u8::from(x - f32::from(whole) >= 0.5)
}

/// The bytes `image::imageops::resize(src, dw, dh, kernel)`
/// holds in the rectangle `(x, y, w, h)` of its output, for
/// an 8-bit buffer of `CH` interleaved channels (at most 4).
/// Only the source rows and columns that rectangle reads are
/// touched. `None` when the buffer is shorter than its size,
/// the rectangle is not inside the output, or the weights are
/// not ones this module computes with: the caller then takes
/// the crate's own resize.
pub(crate) fn resize_window<const CH: usize>(
    src: &[u8],
    (sw, sh): (u32, u32),
    (dw, dh): (u32, u32),
    kernel: Kernel,
    (wx, wy, ww, wh): (u32, u32, u32, u32),
) -> Option<Vec<u8>> {
    if CH == 0 || CH > 4 {
        return None;
    }
    let (sw_us, out_w, out_h) = (sw as usize, ww as usize, wh as usize);
    if src.len() < sw_us.checked_mul(sh as usize)?.checked_mul(CH)? {
        return None;
    }
    if wx.checked_add(ww)? > dw || wy.checked_add(wh)? > dh {
        return None;
    }
    let out_len = out_w.checked_mul(out_h)?.checked_mul(CH)?;
    // The crate answers an empty source with a zeroed buffer
    // and an unchanged size with a copy.
    if sw == 0 || sh == 0 || out_len == 0 {
        return Some(vec![0u8; out_len]);
    }
    if (dw, dh) == (sw, sh) {
        let mut out = Vec::with_capacity(out_len);
        for y in wy as usize..(wy + wh) as usize {
            let at = (y * sw_us + wx as usize) * CH;
            out.extend_from_slice(&src[at..at + out_w * CH]);
        }
        return Some(out);
    }
    let rows = axis_taps(sh, dh, kernel, wy, wy + wh)?;
    let cols = axis_taps(sw, dw, kernel, wx, wx + ww)?;
    let (col_from, col_to) = cols.source_range();
    let span = (col_to - col_from) * CH;
    let mut out = vec![0u8; out_len];
    // One row of the crate's intermediate image at a time,
    // over the columns in use: the vertical pass for this
    // output row, then the horizontal pass from it. The row
    // has room for the four-wide read of its last sample.
    let mut acc = vec![0f32; span + 4 - CH];
    for (y, out_row) in out.chunks_exact_mut(out_w * CH).enumerate() {
        let (first_row, weights) = rows.run(y);
        acc.fill(0.0);
        for (i, &w) in weights.iter().enumerate() {
            let at = ((first_row + i) * sw_us + col_from) * CH;
            for (a, &s) in acc.iter_mut().zip(&src[at..at + span]) {
                *a += f32::from(s) * w;
            }
        }
        for (x, px) in out_row.as_chunks_mut::<CH>().0.iter_mut().enumerate() {
            let (first_col, weights) = cols.run(x);
            let at = (first_col - col_from) * CH;
            // Four lanes at once, whatever `CH` is: a lane past
            // the pixel's own channels reads its neighbour and
            // is dropped. Each lane adds its taps in order.
            let run = &acc[at..at + (weights.len() - 1) * CH + 4];
            let mut t = [0f32; 4];
            for (tap, &w) in run.windows(4).step_by(CH).zip(weights) {
                t[0] += tap[0] * w;
                t[1] += tap[1] * w;
                t[2] += tap[2] * w;
                t[3] += tap[3] * w;
            }
            for (sample, &sum) in px.iter_mut().zip(&t) {
                *sample = to_u8(sum);
            }
        }
    }
    Some(out)
}

/// `image::imageops::resize(src, dw, dh, kernel)` for an
/// 8-bit buffer of `CH` interleaved channels: every byte of
/// the output. `None` as [`resize_window`] answers it.
pub(crate) fn resize<const CH: usize>(
    src: &[u8],
    src_dims: (u32, u32),
    (dw, dh): (u32, u32),
    kernel: Kernel,
) -> Option<Vec<u8>> {
    resize_window::<CH>(src, src_dims, (dw, dh), kernel, (0, 0, dw, dh))
}

/// The image crate's `resize_dimensions`, which it does not
/// export, verbatim: the size `DynamicImage::resize` (`fill`
/// false) and `DynamicImage::resize_to_fill` (`fill` true)
/// scale a `width` by `height` image to, its aspect kept, to
/// fit inside a box of `nwidth` by `nheight` or to cover it.
pub(crate) fn resize_dimensions(
    width: u32,
    height: u32,
    nwidth: u32,
    nheight: u32,
    fill: bool,
) -> (u32, u32) {
    let wratio = f64::from(nwidth) / f64::from(width);
    let hratio = f64::from(nheight) / f64::from(height);
    let ratio = if fill {
        f64::max(wratio, hratio)
    } else {
        f64::min(wratio, hratio)
    };
    let nw = ((f64::from(width) * ratio).round() as u64).max(1);
    let nh = ((f64::from(height) * ratio).round() as u64).max(1);
    if nw > u64::from(u32::MAX) {
        let ratio = f64::from(u32::MAX) / f64::from(width);
        (
            u32::MAX,
            ((f64::from(height) * ratio).round() as u32).max(1),
        )
    } else if nh > u64::from(u32::MAX) {
        let ratio = f64::from(u32::MAX) / f64::from(height);
        (((f64::from(width) * ratio).round() as u32).max(1), u32::MAX)
    } else {
        (nw as u32, nh as u32)
    }
}

/// [`resize_window`] of an RGB8 or RGBA8 image, in its own
/// layout: the window `(x, y, w, h)` of `img.resize_exact(dw,
/// dh, kernel)`. `None` for any other layout, and where
/// [`resize_window`] answers `None`.
fn image_window(
    img: &DynamicImage,
    (dw, dh): (u32, u32),
    kernel: Kernel,
    window: (u32, u32, u32, u32),
) -> Option<DynamicImage> {
    let src_dims = (img.width(), img.height());
    let (_, _, w, h) = window;
    match img {
        DynamicImage::ImageRgb8(rgb) => {
            resize_window::<3>(rgb.as_raw(), src_dims, (dw, dh), kernel, window)
                .and_then(|raw| image::RgbImage::from_raw(w, h, raw))
                .map(DynamicImage::ImageRgb8)
        }
        DynamicImage::ImageRgba8(rgba) => {
            resize_window::<4>(rgba.as_raw(), src_dims, (dw, dh), kernel, window)
                .and_then(|raw| image::RgbaImage::from_raw(w, h, raw))
                .map(DynamicImage::ImageRgba8)
        }
        _ => None,
    }
}

/// `img.resize_exact(nwidth, nheight, kernel)` (issue #811),
/// computed over rows for an RGB8 or RGBA8 image; any other
/// layout takes the crate's own call.
pub(crate) fn resize_exact(
    img: &DynamicImage,
    nwidth: u32,
    nheight: u32,
    kernel: Kernel,
) -> DynamicImage {
    image_window(img, (nwidth, nheight), kernel, (0, 0, nwidth, nheight))
        .unwrap_or_else(|| img.resize_exact(nwidth, nheight, kernel.filter()))
}

/// `img.resize(nwidth, nheight, kernel)` (issue #811): the
/// image scaled to fit inside the box, at the crate's size
/// (see [`resize_dimensions`]), computed over rows for an RGB8
/// or RGBA8 image; any other layout takes the crate's own
/// call.
pub(crate) fn resize_to_fit(
    img: &DynamicImage,
    nwidth: u32,
    nheight: u32,
    kernel: Kernel,
) -> DynamicImage {
    let (w, h) = resize_dimensions(img.width(), img.height(), nwidth, nheight, false);
    image_window(img, (w, h), kernel, (0, 0, w, h))
        .unwrap_or_else(|| img.resize(nwidth, nheight, kernel.filter()))
}

/// `img.resize_to_fill(nwidth, nheight, kernel)` (issue #806):
/// the image scaled to cover the box, at the crate's size (see
/// [`resize_dimensions`]), and the box cut from the middle of
/// the axis that overflows it, where the crate cuts it. The
/// crate scales the whole image and then cuts; an RGB8 or
/// RGBA8 image has only the window that is kept computed, over
/// rows. Any other layout takes the crate's own call.
pub(crate) fn resize_to_fill(
    img: &DynamicImage,
    nwidth: u32,
    nheight: u32,
    kernel: Kernel,
) -> DynamicImage {
    let (w, h) = resize_dimensions(img.width(), img.height(), nwidth, nheight, true);
    let window = if u64::from(nwidth) * u64::from(h) > u64::from(w) * u64::from(nheight) {
        h.checked_sub(nheight)
            .map(|spare| (0, spare / 2, nwidth, nheight))
    } else {
        w.checked_sub(nwidth)
            .map(|spare| (spare / 2, 0, nwidth, nheight))
    };
    window
        .and_then(|window| image_window(img, (w, h), kernel, window))
        .unwrap_or_else(|| img.resize_to_fill(nwidth, nheight, kernel.filter()))
}

/// Whether [`resize_to_fill`] computes only the box it keeps
/// of an image in `layout`, rather than the crate's whole
/// cover: RGB8 and RGBA8, the layouts `image_window`
/// computes. It leaves one of those to the crate only for a
/// cover side past `u32`, which no image and box within the
/// engine's 16384 px a side make, or for weights past
/// `WEIGHT_LIMIT`, which these kernels never make. A blurred
/// pad holds the whole cover of every other layout to the
/// engine's byte budget before it builds it (issue #784).
pub(crate) fn fills_by_window(layout: image::ColorType) -> bool {
    matches!(layout, image::ColorType::Rgb8 | image::ColorType::Rgba8)
}

#[cfg(test)]
mod tests;
