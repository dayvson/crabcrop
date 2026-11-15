//! `image::imageops::resize` for 8-bit buffers, byte for
//! byte, over rows (issue #673).
//!
//! A few frames are still resampled with the image crate's
//! own resize: the 224-crop of the CLIP and ViT preprocessing
//! and the studio composite's subject (issue #674). What they
//! produce feeds a model or is the response, so the kernel
//! cannot change without changing bytes. The crate reads one
//! pixel at a time through `get_pixel`, column by column,
//! into a four-channel float image. This module does the
//! same arithmetic on rows: the same weights from the same
//! expressions, the same accumulation order for every
//! sample, the same clamp and rounding. The tests compare it
//! with the crate.
//!
//! It can compute a window of the output alone, for a
//! caller that crops right after it resizes.

/// The sampling filters the callers use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kernel {
    Triangle,
    CatmullRom,
}

impl Kernel {
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

#[cfg(test)]
mod tests;
