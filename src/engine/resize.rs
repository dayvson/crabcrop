//! Sizing (issue #741): the resize, fit-in and pad steps, the
//! size arithmetic behind them (the `no_upscale` cap, DPR
//! scaling, proportional sides, the fit-in box), and the
//! Lanczos3 resampling they share: whole, windowed, lent,
//! cropped and reduced.
//!
//! engine.rs re-exports what the scaled JPEG decode, the
//! crate and the tests use.

use image::DynamicImage;

use super::limits::{check_output_bytes, check_output_size};
use super::{EngineError, Window, jpeg_scaled};
use crate::url::{Dimension, ImageSize, PadFill};

#[cfg(doc)]
use super::chain::fusion_plan;

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
pub(in crate::engine) enum SizeCap {
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
pub(in crate::engine) fn scale_px(n: u32, dpr: f32) -> u32 {
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
pub(in crate::engine) fn apply_size(
    img: DynamicImage,
    size: &ImageSize,
    cap: Option<SizeCap>,
    dpr: f32,
) -> Result<DynamicImage, EngineError> {
    apply_size_windowed(img, size, cap, dpr, SourceWindow::Whole)
}

/// What part of the frame in hand a resize reads.
#[derive(Clone, Copy)]
pub(in crate::engine) enum SourceWindow {
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
    /// A crop window within a reduced JPEG frame (issue #785):
    /// `window` is the (left, top, width, height) rectangle the
    /// crop decided on, in the full image's (upright)
    /// coordinates, and `reduced` says what part of the frame
    /// that full image fills. The size resolves and is checked
    /// against the window's dimensions, as a cut window's is,
    /// and the part of the frame the window maps to ([`reduced_crop_rect`])
    /// is resampled to the target (see [`resize_lanczos3_reduced`]).
    ReducedCrop {
        reduced: jpeg_scaled::Reduced,
        window: Window,
    },
}

/// [`apply_size`] for the crop+resize fusion (issue #287):
/// `img` is still the whole frame and `window` is the (left,
/// top, width, height) rectangle the chain's crop-rect named.
/// The size resolves against the window's dimensions, exactly
/// as it resolves against the materialized crop in the
/// two-step path, and the resample reads the window straight
/// from the frame.
pub(in crate::engine) fn apply_size_cropped(
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
pub(in crate::engine) fn proportional_side(side: u32, to: u32, from: u32) -> u32 {
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
pub(in crate::engine) fn resize_target(
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

pub(in crate::engine) fn apply_size_windowed(
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
        SourceWindow::Fused((_, _, w, h))
        | SourceWindow::Cut((_, _, w, h))
        | SourceWindow::ReducedCrop {
            window: (_, _, w, h),
            ..
        } => (w, h),
        SourceWindow::Reduced(reduced) => reduced.full,
        SourceWindow::Whole => (img.width(), img.height()),
    };
    let (w, h) = resize_target(size, dpr, (sw, sh), cap);
    check_output_size(w, h, (sw, sh))?;
    // Every branch below allocates its result in the frame's
    // layout, or as RGBA8, which the pixel limit bounds.
    check_output_bytes(w, h, img.color())?;
    let unchanged = w == sw && h == sh;
    let img = match window {
        // A reduced frame is not the size the target was
        // resolved against, so it is resampled whatever the
        // target is.
        SourceWindow::Reduced(reduced) => resize_lanczos3_reduced(img, reduced.window, w, h),
        // A crop window within a reduced frame is likewise
        // always resampled: the window maps to a part of the
        // frame smaller than the target it resolved (issue #785).
        SourceWindow::ReducedCrop { reduced, window } => {
            resize_lanczos3_reduced(img, reduced_crop_rect(&reduced, window), w, h)
        }
        // No resample runs: the two-step path returns
        // the materialized window itself, so the fused
        // path materializes it here.
        SourceWindow::Fused((x, y, ww, wh)) if unchanged => img.crop_imm(x, y, ww, wh),
        SourceWindow::Cut(window) if unchanged => crate::crop::cut(&img, window),
        SourceWindow::Whole if unchanged => img,
        SourceWindow::Fused(window) => resize_lanczos3_cropped_owned(img, window, w, h),
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
pub(in crate::engine) fn apply_size_lent(
    frame: &mut DynamicImage,
    size: &ImageSize,
    cap: Option<SizeCap>,
    dpr: f32,
) -> Result<Option<DynamicImage>, EngineError> {
    check_nonzero_size(size)?;
    let dims = (frame.width(), frame.height());
    let (w, h) = resize_target(size, dpr, dims, cap);
    check_output_size(w, h, dims)?;
    check_output_bytes(w, h, frame.color())?;
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
pub(in crate::engine) fn resize_lanczos3_owned(
    mut img: DynamicImage,
    w: u32,
    h: u32,
) -> DynamicImage {
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
/// transparent black. A frame with no such row is not
/// premultiplied at all (issue #807), since that would change
/// none of its bytes: the backend convolves it as it is and
/// divides the result. Any other layout, and a size that needs
/// no resample, take [`resize_lanczos3`].
pub(in crate::engine) fn resize_lanczos3_lent(
    img: &mut DynamicImage,
    w: u32,
    h: u32,
) -> DynamicImage {
    let (sw, sh) = (img.width(), img.height());
    let resamples = (w, h) != (sw, sh) && w > 0 && h > 0 && sw > 0 && sh > 0;
    let changing = match &*img {
        DynamicImage::ImageRgba8(rgba) if resamples => ChangingRows::of(rgba),
        _ => return resize_lanczos3(img, w, h),
    };
    let mut dst = DynamicImage::new_rgba8(w, h);
    let done = if changing.rows.is_empty() {
        convolve_and_divide(&*img, &mut dst)
    } else {
        resample_premultiplying(img, &mut dst)
    };
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
pub(in crate::engine) struct ChangingRows {
    /// The bytes in a row of the frame.
    stride: usize,
    /// The row indexes, in order.
    pub(in crate::engine) rows: Vec<usize>,
    /// Their bytes, one row after another.
    pub(in crate::engine) bytes: Vec<u8>,
}

impl ChangingRows {
    /// Copy out the rows of `frame` holding a pixel that is
    /// neither opaque nor black.
    pub(in crate::engine) fn of(frame: &image::RgbaImage) -> Self {
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
    #[cfg(test)]
    PREMULTIPLIED.set(PREMULTIPLIED.get() + u64::from(src.width()) * u64::from(src.height()));
    if convolve_and_divide(src, dst) {
        return true;
    }
    let _ = mul_div.divide_alpha_inplace(src);
    false
}

/// The last two of the backend's three alpha steps, on a frame
/// premultiplied already or one its premultiply would leave as
/// it is: its convolution with the alpha step off, and its own
/// divide of the result. `false` when either refuses.
fn convolve_and_divide<S, D>(src: &S, dst: &mut D) -> bool
where
    S: fast_image_resize::IntoImageView,
    D: fast_image_resize::IntoImageViewMut,
{
    convolve_lanczos3(src, dst, None)
        && fast_image_resize::MulDiv::new()
            .divide_alpha_inplace(dst)
            .is_ok()
}

/// The backend's Lanczos3 convolution alone, its alpha step
/// off: what it runs between its own premultiply and divide,
/// through its crop box `crop` when there is one. `false`
/// when it refuses.
fn convolve_lanczos3<S, D>(src: &S, dst: &mut D, crop: Option<Window>) -> bool
where
    S: fast_image_resize::IntoImageView,
    D: fast_image_resize::IntoImageViewMut,
{
    #[cfg(test)]
    if CONVOLVE_REFUSES.get() {
        return false;
    }
    let mut options = lanczos3_options().use_alpha(false);
    if let Some((x, y, w, h)) = crop {
        options = options.crop(f64::from(x), f64::from(y), f64::from(w), f64::from(h));
    }
    fast_image_resize::Resizer::new()
        .resize(src, dst, &options)
        .is_ok()
}

#[cfg(test)]
thread_local! {
    /// Makes [`convolve_lanczos3`] refuse on this thread, so
    /// a test can walk the path behind a backend refusal,
    /// which no input reaches.
    pub(in crate::engine) static CONVOLVE_REFUSES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// How many pixels the resizes here have premultiplied
    /// where they sit on this thread, so a test can see which
    /// premultiplies run and how much of a frame each covers.
    pub(in crate::engine) static PREMULTIPLIED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
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
pub(in crate::engine) fn resize_lanczos3_cut(
    img: DynamicImage,
    window: Window,
    w: u32,
    h: u32,
) -> DynamicImage {
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
                .is_ok_and(|view| convolve_lanczos3(&view, &mut resized, None));
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
pub(in crate::engine) fn view_holds_no_more(
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

/// [`resize_lanczos3_cropped`] of a frame the caller is done
/// with (issue #807), as [`resize_lanczos3_owned`] is of a
/// whole resize. Through a crop box, the backend premultiplies
/// a borrowed frame with alpha into a copy of the whole frame,
/// whatever the box, though its kernel reads only the box and
/// the filter's support around it (see [`kernel_span`]). An
/// RGBA8 frame has that part premultiplied where it sits, by
/// the backend's own routine, and the whole frame is resampled
/// through the same crop box with the alpha step off, then
/// divided by the backend's own routine. The kernel reads the
/// same premultiplied values at the same places, weighted by
/// the coefficients the backend computes for that box and that
/// frame, so the bytes are the same. Any other layout, and a
/// box that needs no resample, take [`resize_lanczos3_cropped`];
/// so does a backend refusal, which no input reaches, once the
/// part is divided back.
pub(in crate::engine) fn resize_lanczos3_cropped_owned(
    img: DynamicImage,
    (x, y, ww, wh): Window,
    w: u32,
    h: u32,
) -> DynamicImage {
    let DynamicImage::ImageRgba8(mut rgba) = img else {
        return resize_lanczos3_cropped(&img, x, y, ww, wh, w, h);
    };
    let (fw, fh) = rgba.dimensions();
    let inside = x as u64 + ww as u64 <= fw as u64 && y as u64 + wh as u64 <= fh as u64;
    if inside && ww > 0 && wh > 0 && w > 0 && h > 0 && (ww, wh) != (w, h) {
        let (left, right) = kernel_span(x, ww, w, fw);
        let (top, bottom) = kernel_span(y, wh, h, fh);
        let read = (left, top, right - left, bottom - top);
        let mut dst = image::RgbaImage::new(w, h);
        if resample_part_premultiplying(&mut rgba, read, (x, y, ww, wh), &mut dst) {
            return DynamicImage::ImageRgba8(dst);
        }
    }
    resize_lanczos3_cropped(&DynamicImage::ImageRgba8(rgba), x, y, ww, wh, w, h)
}

/// The samples of a frame axis `size` long that the backend's
/// Lanczos3 kernel reads when it resamples the `len` samples
/// from `start` to `out`, as (first, one past the last): the
/// box and the filter's radius past each of its ends, which is
/// three samples times the reduction (and three where the box
/// is enlarged), rounded up, with two samples more for the
/// floor and the ceiling the backend's bounds take, within the
/// frame.
pub(in crate::engine) fn kernel_span(start: u32, len: u32, out: u32, size: u32) -> (u32, u32) {
    let radius = 3.0 * (f64::from(len) / f64::from(out)).max(1.0);
    let reach = (radius.ceil() as u32).saturating_add(2);
    let end = start.saturating_add(len).saturating_add(reach);
    (start.saturating_sub(reach), end.min(size))
}

/// [`resample_premultiplying`] through the backend's crop box
/// `crop`, with only the part `read` of `frame` premultiplied,
/// where it sits: the part the kernel reads for that box (see
/// [`resize_lanczos3_cropped_owned`]). `true` when `dst` holds
/// the result. On `false` the part is straight alpha again,
/// as [`resample_premultiplying`] leaves its frame.
fn resample_part_premultiplying(
    frame: &mut image::RgbaImage,
    (px, py, pw, ph): Window,
    crop: Window,
    dst: &mut image::RgbaImage,
) -> bool {
    use fast_image_resize::images::CroppedImageMut;
    let mul_div = fast_image_resize::MulDiv::new();
    let premultiplied = CroppedImageMut::new(frame, px, py, pw, ph)
        .is_ok_and(|mut part| mul_div.multiply_alpha_inplace(&mut part).is_ok());
    if !premultiplied {
        return false;
    }
    #[cfg(test)]
    PREMULTIPLIED.set(PREMULTIPLIED.get() + u64::from(pw) * u64::from(ph));
    if convolve_lanczos3(&*frame, dst, Some(crop)) && mul_div.divide_alpha_inplace(dst).is_ok() {
        return true;
    }
    if let Ok(mut part) = CroppedImageMut::new(frame, px, py, pw, ph) {
        let _ = mul_div.divide_alpha_inplace(&mut part);
    }
    false
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
/// The part of a reduced frame a crop `window` maps to, as the
/// backend's fractional crop box (issue #785). The window is in
/// the full image's upright coordinates; `reduced.window` is the
/// part of the frame that full image fills, so the window is
/// mapped into it in proportion. A resize of this box then
/// resamples the crop straight from the reduced frame, at the
/// crop's own scale and in its place, exactly as
/// [`resize_lanczos3_reduced`] does for the whole image.
fn reduced_crop_rect(
    reduced: &jpeg_scaled::Reduced,
    (x, y, cw, ch): Window,
) -> (f64, f64, f64, f64) {
    let (full_w, full_h) = reduced.full;
    let (fx, fy, fw, fh) = reduced.window;
    let sx = fw / f64::from(full_w.max(1));
    let sy = fh / f64::from(full_h.max(1));
    (
        fx + f64::from(x) * sx,
        fy + f64::from(y) * sy,
        f64::from(cw) * sx,
        f64::from(ch) * sy,
    )
}

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
pub(in crate::engine) fn apply_pad(
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
    let canvas = ImageSize {
        width: Dimension::Pixels(target_w),
        height: Dimension::Pixels(target_h),
        flip_horizontal: false,
        flip_vertical: false,
    };
    // The canvas is RGBA8, which the check above holds to the
    // byte budget too, and the fit checks the image it fits in
    // the frame's layout. The blurred bars scale the frame to
    // cover the canvas and keep the canvas's part of the cover.
    // An RGB8 or RGBA8 frame has only that part computed (issue
    // #806). A frame in any other layout is scaled to the whole
    // cover first, in its layout, so that cover is checked
    // here, before the pad allocates anything (issue #784). Pad
    // never adapts: the canvas is the explicit box, so the fit
    // inside it uses the box as written.
    let dims = (img.width(), img.height());
    let fit = fit_in_target(&canvas, false, cap, 1.0, dims, img.color())?.unwrap_or(dims);
    if fill == PadFill::Blur
        && fit != (target_w, target_h)
        && !crate::ai::resample::fills_by_window(img.color())
    {
        let (cover_w, cover_h) =
            crate::ai::resample::resize_dimensions(dims.0, dims.1, target_w, target_h, true);
        check_output_bytes(cover_w, cover_h, img.color())?;
    }
    let backdrop = match fill {
        PadFill::Color(c) => Backdrop::Solid(c),
        PadFill::Directive => Backdrop::Solid(directive_fill.unwrap_or([0, 0, 0])),
        // Only the blurred fill reads the frame again after
        // the fit, so only it keeps a copy (issue #656).
        PadFill::Blur => Backdrop::Blurred(img.clone()),
    };
    let fitted = apply_fit_in(img, &canvas, false, cap, 1.0)?;
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
            // broad colors survive. The cover is the image
            // crate's Triangle `resize_to_fill`, of which an
            // RGB8 or RGBA8 frame has only the part the canvas
            // keeps computed, over rows (issue #806).
            let sigma = ((target_w.max(target_h) as f32) / 16.0).clamp(2.0, 80.0);
            let cover = crate::ai::resample::resize_to_fill(
                &source,
                target_w,
                target_h,
                crate::ai::resample::Kernel::Triangle,
            );
            crate::filters::blur_fast_within(&cover, sigma, deadline)?.into_rgba8()
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
pub(in crate::engine) fn place_on(base: &mut image::RgbaImage, top: DynamicImage, x: u32, y: u32) {
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
pub(in crate::engine) fn fit_in_box(
    size: &ImageSize,
    sw: u32,
    sh: u32,
    adaptive: bool,
) -> ImageSize {
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
pub(in crate::engine) fn apply_fit_in(
    img: DynamicImage,
    size: &ImageSize,
    adaptive: bool,
    cap: Option<(u32, u32)>,
    dpr: f32,
) -> Result<DynamicImage, EngineError> {
    let dims = (img.width(), img.height());
    match fit_in_target(size, adaptive, cap, dpr, dims, img.color())? {
        Some((w, h)) if (w, h) != dims => Ok(resize_lanczos3_owned(img, w, h)),
        _ => Ok(img),
    }
}

/// [`apply_fit_in`] of a frame lent for the resize (issue
/// #737), as [`apply_size_lent`] is of a resize: `None` when
/// the fit-in leaves the frame as it is.
pub(in crate::engine) fn apply_fit_in_lent(
    frame: &mut DynamicImage,
    size: &ImageSize,
    adaptive: bool,
    cap: Option<(u32, u32)>,
    dpr: f32,
) -> Result<Option<DynamicImage>, EngineError> {
    let dims = (frame.width(), frame.height());
    Ok(
        match fit_in_target(size, adaptive, cap, dpr, dims, frame.color())? {
            Some((w, h)) if (w, h) != dims => Some(resize_lanczos3_lent(frame, w, h)),
            _ => None,
        },
    )
}

/// The size a fit-in to `size` leaves a `sw` by `sh` image in
/// `layout` at, checked against the output bounds, the bytes
/// of a frame in that layout included; `None` when the box
/// bounds neither axis, so the image passes as it is.
fn fit_in_target(
    size: &ImageSize,
    adaptive: bool,
    cap: Option<(u32, u32)>,
    dpr: f32,
    (sw, sh): (u32, u32),
    layout: image::ColorType,
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
    check_output_bytes(w, h, layout)?;
    Ok(Some((w, h)))
}
