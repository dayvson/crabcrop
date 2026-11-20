//! Compositing (issue #741): the blend modes, the overlay and
//! watermark composites, where a placed image lands on its
//! base, and the flatten onto a fill color that JPEG output
//! and the encoders use. The scale behind
//! `overlay(...,width)` is `overlay_scale`.
//!
//! engine.rs re-exports what the still encode, the
//! encoders and the tests use.

use image::DynamicImage;

use super::animated::ScaledOverlays;
use super::limits::MAX_OUTPUT_PIXELS;
use super::{EngineError, ProcessContext, StillFormat, overlay_scale};

/// One channel of a blend mode: the CSS formulas on the
/// 0-255 scale (`b` is the base channel, `o` the overlay's).
pub(in crate::engine) fn blend_channel(mode: crate::url::BlendMode, b: f32, o: f32) -> f32 {
    use crate::url::BlendMode;
    match mode {
        BlendMode::Normal => o,
        BlendMode::Multiply => b * o / 255.0,
        BlendMode::Screen => 255.0 - (255.0 - b) * (255.0 - o) / 255.0,
        BlendMode::Overlay => {
            if b < 127.5 {
                2.0 * b * o / 255.0
            } else {
                255.0 - 2.0 * (255.0 - b) * (255.0 - o) / 255.0
            }
        }
    }
}

/// Composite `ov` over `base` with a blend mode, at (x, y)
/// with an opacity (0-100), optionally after scaling `ov`
/// to `width_pct` percent of the base width (aspect
/// preserved). Anchoring and clipping follow
/// `overlay_watermark`, as does the alpha convention: the
/// effective alpha is the overlay pixel's alpha times the
/// opacity, and the base alpha channel is preserved, so
/// `Normal` reproduces a watermark exactly. A scaled overlay
/// is scaled only where it lands, a row at a time, with the
/// deadline checked before each row and within a row that
/// reads many source samples (see `overlay_scale`): the same
/// pixels the whole scale gave there.
///
/// On a frame of an animated render, `kept` holds the URL the
/// overlay was fetched from and the scaled parts the render's
/// frames share (see [`ScaledOverlays`]): a part an earlier
/// frame scaled is drawn from there, with the same pixels, and
/// one this frame scales is kept there for the frames after
/// it. A still passes `None`.
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn blend_overlay_keeping(
    ctx: &ProcessContext<'_>,
    base: &DynamicImage,
    ov: &crate::filters::WatermarkImage,
    (x, y): (i32, i32),
    alpha: u8,
    mode: crate::url::BlendMode,
    width_pct: Option<u32>,
    kept: Option<(&str, &mut ScaledOverlays)>,
) -> Result<DynamicImage, EngineError> {
    // The deadline is checked before anything is copied or
    // scaled, as well as per row of the blend (issue #604).
    ctx.check_deadline()?;
    let target = match width_pct {
        Some(pct) => {
            let ov = ov.image();
            let target = ((base.width() as u64 * pct as u64) / 100).max(1);
            // The scaled height follows the overlay's aspect
            // with no bound of its own, so an extreme aspect
            // (or a very wide base) grows the intermediate
            // far past the pixel budget that guards the
            // other resize paths, up to aborting the process
            // on the allocation (issue #358). Charge an upper
            // bound of the scaled pair against the shared
            // budget before the resize allocates it.
            let scaled_h = (ov.height() as u64 * target / ov.width().max(1) as u64) + 1;
            if target * scaled_h > MAX_OUTPUT_PIXELS {
                return Err(EngineError::Unsupported(format!(
                    "scaled overlay {target}x{scaled_h} exceeds the {MAX_OUTPUT_PIXELS} pixel limit"
                )));
            }
            Some(target as u32)
        }
        None => None,
    };
    // An overlay that lands off the canvas is neither scaled
    // nor copied (issue #604), and leaves the base as it was,
    // in its own layout (issue #713).
    if !lands_on((base.width(), base.height()), (x, y)) {
        return Ok(base.clone());
    }
    // The scaled size is known before anything is scaled, so
    // only the part that lands is: a scale the pixel budget
    // admits can still take long whole, and a sliver of it on
    // the canvas no longer pays for the rest.
    let source_size = (ov.image().width(), ov.image().height());
    let top_size = match target {
        Some(target) => overlay_scale::scaled_size(source_size, target),
        None => source_size,
    };
    let Some(part) = landing((base.width(), base.height()), top_size, (x, y)) else {
        return Ok(base.clone());
    };
    let top = if top_size == source_size {
        // Scaled to its own size, the overlay is itself.
        ov.rgba_part(part.0)
    } else {
        let scale =
            || overlay_scale::scaled_part(ov.image(), top_size, part.0, || ctx.check_deadline());
        let origin = (part.0.0, part.0.1);
        match kept {
            Some((url, kept)) => {
                let room = u64::from(base.width()) * u64::from(base.height());
                let pixels = kept.part(url, top_size, part.0, room, scale)?;
                crate::filters::RgbaPart::shared(pixels, origin, top_size)
            }
            None => crate::filters::RgbaPart::copied(scale()?, origin, top_size),
        }
    };
    // The blend walks every overlay pixel; a large legal pair
    // still runs long, so the deadline is honored per row
    // (issue #358).
    composite(
        base,
        &top,
        (x, y),
        alpha,
        || ctx.check_deadline(),
        |b, o| blend_channel(mode, b, o),
    )
}

/// [`blend_overlay_keeping`] on a still, with nothing kept:
/// the form the tests call.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn blend_overlay(
    ctx: &ProcessContext<'_>,
    base: &DynamicImage,
    ov: &crate::filters::WatermarkImage,
    x: i32,
    y: i32,
    alpha: u8,
    mode: crate::url::BlendMode,
    width_pct: Option<u32>,
) -> Result<DynamicImage, EngineError> {
    blend_overlay_keeping(ctx, base, ov, (x, y), alpha, mode, width_pct, None)
}

/// Overlay `wm` onto `base` at (x, y) with `alpha` percent
/// opacity, honoring the watermark's own alpha channel.
/// Negative coordinates anchor from the right/bottom edges
/// (x = -1 puts the watermark's right edge one pixel from the
/// base's right edge). Parts falling outside the base are
/// clipped, never a panic. The deadline is checked before
/// anything is copied and per row of the composite, as the
/// overlay checks it, and only the part of the watermark that
/// lands on the base is read or copied (issue #604). A
/// watermark that lands nowhere leaves the base as it was, in
/// its own layout (issue #713).
pub(in crate::engine) fn overlay_watermark(
    ctx: &ProcessContext<'_>,
    base: &DynamicImage,
    wm: &crate::filters::WatermarkImage,
    x: i32,
    y: i32,
    alpha: u8,
) -> Result<DynamicImage, EngineError> {
    ctx.check_deadline()?;
    let top_size = (wm.image().width(), wm.image().height());
    let Some(part) = landing((base.width(), base.height()), top_size, (x, y)) else {
        return Ok(base.clone());
    };
    composite(
        base,
        &wm.rgba_part(part.0),
        (x, y),
        alpha,
        || ctx.check_deadline(),
        |_, o| o,
    )
}

/// Whether an image placed at (x, y) on a base of
/// `(width, height)` covers any base pixel, by the anchoring
/// rule [`composite`] follows. The answer does not depend on
/// the placed image's size, as long as it has a pixel: a
/// coordinate of 0 or more must fall inside the base, and a
/// negative one, which anchors the far edge that many pixels
/// in from the base's far edge, must be less than the base's
/// extent. So it can be asked before an overlay is scaled.
pub(in crate::engine) fn lands_on((width, height): (u32, u32), (x, y): (i32, i32)) -> bool {
    x.unsigned_abs() < width && y.unsigned_abs() < height
}

/// The part of a top image of size `top` placed at (x, y) on
/// a base of size `base` that lands on the base, and the base
/// pixel its top-left corner lands on (issue #604). A
/// negative coordinate anchors the top's right or bottom edge
/// that many pixels in from the base's. `None` when no pixel
/// lands.
pub(in crate::engine) fn landing(
    (bw, bh): (u32, u32),
    (tw, th): (u32, u32),
    (x, y): (i32, i32),
) -> Option<(crate::filters::Rect, (u32, u32))> {
    let (bw, bh, tw, th) = (bw as i64, bh as i64, tw as i64, th as i64);
    let left = if x >= 0 { x as i64 } else { bw - tw + x as i64 };
    let upper = if y >= 0 { y as i64 } else { bh - th + y as i64 };
    let columns = (-left).max(0)..(bw - left).min(tw);
    let rows = (-upper).max(0)..(bh - upper).min(th);
    if columns.is_empty() || rows.is_empty() {
        return None;
    }
    let part = (
        columns.start as u32,
        rows.start as u32,
        (columns.end - columns.start) as u32,
        (rows.end - rows.start) as u32,
    );
    Some((
        part,
        ((left + columns.start) as u32, (upper + rows.start) as u32),
    ))
}

/// The anchor, clip, and alpha loop the overlay and the
/// watermark share (issue #480). `top` sits on `base` at
/// `(x, y)`, a negative coordinate anchoring from the right or
/// bottom edge, and whatever falls outside `base` is clipped.
/// Each covered pixel's effective alpha is the top pixel's own
/// alpha times `alpha` percent, and each color channel moves
/// that far toward `blend(base, top)`, the per-pixel blend of
/// the base channel and the top one; the base alpha channel is
/// kept. Only the part of `top` that lands on `base` is read
/// (issue #604, see [`landing`]), and `each_row` runs before
/// each of its rows, where the callers check the deadline.
/// The blend works in 8-bit RGBA, so a composite that lands
/// is RGBA8; one that lands nowhere is the base as it was, in
/// its own layout (issue #713).
pub(in crate::engine) fn composite<E, T>(
    base: &DynamicImage,
    top: &T,
    (x, y): (i32, i32),
    alpha: u8,
    mut each_row: impl FnMut() -> Result<(), E>,
    blend: impl Fn(f32, f32) -> f32,
) -> Result<DynamicImage, E>
where
    T: image::GenericImageView<Pixel = image::Rgba<u8>>,
{
    let Some(((tx0, ty0, w, h), (bx0, by0))) =
        landing((base.width(), base.height()), top.dimensions(), (x, y))
    else {
        return Ok(base.clone());
    };
    let mut out = base.to_rgba8();
    let opacity = alpha as f32 / 100.0;
    for row in 0..h {
        each_row()?;
        for column in 0..w {
            let top_px = top.get_pixel(tx0 + column, ty0 + row);
            let a = opacity * (top_px[3] as f32 / 255.0);
            if a <= 0.0 {
                continue;
            }
            let bot = out.get_pixel_mut(bx0 + column, by0 + row);
            for c in 0..3 {
                let blended = blend(bot[c] as f32, top_px[c] as f32);
                bot[c] = (bot[c] as f32 * (1.0 - a) + blended * a).round() as u8;
            }
        }
    }
    Ok(DynamicImage::ImageRgba8(out))
}

/// The image a JPEG with a `fill` encodes: `img` flattened over
/// the fill, as the encoder flattens it, when `img` has alpha;
/// `None` for every other format, a JPEG without a fill, and
/// an image with no alpha to flatten, which encode as they
/// are.
pub(in crate::engine) fn flattened_for(
    img: &DynamicImage,
    format: StillFormat,
    fill: Option<[u8; 3]>,
) -> Option<DynamicImage> {
    match (format, fill) {
        (StillFormat::Jpeg, Some(bg)) if img.color().has_alpha() => {
            Some(DynamicImage::ImageRgb8(flatten(img, bg)))
        }
        _ => None,
    }
}

/// Composite an image over a solid background color, dropping the
/// alpha channel.
pub(in crate::engine) fn flatten(img: &DynamicImage, bg: [u8; 3]) -> image::RgbImage {
    let mut base = image::RgbaImage::from_pixel(
        img.width(),
        img.height(),
        image::Rgba([bg[0], bg[1], bg[2], 255]),
    );
    image::imageops::overlay(&mut base, &img.to_rgba8(), 0, 0);
    DynamicImage::ImageRgba8(base).to_rgb8()
}
