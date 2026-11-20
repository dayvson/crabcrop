//! Focal points (issue #741): the pending focal form a
//! `focal:` or `focalpoint:` segment names, mapped through
//! each later step until a crop consumes it, which filters
//! keep it, and the `debug:focal` capture of what a crop
//! used.
//!
//! engine.rs re-exports what the tests use.

use image::DynamicImage;

use super::Window;
use super::detect::filter_preserves_positions;
use crate::url::{Filter, PlainFilter};

/// Whether a filter leaves every pixel where it was, so a
/// pending focal form stays put (issue #465): the
/// position-preserving filters of #373, plus the ones that
/// paint over the frame in place (watermark, overlay, text,
/// border, background removal). Those add content a detector
/// could find, which is why #373 re-detects after them, but
/// they move nothing. Anything else drops the form: a new
/// filter errs toward the crop's own alignment, never toward
/// a stale position.
pub(in crate::engine) fn filter_keeps_focal(filter: &Filter) -> bool {
    filter_preserves_positions(filter)
        || matches!(
            filter,
            Filter::Watermark { .. }
                | Filter::Overlay { .. }
                | Filter::Text { .. }
                | Filter::Plain(PlainFilter::Border(..))
                | Filter::RemoveBg
        )
}

/// A pending focal form (issues #313, #465): the rectangle
/// a `focal:` segment named or the point a `focalpoint:`
/// segment named, carried in the coordinates of the image in
/// hand. The segment names it against the image at its own
/// position in the chain; each later step that moves pixels
/// maps it along (a resize or fit-in scales it, a resize flip
/// mirrors it, a crop-rect or trim cuts it, a pad places
/// it), and a filter that moves pixels any other way drops
/// it, so the consuming crop never reads coordinates of an
/// image that is gone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::engine) struct PendingFocal {
    /// `None` for a rectangle, the zoom for a point.
    zoom: Option<f32>,
    /// (left, top, right, bottom) in the coordinates of a
    /// `basis.0` x `basis.1` image. A point is a rectangle
    /// of no size.
    edges: [f64; 4],
    basis: (u32, u32),
}

/// A pending focal form resolved against the image a crop
/// cuts: a rectangle in its pixels, or a point as fractions
/// of it with the zoom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::engine) enum Focal {
    Rect((u32, u32, u32, u32)),
    Point((f32, f32, f32)),
}

impl PendingFocal {
    pub(in crate::engine) fn rect(
        (left, top, right, bottom): (u32, u32, u32, u32),
        basis: (u32, u32),
    ) -> Self {
        PendingFocal {
            zoom: None,
            edges: [left as f64, top as f64, right as f64, bottom as f64],
            basis,
        }
    }

    pub(in crate::engine) fn point((x, y, zoom): (f32, f32, f32), basis: (u32, u32)) -> Self {
        let (px, py) = (x as f64 * basis.0 as f64, y as f64 * basis.1 as f64);
        PendingFocal {
            zoom: Some(zoom),
            edges: [px, py, px, py],
            basis,
        }
    }

    /// Through a resize of the whole image to `to`, mirrored
    /// afterwards on the flipped axes.
    pub(in crate::engine) fn scaled(self, to: (u32, u32), flip_h: bool, flip_v: bool) -> Self {
        let ratio = |to: u32, from: u32| {
            if from == 0 {
                0.0
            } else {
                to as f64 / from as f64
            }
        };
        let (sx, sy) = (ratio(to.0, self.basis.0), ratio(to.1, self.basis.1));
        let [mut left, mut top, mut right, mut bottom] = self.edges;
        (left, right) = (left * sx, right * sx);
        (top, bottom) = (top * sy, bottom * sy);
        if flip_h {
            (left, right) = (to.0 as f64 - right, to.0 as f64 - left);
        }
        if flip_v {
            (top, bottom) = (to.1 as f64 - bottom, to.1 as f64 - top);
        }
        PendingFocal {
            zoom: self.zoom,
            edges: [left, top, right, bottom],
            basis: to,
        }
    }

    /// Through a cut to `window`.
    /// Whatever falls outside the window lands on its nearest
    /// edge, so a rectangle the cut removed entirely still
    /// pulls the crop toward the side it was on.
    pub(in crate::engine) fn cropped(self, (x, y, w, h): Window) -> Self {
        let [left, top, right, bottom] = self.edges;
        let along =
            |v: f64, origin: u32, extent: u32| (v - origin as f64).clamp(0.0, extent as f64);
        PendingFocal {
            zoom: self.zoom,
            edges: [
                along(left, x, w),
                along(top, y, h),
                along(right, x, w),
                along(bottom, y, h),
            ],
            basis: (w, h),
        }
    }

    /// Through a fit onto a larger canvas: the image scaled
    /// to `placed`'s size and set at its offset on a
    /// `canvas`-sized image.
    pub(in crate::engine) fn placed(self, (x, y, w, h): Window, canvas: (u32, u32)) -> Self {
        let fitted = self.scaled((w, h), false, false);
        let [left, top, right, bottom] = fitted.edges;
        let (dx, dy) = (x as f64, y as f64);
        PendingFocal {
            zoom: self.zoom,
            edges: [left + dx, top + dy, right + dx, bottom + dy],
            basis: canvas,
        }
    }

    /// Resolve against the `dims` image the crop cuts. The
    /// steps keep the basis in step with the image; a size
    /// change no step mapped scales over.
    pub(in crate::engine) fn resolve(self, dims: (u32, u32)) -> Focal {
        let at = if self.basis == dims {
            self
        } else {
            self.scaled(dims, false, false)
        };
        let [left, top, right, bottom] = at.edges;
        match at.zoom {
            None => {
                let px = |v: f64, extent: u32| v.round().clamp(0.0, extent as f64) as u32;
                let (left, top) = (px(left, dims.0), px(top, dims.1));
                let (right, bottom) = (px(right, dims.0).max(left), px(bottom, dims.1).max(top));
                Focal::Rect((left, top, right, bottom))
            }
            Some(zoom) => {
                let frac = |v: f64, extent: u32| {
                    if extent == 0 {
                        0.5
                    } else {
                        (v / extent as f64).clamp(0.0, 1.0) as f32
                    }
                };
                Focal::Point((frac(left, dims.0), frac(top, dims.1), zoom))
            }
        }
    }
}

/// Build the debug capture for a crop that reported a
/// [`crate::crop::CropTrace`]: the traced focal fraction
/// becomes a pixel point on the canvas the crop ran on.
pub(in crate::engine) fn capture_from_trace(
    canvas: DynamicImage,
    trace: crate::crop::CropTrace,
) -> crate::debugview::FocalCapture {
    let (w, h) = (canvas.width(), canvas.height());
    let focal = trace.focal.map(|p| {
        (
            ((p.x * w as f32).round() as u32).min(w.saturating_sub(1)),
            ((p.y * h as f32).round() as u32).min(h.saturating_sub(1)),
        )
    });
    crate::debugview::FocalCapture {
        canvas,
        focal,
        focal_rect: None,
        window: Some(trace.window),
        faces: trace.faces,
        objects: Vec::new(),
    }
}
