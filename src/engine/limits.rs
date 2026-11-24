//! Resource limits (issue #741): the shared decode, output
//! and animation limits a render is held to, the checks that
//! hold each step's output and the final image to them, the
//! animated output budget across frames, the operation-count
//! backstop, and the client hint's DPR factor, which steps
//! down when only it pushes an output past the limits (issue
//! #529).
//!
//! engine.rs re-exports what the pipeline, the crate and the
//! tests use.

use super::animated::webp_animation_params;
use super::decode::{SourceProbe, exif_orientation, is_heic, probe_source};
use super::{EngineError, ProcessContext};
use crate::url::{Dimension, Operation};

/// Shared resource limits for decoding and output sizing
/// (security audit E-1/E-2/E-3/E-4/E-9): one policy, applied at
/// every entry point, so a hostile source or an oversized
/// request fails cheaply instead of allocating gigabytes.
pub(crate) const MAX_DECODE_DIMENSION: u32 = 16_384;
pub(crate) const MAX_DECODE_ALLOC: u64 = 256 * 1024 * 1024;
pub(in crate::engine) const MAX_ANIMATED_FRAMES: usize = 200;
pub(in crate::engine) const MAX_ANIMATED_PIXELS: u64 = 100_000_000;
pub(crate) const MAX_OUTPUT_PIXELS: u64 = 100_000_000;

/// The most bytes the frame a step allocates may hold (issue
/// #784): what an RGBA8 frame at [`MAX_OUTPUT_PIXELS`] holds.
/// A step allocates its frame in the layout it works in, for a
/// resize the layout of the frame in hand, so a 16-bit color
/// or float frame at the pixel limit would hold two to four
/// times as much.
pub(crate) const MAX_OUTPUT_BYTES: u64 = MAX_OUTPUT_PIXELS * 4;

/// Reject an output size past the shared pixel budget, before
/// anything allocates it. u64 math, so DPR-scaled dimensions
/// cannot wrap past the check.
pub(in crate::engine) fn check_output_pixels(width: u32, height: u32) -> Result<(), EngineError> {
    if width as u64 * height as u64 > MAX_OUTPUT_PIXELS {
        return Err(EngineError::Unsupported(format!(
            "requested output {width}x{height} exceeds the {MAX_OUTPUT_PIXELS} pixel limit"
        )));
    }
    Ok(())
}

/// Reject a `width` x `height` frame in `layout` past
/// [`MAX_OUTPUT_BYTES`], before anything allocates it (issue
/// #784). An 8-bit layout holds four bytes a pixel at most, so
/// a frame within the pixel budget is within this one: it
/// binds on 16-bit color and float frames, and on a frame no
/// pixel check holds, the whole cover a blurred pad scales a
/// gray, 16-bit or float image to. The refusal starts
/// `requested output WxH`, as the pixel budget's does, so a
/// hinted DPR factor steps down from it.
pub(in crate::engine) fn check_output_bytes(
    width: u32,
    height: u32,
    layout: image::ColorType,
) -> Result<(), EngineError> {
    let per_pixel = u64::from(layout.bytes_per_pixel());
    if (width as u64 * height as u64).saturating_mul(per_pixel) > MAX_OUTPUT_BYTES {
        let unit = if per_pixel == 1 { "byte" } else { "bytes" };
        return Err(EngineError::Unsupported(format!(
            "requested output {width}x{height} at {per_pixel} {unit} a pixel exceeds the \
             {MAX_OUTPUT_BYTES} byte limit"
        )));
    }
    Ok(())
}

/// Reject a size the engine works out for a step's output
/// past the per-axis limit a size written in the URL is held
/// to ([`crate::url::MAX_DIMENSION`]), then past the pixel
/// budget (issue #521). Every step that resolves its own
/// size checks it here before allocating it: a resize with a
/// proportional or `orig` side, a fit-in scale, a crop or pad
/// target, a DPR factor on any of them, a rotated canvas.
/// The parser bounds only what the URL spells, and a chain
/// can grow a proportional side past it inside the budget:
/// `16383x1/0x2` keeps the aspect ratio at height 2 and asks
/// for 32766x2, which WebP cannot encode. The step
/// is refused, as a written side past the limit is, rather
/// than clamped to a size nobody asked for.
///
/// `input` is the size the step starts from. A step that
/// ends at that same size works nothing out (`0x0`,
/// `fit-in:0x0`, `rotate(180)`, a width hint or `no_upscale`
/// held at the source), so it is held to the budget only: a
/// source at the 16384 px decode ceiling passes through it
/// as decoded, and a later step can still bring it within
/// the limit. Whatever the chain leaves at the end is held
/// to the limit by [`check_final_output`].
pub(in crate::engine) fn check_output_size(
    width: u32,
    height: u32,
    input: (u32, u32),
) -> Result<(), EngineError> {
    let max = crate::url::MAX_DIMENSION;
    if (width, height) != input && (width > max || height > max) {
        return Err(EngineError::Unsupported(format!(
            "requested output {width}x{height} exceeds the {max} pixel limit per axis"
        )));
    }
    check_output_pixels(width, height)
}

/// The last word on size, for the image a chain hands its
/// encoder: within [`crate::url::MAX_DIMENSION`] per axis,
/// then within the pixel budget (issue #528). The steps
/// let a size they did not change pass (see
/// [`check_output_size`]), so a source at the 16384 px
/// decode ceiling reached the encoder as decoded when no
/// step resized it: WebP refused the side with a `500`, and
/// every other format served an output past the documented
/// limit. There is no later step to bring the size within
/// the limit, so this check has no such exemption.
///
/// The refusal starts `requested output WxH exceeds`, as
/// the steps' refusals do: it is the same refusal of an
/// output size, made later, and whatever tells output-size
/// refusals from others by that wording must count it too.
pub(in crate::engine) fn check_final_output(width: u32, height: u32) -> Result<(), EngineError> {
    let max = crate::url::MAX_DIMENSION;
    if width > max || height > max {
        return Err(EngineError::Unsupported(format!(
            "requested output {width}x{height} exceeds the {max} pixel limit per axis; \
             resize or crop it to fit"
        )));
    }
    check_output_pixels(width, height)
}

/// A running total of output pixels across the frames of one
/// animated or video output, charged against the same budget
/// a single still output gets (issue #182, audit R2-01). The
/// per-frame check alone lets 200 individually legal frames
/// retain hundreds of millions of pixels before any encoder
/// runs; charging each frame as it is produced rejects the
/// request once the total passes the budget, so what is
/// retained stays within the budget plus one frame.
pub(in crate::engine) struct OutputBudget {
    limit: u64,
    total: u64,
}

impl OutputBudget {
    pub(in crate::engine) fn new() -> Self {
        Self {
            limit: MAX_OUTPUT_PIXELS,
            total: 0,
        }
    }

    #[cfg(test)]
    pub(in crate::engine) fn with_limit(limit: u64) -> Self {
        Self { limit, total: 0 }
    }

    /// Charge one frame's output pixels against the budget.
    pub(in crate::engine) fn charge(&mut self, width: u32, height: u32) -> Result<(), EngineError> {
        self.total = self
            .total
            .saturating_add(u64::from(width) * u64::from(height));
        if self.total > self.limit {
            return Err(self.refusal());
        }
        Ok(())
    }

    /// Refuse up front when `frames` frames of `width` x
    /// `height` would pass the budget. Every frame of one
    /// animation renders at one size (each starts at the
    /// canvas size, and the trim and smart-crop windows are
    /// shared), so the first frame settles what the run will
    /// charge: the refusal comes after one frame instead of a
    /// budget's worth, with the message the charge gives. A
    /// hinted DPR factor tried and refused for it costs one
    /// frame this way (issue #529).
    pub(in crate::engine) fn check_frames(
        &self,
        width: u32,
        height: u32,
        frames: usize,
    ) -> Result<(), EngineError> {
        let run = (u64::from(width) * u64::from(height)).saturating_mul(frames as u64);
        if self.total.saturating_add(run) > self.limit {
            return Err(self.refusal());
        }
        Ok(())
    }

    /// The refusal of a run past the budget.
    fn refusal(&self) -> EngineError {
        EngineError::Unsupported(format!(
            "animated output exceeds the {} pixel budget across frames",
            self.limit
        ))
    }
}

/// Reject an operation list past
/// [`crate::url::MAX_OPERATIONS`] (issue #182, audit
/// R2-12). The server refuses such a request at parse time,
/// before any fetch (issue #469); this is the backstop for
/// every other caller of the engine, with the same message
/// and the same `unsupported_operation` class.
pub(in crate::engine) fn check_operation_count(ops: &[Operation]) -> Result<(), EngineError> {
    crate::url::check_operation_count(ops).map_err(|e| EngineError::Unsupported(e.to_string()))
}

/// The effective DPR factor: an explicit `dpr(...)` directive
/// wins, then the client hint carried on the context, else 1.
pub(in crate::engine) fn dpr_factor(explicit: Option<f32>, ctx: &ProcessContext<'_>) -> f32 {
    explicit.or(ctx.dpr).unwrap_or(1.0)
}

/// Run `render` for a request whose source is `bytes`, at
/// the DPR factor its hint `hint` settles on (issue #529):
/// first at the factor [`hint_start`] picks from the source's
/// header, then at each lower factor [`hint_step_down`] names
/// while the render still refuses its output size. The
/// result at the largest factor that fits, or the refusal
/// that stands. A refused try stops at the step that asks
/// for the oversized image, before that step allocates it.
pub(in crate::engine) fn step_down_hint<T>(
    bytes: &[u8],
    ops: &[Operation],
    directives: &crate::url::RequestDirectives,
    hint: Option<f32>,
    mut render: impl FnMut(Option<f32>) -> Result<T, EngineError>,
) -> Result<T, EngineError> {
    let mut hint = hint_start(bytes, ops, directives, hint);
    loop {
        match render(hint) {
            Err(err) => match hint_step_down(&err, ops, directives, hint) {
                Some(lower) => hint = Some(lower),
                None => return Err(err),
            },
            done => return done,
        }
    }
}

/// The factor a hinted render starts at (issue #529): the
/// largest of [`crate::url::DPR_FACTORS`] up to `hint` whose
/// sizes the walk of [`crate::explain::settle_hint`] (the
/// `/explain` forecast's) finds within the output limits, so
/// a factor the limits plainly refuse costs no decode. The
/// walk starts from the size the decode hands the chain: the
/// header's, turned by the EXIF orientation where the decode
/// turns it. Where the header states no size, and for an SVG
/// (whose intrinsic size takes a worker of its own) or a
/// HEIC (whose decoder turns it by metadata of its own), the
/// render starts at the hint and steps on its refusals alone.
fn hint_start(
    bytes: &[u8],
    ops: &[Operation],
    directives: &crate::url::RequestDirectives,
    hint: Option<f32>,
) -> Option<f32> {
    let factor = hint?;
    if directives.dpr.is_some()
        || factor <= 1.0
        || !scales_with_dpr(ops)
        || crate::svg::is_svg(bytes)
        || is_heic(bytes)
    {
        return hint;
    }
    let probe = probe_source(bytes);
    let Some((width, height)) = probe.dimensions else {
        return hint;
    };
    // `decode_source` turns a still by its EXIF orientation;
    // GIF and animated WebP frames keep their stored size.
    let turned = probe.kind != "gif"
        && webp_animation_params(bytes).is_none()
        && matches!(exif_orientation(bytes), Some(5..=8));
    let dimensions = Some(if turned {
        (height, width)
    } else {
        (width, height)
    });
    Some(crate::explain::settle_hint(
        ops,
        directives,
        &SourceProbe {
            dimensions,
            ..probe
        },
        factor,
    ))
}

/// The factor a render refused at the hinted DPR factor
/// `hint` tries next, or `None` when the refusal stands
/// (issue #529). A client hint is advisory, so when it alone
/// pushes a size past the per-axis limit or the pixel budget,
/// the render steps down [`crate::url::DPR_FACTORS`] to the
/// largest factor whose sizes fit both; at factor 1 the URL
/// itself is over, and its refusal stands. Nothing else steps:
/// a factor from `dpr(...)` (in the URL, a preset, or the
/// operator's default operations) is the request's own and
/// keeps its `400`, a refusal of any other kind is not the
/// factor's doing, and a chain with no pixel size for the
/// factor to scale renders the same at every factor.
pub(in crate::engine) fn hint_step_down(
    err: &EngineError,
    ops: &[Operation],
    directives: &crate::url::RequestDirectives,
    hint: Option<f32>,
) -> Option<f32> {
    if directives.dpr.is_some() || !is_output_size_refusal(err) || !scales_with_dpr(ops) {
        return None;
    }
    dpr_step_below(hint?)
}

/// The largest of [`crate::url::DPR_FACTORS`] below `factor`,
/// the next one a hinted factor steps down to (issue #529);
/// `None` at factor 1.
pub(crate) fn dpr_step_below(factor: f32) -> Option<f32> {
    crate::url::DPR_FACTORS
        .iter()
        .rev()
        .copied()
        .find(|&lower| lower < factor)
}

/// How every refusal of an output size opens: a step past
/// the per-axis limit, the pixel budget or the byte budget
/// ([`check_output_size`], [`check_output_pixels`],
/// [`check_output_bytes`]), a hinted
/// output past the per-axis limit ([`check_hinted_output`]),
/// a scaled overlay past the budget, and an animated output
/// past the budget across its frames ([`OutputBudget`]). A
/// check that refuses an output size later in the pipeline
/// words its refusal `requested output WxH exceeds ...` too,
/// so the step-down counts it.
const OUTPUT_SIZE_REFUSALS: [&str; 3] = [
    "requested output ",
    "scaled overlay ",
    "animated output exceeds ",
];

/// Whether `err` is a refusal of an output size (see
/// [`OUTPUT_SIZE_REFUSALS`]): the refusals a DPR factor can
/// cause, since it scales the sizes they measure.
fn is_output_size_refusal(err: &EngineError) -> bool {
    match err {
        EngineError::Unsupported(what) => OUTPUT_SIZE_REFUSALS
            .iter()
            .any(|opening| what.starts_with(opening)),
        _ => false,
    }
}

/// Hold the image a render at a hinted DPR factor past 1
/// ends with to [`crate::url::MAX_DIMENSION`] per axis, a
/// size no step changed included (issue #529). The steps let
/// a size they did not change pass ([`check_output_size`]),
/// so a hinted factor could take a resize to a 16384 px
/// source's own size and serve that side (or, as WebP, fail
/// to encode it), past the limit a stepped factor promises to
/// fit. The refusal is an output-size refusal, so the factor
/// steps down. At factor 1 the render is the one the URL gets
/// without the hint, under its own rules.
pub(in crate::engine) fn check_hinted_output(width: u32, height: u32) -> Result<(), EngineError> {
    let max = crate::url::MAX_DIMENSION;
    if width > max || height > max {
        return Err(EngineError::Unsupported(format!(
            "requested output {width}x{height} exceeds the {max} pixel limit per axis; \
             resize or crop it to fit"
        )));
    }
    Ok(())
}

/// Whether any step of `ops` names a pixel size, the one
/// thing a DPR factor scales: a resize, `fit-in`, `crop`, or
/// `pad` with a pixel axis. An SVG source's raster size
/// follows the first of these too.
pub(crate) fn scales_with_dpr(ops: &[Operation]) -> bool {
    ops.iter().any(|op| match op {
        Operation::Size(size)
        | Operation::Crop { size, .. }
        | Operation::Pad { size, .. }
        | Operation::FitIn { size, .. } => {
            matches!(size.width, Dimension::Pixels(_))
                || matches!(size.height, Dimension::Pixels(_))
        }
        _ => false,
    })
}
