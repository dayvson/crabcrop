use image::DynamicImage;

use super::animated::{FrameShare, copy_of_lent};
use super::detect::{
    DetectionMemo, FaceMemo, MAX_SEGMENT_DECODES, SegmentMemo, detect_faces, detect_for_class,
    detection_providers, filter_preserves_positions, invalidate_detection_memos, segment_for_class,
};
use super::focal::{Focal, PendingFocal, capture_from_trace, filter_keeps_focal};
use super::limits::{
    check_final_output, check_hinted_output, check_operation_count, check_output_pixels,
    check_output_size, scales_with_dpr,
};
use super::overlay::{blend_overlay_keeping, overlay_watermark};
use super::resize::{
    SizeCap, SourceWindow, apply_fit_in, apply_fit_in_lent, apply_pad, apply_size,
    apply_size_cropped, apply_size_lent, apply_size_windowed, cap_dimensions, resize_lanczos3,
    scale_px,
};
use super::{
    EngineError, InpaintRecord, ObjectsCropRecord, ProcessContext, SegmentObject, SegmentRecord,
    jpeg_scaled,
};
use crate::filters::EncodeOptions;
use crate::url::{Align, Dimension, Filter, ImageSize, Operation, PlainFilter};

#[cfg(doc)]
use super::Engine;
#[cfg(doc)]
use super::animated::survey_trims;
#[cfg(doc)]
use super::resize::{resize_lanczos3_cut, resize_lanczos3_lent};

/// The largest width or height an AI upscale may produce.
/// Super-resolution multiplies activations by the factor
/// squared; this keeps the result inside a sane memory envelope.
const MAX_UPSCALE_DIM: u64 = 4096;

/// The largest input, in pixels, one AI upscale run may
/// take. A dynamic-size model's activation memory grows
/// with its input pixels, which the output cap above does
/// not bind at all: the campaign's Swin2SR x2 export ran a
/// 196,608-pixel composed run (the second half of an
/// `upscale(4)`) and a 172,800-pixel run fine, and
/// OOM-killed the whole process on a 504,000-pixel input
/// (issue #244). The budget sits at about half that kill
/// size, above every measured working run.
const MAX_UPSCALE_INPUT_PIXELS: u64 = 262_144;

/// AI-upscale `img` to `factor`x its current dimensions. The
/// provider's model runs at its native factor as many times as
/// needed; a final Lanczos pass lands the exact requested size
/// when the native factor does not divide it.
fn upscale_image(
    img: DynamicImage,
    factor: u32,
    provider: &dyn crate::ai::UpscaleProvider,
) -> Result<DynamicImage, EngineError> {
    let (tw, th) = (
        img.width() as u64 * factor as u64,
        img.height() as u64 * factor as u64,
    );
    if tw > MAX_UPSCALE_DIM || th > MAX_UPSCALE_DIM {
        return Err(EngineError::Unsupported(format!(
            "upscale({factor}) would produce {tw}x{th}, over the {MAX_UPSCALE_DIM}px limit; \
             resize the source first"
        )));
    }
    // Budget every run's input before any inference
    // happens (issue #244): a composed factor feeds each
    // run the previous run's output, so the projection
    // walks the same arithmetic the loop below will. A
    // run over the budget is a clean error here, never an
    // out-of-memory kill inside the model.
    let native = provider.factor().max(2) as u64;
    let mut run_pixels = img.width() as u64 * img.height() as u64;
    let mut projected = 1u64;
    while projected < factor as u64 {
        if run_pixels > MAX_UPSCALE_INPUT_PIXELS {
            return Err(EngineError::Unsupported(format!(
                "upscale({factor}) on a {}x{} image would feed a model run {run_pixels} \
                 pixels, over the {MAX_UPSCALE_INPUT_PIXELS} pixel input budget; \
                 resize the source first",
                img.width(),
                img.height()
            )));
        }
        run_pixels *= native * native;
        projected *= native;
    }
    let (target_w, target_h) = (tw as u32, th as u32);
    let mut img = img;
    let mut scale = 1u32;
    while scale < factor {
        let before = (img.width(), img.height());
        img = provider.upscale(&img).map_err(EngineError::Inference)?;
        if img.width() <= before.0 || img.height() <= before.1 {
            return Err(EngineError::Inference(crate::ai::AiError::BadOutput(
                "the upscale model did not enlarge the image".to_string(),
            )));
        }
        scale *= provider.factor().max(2);
    }
    if img.width() != target_w || img.height() != target_h {
        img = resize_lanczos3(&img, target_w, target_h);
    }
    Ok(img)
}

/// The crop-rect stance for explicit regions: a rectangle
/// that overflows the image in effect at this point in the
/// chain fails the request, rather than redacting less than
/// the caller specified.
fn check_region(
    name: &str,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    img: &DynamicImage,
) -> Result<(), EngineError> {
    // The coordinates are parser-supplied u32s: plain
    // addition can wrap (release) or panic (debug) before
    // the comparison ever runs, so compare the far edge
    // with checked arithmetic.
    let in_bounds = x.checked_add(w).is_some_and(|right| right <= img.width())
        && y.checked_add(h)
            .is_some_and(|bottom| bottom <= img.height());
    if !in_bounds {
        return Err(EngineError::Unsupported(format!(
            "{name}({x},{y},{w},{h}) exceeds image bounds {}x{}",
            img.width(),
            img.height()
        )));
    }
    Ok(())
}

/// What [`apply_operations`] hands back: the rendered
/// image, its encode options, and the records the
/// operations produced for the meta response.
#[derive(Debug)]
pub(in crate::engine) struct ApplyOutcome {
    pub(in crate::engine) img: DynamicImage,
    pub(in crate::engine) encode_opts: EncodeOptions,
    /// The studio record (issue #197).
    pub(in crate::engine) studio: Option<crate::studio::StudioRecord>,
    /// The region-removal record (issue #176).
    pub(in crate::engine) inpaint: Option<InpaintRecord>,
    /// The segmentation record (issue #175).
    pub(in crate::engine) segments: Option<SegmentRecord>,
    /// The objects-crop record (issue #318).
    pub(in crate::engine) objects_crop: Option<ObjectsCropRecord>,
}

/// A crop+resize fusion the chain qualifies for (issue
/// #287): the rectangle a `crop-rect` cuts, resampled to the
/// chain's resize target in a single fast_image_resize call
/// through its crop box, instead of materializing the window
/// and resampling that. The backend computes its convolution
/// coefficients against the full-source geometry when a crop
/// box is set, so the fused result differs from the two-step
/// chain by rounding-class deltas, measured at 1.0 to 3.5
/// percent of pixels differing, a max
/// delta of 4 on downscales and 10 on upscales, and a mean
/// delta of at most 0.027, for a 1.5 to 3.6x speedup on the
/// fused shapes. Any chain the gates in [`fusion_plan`] do
/// not accept runs the two-step path untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::engine) struct FusionPlan {
    /// Index of the `crop-rect` operation in the chain.
    pub(in crate::engine) crop_index: usize,
    /// Index of the resize operation in the chain.
    pub(in crate::engine) size_index: usize,
    /// The crop window: (left, top, width, height).
    pub(in crate::engine) window: (u32, u32, u32, u32),
}

/// Decide whether the chain's crop and resize fuse (issue
/// #287). The gates, one per condition:
///
/// - G1: exactly one crop in the chain, and it is a
///   `crop-rect`. Its window is static in the operation; an
///   aspect crop's window depends on focal and detection
///   state decided at run time, so it never fuses. (An
///   aspect crop's own resize reads its window from the
///   frame too, by a route that keeps the two-step bytes:
///   see [`resize_lanczos3_cut`].)
/// - G2: exactly one resize (`Operation::Size`) in the
///   chain.
/// - G3: the crop precedes the resize, and only decode-time
///   no-ops (`frame`, `page`) sit between them: no pixel
///   operation may run on the window before the resample.
/// - G4: the resize is the chain's final pixel operation:
///   only decode-time no-ops follow it.
/// - G5: no operation after the crop consumes the full
///   frame's geometry. A focal rectangle validates against
///   the frame in effect where it sits, a pad builds a
///   canvas from it, filters read regions and detections
///   from it; under fusion the frame at those points is the
///   un-cropped one, so any of them after the crop refuses
///   the fusion. G3 and G4 are this gate's two halves; it is
///   checked as its own scan so the condition stands on its
///   own.
/// - G6: the chain is not a debug chain (`debug:focal`
///   renders a diagnostic of the exact pixels, so it stays
///   on the two-step path), and the rectangle has positive
///   area.
pub(in crate::engine) fn fusion_plan(ops: &[Operation]) -> Option<FusionPlan> {
    // G1: one crop, and it is a crop-rect.
    let mut crop: Option<(usize, (u32, u32, u32, u32))> = None;
    for (index, op) in ops.iter().enumerate() {
        match op {
            Operation::CropRect {
                left,
                top,
                right,
                bottom,
            } => {
                if crop.is_some() {
                    return None;
                }
                crop = Some((index, (*left, *top, *right, *bottom)));
            }
            Operation::Crop { .. } => return None,
            _ => {}
        }
    }
    let (crop_index, (left, top, right, bottom)) = crop?;
    // G2: one resize.
    let mut size_index: Option<usize> = None;
    for (index, op) in ops.iter().enumerate() {
        if matches!(op, Operation::Size(_)) {
            if size_index.is_some() {
                return None;
            }
            size_index = Some(index);
        }
    }
    let size_index = size_index?;
    // G3: the crop precedes the resize, with only decode-time
    // no-ops between them.
    if crop_index > size_index {
        return None;
    }
    let decode_noop = |op: &Operation| matches!(op, Operation::Frame(_) | Operation::Page(_));
    if !ops[crop_index + 1..size_index].iter().all(decode_noop) {
        return None;
    }
    // G4: the resize is the final pixel operation.
    if !ops[size_index + 1..].iter().all(decode_noop) {
        return None;
    }
    // G5: nothing after the crop consumes the full frame.
    for (index, op) in ops.iter().enumerate().skip(crop_index + 1) {
        if index != size_index && !decode_noop(op) {
            return None;
        }
    }
    // G6: no debug chain, and the rectangle has positive
    // area.
    if ops.iter().any(|op| matches!(op, Operation::DebugFocal)) {
        return None;
    }
    if right <= left || bottom <= top {
        return None;
    }
    Some(FusionPlan {
        crop_index,
        size_index,
        window: (left, top, right - left, bottom - top),
    })
}

/// Apply the operations to one image, collecting the encoding
/// directives along the way. Shared by the still pipeline and
/// the animated paths (GIF and WebP), which run it once per
/// frame.
pub(in crate::engine) fn apply_operations(
    img: DynamicImage,
    ops: &[Operation],
    cap: Option<(u32, u32)>,
    dpr: f32,
    ctx: &ProcessContext<'_>,
    directives: &crate::url::RequestDirectives,
    share: Option<&mut FrameShare>,
) -> Result<ApplyOutcome, EngineError> {
    check_operation_count(ops)?;
    ChainState::new(ops, cap, dpr, ctx, directives, share).run(img, ops)
}

/// The running state of one [`apply_operations`] pass: the
/// chain's inputs, the encode options the filters collect,
/// the focal and debug state a crop consumes, the inference
/// memos for the image state in hand, and the records the
/// meta response carries.
pub(in crate::engine) struct ChainState<'r, 'a> {
    /// The `no_upscale` cap: the (oriented) source
    /// dimensions, when the request asked for it.
    cap: Option<(u32, u32)>,
    /// The effective DPR factor (see `dpr_factor`).
    dpr: f32,
    ctx: &'r ProcessContext<'a>,
    directives: &'r crate::url::RequestDirectives,
    /// What an animated render shares across its frames: its
    /// geometry (issues #376, #465) and its scaled overlay
    /// parts; `None` on a still.
    share: Option<&'r mut FrameShare>,
    encode_opts: EncodeOptions,
    face_memo: FaceMemo,
    /// The detection providers in resolution order, with
    /// one detection memo per provider (issue #166).
    detect_providers: Vec<&'a dyn crate::ai::DetectionProvider>,
    detection_memos: Vec<DetectionMemo>,
    /// The pending focal form from a `focal:` or
    /// `focalpoint:` segment (issue #313), consumed by the
    /// next crop. One slot holds both forms, so the last one
    /// in the chain is the one the crop obeys.
    focal: Option<PendingFocal>,
    /// The `debug:focal` state: armed by the segment, captured
    /// by the first crop that follows it. The capture holds
    /// the values that crop actually used (threaded out of the
    /// crop functions, never recomputed), and the response is
    /// the rendered capture instead of the transform result.
    debug_armed: bool,
    debug_capture: Option<crate::debugview::FocalCapture>,
    /// The studio record (issue #197): the last `studio(...)`
    /// filter in the chain wins, since its composite is the
    /// look the response carries.
    studio_record: Option<crate::studio::StudioRecord>,
    /// The region-removal record (issue #176): every
    /// `remove_region(...)` in the chain, in order.
    inpaint_regions: Vec<(u32, u32, u32, u32)>,
    /// The segmentation state (issue #175): the encoder
    /// memo (one encoding per image state, however many
    /// class operations decode against it) and the record
    /// of what was segmented, in chain order.
    segment_memo: SegmentMemo,
    segment_objects: Vec<SegmentObject>,
    /// How many box-prompted decodes the chain has run, held
    /// to [`MAX_SEGMENT_DECODES`] (issue #713).
    segment_decodes: usize,
    /// The objects-crop record (issue #318): the last
    /// `objects` crop in the chain wins, like the studio
    /// record, since its framing is the one the response
    /// carries.
    objects_crop_record: Option<ObjectsCropRecord>,
    /// The crop+resize fusion plan (issue #287): when the
    /// chain qualifies, the crop-rect arm records its window
    /// instead of materializing it, and the resize arm
    /// resamples the window straight from the frame in hand.
    fusion: Option<FusionPlan>,
    fused_window: Option<(u32, u32, u32, u32)>,
    /// The image the frame stands for when the chain starts
    /// from a reduced decode of it (issue #653), until the
    /// chain's first operation, a plain resize, has
    /// resampled the frame to its target (see
    /// [`Engine::still_chain`]).
    pub(in crate::engine) reduced: Option<jpeg_scaled::Reduced>,
}

/// What a class filter paints (see
/// [`ChainState::class_targets`]): the detections with their
/// segmentation masks, and the detections it paints by their
/// boxes. Without a segment provider every detection is a
/// box; with one, every detection is masked until the
/// request's decodes run out (see [`MAX_SEGMENT_DECODES`]).
struct ClassTargets {
    masked: Vec<crate::filters::MaskedDetection>,
    boxes: Vec<crate::ai::Detection>,
}

impl<'r, 'a> ChainState<'r, 'a> {
    pub(in crate::engine) fn new(
        ops: &[Operation],
        cap: Option<(u32, u32)>,
        dpr: f32,
        ctx: &'r ProcessContext<'a>,
        directives: &'r crate::url::RequestDirectives,
        share: Option<&'r mut FrameShare>,
    ) -> Self {
        let detect_providers = detection_providers(ctx);
        let detection_memos = vec![None; detect_providers.len()];
        ChainState {
            cap,
            dpr,
            ctx,
            directives,
            share,
            encode_opts: EncodeOptions::default(),
            face_memo: None,
            detect_providers,
            detection_memos,
            focal: None,
            debug_armed: false,
            debug_capture: None,
            studio_record: None,
            inpaint_regions: Vec::new(),
            segment_memo: None,
            segment_objects: Vec::new(),
            segment_decodes: 0,
            objects_crop_record: None,
            fusion: fusion_plan(ops),
            fused_window: None,
            reduced: None,
        }
    }

    /// Run the whole chain on `img`: every operation in
    /// order, then what the finished chain owes its caller.
    pub(in crate::engine) fn run(
        mut self,
        mut img: DynamicImage,
        ops: &[Operation],
    ) -> Result<ApplyOutcome, EngineError> {
        for (op_index, op) in ops.iter().enumerate() {
            img = self.step(img, op_index, op)?;
        }
        let hinted = self.directives.dpr.is_none() && self.dpr > 1.0 && scales_with_dpr(ops);
        let outcome = self.finish(img)?;
        if hinted {
            check_hinted_output(outcome.img.width(), outcome.img.height())?;
        }
        Ok(outcome)
    }

    /// Run one operation of the chain, with what every
    /// operation owes the chain around it: the deadline, the
    /// detection memos, and the output budget.
    pub(in crate::engine) fn step(
        &mut self,
        img: DynamicImage,
        op_index: usize,
        op: &Operation,
    ) -> Result<DynamicImage, EngineError> {
        self.ctx.check_deadline()?;
        let img = self.apply_op(img, op_index, op)?;
        self.stepped(img, op)
    }

    /// What an operation owes the chain once it has run (see
    /// [`ChainState::step`]): the detection memos and the
    /// output budget.
    fn stepped(&mut self, img: DynamicImage, op: &Operation) -> Result<DynamicImage, EngineError> {
        // Every operation other than the filters segment
        // sizes, crops, or pads, so the detection memos never
        // survive one (issue #373); the filters arm
        // invalidates per filter instead, and the no-op
        // arms (frame, page, debug) simply pay a fresh
        // inference in a chain that mixes them with
        // detection, which no real chain does.
        if !matches!(op, Operation::Filters(_)) {
            invalidate_detection_memos(
                &mut self.face_memo,
                &mut self.detection_memos,
                &mut self.segment_memo,
            );
        }
        // The running image stays inside the output budget
        // after every operation, not only after the ones
        // that size it (issue #182, audit R2-12). While a
        // fused window is pending (issue #287) the image in
        // hand is still the pre-crop frame: the two-step
        // path checks the window at this point, and the
        // fused crop ran that same check on the window when
        // it recorded it.
        if self.fused_window.is_none() {
            check_output_pixels(img.width(), img.height())?;
        }
        Ok(img)
    }

    /// Run the chain up to the filter at `key` (the operation
    /// and filter indexes) on a frame lent for the run, and
    /// return the image that filter receives: the trim
    /// survey's view of a frame (see [`survey_trims`]). The
    /// frame comes back as it was. A first operation that
    /// resamples the whole frame reads it where it is (see
    /// [`ChainState::lent_step`]); after any other, the chain
    /// runs on a copy of the frame (issue #737).
    pub(in crate::engine) fn run_to(
        &mut self,
        frame: &mut DynamicImage,
        ops: &[Operation],
        (op_index, filter_index): (usize, usize),
    ) -> Result<DynamicImage, EngineError> {
        let lent = match ops.first() {
            Some(first) if op_index > 0 => self.lent_step(frame, first)?,
            _ => None,
        };
        let (mut img, next) = match lent {
            Some(img) => (img, 1),
            None => (copy_of_lent(frame), 0),
        };
        for (index, op) in ops.iter().enumerate().take(op_index).skip(next) {
            img = self.step(img, index, op)?;
        }
        if let Some(Operation::Filters(filters)) = ops.get(op_index) {
            for (index, filter) in filters.iter().enumerate().take(filter_index) {
                img = self.filter_step(img, (op_index, index), filter)?;
            }
        }
        Ok(img)
    }

    /// The chain's first operation, `op`, on a frame lent for
    /// it (issue #737): a resize or a fit-in that resamples the
    /// whole frame reads it where it is (see
    /// [`resize_lanczos3_lent`]) and returns the image it
    /// makes, with what [`ChainState::step`] owes the chain
    /// around it. `None` for any other operation, and for a
    /// resize that leaves the frame at its size, which hands
    /// on the frame itself: those run on a copy.
    fn lent_step(
        &mut self,
        frame: &mut DynamicImage,
        op: &Operation,
    ) -> Result<Option<DynamicImage>, EngineError> {
        self.ctx.check_deadline()?;
        let img = match op {
            Operation::Size(size) if self.fused_window.is_none() && self.reduced.is_none() => {
                let cap = self.cap.map(|(cw, ch)| SizeCap::Axes(cw, ch));
                let Some(img) = apply_size_lent(frame, size, cap, self.dpr)? else {
                    return Ok(None);
                };
                self.focal_resized(&img, size.flip_horizontal, size.flip_vertical);
                img
            }
            Operation::FitIn { size, adaptive } => {
                let Some(img) = apply_fit_in_lent(frame, size, *adaptive, self.cap, self.dpr)?
                else {
                    return Ok(None);
                };
                self.focal_resized(&img, false, false);
                img
            }
            _ => return Ok(None),
        };
        self.stepped(img, op).map(Some)
    }

    /// Carry the pending focal form through a resize of the
    /// whole image to `img`, mirrored on the axes it flipped.
    fn focal_resized(&mut self, img: &DynamicImage, flip_h: bool, flip_v: bool) {
        let dims = (img.width(), img.height());
        self.focal = self.focal.map(|f| f.scaled(dims, flip_h, flip_v));
    }

    /// Apply one operation of the chain to the image in hand.
    fn apply_op(
        &mut self,
        mut img: DynamicImage,
        op_index: usize,
        op: &Operation,
    ) -> Result<DynamicImage, EngineError> {
        match op {
            Operation::DebugFocal => self.debug_armed = true,
            // Frame selection already happened at decode
            // (issue #214): the frame is the source image
            // this chain runs on, so there is nothing left
            // to apply here.
            Operation::Frame(_) => {}
            // Page selection likewise (issue #215): the
            // rendered page is the source image this chain
            // runs on.
            Operation::Page(_) => {}
            Operation::Size(size) => {
                // A pending fused window (issue #287) is the
                // plan's crop, waiting for this resize: the plan
                // admits one resize per chain, after the crop.
                img = if let Some(window) = self.fused_window.take() {
                    apply_size_cropped(
                        img,
                        window,
                        size,
                        self.cap.map(|(cw, ch)| SizeCap::Axes(cw, ch)),
                        self.dpr,
                    )?
                } else if let Some(reduced) = self.reduced.take() {
                    // The frame is a reduced decode (issue
                    // #653): the size resolves against the
                    // full image, as it would have on it.
                    apply_size_windowed(
                        img,
                        size,
                        self.cap.map(|(cw, ch)| SizeCap::Axes(cw, ch)),
                        self.dpr,
                        SourceWindow::Reduced(reduced),
                    )?
                } else {
                    apply_size(
                        img,
                        size,
                        self.cap.map(|(cw, ch)| SizeCap::Axes(cw, ch)),
                        self.dpr,
                    )?
                };
                self.focal_resized(&img, size.flip_horizontal, size.flip_vertical);
            }
            Operation::FitIn { size, adaptive } => {
                img = apply_fit_in(img, size, *adaptive, self.cap, self.dpr)?;
                self.focal_resized(&img, false, false);
            }
            Operation::Pad { size, fill } => {
                let (tw, th) = match (size.width, size.height) {
                    (Dimension::Pixels(w), Dimension::Pixels(h)) => {
                        (scale_px(w, self.dpr), scale_px(h, self.dpr))
                    }
                    _ => {
                        return Err(EngineError::Unsupported(
                            "pad with non-pixel dimensions".to_string(),
                        ));
                    }
                };
                let (padded, placed) = apply_pad(
                    img,
                    tw,
                    th,
                    *fill,
                    self.directives.fill,
                    self.cap,
                    self.ctx.deadline,
                )?;
                img = padded;
                let dims = (img.width(), img.height());
                self.focal = self.focal.map(|f| f.placed(placed, dims));
            }
            Operation::FocalRect {
                left,
                top,
                right,
                bottom,
            } => {
                // Same stance as crop-rect: a rectangle that
                // overflows the image in effect at this point
                // fails the request.
                if *right > img.width() || *bottom > img.height() {
                    return Err(EngineError::Unsupported(format!(
                        "focal {left}x{top}:{right}x{bottom} exceeds image bounds {}x{}",
                        img.width(),
                        img.height()
                    )));
                }
                self.focal = Some(PendingFocal::rect(
                    (*left, *top, *right, *bottom),
                    (img.width(), img.height()),
                ));
            }
            Operation::FocalPoint { x, y, zoom } => {
                // Fractions of the image in effect at this
                // point, so there is no bounds failure to
                // make: any image can carry the point.
                self.focal = Some(PendingFocal::point(
                    (*x, *y, *zoom),
                    (img.width(), img.height()),
                ));
            }
            Operation::CropRect {
                left,
                top,
                right,
                bottom,
            } => {
                if *right > img.width() || *bottom > img.height() {
                    return Err(EngineError::Unsupported(format!(
                        "crop-rect {left}x{top}:{right}x{bottom} exceeds image bounds {}x{}",
                        img.width(),
                        img.height()
                    )));
                }
                if self.fusion.is_some_and(|plan| plan.crop_index == op_index) {
                    // Issue #287: the bounds validation above
                    // already ran, so an out-of-bounds
                    // rectangle errors exactly as the two-step
                    // path does. The window itself is cut by
                    // the resize, not here; the budget check
                    // the two-step path runs on the cropped
                    // result at the end of this step runs on
                    // the window now, so an oversized window
                    // refuses the same way.
                    check_output_pixels(right - left, bottom - top)?;
                    self.fused_window = Some((*left, *top, right - left, bottom - top));
                } else {
                    img = img.crop_imm(*left, *top, right - left, bottom - top);
                }
                let window = (*left, *top, right - left, bottom - top);
                self.focal = self.focal.map(|f| f.cropped(window));
            }
            Operation::Crop { size, align } => img = self.crop(img, op_index, size, align)?,
            Operation::Filters(filters) => {
                for (filter_index, filter) in filters.iter().enumerate() {
                    img = self.filter_step(img, (op_index, filter_index), filter)?;
                }
            }
        }
        Ok(img)
    }

    /// One filter of a `filters:` segment, at `key` (the
    /// operation and filter indexes), with what every
    /// filter owes the chain around it: the deadline, the
    /// pending focal form, the detection memos, and the
    /// output budget. Trims run here, since they need their
    /// chain position to find the window an animated render
    /// shares (issue #465); every other filter goes through
    /// [`ChainState::apply_filter`].
    fn filter_step(
        &mut self,
        img: DynamicImage,
        key: (usize, usize),
        filter: &Filter,
    ) -> Result<DynamicImage, EngineError> {
        // The deadline is checked between filters, not only
        // between operations: a segment is one operation to
        // the per-operation check, so without this a long
        // segment (a hundred blurs in one `filters:`) runs
        // far past the deadline the runner already reported
        // to the client.
        self.ctx.check_deadline()?;
        let before = (img.width(), img.height());
        let img = match filter {
            Filter::Plain(PlainFilter::Trim(tolerance)) => self.trim(img, key, *tolerance, [0; 4]),
            Filter::Plain(PlainFilter::TrimSides(tolerance, sides)) => {
                self.trim(img, key, *tolerance, *sides)
            }
            _ => {
                let img = self.apply_filter(img, filter)?;
                // A filter that moves pixels in a way the
                // chain does not track (a rotation, a flip,
                // an upscale, studio) leaves the pending
                // focal form pointing at content that is no
                // longer there: drop it, and the crop falls
                // back to its alignment (issue #465).
                if !(filter_keeps_focal(filter) && (img.width(), img.height()) == before) {
                    self.focal = None;
                }
                img
            }
        };
        // A filter that moves content leaves the detection
        // memos describing a state the pixels have left
        // (issue #373): the next detection consumer must
        // re-detect.
        if !filter_preserves_positions(filter) {
            invalidate_detection_memos(
                &mut self.face_memo,
                &mut self.detection_memos,
                &mut self.segment_memo,
            );
        }
        // Every filter leaves the image inside the output
        // budget: growth is checked before it compounds
        // across a chain, not just at the size operations.
        check_output_pixels(img.width(), img.height())?;
        Ok(img)
    }

    /// A `trim` filter: the window the frame's own tolerance
    /// pass keeps, or on an animated render the window every
    /// frame shares (issue #465), less the named sides. A
    /// pending focal form moves with the cut.
    fn trim(
        &mut self,
        img: DynamicImage,
        key: (usize, usize),
        tolerance: u8,
        sides: [u32; 4],
    ) -> DynamicImage {
        let dims = (img.width(), img.height());
        let found = match self.share.as_deref().and_then(|s| s.trims.get(key, dims)) {
            Some(shared) => shared,
            None => crate::filters::trim_box(&img, tolerance),
        };
        let window = crate::filters::trim_window(dims, found, sides);
        self.focal = self.focal.map(|f| f.cropped(window));
        crate::filters::trim_to(img, found, sides)
    }

    /// The `crop:` operation: crop to the target's aspect
    /// ratio, positioned by a pending focal rectangle or
    /// point, or else by the alignment, then resize to
    /// exactly the target.
    fn crop(
        &mut self,
        img: DynamicImage,
        op_index: usize,
        size: &ImageSize,
        align: &Align,
    ) -> Result<DynamicImage, EngineError> {
        let (tw, th) = match (size.width, size.height) {
            (Dimension::Pixels(w), Dimension::Pixels(h)) => {
                (scale_px(w, self.dpr), scale_px(h, self.dpr))
            }
            _ => {
                return Err(EngineError::Unsupported(
                    "crop with non-pixel dimensions".to_string(),
                ));
            }
        };
        // The resize that closes the crop lands on this size
        // (the same `no_upscale` fit as below): refuse one past
        // the limit here, before any detection or saliency
        // pass runs on a crop that cannot be served (issue
        // #521). A target that keeps the image in hand as it
        // is passes, like any step that keeps its size.
        let target = match self.cap {
            Some(cap) => cap_dimensions(tw, th, cap),
            None => (tw, th),
        };
        check_output_size(target.0, target.1, (img.width(), img.height()))?;
        // The crop consumes any pending focal form, mapped
        // onto the image in hand by the steps since its
        // segment (issue #465). Center and smart crops
        // position on it (smart skips face and saliency
        // detection entirely); face and directional
        // alignments were explicit caller instructions and
        // ignore it.
        let focal = self
            .focal
            .take()
            .filter(|_| *align == Align::Center || *align == Align::Smart)
            .map(|f| f.resolve((img.width(), img.height())));
        // With the debug view armed, this crop is the
        // one the view annotates: clone the canvas it
        // operates on and run the traced crop variants,
        // which report the decision they actually made.
        let want_trace = self.debug_armed && self.debug_capture.is_none();
        let canvas = want_trace.then(|| img.clone());
        // Every arm hands back the frame and the window it
        // decided on. The traced arms, which the debug view
        // and an animation's first frame take, hand back the
        // window already cut.
        let uncut = if let Some(Focal::Rect(rect)) = focal {
            self.focal_rect_crop(img, (tw, th), rect, canvas)
        } else if let Some(Focal::Point(pt)) = focal {
            self.focal_point_crop(img, (tw, th), pt, canvas)
        } else if *align == Align::Face {
            self.face_crop(img, (tw, th), canvas)?
        } else if let Align::Objects { classes, scale } = align {
            self.objects_crop(img, (tw, th), classes, *scale, canvas)?
        } else if let Some(canvas) = canvas {
            let (cropped, trace) = crate::crop::crop_to_aspect_traced(
                img,
                tw,
                th,
                align.clone(),
                self.ctx.face_provider,
                self.ctx.focal_provider,
            );
            if let Some(trace) = trace {
                self.debug_capture = Some(capture_from_trace(canvas, trace));
            }
            crate::crop::Uncut::whole(cropped)
        } else if *align == Align::Smart
            && let Some(windows) = self.share.as_deref_mut().map(|s| &mut s.smart_windows)
        {
            windows.crop(img, (tw, th), op_index, align, self.ctx)
        } else {
            crate::crop::crop_to_aspect_uncut(
                img,
                tw,
                th,
                align.clone(),
                self.ctx.face_provider,
                self.ctx.focal_provider,
            )
        };
        // The resize that closes the crop reads the window
        // straight from the frame (issue #654). It sizes
        // and checks against the window's dimensions, as it
        // did against the cut window's.
        apply_size_windowed(
            uncut.img,
            &ImageSize {
                width: Dimension::Pixels(tw),
                height: Dimension::Pixels(th),
                flip_horizontal: false,
                flip_vertical: false,
            },
            self.cap.map(|(cw, ch)| SizeCap::Fit(cw, ch)),
            1.0,
            match uncut.window {
                Some(window) => SourceWindow::Cut(window),
                None => SourceWindow::Whole,
            },
        )
    }

    /// A center or smart crop positioned on a pending focal
    /// rectangle. `canvas` is set when the debug view
    /// annotates this crop.
    fn focal_rect_crop(
        &mut self,
        img: DynamicImage,
        (tw, th): (u32, u32),
        rect: (u32, u32, u32, u32),
        canvas: Option<DynamicImage>,
    ) -> crate::crop::Uncut {
        if let Some(canvas) = canvas {
            let (cropped, window) = crate::crop::crop_to_aspect_focal_traced(img, tw, th, rect);
            let (cw, ch) = (canvas.width(), canvas.height());
            self.debug_capture = Some(crate::debugview::FocalCapture {
                canvas,
                focal: Some((
                    ((rect.0 + rect.2) / 2).min(cw.saturating_sub(1)),
                    ((rect.1 + rect.3) / 2).min(ch.saturating_sub(1)),
                )),
                focal_rect: Some(rect),
                window,
                faces: Vec::new(),
                objects: Vec::new(),
            });
            crate::crop::Uncut::whole(cropped)
        } else {
            crate::crop::crop_to_aspect_focal_uncut(img, tw, th, rect).0
        }
    }

    /// A center or smart crop centered on a pending focal
    /// point (issue #313). `canvas` is set when the debug
    /// view annotates this crop.
    fn focal_point_crop(
        &mut self,
        img: DynamicImage,
        (tw, th): (u32, u32),
        pt: (f32, f32, f32),
        canvas: Option<DynamicImage>,
    ) -> crate::crop::Uncut {
        if let Some(canvas) = canvas {
            let (cropped, window) = crate::crop::crop_to_aspect_point_traced(img, tw, th, pt);
            let (cw, ch) = (canvas.width(), canvas.height());
            self.debug_capture = Some(crate::debugview::FocalCapture {
                canvas,
                focal: Some((
                    ((pt.0 * cw as f32).round() as u32).min(cw.saturating_sub(1)),
                    ((pt.1 * ch as f32).round() as u32).min(ch.saturating_sub(1)),
                )),
                focal_rect: None,
                window,
                faces: Vec::new(),
                objects: Vec::new(),
            });
            crate::crop::Uncut::whole(cropped)
        } else {
            crate::crop::crop_to_aspect_point_uncut(img, tw, th, pt).0
        }
    }

    /// Crop around the largest detected face. A detection
    /// failure or an empty result falls back to the saliency
    /// smart crop: a crop always exists, so it never fails
    /// the request. `canvas` is set when the debug view
    /// annotates this crop.
    fn face_crop(
        &mut self,
        img: DynamicImage,
        (tw, th): (u32, u32),
        canvas: Option<DynamicImage>,
    ) -> Result<crate::crop::Uncut, EngineError> {
        let faces = match detect_faces(&mut self.face_memo, &img, self.ctx) {
            Ok(faces) => faces,
            Err(EngineError::Inference(_)) => Vec::new(),
            Err(e) => return Err(e),
        };
        let largest = faces
            .iter()
            .max_by_key(|f| f.w as u64 * f.h as u64)
            .copied();
        let cropped = match largest {
            Some(face) => {
                if let Some(canvas) = canvas {
                    let (cropped, window) = crate::crop::crop_around_face_traced(
                        img,
                        tw,
                        th,
                        &face,
                        self.directives.face_padding,
                    );
                    self.debug_capture = Some(crate::debugview::FocalCapture {
                        canvas,
                        focal: Some((face.x + face.w / 2, face.y + face.h / 2)),
                        focal_rect: Some((face.x, face.y, face.x + face.w, face.y + face.h)),
                        window,
                        faces,
                        objects: Vec::new(),
                    });
                    crate::crop::Uncut::whole(cropped)
                } else {
                    crate::crop::crop_around_face_uncut(
                        img,
                        tw,
                        th,
                        &face,
                        self.directives.face_padding,
                    )
                    .0
                }
            }
            None => {
                if let Some(canvas) = canvas {
                    let (cropped, trace) = crate::crop::crop_to_aspect_traced(
                        img,
                        tw,
                        th,
                        Align::Smart,
                        None,
                        self.ctx.focal_provider,
                    );
                    if let Some(trace) = trace {
                        self.debug_capture = Some(capture_from_trace(canvas, trace));
                    }
                    crate::crop::Uncut::whole(cropped)
                } else {
                    crate::crop::crop_to_aspect_uncut(
                        img,
                        tw,
                        th,
                        Align::Smart,
                        None,
                        self.ctx.focal_provider,
                    )
                }
            }
        };
        Ok(cropped)
    }

    /// Crop to the union of the detected objects of the named
    /// classes (issue #318). Preflight already refused the
    /// request when no detection provider is configured or a
    /// named class is unknown, so what remains is the model's
    /// answer: a union to frame, or none. None (and a
    /// detection failure, downgraded exactly like the face
    /// crop's) falls back to the plain crop of the same size;
    /// the record states which path was taken either way.
    /// `canvas` is set when the debug view annotates this
    /// crop.
    fn objects_crop(
        &mut self,
        img: DynamicImage,
        (tw, th): (u32, u32),
        classes: &[String],
        scale: f32,
        canvas: Option<DynamicImage>,
    ) -> Result<crate::crop::Uncut, EngineError> {
        let mut used: Vec<crate::ai::Detection> = Vec::new();
        let mut answered = true;
        for class in classes {
            match detect_for_class(
                &mut self.detection_memos,
                &self.detect_providers,
                &img,
                class,
            ) {
                Ok(found) => used.extend(found),
                Err(EngineError::Inference(_)) => {
                    answered = false;
                    break;
                }
                Err(e) => return Err(e),
            }
        }
        let union = if answered {
            crate::crop::union_box(&used)
        } else {
            None
        };
        let cropped = match union {
            Some(union) => {
                self.objects_crop_record = Some(ObjectsCropRecord {
                    classes: classes.to_vec(),
                    objects: used.clone(),
                    union: Some((union.0, union.1, union.2 - union.0, union.3 - union.1)),
                    scale,
                    fallback: false,
                });
                if let Some(canvas) = canvas {
                    let (cropped, window) =
                        crate::crop::crop_around_objects_traced(img, tw, th, union, scale);
                    self.debug_capture = Some(crate::debugview::FocalCapture {
                        canvas,
                        focal: Some(((union.0 + union.2) / 2, (union.1 + union.3) / 2)),
                        focal_rect: Some(union),
                        window,
                        faces: Vec::new(),
                        objects: used,
                    });
                    crate::crop::Uncut::whole(cropped)
                } else {
                    crate::crop::crop_around_objects_uncut(img, tw, th, union, scale).0
                }
            }
            None => {
                self.objects_crop_record = Some(ObjectsCropRecord {
                    classes: classes.to_vec(),
                    objects: Vec::new(),
                    union: None,
                    scale,
                    fallback: true,
                });
                if let Some(canvas) = canvas {
                    let (cropped, trace) =
                        crate::crop::crop_to_aspect_traced(img, tw, th, Align::Center, None, None);
                    if let Some(trace) = trace {
                        self.debug_capture = Some(capture_from_trace(canvas, trace));
                    }
                    crate::crop::Uncut::whole(cropped)
                } else {
                    crate::crop::crop_to_aspect_uncut(img, tw, th, Align::Center, None, None)
                }
            }
        };
        Ok(cropped)
    }

    /// Apply one filter of a `filters:` segment: a plain
    /// filter runs through `crate::filters` (see
    /// [`ChainState::plain_filter`]), watermark and overlay
    /// filters resolve against the server-fetched images, the
    /// model and font filters run through their providers,
    /// and the region filters check their rectangle against
    /// the image in effect. The match names every variant
    /// (issue #741), so a new filter does not compile until
    /// it is applied here or is a [`PlainFilter`].
    pub(in crate::engine) fn apply_filter(
        &mut self,
        mut img: DynamicImage,
        filter: &Filter,
    ) -> Result<DynamicImage, EngineError> {
        match filter {
            Filter::Plain(plain) => img = self.plain_filter(img, plain)?,
            Filter::Watermark { url, x, y, alpha } => {
                let wm = self
                    .ctx
                    .watermarks
                    .and_then(|m| m.get(url))
                    .ok_or_else(|| {
                        EngineError::Unsupported(format!("watermark was not resolved: {url}"))
                    })?;
                img = overlay_watermark(self.ctx, &img, wm, *x, *y, *alpha)?;
            }
            Filter::Overlay {
                url,
                x,
                y,
                alpha,
                blend,
                width,
            } => {
                let ov = self
                    .ctx
                    .watermarks
                    .and_then(|m| m.get(url))
                    .ok_or_else(|| {
                        EngineError::Unsupported(format!("overlay was not resolved: {url}"))
                    })?;
                // The frames of an animated render share the
                // scaled part (see `ScaledOverlays`).
                let kept = self
                    .share
                    .as_deref_mut()
                    .map(|share| (url.as_str(), &mut share.overlays));
                img = blend_overlay_keeping(
                    self.ctx,
                    &img,
                    ov,
                    (*x, *y),
                    *alpha,
                    *blend,
                    *width,
                    kept,
                )?;
            }
            Filter::RemoveBg => {
                let provider = self.ctx.mask_provider.ok_or_else(|| {
                    EngineError::NotConfigured(
                        "removebg is not configured on this server".to_string(),
                    )
                })?;
                let mask = provider.mask(&img).map_err(EngineError::Inference)?;
                img = crate::ai::apply_mask(&img, &mask);
            }
            Filter::Studio(spec) => img = self.studio(&img, spec)?,
            Filter::BlurFaces => {
                // A detection failure fails the
                // request: serving the image
                // unredacted would be the real
                // failure.
                let faces = detect_faces(&mut self.face_memo, &img, self.ctx)?;
                img = crate::filters::blur_faces(&img, &faces, self.ctx.deadline)?;
            }
            Filter::PixelateFaces => {
                let faces = detect_faces(&mut self.face_memo, &img, self.ctx)?;
                img = crate::filters::pixelate_faces(&img, &faces);
            }
            Filter::BlurClass(class) => {
                let deadline = self.ctx.deadline;
                let ClassTargets { masked, boxes } = self.class_targets(&img, class)?;
                if !masked.is_empty() {
                    img = crate::filters::blur_detections_masked(&img, &masked, deadline)?;
                }
                if !boxes.is_empty() {
                    img = crate::filters::blur_detections(&img, &boxes, deadline)?;
                }
            }
            Filter::PixelateClass(class) => {
                let ClassTargets { masked, boxes } = self.class_targets(&img, class)?;
                if !masked.is_empty() {
                    img = crate::filters::pixelate_detections_masked(&img, &masked);
                }
                if !boxes.is_empty() {
                    img = crate::filters::pixelate_detections(&img, &boxes);
                }
            }
            Filter::RecolorClass(class, color) => {
                // The class rules are the
                // redaction filters': detection
                // resolves the class, and a
                // failure fails the request.
                // The paint follows the
                // silhouette when a segment
                // provider is configured, the
                // box when not (issue #175).
                let ClassTargets { masked, boxes } = self.class_targets(&img, class)?;
                if !masked.is_empty() {
                    img = crate::filters::recolor_detections_masked(&img, &masked, *color);
                }
                if !boxes.is_empty() {
                    img = crate::filters::recolor_detections(&img, &boxes, *color);
                }
            }
            // Request-wide directives, read from the request
            // as a whole: the face crop reads the padding, and
            // the DPR factor scales the size targets before
            // the chain runs. No pixels here.
            Filter::FacePad(_) | Filter::Dpr(_) => {}
            Filter::Upscale(factor) => {
                let provider = self.ctx.upscale_provider.ok_or_else(|| {
                    EngineError::NotConfigured(
                        "upscale is not configured on this server".to_string(),
                    )
                })?;
                img = upscale_image(img, *factor, provider)?;
            }
            Filter::BlurRegion { x, y, w, h } => {
                check_region("blur_region", *x, *y, *w, *h, &img)?;
                img = crate::filters::blur_region(&img, *x, *y, *w, *h, self.ctx.deadline)?;
            }
            Filter::PixelateRegion { x, y, w, h } => {
                check_region("pixelate_region", *x, *y, *w, *h, &img)?;
                img = crate::filters::pixelate_region(&img, *x, *y, *w, *h);
            }
            Filter::RemoveRegion { x, y, w, h } => {
                // Same bounds discipline as the
                // other region filters: a rectangle
                // that overflows the image in
                // effect here fails the request
                // rather than erasing less than
                // the caller named (issue #176).
                check_region("remove_region", *x, *y, *w, *h, &img)?;
                let provider = self.ctx.inpaint_provider.ok_or_else(|| {
                    EngineError::NotConfigured(
                        "remove_region is not configured on this server".to_string(),
                    )
                })?;
                let mask = crate::ai::region_mask(img.width(), img.height(), *x, *y, *w, *h);
                img = provider
                    .inpaint(&img, &mask)
                    .map_err(EngineError::Inference)?;
                self.inpaint_regions.push((*x, *y, *w, *h));
            }
            Filter::Text {
                text,
                size,
                color,
                anchor,
            } => {
                let renderer = self.ctx.text_renderer.ok_or_else(|| {
                    EngineError::NotConfigured(
                        "text is not configured on this server (no font)".to_string(),
                    )
                })?;
                // The renderer checks the deadline between
                // glyphs (issue #606).
                img = renderer
                    .draw(&img, text, *size, *color, *anchor, self.ctx.deadline)
                    .map_err(|e| match e {
                        crate::text::DrawError::Timeout => EngineError::Timeout,
                        crate::text::DrawError::Failed(message) => EngineError::Render(message),
                    })?;
            }
        }
        Ok(img)
    }

    /// Studio runs the removebg mask model, then expands in
    /// `crate::studio`: product mode composites (mask,
    /// defringe, backdrop and shadow, pad; issue #197), scene
    /// mode reframes and corrects (mask, crop, correct; issue
    /// #198). The record rides out with the result for meta.
    fn studio(
        &mut self,
        img: &DynamicImage,
        spec: &crate::url::StudioSpec,
    ) -> Result<DynamicImage, EngineError> {
        let provider = self.ctx.mask_provider.ok_or_else(|| {
            EngineError::NotConfigured(
                "studio is not configured on this server (no mask model)".to_string(),
            )
        })?;
        let t_timing = std::time::Instant::now();
        let mask = provider.mask(img).map_err(EngineError::Inference)?;
        let confidence = provider.confidence(img, &mask);
        crate::studio::timing_mark("inference", t_timing);
        let outcome = match spec.mode {
            // The shadow's blur checks the deadline before
            // each row (issue #614).
            crate::url::StudioMode::Product => crate::studio::composite_within(
                img,
                &mask,
                confidence,
                spec,
                self.ctx.brand_color,
                self.ctx.deadline,
            )?,
            crate::url::StudioMode::Scene => crate::studio::scene(img, &mask, confidence, spec),
        };
        self.studio_record = Some(outcome.record);
        Ok(outcome.image)
    }

    /// What a class filter paints: the detections of `class`
    /// in `img`, segmented into masks when a segment provider
    /// is configured (issue #175: mask-tight, the object's
    /// silhouette rather than its box), with the segmented
    /// objects added to the record. The #42 rule, inherited:
    /// a detection or segmentation failure fails the request
    /// rather than serve the image unredacted.
    ///
    /// A request runs at most [`MAX_SEGMENT_DECODES`] decodes
    /// across its class filters (issue #713), to the
    /// detections in the order the detector gives them, best
    /// first. Past that, the rest are painted by their boxes,
    /// as without a segment provider: a box holds the whole
    /// object, so nothing is left unredacted. The deadline is
    /// checked before each decode.
    fn class_targets(
        &mut self,
        img: &DynamicImage,
        class: &str,
    ) -> Result<ClassTargets, EngineError> {
        let mut boxes = detect_for_class(
            &mut self.detection_memos,
            &self.detect_providers,
            img,
            class,
        )?;
        let Some(provider) = self.ctx.segment_provider else {
            return Ok(ClassTargets {
                masked: Vec::new(),
                boxes,
            });
        };
        let room = MAX_SEGMENT_DECODES.saturating_sub(self.segment_decodes);
        let rest = boxes.split_off(room.min(boxes.len()));
        let ctx = self.ctx;
        let (masked, mut objects) =
            segment_for_class(&mut self.segment_memo, provider, img, &boxes, || {
                ctx.check_deadline()
            })?;
        self.segment_decodes += masked.len();
        self.segment_objects.append(&mut objects);
        Ok(ClassTargets {
            masked,
            boxes: rest,
        })
    }

    /// A plain filter: pixels in, pixels out through
    /// `apply_filters_within`, with any encoding directive it
    /// carries absorbed into the chain's encode options.
    fn plain_filter(
        &mut self,
        img: DynamicImage,
        filter: &PlainFilter,
    ) -> Result<DynamicImage, EngineError> {
        // Arbitrary-angle rotation is
        // the one plain filter that grows
        // the canvas: check the canvas it
        // is about to allocate before it
        // allocates (issue #182, R2-12),
        // per axis too (issue #521).
        if let PlainFilter::Rotate(degrees) = filter {
            let (cw, ch) = crate::filters::rotated_canvas(img.width(), img.height(), *degrees);
            check_output_size(cw, ch, (img.width(), img.height()))?;
        }
        // A blur checks the deadline before each row (issue
        // #614).
        let (new_img, options) = crate::filters::apply_filters_within(
            img,
            std::slice::from_ref(filter),
            self.directives.fill,
            self.ctx.deadline,
        )?;
        self.encode_opts.absorb(options);
        Ok(new_img)
    }

    /// Close the chain: render the debug view when
    /// `debug:focal` armed it, check the final image against
    /// the per-axis limit and the output budget, and hand
    /// back the image with the encode options and the
    /// records.
    pub(in crate::engine) fn finish(
        self,
        mut img: DynamicImage,
    ) -> Result<ApplyOutcome, EngineError> {
        if self.debug_armed {
            // The debug view replaces the transform result. With
            // no capture, no crop followed the segment: the view
            // shows the focal analysis of the final image. With
            // one, it shows the decision that crop actually made,
            // on the canvas it operated on.
            let capture = match self.debug_capture {
                Some(capture) => capture,
                None => {
                    let (point, faces) = crate::focus::focal_point_chained(
                        &img,
                        self.ctx.face_provider,
                        self.ctx.focal_provider,
                    );
                    let (w, h) = (img.width(), img.height());
                    crate::debugview::FocalCapture {
                        canvas: img.clone(),
                        focal: Some((
                            ((point.x * w as f32).round() as u32).min(w.saturating_sub(1)),
                            ((point.y * h as f32).round() as u32).min(h.saturating_sub(1)),
                        )),
                        focal_rect: None,
                        window: None,
                        faces,
                        objects: Vec::new(),
                    }
                }
            };
            img = crate::debugview::render(&capture);
        }
        // The last word on size: whatever the chain did, the
        // image that goes to the encoder is inside the output
        // budget (issue #182, audit R2-12) and the per-axis
        // limit (issue #528). The still, animated and video
        // paths all end here, once per frame.
        check_final_output(img.width(), img.height())?;
        let inpaint_record = if self.inpaint_regions.is_empty() {
            None
        } else {
            Some(InpaintRecord {
                regions: self.inpaint_regions,
            })
        };
        let segment_record = if self.segment_objects.is_empty() {
            None
        } else {
            Some(SegmentRecord {
                objects: self.segment_objects,
            })
        };
        Ok(ApplyOutcome {
            img,
            encode_opts: self.encode_opts,
            studio: self.studio_record,
            inpaint: inpaint_record,
            segments: segment_record,
            objects_crop: self.objects_crop_record,
        })
    }
}
