use image::DynamicImage;

use super::{EngineError, ProcessContext, SegmentObject};
use crate::url::{Filter, PlainFilter};

#[cfg(doc)]
use super::ChainState;

/// Whether a filter leaves every existing pixel where it
/// was (issue #373): the detection memos key on the image
/// dimensions, so a chain step that moves content (a
/// flip, a rotation, a trim, an upscale, a border, an
/// overlay a face could hide behind) must invalidate
/// them, or the next detection consumer is served boxes
/// measured on a state the pixels have left, and a
/// redaction lands where the object no longer is.
/// Photometric filters, the detection consumers
/// themselves, region filters (their coordinates are the
/// request's own), and encode-time directives preserve
/// positions. Anything not listed here invalidates: a
/// new filter variant errs toward a fresh inference,
/// never toward stale boxes.
pub(in crate::engine) fn filter_preserves_positions(filter: &Filter) -> bool {
    matches!(
        filter,
        Filter::Plain(
            PlainFilter::Grayscale
                | PlainFilter::Invert
                | PlainFilter::Blur(_)
                | PlainFilter::Brighten(_)
                | PlainFilter::Contrast(_)
                | PlainFilter::Gamma(_)
                | PlainFilter::Exposure(_)
                | PlainFilter::HighlightsShadows(..)
                | PlainFilter::Vibrance(_)
                | PlainFilter::HueRotate(_)
                | PlainFilter::Sharpen(_)
                | PlainFilter::Saturation(_)
                | PlainFilter::Rgb(..)
                | PlainFilter::Curve(..)
                | PlainFilter::Convolution(..)
                | PlainFilter::Noise(_)
                | PlainFilter::Equalize
                | PlainFilter::Enhance
                | PlainFilter::Sepia
                | PlainFilter::Duotone(..)
                | PlainFilter::Monochrome(_)
                | PlainFilter::Pixelate(_)
                | PlainFilter::Fill(_)
                | PlainFilter::Radius(_)
                | PlainFilter::Mask(_)
                | PlainFilter::KeepProfile
                | PlainFilter::MaxBytes(_)
                | PlainFilter::Quality(_)
                | PlainFilter::Palette(_)
                | PlainFilter::Format(_)
                | PlainFilter::NoUpscale
        ) | Filter::BlurFaces
            | Filter::PixelateFaces
            | Filter::BlurClass(_)
            | Filter::PixelateClass(_)
            | Filter::RecolorClass(..)
            | Filter::FacePad(_)
            | Filter::BlurRegion { .. }
            | Filter::PixelateRegion { .. }
            | Filter::RemoveRegion { .. }
            | Filter::Dpr(_)
    )
}

/// Drop every detection memo: the chain step that just
/// ran moved the content the memos' boxes describe.
pub(in crate::engine) fn invalidate_detection_memos(
    face_memo: &mut FaceMemo,
    detection_memos: &mut [DetectionMemo],
    segment_memo: &mut SegmentMemo,
) {
    *face_memo = None;
    detection_memos.iter_mut().for_each(|memo| *memo = None);
    *segment_memo = None;
}

/// Memoized face detection for one pipeline run: the image
/// dimensions the detections belong to, and the provider's
/// result. Several face operations on the same image state
/// share one inference; an operation that changes the size
/// re-detects against the new state, and a step that moves
/// content invalidates the memo outright (see
/// `filter_preserves_positions`).
pub(in crate::engine) type FaceMemo = Option<(
    u32,
    u32,
    Result<Vec<crate::ai::FaceBox>, crate::ai::AiError>,
)>;

/// Detect faces in `img` through the context's provider,
/// serving the memoized result when the image state matches.
/// The provider's failure surfaces as an inference error;
/// callers with a fallback (the face crop) may downgrade it.
pub(in crate::engine) fn detect_faces(
    memo: &mut FaceMemo,
    img: &DynamicImage,
    ctx: &ProcessContext<'_>,
) -> Result<Vec<crate::ai::FaceBox>, EngineError> {
    let provider = ctx.face_provider.ok_or_else(|| {
        EngineError::NotConfigured(
            "face operations are not configured on this server (no face model)".to_string(),
        )
    })?;
    if let Some((w, h, cached)) = &memo
        && *w == img.width()
        && *h == img.height()
    {
        return cached.clone().map_err(EngineError::Inference);
    }
    let found = provider.detect(img);
    *memo = Some((img.width(), img.height(), found.clone()));
    found.map_err(EngineError::Inference)
}

/// Memoized object detection for one pipeline run, the
/// face memo's twin: chained class filters on the same
/// image state share one inference.
pub(in crate::engine) type DetectionMemo = Option<(
    u32,
    u32,
    Result<Vec<crate::ai::Detection>, crate::ai::AiError>,
)>;

/// Every detection provider that can answer for this
/// request, in resolution order (issue #166): the built-in
/// detector first, then the declared detectors in
/// declaration order.
pub(in crate::engine) fn detection_providers<'a>(
    ctx: &ProcessContext<'a>,
) -> Vec<&'a dyn crate::ai::DetectionProvider> {
    let mut providers: Vec<&dyn crate::ai::DetectionProvider> = Vec::new();
    if let Some(provider) = ctx.detect_provider {
        providers.push(provider);
    }
    providers.extend(ctx.declared_detect_providers.iter().copied());
    providers
}

/// The slot of the first provider (in resolution order)
/// whose vocabulary knows `class`: the one provider a
/// class filter runs.
pub(in crate::engine) fn resolve_detector(
    providers: &[&dyn crate::ai::DetectionProvider],
    class: &str,
) -> Option<usize> {
    providers.iter().position(|p| p.knows_class(class))
}

/// Detect objects in `img` through one provider, memoized
/// like face detection. A provider failure is an inference
/// error: on a redaction request that fails the request
/// (never an unredacted response).
fn detect_objects(
    memo: &mut DetectionMemo,
    img: &DynamicImage,
    provider: &dyn crate::ai::DetectionProvider,
) -> Result<Vec<crate::ai::Detection>, EngineError> {
    if let Some((w, h, cached)) = &memo
        && *w == img.width()
        && *h == img.height()
    {
        return cached.clone().map_err(EngineError::Inference);
    }
    let found = provider.detect(img);
    *memo = Some((img.width(), img.height(), found.clone()));
    found.map_err(EngineError::Inference)
}

/// The detections of one class, through the provider that
/// resolution picks for it (issue #166): each provider has
/// its own memo slot, so chained class filters still share
/// one run per provider per image state.
pub(in crate::engine) fn detect_for_class(
    memos: &mut [DetectionMemo],
    providers: &[&dyn crate::ai::DetectionProvider],
    img: &DynamicImage,
    class: &str,
) -> Result<Vec<crate::ai::Detection>, EngineError> {
    if providers.is_empty() {
        return Err(EngineError::NotConfigured(
            "class redaction is not configured on this server (no detection model)".to_string(),
        ));
    }
    let slot = resolve_detector(providers, class).ok_or_else(|| {
        EngineError::Unsupported(format!(
            "unknown detection class {class:?}: not in the detection model's vocabulary"
        ))
    })?;
    let detections = detect_objects(&mut memos[slot], img, providers[slot])?;
    Ok(detections_of_class(&detections, class))
}

/// The detections of one class (case-insensitive, the
/// vocabulary's spelling is the model's own).
fn detections_of_class(
    detections: &[crate::ai::Detection],
    class: &str,
) -> Vec<crate::ai::Detection> {
    detections
        .iter()
        .filter(|d| d.label.eq_ignore_ascii_case(class))
        .cloned()
        .collect()
}

/// Memoized image encoding for one pipeline run (issue
/// #175): the image dimensions the embedding belongs to,
/// and the segment provider's result. The encoder is the
/// expensive half of segmentation (the decoder runs per
/// box), so chained class operations on the same image
/// state share one encoding, like detection memoization.
pub(in crate::engine) type SegmentMemo = Option<(
    u32,
    u32,
    Result<crate::ai::SegmentEmbedding, crate::ai::AiError>,
)>;

/// Segment one detection's box against the memoized
/// encoding of `img`. A provider failure is an inference
/// error: on a redaction request that fails the request,
/// the same rule detection follows (never an unredacted,
/// or silently rectangle-redacted, response).
fn segment_detection(
    memo: &mut SegmentMemo,
    provider: &dyn crate::ai::SegmentProvider,
    img: &DynamicImage,
    det: &crate::ai::Detection,
) -> Result<image::GrayImage, EngineError> {
    let dims = (img.width(), img.height());
    if memo.as_ref().is_some_and(|(w, h, _)| (*w, *h) != dims) {
        *memo = None;
    }
    let (_, _, cached) = memo.get_or_insert_with(|| (dims.0, dims.1, provider.encode(img)));
    let embedding = cached
        .as_ref()
        .map_err(|e| EngineError::Inference(e.clone()))?;
    provider
        .decode(embedding, dims.0, dims.1, (det.x, det.y, det.w, det.h))
        .map_err(EngineError::Inference)
}

/// The most box-prompted segmentation decodes one request
/// runs (issue #713). Each detection of a class filter is one
/// decode, and a crowd can hold hundreds of detections: an
/// evaluation measured about 1.2 s more for 35 more people in
/// a debug build, all of it holding one of the AI permits. 64
/// covers a group photograph, about two seconds at that rate;
/// the detections past it are redacted by their boxes (see
/// [`ChainState::class_targets`]).
pub(crate) const MAX_SEGMENT_DECODES: usize = 64;

/// The detections of one class operation with their
/// segmentation masks, plus the record entries the masks
/// produced (issue #175). The geometry is measured on
/// each mask and stated in image coordinates: the tight
/// box and centroid are the mask geometry crop anchoring
/// consumes. A mask in which the model found no object
/// (nothing over half strength) still redacts its box's
/// soft shape, it just contributes no geometry, so it
/// leaves no record entry.
pub(in crate::engine) fn segment_for_class(
    memo: &mut SegmentMemo,
    provider: &dyn crate::ai::SegmentProvider,
    img: &DynamicImage,
    detections: &[crate::ai::Detection],
    check_deadline: impl Fn() -> Result<(), EngineError>,
) -> Result<(Vec<crate::filters::MaskedDetection>, Vec<SegmentObject>), EngineError> {
    let mut masked = Vec::with_capacity(detections.len());
    let mut objects = Vec::new();
    for det in detections {
        // Between decodes, not only between filters (issue
        // #713): a request with many objects stops at its
        // deadline instead of decoding on.
        check_deadline()?;
        let mask = segment_detection(memo, provider, img, det)?;
        if let Some(geometry) = crate::ai::mask_geometry(&mask) {
            let (tx, ty, tw, th) = geometry.tight;
            objects.push(SegmentObject {
                label: det.label.clone(),
                score: det.score,
                detection: (det.x, det.y, det.w, det.h),
                tight_box: (det.x + tx, det.y + ty, tw, th),
                coverage: geometry.coverage,
                centroid: (
                    det.x + (geometry.centroid.0 * det.w as f32).round() as u32,
                    det.y + (geometry.centroid.1 * det.h as f32).round() as u32,
                ),
            });
        }
        masked.push(crate::filters::MaskedDetection {
            detection: det.clone(),
            mask,
        });
    }
    Ok((masked, objects))
}
