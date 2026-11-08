//! Metadata responses (issue #345): the meta document
//! the server answers for a signed URL when the client
//! asks with an `Accept: application/json` header, the
//! annotation helpers that compose its sections (tags,
//! detections, moderation, embedding, aesthetic, OCR),
//! and the Accept negotiation that selects them.
//!
//! server.rs re-exports the items the request path and the
//! tests use.

use std::sync::Arc;
use std::time::Instant;

use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};

use super::AppState;
use super::process::AI_GATE_WAIT;
use crate::url::ImageFormat;

/// Whether the client asked for the meta response: an explicit
/// `application/json` in `Accept` that is not refused with q=0.
pub(in crate::server) fn wants_meta(accept: Option<&str>) -> bool {
    let Some(accept) = accept else {
        return false;
    };
    for part in accept.split(',') {
        let mut segments = part.split(';');
        let media = segments.next().unwrap_or("").trim();
        if media != "application/json" {
            continue;
        }
        let refused = segments.any(|p| {
            p.trim()
                .strip_prefix("q=")
                .and_then(|v| v.trim().parse::<f32>().ok())
                .is_some_and(|q| q == 0.0)
        });
        if !refused {
            return true;
        }
    }
    false
}

/// Whether the accepted `application/json` entry also asks for
/// placeholder fields: a `placeholders` parameter on that entry
/// in an opt-in form (see `wants_meta_param`).
pub(in crate::server) fn wants_placeholders(accept: Option<&str>) -> bool {
    wants_meta_param(accept, "placeholders")
}

/// Whether the accepted `application/json` entry asks for
/// detection boxes: a `detections` parameter, same rule as
/// `placeholders`.
pub(in crate::server) fn wants_detections(accept: Option<&str>) -> bool {
    wants_meta_param(accept, "detections")
}

/// Whether the accepted `application/json` entry asks for
/// moderation verdicts: a `moderation` parameter, same
/// rule as `placeholders`.
pub(in crate::server) fn wants_moderation(accept: Option<&str>) -> bool {
    wants_meta_param(accept, "moderation")
}

/// Whether the accepted `application/json` entry asks
/// for the image embedding: an `embeddings` parameter,
/// same rule as `placeholders`.
pub(in crate::server) fn wants_embeddings(accept: Option<&str>) -> bool {
    wants_meta_param(accept, "embeddings")
}

/// Whether the accepted `application/json` entry asks
/// for OCR text: an `ocr` parameter, same rule as
/// `placeholders`.
pub(in crate::server) fn wants_ocr(accept: Option<&str>) -> bool {
    wants_meta_param(accept, "ocr")
}

/// Whether the accepted `application/json` entry carries
/// the named parameter in an opt-in form (issue #151): bare,
/// or set to exactly `1` or `true`, case-insensitively.
/// Anything else (`0`, `false`, or any other value) is not
/// an opt-in. Every `application/json` entry is considered;
/// an entry the client refused with `q=0` does not count.
fn wants_meta_param(accept: Option<&str>, name: &str) -> bool {
    let Some(accept) = accept else {
        return false;
    };
    for part in accept.split(',') {
        let mut segments = part.split(';');
        if segments.next().unwrap_or("").trim() != "application/json" {
            continue;
        }
        let params: Vec<&str> = segments.collect();
        let refused = params.iter().any(|p| {
            p.trim()
                .strip_prefix("q=")
                .and_then(|v| v.trim().parse::<f32>().ok())
                .is_some_and(|q| q == 0.0)
        });
        if refused {
            continue;
        }
        for p in &params {
            let mut kv = p.trim().splitn(2, '=');
            if kv.next().map(str::trim) != Some(name) {
                continue;
            }
            match kv.next().map(|v| v.trim().to_ascii_lowercase()).as_deref() {
                None => return true,
                Some("1" | "true") => return true,
                _ => {}
            }
        }
    }
    false
}

/// The wire name of a format, as used in meta responses.
fn format_name(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Png => "png",
        ImageFormat::WebP => "webp",
        ImageFormat::Jxl => "jxl",
        ImageFormat::Avif => "avif",
        ImageFormat::Gif => "gif",
        ImageFormat::Mp4 => "mp4",
        ImageFormat::WebM => "webm",
    }
}

/// The composed meta annotation: the tags that cleared the
/// threshold, and a readable alt text built from the best
/// labels. Composition lives here, at the call site, not
/// behind the tag provider (issue #152): the provider
/// returns raw scores, so threshold tuning and debugging
/// work from the same numbers the response is built from.
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    pub tags: Vec<crate::ai::TagScore>,
    pub alt_text: String,
}

/// Model output the meta path cannot carry honestly
/// (issue #352): a NaN or infinite score has no JSON
/// representation, and every composition downstream
/// (softmax, rounding, ranking) would propagate it
/// into the response body. A provider answers as a
/// unit, so one non-finite value rejects the whole
/// answer as bad model output; the meta helpers then
/// degrade the section exactly like an inference
/// failure.
fn check_finite_scores(
    producer: &str,
    values: impl IntoIterator<Item = f32>,
) -> Result<(), crate::ai::AiError> {
    if values.into_iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(crate::ai::AiError::BadOutput(format!(
            "{producer} produced a non-finite score"
        )))
    }
}

/// The zero-shot logit scale the composition applies to
/// raw scores before the softmax.
const TAG_LOGIT_SCALE: f32 = 100.0;

/// Turn raw per-label scores into the annotation:
/// probabilities are the softmax of `TAG_LOGIT_SCALE *
/// score`; tags are the labels at or above the threshold,
/// best first, at most 10, with scores rounded to 3
/// decimals; alt text is the top 3 labels joined with
/// commas, first letter capitalized.
pub(in crate::server) fn compose_annotation(
    scores: &[crate::ai::TagScore],
    threshold: f32,
) -> Annotation {
    let max_logit = scores
        .iter()
        .map(|s| s.score * TAG_LOGIT_SCALE)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores
        .iter()
        .map(|s| (s.score * TAG_LOGIT_SCALE - max_logit).exp())
        .collect();
    let total: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / total).collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let tags: Vec<crate::ai::TagScore> = order
        .iter()
        .filter(|&&i| probs[i] >= threshold)
        .take(10)
        .map(|&i| crate::ai::TagScore {
            label: scores[i].label.clone(),
            score: (probs[i] * 1000.0).round() / 1000.0,
        })
        .collect();
    let mut alt = order
        .iter()
        .take(3)
        .map(|&i| scores[i].label.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    if let Some(first) = alt.get_mut(..1) {
        first.make_ascii_uppercase();
    }
    Annotation {
        tags,
        alt_text: alt,
    }
}

/// Annotate a processed image for a meta response, when a
/// tag provider is configured. Bounded by the AI gate like
/// every other inference; any failure (gate busy, decode,
/// inference) omits the annotation rather than failing a
/// response whose image processed fine. When `ai_gated` is
/// set the request already holds an AI permit (an AI
/// transform answering meta), and the annotation runs
/// under it rather than acquiring a second permit (audit
/// R2-29).
pub(in crate::server) async fn annotate_meta(
    state: &AppState,
    processed: &crate::engine::ProcessedImage,
    ai_gated: bool,
) -> Option<Annotation> {
    let tagger = state.tagger.as_ref()?;
    // The permit moves into the blocking task below and
    // drops when the inference returns (issue #357): the
    // caller wraps this helper in the annotation deadline
    // and drops the future when it fires, and a permit
    // held on this side of the await would be freed at
    // that moment while the detached inference runs on
    // unaccounted. Every meta helper holds its permit
    // this way.
    let permit = if ai_gated {
        None
    } else {
        Some(
            tokio::time::timeout(AI_GATE_WAIT, state.ai_gate.clone().acquire_owned())
                .await
                .ok()?
                .ok()?,
        )
    };
    let img = crate::engine::decode_image(&processed.bytes).ok()?;
    let tagger = tagger.clone();
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        tagger.scores(&img)
    })
    .await
    {
        Ok(Ok(scores)) => match check_finite_scores("the tagger", scores.iter().map(|s| s.score)) {
            Ok(()) => Some(compose_annotation(&scores, state.tag_threshold)),
            Err(e) => {
                eprintln!("annotation error: {e}");
                None
            }
        },
        Ok(Err(e)) => {
            eprintln!("annotation error: {e}");
            None
        }
        Err(e) => {
            eprintln!("annotation task failed: {e}");
            None
        }
    }
}

/// Run object detection for a meta response, when at
/// least one detection provider is configured and the
/// request asked (`detections=1`). Detections aggregate
/// every configured provider (issue #166): the built-in
/// detector first (source name `builtin`), then the
/// declared detectors in declaration order, each box
/// carrying its provider's name. Same failure stance as
/// annotation, per provider: one provider's failure omits
/// only its own boxes, and the field appears when at
/// least one configured provider answered.
pub(in crate::server) async fn detect_meta(
    state: &AppState,
    processed: &crate::engine::ProcessedImage,
    ai_gated: bool,
) -> Option<Vec<(String, crate::ai::Detection)>> {
    let mut providers: Vec<(String, Arc<dyn crate::ai::DetectionProvider>)> = Vec::new();
    if let Some(detector) = &state.detect {
        providers.push(("builtin".to_string(), detector.clone()));
    }
    for declared in &state.detectors {
        providers.push((declared.name.clone(), declared.provider.clone()));
    }
    if providers.is_empty() {
        return None;
    }
    // Same permit rule as `annotate_meta`: a request that
    // already holds an AI permit runs its detections under
    // it instead of queueing for a second one (audit R2-29).
    // The permit itself is shared with each provider's
    // blocking task (issue #357), so the slot stays
    // accounted until the inference that is actually
    // running returns, even if the caller has already
    // timed the section out.
    let permit = if ai_gated {
        None
    } else {
        Some(
            tokio::time::timeout(AI_GATE_WAIT, state.ai_gate.clone().acquire_owned())
                .await
                .ok()?
                .ok()?,
        )
    };
    let permit = Arc::new(permit);
    let img = crate::engine::decode_image(&processed.bytes).ok()?;
    let mut aggregated: Vec<(String, crate::ai::Detection)> = Vec::new();
    let mut any_answered = false;
    for (name, provider) in providers {
        let provider_img = img.clone();
        let task_permit = permit.clone();
        match tokio::task::spawn_blocking(move || {
            let _permit = task_permit;
            provider.detect(&provider_img)
        })
        .await
        {
            Ok(Ok(detections)) => {
                match check_finite_scores("a detector", detections.iter().map(|d| d.score)) {
                    Ok(()) => {
                        any_answered = true;
                        aggregated.extend(detections.into_iter().map(|d| (name.clone(), d)));
                    }
                    Err(e) => {
                        eprintln!("detection error from {name}: {e}");
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!("detection error from {name}: {e}");
            }
            Err(e) => {
                eprintln!("detection task failed for {name}: {e}");
            }
        }
    }
    any_answered.then_some(aggregated)
}

/// The moderation field of a meta response that asked
/// for it (`moderation=1`, issue #171): either the
/// provider's report, composed for the wire, or the
/// reason there is none. Like tags and detections, the
/// field is an annotation, so every failure degrades to
/// the unavailable form and the meta request still
/// succeeds.
#[derive(Debug, Clone, PartialEq)]
pub enum ModerationMeta {
    Report(crate::ai::ModerationScores),
    Unavailable(&'static str),
}

/// Compose a moderation report for the response: verdict
/// scores keep the model's output order and round to 4
/// decimals (verdicts are consumed programmatically,
/// and a small probability must not round away);
/// signals sort best first and round to 3 decimals,
/// like tag scores.
pub(in crate::server) fn compose_moderation(
    report: &crate::ai::ModerationScores,
) -> crate::ai::ModerationScores {
    let round_to = |score: f32, places: f32| (score * places).round() / places;
    let mut signals: Vec<crate::ai::TagScore> = report
        .signals
        .iter()
        .map(|s| crate::ai::TagScore {
            label: s.label.clone(),
            score: round_to(s.score, 1000.0),
        })
        .collect();
    // The sort is stable, so equal signals keep their
    // label-file order.
    signals.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    crate::ai::ModerationScores {
        model: report.model.clone(),
        scores: report
            .scores
            .iter()
            .map(|s| crate::ai::TagScore {
                label: s.label.clone(),
                score: round_to(s.score, 10000.0),
            })
            .collect(),
        signals,
    }
}

/// Run moderation for a meta response that asked for it.
/// Same permit rule as `annotate_meta` (a request already
/// holding an AI permit runs under it, audit R2-29),
/// but the outcome is explicit rather than an omitted
/// field: an unconfigured provider, a busy gate, and a
/// failed inference each report unavailable, with a
/// reason, because the client asked for the field by
/// name.
pub(in crate::server) async fn moderate_meta(
    state: &AppState,
    processed: &crate::engine::ProcessedImage,
    ai_gated: bool,
) -> ModerationMeta {
    let Some(provider) = state.moderation.as_ref() else {
        return ModerationMeta::Unavailable("not_configured");
    };
    // The permit moves into the blocking task and drops
    // when the inference returns, as in `annotate_meta`
    // (issue #357).
    let permit = if ai_gated {
        None
    } else {
        match tokio::time::timeout(AI_GATE_WAIT, state.ai_gate.clone().acquire_owned()).await {
            Ok(Ok(permit)) => Some(permit),
            _ => return ModerationMeta::Unavailable("busy"),
        }
    };
    let img = match crate::engine::decode_image(&processed.bytes) {
        Ok(img) => img,
        Err(_) => return ModerationMeta::Unavailable("inference_failed"),
    };
    let provider = provider.clone();
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        provider.moderate(&img)
    })
    .await
    {
        Ok(Ok(report)) => match check_finite_scores(
            "the moderation provider",
            report
                .scores
                .iter()
                .chain(report.signals.iter())
                .map(|s| s.score),
        ) {
            Ok(()) => ModerationMeta::Report(compose_moderation(&report)),
            Err(e) => {
                eprintln!("moderation error: {e}");
                ModerationMeta::Unavailable("inference_failed")
            }
        },
        Ok(Err(e)) => {
            eprintln!("moderation error: {e}");
            ModerationMeta::Unavailable("inference_failed")
        }
        Err(e) => {
            eprintln!("moderation task failed: {e}");
            ModerationMeta::Unavailable("inference_failed")
        }
    }
}

/// A number in a meta response, carried as the exact
/// text the response has always printed for it (issue
/// #346). serde_json renders floats through ryu, which
/// spells a whole value 1.0 where these responses print
/// 1, so a float cannot ride as a plain f32 or f64
/// without changing the body byte for byte. The number
/// is formatted here with the same expressions the
/// hand-built body used (plain Display, {:.1}, {:.3},
/// or the probe rounding below) and carried as a raw
/// JSON value, so the typed structs around it serialize
/// to the identical bytes.
pub(in crate::server) struct MetaNum(Box<serde_json::value::RawValue>);

impl MetaNum {
    fn text(text: String) -> Self {
        // Reachable only from the constructors below,
        // each of which refuses a non-finite value first,
        // so the text is Rust's formatting of a finite
        // value and always spells a valid JSON number.
        Self(
            serde_json::value::RawValue::from_string(text)
                .expect("a formatted finite number is valid JSON"),
        )
    }

    /// Plain Display, the form most float fields use.
    /// `None` for a non-finite value (issue #352): it
    /// has no JSON representation, so the field it would
    /// fill is omitted or its section degrades, at the
    /// caller's choice. Probe readings are in scope, not
    /// just provider output: a container can report a
    /// duration that parses as NaN.
    fn display(value: f32) -> Option<Self> {
        value.is_finite().then(|| Self::text(format!("{value}")))
    }

    /// One decimal place for a 64-bit value: the
    /// verified auto-quality score.
    fn fixed1(value: f64) -> Option<Self> {
        value.is_finite().then(|| Self::text(format!("{value:.1}")))
    }

    /// Three decimal places: segment scores and
    /// coverages.
    fn fixed3(value: f32) -> Option<Self> {
        value.is_finite().then(|| Self::text(format!("{value:.3}")))
    }

    /// Seconds as the video record prints them:
    /// millisecond precision, whole values bare.
    fn seconds(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let rounded = (value * 1000.0).round() / 1000.0;
        if rounded == rounded.trunc() {
            Some(Self::text(format!("{}", rounded as i64)))
        } else {
            Some(Self::text(format!("{rounded}")))
        }
    }

    /// PDF points as the pdf record prints them:
    /// two-decimal precision, whole values bare.
    fn points(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let rounded = (value * 100.0).round() / 100.0;
        if rounded == rounded.trunc() {
            Some(Self::text(format!("{}", rounded as i64)))
        } else {
            Some(Self::text(format!("{rounded}")))
        }
    }
}

impl serde::Serialize for MetaNum {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

/// One scored label in a meta response: a tag, a
/// moderation verdict, or a review signal.
#[derive(serde::Serialize)]
pub(in crate::server) struct ScoreJson {
    label: String,
    score: MetaNum,
}

/// Build the wire form of a score list. A non-finite
/// score has no JSON representation, so the list
/// refuses as a whole (issue #352): the caller
/// degrades the section rather than emit a malformed
/// body. The meta helpers reject non-finite model
/// output before composition, so this is the
/// serializer's own guard for any future caller.
pub(in crate::server) fn score_list(scores: &[crate::ai::TagScore]) -> Option<Vec<ScoreJson>> {
    scores
        .iter()
        .map(|s| {
            Some(ScoreJson {
                label: s.label.clone(),
                score: MetaNum::display(s.score)?,
            })
        })
        .collect()
}

/// The embedding field of a meta response that asked
/// for it (`embeddings=1`, issue #172): either the
/// provider's vector, or the reason there is none.
/// Like moderation, the field is an annotation, so
/// every failure degrades to the unavailable form and
/// the meta request still succeeds.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingMeta {
    Report(crate::ai::ImageEmbedding),
    Unavailable(&'static str),
}

/// Run the embedding provider for a meta response.
/// Same permit rule as `moderate_meta` (a request
/// already holding an AI permit runs under it, audit
/// R2-29), and the outcome is explicit rather than an
/// omitted field: an unconfigured provider, a busy
/// gate, and a failed inference each report
/// unavailable, with a reason, because the client
/// asked for the field by name (directly, or through
/// the aesthetic head that scores the vector).
pub(in crate::server) async fn embed_meta(
    state: &AppState,
    processed: &crate::engine::ProcessedImage,
    ai_gated: bool,
) -> EmbeddingMeta {
    let Some(provider) = state.embeddings.as_ref() else {
        return EmbeddingMeta::Unavailable("not_configured");
    };
    // The permit moves into the blocking task and drops
    // when the inference returns, as in `annotate_meta`
    // (issue #357).
    let permit = if ai_gated {
        None
    } else {
        match tokio::time::timeout(AI_GATE_WAIT, state.ai_gate.clone().acquire_owned()).await {
            Ok(Ok(permit)) => Some(permit),
            _ => return EmbeddingMeta::Unavailable("busy"),
        }
    };
    let img = match crate::engine::decode_image(&processed.bytes) {
        Ok(img) => img,
        Err(_) => return EmbeddingMeta::Unavailable("inference_failed"),
    };
    let provider = provider.clone();
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        provider.embed(&img)
    })
    .await
    {
        Ok(Ok(embedding)) => {
            match check_finite_scores("the embedding provider", embedding.vector.iter().copied()) {
                Ok(()) => EmbeddingMeta::Report(embedding),
                Err(e) => {
                    eprintln!("embedding error: {e}");
                    EmbeddingMeta::Unavailable("inference_failed")
                }
            }
        }
        Ok(Err(e)) => {
            eprintln!("embedding error: {e}");
            EmbeddingMeta::Unavailable("inference_failed")
        }
        Err(e) => {
            eprintln!("embedding task failed: {e}");
            EmbeddingMeta::Unavailable("inference_failed")
        }
    }
}

/// The OCR field of a meta response that asked for it
/// (`ocr=1`, issue #177): either the provider's report,
/// composed for the wire, or the reason there is none.
/// Like moderation, the field is an annotation, so
/// every failure degrades to the unavailable form and
/// the meta request still succeeds.
#[derive(Debug, Clone, PartialEq)]
pub enum OcrMeta {
    Report(crate::ai::OcrReport),
    Unavailable(&'static str),
}

/// Compose an OCR report for the response: line scores
/// round to 3 decimals, like tag scores. Boxes and text
/// pass through untouched.
fn compose_ocr(report: &crate::ai::OcrReport) -> crate::ai::OcrReport {
    crate::ai::OcrReport {
        model: report.model.clone(),
        language: report.language.clone(),
        lines: report
            .lines
            .iter()
            .map(|line| crate::ai::OcrLine {
                text: line.text.clone(),
                score: (line.score * 1000.0).round() / 1000.0,
                x: line.x,
                y: line.y,
                w: line.w,
                h: line.h,
            })
            .collect(),
    }
}

/// Run OCR for a meta response that asked for it. Same
/// permit rule and explicit-outcome stance as
/// `moderate_meta`: an unconfigured provider, a busy
/// gate, and a failed inference each report unavailable,
/// with a reason, because the client asked for the
/// field by name.
pub(in crate::server) async fn ocr_meta(
    state: &AppState,
    processed: &crate::engine::ProcessedImage,
    ai_gated: bool,
    deadline: Instant,
) -> OcrMeta {
    let Some(provider) = state.ocr.as_ref() else {
        return OcrMeta::Unavailable("not_configured");
    };
    // The permit moves into the blocking task and drops
    // when the inference returns, as in `annotate_meta`
    // (issue #357).
    let permit = if ai_gated {
        None
    } else {
        match tokio::time::timeout(AI_GATE_WAIT, state.ai_gate.clone().acquire_owned()).await {
            Ok(Ok(permit)) => Some(permit),
            _ => return OcrMeta::Unavailable("busy"),
        }
    };
    let img = match crate::engine::decode_image(&processed.bytes) {
        Ok(img) => img,
        Err(_) => return OcrMeta::Unavailable("inference_failed"),
    };
    let provider = provider.clone();
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        provider.recognize_deadline(&img, deadline)
    })
    .await
    {
        Ok(Ok(report)) => {
            match check_finite_scores("the OCR provider", report.lines.iter().map(|l| l.score)) {
                Ok(()) => OcrMeta::Report(compose_ocr(&report)),
                Err(e) => {
                    eprintln!("OCR error: {e}");
                    OcrMeta::Unavailable("inference_failed")
                }
            }
        }
        Ok(Err(e)) => {
            eprintln!("OCR error: {e}");
            OcrMeta::Unavailable("inference_failed")
        }
        Err(e) => {
            eprintln!("OCR task failed: {e}");
            OcrMeta::Unavailable("inference_failed")
        }
    }
}

/// The aesthetic field of a meta response (issue
/// #172), present whenever a head is configured: the
/// head's identity always rides along, and the
/// encoder's identity whenever a vector was produced,
/// so a null score stays interpretable.
#[derive(Debug, Clone, PartialEq)]
pub struct AestheticMeta {
    pub head: String,
    pub model: Option<String>,
    pub outcome: AestheticOutcome,
}

/// How an aesthetic composition ended: a score, a
/// dimension mismatch between the head and the
/// encoder (the fail-soft case the issue names), or
/// the embedding run's own unavailability.
#[derive(Debug, Clone, PartialEq)]
pub enum AestheticOutcome {
    Score(f32),
    DimensionMismatch { expected: usize, got: usize },
    Unavailable(&'static str),
}

/// Compose the aesthetic field from the embedding
/// run: the head scores the run's vector, rounded to
/// 3 decimals like the other composed scores. A
/// vector of the wrong dimension is a mismatch, not
/// an error; a run that produced no vector passes its
/// reason through.
pub(in crate::server) fn compose_aesthetic(
    state: &AppState,
    run: Option<&EmbeddingMeta>,
) -> Option<AestheticMeta> {
    let head = state.aesthetic.as_ref()?;
    let (model, outcome) = match run {
        Some(EmbeddingMeta::Report(embedding)) => {
            let outcome = match head.score(&embedding.vector) {
                Some(score) if score.is_finite() => {
                    AestheticOutcome::Score((score * 1000.0).round() / 1000.0)
                }
                // A non-finite head score is bad model
                // output, not a number the response can
                // carry (issue #352); it degrades like
                // an inference failure.
                Some(_) => AestheticOutcome::Unavailable("inference_failed"),
                None => AestheticOutcome::DimensionMismatch {
                    expected: head.expected_dim(),
                    got: embedding.vector.len(),
                },
            };
            (Some(embedding.model.clone()), outcome)
        }
        Some(EmbeddingMeta::Unavailable(reason)) => (None, AestheticOutcome::Unavailable(reason)),
        // The handler runs the embedding whenever a
        // head is configured, so this is unreachable;
        // report it rather than panic.
        None => (None, AestheticOutcome::Unavailable("inference_failed")),
    };
    Some(AestheticMeta {
        head: head.identity().to_string(),
        model,
        outcome,
    })
}

/// Pack an embedding for the wire (issue #172): each
/// f32 narrows to a float16, little-endian, and the
/// bytes are base64-encoded (the crate's encoder, the
/// one placeholder data URIs use). The encoding is
/// named in the response (`f16le-base64`), so a
/// future encoding ships as a new value, never a
/// silent change under this one.
pub(in crate::server) fn pack_embedding(vector: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(vector.len() * 2);
    for value in vector {
        bytes.extend_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
    }
    crate::placeholder::base64_encode(&bytes)
}

/// Narrow an f32 to float16 bits, round to nearest
/// even, with subnormals, overflow to infinity, and
/// NaN kept nonzero and quiet. Embeddings are unit
/// vectors, so the working range is [-1, 1]; the full
/// conversion keeps the packing honest for any input.
pub(in crate::server) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_bits = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp_bits == 0xff {
        // Infinity stays infinity; a NaN keeps a
        // nonzero fraction (quieted) so it cannot
        // collapse into infinity on the wire.
        let narrowed = (frac >> 13) as u16;
        return sign
            | 0x7c00
            | if frac != 0 && narrowed == 0 {
                0x0200
            } else {
                narrowed
            };
    }
    // The f16 exponent field for a normal result.
    let exp = exp_bits - 112;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        // A subnormal f16: shift the significand (with
        // its implicit bit) into subnormal scale,
        // rounding to nearest even on the way.
        let significand = frac | 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut half = significand >> shift;
        let remainder = significand & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && half & 1 == 1) {
            half += 1;
        }
        return sign | half as u16;
    }
    // A normal f16: drop the low 13 fraction bits,
    // rounding to nearest even; a carry out of the
    // fraction correctly increments the exponent
    // through the addition.
    let mut half = sign | ((exp as u16) << 10) | (frac >> 13) as u16;
    let remainder = frac & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    half
}

/// One detection box in a meta response, with the
/// detector that produced it.
#[derive(serde::Serialize)]
struct DetectionJson {
    label: String,
    score: MetaNum,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    detector: String,
}

/// One detection box in an objects-crop record: the
/// same box without the detector attribution.
#[derive(serde::Serialize)]
pub(in crate::server) struct CropObjectJson {
    label: String,
    score: MetaNum,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// The review-only signals inside a moderation report.
#[derive(serde::Serialize)]
pub(in crate::server) struct ModerationSignalsJson {
    review_only: bool,
    scores: Vec<ScoreJson>,
}

/// Serialize as the JSON literal `false`: the
/// unavailable form's `available` field is a constant
/// of the wire shape, not a state a value can be in,
/// so the field carries a unit and prints false.
fn serialize_false<S: serde::Serializer>(_: &(), serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_bool(false)
}

/// The moderation field of a meta response (issue
/// #171): a report carries the model identity, the
/// verdict scores when a classifier ran, and the
/// zero-shot signals (marked review-only: leads, never
/// verdicts) when the CLIP fallback ran. An unavailable
/// moderation is explicit too, since the client opted
/// in by name.
#[derive(serde::Serialize)]
#[serde(untagged)]
pub(in crate::server) enum ModerationJson {
    Report {
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        scores: Option<Vec<ScoreJson>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        signals: Option<ModerationSignalsJson>,
    },
    Unavailable {
        #[serde(serialize_with = "serialize_false")]
        available: (),
        reason: String,
    },
}

/// Compose the moderation field. An empty section is
/// omitted; a section that no longer renders (only
/// possible if composition was skipped by a future
/// caller) degrades the whole field to the unavailable
/// form rather than present a partial one.
pub(in crate::server) fn moderation_json(meta: &ModerationMeta) -> ModerationJson {
    match meta {
        ModerationMeta::Report(report) => {
            let scores = if report.scores.is_empty() {
                Some(None)
            } else {
                score_list(&report.scores).map(Some)
            };
            let signals = if report.signals.is_empty() {
                Some(None)
            } else {
                score_list(&report.signals).map(|scores| {
                    Some(ModerationSignalsJson {
                        review_only: true,
                        scores,
                    })
                })
            };
            match (scores, signals) {
                (Some(scores), Some(signals)) => ModerationJson::Report {
                    model: report.model.clone(),
                    scores,
                    signals,
                },
                _ => ModerationJson::Unavailable {
                    available: (),
                    reason: "inference_failed".to_string(),
                },
            }
        }
        ModerationMeta::Unavailable(reason) => ModerationJson::Unavailable {
            available: (),
            reason: reason.to_string(),
        },
    }
}

/// The embedding slot of a meta response: the packed
/// vector as a string, or an explicit null when the
/// embedding was asked for but not produced.
enum EmbeddingJson {
    Packed(String),
    Absent,
}

impl serde::Serialize for EmbeddingJson {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            EmbeddingJson::Packed(packed) => serializer.serialize_str(packed),
            EmbeddingJson::Absent => serializer.serialize_none(),
        }
    }
}

/// The aesthetic field of a meta response: the head's
/// identity always, the encoder's whenever a vector
/// was produced, and a score or the reason there is
/// none.
#[derive(serde::Serialize)]
struct AestheticJson {
    head: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    score: Option<MetaNum>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    got: Option<usize>,
}

fn aesthetic_json(meta: &AestheticMeta) -> AestheticJson {
    let (score, reason, expected, got) = match &meta.outcome {
        AestheticOutcome::Score(value) if value.is_finite() => {
            (MetaNum::display(*value), None, None, None)
        }
        // A non-finite score has no JSON representation
        // (issue #352): report it the way a failed
        // inference is reported rather than print it.
        AestheticOutcome::Score(_) => (None, Some("inference_failed"), None, None),
        AestheticOutcome::DimensionMismatch { expected, got } => (
            None,
            Some("dimension_mismatch"),
            Some(*expected),
            Some(*got),
        ),
        AestheticOutcome::Unavailable(why) => (None, Some(*why), None, None),
    };
    AestheticJson {
        head: meta.head.clone(),
        model: meta.model.clone(),
        score,
        reason,
        expected,
        got,
    }
}

/// One manifest summary inside the provenance field.
#[derive(serde::Serialize)]
struct ManifestJson {
    state: &'static str,
    signer: Option<String>,
    manifests: usize,
    ingredients: usize,
}

fn manifest_json(summary: &crate::provenance::ManifestSummary) -> ManifestJson {
    ManifestJson {
        state: summary.state,
        signer: summary.signer.clone(),
        manifests: summary.manifests,
        ingredients: summary.ingredients,
    }
}

/// The provenance field of a meta response: C2PA
/// provenance (issue #173), what the source carried
/// and what the output carries, verified inside the
/// process job. Both are facts read from the assets'
/// embedded credentials, the same class as the tags,
/// so a sealed request reports them too (never
/// operation or source text). A valid credential
/// proves origin, not untouched pixels: the states say
/// exactly which of valid, trusted, or invalid each
/// manifest earned.
#[derive(serde::Serialize)]
struct ProvenanceJson {
    signing_configured: bool,
    source: Option<ManifestJson>,
    output: Option<ManifestJson>,
}

/// The video record of a meta response (issue #214):
/// the probed facts of the video source and the
/// resolved timestamp the frame was selected at. The
/// duration is omitted when the container reports none,
/// and a reading that parses as non-finite omits the
/// same way (issue #352). Numbers print at millisecond
/// precision, the same rounding the extractor seeks
/// with.
#[derive(serde::Serialize)]
struct VideoJson {
    width: u32,
    height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    frame_timestamp: Option<MetaNum>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<MetaNum>,
}

/// The pdf record of a meta response (issue #215): the
/// document's page count, the page this response
/// rendered (after canonicalization), and the page size
/// in points. Points print the way the probe reports
/// them: whole values bare, fractions at two decimals.
#[derive(serde::Serialize)]
struct PdfJson {
    pages: u32,
    page: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    page_width: Option<MetaNum>,
    #[serde(skip_serializing_if = "Option::is_none")]
    page_height: Option<MetaNum>,
}

/// The scene facts inside a studio record.
#[derive(serde::Serialize)]
pub(in crate::server) struct SceneJson {
    subject: Option<(u32, u32, u32, u32)>,
    subjects: u32,
    crop: (u32, u32, u32, u32),
    levels: [[u8; 2]; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    saturation: Option<MetaNum>,
}

/// The studio record of a meta response, one variant
/// per mode: a scene run has no background fields to
/// print, and a product run has no scene facts, so the
/// type carries only what the mode can emit.
#[derive(serde::Serialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub(in crate::server) enum StudioJson {
    Product {
        bg: String,
        bg_from: &'static str,
        shadow: String,
        pad: u8,
        steps: Vec<&'static str>,
        fallbacks: Vec<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mask_confidence: Option<MetaNum>,
    },
    Scene {
        pad: u8,
        steps: Vec<&'static str>,
        fallbacks: Vec<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        scene: Option<SceneJson>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mask_confidence: Option<MetaNum>,
    },
}

pub(in crate::server) fn studio_json(record: &crate::studio::StudioRecord) -> StudioJson {
    // A non-finite confidence has no JSON
    // representation (issue #352): the field is
    // omitted rather than printed.
    let mask_confidence = record.confidence.and_then(MetaNum::display);
    match record.mode {
        crate::url::StudioMode::Product => StudioJson::Product {
            bg: match &record.bg {
                Some([r, g, b]) => format!("#{r:02x}{g:02x}{b:02x}"),
                None => "none".to_string(),
            },
            bg_from: record.bg_from,
            shadow: record.shadow.name().to_string(),
            pad: record.pad,
            steps: record.steps.clone(),
            fallbacks: record.fallbacks.clone(),
            mask_confidence,
        },
        crate::url::StudioMode::Scene => StudioJson::Scene {
            pad: record.pad,
            steps: record.steps.clone(),
            fallbacks: record.fallbacks.clone(),
            scene: record.scene.as_ref().map(|scene| SceneJson {
                subject: scene.subject,
                subjects: scene.subjects,
                crop: scene.crop,
                levels: scene.levels,
                // The factor is recorded at millisecond
                // precision: the value itself, not its
                // full f32 expansion. A non-finite factor
                // omits the field (issue #352).
                saturation: MetaNum::display((scene.saturation * 1000.0).round() / 1000.0),
            }),
            mask_confidence,
        },
    }
}

/// The inpaint record of a meta response.
#[derive(serde::Serialize)]
struct InpaintJson {
    regions: Vec<(u32, u32, u32, u32)>,
}

/// One segmented object in a segments record.
#[derive(serde::Serialize)]
struct SegmentObjectJson {
    label: String,
    score: MetaNum,
    #[serde(rename = "box")]
    bounds: (u32, u32, u32, u32),
    tight_box: (u32, u32, u32, u32),
    coverage: MetaNum,
    centroid: (u32, u32),
}

/// The segments record of a meta response. The record
/// carries render-pipeline values, so it gets the same
/// non-finite guarantee as the helper sections (issue
/// #352, scope note on #346): a non-finite score or
/// coverage omits the record rather than print a value
/// JSON cannot represent.
#[derive(serde::Serialize)]
pub(in crate::server) struct SegmentsJson {
    objects: Vec<SegmentObjectJson>,
}

pub(in crate::server) fn segments_json(
    record: &crate::engine::SegmentRecord,
) -> Option<SegmentsJson> {
    // A non-finite score or coverage anywhere refuses
    // the record (issue #352): the number constructors
    // return None and the collect propagates it.
    Some(SegmentsJson {
        objects: record
            .objects
            .iter()
            .map(|o| {
                Some(SegmentObjectJson {
                    label: o.label.clone(),
                    score: MetaNum::fixed3(o.score)?,
                    bounds: o.detection,
                    tight_box: o.tight_box,
                    coverage: MetaNum::fixed3(o.coverage)?,
                    centroid: o.centroid,
                })
            })
            .collect::<Option<Vec<_>>>()?,
    })
}

/// The objects-crop record of a meta response, with
/// the same non-finite guarantee as the segments
/// record above.
#[derive(serde::Serialize)]
pub(in crate::server) struct ObjectsCropJson {
    classes: Vec<String>,
    scale: MetaNum,
    path: &'static str,
    objects: Vec<CropObjectJson>,
    union: Option<(u32, u32, u32, u32)>,
}

pub(in crate::server) fn objects_crop_json(
    record: &crate::engine::ObjectsCropRecord,
) -> Option<ObjectsCropJson> {
    // The same refusal as the segments record: a
    // non-finite scale or score anywhere refuses the
    // whole record (issue #352).
    Some(ObjectsCropJson {
        classes: record.classes.clone(),
        scale: MetaNum::display(record.scale)?,
        path: if record.fallback {
            "fallback"
        } else {
            "objects"
        },
        objects: record
            .objects
            .iter()
            .map(|o| {
                Some(CropObjectJson {
                    label: o.label.clone(),
                    score: MetaNum::display(o.score)?,
                    x: o.x,
                    y: o.y,
                    w: o.w,
                    h: o.h,
                })
            })
            .collect::<Option<Vec<_>>>()?,
        union: record.union,
    })
}

/// One line of recognized text in an OCR field.
#[derive(serde::Serialize)]
pub(in crate::server) struct OcrLineJson {
    text: String,
    score: MetaNum,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// The OCR field of a meta response: the report, or
/// the uniform unavailable object.
#[derive(serde::Serialize)]
#[serde(untagged)]
pub(in crate::server) enum OcrJson {
    Report {
        model: String,
        language: String,
        text: Vec<OcrLineJson>,
        text_content: String,
    },
    Unavailable {
        #[serde(serialize_with = "serialize_false")]
        available: (),
        reason: String,
    },
}

pub(in crate::server) fn ocr_json(meta: &OcrMeta) -> OcrJson {
    match meta {
        OcrMeta::Report(report) => {
            // Last-line guard (issue #352): a report that
            // reached composition with a bad score
            // degrades the same way a failed inference
            // does, never a malformed value. The number
            // constructors refuse non-finite scores and
            // the collect propagates the refusal.
            let lines = report
                .lines
                .iter()
                .map(|l| {
                    Some(OcrLineJson {
                        text: l.text.clone(),
                        score: MetaNum::display(l.score)?,
                        x: l.x,
                        y: l.y,
                        w: l.w,
                        h: l.h,
                    })
                })
                .collect::<Option<Vec<_>>>();
            match lines {
                Some(lines) => OcrJson::Report {
                    model: report.model.clone(),
                    language: report.language.clone(),
                    text: lines,
                    text_content: report
                        .lines
                        .iter()
                        .map(|l| l.text.as_str())
                        .collect::<Vec<_>>()
                        .join("\n"),
                },
                None => OcrJson::Unavailable {
                    available: (),
                    reason: "inference_failed".to_string(),
                },
            }
        }
        OcrMeta::Unavailable(reason) => OcrJson::Unavailable {
            available: (),
            reason: reason.to_string(),
        },
    }
}

/// A meta response body (issue #346): every field the
/// meta surface emits, in emission order, built from
/// typed values and serialized with serde. Field order
/// in this struct is the wire order; optional fields
/// are skipped when absent.
#[derive(serde::Serialize)]
struct MetaBody {
    width: u32,
    height: u32,
    format: &'static str,
    bytes: usize,
    source_width: u32,
    source_height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    sealed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tags: Option<Vec<ScoreJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    alt_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detections: Option<Vec<DetectionJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moderation: Option<ModerationJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding: Option<EmbeddingJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_dim: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_encoding: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_normalized: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aesthetic: Option<AestheticJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ssimulacra2: Option<MetaNum>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provenance: Option<ProvenanceJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    video: Option<VideoJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pdf: Option<PdfJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    studio: Option<StudioJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inpaint: Option<InpaintJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    segments: Option<SegmentsJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    objects_crop: Option<ObjectsCropJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ocr: Option<OcrJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dominant_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    placeholder: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    palette: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thumbhash: Option<String>,
}

/// The meta response: what the pipeline produced, as JSON,
/// without the image bytes.
#[allow(clippy::too_many_arguments)]
pub(in crate::server) fn meta_response(
    processed: &crate::engine::ProcessedImage,
    sealed: bool,
    annotation: Option<&Annotation>,
    detections: Option<&[(String, crate::ai::Detection)]>,
    moderation: Option<&ModerationMeta>,
    embedding: Option<&EmbeddingMeta>,
    aesthetic: Option<&AestheticMeta>,
    ocr: Option<&OcrMeta>,
) -> Response {
    // The sealed audit (issue #255): a request that
    // arrived sealed gets the render facts and every
    // image-derived section (tags, detections,
    // moderation, embedding, aesthetic, OCR, placeholder,
    // ssimulacra2), never a field that reports operation
    // or source text. Below, that means the operation
    // records (video, pdf, studio, inpaint, segments,
    // objects_crop) are skipped: each resolves or echoes
    // a URL argument, the recipe text the token exists
    // to hide. Model identities inside moderation and
    // OCR are deployment facts the plain meta of any URL
    // already publishes, and the base fields above are
    // measured from the render.
    let tags = annotation.and_then(|a| score_list(&a.tags));
    let alt_text = match (annotation, &tags) {
        // The alt text rides whenever the tags render,
        // even when it is empty: the pair is one section.
        (Some(a), Some(_)) => Some(a.alt_text.clone()),
        _ => None,
    };
    let meta = MetaBody {
        width: processed.width,
        height: processed.height,
        format: format_name(processed.format),
        bytes: processed.bytes.len(),
        source_width: processed.source_width,
        source_height: processed.source_height,
        sealed: if sealed { Some(true) } else { None },
        tags,
        alt_text,
        detections: detections.and_then(|pairs| {
            // A non-finite score has no JSON
            // representation (issue #352): the whole
            // section is dropped rather than print a
            // malformed value. The meta helper rejects
            // non-finite boxes before this point, so
            // this is the serializer's own guard: the
            // number constructor refuses and the collect
            // propagates the refusal.
            pairs
                .iter()
                .map(|(detector, d)| {
                    Some(DetectionJson {
                        label: d.label.clone(),
                        score: MetaNum::display(d.score)?,
                        x: d.x,
                        y: d.y,
                        w: d.w,
                        h: d.h,
                        detector: detector.clone(),
                    })
                })
                .collect::<Option<Vec<_>>>()
        }),
        moderation: moderation.map(moderation_json),
        embedding: match embedding {
            Some(EmbeddingMeta::Report(report)) => {
                Some(EmbeddingJson::Packed(pack_embedding(&report.vector)))
            }
            Some(EmbeddingMeta::Unavailable(_)) => Some(EmbeddingJson::Absent),
            None => None,
        },
        embedding_model: match embedding {
            Some(EmbeddingMeta::Report(report)) => Some(report.model.clone()),
            _ => None,
        },
        embedding_dim: match embedding {
            Some(EmbeddingMeta::Report(report)) => Some(report.vector.len()),
            _ => None,
        },
        embedding_encoding: match embedding {
            Some(EmbeddingMeta::Report(_)) => Some("f16le-base64"),
            _ => None,
        },
        embedding_normalized: match embedding {
            Some(EmbeddingMeta::Report(_)) => Some(true),
            _ => None,
        },
        embedding_reason: match embedding {
            Some(EmbeddingMeta::Unavailable(reason)) => Some(*reason),
            _ => None,
        },
        aesthetic: aesthetic.map(aesthetic_json),
        // The verified auto-quality score, on the meta
        // of the response it gated (issue #41). One
        // decimal: the score is a ranking signal, not a
        // measurement. A non-finite score cannot occur
        // (ssimulacra2 returns a bounded dissimilarity),
        // but the field guards anyway rather than print
        // a malformed value.
        ssimulacra2: processed.ssimulacra2.and_then(MetaNum::fixed1),
        provenance: processed.provenance.as_ref().map(|facts| ProvenanceJson {
            signing_configured: facts.signing_configured,
            source: facts.source.as_ref().map(manifest_json),
            output: facts.output.as_ref().map(manifest_json),
        }),
        video: if sealed {
            None
        } else {
            processed.video.as_ref().map(|v| VideoJson {
                width: v.width,
                height: v.height,
                frame_timestamp: MetaNum::seconds(v.frame_seconds),
                duration: v.duration_seconds.and_then(MetaNum::seconds),
            })
        },
        pdf: if sealed {
            None
        } else {
            processed.pdf.as_ref().map(|pdf| PdfJson {
                pages: pdf.pages,
                page: pdf.page,
                page_width: MetaNum::points(pdf.page_width),
                page_height: MetaNum::points(pdf.page_height),
            })
        },
        studio: if sealed {
            None
        } else {
            processed.studio.as_ref().map(studio_json)
        },
        inpaint: if sealed {
            None
        } else {
            processed.inpaint.as_ref().map(|record| InpaintJson {
                regions: record.regions.clone(),
            })
        },
        segments: if sealed {
            None
        } else {
            processed.segments.as_ref().and_then(segments_json)
        },
        objects_crop: if sealed {
            None
        } else {
            processed.objects_crop.as_ref().and_then(objects_crop_json)
        },
        ocr: ocr.map(ocr_json),
        dominant_color: processed
            .placeholder
            .as_ref()
            .map(|ph| ph.dominant_color.clone()),
        placeholder: processed.placeholder.as_ref().map(|ph| ph.data_uri.clone()),
        palette: processed.placeholder.as_ref().map(|ph| ph.palette.clone()),
        thumbhash: processed
            .placeholder
            .as_ref()
            .map(|ph| ph.thumbhash.clone()),
    };
    let body = serde_json::to_string(&meta).unwrap_or_default();
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
    meta_cache_headers((StatusCode::OK, headers, body).into_response())
}

/// The caching headers of an answer to a meta request.
/// Meta is a second representation of the same URL (audit
/// PL-12): never publicly cacheable, so an intermediary can
/// never pin it in place of the image, or the reverse. The
/// `meta_disabled` refusal (issue #453) answers that same
/// representation and carries the same headers.
pub(in crate::server) fn meta_cache_headers(mut response: Response) -> Response {
    let headers = response.headers_mut();
    headers.insert(header::CACHE_CONTROL, "private, no-store".parse().unwrap());
    headers.insert(header::VARY, "Accept, DPR, Sec-CH-DPR, Sec-CH-Width, Sec-CH-Viewport-Width, Width, Viewport-Width, Save-Data".parse().unwrap());
    response
}
