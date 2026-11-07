//! The HTTP server: parse, verify, load, process, respond.
//!
//! `GET /<signature>/<operations...>/<image-url>` verifies the
//! signature, loads the source image, runs the engine, and responds
//! with the optimized bytes. When the URL names no format, the
//! response format is negotiated from the request's `Accept`
//! header. `GET /health` is the health check and
//! `GET /metrics` serves Prometheus metrics. Every image request
//! leaves one structured JSON log line and one metrics update.
//! When an upload destination is configured, a signed
//! `PUT /upload/<signature>/<name>` stores originals in it
//! (issue #168; see [`crate::upload`]).
//!
//! `GET /explain/<signature>/<operations...>/<image-url>`
//! (issue #196) verifies the remainder exactly as an image
//! request does and answers with a report on the URL
//! instead of the image: signature and expiry verdict,
//! probed source facts, every operation glossed, the
//! output forecast, and the canonical cache form. It
//! probes the source through the loader (headers only, no
//! decode, no gates) and writes nothing anywhere.

use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::{
    Router,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};

use crate::cache::{Cache, CachedImage, etag_for_bytes};
use crate::engine::{Engine, EngineError};
use crate::loader::{LoadError, Loader};
use crate::observe::{FailStage, Metrics, RequestRecord};
use crate::url::{ImageFormat, parse_operations_with_expiry, parse_signature};

/// One declared `boxes` detector (issue #166): the name
/// its declaration carries in `CRABCROP_DETECTORS_FILE`
/// and its provider. Kept in declaration order on the
/// state: meta aggregation and class resolution both walk
/// the built-in detector first, then these.
#[derive(Clone)]
pub struct NamedDetector {
    pub name: String,
    pub provider: Arc<dyn crate::ai::DetectionProvider>,
}

impl std::fmt::Debug for NamedDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedDetector")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

mod process;
pub(in crate::server) use process::{
    AI_GATE_WAIT, ANIM_GATE_WAIT, Flight, JobPermits, Lane, MetaFlight, PROCESS_GATE_WAIT,
    ProcessFailure, ProcessJob, SharedMeta, VIDEO_GATE_WAIT, run_process, wants_animation_gate,
    wants_slow_lane,
};
pub use process::{DEFAULT_SLOW_LANE_PERMITS, Gate, Inflight};
#[cfg(test)]
pub(in crate::server) use process::{SLOW_LANE_WAIT, apply_provenance};

/// Shared state for the handlers.
#[derive(Debug, Clone)]
pub struct AppState {
    /// HMAC secret, from `CRABCROP_SECRET`.
    /// The signing keyring (issue #143): the current key
    /// plus, during a rotation window, the previous one.
    pub keyring: crate::url::Keyring,
    /// Whether `unsafe` URLs are honored, from `CRABCROP_ALLOW_UNSAFE`.
    pub allow_unsafe: bool,
    pub loader: Loader,
    pub engine: Engine,
    /// Per-request processing deadline (issue #260), from
    /// `CRABCROP_PROCESS_TIMEOUT_SECS`: the engine checks
    /// it between stages and inside chunked encode loops
    /// and aborts chained work with a 504, and the runner
    /// stops waiting at the deadline plus `DEADLINE_GRACE`
    /// even when a single opaque stage cannot be
    /// interrupted mid-call (issue #322); the job's gates
    /// stay with it until it finishes (issue #357).
    pub process_timeout: std::time::Duration,
    /// Request metrics, shared behind an `Arc` so clones agree.
    pub metrics: Arc<Metrics>,
    /// The bearer token `/metrics` requires (issue #270),
    /// from `CRABCROP_METRICS_TOKEN`. `None` closes the
    /// endpoint: it answers 403 rather than serving the
    /// exposition unauthenticated.
    pub metrics_token: Option<String>,
    /// Processed-response cache, shared like the metrics.
    pub cache: Arc<Cache>,
    /// Background-removal provider, preloaded at boot when
    /// `CRABCROP_REMOVEBG_MODEL` is configured. `None` means the
    /// feature is off; `removebg` requests then fail cleanly.
    pub removebg: Option<Arc<dyn crate::ai::MaskProvider>>,
    /// Upscale provider, preloaded at boot when
    /// `CRABCROP_UPSCALE_MODEL` is configured. `None` means the
    /// feature is off; `upscale` requests then fail cleanly.
    pub upscale: Option<Arc<dyn crate::ai::UpscaleProvider>>,
    /// Inpainting provider, preloaded at boot when
    /// `CRABCROP_INPAINT_MODEL` is configured. `None` means
    /// the feature is off; `remove_region` requests then
    /// fail cleanly.
    pub inpaint: Option<Arc<dyn crate::ai::InpaintProvider>>,
    /// Segment provider, preloaded at boot when the
    /// `CRABCROP_SEGMENT_*_MODEL` pair is configured.
    /// `None` means class operations use detection
    /// rectangles; configured, they follow silhouettes.
    pub segment: Option<Arc<dyn crate::ai::SegmentProvider>>,
    /// Face provider for smart crops, preloaded at boot when
    /// `CRABCROP_FACE_MODEL` is configured. `None` means smart
    /// crops use the saliency estimate.
    pub face: Option<Arc<dyn crate::ai::FaceProvider>>,
    /// CLIP tag provider for meta annotations, preloaded at
    /// boot when the `CRABCROP_CLIP_*` set is configured.
    /// `None` means meta responses carry no tags or alt text.
    pub tagger: Option<Arc<dyn crate::ai::TagProvider>>,
    /// Tag threshold for meta annotations, from
    /// `CRABCROP_CLIP_THRESHOLD` (default 0.05): labels whose
    /// composed probability falls below it are omitted from
    /// meta `tags`. The composition itself lives in
    /// `compose_annotation` (issue #152).
    pub tag_threshold: f32,
    /// Moderation provider for meta responses that opt in
    /// with `moderation=1` (issue #171), preloaded at boot
    /// when `CRABCROP_MODERATION_MODEL` or the labels
    /// fallback is configured. `None` means the opt-in
    /// reports moderation as unavailable. Verdicts are
    /// annotations only: nothing here alters serving.
    pub moderation: Option<Arc<dyn crate::ai::ModerationProvider>>,
    /// Embedding provider for meta responses that opt
    /// in with `embeddings=1` (issue #172), built from
    /// the CLIP tagger at boot. `None` means the opt-in
    /// reports the embedding as unavailable.
    pub embeddings: Option<Arc<dyn crate::ai::EmbeddingProvider>>,
    /// Aesthetic head for meta responses (issue #172),
    /// loaded at boot when `CRABCROP_AESTHETIC_HEAD` is
    /// configured. `None` means meta carries no
    /// aesthetic field. The score is descriptive only:
    /// nothing here alters serving either.
    pub aesthetic: Option<Arc<crate::ai::AestheticHead>>,
    /// OCR provider for meta responses that opt in with
    /// `ocr=1` (issue #177), preloaded at boot when the
    /// `CRABCROP_OCR_*` set is configured. `None` means
    /// the opt-in reports OCR as unavailable. The text
    /// only annotates meta; nothing here alters serving
    /// either: the image a meta request describes is the
    /// same entry an image request returns.
    pub ocr: Option<Arc<dyn crate::ai::OcrProvider>>,
    /// Detection provider for class redaction and meta
    /// detections, preloaded at boot when
    /// `CRABCROP_DETECT_MODEL` is configured. `None` means
    /// class redaction fails cleanly and meta carries no
    /// detections.
    pub detect: Option<Arc<dyn crate::ai::DetectionProvider>>,
    /// Declared `boxes` detectors from
    /// `CRABCROP_DETECTORS_FILE` (issue #166), in
    /// declaration order: class redaction resolves across
    /// the built-in detector and these, and meta detections
    /// aggregate all of them, each box attributed by name.
    pub detectors: Vec<NamedDetector>,
    /// Declared focal provider for smart crops (issue
    /// #166): the first `focal` declaration in
    /// `CRABCROP_DETECTORS_FILE`. Later focal declarations
    /// are probed at boot and stand by. `None` means smart
    /// crops fall back to faces, then saliency.
    pub focal: Option<Arc<dyn crate::ai::FocalProvider>>,
    /// Text renderer, loaded at boot when `CRABCROP_FONT_PATH`
    /// is configured. `None` means `text(...)` requests fail
    /// cleanly.
    pub font: Option<Arc<dyn crate::text::TextRenderer>>,
    /// Fallback image for source failures, when configured.
    pub fallback: Option<Fallback>,
    /// Disk-backed result storage, the persistent tier under
    /// the memory cache, when configured.
    pub store: Option<Arc<crate::storage::DiskStore>>,
    /// Shared result storage (issue #142): the multi-replica
    /// tier behind the local ones, consulted on a local miss
    /// and written through in the background, when configured.
    pub shared_store: Option<Arc<crate::s3store::S3Store>>,
    /// Upload destination (issue #168): where signed
    /// `PUT /upload/...` originals land. `None` means the
    /// endpoint does not exist on this server (the route is
    /// never registered, so a PUT is a 404, not a refusal).
    pub upload: Option<Arc<crate::upload::UploadStore>>,
    /// Whether meta mode is enabled (issue #390):
    /// `CRABCROP_META_MODE`, on by default. A meta request
    /// bypasses every cache tier and runs a fresh render
    /// plus its annotation inferences on every call, so a
    /// deployment that does not use it can close the most
    /// expensive request shape a signed URL can trigger.
    pub meta_enabled: bool,
    /// Bounds concurrent AI inference so it cannot starve the
    /// transform pipeline.
    pub ai_gate: Arc<tokio::sync::Semaphore>,
    /// Bounds concurrent engine runs (audit E-5): cache misses
    /// process on blocking threads under this gate, sized to
    /// the core count at boot. This is the stills lane
    /// (issue #324): ordinary still renders and upload
    /// validation take their permits here, and long-running
    /// work never does, so a saturated slow lane cannot
    /// shrink still capacity at all.
    pub process_gate: Arc<Gate>,
    /// The slow lane (issue #324): a separately bounded
    /// gate for long-running engine work, animated WebP
    /// encodes and the external-tool work behind the video
    /// gate (video encodes, frame extraction, page
    /// renders). Sized by `CRABCROP_SLOW_LANE_PERMITS`
    /// (default 1, sane for a 2 vCPU host: one long encode
    /// leaves the stills lane and a core for stills).
    /// Overflow sheds with the same 503 semantics as the
    /// other gates; it never spills onto the process gate.
    pub slow_gate: Arc<Gate>,
    /// SSIMULACRA2 target for verified `quality(auto)` encodes,
    /// from `CRABCROP_SSIM_TARGET`.
    pub ssim_target: f64,
    pub avif_speed: u8,
    pub jxl_speed: jixel::Speed,
    /// Quantization step for width client hints
    /// (`CRABCROP_WIDTH_STEP`).
    pub width_step: u32,
    /// Metadata policy (`CRABCROP_METADATA=preserve`):
    /// copy source EXIF/XMP into JPEG and PNG outputs.
    pub preserve_metadata: bool,
    /// Operator brand color for `studio(...)` composites
    /// (issue #197), from `CRABCROP_BRAND_COLOR`. `None`
    /// means an unnamed or `bg:brand` backdrop resolves to
    /// white.
    pub brand_color: Option<[u8; 3]>,
    /// Named operation presets from `CRABCROP_PRESETS_FILE`:
    /// name to operation string, validated at boot.
    pub presets: std::collections::HashMap<String, String>,
    /// Presets-only mode, from `CRABCROP_ONLY_PRESETS`: reject
    /// free-form operations in requests.
    pub only_presets: bool,
    /// Named source roots from `CRABCROP_SOURCE_ROOTS_FILE`
    /// (issue #248): name to base URL, validated at boot. A
    /// request whose source is `root:<name>/<path>` is
    /// fetched from the base URL with the path joined under
    /// it; expansion runs after verification, before cache
    /// keying, so the short and full spellings are one
    /// render. The bases are operator configuration: no
    /// response ever prints one.
    pub source_roots: std::collections::HashMap<String, String>,
    /// Server-wide default operations (issue #145), parsed at
    /// boot from `CRABCROP_DEFAULT_OPS` and prepended to every
    /// request's operations before analysis, processing, and
    /// cache keying.
    pub default_ops: Vec<crate::url::Operation>,
    /// External video encoder (issue #162), validated at
    /// boot when `CRABCROP_VIDEO_ENCODER` is configured.
    /// `None` means video output does not exist on this
    /// server; `format(mp4)` and `format(webm)` then fail
    /// with the specific not-configured error.
    pub video_encoder: Option<Arc<crate::video::VideoEncoder>>,
    /// Frame extractor for `frame:` segments on video
    /// sources (issue #214), built at boot when
    /// `CRABCROP_VIDEO_DECODER` is configured. `None` means
    /// frame selection does not exist on this server; a
    /// `frame:` request then fails with the specific
    /// not-configured error. Extraction shares the video
    /// gate below: it is the same class of subprocess
    /// work as an encode.
    pub frame_extractor: Option<Arc<crate::frames::VideoFrameExtractor>>,
    /// PDF renderer for `page:` segments on PDF sources
    /// (issue #215), built at boot when
    /// `CRABCROP_PDF_RENDERER` is configured. `None` means
    /// page selection does not exist on this server; a
    /// `page:` request then fails with the specific
    /// not-configured error. Rendering shares the video
    /// gate below: it is the same class of subprocess
    /// work as frame extraction.
    pub pdf_renderer: Option<Arc<crate::pdf::PdfRenderer>>,
    /// C2PA output signer (issue #173), built and
    /// self-validated at boot when `CRABCROP_C2PA_CERT`
    /// and `CRABCROP_C2PA_KEY` are configured. `None`
    /// means outputs are served unsigned; meta responses
    /// still report the provenance a source carries.
    pub provenance_signer: Option<Arc<crate::provenance::ProvenanceSigner>>,
    /// Bounds concurrent video encodes: the subprocess is
    /// the server's most expensive and least predictable
    /// work, so it gets its own small gate (2 permits, like
    /// the AI gate) on top of the slow lane (issue #324).
    pub video_gate: Arc<tokio::sync::Semaphore>,
    /// Bounds concurrent animated WebP encodes to one
    /// (issue #231): the animation encode is among the
    /// slowest work the engine does, minutes for a modest
    /// file, so it gets a single-permit gate of its own on
    /// top of the slow lane (issue #324), which is what
    /// keeps it from holding stills permits while it runs.
    pub anim_gate: Arc<tokio::sync::Semaphore>,
    /// In-flight miss coalescing (audit R2-09): concurrent
    /// misses for one cache key share one fetch and one
    /// encode instead of stampeding the origin and the
    /// process gate.
    pub inflight: Arc<Inflight>,
    /// Purge generations and tombstones (issue #350):
    /// every purge records its target here before it
    /// touches a tier, and every tier write carries the
    /// generation it started under, so a render or
    /// promotion that began before a purge cannot restore
    /// what the purge removed when its writes land.
    pub purge_guard: Arc<crate::cache::PurgeGuard>,
    /// Fingerprint of the output-shaping configuration
    /// (audit R2-08), computed at boot in `main`: embedded
    /// in every cache key, so changing the metadata
    /// policy, quality tuning, width step, model files,
    /// font, or video encoder starts a fresh cache epoch
    /// instead of silently serving entries rendered under
    /// the old configuration from the persistent tiers.
    pub config_fingerprint: String,
}

/// Build the router.
pub fn app(state: AppState) -> Router {
    let mut router = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/explain/{*path}", get(explain_handle))
        .route("/purge/{*path}", axum::routing::post(purge_handle))
        .route(
            "/purge-group/{*path}",
            axum::routing::post(purge_group_handle),
        );
    // The upload route exists only when a destination is
    // configured (issue #168): an unconfigured server has
    // no PUT surface at all, so the request is a plain 404.
    if state.upload.is_some() {
        router = router.route("/upload/{*path}", axum::routing::put(upload_handle));
    }
    router
        .route("/{*path}", get(handle))
        // A PUT that matched no upload route is a 404, not
        // the 405 the GET catch-all would otherwise produce
        // for it: without an upload destination there is no
        // PUT surface on this server at all (issue #168).
        .route("/{*path}", axum::routing::put(put_not_found))
        .with_state(state)
        .layer(axum::middleware::from_fn(security_headers))
}

/// The PUT fallback: every PUT that is not an upload to a
/// configured destination is simply not found.
async fn put_not_found(headers: HeaderMap) -> Response {
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    error_response(
        StatusCode::NOT_FOUND,
        "not_found",
        "not found".to_string(),
        accept.as_deref(),
    )
}

/// Headers applied to every response (audit PL-5, PL-6):
/// `nosniff` everywhere, and `no-store` on the system
/// endpoints, whose contents are operational data, not
/// cacheable content.
async fn security_headers(req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let system = matches!(req.uri().path(), "/health" | "/metrics");
    let mut response = next.run(req).await;
    let headers = response.headers_mut();
    headers.insert(
        "x-content-type-options",
        axum::http::HeaderValue::from_static("nosniff"),
    );
    if system {
        headers.insert(
            header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-store"),
        );
    }
    response
}

async fn health() -> &'static str {
    "ok"
}

/// Constant-time equality for a bearer token, so the
/// comparison leaks nothing about it through timing.
fn token_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

async fn metrics(State(state): State<AppState>, headers: HeaderMap) -> Response {
    // The exposition is operational data (issue #270,
    // the PL-5 residual): it serves only to a bearer of
    // the configured token, and an unconfigured server
    // keeps it closed rather than public.
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let Some(token) = &state.metrics_token else {
        return error_response(
            StatusCode::FORBIDDEN,
            "metrics_disabled",
            "metrics are disabled: set CRABCROP_METRICS_TOKEN to enable them".to_string(),
            accept.as_deref(),
        );
    };
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    if !presented.is_some_and(|p| token_eq(p.as_bytes(), token.as_bytes())) {
        return error_response(
            StatusCode::UNAUTHORIZED,
            "metrics_unauthorized",
            "metrics require a bearer token".to_string(),
            accept.as_deref(),
        );
    }
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        format!(
            "{}{}{}",
            state.metrics.render(),
            lane_metrics(&state),
            tier_metrics(&state)
        ),
    )
        .into_response()
}

/// Per-lane occupancy for `/metrics` (issue #324): how
/// many permits of each processing lane are in use,
/// against the lane's capacity. Read live from the gates
/// at scrape time, so a lane saturated by long encodes
/// shows itself here while the stills lane stays free.
fn lane_metrics(state: &AppState) -> String {
    let mut out = String::new();
    out.push_str("# HELP crabcrop_lane_in_use Processing lane permits currently in use.\n");
    out.push_str("# TYPE crabcrop_lane_in_use gauge\n");
    out.push_str(&format!(
        "crabcrop_lane_in_use{{lane=\"process\"}} {}\n",
        state.process_gate.in_use()
    ));
    out.push_str(&format!(
        "crabcrop_lane_in_use{{lane=\"slow\"}} {}\n",
        state.slow_gate.in_use()
    ));
    out.push_str("# HELP crabcrop_lane_capacity Processing lane permit capacity.\n");
    out.push_str("# TYPE crabcrop_lane_capacity gauge\n");
    out.push_str(&format!(
        "crabcrop_lane_capacity{{lane=\"process\"}} {}\n",
        state.process_gate.capacity()
    ));
    out.push_str(&format!(
        "crabcrop_lane_capacity{{lane=\"slow\"}} {}\n",
        state.slow_gate.capacity()
    ));
    out
}

/// Cache tier failures for `/metrics` (issue #367): the
/// stores count their own failures as they log them; the
/// exposition reads the counters live at scrape time,
/// one series per configured tier.
fn tier_metrics(state: &AppState) -> String {
    let mut out = String::new();
    out.push_str("# HELP crabcrop_cache_tier_errors_total Cache tier operations that failed.\n");
    out.push_str("# TYPE crabcrop_cache_tier_errors_total counter\n");
    if let Some(store) = &state.store {
        out.push_str(&format!(
            "crabcrop_cache_tier_errors_total{{tier=\"disk\"}} {}\n",
            store.errors()
        ));
    }
    if let Some(store) = &state.shared_store {
        out.push_str(&format!(
            "crabcrop_cache_tier_errors_total{{tier=\"shared\"}} {}\n",
            store.errors()
        ));
    }
    out
}

/// The `DPR` client hint, when it names one of the settled
/// factors. Anything else (absent, unparsable, out of set) is
/// no hint: hints are advisory, never an error.
fn dpr_hint(headers: &HeaderMap) -> Option<f32> {
    let raw = headers
        .get("Sec-CH-DPR")
        .or_else(|| headers.get("DPR"))
        .and_then(|v| v.to_str().ok())?;
    raw.trim()
        .parse::<f32>()
        .ok()
        .filter(|f| [1.0, 1.5, 2.0, 3.0].contains(f))
}

/// The width client hint: the display width the client
/// wants, from the most specific header present
/// (`Sec-CH-Width`, then the legacy `Width`, then the
/// viewport widths). Advisory like every hint: an absent,
/// unparsable, or out-of-range value is no hint, never an
/// error.
fn width_hint(headers: &HeaderMap) -> Option<u32> {
    for name in [
        "Sec-CH-Width",
        "Width",
        "Sec-CH-Viewport-Width",
        "Viewport-Width",
    ] {
        let Some(raw) = headers.get(name).and_then(|v| v.to_str().ok()) else {
            continue;
        };
        if let Ok(w) = raw.trim().parse::<u32>()
            && (1..=crate::url::MAX_DIMENSION).contains(&w)
        {
            return Some(w);
        }
    }
    None
}

/// Round a hinted width UP to the next multiple of the
/// quantization step, so a client never gets less than it
/// asked for and the cache sees bounded cardinality. The
/// result is clamped to the parser's dimension cap: the
/// synthesized op bypasses parsing, so without the clamp
/// a near-cap hint at a coarse step would mint a size the
/// URL grammar itself would reject.
fn quantize_width(width: u32, step: u32) -> u32 {
    let step = step.max(1);
    (width.div_ceil(step) * step).min(crate::url::MAX_DIMENSION)
}

/// Whether the client sent `Save-Data: on`.
fn save_data_hint(headers: &HeaderMap) -> bool {
    headers
        .get("Save-Data")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("on"))
}

/// The operator-configured fallback image: processed and
/// served in place of a source that cannot be loaded or
/// decoded, so a dead origin degrades to a placeholder instead
/// of a broken image.
#[derive(Debug, Clone)]
pub enum Fallback {
    /// A local file, read once at boot.
    Bytes(Arc<Vec<u8>>),
    /// A URL, fetched per fallback event through the loader
    /// (and its bounds and source policy).
    Url(String),
}

/// The fallback's bytes, when one is configured and reachable.
/// A URL fallback that itself fails to load yields `None`, and
/// the caller answers with the original source error.
async fn fallback_source_bytes(state: &AppState) -> Option<Vec<u8>> {
    match &state.fallback {
        Some(Fallback::Bytes(bytes)) => Some(bytes.as_ref().clone()),
        Some(Fallback::Url(url)) => state.loader.load(url).await.ok(),
        None => None,
    }
}

/// `GET /explain/<signature>/<operations...>/<image-url>`:
/// Whether a probe head settles the explain report's
/// source facts: the probe knows the kind, and it either
/// has the dimensions or the kind is one whose dimensions
/// never live in headers (video, PDF). Anything else means
/// the identifying header may still be on its way.
fn probe_settled(head: &[u8]) -> bool {
    let probe = crate::engine::probe_source(head);
    probe.kind != "unknown" && (probe.dimensions.is_some() || matches!(probe.kind, "video" | "pdf"))
}

/// the self-explaining URL report (issue #196; the report
/// itself is built in [`crate::explain`]).
///
/// The checks mirror the image path exactly and run in its
/// order: signature grammar, signature verification, the
/// reserved-segment guard, preset expansion, operation
/// parsing, expiry. Only then is the source touched, once,
/// through the loader with all of its guards, and the
/// facts come from header probing (`crate::engine::
/// probe_source`): no decode, no detection, no process,
/// AI, or video gate, and nothing written to any cache
/// tier. A bad signature or an expired URL therefore costs
/// zero source fetches, and answers with the image path's
/// own status and body.
async fn explain_handle(
    State(state): State<AppState>,
    uri: axum::http::Uri,
    headers: HeaderMap,
) -> Response {
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let Some(tail) = uri.path().strip_prefix("/explain/") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /explain/<signature>/<target>".to_string(),
            accept.as_deref(),
        );
    };
    let full = format!("/{tail}");
    // A sealed spelling is refused wholesale (issue #255):
    // the server never unseals as a service, because an
    // endpoint that opens tokens would be an unsealing
    // oracle for anyone who lifts one off a public page.
    // Operators unseal locally (the CLI's `unseal` verb,
    // under their own secret) and then explain the plain
    // form. The refusal is the uniform one and no source
    // fetch happens for it.
    let first_segment = tail.split('/').next().unwrap_or("");
    if crate::url::looks_sealed(first_segment) {
        return error_response(
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature".to_string(),
            accept.as_deref(),
        );
    }
    let (signature, key_id, rest) = match parse_signature(&full) {
        Ok(v) => v,
        Err(e) => return parse_failure(&e, accept.as_deref()),
    };
    if !state
        .keyring
        .verify(key_id.as_deref(), &signature, rest, state.allow_unsafe)
    {
        return error_response(
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature".to_string(),
            accept.as_deref(),
        );
    }
    // The same domain separation as the image path: a
    // remainder may not begin with a reserved segment.
    if matches!(
        rest.split('/').next(),
        Some("purge" | "purge-group" | "upload")
    ) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "reserved_segment",
            "an image URL may not begin with a purge or upload segment".to_string(),
            accept.as_deref(),
        );
    }
    let groups = crate::url::preset_names(rest);
    // Expansion always runs, even with no presets loaded:
    // a `preset:` segment is a grammar error
    // (`unknown_preset`) on every server, never source text.
    let expanded;
    let rest = match crate::url::expand_presets(rest, &state.presets, state.only_presets) {
        Ok(text) => {
            expanded = text;
            &expanded
        }
        Err(e) => return parse_failure(&e, accept.as_deref()),
    };
    let (ops, source_url, expiry, _) = match parse_operations_with_expiry(rest) {
        Ok(v) => v,
        Err(e) => return parse_failure(&e, accept.as_deref()),
    };
    // A root source expands for the probe fetch only: the
    // report below carries the source exactly as written
    // and the root's name, never the base URL the reference
    // resolves to (operator configuration, issue #248).
    let (fetch_url, root_ref) =
        match crate::url::expand_source_root(source_url, &state.source_roots) {
            Ok(v) => v,
            Err(e) => return parse_failure(&e, accept.as_deref()),
        };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Some(exp) = expiry
        && now >= exp
    {
        return error_response(
            StatusCode::FORBIDDEN,
            "signature_expired",
            "signature expired".to_string(),
            accept.as_deref(),
        );
    }

    // The single bounded source fetch, through the loader
    // and its policy. Watermark and overlay sources are
    // operations' inputs, not the explained URL's source,
    // and are never fetched: no fan-out. A `frame:`
    // segment makes the source a video, so the loader may
    // accept a `video/*` Content-Type for it (issue #214);
    // a `page:` segment does the same for
    // `application/pdf` (issue #215). Explain never runs
    // either tool: the fetch is admitted, the probe stays
    // the raw-bytes probe.
    let admission = crate::loader::MediaAdmission {
        video: ops
            .iter()
            .any(|op| matches!(op, crate::url::Operation::Frame(_))),
        pdf: ops
            .iter()
            .any(|op| matches!(op, crate::url::Operation::Page(_))),
    };
    // The report reads the source's headers only, so the
    // fetch is a probe load: it stops as soon as the head
    // identifies the source instead of downloading the
    // whole body (which also keeps this route off the
    // result cache's coalescing path by construction: it
    // never held a whole body to share).
    let fetched = match state
        .loader
        .load_probe(&fetch_url, admission, &probe_settled)
        .await
    {
        Ok(fetched) => fetched,
        Err(e) => return explain_load_error(e, accept.as_deref()),
    };
    let probe = crate::engine::probe_source(&fetched.head);

    // Each referenced preset with the operations its
    // definition expands to. Definitions were validated at
    // boot; one that no longer parses (it cannot) is listed
    // by name alone rather than failing the report.
    let mut presets = Vec::new();
    for name in &groups {
        let entry_ops = state
            .presets
            .get(name)
            .and_then(|definition| crate::url::parse_default_ops(definition, &state.presets).ok())
            .unwrap_or_default();
        presets.push(crate::explain::preset_entry(name, &entry_ops));
    }

    // The canonical form is the cache key's path form,
    // defaults merged exactly as the image path merges
    // them for its key. The key's config fingerprint
    // suffix is operator state and stays unreported. For a
    // root source the form is built from the source as
    // written: the key itself is keyed on the expansion,
    // but printing that here would publish the root's
    // base URL into the report.
    let canonical = crate::url::canonical_form(
        &crate::url::merge_default_ops(&state.default_ops, ops.clone()),
        source_url,
    );
    let report = crate::explain::Report::new(crate::explain::ReportInput {
        signed: !matches!(signature, crate::url::Signature::Unsafe),
        key_id,
        expiry,
        now,
        source_url: source_url.to_string(),
        source_root: root_ref.map(|r| r.name),
        probe,
        source_bytes: fetched.total_bytes.map(|n| n as usize),
        ops,
        presets,
        canonical,
        cache_group: groups.into_iter().next(),
        image_path: full,
    });

    let (content_type, body) = if crate::explain::wants_json(accept.as_deref()) {
        (
            "application/json",
            serde_json::to_string(&report.to_json()).unwrap_or_default(),
        )
    } else {
        ("text/html; charset=utf-8", report.to_html())
    };
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "no-store"),
            (header::CONTENT_SECURITY_POLICY, "default-src 'none'"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        ],
        body,
    )
        .into_response()
}

/// A probe failure on the explain route: the image path's
/// statuses, with generic bodies throughout. The detailed
/// cause goes to the server log only: a policy refusal can
/// name a resolved address or a bucket, which are the
/// operator's facts, not the caller's (issue #196,
/// requirement 3).
fn explain_load_error(e: LoadError, accept: Option<&str>) -> Response {
    eprintln!("explain load error: {e}");
    let (status, code, message) = load_error_parts(e);
    error_response(status, code, message, accept)
}

/// `POST /purge/<signature>/<path>`: invalidate every stored
/// variant of one result. The removal spans this process's
/// tiers (its memory, its disk store, and the shared store);
/// it does not reach another process's memory, so a replica
/// can keep serving its own copy until it evicts or restarts
/// (issue #351). The signature is an HMAC over the
/// path under a purge subkey derived from the secret (see
/// [`crate::url::purge_key`]), so only the secret holder can
/// purge and an image signature can never be replayed here; a
/// bad signature is a 403 and purges nothing. Idempotent:
/// purging an absent entry answers 200 with a count of 0.
/// One structured line per purge request (audit PL-8): the
/// destructive, secret-gated endpoint must be auditable. The
/// target is hashed; a raw target can carry signed URLs.
/// How many shared-tier objects one purge request scans
/// inline (issue #368): the shared walk reads entry
/// headers one at a time, so its inline cost must not
/// grow with the whole bucket. The tombstone already
/// guarantees the purge's correctness (issue #350); a
/// walk that outgrows the budget finishes in a
/// background task and logs its own count on completion.
const SHARED_PURGE_INLINE_SCAN: usize = 128;

/// Queue the un-scanned remainder of a bounded shared
/// purge, if any, on a background task (issue #368).
fn spawn_shared_purge_rest(
    shared: &Arc<crate::s3store::S3Store>,
    pending: Option<crate::s3store::PendingPurge>,
    event: &'static str,
    target: &str,
) {
    if let Some(pending) = pending {
        let shared = shared.clone();
        let target = target.to_string();
        tokio::spawn(async move {
            let removed = shared.finish_purge(pending).await;
            purge_log(event, &target, Some(removed));
        });
    }
}

fn purge_log(event: &str, target: &str, removed: Option<usize>) {
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(target.as_bytes());
    let hash: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    match removed {
        Some(n) => {
            println!("{{\"event\":\"{event}\",\"target\":\"{hash}\",\"removed\":{n}}}")
        }
        None => println!("{{\"event\":\"{event}\",\"target\":\"{hash}\",\"rejected\":true}}"),
    }
}

async fn purge_handle(
    State(state): State<AppState>,
    uri: axum::http::Uri,
    headers: HeaderMap,
) -> Response {
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let Some(tail) = uri.path().strip_prefix("/purge/") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /purge/<signature>/<target>".to_string(),
            accept.as_deref(),
        );
    };
    let full = format!("/{tail}");
    let (signature, key_id, target) = match parse_signature(&full) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                e.code(),
                e.to_string(),
                accept.as_deref(),
            );
        }
    };
    if !state
        .keyring
        .verify_purge(key_id.as_deref(), &signature, target)
    {
        state.metrics.record_purge();
        purge_log("purge", target, None);
        return error_response(
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature".to_string(),
            accept.as_deref(),
        );
    }
    // A sealed target (issue #255) unseals here, the first
    // step of the chain, now that the purge signature has
    // verified over the target as presented: the steps
    // below continue with the decrypted remainder, so a
    // purge by the sealed spelling evicts the entry its
    // plain spelling serves, and the reverse. The decrypted
    // remainder replaces the target wholesale (anything
    // after the token segment was the cosmetic friendly
    // name). A token that does not open under this keyring
    // falls back to the presented target, which matches
    // nothing: the same quiet no-op an unverifiable spelling
    // gets everywhere else in this handler.
    let unsealed;
    let target = {
        let presented_segment = target.split('/').next().unwrap_or("");
        match crate::url::split_sealed_segment(presented_segment) {
            Some((kid, token)) if presented_segment.len() <= crabcrop_url::MAX_SEALED_SEGMENT => {
                match state.keyring.open_sealed(kid, token) {
                    Some(text) => {
                        unsealed = text;
                        unsealed.as_str()
                    }
                    None => target,
                }
            }
            _ => target,
        }
    };
    // The signature covers the target as presented; the
    // entries, though, are keyed by the canonical form of
    // the EXPANDED remainder (Q1, Q5). Expand presets the
    // way serve does, or a purge of the preset URL an
    // operator actually holds silently removes 0 entries.
    // A target that does not expand (a directly expanded
    // path, an unknown preset) falls back to itself.
    let expanded;
    let target = match crate::url::expand_presets(target, &state.presets, state.only_presets) {
        Ok(text) => {
            expanded = text;
            &expanded
        }
        Err(_) => target,
    };
    // Root sources expand the same way (issue #248): the
    // entry is keyed on the expansion, so a purge naming
    // the short spelling must reach it.
    let root_expanded;
    let target = match crate::url::expand_source_root_in_remainder(target, &state.source_roots) {
        Ok(text) => {
            root_expanded = text;
            &root_expanded
        }
        Err(_) => target,
    };
    // Entries are keyed post-merge (issue #145): with
    // defaults configured, the purge target's key form is
    // the canonical form of the merged operations, or the
    // purge silently removes nothing.
    let key_form = if state.default_ops.is_empty() {
        crate::url::canonical_key_form(target)
    } else {
        match parse_operations_with_expiry(target) {
            Ok((ops, source, _, _)) => {
                let merged = crate::url::merge_default_ops(&state.default_ops, ops);
                crate::url::canonical_form(&merged, source)
            }
            Err(_) => crate::url::canonical_key_form(target),
        }
    };
    // The fan-out runs in a spawned task once the
    // signature has verified (issue #347): a purge that
    // has started completes and is logged even if the
    // caller disconnects mid-purge; dropping the
    // JoinHandle detaches the task instead of cancelling
    // it. The caller waits for the count, so the response
    // is unchanged.
    let task_state = state.clone();
    let task_target = target.to_string();
    let removed = tokio::spawn(async move {
        // The tombstone lands before any tier is touched
        // (issue #350): a write begun before this purge can
        // then see it whenever it tries to land, including
        // after this handler has finished.
        task_state.purge_guard.record_key(&key_form);
        let mut removed = task_state.cache.purge_matching(&key_form);
        if let Some(store) = &task_state.store {
            let store = store.clone();
            let key = key_form.clone();
            removed += store
                .blocking(move |store| store.purge_matching(&key))
                .await
                .unwrap_or(0);
        }
        if let Some(shared) = &task_state.shared_store {
            // The shared tier deletes the current epoch's
            // enumerable variants directly by object name
            // (audit R2-24); the fingerprint-free form drives
            // its matching walk, which also reaches older
            // epochs and hint-variant spellings.
            let epoch_form = format!("{key_form}#cfg{}", task_state.config_fingerprint);
            let (walked, pending) = shared
                .purge_matching_bounded(&key_form, &epoch_form, SHARED_PURGE_INLINE_SCAN)
                .await;
            removed += walked;
            spawn_shared_purge_rest(shared, pending, "purge", &task_target);
        }
        task_state.metrics.record_purge();
        purge_log("purge", &task_target, Some(removed));
        removed
    })
    .await
    .unwrap_or(0);
    (StatusCode::OK, format!("purged {removed}\n")).into_response()
}

/// `POST /purge-group/<signature>/<group>`: invalidate every
/// stored entry in one cache group (RFC 9875), across this
/// process's tiers (its memory, its disk store, and the
/// shared store); another process's memory is not reached,
/// as in the per-URL purge (issue #351). The signature is
/// verified under the purge-group
/// subkey, its own domain beside the purge subkey, over the
/// group name, so neither an image signature nor a per-URL
/// purge signature can be replayed here (audit R2-14). A
/// bad signature is a 403 and removes nothing; an unknown
/// group answers 200 with a count of 0.
async fn purge_group_handle(
    State(state): State<AppState>,
    uri: axum::http::Uri,
    headers: HeaderMap,
) -> Response {
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let Some(tail) = uri.path().strip_prefix("/purge-group/") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /purge-group/<signature>/<group>".to_string(),
            accept.as_deref(),
        );
    };
    let full = format!("/{tail}");
    let (signature, key_id, group) = match parse_signature(&full) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                e.code(),
                e.to_string(),
                accept.as_deref(),
            );
        }
    };
    if group.is_empty() || group.contains('/') {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_purge_group",
            "invalid purge group name".to_string(),
            accept.as_deref(),
        );
    }
    let target = format!("purge-group/{group}");
    if !state
        .keyring
        .verify_purge_group(key_id.as_deref(), &signature, group)
    {
        state.metrics.record_purge();
        purge_log("purge-group", &target, None);
        return error_response(
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature".to_string(),
            accept.as_deref(),
        );
    }
    // The fan-out runs in a spawned task once the
    // signature has verified, as in the per-URL purge
    // (issue #347).
    let task_state = state.clone();
    let task_target = target.clone();
    let group = group.to_string();
    let removed = tokio::spawn(async move {
        // Tombstone first, tiers second (issue #350), as in
        // the per-URL purge.
        task_state.purge_guard.record_group(&group);
        let mut removed = task_state.cache.purge_group(&group);
        if let Some(store) = &task_state.store {
            let store = store.clone();
            let group = group.clone();
            removed += store
                .blocking(move |store| store.purge_group(&group))
                .await
                .unwrap_or(0);
        }
        if let Some(shared) = &task_state.shared_store {
            let (walked, pending) = shared
                .purge_group_bounded(&group, SHARED_PURGE_INLINE_SCAN)
                .await;
            removed += walked;
            spawn_shared_purge_rest(shared, pending, "purge-group", &task_target);
        }
        task_state.metrics.record_purge();
        purge_log("purge-group", &task_target, Some(removed));
        removed
    })
    .await
    .unwrap_or(0);
    (StatusCode::OK, format!("purged {removed}\n")).into_response()
}

/// One structured line per upload request, the purge
/// precedent (audit PL-8): the write endpoint must be
/// auditable. The name is hashed for the same reason the
/// purge target is: a raw name is caller-chosen text.
fn upload_log(name: &str, outcome: &str) {
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(name.as_bytes());
    let hash: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    println!("{{\"event\":\"upload\",\"target\":\"{hash}\",\"outcome\":\"{outcome}\"}}");
}

/// The outcome of running upload validation under the
/// process gate.
enum UploadValidation {
    /// The bytes decode as an image this server can
    /// serve; the prepared form (what to store, and the
    /// media kind it validated as) travels with the
    /// verdict (issue #192).
    Valid(crate::engine::PreparedUpload),
    /// The bytes do not decode; the engine's error, kept
    /// for the server log.
    Invalid(EngineError),
    /// The process gate stayed saturated past its wait:
    /// the upload is shed with a 503, like a transform.
    Busy,
    /// Validation panicked on its blocking thread.
    Panicked,
}

/// Validate uploaded bytes under the process gate on a
/// blocking thread (issue #187). Validation is a real
/// decode (animated formats frame by frame, HEIC through
/// oxideav), so it is budgeted exactly like transform
/// work: without the gate, a burst of signed uploads
/// would multiply decode load past the concurrency bound
/// the rest of the server runs under (audit R2-06).
async fn run_upload_validation(state: &AppState, bytes: Vec<u8>) -> UploadValidation {
    let permit = match tokio::time::timeout(
        PROCESS_GATE_WAIT,
        state.process_gate.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        _ => return UploadValidation::Busy,
    };
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        crate::engine::prepare_upload(&bytes)
    })
    .await
    {
        Ok(Ok(prepared)) => UploadValidation::Valid(prepared),
        Ok(Err(e)) => UploadValidation::Invalid(e),
        Err(_) => UploadValidation::Panicked,
    }
}

/// `PUT /upload/<signature>/<name>`: store one original
/// (issue #168). The route is registered only when a
/// destination is configured, so `upload` here is always
/// `Some`; the handler still answers 404 without one, so
/// the rule holds however the state was built.
///
/// The signature is an HMAC over the name under an upload
/// subkey derived from the secret (see
/// [`crate::url::verify_upload`]): only the secret holder
/// can upload, an image or purge signature can never be
/// replayed here, and `unsafe` is never honored. The
/// expiring form (issue #192) signs
/// `exp:<unix-seconds>/<name>` instead and presents as
/// `/upload/<signature>/exp:<unix-seconds>/<name>`; the
/// expiry is inside the signed string, so it can neither
/// be stripped nor extended, and an expired token is a
/// 403 like an expired image URL. The checks run cheapest
/// first: signature, expiry, name shape, body size (read
/// with the cap as a hard bound, so an oversize body is
/// never fully buffered), the overwrite policy (so a
/// refused duplicate never pays a decode), and image
/// decode through the pipeline's own validation, run
/// under the process gate like any transform. What is
/// stored is the prepared form: the sanitized document
/// for an SVG, the caller's bytes for anything else, and
/// only when the name's extension matches the media kind
/// the body validated as. With overwriting off, the
/// final write is a conditional create, so the 409 the
/// policy promises holds even against a racing upload.
/// On success the body of the response is the exact
/// source reference to embed in transform URLs: `201`
/// for a new object, `200` when an allowed overwrite
/// replaced one.
/// Remove every cached derivative of `source` from all
/// three of this process's tiers; returns the total
/// removed. The upload
/// overwrite path (audit R2-07) shares this with any
/// future source-replacement flow.
async fn purge_source_everywhere(state: &AppState, source: &str) -> usize {
    // The fan-out runs in a spawned task, as in the purge
    // handlers (issue #347): a purge that has started
    // completes and is logged even if the caller is
    // dropped mid-purge. The caller waits for the count.
    let task_state = state.clone();
    let source = source.to_string();
    tokio::spawn(async move {
        // Tombstone first, tiers second (issue #350), as in
        // the per-URL purge.
        task_state.purge_guard.record_source(&source);
        let mut removed = task_state.cache.purge_source(&source);
        if let Some(store) = &task_state.store {
            let store = store.clone();
            let key = source.clone();
            removed += store
                .blocking(move |store| store.purge_source(&key))
                .await
                .unwrap_or(0);
        }
        if let Some(shared) = &task_state.shared_store {
            let (walked, pending) = shared
                .purge_source_bounded(&source, SHARED_PURGE_INLINE_SCAN)
                .await;
            removed += walked;
            spawn_shared_purge_rest(shared, pending, "purge-source", &source);
        }
        purge_log("purge-source", &source, Some(removed));
        removed
    })
    .await
    .unwrap_or(0)
}

async fn upload_handle(State(state): State<AppState>, req: axum::extract::Request) -> Response {
    let accept = req
        .headers()
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let Some(upload) = &state.upload else {
        return error_response(
            StatusCode::NOT_FOUND,
            "uploads_not_configured",
            "uploads are not configured".to_string(),
            accept.as_deref(),
        );
    };
    let (parts, body) = req.into_parts();
    let Some(tail) = parts.uri.path().strip_prefix("/upload/") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /upload/<signature>/<name>".to_string(),
            accept.as_deref(),
        );
    };
    let full = format!("/{tail}");
    // A sealed spelling is refused wholesale (issue #255):
    // the server never unseals as a service, because an
    // endpoint that opens tokens would be an unsealing
    // oracle for anyone who lifts one off a public page.
    // Operators unseal locally (the CLI's `unseal` verb,
    // under their own secret) and then explain the plain
    // form. The refusal is the uniform one and no source
    // fetch happens for it.
    let first_segment = tail.split('/').next().unwrap_or("");
    if crate::url::looks_sealed(first_segment) {
        return error_response(
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature".to_string(),
            accept.as_deref(),
        );
    }
    let (signature, key_id, rest) = match parse_signature(&full) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                e.code(),
                e.to_string(),
                accept.as_deref(),
            );
        }
    };
    // Verification runs over the remainder exactly as
    // presented: for the expiring form the `exp:` segment
    // is part of the signed string, so a tampered or
    // stripped expiry already fails here as a bad
    // signature.
    if !state
        .keyring
        .verify_upload(key_id.as_deref(), &signature, rest)
    {
        state.metrics.record_upload();
        upload_log(rest, "rejected:bad-signature");
        return error_response(
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature".to_string(),
            accept.as_deref(),
        );
    }
    let (expiry, name) = match crate::url::split_upload_remainder(rest) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                e.code(),
                e.to_string(),
                accept.as_deref(),
            );
        }
    };
    // Expiry is enforced after verification and before
    // anything else, like an image URL: the comparison is
    // exact, the token valid strictly before its second.
    if let Some(exp) = expiry {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if now >= exp {
            state.metrics.record_upload();
            upload_log(name, "rejected:expired");
            return error_response(
                StatusCode::FORBIDDEN,
                "signature_expired",
                "signature expired".to_string(),
                accept.as_deref(),
            );
        }
    }
    if let Err(e) = crate::upload::validate_upload_name(name) {
        state.metrics.record_upload();
        upload_log(name, "rejected:bad-name");
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_upload_name",
            e,
            accept.as_deref(),
        );
    }
    // The read is bounded by the cap itself: `to_bytes`
    // fails the moment the body passes it, which is the
    // oversize answer (413) without ever holding more
    // than the cap in memory. Any other read failure is
    // a bad request body (400).
    let bytes = match axum::body::to_bytes(body, upload.max_bytes as usize).await {
        Ok(bytes) => bytes,
        Err(_) => {
            state.metrics.record_upload();
            upload_log(name, "rejected:oversize-or-unreadable");
            return error_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                "upload_too_large",
                format!("upload exceeds the {}-byte limit", upload.max_bytes),
                accept.as_deref(),
            );
        }
    };
    if bytes.is_empty() {
        state.metrics.record_upload();
        upload_log(name, "rejected:not-an-image");
        return error_response(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "upload_not_an_image",
            "upload body is empty: an image is required".to_string(),
            accept.as_deref(),
        );
    }
    // The existence check runs before validation, so an
    // upload that will be refused as a duplicate never
    // pays a full decode first (audit R2-06). The value
    // is reused after the store to pick 200 vs 201 and
    // the log outcome.
    let existed = upload.exists(name).await;
    if existed && !upload.allow_overwrite {
        state.metrics.record_upload();
        upload_log(name, "rejected:exists");
        return error_response(
            StatusCode::CONFLICT,
            "upload_already_exists",
            "an object with this name already exists".to_string(),
            accept.as_deref(),
        );
    }
    // The bytes must decode as an image this server can
    // serve, through the pipeline's own decode branches:
    // anything else is refused before it is stored, so
    // the destination never holds an unservable object.
    // The decode runs under the process gate, like every
    // transform (see `run_upload_validation`).
    let prepared = match run_upload_validation(&state, bytes.to_vec()).await {
        UploadValidation::Valid(prepared) => prepared,
        UploadValidation::Invalid(e) => {
            eprintln!("upload rejected: {e}");
            state.metrics.record_upload();
            upload_log(name, "rejected:not-an-image");
            return error_response(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "upload_not_an_image",
                "upload is not an image this server can decode".to_string(),
                accept.as_deref(),
            );
        }
        UploadValidation::Busy => {
            state.metrics.record_upload();
            upload_log(name, "rejected:busy");
            return busy_response(
                "server busy: too many images processing at once",
                1,
                accept.as_deref(),
            );
        }
        UploadValidation::Panicked => {
            state.metrics.record_upload();
            upload_log(name, "rejected:validation-failed");
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "validation_failed",
                "image validation failed".to_string(),
                accept.as_deref(),
            );
        }
    };
    // The name must present the body as what it is: a
    // validated image may not be stored under an
    // extension belonging to another kind (or to active
    // content), because the destination may be served
    // directly (audit R2-22).
    if let Err(e) = crate::upload::check_upload_media(name, prepared.media) {
        state.metrics.record_upload();
        upload_log(name, "rejected:media-mismatch");
        return error_response(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "upload_media_mismatch",
            e,
            accept.as_deref(),
        );
    }
    // What lands is the prepared form, and with
    // overwriting off the write is a conditional create:
    // the exists check above is the cheap refusal, this
    // is the atomic one, so a name claimed between the
    // two still answers 409 and keeps the winner's bytes
    // (audit R2-21).
    let stored = if upload.allow_overwrite {
        upload.put(name, &prepared.stored).await
    } else {
        upload.put_new(name, &prepared.stored).await
    };
    match stored {
        Ok(()) => {
            state.metrics.record_upload();
            upload_log(name, if existed { "replaced" } else { "stored" });
            if existed {
                // An overwrite replaces the source every
                // cached derivative was rendered from: purge
                // them all, in every tier, so the next
                // request re-renders from the new original
                // instead of serving the stale transform
                // (audit R2-07). A first store has no
                // derivatives to invalidate.
                let source = upload.source_ref(name);
                // Logged inside the purge itself, so the
                // line lands even if this request is dropped
                // while the purge runs (issue #347).
                let _removed = purge_source_everywhere(&state, &source).await;
            }
            let status = if existed {
                StatusCode::OK
            } else {
                StatusCode::CREATED
            };
            let mut response = (status, format!("{}\n", upload.source_ref(name))).into_response();
            response.headers_mut().insert(
                axum::http::HeaderName::from_static("x-crabcrop-source"),
                axum::http::HeaderValue::from_str(&upload.source_ref(name)).unwrap(),
            );
            response
        }
        Err(crate::upload::UploadError::AlreadyExists) => {
            state.metrics.record_upload();
            upload_log(name, "rejected:exists");
            error_response(
                StatusCode::CONFLICT,
                "upload_already_exists",
                "an object with this name already exists".to_string(),
                accept.as_deref(),
            )
        }
        Err(crate::upload::UploadError::Backend(cause)) => {
            eprintln!("upload failed: {cause}");
            state.metrics.record_upload();
            upload_log(name, "rejected:store-failed");
            error_response(
                StatusCode::BAD_GATEWAY,
                "upload_store_unavailable",
                "the upload store is unavailable".to_string(),
                accept.as_deref(),
            )
        }
        Err(crate::upload::UploadError::Io(cause)) => {
            eprintln!("upload failed: {cause}");
            state.metrics.record_upload();
            upload_log(name, "rejected:store-failed");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "upload_store_failed",
                "the upload could not be stored".to_string(),
                accept.as_deref(),
            )
        }
    }
}

/// Quote one string for a structured-fields string list
/// (RFC 8941): backslash and quote are escaped.
fn sf_quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        if ch == '\\' || ch == '"' {
            out.push('\\');
        }
        out.push(ch);
    }
    out.push('"');
    out
}

/// Attach the cache observability headers (RFC 9211
/// Cache-Status, RFC 9875 Cache-Group) to an image response.
fn attach_cache_headers(response: &mut Response, cache_status: &str, groups: &[String]) {
    response.headers_mut().insert(
        axum::http::HeaderName::from_static("cache-status"),
        axum::http::HeaderValue::from_str(cache_status).unwrap(),
    );
    if !groups.is_empty() {
        let list = groups
            .iter()
            .map(|g| sf_quote(g))
            .collect::<Vec<_>>()
            .join(", ");
        response.headers_mut().insert(
            axum::http::HeaderName::from_static("cache-group"),
            axum::http::HeaderValue::from_str(&list).unwrap(),
        );
    }
}

fn generate_request_id() -> String {
    use sha2::Digest as _;
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let digest = sha2::Sha256::digest(format!("{nanos}-{n}").as_bytes());
    hex::encode(&digest[..8])
}

fn request_id_from(inbound: Option<&str>) -> String {
    if let Some(value) = inbound {
        let value = value.trim();
        if (1..=128).contains(&value.len())
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return value.to_string();
        }
    }
    generate_request_id()
}

async fn handle(
    State(state): State<AppState>,
    uri: axum::http::Uri,
    headers: HeaderMap,
) -> Response {
    // The raw path, exactly as sent: the signature covers it
    // byte-for-byte, so no percent-decoding may happen first
    // (the `Path` extractor decodes, which would rewrite e.g.
    // an encoded watermark URL before verification). At most
    // one leading slash is stripped (audit PL-7): a doubled
    // slash leaves an empty signature segment, which the
    // parser rejects, instead of silently serving the same
    // content under a second spelling.
    let path = uri.path().strip_prefix('/').unwrap_or(uri.path());
    let started = Instant::now();
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let request_id = request_id_from(headers.get("x-request-id").and_then(|v| v.to_str().ok()));
    let if_none_match = headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let dpr = dpr_hint(&headers);
    let width = width_hint(&headers);
    let save_data = save_data_hint(&headers);
    let (mut response, mut record) = serve(
        &state,
        path,
        accept.as_deref(),
        if_none_match.as_deref(),
        dpr,
        width,
        save_data,
    )
    .await;
    record.request_id = request_id.clone();
    record.status = response.status().as_u16();
    record.total = started.elapsed();
    response.headers_mut().insert(
        axum::http::HeaderName::from_static("x-request-id"),
        request_id.parse().unwrap(),
    );
    state.metrics.record(&record);
    println!("{}", record.to_json());
    response
}

async fn serve(
    state: &AppState,
    path: &str,
    accept: Option<&str>,
    if_none_match: Option<&str>,
    dpr_hint: Option<f32>,
    width_hint: Option<u32>,
    save_data: bool,
) -> (Response, RequestRecord) {
    let mut record = RequestRecord::default();
    let full = format!("/{path}");

    // A sealed spelling (issue #255) is intercepted before
    // the plain parse: the token sits in the signature slot,
    // and everything about it (an oversized segment, a
    // failed open, and every payload failure after a
    // successful open) answers with the one uniform refusal,
    // so a token is never a validity oracle. On a successful
    // open the decrypted remainder takes the place of `rest`
    // and flows through the exact pipeline a verified plain
    // remainder flows through: because decryption produces
    // the pre-expansion string the HMAC form carries, the
    // two spellings share one cache entry, one ETag, one
    // purge target by construction.
    let sealed_refusal = |record: RequestRecord| {
        (
            (StatusCode::FORBIDDEN, "bad signature").into_response(),
            record,
        )
    };
    let mut sealed = false;
    let mut sealed_remainder: Option<String> = None;
    let nolead = full.strip_prefix('/').unwrap_or(&full);
    let first_segment = nolead.split('/').next().unwrap_or("");
    // Grammar failures answer 400 with the parser's own
    // message as the body: it names only what the caller
    // sent, and it is the difference between a debuggable
    // rejection and an empty one.
    let (signature, key_id, rest) = if crate::url::looks_sealed(first_segment)
        && crate::url::split_sealed_segment(first_segment).is_none()
    {
        // Marker-shaped but not a token this scheme opens
        // (an unknown version): the uniform refusal, never
        // the grammar's 400, so the answer cannot announce
        // that the segment was not a token at all.
        return sealed_refusal(record);
    } else if let Some((kid, token)) = crate::url::split_sealed_segment(first_segment) {
        if first_segment.len() > crabcrop_url::MAX_SEALED_SEGMENT {
            return sealed_refusal(record);
        }
        let key_id = kid.map(str::to_string);
        let verify_start = Instant::now();
        let opened = state.keyring.open_sealed(key_id.as_deref(), token);
        record.verify = verify_start.elapsed();
        match opened {
            Some(text) => {
                sealed = true;
                sealed_remainder = Some(text);
                // The placeholder signature is never read:
                // for a sealed request the open above was
                // the verification the branch below runs
                // for plain spellings.
                (crate::url::Signature::Unsafe, key_id, "")
            }
            None => return sealed_refusal(record),
        }
    } else {
        match parse_signature(&full) {
            Ok(v) => v,
            Err(e) => {
                return (parse_failure(&e, accept), record);
            }
        }
    };
    let rest: &str = if sealed {
        sealed_remainder
            .as_deref()
            .expect("set when the open succeeds")
    } else {
        rest
    };
    if !sealed {
        let verify_start = Instant::now();
        let verified =
            state
                .keyring
                .verify(key_id.as_deref(), &signature, rest, state.allow_unsafe);
        record.verify = verify_start.elapsed();
        if !verified {
            return (
                error_response(
                    StatusCode::FORBIDDEN,
                    "bad_signature",
                    "bad signature".to_string(),
                    accept,
                ),
                record,
            );
        }
    }
    // Defense in depth for the purge domain separation (audit
    // P-1): an image remainder may not begin with the `purge`
    // segment, so the two URL shapes can never be confused.
    // The upload segment (issue #168) is reserved the same
    // way, for the same reason.
    if matches!(
        rest.split('/').next(),
        Some("purge" | "purge-group" | "upload")
    ) {
        if sealed {
            return sealed_refusal(record);
        }
        return (
            error_response(
                StatusCode::BAD_REQUEST,
                "reserved_segment",
                "an image URL may not begin with a purge or upload segment".to_string(),
                accept,
            ),
            record,
        );
    }
    // The cache groups (RFC 9875) are the presets the signed
    // remainder references, read before expansion rewrites it.
    let groups = crate::url::preset_names(rest);
    // Preset expansion happens after verification (the
    // signature covers the preset reference, not its expansion)
    // and before everything else: parsing, the cache key, and
    // the ops summary all see the expanded remainder, so
    // editing a preset produces fresh cache entries naturally.
    // Expansion always runs, even with no presets loaded:
    // a `preset:` segment is a grammar error
    // (`unknown_preset`) on every server, never source text.
    let expanded;
    let rest = match crate::url::expand_presets(rest, &state.presets, state.only_presets) {
        Ok(text) => {
            expanded = text;
            &expanded
        }
        Err(e) => {
            if sealed {
                return sealed_refusal(record);
            }
            return (parse_failure(&e, accept), record);
        }
    };
    let (ops, source_url, expiry, max_age) = match parse_operations_with_expiry(rest) {
        Ok(v) => v,
        Err(e) => {
            if sealed {
                return sealed_refusal(record);
            }
            return (parse_failure(&e, accept), record);
        }
    };
    // The operation summary is everything before the source URL.
    record.ops = rest[..rest.len() - source_url.len()]
        .trim_end_matches('/')
        .to_string();
    // A root source expands here (issue #248): after
    // verification and the ops summary (which reads the
    // source as written), before the cache key and the
    // fetch, so the short and full spellings of one source
    // are one render, sharing a cache entry and a purge
    // target. The loader's checks evaluate the expanded
    // URL unchanged, the allowlist included.
    let expanded_url;
    let source_url = match crate::url::expand_source_root(source_url, &state.source_roots) {
        Ok((expanded, _)) => {
            expanded_url = expanded;
            expanded_url.as_str()
        }
        Err(e) => {
            if sealed {
                return sealed_refusal(record);
            }
            return (parse_failure(&e, accept), record);
        }
    };
    // Server-wide defaults (issue #145) merge here: prepended
    // to the request's own operations, so the directive
    // analysis below, the engine, and the cache key all see
    // the merged list, and the URL wins every last-wins
    // directive (format, quality, fill) by coming later. The
    // signature never covers the defaults: they are operator
    // configuration, not request content.
    let ops = crate::url::merge_default_ops(&state.default_ops, ops);
    // Expiry is enforced after signature verification (the
    // segment is inside the signed remainder, so a tampered
    // expiry already failed as a bad signature) and before the
    // cache: an expired URL never serves, cached or not. The
    // comparison is exact: the URL is valid strictly before
    // its `exp` second.
    if let Some(exp) = expiry {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if now >= exp {
            return (
                error_response(
                    StatusCode::FORBIDDEN,
                    "signature_expired",
                    "signature expired".to_string(),
                    accept,
                ),
                record,
            );
        }
    }

    // One pass over the ops list settles every directive
    // the handler reads (issue #136).
    let directives = crate::url::RequestDirectives::analyze(&ops);

    // The DPR client hint settles sizing when the URL has no
    // explicit `dpr` directive; the effective factor is part of
    // what the response is, so it joins the cache key whenever
    // it changes the output.
    let effective_dpr = directives.dpr.or(dpr_hint).unwrap_or(1.0);

    // The quantized width a width hint resolves to, when the
    // ops size nothing themselves. It is part of what the
    // response is, so it joins the cache key below.
    let hinted_width = if directives.has_size_operation {
        None
    } else {
        width_hint.map(|w| quantize_width(w, state.width_step))
    };
    // Save-Data changes the bytes exactly when the engine
    // steps (hint present, no quality directive in the URL);
    // the key separates those entries under the same rule.
    let save_data_applies = save_data && !directives.quality_directive;

    // Without an explicit format in the URL, the response depends
    // on the Accept class as well as the path, so the cache key
    // carries the class. Explicit-format URLs key on the path alone.
    // The key is the request's canonical form (Q1): the
    // parsed operations re-serialized in their one
    // spelling, so equivalent requests share an entry.
    // Expiry and the cache lifetime are not operations,
    // so neither ever enters it.
    let preferred = negotiate_format(accept);
    let meta = wants_meta(accept);
    if meta && !state.meta_enabled {
        // Meta mode is switched off on this deployment
        // (issue #390): refuse specifically, like the
        // metrics endpoint does, rather than serving a
        // render the caller did not ask for.
        if sealed {
            return sealed_refusal(record);
        }
        return (
            error_response(
                StatusCode::FORBIDDEN,
                "meta_disabled",
                "meta mode is disabled on this server".to_string(),
                accept,
            ),
            record,
        );
    }
    let placeholders = meta && wants_placeholders(accept);
    let wants_video = directives.explicit_format.is_some_and(|f| f.is_video());
    // Meta describes an image; a video result is not one.
    // Refuse specifically rather than describing encoded
    // video bytes as if they were a still image.
    if meta && wants_video {
        if sealed {
            // The plain form's message echoes the payload's
            // own format argument; a sealed payload failure
            // answers the uniform refusal instead (issue
            // #255).
            return sealed_refusal(record);
        }
        return (
            error_response(
                StatusCode::BAD_REQUEST,
                "meta_not_available",
                format!(
                    "meta is not available for video output: format({}) serves encoded video, not an image",
                    crate::url::format_text(directives.explicit_format.unwrap())
                ),
                accept,
            ),
            record,
        );
    }
    // The debug view (issue #164) is a diagnostic computed
    // from this request's own processing: it bypasses the
    // cache in both directions (the meta precedent) and is
    // served no-store. Meta cannot describe it either: the
    // bytes are a marked-up canvas, not the transform result
    // the meta document reports on.
    let debug = directives.debug_focal;
    if meta && debug {
        if sealed {
            return sealed_refusal(record);
        }
        return (
            error_response(
                StatusCode::BAD_REQUEST,
                "meta_not_available",
                "meta is not available for debug output: debug:focal serves a diagnostic image, not a transform result"
                    .to_string(),
                accept,
            ),
            record,
        );
    }
    let key_base = crate::url::canonical_form(&ops, source_url);
    // The key also carries the configuration fingerprint
    // (audit R2-08): entries in the persistent tiers are
    // keyed under the configuration that produced them, so
    // a restart with different output-shaping settings
    // (metadata policy, quality tuning, width step, model
    // or font or encoder files) starts a fresh epoch
    // instead of serving the old configuration's renders.
    // The fingerprint sits inside the base, ahead of the
    // negotiated-class and hint suffixes.
    let key_base = format!("{key_base}#cfg{}", state.config_fingerprint);
    let mut cache_key = if directives.explicit_format.is_some() {
        key_base.clone()
    } else {
        let class = match preferred {
            Some(ImageFormat::Avif) => "avif",
            Some(ImageFormat::Jxl) => "jxl",
            Some(ImageFormat::WebP) => "webp",
            _ => "source",
        };
        format!("{key_base}#{class}")
    };
    if let Some(w) = hinted_width {
        cache_key = format!("{cache_key}#w{w}");
    }
    if save_data_applies {
        cache_key = format!("{cache_key}#sd");
    }
    if effective_dpr != 1.0 {
        cache_key = format!("{cache_key}#dpr{effective_dpr}");
    }
    // The key fully determines the lookup: a stored response can
    // be served without touching the loader or the engine. Meta
    // requests bypass the cache in both directions: they must
    // never be answered with image bytes, and their JSON is never
    // stored as an image entry. Debug requests bypass it too:
    // the diagnostic must never be served from, or stored into,
    // any tier (its key differs from the plain URL's anyway, but
    // the bypass is the rule, not the key's accident).
    let bypass = meta || debug;
    if !bypass {
        record.cache_looked_up = true;
    }
    let work = MissWork {
        ops,
        source_url: source_url.to_string(),
        directives,
        preferred,
        meta,
        debug,
        placeholders,
        hinted_width,
        dpr_hint,
        save_data,
        accept: accept.map(str::to_string),
        groups: groups.clone(),
        cache_key: cache_key.clone(),
        max_age,
        sealed,
    };
    // Debug requests keep bypassing everything, per
    // request, as before.
    if debug {
        let (response, record, _) = serve_miss(state, work, record).await;
        return (response, record);
    }
    // Meta requests coalesce through their own flights
    // (issue #295): the flight key is the cache key plus
    // every fact the document depends on beyond it (the
    // sealed spelling, which redacts the document, the
    // placeholder fields, and the Accept opt-ins), each
    // computed by the same helpers the miss path branches
    // on, so the key can never disagree with the document
    // a flight shares.
    if meta {
        let flight_key = format!(
            "{cache_key}\u{1}meta\u{1}{}{}{}{}{}{}",
            sealed as u8,
            placeholders as u8,
            wants_detections(accept) as u8,
            wants_moderation(accept) as u8,
            wants_embeddings(accept) as u8,
            wants_ocr(accept) as u8,
        );
        return serve_meta_flight(state, work, record, &flight_key).await;
    }
    // Singleflight (audit R2-09): the first miss for a key
    // leads (one fetch, one encode, under the gates below);
    // concurrent requests for the same key wait for the
    // leader, then re-check the tiers and serve its entry.
    // A leader whose result is not cached (an error, a
    // fallback) releases its waiters to take the miss path
    // themselves, one at a time, so a failing origin is
    // retried by the waiters rather than masked. Waits are
    // capped: a waiter whose leaders keep failing stops
    // queueing and runs the miss path itself (issue #375),
    // so it ends with the origin's real answer instead of
    // a busy response masking it.
    let mut waits = 0;
    loop {
        if let Some((hit, cache_status)) = lookup_tiers(state, &cache_key, &groups).await {
            record.cache_hit = true;
            record.bytes_out = hit.bytes.len();
            let etag = hit.etag.clone();
            let mut response = respond(hit.format, hit.quality, hit.bytes, max_age);
            attach_cache_headers(&mut response, cache_status, &groups);
            finalize_validators(&mut response, &etag, if_none_match);
            return (response, record);
        }
        match state.inflight.lead(&cache_key) {
            Flight::Leader(guard) => {
                let (mut response, record, render_etag) = serve_miss(state, work, record).await;
                drop(guard);
                if let Some(etag) = &render_etag {
                    finalize_validators(&mut response, etag, if_none_match);
                }
                return (response, record);
            }
            Flight::Follower(mut changed) => {
                if waits >= MAX_FLIGHT_WAITS {
                    // The wait cap is spent: run the miss
                    // path, the same settlement the meta
                    // flights use (issue #375). Shedding a
                    // busy response here masked the origin's
                    // real answer (its error, or a slow
                    // success) behind a server_busy this
                    // request never earned: the cap exists
                    // to stop queueing behind failing leaders
                    // forever, not to refuse a request that
                    // has not had its own attempt.
                    let (mut response, record, render_etag) = serve_miss(state, work, record).await;
                    if let Some(etag) = &render_etag {
                        finalize_validators(&mut response, etag, if_none_match);
                    }
                    return (response, record);
                }
                waits += 1;
                // The leader always signals on drop; the
                // timeout is only a backstop against a
                // wedged leader, after which the waiter
                // re-checks the tiers and may lead itself.
                let _ = tokio::time::timeout(std::time::Duration::from_secs(60), changed.changed())
                    .await;
            }
        }
    }
}

/// The body size above which a meta document is not
/// shared with a flight's followers (issue #295): real
/// documents are KB-scale, so the cap only ever refuses
/// the pathological.
const META_SHARE_CAP: usize = 4 * 1024 * 1024;

/// Meta singleflight (issue #295): concurrent identical
/// meta requests share one run of the miss path instead
/// of each loading and rendering the same source. The
/// leader runs `serve_miss` unchanged; on a 200 it
/// collects the document into the flight's slot before
/// its guard drops, and followers answer with a
/// reconstructed response (the leader's status, headers,
/// and body) and their own request record, which
/// truthfully shows they ran no load and no process
/// stage. Only 200 documents are shared: a flight that
/// produced anything else leaves its slot empty, and its
/// followers loop to lead or follow the next flight and
/// finally run the miss path themselves, so errors are
/// never shared and every request keeps the error
/// semantics of a solo request. Nothing outlives the
/// flight: meta remains uncached across time, exactly
/// as before.
///
/// The two choices this settles, deliberately: request
/// records stay per request with no flight marker (the
/// record type has no such field, and none is invented;
/// a follower is recognizable by its zero load and
/// process times), and a follower's response headers are
/// the leader's, which is sound because the flight key
/// pins every input the headers and the document were
/// computed from (see `serve`).
async fn serve_meta_flight(
    state: &AppState,
    work: MissWork,
    mut record: RequestRecord,
    key: &str,
) -> (Response, RequestRecord) {
    let mut waits = 0;
    loop {
        match state.inflight.lead_meta(key) {
            MetaFlight::Leader(guard, slot) => {
                let (response, record, _) = serve_miss(state, work, record).await;
                if response.status() == StatusCode::OK {
                    let (response, shared) = share_meta_response(response).await;
                    if let Some(shared) = shared {
                        *slot.shared.lock().unwrap() = Some(shared);
                    }
                    drop(guard);
                    return (response, record);
                }
                drop(guard);
                return (response, record);
            }
            MetaFlight::Follower(mut changed, slot) => {
                if waits >= MAX_FLIGHT_WAITS {
                    let (response, record, _) = serve_miss(state, work, record).await;
                    return (response, record);
                }
                waits += 1;
                // The leader always signals on drop; the
                // timeout is only a backstop against a
                // wedged leader, after which the waiter
                // re-checks the slot and may lead itself.
                let _ = tokio::time::timeout(std::time::Duration::from_secs(60), changed.changed())
                    .await;
                if let Some(shared) = slot.shared.lock().unwrap().clone() {
                    record.bytes_out = shared.body.len();
                    return (respond_shared(&shared), record);
                }
            }
        }
    }
}

/// Rebuild a response from a shared meta document: the
/// leader's status and headers, its body bytes.
fn respond_shared(shared: &SharedMeta) -> Response {
    let mut response = Response::new(axum::body::Body::from(shared.body.clone()));
    *response.status_mut() = shared.status;
    *response.headers_mut() = shared.headers.clone();
    response
}

/// Split a 200 meta response into the shared form a
/// flight slot carries, rebuilding the leader's own
/// response from the same parts. Returns `None` for the
/// shared form when the body is not collected whole
/// within the share cap; meta bodies are in-memory, so
/// in practice only the cap refuses, and the leader's
/// response is rebuilt untouched in that case whenever
/// the body was not consumed trying.
async fn share_meta_response(response: Response) -> (Response, Option<SharedMeta>) {
    let (parts, body) = response.into_parts();
    if <axum::body::Body as axum::body::HttpBody>::size_hint(&body)
        .exact()
        .is_none_or(|n| n > META_SHARE_CAP as u64)
    {
        return (Response::from_parts(parts, body), None);
    }
    match axum::body::to_bytes(body, META_SHARE_CAP).await {
        Ok(bytes) => {
            let shared = SharedMeta {
                status: parts.status,
                headers: parts.headers.clone(),
                body: bytes,
            };
            let response = Response::from_parts(parts, axum::body::Body::from(shared.body.clone()));
            (response, Some(shared))
        }
        Err(_) => {
            // Unreachable for the in-memory bodies meta
            // answers with (the size hint above already
            // vouched for the body); the status and
            // headers survive, the body does not.
            eprintln!("meta flight: leader body could not be collected for sharing");
            (Response::from_parts(parts, axum::body::Body::empty()), None)
        }
    }
}

/// How many times one request may wait on another
/// request's in-flight miss for the same key before it
/// stops waiting and runs the miss path itself (audit
/// R2-09, issue #375).
const MAX_FLIGHT_WAITS: u32 = 3;

/// Whether a purge recorded after `snapshot` covers
/// `entry` (issue #350): the write that carries this
/// snapshot began before that purge, so its result must
/// not stand in any tier.
fn write_is_stale(state: &AppState, snapshot: u64, key: &str, entry: &CachedImage) -> bool {
    state.purge_guard.purge_covers(snapshot, key, &entry.groups)
}

/// Store `entry` in the memory tier under the purge
/// guard (issue #350). The write is skipped when a purge
/// covering it already landed; a purge that lands while
/// the write is in flight is reconciled right after: the
/// entry just written is removed again, so the purge
/// wins however the two interleave. The removal matches
/// on the entry's ETag, so an entry a fresher write has
/// already replaced under the same key is left alone.
fn put_memory_tier(state: &AppState, snapshot: u64, key: &str, entry: &CachedImage) {
    if write_is_stale(state, snapshot, key, entry) {
        return;
    }
    state.cache.put(key.to_string(), entry.clone());
    if write_is_stale(state, snapshot, key, entry)
        && state
            .cache
            .get(key)
            .is_some_and(|held| held.etag == entry.etag)
    {
        state.cache.remove(key);
    }
}

/// The reconcile every guarded disk write and group
/// merge runs after landing (issue #350): when a purge
/// recorded after `snapshot` covers the entry and the
/// disk tier still holds this exact copy, remove it, so
/// the copy the write or merge just made is not left
/// behind for the purge to miss. The ETag match leaves
/// alone an entry a fresher write has already replaced
/// under the same key.
fn reconcile_disk(
    store: &crate::storage::DiskStore,
    guard: &crate::cache::PurgeGuard,
    snapshot: u64,
    key: &str,
    entry: &CachedImage,
) {
    if guard.purge_covers(snapshot, key, &entry.groups)
        && store.get(key).is_some_and(|held| held.etag == entry.etag)
    {
        store.remove(key);
    }
}

/// Store `entry` in the disk tier under the purge guard
/// (issue #350): the same skip-and-reconcile discipline
/// as [`put_memory_tier`], run as one blocking operation
/// (issue #347), so the guard checks and the store calls
/// stay in one sequence that finishes even if the
/// request that asked for it is dropped while it waits.
async fn put_disk_tier(state: &AppState, snapshot: u64, key: &str, entry: &CachedImage) {
    let Some(store) = &state.store else {
        return;
    };
    let store = store.clone();
    let guard = state.purge_guard.clone();
    let key = key.to_string();
    let entry = entry.clone();
    let _ = store
        .blocking(move |store| {
            if guard.purge_covers(snapshot, &key, &entry.groups) {
                return;
            }
            store.put(&key, &entry);
            reconcile_disk(store, &guard, snapshot, &key, &entry);
        })
        .await;
}

/// Write `entry` through to the shared tier in the
/// background under the purge guard (issue #350): the
/// same skip-and-reconcile discipline as the local
/// tiers, applied inside the spawned task, because the
/// purge it must lose to can land at any point during
/// the network write.
fn spawn_shared_put(state: &AppState, snapshot: u64, key: &str, entry: CachedImage) {
    let Some(shared) = &state.shared_store else {
        return;
    };
    let guard = state.purge_guard.clone();
    let shared = shared.clone();
    let key = key.to_string();
    tokio::spawn(async move {
        if guard.purge_covers(snapshot, &key, &entry.groups) {
            return;
        }
        shared.put(&key, &entry).await;
        if guard.purge_covers(snapshot, &key, &entry.groups)
            && shared
                .get(&key)
                .await
                .is_some_and(|held| held.etag == entry.etag)
        {
            shared.remove(&key).await;
        }
    });
}

/// The tier lookup chain for one key: memory, then disk
/// (a hit promotes into memory), then the shared tier (a
/// hit promotes into both local tiers). On any hit, the
/// request's cache groups are unioned into the stored
/// entry (audit R2-15): provenance accumulates, so an
/// entry first stored through a group-free spelling is
/// still removed by a later group purge once any grouped
/// spelling of the same URL has been served. The union
/// reaches every tier that holds the entry, not just the
/// one that served the hit (issue #350): membership that
/// stayed in memory alone would be lost at restart,
/// leaving the persistent copies invisible to the group
/// purge. Every write this chain makes (promotions,
/// merge write-backs) runs under the purge guard, under
/// the generation the lookup started with. Returns the
/// image and the `Cache-Status` value for the tier that
/// served it.
async fn lookup_tiers(
    state: &AppState,
    cache_key: &str,
    groups: &[String],
) -> Option<(CachedImage, &'static str)> {
    let snapshot = state.purge_guard.snapshot();
    if let Some(hit) = state.cache.get(cache_key) {
        if !groups.is_empty() && state.cache.merge_groups(cache_key, groups) {
            // The memory set grew: propagate the new
            // membership to the persistent tiers. A group
            // the memory entry already carried needs no
            // propagation; it reached them when it was
            // first learned.
            let merged = crate::cache::union_groups(&hit.groups, groups);
            if merged != hit.groups {
                // The shared propagation is spawned first,
                // before any wait (issue #347): it does not
                // depend on the disk merge, and a request
                // dropped during that merge must not be
                // able to strand it.
                let mut entry = hit.clone();
                entry.groups = merged.clone();
                spawn_shared_put(state, snapshot, cache_key, entry);
            }
            if let Some(store) = &state.store {
                // The merge and its reconcile run as one
                // blocking operation (issue #347), so the
                // sequence finishes even if this request is
                // dropped while it waits.
                let store = store.clone();
                let guard = state.purge_guard.clone();
                let key = cache_key.to_string();
                let groups = groups.to_vec();
                let mut entry = hit.clone();
                entry.groups = merged;
                let _ = store
                    .blocking(move |store| {
                        store.merge_groups(&key, &groups);
                        reconcile_disk(store, &guard, snapshot, &key, &entry);
                    })
                    .await;
            }
        }
        return Some((hit, "crabcrop; hit"));
    }
    if let Some(store) = &state.store
        && store.contains(cache_key)
    {
        // The disk hit and everything it owes the other
        // tiers run in a spawned task (issue #347):
        // dropping this request's JoinHandle detaches the
        // task instead of cancelling it, so a client that
        // disconnects while the disk answers cannot leave
        // the disk entry holding a group the shared tier
        // never learned (issue #350).
        let task_state = state.clone();
        let key = cache_key.to_string();
        let lookup_groups = groups.to_vec();
        let joined = tokio::spawn(async move {
            lookup_disk_tier(&task_state, snapshot, &key, &lookup_groups).await
        })
        .await;
        if let Ok(Some(image)) = joined {
            return Some((image, "crabcrop; hit; detail=\"disk\""));
        }
    }
    if let Some(shared) = &state.shared_store
        && let Some(mut image) = shared.get(cache_key).await
    {
        if !groups.is_empty() {
            let merged = crate::cache::union_groups(&image.groups, groups);
            if merged != image.groups {
                image.groups = merged;
                // Best-effort, off the response path, like
                // the write-through put.
                spawn_shared_put(state, snapshot, cache_key, image.clone());
            }
        }
        put_memory_tier(state, snapshot, cache_key, &image);
        put_disk_tier(state, snapshot, cache_key, &image).await;
        return Some((image, "crabcrop; hit; detail=\"shared\""));
    }
    None
}

/// The disk tier's half of [`lookup_tiers`], run as its
/// own task so a dropped request cannot strand it
/// mid-chain: read the entry, union the request's groups
/// into it under the purge guard, then pay what the hit
/// owes the other tiers: the group propagation to the
/// shared tier and the promotion into memory, in that
/// order. Returns the image on a hit.
async fn lookup_disk_tier(
    state: &AppState,
    snapshot: u64,
    cache_key: &str,
    groups: &[String],
) -> Option<CachedImage> {
    let store = state.store.clone()?;
    let guard = state.purge_guard.clone();
    let key = cache_key.to_string();
    let lookup_groups = groups.to_vec();
    // The read, its group merge, and the merge reconcile
    // run as one blocking operation (issue #347); the
    // boolean reports whether the merge grew the entry's
    // group set.
    let (image, grew) = store
        .blocking(move |store| {
            let mut image = store.get(&key)?;
            let mut grew = false;
            if !lookup_groups.is_empty() && store.merge_groups(&key, &lookup_groups) {
                image.groups = crate::cache::union_groups(&image.groups, &lookup_groups);
                reconcile_disk(store, &guard, snapshot, &key, &image);
                grew = true;
            }
            Some((image, grew))
        })
        .await
        .flatten()?;
    if grew {
        // The merge grew the disk entry's set; the
        // shared tier's copy learns it too (issue #350).
        spawn_shared_put(state, snapshot, cache_key, image.clone());
    }
    put_memory_tier(state, snapshot, cache_key, &image);
    Some(image)
}

/// Everything the cache-miss path needs, bundled so the
/// singleflight loop in `serve` can hand it to whichever
/// request leads the miss.
struct MissWork {
    ops: Vec<crate::url::Operation>,
    source_url: String,
    directives: crate::url::RequestDirectives,
    preferred: Option<ImageFormat>,
    meta: bool,
    debug: bool,
    placeholders: bool,
    hinted_width: Option<u32>,
    dpr_hint: Option<f32>,
    save_data: bool,
    accept: Option<String>,
    groups: Vec<String>,
    cache_key: String,
    max_age: Option<u32>,
    /// Whether the request arrived in the sealed spelling
    /// (issue #255). Meta for it redacts every operation and
    /// source text. The flag describes this request, not the
    /// render: meta bypasses the cache in both directions,
    /// so a plain meta request on the same render still
    /// reports in full.
    sealed: bool,
}

/// Whether a request runs inference and so takes an AI
/// gate permit (issue #379): every AI directive counts,
/// except a smart crop, which runs a model only when a
/// face or focal provider participates in it. Without
/// either provider the crop's focal point is the built-in
/// saliency statistic, no inference runs, and gating the
/// request would let it consume permits and shed behind
/// real AI work for nothing.
fn ai_gate_applies(
    directives: &crate::url::RequestDirectives,
    face_provider: bool,
    focal_provider: bool,
) -> bool {
    directives.uses_ai_beyond_smart_crop()
        || (directives.smart_crop && (face_provider || focal_provider))
}

/// The cache-miss path: load the source, resolve
/// watermarks, run the engine under the gates, populate
/// the tiers, and respond (a meta document when asked).
/// Runs at most once per key at a time, led by one
/// request while its duplicates wait (see `serve`).
/// The third return value is the ETag of the rendered
/// bytes when the response serves a render, so the
/// caller can answer conditional requests against the
/// bytes themselves; it is `None` for a meta document
/// and for every error.
async fn serve_miss(
    state: &AppState,
    work: MissWork,
    mut record: RequestRecord,
) -> (Response, RequestRecord, Option<String>) {
    // The purge generation this render starts under
    // (issue #350), taken before the load: a purge that
    // lands while the render runs must not be undone by
    // the population writes at its end.
    let write_snapshot = state.purge_guard.snapshot();
    let MissWork {
        ops,
        source_url,
        directives,
        preferred,
        meta,
        debug,
        placeholders,
        hinted_width,
        dpr_hint,
        save_data,
        accept,
        groups,
        cache_key,
        max_age,
        sealed,
    } = work;
    let accept = accept.as_deref();
    let cache_status = "crabcrop; fwd=uri-miss";
    let source_url = source_url.as_str();
    let wants_video = directives.explicit_format.is_some_and(|f| f.is_video());
    // A `frame:` segment makes the source a video (issue
    // #214): only then may the loader accept a `video/*`
    // Content-Type for it. A `page:` segment does the
    // same for `application/pdf` (issue #215).
    let wants_frame = directives.frame.is_some();
    let wants_page = directives.page.is_some();
    record.load_ran = true;
    let load_start = Instant::now();
    let mut used_fallback = false;
    let admission = crate::loader::MediaAdmission {
        video: wants_frame,
        pdf: wants_page,
    };
    let bytes = match state.loader.load_media(source_url, admission).await {
        Ok(b) => b,
        Err(e) => {
            record.load = load_start.elapsed();
            // A policy refusal (Forbidden) is a decision, not a
            // failure: it never falls back. Any other load
            // failure degrades to the configured fallback when
            // there is one.
            let fb = if matches!(e, LoadError::Forbidden(_)) {
                None
            } else {
                fallback_source_bytes(state).await
            };
            match fb {
                Some(b) => {
                    used_fallback = true;
                    b
                }
                None => {
                    record.fail = FailStage::Load;
                    return (
                        sealed_or(sealed, load_error_response(e, accept)),
                        record,
                        None,
                    );
                }
            }
        }
    };
    record.load = load_start.elapsed();
    record.bytes_in = bytes.len();

    // Watermarks are fetched here, before the engine runs:
    // each referenced image goes through the same loader (and
    // its bounds and source policy) as sources, and a
    // watermark that fails to load fails the request rather
    // than being silently skipped. Only the bytes travel
    // with the job: the decode happens inside the gated
    // blocking job with the rest of the engine work, so a
    // watermark that does not decode fails the request from
    // there with the same engine error (audit R2-16).
    let mut watermark_sources: Vec<(String, Vec<u8>)> = Vec::new();
    for wm_url in &directives.watermark_urls {
        record.load_ran = true;
        let wm_bytes = match state.loader.load(wm_url).await {
            Ok(b) => b,
            Err(e) => {
                record.fail = FailStage::Load;
                return (
                    sealed_or(sealed, load_error_response(e, accept)),
                    record,
                    None,
                );
            }
        };
        watermark_sources.push((wm_url.to_string(), wm_bytes));
    }

    record.process_ran = true;
    let process_start = Instant::now();
    let ai_gate_applies = ai_gate_applies(&directives, state.face.is_some(), state.focal.is_some());
    // The request's one process deadline starts here, so
    // the gate waits below and the run itself share a
    // single budget (run_process takes it as a parameter).
    let process_deadline = process_start + state.process_timeout;
    // AI inference is gated process-wide: a burst of removebg
    // requests must not starve the transform pipeline. The
    // permit travels with the process job (run_process
    // holds it until the job actually finishes).
    let ai_permit = if ai_gate_applies {
        // The wait is bounded (audit PL-14): a burst of AI
        // requests sheds with a 503 instead of parking
        // unbounded tasks on the gate.
        match tokio::time::timeout(AI_GATE_WAIT, state.ai_gate.clone().acquire_owned()).await {
            Ok(Ok(permit)) => Some(permit),
            _ => {
                return (
                    sealed_or(
                        sealed,
                        busy_response("server busy: AI processing queue is full", 5, accept),
                    ),
                    record,
                    None,
                );
            }
        }
    } else {
        None
    };
    // Video work sits behind its own gate, on top of the
    // slow lane inside `run_process` (issue #324): the
    // external encoder is the server's most expensive and
    // least predictable work (seconds per job, a
    // subprocess), so a burst of video requests sheds here
    // instead of queueing unboundedly behind the transform
    // pipeline, and its lane permit comes from the slow
    // lane, never from the stills pool.
    // Frame extraction (issue #214) takes the same gate:
    // it runs the same class of subprocess. Page
    // rendering (issue #215) takes it too: it is the
    // external-tool gate in everything but name, and a
    // second gate for the same class of work would only
    // split the protection. Like the AI permit, it
    // travels with the process job and drains with it
    // (issue #357).
    let video_permit = if wants_video || wants_frame || wants_page {
        match tokio::time::timeout(VIDEO_GATE_WAIT, state.video_gate.clone().acquire_owned()).await
        {
            Ok(Ok(permit)) => Some(permit),
            _ => {
                return (
                    sealed_or(
                        sealed,
                        busy_response("server busy: video processing queue is full", 5, accept),
                    ),
                    record,
                    None,
                );
            }
        }
    } else {
        None
    };
    // Animated WebP encodes sit behind a single-permit
    // gate of their own (issue #231): one encode can run
    // for minutes, so without this gate duplicate or
    // distinct animated renders stack encodes until the
    // stills pool is drained. Behind the gate, a second
    // animated request sheds here with an explanation
    // instead of stacking, and the encode's lane permit
    // comes from the slow lane (issue #324), so the still
    // pool keeps every permit throughout. Like the AI
    // permit, it travels with the process job and drains
    // with it (issue #357).
    let anim_permit = if wants_animation_gate(&bytes, &directives) {
        match tokio::time::timeout(ANIM_GATE_WAIT, state.anim_gate.clone().acquire_owned()).await {
            Ok(Ok(permit)) => Some(permit),
            _ => {
                return (
                    sealed_or(
                        sealed,
                        busy_response("server busy: animation encoding queue is full", 5, accept),
                    ),
                    record,
                    None,
                );
            }
        }
    } else {
        None
    };
    // A width hint applies only when the URL sizes nothing
    // itself: it synthesizes the width-only size op a client
    // URL would have carried, ahead of the URL's own ops.
    let job_ops = match hinted_width {
        Some(w) => {
            let mut combined = vec![crate::url::Operation::Size(crate::url::ImageSize {
                width: crate::url::Dimension::Pixels(w),
                height: crate::url::Dimension::Proportional,
                flip_horizontal: false,
                flip_vertical: false,
            })];
            combined.extend(ops.iter().cloned());
            combined
        }
        None => ops.clone(),
    };
    let job = ProcessJob {
        save_data,
        preserve_metadata: state.preserve_metadata,
        ops: job_ops,
        preferred,
        watermark_sources,
        placeholders,
        mask_provider: state.removebg.clone(),
        upscale_provider: state.upscale.clone(),
        inpaint_provider: state.inpaint.clone(),
        segment_provider: state.segment.clone(),
        face_provider: state.face.clone(),
        detect_provider: state.detect.clone(),
        declared_detect_providers: state.detectors.iter().map(|d| d.provider.clone()).collect(),
        focal_provider: state.focal.clone(),
        text_renderer: state.font.clone(),
        video_encoder: state.video_encoder.clone(),
        frame_extractor: state.frame_extractor.clone(),
        pdf_renderer: state.pdf_renderer.clone(),
        provenance_signer: state.provenance_signer.clone(),
        ai_edited: directives.removebg
            || directives.studio
            || directives.remove_region
            || directives.upscale
            || directives.needs_face_model
            || directives.has_class_redaction(),
        want_provenance: meta,
        dpr: dpr_hint,
        ssim_target: state.ssim_target,
        brand_color: state.brand_color,
        avif_speed: state.avif_speed,
        jxl_speed: state.jxl_speed,
    };
    // The lane this run draws its permit from (issue
    // #324): long-running work goes to the slow lane, so
    // its saturation sheds slow requests only and stills
    // keep their full capacity; everything else takes
    // the stills lane as before.
    let lane = if wants_slow_lane(&bytes, &directives, wants_video, wants_frame, wants_page) {
        Lane::Slow
    } else {
        Lane::Process
    };
    let permits = JobPermits {
        ai: ai_permit,
        video: video_permit,
        anim: anim_permit,
    };
    let processed = match run_process(state, &job, &bytes, lane, process_deadline, permits).await {
        Ok(p) => p,
        Err(ProcessFailure::Busy) => {
            return (
                sealed_or(
                    sealed,
                    match lane {
                        Lane::Slow => busy_response(
                            "server busy: long-running processing lane is full",
                            5,
                            accept,
                        ),
                        Lane::Process => busy_response(
                            "server busy: too many images processing at once",
                            1,
                            accept,
                        ),
                    },
                ),
                record,
                None,
            );
        }
        Err(ProcessFailure::Panicked) => {
            record.fail = FailStage::Engine;
            return (
                sealed_or(
                    sealed,
                    error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "processing_failed",
                        "image processing failed".to_string(),
                        accept,
                    ),
                ),
                record,
                None,
            );
        }
        Err(ProcessFailure::TimedOut) => {
            record.fail = FailStage::Engine;
            return (
                sealed_or(
                    sealed,
                    error_response(
                        StatusCode::GATEWAY_TIMEOUT,
                        "processing_timeout",
                        "processing timed out".to_string(),
                        accept,
                    ),
                ),
                record,
                None,
            );
        }
        Err(ProcessFailure::Engine(e)) => {
            record.process = process_start.elapsed();
            // A source that loads but does not decode is the
            // other fallback case. Other engine errors are
            // properties of the request or the configuration,
            // not of the source, and stay honest.
            let retry = if !used_fallback && matches!(e, EngineError::Decode(_)) {
                match fallback_source_bytes(state).await {
                    Some(fb) => {
                        // The fallback run draws on the same
                        // gates as the first: each kind of
                        // work re-acquires its permit (the
                        // first run's permits drained with
                        // its job), and a full gate leaves
                        // the original engine error standing.
                        let mut retry_permits = JobPermits::default();
                        let mut may_retry = true;
                        if ai_gate_applies {
                            match tokio::time::timeout(
                                AI_GATE_WAIT,
                                state.ai_gate.clone().acquire_owned(),
                            )
                            .await
                            {
                                Ok(Ok(p)) => retry_permits.ai = Some(p),
                                _ => may_retry = false,
                            }
                        }
                        if may_retry && (wants_video || wants_frame || wants_page) {
                            match tokio::time::timeout(
                                VIDEO_GATE_WAIT,
                                state.video_gate.clone().acquire_owned(),
                            )
                            .await
                            {
                                Ok(Ok(p)) => retry_permits.video = Some(p),
                                _ => may_retry = false,
                            }
                        }
                        if may_retry && wants_animation_gate(&bytes, &directives) {
                            match tokio::time::timeout(
                                ANIM_GATE_WAIT,
                                state.anim_gate.clone().acquire_owned(),
                            )
                            .await
                            {
                                Ok(Ok(p)) => retry_permits.anim = Some(p),
                                _ => may_retry = false,
                            }
                        }
                        if may_retry {
                            run_process(state, &job, &fb, lane, process_deadline, retry_permits)
                                .await
                                .ok()
                        } else {
                            None
                        }
                    }
                    None => None,
                }
            } else {
                None
            };
            match retry {
                Some(p) => {
                    used_fallback = true;
                    record.process = process_start.elapsed();
                    p
                }
                None => {
                    record.fail = FailStage::Engine;
                    return (
                        sealed_or(sealed, engine_error_response(e, accept)),
                        record,
                        None,
                    );
                }
            }
        }
    };
    record.process = process_start.elapsed();
    // The validator for this render is a digest of its
    // bytes, computed once here: it is stored with the
    // cache entry and returned to the caller, so a hit and
    // a fresh render answer conditional requests against
    // the same value, and a re-render under the same key
    // whose bytes differ (the source changed, the old entry
    // was purged) answers with a different one instead of
    // the key's old validator.
    let render_etag = etag_for_bytes(&processed.bytes);

    // Only successful responses are cached; errors never are.
    // Meta responses are not image entries and are not cached.
    // Fallback responses are not cached either: the placeholder
    // must never be stored as if it were the real image, so the
    // origin gets retried on the next request. Debug responses
    // are not cached: the diagnostic is not the transform
    // result its key would claim.
    if !meta && !used_fallback && !debug {
        let entry = CachedImage {
            bytes: bytes::Bytes::from(processed.bytes.clone()),
            format: processed.format,
            quality: processed.quality,
            groups: groups.clone(),
            etag: render_etag.clone(),
        };
        // Population runs under the purge guard (issue
        // #350): a purge that landed while this render was
        // running keeps the entry out of every tier, while
        // the response below still serves the render
        // itself. The shared tier's write stays off the
        // response path, as before.
        put_memory_tier(state, write_snapshot, &cache_key, &entry);
        // The shared write is spawned before the disk
        // write waits (issue #347): it is already off the
        // response path, and the order means a request
        // dropped during the disk write cannot strand it.
        spawn_shared_put(state, write_snapshot, &cache_key, entry.clone());
        put_disk_tier(state, write_snapshot, &cache_key, &entry).await;
    }
    record.bytes_out = processed.bytes.len();
    // The render's AI permit drained with its job:
    // run_process holds it until the work finishes, which
    // on this success path has already happened, so the
    // gate is free and the annotation paths acquire it per
    // section like any other request. (They once ran under
    // the request's own held permit, audit R2-29; that
    // permit no longer outlives the run.)
    let ai_gated = false;
    // The annotation sections share one deadline, the
    // same process budget the render ran under: a
    // section that has not answered when it passes is
    // left out (tags, detections) or reports itself
    // unavailable with the "timed_out" reason
    // (moderation, embedding, OCR), and the response
    // goes out with the sections that made it. OCR also
    // gets the deadline inside the run, so its per-line
    // recognition stops gathering instead of only being
    // cut off at the response.
    let annotation_deadline = Instant::now() + state.process_timeout;
    let annotation = if meta {
        tokio::time::timeout_at(
            tokio::time::Instant::from_std(annotation_deadline),
            annotate_meta(state, &processed, ai_gated),
        )
        .await
        .ok()
        .flatten()
    } else {
        None
    };
    let detections = if meta && wants_detections(accept) {
        tokio::time::timeout_at(
            tokio::time::Instant::from_std(annotation_deadline),
            detect_meta(state, &processed, ai_gated),
        )
        .await
        .ok()
        .flatten()
    } else {
        None
    };
    let moderation = if meta && wants_moderation(accept) {
        Some(
            match tokio::time::timeout_at(
                tokio::time::Instant::from_std(annotation_deadline),
                moderate_meta(state, &processed, ai_gated),
            )
            .await
            {
                Ok(m) => m,
                Err(_) => ModerationMeta::Unavailable("timed_out"),
            },
        )
    } else {
        None
    };
    // The embedding run (issue #172) serves two
    // fields: the embedding itself, when the client
    // opted in, and the aesthetic score, whenever a
    // head is configured. One run feeds both.
    let want_embedding = meta && wants_embeddings(accept);
    let embedding_run = if meta && (want_embedding || state.aesthetic.is_some()) {
        Some(
            match tokio::time::timeout_at(
                tokio::time::Instant::from_std(annotation_deadline),
                embed_meta(state, &processed, ai_gated),
            )
            .await
            {
                Ok(e) => e,
                Err(_) => EmbeddingMeta::Unavailable("timed_out"),
            },
        )
    } else {
        None
    };
    let aesthetic = compose_aesthetic(state, embedding_run.as_ref());
    let ocr = if meta && wants_ocr(accept) {
        Some(
            match tokio::time::timeout_at(
                tokio::time::Instant::from_std(annotation_deadline),
                ocr_meta(state, &processed, ai_gated, annotation_deadline),
            )
            .await
            {
                Ok(o) => o,
                Err(_) => OcrMeta::Unavailable("timed_out"),
            },
        )
    } else {
        None
    };
    let mut response = if meta {
        meta_response(
            &processed,
            sealed,
            annotation.as_ref(),
            detections.as_deref(),
            moderation.as_ref(),
            if want_embedding {
                embedding_run.as_ref()
            } else {
                None
            },
            aesthetic.as_ref(),
            ocr.as_ref(),
        )
    } else {
        respond(
            processed.format,
            processed.quality,
            bytes::Bytes::from(processed.bytes),
            max_age,
        )
    };
    if !meta {
        attach_cache_headers(&mut response, cache_status, &groups);
    }
    if used_fallback {
        record.fallback = true;
        response
            .headers_mut()
            .insert("x-crabcrop-fallback", "1".parse().unwrap());
        // A fallback is a placeholder for a transient failure:
        // it must never be pinned downstream for the immutable
        // year (audit PL-2).
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    }
    if debug {
        // A diagnostic is computed for this request only: no
        // downstream cache may pin it.
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    }
    // A meta response serves a document about the render,
    // not the render: it carries no validator for the
    // image bytes (and is no-store besides).
    let render_etag = if meta { None } else { Some(render_etag) };
    (response, record, render_etag)
}

/// Pick a format from the client's `Accept` header. Among
/// the negotiable formats (AVIF, JPEG XL, WebP) the client
/// accepts, the highest q-value wins (Q8); ties break by
/// the server preference order (AVIF, then JXL, then
/// WebP). A missing q means 1.0, q=0 is a refusal, and a
/// format named twice is governed by its first entry.
/// `None` keeps the detected source format. Wildcards do
/// not count as an offer: a bare `*/*` from a generic HTTP
/// client should not trigger the expensive AVIF encode.
fn negotiate_format(accept: Option<&str>) -> Option<ImageFormat> {
    let accept = accept?;
    let (mut avif, mut jxl, mut webp) = (None, None, None);
    for part in accept.split(',') {
        let mut segments = part.split(';');
        let media = segments.next().unwrap_or("").trim();
        let slot = match media {
            "image/avif" => &mut avif,
            "image/jxl" => &mut jxl,
            "image/webp" => &mut webp,
            _ => continue,
        };
        if slot.is_some() {
            continue;
        }
        let q = segments
            .filter_map(|p| {
                p.trim()
                    .strip_prefix("q=")
                    .and_then(|v| v.trim().parse::<f32>().ok())
            })
            .next()
            .unwrap_or(1.0);
        *slot = Some(q);
    }
    // Preference order is the array order; a strictly
    // greater q is required to displace the incumbent, so
    // ties go to the earlier (more preferred) format.
    let mut best: Option<(ImageFormat, f32)> = None;
    for (format, q) in [
        (ImageFormat::Avif, avif),
        (ImageFormat::Jxl, jxl),
        (ImageFormat::WebP, webp),
    ] {
        if let Some(q) = q
            && q > 0.0
            && best.is_none_or(|(_, best_q)| q > best_q)
        {
            best = Some((format, q));
        }
    }
    best.map(|(format, _)| format)
}

mod meta;
/// Names the test module uses from meta: re-exported
/// only for test builds, where they are used.
#[cfg(test)]
pub(in crate::server) use meta::{
    AestheticMeta, AestheticOutcome, Annotation, compose_annotation, compose_moderation,
    f32_to_f16_bits, moderation_json, objects_crop_json, ocr_json, pack_embedding, score_list,
    segments_json, studio_json,
};
pub use meta::{EmbeddingMeta, ModerationMeta, OcrMeta};
pub(in crate::server) use meta::{
    annotate_meta, compose_aesthetic, detect_meta, embed_meta, meta_response, moderate_meta,
    ocr_meta, wants_detections, wants_embeddings, wants_meta, wants_moderation, wants_ocr,
    wants_placeholders,
};

/// Build a successful image response: content type, cache
/// headers, and the auto-quality header when quality was chosen.
///
/// The cache lifetime is the immutable year unless the request
/// carried a `maxage:` directive (issue #159), which substitutes
/// its own value for that response only: the stored entry is
/// shared, headers are composed per response. `immutable` rides
/// along at any positive value (during the lifetime the body
/// will not change, so no revalidation is owed); at `max-age=0`
/// it is dropped, because a response that is stale on arrival
/// cannot also promise to never change, and the pair would tell
/// caches to pin what the caller just asked them to revalidate.
fn respond(
    format: ImageFormat,
    quality: Option<u8>,
    bytes: bytes::Bytes,
    max_age: Option<u32>,
) -> Response {
    let content_type = match format {
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::Png => "image/png",
        ImageFormat::WebP => "image/webp",
        ImageFormat::Jxl => "image/jxl",
        ImageFormat::Gif => "image/gif",
        ImageFormat::Avif => "image/avif",
        ImageFormat::Mp4 => "video/mp4",
        ImageFormat::WebM => "video/webm",
    };
    let cache_control = match max_age {
        None => "public, max-age=31536000, immutable".to_string(),
        Some(0) => "public, max-age=0".to_string(),
        Some(n) => format!("public, max-age={n}, immutable"),
    };
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    headers.insert(header::CACHE_CONTROL, cache_control.parse().unwrap());
    // The format may have been negotiated from the request's Accept.
    headers.insert(header::VARY, "Accept, DPR, Sec-CH-DPR, Sec-CH-Width, Sec-CH-Viewport-Width, Width, Viewport-Width, Save-Data".parse().unwrap());
    // Advertise the client hints the server reads, so
    // browsers send them on later requests; without the
    // advertisement only proxies and clients that set the
    // hints manually ever benefit from them.
    headers.insert(
        axum::http::HeaderName::from_static("accept-ch"),
        "Sec-CH-Width, Sec-CH-DPR, Sec-CH-Viewport-Width"
            .parse()
            .unwrap(),
    );
    if let Some(q) = quality {
        headers.insert(
            axum::http::HeaderName::from_static("x-crabcrop-quality"),
            q.to_string().parse().unwrap(),
        );
    }
    (headers, bytes).into_response()
}

fn if_none_match_matches(if_none_match: &str, etag: &str) -> bool {
    let inm = if_none_match.trim();
    if inm == "*" {
        return true;
    }
    inm.split(',').any(|candidate| {
        let candidate = candidate.trim();
        candidate.strip_prefix("W/").unwrap_or(candidate) == etag
    })
}

fn finalize_validators(response: &mut Response, etag: &str, if_none_match: Option<&str>) {
    let cacheable = response
        .headers()
        .get(header::CACHE_CONTROL)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("public"));
    if !cacheable || response.status() != StatusCode::OK {
        return;
    }
    response
        .headers_mut()
        .insert(header::ETAG, etag.parse().unwrap());
    if if_none_match.is_some_and(|inm| if_none_match_matches(inm, etag)) {
        *response.status_mut() = StatusCode::NOT_MODIFIED;
        *response.body_mut() = axum::body::Body::empty();
        response.headers_mut().remove(header::CONTENT_TYPE);
        response.headers_mut().remove(header::CONTENT_LENGTH);
    }
}

/// A 503 for a saturated gate. Retry-After tells clients
/// and CDNs when a retry has a chance, instead of letting
/// them retry straight into the same full queue; the value
/// is sized to the gate (the AI gate's jobs run seconds,
/// the process gate's run in well under one).
fn busy_response(message: &'static str, retry_after_secs: u64, accept: Option<&str>) -> Response {
    let mut response = error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_busy",
        message.to_string(),
        accept,
    );
    response.headers_mut().insert(
        header::RETRY_AFTER,
        retry_after_secs.to_string().parse().unwrap(),
    );
    response
}

/// The uniform refusal for a sealed request (issues #255,
/// #259): once a token has opened, every failure answers
/// with the same 403, so a lifted token reveals nothing
/// about the payload it carries. The real error still
/// reaches the server log through the normal builders.
fn sealed_refusal_response() -> Response {
    (StatusCode::FORBIDDEN, "bad signature").into_response()
}

/// `resp` for a plain request, the uniform sealed refusal
/// for a sealed one.
fn sealed_or(sealed: bool, resp: Response) -> Response {
    if sealed {
        sealed_refusal_response()
    } else {
        resp
    }
}

/// The load failure classes: status, stable code, and the
/// generic client message, shared by the image path and
/// the explain path. The detailed cause (which can embed
/// origin-controlled strings and server-constructed store
/// URLs) goes to the server log; clients get a short
/// generic message (PL-6). A policy refusal is
/// genericized the same way the explain path does it
/// (issue #196, requirement 3): the reason can name a
/// resolved address or a bucket, which are the operator's
/// facts, not the caller's.
fn load_error_parts(e: LoadError) -> (StatusCode, &'static str, String) {
    match e {
        LoadError::Forbidden(_) => (
            StatusCode::FORBIDDEN,
            "source_forbidden",
            "source refused by policy".to_string(),
        ),
        LoadError::NotFound => (
            StatusCode::NOT_FOUND,
            "source_not_found",
            "source not found".to_string(),
        ),
        LoadError::BadStatus(code) => (
            StatusCode::BAD_GATEWAY,
            "source_bad_status",
            format!("source returned status {code}"),
        ),
        LoadError::Network(_) => (
            StatusCode::BAD_GATEWAY,
            "source_fetch_failed",
            "source fetch failed".to_string(),
        ),
        LoadError::WrongContentType(_) => (
            StatusCode::BAD_GATEWAY,
            "source_not_an_image",
            "source did not return an image".to_string(),
        ),
        LoadError::Timeout => (
            StatusCode::GATEWAY_TIMEOUT,
            "source_timeout",
            "source timed out".to_string(),
        ),
        // 502, not 413: the client's request was a valid
        // signed URL; it is the origin object that is too
        // big, like the other origin faults above.
        LoadError::TooLarge { limit } => (
            StatusCode::BAD_GATEWAY,
            "source_too_large",
            format!("source exceeds {limit} bytes"),
        ),
    }
}

fn load_error_response(e: LoadError, accept: Option<&str>) -> Response {
    eprintln!("load error: {e}");
    let (status, code, message) = load_error_parts(e);
    error_response(status, code, message, accept)
}

/// One error response in the v2 contract: the plain-text
/// body every client has always gotten, or the v2 JSON
/// envelope (`{"error":{"code","message"}}`, the code
/// being the failure's stable name) when the client asked
/// for JSON, the same test the meta surface uses.
/// Grammar failures carry the parser's codes; operational
/// failures carry the codes assigned at their builders
/// (issue #356). Codes are frozen once published, like
/// the parser's: a failure class may be renamed in code,
/// but its code never changes. Sealed requests never
/// reach this builder for a payload failure: they answer
/// the uniform refusal instead (see `sealed_or`).
fn error_response(
    status: StatusCode,
    code: &str,
    message: String,
    accept: Option<&str>,
) -> Response {
    if wants_meta(accept) {
        let body = serde_json::json!({
            "error": { "code": code, "message": message }
        });
        (status, axum::Json(body)).into_response()
    } else {
        (status, message).into_response()
    }
}

/// Cap for a parse error echoed to a client (issue
/// #268): the parser's message embeds the offending
/// segment verbatim, and a segment is bounded only by
/// the URL length cap, so a 10 KB segment produced a
/// 10 KB error body. 200 bytes keeps the useful head of
/// every message shape; the full text stays in the
/// server log through the record.
const MAX_ERROR_ECHO: usize = 200;

fn truncate_echo(message: &str) -> String {
    if message.len() <= MAX_ERROR_ECHO {
        return message.to_string();
    }
    let mut end = MAX_ERROR_ECHO;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &message[..end])
}

/// A grammar failure's response: the parser's code and
/// its (truncated) message, as plain text or the v2
/// envelope per the client's Accept.
fn parse_failure(e: &crate::url::ParseError, accept: Option<&str>) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        e.code(),
        truncate_echo(&e.to_string()),
        accept,
    )
}

/// The engine failure classes: status, stable code, and
/// the generic client message.
fn engine_error_parts(e: EngineError) -> (StatusCode, &'static str, String) {
    match e {
        EngineError::Decode(_) => (
            StatusCode::BAD_GATEWAY,
            "decode_failed",
            "source image could not be decoded".to_string(),
        ),
        EngineError::Encode(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "encode_failed",
            "image encoding failed".to_string(),
        ),
        EngineError::Timeout => (
            StatusCode::GATEWAY_TIMEOUT,
            "processing_timeout",
            "processing timed out".to_string(),
        ),
        EngineError::Unsupported(what) => (StatusCode::BAD_REQUEST, "unsupported_operation", what),
        EngineError::NotConfigured(what) => (StatusCode::NOT_IMPLEMENTED, "not_configured", what),
        EngineError::Inference(crate::ai::AiError::BadOutput(_)) => (
            StatusCode::BAD_GATEWAY,
            "inference_bad_output",
            "model produced unexpected output".to_string(),
        ),
        EngineError::Inference(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference_failed",
            "image inference failed".to_string(),
        ),
        EngineError::Render(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "render_failed",
            "image rendering failed".to_string(),
        ),
        // The encoder is the operator's own program: its
        // failure is an upstream fault (502). The body is
        // generic, like every other class here: the cause
        // (the configured program path, the encoder's own
        // stderr) stays in the log line above, where the
        // operator reads it, and never reaches the client
        // (audit R2-30).
        EngineError::Video(_) => (
            StatusCode::BAD_GATEWAY,
            "video_encode_failed",
            "video encoding failed".to_string(),
        ),
        EngineError::Frame(_) => (
            StatusCode::BAD_GATEWAY,
            "frame_extract_failed",
            "video frame extraction failed".to_string(),
        ),
        EngineError::Pdf(_) => (
            StatusCode::BAD_GATEWAY,
            "pdf_render_failed",
            "PDF rendering failed".to_string(),
        ),
    }
}

fn engine_error_response(e: EngineError, accept: Option<&str>) -> Response {
    eprintln!("engine error: {e}");
    let (status, code, message) = engine_error_parts(e);
    error_response(status, code, message, accept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    const SECRET: &[u8] = b"test-secret";

    fn sign(remainder: &str) -> String {
        use hmac::{Hmac, KeyInit, Mac};
        use sha2::Sha256;
        let mut mac = Hmac::<Sha256>::new_from_slice(SECRET).unwrap();
        mac.update(remainder.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    fn test_png() -> Vec<u8> {
        let img = DynamicImage::new_rgb8(100, 60);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    /// One-shot source server: serves `test_png()` as image/png once.
    async fn spawn_image_server() -> String {
        spawn_bytes_server(test_png()).await
    }

    /// Serves a solid blue 10x10 PNG on every connection.
    async fn spawn_watermark_server() -> String {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            10,
            10,
            image::Rgb([0, 0, 255]),
        ));
        let mut png = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        spawn_bytes_server(png).await.replace("src.png", "wm.png")
    }

    async fn spawn_bytes_server(png: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            // Serves every connection: cache-bypassing requests
            // (meta mode) fetch the source more than once.
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    png.len()
                );
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(&png).await;
            }
        });
        format!("http://{addr}/src.png")
    }

    /// Persistent source server that counts its hits.
    async fn spawn_counting_server(
        status: u16,
        content_type: &str,
        body: Vec<u8>,
    ) -> (String, Arc<AtomicUsize>) {
        let hits = Arc::new(AtomicUsize::new(0));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let counted = hits.clone();
        let content_type = content_type.to_string();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                counted.fetch_add(1, Ordering::SeqCst);
                let mut buf = [0u8; 2048];
                let _ = socket.read(&mut buf).await;
                let reason = if status == 200 { "OK" } else { "Not Found" };
                let head = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(&body).await;
            }
        });
        (format!("http://{addr}/src.png"), hits)
    }

    fn test_png_red() -> Vec<u8> {
        let img =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(80, 40, image::Rgb([255, 0, 0])));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    /// Persistent source server whose body can be swapped
    /// between requests, counting its hits: the origin half
    /// of a source that changes under a stable URL.
    async fn spawn_swappable_server(
        body: Vec<u8>,
    ) -> (String, Arc<AtomicUsize>, Arc<std::sync::Mutex<Vec<u8>>>) {
        let hits = Arc::new(AtomicUsize::new(0));
        let current = Arc::new(std::sync::Mutex::new(body));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let counted = hits.clone();
        let served = current.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                counted.fetch_add(1, Ordering::SeqCst);
                let mut buf = [0u8; 2048];
                let _ = socket.read(&mut buf).await;
                let body = served.lock().unwrap().clone();
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.write_all(&body).await;
            }
        });
        (format!("http://{addr}/src.png"), hits, current)
    }

    const TEST_METRICS_TOKEN: &str = "test-metrics-token";

    async fn spawn_app(allow_unsafe: bool) -> String {
        spawn_app_with_metrics(allow_unsafe, Some(TEST_METRICS_TOKEN.to_string())).await
    }

    async fn spawn_app_with_metrics(allow_unsafe: bool, metrics_token: Option<String>) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// Two-frame animated GIF bytes for the video tests.
    fn animated_gif_bytes() -> Vec<u8> {
        let delay = image::Delay::from_numer_denom_ms(100, 1);
        let frames = vec![
            image::Frame::from_parts(
                image::RgbaImage::from_pixel(40, 20, image::Rgba([255, 0, 0, 255])),
                0,
                0,
                delay,
            ),
            image::Frame::from_parts(
                image::RgbaImage::from_pixel(40, 20, image::Rgba([0, 0, 255, 255])),
                0,
                0,
                delay,
            ),
        ];
        let mut out = Vec::new();
        image::codecs::gif::GifEncoder::new(&mut out)
            .encode_frames(frames)
            .unwrap();
        out
    }

    /// Two-frame animated WebP bytes for the animation
    /// gate tests (issue #231), built with the oxideav
    /// encoder like the engine fixtures: the image crate's
    /// WebP encoder writes stills only.
    fn animated_webp_bytes() -> Vec<u8> {
        let frame = |rgb: [u8; 3], delay_ms: u64| {
            let img =
                image::RgbaImage::from_pixel(40, 20, image::Rgba([rgb[0], rgb[1], rgb[2], 255]));
            let image = oxideav_webp::WebpImage::from_rgba8(40, 20, img.as_raw().clone()).unwrap();
            oxideav_webp::Frame::new(image, Some(std::time::Duration::from_millis(delay_ms)))
        };
        let frames = vec![frame([255, 0, 0], 100), frame([0, 0, 255], 200)];
        let opts = oxideav_webp::EncodeOptions::default()
            .with_frame_mode(oxideav_webp::AnimFrameMode::Lossless);
        oxideav_webp::encode_animation(&frames, &opts).unwrap()
    }

    /// Still WebP bytes: the animation probe's negative
    /// case (issue #231).
    fn still_webp_bytes() -> Vec<u8> {
        let img = DynamicImage::new_rgb8(40, 20);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::WebP).unwrap();
        buf.into_inner()
    }

    /// An app whose animation gate the test controls
    /// (issue #231), so the shed and isolation tests can
    /// hold the single permit themselves.
    /// GET /metrics with the test token (issue #270).
    async fn metrics_response(base: &str) -> reqwest::Response {
        client()
            .get(format!("{base}/metrics"))
            .header("authorization", format!("Bearer {TEST_METRICS_TOKEN}"))
            .send()
            .await
            .unwrap()
    }

    async fn metrics_text(base: &str) -> String {
        let resp = metrics_response(base).await;
        assert_eq!(resp.status(), StatusCode::OK);
        resp.text().await.unwrap()
    }

    async fn spawn_app_with_anim_gate(anim_gate: Arc<tokio::sync::Semaphore>) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate,
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn spawn_app_with_ai_gate(ai_gate: Arc<tokio::sync::Semaphore>) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate,
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn spawn_app_with_meta_enabled(meta_enabled: bool) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// An app with a fake video encoder configured (issue
    /// #162): the script drains stdin and answers with its
    /// canned behavior, so the whole video path runs without
    /// a real encoder.
    async fn spawn_app_with_video(script: &str) -> String {
        use std::io::Write as _;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("crabcrop-server-video-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("encoder.sh");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(script.as_bytes()).unwrap();
        drop(file);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let encoder = crate::video::VideoEncoder::new(
            path,
            std::time::Duration::from_secs(30),
            64 * 1024 * 1024,
        )
        .unwrap();
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: Some(Arc::new(encoder)),
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// An app with a fake video decoder configured
    /// (issue #214): its probe prints `report` (printf
    /// arguments, already quoted) on stderr, and its
    /// extraction answers with the test PNG, so the whole
    /// frame path runs without a real decoder.
    async fn spawn_app_with_frame_decoder(report: &str) -> String {
        use std::io::Write as _;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("crabcrop-server-frames-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("frame.png"), test_png()).unwrap();
        let frame_path = dir.join("frame.png").display().to_string();
        let script = format!(
            "#!/bin/sh\ncat > /dev/null\nfor a in \"$@\"; do\n  if [ \"$a\" = \"image2pipe\" ]; then cat \"{frame_path}\"; exit 0; fi\ndone\nprintf '%s\\n' {report} >&2\nexit 1\n"
        );
        let path = dir.join("decoder.sh");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(script.as_bytes()).unwrap();
        drop(file);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let extractor =
            crate::frames::VideoFrameExtractor::new(path, std::time::Duration::from_secs(30))
                .unwrap();
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: Some(Arc::new(extractor)),
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// The fake decoder's probe report for a good video:
    /// 3 seconds, 100x60 (the test PNG's size), 10 fps.
    const FAKE_VIDEO_REPORT: &str = "'  Duration: 00:00:03.00, start: 0.000000' '  Stream #0:0: Video: h264, yuv420p, 100x60, 10 fps'";

    /// Bytes that sniff as no image: an ISO-BMFF header
    /// fragment stands in for a video file.
    const FAKE_VIDEO_SOURCE: &[u8] = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2avc1";

    #[tokio::test]
    async fn frame_urls_render_the_frame_and_cache_it() {
        let (src, hits) = spawn_counting_server(200, "video/mp4", FAKE_VIDEO_SOURCE.to_vec()).await;
        let base = spawn_app_with_frame_decoder(FAKE_VIDEO_REPORT).await;
        let remainder = format!("frame:1.5/40x30/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let first = client().get(&url).send().await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(header(&first, "cache-status"), "crabcrop; fwd=uri-miss");
        let bytes = first.bytes().await.unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!((img.width(), img.height()), (40, 30));
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // The repeat is a cache hit: neither the origin
        // nor the decoder runs again.
        let second = client().get(&url).send().await.unwrap();
        assert_eq!(second.status(), StatusCode::OK);
        assert_eq!(header(&second, "cache-status"), "crabcrop; hit");
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        // The meta form carries the video record.
        let resp = client()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["video"]["duration"], 3.0, "{meta}");
        assert_eq!(meta["video"]["width"], 100, "{meta}");
        assert_eq!(meta["video"]["height"], 60, "{meta}");
        assert_eq!(meta["video"]["frame_timestamp"], 1.5, "{meta}");
        assert_eq!(meta["source_width"], 100, "{meta}");
        assert_eq!(meta["source_height"], 60, "{meta}");
    }

    #[tokio::test]
    async fn missing_sources_answer_404_and_transport_failures_502() {
        // Issue #325: a source that is not there (a missing
        // file under the root, a path escaping the root, an
        // origin answering 404) is a 404 with one generic
        // body, while a genuine transport failure stays 502.
        let dir = upload_dir("notfound");
        let base = spawn_app_with_upload(&dir, 1024 * 1024, false).await;

        let remainder = "50x30/file:///missing.jpg";
        let url = format!("{base}/{}/{remainder}", sign(remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let missing_body = resp.text().await.unwrap();
        assert_eq!(missing_body, "source not found");

        // A symlink out of the root: the canonicalized path
        // escapes, and the answer must be indistinguishable
        // from the missing file above.
        let outside =
            std::env::temp_dir().join(format!("crabcrop-srv-outside-{}", std::process::id()));
        std::fs::write(&outside, b"secret").unwrap();
        std::os::unix::fs::symlink(&outside, dir.join("link.jpg")).unwrap();
        let remainder = "50x30/file:///link.jpg";
        let url = format!("{base}/{}/{remainder}", sign(remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        assert_eq!(resp.text().await.unwrap(), missing_body);

        // The explain route maps the same case the same way.
        let remainder = "50x30/file:///missing.jpg";
        let url = format!("{base}/explain/{}/{remainder}", sign(remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        assert_eq!(resp.text().await.unwrap(), missing_body);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&outside);

        // An origin answering 404 is the same 404...
        let (src, _hits) = spawn_counting_server(404, "text/html", b"nope".to_vec()).await;
        let base = spawn_app(false).await;
        let remainder = format!("50x30/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        assert_eq!(resp.text().await.unwrap(), missing_body);

        // ...while a refused connection stays a 502.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let remainder = format!("50x30/http://127.0.0.1:{port}/img.jpg");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(resp.text().await.unwrap(), "source fetch failed");
    }

    #[tokio::test]
    async fn frame_urls_without_a_decoder_answer_501() {
        let (src, _hits) =
            spawn_counting_server(200, "video/mp4", FAKE_VIDEO_SOURCE.to_vec()).await;
        let base = spawn_app(false).await;
        let remainder = format!("frame:poster/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = resp.text().await.unwrap();
        assert!(body.contains("CRABCROP_VIDEO_DECODER"), "{body}");
    }

    #[tokio::test]
    async fn frame_urls_refuse_image_and_streamless_sources() {
        let base = spawn_app_with_frame_decoder(FAKE_VIDEO_REPORT).await;
        // An image source with a frame segment: the
        // loader serves it (it is an image), and the
        // engine refuses it as a bad source.
        let (img, _h) = spawn_counting_server(200, "image/png", test_png()).await;
        let remainder = format!("frame:poster/{img}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        // A video the decoder finds no stream in fails
        // the same way.
        let base = spawn_app_with_frame_decoder("'no streams in here at all'").await;
        let (vid, _h) = spawn_counting_server(200, "video/mp4", FAKE_VIDEO_SOURCE.to_vec()).await;
        let remainder = format!("frame:poster/{vid}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn a_video_content_type_needs_a_frame_segment() {
        // The same video source, fetched by a URL with no
        // frame segment: the loader's image-only rule
        // refuses it, exactly as before issue #214.
        let (src, _hits) =
            spawn_counting_server(200, "video/mp4", FAKE_VIDEO_SOURCE.to_vec()).await;
        let base = spawn_app_with_frame_decoder(FAKE_VIDEO_REPORT).await;
        let remainder = format!("40x30/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let body = resp.text().await.unwrap();
        assert!(body.contains("did not return an image"), "{body}");
    }

    #[tokio::test]
    async fn frame_grammar_errors_cost_no_fetch() {
        let (src, hits) = spawn_counting_server(200, "video/mp4", FAKE_VIDEO_SOURCE.to_vec()).await;
        let base = spawn_app_with_frame_decoder(FAKE_VIDEO_REPORT).await;
        let remainder = format!("frame:banana/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(json["error"]["code"], "invalid_frame");
        assert_eq!(hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn explain_labels_a_video_source() {
        let (src, hits) = spawn_counting_server(200, "video/mp4", FAKE_VIDEO_SOURCE.to_vec()).await;
        let base = spawn_app(false).await;
        let remainder = format!("frame:1.5/{src}");
        let url = format!("{base}/explain/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(json["source"]["kind"], "video");
        assert!(json["source"]["width"].is_null());
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        let segment = json["operations"][0]["segment"].as_str().unwrap();
        assert_eq!(segment, "frame:1.5");
        let gloss = json["operations"][0]["descriptions"][0].as_str().unwrap();
        assert!(gloss.contains("video frame at that timestamp"), "{gloss}");
    }

    /// An app with a fake PDF renderer configured
    /// (issue #215): the probe prints a fixed 3-page
    /// report and the renderer answers with the test
    /// PNG, so the whole page path runs without a real
    /// renderer. The probe script refuses the way the
    /// real oned does for marked sources: PASSWORD in
    /// the bytes is a password refusal, CORRUPT is
    /// unreadable.
    async fn spawn_app_with_pdf_renderer() -> String {
        use std::io::Write as _;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("crabcrop-server-pages-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("page.png"), test_png()).unwrap();
        let page_path = dir.join("page.png").display().to_string();
        let write_tool = |name: &str, script: String| {
            let path = dir.join(name);
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(script.as_bytes()).unwrap();
            drop(file);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            path
        };
        write_tool(
            "pdfinfo",
            "#!/bin/sh\ninput=$(cat)\ncase \"$input\" in\n  *PASSWORD*) printf '%s\\n' 'Command Line Error: Incorrect password' >&2; exit 1 ;;\n  *CORRUPT*) printf '%s\\n' 'Syntax Error: unreadable' >&2; exit 1 ;;\nesac\nprintf '%s\\n' 'Pages:           3' 'Encrypted:       no' 'Page size:       612 x 792 pts (letter)'\nexit 0\n".to_string(),
        );
        let program = write_tool(
            "pdftocairo",
            format!("#!/bin/sh\ncat > /dev/null\ncat \"{page_path}\"\nexit 0\n"),
        );
        let renderer =
            crate::pdf::PdfRenderer::new(program, std::time::Duration::from_secs(30)).unwrap();
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: Some(Arc::new(renderer)),
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// Bytes that carry the PDF header (the tools are
    /// fake; the document is never really parsed).
    const FAKE_PDF_SOURCE: &[u8] = b"%PDF-1.4\nfake document bytes for the server tests";

    #[tokio::test]
    async fn page_urls_render_the_page_and_cache_it() {
        let (src, hits) =
            spawn_counting_server(200, "application/pdf", FAKE_PDF_SOURCE.to_vec()).await;
        let base = spawn_app_with_pdf_renderer().await;
        let remainder = format!("page:2/40x30/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let first = client().get(&url).send().await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(header(&first, "cache-status"), "crabcrop; fwd=uri-miss");
        let bytes = first.bytes().await.unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!((img.width(), img.height()), (40, 30));
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // The repeat is a cache hit: neither the origin
        // nor the renderer runs again.
        let second = client().get(&url).send().await.unwrap();
        assert_eq!(second.status(), StatusCode::OK);
        assert_eq!(header(&second, "cache-status"), "crabcrop; hit");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // The other spelling of the same page shares the
        // entry: canonicalization happens before the
        // cache key is built.
        let remainder = format!("page:02/40x30/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        // The meta form carries the PDF record.
        let resp = client()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["pdf"]["pages"], 3, "{meta}");
        assert_eq!(meta["pdf"]["page"], 2, "{meta}");
        assert_eq!(meta["pdf"]["page_width"], 612, "{meta}");
        assert_eq!(meta["pdf"]["page_height"], 792, "{meta}");
        assert_eq!(meta["source_width"], 100, "{meta}");
        assert_eq!(meta["source_height"], 60, "{meta}");
        assert!(meta["video"].is_null(), "{meta}");
    }

    #[tokio::test]
    async fn page_urls_without_a_renderer_answer_501() {
        let (src, _hits) =
            spawn_counting_server(200, "application/pdf", FAKE_PDF_SOURCE.to_vec()).await;
        let base = spawn_app(false).await;
        let remainder = format!("page:1/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = resp.text().await.unwrap();
        assert!(body.contains("CRABCROP_PDF_RENDERER"), "{body}");
    }

    #[tokio::test]
    async fn page_urls_refuse_image_and_corrupt_sources() {
        let base = spawn_app_with_pdf_renderer().await;
        // An image source with a page segment: the
        // loader serves it (it is an image), and the
        // engine refuses it as a bad source.
        let (img, _h) = spawn_counting_server(200, "image/png", test_png()).await;
        let remainder = format!("page:1/{img}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        // A PDF the probe cannot read fails the same
        // way, with the generic decode body...
        let (bad, _h) =
            spawn_counting_server(200, "application/pdf", b"%PDF-1.4 CORRUPT junk".to_vec()).await;
        let remainder = format!("page:1/{bad}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let body = resp.text().await.unwrap();
        assert!(body.contains("could not be decoded"), "{body}");
        // ...and the server is unharmed: a good page
        // renders right after.
        let (src, _h) =
            spawn_counting_server(200, "application/pdf", FAKE_PDF_SOURCE.to_vec()).await;
        let remainder = format!("page:1/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn page_urls_answer_the_document_refusals_as_400() {
        let base = spawn_app_with_pdf_renderer().await;
        // Out of range refuses; it never serves the last
        // page under the asked number.
        let (src, _h) =
            spawn_counting_server(200, "application/pdf", FAKE_PDF_SOURCE.to_vec()).await;
        let remainder = format!("page:9/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp.text().await.unwrap();
        assert!(body.contains("page 9"), "{body}");
        assert!(body.contains("3 pages"), "{body}");
        // A document that needs a password is its own
        // refusal, naming the password.
        let (locked, _h) = spawn_counting_server(
            200,
            "application/pdf",
            b"%PDF-1.4 PASSWORD protected".to_vec(),
        )
        .await;
        let remainder = format!("page:1/{locked}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp.text().await.unwrap();
        assert!(body.contains("password-protected"), "{body}");
    }

    #[tokio::test]
    async fn a_pdf_content_type_needs_a_page_segment() {
        // The same PDF source, fetched by a URL with no
        // page segment: the loader's image-only rule
        // refuses it, exactly as before issue #215.
        let (src, _hits) =
            spawn_counting_server(200, "application/pdf", FAKE_PDF_SOURCE.to_vec()).await;
        let base = spawn_app_with_pdf_renderer().await;
        let remainder = format!("40x30/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let body = resp.text().await.unwrap();
        assert!(body.contains("did not return an image"), "{body}");
    }

    #[tokio::test]
    async fn page_grammar_errors_cost_no_fetch() {
        let (src, hits) =
            spawn_counting_server(200, "application/pdf", FAKE_PDF_SOURCE.to_vec()).await;
        let base = spawn_app_with_pdf_renderer().await;
        let remainder = format!("page:banana/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(json["error"]["code"], "invalid_page");
        assert_eq!(hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn explain_labels_a_pdf_source() {
        let (src, hits) =
            spawn_counting_server(200, "application/pdf", FAKE_PDF_SOURCE.to_vec()).await;
        let base = spawn_app(false).await;
        let remainder = format!("page:2/{src}");
        let url = format!("{base}/explain/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(json["source"]["kind"], "pdf");
        assert!(json["source"]["width"].is_null());
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        let segment = json["operations"][0]["segment"].as_str().unwrap();
        assert_eq!(segment, "page:2");
        let gloss = json["operations"][0]["descriptions"][0].as_str().unwrap();
        assert!(gloss.contains("of the PDF source"), "{gloss}");
    }

    #[tokio::test]
    async fn serves_video_from_an_animated_source_and_caches_it() {
        let src = spawn_bytes_server(animated_gif_bytes()).await;
        let base =
            spawn_app_with_video("#!/bin/sh\ncat > /dev/null\nprintf 'SERVER-VIDEO'\n").await;
        let remainder = format!("filters:format(mp4)/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let first = client().get(&url).send().await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(
            first.headers().get(header::CONTENT_TYPE).unwrap(),
            "video/mp4"
        );
        assert_eq!(header(&first, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(first.bytes().await.unwrap().as_ref(), b"SERVER-VIDEO");
        // The repeat is the same stored video: the encoder
        // runs once per unique URL, like every other
        // expensive transform.
        let second = client().get(&url).send().await.unwrap();
        assert_eq!(second.status(), StatusCode::OK);
        assert_eq!(
            second.headers().get(header::CONTENT_TYPE).unwrap(),
            "video/mp4"
        );
        assert_eq!(header(&second, "cache-status"), "crabcrop; hit");
        assert_eq!(second.bytes().await.unwrap().as_ref(), b"SERVER-VIDEO");
    }

    #[tokio::test]
    async fn video_encoder_failures_answer_a_generic_body() {
        // Audit R2-30: the encoder's stderr and the program
        // path are operator details for the server log. The
        // client gets the generic 502 body, like every other
        // engine failure class.
        let gif = spawn_bytes_server(animated_gif_bytes()).await;
        let base = spawn_app_with_video(
            "#!/bin/sh\ncat > /dev/null\necho 'secret encoder detail' >&2\nexit 3\n",
        )
        .await;
        let remainder = format!("filters:format(mp4)/{gif}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let body = resp.text().await.unwrap();
        assert_eq!(body, "video encoding failed");
        assert!(!body.contains("secret encoder detail"), "{body}");
        assert!(!body.contains("encoder.sh"), "{body}");
    }

    #[tokio::test]
    async fn video_requests_fail_specifically() {
        let gif = spawn_bytes_server(animated_gif_bytes()).await;
        let png = spawn_image_server().await;
        // Unconfigured server: 501 naming the configuration,
        // never a silent fallback to the animated GIF.
        let base = spawn_app(false).await;
        let remainder = format!("filters:format(mp4)/{gif}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = resp.text().await.unwrap();
        assert!(body.contains("video output is not configured"), "{body}");
        // Configured server, still source: 400 naming the
        // animated-source requirement.
        let base = spawn_app_with_video("#!/bin/sh\ncat > /dev/null\nprintf 'V'\n").await;
        let remainder = format!("filters:format(webm)/{png}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp.text().await.unwrap();
        assert!(body.contains("animated source"), "{body}");
        // Meta on a video URL: refused specifically rather
        // than describing video bytes as an image.
        let remainder = format!("filters:format(mp4)/{gif}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp.text().await.unwrap();
        assert!(
            body.contains("meta is not available for video output"),
            "{body}"
        );
        // A failing encoder: a generic 502. Its stderr goes
        // to the server log, never to the client (audit
        // R2-30; the dedicated test above pins the body).
        let base =
            spawn_app_with_video("#!/bin/sh\ncat > /dev/null\necho 'broken encoder' >&2\nexit 1\n")
                .await;
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let body = resp.text().await.unwrap();
        assert_eq!(body, "video encoding failed");
    }

    /// An app with server-wide default operations (issue
    /// #145), optionally in presets-only mode.
    async fn spawn_app_with_defaults(defaults: &str, only_presets: bool) -> String {
        let presets = std::collections::HashMap::new();
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets,
            only_presets,
            source_roots: std::collections::HashMap::new(),
            default_ops: crate::url::parse_default_ops(defaults, &std::collections::HashMap::new())
                .unwrap(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// An app in a rotation window: current key `k-new`,
    /// previous key `k-old` (issue #143).
    async fn spawn_app_rotating() -> String {
        let state = AppState {
            keyring: crate::url::Keyring {
                current_id: "k-new".to_string(),
                current: b"the-new-server-secret".to_vec(),
                previous: Some(("k-old".to_string(), b"the-old-server-secret".to_vec())),
            },
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// An app whose fallback image is the red 40x30 PNG the
    /// fallback tests assert against.
    async fn spawn_app_with_fallback() -> String {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            40,
            30,
            image::Rgb([255, 0, 0]),
        ));
        let mut png = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: Some(Fallback::Bytes(Arc::new(png))),
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn client() -> reqwest::Client {
        reqwest::Client::new()
    }

    fn now_unix() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    #[tokio::test]
    async fn purge_evicts_expiry_variants() {
        // Audit P-2: entries are keyed without their exp:
        // segment, so one purge evicts every expiry spelling.
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let plain = format!("100x50/{src}");
        let expiring = format!("exp:{}/{plain}", now_unix() + 3600);
        let url = format!("{base}/{}/{expiring}", sign(&expiring));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "second fetch is a cache hit"
        );
        let purge_sig = crate::url::purge_signature(SECRET, &plain);
        let resp = client()
            .post(format!("{base}/purge/{purge_sig}/{plain}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert!(resp.text().await.unwrap().contains("purged 1"));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "the exp variant was evicted too"
        );
    }

    #[tokio::test]
    async fn equivalent_spellings_share_one_cache_entry() {
        // Q1: the cache key is the canonical form, so the
        // jpg alias and the jpeg spelling are one entry,
        // and a purge in either spelling evicts it.
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let alias = format!("filters:format(jpg)/{src}");
        let canonical = format!("filters:format(jpeg)/{src}");
        for remainder in [&alias, &canonical] {
            let resp = client()
                .get(format!("{base}/{}/{remainder}", sign(remainder)))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
        }
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "the second spelling must be a cache hit"
        );
        let purge_sig = crate::url::purge_signature(SECRET, &alias);
        let resp = client()
            .post(format!("{base}/purge/{purge_sig}/{alias}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert!(resp.text().await.unwrap().contains("purged 1"));
        let resp = client()
            .get(format!("{base}/{}/{canonical}", sign(&canonical)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "the purge evicted the shared entry"
        );
    }

    #[tokio::test]
    async fn purging_a_preset_url_evicts_its_entry() {
        // Q5's trap: entries are keyed by the expanded
        // form, so purging the preset URL itself used to
        // remove 0. The handler expands before matching.
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app_with_presets(preset_test_map(), false).await;
        let remainder = format!("preset:small/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 1, "second fetch is a hit");
        let purge_sig = crate::url::purge_signature(SECRET, &remainder);
        let resp = client()
            .post(format!("{base}/purge/{purge_sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            resp.text().await.unwrap().contains("purged 1"),
            "the preset URL must purge its own entry"
        );
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 2, "the entry was evicted");
    }

    #[tokio::test]
    async fn unexpired_signed_urls_serve() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("exp:{}/300x50/{src}", now_unix() + 3600);
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn expired_signed_urls_are_forbidden() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        // exp in 1970: properly signed, long past.
        let remainder = format!("exp:1/300x50/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(resp.text().await.unwrap(), "signature expired");
    }

    #[tokio::test]
    async fn tampered_expiry_is_a_bad_signature() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let signed = format!("exp:{}/300x50/{src}", now_unix() + 3600);
        let signature = sign(&signed);
        // Extending the expiry without re-signing breaks the HMAC.
        let tampered = format!("exp:{}/300x50/{src}", now_unix() + 7200);
        let resp = client()
            .get(format!("{base}/{signature}/{tampered}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(resp.text().await.unwrap(), "bad signature");
    }

    /// Mint a sealed path for a remainder under the test
    /// secret (issue #255): the token segment, plus the
    /// friendly name when one is given.
    fn sealed_path(remainder: &str, name: Option<&str>) -> String {
        let token = crabcrop_url::seal(SECRET, remainder).unwrap();
        match name {
            Some(name) => format!("{token}/{name}"),
            None => token,
        }
    }

    #[tokio::test]
    async fn sealed_and_plain_share_one_cache_entry() {
        // The token carries the pre-expansion remainder, so
        // the two spellings are one render: one fetch, one
        // entry, one ETag, whichever arrives first.
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        // Plain first, sealed second.
        let plain = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(plain.status(), 200);
        assert_eq!(header(&plain, "cache-status"), "crabcrop; fwd=uri-miss");
        let etag = header(&plain, "etag");
        let plain_bytes = plain.bytes().await.unwrap();
        let sealed = client()
            .get(format!(
                "{base}/{}",
                sealed_path(&remainder, Some("photo.png"))
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(sealed.status(), 200);
        assert_eq!(header(&sealed, "cache-status"), "crabcrop; hit");
        assert_eq!(header(&sealed, "etag"), etag);
        assert_eq!(sealed.bytes().await.unwrap(), plain_bytes);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // The other order, on a fresh source: sealed first
        // (with no friendly name at all), plain second.
        let (src2, hits2) = spawn_counting_server(200, "image/png", test_png()).await;
        let remainder2 = format!("200x40/{src2}");
        let sealed = client()
            .get(format!("{base}/{}", sealed_path(&remainder2, None)))
            .send()
            .await
            .unwrap();
        assert_eq!(sealed.status(), 200);
        let sealed_bytes = sealed.bytes().await.unwrap();
        let plain = client()
            .get(format!("{base}/{}/{remainder2}", sign(&remainder2)))
            .send()
            .await
            .unwrap();
        assert_eq!(plain.status(), 200);
        assert_eq!(header(&plain, "cache-status"), "crabcrop; hit");
        assert_eq!(plain.bytes().await.unwrap(), sealed_bytes);
        assert_eq!(hits2.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_changed_source_rerenders_with_a_new_etag_after_purge() {
        // The ETag names the rendered bytes, not the cache
        // key: when the source changes and the entry is
        // purged, a request revalidating with the previous
        // ETag must get the new render in full (200, new
        // bytes, a new ETag), never a 304 for an image it
        // does not hold.
        let (src, hits, current) = spawn_swappable_server(test_png()).await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let first = client().get(&url).send().await.unwrap();
        assert_eq!(first.status(), 200);
        let etag1 = header(&first, "etag");
        let bytes1 = first.bytes().await.unwrap();
        // The stored entry validates as usual.
        let resp = client()
            .get(&url)
            .header("if-none-match", etag1.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 304);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // The source changes under the same URL, and the
        // entry is purged.
        *current.lock().unwrap() = test_png_red();
        let purge_sig = crate::url::purge_signature(SECRET, &remainder);
        let resp = client()
            .post(format!("{base}/purge/{purge_sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert!(resp.text().await.unwrap().contains("purged 1"));
        // Revalidating with the previous ETag gets the new
        // render in full.
        let second = client()
            .get(&url)
            .header("if-none-match", etag1.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(second.status(), 200, "a changed render is never a 304");
        let etag2 = header(&second, "etag");
        assert_ne!(etag2, etag1);
        assert_ne!(second.bytes().await.unwrap(), bytes1);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "the purge forced a re-render"
        );
        // And the new ETag is the validator now.
        let resp = client()
            .get(&url)
            .header("if-none-match", etag2.as_str())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 304);
    }

    #[tokio::test]
    async fn sealed_failures_are_one_uniform_refusal() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let token = crabcrop_url::seal(SECRET, &remainder).unwrap();
        let flipped = {
            let mut t = token.clone().into_bytes();
            let last = t.len() - 1;
            t[last] = if t[last] == b'A' { b'B' } else { b'A' };
            String::from_utf8(t).unwrap()
        };
        let cases = [
            flipped,
            // Minted under a secret this server does not hold.
            crabcrop_url::seal(b"another-secret", &remainder).unwrap(),
            // An unknown key ID routes nowhere.
            format!("zz:{token}"),
            // A known-shaped but wrong key ID opens nothing.
            format!(
                "k2:{}",
                crabcrop_url::seal(b"another-secret", &remainder).unwrap()
            ),
            // Garbage in the segment.
            "s1.!!!".to_string(),
            "s1.AAAA".to_string(),
            // An unknown version marker.
            "s2.AAAA".to_string(),
            // A truncated token.
            token[..token.len() - 2].to_string(),
            // Oversized: refused before decode.
            format!("s1.{}", "A".repeat(5000)),
        ];
        for case in cases {
            let resp = client()
                .get(format!("{base}/{case}/photo.png"))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::FORBIDDEN, "{case}");
            assert_eq!(resp.text().await.unwrap(), "bad signature", "{case}");
        }
        // The server is unharmed and the genuine token
        // still serves.
        let resp = client()
            .get(format!("{base}/{token}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn sealed_payload_failures_are_one_uniform_refusal() {
        // Each of these remainders opens fine (minted under
        // the server's own secret) and then fails a pipeline
        // stage. The plain form answers some of these with
        // instructive 400s; the sealed form must not: the
        // failure's shape would oracle the payload.
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        for remainder in [
            // Grammar junk: a filter argument that is not
            // a number, and a remainder with no source.
            format!("filters:quality(abc)/{src}"),
            "300x50".to_string(),
            // A reserved segment at the payload head.
            "purge/anything".to_string(),
            "upload/heroes/cat.jpg".to_string(),
            // An unknown preset (a grammar error on every
            // server).
            format!("preset:nope/{src}"),
            // An unknown source root.
            "300x50/root:nope/cat.jpg".to_string(),
        ] {
            let resp = client()
                .get(format!("{base}/{}", sealed_path(&remainder, None)))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::FORBIDDEN, "{remainder}");
            assert_eq!(resp.text().await.unwrap(), "bad signature", "{remainder}");
        }
    }

    #[tokio::test]
    async fn sealed_post_open_failures_are_one_uniform_refusal() {
        // Issue #259: the uniform collapse used to end at
        // the load boundary, so load and engine errors
        // leaked the payload (the hidden host, an origin
        // oracle, the source's dimensions). Each remainder
        // below opens fine and fails after the open; the
        // plain form answers instructively, the sealed
        // form must answer the one refusal.
        let (not_found, _hits) = spawn_counting_server(404, "image/png", test_png()).await;
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        for remainder in [
            // Loader policy refusal (userinfo source).
            "300x50/http://user:pass@127.0.0.1:1/img.jpg".to_string(),
            // Origin answers 404.
            format!("300x50/{not_found}"),
            // Engine refusal echoing the op and the
            // hidden source's bounds.
            format!("crop-rect:0x0:99999x99999/{src}"),
        ] {
            let plain = client()
                .get(format!("{base}/{}/{remainder}", sign(&remainder)))
                .send()
                .await
                .unwrap();
            // The plain form is instructive: a policy
            // refusal is a 403 too, but its body names the
            // reason, and the others are 502/400 shapes.
            // None of them is the sealed refusal.
            assert_ne!(plain.text().await.unwrap(), "bad signature", "{remainder}");
            let sealed = client()
                .get(format!("{base}/{}", sealed_path(&remainder, None)))
                .send()
                .await
                .unwrap();
            assert_eq!(sealed.status(), StatusCode::FORBIDDEN, "{remainder}");
            assert_eq!(sealed.text().await.unwrap(), "bad signature", "{remainder}");
        }
    }

    #[tokio::test]
    async fn sealed_expired_answers_signature_expired() {
        // The one distinct sealed answer: the token opened,
        // and its payload is expired, exactly the plain
        // form's answer for the same remainder.
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("exp:1/300x50/{src}");
        let resp = client()
            .get(format!("{base}/{}", sealed_path(&remainder, None)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(resp.text().await.unwrap(), "signature expired");
    }

    #[tokio::test]
    async fn explain_refuses_sealed_spellings_without_fetching() {
        // Explain never unseals (it would be an oracle for
        // lifted tokens): the uniform refusal, and the
        // origin sees nothing.
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let resp = client()
            .get(format!(
                "{base}/explain/{}",
                sealed_path(&remainder, Some("photo.png"))
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(resp.text().await.unwrap(), "bad signature");
        assert_eq!(hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn sealed_meta_marks_the_request_and_hides_the_recipe() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let body = client()
            .get(format!(
                "{base}/{}",
                sealed_path(&remainder, Some("photo.png"))
            ))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        let doc: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(doc["sealed"], true);
        assert_eq!(doc["width"], 300);
        assert_eq!(doc["height"], 50);
        // No operation or source text: the source URL and
        // the friendly name appear nowhere in the document.
        assert!(!body.contains("src.png"), "{body}");
        assert!(!body.contains("photo.png"), "{body}");
        // The plain meta of the same render carries no
        // sealed marker.
        let plain = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        let doc: serde_json::Value = serde_json::from_str(&plain).unwrap();
        assert_eq!(doc["sealed"], serde_json::Value::Null);
        assert_eq!(doc["width"], 300);
    }

    #[tokio::test]
    async fn sealed_and_plain_purges_reach_each_others_entries() {
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let remainder = format!("100x50/{src}");
        let sealed = sealed_path(&remainder, Some("photo.png"));
        // Render plain, purge by the sealed spelling.
        let resp = get_signed(&base, &remainder).await;
        assert_eq!(resp.status(), 200);
        let resp = get_signed(&base, &remainder).await;
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        let purge_sig = crate::url::purge_signature(SECRET, &sealed);
        let resp = client()
            .post(format!("{base}/purge/{purge_sig}/{sealed}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert!(resp.text().await.unwrap().contains("purged 1"));
        let resp = get_signed(&base, &remainder).await;
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        // Render sealed, purge by the plain spelling.
        let resp = client()
            .get(format!("{base}/{sealed}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        let purge_sig = crate::url::purge_signature(SECRET, &remainder);
        let resp = client()
            .post(format!("{base}/purge/{purge_sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert!(resp.text().await.unwrap().contains("purged 1"));
        let resp = client()
            .get(format!("{base}/{sealed}"))
            .send()
            .await
            .unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    }

    #[tokio::test]
    async fn malformed_expiry_is_a_bad_request() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        // Signed properly over a malformed expiry: verification
        // passes, parsing rejects it.
        let remainder = format!("exp:soon/300x50/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn maxage_overrides_the_response_cache_lifetime() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        // The directive substitutes its own max-age for the
        // immutable year, on this response only.
        let resp = get_signed(&base, &format!("maxage:60/300x50/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            header(&resp, "cache-control"),
            "public, max-age=60, immutable"
        );
        // The cap itself spells exactly today's header.
        let resp = get_signed(&base, &format!("maxage:31536000/300x50/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            header(&resp, "cache-control"),
            "public, max-age=31536000, immutable"
        );
    }

    #[tokio::test]
    async fn absent_maxage_keeps_the_default_header_byte_for_byte() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let resp = get_signed(&base, &format!("300x50/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            header(&resp, "cache-control"),
            "public, max-age=31536000, immutable"
        );
    }

    #[tokio::test]
    async fn maxage_zero_means_revalidate() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        // Zero keeps `public` but drops `immutable`: a response
        // that is stale on arrival cannot also promise never
        // to change.
        let resp = get_signed(&base, &format!("maxage:0/300x50/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(header(&resp, "cache-control"), "public, max-age=0");
    }

    #[tokio::test]
    async fn malformed_maxage_is_a_bad_request() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        // Signed properly over a malformed or out-of-range
        // lifetime: verification passes, parsing rejects it.
        for remainder in [
            format!("maxage:soon/300x50/{src}"),
            format!("maxage:31536001/300x50/{src}"),
            format!("maxage:-1/300x50/{src}"),
        ] {
            let resp = client()
                .get(format!("{base}/{}/{remainder}", sign(&remainder)))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{remainder}");
        }
        // The 400 body is the parser's own message.
        let remainder = format!("maxage:soon/300x50/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.text().await.unwrap(),
            "invalid maxage segment: maxage:soon"
        );
    }

    #[tokio::test]
    async fn maxage_variants_share_one_cache_entry() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        // The lifetime is response dressing, not content: the
        // second spelling is a hit on the first one's entry,
        // and each response still carries its own header.
        let first = get_signed(&base, &format!("maxage:60/300x50/{src}")).await;
        assert_eq!(header(&first, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(
            header(&first, "cache-control"),
            "public, max-age=60, immutable"
        );
        let first_bytes = first.bytes().await.unwrap();
        let second = get_signed(&base, &format!("maxage:120/300x50/{src}")).await;
        assert_eq!(header(&second, "cache-status"), "crabcrop; hit");
        assert_eq!(
            header(&second, "cache-control"),
            "public, max-age=120, immutable"
        );
        assert_eq!(second.bytes().await.unwrap(), first_bytes);
        // The plain spelling is the same entry again.
        let third = get_signed(&base, &format!("300x50/{src}")).await;
        assert_eq!(header(&third, "cache-status"), "crabcrop; hit");
        assert_eq!(
            header(&third, "cache-control"),
            "public, max-age=31536000, immutable"
        );
    }

    #[tokio::test]
    async fn debug_focal_bypasses_the_cache_in_both_directions() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let plain = format!("crop:50x50:smart/{src}");
        let debug = format!("debug:focal/crop:50x50:smart/{src}");
        // Populate the cache with the plain result.
        let first = get_signed(&base, &plain).await;
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(header(&first, "cache-status"), "crabcrop; fwd=uri-miss");
        let first_bytes = first.bytes().await.unwrap();
        let second = get_signed(&base, &plain).await;
        assert_eq!(header(&second, "cache-status"), "crabcrop; hit");
        assert_eq!(second.bytes().await.unwrap(), first_bytes);
        // The debug request is not served the cached entry:
        // it computes the diagnostic (the 100x60 canvas the
        // crop ran on, not the 50x50 result) and serves it
        // no-store.
        let dbg = get_signed(&base, &debug).await;
        assert_eq!(dbg.status(), StatusCode::OK);
        assert_eq!(header(&dbg, "cache-control"), "no-store");
        assert_eq!(header(&dbg, "cache-status"), "crabcrop; fwd=uri-miss");
        let dbg_bytes = dbg.bytes().await.unwrap();
        assert_ne!(dbg_bytes, first_bytes);
        let img = image::load_from_memory(&dbg_bytes).unwrap();
        assert_eq!((img.width(), img.height()), (100, 60));
        // A repeat debug request recomputes the identical
        // diagnostic; it was never stored, so still no hit.
        let dbg2 = get_signed(&base, &debug).await;
        assert_eq!(header(&dbg2, "cache-control"), "no-store");
        assert_eq!(header(&dbg2, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(dbg2.bytes().await.unwrap(), dbg_bytes);
        // The plain entry was never poisoned by the debug
        // responses: still the true result, still a hit.
        let third = get_signed(&base, &plain).await;
        assert_eq!(header(&third, "cache-status"), "crabcrop; hit");
        assert_eq!(third.bytes().await.unwrap(), first_bytes);
    }

    #[tokio::test]
    async fn malformed_debug_is_a_bad_request() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        // Signed properly over an unknown debug view:
        // verification passes, parsing rejects it, and the
        // 400 body is the parser's own message.
        let remainder = format!("debug:banana/300x50/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            resp.text().await.unwrap(),
            "invalid debug segment: debug:banana"
        );
    }

    #[tokio::test]
    async fn meta_is_not_available_for_debug_output() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("debug:focal/crop:50x50:smart/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp.text().await.unwrap();
        assert!(
            body.contains("meta is not available for debug output"),
            "body: {body}"
        );
    }

    #[tokio::test]
    async fn preset_urls_take_maxage_as_the_callers_choice() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_presets(preset_test_map(), false).await;
        // Without the directive a preset response is today's
        // immutable year (Q5: presets are stable templates).
        let resp = get_signed(&base, &format!("preset:small/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            header(&resp, "cache-control"),
            "public, max-age=31536000, immutable"
        );
        // With it, the caller's explicit lifetime wins for
        // that response; the preset itself is untouched.
        let resp = get_signed(&base, &format!("maxage:30/preset:small/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            header(&resp, "cache-control"),
            "public, max-age=30, immutable"
        );
    }

    #[tokio::test]
    async fn grammar_rejections_carry_the_parser_message() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        // The 400 body is the parser's own message, so a
        // signed-URL bug is debuggable from the response
        // alone (review AP-2).
        for (remainder, want) in [
            (
                format!("crop:300x0/{src}"),
                "invalid crop segment: crop:300x0".to_string(),
            ),
            (
                format!("filters:bogus(1)/{src}"),
                "invalid filters segment: filters:bogus(1)".to_string(),
            ),
            (
                format!("exp:soon/300x50/{src}"),
                "invalid expiry segment: exp:soon".to_string(),
            ),
        ] {
            let resp = client()
                .get(format!("{base}/{}/{remainder}", sign(&remainder)))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{remainder}");
            assert_eq!(resp.text().await.unwrap(), want, "{remainder}");
        }
        // Preset expansion failures name the preset.
        let base = spawn_app_with_presets(preset_test_map(), false).await;
        let remainder = format!("preset:nope/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(resp.text().await.unwrap(), "unknown preset: nope");
    }

    #[tokio::test]
    async fn parse_failures_speak_json_when_asked() {
        // The v2 contract: a client asking for JSON gets
        // {"error":{"code","message"}} with the frozen code;
        // everyone else keeps the plain-text body.
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:bogus(1)/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(&url)
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "invalid_filters");
        assert_eq!(
            body["error"]["message"],
            "invalid filters segment: filters:bogus(1)"
        );
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            resp.text().await.unwrap(),
            "invalid filters segment: filters:bogus(1)"
        );
    }

    async fn error_body_text(resp: Response) -> String {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    /// Assert one error response in both contract modes:
    /// the exact legacy plain-text body with no JSON
    /// Accept, and the v2 envelope carrying the class's
    /// frozen code (and the same message) with one.
    async fn assert_error_modes(
        resp_plain: reqwest::Response,
        resp_json: reqwest::Response,
        status: StatusCode,
        code: &str,
        message: &str,
    ) {
        assert_eq!(resp_plain.status(), status);
        assert_eq!(resp_plain.text().await.unwrap(), message);
        assert_eq!(resp_json.status(), status);
        let body: serde_json::Value =
            serde_json::from_str(&resp_json.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], code);
        assert_eq!(body["error"]["message"], message);
    }

    fn load_error_cases() -> Vec<(LoadError, StatusCode, &'static str, String)> {
        vec![
            (
                LoadError::Forbidden("a policy reason".to_string()),
                StatusCode::FORBIDDEN,
                "source_forbidden",
                "source refused by policy".to_string(),
            ),
            (
                LoadError::NotFound,
                StatusCode::NOT_FOUND,
                "source_not_found",
                "source not found".to_string(),
            ),
            (
                LoadError::BadStatus(503),
                StatusCode::BAD_GATEWAY,
                "source_bad_status",
                "source returned status 503".to_string(),
            ),
            (
                LoadError::Network("a cause".to_string()),
                StatusCode::BAD_GATEWAY,
                "source_fetch_failed",
                "source fetch failed".to_string(),
            ),
            (
                LoadError::WrongContentType(Some("text/html".to_string())),
                StatusCode::BAD_GATEWAY,
                "source_not_an_image",
                "source did not return an image".to_string(),
            ),
            (
                LoadError::Timeout,
                StatusCode::GATEWAY_TIMEOUT,
                "source_timeout",
                "source timed out".to_string(),
            ),
            (
                LoadError::TooLarge { limit: 10 },
                StatusCode::BAD_GATEWAY,
                "source_too_large",
                "source exceeds 10 bytes".to_string(),
            ),
        ]
    }

    fn engine_error_cases() -> Vec<(EngineError, StatusCode, &'static str, String)> {
        vec![
            (
                EngineError::Decode("a cause".to_string()),
                StatusCode::BAD_GATEWAY,
                "decode_failed",
                "source image could not be decoded".to_string(),
            ),
            (
                EngineError::Encode("a cause".to_string()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "encode_failed",
                "image encoding failed".to_string(),
            ),
            (
                EngineError::Timeout,
                StatusCode::GATEWAY_TIMEOUT,
                "processing_timeout",
                "processing timed out".to_string(),
            ),
            (
                EngineError::Unsupported("a what".to_string()),
                StatusCode::BAD_REQUEST,
                "unsupported_operation",
                "a what".to_string(),
            ),
            (
                EngineError::NotConfigured("a what".to_string()),
                StatusCode::NOT_IMPLEMENTED,
                "not_configured",
                "a what".to_string(),
            ),
            (
                EngineError::Inference(crate::ai::AiError::BadOutput("a cause".to_string())),
                StatusCode::BAD_GATEWAY,
                "inference_bad_output",
                "model produced unexpected output".to_string(),
            ),
            (
                EngineError::Inference(crate::ai::AiError::Inference("a cause".to_string())),
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
                "image inference failed".to_string(),
            ),
            (
                EngineError::Inference(crate::ai::AiError::LockPoisoned),
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
                "image inference failed".to_string(),
            ),
            (
                EngineError::Render("a cause".to_string()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "render_failed",
                "image rendering failed".to_string(),
            ),
            (
                EngineError::Video("a cause".to_string()),
                StatusCode::BAD_GATEWAY,
                "video_encode_failed",
                "video encoding failed".to_string(),
            ),
            (
                EngineError::Frame("a cause".to_string()),
                StatusCode::BAD_GATEWAY,
                "frame_extract_failed",
                "video frame extraction failed".to_string(),
            ),
            (
                EngineError::Pdf("a cause".to_string()),
                StatusCode::BAD_GATEWAY,
                "pdf_render_failed",
                "PDF rendering failed".to_string(),
            ),
        ]
    }

    #[tokio::test]
    async fn operational_error_classes_carry_stable_codes() {
        // Issue #356: every operational failure class
        // answers with its frozen code in the v2 envelope
        // when the client asked for JSON, and with the
        // exact legacy plain-text body when it did not.
        // The classes are exercised at the builders here;
        // the ones that trigger cheaply over HTTP are also
        // covered end to end below (a source timeout needs
        // a ten-second origin, a shed a saturated gate, a
        // processing panic a panicking job). The builders
        // consume the error, so each mode runs its own
        // pass over fresh cases.
        for (e, status, _code, message) in load_error_cases() {
            let resp = load_error_response(e, None);
            assert_eq!(resp.status(), status);
            assert_eq!(error_body_text(resp).await, message);
        }
        for (e, status, code, message) in load_error_cases() {
            let resp = load_error_response(e, Some("application/json"));
            assert_eq!(resp.status(), status);
            let body: serde_json::Value =
                serde_json::from_str(&error_body_text(resp).await).unwrap();
            assert_eq!(body["error"]["code"], code);
            assert_eq!(body["error"]["message"], message);
        }
        for (e, status, _code, message) in engine_error_cases() {
            let resp = engine_error_response(e, None);
            assert_eq!(resp.status(), status);
            assert_eq!(error_body_text(resp).await, message);
        }
        for (e, status, code, message) in engine_error_cases() {
            let resp = engine_error_response(e, Some("application/json"));
            assert_eq!(resp.status(), status);
            let body: serde_json::Value =
                serde_json::from_str(&error_body_text(resp).await).unwrap();
            assert_eq!(body["error"]["code"], code);
            assert_eq!(body["error"]["message"], message);
        }
        // The shed class keeps its Retry-After in both
        // modes.
        let resp = busy_response("server busy", 5, Some("application/json"));
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers().get(header::RETRY_AFTER).unwrap(), "5");
        let body: serde_json::Value = serde_json::from_str(&error_body_text(resp).await).unwrap();
        assert_eq!(body["error"]["code"], "server_busy");
        assert_eq!(body["error"]["message"], "server busy");
    }

    #[test]
    fn every_error_code_is_documented() {
        // The published list and the code stay in
        // agreement (the AGENTS.md rule), in both
        // directions: every code the server can emit,
        // grammar or operational, appears in
        // docs/responses.md, and every operational code
        // the published table names is one the code
        // derives. The sets are derived from the code
        // under test instead of enumerated by hand, so a
        // code added on either side without the other
        // fails here. The grammar codes are the literals
        // UrlError::code returns. The operational codes
        // are the literals passed in argument position in
        // this module's production code, the shape every
        // envelope builder call site takes; the shape
        // filter (snake_case, an underscore, at least
        // five characters) keeps the neighboring literals
        // that are not codes (JSON keys, header values,
        // meta kinds) out of the set. The scanned files
        // are server.rs plus whatever lives in
        // src/server/, each cut at its test module: a
        // split that moves code into a new file there
        // joins the scan by construction, never by a
        // hand-kept list going stale.
        fn snake_literals(source: &str) -> Vec<(usize, usize)> {
            // (open, close) byte indexes of the quotes of
            // "..." literals whose whole content is
            // snake_case.
            let bytes = source.as_bytes();
            let mut spans = Vec::new();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] != b'"' {
                    i += 1;
                    continue;
                }
                let mut j = i + 1;
                while j < bytes.len()
                    && (bytes[j].is_ascii_lowercase()
                        || bytes[j].is_ascii_digit()
                        || bytes[j] == b'_')
                {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'"' && j > i + 1 {
                    spans.push((i, j));
                    i = j + 1;
                } else {
                    i += 1;
                }
            }
            spans
        }
        // One file's production head: its text up to the
        // test module, when it has one.
        fn production_head(text: &str) -> &str {
            match text.find("#[cfg(test)]\nmod tests") {
                Some(cut) => &text[..cut],
                None => text,
            }
        }
        let mut codes: Vec<String> = Vec::new();
        let url_rs = include_str!("url.rs");
        let fn_start = url_rs.find("pub fn code(&self)").expect("UrlError::code");
        let fn_rest = &url_rs[fn_start..];
        let fn_end = fn_rest.find("\n}\n").expect("end of UrlError::code");
        for (open, close) in snake_literals(&fn_rest[..fn_end]) {
            codes.push(fn_rest[open + 1..close].to_string());
        }
        // Two literals in argument position are not
        // envelope codes: timed_out is a meta unavailable
        // reason (the ModerationMeta, EmbeddingMeta, and
        // OcrMeta unavailable payloads), and
        // dimension_mismatch is an aesthetic reason. Both
        // are meta section vocabulary, never envelopes.
        const NOT_CODES: &[&str] = &["timed_out", "dimension_mismatch"];
        let mut heads: Vec<String> = vec![include_str!("server.rs").to_string()];
        let server_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/server");
        let mut module_files: Vec<std::path::PathBuf> = std::fs::read_dir(server_dir)
            .expect("src/server exists")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
            .collect();
        module_files.sort();
        for path in module_files {
            heads.push(std::fs::read_to_string(&path).expect("read a server module file"));
        }
        for head in &heads {
            let head = production_head(head);
            let head_bytes = head.as_bytes();
            for (open, close) in snake_literals(head) {
                let code = &head[open + 1..close];
                if code.len() < 5 || !code.contains('_') || NOT_CODES.contains(&code) {
                    continue;
                }
                let mut prev = open;
                while prev > 0 && matches!(head_bytes[prev - 1], b' ' | b'\t' | b'\n') {
                    prev -= 1;
                }
                if prev == 0 {
                    continue;
                }
                let prev = prev - 1;
                let mut next = close + 1;
                while next < head_bytes.len() && matches!(head_bytes[next], b' ' | b'\t' | b'\n') {
                    next += 1;
                }
                if next < head_bytes.len()
                    && matches!(head_bytes[prev], b',' | b'(')
                    && matches!(head_bytes[next], b',' | b')')
                {
                    codes.push(code.to_string());
                }
            }
        }
        let mut distinct: Vec<String> = codes;
        distinct.sort();
        distinct.dedup();
        // The derivation must find the set it derived
        // today in bulk; a broken extraction fails loudly
        // here instead of vacuously passing. The floor is
        // on the deduplicated set (63 codes today), so a
        // file dropping out of the scan shows up as codes
        // going missing, not as duplicate slack.
        assert!(
            distinct.len() >= 60,
            "derived only {} error codes from the sources",
            distinct.len()
        );
        let docs = include_str!("../docs/responses.md");
        for code in &distinct {
            assert!(
                docs.contains(&format!("`{code}`")),
                "error code {code} is not documented in docs/responses.md"
            );
        }
        // The other direction: the operational table in
        // the docs (rows of the form | `code` | `NNN` |)
        // names only codes the derivation finds, so a
        // documented code the server no longer emits, or
        // one a split moved out of the scan, fails here
        // too. At the time of writing the two sets agree:
        // 39 documented operational codes, all derived.
        let mut documented: Vec<&str> = Vec::new();
        for line in docs.lines() {
            let Some(rest) = line.trim().strip_prefix("| `") else {
                continue;
            };
            let Some((code, rest)) = rest.split_once("` | `") else {
                continue;
            };
            let status = &rest[..rest.len().min(4)];
            if status.len() == 4
                && status[..3].chars().all(|c| c.is_ascii_digit())
                && status[3..].starts_with('`')
            {
                documented.push(code);
            }
        }
        assert!(
            !documented.is_empty(),
            "parsed no operational rows from docs/responses.md"
        );
        for code in &documented {
            assert!(
                distinct.iter().any(|derived| derived == code),
                "documented error code {code} is not derived from the sources"
            );
        }
    }

    #[tokio::test]
    async fn operational_failures_speak_json_when_asked() {
        // End to end over HTTP: the load and engine
        // families, plus the signature classes, answer
        // with the envelope when asked and keep the exact
        // plain-text bodies when not.
        let base = spawn_app(false).await;
        let src = spawn_image_server().await;
        // Bad signature.
        let remainder = format!("100x50/{src}");
        let url = format!("{base}/deadbeef/{remainder}");
        let plain = client().get(&url).send().await.unwrap();
        let json = client()
            .get(&url)
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_error_modes(
            plain,
            json,
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature",
        )
        .await;
        // Expired signed URL.
        let remainder = format!("exp:1/300x50/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let plain = client().get(&url).send().await.unwrap();
        let json = client()
            .get(&url)
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_error_modes(
            plain,
            json,
            StatusCode::FORBIDDEN,
            "signature_expired",
            "signature expired",
        )
        .await;
        // The origin answers 404.
        let (src404, _hits) = spawn_counting_server(404, "image/png", test_png()).await;
        let remainder = format!("100x50/{src404}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let plain = client().get(&url).send().await.unwrap();
        let json = client()
            .get(&url)
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_error_modes(
            plain,
            json,
            StatusCode::NOT_FOUND,
            "source_not_found",
            "source not found",
        )
        .await;
        // The origin answers a non-2xx status.
        let (src500, _hits) = spawn_counting_server(500, "image/png", test_png()).await;
        let remainder = format!("100x50/{src500}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let json = client()
            .get(&url)
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(json.status(), StatusCode::BAD_GATEWAY);
        let body: serde_json::Value = serde_json::from_str(&json.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "source_bad_status");
        assert_eq!(body["error"]["message"], "source returned status 500");
        // The origin answers a non-image Content-Type.
        let (src_html, _hits) =
            spawn_counting_server(200, "text/html", b"<html></html>".to_vec()).await;
        let remainder = format!("100x50/{src_html}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let json = client()
            .get(&url)
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(json.status(), StatusCode::BAD_GATEWAY);
        let body: serde_json::Value = serde_json::from_str(&json.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "source_not_an_image");
        // The fetched bytes do not decode.
        let (src_bad, _hits) =
            spawn_counting_server(200, "image/png", b"not an image".to_vec()).await;
        let remainder = format!("100x50/{src_bad}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let json = client()
            .get(&url)
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(json.status(), StatusCode::BAD_GATEWAY);
        let body: serde_json::Value = serde_json::from_str(&json.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "decode_failed");
        // A feature this server was not configured for.
        let remainder = format!("filters:blurfaces/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let json = client()
            .get(&url)
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(json.status(), StatusCode::NOT_IMPLEMENTED);
        let body: serde_json::Value = serde_json::from_str(&json.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "not_configured");
    }

    #[tokio::test]
    async fn purge_failures_speak_json_when_asked() {
        let base = spawn_app(false).await;
        let src = spawn_image_server().await;
        let target = format!("300x200/{src}");
        let url = format!("{base}/purge/{}/{target}", "0".repeat(64));
        let plain = client().post(&url).send().await.unwrap();
        let json = client()
            .post(&url)
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_error_modes(
            plain,
            json,
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature",
        )
        .await;
    }

    #[tokio::test]
    async fn upload_failures_speak_json_when_asked() {
        let dir = upload_dir("codes");
        let base = spawn_app_with_upload(&dir, 1024 * 1024, false).await;
        let name = "cat.png";
        // Bad signature.
        let url = format!("{base}/upload/{}/{name}", "0".repeat(64));
        let json = client()
            .put(&url)
            .header("accept", "application/json")
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(json.status(), StatusCode::FORBIDDEN);
        let body: serde_json::Value = serde_json::from_str(&json.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "bad_signature");
        // A body that is not an image.
        let sig = crate::url::upload_signature(SECRET, name);
        let url = format!("{base}/upload/{sig}/{name}");
        let json = client()
            .put(&url)
            .header("accept", "application/json")
            .body(b"not an image".to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(json.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        let body: serde_json::Value = serde_json::from_str(&json.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "upload_not_an_image");
        // A duplicate name with overwriting off.
        let resp = client().put(&url).body(test_png()).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = client()
            .put(&url)
            .header("accept", "application/json")
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(json.status(), StatusCode::CONFLICT);
        let body: serde_json::Value = serde_json::from_str(&json.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "upload_already_exists");
        // A body past the cap.
        let dir = upload_dir("codes-oversize");
        let base = spawn_app_with_upload(&dir, 16, false).await;
        let url = format!("{base}/upload/{sig}/{name}");
        let json = client()
            .put(&url)
            .header("accept", "application/json")
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(json.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body: serde_json::Value = serde_json::from_str(&json.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "upload_too_large");
    }

    #[tokio::test]
    async fn metrics_failures_speak_json_when_asked() {
        // No token configured: metrics are disabled.
        let base = spawn_app_with_metrics(false, None).await;
        let json = client()
            .get(format!("{base}/metrics"))
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(json.status(), StatusCode::FORBIDDEN);
        let body: serde_json::Value = serde_json::from_str(&json.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "metrics_disabled");
        // A token configured but the wrong bearer sent.
        let base = spawn_app_with_metrics(false, Some(TEST_METRICS_TOKEN.to_string())).await;
        let json = client()
            .get(format!("{base}/metrics"))
            .header("accept", "application/json")
            .header("authorization", "Bearer wrong")
            .send()
            .await
            .unwrap();
        assert_eq!(json.status(), StatusCode::UNAUTHORIZED);
        let body: serde_json::Value = serde_json::from_str(&json.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "metrics_unauthorized");
    }

    #[tokio::test]
    async fn unconfigured_features_answer_501_not_400() {
        // The URL is fine; this deployment has no face
        // model. 501 says the server cannot do it, where
        // 400 would blame the request.
        let (src, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:blurfaces/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        assert!(
            resp.text()
                .await
                .unwrap()
                .contains("face operations are not configured on this server")
        );
    }

    fn preset_test_map() -> std::collections::HashMap<String, String> {
        [
            ("small".to_string(), "30x20".to_string()),
            ("gray".to_string(), "filters:grayscale".to_string()),
        ]
        .into_iter()
        .collect()
    }

    async fn spawn_app_with_presets(
        presets: std::collections::HashMap<String, String>,
        only_presets: bool,
    ) -> String {
        spawn_app_with_presets_and_roots(presets, only_presets, std::collections::HashMap::new())
            .await
    }

    /// The root test map points the `photos` root at a test
    /// origin: the base is the origin with `/src.png`
    /// stripped, so `root:photos/src.png` expands to the
    /// origin's own URL.
    fn root_test_map(origin: &str) -> std::collections::HashMap<String, String> {
        let base = origin.strip_suffix("/src.png").unwrap_or(origin);
        [("photos".to_string(), base.to_string())]
            .into_iter()
            .collect()
    }

    async fn spawn_app_with_presets_and_roots(
        presets: std::collections::HashMap<String, String>,
        only_presets: bool,
        source_roots: std::collections::HashMap<String, String>,
    ) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets,
            only_presets,
            source_roots,
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn get_signed(base: &str, remainder: &str) -> reqwest::Response {
        client()
            .get(format!("{base}/{}/{remainder}", sign(remainder)))
            .send()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn preset_urls_serve_the_expanded_transform() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_presets(preset_test_map(), false).await;
        // The source is 100x60; the preset resizes to 30x20.
        let resp = get_signed(&base, &format!("preset:small/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.bytes().await.unwrap();
        let img = image::load_from_memory(&body).unwrap();
        assert_eq!((img.width(), img.height()), (30, 20));
    }

    #[tokio::test]
    async fn preset_urls_and_explicit_chains_serve_identical_bytes() {
        // Issue #353: a preset spelling and the explicit
        // chain it expands to are the same request: identical
        // bytes either way, and the second spelling is served
        // from the first's cache entry (one origin hit).
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let mut presets = preset_test_map();
        presets.insert(
            "card".to_string(),
            "crop:40x30:smart/filters:quality(80)".to_string(),
        );
        let base = spawn_app_with_presets(presets, false).await;
        let chains = [
            ("preset:small", "30x20"),
            ("preset:gray", "filters:grayscale"),
            ("preset:card", "crop:40x30:smart/filters:quality(80)"),
        ];
        let mut want_hits = 0;
        for (preset, chain) in chains {
            want_hits += 1;
            let via_preset = get_signed(&base, &format!("{preset}/{src}")).await;
            assert_eq!(via_preset.status(), StatusCode::OK, "{preset}");
            let preset_bytes = via_preset.bytes().await.unwrap();
            let via_chain = get_signed(&base, &format!("{chain}/{src}")).await;
            assert_eq!(via_chain.status(), StatusCode::OK, "{chain}");
            let chain_bytes = via_chain.bytes().await.unwrap();
            assert_eq!(preset_bytes, chain_bytes, "{preset} vs {chain}");
            assert_eq!(
                hits.load(Ordering::SeqCst),
                want_hits,
                "{preset} vs {chain}"
            );
        }
    }

    #[tokio::test]
    async fn generated_corpus_error_codes_match_the_http_envelope() {
        // Issue #353: one generated remainder per error code
        // in spec/operation-vectors.json, re-targeted at the
        // test origin (the parse fails before any fetch): the
        // HTTP layer must answer 400 with the same code the
        // corpus pins for the parser.
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let doc: serde_json::Value =
            serde_json::from_str(include_str!("../spec/operation-vectors.json")).unwrap();
        let mut seen = std::collections::HashSet::new();
        let mut checked = 0;
        for case in doc["cases"].as_array().unwrap() {
            if case["ok"].as_bool().unwrap() {
                continue;
            }
            let code = case["error"].as_str().unwrap();
            if !seen.insert(code.to_string()) {
                continue;
            }
            let remainder = case["remainder"].as_str().unwrap();
            let ops = remainder
                .strip_suffix("/https://example.com/a.jpg")
                .unwrap_or_else(|| panic!("error case without the corpus source: {remainder}"));
            let remainder = format!("{ops}/{src}");
            let url = format!("{base}/{}/{remainder}", sign(&remainder));
            let resp = client()
                .get(&url)
                .header("accept", "application/json")
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{remainder}");
            let body: serde_json::Value =
                serde_json::from_str(&resp.text().await.unwrap()).unwrap();
            assert_eq!(body["error"]["code"], code, "{remainder}");
            checked += 1;
        }
        assert!(checked >= 12, "only {checked} error codes sampled");
    }

    /// The differential corpora (issue #353): the
    /// hand-written explainer verdicts in
    /// `spec/explain-verdicts.json` and the generated
    /// combinations in `spec/operation-vectors.json`. Every
    /// surface that reads a remainder (the Rust parser, the
    /// TypeScript explainer, the playground's copy) runs the
    /// same files in its own suite; a disagreement anywhere
    /// fails the gate here or there, and the mismatch list is
    /// printed whole so one run shows every divergence.
    /// These tests were put here rather than in url.rs while
    /// the fuzz crate included url.rs textually, without the
    /// explain module or serde_json in scope; since the
    /// library target (issue #417) nothing keeps them here.
    fn load_spec(name: &str) -> serde_json::Value {
        let text = match name {
            "explain-verdicts.json" => include_str!("../spec/explain-verdicts.json"),
            "operation-vectors.json" => include_str!("../spec/operation-vectors.json"),
            _ => unreachable!(),
        };
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn differential_verdict_corpus_agrees_with_the_parser() {
        let doc = load_spec("explain-verdicts.json");
        let mut bad = Vec::new();
        for case in doc["cases"].as_array().unwrap() {
            let remainder = case["remainder"].as_str().unwrap();
            let expect_error = case["error"].as_bool().unwrap();
            let expect_source = case["source"].as_str().unwrap();
            match crate::url::parse_operations_with_expiry(remainder) {
                Ok((_, source, _, _)) => {
                    if expect_error {
                        bad.push(format!("{remainder}: corpus says error, parser accepts"));
                    } else if source != expect_source {
                        bad.push(format!(
                            "{remainder}: source {source:?} != corpus {expect_source:?}"
                        ));
                    }
                }
                Err(e) => {
                    if !expect_error {
                        bad.push(format!(
                            "{remainder}: corpus says ok, parser errors {}",
                            e.code()
                        ));
                    }
                }
            }
        }
        assert!(
            bad.is_empty(),
            "{} mismatches:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// The explainer's per-operation segments for a parsed
    /// chain, with adjacent `filters:` entries merged the way
    /// [`canonical_form`] merges them, so the two walks over
    /// the chain can be compared segment for segment.
    fn explained_segments(ops: &[crate::url::Operation]) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for entry in crate::explain::op_entries(ops) {
            assert!(
                entry.descriptions.iter().all(|d| !d.is_empty()),
                "an operation with an empty gloss: {}",
                entry.segment
            );
            match (out.last_mut(), entry.segment.strip_prefix("filters:")) {
                (Some(last), Some(chain)) if last.starts_with("filters:") => {
                    last.push(':');
                    last.push_str(chain);
                }
                _ => out.push(entry.segment),
            }
        }
        out
    }

    #[test]
    fn differential_generated_cases_agree_across_surfaces() {
        let doc = load_spec("operation-vectors.json");
        let cases = doc["cases"].as_array().unwrap();
        assert!(
            cases.len() > 500,
            "the generated corpus shrank: {}",
            cases.len()
        );
        let mut bad = Vec::new();
        for case in cases {
            let remainder = case["remainder"].as_str().unwrap();
            if case["ok"].as_bool().unwrap() {
                let canonical = case["canonical"].as_str().unwrap();
                match crate::url::parse_operations_with_expiry(remainder) {
                    Ok((ops, source, _, _)) => {
                        // Verdict and canonical form.
                        let form = crate::url::canonical_form(&ops, source);
                        if form != canonical {
                            bad.push(format!(
                                "{remainder}: canonical {form:?} != corpus {canonical:?}"
                            ));
                            continue;
                        }
                        // The canonical form is a fixed point:
                        // parsing it changes nothing.
                        match crate::url::parse_operations_with_expiry(canonical) {
                            Ok((ops2, source2, _, _)) => {
                                let form2 = crate::url::canonical_form(&ops2, source2);
                                if form2 != canonical {
                                    bad.push(format!(
                                        "{canonical}: not a fixed point, re-canonicalizes to {form2:?}"
                                    ));
                                }
                            }
                            Err(e) => bad.push(format!(
                                "{canonical}: canonical form fails to parse ({})",
                                e.code()
                            )),
                        }
                        // The explainer walks the same chain:
                        // its segments, merged like the
                        // canonicalizer merges, are the
                        // canonical operation segments.
                        let ops_part = if canonical == source {
                            ""
                        } else {
                            canonical
                                .strip_suffix(&format!("/{source}"))
                                .unwrap_or(canonical)
                        };
                        let explained = explained_segments(&ops).join("/");
                        if explained != ops_part {
                            bad.push(format!(
                                "{remainder}: explainer segments {explained:?} != canonical operations {ops_part:?}"
                            ));
                        }
                    }
                    Err(e) => bad.push(format!(
                        "{remainder}: corpus says ok, parser errors {}",
                        e.code()
                    )),
                }
            } else {
                let code = case["error"].as_str().unwrap();
                match crate::url::parse_operations_with_expiry(remainder) {
                    Ok(_) => bad.push(format!("{remainder}: corpus says {code}, parser accepts")),
                    Err(e) => {
                        if e.code() != code {
                            bad.push(format!(
                                "{remainder}: corpus says {code}, parser errors {}",
                                e.code()
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            bad.is_empty(),
            "{} mismatches:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    #[test]
    fn differential_builder_segments_parse_to_their_canonical_form() {
        // The TypeScript builders promise the canonical
        // spelling; the SDK suite checks each call's output
        // against `segment`, and here the same segments must
        // parse to the recorded canonical form (or fail with
        // the recorded code, for calls the builder refuses).
        let doc = load_spec("operation-vectors.json");
        let src = "https://example.com/a.jpg";
        let mut bad = Vec::new();
        for case in doc["builder_cases"].as_array().unwrap() {
            if case["rust"].as_bool() == Some(false) {
                continue;
            }
            let call = case["call"].as_str().unwrap();
            let segment = case["segment"].as_str().unwrap();
            let wrap = |text: &str| {
                if call.starts_with("filter.") {
                    format!("filters:{text}")
                } else {
                    text.to_string()
                }
            };
            let remainder = format!("{}/{src}", wrap(segment));
            if case["throws"].as_bool() == Some(true) {
                let code = case["error"].as_str().unwrap();
                match crate::url::parse_operations_with_expiry(&remainder) {
                    Ok(_) => bad.push(format!(
                        "{call} {segment}: corpus says {code}, parser accepts"
                    )),
                    Err(e) => {
                        if e.code() != code {
                            bad.push(format!(
                                "{call} {segment}: corpus says {code}, parser errors {}",
                                e.code()
                            ));
                        }
                    }
                }
            } else {
                let canonical = case["canonical"].as_str().unwrap();
                let want = if canonical.is_empty() {
                    src.to_string()
                } else {
                    format!("{}/{src}", wrap(canonical))
                };
                match crate::url::parse_operations_with_expiry(&remainder) {
                    Ok((ops, source, _, _)) => {
                        let form = crate::url::canonical_form(&ops, source);
                        if form != want {
                            bad.push(format!(
                                "{call} {segment}: canonical {form:?} != corpus {want:?}"
                            ));
                        }
                    }
                    Err(e) => bad.push(format!(
                        "{call} {segment}: corpus says ok, parser errors {}",
                        e.code()
                    )),
                }
            }
        }
        assert!(
            bad.is_empty(),
            "{} mismatches:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    #[test]
    fn preset_spellings_and_their_expansions_share_one_cache_key() {
        // A preset reference and the explicit chain it
        // expands to are one request: the server expands
        // before keying, so both spellings canonicalize to
        // the same cache key form.
        let presets: std::collections::HashMap<String, String> = [
            ("small".to_string(), "30x20".to_string()),
            ("gray".to_string(), "filters:grayscale".to_string()),
            (
                "card".to_string(),
                "crop:1200x630:smart/filters:quality(80)".to_string(),
            ),
        ]
        .into_iter()
        .collect();
        let src = "https://example.com/a.jpg";
        for (name, definition) in &presets {
            let spelled = format!("preset:{name}/{src}");
            let expanded = crate::url::expand_presets(&spelled, &presets, false).unwrap();
            assert_eq!(expanded, format!("{definition}/{src}"));
        }
        // A preset followed by more chain composes with it,
        // and the composed spelling keys like the fully
        // explicit one (adjacent filter segments merge).
        let spelled = format!("preset:gray/filters:blur(2)/{src}");
        let expanded = crate::url::expand_presets(&spelled, &presets, false).unwrap();
        assert_eq!(expanded, format!("filters:grayscale/filters:blur(2)/{src}"));
        assert_eq!(
            crate::url::canonical_key_form(&expanded),
            crate::url::canonical_key_form(&format!("filters:grayscale:blur(2)/{src}")),
        );
        // Expansion failures carry the frozen codes.
        let err =
            crate::url::expand_presets(&format!("preset:nope/{src}"), &presets, false).unwrap_err();
        assert_eq!(err.code(), "unknown_preset");
        let err =
            crate::url::expand_presets(&format!("300x200/{src}"), &presets, true).unwrap_err();
        assert_eq!(err.code(), "presets_only");
    }

    #[tokio::test]
    async fn unknown_presets_are_a_bad_request() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_presets(preset_test_map(), false).await;
        let resp = get_signed(&base, &format!("preset:nope/{src}")).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Issue #247: with NO presets configured, a `preset:`
    /// segment used to skip expansion entirely and reach the
    /// loader as the source (502 `source fetch failed`). Every
    /// name is unknown to such a server, so the answer is the
    /// same 400 `unknown_preset` a configured server gives.
    #[tokio::test]
    async fn preset_segment_without_presets_configured_is_unknown_preset() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_presets(std::collections::HashMap::new(), false).await;
        let remainder = format!("preset:thumb/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "unknown_preset");
        // Plain (non-preset) requests on the same server are
        // unaffected by the always-on expansion.
        let resp = get_signed(&base, &format!("30x20/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn root_sources_serve_the_expanded_origin() {
        let (src, hits) = spawn_counting_image_server().await;
        let base = spawn_app_with_presets_and_roots(
            std::collections::HashMap::new(),
            false,
            root_test_map(&src),
        )
        .await;
        // The source is 100x60; the request resizes to 30x20.
        let resp = get_signed(&base, "30x20/root:photos/src.png").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.bytes().await.unwrap();
        let img = image::load_from_memory(&body).unwrap();
        assert_eq!((img.width(), img.height()), (30, 20));
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn root_and_full_sources_share_one_cache_entry() {
        // Short form first, then the full spelling is a hit.
        let (src, hits) = spawn_counting_image_server().await;
        let base = spawn_app_with_presets_and_roots(
            std::collections::HashMap::new(),
            false,
            root_test_map(&src),
        )
        .await;
        let resp = get_signed(&base, "30x20/root:photos/src.png").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let resp = get_signed(&base, &format!("30x20/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "the full spelling must hit the short spelling's entry"
        );
        // And the reverse, on a fresh app: full first, short hits.
        let (src, hits) = spawn_counting_image_server().await;
        let base = spawn_app_with_presets_and_roots(
            std::collections::HashMap::new(),
            false,
            root_test_map(&src),
        )
        .await;
        let resp = get_signed(&base, &format!("30x20/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let resp = get_signed(&base, "30x20/root:photos/src.png").await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "the short spelling must hit the full spelling's entry"
        );
    }

    #[tokio::test]
    async fn unknown_roots_are_a_bad_request() {
        let (src, _hits) = spawn_counting_image_server().await;
        // A name the server has not configured...
        let base = spawn_app_with_presets_and_roots(
            std::collections::HashMap::new(),
            false,
            root_test_map(&src),
        )
        .await;
        let remainder = "30x20/root:nope/src.png";
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(remainder)))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "unknown_root");
        // ...and every name on a server with no roots at all.
        let base = spawn_app(false).await;
        let remainder = "30x20/root:photos/src.png";
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(remainder)))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "unknown_root");
    }

    /// Send one GET with the path bytes exactly as given:
    /// traversal spellings must reach the server unaltered,
    /// which a URL-normalizing client would prevent.
    async fn get_raw(base: &str, path: &str) -> (u16, String) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let addr = base.strip_prefix("http://").unwrap();
        let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "GET {path} HTTP/1.1\r\nHost: {addr}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
        );
        socket.write_all(request.as_bytes()).await.unwrap();
        let mut raw = String::new();
        socket.read_to_string(&mut raw).await.unwrap();
        let status: u16 = raw
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let body = raw
            .split_once("\r\n\r\n")
            .map(|(_, b)| b.to_string())
            .unwrap_or_default();
        (status, body)
    }

    #[tokio::test]
    async fn root_paths_that_escape_the_root_are_a_bad_request() {
        let (src, hits) = spawn_counting_image_server().await;
        let base = spawn_app_with_presets_and_roots(
            std::collections::HashMap::new(),
            false,
            root_test_map(&src),
        )
        .await;
        for remainder in [
            "30x20/root:photos/../secret.png",
            "30x20/root:photos/%2e%2e/secret.png",
            "30x20/root:photos/a/%2f..%2fsecret.png",
            "30x20/root:photos/..%5csecret.png",
        ] {
            let (status, body) = get_raw(&base, &format!("/{}/{remainder}", sign(remainder))).await;
            assert_eq!(status, 400, "{remainder}: {body}");
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(json["error"]["code"], "invalid_root", "{remainder}");
        }
        assert_eq!(hits.load(Ordering::SeqCst), 0, "nothing was fetched");
        // Dot segments that stay inside the root normalize
        // and serve (raw again: a client would collapse
        // this path before the server ever saw it).
        let remainder = "30x20/root:photos/a/../src.png";
        let (status, _body) = get_raw(&base, &format!("/{}/{remainder}", sign(remainder))).await;
        assert_eq!(status, 200);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    /// Issue #250: the adversarial barrage. Every invalid
    /// root spelling is refused with its frozen code, none
    /// of them reaches the origin, no response carries
    /// operator configuration, and the same process answers
    /// valid requests afterwards: an invalid root is never
    /// a vulnerability and never a crash.
    #[tokio::test]
    async fn invalid_root_barrage_never_fetches_and_the_server_survives() {
        let (src, hits) = spawn_counting_image_server().await;
        let base = spawn_app_with_presets_and_roots(
            std::collections::HashMap::new(),
            false,
            root_test_map(&src),
        )
        .await;
        let origin = src.strip_suffix("/src.png").unwrap().to_string();
        let long_name = "a".repeat(10_000);
        for remainder in [
            "30x20/root:nope/src.png".to_string(),
            "30x20/root:PHOTOS/src.png".to_string(),
            "30x20/root:photo/src.png".to_string(),
            "30x20/root:photoss/src.png".to_string(),
            format!("30x20/root:{long_name}/src.png"),
        ] {
            let (status, body) =
                get_raw(&base, &format!("/{}/{remainder}", sign(&remainder))).await;
            assert_eq!(status, 400, "{remainder}: {body}");
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(json["error"]["code"], "unknown_root", "{remainder}");
            assert!(!body.contains(&origin), "{remainder}: {body}");
        }
        for remainder in [
            "30x20/root:/src.png",
            "30x20/root:photos",
            "30x20/root:photos/..%5csecret.png",
            "30x20/root:photos/%2e%2e%5csecret.png",
            "30x20/root:photos/%/src.png",
            "30x20/root:photos/%ff/src.png",
        ] {
            let (status, body) = get_raw(&base, &format!("/{}/{remainder}", sign(remainder))).await;
            assert_eq!(status, 400, "{remainder}: {body}");
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(json["error"]["code"], "invalid_root", "{remainder}");
            assert!(!body.contains(&origin), "{remainder}: {body}");
        }
        assert_eq!(hits.load(Ordering::SeqCst), 0, "no refusal was fetched");
        // The process is unharmed: a valid root request and
        // a valid full-URL request both render (one shared
        // cache entry, one fetch), and health answers.
        let resp = get_signed(&base, "30x20/root:photos/src.png").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let resp = get_signed(&base, &format!("30x20/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        let resp = client().get(format!("{base}/health")).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Issue #250: a double-encoded traversal expands to a
    /// literal percent-named file under the root. Decoding
    /// happens exactly once, so the fetch is addressed
    /// under the root's base, like any other file name,
    /// and reaches exactly the origin the root names.
    #[tokio::test]
    async fn root_double_encoded_segments_fetch_under_the_root() {
        let (src, hits) = spawn_counting_image_server().await;
        let base = spawn_app_with_presets_and_roots(
            std::collections::HashMap::new(),
            false,
            root_test_map(&src),
        )
        .await;
        let resp = get_signed(&base, "30x20/root:photos/%252e%252e/src.png").await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    /// Issue #250: explain and meta refuse invalid roots
    /// the same way serve does, before any fetch: neither
    /// surface is a side door around the expansion checks.
    #[tokio::test]
    async fn explain_and_meta_refuse_invalid_roots_without_fetching() {
        let (src, hits) = spawn_counting_image_server().await;
        let base = spawn_explain_app_with_roots(
            std::collections::HashMap::new(),
            Loader::with_defaults().unwrap(),
            root_test_map(&src),
        )
        .await;
        // Raw requests: a normalizing client would rewrite
        // the traversal spelling before the server saw it.
        for (remainder, code) in [
            ("30x20/root:nope/src.png", "unknown_root"),
            ("30x20/root:photos/../secret.png", "invalid_root"),
        ] {
            let (status, body) =
                get_raw(&base, &format!("/explain/{}/{remainder}", sign(remainder))).await;
            assert_eq!(status, 400, "{remainder}: {body}");
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(json["error"]["code"], code, "{remainder}");
        }
        assert_eq!(hits.load(Ordering::SeqCst), 0, "explain fetched nothing");
        // Meta rides the image path under an
        // application/json Accept: same refusal, same code.
        let base = spawn_app_with_presets_and_roots(
            std::collections::HashMap::new(),
            false,
            root_test_map(&src),
        )
        .await;
        let remainder = "30x20/root:nope/src.png";
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(remainder)))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "unknown_root");
        assert_eq!(hits.load(Ordering::SeqCst), 0, "meta fetched nothing");
    }

    /// POST with the path bytes exactly as given, the
    /// purge twin of get_raw: purge targets with traversal
    /// spellings must reach the server unaltered, which a
    /// URL-normalizing client would prevent.
    async fn post_raw(base: &str, path: &str) -> (u16, String) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let addr = base.strip_prefix("http://").unwrap();
        let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        socket.write_all(request.as_bytes()).await.unwrap();
        let mut raw = String::new();
        socket.read_to_string(&mut raw).await.unwrap();
        let status: u16 = raw
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let body = raw
            .split_once("\r\n\r\n")
            .map(|(_, b)| b.to_string())
            .unwrap_or_default();
        (status, body)
    }

    /// Issue #250: purge is fail-safe against invalid root
    /// spellings. A purge signature is cheap for an
    /// operator to mint over any text; a bad spelling must
    /// evict nothing and disturb nothing.
    #[tokio::test]
    async fn purge_with_an_invalid_root_spelling_evicts_nothing() {
        let (src, hits) = spawn_counting_image_server().await;
        let base = spawn_app_with_presets_and_roots(
            std::collections::HashMap::new(),
            false,
            root_test_map(&src),
        )
        .await;
        let short = "30x20/root:photos/src.png";
        let resp = get_signed(&base, short).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        for target in [
            "30x20/root:nope/src.png",
            "30x20/root:photos/../src.png",
            "30x20/root:photos/%2e%2e/src.png",
            "30x20/root:photos/a/../../src.png",
        ] {
            let purge_sig = crate::url::purge_signature(SECRET, target);
            let (status, body) = post_raw(&base, &format!("/purge/{purge_sig}/{target}")).await;
            assert_eq!(status, 200, "{target}: {body}");
            assert!(body.contains("purged 0"), "{target}: {body}");
        }
        // The cached entry survived every bad spelling.
        let resp = get_signed(&base, short).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 1, "still a cache hit");
    }

    #[tokio::test]
    async fn purging_either_spelling_evicts_the_shared_root_entry() {
        // Render via the short spelling, purge by it, and
        // the full spelling misses afterwards.
        let (src, hits) = spawn_counting_image_server().await;
        let base = spawn_app_with_presets_and_roots(
            std::collections::HashMap::new(),
            false,
            root_test_map(&src),
        )
        .await;
        let short = "30x20/root:photos/src.png";
        let full = format!("30x20/{src}");
        let resp = get_signed(&base, short).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let resp = get_signed(&base, short).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 1, "second fetch is a hit");
        let purge_sig = crate::url::purge_signature(SECRET, short);
        let resp = client()
            .post(format!("{base}/purge/{purge_sig}/{short}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.text().await.unwrap().contains("purged 1"));
        let resp = get_signed(&base, &full).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 2, "the entry was evicted");
        // The other direction: purge by the full spelling
        // evicts the entry the short spelling reads.
        let purge_sig = crate::url::purge_signature(SECRET, &full);
        let resp = client()
            .post(format!("{base}/purge/{purge_sig}/{full}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.text().await.unwrap().contains("purged 1"));
        let resp = get_signed(&base, short).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 3, "the entry was evicted");
    }

    #[tokio::test]
    async fn explain_reports_the_root_name_and_never_the_base() {
        let (src, hits) = spawn_counting_image_server().await;
        let base = spawn_explain_app_with_roots(
            std::collections::HashMap::new(),
            Loader::with_defaults().unwrap(),
            root_test_map(&src),
        )
        .await;
        let origin = src.strip_suffix("/src.png").unwrap().to_string();
        let remainder = "100x50/root:photos/src.png";
        let resp = client()
            .get(format!("{base}/explain/{}/{remainder}", sign(remainder)))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await.unwrap();
        // The root's base URL is operator configuration:
        // the origin it points at appears nowhere.
        assert!(!body.contains(&origin), "{body}");
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["source"]["url"], "root:photos/src.png");
        assert_eq!(json["source"]["root"], "photos");
        assert_eq!(hits.load(Ordering::SeqCst), 1, "one probe fetch");
        // The HTML form names the root too, and leaks the
        // same nothing.
        let resp = client()
            .get(format!("{base}/explain/{}/{remainder}", sign(remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await.unwrap();
        assert!(body.contains("Source root: photos"), "{body}");
        assert!(!body.contains(&origin), "{body}");
    }

    /// An image server that counts how many times it was hit:
    /// how cache behavior becomes observable from outside.
    async fn spawn_counting_image_server() -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::AsyncWriteExt;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        let png = test_png();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                counter.fetch_add(1, Ordering::SeqCst);
                let body = png.clone();
                tokio::spawn(async move {
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = socket.write_all(head.as_bytes()).await;
                    let _ = socket.write_all(&body).await;
                });
            }
        });
        (format!("http://{addr}/src.png"), hits)
    }

    #[tokio::test]
    async fn presets_and_free_form_share_a_cache_entry() {
        use std::sync::atomic::Ordering;
        let (src, hits) = spawn_counting_image_server().await;
        let base = spawn_app_with_presets(preset_test_map(), false).await;
        // The preset request populates the cache under its
        // expanded form...
        let resp = get_signed(&base, &format!("preset:small/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        // ...so the equivalent free-form request is a cache hit
        // and the source is not fetched again.
        let resp = get_signed(&base, &format!("30x20/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn presets_only_mode_gates_free_form_operations() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_presets(preset_test_map(), true).await;
        // Preset references and bare sources are fine.
        let resp = get_signed(&base, &format!("preset:small/{src}")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let resp = get_signed(&base, &src).await;
        assert_eq!(resp.status(), StatusCode::OK);
        // Expiry metadata alongside a preset is fine too.
        let resp = get_signed(
            &base,
            &format!("exp:{}/preset:gray/{src}", now_unix() + 3600),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        // Free-form operations are refused.
        let resp = get_signed(&base, &format!("30x20/{src}")).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let resp = get_signed(&base, &format!("preset:small/filters:grayscale/{src}")).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn removebg_without_a_configured_model_is_a_clean_501() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:removebg:format(png)/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = resp.text().await.unwrap();
        assert!(body.contains("not configured"), "body: {body}");
    }

    #[tokio::test]
    async fn upscale_without_a_configured_model_is_a_clean_501() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:upscale(2)/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = resp.text().await.unwrap();
        assert!(body.contains("not configured"), "body: {body}");
    }

    #[tokio::test]
    async fn remove_region_without_a_configured_model_is_a_clean_501() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:remove_region(0,0,10,10):format(png)/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = resp.text().await.unwrap();
        assert!(
            body.contains("remove_region is not configured"),
            "body: {body}"
        );
    }

    #[tokio::test]
    async fn remove_region_out_of_bounds_is_a_400() {
        // The image server fixture is 100x60: a rectangle
        // overflowing it fails with the region filters'
        // shared bounds error, a 400, not a partial erase.
        let src = spawn_image_server().await;
        let base = spawn_app_with_inpaint().await;
        let remainder = format!("filters:remove_region(95,55,10,10):format(png)/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp.text().await.unwrap();
        assert!(body.contains("exceeds image bounds"), "body: {body}");
    }

    #[tokio::test]
    async fn serves_optimized_image() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let sig = sign(&remainder);
        let resp = client()
            .get(format!("{base}/{sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/png"
        );
        let body = resp.bytes().await.unwrap();
        let img = image::load_from_memory(&body).unwrap();
        assert_eq!((img.width(), img.height()), (300, 50));
    }

    #[tokio::test]
    async fn serves_fit_in() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("fit-in:30x30/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.bytes().await.unwrap();
        let img = image::load_from_memory(&body).unwrap();
        // 100x60 into a 30x30 box: width rules, aspect preserved.
        assert_eq!((img.width(), img.height()), (30, 18));
    }

    #[tokio::test]
    async fn serves_meta_as_json() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("100x50/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        // The real processed output, fetched first (which also
        // populates the result cache) for comparison.
        let img_resp = client().get(&url).send().await.unwrap();
        assert_eq!(
            img_resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/png"
        );
        let img_body = img_resp.bytes().await.unwrap();
        // The meta response for the same URL: JSON describing the
        // same processed output, never the cached image bytes.
        let resp = client()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["width"].as_u64(), Some(100));
        assert_eq!(meta["height"].as_u64(), Some(50));
        assert_eq!(meta["format"], "png");
        assert_eq!(meta["bytes"].as_u64(), Some(img_body.len() as u64));
        assert_eq!(meta["source_width"].as_u64(), Some(100));
        assert_eq!(meta["source_height"].as_u64(), Some(60));
    }

    #[tokio::test]
    async fn meta_requests_do_not_populate_the_image_cache() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("90x45/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        // Meta first: it must not leave an entry the image request
        // could be answered with.
        let resp = client()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/png"
        );
        let body = resp.bytes().await.unwrap();
        assert_eq!(&body[..4], b"\x89PNG");
    }

    #[tokio::test]
    async fn meta_mode_keeps_error_mapping() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("100x50/{src}");
        let resp = client()
            .get(format!("{base}/deadbeef/{remainder}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[derive(Debug)]
    struct FakeTagger;

    impl crate::ai::TagProvider for FakeTagger {
        fn scores(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<Vec<crate::ai::TagScore>, crate::ai::AiError> {
            // Raw scores (issue #152): the composition turns
            // these into one tag (ocean, probability ~1) and
            // the alt text "Ocean, sky, mountain".
            Ok(vec![
                crate::ai::TagScore {
                    label: "ocean".to_string(),
                    score: 0.30,
                },
                crate::ai::TagScore {
                    label: "sky".to_string(),
                    score: 0.05,
                },
                crate::ai::TagScore {
                    label: "mountain".to_string(),
                    score: 0.02,
                },
            ])
        }
    }

    /// A tagger double whose raw scores are not
    /// finite (issue #352): the composition must
    /// never see them.
    #[derive(Debug)]
    struct NonFiniteTagger;

    impl crate::ai::TagProvider for NonFiniteTagger {
        fn scores(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<Vec<crate::ai::TagScore>, crate::ai::AiError> {
            Ok(vec![
                crate::ai::TagScore {
                    label: "ocean".to_string(),
                    score: f32::NEG_INFINITY,
                },
                crate::ai::TagScore {
                    label: "sky".to_string(),
                    score: 0.05,
                },
            ])
        }
    }

    /// A mask fake: everything is foreground, so a removebg
    /// render keeps the source image.
    #[derive(Debug)]
    struct FakeMask;

    impl crate::ai::MaskProvider for FakeMask {
        fn mask(&self, img: &image::DynamicImage) -> Result<image::GrayImage, crate::ai::AiError> {
            Ok(image::GrayImage::from_pixel(
                img.width(),
                img.height(),
                image::Luma([255]),
            ))
        }
    }

    /// A fake inpainting provider: fills the masked pixels
    /// with one flat color, the deterministic stand-in for
    /// the model's fill.
    #[derive(Debug)]
    struct FakeInpaint;

    impl crate::ai::InpaintProvider for FakeInpaint {
        fn inpaint(
            &self,
            img: &image::DynamicImage,
            mask: &image::GrayImage,
        ) -> Result<image::DynamicImage, crate::ai::AiError> {
            let mut rgb = img.to_rgb8();
            for (x, y, m) in mask.enumerate_pixels() {
                if m[0] > 0 {
                    *rgb.get_pixel_mut(x, y) = image::Rgb([30, 30, 200]);
                }
            }
            Ok(image::DynamicImage::ImageRgb8(rgb))
        }
    }

    async fn spawn_app_with_tagger(tagger: Arc<dyn crate::ai::TagProvider>) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: Some(tagger),
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn meta_non_finite_tag_scores_omit_the_annotation() {
        // A tagger answering with a non-finite raw
        // score (issue #352) produced bad model
        // output: the annotation omits, and the body
        // still parses as JSON.
        let src = spawn_image_server().await;
        let base = spawn_app_with_tagger(Arc::new(NonFiniteTagger)).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert!(meta.get("tags").is_none());
        assert!(meta.get("alt_text").is_none());
    }

    #[tokio::test]
    async fn meta_includes_tags_only_with_a_tagger() {
        let src = spawn_image_server().await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        // With a tagger: tags and alt text are present.
        let base = spawn_app_with_tagger(Arc::new(FakeTagger)).await;
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["tags"][0]["label"], "ocean");
        assert_eq!(meta["tags"][0]["score"], 1.0);
        assert_eq!(meta["tags"].as_array().unwrap().len(), 1);
        assert_eq!(meta["alt_text"], "Ocean, sky, mountain");
        // Without one: both fields absent.
        let base = spawn_app(false).await;
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert!(meta.get("tags").is_none());
        assert!(meta.get("alt_text").is_none());
    }

    #[derive(Debug)]
    struct FakeModeration;

    impl crate::ai::ModerationProvider for FakeModeration {
        fn moderate(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<crate::ai::ModerationScores, crate::ai::AiError> {
            // Raw probabilities (issue #171): the
            // composition rounds them for the wire.
            Ok(crate::ai::ModerationScores {
                model: "fake-model".to_string(),
                scores: vec![
                    crate::ai::TagScore {
                        label: "normal".to_string(),
                        score: 0.99876,
                    },
                    crate::ai::TagScore {
                        label: "nsfw".to_string(),
                        score: 0.00124,
                    },
                ],
                signals: Vec::new(),
            })
        }
    }

    #[derive(Debug)]
    struct FailingModeration;

    impl crate::ai::ModerationProvider for FailingModeration {
        fn moderate(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<crate::ai::ModerationScores, crate::ai::AiError> {
            Err(crate::ai::AiError::Inference("model exploded".to_string()))
        }
    }

    /// A moderation double whose output is not
    /// finite (issue #352): a NaN verdict score and
    /// an infinite signal.
    #[derive(Debug)]
    struct NonFiniteModeration;

    impl crate::ai::ModerationProvider for NonFiniteModeration {
        fn moderate(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<crate::ai::ModerationScores, crate::ai::AiError> {
            Ok(crate::ai::ModerationScores {
                model: "fake-model".to_string(),
                scores: vec![crate::ai::TagScore {
                    label: "normal".to_string(),
                    score: f32::NAN,
                }],
                signals: vec![crate::ai::TagScore {
                    label: "lead".to_string(),
                    score: f32::INFINITY,
                }],
            })
        }
    }

    async fn spawn_app_with_moderation(
        provider: Option<Arc<dyn crate::ai::ModerationProvider>>,
    ) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: provider,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn meta_includes_moderation_verdicts_only_on_opt_in() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_moderation(Some(Arc::new(FakeModeration))).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json; moderation=1")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        let moderation = &meta["moderation"];
        assert_eq!(moderation["model"], "fake-model");
        let scores = moderation["scores"].as_array().unwrap();
        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0]["label"], "normal");
        // Verdicts round to 4 decimals on the wire.
        assert!((scores[0]["score"].as_f64().unwrap() - 0.9988).abs() < 1e-9);
        assert_eq!(scores[1]["label"], "nsfw");
        assert!((scores[1]["score"].as_f64().unwrap() - 0.0012).abs() < 1e-9);
        assert!(moderation.get("signals").is_none());
        // Without the opt-in the field is absent, even
        // with a provider configured.
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert!(meta.get("moderation").is_none());
    }

    #[tokio::test]
    async fn meta_moderation_signals_are_marked_review_only() {
        let src = spawn_image_server().await;
        // The CLIP fallback, wrapped around the fake
        // tagger: its raw scores become signals.
        let fallback =
            crate::ai::ClipModerationProvider::new(Arc::new(FakeTagger), "clip-fake".to_string());
        let base = spawn_app_with_moderation(Some(Arc::new(fallback))).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json; moderation=1")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        let moderation = &meta["moderation"];
        assert_eq!(moderation["model"], "clip-fake");
        // No verdicts from a signals-only provider.
        assert!(moderation.get("scores").is_none());
        let signals = &moderation["signals"];
        assert_eq!(signals["review_only"], true);
        let scores = signals["scores"].as_array().unwrap();
        assert_eq!(scores.len(), 3);
        // Best first, rounded to 3 decimals.
        assert_eq!(scores[0]["label"], "ocean");
        assert!((scores[0]["score"].as_f64().unwrap() - 0.3).abs() < 1e-9);
        assert_eq!(scores[2]["label"], "mountain");
    }

    #[tokio::test]
    async fn meta_model_fields_report_file_names_not_paths() {
        // Providers are built from the operator's
        // configured paths (that is what main passes);
        // the meta response must name the model file,
        // never the filesystem layout it lives in.
        let src = spawn_image_server().await;
        let fallback = crate::ai::ClipModerationProvider::new(
            Arc::new(FakeTagger),
            "/srv/models/clip_vision.onnx".to_string(),
        );
        let base = spawn_app_with_moderation(Some(Arc::new(fallback))).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json; moderation=1")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let text = resp.text().await.unwrap();
        assert!(
            !text.contains("/srv/models"),
            "the configured path must not appear in the response: {text}"
        );
        let meta: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(meta["moderation"]["model"], "clip_vision.onnx");
    }

    #[tokio::test]
    async fn meta_moderation_reports_unavailable_without_a_provider() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_moderation(None).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json; moderation=1")
            .send()
            .await
            .unwrap();
        // The meta request still succeeds; only the
        // field reports its state.
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["moderation"]["available"], false);
        assert_eq!(meta["moderation"]["reason"], "not_configured");
    }

    #[tokio::test]
    async fn meta_moderation_failure_reports_unavailable() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_moderation(Some(Arc::new(FailingModeration))).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json; moderation=1")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["moderation"]["available"], false);
        assert_eq!(meta["moderation"]["reason"], "inference_failed");
    }

    #[tokio::test]
    async fn meta_moderation_non_finite_scores_report_unavailable() {
        // Non-finite model output (issue #352) is
        // bad output, not a number: the field reports
        // unavailable like an inference failure, and
        // the body parses as JSON (a bare NaN in the
        // body would fail this parse).
        let src = spawn_image_server().await;
        let base = spawn_app_with_moderation(Some(Arc::new(NonFiniteModeration))).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json; moderation=1")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["moderation"]["available"], false);
        assert_eq!(meta["moderation"]["reason"], "inference_failed");
    }

    #[test]
    fn moderation_opt_in_takes_exact_tokens() {
        // The #151 rule, for the moderation parameter.
        assert!(wants_moderation(Some("application/json; moderation")));
        assert!(wants_moderation(Some("application/json; moderation=1")));
        assert!(wants_moderation(Some("application/json; moderation=TRUE")));
        assert!(!wants_moderation(Some("application/json")));
        assert!(!wants_moderation(Some("application/json; moderation=0")));
        assert!(!wants_moderation(Some("application/json; moderation=yes")));
        assert!(!wants_moderation(Some("application/json; detections=1")));
        assert!(!wants_moderation(None));
    }

    #[derive(Debug)]
    struct FakeOcr;

    impl crate::ai::OcrProvider for FakeOcr {
        fn recognize(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<crate::ai::OcrReport, crate::ai::AiError> {
            // Raw scores (issue #177): the composition
            // rounds them for the wire.
            Ok(crate::ai::OcrReport {
                model: "fake-recognizer".to_string(),
                language: "en".to_string(),
                lines: vec![
                    crate::ai::OcrLine {
                        text: "CRABCROP MARKET".to_string(),
                        score: 0.98765,
                        x: 41,
                        y: 37,
                        w: 490,
                        h: 42,
                    },
                    crate::ai::OcrLine {
                        text: "OPEN 24 HOURS".to_string(),
                        score: 0.9,
                        x: 40,
                        y: 248,
                        w: 401,
                        h: 41,
                    },
                ],
            })
        }
    }

    #[derive(Debug)]
    struct FailingOcr;

    impl crate::ai::OcrProvider for FailingOcr {
        fn recognize(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<crate::ai::OcrReport, crate::ai::AiError> {
            Err(crate::ai::AiError::Inference("model exploded".to_string()))
        }
    }

    /// An OCR double whose line score is not finite
    /// (issue #352).
    #[derive(Debug)]
    struct NonFiniteOcr;

    impl crate::ai::OcrProvider for NonFiniteOcr {
        fn recognize(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<crate::ai::OcrReport, crate::ai::AiError> {
            Ok(crate::ai::OcrReport {
                model: "fake-recognizer".to_string(),
                language: "en".to_string(),
                lines: vec![crate::ai::OcrLine {
                    text: "CRABCROP MARKET".to_string(),
                    score: f32::INFINITY,
                    x: 41,
                    y: 37,
                    w: 490,
                    h: 42,
                }],
            })
        }
    }

    /// An OCR provider that takes five seconds to
    /// answer: the annotation deadline tests run it
    /// against a far shorter budget.
    #[derive(Debug)]
    struct SlowOcr;

    impl crate::ai::OcrProvider for SlowOcr {
        fn recognize(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<crate::ai::OcrReport, crate::ai::AiError> {
            std::thread::sleep(std::time::Duration::from_secs(5));
            Ok(crate::ai::OcrReport {
                model: "slow-recognizer".to_string(),
                language: "en".to_string(),
                lines: Vec::new(),
            })
        }
    }

    async fn spawn_app_with_ocr(provider: Option<Arc<dyn crate::ai::OcrProvider>>) -> String {
        spawn_app_with_ocr_timeout(provider, std::time::Duration::from_secs(30)).await
    }

    async fn spawn_app_with_ocr_timeout(
        provider: Option<Arc<dyn crate::ai::OcrProvider>>,
        process_timeout: std::time::Duration,
    ) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout,
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: provider,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn meta_includes_ocr_text_only_on_opt_in() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_ocr(Some(Arc::new(FakeOcr))).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json; ocr=1")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        let ocr = &meta["ocr"];
        assert_eq!(ocr["model"], "fake-recognizer");
        assert_eq!(ocr["language"], "en");
        let lines = ocr["text"].as_array().unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["text"], "CRABCROP MARKET");
        // The raw 0.98765 rounds to 3 decimals on the wire.
        assert!((lines[0]["score"].as_f64().unwrap() - 0.988).abs() < 1e-9);
        assert_eq!(lines[0]["x"], 41);
        assert_eq!(lines[0]["y"], 37);
        assert_eq!(lines[0]["w"], 490);
        assert_eq!(lines[0]["h"], 42);
        assert_eq!(lines[1]["text"], "OPEN 24 HOURS");
        // The joined content is the lines in order, one
        // per line.
        assert_eq!(ocr["text_content"], "CRABCROP MARKET\nOPEN 24 HOURS");
        // Without the opt-in the field is absent, even
        // with a provider configured.
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert!(meta.get("ocr").is_none());
    }

    #[tokio::test]
    async fn meta_ocr_reports_unavailable_without_a_provider() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_ocr(None).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json; ocr=1")
            .send()
            .await
            .unwrap();
        // The meta request still succeeds; only the
        // field reports its state.
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["ocr"]["available"], false);
        assert_eq!(meta["ocr"]["reason"], "not_configured");
    }

    #[tokio::test]
    async fn meta_ocr_failure_reports_unavailable() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_ocr(Some(Arc::new(FailingOcr))).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json; ocr=1")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["ocr"]["available"], false);
        assert_eq!(meta["ocr"]["reason"], "inference_failed");
    }

    #[tokio::test]
    async fn meta_ocr_non_finite_scores_report_unavailable() {
        // A non-finite line score (issue #352) is
        // bad model output: the field reports
        // unavailable and the body parses as JSON.
        let src = spawn_image_server().await;
        let base = spawn_app_with_ocr(Some(Arc::new(NonFiniteOcr))).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json; ocr=1")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["ocr"]["available"], false);
        assert_eq!(meta["ocr"]["reason"], "inference_failed");
    }

    #[tokio::test]
    async fn meta_ocr_reports_timed_out_at_the_annotation_deadline() {
        // The annotation sections share one deadline,
        // the process budget: an OCR run that would take
        // five seconds against a 300 ms budget must not
        // hold the response. The section reports itself
        // unavailable with the timed_out reason and the
        // meta answer goes out on time.
        let src = spawn_image_server().await;
        let base = spawn_app_with_ocr_timeout(
            Some(Arc::new(SlowOcr)),
            std::time::Duration::from_millis(300),
        )
        .await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let started = Instant::now();
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json; ocr=1")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(3),
            "the meta response must not wait out the OCR run: {:?}",
            started.elapsed()
        );
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["ocr"]["available"], false);
        assert_eq!(meta["ocr"]["reason"], "timed_out");
    }

    #[tokio::test]
    async fn ocr_meta_requests_leave_the_image_bytes_identical() {
        // OCR is an annotation: the image entry is built
        // and cached exactly as if the opt-in never
        // happened (issue #177). Render the image, ask
        // for OCR meta, render again: the bytes must be
        // the same bytes, and the meta must describe
        // that same length.
        let src = spawn_image_server().await;
        let base = spawn_app_with_ocr(Some(Arc::new(FakeOcr))).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let first = client()
            .get(format!("{base}{url_path}"))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json; ocr=1")
            .send()
            .await
            .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["bytes"].as_u64().unwrap(), first.len() as u64);
        assert!(meta.get("ocr").is_some());
        let second = client()
            .get(format!("{base}{url_path}"))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn ocr_opt_in_takes_exact_tokens() {
        // The #151 rule, for the OCR parameter.
        assert!(wants_ocr(Some("application/json; ocr")));
        assert!(wants_ocr(Some("application/json; ocr=1")));
        assert!(wants_ocr(Some("application/json; ocr=TRUE")));
        assert!(!wants_ocr(Some("application/json")));
        assert!(!wants_ocr(Some("application/json; ocr=0")));
        assert!(!wants_ocr(Some("application/json; ocr=yes")));
        assert!(!wants_ocr(Some("application/json; moderation=1")));
        assert!(!wants_ocr(None));
    }

    #[test]
    fn embeddings_opt_in_takes_exact_tokens() {
        // The #151 option rule, same as placeholders and
        // moderation: a bare token or `=1` opts in,
        // `=TRUE` (any case) also counts, nothing else
        // does.
        assert!(wants_embeddings(Some("application/json; embeddings")));
        assert!(wants_embeddings(Some("application/json; embeddings=1")));
        assert!(wants_embeddings(Some("application/json; embeddings=TRUE")));
        assert!(!wants_embeddings(Some("application/json; embeddings=0")));
        assert!(!wants_embeddings(Some("application/json; embeddings=yes")));
        assert!(!wants_embeddings(Some("application/json; moderation=1")));
        assert!(!wants_embeddings(None));
    }

    #[test]
    fn f16_packing_matches_known_vectors() {
        // IEEE 754 half values, round to nearest even:
        // signs, normals, the largest finite, the
        // overflow tie, subnormals and their ties,
        // infinities, and NaN.
        let cases: &[(f32, u16)] = &[
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-1.0, 0xbc00),
            (2.0, 0x4000),
            (0.5, 0x3800),
            (0.1, 0x2e66),
            (std::f32::consts::PI, 0x4248),
            (65504.0, 0x7bff),
            (65520.0, 0x7c00),
            (f32::INFINITY, 0x7c00),
            (f32::NEG_INFINITY, 0xfc00),
            (6.103_515_6e-5, 0x0400), // 2^-14, smallest normal
            (5.960_464_5e-8, 0x0001), // 2^-24, smallest subnormal
            (2.980_232_2e-8, 0x0000), // 2^-25, ties to zero
            (8.940_697e-8, 0x0002),   // 3 * 2^-25, ties up to even
            (f32::NAN, 0x7e00),
        ];
        for (value, want) in cases {
            assert_eq!(f32_to_f16_bits(*value), *want, "f16({value})");
        }
        // The full packing: little-endian halves,
        // base64 over the bytes.
        assert_eq!(pack_embedding(&[1.0, -1.0, 0.5]), "ADwAvAA4");
        assert_eq!(pack_embedding(&[]), "");
    }

    #[test]
    fn f16_packing_round_trips_unit_vectors() {
        // A test-local decoder: widen an f16 back to
        // f32 exactly (every f16 is representable).
        fn widen(half: u16) -> f32 {
            let sign = if half & 0x8000 != 0 { -1.0 } else { 1.0 };
            let exp = (half >> 10) & 0x1f;
            let frac = (half & 0x3ff) as f32;
            let magnitude = match exp {
                0 => frac * 2f32.powi(-24),
                31 => {
                    if frac == 0.0 {
                        f32::INFINITY
                    } else {
                        f32::NAN
                    }
                }
                e => (1.0 + frac / 1024.0) * 2f32.powi(e as i32 - 15),
            };
            sign * magnitude
        }
        let vector = [
            0.0,
            1.0,
            -1.0,
            0.5,
            -0.25,
            0.1,
            0.9995,
            -std::f32::consts::FRAC_1_SQRT_2,
        ];
        let packed = pack_embedding(&vector);
        // Decode the base64 by hand: the packing test
        // above pins the encoder against known vectors,
        // so here a small decoder keeps the round trip
        // independent of it.
        let bytes = {
            let mut out = Vec::new();
            let mut acc: u32 = 0;
            let mut nbits = 0;
            for c in packed.bytes() {
                let v = match c {
                    b'A'..=b'Z' => (c - b'A') as u32,
                    b'a'..=b'z' => (c - b'a' + 26) as u32,
                    b'0'..=b'9' => (c - b'0' + 52) as u32,
                    b'+' => 62,
                    b'/' => 63,
                    b'=' => break,
                    other => panic!("unexpected base64 byte {other}"),
                };
                acc = (acc << 6) | v;
                nbits += 6;
                if nbits >= 8 {
                    nbits -= 8;
                    out.push((acc >> nbits) as u8);
                }
            }
            out
        };
        assert_eq!(bytes.len(), vector.len() * 2);
        for (chunk, want) in bytes.as_chunks::<2>().0.iter().zip(vector.iter()) {
            let back = widen(u16::from_le_bytes(*chunk));
            // f16 keeps about 3 decimal digits: the
            // round trip must land within half an ulp
            // of the original's magnitude class.
            assert!((back - want).abs() <= 0.001, "{want} -> {back}");
        }
    }

    /// An embedding provider double (issue #172): a
    /// fixed unit vector under a fixed model identity.
    #[derive(Debug)]
    struct FakeEmbedding;

    impl crate::ai::EmbeddingProvider for FakeEmbedding {
        fn embed(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<crate::ai::ImageEmbedding, crate::ai::AiError> {
            Ok(crate::ai::ImageEmbedding {
                model: "fake-encoder".to_string(),
                vector: vec![0.5; 4],
            })
        }
    }

    /// An embedding provider double that always fails.
    #[derive(Debug)]
    struct FailingEmbedding;

    impl crate::ai::EmbeddingProvider for FailingEmbedding {
        fn embed(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<crate::ai::ImageEmbedding, crate::ai::AiError> {
            Err(crate::ai::AiError::Inference("no vector today".to_string()))
        }
    }

    /// An embedding double whose vector is not
    /// finite (issue #352).
    #[derive(Debug)]
    struct NonFiniteEmbedding;

    impl crate::ai::EmbeddingProvider for NonFiniteEmbedding {
        fn embed(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<crate::ai::ImageEmbedding, crate::ai::AiError> {
            Ok(crate::ai::ImageEmbedding {
                model: "fake-encoder".to_string(),
                vector: vec![0.5, f32::NAN, 0.25, 0.5],
            })
        }
    }

    /// Spawn the app with an embedding provider and,
    /// optionally, an aesthetic head, for the meta
    /// tests (issue #172). Mirrors
    /// `spawn_app_with_moderation`.
    async fn spawn_app_with_embeddings(
        provider: Option<Arc<dyn crate::ai::EmbeddingProvider>>,
        aesthetic: Option<Arc<crate::ai::AestheticHead>>,
    ) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: provider,
            ocr: None,
            aesthetic,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// Fetch a signed meta response as JSON with the
    /// given Accept value.
    async fn meta_with(base: &str, src: &str, accept: &str) -> serde_json::Value {
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", accept)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        serde_json::from_str(&resp.text().await.unwrap()).unwrap()
    }

    #[tokio::test]
    async fn meta_non_finite_embedding_reports_unavailable() {
        // A non-finite vector (issue #352) is bad
        // model output: the embedding reports
        // unavailable, and meta_with's parse proves
        // the body is valid JSON.
        let src = spawn_image_server().await;
        let base = spawn_app_with_embeddings(Some(Arc::new(NonFiniteEmbedding)), None).await;
        let meta = meta_with(&base, &src, "application/json; embeddings=1").await;
        assert!(meta["embedding"].is_null(), "{meta}");
        assert_eq!(meta["embedding_reason"], "inference_failed");
    }

    #[tokio::test]
    async fn meta_embedding_only_on_opt_in() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_embeddings(Some(Arc::new(FakeEmbedding)), None).await;
        // Without the opt-in, no embedding fields at
        // all: the provider is not even consulted.
        let plain = meta_with(&base, &src, "application/json").await;
        assert!(plain.get("embedding").is_none(), "{plain}");
        assert!(plain.get("aesthetic").is_none(), "{plain}");
        // With it, the packed vector and every
        // mandatory companion.
        let meta = meta_with(&base, &src, "application/json; embeddings=1").await;
        assert_eq!(meta["embedding"], pack_embedding(&[0.5; 4]), "{meta}");
        assert_eq!(meta["embedding_model"], "fake-encoder", "{meta}");
        assert_eq!(meta["embedding_dim"], 4, "{meta}");
        assert_eq!(meta["embedding_encoding"], "f16le-base64", "{meta}");
        assert_eq!(meta["embedding_normalized"], true, "{meta}");
    }

    #[tokio::test]
    async fn meta_embedding_without_a_provider_reports_unavailable() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_embeddings(None, None).await;
        let meta = meta_with(&base, &src, "application/json; embeddings=1").await;
        assert!(meta["embedding"].is_null(), "{meta}");
        assert_eq!(meta["embedding_reason"], "not_configured", "{meta}");
    }

    #[tokio::test]
    async fn meta_embedding_failure_reports_unavailable() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_embeddings(Some(Arc::new(FailingEmbedding)), None).await;
        let meta = meta_with(&base, &src, "application/json; embeddings=1").await;
        assert!(meta["embedding"].is_null(), "{meta}");
        assert_eq!(meta["embedding_reason"], "inference_failed", "{meta}");
    }

    #[tokio::test]
    async fn meta_aesthetic_scores_when_a_head_is_configured() {
        // A head reading only the first component,
        // plus a 0.5 bias: the fake vector is all
        // 0.5, so the score is exactly 1.0. The field
        // rides the plain meta request: no embeddings
        // opt-in needed, and no embedding fields
        // appear without one.
        let head = crate::ai::AestheticHead::from_weights(
            vec![1.0, 0.0, 0.0, 0.0],
            0.5,
            "fake-head".to_string(),
        )
        .unwrap();
        let src = spawn_image_server().await;
        let base =
            spawn_app_with_embeddings(Some(Arc::new(FakeEmbedding)), Some(Arc::new(head))).await;
        let meta = meta_with(&base, &src, "application/json").await;
        assert_eq!(meta["aesthetic"]["head"], "fake-head", "{meta}");
        assert_eq!(meta["aesthetic"]["model"], "fake-encoder", "{meta}");
        assert_eq!(meta["aesthetic"]["score"], 1.0, "{meta}");
        assert!(meta.get("embedding").is_none(), "{meta}");
    }

    #[tokio::test]
    async fn meta_aesthetic_dimension_mismatch_reports_expected_and_got() {
        // A 3-wide head against the 4-wide fake
        // encoder: the fail-soft case. Null score, the
        // reason, and both dimensions; the request
        // still succeeds.
        let head = crate::ai::AestheticHead::from_weights(
            vec![1.0, 0.0, 0.0],
            0.5,
            "l14-head".to_string(),
        )
        .unwrap();
        let src = spawn_image_server().await;
        let base =
            spawn_app_with_embeddings(Some(Arc::new(FakeEmbedding)), Some(Arc::new(head))).await;
        let meta = meta_with(&base, &src, "application/json").await;
        let aesthetic = &meta["aesthetic"];
        assert!(aesthetic["score"].is_null(), "{meta}");
        assert_eq!(aesthetic["reason"], "dimension_mismatch", "{meta}");
        assert_eq!(aesthetic["expected"], 3, "{meta}");
        assert_eq!(aesthetic["got"], 4, "{meta}");
        assert_eq!(aesthetic["model"], "fake-encoder", "{meta}");
    }

    #[tokio::test]
    async fn meta_aesthetic_without_a_vector_reports_the_reason() {
        let head =
            crate::ai::AestheticHead::from_weights(vec![1.0, 0.0, 0.0, 0.0], 0.5, "h".to_string())
                .unwrap();
        let src = spawn_image_server().await;
        let base =
            spawn_app_with_embeddings(Some(Arc::new(FailingEmbedding)), Some(Arc::new(head))).await;
        let meta = meta_with(&base, &src, "application/json").await;
        let aesthetic = &meta["aesthetic"];
        assert_eq!(aesthetic["head"], "h", "{meta}");
        assert!(aesthetic["score"].is_null(), "{meta}");
        assert_eq!(aesthetic["reason"], "inference_failed", "{meta}");
        // No vector, so no encoder identity either.
        assert!(aesthetic.get("model").is_none(), "{meta}");
    }

    #[test]
    fn score_list_refuses_non_finite_scores() {
        // The serializer's own guard (issue #352):
        // finite lists render as before, and any
        // non-finite score refuses the whole list.
        let finite = vec![
            crate::ai::TagScore {
                label: "a".to_string(),
                score: 0.5,
            },
            crate::ai::TagScore {
                label: "b".to_string(),
                score: 0.25,
            },
        ];
        let rendered = score_list(&finite).expect("finite scores render");
        assert_eq!(
            serde_json::to_string(&rendered).unwrap(),
            "[{\"label\":\"a\",\"score\":0.5},{\"label\":\"b\",\"score\":0.25}]"
        );
        assert_eq!(
            serde_json::to_string(&score_list(&[]).unwrap()).unwrap(),
            "[]"
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let scores = vec![crate::ai::TagScore {
                label: "a".to_string(),
                score: bad,
            }];
            assert!(score_list(&scores).is_none(), "{bad}");
        }
    }

    fn segment_record(score: f32, coverage: f32) -> crate::engine::SegmentRecord {
        crate::engine::SegmentRecord {
            objects: vec![crate::engine::SegmentObject {
                label: "cat".to_string(),
                score,
                detection: (0, 0, 10, 10),
                tight_box: (1, 1, 8, 8),
                coverage,
                centroid: (5, 5),
            }],
        }
    }

    #[test]
    fn segments_json_refuses_non_finite_values() {
        // The record-level guarantee (issue #352): any
        // non-finite score or coverage refuses the whole
        // record, and finite values render at three
        // decimals.
        let rendered = segments_json(&segment_record(0.5, 0.25)).expect("finite renders");
        assert_eq!(
            serde_json::to_string(&rendered).unwrap(),
            "{\"objects\":[{\"label\":\"cat\",\"score\":0.500,\"box\":[0,0,10,10],\
             \"tight_box\":[1,1,8,8],\"coverage\":0.250,\"centroid\":[5,5]}]}"
        );
        assert!(segments_json(&segment_record(f32::NAN, 0.25)).is_none());
        assert!(segments_json(&segment_record(0.5, f32::INFINITY)).is_none());
        assert!(segments_json(&segment_record(f32::NEG_INFINITY, 0.25)).is_none());
    }

    fn objects_crop_record(scale: f32, score: f32) -> crate::engine::ObjectsCropRecord {
        crate::engine::ObjectsCropRecord {
            classes: vec!["cat".to_string()],
            objects: vec![crate::ai::Detection {
                label: "cat".to_string(),
                score,
                x: 0,
                y: 0,
                w: 10,
                h: 10,
            }],
            union: Some((0, 0, 10, 10)),
            scale,
            fallback: false,
        }
    }

    #[test]
    fn objects_crop_json_refuses_non_finite_values() {
        let rendered = objects_crop_json(&objects_crop_record(1.5, 0.5)).expect("finite renders");
        assert_eq!(
            serde_json::to_string(&rendered).unwrap(),
            "{\"classes\":[\"cat\"],\"scale\":1.5,\"path\":\"objects\",\
             \"objects\":[{\"label\":\"cat\",\"score\":0.5,\"x\":0,\"y\":0,\"w\":10,\"h\":10}],\
             \"union\":[0,0,10,10]}"
        );
        assert!(objects_crop_json(&objects_crop_record(f32::NAN, 0.5)).is_none());
        assert!(objects_crop_json(&objects_crop_record(1.5, f32::INFINITY)).is_none());
    }

    fn studio_product_record(confidence: Option<f32>) -> crate::studio::StudioRecord {
        crate::studio::StudioRecord {
            mode: crate::url::StudioMode::Product,
            bg: Some([255, 0, 0]),
            bg_from: "argument",
            shadow: crate::url::StudioShadow::Soft,
            pad: 10,
            steps: vec!["mask", "backdrop"],
            fallbacks: Vec::new(),
            confidence,
            scene: None,
        }
    }

    fn studio_scene_record(saturation: f32) -> crate::studio::StudioRecord {
        crate::studio::StudioRecord {
            mode: crate::url::StudioMode::Scene,
            bg: None,
            bg_from: "argument",
            shadow: crate::url::StudioShadow::Soft,
            pad: 5,
            steps: vec!["mask", "crop"],
            fallbacks: Vec::new(),
            confidence: Some(0.5),
            scene: Some(crate::studio::SceneRecord {
                subject: Some((1, 2, 3, 4)),
                subjects: 2,
                crop: (0, 0, 50, 50),
                levels: [[0, 255], [10, 200], [0, 255]],
                saturation,
            }),
        }
    }

    #[test]
    fn studio_json_omits_non_finite_fields() {
        // A non-finite confidence omits mask_confidence
        // (issue #352); the rest of the product record
        // renders as before.
        let rendered = studio_json(&studio_product_record(Some(f32::NAN)));
        assert_eq!(
            serde_json::to_string(&rendered).unwrap(),
            "{\"mode\":\"product\",\"bg\":\"#ff0000\",\"bg_from\":\"argument\",\
             \"shadow\":\"soft\",\"pad\":10,\"steps\":[\"mask\",\"backdrop\"],\"fallbacks\":[]}"
        );
        // A finite confidence rides, and a scene record
        // carries its facts with no background fields.
        let rendered = studio_json(&studio_product_record(Some(0.25)));
        assert_eq!(
            serde_json::to_string(&rendered).unwrap(),
            "{\"mode\":\"product\",\"bg\":\"#ff0000\",\"bg_from\":\"argument\",\
             \"shadow\":\"soft\",\"pad\":10,\"steps\":[\"mask\",\"backdrop\"],\"fallbacks\":[],\
             \"mask_confidence\":0.25}"
        );
        let rendered = studio_json(&studio_scene_record(1.25));
        assert_eq!(
            serde_json::to_string(&rendered).unwrap(),
            "{\"mode\":\"scene\",\"pad\":5,\"steps\":[\"mask\",\"crop\"],\"fallbacks\":[],\
             \"scene\":{\"subject\":[1,2,3,4],\"subjects\":2,\"crop\":[0,0,50,50],\
             \"levels\":[[0,255],[10,200],[0,255]],\"saturation\":1.25},\
             \"mask_confidence\":0.5}"
        );
        // A non-finite saturation omits just the field.
        let rendered = studio_json(&studio_scene_record(f32::NAN));
        assert_eq!(
            serde_json::to_string(&rendered).unwrap(),
            "{\"mode\":\"scene\",\"pad\":5,\"steps\":[\"mask\",\"crop\"],\"fallbacks\":[],\
             \"scene\":{\"subject\":[1,2,3,4],\"subjects\":2,\"crop\":[0,0,50,50],\
             \"levels\":[[0,255],[10,200],[0,255]]},\"mask_confidence\":0.5}"
        );
    }

    #[test]
    fn unavailable_sections_render_the_uniform_form() {
        // The unavailable object is a wire constant:
        // available is always false, in both sections
        // that use the form.
        let moderation = moderation_json(&ModerationMeta::Unavailable("off"));
        assert_eq!(
            serde_json::to_string(&moderation).unwrap(),
            "{\"available\":false,\"reason\":\"off\"}"
        );
        let ocr = ocr_json(&OcrMeta::Unavailable("off"));
        assert_eq!(
            serde_json::to_string(&ocr).unwrap(),
            "{\"available\":false,\"reason\":\"off\"}"
        );
        // A report whose lines carry a non-finite score
        // degrades to the same form (issue #352).
        let report = crate::ai::OcrReport {
            model: "m".to_string(),
            language: "en".to_string(),
            lines: vec![crate::ai::OcrLine {
                text: "x".to_string(),
                score: f32::NAN,
                x: 0,
                y: 0,
                w: 1,
                h: 1,
            }],
        };
        let ocr = ocr_json(&OcrMeta::Report(report));
        assert_eq!(
            serde_json::to_string(&ocr).unwrap(),
            "{\"available\":false,\"reason\":\"inference_failed\"}"
        );
    }

    /// Read a meta response's body as text.
    async fn meta_body(response: Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    /// A render carrying every operation record, for
    /// the byte-exact body tests.
    fn full_meta_processed() -> crate::engine::ProcessedImage {
        let mut processed = still_processed(test_png(), ImageFormat::Png);
        processed.ssimulacra2 = Some(87.26);
        processed.video = Some(crate::engine::VideoSourceMeta {
            duration_seconds: Some(3.0),
            width: 100,
            height: 60,
            frame_seconds: 1.5,
        });
        processed.pdf = Some(crate::engine::PdfSourceMeta {
            pages: 2,
            page: 1,
            page_width: 612.0,
            page_height: 792.5,
        });
        processed.studio = Some(studio_product_record(Some(0.25)));
        processed.inpaint = Some(crate::engine::InpaintRecord {
            regions: vec![(1, 2, 3, 4)],
        });
        processed.segments = Some(segment_record(0.5, 0.25));
        processed.objects_crop = Some(objects_crop_record(1.5, 0.5));
        processed.placeholder = Some(crate::placeholder::Placeholder {
            dominant_color: "#112233".to_string(),
            data_uri: "data:image/jpeg;base64,AAAA".to_string(),
            palette: vec!["#112233".to_string()],
            thumbhash: "abc".to_string(),
        });
        processed
    }

    fn full_meta_annotation() -> Annotation {
        Annotation {
            tags: vec![crate::ai::TagScore {
                label: "cat".to_string(),
                score: 0.5,
            }],
            alt_text: "a cat".to_string(),
        }
    }

    fn full_meta_detections() -> Vec<(String, crate::ai::Detection)> {
        vec![(
            "yolo".to_string(),
            crate::ai::Detection {
                label: "dog".to_string(),
                score: 0.25,
                x: 1,
                y: 2,
                w: 3,
                h: 4,
            },
        )]
    }

    #[tokio::test]
    async fn meta_response_body_is_byte_exact_unsealed() {
        // The typed body must spell what the hand-built
        // body spelled (issue #346): field order, number
        // forms, and omissions are all pinned here, so a
        // struct edit that moves a field or a float
        // spelling that drifts fails this test, not a
        // client.
        let processed = full_meta_processed();
        let annotation = full_meta_annotation();
        let detections = full_meta_detections();
        let body = meta_body(meta_response(
            &processed,
            false,
            Some(&annotation),
            Some(&detections),
            None,
            None,
            None,
            None,
        ))
        .await;
        let expected = format!(
            "{{\"width\":100,\"height\":60,\"format\":\"png\",\"bytes\":{},\
             \"source_width\":100,\"source_height\":60,\
             \"tags\":[{{\"label\":\"cat\",\"score\":0.5}}],\"alt_text\":\"a cat\",\
             \"detections\":[{{\"label\":\"dog\",\"score\":0.25,\"x\":1,\"y\":2,\"w\":3,\"h\":4,\
             \"detector\":\"yolo\"}}],\
             \"ssimulacra2\":87.3,\
             \"video\":{{\"width\":100,\"height\":60,\"frame_timestamp\":1.5,\"duration\":3}},\
             \"pdf\":{{\"pages\":2,\"page\":1,\"page_width\":612,\"page_height\":792.5}},\
             \"studio\":{{\"mode\":\"product\",\"bg\":\"#ff0000\",\"bg_from\":\"argument\",\
             \"shadow\":\"soft\",\"pad\":10,\"steps\":[\"mask\",\"backdrop\"],\"fallbacks\":[],\
             \"mask_confidence\":0.25}},\
             \"inpaint\":{{\"regions\":[[1,2,3,4]]}},\
             \"segments\":{{\"objects\":[{{\"label\":\"cat\",\"score\":0.500,\
             \"box\":[0,0,10,10],\"tight_box\":[1,1,8,8],\"coverage\":0.250,\
             \"centroid\":[5,5]}}]}},\
             \"objects_crop\":{{\"classes\":[\"cat\"],\"scale\":1.5,\"path\":\"objects\",\
             \"objects\":[{{\"label\":\"cat\",\"score\":0.5,\"x\":0,\"y\":0,\"w\":10,\"h\":10}}],\
             \"union\":[0,0,10,10]}},\
             \"dominant_color\":\"#112233\",\"placeholder\":\"data:image/jpeg;base64,AAAA\",\
             \"palette\":[\"#112233\"],\"thumbhash\":\"abc\"}}",
            test_png().len()
        );
        assert_eq!(body, expected);
    }

    #[tokio::test]
    async fn meta_response_body_is_byte_exact_sealed() {
        // The sealed form (issue #255): the render facts
        // and the image-derived sections ride; every
        // operation record is withheld.
        let processed = full_meta_processed();
        let annotation = full_meta_annotation();
        let detections = full_meta_detections();
        let body = meta_body(meta_response(
            &processed,
            true,
            Some(&annotation),
            Some(&detections),
            None,
            None,
            None,
            None,
        ))
        .await;
        let expected = format!(
            "{{\"width\":100,\"height\":60,\"format\":\"png\",\"bytes\":{},\
             \"source_width\":100,\"source_height\":60,\"sealed\":true,\
             \"tags\":[{{\"label\":\"cat\",\"score\":0.5}}],\"alt_text\":\"a cat\",\
             \"detections\":[{{\"label\":\"dog\",\"score\":0.25,\"x\":1,\"y\":2,\"w\":3,\"h\":4,\
             \"detector\":\"yolo\"}}],\
             \"ssimulacra2\":87.3,\
             \"dominant_color\":\"#112233\",\"placeholder\":\"data:image/jpeg;base64,AAAA\",\
             \"palette\":[\"#112233\"],\"thumbhash\":\"abc\"}}",
            test_png().len()
        );
        assert_eq!(body, expected);
    }

    #[tokio::test]
    async fn meta_response_body_is_byte_exact_for_unavailable_sections() {
        let processed = still_processed(test_png(), ImageFormat::Png);
        let moderation = ModerationMeta::Unavailable("not_configured");
        let embedding = EmbeddingMeta::Unavailable("busy");
        let aesthetic = AestheticMeta {
            head: "h".to_string(),
            model: None,
            outcome: AestheticOutcome::Score(f32::NAN),
        };
        let ocr = OcrMeta::Unavailable("not_configured");
        let body = meta_body(meta_response(
            &processed,
            false,
            None,
            None,
            Some(&moderation),
            Some(&embedding),
            Some(&aesthetic),
            Some(&ocr),
        ))
        .await;
        let expected = format!(
            "{{\"width\":100,\"height\":60,\"format\":\"png\",\"bytes\":{},\
             \"source_width\":100,\"source_height\":60,\
             \"moderation\":{{\"available\":false,\"reason\":\"not_configured\"}},\
             \"embedding\":null,\"embedding_reason\":\"busy\",\
             \"aesthetic\":{{\"head\":\"h\",\"score\":null,\"reason\":\"inference_failed\"}},\
             \"ocr\":{{\"available\":false,\"reason\":\"not_configured\"}}}}",
            test_png().len()
        );
        assert_eq!(body, expected);
    }

    #[tokio::test]
    async fn meta_response_omits_non_finite_probe_readings() {
        // A container can report a duration that parses
        // as NaN (issue #352): the field omits, the body
        // stays valid JSON, and the request succeeds.
        // Before the number constructors refused
        // non-finite values this panicked mid-response.
        let mut processed = still_processed(test_png(), ImageFormat::Png);
        processed.video = Some(crate::engine::VideoSourceMeta {
            duration_seconds: Some(f64::NAN),
            width: 100,
            height: 60,
            frame_seconds: 1.5,
        });
        processed.pdf = Some(crate::engine::PdfSourceMeta {
            pages: 2,
            page: 1,
            page_width: f64::NAN,
            page_height: 792.0,
        });
        let body = meta_body(meta_response(
            &processed, false, None, None, None, None, None, None,
        ))
        .await;
        assert!(
            serde_json::from_str::<serde_json::Value>(&body).is_ok(),
            "{body}"
        );
        assert!(
            body.contains("\"video\":{\"width\":100,\"height\":60,\"frame_timestamp\":1.5}"),
            "{body}"
        );
        assert!(
            body.contains("\"pdf\":{\"pages\":2,\"page\":1,\"page_height\":792}"),
            "{body}"
        );
        // A non-finite frame timestamp omits as well.
        processed.video = Some(crate::engine::VideoSourceMeta {
            duration_seconds: None,
            width: 100,
            height: 60,
            frame_seconds: f64::INFINITY,
        });
        processed.pdf = None;
        let body = meta_body(meta_response(
            &processed, false, None, None, None, None, None, None,
        ))
        .await;
        assert!(
            body.contains("\"video\":{\"width\":100,\"height\":60}"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn meta_response_pins_float_spelling_and_escaping() {
        // The number forms are Rust's Display spellings,
        // not serde's float rendering (issue #346): a
        // whole float prints bare, a small one in full
        // decimal, and the largest f32 in its full
        // expansion. Labels escape as JSON strings do.
        let annotation = Annotation {
            tags: vec![
                crate::ai::TagScore {
                    label: "a\"b".to_string(),
                    score: 1.0,
                },
                crate::ai::TagScore {
                    label: "c".to_string(),
                    score: 1e-7,
                },
                crate::ai::TagScore {
                    label: "d".to_string(),
                    score: -0.0,
                },
                crate::ai::TagScore {
                    label: "e".to_string(),
                    score: f32::MAX,
                },
            ],
            alt_text: String::new(),
        };
        let processed = still_processed(test_png(), ImageFormat::Png);
        let body = meta_body(meta_response(
            &processed,
            false,
            Some(&annotation),
            None,
            None,
            None,
            None,
            None,
        ))
        .await;
        let expected = format!(
            "{{\"width\":100,\"height\":60,\"format\":\"png\",\"bytes\":{},\
             \"source_width\":100,\"source_height\":60,\
             \"tags\":[{{\"label\":\"a\\\"b\",\"score\":1}},\
             {{\"label\":\"c\",\"score\":0.0000001}},\
             {{\"label\":\"d\",\"score\":-0}},\
             {{\"label\":\"e\",\"score\":340282350000000000000000000000000000000}}],\
             \"alt_text\":\"\"}}",
            test_png().len()
        );
        assert_eq!(body, expected);
    }

    #[tokio::test]
    async fn meta_response_empty_annotation_still_rides_alt_text() {
        // The alt text rides whenever the tags render,
        // even when both are empty: the pair is one
        // section.
        let annotation = Annotation {
            tags: Vec::new(),
            alt_text: String::new(),
        };
        let processed = still_processed(test_png(), ImageFormat::Png);
        let body = meta_body(meta_response(
            &processed,
            false,
            Some(&annotation),
            None,
            None,
            None,
            None,
            None,
        ))
        .await;
        assert!(body.contains("\"tags\":[],\"alt_text\":\"\""), "{body}");
    }

    #[test]
    fn compose_moderation_rounds_and_orders() {
        let report = crate::ai::ModerationScores {
            model: "m".to_string(),
            scores: vec![
                crate::ai::TagScore {
                    label: "second".to_string(),
                    score: 0.123456,
                },
                crate::ai::TagScore {
                    label: "first".to_string(),
                    score: 0.999949,
                },
            ],
            signals: vec![
                crate::ai::TagScore {
                    label: "low".to_string(),
                    score: 0.11111,
                },
                crate::ai::TagScore {
                    label: "high".to_string(),
                    score: 0.98765,
                },
                crate::ai::TagScore {
                    label: "mid".to_string(),
                    score: 0.5,
                },
            ],
        };
        let composed = compose_moderation(&report);
        assert_eq!(composed.model, "m");
        // Verdicts keep the model's output order and
        // round to 4 decimals.
        assert_eq!(composed.scores[0].label, "second");
        assert!((composed.scores[0].score - 0.1235).abs() < 1e-6);
        assert_eq!(composed.scores[1].label, "first");
        assert!((composed.scores[1].score - 0.9999).abs() < 1e-6);
        // Signals sort best first and round to 3.
        let labels: Vec<&str> = composed.signals.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, ["high", "mid", "low"]);
        assert!((composed.signals[0].score - 0.988).abs() < 1e-6);
        assert!((composed.signals[2].score - 0.111).abs() < 1e-6);
    }

    /// An app whose AI surface is fully faked (mask,
    /// tagger, detector) behind a one-permit AI gate, for
    /// the permit-reuse test below.
    async fn spawn_app_with_ai_and_meta() -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: Some(Arc::new(FakeMask)),
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: Some(Arc::new(FakeTagger)),
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: Some(Arc::new(FakeDetector)),
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// An app with the mask fake and a brand color
    /// configured, for the studio meta test.
    async fn spawn_app_with_studio() -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: Some(Arc::new(FakeMask)),
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: Some([0xf5, 0xf0, 0xe8]),
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// An app with the inpainting fake configured, for
    /// the remove_region tests.
    async fn spawn_app_with_inpaint() -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: Some(Arc::new(FakeInpaint)),
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// The segmentation fake for the server tests
    /// (issue #175): the silhouette of any box is its
    /// left half at full strength.
    #[derive(Debug)]
    struct FakeSegment;

    impl crate::ai::SegmentProvider for FakeSegment {
        fn encode(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<crate::ai::SegmentEmbedding, crate::ai::AiError> {
            Ok(crate::ai::SegmentEmbedding { data: vec![0.0] })
        }

        fn decode(
            &self,
            _embedding: &crate::ai::SegmentEmbedding,
            _img_w: u32,
            _img_h: u32,
            bbox: (u32, u32, u32, u32),
        ) -> Result<image::GrayImage, crate::ai::AiError> {
            let (_, _, w, h) = bbox;
            Ok(image::GrayImage::from_fn(w, h, |lx, _| {
                if lx < w / 2 {
                    image::Luma([255])
                } else {
                    image::Luma([0])
                }
            }))
        }
    }

    /// An app with the detection and segmentation fakes
    /// configured, for the segment tests.
    async fn spawn_app_with_segment() -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: Some(Arc::new(FakeSegment)),
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: Some(Arc::new(FakeDetector)),
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn meta_includes_the_segments_record() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_segment().await;
        // A mask-tight blurclass: the record names the
        // detection, its silhouette's tight box (the fake
        // silhouette is the box's left half), its coverage
        // and its centroid.
        let remainder = format!("filters:blurclass(person):format(png)/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        let objects = meta["segments"]["objects"].as_array().unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0]["label"], "person");
        assert_eq!(objects[0]["box"], serde_json::json!([1, 2, 30, 20]));
        assert_eq!(objects[0]["tight_box"], serde_json::json!([1, 2, 15, 20]));
        assert_eq!(objects[0]["coverage"], 0.5);
        assert_eq!(objects[0]["centroid"], serde_json::json!([8, 12]));
        // No class operation, no segments record.
        let remainder = format!("40x24/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert!(meta.get("segments").is_none());
    }

    #[tokio::test]
    async fn meta_includes_the_objects_crop_record() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_segment().await;
        // The fake detector finds one person at (1, 2)
        // sized 30x20: the record names the classes, the
        // scale, the path taken, the detection used, and
        // the union box framed (issue #318).
        let remainder = format!("crop:100x100:objects(person):1.5/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        let record = &meta["objects_crop"];
        assert_eq!(record["classes"], serde_json::json!(["person"]));
        assert_eq!(record["scale"], 1.5);
        assert_eq!(record["path"], "objects");
        assert_eq!(record["union"], serde_json::json!([1, 2, 30, 20]));
        let objects = record["objects"].as_array().unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0]["label"], "person");
        assert_eq!(objects[0]["x"], 1);
        assert_eq!(objects[0]["y"], 2);
        assert_eq!(objects[0]["w"], 30);
        assert_eq!(objects[0]["h"], 20);
        // No objects crop, no record.
        let remainder = format!("40x24/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert!(meta.get("objects_crop").is_none());
    }

    #[tokio::test]
    async fn objects_crop_without_a_detector_is_501() {
        let src = spawn_image_server().await;
        // No detector anywhere: the mode is unavailable
        // with the AI features' shared clean 501, never a
        // silent plain crop (issue #318).
        let base = spawn_app(false).await;
        let remainder = format!("crop:100x100:objects(person)/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = resp.text().await.unwrap();
        assert!(
            body.contains("object crop is not configured"),
            "body: {body}"
        );
    }

    #[tokio::test]
    async fn recolorclass_renders_and_needs_a_detector() {
        let src = spawn_image_server().await;
        // Configured: the recolor renders.
        let base = spawn_app_with_segment().await;
        let remainder = format!("filters:recolorclass(person,ff0000):format(png)/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // No detector anywhere: the class operations'
        // shared clean 501.
        let base = spawn_app(false).await;
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = resp.text().await.unwrap();
        assert!(body.contains("not configured"), "body: {body}");
    }

    #[tokio::test]
    async fn meta_includes_the_studio_record() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_studio().await;
        let meta_for = |remainder: String| {
            let base = base.clone();
            async move {
                let resp = client()
                    .get(format!("{base}/{}/{remainder}", sign(&remainder)))
                    .header("Accept", "application/json")
                    .send()
                    .await
                    .unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                serde_json::from_str::<serde_json::Value>(&resp.text().await.unwrap()).unwrap()
            }
        };
        // Defaults: the unnamed backdrop resolves to the
        // configured brand color, recorded as such.
        let meta = meta_for(format!(
            "filters:studio(product,shadow:none):format(png)/{src}"
        ))
        .await;
        let studio = &meta["studio"];
        assert_eq!(studio["mode"], "product");
        assert_eq!(studio["bg"], "#f5f0e8");
        assert_eq!(studio["bg_from"], "brand");
        assert_eq!(studio["shadow"], "none");
        assert_eq!(studio["pad"], 10);
        assert_eq!(
            studio["steps"],
            serde_json::json!(["mask", "defringe", "backdrop", "pad"])
        );
        assert_eq!(studio["fallbacks"], serde_json::json!([]));
        // The mask fake exposes no confidence, so the field
        // is absent rather than invented.
        assert!(studio.get("mask_confidence").is_none());
        // Named arguments resolve on their own terms.
        let meta = meta_for(format!(
            "filters:studio(product,bg:white,shadow:soft,pad:25):format(png)/{src}"
        ))
        .await;
        let studio = &meta["studio"];
        assert_eq!(studio["bg"], "#ffffff");
        assert_eq!(studio["bg_from"], "argument");
        assert_eq!(studio["shadow"], "soft");
        assert_eq!(studio["pad"], 25);
        assert_eq!(
            studio["steps"],
            serde_json::json!(["mask", "defringe", "backdrop", "shadow", "pad"])
        );
        // Scene mode (issue #198): the mask fake covers
        // the whole 100x60 frame, so the subject is the
        // frame itself and the reframe is the identity;
        // the record carries the scene object and no
        // backdrop fields.
        let meta = meta_for(format!("filters:studio(scene):format(png)/{src}")).await;
        let studio = &meta["studio"];
        assert_eq!(studio["mode"], "scene");
        assert_eq!(studio["pad"], 10);
        assert_eq!(
            studio["steps"],
            serde_json::json!(["mask", "crop", "correct"])
        );
        assert_eq!(studio["fallbacks"], serde_json::json!([]));
        assert!(studio.get("bg").is_none());
        let scene = &studio["scene"];
        assert_eq!(scene["subject"], serde_json::json!([0, 0, 100, 60]));
        assert_eq!(scene["subjects"], 1);
        assert_eq!(scene["crop"], serde_json::json!([0, 0, 100, 60]));
        assert_eq!(
            scene["levels"],
            serde_json::json!([[0, 255], [0, 255], [0, 255]]),
            "a black frame's channels are too flat to stretch"
        );
        assert_eq!(scene["saturation"].as_f64(), Some(1.0));
        // No studio filter, no studio record.
        let meta = meta_for(format!("40x24/{src}")).await;
        assert!(meta.get("studio").is_none());
    }

    #[tokio::test]
    async fn meta_includes_the_inpaint_record() {
        let src = spawn_image_server().await;
        let base = spawn_app_with_inpaint().await;
        let meta_for = |remainder: String| {
            let base = base.clone();
            async move {
                let resp = client()
                    .get(format!("{base}/{}/{remainder}", sign(&remainder)))
                    .header("Accept", "application/json")
                    .send()
                    .await
                    .unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                serde_json::from_str::<serde_json::Value>(&resp.text().await.unwrap()).unwrap()
            }
        };
        // One removal: the record names its rectangle.
        let meta = meta_for(format!("filters:remove_region(4,4,10,8):format(png)/{src}")).await;
        assert_eq!(
            meta["inpaint"]["regions"],
            serde_json::json!([[4, 4, 10, 8]])
        );
        // Two removals: both, in chain order.
        let meta = meta_for(format!(
            "filters:remove_region(4,4,10,8):remove_region(20,20,5,5):format(png)/{src}"
        ))
        .await;
        assert_eq!(
            meta["inpaint"]["regions"],
            serde_json::json!([[4, 4, 10, 8], [20, 20, 5, 5]])
        );
        // No removal, no inpaint record.
        let meta = meta_for(format!("40x24/{src}")).await;
        assert!(meta.get("inpaint").is_none());
    }

    #[tokio::test]
    async fn meta_annotations_reuse_the_requests_ai_permit() {
        // A removebg request (AI use) answering meta on a
        // one-permit AI gate: the annotations must not wait
        // out the full gate timeout and silently drop both
        // the tags and the detections (audit R2-29). They
        // once ran under the permit the request held for
        // its render; that permit now drains with the
        // render job itself, so by annotation time the gate
        // is free and the sections acquire it directly.
        let src = spawn_image_server().await;
        let base = spawn_app_with_ai_and_meta().await;
        let remainder = format!("filters:removebg:format(png)/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("Accept", "application/json; detections=1")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["tags"][0]["label"], "ocean");
        let boxes = meta["detections"].as_array().unwrap();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0]["detector"], "builtin");
    }

    #[derive(Debug)]
    struct FakeDetector;

    impl crate::ai::DetectionProvider for FakeDetector {
        fn knows_class(&self, name: &str) -> bool {
            name.eq_ignore_ascii_case("person")
        }
        fn describe(&self) -> Vec<String> {
            vec!["person".to_string()]
        }
        fn detect(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
            Ok(vec![crate::ai::Detection {
                label: "person".to_string(),
                score: 0.87,
                x: 1,
                y: 2,
                w: 30,
                h: 20,
            }])
        }
    }

    #[derive(Debug)]
    struct FailingDetector;

    impl crate::ai::DetectionProvider for FailingDetector {
        fn knows_class(&self, name: &str) -> bool {
            name.eq_ignore_ascii_case("person")
        }
        fn describe(&self) -> Vec<String> {
            vec!["person".to_string()]
        }
        fn detect(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
            Err(crate::ai::AiError::Inference(
                "detector exploded".to_string(),
            ))
        }
    }

    #[test]
    fn compose_annotation_picks_the_clear_winner() {
        let scores = |pairs: &[(&str, f32)]| -> Vec<crate::ai::TagScore> {
            pairs
                .iter()
                .map(|(label, score)| crate::ai::TagScore {
                    label: label.to_string(),
                    score: *score,
                })
                .collect()
        };
        let a = compose_annotation(
            &scores(&[("ocean", 0.30), ("desert", 0.05), ("forest", 0.02)]),
            0.05,
        );
        assert_eq!(a.tags.len(), 1);
        assert_eq!(a.tags[0].label, "ocean");
        assert!(a.tags[0].score > 0.99, "score {}", a.tags[0].score);
        assert_eq!(a.alt_text, "Ocean, desert, forest");
        // Logit gaps of 0.01 in cosine are 1.0 after the
        // scale: probs are roughly 0.66 / 0.24 / 0.09.
        let a = compose_annotation(&scores(&[("a", 0.10), ("b", 0.09), ("c", 0.08)]), 0.10);
        let names: Vec<&str> = a.tags.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
        // A flat distribution clears nothing but alt text
        // still names the best labels.
        let flat = compose_annotation(&scores(&[("a", 0.0), ("b", 0.0), ("c", 0.0)]), 0.9);
        assert!(flat.tags.is_empty());
        assert_eq!(flat.alt_text, "A, b, c");
    }

    #[test]
    fn ai_error_classes_map_to_distinct_statuses() {
        // AP-16: a model that ran but produced unusable output
        // is a 502; a runtime failure or a poisoned session
        // lock stays a 500; the log line keeps the typed cause.
        let status = |e: EngineError| engine_error_response(e, None).status();
        assert_eq!(
            status(EngineError::Inference(crate::ai::AiError::BadOutput(
                "unexpected shape".to_string()
            ))),
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(
            status(EngineError::Inference(crate::ai::AiError::Inference(
                "inference failed: boom".to_string()
            ))),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            status(EngineError::Inference(crate::ai::AiError::LockPoisoned)),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    async fn spawn_app_with_detector(detector: Arc<dyn crate::ai::DetectionProvider>) -> String {
        spawn_app_with_providers(Some(detector), Vec::new()).await
    }

    async fn spawn_app_with_providers(
        detect: Option<Arc<dyn crate::ai::DetectionProvider>>,
        detectors: Vec<NamedDetector>,
    ) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect,
            detectors,
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// An app combining presets and disk storage, with a
    /// caller-chosen memory cache size (0 disables the
    /// memory tier so disk behavior is observable).
    async fn spawn_app_full(
        presets: std::collections::HashMap<String, String>,
        dir: Option<&std::path::Path>,
        cache_bytes: usize,
    ) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(cache_bytes)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: dir.map(|d| {
                Arc::new(
                    crate::storage::DiskStore::open(
                        d,
                        16 * 1024 * 1024,
                        crate::url::entry_key(b"disk-key-test"),
                    )
                    .unwrap(),
                )
            }),
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets,
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn header(resp: &reqwest::Response, name: &str) -> String {
        resp.headers()
            .get(name)
            .map(|v| v.to_str().unwrap().to_string())
            .unwrap_or_default()
    }

    fn headers_of(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                axum::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                value.parse().unwrap(),
            );
        }
        map
    }

    #[test]
    fn width_hint_prefers_the_specific_sec_spelling() {
        // The per-image Sec- width beats everything.
        let map = headers_of(&[
            ("Sec-CH-Width", "800"),
            ("Width", "400"),
            ("Sec-CH-Viewport-Width", "1200"),
        ]);
        assert_eq!(width_hint(&map), Some(800));
        // Legacy spelling next.
        let map = headers_of(&[("Width", "400"), ("Viewport-Width", "1200")]);
        assert_eq!(width_hint(&map), Some(400));
        // Viewport widths are the fallback, Sec- first.
        let map = headers_of(&[("Sec-CH-Viewport-Width", "1200"), ("Viewport-Width", "900")]);
        assert_eq!(width_hint(&map), Some(1200));
        // Junk and out-of-range values are no hint, and a
        // junk specific header falls through to the next.
        let map = headers_of(&[("Sec-CH-Width", "wide"), ("Width", "300")]);
        assert_eq!(width_hint(&map), Some(300));
        let map = headers_of(&[("Sec-CH-Width", "0")]);
        assert_eq!(width_hint(&map), None);
        let map = headers_of(&[("Sec-CH-Width", "20000")]);
        assert_eq!(width_hint(&map), None);
        assert_eq!(width_hint(&HeaderMap::new()), None);
    }

    #[test]
    fn quantize_width_rounds_up_to_the_step() {
        assert_eq!(quantize_width(1, 100), 100);
        assert_eq!(quantize_width(100, 100), 100);
        assert_eq!(quantize_width(101, 100), 200);
        assert_eq!(quantize_width(250, 100), 300);
        assert_eq!(quantize_width(250, 1), 250);
        assert_eq!(quantize_width(250, 0), 250);
        // Rounding up never crosses the parser's cap.
        assert_eq!(quantize_width(16383, 2000), 16383);
        assert_eq!(quantize_width(16383, 100), 16383);
    }

    #[test]
    fn save_data_hint_reads_on_only() {
        assert!(save_data_hint(&headers_of(&[("Save-Data", "on")])));
        assert!(save_data_hint(&headers_of(&[("Save-Data", " On ")])));
        assert!(!save_data_hint(&headers_of(&[("Save-Data", "off")])));
        assert!(!save_data_hint(&HeaderMap::new()));
    }

    async fn get_with_headers(
        base: &str,
        remainder: &str,
        pairs: &[(&str, &str)],
    ) -> reqwest::Response {
        let mut req = client().get(format!("{base}/{}/{remainder}", sign(remainder)));
        for (name, value) in pairs {
            req = req.header(*name, *value);
        }
        req.send().await.unwrap()
    }

    #[test]
    fn oversized_source_is_a_bad_gateway() {
        // The request was a valid signed URL; the origin
        // object is what is too big (review AP-11).
        let resp = load_error_response(LoadError::TooLarge { limit: 10 }, None);
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn policy_refusal_body_names_no_resolution() {
        // The refusal reason can carry a server-side DNS
        // resolution; the image path answers with the same
        // generic body as the explain path (issue #196,
        // requirement 3), and the reason stays in the log.
        let resp = load_error_response(
            LoadError::Forbidden(
                "host localhost resolves to a non-public address (::1)".to_string(),
            ),
            None,
        );
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], b"source refused by policy");
    }

    #[test]
    fn busy_responses_carry_retry_after() {
        // Both gate 503s are built by busy_response; the
        // header is what lets a client back off instead of
        // retrying into the same full queue (review AP-9).
        let resp = busy_response("server busy", 5, None);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers()
                .get(header::RETRY_AFTER)
                .unwrap()
                .to_str()
                .unwrap(),
            "5"
        );
    }

    #[test]
    fn if_none_match_matching_rules() {
        let etag = "\"abc123\"";
        assert!(if_none_match_matches("\"abc123\"", etag));
        assert!(if_none_match_matches("W/\"abc123\"", etag));
        assert!(if_none_match_matches("*", etag));
        assert!(if_none_match_matches("\"other\", \"abc123\"", etag));
        assert!(if_none_match_matches(" \"other\" , W/\"abc123\" ", etag));
        assert!(!if_none_match_matches("\"other\"", etag));
        assert!(!if_none_match_matches("", etag));
    }

    #[tokio::test]
    async fn validators_attach_the_etag_and_answer_304_on_a_match() {
        let etag = etag_for_bytes(b"rendered bytes");
        // No conditional header: the ETag is attached and the
        // response is otherwise untouched.
        let mut resp = respond(
            ImageFormat::Png,
            None,
            bytes::Bytes::from(vec![1, 2, 3]),
            None,
        );
        finalize_validators(&mut resp, &etag, None);
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get(header::ETAG).unwrap(), etag.as_str());
        // A matching If-None-Match: 304, empty body, and the
        // content headers are gone.
        let mut resp = respond(
            ImageFormat::Png,
            None,
            bytes::Bytes::from(vec![1, 2, 3]),
            None,
        );
        finalize_validators(&mut resp, &etag, Some(etag.as_str()));
        assert_eq!(resp.status(), StatusCode::NOT_MODIFIED);
        assert!(resp.headers().get(header::CONTENT_TYPE).is_none());
        assert!(resp.headers().get(header::CONTENT_LENGTH).is_none());
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(body.is_empty());
        // A non-matching value: the full 200 goes out, ETag on.
        let mut resp = respond(
            ImageFormat::Png,
            None,
            bytes::Bytes::from(vec![1, 2, 3]),
            None,
        );
        finalize_validators(&mut resp, &etag, Some("\"other\""));
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], &[1, 2, 3]);
    }

    #[test]
    fn validators_skip_meta_and_error_responses() {
        let etag = etag_for_bytes(b"rendered bytes");
        // Meta responses are private, no-store: no ETag is
        // attached and a matching conditional never 304s them.
        let mut resp = respond(ImageFormat::Png, None, bytes::Bytes::from(vec![1]), None);
        resp.headers_mut()
            .insert(header::CACHE_CONTROL, "private, no-store".parse().unwrap());
        finalize_validators(&mut resp, &etag, Some(etag.as_str()));
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get(header::ETAG).is_none());
        // An error status is left alone too, public or not.
        let mut resp = respond(ImageFormat::Png, None, bytes::Bytes::from(vec![1]), None);
        *resp.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
        finalize_validators(&mut resp, &etag, Some(etag.as_str()));
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(resp.headers().get(header::ETAG).is_none());
    }

    #[tokio::test]
    async fn width_hint_sizes_an_unsized_url_and_separates_the_cache() {
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        // The source is 100x60 and the URL sizes nothing.
        let remainder = format!("filters:format(png)/{src}");
        let width_of = |resp: reqwest::Response| async {
            let bytes = resp.bytes().await.unwrap();
            image::load_from_memory(&bytes).unwrap().width()
        };
        // 250 quantizes to 300 (upscale is the engine's
        // default posture): proof the synthesized op ran.
        let resp = get_with_headers(&base, &remainder, &[("Sec-CH-Width", "250")]).await;
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        // The response advertises the hints it reads, so a
        // browser sends them next time.
        assert_eq!(
            header(&resp, "accept-ch"),
            "Sec-CH-Width, Sec-CH-DPR, Sec-CH-Viewport-Width"
        );
        assert_eq!(width_of(resp).await, 300);
        // 210 is the same quantized class: a hit, no refetch.
        let resp = get_with_headers(&base, &remainder, &[("Sec-CH-Width", "210")]).await;
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // 90 quantizes to 100: a different entry.
        let resp = get_with_headers(&base, &remainder, &[("Width", "90")]).await;
        assert_eq!(width_of(resp).await, 100);
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        // No hint at all: its own entry again.
        let resp = get_with_headers(&base, &remainder, &[]).await;
        assert_eq!(width_of(resp).await, 100);
        assert_eq!(hits.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn an_explicit_size_ignores_the_width_hint() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("30x20/{src}");
        let resp = get_with_headers(&base, &remainder, &[("Sec-CH-Width", "250")]).await;
        let bytes = resp.bytes().await.unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!((img.width(), img.height()), (30, 20));
    }

    #[tokio::test]
    async fn save_data_steps_quality_and_separates_the_cache() {
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:format(jpeg)/{src}");
        // Plain: no quality is reported on the default path.
        let resp = get_with_headers(&base, &remainder, &[]).await;
        assert_eq!(header(&resp, "x-crabcrop-quality"), "");
        let plain = resp.bytes().await.unwrap();
        // Save-Data: the step to 70 is reported and the bytes
        // are a different encode.
        let resp = get_with_headers(&base, &remainder, &[("Save-Data", "on")]).await;
        assert_eq!(header(&resp, "x-crabcrop-quality"), "70");
        let stepped = resp.bytes().await.unwrap();
        assert_ne!(plain, stepped);
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        // A repeat is the stepped entry, not the plain one.
        let resp = get_with_headers(&base, &remainder, &[("Save-Data", "on")]).await;
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        assert_eq!(resp.bytes().await.unwrap(), stepped);
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        // An explicit quality stands under Save-Data and
        // reports nothing, exactly as without the hint.
        let remainder = format!("filters:format(jpeg):quality(90)/{src}");
        let resp = get_with_headers(&base, &remainder, &[("Save-Data", "on")]).await;
        assert_eq!(header(&resp, "x-crabcrop-quality"), "");
    }

    #[tokio::test]
    async fn cli_verbs_work_against_a_live_server() {
        let cli = |parts: &[&str]| parts.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("50x50/{src}");
        // sign: the printed URL serves.
        let signed = crate::cli::exec(&cli(&[
            "sign",
            "--secret",
            "test-secret",
            "--base",
            &base,
            &remainder,
        ]))
        .await
        .unwrap();
        assert!(signed.starts_with(&format!("{base}/")), "{signed}");
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // explain: verified, with the operation described.
        let out = crate::cli::exec(&cli(&["explain", "--secret", "test-secret", &signed]))
            .await
            .unwrap();
        assert!(out.contains("valid for this remainder"), "{out}");
        assert!(out.contains("resize to 50x50"), "{out}");
        // meta: a remainder is signed, fetched, and printed.
        let meta = crate::cli::exec(&cli(&[
            "meta",
            "--secret",
            "test-secret",
            "--base",
            &base,
            &remainder,
        ]))
        .await
        .unwrap();
        assert!(meta.contains("\"width\": 50"), "{meta}");
        // purge: the entry the first fetch cached goes away.
        let out = crate::cli::exec(&cli(&[
            "purge",
            "--secret",
            "test-secret",
            "--base",
            &base,
            &signed,
        ]))
        .await
        .unwrap();
        assert_eq!(out, "purged 1");
        // purge-group: a preset fetch lands in the group,
        // and the group purge removes it.
        let base = spawn_app_with_presets(preset_test_map(), false).await;
        let remainder = format!("preset:small/{src}");
        let signed = crate::cli::exec(&cli(&[
            "sign",
            "--secret",
            "test-secret",
            "--base",
            &base,
            &remainder,
        ]))
        .await
        .unwrap();
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let out = crate::cli::exec(&cli(&[
            "purge-group",
            "--secret",
            "test-secret",
            "--base",
            &base,
            "small",
        ]))
        .await
        .unwrap();
        assert_eq!(out, "purged 1");
    }

    #[tokio::test]
    async fn cli_meta_reports_the_fixture_and_honors_its_flags() {
        let cli = |parts: &[&str]| parts.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("50x30/{src}");
        // The fixture is the 100x60 test PNG: the meta JSON
        // names the result's dimensions, its format, and the
        // source's own dimensions.
        let out = crate::cli::exec(&cli(&[
            "meta",
            "--secret",
            "test-secret",
            "--base",
            &base,
            &remainder,
        ]))
        .await
        .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(meta["width"], 50);
        assert_eq!(meta["height"], 30);
        assert_eq!(meta["source_width"], 100);
        assert_eq!(meta["source_height"], 60);
        assert_eq!(meta["format"], "png");
        assert!(meta["bytes"].as_u64().unwrap() > 0);
        // Neither opt-in was passed: both field groups stay out.
        assert!(meta.get("dominant_color").is_none());
        assert!(meta.get("placeholder").is_none());
        assert!(meta.get("detections").is_none());
        // A full signed URL passes through untouched, so meta
        // needs no secret of its own.
        let signed = crate::cli::exec(&cli(&[
            "sign",
            "--secret",
            "test-secret",
            "--base",
            &base,
            &remainder,
        ]))
        .await
        .unwrap();
        let out = crate::cli::exec(&cli(&["meta", &signed])).await.unwrap();
        let meta: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(meta["width"], 50);
        // --placeholders puts the placeholder fields in.
        let out = crate::cli::exec(&cli(&[
            "meta",
            "--secret",
            "test-secret",
            "--base",
            &base,
            "--placeholders",
            &remainder,
        ]))
        .await
        .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&out).unwrap();
        let color = meta["dominant_color"].as_str().unwrap();
        assert!(color.starts_with('#') && color.len() == 7, "{color}");
        assert!(
            meta["placeholder"]
                .as_str()
                .unwrap()
                .starts_with("data:image/jpeg;base64,")
        );
        assert!(!meta["palette"].as_array().unwrap().is_empty());
        assert!(meta["thumbhash"].as_str().unwrap().len() >= 20);
        // --detections against a server with no detector:
        // asked but unconfigured, the field stays out and the
        // meta still succeeds.
        let out = crate::cli::exec(&cli(&[
            "meta",
            "--secret",
            "test-secret",
            "--base",
            &base,
            "--detections",
            &remainder,
        ]))
        .await
        .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(meta.get("detections").is_none());
        assert_eq!(meta["width"], 50);
        // --detections with a detector configured: the fake's
        // box comes back in the printed JSON.
        let base = spawn_app_with_detector(Arc::new(FakeDetector)).await;
        let out = crate::cli::exec(&cli(&[
            "meta",
            "--secret",
            "test-secret",
            "--base",
            &base,
            "--detections",
            &remainder,
        ]))
        .await
        .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&out).unwrap();
        let boxes = meta["detections"].as_array().unwrap();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0]["label"], "person");
        assert_eq!(boxes[0]["x"], 1);
        assert_eq!(boxes[0]["w"], 30);
    }

    #[tokio::test]
    async fn cli_purge_evicts_so_the_next_fetch_misses() {
        let cli = |parts: &[&str]| parts.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let remainder = format!("40x24/{src}");
        let signed = crate::cli::exec(&cli(&[
            "sign",
            "--secret",
            "test-secret",
            "--base",
            &base,
            &remainder,
        ]))
        .await
        .unwrap();
        // Prime the entry: a miss, then a hit, one source fetch.
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // Purge by the full signed URL: the entry is gone.
        let out = crate::cli::exec(&cli(&[
            "purge",
            "--secret",
            "test-secret",
            "--base",
            &base,
            &signed,
        ]))
        .await
        .unwrap();
        assert_eq!(out, "purged 1");
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        // Purge by bare remainder: the CLI signs the purge
        // itself, and the refetched entry goes away again.
        let out = crate::cli::exec(&cli(&[
            "purge",
            "--secret",
            "test-secret",
            "--base",
            &base,
            &remainder,
        ]))
        .await
        .unwrap();
        assert_eq!(out, "purged 1");
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(hits.load(Ordering::SeqCst), 3);
        // A purge signed with the wrong secret is refused by
        // the server, surfaces as a CLI error, and the entry
        // survives.
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        let err = crate::cli::exec(&cli(&[
            "purge",
            "--secret",
            "the-wrong-secret",
            "--base",
            &base,
            &signed,
        ]))
        .await
        .unwrap_err();
        assert!(err.contains("HTTP 403"), "{err}");
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        assert_eq!(hits.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn cli_purge_group_evicts_every_grouped_entry() {
        let cli = |parts: &[&str]| parts.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app_with_presets(preset_test_map(), false).await;
        // Two distinct entries, both landing in the "small"
        // group, plus one ungrouped entry that must survive.
        let grouped = [
            format!("preset:small/{src}"),
            format!("preset:small/filters:grayscale/{src}"),
        ];
        let plain = format!("40x24/{src}");
        for remainder in grouped.iter().chain(std::iter::once(&plain)) {
            let resp = get_signed(&base, remainder).await;
            assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
            let resp = get_signed(&base, remainder).await;
            assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        }
        assert_eq!(hits.load(Ordering::SeqCst), 3);
        let out = crate::cli::exec(&cli(&[
            "purge-group",
            "--secret",
            "test-secret",
            "--base",
            &base,
            "small",
        ]))
        .await
        .unwrap();
        assert_eq!(out, "purged 2");
        // Both grouped entries miss again; the ungrouped one
        // still hits.
        for remainder in &grouped {
            let resp = get_signed(&base, remainder).await;
            assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        }
        assert_eq!(hits.load(Ordering::SeqCst), 5);
        let resp = get_signed(&base, &plain).await;
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        // An unknown group is a successful no-op.
        let out = crate::cli::exec(&cli(&[
            "purge-group",
            "--secret",
            "test-secret",
            "--base",
            &base,
            "nope",
        ]))
        .await
        .unwrap();
        assert_eq!(out, "purged 0");
    }

    #[tokio::test]
    async fn cli_purge_with_a_key_id_verifies_against_a_rotating_keyring() {
        let cli = |parts: &[&str]| parts.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app_rotating().await;
        let remainder = format!("25x25/{src}");
        // Mint the URL under the previous key, through the CLI.
        let signed = crate::cli::exec(&cli(&[
            "sign",
            "--secret",
            "the-old-server-secret",
            "--key-id",
            "k-old",
            "--base",
            &base,
            &remainder,
        ]))
        .await
        .unwrap();
        assert!(signed.contains("/k-old:"), "{signed}");
        // It serves and caches.
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // Purge naming the old key: the purge signature
        // verifies against the keyring's previous slot and
        // the entry goes away.
        let out = crate::cli::exec(&cli(&[
            "purge",
            "--secret",
            "the-old-server-secret",
            "--key-id",
            "k-old",
            "--base",
            &base,
            &signed,
        ]))
        .await
        .unwrap();
        assert_eq!(out, "purged 1");
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        // The old secret under the new key's name routes to
        // the wrong key: refused, and the entry survives.
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        let err = crate::cli::exec(&cli(&[
            "purge",
            "--secret",
            "the-old-server-secret",
            "--key-id",
            "k-new",
            "--base",
            &base,
            &signed,
        ]))
        .await
        .unwrap_err();
        assert!(err.contains("HTTP 403"), "{err}");
        let resp = client().get(&signed).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        // A purge under the current key verifies too.
        let out = crate::cli::exec(&cli(&[
            "purge",
            "--secret",
            "the-new-server-secret",
            "--key-id",
            "k-new",
            "--base",
            &base,
            &signed,
        ]))
        .await
        .unwrap();
        assert_eq!(out, "purged 1");
        // purge-group signs under a key ID the same way: the
        // rotating app has no presets, so a verified group
        // purge removes nothing, and the same secret under
        // the wrong key's name is refused.
        let out = crate::cli::exec(&cli(&[
            "purge-group",
            "--secret",
            "the-old-server-secret",
            "--key-id",
            "k-old",
            "--base",
            &base,
            "small",
        ]))
        .await
        .unwrap();
        assert_eq!(out, "purged 0");
        let err = crate::cli::exec(&cli(&[
            "purge-group",
            "--secret",
            "the-old-server-secret",
            "--key-id",
            "k-new",
            "--base",
            &base,
            "small",
        ]))
        .await
        .unwrap_err();
        assert!(err.contains("HTTP 403"), "{err}");
    }

    #[tokio::test]
    async fn cli_verbs_fail_loudly_on_bad_targets_and_arguments() {
        let cli = |parts: &[&str]| parts.iter().map(|s| s.to_string()).collect::<Vec<String>>();
        // A base nothing listens on: bind a port, then drop
        // the listener so connections are refused.
        let dead = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            drop(listener);
            format!("http://{addr}")
        };
        let target = "50x50/http://127.0.0.1:9/src.png";
        for verb in ["meta", "purge"] {
            let err = crate::cli::exec(&cli(&[
                verb,
                "--secret",
                "test-secret",
                "--base",
                &dead,
                target,
            ]))
            .await
            .unwrap_err();
            assert!(err.contains("request failed"), "{verb}: {err}");
        }
        let err = crate::cli::exec(&cli(&[
            "purge-group",
            "--secret",
            "test-secret",
            "--base",
            &dead,
            "small",
        ]))
        .await
        .unwrap_err();
        assert!(err.contains("request failed"), "{err}");

        let src = spawn_image_server().await;
        let base = spawn_app_with_presets(preset_test_map(), false).await;
        // Unknown preset: the CLI's own validation steps over
        // preset references, the server answers 400, and the
        // CLI surfaces the status instead of succeeding.
        let err = crate::cli::exec(&cli(&[
            "meta",
            "--secret",
            "test-secret",
            "--base",
            &base,
            &format!("preset:nope/{src}"),
        ]))
        .await
        .unwrap_err();
        assert!(err.contains("HTTP 400"), "{err}");
        // Malformed operations fail in the CLI's grammar
        // check, before any request is made.
        let err = crate::cli::exec(&cli(&[
            "meta",
            "--secret",
            "test-secret",
            "--base",
            &base,
            &format!("300x200/filters:bogus(1)/{src}"),
        ]))
        .await
        .unwrap_err();
        assert!(
            err.contains("invalid filters segment: filters:bogus(1)"),
            "{err}"
        );
        // A well-formed signature from the wrong secret: the
        // server refuses the meta fetch and the CLI reports it.
        let remainder = format!("50x50/{src}");
        let forged = format!(
            "{base}/{}/{remainder}",
            crabcrop_url::sign(b"some-other-secret", &remainder)
        );
        let err = crate::cli::exec(&cli(&["meta", &forged]))
            .await
            .unwrap_err();
        assert!(err.contains("HTTP 403"), "{err}");
        // No secret for a remainder target: refused locally.
        for verb in ["meta", "purge"] {
            let err = crate::cli::exec(&cli(&[verb, "--base", &base, &remainder]))
                .await
                .unwrap_err();
            assert!(err.contains("no secret"), "{verb}: {err}");
        }
        // Argument shape: missing and extra targets.
        let err = crate::cli::exec(&cli(&["meta", "--secret", "test-secret"]))
            .await
            .unwrap_err();
        assert!(err.contains("meta needs an argument"), "{err}");
        let err = crate::cli::exec(&cli(&[
            "purge",
            "--secret",
            "test-secret",
            &remainder,
            &remainder,
        ]))
        .await
        .unwrap_err();
        assert!(err.contains("takes exactly one argument"), "{err}");
        // The meta opt-ins belong to the meta verb only.
        let err = crate::cli::exec(&cli(&[
            "purge",
            "--secret",
            "test-secret",
            "--base",
            &base,
            "--placeholders",
            &remainder,
        ]))
        .await
        .unwrap_err();
        assert!(err.contains("only apply to the meta verb"), "{err}");
    }

    #[tokio::test]
    async fn cache_status_reports_miss_then_memory_hit() {
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let remainder = format!("40x24/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(header(&resp, "cache-group"), "");
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cache_status_reports_a_disk_hit_with_its_tier() {
        let dir = std::env::temp_dir().join(format!("crabcrop-srv-tier-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        // Memory tier disabled: the repeat must come from disk.
        let base = spawn_app_full(preset_test_map(), Some(&dir), 0).await;
        let remainder = format!("40x24/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(
            header(&resp, "cache-status"),
            "crabcrop; hit; detail=\"disk\""
        );
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn preset_responses_carry_the_group_and_group_purge_evicts() {
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app_with_presets(preset_test_map(), false).await;
        let remainder = format!("preset:small/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(header(&resp, "cache-group"), "\"small\"");
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(header(&resp, "cache-group"), "\"small\"");
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        // A bad signature refuses and removes nothing.
        let resp = client()
            .post(format!("{base}/purge-group/deadbeef/small"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 403);
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        // The real purge, signed under the purge-group subkey.
        let sig = crate::url::purge_group_signature(SECRET, "small");
        let resp = client()
            .post(format!("{base}/purge-group/{sig}/small"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert!(resp.text().await.unwrap().contains("purged 1"));
        // And the group is a miss again.
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        // Unknown group: 200, nothing removed.
        let sig = crate::url::purge_group_signature(SECRET, "nope");
        let resp = client()
            .post(format!("{base}/purge-group/{sig}/nope"))
            .send()
            .await
            .unwrap();
        assert!(resp.text().await.unwrap().contains("purged 0"));
    }

    #[tokio::test]
    async fn purge_group_refuses_transplanted_purge_signatures() {
        // Audit R2-14: purge-group verifies under its own
        // subkey, so a per-target purge signature, even one
        // minted over the literal string the endpoint's URL
        // carries, is a 403 there and removes nothing.
        let (src, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app_with_presets(preset_test_map(), false).await;
        let remainder = format!("preset:small/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        for transplanted in [
            crate::url::purge_signature(SECRET, "purge-group/small"),
            crate::url::purge_signature(SECRET, "small"),
        ] {
            let resp = client()
                .post(format!("{base}/purge-group/{transplanted}/small"))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 403);
        }
        // Nothing was evicted: the entry still hits.
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
    }

    #[tokio::test]
    async fn group_purge_reaches_the_disk_tier() {
        let dir = std::env::temp_dir().join(format!("crabcrop-srv-gdisk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app_full(preset_test_map(), Some(&dir), 0).await;
        let remainder = format!("preset:small/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(header(&resp, "cache-group"), "\"small\"");
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(
            header(&resp, "cache-status"),
            "crabcrop; hit; detail=\"disk\""
        );
        let sig = crate::url::purge_group_signature(SECRET, "small");
        let resp = client()
            .post(format!("{base}/purge-group/{sig}/small"))
            .send()
            .await
            .unwrap();
        assert!(resp.text().await.unwrap().contains("purged 1"));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn meta_includes_detections_only_when_asked_and_configured() {
        let src = spawn_image_server().await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let get_meta = |base: String, accept: &'static str| {
            let url_path = url_path.clone();
            async move {
                let resp = client()
                    .get(format!("{base}{url_path}"))
                    .header("Accept", accept)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                serde_json::from_str::<serde_json::Value>(&resp.text().await.unwrap()).unwrap()
            }
        };
        // Asked + configured: boxes present and shaped right.
        let base = spawn_app_with_detector(Arc::new(FakeDetector)).await;
        let meta = get_meta(base, "application/json; detections=1").await;
        let boxes = meta["detections"].as_array().unwrap();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0]["label"], "person");
        assert_eq!(boxes[0]["score"], 0.87);
        assert_eq!(boxes[0]["w"], 30);
        // The built-in detector attributes as "builtin"
        // (issue #166).
        assert_eq!(boxes[0]["detector"], "builtin");
        // Configured but not asked: absent.
        let base = spawn_app_with_detector(Arc::new(FakeDetector)).await;
        let meta = get_meta(base, "application/json").await;
        assert!(meta.get("detections").is_none());
        // Asked and configured but the detector fails: the
        // field is omitted, the response still succeeds.
        let base = spawn_app_with_detector(Arc::new(FailingDetector)).await;
        let meta = get_meta(base, "application/json; detections=1").await;
        assert!(meta.get("detections").is_none());
        assert_eq!(meta["width"], 40);
    }

    /// A declared detector's fake (issue #166): one crate
    /// box, disjoint from the built-in fake's person.
    #[derive(Debug)]
    struct FakeDeclaredDetector;

    impl crate::ai::DetectionProvider for FakeDeclaredDetector {
        fn knows_class(&self, name: &str) -> bool {
            name.eq_ignore_ascii_case("crate")
        }
        fn describe(&self) -> Vec<String> {
            vec!["crate".to_string()]
        }
        fn detect(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
            Ok(vec![crate::ai::Detection {
                label: "crate".to_string(),
                score: 0.5,
                x: 5,
                y: 6,
                w: 10,
                h: 12,
            }])
        }
    }

    #[tokio::test]
    async fn meta_aggregates_declared_detectors_with_attribution() {
        let src = spawn_image_server().await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let get_meta = |base: String| {
            let url_path = url_path.clone();
            async move {
                let resp = client()
                    .get(format!("{base}{url_path}"))
                    .header("Accept", "application/json; detections=1")
                    .send()
                    .await
                    .unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                serde_json::from_str::<serde_json::Value>(&resp.text().await.unwrap()).unwrap()
            }
        };
        let declared = vec![NamedDetector {
            name: "crates".to_string(),
            provider: Arc::new(FakeDeclaredDetector),
        }];
        // Built-in first, then declared in order, each box
        // carrying its provider's name.
        let base = spawn_app_with_providers(Some(Arc::new(FakeDetector)), declared).await;
        let meta = get_meta(base).await;
        let boxes = meta["detections"].as_array().unwrap();
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0]["label"], "person");
        assert_eq!(boxes[0]["detector"], "builtin");
        assert_eq!(boxes[1]["label"], "crate");
        assert_eq!(boxes[1]["detector"], "crates");
        assert_eq!(boxes[1]["x"], 5);
        // Declared alone (no built-in) still reports.
        let declared = vec![NamedDetector {
            name: "crates".to_string(),
            provider: Arc::new(FakeDeclaredDetector),
        }];
        let base = spawn_app_with_providers(None, declared).await;
        let meta = get_meta(base).await;
        let boxes = meta["detections"].as_array().unwrap();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0]["detector"], "crates");
        // A failing declared provider omits only its own
        // boxes; the built-in's still report.
        let declared = vec![NamedDetector {
            name: "broken".to_string(),
            provider: Arc::new(FailingDetector),
        }];
        let base = spawn_app_with_providers(Some(Arc::new(FakeDetector)), declared).await;
        let meta = get_meta(base).await;
        let boxes = meta["detections"].as_array().unwrap();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0]["detector"], "builtin");
    }

    #[test]
    fn meta_opt_ins_take_exact_tokens() {
        // Issue #151: bare, 1, or true (any case) opt in;
        // everything else does not, including values the old
        // loose check used to accept.
        let yes = [
            "application/json; placeholders",
            "application/json; placeholders=1",
            "application/json; placeholders=true",
            "application/json; placeholders=TRUE",
            "application/json; placeholders = 1",
            "text/html, application/json; placeholders=1",
            "application/json;q=0, application/json; placeholders=1",
        ];
        for accept in yes {
            assert!(wants_placeholders(Some(accept)), "{accept}");
        }
        let no = [
            "application/json",
            "application/json; placeholders=0",
            "application/json; placeholders=false",
            "application/json; placeholders=FALSE",
            "application/json; placeholders=banana",
            "application/json; placeholders=yes",
            "application/json; placeholders=2",
            "application/json; placeholders=1;q=0",
            "text/html; placeholders=1",
        ];
        for accept in no {
            assert!(!wants_placeholders(Some(accept)), "{accept}");
        }
        assert!(!wants_placeholders(None));
        assert!(wants_detections(Some("application/json; detections")));
        assert!(!wants_detections(Some("application/json; placeholders=1")));
    }

    #[tokio::test]
    async fn meta_includes_placeholders_only_when_asked() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("40x24/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        // Asked via the Accept parameter: both fields present.
        let resp = client()
            .get(&url)
            .header("Accept", "application/json; placeholders=1")
            .send()
            .await
            .unwrap();
        // Meta is never publicly cacheable (audit PL-12).
        assert_eq!(
            resp.headers().get("cache-control").unwrap(),
            "private, no-store"
        );
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        let color = meta["dominant_color"].as_str().unwrap();
        assert!(color.starts_with('#') && color.len() == 7, "{color}");
        assert!(
            meta["placeholder"]
                .as_str()
                .unwrap()
                .starts_with("data:image/jpeg;base64,")
        );
        let palette = meta["palette"].as_array().unwrap();
        assert!(!palette.is_empty() && palette.len() <= 5);
        for entry in palette {
            let c = entry.as_str().unwrap();
            assert!(c.starts_with('#') && c.len() == 7, "{c}");
        }
        let thumbhash = meta["thumbhash"].as_str().unwrap();
        assert!(thumbhash.len() >= 20, "{thumbhash}");
        // Not asked: both fields absent, plain fields unchanged.
        let resp = client()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert!(meta.get("dominant_color").is_none());
        assert!(meta.get("placeholder").is_none());
        assert!(meta.get("palette").is_none());
        assert!(meta.get("thumbhash").is_none());
        assert_eq!(meta["width"].as_u64(), Some(40));
    }

    #[tokio::test]
    async fn meta_reports_the_verified_score_for_auto_quality() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:quality(auto):format(jpeg)/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        let score = meta["ssimulacra2"].as_f64().expect("score reported");
        assert!((0.0..=100.0).contains(&score), "score {score}");
        // An explicit quality is never scored.
        let remainder = format!("filters:quality(80):format(jpeg)/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert!(meta.get("ssimulacra2").is_none());
    }

    #[tokio::test]
    async fn serves_watermarked_image() {
        let src = spawn_image_server().await;
        let wm = spawn_watermark_server().await;
        let base = spawn_app(false).await;
        // The watermark URL travels percent-encoded inside the
        // filters segment.
        let encoded = wm.replace(':', "%3A").replace('/', "%2F");
        let remainder = format!("filters:watermark({encoded},0,0,100)/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.bytes().await.unwrap();
        let img = image::load_from_memory(&body).unwrap().to_rgb8();
        // The blue 10x10 watermark sits at the top-left corner.
        assert_eq!(img.get_pixel(0, 0), &image::Rgb([0, 0, 255]));
        assert_eq!(img.get_pixel(9, 9), &image::Rgb([0, 0, 255]));
        assert_ne!(img.get_pixel(90, 50), &image::Rgb([0, 0, 255]));
    }

    #[tokio::test]
    async fn unreachable_watermark_fails_the_request() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let encoded = "http%3A%2F%2F127.0.0.1%3A1%2Fwm.png";
        let remainder = format!("filters:watermark({encoded},0,0,100)/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn undecodable_watermark_fails_the_request() {
        let src = spawn_image_server().await;
        // Non-image bytes under an image content type: the
        // fetch succeeds, so the failure must come from the
        // decode, which runs inside the gated engine job
        // and answers with the decode error mapping (audit
        // R2-16).
        let wm = spawn_bytes_server(b"this is not an image".to_vec())
            .await
            .replace("src.png", "wm.png");
        let base = spawn_app(false).await;
        let encoded = wm.replace(':', "%3A").replace('/', "%2F");
        let remainder = format!("filters:watermark({encoded},0,0,100)/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.text().await.unwrap(),
            "source image could not be decoded"
        );
    }

    /// An app with disk result storage rooted at `dir`.
    async fn spawn_app_with_storage(dir: &std::path::Path) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: Some(Arc::new(
                crate::storage::DiskStore::open(
                    dir,
                    16 * 1024 * 1024,
                    crate::url::entry_key(b"disk-key-test"),
                )
                .unwrap(),
            )),
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// An app with shared (S3) result storage against the
    /// fake endpoint, and optionally a disk tier at `dir`.
    /// An app state with a disk tier (when `dir` is set)
    /// and a shared tier at `endpoint`, unserved: the
    /// tier-chain tests drive it directly.
    fn tier_test_state(dir: Option<&std::path::Path>, endpoint: &str) -> AppState {
        AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: dir.map(|d| {
                Arc::new(
                    crate::storage::DiskStore::open(
                        d,
                        16 * 1024 * 1024,
                        crate::url::entry_key(b"disk-key-test"),
                    )
                    .unwrap(),
                )
            }),
            shared_store: Some(Arc::new(crate::s3store::S3Store::new(
                "results".into(),
                String::new(),
                Some(endpoint.into()),
                "us-east-1".into(),
                None,
                crate::url::entry_key(b"disk-key-test"),
            ))),
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        }
    }

    async fn spawn_app_with_shared(dir: Option<&std::path::Path>, endpoint: &str) -> String {
        let state = tier_test_state(dir, endpoint);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// A group-free entry as the tier-chain tests seed it.
    fn tier_test_entry() -> CachedImage {
        CachedImage {
            bytes: bytes::Bytes::copy_from_slice(b"tier-test-pixels"),
            format: crate::url::ImageFormat::WebP,
            quality: Some(80),
            groups: Vec::new(),
            etag: crate::cache::etag_for_bytes(b"tier-test-pixels"),
        }
    }

    /// Poll a lookup future exactly once, then drop it:
    /// the shape of a client disconnecting at the first
    /// await that cannot complete synchronously.
    fn poll_lookup_once(state: &AppState, key: &str, groups: &[String]) {
        let fut = lookup_tiers(state, key, groups);
        let mut fut = Box::pin(fut);
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        let polled = std::future::Future::poll(fut.as_mut(), &mut cx);
        assert!(polled.is_pending(), "the lookup must wait on a tier");
        drop(fut);
    }

    /// Wait until the shared tier's copy of `key` carries
    /// `group`, or fail: the propagation the dropped
    /// request owed the tier lands detached, or not at all.
    async fn shared_learns_group(shared: &crate::s3store::S3Store, key: &str, group: &str) {
        for _ in 0..50 {
            if let Some(image) = shared.get(key).await
                && image.groups.iter().any(|g| g == group)
            {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        panic!("the shared tier never learned the group {group}");
    }

    #[tokio::test]
    async fn dropped_lookup_still_propagates_a_disk_hit_group_merge() {
        let fake = crate::s3store::test_support::start_fake().await;
        let dir = std::env::temp_dir().join(format!("crabcrop-srv-drop3-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = tier_test_state(Some(&dir), &fake.endpoint);
        let store = state.store.clone().unwrap();
        let shared = state.shared_store.clone().unwrap();
        let key = "40x24/example.com/i.png";
        // Both persistent tiers hold the entry, group-free.
        store.put(key, &tier_test_entry());
        shared.put(key, &tier_test_entry()).await;
        // One poll starts the disk chain; the request is
        // then gone (issue #347). The merge, the shared
        // propagation, and the memory promotion still owe
        // (issue #350).
        poll_lookup_once(&state, key, &["thumb".to_string()]);
        shared_learns_group(&shared, key, "thumb").await;
        let promoted = state.cache.get(key).expect("promoted into memory");
        assert!(promoted.groups.iter().any(|g| g == "thumb"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn dropped_lookup_still_propagates_a_memory_hit_group_merge() {
        let fake = crate::s3store::test_support::start_fake().await;
        let dir = std::env::temp_dir().join(format!("crabcrop-srv-drop4-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = tier_test_state(Some(&dir), &fake.endpoint);
        let shared = state.shared_store.clone().unwrap();
        let key = "40x24/example.com/i.png";
        state.cache.put(key.to_string(), tier_test_entry());
        shared.put(key, &tier_test_entry()).await;
        // The memory hit's shared propagation is owed
        // before any wait (issue #347): one poll and a
        // drop cannot strand it.
        poll_lookup_once(&state, key, &["thumb".to_string()]);
        shared_learns_group(&shared, key, "thumb").await;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn disk_task_panic_degrades_and_counts_the_error() {
        let dir = std::env::temp_dir().join(format!("crabcrop-srv-panic-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = std::sync::Arc::new(
            crate::storage::DiskStore::open(
                &dir,
                16 * 1024 * 1024,
                crate::url::entry_key(b"disk-key-test"),
            )
            .unwrap(),
        );
        let before = store.errors();
        // A panicked disk operation degrades to None and
        // moves the tier error counter (issues #347, #367);
        // it does not take the request path down with it.
        let out: Option<usize> = store.blocking(|_| -> usize { panic!("test panic") }).await;
        assert_eq!(out, None);
        assert_eq!(store.errors(), before + 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A disk read that blocks must not stall the runtime
    /// worker (issue #347): the stored entry's file is
    /// swapped for a FIFO, so the read waits until a
    /// writer opens the pipe two seconds later, while
    /// /health has to answer within one.
    #[cfg(unix)]
    #[tokio::test]
    async fn blocked_disk_read_leaves_health_responsive() {
        let dir = std::env::temp_dir().join(format!("crabcrop-srv-fifo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app_with_storage(&dir).await;
        let remainder = format!("25x25/{src}");
        let url_path = format!("{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}/{url_path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        // A second instance on the same directory: its
        // memory is cold, so its requests reach the disk
        // tier. Its index is built now, from the real file.
        let base = spawn_app_with_storage(&dir).await;
        let bin = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .find(|e| e.path().extension().is_some_and(|x| x == "bin"))
            .expect("the stored entry file")
            .path();
        std::fs::remove_file(&bin).unwrap();
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&bin)
                .status()
                .unwrap()
                .success()
        );
        // The writer that releases the read two seconds
        // in. Detached on purpose: if the read never
        // happens the open blocks forever, and joining it
        // would hang the suite instead of failing a check.
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(2));
            let _ = std::fs::OpenOptions::new().write(true).open(&bin);
        });
        let pending_base = base.clone();
        let pending_path = url_path.clone();
        let pending = tokio::spawn(async move {
            client()
                .get(format!("{pending_base}/{pending_path}"))
                .send()
                .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let start = std::time::Instant::now();
        let health = client().get(format!("{base}/health")).send().await.unwrap();
        assert_eq!(health.status(), 200);
        assert!(
            start.elapsed() < std::time::Duration::from_secs(1),
            "/health waited {:?} behind a blocked disk read",
            start.elapsed()
        );
        let resp = pending.await.unwrap().unwrap();
        assert_eq!(resp.status(), 200);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn disk_results_survive_a_fresh_instance_and_purge_invalidates() {
        let dir = std::env::temp_dir().join(format!("crabcrop-srv-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let remainder = format!("25x25/{src}");
        let sig = sign(&remainder);
        let url_path = format!("{sig}/{remainder}");
        // First instance: processes and stores to disk.
        let base1 = spawn_app_with_storage(&dir).await;
        let resp = client()
            .get(format!("{base1}/{url_path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // Second instance, cold memory, same directory: the
        // disk tier answers and the origin is not touched.
        let base2 = spawn_app_with_storage(&dir).await;
        let resp = client()
            .get(format!("{base2}/{url_path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(hits.load(Ordering::SeqCst), 1, "served from disk");
        // A purge with a bad signature is a 403 and removes
        // nothing.
        let bad = client()
            .post(format!("{base2}/purge/deadbeef/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(bad.status(), 403);
        let resp = client()
            .get(format!("{base2}/{url_path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // A correctly signed purge removes it from both tiers,
        // so the next request refetches the origin. Purge
        // signatures use the derived purge key (audit P-1).
        let purge_sig = crate::url::purge_signature(SECRET, &remainder);
        let ok = client()
            .post(format!("{base2}/purge/{purge_sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(ok.status(), 200);
        assert!(ok.text().await.unwrap().contains("purged 2"));
        let resp = client()
            .get(format!("{base2}/{url_path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(hits.load(Ordering::SeqCst), 2, "purged: origin refetched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn default_ops_apply_under_the_url_and_key_post_merge() {
        let (src_host, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app_with_defaults("filters:format(webp):quality(80)", false).await;
        // A bare request takes the house format.
        let bare = src_host.to_string();
        let sig = sign(&bare);
        let resp = client()
            .get(format!("{base}/{sig}/{bare}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("content-type").unwrap(), "image/webp");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // The explicit URL spelling of the same merged
        // request hits the same cache entry: keys are
        // post-merge.
        let explicit = format!("filters:format(webp):quality(80)/{src_host}");
        let sig = sign(&explicit);
        let resp = client()
            .get(format!("{base}/{sig}/{explicit}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(hits.load(Ordering::SeqCst), 1, "post-merge key shared");
        // The URL wins a singleton directive over the default.
        let png = format!("filters:format(png)/{src_host}");
        let sig = sign(&png);
        let resp = client()
            .get(format!("{base}/{sig}/{png}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("content-type").unwrap(), "image/png");
        // A signed purge of the bare URL removes the merged
        // entry, so the next bare request refetches.
        let purge_sig = crate::url::purge_signature(SECRET, &bare);
        let ok = client()
            .post(format!("{base}/purge/{purge_sig}/{bare}"))
            .send()
            .await
            .unwrap();
        assert_eq!(ok.status(), 200);
        assert!(ok.text().await.unwrap().contains("purged 1"));
        let sig = sign(&bare);
        let resp = client()
            .get(format!("{base}/{sig}/{bare}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(hits.load(Ordering::SeqCst), 3, "purged entry refetched");
    }

    #[tokio::test]
    async fn default_ops_apply_in_presets_only_mode() {
        let (src_host, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app_with_defaults("filters:format(webp)", true).await;
        // No operations in the URL at all: presets-only has
        // nothing to reject, and the defaults still apply.
        let bare = src_host.to_string();
        let sig = sign(&bare);
        let resp = client()
            .get(format!("{base}/{sig}/{bare}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("content-type").unwrap(), "image/webp");
    }

    #[tokio::test]
    async fn rotation_window_verifies_both_keys_and_routes_by_key_id() {
        let (src_host, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let remainder = format!("25x25/{src_host}");
        let base = spawn_app_rotating().await;
        let get = |segment: String| {
            let url = format!("{base}/{segment}/{remainder}");
            async move { client().get(url).send().await.unwrap().status() }
        };
        let new_unkeyed = crabcrop_url::sign(b"the-new-server-secret", &remainder);
        let old_unkeyed = crabcrop_url::sign(b"the-old-server-secret", &remainder);
        // Unkeyed URLs from either era verify.
        assert_eq!(get(new_unkeyed.clone()).await, 200);
        assert_eq!(get(old_unkeyed.clone()).await, 200);
        // Keyed URLs verify against their named key.
        assert_eq!(get(format!("k-new:{new_unkeyed}")).await, 200);
        assert_eq!(get(format!("k-old:{old_unkeyed}")).await, 200);
        // A key ID routes strictly: the old MAC under the new
        // key's name is a forgery, and an unknown ID fails the
        // same way.
        assert_eq!(get(format!("k-new:{old_unkeyed}")).await, 403);
        assert_eq!(get(format!("k-who:{new_unkeyed}")).await, 403);
    }

    #[tokio::test]
    async fn shared_results_serve_a_second_replica_and_promote_to_disk() {
        let fake = crate::s3store::test_support::start_fake().await;
        let dir = std::env::temp_dir().join(format!("crabcrop-srv-shared-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let remainder = format!("25x25/{src}");
        let sig = sign(&remainder);
        let url_path = format!("{sig}/{remainder}");
        // Replica A: no local tiers worth the name, processes
        // and writes through to the shared bucket in the
        // background.
        let base_a = spawn_app_with_shared(None, &fake.endpoint).await;
        let resp = client()
            .get(format!("{base_a}/{url_path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // The write-through is spawned; wait for it to land.
        let mut landed = false;
        for _ in 0..50 {
            if !fake.objects.lock().unwrap().is_empty() {
                landed = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert!(landed, "the shared write-through never landed");
        // Replica B: cold memory, empty disk, same bucket. The
        // shared tier answers and the origin is not touched.
        let base_b = spawn_app_with_shared(Some(&dir), &fake.endpoint).await;
        let resp = client()
            .get(format!("{base_b}/{url_path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "served from the shared tier"
        );
        let status = resp
            .headers()
            .get("cache-status")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(
            status.contains("detail=\"shared\""),
            "cache-status: {status}"
        );
        // The shared hit promoted into B's disk: replica C has
        // only that disk and still does not refetch.
        let base_c = spawn_app_with_storage(&dir).await;
        let resp = client()
            .get(format!("{base_c}/{url_path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(hits.load(Ordering::SeqCst), 1, "promoted to disk");
        // A signed purge fans out to every tier B holds
        // (memory, disk, shared): three entries.
        let purge_sig = crate::url::purge_signature(SECRET, &remainder);
        let ok = client()
            .post(format!("{base_b}/purge/{purge_sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(ok.status(), 200);
        assert!(ok.text().await.unwrap().contains("purged 3"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_purge_reaches_the_shared_stores_but_not_another_process_memory() {
        // Issue #351: the purge guarantee is the receiving
        // process plus its configured persistent and shared
        // stores. A purge by one process removes the entry
        // from the shared disk dir and the shared bucket,
        // so a fresh process over the same stores cannot
        // resurrect it; it does not reach into another
        // process's memory, whose copy keeps serving.
        let fake = crate::s3store::test_support::start_fake().await;
        let dir = std::env::temp_dir().join(format!("crabcrop-srv-p351-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let remainder = format!("25x25/{src}");
        let sig = sign(&remainder);
        let url_path = format!("{sig}/{remainder}");
        // Process A renders: the entry lands in A's
        // memory, the shared disk dir, and (in the
        // background) the shared bucket.
        let base_a = spawn_app_with_shared(Some(&dir), &fake.endpoint).await;
        let resp = client()
            .get(format!("{base_a}/{url_path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // The shared write-through is spawned; wait for
        // it to land.
        let mut landed = false;
        for _ in 0..50 {
            if !fake.objects.lock().unwrap().is_empty() {
                landed = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert!(landed, "the shared write-through never landed");
        // Process B: a fresh instance over the same
        // stores, its memory empty. Its purge removes the
        // disk entry and the shared entry: purged 2.
        let base_b = spawn_app_with_shared(Some(&dir), &fake.endpoint).await;
        let purge_sig = crate::url::purge_signature(SECRET, &remainder);
        let ok = client()
            .post(format!("{base_b}/purge/{purge_sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(ok.status(), 200);
        assert!(ok.text().await.unwrap().contains("purged 2"));
        // A still serves its own memory copy without
        // touching the origin: B's purge did not reach
        // into A's process.
        let resp = client()
            .get(format!("{base_a}/{url_path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        assert_eq!(hits.load(Ordering::SeqCst), 1, "A answered from memory");
        // Process C: fresh memory over the purged
        // stores. Nothing is left to resurrect, so the
        // request misses every tier and re-renders from
        // the origin.
        let base_c = spawn_app_with_shared(Some(&dir), &fake.endpoint).await;
        let resp = client()
            .get(format!("{base_c}/{url_path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "C re-rendered from the origin"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn system_endpoints_carry_security_headers() {
        let base = spawn_app(false).await;
        for path in ["/health", "/metrics"] {
            let resp = client().get(format!("{base}{path}")).send().await.unwrap();
            assert_eq!(
                resp.headers().get("x-content-type-options").unwrap(),
                "nosniff"
            );
            assert_eq!(resp.headers().get("cache-control").unwrap(), "no-store");
        }
        // Image responses get nosniff too, and keep their
        // immutable caching (audits PL-5, PL-6).
        let src = spawn_image_server().await;
        let remainder = format!("100x50/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get("x-content-type-options").unwrap(),
            "nosniff"
        );
        assert!(
            resp.headers()
                .get("cache-control")
                .unwrap()
                .to_str()
                .unwrap()
                .contains("immutable")
        );
    }

    #[tokio::test]
    async fn purges_leave_a_metric_trace() {
        // Audit PL-8: accepted and rejected purges both count.
        let base = spawn_app(false).await;
        let target = "100x50/http://example.com/x.jpg";
        let bad = client()
            .post(format!("{base}/purge/deadbeef/{target}"))
            .send()
            .await
            .unwrap();
        assert_eq!(bad.status(), StatusCode::FORBIDDEN);
        let sig = crate::url::purge_signature(SECRET, target);
        let ok = client()
            .post(format!("{base}/purge/{sig}/{target}"))
            .send()
            .await
            .unwrap();
        assert_eq!(ok.status(), 200);
        let metrics = metrics_text(&base).await;
        assert!(metrics.contains("crabcrop_purges_total 2"), "{metrics}");
    }

    /// A process job with the provenance knobs set
    /// explicitly, for the direct `apply_provenance`
    /// tests below (issue #173).
    fn provenance_job(
        signer: Option<Arc<crate::provenance::ProvenanceSigner>>,
        ai_edited: bool,
    ) -> ProcessJob {
        ProcessJob {
            save_data: false,
            preserve_metadata: false,
            brand_color: None,
            ops: Vec::new(),
            preferred: None,
            watermark_sources: Vec::new(),
            placeholders: false,
            mask_provider: None,
            upscale_provider: None,
            inpaint_provider: None,
            segment_provider: None,
            face_provider: None,
            detect_provider: None,
            declared_detect_providers: Vec::new(),
            focal_provider: None,
            text_renderer: None,
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            provenance_signer: signer,
            ai_edited,
            want_provenance: true,
            dpr: None,
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
        }
    }

    fn still_processed(bytes: Vec<u8>, format: ImageFormat) -> crate::engine::ProcessedImage {
        crate::engine::ProcessedImage {
            bytes,
            format,
            quality: None,
            width: 100,
            height: 60,
            source_width: 100,
            source_height: 60,
            placeholder: None,
            studio: None,
            inpaint: None,
            segments: None,
            objects_crop: None,
            ssimulacra2: None,
            video: None,
            pdf: None,
            provenance: None,
        }
    }

    #[test]
    fn provenance_step_signs_a_still_and_records_the_facts() {
        let job = provenance_job(Some(crate::provenance::test_signer()), false);
        let source = test_png();
        let mut processed = still_processed(source.clone(), ImageFormat::Png);
        apply_provenance(&job, &source, &mut processed).unwrap();
        let facts = processed.provenance.expect("facts are recorded");
        assert!(facts.signing_configured);
        assert!(facts.source.is_none(), "the source carried no manifest");
        let output = facts.output.expect("the output is signed");
        assert_eq!(output.state, "valid");
        assert_eq!(output.signer.as_deref(), Some("crabcrop test signer"));
        assert_eq!(output.manifests, 1);
        assert_eq!(output.ingredients, 1, "the source asset is the ingredient");
    }

    #[test]
    fn provenance_step_carries_a_signed_source_chain() {
        let signer = crate::provenance::test_signer();
        let parent = signer
            .sign(
                &test_png(),
                "image/png",
                &crate::provenance::SignRequest {
                    ingredient: None,
                    ai_edited: false,
                },
            )
            .unwrap();
        let job = provenance_job(Some(signer), false);
        let mut processed = still_processed(test_png(), ImageFormat::Png);
        apply_provenance(&job, &parent, &mut processed).unwrap();
        let facts = processed.provenance.expect("facts are recorded");
        let source = facts.source.expect("the source manifest is reported");
        assert_eq!(source.state, "valid");
        let output = facts.output.expect("the output is signed");
        assert_eq!(output.state, "valid");
        assert_eq!(output.manifests, 2, "the parent store rides along");
        assert_eq!(output.ingredients, 1);
    }

    #[test]
    fn provenance_step_marks_ai_edits() {
        let job = provenance_job(Some(crate::provenance::test_signer()), true);
        let source = test_png();
        let mut processed = still_processed(source.clone(), ImageFormat::Png);
        apply_provenance(&job, &source, &mut processed).unwrap();
        let json = crate::provenance::manifest_json(&processed.bytes, "image/png");
        assert!(
            json.contains("compositeWithTrainedAlgorithmicMedia"),
            "the edit action records the IPTC source type: {json}"
        );
    }

    #[test]
    fn provenance_step_leaves_unsignable_outputs_alone() {
        let job = provenance_job(Some(crate::provenance::test_signer()), false);
        let source = test_png();
        // Video output: not a signable format.
        let mut video = still_processed(b"video-bytes".to_vec(), ImageFormat::Mp4);
        apply_provenance(&job, &source, &mut video).unwrap();
        assert_eq!(video.bytes, b"video-bytes");
        let facts = video.provenance.expect("facts are recorded");
        assert!(facts.output.is_none());
        // Animated WebP: a signable format, but the bytes
        // are an animation, and signing covers stills.
        let mut anim = b"RIFF\x00\x00\x00\x00WEBPVP8X".to_vec();
        anim.extend_from_slice(&[0, 0, 0, 0, 0x02, 0, 0, 0]);
        let before = anim.clone();
        let mut processed = still_processed(anim, ImageFormat::WebP);
        apply_provenance(&job, &source, &mut processed).unwrap();
        assert_eq!(processed.bytes, before);
        assert!(processed.provenance.unwrap().output.is_none());
    }

    #[test]
    fn provenance_step_without_a_signer_only_verifies() {
        let signer = crate::provenance::test_signer();
        let parent = signer
            .sign(
                &test_png(),
                "image/png",
                &crate::provenance::SignRequest {
                    ingredient: None,
                    ai_edited: false,
                },
            )
            .unwrap();
        let job = provenance_job(None, false);
        let mut processed = still_processed(test_png(), ImageFormat::Png);
        let before = processed.bytes.clone();
        apply_provenance(&job, &parent, &mut processed).unwrap();
        assert_eq!(processed.bytes, before, "no signer, no signing");
        let facts = processed.provenance.expect("facts are recorded");
        assert!(!facts.signing_configured);
        assert_eq!(facts.source.expect("source reported").state, "valid");
        assert!(facts.output.is_none());
    }

    /// An app whose state carries the C2PA test signing
    /// identity (issue #173).
    async fn spawn_app_with_signer() -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: Some(crate::provenance::test_signer()),
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn meta_reports_provenance_without_a_signer() {
        let signer = crate::provenance::test_signer();
        let signed_source = signer
            .sign(
                &test_png(),
                "image/png",
                &crate::provenance::SignRequest {
                    ingredient: None,
                    ai_edited: false,
                },
            )
            .unwrap();
        let src = spawn_bytes_server(signed_source).await;
        let base = spawn_app(false).await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        let prov = &meta["provenance"];
        assert_eq!(prov["signing_configured"], false);
        assert_eq!(prov["source"]["state"], "valid");
        assert_eq!(prov["source"]["signer"], "crabcrop test signer");
        assert!(prov["output"].is_null());
    }

    #[tokio::test]
    async fn signing_server_signs_image_responses() {
        let base = spawn_app_with_signer().await;
        let src = spawn_image_server().await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(format!("{base}{url_path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let bytes = resp.bytes().await.unwrap().to_vec();
        let summary = crate::provenance::read_manifest(&bytes, "image/png")
            .expect("the response carries a manifest");
        assert_eq!(summary.state, "valid");
        assert_eq!(summary.signer.as_deref(), Some("crabcrop test signer"));
        assert_eq!(summary.ingredients, 1);
        // Meta agrees about the output side.
        let resp = client()
            .get(format!("{base}{url_path}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(meta["provenance"]["signing_configured"], true);
        assert_eq!(meta["provenance"]["output"]["state"], "valid");
        assert!(meta["provenance"]["source"].is_null());
    }

    #[tokio::test]
    async fn processing_gate_bounds_concurrency() {
        // A state whose single process permit the test holds:
        // a gated run must wait, then proceed once released
        // (audit E-5).
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(1)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let job = ProcessJob {
            save_data: false,
            preserve_metadata: false,
            brand_color: None,
            ops: Vec::new(),
            preferred: None,
            watermark_sources: Vec::new(),
            placeholders: false,
            mask_provider: None,
            upscale_provider: None,
            inpaint_provider: None,
            segment_provider: None,
            face_provider: None,
            detect_provider: None,
            declared_detect_providers: Vec::new(),
            focal_provider: None,
            text_renderer: None,
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            provenance_signer: None,
            ai_edited: false,
            want_provenance: false,
            dpr: None,
            ssim_target: state.ssim_target,
            avif_speed: state.avif_speed,
            jxl_speed: state.jxl_speed,
        };
        let held = state.process_gate.clone().acquire_owned().await.unwrap();
        let png = test_png();
        let waiting = run_process(
            &state,
            &job,
            &png,
            Lane::Process,
            std::time::Instant::now() + state.process_timeout,
            JobPermits::default(),
        );
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(300), &mut waiting)
                .await
                .is_err(),
            "the gated run must wait while the only permit is held"
        );
        drop(held);
        let processed = waiting.await.unwrap();
        assert_eq!((processed.width, processed.height), (100, 60));
    }

    #[tokio::test]
    async fn timed_out_encode_releases_the_process_gate() {
        // Issue #322, the #310 soak scenario in miniature:
        // one encode runs far past the processing deadline
        // inside a single blocking stage (here the external
        // video encoder, a script that drains its input,
        // sleeps 9 seconds, and only then answers), and a
        // follow-up request arrives while it is stuck. The
        // stuck request must answer TimedOut at its deadline
        // (plus the grace bound) instead of whenever the
        // encode happens to return, and the follow-up must
        // get a genuine encode attempt on the freed gate
        // instead of the gate-held 503 (Busy) the soak's
        // follow-up requests got.
        use std::io::Write as _;
        let dir = std::env::temp_dir().join(format!(
            "crabcrop-server-stuck-encode-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let script_path = dir.join("encoder.sh");
        let mut file = std::fs::File::create(&script_path).unwrap();
        file.write_all(b"#!/bin/sh\ncat > /dev/null\nsleep 9\nprintf 'x'\n")
            .unwrap();
        drop(file);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let encoder = crate::video::VideoEncoder::new(
            script_path,
            std::time::Duration::from_secs(30),
            64 * 1024 * 1024,
        )
        .unwrap();
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_millis(300),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(1)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let job = |video_encoder: Option<Arc<crate::video::VideoEncoder>>, ops_str: &str| {
            let (ops, _) = crate::url::parse_operations(ops_str).unwrap();
            ProcessJob {
                save_data: false,
                preserve_metadata: false,
                brand_color: None,
                ops,
                preferred: None,
                watermark_sources: Vec::new(),
                placeholders: false,
                mask_provider: None,
                upscale_provider: None,
                inpaint_provider: None,
                segment_provider: None,
                face_provider: None,
                detect_provider: None,
                declared_detect_providers: Vec::new(),
                focal_provider: None,
                text_renderer: None,
                video_encoder,
                frame_extractor: None,
                pdf_renderer: None,
                provenance_signer: None,
                ai_edited: false,
                want_provenance: false,
                dpr: None,
                ssim_target: state.ssim_target,
                avif_speed: state.avif_speed,
                jxl_speed: state.jxl_speed,
            }
        };
        let stuck_job = job(
            Some(Arc::new(encoder)),
            "filters:format(mp4)/example.com/i.gif",
        );
        let follow_job = job(None, "example.com/i.png");
        let gif = animated_gif_bytes();
        let png = test_png();
        let started = Instant::now();
        let (first, (follow_elapsed, follow)) = tokio::join!(
            run_process(
                &state,
                &stuck_job,
                &gif,
                Lane::Slow,
                std::time::Instant::now() + state.process_timeout,
                JobPermits::default()
            ),
            async {
                // Arrive while the first encode is stuck,
                // well past its 300 ms deadline.
                tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                let follow_start = Instant::now();
                let result = run_process(
                    &state,
                    &follow_job,
                    &png,
                    Lane::Process,
                    std::time::Instant::now() + state.process_timeout,
                    JobPermits::default(),
                )
                .await;
                (follow_start.elapsed(), result)
            }
        );
        assert!(
            matches!(first, Err(ProcessFailure::TimedOut)),
            "the stuck encode must answer TimedOut at its deadline, got {first:?}"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(7),
            "the timeout and the follow-up must not wait out the 9 s encode"
        );
        let processed = follow.unwrap_or_else(|e| {
            panic!("the follow-up must get a genuine encode attempt, got {e:?}")
        });
        assert_eq!((processed.width, processed.height), (100, 60));
        assert!(
            follow_elapsed < PROCESS_GATE_WAIT,
            "the follow-up must not shed on the gate the dead request held"
        );
        assert_eq!(state.process_gate.available_permits(), 1);
        // Issue #357: the stuck video encode drew its
        // permit from the slow lane, and the permit stays
        // with the detached job while it drains, so one
        // of the two is still out at this point. It comes
        // back when the job itself finishes (the encoder
        // script exits at nine seconds).
        assert_eq!(state.slow_gate.available_permits(), 1);
        let mut back = false;
        for _ in 0..200 {
            if state.slow_gate.available_permits() == 2 {
                back = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert!(back, "the slow permit must return when the job drains");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An upscale fake that blocks until the test lets it
    /// through, then really doubles the image: the drain
    /// test's stand-in for a long inference run.
    #[derive(Debug)]
    struct BlockingUpscale {
        release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    }

    impl crate::ai::UpscaleProvider for BlockingUpscale {
        fn factor(&self) -> u32 {
            2
        }

        fn upscale(
            &self,
            img: &image::DynamicImage,
        ) -> Result<image::DynamicImage, crate::ai::AiError> {
            let (lock, cvar) = &*self.release;
            let mut open = lock.lock().unwrap();
            while !*open {
                open = cvar.wait(open).unwrap();
            }
            Ok(image::DynamicImage::ImageRgba8(image::imageops::resize(
                &img.to_rgba8(),
                img.width() * 2,
                img.height() * 2,
                image::imageops::FilterType::Nearest,
            )))
        }
    }

    #[tokio::test]
    async fn an_ai_permit_is_held_until_the_job_actually_finishes() {
        // The client is shed at the deadline (plus grace)
        // while the inference is still blocked. The AI
        // permit the run drew must stay with the detached
        // job until the job itself finishes: freed at the
        // shed, it would let new AI work stack onto a
        // provider the gate believes is idle. Only when
        // the job drains does the permit come back.
        let state = lane_test_state(std::time::Duration::from_millis(300), 1, 1);
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let mut job = lane_test_job(None, "filters:upscale(2)/example.com/i.png");
        job.upscale_provider = Some(Arc::new(BlockingUpscale {
            release: release.clone(),
        }));
        let png = test_png();
        let permit = state.ai_gate.clone().acquire_owned().await.unwrap();
        assert_eq!(state.ai_gate.available_permits(), 1);
        let result = run_process(
            &state,
            &job,
            &png,
            Lane::Process,
            std::time::Instant::now() + state.process_timeout,
            JobPermits {
                ai: Some(permit),
                ..Default::default()
            },
        )
        .await;
        assert!(
            matches!(result, Err(ProcessFailure::TimedOut)),
            "the blocked run must answer TimedOut at its deadline, got {result:?}"
        );
        assert_eq!(
            state.ai_gate.available_permits(),
            1,
            "the permit must stay with the detached job after the shed"
        );
        // Let the inference through: the job finishes and
        // the permit comes back.
        {
            let (lock, cvar) = &*release;
            *lock.lock().unwrap() = true;
            cvar.notify_all();
        }
        let mut back = false;
        for _ in 0..100 {
            if state.ai_gate.available_permits() == 2 {
                back = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(back, "the permit must return when the job drains");
    }

    #[tokio::test]
    async fn video_and_animation_permits_are_held_until_the_job_finishes() {
        // Issue #357: the video and animation permits
        // serve_miss acquired before the run are accounted
        // like the AI permit: they travel with the job and
        // drop only when it finishes, so a request shed at
        // its deadline leaves them held while the detached
        // encode drains instead of freeing them for a
        // second encode to stack onto.
        let dir = std::env::temp_dir().join(format!(
            "crabcrop-server-held-permits-{}",
            std::process::id()
        ));
        let encoder = fake_long_encoder(&dir, 9);
        let state = lane_test_state(std::time::Duration::from_millis(300), 2, 1);
        let job = lane_test_job(
            Some(Arc::new(encoder)),
            "filters:format(mp4)/example.com/i.gif",
        );
        let gif = animated_gif_bytes();
        let video_permit = state.video_gate.clone().acquire_owned().await.unwrap();
        let anim_permit = state.anim_gate.clone().acquire_owned().await.unwrap();
        assert_eq!(state.video_gate.available_permits(), 1);
        assert_eq!(state.anim_gate.available_permits(), 0);
        let result = run_process(
            &state,
            &job,
            &gif,
            Lane::Slow,
            std::time::Instant::now() + state.process_timeout,
            JobPermits {
                ai: None,
                video: Some(video_permit),
                anim: Some(anim_permit),
            },
        )
        .await;
        assert!(
            matches!(result, Err(ProcessFailure::TimedOut)),
            "the stuck encode must answer TimedOut at its deadline, got {result:?}"
        );
        assert_eq!(
            state.video_gate.available_permits(),
            1,
            "the video permit must stay with the detached job after the shed"
        );
        assert_eq!(
            state.anim_gate.available_permits(),
            0,
            "the animation permit must stay with the detached job after the shed"
        );
        // Both come back when the job itself finishes
        // (the encoder script exits at nine seconds).
        let mut back = false;
        for _ in 0..200 {
            if state.video_gate.available_permits() == 2 && state.anim_gate.available_permits() == 1
            {
                back = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert!(back, "the permits must return when the job drains");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn repeated_timeouts_never_stack_jobs_beyond_the_lane() {
        // Issue #357 acceptance, at the lane seam: two
        // runs in a row time out against a slow encode.
        // The fake encoder logs a start and an end marker
        // per run, so the worst overlap in the log is the
        // number of live jobs itself: it must never exceed
        // the slow lane's single permit, even though the
        // first run's answer is long gone while its encode
        // is still executing. The timings are chosen so
        // the scenario is real: the encode (6 s) outlasts
        // the first run's deadline plus grace (3.5 + 2 s),
        // so the first run detaches mid-encode, and the
        // second run, arriving at 3 s with the same 3.5 s
        // budget, still has budget left when the lane
        // drains at 6 s, so it enters its own encode
        // instead of aborting at its first checkpoint.
        use std::io::Write as _;
        let dir =
            std::env::temp_dir().join(format!("crabcrop-server-lane-bound-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let log_path = dir.join("runs.log");
        let script_path = dir.join("encoder.sh");
        let mut file = std::fs::File::create(&script_path).unwrap();
        file.write_all(
            format!(
                "#!/bin/sh\ncat > /dev/null\necho S >> {}\nsleep 6\necho E >> {}\nprintf 'x'\n",
                log_path.display(),
                log_path.display()
            )
            .as_bytes(),
        )
        .unwrap();
        drop(file);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let encoder = crate::video::VideoEncoder::new(
            script_path,
            std::time::Duration::from_secs(30),
            64 * 1024 * 1024,
        )
        .unwrap();
        let state = lane_test_state(std::time::Duration::from_millis(3500), 2, 1);
        let job = lane_test_job(
            Some(Arc::new(encoder)),
            "filters:format(mp4)/example.com/i.gif",
        );
        let gif = animated_gif_bytes();
        let (first, second) = tokio::join!(
            run_process(
                &state,
                &job,
                &gif,
                Lane::Slow,
                std::time::Instant::now() + state.process_timeout,
                JobPermits::default()
            ),
            async {
                // Arrive while the first encode is running
                // and queue behind it at the lane gate.
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                run_process(
                    &state,
                    &job,
                    &gif,
                    Lane::Slow,
                    std::time::Instant::now() + state.process_timeout,
                    JobPermits::default(),
                )
                .await
            }
        );
        assert!(
            matches!(first, Err(ProcessFailure::TimedOut)),
            "the first run must answer TimedOut at its deadline, got {first:?}"
        );
        assert!(
            matches!(second, Err(ProcessFailure::TimedOut)),
            "the second run must answer TimedOut at its deadline, got {second:?}"
        );
        // Both encodes run their course (six seconds
        // each, back to back): wait for the log to close.
        let mut log = String::new();
        for _ in 0..200 {
            log = std::fs::read_to_string(&log_path).unwrap_or_default();
            if log.matches('E').count() == 2 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        let mut live = 0i32;
        let mut worst = 0i32;
        for marker in log.chars() {
            match marker {
                'S' => {
                    live += 1;
                    worst = worst.max(live);
                }
                'E' => live -= 1,
                _ => {}
            }
        }
        assert_eq!(
            log.matches('S').count(),
            2,
            "both encodes must have run; log: {log}"
        );
        assert_eq!(
            worst, 1,
            "live jobs must never exceed the lane permit count; log: {log}"
        );
        assert_eq!(state.slow_gate.available_permits(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Open the release the blocking provider doubles
    /// below wait behind (the `BlockingUpscale` pattern,
    /// shared).
    fn open_release(release: &Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>) {
        let (lock, cvar) = &**release;
        *lock.lock().unwrap() = true;
        cvar.notify_all();
    }

    fn wait_for_open(release: &Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>) {
        let (lock, cvar) = &**release;
        let mut open = lock.lock().unwrap();
        // Bounded on purpose: if a test fails before it
        // opens the release, the double still returns, so
        // the suite reports the failure instead of parking
        // a blocking thread (and the runtime's shutdown)
        // forever.
        while !*open {
            let (guard, waited) = cvar
                .wait_timeout(open, std::time::Duration::from_secs(120))
                .unwrap();
            open = guard;
            if waited.timed_out() {
                break;
            }
        }
    }

    /// A tagger double whose inference blocks until the
    /// test opens its release: the meta drain tests'
    /// stand-in for a slow model run.
    #[derive(Debug)]
    struct BlockingTagger {
        release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    }

    impl crate::ai::TagProvider for BlockingTagger {
        fn scores(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<Vec<crate::ai::TagScore>, crate::ai::AiError> {
            wait_for_open(&self.release);
            Ok(vec![crate::ai::TagScore {
                label: "ocean".to_string(),
                score: 0.3,
            }])
        }
    }

    /// A moderation double in the same shape.
    #[derive(Debug)]
    struct BlockingModeration {
        release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    }

    impl crate::ai::ModerationProvider for BlockingModeration {
        fn moderate(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<crate::ai::ModerationScores, crate::ai::AiError> {
            wait_for_open(&self.release);
            Ok(crate::ai::ModerationScores {
                model: "blocking-model".to_string(),
                scores: vec![crate::ai::TagScore {
                    label: "normal".to_string(),
                    score: 0.99,
                }],
                signals: Vec::new(),
            })
        }
    }

    /// A detection double in the same shape.
    #[derive(Debug)]
    struct BlockingDetector {
        release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    }

    impl crate::ai::DetectionProvider for BlockingDetector {
        fn knows_class(&self, _name: &str) -> bool {
            true
        }

        fn describe(&self) -> Vec<String> {
            vec!["object".to_string()]
        }

        fn detect(
            &self,
            _img: &image::DynamicImage,
        ) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
            wait_for_open(&self.release);
            Ok(Vec::new())
        }
    }

    /// Assert the meta drain property for one timed-out
    /// section: its AI gate slot stays accounted while
    /// the inference is still blocked, and returns when
    /// the inference returns. Called by the three tests
    /// below, one per helper shape.
    async fn assert_meta_slot_drains_with_inference(
        state: &AppState,
        release: &Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
        timed_out: bool,
    ) {
        assert!(
            timed_out,
            "the section must stop waiting at the caller's deadline"
        );
        assert_eq!(
            state.ai_gate.available_permits(),
            0,
            "the slot must stay accounted while the inference drains"
        );
        open_release(release);
        let mut back = false;
        for _ in 0..100 {
            if state.ai_gate.available_permits() == 1 {
                back = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(back, "the slot must return when the inference returns");
    }

    #[tokio::test]
    async fn a_timed_out_annotation_keeps_its_ai_gate_slot() {
        // Issue #357 acceptance, at the meta seam: the
        // caller's annotation deadline drops the helper
        // future while its inference is still blocked.
        // The section's gate slot must stay accounted
        // until the inference itself returns: freed at
        // the deadline, it would let the next section
        // stack onto a provider the gate believes idle.
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let mut state = lane_test_state(std::time::Duration::from_secs(30), 4, 2);
        state.ai_gate = Arc::new(tokio::sync::Semaphore::new(1));
        state.tagger = Some(Arc::new(BlockingTagger {
            release: release.clone(),
        }));
        let processed = still_processed(test_png(), ImageFormat::Png);
        let timed_out = tokio::time::timeout_at(
            tokio::time::Instant::now() + std::time::Duration::from_millis(300),
            annotate_meta(&state, &processed, false),
        )
        .await
        .is_err();
        assert_meta_slot_drains_with_inference(&state, &release, timed_out).await;
    }

    #[tokio::test]
    async fn a_timed_out_moderation_keeps_its_ai_gate_slot() {
        // The same property for the explicit-outcome
        // helpers (moderation here; embedding and OCR
        // hold their permits in the same shape).
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let mut state = lane_test_state(std::time::Duration::from_secs(30), 4, 2);
        state.ai_gate = Arc::new(tokio::sync::Semaphore::new(1));
        state.moderation = Some(Arc::new(BlockingModeration {
            release: release.clone(),
        }));
        let processed = still_processed(test_png(), ImageFormat::Png);
        let timed_out = tokio::time::timeout_at(
            tokio::time::Instant::now() + std::time::Duration::from_millis(300),
            moderate_meta(&state, &processed, false),
        )
        .await
        .is_err();
        assert_meta_slot_drains_with_inference(&state, &release, timed_out).await;
    }

    #[tokio::test]
    async fn a_timed_out_detection_keeps_its_ai_gate_slot() {
        // The same property for detection, whose permit
        // is shared across a sequence of provider tasks.
        let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let mut state = lane_test_state(std::time::Duration::from_secs(30), 4, 2);
        state.ai_gate = Arc::new(tokio::sync::Semaphore::new(1));
        state.detect = Some(Arc::new(BlockingDetector {
            release: release.clone(),
        }));
        let processed = still_processed(test_png(), ImageFormat::Png);
        let timed_out = tokio::time::timeout_at(
            tokio::time::Instant::now() + std::time::Duration::from_millis(300),
            detect_meta(&state, &processed, false),
        )
        .await
        .is_err();
        assert_meta_slot_drains_with_inference(&state, &release, timed_out).await;
    }

    /// A state for the slow-lane tests (issue #324): the
    /// two lanes sized explicitly, everything else the
    /// defaults the other test states use.
    fn lane_test_state(
        process_timeout: std::time::Duration,
        process_permits: usize,
        slow_permits: usize,
    ) -> AppState {
        AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout,
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(process_permits)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(slow_permits)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        }
    }

    /// A job for the slow-lane tests: the parsed ops, and
    /// a video encoder when the job is the stuck encode.
    fn lane_test_job(
        video_encoder: Option<Arc<crate::video::VideoEncoder>>,
        ops_str: &str,
    ) -> ProcessJob {
        let (ops, _) = crate::url::parse_operations(ops_str).unwrap();
        ProcessJob {
            save_data: false,
            preserve_metadata: false,
            brand_color: None,
            ops,
            preferred: None,
            watermark_sources: Vec::new(),
            placeholders: false,
            mask_provider: None,
            upscale_provider: None,
            inpaint_provider: None,
            segment_provider: None,
            face_provider: None,
            detect_provider: None,
            declared_detect_providers: Vec::new(),
            focal_provider: None,
            text_renderer: None,
            video_encoder,
            frame_extractor: None,
            pdf_renderer: None,
            provenance_signer: None,
            ai_edited: false,
            want_provenance: false,
            dpr: None,
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
        }
    }

    /// The fake external encoder of the #322 tests: drains
    /// stdin, sleeps, only then answers, so one run is a
    /// deterministic long encode inside a single opaque
    /// stage.
    fn fake_long_encoder(dir: &std::path::Path, sleep_secs: u32) -> crate::video::VideoEncoder {
        use std::io::Write as _;
        std::fs::create_dir_all(dir).unwrap();
        let script_path = dir.join("encoder.sh");
        let mut file = std::fs::File::create(&script_path).unwrap();
        file.write_all(
            format!("#!/bin/sh\ncat > /dev/null\nsleep {sleep_secs}\nprintf 'x'\n").as_bytes(),
        )
        .unwrap();
        drop(file);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        crate::video::VideoEncoder::new(
            script_path,
            std::time::Duration::from_secs(30),
            64 * 1024 * 1024,
        )
        .unwrap()
    }

    #[test]
    fn slow_lane_predicate_routes_long_work_only() {
        // Issue #324: the lane classification. Animated
        // WebP encodes and the external-tool work (video,
        // frame, page) are slow; stills, and animated GIF
        // output (a chunked per-frame encode), are not.
        let directives = |ops_str: &str| {
            let (ops, _) = crate::url::parse_operations(ops_str).unwrap();
            crate::url::RequestDirectives::analyze(&ops)
        };
        let slow = |bytes: &[u8], ops_str: &str| {
            let d = directives(ops_str);
            wants_slow_lane(
                bytes,
                &d,
                d.explicit_format.is_some_and(|f| f.is_video()),
                d.frame.is_some(),
                d.page.is_some(),
            )
        };
        let anim_webp = animated_webp_bytes();
        let anim_gif = animated_gif_bytes();
        assert!(slow(&anim_webp, "example.com/i.webp"));
        assert!(slow(&anim_webp, "filters:format(webp)/example.com/i.webp"));
        assert!(slow(&anim_gif, "filters:format(mp4)/example.com/i.gif"));
        assert!(slow(&anim_gif, "frame:poster/example.com/v.mp4"));
        assert!(slow(&test_png(), "page:1/example.com/doc.pdf"));
        assert!(!slow(&test_png(), "example.com/i.png"));
        assert!(!slow(&anim_gif, "example.com/i.gif"));
        assert!(!slow(&anim_webp, "filters:format(gif)/example.com/i.webp"));
        assert!(!slow(&anim_webp, "filters:format(png)/example.com/i.webp"));
    }

    #[tokio::test]
    async fn slow_lane_saturation_leaves_stills_at_full_capacity() {
        // Issue #324, the acceptance shape: the slow lane
        // is saturated by a deadline-length encode (the
        // fake encoder sleeps 9 s inside one opaque stage;
        // the deadline is 300 ms), and a still render
        // arriving during the saturation still renders
        // promptly on the untouched stills lane, where
        // before this issue the encode's process permit
        // was one of the stills pool's own.
        let dir =
            std::env::temp_dir().join(format!("crabcrop-server-slow-lane-{}", std::process::id()));
        let encoder = fake_long_encoder(&dir, 9);
        let state = lane_test_state(std::time::Duration::from_millis(300), 2, 1);
        let stuck_job = lane_test_job(
            Some(Arc::new(encoder)),
            "filters:format(mp4)/example.com/i.gif",
        );
        let follow_job = lane_test_job(None, "example.com/i.png");
        let gif = animated_gif_bytes();
        let png = test_png();
        let (first, (follow_elapsed, follow)) = tokio::join!(
            run_process(
                &state,
                &stuck_job,
                &gif,
                Lane::Slow,
                std::time::Instant::now() + state.process_timeout,
                JobPermits::default()
            ),
            async {
                // Arrive while the encode is stuck, and
                // prove the saturation is the slow lane's
                // alone before rendering the still.
                tokio::time::sleep(std::time::Duration::from_millis(400)).await;
                assert_eq!(state.slow_gate.in_use(), 1);
                assert_eq!(state.process_gate.in_use(), 0);
                let follow_start = Instant::now();
                let result = run_process(
                    &state,
                    &follow_job,
                    &png,
                    Lane::Process,
                    std::time::Instant::now() + state.process_timeout,
                    JobPermits::default(),
                )
                .await;
                (follow_start.elapsed(), result)
            }
        );
        assert!(
            matches!(first, Err(ProcessFailure::TimedOut)),
            "the stuck encode must answer TimedOut at its deadline, got {first:?}"
        );
        let processed = follow.unwrap_or_else(|e| {
            panic!("the still must render under slow-lane saturation, got {e:?}")
        });
        assert_eq!((processed.width, processed.height), (100, 60));
        assert!(
            follow_elapsed < PROCESS_GATE_WAIT,
            "the still must not wait on the slow lane, took {follow_elapsed:?}"
        );
        // The timed-out run's slow permit stays with its
        // detached job while it drains (issue #357), so
        // the lane is not free for the next slow request
        // yet; it returns when the job finishes (the
        // encoder script exits at nine seconds).
        assert_eq!(state.slow_gate.available_permits(), 0);
        assert_eq!(state.process_gate.available_permits(), 2);
        let mut back = false;
        for _ in 0..200 {
            if state.slow_gate.available_permits() == 1 {
                back = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert!(back, "the slow permit must return when the job drains");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn slow_lane_overflow_waits_in_its_own_lane() {
        // Issue #324: with the slow lane's only permit
        // held, a slow run waits in its own lane (it does
        // not spill onto a stills permit), a still renders
        // meanwhile, and once the lane drains the slow run
        // proceeds: the wait is a queue in the right lane,
        // not a detour through the stills pool.
        let state = lane_test_state(std::time::Duration::from_secs(30), 1, 1);
        let job = lane_test_job(None, "example.com/i.png");
        let png = test_png();
        let held = state.slow_gate.clone().acquire_owned().await.unwrap();
        let waiting = run_process(
            &state,
            &job,
            &png,
            Lane::Slow,
            std::time::Instant::now() + state.process_timeout,
            JobPermits::default(),
        );
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(300), &mut waiting)
                .await
                .is_err(),
            "the slow run must wait while the slow lane is saturated"
        );
        assert_eq!(state.process_gate.in_use(), 0);
        let still = run_process(
            &state,
            &job,
            &png,
            Lane::Process,
            std::time::Instant::now() + state.process_timeout,
            JobPermits::default(),
        )
        .await
        .unwrap();
        assert_eq!((still.width, still.height), (100, 60));
        drop(held);
        let processed = waiting.await.unwrap();
        assert_eq!((processed.width, processed.height), (100, 60));
    }

    /// An app whose two lane gates the test keeps handles
    /// to (issue #324), with the metrics token set so the
    /// lane occupancy exposition can be read back.
    async fn spawn_app_with_lane_gates(process_gate: Arc<Gate>, slow_gate: Arc<Gate>) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: Some(TEST_METRICS_TOKEN.to_string()),
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate,
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate,
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn metrics_expose_per_lane_occupancy() {
        // Issue #324: /metrics reports each lane's in-use
        // permits against its capacity, read live from the
        // gates: holding permits by hand moves the gauges.
        let process_gate = Arc::new(Gate::new(4));
        let slow_gate = Arc::new(Gate::new(2));
        let base = spawn_app_with_lane_gates(process_gate.clone(), slow_gate.clone()).await;
        let _held_process = process_gate.clone().acquire_owned().await.unwrap();
        let held_slow = slow_gate.clone().acquire_owned().await.unwrap();
        let _held_slow2 = slow_gate.clone().acquire_owned().await.unwrap();
        let body = metrics_text(&base).await;
        assert!(
            body.contains("crabcrop_lane_in_use{lane=\"process\"} 1"),
            "{body}"
        );
        assert!(
            body.contains("crabcrop_lane_capacity{lane=\"process\"} 4"),
            "{body}"
        );
        assert!(
            body.contains("crabcrop_lane_in_use{lane=\"slow\"} 2"),
            "{body}"
        );
        assert!(
            body.contains("crabcrop_lane_capacity{lane=\"slow\"} 2"),
            "{body}"
        );
        drop(held_slow);
        let body = metrics_text(&base).await;
        assert!(
            body.contains("crabcrop_lane_in_use{lane=\"slow\"} 1"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn slow_lane_saturation_sheds_slow_requests_and_serves_stills() {
        // Issue #324 at the HTTP seam: with the slow lane
        // saturated (its only permit held, as a long
        // encode holds it), a still renders promptly, and
        // an animated WebP request waits out the slow-lane
        // wait and sheds with the lane's own 503 and
        // Retry-After instead of spilling onto the stills
        // lane or queueing silently.
        let process_gate = Arc::new(Gate::new(2));
        let slow_gate = Arc::new(Gate::new(1));
        let base = spawn_app_with_lane_gates(process_gate.clone(), slow_gate.clone()).await;
        let _held_slow = slow_gate.clone().acquire_owned().await.unwrap();
        // The still renders promptly on its own lane.
        let src = spawn_image_server().await;
        let remainder = format!("20x15/{src}");
        let sig = sign(&remainder);
        let started = Instant::now();
        let resp = client()
            .get(format!("{base}/{sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            started.elapsed() < SLOW_LANE_WAIT,
            "a still render must not wait on the slow lane"
        );
        // The animated request sheds honestly.
        let anim_src = spawn_bytes_server(animated_webp_bytes()).await;
        let sig = sign(&anim_src);
        let started = Instant::now();
        let resp = client()
            .get(format!("{base}/{sig}/{anim_src}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers().get("retry-after").unwrap(), "5");
        let body = resp.text().await.unwrap();
        assert!(
            body.contains("long-running processing lane is full"),
            "{body}"
        );
        assert!(
            started.elapsed() >= SLOW_LANE_WAIT,
            "the shed must come only after the lane wait elapses"
        );
    }

    #[tokio::test]
    async fn watermark_decode_waits_for_the_process_gate() {
        // A job carrying an undecodable watermark: the
        // decode rides inside the gated blocking job, so
        // the run waits for a process permit like any other
        // engine work, and the decode failure then surfaces
        // as the engine's decode error (audit R2-16). The
        // watermark decoded inline in the async handler
        // before the fix, outside this seam entirely.
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(1)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let job = ProcessJob {
            save_data: false,
            preserve_metadata: false,
            brand_color: None,
            ops: Vec::new(),
            preferred: None,
            watermark_sources: vec![(
                "http://img/wm.png".to_string(),
                b"this is not an image".to_vec(),
            )],
            placeholders: false,
            mask_provider: None,
            upscale_provider: None,
            inpaint_provider: None,
            segment_provider: None,
            face_provider: None,
            detect_provider: None,
            declared_detect_providers: Vec::new(),
            focal_provider: None,
            text_renderer: None,
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            provenance_signer: None,
            ai_edited: false,
            want_provenance: false,
            dpr: None,
            ssim_target: state.ssim_target,
            avif_speed: state.avif_speed,
            jxl_speed: state.jxl_speed,
        };
        let held = state.process_gate.clone().acquire_owned().await.unwrap();
        let png = test_png();
        let waiting = run_process(
            &state,
            &job,
            &png,
            Lane::Process,
            std::time::Instant::now() + state.process_timeout,
            JobPermits::default(),
        );
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(300), &mut waiting)
                .await
                .is_err(),
            "the watermark decode must wait behind the process gate"
        );
        drop(held);
        let failure = waiting.await.unwrap_err();
        assert!(
            matches!(failure, ProcessFailure::Engine(EngineError::Decode(_))),
            "{failure:?}"
        );
    }

    #[test]
    fn animation_gate_predicate_matches_the_encoder_trigger() {
        // Issue #231: only an animated WebP source rendered
        // to WebP output reaches the slow lossless animation
        // encode, so only that combination takes the gate.
        let anim = animated_webp_bytes();
        let directives = |ops_str: &str| {
            let (ops, _) = crate::url::parse_operations(ops_str).unwrap();
            crate::url::RequestDirectives::analyze(&ops)
        };
        assert!(wants_animation_gate(
            &anim,
            &directives("example.com/i.webp")
        ));
        assert!(wants_animation_gate(
            &anim,
            &directives("filters:format(webp)/example.com/i.webp")
        ));
        // Explicit non-WebP outputs never reach the
        // animated WebP encoder: GIF output uses the fast
        // GIF encoder, still formats take frame 0, and
        // video takes the video gate instead.
        for fmt in ["gif", "png", "jpeg", "mp4"] {
            assert!(
                !wants_animation_gate(
                    &anim,
                    &directives(&format!("filters:format({fmt})/example.com/i.webp"))
                ),
                "format({fmt}) must not take the animation gate"
            );
        }
        // Still sources are never gated, whatever their
        // container: a still WebP carries no ANIM chunk, a
        // GIF source never reaches the WebP animation path,
        // and a PNG is not WebP at all.
        assert!(!wants_animation_gate(
            &still_webp_bytes(),
            &directives("example.com/i.webp")
        ));
        assert!(!wants_animation_gate(
            &animated_gif_bytes(),
            &directives("example.com/i.gif")
        ));
        assert!(!wants_animation_gate(
            &test_png(),
            &directives("example.com/i.png")
        ));
    }

    #[tokio::test]
    async fn animation_gate_sheds_a_second_animated_render() {
        // Issue #231: with the single animation permit held
        // (one long encode already running), an animated
        // WebP request waits out the gate wait and sheds
        // with the animation-queue 503 instead of stacking
        // a second encode behind the first.
        let gate = Arc::new(tokio::sync::Semaphore::new(1));
        let base = spawn_app_with_anim_gate(gate.clone()).await;
        let src = spawn_bytes_server(animated_webp_bytes()).await;
        let _held = gate.acquire_owned().await.unwrap();
        let sig = sign(&src);
        let started = Instant::now();
        let resp = client()
            .get(format!("{base}/{sig}/{src}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers().get("retry-after").unwrap(), "5");
        let body = resp.text().await.unwrap();
        assert!(body.contains("animation encoding queue is full"), "{body}");
        assert!(
            started.elapsed() >= ANIM_GATE_WAIT,
            "the shed must come only after the gate wait elapses"
        );
    }

    #[tokio::test]
    async fn animation_gate_leaves_still_renders_unaffected() {
        // The #231 regression test: while the animation
        // gate is saturated (one long encode running), an
        // unrelated still render still succeeds promptly
        // instead of shedding behind the animation work.
        let gate = Arc::new(tokio::sync::Semaphore::new(1));
        let base = spawn_app_with_anim_gate(gate.clone()).await;
        let _held = gate.acquire_owned().await.unwrap();
        let src = spawn_image_server().await;
        let remainder = format!("20x15/{src}");
        let sig = sign(&remainder);
        let started = Instant::now();
        let resp = client()
            .get(format!("{base}/{sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            started.elapsed() < ANIM_GATE_WAIT,
            "a still render must not wait on the animation gate"
        );
        let bytes = resp.bytes().await.unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!((img.width(), img.height()), (20, 15));
    }

    #[tokio::test]
    async fn animated_webp_renders_through_the_gate_when_free() {
        // The gated path still delivers the feature: with
        // the animation permit free, a bare animated WebP
        // request renders the full animation.
        let gate = Arc::new(tokio::sync::Semaphore::new(1));
        let base = spawn_app_with_anim_gate(gate).await;
        let src = spawn_bytes_server(animated_webp_bytes()).await;
        let sig = sign(&src);
        let resp = client()
            .get(format!("{base}/{sig}/{src}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("content-type").unwrap(), "image/webp");
        let bytes = resp.bytes().await.unwrap();
        use image::AnimationDecoder as _;
        let decoder = image::codecs::webp::WebPDecoder::new(std::io::Cursor::new(&bytes)).unwrap();
        let frames = decoder.into_frames().collect_frames().unwrap();
        assert_eq!(frames.len(), 2);
    }

    #[tokio::test]
    async fn fallback_serves_through_the_operations_on_a_dead_source() {
        let base = spawn_app_with_fallback().await;
        let (dead, hits) = spawn_counting_server(404, "image/png", test_png()).await;
        let remainder = format!("20x15/{dead}");
        let sig = sign(&remainder);
        let url = format!("{base}/{sig}/{remainder}");
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("x-crabcrop-fallback").unwrap(), "1");
        // The fallback is never pinned downstream (audit PL-2).
        assert_eq!(resp.headers().get("cache-control").unwrap(), "no-store");
        let bytes = resp.bytes().await.unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        // The operations ran on the fallback (red 40x30), not on
        // anything the origin said.
        assert_eq!((img.width(), img.height()), (20, 15));
        assert_eq!(img.to_rgb8().get_pixel(10, 7), &image::Rgb([255, 0, 0]));
        // Never cached: the origin is retried and the header
        // appears again, instead of a cached placeholder being
        // served as if it were the real image.
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.headers().get("x-crabcrop-fallback").unwrap(), "1");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn fallback_covers_undecodable_sources_but_not_bad_signatures() {
        let base = spawn_app_with_fallback().await;
        let (garbage, _) = spawn_counting_server(200, "image/png", b"not an image".to_vec()).await;
        let remainder = format!("10x10/{garbage}");
        let sig = sign(&remainder);
        let resp = client()
            .get(format!("{base}/{sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert!(resp.headers().get("x-crabcrop-fallback").is_some());
        // A bad signature stays an honest 403, no fallback.
        let resp = client()
            .get(format!("{base}/deadbeef/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 403);
        assert!(resp.headers().get("x-crabcrop-fallback").is_none());
    }

    #[tokio::test]
    async fn dpr_header_scales_and_caches_separately() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("20x20/{src}");
        let sig = sign(&remainder);
        let url = format!("{base}/{sig}/{remainder}");
        let dims = |bytes: &[u8]| {
            let img = image::load_from_memory(bytes).unwrap();
            (img.width(), img.height())
        };
        // With the hint, the 20x20 target becomes 40x40.
        let hinted = client()
            .get(&url)
            .header("DPR", "2")
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(dims(&hinted), (40, 40));
        // Without it, the same URL serves 20x20: the two results
        // live in separate cache entries, neither pollutes the
        // other.
        let plain = client()
            .get(&url)
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(dims(&plain), (20, 20));
        let hinted_again = client()
            .get(&url)
            .header("DPR", "2")
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(dims(&hinted_again), (40, 40));
        // A nonsense hint is ignored, never an error.
        let bogus = client()
            .get(&url)
            .header("DPR", "banana")
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(dims(&bogus), (20, 20));
    }

    #[tokio::test]
    async fn text_without_a_font_is_501() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:text(HI,20,ffffff,bottom-right)/{src}");
        let sig = sign(&remainder);
        let resp = client()
            .get(format!("{base}/{sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 501);
        let body = resp.text().await.unwrap();
        assert!(body.contains("text is not configured"), "{body}");
    }

    #[tokio::test]
    async fn serves_jxl_when_asked() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:format(jxl)/{src}");
        let sig = sign(&remainder);
        let resp = client()
            .get(format!("{base}/{sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/jxl"
        );
        let body = resp.bytes().await.unwrap();
        assert!(body.starts_with(&[0xFF, 0x0A]));
    }

    #[tokio::test]
    async fn serves_heic_source_resized_to_png() {
        // The orientation fixture (tests/fixtures/orient6.heic):
        // stored 64x48 with EXIF Orientation=6, so the served
        // image is the upright 48x64.
        let src =
            spawn_bytes_server(include_bytes!("../tests/fixtures/orient6.heic").to_vec()).await;
        let base = spawn_app(false).await;
        let remainder = format!("24x32/filters:format(png)/{src}");
        let sig = sign(&remainder);
        let resp = client()
            .get(format!("{base}/{sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/png"
        );
        let body = resp.bytes().await.unwrap();
        let out = image::load_from_memory(&body).unwrap();
        assert_eq!((out.width(), out.height()), (24, 32));
    }

    #[tokio::test]
    async fn auto_quality_sets_header() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:quality(auto):format(jpeg)/{src}");
        let sig = sign(&remainder);
        let resp = client()
            .get(format!("{base}/{sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // The source image is solid black, so analysis picks the floor.
        assert_eq!(resp.headers().get("x-crabcrop-quality").unwrap(), "50");
    }

    #[tokio::test]
    async fn rejects_bad_signature() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let resp = client()
            .get(format!("{base}/deadbeef/300x50/{src}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn rejects_unsafe_when_disabled() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let resp = client()
            .get(format!("{base}/unsafe/300x50/{src}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn allows_unsafe_when_enabled() {
        let src = spawn_image_server().await;
        let base = spawn_app(true).await;
        let resp = client()
            .get(format!("{base}/unsafe/300x50/{src}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn purges_never_honor_unsafe() {
        // Q3: unsafe mode opens image reads, never the
        // destructive endpoints. An unsafe purge is a 403
        // and evicts nothing; the signed purge still works
        // on the same server.
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(true).await;
        let remainder = format!("300x50/{src}");
        let resp = client()
            .get(format!("{base}/unsafe/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let resp = client()
            .post(format!("{base}/purge/unsafe/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let resp = client()
            .post(format!("{base}/purge-group/unsafe/anygroup"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let resp = client()
            .get(format!("{base}/unsafe/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "the rejected purge evicted nothing"
        );
        let purge_sig = crate::url::purge_signature(SECRET, &remainder);
        let resp = client()
            .post(format!("{base}/purge/{purge_sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.text().await.unwrap().contains("purged 1"));
    }

    #[tokio::test]
    async fn rejects_malformed_path() {
        let base = spawn_app(false).await;
        // Valid signature over an empty remainder, but no operations or URL.
        let sig = sign("");
        let resp = client().get(format!("{base}/{sig}/")).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// An app with uploads configured (issue #168): a
    /// local destination at `dir`, which is also the
    /// loader's file root, exactly as boot wires it. The
    /// size cap is a parameter so the oversize test can
    /// use a small one.
    async fn spawn_app_with_upload(
        dir: &std::path::Path,
        max_bytes: u64,
        allow_overwrite: bool,
    ) -> String {
        spawn_app_with_upload_gate(dir, max_bytes, allow_overwrite, Arc::new(Gate::new(4))).await
    }

    async fn spawn_app_with_upload_gate(
        dir: &std::path::Path,
        max_bytes: u64,
        allow_overwrite: bool,
        process_gate: Arc<Gate>,
    ) -> String {
        std::fs::create_dir_all(dir).unwrap();
        let dir = std::fs::canonicalize(dir).unwrap();
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults()
                .unwrap()
                .with_sources(crate::loader::SourceConfig {
                    file_root: Some(dir.clone()),
                    ..Default::default()
                }),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: Some(TEST_METRICS_TOKEN.to_string()),
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: Some(Arc::new(crate::upload::UploadStore::local(
                dir,
                max_bytes,
                allow_overwrite,
            ))),
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate,
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn upload_dir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("crabcrop-srv-upload-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[tokio::test]
    async fn upload_stores_and_the_object_is_served_by_a_transform_url() {
        let dir = upload_dir("serve");
        let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, false).await;
        let name = "heroes/cat.png";
        let sig = crate::url::upload_signature(SECRET, name);
        let resp = client()
            .put(format!("{base}/upload/{sig}/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        assert_eq!(
            resp.headers().get("x-crabcrop-source").unwrap(),
            "file:///heroes/cat.png"
        );
        let source_ref = resp.text().await.unwrap().trim().to_string();
        assert_eq!(source_ref, "file:///heroes/cat.png");
        // The bytes landed verbatim under the name.
        assert_eq!(
            std::fs::read(dir.join("heroes/cat.png")).unwrap(),
            test_png()
        );
        // And the returned reference serves through a
        // normal signed transform URL: 100x60 resized to
        // exactly 50x30.
        let remainder = format!("50x30/{source_ref}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.bytes().await.unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!((img.width(), img.height()), (50, 30));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn upload_rejects_unsigned_and_wrong_kind_signatures() {
        let dir = upload_dir("auth");
        let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, false).await;
        let name = "cat.png";
        // No signature segment at all: the path does not
        // parse as a signed upload.
        let resp = client()
            .put(format!("{base}/upload/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        // `unsafe` never authorizes a write.
        let resp = client()
            .put(format!("{base}/upload/unsafe/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        // A transform (image) signature over the name.
        let resp = client()
            .put(format!("{base}/upload/{}/{name}", sign(name)))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        // A purge signature over the name.
        let purge_sig = crate::url::purge_signature(SECRET, name);
        let resp = client()
            .put(format!("{base}/upload/{purge_sig}/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        // An upload signature for a different name.
        let other_sig = crate::url::upload_signature(SECRET, "dog.png");
        let resp = client()
            .put(format!("{base}/upload/{other_sig}/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        // Nothing was stored by any of the rejections.
        assert!(!dir.join(name).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn upload_rejects_bad_names_oversize_and_non_images() {
        let dir = upload_dir("bounds");
        // A 64-byte cap: the test PNG is larger, a tiny
        // body is not.
        let base = spawn_app_with_upload(&dir, 64, false).await;
        // Oversize: a real PNG past the configured cap.
        let name = "big.png";
        let sig = crate::url::upload_signature(SECRET, name);
        let resp = client()
            .put(format!("{base}/upload/{sig}/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert!(!dir.join(name).exists());
        // Non-image bytes under the cap.
        let name = "note.png";
        let sig = crate::url::upload_signature(SECRET, name);
        let resp = client()
            .put(format!("{base}/upload/{sig}/{name}"))
            .body(b"this is not an image".to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert!(!dir.join(name).exists());
        // An empty body is not an image either.
        let resp = client()
            .put(format!("{base}/upload/{sig}/{name}"))
            .body(Vec::new())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        // Bad names: traversal and a leading-dot segment.
        // The signature is over the name as presented, so
        // these reach the name check with a valid tag.
        for bad in ["../secret.png", ".hidden.png", "a//b.png"] {
            let sig = crate::url::upload_signature(SECRET, bad);
            let resp = client()
                .put(format!("{base}/upload/{sig}/{bad}"))
                .body(b"x".to_vec())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{bad}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn upload_overwrite_is_refused_by_default_and_allowed_by_flag() {
        let dir = upload_dir("overwrite");
        let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, false).await;
        let name = "cat.png";
        let sig = crate::url::upload_signature(SECRET, name);
        let url = format!("{base}/upload/{sig}/{name}");
        let resp = client().put(&url).body(test_png()).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let resp = client().put(&url).body(test_png()).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        // With the flag, the same second PUT replaces and
        // answers 200.
        let dir2 = upload_dir("overwrite-ok");
        let base = spawn_app_with_upload(&dir2, 10 * 1024 * 1024, true).await;
        let url = format!("{base}/upload/{sig}/{name}");
        let resp = client().put(&url).body(test_png()).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let other = {
            let img = DynamicImage::new_rgb8(10, 10);
            let mut buf = std::io::Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
            buf.into_inner()
        };
        let resp = client().put(&url).body(other.clone()).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(std::fs::read(dir2.join(name)).unwrap(), other);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[tokio::test]
    async fn upload_overwrite_invalidates_cached_derivatives() {
        // Audit R2-07: overwriting a source must not leave
        // its cached derivatives serving the old image.
        let dir = upload_dir("overwrite-purge");
        let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, true).await;
        let name = "cat.png";
        let sig = crate::url::upload_signature(SECRET, name);
        let url = format!("{base}/upload/{sig}/{name}");
        let resp = client().put(&url).body(test_png()).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        // Render a derivative and confirm it is cached.
        let remainder = "50x30/file:///cat.png";
        let get = format!("{base}/{}/{remainder}", sign(remainder));
        let first = client()
            .get(&get)
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let resp = client().get(&get).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        // Overwrite the source with a different image.
        let other = {
            let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                200,
                120,
                image::Rgb([255, 255, 255]),
            ));
            let mut buf = std::io::Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
            buf.into_inner()
        };
        let resp = client().put(&url).body(other).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // The same transform URL re-renders from the new
        // original instead of serving the stale entry.
        let resp = client().get(&get).send().await.unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        let rendered = resp.bytes().await.unwrap();
        assert_ne!(
            rendered, first,
            "the derivative is the new image, not the stale one"
        );
        let img = image::load_from_memory(&rendered).unwrap();
        assert_eq!((img.width(), img.height()), (50, 30));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Source server that counts its hits and answers
    /// slowly, so concurrent requests overlap in flight.
    async fn spawn_slow_counting_server(
        body: Vec<u8>,
        delay_ms: u64,
    ) -> (String, Arc<AtomicUsize>) {
        let hits = Arc::new(AtomicUsize::new(0));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let counted = hits.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                counted.fetch_add(1, Ordering::SeqCst);
                let body = body.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 2048];
                    let _ = socket.read(&mut buf).await;
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = socket.write_all(head.as_bytes()).await;
                    let _ = socket.write_all(&body).await;
                });
            }
        });
        (format!("http://{addr}/src.png"), hits)
    }

    /// Source server that counts its hits and answers
    /// slowly with a 500, so a burst's leaders fail one
    /// after another while their waiters are still queued.
    async fn spawn_slow_failing_server(delay_ms: u64) -> (String, Arc<AtomicUsize>) {
        let hits = Arc::new(AtomicUsize::new(0));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let counted = hits.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                counted.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let mut buf = [0u8; 2048];
                    let _ = socket.read(&mut buf).await;
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    let _ = socket
                        .write_all(
                            b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        )
                        .await;
                });
            }
        });
        (format!("http://{addr}/src.png"), hits)
    }

    #[tokio::test]
    async fn concurrent_misses_share_one_origin_fetch() {
        // Audit R2-09: a burst of identical cache misses
        // collapses into one fetch and one encode; the
        // rest wait for the leader, then serve its entry.
        let (src, hits) = spawn_slow_counting_server(test_png(), 500).await;
        let base = spawn_app(false).await;
        let remainder = format!("50x30/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let mut tasks = Vec::new();
        for _ in 0..8 {
            let url = url.clone();
            tasks.push(tokio::spawn(async move {
                let resp = client().get(&url).send().await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                resp.bytes().await.unwrap()
            }));
        }
        let mut bodies = Vec::new();
        for task in tasks {
            bodies.push(task.await.unwrap());
        }
        assert!(
            bodies.windows(2).all(|w| w[0] == w[1]),
            "every waiter received the leader's bytes"
        );
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "one origin fetch for the whole burst"
        );
    }

    #[test]
    fn ai_gate_counts_a_smart_crop_only_when_a_provider_participates() {
        // Issue #379: the predicate matrix. A smart crop
        // alone gates only with a face or focal provider;
        // real inference directives gate regardless.
        let smart = crate::url::RequestDirectives::analyze(&[crate::url::Operation::Crop {
            size: crate::url::ImageSize {
                width: crate::url::Dimension::Pixels(50),
                height: crate::url::Dimension::Pixels(50),
                flip_horizontal: false,
                flip_vertical: false,
            },
            align: crate::url::Align::Smart,
        }]);
        assert!(smart.smart_crop);
        assert!(!ai_gate_applies(&smart, false, false));
        assert!(ai_gate_applies(&smart, true, false));
        assert!(ai_gate_applies(&smart, false, true));
        let removebg =
            crate::url::RequestDirectives::analyze(&[crate::url::Operation::Filters(vec![
                crate::url::Filter::RemoveBg,
            ])]);
        assert!(ai_gate_applies(&removebg, false, false));
        let plain = crate::url::RequestDirectives::analyze(&[crate::url::Operation::Size(
            crate::url::ImageSize {
                width: crate::url::Dimension::Pixels(50),
                height: crate::url::Dimension::Pixels(50),
                flip_horizontal: false,
                flip_vertical: false,
            },
        )]);
        assert!(!ai_gate_applies(&plain, true, true));
    }

    #[tokio::test]
    async fn meta_mode_disabled_refuses_meta_and_still_serves_the_render() {
        // Issue #390: with meta mode switched off, a meta
        // request answers 403 meta_disabled, and the same
        // signed URL without the meta Accept still serves
        // its render.
        let src = spawn_bytes_server(test_png()).await;
        let base = spawn_app_with_meta_enabled(false).await;
        let remainder = format!("100x0/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let body = resp.text().await.unwrap();
        assert!(body.contains("meta_disabled"), "{body}");
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn a_provider_free_smart_crop_never_touches_the_ai_gate() {
        // Issue #379 at the HTTP level: the app has no
        // face or focal provider and an AI gate with no
        // permits at all, so a gated request could only
        // shed. The smart crop runs on built-in saliency
        // and answers at once.
        let src = spawn_bytes_server(test_png()).await;
        let base = spawn_app_with_ai_gate(Arc::new(tokio::sync::Semaphore::new(0))).await;
        let remainder = format!("crop:20x20:smart/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let started = std::time::Instant::now();
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "the crop answered without waiting on the gate, took {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn flight_waiters_end_with_the_origins_answer_not_a_busy_shed() {
        // Issue #375: when leaders keep failing, a waiter
        // that spent its wait cap once shed with a 503
        // server_busy, masking the origin's real answer.
        // The waiter now runs the miss path itself, so a
        // burst against a slow failing origin ends the way
        // a solo request does, for every request.
        let (src, _hits) = spawn_slow_failing_server(300).await;
        let base = spawn_app(false).await;
        let remainder = format!("50x30/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let solo = client().get(&url).send().await.unwrap();
        let solo_status = solo.status();
        assert_ne!(solo_status, StatusCode::SERVICE_UNAVAILABLE);
        let mut tasks = Vec::new();
        for _ in 0..6 {
            let url = url.clone();
            tasks.push(tokio::spawn(async move {
                client().get(&url).send().await.unwrap().status()
            }));
        }
        for task in tasks {
            assert_eq!(task.await.unwrap(), solo_status);
        }
    }

    /// Fire `n` concurrent meta requests for one URL and
    /// collect their (status, body) pairs.
    async fn meta_burst(url: &str, accept: &str, n: usize) -> Vec<(StatusCode, bytes::Bytes)> {
        let mut tasks = Vec::new();
        for _ in 0..n {
            let url = url.to_string();
            let accept = accept.to_string();
            tasks.push(tokio::spawn(async move {
                let resp = client()
                    .get(&url)
                    .header("Accept", accept.as_str())
                    .send()
                    .await
                    .unwrap();
                let status = resp.status();
                (status, resp.bytes().await.unwrap())
            }));
        }
        let mut out = Vec::new();
        for task in tasks {
            out.push(task.await.unwrap());
        }
        out
    }

    #[tokio::test]
    async fn concurrent_meta_requests_share_one_origin_fetch() {
        // Issue #295: a burst of identical meta requests
        // collapses into one load and one render, like
        // image misses do (audit R2-09); the followers
        // receive the leader's document from its flight.
        let (src, hits) = spawn_slow_counting_server(test_png(), 500).await;
        let base = spawn_app(false).await;
        let remainder = format!("50x30/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let first = meta_burst(&url, "application/json", 8).await;
        assert!(first.iter().all(|(s, _)| *s == StatusCode::OK));
        assert!(
            first.windows(2).all(|w| w[0].1 == w[1].1),
            "every follower received the leader's document"
        );
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "one origin fetch for the whole meta burst"
        );
        // Nothing is shared across time: a second burst,
        // after the flight has ended, leads a fresh flight
        // and fetches exactly once more.
        let second = meta_burst(&url, "application/json", 8).await;
        assert!(second.iter().all(|(s, _)| *s == StatusCode::OK));
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "the second burst fetched again, once"
        );
        assert_eq!(first[0].1, second[0].1, "the same document both times");
    }

    #[tokio::test]
    async fn concurrent_meta_requests_with_different_opt_ins_do_not_share() {
        // Issue #295: the flight key carries the Accept
        // opt-ins, so a burst asking for placeholder
        // fields never receives a document computed
        // without them, nor the other way around: the two
        // bursts lead separate flights, one fetch each.
        let (src, hits) = spawn_slow_counting_server(test_png(), 500).await;
        let base = spawn_app(false).await;
        let remainder = format!("50x30/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let plain = tokio::spawn({
            let url = url.clone();
            async move { meta_burst(&url, "application/json", 4).await }
        });
        let opted =
            tokio::spawn(
                async move { meta_burst(&url, "application/json; placeholders=1", 4).await },
            );
        let plain = plain.await.unwrap();
        let opted = opted.await.unwrap();
        assert!(plain.iter().all(|(s, _)| *s == StatusCode::OK));
        assert!(opted.iter().all(|(s, _)| *s == StatusCode::OK));
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "one fetch per opt-in set, not one per request and not one overall"
        );
        assert!(
            plain
                .iter()
                .all(|(_, b)| !String::from_utf8_lossy(b).contains("\"placeholder\"")),
            "the plain burst's documents carry no placeholder fields"
        );
        assert!(
            opted
                .iter()
                .all(|(_, b)| String::from_utf8_lossy(b).contains("\"placeholder\"")),
            "the opt-in burst's documents all carry placeholder fields"
        );
    }

    #[tokio::test]
    async fn concurrent_meta_requests_keep_solo_error_semantics() {
        // Issue #295: errors are never shared, and no
        // waiter hangs or panics: when the origin fails,
        // every request in the burst ends with the failure
        // a solo request gets, from its own attempt at the
        // miss path.
        let (src, _hits) = spawn_counting_server(404, "image/png", Vec::new()).await;
        let base = spawn_app(false).await;
        let remainder = format!("50x30/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let solo = meta_burst(&url, "application/json", 1).await;
        // The origin answers 404, which is a not-found
        // answer since issue #325, not a gateway failure.
        assert_eq!(solo[0].0, StatusCode::NOT_FOUND);
        let burst = meta_burst(&url, "application/json", 6).await;
        assert!(
            burst
                .iter()
                .all(|(s, b)| *s == solo[0].0 && *b == solo[0].1),
            "every waiter failed the way the solo request did"
        );
    }

    #[tokio::test]
    async fn group_provenance_accumulates_across_spellings() {
        // Audit R2-15: an entry first stored through a
        // group-free spelling joins the group when a
        // grouped spelling of the same URL is served from
        // it, so the group purge still finds it.
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app_with_presets(preset_test_map(), false).await;
        // The plain spelling stores the entry group-free.
        let plain = format!("30x20/{src}");
        let resp = client()
            .get(format!("{base}/{}/{plain}", sign(&plain)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // The preset spelling is the same canonical entry:
        // a hit (no second fetch) that unions the preset's
        // group into the stored entry.
        let preset = format!("preset:small/{src}");
        let resp = client()
            .get(format!("{base}/{}/{preset}", sign(&preset)))
            .send()
            .await
            .unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // The group purge removes the entry even though it
        // was stored before the group was ever seen.
        let sig = crate::url::purge_group_signature(SECRET, "small");
        let resp = client()
            .post(format!("{base}/purge-group/{sig}/small"))
            .send()
            .await
            .unwrap();
        assert!(resp.text().await.unwrap().contains("purged 1"));
        let resp = client()
            .get(format!("{base}/{}/{plain}", sign(&plain)))
            .send()
            .await
            .unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    }

    #[tokio::test]
    async fn group_membership_learned_on_a_hit_survives_a_restart() {
        // Issue #350: membership learned on a memory hit
        // propagates to the persistent tier, so a group
        // purge after a restart still finds the entry.
        // Before the fix the union lived only in the first
        // instance's memory and the purge below removed 0.
        let dir = std::env::temp_dir().join(format!("crabcrop-srv-gprop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app_full(preset_test_map(), Some(&dir), 16 * 1024 * 1024).await;
        // The plain spelling stores the entry group-free
        // in memory and on disk.
        let plain = format!("30x20/{src}");
        let resp = client()
            .get(format!("{base}/{}/{plain}", sign(&plain)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // The preset spelling is the same canonical entry:
        // a memory hit that unions the preset's group in.
        let preset = format!("preset:small/{src}");
        let resp = client()
            .get(format!("{base}/{}/{preset}", sign(&preset)))
            .send()
            .await
            .unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // Restart: a fresh instance over the same
        // directory, its memory cold. The group purge can
        // only see the persistent tier, and it finds the
        // entry there by the membership the first
        // instance's memory learned.
        let base2 = spawn_app_full(preset_test_map(), Some(&dir), 16 * 1024 * 1024).await;
        let sig = crate::url::purge_group_signature(SECRET, "small");
        let resp = client()
            .post(format!("{base2}/purge-group/{sig}/small"))
            .send()
            .await
            .unwrap();
        assert!(resp.text().await.unwrap().contains("purged 1"));
        let resp = client()
            .get(format!("{base2}/{}/{plain}", sign(&plain)))
            .send()
            .await
            .unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn purge_during_a_render_keeps_the_entry_out_of_every_tier() {
        // Issue #350: a purge that lands while a render is
        // in flight is recorded before the render's writes,
        // so the finished render cannot restore the entry
        // to any tier.
        let fake = crate::s3store::test_support::start_fake().await;
        let dir = std::env::temp_dir().join(format!("crabcrop-srv-pdr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, hits) = spawn_slow_counting_server(test_png(), 1200).await;
        let base = spawn_app_with_shared(Some(&dir), &fake.endpoint).await;
        let remainder = format!("25x25/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        // Start the render; it blocks in the origin load.
        let pending = tokio::spawn(async move { client().get(&url).send().await.unwrap() });
        // Wait until the fetch is actually in flight.
        for _ in 0..50 {
            if hits.load(Ordering::SeqCst) == 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert_eq!(hits.load(Ordering::SeqCst), 1, "the render never started");
        // Purge the key mid-render. Nothing is stored yet,
        // so the count is 0; the purge still counts.
        let purge_sig = crate::url::purge_signature(SECRET, &remainder);
        let resp = client()
            .post(format!("{base}/purge/{purge_sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.text().await.unwrap().contains("purged 0"));
        // The render completes and answers its client with
        // the render itself; only its writes are refused.
        let resp = pending.await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Give the background shared-tier write every
        // chance to land: it must never appear.
        tokio::time::sleep(std::time::Duration::from_millis(700)).await;
        assert!(
            fake.objects.lock().unwrap().is_empty(),
            "the shared tier holds the purged render"
        );
        let on_disk = std::fs::read_dir(&dir)
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|e| e.path().extension().is_some_and(|x| x == "bin"))
                    .count()
            })
            .unwrap_or(0);
        assert_eq!(on_disk, 0, "the disk tier holds the purged render");
        // And no tier serves it: the next request misses
        // and renders again.
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn group_purge_during_a_render_keeps_the_entry_out() {
        // Issue #350, the group form: a group purge that
        // lands while a grouped render is in flight keeps
        // the finished render out of the tiers.
        let dir = std::env::temp_dir().join(format!("crabcrop-srv-gpdr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (src, hits) = spawn_slow_counting_server(test_png(), 1200).await;
        let base = spawn_app_full(preset_test_map(), Some(&dir), 16 * 1024 * 1024).await;
        let preset = format!("preset:small/{src}");
        let url = format!("{base}/{}/{preset}", sign(&preset));
        let pending = tokio::spawn(async move { client().get(&url).send().await.unwrap() });
        for _ in 0..50 {
            if hits.load(Ordering::SeqCst) == 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert_eq!(hits.load(Ordering::SeqCst), 1, "the render never started");
        let sig = crate::url::purge_group_signature(SECRET, "small");
        let resp = client()
            .post(format!("{base}/purge-group/{sig}/small"))
            .send()
            .await
            .unwrap();
        assert!(resp.text().await.unwrap().contains("purged 0"));
        let resp = pending.await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // The local writes land before the response, so
        // the outcome is already settled: the next request
        // misses and renders again.
        let resp = client()
            .get(format!("{base}/{}/{preset}", sign(&preset)))
            .send()
            .await
            .unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An app over a disk tier under a chosen
    /// configuration fingerprint (the epoch tests).
    async fn spawn_app_with_store_fingerprint(dir: &std::path::Path, fingerprint: &str) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: Some(Arc::new(
                crate::storage::DiskStore::open(
                    dir,
                    16 * 1024 * 1024,
                    crate::url::entry_key(SECRET),
                )
                .unwrap(),
            )),
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            source_roots: std::collections::HashMap::new(),
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: fingerprint.to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn a_config_change_starts_a_fresh_cache_epoch() {
        // Audit R2-08: servers sharing one disk tier under
        // different configuration fingerprints must not
        // serve each other's renders; the same fingerprint
        // across a restart still hits the disk tier.
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let dir = upload_dir("epoch");
        let remainder = format!("50x30/{src}");
        let path = format!("{}/{remainder}", sign(&remainder));
        let base_a = spawn_app_with_store_fingerprint(&dir, "epoch-a").await;
        let resp = client()
            .get(format!("{base_a}/{path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        // A different fingerprint over the same directory:
        // a miss, refetched, not the stored render.
        let base_b = spawn_app_with_store_fingerprint(&dir, "epoch-b").await;
        let resp = client()
            .get(format!("{base_b}/{path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        // The same fingerprint across a restart: a disk
        // hit, no third fetch. The epoch is a function of
        // the configuration, not of the process.
        let base_a2 = spawn_app_with_store_fingerprint(&dir, "epoch-a").await;
        let resp = client()
            .get(format!("{base_a2}/{path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(
            header(&resp, "cache-status"),
            "crabcrop; hit; detail=\"disk\""
        );
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn upload_conflict_is_answered_before_validation() {
        // The existence check runs before the validation
        // decode (issue #187): a duplicate upload whose
        // body would fail validation is still answered
        // 409, not 415, and the stored object is
        // untouched.
        let dir = upload_dir("exists-first");
        let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, false).await;
        let name = "cat.png";
        let sig = crate::url::upload_signature(SECRET, name);
        let url = format!("{base}/upload/{sig}/{name}");
        let resp = client().put(&url).body(test_png()).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let resp = client()
            .put(&url)
            .body(b"this is not an image".to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert_eq!(std::fs::read(dir.join(name)).unwrap(), test_png());
        // With overwrite allowed, the same invalid body
        // still reaches validation and is refused there,
        // and the stored object is still untouched.
        let dir2 = upload_dir("exists-first-ok");
        let base = spawn_app_with_upload(&dir2, 10 * 1024 * 1024, true).await;
        let url = format!("{base}/upload/{sig}/{name}");
        let resp = client().put(&url).body(test_png()).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let resp = client()
            .put(&url)
            .body(b"this is not an image".to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert_eq!(std::fs::read(dir2.join(name)).unwrap(), test_png());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[tokio::test]
    async fn upload_expiring_tokens_expire_and_stay_bound() {
        // The expiring form (issue #192, audit R2-06):
        // the expiry rides inside the signed string, so a
        // delegated token dies on schedule and cannot be
        // reshaped.
        let dir = upload_dir("expiring");
        let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, false).await;
        let name = "cat.png";
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        // A token inside its window uploads normally.
        let future = now + 3600;
        let sig = crate::url::upload_signature_with_expiry(SECRET, name, future);
        let resp = client()
            .put(format!("{base}/upload/{sig}/exp:{future}/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        // A token past its second is refused as expired,
        // before anything is stored.
        let past = now - 3600;
        let sig = crate::url::upload_signature_with_expiry(SECRET, "dog.png", past);
        let resp = client()
            .put(format!("{base}/upload/{sig}/exp:{past}/dog.png"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(resp.text().await.unwrap(), "signature expired");
        assert!(!dir.join("dog.png").exists());
        // The token is bound to its name: replaying it
        // against another name, expiry and all, fails the
        // signature.
        let sig = crate::url::upload_signature_with_expiry(SECRET, name, future);
        let resp = client()
            .put(format!("{base}/upload/{sig}/exp:{future}/other.png"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        // Extending the expiry by editing the path fails
        // the same way: the presented expiry is signed.
        let later = future + 3600;
        let resp = client()
            .put(format!("{base}/upload/{sig}/exp:{later}/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        // And a plain (non-expiring) token cannot be given
        // an expiry by presentation, nor an expiring one
        // stripped of it.
        let plain = crate::url::upload_signature(SECRET, "plain.png");
        let resp = client()
            .put(format!("{base}/upload/{plain}/exp:{future}/plain.png"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let resp = client()
            .put(format!("{base}/upload/{sig}/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn upload_stores_the_sanitized_svg_not_the_callers_bytes() {
        // What lands in the destination for an SVG is the
        // sanitized document (issue #192, audit R2-22):
        // the script is stripped at store time, not only
        // when crabcrop serves it back.
        let dir = upload_dir("svg-store");
        let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, false).await;
        let name = "icon.svg";
        let sig = crate::url::upload_signature(SECRET, name);
        let hostile = br#"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><script>alert(1)</script><rect width="4" height="4" fill="red"/></svg>"#;
        let resp = client()
            .put(format!("{base}/upload/{sig}/{name}"))
            .body(hostile.to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let stored = std::fs::read(dir.join(name)).unwrap();
        assert_ne!(stored, hostile.to_vec());
        let stored = String::from_utf8(stored).unwrap();
        assert!(!stored.contains("script"), "{stored}");
        assert!(stored.contains("<rect"), "{stored}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn upload_rejects_names_that_misstate_the_media() {
        // The name's extension must match the validated
        // media kind (issue #192, audit R2-22): a PNG is
        // refused as .jpg and as .html alike.
        let dir = upload_dir("media-bind");
        let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, false).await;
        for (name, want) in [
            ("photo.jpg", StatusCode::UNSUPPORTED_MEDIA_TYPE),
            ("promo.html", StatusCode::UNSUPPORTED_MEDIA_TYPE),
            ("photo.png", StatusCode::CREATED),
            ("photo.PNG", StatusCode::CREATED),
            ("photo", StatusCode::CREATED),
        ] {
            let sig = crate::url::upload_signature(SECRET, name);
            let resp = client()
                .put(format!("{base}/upload/{sig}/{name}"))
                .body(test_png())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), want, "{name}");
            assert_eq!(
                dir.join(name).exists(),
                want == StatusCode::CREATED,
                "{name}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn concurrent_uploads_to_one_new_name_create_exactly_once() {
        // Two signed PUTs racing for one new name with
        // overwriting off: exactly one creates, the other
        // is a 409, whichever the scheduler runs first
        // (issue #192, audit R2-21). The conditional
        // create, not the pre-check, is what guarantees
        // this when both pass the check together.
        let dir = upload_dir("race");
        let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, false).await;
        let name = "cat.png";
        let sig = crate::url::upload_signature(SECRET, name);
        let url = format!("{base}/upload/{sig}/{name}");
        let png_of = |w: u32| {
            let img = DynamicImage::new_rgb8(w, 10);
            let mut buf = std::io::Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
            buf.into_inner()
        };
        let (a, b) = tokio::join!(
            client().put(&url).body(png_of(10)).send(),
            client().put(&url).body(png_of(20)).send(),
        );
        let mut statuses: Vec<StatusCode> =
            [a, b].into_iter().map(|r| r.unwrap().status()).collect();
        statuses.sort();
        assert_eq!(statuses, [StatusCode::CREATED, StatusCode::CONFLICT]);
        // The stored object is one of the two bodies,
        // whole: never a torn mix.
        let stored = std::fs::read(dir.join(name)).unwrap();
        assert!(stored == png_of(10) || stored == png_of(20));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn upload_validation_waits_behind_the_process_gate() {
        // Validation decodes run under the process gate
        // (issue #187): while the test holds the only
        // permit, a valid upload cannot complete; once
        // the permit is released it stores normally. The
        // `processing_gate_bounds_concurrency` pattern, at
        // the HTTP seam.
        let dir = upload_dir("gated");
        let gate = Arc::new(Gate::new(1));
        let base = spawn_app_with_upload_gate(&dir, 10 * 1024 * 1024, false, gate.clone()).await;
        let name = "cat.png";
        let sig = crate::url::upload_signature(SECRET, name);
        let held = gate.acquire_owned().await.unwrap();
        let pending = client()
            .put(format!("{base}/upload/{sig}/{name}"))
            .body(test_png())
            .send();
        tokio::pin!(pending);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(300), &mut pending)
                .await
                .is_err(),
            "the upload must wait while the only process permit is held"
        );
        assert!(!dir.join(name).exists());
        drop(held);
        let resp = pending.await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        assert_eq!(std::fs::read(dir.join(name)).unwrap(), test_png());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn upload_validation_is_shed_when_the_gate_stays_saturated() {
        // A gate that never drains sheds the upload with
        // the gate's own 503 and Retry-After once the
        // wait elapses, instead of decoding ungated, and
        // nothing is stored (issue #187).
        let dir = upload_dir("shed");
        let gate = Arc::new(Gate::new(1));
        let base = spawn_app_with_upload_gate(&dir, 10 * 1024 * 1024, false, gate.clone()).await;
        let name = "cat.png";
        let sig = crate::url::upload_signature(SECRET, name);
        let _held = gate.acquire_owned().await.unwrap();
        let resp = client()
            .put(format!("{base}/upload/{sig}/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers().get("retry-after").unwrap(), "1");
        assert!(!dir.join(name).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn upload_route_is_absent_when_not_configured() {
        // The plain test app has no upload destination:
        // the PUT route was never registered, so this is
        // a 404 (no upload surface at all), not a 403.
        let base = spawn_app(false).await;
        let name = "cat.png";
        let sig = crate::url::upload_signature(SECRET, name);
        let resp = client()
            .put(format!("{base}/upload/{sig}/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn uploads_leave_a_metric_trace() {
        let dir = upload_dir("metrics");
        let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, false).await;
        let name = "cat.png";
        let sig = crate::url::upload_signature(SECRET, name);
        let resp = client()
            .put(format!("{base}/upload/deadbeef/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let resp = client()
            .put(format!("{base}/upload/{sig}/{name}"))
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let metrics = metrics_text(&base).await;
        assert!(metrics.contains("crabcrop_uploads_total 2"), "{metrics}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn cli_upload_stores_and_prints_the_source_reference() {
        // The CLI verb end to end, mirroring the CLI purge
        // tests: it mints the upload signature, PUTs the
        // file, and prints the server's source reference.
        let dir = upload_dir("cli");
        let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, false).await;
        let file = dir.join("local-cat.png");
        std::fs::write(&file, test_png()).unwrap();
        let out = crate::cli::exec(&[
            "upload".to_string(),
            "--base".to_string(),
            base.clone(),
            "--secret".to_string(),
            "test-secret".to_string(),
            file.to_string_lossy().to_string(),
            "heroes/cat.png".to_string(),
        ])
        .await
        .unwrap();
        assert_eq!(out, "file:///heroes/cat.png");
        assert_eq!(
            std::fs::read(dir.join("heroes/cat.png")).unwrap(),
            test_png()
        );
        // A second run is the server's 409, surfaced as an
        // error: the CLI never overwrites silently either.
        let err = crate::cli::exec(&[
            "upload".to_string(),
            "--base".to_string(),
            base.clone(),
            "--secret".to_string(),
            "test-secret".to_string(),
            file.to_string_lossy().to_string(),
            "heroes/cat.png".to_string(),
        ])
        .await
        .unwrap_err();
        assert!(err.contains("409"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn second_identical_request_hits_the_cache() {
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let first = client().get(&url).send().await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let first_body = first.bytes().await.unwrap();
        let second = client().get(&url).send().await.unwrap();
        assert_eq!(second.status(), StatusCode::OK);
        assert_eq!(second.bytes().await.unwrap(), first_body);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        let body = metrics_text(&base).await;
        assert!(body.contains("crabcrop_cache_hits_total 1"), "{body}");
        assert!(body.contains("crabcrop_cache_misses_total 1"), "{body}");
    }

    #[tokio::test]
    async fn different_operations_are_different_entries() {
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        for size in ["300x50", "200x50", "300x50"] {
            let remainder = format!("{size}/{src}");
            let resp = client()
                .get(format!("{base}/{}/{remainder}", sign(&remainder)))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
        // Two distinct operations fetched once each; the repeat was cached.
        assert_eq!(hits.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn failed_requests_are_not_cached() {
        let (src, hits) = spawn_counting_server(500, "text/plain", b"nope".to_vec()).await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        for _ in 0..2 {
            let resp = client().get(&url).send().await.unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        }
        // Both attempts reached the source: nothing was cached.
        assert_eq!(hits.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn negotiates_avif_from_accept() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("accept", "image/avif,image/webp,*/*")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/avif"
        );
        assert_eq!(
            resp.headers().get(header::VARY).unwrap(),
            "Accept, DPR, Sec-CH-DPR, Sec-CH-Width, Sec-CH-Viewport-Width, Width, Viewport-Width, Save-Data"
        );
        let body = resp.bytes().await.unwrap();
        assert_eq!(&body[4..8], b"ftyp");
    }

    #[test]
    fn negotiation_honors_q_values() {
        use crate::url::ImageFormat::{Avif, Jxl, WebP};
        // Absent header and bare wildcards keep the source.
        assert_eq!(negotiate_format(None), None);
        assert_eq!(negotiate_format(Some("*/*")), None);
        assert_eq!(negotiate_format(Some("text/html, image/png")), None);
        // A single offer wins; missing q means 1.0.
        assert_eq!(negotiate_format(Some("image/webp")), Some(WebP));
        // Equal q: the server preference order decides.
        assert_eq!(negotiate_format(Some("image/webp, image/avif")), Some(Avif));
        assert_eq!(
            negotiate_format(Some("image/webp;q=0.9, image/jxl;q=0.9")),
            Some(Jxl)
        );
        // Higher q beats server preference.
        assert_eq!(
            negotiate_format(Some("image/avif;q=0.5, image/webp")),
            Some(WebP)
        );
        assert_eq!(
            negotiate_format(Some("image/avif;q=0.5, image/webp;q=0.4")),
            Some(Avif)
        );
        // q=0 refuses, however often the format is listed.
        assert_eq!(negotiate_format(Some("image/avif;q=0")), None);
        assert_eq!(
            negotiate_format(Some("image/avif;q=0, image/avif;q=1")),
            None
        );
        assert_eq!(
            negotiate_format(Some("image/avif;q=0, image/webp;q=0.5")),
            Some(WebP)
        );
        // An unparseable q is not a refusal.
        assert_eq!(negotiate_format(Some("image/webp;q=abc")), Some(WebP));
    }

    #[tokio::test]
    async fn q_values_can_overrule_server_preference() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("accept", "image/avif;q=0.4, image/webp;q=0.9")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/webp"
        );
    }

    #[tokio::test]
    async fn negotiates_webp_from_accept() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("accept", "image/webp")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/webp"
        );
    }

    #[tokio::test]
    async fn negotiates_jxl_from_accept() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("accept", "image/jxl")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/jxl"
        );
        let body = resp.bytes().await.unwrap();
        // Bare codestream signature, or the container signature.
        assert!(
            body.starts_with(&[0xFF, 0x0A])
                || body.starts_with(&[
                    0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A
                ]),
            "not a JXL stream"
        );
    }

    #[tokio::test]
    async fn jxl_sits_between_avif_and_webp_in_negotiation() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        // JXL beats WebP when AVIF is absent.
        let resp = client()
            .get(&url)
            .header("accept", "image/webp, image/jxl")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/jxl"
        );
        // AVIF still wins over JXL.
        let resp = client()
            .get(&url)
            .header("accept", "image/jxl, image/avif")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/avif"
        );
        // A refused JXL falls through to WebP.
        let resp = client()
            .get(&url)
            .header("accept", "image/jxl;q=0, image/webp")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/webp"
        );
    }

    #[tokio::test]
    async fn no_accept_keeps_source_format() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/png"
        );
    }

    #[tokio::test]
    async fn negotiation_respects_refusal() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("accept", "image/avif;q=0, image/webp")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/webp"
        );
    }

    #[tokio::test]
    async fn explicit_format_beats_accept() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:format(jpeg)/{src}");
        let resp = client()
            .get(format!("{base}/{}/{remainder}", sign(&remainder)))
            .header("accept", "image/avif")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/jpeg"
        );
    }

    #[tokio::test]
    async fn accept_classes_have_separate_cache_entries() {
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let remainder = format!("300x50/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let types = ["image/avif", "image/webp", "image/jxl", "image/avif"];
        let expected = ["image/avif", "image/webp", "image/jxl", "image/avif"];
        for (accept, want) in types.iter().zip(expected) {
            let resp = client()
                .get(&url)
                .header("accept", *accept)
                .send()
                .await
                .unwrap();
            assert_eq!(resp.headers().get(header::CONTENT_TYPE).unwrap(), want);
        }
        // Three fetches: one per Accept class; the repeat was cached.
        assert_eq!(hits.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn metrics_require_the_configured_token() {
        // Issue #270: with no token configured the
        // endpoint is closed; with one, it gates.
        let closed = spawn_app_with_metrics(false, None).await;
        let resp = client()
            .get(format!("{closed}/metrics"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        // /health stays open either way: it discloses
        // only liveness.
        let resp = client()
            .get(format!("{closed}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let base = spawn_app(false).await;
        let resp = client()
            .get(format!("{base}/metrics"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let resp = client()
            .get(format!("{base}/metrics"))
            .header("authorization", "Bearer wrong-token")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let resp = metrics_response(&base).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn metrics_reflect_traffic() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        // One served request.
        let remainder = format!("300x50/{src}");
        let sig = sign(&remainder);
        let resp = client()
            .get(format!("{base}/{sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // One failed request.
        let resp = client()
            .get(format!("{base}/deadbeef/300x50/{src}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);

        let body = metrics_text(&base).await;
        assert!(
            body.contains("crabcrop_requests_total{status_class=\"2xx\"} 1"),
            "{body}"
        );
        assert!(
            body.contains("crabcrop_requests_total{status_class=\"4xx\"} 1"),
            "{body}"
        );
        assert!(
            body.contains("crabcrop_request_duration_ms_count 2"),
            "{body}"
        );
        let served: u64 = body
            .lines()
            .find_map(|l| l.strip_prefix("crabcrop_bytes_served_total "))
            .unwrap()
            .parse()
            .unwrap();
        assert!(served > 0);
    }

    #[tokio::test]
    async fn health_check() {
        let base = spawn_app(false).await;
        let resp = client().get(format!("{base}/health")).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.text().await.unwrap(), "ok");
    }

    /// Sign with an arbitrary secret (the rotation app's
    /// previous key is not SECRET).
    fn sign_with(secret: &[u8], remainder: &str) -> String {
        use hmac::{Hmac, KeyInit, Mac};
        use sha2::Sha256;
        let mut mac = Hmac::<Sha256>::new_from_slice(secret).unwrap();
        mac.update(remainder.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// An app with presets and a caller-built loader, for
    /// the explain tests (preset expansion, object-store
    /// sources).
    async fn spawn_explain_app(
        presets: std::collections::HashMap<String, String>,
        loader: Loader,
    ) -> String {
        spawn_explain_app_with_roots(presets, loader, std::collections::HashMap::new()).await
    }

    async fn spawn_explain_app_with_roots(
        presets: std::collections::HashMap<String, String>,
        loader: Loader,
        source_roots: std::collections::HashMap<String, String>,
    ) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader,
            engine: Engine,
            process_timeout: std::time::Duration::from_secs(30),
            metrics: Arc::new(Metrics::default()),
            metrics_token: None,
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            inpaint: None,
            segment: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            moderation: None,
            embeddings: None,
            aesthetic: None,
            ocr: None,
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            meta_enabled: true,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(Gate::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            brand_color: None,
            presets,
            only_presets: false,
            source_roots,
            default_ops: Vec::new(),
            video_encoder: None,
            frame_extractor: None,
            pdf_renderer: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            slow_gate: Arc::new(Gate::new(2)),
            inflight: Arc::new(Inflight::default()),
            purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
            provenance_signer: None,
            config_fingerprint: "test".to_string(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn explain_bad_signature_fetches_nothing() {
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let remainder = format!("100x50/{src}");
        // The image path's own refusal, for comparison: the
        // explain refusal must be the same status and body.
        let image = client()
            .get(format!("{base}/deadbeef/{remainder}"))
            .send()
            .await
            .unwrap();
        let image_status = image.status();
        let image_body = image.text().await.unwrap();
        let resp = client()
            .get(format!("{base}/explain/deadbeef/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), image_status);
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(resp.text().await.unwrap(), image_body);
        assert_eq!(hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn explain_expired_fetches_nothing() {
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let base = spawn_app(false).await;
        let remainder = format!("exp:{}/100x50/{src}", now_unix() - 60);
        let sig = sign(&remainder);
        let image = client()
            .get(format!("{base}/{sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(image.status(), StatusCode::FORBIDDEN);
        assert_eq!(image.text().await.unwrap(), "signature expired");
        let resp = client()
            .get(format!("{base}/explain/{sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(resp.text().await.unwrap(), "signature expired");
        assert_eq!(hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn explain_reports_a_signed_url_and_caches_nothing() {
        let png = test_png();
        let png_len = png.len();
        let (src, hits) = spawn_counting_server(200, "image/png", png).await;
        let base = spawn_app(false).await;
        let remainder = format!("crop:100x50/filters:format(webp)/{src}");
        let sig = sign(&remainder);
        let explain_url = format!("{base}/explain/{sig}/{remainder}");

        let resp = client()
            .get(&explain_url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("cache-control").unwrap(), "no-store");
        assert_eq!(
            resp.headers().get("content-security-policy").unwrap(),
            "default-src 'none'"
        );
        assert_eq!(
            resp.headers().get("x-content-type-options").unwrap(),
            "nosniff"
        );
        let json: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(json["signature"]["status"], "valid");
        assert_eq!(json["signature"]["signed"], true);
        assert!(json["signature"]["key_id"].is_null());
        assert_eq!(json["source"]["url"], src);
        assert_eq!(json["source"]["kind"], "png");
        assert_eq!(json["source"]["width"], 100);
        assert_eq!(json["source"]["height"], 60);
        assert_eq!(json["source"]["bytes"], png_len);
        let first_segment = json["operations"][0]["segment"].as_str().unwrap();
        assert!(first_segment.contains("crop:100x50"), "{first_segment}");
        assert_eq!(json["forecast"]["width"], 100);
        assert_eq!(json["forecast"]["height"], 50);
        assert_eq!(json["forecast"]["format"], "webp");
        assert_eq!(json["forecast"]["format_from"], "operation");
        assert_eq!(json["forecast"]["pixels"], 5000);
        assert_eq!(json["links"]["image"], format!("/{sig}/{remainder}"));
        let debug = json["links"]["debug_focal"].as_str().unwrap();
        assert!(debug.contains("debug:focal"), "{debug}");
        assert!(debug.ends_with(&src), "{debug}");
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        // The HTML form carries the same facts, under the
        // same headers.
        let resp = client()
            .get(&explain_url)
            .header("Accept", "text/html")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            resp.headers()
                .get("content-type")
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );
        let body = resp.text().await.unwrap();
        assert!(body.contains("crabcrop URL report"), "{body}");
        assert!(body.contains(&src), "{body}");
        assert!(body.contains("crop to 100x50"), "{body}");
        assert_eq!(hits.load(Ordering::SeqCst), 2);

        // Explain wrote nothing to any cache tier: the
        // first image request still fetches the source, and
        // the second is served from the entry the first one
        // stored.
        let image_url = format!("{base}/{sig}/{remainder}");
        let resp = client().get(&image_url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 3);
        let resp = client().get(&image_url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(hits.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn explain_key_id_form_reports_the_key() {
        let src = spawn_image_server().await;
        let base = spawn_app_rotating().await;
        let remainder = format!("100x50/{src}");
        let sig = sign_with(b"the-old-server-secret", &remainder);
        let resp = client()
            .get(format!("{base}/explain/k-old:{sig}/{remainder}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(json["signature"]["signed"], true);
        assert_eq!(json["signature"]["key_id"], "k-old");
    }

    #[tokio::test]
    async fn explain_unsafe_form_honors_the_policy() {
        let src = spawn_image_server().await;
        let remainder = format!("100x50/{src}");
        let base = spawn_app(true).await;
        let resp = client()
            .get(format!("{base}/explain/unsafe/{remainder}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(json["signature"]["status"], "valid");
        assert_eq!(json["signature"]["signed"], false);

        let base = spawn_app(false).await;
        let resp = client()
            .get(format!("{base}/explain/unsafe/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(resp.text().await.unwrap(), "bad signature");
    }

    #[tokio::test]
    async fn explain_lists_preset_expansions_and_the_group() {
        let src = spawn_image_server().await;
        let mut presets = std::collections::HashMap::new();
        presets.insert("small".to_string(), "100x50/filters:grayscale".to_string());
        let base = spawn_explain_app(presets, Loader::with_defaults().unwrap()).await;
        let remainder = format!("preset:small/{src}");
        let sig = sign(&remainder);
        let resp = client()
            .get(format!("{base}/explain/{sig}/{remainder}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(json["presets"][0]["name"], "small");
        assert_eq!(json["presets"][0]["operations"][0]["segment"], "100x50");
        assert_eq!(
            json["presets"][0]["operations"][1]["segment"],
            "filters:grayscale"
        );
        assert_eq!(json["operations"][0]["segment"], "100x50");
        assert_eq!(json["operations"][1]["segment"], "filters:grayscale");
        assert_eq!(json["cache_group"], "small");
        assert_eq!(json["canonical"], format!("100x50/filters:grayscale/{src}"));
    }

    #[tokio::test]
    async fn explain_s3_source_reports_facts_only() {
        let png = test_png();
        let png_len = png.len();
        let (store_url, hits) = spawn_counting_server(200, "image/png", png).await;
        let endpoint = store_url.strip_suffix("/src.png").unwrap().to_string();
        let loader = Loader::with_defaults()
            .unwrap()
            .with_sources(crate::loader::SourceConfig {
                s3_endpoint: Some(endpoint.clone()),
                allowed_buckets: vec!["photos".to_string()],
                ..Default::default()
            });
        let base = spawn_explain_app(std::collections::HashMap::new(), loader).await;
        let remainder = "100x50/s3://photos/cat.png".to_string();
        let sig = sign(&remainder);
        let resp = client()
            .get(format!("{base}/explain/{sig}/{remainder}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await.unwrap();
        // The store's own URL (the server's endpoint
        // configuration) is never echoed back.
        assert!(!body.contains(&endpoint), "{body}");
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["source"]["url"], "s3://photos/cat.png");
        assert_eq!(json["source"]["kind"], "png");
        assert_eq!(json["source"]["width"], 100);
        assert_eq!(json["source"]["height"], 60);
        assert_eq!(json["source"]["bytes"], png_len);
        // Kind, dimensions, and size, and nothing else.
        assert_eq!(json["source"].as_object().unwrap().len(), 5);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn explain_escapes_markup_bearing_strings() {
        // The full markup cases (script tags in a source, a
        // preset name, a key id) are covered at the render
        // layer in the explain module's tests; through HTTP
        // the URL grammar itself carries entity text in a
        // source path and an ampersand in a preset name, and
        // both must come back escaped in HTML and verbatim
        // in JSON.
        let raw = spawn_bytes_server(test_png()).await;
        let src = raw.replace("src.png", "a&lt;b.png");
        let mut presets = std::collections::HashMap::new();
        presets.insert("a&b".to_string(), "100x50".to_string());
        let base = spawn_explain_app(presets, Loader::with_defaults().unwrap()).await;
        let remainder = format!("preset:a&b/{src}");
        let sig = sign(&remainder);
        let url = format!("{base}/explain/{sig}/{remainder}");

        let resp = client()
            .get(&url)
            .header("Accept", "text/html")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await.unwrap();
        assert!(body.contains("a&amp;lt;b.png"), "{body}");
        assert!(!body.contains("a&lt;b.png"), "{body}");
        assert!(body.contains("preset:a&amp;b"), "{body}");

        let resp = client()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(json["source"]["url"], src);
        assert_eq!(json["presets"][0]["name"], "a&b");
    }
}
