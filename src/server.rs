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
    AI_GATE_WAIT, ANIM_GATE_WAIT, Flight, FlightSlot, JobPermits, JobSource, Lane, MetaFlight,
    PROCESS_GATE_WAIT, ProcessFailure, ProcessJob, SOURCE_BUDGET_WAIT, SharedFlight,
    SharedResponse, VIDEO_GATE_WAIT, WatermarkSources, run_process, wants_animation_gate,
    wants_slow_lane,
};
pub use process::{DEFAULT_SLOW_LANE_PERMITS, Gate, Inflight};
#[cfg(test)]
pub(in crate::server) use process::{SLOW_LANE_WAIT, apply_provenance};

/// Shared state for the handlers.
#[derive(Debug, Clone)]
pub struct AppState {
    /// The signing keyring (issue #143), from `CRABCROP_SECRET`:
    /// the current key plus, during a rotation window, the
    /// previous one (`CRABCROP_SECRET_PREVIOUS`).
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
            "{}{}{}{}",
            state.metrics.render(),
            lane_metrics(&state),
            source_buffer_metrics(&state),
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

/// Source buffer budget occupancy for `/metrics` (issue
/// #403): the source and watermark bytes cache misses hold
/// right now, against the configured budget, read live at
/// scrape time. In use sitting at capacity while requests
/// shed with `server_busy` is the signal to raise
/// `CRABCROP_SOURCE_BUFFER_MAX_MB`.
fn source_buffer_metrics(state: &AppState) -> String {
    let budget = state.loader.source_budget();
    format!(
        "# HELP crabcrop_source_buffer_bytes Source bytes cache misses hold against the source buffer budget.\n\
         # TYPE crabcrop_source_buffer_bytes gauge\n\
         crabcrop_source_buffer_bytes {}\n\
         # HELP crabcrop_source_buffer_capacity_bytes The source buffer budget.\n\
         # TYPE crabcrop_source_buffer_capacity_bytes gauge\n\
         crabcrop_source_buffer_capacity_bytes {}\n",
        budget.in_use(),
        budget.capacity()
    )
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

/// The widest width a client hint may resolve to (issue
/// #403). Hints are client-controlled and the hinted width
/// joins the cache key, so without a ceiling one URL fans
/// out into a key (and a render) per quantization step all
/// the way to the parser's dimension cap. 3840 is the UHD
/// display width; the DPR hint still multiplies on top,
/// and the hinted op never enlarges past the source either
/// way.
const MAX_HINT_WIDTH: u32 = 3840;
const _: () = assert!(MAX_HINT_WIDTH <= crate::url::MAX_DIMENSION);

/// Round a hinted width UP to the next multiple of the
/// quantization step, so a client never gets less than it
/// asked for and the cache sees bounded cardinality, then
/// clamp it to [`MAX_HINT_WIDTH`]. The clamp happens here,
/// before the cache key is built, so hints past the
/// ceiling all share the ceiling's entry. It also keeps the
/// synthesized op inside the parser's dimension cap: the op
/// bypasses parsing, so a near-cap hint at a coarse step
/// must not mint a size the URL grammar itself would
/// reject.
fn quantize_width(width: u32, step: u32) -> u32 {
    let step = step.max(1);
    (width.div_ceil(step) * step).min(MAX_HINT_WIDTH)
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
/// the caller answers with the original source error. The copy
/// is charged to the request's source budget share like a
/// source (issue #403); no budget within the wait is `None` too.
async fn fallback_source_bytes(
    state: &AppState,
    charge: &crate::loader::SourceCharge,
) -> Option<Vec<u8>> {
    match &state.fallback {
        Some(Fallback::Bytes(bytes)) => {
            charge.grow_to(charge.bytes() + bytes.len()).await.ok()?;
            Some(bytes.as_ref().clone())
        }
        Some(Fallback::Url(url)) => state
            .loader
            .load_media_charged(url, crate::loader::MediaAdmission::default(), charge)
            .await
            .ok(),
        None => None,
    }
}

/// Whether a probe head settles the explain report's
/// source facts: the probe knows the kind, and it either
/// has the dimensions or the kind is one whose dimensions
/// never live in headers (video, PDF). Anything else means
/// the identifying header may still be on its way.
fn probe_settled(head: &[u8]) -> bool {
    let probe = crate::engine::probe_source(head);
    probe.kind != "unknown" && (probe.dimensions.is_some() || matches!(probe.kind, "video" | "pdf"))
}

/// `GET /explain/<signature>/<operations...>/<image-url>`:
/// the self-explaining URL report (issue #196; the report
/// itself is built in [`crate::explain`]).
///
/// The checks mirror the image path exactly and run in its
/// order: signature grammar, signature verification, the
/// reserved-segment guard, preset expansion, operation
/// parsing, the operation cap, expiry. Only then is the
/// source touched, once, through the loader with all of
/// its guards, and the facts come from header probing
/// (`crate::engine::probe_source`): no decode, no
/// detection, no process, AI, or video gate, and nothing
/// written to any cache tier. A bad signature, an over-cap
/// operation list, or an expired URL therefore costs zero
/// source fetches, and answers with the image path's own
/// status and body.
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
    // The operation cap (issue #469), on the list the image
    // path counts (the defaults merged in) and at the same
    // point: an over-cap URL answers its 400 here, before the
    // probe fetch.
    if let Err(e) = crate::url::check_operation_count(&crate::url::merge_default_ops(
        &state.default_ops,
        ops.clone(),
    )) {
        return parse_failure(&e, accept.as_deref());
    }
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

/// How many shared-tier objects one purge request scans
/// inline (issue #368): the shared walk reads entry
/// headers one at a time, so its inline cost must not
/// grow with the whole bucket. A walk that outgrows the
/// budget finishes in a background task and logs its own
/// count on completion. Correctness does not wait for
/// it: the tombstone keeps writes begun before the purge
/// out (issue #350), and the purge stays active until
/// the walk ends, so a lookup that overlaps the walk
/// treats a copy the purge covers as a miss (issue #400).
const SHARED_PURGE_INLINE_SCAN: usize = 128;

/// Queue the un-scanned remainder of a bounded shared
/// purge, if any, on a background task (issue #368). The
/// purge stays active until that task finishes; with
/// nothing left to walk it ends here.
fn spawn_shared_purge_rest(
    shared: &Arc<crate::s3store::S3Store>,
    pending: Option<crate::s3store::PendingPurge>,
    active: crate::cache::ActivePurge,
    event: &'static str,
    target: &str,
) {
    if let Some(pending) = pending {
        let shared = shared.clone();
        let target = target.to_string();
        tokio::spawn(async move {
            let removed = shared.finish_purge(pending).await;
            drop(active);
            purge_log(event, &target, Some(removed));
        });
    }
}

/// One structured line per purge request (audit PL-8): the
/// destructive, secret-gated endpoint must be auditable. The
/// target is hashed; a raw target can carry signed URLs.
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
    // still carries it.
    let task_state = state.clone();
    let task_target = target.to_string();
    let removed = tokio::spawn(async move {
        // The tombstone lands before any tier is touched
        // (issue #350): a write begun before this purge can
        // then see it whenever it tries to land, including
        // after this handler has finished. The purge stays
        // active until the shared walk ends (issue #400).
        let active = task_state
            .purge_guard
            .begin(crate::cache::PurgeTarget::Key(key_form.clone()));
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
            spawn_shared_purge_rest(shared, pending, active, "purge", &task_target);
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
        let active = task_state
            .purge_guard
            .begin(crate::cache::PurgeTarget::Group(group.clone()));
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
            spawn_shared_purge_rest(shared, pending, active, "purge-group", &task_target);
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

/// Read an upload body of at most `cap` bytes, charging it
/// to `charge` as it arrives (issue #403), in steps so a
/// trickling client does not touch the budget per frame.
/// [`LoadError::TooLarge`] past the cap,
/// [`LoadError::Network`] for an unreadable body, and
/// [`LoadError::Busy`] when the request's wait for budget
/// ran out.
async fn read_upload_body(
    mut body: axum::body::Body,
    cap: usize,
    charge: &crate::loader::SourceCharge,
) -> Result<Vec<u8>, LoadError> {
    use axum::body::HttpBody as _;
    const STEP: usize = 256 * 1024;
    let mut out = Vec::new();
    while let Some(frame) =
        std::future::poll_fn(|cx| std::pin::Pin::new(&mut body).poll_frame(cx)).await
    {
        let frame = frame.map_err(|e| LoadError::Network(e.to_string()))?;
        let Ok(data) = frame.into_data() else {
            continue;
        };
        let received = out.len() + data.len();
        if received > cap {
            return Err(LoadError::TooLarge { limit: cap as u64 });
        }
        let step = received.div_ceil(STEP).saturating_mul(STEP);
        charge.grow_to(step.min(cap).max(received)).await?;
        out.extend_from_slice(&data);
    }
    charge.shrink_to(out.len());
    Ok(out)
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
        let active = task_state
            .purge_guard
            .begin(crate::cache::PurgeTarget::Source(source.clone()));
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
            spawn_shared_purge_rest(shared, pending, active, "purge-source", &source);
        }
        purge_log("purge-source", &source, Some(removed));
        removed
    })
    .await
    .unwrap_or(0)
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
    // A buffered upload is a source like a fetched one
    // (issue #403): its body is charged to the source buffer
    // budget as it arrives and the charge is held until the
    // handler is done with the bytes, so a burst of uploads
    // waits at the budget instead of buffering past it, and
    // a client that stalls mid-body holds only what it sent.
    // The read is bounded by the cap itself: it fails the
    // moment the body passes it, which is the oversize
    // answer (413) without ever holding more than the cap in
    // memory; an unreadable body answers the same.
    let charge = state.loader.source_budget().charge(SOURCE_BUDGET_WAIT);
    let cap = usize::try_from(upload.max_bytes).unwrap_or(usize::MAX);
    let bytes = match read_upload_body(body, cap, &charge).await {
        Ok(bytes) => bytes,
        Err(LoadError::Busy) => {
            state.metrics.record_upload();
            upload_log(name, "rejected:busy");
            return source_budget_busy(accept.as_deref());
        }
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
    // The body moves into validation rather than being
    // copied for it, so the charge covers what is held.
    let prepared = match run_upload_validation(&state, bytes).await {
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

/// Run one image request, timing each stage into the returned record.
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
    // Server-wide defaults (issue #145) merge here: prepended
    // to the request's own operations, so the directive
    // analysis below, the engine, and the cache key all see
    // the merged list, and the URL wins every last-wins
    // directive (format, quality, fill) by coming later. The
    // signature never covers the defaults: they are operator
    // configuration, not request content.
    let ops = crate::url::merge_default_ops(&state.default_ops, ops);
    // The operation cap (issue #469) counts this merged list,
    // the one the engine runs, and answers here: before the
    // cache, the gates, and the source fetch, so an over-cap
    // URL (a short one repeating a preset included) costs a
    // parse and nothing more.
    if let Err(e) = crate::url::check_operation_count(&ops) {
        if sealed {
            return sealed_refusal(record);
        }
        return (parse_failure(&e, accept), record);
    }
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
    // Debug requests bypass everything, per request.
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
    // A leader whose result no tier stored (an error, a
    // fallback, a render with the caches off) leaves its
    // response in the flight's slot instead, and a waiter
    // whose request is identical to the leader's answers
    // with it (issue #407): it is the answer the waiter
    // would have produced itself, and taking it keeps a
    // burst on a failing origin at one fetch per flight
    // rather than one per waiter. Identical means the same
    // raw path (the sealing, the cache groups, and the
    // cache lifetime all come from it) and the same header
    // inputs as the miss path sees them: the negotiated
    // format, not the raw Accept (on this path errors are
    // plain text whatever it says), and the quantized
    // width, not the raw hint, so browsers that phrase the
    // same request differently still share one flight. The
    // conditional header is left out: each waiter applies
    // its own. A waiter that cannot take the response
    // (another spelling of the key, a body past the share
    // cap, a leader whose request was dropped, a run a
    // purge of the key overtook) leads or follows the next
    // flight. Waits are capped: a waiter whose flights
    // keep ending without an answer for it stops queueing
    // and runs the miss path itself (issue #375), so it
    // ends with the origin's real answer instead of a busy
    // response masking it.
    let flight_request = format!("{path:?}{preferred:?}{dpr_hint:?}{hinted_width:?}{save_data}");
    let mut waits = 0;
    let mut followed: Option<Arc<FlightSlot>> = None;
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
        let taken = followed
            .take()
            .and_then(|slot| {
                slot.shared
                    .lock()
                    .unwrap()
                    .as_ref()
                    .filter(|shared| shared.request == flight_request)
                    .cloned()
            })
            .filter(|shared| {
                !state
                    .purge_guard
                    .purge_covers(shared.snapshot, &cache_key, &groups)
            });
        if let Some(shared) = taken {
            record.bytes_out = shared.response.body.len();
            record.fallback = shared.fallback;
            let mut response = respond_shared(&shared.response);
            if let Some(etag) = &shared.etag {
                finalize_validators(&mut response, etag, if_none_match);
            }
            return (response, record);
        }
        match state.inflight.lead(&cache_key) {
            Flight::Leader(guard, slot) => {
                let snapshot = state.purge_guard.snapshot();
                let (response, record, render_etag) = serve_miss(state, work, record).await;
                let (mut response, shared) = share_response(response, FLIGHT_SHARE_CAP).await;
                if let Some(shared) = shared {
                    *slot.shared.lock().unwrap() = Some(SharedFlight {
                        request: flight_request,
                        snapshot,
                        response: shared,
                        etag: render_etag.clone(),
                        fallback: record.fallback,
                    });
                }
                drop(guard);
                if let Some(etag) = &render_etag {
                    finalize_validators(&mut response, etag, if_none_match);
                }
                return (response, record);
            }
            Flight::Follower(mut changed, slot) => {
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
                followed = Some(slot);
            }
        }
    }
}

/// The body size above which an image flight's response
/// is not shared with its followers (issue #407). The
/// body is already in memory and followers share the
/// leader's buffer rather than copying it, so the cap
/// only bounds what one flight can pin for its followers;
/// it sits well above any realistic still render.
const FLIGHT_SHARE_CAP: usize = 64 * 1024 * 1024;

/// The body size above which a meta document is not
/// shared with a flight's followers (issue #295): real
/// documents are KB-scale, so the cap only ever refuses
/// the pathological.
const META_SHARE_CAP: usize = 4 * 1024 * 1024;

/// Meta singleflight (issue #295): concurrent identical
/// meta requests share one run of the miss path instead
/// of each loading and rendering the same source. The
/// leader runs `serve_miss` unchanged and collects its
/// response into the flight's slot before its guard
/// drops, and followers answer with a reconstructed
/// response (the leader's status, headers, and body) and
/// their own request record, which truthfully shows they
/// ran no load and no process stage. Errors are shared
/// too (issue #407): meta errors are always the JSON
/// envelope, shaped only by inputs the flight key pins,
/// so a follower receives exactly the failure a solo
/// request gets, and a burst against a failing origin
/// costs one fetch instead of one per follower. A flight
/// that leaves its slot empty (a body past the share
/// cap, a leader whose request was dropped) sends its
/// followers to lead or follow the next flight, and
/// finally to run the miss path themselves. Nothing
/// outlives the flight: meta stays uncached across
/// time.
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
                let (response, shared) = share_response(response, META_SHARE_CAP).await;
                if let Some(shared) = shared {
                    *slot.shared.lock().unwrap() = Some(shared);
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

/// Rebuild a response from a flight's shared response:
/// the leader's status and headers, its body bytes.
fn respond_shared(shared: &SharedResponse) -> Response {
    let mut response = Response::new(axum::body::Body::from(shared.body.clone()));
    *response.status_mut() = shared.status;
    *response.headers_mut() = shared.headers.clone();
    response
}

/// Split a leader's response into the shared form a
/// flight slot carries, rebuilding the leader's own
/// response from the same parts. Returns `None` for the
/// shared form when the body is not collected whole
/// within `cap`; the miss path's bodies are in-memory,
/// so in practice only the cap refuses, and the leader's
/// response is rebuilt untouched in that case whenever
/// the body was not consumed trying.
async fn share_response(response: Response, cap: usize) -> (Response, Option<SharedResponse>) {
    let (parts, body) = response.into_parts();
    if <axum::body::Body as axum::body::HttpBody>::size_hint(&body)
        .exact()
        .is_none_or(|n| n > cap as u64)
    {
        return (Response::from_parts(parts, body), None);
    }
    match axum::body::to_bytes(body, cap).await {
        Ok(bytes) => {
            let shared = SharedResponse {
                status: parts.status,
                headers: parts.headers.clone(),
                body: bytes,
            };
            let response = Response::from_parts(parts, axum::body::Body::from(shared.body.clone()));
            (response, Some(shared))
        }
        Err(_) => {
            // Unreachable for the in-memory bodies the
            // miss path answers with (the size hint above
            // already vouched for the body); the status
            // and headers survive, the body does not.
            eprintln!("flight: leader body could not be collected for sharing");
            (Response::from_parts(parts, axum::body::Body::empty()), None)
        }
    }
}

/// How many times one request may wait on another
/// request's in-flight miss for the same key before it
/// stops waiting and runs the miss path itself (audit
/// R2-09, issue #375). A burst of identical requests
/// settles on the first flight (issue #407), so the
/// cap only comes into play for waiters no flight could
/// answer.
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
/// the generation the lookup started with. A disk or
/// shared copy that a purge covers is a miss, and is
/// deleted on sight, when that purge was running at any
/// point since the lookup began (issue #400): the purge
/// may not have reached the copy when it was read, and
/// serving or promoting it would outlive the purge. The
/// memory tier needs no such check: a purge empties it
/// before any slower tier, and every way back in is a
/// promotion through this chain or a guarded write.
/// Returns the image and the `Cache-Status` value for
/// the tier that served it.
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
        // Checked after the read, against purges running at
        // any point since the snapshot: a walk can delete
        // this object and end while its bytes are in flight.
        if state
            .purge_guard
            .running_covers(snapshot, cache_key, &image.groups)
        {
            shared.remove(cache_key).await;
            return None;
        }
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
            if guard.running_covers(snapshot, &key, &image.groups) {
                // A purge running since the lookup began
                // covers this copy (issue #400): drop it
                // rather than serve and promote it.
                store.remove(&key);
                return None;
            }
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
    // Every byte this miss buffers, the source, its
    // watermarks, a fallback, is charged to the source
    // buffer budget as it arrives (issue #403), and the
    // charge travels with the bytes into the render job,
    // which returns it when it finishes. A burst of distinct
    // misses therefore waits here, before the gates below,
    // instead of each buffering a whole source first; a miss
    // whose wait for budget runs out sheds like a full gate.
    // Nothing is charged before an origin answers, and an
    // origin that answers and then stalls holds only what it
    // has delivered (see `SourceCharge`).
    let charge = state.loader.source_budget().charge(SOURCE_BUDGET_WAIT);
    let load_start = Instant::now();
    let mut used_fallback = false;
    let admission = crate::loader::MediaAdmission {
        video: wants_frame,
        pdf: wants_page,
    };
    let bytes = match state
        .loader
        .load_media_charged(source_url, admission, &charge)
        .await
    {
        Ok(b) => b,
        Err(e) => {
            record.load = load_start.elapsed();
            // A policy refusal (Forbidden) is a decision, not a
            // failure: it never falls back. Neither does an
            // exhausted budget (issue #403): that is overload,
            // and the fallback would need budget too. Any other
            // load failure degrades to the configured fallback
            // when there is one.
            let fb = if matches!(e, LoadError::Forbidden(_) | LoadError::Busy) {
                None
            } else {
                fallback_source_bytes(state, &charge).await
            };
            match fb {
                Some(b) => {
                    used_fallback = true;
                    b
                }
                None => {
                    // A budget shed is overload, like a full
                    // gate, not a loader error (issue #403).
                    if e != LoadError::Busy {
                        record.fail = FailStage::Load;
                    }
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
    // Watermark loads charge the same budget, against the
    // request's one wait (issue #403); their share is split
    // off to travel with the watermark bytes.
    let mut watermark_images: Vec<(String, Vec<u8>)> = Vec::new();
    for wm_url in &directives.watermark_urls {
        record.load_ran = true;
        let wm_bytes = match state
            .loader
            .load_media_charged(wm_url, crate::loader::MediaAdmission::default(), &charge)
            .await
        {
            Ok(b) => b,
            Err(e) => {
                if e != LoadError::Busy {
                    record.fail = FailStage::Load;
                }
                return (
                    sealed_or(sealed, load_error_response(e, accept)),
                    record,
                    None,
                );
            }
        };
        watermark_images.push((wm_url.to_string(), wm_bytes));
    }
    let watermark_bytes: usize = watermark_images.iter().map(|(_, b)| b.len()).sum();
    let watermark_sources = Arc::new(WatermarkSources {
        images: watermark_images,
        share: Some(charge.split_off(watermark_bytes)),
    });

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
    // with it (issue #357). The decision covers animated
    // GIF sources rendered to WebP as well as animated
    // WebP sources, and counts the negotiated format, not
    // only an explicit one (issue #405); it is made once,
    // so the gate, the lane, and a fallback retry agree.
    let animated_webp = wants_animation_gate(&bytes, &directives, preferred);
    let anim_permit = if animated_webp {
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
    // The op carries `no_upscale` semantics (issue #403): a
    // hint says how wide the client displays the image, not
    // that the server should invent pixels, so a hint past
    // the source returns the source width, and the DPR hint
    // runs into the same cap. The directive is request-wide,
    // but the URL has no size op of its own here, so the
    // synthesized op is the only resize it can bound.
    let job_ops = match hinted_width {
        Some(w) => {
            let mut combined = vec![crate::url::Operation::Size(crate::url::ImageSize {
                width: crate::url::Dimension::Pixels(w),
                height: crate::url::Dimension::Proportional,
                flip_horizontal: false,
                flip_vertical: false,
            })];
            if !directives.no_upscale {
                combined.push(crate::url::Operation::Filters(vec![
                    crate::url::Filter::NoUpscale,
                ]));
            }
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
    // the stills lane.
    let lane = if wants_slow_lane(animated_webp, wants_video, wants_frame, wants_page) {
        Lane::Slow
    } else {
        Lane::Process
    };
    let permits = JobPermits {
        ai: ai_permit,
        video: video_permit,
        anim: anim_permit,
    };
    let source = JobSource {
        bytes,
        share: Some(charge),
    };
    let processed = match run_process(state, &job, source, lane, process_deadline, permits).await {
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
            // The first run's source share drained with its
            // job, so the fallback is charged afresh (issue
            // #403), with a wait of its own like each gate
            // below; the watermarks keep the share they
            // travel with. No budget leaves the original
            // engine error standing, like a full gate.
            let retry_charge = state.loader.source_budget().charge(SOURCE_BUDGET_WAIT);
            let retry = if !used_fallback && matches!(e, EngineError::Decode(_)) {
                match fallback_source_bytes(state, &retry_charge).await {
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
                        if may_retry && animated_webp {
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
                            let source = JobSource {
                                bytes: fb,
                                share: Some(retry_charge),
                            };
                            run_process(state, &job, source, lane, process_deadline, retry_permits)
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
    // The source's budget share drained with its run; the
    // watermarks go now with theirs (issue #403), not after
    // the population writes and meta annotations.
    drop(job);
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
        // response path.
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
    // section like any other request. (The request's own
    // AI permit does not outlive the run, audit R2-29.)
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

/// The 503 for an exhausted source buffer budget (issue
/// #403): the same `server_busy` answer as a full gate,
/// with the stills lane's short Retry-After, since shares
/// free as soon as renders end.
fn source_budget_busy(accept: Option<&str>) -> Response {
    busy_response(SOURCE_BUDGET_BUSY, 1, accept)
}

const SOURCE_BUDGET_BUSY: &str = "server busy: too many source images held in memory";

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
        // The overload answer every gate gives (issue #403).
        LoadError::Busy => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_busy",
            SOURCE_BUDGET_BUSY.to_string(),
        ),
    }
}

fn load_error_response(e: LoadError, accept: Option<&str>) -> Response {
    if e == LoadError::Busy {
        return source_budget_busy(accept);
    }
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
mod tests;
