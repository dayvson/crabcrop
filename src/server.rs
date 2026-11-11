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

mod background;
pub use background::{Background, DEFAULT_SHUTDOWN_GRACE_SECS, Drained, SHARED_PUTS_AT_ONCE};
mod listen;
pub use listen::{
    ConnectionLimits, DEFAULT_HEADER_TIMEOUT_SECS, DEFAULT_IDLE_TIMEOUT_SECS, serve_connections,
};
mod process;
pub(in crate::server) use process::{
    AI_GATE_WAIT, ANIM_GATE_WAIT, Flight, FlightSlot, JobPermits, JobSource, Lane, MetaFlight,
    MissProgress, PROCESS_GATE_WAIT, ProcessFailure, ProcessJob, SOURCE_BUDGET_WAIT, SharedFlight,
    SharedResponse, VIDEO_GATE_WAIT, WatermarkSources, run_process, wants_animation_gate,
    wants_slow_lane,
};
pub use process::{DEFAULT_SLOW_LANE_PERMITS, Gate, Inflight};
#[cfg(test)]
pub(in crate::server) use process::{SLOW_LANE_WAIT, apply_provenance};

/// Shared state for the handlers.
#[derive(Clone)]
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
    /// `PUT /upload/...` originals land. `None` means
    /// uploads are off: the route is still registered, and
    /// the handler answers `404 uploads_not_configured`
    /// (issue #467).
    pub upload: Option<Arc<crate::upload::UploadStore>>,
    /// Whether meta mode is enabled (issue #390):
    /// `CRABCROP_META_MODE`, off by default (issue #453). A
    /// meta request bypasses every cache tier and runs a
    /// fresh render plus its annotation inferences on every
    /// call, so the most expensive request shape a signed
    /// URL can trigger is open only where the operator
    /// turned it on.
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
    /// Whether the image path reads the client hint headers
    /// (`CRABCROP_CLIENT_HINTS`, on by default). Off, every
    /// hint header is ignored, image answers vary on
    /// `Accept` alone, and no `Accept-CH` is advertised.
    pub client_hints: bool,
    /// Whether `/explain/...` is served (`CRABCROP_EXPLAIN`,
    /// on by default, issue #591). Off, every `GET` or
    /// `HEAD` to the route answers `404 not_found` before
    /// anything else.
    pub explain_enabled: bool,
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
    pub presets: crate::url::Presets,
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
    /// Bounds concurrent heavy `/explain` probes (an SVG's
    /// parse in the worker, an AVIF's decode, a HEIC's item
    /// graph walk), which run on blocking threads (see
    /// `explain_probe`): a burst of them waits here, or
    /// sheds with a 503, instead of occupying the async
    /// runtime or starting a worker process per request.
    pub probe_gate: Arc<tokio::sync::Semaphore>,
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
    /// The work handlers leave running (issue #463):
    /// shared-tier write-throughs, bounded to
    /// [`SHARED_PUTS_AT_ONCE`] in flight, purge fan-outs,
    /// disk-hit follow-ups and purge walks, all tracked, so
    /// a shutdown can drain the writes up to its deadline.
    pub background: Background,
    /// Fingerprint of the output-shaping configuration
    /// (audit R2-08), computed at boot in `main`: embedded
    /// in every cache key, so changing the metadata
    /// policy, quality tuning, width step, model files,
    /// font, or video encoder starts a fresh cache epoch
    /// instead of silently serving entries rendered under
    /// the old configuration from the persistent tiers.
    pub config_fingerprint: String,
}

impl std::fmt::Debug for AppState {
    /// The fields a derive would print, except that the
    /// `/metrics` token prints as `[redacted]`; the keyring,
    /// the loader's credentials, and the stores redact their
    /// own secrets (issue #455). A new field that holds a
    /// secret must be redacted here too.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = f.debug_struct("AppState");
        // Labels come from the field names themselves, so a
        // label cannot drift from its field, and none is a
        // snake_case string literal, which the error-code
        // inventory (`every_error_code_is_documented`) would
        // read as a response code.
        macro_rules! fields {
            ($($name:ident),* $(,)?) => {
                $(out.field(stringify!($name), &self.$name);)*
            };
        }
        fields!(
            keyring,
            allow_unsafe,
            loader,
            engine,
            process_timeout,
            metrics
        );
        out.field(
            stringify!(metrics_token),
            &self.metrics_token.as_ref().map(|_| crate::redact::Redacted),
        );
        fields!(
            cache,
            removebg,
            upscale,
            inpaint,
            segment,
            face,
            tagger,
            tag_threshold,
            moderation,
            embeddings,
            aesthetic,
            ocr,
            detect,
            detectors,
            focal,
            font,
            fallback,
            store,
            shared_store,
            upload,
            meta_enabled,
            ai_gate,
            process_gate,
            slow_gate,
            ssim_target,
            avif_speed,
            jxl_speed,
            width_step,
            client_hints,
            explain_enabled,
            preserve_metadata,
            brand_color,
            presets,
            only_presets,
            source_roots,
            default_ops,
            video_encoder,
            frame_extractor,
            pdf_renderer,
            provenance_signer,
            video_gate,
            anim_gate,
            inflight,
            purge_guard,
            background,
            config_fingerprint
        );
        out.finish()
    }
}

/// Build the router.
pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/explain/{*path}", get(explain_handle))
        .route("/purge/{*path}", axum::routing::post(purge_handle))
        .route(
            "/purge-group/{*path}",
            axum::routing::post(purge_group_handle),
        )
        // The upload route is registered whether or not a
        // destination is configured (issue #467): without
        // one, the handler answers `uploads_not_configured`,
        // so a client can tell a server with uploads off
        // from a mistyped path.
        .route("/upload/{*path}", axum::routing::put(upload_handle))
        .route("/{*path}", get(handle))
        // A PUT on an image path is a 404, not the 405 the
        // GET catch-all would otherwise produce for it:
        // uploads go to the upload route only (issue #168).
        // A PUT to a named route other than the upload
        // route is the 405 below.
        .route("/{*path}", axum::routing::put(put_not_found))
        // A path no route matches (the catch-all's wildcard
        // cannot match `/` itself) answers the documented
        // 404 instead of axum's bare one, and a method the
        // matched route does not serve answers a documented
        // 405; axum still sets the `Allow` header on the 405
        // (issue #563).
        .fallback(unrouted_path)
        .method_not_allowed_fallback(method_not_allowed)
        .with_state(state)
        .layer(axum::middleware::from_fn(security_headers))
}

/// The `Accept` header's value, for [`error_response`].
fn accept_of(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

/// The image catch-all's PUT handler: a PUT there is
/// simply not found. A PUT to a named route other than
/// `/upload` is a different answer: the router's
/// method-not-allowed fallback below (issue #563).
async fn put_not_found(headers: HeaderMap) -> Response {
    error_response(
        StatusCode::NOT_FOUND,
        "not_found",
        "not found".to_string(),
        accept_of(&headers).as_deref(),
    )
}

/// The router fallback: a path no route matches answers
/// the documented `404 not_found`, in the JSON envelope
/// when the client asked for it, instead of axum's bare
/// 404 with no code (issue #563).
async fn unrouted_path(headers: HeaderMap) -> Response {
    error_response(
        StatusCode::NOT_FOUND,
        "not_found",
        "not found".to_string(),
        accept_of(&headers).as_deref(),
    )
}

/// The router's answer when the path matches a route but
/// the method does not: `405 method_not_allowed` in the
/// same shape as every other refusal, instead of axum's
/// bare 405 (issue #563). axum adds the `Allow` header
/// naming the matched route's methods.
async fn method_not_allowed(headers: HeaderMap) -> Response {
    error_response(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        "method not allowed".to_string(),
        accept_of(&headers).as_deref(),
    )
}

/// Headers applied to every response (audit PL-5, PL-6):
/// `nosniff` everywhere, `no-store` on the system
/// endpoints, whose contents are operational data, not
/// cacheable content, and `no-store` on every error answer
/// that does not already carry it.
///
/// An error describes one request at one moment (a source
/// not there yet, a gate that was full, a header that asked
/// for something this URL cannot serve), never what the URL
/// is. Without cache headers a shared cache applies its own
/// default to an error, and some store one for seconds to
/// an hour under a key that leaves out the request headers
/// the answer depended on, so every viewer of the URL would
/// get it. Setting it here covers every route and the
/// router's own answers (an unrouted method's `405`) in one
/// place, so an error builder added later cannot miss it.
async fn security_headers(req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let system = matches!(req.uri().path(), "/health" | "/metrics");
    let mut response = next.run(req).await;
    let error = response.status().is_client_error() || response.status().is_server_error();
    let headers = response.headers_mut();
    headers.insert(
        "x-content-type-options",
        axum::http::HeaderValue::from_static("nosniff"),
    );
    let stored_nowhere = headers
        .get(header::CACHE_CONTROL)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| has_directive(v, "no-store"));
    if system || (error && !stored_nowhere) {
        headers.insert(
            header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-store"),
        );
    }
    response
}

/// Whether a `Cache-Control` value carries `directive`
/// (compared without case, ignoring any argument).
fn has_directive(cache_control: &str, directive: &str) -> bool {
    cache_control.split(',').any(|d| {
        d.split('=')
            .next()
            .is_some_and(|name| name.trim().eq_ignore_ascii_case(directive))
    })
}

async fn health() -> &'static str {
    "ok"
}

/// Equality for a bearer token that leaks nothing about it
/// through timing, its length included (issue #454): both
/// sides are hashed first, so the comparison always runs
/// over two 32-byte digests, however long the presented
/// token is.
fn token_eq(a: &[u8], b: &[u8]) -> bool {
    use sha2::Digest as _;
    let (a, b) = (sha2::Sha256::digest(a), sha2::Sha256::digest(b));
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
    // The auth scheme is case-insensitive (RFC 9110), so
    // `bearer` and `BEARER` name it too.
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("Bearer"))
        .map(|(_, credentials)| credentials);
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
            "{}{}{}{}{}",
            state.metrics.render(),
            lane_metrics(&state),
            source_buffer_metrics(&state),
            source_lookup_metrics(&state),
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

/// Source host lookup slot occupancy for `/metrics`: the
/// lookups running right now against the slots (see
/// [`crate::loader::LOOKUP_SLOTS`]), read live at scrape time.
/// In flight sitting at capacity while requests shed with
/// `server_busy` means the resolver is not answering for the
/// host names requests carry.
fn source_lookup_metrics(state: &AppState) -> String {
    let (in_flight, capacity) = state.loader.lookup_slots();
    format!(
        "# HELP crabcrop_source_lookups_in_flight Source host lookups running now.\n\
         # TYPE crabcrop_source_lookups_in_flight gauge\n\
         crabcrop_source_lookups_in_flight {in_flight}\n\
         # HELP crabcrop_source_lookups_capacity Source host lookups that can run at once.\n\
         # TYPE crabcrop_source_lookups_capacity gauge\n\
         crabcrop_source_lookups_capacity {capacity}\n"
    )
}

/// Cache tier failures for `/metrics` (issue #367): the
/// stores count their own failures as they log them; the
/// exposition reads the counters live at scrape time,
/// one series per configured tier. With a shared tier, its
/// write-through slots and skips follow (issue #463), then
/// what its last complete listing of the purge tombstones
/// found unread, and when that was (issue #577).
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
    // The disk tier's slots in use and its background work
    // (issue #572): queue lengths, never an index scan.
    if let Some(store) = &state.store {
        let (behind, deletions) = store.queued();
        out.push_str(&format!(
            "# HELP crabcrop_cache_disk_ops_in_flight Disk tier operations holding one of its 32 slots.\n\
             # TYPE crabcrop_cache_disk_ops_in_flight gauge\n\
             crabcrop_cache_disk_ops_in_flight {}\n\
             # HELP crabcrop_cache_disk_groups_behind Disk entries whose file has not caught up with the cache groups they were served under.\n\
             # TYPE crabcrop_cache_disk_groups_behind gauge\n\
             crabcrop_cache_disk_groups_behind {behind}\n\
             # HELP crabcrop_cache_disk_deletions_queued Purged or evicted disk entries whose file is not deleted yet.\n\
             # TYPE crabcrop_cache_disk_deletions_queued gauge\n\
             crabcrop_cache_disk_deletions_queued {deletions}\n",
            store.ops_in_flight()
        ));
    }
    if let Some(store) = &state.shared_store {
        // The write bound (issue #463): in flight against
        // its slots, and the render write-throughs skipped
        // because every slot was taken.
        let background = &state.background;
        out.push_str(&format!(
            "# HELP crabcrop_cache_shared_puts_in_flight Shared tier writes in flight.\n\
             # TYPE crabcrop_cache_shared_puts_in_flight gauge\n\
             crabcrop_cache_shared_puts_in_flight {}\n\
             # HELP crabcrop_cache_shared_puts_capacity Shared tier writes allowed in flight.\n\
             # TYPE crabcrop_cache_shared_puts_capacity gauge\n\
             crabcrop_cache_shared_puts_capacity {}\n\
             # HELP crabcrop_cache_shared_puts_skipped_total Shared tier render write-throughs skipped with every slot in flight.\n\
             # TYPE crabcrop_cache_shared_puts_skipped_total counter\n\
             crabcrop_cache_shared_puts_skipped_total {}\n",
            background.puts_in_flight(),
            background.put_capacity(),
            background.puts_skipped()
        ));
        // The tombstone gap (issue #577): the unread count
        // stays as of the last complete listing, so a listing
        // time that stops moving is the sign of a failing
        // listing.
        let listed_ms = store.tombstones_listed_ms();
        out.push_str(&format!(
            "# HELP crabcrop_cache_shared_tombstones_unread Purge tombstones the last complete listing named that this replica has never read; their purges do not apply here until read.\n\
             # TYPE crabcrop_cache_shared_tombstones_unread gauge\n\
             crabcrop_cache_shared_tombstones_unread {}\n\
             # HELP crabcrop_cache_shared_tombstones_listed_timestamp_seconds When this replica last listed the purge tombstones completely, in Unix seconds; 0 before its first complete listing.\n\
             # TYPE crabcrop_cache_shared_tombstones_listed_timestamp_seconds gauge\n\
             crabcrop_cache_shared_tombstones_listed_timestamp_seconds {}.{:03}\n",
            store.tombstones_unread(),
            listed_ms / 1000,
            listed_ms % 1000
        ));
    }
    out
}

/// The DPR hint headers [`dpr_hint`] reads, the
/// structured one first.
const DPR_HINT_HEADERS: [&str; 2] = ["Sec-CH-DPR", "DPR"];

/// The width hint headers [`width_hint`] reads, in the
/// order it prefers them.
const WIDTH_HINT_HEADERS: [&str; 4] = [
    "Sec-CH-Width",
    "Width",
    "Sec-CH-Viewport-Width",
    "Viewport-Width",
];

/// The header [`save_data_hint`] reads.
const SAVE_DATA_HEADER: &str = "Save-Data";

/// The `Vary` of every image-route answer while the server
/// reads client hints: `Accept` and every header the hint
/// readers above read, since each can change the bytes.
const VARY_WITH_HINTS: &str = "Accept, DPR, Sec-CH-DPR, Sec-CH-Width, Sec-CH-Viewport-Width, Width, Viewport-Width, Save-Data";

/// The `Vary` of every image-route answer with client hints
/// off (`CRABCROP_CLIENT_HINTS=0`): no hint header is read,
/// so only `Accept` can change an answer.
const VARY_WITHOUT_HINTS: &str = "Accept";

/// The `Vary` an image-route answer carries, success or
/// error: an error's body follows `Accept` (plain text or
/// the JSON envelope) and can follow a hint (a hinted size
/// that runs past a limit or the deadline), just as an
/// image does.
fn image_vary(client_hints: bool) -> axum::http::HeaderValue {
    axum::http::HeaderValue::from_static(if client_hints {
        VARY_WITH_HINTS
    } else {
        VARY_WITHOUT_HINTS
    })
}

/// The `DPR` client hint, when it names one of the settled
/// factors. Anything else (absent, unparsable, out of set) is
/// no hint: hints are advisory, never an error. A factor
/// that would push the output past its limits steps down
/// in the engine rather than failing the request (issue
/// #529).
fn dpr_hint(headers: &HeaderMap) -> Option<f32> {
    let [first, legacy] = DPR_HINT_HEADERS;
    let raw = headers
        .get(first)
        .or_else(|| headers.get(legacy))
        .and_then(|v| v.to_str().ok())?;
    raw.trim()
        .parse::<f32>()
        .ok()
        .filter(|f| crate::url::DPR_FACTORS.contains(f))
}

/// The width client hint: the display width the client
/// wants, from the most specific header present
/// (`Sec-CH-Width`, then the legacy `Width`, then the
/// viewport widths). Advisory like every hint: an absent,
/// unparsable, or out-of-range value is no hint, never an
/// error.
fn width_hint(headers: &HeaderMap) -> Option<u32> {
    for name in WIDTH_HINT_HEADERS {
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
        .get(SAVE_DATA_HEADER)
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
///
/// The load asks this after every chunk, on the async
/// runtime, so it never sizes a kind whose probe is heavy
/// ([`crate::engine::probe_is_heavy`]): an SVG's size needs
/// the whole document anyway, an AVIF's a decode, and a
/// HEIC's a walk of its item graph. Those read on to the end
/// of the body (or the head cap), and [`explain_probe`]
/// sizes them once, off the runtime.
fn probe_settled(head: &[u8]) -> bool {
    let kind = crate::engine::probe_kind(head);
    match kind {
        "unknown" => false,
        "video" | "pdf" => true,
        _ if crate::engine::probe_is_heavy(kind) => false,
        _ => crate::engine::probe_source(head).dimensions.is_some(),
    }
}

/// How long an explain probe may wait for a probe slot
/// before the server sheds it with a 503: the stills
/// gate's wait.
const PROBE_GATE_WAIT: std::time::Duration = PROCESS_GATE_WAIT;

/// The explain report's source probe. A heavy probe (an
/// SVG's parse in the worker, which may take the probe's
/// whole 5 s, an AVIF's decode, or a HEIC's item graph
/// walk) runs on a blocking thread under the probe gate,
/// so the async runtime, and every route it serves, never
/// waits on one; the permit moves into the job, so a probe
/// whose request went away still counts until it ends.
/// Every other kind's probe reads headers alone and runs
/// inline. `None` means the gate stayed full past its wait.
async fn explain_probe(state: &AppState, head: Vec<u8>) -> Option<crate::engine::SourceProbe> {
    let kind = crate::engine::probe_kind(&head);
    if !crate::engine::probe_is_heavy(kind) {
        return Some(crate::engine::probe_source(&head));
    }
    let permit = tokio::time::timeout(PROBE_GATE_WAIT, state.probe_gate.clone().acquire_owned())
        .await
        .ok()?
        .ok()?;
    let probe = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        crate::engine::probe_source(&head)
    })
    .await;
    // A probe that panicked states the kind and no size, as
    // one that failed does.
    Some(probe.unwrap_or(crate::engine::SourceProbe {
        kind,
        dimensions: None,
    }))
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
/// detection, no process, AI, or video gate (an SVG, AVIF,
/// or HEIC probe takes the probe gate, see
/// [`explain_probe`]), and nothing written to any cache
/// tier. A bad signature, an over-cap operation list, or an
/// expired URL therefore costs zero
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
    // With `CRABCROP_EXPLAIN=0` the route is off (issue #591):
    // every request answers the router's own `404 not_found`
    // before the signature is read, so the answer is the
    // same whatever the URL, and no source is fetched.
    if !state.explain_enabled {
        return error_response(
            StatusCode::NOT_FOUND,
            "not_found",
            "not found".to_string(),
            accept.as_deref(),
        );
    }
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
    // The reserved segment is refused before verification,
    // as on the image path.
    if crabcrop_url::is_reserved_remainder(rest) {
        let (status, code, message) = reserved_remainder_parts();
        return error_response(status, code, message, accept.as_deref());
    }
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
    // The operation cap (issue #469) and the watermark URL
    // cap (issue #613), on the list the image path counts
    // (the defaults merged in) and at the same point: an
    // over-cap URL answers its 400 here, before the probe
    // fetch.
    let counted = crate::url::merge_default_ops(&state.default_ops, ops.clone());
    if let Err(e) = crate::url::check_operation_count(&counted)
        .and_then(|()| crate::url::check_watermark_count(&counted))
    {
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
    // Server-wide defaults merge here, once, exactly as the
    // image path merges them: the admission, the canonical
    // form, and the forecast all follow the list the image
    // request runs (issue #468). The report's operation
    // list stays `ops`, the URL's own, so it never credits
    // the URL with a default.
    let merged = crate::url::merge_default_ops(&state.default_ops, ops.clone());
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
        video: merged
            .iter()
            .any(|op| matches!(op, crate::url::Operation::Frame(_))),
        pdf: merged
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
    let Some(probe) = explain_probe(&state, fetched.head).await else {
        return busy_response(
            "server busy: too many source probes running at once",
            5,
            accept.as_deref(),
        );
    };

    // Each referenced preset with the operations its
    // definition expands to. Definitions were validated at
    // boot; one that no longer parses (it cannot) is listed
    // by name alone rather than failing the report.
    let mut presets = Vec::new();
    for name in &groups {
        let entry_ops = state
            .presets
            .get(name)
            .and_then(|definition| {
                crate::url::parse_default_ops(definition.as_str(), &state.presets).ok()
            })
            .unwrap_or_default();
        presets.push(crate::explain::preset_entry(name, &entry_ops));
    }

    // The canonical form is the cache key's path form, of
    // the merged list as the image path keys it. The key's
    // config fingerprint suffix is operator state and stays
    // unreported. A form past the key-length bound is shown
    // in full, though the key carries its digest
    // (`cache::bounded_key_base`). For a root source the
    // form is built from the source as written: the key
    // itself is keyed on the expansion, but printing that
    // here would publish the root's base URL into the
    // report.
    let canonical = crate::url::canonical_form(&merged, source_url);
    let report = crate::explain::Report::new(crate::explain::ReportInput {
        signed: !matches!(signature, crate::url::Signature::Unsafe),
        key_id,
        expiry,
        now,
        source_url: source_url.to_string(),
        source_root: root_ref.map(|r| r.name),
        probe,
        // The DPR hint is read as the image path reads it, so
        // the forecast steps a hinted factor down as the image
        // request does (issue #529), and not at all with
        // client hints off.
        dpr_hint: dpr_hint(&headers).filter(|_| state.client_hints),
        source_bytes: fetched.total_bytes.map(|n| n as usize),
        ops,
        merged_ops: merged,
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
    if let Some(busy) = load_busy_response(&e, accept) {
        return busy;
    }
    eprintln!("explain load error: {e}");
    let (status, code, message) = load_error_parts(e);
    error_response(status, code, message, accept)
}

/// How many shared-tier objects one purge request scans
/// inline (issue #368): the shared walk reads one entry
/// header per object (a bounded number at a time, issue
/// #462), so its inline cost must not grow with the whole
/// bucket. A walk that outgrows the budget finishes in a
/// background task, waiting for one of the
/// [`crate::s3store::PURGE_WALKS_AT_ONCE`] turns, and logs
/// its own count on completion. Correctness does not wait
/// for it: the generation tombstone keeps writes begun
/// before the purge out (issue #350), and the purge's
/// stamped tombstone, here and in the bucket, makes every
/// copy written before the purge began a miss on every
/// replica until the walk has removed it (issues #400,
/// #461).
const SHARED_PURGE_INLINE_SCAN: usize = 128;

/// How often a replica re-reads the purge tombstones in the
/// shared bucket (issue #461): the longest another replica's
/// purge can take to reach this one's lookups.
const TOMBSTONE_REFRESH: std::time::Duration = std::time::Duration::from_secs(5);

/// Start a purge of `target`: record it in this process's
/// guard and, with a shared tier, as a tombstone in the
/// bucket (issue #461), before any tier is touched. The
/// bucket record is what keeps the purge effective on
/// other replicas, and on this one after a restart, while
/// its walk runs. Returns the running purge and the
/// tombstone's object name, `None` when there is no shared
/// tier or the write failed (counted and logged by the
/// store; the purge then holds on this process only).
async fn begin_purge(
    state: &AppState,
    target: crate::cache::PurgeTarget,
) -> (crate::cache::ActivePurge, Option<String>) {
    let active = state.purge_guard.begin(target);
    let tombstone = match &state.shared_store {
        Some(shared) => {
            shared
                .put_tombstone(active.tombstone(), active.sequence())
                .await
        }
        None => None,
    };
    (active, tombstone)
}

/// The shared tier's half of a purge (issues #368, #446,
/// #462): remove the covered copies written before the
/// purge began. A per-URL purge first deletes the current
/// epoch's variants by name. Then, when no identical purge
/// is walking the bucket in this process, this one leads a
/// walk: it scans at most [`SHARED_PURGE_INLINE_SCAN`]
/// objects inline and hands the remainder, if any, to a
/// background task that waits for a walk turn, tracked on
/// `background` (issue #463). When one is
/// walking, this purge joins that walk instead (see
/// [`crate::s3store::S3Store::lead_or_join`]), which
/// re-lists the bucket for it once its current pass ends.
/// A purge stays running until the walk that covers it
/// ends, then its tombstone is marked ended if the walk
/// reached every object (see [`end_purge`]). Returns what
/// the inline part removed.
async fn purge_shared_tier(
    shared: &Arc<crate::s3store::S3Store>,
    background: &Background,
    begun: BegunPurge,
    epoch_base: Option<String>,
    event: &'static str,
    log_target: &str,
) -> usize {
    let BegunPurge {
        active,
        tombstone,
        disk,
    } = begun;
    let target = active.tombstone().target.clone();
    let cutoff = active.tombstone().cutoff;
    // The walk lists these variants and reads them again, so
    // a read here that fails is one the walk repairs.
    let mut removed = match &epoch_base {
        Some(epoch_base) => {
            shared
                .purge_epoch_variants(epoch_base, &target, cutoff)
                .await
                .removed
        }
        None => 0,
    };
    let purge = crate::s3store::SharedPurge {
        active,
        tombstone,
        disk,
        event,
        log_target: log_target.to_string(),
    };
    let Some((mut lease, purge)) = shared.lead_or_join(purge) else {
        return removed;
    };
    let (inline, pending) = shared
        .purge_bounded(&target, None, cutoff, SHARED_PURGE_INLINE_SCAN)
        .await;
    removed += inline.removed;
    let shared = shared.clone();
    let tasks = background.clone();
    match pending {
        Some(pending) => {
            background.spawn_walk(async move {
                let rest = {
                    let _turn = shared.walk_turn().await;
                    shared.finish_purge(pending).await
                };
                let (event, log_target) = (purge.event, purge.log_target.clone());
                end_purge_apart(&shared, &tasks, purge, inline.complete && rest.complete).await;
                purge_log(event, &log_target, Some(rest.removed));
                let joined = lease.take_joined();
                walk_for_joined(&shared, &tasks, lease, joined).await;
            });
        }
        None => {
            end_purge_apart(&shared, background, purge, inline.complete).await;
            let joined = lease.take_joined();
            if !joined.is_empty() {
                background.spawn_walk(async move {
                    walk_for_joined(&shared, &tasks, lease, joined).await;
                });
            }
        }
    }
    removed
}

/// Walk the bucket for the purges that joined a running
/// walk of their target (issue #462), one pass for all of
/// them, then again for any that joined during that pass,
/// until none is waiting. Each pass waits for a walk turn,
/// lists the bucket afresh, and judges copies by the
/// latest joined purge's start: the purges come from one
/// process, so its cutoff covers every copy the earlier
/// ones do. Each purge the pass served logs the pass's
/// count under its own target.
async fn walk_for_joined(
    shared: &Arc<crate::s3store::S3Store>,
    background: &Background,
    mut lease: crate::s3store::WalkLease,
    mut joined: Vec<crate::s3store::SharedPurge>,
) {
    while let Some(latest) = joined
        .iter()
        .max_by_key(|purge| purge.active.tombstone().cutoff.start_ms)
    {
        let target = latest.active.tombstone().target.clone();
        let cutoff = latest.active.tombstone().cutoff;
        let swept = {
            let _turn = shared.walk_turn().await;
            shared.relist(&target, cutoff).await
        };
        for purge in joined {
            let (event, log_target) = (purge.event, purge.log_target.clone());
            end_purge_apart(shared, background, purge, swept.complete).await;
            purge_log(event, &log_target, Some(swept.removed));
        }
        joined = lease.take_joined();
    }
}

/// End a purge whose shared walk is done (issue #461):
/// rewrite its bucket tombstone with the walk's end, from
/// which every replica counts the grace period down and
/// then collects it, and end the purge in this process,
/// where its tombstone likewise outlives it for the grace.
/// A tombstone that is never marked (the process stopped
/// mid-walk) stays in force until a later purge of the same
/// target finishes its walk. So does the tombstone of a
/// walk that was not `complete` (issue #462): the store
/// failed to list the bucket or to answer for an object, so
/// a copy the purge covers may still be there, and it must
/// stay a miss after the grace. The tombstone is marked
/// ended only once the purge's disk files are deleted (issue
/// #572): a restart before then indexes them again, and the
/// unended tombstone keeps them misses until a later purge of
/// the target finishes. The wait for them is bounded: when a
/// deletion fails, or they stop making progress, the purge
/// ends with its tombstone in force, as for a walk that was
/// not complete.
async fn end_purge(
    shared: &crate::s3store::S3Store,
    purge: crate::s3store::SharedPurge,
    complete: bool,
) {
    let deleted = match &purge.disk {
        Some(disk) => disk.finished(PURGE_END_IDLE, PURGE_END_CAP).await,
        None => true,
    };
    if complete
        && deleted
        && let Some(object) = &purge.tombstone
    {
        let mut ended = purge.active.tombstone().clone();
        ended.ended_ms = Some(crate::cache::now_ms());
        shared.write_tombstone(object, &ended).await;
    }
    drop(purge);
}

/// [`end_purge`], on a task of its own when the purge still
/// waits for disk deletions (issue #572): the walk lease, and
/// the purges that joined it, do not wait on the disk.
async fn end_purge_apart(
    shared: &Arc<crate::s3store::S3Store>,
    background: &Background,
    purge: crate::s3store::SharedPurge,
    complete: bool,
) {
    if purge.disk.as_ref().is_some_and(|disk| disk.pending()) {
        let shared = shared.clone();
        background.spawn_walk(async move { end_purge(&shared, purge, complete).await });
    } else {
        end_purge(shared, purge, complete).await;
    }
}

/// A purge that has begun, as its shared-tier half takes it:
/// the running purge, its bucket tombstone, and the disk
/// deletions its tombstone waits for (issue #572).
struct BegunPurge {
    active: crate::cache::ActivePurge,
    tombstone: Option<String>,
    disk: Option<crate::storage::DeletionWait>,
}

/// The disk tier's half of a purge (issue #572): remove the
/// records `target` covers from the index at once, so they
/// stop serving and the count is exact, and queue their
/// files' deletion. No disk slot is waited for. Returns the
/// count and the deletions queued.
fn purge_disk_tier(
    state: &AppState,
    target: &crate::cache::PurgeTarget,
) -> (usize, Option<crate::storage::DeletionWait>) {
    let Some(store) = &state.store else {
        return (0, None);
    };
    let (removed, seqs) = store.purge(target);
    store.kick();
    (
        removed,
        Some(crate::storage::DeletionWait {
            store: store.clone(),
            seqs,
        }),
    )
}

/// Wait for a purge's disk deletions (issue #572) while they
/// make progress: on a healthy disk with no deletions queued
/// before them, the files are gone when the purge answers, as
/// before. Once none finishes for [`DISK_WRITE_WAIT`], or after
/// [`PURGE_DELETE_CAP`] in all, the purge answers and the
/// deletions go on in the background. That counts as a stall
/// only on a stuck disk, not on one busy with the deletions
/// queued before these.
async fn wait_for_disk_purge(disk: Option<crate::storage::DeletionWait>) {
    let Some(disk) = disk else {
        return;
    };
    let waited = disk
        .store
        .wait_deletions(disk.seqs.clone(), DISK_WRITE_WAIT, PURGE_DELETE_CAP)
        .await;
    if waited == crate::storage::Waited::Stalled {
        disk.store.note_stall(crate::storage::Stall::PurgeDelete);
    }
}

/// Load the shared bucket's purge tombstones before serving
/// (issue #461), then keep them fresh every
/// `TOMBSTONE_REFRESH` on one background task. Without a
/// shared tier there is nothing to read. Each refresh, the
/// boot one included, ends within
/// [`crate::s3store::TOMBSTONE_REFRESH_DEADLINE`] (issue
/// #549), so a slow or hanging store delays the first
/// request by that much at most. A failed or unfinished
/// boot read is counted and logged, and the refresher
/// retries: the tier stays best-effort, like every other
/// shared read. Serving does not start first: lookups
/// consult the tombstones, so a request served before them
/// could hand out a copy a purge removed.
pub async fn start_purge_tombstones(state: &AppState) {
    let Some(shared) = state.shared_store.clone() else {
        return;
    };
    shared.refresh_tombstones(&state.purge_guard).await;
    let guard = state.purge_guard.clone();
    // One task per process, started at boot and never
    // ending, so it stays off the shutdown drain (issue
    // #463), which waits only for work that finishes; it
    // stops with the process.
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(TOMBSTONE_REFRESH);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tick.tick().await;
        loop {
            tick.tick().await;
            shared.refresh_tombstones(&guard).await;
        }
    });
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
/// purge and an image signature does not verify here (no
/// signer signs the subkey's label); a bad signature is a
/// 403 and purges nothing. Idempotent:
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
    // purge silently removes nothing. The form is bounded
    // the way `cache_key_for` bounds it, so an over-long
    // URL's purge names its digest-form key.
    let key_form = match parse_operations_with_expiry(target) {
        Ok((ops, source, _, _)) => {
            let merged = crate::url::merge_default_ops(&state.default_ops, ops);
            crate::cache::bounded_key_base(crate::url::canonical_form(&merged, source), source)
        }
        Err(_) => crate::cache::bounded_key_base(target.to_string(), ""),
    };
    // The fan-out runs in a spawned task once the
    // signature has verified (issue #347): a purge that
    // has started completes and is logged even if the
    // caller disconnects mid-purge; dropping the
    // JoinHandle detaches the task instead of cancelling
    // it. The caller waits for the count, so the response
    // still carries it. The task is tracked (issue #463),
    // so a shutdown waits for a purge whose caller left.
    let task_state = state.clone();
    let task_target = target.to_string();
    let task = state.background.spawn(async move {
        // The tombstones land before any tier is touched
        // (issues #350, #461): a write begun before this
        // purge can then see it whenever it tries to land,
        // including after this handler has finished, and
        // every replica treats the copies it covers as
        // purged until the shared walk ends (issue #400).
        let (active, tombstone) = begin_purge(
            &task_state,
            crate::cache::PurgeTarget::Key(key_form.clone()),
        )
        .await;
        let mut removed = task_state.cache.purge_matching(&key_form);
        let (on_disk, disk) = purge_disk_tier(
            &task_state,
            &crate::cache::PurgeTarget::Key(key_form.clone()),
        );
        removed += on_disk;
        if let Some(shared) = &task_state.shared_store {
            // The shared tier deletes the current epoch's
            // enumerable variants directly by object name
            // (audit R2-24); the fingerprint-free form drives
            // its matching walk, which also reaches older
            // epochs and hint-variant spellings.
            let epoch_form = format!("{key_form}#cfg{}", task_state.config_fingerprint);
            removed += purge_shared_tier(
                shared,
                &task_state.background,
                BegunPurge {
                    active,
                    tombstone,
                    disk: disk.clone(),
                },
                Some(epoch_form),
                "purge",
                &task_target,
            )
            .await;
        }
        wait_for_disk_purge(disk).await;
        task_state.metrics.record_purge();
        purge_log("purge", &task_target, Some(removed));
        removed
    });
    let removed = task.await.unwrap_or(0);
    (StatusCode::OK, format!("purged {removed}\n")).into_response()
}

/// `POST /purge-group/<signature>/<group>`: invalidate every
/// stored entry in one cache group (RFC 9875), across this
/// process's tiers (its memory, its disk store, and the
/// shared store); another process's memory is not reached,
/// as in the per-URL purge (issue #351). The signature is
/// verified under the purge-group
/// subkey, its own domain beside the purge subkey, over the
/// group name, so a per-URL purge signature does not verify
/// here, and neither does an image signature (no signer
/// signs the subkey's label). A bad signature is a 403 and
/// removes nothing; an unknown group answers 200 with a
/// count of 0.
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
    let task = state.background.spawn(async move {
        // Tombstones first, tiers second (issues #350,
        // #461), as in the per-URL purge.
        let (active, tombstone) =
            begin_purge(&task_state, crate::cache::PurgeTarget::Group(group.clone())).await;
        let mut removed = task_state.cache.purge_group(&group);
        let (on_disk, disk) = purge_disk_tier(
            &task_state,
            &crate::cache::PurgeTarget::Group(group.clone()),
        );
        removed += on_disk;
        if let Some(shared) = &task_state.shared_store {
            removed += purge_shared_tier(
                shared,
                &task_state.background,
                BegunPurge {
                    active,
                    tombstone,
                    disk: disk.clone(),
                },
                None,
                "purge-group",
                &task_target,
            )
            .await;
        }
        wait_for_disk_purge(disk).await;
        task_state.metrics.record_purge();
        purge_log("purge-group", &task_target, Some(removed));
        removed
    });
    let removed = task.await.unwrap_or(0);
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
    let task = state.background.spawn(async move {
        // Tombstones first, tiers second (issues #350,
        // #461), as in the per-URL purge.
        let (active, tombstone) = begin_purge(
            &task_state,
            crate::cache::PurgeTarget::Source(source.clone()),
        )
        .await;
        let mut removed = task_state.cache.purge_source(&source);
        let (on_disk, disk) = purge_disk_tier(
            &task_state,
            &crate::cache::PurgeTarget::Source(source.clone()),
        );
        removed += on_disk;
        if let Some(shared) = &task_state.shared_store {
            removed += purge_shared_tier(
                shared,
                &task_state.background,
                BegunPurge {
                    active,
                    tombstone,
                    disk: disk.clone(),
                },
                None,
                "purge-source",
                &source,
            )
            .await;
        }
        wait_for_disk_purge(disk).await;
        purge_log("purge-source", &source, Some(removed));
        removed
    });
    task.await.unwrap_or(0)
}

/// `PUT /upload/<signature>/<name>`: store one original
/// (issue #168). The route is always registered; without
/// a destination every request answers
/// `404 uploads_not_configured` before anything else is
/// checked (issue #467).
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
/// On success the body of the response is the source
/// reference to embed in transform URLs plus a newline,
/// and the `X-Crabcrop-Source` header is the exact
/// reference: `201` for a new object, `200` when an
/// allowed overwrite replaced one.
async fn upload_handle(State(state): State<AppState>, req: axum::extract::Request) -> Response {
    let accept = req
        .headers()
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let (parts, body) = req.into_parts();
    // A buffered upload is a source like a fetched one
    // (issue #403): its body is charged to the source buffer
    // budget as it arrives and the charge is held until the
    // handler is done with the bytes, so a burst of uploads
    // waits at the budget instead of buffering past it, and
    // a client that stalls mid-body holds only what it sent.
    let charge = state.loader.source_budget().charge(SOURCE_BUDGET_WAIT);
    let outcome = match receive_upload(&state, parts.uri.path(), body, &charge).await {
        Ok(outcome) | Err(outcome) => outcome,
    };
    finish_upload(&state, outcome, accept.as_deref()).await
}

/// How an upload request ended (issue #479). The stages of
/// [`receive_upload`] return every refusal as one of these,
/// and [`finish_upload`] answers it, so the count on
/// `/metrics`, the audit line (audit PL-8), and the answer
/// are made in one place.
enum UploadOutcome {
    /// Refused before the signature check (no destination,
    /// a bad path, a sealed spelling, an unparsable
    /// signature), or for a verified remainder that does
    /// not split into an expiry and a name: answered, and
    /// neither counted nor logged.
    Unaudited(ErrorParts),
    /// Refused from the signature check on: counted, and
    /// logged under `target` (the remainder as presented
    /// for a bad signature, the name after it) with its
    /// `rejected:` outcome.
    Rejected {
        target: String,
        outcome: &'static str,
        answer: ErrorParts,
    },
    /// Shed because the source buffer budget or the process
    /// gate stayed full past its wait: counted and logged as
    /// `rejected:busy`, and answered with a 503 carrying
    /// `message`.
    Busy {
        target: String,
        message: &'static str,
    },
    /// Stored under `name`, at the source reference
    /// `source`; `replaced` when an allowed overwrite
    /// replaced an object.
    Stored {
        name: String,
        source: String,
        replaced: bool,
    },
}

impl UploadOutcome {
    fn rejected(target: &str, outcome: &'static str, answer: ErrorParts) -> Self {
        UploadOutcome::Rejected {
            target: target.to_string(),
            outcome,
            answer,
        }
    }
}

/// Count and log an upload's outcome (audit PL-8), then
/// answer it. A stored upload that replaced an object
/// purges the old one's derivatives before it answers.
async fn finish_upload(state: &AppState, outcome: UploadOutcome, accept: Option<&str>) -> Response {
    match outcome {
        UploadOutcome::Unaudited((status, code, message)) => {
            error_response(status, code, message, accept)
        }
        UploadOutcome::Rejected {
            target,
            outcome,
            answer: (status, code, message),
        } => {
            state.metrics.record_upload();
            upload_log(&target, outcome);
            error_response(status, code, message, accept)
        }
        UploadOutcome::Busy { target, message } => {
            state.metrics.record_upload();
            upload_log(&target, "rejected:busy");
            busy_response(message, 1, accept)
        }
        UploadOutcome::Stored {
            name,
            source,
            replaced,
        } => {
            state.metrics.record_upload();
            upload_log(&name, if replaced { "replaced" } else { "stored" });
            if replaced {
                // An overwrite replaces the source every
                // cached derivative was rendered from: purge
                // them all, in every tier, so the next
                // request re-renders from the new original
                // instead of serving the stale transform
                // (audit R2-07). A first store has no
                // derivatives to invalidate.
                // Logged inside the purge itself, so the
                // line lands even if this request is dropped
                // while the purge runs (issue #347).
                let _removed = purge_source_everywhere(state, &source).await;
            }
            let status = if replaced {
                StatusCode::OK
            } else {
                StatusCode::CREATED
            };
            let mut response = (status, format!("{source}\n")).into_response();
            response.headers_mut().insert(
                axum::http::HeaderName::from_static("x-crabcrop-source"),
                axum::http::HeaderValue::from_str(&source).unwrap(),
            );
            response
        }
    }
}

/// The stages of an upload (issue #479), cheapest first:
/// [`authorize_upload`], [`read_upload`], the overwrite
/// policy (so a refused duplicate never pays a decode),
/// [`validate_upload`], and [`store_upload`]. `Ok` is
/// always [`UploadOutcome::Stored`]; a refusal ends the
/// chain as its [`UploadOutcome`] in `Err`.
async fn receive_upload(
    state: &AppState,
    path: &str,
    body: axum::body::Body,
    charge: &crate::loader::SourceCharge,
) -> Result<UploadOutcome, UploadOutcome> {
    let (upload, name) = authorize_upload(state, path)?;
    let name = name.as_str();
    let bytes = read_upload(body, upload.max_bytes, charge, name).await?;
    // The existence check runs before validation, so an
    // upload that will be refused as a duplicate never
    // pays a full decode first (audit R2-06). The value
    // is reused after the store to pick 200 vs 201 and
    // the log outcome.
    let existed = upload.exists(name).await;
    if existed && !upload.allow_overwrite {
        return Err(upload_exists(name));
    }
    let prepared = validate_upload(state, bytes, name).await?;
    store_upload(upload, name, prepared, existed).await
}

/// The upload's first stage: a configured destination, a
/// well-formed path, a verified signature, an unexpired
/// token, and a well-formed name. Returns the destination
/// and the name.
fn authorize_upload<'s>(
    state: &'s AppState,
    path: &str,
) -> Result<(&'s crate::upload::UploadStore, String), UploadOutcome> {
    let Some(upload) = &state.upload else {
        return Err(UploadOutcome::Unaudited((
            StatusCode::NOT_FOUND,
            "uploads_not_configured",
            "uploads are not configured".to_string(),
        )));
    };
    let Some(tail) = path.strip_prefix("/upload/") else {
        return Err(UploadOutcome::Unaudited((
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /upload/<signature>/<name>".to_string(),
        )));
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
        return Err(UploadOutcome::Unaudited((
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature".to_string(),
        )));
    }
    let (signature, key_id, rest) = match parse_signature(&full) {
        Ok(v) => v,
        Err(e) => {
            return Err(UploadOutcome::Unaudited((
                StatusCode::BAD_REQUEST,
                e.code(),
                e.to_string(),
            )));
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
        return Err(UploadOutcome::rejected(
            rest,
            "rejected:bad-signature",
            (
                StatusCode::FORBIDDEN,
                "bad_signature",
                "bad signature".to_string(),
            ),
        ));
    }
    let (expiry, name) = match crate::url::split_upload_remainder(rest) {
        Ok(v) => v,
        Err(e) => {
            return Err(UploadOutcome::Unaudited((
                StatusCode::BAD_REQUEST,
                e.code(),
                e.to_string(),
            )));
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
            return Err(UploadOutcome::rejected(
                name,
                "rejected:expired",
                (
                    StatusCode::FORBIDDEN,
                    "signature_expired",
                    "signature expired".to_string(),
                ),
            ));
        }
    }
    if let Err(e) = crate::upload::validate_upload_name(name) {
        return Err(UploadOutcome::rejected(
            name,
            "rejected:bad-name",
            (StatusCode::BAD_REQUEST, "invalid_upload_name", e),
        ));
    }
    Ok((upload, name.to_string()))
}

/// Read the upload's body, charged to `charge` as it
/// arrives (issue #403). An empty body is refused here: an
/// image is required.
async fn read_upload(
    body: axum::body::Body,
    max_bytes: u64,
    charge: &crate::loader::SourceCharge,
    name: &str,
) -> Result<Vec<u8>, UploadOutcome> {
    // The read is bounded by the cap itself: it fails the
    // moment the body passes it, which is the oversize
    // answer (413) without ever holding more than the cap in
    // memory. A body that cannot be read to its end (the
    // client aborted it, or framed it badly) is the
    // client's fault but not a size one: a 400 (issue #467).
    let cap = usize::try_from(max_bytes).unwrap_or(usize::MAX);
    let bytes = match read_upload_body(body, cap, charge).await {
        Ok(bytes) => bytes,
        Err(LoadError::Busy) => {
            return Err(UploadOutcome::Busy {
                target: name.to_string(),
                message: SOURCE_BUDGET_BUSY,
            });
        }
        Err(LoadError::TooLarge { .. }) => {
            return Err(UploadOutcome::rejected(
                name,
                "rejected:oversize",
                (
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "upload_too_large",
                    format!("upload exceeds the {max_bytes}-byte limit"),
                ),
            ));
        }
        Err(e) => {
            eprintln!("upload body read failed: {e}");
            return Err(UploadOutcome::rejected(
                name,
                "rejected:unreadable",
                (
                    StatusCode::BAD_REQUEST,
                    "upload_unreadable",
                    "upload body could not be read".to_string(),
                ),
            ));
        }
    };
    if bytes.is_empty() {
        return Err(UploadOutcome::rejected(
            name,
            "rejected:not-an-image",
            (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "upload_not_an_image",
                "upload body is empty: an image is required".to_string(),
            ),
        ));
    }
    Ok(bytes)
}

/// The 409 for a name already taken while overwriting is
/// off: from the cheap existence check, or from the
/// conditional create that settles a race (audit R2-21).
fn upload_exists(name: &str) -> UploadOutcome {
    UploadOutcome::rejected(
        name,
        "rejected:exists",
        (
            StatusCode::CONFLICT,
            "upload_already_exists",
            "an object with this name already exists".to_string(),
        ),
    )
}

/// The bytes must decode as an image this server can
/// serve, through the pipeline's own decode branches:
/// anything else is refused before it is stored, so the
/// destination never holds an unservable object. The
/// decode runs under the process gate, like every
/// transform (see `run_upload_validation`). Then the name
/// must present the body as what it is. Returns the
/// prepared form to store.
async fn validate_upload(
    state: &AppState,
    bytes: Vec<u8>,
    name: &str,
) -> Result<crate::engine::PreparedUpload, UploadOutcome> {
    // The body moves into validation rather than being
    // copied for it, so the charge covers what is held.
    let prepared = match run_upload_validation(state, bytes).await {
        UploadValidation::Valid(prepared) => prepared,
        UploadValidation::Invalid(e) => {
            eprintln!("upload rejected: {e}");
            return Err(UploadOutcome::rejected(
                name,
                "rejected:not-an-image",
                (
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "upload_not_an_image",
                    "upload is not an image this server can decode".to_string(),
                ),
            ));
        }
        UploadValidation::Busy => {
            return Err(UploadOutcome::Busy {
                target: name.to_string(),
                message: "server busy: too many images processing at once",
            });
        }
        UploadValidation::Panicked => {
            return Err(UploadOutcome::rejected(
                name,
                "rejected:validation-failed",
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "validation_failed",
                    "image validation failed".to_string(),
                ),
            ));
        }
    };
    // The name must present the body as what it is: a
    // validated image may not be stored under an
    // extension belonging to another kind (or to active
    // content), because the destination may be served
    // directly (audit R2-22).
    if let Err(e) = crate::upload::check_upload_media(name, prepared.media) {
        return Err(UploadOutcome::rejected(
            name,
            "rejected:media-mismatch",
            (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "upload_media_mismatch",
                e,
            ),
        ));
    }
    Ok(prepared)
}

/// Write the prepared upload to the destination: the
/// stored outcome, or a refusal (a name claimed by a
/// racing upload, a failed write).
async fn store_upload(
    upload: &crate::upload::UploadStore,
    name: &str,
    prepared: crate::engine::PreparedUpload,
    existed: bool,
) -> Result<UploadOutcome, UploadOutcome> {
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
        Ok(()) => Ok(UploadOutcome::Stored {
            name: name.to_string(),
            source: upload.source_ref(name),
            replaced: existed,
        }),
        Err(crate::upload::UploadError::AlreadyExists) => Err(upload_exists(name)),
        Err(crate::upload::UploadError::Backend(cause)) => {
            eprintln!("upload failed: {cause}");
            Err(UploadOutcome::rejected(
                name,
                "rejected:store-failed",
                (
                    StatusCode::BAD_GATEWAY,
                    "upload_store_unavailable",
                    "the upload store is unavailable".to_string(),
                ),
            ))
        }
        Err(crate::upload::UploadError::Io(cause)) => {
            eprintln!("upload failed: {cause}");
            Err(UploadOutcome::rejected(
                name,
                "rejected:store-failed",
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "upload_store_failed",
                    "the upload could not be stored".to_string(),
                ),
            ))
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
/// A sealed request gets no Cache-Group (issue #454): its
/// groups are preset names read from inside the token, and
/// listing them would show anyone who fetches the URL part
/// of the recipe the token hides. Group purges still reach
/// its entry, because the groups are recorded on the cache
/// entry, never read back from this header.
fn attach_cache_headers(
    response: &mut Response,
    cache_status: &str,
    groups: &[String],
    sealed: bool,
) {
    response.headers_mut().insert(
        axum::http::HeaderName::from_static("cache-status"),
        axum::http::HeaderValue::from_str(cache_status).unwrap(),
    );
    if !sealed && !groups.is_empty() {
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

/// The request ID a client chose with `X-Request-Id`, when
/// it is 1 to 128 letters, digits, `-` or `_`.
fn client_request_id(inbound: Option<&str>) -> Option<String> {
    let value = inbound?.trim();
    ((1..=128).contains(&value.len())
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'))
    .then(|| value.to_string())
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
    let chosen_id = client_request_id(headers.get("x-request-id").and_then(|v| v.to_str().ok()));
    let request_id = chosen_id.clone().unwrap_or_else(generate_request_id);
    let if_none_match = headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    // With client hints off, no hint header is read at all:
    // every request for a URL gets the same bytes, the one
    // answer a cache that keys on the URL alone can hold.
    let hints = state.client_hints;
    let dpr = dpr_hint(&headers).filter(|_| hints);
    let width = width_hint(&headers).filter(|_| hints);
    let save_data = hints && save_data_hint(&headers);
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
    // Every answer on this route, errors and meta included,
    // names the request headers it may depend on.
    response
        .headers_mut()
        .insert(header::VARY, image_vary(state.client_hints));
    // An answer a shared cache may store reaches every later
    // viewer of the URL with the headers it was stored with,
    // so it never carries an ID the client chose: that
    // client already has it, and the log line records it.
    // An ID the server generated is still sent.
    let shared = response
        .headers()
        .get(header::CACHE_CONTROL)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| has_directive(v, "public"));
    if chosen_id.is_none() || !shared {
        response.headers_mut().insert(
            axum::http::HeaderName::from_static("x-request-id"),
            request_id.parse().unwrap(),
        );
    }
    state.metrics.record(&record);
    println!("{}", record.to_json());
    response
}

/// Why the image path refused a request before its cache
/// lookup (issue #479). The stages of [`serve`] return
/// every refusal as one of these, and `serve` answers it
/// in one place, [`Refusal::answer`], so a refusal added to
/// any stage keeps the sealed spelling's uniform 403
/// (issues #255, #259) without a branch of its own.
enum Refusal {
    /// A segment in the sealed spelling that does not
    /// open: the uniform refusal, whatever the request.
    Unopened,
    /// A failure of what the remainder asks for: its own
    /// answer for a plain request, the uniform refusal for
    /// a sealed one, so an opened token is never a
    /// validity oracle.
    Payload(ErrorParts),
    /// A request for meta with meta mode off (issues #390,
    /// #453): a payload refusal whose answer, either one,
    /// carries the meta document's caching headers.
    MetaDisabled(ErrorParts),
    /// An answer both spellings give as is: the plain
    /// spelling's own signature failures, which a sealed
    /// request never reaches, and an expired URL, which an
    /// opened token answers exactly as the plain form does
    /// (`signature_expired`).
    AsIs(ErrorParts),
}

impl Refusal {
    /// The answer to this refusal, for a request in the
    /// sealed spelling or not. A sealed payload refusal
    /// answers the uniform refusal without building its
    /// own answer first.
    fn answer(self, sealed: bool, accept: Option<&str>) -> Response {
        let own =
            |(status, code, message): ErrorParts| error_response(status, code, message, accept);
        match self {
            Refusal::Unopened => sealed_refusal_response(),
            Refusal::Payload(_) if sealed => sealed_refusal_response(),
            Refusal::MetaDisabled(_) if sealed => meta_cache_headers(sealed_refusal_response()),
            Refusal::Payload(parts) | Refusal::AsIs(parts) => own(parts),
            Refusal::MetaDisabled(parts) => meta_cache_headers(own(parts)),
        }
    }
}

/// Run one image request, timing each stage into the returned record.
///
/// The request runs in stages (issue #479):
/// [`authenticate`], [`resolve_request`] and [`negotiate`]
/// admit it, each returning a [`Refusal`] that is answered
/// here, the one place that knows the request's spelling;
/// [`cache_key_for`] keys it; and [`run_flight`] serves it
/// from the tiers or the miss path.
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
    let opened = authenticate(state, &full, &mut record);
    let sealed = opened.as_ref().is_ok_and(|opened| opened.sealed);
    let admitted = opened.and_then(|opened| {
        let resolved = resolve_request(state, &opened.remainder, &mut record)?;
        let negotiated = negotiate(
            state,
            &resolved.ops,
            accept,
            dpr_hint,
            width_hint,
            save_data,
        )?;
        Ok((resolved, negotiated))
    });
    let (resolved, negotiated) = match admitted {
        Ok(admitted) => admitted,
        Err(refusal) => return (refusal.answer(sealed, accept), record),
    };
    let hint_shaped = negotiated.hint_shaped();
    let cache_key = cache_key_for(
        &state.config_fingerprint,
        &resolved.ops,
        &resolved.source_url,
        &negotiated,
    );
    // The key fully determines the lookup: a stored response can
    // be served without touching the loader or the engine. Meta
    // requests bypass the cache in both directions: they must
    // never be answered with image bytes, and their JSON is never
    // stored as an image entry. Debug requests bypass it too:
    // the diagnostic must never be served from, or stored into,
    // any tier (its key differs from the plain URL's anyway, but
    // the bypass is the rule, not the key's accident).
    let bypass = negotiated.meta || negotiated.debug;
    if !bypass {
        record.cache_looked_up = true;
    }
    let work = MissWork {
        ops: resolved.ops,
        source_url: resolved.source_url,
        directives: negotiated.directives,
        preferred: negotiated.preferred,
        meta: negotiated.meta,
        debug: negotiated.debug,
        placeholders: negotiated.placeholders,
        hinted_width: negotiated.hinted_width,
        dpr_hint,
        // The engine steps on the same decision the key and
        // the private marking take, never on the raw header:
        // a render the header stepped where the key does not
        // separate it would be stored, and served publicly,
        // as the URL's plain answer.
        save_data: negotiated.save_data_applies,
        accept: accept.map(str::to_string),
        groups: resolved.groups,
        cache_key,
        max_age: resolved.max_age,
        sealed,
        progress: MissProgress::default(),
        disk_handoff: None,
    };
    let (mut response, record) = run_flight(state, path, work, if_none_match, record).await;
    if hint_shaped {
        keep_from_shared_caches(&mut response);
    }
    (response, record)
}

/// Make a public answer private, keeping its lifetime: an
/// answer a client hint gives its own cache entry (see
/// [`Negotiated::hint_shaped`]).
///
/// The hint headers are request headers any client sets,
/// and many shared caches (CDNs among them) key on the URL
/// and not on `Vary`, so a hint-shaped answer stored there
/// would be the answer every later viewer of the URL gets,
/// whatever their own headers say. A private answer is
/// still kept by the browser that asked, for its whole
/// lifetime, and the URL's hint-free answer stays public:
/// it is the answer a cache that keys on the URL alone
/// should hold. Answers that are not public (meta, errors,
/// the fallback, the debug view) are left as they are.
fn keep_from_shared_caches(response: &mut Response) {
    let Some(rest) = response
        .headers()
        .get(header::CACHE_CONTROL)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("public"))
    else {
        return;
    };
    let private = format!("private{rest}");
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, private.parse().unwrap());
}

/// A remainder the request may run.
struct Opened<'a> {
    /// The remainder as presented, or as its sealed token
    /// decrypts.
    remainder: std::borrow::Cow<'a, str>,
    /// Whether the request arrived in the sealed spelling
    /// (issue #255).
    sealed: bool,
}

/// The first stage of [`serve`]: verify the signature over
/// the remainder, or open the sealed token in its slot,
/// timing either into `record.verify`.
///
/// A sealed spelling (issue #255) is intercepted before
/// the plain parse: the token sits in the signature slot,
/// and everything about it (an oversized segment, a
/// failed open, and every payload failure after a
/// successful open) answers with the one uniform refusal,
/// so a token is never a validity oracle. On a successful
/// open the decrypted remainder takes the place of the
/// presented one and flows through the exact pipeline a
/// verified plain remainder flows through: because
/// decryption produces the pre-expansion string the HMAC
/// form carries, the two spellings share one cache entry,
/// one ETag, one purge target by construction.
fn authenticate<'a>(
    state: &AppState,
    full: &'a str,
    record: &mut RequestRecord,
) -> Result<Opened<'a>, Refusal> {
    let nolead = full.strip_prefix('/').unwrap_or(full);
    let first_segment = nolead.split('/').next().unwrap_or("");
    if crate::url::looks_sealed(first_segment)
        && crate::url::split_sealed_segment(first_segment).is_none()
    {
        // Marker-shaped but not a token this scheme opens
        // (an unknown version): the uniform refusal, never
        // the grammar's 400, so the answer cannot announce
        // that the segment was not a token at all.
        return Err(Refusal::Unopened);
    }
    if let Some((key_id, token)) = crate::url::split_sealed_segment(first_segment) {
        if first_segment.len() > crabcrop_url::MAX_SEALED_SEGMENT {
            return Err(Refusal::Unopened);
        }
        let verify_start = Instant::now();
        let opened = state.keyring.open_sealed(key_id, token);
        record.verify = verify_start.elapsed();
        // For a sealed request the open is the
        // verification the plain spelling runs below.
        return match opened {
            Some(text) => Ok(Opened {
                remainder: std::borrow::Cow::Owned(text),
                sealed: true,
            }),
            None => Err(Refusal::Unopened),
        };
    }
    // Grammar failures answer 400 with the parser's own
    // message as the body: it names only what the caller
    // sent, and it is the difference between a debuggable
    // rejection and an empty one.
    let (signature, key_id, rest) =
        parse_signature(full).map_err(|e| Refusal::AsIs(parse_failure_parts(&e)))?;
    // A remainder that begins with the reserved segment is
    // refused before its signature is checked, so the answer
    // is the same whatever the signature: no response depends
    // on an HMAC over a key derivation label.
    if crabcrop_url::is_reserved_remainder(rest) {
        return Err(Refusal::AsIs(reserved_remainder_parts()));
    }
    let verify_start = Instant::now();
    let verified = state
        .keyring
        .verify(key_id.as_deref(), &signature, rest, state.allow_unsafe);
    record.verify = verify_start.elapsed();
    if !verified {
        return Err(Refusal::AsIs((
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature".to_string(),
        )));
    }
    Ok(Opened {
        remainder: std::borrow::Cow::Borrowed(rest),
        sealed: false,
    })
}

/// What a verified remainder resolves to.
struct Resolved {
    /// The operations the engine runs: the remainder's
    /// own, with the server-wide defaults merged in.
    ops: Vec<crate::url::Operation>,
    /// The source to fetch, a root source expanded.
    source_url: String,
    /// The cache groups (RFC 9875).
    groups: Vec<String>,
    /// The `maxage:` cache lifetime (issue #159).
    max_age: Option<u32>,
}

/// The second stage of [`serve`]: expand and parse the
/// remainder, recording its operation summary, and refuse
/// what it may not ask for: a reserved segment, a preset
/// or grammar failure, too many operations, an unknown
/// source root, or an expired URL, in that order.
fn resolve_request(
    state: &AppState,
    rest: &str,
    record: &mut RequestRecord,
) -> Result<Resolved, Refusal> {
    let grammar = |e: crate::url::ParseError| Refusal::Payload(parse_failure_parts(&e));
    // Defense in depth for the purge domain separation (audit
    // P-1): an image remainder may not begin with the `purge`
    // segment, so the two URL shapes can never be confused.
    // The upload segment (issue #168) is reserved the same
    // way, for the same reason.
    if matches!(
        rest.split('/').next(),
        Some("purge" | "purge-group" | "upload")
    ) {
        return Err(Refusal::Payload((
            StatusCode::BAD_REQUEST,
            "reserved_segment",
            "an image URL may not begin with a purge or upload segment".to_string(),
        )));
    }
    // A plain remainder that begins with the key derivation
    // segment never gets this far (`authenticate` refuses it
    // before verification); an opened token's can, and its
    // answer is the uniform refusal like any payload failure.
    if crabcrop_url::is_reserved_remainder(rest) {
        return Err(Refusal::Payload(reserved_remainder_parts()));
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
    let rest =
        crate::url::expand_presets(rest, &state.presets, state.only_presets).map_err(grammar)?;
    let (ops, source_url, expiry, max_age) =
        parse_operations_with_expiry(&rest).map_err(grammar)?;
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
    crate::url::check_operation_count(&ops).map_err(grammar)?;
    // So does the cap on distinct watermark and overlay URLs
    // (issue #613): each one is a fetch and a decode.
    crate::url::check_watermark_count(&ops).map_err(grammar)?;
    // A root source expands here (issue #248): after
    // verification and the ops summary (which reads the
    // source as written), before the cache key and the
    // fetch, so the short and full spellings of one source
    // are one render, sharing a cache entry and a purge
    // target. The loader's checks evaluate the expanded
    // URL unchanged, the allowlist included.
    let (source_url, _) =
        crate::url::expand_source_root(source_url, &state.source_roots).map_err(grammar)?;
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
            return Err(Refusal::AsIs((
                StatusCode::FORBIDDEN,
                "signature_expired",
                "signature expired".to_string(),
            )));
        }
    }
    Ok(Resolved {
        ops,
        source_url,
        groups,
        max_age,
    })
}

/// How the request's directives and headers shape its
/// response.
struct Negotiated {
    directives: crate::url::RequestDirectives,
    /// The format negotiated from `Accept`, when the URL
    /// names none.
    preferred: Option<ImageFormat>,
    /// The DPR factor the response is rendered at.
    effective_dpr: f32,
    /// The quantized width a width hint resolves to.
    hinted_width: Option<u32>,
    /// Whether `Save-Data` applies: the engine steps the
    /// quality of a format that carries one, and the key
    /// separates the entry.
    save_data_applies: bool,
    meta: bool,
    debug: bool,
    placeholders: bool,
}

impl Negotiated {
    /// Whether a client hint gives the answer its own cache
    /// entry: a width hint sizes it, a DPR hint (not the
    /// URL's own `dpr`) scales it, or `Save-Data` applies.
    /// This is exactly when [`cache_key_for`] gives the key
    /// a hint's suffix, so the answer is set apart from the
    /// URL's hint-free answer by the same rule its cache
    /// entry is. It can still match that answer byte for
    /// byte (a width hint at or past the source width, a DPR
    /// hint on a URL that sizes nothing, `Save-Data` on PNG
    /// output), and is set apart all the same: whether the
    /// bytes differ is only known after the render.
    fn hint_shaped(&self) -> bool {
        self.hinted_width.is_some()
            || self.save_data_applies
            || (self.directives.dpr.is_none() && self.effective_dpr != 1.0)
    }
}

/// The third stage of [`serve`]: settle the directives,
/// the client hints, and the negotiated format, and refuse
/// a meta request this server or this output cannot
/// answer.
fn negotiate(
    state: &AppState,
    ops: &[crate::url::Operation],
    accept: Option<&str>,
    dpr_hint: Option<f32>,
    width_hint: Option<u32>,
    save_data: bool,
) -> Result<Negotiated, Refusal> {
    // One pass over the ops list settles every directive
    // the handler reads (issue #136).
    let directives = crate::url::RequestDirectives::analyze(ops);

    // The DPR client hint settles sizing when the URL has no
    // explicit `dpr` directive; the effective factor is part of
    // what the response is, so it joins the cache key whenever
    // it changes the output. A hinted factor the output limits
    // refuse steps down in the engine (issue #529), and that
    // render is stored under the hinted factor's key: the hint
    // and the URL settle the step together, so the entry
    // answers that hint alone and never the hint-less URL's.
    let effective_dpr = directives.dpr.or(dpr_hint).unwrap_or(1.0);

    // The quantized width a width hint resolves to, when the
    // ops size nothing themselves. It is part of what the
    // response is, so it joins the cache key below. A hint
    // is advisory (issue #615): when the ops it adds would
    // take the list past the operation cap the URL passed,
    // it is dropped, and the request renders and caches as
    // the hint-less URL does instead of failing in the
    // engine after the fetch.
    let hinted_width = if directives.has_size_operation {
        None
    } else {
        width_hint
            .map(|w| quantize_width(w, state.width_step))
            .filter(|&w| {
                let added = width_hint_ops(w, directives.no_upscale);
                crate::url::operation_count(ops) + crate::url::operation_count(&added)
                    <= crate::url::MAX_OPERATIONS
            })
    };
    // Save-Data applies when the hint is present and the URL
    // names no quality directive. This one decision drives
    // the engine's step (`MissWork::save_data`), the key's
    // separate entry, and the private marking, so the three
    // cannot disagree: a per-format list that names no value
    // for the output format still counts as a directive, and
    // the engine does not step it either.
    let save_data_applies = save_data && !directives.quality_directive;

    let preferred = negotiate_format(accept);
    let meta = wants_meta(accept);
    if meta && !state.meta_enabled {
        // Meta mode is switched off on this deployment
        // (issue #390), as it is by default (issue #453):
        // refuse specifically, like the metrics endpoint
        // does, rather than serving a render the caller did
        // not ask for. The refusal answers the URL's JSON
        // representation, so it carries the meta document's
        // caching headers: a cache that stored it under the
        // bare URL would serve it to every image viewer.
        return Err(Refusal::MetaDisabled((
            StatusCode::FORBIDDEN,
            "meta_disabled",
            "meta mode is disabled on this server: set CRABCROP_META_MODE=1 to enable it"
                .to_string(),
        )));
    }
    let placeholders = meta && wants_placeholders(accept);
    let wants_video = directives.explicit_format.is_some_and(|f| f.is_video());
    // Meta describes an image; a video result is not one.
    // Refuse specifically rather than describing encoded
    // video bytes as if they were a still image. The
    // message echoes the payload's own format argument, so
    // a sealed request answers the uniform refusal instead
    // (issue #255).
    if meta && wants_video {
        return Err(Refusal::Payload((
            StatusCode::BAD_REQUEST,
            "meta_not_available",
            format!(
                "meta is not available for video output: format({}) serves encoded video, not an image",
                crate::url::format_text(directives.explicit_format.unwrap())
            ),
        )));
    }
    // The debug view (issue #164) is a diagnostic computed
    // from this request's own processing: it bypasses the
    // cache in both directions (the meta precedent) and is
    // served no-store. Meta cannot describe it either: the
    // bytes are a marked-up canvas, not the transform result
    // the meta document reports on.
    let debug = directives.debug_focal;
    if meta && debug {
        return Err(Refusal::Payload((
            StatusCode::BAD_REQUEST,
            "meta_not_available",
            "meta is not available for debug output: debug:focal serves a diagnostic image, not a transform result"
                .to_string(),
        )));
    }
    Ok(Negotiated {
        directives,
        preferred,
        effective_dpr,
        hinted_width,
        save_data_applies,
        meta,
        debug,
        placeholders,
    })
}

/// The fourth stage of [`serve`]: the request's cache key.
///
/// Without an explicit format in the URL, the response depends
/// on the Accept class as well as the path, so the cache key
/// carries the class. Explicit-format URLs key on the path alone.
/// The key is the request's canonical form (Q1): the
/// parsed operations re-serialized in their one
/// spelling, so equivalent requests share an entry.
/// Expiry and the cache lifetime are not operations,
/// so neither ever enters it. A form past
/// [`crate::cache::MAX_KEY_BASE_BYTES`] is carried as its
/// digest ([`crate::cache::bounded_key_base`]), so a padded
/// source cannot make one entry's key, which every tier
/// stores and the cache maps hold twice, tens of kilobytes
/// long.
fn cache_key_for(
    config_fingerprint: &str,
    ops: &[crate::url::Operation],
    source_url: &str,
    negotiated: &Negotiated,
) -> String {
    let key_base =
        crate::cache::bounded_key_base(crate::url::canonical_form(ops, source_url), source_url);
    // The key also carries the configuration fingerprint
    // (audit R2-08): entries in the persistent tiers are
    // keyed under the configuration that produced them, so
    // a restart with different output-shaping settings
    // (metadata policy, quality tuning, width step, model
    // or font or encoder files) starts a fresh epoch
    // instead of serving the old configuration's renders.
    // The fingerprint sits inside the base, ahead of the
    // negotiated-class and hint suffixes.
    let key_base = format!("{key_base}#cfg{config_fingerprint}");
    let mut cache_key = if negotiated.directives.explicit_format.is_some() {
        key_base
    } else {
        let class = match negotiated.preferred {
            Some(ImageFormat::Avif) => "avif",
            Some(ImageFormat::Jxl) => "jxl",
            Some(ImageFormat::WebP) => "webp",
            _ => "source",
        };
        format!("{key_base}#{class}")
    };
    if let Some(w) = negotiated.hinted_width {
        cache_key = format!("{cache_key}#w{w}");
    }
    if negotiated.save_data_applies {
        cache_key = format!("{cache_key}#sd");
    }
    let effective_dpr = negotiated.effective_dpr;
    if effective_dpr != 1.0 {
        cache_key = format!("{cache_key}#dpr{effective_dpr}");
    }
    cache_key
}

/// The flight key of a meta request (issue #295): the
/// cache key plus every fact the document depends on
/// beyond it (the sealed spelling, which redacts the
/// document, the placeholder fields, and the Accept
/// opt-ins), each computed by the same helpers the miss
/// path branches on, so the key can never disagree with
/// the document a flight shares.
fn meta_flight_key(work: &MissWork) -> String {
    let cache_key = &work.cache_key;
    let accept = work.accept.as_deref();
    format!(
        "{cache_key}\u{1}meta\u{1}{}{}{}{}{}{}",
        work.sealed as u8,
        work.placeholders as u8,
        wants_detections(accept) as u8,
        wants_moderation(accept) as u8,
        wants_embeddings(accept) as u8,
        wants_ocr(accept) as u8,
    )
}

/// The last stage of [`serve`]: serve the keyed request.
/// A debug request runs the miss path on its own, a meta
/// request coalesces through its own flights, and an image
/// request reads the tiers and coalesces its miss.
async fn run_flight(
    state: &AppState,
    path: &str,
    mut work: MissWork,
    if_none_match: Option<&str>,
    mut record: RequestRecord,
) -> (Response, RequestRecord) {
    // Debug requests bypass everything, per request.
    if work.debug {
        let (response, record, _) = serve_miss(state, work, record).await;
        return (response, record);
    }
    // Meta requests coalesce through their own flights
    // (issue #295), keyed by [`meta_flight_key`].
    if work.meta {
        let flight_key = meta_flight_key(&work);
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
    // plain text whatever it says), the quantized width,
    // not the raw hint, and whether Save-Data applies, not
    // the raw header, so browsers that phrase the same
    // request differently still share one flight. The
    // conditional header is left out: each waiter applies
    // its own. A leader whose client goes away after its
    // render has started still finishes the flight (see
    // `finish_started_miss`). A waiter that cannot take
    // the response (another spelling of the key, a body
    // past the share cap, a leader whose request was
    // dropped before its render started, a leader that
    // forwent its fallback run because its client had
    // gone, a run a purge of the key overtook) leads or
    // follows the next flight.
    // Waits are capped: a waiter whose flights
    // keep ending without an answer for it stops queueing
    // and runs the miss path itself (issue #375), so it
    // ends with the origin's real answer instead of a busy
    // response masking it.
    let (preferred, dpr_hint, hinted_width, save_data) = (
        work.preferred,
        work.dpr_hint,
        work.hinted_width,
        work.save_data,
    );
    let flight_request = format!("{path:?}{preferred:?}{dpr_hint:?}{hinted_width:?}{save_data}");
    let (cache_key, groups) = (work.cache_key.clone(), work.groups.clone());
    let (max_age, sealed) = (work.max_age, work.sealed);
    let mut waits = 0;
    let mut followed: Option<Arc<FlightSlot>> = None;
    // One disk budget for every tier check this request makes
    // (issue #572).
    let mut budget = DiskBudget::new();
    loop {
        if let Some((hit, cache_status)) =
            lookup_tiers_within(state, &cache_key, &groups, &mut budget).await
        {
            record.cache_hit = true;
            record.bytes_out = hit.bytes.len();
            let etag = hit.etag.clone();
            let mut response = respond(
                hit.format,
                hit.quality,
                hit.bytes,
                max_age,
                state.client_hints,
            );
            attach_cache_headers(&mut response, cache_status, &groups, sealed);
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
                let progress = work.progress.clone();
                let task_progress = progress.clone();
                let task_state = state.clone();
                // The render's disk write waits until the
                // followers are woken (issue #572): they read
                // the render from memory and never wait for it.
                let handoff = DiskHandoff::default();
                work.disk_handoff = Some(handoff.clone());
                // The leader's whole part runs as one miss
                // (see `finish_started_miss`): a render it
                // starts is stored and answers the
                // followers even when this request is gone,
                // and the guard is held until it has. A
                // miss that forwent a render for a client
                // that had gone ends without its request's
                // answer, so it shares nothing.
                let lead = async move {
                    let (response, record, render_etag) =
                        serve_miss(&task_state, work, record).await;
                    let (response, shared) = share_response(response, FLIGHT_SHARE_CAP).await;
                    if let Some(shared) = shared.filter(|_| !task_progress.forwent_render()) {
                        *slot.shared.lock().unwrap() = Some(SharedFlight {
                            request: flight_request,
                            snapshot,
                            response: shared,
                            etag: render_etag.clone(),
                            fallback: record.fallback,
                        });
                    }
                    drop(guard);
                    let pending = handoff.lock().unwrap().take();
                    if let Some(pending) = pending {
                        finish_disk_write(
                            &task_state,
                            pending.snapshot,
                            &pending.key,
                            &pending.entry,
                            pending.claim,
                        )
                        .await;
                    }
                    (response, record, render_etag)
                };
                let (mut response, record, render_etag) =
                    finish_started_miss(state, progress, lead).await;
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
                    let progress = work.progress.clone();
                    let task_state = state.clone();
                    let miss = async move { serve_miss(&task_state, work, record).await };
                    let (mut response, record, render_etag) =
                        finish_started_miss(state, progress, miss).await;
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

/// Run one image miss (`miss`: the miss path for a key,
/// with whatever its flight does around it) on a task of
/// its own, tracked for the shutdown drain, and wait for
/// its answer.
///
/// The task is what lets a render outlive its client
/// usefully. When the client goes away, the server drops
/// this future, and with it the sender of the `gone`
/// channel. If the miss has not started a render by then
/// (`progress`, see [`MissProgress`]), the task drops the
/// miss too, which stops it where it was, as before: a
/// source load or a gate wait ends, its budget share and
/// permits come back, and it starts no render. If a render
/// has started, it runs to its end on its blocking thread
/// anyway, so the task finishes the miss: the render is
/// stored in the tiers (the guarded writes of
/// [`populate_tiers`], unchanged) and the flight's
/// followers receive it, through the tiers or the flight's
/// slot. A replay of the request is then a hit, or a
/// follower of the render still running, instead of a
/// fresh render each time. Finishing starts no further
/// render: a source that did not decode gets no fallback
/// run once the client has gone, and the flight's
/// followers then make their own attempt. The permits a
/// render holds still come back when that render ends
/// (issue #357), whether or not its client is still there;
/// what runs after it (the tier writes, the slot) holds
/// none. Only image misses run here: meta documents and
/// debug views are never stored, so a render made for one
/// has nothing to keep once its client is gone.
///
/// A miss whose client is still there is part of a
/// request in flight, which the shutdown drain counts as
/// such; one whose client has gone is counted as a render
/// left running (see [`Background::spawn_miss`]).
async fn finish_started_miss<F>(
    state: &AppState,
    progress: MissProgress,
    miss: F,
) -> (Response, RequestRecord, Option<String>)
where
    F: std::future::Future<Output = (Response, RequestRecord, Option<String>)> + Send + 'static,
{
    // Never sent on: dropping the sender, with this
    // future, is the signal.
    let (_waiting, gone) = tokio::sync::oneshot::channel::<std::convert::Infallible>();
    let background = state.background.clone();
    let task = state.background.spawn_miss(async move {
        let mut miss = std::pin::pin!(miss);
        tokio::select! {
            // The client's departure is checked first on
            // every wake, so a miss whose client has gone
            // never takes a lane permit that frees up in
            // the same instant and starts a render.
            biased;
            _ = gone => {
                if progress.render_started() {
                    progress.mark_client_gone();
                    let _left = background.client_left();
                    Some(miss.await)
                } else {
                    None
                }
            }
            answer = &mut miss => Some(answer),
        }
    });
    match task.await {
        Ok(Some(answer)) => answer,
        // A panic on the miss path stays the panic of
        // this request, as when the path ran inline.
        Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
        // The task gives up only once this future is gone,
        // and is cancelled only by a runtime that is
        // shutting down, which drops this request too: no
        // client receives this answer.
        _ => (
            StatusCode::SERVICE_UNAVAILABLE.into_response(),
            RequestRecord::default(),
            None,
        ),
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
/// computed from (see [`meta_flight_key`]).
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

/// Whether `entry` must not stand in any tier: a purge
/// recorded after `snapshot` covers it, so the write that
/// carries the snapshot began before that purge (issue
/// #350), or a live tombstone covers the copy by its stamp,
/// as when it was rendered before another replica's purge
/// began (issue #461).
fn write_is_stale(state: &AppState, snapshot: u64, key: &str, entry: &CachedImage) -> bool {
    state
        .purge_guard
        .write_covered(snapshot, key, &entry.groups, entry.written)
}

/// Store `entry` in the memory tier under the purge
/// guard (issue #350). The write is skipped when a purge
/// covering it already landed; a purge that lands while
/// the write is in flight is reconciled right after: the
/// entry just written is removed again, so the purge
/// wins however the two interleave. The removal matches
/// on the entry's ETag and stamp, so an entry a fresher
/// write has already replaced under the same key is left
/// alone. A copy memory held keeps its groups in the new
/// one (issue #572); returns those the slower tiers still
/// have to learn.
fn put_memory_tier(state: &AppState, snapshot: u64, key: &str, entry: &CachedImage) -> Vec<String> {
    if write_is_stale(state, snapshot, key, entry) {
        return Vec::new();
    }
    let took = state.cache.put(key.to_string(), entry.clone());
    let stored = crate::cache::union_groups(&entry.groups, &took);
    if state
        .purge_guard
        .write_covered(snapshot, key, &stored, entry.written)
        && state
            .cache
            .get(key)
            .is_some_and(|held| crate::cache::same_copy(&held, entry))
    {
        state.cache.remove(key);
    }
    took
}

/// Promote a copy a slower tier served into memory under
/// the purge guard (issue #350). A copy memory already
/// holds keeps its bytes and gains the groups (issue
/// #572), so a promotion that lands late never takes back a
/// group a hit taught memory meanwhile. Returns the groups
/// memory held that `entry` lacked, which the slower tiers
/// still have to learn.
fn promote_memory_tier(
    state: &AppState,
    snapshot: u64,
    key: &str,
    entry: &CachedImage,
) -> Vec<String> {
    if write_is_stale(state, snapshot, key, entry) {
        return Vec::new();
    }
    match state.cache.promote(key, entry) {
        crate::cache::Promoted::Inserted => {
            if write_is_stale(state, snapshot, key, entry)
                && state
                    .cache
                    .get(key)
                    .is_some_and(|held| crate::cache::same_copy(&held, entry))
            {
                state.cache.remove(key);
            }
            Vec::new()
        }
        crate::cache::Promoted::Merged {
            groups,
            took,
            grew,
            written,
        } => {
            // A held copy that grows into a group a live purge
            // covers leaves memory, as on a memory hit.
            let covered = if grew {
                state
                    .purge_guard
                    .write_covered(snapshot, key, &groups, written)
            } else {
                state.purge_guard.purge_covers(snapshot, key, &groups)
            };
            if covered {
                state.cache.remove(key);
            }
            took
        }
        crate::cache::Promoted::Skipped => Vec::new(),
    }
}

/// What a learn tells the disk tier about the bucket copy,
/// given how its write went (issue #572): only a write that
/// was spawned, or is queued, carries the groups.
fn bucket_for(sent: Sent, groups: &[String]) -> crate::storage::Bucket<'_> {
    match sent {
        Sent::Spawned => crate::storage::Bucket::Carries(groups),
        Sent::Skipped => crate::storage::Bucket::Skipped,
        Sent::Nothing => crate::storage::Bucket::Unknown,
    }
}

/// Teach the disk tier that a copy of `key` was served
/// under `groups` (issue #572): the index records them at
/// once, with no file I/O, and the drain writes them into
/// the file. When the record grows into a group a purge
/// covers, the disk copy goes: the reconcile a disk merge
/// ran on main, with the record's own stamp and id.
fn learn_disk_groups(
    state: &AppState,
    snapshot: u64,
    key: &str,
    groups: &[String],
    bucket: crate::storage::Bucket<'_>,
) {
    let Some(store) = &state.store else {
        return;
    };
    if let crate::storage::Learned::Indexed {
        grew: true,
        groups,
        written,
        id,
    } = store.learn_groups(key, groups, bucket)
    {
        if state
            .purge_guard
            .write_covered(snapshot, key, &groups, written)
        {
            store.forget_if(key, id);
        }
        store.kick();
    }
}

/// Teach the disk and shared tiers the groups memory held
/// for `key` that `entry`, the copy a caller just wrote to
/// them, lacked (issue #572): memory served the entry under
/// them, so every copy has to carry them.
fn teach_slower_tiers(
    state: &AppState,
    snapshot: u64,
    key: &str,
    entry: &CachedImage,
    took: Vec<String>,
) {
    if took.is_empty() {
        return;
    }
    let copy = CachedImage {
        groups: crate::cache::union_groups(&entry.groups, &took),
        ..entry.clone()
    };
    let sent = spawn_shared_put(state, snapshot, key, copy.clone(), SharedWrite::Groups);
    learn_disk_groups(
        state,
        snapshot,
        key,
        &copy.groups,
        bucket_for(sent, &copy.groups),
    );
}

/// The groups this process knows `key` has been served
/// under (issue #572): the disk record's, or those of a disk
/// write in flight, and the memory copy's. A copy that serves
/// instead of those takes them, so no copy written below
/// lacks one.
fn known_groups(state: &AppState, key: &str) -> Vec<String> {
    let disk = state
        .store
        .as_ref()
        .and_then(|store| store.known_groups(key))
        .unwrap_or_default();
    match state.cache.groups(key) {
        Some(memory) => crate::cache::union_groups(&disk, &memory),
        None => disk,
    }
}

/// Whether `extra` holds a group `existing` lacks.
fn adds_groups(existing: &[String], extra: &[String]) -> bool {
    extra.iter().any(|g| !existing.contains(g))
}

/// Start a disk write of `entry` under the purge guard (issue
/// #572). A key the disk holds is never written again: its
/// record learns the groups, and when that grows it into a
/// group a purge covers, the disk copy goes, as main's merge
/// and reconcile did. A stale copy is not written. Returns
/// the claim of a write to finish with
/// [`finish_disk_write`]. Runs before the copy enters memory,
/// so a memory hit that learns a group meanwhile reaches the
/// write.
fn begin_disk_write(
    state: &AppState,
    snapshot: u64,
    key: &str,
    entry: &CachedImage,
    bucket: crate::storage::Bucket<'_>,
) -> Option<crate::storage::WriteClaim> {
    let store = state.store.as_ref()?;
    let may_claim = !write_is_stale(state, snapshot, key, entry);
    match store.begin_write(key, &entry.groups, bucket, may_claim) {
        crate::storage::WritePlan::Claimed(claim) => Some(claim),
        crate::storage::WritePlan::Held(crate::storage::Learned::Indexed {
            grew: true,
            groups,
            written,
            id,
        }) => {
            if state
                .purge_guard
                .write_covered(snapshot, key, &groups, written)
            {
                store.forget_if(key, id);
            }
            store.kick();
            None
        }
        _ => None,
    }
}

/// Finish a disk write begun by [`begin_disk_write`] (issue
/// #572). The request waits at most [`DISK_WRITE_WAIT`] for
/// it. A write that has no disk slot by then is dropped, with
/// its claim, and counted as a stall; one that has started
/// runs to its end on its thread. The write lands with every
/// group learned meanwhile, and the reconcile after it checks
/// those under the write's own snapshot: a purge that lands
/// while the write is in flight wins.
async fn finish_disk_write(
    state: &AppState,
    snapshot: u64,
    key: &str,
    entry: &CachedImage,
    claim: crate::storage::WriteClaim,
) {
    let Some(store) = state.store.clone() else {
        return;
    };
    let started = tokio::time::Instant::now();
    let permit = match tokio::time::timeout(DISK_WRITE_WAIT, store.permit()).await {
        Ok(Some(permit)) => permit,
        Ok(None) => return,
        Err(_) => {
            store.note_stall(crate::storage::Stall::Write);
            return;
        }
    };
    let guard = state.purge_guard.clone();
    let key = key.to_string();
    let entry = entry.clone();
    let write = store.blocking_with(permit, move |store| {
        if guard.write_covered(snapshot, &key, &entry.groups, entry.written) {
            return;
        }
        if let Some(landed) = store.land(claim, &entry)
            && guard.write_covered(snapshot, &key, &landed.groups, entry.written)
        {
            store.remove_copy(&key, landed.id);
        }
    });
    let _ = tokio::time::timeout(DISK_WRITE_WAIT.saturating_sub(started.elapsed()), write).await;
}

/// A render's disk write that its flight leader finishes
/// after the followers are woken (issue #572).
struct PendingDiskWrite {
    snapshot: u64,
    key: String,
    entry: CachedImage,
    claim: crate::storage::WriteClaim,
}

/// Where [`populate_tiers`] leaves a flight leader's disk
/// write for the leader to finish once its guard is dropped.
type DiskHandoff = Arc<std::sync::Mutex<Option<PendingDiskWrite>>>;

/// Why a shared-tier write is made, which decides what it
/// does when every write slot is taken (issue #463).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedWrite {
    /// A fresh render's write-through, on a miss. Skipped
    /// when every slot is taken: the bucket only misses a
    /// copy that the next replica to miss renders again.
    Render,
    /// A tier hit's write-back of the request's cache groups
    /// onto the stored copy (issue #350). Never skipped: the
    /// local merge has already recorded the groups, so no
    /// later hit sends them again, and a bucket copy that
    /// never learns a group outlives a purge of it on every
    /// replica. It waits for a slot in its own task instead.
    Groups,
}

/// Write `entry` through to the shared tier in the
/// background under the purge guard (issue #350): the
/// same skip-and-reconcile discipline as the local
/// tiers, applied inside the spawned task, because the
/// purge it must lose to can land at any point during
/// the network write. The reconcile deletes only the
/// object it read back (issue #446), so a fresher render
/// that replaced it meanwhile stays.
///
/// At most [`SHARED_PUTS_AT_ONCE`] writes are in flight
/// (issue #463), each holding its entry until the bucket
/// answers. With every slot taken, a stalled bucket's
/// backlog, a [`SharedWrite::Render`] is skipped and counted
/// on /metrics: the local tiers hold the entry, so the cost
/// is a render on the next replica that misses. A
/// [`SharedWrite::Groups`] write-back is never skipped: it
/// waits for a slot in its own task. One for the same copy,
/// snapshot and groups as one already queued is not spawned
/// again (issue #572): the queued one carries them, so a
/// stall cannot queue one per request.
fn spawn_shared_put(
    state: &AppState,
    snapshot: u64,
    key: &str,
    entry: CachedImage,
    write: SharedWrite,
) -> Sent {
    let Some(shared) = &state.shared_store else {
        return Sent::Nothing;
    };
    let guard = state.purge_guard.clone();
    // A write a purge already covers would be dropped in the
    // task anyway: checked before it takes a slot, so it is
    // not counted as a skip.
    if guard.write_covered(snapshot, key, &entry.groups, entry.written) {
        return Sent::Nothing;
    }
    let (slot, queued) = match write {
        SharedWrite::Render => match state.background.put_slot() {
            Some(slot) => (Some(slot), None),
            None => return Sent::Skipped,
        },
        SharedWrite::Groups => match state.background.queue_groups_write(key, snapshot, &entry) {
            Some(queued) => (None, Some(queued)),
            None => return Sent::Spawned,
        },
    };
    let background = state.background.clone();
    let shared = shared.clone();
    let key = key.to_string();
    // The handle is dropped: the write is detached, and
    // tracked for the shutdown drain.
    let _detached = state.background.spawn(async move {
        let _queued = queued;
        let _slot = match slot {
            Some(slot) => slot,
            None => match background.wait_put_slot().await {
                Some(slot) => slot,
                None => return,
            },
        };
        // Checked again with the slot held: a purge can begin
        // while the write waits for one.
        if guard.write_covered(snapshot, &key, &entry.groups, entry.written) {
            return;
        }
        shared.put(&key, &entry).await;
        if guard.write_covered(snapshot, &key, &entry.groups, entry.written)
            && let Some((held, etag)) = shared.get_tagged(&key).await
            && crate::cache::same_copy(&held, &entry)
        {
            shared.remove_if_match(&key, etag.as_deref()).await;
        }
    });
    Sent::Spawned
}

/// How [`spawn_shared_put`] went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sent {
    /// A write was spawned, or one already queued carries the
    /// copy.
    Spawned,
    /// A render's write-through found every slot taken (issue
    /// #463).
    Skipped,
    /// Nothing to write: no shared tier, or a purge covers the
    /// copy.
    Nothing,
}

/// How long one request waits for the disk tier's lookups,
/// in all (issue #572): the waits for a disk slot and for the
/// reads, across every tier check the request makes. A
/// lookup that runs out goes on to the shared tier or a
/// render.
const DISK_LOOKUP_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

/// How long a request waits for its own disk write, once
/// (issue #572). A write with no disk slot by then is
/// skipped.
const DISK_WRITE_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

/// How long a purge waits for its disk deletions in all
/// while they make progress (issue #572). It stops sooner, at
/// [`DISK_WRITE_WAIT`], once none of them finishes for that
/// long.
const PURGE_DELETE_CAP: std::time::Duration = std::time::Duration::from_secs(30);

/// How long the end of a purge, in the background, waits for
/// one of its disk deletions to finish (issue #572). When none
/// does for that long, the purge ends with its bucket
/// tombstone in force.
const PURGE_END_IDLE: std::time::Duration = std::time::Duration::from_secs(30);

/// How long the end of a purge waits for its disk deletions
/// in all while they make progress.
const PURGE_END_CAP: std::time::Duration = std::time::Duration::from_secs(600);

/// What is left of a request's [`DISK_LOOKUP_BUDGET`]. Only
/// time spent waiting on the disk is charged: a request that
/// waited for a long render before its next check still
/// reads a healthy disk.
#[derive(Debug)]
struct DiskBudget {
    left: std::time::Duration,
}

impl DiskBudget {
    fn new() -> Self {
        DiskBudget {
            left: DISK_LOOKUP_BUDGET,
        }
    }

    #[cfg(test)]
    fn with(left: std::time::Duration) -> Self {
        DiskBudget { left }
    }

    fn spent(&self) -> bool {
        self.left.is_zero()
    }

    /// Wait for `wait` within what is left, and charge the
    /// time it took. `None`, at once once spent, when it ran
    /// out.
    async fn spend<F: std::future::Future>(&mut self, wait: F) -> Option<F::Output> {
        if self.spent() {
            return None;
        }
        let started = tokio::time::Instant::now();
        let out = tokio::time::timeout(self.left, wait).await.ok();
        self.left = match out {
            Some(_) => self.left.saturating_sub(started.elapsed()),
            None => std::time::Duration::ZERO,
        };
        out
    }
}

/// Who acts on a disk lookup's answer (issue #572): the
/// request, if it is still waiting, or nobody. A lookup task
/// whose request went on without it promotes nothing and
/// writes nothing back.
#[derive(Debug, Default)]
struct LookupAnswer(std::sync::atomic::AtomicU8);

impl LookupAnswer {
    const WAITING: u8 = 0;
    const TAKEN: u8 = 1;
    const ABANDONED: u8 = 2;

    fn swap_from_waiting(&self, to: u8) -> bool {
        self.0
            .compare_exchange(
                Self::WAITING,
                to,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_ok()
    }

    /// The lookup task takes its answer. False when the
    /// request has gone on.
    fn take(&self) -> bool {
        self.swap_from_waiting(Self::TAKEN)
    }

    /// The request goes on without the answer. False when the
    /// task took it first: it only has memory work left.
    fn abandon(&self) -> bool {
        self.swap_from_waiting(Self::ABANDONED)
    }
}

/// The `Cache-Status` of a disk hit.
const DISK_HIT: &str = "crabcrop; hit; detail=\"disk\"";

/// [`lookup_tiers_within`] with a whole disk budget, as the
/// tier tests drive one lookup.
#[cfg(test)]
async fn lookup_tiers(
    state: &AppState,
    cache_key: &str,
    groups: &[String],
) -> Option<(CachedImage, &'static str)> {
    lookup_tiers_within(state, cache_key, groups, &mut DiskBudget::new()).await
}

/// A memory hit's group learn (audit R2-15, issues #350,
/// #572): the copy memory holds gains the request's groups,
/// and when its set grows, the bucket copy and the disk
/// record learn the whole set. None of it waits on the disk:
/// the record learns in the index, and the drain writes the
/// file. When the grown set falls under a purge, the copy
/// leaves memory too, as the slower tiers' copies do.
fn memory_hit(state: &AppState, snapshot: u64, key: &str, groups: &[String], hit: &CachedImage) {
    if groups.is_empty() {
        return;
    }
    let held = match state.cache.merge_groups(key, groups) {
        Some(held) => held,
        None => {
            // Memory may have dropped the copy since it was read:
            // the request is still served under its groups.
            if !adds_groups(&hit.groups, groups)
                || state
                    .cache
                    .groups(key)
                    .is_some_and(|memory| !adds_groups(&memory, groups))
            {
                return;
            }
            CachedImage {
                groups: crate::cache::union_groups(&hit.groups, groups),
                ..hit.clone()
            }
        }
    };
    let sent = spawn_shared_put(state, snapshot, key, held.clone(), SharedWrite::Groups);
    learn_disk_groups(
        state,
        snapshot,
        key,
        &held.groups,
        bucket_for(sent, &held.groups),
    );
    if state
        .purge_guard
        .write_covered(snapshot, key, &held.groups, held.written)
        && state
            .cache
            .get(key)
            .is_some_and(|now| crate::cache::same_copy(&now, &held))
    {
        state.cache.remove(key);
    }
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
/// shared copy is a miss, and is deleted on sight, when a
/// live tombstone covers it: a purge, begun here or read
/// from the shared bucket, that is still running or within
/// its grace, and that began after the copy was written
/// (issues #400, #446, #461). The purge may not have
/// reached such a copy yet, and serving or promoting it
/// would outlive the purge; a copy rendered after the
/// purge began serves normally. The memory tier needs no
/// such check: a purge empties this process's memory
/// before any slower tier, and every way back in is a
/// promotion through this chain or a guarded write.
///
/// The disk tier is waited for within `budget` (issue
/// #572): a lookup that finds no disk slot, or whose read
/// does not answer, before the budget runs out goes on to
/// the shared tier, and its task promotes nothing. A
/// request learns groups only when it is served a copy, so
/// one dropped while it waits owes no tier anything.
/// Returns the image and the `Cache-Status` value for
/// the tier that served it.
async fn lookup_tiers_within(
    state: &AppState,
    cache_key: &str,
    groups: &[String],
    budget: &mut DiskBudget,
) -> Option<(CachedImage, &'static str)> {
    let snapshot = state.purge_guard.snapshot();
    if let Some(hit) = state.cache.get(cache_key) {
        memory_hit(state, snapshot, cache_key, groups, &hit);
        return Some((hit, "crabcrop; hit"));
    }
    if let Some(store) = &state.store
        && store.contains(cache_key)
        && !budget.spent()
    {
        // The slot is taken before the task starts, so no task
        // is left waiting for one, and at most one per slot
        // can be stuck in the kernel.
        match budget.spend(store.permit()).await {
            None => store.note_stall(crate::storage::Stall::LookupSlot),
            Some(None) => {}
            Some(Some(permit)) => {
                // The disk hit and everything it owes the other
                // tiers run in a spawned task (issue #347):
                // dropping this request's JoinHandle detaches
                // the task instead of cancelling it, so a
                // client that disconnects while the disk
                // answers still has its hit's groups learned
                // (issue #350). Tracked like every detached
                // task (issue #463).
                let answer = Arc::new(LookupAnswer::default());
                let mut task = state.background.spawn(lookup_disk_tier(
                    state.clone(),
                    snapshot,
                    cache_key.to_string(),
                    groups.to_vec(),
                    permit,
                    answer.clone(),
                ));
                match budget.spend(&mut task).await {
                    Some(Ok(Some(image))) => return Some((image, DISK_HIT)),
                    Some(_) => {}
                    None if answer.abandon() => store.note_stall(crate::storage::Stall::LookupRead),
                    // The task took its answer first: only memory
                    // work is left in it.
                    None => {
                        if let Ok(Some(image)) = task.await {
                            return Some((image, DISK_HIT));
                        }
                    }
                }
            }
        }
    }
    if let Some(shared) = &state.shared_store
        && let Some((mut image, etag)) = shared.get_tagged(cache_key).await
    {
        // Checked after the read: a walk can delete this
        // object and end while its bytes are in flight, and
        // an ended purge's tombstone still covers the copy.
        // The delete is conditional on the copy read, so a
        // fresh render that replaced it meanwhile stays.
        if state
            .purge_guard
            .running_covers(cache_key, &image.groups, image.written)
        {
            shared.remove_if_match(cache_key, etag.as_deref()).await;
            return None;
        }
        // The copy takes the request's groups and every group
        // this process knows for the key (issue #572): it
        // serves instead of the copies that were served under
        // them. When a purge covers one of those known groups,
        // the write-back would be dropped and the copy would
        // not learn the request's groups either, so it takes
        // the request's alone.
        let mut wanted = crate::cache::union_groups(groups, &known_groups(state, cache_key));
        let covered = |extra: &[String]| {
            state.purge_guard.write_covered(
                snapshot,
                cache_key,
                &crate::cache::union_groups(&image.groups, extra),
                image.written,
            )
        };
        if covered(&wanted) && !covered(groups) {
            wanted = groups.to_vec();
        }
        let sent = if adds_groups(&image.groups, &wanted) {
            image.groups = crate::cache::union_groups(&image.groups, &wanted);
            // Off the response path, like the write-through
            // put, but never skipped.
            spawn_shared_put(
                state,
                snapshot,
                cache_key,
                image.clone(),
                SharedWrite::Groups,
            )
        } else {
            // The bucket copy already carries them all.
            Sent::Spawned
        };
        // The disk write is claimed before the copy enters
        // memory, so a memory hit that learns a group meanwhile
        // reaches the write (issue #572).
        let disk = begin_disk_write(
            state,
            snapshot,
            cache_key,
            &image,
            bucket_for(sent, &image.groups),
        );
        let took = promote_memory_tier(state, snapshot, cache_key, &image);
        teach_slower_tiers(state, snapshot, cache_key, &image, took);
        if let Some(claim) = disk {
            finish_disk_write(state, snapshot, cache_key, &image, claim).await;
        }
        return Some((image, "crabcrop; hit; detail=\"shared\""));
    }
    None
}

/// The disk tier's half of [`lookup_tiers_within`], run as its
/// own task under the disk slot its request took, so a
/// dropped request cannot strand it mid-chain: read the
/// entry, and if the request is still waiting for it, learn
/// the request's groups in the index under the purge guard,
/// then pay what the hit owes the other tiers: the group
/// write-back to the shared tier and the promotion into
/// memory, in that order. A request that went on without
/// the answer (issue #572) leaves the task nothing to do.
/// Returns the image on a hit.
async fn lookup_disk_tier(
    state: AppState,
    snapshot: u64,
    key: String,
    groups: Vec<String>,
    permit: tokio::sync::OwnedSemaphorePermit,
    answer: Arc<LookupAnswer>,
) -> Option<CachedImage> {
    let store = state.store.clone()?;
    let guard = state.purge_guard.clone();
    let read_key = key.clone();
    let (mut image, id) = store
        .blocking_with(permit, move |store| {
            let (image, id) = store.get_with_id(&read_key)?;
            if guard.running_covers(&read_key, &image.groups, image.written) {
                // A live tombstone covers this copy (issues
                // #400, #461): drop it rather than serve and
                // promote it.
                store.remove_copy(&read_key, id);
                return None;
            }
            Some((image, id))
        })
        .await
        .flatten()?;
    if !answer.take() {
        return None;
    }
    let hit = store.learn_hit(&key, id, &groups)?;
    image.groups = hit.groups;
    if hit.grew || hit.owed {
        // The record grew, or a skipped render write left the
        // bucket copy without some of its groups: the shared
        // tier's copy learns them (issue #350).
        spawn_shared_put(&state, snapshot, &key, image.clone(), SharedWrite::Groups);
    }
    if hit.grew
        && state
            .purge_guard
            .write_covered(snapshot, &key, &image.groups, image.written)
    {
        store.forget_if(&key, id);
    }
    let took = promote_memory_tier(&state, snapshot, &key, &image);
    teach_slower_tiers(&state, snapshot, &key, &image, took);
    store.kick();
    Some(image)
}

/// Everything the cache-miss path needs, bundled so the
/// singleflight loop in `run_flight` can hand it to
/// whichever request leads the miss.
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
    /// Whether Save-Data steps this render's quality:
    /// [`Negotiated::save_data_applies`], the rule the cache
    /// key and the private marking follow, not the raw
    /// header.
    save_data: bool,
    accept: Option<String>,
    groups: Vec<String>,
    cache_key: String,
    max_age: Option<u32>,
    /// Whether the request arrived in the sealed spelling
    /// (issue #255). Meta for it redacts every operation and
    /// source text, and its image responses carry no
    /// Cache-Group (issue #454). The flag describes this
    /// request, not the render: meta bypasses the cache in
    /// both directions, so a plain meta request on the same
    /// render still reports in full, and a plain hit on an
    /// entry first served sealed still names its groups.
    sealed: bool,
    /// How far this miss has got (see [`MissProgress`]);
    /// its render jobs carry clones.
    progress: MissProgress,
    /// Where a flight leader's render leaves its disk write,
    /// to finish once its followers are woken (issue #572).
    /// `None` everywhere else.
    disk_handoff: Option<DiskHandoff>,
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

/// The gates a miss's engine run takes beyond its lane
/// permit (issue #357).
#[derive(Clone, Copy)]
struct GateNeeds {
    /// The AI gate: the request runs inference (see
    /// [`ai_gate_applies`]).
    ai: bool,
    /// The video gate: a video encode, a `frame:`
    /// extraction, or a `page:` render.
    video: bool,
    /// The animation gate: an animated WebP encode (see
    /// [`wants_animation_gate`]).
    anim: bool,
}

/// The gate a run found still full when its wait ran out.
#[derive(Clone, Copy)]
enum Busy {
    Ai,
    Video,
    Anim,
}

impl Busy {
    /// The first run's 503 for the full gate. A fallback
    /// retry that meets one leaves the original engine
    /// error standing instead.
    fn response(self, accept: Option<&str>) -> Response {
        let message = match self {
            Busy::Ai => "server busy: AI processing queue is full",
            Busy::Video => "server busy: video processing queue is full",
            Busy::Anim => "server busy: animation encoding queue is full",
        };
        busy_response(message, 5, accept)
    }
}

/// One permit from `gate`, waiting at most `wait` for it.
async fn acquire_gate(
    gate: &Arc<tokio::sync::Semaphore>,
    wait: std::time::Duration,
) -> Option<tokio::sync::OwnedSemaphorePermit> {
    match tokio::time::timeout(wait, gate.clone().acquire_owned()).await {
        Ok(Ok(permit)) => Some(permit),
        _ => None,
    }
}

/// Take the permits a miss's run needs beyond its lane
/// permit, AI then video then animation: the one
/// acquisition the first run and the fallback retry share
/// (issue #479). Each wait is bounded (audit PL-14): a
/// burst sheds with a 503 instead of parking unbounded
/// tasks on a gate. The permits travel with the process
/// job, and `run_process` holds them until the job
/// actually finishes (issue #357). A gate still full when
/// its wait runs out is the [`Busy`] it names, and the
/// permits already taken drop with it.
async fn acquire_job_permits(state: &AppState, needs: GateNeeds) -> Result<JobPermits, Busy> {
    let mut permits = JobPermits::default();
    // AI inference is gated process-wide: a burst of removebg
    // requests must not starve the transform pipeline.
    if needs.ai {
        permits.ai = Some(
            acquire_gate(&state.ai_gate, AI_GATE_WAIT)
                .await
                .ok_or(Busy::Ai)?,
        );
    }
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
    // split the protection.
    if needs.video {
        permits.video = Some(
            acquire_gate(&state.video_gate, VIDEO_GATE_WAIT)
                .await
                .ok_or(Busy::Video)?,
        );
    }
    // Animated WebP encodes sit behind a single-permit
    // gate of their own (issue #231): one encode can run
    // for minutes, so without this gate duplicate or
    // distinct animated renders stack encodes until the
    // stills pool is drained. Behind the gate, a second
    // animated request sheds here with an explanation
    // instead of stacking, and the encode's lane permit
    // comes from the slow lane (issue #324), so the still
    // pool keeps every permit throughout.
    if needs.anim {
        permits.anim = Some(
            acquire_gate(&state.anim_gate, ANIM_GATE_WAIT)
                .await
                .ok_or(Busy::Anim)?,
        );
    }
    Ok(permits)
}

/// How a miss's engine runs go, decided once per miss so
/// the first run and a fallback retry agree: the lane each
/// takes its permit from (issue #324), the gates each takes
/// beyond it, and when the process phase began, which
/// starts the request's one deadline.
#[derive(Clone, Copy)]
struct RunPlan {
    lane: Lane,
    needs: GateNeeds,
    start: Instant,
}

/// What a miss rendered.
struct Rendered {
    processed: crate::engine::ProcessedImage,
    /// Whether it was rendered from the configured
    /// fallback rather than the source.
    used_fallback: bool,
}

/// The cache-miss path: load the source, resolve
/// watermarks, run the engine under the gates, populate
/// the tiers, and respond (a meta document when asked).
/// Runs at most once per key at a time, led by one
/// request while its duplicates wait (see `run_flight`).
/// The third return value is the ETag of the rendered
/// bytes when the response serves a render, so the
/// caller can answer conditional requests against the
/// bytes themselves; it is `None` for a meta document
/// and for every error.
///
/// Every stage that can fail runs in [`render_miss`], and
/// its failure is answered here, the one place on the
/// miss path that knows the request's spelling (issue
/// #479).
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
    // Its write stamp (issues #446, #461), taken after the
    // snapshot: a purge that begins after this point covers
    // the copies this render writes (it may read the old
    // source), and one that began before it does not, so
    // they outlive that purge's walk.
    let write_stamp = state.purge_guard.stamp();
    let rendered = render_miss(state, &work, &mut record).await;
    let MissWork {
        meta,
        debug,
        accept,
        groups,
        cache_key,
        max_age,
        sealed,
        disk_handoff,
        ..
    } = work;
    let Rendered {
        processed,
        used_fallback,
    } = match rendered {
        Ok(rendered) => rendered,
        // Once a token has opened, every failure answers
        // the uniform refusal (issues #255, #259); a plain
        // request gets the failure's own answer.
        Err(refusal) => return (sealed_or(sealed, *refusal), record, None),
    };
    let accept = accept.as_deref();
    let cache_status = "crabcrop; fwd=uri-miss";
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
            written: Some(write_stamp),
        };
        populate_tiers(
            state,
            write_snapshot,
            &cache_key,
            entry,
            disk_handoff.as_ref(),
        )
        .await;
    }
    record.bytes_out = processed.bytes.len();
    let mut response = if meta {
        meta_document(state, &processed, sealed, accept).await
    } else {
        respond(
            processed.format,
            processed.quality,
            bytes::Bytes::from(processed.bytes),
            max_age,
            state.client_hints,
        )
    };
    if !meta {
        attach_cache_headers(&mut response, cache_status, &groups, sealed);
    }
    if used_fallback {
        record.fallback = true;
        response.headers_mut().insert(
            "x-crabcrop-fallback",
            axum::http::HeaderValue::from_static("1"),
        );
        // A fallback is a placeholder for a transient failure:
        // it must never be pinned downstream for the immutable
        // year (audit PL-2).
        response.headers_mut().insert(
            header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-store"),
        );
    }
    if debug {
        // A diagnostic is computed for this request only: no
        // downstream cache may pin it.
        response.headers_mut().insert(
            header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-store"),
        );
    }
    // A meta response serves a document about the render,
    // not the render: it carries no validator for the
    // image bytes (and is no-store besides).
    let render_etag = if meta { None } else { Some(render_etag) };
    (response, record, render_etag)
}

/// The stages of the miss path that can fail (issue #479):
/// [`load_source`], [`load_watermarks`], [`build_job`], and
/// [`render_with_fallback`], which takes the gates through
/// [`acquire_job_permits`]. A failure is the plain
/// request's answer, which [`serve_miss`] maps for a sealed
/// one. These stages box it: a bare `Response` is 128
/// bytes, too large for an `Err` (clippy's
/// `result_large_err`).
async fn render_miss(
    state: &AppState,
    work: &MissWork,
    record: &mut RequestRecord,
) -> Result<Rendered, Box<Response>> {
    let accept = work.accept.as_deref();
    let directives = &work.directives;
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
    // origin that answers and then falls behind, stalling or
    // trickling, holds only what it has delivered (see
    // `SourceCharge`).
    let charge = state.loader.source_budget().charge(SOURCE_BUDGET_WAIT);
    let admission = crate::loader::MediaAdmission {
        video: wants_frame,
        pdf: wants_page,
    };
    let (bytes, used_fallback) =
        load_source(state, &work.source_url, admission, &charge, accept, record).await?;
    let watermark_sources =
        load_watermarks(state, &directives.watermark_urls, &charge, accept, record).await?;

    record.process_ran = true;
    let process_start = Instant::now();
    let needs = GateNeeds {
        ai: ai_gate_applies(directives, state.face.is_some(), state.focal.is_some()),
        video: wants_video || wants_frame || wants_page,
        // The animation decision covers animated GIF
        // sources rendered to WebP as well as animated WebP
        // sources, and counts the negotiated format, not
        // only an explicit one (issue #405); it is made
        // once, so the gate, the lane, and a fallback retry
        // agree.
        anim: wants_animation_gate(&bytes, directives, work.preferred),
    };
    // The lane this run draws its permit from (issue
    // #324): long-running work goes to the slow lane, so
    // its saturation sheds slow requests only and stills
    // keep their full capacity; everything else takes
    // the stills lane.
    let lane = if wants_slow_lane(needs.anim, wants_video, wants_frame, wants_page) {
        Lane::Slow
    } else {
        Lane::Process
    };
    let plan = RunPlan {
        lane,
        needs,
        start: process_start,
    };
    let job = build_job(state, work, watermark_sources);
    let source = JobSource {
        bytes,
        share: Some(charge),
    };
    let rendered =
        render_with_fallback(state, &job, source, used_fallback, plan, accept, record).await;
    // The source's budget share drained with its run; the
    // watermarks go now with theirs (issue #403), not after
    // the population writes and meta annotations.
    drop(job);
    rendered
}

/// Load the miss's source, charged to `charge` (issue
/// #403), timing the load into the record. Returns the
/// bytes and whether they are the configured fallback's,
/// which stands in for a source that failed to load.
async fn load_source(
    state: &AppState,
    source_url: &str,
    admission: crate::loader::MediaAdmission,
    charge: &crate::loader::SourceCharge,
    accept: Option<&str>,
    record: &mut RequestRecord,
) -> Result<(Vec<u8>, bool), Box<Response>> {
    let load_start = Instant::now();
    let loaded = match state
        .loader
        .load_media_charged(source_url, admission, charge)
        .await
    {
        Ok(b) => (b, false),
        Err(e) => {
            record.load = load_start.elapsed();
            // A policy refusal (Forbidden) is a decision, not a
            // failure: it never falls back. Neither does an
            // exhausted source budget or lookup slots (issue
            // #403): that is overload, and the fallback would
            // need budget too. Nor does source text that does
            // not parse (issue #467): a 400 stays honest, like
            // a grammar failure. Any other load failure
            // degrades to the configured fallback when there
            // is one.
            let fb = if e.is_busy()
                || matches!(e, LoadError::Forbidden(_) | LoadError::InvalidSource(_))
            {
                None
            } else {
                fallback_source_bytes(state, charge).await
            };
            match fb {
                Some(b) => (b, true),
                None => {
                    // A budget shed is overload, like a full
                    // gate, not a loader error (issue #403).
                    if !e.is_busy() {
                        record.fail = FailStage::Load;
                    }
                    return Err(Box::new(load_error_response(e, accept)));
                }
            }
        }
    };
    record.load = load_start.elapsed();
    record.bytes_in = loaded.0.len();
    Ok(loaded)
}

/// Load the miss's watermark images.
///
/// Watermarks are fetched here, before the engine runs:
/// each referenced image goes through the same loader (and
/// its bounds and source policy) as sources, and a
/// watermark that fails to load fails the request rather
/// than being silently skipped. Only the bytes travel
/// with the job: the decode happens inside the gated
/// blocking job with the rest of the engine work, so a
/// watermark that does not decode fails the request from
/// there with the same engine error (audit R2-16).
/// Watermark loads charge the same budget, against the
/// request's one wait (issue #403); their share is split
/// off to travel with the watermark bytes.
async fn load_watermarks(
    state: &AppState,
    urls: &[String],
    charge: &crate::loader::SourceCharge,
    accept: Option<&str>,
    record: &mut RequestRecord,
) -> Result<Arc<WatermarkSources>, Box<Response>> {
    let mut watermark_images: Vec<(String, Vec<u8>)> = Vec::new();
    for wm_url in urls {
        record.load_ran = true;
        let wm_bytes = match state
            .loader
            .load_media_charged(wm_url, crate::loader::MediaAdmission::default(), charge)
            .await
        {
            Ok(b) => b,
            Err(e) => {
                if !e.is_busy() {
                    record.fail = FailStage::Load;
                }
                return Err(Box::new(load_error_response(e, accept)));
            }
        };
        watermark_images.push((wm_url.to_string(), wm_bytes));
    }
    let watermark_bytes: usize = watermark_images.iter().map(|(_, b)| b.len()).sum();
    Ok(Arc::new(WatermarkSources {
        images: watermark_images,
        share: Some(charge.split_off(watermark_bytes)),
    }))
}

/// The operations a width hint puts in front of the URL's
/// own: the width-only size op, plus `no_upscale` unless
/// the request already has it. [`negotiate`] counts them
/// against the operation cap before it keeps the hint
/// (issue #615), and [`build_job`] adds them.
fn width_hint_ops(width: u32, no_upscale: bool) -> Vec<crate::url::Operation> {
    let mut ops = vec![crate::url::Operation::Size(crate::url::ImageSize {
        width: crate::url::Dimension::Pixels(width),
        height: crate::url::Dimension::Proportional,
        flip_horizontal: false,
        flip_vertical: false,
    })];
    if !no_upscale {
        ops.push(crate::url::Operation::Filters(vec![
            crate::url::Filter::NoUpscale,
        ]));
    }
    ops
}

/// The engine job for a miss: the request's operations and
/// directives with the server's providers and settings.
fn build_job(
    state: &AppState,
    work: &MissWork,
    watermark_sources: Arc<WatermarkSources>,
) -> ProcessJob {
    let directives = &work.directives;
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
    let job_ops = match work.hinted_width {
        Some(w) => {
            let mut combined = width_hint_ops(w, directives.no_upscale);
            combined.extend(work.ops.iter().cloned());
            combined
        }
        None => work.ops.clone(),
    };
    ProcessJob {
        save_data: work.save_data,
        preserve_metadata: state.preserve_metadata,
        ops: job_ops,
        preferred: work.preferred,
        watermark_sources,
        placeholders: work.placeholders,
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
        want_provenance: work.meta,
        dpr: work.dpr_hint,
        ssim_target: state.ssim_target,
        brand_color: state.brand_color,
        avif_speed: state.avif_speed,
        jxl_speed: state.jxl_speed,
        progress: work.progress.clone(),
    }
}

/// Run the miss's job under its gates, and once more on the
/// configured fallback when the source loaded but did not
/// decode ([`retry_on_fallback`]). Times the process phase
/// into the record.
async fn render_with_fallback(
    state: &AppState,
    job: &ProcessJob,
    source: JobSource,
    mut used_fallback: bool,
    plan: RunPlan,
    accept: Option<&str>,
    record: &mut RequestRecord,
) -> Result<Rendered, Box<Response>> {
    // The request's one process deadline starts with the
    // process phase, so the gate waits and the run itself
    // share a single budget (run_process takes it as a
    // parameter).
    let deadline = plan.start + state.process_timeout;
    let permits = acquire_job_permits(state, plan.needs)
        .await
        .map_err(|busy| Box::new(busy.response(accept)))?;
    let processed = match run_process(state, job, source, plan.lane, deadline, permits).await {
        Ok(p) => p,
        // A first run never finds its client gone (see
        // `ProcessFailure::ClientGone`); were it to, nobody
        // would receive the busy answer.
        Err(ProcessFailure::Busy | ProcessFailure::ClientGone) => {
            return Err(Box::new(match plan.lane {
                Lane::Slow => busy_response(
                    "server busy: long-running processing lane is full",
                    5,
                    accept,
                ),
                Lane::Process => {
                    busy_response("server busy: too many images processing at once", 1, accept)
                }
            }));
        }
        Err(ProcessFailure::Panicked) => {
            record.fail = FailStage::Engine;
            return Err(Box::new(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "processing_failed",
                "image processing failed".to_string(),
                accept,
            )));
        }
        Err(ProcessFailure::TimedOut) => {
            record.fail = FailStage::Engine;
            return Err(Box::new(error_response(
                StatusCode::GATEWAY_TIMEOUT,
                "processing_timeout",
                "processing timed out".to_string(),
                accept,
            )));
        }
        Err(ProcessFailure::Engine(e)) => {
            record.process = plan.start.elapsed();
            // A source that loads but does not decode is the
            // other fallback case. Other engine errors are
            // properties of the request or the configuration,
            // not of the source, and stay honest.
            let retry = if !used_fallback && matches!(e, EngineError::Decode(_)) {
                retry_on_fallback(state, job, plan, deadline).await
            } else {
                None
            };
            match retry {
                Some(p) => {
                    used_fallback = true;
                    record.process = plan.start.elapsed();
                    p
                }
                None => {
                    record.fail = FailStage::Engine;
                    return Err(Box::new(engine_error_response(e, accept)));
                }
            }
        }
    };
    record.process = plan.start.elapsed();
    Ok(Rendered {
        processed,
        used_fallback,
    })
}

/// The fallback run for a source that loaded but did not
/// decode: the same job on the configured fallback's bytes,
/// under the same deadline. `None` leaves the original
/// engine error standing: no fallback, no budget for it, a
/// full gate, a failed run, or a client that has gone (see
/// [`MissProgress`]), before the run or by the time its
/// lane comes free.
async fn retry_on_fallback(
    state: &AppState,
    job: &ProcessJob,
    plan: RunPlan,
    deadline: Instant,
) -> Option<crate::engine::ProcessedImage> {
    // A fallback is never stored, so for a client that has
    // gone it would be work nobody receives: it is not
    // fetched or charged, and it waits for no gate.
    if state.fallback.is_none() || !job.progress.may_start_render() {
        return None;
    }
    // The first run's source share drained with its job, so
    // the fallback is charged afresh (issue #403), with a
    // wait of its own like each gate below; the watermarks
    // keep the share they travel with. No budget leaves the
    // original engine error standing, like a full gate.
    let charge = state.loader.source_budget().charge(SOURCE_BUDGET_WAIT);
    let bytes = fallback_source_bytes(state, &charge).await?;
    // The fallback run draws on the same gates as the first:
    // each kind of work re-acquires its permit (the first
    // run's permits drained with its job), and a full gate
    // leaves the original engine error standing.
    let permits = acquire_job_permits(state, plan.needs).await.ok()?;
    let source = JobSource {
        bytes,
        share: Some(charge),
    };
    run_process(state, job, source, plan.lane, deadline, permits)
        .await
        .ok()
}

/// Store a fresh render in every tier under the purge
/// guard (issue #350), at the generation the render
/// started under: a purge that landed while this render
/// was running keeps the entry out of every tier, while
/// the response still serves the render itself. The shared
/// tier's write stays off the response path.
///
/// The stored copy takes every group this process knows for
/// the key (issue #572): it serves instead of the copies
/// served under them. The response's own headers still name
/// the request's groups. A key the disk holds is not written
/// again; its record learns the groups. With `handoff`, a
/// flight leader's disk write is left there for the leader
/// to finish after its followers are woken, but only when
/// memory holds this copy for them to read.
async fn populate_tiers(
    state: &AppState,
    snapshot: u64,
    cache_key: &str,
    mut entry: CachedImage,
    handoff: Option<&DiskHandoff>,
) {
    let known = known_groups(state, cache_key);
    if adds_groups(&entry.groups, &known) {
        entry.groups = crate::cache::union_groups(&entry.groups, &known);
    }
    // The shared write is spawned before any wait (issue
    // #347): it is off the response path, and the order means
    // a request dropped during the disk write cannot strand
    // it.
    let sent = spawn_shared_put(
        state,
        snapshot,
        cache_key,
        entry.clone(),
        SharedWrite::Render,
    );
    // Claimed before the copy enters memory, so a memory hit
    // that learns a group meanwhile reaches the write.
    let disk = begin_disk_write(
        state,
        snapshot,
        cache_key,
        &entry,
        bucket_for(sent, &entry.groups),
    );
    let took = put_memory_tier(state, snapshot, cache_key, &entry);
    teach_slower_tiers(state, snapshot, cache_key, &entry, took);
    let Some(claim) = disk else {
        return;
    };
    if let Some(handoff) = handoff
        && state.cache.holds_copy(cache_key, &entry)
    {
        *handoff.lock().unwrap() = Some(PendingDiskWrite {
            snapshot,
            key: cache_key.to_string(),
            entry,
            claim,
        });
        return;
    }
    finish_disk_write(state, snapshot, cache_key, &entry, claim).await;
}

/// The meta document for a render: the annotation sections
/// the request opts into, then the document itself.
async fn meta_document(
    state: &AppState,
    processed: &crate::engine::ProcessedImage,
    sealed: bool,
    accept: Option<&str>,
) -> Response {
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
    let annotation = tokio::time::timeout_at(
        tokio::time::Instant::from_std(annotation_deadline),
        annotate_meta(state, processed),
    )
    .await
    .ok()
    .flatten();
    let detections = if wants_detections(accept) {
        tokio::time::timeout_at(
            tokio::time::Instant::from_std(annotation_deadline),
            detect_meta(state, processed),
        )
        .await
        .ok()
        .flatten()
    } else {
        None
    };
    let moderation = if wants_moderation(accept) {
        Some(
            match tokio::time::timeout_at(
                tokio::time::Instant::from_std(annotation_deadline),
                moderate_meta(state, processed),
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
    let want_embedding = wants_embeddings(accept);
    let embedding_run = if want_embedding || state.aesthetic.is_some() {
        Some(
            match tokio::time::timeout_at(
                tokio::time::Instant::from_std(annotation_deadline),
                embed_meta(state, processed),
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
    let ocr = if wants_ocr(accept) {
        Some(
            match tokio::time::timeout_at(
                tokio::time::Instant::from_std(annotation_deadline),
                ocr_meta(state, processed, annotation_deadline),
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
    meta_response(
        processed,
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
    annotate_meta, compose_aesthetic, detect_meta, embed_meta, meta_cache_headers, meta_response,
    moderate_meta, ocr_meta, wants_detections, wants_embeddings, wants_meta, wants_moderation,
    wants_ocr, wants_placeholders,
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
    client_hints: bool,
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
    headers.insert(
        header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static(content_type),
    );
    headers.insert(header::CACHE_CONTROL, cache_control.parse().unwrap());
    // `Vary` is set by `handle`, on every answer of the
    // image route alike. Advertise the client hints the
    // server reads, so browsers send them on later requests;
    // without the advertisement only proxies and clients
    // that set the hints manually ever benefit from them.
    // With hints off there is nothing to advertise.
    if client_hints {
        headers.insert(
            axum::http::HeaderName::from_static("accept-ch"),
            axum::http::HeaderValue::from_static("Sec-CH-Width, Sec-CH-DPR, Sec-CH-Viewport-Width"),
        );
    }
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

const SOURCE_LOOKUP_BUSY: &str = "server busy: too many source host lookups waiting";

/// The 503 for a load shed as overload, when `e` is one: the
/// source buffer budget's, or the host lookup slots' (see
/// [`crate::loader::LOOKUP_SLOTS`]), which carries the gates'
/// Retry-After, since a slot that stayed taken for the whole
/// wait is held by a lookup the resolver is not answering.
fn load_busy_response(e: &LoadError, accept: Option<&str>) -> Option<Response> {
    match e {
        LoadError::Busy => Some(source_budget_busy(accept)),
        LoadError::LookupBusy => Some(busy_response(SOURCE_LOOKUP_BUSY, 5, accept)),
        _ => None,
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
/// bucket, which is the operator's fact, not the caller's.
/// A host name that resolves to a non-public address is a
/// [`LoadError::Network`] cause, so it answers exactly as a
/// name that does not resolve.
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
        // The caller's own source text is at fault, not an
        // origin (issue #467), so this is a 400 like a
        // grammar failure, not a gateway error.
        LoadError::InvalidSource(_) => (
            StatusCode::BAD_REQUEST,
            "invalid_source",
            "invalid source URL".to_string(),
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
        LoadError::LookupBusy => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_busy",
            SOURCE_LOOKUP_BUSY.to_string(),
        ),
    }
}

fn load_error_response(e: LoadError, accept: Option<&str>) -> Response {
    if let Some(busy) = load_busy_response(&e, accept) {
        return busy;
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
/// but its code never changes. What this builds is never
/// sent to a sealed request for a payload failure, which
/// answers the uniform refusal instead: the image path's
/// [`Refusal::answer`] skips this builder, and the miss
/// path's [`sealed_or`] drops the answer its failure
/// built. Every error answer leaves with `Cache-Control:
/// no-store`, which the router's response-header layer
/// adds on every route, and on the image route with the
/// same `Vary` as an image ([`image_vary`]).
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

/// The parts of an error answer before [`error_response`]
/// shapes them for the client's `Accept`: the status, the
/// stable code, and the message. A refusal that carries
/// them leaves the answer to the one place that knows how
/// the request may be answered (see [`Refusal::answer`]).
type ErrorParts = (StatusCode, &'static str, String);

/// The refusal for an image remainder that begins with
/// [`crabcrop_url::RESERVED_SEGMENT`]. Every derived subkey
/// (purge, purge-group, upload, seal, cache entry) is the
/// HMAC of a label under that segment, so no image URL may
/// begin with it, and no source can: it names no scheme the
/// loader fetches.
fn reserved_remainder_parts() -> ErrorParts {
    (
        StatusCode::BAD_REQUEST,
        "reserved_segment",
        format!(
            "an image URL may not begin with the reserved segment {}",
            crabcrop_url::RESERVED_SEGMENT
        ),
    )
}

/// A grammar failure's parts: the parser's code and its
/// (truncated) message.
fn parse_failure_parts(e: &crate::url::ParseError) -> ErrorParts {
    (
        StatusCode::BAD_REQUEST,
        e.code(),
        truncate_echo(&e.to_string()),
    )
}

/// A grammar failure's response: the parser's code and
/// its (truncated) message, as plain text or the v2
/// envelope per the client's Accept.
fn parse_failure(e: &crate::url::ParseError, accept: Option<&str>) -> Response {
    let (status, code, message) = parse_failure_parts(e);
    error_response(status, code, message, accept)
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
        // A dependency outage, not a broken encoder (issue
        // #467): the output was rendered, but the timestamp
        // authority signing needs did not answer, and a
        // retry may find it back.
        EngineError::ProvenanceUnavailable(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "provenance_unavailable",
            "provenance signing is unavailable".to_string(),
        ),
    }
}

/// `Retry-After` on a `provenance_unavailable` answer, in
/// seconds. Longer than a gate's: an authority outage
/// outlasts a queue, and every retry that meets it holds a
/// process permit for up to the authority's 15-second
/// bound before it fails again.
const PROVENANCE_RETRY_AFTER_SECS: u64 = 30;

fn engine_error_response(e: EngineError, accept: Option<&str>) -> Response {
    eprintln!("engine error: {e}");
    let retry_after = matches!(e, EngineError::ProvenanceUnavailable(_));
    let (status, code, message) = engine_error_parts(e);
    let mut response = error_response(status, code, message, accept);
    if retry_after {
        response.headers_mut().insert(
            header::RETRY_AFTER,
            PROVENANCE_RETRY_AFTER_SECS.to_string().parse().unwrap(),
        );
    }
    response
}

#[cfg(test)]
mod tests;
