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
#[cfg(test)]
use std::time::Instant;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    Router,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};

use crate::cache::Cache;
#[cfg(test)]
use crate::cache::{CachedImage, etag_for_bytes};
use crate::engine::{Engine, EngineError};
use crate::loader::{LoadError, Loader};
use crate::observe::{Metrics, RequestLog, RequestRecord};
#[cfg(test)]
use crate::url::ImageFormat;
use crate::url::{parse_operations_with_expiry, parse_signature};

/// One declared `boxes` detector (issue #166): the name
/// its declaration carries in `CRABCROP_DETECTORS_FILE`
/// and its provider. Kept in declaration order in
/// [`Providers`]: meta aggregation and class resolution
/// both walk the built-in detector first, then these.
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
pub use background::{
    Background, DEFAULT_SHUTDOWN_GRACE_SECS, Drained, SHARED_GROUP_WRITES_WAITING,
    SHARED_PUTS_AT_ONCE,
};
mod listen;
pub use listen::{
    ConnectionLimits, DEFAULT_HEADER_TIMEOUT_SECS, DEFAULT_IDLE_TIMEOUT_SECS, RequestsInFlight,
    bind_listener, serve_connections,
};
mod process;
pub(in crate::server) use process::{
    AI_GATE_WAIT, ANIM_GATE_WAIT, Flight, FlightSlot, JobPermits, JobSource, Lane, LeaderGuard,
    MetaFlight, MissProgress, PROCESS_GATE_WAIT, ProcessFailure, ProcessJob, SOURCE_BUDGET_WAIT,
    SharedFlight, SharedResponse, VIDEO_GATE_WAIT, WatermarkSources, run_process,
    wants_animation_gate, wants_slow_lane,
};
pub use process::{DEFAULT_SLOW_LANE_PERMITS, Gate, Inflight};
#[cfg(test)]
pub(in crate::server) use process::{SLOW_LANE_WAIT, apply_provenance};

mod response;
pub(in crate::server) use response::{
    ErrorParts, SOURCE_BUDGET_BUSY, accept_of, bad_signature_parts, busy_response, error_response,
    load_busy_response, load_error_parts,
};
#[cfg(test)]
pub(in crate::server) use response::{
    engine_error_response, finalize_validators, if_none_match_matches, load_error_response,
    parse_failure_parts, reserved_remainder_parts, respond,
};
mod system;
#[cfg(test)]
pub(in crate::server) use system::token_eq;
pub(in crate::server) use system::{health, metrics};
mod admission;
pub(in crate::server) use admission::{
    Credential, Opened, Refusal, Resolved, authenticate, resolve_request, unix_seconds,
};
#[cfg(test)]
pub(in crate::server) use admission::{Expiry, Negotiated, negotiate_format};
mod request;
#[cfg(test)]
pub(in crate::server) use request::{
    DPR_HINT_HEADERS, MAX_HINT_WIDTH, OwedRecord, SAVE_DATA_HEADER, VARY_WITH_HINTS,
    VARY_WITHOUT_HINTS, WIDTH_HINT_HEADERS, cache_key_for, quantize_width, save_data_hint,
    width_hint,
};
pub(in crate::server) use request::{dpr_hint, handle};
mod flight;
#[cfg(test)]
pub(in crate::server) use flight::MAX_FLIGHT_WAITS;
mod tiers;
pub(in crate::server) use tiers::DISK_WRITE_WAIT;
#[cfg(test)]
pub(in crate::server) use tiers::{
    DISK_HIT, DISK_LOOKUP_BUDGET, DiskBudget, Sent, SharedWrite, begin_disk_write,
    finish_disk_write, lookup_tiers, lookup_tiers_within, memory_hit, populate_tiers,
    put_memory_tier, spawn_shared_put,
};
mod miss;
pub use miss::Fallback;
#[cfg(doc)]
pub(in crate::server) use miss::render_miss;
#[cfg(test)]
pub(in crate::server) use miss::{MissWork, ai_gate_applies, build_job};

/// Shared state for the handlers: one per server, built at
/// boot and shared behind an `Arc` (issue #475). The router,
/// each request, and each task a request starts hold that
/// `Arc`, never a copy of the state.
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
    /// Where the lines on standard output go once the
    /// server serves (issue #642): each request's line, the
    /// purge and upload lines, and the shutdown lines. By
    /// default a queue that one thread writes from; with
    /// `CRABCROP_LOG_SYNC` each line is written where it is
    /// logged. A handler that logs asks it for room first
    /// ([`RequestLog::room`]).
    pub request_log: RequestLog,
    /// The bearer token `/metrics` requires (issue #270),
    /// from `CRABCROP_METRICS_TOKEN`. `None` closes the
    /// endpoint: it answers 403 rather than serving the
    /// exposition unauthenticated. Zeroed when the state is
    /// dropped.
    pub metrics_token: Option<zeroize::Zeroizing<String>>,
    /// Processed-response cache, shared like the metrics.
    pub cache: Arc<Cache>,
    /// The provider and renderer handles (see
    /// [`Providers`]): one `Arc`, shared with every
    /// engine job.
    pub providers: Arc<Providers>,
    /// Tag threshold for meta annotations, from
    /// `CRABCROP_CLIP_THRESHOLD` (default 0.05): labels whose
    /// composed probability falls below it are omitted from
    /// meta `tags`. The composition itself lives in
    /// `compose_annotation` (issue #152).
    pub tag_threshold: f32,
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
    /// transform pipeline. This gate and the video,
    /// animation and probe gates below are [`Gate`]s, which
    /// know their capacity, so `/metrics` reports each as in
    /// use against capacity, as it does the lanes (issue
    /// #707).
    pub ai_gate: Arc<Gate>,
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
    /// How many tiles an AVIF is written in
    /// (`CRABCROP_AVIF_TILES`, issue #688).
    pub avif_tiles: crate::engine::AvifTiles,
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
    /// Whether the purge endpoints take only the expiring
    /// form (`CRABCROP_PURGE_REQUIRE_EXPIRY`, off by
    /// default). On, a purge or purge-group URL with no
    /// `exp:` segment answers `403 expiry_required` once its
    /// signature has verified, and purges nothing.
    pub purge_require_expiry: bool,
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
    /// Bounds concurrent video encodes: the subprocess is
    /// the server's most expensive and least predictable
    /// work, so it gets its own small gate (2 permits, like
    /// the AI gate) on top of the slow lane (issue #324).
    pub video_gate: Arc<Gate>,
    /// Bounds concurrent animated WebP encodes to one
    /// (issue #231): the animation encode is among the
    /// slowest work the engine does, minutes for a modest
    /// file, so it gets a single-permit gate of its own on
    /// top of the slow lane (issue #324), which is what
    /// keeps it from holding stills permits while it runs.
    pub anim_gate: Arc<Gate>,
    /// Bounds concurrent heavy `/explain` probes (an SVG's
    /// parse in the worker, an AVIF's decode, a HEIC's item
    /// graph walk), which run on blocking threads (see
    /// `explain_probe`): a burst of them waits here, or
    /// sheds with a 503, instead of occupying the async
    /// runtime or starting a worker process per request.
    pub probe_gate: Arc<Gate>,
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

// `AppState` is not `Clone` (issue #475): the server holds
// one, behind an `Arc`, and everything that needs it past a
// borrow clones the `Arc`. A derive would let a whole-state
// copy back in; with it, the call below names two impls and
// the build fails as ambiguous.
const _: fn() = || {
    trait AmbiguousIfClone<A> {
        fn check() {}
    }
    impl<T> AmbiguousIfClone<()> for T {}
    impl<T: Clone> AmbiguousIfClone<u8> for T {}
    <AppState as AmbiguousIfClone<_>>::check();
};

/// The provider and renderer handles (issue #475): every
/// model, external tool, font and signer loaded at boot,
/// each `None` (or empty) when it is not configured. The
/// state and every engine job share one `Arc` of them,
/// so a job takes them all with one pointer copy.
#[derive(Debug, Default)]
pub struct Providers {
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
    /// not-configured error. Extraction shares the state's
    /// video gate: it is the same class of subprocess work
    /// as an encode.
    pub frame_extractor: Option<Arc<crate::frames::VideoFrameExtractor>>,
    /// PDF renderer for `page:` segments on PDF sources
    /// (issue #215), built at boot when
    /// `CRABCROP_PDF_RENDERER` is configured. `None` means
    /// page selection does not exist on this server; a
    /// `page:` request then fails with the specific
    /// not-configured error. Rendering shares the state's
    /// video gate: it is the same class of subprocess work
    /// as frame extraction.
    pub pdf_renderer: Option<Arc<crate::pdf::PdfRenderer>>,
    /// C2PA output signer (issue #173), built and
    /// self-validated at boot when `CRABCROP_C2PA_CERT`
    /// and `CRABCROP_C2PA_KEY` are configured. `None`
    /// means outputs are served unsigned; meta responses
    /// still report the provenance a source carries.
    pub provenance_signer: Option<Arc<crate::provenance::ProvenanceSigner>>,
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
            metrics,
            request_log
        );
        out.field(
            stringify!(metrics_token),
            &self.metrics_token.as_ref().map(|_| crate::redact::Redacted),
        );
        // The provider handles print where they printed
        // before they moved into `Providers` (issue #475),
        // so the output is what it was. The pattern names
        // every handle, so a handle added to `Providers`
        // does not build until it is printed below too.
        let Providers {
            removebg,
            upscale,
            inpaint,
            segment,
            face,
            tagger,
            moderation,
            embeddings,
            aesthetic,
            ocr,
            detect,
            detectors,
            focal,
            font,
            video_encoder,
            frame_extractor,
            pdf_renderer,
            provenance_signer,
        } = &*self.providers;
        macro_rules! providers {
            ($($name:ident),* $(,)?) => {
                $(out.field(stringify!($name), $name);)*
            };
        }
        fields!(cache);
        providers!(removebg, upscale, inpaint, segment, face, tagger);
        fields!(tag_threshold);
        providers!(
            moderation, embeddings, aesthetic, ocr, detect, detectors, focal, font
        );
        fields!(
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
            avif_tiles,
            jxl_speed,
            width_step,
            client_hints,
            explain_enabled,
            purge_require_expiry,
            preserve_metadata,
            brand_color,
            presets,
            only_presets,
            source_roots,
            default_ops
        );
        providers!(
            video_encoder,
            frame_extractor,
            pdf_renderer,
            provenance_signer
        );
        fields!(
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

/// Build the router over `state`: an `AppState`, or the
/// `Arc` a caller that keeps its own handle already holds.
/// Every route shares that one `Arc`.
pub fn app(state: impl Into<Arc<AppState>>) -> Router {
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
        .with_state(state.into())
        .layer(axum::middleware::from_fn(security_headers))
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
    // Read only for an error (issue #641): a success keeps
    // its own caching headers whatever they say.
    let stored_nowhere = error
        && headers
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
/// The request runs in stages. [`admit_explained`] admits
/// it through the image path's own [`authenticate`] and
/// [`resolve_request`], so a URL this route admits is one
/// the image route admits, and a refusal is the image
/// path's own status and body (issue #741). Only then is
/// the source touched, once, through the loader with all
/// of its guards ([`probe_explained`]), and the facts come
/// from header probing (`crate::engine::probe_source`): no
/// decode, no detection, no process, AI, or video gate (an
/// SVG, AVIF, or HEIC probe takes the probe gate, see
/// [`explain_probe`]), and nothing written to any cache
/// tier. A bad signature, an over-cap operation list, or
/// an expired URL therefore costs zero source fetches.
/// [`explain_report`] builds the report and
/// [`explain_response`] answers it.
async fn explain_handle(
    State(state): State<Arc<AppState>>,
    uri: axum::http::Uri,
    headers: HeaderMap,
) -> Response {
    let accept = accept_of(&headers);
    let accept = accept.as_deref();
    // With `CRABCROP_EXPLAIN=0` the route is off (issue #591):
    // every request answers the router's own `404 not_found`
    // before the signature is read, so the answer is the
    // same whatever the URL, and no source is fetched.
    if !state.explain_enabled {
        return error_response(
            StatusCode::NOT_FOUND,
            "not_found",
            "not found".to_string(),
            accept,
        );
    }
    let Some(tail) = uri.path().strip_prefix("/explain/") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /explain/<signature>/<target>".to_string(),
            accept,
        );
    };
    let full = format!("/{tail}");
    let (credential, resolved) = match admit_explained(&state, &full) {
        Ok((opened, resolved)) => (opened.credential, resolved),
        // Only plain spellings are admitted here, so every
        // refusal is the plain spelling's own answer, as on
        // the image route.
        Err(refusal) => return refusal.answer(false, accept),
    };
    let (source_bytes, probe) = match probe_explained(&state, &resolved, accept).await {
        Ok(facts) => facts,
        Err(response) => return *response,
    };
    let report = explain_report(
        &state,
        &headers,
        credential,
        resolved,
        source_bytes,
        probe,
        full,
    );
    explain_response(&report, accept)
}

/// The first stage of [`explain_handle`]: the image path's
/// admission, [`authenticate`] and [`resolve_request`]
/// themselves (issue #741), so the two routes cannot
/// disagree on which plain URLs they admit or on how they
/// refuse the rest.
///
/// A sealed spelling is refused before either runs,
/// wholesale (issue #255): the server never unseals as a
/// service, because an endpoint that opens tokens would be
/// an unsealing oracle for anyone who lifts one off a
/// public page. Operators unseal locally (the CLI's
/// `unseal` verb, under their own secret) and then explain
/// the plain form. The refusal is the uniform
/// `bad_signature`, and no source fetch happens for it.
fn admit_explained<'a>(state: &AppState, full: &'a str) -> Result<(Opened<'a>, Resolved), Refusal> {
    let nolead = full.strip_prefix('/').unwrap_or(full);
    if crate::url::looks_sealed(nolead.split('/').next().unwrap_or("")) {
        return Err(Refusal::AsIs(bad_signature_parts()));
    }
    // The stages time their work into a request record. An
    // explain request owes the request log no line, so this
    // one goes nowhere.
    let mut record = RequestRecord::default();
    let opened = authenticate(state, full, &mut record)?;
    let resolved = resolve_request(state, &opened.remainder, &mut record)?;
    Ok((opened, resolved))
}

/// The second stage of [`explain_handle`]: the single
/// bounded source fetch, through the loader and its
/// policy, and the probe of the head it read
/// ([`explain_probe`]). Returns the source's encoded size
/// when the fetch states it, and the probe; a failed fetch,
/// or a probe shed for a full gate, is the answer instead,
/// boxed as [`render_miss`] boxes its own (a bare `Response`
/// is 128 bytes, too large for an `Err`: clippy's
/// `result_large_err`).
async fn probe_explained(
    state: &AppState,
    resolved: &Resolved,
    accept: Option<&str>,
) -> Result<(Option<usize>, crate::engine::SourceProbe), Box<Response>> {
    // Watermark and overlay sources are operations' inputs,
    // not the explained URL's source, and are never fetched:
    // no fan-out. A `frame:` segment makes the source a
    // video, so the loader may accept a `video/*`
    // Content-Type for it (issue #214); a `page:` segment
    // does the same for `application/pdf` (issue #215).
    // Explain never runs either tool: the fetch is admitted,
    // the probe stays the raw-bytes probe. Both are read
    // from the merged list, the one the image request runs.
    let admission = crate::loader::MediaAdmission {
        video: resolved
            .ops
            .iter()
            .any(|op| matches!(op, crate::url::Operation::Frame(_))),
        pdf: resolved
            .ops
            .iter()
            .any(|op| matches!(op, crate::url::Operation::Page(_))),
    };
    // The report reads the source's headers only, so the
    // fetch is a probe load: it stops as soon as the head
    // identifies the source instead of downloading the
    // whole body (which also keeps this route off the
    // result cache's coalescing path by construction: it
    // never held a whole body to share). A root source is
    // fetched at its expansion.
    let crate::loader::ProbeLoad { head, total_bytes } = state
        .loader
        .load_probe(&resolved.source_url, admission, &probe_settled)
        .await
        .map_err(|e| Box::new(explain_load_error(e, accept)))?;
    let Some(probe) = explain_probe(state, head).await else {
        return Err(Box::new(busy_response(
            "server busy: too many source probes running at once",
            5,
            accept,
        )));
    };
    Ok((total_bytes.map(|n| n as usize), probe))
}

/// The third stage of [`explain_handle`]: the report on an
/// admitted URL and its probed source.
///
/// The forecast, the canonical form and the admission all
/// follow the operations the image request runs, the
/// server-wide defaults merged in (issue #468); the
/// report's operation list is the URL's own, so it never
/// credits the URL with a default. The source is reported
/// exactly as written, with a root's name, never the base
/// URL a root reference resolves to (operator
/// configuration, issue #248).
fn explain_report(
    state: &AppState,
    headers: &HeaderMap,
    credential: Credential,
    resolved: Resolved,
    source_bytes: Option<usize>,
    probe: crate::engine::SourceProbe,
    image_path: String,
) -> crate::explain::Report {
    let ops = resolved.own_ops().to_vec();
    // Each referenced preset with the operations its
    // definition expands to. Definitions were validated at
    // boot; one that no longer parses (it cannot) is listed
    // by name alone rather than failing the report.
    let presets = resolved
        .groups
        .iter()
        .map(|name| {
            let entry_ops = state
                .presets
                .get(name)
                .and_then(|definition| {
                    crate::url::parse_default_ops(definition.as_str(), &state.presets).ok()
                })
                .unwrap_or_default();
            crate::explain::preset_entry(name, &entry_ops)
        })
        .collect();
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
    let source_url = resolved.written_source();
    let canonical = crate::url::canonical_form(&resolved.ops, &source_url);
    let (signed, key_id) = match credential {
        Credential::Signature(key_id) => (true, key_id),
        Credential::Unsafe => (false, None),
        // Never admitted here (see `admit_explained`).
        Credential::Sealed => (true, None),
    };
    // The expiry verdict reads the clock the admission read,
    // so the report never calls a URL it admitted expired.
    let (expiry, now) = match resolved.expiry {
        Some(expiry) => (Some(expiry.at), expiry.checked_at),
        None => (None, unix_seconds()),
    };
    crate::explain::Report::new(crate::explain::ReportInput {
        signed,
        key_id,
        expiry,
        now,
        source_url,
        source_root: resolved.root.map(|r| r.name),
        probe,
        // The DPR hint is read as the image path reads it, so
        // the forecast steps a hinted factor down as the image
        // request does (issue #529), and not at all with
        // client hints off.
        dpr_hint: dpr_hint(headers).filter(|_| state.client_hints),
        source_bytes,
        ops,
        merged_ops: resolved.ops,
        presets,
        canonical,
        cache_group: resolved.groups.into_iter().next(),
        image_path,
    })
}

/// The last stage of [`explain_handle`]: the report as the
/// client asked for it, JSON for a JSON `Accept` and the
/// HTML page otherwise, stored nowhere and served under a
/// policy that runs no script.
fn explain_response(report: &crate::explain::Report, accept: Option<&str>) -> Response {
    let (content_type, body) = if crate::explain::wants_json(accept) {
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
    stderr_line!("explain load error: {e}");
    let (status, code, message) = load_error_parts(e);
    error_response(status, code, message, accept)
}

/// How many shared-tier objects one purge request scans
/// inline (issue #368): the shared walk reads one entry
/// header per object (a bounded number at a time, issue
/// #462), so its inline cost must not grow with the whole
/// bucket. Nor does its listing (issue #709): the request
/// lists one page, of no more names than this. A walk that
/// outgrows the budget finishes in a background task,
/// waiting for one of the
/// [`crate::s3store::PURGE_WALKS_AT_ONCE`] turns with the
/// listing's continuation token, and logs its own count on
/// completion. Correctness does not wait
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
    log: &RequestLog,
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
    let log = log.clone();
    match pending {
        Some(pending) => {
            background.spawn_walk(async move {
                let rest = {
                    let _turn = shared.walk_turn().await;
                    shared.finish_purge(pending).await
                };
                let (event, log_target) = (purge.event, purge.log_target.clone());
                end_purge_apart(&shared, &tasks, purge, inline.complete && rest.complete).await;
                purge_log(&log, event, &log_target, Some(rest.removed)).await;
                let joined = lease.take_joined();
                walk_for_joined(&shared, &tasks, &log, lease, joined).await;
            });
        }
        None => {
            end_purge_apart(&shared, background, purge, inline.complete).await;
            let joined = lease.take_joined();
            if !joined.is_empty() {
                background.spawn_walk(async move {
                    walk_for_joined(&shared, &tasks, &log, lease, joined).await;
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
    log: &RequestLog,
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
            purge_log(log, event, &log_target, Some(swept.removed)).await;
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
/// ended only once the disk files of every purge of its
/// target in this process are deleted (issue #572): a
/// restart before then indexes them again, and the unended
/// tombstone keeps them misses. The purge waits for its own
/// while they make progress; when they stop, or one fails,
/// it ends here, and its tombstone is marked by the
/// tombstone refresh after they are gone.
async fn end_purge(
    shared: &crate::s3store::S3Store,
    purge: crate::s3store::SharedPurge,
    complete: bool,
) {
    if let Some(disk) = &purge.disk {
        disk.wait(PURGE_END_IDLE, PURGE_END_CAP).await;
    }
    if complete && let Some(object) = &purge.tombstone {
        shared.end_tombstone(object, purge.active.tombstone()).await;
    }
    drop(purge);
}

/// [`end_purge`], on a task of its own when the purge still
/// waits for disk deletions (issue #572): the walk lease, and
/// the purges that joined it, do not wait on the disk. A
/// purge with no tombstone to mark ended waits for nothing;
/// its deletions hold the target's tombstones from the moment
/// they were queued (see [`purge_disk_tier`]).
async fn end_purge_apart(
    shared: &Arc<crate::s3store::S3Store>,
    background: &Background,
    purge: crate::s3store::SharedPurge,
    complete: bool,
) {
    if !complete || purge.tombstone.is_none() {
        return;
    }
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
/// files' deletion. No disk slot is waited for. The deletions
/// hold the target's bucket tombstones at once: a per-URL
/// purge awaits the bucket before its walk, and a later purge
/// of the URL that ends meanwhile, with no deletions of its
/// own, must not mark its tombstone ended while these files
/// are there. Returns the count and the deletions queued.
fn purge_disk_tier(
    state: &AppState,
    target: &crate::cache::PurgeTarget,
) -> (usize, Option<crate::storage::DeletionWait>) {
    let Some(store) = &state.store else {
        return (0, None);
    };
    let (removed, disk) = store.purge_waiting(target);
    store.kick();
    if let Some(shared) = &state.shared_store {
        shared.hold_for_deletions(target, &disk);
    }
    (removed, Some(disk))
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
    let waited = disk.wait(DISK_WRITE_WAIT, PURGE_DELETE_CAP).await;
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
/// could hand out a copy a purge removed. Each refresh after
/// boot first marks ended the tombstones that waited for disk
/// deletions now done (issue #572).
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
            shared.end_held_purges().await;
            shared.refresh_tombstones(&guard).await;
        }
    });
}

/// One structured line per purge request (audit PL-8): the
/// destructive, secret-gated endpoint must be auditable. The
/// target is hashed; a raw target can carry signed URLs.
///
/// The line is on standard output when this returns, so
/// before the purge is answered (see [`RequestLog::event`]).
async fn purge_log(log: &RequestLog, event: &str, target: &str, removed: Option<usize>) {
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(target.as_bytes());
    let hash: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    let line = match removed {
        Some(n) => format!("{{\"event\":\"{event}\",\"target\":\"{hash}\",\"removed\":{n}}}"),
        None => format!("{{\"event\":\"{event}\",\"target\":\"{hash}\",\"rejected\":true}}"),
    };
    log.event(&line).await;
}

/// The current unix second, as the purge expiry check
/// reads it.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Why a purge whose signature has verified may not run, by
/// its expiry, at the unix second `now`. A purge URL is
/// valid strictly before its `exp:` second, the comparison
/// image and upload URLs use: from that second on it is
/// refused as expired. One with no expiry runs, unless
/// `require_expiry` (`CRABCROP_PURGE_REQUIRE_EXPIRY`) says
/// the server takes only expiring purge URLs.
fn purge_expiry_refusal(expiry: Option<u64>, now: u64, require_expiry: bool) -> Option<ErrorParts> {
    match expiry {
        Some(exp) if now >= exp => Some((
            StatusCode::FORBIDDEN,
            "signature_expired",
            "signature expired".to_string(),
        )),
        Some(_) => None,
        None if require_expiry => Some((
            StatusCode::FORBIDDEN,
            "expiry_required",
            "expiry required: this server accepts only purge URLs that carry an exp: segment"
                .to_string(),
        )),
        None => None,
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
/// 403 and purges nothing. The expiring form,
/// `POST /purge/<signature>/exp:<unix-seconds>/<path>`, signs
/// the `exp:` segment along with the path, and is a 403
/// from its expiry second on (see
/// [`crate::url::split_purge_remainder`]); with
/// `CRABCROP_PURGE_REQUIRE_EXPIRY`, a URL without an expiry
/// is a 403 as well. Idempotent:
/// purging an absent entry answers 200 with a count of 0.
async fn purge_handle(
    State(state): State<Arc<AppState>>,
    uri: axum::http::Uri,
    headers: HeaderMap,
) -> Response {
    // A purge leaves a log line: like an image request, it
    // waits while the request log's queue is full.
    state.request_log.room().await;
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
    let (signature, key_id, presented) = match parse_signature(&full) {
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
    // presented: for the expiring form the `exp:` segment is
    // part of the signed string, so a tampered or stripped
    // expiry already fails here as a bad signature.
    if !state
        .keyring
        .verify_purge(key_id.as_deref(), &signature, presented)
    {
        state.metrics.record_purge();
        purge_log(&state.request_log, "purge", presented, None).await;
        return error_response(
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature".to_string(),
            accept.as_deref(),
        );
    }
    // A leading `exp:` segment is the purge's own expiry,
    // split off the verified remainder as the upload
    // endpoint splits its own, and enforced before anything
    // is touched. What follows it is the target.
    let (expiry, target) = match crate::url::split_purge_remainder(presented) {
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
    if let Some((status, code, message)) =
        purge_expiry_refusal(expiry, unix_now(), state.purge_require_expiry)
    {
        state.metrics.record_purge();
        purge_log(&state.request_log, "purge", target, None).await;
        return error_response(status, code, message, accept.as_deref());
    }
    // A sealed target (issue #255) unseals here, the first
    // step of the chain, now that the purge signature has
    // verified over the target as presented and the purge's
    // own expiry is off its front: the steps below continue
    // with the decrypted remainder, so a purge by the sealed
    // spelling evicts the entry its plain spelling serves,
    // and the reverse. An `exp:` inside the payload is the
    // image's, which the key form drops. The decrypted
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
                &task_state.request_log,
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
        purge_log(
            &task_state.request_log,
            "purge",
            &task_target,
            Some(removed),
        )
        .await;
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
/// count of 0. The expiring form,
/// `POST /purge-group/<signature>/exp:<unix-seconds>/<group>`,
/// signs the `exp:` segment along with the group name, and
/// is a 403 from its expiry second on; the setting that
/// refuses a URL without one applies here too.
async fn purge_group_handle(
    State(state): State<Arc<AppState>>,
    uri: axum::http::Uri,
    headers: HeaderMap,
) -> Response {
    // As in `purge_handle`.
    state.request_log.room().await;
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
    let (signature, key_id, presented) = match parse_signature(&full) {
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
    // The group's shape is checked before its signature, as
    // it always was, so the purge's own expiry comes off the
    // front first (see `crate::url::split_purge_remainder`):
    // what follows it is the group name.
    let (expiry, group) = match crate::url::split_purge_remainder(presented) {
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
    // The signature covers the remainder as presented, the
    // `exp:` segment included.
    if !state
        .keyring
        .verify_purge_group(key_id.as_deref(), &signature, presented)
    {
        state.metrics.record_purge();
        purge_log(&state.request_log, "purge-group", &target, None).await;
        return error_response(
            StatusCode::FORBIDDEN,
            "bad_signature",
            "bad signature".to_string(),
            accept.as_deref(),
        );
    }
    if let Some((status, code, message)) =
        purge_expiry_refusal(expiry, unix_now(), state.purge_require_expiry)
    {
        state.metrics.record_purge();
        purge_log(&state.request_log, "purge-group", &target, None).await;
        return error_response(status, code, message, accept.as_deref());
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
                &task_state.request_log,
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
        purge_log(
            &task_state.request_log,
            "purge-group",
            &task_target,
            Some(removed),
        )
        .await;
        removed
    });
    let removed = task.await.unwrap_or(0);
    (StatusCode::OK, format!("purged {removed}\n")).into_response()
}

/// One structured line per upload request, the purge
/// precedent (audit PL-8): the write endpoint must be
/// auditable. The name is hashed for the same reason the
/// purge target is: a raw name is caller-chosen text.
/// Like a purge's, the line is on standard output before
/// the upload is answered.
async fn upload_log(log: &RequestLog, name: &str, outcome: &str) {
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(name.as_bytes());
    let hash: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    log.event(&format!(
        "{{\"event\":\"upload\",\"target\":\"{hash}\",\"outcome\":\"{outcome}\"}}"
    ))
    .await;
}

/// How long an upload body may take before its pace counts:
/// the body must arrive within this long plus one second for
/// every [`crate::loader::MIN_BODY_RATE`] bytes (64 KiB) it
/// has delivered (see [`upload_body_due`]).
pub const UPLOAD_BODY_GRACE: Duration = Duration::from_secs(10);

/// When an upload body whose read started at `started`, and
/// which has delivered `received` bytes, must have delivered
/// more: [`UPLOAD_BODY_GRACE`] plus one second per
/// [`crate::loader::MIN_BODY_RATE`] bytes, plus `waited`, the
/// time the server itself spent waiting for budget. A body
/// that keeps 64 KiB a second after the grace never meets
/// it, and one that trickles or stalls does, so a client
/// cannot hold its connection and its budget for longer than
/// the body's own size accounts for: a body at the 10 MiB
/// default cap is done within 10 + 160 seconds, plus the
/// budget wait.
fn upload_body_due(
    started: tokio::time::Instant,
    received: usize,
    waited: Duration,
) -> tokio::time::Instant {
    let paced = (received as u64).saturating_mul(1000) / crate::loader::MIN_BODY_RATE as u64;
    started + UPLOAD_BODY_GRACE + Duration::from_millis(paced) + waited
}

/// Read an upload body of at most `cap` bytes, charging it
/// to `charge` as it arrives (issue #403), in steps so a
/// trickling client does not touch the budget per frame.
/// [`LoadError::TooLarge`] past the cap,
/// [`LoadError::Network`] for an unreadable body,
/// [`LoadError::Timeout`] for a body that falls behind
/// [`upload_body_due`], and [`LoadError::Busy`] when the
/// request's wait for budget ran out.
async fn read_upload_body(
    mut body: axum::body::Body,
    cap: usize,
    charge: &crate::loader::SourceCharge,
) -> Result<Vec<u8>, LoadError> {
    use axum::body::HttpBody as _;
    const STEP: usize = 256 * 1024;
    let started = tokio::time::Instant::now();
    let mut waited = Duration::ZERO;
    let mut out = Vec::new();
    loop {
        let due = upload_body_due(started, out.len(), waited);
        let next = std::future::poll_fn(|cx| std::pin::Pin::new(&mut body).poll_frame(cx));
        let frame = match tokio::time::timeout_at(due, next).await {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(_) => return Err(LoadError::Timeout),
        };
        let frame = frame.map_err(|e| LoadError::Network(e.to_string()))?;
        let Ok(data) = frame.into_data() else {
            continue;
        };
        let received = out.len() + data.len();
        if received > cap {
            return Err(LoadError::TooLarge { limit: cap as u64 });
        }
        let step = received.div_ceil(STEP).saturating_mul(STEP);
        let waiting = tokio::time::Instant::now();
        charge.grow_to(step.min(cap).max(received)).await?;
        waited += waiting.elapsed();
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
async fn purge_source_everywhere(state: &Arc<AppState>, source: &str) -> usize {
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
                &task_state.request_log,
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
        purge_log(
            &task_state.request_log,
            "purge-source",
            &source,
            Some(removed),
        )
        .await;
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
async fn upload_handle(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
) -> Response {
    // An upload leaves a log line: like an image request,
    // it waits while the request log's queue is full, before
    // its body is read.
    state.request_log.room().await;
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
async fn finish_upload(
    state: &Arc<AppState>,
    outcome: UploadOutcome,
    accept: Option<&str>,
) -> Response {
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
            upload_log(&state.request_log, &target, outcome).await;
            error_response(status, code, message, accept)
        }
        UploadOutcome::Busy { target, message } => {
            state.metrics.record_upload();
            upload_log(&state.request_log, &target, "rejected:busy").await;
            busy_response(message, 1, accept)
        }
        UploadOutcome::Stored {
            name,
            source,
            replaced,
        } => {
            state.metrics.record_upload();
            upload_log(
                &state.request_log,
                &name,
                if replaced { "replaced" } else { "stored" },
            )
            .await;
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
        Err(LoadError::Timeout) => {
            return Err(UploadOutcome::rejected(
                name,
                "rejected:too-slow",
                (
                    StatusCode::REQUEST_TIMEOUT,
                    "upload_too_slow",
                    format!(
                        "upload body arrived too slowly: it may take {} seconds plus one second \
                         for every {} KiB it sends",
                        UPLOAD_BODY_GRACE.as_secs(),
                        crate::loader::MIN_BODY_RATE / 1024
                    ),
                ),
            ));
        }
        Err(e) => {
            stderr_line!("upload body read failed: {e}");
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
            stderr_line!("upload rejected: {e}");
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
            stderr_line!("upload failed: {cause}");
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
            stderr_line!("upload failed: {cause}");
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

/// How long a purge waits for its disk deletions in all
/// while they make progress (issue #572). It stops sooner, at
/// [`DISK_WRITE_WAIT`], once none of them finishes for that
/// long.
const PURGE_DELETE_CAP: std::time::Duration = std::time::Duration::from_secs(30);

/// How long the end of a purge, in the background, waits for
/// one of its disk deletions to finish (issue #572). When none
/// does for that long, the purge ends, and its bucket
/// tombstone is marked ended by the tombstone refresh once
/// they are done.
const PURGE_END_IDLE: std::time::Duration = std::time::Duration::from_secs(30);

/// How long the end of a purge waits for its disk deletions
/// in all while they make progress.
const PURGE_END_CAP: std::time::Duration = std::time::Duration::from_secs(600);

mod meta;
#[cfg(test)]
pub(in crate::server) use meta::meta_document_from;
/// Names the test module uses from meta: re-exported
/// only for test builds, where they are used.
#[cfg(test)]
pub(in crate::server) use meta::{
    AestheticMeta, AestheticOutcome, Annotation, FrameDecoder, compose_annotation,
    compose_moderation, f32_to_f16_bits, moderation_json, objects_crop_json, ocr_json,
    pack_embedding, score_list, segments_json, studio_json,
};
#[cfg(test)]
pub(in crate::server) use meta::{
    DocumentFrame, annotate_meta, detect_meta, embed_meta, meta_response, moderate_meta, ocr_meta,
};
pub use meta::{EmbeddingMeta, ModerationMeta, OcrMeta};
pub(in crate::server) use meta::{
    meta_cache_headers, wants_detections, wants_embeddings, wants_meta, wants_moderation,
    wants_ocr, wants_placeholders,
};

#[cfg(test)]
mod tests;
