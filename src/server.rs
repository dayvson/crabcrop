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
use std::time::Duration;
#[cfg(test)]
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
#[cfg(test)]
use crate::observe::RequestRecord;
use crate::observe::{Metrics, RequestLog};
use crate::url::parse_signature;
#[cfg(test)]
use crate::url::{ImageFormat, parse_operations_with_expiry};

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
    ErrorParts, SOURCE_BUDGET_BUSY, accept_of, busy_response, error_response,
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
#[cfg(test)]
pub(in crate::server) use admission::{
    Expiry, Negotiated, Refusal, Resolved, negotiate_format, resolve_request,
};
mod request;
pub(in crate::server) use request::handle;
#[cfg(test)]
pub(in crate::server) use request::{
    DPR_HINT_HEADERS, MAX_HINT_WIDTH, OwedRecord, SAVE_DATA_HEADER, VARY_WITH_HINTS,
    VARY_WITHOUT_HINTS, WIDTH_HINT_HEADERS, cache_key_for, quantize_width, save_data_hint,
    width_hint,
};
mod flight;
#[cfg(test)]
pub(in crate::server) use flight::MAX_FLIGHT_WAITS;
mod tiers;
#[cfg(test)]
pub(in crate::server) use tiers::{
    DISK_HIT, DISK_LOOKUP_BUDGET, DISK_WRITE_WAIT, DiskBudget, Sent, SharedWrite, begin_disk_write,
    finish_disk_write, lookup_tiers, lookup_tiers_within, memory_hit, populate_tiers,
    put_memory_tier, spawn_shared_put,
};
mod miss;
pub use miss::Fallback;
#[cfg(test)]
pub(in crate::server) use miss::{MissWork, ai_gate_applies, build_job};
mod explain;
pub(in crate::server) use explain::explain_handle;
#[cfg(test)]
pub(in crate::server) use explain::probe_settled;
mod purge;
pub use purge::start_purge_tombstones;
#[cfg(test)]
pub(in crate::server) use purge::{
    BegunPurge, SHARED_PURGE_INLINE_SCAN, begin_purge, purge_disk_tier, purge_expiry_refusal,
    purge_shared_tier, wait_for_disk_purge,
};
pub(in crate::server) use purge::{purge_group_handle, purge_handle, purge_source_everywhere};

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
pub(in crate::server) async fn read_upload_body(
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
pub(in crate::server) async fn upload_handle(
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
