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
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(test)]
use axum::extract::State;
use axum::{
    Router,
    http::{HeaderMap, StatusCode, header},
    response::Response,
    routing::get,
};

use crate::cache::Cache;
#[cfg(test)]
use crate::cache::{CachedImage, etag_for_bytes};
use crate::engine::Engine;
#[cfg(test)]
use crate::engine::EngineError;
#[cfg(test)]
use crate::loader::LoadError;
use crate::loader::Loader;
#[cfg(test)]
use crate::observe::RequestRecord;
use crate::observe::{Metrics, RequestLog};
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
#[cfg(test)]
pub(in crate::server) use response::{
    ErrorParts, busy_response, engine_error_response, finalize_validators, if_none_match_matches,
    load_error_response, parse_failure_parts, reserved_remainder_parts, respond,
};
pub(in crate::server) use response::{accept_of, error_response};
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
    purge_shared_tier, purge_source_everywhere, wait_for_disk_purge,
};
pub(in crate::server) use purge::{purge_group_handle, purge_handle};
mod upload;
pub use upload::UPLOAD_BODY_GRACE;
#[cfg(test)]
pub(in crate::server) use upload::read_upload_body;
pub(in crate::server) use upload::upload_handle;

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
    /// Processed-response cache, shared like the metrics. It
    /// is this server's own: a copy enters it only once this
    /// server's `loader` policy has admitted the copy's
    /// sources, so a memory hit runs none of the policy's
    /// checks (issue #780). A state that loads under another
    /// policy needs a cache of its own.
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
