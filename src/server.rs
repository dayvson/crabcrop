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

use crate::cache::{Cache, CachedImage};
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

/// In-flight miss coalescing (audit R2-09): the first
/// request to miss the cache for a key becomes its leader
/// and runs the fetch and the encode; concurrent requests
/// for the same key wait on the leader instead of each
/// fetching and encoding the same image, then re-check
/// the tiers. The map holds one watch sender per key in
/// flight; the leader's guard removes the key and wakes
/// the waiters on drop, so every exit path (success,
/// error, fallback) releases them exactly once.
#[derive(Default)]
pub struct Inflight {
    leaders:
        std::sync::Mutex<std::collections::HashMap<String, Arc<tokio::sync::watch::Sender<bool>>>>,
}

impl std::fmt::Debug for Inflight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inflight").finish_non_exhaustive()
    }
}

/// The caller's role for one cache key.
enum Flight {
    /// This request runs the miss path; dropping the
    /// guard wakes the waiters.
    Leader(LeaderGuard),
    /// Another request is already running the miss path
    /// for this key; wait on the channel, then re-check
    /// the tiers.
    Follower(tokio::sync::watch::Receiver<bool>),
}

/// Removes the key from the in-flight map and wakes its
/// waiters when dropped, on every exit path alike.
struct LeaderGuard {
    inflight: Arc<Inflight>,
    key: String,
    sender: Arc<tokio::sync::watch::Sender<bool>>,
}

impl Drop for LeaderGuard {
    fn drop(&mut self) {
        self.inflight.remove(&self.key, &self.sender);
        let _ = self.sender.send(true);
    }
}

impl Inflight {
    /// Claim leadership for `key`, or follow the request
    /// already leading it.
    fn lead(self: &Arc<Self>, key: &str) -> Flight {
        let mut leaders = self.leaders.lock().unwrap();
        if let Some(sender) = leaders.get(key) {
            return Flight::Follower(sender.subscribe());
        }
        let (sender, _) = tokio::sync::watch::channel(false);
        let sender = Arc::new(sender);
        leaders.insert(key.to_string(), sender.clone());
        Flight::Leader(LeaderGuard {
            inflight: self.clone(),
            key: key.to_string(),
            sender,
        })
    }

    /// Remove `key` if `sender` is still its leader (a
    /// waiter that already promoted itself must not lose
    /// its own claim to the previous leader's cleanup).
    fn remove(&self, key: &str, sender: &Arc<tokio::sync::watch::Sender<bool>>) {
        let mut leaders = self.leaders.lock().unwrap();
        if leaders.get(key).is_some_and(|s| Arc::ptr_eq(s, sender)) {
            leaders.remove(key);
        }
    }
}

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
    /// Request metrics, shared behind an `Arc` so clones agree.
    pub metrics: Arc<Metrics>,
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
    /// Bounds concurrent AI inference so it cannot starve the
    /// transform pipeline.
    pub ai_gate: Arc<tokio::sync::Semaphore>,
    /// Bounds concurrent engine runs (audit E-5): cache misses
    /// process on blocking threads under this gate, sized to
    /// the core count at boot.
    pub process_gate: Arc<tokio::sync::Semaphore>,
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
    /// Bounds concurrent video encodes: the subprocess is
    /// the server's most expensive and least predictable
    /// work, so it gets its own small gate (2 permits, like
    /// the AI gate) on top of the process gate.
    pub video_gate: Arc<tokio::sync::Semaphore>,
    /// Bounds concurrent animated WebP encodes to one
    /// (issue #231): the lossless animation encode is the
    /// slowest work the engine does, minutes for a modest
    /// file, so it gets a single-permit gate of its own
    /// instead of holding process permits while it runs.
    pub anim_gate: Arc<tokio::sync::Semaphore>,
    /// In-flight miss coalescing (audit R2-09): concurrent
    /// misses for one cache key share one fetch and one
    /// encode instead of stampeding the origin and the
    /// process gate.
    pub inflight: Arc<Inflight>,
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
async fn put_not_found() -> Response {
    (StatusCode::NOT_FOUND, "not found").into_response()
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

async fn metrics(State(state): State<AppState>) -> Response {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics.render(),
    )
        .into_response()
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
        return (
            StatusCode::BAD_REQUEST,
            "expected /explain/<signature>/<target>",
        )
            .into_response();
    };
    let full = format!("/{tail}");
    let (signature, key_id, rest) = match parse_signature(&full) {
        Ok(v) => v,
        Err(e) => return parse_failure(&e, accept.as_deref()),
    };
    if !state
        .keyring
        .verify(key_id.as_deref(), &signature, rest, state.allow_unsafe)
    {
        return (StatusCode::FORBIDDEN, "bad signature").into_response();
    }
    // The same domain separation as the image path: a
    // remainder may not begin with a reserved segment.
    if matches!(
        rest.split('/').next(),
        Some("purge" | "purge-group" | "upload")
    ) {
        return (
            StatusCode::BAD_REQUEST,
            "an image URL may not begin with a purge or upload segment",
        )
            .into_response();
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
        return (StatusCode::FORBIDDEN, "signature expired").into_response();
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
    let bytes = match state.loader.load_media(&fetch_url, admission).await {
        Ok(bytes) => bytes,
        Err(e) => return explain_load_error(e),
    };
    let probe = crate::engine::probe_source(&bytes);

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
        source_bytes: bytes.len(),
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
fn explain_load_error(e: LoadError) -> Response {
    eprintln!("explain load error: {e}");
    let (status, message) = match e {
        LoadError::Forbidden(_) => (
            StatusCode::FORBIDDEN,
            "source refused by policy".to_string(),
        ),
        LoadError::BadStatus(code) => (
            StatusCode::BAD_GATEWAY,
            format!("source returned status {code}"),
        ),
        LoadError::Network(_) => (StatusCode::BAD_GATEWAY, "source fetch failed".to_string()),
        LoadError::WrongContentType(_) => (
            StatusCode::BAD_GATEWAY,
            "source did not return an image".to_string(),
        ),
        LoadError::Timeout => (StatusCode::GATEWAY_TIMEOUT, "source timed out".to_string()),
        LoadError::TooLarge { limit } => (
            StatusCode::BAD_GATEWAY,
            format!("source exceeds {limit} bytes"),
        ),
    };
    (status, message).into_response()
}

/// `POST /purge/<signature>/<path>`: invalidate every stored
/// variant of one result. The signature is an HMAC over the
/// path under a purge subkey derived from the secret (see
/// [`crate::url::purge_key`]), so only the secret holder can
/// purge and an image signature can never be replayed here; a
/// bad signature is a 403 and purges nothing. Idempotent:
/// purging an absent entry answers 200 with a count of 0.
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

async fn purge_handle(State(state): State<AppState>, uri: axum::http::Uri) -> Response {
    let Some(tail) = uri.path().strip_prefix("/purge/") else {
        return (
            StatusCode::BAD_REQUEST,
            "expected /purge/<signature>/<target>",
        )
            .into_response();
    };
    let full = format!("/{tail}");
    let (signature, key_id, target) = match parse_signature(&full) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    if !state
        .keyring
        .verify_purge(key_id.as_deref(), &signature, target)
    {
        state.metrics.record_purge();
        purge_log("purge", target, None);
        return (StatusCode::FORBIDDEN, "bad signature").into_response();
    }
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
    let mut removed = state.cache.purge_matching(&key_form);
    if let Some(store) = &state.store {
        removed += store.purge_matching(&key_form);
    }
    if let Some(shared) = &state.shared_store {
        // The shared tier deletes the current epoch's
        // enumerable variants directly by object name
        // (audit R2-24); the fingerprint-free form drives
        // its matching walk, which also reaches older
        // epochs and hint-variant spellings.
        let epoch_form = format!("{key_form}#cfg{}", state.config_fingerprint);
        removed += shared.purge_matching(&key_form, &epoch_form).await;
    }
    state.metrics.record_purge();
    purge_log("purge", target, Some(removed));
    (StatusCode::OK, format!("purged {removed}\n")).into_response()
}

/// `POST /purge-group/<signature>/<group>`: invalidate every
/// stored entry in one cache group (RFC 9875), across both
/// tiers. The signature is verified under the purge-group
/// subkey, its own domain beside the purge subkey, over the
/// group name, so neither an image signature nor a per-URL
/// purge signature can be replayed here (audit R2-14). A
/// bad signature is a 403 and removes nothing; an unknown
/// group answers 200 with a count of 0.
async fn purge_group_handle(State(state): State<AppState>, uri: axum::http::Uri) -> Response {
    let Some(tail) = uri.path().strip_prefix("/purge-group/") else {
        return (
            StatusCode::BAD_REQUEST,
            "expected /purge-group/<signature>/<group>",
        )
            .into_response();
    };
    let full = format!("/{tail}");
    let (signature, key_id, group) = match parse_signature(&full) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    if group.is_empty() || group.contains('/') {
        return (StatusCode::BAD_REQUEST, "invalid purge group name").into_response();
    }
    let target = format!("purge-group/{group}");
    if !state
        .keyring
        .verify_purge_group(key_id.as_deref(), &signature, group)
    {
        state.metrics.record_purge();
        purge_log("purge-group", &target, None);
        return (StatusCode::FORBIDDEN, "bad signature").into_response();
    }
    let mut removed = state.cache.purge_group(group);
    if let Some(store) = &state.store {
        removed += store.purge_group(group);
    }
    if let Some(shared) = &state.shared_store {
        removed += shared.purge_group(group).await;
    }
    state.metrics.record_purge();
    purge_log("purge-group", &target, Some(removed));
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
/// three tiers; returns the total removed. The upload
/// overwrite path (audit R2-07) shares this with any
/// future source-replacement flow.
async fn purge_source_everywhere(state: &AppState, source: &str) -> usize {
    let mut removed = state.cache.purge_source(source);
    if let Some(store) = &state.store {
        removed += store.purge_source(source);
    }
    if let Some(shared) = &state.shared_store {
        removed += shared.purge_source(source).await;
    }
    removed
}

async fn upload_handle(State(state): State<AppState>, req: axum::extract::Request) -> Response {
    let Some(upload) = &state.upload else {
        return (StatusCode::NOT_FOUND, "uploads are not configured").into_response();
    };
    let (parts, body) = req.into_parts();
    let Some(tail) = parts.uri.path().strip_prefix("/upload/") else {
        return (
            StatusCode::BAD_REQUEST,
            "expected /upload/<signature>/<name>",
        )
            .into_response();
    };
    let full = format!("/{tail}");
    let (signature, key_id, rest) = match parse_signature(&full) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
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
        return (StatusCode::FORBIDDEN, "bad signature").into_response();
    }
    let (expiry, name) = match crate::url::split_upload_remainder(rest) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
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
            return (StatusCode::FORBIDDEN, "signature expired").into_response();
        }
    }
    if let Err(e) = crate::upload::validate_upload_name(name) {
        state.metrics.record_upload();
        upload_log(name, "rejected:bad-name");
        return (StatusCode::BAD_REQUEST, e).into_response();
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
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("upload exceeds the {}-byte limit", upload.max_bytes),
            )
                .into_response();
        }
    };
    if bytes.is_empty() {
        state.metrics.record_upload();
        upload_log(name, "rejected:not-an-image");
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "upload body is empty: an image is required",
        )
            .into_response();
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
        return (
            StatusCode::CONFLICT,
            "an object with this name already exists",
        )
            .into_response();
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
            return (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "upload is not an image this server can decode",
            )
                .into_response();
        }
        UploadValidation::Busy => {
            state.metrics.record_upload();
            upload_log(name, "rejected:busy");
            return busy_response("server busy: too many images processing at once", 1);
        }
        UploadValidation::Panicked => {
            state.metrics.record_upload();
            upload_log(name, "rejected:validation-failed");
            return (StatusCode::INTERNAL_SERVER_ERROR, "image validation failed").into_response();
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
        return (StatusCode::UNSUPPORTED_MEDIA_TYPE, e).into_response();
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
                let removed = purge_source_everywhere(&state, &source).await;
                purge_log("purge-source", &source, Some(removed));
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
            (
                StatusCode::CONFLICT,
                "an object with this name already exists",
            )
                .into_response()
        }
        Err(crate::upload::UploadError::Backend(cause)) => {
            eprintln!("upload failed: {cause}");
            state.metrics.record_upload();
            upload_log(name, "rejected:store-failed");
            (StatusCode::BAD_GATEWAY, "the upload store is unavailable").into_response()
        }
        Err(crate::upload::UploadError::Io(cause)) => {
            eprintln!("upload failed: {cause}");
            state.metrics.record_upload();
            upload_log(name, "rejected:store-failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "the upload could not be stored",
            )
                .into_response()
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
/// An owned bundle for one engine run: everything
/// [`crate::engine::ProcessContext`] borrows, owned so the work
/// can move to a blocking thread (audit E-5).
#[derive(Clone)]
struct ProcessJob {
    save_data: bool,
    preserve_metadata: bool,
    ops: Vec<crate::url::Operation>,
    preferred: Option<ImageFormat>,
    /// Watermark images as (URL, fetched bytes) pairs: the
    /// decode happens inside the gated blocking job, not in
    /// the async handler (audit R2-16).
    watermark_sources: Vec<(String, Vec<u8>)>,
    placeholders: bool,
    mask_provider: Option<Arc<dyn crate::ai::MaskProvider>>,
    upscale_provider: Option<Arc<dyn crate::ai::UpscaleProvider>>,
    inpaint_provider: Option<Arc<dyn crate::ai::InpaintProvider>>,
    segment_provider: Option<Arc<dyn crate::ai::SegmentProvider>>,
    face_provider: Option<Arc<dyn crate::ai::FaceProvider>>,
    detect_provider: Option<Arc<dyn crate::ai::DetectionProvider>>,
    declared_detect_providers: Vec<Arc<dyn crate::ai::DetectionProvider>>,
    focal_provider: Option<Arc<dyn crate::ai::FocalProvider>>,
    text_renderer: Option<Arc<dyn crate::text::TextRenderer>>,
    video_encoder: Option<Arc<crate::video::VideoEncoder>>,
    frame_extractor: Option<Arc<crate::frames::VideoFrameExtractor>>,
    pdf_renderer: Option<Arc<crate::pdf::PdfRenderer>>,
    dpr: Option<f32>,
    ssim_target: f64,
    brand_color: Option<[u8; 3]>,
    avif_speed: u8,
    jxl_speed: jixel::Speed,
}

/// Why a gated engine run produced no image.
#[derive(Debug)]
enum ProcessFailure {
    /// The engine's own error; fallback logic applies.
    Engine(EngineError),
    /// The process gate stayed saturated past its wait:
    /// the request is shed with a 503.
    Busy,
    /// The engine panicked. The panic is contained at this
    /// seam (audit E-8) and surfaces as a 500.
    Panicked,
}

/// How long a cache miss may wait for a processing slot
/// before the server sheds it with a 503.
const PROCESS_GATE_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// How long an AI-backed request may wait for an inference
/// slot before the server sheds it with a 503 (audit PL-14).
const AI_GATE_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// How long a video request may wait for an encode slot
/// before the server sheds it with a 503. Encodes run for
/// seconds behind a 2-permit gate; the wait is the AI
/// gate's, and a shed request can retry into a drained
/// queue.
const VIDEO_GATE_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// How long an animated WebP request may wait for the
/// single animation encode slot before the server sheds
/// it with a 503 (issue #231). The AI and video gates'
/// wait: long enough to absorb a hiccup, far shorter
/// than queueing behind a minutes-long encode.
const ANIM_GATE_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Whether this miss will encode an animated WebP and so
/// must take the animation gate (issue #231): the source
/// is an animated WebP (a container probe on the loaded
/// bytes, no decode) and the output resolves to WebP.
/// The explicit-format half is conservative: an explicit
/// GIF, still, or video format never reaches the animated
/// WebP encoder, and with no explicit format negotiation
/// keeps an animated WebP source at WebP; if some header
/// combination resolved to a still instead, the permit is
/// held for one fast render, which is harmless.
fn wants_animation_gate(bytes: &[u8], directives: &crate::url::RequestDirectives) -> bool {
    crate::engine::is_animated_webp(bytes)
        && matches!(directives.explicit_format, None | Some(ImageFormat::WebP))
}

/// Run the engine under the process gate on a blocking
/// thread, so CPU-bound work neither stalls the async
/// workers nor multiplies unboundedly under a burst of
/// cache misses. Cache hits never reach this function.
async fn run_process(
    state: &AppState,
    job: &ProcessJob,
    bytes: &[u8],
) -> Result<crate::engine::ProcessedImage, ProcessFailure> {
    let permit = match tokio::time::timeout(
        PROCESS_GATE_WAIT,
        state.process_gate.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        _ => return Err(ProcessFailure::Busy),
    };
    let job = job.clone();
    let bytes = bytes.to_vec();
    let engine = state.engine;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let declared_refs: Vec<&dyn crate::ai::DetectionProvider> = job
            .declared_detect_providers
            .iter()
            .map(|p| p.as_ref())
            .collect();
        // Watermark images decode here, inside the gated
        // blocking job, rather than inline in the async
        // handler: the decode is CPU work a burst of
        // watermarked requests must not run ungated on the
        // runtime workers, and a decoder panic on crafted
        // watermark bytes is contained by the same
        // catch_unwind as the engine run (audit R2-16).
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut watermarks = crate::filters::WatermarkImages::new();
            for (url, wm_bytes) in &job.watermark_sources {
                let img = crate::engine::decode_image(wm_bytes)?;
                watermarks.insert(url.clone(), img);
            }
            let ctx = crate::engine::ProcessContext {
                preferred: job.preferred,
                watermarks: Some(&watermarks),
                placeholders: job.placeholders,
                mask_provider: job.mask_provider.as_deref(),
                upscale_provider: job.upscale_provider.as_deref(),
                inpaint_provider: job.inpaint_provider.as_deref(),
                segment_provider: job.segment_provider.as_deref(),
                face_provider: job.face_provider.as_deref(),
                detect_provider: job.detect_provider.as_deref(),
                declared_detect_providers: &declared_refs,
                focal_provider: job.focal_provider.as_deref(),
                video_encoder: job.video_encoder.as_deref(),
                frame_extractor: job.frame_extractor.as_deref(),
                pdf_renderer: job.pdf_renderer.as_deref(),
                save_data: job.save_data,
                preserve_metadata: job.preserve_metadata,
                text_renderer: job.text_renderer.as_deref(),
                dpr: job.dpr,
                ssim_target: job.ssim_target,
                brand_color: job.brand_color,
                avif_speed: job.avif_speed,
                jxl_speed: job.jxl_speed,
            };
            engine.process_preferred(&bytes, &job.ops, ctx)
        })) {
            Ok(Ok(processed)) => Ok(processed),
            Ok(Err(e)) => Err(ProcessFailure::Engine(e)),
            Err(_) => Err(ProcessFailure::Panicked),
        }
    })
    .await
    .unwrap_or(Err(ProcessFailure::Panicked))
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

    // Grammar failures answer 400 with the parser's own
    // message as the body: it names only what the caller
    // sent, and it is the difference between a debuggable
    // rejection and an empty one.
    let (signature, key_id, rest) = match parse_signature(&full) {
        Ok(v) => v,
        Err(e) => {
            return (parse_failure(&e, accept), record);
        }
    };
    let verify_start = Instant::now();
    let verified = state
        .keyring
        .verify(key_id.as_deref(), &signature, rest, state.allow_unsafe);
    record.verify = verify_start.elapsed();
    if !verified {
        return (
            (StatusCode::FORBIDDEN, "bad signature").into_response(),
            record,
        );
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
        return (
            (
                StatusCode::BAD_REQUEST,
                "an image URL may not begin with a purge or upload segment",
            )
                .into_response(),
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
            return (parse_failure(&e, accept), record);
        }
    };
    let (ops, source_url, expiry, max_age) = match parse_operations_with_expiry(rest) {
        Ok(v) => v,
        Err(e) => {
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
                (StatusCode::FORBIDDEN, "signature expired").into_response(),
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
    let placeholders = meta && wants_placeholders(accept);
    let wants_video = directives.explicit_format.is_some_and(|f| f.is_video());
    // Meta describes an image; a video result is not one.
    // Refuse specifically rather than describing encoded
    // video bytes as if they were a still image.
    if meta && wants_video {
        return (
            (
                StatusCode::BAD_REQUEST,
                format!(
                    "meta is not available for video output: format({}) serves encoded video, not an image",
                    crate::url::format_text(directives.explicit_format.unwrap())
                ),
            )
                .into_response(),
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
        return (
            (
                StatusCode::BAD_REQUEST,
                "meta is not available for debug output: debug:focal serves a diagnostic image, not a transform result"
                    .to_string(),
            )
                .into_response(),
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
    let etag = etag_for(&cache_key);

    // The verified path is content-addressed: a stored response can
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
    };
    // Singleflight (audit R2-09): the first miss for a key
    // leads (one fetch, one encode, under the gates below);
    // concurrent requests for the same key wait for the
    // leader, then re-check the tiers and serve its entry.
    // A leader whose result is not cached (an error, a
    // fallback) releases its waiters to take the miss path
    // themselves, one at a time, so a failing origin is
    // retried by the waiters rather than masked. Waits are
    // capped: a key whose leaders keep failing sheds its
    // remaining waiters with the busy response instead of
    // queueing them forever.
    let mut waits = 0;
    loop {
        if !bypass && let Some((hit, cache_status)) = lookup_tiers(state, &cache_key, &groups).await
        {
            record.cache_hit = true;
            record.bytes_out = hit.bytes.len();
            let mut response = respond(hit.format, hit.quality, hit.bytes, max_age);
            attach_cache_headers(&mut response, cache_status, &groups);
            finalize_validators(&mut response, &etag, if_none_match);
            return (response, record);
        }
        if bypass {
            return serve_miss(state, work, record).await;
        }
        match state.inflight.lead(&cache_key) {
            Flight::Leader(guard) => {
                let (mut response, record) = serve_miss(state, work, record).await;
                drop(guard);
                finalize_validators(&mut response, &etag, if_none_match);
                return (response, record);
            }
            Flight::Follower(mut changed) => {
                if waits >= MAX_FLIGHT_WAITS {
                    return (
                        busy_response("server busy: too many images processing at once", 1),
                        record,
                    );
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

/// How many times one request may wait on another
/// request's in-flight miss for the same key before it
/// sheds with a 503 (audit R2-09).
const MAX_FLIGHT_WAITS: u32 = 3;

/// The tier lookup chain for one key: memory, then disk
/// (a hit promotes into memory), then the shared tier (a
/// hit promotes into both local tiers). On any hit, the
/// request's cache groups are unioned into the stored
/// entry (audit R2-15): provenance accumulates, so an
/// entry first stored through a group-free spelling is
/// still removed by a later group purge once any grouped
/// spelling of the same URL has been served. Returns the
/// image and the `Cache-Status` value for the tier that
/// served it.
async fn lookup_tiers(
    state: &AppState,
    cache_key: &str,
    groups: &[String],
) -> Option<(CachedImage, &'static str)> {
    if let Some(hit) = state.cache.get(cache_key) {
        if !groups.is_empty() {
            state.cache.merge_groups(cache_key, groups);
        }
        return Some((hit, "crabcrop; hit"));
    }
    if let Some(store) = &state.store
        && let Some(mut image) = store.get(cache_key)
    {
        if !groups.is_empty() && store.merge_groups(cache_key, groups) {
            image.groups = crate::cache::union_groups(&image.groups, groups);
        }
        state.cache.put(cache_key.to_string(), image.clone());
        return Some((image, "crabcrop; hit; detail=\"disk\""));
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
                let shared = shared.clone();
                let key = cache_key.to_string();
                let entry = image.clone();
                tokio::spawn(async move { shared.put(&key, &entry).await });
            }
        }
        state.cache.put(cache_key.to_string(), image.clone());
        if let Some(store) = &state.store {
            store.put(cache_key, &image);
        }
        return Some((image, "crabcrop; hit; detail=\"shared\""));
    }
    None
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
}

/// The cache-miss path: load the source, resolve
/// watermarks, run the engine under the gates, populate
/// the tiers, and respond (a meta document when asked).
/// Runs at most once per key at a time, led by one
/// request while its duplicates wait (see `serve`).
async fn serve_miss(
    state: &AppState,
    work: MissWork,
    mut record: RequestRecord,
) -> (Response, RequestRecord) {
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
                    return (load_error_response(e), record);
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
                return (load_error_response(e), record);
            }
        };
        watermark_sources.push((wm_url.to_string(), wm_bytes));
    }

    record.process_ran = true;
    let process_start = Instant::now();
    // AI inference is gated process-wide: a burst of removebg
    // requests must not starve the transform pipeline.
    let _ai_permit = if directives.uses_ai() {
        // The wait is bounded (audit PL-14): a burst of AI
        // requests sheds with a 503 instead of parking
        // unbounded tasks on the gate.
        match tokio::time::timeout(AI_GATE_WAIT, state.ai_gate.clone().acquire_owned()).await {
            Ok(Ok(permit)) => Some(permit),
            _ => {
                return (
                    busy_response("server busy: AI processing queue is full", 5),
                    record,
                );
            }
        }
    } else {
        None
    };
    // Video work sits behind its own gate, on top of the
    // process gate inside `run_process`: the external
    // encoder is the server's most expensive and least
    // predictable work (seconds per job, a subprocess), so
    // a burst of video requests sheds here instead of
    // queueing unboundedly behind the transform pipeline.
    // Frame extraction (issue #214) takes the same gate:
    // it runs the same class of subprocess. Page
    // rendering (issue #215) takes it too: it is the
    // external-tool gate in everything but name, and a
    // second gate for the same class of work would only
    // split the protection.
    let _video_permit = if wants_video || wants_frame || wants_page {
        match tokio::time::timeout(VIDEO_GATE_WAIT, state.video_gate.clone().acquire_owned()).await
        {
            Ok(Ok(permit)) => Some(permit),
            _ => {
                return (
                    busy_response("server busy: video processing queue is full", 5),
                    record,
                );
            }
        }
    } else {
        None
    };
    // Animated WebP encodes sit behind a single-permit
    // gate of their own (issue #231): one encode can run
    // for minutes, so without this gate duplicate or
    // distinct animated renders stack encodes until they
    // hold every process permit and unrelated stills shed.
    // Behind the gate, a second animated request sheds
    // here with an explanation instead of stacking, and
    // the still pool keeps its permits throughout.
    let _anim_permit = if wants_animation_gate(&bytes, &directives) {
        match tokio::time::timeout(ANIM_GATE_WAIT, state.anim_gate.clone().acquire_owned()).await {
            Ok(Ok(permit)) => Some(permit),
            _ => {
                return (
                    busy_response("server busy: animation encoding queue is full", 5),
                    record,
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
        dpr: dpr_hint,
        ssim_target: state.ssim_target,
        brand_color: state.brand_color,
        avif_speed: state.avif_speed,
        jxl_speed: state.jxl_speed,
    };
    let processed = match run_process(state, &job, &bytes).await {
        Ok(p) => p,
        Err(ProcessFailure::Busy) => {
            return (
                busy_response("server busy: too many images processing at once", 1),
                record,
            );
        }
        Err(ProcessFailure::Panicked) => {
            record.fail = FailStage::Engine;
            return (
                (StatusCode::INTERNAL_SERVER_ERROR, "image processing failed").into_response(),
                record,
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
                    Some(fb) => run_process(state, &job, &fb).await.ok(),
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
                    return (engine_error_response(e), record);
                }
            }
        }
    };
    record.process = process_start.elapsed();

    // Only successful responses are cached; errors never are.
    // Meta responses are not image entries and are not cached.
    // Fallback responses are not cached either: the placeholder
    // must never be stored as if it were the real image, so the
    // origin gets retried on the next request. Debug responses
    // are not cached: the diagnostic is not the transform
    // result its key would claim.
    if !meta && !used_fallback && !debug {
        let entry = CachedImage {
            bytes: processed.bytes.clone(),
            format: processed.format,
            quality: processed.quality,
            groups: groups.clone(),
        };
        state.cache.put(cache_key.clone(), entry.clone());
        if let Some(store) = &state.store {
            store.put(&cache_key, &entry);
        }
        if let Some(shared) = &state.shared_store {
            // Write-through off the response path: the local
            // tiers already hold the entry, and the shared
            // tier is best-effort by design.
            let shared = shared.clone();
            let key = cache_key.clone();
            tokio::spawn(async move { shared.put(&key, &entry).await });
        }
    }
    record.bytes_out = processed.bytes.len();
    // When this request already holds an AI permit (an AI
    // transform that is also a meta request), the annotation
    // paths run under it instead of queueing for a second
    // permit that cannot free until this request returns
    // (audit R2-29).
    let ai_gated = _ai_permit.is_some();
    let annotation = if meta {
        annotate_meta(state, &processed, ai_gated).await
    } else {
        None
    };
    let detections = if meta && wants_detections(accept) {
        detect_meta(state, &processed, ai_gated).await
    } else {
        None
    };
    let moderation = if meta && wants_moderation(accept) {
        Some(moderate_meta(state, &processed, ai_gated).await)
    } else {
        None
    };
    // The embedding run (issue #172) serves two
    // fields: the embedding itself, when the client
    // opted in, and the aesthetic score, whenever a
    // head is configured. One run feeds both.
    let want_embedding = meta && wants_embeddings(accept);
    let embedding_run = if meta && (want_embedding || state.aesthetic.is_some()) {
        Some(embed_meta(state, &processed, ai_gated).await)
    } else {
        None
    };
    let aesthetic = compose_aesthetic(state, embedding_run.as_ref());
    let ocr = if meta && wants_ocr(accept) {
        Some(ocr_meta(state, &processed, ai_gated).await)
    } else {
        None
    };
    let mut response = if meta {
        meta_response(
            &processed,
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
            processed.bytes,
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
    (response, record)
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

/// Whether the client asked for the meta response: an explicit
/// `application/json` in `Accept` that is not refused with q=0.
fn wants_meta(accept: Option<&str>) -> bool {
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
fn wants_placeholders(accept: Option<&str>) -> bool {
    wants_meta_param(accept, "placeholders")
}

/// Whether the accepted `application/json` entry asks for
/// detection boxes: a `detections` parameter, same rule as
/// `placeholders`.
fn wants_detections(accept: Option<&str>) -> bool {
    wants_meta_param(accept, "detections")
}

/// Whether the accepted `application/json` entry asks for
/// moderation verdicts: a `moderation` parameter, same
/// rule as `placeholders`.
fn wants_moderation(accept: Option<&str>) -> bool {
    wants_meta_param(accept, "moderation")
}

/// Whether the accepted `application/json` entry asks
/// for the image embedding: an `embeddings` parameter,
/// same rule as `placeholders`.
fn wants_embeddings(accept: Option<&str>) -> bool {
    wants_meta_param(accept, "embeddings")
}

/// Whether the accepted `application/json` entry asks
/// for OCR text: an `ocr` parameter, same rule as
/// `placeholders`.
fn wants_ocr(accept: Option<&str>) -> bool {
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

/// The meta response: what the pipeline produced, as JSON,
/// without the image bytes.
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

/// The zero-shot logit scale the composition applies to
/// raw scores before the softmax.
const TAG_LOGIT_SCALE: f32 = 100.0;

/// Turn raw per-label scores into the annotation:
/// probabilities are the softmax of `TAG_LOGIT_SCALE *
/// score`; tags are the labels at or above the threshold,
/// best first, at most 10, with scores rounded to 3
/// decimals; alt text is the top 3 labels joined with
/// commas, first letter capitalized.
fn compose_annotation(scores: &[crate::ai::TagScore], threshold: f32) -> Annotation {
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
async fn annotate_meta(
    state: &AppState,
    processed: &crate::engine::ProcessedImage,
    ai_gated: bool,
) -> Option<Annotation> {
    let tagger = state.tagger.as_ref()?;
    let _permit = if ai_gated {
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
    match tokio::task::spawn_blocking(move || tagger.scores(&img)).await {
        Ok(Ok(scores)) => Some(compose_annotation(&scores, state.tag_threshold)),
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
async fn detect_meta(
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
    let _permit = if ai_gated {
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
    let mut aggregated: Vec<(String, crate::ai::Detection)> = Vec::new();
    let mut any_answered = false;
    for (name, provider) in providers {
        let provider_img = img.clone();
        match tokio::task::spawn_blocking(move || provider.detect(&provider_img)).await {
            Ok(Ok(detections)) => {
                any_answered = true;
                aggregated.extend(detections.into_iter().map(|d| (name.clone(), d)));
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
fn compose_moderation(report: &crate::ai::ModerationScores) -> crate::ai::ModerationScores {
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
async fn moderate_meta(
    state: &AppState,
    processed: &crate::engine::ProcessedImage,
    ai_gated: bool,
) -> ModerationMeta {
    let Some(provider) = state.moderation.as_ref() else {
        return ModerationMeta::Unavailable("not_configured");
    };
    let _permit = if ai_gated {
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
    match tokio::task::spawn_blocking(move || provider.moderate(&img)).await {
        Ok(Ok(report)) => ModerationMeta::Report(compose_moderation(&report)),
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

/// Render a score list the way meta carries it:
/// `[{"label":...,"score":...}, ...]`.
fn score_list_json(scores: &[crate::ai::TagScore]) -> String {
    let items: Vec<String> = scores
        .iter()
        .map(|s| {
            format!(
                "{{\"label\":{},\"score\":{}}}",
                serde_json::to_string(&s.label).unwrap_or_default(),
                s.score
            )
        })
        .collect();
    format!("[{}]", items.join(","))
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
async fn embed_meta(
    state: &AppState,
    processed: &crate::engine::ProcessedImage,
    ai_gated: bool,
) -> EmbeddingMeta {
    let Some(provider) = state.embeddings.as_ref() else {
        return EmbeddingMeta::Unavailable("not_configured");
    };
    let _permit = if ai_gated {
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
    match tokio::task::spawn_blocking(move || provider.embed(&img)).await {
        Ok(Ok(embedding)) => EmbeddingMeta::Report(embedding),
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
async fn ocr_meta(
    state: &AppState,
    processed: &crate::engine::ProcessedImage,
    ai_gated: bool,
) -> OcrMeta {
    let Some(provider) = state.ocr.as_ref() else {
        return OcrMeta::Unavailable("not_configured");
    };
    let _permit = if ai_gated {
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
    match tokio::task::spawn_blocking(move || provider.recognize(&img)).await {
        Ok(Ok(report)) => OcrMeta::Report(compose_ocr(&report)),
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
fn compose_aesthetic(state: &AppState, run: Option<&EmbeddingMeta>) -> Option<AestheticMeta> {
    let head = state.aesthetic.as_ref()?;
    let (model, outcome) = match run {
        Some(EmbeddingMeta::Report(embedding)) => {
            let outcome = match head.score(&embedding.vector) {
                Some(score) => AestheticOutcome::Score((score * 1000.0).round() / 1000.0),
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
fn pack_embedding(vector: &[f32]) -> String {
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
fn f32_to_f16_bits(value: f32) -> u16 {
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

fn meta_response(
    processed: &crate::engine::ProcessedImage,
    annotation: Option<&Annotation>,
    detections: Option<&[(String, crate::ai::Detection)]>,
    moderation: Option<&ModerationMeta>,
    embedding: Option<&EmbeddingMeta>,
    aesthetic: Option<&AestheticMeta>,
    ocr: Option<&OcrMeta>,
) -> Response {
    let mut body = format!(
        "{{\"width\":{},\"height\":{},\"format\":\"{}\",\"bytes\":{},\"source_width\":{},\"source_height\":{}",
        processed.width,
        processed.height,
        format_name(processed.format),
        processed.bytes.len(),
        processed.source_width,
        processed.source_height,
    );
    if let Some(a) = annotation {
        let tags: Vec<String> = a
            .tags
            .iter()
            .map(|t| {
                format!(
                    "{{\"label\":{},\"score\":{}}}",
                    serde_json::to_string(&t.label).unwrap_or_default(),
                    t.score
                )
            })
            .collect();
        body.push_str(&format!(
            ",\"tags\":[{}],\"alt_text\":{}",
            tags.join(","),
            serde_json::to_string(&a.alt_text).unwrap_or_default()
        ));
    }
    if let Some(detections) = detections {
        // Each box keeps its original fields and gains a
        // trailing `detector`: the provider that produced
        // it (`builtin`, or a declared detector's name from
        // `CRABCROP_DETECTORS_FILE`, issue #166).
        let boxes: Vec<String> = detections
            .iter()
            .map(|(source, d)| {
                format!(
                    "{{\"label\":{},\"score\":{},\"x\":{},\"y\":{},\"w\":{},\"h\":{},\"detector\":{}}}",
                    serde_json::to_string(&d.label).unwrap_or_default(),
                    d.score,
                    d.x,
                    d.y,
                    d.w,
                    d.h,
                    serde_json::to_string(source).unwrap_or_default()
                )
            })
            .collect();
        body.push_str(&format!(",\"detections\":[{}]", boxes.join(",")));
    }
    if let Some(moderation) = moderation {
        // The moderation field (issue #171): a report
        // carries the model identity, the verdict scores
        // when a classifier ran, and the zero-shot
        // signals (marked review-only: leads, never
        // verdicts) when the CLIP fallback ran. An
        // unavailable moderation is explicit too, since
        // the client opted in by name.
        match moderation {
            ModerationMeta::Report(report) => {
                body.push_str(&format!(
                    ",\"moderation\":{{\"model\":{}",
                    serde_json::to_string(&report.model).unwrap_or_default()
                ));
                if !report.scores.is_empty() {
                    body.push_str(&format!(",\"scores\":{}", score_list_json(&report.scores)));
                }
                if !report.signals.is_empty() {
                    body.push_str(&format!(
                        ",\"signals\":{{\"review_only\":true,\"scores\":{}}}",
                        score_list_json(&report.signals)
                    ));
                }
                body.push('}');
            }
            ModerationMeta::Unavailable(reason) => {
                body.push_str(&format!(
                    ",\"moderation\":{{\"available\":false,\"reason\":\"{reason}\"}}"
                ));
            }
        }
    }
    if let Some(embedding) = embedding {
        // The embedding fields (issue #172): the
        // vector, base64-packed float16 little-endian,
        // with the companions that keep it
        // interpretable across operator
        // reconfigurations: the encoder identity, the
        // dimension, the encoding name, and the
        // normalization guarantee. An opt-in that
        // cannot be answered reports null plus a
        // reason instead, like the moderation field.
        match embedding {
            EmbeddingMeta::Report(report) => {
                body.push_str(&format!(
                    ",\"embedding\":{},\"embedding_model\":{},\"embedding_dim\":{},\"embedding_encoding\":\"f16le-base64\",\"embedding_normalized\":true",
                    serde_json::to_string(&pack_embedding(&report.vector)).unwrap_or_default(),
                    serde_json::to_string(&report.model).unwrap_or_default(),
                    report.vector.len()
                ));
            }
            EmbeddingMeta::Unavailable(reason) => {
                body.push_str(&format!(
                    ",\"embedding\":null,\"embedding_reason\":\"{reason}\""
                ));
            }
        }
    }
    if let Some(aesthetic) = aesthetic {
        // The aesthetic field (issue #172): the head's
        // score of the same embedding, with both
        // identities, or a null score carrying the
        // reason (a dimension mismatch names the
        // expected and actual dimensions). The score
        // is descriptive only.
        body.push_str(&format!(
            ",\"aesthetic\":{{\"head\":{}",
            serde_json::to_string(&aesthetic.head).unwrap_or_default()
        ));
        if let Some(model) = &aesthetic.model {
            body.push_str(&format!(
                ",\"model\":{}",
                serde_json::to_string(model).unwrap_or_default()
            ));
        }
        match &aesthetic.outcome {
            AestheticOutcome::Score(score) => {
                body.push_str(&format!(",\"score\":{score}"));
            }
            AestheticOutcome::DimensionMismatch { expected, got } => {
                body.push_str(&format!(
                    ",\"score\":null,\"reason\":\"dimension_mismatch\",\"expected\":{expected},\"got\":{got}"
                ));
            }
            AestheticOutcome::Unavailable(reason) => {
                body.push_str(&format!(",\"score\":null,\"reason\":\"{reason}\""));
            }
        }
        body.push('}');
    }
    if let Some(score) = processed.ssimulacra2 {
        body.push_str(&format!(",\"ssimulacra2\":{score:.1}"));
    }
    if let Some(video) = &processed.video {
        // The video record (issue #214): the probed facts
        // of the video source and the resolved timestamp
        // the frame was selected at. The duration is
        // omitted when the container reports none. Numbers
        // print at millisecond precision, the same
        // rounding the extractor seeks with.
        let seconds = |v: f64| {
            let rounded = (v * 1000.0).round() / 1000.0;
            if rounded == rounded.trunc() {
                format!("{}", rounded as i64)
            } else {
                format!("{rounded}")
            }
        };
        let duration = match video.duration_seconds {
            Some(d) => format!(",\"duration\":{}", seconds(d)),
            None => String::new(),
        };
        body.push_str(&format!(
            ",\"video\":{{\"width\":{},\"height\":{},\"frame_timestamp\":{}{}}}",
            video.width,
            video.height,
            seconds(video.frame_seconds),
            duration
        ));
    }
    if let Some(pdf) = &processed.pdf {
        // The PDF record (issue #215): the document's page
        // count, the page this response rendered (after
        // canonicalization), and the page size in points.
        // Points print the way the probe reports them:
        // whole values bare, fractions at two decimals.
        let points = |v: f64| {
            let rounded = (v * 100.0).round() / 100.0;
            if rounded == rounded.trunc() {
                format!("{}", rounded as i64)
            } else {
                format!("{rounded}")
            }
        };
        body.push_str(&format!(
            ",\"pdf\":{{\"pages\":{},\"page\":{},\"page_width\":{},\"page_height\":{}}}",
            pdf.pages,
            pdf.page,
            points(pdf.page_width),
            points(pdf.page_height)
        ));
    }
    if let Some(studio) = &processed.studio {
        // The studio record (issue #197): the resolved
        // parameters, the steps that ran, the fallbacks
        // taken, and the mask confidence when the provider
        // exposed one. The backdrop is spelled like the
        // placeholder colors (`#rrggbb`), or `none` when the
        // subject kept its transparency.
        let bg = match studio.bg {
            Some(c) => format!("\"#{:02x}{:02x}{:02x}\"", c[0], c[1], c[2]),
            None => "\"none\"".to_string(),
        };
        let steps: Vec<String> = studio.steps.iter().map(|s| format!("\"{s}\"")).collect();
        let fallbacks: Vec<String> = studio
            .fallbacks
            .iter()
            .map(|s| format!("\"{s}\""))
            .collect();
        body.push_str(&format!(
            ",\"studio\":{{\"bg\":{bg},\"bg_from\":\"{}\",\"shadow\":\"{}\",\"pad\":{},\"steps\":[{}],\"fallbacks\":[{}]",
            studio.bg_from,
            studio.shadow.name(),
            studio.pad,
            steps.join(","),
            fallbacks.join(",")
        ));
        if let Some(confidence) = studio.confidence {
            body.push_str(&format!(",\"mask_confidence\":{confidence}"));
        }
        body.push('}');
    }
    if let Some(inpaint) = &processed.inpaint {
        // The region-removal record (issue #176): the
        // rectangles the inpainting model erased, in chain
        // order, each [x, y, w, h] in the pixel coordinates
        // in effect where its filter ran.
        let regions: Vec<String> = inpaint
            .regions
            .iter()
            .map(|(x, y, w, h)| format!("[{x},{y},{w},{h}]"))
            .collect();
        body.push_str(&format!(
            ",\"inpaint\":{{\"regions\":[{}]}}",
            regions.join(",")
        ));
    }
    if let Some(segments) = &processed.segments {
        // The segmentation record (issue #175): every
        // object a class operation segmented, in chain
        // order, with its mask geometry: the detection box
        // it came from, the silhouette's tight box inside
        // it, and the silhouette's coverage of the box.
        let objects: Vec<String> = segments
            .objects
            .iter()
            .map(|o| {
                let (x, y, w, h) = o.detection;
                let (tx, ty, tw, th) = o.tight_box;
                let (cx, cy) = o.centroid;
                let label = serde_json::to_string(&o.label).unwrap_or_default();
                let score = format!("{:.3}", o.score);
                let coverage = format!("{:.3}", o.coverage);
                format!(
                    "{{\"label\":{label},\"score\":{score},\"box\":[{x},{y},{w},{h}],\
                     \"tight_box\":[{tx},{ty},{tw},{th}],\
                     \"coverage\":{coverage},\"centroid\":[{cx},{cy}]}}"
                )
            })
            .collect();
        body.push_str(&format!(
            ",\"segments\":{{\"objects\":[{}]}}",
            objects.join(",")
        ));
    }
    if let Some(ocr) = ocr {
        // The OCR field (issue #177): the model identity
        // (the recognition model: it fixes the script),
        // the operator-declared language, the lines in
        // reading order with their boxes in the rendered
        // image's pixels, and the lines joined as one
        // text content. An unavailable OCR is explicit
        // too, since the client opted in by name.
        match ocr {
            OcrMeta::Report(report) => {
                let lines: Vec<String> = report
                    .lines
                    .iter()
                    .map(|l| {
                        format!(
                            "{{\"text\":{},\"score\":{},\"x\":{},\"y\":{},\"w\":{},\"h\":{}}}",
                            serde_json::to_string(&l.text).unwrap_or_default(),
                            l.score,
                            l.x,
                            l.y,
                            l.w,
                            l.h
                        )
                    })
                    .collect();
                let content = report
                    .lines
                    .iter()
                    .map(|l| l.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                body.push_str(&format!(
                    ",\"ocr\":{{\"model\":{},\"language\":{},\"text\":[{}],\"text_content\":{}}}",
                    serde_json::to_string(&report.model).unwrap_or_default(),
                    serde_json::to_string(&report.language).unwrap_or_default(),
                    lines.join(","),
                    serde_json::to_string(&content).unwrap_or_default()
                ));
            }
            OcrMeta::Unavailable(reason) => {
                body.push_str(&format!(
                    ",\"ocr\":{{\"available\":false,\"reason\":\"{reason}\"}}"
                ));
            }
        }
    }
    if let Some(ph) = &processed.placeholder {
        let palette: Vec<String> = ph.palette.iter().map(|c| format!("\"{c}\"")).collect();
        body.push_str(&format!(
            ",\"dominant_color\":\"{}\",\"placeholder\":\"{}\",\"palette\":[{}],\"thumbhash\":\"{}\"",
            ph.dominant_color,
            ph.data_uri,
            palette.join(","),
            ph.thumbhash
        ));
    }
    body.push('}');
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
    // Meta is a second representation of the same URL (audit
    // PL-12): never publicly cacheable, so an intermediary can
    // never pin it in place of the image, or the reverse.
    headers.insert(header::CACHE_CONTROL, "private, no-store".parse().unwrap());
    headers.insert(header::VARY, "Accept, DPR, Sec-CH-DPR, Sec-CH-Width, Sec-CH-Viewport-Width, Width, Viewport-Width, Save-Data".parse().unwrap());
    (StatusCode::OK, headers, body).into_response()
}

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
    bytes: Vec<u8>,
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

fn etag_for(cache_key: &str) -> String {
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(cache_key.as_bytes());
    format!("\"{}\"", hex::encode(&digest[..16]))
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
fn busy_response(message: &'static str, retry_after_secs: u64) -> Response {
    let mut response = (StatusCode::SERVICE_UNAVAILABLE, message).into_response();
    response.headers_mut().insert(
        header::RETRY_AFTER,
        retry_after_secs.to_string().parse().unwrap(),
    );
    response
}

fn load_error_response(e: LoadError) -> Response {
    // The detailed cause (which can embed origin-controlled
    // strings and server-constructed store URLs) goes to the
    // server log; clients get a short generic message (PL-6).
    eprintln!("load error: {e}");
    let (status, message) = match e {
        LoadError::Forbidden(reason) => (StatusCode::FORBIDDEN, reason),
        LoadError::BadStatus(code) => (
            StatusCode::BAD_GATEWAY,
            format!("source returned status {code}"),
        ),
        LoadError::Network(_) => (StatusCode::BAD_GATEWAY, "source fetch failed".to_string()),
        LoadError::WrongContentType(_) => (
            StatusCode::BAD_GATEWAY,
            "source did not return an image".to_string(),
        ),
        LoadError::Timeout => (StatusCode::GATEWAY_TIMEOUT, "source timed out".to_string()),
        // 502, not 413: the client's request was a valid
        // signed URL; it is the origin object that is too
        // big, like the other origin faults above.
        LoadError::TooLarge { limit } => (
            StatusCode::BAD_GATEWAY,
            format!("source exceeds {limit} bytes"),
        ),
    };
    (status, message).into_response()
}

/// A grammar failure's response: the parser's message as
/// plain text, or the v2 JSON envelope
/// (`{"error":{"code","message"}}`, the code being the
/// error's stable name) when the client asked for JSON,
/// the same test the meta surface uses.
fn parse_failure(e: &crate::url::ParseError, accept: Option<&str>) -> Response {
    if wants_meta(accept) {
        let body = serde_json::json!({
            "error": { "code": e.code(), "message": e.to_string() }
        });
        (StatusCode::BAD_REQUEST, axum::Json(body)).into_response()
    } else {
        (StatusCode::BAD_REQUEST, e.to_string()).into_response()
    }
}

fn engine_error_response(e: EngineError) -> Response {
    eprintln!("engine error: {e}");
    let (status, message) = match e {
        EngineError::Decode(_) => (
            StatusCode::BAD_GATEWAY,
            "source image could not be decoded".to_string(),
        ),
        EngineError::Encode(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "image encoding failed".to_string(),
        ),
        EngineError::Unsupported(what) => (StatusCode::BAD_REQUEST, what),
        EngineError::NotConfigured(what) => (StatusCode::NOT_IMPLEMENTED, what),
        EngineError::Inference(crate::ai::AiError::BadOutput(_)) => (
            StatusCode::BAD_GATEWAY,
            "model produced unexpected output".to_string(),
        ),
        EngineError::Inference(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "image inference failed".to_string(),
        ),
        EngineError::Render(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "image rendering failed".to_string(),
        ),
        // The encoder is the operator's own program: its
        // failure is an upstream fault (502). The body is
        // generic, like every other class here: the cause
        // (the configured program path, the encoder's own
        // stderr) stays in the log line above, where the
        // operator reads it, and never reaches the client
        // (audit R2-30).
        EngineError::Video(_) => (StatusCode::BAD_GATEWAY, "video encoding failed".to_string()),
        EngineError::Frame(_) => (
            StatusCode::BAD_GATEWAY,
            "video frame extraction failed".to_string(),
        ),
        EngineError::Pdf(_) => (StatusCode::BAD_GATEWAY, "PDF rendering failed".to_string()),
    };
    (status, message).into_response()
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

    async fn spawn_app(allow_unsafe: bool) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
    async fn spawn_app_with_anim_gate(anim_gate: Arc<tokio::sync::Semaphore>) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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

    async fn spawn_app_with_tagger() -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
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
            detect: None,
            detectors: Vec::new(),
            focal: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            upload: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
    async fn meta_includes_tags_only_with_a_tagger() {
        let src = spawn_image_server().await;
        let remainder = format!("40x24/{src}");
        let url_path = format!("/{}/{remainder}", sign(&remainder));
        // With a tagger: tags and alt text are present.
        let base = spawn_app_with_tagger().await;
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

    async fn spawn_app_with_moderation(
        provider: Option<Arc<dyn crate::ai::ModerationProvider>>,
    ) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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

    async fn spawn_app_with_ocr(provider: Option<Arc<dyn crate::ai::OcrProvider>>) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
        // one-permit AI gate: the request holds the only
        // permit while its annotations run. They must run
        // under the held permit; queueing for a second one
        // would wait out the full gate timeout and silently
        // drop both the tags and the detections (audit
        // R2-29).
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
        let status = |e: EngineError| engine_error_response(e).status();
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
        let resp = load_error_response(LoadError::TooLarge { limit: 10 });
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn busy_responses_carry_retry_after() {
        // Both gate 503s are built by busy_response; the
        // header is what lets a client back off instead of
        // retrying into the same full queue (review AP-9).
        let resp = busy_response("server busy", 5);
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
    fn etag_for_is_a_stable_quoted_digest_of_the_cache_key() {
        let etag = etag_for("some-cache-key");
        assert!(etag.starts_with('"') && etag.ends_with('"'));
        assert_eq!(etag.len(), 34);
        assert_eq!(etag, etag_for("some-cache-key"));
        assert_ne!(etag, etag_for("another-cache-key"));
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
        let etag = etag_for("key");
        // No conditional header: the ETag is attached and the
        // response is otherwise untouched.
        let mut resp = respond(ImageFormat::Png, None, vec![1, 2, 3], None);
        finalize_validators(&mut resp, &etag, None);
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get(header::ETAG).unwrap(), etag.as_str());
        // A matching If-None-Match: 304, empty body, and the
        // content headers are gone.
        let mut resp = respond(ImageFormat::Png, None, vec![1, 2, 3], None);
        finalize_validators(&mut resp, &etag, Some(etag.as_str()));
        assert_eq!(resp.status(), StatusCode::NOT_MODIFIED);
        assert!(resp.headers().get(header::CONTENT_TYPE).is_none());
        assert!(resp.headers().get(header::CONTENT_LENGTH).is_none());
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(body.is_empty());
        // A non-matching value: the full 200 goes out, ETag on.
        let mut resp = respond(ImageFormat::Png, None, vec![1, 2, 3], None);
        finalize_validators(&mut resp, &etag, Some("\"other\""));
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], &[1, 2, 3]);
    }

    #[test]
    fn validators_skip_meta_and_error_responses() {
        let etag = etag_for("key");
        // Meta responses are private, no-store: no ETag is
        // attached and a matching conditional never 304s them.
        let mut resp = respond(ImageFormat::Png, None, vec![1], None);
        resp.headers_mut()
            .insert(header::CACHE_CONTROL, "private, no-store".parse().unwrap());
        finalize_validators(&mut resp, &etag, Some(etag.as_str()));
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get(header::ETAG).is_none());
        // An error status is left alone too, public or not.
        let mut resp = respond(ImageFormat::Png, None, vec![1], None);
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
    async fn spawn_app_with_shared(dir: Option<&std::path::Path>, endpoint: &str) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
        let metrics = client()
            .get(format!("{base}/metrics"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(metrics.contains("crabcrop_purges_total 2"), "{metrics}");
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(1)),
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
            inflight: Arc::new(Inflight::default()),
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
            dpr: None,
            ssim_target: state.ssim_target,
            avif_speed: state.avif_speed,
            jxl_speed: state.jxl_speed,
        };
        let held = state.process_gate.clone().acquire_owned().await.unwrap();
        let png = test_png();
        let waiting = run_process(&state, &job, &png);
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(1)),
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
            inflight: Arc::new(Inflight::default()),
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
            dpr: None,
            ssim_target: state.ssim_target,
            avif_speed: state.avif_speed,
            jxl_speed: state.jxl_speed,
        };
        let held = state.process_gate.clone().acquire_owned().await.unwrap();
        let png = test_png();
        let waiting = run_process(&state, &job, &png);
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
        spawn_app_with_upload_gate(
            dir,
            max_bytes,
            allow_overwrite,
            Arc::new(tokio::sync::Semaphore::new(4)),
        )
        .await
    }

    async fn spawn_app_with_upload_gate(
        dir: &std::path::Path,
        max_bytes: u64,
        allow_overwrite: bool,
        process_gate: Arc<tokio::sync::Semaphore>,
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
            metrics: Arc::new(Metrics::default()),
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
            inflight: Arc::new(Inflight::default()),
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

    /// An app over a disk tier under a chosen
    /// configuration fingerprint (the epoch tests).
    async fn spawn_app_with_store_fingerprint(dir: &std::path::Path, fingerprint: &str) -> String {
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
        let gate = Arc::new(tokio::sync::Semaphore::new(1));
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
        let gate = Arc::new(tokio::sync::Semaphore::new(1));
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
        let metrics = client()
            .get(format!("{base}/metrics"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
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
        let body = client()
            .get(format!("{base}/metrics"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
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
        let (src, hits) = spawn_counting_server(404, "text/plain", b"nope".to_vec()).await;
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

        let resp = client()
            .get(format!("{base}/metrics"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.text().await.unwrap();
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
            metrics: Arc::new(Metrics::default()),
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
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
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
            inflight: Arc::new(Inflight::default()),
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
