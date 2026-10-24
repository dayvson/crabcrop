//! The HTTP server: parse, verify, load, process, respond.
//!
//! `GET /<signature>/<operations...>/<image-url>` verifies the
//! signature, loads the source image, runs the engine, and responds
//! with the optimized bytes. When the URL names no format, the
//! response format is negotiated from the request's `Accept`
//! header. `GET /health` is the health check and
//! `GET /metrics` serves Prometheus metrics. Every image request
//! leaves one structured JSON log line and one metrics update.

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
    /// Detection provider for class redaction and meta
    /// detections, preloaded at boot when
    /// `CRABCROP_DETECT_MODEL` is configured. `None` means
    /// class redaction fails cleanly and meta carries no
    /// detections.
    pub detect: Option<Arc<dyn crate::ai::DetectionProvider>>,
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
    /// Named operation presets from `CRABCROP_PRESETS_FILE`:
    /// name to operation string, validated at boot.
    pub presets: std::collections::HashMap<String, String>,
    /// Presets-only mode, from `CRABCROP_ONLY_PRESETS`: reject
    /// free-form operations in requests.
    pub only_presets: bool,
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
    /// Bounds concurrent video encodes: the subprocess is
    /// the server's most expensive and least predictable
    /// work, so it gets its own small gate (2 permits, like
    /// the AI gate) on top of the process gate.
    pub video_gate: Arc<tokio::sync::Semaphore>,
}

/// Build the router.
pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/purge/{*path}", axum::routing::post(purge_handle))
        .route(
            "/purge-group/{*path}",
            axum::routing::post(purge_group_handle),
        )
        .route("/{*path}", get(handle))
        .with_state(state)
        .layer(axum::middleware::from_fn(security_headers))
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
    let target = if state.presets.is_empty() && !state.only_presets {
        target
    } else {
        match crate::url::expand_presets(target, &state.presets, state.only_presets) {
            Ok(text) => {
                expanded = text;
                &expanded
            }
            Err(_) => target,
        }
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
        removed += shared.purge_matching(&key_form).await;
    }
    state.metrics.record_purge();
    purge_log("purge", target, Some(removed));
    (StatusCode::OK, format!("purged {removed}\n")).into_response()
}

/// `POST /purge-group/<signature>/<group>`: invalidate every
/// stored entry in one cache group (RFC 9875), across both
/// tiers. The signature is verified with the same purge
/// subkey as `/purge`, over the literal `purge-group/<group>`,
/// so an image or per-URL purge signature cannot be replayed
/// here. A bad signature is a 403 and removes nothing; an
/// unknown group answers 200 with a count of 0.
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
        .verify_purge(key_id.as_deref(), &signature, &target)
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
    let dpr = dpr_hint(&headers);
    let width = width_hint(&headers);
    let save_data = save_data_hint(&headers);
    let (response, mut record) =
        serve(&state, path, accept.as_deref(), dpr, width, save_data).await;
    record.status = response.status().as_u16();
    record.total = started.elapsed();
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
    watermarks: crate::filters::WatermarkImages,
    placeholders: bool,
    mask_provider: Option<Arc<dyn crate::ai::MaskProvider>>,
    upscale_provider: Option<Arc<dyn crate::ai::UpscaleProvider>>,
    face_provider: Option<Arc<dyn crate::ai::FaceProvider>>,
    detect_provider: Option<Arc<dyn crate::ai::DetectionProvider>>,
    text_renderer: Option<Arc<dyn crate::text::TextRenderer>>,
    video_encoder: Option<Arc<crate::video::VideoEncoder>>,
    dpr: Option<f32>,
    ssim_target: f64,
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
        let ctx = crate::engine::ProcessContext {
            preferred: job.preferred,
            watermarks: Some(&job.watermarks),
            placeholders: job.placeholders,
            mask_provider: job.mask_provider.as_deref(),
            upscale_provider: job.upscale_provider.as_deref(),
            face_provider: job.face_provider.as_deref(),
            detect_provider: job.detect_provider.as_deref(),
            video_encoder: job.video_encoder.as_deref(),
            save_data: job.save_data,
            preserve_metadata: job.preserve_metadata,
            text_renderer: job.text_renderer.as_deref(),
            dpr: job.dpr,
            ssim_target: job.ssim_target,
            avif_speed: job.avif_speed,
            jxl_speed: job.jxl_speed,
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
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
    if matches!(rest.split('/').next(), Some("purge" | "purge-group")) {
        return (
            (
                StatusCode::BAD_REQUEST,
                "an image URL may not begin with a purge segment",
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
    let expanded;
    let rest = if state.presets.is_empty() && !state.only_presets {
        rest
    } else {
        match crate::url::expand_presets(rest, &state.presets, state.only_presets) {
            Ok(text) => {
                expanded = text;
                &expanded
            }
            Err(e) => {
                return (parse_failure(&e, accept), record);
            }
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
    let key_base = crate::url::canonical_form(&ops, source_url);
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

    // The verified path is content-addressed: a stored response can
    // be served without touching the loader or the engine. Meta
    // requests bypass the cache in both directions: they must
    // never be answered with image bytes, and their JSON is never
    // stored as an image entry.
    let mut cache_status = "crabcrop; fwd=uri-miss";
    let hit = if meta {
        None
    } else {
        record.cache_looked_up = true;
        state.cache.get(&cache_key)
    };
    // Memory first, disk second: a disk hit is promoted into
    // memory so the next repeat skips the file read.
    let hit = match hit {
        Some(h) => {
            cache_status = "crabcrop; hit";
            Some(h)
        }
        None if meta => None,
        None => match &state.store {
            Some(store) => {
                let found = store.get(&cache_key);
                if let Some(image) = &found {
                    cache_status = "crabcrop; hit; detail=\"disk\"";
                    state.cache.put(cache_key.clone(), image.clone());
                }
                found
            }
            None => None,
        },
    };
    // Shared tier last (issue #142): a hit is promoted into
    // both local tiers so the next repeat is local.
    let hit = match hit {
        Some(h) => Some(h),
        None if meta => None,
        None => match &state.shared_store {
            Some(shared) => {
                let found = shared.get(&cache_key).await;
                if let Some(image) = &found {
                    cache_status = "crabcrop; hit; detail=\"shared\"";
                    state.cache.put(cache_key.clone(), image.clone());
                    if let Some(store) = &state.store {
                        store.put(&cache_key, image);
                    }
                }
                found
            }
            None => None,
        },
    };
    if let Some(hit) = hit {
        record.cache_hit = true;
        record.bytes_out = hit.bytes.len();
        let mut response = respond(hit.format, hit.quality, hit.bytes, max_age);
        attach_cache_headers(&mut response, cache_status, &groups);
        return (response, record);
    }

    record.load_ran = true;
    let load_start = Instant::now();
    let mut used_fallback = false;
    let bytes = match state.loader.load(source_url).await {
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

    // Watermarks are resolved here, before the engine runs: the
    // engine only ever sees pixels. Each referenced image goes
    // through the same loader (and its bounds and source policy)
    // as sources, and a watermark that fails to load or decode
    // fails the request rather than being silently skipped.
    let mut watermark_images = crate::filters::WatermarkImages::new();
    for wm_url in &directives.watermark_urls {
        record.load_ran = true;
        let wm_bytes = match state.loader.load(wm_url).await {
            Ok(b) => b,
            Err(e) => {
                record.fail = FailStage::Load;
                return (load_error_response(e), record);
            }
        };
        match crate::engine::decode_image(&wm_bytes) {
            Ok(img) => {
                watermark_images.insert(wm_url.to_string(), img);
            }
            Err(e) => {
                record.fail = FailStage::Engine;
                return (engine_error_response(e), record);
            }
        }
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
    // Video encodes sit behind their own gate, on top of
    // the process gate inside `run_process`: the external
    // encoder is the server's most expensive and least
    // predictable work (seconds per job, a subprocess), so
    // a burst of video requests sheds here instead of
    // queueing unboundedly behind the transform pipeline.
    let _video_permit = if wants_video {
        match tokio::time::timeout(VIDEO_GATE_WAIT, state.video_gate.clone().acquire_owned()).await
        {
            Ok(Ok(permit)) => Some(permit),
            _ => {
                return (
                    busy_response("server busy: video encoding queue is full", 5),
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
        watermarks: watermark_images,
        placeholders,
        mask_provider: state.removebg.clone(),
        upscale_provider: state.upscale.clone(),
        face_provider: state.face.clone(),
        detect_provider: state.detect.clone(),
        text_renderer: state.font.clone(),
        video_encoder: state.video_encoder.clone(),
        dpr: dpr_hint,
        ssim_target: state.ssim_target,
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
    // origin gets retried on the next request.
    if !meta && !used_fallback {
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
    let annotation = if meta {
        annotate_meta(state, &processed).await
    } else {
        None
    };
    let detections = if meta && wants_detections(accept) {
        detect_meta(state, &processed).await
    } else {
        None
    };
    let mut response = if meta {
        meta_response(&processed, annotation.as_ref(), detections.as_deref())
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
/// response whose image processed fine.
async fn annotate_meta(
    state: &AppState,
    processed: &crate::engine::ProcessedImage,
) -> Option<Annotation> {
    let tagger = state.tagger.as_ref()?;
    let _permit = tokio::time::timeout(AI_GATE_WAIT, state.ai_gate.clone().acquire_owned())
        .await
        .ok()?
        .ok()?;
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

/// Run object detection for a meta response, when a
/// detection provider is configured and the request asked
/// (`detections=1`). Same failure stance as annotation:
/// detection is a debugging surface here, so a failure
/// omits the field rather than failing the response.
async fn detect_meta(
    state: &AppState,
    processed: &crate::engine::ProcessedImage,
) -> Option<Vec<crate::ai::Detection>> {
    let detector = state.detect.as_ref()?;
    let _permit = tokio::time::timeout(AI_GATE_WAIT, state.ai_gate.clone().acquire_owned())
        .await
        .ok()?
        .ok()?;
    let img = crate::engine::decode_image(&processed.bytes).ok()?;
    let detector = detector.clone();
    match tokio::task::spawn_blocking(move || detector.detect(&img)).await {
        Ok(Ok(detections)) => Some(detections),
        Ok(Err(e)) => {
            eprintln!("detection error: {e}");
            None
        }
        Err(e) => {
            eprintln!("detection task failed: {e}");
            None
        }
    }
}

fn meta_response(
    processed: &crate::engine::ProcessedImage,
    annotation: Option<&Annotation>,
    detections: Option<&[crate::ai::Detection]>,
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
        let boxes: Vec<String> = detections
            .iter()
            .map(|d| {
                format!(
                    "{{\"label\":{},\"score\":{},\"x\":{},\"y\":{},\"w\":{},\"h\":{}}}",
                    serde_json::to_string(&d.label).unwrap_or_default(),
                    d.score,
                    d.x,
                    d.y,
                    d.w,
                    d.h
                )
            })
            .collect();
        body.push_str(&format!(",\"detections\":[{}]", boxes.join(",")));
    }
    if let Some(score) = processed.ssimulacra2 {
        body.push_str(&format!(",\"ssimulacra2\":{score:.1}"));
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
        // failure is an upstream fault (502), and the body
        // names it (including its truncated stderr) because
        // the operator is the one who can fix it.
        EngineError::Video(cause) => (
            StatusCode::BAD_GATEWAY,
            format!("video encoder failed: {cause}"),
        ),
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
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            detect: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            default_ops: Vec::new(),
            video_encoder: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
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
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            detect: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            default_ops: Vec::new(),
            video_encoder: Some(Arc::new(encoder)),
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        format!("http://{addr}")
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
        // A failing encoder: 502 naming the encoder and its
        // own error output.
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
        assert!(body.contains("video encoder failed"), "{body}");
        assert!(body.contains("broken encoder"), "{body}");
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
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            detect: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            presets,
            only_presets,
            default_ops: crate::url::parse_default_ops(defaults, &std::collections::HashMap::new())
                .unwrap(),
            video_encoder: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
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
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            detect: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            default_ops: Vec::new(),
            video_encoder: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
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
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            detect: None,
            font: None,
            fallback: Some(Fallback::Bytes(Arc::new(png))),
            store: None,
            shared_store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            default_ops: Vec::new(),
            video_encoder: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
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
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            detect: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            presets,
            only_presets,
            default_ops: Vec::new(),
            video_encoder: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
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
            face: None,
            tagger: Some(Arc::new(FakeTagger)),
            tag_threshold: 0.05,
            detect: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            default_ops: Vec::new(),
            video_encoder: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
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
        let state = AppState {
            keyring: crate::url::Keyring::single(SECRET.to_vec()),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            detect: Some(detector),
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            default_ops: Vec::new(),
            video_encoder: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
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
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            detect: None,
            font: None,
            fallback: None,
            store: dir
                .map(|d| Arc::new(crate::storage::DiskStore::open(d, 16 * 1024 * 1024).unwrap())),
            shared_store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            presets,
            only_presets: false,
            default_ops: Vec::new(),
            video_encoder: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
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
        // The real purge: signed over the literal target.
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
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            detect: None,
            font: None,
            fallback: None,
            store: Some(Arc::new(
                crate::storage::DiskStore::open(dir, 16 * 1024 * 1024).unwrap(),
            )),
            shared_store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            default_ops: Vec::new(),
            video_encoder: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
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
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            detect: None,
            font: None,
            fallback: None,
            store: dir
                .map(|d| Arc::new(crate::storage::DiskStore::open(d, 16 * 1024 * 1024).unwrap())),
            shared_store: Some(Arc::new(crate::s3store::S3Store::new(
                "results".into(),
                String::new(),
                Some(endpoint.into()),
                "us-east-1".into(),
                None,
            ))),
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            default_ops: Vec::new(),
            video_encoder: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
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
            face: None,
            tagger: None,
            tag_threshold: 0.05,
            detect: None,
            font: None,
            fallback: None,
            store: None,
            shared_store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            process_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            avif_speed: crate::engine::DEFAULT_AVIF_SPEED,
            jxl_speed: crate::engine::DEFAULT_JXL_SPEED,
            width_step: 100,
            preserve_metadata: false,
            presets: std::collections::HashMap::new(),
            only_presets: false,
            default_ops: Vec::new(),
            video_encoder: None,
            video_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        };
        let job = ProcessJob {
            save_data: false,
            preserve_metadata: false,
            ops: Vec::new(),
            preferred: None,
            watermarks: crate::filters::WatermarkImages::new(),
            placeholders: false,
            mask_provider: None,
            upscale_provider: None,
            face_provider: None,
            detect_provider: None,
            text_renderer: None,
            video_encoder: None,
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
}
