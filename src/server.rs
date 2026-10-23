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
use crate::url::{
    ImageFormat, explicit_format, parse_operations_with_expiry, parse_signature, verify,
    verify_purge,
};

/// Shared state for the handlers.
#[derive(Debug, Clone)]
pub struct AppState {
    /// HMAC secret, from `CRABCROP_SECRET`.
    pub secret: Vec<u8>,
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
    /// Text renderer, loaded at boot when `CRABCROP_FONT_PATH`
    /// is configured. `None` means `text(...)` requests fail
    /// cleanly.
    pub font: Option<Arc<dyn crate::text::TextRenderer>>,
    /// Fallback image for source failures, when configured.
    pub fallback: Option<Fallback>,
    /// Disk-backed result storage, the persistent tier under
    /// the memory cache, when configured.
    pub store: Option<Arc<crate::storage::DiskStore>>,
    /// Bounds concurrent AI inference so it cannot starve the
    /// transform pipeline.
    pub ai_gate: Arc<tokio::sync::Semaphore>,
    /// SSIMULACRA2 target for verified `quality(auto)` encodes,
    /// from `CRABCROP_SSIM_TARGET`.
    pub ssim_target: f64,
    /// Named operation presets from `CRABCROP_PRESETS_FILE`:
    /// name to operation string, validated at boot.
    pub presets: std::collections::HashMap<String, String>,
    /// Presets-only mode, from `CRABCROP_ONLY_PRESETS`: reject
    /// free-form operations in requests.
    pub only_presets: bool,
}

/// Build the router.
pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/purge/{*path}", axum::routing::post(purge_handle))
        .route("/{*path}", get(handle))
        .with_state(state)
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
    headers
        .get("DPR")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|f| [1.0, 1.5, 2.0, 3.0].contains(f))
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
async fn purge_handle(State(state): State<AppState>, uri: axum::http::Uri) -> Response {
    let Some(tail) = uri.path().strip_prefix("/purge/") else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let full = format!("/{tail}");
    let (signature, target) = match parse_signature(&full) {
        Ok(v) => v,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    if !verify_purge(&state.secret, &signature, target, state.allow_unsafe) {
        return (StatusCode::FORBIDDEN, "bad signature").into_response();
    }
    let mut removed = state.cache.purge_matching(target);
    if let Some(store) = &state.store {
        removed += store.purge_matching(target);
    }
    (StatusCode::OK, format!("purged {removed}\n")).into_response()
}

async fn handle(
    State(state): State<AppState>,
    uri: axum::http::Uri,
    headers: HeaderMap,
) -> Response {
    // The raw path, exactly as sent: the signature covers it
    // byte-for-byte, so no percent-decoding may happen first
    // (the `Path` extractor decodes, which would rewrite e.g.
    // an encoded watermark URL before verification).
    let path = uri.path().trim_start_matches('/');
    let started = Instant::now();
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let dpr = dpr_hint(&headers);
    let (response, mut record) = serve(&state, path, accept.as_deref(), dpr).await;
    record.status = response.status().as_u16();
    record.total = started.elapsed();
    state.metrics.record(&record);
    println!("{}", record.to_json());
    response
}

/// Run one image request, timing each stage into the returned record.
async fn serve(
    state: &AppState,
    path: &str,
    accept: Option<&str>,
    dpr_hint: Option<f32>,
) -> (Response, RequestRecord) {
    let mut record = RequestRecord::default();
    let full = format!("/{path}");

    let (signature, rest) = match parse_signature(&full) {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_REQUEST.into_response(), record),
    };
    let verify_start = Instant::now();
    let verified = verify(&state.secret, &signature, rest, state.allow_unsafe);
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
    if rest.split('/').next() == Some("purge") {
        return (StatusCode::BAD_REQUEST.into_response(), record);
    }
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
            Err(_) => return (StatusCode::BAD_REQUEST.into_response(), record),
        }
    };
    let (ops, source_url, expiry) = match parse_operations_with_expiry(rest) {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_REQUEST.into_response(), record),
    };
    // The operation summary is everything before the source URL.
    record.ops = rest[..rest.len() - source_url.len()]
        .trim_end_matches('/')
        .to_string();
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

    // The DPR client hint settles sizing when the URL has no
    // explicit `dpr` directive; the effective factor is part of
    // what the response is, so it joins the cache key whenever
    // it changes the output.
    let effective_dpr = crate::url::explicit_dpr(&ops).or(dpr_hint).unwrap_or(1.0);

    // Without an explicit format in the URL, the response depends
    // on the Accept class as well as the path, so the cache key
    // carries the class. Explicit-format URLs key on the path alone.
    let preferred = negotiate_format(accept);
    let meta = wants_meta(accept);
    let placeholders = meta && wants_placeholders(accept);
    let mut cache_key = if explicit_format(&ops).is_some() {
        rest.to_string()
    } else {
        let class = match preferred {
            Some(ImageFormat::Avif) => "avif",
            Some(ImageFormat::Jxl) => "jxl",
            Some(ImageFormat::WebP) => "webp",
            _ => "source",
        };
        format!("{rest}#{class}")
    };
    if effective_dpr != 1.0 {
        cache_key = format!("{cache_key}#dpr{effective_dpr}");
    }

    // The verified path is content-addressed: a stored response can
    // be served without touching the loader or the engine. Meta
    // requests bypass the cache in both directions: they must
    // never be answered with image bytes, and their JSON is never
    // stored as an image entry.
    let hit = if meta {
        None
    } else {
        state.cache.get(&cache_key)
    };
    // Memory first, disk second: a disk hit is promoted into
    // memory so the next repeat skips the file read.
    let hit = match hit {
        Some(h) => Some(h),
        None if meta => None,
        None => match &state.store {
            Some(store) => {
                let found = store.get(&cache_key);
                if let Some(image) = &found {
                    state.cache.put(cache_key.clone(), image.clone());
                }
                found
            }
            None => None,
        },
    };
    if let Some(hit) = hit {
        record.cache_hit = true;
        record.bytes_out = hit.bytes.len();
        return (respond(hit.format, hit.quality, hit.bytes), record);
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
    for wm_url in crate::url::watermark_urls(&ops) {
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
    let _ai_permit = if crate::url::has_removebg(&ops)
        || crate::url::has_upscale(&ops)
        || crate::url::has_smart_crop(&ops)
    {
        Some(
            state
                .ai_gate
                .clone()
                .acquire_owned()
                .await
                .expect("the AI gate is never closed"),
        )
    } else {
        None
    };
    let ctx = crate::engine::ProcessContext {
        preferred,
        watermarks: Some(&watermark_images),
        placeholders,
        mask_provider: state.removebg.as_deref(),
        upscale_provider: state.upscale.as_deref(),
        face_provider: state.face.as_deref(),
        text_renderer: state.font.as_deref(),
        dpr: dpr_hint,
        ssim_target: state.ssim_target,
    };
    let processed = match state.engine.process_preferred(&bytes, &ops, ctx.clone()) {
        Ok(p) => p,
        Err(e) => {
            record.process = process_start.elapsed();
            // A source that loads but does not decode is the
            // other fallback case. Other engine errors are
            // properties of the request or the configuration,
            // not of the source, and stay honest.
            let retry = if !used_fallback && matches!(e, EngineError::Decode(_)) {
                fallback_source_bytes(state)
                    .await
                    .and_then(|fb| state.engine.process_preferred(&fb, &ops, ctx).ok())
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
        };
        state.cache.put(cache_key.clone(), entry.clone());
        if let Some(store) = &state.store {
            store.put(&cache_key, &entry);
        }
    }
    record.bytes_out = processed.bytes.len();
    let mut response = if meta {
        meta_response(&processed)
    } else {
        respond(processed.format, processed.quality, processed.bytes)
    };
    if used_fallback {
        record.fallback = true;
        response
            .headers_mut()
            .insert("x-crabcrop-fallback", "1".parse().unwrap());
    }
    (response, record)
}

/// Pick a format from the client's `Accept` header, in negotiation
/// order: AVIF first, then JPEG XL, then WebP. `None` keeps the
/// detected source format. A format the client explicitly refuses
/// (q=0) is never picked, and wildcards do not count as an offer:
/// a bare `*/*` from a generic HTTP client should not trigger the
/// expensive AVIF encode.
fn negotiate_format(accept: Option<&str>) -> Option<ImageFormat> {
    let accept = accept?;
    let (mut avif, mut jxl, mut webp) = (false, false, false);
    for part in accept.split(',') {
        let mut segments = part.split(';');
        let media = segments.next().unwrap_or("").trim();
        let refused = segments.any(|p| {
            p.trim()
                .strip_prefix("q=")
                .and_then(|v| v.trim().parse::<f32>().ok())
                .is_some_and(|q| q == 0.0)
        });
        if refused {
            continue;
        }
        match media {
            "image/avif" => avif = true,
            "image/jxl" => jxl = true,
            "image/webp" => webp = true,
            _ => {}
        }
    }
    if avif {
        Some(ImageFormat::Avif)
    } else if jxl {
        Some(ImageFormat::Jxl)
    } else if webp {
        Some(ImageFormat::WebP)
    } else {
        None
    }
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
/// placeholder fields: a `placeholders` parameter on that entry,
/// bare or set to a truthy value (`0`/`false` decline).
fn wants_placeholders(accept: Option<&str>) -> bool {
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
        return params.iter().any(|p| {
            let mut kv = p.trim().splitn(2, '=');
            kv.next() == Some("placeholders")
                && !matches!(kv.next().map(str::trim), Some("0") | Some("false"))
        });
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
    }
}

/// The meta response: what the pipeline produced, as JSON,
/// without the image bytes.
fn meta_response(processed: &crate::engine::ProcessedImage) -> Response {
    let mut body = format!(
        "{{\"width\":{},\"height\":{},\"format\":\"{}\",\"bytes\":{},\"source_width\":{},\"source_height\":{}",
        processed.width,
        processed.height,
        format_name(processed.format),
        processed.bytes.len(),
        processed.source_width,
        processed.source_height,
    );
    if let Some(score) = processed.ssimulacra2 {
        body.push_str(&format!(",\"ssimulacra2\":{score:.1}"));
    }
    if let Some(ph) = &processed.placeholder {
        body.push_str(&format!(
            ",\"dominant_color\":\"{}\",\"placeholder\":\"{}\"",
            ph.dominant_color, ph.data_uri
        ));
    }
    body.push('}');
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
    headers.insert(
        header::CACHE_CONTROL,
        "public, max-age=31536000, immutable".parse().unwrap(),
    );
    headers.insert(header::VARY, "Accept, DPR".parse().unwrap());
    (StatusCode::OK, headers, body).into_response()
}

/// Build a successful image response: content type, immutable cache
/// headers, and the auto-quality header when quality was chosen.
fn respond(format: ImageFormat, quality: Option<u8>, bytes: Vec<u8>) -> Response {
    let content_type = match format {
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::Png => "image/png",
        ImageFormat::WebP => "image/webp",
        ImageFormat::Jxl => "image/jxl",
        ImageFormat::Gif => "image/gif",
        ImageFormat::Avif => "image/avif",
    };
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    headers.insert(
        header::CACHE_CONTROL,
        "public, max-age=31536000, immutable".parse().unwrap(),
    );
    // The format may have been negotiated from the request's Accept.
    headers.insert(header::VARY, "Accept, DPR".parse().unwrap());
    if let Some(q) = quality {
        headers.insert(
            axum::http::HeaderName::from_static("x-crabcrop-quality"),
            q.to_string().parse().unwrap(),
        );
    }
    (headers, bytes).into_response()
}

fn load_error_response(e: LoadError) -> Response {
    let (status, message) = match e {
        LoadError::Forbidden(reason) => (StatusCode::FORBIDDEN, reason),
        LoadError::BadStatus(code) => (
            StatusCode::BAD_GATEWAY,
            format!("source returned status {code}"),
        ),
        LoadError::Network(cause) => (StatusCode::BAD_GATEWAY, cause),
        LoadError::WrongContentType(ct) => (
            StatusCode::BAD_GATEWAY,
            format!("source is not an image: {ct:?}"),
        ),
        LoadError::Timeout => (StatusCode::GATEWAY_TIMEOUT, "source timed out".to_string()),
        LoadError::TooLarge { limit } => (
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("source exceeds {limit} bytes"),
        ),
    };
    (status, message).into_response()
}

fn engine_error_response(e: EngineError) -> Response {
    let (status, message) = match e {
        EngineError::Decode(cause) => (StatusCode::BAD_GATEWAY, cause),
        EngineError::Encode(cause) => (StatusCode::INTERNAL_SERVER_ERROR, cause),
        EngineError::Unsupported(what) => (StatusCode::BAD_REQUEST, what),
        EngineError::Inference(cause) => (StatusCode::INTERNAL_SERVER_ERROR, cause),
        EngineError::Render(cause) => (StatusCode::INTERNAL_SERVER_ERROR, cause),
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
        use hmac::{Hmac, Mac};
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
            secret: SECRET.to_vec(),
            allow_unsafe,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            face: None,
            font: None,
            fallback: None,
            store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            presets: std::collections::HashMap::new(),
            only_presets: false,
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
            secret: SECRET.to_vec(),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            face: None,
            font: None,
            fallback: Some(Fallback::Bytes(Arc::new(png))),
            store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            presets: std::collections::HashMap::new(),
            only_presets: false,
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
            secret: SECRET.to_vec(),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            face: None,
            font: None,
            fallback: None,
            store: None,
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            presets,
            only_presets,
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
    async fn removebg_without_a_configured_model_is_a_clean_400() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:removebg:format(png)/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp.text().await.unwrap();
        assert!(body.contains("not configured"), "body: {body}");
    }

    #[tokio::test]
    async fn upscale_without_a_configured_model_is_a_clean_400() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:upscale(2)/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
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
        let meta: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        let color = meta["dominant_color"].as_str().unwrap();
        assert!(color.starts_with('#') && color.len() == 7, "{color}");
        assert!(
            meta["placeholder"]
                .as_str()
                .unwrap()
                .starts_with("data:image/jpeg;base64,")
        );
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
            secret: SECRET.to_vec(),
            allow_unsafe: false,
            loader: Loader::with_defaults().unwrap(),
            engine: Engine,
            metrics: Arc::new(Metrics::default()),
            cache: Arc::new(Cache::new(16 * 1024 * 1024)),
            removebg: None,
            upscale: None,
            face: None,
            font: None,
            fallback: None,
            store: Some(Arc::new(
                crate::storage::DiskStore::open(dir, 16 * 1024 * 1024).unwrap(),
            )),
            ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
            ssim_target: crate::quality::DEFAULT_SSIM_TARGET,
            presets: std::collections::HashMap::new(),
            only_presets: false,
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
    async fn fallback_serves_through_the_operations_on_a_dead_source() {
        let base = spawn_app_with_fallback().await;
        let (dead, hits) = spawn_counting_server(404, "image/png", test_png()).await;
        let remainder = format!("20x15/{dead}");
        let sig = sign(&remainder);
        let url = format!("{base}/{sig}/{remainder}");
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("x-crabcrop-fallback").unwrap(), "1");
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
    async fn text_without_a_font_is_400() {
        let src = spawn_image_server().await;
        let base = spawn_app(false).await;
        let remainder = format!("filters:text(HI,20,ffffff,bottom-right)/{src}");
        let sig = sign(&remainder);
        let resp = client()
            .get(format!("{base}/{sig}/{remainder}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
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
        assert_eq!(resp.headers().get(header::VARY).unwrap(), "Accept, DPR");
        let body = resp.bytes().await.unwrap();
        assert_eq!(&body[4..8], b"ftyp");
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
