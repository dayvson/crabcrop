//! The HTTP server: parse, verify, load, process, respond.
//!
//! `GET /<signature>/<operations...>/<image-url>` verifies the
//! signature, loads the source image, runs the engine, and responds
//! with the optimized bytes. `GET /health` is the health check and
//! `GET /metrics` serves Prometheus metrics. Every image request
//! leaves one structured JSON log line and one metrics update.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    Router,
    extract::{Path, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};

use crate::cache::{Cache, CachedImage};
use crate::engine::{Engine, EngineError};
use crate::loader::{LoadError, Loader};
use crate::observe::{FailStage, Metrics, RequestRecord};
use crate::url::{ImageFormat, parse_operations, parse_signature, verify};

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
}

/// Build the router.
pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
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

async fn handle(State(state): State<AppState>, Path(path): Path<String>) -> Response {
    let started = Instant::now();
    let (response, mut record) = serve(&state, &path).await;
    record.status = response.status().as_u16();
    record.total = started.elapsed();
    state.metrics.record(&record);
    println!("{}", record.to_json());
    response
}

/// Run one image request, timing each stage into the returned record.
async fn serve(state: &AppState, path: &str) -> (Response, RequestRecord) {
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
    let (ops, source_url) = match parse_operations(rest) {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_REQUEST.into_response(), record),
    };
    // The operation summary is everything before the source URL.
    record.ops = rest[..rest.len() - source_url.len()]
        .trim_end_matches('/')
        .to_string();

    // The verified path is content-addressed: a stored response can
    // be served without touching the loader or the engine.
    if let Some(hit) = state.cache.get(rest) {
        record.cache_hit = true;
        record.bytes_out = hit.bytes.len();
        return (respond(hit.format, hit.quality, hit.bytes), record);
    }

    record.load_ran = true;
    let load_start = Instant::now();
    let bytes = match state.loader.load(source_url).await {
        Ok(b) => b,
        Err(e) => {
            record.load = load_start.elapsed();
            record.fail = FailStage::Load;
            return (load_error_response(e), record);
        }
    };
    record.load = load_start.elapsed();
    record.bytes_in = bytes.len();

    record.process_ran = true;
    let process_start = Instant::now();
    let processed = match state.engine.process(&bytes, &ops) {
        Ok(p) => p,
        Err(e) => {
            record.process = process_start.elapsed();
            record.fail = FailStage::Engine;
            return (engine_error_response(e), record);
        }
    };
    record.process = process_start.elapsed();

    // Only successful responses are cached; errors never are.
    state.cache.put(
        rest.to_string(),
        CachedImage {
            bytes: processed.bytes.clone(),
            format: processed.format,
            quality: processed.quality,
        },
    );
    record.bytes_out = processed.bytes.len();
    (
        respond(processed.format, processed.quality, processed.bytes),
        record,
    )
}

/// Build a successful image response: content type, immutable cache
/// headers, and the auto-quality header when quality was chosen.
fn respond(format: ImageFormat, quality: Option<u8>, bytes: Vec<u8>) -> Response {
    let content_type = match format {
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::Png => "image/png",
        ImageFormat::WebP => "image/webp",
        ImageFormat::Jxl => "image/jxl",
    };
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    headers.insert(
        header::CACHE_CONTROL,
        "public, max-age=31536000, immutable".parse().unwrap(),
    );
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
        let png = test_png();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                png.len()
            );
            let _ = socket.write_all(head.as_bytes()).await;
            let _ = socket.write_all(&png).await;
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
