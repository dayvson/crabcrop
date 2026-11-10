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

/// Answer one HTTP/1.1 request the way a real origin does:
/// read the request head, write the response, then close the
/// write side. Writing and dropping the socket with the
/// request still unread makes the kernel answer with a reset
/// instead of a clean close, and on macOS that reset can reach
/// the client before it has read the whole response ("error
/// decoding response body", "error sending request").
async fn reply_and_close(socket: &mut tokio::net::TcpStream, head: &[u8], body: &[u8]) {
    let mut buf = [0u8; 1024];
    let mut seen = Vec::new();
    while !seen.windows(4).any(|w| w == b"\r\n\r\n") && seen.len() < 64 * 1024 {
        match socket.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => seen.extend_from_slice(&buf[..n]),
        }
    }
    let _ = socket.write_all(head).await;
    let _ = socket.write_all(body).await;
    let _ = socket.shutdown().await;
}

async fn spawn_bytes_server(png: Vec<u8>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let png = Arc::new(png);
    tokio::spawn(async move {
        // Serves every connection: cache-bypassing requests
        // (meta mode) fetch the source more than once.
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let png = png.clone();
            tokio::spawn(async move {
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    png.len()
                );
                reply_and_close(&mut socket, head.as_bytes(), &png).await;
            });
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
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(80, 40, image::Rgb([255, 0, 0])));
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        let img = image::RgbaImage::from_pixel(40, 20, image::Rgba([rgb[0], rgb[1], rgb[2], 255]));
        let image = oxideav_webp::WebpImage::from_rgba8(40, 20, img.as_raw().clone()).unwrap();
        oxideav_webp::Frame::new(image, Some(std::time::Duration::from_millis(delay_ms)))
    };
    let frames = vec![frame([255, 0, 0], 100), frame([0, 0, 255], 200)];
    let opts = oxideav_webp::EncodeOptions::default()
        .with_frame_mode(oxideav_webp::AnimFrameMode::Lossless);
    oxideav_webp::encode_animation(&frames, &opts).unwrap()
}

/// One-frame GIF bytes: the GIF animation probe's
/// negative case (issue #405).
fn still_gif_bytes() -> Vec<u8> {
    let img = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
        40,
        20,
        image::Rgba([255, 0, 0, 255]),
    ));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Gif).unwrap();
    buf.into_inner()
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
    let dir = crate::test_support::scratch_dir("server-video");
    let path = dir.join("encoder.sh");
    crate::test_support::write_script(&path, script);
    let encoder =
        crate::video::VideoEncoder::new(path, std::time::Duration::from_secs(30), 64 * 1024 * 1024)
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
        provenance_signer: None,
        config_fingerprint: "test".to_string(),
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        // The scripts last as long as the server: the
        // test's runtime drops this task when it ends.
        let _scripts = dir;
        axum::serve(listener, app(state)).await.unwrap();
    });
    format!("http://{addr}")
}

/// An app with a fake video decoder configured
/// (issue #214): its probe prints `report` (printf
/// arguments, already quoted) on stderr, with the
/// probe's `-dump_separator` after each stream's codec
/// name as the reference tool prints it (issue #402),
/// and its extraction answers with the test PNG, so the
/// whole frame path runs without a real decoder.
async fn spawn_app_with_frame_decoder(report: &str) -> String {
    let dir = crate::test_support::scratch_dir("server-frames");
    std::fs::write(dir.join("frame.png"), test_png()).unwrap();
    let frame_path = dir.join("frame.png").display().to_string();
    let script = format!(
        "#!/bin/sh\ncat > /dev/null\nsep=', '\nprev=\nfor a in \"$@\"; do\n  if [ \"$a\" = \"image2pipe\" ]; then cat \"{frame_path}\"; exit 0; fi\n  if [ \"$prev\" = -dump_separator ]; then sep=\"$a\"; fi\n  prev=\"$a\"\ndone\nprintf '%s\\n' {report} | sed \"/^  Stream #/s/, /$sep/\" >&2\nexit 1\n"
    );
    let path = dir.join("decoder.sh");
    crate::test_support::write_script(&path, &script);
    let extractor =
        crate::frames::VideoFrameExtractor::new(path, std::time::Duration::from_secs(30)).unwrap();
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
        provenance_signer: None,
        config_fingerprint: "test".to_string(),
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        // The scripts last as long as the server: the
        // test's runtime drops this task when it ends.
        let _scripts = dir;
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
    let outside_dir = crate::test_support::scratch_dir("srv-outside");
    let outside = outside_dir.join("secret.jpg");
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
    let (src, _hits) = spawn_counting_server(200, "video/mp4", FAKE_VIDEO_SOURCE.to_vec()).await;
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
    let (src, _hits) = spawn_counting_server(200, "video/mp4", FAKE_VIDEO_SOURCE.to_vec()).await;
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
    let dir = crate::test_support::scratch_dir("server-pages");
    std::fs::write(dir.join("page.png"), test_png()).unwrap();
    let page_path = dir.join("page.png").display().to_string();
    let write_tool = |name: &str, script: String| {
        let path = dir.join(name);
        crate::test_support::write_script(&path, &script);
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
        provenance_signer: None,
        config_fingerprint: "test".to_string(),
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        // The scripts last as long as the server: the
        // test's runtime drops this task when it ends.
        let _scripts = dir;
        axum::serve(listener, app(state)).await.unwrap();
    });
    format!("http://{addr}")
}

/// Bytes that carry the PDF header (the tools are
/// fake; the document is never really parsed).
const FAKE_PDF_SOURCE: &[u8] = b"%PDF-1.4\nfake document bytes for the server tests";

#[tokio::test]
async fn page_urls_render_the_page_and_cache_it() {
    let (src, hits) = spawn_counting_server(200, "application/pdf", FAKE_PDF_SOURCE.to_vec()).await;
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
    let (src, _h) = spawn_counting_server(200, "application/pdf", FAKE_PDF_SOURCE.to_vec()).await;
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
    let (src, _h) = spawn_counting_server(200, "application/pdf", FAKE_PDF_SOURCE.to_vec()).await;
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
    let (src, hits) = spawn_counting_server(200, "application/pdf", FAKE_PDF_SOURCE.to_vec()).await;
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
    let (src, hits) = spawn_counting_server(200, "application/pdf", FAKE_PDF_SOURCE.to_vec()).await;
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
    let base = spawn_app_with_video("#!/bin/sh\ncat > /dev/null\nprintf 'SERVER-VIDEO'\n").await;
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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

/// The negotiation of a plain request: no hints, no
/// Accept, no meta.
fn plain_negotiation(ops: &[crate::url::Operation]) -> Negotiated {
    Negotiated {
        directives: crate::url::RequestDirectives::analyze(ops),
        preferred: None,
        effective_dpr: 1.0,
        hinted_width: None,
        save_data_applies: false,
        meta: false,
        debug: false,
        placeholders: false,
    }
}

#[test]
fn an_ordinary_url_keeps_its_cache_key() {
    let remainder = "300x200/filters:grayscale/http://x/heroes/cat.jpg";
    let (ops, source, _, _) = parse_operations_with_expiry(remainder).unwrap();
    let key = cache_key_for("test", &ops, source, &plain_negotiation(&ops));
    assert_eq!(
        key,
        format!(
            "{}#cfgtest#source",
            crate::url::canonical_form(&ops, source)
        )
    );
}

#[test]
fn a_padded_source_is_keyed_by_a_bounded_digest() {
    // Empty path segments survive into the source, so a
    // padded source used to put its whole length into the
    // key: a 58 KB key for a 70-byte entry.
    let source = format!("http://origin{}t.png", "/".repeat(58_000));
    let remainder = format!("filters:format(png)/{source}");
    let (ops, parsed, _, _) = parse_operations_with_expiry(&remainder).unwrap();
    assert_eq!(parsed, source);
    let key = cache_key_for("test", &ops, parsed, &plain_negotiation(&ops));
    assert!(
        key.len() <= crate::cache::MAX_KEY_BASE_BYTES + "#cfgtest".len(),
        "a {}-byte key",
        key.len()
    );
    // Another padding is another entry.
    let longer = format!("http://origin{}t.png", "/".repeat(58_001));
    let other = cache_key_for("test", &ops, &longer, &plain_negotiation(&ops));
    assert_ne!(key, other);
}

/// A source server that reads each whole request head before
/// it answers, however long the request line, and counts the
/// requests.
async fn spawn_reading_counting_server(body: Vec<u8>) -> (String, Arc<AtomicUsize>) {
    let hits = Arc::new(AtomicUsize::new(0));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let counted = hits.clone();
    let body = Arc::new(body);
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            counted.fetch_add(1, Ordering::SeqCst);
            let body = body.clone();
            tokio::spawn(async move {
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                reply_and_close(&mut socket, head.as_bytes(), &body).await;
            });
        }
    });
    (format!("http://{addr}"), hits)
}

#[tokio::test]
async fn an_over_long_url_caches_and_purges_under_its_bounded_key() {
    // The serve path and the purge path bound the key the
    // same way, so a URL past the key bound still caches,
    // and its purge still finds the entry.
    let (origin, hits) = spawn_reading_counting_server(test_png()).await;
    let base = spawn_app(false).await;
    let remainder = format!("300x200/{origin}{}src.png", "/".repeat(6_000));
    assert!(remainder.len() > crate::cache::MAX_KEY_BASE_BYTES);
    let url = format!("{base}/{}/{remainder}", sign(&remainder));
    for _ in 0..2 {
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), 200);
    }
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "the second request is a hit"
    );
    let purge_sig = crate::url::purge_signature(SECRET, &remainder);
    let resp = client()
        .post(format!("{base}/purge/{purge_sig}/{remainder}"))
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
        "the purge evicted the entry"
    );
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
        "crabcrop/purge-key".to_string(),
        format!("crabcrop/300x50/{src}"),
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

/// Every derived subkey (purge, purge-group, upload, seal,
/// cache entry) is the HMAC of a label under the reserved
/// segment. An image or explain request whose remainder
/// begins with that segment answers the same 400 whatever
/// its signature (the right one, a wrong one, a keyed one,
/// `unsafe`): the answer never depends on the signature.
#[tokio::test]
async fn a_reserved_remainder_is_refused_before_its_signature_is_checked() {
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app(true).await;
    for remainder in [
        "crabcrop/purge-key".to_string(),
        "crabcrop/purge-group-key".to_string(),
        "crabcrop/upload-key".to_string(),
        "crabcrop/seal-key".to_string(),
        "crabcrop/entry-key".to_string(),
        "crabcrop".to_string(),
        format!("crabcrop/300x50/{src}"),
    ] {
        let right = sign(&remainder);
        for segment in [
            right.clone(),
            "0".repeat(64),
            format!("k2:{right}"),
            "unsafe".to_string(),
        ] {
            for route in ["", "explain/"] {
                let resp = client()
                    .get(format!("{base}/{route}{segment}/{remainder}"))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(
                    resp.status(),
                    StatusCode::BAD_REQUEST,
                    "{route}{segment}/{remainder}"
                );
                assert_eq!(
                    resp.text().await.unwrap(),
                    "an image URL may not begin with the reserved segment crabcrop",
                    "{route}{segment}/{remainder}"
                );
            }
        }
    }
    assert_eq!(hits.load(Ordering::SeqCst), 0, "nothing was fetched");
    // A remainder that only names a label further in is an
    // ordinary image URL: a wrong signature is still a 403.
    let remainder = format!("300x50/{src}");
    let resp = client()
        .get(format!("{base}/{}/{remainder}", "0".repeat(64)))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
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
    let body: serde_json::Value = serde_json::from_str(&resp_json.text().await.unwrap()).unwrap();
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
            LoadError::InvalidSource("a cause".to_string()),
            StatusCode::BAD_REQUEST,
            "invalid_source",
            "invalid source URL".to_string(),
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
        (
            LoadError::Busy,
            StatusCode::SERVICE_UNAVAILABLE,
            "server_busy",
            "server busy: too many source images held in memory".to_string(),
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
        (
            EngineError::ProvenanceUnavailable("a cause".to_string()),
            StatusCode::SERVICE_UNAVAILABLE,
            "provenance_unavailable",
            "provenance signing is unavailable".to_string(),
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
        let body: serde_json::Value = serde_json::from_str(&error_body_text(resp).await).unwrap();
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
        let body: serde_json::Value = serde_json::from_str(&error_body_text(resp).await).unwrap();
        assert_eq!(body["error"]["code"], code);
        assert_eq!(body["error"]["message"], message);
    }
    // The provenance outage class carries Retry-After in
    // both modes, and only it among the engine classes
    // (issue #467).
    for accept in [None, Some("application/json")] {
        for (e, ..) in engine_error_cases() {
            let outage = matches!(e, EngineError::ProvenanceUnavailable(_));
            let resp = engine_error_response(e, accept);
            let retry = resp.headers().get(header::RETRY_AFTER);
            if outage {
                assert_eq!(retry.unwrap(), "30");
            } else {
                assert!(retry.is_none(), "{:?}", resp.status());
            }
        }
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
    // src/server/ except tests.rs (this test module),
    // each cut at its test module: a split that moves
    // code into a new file there joins the scan by
    // construction, never by a hand-kept list going
    // stale.
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
                && (bytes[j].is_ascii_lowercase() || bytes[j].is_ascii_digit() || bytes[j] == b'_')
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
    let url_rs = include_str!("../url.rs");
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
    let mut heads: Vec<String> = vec![include_str!("../server.rs").to_string()];
    let server_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/server");
    let mut module_files: Vec<std::path::PathBuf> = std::fs::read_dir(server_dir)
        .expect("src/server exists")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .filter(|path| path.file_name().is_some_and(|name| name != "tests.rs"))
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
    // on the deduplicated set (66 codes today), so a
    // file dropping out of the scan shows up as codes
    // going missing, not as duplicate slack.
    assert!(
        distinct.len() >= 60,
        "derived only {} error codes from the sources",
        distinct.len()
    );
    let docs = include_str!("../../docs/responses.md");
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
    // 42 documented operational codes, all derived.
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
    let (src_bad, _hits) = spawn_counting_server(200, "image/png", b"not an image".to_vec()).await;
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
async fn source_text_the_server_cannot_fetch_answers_400_invalid_source() {
    // Issue #467: a source URL that does not parse, names
    // a scheme the loader does not fetch, or is a
    // malformed bucket locator is the caller's fault, so
    // it answers 400, not the 502 a failing origin gets.
    // Signed, so the signature is not what fails. The
    // last two are a segment kind this server does not
    // know, a prefix and a bare form: they read as source
    // text and answer the same (spec/url-signing.md,
    // Versioning).
    let base = spawn_app(false).await;
    for source in [
        "ftp://8.8.8.8/a.jpg",
        "notaurl",
        "s3://",
        "gs://x",
        "newop:7/https://example.com/a.jpg",
        "newop/https://example.com/a.jpg",
    ] {
        let remainder = format!("100x50/{source}");
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
            StatusCode::BAD_REQUEST,
            "invalid_source",
            "invalid source URL",
        )
        .await;
        // The explain probe goes through the same loader
        // and answers the same.
        let url = format!("{base}/explain/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(&url)
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{source}");
        let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "invalid_source", "{source}");
    }
    // Only the first hop is the caller's text: an origin
    // that redirects to ftp:// is a failing origin, 502.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let mut buf = [0u8; 2048];
            let _ = socket.read(&mut buf).await;
            let _ = socket
                .write_all(
                    b"HTTP/1.1 302 Found\r\nLocation: ftp://8.8.8.8/a.jpg\r\n\
                      Content-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await;
        }
    });
    let remainder = format!("100x50/http://{addr}/src.png");
    let url = format!("{base}/{}/{remainder}", sign(&remainder));
    let json = client()
        .get(&url)
        .header("accept", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(json.status(), StatusCode::BAD_GATEWAY);
    let body: serde_json::Value = serde_json::from_str(&json.text().await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "source_fetch_failed");
}

#[tokio::test]
async fn an_invalid_source_does_not_fall_back() {
    // Issue #467: an invalid source is a 400, and a 400
    // stays honest (docs/responses.md, Fallback image):
    // the fallback stands in for a source that could not
    // be fetched or decoded, not for a URL that names
    // nothing fetchable.
    let base = spawn_app_with_fallback().await;
    let remainder = "100x50/notaurl";
    let url = format!("{base}/{}/{remainder}", sign(remainder));
    let resp = client().get(&url).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert!(resp.headers().get("x-crabcrop-fallback").is_none());
    // A source that fails at the origin still falls back.
    let (src404, _hits) = spawn_counting_server(404, "image/png", test_png()).await;
    let remainder = format!("100x50/{src404}");
    let url = format!("{base}/{}/{remainder}", sign(&remainder));
    let resp = client().get(&url).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers().get("x-crabcrop-fallback").unwrap(), "1");
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

fn preset_test_map() -> crate::url::Presets {
    [
        ("small".to_string(), "30x20".parse().unwrap()),
        ("gray".to_string(), "filters:grayscale".parse().unwrap()),
    ]
    .into_iter()
    .collect()
}

async fn spawn_app_with_presets(presets: crate::url::Presets, only_presets: bool) -> String {
    spawn_app_with_presets_and_roots(presets, only_presets, std::collections::HashMap::new()).await
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
    presets: crate::url::Presets,
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        "crop:40x30:smart/filters:quality(80)".parse().unwrap(),
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
        serde_json::from_str(include_str!("../../spec/operation-vectors.json")).unwrap();
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
        let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
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
/// TypeScript explainer, the playground's build of it) runs
/// the same files in its own suite; a disagreement anywhere
/// fails the gate here or there, and the mismatch list is
/// printed whole so one run shows every divergence.
/// These tests were put here rather than in url.rs while
/// the fuzz crate included url.rs textually, without the
/// explain module or serde_json in scope; since the
/// library target (issue #417) nothing keeps them here.
fn load_spec(name: &str) -> serde_json::Value {
    let text = match name {
        "explain-verdicts.json" => include_str!("../../spec/explain-verdicts.json"),
        "operation-vectors.json" => include_str!("../../spec/operation-vectors.json"),
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
    let presets: crate::url::Presets = [
        ("small".to_string(), "30x20".parse().unwrap()),
        ("gray".to_string(), "filters:grayscale".parse().unwrap()),
        (
            "card".to_string(),
            "crop:1200x630:smart/filters:quality(80)".parse().unwrap(),
        ),
    ]
    .into_iter()
    .collect();
    let src = "https://example.com/a.jpg";
    for (name, definition) in &presets {
        let spelled = format!("preset:{name}/{src}");
        let expanded = crate::url::expand_presets(&spelled, &presets, false).unwrap();
        assert_eq!(expanded, format!("{}/{src}", definition.as_str()));
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
    let err = crate::url::expand_presets(&format!("300x200/{src}"), &presets, true).unwrap_err();
    assert_eq!(err.code(), "presets_only");
}

#[tokio::test]
async fn unknown_presets_are_a_bad_request() {
    let src = spawn_image_server().await;
    let base = spawn_app_with_presets(preset_test_map(), false).await;
    let resp = get_signed(&base, &format!("preset:nope/{src}")).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn thousands_of_distinct_unknown_presets_are_still_unknown_preset() {
    // Issue #409: the cache-group scan runs before
    // expansion, and used to de-duplicate quadratically.
    // A presets-only remainder of about 5,400 distinct
    // names (just under hyper's URI cap) answers the same
    // 400 unknown_preset, naming the first unknown, on
    // the image route and on /explain, without a fetch.
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app_with_presets(preset_test_map(), true).await;
    let mut remainder = String::from("preset:small/");
    for i in 0..5_400 {
        remainder.push_str(&format!("preset:{i:03x}/"));
    }
    remainder.push_str(&src);
    assert!(remainder.len() > 60_000, "{}", remainder.len());
    for route in ["", "explain/"] {
        let resp = client()
            .get(format!("{base}/{route}{}/{remainder}", sign(&remainder)))
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "/{route}");
        let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "unknown_preset", "/{route}");
        assert_eq!(body["error"]["message"], "unknown preset: 000", "/{route}");
    }
    assert_eq!(hits.load(Ordering::SeqCst), 0);
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
        let (status, body) = get_raw(&base, &format!("/{}/{remainder}", sign(&remainder))).await;
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
                reply_and_close(&mut socket, head.as_bytes(), &body).await;
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
    let base = spawn_app_with_embeddings(Some(Arc::new(FakeEmbedding)), Some(Arc::new(head))).await;
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
    let head =
        crate::ai::AestheticHead::from_weights(vec![1.0, 0.0, 0.0], 0.5, "l14-head".to_string())
            .unwrap();
    let src = spawn_image_server().await;
    let base = spawn_app_with_embeddings(Some(Arc::new(FakeEmbedding)), Some(Arc::new(head))).await;
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
async fn an_objects_crop_past_the_class_cap_is_a_400_on_both_routes() {
    // Issue #408: a 33rd distinct class is the grammar's
    // 400 invalid_crop on the image route and on
    // /explain, before any fetch, as at 0.9.10, not a
    // silent truncation that reaches the detector (the
    // 501 below) with classes the caller never sees
    // dropped.
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app(false).await;
    let names = |n: usize| {
        (0..n)
            .map(|i| format!("c{i}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    for n in [33, 40] {
        let remainder = format!("crop:100x100:objects({})/{src}", names(n));
        for route in ["", "explain/"] {
            let resp = client()
                .get(format!("{base}/{route}{}/{remainder}", sign(&remainder)))
                .header("accept", "application/json")
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "/{route} {n}");
            let body: serde_json::Value =
                serde_json::from_str(&resp.text().await.unwrap()).unwrap();
            assert_eq!(body["error"]["code"], "invalid_crop", "/{route} {n}");
        }
    }
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    // 32 distinct classes, spelled with repeats, pass
    // the grammar: the image route reaches the missing
    // detector's 501 and /explain reports the folded set.
    let remainder = format!("crop:100x100:objects({},C0,c31)/{src}", names(32));
    let resp = client()
        .get(format!("{base}/{}/{remainder}", sign(&remainder)))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    let resp = client()
        .get(format!("{base}/explain/{}/{remainder}", sign(&remainder)))
        .header("accept", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = resp.text().await.unwrap();
    assert!(
        text.contains(&format!("objects({}):1", names(32))),
        "explain: {text}"
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
async fn ai_render_meta_sections_acquire_the_gate_after_the_render() {
    // A removebg request (AI use) answering meta on a
    // one-permit AI gate: the annotations must not wait
    // out the full gate timeout and silently drop both
    // the tags and the detections (audit R2-29). The
    // render's permit drains with its job, so by
    // annotation time the gate is free and each section
    // acquires the one permit in turn.
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
    presets: crate::url::Presets,
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
    // Hints past the ceiling all resolve to it (issue
    // #403), so they share one cache entry, and rounding
    // up never crosses the ceiling, let alone the
    // parser's cap.
    assert_eq!(quantize_width(3800, 100), 3800);
    assert_eq!(quantize_width(3801, 100), MAX_HINT_WIDTH);
    assert_eq!(quantize_width(5000, 100), MAX_HINT_WIDTH);
    assert_eq!(quantize_width(16383, 2000), MAX_HINT_WIDTH);
    assert_eq!(quantize_width(16383, 1), MAX_HINT_WIDTH);
}

#[test]
fn hinted_widths_have_bounded_cardinality() {
    // Every width the hint headers accept resolves to
    // one of a bounded set of keys (issue #403): at the
    // default step, the 38 multiples of 100 up to 3800
    // plus the 3840 ceiling, where it used to be 164
    // steps up to the parser cap.
    let distinct: std::collections::BTreeSet<u32> = (1..=crate::url::MAX_DIMENSION)
        .map(|w| quantize_width(w, 100))
        .collect();
    assert_eq!(distinct.len(), 39);
    assert_eq!(distinct.last(), Some(&MAX_HINT_WIDTH));
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
        LoadError::Forbidden("host localhost resolves to a non-public address (::1)".to_string()),
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

/// A black PNG of the given size: a source wider than
/// the hints the client-hint tests send.
fn png_of(width: u32, height: u32) -> Vec<u8> {
    let img = DynamicImage::new_rgb8(width, height);
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    buf.into_inner()
}

#[tokio::test]
async fn width_hint_sizes_an_unsized_url_and_separates_the_cache() {
    let (src, hits) = spawn_counting_server(200, "image/png", png_of(640, 480)).await;
    let base = spawn_app(false).await;
    // The source is 640x480 and the URL sizes nothing.
    let remainder = format!("filters:format(png)/{src}");
    let width_of = |resp: reqwest::Response| async {
        let bytes = resp.bytes().await.unwrap();
        image::load_from_memory(&bytes).unwrap().width()
    };
    // 250 quantizes to 300: proof the synthesized op ran.
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
    assert_eq!(width_of(resp).await, 640);
    assert_eq!(hits.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn a_width_hint_never_upscales_past_the_source() {
    let src = spawn_bytes_server(png_of(640, 480)).await;
    let base = spawn_app(false).await;
    let remainder = format!("filters:format(png)/{src}");
    let dims_of = |resp: reqwest::Response| async {
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.bytes().await.unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        (img.width(), img.height())
    };
    // Issue #403: a hint past the source used to upscale
    // to the hinted width (3000x2250 here). It returns
    // the source width now.
    let resp = get_with_headers(&base, &remainder, &[("Sec-CH-Width", "3000")]).await;
    assert_eq!(dims_of(resp).await, (640, 480));
    // The DPR hint runs into the same cap instead of
    // doubling past it (it used to answer 6000x4500).
    let resp = get_with_headers(
        &base,
        &remainder,
        &[("Sec-CH-Width", "3000"), ("Sec-CH-DPR", "2")],
    )
    .await;
    assert_eq!(dims_of(resp).await, (640, 480));
    // A hint whose DPR product crosses the source stops
    // at the source too: 400 * 2 is 800, past 640.
    let resp = get_with_headers(
        &base,
        &remainder,
        &[("Sec-CH-Width", "400"), ("Sec-CH-DPR", "2")],
    )
    .await;
    assert_eq!(dims_of(resp).await, (640, 480));
    // Below the source the hint and DPR still size as
    // before: 300 * 2 is 600.
    let resp = get_with_headers(
        &base,
        &remainder,
        &[("Sec-CH-Width", "300"), ("Sec-CH-DPR", "2")],
    )
    .await;
    assert_eq!(dims_of(resp).await, (600, 450));
    // An explicit size in the URL is not a hint: the
    // engine's upscale posture is unchanged for it.
    let resp = get_with_headers(&base, &format!("1280x0/{src}"), &[]).await;
    assert_eq!(dims_of(resp).await, (1280, 960));
}

#[tokio::test]
async fn a_width_hint_rasterizes_an_svg_at_the_hinted_width() {
    // A vector has no pixels to invent: an SVG rasterizes
    // at the size the request heads for, and the
    // `no_upscale` cap is that raster, the same as for an
    // explicit `filters:no_upscale` in the URL. So a hint
    // past the intrinsic 100x10 still renders crisp at
    // the hinted width, and the hint ceiling (issue #403)
    // is what bounds it.
    let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="10"><rect width="100" height="10" fill="#00ff00"/></svg>"##;
    let (src, _) = spawn_counting_server(200, "image/svg+xml", svg.to_vec()).await;
    let base = spawn_app(false).await;
    let remainder = format!("filters:format(png)/{src}");
    let dims_of = |resp: reqwest::Response| async {
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.bytes().await.unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        (img.width(), img.height())
    };
    let resp = get_with_headers(&base, &remainder, &[("Sec-CH-Width", "300")]).await;
    assert_eq!(dims_of(resp).await, (300, 30));
    let explicit = format!("300x0/filters:no_upscale:format(png)/{src}");
    let resp = get_with_headers(&base, &explicit, &[]).await;
    assert_eq!(dims_of(resp).await, (300, 30));
    let resp = get_with_headers(&base, &remainder, &[("Sec-CH-Width", "9000")]).await;
    assert_eq!(dims_of(resp).await, (MAX_HINT_WIDTH, 384));
}

#[tokio::test]
async fn hints_past_the_ceiling_share_one_cache_entry() {
    let (src, hits) = spawn_counting_server(200, "image/png", png_of(640, 480)).await;
    let base = spawn_app(false).await;
    let remainder = format!("filters:format(png)/{src}");
    // Issue #403: every hint past MAX_HINT_WIDTH resolves
    // to it before the key is built, so a client cycling
    // through large widths gets one render and one entry,
    // not one per quantization step.
    let resp = get_with_headers(&base, &remainder, &[("Sec-CH-Width", "5000")]).await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_eq!(resp.status(), StatusCode::OK);
    for width in ["3900", "9000", "16383"] {
        let resp = get_with_headers(&base, &remainder, &[("Sec-CH-Width", width)]).await;
        assert_eq!(
            header(&resp, "cache-status"),
            "crabcrop; hit",
            "hint {width}"
        );
    }
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    // Below the ceiling, a different step is its own
    // entry as before.
    let resp = get_with_headers(&base, &remainder, &[("Sec-CH-Width", "3700")]).await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_eq!(hits.load(Ordering::SeqCst), 2);
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
    .unwrap_err()
    .to_string();
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
    .unwrap_err()
    .to_string();
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
    .unwrap_err()
    .to_string();
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
        .unwrap_err()
        .to_string();
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
    .unwrap_err()
    .to_string();
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
    .unwrap_err()
    .to_string();
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
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("invalid filters segment: filters:bogus(1)"),
        "{err}"
    );
    // A well-formed signature from the wrong secret: the
    // server refuses the meta fetch and the CLI reports it.
    let remainder = format!("50x50/{src}");
    let forged = format!(
        "{base}/{}/{remainder}",
        crabcrop_url::sign(b"some-other-secret", &remainder).unwrap()
    );
    let err = crate::cli::exec(&cli(&["meta", &forged]))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("HTTP 403"), "{err}");
    // No secret for a remainder target: refused locally.
    for verb in ["meta", "purge"] {
        let err = crate::cli::exec(&cli(&[verb, "--base", &base, &remainder]))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("no secret"), "{verb}: {err}");
    }
    // Argument shape: missing and extra targets.
    let err = crate::cli::exec(&cli(&["meta", "--secret", "test-secret"]))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("meta needs an argument"), "{err}");
    let err = crate::cli::exec(&cli(&[
        "purge",
        "--secret",
        "test-secret",
        &remainder,
        &remainder,
    ]))
    .await
    .unwrap_err()
    .to_string();
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
    .unwrap_err()
    .to_string();
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
    let dir = crate::test_support::scratch_dir("srv-tier");
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
async fn sealed_preset_responses_carry_no_cache_group() {
    // Issue #454: a sealed URL hides its recipe, preset
    // names included, so neither its miss nor its hit
    // lists them in Cache-Group. The entry still joins the
    // group (groups live on the cache entry, not in the
    // header), so purge-group reaches it, and the plain
    // spelling of the same render keeps the header.
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app_with_presets(preset_test_map(), false).await;
    let remainder = format!("preset:small/{src}");
    let sealed_url = format!("{base}/{}", sealed_path(&remainder, Some("photo.png")));
    let resp = client().get(&sealed_url).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert!(resp.headers().get("cache-group").is_none());
    let resp = client().get(&sealed_url).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
    assert!(resp.headers().get("cache-group").is_none());
    // An entry only ever served sealed is in the group.
    let sig = crate::url::purge_group_signature(SECRET, "small");
    let resp = client()
        .post(format!("{base}/purge-group/{sig}/small"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.text().await.unwrap().contains("purged 1"));
    let resp = client().get(&sealed_url).send().await.unwrap();
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert!(resp.headers().get("cache-group").is_none());
    assert_eq!(hits.load(Ordering::SeqCst), 2);
    // The plain spelling hits the same entry and names
    // its group.
    let plain_url = format!("{base}/{}/{remainder}", sign(&remainder));
    let resp = client().get(&plain_url).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
    assert_eq!(header(&resp, "cache-group"), "\"small\"");
    // And a sealed hit after the plain one still names none.
    let resp = client().get(&sealed_url).send().await.unwrap();
    assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
    assert!(resp.headers().get("cache-group").is_none());
    assert_eq!(hits.load(Ordering::SeqCst), 2);
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
    let dir = crate::test_support::scratch_dir("srv-gdisk");
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
    spawn_app_on_store(test_store(dir)).await
}

/// The disk store the storage tests' apps run on, rooted
/// at `dir`.
fn test_store(dir: &std::path::Path) -> Arc<crate::storage::DiskStore> {
    Arc::new(
        crate::storage::DiskStore::open(
            dir,
            16 * 1024 * 1024,
            crate::url::entry_key(b"disk-key-test"),
        )
        .unwrap(),
    )
}

/// An app on the disk store `store`, which the test keeps a
/// handle to.
async fn spawn_app_on_store(store: Arc<crate::storage::DiskStore>) -> String {
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
        store: Some(store),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        shared_store: Some(Arc::new(
            crate::s3store::S3Store::new(
                "results".into(),
                String::new(),
                Some(endpoint.into()),
                "us-east-1".into(),
                None,
                crate::url::entry_key(b"disk-key-test"),
            )
            .unwrap(),
        )),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        written: None,
    }
}

/// How many cache entry objects the fake bucket holds,
/// leaving out purge tombstones (issue #461).
fn entry_objects(fake: &crate::s3store::test_support::FakeS3) -> usize {
    fake.objects
        .lock()
        .unwrap()
        .keys()
        .filter(|k| k.ends_with(".bin"))
        .count()
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
    let dir = crate::test_support::scratch_dir("srv-drop3");
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

/// Holds the only blocking thread of a runtime built with
/// `max_blocking_threads(1)` until it is dropped (issue
/// #546). The pool queues every task past its thread cap
/// until a busy thread is free, so a disk operation the
/// code under test hands to the pool meanwhile waits
/// behind the gate, however the threads are scheduled.
/// Dropping the gate lets the thread go on every exit
/// path, so a failed check cannot leave the runtime's
/// drop waiting for it.
struct BlockingThreadGate {
    /// Never sent on: dropping it ends the held task.
    _release: std::sync::mpsc::Sender<()>,
}

impl BlockingThreadGate {
    /// Takes the pool's thread, and returns once the
    /// thread is held.
    async fn hold() -> Self {
        let (release, released) = std::sync::mpsc::channel::<()>();
        let (held_tx, held_rx) = tokio::sync::oneshot::channel();
        // Detached: the task holds the thread until the
        // gate drops `release`, waited on or not.
        drop(tokio::task::spawn_blocking(move || {
            let _ = held_tx.send(());
            let _ = released.recv();
        }));
        held_rx.await.expect("the gate takes the blocking thread");
        BlockingThreadGate { _release: release }
    }
}

#[test]
fn dropped_lookup_still_propagates_a_memory_hit_group_merge() {
    // One blocking thread, for the gate below to hold
    // (issue #546).
    let runtime = tokio::runtime::Builder::new_current_thread()
        .max_blocking_threads(1)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let fake = crate::s3store::test_support::start_fake().await;
        let dir = crate::test_support::scratch_dir("srv-drop4");
        let state = tier_test_state(Some(&dir), &fake.endpoint);
        let shared = state.shared_store.clone().unwrap();
        let key = "40x24/example.com/i.png";
        state.cache.put(key.to_string(), tier_test_entry());
        shared.put(key, &tier_test_entry()).await;
        // The memory hit's shared propagation is owed
        // before any wait (issue #347): one poll and a
        // drop cannot strand it. The disk merge the hit
        // then waits on queues behind the gate, so the
        // one poll finds the lookup waiting on it
        // (issue #546). Unheld, the merge could finish
        // on the blocking pool before the poll looked,
        // and the lookup returned instead of waiting.
        let gate = BlockingThreadGate::hold().await;
        poll_lookup_once(&state, key, &["thumb".to_string()]);
        drop(gate);
        shared_learns_group(&shared, key, "thumb").await;
        let _ = std::fs::remove_dir_all(&dir);
    });
}

#[tokio::test]
async fn disk_task_panic_degrades_and_counts_the_error() {
    let dir = crate::test_support::scratch_dir("srv-panic");
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

/// Holds every entry read of one disk store, the way a slow
/// disk holds a read, until the test lets them go or drops
/// the gate (issues #543, #545). The test used to hold a
/// read on a FIFO swapped in for an entry's file; the store
/// now refuses one.
///
/// Each held read sends the instant it was held. A hold also
/// ends on its own after [`DiskReadGate::BACKSTOP`], so a
/// read that wrongly runs on the runtime thread cannot keep
/// the test from reaching its checks, which then see that
/// the backstop fired. Dropping the gate lets every hold go,
/// so a failed check cannot leave a `#[tokio::test]`
/// runtime's drop waiting for a held blocking thread.
struct DiskReadGate {
    store: Arc<crate::storage::DiskStore>,
    shared: Arc<DiskReadGateShared>,
}

struct DiskReadGateShared {
    released: std::sync::Mutex<bool>,
    wake: std::sync::Condvar,
    backstopped: std::sync::atomic::AtomicBool,
}

impl DiskReadGate {
    /// How long a read is held when the test does not let it
    /// go itself. A read that blocks the runtime thread holds
    /// up /health until then, so the check fails within this;
    /// a healthy /health answers far inside it, even on a
    /// loaded runner.
    const BACKSTOP: std::time::Duration = std::time::Duration::from_secs(15);

    /// Starts holding every entry read of `store`.
    fn hold(
        store: &Arc<crate::storage::DiskStore>,
    ) -> (
        Self,
        tokio::sync::mpsc::UnboundedReceiver<std::time::Instant>,
    ) {
        let shared = Arc::new(DiskReadGateShared {
            released: std::sync::Mutex::new(false),
            wake: std::sync::Condvar::new(),
            backstopped: std::sync::atomic::AtomicBool::new(false),
        });
        let (held_tx, held_rx) = tokio::sync::mpsc::unbounded_channel();
        let hook = shared.clone();
        store.set_read_hold(Some(Arc::new(move || {
            let _ = held_tx.send(std::time::Instant::now());
            let released = hook.released.lock().unwrap();
            let (_released, wait) = hook
                .wake
                .wait_timeout_while(released, Self::BACKSTOP, |released| !*released)
                .unwrap();
            if wait.timed_out() {
                hook.backstopped
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            }
        })));
        let gate = DiskReadGate {
            store: store.clone(),
            shared,
        };
        (gate, held_rx)
    }

    /// Lets every held read go, and every later one through.
    fn release(&self) {
        *self.shared.released.lock().unwrap() = true;
        self.shared.wake.notify_all();
    }

    /// Whether a hold ended on its backstop instead of on the
    /// test's release.
    fn backstopped(&self) -> bool {
        self.shared
            .backstopped
            .load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Drop for DiskReadGate {
    fn drop(&mut self) {
        self.store.set_read_hold(None);
        self.release();
    }
}

/// Awaits `fut`, failing the test with `what` if it takes
/// longer than `limit` (issue #543): a stuck step names
/// itself instead of holding the suite.
async fn within<F: std::future::Future>(
    limit: std::time::Duration,
    what: &str,
    fut: F,
) -> F::Output {
    match tokio::time::timeout(limit, fut).await {
        Ok(out) => out,
        Err(_) => panic!("waited {limit:?} for {what}"),
    }
}

/// A disk read that blocks must not stall the runtime
/// worker (issue #347): the second instance's entry reads
/// are held, as a slow disk would hold them, and /health has
/// to answer while the pending request's read is. Every
/// request and the wait for the read have a deadline, and
/// the gate lets a held read go on every exit path, so a
/// failure is reported rather than hanging the suite (issue
/// #543).
#[tokio::test]
async fn blocked_disk_read_leaves_health_responsive() {
    let limit = std::time::Duration::from_secs(30);
    let dir = crate::test_support::scratch_dir("srv-held-read");
    let (src, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app_with_storage(&dir).await;
    let remainder = format!("25x25/{src}");
    let url_path = format!("{}/{remainder}", sign(&remainder));
    let resp = within(
        limit,
        "the first request to answer",
        client().get(format!("{base}/{url_path}")).send(),
    )
    .await
    .unwrap();
    assert_eq!(resp.status(), 200);
    // A second instance on the same directory: its
    // memory is cold, so its requests reach the disk
    // tier. Its index is built now, from the real file.
    let store = test_store(&dir);
    let base = spawn_app_on_store(store.clone()).await;
    // Declared after `dir`, so a failed check drops it
    // first: the read is let go before the directory is
    // removed.
    let (gate, mut held) = DiskReadGate::hold(&store);
    let pending_base = base.clone();
    let pending_path = url_path.clone();
    let pending = tokio::spawn(async move {
        client()
            .get(format!("{pending_base}/{pending_path}"))
            .send()
            .await
    });
    let held_at = within(
        limit,
        "the pending request to read the stored entry",
        held.recv(),
    )
    .await
    .expect("the gate stopped before any read came");
    let health = within(
        limit,
        "/health to answer while a disk read is held",
        client().get(format!("{base}/health")).send(),
    )
    .await
    .unwrap();
    assert_eq!(health.status(), 200);
    assert!(
        !gate.backstopped(),
        "/health answered only after the gate's {:?} backstop let the \
         held read go ({:?} after it was held): the runtime thread was \
         blocked while the read was held, by the read itself (#347) or \
         by a starved runner",
        DiskReadGate::BACKSTOP,
        held_at.elapsed()
    );
    gate.release();
    let resp = within(
        limit,
        "the pending request to answer once its read was let go",
        pending,
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        header(&resp, "cache-status"),
        "crabcrop; hit; detail=\"disk\"",
        "the held read was the stored entry's, and it served"
    );
    drop(gate);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn disk_results_survive_a_fresh_instance_and_purge_invalidates() {
    let dir = crate::test_support::scratch_dir("srv-store");
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

/// GET a signed remainder on the image route (`""`) or
/// the explain route (`"explain/"`), asking for the JSON
/// envelope: the status and the parsed body.
async fn get_signed_json(
    base: &str,
    route: &str,
    remainder: &str,
) -> (StatusCode, serde_json::Value) {
    let resp = client()
        .get(format!("{base}/{route}{}/{remainder}", sign(remainder)))
        .header("accept", "application/json")
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let body = serde_json::from_str(&resp.text().await.unwrap()).unwrap_or_default();
    (status, body)
}

#[tokio::test]
async fn over_cap_requests_answer_400_before_any_fetch() {
    // Issue #469: the cap used to be checked by the engine
    // alone, after the source was fetched and decoded, so an
    // over-cap URL cost a fetch, and repeating a preset made
    // one short to spell. It is checked on the expanded list
    // at parse time now: every over-cap spelling below
    // answers 400 on the image and explain routes, and the
    // counting origin sees no request at all.
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app_with_presets(preset_test_map(), false).await;
    let over_cap = [
        // One segment, one filter past the cap.
        format!("filters:{}", vec!["invert"; 101].join(":")),
        // 101 single-operation segments.
        vec!["30x20"; 101].join("/"),
        // The issue's evidence: 4,985 preset references, a
        // path of about 64 KB that expands to 4,985 sizes.
        vec!["preset:small"; 4_985].join("/"),
        // Metadata does not count and does not hide the
        // operations around it.
        format!(
            "exp:{}/{}/maxage:60",
            now_unix() + 3600,
            vec!["preset:gray"; 101].join("/")
        ),
        // The longest single filters segment a URL can carry.
        format!("filters:{}", vec!["invert"; 9_000].join(":")),
    ];
    let counts = [101, 101, 4_985, 101, 9_000];
    for (ops, count) in over_cap.iter().zip(counts) {
        let remainder = format!("{ops}/{src}");
        for route in ["", "explain/"] {
            let (status, body) = get_signed_json(&base, route, &remainder).await;
            let label = format!("{route}{}", &ops[..ops.len().min(40)]);
            assert_eq!(status, StatusCode::BAD_REQUEST, "{label}");
            assert_eq!(body["error"]["code"], "unsupported_operation", "{label}");
            assert_eq!(
                body["error"]["message"],
                format!("request has {count} operations, over the 100 operation limit"),
                "{label}"
            );
        }
    }
    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "an over-cap request fetched its source"
    );
    // Without the JSON ask the body is the plain message.
    let remainder = format!("{}/{src}", over_cap[0]);
    let resp = get_signed(&base, &remainder).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        resp.text().await.unwrap(),
        "request has 101 operations, over the 100 operation limit"
    );
    // A sealed spelling gets the uniform refusal, also
    // without a fetch.
    let resp = client()
        .get(format!("{base}/{}", sealed_path(&remainder, Some("a.png"))))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    assert_eq!(resp.text().await.unwrap(), "bad signature");
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    // Exactly at the cap still renders: the check refuses
    // only past it, and the server survived the barrage.
    let at_cap = format!("{}/{src}", vec!["30x20"; 100].join("/"));
    let resp = get_signed(&base, &at_cap).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn over_cap_counts_the_default_operations_before_any_fetch() {
    // Issue #469: the cap counts the merged list, defaults
    // included, on both routes, and a default the URL
    // overrides is dropped before counting.
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app_with_defaults("filters:format(webp)", false).await;
    // 100 of the URL's own plus the default format: 101.
    let over = format!("{}/{src}", vec!["filters:invert"; 100].join("/"));
    for route in ["", "explain/"] {
        let (status, body) = get_signed_json(&base, route, &over).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{route}");
        assert_eq!(
            body["error"]["message"], "request has 101 operations, over the 100 operation limit",
            "{route}"
        );
    }
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    // 99 plus the default fits.
    let fits = format!("{}/{src}", vec!["filters:invert"; 99].join("/"));
    let resp = get_signed(&base, &fits).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers().get("content-type").unwrap(), "image/webp");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    // The URL's own format drops the default one, so 99
    // plus the URL's format is 100, not 101.
    let own_format = format!(
        "{}/filters:format(png)/{src}",
        vec!["filters:invert"; 99].join("/")
    );
    let resp = get_signed(&base, &own_format).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers().get("content-type").unwrap(), "image/png");
    // A URL whose own operations already pass the cap is
    // refused while its presets expand, before the defaults
    // merge: the count names the URL's own operations.
    let own_over = format!("{}/{src}", vec!["filters:invert"; 101].join("/"));
    let fetched = hits.load(Ordering::SeqCst);
    for route in ["", "explain/"] {
        let (status, body) = get_signed_json(&base, route, &own_over).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{route}");
        assert_eq!(
            body["error"]["message"], "request has 101 operations, over the 100 operation limit",
            "{route}"
        );
    }
    assert_eq!(hits.load(Ordering::SeqCst), fetched);
}

#[tokio::test]
async fn repeated_preset_references_are_refused_before_expanding() {
    // A short URL that repeats a long preset used to expand
    // in full, then parse every operation it expanded to,
    // before the cap refused it: 7,000 references to a
    // 100-operation preset of 5x5 convolutions built and
    // parsed 700,000 operations (about 80 MB of text) per
    // request. Expansion now counts each reference by the
    // count settled when the presets loaded and refuses
    // before building anything, on the image and explain
    // routes alike, with the same answer as before.
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let kernel = format!("convolution({},1)", vec!["0.5"; 25].join(","));
    let presets: crate::url::Presets = [
        (
            "c".to_string(),
            format!("filters:{}", vec![kernel; 100].join(":"))
                .parse()
                .unwrap(),
        ),
        ("small".to_string(), "30x20".parse().unwrap()),
    ]
    .into_iter()
    .collect();
    let base = spawn_app_with_presets(presets, true).await;
    let remainder = format!("{}{src}", "preset:c/".repeat(7_000));
    for route in ["", "explain/"] {
        let started = std::time::Instant::now();
        let (status, body) = get_signed_json(&base, route, &remainder).await;
        let elapsed = started.elapsed();
        assert_eq!(status, StatusCode::BAD_REQUEST, "/{route}");
        assert_eq!(body["error"]["code"], "unsupported_operation", "/{route}");
        assert_eq!(
            body["error"]["message"], "request has 700000 operations, over the 100 operation limit",
            "/{route}"
        );
        // The refusal costs a pass over the URL's own text;
        // expanding and parsing it took seconds.
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "/{route} took {elapsed:?}"
        );
    }
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    // A realistic reference still renders, and the
    // 100-operation preset alone is within the cap.
    let resp = get_signed(&base, &format!("preset:small/{src}")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    let (status, body) = get_signed_json(&base, "explain/", &format!("preset:c/{src}")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn explain_forecasts_the_merged_default_operations() {
    // The image path runs the defaults merged ahead of the
    // URL's own operations, so the forecast has to walk
    // the same list (issue #468): a default format and a
    // default DPR decide the output the image request
    // answers with. The operation list stays the URL's own.
    let (src, _hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app_with_defaults("filters:format(webp):dpr(2)", false).await;
    let remainder = format!("50x0/{src}");
    let sig = sign(&remainder);
    let explain = client()
        .get(format!("{base}/explain/{sig}/{remainder}"))
        .header("Accept", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(explain.status(), StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&explain.text().await.unwrap()).unwrap();
    assert_eq!(json["forecast"]["format"], "webp");
    assert_eq!(json["forecast"]["format_from"], "default");
    assert_eq!(json["forecast"]["width"], 100);
    assert_eq!(json["forecast"]["height"], 60);
    let operations = json["operations"].as_array().unwrap();
    assert_eq!(operations.len(), 1, "{operations:?}");
    assert_eq!(operations[0]["segment"], "50x0");
    let canonical = json["canonical"].as_str().unwrap();
    assert!(
        canonical.starts_with("filters:format(webp):dpr(2)/50x0/"),
        "{canonical}"
    );
    // The image request agrees with the forecast.
    let image = client()
        .get(format!("{base}/{sig}/{remainder}"))
        .send()
        .await
        .unwrap();
    assert_eq!(image.status(), StatusCode::OK);
    assert_eq!(image.headers().get("content-type").unwrap(), "image/webp");
    let decoded = image::load_from_memory(&image.bytes().await.unwrap()).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (100, 60));

    // The HTML form says where the format came from, and
    // never that the URL spelled it.
    let html = client()
        .get(format!("{base}/explain/{sig}/{remainder}"))
        .header("Accept", "text/html")
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        html.contains("<li>Format: webp (from the server's default operations)</li>"),
        "{html}"
    );
    assert!(!html.contains("from the URL"), "{html}");

    // A format the URL names wins over the default, and the
    // report credits the URL.
    let remainder = format!("50x0/filters:format(png)/{src}");
    let sig = sign(&remainder);
    let body = client()
        .get(format!("{base}/explain/{sig}/{remainder}"))
        .header("Accept", "application/json")
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["forecast"]["format"], "png");
    assert_eq!(json["forecast"]["format_from"], "operation");
}

#[tokio::test]
async fn explain_survives_an_overflowing_forecast() {
    // A crop-rect takes u32 corners and each upscale
    // multiplies the running size: the forecast's pixel
    // count overflowed u64 here and the handler panicked,
    // dropping the connection (issue #468). It saturates
    // now, and a saturated count is over the budget.
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app(false).await;
    let sixteen = vec!["upscale(4)"; 16].join(":");
    for chain in [
        "crop-rect:0x0:4294967295x4294967295/filters:upscale(2)".to_string(),
        format!("crop-rect:0x0:4294967295x4294967295/filters:{sixteen}"),
    ] {
        let remainder = format!("{chain}/{src}");
        let sig = sign(&remainder);
        let resp = client()
            .get(format!("{base}/explain/{sig}/{remainder}"))
            .header("Accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{chain}");
        let json: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(json["forecast"]["pixels"], u64::MAX, "{chain}");
        assert_eq!(json["forecast"]["over_budget"], true, "{chain}");
    }
    assert_eq!(hits.load(Ordering::SeqCst), 2);
    let health = client().get(format!("{base}/health")).send().await.unwrap();
    assert_eq!(health.status(), StatusCode::OK);
}

#[tokio::test]
async fn explain_admits_a_video_source_by_a_default_frame() {
    // The image path admits a video source when its merged
    // operations select a frame, a default `frame:`
    // included. The explain probe admits by the same merged
    // list (issue #468), so it reports the source the image
    // request fetches instead of refusing it (on main it
    // answered 502). The operation list stays the URL's own.
    let (src, hits) = spawn_counting_server(200, "video/mp4", FAKE_VIDEO_SOURCE.to_vec()).await;
    let base = spawn_app_with_defaults("frame:1.5", false).await;
    let remainder = format!("100x0/{src}");
    let resp = client()
        .get(format!("{base}/explain/{}/{remainder}", sign(&remainder)))
        .header("Accept", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    assert_eq!(json["source"]["kind"], "video");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    let operations = json["operations"].as_array().unwrap();
    assert_eq!(operations.len(), 1, "{operations:?}");
    assert_eq!(operations[0]["segment"], "100x0");
    let canonical = json["canonical"].as_str().unwrap();
    assert!(canonical.starts_with("frame:1.5/100x0/"), "{canonical}");
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
    let new_unkeyed = crabcrop_url::sign(b"the-new-server-secret", &remainder).unwrap();
    let old_unkeyed = crabcrop_url::sign(b"the-old-server-secret", &remainder).unwrap();
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
    let dir = crate::test_support::scratch_dir("srv-shared");
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
    let dir = crate::test_support::scratch_dir("srv-p351");
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
        watermark_sources: Default::default(),
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
fn provenance_step_tells_an_authority_outage_from_a_signing_failure() {
    // Issue #467: a stalled timestamp authority is its
    // own engine error (a 503 a retry may clear), and a
    // failure that is not the authority's stays an
    // encode failure. The bytes here are not the PNG they
    // claim to be, so that signing fails before the
    // authority is asked.
    let (signer, release, staller) = crate::provenance::test_signer_with_a_stalled_authority();
    let job = provenance_job(Some(signer), false);
    let source = test_png();
    let mut processed = still_processed(source.clone(), ImageFormat::Png);
    let outage = apply_provenance(&job, &source, &mut processed).unwrap_err();
    let mut garbage = still_processed(b"not a png".to_vec(), ImageFormat::Png);
    let broken = apply_provenance(&job, &source, &mut garbage).unwrap_err();
    drop(release);
    assert_eq!(staller.join().unwrap(), 1, "only the real output asked");
    assert!(
        matches!(outage, EngineError::ProvenanceUnavailable(_)),
        "{outage:?}"
    );
    assert!(matches!(broken, EngineError::Encode(_)), "{broken:?}");
}

#[tokio::test]
async fn a_stalled_timestamp_authority_answers_503_provenance_unavailable() {
    // Issue #467: the output rendered, but the authority
    // signing depends on stopped answering after boot.
    // That is a dependency outage, not a broken encoder:
    // 503 provenance_unavailable with Retry-After, not
    // 500 encode_failed, and within the authority's
    // bound.
    let (signer, release, staller) = crate::provenance::test_signer_with_a_stalled_authority();
    let base = spawn_app_with_provenance_signer(signer).await;
    let src = spawn_bytes_server(test_png()).await;
    let remainder = format!("40x24/{src}");
    let url = format!("{base}/{}/{remainder}", sign(&remainder));
    let started = std::time::Instant::now();
    let json = client()
        .get(&url)
        .header("accept", "application/json")
        .send()
        .await
        .unwrap();
    let elapsed = started.elapsed();
    assert_eq!(json.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(json.headers().get(header::RETRY_AFTER).unwrap(), "30");
    let body: serde_json::Value = serde_json::from_str(&json.text().await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "provenance_unavailable");
    assert_eq!(
        body["error"]["message"],
        "provenance signing is unavailable"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "answered at the authority's bound, took {elapsed:?}"
    );
    // Errors are never cached: the plain request asks
    // the authority again and gets the same answer.
    let plain = client().get(&url).send().await.unwrap();
    assert_eq!(plain.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(plain.headers().get(header::RETRY_AFTER).unwrap(), "30");
    assert_eq!(
        plain.text().await.unwrap(),
        "provenance signing is unavailable"
    );
    drop(release);
    assert_eq!(staller.join().unwrap(), 2);
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
fn provenance_step_names_a_signed_source_without_its_store() {
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
    assert_eq!(
        output.manifests, 1,
        "the parent's store does not ride along"
    );
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
    spawn_app_with_provenance_signer(crate::provenance::test_signer()).await
}

/// An app that signs its outputs with `signer`.
async fn spawn_app_with_provenance_signer(
    signer: Arc<crate::provenance::ProvenanceSigner>,
) -> String {
    spawn_signing_app(signer, false).await
}

/// An app that signs its outputs with `signer`, under the
/// given metadata policy (`CRABCROP_METADATA=preserve` when
/// `preserve_metadata`).
async fn spawn_signing_app(
    signer: Arc<crate::provenance::ProvenanceSigner>,
    preserve_metadata: bool,
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
        preserve_metadata,
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
        provenance_signer: Some(signer),
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
async fn a_signed_crop_or_redaction_carries_nothing_of_its_source() {
    // The source is shaped like a camera export: a manifest
    // store whose claim thumbnail is the whole frame, with
    // EXIF location, a body serial and an author, plus an
    // XMP instance ID. A crop keeps only the right half; a
    // pixelated region covers the left. Under either
    // metadata policy the signed output names its parent and
    // carries nothing of the store: no thumbnail, no
    // assertion, no title, and its size does not follow the
    // store. The XMP instance ID is the one fact `preserve`
    // keeps, in the XMP packet that policy copies.
    let source = crate::provenance::test_source_with_credentials(256 * 1024);
    let src = spawn_bytes_server(source).await;
    for preserve in [false, true] {
        let base = spawn_signing_app(crate::provenance::test_signer(), preserve).await;
        for (ops, mime) in [
            ("crop-rect:100x0:200x150", "image/jpeg"),
            ("filters:pixelate_region(0,0,100,150)", "image/jpeg"),
            ("crop-rect:100x0:200x150/filters:format(png)", "image/png"),
        ] {
            let remainder = format!("{ops}/{src}");
            let url_path = format!("/{}/{remainder}", sign(&remainder));
            let resp = client()
                .get(format!("{base}{url_path}"))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200, "{ops}");
            let bytes = resp.bytes().await.unwrap().to_vec();
            for fact in crate::provenance::TEST_SOURCE_FACTS {
                if preserve && fact.starts_with("xmp.iid:") {
                    continue;
                }
                assert!(
                    !bytes.windows(fact.len()).any(|w| w == fact.as_bytes()),
                    "{ops} preserve={preserve} carries {fact}"
                );
            }
            let jpeg_starts = bytes
                .windows(3)
                .filter(|w| *w == [0xFF, 0xD8, 0xFF])
                .count();
            let own = usize::from(mime == "image/jpeg");
            assert_eq!(
                jpeg_starts, own,
                "{ops} preserve={preserve}: no embedded picture"
            );
            assert!(
                bytes.len() < 64 * 1024,
                "{ops} preserve={preserve}: a {}-byte output",
                bytes.len()
            );
            let summary = crate::provenance::read_manifest(&bytes, mime)
                .expect("the response carries a manifest");
            assert_eq!(summary.state, "valid", "{ops} preserve={preserve}");
            assert_eq!(summary.manifests, 1, "{ops} preserve={preserve}");
            assert_eq!(summary.ingredients, 1, "{ops} preserve={preserve}");
        }
    }
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

/// The orientation HEIC fixture with its major brand
/// rewritten to `mif1`, a generic HEIF brand the engine
/// decodes as HEIC like `heic` (issue #564). Its
/// compatible brands still list `heic`.
fn mif1_heic() -> Vec<u8> {
    let mut heic = include_bytes!("../../tests/fixtures/orient6.heic").to_vec();
    assert_eq!(&heic[4..12], b"ftypheic");
    heic[8..12].copy_from_slice(b"mif1");
    heic
}

/// A JPEG XL still as a bare codestream, the form the
/// engine writes.
fn bare_jxl() -> Vec<u8> {
    use crate::url::{Filter, FormatSpec, Operation};
    let jxl = Engine
        .process(
            &test_png(),
            &[Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Jxl,
            ))])],
        )
        .unwrap()
        .bytes;
    assert!(jxl.starts_with(&[0xFF, 0x0A]));
    jxl
}

/// The same still in the JPEG XL box container: the
/// signature box, an `ftyp` box, and the codestream in a
/// `jxlc` box. The container is the only JPEG XL form that
/// can carry a C2PA manifest.
fn container_jxl() -> Vec<u8> {
    let codestream = bare_jxl();
    let mut jxl = vec![
        0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A,
    ];
    jxl.extend_from_slice(&20u32.to_be_bytes());
    jxl.extend_from_slice(b"ftypjxl \0\0\0\0jxl ");
    jxl.extend_from_slice(&(8 + codestream.len() as u32).to_be_bytes());
    jxl.extend_from_slice(b"jxlc");
    jxl.extend_from_slice(&codestream);
    jxl
}

/// A 16x16 still in one of the image crate's formats
/// (RGBA: the ICO decoder refuses an embedded RGB PNG).
fn encoded_still(format: image::ImageFormat) -> Vec<u8> {
    let mut buf = std::io::Cursor::new(Vec::new());
    DynamicImage::new_rgba8(16, 16)
        .write_to(&mut buf, format)
        .unwrap();
    buf.into_inner()
}

/// A 100x60 SVG still.
const SVG_SOURCE: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="60"><rect width="100" height="60" fill="#00ff00"/></svg>"##;

/// `source` with a manifest of its own, signed with the
/// test identity as a `format` asset.
fn signed_source(source: &[u8], format: &str) -> Vec<u8> {
    crate::provenance::test_signer()
        .sign(
            source,
            format,
            &crate::provenance::SignRequest {
                ingredient: None,
                ai_edited: false,
            },
        )
        .unwrap()
}

/// The output half of the provenance a signing server's
/// meta document reports for `source` rendered to PNG, a
/// signed format whatever the source's own is, and the
/// parent ingredient the image response's manifest names.
/// The image response for the same URL is read back: its
/// active manifest must name one ingredient, the parent, in
/// `format`, the source's media type.
async fn signed_png_provenance(
    name: &str,
    source: Vec<u8>,
    format: &str,
) -> (serde_json::Value, serde_json::Value) {
    let base = spawn_app_with_signer().await;
    let src = spawn_bytes_server(source).await;
    let remainder = format!("40x24/filters:format(png)/{src}");
    let url = format!("{base}/{}/{remainder}", sign(&remainder));
    let resp = client()
        .get(&url)
        .header("Accept", "application/json")
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert_eq!(status, 200, "{name}: {body}");
    let meta: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(meta["format"], "png", "{name}");
    let output = meta["provenance"]["output"].clone();
    assert_eq!(output["state"], "valid", "{name}: {output}");
    let resp = client().get(&url).send().await.unwrap();
    assert_eq!(resp.status(), 200, "{name}");
    let png = resp.bytes().await.unwrap();
    let store: serde_json::Value =
        serde_json::from_str(&crate::provenance::manifest_json(&png, "image/png")).unwrap();
    let active = store["active_manifest"].as_str().unwrap();
    let ingredients = &store["manifests"][active]["ingredients"];
    assert_eq!(
        ingredients.as_array().map(Vec::len),
        Some(1),
        "{name}: one parent ingredient: {ingredients} (meta's output: {output})"
    );
    assert_eq!(ingredients[0]["relationship"], "parentOf", "{name}");
    assert_eq!(ingredients[0]["format"], format, "{name}");
    (output, ingredients[0].clone())
}

/// Whether a parent ingredient records that the source's
/// own manifest store was found and its signature checked.
fn parent_store_was_read(parent: &serde_json::Value) -> bool {
    parent["validation_results"]["activeManifest"]["success"]
        .as_array()
        .is_some_and(|codes| {
            codes
                .iter()
                .any(|status| status["code"] == "claimSignature.validated")
        })
}

#[tokio::test]
async fn a_mif1_heic_source_is_the_signed_outputs_parent_ingredient() {
    // Issue #564: the engine decodes every HEIF brand its
    // HEIC sniff accepts, but the provenance sniff knew
    // only `heic` and `heix` and took a `mif1` file for a
    // video, so its output was signed as a fresh image
    // with no parent.
    let (output, _) = signed_png_provenance("mif1 heic", mif1_heic(), "image/heic").await;
    assert_eq!(output["ingredients"], 1, "{output}");
}

#[tokio::test]
async fn a_container_jxl_source_is_the_signed_outputs_parent_ingredient() {
    // Issue #564: the provenance sniff checked a misspelled
    // container signature, so a JPEG XL in its box
    // container had no parent ingredient.
    let (output, _) = signed_png_provenance("container jxl", container_jxl(), "image/jxl").await;
    assert_eq!(output["ingredients"], 1, "{output}");
}

#[tokio::test]
async fn a_signed_container_jxl_sources_store_is_read_for_the_output() {
    // The container is the JPEG XL form that carries a
    // manifest, and it was the one whose store was dropped
    // (issue #564): now the source's own manifest is read,
    // and the output's parent ingredient records how it
    // validated. The store itself stays out of the output
    // (issue #610).
    let source = signed_source(&container_jxl(), "image/jxl");
    let (output, parent) = signed_png_provenance("signed container jxl", source, "image/jxl").await;
    assert_eq!(
        output["manifests"], 1,
        "the source's store stays out: {output}"
    );
    assert_eq!(output["ingredients"], 1, "{output}");
    assert!(parent_store_was_read(&parent), "{parent}");
}

#[tokio::test]
async fn a_signed_svg_or_tiff_sources_store_is_read_for_the_output() {
    // The SDK reads a manifest from an SVG and from a TIFF,
    // but both were signed with no parent (issue #564), so
    // the store a signed one carried was dropped. Now each
    // is the parent ingredient, and the output records how
    // its store validated, without carrying the store
    // (issue #610).
    for (name, source, format) in [
        ("signed svg", SVG_SOURCE.to_vec(), "image/svg+xml"),
        (
            "signed tiff",
            encoded_still(image::ImageFormat::Tiff),
            "image/tiff",
        ),
    ] {
        let source = signed_source(&source, format);
        let (output, parent) = signed_png_provenance(name, source, format).await;
        assert_eq!(
            output["manifests"], 1,
            "{name}: the source's store stays out: {output}"
        );
        assert_eq!(output["ingredients"], 1, "{name}: {output}");
        assert!(parent_store_was_read(&parent), "{name}: {parent}");
    }
}

#[tokio::test]
async fn every_image_source_is_the_signed_outputs_parent_ingredient() {
    // Issue #564: SVG, TIFF, BMP and ICO sources sniffed as
    // application/octet-stream and were signed with no
    // parent. A bare JPEG XL codestream, the form the
    // engine writes, failed the request with 500 instead:
    // the SDK's JPEG XL reader refuses a bare codestream
    // outright. Each is now the parent ingredient; the
    // SVG, TIFF and container JPEG XL readers would carry a
    // manifest along, and the others cannot hold one. The
    // ingredient names the source's media type, the bare
    // codestream's included, though the SDK is handed it as
    // a type no reader claims.
    for (name, source, format) in [
        ("svg", SVG_SOURCE.to_vec(), "image/svg+xml"),
        (
            "tiff",
            encoded_still(image::ImageFormat::Tiff),
            "image/tiff",
        ),
        ("bmp", encoded_still(image::ImageFormat::Bmp), "image/bmp"),
        (
            "ico",
            encoded_still(image::ImageFormat::Ico),
            "image/x-icon",
        ),
        ("bare jxl", bare_jxl(), "image/jxl"),
    ] {
        let (output, _) = signed_png_provenance(name, source, format).await;
        assert_eq!(output["ingredients"], 1, "{name}: {output}");
    }
}

#[test]
fn video_and_pdf_sources_start_a_fresh_chain() {
    // The two sources that are not ingredients: the output
    // is a frame or a page rendered from them, so its
    // manifest opens with a created action and names no
    // parent (issue #564 kept this boundary).
    let job = provenance_job(Some(crate::provenance::test_signer()), false);
    for (name, source) in [
        (
            "mp4",
            b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isommp41".to_vec(),
        ),
        ("pdf", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n".to_vec()),
    ] {
        let mut processed = still_processed(test_png(), ImageFormat::Png);
        apply_provenance(&job, &source, &mut processed).unwrap();
        let output = processed.provenance.unwrap().output.expect("signed");
        assert_eq!(output.ingredients, 0, "{name}");
        let json = crate::provenance::manifest_json(&processed.bytes, "image/png");
        assert!(json.contains("c2pa.created"), "{name}: {json}");
    }
}

#[test]
fn hostile_sources_in_the_newly_read_formats_do_not_take_the_job_down() {
    // Issue #564 hands more source formats to the SDK's
    // readers as ingredients. Malformed ones (truncated,
    // with box and directory sizes past the end) either
    // sign or fail the signing step; none panics or hangs.
    let mut cut_heic = mif1_heic();
    cut_heic.truncate(64);
    let mut huge_box_jxl = container_jxl();
    huge_box_jxl[32..36].copy_from_slice(&u32::MAX.to_be_bytes());
    let mut cut_jxl = container_jxl();
    cut_jxl.truncate(40);
    let mut far_ifd_tiff = encoded_still(image::ImageFormat::Tiff);
    far_ifd_tiff[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut cut_tiff = encoded_still(image::ImageFormat::Tiff);
    cut_tiff.truncate(12);
    let mut cut_bmp = encoded_still(image::ImageFormat::Bmp);
    cut_bmp.truncate(20);
    let mut cut_ico = encoded_still(image::ImageFormat::Ico);
    cut_ico.truncate(10);
    let sources: [(&str, Vec<u8>); 9] = [
        ("cut heic", cut_heic),
        ("jxl box past the end", huge_box_jxl),
        ("cut jxl", cut_jxl),
        ("tiff directory past the end", far_ifd_tiff),
        ("cut tiff", cut_tiff),
        ("cut bmp", cut_bmp),
        ("cut ico", cut_ico),
        (
            "unclosed svg",
            b"<svg xmlns=\"http://www.w3.org/2000/svg\"><metadata><c2pa:manifest>".to_vec(),
        ),
        ("cut codestream", vec![0xFF, 0x0A]),
    ];
    let job = provenance_job(Some(crate::provenance::test_signer()), false);
    for (name, source) in sources {
        let started = std::time::Instant::now();
        let mut processed = still_processed(test_png(), ImageFormat::Png);
        let result = apply_provenance(&job, &source, &mut processed);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "{name}: {result:?}"
        );
        if let Err(e) = result {
            assert!(matches!(e, EngineError::Encode(_)), "{name}: {e:?}");
        }
    }
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
        provenance_signer: None,
        config_fingerprint: "test".to_string(),
    };
    let job = ProcessJob {
        save_data: false,
        preserve_metadata: false,
        brand_color: None,
        ops: Vec::new(),
        preferred: None,
        watermark_sources: Default::default(),
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
    let dir = crate::test_support::scratch_dir("server-stuck-encode");
    let script_path = dir.join("encoder.sh");
    crate::test_support::write_script(
        &script_path,
        "#!/bin/sh\ncat > /dev/null\nsleep 9\nprintf 'x'\n",
    );
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
            watermark_sources: Default::default(),
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
    let processed = follow
        .unwrap_or_else(|e| panic!("the follow-up must get a genuine encode attempt, got {e:?}"));
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
async fn a_source_budget_share_is_held_until_the_job_actually_finishes() {
    // Issue #403: the charge serve_miss holds for a
    // source travels into the run with the bytes. A run
    // shed at its deadline leaves the detached job
    // holding the bytes, so the share must stay held
    // with it, and come back only once the job finishes
    // and has dropped them; freed at the shed, it would
    // let new loads buffer on top of bytes still in
    // memory.
    let state = lane_test_state(std::time::Duration::from_millis(300), 1, 1);
    let budget = crate::loader::SourceBudget::new(1 << 20);
    let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let mut job = lane_test_job(None, "filters:upscale(2)/example.com/i.png");
    job.upscale_provider = Some(Arc::new(BlockingUpscale {
        release: release.clone(),
    }));
    let png = test_png();
    let share = budget.charge(std::time::Duration::from_secs(1));
    share.grow_to(png.len()).await.unwrap();
    assert_eq!(budget.in_use(), png.len());
    let source = JobSource {
        bytes: png.clone(),
        share: Some(share),
    };
    let result = run_process(
        &state,
        &job,
        source,
        Lane::Process,
        std::time::Instant::now() + state.process_timeout,
        JobPermits::default(),
    )
    .await;
    assert!(
        matches!(result, Err(ProcessFailure::TimedOut)),
        "the blocked run must answer TimedOut at its deadline, got {result:?}"
    );
    assert_eq!(
        budget.in_use(),
        png.len(),
        "the share must stay with the detached job after the shed"
    );
    {
        let (lock, cvar) = &*release;
        *lock.lock().unwrap() = true;
        cvar.notify_all();
    }
    let mut back = false;
    for _ in 0..100 {
        if budget.in_use() == 0 {
            back = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(back, "the share must return when the job drains");
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
    let dir = crate::test_support::scratch_dir("server-held-permits");
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
        if state.video_gate.available_permits() == 2 && state.anim_gate.available_permits() == 1 {
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
    let dir = crate::test_support::scratch_dir("server-lane-bound");
    let log_path = dir.join("runs.log");
    let script_path = dir.join("encoder.sh");
    crate::test_support::write_script(
        &script_path,
        &format!(
            "#!/bin/sh\ncat > /dev/null\necho S >> '{}'\nsleep 6\necho E >> '{}'\nprintf 'x'\n",
            log_path.display(),
            log_path.display()
        ),
    );
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
    // The encoder logs its end marker before it exits,
    // and the detached job drops its permit only after
    // the encode returns, so the permit can still be
    // out for a moment after the second "E" lands. Wait
    // for it the way the drain test above does.
    let mut back = false;
    for _ in 0..200 {
        if state.slow_gate.available_permits() == 1 {
            back = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(
        back,
        "the slow-lane permit must return when the second job drains"
    );
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
        annotate_meta(&state, &processed),
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
        moderate_meta(&state, &processed),
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
        detect_meta(&state, &processed),
    )
    .await
    .is_err();
    assert_meta_slot_drains_with_inference(&state, &release, timed_out).await;
}

#[tokio::test]
async fn meta_sections_never_run_inference_without_an_ai_permit() {
    // Every meta section acquires its own AI permit; none
    // runs under one the request claims to hold (issue
    // #456). With the gate's only permit held elsewhere,
    // each section degrades after the gate wait instead
    // of reaching its provider: tags and detections are
    // omitted, and the explicit-outcome sections report
    // "busy". Once the permit returns, the same sections
    // answer, so the degrade is the gate's doing.
    let mut state = lane_test_state(std::time::Duration::from_secs(30), 4, 2);
    state.ai_gate = Arc::new(tokio::sync::Semaphore::new(1));
    state.tagger = Some(Arc::new(FakeTagger));
    state.detect = Some(Arc::new(FakeDetector));
    state.moderation = Some(Arc::new(FakeModeration));
    state.embeddings = Some(Arc::new(FakeEmbedding));
    state.ocr = Some(Arc::new(FakeOcr));
    let processed = still_processed(test_png(), ImageFormat::Png);
    let deadline = Instant::now() + std::time::Duration::from_secs(60);
    let held = state.ai_gate.clone().acquire_owned().await.unwrap();
    let (tags, boxes, moderation, embedding, ocr) = tokio::join!(
        annotate_meta(&state, &processed),
        detect_meta(&state, &processed),
        moderate_meta(&state, &processed),
        embed_meta(&state, &processed),
        ocr_meta(&state, &processed, deadline),
    );
    assert!(tags.is_none(), "tags ran without a permit");
    assert!(boxes.is_none(), "detections ran without a permit");
    assert_eq!(moderation, ModerationMeta::Unavailable("busy"));
    assert_eq!(embedding, EmbeddingMeta::Unavailable("busy"));
    assert_eq!(ocr, OcrMeta::Unavailable("busy"));
    drop(held);
    assert!(annotate_meta(&state, &processed).await.is_some());
    assert!(detect_meta(&state, &processed).await.is_some());
    assert!(matches!(
        moderate_meta(&state, &processed).await,
        ModerationMeta::Report(_)
    ));
    assert!(matches!(
        embed_meta(&state, &processed).await,
        EmbeddingMeta::Report(_)
    ));
    assert!(matches!(
        ocr_meta(&state, &processed, deadline).await,
        OcrMeta::Report(_)
    ));
    assert_eq!(state.ai_gate.available_permits(), 1);
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(slow_permits)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        watermark_sources: Default::default(),
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
    std::fs::create_dir_all(dir).unwrap();
    let script_path = dir.join("encoder.sh");
    crate::test_support::write_script(
        &script_path,
        &format!("#!/bin/sh\ncat > /dev/null\nsleep {sleep_secs}\nprintf 'x'\n"),
    );
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
    let slow_with = |bytes: &[u8], ops_str: &str, preferred: Option<ImageFormat>| {
        let d = directives(ops_str);
        wants_slow_lane(
            wants_animation_gate(bytes, &d, preferred),
            d.explicit_format.is_some_and(|f| f.is_video()),
            d.frame.is_some(),
            d.page.is_some(),
        )
    };
    let slow = |bytes: &[u8], ops_str: &str| slow_with(bytes, ops_str, None);
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
    // Issue #405: an animated GIF rendered to WebP runs
    // the animated WebP encoder, so it is slow whether
    // WebP was named in the URL or negotiated.
    assert!(slow(&anim_gif, "filters:format(webp)/example.com/i.gif"));
    assert!(slow_with(
        &anim_gif,
        "example.com/i.gif",
        Some(ImageFormat::WebP)
    ));
    assert!(!slow_with(
        &anim_gif,
        "example.com/i.gif",
        Some(ImageFormat::Avif)
    ));
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
    let dir = crate::test_support::scratch_dir("server-slow-lane");
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
    let processed = follow
        .unwrap_or_else(|e| panic!("the still must render under slow-lane saturation, got {e:?}"));
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate,
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
        provenance_signer: None,
        config_fingerprint: "test".to_string(),
    };
    let job = ProcessJob {
        save_data: false,
        preserve_metadata: false,
        brand_color: None,
        ops: Vec::new(),
        preferred: None,
        watermark_sources: Arc::new(WatermarkSources {
            images: vec![(
                "http://img/wm.png".to_string(),
                b"this is not an image".to_vec(),
            )],
            share: None,
        }),
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
    // Issue #231: only an animated source rendered to
    // WebP output reaches the slow animated WebP encode,
    // so only that combination takes the gate. Issue
    // #405: since #370 that includes an animated GIF
    // source whose output resolves to WebP, through an
    // explicit format(webp) or a negotiated Accept.
    let anim = animated_webp_bytes();
    let anim_gif = animated_gif_bytes();
    let directives = |ops_str: &str| {
        let (ops, _) = crate::url::parse_operations(ops_str).unwrap();
        crate::url::RequestDirectives::analyze(&ops)
    };
    let gated = |bytes: &[u8], ops_str: &str, preferred: Option<ImageFormat>| {
        wants_animation_gate(bytes, &directives(ops_str), preferred)
    };
    assert!(gated(&anim, "example.com/i.webp", None));
    assert!(gated(
        &anim,
        "filters:format(webp)/example.com/i.webp",
        None
    ));
    // A negotiated AVIF or JXL cannot carry the frames:
    // the animation falls back to WebP and is encoded
    // as one, gate and all.
    for preferred in [ImageFormat::WebP, ImageFormat::Avif, ImageFormat::Jxl] {
        assert!(gated(&anim, "example.com/i.webp", Some(preferred)));
    }
    // An animated GIF to WebP, explicit or negotiated,
    // runs the same encoder and takes the same gate.
    assert!(gated(
        &anim_gif,
        "filters:format(webp)/example.com/i.gif",
        None
    ));
    assert!(gated(
        &anim_gif,
        "example.com/i.gif",
        Some(ImageFormat::WebP)
    ));
    assert!(gated(
        &anim_gif,
        "filters:format(webp)/example.com/i.gif",
        Some(ImageFormat::Avif)
    ));
    // An animated GIF that stays GIF (no preference, or
    // a negotiated AVIF or JXL, which fall back to GIF)
    // uses the chunked GIF encoder, ungated.
    for preferred in [None, Some(ImageFormat::Avif), Some(ImageFormat::Jxl)] {
        assert!(
            !gated(&anim_gif, "example.com/i.gif", preferred),
            "{preferred:?} must not gate an animated GIF"
        );
    }
    // Explicit non-WebP outputs never reach the
    // animated WebP encoder, whatever was negotiated:
    // GIF output uses the GIF encoder, still formats
    // take frame 0, and video takes the video gate
    // instead.
    for fmt in ["gif", "png", "jpeg", "mp4"] {
        for (src, name) in [(&anim, "i.webp"), (&anim_gif, "i.gif")] {
            assert!(
                !gated(
                    src,
                    &format!("filters:format({fmt})/example.com/{name}"),
                    Some(ImageFormat::WebP)
                ),
                "format({fmt}) on {name} must not take the animation gate"
            );
        }
    }
    // Still sources are never gated, whatever their
    // container or output: a still WebP carries no ANIM
    // chunk, a one-frame GIF has no second frame, and a
    // PNG is neither.
    for preferred in [None, Some(ImageFormat::WebP)] {
        assert!(!gated(&still_webp_bytes(), "example.com/i.webp", preferred));
        assert!(!gated(&still_gif_bytes(), "example.com/i.gif", preferred));
        assert!(!gated(
            &still_gif_bytes(),
            "filters:format(webp)/example.com/i.gif",
            preferred
        ));
        assert!(!gated(&test_png(), "example.com/i.png", preferred));
    }
}

#[test]
fn animation_gate_predicate_agrees_with_the_engine() {
    // Issue #405: the predicate drifted from the
    // engine's routing once already, so this runs the
    // real engine over every source, explicit format,
    // and negotiated preference the predicate decides
    // between, and requires the gate exactly when the
    // render comes out as an animated WebP.
    let sources: [(&str, Vec<u8>); 5] = [
        ("animated gif", animated_gif_bytes()),
        ("animated webp", animated_webp_bytes()),
        ("still gif", still_gif_bytes()),
        ("still webp", still_webp_bytes()),
        ("png", test_png()),
    ];
    let formats = [
        "",
        "filters:format(webp)/",
        "filters:format(gif)/",
        "filters:format(png)/",
    ];
    let preferences = [None, Some(ImageFormat::WebP), Some(ImageFormat::Jxl)];
    for (name, src) in &sources {
        for prefix in formats {
            for preferred in preferences {
                let (ops, _) =
                    crate::url::parse_operations(&format!("{prefix}example.com/i")).unwrap();
                let directives = crate::url::RequestDirectives::analyze(&ops);
                let out = Engine
                    .process_preferred(
                        src,
                        &ops,
                        crate::engine::ProcessContext {
                            preferred,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(
                    wants_animation_gate(src, &directives, preferred),
                    crate::engine::is_animated_webp(&out.bytes),
                    "{name} with {prefix:?} and {preferred:?}"
                );
            }
        }
    }
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
async fn animation_gate_sheds_gif_to_webp_renders() {
    // Issue #405: an animated GIF rendered to WebP runs
    // the animated WebP encoder, so with the single
    // animation permit held it sheds with the
    // animation-queue 503 like an animated WebP source
    // does, whether the WebP came from an unsigned
    // Accept header or an explicit format(webp). Before
    // the fix both ran at once on the stills lane.
    let gate = Arc::new(tokio::sync::Semaphore::new(1));
    let base = spawn_app_with_anim_gate(gate.clone()).await;
    let src = spawn_bytes_server(animated_gif_bytes()).await;
    let _held = gate.acquire_owned().await.unwrap();
    let sig = sign(&src);
    let explicit = format!("filters:format(webp)/{src}");
    let explicit_sig = sign(&explicit);
    let started = Instant::now();
    let (negotiated, explicit) = tokio::join!(
        client()
            .get(format!("{base}/{sig}/{src}"))
            .header("accept", "image/webp,*/*")
            .send(),
        client()
            .get(format!("{base}/{explicit_sig}/{explicit}"))
            .send(),
    );
    for (label, resp) in [("negotiated", negotiated), ("explicit", explicit)] {
        let resp = resp.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE, "{label}");
        assert_eq!(resp.headers().get("retry-after").unwrap(), "5", "{label}");
        let body = resp.text().await.unwrap();
        assert!(
            body.contains("animation encoding queue is full"),
            "{label}: {body}"
        );
    }
    assert!(
        started.elapsed() >= ANIM_GATE_WAIT,
        "the shed must come only after the gate wait elapses"
    );
    // The same animated GIF kept as GIF uses the GIF
    // encoder and never waits on the animation gate.
    let started = Instant::now();
    let resp = client()
        .get(format!("{base}/{sig}/{src}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers().get("content-type").unwrap(), "image/gif");
    assert!(
        started.elapsed() < ANIM_GATE_WAIT,
        "GIF output must not wait on the animation gate"
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

/// The status and, on a 200, the decoded dimensions of `resp`.
async fn status_and_dims(resp: reqwest::Response) -> (StatusCode, Option<(u32, u32)>) {
    let status = resp.status();
    let bytes = resp.bytes().await.unwrap();
    let dims = (status == StatusCode::OK).then(|| {
        let img = image::load_from_memory(&bytes).unwrap();
        (img.width(), img.height())
    });
    (status, dims)
}

/// A DPR hint is advisory (issue #529): when the hint alone
/// pushes a size past the per-axis limit or the pixel budget,
/// the request answers 200 at the largest factor that fits,
/// where it used to answer 400 on high-density screens only.
#[tokio::test]
async fn a_dpr_hint_past_the_limits_steps_down() {
    let src = spawn_image_server().await;
    let wide = spawn_bytes_server(png_of(16383, 1)).await;
    let base = spawn_app(false).await;
    for (ops, source, hint, dims) in [
        // 18000x3 at 3 is past the per-axis limit; 2 fits.
        ("6000x1", &src, ("DPR", "3"), (12000, 2)),
        // 16384x2 at 2 is past it; 1.5 fits.
        ("8192x1", &src, ("Sec-CH-DPR", "2"), (12288, 2)),
        // 32766x2 at 2 and at 1.5; 1 keeps the source size.
        ("0x1", &wide, ("DPR", "2"), (16383, 1)),
        // The budget: a 12000x12000 canvas at 1.5 is 144
        // million pixels, past the 100 million budget; at 1
        // the 8000x8000 canvas fits, and the crop-rect after
        // it keeps the response small.
        (
            "pad:8000x8000/crop-rect:0x0:8000x1/filters:no_upscale:format(png)",
            &src,
            ("DPR", "1.5"),
            (8000, 1),
        ),
    ] {
        let remainder = format!("{ops}/{source}");
        let resp = get_with_headers(&base, &remainder, &[hint]).await;
        assert_eq!(
            status_and_dims(resp).await,
            (StatusCode::OK, Some(dims)),
            "{ops} with {hint:?}"
        );
    }
    // A stepped factor never serves a side past the limit. At
    // 2, `8192x1` on a 16384x2 source asks for the source's own
    // size, which no step refuses: served, it was one past the
    // limit, and a WebP negotiated from Accept failed to encode
    // it with a 500. The factor steps to 1.5 in every format.
    let ceiling = spawn_bytes_server(png_of(16384, 2)).await;
    let remainder = format!("8192x1/{ceiling}");
    for (accept, format) in [("image/png", "image/png"), ("image/webp,*/*", "image/webp")] {
        let resp = get_with_headers(&base, &remainder, &[("DPR", "3"), ("Accept", accept)]).await;
        assert_eq!(header(&resp, "content-type"), format);
        assert_eq!(
            status_and_dims(resp).await,
            (StatusCode::OK, Some((12288, 2))),
            "{accept}"
        );
    }
}

/// A stepped-down render keeps the cache entry of the hint
/// that asked for it (issue #529): repeats of that hint hit
/// it, and neither the hint-less URL nor another factor is
/// ever answered from it. The response varies on both DPR
/// headers.
#[tokio::test]
async fn a_stepped_down_hint_render_keys_under_its_hint() {
    let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
    let base = spawn_app(false).await;
    let remainder = format!("6000x1/{src}");
    let resp = get_with_headers(&base, &remainder, &[("DPR", "3")]).await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    let vary = header(&resp, "vary");
    assert!(
        vary.split(", ").any(|h| h == "DPR") && vary.split(", ").any(|h| h == "Sec-CH-DPR"),
        "{vary}"
    );
    assert_eq!(
        status_and_dims(resp).await,
        (StatusCode::OK, Some((12000, 2)))
    );
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    // The same factor from either header is the same entry.
    for hints in [&[("DPR", "3")][..], &[("Sec-CH-DPR", "3")][..]] {
        let resp = get_with_headers(&base, &remainder, hints).await;
        assert_eq!(header(&resp, "cache-status"), "crabcrop; hit", "{hints:?}");
        assert_eq!(
            status_and_dims(resp).await,
            (StatusCode::OK, Some((12000, 2)))
        );
    }
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    // The hint-less URL renders 6000x1 under its own key.
    let resp = get_with_headers(&base, &remainder, &[]).await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_eq!(
        status_and_dims(resp).await,
        (StatusCode::OK, Some((6000, 1)))
    );
    // A hint of 2 renders the same size, under its own key.
    let resp = get_with_headers(&base, &remainder, &[("DPR", "2")]).await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_eq!(
        status_and_dims(resp).await,
        (StatusCode::OK, Some((12000, 2)))
    );
    assert_eq!(hits.load(Ordering::SeqCst), 3);
    // And the hint-less entry still answers 6000x1.
    let resp = get_with_headers(&base, &remainder, &[]).await;
    assert_eq!(header(&resp, "cache-status"), "crabcrop; hit");
    assert_eq!(
        status_and_dims(resp).await,
        (StatusCode::OK, Some((6000, 1)))
    );
}

/// Only the hint is advisory (issue #529): `dpr(...)` in the
/// URL or in the operator's default operations is the
/// request's own factor and keeps its 400, and a URL past the
/// limit at factor 1 keeps its 400 under any hint.
#[tokio::test]
async fn an_explicit_dpr_past_the_limit_still_answers_400() {
    let src = spawn_image_server().await;
    let base = spawn_app(false).await;
    let refused =
        |size: &str| format!("requested output {size} exceeds the 16383 pixel limit per axis");
    let expect_400 = |resp: reqwest::Response, size: &'static str| {
        let message = refused(size);
        async move {
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            assert_eq!(resp.text().await.unwrap(), message);
        }
    };
    let remainder = format!("6000x1/filters:dpr(3)/{src}");
    for hints in [&[][..], &[("DPR", "3")][..], &[("Sec-CH-DPR", "1")][..]] {
        expect_400(get_with_headers(&base, &remainder, hints).await, "18000x3").await;
    }
    let remainder = format!("16383x1/0x2/{src}");
    expect_400(
        get_with_headers(&base, &remainder, &[("DPR", "3")]).await,
        "32766x2",
    )
    .await;
    // A default `dpr(3)` is the operator's explicit factor.
    let base = spawn_app_with_defaults("filters:dpr(3)", false).await;
    let remainder = format!("6000x1/{src}");
    expect_400(
        get_with_headers(&base, &remainder, &[("DPR", "2")]).await,
        "18000x3",
    )
    .await;
}

/// Hostile DPR headers (issue #529): a value outside the
/// factor set, or not a number at all, is no hint and the URL
/// renders as written; a value in the set that the limits
/// refuse steps down. Every one answers 200.
#[tokio::test]
async fn hostile_dpr_hints_render_or_step_down() {
    let src = spawn_image_server().await;
    let base = spawn_app(false).await;
    let remainder = format!("6000x1/{src}");
    for (value, dims) in [
        ("NaN", (6000, 1)),
        ("inf", (6000, 1)),
        ("-3", (6000, 1)),
        ("0", (6000, 1)),
        ("1e30", (6000, 1)),
        ("2.5", (6000, 1)),
        ("3,2", (6000, 1)),
        ("", (6000, 1)),
        ("3e0", (12000, 2)),
        (" 3 ", (12000, 2)),
    ] {
        let resp = get_with_headers(&base, &remainder, &[("DPR", value)]).await;
        assert_eq!(
            status_and_dims(resp).await,
            (StatusCode::OK, Some(dims)),
            "DPR: {value:?}"
        );
    }
    let resp = client()
        .get(format!("{base}/{}/{remainder}", sign(&remainder)))
        .header(
            "Sec-CH-DPR",
            reqwest::header::HeaderValue::from_bytes(b"3\xff").unwrap(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(
        status_and_dims(resp).await,
        (StatusCode::OK, Some((6000, 1)))
    );
}

/// `/explain` reads the DPR hint as the image path does and
/// forecasts the factor the image request applies (issue
/// #529), stepped down like the render.
#[tokio::test]
async fn explain_forecasts_the_stepped_down_hint() {
    let src = spawn_image_server().await;
    let base = spawn_app(true).await;
    let explain = |ops: &'static str, hint: Option<&'static str>| {
        let (base, src) = (base.clone(), src.clone());
        async move {
            let mut req = client()
                .get(format!("{base}/explain/unsafe/{ops}/{src}"))
                .header("accept", "application/json");
            if let Some(hint) = hint {
                req = req.header("DPR", hint);
            }
            let resp = req.send().await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "{ops}");
            let report: serde_json::Value =
                serde_json::from_str(&resp.text().await.unwrap()).unwrap();
            report["forecast"].clone()
        }
    };
    let forecast = explain("6000x1", Some("3")).await;
    assert_eq!(forecast["width"], 12000);
    assert_eq!(forecast["height"], 2);
    assert_eq!(forecast["over_dimension"], false);
    assert_eq!(forecast["dpr"], 2.0);
    assert_eq!(forecast["dpr_from"], "hint");
    assert_eq!(forecast["dpr_hint"], 3.0);
    // The image request agrees.
    let resp = client()
        .get(format!("{base}/unsafe/6000x1/{src}"))
        .header("DPR", "3")
        .send()
        .await
        .unwrap();
    assert_eq!(
        status_and_dims(resp).await,
        (StatusCode::OK, Some((12000, 2)))
    );
    // Without the hint, factor 1.
    let forecast = explain("6000x1", None).await;
    assert_eq!(forecast["width"], 6000);
    assert_eq!(forecast["height"], 1);
    assert_eq!(forecast["dpr"], 1.0);
    assert_eq!(forecast["dpr_from"], "none");
    assert!(forecast["dpr_hint"].is_null());
    // An explicit factor does not step: the forecast flags
    // the refusal the image request answers.
    let forecast = explain("6000x1/filters:dpr(3)", Some("2")).await;
    assert_eq!(forecast["over_dimension"], true);
    assert_eq!(forecast["width"], 18000);
    assert_eq!(forecast["height"], 3);
    assert_eq!(forecast["dpr"], 3.0);
    assert_eq!(forecast["dpr_from"], "operation");
    assert_eq!(forecast["dpr_hint"], 2.0);
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
        spawn_bytes_server(include_bytes!("../../tests/fixtures/orient6.heic").to_vec()).await;
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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

fn upload_dir(tag: &str) -> crate::test_support::ScratchDir {
    crate::test_support::scratch_dir(&format!("srv-upload-{tag}"))
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
    // The body is the same reference plus a newline, as
    // docs/endpoints.md tells clients to expect.
    let body = resp.text().await.unwrap();
    assert_eq!(body, "file:///heroes/cat.png\n");
    let source_ref = body.trim_end();
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
async fn spawn_slow_counting_server(body: Vec<u8>, delay_ms: u64) -> (String, Arc<AtomicUsize>) {
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

/// Source server that answers slowly, with `body` as a
/// PNG or, for `None`, with a 500, so a burst's flights
/// overlap its requests. Returns its URL, its hit
/// count, and the most fetches it ever had open at once
/// (issue #407: a burst on one key must not fetch in
/// parallel).
async fn spawn_slow_peak_server(
    body: Option<Vec<u8>>,
    delay_ms: u64,
) -> (String, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let hits = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let open = Arc::new(AtomicUsize::new(0));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let counted = hits.clone();
    let peaked = peak.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            counted.fetch_add(1, Ordering::SeqCst);
            peaked.fetch_max(open.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
            let open = open.clone();
            let body = body.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 2048];
                let _ = socket.read(&mut buf).await;
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                match &body {
                    Some(body) => {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = socket.write_all(head.as_bytes()).await;
                        let _ = socket.write_all(body).await;
                    }
                    None => {
                        let _ = socket
                            .write_all(
                                b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                            )
                            .await;
                    }
                }
                open.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    (format!("http://{addr}/src.png"), hits, peak)
}

/// Persistent source server that answers slowly with
/// whatever body is current when the request arrives,
/// counting its hits: a source replaced while a render
/// of its old bytes is still in flight.
async fn spawn_slow_swappable_server(
    body: Vec<u8>,
    delay_ms: u64,
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
            // Taken before the hit counts, so a caller
            // that sees the hit can swap the body
            // without racing this request.
            let body = served.lock().unwrap().clone();
            counted.fetch_add(1, Ordering::SeqCst);
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
    (format!("http://{addr}/src.png"), hits, current)
}

/// Fire one GET per `(url, if_none_match)` pair at once
/// and collect each answer's status, ETag, and body, in
/// order.
async fn image_burst(
    requests: Vec<(String, Option<String>)>,
) -> Vec<(StatusCode, String, bytes::Bytes)> {
    image_burst_with(
        requests
            .into_iter()
            .map(|(url, inm)| (url, inm.map(|e| ("If-None-Match", e)).into_iter().collect()))
            .collect(),
    )
    .await
}

/// Fire one GET per `(url, headers)` pair at once and
/// collect each answer's status, ETag, and body, in
/// order.
async fn image_burst_with(
    requests: Vec<(String, Vec<(&'static str, String)>)>,
) -> Vec<(StatusCode, String, bytes::Bytes)> {
    let mut tasks = Vec::new();
    for (url, headers) in requests {
        tasks.push(tokio::spawn(async move {
            let mut req = client().get(&url);
            for (name, value) in headers {
                req = req.header(name, value);
            }
            let resp = req.send().await.unwrap();
            let status = resp.status();
            let etag = header(&resp, "etag");
            (status, etag, resp.bytes().await.unwrap())
        }));
    }
    let mut out = Vec::new();
    for task in tasks {
        out.push(task.await.unwrap());
    }
    out
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
    let removebg = crate::url::RequestDirectives::analyze(&[crate::url::Operation::Filters(vec![
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
async fn the_meta_disabled_refusal_is_never_publicly_cacheable() {
    // Issue #453: with meta mode off by default, any viewer
    // can make the server answer 403 for a URL by sending
    // the meta Accept. Some CDNs cache a 403 that carries
    // no caching headers (CloudFront's error caching keeps
    // it 10 seconds), and the response would then be served
    // to every image viewer. The refusal carries the meta
    // document's headers, plain and sealed alike, and the
    // sealed one keeps the uniform status and body.
    let src = spawn_bytes_server(test_png()).await;
    let base = spawn_app_with_meta_enabled(false).await;
    let remainder = format!("100x0/{src}");
    let plain = format!("{base}/{}/{remainder}", sign(&remainder));
    let sealed_url = format!("{base}/{}", sealed_path(&remainder, Some("photo.png")));
    for (url, sealed) in [(&plain, false), (&sealed_url, true)] {
        let resp = client()
            .get(url)
            .header("Accept", "image/avif, application/json; placeholders=1")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN, "{url}");
        assert_eq!(
            resp.headers()[header::CACHE_CONTROL],
            "private, no-store",
            "{url}"
        );
        let vary = resp.headers()[header::VARY].to_str().unwrap().to_string();
        assert!(vary.starts_with("Accept,"), "{url}: {vary}");
        let body = resp.text().await.unwrap();
        if sealed {
            assert_eq!(body, "bad signature");
        } else {
            assert!(body.contains(r#""code":"meta_disabled""#), "{body}");
        }
    }
    // The plain refusal tells the caller how meta turns on.
    let resp = client()
        .get(&plain)
        .header("Accept", "application/json")
        .send()
        .await
        .unwrap();
    let body = resp.text().await.unwrap();
    assert!(body.contains("CRABCROP_META_MODE=1"), "{body}");
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
    // Every request in a burst against a slow failing
    // origin must end the way a solo request does.
    // Issue #407: and the burst must not cost one
    // origin fetch per request. Before, three flights
    // failed one after another and then every waiter
    // past the wait cap fetched at once (12 requests
    // made 12 fetches, 9 of them in parallel); the
    // waiters now take the leader's answer.
    let (src, hits, peak) = spawn_slow_peak_server(None, 300).await;
    let base = spawn_app(false).await;
    let remainder = format!("50x30/{src}");
    let url = format!("{base}/{}/{remainder}", sign(&remainder));
    let solo = client().get(&url).send().await.unwrap();
    let solo_status = solo.status();
    let solo_body = solo.bytes().await.unwrap();
    assert_eq!(solo_status, StatusCode::BAD_GATEWAY);
    let before = hits.load(Ordering::SeqCst);
    let burst = image_burst(vec![(url.clone(), None); 12]).await;
    for (status, _, body) in &burst {
        assert_eq!(*status, solo_status);
        assert_eq!(*body, solo_body);
    }
    assert_eq!(
        peak.load(Ordering::SeqCst),
        1,
        "fetches of one key overlapped"
    );
    let fetched = hits.load(Ordering::SeqCst) - before;
    assert!(
        fetched <= 2,
        "12 identical requests made {fetched} origin fetches"
    );
}

#[tokio::test]
async fn flight_waiters_share_a_render_no_tier_stored() {
    // Issue #407: with every cache tier off, a leader's
    // render is stored nowhere, so its waiters used to
    // fetch and render again, one flight at a time and
    // then all at once past the wait cap. They now take
    // the leader's response, and each one still answers
    // its own conditional request against the shared
    // render's ETag.
    let (src, hits, peak) = spawn_slow_peak_server(Some(test_png()), 300).await;
    let base = spawn_app_full(std::collections::HashMap::new(), None, 0).await;
    let remainder = format!("50x30/{src}");
    let url = format!("{base}/{}/{remainder}", sign(&remainder));
    let solo = client().get(&url).send().await.unwrap();
    assert_eq!(solo.status(), StatusCode::OK);
    let etag = header(&solo, "etag");
    assert!(!etag.is_empty());
    let solo_body = solo.bytes().await.unwrap();
    let before = hits.load(Ordering::SeqCst);
    // Every other request revalidates the render it
    // already holds; the rest ask for the bytes.
    let requests = (0..8)
        .map(|i| (url.clone(), (i % 2 == 0).then(|| etag.clone())))
        .collect();
    let burst = image_burst(requests).await;
    for (i, (status, got_etag, body)) in burst.iter().enumerate() {
        assert_eq!(*got_etag, etag, "request {i}");
        if i % 2 == 0 {
            assert_eq!(*status, StatusCode::NOT_MODIFIED, "request {i}");
            assert!(body.is_empty(), "request {i}");
        } else {
            assert_eq!(*status, StatusCode::OK, "request {i}");
            assert_eq!(*body, solo_body, "request {i}");
        }
    }
    assert_eq!(
        peak.load(Ordering::SeqCst),
        1,
        "fetches of one key overlapped"
    );
    let fetched = hits.load(Ordering::SeqCst) - before;
    assert!(
        fetched <= 2,
        "8 identical requests made {fetched} origin fetches"
    );
}

#[tokio::test]
async fn flight_waiters_take_only_an_identical_requests_answer() {
    // Issue #407: one cache key serves spellings that
    // answer differently. A sealed request answers a
    // failure with the uniform refusal, so it must never
    // take the plain spelling's 502 from a flight they
    // share, nor the plain one the sealed refusal; each
    // spelling still ends exactly as it does alone.
    let (src, _hits, peak) = spawn_slow_peak_server(None, 300).await;
    let base = spawn_app(false).await;
    let remainder = format!("50x30/{src}");
    let plain = format!("{base}/{}/{remainder}", sign(&remainder));
    let sealed = format!("{base}/{}", sealed_path(&remainder, None));
    let solo = image_burst(vec![(plain.clone(), None), (sealed.clone(), None)]).await;
    assert_eq!(solo[0].0, StatusCode::BAD_GATEWAY);
    assert_eq!(solo[1].0, StatusCode::FORBIDDEN);
    assert_eq!(solo[1].2.as_ref(), b"bad signature");
    let requests = (0..8)
        .map(|i| (if i % 2 == 0 { &plain } else { &sealed }.clone(), None))
        .collect();
    let burst = image_burst(requests).await;
    for (i, (status, _, body)) in burst.iter().enumerate() {
        let expected = &solo[i % 2];
        assert_eq!((*status, body), (expected.0, &expected.2), "request {i}");
    }
    assert_eq!(
        peak.load(Ordering::SeqCst),
        1,
        "fetches of one key overlapped"
    );
}

#[tokio::test]
async fn flight_waiters_share_across_headers_that_do_not_change_the_answer() {
    // Issue #407: browsers phrase one request in
    // different headers. These Accept strings all
    // negotiate WebP, and the URL sizes the image
    // itself, so no width hint applies: every request
    // gets the same answer and must share one flight.
    // Compared as raw headers, each of the five
    // spellings needed a flight of its own, and the
    // waiters of the last ones passed the wait cap and
    // fetched at once.
    let (src, hits, peak) = spawn_slow_peak_server(None, 300).await;
    let base = spawn_app(false).await;
    let remainder = format!("50x30/{src}");
    let url = format!("{base}/{}/{remainder}", sign(&remainder));
    let spellings = [
        ("image/webp", "300"),
        ("image/webp,*/*", "310"),
        ("image/webp,image/*;q=0.8", "320"),
        ("image/png,image/webp", "330"),
        ("image/webp;q=0.9,*/*;q=0.1", "340"),
    ];
    let solo = image_burst_with(vec![(url.clone(), vec![("Accept", "image/webp".into())])]).await;
    assert_eq!(solo[0].0, StatusCode::BAD_GATEWAY);
    let before = hits.load(Ordering::SeqCst);
    let requests = (0..15)
        .map(|i| {
            let (accept, width) = spellings[i % spellings.len()];
            (
                url.clone(),
                vec![
                    ("Accept", accept.to_string()),
                    ("Sec-CH-Width", width.to_string()),
                ],
            )
        })
        .collect();
    let burst = image_burst_with(requests).await;
    for (i, (status, _, body)) in burst.iter().enumerate() {
        assert_eq!((*status, body), (solo[0].0, &solo[0].2), "request {i}");
    }
    assert_eq!(
        peak.load(Ordering::SeqCst),
        1,
        "fetches of one key overlapped"
    );
    let fetched = hits.load(Ordering::SeqCst) - before;
    assert!(
        fetched <= 2,
        "15 requests for one answer made {fetched} origin fetches"
    );
}

#[tokio::test]
async fn flight_waiters_do_not_take_a_render_a_purge_overtook() {
    // Issue #407: a follower takes the leader's unstored
    // response only while no purge of its key has
    // landed since the leader's run began. Here the
    // source is replaced and purged while the leader
    // still renders the old bytes. The purge keeps that
    // render out of the tiers (issue #350), and a
    // request sent after the purge must get the new
    // image, as the purge promises the next request,
    // not the old render through the flight it joined.
    let (src, hits, current) = spawn_slow_swappable_server(test_png(), 800).await;
    let base = spawn_app(false).await;
    let remainder = format!("25x25/{src}");
    let url = format!("{base}/{}/{remainder}", sign(&remainder));
    let leader_url = url.clone();
    let leader = tokio::spawn(async move { client().get(&leader_url).send().await.unwrap() });
    for _ in 0..50 {
        if hits.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(hits.load(Ordering::SeqCst), 1, "the render never started");
    *current.lock().unwrap() = test_png_red();
    let purge_sig = crate::url::purge_signature(SECRET, &remainder);
    let resp = client()
        .post(format!("{base}/purge/{purge_sig}/{remainder}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // The leader is still loading, so this request
    // joins its flight.
    let follower = client().get(&url).send().await.unwrap();
    let leader = leader.await.unwrap();
    assert_eq!(leader.status(), StatusCode::OK);
    assert_eq!(follower.status(), StatusCode::OK);
    assert_ne!(
        header(&follower, "etag"),
        header(&leader, "etag"),
        "the request after the purge got the render the purge overtook"
    );
    assert_eq!(hits.load(Ordering::SeqCst), 2);
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
        tokio::spawn(async move { meta_burst(&url, "application/json; placeholders=1", 4).await });
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
    // Issue #295: no waiter hangs or panics: when the
    // origin fails, every request in the burst ends
    // with the failure a solo request gets (since issue
    // #407, the leader's failure, shared through the
    // flight).
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
async fn concurrent_meta_requests_share_a_failure_instead_of_refetching() {
    // Issue #407: meta followers used to share only a
    // 200 document, so a failing leader sent them to
    // lead the next flights one by one and then, past
    // the wait cap, to fetch all at once. They now take
    // the leader's failure, which is the one they would
    // have computed themselves.
    let (src, hits, peak) = spawn_slow_peak_server(None, 300).await;
    let base = spawn_app(false).await;
    let remainder = format!("50x30/{src}");
    let url = format!("{base}/{}/{remainder}", sign(&remainder));
    let solo = meta_burst(&url, "application/json", 1).await;
    assert_eq!(solo[0].0, StatusCode::BAD_GATEWAY);
    let before = hits.load(Ordering::SeqCst);
    let burst = meta_burst(&url, "application/json", 12).await;
    assert!(
        burst
            .iter()
            .all(|(s, b)| *s == solo[0].0 && *b == solo[0].1),
        "every follower failed the way the solo request did"
    );
    assert_eq!(
        peak.load(Ordering::SeqCst),
        1,
        "fetches of one key overlapped"
    );
    let fetched = hits.load(Ordering::SeqCst) - before;
    assert!(
        fetched <= 2,
        "12 identical meta requests made {fetched} origin fetches"
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
    let dir = crate::test_support::scratch_dir("srv-gprop");
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
    let dir = crate::test_support::scratch_dir("srv-pdr");
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
    assert_eq!(
        entry_objects(&fake),
        0,
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
    let dir = crate::test_support::scratch_dir("srv-gpdr");
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

/// Poll `done` every 20 ms for up to five seconds, or
/// fail with `what`.
async fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    for _ in 0..250 {
        if done() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("timed out waiting until {what}");
}

/// A served app over disk and shared tiers for the
/// background purge walk tests (issue #400): one preset
/// render in group `small`, its bucket copy replaced by
/// distinct "stale" bytes, so a later answer shows which
/// copy served it, behind more filler objects than the
/// inline purge budget, so a group purge returns before
/// its walk reaches the entry.
struct WalkFixture {
    fake: crate::s3store::test_support::FakeS3,
    dir: crate::test_support::ScratchDir,
    hits: Arc<AtomicUsize>,
    probe: AppState,
    server: tokio::task::JoinHandle<()>,
    base: String,
    url: String,
    object: String,
    key: String,
    stale: CachedImage,
}

impl WalkFixture {
    async fn new(name: &str) -> Self {
        let fake = crate::s3store::test_support::start_fake().await;
        let dir = crate::test_support::scratch_dir(&format!("srv-{name}"));
        let (src, hits) = spawn_counting_server(200, "image/png", test_png()).await;
        let mut state = tier_test_state(Some(&dir), &fake.endpoint);
        state.presets = preset_test_map();
        let probe = state.clone();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });
        let preset = format!("preset:small/{src}");
        let url = format!("{base}/{}/{preset}", sign(&preset));
        // One render stores the entry, in group `small`,
        // in memory, on disk, and (written through in the
        // background) in the bucket.
        let resp = client().get(&url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        wait_until("the shared write-through lands", || {
            fake.objects.lock().unwrap().len() == 1
        })
        .await;
        let (object, key, rendered) = {
            let objects = fake.objects.lock().unwrap();
            let (object, data) = objects.iter().next().unwrap();
            let (key, image) =
                crate::storage::parse_entry(data, &crate::url::entry_key(b"disk-key-test"))
                    .unwrap();
            (object.clone(), key, image)
        };
        // Stand the bucket copy in for one the purge must
        // remove.
        let stale = CachedImage {
            bytes: bytes::Bytes::from_static(b"stale-pixels"),
            etag: crate::cache::etag_for_bytes(b"stale-pixels"),
            ..rendered
        };
        assert_eq!(stale.groups, vec!["small".to_string()]);
        probe.shared_store.clone().unwrap().put(&key, &stale).await;
        {
            let mut objects = fake.objects.lock().unwrap();
            for i in 0..SHARED_PURGE_INLINE_SCAN + 40 {
                let filler = format!("0-filler-{i:04}");
                assert!(filler < object, "fillers must list ahead of the entry");
                objects.insert(filler, b"not a cache entry".to_vec());
            }
        }
        WalkFixture {
            fake,
            dir,
            hits,
            probe,
            server,
            base,
            url,
            object,
            key,
            stale,
        }
    }

    /// Whether the bucket copy is the stale one.
    fn bucket_holds_stale(&self) -> bool {
        self.fake
            .objects
            .lock()
            .unwrap()
            .get(&self.object)
            .and_then(|data| {
                crate::storage::parse_entry(data, &crate::url::entry_key(b"disk-key-test"))
            })
            .is_some_and(|(_, image)| image.etag == self.stale.etag)
    }

    /// Put the stale copy back in the bucket, as the walk
    /// left it: unreached.
    async fn restore_stale_bucket_copy(&self) {
        let shared = self.probe.shared_store.clone().unwrap();
        shared.put(&self.key, &self.stale).await;
        assert!(self.bucket_holds_stale());
    }

    /// A replica over the fixture's bucket that did not
    /// receive the purge, with its own disk tier at `dir`
    /// when given.
    fn other_replica(&self, dir: Option<&std::path::Path>) -> AppState {
        let mut state = tier_test_state(dir, &self.fake.endpoint);
        state.presets = preset_test_map();
        state
    }

    /// Purge group `small` and wait until its background
    /// walk is parked on its first header read past the
    /// inline budget.
    async fn purge_and_park_walk(&self) {
        self.fake.ranged_gets.store(0, Ordering::SeqCst);
        *self.fake.hold_ranged_after.lock().unwrap() = Some(SHARED_PURGE_INLINE_SCAN);
        let sig = crate::url::purge_group_signature(SECRET, "small");
        let resp = client()
            .post(format!("{}/purge-group/{sig}/small", self.base))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Memory and disk; the bucket copy waits for the
        // walk.
        assert_eq!(resp.text().await.unwrap(), "purged 2\n");
        wait_until("the background walk parks", || {
            self.fake.ranged_gets.load(Ordering::SeqCst) > SHARED_PURGE_INLINE_SCAN
        })
        .await;
        assert_eq!(self.probe.purge_guard.active_count(), 1);
        assert!(self.fake.objects.lock().unwrap().contains_key(&self.object));
    }

    /// Let the parked walk finish and wait until the
    /// purge ends.
    async fn finish_walk(&self) {
        *self.fake.hold_ranged_after.lock().unwrap() = None;
        wait_until("the purge walk ends", || {
            self.probe.purge_guard.active_count() == 0
        })
        .await;
    }

    /// No tier holds the stale copy, and the URL serves a
    /// fresh render.
    async fn assert_no_tier_serves_the_stale_copy(&self) {
        let is_stale = |image: &CachedImage| image.etag == self.stale.etag;
        let store = self.probe.store.clone().unwrap();
        let shared = self.probe.shared_store.clone().unwrap();
        assert!(
            !self
                .probe
                .cache
                .get(&self.key)
                .is_some_and(|e| is_stale(&e))
        );
        assert!(!store.get(&self.key).is_some_and(|e| is_stale(&e)));
        assert!(!shared.get(&self.key).await.is_some_and(|e| is_stale(&e)));
        let resp = client().get(&self.url).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_ne!(header(&resp, "etag"), self.stale.etag);
        assert_ne!(resp.bytes().await.unwrap().as_ref(), b"stale-pixels");
    }
}

#[tokio::test]
async fn lookups_during_a_background_purge_walk_do_not_resurrect_the_entry() {
    // Issue #400: a shared purge walk that outgrows the
    // inline budget finishes in the background. Before
    // the fix, a GET during that walk missed memory and
    // disk (the purge had emptied them), served the
    // shared copy the walk had not reached yet, and
    // promoted it into both local tiers, where the walk
    // never looks: the purged entry kept serving after
    // the walk deleted the bucket object.
    let fx = WalkFixture::new("walk").await;
    fx.purge_and_park_walk().await;
    // A GET during the walk is a miss: it renders from
    // the origin instead of serving the bucket copy.
    let resp = client().get(&fx.url).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_ne!(header(&resp, "etag"), fx.stale.etag);
    assert_ne!(resp.bytes().await.unwrap().as_ref(), b"stale-pixels");
    assert_eq!(fx.hits.load(Ordering::SeqCst), 2);
    // The lookup deleted the copy on sight, while the
    // walk was still parked.
    assert!(
        fx.fake
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|r| r == &format!("DELETE {} if-match", fx.object)),
        "the lookup did not delete the purged bucket copy"
    );
    fx.finish_walk().await;
    fx.assert_no_tier_serves_the_stale_copy().await;
}

#[tokio::test]
async fn a_lookup_the_purge_walk_ends_under_does_not_resurrect_the_entry() {
    // Issue #400, at the end of the walk: a lookup that
    // starts while the walk runs reads the bucket copy
    // before the walk reaches it, and the walk deletes
    // the object and ends while those bytes are still in
    // flight. A check against only the purges still
    // running when the bytes arrive finds none, and the
    // lookup serves the copy and promotes it into both
    // local tiers after the purge is over.
    let fx = WalkFixture::new("walk-end").await;
    fx.purge_and_park_walk().await;
    fx.fake.hold_object_gets.store(true, Ordering::SeqCst);
    let url = fx.url.clone();
    let get = tokio::spawn(async move { client().get(&url).send().await.unwrap() });
    wait_until("the lookup reads the bucket copy", || {
        fx.fake.held_object_gets.load(Ordering::SeqCst) == 1
    })
    .await;
    fx.finish_walk().await;
    assert!(
        !fx.fake.objects.lock().unwrap().contains_key(&fx.object),
        "the walk did not delete the bucket copy"
    );
    fx.fake.hold_object_gets.store(false, Ordering::SeqCst);
    let resp = get.await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_ne!(header(&resp, "etag"), fx.stale.etag);
    assert_ne!(resp.bytes().await.unwrap().as_ref(), b"stale-pixels");
    assert_eq!(fx.hits.load(Ordering::SeqCst), 2);
    fx.assert_no_tier_serves_the_stale_copy().await;
}

#[tokio::test]
async fn copies_a_running_purge_covers_miss_in_every_persistent_tier() {
    // Issue #400, at the tier chain: while a purge is
    // still running, a disk or bucket copy it covers is
    // a miss for a lookup that starts after the purge,
    // and is deleted on sight; copies it does not cover
    // still serve, and once the purge ends a fresh copy
    // serves again.
    let fake = crate::s3store::test_support::start_fake().await;
    let dir = crate::test_support::scratch_dir("srv-active");
    let state = tier_test_state(Some(&dir), &fake.endpoint);
    let store = state.store.clone().unwrap();
    let shared = state.shared_store.clone().unwrap();
    let key = "40x24/example.com/i.png#cfgtest#w300";
    let other = "40x24/example.com/other.png#cfgtest";
    store.put(key, &tier_test_entry());
    shared.put(key, &tier_test_entry()).await;
    store.put(other, &tier_test_entry());
    let active = state.purge_guard.begin(crate::cache::PurgeTarget::Key(
        "40x24/example.com/i.png".into(),
    ));
    assert!(lookup_tiers(&state, key, &[]).await.is_none());
    assert!(store.get(key).is_none(), "the disk copy survived");
    assert!(shared.get(key).await.is_none(), "the bucket copy survived");
    assert!(state.cache.get(key).is_none(), "the copy was promoted");
    let (_, status) = lookup_tiers(&state, other, &[]).await.unwrap();
    assert_eq!(status, "crabcrop; hit; detail=\"disk\"");
    // A copy only in the bucket is caught there too.
    shared.put(key, &tier_test_entry()).await;
    assert!(lookup_tiers(&state, key, &[]).await.is_none());
    assert!(shared.get(key).await.is_none());
    // A copy rendered after the purge began is not
    // covered (issue #446): it serves while the purge runs.
    let fresh = CachedImage {
        written: Some(state.purge_guard.stamp()),
        ..tier_test_entry()
    };
    shared.put(key, &fresh).await;
    let (_, status) = lookup_tiers(&state, key, &[]).await.unwrap();
    assert_eq!(status, "crabcrop; hit; detail=\"shared\"");
    drop(active);
    // Once the purge has ended, its tombstone still covers
    // copies older than it for the grace period (issue
    // #461), but not fresh ones.
    state.cache.remove(key);
    store.remove(key);
    shared.put(key, &tier_test_entry()).await;
    assert!(lookup_tiers(&state, key, &[]).await.is_none());
    shared.put(key, &fresh).await;
    let (_, status) = lookup_tiers(&state, key, &[]).await.unwrap();
    assert_eq!(status, "crabcrop; hit; detail=\"shared\"");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_render_after_a_purge_began_keeps_caching_during_the_walk() {
    // Issue #446: while a purge's background walk runs,
    // the receiving server treated every disk or bucket
    // copy the purge covers as purged, including copies
    // rendered after the purge began: each lookup deleted
    // the fresh render and rendered again, and the walk
    // deleted (and counted) the render too. A copy written
    // after the purge began now keeps serving from every
    // persistent tier, and the walk leaves it in place.
    let fx = WalkFixture::new("walk-fresh").await;
    fx.purge_and_park_walk().await;
    let resp = client().get(&fx.url).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    let fresh = header(&resp, "etag");
    assert_ne!(fresh, fx.stale.etag);
    assert_eq!(fx.hits.load(Ordering::SeqCst), 2);
    let shared = fx.probe.shared_store.clone().unwrap();
    let store = fx.probe.store.clone().unwrap();
    // The render's write-through lands in the bucket.
    for _ in 0..250 {
        if shared.get(&fx.key).await.is_some_and(|e| e.etag == fresh) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(shared.get(&fx.key).await.is_some_and(|e| e.etag == fresh));
    let deletes = || {
        fx.fake
            .requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.starts_with(&format!("DELETE {}", fx.object)))
            .count()
    };
    let deletes_after_render = deletes();
    // With memory cleared, the next lookup hits the disk
    // tier, while the walk is still parked...
    fx.probe.cache.remove(&fx.key);
    let resp = client().get(&fx.url).send().await.unwrap();
    assert_eq!(
        header(&resp, "cache-status"),
        "crabcrop; hit; detail=\"disk\""
    );
    assert_eq!(header(&resp, "etag"), fresh);
    // ...and with the disk copy gone too, the bucket.
    fx.probe.cache.remove(&fx.key);
    store.remove(&fx.key);
    let resp = client().get(&fx.url).send().await.unwrap();
    assert_eq!(
        header(&resp, "cache-status"),
        "crabcrop; hit; detail=\"shared\""
    );
    assert_eq!(header(&resp, "etag"), fresh);
    assert_eq!(
        fx.hits.load(Ordering::SeqCst),
        2,
        "a fresh copy re-rendered"
    );
    assert_eq!(
        fx.probe.purge_guard.active_count(),
        1,
        "the walk ended early"
    );
    // The walk then reaches the object, reads a copy
    // written after the purge began, and leaves it: no
    // delete, so nothing counted.
    fx.finish_walk().await;
    assert_eq!(deletes(), deletes_after_render);
    assert!(shared.get(&fx.key).await.is_some_and(|e| e.etag == fresh));
    fx.probe.cache.remove(&fx.key);
    store.remove(&fx.key);
    let resp = client().get(&fx.url).send().await.unwrap();
    assert_eq!(
        header(&resp, "cache-status"),
        "crabcrop; hit; detail=\"shared\""
    );
    assert_eq!(fx.hits.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_restart_mid_walk_cannot_undo_the_purge_on_any_replica() {
    // Issue #461: the guard that kept a purge effective
    // during its walk lived only in memory, and so did the
    // walk's remainder. A restart mid-walk lost both: the
    // unreached bucket copy then served again on every
    // replica, the receiving one included. The purge's
    // tombstone in the bucket now keeps it effective.
    let fx = WalkFixture::new("walk-restart").await;
    // A replica already running, which never receives the
    // purge: it read the bucket's tombstones at boot.
    let running = fx.other_replica(None);
    start_purge_tombstones(&running).await;
    fx.purge_and_park_walk().await;
    // The receiving server stops mid-walk: its memory, its
    // guard, and the walk's remainder are gone. The bucket
    // still holds the unreached copy and the tombstone, not
    // marked ended.
    fx.server.abort();
    assert!(fx.bucket_holds_stale());
    let tombstone = {
        let objects = fx.fake.objects.lock().unwrap();
        let (_, data) = objects
            .iter()
            .find(|(name, _)| name.starts_with("purges/"))
            .expect("the purge left a tombstone in the bucket");
        crate::storage::parse_tombstone(data, &crate::url::entry_key(b"disk-key-test")).unwrap()
    };
    assert_eq!(
        tombstone.target,
        crate::cache::PurgeTarget::Group("small".into())
    );
    assert_eq!(tombstone.ended_ms, None);
    // The running replica learns the purge at its next
    // refresh and misses on the unreached copy.
    let shared = running.shared_store.clone().unwrap();
    assert!(shared.refresh_tombstones(&running.purge_guard).await);
    assert!(lookup_tiers(&running, &fx.key, &[]).await.is_none());
    assert!(
        !fx.bucket_holds_stale(),
        "the copy was not removed on sight"
    );
    // A replica that boots after the purge, with a disk of
    // its own that kept a pre-purge copy, misses on both.
    fx.restore_stale_bucket_copy().await;
    let other_dir = crate::test_support::scratch_dir("srv-walk-other");
    let booted = fx.other_replica(Some(&other_dir));
    let booted_disk = booted.store.clone().unwrap();
    booted_disk.put(&fx.key, &fx.stale);
    start_purge_tombstones(&booted).await;
    assert!(lookup_tiers(&booted, &fx.key, &[]).await.is_none());
    assert!(booted_disk.get(&fx.key).is_none(), "the disk copy survived");
    assert!(!fx.bucket_holds_stale(), "the bucket copy survived");
    // The restarted receiving server, over the same disk
    // and bucket with fresh memory, misses on the purged
    // URL and renders it again from the origin.
    fx.restore_stale_bucket_copy().await;
    let restarted = fx.other_replica(Some(&fx.dir));
    start_purge_tombstones(&restarted).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app(restarted)).await.unwrap();
    });
    let url = fx.url.replace(&fx.base, &base);
    let resp = client().get(&url).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    assert_ne!(header(&resp, "etag"), fx.stale.etag);
    assert_ne!(resp.bytes().await.unwrap().as_ref(), b"stale-pixels");
    assert_eq!(fx.hits.load(Ordering::SeqCst), 2);
    assert!(!fx.bucket_holds_stale());
}

#[tokio::test]
async fn a_replica_does_not_write_back_a_copy_older_than_a_purge_it_learned() {
    // Issue #461: a replica that held a copy from before
    // another replica's purge must not write it back to
    // the shared tier once it knows the purge, or the copy
    // would outlive the walk that removed it.
    let fake = crate::s3store::test_support::start_fake().await;
    let holder = tier_test_state(None, &fake.endpoint);
    let purger = tier_test_state(None, &fake.endpoint);
    let key = "40x24/example.com/i.png#cfgtest";
    holder.cache.put(key.to_string(), tier_test_entry());
    let (active, tombstone) = begin_purge(
        &purger,
        crate::cache::PurgeTarget::Key("40x24/example.com/i.png".into()),
    )
    .await;
    assert!(tombstone.is_some());
    let shared = holder.shared_store.clone().unwrap();
    assert!(shared.refresh_tombstones(&holder.purge_guard).await);
    // The memory hit still serves (another process's
    // memory is out of a purge's reach), but the group it
    // learns does not carry the copy back to the bucket.
    let (_, status) = lookup_tiers(&holder, key, &["thumb".to_string()])
        .await
        .unwrap();
    assert_eq!(status, "crabcrop; hit");
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert_eq!(
        entry_objects(&fake),
        0,
        "the pre-purge copy was written back"
    );
    drop(active);
}

/// A served app over a shared tier alone for the purge
/// walk pacing tests (issue #462), its bucket filled to
/// `objects` objects that are not cache entries: the first
/// [`SHARED_PURGE_INLINE_SCAN`], named `0-head-NNNN`, list
/// ahead of every entry object, so a purge's inline scan
/// reads exactly those; the rest, named `z-tail-NNNN`, list
/// after every entry object and are left to the background
/// walk, which a test parks on them with the fake's
/// `hold_prefix`.
async fn spawn_walk_app(
    fake: &crate::s3store::test_support::FakeS3,
    objects: usize,
) -> (String, AppState) {
    let state = tier_test_state(None, &fake.endpoint);
    let probe = state.clone();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app(state)).await.unwrap();
    });
    let mut bucket = fake.objects.lock().unwrap();
    let before = bucket.len();
    for i in 0..objects {
        let name = if i < SHARED_PURGE_INLINE_SCAN {
            format!("0-head-{i:04}")
        } else {
            format!("z-tail-{i:04}")
        };
        bucket.insert(name, b"not a cache entry".to_vec());
    }
    assert_eq!(bucket.len(), before + objects);
    drop(bucket);
    (base, probe)
}

/// Send a signed group purge and return its body.
async fn purge_group_call(base: &str, group: &str) -> String {
    let sig = crate::url::purge_group_signature(SECRET, group);
    let resp = client()
        .post(format!("{base}/purge-group/{sig}/{group}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    resp.text().await.unwrap()
}

/// Wait until every purge `state` started has ended, its
/// walks and re-lists included, or fail. Walks over a few
/// thousand objects can take seconds on a loaded machine,
/// so the limit is generous.
async fn wait_for_purges_to_end(state: &AppState) {
    for _ in 0..3000 {
        if state.purge_guard.active_count() == 0 {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("the purges never ended");
}

/// The purge tombstones in the fake bucket, parsed.
fn bucket_tombstones(fake: &crate::s3store::test_support::FakeS3) -> Vec<crate::cache::Tombstone> {
    let mut tombstones: Vec<crate::cache::Tombstone> = fake
        .objects
        .lock()
        .unwrap()
        .iter()
        .filter(|(name, _)| name.starts_with("purges/"))
        .map(|(_, data)| {
            crate::storage::parse_tombstone(data, &crate::url::entry_key(b"disk-key-test")).unwrap()
        })
        .collect();
    tombstones.sort_by_key(|tombstone| tombstone.cutoff.start_ms);
    tombstones
}

#[tokio::test]
async fn identical_group_purges_share_one_walk_and_one_relist() {
    // Issue #462: five purge-group calls in a row on a
    // 1,000-object bucket each listed the bucket and walked
    // it, all five walks at once. Now the four that arrive
    // while the first one's walk runs join it, and one
    // re-list after it serves all four. That re-list is
    // still needed with tombstones (#461): a copy rendered
    // between the first and the last call is newer than the
    // first purge, so the first walk spares it, and the
    // later purges' tombstones hide it only until they
    // expire; it has to leave the bucket.
    use crate::s3store::PURGE_READS_AT_ONCE;
    const OBJECTS: usize = 1000;
    let fake = crate::s3store::test_support::start_fake().await;
    let (base, probe) = spawn_walk_app(&fake, OBJECTS - 1).await;
    let shared = probe.shared_store.clone().unwrap();
    // One member of `small` from before every purge.
    let stale_key = "40x24/example.com/stale.png#cfgtest";
    let groups = vec!["small".to_string()];
    let member = CachedImage {
        groups: groups.clone(),
        ..tier_test_entry()
    };
    shared.put(stale_key, &member).await;
    *fake.walk.hold_prefix.lock().unwrap() = Some("z-tail-".into());
    assert_eq!(purge_group_call(&base, "small").await, "purged 0\n");
    wait_until("the first walk parks", || {
        fake.walk.held.load(Ordering::SeqCst) == PURGE_READS_AT_ONCE
    })
    .await;
    assert_eq!(fake.walk.lists_of(""), 1);
    // Another replica writes back a render that began after
    // the first purge: the first walk has already listed the
    // bucket, and would spare the copy anyway.
    let fresh_key = "40x24/example.com/fresh.png#cfgtest";
    let fresh = CachedImage {
        written: Some(probe.purge_guard.stamp()),
        ..member.clone()
    };
    shared.put(fresh_key, &fresh).await;
    // Four more identical purges join the parked walk: none
    // of them lists the bucket or reads a header.
    let reads = fake.ranged_gets.load(Ordering::SeqCst);
    for _ in 0..4 {
        assert_eq!(purge_group_call(&base, "small").await, "purged 0\n");
    }
    assert_eq!(fake.walk.lists_of(""), 1);
    assert_eq!(fake.ranged_gets.load(Ordering::SeqCst), reads);
    assert_eq!(probe.purge_guard.active_count(), 5);
    *fake.walk.hold_prefix.lock().unwrap() = None;
    wait_for_purges_to_end(&probe).await;
    // One walk and one re-list: each object's header read
    // once per pass, where five walks read it five times.
    assert_eq!(fake.walk.lists_of(""), 2);
    assert_eq!(fake.ranged_gets.load(Ordering::SeqCst), 2 * OBJECTS);
    assert!(shared.get(stale_key).await.is_none());
    // The copy written between the first and the last call
    // is gone from the bucket, where nothing else would have
    // removed it: once the grace has passed, no tombstone
    // covers it any more.
    assert!(shared.get(fresh_key).await.is_none());
    let past_grace = crate::cache::now_ms() + crate::cache::TOMBSTONE_GRACE_MS + 1;
    assert!(
        !probe
            .purge_guard
            .running_covers_at(past_grace, fresh_key, &groups, fresh.written)
    );
    // Every purge's bucket tombstone was marked ended.
    let tombstones = bucket_tombstones(&fake);
    assert_eq!(tombstones.len(), 5);
    assert!(tombstones.iter().all(|t| t.ended_ms.is_some()));
}

#[tokio::test]
async fn background_purge_walks_take_turns_two_at_a_time() {
    // Issue #462: every purge that outgrew its inline scan
    // walked the rest of the bucket on its own unbounded
    // task, so N different purges ran N walks at once. Now
    // background walks wait for one of PURGE_WALKS_AT_ONCE
    // turns: with every walk's reads parked, exactly two
    // walks' worth arrive, and the other three walks wait.
    use crate::s3store::{PURGE_READS_AT_ONCE, PURGE_WALKS_AT_ONCE};
    const OBJECTS: usize = SHARED_PURGE_INLINE_SCAN + 100;
    let fake = crate::s3store::test_support::start_fake().await;
    let (base, probe) = spawn_walk_app(&fake, OBJECTS).await;
    *fake.walk.hold_prefix.lock().unwrap() = Some("z-tail-".into());
    let calls: Vec<_> = (0..5)
        .map(|i| {
            let base = base.clone();
            tokio::spawn(async move { purge_group_call(&base, &format!("g{i}")).await })
        })
        .collect();
    for call in calls {
        assert_eq!(call.await.unwrap(), "purged 0\n");
    }
    let parked = PURGE_WALKS_AT_ONCE * PURGE_READS_AT_ONCE;
    wait_until("two walks park", || {
        fake.walk.held.load(Ordering::SeqCst) >= parked
    })
    .await;
    // The other walks wait for a turn: no read of theirs
    // arrives while the first two are parked.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert_eq!(fake.walk.held.load(Ordering::SeqCst), parked);
    assert_eq!(probe.purge_guard.active_count(), 5);
    *fake.walk.hold_prefix.lock().unwrap() = None;
    wait_for_purges_to_end(&probe).await;
    assert_eq!(fake.walk.held_peak.load(Ordering::SeqCst), parked);
    // Each purge still listed and walked the whole bucket.
    assert_eq!(fake.walk.lists_of(""), 5);
    assert_eq!(fake.ranged_gets.load(Ordering::SeqCst), 5 * OBJECTS);
}

#[tokio::test]
async fn an_identical_url_purge_joins_the_walk_and_purges_its_variants_inline() {
    // Issue #462: a per-URL purge whose target is already
    // being walked joins that walk, but the current epoch's
    // variants are reached by name, five reads, so it still
    // deletes them itself: a copy re-rendered after the
    // first purge is gone from the bucket when the second
    // purge answers, while the first walk is still parked.
    use crate::s3store::PURGE_READS_AT_ONCE;
    let fx = WalkFixture::new("walk-join-url").await;
    {
        let mut objects = fx.fake.objects.lock().unwrap();
        for i in 0..64 {
            objects.insert(format!("z-tail-{i:04}"), b"not a cache entry".to_vec());
        }
    }
    *fx.fake.walk.hold_prefix.lock().unwrap() = Some("z-tail-".into());
    let path = fx.url.strip_prefix(&fx.base).unwrap();
    let (_, remainder) = path[1..].split_once('/').unwrap();
    let purge_url = format!(
        "{}/purge/{}/{remainder}",
        fx.base,
        crate::url::purge_signature(SECRET, remainder)
    );
    let resp = client().post(&purge_url).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    wait_until("the first walk parks", || {
        fx.fake.walk.held.load(Ordering::SeqCst) == PURGE_READS_AT_ONCE
    })
    .await;
    assert!(!fx.bucket_holds_stale());
    // A render after the first purge began writes its copy
    // through to the bucket.
    let resp = client().get(&fx.url).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(header(&resp, "cache-status"), "crabcrop; fwd=uri-miss");
    wait_until("the re-render lands in the bucket", || {
        fx.fake.objects.lock().unwrap().contains_key(&fx.object)
    })
    .await;
    let lists = fx.fake.walk.lists_of("");
    let resp = client().post(&purge_url).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // Memory, disk, and the bucket copy, by name...
    assert_eq!(resp.text().await.unwrap(), "purged 3\n");
    assert!(!fx.fake.objects.lock().unwrap().contains_key(&fx.object));
    // ...with no walk of its own: it joined the parked one.
    assert_eq!(fx.fake.walk.lists_of(""), lists);
    assert_eq!(
        fx.fake.walk.held.load(Ordering::SeqCst),
        PURGE_READS_AT_ONCE
    );
    assert_eq!(fx.probe.purge_guard.active_count(), 2);
    *fx.fake.walk.hold_prefix.lock().unwrap() = None;
    wait_for_purges_to_end(&fx.probe).await;
    // One re-list served the joined purge.
    assert_eq!(fx.fake.walk.lists_of(""), lists + 1);
    fx.assert_no_tier_serves_the_stale_copy().await;
}

#[tokio::test]
async fn a_purge_joining_an_inline_walk_still_gets_its_relist() {
    // Issue #462 review: the leader's walk ends inline, with
    // no background remainder, and an identical purge joins
    // while that inline scan runs. The joined purge must
    // still get its re-list, or a copy rendered between the
    // two purges stays in the bucket and serves again once
    // the joined purge's tombstone expires.
    use crate::s3store::PURGE_READS_AT_ONCE;
    let fake = crate::s3store::test_support::start_fake().await;
    let (base, probe) = spawn_walk_app(&fake, 40).await;
    let shared = probe.shared_store.clone().unwrap();
    let groups = vec!["small".to_string()];
    let member = CachedImage {
        groups: groups.clone(),
        ..tier_test_entry()
    };
    shared
        .put("40x24/example.com/stale.png#cfgtest", &member)
        .await;
    *fake.walk.hold_prefix.lock().unwrap() = Some("0-head-".into());
    let leader_base = base.clone();
    let leader = tokio::spawn(async move { purge_group_call(&leader_base, "small").await });
    wait_until("the leader's inline scan parks", || {
        fake.walk.held.load(Ordering::SeqCst) == PURGE_READS_AT_ONCE
    })
    .await;
    let fresh_key = "40x24/example.com/fresh.png#cfgtest";
    let fresh = CachedImage {
        written: Some(probe.purge_guard.stamp()),
        ..member.clone()
    };
    shared.put(fresh_key, &fresh).await;
    assert_eq!(purge_group_call(&base, "small").await, "purged 0\n");
    assert_eq!(fake.walk.lists_of(""), 1);
    *fake.walk.hold_prefix.lock().unwrap() = None;
    assert_eq!(leader.await.unwrap(), "purged 1\n");
    wait_for_purges_to_end(&probe).await;
    assert_eq!(fake.walk.lists_of(""), 2);
    assert!(shared.get(fresh_key).await.is_none());
}

#[tokio::test]
async fn the_relist_judges_copies_by_the_latest_joined_purge() {
    // Issue #462 review: two purges join a parked walk, one
    // before and one after a copy is rendered. The shared
    // re-list must judge copies by the later purge's start,
    // or that copy stays in the bucket and serves again once
    // the later purge's tombstone expires.
    use crate::s3store::PURGE_READS_AT_ONCE;
    let fake = crate::s3store::test_support::start_fake().await;
    let (base, probe) = spawn_walk_app(&fake, SHARED_PURGE_INLINE_SCAN + 50).await;
    let shared = probe.shared_store.clone().unwrap();
    let member = CachedImage {
        groups: vec!["small".to_string()],
        ..tier_test_entry()
    };
    *fake.walk.hold_prefix.lock().unwrap() = Some("z-tail-".into());
    assert_eq!(purge_group_call(&base, "small").await, "purged 0\n");
    wait_until("the leader's walk parks", || {
        fake.walk.held.load(Ordering::SeqCst) == PURGE_READS_AT_ONCE
    })
    .await;
    assert_eq!(purge_group_call(&base, "small").await, "purged 0\n");
    let fresh_key = "40x24/example.com/fresh.png#cfgtest";
    let fresh = CachedImage {
        written: Some(probe.purge_guard.stamp()),
        ..member.clone()
    };
    shared.put(fresh_key, &fresh).await;
    assert_eq!(purge_group_call(&base, "small").await, "purged 0\n");
    *fake.walk.hold_prefix.lock().unwrap() = None;
    wait_for_purges_to_end(&probe).await;
    assert_eq!(fake.walk.lists_of(""), 2);
    assert!(shared.get(fresh_key).await.is_none());
}

#[tokio::test]
async fn inline_scans_of_different_purges_share_one_read_budget() {
    // Issue #462 review: walk turns bound the background
    // walks only, and each purge's inline scan read 16
    // headers at a time on its own, so six concurrent purges
    // of different groups put 96 reads on the store at once.
    // Every inline read now holds one of the process's
    // PURGE_INLINE_READS_AT_ONCE: with every inline read
    // parked, exactly that many arrive, and the rest wait.
    use crate::s3store::PURGE_INLINE_READS_AT_ONCE;
    let fake = crate::s3store::test_support::start_fake().await;
    let (base, probe) = spawn_walk_app(&fake, SHARED_PURGE_INLINE_SCAN).await;
    *fake.walk.hold_prefix.lock().unwrap() = Some("0-head-".into());
    let calls: Vec<_> = (0..6)
        .map(|i| {
            let base = base.clone();
            tokio::spawn(async move { purge_group_call(&base, &format!("g{i}")).await })
        })
        .collect();
    wait_until("the inline scans park their reads", || {
        fake.walk.held.load(Ordering::SeqCst) >= PURGE_INLINE_READS_AT_ONCE
    })
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert_eq!(
        fake.walk.held.load(Ordering::SeqCst),
        PURGE_INLINE_READS_AT_ONCE
    );
    assert_eq!(fake.walk.lists_of(""), 6);
    *fake.walk.hold_prefix.lock().unwrap() = None;
    for call in calls {
        assert_eq!(call.await.unwrap(), "purged 0\n");
    }
    assert_eq!(
        fake.walk.held_peak.load(Ordering::SeqCst),
        PURGE_INLINE_READS_AT_ONCE
    );
    assert_eq!(
        fake.ranged_gets.load(Ordering::SeqCst),
        6 * SHARED_PURGE_INLINE_SCAN
    );
    wait_for_purges_to_end(&probe).await;
}

#[tokio::test]
async fn a_purge_request_scan_does_not_wait_behind_background_walks() {
    // Issue #462 review: the inline read budget is the
    // requests' own. With both background turns taken by
    // walks whose reads are parked, a purge of another group
    // still scans its inline part and answers; only its own
    // background walk waits for a turn.
    use crate::s3store::{PURGE_READS_AT_ONCE, PURGE_WALKS_AT_ONCE};
    let fake = crate::s3store::test_support::start_fake().await;
    let (base, probe) = spawn_walk_app(&fake, SHARED_PURGE_INLINE_SCAN + 20).await;
    *fake.walk.hold_prefix.lock().unwrap() = Some("z-tail-".into());
    for i in 0..PURGE_WALKS_AT_ONCE {
        assert_eq!(
            purge_group_call(&base, &format!("g{i}")).await,
            "purged 0\n"
        );
    }
    let parked = PURGE_WALKS_AT_ONCE * PURGE_READS_AT_ONCE;
    wait_until("the background walks park", || {
        fake.walk.held.load(Ordering::SeqCst) == parked
    })
    .await;
    let reads = fake.ranged_gets.load(Ordering::SeqCst);
    // Well inside the store client's 10 s request timeout,
    // which would otherwise free a stuck budget by failing
    // the parked reads.
    let answer = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        purge_group_call(&base, "other"),
    )
    .await
    .expect("the purge request waited behind the parked walks");
    assert_eq!(answer, "purged 0\n");
    assert_eq!(
        fake.ranged_gets.load(Ordering::SeqCst),
        reads + SHARED_PURGE_INLINE_SCAN
    );
    assert_eq!(fake.walk.held.load(Ordering::SeqCst), parked);
    *fake.walk.hold_prefix.lock().unwrap() = None;
    wait_for_purges_to_end(&probe).await;
    assert_eq!(fake.walk.lists_of(""), PURGE_WALKS_AT_ONCE + 1);
}

/// Issue #462 review: purge a group while `fail` makes the
/// store fail part of the walk, and check that the purge
/// stays in force. The walk could not reach the group's
/// copy, so marking the purge ended would let that copy
/// serve again once the tombstone expires; it is left
/// running in the bucket instead, as when the process stops
/// mid-walk, until a later purge of the group completes.
async fn assert_a_failed_walk_leaves_its_purge_in_force(
    fail: fn(&mut crate::s3store::test_support::FailFlags, bool),
) {
    let fake = crate::s3store::test_support::start_fake().await;
    let (base, probe) = spawn_walk_app(&fake, 0).await;
    let shared = probe.shared_store.clone().unwrap();
    let stale_key = "40x24/example.com/stale.png#cfgtest";
    let groups = vec!["small".to_string()];
    let member = CachedImage {
        groups: groups.clone(),
        ..tier_test_entry()
    };
    shared.put(stale_key, &member).await;
    let errors = shared.errors();
    fail(&mut fake.fail.lock().unwrap(), true);
    assert_eq!(purge_group_call(&base, "small").await, "purged 0\n");
    wait_for_purges_to_end(&probe).await;
    fail(&mut fake.fail.lock().unwrap(), false);
    // Counted for /metrics, and the copy is still there.
    assert!(shared.errors() > errors);
    assert!(shared.get(stale_key).await.is_some());
    let tombstones = bucket_tombstones(&fake);
    assert_eq!(tombstones.len(), 1);
    assert_eq!(tombstones[0].ended_ms, None);
    // A replica that reads the tombstone keeps the copy a
    // miss past any grace period.
    assert!(shared.refresh_tombstones(&probe.purge_guard).await);
    let past_grace = crate::cache::now_ms() + crate::cache::TOMBSTONE_GRACE_MS + 1;
    assert!(
        probe
            .purge_guard
            .running_covers_at(past_grace, stale_key, &groups, None)
    );
    // A later purge whose walk completes removes the copy,
    // and its ended tombstone supersedes the one left open.
    assert_eq!(purge_group_call(&base, "small").await, "purged 1\n");
    wait_for_purges_to_end(&probe).await;
    assert!(shared.get(stale_key).await.is_none());
    assert!(shared.refresh_tombstones(&probe.purge_guard).await);
    let tombstones = bucket_tombstones(&fake);
    assert_eq!(tombstones.len(), 1);
    assert!(tombstones[0].ended_ms.is_some());
}

#[tokio::test]
async fn a_purge_whose_walk_cannot_list_the_bucket_stays_in_force() {
    assert_a_failed_walk_leaves_its_purge_in_force(|fail, on| fail.list = on).await;
}

#[tokio::test]
async fn a_purge_whose_header_reads_fail_stays_in_force() {
    assert_a_failed_walk_leaves_its_purge_in_force(|fail, on| fail.get_object = on).await;
}

#[tokio::test]
async fn a_purge_whose_deletes_fail_stays_in_force() {
    assert_a_failed_walk_leaves_its_purge_in_force(|fail, on| fail.delete = on).await;
}

#[tokio::test]
async fn a_failed_relist_leaves_the_joined_purges_in_force() {
    // Issue #462 review: one re-list serves every purge that
    // joined a walk. When the store cannot list the bucket
    // for it, none of those purges may be marked ended: the
    // copy rendered between the purges is still there.
    use crate::s3store::PURGE_READS_AT_ONCE;
    let fake = crate::s3store::test_support::start_fake().await;
    let (base, probe) = spawn_walk_app(&fake, SHARED_PURGE_INLINE_SCAN + 50).await;
    let shared = probe.shared_store.clone().unwrap();
    let groups = vec!["small".to_string()];
    *fake.walk.hold_prefix.lock().unwrap() = Some("z-tail-".into());
    assert_eq!(purge_group_call(&base, "small").await, "purged 0\n");
    wait_until("the first walk parks", || {
        fake.walk.held.load(Ordering::SeqCst) == PURGE_READS_AT_ONCE
    })
    .await;
    let fresh_key = "40x24/example.com/fresh.png#cfgtest";
    let fresh = CachedImage {
        groups: groups.clone(),
        written: Some(probe.purge_guard.stamp()),
        ..tier_test_entry()
    };
    shared.put(fresh_key, &fresh).await;
    assert_eq!(purge_group_call(&base, "small").await, "purged 0\n");
    assert_eq!(probe.purge_guard.active_count(), 2);
    fake.fail.lock().unwrap().list = true;
    *fake.walk.hold_prefix.lock().unwrap() = None;
    wait_for_purges_to_end(&probe).await;
    fake.fail.lock().unwrap().list = false;
    assert!(shared.get(fresh_key).await.is_some());
    // The first purge's own walk completed, so it ended; the
    // joined one did not.
    let tombstones = bucket_tombstones(&fake);
    assert_eq!(tombstones.len(), 2);
    assert!(tombstones[0].ended_ms.is_some());
    assert_eq!(tombstones[1].ended_ms, None);
    assert!(shared.refresh_tombstones(&probe.purge_guard).await);
    let past_grace = crate::cache::now_ms() + crate::cache::TOMBSTONE_GRACE_MS + 1;
    assert!(
        probe
            .purge_guard
            .running_covers_at(past_grace, fresh_key, &groups, fresh.written)
    );
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
            crate::storage::DiskStore::open(dir, 16 * 1024 * 1024, crate::url::entry_key(SECRET))
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
        // Its own stem: on a case-insensitive file system
        // (the macOS default) photo.PNG is the same file as
        // photo.png and would be refused as existing.
        ("upper.PNG", StatusCode::CREATED),
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
async fn undocumented_formats_fail_as_sources_and_uploads() {
    // Issue #466: the image crate's long-tail decoders are
    // not compiled in, and a HEIF file coded with anything
    // but HEVC is refused before decoding. As a source each
    // sample is a 502 decode_failed; as an upload it is a
    // 415 whatever name it claims, and nothing is stored.
    let mut samples = crate::engine::test_formats::undocumented_samples();
    samples.push(("av1-in-mif1", crate::engine::test_formats::av1_in_mif1()));
    let base = spawn_app(false).await;
    for (kind, bytes) in &samples {
        let src = spawn_bytes_server(bytes.clone()).await;
        let remainder = format!("100x50/{src}");
        let url = format!("{base}/{}/{remainder}", sign(&remainder));
        let resp = client()
            .get(&url)
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY, "{kind}");
        let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "decode_failed", "{kind}");
    }
    let dir = upload_dir("long-tail");
    let base = spawn_app_with_upload(&dir, 10 * 1024 * 1024, false).await;
    for (kind, bytes) in &samples {
        for name in [
            format!("{kind}/promo.html"),
            format!("{kind}/page.svg"),
            format!("{kind}/x.png"),
            format!("{kind}/plain"),
        ] {
            let sig = crate::url::upload_signature(SECRET, &name);
            let resp = client()
                .put(format!("{base}/upload/{sig}/{name}"))
                .body(bytes.clone())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE, "{name}");
            let text = resp.text().await.unwrap();
            assert!(
                text.contains("not an image this server can decode"),
                "{name}: {text}"
            );
            assert!(!dir.join(&name).exists(), "{name}");
        }
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
    let mut statuses: Vec<StatusCode> = [a, b].into_iter().map(|r| r.unwrap().status()).collect();
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
async fn uploads_answer_uploads_not_configured_without_a_destination() {
    // Issue #467: the plain test app has no upload
    // destination. The route still exists and says why
    // it refuses, 404 uploads_not_configured, before it
    // looks at the signature, so a bad one gets the same
    // answer. A PUT anywhere else stays a plain
    // not_found.
    let base = spawn_app(false).await;
    let name = "cat.png";
    let sig = crate::url::upload_signature(SECRET, name);
    for sig in [sig.as_str(), "0000", "deadbeef"] {
        let url = format!("{base}/upload/{sig}/{name}");
        let plain = client().put(&url).body(test_png()).send().await.unwrap();
        let json = client()
            .put(&url)
            .header("accept", "application/json")
            .body(test_png())
            .send()
            .await
            .unwrap();
        assert_error_modes(
            plain,
            json,
            StatusCode::NOT_FOUND,
            "uploads_not_configured",
            "uploads are not configured",
        )
        .await;
    }
    let resp = client()
        .put(format!("{base}/elsewhere/{name}"))
        .header("accept", "application/json")
        .body(test_png())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "not_found");
}

#[tokio::test]
async fn an_upload_body_read_error_is_a_400_not_a_413() {
    // Issue #467: only the cap is a size answer. These
    // are signed, valid-name uploads well under the cap
    // whose bodies cannot be read to their end: hostile
    // chunked framings (a chunk size that is not hex, one
    // that overflows, a chunk longer than it declared) and
    // an aborted body. Each answered 413 upload_too_large
    // before; each answers 400 upload_unreadable, stores
    // nothing, and leaves the server serving.
    let dir = upload_dir("unreadable");
    let base = spawn_app_with_upload(&dir, 1024 * 1024, false).await;
    let addr = base.trim_start_matches("http://").to_string();
    let name = "torn.png";
    let sig = crate::url::upload_signature(SECRET, name);
    let framings: [&[u8]; 3] = [
        b"zz\r\nabc\r\n0\r\n\r\n",
        b"ffffffffffffffffffffffff\r\nabc\r\n0\r\n\r\n",
        b"3\r\nabcdef\r\n0\r\n\r\n",
    ];
    for framing in framings {
        let mut socket = tokio::net::TcpStream::connect(&addr).await.unwrap();
        let head = format!(
            "PUT /upload/{sig}/{name} HTTP/1.1\r\nHost: {addr}\r\n\
             Accept: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
        );
        socket.write_all(head.as_bytes()).await.unwrap();
        socket.write_all(framing).await.unwrap();
        let mut reply = Vec::new();
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            socket.read_to_end(&mut reply),
        )
        .await;
        let reply = String::from_utf8_lossy(&reply);
        let shown = String::from_utf8_lossy(framing);
        assert!(reply.starts_with("HTTP/1.1 400 "), "{shown:?}: {reply}");
        assert!(
            reply.contains("\"upload_unreadable\""),
            "{shown:?}: {reply}"
        );
        assert!(!dir.join(name).exists(), "{shown:?}");
    }
    // An aborted body: a client that declares a length,
    // sends part of it, and shuts its half of the
    // connection. Same answer, nothing stored.
    let mut socket = tokio::net::TcpStream::connect(&addr).await.unwrap();
    let head = format!(
        "PUT /upload/{sig}/{name} HTTP/1.1\r\nHost: {addr}\r\n\
         Accept: application/json\r\nContent-Length: 4096\r\n\r\n"
    );
    socket.write_all(head.as_bytes()).await.unwrap();
    socket.write_all(&test_png()[..32]).await.unwrap();
    socket.shutdown().await.unwrap();
    let mut reply = Vec::new();
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        socket.read_to_end(&mut reply),
    )
    .await;
    let reply = String::from_utf8_lossy(&reply);
    assert!(reply.starts_with("HTTP/1.1 400 "), "{reply}");
    assert!(reply.contains("\"upload_unreadable\""), "{reply}");
    assert!(!dir.join(name).exists());
    // The server is still serving, and the same name
    // uploads cleanly once.
    let resp = client()
        .put(format!("{base}/upload/{sig}/{name}"))
        .body(test_png())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let _ = std::fs::remove_dir_all(&dir);
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
    .unwrap_err()
    .to_string();
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

/// A float TIFF source (Rgb32F samples) renders to PNG, plain
/// and interlaced, and to AVIF, named or negotiated from a
/// browser's `Accept` (issue #464). Each of these was a 500
/// `encode_failed`: the encoders take no float samples.
#[tokio::test]
async fn float_sources_render_to_png_and_avif() {
    let float = DynamicImage::ImageRgb32F(image::Rgb32FImage::from_fn(64, 48, |x, y| {
        image::Rgb([x as f32 / 63.0, y as f32 / 47.0, 0.5])
    }));
    let mut tiff = Vec::new();
    float
        .write_to(
            &mut std::io::Cursor::new(&mut tiff),
            image::ImageFormat::Tiff,
        )
        .unwrap();
    let src = spawn_bytes_server(tiff).await;
    let base = spawn_app(false).await;
    for (ops, accept, content_type) in [
        ("filters:format(png)", None, "image/png"),
        ("filters:format(png,interlace)", None, "image/png"),
        ("filters:format(avif)", None, "image/avif"),
        ("32x0", Some("image/avif,image/webp,*/*"), "image/avif"),
    ] {
        let remainder = format!("{ops}/{src}");
        let mut request = client().get(format!("{base}/{}/{remainder}", sign(&remainder)));
        if let Some(accept) = accept {
            request = request.header("accept", accept);
        }
        let resp = request.send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{ops}");
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            content_type,
            "{ops}"
        );
        let body = resp.bytes().await.unwrap();
        if content_type == "image/png" {
            let decoded = image::load_from_memory(&body).unwrap();
            assert_eq!((decoded.width(), decoded.height()), (64, 48), "{ops}");
            assert_eq!(decoded.to_rgb16(), float.to_rgb16(), "{ops}");
        }
    }
}

/// A multi-digit zero side (`00x100`) is valid grammar that
/// asks for an empty image: a `400 unsupported_operation` in
/// both error modes, where it was a 500 from the encoder
/// (issue #464). DPR and fit-in used to round it up to a
/// 1-pixel render instead.
#[tokio::test]
async fn zero_pixel_sides_answer_400() {
    let src = spawn_image_server().await;
    let base = spawn_app(false).await;
    let message =
        "requested size has a zero pixel side; write 0 alone for a proportional or unbounded axis";
    for ops in ["00x100", "100x00", "00x100/filters:dpr(2)", "fit-in:00x100"] {
        let remainder = format!("{ops}/{src}");
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
            StatusCode::BAD_REQUEST,
            "unsupported_operation",
            message,
        )
        .await;
    }
}

/// A size the engine works out past the per-axis limit is
/// refused like a size written past it (issue #521):
/// `/unsafe/16383x1/0x2/...` rendered 32766x2, wider than
/// WebP encodes. `/explain` forecasts the same refusal.
#[tokio::test]
async fn computed_sizes_past_max_dimension_answer_400() {
    let src = spawn_image_server().await;
    let base = spawn_app(true).await;
    for (ops, refused) in [
        ("16383x1/0x2", "32766x2"),
        ("16383x1/fit-in:0x2", "32766x2"),
        ("crop:16383x1:smart/0x2", "32766x2"),
        ("4096x1/0x2/filters:dpr(2)", "16384x4"),
    ] {
        let url = format!("{base}/unsafe/{ops}/{src}");
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
            StatusCode::BAD_REQUEST,
            "unsupported_operation",
            &format!("requested output {refused} exceeds the 16383 pixel limit per axis"),
        )
        .await;
        let resp = client()
            .get(format!("{base}/explain/unsafe/{ops}/{src}"))
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{ops}");
        let report: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(report["forecast"]["over_dimension"], true, "{ops}");
    }
    // A chain that lands exactly on the limit renders.
    let resp = client()
        .get(format!("{base}/unsafe/5461x1/0x3/{src}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let img = image::load_from_memory(&resp.bytes().await.unwrap()).unwrap();
    assert_eq!((img.width(), img.height()), (16383, 3));
}

/// A width hint on a source at the 16384 px decode ceiling
/// answers what the URL answers without it (issue #521):
/// the hint's `Wx0` op is held at the source by its
/// `no_upscale`, so it keeps the size it started from and
/// works nothing out. The first cut of the per-axis check
/// refused it at the step, turning an advisory hint into a
/// 400 for a URL that rendered. Since issue #528 the output
/// itself is held to the limit: a URL that leaves the source
/// at the ceiling answers 400 with and without a hint, and
/// one that brings it within the limit renders with and
/// without one.
#[tokio::test]
async fn a_width_hint_answers_a_decode_ceiling_source_as_the_url_does() {
    let src = spawn_bytes_server(png_of(1, 16384)).await;
    let base = spawn_app(false).await;
    let left = format!("filters:format(png)/{src}");
    let resized = format!("0x8192/filters:format(png)/{src}");
    for hints in [
        &[][..],
        &[("Sec-CH-Width", "400")][..],
        &[("Width", "3840"), ("Sec-CH-DPR", "2")][..],
    ] {
        let resp = get_with_headers(&base, &left, hints).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{hints:?}");
        assert_eq!(
            resp.text().await.unwrap(),
            "requested output 1x16384 exceeds the 16383 pixel limit per axis; \
             resize or crop it to fit",
            "{hints:?}"
        );
    }
    for hints in [
        &[][..],
        &[("Sec-CH-Width", "400")][..],
        &[("Width", "3840")][..],
    ] {
        let resp = get_with_headers(&base, &resized, hints).await;
        assert_eq!(resp.status(), StatusCode::OK, "{hints:?}");
        let img = image::load_from_memory(&resp.bytes().await.unwrap()).unwrap();
        assert_eq!((img.width(), img.height()), (1, 8192), "{hints:?}");
    }
}

/// A source at the 16384 px decode ceiling, one past the
/// output limit, rendered through a chain that keeps that
/// side (issue #528). WebP cannot encode it: a negotiated
/// or named WebP answered `500 image encoding failed`,
/// still or animated, and other formats served 16384 px.
/// Every such request now answers the documented `400`,
/// and a resize within the limit renders, as WebP too.
#[tokio::test]
async fn a_decode_ceiling_source_answers_400_never_500() {
    let still = spawn_bytes_server(png_of(16384, 1)).await;
    let delay = image::Delay::from_numer_denom_ms(100, 1);
    let frames: Vec<image::Frame> = [[255, 0, 0, 255], [0, 0, 255, 255]]
        .into_iter()
        .map(|rgba| {
            let pixels = image::RgbaImage::from_pixel(16384, 1, image::Rgba(rgba));
            image::Frame::from_parts(pixels, 0, 0, delay)
        })
        .collect();
    let mut gif = Vec::new();
    image::codecs::gif::GifEncoder::new(&mut gif)
        .encode_frames(frames)
        .unwrap();
    let animated = spawn_bytes_server(gif).await;
    let base = spawn_app(true).await;
    let browser = "image/avif;q=0,image/webp,image/apng,image/*,*/*;q=0.8";
    let refused =
        "requested output 16384x1 exceeds the 16383 pixel limit per axis; resize or crop it to fit";
    for src in [&still, &animated] {
        for (ops, accept) in [
            ("", browser),
            ("filters:format(webp)/", "*/*"),
            ("", "*/*"),
            ("filters:grayscale()/", browser),
        ] {
            let url = format!("{base}/unsafe/{ops}{src}");
            let plain = client()
                .get(&url)
                .header("accept", accept)
                .send()
                .await
                .unwrap();
            let json = client()
                .get(&url)
                .header("accept", "application/json")
                .send()
                .await
                .unwrap();
            assert_error_modes(
                plain,
                json,
                StatusCode::BAD_REQUEST,
                "unsupported_operation",
                refused,
            )
            .await;
        }
        // `/explain` forecasts the same refusal.
        let resp = client()
            .get(format!("{base}/explain/unsafe/{src}"))
            .header("accept", "application/json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let report: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(report["forecast"]["over_dimension"], true, "{src}");
        // A resize within the limit renders as the
        // negotiated WebP.
        let resp = client()
            .get(format!("{base}/unsafe/8192x0/{src}"))
            .header("accept", browser)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{src}");
        assert_eq!(resp.headers()["content-type"], "image/webp", "{src}");
        let img = image::load_from_memory(&resp.bytes().await.unwrap()).unwrap();
        assert_eq!((img.width(), img.height()), (8192, 1), "{src}");
    }
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
async fn metrics_bearer_scheme_is_case_insensitive() {
    // Issue #454: the auth scheme is case-insensitive
    // (RFC 9110), so every casing of `Bearer` reaches the
    // token check. The token itself stays case-sensitive.
    let base = spawn_app(false).await;
    for scheme in ["Bearer", "bearer", "BEARER", "bEaReR"] {
        let resp = client()
            .get(format!("{base}/metrics"))
            .header("authorization", format!("{scheme} {TEST_METRICS_TOKEN}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{scheme}");
    }
    let resp = client()
        .get(format!("{base}/metrics"))
        .header(
            "authorization",
            format!("bearer {}", TEST_METRICS_TOKEN.to_ascii_uppercase()),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn metrics_refuse_wrong_and_hostile_authorization() {
    // Issue #454: a wrong token answers the same 401
    // whether or not its length matches the configured
    // token's, malformed headers answer it too, and the
    // server keeps serving the real token afterwards.
    let base = spawn_app(false).await;
    let same_length = {
        let mut t = TEST_METRICS_TOKEN.as_bytes().to_vec();
        let last = t.len() - 1;
        t[last] ^= 1;
        String::from_utf8(t).unwrap()
    };
    assert_eq!(same_length.len(), TEST_METRICS_TOKEN.len());
    assert_ne!(same_length, TEST_METRICS_TOKEN);
    let shorter = &TEST_METRICS_TOKEN[..TEST_METRICS_TOKEN.len() - 1];
    let longer = format!("{TEST_METRICS_TOKEN}x");
    let huge = "a".repeat(8 * 1024);
    let mut values: Vec<axum::http::HeaderValue> = [
        format!("Bearer {same_length}"),
        format!("Bearer {shorter}"),
        format!("Bearer {longer}"),
        format!("Bearer {huge}"),
        "Bearer x".to_string(),
        "Bearer".to_string(),
        format!("Bearer{TEST_METRICS_TOKEN}"),
        format!("Bearer\t{TEST_METRICS_TOKEN}"),
        format!("Bearerx {TEST_METRICS_TOKEN}"),
        format!("Basic {TEST_METRICS_TOKEN}"),
        format!("{TEST_METRICS_TOKEN} Bearer"),
        TEST_METRICS_TOKEN.to_string(),
    ]
    .iter()
    .map(|v| axum::http::HeaderValue::from_str(v).unwrap())
    .collect();
    // Not UTF-8: the header cannot be read as text at all.
    let mut raw = b"Bearer \xff".to_vec();
    raw.extend_from_slice(TEST_METRICS_TOKEN.as_bytes());
    values.push(axum::http::HeaderValue::from_bytes(&raw).unwrap());
    for value in values {
        let resp = client()
            .get(format!("{base}/metrics"))
            .header("accept", "application/json")
            .header("authorization", value.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED, "{value:?}");
        let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "metrics_unauthorized", "{value:?}");
    }
    let resp = metrics_response(&base).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[test]
fn token_eq_matches_only_the_exact_token_at_any_length() {
    // Issue #454: equality is unchanged by comparing
    // digests; only the timing stopped depending on the
    // presented length.
    assert!(token_eq(b"secret-token", b"secret-token"));
    assert!(!token_eq(b"secret-token", b"secret-tokem"));
    assert!(!token_eq(b"secret-token", b"secret-toke"));
    assert!(!token_eq(b"secret-toke", b"secret-token"));
    assert!(!token_eq(b"", b"secret-token"));
    assert!(!token_eq(&[0u8; 4096], b"secret-token"));
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
async fn spawn_explain_app(presets: crate::url::Presets, loader: Loader) -> String {
    spawn_explain_app_with_roots(presets, loader, std::collections::HashMap::new()).await
}

async fn spawn_explain_app_with_roots(
    presets: crate::url::Presets,
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        slow_gate: Arc::new(Gate::new(2)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: Default::default(),
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
    presets.insert(
        "small".to_string(),
        "100x50/filters:grayscale".parse().unwrap(),
    );
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
    presets.insert("a&b".to_string(), "100x50".parse().unwrap());
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

/// Serve `state` on an ephemeral port: the tests that read
/// `/metrics` while they drive the state directly.
async fn serve_state(state: &AppState) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = state.clone();
    tokio::spawn(async move {
        axum::serve(listener, app(state)).await.unwrap();
    });
    format!("http://{addr}")
}

/// The `/metrics` exposition, read with `token`.
async fn scrape(base: &str, token: &str) -> String {
    let resp = client()
        .get(format!("{base}/metrics"))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    resp.text().await.unwrap()
}

#[tokio::test]
async fn shared_write_throughs_past_the_bound_are_skipped_and_counted() {
    // Issue #463: every shared write-through was a bare
    // spawn, so a stalled bucket let them pile up, one per
    // miss, each holding a whole entry. Now at most
    // SHARED_PUTS_AT_ONCE are in flight, and one that finds
    // every slot taken is skipped and counted, not queued.
    let fake = crate::s3store::test_support::start_fake().await;
    fake.hold_puts.store(true, Ordering::SeqCst);
    let mut state = tier_test_state(None, &fake.endpoint);
    state.metrics_token = Some("metrics-token".to_string());
    let shared = state.shared_store.clone().unwrap();
    let base = serve_state(&state).await;
    let snapshot = state.purge_guard.snapshot();
    let key = |i: usize| format!("{i}/example.com/a.png");
    for i in 0..SHARED_PUTS_AT_ONCE {
        spawn_shared_put(
            &state,
            snapshot,
            &key(i),
            tier_test_entry(),
            SharedWrite::Render,
        );
    }
    wait_until("every slot's write reaches the stalled bucket", || {
        fake.held_puts.load(Ordering::SeqCst) == SHARED_PUTS_AT_ONCE
    })
    .await;
    let metrics = scrape(&base, "metrics-token").await;
    assert!(
        metrics.contains(&format!(
            "crabcrop_cache_shared_puts_in_flight {SHARED_PUTS_AT_ONCE}\n"
        )),
        "{metrics}"
    );
    assert!(
        metrics.contains(&format!(
            "crabcrop_cache_shared_puts_capacity {SHARED_PUTS_AT_ONCE}\n"
        )),
        "{metrics}"
    );
    assert!(
        metrics.contains("crabcrop_cache_shared_puts_skipped_total 0\n"),
        "{metrics}"
    );

    // Every slot is taken: the next write is skipped on the
    // spot and counted, and nothing more reaches the bucket.
    spawn_shared_put(
        &state,
        snapshot,
        &key(1000),
        tier_test_entry(),
        SharedWrite::Render,
    );
    assert_eq!(state.background.puts_skipped(), 1);
    let metrics = scrape(&base, "metrics-token").await;
    assert!(
        metrics.contains("crabcrop_cache_shared_puts_skipped_total 1\n"),
        "{metrics}"
    );
    assert_eq!(fake.held_puts.load(Ordering::SeqCst), SHARED_PUTS_AT_ONCE);

    // The bucket answers: the held writes land, tracked to
    // the end, and give their slots back.
    fake.hold_puts.store(false, Ordering::SeqCst);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(60);
    let left = state.background.drain(deadline).await;
    assert_eq!(
        left,
        Drained {
            writes: 0,
            walks: 0
        }
    );
    assert_eq!(state.background.puts_in_flight(), 0);
    assert_eq!(entry_objects(&fake), SHARED_PUTS_AT_ONCE);
    assert!(shared.get(&key(0)).await.is_some());
    assert!(shared.get(&key(1000)).await.is_none());

    // With a slot free again, a write goes through.
    spawn_shared_put(
        &state,
        snapshot,
        &key(1001),
        tier_test_entry(),
        SharedWrite::Render,
    );
    state.background.drain(deadline).await;
    assert!(shared.get(&key(1001)).await.is_some());
    assert_eq!(state.background.puts_skipped(), 1);
    let metrics = scrape(&base, "metrics-token").await;
    assert!(
        metrics.contains("crabcrop_cache_shared_puts_in_flight 0\n"),
        "{metrics}"
    );
}

#[tokio::test]
async fn the_shutdown_drain_waits_for_a_write_through_until_its_deadline() {
    // Issue #463: at shutdown the detached writes died with
    // the runtime once the connections drained. The drain
    // now waits for them, and stops waiting at its deadline.
    let fake = crate::s3store::test_support::start_fake().await;
    fake.hold_puts.store(true, Ordering::SeqCst);
    let state = tier_test_state(None, &fake.endpoint);
    let shared = state.shared_store.clone().unwrap();
    let key = "40x24/example.com/i.png";
    spawn_shared_put(
        &state,
        state.purge_guard.snapshot(),
        key,
        tier_test_entry(),
        SharedWrite::Render,
    );
    wait_until("the write reaches the stalled bucket", || {
        fake.held_puts.load(Ordering::SeqCst) == 1
    })
    .await;
    // The bucket holds the write past the deadline: the
    // drain gives up at it and reports the write.
    let grace = std::time::Duration::from_millis(300);
    let started = tokio::time::Instant::now();
    let left = state.background.drain(started + grace).await;
    assert!(started.elapsed() >= grace);
    assert_eq!(
        left,
        Drained {
            writes: 1,
            walks: 0
        }
    );
    assert_eq!(entry_objects(&fake), 0);
    // Answered in time, the write lands before the drain
    // returns.
    fake.hold_puts.store(false, Ordering::SeqCst);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(60);
    let left = state.background.drain(deadline).await;
    assert_eq!(
        left,
        Drained {
            writes: 0,
            walks: 0
        }
    );
    assert!(shared.get(key).await.is_some());
}

#[tokio::test]
async fn the_shutdown_drain_counts_purge_walks_but_never_waits_for_them() {
    // Issue #463: a purge's background walk is tracked, but
    // a shutdown does not wait for it. Cut short, it leaves
    // the purge's tombstone unended, which keeps the purge
    // in force (issue #461).
    let fake = crate::s3store::test_support::start_fake().await;
    {
        let mut objects = fake.objects.lock().unwrap();
        for i in 0..SHARED_PURGE_INLINE_SCAN + 8 {
            objects.insert(format!("0-filler-{i:04}"), b"not a cache entry".to_vec());
        }
    }
    // The request scans the inline budget itself; the
    // background walk parks on its first read past it.
    *fake.hold_ranged_after.lock().unwrap() = Some(SHARED_PURGE_INLINE_SCAN);
    let state = tier_test_state(None, &fake.endpoint);
    purge_source_everywhere(&state, "example.com/i.png").await;
    wait_until("the background walk parks", || {
        fake.ranged_gets.load(Ordering::SeqCst) > SHARED_PURGE_INLINE_SCAN
    })
    .await;
    assert_eq!(state.purge_guard.active_count(), 1);
    // A deadline far off: the drain returns as soon as the
    // writes are done, with the parked walk still running.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(600);
    let left = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        state.background.drain(deadline),
    )
    .await
    .expect("the drain must not wait for the walk");
    assert_eq!(
        left,
        Drained {
            writes: 0,
            walks: 1
        }
    );
    assert_eq!(state.purge_guard.active_count(), 1);
    // Released, the walk ends the purge as before.
    *fake.hold_ranged_after.lock().unwrap() = None;
    wait_until("the purge walk ends", || {
        state.purge_guard.active_count() == 0
    })
    .await;
}

/// Fill every shared write slot against a stalled bucket:
/// `SHARED_PUTS_AT_ONCE` render write-throughs, each held.
async fn hold_every_shared_write_slot(
    state: &AppState,
    fake: &crate::s3store::test_support::FakeS3,
) {
    fake.hold_puts.store(true, Ordering::SeqCst);
    let snapshot = state.purge_guard.snapshot();
    for i in 0..SHARED_PUTS_AT_ONCE {
        spawn_shared_put(
            state,
            snapshot,
            &format!("{i}/example.com/filler.png"),
            tier_test_entry(),
            SharedWrite::Render,
        );
    }
    wait_until("every slot's write reaches the stalled bucket", || {
        fake.held_puts.load(Ordering::SeqCst) == SHARED_PUTS_AT_ONCE
    })
    .await;
}

/// A group-free copy is served to a grouped request while
/// every shared write slot is taken, from memory or from the
/// bucket; then the bucket recovers and the group is purged.
/// The purge must reach the bucket copy, on this replica and
/// on any other.
async fn group_purge_after_a_hit_with_every_write_slot_taken(from_memory: bool) {
    // Issue #463 review: the bound skipped a tier hit's
    // group write-back like a render write-through. The
    // local merge had already recorded the group, so nothing
    // sent it again; the bucket copy never learned it, the
    // group purge left it, and every replica served it.
    let fake = crate::s3store::test_support::start_fake().await;
    let state = tier_test_state(None, &fake.endpoint);
    let shared = state.shared_store.clone().unwrap();
    let key = "40x24/example.com/hero.png";
    let entry = tier_test_entry();
    shared.put(key, &entry).await;
    if from_memory {
        put_memory_tier(&state, state.purge_guard.snapshot(), key, &entry);
    }
    hold_every_shared_write_slot(&state, &fake).await;

    let groups = vec!["hero".to_string()];
    let (_, status) = lookup_tiers(&state, key, &groups).await.unwrap();
    let expected = if from_memory {
        "crabcrop; hit"
    } else {
        "crabcrop; hit; detail=\"shared\""
    };
    assert_eq!(status, expected);
    // The write-back waits for a slot rather than take a
    // sixty-fifth: the bound on writes in flight holds.
    assert_eq!(state.background.puts_in_flight(), SHARED_PUTS_AT_ONCE);
    assert_eq!(fake.held_puts.load(Ordering::SeqCst), SHARED_PUTS_AT_ONCE);
    // The next hit is a memory hit that already carries the
    // group: it sends nothing.
    let (_, status) = lookup_tiers(&state, key, &groups).await.unwrap();
    assert_eq!(status, "crabcrop; hit");

    // The bucket recovers: the held writes land, then the
    // write-back takes a freed slot and lands too.
    fake.hold_puts.store(false, Ordering::SeqCst);
    let far = tokio::time::Instant::now() + std::time::Duration::from_secs(60);
    assert_eq!(state.background.drain(far).await.writes, 0);
    assert_eq!(
        shared.get(key).await.unwrap().groups,
        groups,
        "the bucket copy never learned the group"
    );
    assert_eq!(state.background.puts_skipped(), 0);

    let base = serve_state(&state).await;
    let sig = crate::url::purge_group_signature(SECRET, "hero");
    let resp = client()
        .post(format!("{base}/purge-group/{sig}/hero"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.text().await.unwrap();
    wait_until("the group purge ends", || {
        state.purge_guard.active_count() == 0
    })
    .await;
    assert!(
        shared.get(key).await.is_none(),
        "the purge answered {body:?}, yet the bucket copy survived it"
    );
    assert!(lookup_tiers(&state, key, &groups).await.is_none());
    // Another replica, with the bucket's tombstones read,
    // misses as well.
    let other = tier_test_state(None, &fake.endpoint);
    shared.refresh_tombstones(&other.purge_guard).await;
    assert!(lookup_tiers(&other, key, &groups).await.is_none());
}

#[tokio::test]
async fn a_memory_hit_group_write_back_waits_for_a_write_slot_and_is_purged() {
    group_purge_after_a_hit_with_every_write_slot_taken(true).await;
}

#[tokio::test]
async fn a_shared_hit_group_write_back_waits_for_a_write_slot_and_is_purged() {
    group_purge_after_a_hit_with_every_write_slot_taken(false).await;
}

#[tokio::test]
async fn a_render_write_a_purge_covers_is_dropped_but_not_counted_as_skipped() {
    // Issue #463 review: the skip counter is for renders the
    // bucket missed because every slot was taken. A write a
    // running purge already covers would never land, so it
    // is not one of them.
    let fake = crate::s3store::test_support::start_fake().await;
    let state = tier_test_state(None, &fake.endpoint);
    let snapshot = state.purge_guard.snapshot();
    hold_every_shared_write_slot(&state, &fake).await;
    let covered = "40x24/example.com/covered.png";
    let _purge = state
        .purge_guard
        .begin(crate::cache::PurgeTarget::Key(covered.to_string()));
    spawn_shared_put(
        &state,
        snapshot,
        covered,
        tier_test_entry(),
        SharedWrite::Render,
    );
    assert_eq!(state.background.puts_skipped(), 0);
    // One the purge does not cover is skipped and counted.
    spawn_shared_put(
        &state,
        snapshot,
        "40x24/example.com/other.png",
        tier_test_entry(),
        SharedWrite::Render,
    );
    assert_eq!(state.background.puts_skipped(), 1);
    fake.hold_puts.store(false, Ordering::SeqCst);
    let far = tokio::time::Instant::now() + std::time::Duration::from_secs(60);
    assert_eq!(state.background.drain(far).await.writes, 0);
    assert_eq!(entry_objects(&fake), SHARED_PUTS_AT_ONCE);
}

/// A minimal, valid SVG that sizes instantly but carries
/// [`crate::engine::SVG_PROBE_SLOW_MARKER`], so the probe's
/// test hook sleeps while sizing it: a stand-in for an SVG
/// whose parse is slow, without crafting a slow document.
const SLOW_PROBE_SVG: &[u8] = b"<!--crabcrop-slow-probe--><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><rect width=\"10\" height=\"10\"/></svg>";

/// A blocking HTTP origin on its own OS thread (not the
/// server's runtime), so serving the source never competes
/// with the server for a request thread: the only thing
/// that holds a request thread in these tests is the probe.
fn spawn_blocking_svg_origin() -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            std::thread::spawn(move || {
                let mut buf = [0u8; 2048];
                let _ = stream.read(&mut buf);
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/svg+xml\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    SLOW_PROBE_SVG.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(SLOW_PROBE_SVG);
                let _ = stream.flush();
            });
        }
    });
    format!("http://{addr}/slow.svg")
}

/// Fetch `url` over a plain blocking socket, off the tokio
/// runtime, and return its status and how long it took. Used
/// to time /health while the server's request threads may be
/// blocked: a measurement on the runtime could not run then.
fn blocking_get(url: &str) -> (u16, std::time::Duration) {
    use std::io::{Read, Write};
    let rest = url.strip_prefix("http://").expect("http url");
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let addr: std::net::SocketAddr = authority.parse().expect("host:port");
    let start = std::time::Instant::now();
    let mut stream =
        std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(20)).unwrap();
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(20)))
        .unwrap();
    let req = format!("GET {path} HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).unwrap();
    let mut buf = Vec::new();
    let _ = stream.read_to_end(&mut buf);
    let elapsed = start.elapsed();
    let status = std::str::from_utf8(&buf)
        .ok()
        .and_then(|t| t.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    (status, elapsed)
}

/// Fire several concurrent /explain requests for a
/// slow-to-size SVG (the `urls` are distinct, so the loader
/// does not coalesce them into one probe) and, from off the
/// runtime, time /health while they run. The probes run off
/// the request threads (issue #447), so /health answers
/// promptly; on a server that sizes SVG on the request
/// threads, the probes hold every request thread and /health
/// waits for one to finish. More requests than the two
/// request threads keeps them held for the whole window.
async fn health_stays_responsive_during_slow_explains(urls: Vec<String>, base: &str) {
    let health_url = format!("{base}/health");
    let prober = std::thread::spawn(move || {
        // Let the explains seize the request threads first.
        std::thread::sleep(std::time::Duration::from_millis(400));
        blocking_get(&health_url)
    });
    let mut tasks = Vec::new();
    for url in urls {
        tasks.push(tokio::spawn(async move {
            client()
                .get(url)
                .timeout(std::time::Duration::from_secs(20))
                .send()
                .await
        }));
    }
    let (status, elapsed) = tokio::task::spawn_blocking(move || prober.join().unwrap())
        .await
        .unwrap();
    assert_eq!(status, 200, "/health did not answer 200");
    assert!(
        elapsed < std::time::Duration::from_secs(1),
        "/health took {elapsed:?} while slow SVG explains ran: the probes held the request threads"
    );
    for task in tasks {
        let _ = within(
            std::time::Duration::from_secs(20),
            "a slow explain to finish",
            task,
        )
        .await;
    }
}

/// A burst of /explain probes for a slow-to-size SVG file
/// source leaves /health responsive: the probes run off the
/// request threads (issue #447). On a server that sized SVG
/// on the request threads, these would hold /health until a
/// probe finished.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_slow_svg_file_explains_leave_health_responsive() {
    let dir = crate::test_support::scratch_dir("explain-slow-svg-file");
    let root = std::fs::canonicalize(&dir).unwrap();
    std::fs::write(root.join("slow.svg"), SLOW_PROBE_SVG).unwrap();
    let loader = Loader::with_defaults()
        .unwrap()
        .with_sources(crate::loader::SourceConfig {
            file_root: Some(root),
            ..Default::default()
        });
    let base = spawn_explain_app_with_roots(Default::default(), loader, Default::default()).await;
    // Distinct output sizes, so the requests are distinct and
    // each runs its own probe.
    let urls = (0..6)
        .map(|i| {
            let remainder = format!("64x{}/file:///slow.svg", 60 + i);
            format!("{base}/explain/{}/{remainder}", sign(&remainder))
        })
        .collect();
    health_stays_responsive_during_slow_explains(urls, &base).await;
    drop(dir);
}

/// The same as [`concurrent_slow_svg_file_explains_leave_health_responsive`]
/// over HTTP, where a probe runs for both the per-chunk
/// settle check and the handler on a server that sizes SVG
/// on the request threads.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_slow_svg_http_explains_leave_health_responsive() {
    let origin = spawn_blocking_svg_origin();
    let base = spawn_explain_app_with_roots(
        Default::default(),
        Loader::with_defaults().unwrap(),
        Default::default(),
    )
    .await;
    let urls = (0..6)
        .map(|i| {
            let remainder = format!("64x{}/{origin}", 60 + i);
            format!("{base}/explain/{}/{remainder}", sign(&remainder))
        })
        .collect();
    health_stays_responsive_during_slow_explains(urls, &base).await;
}
