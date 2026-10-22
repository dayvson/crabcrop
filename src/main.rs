//! crabcrop: on-demand image optimization.
//!
//! Every image operation is encoded in the request URL:
//!
//! ```text
//! /<signature>/<operations...>/<image-url>
//! ```
//!
//! The first path segment is the request [`url::Signature`]: either an
//! HMAC-SHA256 of the rest of the path, or the literal `unsafe` when the
//! server allows unsigned URLs.
//!
//! Configuration comes from the environment:
//!
//! - `CRABCROP_SECRET` (required): the HMAC secret.
//! - `CRABCROP_ALLOW_UNSAFE`: `1` or `true` to honor `unsafe` URLs.
//! - `CRABCROP_CACHE_MAX_MB`: result cache budget in megabytes,
//!   default 256; `0` disables the cache.
//! - `CRABCROP_ALLOWED_SOURCES`: comma-separated host suffixes the
//!   loader may fetch from; unset allows every host.
//! - `CRABCROP_BLOCK_PRIVATE_SOURCES`: `1` or `true` to refuse
//!   sources resolving to loopback, private, or link-local
//!   addresses. Off by default.
//! - `CRABCROP_REMOVEBG_MODEL`: path to a U²-Net-family ONNX
//!   model for the `removebg` filter. Preloaded at boot; a set
//!   but unloadable model is a startup error. Unset disables
//!   the filter (requests fail cleanly).
//! - `CRABCROP_UPSCALE_MODEL`: path to a super-resolution ONNX
//!   model for the `upscale` filter. Same preload rule.
//! - `CRABCROP_FACE_MODEL`: path to a YuNet ONNX model for
//!   face-aware smart crops. Same preload rule; unset means
//!   smart crops use the saliency estimate.
//! - `CRABCROP_FONT_PATH`: path to a TTF/OTF font for the
//!   `text(...)` overlay filter, loaded at boot. Unset
//!   disables the filter (requests fail cleanly).
//! - `CRABCROP_FALLBACK_IMAGE`: a URL or local file path for
//!   the fallback image served (through the same operations,
//!   marked `X-Crabcrop-Fallback: 1`) when a source cannot be
//!   loaded or decoded. Unset keeps today's error behavior.
//! - `CRABCROP_RESULT_STORAGE_DIR`: directory for the
//!   disk-backed result store (the persistent tier under the
//!   memory cache). Unset disables it.
//!   `CRABCROP_RESULT_STORAGE_MAX_MB` bounds it (default 512;
//!   `0` disables storage even with a directory set).
//! - `CRABCROP_ORT_DYLIB`: path to the ONNX Runtime shared
//!   library, when it is not on the loader's search path.
//! - `PORT`: listen port, default 3000.

#![forbid(unsafe_code)]

mod ai;
mod cache;
mod crop;
mod engine;
mod filters;
mod focus;
mod loader;
mod observe;
mod placeholder;
mod quality;
mod server;
mod storage;
mod svg;
mod text;
mod url;

use std::env;
use std::sync::Arc;

use crate::ai::UpscaleProvider;
use crate::cache::Cache;
use crate::engine::Engine;
use crate::loader::{Loader, SourcePolicy};
use crate::observe::Metrics;
use crate::server::{AppState, app};

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("crabcrop: {e}");
        std::process::exit(1);
    }
}

/// Read an environment variable as `T`, or fall back to `default`
/// when it is unset. A set-but-invalid value is a startup error
/// naming the variable: silently serving on the wrong port or
/// with the wrong cache budget is worse than refusing to start.
fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String> {
    parse_env_value(name, env::var(name).ok(), default)
}

fn parse_env_value<T: std::str::FromStr>(
    name: &str,
    value: Option<String>,
    default: T,
) -> Result<T, String> {
    match value {
        Some(v) => v.parse().map_err(|_| format!("invalid {name}: {v:?}")),
        None => Ok(default),
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let secret = env::var("CRABCROP_SECRET")
        .map_err(|_| "CRABCROP_SECRET must be set")?
        .into_bytes();
    let allow_unsafe = env::var("CRABCROP_ALLOW_UNSAFE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let port: u16 = parse_env("PORT", 3000)?;
    let cache_mb: usize = parse_env("CRABCROP_CACHE_MAX_MB", 256)?;
    let cache_bytes = cache_mb
        .checked_mul(1024 * 1024)
        .ok_or("CRABCROP_CACHE_MAX_MB is too large")?;
    let allowed_sources: Vec<String> = env::var("CRABCROP_ALLOWED_SOURCES")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let block_private = env::var("CRABCROP_BLOCK_PRIVATE_SOURCES")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let policy = SourcePolicy::new(allowed_sources, block_private);

    // AI models are configured, never downloaded, and preloaded
    // at boot: a model that is set but broken is a startup
    // error, not a surprise for the first request that uses it.
    let ort_dylib = env::var("CRABCROP_ORT_DYLIB")
        .ok()
        .filter(|v| !v.is_empty());
    let removebg: Option<Arc<dyn crate::ai::MaskProvider>> =
        match env::var("CRABCROP_REMOVEBG_MODEL") {
            Ok(path) if !path.is_empty() => {
                let provider = crate::ai::OnnxMaskProvider::load(ort_dylib.as_deref(), &path)
                    .map_err(|e| format!("cannot load the background-removal model: {e}"))?;
                println!("background removal model loaded from {path}");
                Some(Arc::new(provider))
            }
            _ => None,
        };
    let upscale: Option<Arc<dyn crate::ai::UpscaleProvider>> =
        match env::var("CRABCROP_UPSCALE_MODEL") {
            Ok(path) if !path.is_empty() => {
                let provider = crate::ai::OnnxUpscaleProvider::load(ort_dylib.as_deref(), &path)
                    .map_err(|e| format!("cannot load the upscale model: {e}"))?;
                println!(
                    "upscale model loaded from {path} (native factor {}x)",
                    provider.factor()
                );
                Some(Arc::new(provider))
            }
            _ => None,
        };
    let face: Option<Arc<dyn crate::ai::FaceProvider>> = match env::var("CRABCROP_FACE_MODEL") {
        Ok(path) if !path.is_empty() => {
            let provider = crate::ai::OnnxFaceProvider::load(ort_dylib.as_deref(), &path)
                .map_err(|e| format!("cannot load the face detection model: {e}"))?;
            println!("face detection model loaded from {path}");
            Some(Arc::new(provider))
        }
        _ => None,
    };

    // The overlay font follows the same rule as the models:
    // configured by path, loaded at boot, broken is fatal.
    let font: Option<Arc<dyn crate::text::TextRenderer>> = match env::var("CRABCROP_FONT_PATH") {
        Ok(path) if !path.is_empty() => {
            let renderer = crate::text::FontRenderer::load(&path)
                .map_err(|e| format!("cannot load the overlay font: {e}"))?;
            println!("overlay font loaded from {path}");
            Some(Arc::new(renderer))
        }
        _ => None,
    };

    // The fallback image: an http(s) value is fetched per
    // fallback event through the loader; anything else is a
    // local path read once now, broken is fatal like the rest
    // of the boot config.
    let fallback = match env::var("CRABCROP_FALLBACK_IMAGE") {
        Ok(v) if !v.is_empty() => {
            if v.starts_with("http://") || v.starts_with("https://") {
                println!("fallback image will be fetched from {v}");
                Some(server::Fallback::Url(v))
            } else {
                let bytes = std::fs::read(&v)
                    .map_err(|e| format!("cannot read the fallback image {v}: {e}"))?;
                println!("fallback image loaded from {v} ({} bytes)", bytes.len());
                Some(server::Fallback::Bytes(Arc::new(bytes)))
            }
        }
        _ => None,
    };

    // Disk result storage: configured by directory, bounded
    // by its own budget. Opening indexes whatever a previous
    // run left behind.
    let store = match env::var("CRABCROP_RESULT_STORAGE_DIR") {
        Ok(dir) if !dir.is_empty() => {
            let max_mb: usize = parse_env("CRABCROP_RESULT_STORAGE_MAX_MB", 512)?;
            let max_bytes = max_mb
                .checked_mul(1024 * 1024)
                .ok_or_else(|| "CRABCROP_RESULT_STORAGE_MAX_MB is too large".to_string())?;
            let store = crate::storage::DiskStore::open(std::path::Path::new(&dir), max_bytes)?;
            let (entries, bytes) = store.stats();
            println!(
                "result storage at {dir}: {entries} entries, {bytes} bytes, budget {max_mb} MB"
            );
            Some(Arc::new(store))
        }
        _ => None,
    };

    let state = AppState {
        secret,
        allow_unsafe,
        loader: Loader::with_defaults()
            .map_err(|e| format!("cannot build the loader: {e}"))?
            .with_policy(policy),
        engine: Engine,
        metrics: Arc::new(Metrics::default()),
        cache: Arc::new(Cache::new(cache_bytes)),
        removebg,
        upscale,
        face,
        font,
        fallback,
        store,
        ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
    };
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    println!("crabcrop listening on port {port}");
    axum::serve(listener, app(state)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_env_value_defaults_parses_and_rejects() {
        assert_eq!(parse_env_value::<u16>("PORT", None, 3000).unwrap(), 3000);
        assert_eq!(
            parse_env_value::<u16>("PORT", Some("8080".to_string()), 3000).unwrap(),
            8080
        );
        let err = parse_env_value::<u16>("PORT", Some("abc".to_string()), 3000).unwrap_err();
        assert!(err.contains("PORT"), "{err}");
    }
}
