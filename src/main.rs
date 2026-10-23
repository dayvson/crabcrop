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
//! - `CRABCROP_SECRET` (required, at least 16 bytes): the HMAC secret.
//! - `CRABCROP_ALLOW_UNSAFE`: `1` or `true` to honor `unsafe` URLs.
//! - `CRABCROP_CACHE_MAX_MB`: result cache budget in megabytes,
//!   default 256; `0` disables the cache.
//! - `CRABCROP_SSIM_TARGET`: SSIMULACRA2 target (0-100) that
//!   `quality(auto)` verifies against for JPEG and JXL output,
//!   default 75.
//! - `CRABCROP_AVIF_SPEED`: ravif speed tier for AVIF encodes,
//!   1 (slowest, smallest) to 10 (fastest), default 8.
//! - `CRABCROP_METADATA`: `strip` (default) drops embedded
//!   EXIF/XMP/IPTC on output; `preserve` copies them into
//!   JPEG and PNG outputs.
//! - `CRABCROP_WIDTH_STEP`: quantization step in pixels for
//!   width client hints, 1 to 2000, default 100.
//! - `CRABCROP_JXL_SPEED`: jixel speed tier for JPEG XL encodes,
//!   one of ultrafast, fastest, fast, medium, slow, extraslow,
//!   default fast.
//! - `CRABCROP_ALLOWED_SOURCES`: comma-separated host suffixes the
//!   loader may fetch from; unset allows every host.
//! - `CRABCROP_BLOCK_PRIVATE_SOURCES`: `1` or `true` to refuse
//!   sources resolving to loopback, private, or link-local
//!   addresses. Off by default.
//! - `CRABCROP_S3_ENDPOINT`: base URL of an S3-compatible
//!   store for `s3://` sources (path-style addressing).
//!   Unset means AWS S3 (virtual-hosted, region from
//!   `CRABCROP_S3_REGION`, default `us-east-1`). Credentials
//!   come from the ambient `AWS_ACCESS_KEY_ID`,
//!   `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`
//!   variables; both key parts must be set together. Unset
//!   means unsigned fetches (public buckets).
//! - `CRABCROP_GCS_ENDPOINT` / `CRABCROP_GCS_REGION`: the
//!   GCS XML API base for `gs://` sources (default
//!   `https://storage.googleapis.com`) and its signing region
//!   (default `auto`). Signing uses the HMAC pair
//!   `CRABCROP_GCS_ACCESS_KEY` / `CRABCROP_GCS_SECRET_KEY`;
//!   both or neither.
//! - `CRABCROP_ALLOWED_BUCKETS`: comma-separated bucket
//!   names the `s3://` and `gs://` schemes may read; unset
//!   allows every bucket.
//! - `CRABCROP_FILE_ROOT`: directory that `file://` sources
//!   must stay under, canonicalized at boot. Unset disables
//!   file sources.
//! - `CRABCROP_PRESETS_FILE`: path to a presets file, one
//!   `name = operations` entry per line (`#` comments and
//!   blank lines skipped), validated at boot. Requests may
//!   then use `preset:<name>` as an operation.
//! - `CRABCROP_ONLY_PRESETS`: `1` or `true` rejects
//!   free-form operations in requests; only preset references
//!   (and `exp:` metadata) are allowed. Requires a presets
//!   file with at least one entry.
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
//! - `CRABCROP_CLIP_IMAGE_MODEL` / `CRABCROP_CLIP_TEXT_MODEL`:
//!   a CLIP ONNX model pair for meta `tags` and `alt_text`,
//!   with `CRABCROP_CLIP_TOKENIZER` (the export's
//!   `tokenizer.json`) and `CRABCROP_CLIP_LABELS` (one label
//!   per line). All four are required together; a partial set
//!   is a startup error. `CRABCROP_CLIP_THRESHOLD` optionally
//!   sets the tag score floor (default 0.05). Unset disables
//!   annotation; meta responses then carry no tags.
//! - `CRABCROP_FALLBACK_IMAGE`: a URL or local file path for
//!   the fallback image served (through the same operations,
//!   marked `X-Crabcrop-Fallback: 1`) when a source cannot be
//!   loaded or decoded. Unset keeps today's error behavior.
//! - `CRABCROP_RESULT_STORAGE_DIR`: directory for the
//!   disk-backed result store (the persistent tier under the
//!   memory cache). Unset disables it.
//!   `CRABCROP_RESULT_STORAGE_MAX_MB` bounds it (default 512;
//!   `0` disables storage even with a directory set).
//! - `CRABCROP_DETECT_MODEL`: path to a YOLOv8-family ONNX
//!   model for `blurclass` / `pixelateclass` redaction and
//!   meta detections. Same preload rule. The class
//!   vocabulary comes from the model's `names` metadata;
//!   `CRABCROP_DETECT_LABELS` (one class per line) covers
//!   exports without it. `CRABCROP_DETECT_THRESHOLD` sets
//!   the score floor (default 0.5).
//! - `CRABCROP_ORT_DYLIB`: path to the ONNX Runtime shared
//!   library, when it is not on the loader's search path.
//! - `PORT`: listen port, default 3000.

#![forbid(unsafe_code)]

mod ai;
mod cache;
mod cli;
mod color;
mod crop;
mod engine;
mod filters;
mod focus;
mod loader;
mod observe;
mod placeholder;
mod quality;
mod server;
mod sigv4;
mod storage;
mod svg;
mod text;
mod thumbhash;
mod tokenizer;
mod url;

use std::env;
use std::sync::Arc;

use crate::ai::UpscaleProvider;
use crate::cache::Cache;
use crate::engine::Engine;
use crate::loader::{Loader, SourceConfig, SourcePolicy};
use crate::observe::Metrics;
use crate::server::{AppState, app};

#[tokio::main]
async fn main() {
    // CLI verbs run instead of the server when the first
    // argument names one; anything else starts the server.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().is_some_and(|a| {
        cli::VERBS.contains(&a.as_str()) || a == "--help" || a == "-h" || a == "help"
    }) {
        match cli::exec(&argv).await {
            Ok(out) => {
                if !out.is_empty() {
                    println!("{out}");
                }
            }
            Err(e) => {
                // A bare USAGE error prints to stdout-friendly
                // stderr without the prefix duplication.
                if e.starts_with("crabcrop:") {
                    eprintln!("{e}");
                } else {
                    eprintln!("crabcrop: {e}");
                }
                std::process::exit(1);
            }
        }
        return;
    }
    if let Err(e) = run().await {
        eprintln!("crabcrop: {e}");
        std::process::exit(1);
    }
}

/// Load the preset configuration: the presets file (one
/// `name = operations` entry per line) and the presets-only
/// flag, both validated strictly. Presets-only with no
/// presets configured would reject every transformed
/// request, so it is a startup error rather than a policy.
fn build_presets() -> Result<(std::collections::HashMap<String, String>, bool), String> {
    let only_presets = env::var("CRABCROP_ONLY_PRESETS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let presets = match env::var("CRABCROP_PRESETS_FILE") {
        Ok(path) if !path.is_empty() => {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("cannot read CRABCROP_PRESETS_FILE ({path}): {e}"))?;
            parse_presets(&text)?
        }
        _ => std::collections::HashMap::new(),
    };
    if only_presets && presets.is_empty() {
        return Err(
            "CRABCROP_ONLY_PRESETS is set but CRABCROP_PRESETS_FILE defines no presets".to_string(),
        );
    }
    Ok((presets, only_presets))
}

/// Parse preset definitions: one `name = operations` entry
/// per line, blank lines and `#` comments skipped. Names are
/// letters, digits, `_`, and `-`; every operation string must
/// validate as a preset body (operations only: no source, no
/// `exp:`, no nested `preset:`).
fn parse_presets(text: &str) -> Result<std::collections::HashMap<String, String>, String> {
    let mut map = std::collections::HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, ops) = line
            .split_once('=')
            .ok_or_else(|| format!("preset line {} is not `name = operations`", i + 1))?;
        let name = name.trim();
        let ops = ops.trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(format!("invalid preset name on line {}: {name:?}", i + 1));
        }
        if map.contains_key(name) {
            return Err(format!("duplicate preset name: {name}"));
        }
        crate::url::parse_preset_operations(ops)
            .map_err(|e| format!("invalid operations for preset {name}: {e}"))?;
        map.insert(name.to_string(), ops.to_string());
    }
    Ok(map)
}

/// Build the bucket/file source configuration from the
/// environment. Half-configured credentials and a file root
/// that does not exist are startup errors, under the same
/// strict-config rule as everything else here.
/// Fail closed on bucket access (security audit L-1): when
/// signing credentials are configured, bucket fetches sign as
/// the server, so an unset allowlist would make the server a
/// signing oracle for every bucket its identity can read.
/// With credentials present the allowlist must be explicit;
/// `*` is the deliberate any-bucket opt-in.
fn check_bucket_policy(has_credentials: bool, allowed_buckets: &[String]) -> Result<(), String> {
    if has_credentials && allowed_buckets.is_empty() {
        return Err(
            "bucket credentials are configured but CRABCROP_ALLOWED_BUCKETS is unset: \
             refusing to boot, because signed bucket fetches would reach every bucket \
             the credentials can read. Set CRABCROP_ALLOWED_BUCKETS to a comma-separated \
             list, or to * to deliberately allow all buckets."
                .to_string(),
        );
    }
    Ok(())
}

fn build_source_config() -> Result<SourceConfig, String> {
    let mut sources = SourceConfig::default();
    let nonempty = |name: &str| env::var(name).ok().filter(|v| !v.is_empty());

    if let Some(endpoint) = nonempty("CRABCROP_S3_ENDPOINT") {
        let parsed = reqwest::Url::parse(&endpoint)
            .map_err(|e| format!("invalid CRABCROP_S3_ENDPOINT: {e}"))?;
        if parsed.scheme() != "http" && parsed.scheme() != "https" {
            return Err("CRABCROP_S3_ENDPOINT must be an http(s) URL".to_string());
        }
        sources.s3_endpoint = Some(endpoint.trim_end_matches('/').to_string());
    }
    if let Some(region) = nonempty("CRABCROP_S3_REGION") {
        sources.s3_region = region;
    }
    let cred_pair =
        |key_name: &str, secret_name: &str| -> Result<Option<crate::sigv4::Credentials>, String> {
            match (nonempty(key_name), nonempty(secret_name)) {
                (Some(access_key), Some(secret_key)) => Ok(Some(crate::sigv4::Credentials {
                    access_key,
                    secret_key,
                    session_token: None,
                })),
                (None, None) => Ok(None),
                _ => Err(format!("{key_name} and {secret_name} must be set together")),
            }
        };
    sources.aws_creds = cred_pair("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")?;
    if let Some(creds) = &mut sources.aws_creds {
        creds.session_token = nonempty("AWS_SESSION_TOKEN");
    }
    if let Some(endpoint) = nonempty("CRABCROP_GCS_ENDPOINT") {
        reqwest::Url::parse(&endpoint)
            .map_err(|e| format!("invalid CRABCROP_GCS_ENDPOINT: {e}"))?;
        sources.gcs_endpoint = endpoint.trim_end_matches('/').to_string();
    }
    if let Some(region) = nonempty("CRABCROP_GCS_REGION") {
        sources.gcs_region = region;
    }
    sources.gcs_creds = cred_pair("CRABCROP_GCS_ACCESS_KEY", "CRABCROP_GCS_SECRET_KEY")?;
    sources.allowed_buckets = env::var("CRABCROP_ALLOWED_BUCKETS")
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    check_bucket_policy(
        sources.aws_creds.is_some() || sources.gcs_creds.is_some(),
        &sources.allowed_buckets,
    )?;
    if let Some(root) = nonempty("CRABCROP_FILE_ROOT") {
        let canonical = std::fs::canonicalize(&root)
            .map_err(|e| format!("CRABCROP_FILE_ROOT is not usable: {e}"))?;
        if !canonical.is_dir() {
            return Err("CRABCROP_FILE_ROOT is not a directory".to_string());
        }
        sources.file_root = Some(canonical);
    }
    Ok(sources)
}

/// Read an environment variable as `T`, or fall back to `default`
/// when it is unset. A set-but-invalid value is a startup error
/// naming the variable: silently serving on the wrong port or
/// with the wrong cache budget is worse than refusing to start.
/// Parse a `CRABCROP_JXL_SPEED` value into a jixel speed tier.
/// Default quantization step for width client hints.
const DEFAULT_WIDTH_STEP: u32 = 100;

fn parse_jxl_speed(value: &str) -> Result<jixel::Speed, String> {
    match value.to_ascii_lowercase().as_str() {
        "ultrafast" => Ok(jixel::Speed::UltraFast),
        "fastest" => Ok(jixel::Speed::Fastest),
        "fast" => Ok(jixel::Speed::Fast),
        "medium" => Ok(jixel::Speed::Medium),
        "slow" => Ok(jixel::Speed::Slow),
        "extraslow" => Ok(jixel::Speed::ExtraSlow),
        _ => Err(format!(
            "invalid CRABCROP_JXL_SPEED: {value:?} (expected ultrafast, fastest, fast, medium, slow, or extraslow)"
        )),
    }
}

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

/// The minimum `CRABCROP_SECRET` length, in bytes. Shorter
/// secrets are brute-forceable, and an empty one is worse:
/// empty-key HMAC is publicly computable, so every signed URL
/// (and purge) is forgeable (security audit PL-3, P-6).
const MIN_SECRET_BYTES: usize = 16;

fn validate_secret(secret: &[u8]) -> Result<(), String> {
    if secret.len() < MIN_SECRET_BYTES {
        return Err(format!(
            "CRABCROP_SECRET must be at least {MIN_SECRET_BYTES} bytes, got {}",
            secret.len()
        ));
    }
    Ok(())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let secret = env::var("CRABCROP_SECRET")
        .map_err(|_| "CRABCROP_SECRET must be set")?
        .into_bytes();
    validate_secret(&secret)?;
    let allow_unsafe = env::var("CRABCROP_ALLOW_UNSAFE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let port: u16 = parse_env("PORT", 3000)?;
    let ssim_target: f64 = parse_env("CRABCROP_SSIM_TARGET", crate::quality::DEFAULT_SSIM_TARGET)?;
    if !ssim_target.is_finite() || !(0.0..=100.0).contains(&ssim_target) {
        return Err(
            format!("CRABCROP_SSIM_TARGET must be between 0 and 100, got {ssim_target}").into(),
        );
    }
    let avif_speed: u8 = parse_env("CRABCROP_AVIF_SPEED", crate::engine::DEFAULT_AVIF_SPEED)?;
    if !(1..=10).contains(&avif_speed) {
        return Err(
            format!("CRABCROP_AVIF_SPEED must be between 1 and 10, got {avif_speed}").into(),
        );
    }
    let jxl_speed = match env::var("CRABCROP_JXL_SPEED") {
        Ok(v) => parse_jxl_speed(&v)?,
        Err(_) => crate::engine::DEFAULT_JXL_SPEED,
    };
    let width_step: u32 = parse_env("CRABCROP_WIDTH_STEP", DEFAULT_WIDTH_STEP)?;
    if !(1..=2000).contains(&width_step) {
        return Err(
            format!("CRABCROP_WIDTH_STEP must be between 1 and 2000, got {width_step}").into(),
        );
    }
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
    let sources = build_source_config()?;
    let (presets, only_presets) = build_presets()?;
    let preserve_metadata = metadata_config(|name| env::var(name).ok())?;

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
    let tagger = build_tagger(ort_dylib.as_deref())?;
    let detect = build_detector(ort_dylib.as_deref())?;
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
            .with_policy(policy)
            .with_sources(sources),
        engine: Engine,
        metrics: Arc::new(Metrics::default()),
        cache: Arc::new(Cache::new(cache_bytes)),
        removebg,
        upscale,
        face,
        tagger,
        detect,
        font,
        fallback,
        store,
        ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        process_gate: Arc::new(tokio::sync::Semaphore::new(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        )),
        ssim_target,
        avif_speed,
        jxl_speed,
        width_step,
        preserve_metadata,
        presets,
        only_presets,
    };
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    println!("crabcrop listening on port {port}");
    axum::serve(listener, app(state)).await?;
    Ok(())
}

/// The CLIP annotation configuration from the environment:
/// all four paths together, or none of them.
#[derive(Debug)]
struct ClipConfig {
    image_model: String,
    text_model: String,
    tokenizer: String,
    labels: String,
    threshold: f32,
}

/// Resolve the `CRABCROP_CLIP_*` set through `get` (a name
/// to value lookup, so tests can drive it without the
/// process environment). A partial set, or an out-of-range
/// threshold, is an error.
fn clip_config(get: impl Fn(&str) -> Option<String>) -> Result<Option<ClipConfig>, String> {
    const NAMES: [&str; 4] = [
        "CRABCROP_CLIP_IMAGE_MODEL",
        "CRABCROP_CLIP_TEXT_MODEL",
        "CRABCROP_CLIP_TOKENIZER",
        "CRABCROP_CLIP_LABELS",
    ];
    let values: Vec<Option<String>> = NAMES
        .iter()
        .map(|name| get(name).filter(|v| !v.is_empty()))
        .collect();
    if values.iter().all(|v| v.is_none()) {
        return Ok(None);
    }
    let missing: Vec<&str> = NAMES
        .iter()
        .zip(&values)
        .filter(|(_, v)| v.is_none())
        .map(|(name, _)| *name)
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "CLIP annotation is partially configured: missing {}",
            missing.join(", ")
        ));
    }
    let threshold = match get("CRABCROP_CLIP_THRESHOLD").filter(|v| !v.is_empty()) {
        Some(raw) => {
            let parsed: f32 = raw
                .parse()
                .map_err(|_| format!("CRABCROP_CLIP_THRESHOLD is not a number: {raw:?}"))?;
            if !(0.0..=1.0).contains(&parsed) {
                return Err(format!(
                    "CRABCROP_CLIP_THRESHOLD must be within 0..=1, got {parsed}"
                ));
            }
            parsed
        }
        None => 0.05,
    };
    let mut values = values.into_iter();
    Ok(Some(ClipConfig {
        image_model: values.next().unwrap().unwrap(),
        text_model: values.next().unwrap().unwrap(),
        tokenizer: values.next().unwrap().unwrap(),
        labels: values.next().unwrap().unwrap(),
        threshold,
    }))
}

/// Build the detection provider from the environment,
/// when `CRABCROP_DETECT_MODEL` is set: preloaded like
/// every other model, broken configuration fatal. The
/// label file is only a fallback for exports that carry
/// no class-names metadata.
/// The metadata policy from `CRABCROP_METADATA`: `strip`
/// (the default, and the behavior when unset) or `preserve`.
/// Anything else is a startup error naming the value.
fn metadata_config(get: impl Fn(&str) -> Option<String>) -> Result<bool, String> {
    match get("CRABCROP_METADATA") {
        None => Ok(false),
        Some(v) if v.eq_ignore_ascii_case("strip") => Ok(false),
        Some(v) if v.eq_ignore_ascii_case("preserve") => Ok(true),
        Some(v) => Err(format!(
            "CRABCROP_METADATA must be \"strip\" or \"preserve\", got {v:?}"
        )),
    }
}

fn build_detector(
    ort_dylib: Option<&str>,
) -> Result<Option<Arc<dyn crate::ai::DetectionProvider>>, String> {
    let path = match env::var("CRABCROP_DETECT_MODEL") {
        Ok(path) if !path.is_empty() => path,
        _ => return Ok(None),
    };
    let labels = env::var("CRABCROP_DETECT_LABELS")
        .ok()
        .filter(|v| !v.is_empty());
    let threshold = match env::var("CRABCROP_DETECT_THRESHOLD") {
        Ok(raw) if !raw.is_empty() => {
            let parsed: f32 = raw
                .parse()
                .map_err(|_| format!("CRABCROP_DETECT_THRESHOLD is not a number: {raw:?}"))?;
            if !(0.0..=1.0).contains(&parsed) {
                return Err(format!(
                    "CRABCROP_DETECT_THRESHOLD must be within 0..=1, got {parsed}"
                ));
            }
            parsed
        }
        _ => 0.5,
    };
    let provider =
        crate::ai::OnnxDetectionProvider::load(ort_dylib, &path, labels.as_deref(), threshold)
            .map_err(|e| format!("cannot load the detection model: {e}"))?;
    println!(
        "detection model loaded from {path} ({} classes, threshold {threshold})",
        crate::ai::DetectionProvider::classes(&provider).len()
    );
    Ok(Some(Arc::new(provider)))
}

/// Build the CLIP tag provider from the environment, when
/// configured: preloaded like every other model, with a
/// broken configuration failing startup.
fn build_tagger(
    ort_dylib: Option<&str>,
) -> Result<Option<Arc<dyn crate::ai::TagProvider>>, String> {
    let config = clip_config(|name| env::var(name).ok())?;
    match config {
        None => Ok(None),
        Some(config) => {
            let provider = crate::ai::OnnxClipProvider::load(
                ort_dylib,
                &config.image_model,
                &config.text_model,
                &config.tokenizer,
                &config.labels,
                config.threshold,
            )
            .map_err(|e| format!("cannot load the CLIP tagging models: {e}"))?;
            println!(
                "CLIP tagging loaded ({} labels, threshold {})",
                provider.label_count(),
                config.threshold
            );
            Ok(Some(Arc::new(provider)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_config_defaults_to_strip() {
        let none = |_: &str| None;
        assert_eq!(metadata_config(none), Ok(false));
        let preserve = |name: &str| (name == "CRABCROP_METADATA").then(|| "preserve".to_string());
        assert_eq!(metadata_config(preserve), Ok(true));
        let strip = |name: &str| (name == "CRABCROP_METADATA").then(|| "STRIP".to_string());
        assert_eq!(metadata_config(strip), Ok(false));
        let bogus = |name: &str| (name == "CRABCROP_METADATA").then(|| "keep".to_string());
        assert!(metadata_config(bogus).is_err());
    }

    #[test]
    fn clip_config_needs_all_four_paths_or_none() {
        let get = |pairs: &[(&str, &str)]| {
            let map: std::collections::HashMap<String, String> = pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            move |name: &str| map.get(name).cloned()
        };
        // None configured: off.
        assert!(clip_config(get(&[])).unwrap().is_none());
        // Partial: an error naming what is missing.
        let err = clip_config(get(&[("CRABCROP_CLIP_IMAGE_MODEL", "a.onnx")])).unwrap_err();
        assert!(err.contains("CRABCROP_CLIP_TEXT_MODEL"), "{err}");
        assert!(err.contains("CRABCROP_CLIP_LABELS"), "{err}");
        // Full set: defaults and explicit threshold.
        let full = clip_config(get(&[
            ("CRABCROP_CLIP_IMAGE_MODEL", "a.onnx"),
            ("CRABCROP_CLIP_TEXT_MODEL", "b.onnx"),
            ("CRABCROP_CLIP_TOKENIZER", "t.json"),
            ("CRABCROP_CLIP_LABELS", "labels.txt"),
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(full.threshold, 0.05);
        let bad = clip_config(get(&[
            ("CRABCROP_CLIP_IMAGE_MODEL", "a.onnx"),
            ("CRABCROP_CLIP_TEXT_MODEL", "b.onnx"),
            ("CRABCROP_CLIP_TOKENIZER", "t.json"),
            ("CRABCROP_CLIP_LABELS", "labels.txt"),
            ("CRABCROP_CLIP_THRESHOLD", "2.5"),
        ]));
        assert!(bad.is_err());
    }

    #[test]
    fn bucket_policy_fails_closed_with_credentials() {
        assert!(check_bucket_policy(false, &[]).is_ok());
        assert!(check_bucket_policy(false, &["photos".to_string()]).is_ok());
        assert!(check_bucket_policy(true, &["photos".to_string()]).is_ok());
        assert!(check_bucket_policy(true, &["*".to_string()]).is_ok());
        let err = check_bucket_policy(true, &[]).unwrap_err();
        assert!(err.contains("CRABCROP_ALLOWED_BUCKETS"), "{err}");
    }

    #[test]
    fn secrets_below_the_minimum_are_rejected() {
        assert!(validate_secret(b"").is_err());
        assert!(validate_secret(b"fifteen-bytes!!").is_err());
        assert!(validate_secret(b"sixteen-bytes!!!").is_ok());
        assert!(validate_secret(b"a much longer production secret").is_ok());
    }

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

    #[test]
    fn parses_preset_files() {
        let map = parse_presets(
            "# cards\nog-card = crop:1200x630:smart/filters:format(webp):quality(80)\n\nthumb = 300x200\n",
        )
        .unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map["thumb"], "300x200");
        for bad in [
            "no-equals-here",
            "bad name = 300x200",
            "thumb = 300x200\nthumb = 100x100",
            "thumb = preset:other",
            "thumb = example.com/img.jpg",
            "thumb = ",
        ] {
            assert!(parse_presets(bad).is_err(), "{bad}");
        }
    }
}
