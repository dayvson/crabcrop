//! Secrets stay out of `Debug` output (issue #455). The
//! library's public types that hold key material,
//! credentials, or tokens print those fields as
//! `[redacted]`, so a caller's `dbg!`, a `{:?}` in a log
//! line, or a panic message that formats one cannot leak
//! them. Each type is formatted here with real secrets
//! inside, and the output is searched for every spelling a
//! derived `Debug` would have used for them.

use std::sync::Arc;

use crabcrop::cache::{Cache, PurgeGuard};
use crabcrop::engine::Engine;
use crabcrop::loader::{Loader, SourceConfig};
use crabcrop::observe::Metrics;
use crabcrop::s3store::S3Store;
use crabcrop::server::{AppState, Gate, Inflight};
use crabcrop::sigv4::Credentials;
use crabcrop::storage::DiskStore;
use crabcrop::upload::UploadStore;
use crabcrop::url::{Keyring, entry_key};

const CURRENT_KEY: &str = "current-signing-key-0123456789";
const PREVIOUS_KEY: &str = "previous-signing-key-9876543210";
const AWS_SECRET: &str = "aws-secret-key-wJalrXUtnFEMI/K7MDENG";
const AWS_SESSION: &str = "aws-session-token-FwoGZXIvYXdzEJr";
const GCS_SECRET: &str = "gcs-hmac-secret-bGoa+V7g/yqDXvKRqq";
const METRICS_TOKEN: &str = "metrics-bearer-token-5f1c0e9a";

/// Every secret the fixtures below carry, as bytes: the
/// configured ones, plus the entry MAC key the stores
/// derive from the signing key.
fn secrets() -> Vec<Vec<u8>> {
    let mut all: Vec<Vec<u8>> = [
        CURRENT_KEY,
        PREVIOUS_KEY,
        AWS_SECRET,
        AWS_SESSION,
        GCS_SECRET,
        METRICS_TOKEN,
    ]
    .iter()
    .map(|s| s.as_bytes().to_vec())
    .collect();
    all.push(entry_key(CURRENT_KEY.as_bytes()));
    all
}

/// The first secret `rendered` carries, if any: as text, as
/// the decimal byte list a derived `Debug` prints for a
/// `Vec<u8>`, or as hex. The byte list is matched with
/// whitespace ignored, because `{:#?}` (what `dbg!` prints)
/// puts each byte on its own line.
fn leak(rendered: &str) -> Option<String> {
    let compact: String = rendered.split_whitespace().collect();
    for secret in secrets() {
        if let Ok(text) = std::str::from_utf8(&secret)
            && rendered.contains(text)
        {
            return Some(format!("{text:?}"));
        }
        let decimal: String = format!("{secret:?}").split_whitespace().collect();
        let decimal = decimal.trim_start_matches('[').trim_end_matches(']');
        if compact.contains(decimal) {
            return Some(format!("the bytes {decimal}"));
        }
        let hex: String = secret.iter().map(|b| format!("{b:02x}")).collect();
        if rendered.contains(&hex) {
            return Some(format!("the hex {hex}"));
        }
    }
    None
}

/// Fail if `rendered` carries any secret, and require the
/// redaction marker, so an empty rendering cannot pass.
fn assert_redacted(what: &str, rendered: &str) {
    if let Some(leak) = leak(rendered) {
        panic!("{what} leaks {leak}: {rendered}");
    }
    assert!(
        rendered.contains("[redacted]"),
        "{what} shows no redaction marker: {rendered}"
    );
}

fn keyring() -> Keyring {
    Keyring {
        current_id: "2026-11".to_string(),
        current: CURRENT_KEY.as_bytes().to_vec(),
        previous: Some(("2026-10".to_string(), PREVIOUS_KEY.as_bytes().to_vec())),
    }
}

fn aws_creds() -> Credentials {
    Credentials {
        access_key: "AKIDEXAMPLE".to_string(),
        secret_key: AWS_SECRET.to_string(),
        session_token: Some(AWS_SESSION.to_string()),
    }
}

fn sources() -> SourceConfig {
    SourceConfig {
        aws_creds: Some(aws_creds()),
        gcs_creds: Some(Credentials {
            access_key: "GOOGEXAMPLE".to_string(),
            secret_key: GCS_SECRET.to_string(),
            session_token: None,
        }),
        allowed_buckets: vec!["photos".to_string()],
        ..SourceConfig::default()
    }
}

fn s3_store() -> S3Store {
    S3Store::new(
        "results".to_string(),
        "cache/".to_string(),
        Some("http://127.0.0.1:9".to_string()),
        "us-east-1".to_string(),
        Some(aws_creds()),
        entry_key(CURRENT_KEY.as_bytes()),
    )
    .unwrap()
}

fn upload_store() -> UploadStore {
    UploadStore::s3(
        "uploads".to_string(),
        String::new(),
        Some("http://127.0.0.1:9".to_string()),
        "us-east-1".to_string(),
        Some(aws_creds()),
        1024,
        false,
    )
    .unwrap()
}

fn disk_store(dir: &std::path::Path) -> DiskStore {
    DiskStore::open(dir, 1024 * 1024, entry_key(CURRENT_KEY.as_bytes())).unwrap()
}

fn loader() -> Loader {
    Loader::with_defaults().unwrap().with_sources(sources())
}

fn state(dir: &std::path::Path) -> AppState {
    AppState {
        keyring: keyring(),
        allow_unsafe: false,
        loader: loader(),
        engine: Engine,
        process_timeout: std::time::Duration::from_secs(30),
        metrics: Arc::new(Metrics::default()),
        metrics_token: Some(METRICS_TOKEN.to_string()),
        cache: Arc::new(Cache::new(1024 * 1024)),
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
        store: Some(Arc::new(disk_store(dir))),
        shared_store: Some(Arc::new(s3_store())),
        upload: Some(Arc::new(upload_store())),
        meta_enabled: true,
        ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        process_gate: Arc::new(Gate::new(4)),
        ssim_target: crabcrop::quality::DEFAULT_SSIM_TARGET,
        avif_speed: crabcrop::engine::DEFAULT_AVIF_SPEED,
        jxl_speed: crabcrop::engine::DEFAULT_JXL_SPEED,
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
        slow_gate: Arc::new(Gate::new(1)),
        inflight: Arc::new(Inflight::default()),
        purge_guard: Arc::new(PurgeGuard::new()),
        background: Default::default(),
        provenance_signer: None,
        config_fingerprint: "debug-redaction".to_string(),
    }
}

#[test]
fn debug_output_names_each_secret_holder_without_its_secrets() {
    let dir = std::env::temp_dir().join(format!("crabcrop-debug-redaction-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // The detector itself: what a derived `Debug` prints
    // for each secret, compact or pretty, is caught.
    for secret in secrets() {
        assert!(leak(&format!("{secret:?}")).is_some());
        assert!(leak(&format!("{secret:#?}")).is_some());
        if let Ok(text) = String::from_utf8(secret) {
            assert!(leak(&format!("{:#?}", Some(text))).is_some());
        }
    }

    let rendered = format!("{:?}", keyring());
    assert_redacted("Keyring", &rendered);
    // The key IDs are not secret and still print, so a
    // debug print can tell which keys a ring holds.
    assert!(
        rendered.contains("\"2026-11\"") && rendered.contains("\"2026-10\""),
        "{rendered}"
    );

    let rendered = format!("{:?}", aws_creds());
    assert_redacted("Credentials", &rendered);
    // The access key ID names the identity and travels in
    // every signed request's Authorization header; it
    // prints.
    assert!(rendered.contains("AKIDEXAMPLE"), "{rendered}");

    // The containers keep a derived Debug: the redaction
    // lives in the types they hold, so it holds wherever
    // those are nested.
    assert_redacted("SourceConfig", &format!("{:?}", sources()));
    assert_redacted("Loader", &format!("{:?}", loader()));
    assert_redacted("DiskStore", &format!("{:?}", disk_store(&dir)));
    let rendered = format!("{:?}", s3_store());
    assert_redacted("S3Store", &rendered);
    assert!(rendered.contains("\"results\""), "{rendered}");
    assert_redacted("UploadStore", &format!("{:?}", upload_store()));

    let rendered = format!("{:?}", state(&dir));
    assert_redacted("AppState", &rendered);
    assert!(
        rendered.contains("metrics_token: Some([redacted])"),
        "{rendered}"
    );
    // `dbg!` prints the pretty form, through the same impls.
    assert_redacted("AppState (pretty)", &format!("{:#?}", state(&dir)));

    let _ = std::fs::remove_dir_all(&dir);
}
