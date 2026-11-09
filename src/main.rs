//! crabcrop: on-demand image optimization.
//!
//! The `crabcrop` binary. With no arguments it reads its
//! configuration from the environment, builds the server
//! from the library's modules, and serves signed image URLs
//! of the form `/<signature>/<operations...>/<image-url>`.
//! When the first argument names a CLI verb, that verb runs
//! instead of the server (see [`cli`]). The server also
//! starts this binary as its SVG worker, with
//! `--svg-worker` (see `crabcrop::svg`).
//!
//! Boot is three steps (issue #477): [`Config::from_env`]
//! (in `config.rs`) reads and validates every setting,
//! [`open_stores`] and [`build_state`] build the stores and
//! providers from them, and `run` binds and serves. Every
//! environment variable is documented in
//! `docs/configuration.md`, the single reference for
//! configuration.

#![forbid(unsafe_code)]

// The modules live in the library target (`src/lib.rs`);
// importing them at the crate root lets every `crate::`
// path in this file and in `config.rs` resolve to them.
use crabcrop::{
    ai, cache, cli, engine, frames, loader, observe, pdf, provenance, quality, s3store, server,
    sigv4, storage, text, upload, url, video,
};

mod config;

use std::env;
use std::sync::Arc;

use crate::ai::UpscaleProvider;
use crate::cache::Cache;
use crate::config::{
    AiConfig, ClipConfig, Config, ConfigError, DetectConfig, DetectorDeclaration, DetectorKind,
    LabelsDecl, ModerationConfig, OcrConfig, PdfConfig, ProvenanceConfig, SegmentConfig,
    VideoConfig, parse_detector_declarations, with_causes,
};
use crate::engine::Engine;
use crate::loader::Loader;
use crate::observe::Metrics;
use crate::server::{AppState, app};

fn main() {
    // An SVG worker (issue #447) is this binary started by
    // the server to parse and render one document: it caps
    // itself and exits without a runtime or configuration,
    // so it branches before either exists.
    if crabcrop::svg::is_worker_invocation() {
        std::process::exit(crabcrop::svg::worker_main());
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("crabcrop: cannot start the async runtime: {e}");
            std::process::exit(1);
        }
    };
    // A server that has drained (issue #463) stops its
    // runtime by the drain's deadline: blocking work nothing
    // tracks, such as a disk write whose request went or a
    // render past the grace, gets what is left of the grace
    // and is then left behind. Dropping the runtime would
    // wait for that work without bound, past the deadline.
    if let Some(deadline) = runtime.block_on(start()) {
        runtime.shutdown_timeout(deadline.saturating_duration_since(std::time::Instant::now()));
    }
}

/// Run a CLI verb, or the server. For the server, returns
/// the deadline by which its runtime must stop.
async fn start() -> Option<std::time::Instant> {
    // CLI verbs run instead of the server when the first
    // argument names one, and only a bare `crabcrop` starts
    // the server, which reads the environment and takes no
    // arguments. Every argument goes to the CLI: past the
    // help and version flags, a first argument that names
    // no verb, a flag included, is an "unknown verb" error,
    // never a server start. `crabcrop sgin ...` and
    // `crabcrop --verison` must not silently bind the port
    // and serve (#470).
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if !argv.is_empty() {
        match cli::exec(&argv).await {
            Ok(out) => {
                if !out.is_empty() {
                    println!("{out}");
                }
            }
            Err(e) => {
                eprintln!("crabcrop: {e}");
                std::process::exit(1);
            }
        }
        return None;
    }
    // The server parses and renders every SVG in a worker
    // started from its own binary (issue #447).
    match crabcrop::svg::WorkerCommand::current_binary() {
        Ok(worker) => crabcrop::svg::isolate_in_child(worker),
        Err(e) => {
            eprintln!("crabcrop: cannot locate this binary to start SVG workers: {e}");
            std::process::exit(1);
        }
    }
    match run().await {
        Ok(deadline) => Some(deadline),
        Err(e) => {
            eprintln!("crabcrop: {e}");
            std::process::exit(1);
        }
    }
}

/// Boot from the process environment, then serve until a
/// signal stops the server. Returns the deadline by which
/// the runtime must stop.
async fn run() -> Result<std::time::Instant, Box<dyn std::error::Error>> {
    let config = Config::from_env(|name| env::var(name).ok())?;
    // The result stores open, and then the listener binds,
    // before the AI preload: a taken port or an unwritable
    // directory is a cheap failure and should not wait for
    // the model embedding passes to be paid first (bind
    // early, serve late).
    let stores = open_stores(&config)?;
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", config.port)).await?;
    // The port the listener holds, which the listening line
    // names: the configured one, or for PORT=0 the free port
    // the kernel picked (issue #536).
    let port = listener.local_addr()?.port();
    let state = build_state(&config, stores)?;
    println!(
        "source fetch: cap {} MB, timeout {}s, buffer budget {} MB",
        config.fetch_cap_mb,
        config.fetch_timeout_secs,
        config.source_buffer_bytes / (1024 * 1024)
    );
    println!("configuration fingerprint: {}", state.config_fingerprint);
    // Purges recorded in the shared bucket hold here before
    // the first request (issue #461).
    crate::server::start_purge_tombstones(&state).await;
    let background = state.background.clone();
    let signalled = shutdown_signal();
    println!("crabcrop listening on port {port}");
    // Shutdown (issue #463): at SIGTERM or Ctrl-C the
    // listener closes, idle connections close, and requests
    // in flight finish; then the background writes they and
    // earlier requests left (shared-tier write-throughs,
    // purges whose caller left) finish. One deadline, the
    // grace from the signal on, bounds both waits, and the
    // process exits at it with whatever is still running.
    // A second signal ends the drain at once.
    let shutdown_grace = std::time::Duration::from_secs(config.shutdown_grace_secs);
    let stop = tokio_util::sync::CancellationToken::new();
    let mut serve = axum::serve(listener, app(state))
        .with_graceful_shutdown(stop.clone().cancelled_owned())
        .into_future();
    tokio::select! {
        served = &mut serve => {
            served?;
            return Ok(std::time::Instant::now());
        }
        () = signalled => stop.cancel(),
    }
    let deadline = tokio::time::Instant::now() + shutdown_grace;
    let signalled_again = shutdown_signal();
    println!(
        "shutdown: draining requests and background writes for up to {}s",
        shutdown_grace.as_secs()
    );
    let drain = async {
        match tokio::time::timeout_at(deadline, serve).await {
            Ok(served) => served?,
            Err(_) => eprintln!("shutdown: requests still in flight at the deadline are cut off"),
        }
        Ok::<_, std::io::Error>(background.drain(deadline).await)
    };
    // The runtime stops by `stop_by`: the deadline after a
    // drain, at once after a second signal.
    let (left, when, stop_by) = tokio::select! {
        left = drain => (left?, "at the deadline", deadline.into_std()),
        () = signalled_again => {
            eprintln!("shutdown: a second signal ends the drain; what is still running is cut off");
            (background.running(), "at the second signal", std::time::Instant::now())
        }
    };
    if left.writes > 0 {
        eprintln!(
            "shutdown: {} background writes still running {when} are dropped",
            left.writes
        );
    }
    if left.walks > 0 {
        // A walk cut short leaves its purge's bucket
        // tombstone in force (issue #461): copies it covers
        // stay misses until a later purge of the target
        // finishes its walk. A purge whose tombstone the
        // bucket refused held in this process only.
        println!(
            "shutdown: stopping {} background purge walks; a purge whose tombstone reached the bucket stays in force",
            left.walks
        );
    }
    println!("shutdown: done");
    Ok(stop_by)
}

/// Permits on the AI gate: a render whose operations run a
/// model (issue #379) and a meta annotation each hold one,
/// so at most two inferences run at once and a burst of
/// them cannot starve the transform pipeline. The video
/// gate mirrors it (`video::GATE_PERMITS`).
const AI_GATE_PERMITS: usize = 2;

/// Permits on the animation gate (issue #231): animated
/// WebP encodes, among the slowest work the engine does,
/// run one at a time, on top of the slow lane (issue #324).
const ANIM_GATE_PERMITS: usize = 1;

/// The result stores a configuration names, opened (see
/// [`open_stores`]) for [`build_state`].
struct Stores {
    /// Disk result storage (`CRABCROP_RESULT_STORAGE_DIR`).
    disk: Option<Arc<crate::storage::DiskStore>>,
    /// Shared result storage (`CRABCROP_RESULT_BUCKET`).
    shared: Option<Arc<crate::s3store::S3Store>>,
}

/// Open the result stores a validated configuration names,
/// each printing its line. `run` opens them before it binds
/// the listener, as boot always has, and before
/// [`build_state`] loads any model: a store that does not
/// open is a cheap failure, reported ahead of a taken port.
fn open_stores(config: &Config) -> Result<Stores, ConfigError> {
    // The entry MAC key for the persistent tiers (issue
    // #190): derived from the current signing secret, so
    // entries planted by anyone without it read as misses,
    // and a secret rotation invalidates the old entries
    // rather than serving content authenticated under a
    // retired key.
    let entry_key = crate::url::entry_key(&config.keyring.current);
    // Disk result storage, bounded by its own budget.
    // Opening indexes whatever a previous run left behind.
    let disk = match &config.result_storage {
        Some(storage) => {
            let store = crate::storage::DiskStore::open(
                std::path::Path::new(&storage.dir),
                storage.max_bytes,
                entry_key.clone(),
            )
            .map_err(|e| ConfigError::build(Some("CRABCROP_RESULT_STORAGE_DIR"), e))?;
            let (entries, bytes) = store.stats();
            println!(
                "result storage at {}: {entries} entries, {bytes} bytes, budget {} MB",
                storage.dir, storage.max_mb
            );
            Some(Arc::new(store))
        }
        None => None,
    };
    // Shared result storage (issue #142): the bucket tier
    // reuses the source S3 configuration (endpoint, region,
    // and the AWS credentials from the environment).
    let shared = match &config.result_bucket {
        Some(shared) => {
            let store = crate::s3store::S3Store::new(
                shared.bucket.clone(),
                shared.prefix.clone(),
                config.sources.s3_endpoint.clone(),
                config.sources.s3_region.clone(),
                config.sources.aws_creds.clone(),
                entry_key,
            )
            .map_err(|e| {
                ConfigError::build(
                    Some("CRABCROP_RESULT_BUCKET"),
                    format!(
                        "cannot build the shared result store client: {}",
                        with_causes(&e)
                    ),
                )
            })?;
            println!("shared result storage: bucket {}", shared.bucket);
            Some(Arc::new(store))
        }
        None => None,
    };
    Ok(Stores { disk, shared })
}

/// Build the server state from a validated configuration
/// and its opened stores: every configured provider and
/// the loader, in the order boot has always built them,
/// each printing its line when it is configured. A
/// component that is configured but does not build (a
/// model that does not load, a tool that does not run) is
/// fatal here, not a surprise for the first request that
/// needs it.
fn build_state(config: &Config, stores: Stores) -> Result<AppState, ConfigError> {
    // AI models are configured, never downloaded, and preloaded
    // at boot: a model that is set but broken is a startup
    // error, not a surprise for the first request that uses it.
    let ai = build_ai_providers(&config.ai)?;

    // The overlay font follows the same rule as the models:
    // configured by path, loaded at boot, broken is fatal.
    let font: Option<Arc<dyn crate::text::TextRenderer>> = match &config.font_path {
        Some(path) => {
            let renderer = crate::text::FontRenderer::load(path).map_err(|e| {
                ConfigError::build(
                    Some("CRABCROP_FONT_PATH"),
                    format!("cannot load the overlay font: {e}"),
                )
            })?;
            println!("overlay font loaded from {path}");
            Some(Arc::new(renderer))
        }
        None => None,
    };

    // The fallback image, the video tools, the PDF renderer,
    // and the signing identity follow the same boot rule as
    // the models: configured but broken is fatal here, not
    // at the first request that needs them.
    let fallback = build_fallback(config.fallback_image.as_deref())?;
    let video_encoder = build_video_encoder(&config.video)?;
    let frame_extractor = build_video_decoder(&config.video)?;
    let pdf_renderer = build_pdf_renderer(&config.pdf)?;
    let provenance_signer = build_provenance_signer(config.provenance.as_ref())?;
    if config.meta_enabled {
        println!("meta mode: on");
    }

    Ok(AppState {
        keyring: config.keyring.clone(),
        allow_unsafe: config.allow_unsafe,
        loader: Loader::new(
            config.fetch_cap_mb * 1024 * 1024,
            std::time::Duration::from_secs(config.fetch_timeout_secs),
        )
        .map_err(|e| ConfigError::build(None, format!("cannot build the loader: {e}")))?
        .with_policy(config.policy.clone())
        .with_sources(config.sources.clone())
        .with_source_budget(crate::loader::SourceBudget::new(
            usize::try_from(config.source_buffer_bytes).unwrap_or(usize::MAX),
        )),
        engine: Engine,
        process_timeout: std::time::Duration::from_secs(config.process_timeout_secs),
        metrics: Arc::new(Metrics::default()),
        metrics_token: config
            .metrics_token
            .as_ref()
            .map(|token| token.expose().to_string()),
        cache: Arc::new(Cache::new(config.cache_bytes)),
        removebg: ai.removebg,
        upscale: ai.upscale,
        inpaint: ai.inpaint,
        segment: ai.segment,
        face: ai.face,
        tagger: ai.tagger,
        tag_threshold: ai.tag_threshold,
        moderation: ai.moderation,
        embeddings: ai.embeddings,
        aesthetic: ai.aesthetic,
        ocr: ai.ocr,
        detect: ai.detect,
        detectors: ai.detectors,
        focal: ai.focal,
        font,
        fallback,
        store: stores.disk,
        shared_store: stores.shared,
        upload: config.upload.clone(),
        meta_enabled: config.meta_enabled,
        ai_gate: Arc::new(tokio::sync::Semaphore::new(AI_GATE_PERMITS)),
        process_gate: Arc::new(crate::server::Gate::new(config.process_permits)),
        slow_gate: Arc::new(crate::server::Gate::new(config.slow_lane_permits)),
        ssim_target: config.ssim_target,
        avif_speed: config.avif_speed,
        jxl_speed: config.jxl_speed,
        width_step: config.width_step,
        preserve_metadata: config.preserve_metadata,
        brand_color: config.brand_color,
        presets: config.presets.clone(),
        only_presets: config.only_presets,
        source_roots: config.source_roots.clone(),
        default_ops: config.default_ops.clone(),
        video_encoder,
        frame_extractor,
        pdf_renderer,
        provenance_signer,
        video_gate: Arc::new(tokio::sync::Semaphore::new(crate::video::GATE_PERMITS)),
        anim_gate: Arc::new(tokio::sync::Semaphore::new(ANIM_GATE_PERMITS)),
        inflight: Arc::new(crate::server::Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        background: crate::server::Background::default(),
        config_fingerprint: config.config_fingerprint.clone(),
    })
}

/// Resolves at the next SIGTERM or Ctrl-C. The handlers
/// are installed when this is called, not when it is first
/// polled, so a signal that arrives in between still counts;
/// a signal from before the call does not.
#[cfg(unix)]
fn shutdown_signal() -> impl std::future::Future<Output = ()> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut interrupt = signal(SignalKind::interrupt()).expect("install Ctrl-C handler");
    let mut terminate = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    async move {
        tokio::select! {
            _ = interrupt.recv() => {}
            _ = terminate.recv() => {}
        }
    }
}

/// SIGTERM or Ctrl-C, as for unix; only Ctrl-C elsewhere.
#[cfg(not(unix))]
fn shutdown_signal() -> impl std::future::Future<Output = ()> {
    async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl-C handler");
    }
}

/// The rendering version: bumped deliberately in any
/// release that changes rendered output for the same
/// inputs under the same configuration, so an upgraded
/// server starts a fresh cache epoch instead of serving
/// the previous build's renders from the persistent
/// tiers. Configuration alone cannot express that
/// change, so it travels as its own fingerprint
/// component. `scripts/check_rendering_version.py`
/// fails the gate when an upgrade corpus golden was
/// re-baselined since the last release tag and this
/// value did not move (issue #412).
///
/// 2: the first release after 0.9.10, which changes
/// output for existing URLs (sharpen edge replication
/// #378, keep_profile converting to sRGB for formats
/// that cannot carry a profile #374, animation kept
/// under format negotiation #370, one smart crop
/// window per animated render #376, detection memos
/// dropped after geometric steps #373).
///
/// 3: the first release after 0.9.11, which changes output
/// for existing URLs (interlaced PNG output from a 16-bit
/// source writes its samples big-endian instead of
/// byte-swapped #464; focal forms carried through the
/// geometry between their segment and the crop, and one
/// trim window for every frame of an animation #465; a
/// step that works out a size past 16383 px on an axis,
/// such as `16383x1/0x2`, refused instead of rendered
/// #521).
///
/// 4: the first release after 0.10.0, which changes output
/// for existing URLs (an output left past 16383 px on an
/// axis, such as a 16384 px source with no sizing step,
/// refused instead of rendered, a 16383 px video side
/// cropped to 16382 instead of padded to 16384, and PDF
/// pages near the raster cap rendered one DPI lower #528; a
/// `DPR` or `Sec-CH-DPR` client hint past 1 that took a
/// 16384 px source side through unchanged steps down to a
/// factor whose output fits 16383 px, so `8192x1` on a
/// 16384x2 source with `DPR: 2` serves 12288x2 instead of
/// 16384x2 #529; JPEG XL output written by the jixel 0.5.5
/// encoder, whose files run 11 to 19 percent smaller at
/// slightly lower fidelity than 0.5.2 at the same quality
/// setting, PR #493).
const RENDERING_VERSION: u32 = 4;

/// Fingerprint the configuration that shapes rendered
/// output (audit R2-08): the rendering version, the
/// metadata policy, the quality tuning knobs, the
/// width-hint step, the AI behavior thresholds, and the
/// file-based inputs whose contents change results (the
/// AI models, the CLIP set, the overlay font, the video
/// and PDF tools, the detector declarations and the
/// models they name), plus the brand color, whose value
/// shapes studio composites. A file input is identified
/// by a digest of its contents, never by its path alone:
/// replacing a model or a font in place at the same path
/// must start a fresh epoch, and the same contents under
/// a new path must not. The fingerprint is embedded in
/// every cache key (see `serve`), so a restart under
/// changed settings starts a fresh cache epoch instead
/// of serving the old settings' renders from the
/// persistent tiers. Settings that are already visible
/// in the key (presets, default ops: both change the
/// canonical form) are deliberately excluded, and so is
/// the fallback image, whose responses are never cached.
///
/// Resolved through `get` (a name to value lookup, so
/// tests can drive it without the process environment).
/// An unreadable file contributes a marker in place of
/// its digest: boot fails on it later, in the matching
/// builder, with the precise error. The one file never
/// read here is the C2PA signing key: it is named, like
/// any other setting, because its identity is bound to
/// the certificate chain, which is hashed by content (a
/// rotated key pairs with a rotated certificate, and a
/// mismatched pair fails the signer build).
fn config_fingerprint(
    get: impl Fn(&str) -> Option<String>,
    preserve_metadata: bool,
    ssim_target: f64,
    avif_speed: u8,
    jxl_speed: jixel::Speed,
    width_step: u32,
) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    let mut part = |label: &str, value: &str| {
        hasher.update(label.as_bytes());
        hasher.update(b"=");
        hasher.update(value.as_bytes());
        hasher.update(b"\n");
    };
    part("rendering_version", &RENDERING_VERSION.to_string());
    part(
        "preserve_metadata",
        if preserve_metadata { "1" } else { "0" },
    );
    part("ssim_target", &ssim_target.to_string());
    part("avif_speed", &avif_speed.to_string());
    part("jxl_speed", &format!("{jxl_speed:?}"));
    part("width_step", &width_step.to_string());
    // The AI behavior thresholds, at their effective
    // values (the builders' defaults when unset): both
    // gate what detection and tagging report, and
    // detection drives redaction in rendered output.
    part(
        "detect_threshold",
        &effective_threshold(&get, "CRABCROP_DETECT_THRESHOLD", 0.5),
    );
    part(
        "clip_threshold",
        &effective_threshold(&get, "CRABCROP_CLIP_THRESHOLD", 0.05),
    );
    // Settings whose identity is the configured string
    // itself: the brand color, the OCR language, the
    // timestamp authority, and the C2PA key, which is
    // named but never read (see the note above).
    for name in [
        "CRABCROP_BRAND_COLOR",
        "CRABCROP_OCR_LANGUAGE",
        "CRABCROP_C2PA_TSA_URL",
        "CRABCROP_C2PA_KEY",
    ] {
        if let Some(value) = get(name).filter(|v| !v.is_empty()) {
            part(name, &value);
        }
    }
    // File-based inputs, each identified by the digest
    // of its contents.
    for name in [
        "CRABCROP_REMOVEBG_MODEL",
        "CRABCROP_UPSCALE_MODEL",
        "CRABCROP_INPAINT_MODEL",
        "CRABCROP_SEGMENT_ENCODER_MODEL",
        "CRABCROP_SEGMENT_DECODER_MODEL",
        "CRABCROP_FACE_MODEL",
        "CRABCROP_DETECT_MODEL",
        "CRABCROP_DETECT_LABELS",
        "CRABCROP_CLIP_IMAGE_MODEL",
        "CRABCROP_CLIP_TEXT_MODEL",
        "CRABCROP_CLIP_TOKENIZER",
        "CRABCROP_CLIP_LABELS",
        "CRABCROP_MODERATION_MODEL",
        "CRABCROP_MODERATION_LABELS",
        "CRABCROP_AESTHETIC_HEAD",
        "CRABCROP_OCR_DETECT_MODEL",
        "CRABCROP_OCR_RECOGNIZE_MODEL",
        "CRABCROP_OCR_DICT",
        "CRABCROP_FONT_PATH",
        "CRABCROP_VIDEO_ENCODER",
        "CRABCROP_VIDEO_DECODER",
        "CRABCROP_PDF_RENDERER",
        "CRABCROP_C2PA_CERT",
    ] {
        if let Some(path) = get(name).filter(|v| !v.is_empty()) {
            part(name, &content_identity(&path));
        }
    }
    if let Some(path) = get("CRABCROP_DETECTORS_FILE").filter(|v| !v.is_empty()) {
        // The declarations file is small and is read
        // whole at boot anyway (it is capped), so its
        // digest comes from the same read that names the
        // further files to identify: each declared
        // detector's model and labels file, whose
        // contents shape output exactly like the models
        // configured directly above. An unparseable file
        // contributes its own digest only: boot fails on
        // it later, in the detector build, with the
        // precise error.
        match std::fs::read(&path) {
            Ok(bytes) => {
                part(
                    "CRABCROP_DETECTORS_FILE",
                    &sha2::Sha256::digest(&bytes)
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>(),
                );
                if let Ok(text) = std::str::from_utf8(&bytes)
                    && let Ok(decls) = parse_detector_declarations(text)
                {
                    for decl in &decls {
                        part(
                            &format!("CRABCROP_DETECTORS_FILE:{}:model", decl.name),
                            &content_identity(&decl.model),
                        );
                        if let Some(LabelsDecl::Path(labels)) = &decl.labels {
                            part(
                                &format!("CRABCROP_DETECTORS_FILE:{}:labels", decl.name),
                                &content_identity(labels),
                            );
                        }
                    }
                }
            }
            Err(_) => part("CRABCROP_DETECTORS_FILE", &format!("unreadable:{path}")),
        }
    }
    let digest = hasher.finalize();
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// The fingerprint identity of one file-based input: the
/// hex digest of its contents, streamed through the
/// hasher so a model file is never held in memory just
/// to be identified. An unreadable file degrades to a
/// marker carrying its path (see `config_fingerprint`).
fn content_identity(path: &str) -> String {
    use sha2::Digest;
    let digest = (|| {
        let mut file = std::fs::File::open(path).ok()?;
        let mut hasher = sha2::Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = std::io::Read::read(&mut file, &mut buffer).ok()?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Some(
            hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
        )
    })();
    digest.unwrap_or_else(|| format!("unreadable:{path}"))
}

/// The effective value of an AI behavior threshold for
/// the fingerprint: the parsed value when one is
/// configured, the builder's default otherwise, formatted
/// the same either way, so setting a threshold to its
/// default changes nothing. An unparseable or
/// out-of-range value contributes its raw text: boot
/// fails on it later, in the matching builder, with the
/// precise error.
fn effective_threshold(get: &impl Fn(&str) -> Option<String>, name: &str, default: f32) -> String {
    match get(name).filter(|v| !v.is_empty()) {
        Some(raw) => match raw.parse::<f32>() {
            Ok(value) if (0.0..=1.0).contains(&value) => value.to_string(),
            _ => raw,
        },
        None => default.to_string(),
    }
}

/// The AI provider set built at boot. Every provider is
/// configured by path and preloaded; a set but unloadable
/// model is a startup error. `tag_threshold` travels with
/// the tag provider (0.05 when annotation is off) because
/// it is applied where the meta annotation is composed.
struct AiProviders {
    removebg: Option<Arc<dyn crate::ai::MaskProvider>>,
    upscale: Option<Arc<dyn crate::ai::UpscaleProvider>>,
    inpaint: Option<Arc<dyn crate::ai::InpaintProvider>>,
    segment: Option<Arc<dyn crate::ai::SegmentProvider>>,
    face: Option<Arc<dyn crate::ai::FaceProvider>>,
    tagger: Option<Arc<dyn crate::ai::TagProvider>>,
    tag_threshold: f32,
    moderation: Option<Arc<dyn crate::ai::ModerationProvider>>,
    embeddings: Option<Arc<dyn crate::ai::EmbeddingProvider>>,
    aesthetic: Option<Arc<crate::ai::AestheticHead>>,
    ocr: Option<Arc<dyn crate::ai::OcrProvider>>,
    detect: Option<Arc<dyn crate::ai::DetectionProvider>>,
    detectors: Vec<server::NamedDetector>,
    focal: Option<Arc<dyn crate::ai::FocalProvider>>,
}

/// Load every configured AI provider. `Config::from_env`
/// has checked the settings, so what can fail here is a
/// file that does not load.
fn build_ai_providers(config: &AiConfig) -> Result<AiProviders, ConfigError> {
    let ort_dylib = config.ort_dylib.as_deref();
    let removebg: Option<Arc<dyn crate::ai::MaskProvider>> = match &config.removebg_model {
        Some(path) => {
            let provider = crate::ai::OnnxMaskProvider::load(ort_dylib, path).map_err(|e| {
                ConfigError::build(
                    Some("CRABCROP_REMOVEBG_MODEL"),
                    format!("cannot load the background-removal model: {e}"),
                )
            })?;
            println!("background removal model loaded from {path}");
            Some(Arc::new(provider))
        }
        None => None,
    };
    let upscale: Option<Arc<dyn crate::ai::UpscaleProvider>> = match &config.upscale_model {
        Some(path) => {
            let provider = crate::ai::OnnxUpscaleProvider::load(ort_dylib, path).map_err(|e| {
                ConfigError::build(
                    Some("CRABCROP_UPSCALE_MODEL"),
                    format!("cannot load the upscale model: {e}"),
                )
            })?;
            println!(
                "upscale model loaded from {path} (native factor {}x)",
                provider.factor()
            );
            Some(Arc::new(provider))
        }
        None => None,
    };
    let inpaint: Option<Arc<dyn crate::ai::InpaintProvider>> = match &config.inpaint_model {
        Some(path) => {
            let provider = crate::ai::OnnxInpaintProvider::load(ort_dylib, path).map_err(|e| {
                ConfigError::build(
                    Some("CRABCROP_INPAINT_MODEL"),
                    format!("cannot load the inpainting model: {e}"),
                )
            })?;
            println!("inpainting model loaded from {path}");
            Some(Arc::new(provider))
        }
        None => None,
    };
    let segment = build_segment(ort_dylib, config.segment.as_ref())?;
    let (tagger, tag_threshold, embeddings) = match build_tagger(ort_dylib, config.clip.as_ref())? {
        Some(setup) => (
            Some(setup.provider),
            setup.threshold,
            Some(setup.embeddings),
        ),
        None => (None, 0.05, None),
    };
    let moderation = build_moderation(ort_dylib, config.moderation.as_ref())?;
    let aesthetic = build_aesthetic(config.aesthetic_head.as_deref())?;
    let ocr = build_ocr(ort_dylib, config.ocr.as_ref())?;
    let detect = build_detector(ort_dylib, config.detect.as_ref())?;
    let declared = build_declared_detectors(ort_dylib, &config.detectors)?;
    let face: Option<Arc<dyn crate::ai::FaceProvider>> = match &config.face_model {
        Some(path) => {
            let provider = crate::ai::OnnxFaceProvider::load(ort_dylib, path).map_err(|e| {
                ConfigError::build(
                    Some("CRABCROP_FACE_MODEL"),
                    format!("cannot load the face detection model: {e}"),
                )
            })?;
            println!("face detection model loaded from {path}");
            Some(Arc::new(provider))
        }
        None => None,
    };
    Ok(AiProviders {
        removebg,
        upscale,
        inpaint,
        segment,
        face,
        tagger,
        tag_threshold,
        moderation,
        embeddings,
        aesthetic,
        ocr,
        detect,
        detectors: declared.boxes,
        focal: declared.focal,
    })
}

/// The fallback image (`CRABCROP_FALLBACK_IMAGE`): an
/// http(s) value is fetched per fallback event through the
/// loader; anything else is a local path read once now,
/// broken is fatal like the rest of the boot config. The
/// local bytes are not decoded here: decode happens per
/// fallback event, through the same operations as a normal
/// source.
fn build_fallback(value: Option<&str>) -> Result<Option<server::Fallback>, ConfigError> {
    match value {
        Some(v) => {
            if v.starts_with("http://") || v.starts_with("https://") {
                println!("fallback image will be fetched from {v}");
                Ok(Some(server::Fallback::Url(v.to_string())))
            } else {
                let bytes = std::fs::read(v).map_err(|e| {
                    ConfigError::build(
                        Some("CRABCROP_FALLBACK_IMAGE"),
                        format!("cannot read the fallback image {v}: {e}"),
                    )
                })?;
                println!("fallback image loaded from {v} ({} bytes)", bytes.len());
                Ok(Some(server::Fallback::Bytes(Arc::new(bytes))))
            }
        }
        None => Ok(None),
    }
}

/// The video encoder (issue #162): an operator-supplied
/// program, validated now, so a configured but unusable
/// path is a boot error rather than a surprise at the
/// first video request.
fn build_video_encoder(
    config: &VideoConfig,
) -> Result<Option<Arc<crate::video::VideoEncoder>>, ConfigError> {
    match &config.encoder {
        Some(path) => {
            let encoder = crate::video::VideoEncoder::new(
                path.clone(),
                std::time::Duration::from_secs(config.timeout_secs),
                config.max_bytes,
            )
            .map_err(|e| {
                ConfigError::build(
                    Some("CRABCROP_VIDEO_ENCODER"),
                    format!("cannot use CRABCROP_VIDEO_ENCODER ({path}): {e}"),
                )
            })?;
            println!(
                "video encoder: {path} (timeout {}s, output cap {} MB)",
                config.timeout_secs, config.max_mb
            );
            Ok(Some(Arc::new(encoder)))
        }
        None => Ok(None),
    }
}

/// Build the video frame extractor (issue #214) from
/// `CRABCROP_VIDEO_DECODER`: the source-side counterpart
/// of the encoder. Same boot rule: a set path that is
/// not a usable executable is fatal here, not on the
/// first `frame:` request. The timeout is the shared
/// video tool budget.
fn build_video_decoder(
    config: &VideoConfig,
) -> Result<Option<Arc<crate::frames::VideoFrameExtractor>>, ConfigError> {
    match &config.decoder {
        Some(path) => {
            let extractor = crate::frames::VideoFrameExtractor::new(
                std::path::PathBuf::from(path),
                std::time::Duration::from_secs(config.timeout_secs),
            )
            .map_err(|e| {
                ConfigError::build(
                    Some("CRABCROP_VIDEO_DECODER"),
                    format!("cannot use CRABCROP_VIDEO_DECODER ({path}): {e}"),
                )
            })?;
            println!("video decoder: {path} (timeout {}s)", config.timeout_secs);
            Ok(Some(Arc::new(extractor)))
        }
        None => Ok(None),
    }
}

/// Build the C2PA output signer (issue #173) from the
/// configured identity. The identity is proven at boot by
/// a sign-and-verify round trip (see
/// `ProvenanceSigner::from_pem`): a broken identity is
/// fatal here, not at the first signed request.
fn build_provenance_signer(
    config: Option<&ProvenanceConfig>,
) -> Result<Option<Arc<crate::provenance::ProvenanceSigner>>, ConfigError> {
    let Some(config) = config else {
        return Ok(None);
    };
    let signer = crate::provenance::ProvenanceSigner::from_files(
        &config.cert,
        &config.key,
        config.tsa.clone(),
    )
    .map_err(|e| {
        ConfigError::build(
            Some("CRABCROP_C2PA_CERT"),
            format!("cannot use the C2PA signing identity: {e}"),
        )
    })?;
    println!(
        "C2PA provenance signing: {} (certificate {})",
        signer.name, config.cert
    );
    Ok(Some(Arc::new(signer)))
}

/// Build the PDF renderer (issue #215) from
/// `CRABCROP_PDF_RENDERER`: the document-side
/// counterpart of the frame extractor. Same boot rule:
/// a set path that is not a usable renderer, or one
/// whose `pdfinfo` sibling is missing, is fatal here,
/// not on the first `page:` request.
fn build_pdf_renderer(
    config: &PdfConfig,
) -> Result<Option<Arc<crate::pdf::PdfRenderer>>, ConfigError> {
    match &config.renderer {
        Some(path) => {
            let renderer = crate::pdf::PdfRenderer::new(
                std::path::PathBuf::from(path),
                std::time::Duration::from_secs(config.timeout_secs),
            )
            .map_err(|e| {
                ConfigError::build(
                    Some("CRABCROP_PDF_RENDERER"),
                    format!("cannot use CRABCROP_PDF_RENDERER ({path}): {e}"),
                )
            })?;
            println!("PDF renderer: {path} (timeout {}s)", config.timeout_secs);
            Ok(Some(Arc::new(renderer)))
        }
        None => Ok(None),
    }
}

/// Build the segment provider (issue #175) from the
/// encoder and prompt decoder pair, preloaded like every
/// other model.
fn build_segment(
    ort_dylib: Option<&str>,
    config: Option<&SegmentConfig>,
) -> Result<Option<Arc<dyn crate::ai::SegmentProvider>>, ConfigError> {
    let Some(SegmentConfig { encoder, decoder }) = config else {
        return Ok(None);
    };
    let provider =
        crate::ai::OnnxSegmentProvider::load(ort_dylib, encoder, decoder).map_err(|e| {
            ConfigError::build(
                Some("CRABCROP_SEGMENT_ENCODER_MODEL"),
                format!("cannot load the segment model pair: {e}"),
            )
        })?;
    println!("segment models loaded from {encoder} (encoder) and {decoder} (decoder)");
    Ok(Some(Arc::new(provider)))
}

/// Build the detection provider, when
/// `CRABCROP_DETECT_MODEL` is set: preloaded like every
/// other model, broken configuration fatal. The label file
/// is only a fallback for exports that carry no
/// class-names metadata.
fn build_detector(
    ort_dylib: Option<&str>,
    config: Option<&DetectConfig>,
) -> Result<Option<Arc<dyn crate::ai::DetectionProvider>>, ConfigError> {
    let Some(config) = config else {
        return Ok(None);
    };
    let provider = crate::ai::OnnxDetectionProvider::load(
        ort_dylib,
        &config.model,
        config.labels.as_deref(),
        config.threshold,
    )
    .map_err(|e| {
        ConfigError::build(
            Some("CRABCROP_DETECT_MODEL"),
            format!("cannot load the detection model: {e}"),
        )
    })?;
    println!(
        "detection model loaded from {} ({} classes, threshold {})",
        config.model,
        crate::ai::DetectionProvider::describe(&provider).len(),
        config.threshold
    );
    Ok(Some(Arc::new(provider)))
}

/// The declared detectors, built: every `boxes` provider
/// in declaration order, and the one active focal
/// provider (the first `focal` declaration).
struct DeclaredDetectors {
    boxes: Vec<server::NamedDetector>,
    focal: Option<Arc<dyn crate::ai::FocalProvider>>,
}

/// Build the detectors `CRABCROP_DETECTORS_FILE` declares
/// (issue #166). Each declaration is loaded and probed
/// against its built-in contract here, exactly like the
/// configured models, and any failure is a startup error
/// naming the declaration. Additional `focal` declarations
/// are loaded and probed the same way, then stand by: only
/// the first feeds smart crops.
fn build_declared_detectors(
    ort_dylib: Option<&str>,
    declarations: &[DetectorDeclaration],
) -> Result<DeclaredDetectors, ConfigError> {
    let failed = |reason: String| ConfigError::build(Some("CRABCROP_DETECTORS_FILE"), reason);
    let mut boxes = Vec::new();
    let mut focal: Option<Arc<dyn crate::ai::FocalProvider>> = None;
    let mut active_focal: Option<String> = None;
    for decl in declarations {
        match decl.kind {
            DetectorKind::Boxes => {
                // Declared labels win over the model's own
                // vocabulary (issue #166), in both forms: a
                // labels path resolves to its class list
                // here, before the load, rather than taking
                // the fallback slot `CRABCROP_DETECT_LABELS`
                // uses for the built-in detector.
                let declared_classes = match &decl.labels {
                    Some(LabelsDecl::Inline(names)) => Some(names.clone()),
                    Some(LabelsDecl::Path(path)) => {
                        Some(crate::ai::read_labels(path).map_err(|e| {
                            failed(format!(
                                "cannot load declared detector {:?} (boxes) from {}: {e}",
                                decl.name, decl.model
                            ))
                        })?)
                    }
                    None => None,
                };
                // A labels file's vocabulary is held to the
                // same cap the parser puts on inline labels
                // (audit R2-27).
                let label_count = declared_classes.as_ref().map_or(0, Vec::len);
                if label_count > crate::config::MAX_DETECTOR_LABELS {
                    return Err(failed(format!(
                        "declared detector {:?}: the labels name {label_count} classes; the cap \
                         is {}",
                        decl.name,
                        crate::config::MAX_DETECTOR_LABELS
                    )));
                }
                let threshold = decl.threshold.unwrap_or(0.5);
                let provider = crate::ai::OnnxDetectionProvider::load_with_classes(
                    ort_dylib,
                    &decl.model,
                    declared_classes,
                    None,
                    threshold,
                )
                .map_err(|e| {
                    failed(format!(
                        "cannot load declared detector {:?} (boxes) from {}: {e}",
                        decl.name, decl.model
                    ))
                })?;
                println!(
                    "declared detector {} (boxes) loaded from {} ({} classes, threshold {})",
                    decl.name,
                    decl.model,
                    crate::ai::DetectionProvider::describe(&provider).len(),
                    threshold
                );
                boxes.push(server::NamedDetector {
                    name: decl.name.clone(),
                    provider: Arc::new(provider),
                });
            }
            DetectorKind::Focal => {
                let provider =
                    crate::ai::OnnxFocalProvider::load(ort_dylib, &decl.model).map_err(|e| {
                        failed(format!(
                            "cannot load declared detector {:?} (focal) from {}: {e}",
                            decl.name, decl.model
                        ))
                    })?;
                if focal.is_none() {
                    println!(
                        "declared detector {} (focal) loaded from {}",
                        decl.name, decl.model
                    );
                    focal = Some(Arc::new(provider));
                    active_focal = Some(decl.name.clone());
                } else {
                    println!(
                        "declared detector {} (focal) loaded from {} (standby: {} is the active focal provider)",
                        decl.name,
                        decl.model,
                        active_focal.as_deref().unwrap_or_default()
                    );
                }
            }
        }
    }
    Ok(DeclaredDetectors { boxes, focal })
}

/// A loaded tag provider plus the threshold its meta
/// annotations compose with, and the embedding
/// provider built from the same tagger (issue #172).
struct TaggerSetup {
    provider: Arc<dyn crate::ai::TagProvider>,
    threshold: f32,
    embeddings: Arc<dyn crate::ai::EmbeddingProvider>,
}

/// Build the CLIP tag provider, when the CLIP set is
/// configured: preloaded like every other model, with a
/// broken configuration failing startup.
fn build_tagger(
    ort_dylib: Option<&str>,
    config: Option<&ClipConfig>,
) -> Result<Option<TaggerSetup>, ConfigError> {
    let Some(config) = config else {
        return Ok(None);
    };
    let provider = crate::ai::OnnxClipProvider::load(
        ort_dylib,
        &config.image_model,
        &config.text_model,
        &config.tokenizer,
        &config.labels,
    )
    .map_err(|e| {
        ConfigError::build(
            Some("CRABCROP_CLIP_IMAGE_MODEL"),
            format!("cannot load the CLIP tagging models: {e}"),
        )
    })?;
    println!(
        "CLIP tagging loaded ({} labels, threshold {})",
        provider.label_count(),
        config.threshold
    );
    // The threshold is presentation, applied where
    // the meta annotation is composed (issue #152),
    // so it travels with the provider to the server
    // state rather than living inside the provider.
    let provider = Arc::new(provider);
    // The embedding provider wraps the same
    // tagger (issue #172): the vector the tag
    // scores are computed from, surfaced with the
    // image model's identity.
    let embeddings = Arc::new(crate::ai::ClipEmbeddingProvider::new(
        provider.clone(),
        config.image_model.clone(),
    ));
    Ok(Some(TaggerSetup {
        provider,
        threshold: config.threshold,
        embeddings,
    }))
}

/// Build the moderation provider, when configured (issue
/// #171): a verdict classifier from
/// `CRABCROP_MODERATION_MODEL`, or the CLIP zero-shot
/// fallback when only a labels file is set. Preloaded like
/// every other provider, with a broken configuration
/// failing startup.
fn build_moderation(
    ort_dylib: Option<&str>,
    config: Option<&ModerationConfig>,
) -> Result<Option<Arc<dyn crate::ai::ModerationProvider>>, ConfigError> {
    match config {
        None => Ok(None),
        Some(ModerationConfig::Model { model, labels }) => {
            // With a model, the labels file names its output
            // categories; the provider checks the count
            // against its probe at load.
            let labels = match labels {
                Some(path) => Some(crate::ai::read_labels(path).map_err(|e| {
                    ConfigError::build(
                        Some("CRABCROP_MODERATION_LABELS"),
                        format!("cannot read the moderation labels: {e}"),
                    )
                })?),
                None => None,
            };
            let provider = crate::ai::OnnxModerationProvider::load(ort_dylib, model, labels)
                .map_err(|e| {
                    ConfigError::build(
                        Some("CRABCROP_MODERATION_MODEL"),
                        format!("cannot load the moderation model: {e}"),
                    )
                })?;
            println!(
                "moderation verdicts loaded from {model} ({} categories)",
                provider.labels().len()
            );
            Ok(Some(Arc::new(provider)))
        }
        Some(ModerationConfig::ClipPrompts { labels, clip }) => {
            // The same CLIP model files load again with the
            // moderation labels, so the two label sets stay
            // independent.
            let tagger = crate::ai::OnnxClipProvider::load(
                ort_dylib,
                &clip.image_model,
                &clip.text_model,
                &clip.tokenizer,
                labels,
            )
            .map_err(|e| {
                ConfigError::build(
                    Some("CRABCROP_MODERATION_LABELS"),
                    format!("cannot load the CLIP moderation fallback: {e}"),
                )
            })?;
            println!(
                "moderation review signals loaded from {} ({} prompt labels)",
                clip.image_model,
                tagger.label_count()
            );
            let provider =
                crate::ai::ClipModerationProvider::new(Arc::new(tagger), clip.image_model.clone());
            Ok(Some(Arc::new(provider)))
        }
    }
}

/// Build the aesthetic head, when configured (issue #172):
/// an operator-supplied LAION linear head, scored against
/// the CLIP tagger's embedding in meta (`Config::from_env`
/// refuses a head without the CLIP set). A malformed head
/// file fails its load here, at boot, like every other
/// configured artifact. A dimension mismatch between the
/// head and the encoder is NOT an error: meta reports it
/// per request, fail-soft (see `AestheticHead::score`).
fn build_aesthetic(
    path: Option<&str>,
) -> Result<Option<Arc<crate::ai::AestheticHead>>, ConfigError> {
    let Some(path) = path else {
        return Ok(None);
    };
    let head = crate::ai::AestheticHead::load(path).map_err(|e| {
        ConfigError::build(
            Some("CRABCROP_AESTHETIC_HEAD"),
            format!("cannot load the aesthetic head: {e}"),
        )
    })?;
    println!(
        "aesthetic head loaded from {path} (expects a {}-dimensional embedding)",
        head.expected_dim()
    );
    Ok(Some(Arc::new(head)))
}

/// Build the OCR provider, when configured (issue #177): a
/// PP-OCR-family detection and recognition model pair plus
/// the recognition model's character dictionary, preloaded
/// and probed like every other provider.
fn build_ocr(
    ort_dylib: Option<&str>,
    config: Option<&OcrConfig>,
) -> Result<Option<Arc<dyn crate::ai::OcrProvider>>, ConfigError> {
    let Some(OcrConfig {
        detect,
        recognize,
        dict,
        language,
    }) = config
    else {
        return Ok(None);
    };
    let provider = crate::ai::OnnxOcrProvider::load(ort_dylib, detect, recognize, dict, language)
        .map_err(|e| {
        ConfigError::build(
            Some("CRABCROP_OCR_DETECT_MODEL"),
            format!("cannot load the OCR models: {e}"),
        )
    })?;
    println!(
        "OCR loaded from {detect} (detection) and {recognize} (recognition), \
         dictionary {dict} ({} characters), language {language}",
        provider.dict_len()
    );
    Ok(Some(Arc::new(provider)))
}

// The library's test helpers are not visible to this
// crate: its scratch directories come in by path, as in
// the integration tests. Not every item is used here.
#[cfg(test)]
#[allow(dead_code)]
#[path = "test_support/scratch.rs"]
mod test_support;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tests::{env_of, expect_err, missing_path, temp_file};
    use crate::config::{
        DEFAULT_WIDTH_STEP, MAX_DETECTORS_FILE_BYTES, ai_config, metadata_config, pdf_config,
        provenance_config, video_config,
    };

    // The construction tests drive each builder from an
    // environment, as boot does: the configuration step
    // first, then the build, so a refusal from either comes
    // back the way boot reports it.

    fn build_ai_providers(
        get: impl Fn(&str) -> Option<String>,
    ) -> Result<AiProviders, ConfigError> {
        super::build_ai_providers(&ai_config(get)?)
    }

    fn build_fallback(
        get: impl Fn(&str) -> Option<String>,
    ) -> Result<Option<server::Fallback>, ConfigError> {
        super::build_fallback(
            get("CRABCROP_FALLBACK_IMAGE")
                .filter(|v| !v.is_empty())
                .as_deref(),
        )
    }

    fn build_video_encoder(
        get: impl Fn(&str) -> Option<String>,
    ) -> Result<Option<Arc<crate::video::VideoEncoder>>, ConfigError> {
        super::build_video_encoder(&video_config(get)?)
    }

    fn build_video_decoder(
        get: impl Fn(&str) -> Option<String>,
    ) -> Result<Option<Arc<crate::frames::VideoFrameExtractor>>, ConfigError> {
        super::build_video_decoder(&video_config(get)?)
    }

    fn build_pdf_renderer(
        get: impl Fn(&str) -> Option<String>,
    ) -> Result<Option<Arc<crate::pdf::PdfRenderer>>, ConfigError> {
        super::build_pdf_renderer(&pdf_config(get)?)
    }

    fn build_provenance_signer(
        get: impl Fn(&str) -> Option<String>,
    ) -> Result<Option<Arc<crate::provenance::ProvenanceSigner>>, ConfigError> {
        super::build_provenance_signer(provenance_config(get)?.as_ref())
    }

    #[test]
    fn default_fingerprint_leaves_the_0_9_10_epoch() {
        // 0.9.10 printed `configuration fingerprint:
        // 7b993291c4880a4d` under the default
        // configuration. The release after it changes
        // rendered output for existing URLs, so the same
        // default configuration must print a different
        // fingerprint, or upgraded servers keep serving
        // 0.9.10 renders from the disk and shared tiers
        // (issue #412). The arguments are the boot path's
        // own defaults, not the fixed values the other
        // fingerprint tests use.
        let fingerprint = config_fingerprint(
            env_of(&[]),
            metadata_config(env_of(&[])).unwrap(),
            crate::quality::DEFAULT_SSIM_TARGET,
            crate::engine::DEFAULT_AVIF_SPEED,
            crate::engine::DEFAULT_JXL_SPEED,
            DEFAULT_WIDTH_STEP,
        );
        assert_ne!(
            fingerprint, "7b993291c4880a4d",
            "the default fingerprint still matches 0.9.10"
        );
    }

    #[test]
    fn config_fingerprint_tracks_output_shaping_settings() {
        // The fingerprint is a pure function of the
        // settings (issue #190, audit R2-08): stable for
        // identical configuration, different the moment an
        // output-shaping input changes, and blind to
        // settings that do not shape output.
        let speed = crate::engine::DEFAULT_JXL_SPEED;
        let base = config_fingerprint(env_of(&[]), false, 90.0, 6, speed, 100);
        assert_eq!(base.len(), 16);
        assert_eq!(
            base,
            config_fingerprint(env_of(&[]), false, 90.0, 6, speed, 100),
            "identical configuration, identical fingerprint"
        );
        assert_ne!(
            base,
            config_fingerprint(env_of(&[]), true, 90.0, 6, speed, 100),
            "the metadata policy shapes output"
        );
        assert_ne!(
            base,
            config_fingerprint(env_of(&[]), false, 80.0, 6, speed, 100),
            "the SSIM target shapes output"
        );
        assert_ne!(
            base,
            config_fingerprint(env_of(&[]), false, 90.0, 6, speed, 50),
            "the width step shapes output"
        );
        assert_ne!(
            base,
            config_fingerprint(
                env_of(&[("CRABCROP_REMOVEBG_MODEL", "/models/u2net.onnx")]),
                false,
                90.0,
                6,
                speed,
                100
            ),
            "a swapped model file shapes output"
        );
        assert_ne!(
            base,
            config_fingerprint(
                env_of(&[("CRABCROP_BRAND_COLOR", "f5f0e8")]),
                false,
                90.0,
                6,
                speed,
                100
            ),
            "a brand color shapes studio output"
        );
        assert_ne!(
            base,
            config_fingerprint(
                env_of(&[("CRABCROP_C2PA_CERT", "/identity/cert.pem")]),
                false,
                90.0,
                6,
                speed,
                100
            ),
            "a signing identity shapes signed output"
        );
        assert_ne!(
            base,
            config_fingerprint(
                env_of(&[("CRABCROP_C2PA_TSA_URL", "https://tsa.example.com")]),
                false,
                90.0,
                6,
                speed,
                100
            ),
            "a timestamp authority shapes signatures"
        );
        assert_eq!(
            base,
            config_fingerprint(
                env_of(&[("CRABCROP_PORT", "8080"), ("CRABCROP_CACHE_MAX_MB", "64")]),
                false,
                90.0,
                6,
                speed,
                100
            ),
            "serving knobs do not shape output"
        );
        // The detectors file is identified by content:
        // editing a declaration changes the fingerprint,
        // and an unchanged file is stable.
        let file = temp_file("fp-detectors", br#"[{"name":"a"}]"#);
        let fp_file = config_fingerprint(
            env_of(&[("CRABCROP_DETECTORS_FILE", file.as_str())]),
            false,
            90.0,
            6,
            speed,
            100,
        );
        assert_ne!(base, fp_file);
        std::fs::write(&file, br#"[{"name":"b"}]"#).unwrap();
        let fp_edited = config_fingerprint(
            env_of(&[("CRABCROP_DETECTORS_FILE", file.as_str())]),
            false,
            90.0,
            6,
            speed,
            100,
        );
        assert_ne!(fp_file, fp_edited, "edited declarations, new epoch");
        let _ = std::fs::remove_file(&file);
    }

    #[test]
    fn config_fingerprint_identifies_files_by_content() {
        // File-based inputs are identified by a digest of
        // their contents, not by their paths (issue #348):
        // replacing a file in place at the same path starts
        // a fresh epoch, and the same contents under a
        // different path do not.
        let speed = crate::engine::DEFAULT_JXL_SPEED;
        let base = config_fingerprint(env_of(&[]), false, 90.0, 6, speed, 100);
        let model = temp_file("fp-model", b"model weights, first build");
        let fp_model = config_fingerprint(
            env_of(&[("CRABCROP_REMOVEBG_MODEL", model.as_str())]),
            false,
            90.0,
            6,
            speed,
            100,
        );
        assert_ne!(base, fp_model);
        std::fs::write(&model, b"model weights, second build").unwrap();
        let fp_replaced = config_fingerprint(
            env_of(&[("CRABCROP_REMOVEBG_MODEL", model.as_str())]),
            false,
            90.0,
            6,
            speed,
            100,
        );
        assert_ne!(fp_model, fp_replaced, "replaced in place, new epoch");
        std::fs::write(&model, b"model weights, first build").unwrap();
        let moved = temp_file("fp-model-moved", b"model weights, first build");
        assert_eq!(
            fp_model,
            config_fingerprint(
                env_of(&[("CRABCROP_REMOVEBG_MODEL", moved.as_str())]),
                false,
                90.0,
                6,
                speed,
                100
            ),
            "same contents under a new path keep the epoch"
        );
        // The overlay font follows the same rule.
        let font = temp_file("fp-font", b"font bytes, first build");
        let fp_font = config_fingerprint(
            env_of(&[("CRABCROP_FONT_PATH", font.as_str())]),
            false,
            90.0,
            6,
            speed,
            100,
        );
        std::fs::write(&font, b"font bytes, second build").unwrap();
        assert_ne!(
            fp_font,
            config_fingerprint(
                env_of(&[("CRABCROP_FONT_PATH", font.as_str())]),
                false,
                90.0,
                6,
                speed,
                100
            ),
            "a replaced font starts a fresh epoch"
        );
        // A declared detector's model is identified by
        // content too: replacing it leaves the declarations
        // untouched and still starts a fresh epoch.
        let declared = temp_file("fp-declared-model", b"declared weights, first");
        let decls = temp_file(
            "fp-declared-file",
            format!(r#"[{{"name":"things","model":{declared:?},"kind":"boxes"}}]"#).as_bytes(),
        );
        let fp_declared = config_fingerprint(
            env_of(&[("CRABCROP_DETECTORS_FILE", decls.as_str())]),
            false,
            90.0,
            6,
            speed,
            100,
        );
        std::fs::write(&declared, b"declared weights, second").unwrap();
        assert_ne!(
            fp_declared,
            config_fingerprint(
                env_of(&[("CRABCROP_DETECTORS_FILE", decls.as_str())]),
                false,
                90.0,
                6,
                speed,
                100
            ),
            "a replaced declared model starts a fresh epoch"
        );
        let _ = std::fs::remove_file(&model);
        let _ = std::fs::remove_file(&moved);
        let _ = std::fs::remove_file(&font);
        let _ = std::fs::remove_file(&declared);
        let _ = std::fs::remove_file(&decls);
    }

    #[test]
    fn config_fingerprint_tracks_ai_thresholds() {
        // The AI behavior thresholds shape output at their
        // effective values (issue #348): changing one
        // starts a fresh epoch, and setting one to its
        // default changes nothing.
        let speed = crate::engine::DEFAULT_JXL_SPEED;
        let base = config_fingerprint(env_of(&[]), false, 90.0, 6, speed, 100);
        assert_ne!(
            base,
            config_fingerprint(
                env_of(&[("CRABCROP_DETECT_THRESHOLD", "0.2")]),
                false,
                90.0,
                6,
                speed,
                100
            ),
            "the detection threshold shapes redaction"
        );
        assert_eq!(
            base,
            config_fingerprint(
                env_of(&[("CRABCROP_DETECT_THRESHOLD", "0.5")]),
                false,
                90.0,
                6,
                speed,
                100
            ),
            "the default detection threshold is the unset fingerprint"
        );
        assert_ne!(
            base,
            config_fingerprint(
                env_of(&[("CRABCROP_CLIP_THRESHOLD", "0.2")]),
                false,
                90.0,
                6,
                speed,
                100
            ),
            "the tagging threshold shapes annotations"
        );
        assert_eq!(
            base,
            config_fingerprint(
                env_of(&[("CRABCROP_CLIP_THRESHOLD", "0.05")]),
                false,
                90.0,
                6,
                speed,
                100
            ),
            "the default tagging threshold is the unset fingerprint"
        );
    }

    #[test]
    fn fallback_construction_reads_local_files_and_defers_urls() {
        assert!(build_fallback(env_of(&[])).unwrap().is_none());
        // An http(s) value is kept as a URL, fetched per event.
        let fallback = build_fallback(env_of(&[(
            "CRABCROP_FALLBACK_IMAGE",
            "https://example.com/fallback.png",
        )]))
        .unwrap()
        .unwrap();
        assert!(
            matches!(fallback, server::Fallback::Url(ref u) if u == "https://example.com/fallback.png")
        );
        // A missing local file is a startup error.
        let missing = missing_path("fallback");
        let err = expect_err(build_fallback(env_of(&[(
            "CRABCROP_FALLBACK_IMAGE",
            missing.as_str(),
        )])));
        assert!(err.contains("cannot read the fallback image"), "{err}");
        // A readable file is loaded once, bytes as-is: boot
        // does not decode, so undecodable bytes construct
        // fine here. Decode happens per fallback event, and
        // a failure there falls back to the original error.
        let path = temp_file("fallback", b"these are not image bytes");
        let fallback = build_fallback(env_of(&[("CRABCROP_FALLBACK_IMAGE", path.as_str())]))
            .unwrap()
            .unwrap();
        match fallback {
            server::Fallback::Bytes(bytes) => {
                assert_eq!(&bytes[..], b"these are not image bytes")
            }
            server::Fallback::Url(_) => panic!("a local path must load as bytes"),
        }
    }

    #[test]
    fn video_construction_checks_the_bounds_even_without_an_encoder() {
        // Unconfigured and defaulted: off, no error.
        assert!(build_video_encoder(env_of(&[])).unwrap().is_none());
        // The bounds parse unconditionally: out-of-range or
        // unparseable values are boot errors even with no
        // encoder configured (issue #162).
        for (name, value) in [
            ("CRABCROP_VIDEO_TIMEOUT_SECS", "0"),
            ("CRABCROP_VIDEO_TIMEOUT_SECS", "601"),
            ("CRABCROP_VIDEO_TIMEOUT_SECS", "soon"),
            ("CRABCROP_VIDEO_MAX_MB", "0"),
            ("CRABCROP_VIDEO_MAX_MB", "1025"),
            ("CRABCROP_VIDEO_MAX_MB", "lots"),
        ] {
            let err = expect_err(build_video_encoder(env_of(&[(name, value)])));
            assert!(err.contains(name), "{name}={value}: {err}");
        }
        // The range edges are accepted.
        assert!(
            build_video_encoder(env_of(&[
                ("CRABCROP_VIDEO_TIMEOUT_SECS", "600"),
                ("CRABCROP_VIDEO_MAX_MB", "1"),
            ]))
            .unwrap()
            .is_none()
        );
    }

    #[cfg(unix)]
    #[test]
    fn video_construction_validates_the_encoder_program() {
        // A missing program is a boot error naming the variable.
        let missing = missing_path("encoder");
        let err = expect_err(build_video_encoder(env_of(&[(
            "CRABCROP_VIDEO_ENCODER",
            missing.as_str(),
        )])));
        assert!(err.contains("cannot use CRABCROP_VIDEO_ENCODER"), "{err}");
        assert!(err.contains("is not usable"), "{err}");
        // A directory is not a program.
        let dir = std::env::temp_dir().to_string_lossy().into_owned();
        let err = expect_err(build_video_encoder(env_of(&[(
            "CRABCROP_VIDEO_ENCODER",
            dir.as_str(),
        )])));
        assert!(err.contains("is not a file"), "{err}");
        // A non-executable file is refused...
        let plain = temp_file("encoder-plain", b"#!/bin/sh\n");
        let err = expect_err(build_video_encoder(env_of(&[(
            "CRABCROP_VIDEO_ENCODER",
            plain.as_str(),
        )])));
        assert!(err.contains("is not executable"), "{err}");
        // ...and the same file with the executable bit constructs.
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&plain, std::fs::Permissions::from_mode(0o755)).unwrap();
        let encoder = build_video_encoder(env_of(&[
            ("CRABCROP_VIDEO_ENCODER", plain.as_str()),
            ("CRABCROP_VIDEO_TIMEOUT_SECS", "45"),
            ("CRABCROP_VIDEO_MAX_MB", "2"),
        ]))
        .unwrap();
        assert!(encoder.is_some());
    }

    #[test]
    fn video_decoder_construction_follows_the_encoder_rules() {
        // Unset disables frame selection (issue #214).
        assert!(build_video_decoder(env_of(&[])).unwrap().is_none());
        // The shared timeout validates even when only the
        // decoder is configured.
        let err = expect_err(build_video_decoder(env_of(&[(
            "CRABCROP_VIDEO_TIMEOUT_SECS",
            "0",
        )])));
        assert!(err.contains("CRABCROP_VIDEO_TIMEOUT_SECS"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn video_decoder_construction_validates_the_program() {
        // A missing program is a boot error naming the
        // variable, like the encoder's.
        let missing = missing_path("decoder");
        let err = expect_err(build_video_decoder(env_of(&[(
            "CRABCROP_VIDEO_DECODER",
            missing.as_str(),
        )])));
        assert!(err.contains("cannot use CRABCROP_VIDEO_DECODER"), "{err}");
        assert!(err.contains("does not exist"), "{err}");
        // A non-executable file is refused, and the same
        // file with the executable bit constructs.
        let plain = temp_file("decoder-plain", b"#!/bin/sh\n");
        let err = expect_err(build_video_decoder(env_of(&[(
            "CRABCROP_VIDEO_DECODER",
            plain.as_str(),
        )])));
        assert!(err.contains("is not executable"), "{err}");
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&plain, std::fs::Permissions::from_mode(0o755)).unwrap();
        let decoder = build_video_decoder(env_of(&[
            ("CRABCROP_VIDEO_DECODER", plain.as_str()),
            ("CRABCROP_VIDEO_TIMEOUT_SECS", "45"),
        ]))
        .unwrap();
        assert!(decoder.is_some());
    }

    #[test]
    fn provenance_signer_construction_is_all_or_nothing() {
        // Unset: no signing (issue #173).
        assert!(build_provenance_signer(env_of(&[])).unwrap().is_none());
        // A timestamp authority alone is a configuration
        // error, not a silent no-op.
        let err = expect_err(build_provenance_signer(env_of(&[(
            "CRABCROP_C2PA_TSA_URL",
            "https://tsa.example.com",
        )])));
        assert!(err.contains("CRABCROP_C2PA_TSA_URL"), "{err}");
        // A partial pair names the missing half.
        let cert = temp_file("c2pa-cert", crate::provenance::TEST_CHAIN.as_bytes());
        let err = expect_err(build_provenance_signer(env_of(&[(
            "CRABCROP_C2PA_CERT",
            cert.as_str(),
        )])));
        assert!(err.contains("missing CRABCROP_C2PA_KEY"), "{err}");
        let key = temp_file("c2pa-key", crate::provenance::TEST_KEY.as_bytes());
        let err = expect_err(build_provenance_signer(env_of(&[(
            "CRABCROP_C2PA_KEY",
            key.as_str(),
        )])));
        assert!(err.contains("missing CRABCROP_C2PA_CERT"), "{err}");
        // The pair constructs, proven at boot by the
        // sign-and-verify round trip inside the build.
        let signer = build_provenance_signer(env_of(&[
            ("CRABCROP_C2PA_CERT", cert.as_str()),
            ("CRABCROP_C2PA_KEY", key.as_str()),
        ]))
        .unwrap()
        .expect("the pair configures a signer");
        assert_eq!(signer.name, "crabcrop test signer");
        // A set but unreadable certificate is fatal here.
        let missing = missing_path("c2pa-cert");
        let err = expect_err(build_provenance_signer(env_of(&[
            ("CRABCROP_C2PA_CERT", missing.as_str()),
            ("CRABCROP_C2PA_KEY", key.as_str()),
        ])));
        assert!(err.contains("CRABCROP_C2PA_CERT"), "{err}");
    }

    #[test]
    fn pdf_renderer_construction_follows_the_decoder_rules() {
        // Unset disables page selection (issue #215).
        assert!(build_pdf_renderer(env_of(&[])).unwrap().is_none());
        // Its own timeout validates even with no renderer
        // configured.
        for value in ["0", "601", "soon"] {
            let err = expect_err(build_pdf_renderer(env_of(&[(
                "CRABCROP_PDF_TIMEOUT_SECS",
                value,
            )])));
            assert!(err.contains("CRABCROP_PDF_TIMEOUT_SECS"), "{value}: {err}");
        }
        assert!(
            build_pdf_renderer(env_of(&[("CRABCROP_PDF_TIMEOUT_SECS", "600")]))
                .unwrap()
                .is_none()
        );
    }

    #[cfg(unix)]
    #[test]
    fn pdf_renderer_construction_validates_both_programs() {
        use std::os::unix::fs::PermissionsExt;
        // A missing renderer is a boot error naming the
        // variable, like the decoder's.
        let missing = missing_path("renderer");
        let err = expect_err(build_pdf_renderer(env_of(&[(
            "CRABCROP_PDF_RENDERER",
            missing.as_str(),
        )])));
        assert!(err.contains("cannot use CRABCROP_PDF_RENDERER"), "{err}");
        assert!(err.contains("does not exist"), "{err}");
        // An executable renderer whose pdfinfo sibling is
        // missing is refused at boot too: the pair is the
        // contract.
        let dir = crate::test_support::scratch_dir("boot-pdf");
        let renderer_path = dir.join("pdftocairo");
        std::fs::write(&renderer_path, b"#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&renderer_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        let renderer_str = renderer_path.to_string_lossy().into_owned();
        let err = expect_err(build_pdf_renderer(env_of(&[(
            "CRABCROP_PDF_RENDERER",
            renderer_str.as_str(),
        )])));
        assert!(err.contains("cannot use CRABCROP_PDF_RENDERER"), "{err}");
        assert!(err.contains("pdfinfo"), "{err}");
        // The pair in place constructs.
        let info_path = dir.join("pdfinfo");
        std::fs::write(&info_path, b"#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&info_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        let renderer = build_pdf_renderer(env_of(&[
            ("CRABCROP_PDF_RENDERER", renderer_str.as_str()),
            ("CRABCROP_PDF_TIMEOUT_SECS", "45"),
        ]))
        .unwrap();
        assert!(renderer.is_some());
    }

    /// Issue #526: these construction tests once named
    /// their files after the process ID alone, and a later
    /// test process that drew a recycled ID found what an
    /// earlier one had left: a `pdfinfo` next to the
    /// renderer, scripts already marked executable. Their
    /// "missing" and "not executable" refusals then
    /// constructed. Plant those leftovers under this
    /// process's ID and run the three tests over them, each
    /// twice: under a name built from the ID alone, however
    /// spelled, the second run would meet what the first one
    /// left.
    #[cfg(unix)]
    #[test]
    fn construction_tests_ignore_files_left_under_a_reused_pid() {
        use std::io::Write as _;
        use std::os::unix::fs::PermissionsExt;
        let pid = std::process::id();
        let temp = std::env::temp_dir();
        let pdf_dir = temp.join(format!("crabcrop-boot-pdf-{pid}"));
        // Only what this test creates is removed afterwards:
        // a file already under one of these names is the same
        // kind of leftover, and not this test's to delete.
        let made_pdf_dir = std::fs::create_dir(&pdf_dir).is_ok();
        let mut planted = Vec::new();
        for path in [
            pdf_dir.join("pdfinfo"),
            temp.join(format!("crabcrop-boot-encoder-plain-{pid}.tmp")),
            temp.join(format!("crabcrop-boot-decoder-plain-{pid}.tmp")),
        ] {
            let Ok(mut file) = std::fs::File::create_new(&path) else {
                continue;
            };
            file.write_all(b"#!/bin/sh\n").unwrap();
            file.set_permissions(std::fs::Permissions::from_mode(0o755))
                .unwrap();
            planted.push(path);
        }
        let tests: [(&str, fn()); 3] = [
            (
                "pdf_renderer_construction_validates_both_programs",
                pdf_renderer_construction_validates_both_programs,
            ),
            (
                "video_construction_validates_the_encoder_program",
                video_construction_validates_the_encoder_program,
            ),
            (
                "video_decoder_construction_validates_the_program",
                video_decoder_construction_validates_the_program,
            ),
        ];
        let failed: Vec<&str> = tests
            .into_iter()
            .filter(|(_, test)| (0..2).any(|_| std::panic::catch_unwind(test).is_err()))
            .map(|(name, _)| name)
            .collect();
        for path in &planted {
            let _ = std::fs::remove_file(path);
        }
        if made_pdf_dir {
            let _ = std::fs::remove_dir(&pdf_dir);
        }
        assert!(
            failed.is_empty(),
            "failed over the planted files: {failed:?}"
        );
    }

    #[test]
    fn ai_construction_builds_nothing_when_unconfigured() {
        let ai = build_ai_providers(env_of(&[])).unwrap();
        assert!(ai.removebg.is_none());
        assert!(ai.upscale.is_none());
        assert!(ai.face.is_none());
        assert!(ai.tagger.is_none());
        assert!(ai.moderation.is_none());
        assert!(ai.embeddings.is_none());
        assert!(ai.aesthetic.is_none());
        assert!(ai.ocr.is_none());
        assert!(ai.detect.is_none());
        assert!(ai.detectors.is_empty());
        assert!(ai.focal.is_none());
        // The default tag threshold still reaches the state.
        assert_eq!(ai.tag_threshold, 0.05);
    }

    #[test]
    fn ai_construction_refuses_misconfiguration_before_any_load() {
        // The validation that runs before any model load is
        // gate-testable: no runtime is touched on these paths.
        let missing = missing_path("model");
        // A partial CLIP set refuses, naming what is missing.
        let err = expect_err(build_ai_providers(env_of(&[(
            "CRABCROP_CLIP_IMAGE_MODEL",
            missing.as_str(),
        )])));
        assert!(err.contains("partially configured"), "{err}");
        // An out-of-range CLIP threshold refuses too.
        let err = expect_err(build_ai_providers(env_of(&[
            ("CRABCROP_CLIP_IMAGE_MODEL", missing.as_str()),
            ("CRABCROP_CLIP_TEXT_MODEL", missing.as_str()),
            ("CRABCROP_CLIP_TOKENIZER", missing.as_str()),
            ("CRABCROP_CLIP_LABELS", missing.as_str()),
            ("CRABCROP_CLIP_THRESHOLD", "2.5"),
        ])));
        assert!(
            err.contains("CRABCROP_CLIP_THRESHOLD must be within 0..=1"),
            "{err}"
        );
        // The detector threshold is validated before the load.
        let err = expect_err(build_ai_providers(env_of(&[
            ("CRABCROP_DETECT_MODEL", missing.as_str()),
            ("CRABCROP_DETECT_THRESHOLD", "1.5"),
        ])));
        assert!(
            err.contains("CRABCROP_DETECT_THRESHOLD must be within 0..=1"),
            "{err}"
        );
        let err = expect_err(build_ai_providers(env_of(&[
            ("CRABCROP_DETECT_MODEL", missing.as_str()),
            ("CRABCROP_DETECT_THRESHOLD", "high"),
        ])));
        assert!(
            err.contains("CRABCROP_DETECT_THRESHOLD is not a number"),
            "{err}"
        );
        // Moderation labels without a model are zero-shot
        // prompts: without the CLIP set there is nothing to
        // score them with, and the refusal says so.
        let labels = missing_path("labels");
        let err = expect_err(build_ai_providers(env_of(&[(
            "CRABCROP_MODERATION_LABELS",
            labels.as_str(),
        )])));
        assert!(err.contains("need the CRABCROP_CLIP_* set"), "{err}");
        // With a model, an unreadable labels file refuses
        // before the model load is even attempted.
        let err = expect_err(build_ai_providers(env_of(&[
            ("CRABCROP_MODERATION_MODEL", missing.as_str()),
            ("CRABCROP_MODERATION_LABELS", labels.as_str()),
        ])));
        assert!(err.contains("cannot read the moderation labels"), "{err}");
        // An aesthetic head without the CLIP set refuses:
        // the head scores the tagger's embedding, and
        // there is no tagger to score with.
        let head = missing_path("head");
        let err = expect_err(build_ai_providers(env_of(&[(
            "CRABCROP_AESTHETIC_HEAD",
            head.as_str(),
        )])));
        assert!(
            err.contains("CRABCROP_AESTHETIC_HEAD is set but the CRABCROP_CLIP_* set"),
            "{err}"
        );
        // Half a segmentation pair refuses the same way
        // (issue #175): the provider is an encoder and a
        // prompt decoder, and one without the other
        // segments nothing.
        let model = missing_path("segment");
        let err = expect_err(build_ai_providers(env_of(&[(
            "CRABCROP_SEGMENT_ENCODER_MODEL",
            model.as_str(),
        )])));
        assert!(
            err.contains(
                "segmentation is partially configured: missing CRABCROP_SEGMENT_DECODER_MODEL"
            ),
            "{err}"
        );
        let err = expect_err(build_ai_providers(env_of(&[(
            "CRABCROP_SEGMENT_DECODER_MODEL",
            model.as_str(),
        )])));
        assert!(
            err.contains(
                "segmentation is partially configured: missing CRABCROP_SEGMENT_ENCODER_MODEL"
            ),
            "{err}"
        );
        // A partial OCR set refuses the same way (issue
        // #177): the pipeline is a detection model, a
        // recognition model, and its dictionary, and the
        // refusal names every missing piece.
        let model = missing_path("ocr");
        let err = expect_err(build_ai_providers(env_of(&[(
            "CRABCROP_OCR_DETECT_MODEL",
            model.as_str(),
        )])));
        assert!(
            err.contains("OCR is partially configured: missing CRABCROP_OCR_RECOGNIZE_MODEL, CRABCROP_OCR_DICT"),
            "{err}"
        );
        let err = expect_err(build_ai_providers(env_of(&[
            ("CRABCROP_OCR_DETECT_MODEL", model.as_str()),
            ("CRABCROP_OCR_RECOGNIZE_MODEL", model.as_str()),
        ])));
        assert!(
            err.contains("OCR is partially configured: missing CRABCROP_OCR_DICT"),
            "{err}"
        );
        // The language tag alone configures nothing.
        let ai = build_ai_providers(env_of(&[("CRABCROP_OCR_LANGUAGE", "en")])).unwrap();
        assert!(ai.ocr.is_none());
    }

    /// Configured-but-unloadable models refuse construction
    /// with the boot wording. Ignored, gated exactly like the
    /// src/ai.rs load tests (issue #179, item 2): a load only
    /// reaches the model file once the runtime initializes
    /// from an explicit dylib; with no dylib, ort's default
    /// init panics inside the crate instead of erroring, so
    /// the file path cannot be exercised in the plain gate.
    /// CRABCROP_TEST_ORT_DYLIB=...
    /// cargo test -- --ignored ai_construction_refuses_configured
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn ai_construction_refuses_configured_models_that_do_not_load() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let missing = missing_path("model");
        for (name, wording) in [
            (
                "CRABCROP_REMOVEBG_MODEL",
                "cannot load the background-removal model",
            ),
            ("CRABCROP_UPSCALE_MODEL", "cannot load the upscale model"),
            ("CRABCROP_INPAINT_MODEL", "cannot load the inpainting model"),
            (
                "CRABCROP_FACE_MODEL",
                "cannot load the face detection model",
            ),
            ("CRABCROP_DETECT_MODEL", "cannot load the detection model"),
        ] {
            let err = expect_err(build_ai_providers(env_of(&[
                ("CRABCROP_ORT_DYLIB", dylib.as_str()),
                (name, missing.as_str()),
            ])));
            assert!(err.contains(wording), "{name}: {err}");
        }
        // A full CLIP set whose files do not exist refuses too.
        let err = expect_err(build_ai_providers(env_of(&[
            ("CRABCROP_ORT_DYLIB", dylib.as_str()),
            ("CRABCROP_CLIP_IMAGE_MODEL", missing.as_str()),
            ("CRABCROP_CLIP_TEXT_MODEL", missing.as_str()),
            ("CRABCROP_CLIP_TOKENIZER", missing.as_str()),
            ("CRABCROP_CLIP_LABELS", missing.as_str()),
        ])));
        assert!(err.contains("cannot load the CLIP tagging models"), "{err}");
        // A full segmentation pair whose files do not
        // exist refuses too (issue #175).
        let err = expect_err(build_ai_providers(env_of(&[
            ("CRABCROP_ORT_DYLIB", dylib.as_str()),
            ("CRABCROP_SEGMENT_ENCODER_MODEL", missing.as_str()),
            ("CRABCROP_SEGMENT_DECODER_MODEL", missing.as_str()),
        ])));
        assert!(err.contains("cannot load the segment model pair"), "{err}");
        // A full OCR set whose files do not exist refuses
        // too (issue #177).
        let err = expect_err(build_ai_providers(env_of(&[
            ("CRABCROP_ORT_DYLIB", dylib.as_str()),
            ("CRABCROP_OCR_DETECT_MODEL", missing.as_str()),
            ("CRABCROP_OCR_RECOGNIZE_MODEL", missing.as_str()),
            ("CRABCROP_OCR_DICT", missing.as_str()),
        ])));
        assert!(err.contains("cannot load the OCR models"), "{err}");
    }

    /// Full AI construction with real models: every
    /// configured slot of the provider set is populated.
    /// Ignored like the src/ai.rs live tests: it needs the
    /// runtime and the model files, named by the same
    /// CRABCROP_TEST_* variables those tests use:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
    /// CRABCROP_TEST_UPSCALE_MODEL=... CRABCROP_TEST_FACE_MODEL=...
    /// CRABCROP_TEST_DETECT_MODEL=... CRABCROP_TEST_CLIP_IMAGE_MODEL=...
    /// CRABCROP_TEST_CLIP_TEXT_MODEL=... CRABCROP_TEST_CLIP_TOKENIZER=...
    /// CRABCROP_TEST_CLIP_LABELS=...
    /// cargo test -- --ignored ai_construction_with_real_models
    #[test]
    #[ignore = "needs the ONNX runtime and model files (env-configured)"]
    fn ai_construction_with_real_models_builds_every_provider() {
        let var = |name: &str| std::env::var(name).unwrap();
        let mut map: std::collections::HashMap<String, String> = [
            ("CRABCROP_ORT_DYLIB", "CRABCROP_TEST_ORT_DYLIB"),
            ("CRABCROP_REMOVEBG_MODEL", "CRABCROP_TEST_REMOVEBG_MODEL"),
            ("CRABCROP_UPSCALE_MODEL", "CRABCROP_TEST_UPSCALE_MODEL"),
            ("CRABCROP_FACE_MODEL", "CRABCROP_TEST_FACE_MODEL"),
            ("CRABCROP_DETECT_MODEL", "CRABCROP_TEST_DETECT_MODEL"),
            (
                "CRABCROP_CLIP_IMAGE_MODEL",
                "CRABCROP_TEST_CLIP_IMAGE_MODEL",
            ),
            ("CRABCROP_CLIP_TEXT_MODEL", "CRABCROP_TEST_CLIP_TEXT_MODEL"),
            ("CRABCROP_CLIP_TOKENIZER", "CRABCROP_TEST_CLIP_TOKENIZER"),
            ("CRABCROP_CLIP_LABELS", "CRABCROP_TEST_CLIP_LABELS"),
        ]
        .iter()
        .map(|(config, test)| (config.to_string(), var(test)))
        .collect();
        if let Ok(labels) = std::env::var("CRABCROP_TEST_DETECT_LABELS") {
            map.insert("CRABCROP_DETECT_LABELS".to_string(), labels);
        }
        // Moderation joins when its test model is provided
        // (CRABCROP_TEST_MODERATION_MODEL), like the
        // detector labels above.
        let want_moderation = if let Ok(model) = std::env::var("CRABCROP_TEST_MODERATION_MODEL") {
            map.insert("CRABCROP_MODERATION_MODEL".to_string(), model);
            true
        } else {
            false
        };
        // The aesthetic head joins the same way
        // (CRABCROP_TEST_AESTHETIC_HEAD): a pure-Rust
        // load, on top of the CLIP pair above.
        let want_aesthetic = if let Ok(head) = std::env::var("CRABCROP_TEST_AESTHETIC_HEAD") {
            map.insert("CRABCROP_AESTHETIC_HEAD".to_string(), head);
            true
        } else {
            false
        };
        // Inpainting joins the same way
        // (CRABCROP_TEST_INPAINT_MODEL).
        let want_inpaint = if let Ok(model) = std::env::var("CRABCROP_TEST_INPAINT_MODEL") {
            map.insert("CRABCROP_INPAINT_MODEL".to_string(), model);
            true
        } else {
            false
        };
        // Segmentation joins the same way, as a pair
        // (CRABCROP_TEST_SEGMENT_ENCODER_MODEL and
        // CRABCROP_TEST_SEGMENT_DECODER_MODEL).
        let want_segment = match (
            std::env::var("CRABCROP_TEST_SEGMENT_ENCODER_MODEL"),
            std::env::var("CRABCROP_TEST_SEGMENT_DECODER_MODEL"),
        ) {
            (Ok(encoder), Ok(decoder)) => {
                map.insert("CRABCROP_SEGMENT_ENCODER_MODEL".to_string(), encoder);
                map.insert("CRABCROP_SEGMENT_DECODER_MODEL".to_string(), decoder);
                true
            }
            _ => false,
        };
        // OCR joins the same way, as a set of three paths
        // (CRABCROP_TEST_OCR_DETECT_MODEL,
        // CRABCROP_TEST_OCR_RECOGNIZE_MODEL,
        // CRABCROP_TEST_OCR_DICT).
        let want_ocr = match (
            std::env::var("CRABCROP_TEST_OCR_DETECT_MODEL"),
            std::env::var("CRABCROP_TEST_OCR_RECOGNIZE_MODEL"),
            std::env::var("CRABCROP_TEST_OCR_DICT"),
        ) {
            (Ok(detect), Ok(recognize), Ok(dict)) => {
                map.insert("CRABCROP_OCR_DETECT_MODEL".to_string(), detect);
                map.insert("CRABCROP_OCR_RECOGNIZE_MODEL".to_string(), recognize);
                map.insert("CRABCROP_OCR_DICT".to_string(), dict);
                true
            }
            _ => false,
        };
        let ai = build_ai_providers(|name: &str| map.get(name).cloned()).unwrap();
        assert!(ai.removebg.is_some());
        assert!(ai.upscale.is_some());
        assert_eq!(ai.inpaint.is_some(), want_inpaint);
        assert_eq!(ai.segment.is_some(), want_segment);
        assert_eq!(ai.ocr.is_some(), want_ocr);
        assert!(ai.face.is_some());
        assert!(ai.tagger.is_some());
        // The embedding provider rides the tagger.
        assert!(ai.embeddings.is_some());
        assert!(ai.detect.is_some());
        assert_eq!(ai.moderation.is_some(), want_moderation);
        assert_eq!(ai.aesthetic.is_some(), want_aesthetic);
    }

    #[test]
    fn declared_detectors_refuse_an_oversized_file() {
        // A declarations file past the byte cap refuses at
        // the read, before any parsing or load (audit R2-27).
        let oversized = temp_file(
            "detectors-oversized",
            &vec![b' '; MAX_DETECTORS_FILE_BYTES + 1],
        );
        let err = expect_err(build_ai_providers(env_of(&[(
            "CRABCROP_DETECTORS_FILE",
            oversized.as_str(),
        )])));
        assert!(err.contains("byte cap"), "{err}");
    }

    #[test]
    fn declared_detectors_refuse_files_that_fail_before_any_load() {
        // These refusals precede model contact, so they run
        // in the plain gate through full construction.
        let missing = missing_path("detectors");
        let err = expect_err(build_ai_providers(env_of(&[(
            "CRABCROP_DETECTORS_FILE",
            missing.as_str(),
        )])));
        assert!(err.contains("cannot read CRABCROP_DETECTORS_FILE"), "{err}");
        let bad_kind = temp_file(
            "detectors-kind",
            br#"[{"name": "crates", "model": "/m.onnx", "kind": "heatmap"}]"#,
        );
        let err = expect_err(build_ai_providers(env_of(&[(
            "CRABCROP_DETECTORS_FILE",
            bad_kind.as_str(),
        )])));
        assert!(err.contains("unknown kind \"heatmap\""), "{err}");
        assert!(err.contains("\"crates\""), "{err}");
        let dup = temp_file(
            "detectors-dup",
            br#"[{"name": "a", "model": "/m.onnx", "kind": "boxes"},
                 {"name": "a", "model": "/n.onnx", "kind": "focal"}]"#,
        );
        let err = expect_err(build_ai_providers(env_of(&[(
            "CRABCROP_DETECTORS_FILE",
            dup.as_str(),
        )])));
        assert!(err.contains("duplicate detector name: a"), "{err}");
    }

    /// Declared detectors with real models (issue #166):
    /// a boxes declaration with inline classes whose count
    /// matches the export, and two focal declarations (the
    /// second stands by) build into the provider set.
    /// Ignored like the other construction tests:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_DETECT_MODEL=...
    /// CRABCROP_TEST_FOCAL_MODEL=...
    /// cargo test -- --ignored declared_detectors_build_from
    #[test]
    #[ignore = "needs the ONNX runtime and model files (env-configured)"]
    fn declared_detectors_build_from_a_file_with_real_models() {
        let Some(dylib) = crate::ai::env_or_skip("CRABCROP_TEST_ORT_DYLIB") else {
            return;
        };
        let Some(yolo) = crate::ai::env_or_skip("CRABCROP_TEST_DETECT_MODEL") else {
            return;
        };
        let Some(focal_model) = crate::ai::env_or_skip("CRABCROP_TEST_FOCAL_MODEL") else {
            return;
        };
        // The declaration's vocabulary must match the
        // export's class count (the load probe enforces
        // it): 80 generated names.
        let names: Vec<String> = (0..80).map(|i| format!("\"class{i}\"")).collect();
        // The same vocabulary as a labels file, its first
        // name renamed: labels in path form win over the
        // model's metadata exactly like the inline form.
        let labels_text: String = (0..80)
            .map(|i| {
                if i == 0 {
                    "human\n".to_string()
                } else {
                    format!("class{i}\n")
                }
            })
            .collect();
        let labels_file = temp_file("detectors-labels", labels_text.as_bytes());
        let text = format!(
            r#"[
              {{"name": "things", "model": "{yolo}", "kind": "boxes",
                "labels": [{}], "threshold": 0.4}},
              {{"name": "things-file", "model": "{yolo}", "kind": "boxes",
                "labels": "{labels_file}"}},
              {{"name": "saliency", "model": "{focal_model}", "kind": "focal"}},
              {{"name": "saliency-2", "model": "{focal_model}", "kind": "focal"}}
            ]"#,
            names.join(",")
        );
        let file = temp_file("detectors-build", text.as_bytes());
        let ai = build_ai_providers(env_of(&[
            ("CRABCROP_ORT_DYLIB", dylib.as_str()),
            ("CRABCROP_DETECTORS_FILE", file.as_str()),
        ]))
        .unwrap();
        assert_eq!(ai.detectors.len(), 2);
        assert_eq!(ai.detectors[0].name, "things");
        assert_eq!(
            crate::ai::DetectionProvider::describe(&*ai.detectors[0].provider).len(),
            80
        );
        assert_eq!(ai.detectors[1].name, "things-file");
        assert_eq!(
            crate::ai::DetectionProvider::describe(&*ai.detectors[1].provider)[0],
            "human"
        );
        assert!(ai.focal.is_some());
    }

    /// Each load-level refusal of a declared detector
    /// surfaces at construction, naming the declaration
    /// (issue #166). Ignored like the other load tests:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_DETECT_MODEL=...
    /// CRABCROP_TEST_FACE_MODEL=... CRABCROP_TEST_REMOVEBG_MODEL=...
    /// cargo test -- --ignored declared_detectors_refuse_unloadable
    #[test]
    #[ignore = "needs the ONNX runtime and model files (env-configured)"]
    fn declared_detectors_refuse_unloadable_models_at_boot() {
        let var = |name: &str| std::env::var(name).unwrap();
        let dylib = var("CRABCROP_TEST_ORT_DYLIB");
        let missing = missing_path("declared-model");
        let build = |text: String| {
            let file = temp_file("detectors-refuse", text.as_bytes());
            expect_err(build_ai_providers(env_of(&[
                ("CRABCROP_ORT_DYLIB", dylib.as_str()),
                ("CRABCROP_DETECTORS_FILE", file.as_str()),
            ])))
        };
        // A boxes declaration whose model file is missing.
        let err = build(format!(
            r#"[{{"name": "crates", "model": "{missing}", "kind": "boxes", "labels": ["crate"]}}]"#
        ));
        assert!(
            err.contains("cannot load declared detector \"crates\" (boxes)"),
            "{err}"
        );
        // A boxes declaration with no vocabulary anywhere:
        // the model carries no names metadata and the
        // declaration gives no labels.
        let removebg = var("CRABCROP_TEST_REMOVEBG_MODEL");
        let err = build(format!(
            r#"[{{"name": "crates", "model": "{removebg}", "kind": "boxes"}}]"#
        ));
        assert!(
            err.contains("cannot load declared detector \"crates\" (boxes)"),
            "{err}"
        );
        assert!(err.contains("carries no class names metadata"), "{err}");
        // A focal declaration on a multi-output model fails
        // the heatmap probe.
        let yunet = var("CRABCROP_TEST_FACE_MODEL");
        let err = build(format!(
            r#"[{{"name": "saliency", "model": "{yunet}", "kind": "focal"}}]"#
        ));
        assert!(
            err.contains("cannot load declared detector \"saliency\" (focal)"),
            "{err}"
        );
        assert!(err.contains("does not produce a heatmap output"), "{err}");
    }

    #[test]
    fn build_state_builds_the_configured_state_and_sizes_the_gates() {
        let config = Config::from_env(env_of(&[
            ("CRABCROP_SECRET", "a production-grade secret"),
            ("CRABCROP_METRICS_TOKEN", "metrics-token"),
            ("CRABCROP_SLOW_LANE_PERMITS", "3"),
            ("CRABCROP_META_MODE", "1"),
        ]))
        .unwrap();
        let state = build_state(&config, open_stores(&config).unwrap()).unwrap();
        // The gate sizes are the documented constants.
        assert_eq!(state.ai_gate.available_permits(), AI_GATE_PERMITS);
        assert_eq!(state.anim_gate.available_permits(), ANIM_GATE_PERMITS);
        assert_eq!(
            state.video_gate.available_permits(),
            crate::video::GATE_PERMITS
        );
        assert_eq!(state.process_gate.capacity(), config.process_permits);
        assert_eq!(state.slow_gate.capacity(), 3);
        // The settings reach the state as validated.
        assert_eq!(state.metrics_token.as_deref(), Some("metrics-token"));
        assert!(state.meta_enabled);
        assert_eq!(state.config_fingerprint, config.config_fingerprint);
        assert_eq!(
            state.process_timeout,
            std::time::Duration::from_secs(config.process_timeout_secs)
        );
        assert!(state.store.is_none() && state.shared_store.is_none());
        assert!(state.font.is_none() && state.fallback.is_none());
    }

    #[test]
    fn build_state_names_a_component_that_does_not_build() {
        let font = temp_file("state-font", b"not a font");
        let config = Config::from_env(env_of(&[
            ("CRABCROP_SECRET", "a production-grade secret"),
            ("CRABCROP_FONT_PATH", font.as_str()),
        ]))
        .unwrap();
        let err = build_state(&config, open_stores(&config).unwrap())
            .map(|_| ())
            .unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::Build {
                    var: Some("CRABCROP_FONT_PATH"),
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(
            err.to_string()
                .starts_with("cannot load the overlay font: "),
            "{err}"
        );
    }
}
