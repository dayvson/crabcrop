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
    // The worker branched off above and stays dumpable: its
    // environment holds no secret, and it sets its own OOM
    // score through `/proc`.
    not_dumpable();
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

/// Mark the process non-dumpable on Linux (issue #605),
/// before any child starts. The children the server starts
/// (the SVG and AVIF worker, and the tools) run as its user.
/// Without this, a compromised child could read the server's
/// secrets from `/proc/<pid>/environ`, read its memory, or
/// attach to it. Now only a privileged process can, such as
/// one with `CAP_SYS_PTRACE`, and the server leaves no core
/// dump. A child is dumpable again once its exec completes,
/// so the server can still set a tool's OOM score (see
/// `toolrun::limit_child`), and the worker its own. This
/// does not stop a child from signalling the server or
/// reading its files: only a separate user or a sandbox
/// would (see SECURITY.md).
fn not_dumpable() {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    if let Err(e) =
        rustix::process::set_dumpable_behavior(rustix::process::DumpableBehavior::NotDumpable)
    {
        eprintln!("crabcrop: cannot mark the process non-dumpable: {e}");
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
    // The request and source posture (issue #590), with a
    // warning when unsigned URLs may fetch from any host.
    println!("{}", config.posture_line());
    if let Some(warning) = config.posture_warning() {
        eprintln!("warning: {warning}");
    }
    println!(
        "source fetch: cap {} MB, timeout {}s, buffer budget {} MB",
        config.fetch_cap_mb,
        config.fetch_timeout_secs,
        config.source_buffer_bytes / (1024 * 1024)
    );
    println!("configuration fingerprint: {}", state.config_fingerprint);
    // Purges recorded in the shared bucket hold here before
    // the first request (issue #461). A slow or hanging
    // store holds the first request, and the listening line,
    // for 10 s at most (issue #549).
    crate::server::start_purge_tombstones(&state).await;
    let background = state.background.clone();
    let signalled = shutdown_signal();
    println!("crabcrop listening on port {port}");
    // Shutdown (issue #463): at SIGTERM or Ctrl-C the
    // listener closes, idle connections close, and requests
    // in flight finish; then the background work they and
    // earlier requests left (renders whose client left
    // after they started, shared-tier write-throughs,
    // purges whose caller left) finishes. One deadline, the
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
        "shutdown: draining requests and background work for up to {}s",
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
    if left.renders > 0 {
        // A render whose client had gone, cut off before it
        // was stored: the next request for it renders it.
        eprintln!(
            "shutdown: {} renders whose client left, still running {when}, are dropped unstored",
            left.renders
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

/// Permits on the probe gate: heavy `/explain` probes (an
/// SVG's parse in a worker process, an AVIF's decode, a
/// HEIC's item graph walk) run on blocking threads, at most
/// two at once, so a burst of explain requests cannot
/// occupy the async runtime or start a worker per request.
const PROBE_GATE_PERMITS: usize = 2;

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
    if !config.client_hints {
        println!("client hints: off");
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
        client_hints: config.client_hints,
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
        probe_gate: Arc::new(tokio::sync::Semaphore::new(PROBE_GATE_PERMITS)),
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
/// setting, PR #493; `upscale` and `remove_region` on a
/// 16-bit or float source keep its alpha and its depth
/// instead of serving opaque 8-bit RGB, which showed the
/// color stored under transparent pixels #551; SVG sources
/// whose reference fan-out (an id carried by many elements,
/// an `inherit`, `context-*`, or `<use>` value list, or a
/// memoized subtree reached from many sites) or whose
/// stylesheet matching made the in-process pre-parse bound
/// take time more than linear in the document, which usvg
/// would render but are now refused with `502` instead, so
/// the bound stays within the linear work its documentation
/// promises; a PDF page whose crop box is smaller than its
/// media box rendered as its crop box, the size the probe
/// measures, instead of its media box; and a PDF page too
/// large to fit the raster limits at 1 dpi refused instead
/// of rendered past them).
///
/// 5: the first release after 0.10.1, which changes output
/// for existing URLs (a signed output records every image
/// source as its parent ingredient, a container JPEG XL, a
/// HEIF whose major brand is not `heic` or `heix`, and an
/// SVG, TIFF, BMP or ICO source included, instead of
/// starting a fresh chain #564, PR #581, and no longer
/// carries the source's C2PA manifest store, its parent
/// ingredient naming only the source's format and the
/// status codes its store validated with #610, PR #621; a
/// `Save-Data` request for a URL whose per-format
/// `quality(...)` list names no value for the output
/// format, such as `format(webp):quality(jpeg:90)`, encoded
/// at the format's default quality instead of the stepped
/// 70, which the release before could also store as the
/// URL's plain entry #569, PR #618; a `file://` source path
/// with a `..` segment answered `404` even where it stays
/// inside the file root, PR #620; an SVG source draws
/// nothing for an image reference that points outside the
/// document, PR #623; `frame:` refused with `502` on a
/// video stream in a codec off the decoder list (WMV and
/// VC-1, Sorenson H.263 and VP6 in FLV, Microsoft MPEG-4,
/// lossless, raw and screen-capture codecs) #605, on a
/// video with any stream past the decode budget, displayed
/// or coded, and where the decode needs more than the
/// extractor's 3 GiB address-space cap, 8 GiB before; a PDF
/// page drawn without an embedded image whose decode needs
/// more than the PDF tools' 3 GiB cap, a page whose own
/// surfaces need more refused with `502`, and every page of
/// a document that stores a JPEG 2000 image past that cap
/// refused with `502`, PR #624; and an animated GIF whose
/// frames pass the 100 million pixel budget once each
/// counts at its declared size where that is larger than
/// the canvas, a HEIC/HEIF source whose item graph takes
/// too much work to walk or would hold more than 128 MiB,
/// or whose `altr` alternative's gain map fails the checks
/// run before a decode, and a JPEG XL source with a frame
/// holding more pixels than the decode budget allows a
/// whole image, all of which rendered before and are now
/// refused with `502`, PR #623).
///
/// 6: the first release after 0.11.0, which changes output
/// for existing URLs (WebM output written with bitexact
/// muxing: the Matroska muxer writes track UID 1 and the
/// generic `Lavf` and `Lavc` tags instead of a random
/// track UID and the encoder build's own tags, so WebM
/// bytes change once and then repeat render for render on
/// one encoder build #567, PR #585).
const RENDERING_VERSION: u32 = 6;

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
/// every cache key (see `cache_key_for` in `server.rs`),
/// so a restart under changed settings starts a fresh
/// cache epoch instead of serving the old settings'
/// renders from the persistent tiers. Settings that are
/// already visible in the key (presets, default ops: both
/// change the canonical form) are deliberately excluded,
/// and so is the fallback image, whose responses are
/// never cached.
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
#[path = "main/tests.rs"]
mod tests;
