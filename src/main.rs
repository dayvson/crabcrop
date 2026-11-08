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
//! Every environment variable is documented in
//! `docs/configuration.md`, the single reference for
//! configuration.

#![forbid(unsafe_code)]

// The modules live in the library target (`src/lib.rs`);
// importing them at the crate root lets every `crate::`
// path in this file resolve to them.
use crabcrop::{
    ai, cache, cli, engine, frames, loader, observe, pdf, provenance, quality, s3store, server,
    sigv4, storage, text, upload, url, video,
};

use std::env;
use std::sync::Arc;

use crate::ai::UpscaleProvider;
use crate::cache::Cache;
use crate::engine::Engine;
use crate::loader::{Loader, SourceConfig, SourcePolicy};
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
    start();
}

#[tokio::main]
async fn start() {
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
                let sep = if e.starts_with("crabcrop:") {
                    ""
                } else {
                    "crabcrop: "
                };
                eprintln!("{sep}{e}");
                std::process::exit(1);
            }
        }
        return;
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
fn build_presets(
    get: impl Fn(&str) -> Option<String>,
) -> Result<(std::collections::HashMap<String, String>, bool), String> {
    let only_presets =
        parse_bool_value("CRABCROP_ONLY_PRESETS", get("CRABCROP_ONLY_PRESETS"), false)?;
    let presets = match get("CRABCROP_PRESETS_FILE") {
        Some(path) if !path.is_empty() => {
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

/// The most operations one preset body may carry: the
/// per-request operation cap, applied at boot (issue #268)
/// and counted the way requests count it (issue #469), so
/// each filter in a `filters:` segment counts once. A
/// preset larger than any request may legally run would
/// otherwise boot fine and make every request that names
/// it answer `400`.
const MAX_PRESET_OPERATIONS: usize = crate::url::MAX_OPERATIONS;

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
        let parsed = crate::url::parse_preset_operations(ops)
            .map_err(|e| format!("invalid operations for preset {name}: {e}"))?;
        let count = crate::url::operation_count(&parsed);
        if count > MAX_PRESET_OPERATIONS {
            return Err(format!(
                "preset {name} has {count} operations, over the {MAX_PRESET_OPERATIONS} operation cap"
            ));
        }
        map.insert(name.to_string(), ops.to_string());
    }
    Ok(map)
}

/// Load the source root configuration (issue #248): the
/// roots file (one `name = base-url` entry per line),
/// validated strictly. A root binds a name to the base URL
/// that `root:<name>/<path>` request sources expand
/// against; the file shape and validation habits mirror the
/// presets file because the job is the same: a named
/// reference the server expands after verification.
fn build_source_roots(
    get: impl Fn(&str) -> Option<String>,
) -> Result<std::collections::HashMap<String, String>, String> {
    match get("CRABCROP_SOURCE_ROOTS_FILE") {
        Some(path) if !path.is_empty() => {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("cannot read CRABCROP_SOURCE_ROOTS_FILE ({path}): {e}"))?;
            parse_source_roots(&text)
        }
        _ => Ok(std::collections::HashMap::new()),
    }
}

/// Parse source root definitions: one `name = base-url`
/// entry per line, blank lines and `#` comments skipped.
/// Names follow the preset charset (letters, digits, `_`,
/// `-`); a base must be an absolute http(s) URL with no
/// userinfo, query, or fragment: it is a prefix request
/// paths join onto, and the loader refuses userinfo sources
/// in any case. Bases are stored in their parsed form with
/// trailing slashes trimmed, ready for `base + "/" + path`.
fn parse_source_roots(text: &str) -> Result<std::collections::HashMap<String, String>, String> {
    let mut map = std::collections::HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, base) = line
            .split_once('=')
            .ok_or_else(|| format!("source root line {} is not `name = base-url`", i + 1))?;
        let name = name.trim();
        let base = base.trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(format!(
                "invalid source root name on line {}: {name:?}",
                i + 1
            ));
        }
        if map.contains_key(name) {
            return Err(format!("duplicate source root name: {name}"));
        }
        let parsed = reqwest::Url::parse(base)
            .map_err(|e| format!("invalid base URL for source root {name}: {e}"))?;
        if parsed.scheme() != "http" && parsed.scheme() != "https" {
            return Err(format!(
                "base URL for source root {name} must be an http(s) URL: {base:?}"
            ));
        }
        if parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
        {
            return Err(format!(
                "base URL for source root {name} must not carry userinfo: {base:?}"
            ));
        }
        if parsed.query().is_some() || parsed.fragment().is_some() {
            return Err(format!(
                "base URL for source root {name} must not carry a query or fragment: {base:?}"
            ));
        }
        map.insert(
            name.to_string(),
            parsed.to_string().trim_end_matches('/').to_string(),
        );
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

/// The source policy from the environment. Private-address
/// blocking is on by default: a source that is, or resolves
/// to, a non-public address is refused unless its host is on
/// the allowlist (the operator's exception for loopback and
/// internal sources) or the check is turned off with
/// `CRABCROP_BLOCK_PRIVATE_SOURCES=0`.
/// `CRABCROP_ALLOW_ANY_SOURCE` is the deliberate full
/// opt-out: it switches the private-address check off
/// entirely. A configured allowlist still applies either
/// way.
fn build_source_policy(get: impl Fn(&str) -> Option<String>) -> Result<SourcePolicy, String> {
    let allowed_sources: Vec<String> = get("CRABCROP_ALLOWED_SOURCES")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let block_private = parse_bool_value(
        "CRABCROP_BLOCK_PRIVATE_SOURCES",
        get("CRABCROP_BLOCK_PRIVATE_SOURCES"),
        true,
    )?;
    let allow_any = parse_bool_value(
        "CRABCROP_ALLOW_ANY_SOURCE",
        get("CRABCROP_ALLOW_ANY_SOURCE"),
        false,
    )?;
    Ok(SourcePolicy::new(
        allowed_sources,
        block_private && !allow_any,
    ))
}

/// A bucket store's base URL from the variable `name`: an
/// http(s) URL, kept with any trailing slash trimmed. Every
/// bucket fetch is an HTTP request to it, so any other
/// scheme is a startup error, for the GCS endpoint as much
/// as the S3 one (issue #455).
fn store_endpoint(name: &str, endpoint: &str) -> Result<String, String> {
    let parsed = reqwest::Url::parse(endpoint).map_err(|e| format!("invalid {name}: {e}"))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(format!("{name} must be an http(s) URL"));
    }
    Ok(endpoint.trim_end_matches('/').to_string())
}

/// An error followed by its causes, `: `-separated. reqwest
/// reports a client it cannot build only as "builder error";
/// the reason (no CA roots on the host, say) is in the
/// source chain, and a boot error should name it.
fn with_causes(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut cause = error.source();
    while let Some(inner) = cause {
        message.push_str(": ");
        message.push_str(&inner.to_string());
        cause = inner.source();
    }
    message
}

fn build_source_config(get: impl Fn(&str) -> Option<String>) -> Result<SourceConfig, String> {
    let mut sources = SourceConfig::default();
    let nonempty = |name: &str| get(name).filter(|v| !v.is_empty());

    if let Some(endpoint) = nonempty("CRABCROP_S3_ENDPOINT") {
        sources.s3_endpoint = Some(store_endpoint("CRABCROP_S3_ENDPOINT", &endpoint)?);
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
        sources.gcs_endpoint = store_endpoint("CRABCROP_GCS_ENDPOINT", &endpoint)?;
    }
    if let Some(region) = nonempty("CRABCROP_GCS_REGION") {
        sources.gcs_region = region;
    }
    sources.gcs_creds = cred_pair("CRABCROP_GCS_ACCESS_KEY", "CRABCROP_GCS_SECRET_KEY")?;
    sources.allowed_buckets = get("CRABCROP_ALLOWED_BUCKETS")
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
    // A configured store endpoint is where every bucket
    // fetch goes, whatever host it names: the source
    // allowlist governs HTTP sources only, and the bucket
    // allowlist is the gate for bucket sources. Say so at
    // boot, where the endpoint is set, rather than leaving
    // the split implicit.
    if sources.s3_endpoint.is_some() || nonempty("CRABCROP_GCS_ENDPOINT").is_some() {
        println!(
            "note: bucket sources fetch from the configured store endpoints; \
             CRABCROP_ALLOWED_BUCKETS gates them, the host allowlist does not"
        );
    }
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

/// Default quantization step for width client hints.
const DEFAULT_WIDTH_STEP: u32 = 100;

/// Parse a `CRABCROP_JXL_SPEED` value into a jixel speed tier.
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

/// Read a boolean environment variable, strictly: `1`,
/// `true`, `yes`, `on` (any case) are true; `0`, `false`,
/// `no`, `off` are false; anything else is a startup error
/// naming the variable. A lenient reading (only `1` or
/// `true` count) would make a typo like `ture` silently
/// false, which for a safety flag such as
/// CRABCROP_BLOCK_PRIVATE_SOURCES is the unsafe direction.
fn parse_bool(name: &str, default: bool) -> Result<bool, String> {
    parse_bool_value(name, env::var(name).ok(), default)
}

fn parse_bool_value(name: &str, value: Option<String>, default: bool) -> Result<bool, String> {
    match value.as_deref() {
        None => Ok(default),
        Some(v) => match v.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(format!(
                "invalid {name}: {v:?} (expected one of 1, true, yes, on, 0, false, no, off)"
            )),
        },
    }
}

/// The minimum `CRABCROP_SECRET` length, in bytes. Shorter
/// secrets are brute-forceable, and an empty one is worse:
/// empty-key HMAC is publicly computable, so every signed URL
/// (and purge) is forgeable (security audit PL-3, P-6).
const MIN_SECRET_BYTES: usize = 16;

fn validate_secret(secret: &[u8]) -> Result<(), String> {
    validate_secret_named("CRABCROP_SECRET", secret)
}

fn validate_secret_named(name: &str, secret: &[u8]) -> Result<(), String> {
    if secret.len() < MIN_SECRET_BYTES {
        return Err(format!(
            "{name} must be at least {MIN_SECRET_BYTES} bytes, got {}",
            secret.len()
        ));
    }
    Ok(())
}

/// The signing keyring (issue #143): the current secret under
/// `CRABCROP_SECRET_KEY_ID` (default `current`), plus the
/// previous secret under `CRABCROP_SECRET_PREVIOUS_KEY_ID`
/// (default `previous`) while a rotation window is open.
fn build_keyring(get: impl Fn(&str) -> Option<String>) -> Result<crate::url::Keyring, String> {
    let current = get("CRABCROP_SECRET")
        .ok_or("CRABCROP_SECRET must be set")?
        .into_bytes();
    validate_secret(&current)?;
    let current_id = get("CRABCROP_SECRET_KEY_ID")
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "current".to_string());
    if !crate::url::is_valid_key_id(&current_id) {
        return Err(format!(
            "CRABCROP_SECRET_KEY_ID {current_id:?} is not a valid key ID (1 to 32 characters of [A-Za-z0-9_-])"
        ));
    }
    let previous = match get("CRABCROP_SECRET_PREVIOUS") {
        Some(p) if !p.is_empty() => {
            let bytes = p.into_bytes();
            validate_secret_named("CRABCROP_SECRET_PREVIOUS", &bytes)?;
            let id = get("CRABCROP_SECRET_PREVIOUS_KEY_ID")
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| "previous".to_string());
            if !crate::url::is_valid_key_id(&id) {
                return Err(format!(
                    "CRABCROP_SECRET_PREVIOUS_KEY_ID {id:?} is not a valid key ID (1 to 32 characters of [A-Za-z0-9_-])"
                ));
            }
            if id == current_id {
                return Err(
                    "CRABCROP_SECRET_KEY_ID and CRABCROP_SECRET_PREVIOUS_KEY_ID must differ"
                        .to_string(),
                );
            }
            Some((id, bytes))
        }
        _ => None,
    };
    let mut keyring = crate::url::Keyring::single(current);
    keyring.current_id = current_id;
    keyring.previous = previous;
    Ok(keyring)
}

/// The upper bound on `CRABCROP_SOURCE_MAX_MB`, 1 GiB: a
/// source is buffered whole before it renders, and one
/// body's charge against the source buffer budget must
/// stay a single semaphore grant. The video output cap
/// stops at the same figure.
const MAX_SOURCE_MAX_MB: u64 = 1024;

/// The source fetch cap in megabytes (issue #471):
/// `CRABCROP_SOURCE_MAX_MB`, 1 to 1024, defaulting to the
/// loader's 10 MiB. The upload cap and the source buffer
/// budget derive from it, so each reads it here, through
/// the same `get`.
fn source_max_mb(get: &impl Fn(&str) -> Option<String>) -> Result<u64, String> {
    let mb: u64 = parse_env_value(
        "CRABCROP_SOURCE_MAX_MB",
        get("CRABCROP_SOURCE_MAX_MB"),
        crate::loader::DEFAULT_MAX_BYTES / (1024 * 1024),
    )?;
    if !(1..=MAX_SOURCE_MAX_MB).contains(&mb) {
        return Err(format!(
            "CRABCROP_SOURCE_MAX_MB must be between 1 and {MAX_SOURCE_MAX_MB}, got {mb}"
        ));
    }
    Ok(mb)
}

/// The total time one source fetch may take, redirects
/// and body included (issue #471):
/// `CRABCROP_SOURCE_TIMEOUT_SECS`, 1 to 600 like the other
/// wall-clock limits, defaulting to the loader's 10 s.
fn source_timeout_secs(get: &impl Fn(&str) -> Option<String>) -> Result<u64, String> {
    let secs: u64 = parse_env_value(
        "CRABCROP_SOURCE_TIMEOUT_SECS",
        get("CRABCROP_SOURCE_TIMEOUT_SECS"),
        crate::loader::DEFAULT_TIMEOUT.as_secs(),
    )?;
    if !(1..=600).contains(&secs) {
        return Err(format!(
            "CRABCROP_SOURCE_TIMEOUT_SECS must be between 1 and 600, got {secs}"
        ));
    }
    Ok(secs)
}

/// The proxy variables the HTTP client honours, as pairs
/// it reads in order: the first of a pair that is set
/// wins, even when it is empty.
const PROXY_VARIABLES: [[&str; 2]; 3] = [
    ["HTTPS_PROXY", "https_proxy"],
    ["HTTP_PROXY", "http_proxy"],
    ["ALL_PROXY", "all_proxy"],
];

/// The boot warning for a configured proxy that source
/// fetches bypass (issue #471), or `None` when there is
/// nothing to warn about. A pinned fetch connects to the
/// address the private-address check validated and never
/// goes through a proxy, which would resolve the host
/// again on its own; under proxy-only egress those
/// fetches fail with no other hint. Only the variable
/// names are reported: a proxy URL can carry credentials.
fn proxy_bypass_warning(
    get: impl Fn(&str) -> Option<String>,
    policy: &SourcePolicy,
) -> Option<String> {
    if !policy.pins_hostnames() {
        return None;
    }
    let set: Vec<&str> = PROXY_VARIABLES
        .iter()
        .filter_map(|pair| {
            let (name, value) = pair
                .iter()
                .find_map(|name| get(name).map(|value| (*name, value)))?;
            (!value.trim().is_empty()).then_some(name)
        })
        .collect();
    if set.is_empty() {
        return None;
    }
    Some(format!(
        "a proxy is configured ({}), but source fetches bypass it: with private-address \
         blocking on (CRABCROP_BLOCK_PRIVATE_SOURCES) and no CRABCROP_ALLOWED_SOURCES, \
         each source host is pinned to the address the check validated and connected to \
         directly, so under proxy-only egress those fetches fail. List the source hosts \
         in CRABCROP_ALLOWED_SOURCES to fetch them through the proxy \
         (see docs/configuration.md)",
        set.join(", ")
    ))
}

/// Fetch caps per processing lane permit in the default
/// source buffer budget (issue #403): for each permit, room
/// for one source rendering, one loaded and queued behind
/// it, and two loading, all at the cap. A load is charged
/// its declared length, then what it received, so typical
/// sources, far under the cap, fit many more at once.
const SOURCE_BUFFER_CAPS_PER_PERMIT: u64 = 4;

/// The upper bound on `CRABCROP_SOURCE_BUFFER_MAX_MB`,
/// 64 GiB; the computed default is clamped to it too.
const MAX_SOURCE_BUFFER_MB: u64 = 65_536;

/// The source buffer budget in bytes (issue #403), from
/// `CRABCROP_SOURCE_BUFFER_MAX_MB`, read through `get` so
/// tests can drive it. The default scales with the
/// processing lanes (`lane_permits`, the stills lane plus
/// the slow lane): four fetch caps per permit, at the
/// configured cap (issue #471). The floor is one fetch
/// cap, below which no source at the cap could ever load.
fn source_buffer_budget(
    get: impl Fn(&str) -> Option<String>,
    lane_permits: usize,
) -> Result<u64, String> {
    const MB: u64 = 1024 * 1024;
    let cap_mb = source_max_mb(&get)?;
    let default_mb = (cap_mb * SOURCE_BUFFER_CAPS_PER_PERMIT)
        .saturating_mul(lane_permits as u64)
        .clamp(cap_mb, MAX_SOURCE_BUFFER_MB);
    let mb: u64 = parse_env_value(
        "CRABCROP_SOURCE_BUFFER_MAX_MB",
        get("CRABCROP_SOURCE_BUFFER_MAX_MB"),
        default_mb,
    )?;
    if !(cap_mb..=MAX_SOURCE_BUFFER_MB).contains(&mb) {
        return Err(format!(
            "CRABCROP_SOURCE_BUFFER_MAX_MB must be between {cap_mb} (the source fetch cap) \
             and {MAX_SOURCE_BUFFER_MB}, got {mb}"
        ));
    }
    Ok(mb * MB)
}

/// Build the upload destination (issue #168) from the
/// environment, read through `get` so tests can drive it.
/// Exactly one of `CRABCROP_UPLOAD_DIR` /
/// `CRABCROP_UPLOAD_BUCKET` may be set; neither disables
/// uploads. A configured but unusable destination is a
/// startup error, like every other boot config here.
/// `sources` is the already-built source configuration:
/// the bucket destination reuses its S3 endpoint, region,
/// and credentials, and the caller points the loader's
/// file root at a local destination (see `run`).
fn build_upload_store(
    get: impl Fn(&str) -> Option<String>,
    sources: &SourceConfig,
) -> Result<Option<std::sync::Arc<crate::upload::UploadStore>>, String> {
    let nonempty = |name: &str| get(name).filter(|v| !v.is_empty());
    let dir = nonempty("CRABCROP_UPLOAD_DIR");
    let bucket = nonempty("CRABCROP_UPLOAD_BUCKET");
    if dir.is_some() && bucket.is_some() {
        return Err(
            "CRABCROP_UPLOAD_DIR and CRABCROP_UPLOAD_BUCKET are both set: \
             configure exactly one upload destination"
                .to_string(),
        );
    }
    if dir.is_none() && bucket.is_none() {
        return Ok(None);
    }
    // The upload cap defaults to the source fetch cap and
    // may not pass it (issue #168 ties the two, so a stored
    // original is always one the loader will fetch back).
    let source_mb = source_max_mb(&get)?;
    let max_mb: u64 = parse_env_value(
        "CRABCROP_UPLOAD_MAX_MB",
        get("CRABCROP_UPLOAD_MAX_MB"),
        source_mb,
    )?;
    if max_mb == 0 || max_mb > source_mb {
        return Err(format!(
            "CRABCROP_UPLOAD_MAX_MB must be between 1 and {source_mb} \
             (the source fetch cap, CRABCROP_SOURCE_MAX_MB, an uploaded original \
             must stay servable under), got {max_mb}"
        ));
    }
    let max_bytes = max_mb * 1024 * 1024;
    let allow_overwrite = parse_bool_value(
        "CRABCROP_UPLOAD_ALLOW_OVERWRITE",
        get("CRABCROP_UPLOAD_ALLOW_OVERWRITE"),
        false,
    )?;
    match (dir, bucket) {
        (Some(dir), None) => {
            std::fs::create_dir_all(&dir)
                .map_err(|e| format!("CRABCROP_UPLOAD_DIR is not usable: {e}"))?;
            let canonical = std::fs::canonicalize(&dir)
                .map_err(|e| format!("CRABCROP_UPLOAD_DIR is not usable: {e}"))?;
            if !canonical.is_dir() {
                return Err("CRABCROP_UPLOAD_DIR is not a directory".to_string());
            }
            if let Some(root) = &sources.file_root
                && !canonical.starts_with(root)
            {
                return Err("CRABCROP_UPLOAD_DIR must be under CRABCROP_FILE_ROOT \
                     (uploaded objects are served back as file:// sources)"
                    .to_string());
            }
            println!(
                "uploads enabled: directory {dir} (cap {max_mb} MB, overwrite {allow_overwrite})"
            );
            // The reference an upload hands back must name
            // the object under the file root (issue #265):
            // with a root configured above the upload dir,
            // the ref carries the dir's path below the
            // root; with no root, boot points the root at
            // the dir itself and the ref stays bare.
            let ref_prefix = match &sources.file_root {
                Some(root) => match canonical.strip_prefix(root) {
                    Ok(rel) if !rel.as_os_str().is_empty() => {
                        format!("{}/", rel.to_string_lossy().replace('\\', "/"))
                    }
                    _ => String::new(),
                },
                None => String::new(),
            };
            Ok(Some(std::sync::Arc::new(
                crate::upload::UploadStore::local(canonical, max_bytes, allow_overwrite)
                    .with_ref_prefix(ref_prefix),
            )))
        }
        (None, Some(bucket)) => {
            // Bucket fetches must not silently gain a
            // readable bucket (audit L-1's rule, applied to
            // serving the uploads back): the destination
            // bucket must be in the readable allowlist,
            // which bucket loads require with or without
            // credentials.
            if !sources
                .allowed_buckets
                .iter()
                .any(|b| b == &bucket || b == "*")
            {
                return Err(format!(
                    "CRABCROP_UPLOAD_BUCKET {bucket:?} is not in CRABCROP_ALLOWED_BUCKETS: \
                     uploaded objects are served back as s3:// sources, so the bucket \
                     must be a readable source bucket"
                ));
            }
            let prefix = get("CRABCROP_UPLOAD_PREFIX").unwrap_or_default();
            let store = crate::upload::UploadStore::s3(
                bucket.clone(),
                prefix,
                sources.s3_endpoint.clone(),
                sources.s3_region.clone(),
                sources.aws_creds.clone(),
                max_bytes,
                allow_overwrite,
            )
            .map_err(|e| format!("cannot build the upload store client: {}", with_causes(&e)))?;
            println!(
                "uploads enabled: bucket {bucket} (cap {max_mb} MB, overwrite {allow_overwrite})"
            );
            Ok(Some(std::sync::Arc::new(store)))
        }
        _ => unreachable!("the both-set and neither-set cases returned above"),
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let keyring = build_keyring(|name| env::var(name).ok())?;
    let allow_unsafe = parse_bool("CRABCROP_ALLOW_UNSAFE", false)?;
    // CRABCROP_PORT is the namespaced alias for PORT and
    // wins when both are set; PORT alone keeps working for
    // the deployments and charts that already use it.
    let (port_name, port_value) = match env::var("CRABCROP_PORT") {
        Ok(v) => ("CRABCROP_PORT", Some(v)),
        Err(_) => ("PORT", env::var("PORT").ok()),
    };
    let port: u16 = parse_env_value(port_name, port_value, 3000)?;
    let ssim_target: f64 = parse_env("CRABCROP_SSIM_TARGET", crate::quality::DEFAULT_SSIM_TARGET)?;
    if !ssim_target.is_finite() || !(0.0..=100.0).contains(&ssim_target) {
        return Err(
            format!("CRABCROP_SSIM_TARGET must be between 0 and 100, got {ssim_target}").into(),
        );
    }
    // The /metrics bearer token (issue #270): an empty
    // value is no token, and no token closes the
    // endpoint rather than serving it publicly.
    let metrics_token = env::var("CRABCROP_METRICS_TOKEN")
        .ok()
        .filter(|v| !v.is_empty());
    let process_timeout_secs: u64 = parse_env("CRABCROP_PROCESS_TIMEOUT_SECS", 30)?;
    if !(1..=600).contains(&process_timeout_secs) {
        return Err(format!(
            "CRABCROP_PROCESS_TIMEOUT_SECS must be between 1 and 600, got {process_timeout_secs}"
        )
        .into());
    }
    // The slow lane's size (issue #324): how many
    // long-running encodes (animated WebP, video, frame
    // and page work) may run at once, on permits stills
    // never wait for. The default 1 suits a small host:
    // one long encode at a time, with the stills lane and
    // a core left for stills.
    let slow_lane_permits: usize = parse_env(
        "CRABCROP_SLOW_LANE_PERMITS",
        crate::server::DEFAULT_SLOW_LANE_PERMITS,
    )?;
    if !(1..=64).contains(&slow_lane_permits) {
        return Err(format!(
            "CRABCROP_SLOW_LANE_PERMITS must be between 1 and 64, got {slow_lane_permits}"
        )
        .into());
    }
    // The stills lane is sized to the cores at boot; the
    // source buffer budget's default scales with both lanes.
    let process_permits = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // The source fetch bounds (issue #471); the budget and
    // the upload cap derive from the same cap.
    let fetch_cap_mb = source_max_mb(&|name: &str| env::var(name).ok())?;
    let fetch_timeout_secs = source_timeout_secs(&|name: &str| env::var(name).ok())?;
    let source_buffer_bytes = source_buffer_budget(
        |name| env::var(name).ok(),
        process_permits + slow_lane_permits,
    )?;
    let fetch_cap = fetch_cap_mb * 1024 * 1024;
    let fetch_timeout = std::time::Duration::from_secs(fetch_timeout_secs);
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
    let policy = build_source_policy(|name| env::var(name).ok())?;
    if let Some(warning) = proxy_bypass_warning(|name| env::var(name).ok(), &policy) {
        eprintln!("warning: {warning}");
    }
    let mut sources = build_source_config(|name| env::var(name).ok())?;
    // The upload destination (issue #168), built after the
    // source configuration it reuses. A local destination
    // becomes the file source root when none is configured,
    // so uploaded objects serve back without a second knob;
    // a configured root was already checked to contain it.
    let upload = build_upload_store(|name| env::var(name).ok(), &sources)?;
    if let Some(store) = &upload
        && let Some(dir) = store.local_dir()
        && sources.file_root.is_none()
    {
        sources.file_root = Some(dir.to_path_buf());
    }
    let (presets, only_presets) = build_presets(|name| env::var(name).ok())?;
    let source_roots = build_source_roots(|name| env::var(name).ok())?;
    if !source_roots.is_empty() {
        let mut names: Vec<&String> = source_roots.keys().collect();
        names.sort();
        println!(
            "source roots: {}",
            names
                .iter()
                .map(|n| n.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    // Server-wide default operations (issue #145): parsed
    // and validated now, so a bad string is a boot error
    // naming the variable, not a per-request surprise.
    let default_ops = match env::var("CRABCROP_DEFAULT_OPS") {
        Ok(text) if !text.trim().is_empty() => {
            let ops = crate::url::parse_default_ops(&text, &presets)
                .map_err(|e| format!("invalid CRABCROP_DEFAULT_OPS: {e}"))?;
            println!("default operations: {text}");
            ops
        }
        _ => Vec::new(),
    };
    let preserve_metadata = metadata_config(|name| env::var(name).ok())?;
    let brand_color = brand_color_config(|name| env::var(name).ok())?;
    if let Some(c) = brand_color {
        println!("brand color: #{:02x}{:02x}{:02x}", c[0], c[1], c[2]);
    }

    // The entry MAC key for the persistent tiers (issue
    // #190): derived from the current signing secret, so
    // entries planted by anyone without it read as misses,
    // and a secret rotation invalidates the old entries
    // rather than serving content authenticated under a
    // retired key.
    let entry_key = crate::url::entry_key(&keyring.current);
    // The configuration fingerprint (audit R2-08): every
    // cache key carries it, so the persistent tiers never
    // serve an entry rendered under different
    // output-shaping settings across a restart.
    let config_fingerprint = config_fingerprint(
        |name| env::var(name).ok(),
        preserve_metadata,
        ssim_target,
        avif_speed,
        jxl_speed,
        width_step,
    );

    // Disk result storage: configured by directory, bounded
    // by its own budget. Opening indexes whatever a previous
    // run left behind. It opens here, before the AI preload,
    // together with the listener bind: a taken port or an
    // unwritable directory is a cheap failure and should
    // not wait for the model embedding passes to be paid
    // first (bind early, serve late).
    let store = match env::var("CRABCROP_RESULT_STORAGE_DIR") {
        Ok(dir) if !dir.is_empty() => {
            let max_mb: usize = parse_env("CRABCROP_RESULT_STORAGE_MAX_MB", 512)?;
            let max_bytes = max_mb
                .checked_mul(1024 * 1024)
                .ok_or_else(|| "CRABCROP_RESULT_STORAGE_MAX_MB is too large".to_string())?;
            let store = crate::storage::DiskStore::open(
                std::path::Path::new(&dir),
                max_bytes,
                entry_key.clone(),
            )?;
            let (entries, bytes) = store.stats();
            println!(
                "result storage at {dir}: {entries} entries, {bytes} bytes, budget {max_mb} MB"
            );
            Some(Arc::new(store))
        }
        _ => None,
    };
    // Shared result storage (issue #142): one S3 bucket as
    // the multi-replica tier. It reuses the source S3
    // configuration (endpoint, region, and the AWS
    // credentials from the environment); a prefix
    // namespaces the entries.
    let shared_store = match env::var("CRABCROP_RESULT_BUCKET") {
        Ok(bucket) if !bucket.is_empty() => {
            let prefix = env::var("CRABCROP_RESULT_PREFIX").unwrap_or_default();
            let store = crate::s3store::S3Store::new(
                bucket.clone(),
                prefix,
                sources.s3_endpoint.clone(),
                sources.s3_region.clone(),
                sources.aws_creds.clone(),
                entry_key.clone(),
            )
            .map_err(|e| {
                format!(
                    "cannot build the shared result store client: {}",
                    with_causes(&e)
                )
            })?;
            println!("shared result storage: bucket {bucket}");
            Some(Arc::new(store))
        }
        _ => None,
    };
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;

    // AI models are configured, never downloaded, and preloaded
    // at boot: a model that is set but broken is a startup
    // error, not a surprise for the first request that uses it.
    let ai = build_ai_providers(|name| env::var(name).ok())?;

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

    // The fallback image and the video encoder follow the
    // same boot rule as the models: configured but broken
    // is fatal here, not at the first request that needs
    // them. Both builders read through the env lookup.
    let fallback = build_fallback(|name| env::var(name).ok())?;
    let video_encoder = build_video_encoder(|name| env::var(name).ok())?;
    let frame_extractor = build_video_decoder(|name| env::var(name).ok())?;
    let pdf_renderer = build_pdf_renderer(|name| env::var(name).ok())?;
    let provenance_signer = build_provenance_signer(|name| env::var(name).ok())?;
    let meta_enabled = meta_mode_config(|name| env::var(name).ok())?;
    if meta_enabled {
        println!("meta mode: on");
    }

    let state = AppState {
        keyring,
        allow_unsafe,
        loader: Loader::new(fetch_cap, fetch_timeout)
            .map_err(|e| format!("cannot build the loader: {e}"))?
            .with_policy(policy)
            .with_sources(sources)
            .with_source_budget(crate::loader::SourceBudget::new(
                usize::try_from(source_buffer_bytes).unwrap_or(usize::MAX),
            )),
        engine: Engine,
        process_timeout: std::time::Duration::from_secs(process_timeout_secs),
        metrics: Arc::new(Metrics::default()),
        metrics_token,
        cache: Arc::new(Cache::new(cache_bytes)),
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
        store,
        shared_store,
        upload,
        meta_enabled,
        ai_gate: Arc::new(tokio::sync::Semaphore::new(2)),
        process_gate: Arc::new(crate::server::Gate::new(process_permits)),
        slow_gate: Arc::new(crate::server::Gate::new(slow_lane_permits)),
        ssim_target,
        avif_speed,
        jxl_speed,
        width_step,
        preserve_metadata,
        brand_color,
        presets,
        only_presets,
        source_roots,
        default_ops,
        video_encoder,
        frame_extractor,
        pdf_renderer,
        provenance_signer,
        video_gate: Arc::new(tokio::sync::Semaphore::new(crate::video::GATE_PERMITS)),
        anim_gate: Arc::new(tokio::sync::Semaphore::new(1)),
        inflight: Arc::new(crate::server::Inflight::default()),
        purge_guard: Arc::new(crate::cache::PurgeGuard::new()),
        config_fingerprint,
    };
    println!(
        "source fetch: cap {fetch_cap_mb} MB, timeout {fetch_timeout_secs}s, \
         buffer budget {} MB",
        source_buffer_bytes / (1024 * 1024)
    );
    println!("configuration fingerprint: {}", state.config_fingerprint);
    // Purges recorded in the shared bucket hold here before
    // the first request (issue #461).
    crate::server::start_purge_tombstones(&state).await;
    println!("crabcrop listening on port {port}");
    axum::serve(listener, app(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl-C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
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
const RENDERING_VERSION: u32 = 3;

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

/// The operator brand color from `CRABCROP_BRAND_COLOR`
/// (issue #197): six hex digits, an optional leading `#`
/// allowed, resolved through `get` so tests can drive it
/// without the process environment. Unset means studio
/// backdrops resolve to white; a set but malformed value
/// is a startup error naming the variable.
fn brand_color_config(get: impl Fn(&str) -> Option<String>) -> Result<Option<[u8; 3]>, String> {
    let raw = match get("CRABCROP_BRAND_COLOR") {
        Some(v) if !v.is_empty() => v,
        _ => return Ok(None),
    };
    let hex = raw.strip_prefix('#').unwrap_or(&raw);
    if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "CRABCROP_BRAND_COLOR must be a hex color like f5f0e8, got {raw:?}"
        ));
    }
    let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).unwrap();
    Ok(Some([byte(0), byte(2), byte(4)]))
}

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

/// Meta mode from `CRABCROP_META_MODE`, off unless the
/// operator turns it on (issue #453). Every meta request
/// skips the cache tiers and runs a fresh fetch, render,
/// and annotation inferences, and any viewer of a signed
/// URL can ask for one with a header, so a deployment
/// opts in to that cost rather than carrying it unasked.
fn meta_mode_config(get: impl Fn(&str) -> Option<String>) -> Result<bool, String> {
    parse_bool_value("CRABCROP_META_MODE", get("CRABCROP_META_MODE"), false)
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

/// Build every AI provider from the environment, read
/// through `get` so tests can drive construction without
/// the process environment.
fn build_ai_providers(get: impl Fn(&str) -> Option<String>) -> Result<AiProviders, String> {
    let ort_dylib = get("CRABCROP_ORT_DYLIB").filter(|v| !v.is_empty());
    let removebg: Option<Arc<dyn crate::ai::MaskProvider>> = match get("CRABCROP_REMOVEBG_MODEL") {
        Some(path) if !path.is_empty() => {
            let provider = crate::ai::OnnxMaskProvider::load(ort_dylib.as_deref(), &path)
                .map_err(|e| format!("cannot load the background-removal model: {e}"))?;
            println!("background removal model loaded from {path}");
            Some(Arc::new(provider))
        }
        _ => None,
    };
    let upscale: Option<Arc<dyn crate::ai::UpscaleProvider>> = match get("CRABCROP_UPSCALE_MODEL") {
        Some(path) if !path.is_empty() => {
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
    let inpaint: Option<Arc<dyn crate::ai::InpaintProvider>> = match get("CRABCROP_INPAINT_MODEL") {
        Some(path) if !path.is_empty() => {
            let provider = crate::ai::OnnxInpaintProvider::load(ort_dylib.as_deref(), &path)
                .map_err(|e| format!("cannot load the inpainting model: {e}"))?;
            println!("inpainting model loaded from {path}");
            Some(Arc::new(provider))
        }
        _ => None,
    };
    let segment = build_segment(ort_dylib.as_deref(), &get)?;
    let (tagger, tag_threshold, embeddings) = match build_tagger(ort_dylib.as_deref(), &get)? {
        Some(setup) => (
            Some(setup.provider),
            setup.threshold,
            Some(setup.embeddings),
        ),
        None => (None, 0.05, None),
    };
    let moderation = build_moderation(ort_dylib.as_deref(), &get)?;
    let aesthetic = build_aesthetic(&get, tagger.is_some())?;
    let ocr = build_ocr(ort_dylib.as_deref(), &get)?;
    let detect = build_detector(ort_dylib.as_deref(), &get)?;
    let declared = build_declared_detectors(ort_dylib.as_deref(), &get)?;
    let face: Option<Arc<dyn crate::ai::FaceProvider>> = match get("CRABCROP_FACE_MODEL") {
        Some(path) if !path.is_empty() => {
            let provider = crate::ai::OnnxFaceProvider::load(ort_dylib.as_deref(), &path)
                .map_err(|e| format!("cannot load the face detection model: {e}"))?;
            println!("face detection model loaded from {path}");
            Some(Arc::new(provider))
        }
        _ => None,
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

/// The fallback image from the environment: an http(s)
/// value is fetched per fallback event through the loader;
/// anything else is a local path read once now, broken is
/// fatal like the rest of the boot config. The local bytes
/// are not decoded here: decode happens per fallback event,
/// through the same operations as a normal source.
fn build_fallback(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<server::Fallback>, String> {
    match get("CRABCROP_FALLBACK_IMAGE") {
        Some(v) if !v.is_empty() => {
            if v.starts_with("http://") || v.starts_with("https://") {
                println!("fallback image will be fetched from {v}");
                Ok(Some(server::Fallback::Url(v)))
            } else {
                let bytes = std::fs::read(&v)
                    .map_err(|e| format!("cannot read the fallback image {v}: {e}"))?;
                println!("fallback image loaded from {v} ({} bytes)", bytes.len());
                Ok(Some(server::Fallback::Bytes(Arc::new(bytes))))
            }
        }
        _ => Ok(None),
    }
}

/// The video encoder (issue #162): an operator-supplied
/// program, validated now, so a configured but unusable
/// path is a boot error rather than a surprise at the
/// first video request. The bounds parse unconditionally:
/// a typo'd bound is a boot error even before the
/// encoder itself is configured.
fn build_video_encoder(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<Arc<crate::video::VideoEncoder>>, String> {
    let video_timeout_secs = video_timeout_secs(&get)?;
    let video_max_mb: usize = parse_env_value(
        "CRABCROP_VIDEO_MAX_MB",
        get("CRABCROP_VIDEO_MAX_MB"),
        crate::video::DEFAULT_MAX_OUTPUT_MB,
    )?;
    if !(1..=1024).contains(&video_max_mb) {
        return Err(format!(
            "CRABCROP_VIDEO_MAX_MB must be between 1 and 1024, got {video_max_mb}"
        ));
    }
    match get("CRABCROP_VIDEO_ENCODER") {
        Some(path) if !path.is_empty() => {
            let max_bytes = video_max_mb
                .checked_mul(1024 * 1024)
                .ok_or_else(|| "CRABCROP_VIDEO_MAX_MB is too large".to_string())?;
            let encoder = crate::video::VideoEncoder::new(
                path.clone(),
                std::time::Duration::from_secs(video_timeout_secs),
                max_bytes,
            )
            .map_err(|e| format!("cannot use CRABCROP_VIDEO_ENCODER ({path}): {e}"))?;
            println!(
                "video encoder: {path} (timeout {video_timeout_secs}s, output cap {video_max_mb} MB)"
            );
            Ok(Some(Arc::new(encoder)))
        }
        _ => Ok(None),
    }
}

/// The shared wall-clock limit for one external video
/// tool run, encode or frame extraction (issue #214):
/// `CRABCROP_VIDEO_TIMEOUT_SECS`, validated the same way
/// whichever tool ends up configured.
fn video_timeout_secs(get: &impl Fn(&str) -> Option<String>) -> Result<u64, String> {
    let secs: u64 = parse_env_value(
        "CRABCROP_VIDEO_TIMEOUT_SECS",
        get("CRABCROP_VIDEO_TIMEOUT_SECS"),
        crate::video::DEFAULT_TIMEOUT_SECS,
    )?;
    if !(1..=600).contains(&secs) {
        return Err(format!(
            "CRABCROP_VIDEO_TIMEOUT_SECS must be between 1 and 600, got {secs}"
        ));
    }
    Ok(secs)
}

/// Build the video frame extractor (issue #214) from
/// `CRABCROP_VIDEO_DECODER`: the source-side counterpart
/// of the encoder. Same boot rule: a set path that is
/// not a usable executable is fatal here, not on the
/// first `frame:` request. The timeout is the shared
/// video tool budget.
fn build_video_decoder(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<Arc<crate::frames::VideoFrameExtractor>>, String> {
    let timeout_secs = video_timeout_secs(&get)?;
    match get("CRABCROP_VIDEO_DECODER") {
        Some(path) if !path.is_empty() => {
            let extractor = crate::frames::VideoFrameExtractor::new(
                std::path::PathBuf::from(&path),
                std::time::Duration::from_secs(timeout_secs),
            )
            .map_err(|e| format!("cannot use CRABCROP_VIDEO_DECODER ({path}): {e}"))?;
            println!("video decoder: {path} (timeout {timeout_secs}s)");
            Ok(Some(Arc::new(extractor)))
        }
        _ => Ok(None),
    }
}

/// The wall-clock limit for one PDF renderer run
/// (issue #215): `CRABCROP_PDF_TIMEOUT_SECS`, its own
/// budget rather than the video tools', because a page
/// render's cost profile differs from a frame
/// extraction's. Validated even when no renderer is
/// configured, like the video bounds.
fn pdf_timeout_secs(get: &impl Fn(&str) -> Option<String>) -> Result<u64, String> {
    let secs: u64 = parse_env_value(
        "CRABCROP_PDF_TIMEOUT_SECS",
        get("CRABCROP_PDF_TIMEOUT_SECS"),
        crate::video::DEFAULT_TIMEOUT_SECS,
    )?;
    if !(1..=600).contains(&secs) {
        return Err(format!(
            "CRABCROP_PDF_TIMEOUT_SECS must be between 1 and 600, got {secs}"
        ));
    }
    Ok(secs)
}

/// Build the C2PA output signer from the environment
/// (issue #173): the `CRABCROP_C2PA_CERT` /
/// `CRABCROP_C2PA_KEY` pair (a PEM certificate chain and
/// its private key), with `CRABCROP_C2PA_TSA_URL` as an
/// optional timestamp authority. The pair is
/// all-or-nothing, the CLIP rule: a partial set is a
/// startup error naming what is missing, and a timestamp
/// authority without the pair is one too. The identity is
/// proven at boot by a sign-and-verify round trip (see
/// `ProvenanceSigner::from_pem`): a broken identity is
/// fatal here, not at the first signed request.
fn build_provenance_signer(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<Arc<crate::provenance::ProvenanceSigner>>, String> {
    let cert = get("CRABCROP_C2PA_CERT").filter(|v| !v.is_empty());
    let key = get("CRABCROP_C2PA_KEY").filter(|v| !v.is_empty());
    let tsa = get("CRABCROP_C2PA_TSA_URL").filter(|v| !v.is_empty());
    match (cert, key) {
        (None, None) => {
            if tsa.is_some() {
                return Err(
                    "CRABCROP_C2PA_TSA_URL is set but no signing identity is configured: \
                     set CRABCROP_C2PA_CERT and CRABCROP_C2PA_KEY"
                        .to_string(),
                );
            }
            Ok(None)
        }
        (Some(cert), Some(key)) => {
            let signer = crate::provenance::ProvenanceSigner::from_files(&cert, &key, tsa)
                .map_err(|e| format!("cannot use the C2PA signing identity: {e}"))?;
            println!(
                "C2PA provenance signing: {} (certificate {cert})",
                signer.name
            );
            Ok(Some(Arc::new(signer)))
        }
        (None, Some(_)) => {
            Err("C2PA signing is partially configured: missing CRABCROP_C2PA_CERT".to_string())
        }
        (Some(_), None) => {
            Err("C2PA signing is partially configured: missing CRABCROP_C2PA_KEY".to_string())
        }
    }
}

/// Build the PDF renderer (issue #215) from
/// `CRABCROP_PDF_RENDERER`: the document-side
/// counterpart of the frame extractor. Same boot rule:
/// a set path that is not a usable renderer, or one
/// whose `pdfinfo` sibling is missing, is fatal here,
/// not on the first `page:` request.
fn build_pdf_renderer(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<Arc<crate::pdf::PdfRenderer>>, String> {
    let timeout_secs = pdf_timeout_secs(&get)?;
    match get("CRABCROP_PDF_RENDERER") {
        Some(path) if !path.is_empty() => {
            let renderer = crate::pdf::PdfRenderer::new(
                std::path::PathBuf::from(&path),
                std::time::Duration::from_secs(timeout_secs),
            )
            .map_err(|e| format!("cannot use CRABCROP_PDF_RENDERER ({path}): {e}"))?;
            println!("PDF renderer: {path} (timeout {timeout_secs}s)");
            Ok(Some(Arc::new(renderer)))
        }
        _ => Ok(None),
    }
}

/// Build the segment provider from the environment: the
/// `CRABCROP_SEGMENT_*_MODEL` pair (encoder plus prompt
/// decoder, issue #175), preloaded like every other
/// model. The pair is all-or-nothing, the CLIP rule: a
/// partial set is a startup error naming what is missing.
fn build_segment(
    ort_dylib: Option<&str>,
    get: &impl Fn(&str) -> Option<String>,
) -> Result<Option<Arc<dyn crate::ai::SegmentProvider>>, String> {
    let encoder = get("CRABCROP_SEGMENT_ENCODER_MODEL").filter(|v| !v.is_empty());
    let decoder = get("CRABCROP_SEGMENT_DECODER_MODEL").filter(|v| !v.is_empty());
    match (encoder, decoder) {
        (None, None) => Ok(None),
        (Some(encoder), Some(decoder)) => {
            let provider = crate::ai::OnnxSegmentProvider::load(ort_dylib, &encoder, &decoder)
                .map_err(|e| format!("cannot load the segment model pair: {e}"))?;
            println!("segment models loaded from {encoder} (encoder) and {decoder} (decoder)");
            Ok(Some(Arc::new(provider)))
        }
        (None, Some(_)) => Err(
            "segmentation is partially configured: missing CRABCROP_SEGMENT_ENCODER_MODEL"
                .to_string(),
        ),
        (Some(_), None) => Err(
            "segmentation is partially configured: missing CRABCROP_SEGMENT_DECODER_MODEL"
                .to_string(),
        ),
    }
}

/// Build the detection provider from the environment,
/// when `CRABCROP_DETECT_MODEL` is set: preloaded like
/// every other model, broken configuration fatal. The
/// label file is only a fallback for exports that carry
/// no class-names metadata.
fn build_detector(
    ort_dylib: Option<&str>,
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<Arc<dyn crate::ai::DetectionProvider>>, String> {
    let path = match get("CRABCROP_DETECT_MODEL") {
        Some(path) if !path.is_empty() => path,
        _ => return Ok(None),
    };
    let labels = get("CRABCROP_DETECT_LABELS").filter(|v| !v.is_empty());
    let threshold = match get("CRABCROP_DETECT_THRESHOLD") {
        Some(raw) if !raw.is_empty() => {
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
        crate::ai::DetectionProvider::describe(&provider).len()
    );
    Ok(Some(Arc::new(provider)))
}

/// The contract a declared detector binds to (issue #166).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DetectorKind {
    /// A YOLO-family detector: joins class redaction and
    /// meta detections through `DetectionProvider`.
    Boxes,
    /// A heatmap model: feeds smart crops through
    /// `FocalProvider`.
    Focal,
}

/// Where a `boxes` declaration's class vocabulary comes
/// from when the declaration carries one.
#[derive(Debug, Clone, PartialEq)]
enum LabelsDecl {
    /// A path to a labels file (one class per line), like
    /// `CRABCROP_DETECT_LABELS`.
    Path(String),
    /// Class names written inline in the declaration.
    Inline(Vec<String>),
}

/// One validated detector declaration from
/// `CRABCROP_DETECTORS_FILE`.
#[derive(Debug, Clone, PartialEq)]
struct DetectorDeclaration {
    name: String,
    model: String,
    kind: DetectorKind,
    labels: Option<LabelsDecl>,
    threshold: Option<f32>,
}

/// The detectors file's size cap: declarations are small
/// JSON, and the file is read whole at boot (audit R2-27).
const MAX_DETECTORS_FILE_BYTES: usize = 1024 * 1024;

/// The declaration count cap: every declaration loads and
/// probes a full model at boot, and every declared `boxes`
/// detector runs under the one AI permit of a meta request,
/// so the count bounds both the boot cost and the meta
/// fan-out (audit R2-27).
const MAX_DETECTOR_DECLARATIONS: usize = 16;

/// The class vocabulary cap per declaration, applied to
/// inline labels at parse time and to a labels file's
/// resolved classes at build time (audit R2-27).
const MAX_DETECTOR_LABELS: usize = 4096;

/// The length cap for one inline class label (audit R2-27).
const MAX_DETECTOR_LABEL_CHARS: usize = 256;

/// Parse the detectors file's text into declarations,
/// strictly: this runs before any model is touched, so
/// every refusal here is raisable (and testable) without
/// the ONNX runtime. Unknown fields, kinds, and
/// misapplied options are errors naming the declaration.
fn parse_detector_declarations(text: &str) -> Result<Vec<DetectorDeclaration>, String> {
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| format!("CRABCROP_DETECTORS_FILE is not valid JSON: {e}"))?;
    let entries = value.as_array().ok_or_else(|| {
        "CRABCROP_DETECTORS_FILE must hold a JSON array of detector declarations".to_string()
    })?;
    if entries.len() > MAX_DETECTOR_DECLARATIONS {
        return Err(format!(
            "CRABCROP_DETECTORS_FILE declares {} detectors; the cap is {MAX_DETECTOR_DECLARATIONS}",
            entries.len()
        ));
    }
    let mut declarations: Vec<DetectorDeclaration> = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        let at = format!("detector declaration {}", i + 1);
        let obj = entry
            .as_object()
            .ok_or_else(|| format!("{at} is not a JSON object"))?;
        let name = match obj.get("name").and_then(serde_json::Value::as_str) {
            Some(name) if !name.is_empty() => name.to_string(),
            _ => return Err(format!("{at}: \"name\" must be a non-empty string")),
        };
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!("invalid detector name: {name:?}"));
        }
        // `builtin` is the source name meta detections use
        // for the built-in detector's boxes: a declaration
        // taking it would make its boxes indistinguishable
        // from the built-in's (audit R2-27).
        if name == "builtin" {
            return Err(
                "detector name \"builtin\" is reserved for the built-in detector".to_string(),
            );
        }
        if declarations.iter().any(|d| d.name == name) {
            return Err(format!("duplicate detector name: {name}"));
        }
        let named = format!("detector {name:?}");
        for key in obj.keys() {
            if !["name", "model", "kind", "labels", "threshold"].contains(&key.as_str()) {
                return Err(format!("{named}: unknown field {key:?}"));
            }
        }
        let model = match obj.get("model").and_then(serde_json::Value::as_str) {
            Some(model) if !model.is_empty() => model.to_string(),
            _ => {
                return Err(format!(
                    "{named}: \"model\" must be a non-empty path string"
                ));
            }
        };
        let kind = match obj.get("kind").and_then(serde_json::Value::as_str) {
            Some("boxes") => DetectorKind::Boxes,
            Some("focal") => DetectorKind::Focal,
            Some(other) => {
                return Err(format!(
                    "{named}: unknown kind {other:?} (expected \"boxes\" or \"focal\")"
                ));
            }
            None => {
                return Err(format!("{named}: \"kind\" must be \"boxes\" or \"focal\""));
            }
        };
        let labels = match obj.get("labels") {
            None => None,
            Some(_) if kind == DetectorKind::Focal => {
                return Err(format!(
                    "{named}: \"labels\" applies only to \"boxes\" detectors"
                ));
            }
            Some(serde_json::Value::String(path)) if !path.is_empty() => {
                Some(LabelsDecl::Path(path.clone()))
            }
            Some(serde_json::Value::Array(items)) => {
                if items.is_empty() {
                    return Err(format!("{named}: \"labels\" must name at least one class"));
                }
                if items.len() > MAX_DETECTOR_LABELS {
                    return Err(format!(
                        "{named}: \"labels\" names {} classes; the cap is {MAX_DETECTOR_LABELS}",
                        items.len()
                    ));
                }
                let mut names = Vec::new();
                for item in items {
                    match item.as_str() {
                        Some(class) if !class.trim().is_empty() => {
                            if class.chars().count() > MAX_DETECTOR_LABEL_CHARS {
                                return Err(format!(
                                    "{named}: \"labels\" entries must be at most \
                                     {MAX_DETECTOR_LABEL_CHARS} characters"
                                ));
                            }
                            names.push(class.to_string());
                        }
                        _ => {
                            return Err(format!(
                                "{named}: \"labels\" entries must be non-empty strings"
                            ));
                        }
                    }
                }
                Some(LabelsDecl::Inline(names))
            }
            Some(_) => {
                return Err(format!(
                    "{named}: \"labels\" must be a file path string or an array of class names"
                ));
            }
        };
        let threshold = match obj.get("threshold") {
            None => None,
            Some(_) if kind == DetectorKind::Focal => {
                return Err(format!(
                    "{named}: \"threshold\" applies only to \"boxes\" detectors"
                ));
            }
            Some(value) => match value.as_f64() {
                Some(v) if (0.0..=1.0).contains(&v) => Some(v as f32),
                Some(v) => {
                    return Err(format!(
                        "{named}: \"threshold\" must be within 0..=1, got {v}"
                    ));
                }
                None => {
                    return Err(format!("{named}: \"threshold\" must be a number"));
                }
            },
        };
        declarations.push(DetectorDeclaration {
            name,
            model,
            kind,
            labels,
            threshold,
        });
    }
    Ok(declarations)
}

/// The declared detectors, built: every `boxes` provider
/// in declaration order, and the one active focal
/// provider (the first `focal` declaration).
struct DeclaredDetectors {
    boxes: Vec<server::NamedDetector>,
    focal: Option<Arc<dyn crate::ai::FocalProvider>>,
}

/// Build the declared detectors from
/// `CRABCROP_DETECTORS_FILE`, when set (issue #166). The
/// file only names models and options: each declaration
/// is loaded and probed against its built-in contract
/// here, exactly like the configured models, and any
/// failure is a startup error naming the declaration.
/// Additional `focal` declarations are loaded and probed
/// the same way, then stand by: only the first feeds
/// smart crops.
fn build_declared_detectors(
    ort_dylib: Option<&str>,
    get: impl Fn(&str) -> Option<String>,
) -> Result<DeclaredDetectors, String> {
    let path = match get("CRABCROP_DETECTORS_FILE") {
        Some(path) if !path.is_empty() => path,
        _ => {
            return Ok(DeclaredDetectors {
                boxes: Vec::new(),
                focal: None,
            });
        }
    };
    // The file is read under a size cap, like the cache
    // entry parser's: declarations are small JSON, and an
    // unbounded read at boot is an unbounded allocation
    // (audit R2-27).
    use std::io::Read as _;
    let mut file = std::fs::File::open(&path)
        .map_err(|e| format!("cannot read CRABCROP_DETECTORS_FILE ({path}): {e}"))?;
    let mut bytes = Vec::new();
    file.by_ref()
        .take((MAX_DETECTORS_FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read CRABCROP_DETECTORS_FILE ({path}): {e}"))?;
    if bytes.len() > MAX_DETECTORS_FILE_BYTES {
        return Err(format!(
            "CRABCROP_DETECTORS_FILE ({path}) exceeds the {MAX_DETECTORS_FILE_BYTES} byte cap"
        ));
    }
    let text = String::from_utf8(bytes)
        .map_err(|e| format!("CRABCROP_DETECTORS_FILE ({path}) is not valid UTF-8: {e}"))?;
    let declarations = parse_detector_declarations(&text)?;
    let mut boxes = Vec::new();
    let mut focal: Option<Arc<dyn crate::ai::FocalProvider>> = None;
    let mut active_focal: Option<String> = None;
    for decl in &declarations {
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
                            format!(
                                "cannot load declared detector {:?} (boxes) from {}: {e}",
                                decl.name, decl.model
                            )
                        })?)
                    }
                    None => None,
                };
                // A labels file's vocabulary is held to the
                // same cap the parser puts on inline labels
                // (audit R2-27).
                let label_count = declared_classes.as_ref().map_or(0, Vec::len);
                if label_count > MAX_DETECTOR_LABELS {
                    return Err(format!(
                        "declared detector {:?}: the labels name {label_count} classes; the cap \
                         is {MAX_DETECTOR_LABELS}",
                        decl.name
                    ));
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
                    format!(
                        "cannot load declared detector {:?} (boxes) from {}: {e}",
                        decl.name, decl.model
                    )
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
                        format!(
                            "cannot load declared detector {:?} (focal) from {}: {e}",
                            decl.name, decl.model
                        )
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

/// Build the CLIP tag provider from the environment, when
/// configured: preloaded like every other model, with a
/// broken configuration failing startup.
fn build_tagger(
    ort_dylib: Option<&str>,
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<TaggerSetup>, String> {
    let config = clip_config(get)?;
    match config {
        None => Ok(None),
        Some(config) => {
            let provider = crate::ai::OnnxClipProvider::load(
                ort_dylib,
                &config.image_model,
                &config.text_model,
                &config.tokenizer,
                &config.labels,
            )
            .map_err(|e| format!("cannot load the CLIP tagging models: {e}"))?;
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
    }
}

/// Build the moderation provider from the environment,
/// when configured (issue #171): a verdict classifier
/// from `CRABCROP_MODERATION_MODEL`, or the CLIP
/// zero-shot fallback when only a labels file is set.
/// Preloaded like every other provider, with a broken
/// configuration failing startup.
fn build_moderation(
    ort_dylib: Option<&str>,
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<Arc<dyn crate::ai::ModerationProvider>>, String> {
    let model = get("CRABCROP_MODERATION_MODEL").filter(|v| !v.is_empty());
    let labels_path = get("CRABCROP_MODERATION_LABELS").filter(|v| !v.is_empty());
    if model.is_none() && labels_path.is_none() {
        return Ok(None);
    }
    if let Some(model) = model {
        // With a model, the labels file names its output
        // categories; the provider checks the count
        // against its probe at load.
        let labels = match &labels_path {
            Some(path) => Some(
                crate::ai::read_labels(path)
                    .map_err(|e| format!("cannot read the moderation labels: {e}"))?,
            ),
            None => None,
        };
        let provider = crate::ai::OnnxModerationProvider::load(ort_dylib, &model, labels)
            .map_err(|e| format!("cannot load the moderation model: {e}"))?;
        println!(
            "moderation verdicts loaded from {model} ({} categories)",
            provider.labels().len()
        );
        return Ok(Some(Arc::new(provider)));
    }
    // Labels without a model are zero-shot prompts for
    // the CLIP fallback, which needs the CLIP set: the
    // same model files load again with the moderation
    // labels, so the two label sets stay independent.
    let labels_path = labels_path.unwrap_or_default();
    let clip = clip_config(get)?.ok_or_else(|| {
        "CRABCROP_MODERATION_LABELS is set without CRABCROP_MODERATION_MODEL: the labels then name zero-shot prompts, which need the CRABCROP_CLIP_* set".to_string()
    })?;
    let tagger = crate::ai::OnnxClipProvider::load(
        ort_dylib,
        &clip.image_model,
        &clip.text_model,
        &clip.tokenizer,
        &labels_path,
    )
    .map_err(|e| format!("cannot load the CLIP moderation fallback: {e}"))?;
    println!(
        "moderation review signals loaded from {} ({} prompt labels)",
        clip.image_model,
        tagger.label_count()
    );
    let provider =
        crate::ai::ClipModerationProvider::new(Arc::new(tagger), clip.image_model.clone());
    Ok(Some(Arc::new(provider)))
}

/// Build the aesthetic head from the environment, when
/// configured (issue #172): an operator-supplied LAION
/// linear head, scored against the CLIP tagger's
/// embedding in meta. The head's input is the tagger's
/// embedding, so a head without the CLIP set is a
/// startup error, and a malformed head file fails its
/// load here, at boot, like every other configured
/// artifact. A dimension mismatch between the head and
/// the encoder is NOT an error: meta reports it per
/// request, fail-soft (see `AestheticHead::score`).
fn build_aesthetic(
    get: impl Fn(&str) -> Option<String>,
    tagger_configured: bool,
) -> Result<Option<Arc<crate::ai::AestheticHead>>, String> {
    let path = match get("CRABCROP_AESTHETIC_HEAD").filter(|v| !v.is_empty()) {
        Some(path) => path,
        None => return Ok(None),
    };
    if !tagger_configured {
        return Err(
            "CRABCROP_AESTHETIC_HEAD is set but the CRABCROP_CLIP_* set is not configured: the head scores the CLIP tagger's embedding, so an encoder must be configured"
                .to_string(),
        );
    }
    let head = crate::ai::AestheticHead::load(&path)
        .map_err(|e| format!("cannot load the aesthetic head: {e}"))?;
    println!(
        "aesthetic head loaded from {path} (expects a {}-dimensional embedding)",
        head.expected_dim()
    );
    Ok(Some(Arc::new(head)))
}

/// Build the OCR provider from the environment, when
/// configured (issue #177): a PP-OCR-family detection
/// and recognition model pair plus the recognition
/// model's character dictionary. The three paths are
/// all-or-nothing, the segment pair's rule: a partial
/// set is a startup error naming what is missing,
/// because half a pipeline reads nothing and a
/// dictionary without its model aligns with nothing.
/// `CRABCROP_OCR_LANGUAGE` optionally declares the
/// language the recognition model serves (a BCP-47
/// tag), echoed in meta so the response describes
/// itself; unset reports `und`. Preloaded and probed
/// like every other provider.
fn build_ocr(
    ort_dylib: Option<&str>,
    get: &impl Fn(&str) -> Option<String>,
) -> Result<Option<Arc<dyn crate::ai::OcrProvider>>, String> {
    let detect = get("CRABCROP_OCR_DETECT_MODEL").filter(|v| !v.is_empty());
    let recognize = get("CRABCROP_OCR_RECOGNIZE_MODEL").filter(|v| !v.is_empty());
    let dict = get("CRABCROP_OCR_DICT").filter(|v| !v.is_empty());
    let missing: Vec<&str> = [
        ("CRABCROP_OCR_DETECT_MODEL", detect.is_none()),
        ("CRABCROP_OCR_RECOGNIZE_MODEL", recognize.is_none()),
        ("CRABCROP_OCR_DICT", dict.is_none()),
    ]
    .iter()
    .filter(|(_, missing)| *missing)
    .map(|(name, _)| *name)
    .collect();
    if missing.len() == 3 {
        return Ok(None);
    }
    if !missing.is_empty() {
        return Err(format!(
            "OCR is partially configured: missing {}",
            missing.join(", ")
        ));
    }
    let (detect, recognize, dict) = (detect.unwrap(), recognize.unwrap(), dict.unwrap());
    let language = get("CRABCROP_OCR_LANGUAGE")
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "und".to_string());
    let provider =
        crate::ai::OnnxOcrProvider::load(ort_dylib, &detect, &recognize, &dict, &language)
            .map_err(|e| format!("cannot load the OCR models: {e}"))?;
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

    #[test]
    fn brand_color_config_parses_hex_and_rejects_the_rest() {
        let none = |_: &str| None;
        assert_eq!(brand_color_config(none), Ok(None));
        let plain = |name: &str| (name == "CRABCROP_BRAND_COLOR").then(|| "f5f0e8".to_string());
        assert_eq!(brand_color_config(plain), Ok(Some([0xf5, 0xf0, 0xe8])));
        // A leading # and uppercase hex are the same color.
        let hashed = |name: &str| (name == "CRABCROP_BRAND_COLOR").then(|| "#F5F0E8".to_string());
        assert_eq!(brand_color_config(hashed), Ok(Some([0xf5, 0xf0, 0xe8])));
        // Set but malformed is a startup error naming the
        // variable, never a silent default.
        for bad in ["f5f0e", "f5f0e81", "zzzzzz", "white"] {
            let get = |name: &str| (name == "CRABCROP_BRAND_COLOR").then(|| bad.to_string());
            let err = brand_color_config(get).unwrap_err();
            assert!(err.contains("CRABCROP_BRAND_COLOR"), "{err}");
        }
        // An empty value is unset, like the other settings.
        let empty = |name: &str| (name == "CRABCROP_BRAND_COLOR").then(String::new);
        assert_eq!(brand_color_config(empty), Ok(None));
    }

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
    fn meta_mode_defaults_to_off_and_turns_on_by_the_boolean_spellings() {
        // Issue #453: unset is off.
        assert_eq!(meta_mode_config(env_of(&[])), Ok(false));
        let set = |v: &str| meta_mode_config(env_of(&[("CRABCROP_META_MODE", v)]));
        for v in ["1", "true", "YES", "on"] {
            assert_eq!(set(v), Ok(true), "{v}");
        }
        for v in ["0", "false", "no", "OFF"] {
            assert_eq!(set(v), Ok(false), "{v}");
        }
        // A typo, an empty value, or a stray space is a boot
        // error naming the variable, never a silent default.
        for bad in ["", " 1", "enabled", "2", "on\n"] {
            let err = expect_err(set(bad));
            assert!(err.contains("CRABCROP_META_MODE"), "{bad:?}: {err}");
        }
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
    fn source_policy_blocks_private_addresses_by_default() {
        let get = |pairs: &[(&str, &str)]| {
            let map: std::collections::HashMap<String, String> = pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            move |name: &str| map.get(name).cloned()
        };
        // No configuration: no allowlist, blocking on. Boot
        // needs no posture setting.
        let policy = build_source_policy(get(&[])).unwrap();
        assert!(policy.allowed_suffixes.is_empty());
        assert!(policy.block_private);
        // The explicit switch turns blocking off.
        let policy = build_source_policy(get(&[("CRABCROP_BLOCK_PRIVATE_SOURCES", "0")])).unwrap();
        assert!(!policy.block_private);
        // The any-source opt-out turns blocking off even
        // with the switch on.
        let policy = build_source_policy(get(&[
            ("CRABCROP_BLOCK_PRIVATE_SOURCES", "1"),
            ("CRABCROP_ALLOW_ANY_SOURCE", "1"),
        ]))
        .unwrap();
        assert!(!policy.block_private);
        // The allowlist passes through normalized, and blank
        // entries normalize away.
        let policy = build_source_policy(get(&[(
            "CRABCROP_ALLOWED_SOURCES",
            " Example.com , ,127.0.0.1 ",
        )]))
        .unwrap();
        assert_eq!(policy.allowed_suffixes, ["example.com", "127.0.0.1"]);
        assert!(policy.block_private);
        // A bad boolean is a startup error naming the
        // variable, never a silent default.
        let err =
            build_source_policy(get(&[("CRABCROP_BLOCK_PRIVATE_SOURCES", "ture")])).unwrap_err();
        assert!(err.contains("CRABCROP_BLOCK_PRIVATE_SOURCES"), "{err}");
    }

    #[test]
    fn a_proxy_that_pinned_fetches_bypass_is_warned_about() {
        let proxied = |name: &'static str| {
            move |n: &str| (n == name).then(|| "http://proxy.internal:3128".to_string())
        };
        // The default policy pins every source hostname to
        // its validated address (issue #471): a proxy set
        // alongside it is named in a warning.
        let default_policy = build_source_policy(env_of(&[])).unwrap();
        let warning = proxy_bypass_warning(
            env_of(&[("HTTPS_PROXY", "http://user:hunter2@proxy.internal:3128")]),
            &default_policy,
        )
        .expect("a warning under the default policy");
        assert!(warning.contains("(HTTPS_PROXY)"), "{warning}");
        assert!(warning.contains("CRABCROP_ALLOWED_SOURCES"), "{warning}");
        // The value is never echoed: a proxy URL can carry
        // credentials.
        assert!(!warning.contains("hunter2"), "{warning}");
        assert!(!warning.contains("proxy.internal"), "{warning}");
        // Every spelling the HTTP client reads counts.
        for name in [
            "HTTPS_PROXY",
            "https_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "ALL_PROXY",
            "all_proxy",
        ] {
            let warning = proxy_bypass_warning(proxied(name), &default_policy)
                .unwrap_or_else(|| panic!("no warning for {name}"));
            assert!(warning.contains(&format!("({name})")), "{warning}");
        }
        let warning = proxy_bypass_warning(
            env_of(&[("HTTPS_PROXY", "http://p:1"), ("all_proxy", "http://p:1")]),
            &default_policy,
        )
        .unwrap();
        assert!(warning.contains("(HTTPS_PROXY, all_proxy)"), "{warning}");
        // No proxy is nothing to warn about, and neither is
        // an empty one: the client reads it as none, and an
        // empty uppercase spelling hides the lowercase one.
        for pairs in [
            &[][..],
            &[("HTTPS_PROXY", "")],
            &[("HTTPS_PROXY", ""), ("https_proxy", "http://p:1")],
            &[("NO_PROXY", "*")],
        ] {
            assert!(
                proxy_bypass_warning(env_of(pairs), &default_policy).is_none(),
                "{pairs:?}"
            );
        }
        // Nothing is pinned with an allowlist (every host
        // fetched is listed, and listed hosts are exempt),
        // with blocking off, or with the any-source opt-out:
        // every fetch then honours the proxy.
        for pairs in [
            &[("CRABCROP_ALLOWED_SOURCES", "images.example.com")][..],
            &[("CRABCROP_BLOCK_PRIVATE_SOURCES", "0")],
            &[("CRABCROP_ALLOW_ANY_SOURCE", "1")],
        ] {
            let policy = build_source_policy(env_of(pairs)).unwrap();
            assert!(!policy.pins_hostnames(), "{pairs:?}");
            assert!(
                proxy_bypass_warning(proxied("HTTPS_PROXY"), &policy).is_none(),
                "{pairs:?}"
            );
        }
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
    fn parse_bool_value_accepts_the_spellings_and_rejects_typos() {
        for v in ["1", "true", "TRUE", "yes", "Yes", "on", "ON"] {
            assert!(
                parse_bool_value("CRABCROP_ALLOW_UNSAFE", Some(v.to_string()), false).unwrap(),
                "{v}"
            );
        }
        for v in ["0", "false", "False", "no", "off", "OFF"] {
            assert!(
                !parse_bool_value("CRABCROP_ALLOW_UNSAFE", Some(v.to_string()), true).unwrap(),
                "{v}"
            );
        }
        assert!(!parse_bool_value("X", None, false).unwrap());
        assert!(parse_bool_value("X", None, true).unwrap());
        // A typo is a boot error naming the variable, not a
        // silent false.
        let err =
            parse_bool_value("CRABCROP_ALLOW_UNSAFE", Some("ture".to_string()), false).unwrap_err();
        assert!(err.contains("CRABCROP_ALLOW_UNSAFE"), "{err}");
        assert!(err.contains("ture"), "{err}");
    }

    /// An environment lookup over literal pairs: the
    /// construction tests drive the `build_*` functions
    /// through this instead of the process environment,
    /// so parallel tests cannot race each other.
    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: std::collections::HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name: &str| map.get(name).cloned()
    }

    /// The startup error a construction produced, or a
    /// panic: every refusal below is pinned by its message.
    fn expect_err<T>(result: Result<T, String>) -> String {
        match result {
            Err(e) => e,
            Ok(_) => panic!("expected a startup error, but construction succeeded"),
        }
    }

    /// A temp file with the bytes a construction test gave
    /// it, in a scratch directory of its own that lives as
    /// long as this does.
    struct TempFile {
        path: String,
        _dir: crate::test_support::ScratchDir,
    }

    impl TempFile {
        fn as_str(&self) -> &str {
            &self.path
        }
    }

    impl AsRef<std::path::Path> for TempFile {
        fn as_ref(&self) -> &std::path::Path {
            self.path.as_ref()
        }
    }

    /// Both formats print the path alone, as the `String`
    /// this replaced did: the detector tests quote it into
    /// JSON with `{file:?}` and splice it in with `{file}`.
    impl std::fmt::Debug for TempFile {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            std::fmt::Debug::fmt(&self.path, f)
        }
    }

    impl std::fmt::Display for TempFile {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.path)
        }
    }

    /// Write bytes to a fresh temp file: construction tests
    /// need real files (a presets file, a fallback image, an
    /// encoder program) without committing fixtures. Each
    /// starts in an empty directory, so no test sees a file
    /// an earlier test process left (issue #526). The file
    /// is named after its tag too, so two files that must
    /// differ only in content (the fingerprint tests) differ
    /// in more than their directory.
    fn temp_file(tag: &str, contents: &[u8]) -> TempFile {
        let dir = crate::test_support::scratch_dir(&format!("boot-{tag}"));
        let path = dir
            .join(format!("{tag}.tmp"))
            .to_string_lossy()
            .into_owned();
        std::fs::write(&path, contents).unwrap();
        TempFile { path, _dir: dir }
    }

    /// A path that does not exist: a name inside a scratch
    /// directory that is removed again before this returns.
    fn missing_path(tag: &str) -> String {
        crate::test_support::scratch_dir(&format!("boot-missing-{tag}"))
            .join("missing.tmp")
            .to_string_lossy()
            .into_owned()
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
    fn keyring_construction_requires_a_secret_of_at_least_16_bytes() {
        // Unset: a hard error naming the variable.
        let err = expect_err(build_keyring(env_of(&[])));
        assert_eq!(err, "CRABCROP_SECRET must be set");
        // Below the minimum (security audit PL-3): refused.
        let err = expect_err(build_keyring(env_of(&[("CRABCROP_SECRET", "too-short")])));
        assert!(
            err.contains("CRABCROP_SECRET must be at least 16 bytes"),
            "{err}"
        );
        // Valid: the key ID defaults to `current`, no window.
        let keyring =
            build_keyring(env_of(&[("CRABCROP_SECRET", "a production-grade secret")])).unwrap();
        assert_eq!(keyring.current_id, "current");
        assert!(keyring.previous.is_none());
    }

    #[test]
    fn keyring_construction_opens_a_rotation_window() {
        let keyring = build_keyring(env_of(&[
            ("CRABCROP_SECRET", "the current production secret"),
            ("CRABCROP_SECRET_KEY_ID", "2026-10"),
            ("CRABCROP_SECRET_PREVIOUS", "the outgoing production secret"),
            ("CRABCROP_SECRET_PREVIOUS_KEY_ID", "2026-06"),
        ]))
        .unwrap();
        assert_eq!(keyring.current_id, "2026-10");
        let (id, _) = keyring.previous.as_ref().unwrap();
        assert_eq!(id, "2026-06");
        // The previous key ID defaults to `previous`.
        let keyring = build_keyring(env_of(&[
            ("CRABCROP_SECRET", "the current production secret"),
            ("CRABCROP_SECRET_PREVIOUS", "the outgoing production secret"),
        ]))
        .unwrap();
        assert_eq!(keyring.previous.as_ref().unwrap().0, "previous");
        // The previous secret is held to the same minimum.
        let err = expect_err(build_keyring(env_of(&[
            ("CRABCROP_SECRET", "the current production secret"),
            ("CRABCROP_SECRET_PREVIOUS", "short"),
        ])));
        assert!(
            err.contains("CRABCROP_SECRET_PREVIOUS must be at least 16 bytes"),
            "{err}"
        );
        // The two IDs must differ, or rotation routing is ambiguous.
        let err = expect_err(build_keyring(env_of(&[
            ("CRABCROP_SECRET", "the current production secret"),
            ("CRABCROP_SECRET_PREVIOUS", "the outgoing production secret"),
            ("CRABCROP_SECRET_PREVIOUS_KEY_ID", "current"),
        ])));
        assert!(err.contains("must differ"), "{err}");
        // A malformed key ID is refused, naming the variable.
        let err = expect_err(build_keyring(env_of(&[
            ("CRABCROP_SECRET", "the current production secret"),
            ("CRABCROP_SECRET_KEY_ID", "not a key id!"),
        ])));
        assert!(err.contains("CRABCROP_SECRET_KEY_ID"), "{err}");
    }

    #[test]
    fn source_config_construction_enforces_pairing_and_the_bucket_policy() {
        // Half a credential pair is a startup error.
        let err = expect_err(build_source_config(env_of(&[(
            "AWS_ACCESS_KEY_ID",
            "AKID",
        )])));
        assert!(err.contains("must be set together"), "{err}");
        let err = expect_err(build_source_config(env_of(&[(
            "CRABCROP_GCS_SECRET_KEY",
            "s",
        )])));
        assert!(err.contains("must be set together"), "{err}");
        // Full credentials with no allowlist: fail closed
        // (security audit L-1) at construction time.
        let err = expect_err(build_source_config(env_of(&[
            ("AWS_ACCESS_KEY_ID", "AKID"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
        ])));
        assert!(err.contains("CRABCROP_ALLOWED_BUCKETS"), "{err}");
        // Credentials plus an allowlist construct fine, and
        // the session token is attached to the pair.
        let sources = build_source_config(env_of(&[
            ("AWS_ACCESS_KEY_ID", "AKID"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("AWS_SESSION_TOKEN", "token"),
            ("CRABCROP_ALLOWED_BUCKETS", "photos, backups"),
        ]))
        .unwrap();
        assert_eq!(sources.allowed_buckets, ["photos", "backups"]);
        let creds = sources.aws_creds.as_ref().unwrap();
        assert_eq!(creds.session_token.as_deref(), Some("token"));
        // Nothing configured also constructs: unsigned
        // fetches, every bucket allowed.
        let sources = build_source_config(env_of(&[])).unwrap();
        assert!(sources.aws_creds.is_none() && sources.allowed_buckets.is_empty());
    }

    #[test]
    fn source_config_construction_validates_endpoints_and_the_file_root() {
        // Both store endpoints take the same check (issue
        // #455: the GCS one used to boot with any scheme):
        // a non-http(s) URL is refused, naming the variable,
        // and so is a value that is not a URL at all.
        for name in ["CRABCROP_S3_ENDPOINT", "CRABCROP_GCS_ENDPOINT"] {
            for bad in [
                "ftp://storage.example",
                "file:///etc/passwd",
                "gs://bucket",
                "s3://bucket",
                "data:text/plain,x",
                "javascript:alert(1)",
                "unix:/var/run/store.sock",
            ] {
                let err = expect_err(build_source_config(env_of(&[(name, bad)])));
                assert_eq!(err, format!("{name} must be an http(s) URL"), "{bad}");
            }
            for bad in [
                "storage.example",
                "//storage.example",
                "http://",
                "http://[::1",
                " ",
            ] {
                let err = expect_err(build_source_config(env_of(&[(name, bad)])));
                assert!(
                    err.starts_with(&format!("invalid {name}: ")),
                    "{bad}: {err}"
                );
            }
        }
        // A valid endpoint is stored with the trailing
        // slash trimmed.
        let sources = build_source_config(env_of(&[
            ("CRABCROP_S3_ENDPOINT", "http://127.0.0.1:9000/"),
            ("CRABCROP_GCS_ENDPOINT", "https://gcs.example/"),
        ]))
        .unwrap();
        assert_eq!(
            sources.s3_endpoint.as_deref(),
            Some("http://127.0.0.1:9000")
        );
        assert_eq!(sources.gcs_endpoint, "https://gcs.example");
        // A file root that does not exist is refused...
        let missing = missing_path("root");
        let err = expect_err(build_source_config(env_of(&[(
            "CRABCROP_FILE_ROOT",
            missing.as_str(),
        )])));
        assert!(err.contains("CRABCROP_FILE_ROOT is not usable"), "{err}");
        // ...as is one that is not a directory.
        let file = temp_file("root-file", b"not a directory");
        let err = expect_err(build_source_config(env_of(&[(
            "CRABCROP_FILE_ROOT",
            file.as_str(),
        )])));
        assert!(err.contains("is not a directory"), "{err}");
        // A real directory is canonicalized into the config.
        let dir = std::env::temp_dir().to_string_lossy().into_owned();
        let sources = build_source_config(env_of(&[("CRABCROP_FILE_ROOT", dir.as_str())])).unwrap();
        assert!(sources.file_root.is_some());
    }

    #[test]
    fn upload_store_construction_follows_the_destination_rules() {
        let plain = SourceConfig::default();
        // Built where the temp dir is, never at a fixed
        // /tmp: TMPDIR moves it (issue #526).
        let scratch = crate::test_support::scratch_dir("boot-upload");
        let root_dir = scratch.join("root");
        std::fs::create_dir(&root_dir).unwrap();
        let root = root_dir.to_string_lossy().into_owned();
        let uploads = root_dir.join("uploads").to_string_lossy().into_owned();
        // Its name starts with the root's, and it is not
        // under the root.
        let sibling = scratch.join("root-other").to_string_lossy().into_owned();
        // Neither destination: uploads are off, not an error.
        assert!(build_upload_store(env_of(&[]), &plain).unwrap().is_none());
        // Both destinations: a startup error.
        let err = expect_err(build_upload_store(
            env_of(&[
                ("CRABCROP_UPLOAD_DIR", uploads.as_str()),
                ("CRABCROP_UPLOAD_BUCKET", "uploads"),
            ]),
            &plain,
        ));
        assert!(err.contains("exactly one upload destination"), "{err}");
        // A local destination constructs with the defaults:
        // 10 MB cap (the source cap), overwrite off.
        let store =
            build_upload_store(env_of(&[("CRABCROP_UPLOAD_DIR", uploads.as_str())]), &plain)
                .unwrap()
                .unwrap();
        assert_eq!(store.max_bytes, 10 * 1024 * 1024);
        assert!(!store.allow_overwrite);
        assert!(store.local_dir().is_some());
        // The cap may only move down from the source cap.
        let err = expect_err(build_upload_store(
            env_of(&[
                ("CRABCROP_UPLOAD_DIR", uploads.as_str()),
                ("CRABCROP_UPLOAD_MAX_MB", "11"),
            ]),
            &plain,
        ));
        assert!(err.contains("CRABCROP_UPLOAD_MAX_MB"), "{err}");
        let store = build_upload_store(
            env_of(&[
                ("CRABCROP_UPLOAD_DIR", uploads.as_str()),
                ("CRABCROP_UPLOAD_MAX_MB", "2"),
                ("CRABCROP_UPLOAD_ALLOW_OVERWRITE", "true"),
            ]),
            &plain,
        )
        .unwrap()
        .unwrap();
        assert_eq!(store.max_bytes, 2 * 1024 * 1024);
        assert!(store.allow_overwrite);
        // A local destination outside a configured file
        // root is refused: its objects could never serve.
        let rooted = build_source_config(env_of(&[("CRABCROP_FILE_ROOT", root.as_str())])).unwrap();
        // The root itself and a directory inside it are under
        // it, and their objects are named from the root.
        for (dir, object) in [
            (root.as_str(), "file:///a.jpg"),
            (uploads.as_str(), "file:///uploads/a.jpg"),
        ] {
            let store = build_upload_store(env_of(&[("CRABCROP_UPLOAD_DIR", dir)]), &rooted)
                .unwrap()
                .unwrap();
            assert_eq!(store.source_ref("a.jpg"), object, "{dir}");
        }
        // The check compares path components, not strings:
        // the sibling is refused like `/`.
        for outside in ["/", sibling.as_str()] {
            let err = expect_err(build_upload_store(
                env_of(&[("CRABCROP_UPLOAD_DIR", outside)]),
                &rooted,
            ));
            assert!(err.contains("CRABCROP_FILE_ROOT"), "{outside}: {err}");
        }
        // A bucket destination with credentials must be in
        // the readable allowlist, or its objects could
        // never serve back.
        let signed = build_source_config(env_of(&[
            ("AWS_ACCESS_KEY_ID", "AKID"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("CRABCROP_ALLOWED_BUCKETS", "photos"),
        ]))
        .unwrap();
        let err = expect_err(build_upload_store(
            env_of(&[("CRABCROP_UPLOAD_BUCKET", "uploads")]),
            &signed,
        ));
        assert!(err.contains("CRABCROP_ALLOWED_BUCKETS"), "{err}");
        let store = build_upload_store(
            env_of(&[
                ("CRABCROP_UPLOAD_BUCKET", "photos"),
                ("CRABCROP_UPLOAD_PREFIX", "orig"),
            ]),
            &signed,
        )
        .unwrap()
        .unwrap();
        assert_eq!(store.source_ref("a/b.jpg"), "s3://photos/orig/a/b.jpg");
        // The same rule without credentials: bucket loads
        // require an allowlist entry either way, so an
        // upload bucket outside the list could never serve
        // back.
        let err = expect_err(build_upload_store(
            env_of(&[("CRABCROP_UPLOAD_BUCKET", "uploads")]),
            &plain,
        ));
        assert!(err.contains("CRABCROP_ALLOWED_BUCKETS"), "{err}");
    }

    #[test]
    fn the_upload_cap_follows_the_source_fetch_cap() {
        const MB: u64 = 1024 * 1024;
        let plain = SourceConfig::default();
        let scratch = crate::test_support::scratch_dir("boot-upload-cap");
        let dir = scratch.to_string_lossy().into_owned();
        let upload = |pairs: &[(&str, &str)]| {
            let mut all = vec![("CRABCROP_UPLOAD_DIR", dir.as_str())];
            all.extend_from_slice(pairs);
            build_upload_store(env_of(&all), &plain)
        };
        // A raised fetch cap (issue #471) raises the upload
        // cap's default and its maximum with it.
        let store = upload(&[("CRABCROP_SOURCE_MAX_MB", "64")])
            .unwrap()
            .unwrap();
        assert_eq!(store.max_bytes, 64 * MB);
        let err = expect_err(upload(&[
            ("CRABCROP_SOURCE_MAX_MB", "64"),
            ("CRABCROP_UPLOAD_MAX_MB", "65"),
        ]));
        assert!(err.contains("between 1 and 64"), "{err}");
        // A lowered one lowers both: an upload cap the
        // default allowed is refused, so a stored original
        // can never outgrow what the loader fetches back.
        let store = upload(&[("CRABCROP_SOURCE_MAX_MB", "2")]).unwrap().unwrap();
        assert_eq!(store.max_bytes, 2 * MB);
        let err = expect_err(upload(&[
            ("CRABCROP_SOURCE_MAX_MB", "2"),
            ("CRABCROP_UPLOAD_MAX_MB", "10"),
        ]));
        assert!(err.contains("CRABCROP_UPLOAD_MAX_MB"), "{err}");
        assert!(err.contains("CRABCROP_SOURCE_MAX_MB"), "{err}");
        // A bad fetch cap fails the build, naming the cap.
        let err = expect_err(upload(&[("CRABCROP_SOURCE_MAX_MB", "2048")]));
        assert!(err.contains("CRABCROP_SOURCE_MAX_MB"), "{err}");
    }

    #[test]
    fn presets_construction_reads_and_validates_the_file() {
        // Nothing configured: no presets, free-form allowed.
        let (presets, only) = build_presets(env_of(&[])).unwrap();
        assert!(presets.is_empty() && !only);
        // A missing file is a startup error naming the variable.
        let missing = missing_path("presets");
        let err = expect_err(build_presets(env_of(&[(
            "CRABCROP_PRESETS_FILE",
            missing.as_str(),
        )])));
        assert!(err.contains("cannot read CRABCROP_PRESETS_FILE"), "{err}");
        // Malformed content is refused at boot...
        let bad = temp_file("presets-bad", b"this line has no equals\n");
        let err = expect_err(build_presets(env_of(&[(
            "CRABCROP_PRESETS_FILE",
            bad.as_str(),
        )])));
        assert!(err.contains("not `name = operations`"), "{err}");
        // ...and so are operations that are not a preset body.
        let bad_ops = temp_file("presets-bad-ops", b"thumb = preset:other\n");
        assert!(build_presets(env_of(&[("CRABCROP_PRESETS_FILE", bad_ops.as_str())])).is_err());
        // A valid file loads, and presets-only is honored with it...
        let good = temp_file("presets-good", b"thumb = 300x200\n");
        let (presets, only) = build_presets(env_of(&[
            ("CRABCROP_PRESETS_FILE", good.as_str()),
            ("CRABCROP_ONLY_PRESETS", "1"),
        ]))
        .unwrap();
        assert_eq!(presets["thumb"], "300x200");
        assert!(only);
        // ...but refused without one: it would reject every
        // transformed request.
        let err = expect_err(build_presets(env_of(&[("CRABCROP_ONLY_PRESETS", "true")])));
        assert!(err.contains("defines no presets"), "{err}");
        // A typo'd flag is an error, not a silent false.
        let err = expect_err(build_presets(env_of(&[("CRABCROP_ONLY_PRESETS", "ture")])));
        assert!(err.contains("CRABCROP_ONLY_PRESETS"), "{err}");
    }

    #[test]
    fn source_roots_construction_reads_and_validates_the_file() {
        // Nothing configured: no roots.
        assert!(build_source_roots(env_of(&[])).unwrap().is_empty());
        // A missing file is a startup error naming the variable.
        let missing = missing_path("roots");
        let err = expect_err(build_source_roots(env_of(&[(
            "CRABCROP_SOURCE_ROOTS_FILE",
            missing.as_str(),
        )])));
        assert!(
            err.contains("cannot read CRABCROP_SOURCE_ROOTS_FILE"),
            "{err}"
        );
        // A valid file loads, comments and blank lines
        // skipped, bases stored in parsed form with the
        // trailing slash trimmed.
        let good = temp_file(
            "roots-good",
            b"# origins\n\nphotos = https://cdn.example.com/photos/\nmedia = http://assets.example.com:8080\n",
        );
        let roots =
            build_source_roots(env_of(&[("CRABCROP_SOURCE_ROOTS_FILE", good.as_str())])).unwrap();
        assert_eq!(roots["photos"], "https://cdn.example.com/photos");
        assert_eq!(roots["media"], "http://assets.example.com:8080");
        // Malformed content is refused at boot, line by line.
        for (tag, text) in [
            ("roots-noeq", "photos https://cdn.example.com\n"),
            ("roots-badname", "my photos = https://cdn.example.com\n"),
            (
                "roots-dup",
                "a = https://x.example.com\na = https://y.example.com\n",
            ),
            ("roots-noturl", "a = cdn.example.com/photos\n"),
            ("roots-scheme", "a = s3://bucket/key\n"),
            ("roots-userinfo", "a = https://user:pass@cdn.example.com\n"),
            ("roots-query", "a = https://cdn.example.com/?v=1\n"),
            ("roots-fragment", "a = https://cdn.example.com/#top\n"),
            // Issue #250: the name charset and the base
            // shape admit no lookalikes either.
            ("roots-dotname", "my.photos = https://cdn.example.com\n"),
            ("roots-uniname", "fötös = https://cdn.example.com\n"),
            ("roots-emptyname", "= https://cdn.example.com\n"),
            ("roots-ftp", "a = ftp://cdn.example.com\n"),
            ("roots-useronly", "a = https://user@cdn.example.com\n"),
            ("roots-nohost", "a = https://\n"),
        ] {
            let path = temp_file(tag, text.as_bytes());
            assert!(
                build_source_roots(env_of(&[("CRABCROP_SOURCE_ROOTS_FILE", path.as_str())]))
                    .is_err(),
                "{tag}"
            );
        }
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
    fn the_source_buffer_budget_scales_with_the_lanes_and_checks_its_bounds() {
        const MB: u64 = 1024 * 1024;
        // Unset: four fetch caps per lane permit, so a
        // one-core host with the default slow lane (two
        // permits) gets 80 MB and a four-core one 200 MB.
        assert_eq!(source_buffer_budget(env_of(&[]), 2).unwrap(), 80 * MB);
        assert_eq!(source_buffer_budget(env_of(&[]), 5).unwrap(), 200 * MB);
        // The default never leaves the accepted range.
        assert_eq!(
            source_buffer_budget(env_of(&[]), usize::MAX).unwrap(),
            MAX_SOURCE_BUFFER_MB * MB
        );
        // Set: megabytes, with one fetch cap as the floor.
        let set =
            |v: &str| source_buffer_budget(env_of(&[("CRABCROP_SOURCE_BUFFER_MAX_MB", v)]), 2);
        assert_eq!(set("10").unwrap(), 10 * MB);
        assert_eq!(set("65536").unwrap(), 65_536 * MB);
        for bad in ["0", "9", "65537", "lots", "-1", ""] {
            let err = expect_err(set(bad));
            assert!(
                err.contains("CRABCROP_SOURCE_BUFFER_MAX_MB"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn source_fetch_bounds_default_to_the_loader_and_check_their_ranges() {
        // Unset: the loader's own defaults, 10 MiB and 10 s.
        assert_eq!(source_max_mb(&env_of(&[])).unwrap(), 10);
        assert_eq!(source_timeout_secs(&env_of(&[])).unwrap(), 10);
        let cap = |v: &str| source_max_mb(&env_of(&[("CRABCROP_SOURCE_MAX_MB", v)]));
        let timeout =
            |v: &str| source_timeout_secs(&env_of(&[("CRABCROP_SOURCE_TIMEOUT_SECS", v)]));
        // The range edges are accepted.
        assert_eq!(cap("1").unwrap(), 1);
        assert_eq!(cap("1024").unwrap(), 1024);
        assert_eq!(timeout("1").unwrap(), 1);
        assert_eq!(timeout("600").unwrap(), 600);
        // Out of range, unparseable, and hostile spellings
        // are boot errors naming the variable, never a
        // silent default.
        for bad in [
            "0",
            "1025",
            "-1",
            "10.5",
            " 10",
            "10MB",
            "",
            "lots",
            "18446744073709551616",
            "1\u{0}",
        ] {
            let err = expect_err(cap(bad));
            assert!(err.contains("CRABCROP_SOURCE_MAX_MB"), "{bad:?}: {err}");
        }
        for bad in [
            "0",
            "601",
            "-1",
            "1.5",
            "10s",
            "",
            "soon",
            "18446744073709551616",
        ] {
            let err = expect_err(timeout(bad));
            assert!(
                err.contains("CRABCROP_SOURCE_TIMEOUT_SECS"),
                "{bad:?}: {err}"
            );
        }
    }

    #[test]
    fn the_source_buffer_budget_follows_the_configured_fetch_cap() {
        const MB: u64 = 1024 * 1024;
        let budget = |pairs: &[(&str, &str)], permits| source_buffer_budget(env_of(pairs), permits);
        // Unset: four fetch caps per lane permit at the
        // configured cap (issue #471), up or down.
        assert_eq!(
            budget(&[("CRABCROP_SOURCE_MAX_MB", "64")], 2).unwrap(),
            512 * MB
        );
        assert_eq!(
            budget(&[("CRABCROP_SOURCE_MAX_MB", "1")], 2).unwrap(),
            8 * MB
        );
        // The largest cap on a large host stays inside the
        // accepted range.
        assert_eq!(
            budget(&[("CRABCROP_SOURCE_MAX_MB", "1024")], 64).unwrap(),
            MAX_SOURCE_BUFFER_MB * MB
        );
        // Set: the floor is one configured cap, so a budget
        // that could never hold a source at the cap is
        // refused, and a lower cap lowers the floor.
        let err = expect_err(budget(
            &[
                ("CRABCROP_SOURCE_MAX_MB", "64"),
                ("CRABCROP_SOURCE_BUFFER_MAX_MB", "63"),
            ],
            2,
        ));
        assert!(err.contains("between 64"), "{err}");
        assert_eq!(
            budget(
                &[
                    ("CRABCROP_SOURCE_MAX_MB", "64"),
                    ("CRABCROP_SOURCE_BUFFER_MAX_MB", "64"),
                ],
                2
            )
            .unwrap(),
            64 * MB
        );
        assert_eq!(
            budget(
                &[
                    ("CRABCROP_SOURCE_MAX_MB", "1"),
                    ("CRABCROP_SOURCE_BUFFER_MAX_MB", "1"),
                ],
                2
            )
            .unwrap(),
            MB
        );
        // A bad cap fails the budget too, naming the cap.
        let err = expect_err(budget(&[("CRABCROP_SOURCE_MAX_MB", "0")], 2));
        assert!(err.contains("CRABCROP_SOURCE_MAX_MB"), "{err}");
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
    fn parses_detector_declarations() {
        let decls = parse_detector_declarations(
            r#"[
              {"name": "crates", "model": "/models/crates.onnx", "kind": "boxes",
               "labels": "/models/crates.txt", "threshold": 0.4},
              {"name": "logos", "model": "/models/logos.onnx", "kind": "boxes",
               "labels": ["nike", "adidas"]},
              {"name": "saliency", "model": "/models/saliency.onnx", "kind": "focal"}
            ]"#,
        )
        .unwrap();
        assert_eq!(decls.len(), 3);
        assert_eq!(decls[0].kind, DetectorKind::Boxes);
        assert_eq!(
            decls[0].labels,
            Some(LabelsDecl::Path("/models/crates.txt".to_string()))
        );
        assert_eq!(decls[0].threshold, Some(0.4));
        assert_eq!(
            decls[1].labels,
            Some(LabelsDecl::Inline(vec![
                "nike".to_string(),
                "adidas".to_string()
            ]))
        );
        assert_eq!(decls[1].threshold, None);
        assert_eq!(decls[2].kind, DetectorKind::Focal);
        assert_eq!(decls[2].labels, None);
        // An empty file declares nothing, validly.
        assert!(parse_detector_declarations("[]").unwrap().is_empty());
    }

    #[test]
    fn detector_declaration_refusals_name_the_declaration() {
        // Every refusal class that precedes model contact,
        // each pinned by its wording (issue #166).
        let cases: &[(&str, &str)] = &[
            ("not json", "not valid JSON"),
            ("{}", "must hold a JSON array"),
            (r#"["crates"]"#, "is not a JSON object"),
            (
                r#"[{"model": "/m.onnx", "kind": "boxes"}]"#,
                "\"name\" must be a non-empty string",
            ),
            (
                r#"[{"name": "bad name!", "model": "/m.onnx", "kind": "boxes"}]"#,
                "invalid detector name",
            ),
            (
                r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes"},
                    {"name": "a", "model": "/n.onnx", "kind": "focal"}]"#,
                "duplicate detector name: a",
            ),
            (
                r#"[{"name": "a", "kind": "boxes"}]"#,
                "\"model\" must be a non-empty path string",
            ),
            (
                r#"[{"name": "a", "model": "/m.onnx"}]"#,
                "\"kind\" must be \"boxes\" or \"focal\"",
            ),
            (
                r#"[{"name": "a", "model": "/m.onnx", "kind": "heatmap"}]"#,
                "unknown kind \"heatmap\"",
            ),
            (
                r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes", "extra": 1}]"#,
                "unknown field \"extra\"",
            ),
            (
                r#"[{"name": "a", "model": "/m.onnx", "kind": "focal", "labels": "/l.txt"}]"#,
                "\"labels\" applies only to \"boxes\" detectors",
            ),
            (
                r#"[{"name": "a", "model": "/m.onnx", "kind": "focal", "threshold": 0.4}]"#,
                "\"threshold\" applies only to \"boxes\" detectors",
            ),
            (
                r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes", "labels": 42}]"#,
                "\"labels\" must be a file path string or an array of class names",
            ),
            (
                r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes", "labels": []}]"#,
                "\"labels\" must name at least one class",
            ),
            (
                r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes", "labels": ["ok", " "]}]"#,
                "\"labels\" entries must be non-empty strings",
            ),
            (
                r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes", "threshold": "high"}]"#,
                "\"threshold\" must be a number",
            ),
            (
                r#"[{"name": "a", "model": "/m.onnx", "kind": "boxes", "threshold": 1.5}]"#,
                "\"threshold\" must be within 0..=1",
            ),
        ];
        for (text, wording) in cases {
            let err = expect_err(parse_detector_declarations(text));
            assert!(err.contains(wording), "{text}: {err}");
        }
        // The unknown-kind refusal names the declaration.
        let err = expect_err(parse_detector_declarations(
            r#"[{"name": "crates", "model": "/m.onnx", "kind": "heatmap"}]"#,
        ));
        assert!(err.contains("\"crates\""), "{err}");
    }

    #[test]
    fn detector_declarations_enforce_resource_limits() {
        // The caps from audit R2-27: declaration count,
        // label count and length, and the reserved name.
        let decl = |i: usize| format!(r#"{{"name": "d{i}", "model": "/m.onnx", "kind": "focal"}}"#);
        let many = (0..=MAX_DETECTOR_DECLARATIONS)
            .map(decl)
            .collect::<Vec<_>>()
            .join(",");
        let err = expect_err(parse_detector_declarations(&format!("[{many}]")));
        assert!(err.contains("the cap is"), "{err}");
        // Exactly at the cap parses.
        let at_cap = (0..MAX_DETECTOR_DECLARATIONS)
            .map(decl)
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(
            parse_detector_declarations(&format!("[{at_cap}]"))
                .unwrap()
                .len(),
            MAX_DETECTOR_DECLARATIONS
        );
        // `builtin` is reserved for the built-in detector's
        // meta attribution.
        let err = expect_err(parse_detector_declarations(
            r#"[{"name": "builtin", "model": "/m.onnx", "kind": "boxes"}]"#,
        ));
        assert!(err.contains("reserved"), "{err}");
        // Inline labels: one over the count cap refuses, at
        // the cap parses, and an overlong label refuses.
        let labels = |n: usize| {
            (0..n)
                .map(|i| format!("\"c{i}\""))
                .collect::<Vec<_>>()
                .join(",")
        };
        let over = format!(
            r#"[{{"name": "a", "model": "/m.onnx", "kind": "boxes", "labels": [{}]}}]"#,
            labels(MAX_DETECTOR_LABELS + 1)
        );
        let err = expect_err(parse_detector_declarations(&over));
        assert!(err.contains("the cap is"), "{err}");
        let at = format!(
            r#"[{{"name": "a", "model": "/m.onnx", "kind": "boxes", "labels": [{}]}}]"#,
            labels(MAX_DETECTOR_LABELS)
        );
        assert!(parse_detector_declarations(&at).is_ok());
        let long_label = "x".repeat(MAX_DETECTOR_LABEL_CHARS + 1);
        let err = expect_err(parse_detector_declarations(&format!(
            r#"[{{"name": "a", "model": "/m.onnx", "kind": "boxes", "labels": ["{long_label}"]}}]"#
        )));
        assert!(err.contains("at most"), "{err}");
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
        // A body over the operation cap refuses at boot
        // (issue #268): at the cap it parses, one over it
        // does not.
        let at_cap = format!("big = {}", vec!["300x200"; MAX_PRESET_OPERATIONS].join("/"));
        assert!(parse_presets(&at_cap).is_ok());
        let over_cap = format!(
            "big = {}",
            vec!["300x200"; MAX_PRESET_OPERATIONS + 1].join("/")
        );
        let err = parse_presets(&over_cap).unwrap_err();
        assert!(err.contains("operation cap"), "{err}");
        // The count is the one requests are held to (issue
        // #469): each filter counts, so one `filters:`
        // segment carrying 101 filters is over the cap too.
        let filters = |n: usize| format!("big = filters:{}", vec!["invert"; n].join(":"));
        assert!(parse_presets(&filters(MAX_PRESET_OPERATIONS)).is_ok());
        let err = parse_presets(&filters(MAX_PRESET_OPERATIONS + 1)).unwrap_err();
        assert_eq!(
            err,
            "preset big has 101 operations, over the 100 operation cap"
        );
    }
}
