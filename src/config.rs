//! The server's configuration (issue #477): every
//! environment variable the server reads, parsed and
//! validated in one place, [`Config::from_env`], before
//! any store opens or any model loads. `open_stores` and
//! `build_state` in `main.rs` then build the stores and
//! providers from a [`Config`], and `run` binds and serves.
//!
//! Every lookup goes through an injected `get` (a name to
//! value lookup), so each step here is unit-tested without
//! the process environment, and every refusal is a
//! [`ConfigError`] whose `Display` is the boot error's
//! text. Every variable is documented in
//! `docs/configuration.md`.
//!
//! This module belongs to the binary, not the library
//! target: `main.rs` declares it, and its `crate::` paths
//! resolve through the library modules `main.rs` imports.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::loader::{SourceConfig, SourcePolicy};

/// Why the environment does not configure a server that
/// can boot. `Display` is the message boot prints after
/// `crabcrop: `, and every variant names the variable at
/// fault, so a caller can match on a failure instead of
/// reading its text. No variant carries a secret's value
/// (issue #455): [`ConfigValue::Secret`] stands in for it,
/// and no message quotes one.
// Boot only prints the message: the fields are for code
// that matches on a failure, the tests today.
#[allow(dead_code)]
#[derive(Debug)]
pub enum ConfigError {
    /// A variable the server cannot boot without is unset.
    Missing { var: &'static str },
    /// A variable's value is refused: it does not parse, is
    /// out of range, or names a directory that is not
    /// usable. `reason` is the whole message.
    Invalid {
        var: &'static str,
        value: ConfigValue,
        reason: String,
    },
    /// Variables refused together: a set configured half
    /// way, two settings that exclude each other, or one
    /// that needs another. `vars` are the variables the
    /// message names; `reason` is the whole message.
    Conflict {
        vars: Vec<&'static str>,
        reason: String,
    },
    /// The configuration file `var` names cannot be read.
    Io {
        var: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
    /// The configuration file `var` names holds something
    /// its parser refuses. `reason` is the whole message.
    Malformed { var: &'static str, reason: String },
    /// A component the configuration names does not build:
    /// a model, font, or fallback image that does not load,
    /// a tool that does not run, a store or an HTTP client
    /// that cannot be opened. `var` is the variable that
    /// configured it, if one did; `reason` is the whole
    /// message.
    Build {
        var: Option<&'static str>,
        reason: String,
    },
}

/// A refused variable's value as a [`ConfigError`] carries
/// it: the text it was set to, or, for a variable that
/// holds a secret, only the fact that it was set.
#[derive(Clone, PartialEq, Eq)]
pub enum ConfigValue {
    Text(String),
    /// Prints as `[redacted]`.
    Secret,
}

impl std::fmt::Debug for ConfigValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigValue::Text(text) => std::fmt::Debug::fmt(text, f),
            ConfigValue::Secret => f.write_str("[redacted]"),
        }
    }
}

/// The variables that hold secrets: a [`ConfigError`] keeps
/// none of their values, and [`Config`] holds them only in
/// types whose `Debug` prints `[redacted]`.
const SECRET_VARIABLES: [&str; 6] = [
    "CRABCROP_SECRET",
    "CRABCROP_SECRET_PREVIOUS",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "CRABCROP_GCS_SECRET_KEY",
    "CRABCROP_METRICS_TOKEN",
];

impl ConfigError {
    /// `var`'s `value` refused, for `reason`. The value of a
    /// variable that holds a secret is never kept.
    fn invalid(var: &'static str, value: &str, reason: impl Into<String>) -> Self {
        let value = if SECRET_VARIABLES.contains(&var) {
            ConfigValue::Secret
        } else {
            ConfigValue::Text(value.to_string())
        };
        ConfigError::Invalid {
            var,
            value,
            reason: reason.into(),
        }
    }

    fn conflict(vars: &[&'static str], reason: impl Into<String>) -> Self {
        ConfigError::Conflict {
            vars: vars.to_vec(),
            reason: reason.into(),
        }
    }

    fn malformed(var: &'static str, reason: impl Into<String>) -> Self {
        ConfigError::Malformed {
            var,
            reason: reason.into(),
        }
    }

    /// The component `var` configured does not build, for
    /// `reason`.
    pub fn build(var: Option<&'static str>, reason: impl Into<String>) -> Self {
        ConfigError::Build {
            var,
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Missing { var } => write!(f, "{var} must be set"),
            ConfigError::Io { var, path, source } => {
                write!(f, "cannot read {var} ({}): {source}", path.display())
            }
            ConfigError::Invalid { reason, .. }
            | ConfigError::Conflict { reason, .. }
            | ConfigError::Malformed { reason, .. }
            | ConfigError::Build { reason, .. } => f.write_str(reason),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// A secret setting's value. `Debug` prints `[redacted]`
/// (issue #455).
#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[redacted]")
    }
}

/// The validated configuration: every setting the server
/// reads, parsed and checked, with nothing built from it
/// yet but the upload destination (see
/// [`Config::from_env`]). `Debug` prints every secret as
/// `[redacted]`.
#[derive(Debug)]
pub struct Config {
    /// The signing keyring (issue #143).
    pub keyring: crate::url::Keyring,
    pub allow_unsafe: bool,
    /// The configured listen port: `CRABCROP_PORT`, else
    /// `PORT`. 0 binds a free port (issue #536).
    pub port: u16,
    pub ssim_target: f64,
    /// The `/metrics` bearer token (issue #270): none closes
    /// the endpoint.
    pub metrics_token: Option<Secret>,
    pub process_timeout_secs: u64,
    /// The stills lane's size: the cores at boot.
    pub process_permits: usize,
    /// The slow lane's size (issue #324).
    pub slow_lane_permits: usize,
    /// The source fetch cap, in megabytes (issue #471).
    pub fetch_cap_mb: u64,
    pub fetch_timeout_secs: u64,
    /// How long a stopping server drains (issue #463).
    pub shutdown_grace_secs: u64,
    /// The source buffer budget, in bytes (issue #403).
    pub source_buffer_bytes: u64,
    pub avif_speed: u8,
    pub jxl_speed: jixel::Speed,
    pub width_step: u32,
    pub cache_bytes: usize,
    pub policy: SourcePolicy,
    /// The bucket and file source settings, the file root
    /// pointed at a local upload directory when none is
    /// configured.
    pub sources: SourceConfig,
    /// The upload destination (issue #168).
    pub upload: Option<Arc<crate::upload::UploadStore>>,
    pub presets: HashMap<String, String>,
    pub only_presets: bool,
    pub source_roots: HashMap<String, String>,
    /// Server-wide default operations (issue #145).
    pub default_ops: Vec<crate::url::Operation>,
    pub preserve_metadata: bool,
    pub brand_color: Option<[u8; 3]>,
    /// The fingerprint every cache key carries (audit
    /// R2-08).
    pub config_fingerprint: String,
    pub result_storage: Option<ResultStorage>,
    pub result_bucket: Option<ResultBucket>,
    pub ai: AiConfig,
    pub font_path: Option<String>,
    pub fallback_image: Option<String>,
    pub video: VideoConfig,
    pub pdf: PdfConfig,
    pub provenance: Option<ProvenanceConfig>,
    /// Meta mode (issue #453).
    pub meta_enabled: bool,
}

/// Disk result storage: `CRABCROP_RESULT_STORAGE_DIR`, with
/// its budget.
#[derive(Debug)]
pub struct ResultStorage {
    pub dir: String,
    pub max_mb: usize,
    pub max_bytes: usize,
}

/// Shared result storage (issue #142): one bucket, with a
/// key prefix.
#[derive(Debug)]
pub struct ResultBucket {
    pub bucket: String,
    pub prefix: String,
}

impl Config {
    /// Read, parse, and validate every setting through `get`
    /// (a name to value lookup; boot passes the process
    /// environment). Nothing is built here but the upload
    /// destination, whose validation is its construction
    /// (the directory is created and checked against the
    /// file root, which it may become, and the directory
    /// is created when it does not exist yet); the result
    /// stores are `open_stores`'s, and the providers
    /// `build_state`'s.
    ///
    /// The settings are read in the order boot has always
    /// read them, so of two bad settings the same one is
    /// reported. The lines boot prints while reading them
    /// print here, in the same order: the proxy warning on
    /// stderr, then the bucket endpoint note, the upload
    /// destination, the source roots, the default
    /// operations, and the brand color.
    pub fn from_env(get: impl Fn(&str) -> Option<String>) -> Result<Config, ConfigError> {
        let keyring = build_keyring(&get)?;
        let allow_unsafe =
            parse_bool_value("CRABCROP_ALLOW_UNSAFE", get("CRABCROP_ALLOW_UNSAFE"), false)?;
        // CRABCROP_PORT is the namespaced alias for PORT and
        // wins when both are set; PORT alone keeps working for
        // the deployments and charts that already use it.
        let (port_name, port_value) = match get("CRABCROP_PORT") {
            Some(v) => ("CRABCROP_PORT", Some(v)),
            None => ("PORT", get("PORT")),
        };
        let port: u16 = parse_env_value(port_name, port_value, 3000)?;
        let ssim_target: f64 = parse_env_value(
            "CRABCROP_SSIM_TARGET",
            get("CRABCROP_SSIM_TARGET"),
            crate::quality::DEFAULT_SSIM_TARGET,
        )?;
        if !ssim_target.is_finite() || !(0.0..=100.0).contains(&ssim_target) {
            return Err(ConfigError::invalid(
                "CRABCROP_SSIM_TARGET",
                &get("CRABCROP_SSIM_TARGET").unwrap_or_default(),
                format!("CRABCROP_SSIM_TARGET must be between 0 and 100, got {ssim_target}"),
            ));
        }
        // The /metrics bearer token (issue #270): an empty
        // value is no token, and no token closes the
        // endpoint rather than serving it publicly.
        let metrics_token = get("CRABCROP_METRICS_TOKEN")
            .filter(|v| !v.is_empty())
            .map(Secret);
        let process_timeout_secs: u64 = parse_env_value(
            "CRABCROP_PROCESS_TIMEOUT_SECS",
            get("CRABCROP_PROCESS_TIMEOUT_SECS"),
            30,
        )?;
        if !(1..=600).contains(&process_timeout_secs) {
            return Err(ConfigError::invalid(
                "CRABCROP_PROCESS_TIMEOUT_SECS",
                &process_timeout_secs.to_string(),
                format!(
                    "CRABCROP_PROCESS_TIMEOUT_SECS must be between 1 and 600, got {process_timeout_secs}"
                ),
            ));
        }
        // The slow lane's size (issue #324): how many
        // long-running encodes (animated WebP, video, frame
        // and page work) may run at once, on permits stills
        // never wait for. The default 1 suits a small host:
        // one long encode at a time, with the stills lane and
        // a core left for stills.
        let slow_lane_permits: usize = parse_env_value(
            "CRABCROP_SLOW_LANE_PERMITS",
            get("CRABCROP_SLOW_LANE_PERMITS"),
            crate::server::DEFAULT_SLOW_LANE_PERMITS,
        )?;
        if !(1..=64).contains(&slow_lane_permits) {
            return Err(ConfigError::invalid(
                "CRABCROP_SLOW_LANE_PERMITS",
                &slow_lane_permits.to_string(),
                format!(
                    "CRABCROP_SLOW_LANE_PERMITS must be between 1 and 64, got {slow_lane_permits}"
                ),
            ));
        }
        // The stills lane is sized to the cores at boot; the
        // source buffer budget's default scales with both lanes.
        let process_permits = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        // The source fetch bounds (issue #471); the budget and
        // the upload cap derive from the same cap.
        let fetch_cap_mb = source_max_mb(&get)?;
        let fetch_timeout_secs = source_timeout_secs(&get)?;
        let shutdown_grace_secs = shutdown_grace_secs(&get)?;
        let source_buffer_bytes = source_buffer_budget(&get, process_permits + slow_lane_permits)?;
        let avif_speed: u8 = parse_env_value(
            "CRABCROP_AVIF_SPEED",
            get("CRABCROP_AVIF_SPEED"),
            crate::engine::DEFAULT_AVIF_SPEED,
        )?;
        if !(1..=10).contains(&avif_speed) {
            return Err(ConfigError::invalid(
                "CRABCROP_AVIF_SPEED",
                &avif_speed.to_string(),
                format!("CRABCROP_AVIF_SPEED must be between 1 and 10, got {avif_speed}"),
            ));
        }
        let jxl_speed = match get("CRABCROP_JXL_SPEED") {
            Some(v) => parse_jxl_speed(&v)?,
            None => crate::engine::DEFAULT_JXL_SPEED,
        };
        let width_step: u32 = parse_env_value(
            "CRABCROP_WIDTH_STEP",
            get("CRABCROP_WIDTH_STEP"),
            DEFAULT_WIDTH_STEP,
        )?;
        if !(1..=2000).contains(&width_step) {
            return Err(ConfigError::invalid(
                "CRABCROP_WIDTH_STEP",
                &width_step.to_string(),
                format!("CRABCROP_WIDTH_STEP must be between 1 and 2000, got {width_step}"),
            ));
        }
        let cache_mb: usize =
            parse_env_value("CRABCROP_CACHE_MAX_MB", get("CRABCROP_CACHE_MAX_MB"), 256)?;
        let cache_bytes = cache_mb.checked_mul(1024 * 1024).ok_or_else(|| {
            ConfigError::invalid(
                "CRABCROP_CACHE_MAX_MB",
                &cache_mb.to_string(),
                "CRABCROP_CACHE_MAX_MB is too large",
            )
        })?;
        let policy = build_source_policy(&get)?;
        if let Some(warning) = proxy_bypass_warning(&get, &policy) {
            eprintln!("warning: {warning}");
        }
        let mut sources = build_source_config(&get)?;
        // The upload destination (issue #168), resolved after
        // the source configuration it reuses. A local
        // destination becomes the file source root when none
        // is configured, so uploaded objects serve back
        // without a second knob; a configured root was
        // already checked to contain it.
        let upload = build_upload_store(&get, &sources)?;
        if let Some(store) = &upload
            && let Some(dir) = store.local_dir()
            && sources.file_root.is_none()
        {
            sources.file_root = Some(dir.to_path_buf());
        }
        let (presets, only_presets) = build_presets(&get)?;
        let source_roots = build_source_roots(&get)?;
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
        let default_ops = match get("CRABCROP_DEFAULT_OPS") {
            Some(text) if !text.trim().is_empty() => {
                let ops = crate::url::parse_default_ops(&text, &presets).map_err(|e| {
                    ConfigError::invalid(
                        "CRABCROP_DEFAULT_OPS",
                        &text,
                        format!("invalid CRABCROP_DEFAULT_OPS: {e}"),
                    )
                })?;
                println!("default operations: {text}");
                ops
            }
            _ => Vec::new(),
        };
        let preserve_metadata = metadata_config(&get)?;
        let brand_color = brand_color_config(&get)?;
        if let Some(c) = brand_color {
            println!("brand color: #{:02x}{:02x}{:02x}", c[0], c[1], c[2]);
        }
        // The configuration fingerprint (audit R2-08): every
        // cache key carries it, so the persistent tiers never
        // serve an entry rendered under different
        // output-shaping settings across a restart. It is
        // computed in `main.rs`, beside the rendering version
        // it hashes.
        let config_fingerprint = crate::config_fingerprint(
            &get,
            preserve_metadata,
            ssim_target,
            avif_speed,
            jxl_speed,
            width_step,
        );
        // Disk result storage: configured by directory,
        // bounded by its own budget, whose variable is read
        // only when a directory is set.
        let result_storage = match get("CRABCROP_RESULT_STORAGE_DIR") {
            Some(dir) if !dir.is_empty() => {
                let max_mb: usize = parse_env_value(
                    "CRABCROP_RESULT_STORAGE_MAX_MB",
                    get("CRABCROP_RESULT_STORAGE_MAX_MB"),
                    512,
                )?;
                let max_bytes = max_mb.checked_mul(1024 * 1024).ok_or_else(|| {
                    ConfigError::invalid(
                        "CRABCROP_RESULT_STORAGE_MAX_MB",
                        &max_mb.to_string(),
                        "CRABCROP_RESULT_STORAGE_MAX_MB is too large",
                    )
                })?;
                Some(ResultStorage {
                    dir,
                    max_mb,
                    max_bytes,
                })
            }
            _ => None,
        };
        // Shared result storage (issue #142): one bucket as
        // the multi-replica tier, reusing the source S3
        // configuration; a prefix namespaces the entries.
        let result_bucket = match get("CRABCROP_RESULT_BUCKET") {
            Some(bucket) if !bucket.is_empty() => Some(ResultBucket {
                bucket,
                prefix: get("CRABCROP_RESULT_PREFIX").unwrap_or_default(),
            }),
            _ => None,
        };
        let ai = ai_config(&get)?;
        let font_path = get("CRABCROP_FONT_PATH").filter(|v| !v.is_empty());
        let fallback_image = get("CRABCROP_FALLBACK_IMAGE").filter(|v| !v.is_empty());
        let video = video_config(&get)?;
        let pdf = pdf_config(&get)?;
        let provenance = provenance_config(&get)?;
        let meta_enabled = meta_mode_config(&get)?;
        Ok(Config {
            keyring,
            allow_unsafe,
            port,
            ssim_target,
            metrics_token,
            process_timeout_secs,
            process_permits,
            slow_lane_permits,
            fetch_cap_mb,
            fetch_timeout_secs,
            shutdown_grace_secs,
            source_buffer_bytes,
            avif_speed,
            jxl_speed,
            width_step,
            cache_bytes,
            policy,
            sources,
            upload,
            presets,
            only_presets,
            source_roots,
            default_ops,
            preserve_metadata,
            brand_color,
            config_fingerprint,
            result_storage,
            result_bucket,
            ai,
            font_path,
            fallback_image,
            video,
            pdf,
            provenance,
            meta_enabled,
        })
    }
}

/// Load the preset configuration: the presets file (one
/// `name = operations` entry per line) and the presets-only
/// flag, both validated strictly. Presets-only with no
/// presets configured would reject every transformed
/// request, so it is a startup error rather than a policy.
pub fn build_presets(
    get: impl Fn(&str) -> Option<String>,
) -> Result<(HashMap<String, String>, bool), ConfigError> {
    let only_presets =
        parse_bool_value("CRABCROP_ONLY_PRESETS", get("CRABCROP_ONLY_PRESETS"), false)?;
    let presets = match get("CRABCROP_PRESETS_FILE") {
        Some(path) if !path.is_empty() => {
            let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Io {
                var: "CRABCROP_PRESETS_FILE",
                path: PathBuf::from(&path),
                source,
            })?;
            parse_presets(&text)?
        }
        _ => HashMap::new(),
    };
    if only_presets && presets.is_empty() {
        return Err(ConfigError::conflict(
            &["CRABCROP_ONLY_PRESETS", "CRABCROP_PRESETS_FILE"],
            "CRABCROP_ONLY_PRESETS is set but CRABCROP_PRESETS_FILE defines no presets",
        ));
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
pub fn parse_presets(text: &str) -> Result<HashMap<String, String>, ConfigError> {
    let malformed = |reason: String| ConfigError::malformed("CRABCROP_PRESETS_FILE", reason);
    let mut map = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, ops) = line.split_once('=').ok_or_else(|| {
            malformed(format!("preset line {} is not `name = operations`", i + 1))
        })?;
        let name = name.trim();
        let ops = ops.trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(malformed(format!(
                "invalid preset name on line {}: {name:?}",
                i + 1
            )));
        }
        if map.contains_key(name) {
            return Err(malformed(format!("duplicate preset name: {name}")));
        }
        let parsed = crate::url::parse_preset_operations(ops)
            .map_err(|e| malformed(format!("invalid operations for preset {name}: {e}")))?;
        let count = crate::url::operation_count(&parsed);
        if count > MAX_PRESET_OPERATIONS {
            return Err(malformed(format!(
                "preset {name} has {count} operations, over the {MAX_PRESET_OPERATIONS} operation cap"
            )));
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
pub fn build_source_roots(
    get: impl Fn(&str) -> Option<String>,
) -> Result<HashMap<String, String>, ConfigError> {
    match get("CRABCROP_SOURCE_ROOTS_FILE") {
        Some(path) if !path.is_empty() => {
            let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Io {
                var: "CRABCROP_SOURCE_ROOTS_FILE",
                path: PathBuf::from(&path),
                source,
            })?;
            parse_source_roots(&text)
        }
        _ => Ok(HashMap::new()),
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
pub fn parse_source_roots(text: &str) -> Result<HashMap<String, String>, ConfigError> {
    let malformed = |reason: String| ConfigError::malformed("CRABCROP_SOURCE_ROOTS_FILE", reason);
    let mut map = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, base) = line.split_once('=').ok_or_else(|| {
            malformed(format!(
                "source root line {} is not `name = base-url`",
                i + 1
            ))
        })?;
        let name = name.trim();
        let base = base.trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(malformed(format!(
                "invalid source root name on line {}: {name:?}",
                i + 1
            )));
        }
        if map.contains_key(name) {
            return Err(malformed(format!("duplicate source root name: {name}")));
        }
        let parsed = reqwest::Url::parse(base)
            .map_err(|e| malformed(format!("invalid base URL for source root {name}: {e}")))?;
        if parsed.scheme() != "http" && parsed.scheme() != "https" {
            return Err(malformed(format!(
                "base URL for source root {name} must be an http(s) URL: {base:?}"
            )));
        }
        if parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
        {
            return Err(malformed(format!(
                "base URL for source root {name} must not carry userinfo: {base:?}"
            )));
        }
        if parsed.query().is_some() || parsed.fragment().is_some() {
            return Err(malformed(format!(
                "base URL for source root {name} must not carry a query or fragment: {base:?}"
            )));
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
pub fn check_bucket_policy(
    has_credentials: bool,
    allowed_buckets: &[String],
) -> Result<(), ConfigError> {
    if has_credentials && allowed_buckets.is_empty() {
        return Err(ConfigError::conflict(
            &["CRABCROP_ALLOWED_BUCKETS"],
            "bucket credentials are configured but CRABCROP_ALLOWED_BUCKETS is unset: \
             refusing to boot, because signed bucket fetches would reach every bucket \
             the credentials can read. Set CRABCROP_ALLOWED_BUCKETS to a comma-separated \
             list, or to * to deliberately allow all buckets.",
        ));
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
pub fn build_source_policy(
    get: impl Fn(&str) -> Option<String>,
) -> Result<SourcePolicy, ConfigError> {
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
pub fn store_endpoint(name: &'static str, endpoint: &str) -> Result<String, ConfigError> {
    let parsed = reqwest::Url::parse(endpoint)
        .map_err(|e| ConfigError::invalid(name, endpoint, format!("invalid {name}: {e}")))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(ConfigError::invalid(
            name,
            endpoint,
            format!("{name} must be an http(s) URL"),
        ));
    }
    Ok(endpoint.trim_end_matches('/').to_string())
}

/// An error followed by its causes, `: `-separated. reqwest
/// reports a client it cannot build only as "builder error";
/// the reason (no CA roots on the host, say) is in the
/// source chain, and a boot error should name it.
pub fn with_causes(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut cause = error.source();
    while let Some(inner) = cause {
        message.push_str(": ");
        message.push_str(&inner.to_string());
        cause = inner.source();
    }
    message
}

pub fn build_source_config(
    get: impl Fn(&str) -> Option<String>,
) -> Result<SourceConfig, ConfigError> {
    let mut sources = SourceConfig::default();
    let nonempty = |name: &str| get(name).filter(|v| !v.is_empty());

    if let Some(endpoint) = nonempty("CRABCROP_S3_ENDPOINT") {
        sources.s3_endpoint = Some(store_endpoint("CRABCROP_S3_ENDPOINT", &endpoint)?);
    }
    if let Some(region) = nonempty("CRABCROP_S3_REGION") {
        sources.s3_region = region;
    }
    let cred_pair = |key_name: &'static str,
                     secret_name: &'static str|
     -> Result<Option<crate::sigv4::Credentials>, ConfigError> {
        match (nonempty(key_name), nonempty(secret_name)) {
            (Some(access_key), Some(secret_key)) => Ok(Some(crate::sigv4::Credentials {
                access_key,
                secret_key,
                session_token: None,
            })),
            (None, None) => Ok(None),
            _ => Err(ConfigError::conflict(
                &[key_name, secret_name],
                format!("{key_name} and {secret_name} must be set together"),
            )),
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
        let canonical = std::fs::canonicalize(&root).map_err(|e| {
            ConfigError::invalid(
                "CRABCROP_FILE_ROOT",
                &root,
                format!("CRABCROP_FILE_ROOT is not usable: {e}"),
            )
        })?;
        if !canonical.is_dir() {
            return Err(ConfigError::invalid(
                "CRABCROP_FILE_ROOT",
                &root,
                "CRABCROP_FILE_ROOT is not a directory",
            ));
        }
        sources.file_root = Some(canonical);
    }
    Ok(sources)
}

/// Default quantization step for width client hints.
pub const DEFAULT_WIDTH_STEP: u32 = 100;

/// Parse a `CRABCROP_JXL_SPEED` value into a jixel speed tier.
pub fn parse_jxl_speed(value: &str) -> Result<jixel::Speed, ConfigError> {
    match value.to_ascii_lowercase().as_str() {
        "ultrafast" => Ok(jixel::Speed::UltraFast),
        "fastest" => Ok(jixel::Speed::Fastest),
        "fast" => Ok(jixel::Speed::Fast),
        "medium" => Ok(jixel::Speed::Medium),
        "slow" => Ok(jixel::Speed::Slow),
        "extraslow" => Ok(jixel::Speed::ExtraSlow),
        _ => Err(ConfigError::invalid(
            "CRABCROP_JXL_SPEED",
            value,
            format!(
                "invalid CRABCROP_JXL_SPEED: {value:?} (expected ultrafast, fastest, fast, medium, slow, or extraslow)"
            ),
        )),
    }
}

/// Read the variable `name`'s `value` as `T`, or fall back
/// to `default` when it is unset. A set-but-invalid value
/// is a startup error naming the variable: silently serving
/// on the wrong port or with the wrong cache budget is
/// worse than refusing to start.
pub fn parse_env_value<T: std::str::FromStr>(
    name: &'static str,
    value: Option<String>,
    default: T,
) -> Result<T, ConfigError> {
    match value {
        Some(v) => v
            .parse()
            .map_err(|_| ConfigError::invalid(name, &v, format!("invalid {name}: {v:?}"))),
        None => Ok(default),
    }
}

/// Read the boolean variable `name`'s `value`, strictly:
/// `1`, `true`, `yes`, `on` (any case) are true; `0`,
/// `false`, `no`, `off` are false; anything else is a
/// startup error naming the variable. A lenient reading
/// (only `1` or `true` count) would make a typo like `ture`
/// silently false, which for a safety flag such as
/// CRABCROP_BLOCK_PRIVATE_SOURCES is the unsafe direction.
pub fn parse_bool_value(
    name: &'static str,
    value: Option<String>,
    default: bool,
) -> Result<bool, ConfigError> {
    match value.as_deref() {
        None => Ok(default),
        Some(v) => match v.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(ConfigError::invalid(
                name,
                v,
                format!(
                    "invalid {name}: {v:?} (expected one of 1, true, yes, on, 0, false, no, off)"
                ),
            )),
        },
    }
}

/// The minimum `CRABCROP_SECRET` length, in bytes. Shorter
/// secrets are brute-forceable, and an empty one is worse:
/// empty-key HMAC is publicly computable, so every signed URL
/// (and purge) is forgeable (security audit PL-3, P-6).
const MIN_SECRET_BYTES: usize = 16;

pub fn validate_secret(secret: &[u8]) -> Result<(), ConfigError> {
    validate_secret_named("CRABCROP_SECRET", secret)
}

/// The refusal names the variable and the length, never the
/// secret (issue #455).
fn validate_secret_named(name: &'static str, secret: &[u8]) -> Result<(), ConfigError> {
    if secret.len() < MIN_SECRET_BYTES {
        return Err(ConfigError::Invalid {
            var: name,
            value: ConfigValue::Secret,
            reason: format!(
                "{name} must be at least {MIN_SECRET_BYTES} bytes, got {}",
                secret.len()
            ),
        });
    }
    Ok(())
}

/// The signing keyring (issue #143): the current secret under
/// `CRABCROP_SECRET_KEY_ID` (default `current`), plus the
/// previous secret under `CRABCROP_SECRET_PREVIOUS_KEY_ID`
/// (default `previous`) while a rotation window is open.
pub fn build_keyring(
    get: impl Fn(&str) -> Option<String>,
) -> Result<crate::url::Keyring, ConfigError> {
    let current = get("CRABCROP_SECRET")
        .ok_or(ConfigError::Missing {
            var: "CRABCROP_SECRET",
        })?
        .into_bytes();
    validate_secret(&current)?;
    let current_id = get("CRABCROP_SECRET_KEY_ID")
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "current".to_string());
    if !crate::url::is_valid_key_id(&current_id) {
        return Err(ConfigError::invalid(
            "CRABCROP_SECRET_KEY_ID",
            &current_id,
            format!(
                "CRABCROP_SECRET_KEY_ID {current_id:?} is not a valid key ID (1 to 32 characters of [A-Za-z0-9_-])"
            ),
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
                return Err(ConfigError::invalid(
                    "CRABCROP_SECRET_PREVIOUS_KEY_ID",
                    &id,
                    format!(
                        "CRABCROP_SECRET_PREVIOUS_KEY_ID {id:?} is not a valid key ID (1 to 32 characters of [A-Za-z0-9_-])"
                    ),
                ));
            }
            if id == current_id {
                return Err(ConfigError::conflict(
                    &["CRABCROP_SECRET_KEY_ID", "CRABCROP_SECRET_PREVIOUS_KEY_ID"],
                    "CRABCROP_SECRET_KEY_ID and CRABCROP_SECRET_PREVIOUS_KEY_ID must differ",
                ));
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
pub fn source_max_mb(get: &impl Fn(&str) -> Option<String>) -> Result<u64, ConfigError> {
    let mb: u64 = parse_env_value(
        "CRABCROP_SOURCE_MAX_MB",
        get("CRABCROP_SOURCE_MAX_MB"),
        crate::loader::DEFAULT_MAX_BYTES / (1024 * 1024),
    )?;
    if !(1..=MAX_SOURCE_MAX_MB).contains(&mb) {
        return Err(ConfigError::invalid(
            "CRABCROP_SOURCE_MAX_MB",
            &mb.to_string(),
            format!("CRABCROP_SOURCE_MAX_MB must be between 1 and {MAX_SOURCE_MAX_MB}, got {mb}"),
        ));
    }
    Ok(mb)
}

/// The total time one source fetch may take, redirects
/// and body included (issue #471):
/// `CRABCROP_SOURCE_TIMEOUT_SECS`, 1 to 600 like the other
/// wall-clock limits, defaulting to the loader's 10 s.
pub fn source_timeout_secs(get: &impl Fn(&str) -> Option<String>) -> Result<u64, ConfigError> {
    let secs: u64 = parse_env_value(
        "CRABCROP_SOURCE_TIMEOUT_SECS",
        get("CRABCROP_SOURCE_TIMEOUT_SECS"),
        crate::loader::DEFAULT_TIMEOUT.as_secs(),
    )?;
    if !(1..=600).contains(&secs) {
        return Err(ConfigError::invalid(
            "CRABCROP_SOURCE_TIMEOUT_SECS",
            &secs.to_string(),
            format!("CRABCROP_SOURCE_TIMEOUT_SECS must be between 1 and 600, got {secs}"),
        ));
    }
    Ok(secs)
}

/// How long a stopping server drains (issue #463):
/// `CRABCROP_SHUTDOWN_GRACE_SECS`, 1 to 600 like the other
/// wall-clock limits, defaulting to 8 s, under the 10 s
/// `docker stop` and Cloud Run allow. It bounds the
/// connection drain and the background write drain
/// together, from the signal on.
pub fn shutdown_grace_secs(get: &impl Fn(&str) -> Option<String>) -> Result<u64, ConfigError> {
    let secs: u64 = parse_env_value(
        "CRABCROP_SHUTDOWN_GRACE_SECS",
        get("CRABCROP_SHUTDOWN_GRACE_SECS"),
        crate::server::DEFAULT_SHUTDOWN_GRACE_SECS,
    )?;
    if !(1..=600).contains(&secs) {
        return Err(ConfigError::invalid(
            "CRABCROP_SHUTDOWN_GRACE_SECS",
            &secs.to_string(),
            format!("CRABCROP_SHUTDOWN_GRACE_SECS must be between 1 and 600, got {secs}"),
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
pub fn proxy_bypass_warning(
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
pub fn source_buffer_budget(
    get: impl Fn(&str) -> Option<String>,
    lane_permits: usize,
) -> Result<u64, ConfigError> {
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
        return Err(ConfigError::invalid(
            "CRABCROP_SOURCE_BUFFER_MAX_MB",
            &mb.to_string(),
            format!(
                "CRABCROP_SOURCE_BUFFER_MAX_MB must be between {cap_mb} (the source fetch cap) \
                 and {MAX_SOURCE_BUFFER_MB}, got {mb}"
            ),
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
/// file root at a local destination (see
/// [`Config::from_env`]).
pub fn build_upload_store(
    get: impl Fn(&str) -> Option<String>,
    sources: &SourceConfig,
) -> Result<Option<Arc<crate::upload::UploadStore>>, ConfigError> {
    let nonempty = |name: &str| get(name).filter(|v| !v.is_empty());
    let dir = nonempty("CRABCROP_UPLOAD_DIR");
    let bucket = nonempty("CRABCROP_UPLOAD_BUCKET");
    if dir.is_some() && bucket.is_some() {
        return Err(ConfigError::conflict(
            &["CRABCROP_UPLOAD_DIR", "CRABCROP_UPLOAD_BUCKET"],
            "CRABCROP_UPLOAD_DIR and CRABCROP_UPLOAD_BUCKET are both set: \
             configure exactly one upload destination",
        ));
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
        return Err(ConfigError::invalid(
            "CRABCROP_UPLOAD_MAX_MB",
            &max_mb.to_string(),
            format!(
                "CRABCROP_UPLOAD_MAX_MB must be between 1 and {source_mb} \
                 (the source fetch cap, CRABCROP_SOURCE_MAX_MB, an uploaded original \
                 must stay servable under), got {max_mb}"
            ),
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
            let unusable = |e: std::io::Error| {
                ConfigError::invalid(
                    "CRABCROP_UPLOAD_DIR",
                    &dir,
                    format!("CRABCROP_UPLOAD_DIR is not usable: {e}"),
                )
            };
            std::fs::create_dir_all(&dir).map_err(unusable)?;
            let canonical = std::fs::canonicalize(&dir).map_err(unusable)?;
            if !canonical.is_dir() {
                return Err(ConfigError::invalid(
                    "CRABCROP_UPLOAD_DIR",
                    &dir,
                    "CRABCROP_UPLOAD_DIR is not a directory",
                ));
            }
            if let Some(root) = &sources.file_root
                && !canonical.starts_with(root)
            {
                return Err(ConfigError::conflict(
                    &["CRABCROP_UPLOAD_DIR", "CRABCROP_FILE_ROOT"],
                    "CRABCROP_UPLOAD_DIR must be under CRABCROP_FILE_ROOT \
                     (uploaded objects are served back as file:// sources)",
                ));
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
            Ok(Some(Arc::new(
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
                return Err(ConfigError::conflict(
                    &["CRABCROP_UPLOAD_BUCKET", "CRABCROP_ALLOWED_BUCKETS"],
                    format!(
                        "CRABCROP_UPLOAD_BUCKET {bucket:?} is not in CRABCROP_ALLOWED_BUCKETS: \
                         uploaded objects are served back as s3:// sources, so the bucket \
                         must be a readable source bucket"
                    ),
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
            .map_err(|e| {
                ConfigError::build(
                    Some("CRABCROP_UPLOAD_BUCKET"),
                    format!("cannot build the upload store client: {}", with_causes(&e)),
                )
            })?;
            println!(
                "uploads enabled: bucket {bucket} (cap {max_mb} MB, overwrite {allow_overwrite})"
            );
            Ok(Some(Arc::new(store)))
        }
        _ => unreachable!("the both-set and neither-set cases returned above"),
    }
}

/// The CLIP annotation configuration from the environment:
/// all four paths together, or none of them.
#[derive(Debug, Clone)]
pub struct ClipConfig {
    pub image_model: String,
    pub text_model: String,
    pub tokenizer: String,
    pub labels: String,
    pub threshold: f32,
}

/// Resolve the `CRABCROP_CLIP_*` set through `get` (a name
/// to value lookup, so tests can drive it without the
/// process environment). A partial set, or an out-of-range
/// threshold, is an error.
pub fn clip_config(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<ClipConfig>, ConfigError> {
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
    let missing: Vec<&'static str> = NAMES
        .iter()
        .zip(&values)
        .filter(|(_, v)| v.is_none())
        .map(|(name, _)| *name)
        .collect();
    if !missing.is_empty() {
        return Err(ConfigError::conflict(
            &missing,
            format!(
                "CLIP annotation is partially configured: missing {}",
                missing.join(", ")
            ),
        ));
    }
    let threshold = match get("CRABCROP_CLIP_THRESHOLD").filter(|v| !v.is_empty()) {
        Some(raw) => {
            let parsed: f32 = raw.parse().map_err(|_| {
                ConfigError::invalid(
                    "CRABCROP_CLIP_THRESHOLD",
                    &raw,
                    format!("CRABCROP_CLIP_THRESHOLD is not a number: {raw:?}"),
                )
            })?;
            if !(0.0..=1.0).contains(&parsed) {
                return Err(ConfigError::invalid(
                    "CRABCROP_CLIP_THRESHOLD",
                    &raw,
                    format!("CRABCROP_CLIP_THRESHOLD must be within 0..=1, got {parsed}"),
                ));
            }
            parsed
        }
        None => 0.05,
    };
    let mut values = values.into_iter().flatten();
    let (Some(image_model), Some(text_model), Some(tokenizer), Some(labels)) =
        (values.next(), values.next(), values.next(), values.next())
    else {
        unreachable!("all four CLIP paths were checked present above");
    };
    Ok(Some(ClipConfig {
        image_model,
        text_model,
        tokenizer,
        labels,
        threshold,
    }))
}

/// The operator brand color from `CRABCROP_BRAND_COLOR`
/// (issue #197): six hex digits, an optional leading `#`
/// allowed, resolved through `get` so tests can drive it
/// without the process environment. Unset means studio
/// backdrops resolve to white; a set but malformed value
/// is a startup error naming the variable.
pub fn brand_color_config(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<[u8; 3]>, ConfigError> {
    let raw = match get("CRABCROP_BRAND_COLOR") {
        Some(v) if !v.is_empty() => v,
        _ => return Ok(None),
    };
    let hex = raw.strip_prefix('#').unwrap_or(&raw);
    if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ConfigError::invalid(
            "CRABCROP_BRAND_COLOR",
            &raw,
            format!("CRABCROP_BRAND_COLOR must be a hex color like f5f0e8, got {raw:?}"),
        ));
    }
    let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).unwrap();
    Ok(Some([byte(0), byte(2), byte(4)]))
}

/// The metadata policy from `CRABCROP_METADATA`: `strip`
/// (the default, and the behavior when unset) or `preserve`.
/// Anything else is a startup error naming the value.
pub fn metadata_config(get: impl Fn(&str) -> Option<String>) -> Result<bool, ConfigError> {
    match get("CRABCROP_METADATA") {
        None => Ok(false),
        Some(v) if v.eq_ignore_ascii_case("strip") => Ok(false),
        Some(v) if v.eq_ignore_ascii_case("preserve") => Ok(true),
        Some(v) => Err(ConfigError::invalid(
            "CRABCROP_METADATA",
            &v,
            format!("CRABCROP_METADATA must be \"strip\" or \"preserve\", got {v:?}"),
        )),
    }
}

/// Meta mode from `CRABCROP_META_MODE`, off unless the
/// operator turns it on (issue #453). Every meta request
/// skips the cache tiers and runs a fresh fetch, render,
/// and annotation inferences, and any viewer of a signed
/// URL can ask for one with a header, so a deployment
/// opts in to that cost rather than carrying it unasked.
pub fn meta_mode_config(get: impl Fn(&str) -> Option<String>) -> Result<bool, ConfigError> {
    parse_bool_value("CRABCROP_META_MODE", get("CRABCROP_META_MODE"), false)
}

/// The AI settings: every model path, checked for
/// completeness and with its options parsed, so a provider
/// that `build_state` loads fails only on its files. Every
/// provider is configured by path, never downloaded.
#[derive(Debug)]
pub struct AiConfig {
    /// The ONNX Runtime library to load, when not the
    /// default search.
    pub ort_dylib: Option<String>,
    pub removebg_model: Option<String>,
    pub upscale_model: Option<String>,
    pub inpaint_model: Option<String>,
    pub segment: Option<SegmentConfig>,
    pub clip: Option<ClipConfig>,
    pub moderation: Option<ModerationConfig>,
    /// The aesthetic head (issue #172), which scores the
    /// CLIP set's embedding, so it comes only with `clip`.
    pub aesthetic_head: Option<String>,
    pub ocr: Option<OcrConfig>,
    pub detect: Option<DetectConfig>,
    /// The declarations `CRABCROP_DETECTORS_FILE` holds
    /// (issue #166), in file order.
    pub detectors: Vec<DetectorDeclaration>,
    pub face_model: Option<String>,
}

/// The segmentation model pair (issue #175): an encoder and
/// its prompt decoder.
#[derive(Debug)]
pub struct SegmentConfig {
    pub encoder: String,
    pub decoder: String,
}

/// Where moderation verdicts come from (issue #171).
#[derive(Debug)]
pub enum ModerationConfig {
    /// A verdict classifier, with a labels file naming its
    /// output categories when one is set.
    Model {
        model: String,
        labels: Option<String>,
    },
    /// No model: the labels file holds zero-shot prompts for
    /// the CLIP fallback, which loads the CLIP set again
    /// with these labels.
    ClipPrompts { labels: String, clip: ClipConfig },
}

/// The OCR pipeline (issue #177): detection and recognition
/// models, the recognition model's dictionary, and the
/// language it serves (`und` when unset).
#[derive(Debug)]
pub struct OcrConfig {
    pub detect: String,
    pub recognize: String,
    pub dict: String,
    pub language: String,
}

/// The built-in detector: its model, the fallback labels
/// file for exports that carry no class names, and its
/// threshold.
#[derive(Debug)]
pub struct DetectConfig {
    pub model: String,
    pub labels: Option<String>,
    pub threshold: f32,
}

/// The AI settings from the environment, through `get`. The
/// checks run in the order the providers have always been
/// built in: the segmentation pair, the CLIP set, the
/// moderation source, the aesthetic head, the OCR set, the
/// detector threshold, and the detector declarations.
pub fn ai_config(get: impl Fn(&str) -> Option<String>) -> Result<AiConfig, ConfigError> {
    let path = |name: &str| get(name).filter(|v| !v.is_empty());
    let segment = match (
        path("CRABCROP_SEGMENT_ENCODER_MODEL"),
        path("CRABCROP_SEGMENT_DECODER_MODEL"),
    ) {
        (None, None) => None,
        (Some(encoder), Some(decoder)) => Some(SegmentConfig { encoder, decoder }),
        (None, Some(_)) => {
            return Err(ConfigError::conflict(
                &["CRABCROP_SEGMENT_ENCODER_MODEL"],
                "segmentation is partially configured: missing CRABCROP_SEGMENT_ENCODER_MODEL",
            ));
        }
        (Some(_), None) => {
            return Err(ConfigError::conflict(
                &["CRABCROP_SEGMENT_DECODER_MODEL"],
                "segmentation is partially configured: missing CRABCROP_SEGMENT_DECODER_MODEL",
            ));
        }
    };
    let clip = clip_config(&get)?;
    let moderation = match (
        path("CRABCROP_MODERATION_MODEL"),
        path("CRABCROP_MODERATION_LABELS"),
    ) {
        (None, None) => None,
        (Some(model), labels) => Some(ModerationConfig::Model { model, labels }),
        // Labels without a model are zero-shot prompts for
        // the CLIP fallback, which needs the CLIP set.
        (None, Some(labels)) => match &clip {
            Some(clip) => Some(ModerationConfig::ClipPrompts {
                labels,
                clip: clip.clone(),
            }),
            None => {
                return Err(ConfigError::conflict(
                    &["CRABCROP_MODERATION_LABELS", "CRABCROP_MODERATION_MODEL"],
                    "CRABCROP_MODERATION_LABELS is set without CRABCROP_MODERATION_MODEL: the labels then name zero-shot prompts, which need the CRABCROP_CLIP_* set",
                ));
            }
        },
    };
    let aesthetic_head = path("CRABCROP_AESTHETIC_HEAD");
    if aesthetic_head.is_some() && clip.is_none() {
        return Err(ConfigError::conflict(
            &["CRABCROP_AESTHETIC_HEAD"],
            "CRABCROP_AESTHETIC_HEAD is set but the CRABCROP_CLIP_* set is not configured: the head scores the CLIP tagger's embedding, so an encoder must be configured",
        ));
    }
    let ocr = ocr_config(&get)?;
    let detect = match path("CRABCROP_DETECT_MODEL") {
        Some(model) => {
            let threshold = match get("CRABCROP_DETECT_THRESHOLD") {
                Some(raw) if !raw.is_empty() => {
                    let parsed: f32 = raw.parse().map_err(|_| {
                        ConfigError::invalid(
                            "CRABCROP_DETECT_THRESHOLD",
                            &raw,
                            format!("CRABCROP_DETECT_THRESHOLD is not a number: {raw:?}"),
                        )
                    })?;
                    if !(0.0..=1.0).contains(&parsed) {
                        return Err(ConfigError::invalid(
                            "CRABCROP_DETECT_THRESHOLD",
                            &raw,
                            format!("CRABCROP_DETECT_THRESHOLD must be within 0..=1, got {parsed}"),
                        ));
                    }
                    parsed
                }
                _ => 0.5,
            };
            Some(DetectConfig {
                model,
                labels: path("CRABCROP_DETECT_LABELS"),
                threshold,
            })
        }
        None => None,
    };
    let detectors = match path("CRABCROP_DETECTORS_FILE") {
        Some(file) => read_detector_declarations(&file)?,
        None => Vec::new(),
    };
    Ok(AiConfig {
        ort_dylib: path("CRABCROP_ORT_DYLIB"),
        removebg_model: path("CRABCROP_REMOVEBG_MODEL"),
        upscale_model: path("CRABCROP_UPSCALE_MODEL"),
        inpaint_model: path("CRABCROP_INPAINT_MODEL"),
        segment,
        clip,
        moderation,
        aesthetic_head,
        ocr,
        detect,
        detectors,
        face_model: path("CRABCROP_FACE_MODEL"),
    })
}

/// The OCR set (issue #177): the three paths are
/// all-or-nothing, the segment pair's rule: a partial set
/// is a startup error naming what is missing, because half
/// a pipeline reads nothing and a dictionary without its
/// model aligns with nothing. `CRABCROP_OCR_LANGUAGE`
/// optionally declares the language the recognition model
/// serves (a BCP-47 tag), echoed in meta so the response
/// describes itself; unset reports `und`.
fn ocr_config(get: &impl Fn(&str) -> Option<String>) -> Result<Option<OcrConfig>, ConfigError> {
    let detect = get("CRABCROP_OCR_DETECT_MODEL").filter(|v| !v.is_empty());
    let recognize = get("CRABCROP_OCR_RECOGNIZE_MODEL").filter(|v| !v.is_empty());
    let dict = get("CRABCROP_OCR_DICT").filter(|v| !v.is_empty());
    let missing: Vec<&'static str> = [
        ("CRABCROP_OCR_DETECT_MODEL", detect.is_none()),
        ("CRABCROP_OCR_RECOGNIZE_MODEL", recognize.is_none()),
        ("CRABCROP_OCR_DICT", dict.is_none()),
    ]
    .iter()
    .filter(|(_, missing)| *missing)
    .map(|(name, _)| *name)
    .collect();
    let (Some(detect), Some(recognize), Some(dict)) = (detect, recognize, dict) else {
        if missing.len() == 3 {
            return Ok(None);
        }
        return Err(ConfigError::conflict(
            &missing,
            format!(
                "OCR is partially configured: missing {}",
                missing.join(", ")
            ),
        ));
    };
    let language = get("CRABCROP_OCR_LANGUAGE")
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "und".to_string());
    Ok(Some(OcrConfig {
        detect,
        recognize,
        dict,
        language,
    }))
}

/// The contract a declared detector binds to (issue #166).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorKind {
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
pub enum LabelsDecl {
    /// A path to a labels file (one class per line), like
    /// `CRABCROP_DETECT_LABELS`.
    Path(String),
    /// Class names written inline in the declaration.
    Inline(Vec<String>),
}

/// One validated detector declaration from
/// `CRABCROP_DETECTORS_FILE`.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorDeclaration {
    pub name: String,
    pub model: String,
    pub kind: DetectorKind,
    pub labels: Option<LabelsDecl>,
    pub threshold: Option<f32>,
}

/// The detectors file's size cap: declarations are small
/// JSON, and the file is read whole at boot (audit R2-27).
pub const MAX_DETECTORS_FILE_BYTES: usize = 1024 * 1024;

/// The declaration count cap: every declaration loads and
/// probes a full model at boot, and every declared `boxes`
/// detector runs under the one AI permit of a meta request,
/// so the count bounds both the boot cost and the meta
/// fan-out (audit R2-27).
const MAX_DETECTOR_DECLARATIONS: usize = 16;

/// The class vocabulary cap per declaration, applied to
/// inline labels at parse time and to a labels file's
/// resolved classes at build time (audit R2-27).
pub const MAX_DETECTOR_LABELS: usize = 4096;

/// The length cap for one inline class label (audit R2-27).
const MAX_DETECTOR_LABEL_CHARS: usize = 256;

/// Read and parse `CRABCROP_DETECTORS_FILE` (issue #166).
/// The file only names models and options, so nothing
/// loads here: each declaration is loaded and probed
/// against its contract in `build_state`.
fn read_detector_declarations(path: &str) -> Result<Vec<DetectorDeclaration>, ConfigError> {
    const VAR: &str = "CRABCROP_DETECTORS_FILE";
    // The file is read under a size cap, like the cache
    // entry parser's: declarations are small JSON, and an
    // unbounded read at boot is an unbounded allocation
    // (audit R2-27).
    use std::io::Read as _;
    let unreadable = |source| ConfigError::Io {
        var: VAR,
        path: PathBuf::from(path),
        source,
    };
    let mut file = std::fs::File::open(path).map_err(unreadable)?;
    let mut bytes = Vec::new();
    file.by_ref()
        .take((MAX_DETECTORS_FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(unreadable)?;
    if bytes.len() > MAX_DETECTORS_FILE_BYTES {
        return Err(ConfigError::malformed(
            VAR,
            format!("{VAR} ({path}) exceeds the {MAX_DETECTORS_FILE_BYTES} byte cap"),
        ));
    }
    let text = String::from_utf8(bytes).map_err(|e| {
        ConfigError::malformed(VAR, format!("{VAR} ({path}) is not valid UTF-8: {e}"))
    })?;
    parse_detector_declarations(&text)
}

/// Parse the detectors file's text into declarations,
/// strictly: this runs before any model is touched, so
/// every refusal here is raisable (and testable) without
/// the ONNX runtime. Unknown fields, kinds, and
/// misapplied options are errors naming the declaration.
pub fn parse_detector_declarations(text: &str) -> Result<Vec<DetectorDeclaration>, ConfigError> {
    let malformed = |reason: String| ConfigError::malformed("CRABCROP_DETECTORS_FILE", reason);
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| malformed(format!("CRABCROP_DETECTORS_FILE is not valid JSON: {e}")))?;
    let entries = value.as_array().ok_or_else(|| {
        malformed(
            "CRABCROP_DETECTORS_FILE must hold a JSON array of detector declarations".to_string(),
        )
    })?;
    if entries.len() > MAX_DETECTOR_DECLARATIONS {
        return Err(malformed(format!(
            "CRABCROP_DETECTORS_FILE declares {} detectors; the cap is {MAX_DETECTOR_DECLARATIONS}",
            entries.len()
        )));
    }
    let mut declarations: Vec<DetectorDeclaration> = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        let at = format!("detector declaration {}", i + 1);
        let obj = entry
            .as_object()
            .ok_or_else(|| malformed(format!("{at} is not a JSON object")))?;
        let name = match obj.get("name").and_then(serde_json::Value::as_str) {
            Some(name) if !name.is_empty() => name.to_string(),
            _ => {
                return Err(malformed(format!(
                    "{at}: \"name\" must be a non-empty string"
                )));
            }
        };
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(malformed(format!("invalid detector name: {name:?}")));
        }
        // `builtin` is the source name meta detections use
        // for the built-in detector's boxes: a declaration
        // taking it would make its boxes indistinguishable
        // from the built-in's (audit R2-27).
        if name == "builtin" {
            return Err(malformed(
                "detector name \"builtin\" is reserved for the built-in detector".to_string(),
            ));
        }
        if declarations.iter().any(|d| d.name == name) {
            return Err(malformed(format!("duplicate detector name: {name}")));
        }
        let named = format!("detector {name:?}");
        for key in obj.keys() {
            if !["name", "model", "kind", "labels", "threshold"].contains(&key.as_str()) {
                return Err(malformed(format!("{named}: unknown field {key:?}")));
            }
        }
        let model = match obj.get("model").and_then(serde_json::Value::as_str) {
            Some(model) if !model.is_empty() => model.to_string(),
            _ => {
                return Err(malformed(format!(
                    "{named}: \"model\" must be a non-empty path string"
                )));
            }
        };
        let kind = match obj.get("kind").and_then(serde_json::Value::as_str) {
            Some("boxes") => DetectorKind::Boxes,
            Some("focal") => DetectorKind::Focal,
            Some(other) => {
                return Err(malformed(format!(
                    "{named}: unknown kind {other:?} (expected \"boxes\" or \"focal\")"
                )));
            }
            None => {
                return Err(malformed(format!(
                    "{named}: \"kind\" must be \"boxes\" or \"focal\""
                )));
            }
        };
        let labels = match obj.get("labels") {
            None => None,
            Some(_) if kind == DetectorKind::Focal => {
                return Err(malformed(format!(
                    "{named}: \"labels\" applies only to \"boxes\" detectors"
                )));
            }
            Some(serde_json::Value::String(path)) if !path.is_empty() => {
                Some(LabelsDecl::Path(path.clone()))
            }
            Some(serde_json::Value::Array(items)) => {
                if items.is_empty() {
                    return Err(malformed(format!(
                        "{named}: \"labels\" must name at least one class"
                    )));
                }
                if items.len() > MAX_DETECTOR_LABELS {
                    return Err(malformed(format!(
                        "{named}: \"labels\" names {} classes; the cap is {MAX_DETECTOR_LABELS}",
                        items.len()
                    )));
                }
                let mut names = Vec::new();
                for item in items {
                    match item.as_str() {
                        Some(class) if !class.trim().is_empty() => {
                            if class.chars().count() > MAX_DETECTOR_LABEL_CHARS {
                                return Err(malformed(format!(
                                    "{named}: \"labels\" entries must be at most \
                                     {MAX_DETECTOR_LABEL_CHARS} characters"
                                )));
                            }
                            names.push(class.to_string());
                        }
                        _ => {
                            return Err(malformed(format!(
                                "{named}: \"labels\" entries must be non-empty strings"
                            )));
                        }
                    }
                }
                Some(LabelsDecl::Inline(names))
            }
            Some(_) => {
                return Err(malformed(format!(
                    "{named}: \"labels\" must be a file path string or an array of class names"
                )));
            }
        };
        let threshold = match obj.get("threshold") {
            None => None,
            Some(_) if kind == DetectorKind::Focal => {
                return Err(malformed(format!(
                    "{named}: \"threshold\" applies only to \"boxes\" detectors"
                )));
            }
            Some(value) => match value.as_f64() {
                Some(v) if (0.0..=1.0).contains(&v) => Some(v as f32),
                Some(v) => {
                    return Err(malformed(format!(
                        "{named}: \"threshold\" must be within 0..=1, got {v}"
                    )));
                }
                None => {
                    return Err(malformed(format!(
                        "{named}: \"threshold\" must be a number"
                    )));
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

/// The video tools' settings: the shared wall-clock limit
/// (issue #214), the encoder's output cap, and the encoder
/// (issue #162) and frame extractor programs. The bounds
/// parse whether or not a tool is configured: a typo'd
/// bound is a boot error even before the tool is set.
#[derive(Debug)]
pub struct VideoConfig {
    pub timeout_secs: u64,
    pub max_mb: usize,
    pub max_bytes: usize,
    pub encoder: Option<String>,
    pub decoder: Option<String>,
}

pub fn video_config(get: impl Fn(&str) -> Option<String>) -> Result<VideoConfig, ConfigError> {
    let timeout_secs = video_timeout_secs(&get)?;
    let max_mb: usize = parse_env_value(
        "CRABCROP_VIDEO_MAX_MB",
        get("CRABCROP_VIDEO_MAX_MB"),
        crate::video::DEFAULT_MAX_OUTPUT_MB,
    )?;
    if !(1..=1024).contains(&max_mb) {
        return Err(ConfigError::invalid(
            "CRABCROP_VIDEO_MAX_MB",
            &max_mb.to_string(),
            format!("CRABCROP_VIDEO_MAX_MB must be between 1 and 1024, got {max_mb}"),
        ));
    }
    let max_bytes = max_mb.checked_mul(1024 * 1024).ok_or_else(|| {
        ConfigError::invalid(
            "CRABCROP_VIDEO_MAX_MB",
            &max_mb.to_string(),
            "CRABCROP_VIDEO_MAX_MB is too large",
        )
    })?;
    Ok(VideoConfig {
        timeout_secs,
        max_mb,
        max_bytes,
        encoder: get("CRABCROP_VIDEO_ENCODER").filter(|v| !v.is_empty()),
        decoder: get("CRABCROP_VIDEO_DECODER").filter(|v| !v.is_empty()),
    })
}

/// The shared wall-clock limit for one external video
/// tool run, encode or frame extraction (issue #214):
/// `CRABCROP_VIDEO_TIMEOUT_SECS`, validated the same way
/// whichever tool ends up configured.
fn video_timeout_secs(get: &impl Fn(&str) -> Option<String>) -> Result<u64, ConfigError> {
    let secs: u64 = parse_env_value(
        "CRABCROP_VIDEO_TIMEOUT_SECS",
        get("CRABCROP_VIDEO_TIMEOUT_SECS"),
        crate::video::DEFAULT_TIMEOUT_SECS,
    )?;
    if !(1..=600).contains(&secs) {
        return Err(ConfigError::invalid(
            "CRABCROP_VIDEO_TIMEOUT_SECS",
            &secs.to_string(),
            format!("CRABCROP_VIDEO_TIMEOUT_SECS must be between 1 and 600, got {secs}"),
        ));
    }
    Ok(secs)
}

/// The PDF renderer's settings (issue #215): its program,
/// and its own wall-clock limit rather than the video
/// tools', because a page render's cost profile differs
/// from a frame extraction's. The limit parses even when
/// no renderer is configured, like the video bounds.
#[derive(Debug)]
pub struct PdfConfig {
    pub timeout_secs: u64,
    pub renderer: Option<String>,
}

pub fn pdf_config(get: impl Fn(&str) -> Option<String>) -> Result<PdfConfig, ConfigError> {
    let timeout_secs: u64 = parse_env_value(
        "CRABCROP_PDF_TIMEOUT_SECS",
        get("CRABCROP_PDF_TIMEOUT_SECS"),
        crate::video::DEFAULT_TIMEOUT_SECS,
    )?;
    if !(1..=600).contains(&timeout_secs) {
        return Err(ConfigError::invalid(
            "CRABCROP_PDF_TIMEOUT_SECS",
            &timeout_secs.to_string(),
            format!("CRABCROP_PDF_TIMEOUT_SECS must be between 1 and 600, got {timeout_secs}"),
        ));
    }
    Ok(PdfConfig {
        timeout_secs,
        renderer: get("CRABCROP_PDF_RENDERER").filter(|v| !v.is_empty()),
    })
}

/// The C2PA signing identity (issue #173): a PEM
/// certificate chain and its private key, with an optional
/// timestamp authority.
#[derive(Debug)]
pub struct ProvenanceConfig {
    pub cert: String,
    pub key: String,
    pub tsa: Option<String>,
}

/// The `CRABCROP_C2PA_CERT` / `CRABCROP_C2PA_KEY` pair and
/// `CRABCROP_C2PA_TSA_URL`. The pair is all-or-nothing, the
/// CLIP rule: a partial set is a startup error naming what
/// is missing, and a timestamp authority without the pair
/// is one too.
pub fn provenance_config(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<ProvenanceConfig>, ConfigError> {
    let cert = get("CRABCROP_C2PA_CERT").filter(|v| !v.is_empty());
    let key = get("CRABCROP_C2PA_KEY").filter(|v| !v.is_empty());
    let tsa = get("CRABCROP_C2PA_TSA_URL").filter(|v| !v.is_empty());
    match (cert, key) {
        (None, None) => {
            if tsa.is_some() {
                return Err(ConfigError::conflict(
                    &[
                        "CRABCROP_C2PA_TSA_URL",
                        "CRABCROP_C2PA_CERT",
                        "CRABCROP_C2PA_KEY",
                    ],
                    "CRABCROP_C2PA_TSA_URL is set but no signing identity is configured: \
                     set CRABCROP_C2PA_CERT and CRABCROP_C2PA_KEY",
                ));
            }
            Ok(None)
        }
        (Some(cert), Some(key)) => Ok(Some(ProvenanceConfig { cert, key, tsa })),
        (None, Some(_)) => Err(ConfigError::conflict(
            &["CRABCROP_C2PA_CERT"],
            "C2PA signing is partially configured: missing CRABCROP_C2PA_CERT",
        )),
        (Some(_), None) => Err(ConfigError::conflict(
            &["CRABCROP_C2PA_KEY"],
            "C2PA signing is partially configured: missing CRABCROP_C2PA_KEY",
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests;
