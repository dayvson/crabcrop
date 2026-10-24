//! Operator CLI verbs, dispatched from `main` before the
//! server starts: `sign`, `explain`, `meta`, `purge`,
//! `purge-group`, and `convert-thumbor`. Configuration comes
//! from `--base` / `--secret` flags over `CRABCROP_BASE` /
//! `CRABCROP_SECRET`, the same convention as the MCP server.
//! The verbs use the server's own URL grammar, so what the
//! CLI validates is what the server will do.

use crate::url::{
    Dimension, Filter, FlipDirection, ImageFormat, MaskShape, Operation, Quality, Signature,
};

const USAGE: &str = "\
crabcrop: on-demand image optimization service

Server:
  crabcrop                      Start the server (configuration via environment).

CLI verbs (flags: --base <url>, --secret <secret>; env: CRABCROP_BASE, CRABCROP_SECRET):
  crabcrop sign <remainder>           Print the signed URL for an operations + source remainder.
  crabcrop explain <url-or-path>      Break a URL down segment by segment.
  crabcrop meta <url-or-remainder>    Fetch and print the meta JSON for an image.
                                      (--placeholders / --detections ask for those fields.)
  crabcrop purge <url-or-remainder>   Purge one URL's cached variants.
  crabcrop purge-group <name>         Purge every cached entry in a preset group.
  crabcrop convert-thumbor <url>      Convert a Thumbor URL to a crabcrop URL.";

struct Config {
    base: String,
    secret: Option<Vec<u8>>,
    /// Key ID to stamp on minted signatures (issue #143),
    /// from `--key-id` or `CRABCROP_SECRET_KEY_ID`.
    key_id: Option<String>,
    /// Meta opt-ins (issue #151): ask the meta endpoint for
    /// placeholder fields / detection boxes. Only the `meta`
    /// verb accepts them.
    placeholders: bool,
    detections: bool,
}

fn parse_flags(args: &[String]) -> Result<(Config, Vec<String>), String> {
    let mut base: Option<String> = None;
    let mut secret: Option<String> = None;
    let mut key_id: Option<String> = None;
    let mut placeholders = false;
    let mut detections = false;
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let (flag, inline_value) = match arg.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_string())),
            _ => (arg.as_str(), None),
        };
        match flag {
            "--base" | "--secret" | "--key-id" => {
                let value = match inline_value {
                    Some(v) => v,
                    None => {
                        i += 1;
                        args.get(i)
                            .cloned()
                            .ok_or_else(|| format!("{flag} needs a value"))?
                    }
                };
                if flag == "--base" {
                    base = Some(value);
                } else if flag == "--key-id" {
                    key_id = Some(value);
                } else {
                    secret = Some(value);
                }
            }
            "--placeholders" | "--detections" => {
                if inline_value.is_some() {
                    return Err(format!("{flag} takes no value\n\n{USAGE}"));
                }
                if flag == "--placeholders" {
                    placeholders = true;
                } else {
                    detections = true;
                }
            }
            _ if arg.starts_with("--") => return Err(format!("unknown flag {arg:?}\n\n{USAGE}")),
            _ => positional.push(arg.clone()),
        }
        i += 1;
    }
    let base = base
        .or_else(|| std::env::var("CRABCROP_BASE").ok())
        .unwrap_or_else(|| "http://localhost:3000".to_string())
        .trim_end_matches('/')
        .to_string();
    let secret = secret
        .or_else(|| std::env::var("CRABCROP_SECRET").ok())
        .map(String::into_bytes);
    let key_id = key_id.or_else(|| std::env::var("CRABCROP_SECRET_KEY_ID").ok());
    if let Some(id) = &key_id
        && !crate::url::is_valid_key_id(id)
    {
        return Err(format!(
            "invalid key ID {id:?}: 1 to 32 characters of [A-Za-z0-9_-]"
        ));
    }
    Ok((
        Config {
            base,
            secret,
            key_id,
            placeholders,
            detections,
        },
        positional,
    ))
}

fn require_secret(config: &Config) -> Result<&[u8], String> {
    config
        .secret
        .as_deref()
        .ok_or_else(|| "no secret: pass --secret or set CRABCROP_SECRET".to_string())
}

/// Split a URL or path into (signature, remainder). A full
/// URL loses its scheme and host; a bare remainder comes
/// back with no signature.
fn split_signed(input: &str, base: &str) -> (Option<Signature>, Option<String>, String) {
    let input = input.trim();
    // A URL on the configured base: strip the base, then
    // read the signature segment.
    if let Some(rest) = input.strip_prefix(&format!("{base}/")) {
        return match crate::url::parse_signature(rest) {
            Ok((sig, kid, remainder)) => (Some(sig), kid, remainder.to_string()),
            Err(_) => (None, None, rest.to_string()),
        };
    }
    // A full URL on any other host counts only when its
    // path really starts with a signature segment: a bare
    // remainder also contains "://" (in its source), so
    // anything less strict would mangle it.
    if let Some((_, after_scheme)) = input.split_once("://") {
        let path = after_scheme.split_once('/').map(|(_, p)| p).unwrap_or("");
        if let Ok((sig, kid, remainder)) = crate::url::parse_signature(path) {
            return (Some(sig), kid, remainder.to_string());
        }
        return (None, None, input.to_string());
    }
    let path = input.strip_prefix('/').unwrap_or(input);
    match crate::url::parse_signature(path) {
        Ok((sig, kid, rest)) => (Some(sig), kid, rest.to_string()),
        Err(_) => (None, None, path.to_string()),
    }
}

/// What validating a CLI remainder yields: how many leading
/// segments preset expansion dropped, then the parse (the
/// operations, the source, and the `exp:` / `maxage:`
/// metadata).
type PresetParse = (usize, Vec<Operation>, String, Option<u64>, Option<u32>);

/// Parse a remainder that may lead with preset references:
/// drop leading segments until the grammar accepts the
/// rest, returning how many were dropped. Server presets
/// expand before parsing, so the CLI cannot parse them
/// itself; it can only step aside.
fn parse_with_presets(rest: &str) -> Result<PresetParse, String> {
    let segments: Vec<&str> = rest.split('/').collect();
    let mut first_error = String::new();
    for dropped in 0..segments.len() {
        // Only `preset:` references may be stepped over: the
        // server expands them before parsing, and nothing
        // else is allowed to lead a remainder unparsed.
        if dropped > 0 && !segments[dropped - 1].starts_with("preset:") {
            break;
        }
        let candidate = segments[dropped..].join("/");
        match crate::url::parse_operations_with_expiry(&candidate) {
            Ok((ops, source, expiry, max_age)) => {
                return Ok((dropped, ops, source.to_string(), expiry, max_age));
            }
            Err(e) => {
                if dropped == 0 {
                    first_error = e.to_string();
                }
            }
        }
    }
    Err(first_error)
}

/// Run one CLI verb. `args` includes the verb itself.
/// Returns what to print on stdout.
pub async fn exec(args: &[String]) -> Result<String, String> {
    let verb = args.first().map(String::as_str).unwrap_or("");
    if verb == "--help" || verb == "-h" || verb == "help" {
        return Ok(USAGE.to_string());
    }
    // Help in any position is help: `crabcrop sign --help`
    // prints usage and exits 0, like the top-level spelling,
    // instead of travelling as an error to stderr.
    if args.iter().skip(1).any(|a| a == "--help" || a == "-h") {
        return Ok(USAGE.to_string());
    }
    let (config, positional) = parse_flags(&args[1..])?;
    if verb != "meta" && (config.placeholders || config.detections) {
        return Err(format!(
            "--placeholders and --detections only apply to the meta verb\n\n{USAGE}"
        ));
    }
    match verb {
        "sign" => sign(&config, &positional),
        "explain" => Ok(explain(&config, &positional)?),
        "meta" => meta(&config, &positional).await,
        "purge" => purge(&config, &positional).await,
        "purge-group" => purge_group(&config, &positional).await,
        "convert-thumbor" => convert_thumbor_cmd(&config, &positional),
        _ => Err(format!("unknown verb {verb:?}\n\n{USAGE}")),
    }
}

fn one_target(positional: &[String], verb: &str) -> Result<String, String> {
    match positional {
        [only] => Ok(only.clone()),
        [] => Err(format!("{verb} needs an argument\n\n{USAGE}")),
        _ => Err(format!("{verb} takes exactly one argument\n\n{USAGE}")),
    }
}

fn sign(config: &Config, positional: &[String]) -> Result<String, String> {
    let remainder = one_target(positional, "sign")?;
    let secret = require_secret(config)?;
    // Validate with the server's grammar before signing, so
    // a bad URL fails here instead of at first request.
    parse_with_presets(&remainder)?;
    let sig = crabcrop_url::sign(secret, &remainder);
    let segment = match &config.key_id {
        Some(kid) => format!("{kid}:{sig}"),
        None => sig,
    };
    Ok(format!("{}/{segment}/{remainder}", config.base))
}

fn explain(config: &Config, positional: &[String]) -> Result<String, String> {
    let input = one_target(positional, "explain")?;
    let (signature, key_id, rest) = split_signed(&input, &config.base);
    let mut lines: Vec<String> = Vec::new();
    match &signature {
        Some(Signature::Hmac(hex)) => {
            let note = match &config.secret {
                Some(secret) => {
                    if crate::url::verify(secret, &Signature::Hmac(hex.clone()), &rest, false) {
                        "valid for this remainder"
                    } else {
                        "DOES NOT MATCH this remainder"
                    }
                }
                None => "not verified (no secret given)",
            };
            let kid_note = match &key_id {
                Some(kid) => format!("key {kid}; "),
                None => String::new(),
            };
            lines.push(format!("signature: {hex} ({kid_note}{note})"));
        }
        Some(Signature::Unsafe) => lines.push("signature: unsafe (unsigned URL)".to_string()),
        None => lines.push("signature: none (bare remainder)".to_string()),
    }
    let segments: Vec<&str> = rest.split('/').collect();
    match parse_with_presets(&rest) {
        Ok((dropped, ops, source, expiry, max_age)) => {
            for seg in &segments[..dropped] {
                lines.push(format!(
                    "  {seg}  preset reference (expanded by the server)"
                ));
            }
            let mut op_iter = ops.iter();
            let mut expiry_printed = false;
            let mut max_age_printed = false;
            for seg in &segments[dropped..segments.len() - source.matches('/').count() - 1] {
                if let Some(ts) = seg.strip_prefix("exp:") {
                    lines.push(format!("  {seg}  link expires at unix time {ts}"));
                    expiry_printed = true;
                    continue;
                }
                if let Some(secs) = seg.strip_prefix("maxage:") {
                    lines.push(format!("  {seg}  response cache lifetime: {secs} seconds"));
                    max_age_printed = true;
                    continue;
                }
                match op_iter.next() {
                    Some(op) => lines.extend(describe_operation(seg, op)),
                    None => lines.push(format!("  {seg}")),
                }
            }
            if expiry.is_some() && !expiry_printed {
                lines.push(format!("  exp:{}  link expiry", expiry.unwrap_or_default()));
            }
            if max_age.is_some() && !max_age_printed {
                lines.push(format!(
                    "  maxage:{}  response cache lifetime",
                    max_age.unwrap_or_default()
                ));
            }
            lines.push(format!("source: {source}"));
            lines.push(format!(
                "verdict: valid, {} operation{}",
                ops.len(),
                if ops.len() == 1 { "" } else { "s" }
            ));
        }
        Err(e) => {
            for seg in &segments {
                lines.push(format!("  {seg}"));
            }
            lines.push(format!("verdict: invalid: {e}"));
        }
    }
    Ok(lines.join("\n"))
}

fn dim(d: Dimension) -> String {
    match d {
        Dimension::Pixels(n) => n.to_string(),
        Dimension::Proportional => "proportional".to_string(),
        Dimension::Original => "original".to_string(),
    }
}

fn size_text(size: &crate::url::ImageSize) -> String {
    let mut text = format!("{}x{}", dim(size.width), dim(size.height));
    if size.flip_horizontal {
        text.push_str(", flipped horizontally");
    }
    if size.flip_vertical {
        text.push_str(", flipped vertically");
    }
    text
}

fn describe_operation(segment: &str, op: &Operation) -> Vec<String> {
    match op {
        Operation::Size(size) => vec![format!("  {segment}  resize to {}", size_text(size))],
        Operation::Crop { size, align } => vec![format!(
            "  {segment}  crop to {}, aligned {align:?}",
            size_text(size)
        )],
        Operation::FitIn(size) => vec![format!("  {segment}  fit inside {}", size_text(size))],
        Operation::Pad { size, .. } => vec![format!("  {segment}  pad to {}", size_text(size))],
        Operation::CropRect {
            left,
            top,
            right,
            bottom,
        } => vec![format!(
            "  {segment}  crop the rectangle {left},{top} to {right},{bottom}"
        )],
        Operation::FocalRect {
            left,
            top,
            right,
            bottom,
        } => vec![format!(
            "  {segment}  set the focal rectangle {left},{top} to {right},{bottom} for the next crop"
        )],
        Operation::Filters(filters) => {
            let mut lines = vec![format!("  {segment}")];
            for filter in filters {
                lines.push(format!("    {}", describe_filter(filter)));
            }
            lines
        }
    }
}

fn hex3(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn describe_filter(filter: &Filter) -> String {
    match filter {
        Filter::Grayscale => "grayscale: convert to grayscale".into(),
        Filter::Invert => "invert: invert the colors".into(),
        Filter::Blur(v) => format!("blur({v}): gaussian blur"),
        Filter::Brighten(v) => format!("brighten({v}): adjust brightness"),
        Filter::Contrast(v) => format!("contrast({v}): adjust contrast"),
        Filter::HueRotate(v) => format!("huerotate({v}): rotate the hue"),
        Filter::Sharpen(v) => format!("sharpen({v}): sharpen"),
        Filter::Saturation(v) => format!("saturation({v}): adjust saturation"),
        Filter::Rgb(r, g, b) => format!("rgb({r},{g},{b}): shift the channels"),
        Filter::Curve(channel, points) => {
            let mut text = format!("curve({}", channel.name());
            for (x, y) in points {
                text.push_str(&format!(",{x},{y}"));
            }
            text.push_str("): remap tones through the control points");
            text
        }
        Filter::Equalize => "equalize: histogram equalization".into(),
        Filter::Rotate(v) => format!("rotate({v}): rotate by a multiple of 90 degrees"),
        Filter::Trim(v) => format!("trim({v}): trim near-uniform borders"),
        Filter::Watermark { url, x, y, alpha } => {
            format!("watermark: {url} at {x},{y}, alpha {alpha}")
        }
        Filter::Overlay {
            url,
            x,
            y,
            alpha,
            blend,
            width,
        } => format!(
            "overlay: {url} at {x},{y}, alpha {alpha}, blend {blend:?}{}",
            width.map(|w| format!(", width {w}")).unwrap_or_default()
        ),
        Filter::RemoveBg => "removebg: remove the background (AI)".into(),
        Filter::KeepProfile => "keep_profile: keep the source ICC profile".into(),
        Filter::BlurFaces => "blurfaces: blur detected faces (AI)".into(),
        Filter::PixelateFaces => "pixelatefaces: pixelate detected faces (AI)".into(),
        Filter::BlurClass(name) => format!("blurclass({name}): blur detected {name} (AI)"),
        Filter::PixelateClass(name) => {
            format!("pixelateclass({name}): pixelate detected {name} (AI)")
        }
        Filter::FacePad(v) => format!("facepad({v}): extra padding around faces"),
        Filter::Upscale(v) => format!("upscale({v}): AI upscale"),
        Filter::Dpr(v) => format!("dpr({v}): device pixel ratio for the whole request"),
        Filter::Text { text, .. } => format!("text: draw the text {text:?}"),
        Filter::Fill(c) => format!("fill({}): flatten transparency onto this color", hex3(*c)),
        Filter::Radius(corners) => format!(
            "radius: round the corners ({}, {}, {}, {})",
            corners[0], corners[1], corners[2], corners[3]
        ),
        Filter::Border(w, c) => format!("border({w}, {}): paint an inside border", hex3(*c)),
        Filter::Mask(shape) => format!(
            "mask({}): clip to a shape, keeping alpha",
            match shape {
                MaskShape::Circle => "circle",
                MaskShape::Ellipse => "ellipse",
            }
        ),
        Filter::Sepia => "sepia: sepia tone".into(),
        Filter::Duotone(a, b) => {
            format!("duotone({}, {}): two-tone gradient map", hex3(*a), hex3(*b))
        }
        Filter::Monochrome(c) => format!("monochrome({}): single-hue tint", hex3(*c)),
        Filter::Pixelate(v) => format!("pixelate({v}): mosaic into blocks"),
        Filter::Flip(d) => format!(
            "flip({}): mirror the image",
            match d {
                FlipDirection::Horizontal => "h",
                FlipDirection::Vertical => "v",
            }
        ),
        Filter::BlurRegion { x, y, w, h } => {
            format!("blur_region({x},{y},{w},{h}): blur one rectangle")
        }
        Filter::PixelateRegion { x, y, w, h } => {
            format!("pixelate_region({x},{y},{w},{h}): pixelate one rectangle")
        }
        Filter::MaxBytes(v) => format!("max_bytes({v}): cap the output size in bytes"),
        Filter::Quality(q) => match q {
            Quality::Auto => "quality(auto): encode quality".into(),
            Quality::Value(v) => format!("quality({v}): encode quality"),
        },
        Filter::Format(f) => format!("format({}): output as {}", format_name(*f), format_name(*f)),
        Filter::NoUpscale => "no_upscale: never enlarge past the source".into(),
    }
}

fn format_name(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Gif => "gif",
        ImageFormat::WebP => "webp",
        ImageFormat::Avif => "avif",
        ImageFormat::Jxl => "jxl",
    }
}

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("cannot build the HTTP client: {e}"))
}

/// Resolve a verb target to a full URL: full URLs pass
/// through; a remainder is signed first.
fn target_url(config: &Config, input: &str) -> Result<String, String> {
    if input.starts_with("http://") || input.starts_with("https://") {
        return Ok(input.to_string());
    }
    let secret = require_secret(config)?;
    let (_, _, rest) = split_signed(input, &config.base);
    parse_with_presets(&rest)?;
    let sig = crabcrop_url::sign(secret, &rest);
    Ok(format!("{}/{sig}/{rest}", config.base))
}

async fn meta(config: &Config, positional: &[String]) -> Result<String, String> {
    let input = one_target(positional, "meta")?;
    let url = target_url(config, &input)?;
    let mut accept = "application/json".to_string();
    if config.placeholders {
        accept.push_str("; placeholders=1");
    }
    if config.detections {
        accept.push_str("; detections=1");
    }
    let response = http_client()?
        .get(&url)
        .header("accept", accept)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!("HTTP {status}: {}", body.trim()));
    }
    let json: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("the response was not JSON: {e}"))?;
    serde_json::to_string_pretty(&json).map_err(|e| e.to_string())
}

async fn purge(config: &Config, positional: &[String]) -> Result<String, String> {
    let input = one_target(positional, "purge")?;
    let secret = require_secret(config)?;
    let (_, _, rest) = split_signed(&input, &config.base);
    let sig = crate::url::purge_signature(secret, &rest);
    let segment = match &config.key_id {
        Some(kid) => format!("{kid}:{sig}"),
        None => sig,
    };
    post_purge(config, &format!("{}/purge/{segment}/{rest}", config.base)).await
}

async fn purge_group(config: &Config, positional: &[String]) -> Result<String, String> {
    let name = one_target(positional, "purge-group")?;
    let secret = require_secret(config)?;
    let sig = crate::url::purge_group_signature(secret, &name);
    let segment = match &config.key_id {
        Some(kid) => format!("{kid}:{sig}"),
        None => sig,
    };
    post_purge(
        config,
        &format!("{}/purge-group/{segment}/{name}", config.base),
    )
    .await
}

async fn post_purge(config: &Config, url: &str) -> Result<String, String> {
    let _ = config;
    let response = http_client()?
        .post(url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!("HTTP {status}: {}", body.trim()));
    }
    Ok(body.trim().to_string())
}

fn convert_thumbor_cmd(config: &Config, positional: &[String]) -> Result<String, String> {
    let input = one_target(positional, "convert-thumbor")?;
    let (url, warnings) = convert_thumbor(&input, &config.base, config.secret.as_deref())?;
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }
    Ok(url)
}

/// One Thumbor size axis: a pixel count (negative means
/// flip on that axis), or the original dimension.
#[derive(Clone, Copy, PartialEq, Debug)]
enum ThumborDim {
    Pixels(i64),
    Orig,
}

fn parse_thumbor_dim(text: &str) -> Option<ThumborDim> {
    if text == "orig" {
        return Some(ThumborDim::Orig);
    }
    text.parse::<i64>().ok().map(ThumborDim::Pixels)
}

/// Parse a Thumbor size segment (`<w>x<h>`) when the whole
/// segment is one.
fn parse_thumbor_size(seg: &str) -> Option<(ThumborDim, ThumborDim)> {
    let (w, h) = seg.split_once('x')?;
    Some((parse_thumbor_dim(w)?, parse_thumbor_dim(h)?))
}

/// Parse a Thumbor crop segment (`<l>x<t>:<r>x<b>`) when
/// the whole segment is one.
fn parse_thumbor_crop(seg: &str) -> Option<(u32, u32, u32, u32)> {
    let (lt, rb) = seg.split_once(':')?;
    let point = |s: &str| -> Option<(u32, u32)> {
        let (x, y) = s.split_once('x')?;
        Some((x.parse().ok()?, y.parse().ok()?))
    };
    let (l, t) = point(lt)?;
    let (r, b) = point(rb)?;
    if r <= l || b <= t {
        return None;
    }
    Some((l, t, r, b))
}

/// Split a filter chain on ':' at parenthesis depth zero:
/// filter arguments may contain ':' themselves (URLs).
fn split_filter_chain(chain: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (i, c) in chain.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ':' if depth == 0 => {
                parts.push(&chain[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&chain[start..]);
    parts.into_iter().filter(|p| !p.is_empty()).collect()
}

/// Percent-encode a URL used as a filter argument, so it
/// survives inside a crabcrop path segment; the server
/// decodes it back.
fn encode_url_arg(url: &str) -> String {
    let mut out = String::with_capacity(url.len());
    for b in url.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Format a scale factor compactly, with no trailing zeros
/// (`1.2`, `0.5`, `2`).
fn format_factor(f: f64) -> String {
    let s = format!("{f:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-0" {
        "0".to_string()
    } else {
        s.to_string()
    }
}

/// Convert one Thumbor filter to its crabcrop filter text.
/// Anything with no faithful equivalent is dropped and
/// named in `warnings` instead of being guessed at. A
/// `focal(...)` filter is not a crabcrop filter: it lands in
/// the `focal` side channel (the last one in a chain wins)
/// and becomes a `focal:` segment ahead of the crop.
fn convert_thumbor_filter(
    item: &str,
    warnings: &mut Vec<String>,
    focal: &mut Option<(u32, u32, u32, u32)>,
) -> Option<String> {
    let (name, args) = match item.split_once('(') {
        Some((n, rest)) => (n, rest.strip_suffix(')').unwrap_or(rest)),
        None => (item, ""),
    };
    let drop_it = |warnings: &mut Vec<String>, why: String| -> Option<String> {
        warnings.push(why);
        None
    };
    match name {
        "quality" => match args.parse::<u8>() {
            Ok(q) if q <= 100 => Some(format!("quality({q})")),
            _ => drop_it(
                warnings,
                format!("quality({args}) is out of range; dropped"),
            ),
        },
        "format" => match args {
            "jpeg" | "png" | "gif" | "webp" | "avif" => Some(format!("format({args})")),
            _ => drop_it(
                warnings,
                format!("format({args}) has no crabcrop equivalent; dropped"),
            ),
        },
        "max_bytes" => match args.parse::<u64>() {
            Ok(n) if n > 0 => Some(format!("max_bytes({n})")),
            _ => drop_it(
                warnings,
                format!("max_bytes({args}) is not a positive size; dropped"),
            ),
        },
        "no_upscale" => Some("no_upscale".to_string()),
        "grayscale" => Some("grayscale".to_string()),
        "equalize" => Some("equalize".to_string()),
        "rotate" => match args.parse::<i32>() {
            Ok(n) if n % 90 == 0 => Some(format!("rotate({n})")),
            _ => drop_it(
                warnings,
                format!("rotate({args}): crabcrop rotates in 90 degree steps only; dropped"),
            ),
        },
        "blur" => {
            let mut parts = args.split(',');
            let first = parts.next().unwrap_or("");
            match first.parse::<f32>() {
                Ok(v) if v.is_finite() && v > 0.0 => {
                    if parts.next().is_some() {
                        warnings.push(
                            "blur: the second argument (sigma) was dropped; crabcrop blur \
                             takes one value"
                                .to_string(),
                        );
                    }
                    Some(format!("blur({first})"))
                }
                _ => drop_it(warnings, format!("blur({args}) is not usable; dropped")),
            }
        }
        "brightness" => match args.parse::<i32>() {
            Ok(n) => {
                let m = (2.55 * f64::from(n)).round() as i32;
                warnings.push(format!(
                    "brightness({n}) approximated as brighten({m}): Thumbor scales \
                     brightness, crabcrop shifts it"
                ));
                Some(format!("brighten({m})"))
            }
            _ => drop_it(
                warnings,
                format!("brightness({args}) is not a number; dropped"),
            ),
        },
        "contrast" => match args.parse::<i32>() {
            Ok(n) => Some(format!(
                "contrast({})",
                format_factor(1.0 + f64::from(n) / 100.0)
            )),
            _ => drop_it(
                warnings,
                format!("contrast({args}) is not a number; dropped"),
            ),
        },
        "saturation" => match args.parse::<i32>() {
            Ok(n) => {
                let factor = 1.0 + f64::from(n) / 100.0;
                if factor < 0.0 {
                    drop_it(
                        warnings,
                        format!("saturation({n}) is below crabcrop's range; dropped"),
                    )
                } else {
                    Some(format!("saturation({})", format_factor(factor)))
                }
            }
            _ => drop_it(
                warnings,
                format!("saturation({args}) is not a number; dropped"),
            ),
        },
        "rgb" => {
            let parts: Vec<i32> = args.split(',').filter_map(|p| p.parse().ok()).collect();
            if parts.len() == 3 {
                let offsets: Vec<String> = parts
                    .iter()
                    .map(|n| format!("{}", (2.55 * f64::from(*n)).round() as i32))
                    .collect();
                warnings.push(format!(
                    "rgb({args}) approximated as additive offsets rgb({}): Thumbor scales \
                     channels, crabcrop shifts them",
                    offsets.join(",")
                ));
                Some(format!("rgb({})", offsets.join(",")))
            } else {
                drop_it(warnings, format!("rgb({args}) needs three values; dropped"))
            }
        }
        "sharpen" => {
            let mut parts = args.split(',');
            let first = parts.next().unwrap_or("");
            match first.parse::<f32>() {
                Ok(v) if v.is_finite() && v > 0.0 => {
                    if parts.next().is_some() {
                        warnings
                            .push("sharpen: the luminance-only argument was dropped".to_string());
                    }
                    Some(format!("sharpen({first})"))
                }
                _ => drop_it(warnings, format!("sharpen({args}) is not usable; dropped")),
            }
        }
        "watermark" => {
            // <url>,<x>,<y>,<alpha>: split from the right,
            // where the numeric fields are.
            let parts: Vec<&str> = args.rsplitn(4, ',').collect();
            let parsed = (|| {
                if parts.len() != 4 {
                    return None;
                }
                let alpha: u32 = parts[0].parse().ok()?;
                if alpha > 100 {
                    return None;
                }
                let y: i32 = parts[1].parse().ok()?;
                let x: i32 = parts[2].parse().ok()?;
                let url = parts[3];
                if url.is_empty() {
                    return None;
                }
                Some((url, x, y, alpha))
            })();
            match parsed {
                Some((url, x, y, alpha)) => Some(format!(
                    "watermark({},{x},{y},{alpha})",
                    encode_url_arg(url)
                )),
                None => drop_it(warnings, format!("watermark({args}) is malformed; dropped")),
            }
        }
        "fill" => {
            if args == "auto" {
                drop_it(
                    warnings,
                    "fill(auto) has no crabcrop equivalent; dropped".to_string(),
                )
            } else if args.len() == 6 && args.bytes().all(|b| b.is_ascii_hexdigit()) {
                Some(format!("fill({})", args.to_lowercase()))
            } else {
                drop_it(
                    warnings,
                    format!("fill({args}): crabcrop fill takes a 6-digit hex color; dropped"),
                )
            }
        }
        "strip_exif" | "strip_icc" => drop_it(
            warnings,
            format!(
                "{name}: metadata handling is server policy in crabcrop \
                 (CRABCROP_METADATA), not a URL option; nothing converted"
            ),
        ),
        "background_color" => drop_it(
            warnings,
            "background_color(...) has no per-URL equivalent; see the pad operation \
             and the fill filter"
                .to_string(),
        ),
        "focal" => match parse_thumbor_crop(args) {
            Some(rect) => {
                *focal = Some(rect);
                None
            }
            None => drop_it(
                warnings,
                format!("focal({args}): not a <left>x<top>:<right>x<bottom> rectangle; dropped"),
            ),
        },
        _ => drop_it(
            warnings,
            format!("{item} has no crabcrop equivalent; dropped"),
        ),
    }
}

/// Convert a Thumbor URL to a crabcrop URL on `base`,
/// signed with `secret` when given (an `unsafe` URL, plus
/// a warning, otherwise). Returns the URL and the losses
/// and notes the conversion produced, in order. Thumbor
/// signatures never convert: the scheme differs, so the
/// output is always freshly signed.
pub fn convert_thumbor(
    input: &str,
    base: &str,
    secret: Option<&[u8]>,
) -> Result<(String, Vec<String>), String> {
    let mut warnings: Vec<String> = Vec::new();
    let input = input.trim();
    // A full URL loses its scheme and host; a bare path is
    // used as-is. The source inside either form contains
    // its own "://", so a scheme only counts when it
    // starts the input, before the first '/'.
    let as_url = match (input.find("://"), input.find('/')) {
        (Some(scheme), Some(slash)) => scheme <= slash,
        (Some(_), None) => true,
        _ => false,
    };
    let path = if as_url {
        let after = input.split_once("://").map(|(_, a)| a).unwrap_or("");
        after
            .split_once('/')
            .map(|(_, p)| p)
            .ok_or_else(|| format!("not a Thumbor URL: {input:?} has no path"))?
    } else {
        input.strip_prefix('/').unwrap_or(input)
    };
    let segments: Vec<&str> = path.split('/').collect();
    if segments.len() < 2 {
        return Err(format!(
            "not a Thumbor URL: expected /<signature>/<operations>/<source>, got {input:?}"
        ));
    }
    // Segment 0 is the signature: "unsafe", or a base64
    // HMAC-SHA1 that cannot be carried over. It is dropped
    // either way; the output is signed below.
    let mut idx = 1;
    if segments.get(idx) == Some(&"meta") {
        warnings.push(
            "the meta segment has no URL form in crabcrop: fetch the converted URL with \
             Accept: application/json, or run crabcrop meta on it"
                .to_string(),
        );
        idx += 1;
    }

    // The operations, in Thumbor's canonical order; each is
    // optional. The first segment that matches none of them
    // starts the source.
    let mut trim: Option<Option<u8>> = None;
    let mut crop_rect: Option<(u32, u32, u32, u32)> = None;
    let mut fit_in = false;
    let mut size: Option<(ThumborDim, ThumborDim)> = None;
    let mut halign: Option<&str> = None;
    let mut valign: Option<&str> = None;
    let mut smart = false;
    let mut focal: Option<(u32, u32, u32, u32)> = None;
    let mut filters_text: Vec<String> = Vec::new();

    while idx < segments.len() {
        let seg = segments[idx];
        if seg == "trim" || seg.starts_with("trim:") {
            // trim[:<corner>][:<tolerance>]
            let mut tolerance = None;
            for part in seg.split(':').skip(1) {
                if let Ok(t) = part.parse::<u8>() {
                    tolerance = Some(t);
                } else {
                    warnings.push(format!(
                        "trim corner {part:?} dropped: crabcrop trim reads the border itself"
                    ));
                }
            }
            trim = Some(tolerance);
            idx += 1;
        } else if let Some(rect) = parse_thumbor_crop(seg) {
            crop_rect = Some(rect);
            idx += 1;
        } else if matches!(seg, "fit-in" | "adaptive-fit-in" | "full-fit-in") {
            fit_in = true;
            if seg == "adaptive-fit-in" {
                warnings.push(
                    "adaptive-fit-in converted to fit-in: crabcrop has no adaptive mode"
                        .to_string(),
                );
            } else if seg == "full-fit-in" {
                warnings.push(
                    "full-fit-in converted to fit-in: the whole image stays inside the box"
                        .to_string(),
                );
            }
            idx += 1;
        } else if size.is_none() && parse_thumbor_size(seg).is_some() {
            size = parse_thumbor_size(seg);
            idx += 1;
        } else if matches!(seg, "left" | "center" | "right") && halign.is_none() {
            halign = Some(seg);
            idx += 1;
        } else if matches!(seg, "top" | "middle" | "bottom") && valign.is_none() {
            valign = Some(seg);
            idx += 1;
        } else if seg == "smart" {
            smart = true;
            idx += 1;
        } else if let Some(chain) = seg.strip_prefix("filters:") {
            // A filter argument can be a URL with slashes
            // of its own (watermark): keep joining segments
            // until the chain's parentheses balance.
            let mut chain = chain.to_string();
            while chain.matches('(').count() > chain.matches(')').count()
                && idx + 1 < segments.len()
            {
                idx += 1;
                chain.push('/');
                chain.push_str(segments[idx]);
            }
            for item in split_filter_chain(&chain) {
                if let Some(text) = convert_thumbor_filter(item, &mut warnings, &mut focal) {
                    filters_text.push(text);
                }
            }
            idx += 1;
        } else {
            break;
        }
    }
    let source = segments[idx..].join("/");
    if source.is_empty() {
        return Err(format!("not a Thumbor URL: no source image in {input:?}"));
    }

    let align = if smart {
        if halign.is_some() || valign.is_some() {
            warnings.push(
                "the explicit alignment was dropped: smart cropping overrides it".to_string(),
            );
        }
        "smart"
    } else {
        match (halign, valign) {
            (Some("left"), Some("top")) => "top-left",
            (Some("right"), Some("top")) => "top-right",
            (Some("left"), Some("bottom")) => "bottom-left",
            (Some("right"), Some("bottom")) => "bottom-right",
            (_, Some("top")) => "top",
            (_, Some("bottom")) => "bottom",
            (Some("left"), _) => "left",
            (Some("right"), _) => "right",
            _ => "center",
        }
    };

    // Flips ride inside a plain crabcrop size. On fit-in
    // and on crop (both exact-box operations, whose sizes
    // take no flips) they become flip filters instead: for
    // fit-in, in the converted filter chain; for crop, in
    // their own segment ahead of the crop, so an alignment
    // anchors the same region it anchored in Thumbor.
    let mut lead_filters: Vec<String> = Vec::new();
    let mut crop_flips: Vec<String> = Vec::new();
    let size_op = size.map(|(w, h)| {
        let axis = |d: ThumborDim| -> (String, bool, bool) {
            match d {
                ThumborDim::Orig => ("orig".to_string(), false, false),
                ThumborDim::Pixels(v) => {
                    let flip = v < 0;
                    let mag = v.unsigned_abs();
                    let text = if flip {
                        format!("-{mag}")
                    } else {
                        mag.to_string()
                    };
                    (text, mag > 0, flip)
                }
            }
        };
        let (wt, w_live, w_flip) = axis(w);
        let (ht, h_live, h_flip) = axis(h);
        let dims = format!("{wt}x{ht}");
        if fit_in {
            if w_flip {
                lead_filters.push("flip(h)".to_string());
            }
            if h_flip {
                lead_filters.push("flip(v)".to_string());
            }
            let plain = {
                let strip = |t: &str| t.strip_prefix('-').unwrap_or(t).to_string();
                format!("{}x{}", strip(&wt), strip(&ht))
            };
            format!("fit-in:{plain}")
        } else if w_live && h_live {
            if w_flip {
                crop_flips.push("flip(h)".to_string());
            }
            if h_flip {
                crop_flips.push("flip(v)".to_string());
            }
            let plain = {
                let strip = |t: &str| t.strip_prefix('-').unwrap_or(t).to_string();
                format!("{}x{}", strip(&wt), strip(&ht))
            };
            if align == "center" {
                format!("crop:{plain}")
            } else {
                format!("crop:{plain}:{align}")
            }
        } else {
            dims
        }
    });

    let mut ops: Vec<String> = Vec::new();
    if let Some(tolerance) = trim {
        // Trimming happens before any resize in Thumbor, so
        // it gets its own filters segment ahead of the rest.
        ops.push(match tolerance {
            Some(t) => format!("filters:trim({t})"),
            None => "filters:trim".to_string(),
        });
    }
    if let Some((l, t, r, b)) = crop_rect {
        ops.push(format!("crop-rect:{l}x{t}:{r}x{b}"));
    }
    if !crop_flips.is_empty() {
        ops.push(format!("filters:{}", crop_flips.join(":")));
    }
    if let Some((l, t, r, b)) = focal {
        // Ahead of the size op, so the crop it steers is the
        // next one in the converted URL.
        ops.push(format!("focal:{l}x{t}:{r}x{b}"));
    }
    if let Some(op) = size_op {
        ops.push(op);
    }
    let all_filters: Vec<String> = lead_filters.into_iter().chain(filters_text).collect();
    if !all_filters.is_empty() {
        ops.push(format!("filters:{}", all_filters.join(":")));
    }
    let remainder = if ops.is_empty() {
        source.clone()
    } else {
        format!("{}/{source}", ops.join("/"))
    };

    // The converter's own output must satisfy the server
    // grammar; a failure here is a converter bug, not input.
    parse_with_presets(&remainder)
        .map_err(|e| format!("converted URL failed validation ({e}): {remainder}"))?;

    let url = match secret {
        Some(secret) => format!(
            "{base}/{}/{remainder}",
            crabcrop_url::sign(secret, &remainder)
        ),
        None => {
            warnings.push(
                "no secret given: emitted an unsafe URL; sign it before production use".to_string(),
            );
            format!("{base}/unsafe/{remainder}")
        }
    };
    Ok((url, warnings))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[tokio::test]
    async fn help_after_a_verb_is_usage_not_an_error() {
        // `crabcrop sign --help` answers Ok(USAGE) like the
        // top-level spelling, not an Err to stderr.
        let out = exec(&args(&["sign", "--help"])).await.unwrap();
        assert!(out.contains("crabcrop sign"), "{out}");
    }

    #[tokio::test]
    async fn an_unknown_verb_is_an_error() {
        // The dispatch contract behind main's guard: a
        // misspelled verb errors here instead of the
        // process ever reaching the server.
        let err = exec(&args(&["sgin", "300x200/https://e.com/a.jpg"]))
            .await
            .unwrap_err();
        assert!(err.contains("unknown verb \"sgin\""), "{err}");
    }

    #[tokio::test]
    async fn sign_with_a_key_id_stamps_the_segment() {
        let out = exec(&args(&[
            "sign",
            "--secret",
            "test-secret",
            "--key-id",
            "k2",
            "--base",
            "http://img.test",
            "300x200/https://example.com/a.jpg",
        ]))
        .await
        .unwrap();
        let expected_sig = crabcrop_url::sign(b"test-secret", "300x200/https://example.com/a.jpg");
        assert_eq!(
            out,
            format!("http://img.test/k2:{expected_sig}/300x200/https://example.com/a.jpg")
        );
        // And explain names the key on the way back in.
        let explained = exec(&args(&["explain", "--secret", "test-secret", &out]))
            .await
            .unwrap();
        assert!(
            explained.contains("key k2; valid for this remainder"),
            "{explained}"
        );
        // A malformed key ID is refused at flag parsing.
        assert!(
            exec(&args(&[
                "sign",
                "--secret",
                "test-secret",
                "--key-id",
                "bad id",
                "300x200/https://example.com/a.jpg",
            ]))
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn sign_prints_a_validatable_url() {
        let out = exec(&args(&[
            "sign",
            "--secret",
            "test-secret",
            "--base",
            "http://img.test",
            "300x200/https://example.com/a.jpg",
        ]))
        .await
        .unwrap();
        let expected_sig = crabcrop_url::sign(b"test-secret", "300x200/https://example.com/a.jpg");
        assert_eq!(
            out,
            format!("http://img.test/{expected_sig}/300x200/https://example.com/a.jpg")
        );
        // A malformed remainder fails here, not at request time.
        assert!(
            exec(&args(&[
                "sign",
                "--secret",
                "s",
                "300x200/filters:bogus(1)/https://e.com/a.jpg"
            ]))
            .await
            .is_err()
        );
        // No secret, no signature.
        assert!(
            exec(&args(&["sign", "300x200/https://e.com/a.jpg"]))
                .await
                .is_err()
        );
    }

    #[test]
    fn described_filters_round_trip_through_the_parser() {
        // explain's spellings must be grammar: the text
        // before the colon is a filter segment the parser
        // itself accepts (review CL-5).
        for filter in [
            Filter::HueRotate(90),
            Filter::Curve(crate::url::CurveChannel::Rgb, vec![(0, 0), (255, 255)]),
            Filter::Flip(FlipDirection::Horizontal),
            Filter::Flip(FlipDirection::Vertical),
            Filter::Quality(Quality::Value(80)),
            Filter::Quality(Quality::Auto),
            Filter::Mask(MaskShape::Circle),
            Filter::Mask(MaskShape::Ellipse),
        ] {
            let described = describe_filter(&filter);
            let spelling = described.split(':').next().unwrap();
            let rest = format!("filters:{spelling}/https://e.com/a.jpg");
            assert!(
                crate::url::parse_operations(&rest).is_ok(),
                "{spelling} did not re-parse"
            );
        }
    }

    #[tokio::test]
    async fn explain_describes_the_segments_and_verifies() {
        let secret = "test-secret";
        let remainder = "300x200/filters:grayscale:format(webp)/https://example.com/a.jpg";
        let sig = crabcrop_url::sign(secret.as_bytes(), remainder);
        let out = exec(&args(&[
            "explain",
            "--secret",
            secret,
            &format!("http://img.test/{sig}/{remainder}"),
        ]))
        .await
        .unwrap();
        assert!(out.contains("valid for this remainder"), "{out}");
        assert!(out.contains("resize to 300x200"), "{out}");
        assert!(out.contains("grayscale: convert to grayscale"), "{out}");
        assert!(out.contains("format(webp): output as webp"), "{out}");
        assert!(out.contains("source: https://example.com/a.jpg"), "{out}");
        assert!(out.contains("verdict: valid, 2 operations"), "{out}");
        // A tampered remainder fails verification.
        let out = exec(&args(&[
            "explain",
            "--secret",
            secret,
            &format!("http://img.test/{sig}/100x100/https://example.com/a.jpg"),
        ]))
        .await
        .unwrap();
        assert!(out.contains("DOES NOT MATCH"), "{out}");
    }

    #[test]
    fn convert_thumbor_maps_sizes_and_crops() {
        let base = "http://img.test";
        // A fixed size is a crop-to-fill in Thumbor.
        let (url, warnings) = convert_thumbor(
            "unsafe/300x200/https://example.com/cat.jpg",
            base,
            Some(b"s"),
        )
        .unwrap();
        assert_eq!(
            url,
            format!(
                "http://img.test/{}/crop:300x200/https://example.com/cat.jpg",
                crabcrop_url::sign(b"s", "crop:300x200/https://example.com/cat.jpg")
            )
        );
        assert!(warnings.is_empty(), "{warnings:?}");
        // A full URL on the Thumbor host converts the same.
        let (url2, _) = convert_thumbor(
            "https://thumbor.old/unsafe/300x200/https://example.com/cat.jpg",
            base,
            Some(b"s"),
        )
        .unwrap();
        assert_eq!(url, url2);
        // Proportional sizes are plain resizes in both.
        let (url, _) =
            convert_thumbor("unsafe/300x0/https://e.com/a.jpg", base, Some(b"s")).unwrap();
        assert!(url.ends_with("/300x0/https://e.com/a.jpg"), "{url}");
        // fit-in stays fit-in.
        let (url, _) = convert_thumbor(
            "unsafe/fit-in/300x200/https://e.com/a.jpg",
            base,
            Some(b"s"),
        )
        .unwrap();
        assert!(url.contains("/fit-in:300x200/"), "{url}");
        // An explicit crop rectangle is a rectangle, cut
        // before the size's own crop.
        let (url, _) = convert_thumbor(
            "unsafe/10x20:110x220/100x100/https://e.com/a.jpg",
            base,
            Some(b"s"),
        )
        .unwrap();
        assert!(
            url.contains("/crop-rect:10x20:110x220/crop:100x100/"),
            "{url}"
        );
        // Negative dimensions are flips. On a crop they
        // become a flip segment ahead of it (crop sizes
        // take no flips); on a plain size they ride inside.
        let (url, _) =
            convert_thumbor("unsafe/-300x200/https://e.com/a.jpg", base, Some(b"s")).unwrap();
        assert!(url.contains("/filters:flip(h)/crop:300x200/"), "{url}");
        let (url, _) =
            convert_thumbor("unsafe/-300x0/https://e.com/a.jpg", base, Some(b"s")).unwrap();
        assert!(url.contains("/-300x0/"), "{url}");
        // ...except on fit-in, where they become flip filters.
        let (url, _) = convert_thumbor(
            "unsafe/fit-in/-300x200/https://e.com/a.jpg",
            base,
            Some(b"s"),
        )
        .unwrap();
        assert!(url.contains("/fit-in:300x200/filters:flip(h)/"), "{url}");
        // No operations at all: the source passes through.
        let (url, warnings) =
            convert_thumbor("unsafe/https://e.com/a.jpg", base, Some(b"s")).unwrap();
        assert!(url.ends_with("/https://e.com/a.jpg"), "{url}");
        assert!(warnings.is_empty(), "{warnings:?}");
        // No secret: an unsafe URL and a warning, never a
        // carried-over Thumbor signature.
        let (url, warnings) =
            convert_thumbor("AbCdEf/300x200/https://e.com/a.jpg", base, None).unwrap();
        assert!(
            url.starts_with("http://img.test/unsafe/crop:300x200/"),
            "{url}"
        );
        assert!(
            warnings.iter().any(|w| w.contains("unsafe")),
            "{warnings:?}"
        );
        // Not a URL at all.
        assert!(convert_thumbor("hello", base, None).is_err());
    }

    #[test]
    fn convert_thumbor_maps_alignment_and_smart() {
        let base = "http://img.test";
        let (url, warnings) =
            convert_thumbor("unsafe/300x200/right/https://e.com/a.jpg", base, Some(b"s")).unwrap();
        assert!(url.contains("/crop:300x200:right/"), "{url}");
        assert!(warnings.is_empty(), "{warnings:?}");
        let (url, _) =
            convert_thumbor("unsafe/300x200/smart/https://e.com/a.jpg", base, Some(b"s")).unwrap();
        assert!(url.contains("/crop:300x200:smart/"), "{url}");
        // A corner pair maps to the corner alignment, exactly.
        let (url, warnings) = convert_thumbor(
            "unsafe/300x200/left/top/https://e.com/a.jpg",
            base,
            Some(b"s"),
        )
        .unwrap();
        assert!(url.contains("/crop:300x200:top-left/"), "{url}");
        assert!(warnings.is_empty(), "{warnings:?}");
        let (url, _) = convert_thumbor(
            "unsafe/300x200/right/bottom/https://e.com/a.jpg",
            base,
            Some(b"s"),
        )
        .unwrap();
        assert!(url.contains("/crop:300x200:bottom-right/"), "{url}");
        // A centered horizontal still collapses to the plain side.
        let (url, _) = convert_thumbor(
            "unsafe/300x200/center/top/https://e.com/a.jpg",
            base,
            Some(b"s"),
        )
        .unwrap();
        assert!(url.contains("/crop:300x200:top/"), "{url}");
    }

    #[test]
    fn convert_thumbor_maps_filters_and_reports_losses() {
        let base = "http://img.test";
        let conv = |u: &str| convert_thumbor(u, base, Some(b"s")).unwrap();
        let (url, warnings) =
            conv("unsafe/300x200/filters:quality(80):grayscale():format(webp)/https://e.com/a.jpg");
        assert!(
            url.contains("/filters:quality(80):grayscale:format(webp)/"),
            "{url}"
        );
        assert!(warnings.is_empty(), "{warnings:?}");
        // Percent scales become factors.
        let (url, _) = conv("unsafe/filters:contrast(20):saturation(-50)/https://e.com/a.jpg");
        assert!(
            url.contains("contrast(1.2)") && url.contains("saturation(0.5)"),
            "{url}"
        );
        // Brightness is an approximation, and says so.
        let (url, warnings) = conv("unsafe/filters:brightness(20)/https://e.com/a.jpg");
        assert!(url.contains("brighten(51)"), "{url}");
        assert!(
            warnings.iter().any(|w| w.contains("approximated")),
            "{warnings:?}"
        );
        // A watermark URL is percent-encoded for its segment.
        let (url, _) =
            conv("unsafe/filters:watermark(https://e.com/logo.png,10,-10,50)/https://e.com/a.jpg");
        assert!(
            url.contains("watermark(https%3A%2F%2Fe.com%2Flogo.png,10,-10,50)"),
            "{url}"
        );
        // The unconvertible are dropped and named. (Comma-form
        // focal arguments are not the rectangle grammar, so
        // this focal is malformed and drops like the rest.)
        let (url, warnings) =
            conv("unsafe/filters:focal(1,2,3,4):rotate(45):fill(auto)/https://e.com/a.jpg");
        assert!(!url.contains("focal") && !url.contains("rotate"), "{url}");
        assert_eq!(warnings.len(), 3, "{warnings:?}");
        // A rectangle focal converts to a focal: segment ahead
        // of the crop it steers.
        let (url, warnings) =
            conv("unsafe/300x200/smart/filters:focal(10x20:110x220)/https://e.com/a.jpg");
        assert!(
            url.contains("/focal:10x20:110x220/crop:300x200:smart/"),
            "{url}"
        );
        assert!(warnings.is_empty(), "{warnings:?}");
        // strip_exif is a policy note, not a filter.
        let (_, warnings) = conv("unsafe/filters:strip_exif()/https://e.com/a.jpg");
        assert!(
            warnings.iter().any(|w| w.contains("CRABCROP_METADATA")),
            "{warnings:?}"
        );
        // Trim leads, in its own segment, before the resize.
        let (url, warnings) = conv("unsafe/trim:top-left:25/300x200/https://e.com/a.jpg");
        assert!(url.contains("/filters:trim(25)/crop:300x200/"), "{url}");
        assert!(
            warnings.iter().any(|w| w.contains("trim corner")),
            "{warnings:?}"
        );
        // Meta becomes a note; the URL itself converts.
        let (url, warnings) = conv("unsafe/meta/300x200/https://e.com/a.jpg");
        assert!(url.contains("/crop:300x200/"), "{url}");
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("Accept: application/json")),
            "{warnings:?}"
        );
    }

    #[test]
    fn meta_opt_in_flags_parse_and_default_off() {
        let (config, _) = parse_flags(&args(&["--placeholders", "--detections"])).unwrap();
        assert!(config.placeholders && config.detections);
        let (config, _) = parse_flags(&args(&[])).unwrap();
        assert!(!config.placeholders && !config.detections);
        assert!(parse_flags(&args(&["--placeholders=1"])).is_err());
    }

    #[test]
    fn flags_override_env_and_positional_split_works() {
        let (config, positional) = parse_flags(&args(&[
            "--base=https://cdn.test/",
            "--secret",
            "flag-secret",
            "300x200/https://e.com/a.jpg",
        ]))
        .unwrap();
        assert_eq!(config.base, "https://cdn.test");
        assert_eq!(config.secret.as_deref(), Some(b"flag-secret".as_slice()));
        assert_eq!(positional, ["300x200/https://e.com/a.jpg"]);
    }
}
