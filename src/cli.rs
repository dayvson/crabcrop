//! Operator CLI verbs, dispatched from `main` before the
//! server starts: `sign`, `explain`, `meta`, `purge`, and
//! `purge-group`. Configuration comes from `--base` /
//! `--secret` flags over `CRABCROP_BASE` / `CRABCROP_SECRET`,
//! the same convention as the MCP server. The verbs use the
//! server's own URL grammar, so what the CLI validates is
//! what the server will do.

use crate::url::{Dimension, Filter, ImageFormat, Operation, Signature};

pub const VERBS: [&str; 5] = ["sign", "explain", "meta", "purge", "purge-group"];

const USAGE: &str = "\
crabcrop: on-demand image optimization service

Server:
  crabcrop                      Start the server (configuration via environment).

CLI verbs (flags: --base <url>, --secret <secret>; env: CRABCROP_BASE, CRABCROP_SECRET):
  crabcrop sign <remainder>           Print the signed URL for an operations + source remainder.
  crabcrop explain <url-or-path>      Break a URL down segment by segment.
  crabcrop meta <url-or-remainder>    Fetch and print the meta JSON for an image.
  crabcrop purge <url-or-remainder>   Purge one URL's cached variants.
  crabcrop purge-group <name>         Purge every cached entry in a preset group.";

struct Config {
    base: String,
    secret: Option<Vec<u8>>,
}

fn parse_flags(args: &[String]) -> Result<(Config, Vec<String>), String> {
    let mut base: Option<String> = None;
    let mut secret: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let (flag, inline_value) = match arg.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_string())),
            _ => (arg.as_str(), None),
        };
        match flag {
            "--base" | "--secret" => {
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
                } else {
                    secret = Some(value);
                }
            }
            "--help" | "-h" => return Err(USAGE.to_string()),
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
    Ok((Config { base, secret }, positional))
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
fn split_signed(input: &str, base: &str) -> (Option<Signature>, String) {
    let input = input.trim();
    // A URL on the configured base: strip the base, then
    // read the signature segment.
    if let Some(rest) = input.strip_prefix(&format!("{base}/")) {
        return match crate::url::parse_signature(rest) {
            Ok((sig, remainder)) => (Some(sig), remainder.to_string()),
            Err(_) => (None, rest.to_string()),
        };
    }
    // A full URL on any other host counts only when its
    // path really starts with a signature segment: a bare
    // remainder also contains "://" (in its source), so
    // anything less strict would mangle it.
    if let Some((_, after_scheme)) = input.split_once("://") {
        let path = after_scheme.split_once('/').map(|(_, p)| p).unwrap_or("");
        if let Ok((sig, remainder)) = crate::url::parse_signature(path) {
            return (Some(sig), remainder.to_string());
        }
        return (None, input.to_string());
    }
    let path = input.strip_prefix('/').unwrap_or(input);
    match crate::url::parse_signature(path) {
        Ok((sig, rest)) => (Some(sig), rest.to_string()),
        Err(_) => (None, path.to_string()),
    }
}

/// Parse a remainder that may lead with preset references:
/// drop leading segments until the grammar accepts the
/// rest, returning how many were dropped. Server presets
/// expand before parsing, so the CLI cannot parse them
/// itself; it can only step aside.
fn parse_with_presets(rest: &str) -> Result<(usize, Vec<Operation>, String, Option<u64>), String> {
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
            Ok((ops, source, expiry)) => {
                return Ok((dropped, ops, source.to_string(), expiry));
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
    let (config, positional) = parse_flags(&args[1..])?;
    match verb {
        "sign" => sign(&config, &positional),
        "explain" => Ok(explain(&config, &positional)?),
        "meta" => meta(&config, &positional).await,
        "purge" => purge(&config, &positional).await,
        "purge-group" => purge_group(&config, &positional).await,
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
    Ok(format!("{}/{sig}/{remainder}", config.base))
}

fn explain(config: &Config, positional: &[String]) -> Result<String, String> {
    let input = one_target(positional, "explain")?;
    let (signature, rest) = split_signed(&input, &config.base);
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
            lines.push(format!("signature: {hex} ({note})"));
        }
        Some(Signature::Unsafe) => lines.push("signature: unsafe (unsigned URL)".to_string()),
        None => lines.push("signature: none (bare remainder)".to_string()),
    }
    let segments: Vec<&str> = rest.split('/').collect();
    match parse_with_presets(&rest) {
        Ok((dropped, ops, source, expiry)) => {
            for seg in &segments[..dropped] {
                lines.push(format!(
                    "  {seg}  preset reference (expanded by the server)"
                ));
            }
            let mut op_iter = ops.iter();
            let mut expiry_printed = false;
            for seg in &segments[dropped..segments.len() - source.matches('/').count() - 1] {
                if let Some(ts) = seg.strip_prefix("exp:") {
                    lines.push(format!("  {seg}  link expires at unix time {ts}"));
                    expiry_printed = true;
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
        Filter::HueRotate(v) => format!("hue_rotate({v}): rotate the hue"),
        Filter::Sharpen(v) => format!("sharpen({v}): sharpen"),
        Filter::Saturation(v) => format!("saturation({v}): adjust saturation"),
        Filter::Rgb(r, g, b) => format!("rgb({r},{g},{b}): shift the channels"),
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
        Filter::Mask(shape) => format!("mask({shape:?}): clip to a shape, keeping alpha"),
        Filter::Sepia => "sepia: sepia tone".into(),
        Filter::Duotone(a, b) => {
            format!("duotone({}, {}): two-tone gradient map", hex3(*a), hex3(*b))
        }
        Filter::Monochrome(c) => format!("monochrome({}): single-hue tint", hex3(*c)),
        Filter::Pixelate(v) => format!("pixelate({v}): mosaic into blocks"),
        Filter::Flip(d) => format!("flip({d:?}): mirror the image"),
        Filter::BlurRegion { x, y, w, h } => {
            format!("blur_region({x},{y},{w},{h}): blur one rectangle")
        }
        Filter::PixelateRegion { x, y, w, h } => {
            format!("pixelate_region({x},{y},{w},{h}): pixelate one rectangle")
        }
        Filter::MaxBytes(v) => format!("max_bytes({v}): cap the output size in bytes"),
        Filter::Quality(q) => format!("quality({q:?}): encode quality"),
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
    let (_, rest) = split_signed(input, &config.base);
    parse_with_presets(&rest)?;
    let sig = crabcrop_url::sign(secret, &rest);
    Ok(format!("{}/{sig}/{rest}", config.base))
}

async fn meta(config: &Config, positional: &[String]) -> Result<String, String> {
    let input = one_target(positional, "meta")?;
    let url = target_url(config, &input)?;
    let response = http_client()?
        .get(&url)
        .header("accept", "application/json")
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
    let (_, rest) = split_signed(&input, &config.base);
    let sig = crate::url::purge_signature(secret, &rest);
    post_purge(config, &format!("{}/purge/{sig}/{rest}", config.base)).await
}

async fn purge_group(config: &Config, positional: &[String]) -> Result<String, String> {
    let name = one_target(positional, "purge-group")?;
    let secret = require_secret(config)?;
    let sig = crate::url::purge_signature(secret, &format!("purge-group/{name}"));
    post_purge(config, &format!("{}/purge-group/{sig}/{name}", config.base)).await
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
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
