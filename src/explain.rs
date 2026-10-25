//! The self-explaining URL report (issue #196): what the
//! server's `GET /explain/<remainder>` route renders once
//! the remainder has been verified exactly like an image
//! request and its source probed through the loader.
//!
//! Everything in the report is derivable from the URL
//! itself plus the source's own headers. It carries no
//! secret or derived key material, no configuration values
//! (no model or file paths, no upload roots), no config
//! fingerprint, no filesystem paths, and no internal error
//! text: the handler maps failures to the frozen codes and
//! generic messages before anything reaches this module.
//! The one judgement the report makes beyond the URL's own
//! text is the forecast, a walk over the operation chain
//! in the engine's own arithmetic; it is a forecast, and
//! the response headers say `no-store` because it is one.

use crate::engine::SourceProbe;
use crate::url::{Dimension, Filter, Operation, RequestDirectives, canonical_form, format_text};

/// One operation in the report: its canonical segment and
/// its plain-language gloss lines, from the CLI explain
/// gloss table (the single source both surfaces share).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpEntry {
    pub segment: String,
    pub descriptions: Vec<String>,
}

/// One referenced preset: its name and the operations its
/// definition expands to, glossed like any other chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresetEntry {
    pub name: String,
    pub operations: Vec<OpEntry>,
}

/// The forecast: what the operation chain determines about
/// the output. Dimensions and pixels are `None` when the
/// chain's result depends on facts the probe does not have
/// (an unprobeable source size, a fit against an unknown
/// aspect). The pixel count is per frame; the budget it is
/// compared against is the engine's aggregate output
/// budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Forecast {
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// The output format: the chain's `format(...)` when it
    /// names one, otherwise the format a plain request with
    /// no usable `Accept` preference gets, which is the
    /// detected source format (JPEG when the source kind is
    /// not one the server emits).
    pub format: String,
    /// True when `format` came from the chain itself.
    pub format_from_operation: bool,
    pub pixels: Option<u64>,
    pub over_budget: bool,
}

/// Everything the handler verified and probed, ready to
/// render. Built only after signature and expiry checks
/// have passed, so the verdict it carries is always the
/// positive one; refusals never reach this module.
pub struct ReportInput {
    /// False for an `unsafe` remainder.
    pub signed: bool,
    pub key_id: Option<String>,
    pub expiry: Option<u64>,
    /// Current unix time, for the expiry verdict.
    pub now: u64,
    /// The source exactly as written in the URL.
    pub source_url: String,
    /// Header facts about the probed source.
    pub probe: SourceProbe,
    /// The probed body's encoded size in bytes.
    pub source_bytes: usize,
    /// The request's own operations, presets expanded.
    pub ops: Vec<Operation>,
    pub presets: Vec<PresetEntry>,
    /// The canonical remainder (the cache key's path form;
    /// the fingerprint suffix the key carries is operator
    /// state and is never reported).
    pub canonical: String,
    pub cache_group: Option<String>,
    /// The image request path exactly as presented, for
    /// the report's link back to the image itself.
    pub image_path: String,
}

/// A built report: the input plus its forecast and its
/// debug-view remainder.
pub struct Report {
    input: ReportInput,
    forecast: Forecast,
    debug_focal_remainder: Option<String>,
}

/// Report entries for a chain of operations.
pub fn op_entries(ops: &[Operation]) -> Vec<OpEntry> {
    ops.iter()
        .map(|op| OpEntry {
            segment: canonical_segment(op),
            descriptions: crate::cli::describe_operation_text(op),
        })
        .collect()
}

/// A preset entry from its name and its definition's
/// parsed operations.
pub fn preset_entry(name: &str, ops: &[Operation]) -> PresetEntry {
    PresetEntry {
        name: name.to_string(),
        operations: op_entries(ops),
    }
}

/// The canonical spelling of a single operation: the
/// whole-chain canonicalizer applied to the one operation,
/// minus the empty source it appends.
fn canonical_segment(op: &Operation) -> String {
    let text = canonical_form(std::slice::from_ref(op), "");
    text.trim_end_matches('/').to_string()
}

/// The remainder that renders this chain's debug focal
/// view: the canonical segments with `debug:focal`
/// inserted before the first crop, plus the source. A
/// remainder like any other, so it is signed like any
/// other before a server will serve it. `None` when the
/// chain has no crop for the view to annotate.
pub fn debug_focal_remainder(ops: &[Operation], source: &str) -> Option<String> {
    let crop_at = ops
        .iter()
        .position(|op| matches!(op, Operation::Crop { .. }))?;
    let mut segments: Vec<String> = ops.iter().map(canonical_segment).collect();
    segments.insert(crop_at, "debug:focal".to_string());
    Some(format!("{}/{source}", segments.join("/")))
}

/// Walk the chain over the probed source size and settle
/// what the output will be. The arithmetic mirrors the
/// engine's own size resolution (DPR scaling of pixel
/// axes, the adaptive fit-in box swap, proportional axes
/// resolved against the running size, `no_upscale` capping
/// resizes at the source), simplified where the engine's
/// answer depends on decoded pixels: a step it cannot
/// settle makes the dimensions unknown from there on,
/// never a guess.
fn forecast(ops: &[Operation], probe: &SourceProbe) -> Forecast {
    let directives = RequestDirectives::analyze(ops);
    let dpr = directives.dpr.unwrap_or(1.0);
    let mut cur: Option<(u64, u64)> = probe.dimensions.map(|(w, h)| (w as u64, h as u64));
    for op in ops {
        cur = match op {
            Operation::Size(size) => {
                let resolved = resolve_size(size, cur, dpr);
                match (resolved, cur) {
                    (Some((w, h)), Some((cw, ch))) if directives.no_upscale => {
                        Some((w.min(cw), h.min(ch)))
                    }
                    (resolved, _) => resolved,
                }
            }
            Operation::Crop { size, .. } | Operation::Pad { size, .. } => {
                resolve_size(size, cur, dpr)
            }
            Operation::FitIn { size, adaptive } => fit_in_forecast(size, *adaptive, cur, dpr),
            Operation::CropRect {
                left,
                top,
                right,
                bottom,
            } => Some(((*right - *left) as u64, (*bottom - *top) as u64)),
            Operation::FocalRect { .. } | Operation::DebugFocal => cur,
            Operation::Filters(filters) => {
                let mut cur = cur;
                for filter in filters {
                    if let Filter::Upscale(n) = filter {
                        cur = cur.map(|(w, h)| (w * *n as u64, h * *n as u64));
                    }
                }
                cur
            }
        };
    }
    let (width, height) = match cur {
        Some((w, h)) => (
            Some(w.min(u32::MAX as u64) as u32),
            Some(h.min(u32::MAX as u64) as u32),
        ),
        None => (None, None),
    };
    let pixels = cur.map(|(w, h)| w * h);
    let (format, format_from_operation) = match directives.explicit_format {
        Some(format) => (format_text(format).to_string(), true),
        None => (default_format(probe.kind).to_string(), false),
    };
    Forecast {
        width,
        height,
        format,
        format_from_operation,
        pixels,
        over_budget: pixels.is_some_and(|p| p > crate::engine::MAX_OUTPUT_PIXELS),
    }
}

/// The container a request with no explicit format and no
/// usable `Accept` preference is served in: the detected
/// source format when the server emits that kind, JPEG
/// otherwise (the engine's `detect_format` rule).
fn default_format(kind: &str) -> &'static str {
    match kind {
        "jpeg" => "jpeg",
        "png" => "png",
        "gif" => "gif",
        "webp" => "webp",
        "avif" => "avif",
        "jxl" => "jxl",
        _ => "jpeg",
    }
}

/// Scale one pixel axis by the DPR factor, the engine's
/// rounding (`scale_px`): identity at 1.0, otherwise round
/// and never below 1.
fn scale_px(n: u32, dpr: f32) -> u64 {
    if dpr == 1.0 {
        n as u64
    } else {
        ((n as f32 * dpr).round() as u64).max(1)
    }
}

/// Resolve a size operation's box against the running
/// size: pixel axes are DPR-scaled, `orig` axes take the
/// running size, and a proportional axis follows the other
/// axis's change. Without a running size only a fully
/// pinned box resolves.
fn resolve_size(
    size: &crate::url::ImageSize,
    cur: Option<(u64, u64)>,
    dpr: f32,
) -> Option<(u64, u64)> {
    let width = match size.width {
        Dimension::Pixels(n) => Some(scale_px(n, dpr)),
        Dimension::Original => cur.map(|(w, _)| w),
        Dimension::Proportional => None,
    };
    let height = match size.height {
        Dimension::Pixels(n) => Some(scale_px(n, dpr)),
        Dimension::Original => cur.map(|(_, h)| h),
        Dimension::Proportional => None,
    };
    match (width, height) {
        (Some(w), Some(h)) => Some((w, h)),
        (Some(w), None) => cur.map(|(cw, ch)| (w, (ch * w / cw.max(1)).max(1))),
        (None, Some(h)) => cur.map(|(cw, ch)| ((cw * h / ch.max(1)).max(1), h)),
        (None, None) => cur,
    }
}

/// The fit-in forecast: swap the box when `adaptive` and
/// the orientations disagree (the engine's `fit_in_box`
/// rule: only a fully pinned box has an orientation), then
/// scale the running size by the tightest axis ratio, an
/// `orig` axis bounding the scale at 1.
fn fit_in_forecast(
    size: &crate::url::ImageSize,
    adaptive: bool,
    cur: Option<(u64, u64)>,
    dpr: f32,
) -> Option<(u64, u64)> {
    let (cw, ch) = cur?;
    let (mut bw, mut bh) = (size.width, size.height);
    if adaptive
        && let (Dimension::Pixels(w), Dimension::Pixels(h)) = (bw, bh)
        && ((ch > cw && w > h) || (cw > ch && h > w))
    {
        std::mem::swap(&mut bw, &mut bh);
    }
    let mut scale = f64::INFINITY;
    match bw {
        Dimension::Pixels(n) => scale = scale.min(scale_px(n, dpr) as f64 / cw as f64),
        Dimension::Original => scale = scale.min(1.0),
        Dimension::Proportional => {}
    }
    match bh {
        Dimension::Pixels(n) => scale = scale.min(scale_px(n, dpr) as f64 / ch as f64),
        Dimension::Original => scale = scale.min(1.0),
        Dimension::Proportional => {}
    }
    if !scale.is_finite() {
        return cur;
    }
    Some((
        ((cw as f64 * scale).round() as u64).max(1),
        ((ch as f64 * scale).round() as u64).max(1),
    ))
}

impl Report {
    pub fn new(input: ReportInput) -> Self {
        let forecast = forecast(&input.ops, &input.probe);
        let debug_focal_remainder = debug_focal_remainder(&input.ops, &input.source_url);
        Self {
            input,
            forecast,
            debug_focal_remainder,
        }
    }

    /// The JSON form: the same facts the HTML page shows,
    /// as data.
    pub fn to_json(&self) -> serde_json::Value {
        let input = &self.input;
        let ops_json = |entries: &[OpEntry]| {
            serde_json::Value::Array(
                entries
                    .iter()
                    .map(|entry| {
                        serde_json::json!({
                            "segment": entry.segment,
                            "descriptions": entry.descriptions,
                        })
                    })
                    .collect(),
            )
        };
        serde_json::json!({
            "signature": {
                "status": "valid",
                "signed": input.signed,
                "key_id": input.key_id,
            },
            "expiry": input.expiry.map(|exp| serde_json::json!({
                "expires_at": exp,
                "expired": input.now >= exp,
            })),
            "source": {
                "url": input.source_url,
                "kind": input.probe.kind,
                "width": input.probe.dimensions.map(|(w, _)| w),
                "height": input.probe.dimensions.map(|(_, h)| h),
                "bytes": input.source_bytes,
            },
            "operations": ops_json(&op_entries(&input.ops)),
            "presets": input.presets.iter().map(|preset| serde_json::json!({
                "name": preset.name,
                "operations": ops_json(&preset.operations),
            })).collect::<Vec<_>>(),
            "forecast": {
                "width": self.forecast.width,
                "height": self.forecast.height,
                "format": self.forecast.format,
                "format_from": if self.forecast.format_from_operation { "operation" } else { "source" },
                "pixels": self.forecast.pixels,
                "pixel_budget": crate::engine::MAX_OUTPUT_PIXELS,
                "over_budget": self.forecast.over_budget,
            },
            "canonical": input.canonical,
            "cache_group": input.cache_group,
            "links": {
                "image": input.image_path,
                "debug_focal": self.debug_focal_remainder,
            },
        })
    }

    /// The HTML form: a small script-free page. Every
    /// interpolated string passes through
    /// [`escape_html`]; the handler serves it under
    /// `Content-Security-Policy: default-src 'none'`, so
    /// even a missed escape has no script to run.
    pub fn to_html(&self) -> String {
        let input = &self.input;
        let mut out = String::with_capacity(4096);
        out.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<title>crabcrop URL report</title>\n</head>\n<body>\n<h1>crabcrop URL report</h1>\n");

        out.push_str("<h2>Verdict</h2>\n<ul>\n");
        if input.signed {
            out.push_str("<li>Signature: valid");
            if let Some(key_id) = &input.key_id {
                out.push_str(&format!(" (key {})", escape_html(key_id)));
            }
            out.push_str("</li>\n");
        } else {
            out.push_str(
                "<li>Signature: unsafe (unsigned URL, honored by this server's policy)</li>\n",
            );
        }
        if let Some(exp) = input.expiry {
            out.push_str(&format!(
                "<li>Expires at unix time {exp} (not expired)</li>\n"
            ));
        }
        out.push_str("</ul>\n");

        out.push_str("<h2>Source</h2>\n<ul>\n");
        out.push_str(&format!("<li>{}</li>\n", escape_html(&input.source_url)));
        let dims = match input.probe.dimensions {
            Some((w, h)) => format!(", {w}x{h}"),
            None => String::new(),
        };
        out.push_str(&format!(
            "<li>{}{}, {} bytes</li>\n",
            escape_html(input.probe.kind),
            dims,
            input.source_bytes
        ));
        out.push_str("</ul>\n");

        out.push_str("<h2>Operations, in order</h2>\n<ol>\n");
        for entry in op_entries(&input.ops) {
            out.push_str(&format!("<li><code>{}</code>", escape_html(&entry.segment)));
            for description in &entry.descriptions {
                out.push_str(&format!("<br>{}", escape_html(description)));
            }
            out.push_str("</li>\n");
        }
        out.push_str("</ol>\n");

        if !input.presets.is_empty() {
            out.push_str("<h2>Presets</h2>\n");
            for preset in &input.presets {
                out.push_str(&format!(
                    "<h3>preset:{}</h3>\n<ol>\n",
                    escape_html(&preset.name)
                ));
                for entry in &preset.operations {
                    out.push_str(&format!("<li><code>{}</code>", escape_html(&entry.segment)));
                    for description in &entry.descriptions {
                        out.push_str(&format!("<br>{}", escape_html(description)));
                    }
                    out.push_str("</li>\n");
                }
                out.push_str("</ol>\n");
            }
        }

        out.push_str("<h2>Forecast</h2>\n<ul>\n");
        match (self.forecast.width, self.forecast.height) {
            (Some(w), Some(h)) => {
                out.push_str(&format!("<li>Output: {w}x{h}</li>\n"));
            }
            _ => {
                out.push_str("<li>Output dimensions: set by the source</li>\n");
            }
        }
        out.push_str(&format!(
            "<li>Format: {} ({})</li>\n",
            escape_html(&self.forecast.format),
            if self.forecast.format_from_operation {
                "from the URL"
            } else {
                "the source format by default; Accept negotiation may change it"
            }
        ));
        if let Some(pixels) = self.forecast.pixels {
            out.push_str(&format!(
                "<li>Output pixels: {pixels} of {} budget{}</li>\n",
                crate::engine::MAX_OUTPUT_PIXELS,
                if self.forecast.over_budget {
                    "; over the budget, so the image request itself would be refused"
                } else {
                    ""
                }
            ));
        }
        out.push_str("</ul>\n");

        out.push_str("<h2>Canonical form</h2>\n");
        out.push_str(&format!(
            "<p><code>{}</code></p>\n",
            escape_html(&input.canonical)
        ));
        out.push_str(&format!(
            "<p>Cache group: {}</p>\n",
            match &input.cache_group {
                Some(group) => escape_html(group),
                None => "(none)".to_string(),
            }
        ));

        out.push_str("<h2>Links</h2>\n<ul>\n");
        out.push_str(&format!(
            "<li><a href=\"{}\">The image itself</a></li>\n",
            escape_html(&input.image_path)
        ));
        if let Some(remainder) = &self.debug_focal_remainder {
            out.push_str(&format!(
                "<li>Debug focal view: sign this remainder like any URL to render it: <code>{}</code></li>\n",
                escape_html(remainder)
            ));
        }
        out.push_str("</ul>\n</body>\n</html>\n");
        out
    }
}

/// Escape a string for HTML text and attribute contexts:
/// the five markup-significant characters become entities,
/// so caller-chosen text (a source URL, a preset name, a
/// filter argument) renders as text, never as markup.
pub fn escape_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Whether the client asked for the JSON report: an
/// `application/json` entry that is not refused with q=0,
/// with no `text/html` entry beside it. A browser's Accept
/// names HTML, so it gets the page; a bare `*/*` gets the
/// page too, the readable form being the point of the
/// route.
pub fn wants_json(accept: Option<&str>) -> bool {
    let Some(accept) = accept else {
        return false;
    };
    let (mut json, mut html) = (false, false);
    for part in accept.split(',') {
        let mut segments = part.split(';');
        let media = segments.next().unwrap_or("").trim();
        let refused = segments.any(|p| {
            p.trim()
                .strip_prefix("q=")
                .and_then(|v| v.trim().parse::<f32>().ok())
                .is_some_and(|q| q == 0.0)
        });
        if refused {
            continue;
        }
        match media {
            "application/json" => json = true,
            "text/html" => html = true,
            _ => {}
        }
    }
    json && !html
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::url::parse_operations_with_expiry;

    fn report_for(remainder: &str, probe: SourceProbe) -> Report {
        let (ops, source, expiry, _) = parse_operations_with_expiry(remainder).unwrap();
        Report::new(ReportInput {
            signed: true,
            key_id: None,
            expiry,
            now: 1_700_000_000,
            source_url: source.to_string(),
            probe,
            source_bytes: 412_000,
            presets: Vec::new(),
            canonical: crate::url::canonical_form(&ops, source),
            cache_group: None,
            image_path: format!("/sig/{remainder}"),
            ops,
        })
    }

    fn png_probe() -> SourceProbe {
        SourceProbe {
            kind: "png",
            dimensions: Some((2400, 1600)),
        }
    }

    #[test]
    fn forecast_follows_the_chain() {
        let report = report_for(
            "crop:1200x630:smart/filters:format(avif):quality(auto)/https://example.com/photo.jpg",
            png_probe(),
        );
        assert_eq!(report.forecast.width, Some(1200));
        assert_eq!(report.forecast.height, Some(630));
        assert_eq!(report.forecast.format, "avif");
        assert!(report.forecast.format_from_operation);
        assert_eq!(report.forecast.pixels, Some(1200 * 630));
        assert!(!report.forecast.over_budget);
        assert_eq!(
            report.debug_focal_remainder.as_deref(),
            Some(
                "debug:focal/crop:1200x630:smart/filters:format(avif):quality(auto)/https://example.com/photo.jpg"
            )
        );
    }

    #[test]
    fn forecast_resolves_proportional_and_fit_in() {
        let report = report_for("800x-/https://example.com/photo.jpg", png_probe());
        assert_eq!(
            (report.forecast.width, report.forecast.height),
            (Some(800), Some(533))
        );
        let report = report_for("fit-in:300x300/https://example.com/photo.jpg", png_probe());
        assert_eq!(
            (report.forecast.width, report.forecast.height),
            (Some(300), Some(200))
        );
        // No crop anywhere: no debug remainder.
        assert!(report.debug_focal_remainder.is_none());
    }

    #[test]
    fn forecast_defaults_format_to_the_source_kind() {
        let report = report_for("100x100/https://example.com/photo.jpg", png_probe());
        assert_eq!(report.forecast.format, "png");
        assert!(!report.forecast.format_from_operation);
    }

    #[test]
    fn json_carries_the_verdict_source_and_forecast() {
        let report = report_for(
            "exp:1893456000/100x50/https://example.com/photo.jpg",
            png_probe(),
        );
        let json = report.to_json();
        assert_eq!(json["signature"]["status"], "valid");
        assert_eq!(json["signature"]["signed"], true);
        assert_eq!(json["expiry"]["expires_at"], 1893456000);
        assert_eq!(json["expiry"]["expired"], false);
        assert_eq!(json["source"]["kind"], "png");
        assert_eq!(json["source"]["width"], 2400);
        assert_eq!(json["source"]["bytes"], 412_000);
        assert_eq!(json["operations"][0]["segment"], "100x50");
        assert_eq!(json["forecast"]["width"], 100);
        assert_eq!(json["forecast"]["pixel_budget"], 100_000_000);
        assert_eq!(
            json["links"]["image"],
            "/sig/exp:1893456000/100x50/https://example.com/photo.jpg"
        );
    }

    #[test]
    fn hostile_strings_escape_in_html_and_stay_verbatim_in_json() {
        let (ops, source, expiry, _) =
            parse_operations_with_expiry("100x50/https://example.com/a&lt;b.png").unwrap();
        let mut input = ReportInput {
            signed: true,
            key_id: Some("<script>alert(1)</script>".to_string()),
            expiry,
            now: 1_700_000_000,
            source_url: source.to_string(),
            probe: png_probe(),
            source_bytes: 10,
            presets: vec![PresetEntry {
                name: "<img src=x onerror=alert(1)>".to_string(),
                operations: op_entries(&ops),
            }],
            canonical: crate::url::canonical_form(&ops, source),
            cache_group: Some("\"><script>alert(2)</script>".to_string()),
            image_path: "/sig/100x50/https://example.com/a&lt;b.png".to_string(),
            ops,
        };
        input.source_url = "https://example.com/<script>alert(3)</script>.png".to_string();
        let report = Report::new(input);
        let html = report.to_html();
        assert!(!html.contains("<script>"), "{html}");
        assert!(!html.contains("<img src=x"), "{html}");
        assert!(html.contains("&lt;script&gt;alert(3)"), "{html}");
        assert!(html.contains("a&amp;lt;b.png"), "{html}");
        let json = report.to_json();
        assert_eq!(
            json["source"]["url"],
            "https://example.com/<script>alert(3)</script>.png"
        );
        assert_eq!(json["presets"][0]["name"], "<img src=x onerror=alert(1)>");
        assert_eq!(json["cache_group"], "\"><script>alert(2)</script>");
    }

    #[test]
    fn json_negotiation_needs_json_without_html() {
        assert!(wants_json(Some("application/json")));
        assert!(wants_json(Some("application/json, */*;q=0.8")));
        assert!(!wants_json(Some(
            "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8"
        )));
        assert!(!wants_json(Some("text/html")));
        assert!(!wants_json(Some("*/*")));
        assert!(!wants_json(Some("application/json;q=0")));
        assert!(!wants_json(None));
    }

    #[test]
    fn escape_html_covers_the_five() {
        assert_eq!(
            escape_html("<a href=\"x\">&'</a>"),
            "&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;"
        );
    }
}
