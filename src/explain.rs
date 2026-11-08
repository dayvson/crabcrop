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
//! The chain it walks is the one the image request runs,
//! the server's default operations merged in, because the
//! defaults decide the output too (issue #468). The
//! canonical form already spells them; the operation list
//! stays the URL's own, so the report never credits the
//! URL with a default.

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
    /// True when `format` came from the URL's own operations.
    pub format_from_operation: bool,
    /// True when `format` came from the server's default
    /// operations instead. The merge drops a default format
    /// whenever the URL names one, so at most one of the
    /// two is set; neither means the source format.
    pub format_from_default: bool,
    pub pixels: Option<u64>,
    pub over_budget: bool,
    /// True when a resize or fit-in asks for a zero pixel side
    /// (`00x100`), which the engine refuses (issue #464). The
    /// walk stops there, so the dimensions and pixels are
    /// `None`: the image request has no output to forecast.
    pub zero_side: bool,
    /// True when a resize, fit-in, crop, pad, or `rotate(...)`
    /// works out a size past [`crate::url::MAX_DIMENSION`] on
    /// an axis (`16383x1/0x2` asks for 32766x2), which the
    /// engine refuses wherever the step sits in the chain
    /// (issue #521). A step that keeps the size it started
    /// from is not flagged, as the engine does not refuse
    /// one, but the size the chain ends at is held to the
    /// limit too (issue #528): a source at the 16384 px
    /// decode ceiling that no step resizes is flagged. The
    /// walk stops at a flagged step, so the dimensions are
    /// the size the engine refuses.
    pub over_dimension: bool,
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
    /// The source root's name when the source was written
    /// in the `root:<name>/<path>` short form (issue #248).
    /// The name is request content and safe to report; the
    /// base URL it expands against is operator
    /// configuration and is never part of the report.
    pub source_root: Option<String>,
    /// Header facts about the probed source.
    pub probe: SourceProbe,
    /// The source body's encoded size in bytes, when it is
    /// known from the fetch itself (a declared length, a
    /// file's size, or a body that ended inside the probed
    /// head). The probe never reads a whole body just to
    /// count it, so a large source can honestly report no
    /// size; `None` is never a partial count.
    pub source_bytes: Option<usize>,
    /// The request's own operations, presets expanded: the
    /// operation list the report shows, since they are what
    /// the URL spells.
    pub ops: Vec<Operation>,
    /// The operations the image request runs: the server's
    /// default operations merged ahead of `ops` exactly as
    /// the image path merges them
    /// ([`crate::url::merge_default_ops`]), or `ops` again
    /// when none are configured. The forecast walks this
    /// list.
    pub merged_ops: Vec<Operation>,
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
/// what the output will be. `ops` is the merged chain the
/// image request runs and `url_ops` the URL's own part of
/// it, which tells a URL's format from a default one. The
/// arithmetic mirrors the engine's own size resolution
/// (an SVG source's raster size, DPR scaling of pixel axes,
/// the adaptive fit-in box swap, proportional axes resolved
/// against the running size, `no_upscale` capping resizes,
/// fit-ins and crops at the source, a rotated canvas),
/// simplified where the engine's answer depends on decoded
/// pixels: a step it cannot settle (a trim, a cap against a
/// source of unknown size) makes the dimensions unknown
/// from there on, never a guess. The probe reads the stored
/// dimensions, before any EXIF orientation turns them, and
/// the client hints are not part of the URL. Sizes and the
/// pixel count saturate rather than overflow: a crop-rect
/// takes u32 corners and each upscale multiplies the
/// running size, so a hostile chain can pass `u64::MAX`,
/// and a saturated count is over the budget (issue #468).
fn forecast(ops: &[Operation], url_ops: &[Operation], probe: &SourceProbe) -> Forecast {
    let directives = RequestDirectives::analyze(ops);
    let dpr = directives.dpr.unwrap_or(1.0);
    // An SVG source is rasterized before the chain runs, at
    // the size the engine picks for it (held to 4096 px per
    // axis), and every step resolves against that raster.
    let source = probe.dimensions.map(|(w, h)| match probe.kind {
        "svg" => crate::engine::svg_raster_target(ops, (w as f32, h as f32), dpr),
        _ => (w, h),
    });
    // The engine's `no_upscale` cap: the source size, per axis
    // for a resize, proportionally for a fit-in or a crop.
    // Without a source size the capped result is unknown.
    let capped = |resolved: Option<(u64, u64)>, per_axis: bool| {
        if !directives.no_upscale {
            return resolved;
        }
        let ((w, h), (cw, ch)) = (resolved?, source?);
        Some(if per_axis {
            (w.min(cw as u64), h.min(ch as u64))
        } else {
            let (w, h) = crate::engine::cap_dimensions(saturate(w), saturate(h), (cw, ch));
            (w as u64, h as u64)
        })
    };
    let mut cur: Option<(u64, u64)> = source.map(|(w, h)| (w as u64, h as u64));
    let mut zero_side = false;
    let mut over_dimension = false;
    for op in ops {
        // The engine refuses the request here, whatever follows.
        if let Operation::Size(size) | Operation::FitIn { size, .. } = op
            && crate::engine::has_zero_side(size)
        {
            zero_side = true;
            cur = None;
            break;
        }
        let before = cur;
        cur = match op {
            Operation::Size(size) => capped(resolve_size(size, cur, dpr), true),
            Operation::Crop { size, .. } => capped(resolve_size(size, cur, dpr), false),
            // A pad canvas is never capped: `no_upscale` only
            // fits the image inside it.
            Operation::Pad { size, .. } => resolve_size(size, cur, dpr),
            Operation::FitIn { size, adaptive } => {
                capped(fit_in_forecast(size, *adaptive, cur, dpr), false)
            }
            Operation::CropRect {
                left,
                top,
                right,
                bottom,
            } => Some(((*right - *left) as u64, (*bottom - *top) as u64)),
            // A frame selection (issue #214) keeps the
            // source's dimensions: the frame is the video
            // at full size. From headers alone those are
            // unknown for a video, so `cur` is already None
            // and stays that way. A page selection (issue
            // #215) is the same shape: the rendered page is
            // the source at its render scale, which the
            // raw-bytes probe does not know.
            Operation::FocalRect { .. }
            | Operation::FocalPoint { .. }
            | Operation::DebugFocal
            | Operation::Frame(_)
            | Operation::Page(_) => cur,
            Operation::Filters(filters) => {
                let mut cur = cur;
                for filter in filters {
                    let before = cur;
                    match filter {
                        Filter::Upscale(n) => {
                            let n = *n as u64;
                            cur = cur.map(|(w, h)| (w.saturating_mul(n), h.saturating_mul(n)));
                        }
                        Filter::Rotate(degrees) => {
                            cur = cur.map(|(w, h)| {
                                let (w, h) = crate::filters::rotated_canvas(
                                    saturate(w),
                                    saturate(h),
                                    *degrees,
                                );
                                (w as u64, h as u64)
                            });
                            if past_dimension_limit(before, cur) {
                                over_dimension = true;
                                break;
                            }
                        }
                        // The trimmed size depends on the pixels.
                        Filter::Trim(_) | Filter::TrimSides(..) => cur = None,
                        _ => {}
                    }
                }
                cur
            }
        };
        // The engine checks the size each of these steps
        // resolves against the per-axis limit, and refuses the
        // request there, whatever follows (issue #521).
        let resolves = matches!(
            op,
            Operation::Size(_)
                | Operation::FitIn { .. }
                | Operation::Crop { .. }
                | Operation::Pad { .. }
        );
        if resolves && past_dimension_limit(before, cur) {
            over_dimension = true;
        }
        if over_dimension {
            break;
        }
    }
    let (width, height) = match cur {
        Some((w, h)) => (
            Some(w.min(u32::MAX as u64) as u32),
            Some(h.min(u32::MAX as u64) as u32),
        ),
        None => (None, None),
    };
    // The engine holds the image it encodes to the limit
    // as well, a size no step changed included (issue #528).
    if cur.is_some_and(|(w, h)| w.max(h) > u64::from(crate::url::MAX_DIMENSION)) {
        over_dimension = true;
    }
    let pixels = cur.map(|(w, h)| w.saturating_mul(h));
    let url_names_format = RequestDirectives::analyze(url_ops)
        .explicit_format
        .is_some();
    let (format, format_from_operation) = match directives.explicit_format {
        Some(format) => (format_text(format).to_string(), url_names_format),
        None => (default_format(probe.kind).to_string(), false),
    };
    let format_from_default = directives.explicit_format.is_some() && !url_names_format;
    Forecast {
        width,
        height,
        format,
        format_from_default,
        format_from_operation,
        zero_side,
        over_dimension,
        pixels,
        over_budget: pixels.is_some_and(|p| p > crate::engine::MAX_OUTPUT_PIXELS),
    }
}

/// Whether a step that took the running size from `before`
/// to `after` works out a side past
/// [`crate::url::MAX_DIMENSION`]: the engine's per-axis
/// rule, which passes a step that keeps the size it started
/// from (issue #521).
fn past_dimension_limit(before: Option<(u64, u64)>, after: Option<(u64, u64)>) -> bool {
    after.is_some_and(|(w, h)| before != after && w.max(h) > u64::from(crate::url::MAX_DIMENSION))
}

/// A running size as the engine's `u32` arithmetic sees it.
fn saturate(n: u64) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
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
        (Some(w), None) => cur.map(|(cw, ch)| (w, proportional(ch, w, cw))),
        (None, Some(h)) => cur.map(|(cw, ch)| (proportional(cw, h, ch), h)),
        (None, None) => cur,
    }
}

/// `axis * to / from`, the proportional axis following the
/// other axis from `from` to `to`, never below 1. The
/// product runs in u128 because a saturated running size
/// times a pixel axis overflows u64; the quotient
/// saturates (issue #468).
fn proportional(axis: u64, to: u64, from: u64) -> u64 {
    let scaled = axis as u128 * to as u128 / from.max(1) as u128;
    u64::try_from(scaled).unwrap_or(u64::MAX).max(1)
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

/// The source object: the source as written, the facts
/// the probe read off it, and, for a root source, the
/// root's name. The key appears only for a root source:
/// the object carries no more than the facts that apply,
/// and the root's base URL is never one of them.
fn source_json(input: &ReportInput) -> serde_json::Value {
    let mut source = serde_json::json!({
        "url": input.source_url,
        "kind": input.probe.kind,
        "width": input.probe.dimensions.map(|(w, _)| w),
        "height": input.probe.dimensions.map(|(_, h)| h),
        "bytes": input.source_bytes,
    });
    if let Some(root) = &input.source_root {
        source["root"] = serde_json::json!(root);
    }
    source
}

impl Report {
    pub fn new(input: ReportInput) -> Self {
        let forecast = forecast(&input.merged_ops, &input.ops, &input.probe);
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
            "source": source_json(input),
            "operations": ops_json(&op_entries(&input.ops)),
            "presets": input.presets.iter().map(|preset| serde_json::json!({
                "name": preset.name,
                "operations": ops_json(&preset.operations),
            })).collect::<Vec<_>>(),
            "forecast": {
                "width": self.forecast.width,
                "height": self.forecast.height,
                "format": self.forecast.format,
                "format_from": if self.forecast.format_from_operation {
                    "operation"
                } else if self.forecast.format_from_default {
                    "default"
                } else {
                    "source"
                },
                "pixels": self.forecast.pixels,
                "pixel_budget": crate::engine::MAX_OUTPUT_PIXELS,
                "over_budget": self.forecast.over_budget,
                "zero_side": self.forecast.zero_side,
                "over_dimension": self.forecast.over_dimension,
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
        if let Some(root) = &input.source_root {
            out.push_str(&format!("<li>Source root: {}</li>\n", escape_html(root)));
        }
        let dims = match input.probe.dimensions {
            Some((w, h)) => format!(", {w}x{h}"),
            None => String::new(),
        };
        let size = match input.source_bytes {
            Some(bytes) => format!(", {bytes} bytes"),
            None => String::new(),
        };
        out.push_str(&format!(
            "<li>{}{dims}{size}</li>\n",
            escape_html(input.probe.kind)
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
            _ if self.forecast.zero_side => {
                out.push_str("<li>Output: none; a size asks for a zero pixel side, so the image request itself would be refused</li>\n");
            }
            (Some(w), Some(h)) if self.forecast.over_dimension => {
                out.push_str(&format!(
                    "<li>Output: {w}x{h}, past the {} pixel limit per axis, so the image request itself would be refused</li>\n",
                    crate::url::MAX_DIMENSION
                ));
            }
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
            } else if self.forecast.format_from_default {
                "from the server's default operations"
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
        report_with_defaults("", remainder, probe)
    }

    /// A report on a server whose `CRABCROP_DEFAULT_OPS` is
    /// `defaults`, merged the way the handler merges them.
    fn report_with_defaults(defaults: &str, remainder: &str, probe: SourceProbe) -> Report {
        let defaults =
            crate::url::parse_default_ops(defaults, &std::collections::HashMap::new()).unwrap();
        let (ops, source, expiry, _) = parse_operations_with_expiry(remainder).unwrap();
        let merged_ops = crate::url::merge_default_ops(&defaults, ops.clone());
        Report::new(ReportInput {
            signed: true,
            key_id: None,
            expiry,
            now: 1_700_000_000,
            source_url: source.to_string(),
            source_root: None,
            probe,
            source_bytes: Some(412_000),
            presets: Vec::new(),
            canonical: crate::url::canonical_form(&merged_ops, source),
            cache_group: None,
            image_path: format!("/sig/{remainder}"),
            merged_ops,
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
    fn forecast_agrees_with_the_zero_side_refusal() {
        // A multi-digit zero side parses as an exact zero,
        // which the engine refuses (issue #464). The forecast
        // used to read "Output: 0x100" (or the 1-pixel side
        // DPR rounded it to); now it says the request has no
        // output, whatever follows the step.
        for remainder in [
            "00x100",
            "100x00",
            "00xorig",
            "00x100/filters:dpr(2)",
            "fit-in:00x100",
            "fit-in:100x00:adaptive",
            "00x100/crop:50x50",
            "crop-rect:0x0:200x200/00x10",
        ] {
            let report = report_for(
                &format!("{remainder}/https://example.com/photo.jpg"),
                png_probe(),
            );
            assert!(report.forecast.zero_side, "{remainder}");
            assert_eq!(
                (
                    report.forecast.width,
                    report.forecast.height,
                    report.forecast.pixels
                ),
                (None, None, None),
                "{remainder}"
            );
            assert_eq!(report.to_json()["forecast"]["zero_side"], true);
            let html = report.to_html();
            assert!(
                html.contains("zero pixel side, so the image request itself would be refused"),
                "{html}"
            );
            assert!(!html.contains("<li>Output: 0x"), "{html}");
        }
        // A lone 0 is proportional, not a zero side.
        let report = report_for("0x100/https://example.com/photo.jpg", png_probe());
        assert!(!report.forecast.zero_side);
        assert_eq!(report.to_json()["forecast"]["zero_side"], false);
        assert_eq!(
            (report.forecast.width, report.forecast.height),
            (Some(150), Some(100))
        );
    }

    #[test]
    fn forecast_agrees_with_the_dimension_refusal() {
        // A step that resolves a size past MAX_DIMENSION on an
        // axis is refused by the engine wherever it sits in
        // the chain (issue #521). The forecast used to report
        // 32766x2 as a servable output; now it flags the step
        // and stops there, on the size the engine refuses.
        for (remainder, refused) in [
            ("16383x1/0x2", (32766, 2)),
            ("16383x1/0x2/100x100", (32766, 2)),
            ("16383x1/fit-in:0x2", (32766, 2)),
            ("crop:16383x1:smart/0x2", (32766, 2)),
            ("4096x1/0x2/filters:dpr(2)", (16384, 4)),
            ("12000x10/filters:dpr(2)", (24000, 20)),
            ("pad:9000x100/filters:dpr(2)", (18000, 200)),
        ] {
            let report = report_for(
                &format!("{remainder}/https://example.com/photo.jpg"),
                png_probe(),
            );
            assert!(report.forecast.over_dimension, "{remainder}");
            assert_eq!(
                (report.forecast.width, report.forecast.height),
                (Some(refused.0), Some(refused.1)),
                "{remainder}"
            );
            assert_eq!(report.to_json()["forecast"]["over_dimension"], true);
            let html = report.to_html();
            assert!(
                html.contains(&format!(
                    "<li>Output: {}x{}, past the 16383 pixel limit per axis, so the image request itself would be refused</li>",
                    refused.0, refused.1
                )),
                "{html}"
            );
        }
        // Exactly the limit renders.
        for (remainder, size) in [("5461x1/0x3", (16383, 3)), ("16383x1/0x1", (16383, 1))] {
            let report = report_for(
                &format!("{remainder}/https://example.com/photo.jpg"),
                png_probe(),
            );
            assert!(!report.forecast.over_dimension, "{remainder}");
            assert_eq!(
                (report.forecast.width, report.forecast.height),
                (Some(size.0), Some(size.1)),
                "{remainder}"
            );
            assert_eq!(report.to_json()["forecast"]["over_dimension"], false);
            let html = report.to_html();
            assert!(
                html.contains(&format!("<li>Output: {}x{}</li>", size.0, size.1)),
                "{html}"
            );
        }
        // `no_upscale` holds a crop's target inside the source
        // before the engine checks it, so the request renders,
        // at the size the engine's proportional cap leaves.
        let report = report_for(
            "crop:16383x10/filters:dpr(2):no_upscale/https://example.com/photo.jpg",
            png_probe(),
        );
        assert!(!report.forecast.over_dimension);
        assert_eq!(
            (report.forecast.width, report.forecast.height),
            (Some(2400), Some(1))
        );
    }

    /// A PNG probe of the given stored size.
    fn png_of(width: u32, height: u32) -> SourceProbe {
        SourceProbe {
            kind: "png",
            dimensions: Some((width, height)),
        }
    }

    /// The forecast for `remainder` on `probe`: the dimensions
    /// and the `over_dimension` flag.
    fn forecast_of(remainder: &str, probe: SourceProbe) -> ((Option<u32>, Option<u32>), bool) {
        let report = report_for(&format!("{remainder}/https://example.com/a.png"), probe);
        (
            (report.forecast.width, report.forecast.height),
            report.forecast.over_dimension,
        )
    }

    #[test]
    fn forecast_caps_no_upscale_at_the_source_like_the_engine() {
        // The engine caps a resize per axis and a fit-in or a
        // crop proportionally at the source size, before the
        // per-axis check (issue #521). The forecast used to cap
        // a resize at the running size and a fit-in or crop
        // not at all, so it flagged chains that render.
        // 16383x1: the fit-in caps at the source, and so does
        // the resize after it.
        assert_eq!(
            forecast_of("fit-in:0x2/0x2/filters:no_upscale", png_of(16383, 1)),
            ((Some(16383), Some(1)), false)
        );
        assert_eq!(
            forecast_of("fit-in:0x16383/0x2/filters:no_upscale", png_of(16383, 1)),
            ((Some(16383), Some(1)), false)
        );
        // A step the cap holds at the source's own size keeps
        // it, so the step passes a source at the 16384 px
        // decode ceiling, but the output is held to the limit
        // too (issue #528).
        assert_eq!(
            forecast_of("fit-in:0x2/filters:no_upscale", png_of(16384, 1)),
            ((Some(16384), Some(1)), true)
        );
        // A step that grows a side to the ceiling is refused:
        // 8192x1 doubles to 16384x2, capped per axis to 16384x1.
        assert_eq!(
            forecast_of("8192x1/0x2/filters:no_upscale", png_of(16384, 1)),
            ((Some(16384), Some(1)), true)
        );
        // Without a source size the cap is unknown, and so is
        // the output: never a guess.
        let unknown = SourceProbe {
            kind: "png",
            dimensions: None,
        };
        assert_eq!(
            forecast_of("300x200/filters:no_upscale", unknown),
            ((None, None), false)
        );
        // A pad canvas is never capped, so `no_upscale` does
        // not save one past the limit.
        assert_eq!(
            forecast_of("pad:9000x100/filters:dpr(2):no_upscale", png_probe()),
            ((Some(18000), Some(200)), true)
        );
    }

    #[test]
    fn forecast_holds_the_output_to_the_limit() {
        // The engine refuses an output past 16383 px on an
        // axis, even when no step changed the size (issue
        // #528): a source at the 16384 px decode ceiling
        // needs a step that brings it within the limit.
        for (remainder, probe, size) in [
            ("filters:grayscale", png_of(16384, 1), (16384, 1)),
            ("0x0", png_of(1, 16384), (1, 16384)),
            (
                "fit-in:0x0/filters:format(webp)",
                png_of(16384, 2),
                (16384, 2),
            ),
        ] {
            let report = report_for(&format!("{remainder}/https://example.com/a.png"), probe);
            assert!(report.forecast.over_dimension, "{remainder}");
            assert_eq!(
                (report.forecast.width, report.forecast.height),
                (Some(size.0), Some(size.1)),
                "{remainder}"
            );
            assert!(
                report.to_html().contains(&format!(
                    "<li>Output: {}x{}, past the 16383 pixel limit per axis, so the image request itself would be refused</li>",
                    size.0, size.1
                )),
                "{remainder}"
            );
        }
        assert_eq!(
            forecast_of("8192x0", png_of(16384, 1)),
            ((Some(8192), Some(1)), false)
        );
        assert_eq!(
            forecast_of("filters:grayscale", png_of(16383, 1)),
            ((Some(16383), Some(1)), false)
        );
    }

    #[test]
    fn forecast_resolves_against_the_svg_raster() {
        // The engine rasterizes an SVG at the first size the
        // chain asks for, held to 4096 px per axis, and runs
        // the chain on that raster (issue #521): 0x2 on a
        // 16383x1 vector rasterizes at 4096x1 and renders
        // 8192x2, so the forecast must not flag 32766x2.
        let svg = |width, height| SourceProbe {
            kind: "svg",
            dimensions: Some((width, height)),
        };
        assert_eq!(
            forecast_of("0x2", svg(16383, 1)),
            ((Some(8192), Some(2)), false)
        );
        assert_eq!(
            forecast_of("fit-in:0x2", svg(16383, 1)),
            ((Some(8192), Some(2)), false)
        );
        assert_eq!(
            forecast_of("0x0", svg(16384, 1)),
            ((Some(4096), Some(1)), false)
        );
        assert_eq!(
            forecast_of("2x0", svg(1, 100_000)),
            ((Some(2), Some(8192)), false)
        );
        // A vector inside the raster bound renders at its size.
        assert_eq!(
            forecast_of("filters:grayscale", svg(300, 200)),
            ((Some(300), Some(200)), false)
        );
    }

    #[test]
    fn forecast_turns_rotations_and_leaves_trims_unknown() {
        // The engine checks the canvas a rotation draws on per
        // axis (issue #521): 16383x1000 turned 3.5 degrees
        // needs a canvas past the limit.
        let canvas = crate::filters::rotated_canvas(16383, 1000, 3.5);
        assert_eq!(
            forecast_of("16383x1000/filters:rotate(3.5)", png_probe()),
            ((Some(canvas.0), Some(canvas.1)), true)
        );
        // A quarter turn swaps the sides, and the resize after
        // it scales the long side on from there; a half turn
        // keeps the size, which the step lets through and the
        // output check does not (issue #528).
        assert_eq!(
            forecast_of("filters:rotate(90)/2x0", png_of(16383, 1)),
            ((Some(2), Some(32766)), true)
        );
        assert_eq!(
            forecast_of("filters:rotate(180)", png_of(16384, 2)),
            ((Some(16384), Some(2)), true)
        );
        assert_eq!(
            forecast_of("filters:rotate(180)/8192x0", png_of(16384, 2)),
            ((Some(8192), Some(1)), false)
        );
        assert_eq!(
            forecast_of("filters:rotate(90)", png_of(16384, 2)),
            ((Some(2), Some(16384)), true)
        );
        // A trim's size depends on the pixels: unknown from
        // there on, never flagged on a guess.
        assert_eq!(
            forecast_of("filters:trim/0x2", png_of(16383, 3)),
            ((None, None), false)
        );
    }

    #[test]
    fn forecast_passes_steps_that_keep_a_decode_ceiling_size() {
        // A source can decode at 16384 px per axis. A step
        // that keeps that size works nothing out, and the
        // engine passes it, so a later step can bring the
        // size within the limit; one that changes the size is
        // held to the limit (issue #521). A chain that ends
        // at the ceiling is refused (issue #528).
        for remainder in [
            "0x0",
            "fit-in:0x0",
            "filters:rotate(0)",
            "filters:grayscale",
        ] {
            assert_eq!(
                forecast_of(&format!("{remainder}/8192x0"), png_of(16384, 2)),
                ((Some(8192), Some(1)), false),
                "{remainder}"
            );
            assert_eq!(
                forecast_of(remainder, png_of(16384, 2)),
                ((Some(16384), Some(2)), true),
                "{remainder}"
            );
        }
        assert_eq!(
            forecast_of("origx1", png_of(16384, 2)),
            ((Some(16384), Some(1)), true)
        );
        assert_eq!(
            forecast_of("crop:8192x1/filters:dpr(2)", png_of(16384, 4)),
            ((Some(16384), Some(2)), true)
        );
        assert_eq!(
            forecast_of("crop:8192x1/4096x0/filters:dpr(2)", png_of(16384, 2)),
            ((Some(8192), Some(1)), false)
        );
        assert_eq!(
            forecast_of("crop:8192x1/filters:dpr(2)", png_of(16384, 2)),
            ((Some(16384), Some(2)), true)
        );
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
    fn an_unknown_source_size_reports_no_size() {
        // The probe does not read a whole body just to count
        // it: when the fetch could not state the size, the
        // report carries none, in either form, rather than a
        // partial head count dressed as the total.
        let (ops, source, expiry, _) =
            parse_operations_with_expiry("100x50/https://example.com/photo.jpg").unwrap();
        let report = Report::new(ReportInput {
            signed: true,
            key_id: None,
            expiry,
            now: 1_700_000_000,
            source_url: source.to_string(),
            source_root: None,
            probe: png_probe(),
            source_bytes: None,
            presets: Vec::new(),
            canonical: crate::url::canonical_form(&ops, source),
            cache_group: None,
            image_path: "/sig/100x50/https://example.com/photo.jpg".to_string(),
            merged_ops: ops.clone(),
            ops,
        });
        let json = report.to_json();
        assert!(json["source"]["bytes"].is_null());
        assert_eq!(json["source"]["kind"], "png");
        let html = report.to_html();
        assert!(html.contains("<li>png, 2400x1600</li>"), "{html}");
    }

    #[test]
    fn a_root_source_reports_its_name_and_never_a_base() {
        let (ops, source, expiry, _) =
            parse_operations_with_expiry("100x50/root:photos/cat.jpg").unwrap();
        let report = Report::new(ReportInput {
            signed: true,
            key_id: None,
            expiry,
            now: 1_700_000_000,
            source_url: source.to_string(),
            source_root: Some("photos".to_string()),
            probe: png_probe(),
            source_bytes: Some(10),
            presets: Vec::new(),
            canonical: crate::url::canonical_form(&ops, source),
            cache_group: None,
            image_path: "/sig/100x50/root:photos/cat.jpg".to_string(),
            merged_ops: ops.clone(),
            ops,
        });
        let json = report.to_json();
        assert_eq!(json["source"]["url"], "root:photos/cat.jpg");
        assert_eq!(json["source"]["root"], "photos");
        let html = report.to_html();
        assert!(html.contains("Source root: photos"), "{html}");
        // A plain source carries no root key at all.
        let plain = report_for("100x50/https://example.com/photo.jpg", png_probe());
        assert!(plain.to_json()["source"].get("root").is_none());
        assert!(!plain.to_html().contains("Source root"));
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
            source_root: None,
            probe: png_probe(),
            source_bytes: Some(10),
            presets: vec![PresetEntry {
                name: "<img src=x onerror=alert(1)>".to_string(),
                operations: op_entries(&ops),
            }],
            canonical: crate::url::canonical_form(&ops, source),
            cache_group: Some("\"><script>alert(2)</script>".to_string()),
            image_path: "/sig/100x50/https://example.com/a&lt;b.png".to_string(),
            merged_ops: ops.clone(),
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

    #[test]
    fn hostile_chains_saturate_and_report_over_budget() {
        // The walk multiplies u64 sizes, and crop-rect takes
        // u32 corners (issue #468). The first chain overflows
        // the pixel count; the second wraps it to exactly
        // zero in a build without overflow checks, which read
        // as under the budget. A saturated count is over the
        // budget, so both report it, and neither panics.
        let corner = "crop-rect:0x0:4294967295x4294967295";
        let sixteen = vec!["upscale(4)"; 16].join(":");
        for remainder in [
            format!("{corner}/filters:upscale(2)"),
            format!("{corner}/filters:{sixteen}"),
            format!("filters:{sixteen}:{sixteen}:{sixteen}"),
        ] {
            let report = report_for(
                &format!("{remainder}/https://example.com/photo.jpg"),
                png_probe(),
            );
            assert_eq!(report.forecast.pixels, Some(u64::MAX), "{remainder}");
            assert!(report.forecast.over_budget, "{remainder}");
            assert_eq!(report.forecast.width, Some(u32::MAX), "{remainder}");
            assert_eq!(report.forecast.height, Some(u32::MAX), "{remainder}");
            assert_eq!(report.to_json()["forecast"]["over_budget"], true);
            assert!(report.to_html().contains("over the budget"));
        }
        // A proportional resize after a saturated size
        // scales in wide arithmetic rather than overflowing,
        // and lands on the size it names.
        for remainder in [
            format!("filters:{sixteen}:{sixteen}:{sixteen}/100x0"),
            format!("filters:{sixteen}:{sixteen}:{sixteen}/0x100"),
            format!("{corner}/filters:{sixteen}/100x0"),
        ] {
            let report = report_for(
                &format!("{remainder}/https://example.com/photo.jpg"),
                png_probe(),
            );
            assert_eq!(
                (report.forecast.width, report.forecast.height),
                (Some(100), Some(100)),
                "{remainder}"
            );
            assert!(!report.forecast.over_budget, "{remainder}");
        }
    }

    #[test]
    fn forecast_walks_the_default_operations() {
        // A default format decides the output when the URL
        // names none, and the report says so instead of
        // crediting the URL or the source (issue #468). A
        // default DPR scales the forecast like the engine.
        let report = report_with_defaults(
            "filters:format(webp):dpr(2)",
            "100x0/https://example.com/photo.jpg",
            png_probe(),
        );
        assert_eq!(report.forecast.format, "webp");
        assert!(report.forecast.format_from_default);
        assert!(!report.forecast.format_from_operation);
        assert_eq!(
            (report.forecast.width, report.forecast.height),
            (Some(200), Some(133))
        );
        let json = report.to_json();
        assert_eq!(json["forecast"]["format_from"], "default");
        // The operation list is the URL's own; the defaults
        // show in the canonical form only.
        assert_eq!(json["operations"].as_array().unwrap().len(), 1);
        assert_eq!(json["operations"][0]["segment"], "100x0");
        assert_eq!(
            json["canonical"],
            "filters:format(webp):dpr(2)/100x0/https://example.com/photo.jpg"
        );
        let html = report.to_html();
        assert!(
            html.contains("<li>Format: webp (from the server's default operations)</li>"),
            "{html}"
        );

        // The URL's own format wins, and the report credits it.
        let report = report_with_defaults(
            "filters:format(webp)",
            "100x0/filters:format(avif)/https://example.com/photo.jpg",
            png_probe(),
        );
        assert_eq!(report.forecast.format, "avif");
        assert!(report.forecast.format_from_operation);
        assert!(!report.forecast.format_from_default);
        assert_eq!(report.to_json()["forecast"]["format_from"], "operation");

        // Defaults that name no format leave it to the source.
        let report = report_with_defaults(
            "filters:quality(80)",
            "100x0/https://example.com/photo.jpg",
            png_probe(),
        );
        assert_eq!(report.forecast.format, "png");
        assert!(!report.forecast.format_from_operation);
        assert!(!report.forecast.format_from_default);
        assert_eq!(report.to_json()["forecast"]["format_from"], "source");

        // The debug view is a remainder to sign, and the
        // image path merges the defaults into it again, so
        // it keeps to the URL's own operations while the
        // canonical form carries the defaults.
        let report = report_with_defaults(
            "filters:format(webp):dpr(2)",
            "crop:100x100/https://example.com/photo.jpg",
            png_probe(),
        );
        assert_eq!(
            report.debug_focal_remainder.as_deref(),
            Some("debug:focal/crop:100x100/https://example.com/photo.jpg")
        );
        assert_eq!(
            report.to_json()["canonical"],
            "filters:format(webp):dpr(2)/crop:100x100/https://example.com/photo.jpg"
        );
    }
}
