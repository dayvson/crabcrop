//! SVG sources: content sniffing, sanitization, and
//! rasterization.
//!
//! An SVG is a program as much as a picture: it can carry
//! scripts, event handlers, and references to external
//! resources. The pipeline therefore never hands source SVG to
//! the renderer as-is. [`prepare`] runs the sanitizer first:
//! script and other active elements are removed, `on*`
//! handler attributes are dropped, and every reference
//! (`href`, CSS `url(...)`) that does not point at a fragment
//! inside the document or an embedded `data:` payload is
//! stripped: an `href` is kept only when it names an element
//! of the document (`#id`) or is a well-formed `data:` URL.
//! Rasterization is resvg/usvg (pure Rust), whose render model
//! has no scripts or network access to begin with. Its image
//! loader, though, reads any `href` that is not a data URL as
//! a file path, so [`build`] replaces that loader with one
//! that resolves nothing: whatever the sanitizer keeps, the
//! renderer draws only payloads the document embeds, and a
//! sanitized SVG can only draw what it carries.
//!
//! The server never runs usvg in its own process (issue
//! #447). [`screen`] sanitizes and runs the pre-parse bound
//! here, the cheap first line, and [`Screened`] hands the
//! sanitized document to a capped child process that parses
//! and rasterizes it (the `worker` module), so a document
//! that gets past the bound can abort or exhaust only that
//! child.

use std::ops::ControlFlow;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Bounds on SVG document complexity (audit E-7). The raster
/// target was already capped, but parse and `<use>` expansion
/// work was not: these caps bound the source document, the
/// expansion usvg performs while building its tree, and the
/// declared dimensions, before any rendering happens.
const MAX_SVG_NODES: usize = 10_000;
/// The deepest element nesting the bounds walk (and, after
/// it, the writer) will recurse into. The node cap alone
/// permits a 10,000-deep nest; real drawings nest a few
/// dozen levels at most.
const MAX_SVG_DEPTH: usize = 64;
const MAX_USE_ELEMENTS: usize = 256;
const MAX_USE_DEPTH: usize = 16;
/// The most nodes the renderer may draw, as
/// [`count_expanded`] counts them: `<use>` clones, plus
/// pattern, mask, clip-path, and `feImage` contents once
/// for each reference resvg draws them through (issue
/// #398). The expansion bound holds usvg to it before the
/// tree is built, too (issue #443).
const MAX_EXPANDED_NODES: usize = 100_000;
const MAX_SVG_DIMENSION: f64 = 100_000.0;
/// The deepest the conversion may recurse: element nesting
/// plus every reference it follows (issue #443). usvg
/// recurses on the thread's stack with no limit of its own;
/// a debug build overflows a 2 MiB stack past about 115
/// chained `feImage` references, which this caps at 63. SVG
/// images nested in a document share it: usvg converts one
/// on top of the conversion that reached it.
const MAX_CONVERSION_DEPTH: usize = 128;
/// The levels a nested SVG image's parse adds between the
/// `<image>` that holds it and its own root, as the depth
/// budget charges them (like a reference hop).
const NESTED_IMAGE_LEVELS: usize = 2;

mod expansion;
mod worker;

// Used only by the AVIF worker decode default timeout.
#[cfg(feature = "avif-decode")]
pub(crate) use worker::WORKER_TIMEOUT;
pub use worker::{
    RenderError, WORKER_ARG, WorkerCommand, is_worker_invocation, isolate_in_child, worker_main,
};

use image::DynamicImage;

/// Decode an AVIF source in the capped worker when SVG work is
/// isolated (issue #522): the AV1 decode runs in the worker
/// under its caps, killed at `deadline`, as an SVG render is
/// (issue #447). `None` means no worker is
/// configured (the library, the fuzz harness, the CLI), so the
/// caller decodes in process. The limits are the engine's
/// decode bounds, passed through to the worker.
pub fn decode_avif_in_worker(
    bytes: &[u8],
    max_dimension: u32,
    max_alloc: u64,
    deadline: Instant,
) -> Option<Result<DynamicImage, RenderError>> {
    worker::configured()
        .map(|command| worker::run_avif(command, bytes, max_dimension, max_alloc, deadline))
}

/// [`decode_avif_in_worker`] against a given worker command,
/// for tests that drive a worker directly.
pub fn decode_avif_in(
    command: &WorkerCommand,
    bytes: &[u8],
    max_dimension: u32,
    max_alloc: u64,
    deadline: Instant,
) -> Result<DynamicImage, RenderError> {
    worker::run_avif(command, bytes, max_dimension, max_alloc, deadline)
}
use resvg::usvg;

/// Sniff whether `bytes` are an SVG document. The declared
/// content type can lie, so this looks at the content: an
/// optional BOM, XML prolog, comments, and doctype may precede
/// the `<svg` root element.
pub fn is_svg(bytes: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    let mut rest = text.trim_start_matches('\u{feff}').trim_start();
    loop {
        if let Some(after) = rest.strip_prefix("<?") {
            match after.find("?>") {
                Some(i) => rest = after[i + 2..].trim_start(),
                None => return false,
            }
        } else if let Some(after) = rest.strip_prefix("<!--") {
            match after.find("-->") {
                Some(i) => rest = after[i + 3..].trim_start(),
                None => return false,
            }
        } else if let Some(after) = rest.strip_prefix("<!") {
            match after.find('>') {
                Some(i) => rest = after[i + 1..].trim_start(),
                None => return false,
            }
        } else {
            break;
        }
    }
    rest.starts_with("<svg")
        && matches!(
            rest.as_bytes().get(4),
            Some(b' ' | b'>' | b'\t' | b'\n' | b'\r' | b'/')
        )
}

/// Elements that can act, embed, or animate references: never
/// rendered, removed with their whole subtree.
const BANNED_ELEMENTS: &[&str] = &[
    "script",
    "foreignobject",
    "animate",
    "set",
    "animatetransform",
    "animatemotion",
    "iframe",
    "audio",
    "video",
];

fn is_banned(name: &str) -> bool {
    BANNED_ELEMENTS.iter().any(|b| b.eq_ignore_ascii_case(name))
}

/// The first position of `needle` in `haystack`, comparing
/// ASCII case-insensitively, so the case checks below do
/// not allocate a lowercased copy of untrusted text.
fn find_ascii_ci(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .as_bytes()
        .windows(needle.len())
        .position(|w| w.eq_ignore_ascii_case(needle.as_bytes()))
}

/// Whether `value` begins with `prefix`, comparing ASCII
/// case-insensitively.
fn starts_with_ci(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

/// Whether a CSS fragment references anything besides an
/// in-document fragment or an embedded `data:` payload.
fn has_external_url_ref(value: &str) -> bool {
    let mut rest = value;
    while let Some(i) = find_ascii_ci(rest, "url(") {
        let after = &rest[i + 4..];
        let target = after.trim_start_matches([' ', '\t', '"', '\'']);
        if !(target.starts_with('#') || starts_with_ci(target, "data:")) {
            return true;
        }
        rest = after;
    }
    false
}

/// Whether an `href` value points inside the document: a
/// fragment reference to an element ID (`#id`, which holds no
/// path separator), or a `data:` URL with the comma every data
/// URL has between its media type and its payload. usvg reads
/// anything else as a file path, so a value that only starts
/// like one of these (`#/x`, `data:x.png`) is dropped here, and
/// the renderer's own loader refuses it as well (see
/// [`image_href_resolver`]).
fn is_document_href(value: &str) -> bool {
    if let Some(id) = value.strip_prefix('#') {
        return !id.is_empty() && !id.contains(['/', '\\']);
    }
    starts_with_ci(value, "data:") && value.contains(',')
}

fn keep_attribute(attr: roxmltree::Attribute) -> bool {
    let name = attr.name();
    // Event handlers (onclick, onload, ...) are behavior, not
    // drawing.
    if starts_with_ci(name, "on") {
        return false;
    }
    if name.eq_ignore_ascii_case("href") {
        return is_document_href(attr.value().trim());
    }
    if has_external_url_ref(attr.value()) {
        return false;
    }
    true
}

fn escape_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn escape_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The text of a `<style>` element is kept only when it cannot
/// pull anything in: no imports, no external url() targets.
fn style_is_safe(node: roxmltree::Node) -> bool {
    let text: String = node
        .children()
        .filter(|c| c.is_text())
        .filter_map(|c| c.text())
        .collect();
    find_ascii_ci(&text, "@import").is_none() && !has_external_url_ref(&text)
}

fn write_node(node: roxmltree::Node, out: &mut String, is_root: bool) {
    let tag = node.tag_name().name();
    out.push('<');
    out.push_str(tag);
    if is_root {
        out.push_str(" xmlns=\"http://www.w3.org/2000/svg\"");
        out.push_str(" xmlns:xlink=\"http://www.w3.org/1999/xlink\"");
    }
    for attr in node.attributes() {
        if attr.name() == "xmlns" || attr.name().starts_with("xmlns:") {
            continue; // redeclared above on the root
        }
        if !keep_attribute(attr) {
            continue;
        }
        out.push(' ');
        if attr.namespace() == Some("http://www.w3.org/1999/xlink") {
            out.push_str("xlink:");
        }
        out.push_str(attr.name());
        out.push_str("=\"");
        out.push_str(&escape_attr(attr.value()));
        out.push('"');
    }
    let children: Vec<roxmltree::Node> = node.children().collect();
    if children.is_empty() {
        out.push_str("/>");
        return;
    }
    out.push('>');
    for child in children {
        if child.is_element() {
            let name = child.tag_name().name();
            if is_banned(name) {
                continue;
            }
            if name.eq_ignore_ascii_case("style") && !style_is_safe(child) {
                continue;
            }
            write_node(child, out, false);
        } else if child.is_text()
            && let Some(text) = child.text()
        {
            out.push_str(&escape_text(text));
        }
        // Comments, processing instructions, and doctypes are
        // not part of the drawing and are dropped.
    }
    out.push_str("</");
    out.push_str(tag);
    out.push('>');
}

/// Rewrite `text` into an inert SVG: active elements, event
/// handlers, and external references removed (see the module
/// docs). Errors when the document is malformed or its root
/// is not `<svg>`.
pub fn sanitize(text: &str) -> Result<String, String> {
    // A DOCTYPE with an internal subset can declare entities;
    // expansion behavior differs across parsers, so the subset
    // form is rejected outright (audit E-7). A bare DOCTYPE is
    // fine: the writer below does not re-emit it.
    if let Some(start) = text.find("<!DOCTYPE") {
        let decl = &text[start..text[start..].find('>').map_or(text.len(), |i| start + i)];
        if decl.contains('[') {
            return Err("SVG with a DOCTYPE internal subset is not supported".to_string());
        }
    }
    let doc = roxmltree::Document::parse(text).map_err(|e| format!("invalid SVG: {e}"))?;
    let root = doc.root_element();
    if !root.tag_name().name().eq_ignore_ascii_case("svg") {
        return Err("not an SVG document".to_string());
    }
    check_document_bounds(root)?;
    check_dimensions(root)?;
    let mut out = String::with_capacity(text.len());
    write_node(root, &mut out, true);
    Ok(out)
}

/// Enforce the source-side bounds: total node count and
/// nesting depth, and `<use>` count and nesting depth, all
/// before tree construction.
fn check_document_bounds<'a, 'input>(root: roxmltree::Node<'a, 'input>) -> Result<(), String> {
    fn walk<'a, 'input>(
        node: roxmltree::Node<'a, 'input>,
        counts: &mut (usize, usize),
        ids: &mut std::collections::HashMap<String, roxmltree::Node<'a, 'input>>,
        depth: usize,
    ) -> Result<(), String> {
        counts.0 += 1;
        if counts.0 > MAX_SVG_NODES {
            return Err(format!("SVG has more than {MAX_SVG_NODES} nodes"));
        }
        if depth > MAX_SVG_DEPTH {
            return Err(format!("SVG nests deeper than {MAX_SVG_DEPTH} elements"));
        }
        if node.tag_name().name().eq_ignore_ascii_case("use") {
            counts.1 += 1;
            if counts.1 > MAX_USE_ELEMENTS {
                return Err(format!("SVG has more than {MAX_USE_ELEMENTS} use elements"));
            }
        }
        if let Some(id) = node.attribute("id") {
            ids.insert(id.to_string(), node);
        }
        for child in node.children() {
            walk(child, counts, ids, depth + 1)?;
        }
        Ok(())
    }
    // The expansion depth of the content an id names: the
    // longest chain of uses reachable from it. Cycles are
    // errors (usvg rejects them too, later).
    fn depth_of<'a, 'input>(
        id: &str,
        ids: &std::collections::HashMap<String, roxmltree::Node<'a, 'input>>,
        visiting: &mut Vec<String>,
        memo: &mut std::collections::HashMap<String, usize>,
    ) -> Result<usize, String> {
        if let Some(depth) = memo.get(id) {
            return Ok(*depth);
        }
        if visiting.iter().any(|v| v == id) {
            return Err("SVG use elements reference each other cyclically".to_string());
        }
        // The chain being walked is already longer than any
        // depth the caller would accept: fail here instead
        // of recursing to the end of it first.
        if visiting.len() > MAX_USE_DEPTH {
            return Err(format!("SVG use elements nest deeper than {MAX_USE_DEPTH}"));
        }
        let Some(node) = ids.get(id) else {
            return Ok(0);
        };
        visiting.push(id.to_string());
        let mut deepest = 0;
        for desc in node.descendants() {
            if desc.tag_name().name().eq_ignore_ascii_case("use")
                && let Some(target) = use_target(&desc)
            {
                deepest = deepest.max(1 + depth_of(target, ids, visiting, memo)?);
            }
        }
        visiting.pop();
        memo.insert(id.to_string(), deepest);
        Ok(deepest)
    }
    fn use_target<'x>(node: &roxmltree::Node<'x, '_>) -> Option<&'x str> {
        node.attribute("href")
            .or_else(|| node.attribute(("http://www.w3.org/1999/xlink", "href")))
            .and_then(|h| h.strip_prefix('#'))
    }
    let mut ids = std::collections::HashMap::new();
    walk(root, &mut (0, 0), &mut ids, 0)?;
    let mut memo = std::collections::HashMap::new();
    for desc in root.descendants() {
        if desc.tag_name().name().eq_ignore_ascii_case("use")
            && let Some(target) = use_target(&desc)
        {
            let depth = 1 + depth_of(target, &ids, &mut Vec::new(), &mut memo)?;
            if depth > MAX_USE_DEPTH {
                return Err(format!("SVG use elements nest deeper than {MAX_USE_DEPTH}"));
            }
        }
    }
    Ok(())
}

/// Enforce the declared root dimensions (width/height/viewBox).
fn check_dimensions(root: roxmltree::Node) -> Result<(), String> {
    for attr in ["width", "height"] {
        if let Some(value) = root.attribute(attr) {
            let number: String = value
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if let Ok(n) = number.parse::<f64>()
                && n > MAX_SVG_DIMENSION
            {
                return Err(format!(
                    "SVG {attr} exceeds the {MAX_SVG_DIMENSION} unit limit"
                ));
            }
        }
    }
    if let Some(view_box) = root.attribute("viewBox") {
        for part in view_box.split(|c: char| c.is_whitespace() || c == ',') {
            if let Ok(n) = part.parse::<f64>()
                && n.abs() > MAX_SVG_DIMENSION
            {
                return Err(format!(
                    "SVG viewBox exceeds the {MAX_SVG_DIMENSION} unit limit"
                ));
            }
        }
    }
    Ok(())
}

/// Visit every node the renderer draws from a constructed
/// usvg tree, depth first, until `f` breaks.
///
/// `Group::children()` alone is not that set (issue
/// #398): usvg keeps pattern contents (path fills and
/// strokes), mask and clip-path contents, filter
/// `feImage` sources, the flattened form of text, and the
/// tree of a nested SVG image outside it, and resvg
/// draws all of them. A mask or clip path can itself be
/// masked or clipped, and resvg follows that chain to
/// its end, so this does too (usvg's `Node::subroots`
/// stops after the second link). A subtree shared by
/// several references, one pattern filling many paths,
/// is visited once per reference, because resvg draws it
/// once per reference.
fn visit_rendered<F>(group: &usvg::Group, f: &mut F) -> ControlFlow<()>
where
    F: FnMut(&usvg::Node) -> ControlFlow<()>,
{
    let mut clip = group.clip_path();
    while let Some(clip_path) = clip {
        visit_rendered(clip_path.root(), f)?;
        clip = clip_path.clip_path();
    }
    let mut next_mask = group.mask();
    while let Some(mask) = next_mask {
        visit_rendered(mask.root(), f)?;
        next_mask = mask.mask();
    }
    for filter in group.filters() {
        for primitive in filter.primitives() {
            if let usvg::filter::Kind::Image(image) = primitive.kind() {
                visit_rendered(image.root(), f)?;
            }
        }
    }
    for node in group.children() {
        f(node)?;
        match node {
            usvg::Node::Group(child) => visit_rendered(child, f)?,
            usvg::Node::Path(path) => {
                let paints = [
                    path.fill().map(usvg::Fill::paint),
                    path.stroke().map(usvg::Stroke::paint),
                ];
                for paint in paints.into_iter().flatten() {
                    if let usvg::Paint::Pattern(pattern) = paint {
                        visit_rendered(pattern.root(), f)?;
                    }
                }
            }
            usvg::Node::Image(image) => {
                if let usvg::ImageKind::SVG(tree) = image.kind() {
                    visit_rendered(tree.root(), f)?;
                }
            }
            usvg::Node::Text(text) => visit_rendered(text.flattened(), f)?,
        }
    }
    ControlFlow::Continue(())
}

/// Count the nodes the renderer draws from a constructed
/// tree, `<use>` clones and every subtree
/// [`visit_rendered`] reaches included: the expansion
/// bound, checked before rendering. The count stops at
/// `cap + 1`, so a tree that multiplies a shared subtree
/// through many references costs at most the cap to
/// refuse, however far it would expand.
fn count_expanded(root: &usvg::Group, cap: usize) -> usize {
    let mut count = 0usize;
    let _ = visit_rendered(root, &mut |_| {
        count += 1;
        if count > cap {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    count
}

/// The decoded byte size of the raster payloads embedded
/// in a constructed usvg tree (issues #360, #398). An
/// `<image>` element counts as one node in the expansion
/// accounting above, but the renderer decodes its payload
/// at the payload's own dimensions, so those pixels are
/// charged here at four bytes per pixel (the RGBA
/// decode), the same accounting the stills decode budget
/// uses. Every raster the renderer would decode counts,
/// wherever [`visit_rendered`] finds it: in a pattern, a
/// mask, a filter's `feImage`, or a nested SVG payload,
/// once per time it is drawn.
fn embedded_raster_bytes(root: &usvg::Group) -> u64 {
    let mut total = 0u64;
    let _ = visit_rendered(root, &mut |node| {
        if let usvg::Node::Image(image) = node
            && !matches!(image.kind(), usvg::ImageKind::SVG(_))
        {
            let size = image.size();
            let bytes = (size.width() as u64) * (size.height() as u64) * 4;
            total = total.saturating_add(bytes);
        }
        ControlFlow::Continue(())
    });
    total
}

/// Refuse an SVG whose embedded rasters decode past the
/// shared decode allocation budget (issue #360).
fn check_raster_budget(decoded_bytes: u64) -> Result<(), String> {
    if decoded_bytes > crate::engine::MAX_DECODE_ALLOC {
        return Err(decode_budget_error());
    }
    Ok(())
}

fn decode_budget_error() -> String {
    format!(
        "SVG embedded images decode past the {} byte decode budget",
        crate::engine::MAX_DECODE_ALLOC
    )
}

fn expansion_error() -> String {
    format!("SVG expands past the {MAX_EXPANDED_NODES} node limit")
}

/// Bound what usvg will build for a parsed document before it
/// builds it (issue #443), refusing past the node limit, the
/// decode budget (`data:` payloads, once per image instance),
/// `depth` levels of conversion, or on a reference cycle. See
/// [`expansion`] for the model.
fn check_expansion(doc: &roxmltree::Document, depth: usize) -> Result<expansion::Cost, String> {
    let limits = expansion::Limits {
        nodes: MAX_EXPANDED_NODES as u64,
        bytes: crate::engine::MAX_DECODE_ALLOC,
        depth,
    };
    expansion::bound(doc, limits).map_err(|refusal| match refusal {
        expansion::Refusal::Nodes => expansion_error(),
        expansion::Refusal::Bytes => decode_budget_error(),
        expansion::Refusal::Cycle => "SVG references form a cycle".to_string(),
        expansion::Refusal::Depth => format!(
            "SVG nests deeper than {MAX_CONVERSION_DEPTH} levels once references are followed"
        ),
        expansion::Refusal::Work => {
            "SVG style rules match too many elements to bound before rendering".to_string()
        }
    })
}

/// What the SVG images nested in a document (`data:` payloads
/// usvg parses as SVG) may still add once the document's own
/// bound is charged, and the first refusal one of them hit.
/// usvg parses each nested image only while it builds the
/// outer tree, so the resolver below bounds each one before
/// usvg parses it and charges it here, once per instance.
struct NestedBudget {
    nodes: u64,
    bytes: u64,
    /// The conversion depth left for the next nested image: the
    /// cap, less what the documents it sits in may already have
    /// on the stack when usvg reaches it.
    depth: usize,
    refusal: Option<String>,
}

impl NestedBudget {
    /// Charge a nested document's bound; false (and the
    /// refusal recorded) once the budget is spent.
    fn charge(&mut self, cost: Result<expansion::Cost, String>) -> bool {
        if self.refusal.is_some() {
            return false;
        }
        let refusal = match cost {
            Ok(cost) if cost.nodes > self.nodes => expansion_error(),
            Ok(cost) if cost.bytes > self.bytes => decode_budget_error(),
            Ok(cost) => {
                self.nodes -= cost.nodes;
                self.bytes -= cost.bytes;
                return true;
            }
            Err(refusal) => refusal,
        };
        self.refusal = Some(refusal);
        false
    }
}

/// The depth left for SVG images nested in a document that
/// reaches `height` levels, with `depth` available to it.
fn nested_depth(depth: usize, height: usize) -> usize {
    depth.saturating_sub(height + NESTED_IMAGE_LEVELS)
}

/// The bound of one nested SVG image, parsed the way usvg
/// parses it (DTDs allowed), held to the same source bounds as
/// the document itself; `Ok(None)` when it does not parse, in
/// which case usvg draws nothing for it either.
fn nested_cost(text: &str, depth: usize) -> Result<Option<expansion::Cost>, String> {
    let options = roxmltree::ParsingOptions {
        allow_dtd: true,
        // One past the node cap: the walk below reports it.
        nodes_limit: (MAX_SVG_NODES + 2) as u32,
        ..Default::default()
    };
    let doc = match roxmltree::Document::parse_with_options(text, options) {
        Ok(doc) => doc,
        Err(roxmltree::Error::NodesLimitReached) => {
            return Err(format!("SVG has more than {MAX_SVG_NODES} nodes"));
        }
        Err(_) => return Ok(None),
    };
    check_document_bounds(doc.root_element())?;
    check_expansion(&doc, depth).map(Some)
}

/// The `data:` resolver usvg calls for every image payload:
/// rasters go to usvg's default handling, and a payload usvg
/// would parse as a nested SVG is bounded and charged against
/// `budget` first. A gzip-compressed (svgz) payload is not
/// parsed at all: usvg inflates it with no size limit.
fn nested_svg_resolver(budget: Arc<Mutex<NestedBudget>>) -> usvg::ImageHrefDataResolverFn<'static> {
    let default = usvg::ImageHrefResolver::default_data_resolver();
    Box::new(move |mime, data, options| {
        let svg = match mime {
            "image/svg+xml" => true,
            // usvg sniffs a text/plain payload, and parses it
            // as SVG unless it carries a raster signature.
            "text/plain" => !has_raster_signature(&data),
            _ => false,
        };
        if !svg {
            return default(mime, data, options);
        }
        if data.starts_with(&[0x1f, 0x8b]) {
            return None;
        }
        let text = std::str::from_utf8(&data).ok()?;
        let lock = || budget.lock().unwrap_or_else(|e| e.into_inner());
        let depth = lock().depth;
        let cost = match nested_cost(text, depth) {
            Ok(Some(cost)) => Ok(cost),
            Ok(None) => return None,
            Err(refusal) => Err(refusal),
        };
        let height = cost.as_ref().map_or(0, |c| c.height);
        {
            let mut budget = lock();
            if !budget.charge(cost) {
                return None;
            }
            // Images nested in this one convert on top of it.
            budget.depth = nested_depth(depth, height);
        }
        let tree = usvg::Tree::from_data_nested(&data, options);
        lock().depth = depth;
        tree.ok().map(usvg::ImageKind::SVG)
    })
}

/// The raster signatures usvg's sniffing recognizes in a
/// text/plain payload: JPEG, PNG, GIF, and WebP.
fn has_raster_signature(data: &[u8]) -> bool {
    data.starts_with(&[0xff, 0xd8, 0xff])
        || data.starts_with(b"\x89PNG")
        || data.starts_with(b"GIF8")
        || (data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP")
}

/// A sanitized, parsed SVG ready to rasterize.
pub struct Prepared {
    tree: usvg::Tree,
}

/// The sanitized form of an SVG document, as bytes: the
/// exact output of [`sanitize`] on the decoded text. This
/// is what the upload endpoint stores for an SVG upload
/// (issue #192, audit R2-22): an active document a caller
/// uploaded is never what lands in the destination, even
/// though serving would sanitize it again on the way out.
/// Callers gate on [`is_svg`] first, and validate with
/// [`prepare`] as well: this function alone does not
/// prove the document parses.
pub fn sanitized_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| "SVG source is not valid UTF-8".to_string())?;
    Ok(sanitize(text)?.into_bytes())
}

/// Sanitize and parse `bytes`, in this process. Callers gate
/// on [`is_svg`] first; other XML is rejected here as
/// not-an-SVG. The server goes through [`screen`] instead,
/// which parses in the worker; this is the in-process form
/// the worker itself and the fuzz harness use.
pub fn prepare(bytes: &[u8]) -> Result<Prepared, String> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| "SVG source is not valid UTF-8".to_string())?;
    build(&sanitize(text)?)
}

/// A document that passed the sanitizer and the pre-parse
/// bound in this process (issue #447): the checks that need
/// no usvg, so a document they refuse costs no worker. The
/// parse and the render happen in a worker when SVG work is
/// isolated (the binary always isolates it), in this process
/// otherwise.
pub struct Screened {
    clean: String,
}

/// Sanitize `bytes` and bound the sanitized document's
/// expansion (issue #443), without parsing it into a usvg
/// tree. Callers gate on [`is_svg`] first.
pub fn screen(bytes: &[u8]) -> Result<Screened, String> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| "SVG source is not valid UTF-8".to_string())?;
    let clean = sanitize(text)?;
    let parsed =
        roxmltree::Document::parse(&clean).map_err(|e| format!("cannot parse SVG: {e}"))?;
    check_expansion(&parsed, MAX_CONVERSION_DEPTH)?;
    drop(parsed);
    Ok(Screened { clean })
}

impl Screened {
    /// The sanitized document, exactly as [`sanitized_bytes`]
    /// spells it.
    pub fn into_bytes(self) -> Vec<u8> {
        self.clean.into_bytes()
    }

    /// Parse the document, held to every bound [`prepare`]
    /// enforces, and report its intrinsic size. An isolated
    /// worker is killed at `deadline` (a default timeout when
    /// `None`).
    pub fn intrinsic_size(&self, deadline: Option<Instant>) -> Result<(f32, f32), RenderError> {
        self.intrinsic_size_in(worker::configured(), deadline)
    }

    /// [`Screened::intrinsic_size`] in the given worker, or
    /// in this process for `None`.
    pub fn intrinsic_size_in(
        &self,
        worker: Option<&WorkerCommand>,
        deadline: Option<Instant>,
    ) -> Result<(f32, f32), RenderError> {
        self.run(worker, deadline, |_| None).map(|(size, _)| size)
    }

    /// Parse the document and rasterize it at the size
    /// `target` picks from its intrinsic size (see
    /// [`Prepared::render`]).
    pub fn rasterize(
        &self,
        deadline: Option<Instant>,
        target: impl FnOnce((f32, f32)) -> (u32, u32),
    ) -> Result<DynamicImage, RenderError> {
        self.rasterize_in(worker::configured(), deadline, target)
    }

    /// [`Screened::rasterize`] in the given worker, or in
    /// this process for `None`.
    pub fn rasterize_in(
        &self,
        worker: Option<&WorkerCommand>,
        deadline: Option<Instant>,
        target: impl FnOnce((f32, f32)) -> (u32, u32),
    ) -> Result<DynamicImage, RenderError> {
        let (_, raster) = self.run(worker, deadline, |size| Some(target(size)))?;
        raster
            .map(DynamicImage::ImageRgba8)
            .ok_or_else(|| RenderError::Refused("the SVG renderer returned no raster".to_string()))
    }

    fn run(
        &self,
        worker: Option<&WorkerCommand>,
        deadline: Option<Instant>,
        target: impl FnOnce((f32, f32)) -> Option<(u32, u32)>,
    ) -> Result<worker::Output, RenderError> {
        if let Some(command) = worker {
            let deadline = deadline.unwrap_or_else(|| Instant::now() + worker::WORKER_TIMEOUT);
            return worker::run(command, &self.clean, deadline, target);
        }
        let prepared = build(&self.clean).map_err(RenderError::Refused)?;
        let size = prepared.intrinsic();
        let raster = match target(size) {
            Some((w, h)) => Some(
                prepared
                    .render(w, h)
                    .map_err(RenderError::Refused)?
                    .into_rgba8(),
            ),
            None => None,
        };
        Ok((size, raster))
    }
}

/// Parse a sanitized document into a usvg tree, held to the
/// expansion bound before usvg builds anything and to the
/// node and decode budgets after. This is the work that runs
/// in the worker.
fn build(clean: &str) -> Result<Prepared, String> {
    // The bound runs on the sanitized text because that is what
    // usvg builds from (this very parse): the sanitizer removes
    // elements, which moves what `:first-child` and sibling
    // selectors match, and drops namespace prefixes.
    let parsed = roxmltree::Document::parse(clean).map_err(|e| format!("cannot parse SVG: {e}"))?;
    let cost = check_expansion(&parsed, MAX_CONVERSION_DEPTH)?;
    let budget = Arc::new(Mutex::new(NestedBudget {
        nodes: MAX_EXPANDED_NODES as u64 - cost.nodes,
        bytes: crate::engine::MAX_DECODE_ALLOC - cost.bytes,
        depth: nested_depth(MAX_CONVERSION_DEPTH, cost.height),
        refusal: None,
    }));
    let opt = usvg::Options {
        font_resolver: font_resolver(),
        image_href_resolver: image_href_resolver(budget.clone()),
        ..Default::default()
    };
    let tree =
        usvg::Tree::from_xmltree(&parsed, &opt).map_err(|e| format!("cannot parse SVG: {e}"))?;
    drop(parsed);
    let nested_refusal = budget
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .refusal
        .take();
    if let Some(refusal) = nested_refusal {
        return Err(refusal);
    }
    // A second line behind the bound above: the count of the
    // tree as built.
    if count_expanded(tree.root(), MAX_EXPANDED_NODES) > MAX_EXPANDED_NODES {
        return Err(expansion_error());
    }
    // Runs only once the count above has passed, so this
    // walk visits at most MAX_EXPANDED_NODES nodes.
    check_raster_budget(embedded_raster_bytes(tree.root()))?;
    Ok(Prepared { tree })
}

/// How the renderer loads what an `<image>` or `feImage` href
/// names. An embedded `data:` payload goes through the nested
/// budget ([`nested_svg_resolver`]). Every other href resolves
/// to nothing: usvg's default treats it as a file path and
/// reads it from disk, relative to the working directory the
/// worker inherits, and a source SVG never draws from the
/// filesystem. usvg resolves the hrefs of an SVG nested in an
/// image the same way.
fn image_href_resolver(budget: Arc<Mutex<NestedBudget>>) -> usvg::ImageHrefResolver<'static> {
    usvg::ImageHrefResolver {
        resolve_data: nested_svg_resolver(budget),
        resolve_string: Box::new(|_, _| None),
    }
}

/// usvg's own font lookups, with the system fonts loaded the
/// first time a document asks for a font. A document without
/// text never does, and each worker is a fresh process, so an
/// icon does not pay for scanning the font directories.
/// usvg hands each lookup the database it parses with, which
/// starts empty (the `Options` default); the first lookup
/// swaps in the system database, the one an eager load would
/// have passed, so the fonts and their order are the same.
fn font_resolver() -> usvg::FontResolver<'static> {
    let select_font = usvg::FontResolver::default_font_selector();
    let select_fallback = usvg::FontResolver::default_fallback_selector();
    let with_system_fonts = |db: &mut Arc<usvg::fontdb::Database>| {
        // Empty means nothing was selected from it yet, so
        // no font ID can refer to the database replaced.
        if db.is_empty() {
            *db = font_database();
        }
    };
    usvg::FontResolver {
        select_font: Box::new(move |font, db| {
            with_system_fonts(db);
            select_font(font, db)
        }),
        select_fallback: Box::new(move |c, used, db| {
            with_system_fonts(db);
            select_fallback(c, used, db)
        }),
    }
}

/// System fonts, loaded once per process: SVG text elements
/// need a font database, and scanning the font directories on
/// every request would dwarf the render itself.
fn font_database() -> Arc<usvg::fontdb::Database> {
    static DB: OnceLock<Arc<usvg::fontdb::Database>> = OnceLock::new();
    DB.get_or_init(|| {
        let mut db = usvg::fontdb::Database::new();
        db.load_system_fonts();
        Arc::new(db)
    })
    .clone()
}

impl Prepared {
    /// The document's own size in SVG units.
    pub fn intrinsic(&self) -> (f32, f32) {
        let size = self.tree.size();
        (size.width(), size.height())
    }

    /// Rasterize at exactly `w` x `h` device pixels (the SVG
    /// is scaled non-uniformly to the target when the caller
    /// asked for a specific frame; aspect decisions belong to
    /// the caller's target computation).
    pub fn render(&self, w: u32, h: u32) -> Result<DynamicImage, String> {
        let (iw, ih) = self.intrinsic();
        if !(iw > 0.0 && ih > 0.0) {
            return Err("SVG has no intrinsic size".to_string());
        }
        let mut pixmap = resvg::tiny_skia::Pixmap::new(w, h)
            .ok_or_else(|| "SVG raster size out of range".to_string())?;
        let transform = resvg::tiny_skia::Transform::from_scale(w as f32 / iw, h as f32 / ih);
        resvg::render(&self.tree, transform, &mut pixmap.as_mut());
        let img = image::RgbaImage::from_raw(w, h, pixmap.take())
            .ok_or_else(|| "SVG raster buffer size mismatch".to_string())?;
        Ok(DynamicImage::ImageRgba8(img))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><rect width="100" height="100" fill="#00ff00"/><circle cx="50" cy="50" r="20" fill="#ff0000"/></svg>"##;

    const HOSTILE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="40" height="40" onload="alert(1)">
<script>alert("xss")</script>
<rect width="40" height="40" fill="#0000ff" onclick="alert(2)"/>
<image xlink:href="http://127.0.0.1:9/evil.png" href="https://example.com/evil.png" width="40" height="40"/>
<use href="http://example.com/sprite.svg#frag"/>
<use href="#local"/>
<foreignObject><body xmlns="http://www.w3.org/1999/xhtml">hi</body></foreignObject>
<rect width="10" height="10" fill="url(http://example.com/paint.svg)"/>
<style>@import url("http://example.com/x.css");</style>
</svg>"##;

    #[test]
    fn sniffing_sees_through_prologs_and_rejects_non_svg() {
        assert!(is_svg(SIMPLE.as_bytes()));
        assert!(is_svg(
            "<?xml version=\"1.0\"?>\n<!-- c -->\n<svg xmlns=\"http://www.w3.org/2000/svg\"/>"
                .as_bytes()
        ));
        assert!(!is_svg(b"<note><to>x</to></note>"));
        assert!(!is_svg(b"\x89PNG\r\n\x1a\n"));
        assert!(!is_svg(b""));
    }

    #[test]
    fn sanitizer_strips_active_content_and_external_refs() {
        let clean = sanitize(HOSTILE).unwrap();
        for banned in [
            "script",
            "onload",
            "onclick",
            "alert",
            "foreignObject",
            "127.0.0.1",
            "example.com",
            "@import",
        ] {
            assert!(!clean.contains(banned), "still contains {banned}: {clean}");
        }
        // The inert drawing survives, including the fragment
        // reference and the data-free fills.
        assert!(clean.contains("#local"));
        assert!(clean.contains("#0000ff"));
    }

    /// Base64-encode (standard alphabet, padded): enough
    /// to build embedded-image fixtures in tests.
    fn b64(data: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in data.chunks(3) {
            let b = [
                chunk[0] as u32,
                *chunk.get(1).unwrap_or(&0) as u32,
                *chunk.get(2).unwrap_or(&0) as u32,
            ];
            let n = (b[0] << 16) | (b[1] << 8) | b[2];
            out.push(ALPHABET[(n >> 18) as usize & 63] as char);
            out.push(ALPHABET[(n >> 12) as usize & 63] as char);
            out.push(if chunk.len() > 1 {
                ALPHABET[(n >> 6) as usize & 63] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                ALPHABET[n as usize & 63] as char
            } else {
                '='
            });
        }
        out
    }

    fn png_bytes(w: u32, h: u32) -> Vec<u8> {
        let png = image::RgbImage::from_pixel(w, h, image::Rgb([0, 128, 255]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(png)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        bytes
    }

    /// A PNG that is only its signature and a well-formed
    /// IHDR claiming `w` x `h`. usvg sizes an embedded
    /// raster from its header, and prepare has to refuse
    /// on that size before anything decodes, so a hostile
    /// fixture can claim a gigabyte of pixels in 33 bytes.
    fn png_header_only(w: u32, h: u32) -> Vec<u8> {
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        let mut crc = flate2::Crc::new();
        crc.update(b"IHDR");
        crc.update(&ihdr);
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&13u32.to_be_bytes());
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&ihdr);
        png.extend_from_slice(&crc.sum().to_be_bytes());
        png
    }

    fn png_data_uri(png: &[u8]) -> String {
        format!("data:image/png;base64,{}", b64(png))
    }

    fn svg_document(body: &str) -> String {
        format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\">{body}</svg>")
    }

    fn svg_with_embedded_png(w: u32, h: u32) -> String {
        svg_document(&format!(
            "<g><image href=\"{}\" width=\"10\" height=\"10\"/></g>",
            png_data_uri(&png_bytes(w, h))
        ))
    }

    /// The places an SVG can draw an embedded raster from
    /// outside `Group::children()` (issue #398), as body
    /// templates whose `{IMG}` is an image `href`.
    const SUBROOT_WRAPPERS: &[(&str, &str)] = &[
        (
            "pattern fill",
            "<defs><pattern id=\"p\" width=\"10\" height=\"10\" patternUnits=\"userSpaceOnUse\">\
             <image href=\"{IMG}\" width=\"10\" height=\"10\"/></pattern></defs>\
             <rect width=\"10\" height=\"10\" fill=\"url(#p)\"/>",
        ),
        (
            "pattern stroke",
            "<defs><pattern id=\"p\" width=\"10\" height=\"10\" patternUnits=\"userSpaceOnUse\">\
             <image href=\"{IMG}\" width=\"10\" height=\"10\"/></pattern></defs>\
             <rect x=\"2\" y=\"2\" width=\"6\" height=\"6\" fill=\"none\" stroke=\"url(#p)\" stroke-width=\"2\"/>",
        ),
        (
            "mask",
            "<defs><mask id=\"m\"><image href=\"{IMG}\" width=\"10\" height=\"10\"/></mask></defs>\
             <rect width=\"10\" height=\"10\" fill=\"#ff0000\" mask=\"url(#m)\"/>",
        ),
        (
            // Three links: usvg's Node::subroots stops
            // after the second, resvg draws all three.
            "mask of a mask of a mask",
            "<defs><mask id=\"m3\"><image href=\"{IMG}\" width=\"10\" height=\"10\"/></mask>\
             <mask id=\"m2\" mask=\"url(#m3)\"><rect width=\"10\" height=\"10\" fill=\"#ffffff\"/></mask>\
             <mask id=\"m1\" mask=\"url(#m2)\"><rect width=\"10\" height=\"10\" fill=\"#ffffff\"/></mask></defs>\
             <rect width=\"10\" height=\"10\" fill=\"#ff0000\" mask=\"url(#m1)\"/>",
        ),
        (
            "feImage",
            "<defs><filter id=\"f\"><feImage href=\"{IMG}\"/></filter></defs>\
             <rect width=\"10\" height=\"10\" filter=\"url(#f)\"/>",
        ),
        (
            "feImage naming an image element",
            "<defs><image id=\"i\" href=\"{IMG}\" width=\"10\" height=\"10\"/>\
             <filter id=\"f\"><feImage href=\"#i\"/></filter></defs>\
             <rect width=\"10\" height=\"10\" filter=\"url(#f)\"/>",
        ),
    ];

    #[test]
    fn embedded_rasters_are_charged_at_their_own_dimensions() {
        // Issue #360: the <image> element is one node, but
        // its payload decodes at the payload's dimensions
        // (100x50 here, drawn into a 10x10 viewport), and
        // the accounting charges the decoded RGBA bytes.
        let svg = svg_with_embedded_png(100, 50);
        let prepared = prepare(svg.as_bytes()).unwrap();
        assert_eq!(embedded_raster_bytes(prepared.tree.root()), 100 * 50 * 4);
        // An SVG with no embedded raster charges nothing.
        let prepared = prepare(SIMPLE.as_bytes()).unwrap();
        assert_eq!(embedded_raster_bytes(prepared.tree.root()), 0);
    }

    #[test]
    fn rasters_outside_group_children_are_charged() {
        // Issue #398: pattern, mask, and feImage contents
        // live in usvg subroots, not Group::children(). A
        // small payload in each wrapper still prepares, and
        // is charged at its decoded size.
        let small = png_data_uri(&png_bytes(100, 50));
        for (name, template) in SUBROOT_WRAPPERS {
            let svg = svg_document(&template.replace("{IMG}", &small));
            let prepared = prepare(svg.as_bytes()).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                embedded_raster_bytes(prepared.tree.root()),
                100 * 50 * 4,
                "{name}"
            );
        }
    }

    #[test]
    fn hostile_rasters_outside_group_children_are_refused() {
        // Issue #398: a 16000x16000 payload (about 1 GB as
        // RGBA) was refused as a plain <image> but decoded
        // at full size on the stills lane when wrapped in a
        // pattern, a mask, or an feImage. Every wrapper is
        // now refused by prepare with the budget error.
        let plain = svg_document(&format!(
            "<image href=\"{}\" width=\"10\" height=\"10\"/>",
            png_data_uri(&png_header_only(16_000, 16_000))
        ));
        let err = prepare(plain.as_bytes()).err().unwrap();
        assert!(err.contains("decode budget"), "plain image: {err}");
        let huge = png_data_uri(&png_header_only(16_000, 16_000));
        for (name, template) in SUBROOT_WRAPPERS {
            let svg = svg_document(&template.replace("{IMG}", &huge));
            let err = prepare(svg.as_bytes())
                .err()
                .unwrap_or_else(|| panic!("{name}: prepared"));
            assert!(err.contains("decode budget"), "{name}: {err}");
        }
    }

    #[test]
    fn hostile_raster_in_a_nested_svg_pattern_is_refused() {
        // A nested SVG payload is parsed by usvg straight
        // from its data: URI; a pattern inside it is a
        // subroot of the nested tree, and its rasters count.
        let inner = svg_document(
            &SUBROOT_WRAPPERS[0]
                .1
                .replace("{IMG}", &png_data_uri(&png_header_only(16_000, 16_000))),
        );
        let svg = svg_document(&format!(
            "<image href=\"data:image/svg+xml;base64,{}\" width=\"10\" height=\"10\"/>",
            b64(inner.as_bytes())
        ));
        let err = prepare(svg.as_bytes()).err().unwrap();
        assert!(err.contains("decode budget"), "{err}");
    }

    #[test]
    fn a_pattern_drawn_many_times_is_charged_each_time() {
        // resvg renders a pattern's tile once per path it
        // fills, decoding the tile's raster each time, so a
        // shared pattern is charged once per reference:
        // 100 fills of a 1000x1000 payload is 400 MB of
        // decoding even though the payload appears once.
        let mut body = format!(
            "<defs><pattern id=\"p\" width=\"10\" height=\"10\" patternUnits=\"userSpaceOnUse\">\
             <image href=\"{}\" width=\"10\" height=\"10\"/></pattern></defs>",
            png_data_uri(&png_header_only(1_000, 1_000))
        );
        body.push_str(&"<rect width=\"10\" height=\"10\" fill=\"url(#p)\"/>".repeat(100));
        let err = prepare(svg_document(&body).as_bytes()).err().unwrap();
        assert!(err.contains("decode budget"), "{err}");
    }

    #[test]
    fn pattern_expansion_is_capped_after_parse() {
        // The expansion count walks subroots too: a
        // 1_000-rect pattern filling 150 rects is drawn as
        // ~150k nodes, with no <use> in the document.
        let mut doc = String::from(
            "<defs><pattern id=\"p\" width=\"10\" height=\"10\" patternUnits=\"userSpaceOnUse\">",
        );
        doc.push_str(&"<rect width=\"1\" height=\"1\"/>".repeat(1_000));
        doc.push_str("</pattern></defs>");
        let once = format!("{doc}<rect width=\"10\" height=\"10\" fill=\"url(#p)\"/>");
        assert!(prepare(svg_document(&once).as_bytes()).is_ok());
        doc.push_str(&"<rect width=\"10\" height=\"10\" fill=\"url(#p)\"/>".repeat(150));
        let err = prepare(svg_document(&doc).as_bytes()).err().unwrap();
        assert!(err.contains("node limit"), "{err}");
    }

    #[test]
    fn expansion_count_stops_at_the_cap() {
        // A chain of patterns, each filling ten rects with
        // the one before, draws 10^6 rects from a ~60-node
        // document; the count stops one past the cap.
        let mut body = String::from(
            "<defs><pattern id=\"p0\" width=\"10\" height=\"10\" patternUnits=\"userSpaceOnUse\">",
        );
        body.push_str(&"<rect width=\"1\" height=\"1\"/>".repeat(10));
        body.push_str("</pattern>");
        for level in 1..6 {
            body.push_str(&format!(
                "<pattern id=\"p{level}\" width=\"10\" height=\"10\" patternUnits=\"userSpaceOnUse\">"
            ));
            body.push_str(
                &format!(
                    "<rect width=\"1\" height=\"1\" fill=\"url(#p{})\"/>",
                    level - 1
                )
                .repeat(10),
            );
            body.push_str("</pattern>");
        }
        body.push_str("</defs><rect width=\"10\" height=\"10\" fill=\"url(#p5)\"/>");
        let text = sanitize(&svg_document(&body)).unwrap();
        let opt = usvg::Options::default();
        let tree = usvg::Tree::from_str(&text, &opt).unwrap();
        assert_eq!(count_expanded(tree.root(), 1_000), 1_001);
        let err = prepare(svg_document(&body).as_bytes()).err().unwrap();
        assert!(err.contains("node limit"), "{err}");
    }

    #[test]
    fn raster_budget_refuses_past_the_decode_budget() {
        // Issue #360: the accounting feeds the shared
        // decode allocation budget; past it, prepare
        // refuses instead of letting the renderer decode.
        assert!(check_raster_budget(crate::engine::MAX_DECODE_ALLOC).is_ok());
        let err = check_raster_budget(crate::engine::MAX_DECODE_ALLOC + 4).unwrap_err();
        assert!(err.contains("decode budget"), "{err}");
    }

    #[test]
    fn sanitized_bytes_are_the_sanitizers_output() {
        // The upload endpoint stores exactly this form: the
        // active content is gone, the drawing survives.
        let stored = sanitized_bytes(HOSTILE.as_bytes()).unwrap();
        let stored = String::from_utf8(stored).unwrap();
        assert_eq!(stored, sanitize(HOSTILE).unwrap());
        assert!(!stored.contains("script"), "{stored}");
        assert!(stored.contains("#0000ff"), "{stored}");
        // Non-UTF-8 input is refused, like prepare.
        assert!(sanitized_bytes(&[0xff, 0xfe]).is_err());
    }

    #[test]
    fn sanitizer_rejects_deep_nesting() {
        // The node cap alone would allow this nest; the
        // depth cap is what bounds the recursive passes.
        let mut svg = String::from("<svg xmlns=\"http://www.w3.org/2000/svg\">");
        for _ in 0..100 {
            svg.push_str("<g>");
        }
        svg.push_str("<rect width=\"1\" height=\"1\"/>");
        for _ in 0..100 {
            svg.push_str("</g>");
        }
        svg.push_str("</svg>");
        let err = sanitize(&svg).unwrap_err();
        assert!(err.contains("nests deeper"), "{err}");
    }

    #[test]
    fn case_variants_of_active_content_are_still_stripped() {
        // The case-insensitive checks allocate nothing, but
        // they must keep catching the uppercase spellings.
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" OnLoad=\"alert(1)\"><rect width=\"1\" height=\"1\" fill=\"URL(https://example.com/x)\"/><image href=\"DATA:image/png;base64,AA==\" width=\"1\" height=\"1\"/></svg>";
        let clean = sanitize(svg).unwrap();
        assert!(!clean.contains("OnLoad"), "{clean}");
        assert!(!clean.contains("example.com"), "{clean}");
        assert!(clean.contains("DATA:image/png"), "{clean}");
    }

    #[test]
    fn sanitizer_keeps_fragment_refs_and_data_payloads() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><defs><linearGradient id="g"><stop offset="0" stop-color="#ffffff"/></linearGradient></defs><rect width="10" height="10" fill="url(#g)"/><image href="data:image/png;base64,iVBORw0KGgo=" width="10" height="10"/></svg>"##;
        let clean = sanitize(svg).unwrap();
        assert!(clean.contains("url(#g)"), "{clean}");
        assert!(clean.contains("data:image/png"), "{clean}");
    }

    #[test]
    fn the_renderer_reads_no_file_an_href_names() {
        // usvg hands every href that is not a data URL to the
        // string resolver, whose default reads it as a file
        // path. The renderer's resolver must find nothing, even
        // for an absolute path to a readable PNG.
        let dir = crate::test_support::scratch_dir("svg-href-file");
        let png = dir.join("red.png");
        DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            4,
            4,
            image::Rgba([255, 0, 0, 255]),
        ))
        .save(&png)
        .unwrap();
        let budget = Arc::new(Mutex::new(NestedBudget {
            nodes: MAX_EXPANDED_NODES as u64,
            bytes: crate::engine::MAX_DECODE_ALLOC,
            depth: MAX_CONVERSION_DEPTH,
            refusal: None,
        }));
        let resolver = image_href_resolver(budget);
        let options = usvg::Options::default();
        let path = png.to_str().unwrap();
        assert!((resolver.resolve_string)(path, &options).is_none());
        // The same bytes embedded as a data payload still draw.
        let payload = Arc::new(std::fs::read(&png).unwrap());
        assert!(
            (resolver.resolve_data)("image/png", payload, &options).is_some(),
            "a data payload must still draw"
        );
    }

    #[test]
    fn sanitizer_keeps_only_hrefs_that_point_inside_the_document() {
        for kept in [
            "#g",
            "#a.b-c_d",
            "data:image/png;base64,iVBORw0KGgo=",
            "DATA:image/svg+xml,%3Csvg/%3E",
            "data:,",
        ] {
            assert!(is_document_href(kept), "{kept}");
        }
        for dropped in [
            "#",
            "#/../x.png",
            "#a/b",
            "#a\\b",
            "data:x.png",
            "data:/../../etc/x.png",
            "/etc/x.png",
            "x.png",
            "file:///etc/x.png",
        ] {
            assert!(!is_document_href(dropped), "{dropped}");
        }
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><image href="data:x.png" width="10" height="10"/><image href="#/../x.png" width="10" height="10"/><use href="#r"/><rect id="r" width="1" height="1"/></svg>"##;
        let clean = sanitize(svg).unwrap();
        assert!(!clean.contains("data:x.png"), "{clean}");
        assert!(!clean.contains("#/../"), "{clean}");
        assert!(clean.contains("href=\"#r\""), "{clean}");
    }

    #[test]
    fn rasterizes_expected_pixels_at_intrinsic_size() {
        let prep = prepare(SIMPLE.as_bytes()).unwrap();
        assert_eq!(prep.intrinsic(), (100.0, 100.0));
        let img = prep.render(100, 100).unwrap().to_rgba8();
        assert_eq!(img.get_pixel(50, 50), &image::Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(5, 5), &image::Rgba([0, 255, 0, 255]));
    }

    #[test]
    fn hostile_svg_renders_inert() {
        let prep = prepare(HOSTILE.as_bytes()).unwrap();
        let img = prep.render(40, 40).unwrap().to_rgba8();
        // The blue rect is drawn; the external image and the
        // url() paint contribute nothing.
        assert_eq!(img.get_pixel(20, 20), &image::Rgba([0, 0, 255, 255]));
    }

    #[test]
    fn non_svg_xml_is_rejected() {
        assert!(prepare(b"<note><to>x</to></note>").is_err());
        assert!(sanitize("<html><body>x</body></html>").is_err());
    }

    #[test]
    fn doctype_with_an_internal_subset_is_rejected() {
        let subset = r#"<!DOCTYPE svg [ <!ENTITY x "boom"> ]><svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>"#;
        assert!(sanitize(subset).is_err());
        // A bare DOCTYPE is rejected too: the XML parser
        // refuses DTDs outright, which the sanitizer surfaces.
        let bare = r#"<!DOCTYPE svg><svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>"#;
        assert!(sanitize(bare).is_err());
    }

    #[test]
    fn document_complexity_is_capped() {
        // Node cap: 10_001 elements is past it.
        let many = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\">{}</svg>",
            "<rect width=\"1\" height=\"1\"/>".repeat(10_001)
        );
        assert!(sanitize(&many).is_err());
        // Declared-dimension cap.
        let huge = r##"<svg xmlns="http://www.w3.org/2000/svg" width="200000" height="10"></svg>"##;
        assert!(sanitize(huge).is_err());
        // Use nesting depth: a chain of groups, each using the
        // previous one, 20 deep.
        let mut doc = String::from(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><defs><g id=\"g0\"><rect width=\"1\" height=\"1\"/></g>",
        );
        for i in 1..20 {
            doc.push_str(&format!("<g id=\"g{i}\"><use href=\"#g{}\"/></g>", i - 1));
        }
        doc.push_str("</defs></svg>");
        assert!(sanitize(&doc).is_err());
    }

    #[test]
    fn use_expansion_is_capped_after_parse() {
        // A 1_000-rect group referenced by 150 uses is a small
        // source document but expands to ~150k tree nodes.
        let mut doc = String::from(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><defs><g id=\"big\">",
        );
        doc.push_str(&"<rect width=\"1\" height=\"1\"/>".repeat(1_000));
        doc.push_str("</g></defs>");
        doc.push_str(&"<use href=\"#big\"/>".repeat(150));
        doc.push_str("</svg>");
        assert!(sanitize(&doc).is_ok());
        assert!(prepare(doc.as_bytes()).is_err());
    }

    /// The issue #443 document: level `i` holds a rect drawn
    /// through a filter of `k` feImages, each naming the rect
    /// of level `i - 1`, so the top rect converts k^n rects.
    /// `href` names the reference attribute.
    fn feimage_chain(k: usize, n: usize, href: &str) -> String {
        let mut body =
            String::from("<defs><rect id=\"r0\" width=\"10\" height=\"10\" fill=\"#ff0000\"/>");
        for i in 1..=n {
            body.push_str(&format!("<filter id=\"f{i}\">"));
            for _ in 0..k {
                body.push_str(&format!("<feImage {href}=\"#r{}\"/>", i - 1));
            }
            body.push_str(&format!(
                "</filter><rect id=\"r{i}\" width=\"10\" height=\"10\" filter=\"url(#f{i})\"/>"
            ));
        }
        body.push_str(&format!(
            "</defs><rect width=\"10\" height=\"10\" filter=\"url(#f{n})\"/>"
        ));
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"10\" height=\"10\">{body}</svg>"
        )
    }

    /// The bound alone, as `prepare` runs it before usvg.
    fn bound_refusal(svg: &str) -> String {
        let clean = sanitize(svg).unwrap();
        let doc = roxmltree::Document::parse(&clean).unwrap();
        check_expansion(&doc, MAX_CONVERSION_DEPTH).expect_err("the bound accepted it")
    }

    #[test]
    fn feimage_chain_is_refused_before_parsing() {
        // Issue #443: K=10, N=7 is a ~2 KB document that built
        // 10^7 nodes inside usvg and aborted the process on
        // allocation failure before the post-parse count ran.
        let svg = feimage_chain(10, 7, "href");
        assert!(svg.len() < 4_096, "{}", svg.len());
        assert!(bound_refusal(&svg).contains("node limit"));
        let started = std::time::Instant::now();
        let err = prepare(svg.as_bytes()).err().unwrap();
        assert!(err.contains("node limit"), "{err}");
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        // Narrower and deeper, through xlink:href, and both
        // spellings at once: all refused by the bound.
        for svg in [
            feimage_chain(2, 20, "href"),
            feimage_chain(10, 7, "xlink:href"),
            feimage_chain(10, 7, "href=\"#r0\" xlink:href"),
            feimage_chain(10, 6, "href"),
        ] {
            assert!(bound_refusal(&svg).contains("node limit"), "{svg}");
        }
        // Small chains still render: 10^3 rects is within the
        // limit, and the bound agrees with the count of the
        // tree as built.
        let small = feimage_chain(10, 3, "href");
        let prepared = prepare(small.as_bytes()).unwrap();
        assert!(count_expanded(prepared.tree.root(), MAX_EXPANDED_NODES) < 2_000);
    }

    #[test]
    fn feimage_multiplied_through_use_is_refused() {
        // Each level's feImage names a group of ten <use>
        // copies of the level below: <use> multiplies with
        // feImage even though each is within its own caps.
        let mut body = String::from("<defs><rect id=\"r0\" width=\"10\" height=\"10\"/>");
        for i in 1..=7 {
            body.push_str(&format!("<g id=\"u{i}\">"));
            body.push_str(&format!("<use href=\"#r{}\"/>", i - 1).repeat(10));
            body.push_str(&format!(
                "</g><filter id=\"f{i}\"><feImage href=\"#u{i}\"/></filter>\
                 <rect id=\"r{i}\" width=\"10\" height=\"10\" filter=\"url(#f{i})\"/>"
            ));
        }
        body.push_str("</defs><rect width=\"10\" height=\"10\" filter=\"url(#f7)\"/>");
        let svg = svg_document(&body);
        assert!(bound_refusal(&svg).contains("node limit"));
        assert!(prepare(svg.as_bytes()).is_err());
    }

    #[test]
    fn masks_clip_paths_and_markers_chains_are_refused() {
        // The same multiplication through the other references
        // usvg converts per use: masks (objectBoundingBox
        // units, the default), clip paths in objectBoundingBox
        // units, and markers (once per vertex).
        let mut mask =
            String::from("<defs><mask id=\"m0\"><rect width=\"1\" height=\"1\"/></mask>");
        let mut clip = String::from(
            "<defs><clipPath id=\"c0\" clipPathUnits=\"objectBoundingBox\"><rect width=\"1\" height=\"1\"/></clipPath>",
        );
        let mut marker =
            String::from("<defs><marker id=\"k0\"><rect width=\"1\" height=\"1\"/></marker>");
        let points = "0,0 1,1 2,2 3,3 4,4 5,5 6,6 7,7 8,8 9,9";
        for i in 1..=7 {
            let p = i - 1;
            mask.push_str(&format!("<mask id=\"m{i}\">"));
            mask.push_str(
                &format!("<rect width=\"1\" height=\"1\" mask=\"url(#m{p})\"/>").repeat(10),
            );
            mask.push_str("</mask>");
            clip.push_str(&format!(
                "<clipPath id=\"c{i}\" clipPathUnits=\"objectBoundingBox\">"
            ));
            clip.push_str(
                &format!("<rect width=\"1\" height=\"1\" clip-path=\"url(#c{p})\"/>").repeat(10),
            );
            clip.push_str("</clipPath>");
            marker.push_str(&format!(
                "<marker id=\"k{i}\"><polyline points=\"{points}\" marker-mid=\"url(#k{p})\"/></marker>"
            ));
        }
        mask.push_str("</defs><rect width=\"10\" height=\"10\" mask=\"url(#m7)\"/>");
        clip.push_str("</defs><rect width=\"10\" height=\"10\" clip-path=\"url(#c7)\"/>");
        marker.push_str(&format!(
            "</defs><polyline points=\"{points}\" marker-mid=\"url(#k7)\"/>"
        ));
        for body in [mask, clip, marker] {
            let svg = svg_document(&body);
            assert!(bound_refusal(&svg).contains("node limit"), "{svg}");
            assert!(prepare(svg.as_bytes()).is_err());
        }
        // A marker chain declared through CSS rules, the
        // `marker` shorthand, and inheritance (the polyline in
        // each marker inherits the marker element's value) is
        // followed the same way.
        let mut css = String::from("<style>");
        for i in 1..=7 {
            css.push_str(&format!("#k{i} {{ marker-mid: url(#k{}) }}", i - 1));
        }
        css.push_str("</style><defs><marker id=\"k0\"><rect width=\"1\" height=\"1\"/></marker>");
        for i in 1..=7 {
            css.push_str(&format!(
                "<marker id=\"k{i}\"><polyline points=\"{points}\"/></marker>"
            ));
        }
        css.push_str(
            "</defs><g style=\"marker: url(#k7)\"><path d=\"M0 0 L1 1 L2 2 L3 3 L4 4 L5 5 L6 6 L7 7 L8 8 L9 9\"/></g>",
        );
        assert!(bound_refusal(&svg_document(&css)).contains("node limit"));
    }

    #[test]
    fn reference_cycles_and_long_chains_are_refused() {
        // Three-link cycles that usvg's own self-reference
        // fixes miss: before #443 each overflowed the stack and
        // aborted the process from a ~400-byte document.
        let cycles = [
            "<defs><filter id=\"f1\"><feImage href=\"#r2\"/></filter><filter id=\"f2\"><feImage href=\"#r3\"/></filter>\
             <filter id=\"f3\"><feImage href=\"#r1\"/></filter><rect id=\"r1\" width=\"1\" height=\"1\" filter=\"url(#f1)\"/>\
             <rect id=\"r2\" width=\"1\" height=\"1\" filter=\"url(#f2)\"/><rect id=\"r3\" width=\"1\" height=\"1\" filter=\"url(#f3)\"/></defs>\
             <rect width=\"1\" height=\"1\" filter=\"url(#f1)\"/>",
            "<defs><mask id=\"m1\"><rect width=\"1\" height=\"1\" mask=\"url(#m2)\"/></mask>\
             <mask id=\"m2\"><rect width=\"1\" height=\"1\" mask=\"url(#m3)\"/></mask>\
             <mask id=\"m3\"><rect width=\"1\" height=\"1\" mask=\"url(#m1)\"/></mask></defs>\
             <rect width=\"1\" height=\"1\" mask=\"url(#m1)\"/>",
            "<defs><pattern id=\"p1\" width=\"1\" height=\"1\"><rect width=\"1\" height=\"1\" fill=\"url(#p2)\"/></pattern>\
             <pattern id=\"p2\" width=\"1\" height=\"1\"><rect width=\"1\" height=\"1\" fill=\"url(#p3)\"/></pattern>\
             <pattern id=\"p3\" width=\"1\" height=\"1\"><rect width=\"1\" height=\"1\" fill=\"url(#p1)\"/></pattern></defs>\
             <rect width=\"1\" height=\"1\" fill=\"url(#p1)\"/>",
        ];
        for body in cycles {
            let svg = svg_document(body);
            assert!(bound_refusal(&svg).contains("cycle"), "{svg}");
            assert!(prepare(svg.as_bytes()).err().unwrap().contains("cycle"));
        }
        // `href` chains that loop without passing through the
        // element usvg started resolving (`#a`→`#b`→`#c`→`#b`).
        // usvg's own guard misses these and follows them with
        // no end, so before #443 each hung the blocking thread
        // from a few hundred bytes: a gradient (the issue's
        // named case), and a pattern and a filter whose entry
        // carries children (usvg still walks the chain to
        // resolve absent tiling and region attributes).
        let href_cycles = [
            "<defs><linearGradient id=\"a\" href=\"#b\"/><linearGradient id=\"b\" href=\"#c\"/>\
             <linearGradient id=\"c\" href=\"#b\"/></defs><rect width=\"10\" height=\"10\" fill=\"url(#a)\"/>",
            "<defs><pattern id=\"a\" href=\"#b\" width=\"1\" height=\"1\"><rect width=\"1\" height=\"1\"/></pattern>\
             <pattern id=\"b\" href=\"#c\"/><pattern id=\"c\" href=\"#b\"/></defs>\
             <rect width=\"10\" height=\"10\" fill=\"url(#a)\"/>",
            "<defs><filter id=\"a\" href=\"#b\"><feFlood flood-color=\"#00ff00\"/></filter>\
             <filter id=\"b\" href=\"#c\"/><filter id=\"c\" href=\"#b\"/></defs>\
             <rect width=\"10\" height=\"10\" filter=\"url(#a)\"/>",
        ];
        for body in href_cycles {
            let svg = svg_document(body);
            assert!(bound_refusal(&svg).contains("cycle"), "{svg}");
            let started = std::time::Instant::now();
            assert!(
                prepare(svg.as_bytes()).err().unwrap().contains("cycle"),
                "{svg}"
            );
            assert!(started.elapsed() < std::time::Duration::from_secs(5));
        }
        // A linear chain of 200 feImage references overflowed a
        // 2 MiB stack too; it is refused for depth.
        let svg = feimage_chain(1, 200, "href");
        let err = prepare(svg.as_bytes()).err().unwrap();
        assert!(err.contains("levels once references are followed"), "{err}");
        // The longest chain the depth cap admits converts on
        // this (debug, 2 MiB) test thread without overflowing.
        assert!(prepare(feimage_chain(1, 63, "href").as_bytes()).is_ok());
        assert!(prepare(feimage_chain(1, 64, "href").as_bytes()).is_err());
    }

    #[test]
    fn nested_svg_images_are_held_to_the_bound() {
        // The chain inside a data: SVG image never reaches the
        // sanitizer; the resolver bounds it before usvg parses
        // it, and its refusal refuses the document.
        let inner = feimage_chain(10, 7, "href");
        for mime in ["image/svg+xml", "text/plain"] {
            let svg = svg_document(&format!(
                "<image width=\"10\" height=\"10\" href=\"data:{mime};base64,{}\"/>",
                b64(inner.as_bytes())
            ));
            let started = std::time::Instant::now();
            let err = prepare(svg.as_bytes()).err().unwrap();
            assert!(err.contains("node limit"), "{mime}: {err}");
            assert!(started.elapsed() < std::time::Duration::from_secs(5));
        }
        // Nested images are charged once per instance: a nested
        // document of ~1,100 nodes is fine once and refused when
        // the outer document draws it 100 times.
        let medium = svg_document(&format!(
            "<image id=\"i\" width=\"10\" height=\"10\" href=\"data:image/svg+xml;base64,{}\"/>",
            b64(feimage_chain(10, 3, "href").as_bytes())
        ));
        assert!(prepare(medium.as_bytes()).is_ok());
        let many = medium.replace(
            "</svg>",
            &format!("{}</svg>", "<use href=\"#i\"/>".repeat(100)),
        );
        let err = prepare(many.as_bytes()).err().unwrap();
        assert!(err.contains("node limit"), "{err}");
        // Nested images share the depth cap, because usvg
        // converts one on top of the conversion that reached it:
        // a 60-hop chain ending in an image that holds another
        // 60-hop chain overflowed a 2 MiB stack, though each
        // chain passes the cap alone.
        let chain = |leaf: &str| {
            let mut body = format!("<defs>{leaf}");
            for i in 1..=60 {
                body.push_str(&format!(
                    "<filter id=\"f{i}\"><feImage href=\"#r{}\"/></filter>\
                     <rect id=\"r{i}\" width=\"10\" height=\"10\" filter=\"url(#f{i})\"/>",
                    i - 1
                ));
            }
            body.push_str("</defs><rect width=\"10\" height=\"10\" filter=\"url(#f60)\"/>");
            svg_document(&body)
        };
        let inner = chain("<rect id=\"r0\" width=\"10\" height=\"10\"/>");
        let image = format!(
            "<image id=\"r0\" width=\"10\" height=\"10\" href=\"data:image/svg+xml;base64,{}\"/>",
            b64(inner.as_bytes())
        );
        assert!(prepare(svg_document(&image).as_bytes()).is_ok());
        let err = prepare(chain(&image).as_bytes()).err().unwrap();
        assert!(err.contains("levels once references are followed"), "{err}");
        // A gzip-compressed nested SVG is not inflated (usvg
        // would inflate it with no size limit): it draws
        // nothing, and the document still renders.
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut gz, SIMPLE.as_bytes()).unwrap();
        let svgz = svg_document(&format!(
            "<rect width=\"10\" height=\"10\" fill=\"#0000ff\"/><image width=\"10\" height=\"10\" href=\"data:image/svg+xml;base64,{}\"/>",
            b64(&gz.finish().unwrap())
        ));
        let img = prepare(svgz.as_bytes())
            .unwrap()
            .render(10, 10)
            .unwrap()
            .to_rgba8();
        assert_eq!(img.get_pixel(5, 5), &image::Rgba([0, 0, 255, 255]));
    }

    #[test]
    fn data_payloads_are_bounded_per_instance_before_parsing() {
        // usvg decodes an image's data: payload into a fresh
        // buffer for every instance; 300 instances of a 1 MB
        // payload pass the decode budget before anything is
        // decoded.
        let payload = format!("data:image/png;base64,{}", "A".repeat(1 << 20));
        let mut body = format!(
            "<defs><image id=\"i\" href=\"{payload}\" width=\"1\" height=\"1\"/><g id=\"g\">{}</g></defs>",
            "<use href=\"#i\"/>".repeat(20)
        );
        body.push_str(&"<use href=\"#g\"/>".repeat(15));
        let svg = svg_document(&body);
        assert!(bound_refusal(&svg).contains("decode budget"));
    }

    #[test]
    fn legitimate_feimage_documents_still_render() {
        // An feImage naming an in-document element draws it:
        // the green rect replaces the red one it filters.
        let single = svg_document(
            "<defs><rect id=\"g\" width=\"10\" height=\"10\" fill=\"#00ff00\"/>\
             <filter id=\"f\" x=\"0\" y=\"0\" width=\"1\" height=\"1\"><feImage href=\"#g\"/></filter></defs>\
             <rect width=\"10\" height=\"10\" fill=\"#ff0000\" filter=\"url(#f)\"/>",
        );
        let img = prepare(single.as_bytes())
            .unwrap()
            .render(10, 10)
            .unwrap()
            .to_rgba8();
        assert_eq!(img.get_pixel(5, 5), &image::Rgba([0, 255, 0, 255]));
        // A few references, a data: raster, and ordinary filter
        // primitives all prepare.
        let few = svg_document(
            "<defs><rect id=\"g\" width=\"10\" height=\"10\" fill=\"#00ff00\"/>\
             <filter id=\"f\"><feImage href=\"#g\"/><feImage xlink:href=\"#g\"/><feGaussianBlur stdDeviation=\"1\"/></filter>\
             <filter id=\"b\"><feGaussianBlur stdDeviation=\"2\"/><feOffset dx=\"1\"/></filter></defs>\
             <rect width=\"10\" height=\"10\" filter=\"url(#f)\"/><rect width=\"10\" height=\"10\" filter=\"url(#f)\"/>\
             <circle r=\"4\" filter=\"url(#b) blur(1)\"/>",
        )
        .replace("<svg ", "<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" ");
        assert!(prepare(few.as_bytes()).is_ok());
        let raster = svg_document(&format!(
            "<defs><filter id=\"f\"><feImage href=\"{}\"/></filter></defs><rect width=\"10\" height=\"10\" filter=\"url(#f)\"/>",
            png_data_uri(&png_bytes(10, 10))
        ));
        assert!(prepare(raster.as_bytes()).is_ok());
        // A marker on an arrowed path, a pattern fill, and a
        // mask and clip path drawn by many shapes stay well
        // within the bound.
        let mut everyday = String::from(
            "<defs><marker id=\"a\" markerWidth=\"4\" markerHeight=\"4\"><path d=\"M0 0 L4 2 L0 4 z\"/></marker>\
             <pattern id=\"p\" width=\"4\" height=\"4\" patternUnits=\"userSpaceOnUse\"><rect width=\"2\" height=\"2\"/></pattern>\
             <mask id=\"m\"><rect width=\"10\" height=\"10\" fill=\"#ffffff\"/></mask>\
             <clipPath id=\"c\"><circle r=\"5\" cx=\"5\" cy=\"5\"/></clipPath></defs>",
        );
        everyday.push_str(
            &"<path d=\"M0 0 C 2 4 6 4 8 0\" marker-end=\"url(#a)\" fill=\"url(#p)\" mask=\"url(#m)\" clip-path=\"url(#c)\"/>"
                .repeat(500),
        );
        assert!(prepare(svg_document(&everyday).as_bytes()).is_ok());
    }
}
