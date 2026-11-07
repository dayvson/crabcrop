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
//! stripped, so a sanitized SVG can only draw what it carries.
//! Rasterization is resvg/usvg (pure Rust), whose render model
//! has no scripts or network access to begin with: defense in
//! depth, with the sanitizer as the explicit, tested layer.

use std::ops::ControlFlow;
use std::sync::{Arc, OnceLock};

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
const MAX_EXPANDED_NODES: usize = 100_000;
const MAX_SVG_DIMENSION: f64 = 100_000.0;

use image::DynamicImage;
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

fn keep_attribute(attr: roxmltree::Attribute) -> bool {
    let name = attr.name();
    // Event handlers (onclick, onload, ...) are behavior, not
    // drawing.
    if starts_with_ci(name, "on") {
        return false;
    }
    if name.eq_ignore_ascii_case("href") {
        let value = attr.value().trim();
        return value.starts_with('#') || starts_with_ci(value, "data:");
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
    let mut out = String::with_capacity(text.len());
    write_node(root, &mut out, true);
    Ok(out)
}

/// Enforce the source-side bounds: total node count, `<use>`
/// count and nesting depth, and the declared root dimensions
/// (width/height/viewBox), all before tree construction.
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
        return Err(format!(
            "SVG embedded images decode past the {} byte decode budget",
            crate::engine::MAX_DECODE_ALLOC
        ));
    }
    Ok(())
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

/// Sanitize and parse `bytes`. Callers gate on [`is_svg`]
/// first; other XML is rejected here as not-an-SVG.
pub fn prepare(bytes: &[u8]) -> Result<Prepared, String> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| "SVG source is not valid UTF-8".to_string())?;
    let clean = sanitize(text)?;
    let opt = usvg::Options {
        fontdb: font_database(),
        ..Default::default()
    };
    let tree = usvg::Tree::from_str(&clean, &opt).map_err(|e| format!("cannot parse SVG: {e}"))?;
    if count_expanded(tree.root(), MAX_EXPANDED_NODES) > MAX_EXPANDED_NODES {
        return Err(format!(
            "SVG expands past the {MAX_EXPANDED_NODES} node limit"
        ));
    }
    // Runs only once the count above has passed, so this
    // walk visits at most MAX_EXPANDED_NODES nodes.
    check_raster_budget(embedded_raster_bytes(tree.root()))?;
    Ok(Prepared { tree })
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
        assert!(prepare(svg_document(&body).as_bytes()).is_err());
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
}
