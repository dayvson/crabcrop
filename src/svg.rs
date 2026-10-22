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

use std::sync::{Arc, OnceLock};

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
    BANNED_ELEMENTS.contains(&name.to_lowercase().as_str())
}

/// Whether a CSS fragment references anything besides an
/// in-document fragment or an embedded `data:` payload.
fn has_external_url_ref(value: &str) -> bool {
    let lower = value.to_lowercase();
    let mut rest = lower.as_str();
    while let Some(i) = rest.find("url(") {
        let after = &rest[i + 4..];
        let target = after.trim_start_matches([' ', '\t', '"', '\'']);
        if !(target.starts_with('#') || target.starts_with("data:")) {
            return true;
        }
        rest = after;
    }
    false
}

fn keep_attribute(attr: roxmltree::Attribute) -> bool {
    let name = attr.name().to_lowercase();
    // Event handlers (onclick, onload, ...) are behavior, not
    // drawing.
    if name.starts_with("on") {
        return false;
    }
    if name == "href" {
        let value = attr.value().trim();
        return value.starts_with('#') || value.to_lowercase().starts_with("data:");
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
    !text.to_lowercase().contains("@import") && !has_external_url_ref(&text)
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
    let doc = roxmltree::Document::parse(text).map_err(|e| format!("invalid SVG: {e}"))?;
    let root = doc.root_element();
    if !root.tag_name().name().eq_ignore_ascii_case("svg") {
        return Err("not an SVG document".to_string());
    }
    let mut out = String::with_capacity(text.len());
    write_node(root, &mut out, true);
    Ok(out)
}

/// A sanitized, parsed SVG ready to rasterize.
pub struct Prepared {
    tree: usvg::Tree,
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
}
