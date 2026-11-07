//! The fuzz harness for the untrusted surfaces in front
//! of the server: the URL parser (issue #146), the
//! sealed-token opener (issue #255), and the byte
//! surfaces behind them (issue #387): image decode,
//! source metadata extraction, and filter chains.
//!
//! The harness depends on the server's library target
//! (issue #417), so every check runs the server's own
//! code: the parser in `crabcrop::url`, the decode
//! dispatch in `crabcrop::engine`, the metadata code in
//! `crabcrop::color`, the filters in `crabcrop::filters`,
//! and the SVG pipeline in `crabcrop::svg`. The opener
//! lives in `crabcrop-url`, a direct dependency.
//! `check` drives the parser target, `check_seal` the
//! opener target, and `check_decode`, `check_meta`, and
//! `check_chain` the byte targets; all of them also run
//! under the deterministic smoke test.

pub use crabcrop::{color, filters, svg, url};

use crabcrop::engine;

use url::Signature;

/// The round-trip invariants every input must satisfy,
/// however malformed:
///
/// - Nothing panics: parsing, canonicalizing, and
///   re-serializing untrusted text is total.
/// - A path that parses canonicalizes to a fixed point:
///   canonical form, re-parsed and re-canonicalized, is
///   byte-identical.
/// - The canonical key form is likewise a fixed point.
pub fn check(input: &str) {
    if let Ok((signature, key_id, request)) = url::parse_request(input) {
        let canonical = url::canonical_form(&request.operations, request.source_url);
        let segment = match &signature {
            Signature::Unsafe => "unsafe".to_string(),
            Signature::Hmac(hex) => match &key_id {
                Some(kid) => format!("{kid}:{hex}"),
                None => hex.clone(),
            },
        };
        let rebuilt = format!("/{segment}/{canonical}");
        let reparsed = url::parse_request(&rebuilt);
        let (sig2, kid2, request2) = reparsed.expect("the canonical form must re-parse");
        assert_eq!(sig2, signature, "signature changed across canonicalization");
        assert_eq!(kid2, key_id, "key ID changed across canonicalization");
        let canonical2 = url::canonical_form(&request2.operations, request2.source_url);
        assert_eq!(canonical, canonical2, "canonical form is not a fixed point");
    }
    if let Ok((_, _, rest)) = url::parse_signature(input) {
        let key1 = url::canonical_key_form(rest);
        let key2 = url::canonical_key_form(&key1);
        assert_eq!(key1, key2, "canonical key form is not a fixed point");
    }
    // The remaining entry points over untrusted text: total
    // by assertion (a panic fails the run).
    let _ = url::parse_signature(input);
    let _ = url::expand_presets(input, &std::collections::HashMap::new(), false);
    let _ = url::preset_names(input);
    // Root expansion (issue #250): total over untrusted
    // text, and a successful expansion stays under the
    // named root's base by construction. The map is a
    // fixed stand-in for operator configuration.
    let roots: std::collections::HashMap<String, String> = [
        (
            "photos".to_string(),
            "https://cdn.example.com/photos".to_string(),
        ),
        (
            "media".to_string(),
            "https://assets.example.com".to_string(),
        ),
    ]
    .into_iter()
    .collect();
    let _ = url::expand_source_root(input, &roots);
    let _ = url::expand_source_root_in_remainder(input, &roots);
    if let Ok((_, _, request)) = url::parse_request(input) {
        if let Ok((expanded, Some(reference))) = url::expand_source_root(request.source_url, &roots)
        {
            let base = &roots[&reference.name];
            assert!(
                expanded.starts_with(&format!("{base}/")),
                "root expansion escaped its base: {expanded}"
            );
        }
    }
}

/// The harness secret for the seal invariants, and a distinct
/// second secret the same token must never open under.
const SEAL_SECRET: &[u8] = b"crabcrop-fuzz-seal-secret-000001";
const SEAL_OTHER: &[u8] = b"crabcrop-fuzz-seal-secret-000002";

/// The seal-and-open invariants every input must satisfy
/// (issue #255): the opener is the second untrusted-text
/// surface, so opening attacker-controlled bytes is total,
/// and a token's confidentiality and integrity hold.
///
/// - Nothing panics: opening a token segment under a secret
///   is total, however malformed the bytes, keyed or bare.
/// - Round trip: a remainder within the cap seals and reopens
///   to itself under the minting secret, keyed and bare.
/// - Integrity: a token never opens to its remainder under a
///   different secret, so a wrong key or a tampered body is a
///   refusal, never another plaintext.
pub fn check_seal(data: &[u8]) {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };

    let _ = crabcrop_url::unseal(SEAL_SECRET, text);
    let _ = crabcrop_url::unseal(SEAL_OTHER, text);
    let _ = crabcrop_url::open_token(SEAL_SECRET, text);

    if text.len() <= crabcrop_url::MAX_SEALED_REMAINDER {
        if let Ok(token) = crabcrop_url::seal(SEAL_SECRET, text) {
            assert_eq!(
                crabcrop_url::unseal(SEAL_SECRET, &token).as_deref(),
                Ok(text),
                "a sealed token must reopen to its remainder",
            );
            assert_ne!(
                crabcrop_url::unseal(SEAL_OTHER, &token).as_deref(),
                Ok(text),
                "a token must not open to its remainder under a different secret",
            );
        }
        if let Ok(keyed) = crabcrop_url::seal_with_key_id(SEAL_SECRET, "fuzzkey", text) {
            assert_eq!(
                crabcrop_url::unseal(SEAL_SECRET, &keyed).as_deref(),
                Ok(text),
                "a keyed sealed token must reopen to its remainder",
            );
        }
    }
}

/// A token sealed under the harness secret with a fixed
/// nonce, for the smoke test's truncation and mutation sweeps
/// over a well-formed token. Deterministic, so a failing case
/// reproduces.
pub fn sample_token(remainder: &str) -> String {
    crabcrop_url::seal_with_nonce(SEAL_SECRET, None, &[0x5a; 12], remainder)
        .expect("a remainder within the cap seals")
}

/// The decode invariants (issues #387, #417): decoding
/// attacker-controlled bytes is total. An SVG-shaped
/// input goes through the sanitizer and rasterizer the
/// server uses. Anything else goes through the server's
/// decode dispatch, the branches a render walks: HEIC
/// through its decoder, animated GIF and WebP frame by
/// frame under the animation caps, JPEG XL through its
/// header probe, and the bounded still decode
/// (`engine::decode_image`: the server's dimension and
/// allocation limits, the zune-jpeg and libwebp fast
/// paths) for everything else. The still decode then runs
/// on its own as well, since a still render of an
/// animated source takes it too. Errors are the expected
/// outcome for most inputs; a panic or an unbounded
/// allocation is the failure. The SVG render size is
/// clamped so a well-formed but enormous SVG cannot turn
/// the harness itself into the allocation victim.
pub fn check_decode(data: &[u8]) {
    if svg::is_svg(data) {
        if let Ok(prepared) = svg::prepare(data) {
            let (w, h) = prepared.intrinsic();
            let w = (w.ceil() as u32).clamp(1, 128);
            let h = (h.ceil() as u32).clamp(1, 128);
            let _ = prepared.render(w, h);
        }
        return;
    }
    let _ = engine::validate_source_image(data);
    let _ = engine::decode_image(data);
}

/// The source-metadata invariants (issue #387):
/// extracting and re-injecting the metadata of
/// attacker-controlled bytes is total. Extraction reads
/// the JPEG and PNG segment structures; the TIFF patch
/// walks the extracted EXIF stream; the injectors walk
/// the container again on the way out. When the bytes
/// carry a profile and decode as a small image, the
/// color conversion runs too, since it parses the
/// profile with a real color engine.
pub fn check_meta(data: &[u8]) {
    let meta = color::extract_source_meta(data);
    if let Some(exif) = &meta.exif_tiff {
        let mut tiff = exif.clone();
        color::patch_tiff_orientation(&mut tiff);
    }
    let _ = color::inject_jpeg(data, meta.icc.as_deref(), &meta, true);
    let _ = color::inject_png(data, meta.icc.as_deref(), &meta, true);
    if let Some(icc) = &meta.icc {
        let dims = image::ImageReader::new(std::io::Cursor::new(data))
            .with_guessed_format()
            .ok()
            .and_then(|r| r.into_dimensions().ok());
        if let Some((w, h)) = dims {
            if w <= 1024 && h <= 1024 {
                if let Ok(img) = image::load_from_memory(data) {
                    let _ = color::convert_to_srgb(img, icc);
                }
            }
        }
    }
}

/// The largest canvas the chain harness lets a rotation
/// allocate. The engine checks every rotation against the
/// output pixel budget before the canvas exists (issue
/// #182); the harness applies filters directly, so it
/// applies the same kind of check with a budget sized for
/// fuzzing throughput: a chain whose next rotation would
/// land on a larger canvas stops there.
const CHAIN_MAX_PIXELS: u64 = 1 << 18;

/// The operation-chain invariants (issues #387, #415): a
/// parsed filter chain applied to an image is total, and
/// every filter lands on the canvas its kind predicts:
///
/// - a rotation lands on [`filters::rotated_canvas`] of the
///   canvas it turns: quarter turns swap the axes, any
///   other angle grows to the rotated bounding box;
/// - a trim never grows either axis;
/// - every other filter keeps the dimensions;
/// - no filter leaves a zero dimension.
///
/// The chain runs on a small synthetic image (the
/// harness's stand-in source), one filter at a time, in URL
/// order across every `filters:` segment, with the
/// request's `fill` color, resolved the way the engine
/// resolves it, for the corners an arbitrary rotation
/// uncovers. Resize, crop, and focal operations
/// are not applied: they run in the engine, which this
/// package cannot link while the server is binary-only
/// (issue #417).
pub fn check_chain(input: &str) {
    let Ok((_, _, request)) = url::parse_request(input) else {
        return;
    };
    let chain: Vec<&url::Filter> = request
        .operations
        .iter()
        .filter_map(|op| match op {
            url::Operation::Filters(filters) => Some(filters.iter()),
            _ => None,
        })
        .flatten()
        .collect();
    let fill = url::RequestDirectives::analyze(&request.operations).fill;
    let mut img = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(24, 18, |x, y| {
        image::Rgb([(x * 7) as u8, (y * 11) as u8, ((x + y) * 3) as u8])
    }));
    for filter in chain {
        let (w, h) = (img.width(), img.height());
        let predicted = match filter {
            url::Filter::Rotate(degrees) => {
                let (rw, rh) = filters::rotated_canvas(w, h, *degrees);
                if rw as u64 * rh as u64 > CHAIN_MAX_PIXELS {
                    return;
                }
                Some((rw, rh))
            }
            url::Filter::Trim(_) | url::Filter::TrimSides(..) => None,
            _ => Some((w, h)),
        };
        let (out, _) = filters::apply_filters_filled(img, std::slice::from_ref(filter), fill);
        let (ow, oh) = (out.width(), out.height());
        assert!(
            ow > 0 && oh > 0,
            "{filter:?} collapsed the canvas from {w}x{h} to {ow}x{oh}"
        );
        match predicted {
            Some(size) => assert_eq!(
                (ow, oh),
                size,
                "{filter:?} turned a {w}x{h} canvas into {ow}x{oh}, not the predicted {}x{}",
                size.0,
                size.1
            ),
            None => assert!(
                ow <= w && oh <= h,
                "{filter:?} grew the canvas from {w}x{h} to {ow}x{oh}"
            ),
        }
        img = out;
    }
}
