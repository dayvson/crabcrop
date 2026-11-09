//! The fuzz harness for the untrusted surfaces in front
//! of the server: the URL parser (issue #146), the
//! sealed-token opener (issue #255), and the byte
//! surfaces behind them (issue #387): image decode,
//! source metadata extraction, and operation chains.
//!
//! The harness depends on the server's library target
//! (issue #417), so every check runs the server's own
//! code: the parser in `crabcrop::url`, the decode
//! dispatch and the render pipeline in `crabcrop::engine`,
//! the metadata code in `crabcrop::color`, the filters in
//! `crabcrop::filters`, and the SVG pipeline in
//! `crabcrop::svg`. The opener lives in `crabcrop-url`, a
//! direct dependency. `check` drives the parser target,
//! `check_seal` the opener target, and `check_decode`,
//! `check_meta`, and `check_chain` the byte targets; all of
//! them also run under the deterministic smoke test.

pub use crabcrop::{color, engine, filters, svg, url};

use std::collections::HashMap;
use std::sync::OnceLock;

use url::{Dimension, Filter, Operation, Signature};

/// The preset map `check` expands requests against: a
/// fixed stand-in for operator configuration (issue
/// #476), the way the root map below stands in for
/// `CRABCROP_SOURCE_ROOTS`. Every definition is one the
/// server would accept at boot (`Preset::parse` runs the
/// boot-time validation; the smoke checks it too), so
/// expansion splices real operation text into the request.
pub fn presets() -> &'static url::Presets {
    static PRESETS: OnceLock<url::Presets> = OnceLock::new();
    PRESETS.get_or_init(|| {
        [
            ("thumb", "300x200/filters:quality(80)"),
            (
                "card",
                "crop:160x90:smart/filters:format(webp):sharpen(0.5)",
            ),
            ("hero", "fit-in:1200x630:adaptive/filters:dpr(2)"),
            ("avatar", "crop:96x96:face/filters:mask(circle)"),
        ]
        .into_iter()
        .map(|(name, ops)| (name.to_string(), ops.parse().expect("a valid preset")))
        .collect()
    })
}

/// The signature segment a parsed signature re-serializes
/// to, key ID included.
fn signature_segment(signature: &Signature, key_id: Option<&str>) -> String {
    match signature {
        Signature::Unsafe => "unsafe".to_string(),
        Signature::Hmac(hex) => match key_id {
            Some(kid) => format!("{kid}:{hex}"),
            None => hex.clone(),
        },
    }
}

/// A path that parses canonicalizes to a fixed point:
/// canonical form, re-parsed and re-canonicalized, is
/// byte-identical, signature and key ID included.
fn check_round_trip(input: &str) {
    if let Ok((signature, key_id, request)) = url::parse_request(input) {
        let canonical = url::canonical_form(&request.operations, request.source_url);
        let rebuilt = format!(
            "/{}/{canonical}",
            signature_segment(&signature, key_id.as_deref())
        );
        let reparsed = url::parse_request(&rebuilt);
        let (sig2, kid2, request2) = reparsed.expect("the canonical form must re-parse");
        assert_eq!(sig2, signature, "signature changed across canonicalization");
        assert_eq!(kid2, key_id, "key ID changed across canonicalization");
        let canonical2 = url::canonical_form(&request2.operations, request2.source_url);
        assert_eq!(canonical, canonical2, "canonical form is not a fixed point");
    }
}

/// The round-trip invariants every input must satisfy,
/// however malformed:
///
/// - Nothing panics: parsing, canonicalizing, and
///   re-serializing untrusted text is total.
/// - A path that parses canonicalizes to a fixed point:
///   canonical form, re-parsed and re-canonicalized, is
///   byte-identical.
/// - The canonical key form is likewise a fixed point.
/// - Preset expansion against [`presets`] is total, with
///   and without the presets-only policy, and a path the
///   server would parse after expansion satisfies the same
///   fixed point: the expanded text is what the server
///   parses and keys the cache on.
/// - Expansion, which counts a remainder before it builds
///   anything, agrees with splicing every reference and
///   parsing the result (see [`check_counted_expansion`]).
pub fn check(input: &str) {
    check_round_trip(input);
    if let Ok((_, _, rest)) = url::parse_signature(input) {
        let key1 = url::canonical_key_form(rest);
        let key2 = url::canonical_key_form(&key1);
        assert_eq!(key1, key2, "canonical key form is not a fixed point");
    }
    // The remaining entry points over untrusted text: total
    // by assertion (a panic fails the run).
    let _ = url::parse_signature(input);
    let _ = url::expand_presets(input, &url::Presets::new(), false);
    let _ = url::preset_names(input);
    if let Ok((signature, key_id, rest)) = url::parse_signature(input) {
        let segment = signature_segment(&signature, key_id.as_deref());
        for only_presets in [false, true] {
            if let Ok(expanded) = url::expand_presets(rest, presets(), only_presets) {
                check_round_trip(&format!("/{segment}/{expanded}"));
            }
        }
        check_counted_expansion(rest);
    }
    // Root expansion (issue #250): total over untrusted
    // text, and a successful expansion stays under the
    // named root's base by construction. The map is a
    // fixed stand-in for operator configuration.
    let roots: HashMap<String, String> = [
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
    if let Ok((_, _, request)) = url::parse_request(input)
        && let Ok((expanded, Some(reference))) = url::expand_source_root(request.source_url, &roots)
    {
        let base = &roots[&reference.name];
        assert!(
            expanded.starts_with(&format!("{base}/")),
            "root expansion escaped its base: {expanded}"
        );
    }
}

/// Splice every `preset:` reference before the source into
/// its definition, counting nothing: the plain expansion
/// that [`check_counted_expansion`] holds `expand_presets`
/// to. A segment is an operation when it parses as a
/// one-segment preset body, which is the operation dispatch
/// minus the metadata and reference prefixes handled first.
/// `None` for an unknown name.
fn spliced(rest: &str, presets: &url::Presets) -> Option<String> {
    let mut out = String::new();
    let mut tail = rest;
    loop {
        let (seg, remaining) = tail.split_once('/').unwrap_or((tail, ""));
        if let Some(name) = seg.strip_prefix("preset:") {
            out.push_str(presets.get(name)?.as_str());
        } else if seg.starts_with("exp:")
            || seg.starts_with("maxage:")
            || url::parse_preset_operations(seg).is_ok()
        {
            out.push_str(seg);
        } else {
            out.push_str(tail);
            return Some(out);
        }
        if remaining.is_empty() {
            return Some(out);
        }
        out.push('/');
        tail = remaining;
    }
}

/// `expand_presets` counts a remainder's operations and the
/// length of its expansion before building it. Whatever it
/// answers must match the plain expansion ([`spliced`]):
///
/// - Accepted: the same text, within the length cap, and a
///   parse of it counts at most the operation cap.
/// - Refused for the count: over the cap, and a parse of
///   the plain expansion, when it parses, counts the same.
/// - Refused for the length: the plain expansion's length,
///   over the cap.
/// - An unknown name is refused as unknown.
///
/// With the presets-only policy, any answer other than the
/// policy's own refusal is the answer without it.
pub fn check_counted_expansion(rest: &str) {
    let presets = presets();
    let counted = url::expand_presets(rest, presets, false);
    match url::expand_presets(rest, presets, true) {
        Err(url::ParseError::PresetsOnly(_)) => {}
        only => assert_eq!(
            only, counted,
            "presets-only changed the answer for {rest:?}"
        ),
    }
    let Some(full) = spliced(rest, presets) else {
        assert!(
            matches!(counted, Err(url::ParseError::UnknownPreset(_))),
            "an unknown reference got {counted:?} for {rest:?}"
        );
        return;
    };
    let parsed = url::parse_operations_with_expiry(&full)
        .ok()
        .map(|(ops, _, _, _)| url::operation_count(&ops));
    match counted {
        Ok(expanded) => {
            assert_eq!(expanded, full, "expansion differs from the splice");
            assert!(expanded.len() <= url::MAX_EXPANDED_LEN);
            if let Some(count) = parsed {
                assert!(
                    count <= url::MAX_OPERATIONS,
                    "accepted {count} operations for {rest:?}"
                );
            }
        }
        Err(url::ParseError::TooManyOperations(count)) => {
            assert!(count > url::MAX_OPERATIONS, "refused {count} for {rest:?}");
            if let Some(parsed) = parsed {
                assert_eq!(parsed, count, "count differs from the parse for {rest:?}");
            }
        }
        Err(url::ParseError::ExpansionTooLong(len)) => {
            assert_eq!(len, full.len(), "length differs from the splice");
            assert!(len > url::MAX_EXPANDED_LEN);
        }
        Err(e) => panic!("unexpected refusal {e:?} for {rest:?}"),
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
    let meta = color::extract_source_meta(data, true);
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
        if let Some((w, h)) = dims
            && w <= 1024
            && h <= 1024
            && let Ok(img) = image::load_from_memory(data)
        {
            let _ = color::convert_to_srgb(img, icc);
        }
    }
}

/// The largest canvas the chain harness lets a chain
/// allocate. The engine checks every canvas against its
/// output pixel budget (100 million pixels) before it
/// allocates (issue #182); the harness holds both legs of
/// `check_chain` to this much smaller budget, sized for
/// fuzzing throughput and for the fuzzer's per-allocation
/// limit: a rotation in the filter leg whose canvas would
/// exceed it ends that leg, and a request whose canvas
/// could exceed it at any step never reaches the engine.
pub const CHAIN_MAX_PIXELS: u64 = 1 << 18;

/// The most blur work, in canvas pixels times kernel taps
/// summed over the chain, that either leg of `check_chain`
/// may do. A blur's cost grows with its sigma (up to
/// `url::MAX_BLUR_SIGMA`, a kernel of thousands of taps) as
/// well as with its canvas, and the fuzzing build (address
/// sanitizer plus the fuzzer's comparison tracing) runs its
/// inner loop over a hundred times slower than a debug
/// build: `blur(100)` on a 500x500 canvas took three
/// minutes there and about a second in the smoke, so
/// libFuzzer would report a request the server renders in
/// a second as a hang. A blur that would pass
/// this budget ends the filter leg, and a request whose
/// blurs could pass it never reaches the engine, as with
/// [`CHAIN_MAX_PIXELS`]; smaller sigmas and canvases still
/// reach every blur path.
pub const CHAIN_MAX_BLUR_WORK: u64 = 1 << 24;

/// The work of one blur at `sigma` over `pixels`: the
/// canvas times an upper bound on the kernel length the
/// filters size for that sigma (`blur_kernel_size`, after
/// the `url::MAX_BLUR_SIGMA` clamp).
fn blur_work(pixels: u64, sigma: f32) -> u64 {
    let sigma = if sigma.is_normal() && sigma > 0.0 {
        sigma.min(url::MAX_BLUR_SIGMA)
    } else {
        0.8
    };
    let taps = ((((sigma - 0.8) / 0.3) + 1.0) * 2.0 + 2.0).max(3.0) as u64;
    pixels.saturating_mul(taps)
}

/// The size of the harness's stand-in source.
const SOURCE_WIDTH: u32 = 24;
const SOURCE_HEIGHT: u32 = 18;

/// The harness's stand-in source: a small gradient, so
/// every filter and crop has varied pixels to work on.
fn synthetic_image() -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(
        SOURCE_WIDTH,
        SOURCE_HEIGHT,
        |x, y| image::Rgb([(x * 7) as u8, (y * 11) as u8, ((x + y) * 3) as u8]),
    ))
}

/// [`synthetic_image`] as the PNG body the engine decodes,
/// encoded once per process.
fn synthetic_png() -> &'static [u8] {
    static PNG: OnceLock<Vec<u8>> = OnceLock::new();
    PNG.get_or_init(|| {
        let mut out = std::io::Cursor::new(Vec::new());
        synthetic_image()
            .write_to(&mut out, image::ImageFormat::Png)
            .expect("the synthetic source encodes");
        out.into_inner()
    })
}

/// What [`check_chain`] did with an input, so the smoke can
/// tell an engine render from a refusal or a skip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainOutcome {
    /// The input does not parse: nothing ran.
    Unparsed,
    /// The filter leg ran; the request could outgrow
    /// [`CHAIN_MAX_PIXELS`] somewhere in its chain, or its
    /// blurs could pass [`CHAIN_MAX_BLUR_WORK`], so the
    /// engine leg was skipped.
    OverBudget,
    /// The engine refused the request.
    Refused(engine::EngineError),
    /// The engine rendered the request at this size.
    Rendered { width: u32, height: u32 },
}

/// The operation-chain invariants (issues #387, #415,
/// #476). A parsed request runs through two legs.
///
/// The filter leg applies the filters of every `filters:`
/// segment, one at a time in URL order, to the synthetic
/// source with the request's `fill` color (resolved the way
/// the engine resolves it) for the corners an arbitrary
/// rotation uncovers. Application is total, and every
/// filter lands on the canvas its kind predicts:
///
/// - a rotation lands on [`filters::rotated_canvas`] of the
///   canvas it turns: quarter turns swap the axes, any
///   other angle grows to the rotated bounding box;
/// - a trim never grows either axis;
/// - every other filter keeps the dimensions;
/// - no filter leaves a zero dimension.
///
/// A rotation whose canvas would exceed
/// [`CHAIN_MAX_PIXELS`], or a blur that would take the
/// leg past [`CHAIN_MAX_BLUR_WORK`], ends this leg.
///
/// The engine leg runs the whole operation list (resize,
/// crop, fit-in, pad, crop-rect, focal, debug view, filters
/// and the encode) through `engine::Engine::process` on the
/// synthetic source as a PNG, under the default context: no
/// AI providers, watermarks, text renderer, frame extractor
/// or PDF renderer, so those operations refuse. A request
/// whose canvas could exceed [`CHAIN_MAX_PIXELS`] at any
/// step, or whose blurs could pass [`CHAIN_MAX_BLUR_WORK`]
/// (see [`canvas_bound`]), skips this leg. Otherwise
/// the engine either refuses with an `EngineError` or
/// renders an image inside [`check_render_size`]'s bounds;
/// a panic in either leg is the failure.
pub fn check_chain(input: &str) -> ChainOutcome {
    let Ok((_, _, request)) = url::parse_request(input) else {
        return ChainOutcome::Unparsed;
    };
    let _ = check_filter_leg(&request.operations);
    let Some(bound) = canvas_bound(&request.operations) else {
        return ChainOutcome::OverBudget;
    };
    match engine::Engine.process(synthetic_png(), &request.operations) {
        Err(error) => ChainOutcome::Refused(error),
        Ok(out) => {
            assert!(!out.bytes.is_empty(), "the engine rendered an empty body");
            check_render_size(out.width, out.height, bound);
            ChainOutcome::Rendered {
                width: out.width,
                height: out.height,
            }
        }
    }
}

/// The filter leg of [`check_chain`]. Returns how many
/// filters it applied before the chain ended or a budget
/// stopped it.
fn check_filter_leg(ops: &[Operation]) -> usize {
    let chain: Vec<&Filter> = ops
        .iter()
        .filter_map(|op| match op {
            Operation::Filters(filters) => Some(filters.iter()),
            _ => None,
        })
        .flatten()
        .collect();
    let fill = url::RequestDirectives::analyze(ops).fill;
    let mut img = synthetic_image();
    let mut blur_spent: u64 = 0;
    let mut applied = 0;
    for filter in chain {
        let (w, h) = (img.width(), img.height());
        let predicted = match filter {
            Filter::Rotate(degrees) => {
                let (rw, rh) = filters::rotated_canvas(w, h, *degrees);
                if rw as u64 * rh as u64 > CHAIN_MAX_PIXELS {
                    return applied;
                }
                Some((rw, rh))
            }
            Filter::Trim(_) | Filter::TrimSides(..) => None,
            Filter::Blur(sigma) => {
                blur_spent =
                    blur_spent.saturating_add(blur_work(u64::from(w) * u64::from(h), *sigma));
                if blur_spent > CHAIN_MAX_BLUR_WORK {
                    return applied;
                }
                Some((w, h))
            }
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
        applied += 1;
    }
    applied
}

/// The size invariant on an engine render: both axes are
/// at least 1 and at most `url::MAX_DIMENSION`, the largest
/// dimension a request may name and the strictest encoder
/// limit, and neither axis is past the per-axis `bound`
/// [`canvas_bound`] proved for the request, so the guard
/// that keeps the engine leg inside the harness budget is
/// itself checked on every render.
pub fn check_render_size(width: u32, height: u32, bound: (u64, u64)) {
    let max = url::MAX_DIMENSION;
    assert!(
        (1..=max).contains(&width) && (1..=max).contains(&height),
        "the engine rendered {width}x{height}, outside 1..={max} per axis"
    );
    assert!(
        u64::from(width) <= bound.0 && u64::from(height) <= bound.1,
        "the engine rendered {width}x{height}, past the harness bound {}x{}",
        bound.0,
        bound.1
    );
}

/// Bounds on one axis of the engine's running canvas: the
/// real size lies within `lo..=hi`, and `lo` is at least 1.
#[derive(Debug, Clone, Copy)]
struct Axis {
    lo: u64,
    hi: u64,
}

impl Axis {
    fn exact(n: u64) -> Self {
        let n = n.max(1);
        Axis { lo: n, hi: n }
    }

    /// Bounds on a real-valued size, padded a pixel each way
    /// for the engine's rounding. The float-to-int casts
    /// saturate, so an infinite or huge bound stays huge.
    fn around(lo: f64, hi: f64) -> Self {
        let hi = (hi.ceil() + 1.0).max(1.0) as u64;
        let lo = ((lo.floor() - 1.0).max(1.0) as u64).min(hi);
        Axis { lo, hi }
    }

    /// A pixel target under the DPR factor (the engine's
    /// `scale_px`: identity at 1, else rounded, never 0).
    fn pixels(n: u32, dpr: f64) -> Self {
        if dpr == 1.0 {
            Axis::exact(u64::from(n))
        } else {
            let scaled = f64::from(n) * dpr;
            Axis::around(scaled, scaled)
        }
    }

    /// `no_upscale` caps a resize at the source, which can
    /// only shrink it: keep the upper bound, give up the
    /// lower one.
    fn capped(self, no_upscale: bool) -> Self {
        if no_upscale {
            Axis { lo: 1, hi: self.hi }
        } else {
            self
        }
    }
}

/// The proportional axis of a resize: the running axis
/// scaled by the named axis over the running other axis
/// (the engine's `apply_size`), bounded from both ends.
fn proportional(axis: Axis, other: Axis, target: Axis) -> Axis {
    Axis::around(
        axis.lo as f64 * target.lo as f64 / other.hi as f64,
        axis.hi as f64 * target.hi as f64 / other.lo as f64,
    )
}

/// Conservative per-axis bounds on every canvas the engine
/// builds for `ops` from the synthetic source: `None` when
/// some step's canvas could exceed [`CHAIN_MAX_PIXELS`] or
/// the chain's blurs (blur filters, blurred regions, and
/// `pad:...:blur` bars) could pass [`CHAIN_MAX_BLUR_WORK`],
/// else the largest upper bound each axis reaches anywhere
/// in the chain (the output, and the `debug:focal` view,
/// which draws on a canvas the chain passed through, both
/// sit inside it). The walk mirrors the engine's size
/// resolution: DPR-scaled pixel targets, proportional and
/// `orig` axes against the running canvas, the adaptive
/// fit-in box swap, rotated bounding boxes, and trims and
/// `no_upscale` that only ever shrink. The filters whose
/// size it cannot predict without running them (the AI,
/// overlay and text filters) need a provider the default
/// context does not have, so they refuse before touching
/// pixels; it keeps their dimensions, except `upscale`,
/// which it scales anyway.
pub fn canvas_bound(ops: &[Operation]) -> Option<(u64, u64)> {
    let directives = url::RequestDirectives::analyze(ops);
    let dpr = f64::from(directives.dpr.unwrap_or(1.0));
    let cap = directives.no_upscale;
    let mut cur = (
        Axis::exact(u64::from(SOURCE_WIDTH)),
        Axis::exact(u64::from(SOURCE_HEIGHT)),
    );
    let mut peak = (cur.0.hi, cur.1.hi);
    let mut blur_spent: u64 = 0;
    let mut spend = |pixels: u64, sigma: f32| -> Option<()> {
        blur_spent = blur_spent.saturating_add(blur_work(pixels, sigma));
        (blur_spent <= CHAIN_MAX_BLUR_WORK).then_some(())
    };
    let mut step = |cur: (Axis, Axis)| -> Option<(Axis, Axis)> {
        if cur.0.hi.saturating_mul(cur.1.hi) > CHAIN_MAX_PIXELS {
            return None;
        }
        peak = (peak.0.max(cur.0.hi), peak.1.max(cur.1.hi));
        Some(cur)
    };
    for op in ops {
        match op {
            Operation::Size(size) => {
                let w = match (size.width, size.height) {
                    (Dimension::Pixels(n), _) => Axis::pixels(n, dpr),
                    (Dimension::Proportional, Dimension::Pixels(m)) => {
                        proportional(cur.0, cur.1, Axis::pixels(m, dpr))
                    }
                    _ => cur.0,
                };
                let h = match (size.height, size.width) {
                    (Dimension::Pixels(n), _) => Axis::pixels(n, dpr),
                    (Dimension::Proportional, Dimension::Pixels(m)) => {
                        proportional(cur.1, cur.0, Axis::pixels(m, dpr))
                    }
                    _ => cur.1,
                };
                cur = step((w.capped(cap), h.capped(cap)))?;
            }
            // Crop and pad take an exact box; the parser
            // refuses any other spelling.
            Operation::Crop { size, .. } | Operation::Pad { size, .. } => {
                if let (Dimension::Pixels(w), Dimension::Pixels(h)) = (size.width, size.height) {
                    let (bw, bh) = (Axis::pixels(w, dpr), Axis::pixels(h, dpr));
                    if let Operation::Pad {
                        fill: url::PadFill::Blur,
                        ..
                    } = op
                    {
                        // The ambient bars blur the whole box
                        // at a sigma that grows with its longer
                        // side (the engine's `apply_pad`).
                        let sigma = (bw.hi.max(bh.hi) as f32 / 16.0).clamp(2.0, 80.0);
                        spend(bw.hi.saturating_mul(bh.hi), sigma)?;
                    }
                    cur = step((bw.capped(cap), bh.capped(cap)))?;
                }
            }
            Operation::FitIn { size, adaptive } => {
                cur = step(fit_in_bound(cur, size, *adaptive, dpr, cap))?;
            }
            Operation::CropRect {
                left,
                top,
                right,
                bottom,
            } => {
                cur = step((
                    Axis::exact(u64::from(right.saturating_sub(*left))),
                    Axis::exact(u64::from(bottom.saturating_sub(*top))),
                ))?;
            }
            Operation::FocalRect { .. }
            | Operation::FocalPoint { .. }
            | Operation::DebugFocal
            | Operation::Frame(_)
            | Operation::Page(_) => {}
            Operation::Filters(filters) => {
                for filter in filters {
                    cur = match filter {
                        Filter::Rotate(degrees) => {
                            // The rotated bounding box grows with
                            // either side, so the bounds rotate
                            // with their ends.
                            let lo = rotated(cur.0.lo, cur.1.lo, *degrees);
                            let hi = rotated(cur.0.hi, cur.1.hi, *degrees);
                            step((
                                Axis::around(lo.0 as f64, hi.0 as f64),
                                Axis::around(lo.1 as f64, hi.1 as f64),
                            ))?
                        }
                        Filter::Trim(_) | Filter::TrimSides(..) => (
                            Axis {
                                lo: 1,
                                hi: cur.0.hi,
                            },
                            Axis {
                                lo: 1,
                                hi: cur.1.hi,
                            },
                        ),
                        Filter::Blur(sigma) => {
                            spend(cur.0.hi.saturating_mul(cur.1.hi), *sigma)?;
                            cur
                        }
                        // The region blurs at a sigma set by its
                        // shorter side (`filters::blur_region`);
                        // the engine refuses a region outside
                        // the canvas before it blurs.
                        Filter::BlurRegion { w, h, .. } => {
                            let sigma = ((*w).min(*h) as f32 / 8.0).clamp(1.0, 30.0);
                            let area = u64::from(*w).saturating_mul(u64::from(*h));
                            spend(area.min(cur.0.hi.saturating_mul(cur.1.hi)), sigma)?;
                            cur
                        }
                        Filter::Upscale(factor) => {
                            let f = u64::from(*factor);
                            step((
                                Axis {
                                    lo: cur.0.lo.saturating_mul(f),
                                    hi: cur.0.hi.saturating_mul(f),
                                },
                                Axis {
                                    lo: cur.1.lo.saturating_mul(f),
                                    hi: cur.1.hi.saturating_mul(f),
                                },
                            ))?
                        }
                        _ => cur,
                    };
                }
            }
        }
    }
    Some(peak)
}

/// [`filters::rotated_canvas`] on bounds the budget has
/// already capped (each axis is at most
/// [`CHAIN_MAX_PIXELS`], so it fits a `u32`).
fn rotated(w: u64, h: u64, degrees: f32) -> (u32, u32) {
    let fit = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
    filters::rotated_canvas(fit(w), fit(h), degrees)
}

/// The fit-in step of [`canvas_bound`]: the engine scales
/// the canvas by the smallest of the box-over-canvas ratios
/// of the constrained axes (a pixel axis bounds at the box,
/// `orig` at a scale of 1, a proportional axis bounds
/// nothing), after the adaptive swap. Bounding the swap by
/// the larger box side for the upper bound and the smaller
/// for the lower covers both orientations.
fn fit_in_bound(
    cur: (Axis, Axis),
    size: &url::ImageSize,
    adaptive: bool,
    dpr: f64,
    cap: bool,
) -> (Axis, Axis) {
    let both = match (size.width, size.height) {
        (Dimension::Pixels(w), Dimension::Pixels(h)) => {
            Some((Axis::pixels(w, dpr), Axis::pixels(h, dpr)))
        }
        _ => None,
    };
    // (scale lower bound, scale upper bound) for one axis.
    let ratio = |d: Dimension, axis: Axis| -> Option<(f64, f64)> {
        let b = match (d, both) {
            (Dimension::Proportional, _) => return None,
            (Dimension::Original, _) => return Some((1.0, 1.0)),
            (Dimension::Pixels(_), Some((bw, bh))) if adaptive => Axis {
                lo: bw.lo.min(bh.lo),
                hi: bw.hi.max(bh.hi),
            },
            (Dimension::Pixels(n), _) => Axis::pixels(n, dpr),
        };
        Some((b.lo as f64 / axis.hi as f64, b.hi as f64 / axis.lo as f64))
    };
    let ratios = [ratio(size.width, cur.0), ratio(size.height, cur.1)];
    let mut scale: Option<(f64, f64)> = None;
    for (lo, hi) in ratios.into_iter().flatten() {
        scale = Some(match scale {
            Some((l, h)) => (l.min(lo), h.min(hi)),
            None => (lo, hi),
        });
    }
    let Some((lo, hi)) = scale else {
        return cur;
    };
    (
        Axis::around(cur.0.lo as f64 * lo, cur.0.hi as f64 * hi).capped(cap),
        Axis::around(cur.1.lo as f64 * lo, cur.1.hi as f64 * hi).capped(cap),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn operations(path: &str) -> Vec<Operation> {
        url::parse_request(path)
            .expect("the test request parses")
            .2
            .operations
    }

    #[test]
    fn the_filter_leg_stops_before_a_blur_past_the_work_budget() {
        // Four 45-degree turns grow the 24x18 source to about
        // 87x87; blur(512) there would cost about 26 million
        // pixel-taps, past CHAIN_MAX_BLUR_WORK, so the leg
        // ends before it. A small blur on the same canvas,
        // and blur(512) after three turns, still run.
        let turns = "rotate(45):rotate(45):rotate(45):rotate(45)";
        let leg = |filters: &str| {
            check_filter_leg(&operations(&format!(
                "/unsafe/filters:{filters}/example.com/a.png"
            )))
        };
        assert_eq!(leg(&format!("{turns}:blur(512)")), 4);
        assert_eq!(leg(&format!("{turns}:blur(5)")), 5);
        assert_eq!(leg("rotate(45):rotate(45):rotate(45):blur(512)"), 4);
    }
}
