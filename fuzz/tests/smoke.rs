//! The deterministic fuzz smoke (issue #146): every seed in
//! the corpus, every truncation of the representative seeds,
//! single-byte mutations of grammar-rich seeds, and a short
//! fixed-seed random mutation pass over the chain seeds, all
//! run through the same invariant harness as the cargo-fuzz
//! targets. Bounded and fast, so it belongs on the local
//! checklist (scripts/test_all.sh); the open-ended fuzzing
//! itself runs under cargo-fuzz on a nightly toolchain, every
//! night in .github/workflows/fuzz.yml (see fuzz/README.md).

use std::collections::BTreeSet;
use std::path::PathBuf;

use crabcrop_fuzz::ChainOutcome;
use crabcrop_fuzz::engine::EngineError;
use crabcrop_fuzz::url::{self, Align, Filter, Operation};

fn corpus_inputs() -> Vec<String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/parse_request");
    let mut inputs: Vec<String> = std::fs::read_dir(&dir)
        .expect("the seed corpus directory exists")
        .filter_map(|e| e.ok())
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .collect();
    inputs.sort();
    assert!(inputs.len() >= 25, "the corpus should stay seeded");
    inputs
}

#[test]
fn seeds_satisfy_the_invariants() {
    for input in corpus_inputs() {
        crabcrop_fuzz::check(&input);
    }
}

/// The family an operation belongs to, for the seed
/// coverage check below. The match is exhaustive on
/// purpose: a new operation, alignment or filter fails to
/// compile here until it has a family name, a spot in
/// `FAMILIES`, and a parse_request seed (issue #476).
fn op_families(op: &Operation, out: &mut BTreeSet<&'static str>) {
    let family = match op {
        Operation::Size(_) => "size",
        Operation::Crop { align, .. } => match align {
            Align::Smart => "crop:smart",
            Align::Face => "crop:face",
            Align::Objects { .. } => "crop:objects",
            Align::Center
            | Align::Top
            | Align::Bottom
            | Align::Left
            | Align::Right
            | Align::TopLeft
            | Align::TopRight
            | Align::BottomLeft
            | Align::BottomRight => "crop",
        },
        Operation::FitIn { .. } => "fit-in",
        Operation::Pad { .. } => "pad",
        Operation::CropRect { .. } => "crop-rect",
        Operation::FocalRect { .. } => "focal",
        Operation::FocalPoint { .. } => "focalpoint",
        Operation::DebugFocal => "debug:focal",
        Operation::Frame(_) => "frame",
        Operation::Page(_) => "page",
        Operation::Filters(filters) => {
            for filter in filters {
                out.insert(filter_family(filter));
            }
            "filters"
        }
    };
    out.insert(family);
}

fn filter_family(filter: &Filter) -> &'static str {
    match filter {
        Filter::Grayscale => "grayscale",
        Filter::Invert => "invert",
        Filter::Blur(_) => "blur",
        Filter::Brighten(_) => "brighten",
        Filter::Contrast(_) => "contrast",
        Filter::Gamma(_) => "gamma",
        Filter::Exposure(_) => "exposure",
        Filter::HighlightsShadows(..) => "highlights_shadows",
        Filter::Vibrance(_) => "vibrance",
        Filter::HueRotate(_) => "huerotate",
        Filter::Sharpen(_) => "sharpen",
        Filter::Saturation(_) => "saturation",
        Filter::Rgb(..) => "rgb",
        Filter::Curve(..) => "curve",
        Filter::Convolution(..) => "convolution",
        Filter::Noise(_) => "noise",
        Filter::Equalize => "equalize",
        Filter::Enhance => "enhance",
        Filter::Rotate(_) => "rotate",
        Filter::Trim(_) => "trim",
        Filter::TrimSides(..) => "trim(sides)",
        Filter::Watermark { .. } => "watermark",
        Filter::Overlay { .. } => "overlay",
        Filter::RemoveBg => "removebg",
        Filter::Studio(_) => "studio",
        Filter::KeepProfile => "keep_profile",
        Filter::BlurFaces => "blurfaces",
        Filter::PixelateFaces => "pixelatefaces",
        Filter::BlurClass(_) => "blurclass",
        Filter::PixelateClass(_) => "pixelateclass",
        Filter::RecolorClass(..) => "recolorclass",
        Filter::FacePad(_) => "facepad",
        Filter::Upscale(_) => "upscale",
        Filter::Dpr(_) => "dpr",
        Filter::Text { .. } => "text",
        Filter::Fill(_) => "fill",
        Filter::Radius(_) => "radius",
        Filter::Border(..) => "border",
        Filter::Mask(_) => "mask",
        Filter::Sepia => "sepia",
        Filter::Duotone(..) => "duotone",
        Filter::Monochrome(_) => "monochrome",
        Filter::Pixelate(_) => "pixelate",
        Filter::Flip(_) => "flip",
        Filter::BlurRegion { .. } => "blur_region",
        Filter::PixelateRegion { .. } => "pixelate_region",
        Filter::RemoveRegion { .. } => "remove_region",
        Filter::MaxBytes(_) => "max_bytes",
        Filter::Quality(_) => "quality",
        Filter::Palette(_) => "palette",
        Filter::Format(_) => "format",
        Filter::NoUpscale => "no_upscale",
    }
}

/// Every family the grammar has: the operations (crop by
/// alignment kind), every filter, and the per-request
/// segments that are not operations (`exp:`, `maxage:`,
/// `preset:`, `root:`).
const FAMILIES: &[&str] = &[
    "size",
    "crop",
    "crop:smart",
    "crop:face",
    "crop:objects",
    "fit-in",
    "pad",
    "crop-rect",
    "focal",
    "focalpoint",
    "debug:focal",
    "frame",
    "page",
    "filters",
    "grayscale",
    "invert",
    "blur",
    "brighten",
    "contrast",
    "gamma",
    "exposure",
    "highlights_shadows",
    "vibrance",
    "huerotate",
    "sharpen",
    "saturation",
    "rgb",
    "curve",
    "convolution",
    "noise",
    "equalize",
    "enhance",
    "rotate",
    "trim",
    "trim(sides)",
    "watermark",
    "overlay",
    "removebg",
    "studio",
    "keep_profile",
    "blurfaces",
    "pixelatefaces",
    "blurclass",
    "pixelateclass",
    "recolorclass",
    "facepad",
    "upscale",
    "dpr",
    "text",
    "fill",
    "radius",
    "border",
    "mask",
    "sepia",
    "duotone",
    "monochrome",
    "pixelate",
    "flip",
    "blur_region",
    "pixelate_region",
    "remove_region",
    "max_bytes",
    "quality",
    "palette",
    "format",
    "no_upscale",
    "exp:",
    "maxage:",
    "preset:",
    "root:",
];

#[test]
fn parse_request_seeds_cover_every_operation_family() {
    // Issue #476: the parser fuzzer starts from seeds, and a
    // family no seed reaches waits on the fuzzer to spell it
    // from scratch. Every family needs a seed that parses
    // (after preset expansion, the way the server parses).
    let mut seen: BTreeSet<&'static str> = BTreeSet::new();
    for input in corpus_inputs() {
        let Ok((_, _, rest)) = url::parse_signature(&input) else {
            continue;
        };
        let Ok(expanded) = url::expand_presets(rest, crabcrop_fuzz::presets(), false) else {
            continue;
        };
        let Ok((ops, source, expiry, max_age)) = url::parse_operations_with_expiry(&expanded)
        else {
            continue;
        };
        for op in &ops {
            op_families(op, &mut seen);
        }
        if expiry.is_some() {
            seen.insert("exp:");
        }
        if max_age.is_some() {
            seen.insert("maxage:");
        }
        if !url::preset_names(rest).is_empty() {
            seen.insert("preset:");
        }
        if source.starts_with("root:") {
            seen.insert("root:");
        }
    }
    let missing: Vec<&str> = FAMILIES
        .iter()
        .copied()
        .filter(|f| !seen.contains(f))
        .collect();
    assert!(
        missing.is_empty(),
        "no parse_request seed parses into these families: {missing:?}"
    );
    let unlisted: Vec<&str> = seen
        .iter()
        .copied()
        .filter(|f| !FAMILIES.contains(f))
        .collect();
    assert!(
        unlisted.is_empty(),
        "families missing from FAMILIES: {unlisted:?}"
    );
}

#[test]
fn the_preset_map_holds_definitions_the_server_accepts() {
    // The stand-in map is only useful if expansion splices
    // real operations: each definition passes the boot-time
    // validation, and a request naming one expands and parses.
    for (name, definition) in crabcrop_fuzz::presets() {
        let definition = definition.as_str();
        assert!(
            url::parse_preset_operations(definition).is_ok(),
            "preset {name} = {definition} would be refused at boot"
        );
        let request = format!("/unsafe/preset:{name}/example.com/img.jpg");
        let (_, _, rest) = url::parse_signature(&request).expect("the request parses");
        let expanded = url::expand_presets(rest, crabcrop_fuzz::presets(), true)
            .expect("a known preset expands");
        assert!(
            url::parse_operations(&expanded).is_ok(),
            "{name} expands to {expanded}, which does not parse"
        );
        crabcrop_fuzz::check(&request);
    }
}

#[test]
fn repeated_references_satisfy_the_counted_expansion_invariants() {
    // Expansion counts before it builds. Around and far past
    // the operation cap, by references alone, interleaved with
    // free-form operations or metadata, and followed by every
    // kind of tail (a source, none, an empty segment, a
    // malformed operation, an unknown name, a bad expiry, a
    // source holding `preset:` text), its answer must match
    // splicing every reference and parsing the result.
    let tails = [
        "example.com/img.jpg",
        "",
        "/example.com/img.jpg",
        "filters:bogus/x.jpg",
        "preset:nope/x.jpg",
        "exp:x/example.com/img.jpg",
        "example.com/preset:thumb/img.jpg",
    ];
    for name in crabcrop_fuzz::presets().keys() {
        for unit in [
            format!("preset:{name}/"),
            format!("preset:{name}/filters:invert/"),
            format!("preset:{name}/exp:9/maxage:60/"),
        ] {
            for refs in [1, 33, 34, 50, 51, 99, 100, 101, 400] {
                let ops = unit.repeat(refs);
                for tail in tails {
                    crabcrop_fuzz::check(&format!("/unsafe/{ops}{tail}"));
                }
            }
        }
    }
}

#[test]
fn truncations_and_mutations_satisfy_the_invariants() {
    let rich = "/unsafe/filters:rotate(45.5):quality(80):format(webp)/300x200/example.com/img.jpg";
    // Every truncation.
    for end in 0..=rich.len() {
        if rich.is_char_boundary(end) {
            crabcrop_fuzz::check(&rich[..end]);
        }
    }
    // Single-byte substitutions with grammar-significant
    // characters.
    let mut bytes = rich.as_bytes().to_vec();
    for i in 0..bytes.len() {
        for &replacement in b"/:()%0x-." {
            let original = bytes[i];
            bytes[i] = replacement;
            if let Ok(text) = std::str::from_utf8(&bytes) {
                crabcrop_fuzz::check(text);
            }
            bytes[i] = original;
        }
    }
}

fn byte_corpus_inputs(name: &str) -> Vec<Vec<u8>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("corpus/{name}"));
    let mut inputs: Vec<Vec<u8>> = std::fs::read_dir(&dir)
        .expect("the byte seed corpus directory exists")
        .filter_map(|e| e.ok())
        .filter_map(|e| std::fs::read(e.path()).ok())
        .collect();
    inputs.sort();
    assert!(inputs.len() >= 4, "the {name} corpus should stay seeded");
    inputs
}

#[test]
fn decode_seeds_satisfy_the_invariants() {
    for input in byte_corpus_inputs("decode") {
        crabcrop_fuzz::check_decode(&input);
    }
}

#[test]
fn decode_truncations_and_mutations_satisfy_the_invariants() {
    // The smallest rich seed: a real PNG with header,
    // chunks, and pixel data, so truncations land inside
    // every structure the decoder walks.
    let seed = byte_corpus_inputs("decode")
        .into_iter()
        .find(|s| s.starts_with(b"\x89PNG"))
        .expect("a PNG seed");
    for end in 0..=seed.len() {
        crabcrop_fuzz::check_decode(&seed[..end]);
    }
    let mut bytes = seed.clone();
    for i in 0..bytes.len() {
        for replacement in [0x00u8, 0xff, 0x89, b'P', b'I', b't'] {
            let original = bytes[i];
            bytes[i] = replacement;
            crabcrop_fuzz::check_decode(&bytes);
            bytes[i] = original;
        }
    }
}

#[test]
fn source_meta_seeds_satisfy_the_invariants() {
    for input in byte_corpus_inputs("source_meta") {
        crabcrop_fuzz::check_meta(&input);
    }
}

#[test]
fn source_meta_truncations_and_mutations_satisfy_the_invariants() {
    // The EXIF seed's front: the metadata segments live
    // there, so truncations land inside the JPEG segments
    // and the TIFF stream the patcher walks. The sweep
    // works on the front 8 KB so it stays fast; the full
    // seeds run whole in the seeds test above.
    let seed: Vec<u8> = byte_corpus_inputs("source_meta")
        .into_iter()
        .find(|s| s.starts_with(b"\xff\xd8"))
        .expect("a JPEG seed")
        .into_iter()
        .take(8192)
        .collect();
    for end in 0..=seed.len() {
        crabcrop_fuzz::check_meta(&seed[..end]);
    }
    let mut bytes = seed.clone();
    for i in 0..bytes.len().min(4096) {
        for replacement in [0x00u8, 0xff, 0xe1, b'E', b'x'] {
            let original = bytes[i];
            bytes[i] = replacement;
            crabcrop_fuzz::check_meta(&bytes);
            bytes[i] = original;
        }
    }
}

fn op_chain_corpus_inputs() -> Vec<String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/op_chain");
    let mut inputs: Vec<String> = std::fs::read_dir(&dir)
        .expect("the chain seed corpus directory exists")
        .filter_map(|e| e.ok())
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .collect();
    inputs.sort();
    assert!(inputs.len() >= 9, "the chain corpus should stay seeded");
    inputs
}

#[test]
fn op_chain_seeds_satisfy_the_invariants() {
    let mut rendered = 0;
    for input in op_chain_corpus_inputs() {
        // A seed that does not parse never reaches a filter,
        // so it seeds nothing (issue #415: a misspelled
        // `convolve` seed sat in the corpus unparsed).
        assert!(
            crabcrop_fuzz::url::parse_request(&input).is_ok(),
            "op_chain seed does not parse: {input}"
        );
        if let ChainOutcome::Rendered { .. } = crabcrop_fuzz::check_chain(&input) {
            rendered += 1;
        }
    }
    // Issue #476: the corpus drives the engine, not only the
    // filter leg, so most seeds render.
    assert!(
        rendered >= 10,
        "only {rendered} op_chain seeds rendered through the engine"
    );
}

#[test]
fn op_chain_drives_resize_and_crop_through_the_engine() {
    // Issue #476: check_chain used to apply only the filters,
    // so a size, crop, fit-in, pad or crop-rect never ran.
    // Each request here renders at a size only the engine's
    // own geometry produces from the 24x18 synthetic source:
    // the filter leg alone would leave it 24x18 (or 18x24).
    for (input, size) in [
        // Center crop to 1:3 (src/crop.rs), then resize.
        ("/unsafe/crop:10x30/example.com/a.png", (10, 30)),
        // Saliency smart crop, positioned on a focal rectangle.
        (
            "/unsafe/focal:0x0:6x6/crop:12x6:smart/example.com/a.png",
            (12, 6),
        ),
        // Focal point with zoom.
        (
            "/unsafe/focalpoint:0.2x0.8:2/crop:8x8/example.com/a.png",
            (8, 8),
        ),
        // Directional crop under DPR 2.
        (
            "/unsafe/crop:6x12:top-left/filters:dpr(2)/example.com/a.png",
            (12, 24),
        ),
        // Proportional resize, then a quarter turn.
        (
            "/unsafe/40x0/filters:rotate(90)/example.com/a.png",
            (30, 40),
        ),
        // Flipped proportional resize.
        ("/unsafe/-30x0/example.com/a.png", (30, 22)),
        // Fit inside a box.
        ("/unsafe/fit-in:20x20/example.com/a.png", (20, 15)),
        // Adaptive fit-in: a landscape source into a portrait
        // box fits the box's transpose (20x10).
        ("/unsafe/fit-in:10x20:adaptive/example.com/a.png", (13, 10)),
        // Pad to an exact box.
        ("/unsafe/pad:30x30:ff0000/example.com/a.png", (30, 30)),
        // Crop-rect, fused with the resize after it.
        ("/unsafe/crop-rect:2x3:14x9/6x0/example.com/a.png", (6, 3)),
        // no_upscale caps the resize at the source.
        (
            "/unsafe/100x0/filters:no_upscale/example.com/a.png",
            (24, 18),
        ),
        // An encode to another format.
        (
            "/unsafe/16x16/filters:format(webp):quality(60)/example.com/a.png",
            (16, 16),
        ),
    ] {
        assert_eq!(
            crabcrop_fuzz::check_chain(input),
            ChainOutcome::Rendered {
                width: size.0,
                height: size.1
            },
            "{input}"
        );
        let ops = url::parse_request(input).unwrap().2.operations;
        let bound = crabcrop_fuzz::canvas_bound(&ops).expect("inside the budget");
        assert!(
            bound.0 >= size.0 as u64 && bound.1 >= size.1 as u64,
            "{input}: bound {bound:?} is under the render {size:?}"
        );
    }
}

#[test]
fn op_chain_skips_the_engine_past_the_budget() {
    // A request whose canvas could pass the harness budget,
    // or whose blurs could pass the blur work budget, never
    // reaches the engine: the harness, not the engine's
    // 100-megapixel budget, bounds a fuzz run's allocations,
    // and a sanitizer build's blur time stays far below the
    // fuzzer's hang timeout.
    for input in [
        "/unsafe/16383x16383/example.com/a.png",
        "/unsafe/600x600/example.com/a.png",
        "/unsafe/crop:600x600/filters:grayscale/example.com/a.png",
        "/unsafe/pad:600x500/example.com/a.png",
        "/unsafe/500x400/filters:rotate(45)/example.com/a.png",
        "/unsafe/300x0/filters:dpr(3)/example.com/a.png",
        "/unsafe/200x200/filters:upscale(4)/example.com/a.png",
        "/unsafe/fit-in:4000x4000/example.com/a.png",
        // Proportional axes resolve against the running
        // canvas: 1 pixel tall, then 2000 tall.
        "/unsafe/400x1/0x2000/example.com/a.png",
        // After a trim the canvas could be 1 pixel tall.
        "/unsafe/filters:trim(5)/400x0/example.com/a.png",
        // Inside the pixel budget, past the blur work budget:
        // a large-sigma blur, the ambient bars of a large
        // pad, and a large blurred region.
        "/unsafe/500x500/filters:blur(100)/example.com/a.png",
        "/unsafe/pad:400x400:blur/example.com/a.png",
        "/unsafe/400x400/filters:blur_region(0,0,400,400)/example.com/a.png",
    ] {
        assert_eq!(
            crabcrop_fuzz::check_chain(input),
            ChainOutcome::OverBudget,
            "{input}"
        );
    }
    // The blur budget leaves small blurs to the engine: the
    // largest sigma on the source, and blurred pad bars on a
    // small box.
    for (input, size) in [
        ("/unsafe/filters:blur(512)/example.com/a.png", (24, 18)),
        ("/unsafe/pad:100x100:blur/example.com/a.png", (100, 100)),
    ] {
        assert_eq!(
            crabcrop_fuzz::check_chain(input),
            ChainOutcome::Rendered {
                width: size.0,
                height: size.1
            },
            "{input}"
        );
    }
}

#[test]
fn op_chain_engine_refusals_are_errors_not_panics() {
    // The default context has no providers, extractor or
    // renderer: those operations refuse, and so does
    // geometry the source cannot satisfy.
    for (input, expect) in [
        (
            "/unsafe/filters:removebg/example.com/a.png",
            "not configured",
        ),
        (
            "/unsafe/filters:studio(product)/example.com/a.png",
            "not configured",
        ),
        ("/unsafe/frame:poster/example.com/a.mp4", "not configured"),
        ("/unsafe/page:2/example.com/a.pdf", "not configured"),
        (
            "/unsafe/crop-rect:0x0:100x100/example.com/a.png",
            "unsupported",
        ),
        (
            "/unsafe/filters:format(mp4)/example.com/a.png",
            "not configured",
        ),
        (
            "/unsafe/filters:watermark(https%3A%2F%2Fexample.com%2Fw.png,0,0,50)/example.com/a.png",
            "unsupported",
        ),
    ] {
        match crabcrop_fuzz::check_chain(input) {
            ChainOutcome::Refused(error) => {
                let kind = match error {
                    EngineError::NotConfigured(_) => "not configured",
                    EngineError::Unsupported(_) => "unsupported",
                    other => panic!("{input}: unexpected refusal {other:?}"),
                };
                assert_eq!(kind, expect, "{input}");
            }
            other => panic!("{input}: expected a refusal, got {other:?}"),
        }
    }
}

#[test]
#[should_panic(expected = "outside 1..=")]
fn render_size_check_rejects_a_zero_axis() {
    crabcrop_fuzz::check_render_size(0, 10, (100, 100));
}

#[test]
#[should_panic(expected = "outside 1..=")]
fn render_size_check_rejects_an_axis_past_max_dimension() {
    crabcrop_fuzz::check_render_size(url::MAX_DIMENSION + 1, 1, (u64::MAX, u64::MAX));
}

#[test]
#[should_panic(expected = "past the harness bound")]
fn render_size_check_rejects_a_render_past_the_bound() {
    crabcrop_fuzz::check_render_size(30, 10, (29, 10));
}

#[test]
#[cfg(feature = "avif-decode")]
fn the_decode_corpus_reaches_the_shipped_avif_decoder() {
    // Issue #476: with the avif-decode feature (the fuzz
    // crate's default, as in every shipped build) the AVIF
    // seed decodes through dav1d, so the decode target
    // fuzzes the decoder the server runs.
    let seed = byte_corpus_inputs("decode")
        .into_iter()
        .find(|s| s.get(4..12) == Some(b"ftypavif".as_slice()))
        .expect("an AVIF seed");
    let img = crabcrop_fuzz::engine::decode_image(&seed).expect("the AVIF seed decodes");
    assert!(img.width() > 0 && img.height() > 0);
    crabcrop_fuzz::check_decode(&seed);
}

#[test]
fn op_chain_mutations_satisfy_the_invariants() {
    // Single-byte substitutions over every seed, digits
    // included, so the one-byte neighbors a real fuzzer
    // tries first (`rotate(90)` to `rotate(91)`) run here.
    for seed in op_chain_corpus_inputs() {
        let mut bytes = seed.into_bytes();
        for i in 0..bytes.len() {
            for &replacement in b"/:(),.-0159" {
                let original = bytes[i];
                bytes[i] = replacement;
                if let Ok(text) = std::str::from_utf8(&bytes) {
                    crabcrop_fuzz::check_chain(text);
                }
                bytes[i] = original;
            }
        }
    }
}

#[test]
fn arbitrary_angle_rotations_satisfy_the_chain_invariants() {
    // Issue #415: the harness asserted that a filter chain
    // keeps the canvas or swaps it, which no arbitrary
    // angle does; rotate(91) and rotate(45) stopped the
    // first real fuzz run on a non-bug.
    for chain in [
        "rotate(91)",
        "rotate(45)",
        "rotate(-30.25)",
        "rotate(0.5)",
        "rotate(359.9)",
        "rotate(-113.0128)",
        "rotate(45):rotate(45):rotate(45)",
        "rotate(45):trim(0):rotate(-91)",
        "border(4,ffffff):trim(0)",
        "fill(102030):rotate(30):grayscale:rotate(90)",
    ] {
        let input = format!("/unsafe/filters:{chain}/example.com/a.jpg");
        assert!(
            crabcrop_fuzz::url::parse_request(&input).is_ok(),
            "{input} should parse"
        );
        crabcrop_fuzz::check_chain(&input);
    }
}

#[test]
fn op_chain_random_mutations_satisfy_the_invariants() {
    // A short, deterministic stand-in for `cargo fuzz run
    // op_chain` from the seeds: a fixed-seed generator
    // substitutes, inserts, and deletes bytes drawn from
    // the grammar's alphabet, a few edits per input, so a
    // false invariant surfaces here without a nightly
    // toolchain. Fixed seed, so a failure reproduces.
    const ALPHABET: &[u8] = b"/:(),.-0123456789abcdefrotaimxwh";
    let seeds = op_chain_corpus_inputs();
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = move |bound: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % bound as u64) as usize
    };
    for _ in 0..4000 {
        let mut bytes = seeds[next(seeds.len())].clone().into_bytes();
        for _ in 0..1 + next(4) {
            let at = next(bytes.len() + 1);
            let byte = ALPHABET[next(ALPHABET.len())];
            match next(3) {
                0 if at < bytes.len() => bytes[at] = byte,
                1 if at < bytes.len() => {
                    bytes.remove(at);
                }
                _ => bytes.insert(at, byte),
            }
        }
        if let Ok(text) = std::str::from_utf8(&bytes) {
            crabcrop_fuzz::check_chain(text);
        }
    }
}

fn seal_corpus_inputs() -> Vec<String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/unseal");
    let mut inputs: Vec<String> = std::fs::read_dir(&dir)
        .expect("the seal seed corpus directory exists")
        .filter_map(|e| e.ok())
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .collect();
    inputs.sort();
    assert!(inputs.len() >= 8, "the seal corpus should stay seeded");
    inputs
}

#[test]
fn seal_seeds_satisfy_the_invariants() {
    for input in seal_corpus_inputs() {
        crabcrop_fuzz::check_seal(input.as_bytes());
    }
}

#[test]
fn seal_truncations_and_mutations_satisfy_the_invariants() {
    let remainder = "filters:rotate(45.5):quality(80):format(webp)/300x200/example.com/img.jpg";
    crabcrop_fuzz::check_seal(remainder.as_bytes());
    let token = crabcrop_fuzz::sample_token(remainder);
    for end in 0..=token.len() {
        if token.is_char_boundary(end) {
            crabcrop_fuzz::check_seal(&token.as_bytes()[..end]);
        }
    }
    let mut bytes = token.into_bytes();
    for i in 0..bytes.len() {
        for &replacement in b".:A0_-/%" {
            let original = bytes[i];
            bytes[i] = replacement;
            if let Ok(text) = std::str::from_utf8(&bytes) {
                crabcrop_fuzz::check_seal(text.as_bytes());
            }
            bytes[i] = original;
        }
    }
}
