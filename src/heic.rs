//! HEIC/HEIF source decoding (issue #163): the pure-Rust
//! `oxideav-heif` container + `oxideav-h265` codec, compiled in
//! unconditionally — no cargo feature, no configuration, no
//! external tool. Decode speed is what it is (the evaluation's
//! spike measured the stack at roughly 3.7x FFmpeg's wall time on
//! large single-stream photos); Max accepted that trade for a
//! decoder that ships in every artifact.
//!
//! The decoded pixels join the normal pipeline: this module
//! returns the image with the EXIF orientation already applied
//! (from the decoder's EXIF blob, through the engine's own
//! orientation machinery — the container bytes carry no
//! orientation the generic reader can see) plus the
//! [`SourceMeta`] the pipeline's color conversion and metadata
//! preservation consume. Container transforms (`irot`, mirror,
//! crop) are the decoder's own work and are never re-applied.

use image::DynamicImage;

use crate::color::SourceMeta;
use crate::engine::{EngineError, MAX_DECODE_ALLOC, MAX_DECODE_DIMENSION};

/// The decoded image's own axes are bounded through the decoder's
/// limits, set from the same constants the image-crate path uses,
/// so a HEIC source cannot allocate past the shared budget:
/// 16384 px per axis and 256 MiB of RGBA pixels.
fn decode_options() -> oxideav_heif::DecodeOptions {
    oxideav_heif::DecodeOptions::default()
        .with_max_width(Some(MAX_DECODE_DIMENSION))
        .with_max_height(Some(MAX_DECODE_DIMENSION))
        .with_max_pixels(Some(MAX_DECODE_ALLOC / 4))
}

fn decode_failed(cause: impl std::fmt::Display) -> EngineError {
    EngineError::Decode(format!("HEIC/HEIF source: {cause}"))
}

/// Decode a HEIC/HEIF source to an upright image plus its source
/// metadata. Failures are the pipeline's decode class (a 502
/// upstream), naming HEIC, never a panic: the decoder is pure
/// Rust and its error surface on hostile input is errors, not
/// aborts (320 mutants, zero panics, in the issue #163 spike).
pub(crate) fn decode_heic(bytes: &[u8]) -> Result<(DynamicImage, SourceMeta), EngineError> {
    let mut img = oxideav_heif::decode_with(bytes, &decode_options()).map_err(decode_failed)?;
    // The colour the decoder assumed is only a default when the
    // file carries no nclx colr box; the bitstream's own word on
    // the matter (the SPS VUI) wins where it exists.
    if let Some(color) = vui_color(bytes) {
        img.color = color;
    }
    let (width, height) = (img.width, img.height);
    // The decoder's limits are the first line; the decoded size
    // is re-checked against the shared budget in our own code,
    // the same defence in depth decode_image applies.
    crate::engine::check_decoded_dimensions(width as u64, height as u64)?;
    if width as u64 * height as u64 * 4 > MAX_DECODE_ALLOC {
        return Err(decode_failed(format!(
            "image dimensions {width}x{height} exceed the decode allocation limit"
        )));
    }
    let rgba = img.try_to_rgba8().map_err(decode_failed)?;
    let buffer = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(width, height, rgba)
        .ok_or_else(|| decode_failed("decoded planes do not match the decoded size"))?;
    let mut image = DynamicImage::ImageRgba8(buffer);
    // Orientation: the decoder resolves the HEIF offset word and
    // hands back the TIFF stream, which the engine's EXIF reader
    // (the same one JPEG sources go through) reads the tag from.
    if let Some(exif) = &img.metadata.exif
        && let Some(orientation) = crate::engine::exif_orientation(exif)
    {
        image = crate::engine::apply_orientation(image, orientation);
    }
    let mut meta = SourceMeta {
        icc: img.metadata.icc.clone(),
        exif_tiff: img.metadata.exif.clone(),
        xmp: img.metadata.xmp.clone(),
        photoshop: None,
    };
    // The pixels are upright now; a preserved EXIF must say so,
    // exactly as the JPEG/PNG extraction path records it.
    if let Some(tiff) = &mut meta.exif_tiff {
        crate::color::patch_tiff_orientation(tiff);
    }
    Ok((image, meta))
}

/// The colour description the SPS VUI prescribes, when the
/// container leaves the choice to the bitstream.
///
/// oxideav-heif converts with the file's `nclx` `colr` box, or
/// the MIAF default (BT.601 matrix, full range) when there is
/// none. It never consults the SPS VUI, so a file whose only
/// colour description lives in the bitstream decodes washed out
/// next to every other tool: FFmpeg and libheif both fall back
/// to the VUI. This reads it: the primary item's first SPS,
/// parsed with oxideav-h265. An `nclx` box suppresses the
/// override entirely (the file's explicit word wins, byte for
/// byte), and any parse failure along the way leaves the
/// decoder's default in place — the VUI is a correction, never
/// a new failure mode.
fn vui_color(bytes: &[u8]) -> Option<oxideav_heif::ColorInfo> {
    let file = oxideav_heif::HeifFile::parse_borrowed(bytes).ok()?;
    let node = oxideav_heif::build_primary_graph(&file).ok()?;
    // The colour node mirrors the decoder's own choice: a
    // tone-mapped primary describes its base rendition's colour.
    let colour_node = match (&node.kind, node.inputs.first()) {
        (oxideav_heif::ImageKind::ToneMap(_), Some(base)) => base,
        _ => &node,
    };
    if colour_node.properties.nclx().is_some() {
        return None;
    }
    let sps = find_sps(&node)?;
    let signal = sps
        .vui_parameters
        .as_ref()
        .and_then(|vui| vui.video_signal_type.as_ref());
    Some(color_from_signal(signal))
}

/// What a (possibly absent) VUI video signal block means for the
/// conversion: the signalled range and colour description, with
/// two spec inferences filled in. A VUI that omits the colour
/// description leaves the code points unspecified, for which the
/// MIAF default description (BT.709 primaries, sRGB transfer,
/// BT.601 matrix) is the standing assumption; and a stream that
/// signals no video signal type at all infers limited range
/// (H.265 §E.3.1: `video_full_range_flag` is inferred 0), which
/// is the case the whole correction exists for — older encoders
/// wrote limited-range pictures and said nothing anywhere.
fn color_from_signal(
    signal: Option<&oxideav_h265::vui::VideoSignalType>,
) -> oxideav_heif::ColorInfo {
    let (full_range, description) = match signal {
        Some(signal) => (
            signal.video_full_range_flag,
            signal.colour_description.map(|d| {
                (
                    d.colour_primaries,
                    d.transfer_characteristics,
                    d.matrix_coeffs,
                )
            }),
        ),
        None => (false, None),
    };
    let (primaries, transfer, matrix) = description.unwrap_or((1, 13, 6));
    oxideav_heif::ColorInfo::new(
        if full_range {
            oxideav_heif::ColorRange::Full
        } else {
            oxideav_heif::ColorRange::Limited
        },
        primaries,
        transfer,
        matrix,
    )
}

/// The first SPS in the primary item's derivation tree (grid and
/// overlay inputs included: their tiles carry the same parameter
/// sets as the composition they feed).
fn find_sps(node: &oxideav_heif::ImageNode) -> Option<oxideav_h265::sps::SeqParameterSet> {
    if let Some(hvcc) = node.properties.hvcc() {
        for array in &hvcc.arrays {
            if array.nal_unit_type != 33 {
                continue; // not an SPS array
            }
            for nal in &array.nal_units {
                let unescaped = oxideav_h265::nal::strip_emulation_prevention(nal);
                if unescaped.len() > 2
                    && let Ok(sps) = oxideav_h265::sps::SeqParameterSet::parse(&unescaped[2..])
                {
                    return Some(sps);
                }
            }
        }
    }
    for input in &node.inputs {
        if let Some(sps) = find_sps(input) {
            return Some(sps);
        }
    }
    if let Some(alpha) = &node.alpha
        && let Some(sps) = find_sps(alpha)
    {
        return Some(sps);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fixture provenance (tests/fixtures/): every HEIC fixture was
    // generated for this test suite on 2026-10-24 with the oxideav
    // encoder (the crates this module decodes with), from synthetic
    // patterns written by the generator — no third-party images.
    //   grid.heic       512x384 quadrant pattern, 256 px grid tiles
    //   main10.heif     128x96 YUV 4:2:0 10-bit gradient/blocks
    //   vui-limited.heic 320x240, colour only in the SPS VUI
    //                   (limited range, BT.709), no colr box at all
    //   p3.heic         64x64 muted blocks under a Display P3 ICC
    //                   profile generated by moxcms
    //   orient6.heic    64x48 quadrants, EXIF Orientation=6, no irot
    // Reference pixel values below were cross-checked against
    // FFmpeg 8.1.2's decode of the same files (max abs error 1 on
    // the VUI fixture once the correction applies, 24 on one
    // 10-bit gradient edge): the tolerances carry that margin.
    const GRID: &[u8] = include_bytes!("../tests/fixtures/grid.heic");
    const MAIN10: &[u8] = include_bytes!("../tests/fixtures/main10.heif");
    const VUI_LIMITED: &[u8] = include_bytes!("../tests/fixtures/vui-limited.heic");
    const P3: &[u8] = include_bytes!("../tests/fixtures/p3.heic");
    const ORIENT6: &[u8] = include_bytes!("../tests/fixtures/orient6.heic");

    fn pixel(img: &DynamicImage, x: u32, y: u32) -> (u8, u8, u8) {
        let p = image::GenericImageView::get_pixel(img, x, y);
        (p[0], p[1], p[2])
    }

    fn assert_near(actual: (u8, u8, u8), expected: (u8, u8, u8), tol: i32, what: &str) {
        for (a, e) in [actual.0, actual.1, actual.2]
            .iter()
            .zip([expected.0, expected.1, expected.2].iter())
        {
            assert!(
                (*a as i32 - *e as i32).abs() <= tol,
                "{what}: got {actual:?}, expected {expected:?} within {tol}"
            );
        }
    }

    #[test]
    fn grid_fixture_composes_the_full_canvas() {
        let (img, _) = decode_heic(GRID).unwrap();
        assert_eq!((img.width(), img.height()), (512, 384));
        // One spot per tile quadrant: tiles the composition had
        // to place, not one coded picture.
        assert_near(pixel(&img, 128, 96), (230, 29, 29), 4, "top-left tile");
        assert_near(pixel(&img, 384, 96), (30, 199, 30), 4, "top-right tile");
        assert_near(pixel(&img, 128, 288), (30, 29, 230), 4, "bottom-left tile");
        assert_near(pixel(&img, 384, 288), (240, 240, 240), 4, "bottom-right");
    }

    #[test]
    fn ten_bit_fixture_decodes_with_its_depth_intact() {
        let raw = oxideav_heif::decode_with(MAIN10, &decode_options()).unwrap();
        assert_eq!(format!("{:?}", raw.format), "Yuv420P10Le");
        let (img, _) = decode_heic(MAIN10).unwrap();
        assert_eq!((img.width(), img.height()), (128, 96));
        // The luma gradient runs dark to bright left to right.
        let (left, _, _) = pixel(&img, 4, 24);
        let (right, _, _) = pixel(&img, 124, 24);
        assert!(right > left + 40, "gradient: {left} -> {right}");
        assert_near(pixel(&img, 64, 24), (83, 171, 15), 12, "gradient middle");
    }

    #[test]
    fn exif_orientation_rotates_the_output() {
        let (img, meta) = decode_heic(ORIENT6).unwrap();
        // Stored 64x48; Orientation=6 turns it upright at 48x64.
        assert_eq!((img.width(), img.height()), (48, 64));
        // A clockwise quarter turn moves the stored quadrants:
        // BL (blue) -> TL, TL (red) -> TR, BR (white) -> BL,
        // TR (green) -> BR.
        let (r, g, b) = pixel(&img, 8, 8);
        assert!(
            b > r && b > g,
            "top-left should be blue, got {:?}",
            (r, g, b)
        );
        let (r, g, b) = pixel(&img, 40, 8);
        assert!(
            r > g && r > b,
            "top-right should be red, got {:?}",
            (r, g, b)
        );
        let (r, g, b) = pixel(&img, 40, 56);
        assert!(
            g > r && g > b,
            "bottom-right should be green, got {:?}",
            (r, g, b)
        );
        // The preserved EXIF reports the orientation the pixels
        // now have (1), not the one they were stored with.
        let tiff = meta.exif_tiff.expect("EXIF survives");
        assert_eq!(crate::engine::exif_orientation(&tiff), Some(1));
    }

    #[test]
    fn vui_limited_range_matches_the_reference_decode() {
        // Without the correction the same file decodes visibly
        // washed out: the dark block lands ~13 levels high on
        // average against FFmpeg. Pin both sides so the fixture
        // cannot silently stop exercising the gap.
        let plain = oxideav_heif::decode_with(VUI_LIMITED, &decode_options()).unwrap();
        let plain = DynamicImage::ImageRgba8(
            image::ImageBuffer::from_raw(plain.width, plain.height, plain.try_to_rgba8().unwrap())
                .unwrap(),
        );
        let (dark, _, _) = pixel(&plain, 10, 10);
        assert!(
            dark > 48,
            "uncorrected decode should run bright, got {dark}"
        );

        let (img, _) = decode_heic(VUI_LIMITED).unwrap();
        assert_eq!((img.width(), img.height()), (320, 240));
        assert_near(pixel(&img, 10, 10), (39, 39, 39), 3, "dark block");
        assert_near(pixel(&img, 200, 60), (200, 29, 30), 3, "red block");
        assert_near(pixel(&img, 200, 180), (235, 235, 235), 3, "bright block");
    }

    #[test]
    fn signal_inference_follows_the_spec() {
        use oxideav_h265::vui::{ColourDescription, VideoSignalType};
        let signal = |full: bool, desc: Option<(u8, u8, u8)>| VideoSignalType {
            video_format: 5,
            video_full_range_flag: full,
            colour_description: desc.map(|(p, t, m)| ColourDescription {
                colour_primaries: p,
                transfer_characteristics: t,
                matrix_coeffs: m,
            }),
        };
        // No signal at all: limited range is the H.265 inference,
        // the MIAF description the standing assumption. This is
        // the Nokia cheers case from the spike (that file's terms
        // do not allow vendoring it, so the rule is pinned here
        // and the present-VUI path end to end on the fixture).
        let c = color_from_signal(None);
        assert_eq!(c.range, oxideav_heif::ColorRange::Limited);
        assert_eq!((c.primaries, c.transfer, c.matrix), (1, 13, 6));
        // A signalled description is honored whole.
        let c = color_from_signal(Some(&signal(false, Some((9, 16, 9)))));
        assert_eq!(c.range, oxideav_heif::ColorRange::Limited);
        assert_eq!((c.primaries, c.transfer, c.matrix), (9, 16, 9));
        // Full range signalled, description omitted: unspecified
        // code points keep the standing assumption.
        let c = color_from_signal(Some(&signal(true, None)));
        assert_eq!(c.range, oxideav_heif::ColorRange::Full);
        assert_eq!((c.primaries, c.transfer, c.matrix), (1, 13, 6));
    }

    #[test]
    fn a_colr_box_suppresses_the_override() {
        // grid.heic carries an nclx box: its pixels must come out
        // exactly as the bare decoder produces them.
        assert!(vui_color(GRID).is_none());
        assert!(vui_color(P3).is_none());
        let raw = oxideav_heif::decode_with(GRID, &decode_options()).unwrap();
        let raw = raw.try_to_rgba8().unwrap();
        let (img, _) = decode_heic(GRID).unwrap();
        assert_eq!(img.to_rgba8().into_raw(), raw);
    }

    #[test]
    fn the_icc_profile_reaches_the_source_meta() {
        let (_, meta) = decode_heic(P3).unwrap();
        let icc = meta.icc.expect("the P3 profile is extracted");
        assert!(icc.len() > 100 && &icc[36..40] == b"acsp", "ICC header");
        // And the raw decode underneath keeps the stored values:
        // conversion to sRGB is the engine's step, not this one.
        let (img, _) = decode_heic(P3).unwrap();
        assert_near(pixel(&img, 16, 32), (200, 118, 60), 4, "stored P3 values");
    }

    #[test]
    fn the_engine_converts_heic_pixels_to_srgb() {
        // Through the full pipeline the P3 fixture's stored values
        // are reinterpreted by their profile: the served pixel is
        // the sRGB conversion of (200, 118, 60), measurably not
        // the stored triplet.
        let engine = crate::engine::Engine;
        let out = engine
            .process(
                P3,
                &[crate::url::Operation::Filters(vec![
                    crate::url::Filter::Format(crate::url::ImageFormat::Png),
                ])],
            )
            .unwrap();
        let img = image::load_from_memory(&out.bytes).unwrap();
        assert_near(pixel(&img, 16, 32), (213, 115, 44), 6, "P3 red in sRGB");
        assert_near(pixel(&img, 48, 32), (27, 122, 206), 6, "P3 blue in sRGB");
    }

    #[test]
    fn truncated_and_corrupt_files_fail_cleanly() {
        for bytes in [&ORIENT6[..ORIENT6.len() / 2], &ORIENT6[..24], &GRID[..100]] {
            match decode_heic(bytes) {
                Err(EngineError::Decode(msg)) => assert!(msg.contains("HEIC"), "{msg}"),
                other => panic!("expected a HEIC decode error, got {other:?}"),
            }
        }
    }

    #[test]
    fn decode_limits_reject_oversized_declared_dimensions() {
        // Rewrite the ispe box of the orientation fixture: the
        // declared size alone must trip the shared decode limit
        // before any pixel is produced.
        for (w, h) in [(20_000u32, 48u32), (64, 20_000)] {
            let mut bytes = ORIENT6.to_vec();
            let pos = bytes
                .windows(4)
                .position(|w| w == b"ispe")
                .expect("fixture carries ispe");
            // Past the type: version/flags (4), then width, height.
            bytes[pos + 8..pos + 12].copy_from_slice(&w.to_be_bytes());
            bytes[pos + 12..pos + 16].copy_from_slice(&h.to_be_bytes());
            match decode_heic(&bytes) {
                Err(EngineError::Decode(_)) => {}
                other => panic!("{w}x{h}: expected a decode error, got {other:?}"),
            }
        }
    }
}
