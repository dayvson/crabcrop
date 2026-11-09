use super::*;

fn photo(w: u32, h: u32) -> DynamicImage {
    let mut buf = vec![0u8; (w * h * 3) as usize];
    let mut state: u64 = 0x0fed_cba9_8765_4321;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u8
    };
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            buf[i] = ((x * 255) / w.max(1)) as u8 ^ (next() >> 3);
            buf[i + 1] = ((y * 255) / h.max(1)) as u8 ^ (next() >> 3);
            buf[i + 2] = (((x + y) * 255) / (w + h).max(1)) as u8 ^ (next() >> 3);
        }
    }
    DynamicImage::ImageRgb8(image::RgbImage::from_raw(w, h, buf).unwrap())
}

fn bytes(img: &DynamicImage) -> (std::mem::Discriminant<DynamicImage>, Vec<u8>) {
    let raw: Vec<u8> = match img {
        DynamicImage::ImageLuma8(i) => i.as_raw().clone(),
        DynamicImage::ImageLumaA8(i) => i.as_raw().clone(),
        DynamicImage::ImageRgb8(i) => i.as_raw().clone(),
        DynamicImage::ImageRgba8(i) => i.as_raw().clone(),
        other => other.to_rgba8().as_raw().clone(),
    };
    (std::mem::discriminant(img), raw)
}

// Frozen pre-wave-4 reference: the clone-into-map_rgb form.
fn seq_exposure_reference(img: &DynamicImage, stops: f32) -> DynamicImage {
    let lut = exposure_lut(stops);
    map_rgb(img.clone(), |r, g, b| {
        [
            lut[usize::from(r)],
            lut[usize::from(g)],
            lut[usize::from(b)],
        ]
    })
}

// Frozen pre-wave-4 reference: the clone-into-map_rgb form.
fn seq_gamma_reference(img: &DynamicImage, gamma: f32) -> DynamicImage {
    let lut = gamma_lut(gamma);
    map_rgb(img.clone(), |r, g, b| {
        [
            lut[usize::from(r)],
            lut[usize::from(g)],
            lut[usize::from(b)],
        ]
    })
}

fn seq_auto_levels_reference(img: image::RgbImage) -> image::RgbImage {
    let total = (img.width() * img.height()) as u64;
    let mut hists = [[0u64; 256]; 3];
    for p in img.pixels() {
        hists[0][p[0] as usize] += 1;
        hists[1][p[1] as usize] += 1;
        hists[2][p[2] as usize] += 1;
    }
    let mut maps: [[u8; 256]; 3] = [[0; 256]; 3];
    for c in 0..3 {
        let lo = channel_percentile(&hists[c], total, 1, 200);
        let hi = channel_percentile(&hists[c], total, 199, 200);
        let (lo, hi) = if hi.saturating_sub(lo) < AUTO_LEVELS_MIN_SPAN {
            (0usize, 255usize)
        } else {
            (usize::from(lo), usize::from(hi))
        };
        for (v, slot) in maps[c].iter_mut().enumerate() {
            *slot = if v <= lo {
                0
            } else if v >= hi {
                255
            } else {
                ((v - lo) * 255 + (hi - lo) / 2) / (hi - lo)
            } as u8;
        }
    }
    let mut out = img;
    for p in out.pixels_mut() {
        for c in 0..3 {
            p[c] = maps[c][p[c] as usize];
        }
    }
    out
}

#[test]
fn w4_auto_levels_banded_matches_sequential() {
    // The 1200x900 case crosses the banding thresholds; the
    // smaller ones stay sequential. A compressed-range frame
    // exercises the stretch; a flat frame the span floor.
    for (w, h) in [(64u32, 48u32), (333, 257), (1200, 900)] {
        let rgb = photo(w, h).to_rgb8();
        assert_eq!(
            auto_levels(rgb.clone()).as_raw(),
            seq_auto_levels_reference(rgb).as_raw(),
            "auto_levels photo {w}x{h}"
        );
        let mut flat = image::RgbImage::new(w, h);
        for p in flat.pixels_mut() {
            *p = image::Rgb([100, 110, 120]);
        }
        assert_eq!(
            auto_levels(flat.clone()).as_raw(),
            seq_auto_levels_reference(flat).as_raw(),
            "auto_levels flat {w}x{h}"
        );
    }
}

#[test]
fn w4_exposure_owned_matches_cloned() {
    // Sizes straddle the map_rgb_bands thresholds (rows >= 256
    // and buf >= 3 MB) so both the sequential and banded paths
    // are exercised, in RGB and RGBA variants.
    for (w, h) in [(64u32, 48u32), (333, 257), (1200, 900)] {
        let rgb = photo(w, h);
        let rgba = DynamicImage::ImageRgba8(rgb.to_rgba8());
        for stops in [-2.0f32, -0.7, 0.0, 1.0, 3.5] {
            assert_eq!(
                bytes(&exposure_correct(rgb.clone(), stops)),
                bytes(&seq_exposure_reference(&rgb, stops)),
                "exposure {stops} rgb {w}x{h}"
            );
            assert_eq!(
                bytes(&exposure_correct(rgba.clone(), stops)),
                bytes(&seq_exposure_reference(&rgba, stops)),
                "exposure {stops} rgba {w}x{h}"
            );
        }
    }
}

#[test]
fn w4_gamma_owned_matches_cloned() {
    // Same threshold-straddling sizes as the exposure proof,
    // in RGB and RGBA variants.
    for (w, h) in [(64u32, 48u32), (333, 257), (1200, 900)] {
        let rgb = photo(w, h);
        let rgba = DynamicImage::ImageRgba8(rgb.to_rgba8());
        for gamma in [0.5f32, 1.0, 2.2, 4.0] {
            assert_eq!(
                bytes(&gamma_correct(rgb.clone(), gamma)),
                bytes(&seq_gamma_reference(&rgb, gamma)),
                "gamma {gamma} rgb {w}x{h}"
            );
            assert_eq!(
                bytes(&gamma_correct(rgba.clone(), gamma)),
                bytes(&seq_gamma_reference(&rgba, gamma)),
                "gamma {gamma} rgba {w}x{h}"
            );
        }
    }
}
