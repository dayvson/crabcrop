//! ThumbHash encoding, implemented in-tree (issue #60): the
//! published crate is dormant, and the `fast-thumbhash`
//! crate's encoder output diverges from the reference
//! implementation on most test fixtures, including a solid
//! color. This is Evan Wallace's published algorithm,
//! ported operation for operation so the hashes are
//! byte-identical to the reference encoder; the test
//! vectors below were produced by that reference encoder
//! over the same fixture pixels.
//!
//! A ThumbHash is a compact string (base64 of ~20-25 bytes)
//! that client libraries decode into a blurred preview with
//! the image's approximate aspect ratio and average color.

use image::{DynamicImage, imageops::FilterType};

/// The largest input the encoder accepts, per axis.
const MAX_INPUT: u32 = 100;

/// The ThumbHash of a processed image, base64-encoded: the
/// image is downscaled to fit within 100x100 first (the
/// encoder's input contract), with the pipeline's usual
/// high-quality resampler.
pub fn thumbhash_base64(img: &DynamicImage) -> String {
    let resized;
    let img = if img.width() > MAX_INPUT || img.height() > MAX_INPUT {
        resized = img.resize(MAX_INPUT, MAX_INPUT, FilterType::Lanczos3);
        &resized
    } else {
        img
    };
    let rgba = img.to_rgba8();
    let hash = encode_rgba(rgba.width(), rgba.height(), rgba.as_raw());
    crate::placeholder::base64_encode(&hash)
}

/// Encode raw RGBA pixels (not premultiplied) to a ThumbHash.
/// Both dimensions must be at most 100.
pub fn encode_rgba(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    assert!(
        w <= MAX_INPUT && h <= MAX_INPUT,
        "thumbhash input {w}x{h} exceeds 100x100"
    );
    assert_eq!(rgba.len(), (w * h * 4) as usize, "rgba length mismatch");
    let (w, h) = (w as usize, h as usize);

    // The average color, weighted by alpha.
    let (mut avg_r, mut avg_g, mut avg_b, mut avg_a) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..w * h {
        let alpha = rgba[i * 4 + 3] as f64 / 255.0;
        avg_r += alpha / 255.0 * rgba[i * 4] as f64;
        avg_g += alpha / 255.0 * rgba[i * 4 + 1] as f64;
        avg_b += alpha / 255.0 * rgba[i * 4 + 2] as f64;
        avg_a += alpha;
    }
    if avg_a > 0.0 {
        avg_r /= avg_a;
        avg_g /= avg_a;
        avg_b /= avg_a;
    }

    let has_alpha = avg_a < (w * h) as f64;
    let l_limit = if has_alpha { 5.0 } else { 7.0 };
    let max_dim = w.max(h) as f64;
    let lx = ((l_limit * w as f64 / max_dim).round() as usize).max(1);
    let ly = ((l_limit * h as f64 / max_dim).round() as usize).max(1);

    // Convert to LPQA, composited atop the average color.
    let (mut l, mut p, mut q, mut a) = (
        vec![0.0f64; w * h],
        vec![0.0f64; w * h],
        vec![0.0f64; w * h],
        vec![0.0f64; w * h],
    );
    for i in 0..w * h {
        let alpha = rgba[i * 4 + 3] as f64 / 255.0;
        let r = avg_r * (1.0 - alpha) + alpha / 255.0 * rgba[i * 4] as f64;
        let g = avg_g * (1.0 - alpha) + alpha / 255.0 * rgba[i * 4 + 1] as f64;
        let b = avg_b * (1.0 - alpha) + alpha / 255.0 * rgba[i * 4 + 2] as f64;
        l[i] = (r + g + b) / 3.0;
        p[i] = (r + g) / 2.0 - b;
        q[i] = r - g;
        a[i] = alpha;
    }

    // DCT per channel into a DC term and normalized AC terms.
    fn encode_channel(
        channel: &[f64],
        w: usize,
        h: usize,
        nx: usize,
        ny: usize,
    ) -> (f64, Vec<f64>, f64) {
        let pi = std::f64::consts::PI;
        let (mut dc, mut ac, mut scale) = (0.0f64, Vec::new(), 0.0f64);
        for cy in 0..ny {
            let mut cx = 0usize;
            while cx * ny < nx * (ny - cy) {
                let fx: Vec<f64> = (0..w)
                    .map(|x| (pi / w as f64 * cx as f64 * (x as f64 + 0.5)).cos())
                    .collect();
                let mut f = 0.0f64;
                for y in 0..h {
                    let fy = (pi / h as f64 * cy as f64 * (y as f64 + 0.5)).cos();
                    for x in 0..w {
                        f += channel[x + y * w] * fx[x] * fy;
                    }
                }
                f /= (w * h) as f64;
                if cx == 0 && cy == 0 {
                    dc = f;
                } else {
                    ac.push(f);
                    scale = scale.max(f.abs());
                }
                cx += 1;
            }
        }
        if scale > 0.0 {
            for f in ac.iter_mut() {
                *f = 0.5 + 0.5 / scale * *f;
            }
        }
        (dc, ac, scale)
    }

    let (l_dc, l_ac, l_scale) = encode_channel(&l, w, h, lx.max(3), ly.max(3));
    let (p_dc, p_ac, p_scale) = encode_channel(&p, w, h, 3, 3);
    let (q_dc, q_ac, q_scale) = encode_channel(&q, w, h, 3, 3);
    let (a_dc, a_ac, a_scale) = if has_alpha {
        encode_channel(&a, w, h, 5, 5)
    } else {
        (0.0, Vec::new(), 0.0)
    };
    // Pack the headers.
    let is_landscape = w > h;
    let header24 = (63.0 * l_dc).round() as u32
        | ((31.5 + 31.5 * p_dc).round() as u32) << 6
        | ((31.5 + 31.5 * q_dc).round() as u32) << 12
        | ((31.0 * l_scale).round() as u32) << 18
        | (has_alpha as u32) << 23;
    let header16 = (if is_landscape { ly } else { lx }) as u32
        | ((63.0 * p_scale).round() as u32) << 3
        | ((63.0 * q_scale).round() as u32) << 9
        | (is_landscape as u32) << 15;
    let ac_start = if has_alpha { 6 } else { 5 };
    let total_ac = l_ac.len() + p_ac.len() + q_ac.len() + a_ac.len();
    let mut hash = vec![0u8; ac_start + total_ac.div_ceil(2)];
    hash[0] = (header24 & 255) as u8;
    hash[1] = ((header24 >> 8) & 255) as u8;
    hash[2] = (header24 >> 16) as u8;
    hash[3] = (header16 & 255) as u8;
    hash[4] = (header16 >> 8) as u8;
    if has_alpha {
        hash[5] = ((15.0 * a_dc).round() as u8) | (((15.0 * a_scale).round() as u8) << 4);
    }

    // Pack the AC terms as nibbles.
    let mut ac_index = 0usize;
    for ac in [&l_ac, &p_ac, &q_ac, &a_ac] {
        for f in ac.iter() {
            hash[ac_start + (ac_index >> 1)] |= ((15.0 * f).round() as u8) << ((ac_index & 1) << 2);
            ac_index += 1;
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    type Fixture = fn() -> (u32, u32, Vec<u8>);

    // Fixtures are defined by integer formulas, so the test
    // pixels are exactly the ones the reference encoder ran
    // over in the verification spike (issue #60).
    fn fixture_a() -> (u32, u32, Vec<u8>) {
        let (w, h) = (24u32, 18u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let j = ((x + y * w) * 4) as usize;
                rgba[j] = (x * 255 / (w - 1)) as u8;
                rgba[j + 1] = (y * 255 / (h - 1)) as u8;
                rgba[j + 2] = ((x + y) * 255 / (w + h - 2)) as u8;
                rgba[j + 3] = 255;
            }
        }
        (w, h, rgba)
    }

    fn fixture_b() -> (u32, u32, Vec<u8>) {
        let (w, h) = (17u32, 13u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let j = ((x + y * w) * 4) as usize;
                rgba[j] = 200;
                rgba[j + 1] = (40 + ((x * 10) % 160)) as u8;
                rgba[j + 2] = 90;
                rgba[j + 3] = if ((x >> 2) + (y >> 2)) % 2 == 0 {
                    255
                } else {
                    128
                };
            }
        }
        (w, h, rgba)
    }

    fn fixture_c() -> (u32, u32, Vec<u8>) {
        let (w, h) = (5u32, 5u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            rgba[i * 4] = 18;
            rgba[i * 4 + 1] = 52;
            rgba[i * 4 + 2] = 86;
            rgba[i * 4 + 3] = 255;
        }
        (w, h, rgba)
    }

    fn fixture_d() -> (u32, u32, Vec<u8>) {
        let (w, h) = (100u32, 100u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let j = ((x + y * w) * 4) as usize;
                rgba[j] = (x * 255 / 99) as u8;
                rgba[j + 1] = (y * 255 / 99) as u8;
                rgba[j + 2] = 128;
                rgba[j + 3] = (128 + x * 127 / 99) as u8;
            }
        }
        (w, h, rgba)
    }

    #[test]
    fn matches_the_reference_encoders_vectors() {
        // Base64 of the reference implementation's output for
        // the identical pixels (verification spike, #60).
        let cases: [(Fixture, &str); 3] = [
            (fixture_a, "H/gNDZyAh4dxeHiHiHiHiI+AB/iI"),
            (fixture_b, "4bmGFIoLQFmHh4d3CHR4v3jI+Y6Jqq7ZBQ=="),
            (fixture_d, "IRiKLRQrkIeQh3h4iJCQCPl2gIeIiHeIiA=="),
        ];
        for (make, want) in cases {
            let (w, h, rgba) = make();
            let got = crate::placeholder::base64_encode(&encode_rgba(w, h, &rgba));
            assert_eq!(got, want, "fixture {w}x{h}");
        }
    }

    /// The reference aspect-ratio extraction, for round-trip
    /// assertions.
    fn approximate_aspect_ratio(hash: &[u8]) -> f64 {
        let header = hash[3] as u32;
        let has_alpha = hash[2] & 0x80 != 0;
        let is_landscape = hash[4] & 0x80 != 0;
        let lx = if is_landscape {
            if has_alpha { 5 } else { 7 }
        } else {
            header & 7
        };
        let ly = if is_landscape {
            header & 7
        } else if has_alpha {
            5
        } else {
            7
        };
        lx as f64 / ly as f64
    }

    /// The reference average-color extraction (0..1 RGBA).
    fn average_rgba(hash: &[u8]) -> [f64; 4] {
        let header = (hash[0] as u32) | (hash[1] as u32) << 8 | (hash[2] as u32) << 16;
        let l = (header & 63) as f64 / 63.0;
        let p = ((header >> 6) & 63) as f64 / 31.5 - 1.0;
        let q = ((header >> 12) & 63) as f64 / 31.5 - 1.0;
        let has_alpha = header >> 23 != 0;
        let a = if has_alpha {
            (hash[5] & 15) as f64 / 15.0
        } else {
            1.0
        };
        let b = l - 2.0 / 3.0 * p;
        let r = (3.0 * l - b + q) / 2.0;
        let g = r - q;
        [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0), a]
    }

    #[test]
    fn solid_fixture_matches_the_reference_where_it_is_stable() {
        // The reference's AC nibbles for a solid image are
        // amplified float noise (see the module docs); its
        // 5-byte header (DC color, zero scales) is stable
        // and must match exactly, and the hash must decode
        // to the solid color at aspect 1.
        let (w, h, rgba) = fixture_c();
        let hash = encode_rgba(w, h, &rgba);
        assert_eq!(&hash[..5], &[77, 182, 1, 7, 0]);
        let avg = average_rgba(&hash);
        let want = [18.0 / 255.0, 52.0 / 255.0, 86.0 / 255.0, 1.0];
        for (i, c) in avg.iter().enumerate() {
            assert!(
                (c - want[i]).abs() < 0.02,
                "channel {i}: {c} vs {}",
                want[i]
            );
        }
        assert!((approximate_aspect_ratio(&hash) - 1.0).abs() < 0.01);
    }

    #[test]
    fn round_trip_preserves_aspect_and_average_color() {
        let (w, h, rgba) = fixture_a();
        let hash = encode_rgba(w, h, &rgba);
        let ratio = approximate_aspect_ratio(&hash);
        assert!((ratio - 24.0 / 18.0).abs() < 0.1, "ratio {ratio}");
        // The gradient averages to mid gray on every color
        // channel (alpha is 1: the fixture is opaque).
        let avg = average_rgba(&hash);
        for (i, c) in avg[..3].iter().enumerate() {
            assert!((c - 0.5).abs() < 0.08, "channel {i}: {c}");
        }
        assert_eq!(avg[3], 1.0);

        let (w, h, rgba) = fixture_b();
        let hash = encode_rgba(w, h, &rgba);
        let ratio = approximate_aspect_ratio(&hash);
        assert!((ratio - 17.0 / 13.0).abs() < 0.1, "ratio {ratio}");
        // Half the blocks are at alpha 128: mean alpha ~0.75.
        let avg = average_rgba(&hash);
        assert!((avg[3] - 0.75).abs() < 0.1, "alpha {}", avg[3]);
    }

    #[test]
    fn thumbhash_base64_downscales_large_inputs() {
        let img = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            400,
            200,
            image::Rgba([30, 90, 160, 255]),
        ));
        let s = thumbhash_base64(&img);
        assert!(s.len() >= 20, "hash string: {s}");
        let img = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            40,
            20,
            image::Rgba([30, 90, 160, 255]),
        ));
        assert_eq!(thumbhash_base64(&img).len(), s.len());
    }
}
