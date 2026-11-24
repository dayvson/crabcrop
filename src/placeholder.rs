//! Placeholder data for meta responses: the processed image's
//! dominant color and a tiny blurred preview, computed from the
//! final pixels so both match the requested operations exactly.

use image::{DynamicImage, codecs::jpeg::JpegEncoder};

use crate::ai::resample::{self, Kernel};

/// Placeholder data attached to a processed image when the
/// request asked for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placeholder {
    /// `#rrggbb`: the image's average color (a 1x1 resample).
    pub dominant_color: String,
    /// `data:image/jpeg;base64,...`: a 16px-wide blurred preview.
    pub data_uri: String,
    /// The five most prominent colors as `#rrggbb`, most
    /// prominent first.
    pub palette: Vec<String>,
    /// A ThumbHash of the image, base64: decodable by
    /// ThumbHash client libraries into an aspect-correct
    /// blurred preview.
    pub thumbhash: String,
}

/// Compute the placeholder for a processed image. Its three
/// resizes are the image crate's Triangle resizes, computed
/// over rows (issue #811, see `crate::ai::resample`): the
/// same bytes.
pub fn compute(img: &DynamicImage) -> Placeholder {
    Placeholder {
        dominant_color: dominant_color(img),
        data_uri: data_uri(img),
        palette: palette(img),
        thumbhash: crate::thumbhash::thumbhash_base64(img),
    }
}

/// The top colors of the image: a 4-bit-per-channel
/// histogram over a <=100x100 resample, pixels weighted by
/// alpha, each reported color its bin's weighted average.
/// At most five entries, most prominent first; ties break
/// by bin so the result is deterministic. A fully
/// transparent image has no prominent colors and reports
/// just its (RGB) dominant color.
fn palette(img: &DynamicImage) -> Vec<String> {
    let small = resample::resize_to_fit(img, 100, 100, Kernel::Triangle).to_rgba8();
    let mut sums = vec![[0.0f64; 3]; 4096];
    let mut weights = vec![0.0f64; 4096];
    for px in small.pixels() {
        let a = px[3] as f64 / 255.0;
        if a == 0.0 {
            continue;
        }
        let bin =
            ((px[0] as usize >> 4) << 8) | ((px[1] as usize >> 4) << 4) | (px[2] as usize >> 4);
        for c in 0..3 {
            sums[bin][c] += a * px[c] as f64;
        }
        weights[bin] += a;
    }
    let mut bins: Vec<usize> = (0..4096).filter(|&b| weights[b] > 0.0).collect();
    bins.sort_by(|&x, &y| {
        weights[y]
            .partial_cmp(&weights[x])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.cmp(&y))
    });
    let out: Vec<String> = bins
        .iter()
        .take(5)
        .map(|&b| {
            let w = weights[b];
            format!(
                "#{:02x}{:02x}{:02x}",
                (sums[b][0] / w).round() as u8,
                (sums[b][1] / w).round() as u8,
                (sums[b][2] / w).round() as u8
            )
        })
        .collect();
    if out.is_empty() {
        vec![dominant_color(img)]
    } else {
        out
    }
}

/// The average color: a 1x1 triangle resample is a true mean,
/// so a solid image reports its exact color.
fn dominant_color(img: &DynamicImage) -> String {
    let one = resample::resize_exact(
        &DynamicImage::ImageRgb8(img.to_rgb8()),
        1,
        1,
        Kernel::Triangle,
    );
    let px = one.to_rgb8().get_pixel(0, 0).0;
    format!("#{:02x}{:02x}{:02x}", px[0], px[1], px[2])
}

/// A 16px-wide JPEG preview, blurred after downscaling (the
/// downscale itself already averages most detail away), as a
/// base64 data URI.
fn data_uri(img: &DynamicImage) -> String {
    let (w, h) = (img.width().max(1), img.height().max(1));
    let tiny_h = ((16.0 * h as f32 / w as f32).round() as u32).max(1);
    let tiny = resample::resize_exact(
        &DynamicImage::ImageRgb8(img.to_rgb8()),
        16,
        tiny_h,
        Kernel::Triangle,
    )
    .blur(1.0);
    let mut bytes = Vec::new();
    // A 16px-wide in-memory JPEG encode does not fail in
    // practice; if it ever did, an empty preview beats an error.
    let _ = JpegEncoder::new_with_quality(&mut bytes, 60).encode_image(&tiny);
    format!("data:image/jpeg;base64,{}", base64_encode(&bytes))
}

/// Standard base64 with padding (RFC 4648, section 4), from the
/// base64 crate (issue #737): the encoding of the placeholder's
/// data URI and ThumbHash, and of the embeddings the meta
/// response packs.
pub fn base64_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(r: u8, g: u8, b: u8, w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([r, g, b])))
    }

    #[test]
    fn base64_matches_reference_vectors() {
        // RFC 4648 test vectors.
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64_encode(&[0xFF, 0xEE, 0xDD]), "/+7d");
    }

    const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    /// The encoder this module carried before issue #737, kept
    /// as the oracle for the crate that replaced it.
    fn base64_by_hand(data: &[u8]) -> String {
        let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
        for chunk in data.chunks(3) {
            let b = [
                chunk[0] as u32,
                *chunk.get(1).unwrap_or(&0) as u32,
                *chunk.get(2).unwrap_or(&0) as u32,
            ];
            let n = (b[0] << 16) | (b[1] << 8) | b[2];
            out.push(BASE64[(n >> 18) as usize & 63] as char);
            out.push(BASE64[(n >> 12) as usize & 63] as char);
            out.push(if chunk.len() > 1 {
                BASE64[(n >> 6) as usize & 63] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                BASE64[n as usize & 63] as char
            } else {
                '='
            });
        }
        out
    }

    #[test]
    fn base64_matches_the_hand_written_encoder() {
        // Issue #737: the base64 crate replaced the encoder
        // above, and must write what it wrote. Random bytes at
        // every length to 600, so each remainder mod 3 (no
        // padding, two `=`, one `=`) comes up at least 200
        // times; inputs of a megabyte at each remainder; every
        // one- and two-byte input, which are the padded tails;
        // and every byte value in each place of a full group.
        let mut rng = crate::test_support::Rng::new(112);
        let mut cases = 0;
        let mut same = |data: &[u8]| {
            assert!(
                base64_encode(data) == base64_by_hand(data),
                "{} bytes, starting {:02x?}",
                data.len(),
                &data[..data.len().min(8)]
            );
            cases += 1;
        };
        for len in 0..=600 {
            same(&rng.bytes(len));
        }
        for len in [1 << 20, (1 << 20) + 1, (1 << 20) + 2] {
            same(&rng.bytes(len));
        }
        for a in 0..=255u8 {
            same(&[a]);
            for b in 0..=255u8 {
                same(&[a, b]);
            }
            for place in 0..3 {
                let mut group = [0x5A; 3];
                group[place] = a;
                same(&group);
            }
        }
        assert_eq!(cases, 601 + 3 + 256 * (1 + 256 + 3));
    }

    /// Minimal decoder for round-trip assertions.
    fn base64_decode(s: &str) -> Vec<u8> {
        let mut vals = Vec::new();
        for c in s.bytes() {
            vals.push(match c {
                b'A'..=b'Z' => c - b'A',
                b'a'..=b'z' => c - b'a' + 26,
                b'0'..=b'9' => c - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' => 64,
                _ => panic!("bad base64 in test"),
            } as u32);
        }
        let mut out = Vec::new();
        for chunk in vals.chunks(4) {
            let n = (chunk[0] << 18) | (chunk[1] << 12) | (chunk[2] << 6) | chunk[3];
            out.push((n >> 16) as u8);
            if chunk[2] != 64 {
                out.push((n >> 8) as u8);
            }
            if chunk[3] != 64 {
                out.push(n as u8);
            }
        }
        out
    }

    #[test]
    fn dominant_color_of_a_solid_image_is_exact() {
        assert_eq!(compute(&solid(255, 0, 0, 40, 30)).dominant_color, "#ff0000");
        assert_eq!(compute(&solid(18, 52, 86, 7, 3)).dominant_color, "#123456");
    }

    #[test]
    fn palette_of_a_solid_image_is_that_color() {
        let p = compute(&solid(18, 52, 86, 40, 30));
        assert_eq!(p.palette, vec!["#123456".to_string()]);
        assert!(!p.thumbhash.is_empty());
    }

    #[test]
    fn palette_ranks_colors_by_prominence() {
        // Three quarters red (left 30 of 40 columns), one
        // quarter blue.
        let mut img = image::RgbImage::new(40, 40);
        for (x, _y, px) in img.enumerate_pixels_mut() {
            *px = if x < 30 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 0, 255])
            };
        }
        let p = compute(&DynamicImage::ImageRgb8(img));
        assert_eq!(p.palette[0], "#ff0000");
        assert!(
            p.palette.contains(&"#0000ff".to_string()),
            "{:?}",
            p.palette
        );
    }

    /// A render in each layout the placeholder can be handed,
    /// with every alpha level where it has alpha.
    fn renders(w: u32, h: u32) -> Vec<DynamicImage> {
        let mut rng = crate::test_support::Rng::new(811);
        let rgb = image::RgbImage::from_raw(w, h, rng.bytes((w * h * 3) as usize)).unwrap();
        let rgba = image::RgbaImage::from_raw(w, h, rng.bytes((w * h * 4) as usize)).unwrap();
        let (rgb, rgba) = (DynamicImage::ImageRgb8(rgb), DynamicImage::ImageRgba8(rgba));
        vec![
            DynamicImage::ImageLuma8(rgb.to_luma8()),
            DynamicImage::ImageRgba16(rgba.to_rgba16()),
            rgb,
            rgba,
        ]
    }

    #[test]
    fn the_resizes_are_the_image_crate_s() {
        // Issue #811: the three resizes `compute` makes, in each
        // layout, at a render's size and at odd, small, long and
        // already fitting ones: the crate's calls they stand for,
        // layout and bytes.
        use image::imageops::FilterType;
        let same = |a: &DynamicImage, b: &DynamicImage| {
            a.color() == b.color()
                && (a.width(), a.height()) == (b.width(), b.height())
                && a.as_bytes() == b.as_bytes()
        };
        for (w, h) in [
            (1, 1),
            (90, 40),
            (100, 100),
            (101, 100),
            (333, 1001),
            (800, 533),
        ] {
            for img in renders(w, h) {
                let rgb = DynamicImage::ImageRgb8(img.to_rgb8());
                let tiny_h = ((16.0 * h as f32 / w as f32).round() as u32).max(1);
                for (nw, nh) in [(1, 1), (16, tiny_h)] {
                    let ours = resample::resize_exact(&rgb, nw, nh, Kernel::Triangle);
                    let theirs = rgb.resize_exact(nw, nh, FilterType::Triangle);
                    assert!(
                        same(&ours, &theirs),
                        "{:?} {w}x{h} to {nw}x{nh}",
                        img.color()
                    );
                }
                let ours = resample::resize_to_fit(&img, 100, 100, Kernel::Triangle);
                let theirs = img.resize(100, 100, FilterType::Triangle);
                assert!(
                    same(&ours, &theirs),
                    "{:?} {w}x{h} into 100x100",
                    img.color()
                );
            }
        }
    }

    #[test]
    fn placeholder_decodes_to_a_tiny_jpeg() {
        let p = compute(&solid(0, 128, 255, 100, 60));
        let b64 = p.data_uri.strip_prefix("data:image/jpeg;base64,").unwrap();
        let bytes = base64_decode(b64);
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!(img.width(), 16);
        assert_eq!(img.height(), 10); // 16 * 60/100 = 9.6, rounded
        assert_eq!(
            image::guess_format(&bytes).unwrap(),
            image::ImageFormat::Jpeg
        );
    }
}
