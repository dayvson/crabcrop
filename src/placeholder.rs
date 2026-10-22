//! Placeholder data for meta responses: the processed image's
//! dominant color and a tiny blurred preview, computed from the
//! final pixels so both match the requested operations exactly.

use image::{DynamicImage, codecs::jpeg::JpegEncoder, imageops::FilterType};

/// Placeholder data attached to a processed image when the
/// request asked for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placeholder {
    /// `#rrggbb`: the image's average color (a 1x1 resample).
    pub dominant_color: String,
    /// `data:image/jpeg;base64,...`: a 16px-wide blurred preview.
    pub data_uri: String,
}

/// Compute the placeholder for a processed image.
pub fn compute(img: &DynamicImage) -> Placeholder {
    Placeholder {
        dominant_color: dominant_color(img),
        data_uri: data_uri(img),
    }
}

/// The average color: a 1x1 triangle resample is a true mean,
/// so a solid image reports its exact color.
fn dominant_color(img: &DynamicImage) -> String {
    let one = DynamicImage::ImageRgb8(img.to_rgb8()).resize_exact(1, 1, FilterType::Triangle);
    let px = one.to_rgb8().get_pixel(0, 0).0;
    format!("#{:02x}{:02x}{:02x}", px[0], px[1], px[2])
}

/// A 16px-wide JPEG preview, blurred after downscaling (the
/// downscale itself already averages most detail away), as a
/// base64 data URI.
fn data_uri(img: &DynamicImage) -> String {
    let (w, h) = (img.width().max(1), img.height().max(1));
    let tiny_h = ((16.0 * h as f32 / w as f32).round() as u32).max(1);
    let tiny = DynamicImage::ImageRgb8(img.to_rgb8())
        .resize_exact(16, tiny_h, FilterType::Triangle)
        .blur(1.0);
    let mut bytes = Vec::new();
    // A 16px-wide in-memory JPEG encode does not fail in
    // practice; if it ever did, an empty preview beats an error.
    let _ = JpegEncoder::new_with_quality(&mut bytes, 60).encode_image(&tiny);
    format!("data:image/jpeg;base64,{}", base64_encode(&bytes))
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding. (The crate carries no base64
/// dependency; this is all the meta response needs.)
pub fn base64_encode(data: &[u8]) -> String {
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
