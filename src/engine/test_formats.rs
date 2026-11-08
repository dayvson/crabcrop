//! Sample files for the decode-surface tests (issue #466),
//! built in code rather than committed as fixtures: each is a
//! short header plus a few raw pixels.
//!
//! The first group covers the formats the image crate detects
//! by magic bytes but crabcrop does not decode (OpenEXR, QOI,
//! farbfeld, PNM, Radiance HDR, DDS). Every sample is a valid
//! file in its format, which the image crate's default
//! decoders accept, so a refusal proves the decoder is not
//! compiled in rather than that the bytes are broken. The last
//! sample is an AV1 still whose major brand claims generic
//! HEIF (`mif1`), the shape that once carried AV1 bytes into
//! the HEIC path.

use image::ImageEncoder;

/// The sample pixel at (x, y): distinct per pixel, never the
/// (1, 1, 1) or (2, 2, _) run markers Radiance HDR scanlines
/// reserve.
fn pixel(x: u32, y: u32) -> [u8; 3] {
    [(x * 60) as u8, (y * 60) as u8, 128]
}

/// Every format the image crate can sniff that crabcrop does
/// not decode, as (name, a 4x4 sample).
pub(crate) fn undocumented_samples() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("openexr", openexr(4, 4)),
        ("qoi", qoi(4, 4)),
        ("farbfeld", farbfeld(4, 4)),
        ("pnm", ppm(4, 4)),
        ("radiance-hdr", radiance_hdr(4, 4)),
        ("dds", dds_dxt1(4, 4)),
    ]
}

/// A scanline OpenEXR file: B, G, R float channels, no
/// compression, one line per block.
pub(crate) fn openexr(width: u32, height: u32) -> Vec<u8> {
    fn attribute(out: &mut Vec<u8>, name: &str, kind: &str, value: &[u8]) {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(kind.as_bytes());
        out.push(0);
        out.extend_from_slice(&(value.len() as i32).to_le_bytes());
        out.extend_from_slice(value);
    }
    // Magic, then version 2 with no flags: single-part scanline.
    let mut out = vec![0x76, 0x2f, 0x31, 0x01, 2, 0, 0, 0];
    let mut channels = Vec::new();
    for name in ["B", "G", "R"] {
        channels.extend_from_slice(name.as_bytes());
        channels.push(0);
        channels.extend_from_slice(&2i32.to_le_bytes()); // FLOAT
        channels.extend_from_slice(&[0, 0, 0, 0]); // pLinear, reserved
        channels.extend_from_slice(&1i32.to_le_bytes()); // x sampling
        channels.extend_from_slice(&1i32.to_le_bytes()); // y sampling
    }
    channels.push(0);
    attribute(&mut out, "channels", "chlist", &channels);
    attribute(&mut out, "compression", "compression", &[0]);
    let window: Vec<u8> = [0, 0, width as i32 - 1, height as i32 - 1]
        .iter()
        .flat_map(|v: &i32| v.to_le_bytes())
        .collect();
    attribute(&mut out, "dataWindow", "box2i", &window);
    attribute(&mut out, "displayWindow", "box2i", &window);
    attribute(&mut out, "lineOrder", "lineOrder", &[0]);
    attribute(&mut out, "pixelAspectRatio", "float", &1f32.to_le_bytes());
    attribute(&mut out, "screenWindowCenter", "v2f", &[0; 8]);
    attribute(&mut out, "screenWindowWidth", "float", &1f32.to_le_bytes());
    out.push(0);
    let line_bytes = width as usize * 3 * 4;
    let first_block = out.len() + height as usize * 8;
    for y in 0..height as usize {
        let offset = first_block + y * (8 + line_bytes);
        out.extend_from_slice(&(offset as u64).to_le_bytes());
    }
    for y in 0..height {
        out.extend_from_slice(&(y as i32).to_le_bytes());
        out.extend_from_slice(&(line_bytes as i32).to_le_bytes());
        for channel in [2, 1, 0] {
            for x in 0..width {
                let value = pixel(x, y)[channel] as f32 / 255.0;
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    out
}

/// A QOI file: every pixel spelled as a full RGB op.
pub(crate) fn qoi(width: u32, height: u32) -> Vec<u8> {
    let mut out = b"qoif".to_vec();
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&[3, 0]); // RGB, sRGB
    for y in 0..height {
        for x in 0..width {
            out.push(0xfe);
            out.extend_from_slice(&pixel(x, y));
        }
    }
    out.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
    out
}

/// A farbfeld file: 16-bit big-endian RGBA.
pub(crate) fn farbfeld(width: u32, height: u32) -> Vec<u8> {
    let mut out = b"farbfeld".to_vec();
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&height.to_be_bytes());
    for y in 0..height {
        for x in 0..width {
            let [r, g, b] = pixel(x, y);
            for channel in [r, g, b, 255] {
                out.extend_from_slice(&(channel as u16 * 257).to_be_bytes());
            }
        }
    }
    out
}

/// A binary PPM (PNM) file.
pub(crate) fn ppm(width: u32, height: u32) -> Vec<u8> {
    let mut out = format!("P6\n{width} {height}\n255\n").into_bytes();
    for y in 0..height {
        for x in 0..width {
            out.extend_from_slice(&pixel(x, y));
        }
    }
    out
}

/// A Radiance HDR file with flat (unencoded) scanlines, which
/// the format uses below 8 pixels of width.
pub(crate) fn radiance_hdr(width: u32, height: u32) -> Vec<u8> {
    let mut out =
        format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n").into_bytes();
    for y in 0..height {
        for x in 0..width {
            out.extend_from_slice(&pixel(x, y));
            out.push(128); // exponent: each mantissa over 256
        }
    }
    out
}

/// A DDS file holding one DXT1 mip level.
pub(crate) fn dds_dxt1(width: u32, height: u32) -> Vec<u8> {
    let blocks = width.div_ceil(4) * height.div_ceil(4);
    let mut header = [0u32; 31];
    header[0] = 124; // header size
    header[1] = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000; // caps, height, width, pixel format, linear size
    header[2] = height;
    header[3] = width;
    header[4] = blocks * 8;
    header[18] = 32; // pixel format size
    header[19] = 0x4; // a FourCC follows
    header[20] = u32::from_le_bytes(*b"DXT1");
    header[26] = 0x1000; // a texture
    let mut out = b"DDS ".to_vec();
    for word in header {
        out.extend_from_slice(&word.to_le_bytes());
    }
    for _ in 0..blocks {
        // Endpoints red and blue (RGB565), every texel the first.
        out.extend_from_slice(&[0x00, 0xf8, 0x1f, 0x00, 0, 0, 0, 0]);
    }
    out
}

/// An AV1 still (an AVIF file from the encoder crabcrop ships)
/// with its major brand rewritten from `avif` to `mif1`: a
/// well-formed HEIF file whose only image item is `av01`.
pub(crate) fn av1_in_mif1() -> Vec<u8> {
    let img = image::RgbImage::from_fn(24, 16, |x, y| image::Rgb(pixel(x / 6, y / 4)));
    let mut bytes = Vec::new();
    image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut bytes, 10, 80)
        .write_image(img.as_raw(), 24, 16, image::ExtendedColorType::Rgb8)
        .unwrap();
    assert_eq!(
        &bytes[4..12],
        b"ftypavif",
        "the encoder writes an AVIF ftyp"
    );
    bytes[8..12].copy_from_slice(b"mif1");
    bytes
}
