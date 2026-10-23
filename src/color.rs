//! Color management and embedded metadata, handled at the
//! byte level the way the EXIF orientation reader already
//! works: profiles and metadata segments are read from the
//! source bytes, pixels are converted with moxcms, and
//! anything preserved is injected into the encoded output
//! after the encoders have run, so no encoder needs to know
//! about any of it.

use image::DynamicImage;

/// The metadata a source carries, extracted once per
/// request: the ICC profile bytes, the EXIF TIFF stream
/// (orientation already patched to 1, because the pixels
/// were rotated at decode), the XMP packet, and the JPEG
/// Photoshop segment (home of IPTC), when present.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SourceMeta {
    pub icc: Option<Vec<u8>>,
    pub exif_tiff: Option<Vec<u8>>,
    pub xmp: Option<Vec<u8>>,
    pub photoshop: Option<Vec<u8>>,
}

const ICC_JPEG_HEADER: &[u8] = b"ICC_PROFILE\0";
const XMP_JPEG_HEADER: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const EXIF_JPEG_HEADER: &[u8] = b"Exif\0\0";
const PHOTOSHOP_JPEG_HEADER: &[u8] = b"Photoshop 3.0\0";
/// Sanity caps so a hostile source cannot make extraction
/// allocate without bound.
const MAX_ICC_BYTES: usize = 4 * 1024 * 1024;
const MAX_META_BYTES: usize = 16 * 1024 * 1024;

/// Extract the ICC profile and preservable metadata from a
/// JPEG or PNG source. Other formats yield an empty result:
/// their extraction is not supported in this pass.
pub fn extract_source_meta(bytes: &[u8]) -> SourceMeta {
    if bytes.starts_with(&[0xFF, 0xD8]) {
        extract_jpeg_meta(bytes)
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        extract_png_meta(bytes)
    } else {
        SourceMeta::default()
    }
}

/// Walk JPEG segments up to SOS, collecting the ICC chunks
/// (reassembled in sequence order), EXIF, XMP, and the
/// Photoshop segment.
fn extract_jpeg_meta(bytes: &[u8]) -> SourceMeta {
    let mut meta = SourceMeta::default();
    let mut icc_chunks: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut pos = 2; // past SOI
    while pos + 4 <= bytes.len() {
        if bytes[pos] != 0xFF {
            break;
        }
        let marker = bytes[pos + 1];
        if marker == 0xDA || marker == 0xD9 {
            break; // SOS or EOI: the metadata region ended
        }
        if marker == 0x01 || (0xD0..=0xD8).contains(&marker) {
            pos += 2; // standalone markers carry no length
            continue;
        }
        let len = u16::from_be_bytes([bytes[pos + 2], bytes[pos + 3]]) as usize;
        if len < 2 || pos + 2 + len > bytes.len() {
            break;
        }
        let payload = &bytes[pos + 4..pos + 2 + len];
        match marker {
            0xE1 if payload.starts_with(EXIF_JPEG_HEADER) => {
                let mut tiff = payload[EXIF_JPEG_HEADER.len()..].to_vec();
                patch_tiff_orientation(&mut tiff);
                meta.exif_tiff = Some(tiff);
            }
            0xE1 if payload.starts_with(XMP_JPEG_HEADER) => {
                meta.xmp = Some(payload[XMP_JPEG_HEADER.len()..].to_vec());
            }
            0xE2 if payload.len() > 14 && payload.starts_with(ICC_JPEG_HEADER) => {
                icc_chunks.push((payload[12], payload[14..].to_vec()));
            }
            0xED if payload.starts_with(PHOTOSHOP_JPEG_HEADER) => {
                meta.photoshop = Some(payload.to_vec());
            }
            _ => {}
        }
        pos += 2 + len;
    }
    if !icc_chunks.is_empty() {
        icc_chunks.sort_by_key(|(seq, _)| *seq);
        let mut icc = Vec::new();
        for (_, chunk) in icc_chunks {
            icc.extend_from_slice(&chunk);
            if icc.len() > MAX_ICC_BYTES {
                icc.clear();
                break;
            }
        }
        if !icc.is_empty() {
            meta.icc = Some(icc);
        }
    }
    meta
}

/// Walk PNG chunks up to IDAT, collecting the ICC profile
/// (iCCP, inflated), EXIF (eXIf), and XMP (the iTXt whose
/// keyword is `XML:com.adobe.xmp`).
fn extract_png_meta(bytes: &[u8]) -> SourceMeta {
    let mut meta = SourceMeta::default();
    let mut pos = 8; // past the signature
    while pos + 12 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let kind = &bytes[pos + 4..pos + 8];
        let Some(data) = bytes.get(pos + 8..pos + 8 + len) else {
            break;
        };
        match kind {
            b"iCCP" => {
                if let Some(icc) = parse_iccp(data) {
                    meta.icc = Some(icc);
                }
            }
            b"eXIf" => {
                let mut tiff = data.to_vec();
                patch_tiff_orientation(&mut tiff);
                meta.exif_tiff = Some(tiff);
            }
            b"iTXt" if data.starts_with(b"XML:com.adobe.xmp\0") => {
                if let Some(xmp) = parse_itxt_text(data) {
                    meta.xmp = Some(xmp);
                }
            }
            b"IDAT" => break,
            _ => {}
        }
        pos += 12 + len;
    }
    meta
}

/// The profile inside an iCCP chunk payload: profile name,
/// a compression byte (0 = zlib), then the stream.
fn parse_iccp(data: &[u8]) -> Option<Vec<u8>> {
    let nul = data.iter().position(|&b| b == 0)?;
    if data.get(nul + 1) != Some(&0) {
        return None;
    }
    let stream = &data[nul + 2..];
    use std::io::Read as _;
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(stream)
        .take(MAX_ICC_BYTES as u64 + 1)
        .read_to_end(&mut out)
        .ok()?;
    (out.len() <= MAX_ICC_BYTES && !out.is_empty()).then_some(out)
}

/// The text of an iTXt chunk: keyword, compression flag and
/// method, language and translated keywords (each
/// null-terminated), then the (possibly compressed) text.
fn parse_itxt_text(data: &[u8]) -> Option<Vec<u8>> {
    let mut cursor = 0;
    let skip_cstr = |data: &[u8], cursor: &mut usize| -> Option<()> {
        let nul = data[*cursor..].iter().position(|&b| b == 0)?;
        *cursor += nul + 1;
        Some(())
    };
    skip_cstr(data, &mut cursor)?; // keyword
    let compressed = *data.get(cursor)?;
    cursor += 2; // flag + method
    skip_cstr(data, &mut cursor)?; // language
    skip_cstr(data, &mut cursor)?; // translated keyword
    let text = &data[cursor..];
    if compressed == 0 {
        Some(text.to_vec())
    } else {
        use std::io::Read as _;
        let mut out = Vec::new();
        flate2::read::ZlibDecoder::new(text)
            .take(MAX_META_BYTES as u64 + 1)
            .read_to_end(&mut out)
            .ok()?;
        Some(out)
    }
}

/// Rewrite the Orientation tag (0x0112) of a TIFF stream to
/// 1, in place. The pixels a preserved EXIF describes were
/// already rotated at decode; copying the original value
/// would make every viewer rotate them a second time. Any
/// inconsistency leaves the stream untouched.
pub fn patch_tiff_orientation(tiff: &mut [u8]) {
    let little = match tiff.get(0..2) {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return,
    };
    let u16_at = |b: &[u8], off: usize| -> Option<u16> {
        let raw: [u8; 2] = b.get(off..off + 2)?.try_into().ok()?;
        Some(if little {
            u16::from_le_bytes(raw)
        } else {
            u16::from_be_bytes(raw)
        })
    };
    let u32_at = |b: &[u8], off: usize| -> Option<u32> {
        let raw: [u8; 4] = b.get(off..off + 4)?.try_into().ok()?;
        Some(if little {
            u32::from_le_bytes(raw)
        } else {
            u32::from_be_bytes(raw)
        })
    };
    if u16_at(tiff, 2) != Some(42) {
        return;
    }
    let Some(ifd) = u32_at(tiff, 4).map(|v| v as usize) else {
        return;
    };
    let Some(count) = u16_at(tiff, ifd) else {
        return;
    };
    for i in 0..count as usize {
        let entry = ifd + 2 + i * 12;
        if u16_at(tiff, entry) == Some(0x0112)
            && u16_at(tiff, entry + 2) == Some(3) // SHORT
            && u32_at(tiff, entry + 4) == Some(1)
        {
            // The value sits in the first two bytes of the
            // four-byte value field, in stream endianness.
            let bytes = if little {
                1u16.to_le_bytes()
            } else {
                1u16.to_be_bytes()
            };
            if let Some(slot) = tiff.get_mut(entry + 8..entry + 10) {
                slot.copy_from_slice(&bytes);
            }
            return;
        }
    }
}

/// The sRGB profile moxcms ships, encoded once: a source
/// profile byte-equal to it is already sRGB and needs no
/// transform.
fn srgb_profile_bytes() -> &'static [u8] {
    static BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    BYTES.get_or_init(|| {
        moxcms::ColorProfile::new_srgb()
            .encode()
            .expect("the built-in sRGB profile encodes")
    })
}

/// Convert decoded pixels from the embedded profile to
/// sRGB, for RGB(A) images at 8 or 16 bits. Profiles of
/// other color families, and unparseable profiles, leave
/// the image unchanged (the CMYK route is the decoder's,
/// and a broken profile must not fail the request).
pub fn convert_to_srgb(img: DynamicImage, icc: &[u8]) -> DynamicImage {
    if icc == srgb_profile_bytes() {
        return img;
    }
    let Ok(source) = moxcms::ColorProfile::new_from_slice(icc) else {
        return img;
    };
    if source.color_space != moxcms::DataColorSpace::Rgb {
        return img;
    }
    let srgb = moxcms::ColorProfile::new_srgb();
    let options = moxcms::TransformOptions::default();
    use moxcms::Layout;
    match img {
        DynamicImage::ImageRgb8(mut buf) => {
            let Ok(t) = source.create_transform_8bit(Layout::Rgb, &srgb, Layout::Rgb, options)
            else {
                return DynamicImage::ImageRgb8(buf);
            };
            let mut out = vec![0u8; buf.len()];
            if t.transform(buf.as_raw(), &mut out).is_ok() {
                buf.copy_from_slice(&out);
            }
            DynamicImage::ImageRgb8(buf)
        }
        DynamicImage::ImageRgba8(mut buf) => {
            let Ok(t) = source.create_transform_8bit(Layout::Rgba, &srgb, Layout::Rgba, options)
            else {
                return DynamicImage::ImageRgba8(buf);
            };
            let mut out = vec![0u8; buf.len()];
            if t.transform(buf.as_raw(), &mut out).is_ok() {
                buf.copy_from_slice(&out);
            }
            DynamicImage::ImageRgba8(buf)
        }
        DynamicImage::ImageRgb16(mut buf) => {
            let Ok(t) = source.create_transform_16bit(Layout::Rgb, &srgb, Layout::Rgb, options)
            else {
                return DynamicImage::ImageRgb16(buf);
            };
            let mut out = vec![0u16; buf.len()];
            if t.transform(buf.as_raw(), &mut out).is_ok() {
                buf.copy_from_slice(&out);
            }
            DynamicImage::ImageRgb16(buf)
        }
        DynamicImage::ImageRgba16(mut buf) => {
            let Ok(t) = source.create_transform_16bit(Layout::Rgba, &srgb, Layout::Rgba, options)
            else {
                return DynamicImage::ImageRgba16(buf);
            };
            let mut out = vec![0u16; buf.len()];
            if t.transform(buf.as_raw(), &mut out).is_ok() {
                buf.copy_from_slice(&out);
            }
            DynamicImage::ImageRgba16(buf)
        }
        other => other,
    }
}

// ----------------------------------------------------------------
// Output injection
// ----------------------------------------------------------------

/// One JPEG APP segment (marker + payload), length-checked.
fn jpeg_segment(marker: u8, payload: &[u8]) -> Option<Vec<u8>> {
    if payload.len() + 2 > u16::MAX as usize {
        return None;
    }
    let mut out = Vec::with_capacity(payload.len() + 4);
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
    Some(out)
}

/// Inject the ICC profile and any preserved metadata into
/// encoded JPEG bytes, right after SOI. Segments that do
/// not fit the 64 KiB segment limit are dropped (an ICC
/// profile is chunked to fit; oversized metadata is
/// skipped), and a malformed input passes through.
pub fn inject_jpeg(bytes: &[u8], icc: Option<&[u8]>, meta: &SourceMeta, preserve: bool) -> Vec<u8> {
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return bytes.to_vec();
    }
    let mut inserts: Vec<u8> = Vec::new();
    if preserve {
        if let Some(tiff) = &meta.exif_tiff {
            let mut payload = EXIF_JPEG_HEADER.to_vec();
            payload.extend_from_slice(tiff);
            if let Some(seg) = jpeg_segment(0xE1, &payload) {
                inserts.extend_from_slice(&seg);
            }
        }
        if let Some(xmp) = &meta.xmp {
            let mut payload = XMP_JPEG_HEADER.to_vec();
            payload.extend_from_slice(xmp);
            if let Some(seg) = jpeg_segment(0xE1, &payload) {
                inserts.extend_from_slice(&seg);
            }
        }
        if let Some(ps) = &meta.photoshop
            && let Some(seg) = jpeg_segment(0xED, ps)
        {
            inserts.extend_from_slice(&seg);
        }
    }
    if let Some(profile) = icc {
        // ICC in JPEG travels in APP2 chunks of at most
        // 65519 data bytes, each headed by the 14-byte
        // signature, sequence number, and chunk count.
        let chunks: Vec<&[u8]> = profile.chunks(65519).collect();
        if chunks.len() <= 255 {
            for (i, chunk) in chunks.iter().enumerate() {
                let mut payload = ICC_JPEG_HEADER.to_vec();
                payload.push(i as u8 + 1);
                payload.push(chunks.len() as u8);
                payload.extend_from_slice(chunk);
                if let Some(seg) = jpeg_segment(0xE2, &payload) {
                    inserts.extend_from_slice(&seg);
                }
            }
        }
    }
    if inserts.is_empty() {
        return bytes.to_vec();
    }
    let mut out = Vec::with_capacity(bytes.len() + inserts.len());
    out.extend_from_slice(&bytes[..2]);
    out.extend_from_slice(&inserts);
    out.extend_from_slice(&bytes[2..]);
    out
}

fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (i, entry) in table.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *entry = c;
        }
        table
    });
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = table[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

fn png_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 12);
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = kind.to_vec();
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
    out
}

/// Inject the ICC profile (`iCCP`) and preserved metadata
/// (`eXIf`, XMP `iTXt`) into encoded PNG bytes, right after
/// IHDR, where the spec wants them (before PLTE and IDAT).
pub fn inject_png(bytes: &[u8], icc: Option<&[u8]>, meta: &SourceMeta, preserve: bool) -> Vec<u8> {
    // Signature + IHDR chunk = 8 + 25 bytes; anything else
    // is not the encoder output we expect.
    if bytes.len() < 33 || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") || &bytes[12..16] != b"IHDR" {
        return bytes.to_vec();
    }
    let mut inserts: Vec<u8> = Vec::new();
    if let Some(profile) = icc {
        use std::io::Write as _;
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        let compressed = if encoder.write_all(profile).is_ok() {
            encoder.finish().unwrap_or_default()
        } else {
            Vec::new()
        };
        if !compressed.is_empty() {
            let mut data = b"ICC Profile\0".to_vec();
            data.push(0);
            data.extend_from_slice(&compressed);
            inserts.extend_from_slice(&png_chunk(b"iCCP", &data));
        }
    }
    if preserve {
        if let Some(tiff) = &meta.exif_tiff {
            inserts.extend_from_slice(&png_chunk(b"eXIf", tiff));
        }
        if let Some(xmp) = &meta.xmp {
            let mut data = b"XML:com.adobe.xmp\0".to_vec();
            data.extend_from_slice(&[0, 0, 0, 0]); // uncompressed, no language/translation
            // iTXt layout: keyword NUL, compression flag,
            // method, language NUL, translated NUL, text. The
            // four bytes above are flag+method+two NULs.
            data.extend_from_slice(xmp);
            inserts.extend_from_slice(&png_chunk(b"iTXt", &data));
        }
    }
    if inserts.is_empty() {
        return bytes.to_vec();
    }
    let mut out = Vec::with_capacity(bytes.len() + inserts.len());
    out.extend_from_slice(&bytes[..33]);
    out.extend_from_slice(&inserts);
    out.extend_from_slice(&bytes[33..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal TIFF with an Orientation SHORT of 6.
    fn tiff_with_orientation(value: u16, little: bool) -> Vec<u8> {
        let mut t = Vec::new();
        if little {
            t.extend_from_slice(b"II");
            t.extend_from_slice(&42u16.to_le_bytes());
            t.extend_from_slice(&8u32.to_le_bytes());
            t.extend_from_slice(&1u16.to_le_bytes());
            t.extend_from_slice(&0x0112u16.to_le_bytes());
            t.extend_from_slice(&3u16.to_le_bytes());
            t.extend_from_slice(&1u32.to_le_bytes());
            t.extend_from_slice(&value.to_le_bytes());
            t.extend_from_slice(&0u16.to_le_bytes());
        } else {
            t.extend_from_slice(b"MM");
            t.extend_from_slice(&42u16.to_be_bytes());
            t.extend_from_slice(&8u32.to_be_bytes());
            t.extend_from_slice(&1u16.to_be_bytes());
            t.extend_from_slice(&0x0112u16.to_be_bytes());
            t.extend_from_slice(&3u16.to_be_bytes());
            t.extend_from_slice(&1u32.to_be_bytes());
            t.extend_from_slice(&value.to_be_bytes());
            t.extend_from_slice(&0u16.to_be_bytes());
        }
        t
    }

    #[test]
    fn orientation_patch_rewrites_both_endiannesses() {
        for little in [true, false] {
            let mut tiff = tiff_with_orientation(6, little);
            patch_tiff_orientation(&mut tiff);
            let off = 8 + 2;
            let value = if little {
                u16::from_le_bytes(tiff[off + 8..off + 10].try_into().unwrap())
            } else {
                u16::from_be_bytes(tiff[off + 8..off + 10].try_into().unwrap())
            };
            assert_eq!(value, 1, "little={little}");
        }
        // Garbage is left alone.
        let mut junk = vec![1u8, 2, 3];
        patch_tiff_orientation(&mut junk);
        assert_eq!(junk, [1, 2, 3]);
    }

    #[test]
    fn jpeg_meta_round_trips_through_injection() {
        // Build a bare JPEG: SOI + one DQT-ish segment + EOI.
        let mut src = vec![0xFF, 0xD8];
        src.extend_from_slice(&jpeg_segment(0xE0, b"JFIF\0").unwrap());
        src.extend_from_slice(&[0xFF, 0xD9]);
        let tiff = tiff_with_orientation(6, true);
        let meta_in = SourceMeta {
            icc: Some(vec![7u8; 70000]), // forces two APP2 chunks
            exif_tiff: Some(tiff),
            xmp: Some(b"<x:xmpmeta/>".to_vec()),
            photoshop: Some(b"Photoshop 3.0\0rest".to_vec()),
        };
        let out = inject_jpeg(&src, meta_in.icc.as_deref(), &meta_in, true);
        let got = extract_jpeg_meta(&out);
        assert_eq!(got.icc, meta_in.icc);
        assert_eq!(got.xmp, meta_in.xmp);
        assert_eq!(got.photoshop, meta_in.photoshop);
        // The orientation was 6 going in; extraction patches
        // it to 1 (it patches again, idempotently).
        let tiff = got.exif_tiff.unwrap();
        assert_eq!(
            u16::from_le_bytes(tiff[8 + 2 + 8..8 + 2 + 10].try_into().unwrap()),
            1
        );
        // Without preservation only the ICC survives.
        let out = inject_jpeg(&src, meta_in.icc.as_deref(), &meta_in, false);
        let got = extract_jpeg_meta(&out);
        assert_eq!(got.icc, meta_in.icc);
        assert!(got.exif_tiff.is_none() && got.xmp.is_none());
    }

    #[test]
    fn png_meta_round_trips_through_injection() {
        // A minimal PNG: signature + IHDR + IEND.
        let mut src = b"\x89PNG\r\n\x1a\n".to_vec();
        src.extend_from_slice(&png_chunk(
            b"IHDR",
            &[0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0],
        ));
        src.extend_from_slice(&png_chunk(b"IEND", &[]));
        let meta_in = SourceMeta {
            icc: Some(b"fake-profile-bytes".to_vec()),
            exif_tiff: Some(tiff_with_orientation(3, true)),
            xmp: Some(b"<x:xmpmeta/>".to_vec()),
            photoshop: None,
        };
        let out = inject_png(&src, meta_in.icc.as_deref(), &meta_in, true);
        let got = extract_png_meta(&out);
        assert_eq!(got.icc, meta_in.icc);
        assert_eq!(got.xmp, meta_in.xmp);
        assert!(got.exif_tiff.is_some());
    }

    #[test]
    fn p3_pixels_convert_to_the_reference_srgb_values() {
        // Cross-checked against littlecms during development:
        // these are the colorimetric Display P3 -> sRGB
        // results, not the library compared with itself.
        let icc = moxcms::ColorProfile::new_display_p3().encode().unwrap();
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(3, 1, |x, _| {
            [
                image::Rgb([200, 80, 40]),
                image::Rgb([40, 120, 200]),
                image::Rgb([90, 160, 90]),
            ][x as usize]
        }));
        let out = convert_to_srgb(img, &icc).to_rgb8();
        let expected: [[u8; 3]; 3] = [[216, 69, 17], [0, 122, 206], [61, 162, 81]];
        for (i, want) in expected.iter().enumerate() {
            let got = out.get_pixel(i as u32, 0).0;
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(
                    (i32::from(*g) - i32::from(*w)).abs() <= 2,
                    "pixel {i}: {got:?} vs {want:?}"
                );
            }
        }
        // The sRGB profile itself is a no-op.
        let img =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(1, 1, image::Rgb([10, 20, 30])));
        let out = convert_to_srgb(img, srgb_profile_bytes()).to_rgb8();
        assert_eq!(out.get_pixel(0, 0).0, [10, 20, 30]);
    }
}
