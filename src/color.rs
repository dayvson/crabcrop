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
pub(crate) const MAX_ICC_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const MAX_META_BYTES: usize = 16 * 1024 * 1024;
/// The most zlib output one source may inflate across all
/// of its metadata chunks (issue #399): one ICC profile and
/// one XMP packet, each at its own cap. The caps alone are
/// per chunk, and 16 MiB of zeros deflates to about 16 KB,
/// so a PNG under the fetch cap could otherwise carry
/// hundreds of chunks that each inflate to the cap.
pub(crate) const MAX_INFLATE_BYTES: usize = MAX_ICC_BYTES + MAX_META_BYTES;

/// Extract the ICC profile and, when `preserve` is set (the
/// operator's metadata policy will copy them into the
/// output), the preservable metadata from a JPEG or PNG
/// source. The profile is always read: it drives color
/// conversion. Other formats yield an empty result: their
/// extraction is not supported in this pass.
pub fn extract_source_meta(bytes: &[u8], preserve: bool) -> SourceMeta {
    if bytes.starts_with(&[0xFF, 0xD8]) {
        extract_jpeg_meta(bytes, preserve)
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        let mut inflate_budget = MAX_INFLATE_BYTES;
        extract_png_meta(bytes, preserve, &mut inflate_budget)
    } else {
        SourceMeta::default()
    }
}

/// Walk JPEG segments up to SOS, collecting the ICC chunks
/// (reassembled in sequence order) and, when `preserve` is
/// set, EXIF, XMP, and the Photoshop segment.
fn extract_jpeg_meta(bytes: &[u8], preserve: bool) -> SourceMeta {
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
            0xE1 if preserve && payload.starts_with(EXIF_JPEG_HEADER) => {
                let mut tiff = payload[EXIF_JPEG_HEADER.len()..].to_vec();
                patch_tiff_orientation(&mut tiff);
                meta.exif_tiff = Some(tiff);
            }
            0xE1 if preserve && payload.starts_with(XMP_JPEG_HEADER) => {
                meta.xmp = Some(payload[XMP_JPEG_HEADER.len()..].to_vec());
            }
            0xE2 if payload.len() > 14 && payload.starts_with(ICC_JPEG_HEADER) => {
                icc_chunks.push((payload[12], payload[14..].to_vec()));
            }
            0xED if preserve && payload.starts_with(PHOTOSHOP_JPEG_HEADER) => {
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
/// (iCCP, inflated) and, when `preserve` is set, EXIF
/// (eXIf) and XMP (the iTXt whose keyword is
/// `XML:com.adobe.xmp`). Only the first chunk of each kind
/// is read (the spec allows one iCCP and one eXIf, and
/// libpng ignores repeats of either): later ones are
/// skipped unread, so a source cannot repeat a compressed
/// chunk to multiply the inflate work (issue #399). Every
/// inflate is charged to `inflate_budget`, the bytes of
/// zlib output this source may still produce.
fn extract_png_meta(bytes: &[u8], preserve: bool, inflate_budget: &mut usize) -> SourceMeta {
    let mut meta = SourceMeta::default();
    let (mut seen_iccp, mut seen_exif, mut seen_xmp) = (false, false, false);
    let mut pos = 8; // past the signature
    while pos + 12 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let kind = &bytes[pos + 4..pos + 8];
        let Some(data) = bytes.get(pos + 8..pos + 8 + len) else {
            break;
        };
        match kind {
            b"iCCP" if !seen_iccp => {
                seen_iccp = true;
                meta.icc = parse_iccp(data, inflate_budget);
            }
            b"eXIf" if preserve && !seen_exif => {
                seen_exif = true;
                let mut tiff = data.to_vec();
                patch_tiff_orientation(&mut tiff);
                meta.exif_tiff = Some(tiff);
            }
            b"iTXt" if preserve && !seen_xmp && data.starts_with(b"XML:com.adobe.xmp\0") => {
                seen_xmp = true;
                meta.xmp = parse_itxt_text(data, inflate_budget);
            }
            b"IDAT" => break,
            _ => {}
        }
        pos += 12 + len;
    }
    meta
}

/// Inflate a zlib stream whose output may be at most `cap`
/// bytes, charging the output to `budget`. None when the
/// stream is corrupt or inflates past `cap` or past what
/// is left of the budget; inflating stops one byte past
/// the smaller of the two either way.
fn inflate_capped(stream: &[u8], cap: usize, budget: &mut usize) -> Option<Vec<u8>> {
    use std::io::Read as _;
    let limit = cap.min(*budget);
    let mut out = Vec::new();
    let read = flate2::read::ZlibDecoder::new(stream)
        .take(limit as u64 + 1)
        .read_to_end(&mut out);
    // The byte past the limit only detects the overflow and
    // is not charged: a refused stream must not eat into
    // the cap of the next one, or the profile (which drives
    // the pixels) would depend on whether an XMP packet
    // ahead of it was read, that is, on the policy.
    *budget -= out.len().min(limit);
    read.ok()?;
    (out.len() <= limit).then_some(out)
}

/// The profile inside an iCCP chunk payload: profile name,
/// a compression byte (0 = zlib), then the stream.
fn parse_iccp(data: &[u8], inflate_budget: &mut usize) -> Option<Vec<u8>> {
    let nul = data.iter().position(|&b| b == 0)?;
    if data.get(nul + 1) != Some(&0) {
        return None;
    }
    let stream = &data[nul + 2..];
    inflate_capped(stream, MAX_ICC_BYTES, inflate_budget).filter(|out| !out.is_empty())
}

/// The text of an iTXt chunk: keyword, compression flag and
/// method, language and translated keywords (each
/// null-terminated), then the (possibly compressed) text.
fn parse_itxt_text(data: &[u8], inflate_budget: &mut usize) -> Option<Vec<u8>> {
    // Every cursor step is bounds-checked (issue #372):
    // a chunk shorter than the fixed fields it is walked
    // through yields no text, never a slice panic.
    let mut cursor = 0;
    let skip_cstr = |data: &[u8], cursor: &mut usize| -> Option<()> {
        let nul = data.get(*cursor..)?.iter().position(|&b| b == 0)?;
        *cursor += nul + 1;
        Some(())
    };
    skip_cstr(data, &mut cursor)?; // keyword
    let compressed = *data.get(cursor)?;
    cursor += 2; // flag + method
    if cursor > data.len() {
        return None;
    }
    skip_cstr(data, &mut cursor)?; // language
    skip_cstr(data, &mut cursor)?; // translated keyword
    let text = data.get(cursor..)?;
    if compressed == 0 {
        (text.len() <= MAX_META_BYTES).then(|| text.to_vec())
    } else {
        inflate_capped(text, MAX_META_BYTES, inflate_budget)
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
        let got = extract_jpeg_meta(&out, true);
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
        let got = extract_jpeg_meta(&out, true);
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
        let got = extract_source_meta(&out, true);
        assert_eq!(got.icc, meta_in.icc);
        assert_eq!(got.xmp, meta_in.xmp);
        assert!(got.exif_tiff.is_some());
    }

    #[test]
    fn short_itxt_chunks_yield_no_text_instead_of_panicking() {
        // Issue #372: a chunk that ends inside the fixed
        // fields (keyword, compression flag and method,
        // language, translated keyword) walked the cursor
        // past the end of the data and panicked the slice.
        // Every truncation of a well-formed payload must
        // answer None; the full payload still parses.
        let mut full = b"XML:com.adobe.xmp\0".to_vec();
        full.extend_from_slice(&[0, 0, 0, 0]);
        full.extend_from_slice(b"<x:xmpmeta/>");
        let mut budget = MAX_INFLATE_BYTES;
        assert_eq!(
            parse_itxt_text(&full, &mut budget),
            Some(b"<x:xmpmeta/>".to_vec())
        );
        // The fixed fields end at byte 22; every shorter
        // truncation answers None.
        for len in 0..22 {
            assert_eq!(
                parse_itxt_text(&full[..len], &mut budget),
                None,
                "len {len}"
            );
        }
        // End to end: the 19-byte chunk (keyword plus the
        // compression flag, no method byte) inside a PNG
        // extracts as no XMP, not a panic.
        let mut src = b"\x89PNG\r\n\x1a\n".to_vec();
        src.extend_from_slice(&png_chunk(
            b"IHDR",
            &[0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0],
        ));
        let mut chunk = b"XML:com.adobe.xmp\0".to_vec();
        chunk.push(0);
        assert_eq!(chunk.len(), 19);
        src.extend_from_slice(&png_chunk(b"iTXt", &chunk));
        src.extend_from_slice(&png_chunk(b"IEND", &[]));
        assert!(extract_source_meta(&src, true).xmp.is_none());
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

    fn one_pixel(r: u8, g: u8, b: u8) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(1, 1, image::Rgb([r, g, b])))
    }

    #[test]
    fn a_cmyk_profile_is_not_honored_for_rgb_pixels() {
        // The profile family decides, not the pixels: a
        // CMYK source profile on RGB pixels leaves the
        // image unchanged (the CMYK route belongs to the
        // decoder). The fixture is a real CMYK-family
        // profile, so the no-op pins the colour-family
        // branch rather than a parse failure.
        // (moxcms will not *encode* a mismatched profile,
        // so the fixture patches the data colour space
        // signature of an encoded profile to CMYK, the way
        // a real CMYK profile arrives: as bytes.)
        let mut icc = moxcms::ColorProfile::new_srgb().encode().unwrap();
        icc[16..20].copy_from_slice(b"CMYK");
        let parsed = moxcms::ColorProfile::new_from_slice(&icc).unwrap();
        assert_eq!(parsed.color_space, moxcms::DataColorSpace::Cmyk);
        let out = convert_to_srgb(one_pixel(200, 80, 40), &icc).to_rgb8();
        assert_eq!(out.get_pixel(0, 0).0, [200, 80, 40]);
    }

    #[test]
    fn a_grayscale_profile_is_not_honored() {
        // Same family rule for grayscale profiles: parsed
        // fine, declined by family, pixels unchanged, and a
        // grayscale image itself passes through untouched.
        let icc = moxcms::ColorProfile::new_gray_with_gamma(2.2)
            .encode()
            .unwrap();
        let parsed = moxcms::ColorProfile::new_from_slice(&icc).unwrap();
        assert_eq!(parsed.color_space, moxcms::DataColorSpace::Gray);
        let out = convert_to_srgb(one_pixel(90, 160, 90), &icc).to_rgb8();
        assert_eq!(out.get_pixel(0, 0).0, [90, 160, 90]);
        let gray = DynamicImage::ImageLuma8(image::GrayImage::from_pixel(1, 1, image::Luma([77])));
        let out = convert_to_srgb(gray, &icc).to_luma8();
        assert_eq!(out.get_pixel(0, 0).0, [77]);
    }

    #[test]
    fn an_unparseable_profile_leaves_the_image_unchanged() {
        let out = convert_to_srgb(one_pixel(10, 20, 30), b"not a profile at all").to_rgb8();
        assert_eq!(out.get_pixel(0, 0).0, [10, 20, 30]);
    }

    #[test]
    fn webp_bytes_pass_through_both_injectors_unchanged() {
        // Preserved metadata (and the ICC) are injected for
        // JPEG and PNG outputs only. WebP output bytes are
        // not a shape either injector recognizes, so a
        // preserved WebP result carries none of it.
        let webp: &[u8] = b"RIFF\x1a\x00\x00\x00WEBPVP8L\x0d\x00\x00\x00\x2f\x00\x00\x00\x00\x07\x10\x11\x11\x88\x07\xa9\xbb\xee\x01\x00";
        let meta = SourceMeta {
            icc: Some(b"fake-profile-bytes".to_vec()),
            exif_tiff: Some(tiff_with_orientation(6, true)),
            xmp: Some(b"<x:xmpmeta/>".to_vec()),
            photoshop: Some(b"Photoshop 3.0\0rest".to_vec()),
        };
        assert_eq!(inject_jpeg(webp, meta.icc.as_deref(), &meta, true), webp);
        assert_eq!(inject_png(webp, meta.icc.as_deref(), &meta, true), webp);
    }

    #[test]
    fn malformed_tiff_streams_are_left_untouched() {
        let mut cases: Vec<(&str, Vec<u8>)> = vec![
            ("empty", Vec::new()),
            ("one byte", vec![b'I']),
            ("byte order only", b"II".to_vec()),
        ];
        // Wrong magic where 42 belongs.
        let mut bad_magic = b"II".to_vec();
        bad_magic.extend_from_slice(&43u16.to_le_bytes());
        bad_magic.extend_from_slice(&8u32.to_le_bytes());
        bad_magic.extend_from_slice(&0u16.to_le_bytes());
        cases.push(("wrong magic", bad_magic));
        // The IFD offset points past the end of the stream.
        let mut bad_offset = b"II".to_vec();
        bad_offset.extend_from_slice(&42u16.to_le_bytes());
        bad_offset.extend_from_slice(&1000u32.to_le_bytes());
        cases.push(("ifd offset past the end", bad_offset));
        // The entry count promises entries the stream does
        // not contain.
        let mut lying_count = b"II".to_vec();
        lying_count.extend_from_slice(&42u16.to_le_bytes());
        lying_count.extend_from_slice(&8u32.to_le_bytes());
        lying_count.extend_from_slice(&5u16.to_le_bytes());
        cases.push(("entry count past the end", lying_count));
        // An orientation tag of the wrong type (LONG, not
        // SHORT) is not a patchable orientation.
        let mut wrong_type = tiff_with_orientation(6, true);
        wrong_type[8 + 2 + 2] = 4;
        wrong_type[8 + 2 + 3] = 0;
        cases.push(("orientation with the wrong type", wrong_type));
        // The entry header is complete but the stream ends
        // inside the value field the patch would rewrite.
        let mut truncated_value = tiff_with_orientation(6, true);
        truncated_value.truncate(8 + 2 + 8 + 1);
        cases.push(("value field truncated", truncated_value));
        for (name, mut tiff) in cases {
            let before = tiff.clone();
            patch_tiff_orientation(&mut tiff);
            assert_eq!(tiff, before, "{name}");
        }
    }

    #[test]
    fn malformed_exif_segments_extract_without_panicking() {
        // A truncated TIFF after the Exif header is carried
        // as-is: the orientation patch no-ops on it and the
        // bytes survive extraction untouched.
        let mut src = vec![0xFF, 0xD8];
        src.extend_from_slice(&jpeg_segment(0xE1, b"Exif\0\0II").unwrap());
        src.extend_from_slice(&[0xFF, 0xD9]);
        let meta = extract_jpeg_meta(&src, true);
        assert_eq!(meta.exif_tiff.as_deref(), Some(b"II".as_slice()));
        // A segment whose declared length runs past the end
        // of the data ends the walk: nothing is extracted.
        let mut src = vec![0xFF, 0xD8, 0xFF, 0xE1, 0x10, 0x00];
        src.extend_from_slice(b"Exif\0\0II*\0");
        let meta = extract_jpeg_meta(&src, true);
        assert!(meta.exif_tiff.is_none() && meta.icc.is_none());
        // Even a bare SOI (or nothing) extracts to empty.
        assert_eq!(
            extract_source_meta(&[0xFF, 0xD8], true),
            SourceMeta::default()
        );
        assert_eq!(extract_source_meta(&[], true), SourceMeta::default());
        // PNG: a truncated eXIf TIFF is carried the same
        // way, and a corrupt iCCP profile (not zlib) drops
        // to no profile instead of failing.
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&png_chunk(
            b"IHDR",
            &[0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0],
        ));
        png.extend_from_slice(&png_chunk(b"eXIf", b"MM"));
        png.extend_from_slice(&png_chunk(b"iCCP", b"name\0\0not-zlib"));
        png.extend_from_slice(&png_chunk(b"IEND", &[]));
        let meta = extract_source_meta(&png, true);
        assert_eq!(meta.exif_tiff.as_deref(), Some(b"MM".as_slice()));
        assert!(meta.icc.is_none());
    }

    /// A zlib stream of `len` zero bytes: about a
    /// thousandth of `len` on the wire, `len` once inflated.
    fn zlib_zeros(len: usize) -> Vec<u8> {
        use std::io::Write as _;
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
        let block = vec![0u8; 1024 * 1024];
        let mut left = len;
        while left > 0 {
            let n = left.min(block.len());
            encoder.write_all(&block[..n]).unwrap();
            left -= n;
        }
        encoder.finish().unwrap()
    }

    /// A 1x1 PNG with `chunks` between IHDR and IDAT.
    fn png_with_chunks(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&png_chunk(
            b"IHDR",
            &[0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0],
        ));
        for (kind, data) in chunks {
            png.extend_from_slice(&png_chunk(kind, data));
        }
        png.extend_from_slice(&png_chunk(b"IDAT", &zlib_zeros(4)));
        png.extend_from_slice(&png_chunk(b"IEND", &[]));
        png
    }

    fn iccp_payload(stream: &[u8]) -> Vec<u8> {
        let mut data = b"ICC Profile\0\0".to_vec();
        data.extend_from_slice(stream);
        data
    }

    fn compressed_xmp_payload(stream: &[u8]) -> Vec<u8> {
        let mut data = b"XML:com.adobe.xmp\0".to_vec();
        data.extend_from_slice(&[1, 0, 0, 0]); // compressed, zlib, no language/translation
        data.extend_from_slice(stream);
        data
    }

    #[test]
    fn repeated_metadata_bombs_inflate_once_per_kind() {
        // Issue #399: every iCCP and every XMP iTXt before
        // IDAT used to be inflated to its own cap, so 40 of
        // each (a 700 KB file) cost 40 x 20 MiB of inflate.
        // Only the first of each kind is read now, and the
        // whole walk stays inside the per-source budget.
        let iccp = iccp_payload(&zlib_zeros(MAX_ICC_BYTES));
        let itxt = compressed_xmp_payload(&zlib_zeros(MAX_META_BYTES));
        let mut chunks: Vec<(&[u8; 4], &[u8])> = Vec::new();
        for _ in 0..40 {
            chunks.push((b"iCCP", &iccp));
            chunks.push((b"iTXt", &itxt));
        }
        let png = png_with_chunks(&chunks);
        // A small fraction of the default 10 MiB fetch cap.
        assert!(png.len() < 1024 * 1024, "{} bytes", png.len());
        // Policy keeps metadata: one profile, one packet.
        let mut budget = MAX_INFLATE_BYTES;
        let meta = extract_png_meta(&png, true, &mut budget);
        assert_eq!(MAX_INFLATE_BYTES - budget, MAX_ICC_BYTES + MAX_META_BYTES);
        assert_eq!(meta.icc.map(|icc| icc.len()), Some(MAX_ICC_BYTES));
        assert_eq!(meta.xmp.map(|xmp| xmp.len()), Some(MAX_META_BYTES));
        // Default policy (strip): the XMP is never
        // inflated; only the first profile is.
        let mut budget = MAX_INFLATE_BYTES;
        let meta = extract_png_meta(&png, false, &mut budget);
        assert_eq!(MAX_INFLATE_BYTES - budget, MAX_ICC_BYTES);
        assert!(meta.xmp.is_none());
    }

    #[test]
    fn the_inflate_budget_is_shared_across_chunks() {
        // One budget spans every chunk of a source: what
        // the profile inflates is no longer available to
        // the XMP packet, and an exhausted budget refuses
        // the next stream after at most one byte.
        let xmp = b"<x:xmpmeta>budget</x:xmpmeta>";
        let xmp_stream = {
            use std::io::Write as _;
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(xmp).unwrap();
            encoder.finish().unwrap()
        };
        let png = png_with_chunks(&[
            (b"iCCP", &iccp_payload(&zlib_zeros(1000))),
            (b"iTXt", &compressed_xmp_payload(&xmp_stream)),
        ]);
        // Room for both.
        let mut budget = 1000 + xmp.len();
        let meta = extract_png_meta(&png, true, &mut budget);
        assert_eq!(meta.icc.map(|icc| icc.len()), Some(1000));
        assert_eq!(meta.xmp.as_deref(), Some(xmp.as_slice()));
        assert_eq!(budget, 0);
        // The profile leaves too little for the packet.
        let mut budget = 1000 + xmp.len() - 1;
        let meta = extract_png_meta(&png, true, &mut budget);
        assert!(meta.icc.is_some());
        assert!(meta.xmp.is_none());
        assert_eq!(budget, 0);
        // Too little for the profile: refused one byte
        // past the budget, and nothing is left for the
        // packet.
        let mut budget = 999;
        let meta = extract_png_meta(&png, true, &mut budget);
        assert!(meta.icc.is_none() && meta.xmp.is_none());
        assert_eq!(budget, 0);
    }

    #[test]
    fn only_the_first_chunk_of_each_kind_is_read() {
        // As in libpng, the first iCCP is the profile and
        // later ones are ignored, even when the first is
        // corrupt; the same holds for eXIf and the XMP
        // packet.
        let profile = |byte: u8| {
            use std::io::Write as _;
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(&[byte; 64]).unwrap();
            iccp_payload(&encoder.finish().unwrap())
        };
        let mut xmp_first = b"XML:com.adobe.xmp\0\0\0\0\0".to_vec();
        xmp_first.extend_from_slice(b"<first/>");
        let mut xmp_second = b"XML:com.adobe.xmp\0\0\0\0\0".to_vec();
        xmp_second.extend_from_slice(b"<second/>");
        let png = png_with_chunks(&[
            (b"iCCP", &profile(1)),
            (b"eXIf", b"MM-first"),
            (b"iTXt", &xmp_first),
            (b"iCCP", &profile(2)),
            (b"eXIf", b"MM-second"),
            (b"iTXt", &xmp_second),
        ]);
        let meta = extract_source_meta(&png, true);
        assert_eq!(meta.icc, Some(vec![1u8; 64]));
        assert_eq!(meta.exif_tiff.as_deref(), Some(b"MM-first".as_slice()));
        assert_eq!(meta.xmp.as_deref(), Some(b"<first/>".as_slice()));
        let png = png_with_chunks(&[(b"iCCP", b"name\0\0not-zlib"), (b"iCCP", &profile(2))]);
        assert!(extract_source_meta(&png, true).icc.is_none());
    }

    #[test]
    fn metadata_is_read_only_when_the_policy_keeps_it() {
        // The profile drives color conversion and is always
        // read; EXIF, XMP, and the Photoshop segment are
        // only read when the policy will copy them.
        let meta_in = SourceMeta {
            icc: Some(b"fake-profile-bytes".to_vec()),
            exif_tiff: Some(tiff_with_orientation(3, true)),
            xmp: Some(b"<x:xmpmeta/>".to_vec()),
            photoshop: Some(b"Photoshop 3.0\0fixture".to_vec()),
        };
        let png = inject_png(
            &png_with_chunks(&[]),
            meta_in.icc.as_deref(),
            &meta_in,
            true,
        );
        let jpeg = inject_jpeg(
            &[0xFF, 0xD8, 0xFF, 0xD9],
            meta_in.icc.as_deref(),
            &meta_in,
            true,
        );
        for (name, src) in [("png", &png), ("jpeg", &jpeg)] {
            let kept = extract_source_meta(src, true);
            assert_eq!(kept.icc, meta_in.icc, "{name}");
            assert!(kept.exif_tiff.is_some() && kept.xmp.is_some(), "{name}");
            let stripped = extract_source_meta(src, false);
            assert_eq!(stripped.icc, meta_in.icc, "{name}");
            assert_eq!(
                (stripped.exif_tiff, stripped.xmp, stripped.photoshop),
                (None, None, None),
                "{name}"
            );
        }
        assert!(extract_source_meta(&jpeg, true).photoshop.is_some());
    }

    #[test]
    fn an_xmp_packet_past_its_cap_is_dropped_not_truncated() {
        // The packet used to come back as the first
        // MAX_META_BYTES + 1 bytes of the stream, a
        // truncated document carried into the output.
        let itxt = compressed_xmp_payload(&zlib_zeros(MAX_META_BYTES + 1));
        let png = png_with_chunks(&[(b"iTXt", &itxt)]);
        let mut budget = MAX_INFLATE_BYTES;
        assert!(extract_png_meta(&png, true, &mut budget).xmp.is_none());
        // Charged up to the cap; the byte that detected the
        // overflow is not.
        assert_eq!(MAX_INFLATE_BYTES - budget, MAX_META_BYTES);
    }

    #[test]
    fn the_profile_does_not_depend_on_the_metadata_policy() {
        // The profile drives the pixels, so both policies
        // must read the same one. An over-cap XMP packet
        // ahead of the iCCP chunk (iTXt may come anywhere
        // before IDAT) is read only under preserve; it must
        // not leave a profile at its cap short of budget.
        let itxt = compressed_xmp_payload(&zlib_zeros(MAX_META_BYTES + 1));
        let iccp = iccp_payload(&zlib_zeros(MAX_ICC_BYTES));
        let png = png_with_chunks(&[(b"iTXt", &itxt), (b"iCCP", &iccp)]);
        let stripped = extract_source_meta(&png, false);
        assert_eq!(stripped.icc.as_ref().map(Vec::len), Some(MAX_ICC_BYTES));
        let kept = extract_source_meta(&png, true);
        assert!(kept.xmp.is_none());
        assert_eq!(kept.icc.as_ref().map(Vec::len), Some(MAX_ICC_BYTES));
        assert!(kept.icc == stripped.icc);
    }
}
