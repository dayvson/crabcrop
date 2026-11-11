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
/// were rotated at decode, and its thumbnails removed), the
/// XMP packet, and the JPEG Photoshop segment (home of
/// IPTC, its thumbnails removed), when present.
/// `xmp_compressed` records that the source stored the
/// packet zlib-compressed (a PNG `iTXt` can), so a PNG
/// output stores it the same way (issue #445).
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SourceMeta {
    pub icc: Option<Vec<u8>>,
    pub exif_tiff: Option<Vec<u8>>,
    pub xmp: Option<Vec<u8>>,
    pub xmp_compressed: bool,
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
/// The largest XMP packet read from a source, and so the
/// largest one written back out (issue #445). The packet is
/// read only to be copied, a PNG can store it compressed,
/// and a JPEG output already drops anything past one 64 KiB
/// segment; a larger packet is dropped, never truncated.
pub(crate) const MAX_XMP_BYTES: usize = 256 * 1024;
/// The largest ICC profile embedded into an output (issue
/// #445). JPEG carries the profile raw, so a source profile
/// that compresses a thousandfold would otherwise come back
/// a thousand times larger. Working-space and camera
/// profiles are a few KB; a profile past this cap still
/// drives the conversion to sRGB, it is just not embedded.
pub(crate) const MAX_EMBED_ICC_BYTES: usize = 256 * 1024;
/// The most zlib output one source may inflate across all
/// of its metadata chunks (issue #399): one ICC profile and
/// one XMP packet, each at its own cap. The caps alone are
/// per chunk, and 4 MiB of zeros deflates to about 4 KB,
/// so a PNG under the fetch cap could otherwise carry
/// hundreds of chunks that each inflate to the cap.
pub(crate) const MAX_INFLATE_BYTES: usize = MAX_ICC_BYTES + MAX_XMP_BYTES;

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
                sanitize_preserved_exif(&mut tiff);
                meta.exif_tiff = Some(tiff);
            }
            0xE1 if preserve && payload.starts_with(XMP_JPEG_HEADER) => {
                meta.xmp = Some(payload[XMP_JPEG_HEADER.len()..].to_vec());
            }
            0xE2 if payload.len() > 14 && payload.starts_with(ICC_JPEG_HEADER) => {
                icc_chunks.push((payload[12], payload[14..].to_vec()));
            }
            0xED if preserve && payload.starts_with(PHOTOSHOP_JPEG_HEADER) => {
                meta.photoshop = strip_photoshop_thumbnails(payload);
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
                sanitize_preserved_exif(&mut tiff);
                meta.exif_tiff = Some(tiff);
            }
            b"iTXt" if preserve && !seen_xmp && data.starts_with(b"XML:com.adobe.xmp\0") => {
                seen_xmp = true;
                if let Some((text, compressed)) = parse_itxt_text(data, inflate_budget) {
                    meta.xmp = Some(text);
                    meta.xmp_compressed = compressed;
                }
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

/// The text of an iTXt chunk, and whether it was stored
/// compressed: keyword, compression flag and method,
/// language and translated keywords (each
/// null-terminated), then the (possibly compressed) text.
fn parse_itxt_text(data: &[u8], inflate_budget: &mut usize) -> Option<(Vec<u8>, bool)> {
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
        (text.len() <= MAX_XMP_BYTES).then(|| (text.to_vec(), false))
    } else {
        inflate_capped(text, MAX_XMP_BYTES, inflate_budget).map(|text| (text, true))
    }
}

/// Byte-order-aware reads and writes of a TIFF stream, the
/// container EXIF travels in. Every read and write is
/// bounds-checked: a read past the stream is `None`, a
/// write past it does nothing.
#[derive(Clone, Copy)]
struct TiffOrder {
    little: bool,
}

/// One IFD entry: where it starts, its tag, field type and
/// value count.
struct IfdEntry {
    pos: usize,
    tag: u16,
    kind: u16,
    count: u32,
}

/// The size of one value of a TIFF field type, or `None` for
/// a type TIFF and EXIF do not define.
fn tiff_type_size(kind: u16) -> Option<usize> {
    match kind {
        1 | 2 | 6 | 7 => Some(1),   // BYTE, ASCII, SBYTE, UNDEFINED
        3 | 8 => Some(2),           // SHORT, SSHORT
        4 | 9 | 11 | 13 => Some(4), // LONG, SLONG, FLOAT, IFD
        5 | 10 | 12 => Some(8),     // RATIONAL, SRATIONAL, DOUBLE
        _ => None,
    }
}

impl TiffOrder {
    /// The byte order of a stream that opens with a TIFF
    /// header (`II` or `MM`, then 42).
    fn of(tiff: &[u8]) -> Option<Self> {
        let little = match tiff.get(0..2)? {
            b"II" => true,
            b"MM" => false,
            _ => return None,
        };
        let order = TiffOrder { little };
        (order.u16(tiff, 2) == Some(42)).then_some(order)
    }

    fn u16(self, tiff: &[u8], off: usize) -> Option<u16> {
        let raw: [u8; 2] = tiff.get(off..off.checked_add(2)?)?.try_into().ok()?;
        Some(if self.little {
            u16::from_le_bytes(raw)
        } else {
            u16::from_be_bytes(raw)
        })
    }

    fn u32(self, tiff: &[u8], off: usize) -> Option<u32> {
        let raw: [u8; 4] = tiff.get(off..off.checked_add(4)?)?.try_into().ok()?;
        Some(if self.little {
            u32::from_le_bytes(raw)
        } else {
            u32::from_be_bytes(raw)
        })
    }

    fn put_u16(self, tiff: &mut [u8], off: usize, v: u16) {
        let bytes = if self.little {
            v.to_le_bytes()
        } else {
            v.to_be_bytes()
        };
        if let Some(slot) = off.checked_add(2).and_then(|end| tiff.get_mut(off..end)) {
            slot.copy_from_slice(&bytes);
        }
    }

    fn put_u32(self, tiff: &mut [u8], off: usize, v: u32) {
        let bytes = if self.little {
            v.to_le_bytes()
        } else {
            v.to_be_bytes()
        };
        if let Some(slot) = off.checked_add(4).and_then(|end| tiff.get_mut(off..end)) {
            slot.copy_from_slice(&bytes);
        }
    }

    /// The offset of IFD0, which lies past the header.
    fn ifd0(self, tiff: &[u8]) -> Option<usize> {
        self.u32(tiff, 4)
            .map(|v| v as usize)
            .filter(|&ifd0| ifd0 >= TIFF_HEADER_BYTES)
    }

    /// The entries of the IFD at `ifd`, up to the first one
    /// the stream does not hold whole.
    fn entries(self, tiff: &[u8], ifd: usize) -> Vec<IfdEntry> {
        let count = self.u16(tiff, ifd).unwrap_or(0) as usize;
        (0..count)
            .map_while(|i| {
                let pos = ifd.checked_add(2 + 12 * i)?;
                tiff.get(pos..pos.checked_add(12)?)?;
                Some(IfdEntry {
                    pos,
                    tag: self.u16(tiff, pos)?,
                    kind: self.u16(tiff, pos + 2)?,
                    count: self.u32(tiff, pos + 4)?,
                })
            })
            .collect()
    }

    /// Where the IFD at `ifd` keeps its next-IFD offset, when
    /// the stream holds its whole table.
    fn next_field(self, tiff: &[u8], ifd: usize) -> Option<usize> {
        let count = self.u16(tiff, ifd)? as usize;
        let field = ifd.checked_add(2 + 12 * count)?;
        self.u32(tiff, field).map(|_| field)
    }

    /// Where an entry's values are, when they do not fit in
    /// the entry itself: the range its value field points
    /// at. `None` for inline values and unknown types.
    fn out_of_line(self, tiff: &[u8], e: &IfdEntry) -> Option<std::ops::Range<usize>> {
        let len = tiff_type_size(e.kind)?.checked_mul(e.count as usize)?;
        if len <= 4 {
            return None;
        }
        let at = self.u32(tiff, e.pos + 8)? as usize;
        Some(at..at.saturating_add(len))
    }

    /// The first `limit` SHORT or LONG values of an entry,
    /// as far as the stream holds them.
    fn values(self, tiff: &[u8], e: &IfdEntry, limit: usize) -> Vec<usize> {
        let size: usize = match e.kind {
            3 => 2,
            4 => 4,
            _ => return Vec::new(),
        };
        let at = match size.checked_mul(e.count as usize) {
            Some(len) if len <= 4 => e.pos + 8,
            Some(_) => match self.u32(tiff, e.pos + 8) {
                Some(at) => at as usize,
                None => return Vec::new(),
            },
            None => return Vec::new(),
        };
        (0..(e.count as usize).min(limit))
            .map_while(|i| {
                let off = at.checked_add(i * size)?;
                if size == 2 {
                    self.u16(tiff, off).map(usize::from)
                } else {
                    self.u32(tiff, off).map(|v| v as usize)
                }
            })
            .collect()
    }
}

/// Rewrite the Orientation tag (0x0112) of a TIFF stream to
/// 1, in place. The pixels a preserved EXIF describes were
/// already rotated at decode; copying the original value
/// would make every viewer rotate them a second time. Any
/// inconsistency leaves the stream untouched.
pub fn patch_tiff_orientation(tiff: &mut [u8]) {
    let Some(order) = TiffOrder::of(tiff) else {
        return;
    };
    let Some(ifd0) = order.ifd0(tiff) else {
        return;
    };
    let orientation = order
        .entries(tiff, ifd0)
        .into_iter()
        .find(|e| e.tag == 0x0112 && e.kind == 3 && e.count == 1);
    // The value sits in the first two bytes of the
    // four-byte value field, in stream endianness.
    if let Some(e) = orientation {
        order.put_u16(tiff, e.pos + 8, 1);
    }
}

/// Prepare a source's EXIF for a preserved copy: the
/// orientation goes to 1 ([`patch_tiff_orientation`]) and
/// the previews go ([`strip_tiff_thumbnails`]). The JPEG,
/// PNG and HEIC paths all call this.
pub fn sanitize_preserved_exif(tiff: &mut Vec<u8>) {
    patch_tiff_orientation(tiff);
    strip_tiff_thumbnails(tiff);
}

/// The most IFDs followed past IFD0. EXIF defines one, IFD1,
/// the thumbnail's; the bound keeps a long chain cheap.
const MAX_CHAINED_IFDS: usize = 8;

/// The most strips or tiles read from one IFD. A thumbnail
/// has one strip or a few; the bound keeps the walk's work
/// and memory small whatever the counts claim.
const MAX_IMAGE_PARTS: usize = 1024;

/// The TIFF header, which the sanitizer never blanks or
/// cuts.
const TIFF_HEADER_BYTES: usize = 8;

/// Remove the previews a preserved EXIF stream carries
/// (issue #611): small copies of the whole uncut source,
/// which a crop or a redaction must not ship. In place, and
/// without rebuilding the stream, so the offsets that IFD0,
/// its sub-IFDs and maker notes hold stay valid, and a
/// stream with no preview comes out byte for byte:
/// - every IFD chained after IFD0 (IFD1, the thumbnail's)
///   is zeroed with its values and its image data (a JPEG,
///   or uncompressed strips or tiles), and IFD0's link to
///   it is set to 0;
/// - any JPEG left in the stream is zeroed from its SOI to
///   its EOI: maker notes carry previews too;
/// - zeroed bytes at the end of the stream, where cameras
///   put the thumbnail, are cut off.
///
/// A stream that is not TIFF gets the JPEG pass only.
pub fn strip_tiff_thumbnails(tiff: &mut Vec<u8>) {
    let mut blanked = Vec::new();
    if let Some(order) = TiffOrder::of(tiff) {
        blanked = chained_ifd_ranges(order, tiff);
        if let Some(link) = order
            .ifd0(tiff)
            .and_then(|ifd0| order.next_field(tiff, ifd0))
        {
            order.put_u32(tiff, link, 0);
        }
    }
    for range in &mut blanked {
        range.end = range.end.min(tiff.len());
        range.start = range
            .start
            .clamp(TIFF_HEADER_BYTES, range.end.max(TIFF_HEADER_BYTES));
        if let Some(bytes) = tiff.get_mut(range.clone()) {
            bytes.fill(0);
        }
    }
    blank_embedded_jpegs(tiff, &mut blanked);
    cut_blanked_tail(tiff, blanked);
}

/// The bytes of every IFD chained after IFD0: each one's
/// table, its out-of-line values, and the image data it
/// points at. The walk stops at a loop, at an IFD the
/// stream does not hold, and after `MAX_CHAINED_IFDS`.
fn chained_ifd_ranges(order: TiffOrder, tiff: &[u8]) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let Some(ifd0) = order.ifd0(tiff) else {
        return ranges;
    };
    let mut seen = vec![ifd0];
    let mut next = order
        .next_field(tiff, ifd0)
        .and_then(|link| order.u32(tiff, link))
        .unwrap_or(0) as usize;
    while next != 0 && seen.len() <= MAX_CHAINED_IFDS && !seen.contains(&next) {
        let ifd = next;
        seen.push(ifd);
        let Some(count) = order.u16(tiff, ifd) else {
            break;
        };
        ranges.push(ifd..ifd.saturating_add(2 + 12 * count as usize + 4));
        let entries = order.entries(tiff, ifd);
        let value = |tag: u16| {
            entries
                .iter()
                .find(|e| e.tag == tag)
                .map(|e| order.values(tiff, e, MAX_IMAGE_PARTS))
                .unwrap_or_default()
        };
        // JPEGInterchangeFormat and its length; StripOffsets
        // and StripByteCounts; TileOffsets and TileByteCounts.
        for (offsets, lengths) in [(0x0201, 0x0202), (0x0111, 0x0117), (0x0144, 0x0145)] {
            for (at, len) in value(offsets).into_iter().zip(value(lengths)) {
                ranges.push(at..at.saturating_add(len));
            }
        }
        ranges.extend(entries.iter().filter_map(|e| order.out_of_line(tiff, e)));
        next = order
            .next_field(tiff, ifd)
            .and_then(|link| order.u32(tiff, link))
            .unwrap_or(0) as usize;
    }
    ranges
}

/// The most ranges the JPEG pass records for the tail cut.
/// Past it, a JPEG is still zeroed, just not cut; real
/// streams carry one or two.
const MAX_RECORDED_JPEGS: usize = 1024;

/// Zero every JPEG embedded in `bytes`, recording the ranges
/// in `blanked`. A run that opens like a JPEG but reaches no
/// scan is not an image and stays. The search resumes past
/// each run it walked, so the pass stays linear in the
/// stream.
fn blank_embedded_jpegs(bytes: &mut [u8], blanked: &mut Vec<std::ops::Range<usize>>) {
    let mut pos = 0;
    let mut recorded = 0;
    while let Some(found) = bytes
        .get(pos..)
        .and_then(|rest| rest.windows(3).position(|w| w == [0xFF, 0xD8, 0xFF]))
    {
        let start = pos + found;
        let (len, image) = jpeg_extent(&bytes[start..]);
        if image {
            bytes[start..start + len].fill(0);
            if recorded < MAX_RECORDED_JPEGS {
                blanked.push(start..start + len);
                recorded += 1;
            }
        }
        pos = start + len.max(2);
    }
}

/// How far the JPEG at the start of `b` (at its SOI) runs,
/// walked the way a decoder walks it: segment by segment,
/// through the entropy-coded data after each SOS, to the
/// EOI. Also whether it reached a scan, that is, whether it
/// is an image. A JPEG cut short runs to the end of `b`; one
/// whose structure breaks ends there.
fn jpeg_extent(b: &[u8]) -> (usize, bool) {
    let mut pos = 2;
    let mut scanned = false;
    loop {
        if b.get(pos) != Some(&0xFF) {
            return (pos.min(b.len()), scanned);
        }
        // A marker: 0xFF, any fill bytes, then its code.
        while b.get(pos) == Some(&0xFF) {
            pos += 1;
        }
        let Some(&code) = b.get(pos) else {
            return (b.len(), scanned);
        };
        pos += 1;
        match code {
            0xD9 => return (pos, scanned), // EOI
            0x01 | 0xD0..=0xD7 => {}       // no length
            0x00 | 0xD8 => return (pos, scanned),
            _ => {
                let Some(len) = b.get(pos..pos + 2) else {
                    return (b.len(), scanned);
                };
                let len = u16::from_be_bytes([len[0], len[1]]) as usize;
                if len < 2 {
                    return (pos, scanned);
                }
                pos += len;
                if pos >= b.len() {
                    return (b.len(), scanned);
                }
                if code == 0xDA {
                    // Entropy-coded data runs to the next
                    // marker: an 0xFF not followed by a
                    // stuffed 0x00 or a restart code.
                    scanned = true;
                    loop {
                        let Some(ff) = b[pos..].iter().position(|&x| x == 0xFF) else {
                            return (b.len(), true);
                        };
                        pos += ff;
                        match b.get(pos + 1) {
                            None => return (b.len(), true),
                            Some(0x00 | 0xD0..=0xD7) => pos += 2,
                            Some(_) => break,
                        }
                    }
                }
            }
        }
    }
}

/// Cut the stream where its blanked tail starts: the zeros
/// a removed thumbnail leaves at the end carry nothing. The
/// header always stays.
fn cut_blanked_tail(tiff: &mut Vec<u8>, mut blanked: Vec<std::ops::Range<usize>>) {
    blanked.retain(|r| r.start < r.end);
    blanked.sort_unstable_by_key(|r| r.start);
    let mut last: Option<std::ops::Range<usize>> = None;
    for range in blanked {
        match &mut last {
            Some(merged) if range.start <= merged.end => merged.end = merged.end.max(range.end),
            _ => last = Some(range),
        }
    }
    if let Some(tail) = last
        && tail.end >= tiff.len()
    {
        tiff.truncate(tail.start.max(TIFF_HEADER_BYTES));
    }
}

/// Rewrite the size tags of a preserved EXIF stream to the
/// output's size, in place (issue #611): ImageWidth and
/// ImageLength in IFD0, PixelXDimension and
/// PixelYDimension in the Exif IFD. Copied as they were,
/// they give a crop the source's size. A SHORT too small
/// for the new value becomes a LONG. Tags of another type
/// or count, and a malformed stream, are left alone.
pub fn patch_tiff_dimensions(tiff: &mut [u8], width: u32, height: u32) {
    let Some(order) = TiffOrder::of(tiff) else {
        return;
    };
    let Some(ifd0) = order.ifd0(tiff) else {
        return;
    };
    let exif = order
        .entries(tiff, ifd0)
        .into_iter()
        .find(|e| e.tag == 0x8769 && matches!(e.kind, 4 | 13) && e.count == 1)
        .and_then(|e| order.u32(tiff, e.pos + 8));
    set_size_tags(order, tiff, ifd0, [(0x0100, width), (0x0101, height)]);
    if let Some(exif) = exif {
        set_size_tags(
            order,
            tiff,
            exif as usize,
            [(0xA002, width), (0xA003, height)],
        );
    }
}

fn set_size_tags(order: TiffOrder, tiff: &mut [u8], ifd: usize, tags: [(u16, u32); 2]) {
    for e in order.entries(tiff, ifd) {
        let Some(&(_, value)) = tags.iter().find(|(tag, _)| *tag == e.tag) else {
            continue;
        };
        match (e.kind, e.count) {
            (3, 1) if value <= u32::from(u16::MAX) => {
                order.put_u16(tiff, e.pos + 8, value as u16);
            }
            (3 | 4, 1) => {
                order.put_u16(tiff, e.pos + 2, 4);
                order.put_u32(tiff, e.pos + 8, value);
            }
            _ => {}
        }
    }
}

/// Photoshop image resources a preserved segment drops
/// (issue #611): the thumbnails (0x0409 from Photoshop 4,
/// 0x040C from Photoshop 5 on) and the copies of the EXIF
/// (0x0422, 0x0423), which would bring back its thumbnail
/// and the source's size.
const DROPPED_PHOTOSHOP_RESOURCES: [u16; 4] = [0x0409, 0x040C, 0x0422, 0x0423];

/// A Photoshop segment payload without the resources in
/// `DROPPED_PHOTOSHOP_RESOURCES`. The others are copied as
/// they were, in order, so a well-formed payload with
/// nothing to drop comes back byte for byte. The walk stops
/// at the first resource that does not parse and drops the
/// rest, since a thumbnail there could not be told apart.
/// `None` when no resource is left.
fn strip_photoshop_thumbnails(payload: &[u8]) -> Option<Vec<u8>> {
    let body = payload.strip_prefix(PHOTOSHOP_JPEG_HEADER)?;
    let mut kept = PHOTOSHOP_JPEG_HEADER.to_vec();
    let mut pos = 0;
    while let Some(len) = photoshop_resource_len(&body[pos..]) {
        let id = u16::from_be_bytes([body[pos + 4], body[pos + 5]]);
        if !DROPPED_PHOTOSHOP_RESOURCES.contains(&id) {
            kept.extend_from_slice(&body[pos..pos + len]);
        }
        pos += len;
    }
    (kept.len() > PHOTOSHOP_JPEG_HEADER.len()).then_some(kept)
}

/// The length of the Photoshop image resource at the start
/// of `b`: a signature, the id, a Pascal name padded to even
/// length, the data size, and the data padded to even (the
/// last resource may leave its pad off).
fn photoshop_resource_len(b: &[u8]) -> Option<usize> {
    if !matches!(
        b.get(0..4)?,
        b"8BIM" | b"MeSa" | b"PHUT" | b"AgHg" | b"DCSR"
    ) {
        return None;
    }
    let size_at = 6 + (1 + *b.get(6)? as usize).next_multiple_of(2);
    let size = u32::from_be_bytes(b.get(size_at..size_at + 4)?.try_into().ok()?) as usize;
    let end = (size_at + 4).checked_add(size)?;
    (end <= b.len()).then(|| (end + end % 2).min(b.len()))
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

/// Whether a profile may be embedded into an output (issue
/// #445): at most `MAX_EMBED_ICC_BYTES`, with an ICC header
/// whose size field matches the bytes, the `acsp`
/// signature in place, and room for the tag table the
/// header's tag count promises. The bytes come from the
/// source, so anything else is not copied out. Under
/// `keep_profile` the engine converts a profile this
/// refuses, as it does for outputs that cannot carry one.
pub fn icc_embeddable(icc: &[u8]) -> bool {
    const HEADER: usize = 128;
    if icc.len() < HEADER + 4 || icc.len() > MAX_EMBED_ICC_BYTES {
        return false;
    }
    let field = |at: usize| u32::from_be_bytes(icc[at..at + 4].try_into().unwrap()) as usize;
    let tag_table_end = field(HEADER)
        .checked_mul(12)
        .and_then(|tags| tags.checked_add(HEADER + 4));
    field(0) == icc.len()
        && &icc[36..40] == b"acsp"
        && tag_table_end.is_some_and(|end| end <= icc.len())
}

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
/// skipped), a profile `icc_embeddable` refuses is left
/// out, and a malformed input passes through.
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
    if let Some(profile) = icc.filter(|icc| icc_embeddable(icc)) {
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

/// `data` as a zlib stream, the encoding of PNG's
/// compressed chunks (`iCCP`, compressed `iTXt`).
fn zlib(data: &[u8]) -> Option<Vec<u8>> {
    use std::io::Write as _;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(data).ok()?;
    encoder.finish().ok()
}

/// Inject the ICC profile (`iCCP`) and preserved metadata
/// (`eXIf`, XMP `iTXt`) into encoded PNG bytes, right after
/// IHDR, where the spec wants them (before PLTE and IDAT).
/// Neither the profile nor the XMP packet weighs more than
/// it did in the source (issue #445): a source can store
/// either compressed, and writing it back raw amplified it
/// up to a thousandfold for a packet of padding. The
/// profile is always compressed (`iCCP` has no other
/// form); the packet is compressed when the source's was,
/// and otherwise stays plain text, the form XMP packet
/// scanners and some readers (ImageMagick 6 among them)
/// expect. A profile `icc_embeddable` refuses, or a packet
/// past `MAX_XMP_BYTES`, is left out.
pub fn inject_png(bytes: &[u8], icc: Option<&[u8]>, meta: &SourceMeta, preserve: bool) -> Vec<u8> {
    // Signature + IHDR chunk = 8 + 25 bytes; anything else
    // is not the encoder output we expect.
    if bytes.len() < 33 || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") || &bytes[12..16] != b"IHDR" {
        return bytes.to_vec();
    }
    let mut inserts: Vec<u8> = Vec::new();
    if let Some(profile) = icc.filter(|icc| icc_embeddable(icc))
        && let Some(compressed) = zlib(profile)
    {
        // iCCP layout: profile name NUL, compression method
        // (0, zlib), then the stream.
        let mut data = b"ICC Profile\0".to_vec();
        data.push(0);
        data.extend_from_slice(&compressed);
        inserts.extend_from_slice(&png_chunk(b"iCCP", &data));
    }
    if preserve {
        if let Some(tiff) = &meta.exif_tiff {
            inserts.extend_from_slice(&png_chunk(b"eXIf", tiff));
        }
        let xmp = meta.xmp.as_deref().filter(|xmp| xmp.len() <= MAX_XMP_BYTES);
        let text = match xmp {
            Some(xmp) if meta.xmp_compressed => zlib(xmp).map(|text| (text, 1)),
            Some(xmp) => Some((xmp.to_vec(), 0)),
            None => None,
        };
        if let Some((text, flag)) = text {
            // iTXt layout: keyword NUL, compression flag and
            // method (0, zlib), an empty language tag and
            // translated keyword (two NULs), then the text.
            let mut data = b"XML:com.adobe.xmp\0".to_vec();
            data.extend_from_slice(&[flag, 0, 0, 0]);
            data.extend_from_slice(&text);
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

/// EXIF and Photoshop fixtures for the color, engine and
/// HEIC tests (issue #611): TIFF streams laid out the way
/// camera EXIF is, and readers for what an output carries
/// that go straight to its segments, around the extractor
/// under test.
#[cfg(test)]
pub(crate) mod fixture {
    /// A TIFF stream under construction, in one byte order.
    pub struct Tiff {
        pub little: bool,
        pub bytes: Vec<u8>,
    }

    /// One IFD entry: tag, field type, count, and the value
    /// bytes in stream order. Up to 4 bytes go in the
    /// entry; longer values go after the IFD's table.
    pub struct Entry {
        pub tag: u16,
        pub kind: u16,
        pub count: u32,
        pub data: Vec<u8>,
    }

    impl Tiff {
        /// The 8-byte header, with IFD0 at offset 8.
        pub fn new(little: bool) -> Self {
            let mut t = Tiff {
                little,
                bytes: if little { b"II" } else { b"MM" }.to_vec(),
            };
            let magic = t.u16(42);
            let ifd0 = t.u32(8);
            t.bytes.extend_from_slice(&magic);
            t.bytes.extend_from_slice(&ifd0);
            t
        }

        pub fn u16(&self, v: u16) -> [u8; 2] {
            if self.little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        }

        pub fn u32(&self, v: u32) -> [u8; 4] {
            if self.little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        }

        pub fn short(&self, tag: u16, v: u16) -> Entry {
            Entry {
                tag,
                kind: 3,
                count: 1,
                data: self.u16(v).to_vec(),
            }
        }

        pub fn long(&self, tag: u16, v: u32) -> Entry {
            self.longs(tag, &[v])
        }

        pub fn longs(&self, tag: u16, vs: &[u32]) -> Entry {
            Entry {
                tag,
                kind: 4,
                count: vs.len() as u32,
                data: vs.iter().flat_map(|v| self.u32(*v)).collect(),
            }
        }

        pub fn ascii(&self, tag: u16, s: &str) -> Entry {
            let mut data = s.as_bytes().to_vec();
            data.push(0);
            Entry {
                tag,
                kind: 2,
                count: data.len() as u32,
                data,
            }
        }

        pub fn rational(&self, tag: u16, num: u32, den: u32) -> Entry {
            let mut data = self.u32(num).to_vec();
            data.extend_from_slice(&self.u32(den));
            Entry {
                tag,
                kind: 5,
                count: 1,
                data,
            }
        }

        pub fn undefined(&self, tag: u16, data: &[u8]) -> Entry {
            Entry {
                tag,
                kind: 7,
                count: data.len() as u32,
                data: data.to_vec(),
            }
        }

        /// Append an IFD, word aligned: its table, then the
        /// values that do not fit in their entries. Returns
        /// the IFD's offset and the position of each entry's
        /// value field, for pointers set later.
        pub fn ifd(&mut self, entries: &[Entry], next: u32) -> (u32, Vec<usize>) {
            self.align();
            let start = self.bytes.len();
            let table = 2 + 12 * entries.len() + 4;
            let mut values = Vec::new();
            let mut fields = Vec::new();
            let count = self.u16(entries.len() as u16);
            self.bytes.extend_from_slice(&count);
            for e in entries {
                let head = [self.u16(e.tag), self.u16(e.kind)].concat();
                let count = self.u32(e.count);
                self.bytes.extend_from_slice(&head);
                self.bytes.extend_from_slice(&count);
                fields.push(self.bytes.len());
                if e.data.len() <= 4 {
                    let mut inline = e.data.clone();
                    inline.resize(4, 0);
                    self.bytes.extend_from_slice(&inline);
                } else {
                    let at = self.u32((start + table + values.len()) as u32);
                    self.bytes.extend_from_slice(&at);
                    values.extend_from_slice(&e.data);
                    if values.len() % 2 == 1 {
                        values.push(0);
                    }
                }
            }
            let next = self.u32(next);
            self.bytes.extend_from_slice(&next);
            self.bytes.extend_from_slice(&values);
            (start as u32, fields)
        }

        /// Append raw bytes, word aligned; returns their
        /// offset.
        pub fn append(&mut self, data: &[u8]) -> u32 {
            self.align();
            let at = self.bytes.len();
            self.bytes.extend_from_slice(data);
            at as u32
        }

        pub fn set_u32(&mut self, pos: usize, v: u32) {
            let v = self.u32(v);
            self.bytes[pos..pos + 4].copy_from_slice(&v);
        }

        fn align(&mut self) {
            if self.bytes.len() % 2 == 1 {
                self.bytes.push(0);
            }
        }
    }

    /// The position of the next-IFD field of an IFD at
    /// `ifd` with `entries` entries.
    pub fn next_field(ifd: u32, entries: usize) -> usize {
        ifd as usize + 2 + 12 * entries
    }

    /// A camera-style EXIF stream.
    pub struct Camera {
        pub tiff: Vec<u8>,
        /// Where IFD1 starts. Everything from here to the
        /// end is IFD1, its values, and the thumbnail.
        pub ifd1: usize,
        /// The position of IFD0's next-IFD field.
        pub ifd0_next: usize,
    }

    /// IFD0 (Make, Orientation, the Exif IFD pointer), the
    /// Exif IFD (PixelXDimension as a LONG and
    /// PixelYDimension as a SHORT, as cameras vary), then
    /// IFD1 (JPEG compression, a resolution, and the
    /// thumbnail's offset and length) and the thumbnail
    /// itself, last, where cameras put it.
    pub fn camera(little: bool, orientation: u16, size: (u32, u16), thumbnail: &[u8]) -> Camera {
        let mut t = Tiff::new(little);
        let ifd0 = [
            t.ascii(0x010F, "Crabcam"),
            t.short(0x0112, orientation),
            t.long(0x8769, 0),
        ];
        let (ifd0_at, ifd0_fields) = t.ifd(&ifd0, 0);
        let exif = [t.long(0xA002, size.0), t.short(0xA003, size.1)];
        let (exif_at, _) = t.ifd(&exif, 0);
        t.set_u32(ifd0_fields[2], exif_at);
        let ifd1 = [
            t.short(0x0103, 6),
            t.rational(0x011A, 72, 1),
            t.long(0x0201, 0),
            t.long(0x0202, thumbnail.len() as u32),
        ];
        let (ifd1_at, ifd1_fields) = t.ifd(&ifd1, 0);
        let ifd0_next = next_field(ifd0_at, ifd0.len());
        t.set_u32(ifd0_next, ifd1_at);
        let thumb_at = t.append(thumbnail);
        t.set_u32(ifd1_fields[2], thumb_at);
        Camera {
            tiff: t.bytes,
            ifd1: ifd1_at as usize,
            ifd0_next,
        }
    }

    /// A small JPEG to stand in for a thumbnail: a red
    /// block beside gray, the kind of pixels a crop or a
    /// redaction takes out.
    pub fn thumbnail_jpeg() -> Vec<u8> {
        let img = image::RgbImage::from_fn(16, 12, |x, _| {
            if x < 8 {
                image::Rgb([220, 10, 10])
            } else {
                image::Rgb([128, 128, 128])
            }
        });
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut out),
                image::ImageFormat::Jpeg,
            )
            .unwrap();
        out
    }

    fn order_u16(tiff: &[u8], off: usize) -> Option<u16> {
        let raw: [u8; 2] = tiff.get(off..off + 2)?.try_into().ok()?;
        Some(if tiff.starts_with(b"II") {
            u16::from_le_bytes(raw)
        } else {
            u16::from_be_bytes(raw)
        })
    }

    fn order_u32(tiff: &[u8], off: usize) -> Option<u32> {
        let raw: [u8; 4] = tiff.get(off..off + 4)?.try_into().ok()?;
        Some(if tiff.starts_with(b"II") {
            u32::from_le_bytes(raw)
        } else {
            u32::from_be_bytes(raw)
        })
    }

    /// The type and the inline SHORT or LONG value of
    /// `tag` in the IFD at `ifd`.
    pub fn typed_tag(tiff: &[u8], ifd: usize, tag: u16) -> Option<(u16, u32)> {
        let count = order_u16(tiff, ifd)? as usize;
        (0..count).find_map(|i| {
            let entry = ifd + 2 + 12 * i;
            if order_u16(tiff, entry)? != tag {
                return None;
            }
            match order_u16(tiff, entry + 2)? {
                3 => Some((3, u32::from(order_u16(tiff, entry + 8)?))),
                kind => Some((kind, order_u32(tiff, entry + 8)?)),
            }
        })
    }

    /// The inline SHORT or LONG value of `tag` in the IFD
    /// at `ifd`.
    pub fn tag(tiff: &[u8], ifd: usize, tag: u16) -> Option<u32> {
        typed_tag(tiff, ifd, tag).map(|(_, v)| v)
    }

    /// The next-IFD offset of the IFD at `ifd`.
    pub fn next_ifd(tiff: &[u8], ifd: usize) -> Option<u32> {
        let count = order_u16(tiff, ifd)? as usize;
        order_u32(tiff, ifd + 2 + 12 * count)
    }

    pub fn contains(hay: &[u8], needle: &[u8]) -> bool {
        hay.windows(needle.len()).any(|w| w == needle)
    }

    /// The EXIF TIFF stream an encoded JPEG (APP1) or PNG
    /// (`eXIf`) carries, read from its segments as is.
    pub fn carried_exif(bytes: &[u8]) -> Option<Vec<u8>> {
        if bytes.starts_with(&[0xFF, 0xD8]) {
            jpeg_segment_payload(bytes, 0xE1, super::EXIF_JPEG_HEADER)
                .map(|p| p[super::EXIF_JPEG_HEADER.len()..].to_vec())
        } else {
            let mut pos = 8;
            while pos + 12 <= bytes.len() {
                let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                if &bytes[pos + 4..pos + 8] == b"eXIf" {
                    return Some(bytes[pos + 8..pos + 8 + len].to_vec());
                }
                pos += 12 + len;
            }
            None
        }
    }

    /// The Photoshop (APP13) payload an encoded JPEG
    /// carries, read from its segments as is.
    pub fn carried_photoshop(bytes: &[u8]) -> Option<Vec<u8>> {
        jpeg_segment_payload(bytes, 0xED, super::PHOTOSHOP_JPEG_HEADER)
    }

    fn jpeg_segment_payload(bytes: &[u8], marker: u8, header: &[u8]) -> Option<Vec<u8>> {
        let mut pos = 2;
        while pos + 4 <= bytes.len() && bytes[pos] == 0xFF && bytes[pos + 1] != 0xDA {
            let len = u16::from_be_bytes([bytes[pos + 2], bytes[pos + 3]]) as usize;
            let payload = &bytes[pos + 4..pos + 2 + len];
            if bytes[pos + 1] == marker && payload.starts_with(header) {
                return Some(payload.to_vec());
            }
            pos += 2 + len;
        }
        None
    }

    /// A Photoshop image resource: signature, id, a Pascal
    /// name padded to even, the size, and the data padded
    /// to even.
    pub fn irb_named(id: u16, name: &[u8], data: &[u8]) -> Vec<u8> {
        let mut out = b"8BIM".to_vec();
        out.extend_from_slice(&id.to_be_bytes());
        out.push(name.len() as u8);
        out.extend_from_slice(name);
        if out.len() % 2 == 1 {
            out.push(0);
        }
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(data);
        if data.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    pub fn irb(id: u16, data: &[u8]) -> Vec<u8> {
        irb_named(id, b"", data)
    }

    /// A Photoshop thumbnail resource's data: the 28-byte
    /// header (format 1, JPEG), then the JPEG.
    pub fn irb_thumbnail(jpeg: &[u8]) -> Vec<u8> {
        let mut data = 1u32.to_be_bytes().to_vec();
        data.resize(28, 0);
        data.extend_from_slice(jpeg);
        data
    }

    /// The ids of the resources in a Photoshop payload, in
    /// order, as far as they parse.
    pub fn irb_ids(payload: &[u8]) -> Vec<u16> {
        let mut ids = Vec::new();
        let mut pos = super::PHOTOSHOP_JPEG_HEADER.len();
        while pos + 12 <= payload.len() && &payload[pos..pos + 4] == b"8BIM" {
            ids.push(u16::from_be_bytes([payload[pos + 4], payload[pos + 5]]));
            let name = (1 + payload[pos + 6] as usize).next_multiple_of(2);
            let size_at = pos + 6 + name;
            let size =
                u32::from_be_bytes(payload[size_at..size_at + 4].try_into().unwrap()) as usize;
            pos = size_at + 4 + size.next_multiple_of(2);
        }
        ids
    }
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

    /// A Photoshop segment payload holding one IPTC record
    /// (resource 0x0404).
    fn iptc_photoshop() -> Vec<u8> {
        [
            PHOTOSHOP_JPEG_HEADER,
            &fixture::irb(0x0404, b"\x1c\x02\x00\x00\x02\x00\x04"),
        ]
        .concat()
    }

    /// `len` bytes that pass `icc_embeddable`: an ICC header
    /// with the size field and signature in place, no tags,
    /// then a byte pattern standing in for tag data.
    fn icc_fixture(len: usize) -> Vec<u8> {
        let mut icc: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        icc[0..4].copy_from_slice(&(len as u32).to_be_bytes());
        icc[36..40].copy_from_slice(b"acsp");
        icc[128..132].copy_from_slice(&0u32.to_be_bytes());
        icc
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
            icc: Some(icc_fixture(70000)), // forces two APP2 chunks
            exif_tiff: Some(tiff),
            xmp: Some(b"<x:xmpmeta/>".to_vec()),
            xmp_compressed: false,
            photoshop: Some(iptc_photoshop()),
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
            icc: Some(moxcms::ColorProfile::new_display_p3().encode().unwrap()),
            exif_tiff: Some(tiff_with_orientation(3, true)),
            xmp: Some(b"<x:xmpmeta/>".to_vec()),
            xmp_compressed: false,
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
            Some((b"<x:xmpmeta/>".to_vec(), false))
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
            xmp_compressed: false,
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
        // IFD0 inside the header.
        let mut ifd0_in_header = tiff_with_orientation(6, true);
        ifd0_in_header[4..8].copy_from_slice(&0u32.to_le_bytes());
        cases.push(("IFD0 inside the header", ifd0_in_header));
        for (name, mut tiff) in cases {
            let before = tiff.clone();
            patch_tiff_orientation(&mut tiff);
            assert_eq!(tiff, before, "{name}");
            // The other in-place edits of a preserved EXIF
            // (issue #611) leave it alone too.
            patch_tiff_dimensions(&mut tiff, 40, 30);
            assert_eq!(tiff, before, "{name}: sizes");
            strip_tiff_thumbnails(&mut tiff);
            assert_eq!(tiff, before, "{name}: thumbnails");
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
        let itxt = compressed_xmp_payload(&zlib_zeros(MAX_XMP_BYTES));
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
        assert_eq!(MAX_INFLATE_BYTES - budget, MAX_ICC_BYTES + MAX_XMP_BYTES);
        assert_eq!(meta.icc.map(|icc| icc.len()), Some(MAX_ICC_BYTES));
        assert_eq!(meta.xmp.map(|xmp| xmp.len()), Some(MAX_XMP_BYTES));
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
            icc: Some(icc_fixture(600)),
            exif_tiff: Some(tiff_with_orientation(3, true)),
            xmp: Some(b"<x:xmpmeta/>".to_vec()),
            xmp_compressed: false,
            photoshop: Some(iptc_photoshop()),
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
        // cap + 1 bytes of the stream, a truncated document
        // carried into the output.
        let itxt = compressed_xmp_payload(&zlib_zeros(MAX_XMP_BYTES + 1));
        let png = png_with_chunks(&[(b"iTXt", &itxt)]);
        let mut budget = MAX_INFLATE_BYTES;
        assert!(extract_png_meta(&png, true, &mut budget).xmp.is_none());
        // Charged up to the cap; the byte that detected the
        // overflow is not.
        assert_eq!(MAX_INFLATE_BYTES - budget, MAX_XMP_BYTES);
        // An uncompressed packet answers to the same cap.
        for (len, kept) in [(MAX_XMP_BYTES, true), (MAX_XMP_BYTES + 1, false)] {
            let mut itxt = b"XML:com.adobe.xmp\0\0\0\0\0".to_vec();
            itxt.resize(itxt.len() + len, b' ');
            let png = png_with_chunks(&[(b"iTXt", &itxt)]);
            let xmp = extract_source_meta(&png, true).xmp;
            assert_eq!(xmp.map(|xmp| xmp.len()), kept.then_some(len), "{len}");
        }
    }

    #[test]
    fn the_profile_does_not_depend_on_the_metadata_policy() {
        // The profile drives the pixels, so both policies
        // must read the same one. An over-cap XMP packet
        // ahead of the iCCP chunk (iTXt may come anywhere
        // before IDAT) is read only under preserve; it must
        // not leave a profile at its cap short of budget.
        let itxt = compressed_xmp_payload(&zlib_zeros(MAX_XMP_BYTES + 1));
        let iccp = iccp_payload(&zlib_zeros(MAX_ICC_BYTES));
        let png = png_with_chunks(&[(b"iTXt", &itxt), (b"iCCP", &iccp)]);
        let stripped = extract_source_meta(&png, false);
        assert_eq!(stripped.icc.as_ref().map(Vec::len), Some(MAX_ICC_BYTES));
        let kept = extract_source_meta(&png, true);
        assert!(kept.xmp.is_none());
        assert_eq!(kept.icc.as_ref().map(Vec::len), Some(MAX_ICC_BYTES));
        assert!(kept.icc == stripped.icc);
    }

    #[test]
    fn icc_embeddable_checks_the_header_and_the_cap() {
        // Issue #445: the profile is source bytes, so only a
        // header that describes exactly these bytes, under
        // the embed cap, is copied into an output.
        for real in [
            moxcms::ColorProfile::new_srgb().encode().unwrap(),
            moxcms::ColorProfile::new_display_p3().encode().unwrap(),
            moxcms::ColorProfile::new_gray_with_gamma(2.2)
                .encode()
                .unwrap(),
        ] {
            assert!(icc_embeddable(&real), "{} bytes", real.len());
        }
        assert!(icc_embeddable(&icc_fixture(132)));
        assert!(icc_embeddable(&icc_fixture(MAX_EMBED_ICC_BYTES)));
        let p3 = moxcms::ColorProfile::new_display_p3().encode().unwrap();
        let mut refused: Vec<(&str, Vec<u8>)> = vec![
            ("empty", Vec::new()),
            ("shorter than the header", p3[..131].to_vec()),
            ("all zeros, as the issue's 4 MiB iCCP", vec![0; 4096]),
            ("past the embed cap", icc_fixture(MAX_EMBED_ICC_BYTES + 4)),
            (
                "over the cap and over the extraction cap",
                icc_fixture(MAX_ICC_BYTES),
            ),
        ];
        let mut trailing = p3.clone();
        trailing.push(0);
        refused.push(("a byte past the declared size", trailing));
        refused.push((
            "cut short of the declared size",
            p3[..p3.len() - 1].to_vec(),
        ));
        let mut unsigned = p3.clone();
        unsigned[36..40].copy_from_slice(b"ACSP");
        refused.push(("no acsp signature", unsigned));
        let mut tags = icc_fixture(1000);
        tags[128..132].copy_from_slice(&80u32.to_be_bytes()); // 132 + 80 * 12 > 1000
        refused.push(("a tag table past the end", tags));
        let mut tags = icc_fixture(1000);
        tags[128..132].copy_from_slice(&u32::MAX.to_be_bytes());
        refused.push(("a tag count that overflows", tags));
        for (name, icc) in refused {
            assert!(!icc_embeddable(&icc), "{name}");
        }
    }

    #[test]
    fn injectors_leave_out_a_profile_that_fails_validation() {
        // Issue #445: the issue's 4 MiB iCCP of zeros went
        // into JPEG outputs as 65 raw APP2 segments. A
        // refused profile is left out of both containers;
        // with nothing else to add, the output is the input.
        let jpeg = [0xFF, 0xD8, 0xFF, 0xD9];
        let png = png_with_chunks(&[]);
        let none = SourceMeta::default();
        for bad in [
            vec![0u8; MAX_ICC_BYTES],
            icc_fixture(MAX_EMBED_ICC_BYTES + 4),
            b"fake-profile-bytes".to_vec(),
        ] {
            assert_eq!(inject_jpeg(&jpeg, Some(&bad), &none, false), jpeg);
            assert_eq!(inject_png(&png, Some(&bad), &none, false), png);
        }
        // A profile at the cap goes in, whole, to both.
        let at_cap = icc_fixture(MAX_EMBED_ICC_BYTES);
        let out = inject_jpeg(&jpeg, Some(&at_cap), &none, false);
        assert_eq!(extract_source_meta(&out, false).icc, Some(at_cap.clone()));
        let out = inject_png(&png, Some(&at_cap), &none, false);
        assert_eq!(extract_source_meta(&out, false).icc, Some(at_cap));
    }

    /// The data of every `kind` chunk in a PNG.
    fn png_chunks<'a>(png: &'a [u8], kind: &[u8; 4]) -> Vec<&'a [u8]> {
        let mut found = Vec::new();
        let mut pos = 8;
        while pos + 12 <= png.len() {
            let len = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
            if &png[pos + 4..pos + 8] == kind {
                found.push(&png[pos + 8..pos + 8 + len]);
            }
            pos += 12 + len;
        }
        found
    }

    #[test]
    fn png_xmp_goes_back_out_as_the_source_stored_it() {
        // Issue #445: a packet the source stored compressed
        // (16 MiB of padding in a 16 KB stream) came back as
        // an uncompressed iTXt of its full size. It goes
        // back out compressed now, weighing about what it
        // weighed in the source; a packet the source stored
        // as plain text stays plain text, the form packet
        // scanners and some readers expect. Either way it
        // reads back intact.
        let mut xmp = b"<x:xmpmeta xmlns:x='adobe:ns:meta/'>".to_vec();
        xmp.resize(MAX_XMP_BYTES - 13, b' ');
        xmp.extend_from_slice(b"</x:xmpmeta>\n");
        assert_eq!(xmp.len(), MAX_XMP_BYTES);
        let base = png_with_chunks(&[]);
        let header = b"XML:com.adobe.xmp\0";
        for compressed in [true, false] {
            let meta = SourceMeta {
                xmp: Some(xmp.clone()),
                xmp_compressed: compressed,
                ..SourceMeta::default()
            };
            let out = inject_png(&base, None, &meta, true);
            let itxt = png_chunks(&out, b"iTXt");
            assert_eq!(itxt.len(), 1);
            assert!(itxt[0].starts_with(header));
            let flag = u8::from(compressed);
            assert_eq!(&itxt[0][header.len()..header.len() + 4], &[flag, 0, 0, 0]);
            if compressed {
                assert!(out.len() < base.len() + 2048, "{} bytes", out.len());
            } else {
                assert!(itxt[0].ends_with(&xmp));
            }
            let back = extract_source_meta(&out, true);
            assert_eq!(back.xmp.as_ref(), Some(&xmp));
            assert_eq!(back.xmp_compressed, compressed);
        }
        // From a source PNG to the output, through the
        // extractor: the compressed packet stays compressed.
        let src = png_with_chunks(&[(b"iTXt", &compressed_xmp_payload(&zlib(&xmp).unwrap()))]);
        let meta = extract_source_meta(&src, true);
        assert!(meta.xmp_compressed);
        let out = inject_png(&base, None, &meta, true);
        assert!(out.len() < base.len() + 2048, "{} bytes", out.len());
        // A packet past the cap, as a decoder could hand
        // over, is left out rather than written.
        let mut over = xmp;
        over.push(b' ');
        for compressed in [true, false] {
            let meta = SourceMeta {
                xmp: Some(over.clone()),
                xmp_compressed: compressed,
                ..SourceMeta::default()
            };
            assert_eq!(inject_png(&base, None, &meta, true), base);
        }
    }

    /// A bare JPEG whose only segment is an APP1 EXIF
    /// carrying `tiff`.
    fn jpeg_with_exif(tiff: &[u8]) -> Vec<u8> {
        let mut src = vec![0xFF, 0xD8];
        src.extend_from_slice(&jpeg_segment(0xE1, &[EXIF_JPEG_HEADER, tiff].concat()).unwrap());
        src.extend_from_slice(&[0xFF, 0xD9]);
        src
    }

    /// The EXIF a preserving JPEG and PNG source yield.
    fn preserved_exif(tiff: &[u8]) -> [(&'static str, Vec<u8>); 2] {
        let from_jpeg = extract_source_meta(&jpeg_with_exif(tiff), true);
        let from_png = extract_source_meta(&png_with_chunks(&[(b"eXIf", tiff)]), true);
        [
            ("jpeg", from_jpeg.exif_tiff.expect("jpeg exif")),
            ("png", from_png.exif_tiff.expect("png exif")),
        ]
    }

    #[test]
    fn exif_thumbnails_are_cut_from_preserved_exif() {
        // Issue #611: the IFD1 thumbnail, a small copy of
        // the whole uncut source, rode into every output
        // under preserve. The preserved copy now ends where
        // IFD1 began: IFD0 no longer links to it, and IFD1,
        // its values, and the thumbnail are gone. Everything
        // before IFD1 is copied as is.
        let thumbnail = fixture::thumbnail_jpeg();
        for little in [true, false] {
            let cam = fixture::camera(little, 6, (4000, 3000), &thumbnail);
            let mut want = cam.tiff[..cam.ifd1].to_vec();
            want[cam.ifd0_next..cam.ifd0_next + 4].fill(0);
            patch_tiff_orientation(&mut want);
            for (name, tiff) in preserved_exif(&cam.tiff) {
                assert!(
                    !fixture::contains(&tiff, &thumbnail),
                    "{name} little={little}: the thumbnail is still there"
                );
                assert_eq!(fixture::next_ifd(&tiff, 8), Some(0), "{name} {little}");
                assert_eq!(tiff, want, "{name} little={little}");
            }
        }
    }

    #[test]
    fn exif_without_a_thumbnail_is_copied_byte_for_byte() {
        // The sanitizer edits in place and never rebuilds
        // the stream, so a stream with nothing to remove
        // comes out as it went in, orientation aside. The
        // upgrade corpus's metadata-preserve entry renders
        // this fixture, so its bytes must not move.
        let corpus = include_bytes!("../tests/functional/corpus/exif-orient.jpg");
        let mut want = fixture::carried_exif(corpus).expect("fixture exif");
        patch_tiff_orientation(&mut want);
        let got = extract_source_meta(corpus, true).exif_tiff;
        assert_eq!(got, Some(want));
        // A camera stream without IFD1, in both orders.
        for little in [true, false] {
            let mut t = fixture::Tiff::new(little);
            let ifd0 = [t.ascii(0x010F, "Crabcam"), t.short(0x0112, 3)];
            t.ifd(&ifd0, 0);
            let mut want = t.bytes.clone();
            patch_tiff_orientation(&mut want);
            for (name, tiff) in preserved_exif(&t.bytes) {
                assert_eq!(tiff, want, "{name} little={little}");
            }
        }
    }

    #[test]
    fn uncompressed_and_chained_thumbnails_are_blanked() {
        // IFD1 can hold an uncompressed thumbnail in strips,
        // and a chain can go on past IFD1. Every IFD after
        // IFD0 goes with its image data. Data that IFD0
        // still uses lies after them here, so the removed
        // bytes are zeroed in place rather than cut.
        let thumbnail = fixture::thumbnail_jpeg();
        let strips: [&[u8]; 2] = [&[0xAB; 30], &[0xCD; 20]];
        for little in [true, false] {
            let mut t = fixture::Tiff::new(little);
            let artist = fixture::Entry {
                tag: 0x013B,
                kind: 2,
                count: 18,
                data: vec![0; 4],
            };
            let ifd0 = [t.short(0x0112, 1), artist];
            let (ifd0_at, ifd0_fields) = t.ifd(&ifd0, 0);
            let ifd1 = [
                t.short(0x0103, 1),
                t.longs(0x0111, &[0, 0]),
                fixture::Entry {
                    tag: 0x0117,
                    kind: 3,
                    count: 2,
                    data: [t.u16(30), t.u16(20)].concat(),
                },
            ];
            let (ifd1_at, _) = t.ifd(&ifd1, 0);
            let link = fixture::next_field(ifd0_at, ifd0.len());
            t.set_u32(link, ifd1_at);
            let strip0 = t.append(strips[0]);
            let strip1 = t.append(strips[1]);
            // StripOffsets is out of line: its value field
            // holds where the two values are. Point them at
            // the strips.
            let values = fixture::tag(&t.bytes, ifd1_at as usize, 0x0111).unwrap() as usize;
            t.set_u32(values, strip0);
            t.set_u32(values + 4, strip1);
            let ifd2 = [t.long(0x0201, 0), t.long(0x0202, thumbnail.len() as u32)];
            let (ifd2_at, ifd2_fields) = t.ifd(&ifd2, 0);
            t.set_u32(fixture::next_field(ifd1_at, ifd1.len()), ifd2_at);
            let thumb_at = t.append(&thumbnail);
            t.set_u32(ifd2_fields[0], thumb_at);
            let artist_at = t.append(b"Crab Photographer\0");
            t.set_u32(ifd0_fields[1], artist_at);
            let mut want = t.bytes.clone();
            want[ifd1_at as usize..artist_at as usize].fill(0);
            want[link..link + 4].fill(0);
            for (name, tiff) in preserved_exif(&t.bytes) {
                for data in [strips[0], strips[1], &thumbnail] {
                    assert!(!fixture::contains(&tiff, data), "{name} {little}");
                }
                assert_eq!(tiff, want, "{name} little={little}");
            }
        }
    }

    #[test]
    fn maker_note_previews_are_blanked() {
        // Camera maker notes carry JPEG previews of the
        // whole frame too (Nikon, Olympus and Pentax among
        // others). Any JPEG left in the stream is zeroed,
        // from its SOI to its EOI; the maker note's other
        // bytes stay.
        let preview = fixture::thumbnail_jpeg();
        for little in [true, false] {
            let mut t = fixture::Tiff::new(little);
            let ifd0 = [t.short(0x0112, 1), t.long(0x8769, 0)];
            let (_, ifd0_fields) = t.ifd(&ifd0, 0);
            let note = [b"Nikon\0\x02\x10\0\0".as_slice(), &preview, b"lens data"].concat();
            let exif = [t.undefined(0x927C, &note)];
            let (exif_at, _) = t.ifd(&exif, 0);
            t.set_u32(ifd0_fields[1], exif_at);
            let at = t
                .bytes
                .windows(preview.len())
                .position(|w| w == preview)
                .unwrap();
            let mut want = t.bytes.clone();
            want[at..at + preview.len()].fill(0);
            for (name, tiff) in preserved_exif(&t.bytes) {
                assert_eq!(tiff, want, "{name} little={little}");
            }
        }
    }

    #[test]
    fn odd_exif_chains_are_cut_without_harm() {
        // Edge cases of the IFD walk. Each must end with
        // IFD0 unlinked, IFD0's own tags intact, the header
        // in place, and no panic.
        let thumbnail = fixture::thumbnail_jpeg();
        let cam = fixture::camera(true, 1, (64, 48), &thumbnail);
        let ifd1 = cam.ifd1;
        let ifd1_next = fixture::next_field(ifd1 as u32, 4);
        let thumb_len_field = ifd1 + 2 + 12 * 3 + 8;
        let thumb_at_field = ifd1 + 2 + 12 * 2 + 8;
        let set = |tiff: &mut Vec<u8>, pos: usize, v: u32| {
            tiff[pos..pos + 4].copy_from_slice(&v.to_le_bytes())
        };
        let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
        let mut t = cam.tiff.clone();
        set(&mut t, ifd1_next, 8);
        cases.push(("IFD1 links back to IFD0", t));
        let mut t = cam.tiff.clone();
        set(&mut t, ifd1_next, ifd1 as u32);
        cases.push(("IFD1 links to itself", t));
        let mut t = cam.tiff.clone();
        set(&mut t, thumb_len_field, u32::MAX);
        cases.push(("thumbnail length past the end", t));
        let mut t = cam.tiff.clone();
        set(&mut t, thumb_len_field, 0);
        t[thumb_len_field - 8..thumb_len_field - 6].copy_from_slice(&0x9999u16.to_le_bytes());
        cases.push(("no thumbnail length tag", t));
        let mut t = cam.tiff.clone();
        t.truncate(ifd1 + 2);
        cases.push(("IFD1 cut short", t));
        for (name, tiff) in cases {
            for (kind, got) in preserved_exif(&tiff) {
                assert!(got.len() >= 8, "{name} ({kind})");
                assert_eq!(&got[..8], &tiff[..8], "{name} ({kind}): header");
                assert_eq!(fixture::next_ifd(&got, 8), Some(0), "{name} ({kind})");
                assert_eq!(fixture::tag(&got, 8, 0x0112), Some(1), "{name} ({kind})");
                assert!(
                    !fixture::contains(&got, &thumbnail[..64]),
                    "{name} ({kind}): the thumbnail is still there"
                );
            }
        }
        // A thumbnail range that claims the whole stream:
        // all of it goes but the header.
        let mut t = cam.tiff.clone();
        set(&mut t, thumb_at_field, 0);
        set(&mut t, thumb_len_field, u32::MAX);
        for (kind, got) in preserved_exif(&t) {
            assert_eq!(got, &t[..8], "range over the header ({kind})");
        }
        // IFD0 links past the end of the stream: the link
        // goes, nothing else changes.
        let mut t = cam.tiff[..cam.ifd1].to_vec();
        set(&mut t, cam.ifd0_next, 1 << 20);
        let mut want = t.clone();
        want[cam.ifd0_next..cam.ifd0_next + 4].fill(0);
        for (kind, got) in preserved_exif(&t) {
            assert_eq!(got, want, "dangling link ({kind})");
        }
        // Bytes that open like a JPEG but reach no scan are
        // not one: they stay.
        let mut t = fixture::Tiff::new(true);
        let lookalike = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0x01, 0x02, 0x55, 0x66];
        let ifd0 = [t.short(0x0112, 1), t.undefined(0x927C, &lookalike)];
        t.ifd(&ifd0, 0);
        for (kind, got) in preserved_exif(&t.bytes) {
            assert_eq!(got, t.bytes, "lookalike ({kind})");
        }
    }

    #[test]
    fn the_preview_scan_stays_linear() {
        // A stream packed with JPEG starts whose segments
        // chain to the end without a scan: walking again
        // from every start would be quadratic in the
        // stream. One walk covers them all, and nothing is
        // an image, so nothing changes.
        let unit = [0xFF, 0xE0, 0x00, 0x05, 0xFF, 0xD8, 0xFF];
        // An empty IFD0, then the run.
        let mut tiff = vec![b'I', b'I', 42, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xD8];
        while tiff.len() < 2 * 1024 * 1024 {
            tiff.extend_from_slice(&unit);
        }
        let start = std::time::Instant::now();
        let meta = extract_source_meta(&png_with_chunks(&[(b"eXIf", &tiff)]), true);
        assert!(start.elapsed() < std::time::Duration::from_secs(5));
        assert_eq!(meta.exif_tiff, Some(tiff));
        // A stream packed with tiny JPEGs: every one is
        // zeroed. Past the first 1024 their ranges are not
        // kept for the tail cut, so the zeros stay in.
        let tiny = [0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02, 0x01, 0xFF, 0xD9];
        let mut tiff = vec![b'I', b'I', 42, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        for _ in 0..2000 {
            tiff.extend_from_slice(&tiny);
        }
        let got = extract_source_meta(&png_with_chunks(&[(b"eXIf", &tiff)]), true)
            .exif_tiff
            .unwrap();
        assert_eq!(got.len(), tiff.len());
        assert_eq!(got[..14], tiff[..14]);
        assert!(got[14..].iter().all(|&b| b == 0));
    }

    #[test]
    fn photoshop_thumbnail_resources_are_dropped() {
        // Issue #611: resources 0x0409 and 0x040C hold
        // Photoshop's thumbnail of the source. They go, and
        // so do 0x0422 and 0x0423, copies of the EXIF that
        // carry its thumbnail and size again. IPTC (0x0404)
        // and every other resource stay, as they were and
        // in order.
        let thumbnail = fixture::thumbnail_jpeg();
        let thumb = fixture::irb_thumbnail(&thumbnail);
        let cam = fixture::camera(false, 1, (4000, 3000), &thumbnail);
        let resolution = fixture::irb(0x03ED, &[0, 72, 0, 0, 0, 1, 0, 1]);
        let iptc = fixture::irb(
            0x0404,
            b"\x1c\x02\x00\x00\x02\x00\x04\x1c\x02\x50\x00\x03Ann",
        );
        let digest = fixture::irb_named(0x0425, b"ab", &[7; 16]);
        let odd = fixture::irb(0x0428, &[1, 2, 3]);
        let segment = [
            PHOTOSHOP_JPEG_HEADER,
            &resolution,
            &fixture::irb(0x0409, &thumb),
            &iptc,
            &fixture::irb(0x040C, &thumb),
            &fixture::irb(0x0422, &cam.tiff),
            &fixture::irb(0x0423, &cam.tiff),
            &digest,
            &odd,
        ]
        .concat();
        let src = |payload: &[u8]| {
            let mut src = vec![0xFF, 0xD8];
            src.extend_from_slice(&jpeg_segment(0xED, payload).unwrap());
            src.extend_from_slice(&[0xFF, 0xD9]);
            src
        };
        let got = extract_source_meta(&src(&segment), true).photoshop;
        let want = [PHOTOSHOP_JPEG_HEADER, &resolution, &iptc, &digest, &odd].concat();
        assert_eq!(
            got.as_deref().map(fixture::irb_ids),
            Some(vec![0x03ED, 0x0404, 0x0425, 0x0428])
        );
        assert_eq!(got, Some(want.clone()));
        // Nothing to drop: copied byte for byte, the last
        // resource's missing pad byte included.
        let unpadded = &want[..want.len() - 1];
        let got = extract_source_meta(&src(unpadded), true).photoshop;
        assert_eq!(got.as_deref(), Some(unpadded));
        // Only thumbnails: no segment at all.
        let only = [PHOTOSHOP_JPEG_HEADER, &fixture::irb(0x040C, &thumb)].concat();
        assert_eq!(extract_source_meta(&src(&only), true).photoshop, None);
        // A tail that does not parse (here the rest of a
        // thumbnail resource split across segments) could
        // be anything: it goes, the resources before it
        // stay.
        let split = [PHOTOSHOP_JPEG_HEADER, &iptc, &thumb[..40]].concat();
        let got = extract_source_meta(&src(&split), true).photoshop;
        assert_eq!(got, Some([PHOTOSHOP_JPEG_HEADER, &iptc].concat()));
        let tail = [PHOTOSHOP_JPEG_HEADER, &thumb[40..]].concat();
        assert_eq!(extract_source_meta(&src(&tail), true).photoshop, None);
    }

    #[test]
    fn size_tags_follow_the_output() {
        // Issue #611: the EXIF sizes are rewritten to the
        // output's, in IFD0 (ImageWidth, ImageLength) and
        // the Exif IFD (PixelXDimension, PixelYDimension),
        // as SHORT or LONG as they were. A SHORT that cannot
        // hold the new value becomes a LONG. Nothing else in
        // the stream moves.
        for little in [true, false] {
            let mut t = fixture::Tiff::new(little);
            let ifd0 = [
                t.short(0x0100, 4000),
                t.long(0x0101, 3000),
                t.short(0x0112, 1),
                t.long(0x8769, 0),
            ];
            let (_, fields) = t.ifd(&ifd0, 0);
            let exif = [
                t.long(0xA002, 4000),
                t.short(0xA003, 3000),
                fixture::Entry {
                    tag: 0xA004,
                    kind: 3,
                    count: 1,
                    data: t.u16(5).to_vec(),
                },
            ];
            let (exif_at, _) = t.ifd(&exif, 0);
            t.set_u32(fields[3], exif_at);
            let exif_at = exif_at as usize;
            let mut tiff = t.bytes.clone();
            patch_tiff_dimensions(&mut tiff, 40, 70_000);
            assert_eq!(tiff.len(), t.bytes.len());
            let tag = |tag| fixture::typed_tag(&tiff, 8, tag);
            assert_eq!(tag(0x0100), Some((3, 40)), "little={little}");
            assert_eq!(tag(0x0101), Some((4, 70_000)), "little={little}");
            assert_eq!(tag(0x0112), Some((3, 1)), "little={little}");
            let tag = |tag| fixture::typed_tag(&tiff, exif_at, tag);
            assert_eq!(tag(0xA002), Some((4, 40)), "little={little}");
            assert_eq!(tag(0xA003), Some((4, 70_000)), "little={little}");
            assert_eq!(tag(0xA004), Some((3, 5)), "little={little}");
            // Back within a SHORT, the converted tag stays
            // a LONG and the others keep their type.
            patch_tiff_dimensions(&mut tiff, 40, 30);
            let tag = |tag| fixture::typed_tag(&tiff, exif_at, tag);
            assert_eq!(tag(0xA003), Some((4, 30)), "little={little}");
            assert_eq!(tag(0xA002), Some((4, 40)), "little={little}");
        }
        // A size tag of another type or count is left as it
        // is; the stream without size tags is unchanged.
        let mut t = fixture::Tiff::new(true);
        let ifd0 = [t.rational(0x0100, 4000, 1), t.longs(0x0101, &[1, 2])];
        t.ifd(&ifd0, 0);
        let mut tiff = t.bytes.clone();
        patch_tiff_dimensions(&mut tiff, 40, 30);
        assert_eq!(tiff, t.bytes);
        let corpus = include_bytes!("../tests/functional/corpus/exif-orient.jpg");
        let mut tiff = extract_source_meta(corpus, true).exif_tiff.unwrap();
        let before = tiff.clone();
        patch_tiff_dimensions(&mut tiff, 40, 30);
        assert_eq!(tiff, before);
    }
}
