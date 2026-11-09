//! AVIF container screening (issue #522).
//!
//! An AVIF source reaches mp4parse, the ISO-BMFF parser behind
//! the image crate's AVIF decoder, before any decode limit
//! applies: the decode starts by parsing the container.
//! mp4parse 0.17 sizes several reads by
//! what the file declares instead of by the bytes it holds. A
//! box it reads into memory (`hdlr`, `iloc`, `idat`, the item
//! properties, `mdat`) reserves its declared size first, and two
//! 32-bit entry counts (`iinf`, and `iloc` from version 2)
//! reserve a slot per declared entry. A 291-byte file found by
//! the decode fuzzer declared a 967 MiB `hdlr`, and the parser
//! reserved all of it.
//!
//! [`screen`] runs first, on every path that hands AVIF bytes to
//! the image crate: [`crate::engine::decode_image`], and
//! `decode_source_image`, which screens a source in the server
//! before the capped worker decodes it (the render, upload, and
//! watermark paths). It walks the boxes mp4parse reads, then the
//! item table, and allocates nothing:
//!
//! - Every box must fit inside its parent (the file, for a
//!   top-level box), whichever way its size is written: 32 bits, 64
//!   bits (`largesize`), or 0, which runs to the end of the
//!   parent and so always fits. The walk enters the containers
//!   mp4parse enters: the top-level `meta`, its `iinf`, `iref`,
//!   and `iprp`, and the `ipco` inside `iprp`.
//! - A 32-bit entry count above the 16-bit range must be backed
//!   by bytes: every `iinf` entry is a box (8 bytes at least),
//!   every version 2 `iloc` item 10 bytes at least.
//! - Every item in `iloc` (versions 0, 1, and 2, every field
//!   size the format allows) has at most [`MAX_EXTENTS`]
//!   extents, and its extent lengths add up to no more than the
//!   bytes the item can come from: the top-level `mdat`
//!   payloads for an item stored in the file (construction
//!   method 0), the `idat` payload for one stored there (method
//!   1). A length of 0 counts as the rest of the `mdat` that
//!   holds the extent's offset (its base offset plus its own),
//!   or of the `idat`, as the parser resolves it. An item built
//!   from other items (method 2) counts toward the cap only.
//! - The primary item's `ispe` (found through `pitm`, its `ipma`
//!   entry, and `ipco`) must declare a size within the engine's
//!   decode limits: [`crate::engine::MAX_DECODE_DIMENSION`] a
//!   side and [`crate::engine::MAX_DECODE_ALLOC`] at four bytes
//!   a pixel. The decode checks the decoded picture against the
//!   same limits; this check comes before any decoding. A
//!   primary item without an `ispe` is left to the decoder, as
//!   before.
//!
//! Every reservation the parser sizes from a declared box size or
//! a 32-bit count is then bounded by the size of the file, and
//! the data the parser assembles for an item by the bytes the
//! file holds; the 16-bit counts stay as they are (about 5 MiB
//! at most, for 65,535 `iloc` items). A refused source fails as any
//! undecodable source does (`502 decode_failed`; `415` for an
//! upload). The parser refuses a box that overruns its parent
//! too, once it has made the reservation, so the walk mostly
//! changes when those files fail, not whether. There are two
//! exceptions. mp4parse never checks that `ipco` ends inside its
//! `iprp`: an `ipco` that runs past it swallows the `ipma`, so
//! the image loses the properties an AV1 item must have and fails
//! later instead. And mp4parse reads the `iinf` count as a
//! capacity hint only, which is why a count is refused only above
//! the 16-bit range (at most 65,535 small slots, which a version
//! 0 box can ask for anyway) and only when the box could not hold
//! that many entries.
//!
//! The walk follows mp4parse's own rules where they are lenient,
//! in the default strictness the image crate uses, so it refuses
//! nothing else the parser reads without such a reservation: a
//! top-level box the parser only skips (anything after the first
//! box but `meta` and `mdat`) may run past the end of the file,
//! where the parser stops reading, and a run of boxes may end in
//! fewer bytes than a box header. An item table the parser
//! refuses outright (a field size other than 0, 4, or 8 bytes,
//! an item with no extents, a table cut short) is left to it.
//!
//! An image-sequence brand (`avis` or `msf1`) makes mp4parse
//! parse the movie box (`moov`) as well, whose readers size reads
//! from fields too: a `pssh` box's `data_size`, the sample
//! tables' entry counts. The image crate decodes the primary
//! image only and never uses the sequence. So instead of
//! mirroring that parser, the screen hands mp4parse a copy with
//! those brands blanked: the parser skips the movie box, as it
//! does any box it does not read, and the still decodes as
//! before.
//!
//! [`declared_size`] reads the same `ispe` for the explain
//! probe and the DPR hint, so sizing an AVIF takes no decode.

use std::borrow::Cow;
use std::fmt;

use crate::engine::EngineError;

/// The compatible brands that make mp4parse parse the movie box.
const SEQUENCE_BRANDS: [[u8; 4]; 2] = [*b"avis", *b"msf1"];

/// What a hidden sequence brand reads as: no brand at all.
const BLANK_BRAND: [u8; 4] = [0; 4];

/// The smallest `iinf` entry: a bare box header.
const MIN_IINF_ENTRY: u64 = 8;

/// The smallest version 2 `iloc` item: a 32-bit item ID, the
/// construction method, the data reference index, and the
/// extent count, with every offset and length field sized 0.
const MIN_ILOC_V2_ITEM: u64 = 10;

/// The most extents one item may list. A grid or overlay image
/// reaches the primary image through many items, but each is one
/// extent; a single image item stored in a handful of extents is
/// the usual multi-extent case. The cap is generous for that and
/// still bounds the per-item extent list, which mp4parse reads
/// into a vector before it copies anything.
const MAX_EXTENTS: u64 = 32;

/// Screen source bytes before the image crate's AVIF decoder
/// sees them. Bytes the image crate does not read as AVIF come
/// back as they are, borrowed. An AVIF container that passes
/// the walk comes back borrowed too, or, when it carries an
/// image-sequence brand, as a copy with those brands blanked.
/// A container the walk refuses is a decode error.
pub(crate) fn screen(bytes: &[u8]) -> Result<Cow<'_, [u8]>, EngineError> {
    if !matches!(image::guess_format(bytes), Ok(image::ImageFormat::Avif)) {
        return Ok(Cow::Borrowed(bytes));
    }
    walk(bytes, Level::File)
        .map_err(|refusal| EngineError::Decode(format!("AVIF container refused: {refusal}")))?;
    let inspect = inspect(bytes)
        .map_err(|refusal| EngineError::Decode(format!("AVIF container refused: {refusal}")))?;
    if let Some((width, height)) = inspect.primary_size
        && let Err(refusal) = within_decode_limits(width, height)
    {
        return Err(EngineError::Decode(format!(
            "AVIF container refused: {refusal}"
        )));
    }
    if sequence_brands(bytes).next().is_none() {
        return Ok(Cow::Borrowed(bytes));
    }
    let mut blanked = bytes.to_vec();
    for at in sequence_brands(bytes) {
        blanked[at..at + 4].copy_from_slice(&BLANK_BRAND);
    }
    Ok(Cow::Owned(blanked))
}

/// The offsets of the image-sequence brands among the
/// compatible brands of the leading `ftyp`, which the walk has
/// already fitted to the file. `bytes` matched the image
/// crate's AVIF signature, so it starts with that box's 32-bit
/// size and holds at least its first 12 bytes.
fn sequence_brands(bytes: &[u8]) -> impl Iterator<Item = usize> + '_ {
    let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    // A size of 0, or one short of a brand list, is an `ftyp`
    // mp4parse refuses outright.
    let end = if (16..=bytes.len()).contains(&size) {
        size
    } else {
        16
    };
    (16..end).step_by(4).filter(move |&at| {
        at + 4 <= end
            && SEQUENCE_BRANDS
                .iter()
                .any(|brand| bytes[at..at + 4] == brand[..])
    })
}

/// Why the walk refused a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    /// A box declared more bytes than remain in its parent.
    Overrun {
        kind: [u8; 4],
        declared: u64,
        room: u64,
    },
    /// A box declared fewer bytes than its own header takes.
    Undersized { kind: [u8; 4], declared: u64 },
    /// A 32-bit entry count the box has no room for.
    Count {
        kind: [u8; 4],
        count: u32,
        room: u64,
    },
    /// An item listing more extents than the cap allows.
    Extents { item: u32, extents: u64 },
    /// An item whose extent lengths sum past the data the file
    /// holds for it.
    Amplified { item: u32, needs: u64, holds: u64 },
    /// The primary item declares a size past the decode limits.
    Dimension { width: u32, height: u32 },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::Overrun {
                kind,
                declared,
                room,
            } => write!(
                f,
                "box '{}' declares {declared} bytes, but only {room} remain in its parent",
                kind.escape_ascii()
            ),
            Refusal::Undersized { kind, declared } => write!(
                f,
                "box '{}' declares {declared} bytes, less than its own header",
                kind.escape_ascii()
            ),
            Refusal::Count { kind, count, room } => write!(
                f,
                "box '{}' counts {count} entries in {room} bytes",
                kind.escape_ascii()
            ),
            Refusal::Extents { item, extents } => {
                write!(f, "item {item} lists {extents} extents")
            }
            Refusal::Amplified { item, needs, holds } => write!(
                f,
                "item {item} sums {needs} extent bytes, more than the {holds} the file holds"
            ),
            Refusal::Dimension { width, height } => {
                write!(
                    f,
                    "the primary item declares {width}x{height}, past the decode limits"
                )
            }
        }
    }
}

/// Where in the box tree a run of boxes sits, which decides the
/// boxes the walk enters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    /// The top level of the file.
    File,
    /// Inside the top-level `meta`.
    Meta,
    /// Inside `iprp`.
    Iprp,
    /// Boxes mp4parse reads one by one without entering any:
    /// the entries of `iinf` and `iref`, the properties in
    /// `ipco`.
    Leaves,
}

/// One box header, read the way mp4parse reads it.
struct Header {
    kind: [u8; 4],
    /// Header bytes: 8, or 16 with a `largesize`.
    len: usize,
    /// The declared size, header included; `None` for size 0,
    /// which runs to the end of the parent.
    size: Option<u64>,
}

impl Header {
    /// The header at the start of `rest`, or `None` when `rest`
    /// is too short to hold one: mp4parse stops reading the
    /// parent there, reserving nothing.
    fn read(rest: &[u8]) -> Result<Option<Header>, Refusal> {
        let Some(&[s0, s1, s2, s3, k0, k1, k2, k3]) = rest.first_chunk::<8>() else {
            return Ok(None);
        };
        let kind = [k0, k1, k2, k3];
        let (len, size) = match u32::from_be_bytes([s0, s1, s2, s3]) {
            0 => (8, None),
            1 => {
                let Some(large) = rest.get(8..16) else {
                    return Ok(None);
                };
                let large = u64::from_be_bytes(large.try_into().expect("an 8-byte slice"));
                (16, Some(large))
            }
            small => (8, Some(u64::from(small))),
        };
        match size {
            Some(declared) if declared < len as u64 => Err(Refusal::Undersized { kind, declared }),
            _ => Ok(Some(Header { kind, len, size })),
        }
    }
}

/// Walk one run of sibling boxes (`data` is exactly their
/// parent's remaining payload), entering the containers mp4parse
/// enters at this level.
fn walk(data: &[u8], level: Level) -> Result<(), Refusal> {
    let mut rest = data;
    let mut first = true;
    while let Some(header) = Header::read(rest)? {
        let room = rest.len() as u64;
        let size = header.size.unwrap_or(room);
        if size > room {
            // A top-level box the parser only skips is trailing
            // data to it: it skips what is there and stops.
            if level == Level::File && !first && !matches!(&header.kind, b"meta" | b"mdat") {
                return Ok(());
            }
            return Err(Refusal::Overrun {
                kind: header.kind,
                declared: size,
                room,
            });
        }
        // size <= room, which is a slice length.
        let (this, next) = rest.split_at(size as usize);
        enter(header.kind, &this[header.len..], level)?;
        rest = next;
        first = false;
    }
    Ok(())
}

/// Check the body of one box that fits its parent: walk into it
/// if mp4parse does, and check the entry counts it reserves by.
/// A body too short for the fields in front of its children is
/// one the parser fails to read before reserving anything.
fn enter(kind: [u8; 4], body: &[u8], level: Level) -> Result<(), Refusal> {
    match (level, &kind) {
        // `meta` and `iref` are full boxes: version and flags
        // come before the children.
        (Level::File, b"meta") => walk(body.get(4..).unwrap_or_default(), Level::Meta),
        (Level::Meta, b"iref") => walk(body.get(4..).unwrap_or_default(), Level::Leaves),
        (Level::Meta, b"iprp") => walk(body, Level::Iprp),
        (Level::Iprp, b"ipco") => walk(body, Level::Leaves),
        (Level::Meta, b"iinf") => {
            // Version 0 counts entries in 16 bits, later
            // versions in 32.
            let Some(&version) = body.first() else {
                return Ok(());
            };
            let entries_at = if version == 0 { 6 } else { 8 };
            let (Some(count), Some(entries)) = (body.get(4..entries_at), body.get(entries_at..))
            else {
                return Ok(());
            };
            let count = count.iter().fold(0u32, |n, &b| (n << 8) | u32::from(b));
            check_count(kind, count, entries.len() as u64, MIN_IINF_ENTRY)?;
            walk(entries, Level::Leaves)
        }
        (Level::Meta, b"iloc") => {
            // Versions 0 and 1 count items in 16 bits. Version 2
            // counts in 32, after the version and flags and two
            // bytes of field sizes.
            if body.first() != Some(&2) {
                return Ok(());
            }
            let (Some(count), Some(items)) = (body.get(6..10), body.get(10..)) else {
                return Ok(());
            };
            let count = u32::from_be_bytes(count.try_into().expect("a 4-byte slice"));
            check_count(kind, count, items.len() as u64, MIN_ILOC_V2_ITEM)
        }
        _ => Ok(()),
    }
}

/// Refuse a count above the 16-bit range whose entries, at
/// `min_entry` bytes each, could not fit in `room`.
fn check_count(kind: [u8; 4], count: u32, room: u64, min_entry: u64) -> Result<(), Refusal> {
    if count <= u32::from(u16::MAX) || u64::from(count) * min_entry <= room {
        return Ok(());
    }
    Err(Refusal::Count { kind, count, room })
}

/// What [`inspect`] reads from the item table ahead of the
/// decode.
struct Inspect {
    /// The primary item's declared size, from its `ispe`
    /// property, when the container names one.
    primary_size: Option<(u32, u32)>,
}

/// A forward byte cursor that reads big-endian integers and
/// returns `None` the moment the slice runs short, so a
/// truncated box stops the read without a panic.
struct Cur<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Cur<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    /// The next `n` bytes as a big-endian integer (`n` at most
    /// 8); `n` of 0 reads nothing and is 0, the width the item
    /// format gives an omitted field.
    fn uint(&mut self, n: usize) -> Option<u64> {
        let end = self.at.checked_add(n)?;
        let slice = self.data.get(self.at..end)?;
        self.at = end;
        Some(slice.iter().fold(0u64, |v, &b| (v << 8) | u64::from(b)))
    }

    /// Advance past `n` bytes, or `None` past the end.
    fn skip(&mut self, n: usize) -> Option<()> {
        let end = self.at.checked_add(n)?;
        if end > self.data.len() {
            return None;
        }
        self.at = end;
        Some(())
    }
}

/// Call `f` with the kind, the body, and the body's offset in
/// `data` of each box in `data`, in order, the way [`walk`]
/// reads them: the body excludes the header, a size-0 box runs
/// to the end of `data`, and the iteration stops at a header
/// the remaining bytes cannot hold or a box that overruns
/// `data` (which [`walk`] has already refused for the boxes the
/// parser reads). Allocates nothing.
fn each_box<'a>(data: &'a [u8], mut f: impl FnMut([u8; 4], &'a [u8], usize)) {
    let mut rest = data;
    while let Ok(Some(header)) = Header::read(rest) {
        let room = rest.len() as u64;
        let size = header.size.unwrap_or(room);
        if size > room {
            break;
        }
        let at = data.len() - rest.len() + header.len;
        let (this, next) = rest.split_at(size as usize);
        f(header.kind, &this[header.len..], at);
        rest = next;
    }
}

/// The most top-level `mdat` boxes whose place in the file
/// [`inspect`] keeps, to resolve a length-0 extent to the rest
/// of the `mdat` that holds its offset. An encoder writes one.
/// Past this many, a length-0 extent in an `mdat` not kept
/// counts as every byte of the `mdat` boxes not kept, which can
/// only overstate what the parser copies.
const KEPT_MDATS: usize = 16;

/// Where the file holds the data an item is copied from: the
/// top-level `mdat` payloads (construction method 0) and the
/// `idat` payload (method 1). Fixed size, so reading it
/// allocates nothing.
struct ItemData {
    /// The file offset and length of the first [`KEPT_MDATS`]
    /// `mdat` payloads, in file order (the order mp4parse
    /// searches them in).
    mdats: [(u64, u64); KEPT_MDATS],
    kept: usize,
    /// Every `mdat` payload byte, kept or not.
    total_mdat: u64,
    idat_len: u64,
}

impl ItemData {
    /// The bytes the file holds for an item built with
    /// `method`.
    fn holds(&self, method: u8) -> u64 {
        if method == 1 {
            self.idat_len
        } else {
            self.total_mdat
        }
    }

    /// The bytes a length-0 extent at `offset` copies: the rest
    /// of the `idat` (method 1, where offsets count from the
    /// start of the `idat` payload), or the rest of the first
    /// `mdat` whose payload holds the file offset `offset`, end
    /// included, as mp4parse resolves it. An offset outside the
    /// data is one the parser refuses (it copies nothing), so it
    /// counts as 0.
    fn rest(&self, method: u8, offset: u64) -> u64 {
        if method == 1 {
            return self.idat_len.saturating_sub(offset);
        }
        let kept = &self.mdats[..self.kept];
        for &(start, len) in kept {
            if let Some(into) = offset.checked_sub(start)
                && into <= len
            {
                return len - into;
            }
        }
        let kept_bytes = kept
            .iter()
            .fold(0u64, |sum, &(_, len)| sum.saturating_add(len));
        self.total_mdat.saturating_sub(kept_bytes)
    }
}

/// Read the item table the decoder builds from: refuse an item
/// whose extents overrun the cap or sum past the data the file
/// holds for it, and report the primary item's declared size.
/// Bounds-checked throughout, so a truncated or malformed table
/// stops without a panic and is left to the parser.
fn inspect(bytes: &[u8]) -> Result<Inspect, Refusal> {
    // The top level holds the data an item is copied from (the
    // `mdat` payloads) and the `meta` box that describes it.
    let mut data = ItemData {
        mdats: [(0, 0); KEPT_MDATS],
        kept: 0,
        total_mdat: 0,
        idat_len: 0,
    };
    let mut meta_body: Option<&[u8]> = None;
    each_box(bytes, |kind, body, at| match &kind {
        b"mdat" => {
            let len = body.len() as u64;
            if data.kept < KEPT_MDATS {
                data.mdats[data.kept] = (at as u64, len);
                data.kept += 1;
            }
            data.total_mdat = data.total_mdat.saturating_add(len);
        }
        b"meta" => meta_body = meta_body.or(Some(body)),
        _ => {}
    });
    let Some(meta) = meta_body else {
        return Ok(Inspect { primary_size: None });
    };
    // `meta` is a full box: version and flags before its
    // children.
    let meta = meta.get(4..).unwrap_or_default();
    let mut iloc: Option<&[u8]> = None;
    let mut iprp: Option<&[u8]> = None;
    let mut pitm: Option<&[u8]> = None;
    each_box(meta, |kind, body, _| match &kind {
        b"idat" => data.idat_len = body.len() as u64,
        b"iloc" => iloc = iloc.or(Some(body)),
        b"iprp" => iprp = iprp.or(Some(body)),
        b"pitm" => pitm = pitm.or(Some(body)),
        _ => {}
    });
    if let Some(iloc) = iloc {
        check_iloc(iloc, &data)?;
    }
    let primary_size = match (pitm.and_then(primary_item_id), iprp) {
        (Some(id), Some(iprp)) => primary_ispe(iprp, id),
        _ => None,
    };
    Ok(Inspect { primary_size })
}

/// Walk `iloc` (versions 0, 1, 2, every field-size variant),
/// refusing an item with too many extents or whose extents sum
/// past the bytes the file holds for its construction method.
/// A table mp4parse refuses outright (a field size other than
/// 0, 4, or 8 bytes, an offset past 64 bits, a truncated item)
/// is left to it.
fn check_iloc(body: &[u8], data: &ItemData) -> Result<(), Refusal> {
    let mut c = Cur::new(body);
    let Some(version) = c.uint(1) else {
        return Ok(());
    };
    if version > 2 || c.skip(3).is_none() {
        return Ok(());
    }
    let (Some(sizes0), Some(sizes1)) = (c.uint(1), c.uint(1)) else {
        return Ok(());
    };
    let offset_size = (sizes0 >> 4) as usize;
    let length_size = (sizes0 & 0xf) as usize;
    let base_offset_size = (sizes1 >> 4) as usize;
    let index_size = (sizes1 & 0xf) as usize;
    // Versions 0 and 1 carry 16-bit item IDs and counts, version
    // 2 carries 32-bit. A size other than 0, 4, or 8 bytes, or a
    // version 0 index nibble aside (reserved there), is a table
    // the parser refuses before it copies anything.
    let widths = if version == 0 {
        &[offset_size, length_size, base_offset_size][..]
    } else {
        &[offset_size, length_size, base_offset_size, index_size][..]
    };
    if widths.iter().any(|&s| !matches!(s, 0 | 4 | 8)) {
        return Ok(());
    }
    let id_bytes = if version == 2 { 4 } else { 2 };
    let Some(item_count) = c.uint(id_bytes) else {
        return Ok(());
    };
    for _ in 0..item_count {
        let Some(item_id) = c.uint(id_bytes) else {
            return Ok(());
        };
        let item_id = item_id as u32;
        let method = if version == 0 {
            0
        } else {
            match c.uint(2) {
                Some(field) => (field & 0xf) as u8,
                None => return Ok(()),
            }
        };
        // Data reference index, then the base offset.
        if c.skip(2).is_none() {
            return Ok(());
        }
        let Some(base_offset) = c.uint(base_offset_size) else {
            return Ok(());
        };
        let Some(extent_count) = c.uint(2) else {
            return Ok(());
        };
        if extent_count > MAX_EXTENTS {
            return Err(Refusal::Extents {
                item: item_id,
                extents: extent_count,
            });
        }
        // An explicit length is what the parser copies; a length
        // of 0 copies the rest of the data that holds the
        // extent's offset (the base offset plus its own). The sum
        // saturates: past 64 bits it is past any file.
        let mut needs: u64 = 0;
        for _ in 0..extent_count {
            if version != 0 && c.skip(index_size).is_none() {
                return Ok(());
            }
            let (Some(offset), Some(len)) = (c.uint(offset_size), c.uint(length_size)) else {
                return Ok(());
            };
            let Some(offset) = base_offset.checked_add(offset) else {
                // The parser refuses the table on this overflow.
                return Ok(());
            };
            let copies = if len == 0 {
                data.rest(method, offset)
            } else {
                len
            };
            needs = needs.saturating_add(copies);
        }
        // Construction method 2 builds the item from other
        // items; the decoder refuses it, so only the cap applies.
        if method == 2 {
            continue;
        }
        let holds = data.holds(method);
        if needs > holds {
            return Err(Refusal::Amplified {
                item: item_id,
                needs,
                holds,
            });
        }
    }
    Ok(())
}

/// The primary item's id, from `pitm` (both id widths).
fn primary_item_id(body: &[u8]) -> Option<u32> {
    let mut c = Cur::new(body);
    let version = c.uint(1)?;
    c.skip(3)?;
    if version == 0 {
        c.uint(2).map(|id| id as u32)
    } else {
        c.uint(4).map(|id| id as u32)
    }
}

/// The primary item's declared size, from the `ispe` property
/// its `ipma` entry points at inside `ipco`. `None` when the
/// container names none, which leaves the sizing to the decoder.
fn primary_ispe(iprp: &[u8], primary: u32) -> Option<(u32, u32)> {
    let mut ipco: Option<&[u8]> = None;
    // The property indices (1-based) associated with the primary
    // item, from every `ipma`. Bounded so a flood of
    // associations cannot grow this without limit.
    let mut indices: Vec<u16> = Vec::new();
    each_box(iprp, |kind, body, _| match &kind {
        b"ipco" => ipco = ipco.or(Some(body)),
        b"ipma" => collect_ipma(body, primary, &mut indices),
        _ => {}
    });
    let ipco = ipco?;
    if indices.is_empty() {
        return None;
    }
    let mut found = None;
    let mut index: u16 = 1;
    each_box(ipco, |kind, body, _| {
        if found.is_none() && &kind == b"ispe" && indices.contains(&index) {
            found = read_ispe(body);
        }
        index = index.saturating_add(1);
    });
    found
}

/// Add the property indices `ipma` associates with the primary
/// item to `out` (both property-index widths from the flags,
/// both item-id widths from the version), up to a small cap.
fn collect_ipma(body: &[u8], primary: u32, out: &mut Vec<u16>) {
    let mut c = Cur::new(body);
    let (Some(version), Some(flags)) = (c.uint(1), c.uint(3)) else {
        return;
    };
    let assoc_bytes = if flags & 1 == 1 { 2 } else { 1 };
    let index_bits = assoc_bytes * 8 - 1;
    let Some(entry_count) = c.uint(4) else {
        return;
    };
    for _ in 0..entry_count {
        let id = if version == 0 { c.uint(2) } else { c.uint(4) };
        let (Some(id), Some(assoc_count)) = (id, c.uint(1)) else {
            return;
        };
        for _ in 0..assoc_count {
            let Some(association) = c.uint(assoc_bytes) else {
                return;
            };
            if id as u32 == primary && out.len() < 256 {
                out.push((association & ((1u64 << index_bits) - 1)) as u16);
            }
        }
    }
}

/// The width and height an `ispe` declares.
fn read_ispe(body: &[u8]) -> Option<(u32, u32)> {
    let mut c = Cur::new(body);
    c.skip(4)?;
    Some((c.uint(4)? as u32, c.uint(4)? as u32))
}

/// Refuse a declared size the decoder's own checks would refuse
/// after it allocated the picture: past the per-axis cap, or
/// past the decode allocation budget at four bytes a pixel.
fn within_decode_limits(width: u32, height: u32) -> Result<(), Refusal> {
    let (w, h) = (u64::from(width), u64::from(height));
    if w > u64::from(crate::engine::MAX_DECODE_DIMENSION)
        || h > u64::from(crate::engine::MAX_DECODE_DIMENSION)
        || w.saturating_mul(h).saturating_mul(4) > crate::engine::MAX_DECODE_ALLOC
    {
        return Err(Refusal::Dimension { width, height });
    }
    Ok(())
}

/// The primary item's declared size from an AVIF container's
/// `ispe`, or `None` when the bytes are not an AVIF the walk
/// accepts or the container names no size. Reads headers only:
/// sizing an AVIF no longer needs a decode (issue #522).
pub(crate) fn declared_size(bytes: &[u8]) -> Option<(u32, u32)> {
    if !matches!(image::guess_format(bytes), Ok(image::ImageFormat::Avif)) {
        return None;
    }
    walk(bytes, Level::File).ok()?;
    inspect(bytes).ok()?.primary_size
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decode fuzzer's input from the issue: a `meta`
    /// whose `hdlr` declares 1,014,169,633 bytes.
    const FUZZER_INPUT: &[u8] = include_bytes!("../fuzz/corpus/decode/avif-hdlr-overrun");

    fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        claims(kind, (8 + body.len()) as u32, body)
    }

    /// A box whose 32-bit size field says `declared`.
    fn claims(kind: &[u8; 4], declared: u32, body: &[u8]) -> Vec<u8> {
        let mut out = declared.to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    /// A box with a 64-bit size.
    fn claims_large(kind: &[u8; 4], declared: u64, body: &[u8]) -> Vec<u8> {
        let mut out = 1u32.to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(&declared.to_be_bytes());
        out.extend_from_slice(body);
        out
    }

    fn full(version: u8, rest: &[u8]) -> Vec<u8> {
        let mut out = vec![version, 0, 0, 0];
        out.extend_from_slice(rest);
        out
    }

    fn concat(parts: &[Vec<u8>]) -> Vec<u8> {
        parts.concat()
    }

    fn ftyp(brands: &[u8]) -> Vec<u8> {
        let mut body = b"avif\0\0\0\0".to_vec();
        body.extend_from_slice(brands);
        bx(b"ftyp", &body)
    }

    fn hdlr() -> Vec<u8> {
        let mut handler = vec![0; 4];
        handler.extend_from_slice(b"pict");
        handler.extend_from_slice(&[0; 13]);
        bx(b"hdlr", &full(0, &handler))
    }

    fn meta(children: &[Vec<u8>]) -> Vec<u8> {
        bx(b"meta", &full(0, &concat(children)))
    }

    /// The layout an AVIF encoder writes: every container the
    /// walk enters, each well formed.
    fn well_formed() -> Vec<u8> {
        let mut iloc = vec![0x44, 0x00];
        iloc.extend_from_slice(&[0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0x50, 0, 0, 0, 4]);
        concat(&[
            ftyp(b"mif1miaf"),
            meta(&[
                hdlr(),
                bx(b"pitm", &full(0, &[0, 1])),
                bx(b"iloc", &full(0, &iloc)),
                bx(
                    b"iinf",
                    &full(
                        0,
                        &concat(&[vec![0, 1], bx(b"infe", &full(2, b"\0\x01\0\0av01\0"))]),
                    ),
                ),
                bx(b"iref", &full(0, &bx(b"auxl", &[0, 2, 0, 1, 0, 1]))),
                bx(
                    b"iprp",
                    &concat(&[
                        bx(
                            b"ipco",
                            &concat(&[
                                bx(b"ispe", &full(0, &[0, 0, 0, 8, 0, 0, 0, 6])),
                                bx(b"av1C", &[0x81, 0x3f, 0, 0]),
                                bx(b"pixi", &full(0, &[3, 8, 8, 8])),
                            ]),
                        ),
                        bx(b"ipma", &full(0, &[0, 0, 0, 1, 0, 1, 2, 0x81, 0x02])),
                    ]),
                ),
            ]),
            bx(b"mdat", &[0x12, 0, 0x0a, 0x08]),
        ])
    }

    fn refusal(bytes: &[u8]) -> Option<Refusal> {
        walk(bytes, Level::File).err()
    }

    fn overrun_of(bytes: &[u8]) -> [u8; 4] {
        match refusal(bytes) {
            Some(Refusal::Overrun { kind, .. }) => kind,
            other => panic!("expected an overrun, got {other:?}"),
        }
    }

    #[test]
    fn the_fuzzer_input_is_refused_at_its_hdlr() {
        assert_eq!(
            refusal(FUZZER_INPUT),
            Some(Refusal::Overrun {
                kind: *b"hdlr",
                declared: 1_014_169_633,
                room: 198,
            })
        );
        let Err(EngineError::Decode(message)) = screen(FUZZER_INPUT) else {
            panic!("the screen let the fuzzer input through");
        };
        assert_eq!(
            message,
            "AVIF container refused: box 'hdlr' declares 1014169633 bytes, but only 198 remain in its parent"
        );
    }

    #[test]
    fn a_well_formed_layout_passes_borrowed() {
        let bytes = well_formed();
        assert_eq!(refusal(&bytes), None);
        let screened = screen(&bytes).unwrap();
        assert!(matches!(screened, Cow::Borrowed(b) if b.as_ptr() == bytes.as_ptr()));
    }

    #[test]
    fn every_box_the_parser_reads_must_fit_its_parent() {
        let past = |kind: &[u8; 4], body: &[u8]| claims(kind, 0x7FFF_0000, body);
        // A top-level meta or mdat past the end of the file.
        assert_eq!(
            overrun_of(&concat(&[ftyp(b""), past(b"meta", &full(0, &[]))])),
            *b"meta"
        );
        assert_eq!(
            overrun_of(&concat(&[ftyp(b""), meta(&[hdlr()]), past(b"mdat", b"")])),
            *b"mdat"
        );
        // The ftyp itself.
        assert_eq!(overrun_of(&past(b"ftyp", b"avif\0\0\0\0mif1")), *b"ftyp");
        // Each child of meta, read or skipped.
        for kind in [
            b"hdlr", b"pitm", b"iloc", b"idat", b"iinf", b"iref", b"iprp", b"dinf",
        ] {
            let bytes = concat(&[ftyp(b""), meta(&[hdlr(), past(kind, &full(0, &[0; 8]))])]);
            assert_eq!(overrun_of(&bytes), *kind);
        }
        // One level down: an iinf entry, an iref reference, the
        // children of iprp.
        let iinf = bx(
            b"iinf",
            &full(0, &concat(&[vec![0, 1], past(b"infe", b"")])),
        );
        let iref = bx(b"iref", &full(0, &past(b"auxl", b"")));
        let ipma = bx(b"iprp", &concat(&[bx(b"ipco", b""), past(b"ipma", b"")]));
        // mp4parse lets an ipco run past its iprp (the ipma
        // behind it is lost, so the image fails later); the walk
        // refuses it up front.
        let ipco = bx(b"iprp", &past(b"ipco", &bx(b"ispe", &full(0, &[0; 8]))));
        for (child, kind) in [
            (iinf, b"infe"),
            (iref, b"auxl"),
            (ipma, b"ipma"),
            (ipco, b"ipco"),
        ] {
            let bytes = concat(&[ftyp(b""), meta(&[hdlr(), child])]);
            assert_eq!(overrun_of(&bytes), *kind);
        }
        // Three levels down: a property inside ipco inside iprp.
        let deep = bx(b"iprp", &bx(b"ipco", &past(b"colr", b"prof")));
        assert_eq!(
            overrun_of(&concat(&[ftyp(b""), meta(&[hdlr(), deep])])),
            *b"colr"
        );
        // A box that fits the file but not its own parent.
        let mut ipco = bx(b"ipco", &bx(b"ispe", &full(0, &[0; 8])));
        let ispe_at = 8;
        ipco[ispe_at + 3] += 4;
        let tail = bx(b"free", &[0; 16]);
        let bytes = concat(&[ftyp(b""), meta(&[hdlr(), bx(b"iprp", &ipco)]), tail]);
        assert_eq!(overrun_of(&bytes), *b"ispe");
    }

    #[test]
    fn a_largesize_is_held_to_the_same_bound() {
        let with = |idat: Vec<u8>| concat(&[ftyp(b""), meta(&[hdlr(), idat])]);
        assert_eq!(refusal(&with(claims_large(b"idat", 16 + 4, &[0; 4]))), None);
        assert_eq!(
            overrun_of(&with(claims_large(b"idat", 1 << 36, &[0; 4]))),
            *b"idat"
        );
        assert_eq!(
            overrun_of(&with(claims_large(b"idat", u64::MAX, &[0; 4]))),
            *b"idat"
        );
        assert_eq!(
            refusal(&with(claims_large(b"idat", 15, &[0; 4]))),
            Some(Refusal::Undersized {
                kind: *b"idat",
                declared: 15,
            })
        );
        // At the top level too, mdat included.
        let mdat = |size| {
            concat(&[
                ftyp(b""),
                meta(&[hdlr()]),
                claims_large(b"mdat", size, &[7; 8]),
            ])
        };
        assert_eq!(refusal(&mdat(24)), None);
        assert_eq!(overrun_of(&mdat(1 << 40)), *b"mdat");
        // A largesize cut off by the end of its parent is a run
        // of boxes ending short of a header: the parser stops.
        let cut = concat(&[
            ftyp(b""),
            meta(&[hdlr(), vec![0, 0, 0, 1, b'i', b'd', b'a', b't', 0, 0]]),
        ]);
        assert_eq!(refusal(&cut), None);
    }

    #[test]
    fn size_zero_runs_to_the_end_of_its_parent() {
        // A trailing mdat of size 0 is how a streaming writer
        // ends a file.
        let bytes = concat(&[ftyp(b""), meta(&[hdlr()]), claims(b"mdat", 0, &[9; 40])]);
        assert_eq!(refusal(&bytes), None);
        // A size-0 container is entered all the same.
        let open_meta = claims(
            b"meta",
            0,
            &full(0, &concat(&[hdlr(), claims(b"idat", 0x10_0000, b"")])),
        );
        assert_eq!(overrun_of(&concat(&[ftyp(b""), open_meta])), *b"idat");
        let open_ipco = bx(
            b"iprp",
            &claims(b"ipco", 0, &claims(b"colr", 0x10_0000, b"prof")),
        );
        assert_eq!(
            overrun_of(&concat(&[ftyp(b""), meta(&[hdlr(), open_ipco])])),
            *b"colr"
        );
        // And a size-0 box fits, wherever it is.
        let open_ispe = bx(b"iprp", &bx(b"ipco", &claims(b"ispe", 0, &[0; 12])));
        assert_eq!(
            refusal(&concat(&[ftyp(b""), meta(&[hdlr(), open_ispe])])),
            None
        );
    }

    #[test]
    fn a_size_smaller_than_its_header_is_refused() {
        for declared in 2..8 {
            let bytes = concat(&[
                ftyp(b""),
                meta(&[hdlr(), claims(b"pitm", declared, &[0; 8])]),
            ]);
            assert_eq!(
                refusal(&bytes),
                Some(Refusal::Undersized {
                    kind: *b"pitm",
                    declared: u64::from(declared),
                })
            );
        }
    }

    #[test]
    fn trailing_data_the_parser_skips_is_left_to_it() {
        let file = concat(&[ftyp(b""), meta(&[hdlr()]), bx(b"mdat", &[1; 8])]);
        // A box the parser only skips may run past the end.
        for kind in [b"free", b"moov", b"ftyp", b"uuid"] {
            let bytes = concat(&[file.clone(), claims(kind, 0x7FFF_0000, &[0; 8])]);
            assert_eq!(refusal(&bytes), None, "{}", kind.escape_ascii());
        }
        // So may a few bytes too short for any box header.
        for tail in 1..8 {
            let bytes = concat(&[file.clone(), vec![0xAA; tail]]);
            assert_eq!(refusal(&bytes), None);
        }
        // Nothing after it is read.
        let bytes = concat(&[
            file.clone(),
            claims(b"free", 0x7FFF_0000, &[0; 8]),
            claims(b"meta", 0x7FFF_0000, b""),
        ]);
        assert_eq!(refusal(&bytes), None);
        // A box the parser reads may not.
        for kind in [b"meta", b"mdat"] {
            let bytes = concat(&[file.clone(), claims(kind, 0x7FFF_0000, &[0; 8])]);
            assert_eq!(overrun_of(&bytes), *kind);
        }
    }

    #[test]
    fn a_32_bit_entry_count_must_be_backed_by_bytes() {
        let with = |child: Vec<u8>| concat(&[ftyp(b""), meta(&[hdlr(), child])]);
        let infe = || bx(b"infe", &full(2, b"\0\x01\0\0av01\0"));
        let iinf = |version: u8, count: u32, entries: usize| {
            let mut body = if version == 0 {
                (count as u16).to_be_bytes().to_vec()
            } else {
                count.to_be_bytes().to_vec()
            };
            for _ in 0..entries {
                body.extend(infe());
            }
            bx(b"iinf", &full(version, &body))
        };
        // Version 0 cannot count past the 16-bit range.
        assert_eq!(refusal(&with(iinf(0, 0xFFFF, 1))), None);
        // Version 1 may overstate within it, as the parser allows.
        assert_eq!(refusal(&with(iinf(1, 0xFFFF, 1))), None);
        assert_eq!(
            refusal(&with(iinf(1, 0x1_0000, 1))),
            Some(Refusal::Count {
                kind: *b"iinf",
                count: 0x1_0000,
                room: 21,
            })
        );
        assert!(matches!(
            refusal(&with(iinf(1, u32::MAX, 0))),
            Some(Refusal::Count { .. })
        ));
        // A count past 65,535 the entries do back.
        let many = 0x1_0000;
        let mut body = (many as u32).to_be_bytes().to_vec();
        for _ in 0..many {
            body.extend(bx(b"free", b""));
        }
        assert_eq!(refusal(&with(bx(b"iinf", &full(1, &body)))), None);

        let iloc = |version: u8, count: u32, item_bytes: usize| {
            let mut body = vec![0, 0];
            body.extend_from_slice(&count.to_be_bytes());
            body.resize(body.len() + item_bytes, 0);
            bx(b"iloc", &full(version, &body))
        };
        assert_eq!(refusal(&with(iloc(2, 0xFFFF, 0))), None);
        assert_eq!(
            refusal(&with(iloc(2, 0x1000_0000, 0))),
            Some(Refusal::Count {
                kind: *b"iloc",
                count: 0x1000_0000,
                room: 0,
            })
        );
        assert_eq!(refusal(&with(iloc(2, 0x1_0000, 10 * 0x1_0000))), None);
        assert!(matches!(
            refusal(&with(iloc(2, 0x1_0000, 10 * 0x1_0000 - 1))),
            Some(Refusal::Count { .. })
        ));
        // Versions 0 and 1 hold a 16-bit count there.
        assert_eq!(refusal(&with(iloc(1, 0x1000_0000, 0))), None);
        // A box too short to hold its count is the parser's to
        // refuse.
        assert_eq!(refusal(&with(bx(b"iinf", &full(1, &[0xFF, 0xFF])))), None);
        assert_eq!(refusal(&with(bx(b"iloc", &full(2, &[0, 0, 0xFF])))), None);
    }

    /// An `iloc` item for the test builder: an id, a
    /// construction method, and its `(offset, length)` extents.
    type IlocItem<'a> = (u32, u8, &'a [(u32, u32)]);

    /// A version 1 `iloc` with 4-byte offsets and lengths, no
    /// base offset and no index, holding `items`.
    fn iloc_v1(items: &[IlocItem]) -> Vec<u8> {
        let mut body = vec![0x44, 0x00];
        body.extend_from_slice(&(items.len() as u16).to_be_bytes());
        for (id, method, extents) in items {
            body.extend_from_slice(&(*id as u16).to_be_bytes());
            body.extend_from_slice(&u16::from(*method).to_be_bytes());
            body.extend_from_slice(&0u16.to_be_bytes());
            body.extend_from_slice(&(extents.len() as u16).to_be_bytes());
            for (offset, length) in *extents {
                body.extend_from_slice(&offset.to_be_bytes());
                body.extend_from_slice(&length.to_be_bytes());
            }
        }
        bx(b"iloc", &full(1, &body))
    }

    /// `pitm` + `ipma` naming item 1's one property (index 1)
    /// and an `iprp`/`ipco` whose first property is that `ispe`.
    fn primary_with_ispe(width: u32, height: u32) -> Vec<Vec<u8>> {
        let iprp = bx(
            b"iprp",
            &concat(&[
                bx(b"ipco", &{
                    let mut body = width.to_be_bytes().to_vec();
                    body.extend_from_slice(&height.to_be_bytes());
                    bx(b"ispe", &full(0, &body))
                }),
                bx(b"ipma", &full(0, &[0, 0, 0, 1, 0, 1, 1, 0x01])),
            ]),
        );
        vec![bx(b"pitm", &full(0, &[0, 1])), iprp]
    }

    /// `meta` holding `hdlr` and then `children`.
    fn meta_with(children: &[Vec<u8>]) -> Vec<u8> {
        let mut inside = vec![hdlr()];
        inside.extend_from_slice(children);
        meta(&inside)
    }

    /// `ftyp` + `meta(children)` + an `mdat` of `mdat_len` bytes.
    fn container(children: &[Vec<u8>], mdat_len: usize) -> Vec<u8> {
        concat(&[
            ftyp(b"mif1miaf"),
            meta_with(children),
            bx(b"mdat", &vec![0u8; mdat_len]),
        ])
    }

    fn item_refusal(bytes: &[u8]) -> Option<Refusal> {
        inspect(bytes).err()
    }

    const MAX_DECODE_SIDE: u32 = crate::engine::MAX_DECODE_DIMENSION;

    #[test]
    fn consistent_extents_pass() {
        // One extent filling the mdat, and two partitioning it.
        let single = container(&[iloc_v1(&[(1, 0, &[(0, 100)])])], 100);
        assert_eq!(item_refusal(&single), None);
        let multi = container(&[iloc_v1(&[(1, 0, &[(0, 60), (60, 40)])])], 100);
        assert_eq!(item_refusal(&multi), None);
        assert_eq!(screen(&single).map(|_| ()), Ok(()));
    }

    #[test]
    fn extents_summing_past_the_data_are_refused() {
        // Two extents of 64 over the same 100-byte mdat: 128 > 100.
        let bytes = container(&[iloc_v1(&[(1, 0, &[(0, 64), (0, 64)])])], 100);
        assert_eq!(
            item_refusal(&bytes),
            Some(Refusal::Amplified {
                item: 1,
                needs: 128,
                holds: 100,
            })
        );
        // The amplification shape: one 64-byte range listed many
        // times still sums past the mdat.
        let extents: Vec<(u32, u32)> = (0..8).map(|_| (0, 64)).collect();
        let many = container(&[iloc_v1(&[(1, 0, &extents)])], 100);
        assert!(matches!(
            item_refusal(&many),
            Some(Refusal::Amplified { item: 1, .. })
        ));
    }

    /// Build a container whose layout does not depend on the
    /// offsets it carries (first with a placeholder offset), then
    /// build it again with the file offset of its last box's
    /// payload (the `mdat` of `mdat_len` bytes) passed in, so
    /// extents can point into it.
    fn pointing_into_mdat(mdat_len: usize, build: impl Fn(u64) -> Vec<u8>) -> Vec<u8> {
        let at = (build(1 << 20).len() - mdat_len) as u64;
        let bytes = build(at);
        assert_eq!(bytes.len() - mdat_len, at as usize, "the layout moved");
        bytes
    }

    #[test]
    fn a_zero_length_extent_counts_as_the_rest_of_the_data() {
        let in_mdat = |extents: &'static [(u32, u32)]| {
            pointing_into_mdat(100, move |at| {
                let at = at as u32;
                let extents: Vec<(u32, u32)> = extents.iter().map(|&(o, l)| (at + o, l)).collect();
                container(&[iloc_v1(&[(1, 0, &extents)])], 100)
            })
        };
        // A single length-0 extent at the start of the mdat runs
        // to its end: exactly what the file holds.
        assert_eq!(item_refusal(&in_mdat(&[(0, 0)])), None);
        // A second one copies the mdat twice.
        assert_eq!(
            item_refusal(&in_mdat(&[(0, 0), (0, 0)])),
            Some(Refusal::Amplified {
                item: 1,
                needs: 200,
                holds: 100,
            })
        );
        // An item split in two, the tail running to the end:
        // 60 bytes, then the remaining 40. Counting a length-0
        // extent as the whole mdat would refuse this valid
        // layout.
        assert_eq!(item_refusal(&in_mdat(&[(0, 60), (60, 0)])), None);
        // The tail starting inside the first part copies the
        // overlap twice: 60 + 70.
        assert_eq!(
            item_refusal(&in_mdat(&[(0, 60), (30, 0)])),
            Some(Refusal::Amplified {
                item: 1,
                needs: 130,
                holds: 100,
            })
        );
        // With two mdats, a length-0 extent runs to the end of
        // the one that holds its offset, not of the file.
        let two = pointing_into_mdat(30, |second| {
            let second = second as u32;
            let first = second - 8 - 70;
            concat(&[
                ftyp(b"mif1miaf"),
                meta_with(&[iloc_v1(&[(1, 0, &[(first, 70), (second, 0)])])]),
                bx(b"mdat", &[0u8; 70]),
                bx(b"mdat", &[0u8; 30]),
            ])
        });
        assert_eq!(item_refusal(&two), None);
        // Method 1 counts against the idat, offsets from its
        // start, not against the mdat.
        let idat = |extents: &[(u32, u32)]| {
            concat(&[
                ftyp(b"mif1miaf"),
                meta_with(&[bx(b"idat", &[0u8; 40]), iloc_v1(&[(1, 1, extents)])]),
                bx(b"mdat", &[0u8; 100]),
            ])
        };
        assert_eq!(item_refusal(&idat(&[(0, 0)])), None);
        assert_eq!(item_refusal(&idat(&[(0, 25), (25, 0)])), None);
        assert_eq!(
            item_refusal(&idat(&[(0, 0), (0, 0)])),
            Some(Refusal::Amplified {
                item: 1,
                needs: 80,
                holds: 40,
            })
        );
    }

    /// An `iloc` item for [`iloc_with`]: an id, a construction
    /// method, a base offset, and its `(offset, length)` extents.
    type AnyItem<'a> = (u32, u8, u64, &'a [(u64, u64)]);

    /// An `iloc` of any version and field sizes. Every index
    /// field (versions 1 and 2) carries a nonzero filler, and the
    /// version 0 index nibble (reserved there) is set, so a
    /// reader that misplaces either field misreads what follows.
    fn iloc_with(version: u8, sizes: [usize; 4], items: &[AnyItem]) -> Vec<u8> {
        let [offset_size, length_size, base_size, index_size] = sizes;
        let field = |out: &mut Vec<u8>, value: u64, size: usize| {
            if size > 0 {
                out.extend_from_slice(&value.to_be_bytes()[8 - size..]);
            }
        };
        let index_nibble = if version == 0 { 4 } else { index_size };
        let mut body = vec![
            ((offset_size << 4) | length_size) as u8,
            ((base_size << 4) | index_nibble) as u8,
        ];
        let id_size = if version == 2 { 4 } else { 2 };
        field(&mut body, items.len() as u64, id_size);
        for &(id, method, base, extents) in items {
            field(&mut body, u64::from(id), id_size);
            if version != 0 {
                body.extend_from_slice(&u16::from(method).to_be_bytes());
            }
            body.extend_from_slice(&0u16.to_be_bytes());
            field(&mut body, base, base_size);
            body.extend_from_slice(&(extents.len() as u16).to_be_bytes());
            for &(offset, length) in extents {
                if version != 0 {
                    field(&mut body, 0xA5A5_A5A5_A5A5_A5A5, index_size);
                }
                field(&mut body, offset, offset_size);
                field(&mut body, length, length_size);
            }
        }
        bx(b"iloc", &full(version, &body))
    }

    #[test]
    fn every_iloc_version_and_field_size_reads_alike() {
        // The same tables, written in each version with every
        // offset, length, base offset, and index size: a
        // consistent pair of items passes, an item whose extents
        // sum past the data is refused by its own id, a length-0
        // tail resolves through the base offset, and the cap
        // applies. The item ids need 32 bits in version 2.
        let mut combinations = 0;
        for version in 0..=2u8 {
            let id = |n: u32| if version == 2 { 0x0001_0000 + n } else { n };
            let indices: &[usize] = if version == 0 { &[0] } else { &[0, 4, 8] };
            for offset_size in [4, 8] {
                for length_size in [4, 8] {
                    for base_size in [0, 4, 8] {
                        for &index_size in indices {
                            let sizes = [offset_size, length_size, base_size, index_size];
                            let build = |at: u64, b_extents: &[(u64, u64)]| {
                                // The base offset, when there is a
                                // field for it, sits 10 bytes before
                                // the mdat payload.
                                let base = if base_size > 0 { at - 10 } else { 0 };
                                let local = |o: u64| at + o - base;
                                let a = [(local(0), 30), (local(30), 30)];
                                let b: Vec<(u64, u64)> =
                                    b_extents.iter().map(|&(o, l)| (local(o), l)).collect();
                                container(
                                    &[iloc_with(
                                        version,
                                        sizes,
                                        &[(id(1), 0, base, &a), (id(2), 0, base, &b)],
                                    )],
                                    100,
                                )
                            };
                            let case = |b: &'static [(u64, u64)]| {
                                item_refusal(&pointing_into_mdat(100, |at| build(at, b)))
                            };
                            let what = format!("v{version} sizes {sizes:?}");
                            assert_eq!(case(&[(60, 40)]), None, "{what}");
                            assert_eq!(case(&[(0, 60), (60, 0)]), None, "{what}");
                            assert_eq!(
                                case(&[(0, 64), (0, 64)]),
                                Some(Refusal::Amplified {
                                    item: id(2),
                                    needs: 128,
                                    holds: 100,
                                }),
                                "{what}"
                            );
                            assert_eq!(
                                case(&[(0, 60), (30, 0)]),
                                Some(Refusal::Amplified {
                                    item: id(2),
                                    needs: 130,
                                    holds: 100,
                                }),
                                "{what}"
                            );
                            let over: Vec<(u64, u64)> = (0..=MAX_EXTENTS).map(|i| (i, 1)).collect();
                            let capped = container(
                                &[iloc_with(version, sizes, &[(id(3), 0, 0, &over)])],
                                100,
                            );
                            assert_eq!(
                                item_refusal(&capped),
                                Some(Refusal::Extents {
                                    item: id(3),
                                    extents: MAX_EXTENTS + 1,
                                }),
                                "{what}"
                            );
                            if version != 0 {
                                // Construction method 1 counts
                                // against the idat; method 2 only
                                // against the cap.
                                let idat = |method: u8, extents: &[(u64, u64)]| {
                                    concat(&[
                                        ftyp(b"mif1miaf"),
                                        meta_with(&[
                                            bx(b"idat", &[0u8; 40]),
                                            iloc_with(
                                                version,
                                                sizes,
                                                &[(id(4), method, 0, extents)],
                                            ),
                                        ]),
                                        bx(b"mdat", &[0u8; 100]),
                                    ])
                                };
                                assert_eq!(item_refusal(&idat(1, &[(0, 20), (20, 20)])), None);
                                assert_eq!(
                                    item_refusal(&idat(1, &[(0, 30), (0, 30)])),
                                    Some(Refusal::Amplified {
                                        item: id(4),
                                        needs: 60,
                                        holds: 40,
                                    }),
                                    "{what}"
                                );
                                assert_eq!(item_refusal(&idat(2, &[(0, 30), (0, 30)])), None);
                            }
                            combinations += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(combinations, 4 * 3 * (1 + 3 + 3));
        // One extent may leave its offset or its length out (a
        // field size of 0): an implied offset of 0 plus the base
        // offset, an implied length of 0, the rest of the data.
        for version in 0..=2u8 {
            for base_size in [0, 4, 8] {
                let one = pointing_into_mdat(100, |at| {
                    let base = if base_size > 0 { at } else { 0 };
                    let offset = at - base;
                    let (offset_size, extent) = if base_size > 0 {
                        (0, (0, 100))
                    } else {
                        (4, (offset, 100))
                    };
                    container(
                        &[iloc_with(
                            version,
                            [offset_size, 4, base_size, 0],
                            &[(1, 0, base, &[extent])],
                        )],
                        100,
                    )
                });
                assert_eq!(item_refusal(&one), None, "v{version} base {base_size}");
                let to_end = pointing_into_mdat(100, |at| {
                    container(
                        &[iloc_with(
                            version,
                            [4, 0, base_size, 0],
                            &[(1, 0, 0, &[(at, 0)])],
                        )],
                        100,
                    )
                });
                assert_eq!(item_refusal(&to_end), None, "v{version} base {base_size}");
                // A length past the data, written in a single field.
                let long = container(
                    &[iloc_with(
                        version,
                        [4, 8, base_size, 0],
                        &[(1, 0, 0, &[(0, 101)])],
                    )],
                    100,
                );
                assert_eq!(
                    item_refusal(&long),
                    Some(Refusal::Amplified {
                        item: 1,
                        needs: 101,
                        holds: 100,
                    }),
                    "v{version} base {base_size}"
                );
            }
        }
        // Lengths whose sum passes 64 bits saturate, and an
        // offset whose base pushes it past 64 bits is the
        // parser's to refuse.
        let huge = container(
            &[iloc_with(
                2,
                [8, 8, 0, 0],
                &[(7, 0, 0, &[(0, u64::MAX), (0, u64::MAX)])],
            )],
            100,
        );
        assert_eq!(
            item_refusal(&huge),
            Some(Refusal::Amplified {
                item: 7,
                needs: u64::MAX,
                holds: 100,
            })
        );
        let wrapped = container(
            &[iloc_with(2, [8, 0, 8, 0], &[(7, 0, u64::MAX, &[(1, 0)])])],
            100,
        );
        assert_eq!(item_refusal(&wrapped), None);
    }

    /// `pitm` naming `primary` (in `pitm_version`), and an
    /// `iprp` whose `ipma` (in `ipma_version`, with 2-byte
    /// associations when `wide`) gives item 2 a decoy `ispe` and
    /// the primary an `ispe` of `width` x `height` behind a
    /// codec property, marked essential.
    fn primary_ispe_variant(
        pitm_version: u8,
        ipma_version: u8,
        wide: bool,
        primary: u32,
        (width, height): (u32, u32),
    ) -> Vec<Vec<u8>> {
        let ispe = |w: u32, h: u32| {
            let mut body = w.to_be_bytes().to_vec();
            body.extend_from_slice(&h.to_be_bytes());
            bx(b"ispe", &full(0, &body))
        };
        let ipco = bx(
            b"ipco",
            &concat(&[
                ispe(30_000, 30_000),
                bx(b"av1C", &[0x81, 0x3f, 0, 0]),
                ispe(width, height),
            ]),
        );
        let id = |out: &mut Vec<u8>, id: u32| {
            if ipma_version == 0 {
                out.extend_from_slice(&(id as u16).to_be_bytes());
            } else {
                out.extend_from_slice(&id.to_be_bytes());
            }
        };
        let assoc = |out: &mut Vec<u8>, index: u16, essential: bool| {
            if wide {
                out.extend_from_slice(&((u16::from(essential) << 15) | index).to_be_bytes());
            } else {
                out.push((u8::from(essential) << 7) | index as u8);
            }
        };
        // Entries in item order, as the parser requires.
        let decoy = |out: &mut Vec<u8>| {
            id(out, 2);
            out.push(1);
            assoc(out, 1, false);
        };
        let mut ipma = 2u32.to_be_bytes().to_vec();
        if primary > 2 {
            decoy(&mut ipma);
        }
        id(&mut ipma, primary);
        ipma.push(2);
        assoc(&mut ipma, 2, true);
        assoc(&mut ipma, 3, false);
        if primary < 2 {
            decoy(&mut ipma);
        }
        let mut ipma_box = vec![ipma_version, 0, 0, u8::from(wide)];
        ipma_box.extend_from_slice(&ipma);
        let pitm = if pitm_version == 0 {
            full(0, &(primary as u16).to_be_bytes())
        } else {
            full(1, &primary.to_be_bytes())
        };
        vec![
            bx(b"pitm", &pitm),
            bx(b"iprp", &concat(&[ipco, bx(b"ipma", &ipma_box)])),
        ]
    }

    #[test]
    fn the_primary_ispe_is_found_through_every_id_and_index_width() {
        for (pitm_version, ipma_version, primary) in [(0, 0, 1), (0, 1, 1), (1, 1, 0x0001_0001)] {
            for wide in [false, true] {
                let what = format!("pitm v{pitm_version} ipma v{ipma_version} wide {wide}");
                let small = container(
                    &primary_ispe_variant(pitm_version, ipma_version, wide, primary, (8, 6)),
                    16,
                );
                assert_eq!(
                    inspect(&small).unwrap().primary_size,
                    Some((8, 6)),
                    "{what}"
                );
                assert_eq!(declared_size(&small), Some((8, 6)), "{what}");
                assert_eq!(screen(&small).map(|_| ()), Ok(()), "{what}");
                let huge = container(
                    &primary_ispe_variant(
                        pitm_version,
                        ipma_version,
                        wide,
                        primary,
                        (MAX_DECODE_SIDE + 1, 1),
                    ),
                    16,
                );
                assert!(screen(&huge).is_err(), "{what}");
            }
        }
        // Past the byte budget at four bytes a pixel, though each
        // side is within the cap.
        let side = MAX_DECODE_SIDE;
        assert!(u64::from(side) * u64::from(side) * 4 > crate::engine::MAX_DECODE_ALLOC);
        let wide_and_tall = container(&primary_ispe_variant(0, 0, false, 1, (side, side)), 16);
        assert!(matches!(
            inspect(&wide_and_tall)
                .map(|i| i.primary_size.map(|(w, h)| within_decode_limits(w, h))),
            Ok(Some(Err(Refusal::Dimension { .. })))
        ));
        // An item the primary is not: its ispe is not the
        // primary's size.
        let elsewhere = container(&primary_ispe_variant(0, 0, false, 5, (8, 6)), 16);
        let mut retargeted = elsewhere.clone();
        let pitm_at = retargeted
            .windows(4)
            .position(|w| w == b"pitm")
            .expect("a pitm");
        retargeted[pitm_at + 8..pitm_at + 10].copy_from_slice(&3u16.to_be_bytes());
        assert_eq!(inspect(&retargeted).unwrap().primary_size, None);
    }

    #[test]
    fn the_fuzz_seeds_are_refused_for_their_shapes() {
        // The decode corpus seeds one container per item-table
        // refusal, so the fuzzer starts from each shape.
        let amplified = include_bytes!("../fuzz/corpus/decode/avif-iloc-extent-amplified");
        assert!(matches!(
            item_refusal(amplified),
            Some(Refusal::Amplified { item: 1, .. })
        ));
        let to_end = include_bytes!("../fuzz/corpus/decode/avif-iloc-extent-to-end-twice");
        assert_eq!(
            item_refusal(to_end),
            Some(Refusal::Amplified {
                item: 1,
                needs: 200,
                holds: 100,
            })
        );
        let capped = include_bytes!("../fuzz/corpus/decode/avif-iloc-extent-cap");
        assert!(matches!(
            item_refusal(capped),
            Some(Refusal::Extents { item: 1, .. })
        ));
        let ispe = include_bytes!("../fuzz/corpus/decode/avif-ispe-too-large");
        assert_eq!(declared_size(ispe), Some((20_000, 20_000)));
        assert!(screen(ispe).is_err());
    }

    #[test]
    fn the_extent_cap_applies() {
        let under: Vec<(u32, u32)> = (0..MAX_EXTENTS).map(|i| (i as u32, 1)).collect();
        let ok = container(&[iloc_v1(&[(1, 0, &under)])], MAX_EXTENTS as usize);
        assert_eq!(item_refusal(&ok), None);
        let over: Vec<(u32, u32)> = (0..=MAX_EXTENTS).map(|i| (i as u32, 1)).collect();
        let bytes = container(&[iloc_v1(&[(1, 0, &over)])], MAX_EXTENTS as usize + 1);
        assert_eq!(
            item_refusal(&bytes),
            Some(Refusal::Extents {
                item: 1,
                extents: MAX_EXTENTS + 1,
            })
        );
    }

    #[test]
    fn an_ispe_past_the_limits_is_refused() {
        let within = container(&primary_with_ispe(8, 6), 16);
        assert_eq!(inspect(&within).unwrap().primary_size, Some((8, 6)));
        assert_eq!(screen(&within).map(|_| ()), Ok(()));
        let huge = 20_000;
        let bytes = container(&primary_with_ispe(huge, huge), 16);
        assert_eq!(inspect(&bytes).unwrap().primary_size, Some((huge, huge)));
        let Err(EngineError::Decode(message)) = screen(&bytes) else {
            panic!("the oversized ispe was not refused");
        };
        assert!(message.contains("past the decode limits"), "{message}");
        // The declared size is still reported for the probe, so
        // explain can state it.
        assert_eq!(declared_size(&bytes), Some((huge, huge)));
    }

    #[test]
    fn a_missing_ispe_behaves_as_before() {
        // No properties at all: nothing to size from, and the
        // screen passes as it did before the ispe check.
        let bytes = container(&[iloc_v1(&[(1, 0, &[(0, 16)])])], 16);
        assert_eq!(inspect(&bytes).unwrap().primary_size, None);
        assert_eq!(screen(&bytes).map(|_| ()), Ok(()));
        assert_eq!(declared_size(&bytes), None);
    }

    #[test]
    fn sequence_brands_are_hidden_from_the_parser() {
        let movie = bx(b"moov", &bx(b"pssh", &full(0, &[0; 20])));
        let bytes = concat(&[ftyp(b"mif1msf1miafavis"), meta(&[hdlr()]), movie]);
        let Cow::Owned(screened) = screen(&bytes).unwrap() else {
            panic!("the sequence brands were left in place");
        };
        let mut expected = bytes.clone();
        expected[20..24].copy_from_slice(&BLANK_BRAND);
        expected[28..32].copy_from_slice(&BLANK_BRAND);
        assert_eq!(screened, expected);
        // The major brand and the other brands stay; a file with
        // no sequence brand comes back borrowed.
        let still = concat(&[ftyp(b"mif1miaf"), meta(&[hdlr()])]);
        assert!(matches!(screen(&still).unwrap(), Cow::Borrowed(_)));
        // A brand cut short is no brand: the parser refuses the
        // ftyp itself.
        let ragged = concat(&[ftyp(b"mif1msf"), meta(&[hdlr()])]);
        assert!(matches!(screen(&ragged).unwrap(), Cow::Borrowed(_)));
    }

    #[test]
    fn other_formats_pass_through_untouched() {
        // HEIC shares the container and has its own decoder;
        // video containers are not decoded as images at all.
        let mut heic = concat(&[ftyp(b"mif1heic"), claims(b"meta", 0x7FFF_0000, b"")]);
        heic[8..12].copy_from_slice(b"heic");
        let mp4 = {
            let mut bytes = heic.clone();
            bytes[8..12].copy_from_slice(b"isom");
            bytes
        };
        for bytes in [heic, mp4, b"\x89PNG\r\n\x1a\n".to_vec(), Vec::new()] {
            assert!(matches!(screen(&bytes).unwrap(), Cow::Borrowed(b) if b == bytes.as_slice()));
        }
        // The real HEIF files the HEIC tests decode, too.
        let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let mut heif = 0;
        for entry in std::fs::read_dir(fixtures).unwrap() {
            let path = entry.unwrap().path();
            if !matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("heic" | "heif")
            ) {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            let screened = screen(&bytes).unwrap();
            assert!(
                matches!(screened, Cow::Borrowed(b) if b.as_ptr() == bytes.as_ptr()),
                "{}",
                path.display()
            );
            heif += 1;
        }
        assert!(heif >= 5, "the HEIF fixtures moved: found {heif}");
    }

    #[test]
    fn the_walk_is_total() {
        let seeds = [FUZZER_INPUT.to_vec(), well_formed()];
        for seed in &seeds {
            for end in 0..=seed.len() {
                let _ = screen(&seed[..end]);
            }
            let mut bytes = seed.clone();
            for i in 0..bytes.len() {
                for replacement in [0x00, 0x01, 0x07, 0x10, 0x7F, 0xFF] {
                    let original = bytes[i];
                    bytes[i] = replacement;
                    let _ = screen(&bytes);
                    bytes[i] = original;
                }
            }
        }
    }

    #[test]
    fn the_explain_probe_screens_avif_sources() {
        let probe = crate::engine::probe_source(FUZZER_INPUT);
        assert_eq!((probe.kind, probe.dimensions), ("avif", None));
    }

    #[cfg(feature = "avif-decode")]
    mod decoding {
        use super::*;
        use image::ImageEncoder;

        fn encoded() -> Vec<u8> {
            let img = image::RgbImage::from_fn(24, 16, |x, y| {
                image::Rgb([(x * 10) as u8, (y * 15) as u8, ((x + y) * 6) as u8])
            });
            let mut bytes = Vec::new();
            image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut bytes, 10, 90)
                .write_image(img.as_raw(), 24, 16, image::ExtendedColorType::Rgb8)
                .unwrap();
            bytes
        }

        #[test]
        fn encoder_output_passes_borrowed() {
            let bytes = encoded();
            assert_eq!(refusal(&bytes), None);
            assert!(matches!(screen(&bytes).unwrap(), Cow::Borrowed(_)));
        }

        #[test]
        fn the_explain_probe_reads_a_screened_container() {
            let bytes = encoded();
            let probe = crate::engine::probe_source(&bytes);
            assert_eq!((probe.kind, probe.dimensions), ("avif", Some((24, 16))));
            // A sequence brand in front of a movie mp4parse
            // cannot parse: the probe reads the still all the
            // same, because the brand is hidden from the parser.
            let mut sequence = bytes.clone();
            sequence[20..24].copy_from_slice(b"msf1");
            sequence.extend(bx(b"moov", &bx(b"trak", &claims(b"tkhd", 0x100, b""))));
            assert!(
                image::ImageReader::new(std::io::Cursor::new(&sequence))
                    .with_guessed_format()
                    .unwrap()
                    .into_dimensions()
                    .is_err()
            );
            let probe = crate::engine::probe_source(&sequence);
            assert_eq!(probe.dimensions, Some((24, 16)));
        }

        #[test]
        fn a_refused_container_is_a_decode_error() {
            let Err(EngineError::Decode(message)) = crate::engine::decode_image(FUZZER_INPUT)
            else {
                panic!("the fuzzer input was not a decode error");
            };
            assert!(
                message.starts_with("AVIF container refused: box 'hdlr'"),
                "{message}"
            );
            assert!(crate::engine::validate_source_image(FUZZER_INPUT).is_err());
        }
    }
}
