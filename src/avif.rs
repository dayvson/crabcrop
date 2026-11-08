//! AVIF container screening (issue #522).
//!
//! An AVIF source reaches mp4parse, the ISO-BMFF parser behind
//! the image crate's AVIF decoder, before any decode limit
//! applies: the dimension probe and the decode both start by
//! parsing the container. mp4parse 0.17 sizes several reads by
//! what the file declares instead of by the bytes it holds. A
//! box it reads into memory (`hdlr`, `iloc`, `idat`, the item
//! properties, `mdat`) reserves its declared size first, and two
//! 32-bit entry counts (`iinf`, and `iloc` from version 2)
//! reserve a slot per declared entry. A 291-byte file found by
//! the decode fuzzer declared a 967 MiB `hdlr`, and the parser
//! reserved all of it.
//!
//! [`screen`] runs first, on every path that hands AVIF bytes to
//! the image crate: [`crate::engine::decode_image`] (which the
//! render, upload, watermark, and metadata paths share) and the
//! explain probe. It walks the boxes mp4parse reads, in one pass,
//! and allocates nothing:
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
//!
//! Every reservation the parser sizes from a declared box size or
//! a 32-bit count is then bounded by the size of the file; the
//! 16-bit counts stay as they are (about 5 MiB at most, for
//! 65,535 `iloc` items). A refused source fails as any
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
//! The walk bounds reservations, not copies: an item whose
//! `iloc` lists the same bytes as many extents is still copied
//! once per extent. That is a separate hazard, outside this
//! screen.
//!
//! The walk follows mp4parse's own rules where they are lenient,
//! in the default strictness the image crate uses, so it refuses
//! nothing else the parser reads without such a reservation: a
//! top-level box the parser only skips (anything after the first
//! box but `meta` and `mdat`) may run past the end of the file,
//! where the parser stops reading, and a run of boxes may end in
//! fewer bytes than a box header.
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
