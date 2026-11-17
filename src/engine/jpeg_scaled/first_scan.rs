//! zune-jpeg's reading of the first scan of a progressive
//! file (issue #713), so that the scaled decode leaves a file
//! zune-jpeg refuses there to the full-size path.
//!
//! zune-jpeg decodes a progressive file's first scan and then
//! requires its bit reader to have come to the marker that
//! ends the scan, and refuses the file when it has not
//! ("Marker missing where expected"); after every later scan
//! it looks for the next marker instead. The reader holds at
//! most 63 bits, and fills itself, four bytes at a time, only
//! when it holds fewer than 32. So a first scan whose data
//! the decode is done with a few bytes before its marker is
//! refused. libjpeg-turbo, whose reader looks further ahead,
//! drops such bytes without a warning. A damaged first scan
//! can decode in fewer bits than it holds, and such a file
//! rendered at the sizes the scaled decode serves and failed
//! at full size: the one way the two decoders parted in a
//! differential run over 42,000 mutated files (issue #653),
//! 9 files, all progressive.
//!
//! [`reaches_its_marker`] reads the first scan as that reader
//! does, without decoding a pixel. The first scan of a
//! progressive file is the first pass over the DC
//! coefficients, a Huffman code and at most 15 more bits for
//! each block, so the reading is one table lookup a block,
//! and it ends at the end of the scan's data at the latest.
//! It follows zune-jpeg 0.5 (`bitstream.rs` and `mcu_prog.rs`);
//! a test holds it against zune-jpeg's own decode.

/// The bits a code is first looked up by, as zune-jpeg looks
/// it up (its `HUFF_LOOKAHEAD`); a longer code is resolved
/// from 16 bits.
const LOOKAHEAD: u32 = 9;

/// Whether zune-jpeg's bit reader comes to a marker while it
/// decodes the first scan of the progressive file `bytes`:
/// where it does not, zune-jpeg refuses the file. `false` too
/// for whatever this reading does not follow (see
/// [`marker_in_first_scan`]), which the full-size path then
/// answers.
pub(super) fn reaches_its_marker(bytes: &[u8]) -> bool {
    marker_in_first_scan(bytes) == Some(true)
}

/// zune-jpeg's reading of the first scan of `bytes`:
/// `Some(true)` where its reader comes to a marker in the
/// scan, `Some(false)` where it decodes every block of the
/// scan without coming to one, the file zune-jpeg refuses.
/// `None` for what this reading does not follow: a first
/// scan that is not the first pass over the DC coefficients,
/// restart intervals, a component or a table the scan names
/// and the header does not define, a table either decoder
/// refuses, a code not in its table, and a scan that runs
/// into the end of the file.
pub(super) fn marker_in_first_scan(bytes: &[u8]) -> Option<bool> {
    FirstScan::of(bytes)?.read(bytes)
}

/// One of the three components of the frame: its identifier
/// and its sampling factors.
#[derive(Debug, Clone, Copy, Default)]
struct Component {
    id: u8,
    h: u32,
    v: u32,
}

/// What reading the first scan takes from the header: the
/// frame's size and components, the DC table each component
/// of the scan is coded with, and where its data starts.
#[derive(Debug)]
struct FirstScan<'a> {
    size: (u32, u32),
    frame: [Component; 3],
    /// The scan's components, in its order, each with the DC
    /// table it is coded with: one component, or two or three
    /// interleaved.
    scan: Vec<(Component, DcTable<'a>)>,
    /// The offset of the scan's first byte of data.
    data: usize,
}

impl<'a> FirstScan<'a> {
    /// The first scan of `bytes`, from a walk over the marker
    /// segments up to it, as [`super::glance`] walks them; or
    /// `None` for a header this reading does not follow.
    fn of(bytes: &'a [u8]) -> Option<FirstScan<'a>> {
        if bytes.get(..2)? != [0xFF, 0xD8] {
            return None;
        }
        // The offset of each DC table slot's sixteen code
        // counts: a slot defined again is the later table.
        let mut tables: [Option<usize>; 4] = [None; 4];
        let mut frame = None;
        let mut at = 2;
        loop {
            if *bytes.get(at)? != 0xFF {
                return None;
            }
            while *bytes.get(at + 1)? == 0xFF {
                at += 1;
            }
            let code = bytes[at + 1];
            let length = usize::from(u16::from_be_bytes([
                *bytes.get(at + 2)?,
                *bytes.get(at + 3)?,
            ]));
            let start = at + 4;
            let segment = bytes.get(start..(at + 2).checked_add(length)?)?;
            match code {
                0xC2 => {
                    let &[8, h0, h1, w0, w1, 3, ref components @ ..] = segment else {
                        return None;
                    };
                    if components.len() < 9 || frame.is_some() {
                        return None;
                    }
                    let mut parsed = [Component::default(); 3];
                    for (slot, spec) in parsed.iter_mut().zip(components.as_chunks::<3>().0) {
                        let (h, v) = (u32::from(spec[1] >> 4), u32::from(spec[1] & 15));
                        if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
                            return None;
                        }
                        *slot = Component { id: spec[0], h, v };
                    }
                    let ids = parsed.map(|c| c.id);
                    if ids[0] == ids[1] || ids[0] == ids[2] || ids[1] == ids[2] {
                        return None;
                    }
                    let width = u32::from(u16::from_be_bytes([w0, w1]));
                    let height = u32::from(u16::from_be_bytes([h0, h1]));
                    frame = Some(((width, height), parsed));
                }
                0xC4 => {
                    let mut rest = start;
                    while rest < start + segment.len() {
                        let id = bytes[rest];
                        let counts = bytes.get(rest + 1..rest + 17)?;
                        let symbols: usize = counts.iter().map(|n| usize::from(*n)).sum();
                        if id >> 4 == 0 {
                            *tables.get_mut(usize::from(id & 15))? = Some(rest + 1);
                        }
                        rest += 17 + symbols;
                    }
                    if rest != start + segment.len() {
                        return None;
                    }
                }
                // Restart intervals: a reader of its own between
                // them, which this reading does not follow.
                0xDD => {
                    if segment.get(..2)? != [0, 0] {
                        return None;
                    }
                }
                0xDA => {
                    let ((width, height), frame) = frame?;
                    let (&count, selectors) = segment.split_first()?;
                    let count = usize::from(count);
                    let (selectors, band) = selectors.split_at_checked(2 * count)?;
                    // The first pass over the DC coefficients:
                    // no AC band, no refinement.
                    if !(1..=3).contains(&count) || band.get(..3)? != [0, 0, band[2] & 15] {
                        return None;
                    }
                    let mut scan = Vec::with_capacity(count);
                    for &[id, slots] in selectors.as_chunks::<2>().0 {
                        let component = *frame.iter().find(|c| c.id == id)?;
                        if scan.iter().any(|(c, _): &(Component, _)| c.id == id) {
                            return None;
                        }
                        let table = (*tables.get(usize::from(slots >> 4))?)?;
                        scan.push((component, DcTable::read(bytes, table)?));
                    }
                    return Some(FirstScan {
                        size: (width, height),
                        frame,
                        scan,
                        data: at + 2 + length,
                    });
                }
                // Markers that stand alone, and every frame
                // header but the progressive one.
                0x00 | 0x01 | 0xD0..=0xD9 | 0xC0 | 0xC1 | 0xC3 | 0xC5..=0xCF => return None,
                _ => {}
            }
            at += 2 + length;
        }
    }

    /// Read the scan as zune-jpeg does, block by block in its
    /// order, to the first marker its reader comes to or the
    /// last block (see [`marker_in_first_scan`]). zune-jpeg's
    /// MCU grid: the frame's largest sampling factors make an
    /// MCU, and a scan of one component walks that
    /// component's own blocks instead, rounded up from the
    /// part of the frame it covers.
    fn read(&self, bytes: &[u8]) -> Option<bool> {
        let (width, height) = (u64::from(self.size.0), u64::from(self.size.1));
        let h_max = u64::from(self.frame.iter().map(|c| c.h).max().unwrap_or(1));
        let v_max = u64::from(self.frame.iter().map(|c| c.v).max().unwrap_or(1));
        let mut reader = Reader {
            bytes,
            at: self.data,
            buffer: 0,
            bits: 0,
        };
        if let [(component, table)] = self.scan.as_slice() {
            let across = (width * u64::from(component.h)).div_ceil(h_max * 8);
            let down = (height * u64::from(component.v)).div_ceil(v_max * 8);
            for _ in 0..across * down {
                if reader.block(table)? {
                    return Some(true);
                }
            }
        } else {
            let mcus = width.div_ceil(h_max * 8) * height.div_ceil(v_max * 8);
            for _ in 0..mcus {
                for (component, table) in &self.scan {
                    for _ in 0..component.h * component.v {
                        if reader.block(table)? {
                            return Some(true);
                        }
                    }
                }
            }
        }
        // Every block is decoded, and the reader has not come
        // to the marker: the scan zune-jpeg refuses.
        Some(false)
    }
}

/// A DC Huffman table derived as zune-jpeg derives it, for
/// the lengths of its codes and the sizes of the differences
/// they stand for.
#[derive(Debug)]
struct DcTable<'a> {
    /// For each run of [`LOOKAHEAD`] bits a code of that many
    /// bits or fewer starts: the code's length above its
    /// symbol, a byte each. 0 where no such code starts.
    lookup: [u16; 1 << LOOKAHEAD],
    /// For each length, one more than its last code, shifted
    /// to 16 bits; -1 for a length with no code.
    maxcode: [i32; 17],
    /// For each length, the index of its first symbol less its
    /// first code.
    offset: [i32; 17],
    symbols: &'a [u8],
}

impl<'a> DcTable<'a> {
    /// The table whose sixteen code counts are at `at`, or
    /// `None` for one this reading does not take: its codes
    /// overflow their lengths, which both decoders refuse, or
    /// a difference of more than 15 bits, which libjpeg-turbo
    /// refuses in a DC table.
    fn read(bytes: &'a [u8], at: usize) -> Option<DcTable<'a>> {
        let counts = bytes.get(at..at + 16)?;
        let total: usize = counts.iter().map(|n| usize::from(*n)).sum();
        let symbols = bytes.get(at + 16..at + 16 + total)?;
        if total > 256 || symbols.iter().any(|s| *s > 15) {
            return None;
        }
        let mut table = DcTable {
            lookup: [0; 1 << LOOKAHEAD],
            maxcode: [-1; 17],
            offset: [0; 17],
            symbols,
        };
        let mut code = 0u32;
        let mut index = 0usize;
        for (length, count) in (1..=16u32).zip(counts) {
            let count = u32::from(*count);
            // No code may be all ones, and none may overflow
            // its length. Checked before the codes are laid
            // into the lookup, which they would overrun.
            if code + count >= 1 << length {
                return None;
            }
            if count > 0 {
                table.offset[length as usize] = index as i32 - code as i32;
                if length <= LOOKAHEAD {
                    for k in 0..count {
                        let first = ((code + k) << (LOOKAHEAD - length)) as usize;
                        let entry = (length as u16) << 8 | u16::from(symbols[index + k as usize]);
                        table.lookup[first..first + (1 << (LOOKAHEAD - length))].fill(entry);
                    }
                }
                code += count;
                index += count as usize;
                table.maxcode[length as usize] = (code << (16 - length)) as i32;
            }
            code <<= 1;
        }
        Some(table)
    }

    /// The length of the code `peek` (the next 16 bits) starts
    /// with, and the size of the difference it stands for, or
    /// `None` for bits no code starts.
    fn decode(&self, peek: u32) -> Option<(u32, u32)> {
        let entry = self.lookup[(peek >> (16 - LOOKAHEAD)) as usize];
        if entry != 0 {
            return Some((u32::from(entry >> 8), u32::from(entry & 0xFF)));
        }
        let length = (LOOKAHEAD + 1..=16).find(|&l| (peek as i32) < self.maxcode[l as usize])?;
        let index = (peek >> (16 - length)) as i32 + self.offset[length as usize];
        let symbol = *self.symbols.get(usize::try_from(index).ok()?)?;
        Some((length, u32::from(symbol)))
    }
}

/// zune-jpeg's bit reader, as its decode of a first scan
/// moves it.
struct Reader<'a> {
    bytes: &'a [u8],
    /// The offset of the next byte it reads.
    at: usize,
    /// The bits it holds, the last read in the lowest.
    buffer: u64,
    bits: u32,
}

impl Reader<'_> {
    /// One block's DC difference: `Some(true)` when the reader
    /// comes to a marker, `Some(false)` when it reads on, and
    /// `None` for a code not in the table or the end of the
    /// file. zune-jpeg fills the reader before each block that
    /// finds it holding fewer than 32 bits, which a block's
    /// code and difference (at most 16 and 15 bits) never
    /// outrun.
    fn block(&mut self, table: &DcTable<'_>) -> Option<bool> {
        if self.bits < 32 && self.fill()? {
            return Some(true);
        }
        let peek = ((self.buffer << (64 - self.bits)) >> 48) as u32;
        let (length, size) = table.decode(peek)?;
        self.bits -= length + size;
        Some(false)
    }

    /// zune-jpeg's refill: four bytes at once when the next
    /// four hold no 0xFF, otherwise four bytes one at a time,
    /// each 0xFF followed by a zero (and by any 0xFF fill
    /// before it) being a byte of data, and any other 0xFF
    /// the marker, where the reader stops. `Some(true)` at the
    /// marker; `None` at the end of the file, where zune-jpeg
    /// stops filling and never comes to one.
    fn fill(&mut self) -> Option<bool> {
        if let Some(four) = self.bytes.get(self.at..self.at + 4)
            && !four.contains(&0xFF)
        {
            let four: [u8; 4] = four.try_into().ok()?;
            self.buffer = self.buffer << 32 | u64::from(u32::from_be_bytes(four));
            self.bits += 32;
            self.at += 4;
            return Some(false);
        }
        for _ in 0..4 {
            let byte = *self.bytes.get(self.at)?;
            self.at += 1;
            if byte == 0xFF {
                let mut next = *self.bytes.get(self.at)?;
                self.at += 1;
                while next == 0xFF {
                    next = *self.bytes.get(self.at)?;
                    self.at += 1;
                }
                if next != 0x00 {
                    return Some(true);
                }
            } else if self.at == self.bytes.len() {
                // The file's last byte: zune-jpeg counts it as
                // read past the end, and fills no more.
                return None;
            }
            self.buffer = self.buffer << 8 | u64::from(byte);
            self.bits += 8;
        }
        Some(false)
    }
}
