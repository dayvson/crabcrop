//! The walk the coefficient charge reads a JPEG's frame header
//! and first scan header with ([`frame_sampling`]), against
//! zune-jpeg's own header reader, on headers put together at
//! random from the segments, fill bytes, stray bytes and
//! lengths that reader takes.

use super::*;

/// A xorshift generator, the same on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }
}

fn segment(code: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![0xFF, code];
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// A frame header of type `code`, components 1, 2 and so on at
/// `factors`, all on quantization table 0.
fn frame(code: u8, width: u16, height: u16, factors: &[u8]) -> Vec<u8> {
    let mut payload = vec![8];
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(factors.len() as u8);
    for (id, factors) in (1..).zip(factors) {
        payload.extend_from_slice(&[id, *factors, 0]);
    }
    segment(code, &payload)
}

/// A scan header for the components `ids`.
fn scan(ids: &[u8]) -> Vec<u8> {
    let mut payload = vec![ids.len() as u8];
    for id in ids {
        payload.extend_from_slice(&[*id, 0]);
    }
    payload.extend_from_slice(&[0, 63, 0]);
    segment(0xDA, &payload)
}

/// A quantization table and the two Huffman tables the scan
/// names, then the first scan header for the components `ids`,
/// a few bytes of its data and the end of the image.
fn tables_and_first_scan(ids: &[u8]) -> Vec<u8> {
    let mut out = segment(0xDB, &[&[0u8][..], &[1u8; 64][..]].concat());
    let mut table = vec![0x00];
    let mut counts = [0u8; 16];
    counts[1] = 1;
    table.extend_from_slice(&counts);
    table.push(0);
    out.extend(segment(0xC4, &table));
    table[0] = 0x10;
    out.extend(segment(0xC4, &table));
    out.extend(scan(ids));
    out.extend_from_slice(&[0, 0, 0, 0, 0xFF, 0xD9]);
    out
}

/// A random piece of header: a segment of a kind zune-jpeg
/// reads, which may carry one of the `decoys` in its payload,
/// fill bytes, stray bytes, a stuffed zero, a length a few bytes
/// off, or a decoy on its own.
fn piece(rng: &mut Rng, decoys: &[Vec<u8>]) -> Vec<u8> {
    let carry = |prefix: &[u8], rng: &mut Rng| -> Vec<u8> {
        let mut payload = prefix.to_vec();
        for _ in 0..rng.below(3) {
            payload.push(*rng.pick(&[0u8, 0xFF, 0x12]));
        }
        if rng.below(4) != 0 {
            payload.extend_from_slice(rng.pick(decoys));
        }
        for _ in 0..rng.below(3) {
            payload.push(*rng.pick(&[0u8, 0xFF, 0xD9]));
        }
        payload
    };
    match rng.below(24) {
        0 => segment(0xE0, b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0"),
        // An APP0 segment of length 6, which zune-jpeg reads a
        // byte past.
        1 => [&[0xFF, 0xE0, 0, 6][..], &carry(b"JFIF", rng)[..4]].concat(),
        2 => segment(0xE0, &carry(b"", rng)),
        3 => segment(0xE1, &carry(b"Exif\0\0", rng)),
        4 => segment(0xE1, &carry(b"http://ns.adobe.com/xap/1.0/\0", rng)),
        5 => segment(
            0xE1,
            &carry(
                &[&b"http://ns.adobe.com/xmp/extension/\0"[..], &[7u8; 40][..]].concat(),
                rng,
            ),
        ),
        6 => segment(0xE2, &carry(b"ICC_PROFILE\0\x01\x01", rng)),
        7 => segment(0xE2, &carry(b"urn:iso:std:iso:ts:21496:-1\0", rng)),
        8 => segment(0xE2, &carry(b"MPF\0", rng)),
        9 => segment(0xED, &carry(b"Photoshop 3.0\0", rng)),
        10 => segment(0xEE, &carry(b"Adobe\0\x64\0\0\0\0\x01", rng)),
        11 => segment(0xFE, &carry(b"", rng)),
        // Codes zune-jpeg steps over by their length: other
        // application segments, the frame headers it does not
        // decode, markers that stand alone in a scan.
        12 => {
            let code = *rng.pick(&[
                0xE5u8, 0xEF, 0xF0, 0xFD, 0xC3, 0xC5, 0xC8, 0xCB, 0xCF, 0xD0, 0xD7, 0xD8, 0x01,
                0x02, 0xBF, 0xDE, 0xDF,
            ]);
            segment(code, &carry(b"", rng))
        }
        13 => segment(0xDD, &[0, 16]),
        14 => vec![0xFF],
        15 => vec![*rng.pick(&[0u8, 0x12, 0xC2, 0xD8])],
        16 => vec![0xFF, 0x00],
        17 => rng.pick(decoys).clone(),
        18 => {
            let code = *rng.pick(&[0xE1u8, 0xFE, 0xE5, 0xE0]);
            let mut off = segment(code, &carry(b"Exif\0\0", rng));
            let length = i32::from(u16::from_be_bytes([off[2], off[3]]));
            let length = (length + rng.below(7) as i32 - 3).clamp(0, 65535) as u16;
            off[2..4].copy_from_slice(&length.to_be_bytes());
            off
        }
        19 => segment(0xDB, &[&[1u8][..], &[1u8; 64][..]].concat()),
        20 => vec![0xFF, 0xFF],
        21 => [
            &[0xFF, 0xE0, 0, 6, 1, 2, 3, 4][..],
            &[*rng.pick(&[0xFFu8, 0x00, 0xC2])][..],
        ]
        .concat(),
        22 => segment(0xE0, &carry(b"AVI1\0", rng)),
        _ => segment(0xE1, &carry(b"Exif\0\0", rng)),
    }
}

#[test]
fn the_frame_walk_reads_the_headers_zune_jpeg_reads() {
    // The real frame is 2000x1500 at 4:2:0, progressive or
    // sequential, and its first scan carries one component or
    // all three. Decoy frame headers state other sizes and
    // factors, and decoy scan headers name a component no frame
    // has, so zune-jpeg refuses any file whose first scan it
    // reads from a decoy: every file it reads, it read the real
    // first scan of. Wherever the walk reads a frame, it must be
    // the frame zune-jpeg read, and wherever it reads a first
    // scan, the real one.
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut read, mut to_the_scan, mut to_the_frame, mut stopped) = (0, 0, 0, 0);
    for _ in 0..40_000 {
        let first_scan: &[u8] = if rng.below(2) == 0 { &[1] } else { &[1, 2, 3] };
        let decoys = [
            frame(
                0xC2,
                1000 + rng.below(50) as u16,
                900 + rng.below(50) as u16,
                &[*rng.pick(&[0x11u8, 0x22, 0x21]), 0x11, 0x11],
            ),
            scan(&[9]),
            scan(&[9, 1, 2]),
        ];
        let mut bytes = vec![0xFF, 0xD8];
        for _ in 0..rng.below(6) {
            bytes.extend(piece(&mut rng, &decoys));
        }
        bytes.extend(frame(
            *rng.pick(&[0xC2u8, 0xC0]),
            2000,
            1500,
            &[0x22, 0x11, 0x11],
        ));
        for _ in 0..rng.below(3) {
            bytes.extend(piece(&mut rng, &decoys));
        }
        bytes.extend(tables_and_first_scan(first_scan));
        let mut zune = zune_jpeg::JpegDecoder::new_with_options(
            zune_core::bytestream::ZCursor::new(&bytes[..]),
            zune_jpeg_options(),
        );
        if zune.decode_headers().is_err() {
            continue;
        }
        read += 1;
        let (width, height) = zune.dimensions().unwrap();
        let Some(walked) = frame_sampling(&bytes) else {
            stopped += 1;
            continue;
        };
        assert_eq!(
            (usize::from(walked.width), usize::from(walked.height)),
            (width, height),
            "the walk read another frame: {bytes:02X?}"
        );
        match walked.first_scan {
            Some(components) => {
                assert_eq!(
                    components,
                    first_scan.len(),
                    "the walk read another first scan: {bytes:02X?}"
                );
                to_the_scan += 1;
            }
            None => to_the_frame += 1,
        }
    }
    // Visible under `--nocapture`; captured in the gate.
    eprintln!(
        "zune-jpeg read {read} headers; the walk read the same frame and first scan for \
         {to_the_scan}, the same frame alone for {to_the_frame}, and stopped short of the \
         frame for {stopped}"
    );
    assert!(
        to_the_scan > read / 4,
        "too few walks reached the first scan: {to_the_scan} of {read}"
    );
}
