//! AVIF container screening (issue #522), from outside the
//! crate and under a recording allocator. mp4parse, the
//! container parser behind the image crate's AVIF decoder,
//! reserves what a file declares (a box's size, an entry
//! count) before it reads anything, so a 291-byte file the
//! decode fuzzer found made it reserve 967 MiB ahead of every
//! decode limit. The server now walks an AVIF's boxes first.
//!
//! These tests watch every allocation the decode entry points
//! make for hostile containers and assert that none is large.
//! With the `avif-decode` feature (every shipped build, and
//! CI's second test run) they also hand the same bytes to the
//! image crate directly and assert that it does attempt the
//! large reservation, so each input is known to reach the
//! hazard the screen keeps it from.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use crabcrop::engine::{decode_image, validate_source_image};

/// The global allocator of this test binary: the system
/// allocator, plus a record of the largest single request
/// made on a thread that armed it. While armed, a request
/// past [`REFUSED_ABOVE`] is recorded and refused (a null
/// return): fallible reservations, which is how mp4parse
/// makes them, then fail cleanly, so a regression shows up
/// as a recorded size rather than as gigabytes of reserved
/// address space.
struct Recorder;

/// Requests past this size are refused while armed.
const REFUSED_ABOVE: usize = 64 << 20;

/// Nothing a well-formed or refused input of these sizes
/// needs comes near this; every reservation the hostile
/// inputs provoke without the screen is far above it.
const LARGE: usize = 1 << 20;

thread_local! {
    /// `None` while disarmed, else the largest request seen.
    /// A const-initialized Cell needs no lazy setup and no
    /// destructor, so the allocator can touch it.
    static LARGEST: Cell<Option<usize>> = const { Cell::new(None) };
}

impl Recorder {
    /// Record one request; false refuses it.
    fn admit(size: usize) -> bool {
        LARGEST
            .try_with(|largest| match largest.get() {
                Some(seen) => {
                    largest.set(Some(seen.max(size)));
                    size <= REFUSED_ABOVE
                }
                None => true,
            })
            .unwrap_or(true)
    }
}

// SAFETY: every method forwards to `System` unchanged, or
// returns null (which `GlobalAlloc` permits for any request).
unsafe impl GlobalAlloc for Recorder {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !Self::admit(layout.size()) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !Self::admit(layout.size()) {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if !Self::admit(new_size) {
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Recorder = Recorder;

/// Run `f` with the recorder armed on this thread; return
/// its result and the largest single request it made.
fn largest_allocation<T>(f: impl FnOnce() -> T) -> (T, usize) {
    LARGEST.with(|largest| largest.set(Some(0)));
    let out = f();
    let largest = LARGEST.with(|largest| largest.replace(None)).unwrap_or(0);
    (out, largest)
}

/// The input from the issue, kept as the decode fuzzer's
/// regression seed: a `meta` whose `hdlr` declares
/// 1,014,169,633 bytes inside a 291-byte file.
const FUZZER_INPUT: &[u8] = include_bytes!("../fuzz/corpus/decode/avif-hdlr-overrun");

fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    claims(kind, (8 + body.len()) as u32, body)
}

/// A box whose 32-bit size field says `declared`, whatever
/// its body holds.
fn claims(kind: &[u8; 4], declared: u32, body: &[u8]) -> Vec<u8> {
    let mut out = declared.to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

/// A box with a 64-bit size (`size` 1, then `largesize`).
fn claims_large(kind: &[u8; 4], declared: u64, body: &[u8]) -> Vec<u8> {
    let mut out = 1u32.to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(&declared.to_be_bytes());
    out.extend_from_slice(body);
    out
}

/// A full box body: version, zero flags, then `rest`.
fn full(version: u8, rest: &[u8]) -> Vec<u8> {
    let mut out = vec![version, 0, 0, 0];
    out.extend_from_slice(rest);
    out
}

/// `ftyp` + `meta` (a `pict` handler, then `children`) +
/// `tail`: just enough container for mp4parse to read
/// every child it is handed.
fn avif(children: &[Vec<u8>], tail: &[u8]) -> Vec<u8> {
    let mut handler = vec![0; 4];
    handler.extend_from_slice(b"pict");
    handler.extend_from_slice(&[0; 13]);
    let mut meta = full(0, &[]);
    meta.extend(bx(b"hdlr", &full(0, &handler)));
    for child in children {
        meta.extend_from_slice(child);
    }
    let mut out = bx(b"ftyp", b"avif\0\0\0\0mif1miaf");
    out.extend(bx(b"meta", &meta));
    out.extend_from_slice(tail);
    out
}

/// Each hostile container, named for what it declares.
fn hostile_inputs() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("the fuzzer's hdlr past its parent", FUZZER_INPUT.to_vec()),
        (
            "an idat past its parent",
            avif(&[claims(b"idat", 0x7000_0000, b"")], b""),
        ),
        (
            "a 64-bit idat past its parent",
            avif(&[claims_large(b"idat", 1 << 36, b"")], b""),
        ),
        (
            "a colr three boxes deep, past its parent",
            avif(
                &[bx(
                    b"iprp",
                    &bx(b"ipco", &claims(b"colr", 0x6000_0000, b"prof")),
                )],
                b"",
            ),
        ),
        (
            "an mdat past the end of the file",
            avif(&[], &claims(b"mdat", 0xFFFF_FFF0, b"")),
        ),
        (
            "a 64-bit mdat past the end of the file",
            avif(&[], &claims_large(b"mdat", 1 << 40, b"")),
        ),
        (
            "an iinf counting 2^28 entries, holding none",
            avif(&[bx(b"iinf", &full(1, &0x1000_0000u32.to_be_bytes()))], b""),
        ),
        (
            "an iloc counting 2^28 items, holding none",
            avif(&[bx(b"iloc", &full(2, &[0, 0, 0x10, 0, 0, 0]))], b""),
        ),
    ]
}

#[test]
fn hostile_containers_are_refused_without_a_large_reservation() {
    // Builds without AVIF decoding refuse every AVIF source by
    // name, before any parsing.
    let refusal = if cfg!(feature = "avif-decode") {
        "AVIF container refused"
    } else {
        "avif-decode"
    };
    for (what, bytes) in hostile_inputs() {
        let (decoded, largest) = largest_allocation(|| decode_image(&bytes));
        let err = decoded.err().unwrap_or_else(|| panic!("{what}: decoded"));
        assert!(err.to_string().contains(refusal), "{what}: {err}");
        assert!(
            largest < LARGE,
            "{what}: decode_image asked for {largest} bytes at once"
        );
        let (validated, largest) = largest_allocation(|| validate_source_image(&bytes));
        let err = validated
            .err()
            .unwrap_or_else(|| panic!("{what}: validated"));
        assert!(err.to_string().contains(refusal), "{what}: {err}");
        assert!(
            largest < LARGE,
            "{what}: validate_source_image asked for {largest} bytes at once"
        );
    }
}

#[cfg(feature = "avif-decode")]
#[test]
fn every_hostile_container_reaches_the_reservation_unscreened() {
    // The image crate's own decoder, handed the same bytes,
    // attempts the large reservation (refused here by the
    // recorder). If mp4parse ever bounds these reads itself,
    // this fails, and the screen can be revisited.
    for (what, bytes) in hostile_inputs() {
        let (decoded, largest) = largest_allocation(|| image::load_from_memory(&bytes));
        assert!(decoded.is_err(), "{what}: decoded unscreened");
        assert!(
            largest > REFUSED_ABOVE,
            "{what}: the unscreened decoder asked for only {largest} bytes at once"
        );
    }
}

/// A small, well-formed AVIF from the always-available
/// encoder: a gradient, so a decode that went wrong could
/// not match by accident.
#[cfg(feature = "avif-decode")]
fn encoded_avif() -> Vec<u8> {
    use image::ImageEncoder;
    let img = image::RgbImage::from_fn(24, 16, |x, y| {
        image::Rgb([(x * 10) as u8, (y * 15) as u8, ((x + y) * 6) as u8])
    });
    let mut bytes = Vec::new();
    image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut bytes, 10, 90)
        .write_image(img.as_raw(), 24, 16, image::ExtendedColorType::Rgb8)
        .unwrap();
    assert_eq!(&bytes[4..24], b"ftypavif\0\0\0\0mif1miaf");
    bytes
}

#[cfg(feature = "avif-decode")]
#[test]
fn well_formed_avif_decodes_as_the_image_crate_decodes_it() {
    let bytes = encoded_avif();
    let reference = image::load_from_memory(&bytes).unwrap();
    let (decoded, largest) = largest_allocation(|| decode_image(&bytes));
    let decoded = decoded.unwrap();
    assert_eq!((decoded.width(), decoded.height()), (24, 16));
    assert_eq!(decoded.color(), reference.color());
    assert_eq!(decoded.as_bytes(), reference.as_bytes());
    assert!(
        largest < LARGE,
        "the decode asked for {largest} bytes at once"
    );
    validate_source_image(&bytes).unwrap();

    // The same file with its trailing mdat sized 0 ("to the
    // end of the file") reads the same bytes.
    let mdat = bytes.windows(4).rposition(|w| w == b"mdat").unwrap() - 4;
    let mut to_end = bytes.clone();
    to_end[mdat..mdat + 4].copy_from_slice(&[0; 4]);
    let decoded = decode_image(&to_end).unwrap();
    assert_eq!(decoded.as_bytes(), reference.as_bytes());
}

#[cfg(feature = "avif-decode")]
#[test]
fn a_sequence_brand_does_not_reach_the_movie_parser() {
    // An image-sequence brand makes mp4parse parse the
    // movie box as well, whose readers size several reads
    // from fields: a pssh box's data_size alone reserves up
    // to 4 GiB. The image crate decodes the primary image
    // and never uses the sequence, so the screen hides the
    // brand from the parser; the still decodes as it would
    // without the movie.
    let still = encoded_avif();
    let reference = image::load_from_memory(&still).unwrap();
    let with_sequence = |movie: &[u8]| {
        let mut bytes = still.clone();
        bytes[20..24].copy_from_slice(b"msf1");
        bytes.extend(bx(b"moov", movie));
        bytes
    };

    // A movie mp4parse accepts: identical pixels either way.
    let benign = with_sequence(b"");
    let unscreened = image::load_from_memory(&benign).unwrap();
    assert_eq!(unscreened.as_bytes(), reference.as_bytes());
    assert_eq!(
        decode_image(&benign).unwrap().as_bytes(),
        reference.as_bytes()
    );

    // A pssh whose data_size claims 0xFFFF_FFF0 bytes.
    let mut pssh = full(0, &[0x11; 16]);
    pssh.extend_from_slice(&0xFFFF_FFF0u32.to_be_bytes());
    let hostile = with_sequence(&bx(b"pssh", &pssh));
    let (unscreened, largest) = largest_allocation(|| image::load_from_memory(&hostile));
    assert!(unscreened.is_err());
    assert!(largest > REFUSED_ABOVE, "unscreened: {largest} bytes");
    let (decoded, largest) = largest_allocation(|| decode_image(&hostile));
    assert_eq!(decoded.unwrap().as_bytes(), reference.as_bytes());
    assert!(
        largest < LARGE,
        "the decode asked for {largest} bytes at once"
    );
}
