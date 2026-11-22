//! What a decode holds at once, measured from outside the
//! crate under a recording allocator. EXIF orientations 5 and
//! 7 turn the decoded frame and mirror the turned frame where
//! it is, so they hold two frames at most, as 6 and 8 do,
//! where mirroring into a frame of its own held three. And the
//! libwebp walk over an animated WebP hands its canvas itself
//! to the last frame instead of a copy of it, so a one-frame
//! animation holds one canvas on the heap, where it held two
//! (the frame libwebp decodes is in libwebp's own allocation,
//! which this allocator does not see).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use crabcrop::engine::Engine;
use crabcrop::url::parse_request;

/// The global allocator of this test binary: the system
/// allocator, plus the bytes live on a thread that armed it
/// and the most that were live at once.
struct Recorder;

thread_local! {
    /// `None` while disarmed, else the bytes allocated less the
    /// bytes freed on this thread since it was armed, and the
    /// most that sum has been. A const-initialized Cell needs
    /// no lazy setup and no destructor, so the allocator can
    /// touch it.
    static LIVE: Cell<Option<(i64, i64)>> = const { Cell::new(None) };
}

impl Recorder {
    fn change(delta: i64) {
        let _ = LIVE.try_with(|live| {
            if let Some((now, most)) = live.get() {
                let now = now.saturating_add(delta);
                live.set(Some((now, most.max(now))));
            }
        });
    }
}

fn signed(size: usize) -> i64 {
    i64::try_from(size).unwrap_or(i64::MAX)
}

// SAFETY: every method forwards to `System` unchanged.
unsafe impl GlobalAlloc for Recorder {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::change(signed(layout.size()));
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::change(signed(layout.size()));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::change(signed(new_size).saturating_sub(signed(layout.size())));
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        Self::change(-signed(layout.size()));
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Recorder = Recorder;

/// Run `f` with the recorder armed on this thread; return its
/// result and the most bytes it held at once.
fn peak<T>(f: impl FnOnce() -> T) -> (T, usize) {
    LIVE.with(|live| live.set(Some((0, 0))));
    let out = f();
    let (_, most) = LIVE.with(|live| live.replace(None)).unwrap_or((0, 0));
    (out, usize::try_from(most).unwrap_or(0))
}

/// Render `bytes` (a source named `name`) through the URL
/// `remainder`, and return the most it held at once.
fn render_peak(bytes: &[u8], name: &str, remainder: &str) -> usize {
    let path = format!("/unsafe/{remainder}/example.com/{name}");
    let (_, _, request) = parse_request(&path).unwrap();
    let (out, most) = peak(|| Engine.process(bytes, &request.operations));
    out.unwrap_or_else(|e| panic!("{name} through {remainder}: {e:?}"));
    most
}

/// An opaque `width` x `height` RGB PNG of noise, with an eXIf
/// chunk after IHDR whose one entry is the EXIF orientation.
fn png_with_orientation(width: u32, height: u32, orientation: u16) -> Vec<u8> {
    let mut state: u32 = 0x2545_f491;
    let img = image::RgbImage::from_fn(width, height, |_, _| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        image::Rgb([state as u8, (state >> 8) as u8, (state >> 16) as u8])
    });
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    // A big-endian TIFF header, one IFD with one SHORT entry
    // (0x0112, the orientation), and no next IFD.
    let mut tiff = b"MM\0\x2a".to_vec();
    tiff.extend_from_slice(&8u32.to_be_bytes());
    tiff.extend_from_slice(&1u16.to_be_bytes());
    tiff.extend_from_slice(&0x0112u16.to_be_bytes());
    tiff.extend_from_slice(&3u16.to_be_bytes());
    tiff.extend_from_slice(&1u32.to_be_bytes());
    tiff.extend_from_slice(&orientation.to_be_bytes());
    tiff.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    let mut chunk = (tiff.len() as u32).to_be_bytes().to_vec();
    chunk.extend_from_slice(b"eXIf");
    chunk.extend_from_slice(&tiff);
    let mut crc = crc32fast::Hasher::new();
    crc.update(b"eXIf");
    crc.update(&tiff);
    chunk.extend_from_slice(&crc.finalize().to_be_bytes());
    // The signature and the IHDR chunk (13 bytes of data).
    let after_ihdr = 8 + 8 + 13 + 4;
    [&png[..after_ihdr], &chunk, &png[after_ihdr..]].concat()
}

#[test]
fn exif_orientations_5_and_7_hold_no_more_than_6_and_8() {
    // A 1600x1000 RGB frame, 4.8 MB: past the size where the
    // turns take the fast permutations. Every render decodes
    // the frame, turns it upright and shrinks it to 64 wide, so
    // what differs between two of them is the orientation step.
    // A turn needs a second frame; 5 and 7 then mirror, and a
    // mirror into a frame of its own held a third.
    let (width, height) = (1600, 1000);
    let frame = (width * height * 3) as usize;
    let held = |orientation| {
        let png = png_with_orientation(width, height, orientation);
        render_peak(&png, "a.png", "64x0/filters:format(png)")
    };
    let (upright, five, six, seven, eight) = (held(1), held(5), held(6), held(7), held(8));
    // Visible under `--nocapture`; captured in the gate.
    eprintln!(
        "orientation peaks for a {frame} byte frame: 1 {upright}, 5 {five}, 6 {six}, \
         7 {seven}, 8 {eight}"
    );
    // The turns hold the frame and its turned copy.
    assert!(
        six >= upright + frame / 2,
        "orientation 6 held {six} bytes, orientation 1 {upright}: the turn made no copy?"
    );
    for (mirrored, turned, values) in [(five, six, "5 and 6"), (seven, eight, "7 and 8")] {
        assert!(
            mirrored <= turned + frame / 4,
            "orientations {values} held {mirrored} and {turned} bytes, for a frame of {frame}"
        );
    }
}

/// `bytes`, an animated WebP, with its VP8X canvas declaring
/// `width` x `height`: the frames stay where they are, as
/// sub-rectangles of the larger canvas.
fn with_canvas(mut bytes: Vec<u8>, width: u32, height: u32) -> Vec<u8> {
    let vp8x = bytes
        .windows(4)
        .position(|chunk| chunk == b"VP8X")
        .expect("an animated WebP has a VP8X chunk");
    let body = vp8x + 8;
    bytes[body + 4..body + 7].copy_from_slice(&(width - 1).to_le_bytes()[..3]);
    bytes[body + 7..body + 10].copy_from_slice(&(height - 1).to_le_bytes()[..3]);
    bytes
}

#[test]
fn a_one_frame_animated_webp_holds_one_canvas() {
    // One lossless 4x4 frame on a 1024x1024 canvas (4 MiB at
    // four bytes a pixel). A one-frame animation continues as a
    // still, which the render shrinks where it is, so what it
    // holds at most is what the decode holds. The walk holds
    // its canvas; handing a copy of it to the frame held a
    // second canvas.
    let img = image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]));
    let image = oxideav_webp::WebpImage::from_rgba8(4, 4, img.into_raw()).unwrap();
    let frame = oxideav_webp::Frame::new(image, Some(std::time::Duration::from_millis(100)));
    let opts = oxideav_webp::EncodeOptions::default()
        .with_frame_mode(oxideav_webp::AnimFrameMode::Lossless);
    let one = oxideav_webp::encode_animation(&[frame], &opts).unwrap();
    let (width, height) = (1024, 1024);
    let canvas = (width * height * 4) as usize;
    let webp = with_canvas(one, width, height);
    let held = render_peak(&webp, "a.webp", "64x64/filters:format(png)");
    // Visible under `--nocapture`; captured in the gate.
    eprintln!("one-frame animated WebP peak for a {canvas} byte canvas: {held}");
    assert!(
        held < canvas + canvas / 2,
        "the render held {held} bytes for a canvas of {canvas}"
    );
}
