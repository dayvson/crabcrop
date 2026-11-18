//! The trim survey of an animated render (issue #465) runs
//! the steps before each trim on every frame, before any
//! frame renders. Until issue #737 it ran them on a copy of
//! each frame when anything came before the trim. Here, under
//! a recording allocator and from outside the crate, a trim
//! after a resize surveys an animated GIF without copying its
//! frames.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use crabcrop::engine::Engine;
use crabcrop::url::{ImageFormat, parse_request};

/// The global allocator of this test binary: the system
/// allocator, plus a running total of the bytes requested on
/// a thread that armed it.
struct Recorder;

thread_local! {
    /// `None` while disarmed, else the bytes requested so far.
    /// A const-initialized Cell needs no lazy setup and no
    /// destructor, so the allocator can touch it.
    static REQUESTED: Cell<Option<usize>> = const { Cell::new(None) };
}

impl Recorder {
    fn record(size: usize) {
        let _ = REQUESTED.try_with(|total| {
            if let Some(seen) = total.get() {
                total.set(Some(seen.saturating_add(size)));
            }
        });
    }
}

// SAFETY: every method forwards to `System` unchanged.
unsafe impl GlobalAlloc for Recorder {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Recorder = Recorder;

/// Run `f` with the recorder armed on this thread; return its
/// result and the bytes it requested in total.
fn requested<T>(f: impl FnOnce() -> T) -> (T, usize) {
    REQUESTED.with(|total| total.set(Some(0)));
    let out = f();
    let total = REQUESTED.with(|total| total.replace(None)).unwrap_or(0);
    (out, total)
}

/// An animated GIF of `count` opaque `side` x `side` frames,
/// each a different pattern of 64 colors that reaches every
/// edge, so a trim keeps the whole frame.
fn animation(count: u32, side: u32) -> Vec<u8> {
    let delay = image::Delay::from_numer_denom_ms(100, 1);
    let frames: Vec<image::Frame> = (0..count)
        .map(|i| {
            let buffer = image::RgbaImage::from_fn(side, side, |x, y| {
                let cell = (x / 9 + y / 7 + i * 5) % 64;
                image::Rgba([
                    (cell * 4) as u8,
                    (255 - cell * 3) as u8,
                    (cell * 7) as u8,
                    255,
                ])
            });
            image::Frame::from_parts(buffer, 0, 0, delay)
        })
        .collect();
    let mut out = Vec::new();
    image::codecs::gif::GifEncoder::new(&mut out)
        .encode_frames(frames)
        .unwrap();
    out
}

#[test]
fn a_trim_after_a_resize_surveys_the_frames_without_copying_them() {
    // Three 600x600 frames, 1.44 MB each as RGBA, resized to
    // 30x30 and then trimmed. The renders with and without the
    // trim decode, resize and encode the same frames, so what
    // the trim adds is its survey, which resizes each frame
    // once more, and its cut of each 30x30 frame. Copying the
    // frames for the survey cost a frame's bytes per frame;
    // lent, the survey takes the resize's own buffers, which
    // follow the output and the frame's width, not its area,
    // and the whole trim costs less than one frame (about a
    // third of one, measured).
    let (count, side) = (3, 600);
    let gif = animation(count, side);
    let frame_bytes = (side * side * 4) as usize;
    let render = |remainder: &str| {
        let path = format!("/unsafe/{remainder}/example.com/a.gif");
        let (_, _, request) = parse_request(&path).unwrap();
        let (out, bytes) = requested(|| Engine.process(&gif, &request.operations).unwrap());
        assert_eq!((out.width, out.height), (30, 30), "{remainder}");
        assert_eq!(out.format, ImageFormat::Gif, "{remainder}");
        let decoder = image::codecs::gif::GifDecoder::new(std::io::Cursor::new(&out.bytes));
        let frames = image::AnimationDecoder::into_frames(decoder.unwrap()).count();
        assert_eq!(
            frames, count as usize,
            "{remainder}: an animation in, an animation out"
        );
        bytes
    };
    let resized = render("30x30");
    let trimmed = render("30x30/filters:trim(10)");
    let trim_cost = trimmed.saturating_sub(resized);
    assert!(
        trim_cost < frame_bytes,
        "the trim requested {trim_cost} bytes over the resize's {resized}, \
         for {count} frames of {frame_bytes} bytes"
    );
}
