//! What a render holds at once when a step's output frame
//! would pass the byte budget (issue #784), measured from
//! outside the crate under a recording allocator. The issue's
//! four sources, 2x2 and resized to 10000x10000 as JPEG, hold
//! 4, 8, 12 and 16 bytes a pixel: the 8-bit one renders, and
//! each of the others either peaks within 1.2 times what the
//! 8-bit one holds or is refused before its frame is
//! allocated. So is each step that would allocate a deep
//! frame past the budget.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::Cursor;

use crabcrop::engine::{Engine, EngineError};
use crabcrop::url::parse_request;
use image::DynamicImage;

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
/// `remainder`; return the outcome and the most it held at
/// once. The output is dropped inside the measurement.
fn render(bytes: &[u8], name: &str, remainder: &str) -> (Result<(), EngineError>, usize) {
    let path = format!("/unsafe/{remainder}/example.com/{name}");
    let (_, _, request) = parse_request(&path).unwrap();
    peak(|| Engine.process(bytes, &request.operations).map(drop))
}

/// The most a refused render may hold: the decode of a 2x2
/// source and the request around it, far below one frame of
/// any output it was refused.
const REFUSED_PEAK: usize = 16 << 20;

/// A 2x2 source in `layout`, encoded as `format`.
fn source(layout: image::ColorType, format: image::ImageFormat) -> Vec<u8> {
    let rgba = image::RgbaImage::from_fn(2, 2, |x, y| {
        image::Rgba([(x * 200) as u8, (y * 200) as u8, 90, 255])
    });
    let img = DynamicImage::ImageRgba8(rgba);
    let img = match layout {
        image::ColorType::Rgba8 => img,
        image::ColorType::Rgba16 => DynamicImage::ImageRgba16(img.to_rgba16()),
        image::ColorType::Rgb32F => DynamicImage::ImageRgb32F(img.to_rgb32f()),
        image::ColorType::Rgba32F => DynamicImage::ImageRgba32F(img.to_rgba32f()),
        other => panic!("no source for {other:?}"),
    };
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), format).unwrap();
    assert_eq!(image::load_from_memory(&out).unwrap().color(), layout);
    out
}

/// Whether `outcome` is the refusal of a frame past the byte
/// budget.
fn refused_for_bytes(outcome: &Result<(), EngineError>) -> bool {
    matches!(outcome, Err(EngineError::Unsupported(why))
        if why.starts_with("requested output ") && why.ends_with("exceeds the 400000000 byte limit"))
}

#[test]
fn a_deep_source_at_the_pixel_limit_peaks_like_an_eight_bit_one_or_is_refused() {
    // The issue's measurements, on a release build's RSS:
    // 707 MB for the RGBA8 PNG, 1.10 GB for the 16-bit PNG,
    // 1.49 GB for the RGB float TIFF and 1.88 GB for the RGBA
    // float TIFF, for the same JPEG.
    let remainder = "10000x10000/filters:format(jpeg)";
    let deep = [
        (
            "rgba16.png",
            source(image::ColorType::Rgba16, image::ImageFormat::Png),
        ),
        (
            "rgb32f.tif",
            source(image::ColorType::Rgb32F, image::ImageFormat::Tiff),
        ),
        (
            "rgba32f.tif",
            source(image::ColorType::Rgba32F, image::ImageFormat::Tiff),
        ),
    ];
    // The 8-bit render is measured only when a deep one is
    // served: one refused before its frame needs no yardstick.
    let mut eight_bit = None;
    let mut eight_bit_peak = || {
        *eight_bit.get_or_insert_with(|| {
            let png = source(image::ColorType::Rgba8, image::ImageFormat::Png);
            let (outcome, held) = render(&png, "rgba8.png", remainder);
            outcome.unwrap_or_else(|e| panic!("the 8-bit render: {e}"));
            held
        })
    };
    for (name, bytes) in &deep {
        let (outcome, held) = render(bytes, name, remainder);
        // Visible under `--nocapture`; captured in the gate.
        eprintln!("{name}: held {held} bytes at most, {outcome:?}");
        if refused_for_bytes(&outcome) {
            assert!(
                held < REFUSED_PEAK,
                "{name} was refused after holding {held} bytes"
            );
            continue;
        }
        outcome.unwrap_or_else(|e| panic!("{name}: {e}"));
        let eight = eight_bit_peak();
        assert!(
            held as f64 <= 1.2 * eight as f64,
            "{name} held {held} bytes at most, the 8-bit source {eight}"
        );
    }
}

#[test]
fn every_step_past_the_byte_limit_is_refused_before_it_allocates() {
    // A float source, and steps each of which would allocate
    // a frame just past the budget in its layout: what the
    // refused render holds is the 2x2 decode, never the frame.
    let square = source(image::ColorType::Rgba32F, image::ImageFormat::Tiff);
    let wide = {
        let img = DynamicImage::ImageRgba32F(image::Rgba32FImage::from_pixel(
            2,
            1,
            image::Rgba([0.2, 0.4, 0.6, 1.0]),
        ));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Tiff)
            .unwrap();
        out
    };
    for (bytes, remainder) in [
        (&square, "5000x5001/filters:format(jpeg)"),
        (&square, "fit-in:5001x5001/filters:format(jpeg)"),
        (&square, "crop:5000x5001/filters:format(jpeg)"),
        (&square, "pad:5001x5001/filters:format(jpeg)"),
        (&wide, "pad:4000x4000:blur/filters:format(jpeg)"),
    ] {
        let (outcome, held) = render(bytes, "a.tif", remainder);
        eprintln!("{remainder}: held {held} bytes at most, {outcome:?}");
        assert!(refused_for_bytes(&outcome), "{remainder}: {outcome:?}");
        assert!(
            held < REFUSED_PEAK,
            "{remainder} was refused after holding {held} bytes"
        );
    }
}

#[test]
fn a_blurred_pad_builds_its_whole_cover_only_within_the_budget() {
    // `pad:1000x1000:blur` over a 5x1000 strip scales it to a
    // 1000x200000 cover and keeps the canvas's part of it. An
    // RGBA8 strip has only that part computed (issue #806): it
    // renders, holding a small part of the 800 MB the whole
    // cover would. A 16-bit strip is scaled to the whole cover,
    // in its own layout, 1.6 GB: it is refused before anything
    // of it is allocated.
    let strip = |layout: image::ColorType| {
        let rgba = image::RgbaImage::from_fn(5, 1000, |x, y| {
            image::Rgba([(x * 50) as u8, (y % 256) as u8, 90, 255])
        });
        let img = DynamicImage::ImageRgba8(rgba);
        let img = match layout {
            image::ColorType::Rgba8 => img,
            image::ColorType::Rgba16 => DynamicImage::ImageRgba16(img.to_rgba16()),
            other => panic!("no strip for {other:?}"),
        };
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        out
    };
    let remainder = "pad:1000x1000:blur/filters:format(jpeg)";
    let (outcome, held) = render(&strip(image::ColorType::Rgba8), "strip.png", remainder);
    eprintln!("the RGBA8 strip: held {held} bytes at most, {outcome:?}");
    outcome.unwrap_or_else(|e| panic!("the RGBA8 strip: {e}"));
    assert!(held < 100 << 20, "the RGBA8 strip held {held} bytes");
    let (outcome, held) = render(&strip(image::ColorType::Rgba16), "strip.png", remainder);
    eprintln!("the RGBA16 strip: held {held} bytes at most, {outcome:?}");
    assert!(refused_for_bytes(&outcome), "the RGBA16 strip: {outcome:?}");
    assert!(
        held < REFUSED_PEAK,
        "the RGBA16 strip was refused after holding {held} bytes"
    );
}

#[test]
fn a_deep_overlay_is_held_like_an_eight_bit_one_or_refused() {
    // `overlay(...,width)` scales the part of the overlay that
    // lands on the image in the overlay's own layout, before it
    // becomes RGBA8. A 2x1 overlay at the full width of a
    // 10000x5000 image lands a 10000x5000 part: 200 MB as RGBA8,
    // 800 MB as float, past the budget. Like a step's image, it
    // is refused, or holds no more than 1.2 times what the 8-bit
    // overlay holds.
    use crabcrop::engine::ProcessContext;
    use crabcrop::filters::{WatermarkImage, WatermarkImages};
    let base = source(image::ColorType::Rgba8, image::ImageFormat::Png);
    let url = "http://wm.example/o.png";
    let render_with = |layout: image::ColorType, (w, h): (u32, u32)| {
        let rgba = image::RgbaImage::from_pixel(2, 1, image::Rgba([200, 30, 30, 255]));
        let img = match layout {
            image::ColorType::Rgba8 => DynamicImage::ImageRgba8(rgba),
            _ => DynamicImage::ImageRgba32F(DynamicImage::ImageRgba8(rgba).to_rgba32f()),
        };
        let mut overlays = WatermarkImages::new();
        overlays.insert(url.to_string(), WatermarkImage::new(img));
        let path = format!(
            "/unsafe/{w}x{h}/filters:overlay(http%3A%2F%2Fwm.example%2Fo.png,0,0,100,normal,100):format(jpeg)/example.com/a.png"
        );
        let (_, _, request) = parse_request(&path).unwrap();
        let ctx = ProcessContext {
            watermarks: Some(&overlays),
            ..Default::default()
        };
        peak(|| {
            Engine
                .process_preferred(&base, &request.operations, ctx)
                .map(drop)
        })
    };
    // Inside the budget, both render.
    for layout in [image::ColorType::Rgba8, image::ColorType::Rgba32F] {
        let (outcome, held) = render_with(layout, (4000, 2000));
        eprintln!("4000x2000 with a {layout:?} overlay: held {held} bytes at most, {outcome:?}");
        outcome.unwrap_or_else(|e| panic!("4000x2000, {layout:?}: {e}"));
    }
    // Past it, the 8-bit render is measured only when the float
    // one is served: one refused needs no yardstick.
    let (deep, deep_held) = render_with(image::ColorType::Rgba32F, (10000, 5000));
    eprintln!("10000x5000 with a float overlay: held {deep_held} bytes at most, {deep:?}");
    if refused_for_bytes(&deep) {
        return;
    }
    deep.unwrap_or_else(|e| panic!("10000x5000, float: {e}"));
    let (eight, eight_held) = render_with(image::ColorType::Rgba8, (10000, 5000));
    eight.unwrap_or_else(|e| panic!("10000x5000, 8-bit: {e}"));
    assert!(
        deep_held as f64 <= 1.2 * eight_held as f64,
        "10000x5000: the float overlay held {deep_held} bytes at most, the 8-bit one {eight_held}"
    );
}
