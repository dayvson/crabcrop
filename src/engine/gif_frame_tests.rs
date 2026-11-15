//! Animated GIF frames prepared on several threads (issue
//! #667): the file is the bytes one thread writes, and the
//! deadline, a frame's failure and a panic end the encode as
//! they do on one thread.
//!
//! Every test names its thread count, so none depends on the
//! cores of the host it runs on.

use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Wait until another thread has set `flag`. The wait ends
/// after ten seconds whatever happened, so a test whose other
/// thread never comes fails on its assertions and does not
/// hang.
fn wait_for(flag: &AtomicBool) {
    let waited = Instant::now();
    while !flag.load(Ordering::SeqCst) && waited.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// `count` frames of `width` x `height`: a pattern that
/// moves from frame to frame, opaque or with clear and
/// translucent pixels, each frame with a delay of its own.
fn animation(count: usize, width: u32, height: u32, transparency: bool) -> Frames {
    let frame = |i: usize| {
        let shift = i as u32;
        let buffer = image::RgbaImage::from_fn(width, height, |x, y| {
            let alpha = match (x + 2 * y + shift) % 7 {
                0 if transparency => 0,
                1 if transparency => 128,
                _ => 255,
            };
            image::Rgba([
                (x * 5 + shift * 31) as u8,
                (y * 7 + shift * 3) as u8,
                ((x ^ y) * 3 + shift * 11) as u8,
                alpha,
            ])
        });
        image::Frame::from_parts(
            buffer,
            0,
            0,
            image::Delay::from_numer_denom_ms(40 + 30 * shift, 1),
        )
    };
    Frames {
        first: frame(0),
        rest: (1..count).map(frame).collect(),
    }
}

/// The delay `animation` gives frame `i`, as the GIF stores
/// it: hundredths of a second. A test's stand-in quantizer
/// reads it to tell which frame it was handed.
fn delay_of(i: usize) -> u16 {
    (4 + 3 * i) as u16
}

fn frame_of(delay: u16) -> usize {
    usize::from(delay - 4) / 3
}

/// What a GIF decoder reads back: per frame, its delay, its
/// disposal method, and whether it names a transparent color.
fn frames_written(gif: &[u8]) -> Vec<(u16, gif::DisposalMethod, bool)> {
    let mut decoder = gif::DecodeOptions::new()
        .read_info(std::io::Cursor::new(gif))
        .expect("the GIF parses");
    let mut frames = Vec::new();
    while let Some(frame) = decoder.read_next_frame().expect("the frame decodes") {
        frames.push((frame.delay, frame.dispose, frame.transparent.is_some()));
    }
    frames
}

#[test]
fn frames_prepared_on_threads_write_the_bytes_one_thread_writes() {
    // One frame, two frames at the 64x64 threshold, a canvas
    // far under it and an odd one a pixel under it, an odd
    // canvas past it, and more frames than threads; frames
    // with and without transparency.
    for (count, width, height, transparency) in [
        (1, 80, 64, true),
        (2, 64, 64, true),
        (2, 5, 3, true),
        (3, 65, 63, false),
        (5, 71, 65, true),
        (6, 96, 64, false),
    ] {
        let frames = animation(count, width, height, transparency);
        let what = format!("{count} frames of {width}x{height}");
        let one = encode_animated_gif_on(&frames, None, 1, gif_frame_quantized).unwrap();
        // More threads than frames are never asked for, and
        // would find nothing to do if they were.
        for workers in [2, 4, 9] {
            let threaded =
                encode_animated_gif_on(&frames, None, workers, gif_frame_quantized).unwrap();
            assert!(
                threaded == one,
                "{what} on {workers} threads: {} bytes against {}",
                threaded.len(),
                one.len()
            );
        }
        // The request path, whatever this host's cores, with
        // a deadline that never comes.
        let far = Instant::now() + Duration::from_secs(3600);
        assert!(
            encode_animated_gif(&frames, Some(far)).unwrap() == one,
            "{what}"
        );
        // The file holds every frame in order, with its own
        // delay, the one disposal the engine writes, and a
        // transparent color exactly where a frame has clear
        // pixels.
        let written = frames_written(&one);
        assert_eq!(written.len(), count, "{what}");
        for (i, (delay, dispose, transparent)) in written.into_iter().enumerate() {
            assert_eq!(delay, delay_of(i), "{what}, frame {i}");
            assert_eq!(dispose, gif::DisposalMethod::Background, "{what}");
            assert_eq!(transparent, transparency, "{what}, frame {i}");
        }
        let decoded = decode_gif_frames(&one, None).unwrap();
        assert_eq!(decoded.len(), count, "{what}");
        assert_eq!(decoded.dimensions(), (width, height), "{what}");
    }
}

#[test]
fn the_thread_count_follows_the_cores_the_frames_and_the_size() {
    // The cores, four at most.
    assert_eq!(gif_frame_workers(1, 12, 480, 320), 1);
    assert_eq!(gif_frame_workers(2, 12, 480, 320), 2);
    assert_eq!(gif_frame_workers(3, 12, 480, 320), 3);
    assert_eq!(gif_frame_workers(4, 12, 480, 320), 4);
    assert_eq!(gif_frame_workers(8, 12, 480, 320), 4);
    assert_eq!(gif_frame_workers(256, 200, 480, 320), 4);
    // Never more than the frames, and one for a single frame.
    assert_eq!(gif_frame_workers(8, 3, 480, 320), 3);
    assert_eq!(gif_frame_workers(8, 2, 480, 320), 2);
    assert_eq!(gif_frame_workers(8, 1, 480, 320), 1);
    // One under 64x64 pixels, whatever the shape.
    assert_eq!(gif_frame_workers(8, 12, 64, 64), 4);
    assert_eq!(gif_frame_workers(8, 12, 63, 64), 1);
    assert_eq!(gif_frame_workers(8, 12, 4095, 1), 1);
    assert_eq!(gif_frame_workers(8, 12, 4096, 1), 4);
    assert_eq!(gif_frame_workers(8, 12, 16383, 16383), 4);
    // A host that reports no core still gets the one thread.
    assert_eq!(gif_frame_workers(0, 12, 480, 320), 1);
}

#[test]
fn one_thread_prepares_every_frame_on_the_request_thread() {
    let frames = animation(6, 64, 64, true);
    let request_thread = std::thread::current().id();
    let elsewhere = AtomicUsize::new(0);
    encode_animated_gif_on(&frames, None, 1, |rgba, width, height, delay| {
        if std::thread::current().id() != request_thread {
            elsewhere.fetch_add(1, Ordering::SeqCst);
        }
        gif_frame_quantized(rgba, width, height, delay)
    })
    .unwrap();
    assert_eq!(elsewhere.load(Ordering::SeqCst), 0);
}

#[test]
fn no_more_threads_prepare_frames_than_were_asked_for() {
    let frames = animation(12, 64, 64, false);
    let request_thread = std::thread::current().id();
    for workers in [2, 4] {
        let (in_work, most, calls) = (
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
        );
        let threads = std::sync::Mutex::new(std::collections::HashSet::new());
        let request_has_a_frame = AtomicBool::new(false);
        encode_animated_gif_on(&frames, None, workers, |rgba, width, height, delay| {
            calls.fetch_add(1, Ordering::SeqCst);
            threads.lock().unwrap().insert(std::thread::current().id());
            most.fetch_max(in_work.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
            // Every other thread holds its frame until the
            // request's thread has one, so the frames in work
            // at once are counted with all the threads there
            // are, however late one starts.
            if std::thread::current().id() == request_thread {
                request_has_a_frame.store(true, Ordering::SeqCst);
            } else {
                wait_for(&request_has_a_frame);
            }
            std::thread::sleep(Duration::from_millis(5));
            let frame = gif_frame_quantized(rgba, width, height, delay);
            in_work.fetch_sub(1, Ordering::SeqCst);
            frame
        })
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 12, "each frame once");
        assert!(
            (1..=workers).contains(&most.load(Ordering::SeqCst)),
            "{} frames in work at once on {workers} threads",
            most.load(Ordering::SeqCst)
        );
        // The request's own thread is one of the threads.
        let threads = threads.into_inner().unwrap();
        assert!(
            threads.len() <= workers && threads.contains(&request_thread),
            "{} threads prepared frames where {workers} were asked for",
            threads.len()
        );
        // The threads borrow these counters, which the
        // compiler allows only because every one of them has
        // ended by the time the encode returns.
        assert_eq!(in_work.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn a_deadline_that_passes_during_the_encode_stops_it_there() {
    let frames = animation(12, 64, 64, true);
    for workers in [1, 2, 4] {
        let deadline = Instant::now() + Duration::from_millis(200);
        let started = AtomicUsize::new(0);
        let result = encode_animated_gif_on(
            &frames,
            Some(deadline),
            workers,
            |rgba, width, height, delay| {
                // Every frame in work when the deadline
                // passes is still in work: each thread holds
                // its frame until then.
                started.fetch_add(1, Ordering::SeqCst);
                while Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(2));
                }
                gif_frame_quantized(rgba, width, height, delay)
            },
        );
        assert_eq!(
            result.unwrap_err(),
            EngineError::Timeout,
            "{workers} threads"
        );
        // A thread takes its next frame only after the
        // deadline, where the check stops it: at most one
        // frame per thread was ever prepared, and the other
        // frames never were.
        let started = started.load(Ordering::SeqCst);
        assert!(
            started <= workers,
            "{started} of 12 frames prepared on {workers} threads"
        );
    }
    // A deadline already behind the encode prepares nothing.
    let started = AtomicUsize::new(0);
    let past = Instant::now() - Duration::from_millis(1);
    for workers in [1, 4] {
        let result = encode_animated_gif_on(
            &frames,
            Some(past),
            workers,
            |rgba, width, height, delay| {
                started.fetch_add(1, Ordering::SeqCst);
                gif_frame_quantized(rgba, width, height, delay)
            },
        );
        assert_eq!(result.unwrap_err(), EngineError::Timeout);
    }
    assert_eq!(started.load(Ordering::SeqCst), 0);
}

#[test]
fn a_frame_that_fails_ends_the_encode_with_the_first_failure_in_frame_order() {
    let frames = animation(12, 64, 64, false);
    for workers in [1, 2, 3, 4] {
        let prepared = AtomicUsize::new(0);
        let (later_frame_failed, frame_3_failed) = (AtomicBool::new(false), AtomicBool::new(false));
        let result =
            encode_animated_gif_on(&frames, None, workers, |rgba, width, height, delay| {
                match frame_of(delay) {
                    // Frames 3 and 5 both fail. With several
                    // threads frame 3 holds its failure until
                    // frame 5 has failed, so the later
                    // frame's failure is the first to happen.
                    3 => {
                        if workers > 1 {
                            wait_for(&later_frame_failed);
                        }
                        frame_3_failed.store(true, Ordering::SeqCst);
                        Err(EngineError::Encode("frame 3".to_string()))
                    }
                    5 => {
                        later_frame_failed.store(true, Ordering::SeqCst);
                        Err(EngineError::Encode("frame 5".to_string()))
                    }
                    frame => {
                        // A frame past both holds until frame
                        // 3 has failed, so no thread can be
                        // through with one and take another
                        // before both failures are in.
                        if frame > 5 {
                            wait_for(&frame_3_failed);
                        }
                        prepared.fetch_add(1, Ordering::SeqCst);
                        gif_frame_quantized(rgba, width, height, delay)
                    }
                }
            });
        assert_eq!(
            result.unwrap_err(),
            EngineError::Encode("frame 3".to_string()),
            "{workers} threads"
        );
        // One thread stops at frame 3. Several prepare frames
        // 0, 1, 2 and 4, and every thread but the ones on
        // frames 3 and 5 can have taken one more: the frames
        // behind are never prepared.
        let prepared = prepared.load(Ordering::SeqCst);
        if workers == 1 {
            assert_eq!(prepared, 3);
            assert!(!later_frame_failed.load(Ordering::SeqCst));
        } else {
            assert!(later_frame_failed.load(Ordering::SeqCst));
            assert!(
                (4..=2 + workers).contains(&prepared),
                "{prepared} frames prepared on {workers} threads"
            );
        }
    }
}

#[test]
fn a_frame_the_format_cannot_hold_fails_the_same_on_any_thread_count() {
    // A frame with no pixels is one the real quantizer
    // refuses.
    let mut frames = animation(6, 64, 64, true);
    frames.rest[2] = image::Frame::new(image::RgbaImage::new(0, 0));
    for workers in [1, 2, 4] {
        assert_eq!(
            encode_animated_gif_on(&frames, None, workers, gif_frame_quantized).unwrap_err(),
            EngineError::Encode("gif: zero dimension".to_string()),
            "{workers} threads"
        );
    }
    // And a canvas past the format's limit is refused before
    // any frame is prepared.
    let wide = Frames {
        first: image::Frame::new(image::RgbaImage::new(65536, 1)),
        rest: vec![image::Frame::new(image::RgbaImage::new(65536, 1))],
    };
    let prepared = AtomicUsize::new(0);
    for workers in [1, 2] {
        let err = encode_animated_gif_on(&wide, None, workers, |rgba, width, height, delay| {
            prepared.fetch_add(1, Ordering::SeqCst);
            gif_frame_quantized(rgba, width, height, delay)
        })
        .unwrap_err();
        assert_eq!(
            err,
            EngineError::Encode("gif: width exceeds the 65535 format limit".to_string())
        );
    }
    assert_eq!(prepared.load(Ordering::SeqCst), 0);
}

/// What a caught panic said.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

#[test]
fn a_panic_while_preparing_a_frame_unwinds_on_the_request_thread() {
    let frames = animation(12, 64, 64, false);
    let request_thread = std::thread::current().id();
    // One thread: the request's own, which never goes past
    // the frame that panics.
    let started = AtomicUsize::new(0);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_animated_gif_on(&frames, None, 1, |rgba, width, height, delay| {
            started.fetch_add(1, Ordering::SeqCst);
            if frame_of(delay) == 2 {
                panic!("frame 2 cannot be prepared");
            }
            gif_frame_quantized(rgba, width, height, delay)
        })
    }));
    let panic = outcome.expect_err("the panic reaches the caller");
    assert_eq!(panic_message(&*panic), "frame 2 cannot be prepared");
    assert_eq!(started.load(Ordering::SeqCst), 3);
    // Several threads: every frame another thread takes
    // panics, and the request's thread holds its own frame
    // until one has. The panic is raised again on the
    // request's thread, with what it said.
    for workers in [2, 4] {
        let another_thread_panicked = AtomicBool::new(false);
        let in_work = AtomicUsize::new(0);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            encode_animated_gif_on(&frames, None, workers, |rgba, width, height, delay| {
                if std::thread::current().id() != request_thread {
                    another_thread_panicked.store(true, Ordering::SeqCst);
                    panic!("a frame cannot be prepared on another thread");
                }
                in_work.fetch_add(1, Ordering::SeqCst);
                wait_for(&another_thread_panicked);
                let frame = gif_frame_quantized(rgba, width, height, delay);
                in_work.fetch_sub(1, Ordering::SeqCst);
                frame
            })
        }));
        let panic = outcome.expect_err("the panic reaches the caller");
        assert_eq!(
            panic_message(&*panic),
            "a frame cannot be prepared on another thread",
            "{workers} threads"
        );
        // Every thread has ended, the request's own after it
        // finished the frame it had in work.
        assert!(another_thread_panicked.load(Ordering::SeqCst));
        assert_eq!(in_work.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn a_panic_stops_the_other_threads_before_their_next_item() {
    /// Says when it is dropped.
    struct Dropped<'a>(&'a AtomicBool);
    impl Drop for Dropped<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    for workers in [2, 4] {
        // The first item asked for is returned at once, and
        // the thread that took it panics on its next. Every
        // other thread holds its item until that first result
        // is dropped, which the panic's unwinding does once
        // it has told the others to stop.
        let unwound = AtomicBool::new(false);
        let calls = AtomicUsize::new(0);
        let first_thread = std::sync::Mutex::new(None);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            in_index_order(64, workers, |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                let me = std::thread::current().id();
                let mut first = first_thread.lock().unwrap();
                match *first {
                    None => {
                        *first = Some(me);
                        return Ok(Some(Dropped(&unwound)));
                    }
                    Some(thread) if thread == me => {
                        drop(first);
                        panic!("the second item cannot be prepared");
                    }
                    Some(_) => drop(first),
                }
                while !unwound.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Ok(None)
            })
        }));
        let panic = outcome.err().expect("the panic reaches the caller");
        assert_eq!(
            panic.downcast_ref::<&str>().copied(),
            Some("the second item cannot be prepared")
        );
        // Two items on the first thread and one on each of
        // the others, of 64: without the stop the others
        // would have gone on to the last.
        let calls = calls.load(Ordering::SeqCst);
        assert!(
            (2..=workers + 1).contains(&calls),
            "{calls} items taken on {workers} threads"
        );
    }
}

#[test]
fn work_in_index_order_returns_every_result_in_place() {
    for (count, workers) in [(0, 4), (1, 4), (7, 1), (7, 3), (40, 4), (3, 9)] {
        let squares = in_index_order(count, workers, |i| Ok(i * i)).unwrap();
        let expected: Vec<usize> = (0..count).map(|i| i * i).collect();
        assert_eq!(squares, expected, "{count} items on {workers} threads");
    }
    // The first failure in index order is the one returned.
    for workers in [1, 2, 4] {
        let err = in_index_order(40, workers, |i| {
            if i % 10 == 9 {
                Err(EngineError::Encode(format!("item {i}")))
            } else {
                Ok(i)
            }
        })
        .unwrap_err();
        assert_eq!(err, EngineError::Encode("item 9".to_string()));
    }
}
