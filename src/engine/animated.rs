//! Animated sources (issue #741): decoding GIF and WebP
//! frames under the animation budgets, the geometry and
//! overlay parts the frames of one render share (smart crop
//! and trim windows, scaled overlays), running the chain over
//! every frame, and encoding the frames as an animated GIF or
//! WebP. The video path in engine.rs renders its frames
//! through the same functions.
//!
//! engine.rs re-exports what the pipeline, the server and the
//! tests use.

use std::io::Cursor;

use image::DynamicImage;

use super::chain::{ChainState, apply_operations};
use super::decode::decode_limits;
use super::limits::{
    MAX_ANIMATED_FRAMES, MAX_ANIMATED_PIXELS, MAX_DECODE_ALLOC, MAX_DECODE_DIMENSION, OutputBudget,
    check_operation_count, dpr_factor,
};
use super::{EngineError, ProcessContext, Window, gif_frame_quantized, state, webp_anim};
use crate::filters::EncodeOptions;
use crate::url::{Align, Filter, FormatOptions, ImageFormat, Operation, PlainFilter};

#[cfg(doc)]
use super::resize::resize_lanczos3_lent;

/// Inference-class operations do not run on animated
/// sources (issue #261): the animated paths apply the
/// chain per frame, so one request would run inference
/// up to 200 times and hold an AI gate permit for
/// minutes to hours. The set is the AI gate's own
/// (`RequestDirectives::uses_ai_beyond_smart_crop`): a
/// smart crop stays available on animated sources under
/// the processing deadline. Its focal analysis is the
/// built-in saliency statistic, or, with a face or focal
/// provider configured, an inference that takes the AI
/// gate like any other (issue #379). An animated render,
/// image or video, runs that analysis once, on the first
/// frame, and replays the window (issues #376, #713). A
/// still spelling of the same source (a `frame:` selection
/// renders one frame) runs these operations normally.
fn refuse_animated_ai(directives: &crate::url::RequestDirectives) -> Result<(), EngineError> {
    let inference_bound = directives.removebg
        || directives.studio
        || directives.remove_region
        || directives.upscale
        || directives.needs_face_model
        || directives.has_class_redaction()
        || directives.objects_crop
        || directives.debug_focal;
    if inference_bound {
        return Err(EngineError::Unsupported(
            "AI filters do not run on animated sources".to_string(),
        ));
    }
    Ok(())
}

/// The placeholder of an animated or video render, when the
/// caller asked for one: it describes the first processed
/// frame.
pub(in crate::engine) fn first_frame_placeholder(
    frames: &mut Frames,
    ctx: &ProcessContext<'_>,
) -> Option<crate::placeholder::Placeholder> {
    ctx.placeholders
        .then(|| frames.with_first(crate::placeholder::compute))
}

/// Run the operations over every frame of an animated render,
/// for the animated image and the video paths alike: the
/// inference-class operations refused first (see
/// [`refuse_animated_ai`]), the frames' shared geometry in
/// `share` settled before any frame renders (see
/// [`survey_trims`]), and the run's pixels charged against one
/// output budget, which the first frame's size settles up front
/// (see [`OutputBudget::check_frames`]). The rendered frames
/// keep their delays. The encoding directives are a property of
/// the ops, not of a frame, so every frame reports the same
/// ones; the first frame's copy is the request's and comes back
/// with the frames.
pub(in crate::engine) fn render_frames(
    mut frames: Frames,
    ops: &[Operation],
    ctx: &ProcessContext<'_>,
    directives: &crate::url::RequestDirectives,
    mut share: FrameShare,
) -> Result<(Frames, EncodeOptions), EngineError> {
    refuse_animated_ai(directives)?;
    let cap = if directives.no_upscale {
        Some(frames.dimensions())
    } else {
        None
    };
    let dpr = dpr_factor(directives.dpr, ctx);
    survey_trims(&mut frames, ops, cap, dpr, ctx, directives, &mut share)?;
    let frame_count = frames.len();
    let mut budget = OutputBudget::new();
    let mut render = |frame: image::Frame| {
        ctx.check_deadline()?;
        let delay = frame.delay();
        let img = DynamicImage::ImageRgba8(frame.into_buffer());
        let outcome = apply_operations(img, ops, cap, dpr, ctx, directives, Some(&mut share))?;
        Ok::<_, EngineError>((outcome.img, outcome.encode_opts, delay))
    };
    let (img, encode_opts, delay) = render(frames.first)?;
    budget.check_frames(img.width(), img.height(), frame_count)?;
    budget.charge(img.width(), img.height())?;
    // A rendered frame is the chain's own image, so an RGBA8
    // one becomes the frame without a copy (issue #713); any
    // other layout converts as it always did.
    let first = image::Frame::from_parts(img.into_rgba8(), 0, 0, delay);
    let mut rest = Vec::with_capacity(frames.rest.len());
    for frame in frames.rest {
        let (img, _, delay) = render(frame)?;
        budget.charge(img.width(), img.height())?;
        rest.push(image::Frame::from_parts(img.into_rgba8(), 0, 0, delay));
    }
    Ok((Frames { first, rest }, encode_opts))
}

/// Encode rendered frames as an animated GIF: one screen the
/// first frame's size, and each frame quantized onto a palette
/// of its own (see [`gif_frame_quantized`]) with its delay in
/// hundredths of a second. GIF output has no quality lever and
/// keeps the GIF encoder's defaults otherwise, as it always has.
/// The frames are prepared on up to two threads (see
/// [`gif_frame_workers`]) and written in frame order.
pub(in crate::engine) fn encode_animated_gif(
    frames: &Frames,
    deadline: Option<std::time::Instant>,
) -> Result<Vec<u8>, EngineError> {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let (width, height) = frames.dimensions();
    let workers = gif_frame_workers(cores, frames.len(), width, height);
    encode_animated_gif_on(frames, deadline, workers, gif_frame_quantized)
}

/// The most threads that prepare the frames of one animated
/// GIF (issue #695). An animated GIF render holds one permit of
/// the stills lane, like any still, so its threads share the
/// cores with the still renders beside it. With four, a still
/// render beside one GIF encode took 1.13 to 1.29 times as long
/// as beside the single-thread encoder, and its 90th percentile
/// about 1.6 times; with two it took 0.98 times as long, and a
/// GIF render still got 1.76 times the throughput of one thread
/// (four gave 2.34). The bytes are the same at any count.
pub(in crate::engine) const GIF_FRAME_THREADS: usize = 2;

/// How many threads prepare the frames of an animated GIF on
/// a host with `cores` cores (issue #667): the cores, at most
/// [`GIF_FRAME_THREADS`], never more than the frames, and one
/// for a single frame or for an animation under 64x64 pixels,
/// too small to repay a thread.
pub(in crate::engine) fn gif_frame_workers(
    cores: usize,
    frames: usize,
    width: u32,
    height: u32,
) -> usize {
    if frames < 2 || u64::from(width) * u64::from(height) < 64 * 64 {
        return 1;
    }
    cores.clamp(1, GIF_FRAME_THREADS).min(frames)
}

/// [`encode_animated_gif`] on `workers` threads. `quantize`
/// turns one frame's RGBA pixels, width, height and delay
/// into its GIF frame: always [`gif_frame_quantized`], and a
/// parameter only so a test can see which frames are
/// prepared.
///
/// Each frame is quantized onto its own palette and compressed
/// by itself, with nothing carried from one frame to the next,
/// so the frames can be prepared in any order: one thread
/// writes each as it goes, and several prepare them side by
/// side (see [`in_index_order`]) and then write them in frame
/// order, the LZW pass `write_frame` runs having run on the
/// thread that quantized the frame. The file is the same
/// bytes either way.
///
/// The prepared frames wait for the write as palette and
/// LZW data, the file's own image data: at most about a byte
/// and a half per pixel, where the frames they come from hold
/// four, so the frame cap and the pixel budgets that bound the
/// frames bound them too. Each thread works on one frame at a
/// time.
pub(in crate::engine) fn encode_animated_gif_on(
    frames: &Frames,
    deadline: Option<std::time::Instant>,
    workers: usize,
    quantize: impl Fn(&[u8], u32, u32, u16) -> Result<gif::Frame<'static>, EngineError> + Sync,
) -> Result<Vec<u8>, EngineError> {
    let mut out = Vec::new();
    let (canvas_w, canvas_h) = frames.dimensions();
    let mut enc = gif::Encoder::new(
        &mut out,
        u16::try_from(canvas_w).map_err(|_| {
            EngineError::Encode("gif: width exceeds the 65535 format limit".to_string())
        })?,
        u16::try_from(canvas_h).map_err(|_| {
            EngineError::Encode("gif: height exceeds the 65535 format limit".to_string())
        })?,
        &[],
    )
    .map_err(|e| EngineError::Encode(e.to_string()))?;
    let prepare = |f: &image::Frame| {
        // The deadline is checked per frame inside the encode
        // loop itself (issue #322), on whichever thread takes
        // the frame: a many-frame GIF encode is chunked work,
        // so it aborts at the deadline instead of running
        // every frame's quantization first and being noticed
        // only afterwards.
        state::check_deadline(deadline)?;
        // GIF delays are hundredths of a second; the image
        // crate saturated this conversion at u16::MAX, and so
        // does this.
        let (numer, denom) = f.delay().numer_denom_ms();
        let delay = u16::try_from(numer / denom / 10).unwrap_or(u16::MAX);
        let buffer = f.buffer();
        quantize(buffer.as_raw(), buffer.width(), buffer.height(), delay)
    };
    if workers <= 1 {
        for f in frames.iter() {
            let frame = prepare(f)?;
            enc.write_frame(&frame)
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
    } else {
        let all: Vec<&image::Frame> = frames.iter().collect();
        let prepared = in_index_order(all.len(), workers, |index| {
            let mut frame = prepare(all[index])?;
            frame.make_lzw_pre_encoded();
            Ok(frame)
        })?;
        for frame in &prepared {
            enc.write_lzw_pre_encoded_frame(frame)
                .map_err(|e| EngineError::Encode(e.to_string()))?;
        }
    }
    drop(enc);
    Ok(out)
}

/// Run `work` for every index below `count` on up to `workers`
/// threads, the calling thread among them, and return the
/// results in index order.
///
/// The threads take the indexes in order from a shared
/// counter. A failure stops every thread before its next
/// index, so the work behind it is never started, and the
/// failure returned is the first in index order: the one a
/// loop over the indexes would have met. (An index is left
/// untaken only behind one that failed, and every index taken
/// is finished, so no index below the first failure is
/// missing.) A panic in `work` stops the other threads the
/// same way and is raised again on the calling thread, where
/// it unwinds as it would have from a loop.
///
/// The threads are scoped: all of them have ended when this
/// returns, however it returns. A thread the host cannot
/// start is one helper fewer, not a failure, since the
/// calling thread takes indexes too.
pub(in crate::engine) fn in_index_order<T: Send>(
    count: usize,
    workers: usize,
    work: impl Fn(usize) -> Result<T, EngineError> + Sync,
) -> Result<Vec<T>, EngineError> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    /// Sets the flag when dropped by a panic's unwinding.
    struct StopOnPanic<'a>(&'a AtomicBool);
    impl Drop for StopOnPanic<'_> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                self.0.store(true, Ordering::Relaxed);
            }
        }
    }
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let take = || {
        let mut done: Vec<(usize, Result<T, EngineError>)> = Vec::new();
        // Declared after `done`, so a panic's unwinding sets
        // the flag before it drops what this thread finished.
        let _stop_on_panic = StopOnPanic(&stop);
        while !stop.load(Ordering::Relaxed) {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= count {
                break;
            }
            let result = work(index);
            if result.is_err() {
                stop.store(true, Ordering::Relaxed);
            }
            done.push((index, result));
        }
        done
    };
    let mut results: Vec<Option<Result<T, EngineError>>> = (0..count).map(|_| None).collect();
    std::thread::scope(|scope| {
        let helpers: Vec<_> = (1..workers.min(count))
            .filter_map(|_| std::thread::Builder::new().spawn_scoped(scope, take).ok())
            .collect();
        let mut done = take();
        for helper in helpers {
            match helper.join() {
                Ok(more) => done.extend(more),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        for (index, result) in done {
            results[index] = Some(result);
        }
    });
    results
        .into_iter()
        .map(|result| result.expect("an index is left untaken only behind a failed one"))
        .collect()
}

/// Collect an animation decoder's frames under the shared
/// limits: a frame cap, a total-decoded-pixels budget, and
/// the decode allocation budget charged on the retained
/// RGBA bytes, so a small file cannot expand into gigabytes
/// of frames. The pixel budget alone is not a memory bound:
/// at its 100M-pixel cap the retained frames are 400 MiB,
/// well past the 256 MiB allocation budget every still
/// decode honors. The request's `deadline` is checked before
/// each frame decodes, so a long animation stops with
/// [`EngineError::Timeout`] at the deadline instead of
/// decoding on to its last frame. Errors when the source is
/// malformed, over budget, or carries no frames. `format`
/// names the source format in error messages.
pub(in crate::engine) fn collect_frames(
    mut frames: impl Iterator<Item = Result<image::Frame, image::ImageError>>,
    format: &str,
    deadline: Option<std::time::Instant>,
) -> Result<Frames, EngineError> {
    let mut collected: Option<Frames> = None;
    let mut total_pixels: u64 = 0;
    let mut total_bytes: u64 = 0;
    loop {
        state::check_deadline(deadline)?;
        let Some(frame) = frames.next() else {
            break;
        };
        let frame = frame.map_err(|e| EngineError::Decode(e.to_string()))?;
        if collected.as_ref().map_or(0, Frames::len) >= MAX_ANIMATED_FRAMES {
            return Err(EngineError::Decode(format!(
                "{format} has more than {MAX_ANIMATED_FRAMES} frames"
            )));
        }
        let frame_pixels = frame.buffer().width() as u64 * frame.buffer().height() as u64;
        total_pixels += frame_pixels;
        if total_pixels > MAX_ANIMATED_PIXELS {
            return Err(EngineError::Decode(format!(
                "{format} frames exceed the {MAX_ANIMATED_PIXELS} decoded-pixel limit"
            )));
        }
        total_bytes += frame_pixels * 4;
        if total_bytes > MAX_DECODE_ALLOC {
            return Err(EngineError::Decode(format!(
                "{format} frames exceed the {MAX_DECODE_ALLOC} byte decode allocation limit"
            )));
        }
        match &mut collected {
            None => {
                collected = Some(Frames {
                    first: frame,
                    rest: Vec::new(),
                })
            }
            Some(frames) => frames.rest.push(frame),
        }
    }
    collected.ok_or_else(|| EngineError::Decode(format!("{format} has no frames")))
}

/// The frames of an animated source (issue #480): never empty,
/// so the first frame is always there to read, the one a still
/// conversion keeps and whose size an animated render starts
/// from.
pub(in crate::engine) struct Frames {
    pub(in crate::engine) first: image::Frame,
    pub(in crate::engine) rest: Vec<image::Frame>,
}

impl Frames {
    pub(in crate::engine) fn len(&self) -> usize {
        1 + self.rest.len()
    }

    pub(in crate::engine) fn iter(&self) -> impl Iterator<Item = &image::Frame> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut image::Frame> {
        std::iter::once(&mut self.first).chain(&mut self.rest)
    }

    /// The first frame's width and height.
    pub(in crate::engine) fn dimensions(&self) -> (u32, u32) {
        self.first.buffer().dimensions()
    }

    /// `f` of the first frame as a `DynamicImage`, lent for the
    /// call (see [`lend_frame`]), where it was copied once for
    /// each look (issue #713): the quality pick and the
    /// placeholder each read the whole frame.
    pub(in crate::engine) fn with_first<T>(&mut self, f: impl FnOnce(&DynamicImage) -> T) -> T {
        lend_frame(&mut self.first, |img| f(img))
    }
}

/// `f` of `frame` as a `DynamicImage`: the frame's buffer is
/// lent for the call and put back, with the frame's position
/// and delay, instead of copied (issues #713, #737). `f` may
/// work on the pixels where they sit, as the trim survey's
/// resize does (see [`resize_lanczos3_lent`]), as long as it
/// leaves them as they came.
fn lend_frame<T>(frame: &mut image::Frame, f: impl FnOnce(&mut DynamicImage) -> T) -> T {
    let (left, top, delay) = (frame.left(), frame.top(), frame.delay());
    let empty = image::Frame::new(image::RgbaImage::new(0, 0));
    let mut img = DynamicImage::ImageRgba8(std::mem::replace(frame, empty).into_buffer());
    let out = f(&mut img);
    *frame = image::Frame::from_parts(img.into_rgba8(), left, top, delay);
    out
}

/// A copy of a lent frame, for a chain whose first step cannot
/// read the frame where it is (see [`ChainState::run_to`]).
pub(in crate::engine) fn copy_of_lent(frame: &DynamicImage) -> DynamicImage {
    #[cfg(test)]
    LENT_FRAME_COPIES.set(LENT_FRAME_COPIES.get() + 1);
    frame.clone()
}

#[cfg(test)]
thread_local! {
    /// How many lent frames [`copy_of_lent`] copied on this
    /// thread, so a test can see which chains read their frames
    /// where they are.
    pub(in crate::engine) static LENT_FRAME_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl IntoIterator for Frames {
    type Item = image::Frame;
    type IntoIter =
        std::iter::Chain<std::iter::Once<image::Frame>, std::vec::IntoIter<image::Frame>>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self.first).chain(self.rest)
    }
}

/// Decode all frames of a GIF source, under the shared limits:
/// the frames' declared sizes are charged first (see
/// [`check_gif_frame_budget`]), the decoder gets explicit
/// dimension/allocation limits (its defaults are none), and
/// collection enforces the shared frame and pixel budgets and
/// the request's deadline (see [`collect_frames`]).
pub(in crate::engine) fn decode_gif_frames(
    bytes: &[u8],
    deadline: Option<std::time::Instant>,
) -> Result<Frames, EngineError> {
    use image::{AnimationDecoder, ImageDecoder};
    check_gif_frame_budget(bytes)?;
    let mut decoder = image::codecs::gif::GifDecoder::new(Cursor::new(bytes))
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    decoder
        .set_limits(decode_limits())
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    collect_frames(decoder.into_frames(), "GIF", deadline)
}

/// Charge a GIF's frames to the animation budgets from their
/// descriptors, before any frame decodes. The decoder hands
/// back every frame composited onto the logical screen, so
/// [`collect_frames`] sees each at the screen's size. Each
/// frame is decoded at its own declared size first, though,
/// and a descriptor may declare one far larger than the
/// screen (the gif crate leaves frames unchecked against it):
/// a 1x1 screen can carry 64-megapixel frames, each decoded in
/// full and then cut down to one pixel. So each frame counts
/// here at the larger of the screen and its own declared size,
/// against the same frame cap and decoded-pixel budget. A frame
/// that fits the screen counts as it always has. The walk reads
/// block structure only, with LZW decoding skipped, as
/// [`is_animated_gif`] does. A file the walk cannot read to its
/// end is left to the decoder, which stops at the same point
/// with its own error, past only frames counted here.
pub(in crate::engine) fn check_gif_frame_budget(bytes: &[u8]) -> Result<(), EngineError> {
    let mut options = gif::DecodeOptions::new();
    options.skip_frame_decoding(true);
    let Ok(mut decoder) = options.read_info(bytes) else {
        return Ok(());
    };
    let screen = u64::from(decoder.width()) * u64::from(decoder.height());
    let mut frames: usize = 0;
    let mut total_pixels: u64 = 0;
    let mut kept_bytes: u64 = 0;
    while let Ok(Some(frame)) = decoder.next_frame_info() {
        frames += 1;
        if frames > MAX_ANIMATED_FRAMES {
            return Err(EngineError::Decode(format!(
                "GIF has more than {MAX_ANIMATED_FRAMES} frames"
            )));
        }
        let declared = u64::from(frame.width) * u64::from(frame.height);
        total_pixels += screen.max(declared);
        if total_pixels > MAX_ANIMATED_PIXELS {
            return Err(EngineError::Decode(format!(
                "GIF frames exceed the {MAX_ANIMATED_PIXELS} decoded-pixel limit"
            )));
        }
        // Each frame comes back composited onto the screen, and the
        // collector charges the bytes it keeps against
        // MAX_DECODE_ALLOC, so the screen's bytes per frame are
        // charged here too, before any frame decodes.
        kept_bytes += screen * 4;
        if kept_bytes > MAX_DECODE_ALLOC {
            return Err(EngineError::Decode(format!(
                "GIF frames exceed the {MAX_DECODE_ALLOC} byte decode allocation limit"
            )));
        }
    }
    Ok(())
}

/// Charge an animated WebP's frames to the animation budgets
/// from its VP8X canvas and its ANMF headers, before any frame
/// decodes, as [`check_gif_frame_budget`] charges a GIF from its
/// descriptors. Every frame composites onto the full canvas, so
/// the frame count times the canvas pixels is the animation's
/// decoded-pixel total, charged against [`MAX_ANIMATED_PIXELS`]
/// as the collector charges the frames it retains; the walk and
/// the image crate's decoder each hold a canvas-sized working
/// buffer and the frame they decode beside the frames they keep,
/// so the canvas is charged against [`MAX_DECODE_ALLOC`] at four
/// bytes a pixel, alone and as many times as there are frames
/// and two more. The image crate's WebP decoder enforces only
/// the per-axis dimension limit, not the allocation budget, so
/// without this charge an over-budget canvas would be allocated
/// before the collector could refuse it. The canvas and the
/// frame count are read by [`webp_anim::header`], the way
/// image-webp reads them, so the pixels charged here are the
/// pixels it allocates. A container that does not parse this
/// far states no canvas; the decoder then opens it and fails
/// before it allocates a canvas.
pub(in crate::engine) fn check_webp_frame_budget(bytes: &[u8]) -> Result<(), EngineError> {
    let Some(header) = webp_anim::header(bytes) else {
        return Ok(());
    };
    // A canvas past the per-axis dimension limit is refused by the
    // image crate's decoder (`set_limits` in `open_webp_animation`)
    // before it allocates, so it is left that error. This charge is
    // for a canvas within the dimension limit, which that decoder
    // admits and would then allocate past the allocation budget it
    // does not enforce.
    if header.width > MAX_DECODE_DIMENSION || header.height > MAX_DECODE_DIMENSION {
        return Ok(());
    }
    let canvas = u64::from(header.width) * u64::from(header.height);
    if canvas.saturating_mul(4) > MAX_DECODE_ALLOC {
        return Err(EngineError::Decode(format!(
            "WebP canvas {}x{} exceeds the {MAX_DECODE_ALLOC} byte decode allocation limit",
            header.width, header.height
        )));
    }
    let total_pixels = u64::from(header.frames).saturating_mul(canvas);
    if total_pixels > MAX_ANIMATED_PIXELS {
        return Err(EngineError::Decode(format!(
            "WebP frames exceed the {MAX_ANIMATED_PIXELS} decoded-pixel limit"
        )));
    }
    // The collector keeps every frame at the canvas's size and
    // charges the bytes it keeps against MAX_DECODE_ALLOC, frame by
    // frame. Charging that total here refuses, before any frame
    // decodes, exactly the animations the collector would refuse
    // partway through.
    if total_pixels.saturating_mul(4) > MAX_DECODE_ALLOC {
        return Err(EngineError::Decode(format!(
            "WebP frames exceed the {MAX_DECODE_ALLOC} byte decode allocation limit"
        )));
    }
    // While its last frame decodes, the image crate's iterator
    // holds two canvas-sized buffers beyond the frames it hands
    // back: its canvas and the frame it decodes, beside the
    // buffer it copies the canvas into for that frame. The walk
    // holds as much at most: its canvas, the frame libwebp
    // decodes, and libwebp's own buffer of a lossless frame's
    // size while it decodes one; its last frame takes its canvas
    // (see `webp_anim`). So the decode is charged two canvases
    // more than the frames it keeps.
    check_webp_held(&header, header.frames)
}

/// [`check_webp_frame_budget`] for a decode that keeps the first
/// frame of an animated WebP alone: the image crate's still
/// decode, which `decode_image` runs for a watermark or overlay
/// image and for a render meta mode reads back. It composites
/// that frame onto a canvas of its own, as its frame iterator
/// does, beside the frame it decodes and the buffer it hands
/// back, so it is charged as a one-frame animation. The canvas
/// is held to the dimension and allocation limits by the probe
/// ahead of that decode.
pub(in crate::engine) fn check_webp_first_frame_budget(bytes: &[u8]) -> Result<(), EngineError> {
    let Some(header) = webp_anim::header(bytes) else {
        return Ok(());
    };
    check_webp_held(&header, header.frames.min(1))
}

/// Refuse a decode that keeps `frames` frames of the animated
/// WebP `header` describes when, with the two canvas-sized
/// buffers a decoder holds beside them (see
/// [`check_webp_frame_budget`]), it would hold more than
/// [`MAX_DECODE_ALLOC`]. A file with no frames allocates no
/// canvas, and its decode refuses it.
fn check_webp_held(header: &webp_anim::Header, frames: u32) -> Result<(), EngineError> {
    let canvas = u64::from(header.width) * u64::from(header.height);
    let held = u64::from(frames)
        .saturating_add(2)
        .saturating_mul(canvas)
        .saturating_mul(4);
    if frames > 0 && held > MAX_DECODE_ALLOC {
        return Err(EngineError::Decode(format!(
            "WebP decode of {frames} frames on a {}x{} canvas would hold {held} bytes, past the \
             {MAX_DECODE_ALLOC} byte decode allocation limit",
            header.width, header.height
        )));
    }
    Ok(())
}

/// The ANIM parameters of a WebP source: `(loop count,
/// background color as [R, G, B, A])`, or `None` when the
/// source carries no ANIM chunk (a still) or its container
/// does not parse; genuinely malformed files fail at decode
/// either way.
pub(in crate::engine) fn webp_animation_params(bytes: &[u8]) -> Option<(u16, [u8; 4])> {
    oxideav_webp::animation_params(bytes).ok().flatten()
}

/// Whether the source bytes are an animated WebP (they
/// carry an ANIM chunk). A container probe, not a decode:
/// the server uses it to route animated WebP renders to
/// their own gate (issue #231) before any engine work runs.
pub(crate) fn is_animated_webp(bytes: &[u8]) -> bool {
    webp_animation_params(bytes).is_some()
}

/// Whether the source bytes are an animated GIF: a GIF with
/// a second frame, which is what sends [`decode_gif_frames`]
/// output down the animated paths. A block walk, not a
/// decode: the gif crate (the parser the decoder itself
/// uses) reads frame descriptors with LZW decoding skipped,
/// and the walk stops at the second frame, so it costs one
/// pass over the first frame's compressed bytes and
/// allocates no pixels. The server uses it, beside
/// [`is_animated_webp`], to route animated WebP encodes from
/// GIF sources to their gate (issue #405) before any engine
/// work runs. A file that does not parse that far reads as
/// not animated; the decode then refuses it anyway, or
/// renders a still.
pub(crate) fn is_animated_gif(bytes: &[u8]) -> bool {
    if !bytes.starts_with(b"GIF") {
        return false;
    }
    let mut options = gif::DecodeOptions::new();
    options.skip_frame_decoding(true);
    let Ok(mut decoder) = options.read_info(bytes) else {
        return false;
    };
    for _ in 0..2 {
        if !matches!(decoder.next_frame_info(), Ok(Some(_))) {
            return false;
        }
    }
    true
}

/// The format an animated source renders to while it keeps
/// its frames (or goes to video): an explicit `format(...)`
/// first, then the negotiated preference, then the source's
/// own format. Negotiation never flattens an animation
/// (issue #370): a negotiated GIF or WebP keeps the frames
/// in that format, and any other negotiated format (AVIF and
/// JPEG XL cannot carry them) falls back to the source
/// format. An explicit format always wins; a still format
/// then converts the first frame, as documented for still
/// conversions. The engine's animated branches and the
/// server's animation gate predicate (issue #405) both
/// resolve through this one function, so the gate cannot
/// drift from the encoder it guards again.
pub(crate) fn animated_output_format(
    source: ImageFormat,
    explicit: Option<ImageFormat>,
    preferred: Option<ImageFormat>,
) -> ImageFormat {
    match (explicit, preferred) {
        (Some(format), _) => format,
        (None, Some(format @ (ImageFormat::Gif | ImageFormat::WebP))) => format,
        (None, _) => source,
    }
}

/// The formats an animation keeps its frames in (issue #480):
/// an animated source's own format, and the format its frames
/// render to on the animated path. The video formats
/// ([`crate::video::VideoFormat`]) take the frames too, on the
/// video path, and every other format takes the first frame as
/// a still.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::engine) enum AnimatedFormat {
    Gif,
    WebP,
}

impl AnimatedFormat {
    /// The animated format `format` names, or `None` for one
    /// that cannot keep frames. Every format is listed, so a
    /// new one has to say which it is.
    pub(in crate::engine) fn of(format: ImageFormat) -> Option<Self> {
        match format {
            ImageFormat::Gif => Some(AnimatedFormat::Gif),
            ImageFormat::WebP => Some(AnimatedFormat::WebP),
            ImageFormat::Jpeg
            | ImageFormat::Png
            | ImageFormat::Jxl
            | ImageFormat::Avif
            | ImageFormat::Mp4
            | ImageFormat::WebM => None,
        }
    }
}

impl From<AnimatedFormat> for ImageFormat {
    fn from(format: AnimatedFormat) -> Self {
        match format {
            AnimatedFormat::Gif => ImageFormat::Gif,
            AnimatedFormat::WebP => ImageFormat::WebP,
        }
    }
}

/// Decode all frames of an animated WebP source, under the
/// same limits as GIF. Every frame comes back composited onto
/// the full canvas the way the image crate's decoder
/// (image-webp) composites it, honoring sub-rectangle frames
/// and the blending flag; disposal of a frame to the
/// background color is honored only when the file's ANIM
/// background color has been set on that decoder first (it
/// otherwise leaves the disposed rectangle in place), so the
/// color is read from the container and set before any frame
/// decodes (see [`open_webp_animation`]).
///
/// The frames' bitstreams go through libwebp, which decodes
/// them to the same pixels about twice as fast (issue #646,
/// see [`webp_anim`]). The image crate's iterator still takes,
/// from the start, any file that walk does not handle, so an
/// unusual or broken file gets the frames or the error it
/// always got.
pub(in crate::engine) fn decode_webp_frames(
    bytes: &[u8],
    deadline: Option<std::time::Instant>,
) -> Result<Frames, EngineError> {
    use image::AnimationDecoder;
    // Charge the canvas and the frame count from the headers
    // before either decoder opens the file, as the GIF path does
    // (see [`check_webp_frame_budget`]). An over-budget canvas is
    // refused here, never handed to the image crate's iterator.
    check_webp_frame_budget(bytes)?;
    let (decoder, background) = open_webp_animation(bytes)?;
    if let Some(frames) = webp_frames_through_libwebp(bytes, background, deadline) {
        return frames;
    }
    collect_frames(decoder.into_frames(), "WebP", deadline)
}

/// An animated WebP the image crate's decoder has opened, and
/// the ANIM background color set on that decoder, if any.
type OpenedWebp<'a> = (
    image::codecs::webp::WebPDecoder<Cursor<&'a [u8]>>,
    Option<[u8; 4]>,
);

/// Open an animated WebP with the image crate's decoder: the
/// container checks and their errors, the decode limits, and
/// the ANIM background color. Returns the decoder and the
/// color it took, if any.
pub(in crate::engine) fn open_webp_animation(bytes: &[u8]) -> Result<OpenedWebp<'_>, EngineError> {
    use image::ImageDecoder;
    let mut decoder = image::codecs::webp::WebPDecoder::new(Cursor::new(bytes))
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    decoder
        .set_limits(decode_limits())
        .map_err(|e| EngineError::Decode(e.to_string()))?;
    let mut background = None;
    if let Some((_, color)) = webp_animation_params(bytes) {
        // Losing the fill on a decoder refusal only costs the
        // disposal behavior; the frames still decode.
        if decoder.set_background_color(image::Rgba(color)).is_ok() {
            background = Some(color);
        }
    }
    Ok((decoder, background))
}

/// The frames of an animated WebP that [`open_webp_animation`]
/// has opened, each bitstream decoded by libwebp and composited
/// by [`webp_anim`], under the limits and the deadline
/// [`collect_frames`] holds every animation to. `None` when the
/// walk met something it does not handle, at any frame: the
/// frames it had are dropped and the image crate's iterator
/// decides.
pub(in crate::engine) fn webp_frames_through_libwebp(
    bytes: &[u8],
    background: Option<[u8; 4]>,
    deadline: Option<std::time::Instant>,
) -> Option<Result<Frames, EngineError>> {
    let mut animation = webp_anim::Animation::open(bytes, background)?;
    let mut unsupported = false;
    let frames = std::iter::from_fn(|| match animation.next_frame() {
        Ok(frame) => Some(Ok(frame)),
        Err(webp_anim::Stop::Done) => None,
        Err(webp_anim::Stop::Unsupported) => {
            unsupported = true;
            None
        }
    });
    let collected = collect_frames(frames, "WebP", deadline);
    (!unsupported).then_some(collected)
}

/// Encode frames as an animated WebP through libwebp's
/// animated encoder (the `webp` crate's `AnimEncoder`, method
/// 4 by default; issue #323). The pure-Rust alternative
/// (oxideav-webp) writes lossless VP8L frames only and has no lossy
/// animation mode, so animated output through it has no
/// quality lever, and a cold encode of a real animation ran past the
/// 600 s processing deadline on the production soak host
/// (issue #310). The quality mapping matches the still path
/// in `encode::encode_webp`: the default quality is 80, quality
/// below 100 is a lossy encode at that quality, and quality
/// 100 is a lossless encode through the same libwebp encoder.
/// The lossless corner stays on libwebp because its animated
/// lossless encode is faster than the oxideav lossless
/// encode, not slower: on the soak site files the oxideav
/// encode did not finish inside the 600 s deadline at all,
/// while libwebp lossless took 23.6 s for the 40 frame mascot
/// and 148.5 s for the 120 frame snap-toy (measured in the
/// bench recorded on issue #323). That is the opposite of the
/// roughly 92x gap the still lossless path measured in issue
/// #292, so no second encoder stays wired in for one corner.
/// The swap follows the #292 pattern and is a temporary
/// measure: the animated encode returns to oxideav-webp once
/// a release closes the gap to roughly 1.3x of libwebp method
/// 4 at equal or better size and quality.
/// `loop_count` is carried over from the source (0 loops
/// forever) and set on the muxer, which is where the ANIM
/// loop count lives. Frames arrive already composited onto
/// the full canvas by the decode side, so each one is handed
/// to the encoder as a full-canvas RGBA frame and alpha is
/// preserved.
/// Frame timestamps are the accumulated start times of each
/// frame, with one adjustment: the timeline starts at minus
/// the total duration instead of at zero. The `webp` crate
/// finishes an animation by adding a null frame at timestamp
/// 0, and libwebp takes the last frame's duration from that
/// final timestamp; starting below zero makes the final call
/// land exactly on the end of the last frame through the
/// uint32 wrap-around libwebp's timestamp arithmetic allows,
/// so the last frame keeps its own duration instead of being
/// assigned the average of the frames before it. A total
/// duration beyond the i32 timestamp range (an animation
/// longer than about 24 days) falls back to a zero-based
/// timeline and that average.
pub(in crate::engine) fn encode_animated_webp(
    frames: &Frames,
    loop_count: u16,
    quality: Option<u8>,
    format_options: FormatOptions,
    deadline: Option<std::time::Instant>,
) -> Result<Vec<u8>, EngineError> {
    let (width, height) = frames.dimensions();
    // VP8 frames carry 14-bit dimensions, as on the still path.
    if width > 16383 || height > 16383 {
        return Err(EngineError::Encode(
            "webp: dimension exceeds the 16383 format limit".into(),
        ));
    }
    let q = quality.unwrap_or(80);
    let mut config = webp::WebPConfig::new().map_err(|_| {
        EngineError::Encode("webp: libwebp rejected its default configuration".into())
    })?;
    config.quality = q as f32;
    config.method = format_options.webp_method.map(i32::from).unwrap_or(4);
    if let Some(sns) = format_options.webp_sns {
        config.sns_strength = i32::from(sns);
    }
    if q >= 100 {
        config.lossless = 1;
    }
    let delays_ms: Vec<i64> = frames
        .iter()
        .map(|frame| {
            let delay: std::time::Duration = frame.delay().into();
            i64::try_from(delay.as_millis()).unwrap_or(i64::MAX)
        })
        .collect();
    let total_ms: i64 = delays_ms.iter().fold(0, |acc, d| acc.saturating_add(*d));
    let mut timestamp_ms: i64 = if total_ms <= i64::from(i32::MAX) {
        -total_ms
    } else {
        0
    };
    // libwebp's animation options (`minimize_size`,
    // `allow_mixed`, `kmin`/`kmax`) stay at their defaults
    // (issue #340). The `webp` crate fills them in inside
    // `try_encode` and exposes none of them, and reaching
    // them needs unsafe FFI, which `#![forbid(unsafe_code)]`
    // rules out. Measured through libwebp-sys,
    // `minimize_size` is the one that matters: 13.0% fewer bytes (75,826
    // against 87,164) and +0.21 dB, for 46% more encode time
    // (1,031 ms against 708 ms). `allow_mixed` left photo
    // content byte-identical, kmax 17 with kmin 9 saved 1.2%,
    // and both flags together took 10.2x the time of
    // `minimize_size` alone for the same bytes. That saving
    // is not worth a new unsafe FFI surface plus a fork of
    // the crate's assemble path, so the defaults stay.
    let mut encoder = webp::AnimEncoder::new(width, height, &config);
    encoder.set_loop_count(i32::from(loop_count));
    for (frame, delay_ms) in frames.iter().zip(&delays_ms) {
        // The deadline is checked per frame while the
        // frames are handed over (issue #322). The final
        // `try_encode` below is a single libwebp call with
        // no cancellation hook, so it remains one opaque
        // stage: it cannot be interrupted mid-call, and
        // the server side bounds what it can hold (see
        // `run_process` in server/process.rs).
        state::check_deadline(deadline)?;
        let buffer = frame.buffer();
        if buffer.width() != width || buffer.height() != height {
            return Err(EngineError::Encode(
                "webp: animated frames do not share one canvas".into(),
            ));
        }
        encoder.add_frame(webp::AnimFrame::from_rgba(
            buffer.as_raw(),
            width,
            height,
            i32::try_from(timestamp_ms).unwrap_or(i32::MAX),
        ));
        timestamp_ms = timestamp_ms.saturating_add(*delay_ms);
    }
    let encoded = encoder
        .try_encode()
        .map_err(|e| EngineError::Encode(format!("webp: {e:?}")))?;
    Ok(encoded.to_vec())
}

/// One shared smart crop window: the chain position and
/// frame dimensions it was decided on, and the window
/// itself as (x, y, width, height).
#[derive(Debug)]
struct SmartWindow {
    op_index: usize,
    width: u32,
    height: u32,
    window: (u32, u32, u32, u32),
}

/// Smart crop windows shared across the frames of one
/// animated render (issue #376): the first frame's smart
/// crop decides its window, and every later frame replays
/// that window instead of running the focal analysis
/// again, so the crop cannot jitter frame to frame and
/// the analysis behind it (saliency, plus face detection
/// when a provider feeds it) runs once per render, not
/// once per frame. Entries key on the crop's position in
/// the chain and the frame dimensions it decided on, so
/// a frame of a different size decides for itself.
#[derive(Debug, Default)]
pub(in crate::engine) struct SmartWindows {
    entries: Vec<SmartWindow>,
}

impl SmartWindows {
    pub(in crate::engine) fn get(&self, key: (usize, u32, u32)) -> Option<(u32, u32, u32, u32)> {
        self.entries
            .iter()
            .find(|e| (e.op_index, e.width, e.height) == key)
            .map(|e| e.window)
    }

    /// A smart crop in an animated render sharing windows
    /// across frames (issue #376): the first frame decides
    /// through the traced crop (the trace reports the very
    /// window the crop used) and later frames replay that
    /// window verbatim, so the analysis runs once and the
    /// framing cannot jitter. `op_index` is the crop's
    /// position in the chain. A replayed window is handed
    /// back uncut, for the resize that closes the crop to
    /// read from the frame (issue #654).
    pub(in crate::engine) fn crop(
        &mut self,
        img: DynamicImage,
        (tw, th): (u32, u32),
        op_index: usize,
        align: &Align,
        ctx: &ProcessContext<'_>,
    ) -> crate::crop::Uncut {
        let key = (op_index, img.width(), img.height());
        if let Some(window) = self.get(key) {
            crate::crop::Uncut {
                img,
                window: Some(window),
            }
        } else {
            let (cropped, trace) = crate::crop::crop_to_aspect_traced(
                img,
                tw,
                th,
                align.clone(),
                ctx.face_provider,
                ctx.focal_provider,
            );
            if let Some(trace) = trace {
                self.set(key, trace.window);
            }
            crate::crop::Uncut::whole(cropped)
        }
    }

    fn set(&mut self, key: (usize, u32, u32), window: (u32, u32, u32, u32)) {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|e| (e.op_index, e.width, e.height) == key)
        {
            entry.window = window;
        } else {
            self.entries.push(SmartWindow {
                op_index: key.0,
                width: key.1,
                height: key.2,
                window,
            });
        }
    }
}

/// One trim's shared window: the trim's position in the
/// chain (operation and filter indexes), the frame
/// dimensions it was surveyed at, and the union of the boxes
/// the frames' own tolerance passes keep; `None` when every
/// frame was uniform.
#[derive(Debug)]
pub(in crate::engine) struct TrimWindow {
    key: (usize, usize),
    pub(in crate::engine) dims: (u32, u32),
    pub(in crate::engine) found: Option<Window>,
}

/// Trim windows shared across the frames of one animated
/// render (issue #465). Trim cuts a box that depends on the
/// content, so frames trimmed one at a time came out at
/// different sizes: the GIF screen took the first frame's
/// size and misplaced or clipped the rest, and the animated
/// WebP and video encoders refused the frames. Each window
/// is settled by [`survey_trims`] before any frame renders,
/// as the union of every frame's own box, so one cut holds
/// every frame's content and the frames stay registered on
/// one canvas. Entries key on the trim's chain position and
/// the frame dimensions, so a frame of a different size
/// trims for itself.
#[derive(Debug, Default)]
pub(in crate::engine) struct TrimWindows {
    pub(in crate::engine) entries: Vec<TrimWindow>,
}

impl TrimWindows {
    /// The settled box for the trim at `key` on a frame of
    /// `dims`, or `None` when no window is settled for it.
    pub(in crate::engine) fn get(
        &self,
        key: (usize, usize),
        dims: (u32, u32),
    ) -> Option<Option<Window>> {
        self.entries
            .iter()
            .find(|e| e.key == key && e.dims == dims)
            .map(|e| e.found)
    }

    fn settle(&mut self, key: (usize, usize), dims: (u32, u32), found: Option<Window>) {
        self.entries.retain(|e| e.key != key);
        self.entries.push(TrimWindow { key, dims, found });
    }
}

/// The geometry the frames of one animated render share, so
/// the frames come out on one canvas, cut from the same part
/// of the scene, and the scaled overlay parts they all draw.
#[derive(Debug, Default)]
pub(in crate::engine) struct FrameShare {
    /// One smart crop window per crop (issue #376), on the
    /// animated image and video paths alike: the video path
    /// let each frame decide its own until issue #713.
    pub(in crate::engine) smart_windows: SmartWindows,
    /// One trim window per trim filter (issue #465).
    pub(in crate::engine) trims: TrimWindows,
    /// The parts `overlay(...,width)` scaled, kept for the
    /// frames after the one that scaled them.
    pub(in crate::engine) overlays: ScaledOverlays,
}

/// The scaled overlay parts the frames of one animated render
/// share. `overlay(...,width)` scales the part of the overlay
/// that lands on the image, and those pixels depend on the
/// overlay, the size it is scaled to and the part that lands,
/// nothing else. The frames of a render share one canvas, so
/// they all ask for the same part: the first frame scales it,
/// and the frames after it draw the kept pixels, where each
/// scaled them again. A part is kept under all three, so a
/// frame of another size, or another overlay, asks for a part
/// of its own and scales it. The overlay is named by the URL
/// it was fetched from, which names one image for the whole
/// render.
///
/// The frames render in turn, each chain borrowing the render's
/// one share for its run (see [`render_frames`] and
/// [`survey_trims`]), so the parts are never read and written at
/// once. They hold at most as many pixels as the image they land
/// on: a part past that is scaled for each frame, as before, so
/// a chain of many scaled overlays keeps no more than one
/// frame's worth of them.
#[derive(Debug, Default)]
pub(in crate::engine) struct ScaledOverlays {
    entries: Vec<ScaledOverlay>,
}

/// One kept part: what its pixels depend on, and the pixels.
#[derive(Debug)]
struct ScaledOverlay {
    url: String,
    size: (u32, u32),
    part: crate::filters::Rect,
    pixels: std::sync::Arc<image::RgbaImage>,
}

impl ScaledOverlays {
    /// The pixels of `part` of the overlay at `url` scaled to
    /// `size`: the kept ones when an earlier frame scaled them,
    /// or else what `scale` makes, kept while the kept parts
    /// still hold no more than `room` pixels with them.
    pub(in crate::engine) fn part<E>(
        &mut self,
        url: &str,
        size: (u32, u32),
        part: crate::filters::Rect,
        room: u64,
        scale: impl FnOnce() -> Result<image::RgbaImage, E>,
    ) -> Result<std::sync::Arc<image::RgbaImage>, E> {
        let pixels_of = |(_, _, w, h): crate::filters::Rect| u64::from(w) * u64::from(h);
        if let Some(kept) = self
            .entries
            .iter()
            .find(|e| e.url == url && e.size == size && e.part == part)
        {
            return Ok(std::sync::Arc::clone(&kept.pixels));
        }
        let pixels = std::sync::Arc::new(scale()?);
        let held: u64 = self.entries.iter().map(|e| pixels_of(e.part)).sum();
        if held + pixels_of(part) <= room {
            self.entries.push(ScaledOverlay {
                url: url.to_owned(),
                size,
                part,
                pixels: std::sync::Arc::clone(&pixels),
            });
        }
        Ok(pixels)
    }
}

/// The smallest box holding both boxes; `None` stands for no
/// box at all.
fn union_box(a: Option<Window>, b: Option<Window>) -> Option<Window> {
    match (a, b) {
        (Some((ax, ay, aw, ah)), Some((bx, by, bw, bh))) => {
            let (left, top) = (ax.min(bx), ay.min(by));
            let right = (ax + aw).max(bx + bw);
            let bottom = (ay + ah).max(by + bh);
            Some((left, top, right - left, bottom - top))
        }
        (one, None) | (None, one) => one,
    }
}

/// Settle the trim windows of an animated render (issue
/// #465) before its frames render: for each trim filter in
/// chain order, run every frame up to that filter (the trims
/// before it replaying their settled windows, so the survey
/// sees each frame as the render will) and record the union
/// of the boxes the frames' own tolerance passes keep. The
/// cost is one extra chain prefix per trim per frame, under
/// the render's own deadline. A lone frame has nothing to
/// share and trims for itself.
///
/// The render needs the frames as they came, and a prefix's
/// steps work on the image they are handed (a resize
/// premultiplies its own frame where it sits), so each frame
/// is lent to its prefix and comes back unchanged (issue
/// #737). A prefix that starts with a resize or a fit-in that
/// resamples the whole frame reads the lent frame where it is
/// (see [`ChainState::run_to`]); any other prefix runs on a
/// copy of the frame. A trim nothing runs before reads the
/// frames where they are (issue #713): its survey is the
/// tolerance pass alone.
pub(in crate::engine) fn survey_trims(
    frames: &mut Frames,
    ops: &[Operation],
    cap: Option<(u32, u32)>,
    dpr: f32,
    ctx: &ProcessContext<'_>,
    directives: &crate::url::RequestDirectives,
    share: &mut FrameShare,
) -> Result<(), EngineError> {
    if frames.len() < 2 {
        return Ok(());
    }
    check_operation_count(ops)?;
    for (op_index, op) in ops.iter().enumerate() {
        let Operation::Filters(filters) = op else {
            continue;
        };
        for (filter_index, filter) in filters.iter().enumerate() {
            let tolerance = match filter {
                Filter::Plain(
                    PlainFilter::Trim(tolerance) | PlainFilter::TrimSides(tolerance, _),
                ) => *tolerance,
                _ => continue,
            };
            let key = (op_index, filter_index);
            let first_step = op_index == 0 && filter_index == 0;
            // The first frame's dimensions and the union so far.
            let mut surveyed: Option<((u32, u32), Option<Window>)> = None;
            for frame in frames.iter_mut() {
                ctx.check_deadline()?;
                let (dims, found) = if first_step {
                    let rgba = frame.buffer();
                    (
                        rgba.dimensions(),
                        crate::filters::trim_box_rgba(rgba, tolerance),
                    )
                } else {
                    lend_frame(frame, |lent| {
                        let mut chain =
                            ChainState::new(ops, cap, dpr, ctx, directives, Some(&mut *share));
                        let img = chain.run_to(lent, ops, key)?;
                        Ok::<_, EngineError>((
                            (img.width(), img.height()),
                            crate::filters::trim_box(&img, tolerance),
                        ))
                    })?
                };
                match &mut surveyed {
                    None => surveyed = Some((dims, found)),
                    Some((first, union)) if *first == dims => *union = union_box(*union, found),
                    // Dimensions at a trim follow from the
                    // chain alone, so frames agree on them;
                    // one that does not trims for itself.
                    Some(_) => {}
                }
            }
            if let Some((dims, found)) = surveyed {
                share.trims.settle(key, dims, found);
            }
        }
    }
    Ok(())
}
