//! Video frame extraction (issue #214): a `frame:`
//! segment turns a video the server fetched into an
//! image source by pulling one frame out of it.
//!
//! Decoding a video container (demux, seek, decode) is
//! a job for a real video tool, so extraction follows
//! the pattern the encode side set with
//! `CRABCROP_VIDEO_ENCODER`: an operator-configured
//! external program (FFmpeg is the reference) runs as a
//! bounded subprocess, never a shell, with the source
//! bytes on stdin, an empty environment, its own process
//! group (killed whole on timeout), a wall-clock limit,
//! and a cap on what it may write back. Stdin is a
//! seekable file, not a pipe (issue #411): the bytes are
//! staged once in an unlinked temporary file, because an
//! MP4 whose index (the moov box) follows its samples,
//! FFmpeg's own default layout, can only be read by
//! seeking back to the samples after the index. The tool
//! reads it through the `fd` protocol (FFmpeg 6.0 and
//! later), the one input protocol that seeks on a regular
//! file without naming a path. The argv contract is fixed
//! and built entirely here:
//!
//! - Probe: `-nostdin -v verbose -protocol_whitelist fd
//!   -format_whitelist <containers> -threads 2
//!   -max_pixels <bound> -codec_whitelist <decoders>
//!   -dump_separator <token> -i fd:`. No output target is
//!   named, so the tool prints its input report (duration,
//!   stream list) to stderr and exits nonzero; the probe
//!   reads its facts from that report and treats the exit
//!   status as meaningless. At `verbose` the report gives
//!   a stream's coded size too, in parentheses after its
//!   displayed size, where the two differ. The report also
//!   prints the source's own metadata, so the token, random
//!   per probe, marks the real stream lines (see
//!   [`parse_probe`]). A source with no video stream in the
//!   report (a non-video, an audio-only file, a corrupt
//!   container) is [`FrameError::NotVideo`].
//! - Extract: `-nostdin -v info -protocol_whitelist fd
//!   -format_whitelist <containers> -filter_threads 2
//!   -threads <n> -max_pixels <bound> -codec_whitelist
//!   <decoders> -i fd: -ss <t> -frames:v 1 -an -sn -dn
//!   -threads 2 -f image2pipe -vcodec png pipe:1`. The
//!   seek is output-side, which is the accurate form:
//!   the tool decodes forward and returns the first frame
//!   at or after `t`, and nothing when `t` is past the
//!   last frame's timestamp. That is why resolved
//!   timestamps clamp to `duration - 1/fps` and why an
//!   empty result steps back toward the tail (see
//!   [`VideoFrameExtractor::extract_frame`]): a request
//!   past the end clamps to the last frame, as the
//!   grammar promises.
//!
//! The whitelists keep the tool inside the loader's
//! policy (issue #361): the bytes on stdin are the only
//! input it can reach (`fd` opens a descriptor the tool
//! already holds, and refuses a descriptor named in a URL),
//! so container indirection (a manifest naming a `file://`
//! or network target) has nowhere to go.
//!
//! The decoder bounds hold the tool to the decode budget
//! the extracted PNG has to pass anyway (issue #605 for the
//! decoder list). The probe reads two sizes: the picture a
//! stream displays, and its coded frame, the one its decoder
//! allocates, which can be larger than the picture. A
//! displayed size past the budget, or a coded size past the
//! decoder's pixel bound, is refused before any extraction
//! runs. `-max_pixels` makes the decoder refuse a larger
//! output picture wherever it appears (a stream can change
//! size after the frames the probe reads), the thread counts
//! keep the frames a decoder holds in flight from scaling
//! with the host's cores for large frames, and
//! `-codec_whitelist` lets the tool open only the decoders
//! of the video codecs it names. On Linux the kernel caps
//! the tool's address space at 3 GiB behind all of that:
//! anything the checks do not size, the probe's own decode
//! included, is bounded by that cap, and the tool is the OOM
//! killer's first choice.
//!
//! The extracted PNG re-enters the image pipeline as
//! the source, so every downstream budget (decode
//! dimensions and allocation, output pixels) applies to
//! it unchanged.

use std::fs::File;
use std::io::{Seek, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::toolrun::{ToolCommand, ToolError, ToolInput, ToolRun, run_tool, tool_cpu_limit};
use crate::url::FrameSelection;

/// Cap on the extracted frame's PNG bytes. Tied to the
/// engine's decode allocation budget (256 MiB): a frame
/// whose PNG passes this can never survive the decode
/// budgets, so anything larger is a broken or hostile
/// tool, not a frame.
pub const MAX_FRAME_PNG_BYTES: usize = 256 * 1024 * 1024;

/// The input formats the extractor accepts (issue #361):
/// the container demuxers a video source can legitimately
/// arrive through. Manifest formats (dash, hls) are absent
/// on purpose: their whole job is to point at further
/// inputs, and the only input this tool may consume is
/// the byte stream the loader already validated, which is
/// also why the protocol whitelist names only `fd`.
const FORMAT_WHITELIST: &str = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpeg,mpegts,flv,asf,ogg";

/// The decoders the tool may open (issue #605), by FFmpeg
/// decoder name: H.264 (`h264`, or `libopenh264` in builds
/// without the native decoder), HEVC, VP8 and VP9 (native or
/// libvpx), AV1 (dav1d, libaom, or the native decoder),
/// MPEG-4 Part 2, H.263 (the 3GP phone codec), MPEG-1 and
/// MPEG-2 video, Theora, Motion JPEG, ProRes, and DV: the
/// video codecs the accepted containers carry from the web,
/// broadcast, phones, and cameras. A stream in any other
/// codec has no decoder the tool may open, so it yields no
/// frame: Windows Media (WMV, VC-1) in ASF, Sorenson H.263
/// and VP6 in FLV, and lossless, raw, and screen codecs
/// among them. ASF stays an accepted container, as it can
/// carry a listed codec. Audio and subtitle decoders are
/// absent too: the extraction maps no such stream
/// (`-an -sn -dn`), and the probe only logs that it cannot
/// open them.
const CODEC_WHITELIST: &str = "h264,libopenh264,hevc,vp8,libvpx,vp9,libvpx-vp9,av1,libdav1d,\
libaom-av1,mpeg4,h263,mpeg1video,mpeg2video,theora,mjpeg,prores,dvvideo";

/// The most pixels a video frame may hold: the engine's
/// decode allocation budget at four bytes a pixel
/// ([`crate::engine::MAX_DECODE_ALLOC`], 64 Mi pixels), the
/// budget the extracted PNG has to pass anyway. A probed
/// frame size past it, or past
/// [`crate::engine::MAX_DECODE_DIMENSION`] on a side, is
/// refused before any extraction runs.
const MAX_FRAME_PIXELS: u64 = crate::engine::MAX_DECODE_ALLOC / 4;

/// The pixel bound the tool's decoders get as `-max_pixels`:
/// the frame budget, plus 63 pixels a row for the tallest
/// frame the budget allows. FFmpeg checks a picture's width
/// rounded up to its buffer alignment (up to 64 pixels)
/// times its height, so a frame within the budget whose
/// width is not a multiple of 64, such as 16376x4098, would
/// be refused at the exact budget. The probed displayed size
/// is checked against the exact budget, and so is the
/// extracted PNG; a frame the probe did not see can pass it
/// by the alignment room, at most 1.6%. The probed coded
/// size is checked against this bound: a codec pads its
/// coded frame to whole blocks (1920x1080 H.264 is coded
/// 1920x1088), and this room holds that padding for frames
/// within the budget.
const DECODER_MAX_PIXELS: u64 = MAX_FRAME_PIXELS + 63 * crate::engine::MAX_DECODE_DIMENSION as u64;

/// The threads the probe's decoder, the extraction's filters
/// and PNG encoder, and the extraction's decoder for a frame
/// past [`MANY_THREADS_MAX_PIXELS`] get. A decoder holds
/// about one frame in flight per thread, so for large frames
/// FFmpeg's default of one thread per core made a run's
/// memory grow with the host: an 8K 10-bit HEVC extraction
/// peaked at 2.6 GB with the default on a 4-core host, and
/// 1.9 GB on two threads.
const FEW_THREADS: usize = 2;

/// The largest frame the extraction's decoder runs on
/// FFmpeg's default count of threads for, DCI 4K
/// (4096x2160). The extraction seeks on its output side, so
/// it decodes every frame up to the timestamp, and a far
/// seek in a long clip is as fast as the decoder's threads
/// make it: on a 4-core host, a seek 19.5 s into a 4K H.264
/// clip took 8.7 to 13.2 s on two threads and 6.2 to 7.4 s
/// on the default five, and the gap grows with the cores.
/// Frames up to this size hold little in flight: the same
/// seek peaked at 0.45 GB on five threads, and under 1 GB on
/// sixteen.
const MANY_THREADS_MAX_PIXELS: u64 = 4096 * 2160;

/// The most threads the extraction's decoder gets, FFmpeg's
/// own ceiling for its default count.
const MAX_DECODER_THREADS: usize = 16;

/// The tool's address-space cap on Linux, 3 GiB. An
/// extraction at the frame budget needs about 1.7 GiB of
/// address space (an 8192x8192 H.264 frame), and an 8K
/// 10-bit HEVC frame, the heaviest common source, about
/// 2.4 GiB, most of it decoder frames and the reservations
/// of the tool's threads; the cap leaves headroom over that
/// and turns any larger allocation into a tool error.
const DECODER_MAX_ADDRESS_SPACE: u64 = 3 << 30;

/// What FFmpeg logs when a decoder refuses a frame past
/// `-max_pixels`. A run whose report says so and that
/// produced no frame met the frame budget, which makes the
/// source a bad one, not the tool a broken one.
const PIXEL_BOUND_REPORT: &str = "exceeds specified max pixel count";

/// How much of a tool's stderr report the probe keeps.
/// The input report sits at the head of the output, and a
/// real one is a few kilobytes; this is generous headroom
/// for chatty builds and metadata-heavy sources. A probe
/// report past the cap is refused, not parsed (issue
/// #402): source metadata could have pushed the real facts
/// past the cut and left only planted lines in view.
const PROBE_REPORT_CAP: usize = 256 * 1024;

/// What a probe learned about a video source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoProbe {
    /// Container duration in seconds, when the tool
    /// reports one (some containers do not).
    pub duration: Option<f64>,
    /// The first video stream's displayed width.
    pub width: u32,
    /// The first video stream's displayed height.
    pub height: u32,
    /// The first video stream's coded width: that of the
    /// frame its decoder allocates, which can be wider than
    /// the displayed picture (block padding, or a crop). At
    /// least `width`; `width` when the report names no other.
    pub coded_width: u32,
    /// The first video stream's coded height, at least
    /// `height`, as `coded_width` is.
    pub coded_height: u32,
    /// The first video stream's frame rate, when the
    /// report names one.
    pub fps: Option<f64>,
}

/// One extracted frame and the facts meta reports.
#[derive(Debug)]
pub struct ExtractedFrame {
    /// The frame as PNG bytes, ready to decode as the
    /// source image.
    pub png: Vec<u8>,
    /// The probed duration, for meta.
    pub duration: Option<f64>,
    /// The video stream's displayed width, for meta.
    pub width: u32,
    /// The video stream's displayed height, for meta.
    pub height: u32,
    /// The resolved timestamp the frame was selected at,
    /// in seconds (after clamping), for meta.
    pub timestamp: f64,
}

/// Why frame extraction failed. The class decides the
/// response: a source that is not a usable video fails
/// the way bad sources already fail, a tool that cannot
/// do its job is a server-side failure, and a selection
/// the source cannot answer is the request's problem.
#[derive(Debug)]
pub enum FrameError {
    /// The probe found no video stream: the source is
    /// not a video this tool can read (a non-video, an
    /// audio-only file, a corrupt container).
    NotVideo(String),
    /// The selection cannot be resolved against this
    /// source (a percentage of a duration the container
    /// does not report).
    Unsupported(String),
    /// The tool itself failed: it could not be spawned,
    /// timed out, exited nonzero, produced no frame, or
    /// produced output past the cap.
    Failed(String),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::NotVideo(why) => write!(f, "{why}"),
            FrameError::Unsupported(why) => write!(f, "{why}"),
            FrameError::Failed(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for FrameError {}

/// A configured frame extractor: an external decoder
/// program plus the bounds every run obeys. Cheap to
/// share behind an `Arc`; each extraction spawns fresh
/// processes.
#[derive(Debug)]
pub struct VideoFrameExtractor {
    program: PathBuf,
    timeout: Duration,
}

impl VideoFrameExtractor {
    /// Validate the program path the way the video
    /// encoder's is validated: it must exist and be
    /// executable, so a misconfiguration fails at boot,
    /// not on the first `frame:` request.
    pub fn new(program: PathBuf, timeout: Duration) -> Result<Self, FrameError> {
        if !program.is_file() {
            return Err(FrameError::Failed(format!(
                "video decoder {} does not exist or is not a file",
                program.display()
            )));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&program)
                .map(|m| m.permissions().mode())
                .unwrap_or(0);
            if mode & 0o111 == 0 {
                return Err(FrameError::Failed(format!(
                    "video decoder {} is not executable",
                    program.display()
                )));
            }
        }
        Ok(Self { program, timeout })
    }

    /// Probe the source: duration, displayed and coded
    /// sizes, and frame rate of its first video stream.
    /// `NotVideo` when the report names no usable video
    /// stream. The extraction path probes through
    /// `probe_until` (the shared deadline); this spelling
    /// stays as the single-probe entry the test suite
    /// exercises.
    #[allow(dead_code)]
    pub fn probe(&self, video: &[u8]) -> Result<VideoProbe, FrameError> {
        let input = stage_input(video)?;
        self.probe_until(&input, Instant::now() + self.timeout)
    }

    /// The probe under a caller-supplied deadline, so a
    /// stacked sequence (probe, then extraction attempts)
    /// shares one budget instead of a fresh timeout per
    /// invocation (issue #362).
    fn probe_until(&self, input: &File, deadline: Instant) -> Result<VideoProbe, FrameError> {
        let separator = probe_separator();
        let threads = FEW_THREADS.to_string();
        let max_pixels = DECODER_MAX_PIXELS.to_string();
        let run = self.run(
            &[
                "-nostdin",
                "-v",
                "verbose",
                "-protocol_whitelist",
                "fd",
                "-format_whitelist",
                FORMAT_WHITELIST,
                "-threads",
                &threads,
                "-max_pixels",
                &max_pixels,
                "-codec_whitelist",
                CODEC_WHITELIST,
                "-dump_separator",
                &separator,
                "-i",
                "fd:",
            ],
            input,
            0,
            deadline,
        )?;
        if run.stderr_truncated {
            return Err(FrameError::NotVideo(format!(
                "the decoder's report on the source passed the {PROBE_REPORT_CAP} byte cap, \
                 so its facts cannot be checked"
            )));
        }
        parse_probe(&run.stderr, &separator).ok_or_else(|| {
            if run.stderr.contains("Protocol not found") {
                // FFmpeg before 6.0 has no `fd` protocol: say
                // so, rather than blame the source.
                return FrameError::Failed(format!(
                    "the video decoder cannot read the fd protocol \
                     (frame extraction needs FFmpeg 6.0 or newer): {}",
                    run.stderr_excerpt()
                ));
            }
            if met_the_pixel_bound(&run) {
                // The decoder refused the stream's frame size,
                // so the report may carry no size to check.
                return frame_past_the_budget();
            }
            FrameError::NotVideo(
                "the source has no video stream the decoder can read \
                 (not a video, audio only, or a corrupt container)"
                    .to_string(),
            )
        })
    }

    /// Extract the selected frame as PNG bytes. Probes
    /// first, resolves the selection against the probe,
    /// then seeks. A seek that lands past the last frame
    /// (a variable-rate tail, a rounded duration) comes
    /// back empty; the selection steps back one frame
    /// interval at a time, a few times, so a request past
    /// the end still lands on the last frame instead of
    /// failing. The returned timestamp is the one actually
    /// sought.
    pub fn extract_frame(
        &self,
        video: &[u8],
        selection: FrameSelection,
    ) -> Result<ExtractedFrame, FrameError> {
        // One deadline for the whole extraction (issue
        // #362): the probe and every step-back attempt draw
        // from the same budget, instead of each invocation
        // restarting the timeout.
        let deadline = Instant::now() + self.timeout;
        let input = stage_input(video)?;
        let probe = self.probe_until(&input, deadline)?;
        // A frame the decode budget refuses is refused here,
        // before the extraction decodes it.
        check_frame_size(&probe)?;
        let mut timestamp = resolve_timestamp(selection, &probe)?;
        let step = frame_interval(probe.fps);
        let threads = decoder_threads(probe.coded_width, probe.coded_height);
        for attempt in 0..=3 {
            let png = self.extract_at(&input, timestamp, threads, deadline)?;
            if !png.is_empty() {
                if !png.starts_with(b"\x89PNG\r\n\x1a\n") {
                    return Err(FrameError::Failed(
                        "the decoder's frame output is not a PNG".to_string(),
                    ));
                }
                return Ok(ExtractedFrame {
                    png,
                    duration: probe.duration,
                    width: probe.width,
                    height: probe.height,
                    timestamp,
                });
            }
            if timestamp <= 0.0 || attempt == 3 {
                break;
            }
            timestamp = (timestamp - step).max(0.0);
        }
        Err(FrameError::Failed(format!(
            "the decoder produced no frame at {}s in a {}s video",
            format_seconds(timestamp),
            probe
                .duration
                .map(format_seconds)
                .unwrap_or_else(|| "unknown-length".to_string())
        )))
    }

    /// One extraction run at an already-resolved
    /// timestamp. An empty `Ok` means the tool ran fine
    /// and there is no frame at that time (the caller
    /// decides whether to step back). No frame from a run
    /// whose decoder refused a frame past the budget is
    /// `NotVideo`, whatever the exit status: an earlier
    /// timestamp would only answer with a different frame
    /// than the one asked for. Any other failure is
    /// `Failed`. The decoder runs on `decoder_threads`
    /// threads (see [`decoder_threads`]).
    fn extract_at(
        &self,
        input: &File,
        timestamp: f64,
        decoder_threads: usize,
        deadline: Instant,
    ) -> Result<Vec<u8>, FrameError> {
        let seek = format_seconds(timestamp);
        let few = FEW_THREADS.to_string();
        let decoder = decoder_threads.to_string();
        let max_pixels = DECODER_MAX_PIXELS.to_string();
        let run = self.run(
            &[
                "-nostdin",
                "-v",
                "info",
                "-protocol_whitelist",
                "fd",
                "-format_whitelist",
                FORMAT_WHITELIST,
                "-filter_threads",
                &few,
                "-threads",
                &decoder,
                "-max_pixels",
                &max_pixels,
                "-codec_whitelist",
                CODEC_WHITELIST,
                "-i",
                "fd:",
                "-ss",
                &seek,
                "-frames:v",
                "1",
                "-an",
                "-sn",
                "-dn",
                "-threads",
                &few,
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ],
            input,
            MAX_FRAME_PNG_BYTES,
            deadline,
        )?;
        if run.stdout.is_empty() && met_the_pixel_bound(&run) {
            return Err(frame_past_the_budget());
        }
        if !run.status.success() {
            return Err(FrameError::Failed(format!(
                "video decoder exited with {}: {}",
                run.status,
                run.stderr_excerpt()
            )));
        }
        Ok(run.stdout)
    }

    /// Run the program once through the shared runner
    /// ([`crate::toolrun::run_tool`]): the staged source on
    /// stdin, read from its start (every run shares the
    /// file's offset), stdout capped at `max_stdout`, and the
    /// probe report kept up to its cap.
    fn run(
        &self,
        args: &[&str],
        input: &File,
        max_stdout: usize,
        deadline: Instant,
    ) -> Result<ToolRun, FrameError> {
        let staging_failed = |e: std::io::Error| {
            FrameError::Failed(format!("staging the video for the decoder failed: {e}"))
        };
        let mut stdin = input.try_clone().map_err(staging_failed)?;
        stdin.rewind().map_err(staging_failed)?;
        run_tool(ToolCommand {
            program: &self.program,
            args,
            input: ToolInput::File(stdin),
            max_stdout,
            max_stderr: PROBE_REPORT_CAP,
            deadline,
            cpu_limit: tool_cpu_limit(self.timeout),
            max_address_space: DECODER_MAX_ADDRESS_SPACE,
        })
        .map_err(|e| {
            FrameError::Failed(match e {
                ToolError::Spawn(e) => format!(
                    "could not start the video decoder {}: {e}",
                    self.program.display()
                ),
                ToolError::OutputCap(cap) => {
                    format!("the decoder's output passed the {cap} byte cap")
                }
                ToolError::Read(e) => format!("reading the decoder's output failed: {e}"),
                ToolError::TimedOut => {
                    format!("video decoder timed out after {}s", self.timeout.as_secs())
                }
                ToolError::Wait(e) => format!("waiting on the video decoder failed: {e}"),
            })
        })
    }
}

/// Refuse a probed frame past the decode budget, as a bad
/// source. The displayed size meets the budget itself (see
/// [`MAX_FRAME_PIXELS`]): the PNG of a larger frame could
/// never pass the engine's decode limits. The coded size,
/// the frame the decoder allocates whatever part of it the
/// stream displays, meets the decoder's pixel bound (see
/// [`DECODER_MAX_PIXELS`]), which `-max_pixels` applies only
/// to the output picture, and the same limit on a side.
fn check_frame_size(probe: &VideoProbe) -> Result<(), FrameError> {
    let side = crate::engine::MAX_DECODE_DIMENSION;
    let (width, height) = (probe.width, probe.height);
    if width > side || height > side || u64::from(width) * u64::from(height) > MAX_FRAME_PIXELS {
        return Err(FrameError::NotVideo(format!(
            "the video's frames are {width}x{height}, past the decode budget \
             ({side} px a side, {MAX_FRAME_PIXELS} pixels)"
        )));
    }
    let (coded_width, coded_height) = (probe.coded_width, probe.coded_height);
    if coded_width > side
        || coded_height > side
        || u64::from(coded_width) * u64::from(coded_height) > DECODER_MAX_PIXELS
    {
        return Err(FrameError::NotVideo(format!(
            "the video's frames are coded at {coded_width}x{coded_height} \
             (displayed at {width}x{height}), past the decode budget \
             ({side} px a side, {DECODER_MAX_PIXELS} pixels with the decoder's \
             alignment room)"
        )));
    }
    Ok(())
}

/// The threads the extraction's decoder gets for a coded
/// frame of `width` by `height` (the frames a decoder holds
/// in flight are coded frames): up to
/// [`MANY_THREADS_MAX_PIXELS`], FFmpeg's own default count,
/// one more than the cores (here the cores the server may
/// use, which a container's CPU quota can lower), at most
/// [`MAX_DECODER_THREADS`]; past it, [`FEW_THREADS`]. A
/// stream that grows past the probed size keeps the probed
/// size's count, and the address-space cap bounds what its
/// larger frames hold (a stream that grew from 64x48 to
/// 8192x8192 peaked at 0.9 GB on sixteen threads).
fn decoder_threads(width: u32, height: u32) -> usize {
    if u64::from(width) * u64::from(height) > MANY_THREADS_MAX_PIXELS {
        return FEW_THREADS;
    }
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    (cores + 1).clamp(FEW_THREADS, MAX_DECODER_THREADS)
}

/// The refusal for a frame the decoder refused for its
/// size, which the run does not name.
fn frame_past_the_budget() -> FrameError {
    FrameError::NotVideo(format!(
        "a frame of the video is past the decode budget ({MAX_FRAME_PIXELS} pixels), \
         and the decoder refused it"
    ))
}

/// Whether a run's report says the decoder refused a frame
/// past `-max_pixels` (see [`PIXEL_BOUND_REPORT`]). The
/// whole report is read. It also carries the source's own
/// metadata, and a match there affects only that source's
/// own run, which then ends as a refusal without step-backs
/// to earlier frames.
fn met_the_pixel_bound(run: &ToolRun) -> bool {
    run.stderr.contains(PIXEL_BOUND_REPORT) || run.stderr_excerpt().contains(PIXEL_BOUND_REPORT)
}

/// Stage the source bytes for the tool's stdin (issue
/// #411): a fresh file in the temporary directory, created
/// exclusively and private to this user, unlinked as soon
/// as it exists, so nothing is left behind whatever happens
/// next and only the open descriptor reaches the bytes.
/// The file disappears when the last descriptor (ours, or
/// a run's stdin) closes.
fn stage_input(video: &[u8]) -> Result<File, FrameError> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let failed = |e: std::io::Error| {
        FrameError::Failed(format!("staging the video for the decoder failed: {e}"))
    };
    let dir = std::env::temp_dir();
    let mut attempt = 0;
    let (mut file, path) = loop {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let path = dir.join(format!(
            "crabcrop-frame-{}-{}-{nanos:08x}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&path) {
            Ok(file) => break (file, path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && attempt < 8 => attempt += 1,
            Err(e) => return Err(failed(e)),
        }
    };
    std::fs::remove_file(&path).map_err(failed)?;
    file.write_all(video).map_err(failed)?;
    Ok(file)
}

/// The frame interval a probe implies, for clamping and
/// step-backs. Without a reported rate, a tenth of a
/// second: short enough to land in the tail of almost
/// any real video, long enough to reach it in a few
/// steps.
fn frame_interval(fps: Option<f64>) -> f64 {
    match fps {
        Some(rate) if rate > 0.0 && rate.is_finite() => 1.0 / rate,
        _ => 0.1,
    }
}

/// Resolve a selection to the timestamp to seek, in
/// seconds. Poster is the first frame; a percentage
/// needs the probed duration; seconds and percentages
/// past the end clamp to the last frame's position
/// (duration minus one frame interval), because the
/// decoder's accurate seek returns nothing past it.
pub fn resolve_timestamp(selection: FrameSelection, probe: &VideoProbe) -> Result<f64, FrameError> {
    let raw = match selection {
        FrameSelection::Poster => 0.0,
        FrameSelection::Seconds(s) => s,
        FrameSelection::Percent(p) => match probe.duration {
            Some(duration) => duration * p / 100.0,
            None => {
                return Err(FrameError::Unsupported(
                    "a frame percentage needs the video's duration, and this \
                     video does not report one"
                        .to_string(),
                ));
            }
        },
    };
    Ok(match probe.duration {
        Some(duration) if duration > 0.0 => {
            raw.min((duration - frame_interval(probe.fps)).max(0.0))
        }
        _ => raw,
    })
}

/// Format a timestamp for the tool's argv and for error
/// text: milliseconds are far below one frame interval
/// at any real rate, and the value round-trips cleanly.
fn format_seconds(value: f64) -> String {
    let rounded = (value * 1000.0).round() / 1000.0;
    if rounded == rounded.trunc() {
        format!("{}", rounded as i64)
    } else {
        format!("{rounded}")
    }
}

/// Demuxers that carry raw still images rather than
/// video (issue #266): ffmpeg promotes a JPEG embedded
/// in an arbitrary blob (a PDF, a zip, junk bytes) to a
/// "video" stream through one of these, so a report
/// whose input is only one of them is not a video, no
/// matter what its stream list says. A real video, even
/// an MJPEG one, arrives through a container demuxer
/// (avi, mov, matroska, ...), never one of these alone.
const IMAGE_DEMUXERS: &[&str] = &[
    "image2",
    "image2pipe",
    "mjpeg",
    "jpeg_pipe",
    "jpegls_pipe",
    "jpeg2000_pipe",
    "png_pipe",
    "bmp_pipe",
    "webp_pipe",
    "gif",
    "gif_pipe",
    "tiff_pipe",
    "avif_pipe",
    "exr_pipe",
    "hdr_pipe",
    "dpx_pipe",
    "fits_pipe",
    "pam_pipe",
    "pbm_pipe",
    "pgm_pipe",
    "ppm_pipe",
    "qoi_pipe",
    "sunrast_pipe",
    "xbm_pipe",
    "xpm_pipe",
    "svg_pipe",
];

/// The media type markers a stream line can carry, as the
/// report spells them (the type name, capitalized, between
/// `: ` and `: `).
const STREAM_KINDS: &[&str] = &[
    ": Video: ",
    ": Audio: ",
    ": Subtitle: ",
    ": Data: ",
    ": Attachment: ",
    ": Unknown: ",
];

/// A fresh field separator for one probe (issue #402),
/// ` ~<16 hex digits>~ `, passed as the tool's
/// `-dump_separator`. The tool prints it inside each
/// stream's codec description and nowhere else, which is
/// how [`parse_probe`] tells a real stream line from text
/// the source planted, so whoever made the source must not
/// be able to predict it: `RandomState` keys its hasher
/// from the OS's random source and steps the keys for
/// every instance, so even the digest of no input is a
/// fresh 64-bit secret.
fn probe_separator() -> String {
    use std::hash::{BuildHasher, Hasher};
    let token = std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish();
    format!(" ~{token:016x}~ ")
}

/// Parse the probe facts out of the tool's stderr report:
/// the input's `Duration:` line and its first video stream
/// that carries dimensions. Returns `None` when no usable
/// video stream appears, which is the not-a-video answer
/// for every corrupt, audio-only, or non-video source, and
/// when the report carries a planted `Duration:` line,
/// which is the same answer for a forged one.
///
/// The report interleaves the facts with source-controlled
/// text: container, chapter, program, and stream metadata,
/// and each stream's language tag inside its own stream
/// line. A metadata value's continuation lines start with
/// `: `, but keys, program names, and language tags print
/// raw, newlines included, so a source can plant whole
/// lines shaped like the facts, indentation and all, and a
/// language tag can rewrite the start of a real stream line
/// and push the real codec text onto a line of its own
/// (issue #402, the video twin of the forged PDF page size
/// in #363). No layout rule survives that, so the parse
/// leans only on what a source cannot produce:
///
/// - The duration comes from the report's one `Duration:`
///   line at exactly two spaces of indent (metadata
///   entries print at four or more, so a long key that
///   starts with `Duration: ` is not it). The real line
///   always prints, so a second one is a planted copy, and
///   the report is refused.
/// - Stream facts come only from lines that carry
///   `separator`, the probe's random field separator
///   ([`probe_separator`]): the tool prints it inside each
///   stream's codec description, right after the codec
///   name, and a source cannot know it. On such a line the
///   media type is the last type marker before the first
///   separator; text a language tag adds sits before the
///   real marker, and only table text (codec, profile,
///   codec tag) sits between the marker and the separator.
pub fn parse_probe(report: &str, separator: &str) -> Option<VideoProbe> {
    let lines: Vec<&str> = report.lines().collect();
    let demuxers: Vec<&str> = lines
        .iter()
        .find_map(|line| {
            let rest = line.trim_start().strip_prefix("Input #")?;
            let (_, after) = rest.split_once(", ")?;
            let (list, _) = after.split_once(", from ")?;
            Some(list.split(',').map(str::trim).collect())
        })
        .unwrap_or_default();
    if !demuxers.is_empty() && demuxers.iter().all(|d| IMAGE_DEMUXERS.contains(d)) {
        return None;
    }

    let mut duration_lines = lines
        .iter()
        .enumerate()
        .filter_map(|(at, line)| Some((at, line.strip_prefix("  Duration: ")?)));
    let (duration_at, duration_rest) = duration_lines.next()?;
    if duration_lines.next().is_some() {
        return None;
    }
    let duration = parse_duration(duration_rest);

    // Streams print after the Duration line, in the order
    // the tool lists them; planted lines never carry the
    // separator, so they never reach the stream parse.
    let stream = lines[duration_at + 1..]
        .iter()
        .find_map(|line| parse_video_stream(line, separator))?;
    let (width, height) = stream.size;
    let (coded_width, coded_height) = stream.coded.unwrap_or(stream.size);
    Some(VideoProbe {
        duration,
        width,
        height,
        coded_width: coded_width.max(width),
        coded_height: coded_height.max(height),
        fps: stream.fps,
    })
}

/// The facts one video stream's codec description gives.
#[derive(Debug, PartialEq)]
struct StreamFacts {
    /// The displayed size.
    size: (u32, u32),
    /// The coded size, when the report gives one (only where
    /// it differs from the displayed size).
    coded: Option<(u32, u32)>,
    /// The frame rate, when the report gives one.
    fps: Option<f64>,
}

/// Parse an `HH:MM:SS.cc` duration token (the text right
/// after `Duration: `, up to the comma). `N/A` and
/// anything malformed mean the container reports no
/// duration, which is a fact, not a failure. Every part
/// must be plain digits: `str::parse::<f64>` alone would
/// also take `NaN`, `inf`, signs, and exponents.
fn parse_duration(rest: &str) -> Option<f64> {
    let token = rest.split(',').next()?.trim();
    let mut parts = token.split(':');
    let (hours, minutes, seconds) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    let digits = |part: &str| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit());
    let (whole, fraction) = seconds.split_once('.').unwrap_or((seconds, "0"));
    if !digits(hours) || !digits(minutes) || !digits(whole) || !digits(fraction) {
        return None;
    }
    let total = hours.parse::<f64>().ok()? * 3600.0
        + minutes.parse::<f64>().ok()? * 60.0
        + seconds.parse::<f64>().ok()?;
    total.is_finite().then_some(total)
}

/// Parse one report line as a video stream's codec
/// description. The line must carry the probe's
/// `separator`, and its media type is the last type marker
/// before the first one (see [`parse_probe`]); an attached
/// picture (album art) is not the video stream, even when
/// it is the only one. From the text after the marker,
/// with each separator read as the tool's default `, `:
/// the first comma field whose leading token is exactly
/// `<w>x<h>`, the displayed size (the field usually
/// continues with `[SAR ... DAR ...]`, and at `verbose`, where
/// the coded size differs, with `(<w>x<h>)` right after it),
/// and the `<n> fps` field when present (with `<n> tbr` as
/// the fallback rate). Leading-token matching plus a nonzero
/// check keeps hex codec tags (`0x31637661`, which would
/// split as 0 by 31637661) from reading as dimensions.
fn parse_video_stream(line: &str, separator: &str) -> Option<StreamFacts> {
    let head = &line[..line.find(separator)?];
    let (at, kind) = STREAM_KINDS
        .iter()
        .filter_map(|kind| Some((head.rfind(kind)?, *kind)))
        .max_by_key(|(at, _)| *at)?;
    if kind != ": Video: " {
        return None;
    }
    let after = line[at + kind.len()..].replace(separator, ", ");
    if after.contains("(attached pic)") {
        return None;
    }
    let rate = |token: &str| {
        token
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && *v > 0.0)
    };
    let nonzero = |token: &str| parse_dims(token).filter(|&(w, h)| w > 0 && h > 0);
    let mut dims = None;
    let mut coded = None;
    let mut fps = None;
    let mut tbr = None;
    for field in after.split(',') {
        let tokens: Vec<&str> = field.split_whitespace().collect();
        if dims.is_none()
            && let Some(first) = tokens.first()
            && let Some(size) = nonzero(first)
        {
            dims = Some(size);
            coded = tokens
                .get(1)
                .and_then(|token| token.strip_prefix('(')?.strip_suffix(')'))
                .and_then(nonzero);
        }
        if tokens.len() == 2 {
            match tokens[1] {
                "fps" => fps = rate(tokens[0]),
                "tbr" => tbr = rate(tokens[0]),
                _ => {}
            }
        }
    }
    Some(StreamFacts {
        size: dims?,
        coded,
        fps: fps.or(tbr),
    })
}

/// Parse a strict `<digits>x<digits>` dimension token.
fn parse_dims(token: &str) -> Option<(u32, u32)> {
    let (w, h) = token.split_once('x')?;
    if w.is_empty()
        || h.is_empty()
        || !w.bytes().all(|b| b.is_ascii_digit())
        || !h.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    Some((w.parse().ok()?, h.parse().ok()?))
}

#[cfg(test)]
mod tests;
