//! Video output: animated sources encoded to MP4 or WebM by
//! an external encoder the operator configures (issue #162,
//! option C).
//!
//! crabcrop ships no video codec. The operator points
//! `CRABCROP_VIDEO_ENCODER` at a program (FFmpeg is the
//! reference implementation) and crabcrop runs it under a
//! fixed pipe contract: the argv below, which crabcrop
//! constructs in full, frames in as rawvideo on stdin,
//! encoded bytes out on stdout. The configured path is the
//! only operator input; no shell is involved and no request
//! data reaches the argv beyond numbers crabcrop computed
//! itself (dimensions, frame rate, the output format).
//!
//! The subprocess is the most heavily bounded path in the
//! server: a dedicated gate in the server bounds concurrency,
//! a hard wall-clock timeout kills the encode (the whole
//! process group, so a wrapper script's children die with
//! it), and stdout is capped while it is read, so a runaway
//! encoder cannot fill memory. The child runs with an empty
//! environment, in the temporary directory, and is always
//! reaped: it goes through the runner the frame and PDF
//! tools use ([`crate::toolrun`]).
//!
//! Timing: source frame delays are variable, encoders want a
//! constant frame rate. The schedule picks the smallest rate
//! that can express the fastest frame (the reciprocal of the
//! shortest delay, at most [`MAX_FPS`]) and repeats each
//! frame for the rounded number of samples its delay spans,
//! at least one. Total duration is preserved to within one
//! sample per frame (at 100 fps, 10 ms), and the sum of the
//! rounded samples is the output's duration. Looping is not
//! carried over: neither MP4 nor WebM has a loop flag
//! browsers honor, so looping is the embedding page's
//! `<video loop>` attribute.
//!
//! Frames are composited over black before encoding: H.264
//! carries no alpha in any browser-playable profile, and
//! VP9 alpha is a Chrome-only behavior, so transparency
//! cannot survive the trip. 4:2:0 chroma subsampling needs
//! even width and height, so an odd side is padded by one
//! pixel (edge-replicated), except at the per-axis output
//! limit, where padding would pass it and the last row or
//! column is cropped off instead (see [`even_dimension`]).

use std::io::Write;
use std::path::PathBuf;
use std::process::ChildStdin;
use std::time::{Duration, Instant};

use crate::toolrun::{ToolCommand, ToolError, ToolInput, run_tool, tool_cpu_limit};
use crate::url::ImageFormat;

/// Default encode wall-clock budget, in seconds
/// (`CRABCROP_VIDEO_TIMEOUT_SECS`).
pub const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Default encoded-output cap, in megabytes
/// (`CRABCROP_VIDEO_MAX_MB`).
pub const DEFAULT_MAX_OUTPUT_MB: usize = 64;

/// Permits on the server's dedicated video gate. Mirrors
/// the AI gate: encodes are seconds-scale work, and a burst
/// of them must not starve the transform pipeline.
pub const GATE_PERMITS: usize = 2;

/// The fastest frame rate the schedule will pick. GIF and
/// WebP delays are millisecond-grained, so 100 fps (a 10 ms
/// grid) expresses every delay a real animation carries.
pub const MAX_FPS: u32 = 100;

/// The longest animation video output accepts, in
/// milliseconds of summed frame delays. The input bounds
/// (200 frames) cap frame count, not runtime: 200 frames at
/// the GIF maximum delay run for hours, and no embedding
/// page wants that video.
pub const MAX_DURATION_MS: u64 = 600_000;

/// One frame on its way to the encoder: composited RGBA
/// pixels (post-operation, full canvas) and the delay the
/// source assigned it, in milliseconds.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub pixels: image::RgbaImage,
    pub delay_ms: u64,
}

/// How a video encode failed, split by who is at fault so
/// the engine can map the classes to responses: a plan the
/// encoder never saw (the request or the source is wrong,
/// a 400) versus the encoder itself failing (a 502).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoError {
    /// The frames cannot be encoded as planned: mixed
    /// dimensions, no frames, an animation past the
    /// duration limit. The encoder was never started.
    Unsupported(String),
    /// The encoder was started (or should have been) and
    /// failed: spawn failure, non-zero exit, timeout,
    /// output past the cap, or no output at all.
    Failed(String),
}

impl std::fmt::Display for VideoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VideoError::Unsupported(what) | VideoError::Failed(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for VideoError {}

/// The constant-rate schedule for variable source delays:
/// the frame rate, and how many samples each frame occupies
/// at that rate. Pure and separately tested: the duration
/// math is the contract the docs state.
///
/// The rate is the smallest one that expresses the shortest
/// positive delay (`ceil(1000 / min_delay)`), capped at
/// [`MAX_FPS`]; with no positive delay at all it is 10 fps.
/// Each frame then spans `round(delay * fps / 1000)` samples,
/// at least one: a zero-delay frame still shows once.
pub fn frame_schedule(delays_ms: &[u64]) -> (u32, Vec<u32>) {
    let fps = match delays_ms.iter().copied().filter(|&d| d > 0).min() {
        Some(shortest) => (1000u64.div_ceil(shortest)).clamp(1, MAX_FPS as u64) as u32,
        None => 10,
    };
    let counts = delays_ms
        .iter()
        .map(|&d| {
            if d == 0 {
                1
            } else {
                // Round half up, in exact integer arithmetic.
                ((d * fps as u64 + 500) / 1000).max(1) as u32
            }
        })
        .collect();
    (fps, counts)
}

/// The even side the encoder gets for a frame side, the
/// 4:2:0 requirement. An odd side pads up by one pixel,
/// unless that would pass [`crate::url::MAX_DIMENSION`],
/// the limit every output is held to (issue #528): the
/// limit is odd itself, so a 16383 px side pads to 16384,
/// one past it. That side crops down to 16382 instead,
/// losing its last row or column.
pub fn even_dimension(dimension: u32) -> u32 {
    if dimension.is_multiple_of(2) {
        dimension
    } else if dimension < crate::url::MAX_DIMENSION {
        dimension + 1
    } else {
        dimension - 1
    }
}

/// What one encode will run: the even output dimensions,
/// the constant frame rate, and the per-frame sample
/// counts.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Plan {
    width: u32,
    height: u32,
    fps: u32,
    counts: Vec<u32>,
}

fn plan(frames: &[VideoFrame]) -> Result<Plan, VideoError> {
    let first = frames.first().ok_or_else(|| {
        VideoError::Unsupported("video output needs an animated source".to_string())
    })?;
    let (width, height) = (first.pixels.width(), first.pixels.height());
    if frames
        .iter()
        .any(|f| f.pixels.width() != width || f.pixels.height() != height)
    {
        return Err(VideoError::Unsupported(
            "video frames disagree on dimensions".to_string(),
        ));
    }
    // The engine's chain already holds every frame to the
    // limit (issue #528); the plan checks it again, because
    // `even_dimension` keeps a side within the limit only
    // when it starts there.
    let max = crate::url::MAX_DIMENSION;
    if width > max || height > max {
        return Err(VideoError::Unsupported(format!(
            "video frames of {width}x{height} exceed the {max} pixel limit per axis"
        )));
    }
    let delays: Vec<u64> = frames.iter().map(|f| f.delay_ms).collect();
    let total_ms: u64 = delays.iter().sum();
    if total_ms > MAX_DURATION_MS {
        return Err(VideoError::Unsupported(format!(
            "the animation runs {}s, past the {}s limit for video output",
            total_ms / 1000,
            MAX_DURATION_MS / 1000
        )));
    }
    let (fps, counts) = frame_schedule(&delays);
    Ok(Plan {
        width: even_dimension(width),
        height: even_dimension(height),
        fps,
        counts,
    })
}

/// Composite one frame over black and fit it to the plan's
/// dimensions, padding by edge replication or cropping the
/// last row or column at the limit: the exact bytes one
/// sample sends the encoder.
fn sample_bytes(frame: &VideoFrame, plan: &Plan) -> Vec<u8> {
    let (sw, sh) = (frame.pixels.width(), frame.pixels.height());
    let mut out = vec![0u8; (plan.width * plan.height * 4) as usize];
    for y in 0..plan.height {
        let sy = y.min(sh - 1);
        for x in 0..plan.width {
            let sx = x.min(sw - 1);
            let px = frame.pixels.get_pixel(sx, sy);
            let a = px[3] as u32;
            let at = ((y * plan.width + x) * 4) as usize;
            // Over black: scale by alpha, rounding to nearest.
            out[at] = ((px[0] as u32 * a + 127) / 255) as u8;
            out[at + 1] = ((px[1] as u32 * a + 127) / 255) as u8;
            out[at + 2] = ((px[2] as u32 * a + 127) / 255) as u8;
            out[at + 3] = 255;
        }
    }
    out
}

/// The operator-configured external encoder: a program path
/// plus the bounds every encode runs under. Built once at
/// boot; a configured path that is missing or not executable
/// fails startup there, not the first request.
#[derive(Debug, Clone)]
pub struct VideoEncoder {
    program: PathBuf,
    timeout: Duration,
    max_output_bytes: usize,
}

impl VideoEncoder {
    /// Build an encoder, validating the path the way boot
    /// needs: the file must exist and be executable.
    pub fn new(
        program: impl Into<PathBuf>,
        timeout: Duration,
        max_output_bytes: usize,
    ) -> Result<Self, String> {
        let program = program.into();
        let metadata = std::fs::metadata(&program)
            .map_err(|e| format!("{} is not usable: {e}", program.display()))?;
        if !metadata.is_file() {
            return Err(format!("{} is not a file", program.display()));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o111 == 0 {
                return Err(format!("{} is not executable", program.display()));
            }
        }
        Ok(VideoEncoder {
            program,
            timeout,
            max_output_bytes,
        })
    }

    /// An encoder with the default bounds (30 s, 64 MiB).
    #[cfg(test)]
    fn with_defaults(program: impl Into<PathBuf>) -> Result<Self, String> {
        Self::new(
            program,
            Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            DEFAULT_MAX_OUTPUT_MB * 1024 * 1024,
        )
    }

    /// The argv crabcrop constructs for one encode, in full:
    /// the fixed pipe contract. Frames arrive as rawvideo
    /// (RGBA, the plan's dimensions and rate) on stdin; the
    /// encoded video leaves on stdout. MP4 over a pipe cannot
    /// carry a seek-back index, so the fragmented flags are
    /// part of the contract, not a tuning choice.
    fn argv(&self, plan: &Plan, format: ImageFormat) -> Vec<String> {
        let mut args: Vec<String> = [
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{}x{}", plan.width, plan.height),
            "-r",
            &plan.fps.to_string(),
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            "format=yuv420p",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        match format {
            ImageFormat::Mp4 => args.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-movflags",
                    "frag_keyframe+empty_moov+default_base_moof",
                    "-f",
                    "mp4",
                ]
                .iter()
                .map(|s| s.to_string()),
            ),
            ImageFormat::WebM => args.extend(
                [
                    "-c:v",
                    "libvpx-vp9",
                    "-b:v",
                    "0",
                    "-crf",
                    "30",
                    "-f",
                    "webm",
                ]
                .iter()
                .map(|s| s.to_string()),
            ),
            _ => unreachable!("video output is mp4 or webm"),
        }
        args.push("pipe:1".to_string());
        args
    }

    /// Encode frames to the requested video format under
    /// the configured bounds.
    pub fn encode(
        &self,
        frames: &[VideoFrame],
        format: ImageFormat,
    ) -> Result<Vec<u8>, VideoError> {
        let plan = plan(frames)?;
        let argv = self.argv(&plan, format);
        self.run(&argv, frames, &plan)
    }

    /// Run the encoder through the shared runner
    /// ([`crate::toolrun::run_tool`]): the scheduled samples
    /// fed on stdin from a writer thread, the video read from
    /// stdout under the output cap, and the whole run held to
    /// the wall-clock timeout. Stderr is quoted in the error
    /// message and never trusted beyond that.
    fn run(
        &self,
        argv: &[String],
        frames: &[VideoFrame],
        plan: &Plan,
    ) -> Result<Vec<u8>, VideoError> {
        // One composited sample per frame, computed once;
        // the writer repeats each for its scheduled count.
        let samples: Vec<Vec<u8>> = frames.iter().map(|f| sample_bytes(f, plan)).collect();
        let counts = plan.counts.clone();
        let feed = move |stdin: &mut ChildStdin| {
            for (sample, &count) in samples.iter().zip(&counts) {
                for _ in 0..count {
                    stdin.write_all(sample)?;
                }
            }
            Ok(())
        };
        let args: Vec<&str> = argv.iter().map(String::as_str).collect();
        let run = run_tool(ToolCommand {
            program: &self.program,
            args: &args,
            input: ToolInput::Writer(Box::new(feed)),
            max_stdout: self.max_output_bytes,
            // Only the excerpt is read, and it comes from the
            // end of the stream, whatever this keeps.
            max_stderr: 8192,
            deadline: Instant::now() + self.timeout,
            cpu_limit: tool_cpu_limit(self.timeout),
        })
        .map_err(|e| {
            VideoError::Failed(match e {
                ToolError::Spawn(e) => {
                    format!("could not be started ({}): {e}", self.program.display())
                }
                ToolError::OutputCap(cap) => {
                    format!("output exceeded the {cap} byte cap and the encode was aborted")
                }
                ToolError::Read(e) => format!("reading its output failed: {e}"),
                ToolError::TimedOut => {
                    format!("timed out after {}s and was killed", self.timeout.as_secs())
                }
                ToolError::Wait(_) => "could not be waited on after it finished".to_string(),
            })
        })?;
        if !run.status.success() {
            return Err(VideoError::Failed(format!(
                "exited with status {}: {}",
                run.status,
                run.stderr_excerpt()
            )));
        }
        if run.stdout.is_empty() {
            return Err(VideoError::Failed("produced no output".to_string()));
        }
        Ok(run.stdout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// A fresh temp directory for one test's scripts and
    /// recordings.
    fn test_dir(name: &str) -> crate::test_support::ScratchDir {
        crate::test_support::scratch_dir(&format!("video-{name}"))
    }

    /// Write a fake encoder script and make it executable.
    fn fake_encoder(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("fake-encoder.sh");
        crate::test_support::write_script(&path, body);
        path
    }

    /// A solid-color frame of the given size and delay.
    fn frame(width: u32, height: u32, rgba: [u8; 4], delay_ms: u64) -> VideoFrame {
        VideoFrame {
            pixels: image::RgbaImage::from_pixel(width, height, image::Rgba(rgba)),
            delay_ms,
        }
    }

    #[test]
    fn schedule_picks_the_rate_from_the_fastest_frame() {
        // Uniform 100 ms delays: 10 fps, one sample each.
        assert_eq!(frame_schedule(&[100, 100, 100]), (10, vec![1, 1, 1]));
        // The 100 ms frame sets the rate; the 300 ms frame
        // spans three samples. Total duration is exact.
        assert_eq!(frame_schedule(&[100, 300]), (10, vec![1, 3]));
        // A 10 ms frame pushes the rate to the 100 fps cap.
        assert_eq!(frame_schedule(&[10, 40]), (100, vec![1, 4]));
        // Slower than 1 fps clamps to 1 fps.
        assert_eq!(frame_schedule(&[2000]), (1, vec![2]));
    }

    #[test]
    fn schedule_rounds_to_the_nearest_sample_and_never_drops_a_frame() {
        // 150 ms at 10 fps is 1.5 samples: rounds to 2
        // (25 ms of drift on a 2.5 s clip is the documented
        // one-sample bound).
        assert_eq!(frame_schedule(&[100, 150]), (10, vec![1, 2]));
        // Zero-delay frames still show once, and an all-zero
        // animation runs at the 10 fps default.
        assert_eq!(frame_schedule(&[0, 0]), (10, vec![1, 1]));
        assert_eq!(frame_schedule(&[0, 100]), (10, vec![1, 1]));
        // A delay below half a sample still gets its frame.
        assert_eq!(frame_schedule(&[100, 30]), (34, vec![3, 1]));
    }

    #[test]
    fn boot_validation_rejects_missing_and_unexecutable_paths() {
        let dir = test_dir("boot");
        let missing = dir.join("nope");
        let err = VideoEncoder::with_defaults(&missing).unwrap_err();
        assert!(err.contains("not usable"), "{err}");
        let plain = dir.join("plain-file");
        std::fs::write(&plain, b"not a program").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&plain, std::fs::Permissions::from_mode(0o644)).unwrap();
            let err = VideoEncoder::with_defaults(&plain).unwrap_err();
            assert!(err.contains("not executable"), "{err}");
        }
        let err = VideoEncoder::with_defaults(dir.path()).unwrap_err();
        assert!(err.contains("not a file"), "{err}");
    }

    #[test]
    fn plan_rejects_overlong_and_mixed_animations() {
        let frames = vec![frame(4, 2, [1, 2, 3, 255], MAX_DURATION_MS + 1)];
        let err = plan(&frames).unwrap_err();
        assert!(matches!(err, VideoError::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("video output"), "{err}");
        let frames = vec![
            frame(4, 2, [1, 2, 3, 255], 100),
            frame(6, 2, [1, 2, 3, 255], 100),
        ];
        assert!(matches!(
            plan(&frames).unwrap_err(),
            VideoError::Unsupported(_)
        ));
        assert!(matches!(plan(&[]).unwrap_err(), VideoError::Unsupported(_)));
    }

    #[test]
    fn encode_runs_the_pipe_contract() {
        let dir = test_dir("contract");
        let argv_file = dir.join("argv.txt");
        let stdin_file = dir.join("stdin.bin");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\ncat > '{}'\nprintf 'FAKE-MP4-BYTES'\n",
            argv_file.display(),
            stdin_file.display()
        );
        let encoder = VideoEncoder::with_defaults(fake_encoder(&dir, &script)).unwrap();
        // Two 5x3 frames (odd dimensions pad to 6x4), delays
        // 100 and 300 ms: 10 fps, samples 1 and 3. The first
        // frame carries a half-transparent pixel to prove
        // compositing over black.
        let mut first = frame(5, 3, [200, 100, 50, 255], 100);
        first
            .pixels
            .put_pixel(0, 0, image::Rgba([100, 200, 50, 128]));
        let frames = vec![first, frame(5, 3, [0, 0, 255, 255], 300)];
        let out = encoder.encode(&frames, ImageFormat::Mp4).unwrap();
        assert_eq!(out, b"FAKE-MP4-BYTES");
        let argv = std::fs::read_to_string(&argv_file).unwrap();
        let args: Vec<&str> = argv.lines().collect();
        assert!(args.contains(&"rawvideo"), "{argv}");
        assert!(args.contains(&"rgba"), "{argv}");
        assert!(args.contains(&"6x4"), "{argv}");
        assert!(args.contains(&"10"), "{argv}");
        assert!(
            args.contains(&"frag_keyframe+empty_moov+default_base_moof"),
            "{argv}"
        );
        assert!(args.contains(&"mp4"), "{argv}");
        assert_eq!(args.last(), Some(&"pipe:1"), "{argv}");
        // 4 samples of 6x4 RGBA.
        let stdin = std::fs::read(&stdin_file).unwrap();
        assert_eq!(
            stdin.len(),
            4 * 6 * 4 * 4,
            "stdin was {} bytes",
            stdin.len()
        );
        // Sample 1 is frame 1: the half-transparent pixel is
        // composited over black (100*128/255 rounds to 50,
        // 200*128/255 to 100, 50*128/255 to 25), alpha 255.
        assert_eq!(&stdin[0..4], &[50, 100, 25, 255]);
        // The rest of frame 1 is the opaque color as-is.
        assert_eq!(&stdin[4..8], &[200, 100, 50, 255]);
        // Edge replication: the padded column repeats the
        // last source column, the padded row the last row.
        assert_eq!(&stdin[20..24], &[200, 100, 50, 255]); // x=5 (pad), y=0
        let row3 = 3 * 6 * 4;
        assert_eq!(&stdin[row3..row3 + 4], &[200, 100, 50, 255]); // y=3 (pad row) replicates source row 2
        // Samples 2-4 are frame 2, solid blue.
        let sample2 = 6 * 4 * 4;
        assert_eq!(&stdin[sample2..sample2 + 4], &[0, 0, 255, 255]);
        assert_eq!(&stdin[stdin.len() - 4..], &[0, 0, 255, 255]);
    }

    #[test]
    fn even_dimension_pads_up_and_crops_down_at_the_limit() {
        // Issue #528: the limit is odd, so padding its side
        // up passed it; that one side crops down instead.
        let max = crate::url::MAX_DIMENSION;
        assert_eq!(even_dimension(1), 2);
        assert_eq!(even_dimension(2), 2);
        assert_eq!(even_dimension(5), 6);
        assert_eq!(even_dimension(max - 2), max - 1);
        assert_eq!(even_dimension(max - 1), max - 1);
        assert_eq!(even_dimension(max), max - 1);
        for side in 1..=max {
            let even = even_dimension(side);
            assert!(
                even.is_multiple_of(2) && even <= max && even.abs_diff(side) <= 1,
                "{side} -> {even}"
            );
        }
    }

    #[test]
    fn plan_holds_frames_to_the_output_limit() {
        let plan_of = |width, height| plan(&[frame(width, height, [1, 2, 3, 255], 100)]);
        let planned = plan_of(16_383, 3).unwrap();
        assert_eq!((planned.width, planned.height), (16_382, 4));
        let planned = plan_of(3, 16_383).unwrap();
        assert_eq!((planned.width, planned.height), (4, 16_382));
        // A frame past the limit never reaches the encoder.
        for (width, height) in [(16_384, 2), (2, 16_384), (16_385, 1)] {
            let err = plan_of(width, height).unwrap_err();
            assert!(matches!(err, VideoError::Unsupported(_)), "{err:?}");
            assert_eq!(
                err.to_string(),
                format!("video frames of {width}x{height} exceed the 16383 pixel limit per axis")
            );
        }
    }

    #[test]
    fn encode_crops_the_odd_edge_at_the_limit() {
        // A 16383x3 frame goes to the encoder as 16382x4: the
        // last column is cropped off, the odd row still pads
        // by replicating the last one (issue #528).
        let dir = test_dir("crop-at-limit");
        let argv_file = dir.join("argv.txt");
        let stdin_file = dir.join("stdin.bin");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\ncat > '{}'\nprintf 'FAKE-MP4-BYTES'\n",
            argv_file.display(),
            stdin_file.display()
        );
        let encoder = VideoEncoder::with_defaults(fake_encoder(&dir, &script)).unwrap();
        let (kept, dropped) = ([0, 250, 0, 255], [250, 0, 0, 255]);
        let mut wide = frame(16_383, 3, [10, 20, 30, 255], 100);
        for y in 0..3 {
            wide.pixels.put_pixel(16_381, y, image::Rgba(kept));
            wide.pixels.put_pixel(16_382, y, image::Rgba(dropped));
        }
        encoder.encode(&[wide], ImageFormat::Mp4).unwrap();
        let argv = std::fs::read_to_string(&argv_file).unwrap();
        assert!(argv.lines().any(|arg| arg == "16382x4"), "{argv}");
        let stdin = std::fs::read(&stdin_file).unwrap();
        assert_eq!(stdin.len(), 16_382 * 4 * 4);
        let row = 16_382 * 4;
        for y in 0..4 {
            let last = y * row + row - 4;
            assert_eq!(&stdin[last..last + 4], &kept, "row {y}");
        }
        assert!(
            stdin.chunks(4).all(|px| px != dropped),
            "the cropped column reached the encoder"
        );
    }

    #[test]
    fn encode_webm_uses_the_vp9_argv() {
        let dir = test_dir("webm");
        let argv_file = dir.join("argv.txt");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\ncat > /dev/null\nprintf 'FAKE-WEBM'\n",
            argv_file.display()
        );
        let encoder = VideoEncoder::with_defaults(fake_encoder(&dir, &script)).unwrap();
        let frames = vec![frame(4, 4, [9, 9, 9, 255], 100)];
        let out = encoder.encode(&frames, ImageFormat::WebM).unwrap();
        assert_eq!(out, b"FAKE-WEBM");
        let argv = std::fs::read_to_string(&argv_file).unwrap();
        assert!(argv.contains("libvpx-vp9"), "{argv}");
        assert!(argv.contains("webm"), "{argv}");
    }

    #[test]
    fn encoder_failures_are_specific() {
        // Non-zero exit, with its stderr in the message.
        let dir = test_dir("fail-exit");
        let script = "#!/bin/sh\ncat > /dev/null\necho 'encoder exploded' >&2\nexit 3\n";
        let encoder = VideoEncoder::with_defaults(fake_encoder(&dir, script)).unwrap();
        let frames = vec![frame(4, 2, [1, 2, 3, 255], 100)];
        let err = encoder.encode(&frames, ImageFormat::Mp4).unwrap_err();
        assert!(matches!(err, VideoError::Failed(_)), "{err:?}");
        assert!(err.to_string().contains("exit status: 3"), "{err}");
        assert!(err.to_string().contains("encoder exploded"), "{err}");

        // A configured path that vanishes after boot: spawn
        // failure, named as such.
        let dir = test_dir("fail-spawn");
        let script_path = fake_encoder(&dir, "#!/bin/sh\nprintf 'x'\n");
        let encoder = VideoEncoder::with_defaults(&script_path).unwrap();
        std::fs::remove_file(&script_path).unwrap();
        let err = encoder.encode(&frames, ImageFormat::Mp4).unwrap_err();
        assert!(err.to_string().contains("could not be started"), "{err}");

        // Empty output is a failure, not an empty video.
        let dir = test_dir("fail-empty");
        let script = "#!/bin/sh\ncat > /dev/null\n";
        let encoder = VideoEncoder::with_defaults(fake_encoder(&dir, script)).unwrap();
        let err = encoder.encode(&frames, ImageFormat::Mp4).unwrap_err();
        assert!(err.to_string().contains("no output"), "{err}");
    }

    #[test]
    fn encoder_runs_in_the_temp_directory() {
        // Issue #481: the encoder's working directory is the
        // temporary directory, as the frame and PDF tools'
        // is, not the server's own.
        let dir = test_dir("cwd");
        let cwd_file = dir.join("cwd.txt");
        let script = format!(
            "#!/bin/sh\npwd -P > '{}'\ncat > /dev/null\nprintf 'FAKE-MP4'\n",
            cwd_file.display()
        );
        let encoder = VideoEncoder::with_defaults(fake_encoder(&dir, &script)).unwrap();
        let frames = vec![frame(4, 2, [1, 2, 3, 255], 100)];
        let out = encoder.encode(&frames, ImageFormat::Mp4).unwrap();
        assert_eq!(out, b"FAKE-MP4");
        let cwd = std::fs::read_to_string(&cwd_file).unwrap();
        assert_eq!(
            std::fs::canonicalize(cwd.trim()).unwrap(),
            std::fs::canonicalize(std::env::temp_dir()).unwrap()
        );
    }

    #[test]
    fn encoder_errors_quote_the_end_of_stderr() {
        // Issue #481: the error quotes the last 500 characters
        // of stderr, where the encoder reports its failure,
        // as the frame and PDF errors do.
        let dir = test_dir("fail-tail");
        let script = "#!/bin/sh\ncat > /dev/null\nhead -c 2000 /dev/zero | tr '\\000' x >&2\necho ' the real failure' >&2\nexit 1\n";
        let encoder = VideoEncoder::with_defaults(fake_encoder(&dir, script)).unwrap();
        let frames = vec![frame(4, 2, [1, 2, 3, 255], 100)];
        let err = encoder.encode(&frames, ImageFormat::Mp4).unwrap_err();
        assert!(err.to_string().ends_with("x the real failure"), "{err}");
    }

    #[test]
    fn encoder_output_cap_aborts_the_encode() {
        let dir = test_dir("fail-cap");
        let script = "#!/bin/sh\ncat > /dev/null\nhead -c 100000 /dev/zero\n";
        let encoder =
            VideoEncoder::new(fake_encoder(&dir, script), Duration::from_secs(30), 1024).unwrap();
        let frames = vec![frame(4, 2, [1, 2, 3, 255], 100)];
        let err = encoder.encode(&frames, ImageFormat::Mp4).unwrap_err();
        assert!(
            err.to_string().contains("exceeded the 1024 byte cap"),
            "{err}"
        );
    }

    #[test]
    fn encoder_timeout_kills_the_encode() {
        let dir = test_dir("fail-timeout");
        let script = "#!/bin/sh\nsleep 5\nprintf 'too late'\n";
        let encoder = VideoEncoder::new(
            fake_encoder(&dir, script),
            Duration::from_millis(300),
            1024 * 1024,
        )
        .unwrap();
        let frames = vec![frame(4, 2, [1, 2, 3, 255], 100)];
        let started = Instant::now();
        let err = encoder.encode(&frames, ImageFormat::Mp4).unwrap_err();
        assert!(err.to_string().contains("timed out"), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "the encode was not killed at the timeout: {:?}",
            started.elapsed()
        );
    }

    #[cfg(unix)]
    #[test]
    fn encoder_timeout_kills_the_whole_process_group() {
        // Audit R2-31: the encoder is a wrapper that forks a
        // long-lived grandchild and waits on it. Killing only
        // the direct child would leave the grandchild running
        // past the timeout; the group kill reaches it.
        let dir = test_dir("fail-group");
        let pid_file = dir.join("grandchild.pid");
        let script = format!(
            "#!/bin/sh\nsleep 30 &\necho $! > '{}'\nwait\n",
            pid_file.display()
        );
        let encoder = VideoEncoder::new(
            fake_encoder(&dir, &script),
            Duration::from_millis(300),
            1024 * 1024,
        )
        .unwrap();
        let frames = vec![frame(4, 2, [1, 2, 3, 255], 100)];
        let started = Instant::now();
        let err = encoder.encode(&frames, ImageFormat::Mp4).unwrap_err();
        assert!(err.to_string().contains("timed out"), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "the encode was not killed at the timeout: {:?}",
            started.elapsed()
        );
        let pid: i32 = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        // Signal 0 finds nothing once the grandchild is gone;
        // give the kill a moment to land before concluding.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let alive = std::process::Command::new("kill")
                .args(["-0", &pid.to_string()])
                .status()
                .is_ok_and(|status| status.success());
            if !alive {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "grandchild {pid} survived the process-group kill"
            );
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    /// Encode two frames with the real FFmpeg and check the
    /// containers: an MP4 carries the ftyp box, a WebM the
    /// EBML header. Run it with:
    /// `cargo test -- --ignored real_ffmpeg`
    #[test]
    #[ignore = "needs a real encoder binary (ffmpeg) on this machine"]
    fn real_ffmpeg_encodes_mp4_and_webm() {
        let program =
            std::env::var("CRABCROP_TEST_FFMPEG").unwrap_or_else(|_| "/usr/bin/ffmpeg".to_string());
        let encoder = VideoEncoder::with_defaults(program).unwrap();
        let frames = vec![
            frame(64, 48, [200, 40, 40, 255], 100),
            frame(64, 48, [40, 40, 200, 255], 100),
        ];
        let mp4 = encoder.encode(&frames, ImageFormat::Mp4).unwrap();
        assert!(mp4.len() > 100, "suspiciously small MP4: {}", mp4.len());
        assert_eq!(&mp4[4..8], b"ftyp", "MP4 has no ftyp box");
        let webm = encoder.encode(&frames, ImageFormat::WebM).unwrap();
        assert!(webm.len() > 100, "suspiciously small WebM: {}", webm.len());
        assert_eq!(
            &webm[0..4],
            &[0x1a, 0x45, 0xdf, 0xa3],
            "WebM has no EBML header"
        );
    }
}
