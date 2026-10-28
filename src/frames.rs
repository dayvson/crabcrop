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
//! and a cap on what it may write back. The argv contract
//! is fixed and built entirely here:
//!
//! - Probe: `-nostdin -v info -i pipe:0`. No output
//!   target is named, so the tool prints its input report
//!   (duration, stream list) to stderr and exits nonzero;
//!   the probe reads its facts from that report and treats
//!   the exit status as meaningless. A source with no
//!   video stream in the report (a non-video, an
//!   audio-only file, a corrupt container) is
//!   [`FrameError::NotVideo`].
//! - Extract: `-nostdin -v info -i pipe:0 -ss <t>
//!   -frames:v 1 -f image2pipe -vcodec png pipe:1`. The
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
//! The extracted PNG re-enters the image pipeline as
//! the source, so every downstream budget (decode
//! dimensions and allocation, output pixels) applies to
//! it unchanged.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::url::FrameSelection;

/// Cap on the extracted frame's PNG bytes. Tied to the
/// engine's decode allocation budget (256 MiB): a frame
/// whose PNG passes this can never survive the decode
/// budgets, so anything larger is a broken or hostile
/// tool, not a frame.
pub const MAX_FRAME_PNG_BYTES: usize = 256 * 1024 * 1024;

/// How much of a tool's stderr report the probe keeps.
/// The input report sits at the head of the output; this
/// is generous headroom for chatty builds.
const PROBE_REPORT_CAP: usize = 64 * 1024;

/// What a probe learned about a video source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoProbe {
    /// Container duration in seconds, when the tool
    /// reports one (some containers do not).
    pub duration: Option<f64>,
    /// The first video stream's coded width.
    pub width: u32,
    /// The first video stream's coded height.
    pub height: u32,
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
    /// The video stream's coded width, for meta.
    pub width: u32,
    /// The video stream's coded height, for meta.
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

    /// Probe the source: duration, dimensions, and frame
    /// rate of its first video stream. `NotVideo` when
    /// the report names no usable video stream.
    pub fn probe(&self, video: &[u8]) -> Result<VideoProbe, FrameError> {
        let run = self.run(&["-nostdin", "-v", "info", "-i", "pipe:0"], video, 0)?;
        parse_probe(&run.stderr).ok_or_else(|| {
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
        let probe = self.probe(video)?;
        let mut timestamp = resolve_timestamp(selection, &probe)?;
        let step = frame_interval(probe.fps);
        for attempt in 0..=3 {
            let png = self.extract_at(video, timestamp)?;
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
    /// decides whether to step back); any other failure
    /// is `Failed`.
    fn extract_at(&self, video: &[u8], timestamp: f64) -> Result<Vec<u8>, FrameError> {
        let seek = format_seconds(timestamp);
        let run = self.run(
            &[
                "-nostdin",
                "-v",
                "info",
                "-i",
                "pipe:0",
                "-ss",
                &seek,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ],
            video,
            MAX_FRAME_PNG_BYTES,
        )?;
        if !run.status.success() {
            return Err(FrameError::Failed(format!(
                "video decoder exited with {}: {}",
                run.status,
                stderr_excerpt(&run.stderr)
            )));
        }
        Ok(run.stdout)
    }

    /// Run the program once under the standing bounds:
    /// the source bytes on stdin, stdout capped at
    /// `max_stdout` (past the cap the process group is
    /// killed and the run fails), stderr kept up to the
    /// report cap while the rest is drained so the tool
    /// never blocks on a full pipe, an empty environment,
    /// and the wall-clock timeout.
    fn run(
        &self,
        args: &[&str],
        stdin_bytes: &[u8],
        max_stdout: usize,
    ) -> Result<ToolRun, FrameError> {
        let mut cmd = Command::new(&self.program);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear()
            .current_dir(std::env::temp_dir());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let mut child = cmd.spawn().map_err(|e| {
            FrameError::Failed(format!(
                "could not start the video decoder {}: {e}",
                self.program.display()
            ))
        })?;
        let pid = child.id();

        let mut stdin_pipe = child.stdin.take().expect("stdin was piped");
        let input = stdin_bytes.to_vec();
        let writer = std::thread::spawn(move || {
            let _ = stdin_pipe.write_all(&input);
        });

        let mut stdout_pipe = child.stdout.take().expect("stdout was piped");
        let (out_tx, out_rx) = std::sync::mpsc::channel::<Result<Vec<u8>, String>>();
        let stdout_thread = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let mut chunk = [0u8; 64 * 1024];
            let result = loop {
                match stdout_pipe.read(&mut chunk) {
                    Ok(0) => break Ok(buf),
                    Ok(n) => {
                        buf.extend_from_slice(&chunk[..n]);
                        if buf.len() > max_stdout {
                            break Err(format!(
                                "the decoder's output passed the {max_stdout} byte cap"
                            ));
                        }
                    }
                    Err(e) => break Err(format!("reading the decoder's output failed: {e}")),
                }
            };
            let _ = out_tx.send(result);
        });

        let mut stderr_pipe = child.stderr.take().expect("stderr was piped");
        let (err_tx, err_rx) = std::sync::mpsc::channel::<String>();
        let stderr_thread = std::thread::spawn(move || {
            let mut kept = Vec::new();
            let mut chunk = [0u8; 16 * 1024];
            loop {
                match stderr_pipe.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if kept.len() < PROBE_REPORT_CAP {
                            let take = (PROBE_REPORT_CAP - kept.len()).min(n);
                            kept.extend_from_slice(&chunk[..take]);
                        }
                    }
                }
            }
            let _ = err_tx.send(String::from_utf8_lossy(&kept).into_owned());
        });

        let deadline = Instant::now() + self.timeout;
        // Wait for the exit status without ever blocking
        // past the deadline: a tool that closed its stdout
        // and then hung still gets killed on time.
        let wait_status =
            |child: &mut std::process::Child| -> Result<std::process::ExitStatus, FrameError> {
                loop {
                    match child.try_wait() {
                        Ok(Some(status)) => return Ok(status),
                        Ok(None) => {
                            if Instant::now() >= deadline {
                                kill_process_group(pid);
                                let _ = child.wait();
                                return Err(FrameError::Failed(format!(
                                    "video decoder timed out after {}s",
                                    self.timeout.as_secs()
                                )));
                            }
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(e) => {
                            return Err(FrameError::Failed(format!(
                                "waiting on the video decoder failed: {e}"
                            )));
                        }
                    }
                }
            };
        let (status, stdout) = loop {
            match out_rx.try_recv() {
                Ok(Ok(bytes)) => {
                    let status = wait_status(&mut child)?;
                    break (status, bytes);
                }
                Ok(Err(why)) => {
                    kill_process_group(pid);
                    let _ = child.wait();
                    return Err(FrameError::Failed(why));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    if Instant::now() >= deadline {
                        kill_process_group(pid);
                        let _ = child.wait();
                        return Err(FrameError::Failed(format!(
                            "video decoder timed out after {}s",
                            self.timeout.as_secs()
                        )));
                    }
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            // The process is gone; its stdout
                            // reader finishes at EOF. Give it
                            // a moment, then take what arrived.
                            match out_rx.recv_timeout(Duration::from_secs(2)) {
                                Ok(Ok(bytes)) => break (status, bytes),
                                Ok(Err(why)) => return Err(FrameError::Failed(why)),
                                Err(_) => break (status, Vec::new()),
                            }
                        }
                        Ok(None) => std::thread::sleep(Duration::from_millis(5)),
                        Err(e) => {
                            return Err(FrameError::Failed(format!(
                                "waiting on the video decoder failed: {e}"
                            )));
                        }
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    let status = wait_status(&mut child)?;
                    break (status, Vec::new());
                }
            }
        };
        let _ = writer.join();
        // The reader threads are deliberately never
        // joined (issue #263, the video.rs pattern): a
        // grandchild that inherited the pipes and holds
        // them open would block a join for its whole
        // lifetime, defeating the timeout above. stderr
        // is worth a short grace through its channel, no
        // more; the stdout result already arrived through
        // its own channel.
        let stderr = err_rx
            .recv_timeout(Duration::from_millis(200))
            .unwrap_or_default();
        drop(stdout_thread);
        drop(stderr_thread);
        Ok(ToolRun {
            status,
            stdout,
            stderr,
        })
    }
}

/// One finished tool run.
struct ToolRun {
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: String,
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

/// Parse the probe facts out of the tool's stderr
/// report: the `Duration:` line and the first
/// `Stream ...: Video:` line that carries dimensions.
/// Returns `None` when no usable video stream appears,
/// which is the not-a-video answer for every corrupt,
/// audio-only, or non-video source.
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

pub fn parse_probe(report: &str) -> Option<VideoProbe> {
    let mut duration = None;
    let mut stream: Option<(u32, u32, Option<f64>)> = None;
    let mut demuxers: Vec<&str> = Vec::new();
    for line in report.lines() {
        if demuxers.is_empty()
            && let Some(rest) = line.trim_start().strip_prefix("Input #")
            && let Some((_, after)) = rest.split_once(", ")
            && let Some((list, _)) = after.split_once(", from ")
        {
            demuxers = list.split(',').map(str::trim).collect();
        }
        if duration.is_none()
            && let Some(rest) = line.split("Duration: ").nth(1)
        {
            duration = parse_duration(rest);
        }
        // An attached picture (album art) is not the
        // video stream, even when it is the only one.
        if stream.is_none()
            && line.contains("Stream #")
            && line.contains(": Video: ")
            && !line.contains("(attached pic)")
        {
            stream = parse_video_stream(line);
        }
    }
    if !demuxers.is_empty() && demuxers.iter().all(|d| IMAGE_DEMUXERS.contains(d)) {
        return None;
    }
    let (width, height, fps) = stream?;
    Some(VideoProbe {
        duration,
        width,
        height,
        fps,
    })
}

/// Parse an `HH:MM:SS.cc` duration token (the text right
/// after `Duration: `, up to the comma). `N/A` and
/// anything malformed mean the container reports no
/// duration, which is a fact, not a failure.
fn parse_duration(rest: &str) -> Option<f64> {
    let token = rest.split(',').next()?.trim();
    let mut parts = token.split(':');
    let hours: f64 = parts.next()?.parse().ok()?;
    let minutes: f64 = parts.next()?.parse().ok()?;
    let seconds: f64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || hours < 0.0 || minutes < 0.0 || seconds < 0.0 {
        return None;
    }
    Some(hours * 3600.0 + minutes * 60.0 + seconds)
}

/// Parse one video stream line: the first comma
/// field whose leading token is exactly `<w>x<h>` (the
/// field usually continues with `[SAR ... DAR ...]`),
/// and the `<n> fps` field when present (with `<n> tbr`
/// as the fallback rate). Leading-token matching plus a
/// nonzero check keeps hex codec tags (`0x31637661`,
/// which would split as 0 by 31637661) from reading as
/// dimensions.
fn parse_video_stream(line: &str) -> Option<(u32, u32, Option<f64>)> {
    let after = line.split(": Video: ").nth(1)?;
    let mut dims = None;
    let mut fps = None;
    let mut tbr = None;
    for field in after.split(',') {
        let tokens: Vec<&str> = field.split_whitespace().collect();
        if dims.is_none()
            && let Some(first) = tokens.first()
            && let Some((w, h)) = parse_dims(first)
            && w > 0
            && h > 0
        {
            dims = Some((w, h));
        }
        if tokens.len() == 2 {
            match tokens[1] {
                "fps" => fps = tokens[0].parse::<f64>().ok().filter(|v| *v > 0.0),
                "tbr" => tbr = tokens[0].parse::<f64>().ok().filter(|v| *v > 0.0),
                _ => {}
            }
        }
    }
    let (width, height) = dims?;
    Some((width, height, fps.or(tbr)))
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

/// The tail of a tool's stderr, for error text: capped,
/// the same as the encoder's excerpts.
fn stderr_excerpt(stderr: &str) -> String {
    let tail: String = stderr
        .chars()
        .rev()
        .take(500)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    let tail = tail.trim();
    if tail.is_empty() {
        "no error output".to_string()
    } else {
        tail.to_string()
    }
}

/// Kill a runaway tool's whole process group (it may
/// have spawned helpers). Mirrors the encoder's
/// discipline exactly. The group kill is the whole
/// mechanism (issue #272): a descendant that detaches
/// (setsid, double-fork) escapes it by design of the
/// tool contract, which forbids daemonizing.
#[cfg(unix)]
fn kill_process_group(pid: u32) {
    if let Ok(pid) = i32::try_from(pid)
        && let Some(pid) = rustix::process::Pid::from_raw(pid)
    {
        let _ = rustix::process::kill_process_group(pid, rustix::process::Signal::KILL);
    }
}

#[cfg(not(unix))]
fn kill_process_group(_pid: u32) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe report the extractor parses, recorded
    /// from the reference tool on a real file.
    const MP4_REPORT: &str = "\
ffmpeg version 8.1.2 Copyright (c) 2000-2026 the FFmpeg developers
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'pipe:0':
  Metadata:
    major_brand     : isom
  Duration: 00:00:03.00, start: 0.000000, bitrate: N/A
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 64x48 [SAR 1:1 DAR 4:3], 3 kb/s, 10 fps, 10 tbr, 10240 tbn (default)
";

    const AUDIO_REPORT: &str = "\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'pipe:0':
  Duration: 00:00:02.00, start: 0.000000, bitrate: N/A
  Stream #0:0[0x1](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, mono, fltp, 69 kb/s (default)
";

    fn probe_of(duration: Option<f64>, fps: Option<f64>) -> VideoProbe {
        VideoProbe {
            duration,
            width: 64,
            height: 48,
            fps,
        }
    }

    #[test]
    fn probe_parses_the_reference_report() {
        let probe = parse_probe(MP4_REPORT).expect("a video stream parses");
        assert_eq!(probe.duration, Some(3.0));
        assert_eq!((probe.width, probe.height), (64, 48));
        assert_eq!(probe.fps, Some(10.0));
    }

    #[test]
    fn probe_finds_no_video_in_audio_only_or_junk() {
        // Audio-only: a duration and streams, but no video.
        assert!(parse_probe(AUDIO_REPORT).is_none());
        // A corrupt container: no input report at all.
        assert!(
            parse_probe(
                "[in#0 @ 0x1] Error opening input: Invalid data found when processing input\n"
            )
            .is_none()
        );
        assert!(parse_probe("").is_none());
    }

    #[test]
    fn probe_rejects_still_images_promoted_to_video() {
        // Issue #266: ffmpeg probes a JPEG embedded in a
        // PDF, a zip, or junk bytes through the raw mjpeg
        // demuxer and calls it a video stream. Recorded
        // from the reference tool on such a blob.
        let report = "\
Input #0, mjpeg, from 'pipe:0':
  Duration: N/A, bitrate: N/A
  Stream #0:0: Video: mjpeg (Baseline), yuvj420p(pc), 800x600, 25 fps, 1200k tbr
";
        assert!(parse_probe(report).is_none());
        // A plain JPEG probes through jpeg_pipe: also a
        // still, also refused.
        let report = "\
Input #0, jpeg_pipe, from 'pipe:0':
  Duration: N/A, bitrate: N/A
  Stream #0:0: Video: mjpeg (Baseline), yuvj420p(pc), 800x600, 25 fps, 25 tbr
";
        assert!(parse_probe(report).is_none());
        // But MJPEG is a real video codec inside a
        // container: an AVI of it parses normally.
        let report = "\
Input #0, avi, from 'pipe:0':
  Duration: 00:00:05.00, start: 0.000000, bitrate: N/A
  Stream #0:0: Video: mjpeg (Baseline), yuvj420p(pc), 640x480, 25 fps, 25 tbr
";
        let probe = parse_probe(report).expect("container MJPEG is a video");
        assert_eq!((probe.width, probe.height), (640, 480));
        // An attached picture alone is not a video.
        let report = "\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'pipe:0':
  Duration: 00:00:02.00, start: 0.000000, bitrate: N/A
  Stream #0:0: Audio: aac (LC), 44100 Hz, mono, fltp, 69 kb/s (default)
  Stream #0:1: Video: mjpeg (Baseline), yuvj420p(pc), 500x500, 90k tbr (attached pic)
";
        assert!(parse_probe(report).is_none());
    }

    #[test]
    fn probe_tolerates_an_unreported_duration() {
        let report = "\
Input #0, matroska,webm, from 'pipe:0':
  Duration: N/A, start: 0.000000, bitrate: N/A
  Stream #0:0: Video: vp9 (Profile 0), yuv420p(tv), 320x240, SAR 1:1 DAR 4:3, 25 fps, 25 tbr (default)
";
        let probe = parse_probe(report).expect("a video stream parses");
        assert_eq!(probe.duration, None);
        assert_eq!((probe.width, probe.height), (320, 240));
        assert_eq!(probe.fps, Some(25.0));
    }

    #[test]
    fn probe_does_not_read_codec_tags_as_dimensions() {
        // The hex tag 0x31637661 must not parse as 0x3 or
        // similar; only the whole-field 64x48 counts.
        let report = "\
  Stream #0:0: Video: h264 (avc1 / 0x31637661), yuv420p, 64x48, 10 fps
";
        let probe = parse_probe(report).unwrap();
        assert_eq!((probe.width, probe.height), (64, 48));
    }

    #[test]
    fn resolve_clamps_past_the_end_to_the_last_frame() {
        let probe = probe_of(Some(3.0), Some(10.0));
        assert_eq!(
            resolve_timestamp(FrameSelection::Poster, &probe).unwrap(),
            0.0
        );
        assert_eq!(
            resolve_timestamp(FrameSelection::Seconds(1.5), &probe).unwrap(),
            1.5
        );
        // 100% and a wild seconds value both clamp to
        // duration - one frame interval (2.9 at 10 fps).
        let clamped = resolve_timestamp(FrameSelection::Percent(100.0), &probe).unwrap();
        assert!((clamped - 2.9).abs() < 1e-9, "{clamped}");
        let clamped = resolve_timestamp(FrameSelection::Seconds(999.0), &probe).unwrap();
        assert!((clamped - 2.9).abs() < 1e-9, "{clamped}");
        // Mid percentages resolve against the duration.
        assert_eq!(
            resolve_timestamp(FrameSelection::Percent(50.0), &probe).unwrap(),
            1.5
        );
    }

    #[test]
    fn resolve_a_percentage_needs_a_duration() {
        let probe = probe_of(None, Some(25.0));
        assert_eq!(
            resolve_timestamp(FrameSelection::Seconds(2.0), &probe).unwrap(),
            2.0
        );
        let err = resolve_timestamp(FrameSelection::Percent(50.0), &probe).unwrap_err();
        assert!(matches!(err, FrameError::Unsupported(_)), "{err}");
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "crabcrop-frames-test-{}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fake_tool(name: &str, script: &str) -> PathBuf {
        let dir = temp_dir(name);
        let path = dir.join("fake-decoder");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(script.as_bytes()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    /// A stand-in decoder with the reference tool's shape:
    /// the report on stderr and exit 1 for a probe, a PNG
    /// on stdout for an extraction.
    const FAKE_OK: &str = "#!/bin/sh\n\
cat > /dev/null\n\
for a in \"$@\"; do\n\
  if [ \"$a\" = \"image2pipe\" ]; then\n\
    printf '\\211PNG\\015\\012\\032\\012fakepng'\n\
    exit 0\n\
  fi\n\
done\n\
printf '%s\\n' 'Input #0, mov,mp4, from pipe:0:' '  Duration: 00:00:03.00, start: 0.000000, bitrate: N/A' '  Stream #0:0: Video: h264, yuv420p, 64x48, 10 fps, 10 tbr' >&2\n\
exit 1\n";

    #[test]
    fn extractor_validates_the_program_at_construction() {
        let err = VideoFrameExtractor::new(
            PathBuf::from("/nonexistent/no-such-decoder"),
            Duration::from_secs(5),
        )
        .unwrap_err();
        assert!(matches!(err, FrameError::Failed(_)));
        assert!(err.to_string().contains("does not exist"), "{err}");
    }

    #[test]
    fn extract_runs_probe_then_extraction() {
        let program = fake_tool("ok", FAKE_OK);
        let extractor = VideoFrameExtractor::new(program, Duration::from_secs(10)).unwrap();
        let frame = extractor
            .extract_frame(b"video bytes", FrameSelection::Seconds(1.5))
            .unwrap();
        assert!(frame.png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(frame.duration, Some(3.0));
        assert_eq!((frame.width, frame.height), (64, 48));
        assert_eq!(frame.timestamp, 1.5);
        // A clamped selection reports the clamped time.
        let frame = extractor
            .extract_frame(b"video bytes", FrameSelection::Percent(100.0))
            .unwrap();
        assert!((frame.timestamp - 2.9).abs() < 1e-9, "{}", frame.timestamp);
    }

    #[test]
    fn extract_maps_a_streamless_probe_to_not_video() {
        let program = fake_tool(
            "junk",
            "#!/bin/sh\ncat > /dev/null\nprintf 'not a video at all\\n' >&2\nexit 1\n",
        );
        let extractor = VideoFrameExtractor::new(program, Duration::from_secs(10)).unwrap();
        let err = extractor
            .extract_frame(b"junk", FrameSelection::Poster)
            .unwrap_err();
        assert!(matches!(err, FrameError::NotVideo(_)), "{err}");
    }

    #[test]
    fn extract_maps_tool_failure_and_empty_output() {
        // Nonzero exit on the extraction run.
        let program = fake_tool(
            "failing",
            "#!/bin/sh\n\
cat > /dev/null\n\
for a in \"$@\"; do\n\
  if [ \"$a\" = \"image2pipe\" ]; then echo 'boom on stderr' >&2; exit 3; fi\n\
done\n\
printf '%s\\n' '  Duration: 00:00:03.00, start: 0.000000' '  Stream #0:0: Video: h264, yuv420p, 64x48, 10 fps' >&2\n\
exit 1\n",
        );
        let extractor = VideoFrameExtractor::new(program, Duration::from_secs(10)).unwrap();
        let err = extractor
            .extract_frame(b"v", FrameSelection::Poster)
            .unwrap_err();
        assert!(matches!(err, FrameError::Failed(_)), "{err}");
        assert!(err.to_string().contains("boom on stderr"), "{err}");

        // A clean run that yields no frame at any stepped
        // timestamp is a failure, not an empty image.
        let program = fake_tool(
            "empty",
            "#!/bin/sh\n\
cat > /dev/null\n\
for a in \"$@\"; do\n\
  if [ \"$a\" = \"image2pipe\" ]; then exit 0; fi\n\
done\n\
printf '%s\\n' '  Duration: 00:00:03.00, start: 0.000000' '  Stream #0:0: Video: h264, yuv420p, 64x48, 10 fps' >&2\n\
exit 1\n",
        );
        let extractor = VideoFrameExtractor::new(program, Duration::from_secs(10)).unwrap();
        let err = extractor
            .extract_frame(b"v", FrameSelection::Seconds(2.0))
            .unwrap_err();
        assert!(matches!(err, FrameError::Failed(_)), "{err}");
        assert!(err.to_string().contains("produced no frame"), "{err}");
    }

    #[test]
    fn extract_times_out_on_a_hung_decoder() {
        let program = fake_tool("sleeper", "#!/bin/sh\nsleep 30\n");
        let extractor = VideoFrameExtractor::new(program, Duration::from_secs(1)).unwrap();
        let started = Instant::now();
        let err = extractor
            .extract_frame(b"v", FrameSelection::Poster)
            .unwrap_err();
        assert!(matches!(err, FrameError::Failed(_)), "{err}");
        assert!(err.to_string().contains("timed out"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    /// End to end against the real reference tool when it
    /// is on PATH (CI installs it; a machine without it
    /// skips). The generated clip is a solid color per
    /// second, red, green, blue, so the selected frame is
    /// provable from its pixels.
    /// The reference tool's absolute path, resolved from
    /// PATH (the extractor validates a real file path,
    /// it does not search).
    fn ffmpeg_path() -> Option<PathBuf> {
        let path = std::env::var_os("PATH")?;
        std::env::split_paths(&path)
            .map(|dir| dir.join("ffmpeg"))
            .find(|candidate| candidate.is_file())
    }

    #[test]
    fn real_tool_extracts_the_selected_frames() {
        let Some(ffmpeg) = ffmpeg_path() else {
            return;
        };
        let dir = temp_dir("real");
        let clip = dir.join("clip.mp4");
        let status = Command::new(&ffmpeg)
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=red:size=64x48:rate=10:duration=1",
                "-f",
                "lavfi",
                "-i",
                "color=green:size=64x48:rate=10:duration=1",
                "-f",
                "lavfi",
                "-i",
                "color=blue:size=64x48:rate=10:duration=1",
                "-filter_complex",
                "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
                "-map",
                "[v]",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&clip)
            .status()
            .unwrap();
        assert!(status.success());
        let bytes = std::fs::read(&clip).unwrap();
        let extractor = VideoFrameExtractor::new(ffmpeg, Duration::from_secs(30)).unwrap();

        let probe = extractor.probe(&bytes).unwrap();
        assert_eq!(probe.duration, Some(3.0));
        assert_eq!((probe.width, probe.height), (64, 48));

        let center = |frame: &ExtractedFrame| {
            let img = image::load_from_memory(&frame.png).unwrap().to_rgb8();
            let p = img.get_pixel(32, 24);
            (p[0], p[1], p[2])
        };
        let (r, g, b) = center(
            &extractor
                .extract_frame(&bytes, FrameSelection::Poster)
                .unwrap(),
        );
        assert!(
            r > 200 && g < 60 && b < 60,
            "poster is red: {:?}",
            (r, g, b)
        );
        let (r, g, b) = center(
            &extractor
                .extract_frame(&bytes, FrameSelection::Seconds(1.5))
                .unwrap(),
        );
        assert!(
            g > 100 && r < 60 && b < 60,
            "1.5s is green: {:?}",
            (r, g, b)
        );
        let (r, g, b) = center(
            &extractor
                .extract_frame(&bytes, FrameSelection::Seconds(2.5))
                .unwrap(),
        );
        assert!(b > 200 && r < 60 && g < 60, "2.5s is blue: {:?}", (r, g, b));
        let frame = extractor
            .extract_frame(&bytes, FrameSelection::Percent(50.0))
            .unwrap();
        let (r, g, b) = center(&frame);
        assert!(g > 100 && r < 60 && b < 60, "50% is green: {:?}", (r, g, b));
        assert_eq!(frame.timestamp, 1.5);
        // Past the end clamps to the last (blue) frame.
        let frame = extractor
            .extract_frame(&bytes, FrameSelection::Seconds(999.0))
            .unwrap();
        let (r, g, b) = center(&frame);
        assert!(
            b > 200 && r < 60 && g < 60,
            "clamped is blue: {:?}",
            (r, g, b)
        );
        assert!((frame.timestamp - 2.9).abs() < 0.01, "{}", frame.timestamp);
    }
}
