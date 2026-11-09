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
//! - Probe: `-nostdin -v info -protocol_whitelist fd
//!   -format_whitelist <containers> -dump_separator <token>
//!   -i fd:`. No output target is named, so the tool
//!   prints its input report (duration, stream list) to
//!   stderr and exits nonzero; the probe reads its facts
//!   from that report and treats the exit status as
//!   meaningless. The report also prints the source's own
//!   metadata, so the token, random per probe, marks the
//!   real stream lines (see [`parse_probe`]). A source with
//!   no video stream in the report (a non-video, an
//!   audio-only file, a corrupt container) is
//!   [`FrameError::NotVideo`].
//! - Extract: `-nostdin -v info -protocol_whitelist fd
//!   -format_whitelist <containers> -i fd: -ss <t>
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
//! The whitelists keep the tool inside the loader's
//! policy (issue #361): the bytes on stdin are the only
//! input it can reach (`fd` opens a descriptor the tool
//! already holds, and refuses a descriptor named in a URL),
//! so container indirection (a manifest naming a `file://`
//! or network target) has nowhere to go.
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
    /// the report names no usable video stream. The
    /// extraction path probes through `probe_until` (the
    /// shared deadline); this spelling stays as the
    /// single-probe entry the test suite exercises.
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
        let run = self.run(
            &[
                "-nostdin",
                "-v",
                "info",
                "-protocol_whitelist",
                "fd",
                "-format_whitelist",
                FORMAT_WHITELIST,
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
        let mut timestamp = resolve_timestamp(selection, &probe)?;
        let step = frame_interval(probe.fps);
        for attempt in 0..=3 {
            let png = self.extract_at(&input, timestamp, deadline)?;
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
    fn extract_at(
        &self,
        input: &File,
        timestamp: f64,
        deadline: Instant,
    ) -> Result<Vec<u8>, FrameError> {
        let seek = format_seconds(timestamp);
        let run = self.run(
            &[
                "-nostdin",
                "-v",
                "info",
                "-protocol_whitelist",
                "fd",
                "-format_whitelist",
                FORMAT_WHITELIST,
                "-i",
                "fd:",
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
            input,
            MAX_FRAME_PNG_BYTES,
            deadline,
        )?;
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
    let (width, height, fps) = lines[duration_at + 1..]
        .iter()
        .find_map(|line| parse_video_stream(line, separator))?;
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
/// `<w>x<h>` (the field usually continues with
/// `[SAR ... DAR ...]`), and the `<n> fps` field when
/// present (with `<n> tbr` as the fallback rate).
/// Leading-token matching plus a nonzero check keeps hex
/// codec tags (`0x31637661`, which would split as 0 by
/// 31637661) from reading as dimensions.
fn parse_video_stream(line: &str, separator: &str) -> Option<(u32, u32, Option<f64>)> {
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
                "fps" => fps = rate(tokens[0]),
                "tbr" => tbr = rate(tokens[0]),
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

#[cfg(test)]
mod tests;
