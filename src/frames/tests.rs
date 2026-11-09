use super::*;
use std::process::Command;

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
    let probe = parse_probe(MP4_REPORT, DEFAULT_SEP).expect("a video stream parses");
    assert_eq!(probe.duration, Some(3.0));
    assert_eq!((probe.width, probe.height), (64, 48));
    assert_eq!(probe.fps, Some(10.0));
}

#[test]
fn probe_finds_no_video_in_audio_only_or_junk() {
    // Audio-only: a duration and streams, but no video.
    assert!(parse_probe(AUDIO_REPORT, DEFAULT_SEP).is_none());
    // A corrupt container: no input report at all.
    assert!(
        parse_probe(
            "[in#0 @ 0x1] Error opening input: Invalid data found when processing input\n",
            DEFAULT_SEP
        )
        .is_none()
    );
    assert!(parse_probe("", DEFAULT_SEP).is_none());
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
    assert!(parse_probe(report, DEFAULT_SEP).is_none());
    // A plain JPEG probes through jpeg_pipe: also a
    // still, also refused.
    let report = "\
Input #0, jpeg_pipe, from 'pipe:0':
  Duration: N/A, bitrate: N/A
  Stream #0:0: Video: mjpeg (Baseline), yuvj420p(pc), 800x600, 25 fps, 25 tbr
";
    assert!(parse_probe(report, DEFAULT_SEP).is_none());
    // But MJPEG is a real video codec inside a
    // container: an AVI of it parses normally.
    let report = "\
Input #0, avi, from 'pipe:0':
  Duration: 00:00:05.00, start: 0.000000, bitrate: N/A
  Stream #0:0: Video: mjpeg (Baseline), yuvj420p(pc), 640x480, 25 fps, 25 tbr
";
    let probe = parse_probe(report, DEFAULT_SEP).expect("container MJPEG is a video");
    assert_eq!((probe.width, probe.height), (640, 480));
    // An attached picture alone is not a video.
    let report = "\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'pipe:0':
  Duration: 00:00:02.00, start: 0.000000, bitrate: N/A
  Stream #0:0: Audio: aac (LC), 44100 Hz, mono, fltp, 69 kb/s (default)
  Stream #0:1: Video: mjpeg (Baseline), yuvj420p(pc), 500x500, 90k tbr (attached pic)
";
    assert!(parse_probe(report, DEFAULT_SEP).is_none());
}

#[test]
fn probe_tolerates_an_unreported_duration() {
    let report = "\
Input #0, matroska,webm, from 'pipe:0':
  Duration: N/A, start: 0.000000, bitrate: N/A
  Stream #0:0: Video: vp9 (Profile 0), yuv420p(tv), 320x240, SAR 1:1 DAR 4:3, 25 fps, 25 tbr (default)
";
    let probe = parse_probe(report, DEFAULT_SEP).expect("a video stream parses");
    assert_eq!(probe.duration, None);
    assert_eq!((probe.width, probe.height), (320, 240));
    assert_eq!(probe.fps, Some(25.0));
}

#[test]
fn probe_does_not_read_codec_tags_as_dimensions() {
    // The hex tag 0x31637661 must not parse as 0x3 or
    // similar; only the whole-field 64x48 counts.
    let report = "\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'pipe:0':
  Duration: 00:00:03.00, start: 0.000000, bitrate: N/A
  Stream #0:0: Video: h264 (avc1 / 0x31637661), yuv420p, 64x48, 10 fps
";
    let probe = parse_probe(report, DEFAULT_SEP).unwrap();
    assert_eq!((probe.width, probe.height), (64, 48));
}

/// The field separator the hostile reports below were
/// recorded with (`-dump_separator`), standing in for
/// the probe's random one.
const SEP: &str = " ~0123456789abcdef~ ";

/// The tool's own default separator, which the honest
/// recorded reports above use and a planted line can
/// imitate.
const DEFAULT_SEP: &str = ", ";

/// The report the reference tool (6.1) printed for a 2 s
/// 64x48 10 fps MP4 whose container metadata carries the
/// given lines, in the layout the metadata block uses,
/// probed with `SEP`.
fn report_with_metadata(metadata: &str) -> String {
    format!(
        "\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'pipe:0':
  Metadata:
    major_brand     : isom
{metadata}
    encoder         : Lavf60.16.100
  Duration: 00:00:02.00, start: 0.000000, bitrate: N/A
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661){SEP}yuv420p(progressive), 64x48 [SAR 1:1 DAR 4:3], 4 kb/s{SEP}10 fps, 10 tbr, 10240 tbn (default)
    Metadata:
      handler_name    : VideoHandler
At least one output file must be specified
"
    )
}

/// The report the reference tool (6.1) printed for a 2 s
/// Matroska clip, probed with `SEP`, with the given
/// stream lines.
fn mkv_report(streams: &str) -> String {
    format!(
        "\
Input #0, matroska,webm, from 'pipe:0':
  Metadata:
    ENCODER         : Lavf60.16.100
  Duration: 00:00:02.00, start: 0.000000, bitrate: N/A
{streams}
At least one output file must be specified
"
    )
}

#[test]
fn probe_ignores_metadata_lines_shaped_like_facts() {
    // Issue #402: container metadata prints above the
    // real Duration line, and text there that looks like
    // a fact must not steer the parse: finite, NaN, and
    // infinite durations and a forged stream line in a
    // value, a long key that starts like a fact (keys pad
    // to 16 columns and print whole), a multi-line value
    // (its lines continue with `: `), and a key holding
    // newlines that plants a whole stream line at the
    // report's own indent (an MP4 freeform key written
    // with `-movflags use_metadata_tags`). Recorded from
    // the reference tool on clips made with `-metadata`.
    for metadata in [
        "    comment         : Duration: 00:00:00.05, x",
        "    title           : Duration: NaN:00:00.00, x",
        "    title           : Duration: inf:00:00.00, x",
        "    title           : Stream #0:0: Video: h264, 4000x3000, 0.001 fps",
        "    Duration: 00:00:00.05, x: v",
        "    title           : line one
                    :   Duration: 00:00:00.05, start
                    :   Stream #0:0: Video: h264, 4000x3000, 0.001 fps",
        "    k
  Stream #0:0: Video: h264, yuv420p, 4000x3000, 0.001 fps
  z: v",
    ] {
        let probe = parse_probe(&report_with_metadata(metadata), SEP)
            .unwrap_or_else(|| panic!("the real facts parse under {metadata:?}"));
        assert_eq!(probe.duration, Some(2.0), "{metadata:?}");
        assert_eq!((probe.width, probe.height), (64, 48), "{metadata:?}");
        assert_eq!(probe.fps, Some(10.0), "{metadata:?}");
    }
}

#[test]
fn probe_refuses_a_planted_duration_line() {
    // Issue #402: a key holding newlines plants a whole
    // Duration line at the report's own indent (recorded
    // from the reference tool, MP4 freeform key). The
    // real Duration line still prints after it, so two
    // Duration lines are a forged report, refused
    // whatever the planted one says.
    for planted in [
        "    k\n  Duration: 00:00:00.05, start: 0.000000, bitrate: N/A\n  z: v",
        "    k\n  Duration: N/A, start: 0.000000, bitrate: N/A\n  z: v",
        "    k\n  Duration: 00:00:02.00, start: 0.000000, bitrate: N/A\n  z: v",
    ] {
        assert!(
            parse_probe(&report_with_metadata(planted), SEP).is_none(),
            "{planted:?}"
        );
    }
}

#[test]
fn probe_reads_streams_only_through_the_separator() {
    // Issue #402: a Matroska language tag prints raw
    // inside its own stream line, newlines included. It
    // can put a forged type marker ahead of the real one,
    // rewrite the whole first line of the real stream and
    // push the real codec text onto a line of its own, or
    // move that text under a fresh stream number; no
    // layout rule tells the last two apart from a real
    // report. The probe's random separator does: the tool
    // prints it inside real codec descriptions alone. The
    // stream lines are the reference tool's (6.1), for a
    // 64x48 10 fps clip whose video stream carries the
    // language tag shown.
    let real = format!(
        "h264 (High){SEP}yuv420p(progressive), 64x48 [SAR 1:1 DAR 4:3]{SEP}10 fps, 10 tbr, 1k tbn"
    );
    for streams in [
        format!("  Stream #0:0(x): Video: h264, yuv420p, 4000x3000, 0.001 fps (): Video: {real}"),
        format!(
            "  Stream #0:0(x): Video: h264, yuv420p, 4000x3000, 0.001 fps\n    z): Video: {real}"
        ),
        format!(
            "  Stream #0:0(x): Video: h264, yuv420p, 4000x3000, 0.001 fps\n  Stream #0:1(eng): Video: {real}"
        ),
    ] {
        let probe = parse_probe(&mkv_report(&streams), SEP)
            .unwrap_or_else(|| panic!("the real stream parses under {streams:?}"));
        assert_eq!(probe.duration, Some(2.0), "{streams:?}");
        assert_eq!((probe.width, probe.height), (64, 48), "{streams:?}");
        assert_eq!(probe.fps, Some(10.0), "{streams:?}");
    }
    // A language tag cannot mark the real stream as an
    // attached picture: the disposition is read only
    // after the real marker.
    let streams = format!("  Stream #0:0((attached pic)): Video: {real}");
    assert!(parse_probe(&mkv_report(&streams), SEP).is_some());
    // An audio-only source whose language tag plants a
    // video stream line still has no video stream.
    let streams = format!(
        "  Stream #0:0(eng\n  Stream #0:5: Video: h264, yuv420p, 4000x3000, 0.001 fps\n  x): Audio: vorbis{SEP}44100 Hz, mono, fltp"
    );
    assert!(parse_probe(&mkv_report(&streams), SEP).is_none());
    // A report without the separator (a tool that ignored
    // `-dump_separator`, or planted lines alone) names no
    // stream the probe can take at its word.
    assert!(parse_probe(MP4_REPORT, SEP).is_none());

    // Real layouts still parse: a stream listed under two
    // programs prints twice (recorded, mpegts), and colour
    // details that need a break put the separator before
    // the dimensions too (recorded from 6.1; current
    // FFmpeg's source prints the same).
    let report = format!(
        "\
Input #0, mpegts, from 'pipe:0':
  Duration: N/A, start: 1.600000, bitrate: N/A
  Program 1
    Metadata:
      service_name    : One
  Stream #0:0[0x100]: Video: h264 (High) ([27][0][0][0] / 0x001B){SEP}yuv420p(progressive), 64x48 [SAR 1:1 DAR 4:3]{SEP}10 fps, 10 tbr, 90k tbn
  Program 2
    Metadata:
      service_name    : Two
  Stream #0:0[0x100]: Video: h264 (High) ([27][0][0][0] / 0x001B){SEP}yuv420p(progressive), 64x48 [SAR 1:1 DAR 4:3]{SEP}10 fps, 10 tbr, 90k tbn
"
    );
    let probe = parse_probe(&report, SEP).expect("a repeated stream parses");
    assert_eq!((probe.width, probe.height), (64, 48));
    assert_eq!(probe.duration, None);
    let streams = format!(
        "  Stream #0:0: Video: h264 (High){SEP}yuv420p(tv, bt2020nc/bt2020/smpte2084, progressive){SEP}64x48 [SAR 1:1 DAR 4:3]{SEP}24 fps, 24 tbr, 1k tbn"
    );
    let probe = parse_probe(&mkv_report(&streams), SEP).expect("an HDR stream parses");
    assert_eq!((probe.width, probe.height), (64, 48));
    assert_eq!(probe.fps, Some(24.0));
    // And album art is still not a video (recorded, M4A).
    let report = format!(
        "\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'pipe:0':
  Duration: 00:00:01.02, start: 0.000000, bitrate: N/A
  Stream #0:0[0x1](und): Audio: aac (LC) (mp4a / 0x6134706D){SEP}44100 Hz, mono, fltp, 70 kb/s (default)
  Stream #0:1[0x0]: Video: mjpeg (Baseline){SEP}yuvj420p(pc, bt470bg/unknown/unknown){SEP}50x50 [SAR 1:1 DAR 1:1]{SEP}90k tbr, 90k tbn (attached pic)
"
    );
    assert!(parse_probe(&report, SEP).is_none());
}

#[test]
fn probe_separators_are_fresh_and_unforgeable() {
    // Issue #402: the separator is the one thing in the
    // report a source cannot write, so it must differ per
    // probe and never be the tool's default.
    let first = probe_separator();
    let second = probe_separator();
    assert_ne!(first, second);
    for separator in [&first, &second] {
        let token = separator
            .strip_prefix(" ~")
            .and_then(|rest| rest.strip_suffix("~ "))
            .unwrap_or_else(|| panic!("{separator:?}"));
        assert_eq!(token.len(), 16, "{separator:?}");
        assert!(
            token.bytes().all(|b| b.is_ascii_hexdigit()),
            "{separator:?}"
        );
    }
}

#[test]
fn probe_rejects_non_finite_and_malformed_numbers() {
    // Issue #402: `str::parse::<f64>` takes NaN, inf,
    // signs, and exponents; none of them is a duration
    // the report prints, so each reads as no duration.
    for token in [
        "NaN:00:00.00",
        "inf:00:00.00",
        "00:00:NaN",
        "00:00:inf",
        "00:00:infinity",
        "-1:00:00.00",
        "+1:00:00.00",
        "00:00:1e3",
        "00:00:.5",
        "00:00:5.",
        "00::05.00",
        "00:00:05.00:00",
        "N/A",
        "",
    ] {
        assert_eq!(
            parse_duration(&format!("{token}, start: 0")),
            None,
            "{token}"
        );
    }
    // Digits that overflow to infinity are refused too.
    let huge = format!("{}:00:00.00", "9".repeat(400));
    assert_eq!(parse_duration(&huge), None);
    // The real spellings still parse, past 99 hours too.
    assert_eq!(parse_duration("00:00:02.00, start: 0"), Some(2.0));
    assert_eq!(parse_duration("123:04:05.50, start: 0"), Some(443045.5));
    assert_eq!(parse_duration("00:00:07, start: 0"), Some(7.0));

    // A frame rate must be finite and positive; the tbr
    // fallback follows the same rule.
    for rate in ["inf fps", "NaN fps", "0 fps", "-10 fps", "infinity fps"] {
        let line = format!("  Stream #0:0: Video: h264{SEP}yuv420p, 64x48{SEP}{rate}, inf tbr");
        assert_eq!(
            parse_video_stream(&line, SEP),
            Some((64, 48, None)),
            "{rate}"
        );
    }
    let line = format!("  Stream #0:0: Video: h264{SEP}yuv420p, 64x48{SEP}inf fps, 25 tbr");
    assert_eq!(parse_video_stream(&line, SEP), Some((64, 48, Some(25.0))));
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

fn temp_dir(name: &str) -> crate::test_support::ScratchDir {
    crate::test_support::scratch_dir(&format!("frames-test-{name}"))
}

/// The fake tool's path, and the directory it lives in.
fn fake_tool(name: &str, script: &str) -> (crate::test_support::ScratchDir, PathBuf) {
    let dir = temp_dir(name);
    let path = dir.join("fake-decoder");
    crate::test_support::write_script(&path, script);
    (dir, path)
}

/// A stand-in decoder with the reference tool's shape:
/// the report on stderr and exit 1 for a probe, a PNG
/// on stdout for an extraction. Like the reference tool,
/// it prints the probe's `-dump_separator` after the
/// codec name (issue #402).
const FAKE_OK: &str = "#!/bin/sh\n\
cat > /dev/null\n\
sep=', '\n\
prev=\n\
for a in \"$@\"; do\n\
  if [ \"$a\" = \"image2pipe\" ]; then\n\
    printf '\\211PNG\\015\\012\\032\\012fakepng'\n\
    exit 0\n\
  fi\n\
  if [ \"$prev\" = -dump_separator ]; then sep=\"$a\"; fi\n\
  prev=\"$a\"\n\
done\n\
printf '%s\\n' 'Input #0, mov,mp4, from pipe:0:' '  Duration: 00:00:03.00, start: 0.000000, bitrate: N/A' '  Stream #0:0: Video: h264, yuv420p, 64x48, 10 fps, 10 tbr' | sed \"/^  Stream #/s/, /$sep/\" >&2\n\
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
    let (_tool, program) = fake_tool("ok", FAKE_OK);
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
fn the_probe_parses_a_report_that_arrives_late() {
    // Issue #566: the probe reads its report from stderr,
    // which the run gave up on 200 ms after the tool
    // exited, and an empty report reads as no video
    // stream. Held back past that, the report still
    // arrives and parses.
    let _hold = crate::toolrun::hold_readers(Duration::ZERO, Duration::from_millis(500));
    let (_tool, program) = fake_tool("late-report", FAKE_OK);
    let extractor = VideoFrameExtractor::new(program, Duration::from_secs(10)).unwrap();
    let frame = extractor
        .extract_frame(b"video bytes", FrameSelection::Seconds(1.5))
        .unwrap();
    assert_eq!(frame.duration, Some(3.0));
    assert_eq!((frame.width, frame.height), (64, 48));
    assert!(frame.png.starts_with(b"\x89PNG\r\n\x1a\n"));
}

#[test]
fn runs_carry_the_input_whitelists() {
    // Issue #361: every run (the probe and the
    // extraction) restricts the tool to the fd protocol
    // on its stdin (issue #411) and the accepted
    // container formats, so
    // a manifest inside the source bytes cannot point
    // it at inputs the loader never validated. The
    // stand-in tool records each invocation's argv, one
    // line per run, and otherwise behaves like FAKE_OK.
    let records = temp_dir("whitelists");
    let record = records.join("argv.log");
    let script = format!(
        "#!/bin/sh\ncat > /dev/null\necho \"$@\" >> '{}'\nsep=', '\nprev=\nfor a in \"$@\"; do\n  if [ \"$a\" = \"image2pipe\" ]; then\n    printf '\\211PNG\\015\\012\\032\\012fakepng'\n    exit 0\n  fi\n  if [ \"$prev\" = -dump_separator ]; then sep=\"$a\"; fi\n  prev=\"$a\"\ndone\nprintf '%s\\n' 'Input #0, mov,mp4, from pipe:0:' '  Duration: 00:00:03.00, start: 0.000000, bitrate: N/A' '  Stream #0:0: Video: h264, yuv420p, 64x48, 10 fps, 10 tbr' | sed \"/^  Stream #/s/, /$sep/\" >&2\nexit 1\n",
        record.display()
    );
    let (_tool, program) = fake_tool("whitelists", &script);
    let extractor = VideoFrameExtractor::new(program, Duration::from_secs(10)).unwrap();
    let frame = extractor
        .extract_frame(b"video bytes", FrameSelection::Poster)
        .unwrap();
    assert!(frame.png.starts_with(b"\x89PNG\r\n\x1a\n"));
    let recorded = std::fs::read_to_string(&record).unwrap();
    let lines: Vec<&str> = recorded.lines().collect();
    assert_eq!(lines.len(), 2, "{recorded}");
    for line in &lines {
        assert!(line.contains("-protocol_whitelist fd "), "{line}");
        let argv: Vec<&str> = line.split_whitespace().collect();
        assert!(argv.windows(2).any(|w| w == ["-i", "fd:"]), "{line}");
        assert!(!line.contains("pipe:0"), "{line}");
        assert!(
            line.contains(&format!("-format_whitelist {FORMAT_WHITELIST}")),
            "{line}"
        );
    }
    // Issue #402: the probe, and only the probe, asks for
    // its random field separator, which this stand-in
    // echoes the way the reference tool does.
    assert!(lines[0].contains("-dump_separator  ~"), "{}", lines[0]);
    assert!(!lines[1].contains("-dump_separator"), "{}", lines[1]);
}

#[cfg(target_os = "linux")]
#[test]
fn runs_read_a_seekable_staged_copy() {
    // Issue #411: every run gets the source as a regular
    // file on stdin (which the fd protocol can seek),
    // already unlinked, and each run reads it from the
    // start, though the runs share one staged file. The
    // stand-in records, per run, how many bytes it read,
    // whether stdin is a regular file, and what the
    // descriptor points at. Its probe report carries
    // the run's dump separator the way the reference
    // tool prints it (issue #402).
    let records = temp_dir("staged");
    let record = records.join("stdin.log");
    let script = format!(
        "#!/bin/sh\n\
n=$(cat | wc -c)\n\
if [ -f /dev/stdin ]; then kind=file; else kind=other; fi\n\
echo \"$n $kind $(readlink /proc/self/fd/0)\" >> '{}'\n\
sep=', '\n\
prev=\n\
for a in \"$@\"; do\n  if [ \"$a\" = \"image2pipe\" ]; then\n    printf '\\211PNG\\015\\012\\032\\012fakepng'\n    exit 0\n  fi\n  if [ \"$prev\" = -dump_separator ]; then sep=\"$a\"; fi\n  prev=\"$a\"\ndone\n\
printf '%s\\n' 'Input #0, mov,mp4, from fd:' '  Duration: 00:00:03.00, start: 0.000000, bitrate: N/A' '  Stream #0:0: Video: h264, yuv420p, 64x48, 10 fps, 10 tbr' | sed \"/^  Stream #/s/, /$sep/\" >&2\n\
exit 1\n",
        record.display()
    );
    let (_tool, program) = fake_tool("staged", &script);
    let extractor = VideoFrameExtractor::new(program, Duration::from_secs(10)).unwrap();
    let video = vec![7u8; 100_000];
    let frame = extractor
        .extract_frame(&video, FrameSelection::Seconds(1.5))
        .unwrap();
    assert!(frame.png.starts_with(b"\x89PNG\r\n\x1a\n"));
    let recorded = std::fs::read_to_string(&record).unwrap();
    let lines: Vec<&str> = recorded.lines().collect();
    assert_eq!(lines.len(), 2, "{recorded}");
    for line in lines {
        let mut fields = line.splitn(3, ' ');
        assert_eq!(fields.next(), Some("100000"), "{line}");
        assert_eq!(fields.next(), Some("file"), "{line}");
        let target = fields.next().unwrap_or_default();
        assert!(target.contains("crabcrop-frame-"), "{line}");
        assert!(target.ends_with("(deleted)"), "{line}");
    }
}

#[test]
fn probe_names_a_decoder_without_the_fd_protocol() {
    // FFmpeg before 6.0 has no fd protocol and refuses
    // the input outright; that is the decoder's
    // shortcoming, not a bad source, and the error says
    // what is needed. Recorded from FFmpeg 5.1.
    let (_tool, program) = fake_tool(
        "oldffmpeg",
        "#!/bin/sh\ncat > /dev/null\nprintf '%s\\n' 'fd:: Protocol not found' 'Did you mean file:fd:?' >&2\nexit 1\n",
    );
    let extractor = VideoFrameExtractor::new(program, Duration::from_secs(10)).unwrap();
    let err = extractor
        .extract_frame(b"video bytes", FrameSelection::Poster)
        .unwrap_err();
    assert!(matches!(err, FrameError::Failed(_)), "{err}");
    assert!(err.to_string().contains("FFmpeg 6.0 or newer"), "{err}");
}

#[test]
fn extract_maps_a_streamless_probe_to_not_video() {
    let (_tool, program) = fake_tool(
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
    let (_tool, program) = fake_tool(
        "failing",
        "#!/bin/sh\n\
cat > /dev/null\n\
sep=', '\n\
prev=\n\
for a in \"$@\"; do\n\
  if [ \"$a\" = \"image2pipe\" ]; then echo 'boom on stderr' >&2; exit 3; fi\n\
  if [ \"$prev\" = -dump_separator ]; then sep=\"$a\"; fi\n\
  prev=\"$a\"\n\
done\n\
printf '%s\\n' '  Duration: 00:00:03.00, start: 0.000000' '  Stream #0:0: Video: h264, yuv420p, 64x48, 10 fps' | sed \"/^  Stream #/s/, /$sep/\" >&2\n\
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
    let (_tool, program) = fake_tool(
        "empty",
        "#!/bin/sh\n\
cat > /dev/null\n\
sep=', '\n\
prev=\n\
for a in \"$@\"; do\n\
  if [ \"$a\" = \"image2pipe\" ]; then exit 0; fi\n\
  if [ \"$prev\" = -dump_separator ]; then sep=\"$a\"; fi\n\
  prev=\"$a\"\n\
done\n\
printf '%s\\n' '  Duration: 00:00:03.00, start: 0.000000' '  Stream #0:0: Video: h264, yuv420p, 64x48, 10 fps' | sed \"/^  Stream #/s/, /$sep/\" >&2\n\
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
fn probe_refuses_a_report_past_the_cap() {
    // Issue #402: source metadata can be long enough to
    // push the real facts past the report cap, leaving
    // only planted lines in the kept part. A report that
    // passed the cap is refused instead of parsed.
    let (_tool, program) = fake_tool(
        "longreport",
        &format!(
            "#!/bin/sh\ncat > /dev/null\n\
printf '%s\\n' 'Input #0, mov,mp4, from pipe:0:' '  Metadata:' '    k' \
'  Duration: 00:00:00.05, start: 0.000000, bitrate: N/A' \
'  Stream #0:0: Video: h264, yuv420p, 4000x3000, 0.001 fps' >&2\n\
head -c {} /dev/zero | tr '\\000' 'x' >&2\n\
printf '\\n%s\\n' '  Duration: 00:00:03.00, start: 0.000000, bitrate: N/A' \
'  Stream #0:0: Video: h264, yuv420p, 64x48, 10 fps, 10 tbr' >&2\n\
exit 1\n",
            PROBE_REPORT_CAP
        ),
    );
    let extractor = VideoFrameExtractor::new(program, Duration::from_secs(10)).unwrap();
    let err = extractor.probe(b"video bytes").unwrap_err();
    assert!(matches!(err, FrameError::NotVideo(_)), "{err}");
    assert!(err.to_string().contains("byte cap"), "{err}");
}

#[test]
fn stacked_invocations_share_one_deadline() {
    // Issue #362: the probe and the extraction draw
    // from one budget. Each invocation of this stand-in
    // sleeps 0.6s and then answers correctly; with a 1s
    // budget the probe succeeds and the extraction is
    // cut off by the shared deadline, where a fresh
    // timeout per invocation would have let both run
    // (1.2s in total) and succeed.
    let (_tool, program) = fake_tool(
        "slow",
        "#!/bin/sh\ncat > /dev/null\nsleep 0.6\nsep=', '\nprev=\nfor a in \"$@\"; do\n  if [ \"$a\" = \"image2pipe\" ]; then\n    printf '\\211PNG\\015\\012\\032\\012fakepng'\n    exit 0\n  fi\n  if [ \"$prev\" = -dump_separator ]; then sep=\"$a\"; fi\n  prev=\"$a\"\ndone\nprintf '%s\\n' 'Input #0, mov,mp4, from pipe:0:' '  Duration: 00:00:03.00, start: 0.000000, bitrate: N/A' '  Stream #0:0: Video: h264, yuv420p, 64x48, 10 fps, 10 tbr' | sed \"/^  Stream #/s/, /$sep/\" >&2\nexit 1\n",
    );
    let extractor = VideoFrameExtractor::new(program, Duration::from_secs(1)).unwrap();
    let started = Instant::now();
    let err = extractor
        .extract_frame(b"video bytes", FrameSelection::Poster)
        .unwrap_err();
    assert!(matches!(err, FrameError::Failed(_)), "{err}");
    assert!(err.to_string().contains("timed out"), "{err}");
    assert!(
        started.elapsed() < Duration::from_millis(1500),
        "took {:?}",
        started.elapsed()
    );
}

#[test]
fn extract_times_out_on_a_hung_decoder() {
    let (_tool, program) = fake_tool("sleeper", "#!/bin/sh\nsleep 30\n");
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

#[test]
fn real_tool_reads_an_mp4_with_its_index_at_the_end() {
    // Issue #411: FFmpeg writes an MP4's index (the moov
    // box) after its samples unless told to "faststart",
    // and a demuxer reading such a file from a pipe could
    // not get back to the samples once it was past what
    // its buffer holds. A few megabytes of noise makes
    // sure this clip is well past that.
    let Some(ffmpeg) = ffmpeg_path() else {
        return;
    };
    let dir = temp_dir("moov-at-end");
    let clip = dir.join("clip.mp4");
    let status = Command::new(&ffmpeg)
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x480:rate=10:duration=10",
            "-vf",
            "noise=alls=30:allf=t",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            "3300k",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&clip)
        .status()
        .unwrap();
    assert!(status.success());
    let bytes = std::fs::read(&clip).unwrap();
    assert!(bytes.len() > 2 << 20, "{} bytes", bytes.len());
    let mut boxes = Vec::new();
    let mut at = 0;
    while at + 8 <= bytes.len() {
        let size = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        boxes.push(bytes[at + 4..at + 8].to_vec());
        if size < 8 {
            break;
        }
        at += size;
    }
    let mdat = boxes
        .iter()
        .position(|b| b == b"mdat")
        .expect("an mdat box");
    let moov = boxes.iter().position(|b| b == b"moov").expect("a moov box");
    assert!(moov > mdat, "the index sits at the end: {boxes:?}");

    let extractor = VideoFrameExtractor::new(ffmpeg, Duration::from_secs(30)).unwrap();
    let probe = extractor.probe(&bytes).unwrap();
    assert_eq!((probe.width, probe.height), (640, 480));
    assert_eq!(probe.duration, Some(10.0));
    for selection in [
        FrameSelection::Poster,
        FrameSelection::Seconds(1.0),
        FrameSelection::Percent(100.0),
    ] {
        let frame = extractor
            .extract_frame(&bytes, selection)
            .unwrap_or_else(|e| panic!("{selection:?}: {e}"));
        let img = image::load_from_memory(&frame.png).unwrap();
        assert_eq!((img.width(), img.height()), (640, 480), "{selection:?}");
    }
}

#[test]
fn real_tool_refuses_a_playlist_that_names_a_local_file() {
    // Hostile input on the seekable channel (issues #361
    // and #411): a source that is a concat script naming
    // a video on the server's disk must not reach it. The
    // format whitelist refuses the concat demuxer, and a
    // nested open would fail anyway: a relative name
    // resolves against `fd:` to `fd:<name>`, which the
    // fd protocol refuses, and `file` is not whitelisted.
    // The decoy is a readable clip, so a leak would come
    // back as a frame instead of a refusal.
    let Some(ffmpeg) = ffmpeg_path() else {
        return;
    };
    let dir = temp_dir("playlist");
    let decoy = dir.join("decoy.mp4");
    let status = Command::new(&ffmpeg)
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:size=64x48:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&decoy)
        .status()
        .unwrap();
    assert!(status.success());
    let extractor = VideoFrameExtractor::new(ffmpeg, Duration::from_secs(30)).unwrap();
    extractor
        .extract_frame(&std::fs::read(&decoy).unwrap(), FrameSelection::Poster)
        .unwrap();

    // The tool runs in the temporary directory, so the
    // relative name is the one concat's safe mode allows.
    let relative = decoy.strip_prefix(std::env::temp_dir()).unwrap();
    for script in [
        format!("ffconcat version 1.0\nfile {}\n", relative.display()),
        format!("ffconcat version 1.0\nfile 'file:{}'\n", decoy.display()),
    ] {
        let err = extractor
            .extract_frame(script.as_bytes(), FrameSelection::Poster)
            .unwrap_err();
        assert!(matches!(err, FrameError::NotVideo(_)), "{script:?}: {err}");
    }
}

#[test]
fn real_tool_ignores_forged_metadata() {
    // Issue #402 end to end: clips whose metadata
    // imitates the report's fact lines. A forged value
    // or language tag is ignored and the selection lands
    // where the real duration and frame rate put it; a
    // key that plants a second Duration line makes the
    // report a forgery, refused as a bad source.
    let Some(ffmpeg) = ffmpeg_path() else {
        return;
    };
    let dir = temp_dir("forged");
    let make = |name: &str, extra: &[&str]| {
        let clip = dir.join(name);
        let status = Command::new(&ffmpeg)
            .args([
                "-y",
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
            .args(extra)
            .arg(&clip)
            .status()
            .unwrap();
        assert!(status.success(), "{name}");
        std::fs::read(&clip).unwrap()
    };
    let extractor = VideoFrameExtractor::new(ffmpeg.clone(), Duration::from_secs(30)).unwrap();
    for (name, extra) in [
        (
            "comment.mp4",
            ["-metadata", "comment=Duration: 00:00:00.05, x"],
        ),
        ("nan.mp4", ["-metadata", "title=Duration: NaN:00:00.00, x"]),
        ("inf.mp4", ["-metadata", "title=Duration: inf:00:00.00, x"]),
        (
            "stream.mp4",
            [
                "-metadata",
                "title=Stream #0:0: Video: h264, 4000x3000, 0.001 fps",
            ],
        ),
        // Matroska language tags print raw inside the real
        // stream line: a forged marker ahead of the real
        // one, a newline that pushes the real codec text
        // onto a line of its own, and one that moves it
        // under a fresh stream number.
        (
            "sameline.mkv",
            [
                "-metadata:s:v:0",
                "language=x): Video: h264, yuv420p, 4000x3000, 0.001 fps (",
            ],
        ),
        (
            "split.mkv",
            [
                "-metadata:s:v:0",
                "language=x): Video: h264, yuv420p, 4000x3000, 0.001 fps\n    z",
            ],
        ),
        (
            "renumbered.mkv",
            [
                "-metadata:s:v:0",
                "language=x): Video: h264, yuv420p, 4000x3000, 0.001 fps\n  Stream #0:1(eng",
            ],
        ),
    ] {
        let bytes = make(name, &extra);
        let frame = extractor
            .extract_frame(&bytes, FrameSelection::Seconds(1.5))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(frame.duration, Some(3.0), "{name}");
        assert_eq!((frame.width, frame.height), (64, 48), "{name}");
        assert_eq!(frame.timestamp, 1.5, "{name}");
        let img = image::load_from_memory(&frame.png).unwrap().to_rgb8();
        let p = img.get_pixel(32, 24);
        assert!(
            p[1] > 100 && p[0] < 60 && p[2] < 60,
            "{name}: 1.5s is green: {p:?}"
        );
    }
    let bytes = make(
        "planted.mp4",
        &[
            "-movflags",
            "use_metadata_tags",
            "-metadata",
            "k\n  Duration: 00:00:00.05, start: 0.000000, bitrate: N/A\n  z=v",
        ],
    );
    let err = extractor
        .extract_frame(&bytes, FrameSelection::Seconds(1.5))
        .unwrap_err();
    assert!(matches!(err, FrameError::NotVideo(_)), "{err}");
    // An audio-only clip whose language tag plants a video
    // stream line is still not a video.
    let clip = dir.join("audio.mkv");
    let status = Command::new(&ffmpeg)
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=2",
            "-c:a",
            "flac",
            "-metadata:s:a:0",
            "language=eng\n  Stream #0:5: Video: h264, yuv420p, 4000x3000, 0.001 fps\n  x",
        ])
        .arg(&clip)
        .status()
        .unwrap();
    assert!(status.success());
    let err = extractor
        .extract_frame(&std::fs::read(&clip).unwrap(), FrameSelection::Poster)
        .unwrap_err();
    assert!(matches!(err, FrameError::NotVideo(_)), "{err}");
}
