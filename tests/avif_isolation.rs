//! AVIF decoding in the capped worker process (issue #522),
//! end to end. The dav1d decode can loop forever on a crafted
//! AVIF; before, that loop ran on a render thread and a few
//! such requests exhausted the render lanes behind their
//! deadlines. Now each AVIF decodes in a worker the server
//! spawns from its own binary, killed at the request's
//! deadline, so a decode that never finishes costs one worker
//! and one response while the server carries on. A valid file
//! decodes to the same pixels in the worker as in process.
#![cfg(feature = "avif-decode")]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use crabcrop::engine::decode_image;
use crabcrop::svg::{WORKER_ARG, WorkerCommand, decode_avif_in};
use image::{ImageEncoder, RgbImage, RgbaImage};

// Not every shared item is used here.
#[allow(dead_code)]
mod common;

/// The engine's decode limits, passed to the worker. They are
/// `engine::MAX_DECODE_DIMENSION` and `MAX_DECODE_ALLOC`, which
/// are crate-private, so the end-to-end test names them.
const MAX_DIMENSION: u32 = 16_384;
const MAX_ALLOC: u64 = 256 * 1024 * 1024;

/// A valid AVIF the always-available encoder writes from an RGB
/// gradient of `w` x `h`.
fn rgb_avif(w: u32, h: u32) -> Vec<u8> {
    let img = RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([(x * 7) as u8, (y * 9) as u8, ((x + y) * 5) as u8])
    });
    let mut bytes = Vec::new();
    image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut bytes, 10, 88)
        .write_image(img.as_raw(), w, h, image::ExtendedColorType::Rgb8)
        .unwrap();
    bytes
}

/// A valid AVIF with an alpha channel.
fn rgba_avif(w: u32, h: u32) -> Vec<u8> {
    let img = RgbaImage::from_fn(w, h, |x, y| {
        image::Rgba([(x * 6) as u8, (y * 8) as u8, 64, ((x + y) * 10) as u8])
    });
    let mut bytes = Vec::new();
    image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut bytes, 10, 88)
        .write_image(img.as_raw(), w, h, image::ExtendedColorType::Rgba8)
        .unwrap();
    bytes
}

/// Valid AVIFs: the repo's decode seed and a spread of encoder
/// output (sizes, with and without alpha).
fn valid_avifs() -> Vec<(String, Vec<u8>)> {
    let mut out = vec![(
        "fuzz/corpus/decode/avif-small".to_string(),
        std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/corpus/decode/avif-small"),
        )
        .unwrap(),
    )];
    for (w, h) in [(1, 1), (2, 3), (20, 12), (64, 48), (129, 70)] {
        out.push((format!("rgb-{w}x{h}"), rgb_avif(w, h)));
        out.push((format!("rgba-{w}x{h}"), rgba_avif(w, h)));
    }
    // A broad corpus for the parity measurement: every `.avif`
    // under the directory this test-only variable names (e.g.
    // files produced with ffmpeg at various depths and
    // subsamplings). Unset in the gate, where the encoder output
    // and the repo seed stand in.
    if let Some(dir) = std::env::var_os("CRABCROP_TEST_AVIF_CORPUS_DIR") {
        for entry in std::fs::read_dir(&dir)
            .expect("the corpus directory exists")
            .flatten()
        {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("avif") {
                out.push((path.display().to_string(), std::fs::read(&path).unwrap()));
            }
        }
    }
    out
}

#[test]
fn the_worker_decodes_what_the_process_decodes() {
    let worker = WorkerCommand {
        program: PathBuf::from(env!("CARGO_BIN_EXE_crabcrop")),
        args: vec![WORKER_ARG.into()],
    };
    let mut checked = 0;
    for (name, avif) in valid_avifs() {
        // A corpus file the in-process path does not decode (a
        // color type this build's decoder does not surface) is
        // not a parity case; skip it. The encoder output and the
        // repo seed always decode, so enough cases remain.
        let Ok(in_process) = decode_image(&avif) else {
            continue;
        };
        let in_worker = decode_avif_in(
            &worker,
            &avif,
            MAX_DIMENSION,
            MAX_ALLOC,
            Instant::now() + Duration::from_secs(60),
        )
        .unwrap_or_else(|e| panic!("{name}: in worker: {e}"));
        assert_eq!(in_worker.color(), in_process.color(), "{name}: color");
        assert_eq!(
            (in_worker.width(), in_worker.height()),
            (in_process.width(), in_process.height()),
            "{name}: dimensions",
        );
        assert_eq!(
            in_worker.as_bytes(),
            in_process.as_bytes(),
            "{name}: the worker decoded different pixels",
        );
        checked += 1;
    }
    // Visible under `--nocapture`; captured in the gate.
    eprintln!("parity: {checked} AVIFs decoded in process and in the worker to the same pixels");
    assert!(checked >= 5, "too few AVIFs were checked: {checked}");
}

/// One raw HTTP/1.0 GET: (status, body).
fn get(port: u16, path: &str) -> (u16, Vec<u8>) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(60)))
        .unwrap();
    write!(stream, "GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n").unwrap();
    let mut raw = Vec::new();
    let _ = stream.read_to_end(&mut raw);
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or_else(|| panic!("no response head for {path}"));
    let head = String::from_utf8_lossy(&raw[..split]);
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap();
    (status, raw[split + 4..].to_vec())
}

#[test]
fn a_hung_avif_decode_is_a_504_and_the_server_keeps_serving() {
    let dir = common::scratch_dir("avif-isolation-deadline");
    std::fs::write(dir.join("slow.avif"), rgb_avif(16, 16)).unwrap();
    // A PNG source proves the server keeps serving in-process
    // decodes while an AVIF decode is stuck in its worker.
    let png = {
        let img = RgbImage::from_fn(8, 8, |x, y| {
            image::Rgb([(x * 30) as u8, (y * 30) as u8, 90])
        });
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        bytes
    };
    std::fs::write(dir.join("ok.png"), &png).unwrap();

    let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
    command
        .env_clear()
        .env("CRABCROP_SECRET", "avif-isolation-secret-0123456789")
        .env("CRABCROP_ALLOW_UNSAFE", "1")
        .env("CRABCROP_FILE_ROOT", dir.path())
        .env("CRABCROP_CACHE_MAX_MB", "0")
        .env("CRABCROP_PROCESS_TIMEOUT_SECS", "1")
        // The worker stalls its AVIF job far past the deadline,
        // so the parent's kill is exercised without a special
        // crafted file.
        .env("CRABCROP_TEST_AVIF_WORKER_DELAY_MS", "30000");
    let (mut server, port, _) = common::serve(&mut command);

    // The stuck AVIF decode answers 504, at the deadline plus
    // the runner's grace, not minutes from now.
    let stuck = std::thread::spawn(move || {
        let started = Instant::now();
        let (status, _) = get(port, "/unsafe/32x32/file:///slow.avif");
        (status, started.elapsed())
    });
    // While it runs, the server answers health checks and
    // renders a PNG in process.
    let mut served = 0;
    while !stuck.is_finished() {
        assert_eq!(get(port, "/health").0, 200);
        let (status, body) = get(port, "/unsafe/8x8/file:///ok.png");
        assert_eq!(status, 200);
        assert!(body.starts_with(b"\x89PNG"));
        served += 1;
        std::thread::sleep(Duration::from_millis(100));
    }
    let (status, took) = stuck.join().unwrap();
    assert_eq!(status, 504, "the stuck AVIF decode did not time out");
    assert!(took < Duration::from_secs(15), "the 504 took {took:?}");
    assert!(served > 0, "nothing was served while the AVIF decode hung");
    assert!(server.0.try_wait().unwrap().is_none(), "the server exited");
    assert_eq!(get(port, "/health").0, 200);
    assert_eq!(get(port, "/unsafe/8x8/file:///ok.png").0, 200);
}

/// The added latency of decoding an AVIF in the worker versus
/// in process, for a small and a large valid file. Ignored, so
/// it runs on request (`cargo test --features avif-decode
/// --ignored worker_decode_latency -- --nocapture`), not in the
/// gate.
#[test]
#[ignore = "latency measurement, run by hand"]
fn worker_decode_latency() {
    fn median(mut v: Vec<Duration>) -> Duration {
        v.sort();
        v[v.len() / 2]
    }
    let worker = WorkerCommand {
        program: PathBuf::from(env!("CARGO_BIN_EXE_crabcrop")),
        args: vec![WORKER_ARG.into()],
    };
    for (label, avif) in [
        ("small 20x12", rgb_avif(20, 12)),
        ("large 1200x800", rgb_avif(1200, 800)),
    ] {
        let runs = 25;
        let mut in_process = Vec::new();
        let mut in_worker = Vec::new();
        for _ in 0..runs {
            let t = Instant::now();
            decode_image(&avif).unwrap();
            in_process.push(t.elapsed());
            let t = Instant::now();
            decode_avif_in(
                &worker,
                &avif,
                MAX_DIMENSION,
                MAX_ALLOC,
                Instant::now() + Duration::from_secs(60),
            )
            .unwrap();
            in_worker.push(t.elapsed());
        }
        let ip = median(in_process);
        let iw = median(in_worker);
        println!(
            "avif decode latency [{label}]: in-process {ip:?}, in-worker {iw:?}, added {:?} (median of {runs})",
            iw.saturating_sub(ip)
        );
    }
}
