//! SVG parsing and rendering in a capped child process
//! (issue #447), end to end through the binary. Before, a
//! document that got past the pre-parse bound took the
//! server down with it: a 130-byte arc made kurbo allocate
//! gigabytes, the allocation failed, and the process
//! aborted with every request in flight. Now each SVG parses
//! and renders in a worker the server spawns from its own
//! binary, so the arc costs one worker and one `502`, a
//! render past the deadline costs one killed worker and one
//! `504`, and the server answers everything else throughout.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crabcrop::svg::{WORKER_ARG, WorkerCommand, prepare, screen};

// Not every shared item is used here.
#[allow(dead_code)]
mod common;

/// One raw HTTP/1.0 GET: (status, body, time taken).
fn get(port: u16, path: &str) -> (u16, Vec<u8>, Duration) {
    get_with(port, path, "")
}

/// [`get`] with extra header lines, each ending in CRLF.
fn get_with(port: u16, path: &str, headers: &str) -> (u16, Vec<u8>, Duration) {
    let started = Instant::now();
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(90)))
        .unwrap();
    write!(
        stream,
        "GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n{headers}\r\n"
    )
    .unwrap();
    let mut raw = Vec::new();
    let _ = stream.read_to_end(&mut raw);
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or_else(|| panic!("no response head for {path}: {raw:?}"));
    let head = String::from_utf8_lossy(&raw[..split]);
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap();
    (status, raw[split + 4..].to_vec(), started.elapsed())
}

/// A directory of SVG sources the server reads through
/// `file://` URLs. Bound before the server's guard, it is
/// removed after the server exits (locals drop in reverse).
fn source_dir(name: &str) -> common::ScratchDir {
    let dir = common::scratch_dir(&format!("svg-isolation-{name}"));
    std::fs::write(dir.join("ok.svg"), OK_SVG).unwrap();
    std::fs::write(dir.join("arc.svg"), ARC_SVG).unwrap();
    std::fs::write(dir.join("nested.svg"), nested_chain()).unwrap();
    std::fs::write(dir.join("slow.svg"), slow_svg()).unwrap();
    dir
}

fn boot(root: &Path, extra_env: &[(&str, &str)]) -> (common::ServerGuard, u16) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
    command
        .env_clear()
        .env("CRABCROP_SECRET", "svg-isolation-secret-0123456789")
        .env("CRABCROP_ALLOW_UNSAFE", "1")
        .env("CRABCROP_FILE_ROOT", root)
        .env("CRABCROP_CACHE_MAX_MB", "0")
        .env("CRABCROP_METRICS_TOKEN", METRICS_TOKEN)
        .stderr(Stdio::null());
    for (name, value) in extra_env {
        command.env(name, value);
    }
    let (server, port, _) = common::serve(&mut command);
    (server, port)
}

const METRICS_TOKEN: &str = "svg-isolation-metrics-token";

/// The process lane permits in use, from `/metrics`.
fn process_lane_in_use(port: u16) -> u32 {
    let (status, body, _) = get_with(
        port,
        "/metrics",
        &format!("Authorization: Bearer {METRICS_TOKEN}\r\n"),
    );
    assert_eq!(status, 200);
    String::from_utf8_lossy(&body)
        .lines()
        .find_map(|line| line.strip_prefix("crabcrop_lane_in_use{lane=\"process\"} "))
        .and_then(|n| n.trim().parse().ok())
        .expect("a process lane gauge")
}

const OK_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><rect width="200" height="100" fill="#336699"/><circle cx="100" cy="50" r="30" fill="#ffcc00"/></svg>"##;

/// The issue's 130-byte arc: a tiny rx scales ry up by the
/// chord over rx, and kurbo flattens the arc into a number
/// of cubics no allocation can hold.
const ARC_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><path d="M0 0 A 2e-5 1e60 0 1 1 1 0" stroke="black"/></svg>"#;

/// Base64 (standard alphabet, padded).
fn b64(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let n = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// The issue #443 feImage chain (K=10, N=7: ten million
/// nodes once expanded), nested in a `data:image/svg+xml`
/// image, where it never meets the sanitizer: usvg parses
/// it inside the outer parse. Before #443 it aborted the
/// process after 23 s.
fn nested_chain() -> String {
    let mut body = String::from("<defs><rect id=\"r0\" width=\"10\" height=\"10\"/>");
    for i in 1..=7 {
        body.push_str(&format!("<filter id=\"f{i}\">"));
        body.push_str(&format!("<feImage href=\"#r{}\"/>", i - 1).repeat(10));
        body.push_str(&format!(
            "</filter><rect id=\"r{i}\" width=\"10\" height=\"10\" filter=\"url(#f{i})\"/>"
        ));
    }
    body.push_str("</defs><rect width=\"10\" height=\"10\" filter=\"url(#f7)\"/>");
    let inner = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\">{body}</svg>"
    );
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><image width=\"10\" height=\"10\" href=\"data:image/svg+xml;base64,{}\"/></svg>",
        b64(inner.as_bytes())
    )
}

/// An ordinary-looking document that takes far longer to
/// render than any deadline below: five thousand
/// translucent rects, each covering a 4096 px canvas, with
/// little memory to its name.
fn slow_svg() -> String {
    let mut svg =
        String::from("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"4096\" height=\"4096\">");
    svg.push_str(
        &"<rect width=\"4096\" height=\"4096\" fill=\"#336699\" opacity=\"0.5\"/>".repeat(5_000),
    );
    svg.push_str("</svg>");
    svg
}

#[test]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "only Linux enforces the worker's address-space cap; elsewhere the arc allocates until the deadline"
)]
fn hostile_svgs_fail_alone_while_the_server_keeps_serving() {
    let dir = source_dir("hostile");
    let (mut server, port) = boot(&dir, &[]);
    let arc = std::thread::spawn(move || get(port, "/unsafe/64x64/file:///arc.svg"));
    let nested = std::thread::spawn(move || get(port, "/unsafe/64x64/file:///nested.svg"));
    // While the hostile renders run, the server answers
    // health checks and renders an ordinary SVG.
    let mut served = 0;
    while !(arc.is_finished() && nested.is_finished()) {
        assert_eq!(get(port, "/health").0, 200);
        let (status, body, _) = get(port, "/unsafe/32x32/file:///ok.svg");
        assert_eq!(status, 200);
        assert!(body.starts_with(b"\x89PNG"), "an SVG renders to PNG");
        served += 1;
        std::thread::sleep(Duration::from_millis(100));
    }
    let (status, body, took) = arc.join().unwrap();
    assert_eq!(status, 502, "{}", String::from_utf8_lossy(&body));
    assert!(took < Duration::from_secs(60), "the arc took {took:?}");
    let (status, _, took) = nested.join().unwrap();
    assert_eq!(status, 502);
    assert!(
        took < Duration::from_secs(30),
        "the nested chain took {took:?}"
    );
    assert!(served > 0, "nothing was served while the arc rendered");
    // The server is the same process, still serving.
    assert!(server.0.try_wait().unwrap().is_none(), "the server exited");
    assert_eq!(get(port, "/health").0, 200);
    assert_eq!(get(port, "/unsafe/32x32/file:///ok.svg").0, 200);
    // The explain report reads an SVG's size through a
    // worker too, and a hostile one costs it only the size.
    let (status, body, _) = get(port, "/explain/unsafe/64x64/file:///ok.svg");
    assert_eq!(status, 200);
    assert!(String::from_utf8_lossy(&body).contains("svg, 200x100"));
    let (status, body, took) = get(port, "/explain/unsafe/64x64/file:///arc.svg");
    assert_eq!(status, 200);
    let report = String::from_utf8_lossy(&body);
    assert!(
        report.contains(&format!("svg, {} bytes", ARC_SVG.len())),
        "{report}"
    );
    assert!(took < Duration::from_secs(30), "explain took {took:?}");
    assert!(server.0.try_wait().unwrap().is_none(), "the server exited");
}

/// The server's live child processes (Linux: the kernel
/// lists each thread's children).
#[cfg(target_os = "linux")]
fn children_of(pid: u32) -> Vec<String> {
    let mut children = Vec::new();
    if let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) {
        for task in tasks.flatten() {
            if let Ok(list) = std::fs::read_to_string(task.path().join("children")) {
                children.extend(list.split_whitespace().map(str::to_string));
            }
        }
    }
    children
}

#[test]
fn a_render_past_the_deadline_is_a_504_and_its_worker_dies() {
    let dir = source_dir("deadline");
    let (mut server, port) = boot(&dir, &[("CRABCROP_PROCESS_TIMEOUT_SECS", "1")]);
    let (status, _, took) = get(port, "/unsafe/file:///slow.svg");
    assert_eq!(status, 504);
    // The deadline, the runner's two-second grace, and slack
    // for a loaded test host.
    assert!(took < Duration::from_secs(15), "the 504 took {took:?}");
    // The worker was killed at the deadline, not left to
    // render on behind the 504: the render gives its process
    // lane permit back, where an in-process render detached
    // at its 504 would hold it until it finished, minutes
    // from now.
    let deadline = Instant::now() + Duration::from_secs(5);
    while process_lane_in_use(port) != 0 {
        assert!(
            Instant::now() < deadline,
            "the render still holds its lane after the 504"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    #[cfg(target_os = "linux")]
    {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !children_of(server.0.id()).is_empty() {
            assert!(
                Instant::now() < deadline,
                "the worker outlived the deadline: {:?}",
                children_of(server.0.id())
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    assert!(server.0.try_wait().unwrap().is_none(), "the server exited");
    assert_eq!(get(port, "/unsafe/32x32/file:///ok.svg").0, 200);
}

/// A small PNG, embedded as a `data:` URI below.
fn png_data_uri() -> String {
    let img = image::RgbaImage::from_fn(8, 8, |x, y| {
        image::Rgba([(x * 30) as u8, (y * 30) as u8, 128, 200])
    });
    let mut png = Vec::new();
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    format!("data:image/png;base64,{}", b64(&png))
}

/// Ordinary documents covering what usvg and resvg draw:
/// shapes and paths, gradients, patterns, masks, clip paths,
/// filters, markers, text, an embedded raster, and a nested
/// SVG image.
fn ordinary_documents() -> Vec<String> {
    let svg = |body: &str| {
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"120\" height=\"80\" viewBox=\"0 0 60 40\">{body}</svg>"
        )
    };
    vec![
        OK_SVG.to_string(),
        svg(
            "<path d=\"M5 5 L55 5 Q58 20 55 35 C40 38 20 38 5 35 Z\" fill=\"#c33\" stroke=\"#000\" stroke-width=\"1.5\" stroke-dasharray=\"3 1\"/><ellipse cx=\"30\" cy=\"20\" rx=\"12\" ry=\"6\" fill=\"#fff\" fill-opacity=\"0.6\"/>",
        ),
        svg(
            "<defs><linearGradient id=\"l\" x2=\"1\" y2=\"1\"><stop offset=\"0\" stop-color=\"#f80\"/><stop offset=\"1\" stop-color=\"#08f\" stop-opacity=\"0.4\"/></linearGradient><radialGradient id=\"r\"><stop offset=\"0\" stop-color=\"#fff\"/><stop offset=\"1\" stop-color=\"#000\"/></radialGradient></defs><rect width=\"60\" height=\"40\" fill=\"url(#l)\"/><circle cx=\"30\" cy=\"20\" r=\"15\" fill=\"url(#r)\"/>",
        ),
        svg(
            "<defs><pattern id=\"p\" width=\"6\" height=\"6\" patternUnits=\"userSpaceOnUse\"><rect width=\"3\" height=\"3\" fill=\"#393\"/></pattern><mask id=\"m\"><circle cx=\"30\" cy=\"20\" r=\"18\" fill=\"#fff\"/></mask><clipPath id=\"c\"><rect x=\"5\" y=\"5\" width=\"50\" height=\"30\" rx=\"6\"/></clipPath></defs><g clip-path=\"url(#c)\"><rect width=\"60\" height=\"40\" fill=\"url(#p)\" mask=\"url(#m)\"/></g>",
        ),
        svg(
            "<defs><filter id=\"f\"><feGaussianBlur stdDeviation=\"1.5\"/><feOffset dx=\"2\" dy=\"1\" result=\"o\"/><feMerge><feMergeNode in=\"o\"/><feMergeNode in=\"SourceGraphic\"/></feMerge></filter><marker id=\"a\" markerWidth=\"4\" markerHeight=\"4\" refX=\"2\" refY=\"2\" orient=\"auto\"><path d=\"M0 0 L4 2 L0 4 z\" fill=\"#000\"/></marker></defs><rect x=\"10\" y=\"10\" width=\"30\" height=\"15\" fill=\"#36c\" filter=\"url(#f)\"/><polyline points=\"5,35 20,25 35,32 55,8\" fill=\"none\" stroke=\"#000\" marker-mid=\"url(#a)\" marker-end=\"url(#a)\"/>",
        ),
        svg(
            "<text x=\"4\" y=\"24\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#222\">crab 123</text><text x=\"4\" y=\"36\" font-family=\"serif\" font-weight=\"bold\" font-size=\"9\">Text</text>",
        ),
        svg(&format!(
            "<image href=\"{}\" x=\"5\" y=\"5\" width=\"30\" height=\"30\"/><image xlink:href=\"data:image/svg+xml;base64,{}\" x=\"30\" y=\"5\" width=\"25\" height=\"25\"/>",
            png_data_uri(),
            b64(OK_SVG.as_bytes())
        )),
    ]
}

#[test]
fn the_worker_renders_what_the_process_renders() {
    // The binary's worker against the library's in-process
    // parse and render: the same intrinsic size and the same
    // RGBA bytes, at the intrinsic size and scaled.
    let worker = WorkerCommand {
        program: PathBuf::from(env!("CARGO_BIN_EXE_crabcrop")),
        args: vec![WORKER_ARG.into()],
    };
    for document in ordinary_documents() {
        let screened = screen(document.as_bytes()).unwrap();
        let local = prepare(document.as_bytes()).unwrap();
        let size = screened.intrinsic_size_in(Some(&worker), None).unwrap();
        assert_eq!(size, local.intrinsic(), "{document}");
        for scale in [1.0, 2.5] {
            let target = |(w, h): (f32, f32)| ((w * scale) as u32, (h * scale) as u32);
            let isolated = screened
                .rasterize_in(Some(&worker), None, target)
                .unwrap_or_else(|e| panic!("{document}: {e}"));
            let (w, h) = target(local.intrinsic());
            let in_process = local.render(w, h).unwrap();
            assert_eq!(
                isolated.as_bytes(),
                in_process.as_bytes(),
                "the worker rendered different pixels for {document}"
            );
        }
    }
    // A refusal from the worker's own bounds is the same
    // refusal as in-process.
    let nested = nested_chain();
    let screened = screen(nested.as_bytes()).unwrap();
    let isolated = screened
        .intrinsic_size_in(Some(&worker), None)
        .unwrap_err()
        .to_string();
    let local = prepare(nested.as_bytes()).err().unwrap();
    assert_eq!(isolated, local);
}
