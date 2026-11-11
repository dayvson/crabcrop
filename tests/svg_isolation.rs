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
//!
//! The last test covers what a child can reach in the server
//! (issue #605): the server marks itself non-dumpable, so a
//! child cannot read its environment through `/proc`, and the
//! worker still starts and renders.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Command;
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
    boot_in(root, None, extra_env)
}

/// [`boot`] with the server started from `cwd`, which its SVG
/// workers inherit, instead of the test's own directory.
fn boot_in(
    root: &Path,
    cwd: Option<&Path>,
    extra_env: &[(&str, &str)],
) -> (common::ServerGuard, u16) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    boot_command(command, root, extra_env)
}

/// Boot `command`, which starts the binary, with the
/// environment every test here gives the server.
fn boot_command(
    mut command: Command,
    root: &Path,
    extra_env: &[(&str, &str)],
) -> (common::ServerGuard, u16) {
    command
        .env_clear()
        .env("CRABCROP_SECRET", "svg-isolation-secret-0123456789")
        .env("CRABCROP_ALLOW_UNSAFE", "1")
        .env("CRABCROP_FILE_ROOT", root)
        .env("CRABCROP_CACHE_MAX_MB", "0")
        .env("CRABCROP_METRICS_TOKEN", METRICS_TOKEN);
    for (name, value) in extra_env {
        command.env(name, value);
    }
    let (server, port, _) = common::serve(&mut command);
    (server, port)
}

const METRICS_TOKEN: &str = "svg-isolation-metrics-token";

/// One process lane gauge from `/metrics`: `in_use` or
/// `capacity`.
fn process_lane(port: u16, gauge: &str) -> u32 {
    let (status, body, _) = get_with(
        port,
        "/metrics",
        &format!("Authorization: Bearer {METRICS_TOKEN}\r\n"),
    );
    assert_eq!(status, 200);
    let series = format!("crabcrop_lane_{gauge}{{lane=\"process\"}} ");
    String::from_utf8_lossy(&body)
        .lines()
        .find_map(|line| line.strip_prefix(series.as_str()))
        .and_then(|n| n.trim().parse().ok())
        .unwrap_or_else(|| panic!("no process lane {gauge} gauge"))
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
    // The nested chain is refused inside its worker in
    // milliseconds. It runs alone, so it never waits for a
    // stills permit behind the arc.
    let (status, _, took) = get(port, "/unsafe/64x64/file:///nested.svg");
    assert_eq!(status, 502);
    assert!(
        took < Duration::from_secs(30),
        "the nested chain took {took:?}"
    );
    assert!(server.0.try_wait().unwrap().is_none(), "the server exited");
    // The arc's render holds a stills lane permit for
    // seconds, as any long render does. The lane has one
    // permit per core, so a render beside it is served only
    // when the lane has a second permit. On a one-core host
    // it waits the lane's 5 s and is shed with 503, as it
    // would be behind a slow raster render (issue #547).
    let spare_permit = process_lane(port, "capacity") > 1;
    let arc = std::thread::spawn(move || get(port, "/unsafe/64x64/file:///arc.svg"));
    let (mut rendered, mut explained) = (0, 0);
    while !arc.is_finished() {
        assert_eq!(get(port, "/health").0, 200);
        // The explain report parses an ordinary SVG in a
        // worker of its own, under the probe gate, which no
        // render holds: this works on every host.
        let (status, body, _) = get(port, "/explain/unsafe/64x64/file:///ok.svg");
        assert_eq!(status, 200);
        assert!(String::from_utf8_lossy(&body).contains("svg, 200x100"));
        explained += 1;
        let (status, body, _) = get(port, "/unsafe/32x32/file:///ok.svg");
        if status == 200 {
            assert!(body.starts_with(b"\x89PNG"), "an SVG renders to PNG");
            rendered += 1;
        } else {
            assert!(
                !spare_permit,
                "a render was shed with a stills permit free: {status}"
            );
            assert_eq!(status, 503);
            assert_eq!(
                String::from_utf8_lossy(&body),
                "server busy: too many images processing at once"
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let (status, body, took) = arc.join().unwrap();
    assert_eq!(status, 502, "{}", String::from_utf8_lossy(&body));
    assert!(took < Duration::from_secs(60), "the arc took {took:?}");
    assert!(
        explained > 0,
        "nothing was explained while the arc rendered"
    );
    if spare_permit {
        assert!(rendered > 0, "nothing was rendered while the arc rendered");
    }
    // The server is the same process, still serving.
    assert!(server.0.try_wait().unwrap().is_none(), "the server exited");
    assert_eq!(get(port, "/health").0, 200);
    assert_eq!(get(port, "/unsafe/32x32/file:///ok.svg").0, 200);
    // A hostile SVG costs the explain report only the size.
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
    while process_lane(port, "in_use") != 0 {
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

/// A solid PNG of `color`.
fn solid_png(color: [u8; 4]) -> Vec<u8> {
    let img = image::RgbaImage::from_pixel(16, 16, image::Rgba(color));
    let mut png = Vec::new();
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    png
}

#[test]
fn an_svg_draws_no_file_its_hrefs_name_whatever_the_working_directory_holds() {
    // usvg reads an image href that is not a data URL as a
    // file path, relative to the working directory the worker
    // inherits from the server. Start the server from a
    // directory that holds a `data:` directory, a file named
    // like a fragment reference, and one under a directory
    // named like one, with a red PNG outside the file root:
    // none of these hrefs may draw a file.
    let work = common::scratch_dir("svg-isolation-href");
    let (root, cwd, outside) = (work.join("root"), work.join("cwd"), work.join("outside"));
    for dir in [&root, &cwd, &outside, &cwd.join("data:"), &cwd.join("#d")] {
        std::fs::create_dir(dir).unwrap();
    }
    let red = solid_png([255, 0, 0, 255]);
    let secret = outside.join("secret.png");
    std::fs::write(&secret, &red).unwrap();
    std::fs::write(cwd.join("#red.png"), &red).unwrap();
    std::fs::write(cwd.join("#d").join("red.png"), &red).unwrap();
    // `cwd/data:/..` is `cwd`; enough further `..` reach `/`.
    let ups = "../".repeat(cwd.components().count() + 2);
    let through_data = format!(
        "data:/{ups}{}",
        secret.to_str().unwrap().trim_start_matches('/')
    );
    let svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\"><rect width=\"100\" height=\"100\" fill=\"#00ff00\"/><image href=\"{through_data}\" width=\"50\" height=\"100\"/><image href=\"#red.png\" x=\"50\" width=\"50\" height=\"100\"/><image href=\"#d/red.png\" x=\"25\" width=\"50\" height=\"50\"/><filter id=\"f\"><feImage href=\"{through_data}\"/></filter><rect y=\"80\" width=\"100\" height=\"20\" filter=\"url(#f)\"/></svg>"
    );
    std::fs::write(root.join("href.svg"), svg).unwrap();
    let (mut server, port) = boot_in(&root, Some(&cwd), &[]);
    let (status, body, _) = get(port, "/unsafe/filters:format(png)/file:///href.svg");
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    let out = image::load_from_memory(&body).unwrap().to_rgba8();
    assert_eq!(out.dimensions(), (100, 100));
    let red_pixels = out
        .pixels()
        .filter(|p| p[0] > 200 && p[1] < 60 && p[2] < 60)
        .count();
    assert_eq!(red_pixels, 0, "an href drew a file from disk");
    assert_eq!(out.get_pixel(25, 40), &image::Rgba([0, 255, 0, 255]));
    assert_eq!(out.get_pixel(75, 40), &image::Rgba([0, 255, 0, 255]));
    assert_eq!(out.get_pixel(50, 20), &image::Rgba([0, 255, 0, 255]));
    assert!(server.0.try_wait().unwrap().is_none(), "the server exited");
}

/// Whether this process holds any capability, as a test run
/// as root does. `CAP_SYS_PTRACE`, `CAP_SYS_ADMIN` and
/// `CAP_PERFMON` each let a process read another's `/proc`
/// files, a non-dumpable one's included.
#[cfg(target_os = "linux")]
fn holds_capabilities() -> bool {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("CapEff:"))
                .and_then(|hex| u64::from_str_radix(hex.trim(), 16).ok())
        })
        .is_some_and(|caps| caps != 0)
}

/// The binary, started as the images start it: with no
/// capabilities. When this test holds some, `setpriv` drops
/// them all before the exec.
#[cfg(target_os = "linux")]
fn server_without_capabilities() -> Command {
    let binary = env!("CARGO_BIN_EXE_crabcrop");
    if !holds_capabilities() {
        return Command::new(binary);
    }
    let setpriv = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|dir| dir.join("setpriv"))
        .find(|path| path.is_file())
        .expect("this test holds capabilities (run as root?) and needs setpriv to drop them");
    let mut command = Command::new(setpriv);
    command
        .args(["--inh-caps", "-all", "--bounding-set", "-all", "--"])
        .arg(binary);
    command
}

#[test]
#[cfg(target_os = "linux")]
fn a_child_cannot_read_the_servers_environment_and_the_worker_still_renders() {
    // Issue #605: a child runs as the server's user, so a
    // compromised decoder could read the server's secrets
    // from `/proc/<server pid>/environ` while the server was
    // dumpable. This fake decoder tries that, then reports the
    // OOM score the server gave it, and fails the probe. It
    // runs ten times: `spawn` can return before the child's
    // exec makes it dumpable again, and the server's OOM score
    // write must still land then.
    use std::os::unix::fs::PermissionsExt;
    let dir = source_dir("not-dumpable");
    std::fs::write(dir.join("clip.mp4"), b"not a real video").unwrap();
    let (log, seen) = (dir.join("decoder.log"), dir.join("environ"));
    let decoder = dir.join("decoder.sh");
    std::fs::write(
        &decoder,
        format!(
            "#!/bin/sh\n\
             PATH=/usr/bin:/bin\n\
             echo \"ppid $PPID\" >> '{log}'\n\
             if cat \"/proc/$PPID/environ\" > '{seen}' 2>/dev/null; then\n\
             echo 'environ read' >> '{log}'\n\
             else\n\
             echo 'environ refused' >> '{log}'\n\
             fi\n\
             sleep 0.3\n\
             echo \"oom_score_adj $(cat /proc/$$/oom_score_adj)\" >> '{log}'\n\
             exit 1\n",
            log = log.display(),
            seen = seen.display(),
        ),
    )
    .unwrap();
    std::fs::set_permissions(&decoder, std::fs::Permissions::from_mode(0o755)).unwrap();
    let (mut server, port) = boot_command(
        server_without_capabilities(),
        &dir,
        &[("CRABCROP_VIDEO_DECODER", decoder.to_str().unwrap())],
    );
    const RUNS: usize = 10;
    for _ in 0..RUNS {
        let (status, body, _) = get(port, "/unsafe/frame:poster/50x50/file:///clip.mp4");
        assert_ne!(status, 200, "{}", String::from_utf8_lossy(&body));
    }
    let log = std::fs::read_to_string(&log).expect("the decoder ran");
    let parent = format!("ppid {}", server.0.id());
    let parents: Vec<&str> = log.lines().filter(|l| l.starts_with("ppid ")).collect();
    assert_eq!(parents.len(), RUNS, "{log}");
    assert!(
        parents.iter().all(|line| *line == parent),
        "the decoder's parent is not the server: {log}"
    );
    let seen = std::fs::read(&seen).unwrap_or_default();
    assert!(
        !String::from_utf8_lossy(&seen).contains("CRABCROP_SECRET="),
        "a child read the server's secret from /proc: {log}"
    );
    assert_eq!(log.matches("environ refused\n").count(), RUNS, "{log}");
    assert!(!log.contains("environ read"), "{log}");
    // The server still sets the OOM score of every child it
    // starts.
    assert_eq!(log.matches("oom_score_adj 1000\n").count(), RUNS, "{log}");
    // The worker starts from the server's own image and renders.
    let (status, body, _) = get(port, "/unsafe/32x32/file:///ok.svg");
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    assert!(body.starts_with(b"\x89PNG"), "an SVG renders to PNG");
    assert!(server.0.try_wait().unwrap().is_none(), "the server exited");
}
