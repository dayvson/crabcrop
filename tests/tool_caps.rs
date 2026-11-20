//! The external tools' address-space caps (issue #738), end
//! to end through the binary. Each tool's cap was a constant;
//! now a variable sets it. What a variable sets reaches the
//! kernel limit its tool runs under, and a value under what
//! the tool needs to start stops the boot with the variable
//! named. The kernel limits exist on Linux alone.

#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

// Not every shared item is used here.
#[allow(dead_code)]
mod common;

const SECRET: &str = "tool-caps-secret-0123456789";

/// Long enough for anything that must happen on a loaded
/// machine; a test that passes never waits it out.
const LONG: Duration = Duration::from_secs(60);

const MB: u64 = 1024 * 1024;

/// Each cap's variable, the least it may be set to, in
/// megabytes, and the tool the refusal names.
const CAPS: [(&str, u64, &str); 4] = [
    (
        "CRABCROP_VIDEO_ENCODER_ADDRESS_SPACE_MAX_MB",
        1024,
        "video encoder",
    ),
    (
        "CRABCROP_VIDEO_DECODER_ADDRESS_SPACE_MAX_MB",
        1024,
        "video decoder",
    ),
    (
        "CRABCROP_PDF_RENDERER_ADDRESS_SPACE_MAX_MB",
        256,
        "PDF renderer",
    ),
    (
        "CRABCROP_SVG_WORKER_ADDRESS_SPACE_MAX_MB",
        512,
        "SVG worker",
    ),
];

/// The binary with the environment every test here gives
/// the server: unsigned URLs over the `file://` sources in
/// `root`, and no memory cache, so every request renders.
fn binary(root: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
    command
        .env_clear()
        .env("CRABCROP_SECRET", SECRET)
        .env("CRABCROP_ALLOW_UNSAFE", "1")
        .env("CRABCROP_FILE_ROOT", root)
        .env("CRABCROP_CACHE_MAX_MB", "0");
    command
}

/// Boot `command` and wait for the boot to stop: its exit
/// status and what it wrote to stderr. A server that listens
/// instead, with `what` in its environment, fails the test.
fn boot_stops(command: &mut Command, what: &str) -> (std::process::ExitStatus, String) {
    let mut child = command
        .env("PORT", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawning crabcrop");
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut stderr = child.stderr.take().unwrap();
    let mut server = common::ServerGuard::new(child);
    // A server that boots prints its listening line: that
    // line, or the end of stdout, ends the wait.
    let mut printed = String::new();
    let mut line = String::new();
    while stdout.read_line(&mut line).unwrap_or(0) > 0 {
        assert!(
            !line.starts_with("crabcrop listening on port"),
            "the server booted with {what}: {printed}{line}"
        );
        printed.push_str(&line);
        line.clear();
    }
    let waited = Instant::now();
    let status = loop {
        if let Some(status) = server.0.try_wait().unwrap() {
            break status;
        }
        assert!(waited.elapsed() < LONG, "the server did not exit");
        std::thread::sleep(Duration::from_millis(10));
    };
    let mut said = String::new();
    stderr.read_to_string(&mut said).unwrap();
    (status, said)
}

/// One HTTP/1.0 GET: the status of its answer.
fn get(port: u16, path: &str) -> u16 {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.set_read_timeout(Some(LONG)).unwrap();
    write!(stream, "GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n").unwrap();
    let mut response = Vec::new();
    let _ = stream.read_to_end(&mut response);
    let response = String::from_utf8_lossy(&response);
    response
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status for {path}: {response:?}"))
}

/// Write an executable script.
fn script(path: &Path, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, body).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path.to_path_buf()
}

/// Shell lines that append the running script's own
/// address-space limit, in bytes, to `record`. The pause
/// covers the moment between the spawn and the limits the
/// server sets on the child right after it.
fn record_limit(record: &Path) -> String {
    format!(
        "sleep 0.3\n\
         grep '^Max address space' /proc/$$/limits | awk '{{print $4}}' >> '{}'\n",
        record.display()
    )
}

/// The limits a record holds, one per run.
fn recorded(record: &Path) -> Vec<u64> {
    std::fs::read_to_string(record)
        .unwrap_or_default()
        .lines()
        .map(|line| line.parse().unwrap_or_else(|_| panic!("{line:?}")))
        .collect()
}

/// A 16x16 PNG, for a stand-in renderer to print.
fn png() -> Vec<u8> {
    let mut out = std::io::Cursor::new(Vec::new());
    image::RgbaImage::from_pixel(16, 16, image::Rgba([255, 255, 255, 255]))
        .write_to(&mut out, image::ImageFormat::Png)
        .unwrap();
    out.into_inner()
}

/// A two-frame animated GIF, a source video output accepts.
fn animated_gif() -> Vec<u8> {
    use image::codecs::gif::GifEncoder;
    use image::{Delay, Frame, Rgba, RgbaImage};
    let mut out = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut out);
        for color in [[200, 30, 30, 255], [30, 30, 200, 255]] {
            let frame = Frame::from_parts(
                RgbaImage::from_pixel(8, 8, Rgba(color)),
                0,
                0,
                Delay::from_numer_denom_ms(100, 1),
            );
            encoder.encode_frame(frame).unwrap();
        }
    }
    out
}

/// An SVG that renders for longer than any deadline below
/// (the one tests/svg_isolation.rs uses): five thousand
/// translucent rects, each covering a 4096 px canvas.
fn slow_svg() -> String {
    let mut svg =
        String::from("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"4096\" height=\"4096\">");
    svg.push_str(
        &"<rect width=\"4096\" height=\"4096\" fill=\"#336699\" opacity=\"0.5\"/>".repeat(5_000),
    );
    svg.push_str("</svg>");
    svg
}

/// The pids of `parent`'s children, from `/proc`.
fn children_of(parent: u32) -> Vec<u32> {
    let mut children = Vec::new();
    for entry in std::fs::read_dir("/proc").unwrap().flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            continue;
        };
        // `pid (comm) state ppid ...`: the name can hold
        // spaces and parentheses, so the fields after it are
        // read from the last `)`.
        let Some((_, rest)) = stat.rsplit_once(')') else {
            continue;
        };
        if rest.split_whitespace().nth(1) == Some(&parent.to_string()) {
            children.push(pid);
        }
    }
    children
}

/// A process's soft address-space limit, from its
/// `/proc/<pid>/limits`, or `None` when it is unlimited or
/// the process is gone.
fn address_space_limit(pid: u32) -> Option<u64> {
    let limits = std::fs::read_to_string(format!("/proc/{pid}/limits")).ok()?;
    let line = limits
        .lines()
        .find(|line| line.starts_with("Max address space"))?;
    line.split_whitespace().nth(3)?.parse().ok()
}

#[test]
fn each_cap_reaches_its_tools_limit() {
    // Each stand-in tool writes down the address-space limit
    // it runs under: the PDF tools at the boot check, the
    // decoder at a `frame:` request's probe, the encoder at
    // an MP4 request. The SVG worker is the server's own
    // binary, which sets its limit itself from the cap the
    // server passes it: its `/proc` entry is read while it
    // renders a document slower than the deadline. Every cap
    // is set to a value no default has. A boot with no cap
    // set comes first: its boot lines are the ones the tools
    // had before the caps were settings.
    let dir = common::scratch_dir("tool-caps-reach");
    let record = |name: &str| dir.join(format!("{name}.limits"));
    std::fs::write(dir.join("page.png"), png()).unwrap();
    std::fs::write(dir.join("clip.mp4"), b"not a real video").unwrap();
    std::fs::write(dir.join("anim.gif"), animated_gif()).unwrap();
    std::fs::write(dir.join("slow.svg"), slow_svg()).unwrap();
    script(
        &dir.join("pdfinfo"),
        &format!(
            "#!/bin/sh\ncat > /dev/null\n{}\
             printf '%s\\n' 'Pages:           1' 'Encrypted:       no' 'Page    1 size:  8 x 8 pts'\n",
            record_limit(&record("pdfinfo"))
        ),
    );
    let renderer = script(
        &dir.join("pdftocairo"),
        &format!(
            "#!/bin/sh\ncat > /dev/null\n{}cat '{}'\n",
            record_limit(&record("pdftocairo")),
            dir.join("page.png").display()
        ),
    );
    let decoder = script(
        &dir.join("decoder.sh"),
        &format!(
            "#!/bin/sh\ncat > /dev/null\n{}exit 1\n",
            record_limit(&record("decoder"))
        ),
    );
    let encoder = script(
        &dir.join("encoder.sh"),
        &format!(
            "#!/bin/sh\ncat > /dev/null\n{}printf 'not a real video'\n",
            record_limit(&record("encoder"))
        ),
    );
    let caps = [
        ("CRABCROP_VIDEO_ENCODER_ADDRESS_SPACE_MAX_MB", 1536),
        ("CRABCROP_VIDEO_DECODER_ADDRESS_SPACE_MAX_MB", 1280),
        ("CRABCROP_PDF_RENDERER_ADDRESS_SPACE_MAX_MB", 768),
        ("CRABCROP_SVG_WORKER_ADDRESS_SPACE_MAX_MB", 2560),
    ];
    let mut command = binary(&dir);
    command
        .env("CRABCROP_PDF_RENDERER", &renderer)
        .env("CRABCROP_VIDEO_DECODER", &decoder)
        .env("CRABCROP_VIDEO_ENCODER", &encoder)
        .env("CRABCROP_PROCESS_TIMEOUT_SECS", "5");
    // No cap set: the boot lines name none, and the boot
    // check runs the PDF tools under the 3 GiB they always
    // had.
    {
        let (_server, _, _, printed) = common::serve_printing(&mut command);
        for line in [
            format!("PDF renderer: {} (timeout 30s)", renderer.display()),
            format!("video decoder: {} (timeout 30s)", decoder.display()),
            format!(
                "video encoder: {} (timeout 30s, output cap 64 MB)",
                encoder.display()
            ),
        ] {
            assert!(printed.contains(&line), "no {line:?} in {printed:#?}");
        }
    }
    assert_eq!(recorded(&record("pdfinfo")), [3072 * MB]);
    assert_eq!(recorded(&record("pdftocairo")), [3072 * MB]);
    for (name, mb) in caps {
        command.env(name, mb.to_string());
    }
    let (mut server, port, _, printed) = common::serve_printing(&mut command);
    // The boot lines name the caps set.
    for line in [
        format!(
            "PDF renderer: {} (timeout 30s, address space 768 MB)",
            renderer.display()
        ),
        format!(
            "video decoder: {} (timeout 30s, address space 1280 MB)",
            decoder.display()
        ),
        format!(
            "video encoder: {} (timeout 30s, output cap 64 MB, address space 1536 MB)",
            encoder.display()
        ),
    ] {
        assert!(printed.contains(&line), "no {line:?} in {printed:#?}");
    }
    // Its boot check ran both PDF tools, once each.
    assert_eq!(recorded(&record("pdfinfo")), [3072 * MB, 768 * MB]);
    assert_eq!(recorded(&record("pdftocairo")), [3072 * MB, 768 * MB]);
    // The probe fails, so the request fails, after the run.
    assert_ne!(
        get(port, "/unsafe/frame:poster/40x30/file:///clip.mp4"),
        200
    );
    assert_eq!(recorded(&record("decoder")), [1280 * MB]);
    assert_eq!(
        get(port, "/unsafe/filters:format(mp4)/file:///anim.gif"),
        200
    );
    assert_eq!(recorded(&record("encoder")), [1536 * MB]);
    // The worker: found among the server's children while it
    // renders, its limit read once it has set it, which is
    // when the limit differs from the one it inherited from
    // the server.
    let server_pid = server.0.id();
    let inherited = address_space_limit(server_pid);
    let render = std::thread::spawn(move || get(port, "/unsafe/file:///slow.svg"));
    let deadline = Instant::now() + LONG;
    let limit = loop {
        let worker = children_of(server_pid).into_iter().find(|&pid| {
            std::fs::read(format!("/proc/{pid}/cmdline"))
                .is_ok_and(|cmdline| cmdline.split(|&b| b == 0).any(|arg| arg == b"--svg-worker"))
        });
        if let Some(limit) = worker
            .and_then(address_space_limit)
            .filter(|&limit| Some(limit) != inherited)
        {
            break limit;
        }
        assert!(
            Instant::now() < deadline,
            "no SVG worker with a limit appeared"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(limit, 2560 * MB);
    // The render runs past the 5 s deadline.
    assert_eq!(render.join().unwrap(), 504);
    assert!(server.0.try_wait().unwrap().is_none(), "the server exited");
}

#[test]
fn a_cap_under_what_its_tool_needs_to_start_stops_the_boot() {
    // Each cap one megabyte under its least: the boot stops
    // before the server listens, with an error that names the
    // variable and the least it may be.
    let dir = common::scratch_dir("tool-caps-least");
    for (name, least, tool) in CAPS {
        let value = (least - 1).to_string();
        let (status, said) = boot_stops(binary(&dir).env(name, &value), &format!("{name}={value}"));
        assert_eq!(status.code(), Some(1), "{name}: {said}");
        assert_eq!(
            said.trim_end(),
            format!(
                "crabcrop: {name} must be between {least} (what the {tool} needs to start) \
                 and 65536, got {}",
                least - 1
            ),
            "{name}"
        );
    }
    // At the least, each boots.
    let mut command = binary(&dir);
    for (name, least, _) in CAPS {
        command.env(name, least.to_string());
    }
    let (_server, port, _) = common::serve(&mut command);
    assert_eq!(get(port, "/health"), 200);
}

#[test]
fn a_pdf_renderer_that_cannot_work_under_its_cap_stops_the_boot() {
    // The boot check renders under the configured cap, so a
    // renderer that cannot work in it stops the boot, and the
    // error names the cap's variable. The stand-in renderer
    // fails as one out of memory does whenever its limit is
    // under 1 GiB; its probe answers whatever its limit.
    let dir = common::scratch_dir("tool-caps-boot-check");
    std::fs::write(dir.join("page.png"), png()).unwrap();
    let check = "sleep 0.3\n\
         limit=$(grep '^Max address space' /proc/$$/limits | awk '{print $4}')\n\
         if [ \"$limit\" -lt 1073741824 ]; then echo 'Out of memory' >&2; exit 1; fi\n";
    script(
        &dir.join("pdfinfo"),
        "#!/bin/sh\ncat > /dev/null\n\
         printf '%s\\n' 'Pages:           1' 'Encrypted:       no' 'Page    1 size:  8 x 8 pts'\n",
    );
    let renderer = script(
        &dir.join("pdftocairo"),
        &format!(
            "#!/bin/sh\ncat > /dev/null\n{check}cat '{}'\n",
            dir.join("page.png").display()
        ),
    );
    let (status, said) = boot_stops(
        binary(&dir)
            .env("CRABCROP_PDF_RENDERER", &renderer)
            .env("CRABCROP_PDF_RENDERER_ADDRESS_SPACE_MAX_MB", "512"),
        "CRABCROP_PDF_RENDERER_ADDRESS_SPACE_MAX_MB=512",
    );
    assert_eq!(status.code(), Some(1), "{said}");
    assert!(
        said.contains(&format!(
            "crabcrop: cannot use CRABCROP_PDF_RENDERER ({}) under the 512 MB address-space cap \
             CRABCROP_PDF_RENDERER_ADDRESS_SPACE_MAX_MB sets: a one-page test document did not render",
            renderer.display()
        )),
        "{said}"
    );
    assert!(said.contains("Out of memory"), "{said}");
}
