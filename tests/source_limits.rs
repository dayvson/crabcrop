//! The source fetch bounds (issue #471), end to end
//! through the binary. Both used to be constants: 10 MiB
//! and 10 seconds, with no variable to move them. Now
//! `CRABCROP_SOURCE_MAX_MB` and
//! `CRABCROP_SOURCE_TIMEOUT_SECS` reach the loader: a
//! source over the configured cap answers
//! `502 source_too_large`, a raised cap serves a source
//! the default refuses, and an origin slower than the
//! configured timeout answers `504 source_timeout` at that
//! time, not at the default's.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Kill the child on drop, so a failed assert cannot
/// leak a listening server into later runs.
struct ServerGuard(Child);

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    TcpListener::bind(("127.0.0.1", 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// One raw HTTP/1.0 GET: (status, raw head, body).
fn get(port: u16, path: &str, headers: &str) -> (u16, String, Vec<u8>) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
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
        .expect("a complete response head");
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap();
    (status, head, raw[split + 4..].to_vec())
}

/// Boot the binary in unsafe mode with an allowlist that
/// admits the local origin, plus `env`.
fn boot(env: &[(&str, &str)]) -> (ServerGuard, u16) {
    let port = free_port();
    let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
    command
        .env_clear()
        .env("CRABCROP_SECRET", "source-limits-secret-0123456789")
        .env("PORT", port.to_string())
        .env("CRABCROP_ALLOW_UNSAFE", "1")
        .env("CRABCROP_ALLOWED_SOURCES", "127.0.0.1")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    for (name, value) in env {
        command.env(name, value);
    }
    let mut server = ServerGuard(command.spawn().unwrap());
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Some(status) = server.0.try_wait().unwrap() {
            panic!("the binary exited before serving: {status}");
        }
        if TcpStream::connect(("127.0.0.1", port)).is_ok() && get(port, "/health", "").0 == 200 {
            return (server, port);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("no /health before the deadline");
}

/// An uncompressed RGB TIFF: `width * height * 3` bytes of
/// pixels plus a small header, so its size is exact.
fn tiff(width: u32, height: u32) -> Vec<u8> {
    let img = image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8])
    });
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Tiff).unwrap();
    out.into_inner()
}

/// A local origin: a path containing `hang` reads the
/// request and then answers nothing for 8 s; any other
/// path whose name is in `bodies` gets that body as
/// `image/tiff`.
fn spawn_origin(bodies: Vec<(&'static str, Vec<u8>)>) -> u16 {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    let bodies = std::sync::Arc::new(bodies);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let bodies = bodies.clone();
            std::thread::spawn(move || {
                let mut request = Vec::new();
                let mut buf = [0u8; 1024];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    match stream.read(&mut buf) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => request.extend_from_slice(&buf[..n]),
                    }
                }
                let path = String::from_utf8_lossy(&request)
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("")
                    .to_string();
                if path.contains("hang") {
                    std::thread::sleep(Duration::from_secs(8));
                    return;
                }
                let Some((_, body)) = bodies.iter().find(|(name, _)| path.ends_with(name)) else {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    return;
                };
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/tiff\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(body);
            });
        }
    });
    port
}

/// The structured error code of a JSON error body.
fn error_code(body: &[u8]) -> String {
    let doc: serde_json::Value = serde_json::from_slice(body)
        .unwrap_or_else(|e| panic!("not JSON ({e}): {}", String::from_utf8_lossy(body)));
    doc["error"]["code"].as_str().unwrap_or("").to_string()
}

#[test]
fn the_fetch_cap_follows_crabcrop_source_max_mb() {
    const MIB: usize = 1024 * 1024;
    // 2000x1900 RGB is about 10.9 MiB, over the default
    // 10 MiB cap and under 12; 640x640 is about 1.2 MiB,
    // over a 1 MiB cap and far under the default.
    let big = tiff(2000, 1900);
    let medium = tiff(640, 640);
    assert!(
        big.len() > 10 * MIB && big.len() < 12 * MIB,
        "{}",
        big.len()
    );
    assert!(
        medium.len() > MIB && medium.len() < 2 * MIB,
        "{}",
        medium.len()
    );
    let origin = spawn_origin(vec![("big.tiff", big), ("medium.tiff", medium)]);
    let big_path = format!("/unsafe/50x50/http://127.0.0.1:{origin}/big.tiff");
    let medium_path = format!("/unsafe/50x50/http://127.0.0.1:{origin}/medium.tiff");
    let json = "Accept: application/json\r\n";

    // Unset, the cap stays the 10 MiB default: the big
    // source is refused, the medium one serves.
    {
        let (_server, port) = boot(&[]);
        let (status, _, body) = get(port, &big_path, "");
        assert_eq!(status, 502, "{}", String::from_utf8_lossy(&body));
        assert_eq!(
            String::from_utf8_lossy(&body),
            format!("source exceeds {} bytes", 10 * MIB)
        );
        let (status, head, _) = get(port, &medium_path, "");
        assert_eq!(status, 200, "{head}");
    }

    // Raised to 12 MB, the big source the default refuses
    // is fetched and rendered.
    {
        let (_server, port) = boot(&[("CRABCROP_SOURCE_MAX_MB", "12")]);
        let (status, head, body) = get(port, &big_path, "");
        assert_eq!(status, 200, "{head}\n{}", String::from_utf8_lossy(&body));
        assert!(
            head.to_ascii_lowercase().contains("content-type: image/"),
            "{head}"
        );
    }

    // Lowered to 1 MB, the medium source the default serves
    // is refused with the documented code, naming the
    // configured limit.
    {
        let (_server, port) = boot(&[("CRABCROP_SOURCE_MAX_MB", "1")]);
        let (status, _, body) = get(port, &medium_path, "");
        assert_eq!(status, 502, "{}", String::from_utf8_lossy(&body));
        assert_eq!(
            String::from_utf8_lossy(&body),
            format!("source exceeds {MIB} bytes")
        );
        let (status, _, body) = get(port, &medium_path, json);
        assert_eq!(status, 502);
        assert_eq!(error_code(&body), "source_too_large");
    }
}

#[test]
fn a_slow_source_times_out_at_crabcrop_source_timeout_secs() {
    // The origin answers nothing for 8 s. With a 2 s
    // timeout the request ends at 2 s, not before and not
    // at the 10 s default.
    let origin = spawn_origin(Vec::new());
    let (_server, port) = boot(&[("CRABCROP_SOURCE_TIMEOUT_SECS", "2")]);
    for (i, accept) in ["", "Accept: application/json\r\n"].iter().enumerate() {
        // A fresh path each time, so the second request
        // cannot join the first one's flight.
        let path = format!("/unsafe/50x50/http://127.0.0.1:{origin}/hang{i}.png");
        let started = Instant::now();
        let (status, _, body) = get(port, &path, accept);
        let elapsed = started.elapsed();
        assert_eq!(status, 504, "{}", String::from_utf8_lossy(&body));
        assert!(
            elapsed >= Duration::from_secs(2) && elapsed < Duration::from_secs(6),
            "answered after {elapsed:?}"
        );
        if accept.is_empty() {
            assert_eq!(String::from_utf8_lossy(&body), "source timed out");
        } else {
            assert_eq!(error_code(&body), "source_timeout");
        }
    }
}
