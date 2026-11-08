//! The source buffer budget (issue #403), end to end
//! through the binary. Before the budget, every cache miss
//! buffered its whole source before any gate was consulted,
//! so the sources in memory grew with the number of
//! requests. Now a miss charges its body to the budget as
//! the bytes arrive: an origin that answers and then stalls
//! holds only what it delivered, so stalled origins cannot
//! starve other requests, while a budget genuinely full of
//! buffered bytes makes new misses wait, then shed with the
//! documented `503 server_busy`.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// The library's test helpers are not visible to an
// integration test: its scratch directories come in by
// path. Not every item is used here.
#[allow(dead_code)]
#[path = "../src/test_support/scratch.rs"]
mod test_support;

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

const SECRET: &str = "source-budget-secret-0123456789";

/// One raw HTTP/1.0 exchange: (status, raw head, body).
fn get(port: u16, path: &str, headers: &str) -> (u16, String, Vec<u8>) {
    exchange(port, "GET", path, headers, &[])
}

fn exchange(
    port: u16,
    method: &str,
    path: &str,
    headers: &str,
    body: &[u8],
) -> (u16, String, Vec<u8>) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let length = if body.is_empty() {
        String::new()
    } else {
        format!("Content-Length: {}\r\n", body.len())
    };
    write!(
        stream,
        "{method} {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n{length}{headers}\r\n"
    )
    .unwrap();
    stream.write_all(body).unwrap();
    // A reset after the response (a shed upload's unread
    // body) must not hide what was already received.
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

/// Boot the binary with a source budget of `budget_mb`, an
/// allowlist that admits the local origin, uploads to
/// `upload_dir`, and meta mode on (off by default, issue
/// #453), so a meta request reaches the budget.
fn boot(budget_mb: u32, upload_dir: &std::path::Path) -> (ServerGuard, u16) {
    let port = free_port();
    let child = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
        .env_clear()
        .env("CRABCROP_SECRET", SECRET)
        .env("CRABCROP_UPLOAD_DIR", upload_dir)
        .env("PORT", port.to_string())
        .env("CRABCROP_ALLOW_UNSAFE", "1")
        .env("CRABCROP_ALLOWED_SOURCES", "127.0.0.1")
        .env("CRABCROP_METRICS_TOKEN", "budget-token")
        .env("CRABCROP_SOURCE_BUFFER_MAX_MB", budget_mb.to_string())
        .env("CRABCROP_META_MODE", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut server = ServerGuard(child);
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

/// A fresh per-test directory.
fn scratch_dir(tag: &str) -> test_support::ScratchDir {
    test_support::scratch_dir(&format!("source-budget-{tag}"))
}

fn tiny_png() -> Vec<u8> {
    let img = image::RgbImage::from_pixel(64, 48, image::Rgb([200, 80, 40]));
    let mut png = std::io::Cursor::new(Vec::new());
    img.write_to(&mut png, image::ImageFormat::Png).unwrap();
    png.into_inner()
}

/// A local origin. A path containing `stall` answers with
/// a declared length of `declared` bytes, sends `sent` of
/// them, then holds the connection until `release` is set
/// (or 9 s pass); every other path gets the tiny PNG. It
/// records every path it is asked for.
struct Origin {
    port: u16,
    paths: Arc<Mutex<Vec<String>>>,
    release: Arc<AtomicBool>,
}

fn spawn_origin(declared: usize, sent: usize) -> Origin {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    let paths = Arc::new(Mutex::new(Vec::new()));
    let release = Arc::new(AtomicBool::new(false));
    let png = Arc::new(tiny_png());
    {
        let (paths, release) = (paths.clone(), release.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let (paths, release, png) = (paths.clone(), release.clone(), png.clone());
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
                    paths.lock().unwrap().push(path.clone());
                    if path.contains("stall") {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {declared}\r\nConnection: close\r\n\r\n"
                        );
                        let _ = stream.write_all(head.as_bytes());
                        let _ = stream.write_all(&vec![0u8; sent]);
                        let until = Instant::now() + Duration::from_secs(9);
                        while !release.load(Ordering::SeqCst) && Instant::now() < until {
                            std::thread::sleep(Duration::from_millis(20));
                        }
                        return;
                    }
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        png.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(&png);
                });
            }
        });
    }
    Origin {
        port,
        paths,
        release,
    }
}

/// Wait until the origin has been asked for `n` paths.
fn wait_for_requests(origin: &Origin, n: usize) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while origin.paths.lock().unwrap().len() < n {
        assert!(
            Instant::now() < deadline,
            "the origin never saw {n} requests"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn metrics(port: u16) -> String {
    let (status, _, body) = get(port, "/metrics", "Authorization: Bearer budget-token\r\n");
    assert_eq!(status, 200);
    String::from_utf8(body).unwrap()
}

/// The `crabcrop_source_buffer_bytes` gauge.
fn held_bytes(port: u16) -> u64 {
    metrics(port)
        .lines()
        .find_map(|l| l.strip_prefix("crabcrop_source_buffer_bytes "))
        .and_then(|v| v.parse().ok())
        .expect("the gauge is exported")
}

#[test]
fn stalled_origins_do_not_block_a_normal_request() {
    // Four misses whose origin answers with a 10 MB
    // Content-Length and then sends nothing, under a budget
    // of two fetch caps (20 MB). Charging the fetch cap up
    // front would let two of them hold the whole budget for
    // the fetch timeout, and the normal request below would
    // shed after its 5 s wait. Charged as bytes arrive, a
    // stall holds its declared length only for the one-second
    // grant, then only what it delivered: nothing.
    let origin = spawn_origin(10 * 1024 * 1024, 0);
    let uploads = scratch_dir("stall");
    let (_server, port) = boot(20, &uploads);
    // An upload client that announces 10 MB, sends 1000
    // bytes, and stalls holds one charge step, not the
    // upload cap.
    let sig = crabcrop::url::upload_signature(SECRET.as_bytes(), "stall.png");
    let mut stalled_upload = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        stalled_upload,
        "PUT /upload/{sig}/stall.png HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\n\r\n",
        10 * 1024 * 1024
    )
    .unwrap();
    stalled_upload.write_all(&[0u8; 1000]).unwrap();
    let stalled: Vec<_> = (0..4)
        .map(|i| {
            let path = format!(
                "/unsafe/20x20/http://127.0.0.1:{}/stall{i}.png",
                origin.port
            );
            std::thread::spawn(move || get(port, &path, "").0)
        })
        .collect();
    wait_for_requests(&origin, 4);
    let start = Instant::now();
    let path = format!("/unsafe/20x20/http://127.0.0.1:{}/normal.png", origin.port);
    let (status, head, _) = get(port, &path, "");
    let waited = start.elapsed();
    assert_eq!(status, 200, "{head}");
    assert!(waited < Duration::from_secs(4), "served after {waited:?}");
    // Once their grants end, the stalls, still open, hold
    // nothing they did not receive: the origin stalls
    // nothing, the upload one 256 KiB step for its 1000
    // bytes.
    let deadline = Instant::now() + Duration::from_secs(3);
    while held_bytes(port) > 256 * 1024 {
        assert!(
            Instant::now() < deadline,
            "stalls still hold {} bytes",
            held_bytes(port)
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(stalled_upload);
    origin.release.store(true, Ordering::SeqCst);
    for t in stalled {
        assert_ne!(t.join().unwrap(), 200, "a stalled source never renders");
    }
    assert_eq!(held_bytes(port), 0, "every charge came back");
    let body = metrics(port);
    assert!(
        body.contains("crabcrop_source_buffer_capacity_bytes 20971520\n"),
        "{body}"
    );
    assert!(!uploads.join("stall.png").exists());
    let _ = std::fs::remove_dir_all(&uploads);
}

#[test]
fn a_budget_full_of_buffered_bytes_sheds_new_misses_with_server_busy() {
    // One fetch cap of budget (10 MB), filled by a miss
    // whose origin delivers all but ten bytes of a 10 MB
    // body and stalls: those bytes really are in memory, so
    // they stay charged. Misses behind it wait out the bound
    // (the stills gate's five seconds), then shed with the
    // documented busy answer. An image request, a meta
    // request, and an upload (never stored) shed alike.
    const CAP: usize = 10 * 1024 * 1024;
    let origin = spawn_origin(CAP, CAP - 10);
    let uploads = scratch_dir("shed");
    let (_server, port) = boot(10, &uploads);
    let stalled = {
        let path = format!("/unsafe/20x20/http://127.0.0.1:{}/stall.png", origin.port);
        std::thread::spawn(move || get(port, &path, "").0)
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while held_bytes(port) < (CAP - 10) as u64 {
        assert!(Instant::now() < deadline, "the stalled body never arrived");
        std::thread::sleep(Duration::from_millis(20));
    }
    let start = Instant::now();
    let image = {
        let path = format!("/unsafe/20x20/http://127.0.0.1:{}/image.png", origin.port);
        std::thread::spawn(move || get(port, &path, ""))
    };
    let meta = {
        let path = format!("/unsafe/20x20/http://127.0.0.1:{}/meta.png", origin.port);
        std::thread::spawn(move || get(port, &path, "Accept: application/json\r\n"))
    };
    let upload = std::thread::spawn(move || {
        let sig = crabcrop::url::upload_signature(SECRET.as_bytes(), "shed.png");
        exchange(
            port,
            "PUT",
            &format!("/upload/{sig}/shed.png"),
            "",
            &tiny_png(),
        )
        .0
    });
    let (status, head, body) = image.join().unwrap();
    assert_eq!(status, 503, "{head}");
    assert!(
        head.to_ascii_lowercase().contains("retry-after: 1"),
        "{head}"
    );
    assert_eq!(
        String::from_utf8_lossy(&body),
        "server busy: too many source images held in memory"
    );
    let (status, _, body) = meta.join().unwrap();
    assert_eq!(status, 503);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["code"], "server_busy");
    assert_eq!(upload.join().unwrap(), 503);
    assert!(
        !uploads.join("shed.png").exists(),
        "a shed upload is not stored"
    );
    let waited = start.elapsed();
    assert!(
        waited >= Duration::from_millis(4500),
        "shed after {waited:?}, before the wait bound"
    );
    origin.release.store(true, Ordering::SeqCst);
    assert_ne!(stalled.join().unwrap(), 200);
    assert_eq!(held_bytes(port), 0, "the stalled charge came back");
    let _ = std::fs::remove_dir_all(&uploads);
}
