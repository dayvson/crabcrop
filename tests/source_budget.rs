//! The source byte budget (issue #403), end to end through
//! the binary: a burst of distinct cache misses against a
//! slow origin. Before the budget, every miss buffered its
//! whole source before any gate was consulted, so the
//! sources in memory grew with the number of requests;
//! with it, a miss reserves its share before it fetches a
//! byte, and the origin never sees more fetches at once
//! than the budget holds. A miss that cannot reserve within
//! the wait sheds with the documented `503 server_busy`.
//! An upload body is budgeted the same way before it is
//! read.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
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
/// allowlist that admits the local origin, and uploads to
/// `upload_dir`.
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

/// A fresh per-test, per-process directory.
fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "crabcrop-source-budget-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tiny_png() -> Vec<u8> {
    let img = image::RgbImage::from_pixel(64, 48, image::Rgb([200, 80, 40]));
    let mut png = std::io::Cursor::new(Vec::new());
    img.write_to(&mut png, image::ImageFormat::Png).unwrap();
    png.into_inner()
}

/// A local origin that records every path it is asked for
/// and how many bodies it is serving at once. A path
/// containing `stall` is held until `release` is set;
/// every other body goes out after `delay`.
struct Origin {
    port: u16,
    paths: Arc<Mutex<Vec<String>>>,
    peak: Arc<AtomicUsize>,
    release: Arc<AtomicBool>,
}

fn spawn_origin(delay: Duration) -> Origin {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    let paths = Arc::new(Mutex::new(Vec::new()));
    let serving = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let release = Arc::new(AtomicBool::new(false));
    let png = Arc::new(tiny_png());
    {
        let (paths, peak, release) = (paths.clone(), peak.clone(), release.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let (paths, serving, peak, release, png) = (
                    paths.clone(),
                    serving.clone(),
                    peak.clone(),
                    release.clone(),
                    png.clone(),
                );
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
                    let now = serving.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    paths.lock().unwrap().push(path.clone());
                    if path.contains("stall") {
                        let until = Instant::now() + Duration::from_secs(9);
                        while !release.load(Ordering::SeqCst) && Instant::now() < until {
                            std::thread::sleep(Duration::from_millis(20));
                        }
                    } else {
                        std::thread::sleep(delay);
                    }
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        png.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(&png);
                    serving.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });
    }
    Origin {
        port,
        paths,
        peak,
        release,
    }
}

fn metrics(port: u16) -> String {
    let (status, _, body) = get(port, "/metrics", "Authorization: Bearer budget-token\r\n");
    assert_eq!(status, 200);
    String::from_utf8(body).unwrap()
}

#[test]
fn a_burst_of_distinct_misses_never_fetches_more_than_the_budget_holds() {
    // Two fetch caps of budget (20 MB): however many
    // distinct misses arrive together, at most two sources
    // are being fetched at once. Before the budget all
    // eight were fetched, and buffered, at the same time.
    let origin = spawn_origin(Duration::from_millis(300));
    let uploads = scratch_dir("burst");
    let (_server, port) = boot(20, &uploads);
    let requests: Vec<_> = (0..8)
        .map(|i| {
            let path = format!("/unsafe/20x20/http://127.0.0.1:{}/s{i}.png", origin.port);
            std::thread::spawn(move || get(port, &path, "").0)
        })
        .collect();
    let statuses: Vec<u16> = requests.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(statuses, vec![200; 8], "every miss is served, in turn");
    assert_eq!(origin.paths.lock().unwrap().len(), 8);
    let peak = origin.peak.load(Ordering::SeqCst);
    assert!(
        peak <= 2,
        "{peak} sources fetched at once under a two-cap budget"
    );
    // Every share came back with its job; the exposition
    // reports the budget and what is held against it.
    let body = metrics(port);
    assert!(
        body.contains("crabcrop_source_buffer_capacity_bytes 20971520\n"),
        "{body}"
    );
    assert!(body.contains("crabcrop_source_buffer_bytes 0\n"), "{body}");
    let _ = std::fs::remove_dir_all(&uploads);
}

#[test]
fn a_miss_that_cannot_reserve_in_time_sheds_with_server_busy() {
    // One fetch cap of budget (10 MB), held by a miss whose
    // origin stalls: misses behind it wait out the bound
    // (the stills gate's five seconds), then shed with the
    // documented busy answer, never having fetched. An
    // image request and a meta request shed alike, and so
    // does an upload, whose body is never read or stored.
    let origin = spawn_origin(Duration::ZERO);
    let uploads = scratch_dir("shed");
    let (_server, port) = boot(10, &uploads);
    let stalled = {
        let path = format!("/unsafe/20x20/http://127.0.0.1:{}/stall.png", origin.port);
        std::thread::spawn(move || get(port, &path, "").0)
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while origin.paths.lock().unwrap().is_empty() {
        assert!(Instant::now() < deadline, "the stalled fetch never arrived");
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
    assert_eq!(
        *origin.paths.lock().unwrap(),
        vec!["/stall.png".to_string()],
        "a shed miss never reaches the origin"
    );
    origin.release.store(true, Ordering::SeqCst);
    assert_eq!(stalled.join().unwrap(), 200);
    let _ = std::fs::remove_dir_all(&uploads);
}
