//! Meta mode's default, through the compiled binary
//! (issue #453). The server unit tests build their state
//! by hand, so only a booted binary proves what the real
//! configuration parsing does with `CRABCROP_META_MODE`:
//! unset, a meta request on a signed URL answers
//! `403 meta_disabled` without fetching or rendering
//! anything, and set to `1`, it answers the meta document.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

// Not every shared item is used here.
#[allow(dead_code)]
mod common;

const SECRET: &str = "meta-mode-secret-0123456789";

/// A local origin serving the portrait fixture to every
/// GET, counting the requests it answers: the count is how
/// the tests see whether the server fetched the source.
fn spawn_origin() -> (String, Arc<AtomicUsize>) {
    let path = format!("{}/tests/fixtures/portrait.jpg", env!("CARGO_MANIFEST_DIR"));
    let image = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let origin = format!("http://{}/portrait.jpg", listener.local_addr().unwrap());
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                match stream.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => request.extend_from_slice(&buf[..n]),
                }
            }
            counter.fetch_add(1, Ordering::SeqCst);
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                image.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&image);
        }
    });
    (origin, hits)
}

/// Boot the binary on a hermetic environment: the secret,
/// the port, the local origin allowed as a source, and the
/// given extra variables. Nothing else from the test shell
/// can reach the configuration under test.
fn boot(extra: &[(&str, &str)]) -> (common::ServerGuard, u16) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
    command
        .env_clear()
        .env("CRABCROP_SECRET", SECRET)
        .env("CRABCROP_ALLOWED_SOURCES", "127.0.0.1");
    for (name, value) in extra {
        command.env(name, value);
    }
    let (server, port, _) = common::serve(&mut command);
    (server, port)
}

struct Reply {
    status: u16,
    head: String,
    body: Vec<u8>,
}

impl Reply {
    /// A response header's value, by case-insensitive name.
    fn header(&self, name: &str) -> Option<&str> {
        self.head.lines().skip(1).find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim()
                .eq_ignore_ascii_case(name)
                .then_some(value.trim())
        })
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// One HTTP/1.0 GET, so the server closes the connection
/// and the reply is everything read until then.
fn get(port: u16, path: &str, accept: Option<&str>) -> Option<Reply> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .ok()?;
    let mut request = format!("GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n");
    if let Some(accept) = accept {
        request.push_str(&format!("Accept: {accept}\r\n"));
    }
    request.push_str("\r\n");
    stream.write_all(request.as_bytes()).ok()?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).ok()?;
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let status = head.split(' ').nth(1)?.parse().ok()?;
    Some(Reply {
        status,
        head,
        body: raw[split + 4..].to_vec(),
    })
}

/// The signed path for a remainder, minted the way every
/// client does it.
fn signed(remainder: &str) -> String {
    format!(
        "/{}/{remainder}",
        crabcrop_url::sign(SECRET.as_bytes(), remainder).unwrap()
    )
}

#[test]
fn meta_mode_is_off_by_default_and_refuses_before_any_work() {
    let (origin, hits) = spawn_origin();
    let (_server, port) = boot(&[]);
    let path = signed(&format!("100x0/filters:format(png)/{origin}"));

    // The render itself is unaffected: one fetch.
    let image = get(port, &path, Some("image/png")).unwrap();
    assert_eq!(image.status, 200, "{}", image.head);
    assert_eq!(hits.load(Ordering::SeqCst), 1);

    // Every spelling the server reads as a meta request is
    // refused with the code, and none of them reaches the
    // origin: the refusal comes before the fetch and the
    // render a meta request would otherwise pay for afresh.
    let opt_ins = format!("application/json; {}", ["placeholders=1"; 64].join("; "));
    let flood = format!("{}, application/json", ["image/webp;q=0.9"; 200].join(", "));
    let meta_spellings = [
        "application/json",
        "image/avif, application/json;q=0.1",
        "application/json; placeholders=1; detections=1; moderation=1; embeddings=1; ocr=1",
        // An unparseable q is not a refusal.
        "application/json;q=lots",
        // A refused entry does not cancel a later one.
        "application/json;q=0, application/json",
        opt_ins.as_str(),
        flood.as_str(),
    ];
    for accept in meta_spellings {
        for _ in 0..3 {
            let reply = get(port, &path, Some(accept)).unwrap();
            assert_eq!(reply.status, 403, "{accept}: {}", reply.head);
            let body = reply.text();
            assert!(
                body.contains(r#""code":"meta_disabled""#),
                "{accept}: {body}"
            );
            assert!(body.contains("CRABCROP_META_MODE=1"), "{accept}: {body}");
            assert_eq!(
                reply.header("cache-control"),
                Some("private, no-store"),
                "{accept}"
            );
        }
    }
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "a refused meta request fetched the source"
    );

    // A JSON entry refused with q=0 is not a meta request:
    // the image answers, from the cache.
    let reply = get(port, &path, Some("application/json;q=0, image/png")).unwrap();
    assert_eq!(reply.status, 200, "{}", reply.head);
    assert_eq!(hits.load(Ordering::SeqCst), 1);

    // The server survived all of it.
    assert_eq!(get(port, "/health", None).unwrap().status, 200);
}

#[test]
fn meta_mode_on_serves_the_meta_document() {
    let (origin, _hits) = spawn_origin();
    for value in ["1", "on"] {
        let (_server, port) = boot(&[("CRABCROP_META_MODE", value)]);
        let path = signed(&format!("100x0/filters:format(png)/{origin}"));
        let reply = get(port, &path, Some("application/json")).unwrap();
        assert_eq!(reply.status, 200, "{value}: {}", reply.head);
        assert_eq!(reply.header("content-type"), Some("application/json"));
        let body = reply.text();
        assert!(body.contains(r#""width":100"#), "{value}: {body}");
        assert!(body.contains(r#""format":"png""#), "{value}: {body}");
    }
}

#[test]
fn an_invalid_meta_mode_is_a_startup_error() {
    // A typo must not silently pick either side: the
    // binary refuses to start and names the variable.
    for value in ["maybe", ""] {
        let out = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .env_clear()
            .env("CRABCROP_SECRET", SECRET)
            .env("PORT", "0")
            .env("CRABCROP_META_MODE", value)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(1), "{value:?}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("CRABCROP_META_MODE"), "{value:?}: {stderr}");
    }
}
