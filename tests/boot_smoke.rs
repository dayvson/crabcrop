//! Boot smoke: the compiled binary, launched with a
//! minimal valid environment, answers `GET /health`.
//! This is the one test that exercises process entry
//! itself (argument handling, `run()`, the listener)
//! rather than the construction functions `main.rs`
//! factors out of it: those have their own unit tests,
//! and this pins that the wiring between them serves.

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

/// An ephemeral port, released before the server binds:
/// the small race is the standard one, and the deadline
/// loop below absorbs a lost race only by failing loudly.
fn free_port() -> u16 {
    TcpListener::bind(("127.0.0.1", 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[test]
fn the_binary_boots_and_answers_health() {
    let port = free_port();
    let workdir = std::env::temp_dir().join(format!("crabcrop-boot-smoke-{}", std::process::id()));
    std::fs::create_dir_all(&workdir).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
        .current_dir(&workdir)
        // A hermetic environment: only what boot needs, so
        // no operator variable from the test shell can leak
        // into the configuration under test.
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
        .env("PORT", port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut server = ServerGuard(child);

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut status_line = String::new();
    while Instant::now() < deadline {
        if let Some(status) = server.0.try_wait().unwrap() {
            panic!("the binary exited before serving: {status}");
        }
        if let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) {
            let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
            if stream
                .write_all(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
                .is_ok()
            {
                let mut response = String::new();
                if stream.read_to_string(&mut response).is_ok() && !response.is_empty() {
                    status_line = response.lines().next().unwrap_or("").to_string();
                    break;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        status_line.contains("200"),
        "no 200 from /health before the deadline; got status line {status_line:?}"
    );
}

#[test]
fn an_unknown_dash_argument_exits_with_an_error_and_never_serves() {
    // #470: main handed the CLI only bare words and the
    // help and version flags, so any other dash argument
    // fell through to the server start: with a secret set,
    // `crabcrop --verison` bound the port and served. The
    // server takes no arguments, so each of these must be
    // the CLI's error exit, with nothing ever listening.
    for arg in ["--verison", "-x", "--", "-", "--base", "--port=8080"] {
        let port = free_port();
        let child = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .arg(arg)
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
            .env("PORT", port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut process = ServerGuard(child);

        let deadline = Instant::now() + Duration::from_secs(30);
        let status = loop {
            if let Some(status) = process.0.try_wait().unwrap() {
                break status;
            }
            assert!(
                TcpStream::connect(("127.0.0.1", port)).is_err(),
                "`crabcrop {arg}` started a server on port {port}"
            );
            assert!(
                Instant::now() < deadline,
                "`crabcrop {arg}` neither exited nor served before the deadline"
            );
            std::thread::sleep(Duration::from_millis(50));
        };
        assert!(
            TcpStream::connect(("127.0.0.1", port)).is_err(),
            "`crabcrop {arg}`: something is listening on port {port}"
        );
        let mut stdout = String::new();
        let mut stderr = String::new();
        process
            .0
            .stdout
            .take()
            .unwrap()
            .read_to_string(&mut stdout)
            .unwrap();
        process
            .0
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut stderr)
            .unwrap();
        assert_eq!(status.code(), Some(1), "`crabcrop {arg}`: {stderr}");
        assert!(stdout.is_empty(), "`crabcrop {arg}`: {stdout}");
        assert!(
            stderr.starts_with(&format!("crabcrop: unknown verb {arg:?}")),
            "`crabcrop {arg}`: {stderr}"
        );
    }
}

#[test]
fn help_names_every_flag_the_cli_accepts() {
    // #470: the parser took `--key-id` while the help never
    // mentioned it. These are every flag parse_flags reads.
    let out = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
        .arg("--help")
        .env_clear()
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    let stdout = String::from_utf8(out.stdout).unwrap();
    for flag in [
        "--base",
        "--secret",
        "--key-id",
        "--expires",
        "--placeholders",
        "--detections",
    ] {
        assert!(
            stdout.contains(flag),
            "--help does not name {flag}:\n{stdout}"
        );
    }
}

#[test]
fn boot_with_no_source_configuration_serves_health() {
    // No allowlist, no private-blocking setting, no
    // any-source opt-in: boot succeeds on the defaults
    // (private-address blocking on) and answers health.
    // The posture lives in the fetch policy, not in a
    // boot gate.
    let port = free_port();
    let child = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
        .env("PORT", port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut server = ServerGuard(child);

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut status_line = String::new();
    while Instant::now() < deadline {
        if let Some(status) = server.0.try_wait().unwrap() {
            panic!("the binary exited before serving: {status}");
        }
        if let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) {
            let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
            if stream
                .write_all(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
                .is_ok()
            {
                let mut response = String::new();
                if stream.read_to_string(&mut response).is_ok() && !response.is_empty() {
                    status_line = response.lines().next().unwrap_or("").to_string();
                    break;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        status_line.contains("200"),
        "no 200 from /health before the deadline; got status line {status_line:?}"
    );
}

/// Boot with a minimal environment plus `env`, wait for
/// `/health`, stop the server, and return what it wrote
/// to stderr while booting.
fn boot_stderr(env: &[(&str, &str)]) -> String {
    let port = free_port();
    let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
    command
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
        .env("PORT", port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    for (name, value) in env {
        command.env(name, value);
    }
    let mut server = ServerGuard(command.spawn().unwrap());
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut healthy = false;
    while Instant::now() < deadline && !healthy {
        if let Some(status) = server.0.try_wait().unwrap() {
            panic!("the binary exited before serving: {status}");
        }
        if let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) {
            let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
            let mut response = String::new();
            healthy = stream
                .write_all(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
                .is_ok()
                && stream.read_to_string(&mut response).is_ok()
                && response.lines().next().unwrap_or("").contains("200");
        }
        if !healthy {
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    assert!(healthy, "no 200 from /health before the deadline");
    // Boot writes its warnings before it listens, so they
    // are all in the pipe once health answers.
    let _ = server.0.kill();
    let _ = server.0.wait();
    let mut stderr = String::new();
    server
        .0
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    stderr
}

#[test]
fn boot_warns_when_pinned_source_fetches_bypass_a_proxy() {
    // Issue #471: under the default policy every source
    // hostname is pinned to its validated address and
    // fetched directly, so a proxy in the environment is
    // bypassed. Boot says so, names the variable but never
    // its value (a proxy URL can carry credentials), and
    // still serves.
    let proxy = ("HTTPS_PROXY", "http://proxy-user:proxy-pass@127.0.0.1:9");
    let stderr = boot_stderr(&[proxy]);
    assert!(
        stderr
            .contains("warning: a proxy is configured (HTTPS_PROXY), but source fetches bypass it"),
        "{stderr}"
    );
    assert!(stderr.contains("CRABCROP_ALLOWED_SOURCES"), "{stderr}");
    assert!(!stderr.contains("proxy-pass"), "{stderr}");
    // With an allowlist nothing is pinned, every fetch
    // honours the proxy, and there is nothing to warn about.
    let stderr = boot_stderr(&[proxy, ("CRABCROP_ALLOWED_SOURCES", "images.example.com")]);
    assert!(!stderr.contains("proxy"), "{stderr}");
    // No proxy, no warning.
    let stderr = boot_stderr(&[]);
    assert!(!stderr.contains("proxy"), "{stderr}");
}

/// A model file that exists, so a boot failure is about
/// the runtime, never about a missing model.
fn configured_model() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("crabcrop-boot-model-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.onnx");
    std::fs::write(&model, b"not a real model").unwrap();
    model
}

#[test]
fn boot_with_a_model_but_no_loadable_runtime_fails_with_the_diagnostic() {
    // #180: with a model configured but no loadable ONNX
    // Runtime, boot used to die on a panic inside ort (exit
    // 101). ORT_DYLIB_PATH pointing nowhere real is exactly
    // the fallback resolution ort performs when
    // CRABCROP_ORT_DYLIB is unset, and it keeps the test
    // hermetic on hosts that do have a system runtime.
    let model = configured_model();
    let out = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
        .env("PORT", free_port().to_string())
        .env("CRABCROP_REMOVEBG_MODEL", &model)
        .env("ORT_DYLIB_PATH", "/nonexistent/libonnxruntime.so")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains(
            "crabcrop: cannot load the background-removal model: \
             cannot load the ONNX runtime from /nonexistent/libonnxruntime.so"
        ),
        "{stderr}"
    );
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
#[ignore = "host-dependent: only fails where no system ONNX Runtime is installed"]
fn boot_with_a_model_but_no_runtime_anywhere_fails_with_the_diagnostic() {
    // The unvarnished #180 scenario: a model configured, no
    // CRABCROP_ORT_DYLIB, no ORT_DYLIB_PATH, and no system
    // runtime for the loader to find. On a host with a
    // system runtime the boot succeeds instead, which is
    // why this is ignored by default. Run it where no
    // system runtime exists:
    // cargo test --test boot_smoke -- --ignored boot_with_a_model_but_no_runtime_anywhere
    let model = configured_model();
    let out = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
        .env("PORT", free_port().to_string())
        .env("CRABCROP_REMOVEBG_MODEL", &model)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("cannot load the ONNX runtime from"),
        "{stderr}"
    );
    assert!(stderr.contains("onnxruntime"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
fn boot_refuses_default_operations_with_metadata_or_over_the_cap() {
    // Issue #469: `exp:` and `maxage:` in CRABCROP_DEFAULT_OPS
    // used to boot cleanly and be dropped, so a default
    // lifetime never applied, and defaults alone over the
    // operation cap booted a server that refused every
    // request. Each is a boot error naming the variable now.
    // The child runs under a deadline: a regression boots and
    // serves, and must fail here instead of hanging the run.
    let over_cap = format!("filters:{}", vec!["invert"; 101].join(":"));
    let cases = [
        (
            "filters:format(webp)/maxage:60",
            "invalid maxage segment: maxage:60",
        ),
        (
            "exp:1893456000/filters:format(webp)",
            "invalid expiry segment: exp:1893456000",
        ),
        (
            over_cap.as_str(),
            "request has 101 operations, over the 100 operation limit",
        ),
    ];
    for (defaults, detail) in cases {
        let child = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
            .env("PORT", free_port().to_string())
            .env("CRABCROP_DEFAULT_OPS", defaults)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut server = ServerGuard(child);
        let deadline = Instant::now() + Duration::from_secs(30);
        let status = loop {
            if let Some(status) = server.0.try_wait().unwrap() {
                break status;
            }
            assert!(
                Instant::now() < deadline,
                "booted with CRABCROP_DEFAULT_OPS={defaults}"
            );
            std::thread::sleep(Duration::from_millis(50));
        };
        assert_eq!(status.code(), Some(1), "{defaults}");
        let mut stderr = String::new();
        server
            .0
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut stderr)
            .unwrap();
        assert!(
            stderr.contains(&format!("crabcrop: invalid CRABCROP_DEFAULT_OPS: {detail}")),
            "{defaults}: {stderr}"
        );
        let mut stdout = String::new();
        server
            .0
            .stdout
            .take()
            .unwrap()
            .read_to_string(&mut stdout)
            .unwrap();
        assert!(!stdout.contains("default operations:"), "{stdout}");
    }
}

#[test]
fn cli_verbs_fail_with_an_exit_code_and_the_error_on_stderr() {
    // The in-process contract (cli::exec returning Err) is
    // covered by unit tests; this pins the process shell
    // around it: exit code 1, the error on stderr prefixed
    // like every other crabcrop failure, nothing on stdout.
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .args(args)
            // Hermetic: no CRABCROP_BASE / CRABCROP_SECRET
            // from the test shell may leak in.
            .env_clear()
            .output()
            .unwrap()
    };
    let dead_base = format!("http://127.0.0.1:{}", free_port());
    let target = "50x50/http://127.0.0.1:9/src.png";

    // Unreachable server: the request error, reported.
    let out = run(&[
        "meta",
        "--base",
        dead_base.as_str(),
        "--secret",
        "cli-smoke-secret",
        target,
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("crabcrop: request failed"), "{stderr}");

    // No secret anywhere: purge refuses before any request.
    let out = run(&["purge", "--base", dead_base.as_str(), target]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("no secret"), "{stderr}");

    // An unknown verb is an error, never a server start.
    let out = run(&["sgin", target]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("unknown verb \"sgin\""), "{stderr}");

    // Help stays a success: usage on stdout, exit 0.
    let out = run(&["--help"]);
    assert_eq!(out.status.code(), Some(0));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("crabcrop purge"), "{stdout}");
}

/// Boot with an empty CA bundle: on Linux the TLS verifier
/// reads only `SSL_CERT_FILE` when it is set, finds no
/// roots, and refuses to build, so every HTTP client fails
/// to construct. The bucket stores build theirs before the
/// loader does.
#[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "android")))]
fn boot_without_ca_roots(vars: &[(&str, &str)]) -> std::process::Output {
    let dir = std::env::temp_dir().join(format!("crabcrop-boot-no-roots-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bundle = dir.join("empty-ca.pem");
    std::fs::write(&bundle, b"").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
        .env("PORT", free_port().to_string())
        .env("SSL_CERT_FILE", &bundle)
        .envs(vars.iter().copied())
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    out
}

#[test]
#[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "android")))]
fn boot_fails_cleanly_when_a_bucket_store_client_cannot_be_built() {
    // #455: the shared result store and the S3 upload store
    // built their HTTP client with `unwrap_or_default()`,
    // whose fallback, reqwest's `Client::new()`, panics on
    // the same failure (exit 101). Boot now stops with the
    // error and its cause instead.
    let cases: [(&[(&str, &str)], &str); 2] = [
        (
            &[("CRABCROP_RESULT_BUCKET", "results")],
            "crabcrop: cannot build the shared result store client: builder error: ",
        ),
        (
            &[
                ("CRABCROP_UPLOAD_BUCKET", "uploads"),
                ("CRABCROP_ALLOWED_BUCKETS", "uploads"),
            ],
            "crabcrop: cannot build the upload store client: builder error: ",
        ),
    ];
    for (vars, expected) in cases {
        let out = boot_without_ca_roots(vars);
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert_eq!(out.status.code(), Some(1), "{stderr}");
        assert!(stderr.contains(expected), "{stderr}");
        assert!(!stderr.contains("panicked"), "{stderr}");
    }
}

/// A slow shared bucket and the image origin on one
/// listener, for the shutdown tests (issue #463). It answers
/// listings (empty) and object reads (404) at once, serves
/// the source image, and holds every PUT, reporting it as it
/// arrives, until the test lets the PUTs through.
#[cfg(unix)]
struct SlowBucket {
    endpoint: String,
    /// One message per PUT, sent when it arrives.
    puts: std::sync::mpsc::Receiver<String>,
    /// Set to answer the held PUTs.
    release: std::sync::Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    /// PUTs answered so far.
    answered: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(unix)]
impl SlowBucket {
    fn start() -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Condvar, Mutex, mpsc};
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (put_tx, puts) = mpsc::channel();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let answered = Arc::new(AtomicUsize::new(0));
        let (held, done) = (release.clone(), answered.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let (put_tx, held, done) = (put_tx.clone(), held.clone(), done.clone());
                std::thread::spawn(move || {
                    let Some((method, target)) = read_request(&mut stream) else {
                        return;
                    };
                    let (status, body): (&str, &[u8]) = if target == "/img/src.jpg" {
                        ("200 OK", include_bytes!("fixtures/portrait.jpg"))
                    } else if method == "PUT" {
                        let _ = put_tx.send(target);
                        let (lock, cvar) = &*held;
                        let mut go = lock.lock().unwrap();
                        while !*go {
                            go = cvar.wait(go).unwrap();
                        }
                        done.fetch_add(1, Ordering::SeqCst);
                        ("200 OK", b"")
                    } else if target.contains("list-type=2") {
                        (
                            "200 OK",
                            b"<ListBucketResult><IsTruncated>false</IsTruncated></ListBucketResult>",
                        )
                    } else if method == "GET" {
                        ("404 Not Found", b"")
                    } else {
                        ("204 No Content", b"")
                    };
                    let head = format!(
                        "HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(body);
                });
            }
        });
        SlowBucket {
            endpoint,
            puts,
            release,
            answered,
        }
    }

    /// Answer every held PUT, and every later one at once.
    fn let_puts_through(&self) {
        let (lock, cvar) = &*self.release;
        *lock.lock().unwrap() = true;
        cvar.notify_all();
    }
}

/// Read one request's head and its body; the method and
/// the request target.
#[cfg(unix)]
fn read_request(stream: &mut TcpStream) -> Option<(String, String)> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        let n = stream.read(&mut chunk).ok().filter(|&n| n > 0)?;
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let length: usize = head
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    while buf.len() < head_end + length {
        let n = stream.read(&mut chunk).ok().filter(|&n| n > 0)?;
        buf.extend_from_slice(&chunk[..n]);
    }
    let mut parts = head.split_whitespace();
    Some((parts.next()?.to_string(), parts.next()?.to_string()))
}

/// A server over `bucket` with a shutdown grace of
/// `grace_secs`, which has rendered one image and has its
/// write-through to the bucket in flight; and the lines it
/// prints, as they come.
#[cfg(unix)]
fn serving_with_a_held_put(
    bucket: &SlowBucket,
    grace_secs: u64,
) -> (ServerGuard, std::sync::mpsc::Receiver<String>) {
    use std::io::BufRead;
    let port = free_port();
    let child = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
        .env("PORT", port.to_string())
        .env("CRABCROP_ALLOW_UNSAFE", "1")
        .env("CRABCROP_ALLOWED_SOURCES", "127.0.0.1")
        .env("CRABCROP_RESULT_BUCKET", "results")
        .env("CRABCROP_S3_ENDPOINT", &bucket.endpoint)
        .env("CRABCROP_SHUTDOWN_GRACE_SECS", grace_secs.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut server = ServerGuard(child);
    let stdout = server.0.stdout.take().unwrap();
    let (line_tx, lines) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout).lines() {
            let Ok(line) = line else { return };
            if line_tx.send(line).is_err() {
                return;
            }
        }
    });
    wait_for_line(&lines, "crabcrop listening on port");
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let request = format!(
        "GET /unsafe/40x30/{}/img/src.jpg HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n",
        bucket.endpoint
    );
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    let response = String::from_utf8_lossy(&response);
    assert!(
        response.lines().next().unwrap_or("").contains("200"),
        "{}",
        response.lines().next().unwrap_or("")
    );
    // The render's write-through reaches the bucket, which
    // holds it.
    let put = bucket
        .puts
        .recv_timeout(Duration::from_secs(30))
        .expect("the render's write-through reaches the bucket");
    assert!(put.starts_with("/results/"), "{put}");
    (server, lines)
}

/// Wait for a printed line that contains `needle`, and
/// return it.
#[cfg(unix)]
fn wait_for_line(lines: &std::sync::mpsc::Receiver<String>, needle: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut seen = Vec::new();
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        match lines.recv_timeout(left) {
            Ok(line) if line.contains(needle) => return line,
            Ok(line) => seen.push(line),
            Err(_) => break,
        }
    }
    panic!("no line with {needle:?}; saw {seen:#?}");
}

/// Send SIGTERM to the server, as a platform stopping it does.
#[cfg(unix)]
fn terminate(server: &ServerGuard) {
    rustix::process::kill_process(
        rustix::process::Pid::from_child(&server.0),
        rustix::process::Signal::TERM,
    )
    .unwrap();
}

/// Wait for the server to exit; its status and stderr.
#[cfg(unix)]
fn exit_of(server: &mut ServerGuard, within: Duration) -> (std::process::ExitStatus, String) {
    let deadline = Instant::now() + within;
    let status = loop {
        if let Some(status) = server.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "the server did not exit within {within:?}"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut stderr = String::new();
    server
        .0
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    (status, stderr)
}

#[test]
#[cfg(unix)]
fn sigterm_waits_for_an_in_flight_shared_write_then_exits() {
    // Issue #463: at SIGTERM the connections drained and the
    // process exited, dropping the write-throughs still in
    // flight. Now it waits for them, within its grace.
    let bucket = SlowBucket::start();
    let (mut server, lines) = serving_with_a_held_put(&bucket, 120);
    terminate(&server);
    wait_for_line(&lines, "shutdown: draining");
    // Draining, with the write held at the bucket: the
    // process is still there.
    assert!(server.0.try_wait().unwrap().is_none());
    assert_eq!(bucket.answered.load(std::sync::atomic::Ordering::SeqCst), 0);
    // The bucket answers; the write lands, and the process
    // exits on its own, long before the grace is spent.
    let released = Instant::now();
    bucket.let_puts_through();
    let (status, stderr) = exit_of(&mut server, Duration::from_secs(60));
    assert_eq!(status.code(), Some(0), "{stderr}");
    assert!(released.elapsed() < Duration::from_secs(60));
    assert_eq!(bucket.answered.load(std::sync::atomic::Ordering::SeqCst), 1);
    wait_for_line(&lines, "shutdown: done");
    assert!(!stderr.contains("dropped"), "{stderr}");
}

#[test]
#[cfg(unix)]
fn sigterm_stops_waiting_for_a_stalled_shared_write_at_the_grace() {
    // Issue #463: a write the bucket never answers holds the
    // shutdown only until the grace, CRABCROP_SHUTDOWN_GRACE_SECS
    // from the signal on, and then the process exits.
    let grace = Duration::from_secs(2);
    let bucket = SlowBucket::start();
    let (mut server, lines) = serving_with_a_held_put(&bucket, grace.as_secs());
    let signalled = Instant::now();
    terminate(&server);
    wait_for_line(&lines, "shutdown: draining");
    let (status, stderr) = exit_of(&mut server, Duration::from_secs(60));
    let took = signalled.elapsed();
    assert_eq!(status.code(), Some(0), "{stderr}");
    assert!(took >= grace, "exited after {took:?}, inside the grace");
    assert!(
        took < grace + Duration::from_secs(30),
        "exited after {took:?}"
    );
    assert!(
        stderr.contains("shutdown: 1 background writes still running at the deadline are dropped"),
        "{stderr}"
    );
    assert_eq!(bucket.answered.load(std::sync::atomic::Ordering::SeqCst), 0);
    wait_for_line(&lines, "shutdown: done");
    // Let the abandoned PUT's thread finish.
    bucket.let_puts_through();
}
