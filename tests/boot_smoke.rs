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
