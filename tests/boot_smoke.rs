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
