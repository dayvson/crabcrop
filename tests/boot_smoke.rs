//! Boot smoke: the compiled binary, launched with a
//! minimal valid environment, answers `GET /health`.
//! This is the one test that exercises process entry
//! itself (argument handling, `run()`, the listener)
//! rather than the construction functions `main.rs`
//! factors out of it: those have their own unit tests,
//! and this pins that the wiring between them serves.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

// Not every shared item is used here.
#[allow(dead_code)]
mod common;

use common::ServerGuard;

/// One HTTP/1.0 GET on `port`, with `headers` (each line
/// ending in CRLF): the response's status line.
fn status_line(port: u16, path: &str, headers: &str) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    write!(
        stream,
        "GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n{headers}\r\n"
    )
    .unwrap();
    let mut response = Vec::new();
    let _ = stream.read_to_end(&mut response);
    let response = String::from_utf8_lossy(&response);
    response.lines().next().unwrap_or("").to_string()
}

#[test]
fn the_binary_boots_and_answers_health() {
    let workdir = common::scratch_dir("boot-smoke");
    let (_server, port, _) = common::serve(
        Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .current_dir(&workdir)
            // A hermetic environment: only what boot needs, so
            // no operator variable from the test shell can leak
            // into the configuration under test.
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
            .stderr(Stdio::null()),
    );
    let status = status_line(port, "/health", "");
    assert!(status.contains("200"), "no 200 from /health: {status:?}");
}

#[test]
fn port_zero_binds_a_free_port_and_the_listening_line_names_it() {
    // Issue #536: the listening line printed the configured
    // port, so with PORT=0 it named port 0 while the server
    // held another. It names the bound port now. The
    // metrics token is this boot's own: a 200 for it on the
    // named port comes from this server, never from another
    // test's on a port the line got wrong.
    let token = format!("boot-port-zero-{}", std::process::id());
    let (_server, port, _) = common::serve(
        Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
            // What common::serve sets for every boot, spelled
            // out because it is what this test is about.
            .env("PORT", "0")
            .env("CRABCROP_METRICS_TOKEN", &token)
            .stderr(Stdio::null()),
    );
    assert_ne!(port, 0);
    let status = status_line(
        port,
        "/metrics",
        &format!("Authorization: Bearer {token}\r\n"),
    );
    assert!(
        status.contains("200"),
        "/metrics on port {port}: {status:?}"
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
        let child = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .arg(arg)
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
            .env("PORT", "0")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut process = ServerGuard(child);

        // No probe of a port here: with PORT=0 a server
        // would hold a port only it names, and a probe of any
        // other can reach another test's server and fail this
        // one for nothing. A binary that served would never
        // exit, so the deadline catches it, and a server
        // would print its listening line, which the empty
        // stdout below rules out.
        let deadline = Instant::now() + Duration::from_secs(30);
        let status = loop {
            if let Some(status) = process.0.try_wait().unwrap() {
                break status;
            }
            assert!(
                Instant::now() < deadline,
                "`crabcrop {arg}` did not exit before the deadline (it may be serving)"
            );
            std::thread::sleep(Duration::from_millis(50));
        };
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
    let (_server, port, _) = common::serve(
        Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
            .stderr(Stdio::null()),
    );
    let status = status_line(port, "/health", "");
    assert!(status.contains("200"), "no 200 from /health: {status:?}");
}

/// Boot with a minimal environment plus `env`, check
/// `/health`, stop the server, and return what it wrote
/// to stderr while booting.
fn boot_stderr(env: &[(&str, &str)]) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
    command
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
        .stderr(Stdio::piped());
    for (name, value) in env {
        command.env(name, value);
    }
    let (mut server, port, _) = common::serve(&mut command);
    let status = status_line(port, "/health", "");
    assert!(status.contains("200"), "no 200 from /health: {status:?}");
    // Boot writes its warnings before it listens, so they
    // are all in the pipe once the listening line is out.
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
/// the runtime, never about a missing model, and the
/// directory it lives in.
fn configured_model() -> (common::ScratchDir, std::path::PathBuf) {
    let dir = common::scratch_dir("boot-model");
    let model = dir.join("model.onnx");
    std::fs::write(&model, b"not a real model").unwrap();
    (dir, model)
}

#[test]
fn boot_with_a_model_but_no_loadable_runtime_fails_with_the_diagnostic() {
    // #180: with a model configured but no loadable ONNX
    // Runtime, boot used to die on a panic inside ort (exit
    // 101). ORT_DYLIB_PATH pointing nowhere real is exactly
    // the fallback resolution ort performs when
    // CRABCROP_ORT_DYLIB is unset, and it keeps the test
    // hermetic on hosts that do have a system runtime.
    let (_dir, model) = configured_model();
    let out = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
        .env("PORT", "0")
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
    let (_dir, model) = configured_model();
    let out = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
        .env("PORT", "0")
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
            .env("PORT", "0")
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
    // Port 0: nothing ever listens there (binding port 0
    // picks another), so the connection is refused.
    let dead_base = "http://127.0.0.1:0";
    let target = "50x50/http://127.0.0.1:9/src.png";

    // Unreachable server: the request error, reported.
    let out = run(&[
        "meta",
        "--base",
        dead_base,
        "--secret",
        "cli-smoke-secret",
        target,
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("crabcrop: request failed"), "{stderr}");

    // No secret anywhere: purge refuses before any request.
    let out = run(&["purge", "--base", dead_base, target]);
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
    let dir = common::scratch_dir("boot-no-roots");
    let bundle = dir.join("empty-ca.pem");
    std::fs::write(&bundle, b"").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
        .env("PORT", "0")
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
