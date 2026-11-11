//! Boot smoke: the compiled binary, launched with a
//! minimal valid environment, answers `GET /health`.
//! This is the one test that exercises process entry
//! itself (argument handling, `run()`, the listener)
//! rather than the construction functions `main.rs`
//! factors out of it: those have their own unit tests,
//! and this pins that the wiring between them serves.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
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
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789"),
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
            .env("CRABCROP_METRICS_TOKEN", &token),
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
#[should_panic(expected = "stderr: crabcrop: invalid CRABCROP_DEFAULT_OPS")]
fn a_boot_that_exits_before_it_listens_fails_with_its_stderr() {
    // Issue #536 review: common::serve read stderr only when
    // the caller piped it, and most passed Stdio::null(), so
    // a boot that failed (a bad setting, a port in use) said
    // only `exit status: 1`. It reads stderr itself now and
    // puts it in the failure.
    common::serve(
        Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
            .env(
                "CRABCROP_DEFAULT_OPS",
                "exp:1893456000/filters:format(webp)",
            ),
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
        let mut process = ServerGuard::new(child);

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
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789"),
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
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789");
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
    server.stderr()
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

/// Boot with a minimal environment plus `env` and expect
/// the boot to fail: the exit code is 1, nothing listens,
/// and the result is what the server wrote to stderr. A
/// boot that serves instead fails the test at the deadline.
fn refused_boot(env: &[(&str, &str)]) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
    command
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
        .env("PORT", "0");
    for (name, value) in env {
        command.env(name, value);
    }
    let child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut server = ServerGuard::new(child);
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = server.0.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "booted with {env:?}");
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut stdout = String::new();
    let mut stderr = String::new();
    let _ = server.0.stdout.take().unwrap().read_to_string(&mut stdout);
    let _ = server.0.stderr.take().unwrap().read_to_string(&mut stderr);
    assert_eq!(status.code(), Some(1), "{env:?}: {stderr}");
    assert!(!stdout.contains("crabcrop listening"), "{env:?}: {stdout}");
    stderr
}

#[test]
fn placeholder_secrets_from_the_docs_stop_the_boot() {
    // Issue #588: the README, the VM and Container Apps
    // recipes and the compose metrics line shipped these
    // values, and each one booted. Boot now refuses each, and
    // the error names the variable but never the value.
    let secret = ("CRABCROP_SECRET", "boot-smoke-secret-0123456789");
    let cases: [(&str, &str); 6] = [
        ("CRABCROP_SECRET", "your-secret-change-me"),
        ("CRABCROP_SECRET", "replace-with-a-long-random-value"),
        ("CRABCROP_SECRET", "REPLACE_ME_signing_secret"),
        ("CRABCROP_SECRET", "local-dev-secret"),
        ("CRABCROP_SECRET_PREVIOUS", "your-secret-change-me"),
        (
            "CRABCROP_METRICS_TOKEN",
            "CHANGE-ME-generate-with-openssl-rand-hex-32",
        ),
    ];
    for (var, value) in cases {
        let env = if var == "CRABCROP_SECRET" {
            vec![(var, value)]
        } else {
            vec![secret, (var, value)]
        };
        let stderr = refused_boot(&env);
        assert!(
            stderr.starts_with(&format!("crabcrop: {var} is a placeholder")),
            "{var}: {stderr}"
        );
        assert!(!stderr.contains(value), "{var}: {stderr}");
    }
}

#[test]
fn a_short_metrics_token_warns_at_boot_and_still_serves() {
    // Issue #588: a metrics token only had to be non-empty.
    // Under 16 bytes it warns for this release; a later one
    // refuses it. The warning never quotes the token.
    let stderr = boot_stderr(&[("CRABCROP_METRICS_TOKEN", "fifteen-bytes!!")]);
    assert!(
        stderr.contains("warning: CRABCROP_METRICS_TOKEN is 15 bytes"),
        "{stderr}"
    );
    assert!(!stderr.contains("fifteen-bytes!!"), "{stderr}");
    let stderr = boot_stderr(&[("CRABCROP_METRICS_TOKEN", "sixteen-bytes!!!")]);
    assert!(!stderr.contains("CRABCROP_METRICS_TOKEN"), "{stderr}");
}

#[test]
fn a_blank_allowed_sources_stops_the_boot_naming_the_variable() {
    // Issue #590: a set value with no host in it read as no
    // allowlist, so a template whose host variable was empty
    // let every public host through without a word.
    for blank in ["", " ", ",", "."] {
        let stderr = refused_boot(&[("CRABCROP_ALLOWED_SOURCES", blank)]);
        assert!(
            stderr.starts_with("crabcrop: CRABCROP_ALLOWED_SOURCES is set but names no host"),
            "{blank:?}: {stderr}"
        );
    }
    // The any-source opt-out does not make a blank list fine.
    refused_boot(&[
        ("CRABCROP_ALLOWED_SOURCES", ","),
        ("CRABCROP_ALLOW_ANY_SOURCE", "1"),
    ]);
}

/// Boot with a minimal environment plus `env`, and return
/// the lines it printed to stdout before its listening
/// line, and what it wrote to stderr while booting.
fn boot_output(env: &[(&str, &str)]) -> (Vec<String>, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
    command
        .env_clear()
        .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789");
    for (name, value) in env {
        command.env(name, value);
    }
    let (mut server, _, _, printed) = common::serve_printing(&mut command);
    let _ = server.0.kill();
    let _ = server.0.wait();
    (printed, server.stderr())
}

#[test]
fn boot_prints_the_request_mode_and_warns_when_anyone_can_fetch_any_host() {
    // Issue #590: boot named no part of the request and
    // source posture. One line now names the request mode,
    // the source allowlist and the private-address check,
    // and unsigned URLs with no allowlist warn on stderr.
    let dir = common::scratch_dir("boot-posture");
    let presets = dir.join("presets.conf");
    std::fs::write(&presets, "thumb = 300x200\n").unwrap();
    let presets = presets.to_str().unwrap();
    let unsafe_on = ("CRABCROP_ALLOW_UNSAFE", "1");
    let only_presets = ("CRABCROP_ONLY_PRESETS", "1");
    let presets_file = ("CRABCROP_PRESETS_FILE", presets);
    // The extra environment, the posture line, and the
    // start of the one warning expected, if any.
    type Case<'a> = (Vec<(&'a str, &'a str)>, &'a str, Option<&'a str>);
    let cases: Vec<Case> = vec![
        (
            vec![],
            "request mode: signed; source allowlist: none; private-address check: on",
            None,
        ),
        (
            vec![unsafe_on],
            "request mode: unsafe; source allowlist: none; private-address check: on",
            Some("warning: unsafe mode is on with no CRABCROP_ALLOWED_SOURCES"),
        ),
        (
            vec![
                unsafe_on,
                (
                    "CRABCROP_ALLOWED_SOURCES",
                    "images.example.com, cdn.example.com",
                ),
            ],
            "request mode: unsafe; source allowlist: 2 hosts; private-address check: on",
            None,
        ),
        (
            vec![unsafe_on, ("CRABCROP_ALLOW_ANY_SOURCE", "1")],
            "request mode: unsafe; source allowlist: none; private-address check: off",
            Some("warning: unsafe mode is on with no CRABCROP_ALLOWED_SOURCES"),
        ),
        (
            vec![unsafe_on, only_presets, presets_file],
            "request mode: presets-only; source allowlist: none; private-address check: on",
            Some("warning: presets-only mode takes unsigned URLs with no CRABCROP_ALLOWED_SOURCES"),
        ),
        (
            vec![
                unsafe_on,
                only_presets,
                presets_file,
                ("CRABCROP_ALLOWED_SOURCES", "images.example.com"),
            ],
            "request mode: presets-only; source allowlist: 1 host; private-address check: on",
            None,
        ),
        (
            vec![only_presets, presets_file],
            "request mode: signed presets-only; source allowlist: none; private-address check: on",
            None,
        ),
        (
            vec![("CRABCROP_BLOCK_PRIVATE_SOURCES", "0")],
            "request mode: signed; source allowlist: none; private-address check: off",
            None,
        ),
    ];
    for (env, line, warning) in cases {
        let (printed, stderr) = boot_output(&env);
        let posture: Vec<&String> = printed
            .iter()
            .filter(|l| l.starts_with("request mode: "))
            .collect();
        assert_eq!(posture, [line], "{env:?}: {printed:#?}");
        match warning {
            Some(warning) => {
                let warnings: Vec<&str> = stderr
                    .lines()
                    .filter(|l| l.starts_with("warning: "))
                    .collect();
                assert_eq!(warnings.len(), 1, "{env:?}: {stderr}");
                assert!(warnings[0].starts_with(warning), "{env:?}: {stderr}");
                // The existing proxy warning test reads "proxy"
                // in stderr as its own warning.
                assert!(!stderr.contains("proxy"), "{env:?}: {stderr}");
            }
            None => assert!(!stderr.contains("warning:"), "{env:?}: {stderr}"),
        }
    }
}

#[test]
fn explain_off_answers_404_while_images_still_render() {
    // Issue #591: `/explain` could not be turned off. With
    // `CRABCROP_EXPLAIN=0` it answers 404, boot says so, and
    // the image route serves as before. Unset, the report
    // is served as it always was.
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let image = "/unsafe/40x30/file:///portrait.jpg";
    let explain = format!("/explain{image}");
    for (value, explain_status, says_off) in [(None, "200", false), (Some("0"), "404", true)] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
        command
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
            .env("CRABCROP_ALLOW_UNSAFE", "1")
            .env("CRABCROP_FILE_ROOT", fixtures);
        if let Some(value) = value {
            command.env("CRABCROP_EXPLAIN", value);
        }
        let (_server, port, _, printed) = common::serve_printing(&mut command);
        let status = status_line(port, image, "");
        assert!(status.contains("200"), "{value:?}: image {status:?}");
        let status = status_line(port, &explain, "");
        assert!(
            status.contains(explain_status),
            "{value:?}: explain {status:?}"
        );
        assert_eq!(
            printed.iter().any(|l| l == "explain: off"),
            says_off,
            "{value:?}: {printed:#?}"
        );
    }
    // A typo stops the boot and names the variable, rather
    // than leaving the route on.
    let stderr = refused_boot(&[("CRABCROP_EXPLAIN", "of")]);
    assert!(
        stderr.starts_with("crabcrop: invalid CRABCROP_EXPLAIN"),
        "{stderr}"
    );
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
        let mut server = ServerGuard::new(child);
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

#[test]
fn the_health_verb_probes_the_server_on_its_port() {
    // Issue #592: the images' HEALTHCHECK ran curl, the one
    // reason the runtime images carried it. `crabcrop health`
    // probes 127.0.0.1 at the server's port and exits 0 for a
    // 200, 1 otherwise, so the HEALTHCHECK runs it instead.
    let (_server, port, _) = common::serve(
        Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789"),
    );
    let health = |vars: &[(&str, &str)]| {
        Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .arg("health")
            .env_clear()
            .envs(vars.iter().copied())
            .output()
            .unwrap()
    };
    let port = port.to_string();
    let out = health(&[("PORT", &port)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "ok");
    // CRABCROP_PORT wins over PORT, as it does for the server.
    let out = health(&[("CRABCROP_PORT", &port), ("PORT", "1")]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    // CRABCROP_BIND=:: probes ::1, then 127.0.0.1 when the
    // connection to ::1 fails. This server listens on IPv4
    // only, so ::1 fails here as it does on a host with IPv6
    // off on loopback, and the probe passes at 127.0.0.1.
    let out = health(&[("PORT", &port), ("CRABCROP_BIND", "::")]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    // `::1` named alone gets no second address: a server
    // bound to it does not answer on 127.0.0.1.
    let out = health(&[("PORT", &port), ("CRABCROP_BIND", "::1")]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("crabcrop: request failed"), "{stderr}");

    // A port that never answers: the verb waits out its own
    // 2 s timeout, then exits 1 with the error on stderr. The
    // upper bound leaves room for a loaded runner; the cli.rs
    // tests keep the timeout under the HEALTHCHECK's 3 s.
    let silent = TcpListener::bind("127.0.0.1:0").unwrap();
    let silent_port = silent.local_addr().unwrap().port().to_string();
    let started = Instant::now();
    let out = health(&[("PORT", &silent_port)]);
    let elapsed = started.elapsed();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(out.stdout.is_empty());
    assert!(stderr.starts_with("crabcrop: request failed"), "{stderr}");
    assert!(
        elapsed >= Duration::from_secs(2) && elapsed < Duration::from_secs(10),
        "{elapsed:?}"
    );
    drop(silent);
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
    /// PUTs whose answer was written out so far.
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
                    let put = method == "PUT";
                    let (status, body): (&str, &[u8]) = if target == "/img/src.jpg" {
                        ("200 OK", include_bytes!("fixtures/portrait.jpg"))
                    } else if put {
                        let _ = put_tx.send(target);
                        let (lock, cvar) = &*held;
                        let mut go = lock.lock().unwrap();
                        while !*go {
                            go = cvar.wait(go).unwrap();
                        }
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
                    let written = stream
                        .write_all(head.as_bytes())
                        .and_then(|()| stream.write_all(body));
                    if put && written.is_ok() {
                        done.fetch_add(1, Ordering::SeqCst);
                    }
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
    let (server, port, lines) = serving_unsafe_urls(&[
        ("CRABCROP_RESULT_BUCKET", "results"),
        ("CRABCROP_S3_ENDPOINT", &bucket.endpoint),
        ("CRABCROP_SHUTDOWN_GRACE_SECS", &grace_secs.to_string()),
    ]);
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

/// A server that takes `unsafe` URLs for sources on
/// 127.0.0.1, with `envs` on top, once it listens; its
/// port, and the lines it prints, as they come.
#[cfg(unix)]
fn serving_unsafe_urls(
    envs: &[(&str, &str)],
) -> (ServerGuard, u16, std::sync::mpsc::Receiver<String>) {
    common::serve(
        Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
            .env("CRABCROP_ALLOW_UNSAFE", "1")
            .env("CRABCROP_ALLOWED_SOURCES", "127.0.0.1")
            .envs(envs.iter().copied()),
    )
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
    signal(server, rustix::process::Signal::TERM);
}

/// Send `sig` to the server.
#[cfg(unix)]
fn signal(server: &ServerGuard, sig: rustix::process::Signal) {
    rustix::process::kill_process(rustix::process::Pid::from_child(&server.0), sig).unwrap();
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
    (status, server.stderr())
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
    // process stays. A drain that does not wait exits within
    // milliseconds of that line, so a second of staying is
    // the proof; nothing else marks a wait that is going on.
    let watched = Instant::now();
    while watched.elapsed() < Duration::from_secs(1) {
        assert!(
            server.0.try_wait().unwrap().is_none(),
            "the server exited with its write-through still held"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(bucket.answered.load(std::sync::atomic::Ordering::SeqCst), 0);
    // The bucket answers; the write lands, and the process
    // exits on its own, long before the 120 s grace (and
    // the S3 client's own 10 s request timeout) is spent.
    let released = Instant::now();
    bucket.let_puts_through();
    let (status, stderr) = exit_of(&mut server, Duration::from_secs(60));
    assert_eq!(status.code(), Some(0), "{stderr}");
    let took = released.elapsed();
    assert!(
        took < Duration::from_secs(5),
        "exited {took:?} after the write landed"
    );
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

#[test]
#[cfg(unix)]
fn sigterm_cuts_off_a_hung_request_at_the_grace() {
    // Issue #463: the connection drain had no deadline, so
    // one request stuck on a slow origin held the shutdown
    // until the platform killed the process. Now the grace
    // bounds it too.
    let origin = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let origin_addr = origin.local_addr().unwrap();
    let (fetched_tx, fetched) = std::sync::mpsc::channel();
    // An origin that accepts the fetch and never answers.
    std::thread::spawn(move || {
        let mut held = Vec::new();
        for stream in origin.incoming() {
            let Ok(stream) = stream else { return };
            let _ = fetched_tx.send(());
            held.push(stream);
        }
    });
    let grace = Duration::from_secs(2);
    let (mut server, port, lines) = serving_unsafe_urls(&[
        ("CRABCROP_SOURCE_TIMEOUT_SECS", "600"),
        ("CRABCROP_SHUTDOWN_GRACE_SECS", &grace.as_secs().to_string()),
    ]);
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let request = format!(
        "GET /unsafe/40x30/http://{origin_addr}/img/hang.jpg HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).unwrap();
    fetched
        .recv_timeout(Duration::from_secs(30))
        .expect("the server fetches the source");
    let signalled = Instant::now();
    terminate(&server);
    wait_for_line(&lines, "shutdown: draining");
    let (status, stderr) = exit_of(&mut server, Duration::from_secs(60));
    let took = signalled.elapsed();
    assert_eq!(status.code(), Some(0), "{stderr}");
    assert!(took >= grace, "exited after {took:?}, inside the grace");
    assert!(
        took < grace + Duration::from_secs(8),
        "exited after {took:?}"
    );
    assert!(
        stderr.contains("shutdown: requests still in flight at the deadline are cut off"),
        "{stderr}"
    );
    // The request's miss belongs to the request: it is not
    // reported again as background work left behind.
    assert!(!stderr.contains("background writes"), "{stderr}");
    assert!(!stderr.contains("renders whose client left"), "{stderr}");
    wait_for_line(&lines, "shutdown: done");
}

#[test]
#[cfg(unix)]
fn a_second_signal_ends_the_drain_at_once() {
    // Issue #463 review: the signal handlers stay installed
    // once the drain starts, so with a long grace a second
    // Ctrl-C did nothing and only SIGKILL stopped the server.
    // Now a second signal abandons the drain.
    let bucket = SlowBucket::start();
    let (mut server, lines) = serving_with_a_held_put(&bucket, 120);
    terminate(&server);
    wait_for_line(&lines, "shutdown: draining");
    let signalled = Instant::now();
    signal(&server, rustix::process::Signal::INT);
    let (status, stderr) = exit_of(&mut server, Duration::from_secs(60));
    let took = signalled.elapsed();
    assert_eq!(status.code(), Some(0), "{stderr}");
    // Well inside the grace, and before the S3 client's own
    // 10 s request timeout could end the held write.
    assert!(took < Duration::from_secs(5), "exited {took:?} after it");
    assert!(
        stderr.contains("shutdown: a second signal ends the drain"),
        "{stderr}"
    );
    assert!(
        stderr.contains("1 background writes still running"),
        "{stderr}"
    );
    assert_eq!(bucket.answered.load(std::sync::atomic::Ordering::SeqCst), 0);
    bucket.let_puts_through();
}

/// A shared bucket that lists `count` purge tombstones and
/// never answers a read of one (issue #549): such a read
/// waits until the server gives up on it. A listing answers
/// at once, every other read is a 404, and every other
/// request succeeds with an empty body.
#[cfg(unix)]
fn bucket_with_hanging_tombstones(count: u64) -> String {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let mut listing = String::from("<ListBucketResult><IsTruncated>false</IsTruncated>");
    for i in 0..count {
        listing += &format!(
            "<Contents><Key>purges/{}-{:016x}-{i}</Key><ETag>&quot;{i}&quot;</ETag></Contents>",
            1_762_000_000_000 + i,
            0xabc
        );
    }
    listing += "</ListBucketResult>";
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let listing = listing.clone();
            std::thread::spawn(move || {
                let Some((method, target)) = read_request(&mut stream) else {
                    return;
                };
                if method == "GET" && target.starts_with("/results/purges/") {
                    let mut rest = [0u8; 64];
                    while matches!(stream.read(&mut rest), Ok(n) if n > 0) {}
                    return;
                }
                let (status, body) = if target.contains("list-type=2") {
                    ("200 OK", listing.as_bytes())
                } else if method == "GET" {
                    ("404 Not Found", &b""[..])
                } else {
                    ("200 OK", &b""[..])
                };
                let head = format!(
                    "HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream
                    .write_all(head.as_bytes())
                    .and_then(|()| stream.write_all(body));
            });
        }
    });
    endpoint
}

#[test]
#[cfg(unix)]
fn boot_serves_within_its_deadline_while_tombstone_reads_hang() {
    // Issue #549: boot read the shared bucket's purge
    // tombstones one at a time before it served, each up to
    // the store's 10 s request bound. With five tombstones and
    // hanging reads, the listening line came after 50 s, past
    // the Helm liveness probe's restart at about 45 s, and
    // nothing was counted or logged. The read now ends within
    // 10 s whatever the bucket does. These 40 would take 30 s
    // even at the 16 reads at a time it now keeps in flight.
    let endpoint = bucket_with_hanging_tombstones(40);
    let token = format!("boot-tombstones-{}", std::process::id());
    let began = Instant::now();
    let (mut server, port, _) = common::serve(
        Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
            .env("CRABCROP_RESULT_BUCKET", "results")
            .env("CRABCROP_S3_ENDPOINT", &endpoint)
            .env("CRABCROP_METRICS_TOKEN", &token),
    );
    let took = began.elapsed();
    // Boot waits for the refresh, as the docs say, and for
    // its 10 s deadline at most.
    assert!(took >= Duration::from_secs(9), "listened after {took:?}");
    assert!(took < Duration::from_secs(20), "listened after {took:?}");
    assert!(status_line(port, "/health", "").contains("200"));
    // The reads the deadline left count once together on
    // /metrics, not once per tombstone: a read that failed
    // before it (the client's own 10 s bound can win the
    // race) counts on its own, and at most 16 run at once.
    // The refresh logs them on one line.
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    write!(
        stream,
        "GET /metrics HTTP/1.0\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\n\r\n"
    )
    .unwrap();
    let mut metrics = String::new();
    let _ = stream.read_to_string(&mut metrics);
    let errors: u64 = metrics
        .lines()
        .find_map(|l| l.strip_prefix("crabcrop_cache_tier_errors_total{tier=\"shared\"} "))
        .and_then(|n| n.trim().parse().ok())
        .unwrap_or_else(|| panic!("no shared tier error count: {metrics}"));
    assert!((1..=17).contains(&errors), "{metrics}");
    // Issue #577: /metrics shows that the replica listed all 40
    // and has read none of them, and when it listed them.
    assert!(
        metrics.contains("\ncrabcrop_cache_shared_tombstones_unread 40\n"),
        "{metrics}"
    );
    let listed: f64 = metrics
        .lines()
        .find_map(|l| l.strip_prefix("crabcrop_cache_shared_tombstones_listed_timestamp_seconds "))
        .and_then(|n| n.trim().parse().ok())
        .unwrap_or_else(|| panic!("no tombstone listing time: {metrics}"));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    assert!((now - listed).abs() < 60.0, "listed at {listed}, now {now}");
    let _ = server.0.kill();
    let _ = server.0.wait();
    let stderr = server.stderr();
    let line = stderr
        .lines()
        .find(|l| l.starts_with("shared storage: cannot read purge tombstones: "))
        .unwrap_or_else(|| panic!("no tombstone read line: {stderr}"));
    assert!(
        line.contains(
            "of 40 not done within 10s; kept as last read: 0; not in force until read: 40"
        ),
        "{line}"
    );
    // Issue #577: the line ends with the three tombstones not
    // in force that began first, by name.
    assert!(
        line.ends_with(
            "not in force until read: 40; oldest not in force: \
             purges/1762000000000-0000000000000abc-0, \
             purges/1762000000001-0000000000000abc-1, \
             purges/1762000000002-0000000000000abc-2 and 37 more"
        ),
        "{line}"
    );
}

/// A connection to the server on `port`.
fn connect(port: u16) -> TcpStream {
    let stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    stream
}

/// Read one response from `stream`, its head and a body of
/// `Content-Length` bytes, and leave the connection open:
/// the status line.
fn read_response(stream: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        match stream.read(&mut chunk) {
            Ok(0) => panic!("closed before a response: {buf:?}"),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) => panic!("no response: {e}"),
        }
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
                .map(|v| v.trim().parse().unwrap())
        })
        .expect("a content-length");
    while buf.len() < head_end + length {
        match stream.read(&mut chunk) {
            Ok(0) => panic!("closed inside a body"),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) => panic!("no body: {e}"),
        }
    }
    head.lines().next().unwrap_or("").to_string()
}

/// One keep-alive `GET /health` on `stream`: the status
/// line. The connection stays open.
fn health_on(stream: &mut TcpStream) -> String {
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        .unwrap();
    read_response(stream)
}

/// Read from `stream` until the server closes it: how long
/// that took, or `None` when it is still open after
/// `within`. Bytes the server sends first are read past.
fn wait_for_close(stream: &mut TcpStream, within: Duration) -> Option<Duration> {
    let began = Instant::now();
    let mut chunk = [0u8; 4096];
    loop {
        let left = within.checked_sub(began.elapsed())?;
        stream
            .set_read_timeout(Some(left.max(Duration::from_millis(1))))
            .unwrap();
        match stream.read(&mut chunk) {
            Ok(0) => return Some(began.elapsed()),
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                return None;
            }
            // A reset is a close too.
            Err(_) => return Some(began.elapsed()),
        }
    }
}

/// A server with a minimal environment plus `env`, once it
/// listens: the server, its port, and its boot lines.
fn serving(env: &[(&str, &str)]) -> (ServerGuard, u16, Vec<String>) {
    let (server, port, _, printed) = common::serve_printing(
        Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
            .envs(env.iter().copied()),
    );
    (server, port, printed)
}

#[test]
fn a_connection_that_never_finishes_its_request_head_is_closed_at_the_header_timeout() {
    // Issue #607: the server never set a timer on its
    // connections, so one that sent nothing, stopped half
    // way through a request head, or sent the head a byte at
    // a time was held open for good, and a client could hold
    // the process's whole file table that way. The header
    // timeout counts from the connection's start, so bytes
    // that keep arriving do not extend it.
    let (_server, port, _) = serving(&[("CRABCROP_HEADER_TIMEOUT_SECS", "1")]);
    let mut silent = connect(port);
    let mut partial = connect(port);
    partial
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n")
        .unwrap();
    let mut drip = connect(port);
    let mut writer = drip.try_clone().unwrap();
    std::thread::spawn(move || {
        for byte in b"GET /health HTTP/1.1\r\nX-Slow: one byte at a time, five a second\r\n" {
            if writer.write_all(&[*byte]).is_err() {
                return;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    });
    for (name, stream) in [
        ("silent", &mut silent),
        ("partial head", &mut partial),
        ("byte-at-a-time head", &mut drip),
    ] {
        assert!(
            wait_for_close(stream, Duration::from_secs(10)).is_some(),
            "the {name} connection is still open 10 s after it started, past a 1 s header timeout"
        );
    }
    // The server still serves.
    assert!(status_line(port, "/health", "").contains("200"));
}

#[test]
fn an_idle_keep_alive_connection_is_closed_at_the_idle_timeout_and_not_before() {
    // Issue #607: a keep-alive connection that went quiet
    // after its response was held open for good. It now
    // closes at the idle timeout, which is apart from the
    // header timeout: an edge that keeps the connection in
    // its pool past the header timeout still finds it open.
    let (_server, port, _) = serving(&[
        ("CRABCROP_HEADER_TIMEOUT_SECS", "1"),
        ("CRABCROP_IDLE_TIMEOUT_SECS", "4"),
    ]);
    let mut stream = connect(port);
    assert!(health_on(&mut stream).contains("200"));
    std::thread::sleep(Duration::from_secs(2));
    assert!(
        health_on(&mut stream).contains("200"),
        "a connection idle past the header timeout but inside the idle timeout serves"
    );
    let closed = wait_for_close(&mut stream, Duration::from_secs(15));
    let closed = closed.expect(
        "the connection is still open 15 s after its last response, past a 4 s idle timeout",
    );
    assert!(
        closed >= Duration::from_secs(3),
        "closed {closed:?} after its last response, before the 4 s idle timeout"
    );
}

#[test]
fn at_the_connection_cap_a_new_connection_is_served_once_a_slot_frees() {
    // Issue #607: nothing capped the open connections. At
    // CRABCROP_MAX_CONNECTIONS the server stops accepting:
    // a new connection waits in the kernel's queue, and is
    // served as soon as an open one closes.
    let (_server, port, _) = serving(&[("CRABCROP_MAX_CONNECTIONS", "2")]);
    let mut first = connect(port);
    assert!(health_on(&mut first).contains("200"));
    let mut second = connect(port);
    assert!(health_on(&mut second).contains("200"));
    let mut third = connect(port);
    third
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        .unwrap();
    third
        .set_read_timeout(Some(Duration::from_millis(1500)))
        .unwrap();
    let mut byte = [0u8; 1];
    match third.read(&mut byte) {
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ) => {}
        other => panic!("a third connection was answered with two open and a cap of 2: {other:?}"),
    }
    drop(first);
    third
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    assert!(read_response(&mut third).contains("200"));
    // The second is still served on its slot.
    assert!(health_on(&mut second).contains("200"));
}

#[test]
#[cfg(unix)]
fn silent_connections_past_the_open_file_limit_do_not_shut_out_health() {
    // Issue #607, as reproduced there: under a low open-file
    // limit, connections that send nothing took every file
    // descriptor and a fresh /health timed out for as long
    // as they stayed. The default cap is three quarters of
    // the limit, so the server keeps descriptors for its own
    // work, and the header timeout frees the slots.
    let (_server, port, _, printed) = common::serve_printing(
        Command::new("/bin/sh")
            .arg("-c")
            .arg("ulimit -n 64 && exec \"$0\"")
            .arg(env!("CARGO_BIN_EXE_crabcrop"))
            .env_clear()
            .env("CRABCROP_SECRET", "boot-smoke-secret-0123456789")
            .env("CRABCROP_HEADER_TIMEOUT_SECS", "2"),
    );
    let silent: Vec<TcpStream> = (0..80).map(|_| connect(port)).collect();
    let mut fresh = connect(port);
    fresh
        .set_read_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    fresh
        .write_all(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        .unwrap();
    let mut response = Vec::new();
    let _ = fresh.read_to_end(&mut response);
    let response = String::from_utf8_lossy(&response);
    assert!(
        response.starts_with("HTTP/1.0 200"),
        "/health with 80 silent connections open and a limit of 64 files: {response:?}"
    );
    drop(silent);
    assert!(
        printed
            .iter()
            .any(|l| l.starts_with("connections: ") && l.contains("at most 48 open")),
        "{printed:#?}"
    );
}

#[test]
#[cfg(target_os = "linux")]
fn crabcrop_bind_listens_on_the_named_address_only() {
    // Issue #592: the server always listened on every
    // interface, so on a VM the port the proxy fronts was
    // open to anyone who could reach the host. 127.0.0.2 is
    // a loopback address on Linux that a listener on every
    // interface answers on, and one on 127.0.0.1 does not.
    let (_server, port, _) = serving(&[]);
    assert!(
        TcpStream::connect(("127.0.0.2", port)).is_ok(),
        "by default the server listens on every IPv4 interface"
    );
    let (_server, port, printed) = serving(&[("CRABCROP_BIND", "127.0.0.1")]);
    assert!(status_line(port, "/health", "").contains("200"));
    let refused = TcpStream::connect(("127.0.0.2", port)).map(|_| ());
    assert_eq!(
        refused.map_err(|e| e.kind()),
        Err(std::io::ErrorKind::ConnectionRefused),
        "CRABCROP_BIND=127.0.0.1 answers on 127.0.0.2"
    );
    assert!(
        printed
            .iter()
            .any(|l| l.starts_with("connections: bind 127.0.0.1, ")),
        "{printed:#?}"
    );
}

#[test]
fn health_probes_pass_under_the_tightest_connection_bounds() {
    // Issues #592 and #607: the images' HEALTHCHECK runs
    // `crabcrop health`, and the Kubernetes and platform
    // recipes probe `GET /health` over HTTP. A probe sends
    // its whole head at once and closes after the answer, so
    // the smallest header and idle timeouts do not cut it
    // off, and its slot frees when it closes: with a cap of
    // one connection, probe after probe still passes.
    let (_server, port, _) = serving(&[
        ("CRABCROP_HEADER_TIMEOUT_SECS", "1"),
        ("CRABCROP_IDLE_TIMEOUT_SECS", "1"),
        ("CRABCROP_MAX_CONNECTIONS", "1"),
    ]);
    let started = Instant::now();
    let port_text = port.to_string();
    for _ in 0..3 {
        let out = Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .arg("health")
            .env_clear()
            .env("PORT", &port_text)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(0), "{stderr}");
    }
    // A kubelet probe: HTTP/1.1 with `Connection: close`.
    for _ in 0..3 {
        let mut stream = connect(port);
        stream
            .write_all(
                b"GET /health HTTP/1.1\r\nHost: 10.0.0.1:3000\r\nUser-Agent: kube-probe/1.31\r\n\
                  Accept: */*\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
        let mut response = Vec::new();
        let _ = stream.read_to_end(&mut response);
        let response = String::from_utf8_lossy(&response);
        assert!(response.starts_with("HTTP/1.1 200 "), "{response}");
        assert!(status_line(port, "/health", "").contains("200"));
    }
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "{:?}",
        started.elapsed()
    );
}

#[test]
#[cfg(target_os = "linux")]
fn the_health_verb_probes_the_address_crabcrop_bind_names() {
    // Issues #592 and #607: `crabcrop health` probed
    // 127.0.0.1, and a server bound to one other address
    // does not answer there, so the images' HEALTHCHECK
    // failed on a healthy server. The verb reads
    // CRABCROP_BIND as the server does and probes that
    // address. 127.0.0.2 is a loopback address on Linux.
    let (_server, port, _) = serving(&[("CRABCROP_BIND", "127.0.0.2")]);
    let health = |vars: &[(&str, &str)]| {
        Command::new(env!("CARGO_BIN_EXE_crabcrop"))
            .arg("health")
            .env_clear()
            .envs(vars.iter().copied())
            .output()
            .unwrap()
    };
    let port = port.to_string();
    let out = health(&[("PORT", &port), ("CRABCROP_BIND", "127.0.0.2")]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "ok");
    // Without the setting the verb probes 127.0.0.1, where
    // this server does not listen.
    let out = health(&[("PORT", &port)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("crabcrop: request failed"), "{stderr}");
    // A value the server refuses is an error here too.
    let out = health(&[("PORT", &port), ("CRABCROP_BIND", "localhost")]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.starts_with("crabcrop: invalid CRABCROP_BIND: \"localhost\""),
        "{stderr}"
    );
}

#[test]
fn bad_connection_settings_stop_the_boot_naming_the_variable() {
    // Issue #607 and #592: each new setting is checked at
    // boot like the others. An empty CRABCROP_BIND, as a
    // template with an unset value writes it, is refused
    // rather than read as every interface.
    let cases = [
        (
            "CRABCROP_BIND",
            "localhost",
            "crabcrop: CRABCROP_BIND must be an IP address",
        ),
        (
            "CRABCROP_BIND",
            "",
            "crabcrop: CRABCROP_BIND must be an IP address",
        ),
        (
            "CRABCROP_BIND",
            "127.0.0.1:3000",
            "crabcrop: CRABCROP_BIND must be an IP address",
        ),
        (
            "CRABCROP_HEADER_TIMEOUT_SECS",
            "0",
            "crabcrop: CRABCROP_HEADER_TIMEOUT_SECS must be between 1 and 600",
        ),
        (
            "CRABCROP_HEADER_TIMEOUT_SECS",
            "601",
            "crabcrop: CRABCROP_HEADER_TIMEOUT_SECS must be between 1 and 600",
        ),
        (
            "CRABCROP_IDLE_TIMEOUT_SECS",
            "0",
            "crabcrop: CRABCROP_IDLE_TIMEOUT_SECS must be between 1 and 86400",
        ),
        (
            "CRABCROP_MAX_CONNECTIONS",
            "0",
            "crabcrop: CRABCROP_MAX_CONNECTIONS must be between 1 and 1000000",
        ),
        (
            "CRABCROP_MAX_CONNECTIONS",
            "-1",
            "crabcrop: invalid CRABCROP_MAX_CONNECTIONS",
        ),
    ];
    for (var, value, error) in cases {
        let stderr = refused_boot(&[(var, value)]);
        assert!(stderr.starts_with(error), "{var}={value:?}: {stderr}");
    }
}
