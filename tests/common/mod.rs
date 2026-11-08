//! Code the integration tests share. Each file under
//! `tests/` is a crate of its own, so each includes this
//! module (`mod common;`) and uses what it needs: not every
//! item is used by every test.

use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

// The library's test helpers are not visible to an
// integration test: its scratch directories come in by
// path, the same file the library's and the binary's unit
// tests use. A test that boots the server and writes no
// files has no use for them.
#[path = "../../src/test_support/scratch.rs"]
mod scratch;
#[allow(unused_imports)]
pub(crate) use scratch::{ScratchDir, scratch_dir};

/// What the server prints, with the port after it, once
/// its listener is bound.
const LISTENING: &str = "crabcrop listening on port ";

/// How long a boot may take to reach its listening line.
const BOOT_DEADLINE: Duration = Duration::from_secs(30);

/// Kill the child on drop, so a failed assert cannot leak a
/// listening server into later runs.
pub(crate) struct ServerGuard(pub(crate) Child);

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Start `command`, the compiled binary with the
/// environment the test gives it, on a port the kernel
/// picks, and wait until it listens. Returns the server,
/// the port its listening line names, and the lines it
/// prints to stdout after that one, as it prints them.
///
/// The command runs with `PORT=0` (and no `CRABCROP_PORT`)
/// and its stdout piped here; where its stderr goes is the
/// caller's choice. The server binds a free port itself and
/// prints it. Picking a free port in the test, releasing
/// it, and handing it to the server raced (issue #536):
/// another process could take the port in between, or
/// another test's server could answer on it. A server that
/// exits before it listens, or does not listen within 30
/// seconds, fails the test with what it printed.
///
/// A thread reads stdout to its end whether or not the
/// lines are received, so a test that has no use for them
/// drops the receiver, and a server that keeps printing
/// never blocks on a full pipe.
pub(crate) fn serve(command: &mut Command) -> (ServerGuard, u16, Receiver<String>) {
    let mut child = command
        .env_remove("CRABCROP_PORT")
        .env("PORT", "0")
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawning crabcrop");
    let pipe = child.stdout.take().expect("a piped stdout");
    let (sender, lines) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(pipe).lines() {
            let Ok(line) = line else { return };
            // Read on after the receiver is gone: a pipe left
            // unread fills, and the server's next print blocks.
            let _ = sender.send(line);
        }
    });
    let mut server = ServerGuard(child);
    let deadline = Instant::now() + BOOT_DEADLINE;
    let mut printed = Vec::new();
    loop {
        match lines.recv_timeout(Duration::from_millis(50)) {
            Ok(line) => {
                if let Some(port) = line.strip_prefix(LISTENING) {
                    let port: u16 = port
                        .parse()
                        .unwrap_or_else(|_| panic!("a listening line without a port: {line:?}"));
                    assert_ne!(
                        port, 0,
                        "the listening line names port 0, not the bound one"
                    );
                    return (server, port, lines);
                }
                printed.push(line);
            }
            Err(RecvTimeoutError::Timeout) => {}
            // Stdout closed: the server is on its way out,
            // which the check below sees.
            Err(RecvTimeoutError::Disconnected) => std::thread::sleep(Duration::from_millis(50)),
        }
        if let Some(status) = server.0.try_wait().unwrap() {
            while let Ok(line) = lines.recv_timeout(Duration::from_secs(1)) {
                printed.push(line);
            }
            let mut stderr = String::new();
            if let Some(mut pipe) = server.0.stderr.take() {
                let _ = pipe.read_to_string(&mut stderr);
            }
            panic!(
                "crabcrop exited before it listened: {status}\nstdout: {printed:#?}\nstderr: {stderr}"
            );
        }
        assert!(
            Instant::now() < deadline,
            "crabcrop did not listen within {BOOT_DEADLINE:?}; stdout: {printed:#?}"
        );
    }
}
