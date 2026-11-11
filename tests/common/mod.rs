//! Code the integration tests share. Each file under
//! `tests/` is a crate of its own, so each includes this
//! module (`mod common;`) and uses what it needs: not every
//! item is used by every test.

use std::io::{BufRead, BufReader, ErrorKind, Read};
use std::process::{Child, ChildStderr, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
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

/// How long to wait for a server's stderr to reach its end
/// once the server has exited.
const STDERR_GRACE: Duration = Duration::from_secs(10);

/// Kill the child on drop, so a failed assert cannot leak a
/// listening server into later runs. A guard from [`serve`]
/// also holds what the server writes to stderr.
pub(crate) struct ServerGuard(pub(crate) Child, Option<StderrDrain>);

impl ServerGuard {
    /// Guard `child`, whose pipes, if any, the caller reads.
    pub(crate) fn new(child: Child) -> Self {
        ServerGuard(child, None)
    }

    /// All the server wrote to stderr, for a server from
    /// [`serve`] that has exited, or was killed and reaped.
    /// It waits up to 10 seconds for the pipe to reach its
    /// end; called earlier, it returns what is written so
    /// far.
    pub(crate) fn stderr(&self) -> String {
        let drain = self
            .1
            .as_ref()
            .expect("a server from common::serve, whose stderr is read here");
        // The drain thread drops its sender at the pipe's end,
        // so this returns then, or at the timeout.
        let _ = drain.done.recv_timeout(STDERR_GRACE);
        drain.text()
    }
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A server's stderr, read on a thread as it is written.
struct StderrDrain {
    captured: Arc<Mutex<Vec<u8>>>,
    /// Disconnected once the pipe reached its end.
    done: Receiver<()>,
}

impl StderrDrain {
    fn start(mut pipe: ChildStderr) -> Self {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let (finished, done) = mpsc::channel::<()>();
        let sink = Arc::clone(&captured);
        std::thread::spawn(move || {
            let _finished = finished;
            let mut chunk = [0u8; 8192];
            loop {
                match pipe.read(&mut chunk) {
                    Ok(0) => return,
                    Ok(n) => sink.lock().unwrap().extend_from_slice(&chunk[..n]),
                    Err(e) if e.kind() == ErrorKind::Interrupted => {}
                    Err(_) => return,
                }
            }
        });
        StderrDrain { captured, done }
    }

    /// What has been read so far.
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.captured.lock().unwrap()).into_owned()
    }
}

/// Start `command`, the compiled binary with the
/// environment the test gives it, on a port the kernel
/// picks, and wait until it listens. Returns the server,
/// the port its listening line names, and the lines it
/// prints to stdout after that one, as it prints them.
///
/// The command runs with `PORT=0` (and no `CRABCROP_PORT`)
/// and its stdout and stderr piped here. The server binds a
/// free port itself and prints it. Picking a free port in
/// the test, releasing it, and handing it to the server
/// raced (issue #536): another process could take the port
/// in between, or another test's server could answer on it.
/// A server that exits before it listens, or does not
/// listen within 30 seconds, fails the test with what it
/// printed to both.
///
/// Threads read stdout and stderr to their ends whether or
/// not anyone looks at what they read, so a test that has
/// no use for the lines drops the receiver, and a server
/// that keeps printing never blocks on a full pipe.
/// [`ServerGuard::stderr`] returns what stderr carried.
pub(crate) fn serve(command: &mut Command) -> (ServerGuard, u16, Receiver<String>) {
    let (server, port, lines, _) = serve_printing(command);
    (server, port, lines)
}

/// [`serve`], plus the lines the server printed to stdout
/// before its listening line: the boot lines, in order.
pub(crate) fn serve_printing(
    command: &mut Command,
) -> (ServerGuard, u16, Receiver<String>, Vec<String>) {
    let mut child = command
        .env_remove("CRABCROP_PORT")
        .env("PORT", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawning crabcrop");
    let stdout = child.stdout.take().expect("a piped stdout");
    let stderr = StderrDrain::start(child.stderr.take().expect("a piped stderr"));
    let (sender, lines) = mpsc::channel();
    std::thread::spawn(move || {
        let mut stdout = BufReader::new(stdout);
        let mut line = Vec::new();
        loop {
            line.clear();
            match stdout.read_until(b'\n', &mut line) {
                Ok(0) => return,
                Ok(_) => {
                    let text = String::from_utf8_lossy(&line);
                    let text = text.strip_suffix('\n').unwrap_or(&text);
                    let text = text.strip_suffix('\r').unwrap_or(text);
                    // Read on after the receiver is gone, and
                    // past bytes that are not UTF-8: a pipe left
                    // unread fills, and the server's next print
                    // blocks, or fails once the pipe is closed.
                    let _ = sender.send(text.to_string());
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(_) => return,
            }
        }
    });
    let mut server = ServerGuard(child, Some(stderr));
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
                    return (server, port, lines, printed);
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
            panic!(
                "crabcrop exited before it listened: {status}\nstdout: {printed:#?}\nstderr: {}",
                server.stderr()
            );
        }
        if Instant::now() >= deadline {
            let stderr = server.1.as_ref().map(StderrDrain::text).unwrap_or_default();
            panic!(
                "crabcrop did not listen within {BOOT_DEADLINE:?}\nstdout: {printed:#?}\nstderr: {stderr}"
            );
        }
    }
}
