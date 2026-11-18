//! The request log on the compiled binary's standard
//! output (issue #642): what a reader of the pipe gets
//! from the writer thread and from `CRABCROP_LOG_SYNC`,
//! and what the server does while the pipe is not read,
//! once it is closed, and at shutdown.

#![cfg(unix)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Not every shared item is used here.
#[allow(dead_code)]
mod common;

use common::ServerGuard;

const SECRET: &str = "request-log-secret-0123456789";

/// Long enough for anything that must happen on a loaded
/// machine; a test that passes never waits it out.
const LONG: Duration = Duration::from_secs(60);

/// How many health checks the server answers, one after
/// the other, while a request stays unanswered, before the
/// request counts as waiting. A fixed time would take a
/// slow answer on a loaded machine for a wait; a request
/// that is only slow is answered long before the same
/// server, on the same machine, has answered this many
/// others.
const HEALTH_CHECKS: usize = 200;

fn binary() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_crabcrop"));
    command.env_clear().env("CRABCROP_SECRET", SECRET);
    command
}

/// One HTTP/1.0 GET on `port`: the response's head, or
/// `None` if no answer came within `within`.
fn get_within(port: u16, path: &str, headers: &str, within: Duration) -> Option<String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.set_read_timeout(Some(within)).unwrap();
    write!(
        stream,
        "GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n{headers}\r\n"
    )
    .unwrap();
    let mut response = Vec::new();
    match stream.read_to_end(&mut response) {
        Ok(_) => {}
        // The read timed out with nothing, or part way.
        Err(_) if response.is_empty() => return None,
        Err(e) => panic!("{path}: {e}"),
    }
    let response = String::from_utf8_lossy(&response);
    Some(response.split("\r\n\r\n").next().unwrap_or("").to_string())
}

/// One GET: the status code of its answer.
fn status(port: u16, path: &str, headers: &str) -> u16 {
    let head = get_within(port, path, headers, LONG).expect("an answer");
    status_of(&head)
}

fn status_of(head: &str) -> u16 {
    head.split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status in {head:?}"))
}

/// An image request that is refused at once and logs a
/// line of about 60 KB: its operations name a watermark
/// with a very long URL, and the log line carries the
/// operations. `n` marks the request, and its line.
fn long_line_request(n: usize) -> String {
    format!(
        "/unsafe/filters:watermark(http%3A%2F%2F127.0.0.1%3A9%2F{}-{n:04}.png,0,0,50)/http://127.0.0.1:9/x.png",
        "w".repeat(60_000)
    )
}

/// The request number a [`long_line_request`]'s log line
/// carries.
fn number_in(line: &str) -> usize {
    let record: serde_json::Value =
        serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {} bytes", line.len()));
    let ops = record["ops"].as_str().unwrap();
    let mark = ops.rfind(".png").unwrap();
    ops[mark - 4..mark].parse().unwrap()
}

/// A server whose standard output is a pipe that only the
/// test reads, when it chooses to.
struct Unread {
    server: ServerGuard,
    port: u16,
    /// The pipe, read up to the listening line.
    stdout: Option<BufReader<ChildStdout>>,
    stderr: Arc<Mutex<Vec<u8>>>,
    /// Disconnected once stderr has been read to its end.
    stderr_ended: Receiver<()>,
}

impl Unread {
    fn boot(env: &[(&str, &str)]) -> Self {
        let mut child = binary()
            .env("PORT", "0")
            .envs(env.iter().copied())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawning crabcrop");
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let mut pipe = child.stderr.take().unwrap();
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&stderr);
        let (ended, stderr_ended) = mpsc::channel::<()>();
        std::thread::spawn(move || {
            let _ended = ended;
            let mut chunk = [0u8; 8192];
            while let Ok(n @ 1..) = pipe.read(&mut chunk) {
                sink.lock().unwrap().extend_from_slice(&chunk[..n]);
            }
        });
        let server = ServerGuard::new(child);
        // The boot lines, on a thread, so a boot that never
        // listens fails the test instead of hanging it.
        let (listening, listened) = mpsc::channel();
        std::thread::spawn(move || {
            let mut line = String::new();
            loop {
                line.clear();
                match stdout.read_line(&mut line) {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {}
                }
                if let Some(port) = line.trim_end().strip_prefix("crabcrop listening on port ") {
                    let _ = listening.send((port.parse::<u16>().unwrap(), stdout));
                    return;
                }
            }
        });
        let (port, stdout) = listened.recv_timeout(LONG).unwrap_or_else(|_| {
            panic!(
                "crabcrop did not listen: {}",
                String::from_utf8_lossy(&stderr.lock().unwrap())
            )
        });
        Unread {
            server,
            port,
            stdout: Some(stdout),
            stderr,
            stderr_ended,
        }
    }

    /// Start reading standard output: its lines, as they
    /// come, to the end of the pipe.
    fn read(&mut self) -> Receiver<String> {
        let mut stdout = self.stdout.take().expect("standard output is not read yet");
        let (sender, lines) = mpsc::channel();
        std::thread::spawn(move || {
            let mut line = String::new();
            loop {
                line.clear();
                match stdout.read_line(&mut line) {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {
                        let _ = sender.send(line.trim_end_matches('\n').to_string());
                    }
                }
            }
        });
        lines
    }

    /// Close the read end of standard output.
    fn close(&mut self) {
        self.stdout = None;
    }

    fn stderr(&self) -> String {
        String::from_utf8_lossy(&self.stderr.lock().unwrap()).into_owned()
    }

    /// Wait until stderr holds `needle`.
    fn stderr_says(&self, needle: &str) -> String {
        let deadline = Instant::now() + LONG;
        loop {
            let stderr = self.stderr();
            if stderr.contains(needle) {
                return stderr;
            }
            assert!(Instant::now() < deadline, "no {needle:?} in {stderr}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn terminate(&self) {
        rustix::process::kill_process(
            rustix::process::Pid::from_child(&self.server.0),
            rustix::process::Signal::TERM,
        )
        .unwrap();
    }

    /// Wait for the server to exit: its status, and how long
    /// it took from now. Its stderr has been read to the end
    /// when this returns: the server's last words are
    /// written just before it exits, and the thread that
    /// reads them can be behind.
    fn exit(&mut self) -> (std::process::ExitStatus, Duration) {
        let waited = Instant::now();
        loop {
            if let Some(status) = self.server.0.try_wait().unwrap() {
                let took = waited.elapsed();
                let _ = self.stderr_ended.recv_timeout(LONG);
                return (status, took);
            }
            assert!(
                waited.elapsed() < LONG,
                "the server did not exit: {}",
                self.stderr()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Send [`long_line_request`]s, one after the other,
    /// until one waits (see [`HEALTH_CHECKS`]): how many
    /// were answered, and the one that waits. `most` is
    /// more than can fit. The health checks only have to
    /// be answered: once the queue has stayed full for 5
    /// seconds, which a slow machine can take to run them,
    /// they answer 503 (issue #707).
    fn fill(&self, most: usize) -> (usize, std::thread::JoinHandle<Option<String>>) {
        let port = self.port;
        for n in 0..most {
            let (answer, answered) = mpsc::channel();
            let request = std::thread::spawn(move || {
                let head = get_within(port, &long_line_request(n), "", LONG);
                let _ = answer.send(());
                head
            });
            let mut checks = 0;
            while answered.recv_timeout(Duration::from_millis(1)).is_err() {
                let health = status(port, "/health", "");
                assert!(health == 200 || health == 503, "/health answered {health}");
                checks += 1;
                if checks == HEALTH_CHECKS {
                    return (n, request);
                }
            }
            assert_eq!(status_of(&request.join().unwrap().unwrap()), 403);
        }
        panic!("{most} requests were answered with nobody reading standard output");
    }
}

/// Every line `lines` gives up to the end of the pipe.
fn rest_of(lines: &Receiver<String>) -> Vec<String> {
    let deadline = Instant::now() + LONG;
    let mut all = Vec::new();
    loop {
        match lines.recv_timeout(Duration::from_millis(100)) {
            Ok(line) => all.push(line),
            Err(mpsc::RecvTimeoutError::Disconnected) => return all,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                assert!(Instant::now() < deadline, "standard output never ended");
            }
        }
    }
}

/// A log's lines with what differs from run to run taken
/// out: a request line's clock fields, and a request ID
/// the server generated.
fn comparable(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .map(|line| {
            let Ok(mut record) = serde_json::from_str::<serde_json::Value>(line) else {
                return line.clone();
            };
            if record.get("ts").is_none() {
                return line.clone();
            }
            for clock in ["ts", "verify_ms", "load_ms", "process_ms", "total_ms"] {
                assert!(record[clock].is_number(), "{line}");
                record[clock] = 0.into();
            }
            let id = record["request_id"].as_str().unwrap().to_string();
            if id.len() == 16 && id.bytes().all(|b| b.is_ascii_hexdigit()) {
                record["request_id"] = "generated".into();
            }
            record.to_string()
        })
        .collect()
}

#[test]
fn a_replay_leaves_the_same_log_from_the_writer_thread_as_from_synchronous_lines() {
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let image = "/unsafe/40x30/file:///portrait.jpg";
    let mut logs = Vec::new();
    for sync in [false, true] {
        let mut command = binary();
        command
            .env("CRABCROP_ALLOW_UNSAFE", "1")
            .env("CRABCROP_FILE_ROOT", fixtures);
        if sync {
            command.env("CRABCROP_LOG_SYNC", "1");
        }
        let (mut server, port, lines, printed) = common::serve_printing(&mut command);
        // Boot names the setting when it is on.
        assert_eq!(
            printed
                .iter()
                .any(|line| line == "request log: synchronous"),
            sync,
            "{printed:#?}"
        );
        // One request after the other: a miss, a hit, a
        // revalidation, three refusals, a hit under an ID
        // the client chose, a health check (which logs
        // nothing), a purge, and the miss after it.
        let head = get_within(port, image, "", LONG).unwrap();
        assert_eq!(status_of(&head), 200, "{head}");
        let etag = head
            .lines()
            .find_map(|line| line.strip_prefix("etag: "))
            .unwrap_or_else(|| panic!("no etag in {head}"));
        assert_eq!(status(port, image, ""), 200);
        assert_eq!(
            status(port, image, &format!("If-None-Match: {etag}\r\n")),
            304
        );
        let unsigned = format!("/{}/40x30/file:///portrait.jpg", "0".repeat(64));
        assert_eq!(status(port, &unsigned, ""), 403);
        assert_eq!(
            status(port, "/unsafe/filters:nope()/file:///portrait.jpg", ""),
            400
        );
        assert_eq!(status(port, "/unsafe/40x30/file:///absent.jpg", ""), 404);
        assert_eq!(
            status(port, image, "X-Request-Id: chosen-by-client\r\n"),
            200
        );
        assert_eq!(status(port, "/health", ""), 200);
        let purge = binary()
            .env("CRABCROP_BASE", format!("http://127.0.0.1:{port}"))
            .args(["purge", "40x30/file:///portrait.jpg"])
            .output()
            .unwrap();
        assert!(purge.status.success(), "{purge:?}");
        assert_eq!(status(port, image, ""), 200);
        // A clean shutdown writes every line before its last.
        rustix::process::kill_process(
            rustix::process::Pid::from_child(&server.0),
            rustix::process::Signal::TERM,
        )
        .unwrap();
        let waited = Instant::now();
        let exit = loop {
            if let Some(exit) = server.0.try_wait().unwrap() {
                break exit;
            }
            assert!(waited.elapsed() < LONG, "the server did not exit");
            std::thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(exit.code(), Some(0), "{}", server.stderr());
        logs.push(rest_of(&lines));
    }
    let (queued, sync) = (comparable(&logs[0]), comparable(&logs[1]));
    assert_eq!(queued, sync);
    // And the log is what the requests were, in their order,
    // ended by the shutdown's lines.
    let told: Vec<String> = queued
        .iter()
        .map(
            |line| match serde_json::from_str::<serde_json::Value>(line) {
                Ok(record) => match record.get("event") {
                    Some(event) => format!("{} {}", event.as_str().unwrap(), record["removed"]),
                    None => format!("{} {}", record["status"], record["cache"].as_str().unwrap()),
                },
                Err(_) => line.clone(),
            },
        )
        .collect();
    assert_eq!(
        told,
        [
            "200 miss",
            "200 hit",
            "304 hit",
            "403 miss",
            "400 miss",
            "404 miss",
            "200 hit",
            "purge 1",
            "200 miss",
            "shutdown: draining requests and background work for up to 8s",
            "shutdown: done",
        ]
    );
    assert!(
        queued[6].contains("\"request_id\":\"chosen-by-client\""),
        "{}",
        queued[6]
    );
}

#[test]
fn a_bad_log_sync_value_stops_the_boot() {
    let out = binary()
        .env("PORT", "0")
        .env("CRABCROP_LOG_SYNC", "sync")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.starts_with("crabcrop: invalid CRABCROP_LOG_SYNC"),
        "{stderr}"
    );
    assert!(out.stdout.is_empty());
}

#[test]
fn with_log_sync_a_request_writes_its_own_line_before_it_is_answered() {
    let mut server = Unread::boot(&[("CRABCROP_LOG_SYNC", "1")]);
    let unsigned = format!("/{}/40x30/http://127.0.0.1:9/a.png", "0".repeat(64));
    // While standard output takes lines, a request is
    // answered and its line is there.
    assert_eq!(
        status(server.port, &unsigned, "X-Request-Id: answered\r\n"),
        403
    );
    let mut line = String::new();
    server
        .stdout
        .as_mut()
        .unwrap()
        .read_line(&mut line)
        .unwrap();
    assert!(line.contains("\"request_id\":\"answered\""), "{line}");
    // Standard output is closed. The request's handler
    // writes the line itself, before the answer: the write
    // fails there, and the request is cut off with no
    // answer, as in releases through 0.12. (The writer
    // thread drops the line and the request is answered:
    // see the test of a closed standard output below.)
    server.close();
    let cut_off = get_within(server.port, &unsigned, "", LONG);
    assert!(
        cut_off.as_ref().is_none_or(|head| head.is_empty()),
        "{cut_off:?}"
    );
    server.stderr_says("failed printing to stdout");
    // The process is still there.
    assert_eq!(status(server.port, "/health", ""), 200);
}

#[test]
fn a_full_queue_makes_requests_wait_and_they_go_on_when_the_pipe_is_read() {
    // Lines of 60 KB into a pipe nobody reads: 8 MiB of
    // them are queued, about 140, and the request after
    // that is not answered.
    let mut server = Unread::boot(&[("CRABCROP_ALLOW_UNSAFE", "1")]);
    let (answered, waiting) = server.fill(200);
    assert!(
        (130..=150).contains(&answered),
        "{answered} lines of 60 KB were taken: the queue is full at 8 MiB"
    );
    // The server is not stuck: it answered a request that
    // logs nothing, the health check, 200 times while that
    // one waited.
    assert!(!waiting.is_finished());
    // The pipe is read: the request is answered, and every
    // line is there, whole, once, in the order of the
    // requests.
    let lines = server.read();
    assert_eq!(status_of(&waiting.join().unwrap().unwrap()), 403);
    server.terminate();
    assert_eq!(server.exit().0.code(), Some(0), "{}", server.stderr());
    let lines = rest_of(&lines);
    assert_eq!(lines.len(), answered + 3, "{} lines", lines.len());
    let numbers: Vec<usize> = lines[..=answered].iter().map(|l| number_in(l)).collect();
    assert_eq!(numbers, (0..=answered).collect::<Vec<_>>());
    assert_eq!(lines[answered + 2], "shutdown: done");
    assert!(
        !server.stderr().contains("request log"),
        "{}",
        server.stderr()
    );
}

#[test]
fn a_shutdown_ends_within_its_bound_when_nobody_reads_standard_output() {
    // The queue is full, a request waits for room in it,
    // and nobody reads the pipe, now or later.
    let mut server = Unread::boot(&[
        ("CRABCROP_ALLOW_UNSAFE", "1"),
        ("CRABCROP_SHUTDOWN_GRACE_SECS", "1"),
    ]);
    let (answered, waiting) = server.fill(200);
    server.terminate();
    let (status, took) = server.exit();
    // The drain gives the waiting request its grace, 1 s,
    // and the log's queue 1 s more: then the process is
    // gone, with nothing read.
    assert_eq!(status.code(), Some(0), "{}", server.stderr());
    assert!(took >= Duration::from_millis(1900), "exited after {took:?}");
    assert!(took < LONG / 2, "exited after {took:?}");
    let stderr = server.stderr();
    // The one request that waited, counted (issue #715).
    assert!(
        stderr.contains("shutdown: 1 requests still in flight at the deadline are cut off"),
        "{stderr}"
    );
    // Stderr says how many lines went unwritten: all but
    // the one or two the pipe took.
    let left: usize = stderr
        .lines()
        .find_map(|line| {
            line.strip_prefix(
                "shutdown: standard output did not take the log within 1s; the last ",
            )?
            .strip_suffix(" log lines are not written")?
            .parse()
            .ok()
        })
        .unwrap_or_else(|| panic!("no count of unwritten lines in {stderr}"));
    assert!(
        (answered..=answered + 3).contains(&left),
        "{left} of {answered} lines left"
    );
    // The request that waited was cut off at the deadline.
    let _ = waiting.join();
}

#[test]
fn a_closed_standard_output_costs_the_lines_and_nothing_else() {
    let mut server = Unread::boot(&[]);
    server.close();
    let unsigned = format!("/{}/40x30/http://127.0.0.1:9/a.png", "0".repeat(64));
    // Every request is answered, though no line can be
    // written.
    for _ in 0..40 {
        assert_eq!(status(server.port, &unsigned, ""), 403);
    }
    let stderr = server.stderr_says("request log: cannot write to standard output: Broken pipe");
    assert!(stderr.contains(" lines dropped"), "{stderr}");
    // At the first failure and then every ten seconds, not
    // once per line.
    assert!(stderr.matches("cannot write to standard output").count() < 10);
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert_eq!(status(server.port, "/health", ""), 200);
    // And the shutdown is clean, and prompt.
    server.terminate();
    let (status, took) = server.exit();
    assert_eq!(status.code(), Some(0), "{}", server.stderr());
    assert!(took < LONG / 2, "exited after {took:?}");
    let stderr = server.stderr();
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(!stderr.contains("did not take the log"), "{stderr}");
}

/// The writer thread of process `pid`: its ID, or `None`
/// if the process has none.
#[cfg(target_os = "linux")]
fn writer_thread(pid: u32) -> Option<String> {
    std::fs::read_dir(format!("/proc/{pid}/task"))
        .unwrap()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .find(|tid| {
            std::fs::read_to_string(format!("/proc/{pid}/task/{tid}/comm"))
                .is_ok_and(|name| name.trim() == "crabcrop-log")
        })
}

/// Wait until thread `tid` of `pid` sleeps.
#[cfg(target_os = "linux")]
fn asleep(pid: u32, tid: &str) {
    let deadline = Instant::now() + LONG;
    loop {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/task/{tid}/stat")).unwrap();
        // The state is the first field after the name.
        if stat[stat.rfind(')').unwrap() + 2..].starts_with('S') {
            return;
        }
        assert!(Instant::now() < deadline, "the thread never slept: {stat}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// How often thread `tid` of `pid` has been taken off a
/// CPU, by its own choice and not, and the CPU time it
/// has used, in clock ticks.
#[cfg(target_os = "linux")]
fn thread_use(pid: u32, tid: &str) -> (u64, u64) {
    let status = std::fs::read_to_string(format!("/proc/{pid}/task/{tid}/status")).unwrap();
    let switches: u64 = status
        .lines()
        .filter(|line| line.contains("ctxt_switches"))
        .map(|line| {
            line.rsplit('\t')
                .next()
                .unwrap()
                .trim()
                .parse::<u64>()
                .unwrap()
        })
        .sum();
    let stat = std::fs::read_to_string(format!("/proc/{pid}/task/{tid}/stat")).unwrap();
    // The fields after the name, which can hold spaces:
    // utime and stime are the 12th and 13th of them.
    let fields: Vec<&str> = stat[stat.rfind(')').unwrap() + 2..].split(' ').collect();
    let ticks = fields[11].parse::<u64>().unwrap() + fields[12].parse::<u64>().unwrap();
    (switches, ticks)
}

#[test]
#[cfg(target_os = "linux")]
fn the_writer_thread_has_a_name_and_does_nothing_at_rest() {
    let (server, port, lines) = common::serve(&mut binary());
    let pid = server.0.id();
    let unsigned = format!("/{}/40x30/http://127.0.0.1:9/a.png", "0".repeat(64));
    for _ in 0..5 {
        assert_eq!(status(port, &unsigned, ""), 403);
    }
    // Its lines are out: the thread has nothing to do.
    for _ in 0..5 {
        let line = lines.recv_timeout(LONG).unwrap();
        assert!(line.contains("\"status\":403"), "{line}");
    }
    let tid = writer_thread(pid).expect("a thread named crabcrop-log");
    asleep(pid, &tid);
    let before = thread_use(pid, &tid);
    std::thread::sleep(Duration::from_millis(1500));
    let after = thread_use(pid, &tid);
    // It slept the whole time: it was never scheduled, so
    // it used no CPU. A thread that woke every millisecond
    // to look at its queue would have run 1500 times.
    assert_eq!(after, before, "the writer thread ran with nothing to write");
    // A line still gets it up.
    assert_eq!(status(port, &unsigned, ""), 403);
    lines.recv_timeout(LONG).unwrap();
    asleep(pid, &tid);
    assert!(thread_use(pid, &tid).0 > before.0);

    // With `CRABCROP_LOG_SYNC` there is no such thread.
    let (server, port, lines) = common::serve(binary().env("CRABCROP_LOG_SYNC", "1"));
    assert_eq!(status(port, &unsigned, ""), 403);
    lines.recv_timeout(LONG).unwrap();
    assert_eq!(writer_thread(server.0.id()), None);
}

/// The `/metrics` token the stalled-log tests configure.
const METRICS_TOKEN: &str = "request-log-metrics-token-0123456789";

/// One HTTP/1.0 GET on `port`: the answer's status and body.
fn get_body(port: u16, path: &str, headers: &str) -> (u16, String) {
    try_get_body(port, path, headers).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// [`get_body`], or the error that kept the answer away.
fn try_get_body(port: u16, path: &str, headers: &str) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(LONG))?;
    write!(
        stream,
        "GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n{headers}\r\n"
    )?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    let response = String::from_utf8_lossy(&response).into_owned();
    let (head, body) = response
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("{path}: no head in {response:?}"));
    Ok((status_of(head), body.to_string()))
}

/// The exposition, scraped with [`METRICS_TOKEN`].
fn scrape(port: u16) -> String {
    let (status, body) = get_body(
        port,
        "/metrics",
        &format!("Authorization: Bearer {METRICS_TOKEN}\r\n"),
    );
    assert_eq!(status, 200, "{body}");
    body
}

/// The value of the metric `name` in `exposition`.
fn metric(exposition: &str, name: &str) -> f64 {
    exposition
        .lines()
        .find_map(|line| line.strip_prefix(name)?.strip_prefix(' ')?.parse().ok())
        .unwrap_or_else(|| panic!("no {name} in:\n{exposition}"))
}

/// How long the server keeps its queue full before its
/// health check fails: `STALLED_AFTER`.
const STALLED_AFTER: Duration = Duration::from_secs(5);

#[test]
fn a_standard_output_nobody_reads_fails_the_health_check_until_it_is_read() {
    // Issue #707: with standard output a pipe nobody reads,
    // the queue fills and every image request waits, and
    // the health check kept answering 200, so a liveness
    // probe never restarted a replica that served no
    // image. Now it answers 503 once the queue has stayed
    // full for 5 seconds, `/metrics` shows the queue, and
    // everything goes on once the pipe is read: the
    // request that waited is answered, no line is lost,
    // and the check passes again.
    let mut server = Unread::boot(&[
        ("CRABCROP_ALLOW_UNSAFE", "1"),
        ("CRABCROP_METRICS_TOKEN", METRICS_TOKEN),
    ]);
    let port = server.port;
    let before_filling = Instant::now();
    let (answered, waiting) = server.fill(200);
    let metrics = scrape(port);
    assert!(
        metric(&metrics, "crabcrop_request_log_queue_bytes") >= (8 * 1024 * 1024) as f64,
        "{metrics}"
    );
    assert_eq!(
        metric(&metrics, "crabcrop_request_log_queue_capacity_bytes"),
        (8 * 1024 * 1024) as f64
    );
    assert_eq!(
        metric(&metrics, "crabcrop_request_log_requests_waiting"),
        1.0
    );
    let scraped = Instant::now();
    // The health check is answered all along, 200 and
    // then 503 once the queue has been full for 5 seconds.
    let stalled = loop {
        let (health, body) = get_body(port, "/health", "Accept: application/json\r\n");
        if health == 503 {
            break body;
        }
        assert_eq!((health, body.as_str()), (200, "ok"));
        assert!(scraped.elapsed() < LONG, "the health check never failed");
        std::thread::sleep(Duration::from_millis(50));
    };
    let error: serde_json::Value = serde_json::from_str(&stalled).unwrap();
    assert_eq!(error["error"]["code"], "request_log_stalled");
    // No sooner than the bound after the queue filled,
    // which was after `before_filling`.
    assert!(
        before_filling.elapsed() >= STALLED_AFTER,
        "it failed {:?} after the queue began to fill",
        before_filling.elapsed()
    );
    let metrics = scrape(port);
    assert!(
        metric(&metrics, "crabcrop_request_log_queue_full_seconds_total")
            >= STALLED_AFTER.as_secs_f64(),
        "{metrics}"
    );
    assert_eq!(
        metric(&metrics, "crabcrop_request_log_requests_waiting"),
        1.0
    );
    // The images' HEALTHCHECK fails with it.
    let check = binary()
        .env("PORT", port.to_string())
        .arg("health")
        .output()
        .unwrap();
    assert_eq!(check.status.code(), Some(1), "{check:?}");
    let said = String::from_utf8_lossy(&check.stderr);
    assert!(
        said.contains("HTTP 503 Service Unavailable: request log stalled"),
        "{said}"
    );
    // The request still waits: it is not dropped.
    assert!(!waiting.is_finished());
    // The pipe is read: the request is answered, the queue
    // drains, and the check passes again. (The writer takes
    // the whole queue as one write when the reader comes
    // back, so the bytes it holds fall when that write
    // ends.)
    let lines = server.read();
    assert_eq!(status_of(&waiting.join().unwrap().unwrap()), 403);
    let drained = Instant::now();
    loop {
        let metrics = scrape(port);
        if metric(&metrics, "crabcrop_request_log_queue_bytes") < (8 * 1024 * 1024) as f64 {
            assert_eq!(
                metric(&metrics, "crabcrop_request_log_requests_waiting"),
                0.0
            );
            break;
        }
        assert!(drained.elapsed() < LONG, "the queue never drained");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(get_body(port, "/health", ""), (200, "ok".to_string()));
    // Requests are answered at once again.
    assert_eq!(status(port, &long_line_request(9999), ""), 403);
    server.terminate();
    assert_eq!(server.exit().0.code(), Some(0), "{}", server.stderr());
    // Every line is there, once: the requests that filled
    // the queue, the one that waited, the one after, and
    // the shutdown's two.
    let lines = rest_of(&lines);
    assert_eq!(lines.len(), answered + 4, "{} lines", lines.len());
    let numbers: Vec<usize> = lines[..answered + 2].iter().map(|l| number_in(l)).collect();
    let mut expected: Vec<usize> = (0..=answered).collect();
    expected.push(9999);
    assert_eq!(numbers, expected);
    assert_eq!(lines[answered + 3], "shutdown: done");
}

/// A TCP port on 127.0.0.1 that nothing listens on right now,
/// for a server whose listening line cannot reach the test.
/// The server's own sockets cannot be read from `/proc`
/// instead: it marks itself non-dumpable at startup (issue
/// #605), so `/proc/<pid>/fd` is readable by root alone, and a
/// test run as any other user would never find the port.
#[cfg(target_os = "linux")]
fn unused_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port on 127.0.0.1")
        .port()
}

#[test]
#[cfg(target_os = "linux")]
fn a_server_whose_standard_output_reader_left_during_boot_serves_on() {
    // Issue #707: the boot lines were written with
    // `println!`, which panics when a write fails, so a
    // reader of the pipe that had gone before boot was
    // done (a log collector that exited) ended the server
    // before it served. A boot line that fails is now lost,
    // with one line on stderr, and the server serves; the
    // request log counts the lines it cannot write, as it
    // does once the server serves. The reader here is gone
    // before the first line, which would have named the
    // server's port, so the test picks the port and waits
    // until the server accepts on it.
    let port = unused_port();
    let mut child = binary()
        .env("PORT", port.to_string())
        .env("CRABCROP_BIND", "127.0.0.1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawning crabcrop");
    drop(child.stdout.take());
    let mut pipe = child.stderr.take().unwrap();
    let stderr = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&stderr);
    std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        while let Ok(n @ 1..) = pipe.read(&mut chunk) {
            sink.lock().unwrap().extend_from_slice(&chunk[..n]);
        }
    });
    let said = || String::from_utf8_lossy(&stderr.lock().unwrap()).into_owned();
    let mut server = ServerGuard::new(child);
    let deadline = Instant::now() + LONG;
    while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
        if let Some(exit) = server.0.try_wait().unwrap() {
            panic!("crabcrop exited before it listened: {exit}\n{}", said());
        }
        assert!(
            Instant::now() < deadline,
            "crabcrop never listened on {port}\n{}",
            said()
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    // It answers: the health check, and an image request,
    // whose line cannot be written.
    let (health, _) = try_get_body(port, "/health", "")
        .unwrap_or_else(|e| panic!("no answer: {e}\nstderr: {}", said()));
    assert_eq!(health, 200, "{}", said());
    let unsigned = format!("/{}/40x30/http://127.0.0.1:9/a.png", "0".repeat(64));
    assert_eq!(get_body(port, &unsigned, "").0, 403);
    let deadline = Instant::now() + LONG;
    while !said().contains("request log: cannot write to standard output") {
        assert!(
            Instant::now() < deadline,
            "no request log failure in {}",
            said()
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let stderr_text = said();
    assert_eq!(
        stderr_text
            .matches("crabcrop: cannot write the boot lines to standard output: Broken pipe")
            .count(),
        1,
        "{stderr_text}"
    );
    assert!(!stderr_text.contains("panicked"), "{stderr_text}");
    // And it stops cleanly.
    rustix::process::kill_process(
        rustix::process::Pid::from_child(&server.0),
        rustix::process::Signal::TERM,
    )
    .unwrap();
    let waited = Instant::now();
    let exit = loop {
        if let Some(exit) = server.0.try_wait().unwrap() {
            break exit;
        }
        assert!(waited.elapsed() < LONG, "the server did not exit");
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(exit.code(), Some(0), "{}", said());
}

#[test]
#[cfg(target_os = "linux")]
fn a_server_whose_standard_error_reader_left_during_boot_serves_on() {
    // Issue #738: the server's lines on stderr were written
    // with `eprintln!`, which panics when a write fails, so a
    // reader of stderr that had gone (a log collector that
    // exited) ended the server at its first line there. Unsafe
    // mode with no allowlist writes a warning there during
    // boot, and a source that cannot be fetched writes a load
    // error once the server serves. Each line is lost now,
    // standard output says so once, and the server serves and
    // stops as it would.
    let mut child = binary()
        .env("PORT", "0")
        .env("CRABCROP_ALLOW_UNSAFE", "1")
        .env("CRABCROP_BLOCK_PRIVATE_SOURCES", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawning crabcrop");
    drop(child.stderr.take());
    let stdout = BufReader::new(child.stdout.take().unwrap());
    let (sender, lines) = mpsc::channel();
    std::thread::spawn(move || {
        for line in stdout.lines() {
            let Ok(line) = line else { return };
            if sender.send(line).is_err() {
                return;
            }
        }
    });
    let mut server = ServerGuard::new(child);
    let mut printed = Vec::new();
    let deadline = Instant::now() + LONG;
    let port = loop {
        match lines.recv_timeout(Duration::from_millis(50)) {
            Ok(line) => match line.strip_prefix("crabcrop listening on port ") {
                Some(port) => break port.parse::<u16>().unwrap(),
                None => printed.push(line),
            },
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                std::thread::sleep(Duration::from_millis(50))
            }
        }
        if let Some(exit) = server.0.try_wait().unwrap() {
            panic!("crabcrop exited before it listened: {exit}\n{printed:#?}");
        }
        assert!(
            Instant::now() < deadline,
            "crabcrop never listened\n{printed:#?}"
        );
    };
    // It answers: the health check, and two image requests
    // whose sources nothing serves (port 9 refuses the
    // connection), each of which writes its load error.
    assert_eq!(get_body(port, "/health", "").0, 200);
    for name in ["a", "b"] {
        let path = format!("/unsafe/40x30/http://127.0.0.1:9/{name}.png");
        assert_eq!(status(port, &path, ""), 502, "{path}");
    }
    // And it stops cleanly.
    rustix::process::kill_process(
        rustix::process::Pid::from_child(&server.0),
        rustix::process::Signal::TERM,
    )
    .unwrap();
    let waited = Instant::now();
    let exit = loop {
        if let Some(exit) = server.0.try_wait().unwrap() {
            break exit;
        }
        assert!(waited.elapsed() < LONG, "the server did not exit");
        std::thread::sleep(Duration::from_millis(10));
    };
    printed.extend(rest_of(&lines));
    assert_eq!(exit.code(), Some(0), "{printed:#?}");
    // Standard output said once that stderr takes no lines,
    // at the first one, the boot's warning.
    let notes: Vec<&String> = printed
        .iter()
        .filter(|line| line.starts_with("crabcrop: cannot write to standard error"))
        .collect();
    assert_eq!(notes.len(), 1, "{printed:#?}");
    assert!(notes[0].contains("Broken pipe"), "{}", notes[0]);
    assert_eq!(printed.last().map(String::as_str), Some("shutdown: done"));
}

/// Not a test on its own: the process that
/// `a_line_stderr_refuses_lets_go_of_stderr_before_its_note_waits`
/// starts, by running this test binary again with a stderr
/// whose reader is gone. It exits 0 once both of its threads
/// are through.
#[test]
#[ignore = "a helper the stderr lock test runs as its child"]
#[cfg(target_os = "linux")]
fn stderr_note_helper() {
    if std::env::var_os("CRABCROP_TEST_STDERR_NOTE").is_none() {
        return;
    }
    let pid = std::process::id();
    // This thread holds standard output, as a failed boot
    // line does while it says so on stderr.
    let out = std::io::stdout().lock();
    let writer = std::thread::Builder::new()
        .name("stderr-line".to_string())
        .spawn(|| crabcrop::observe::print_stderr_line(format_args!("a line stderr refuses")))
        .unwrap();
    let deadline = Instant::now() + LONG;
    let tid = loop {
        let found = std::fs::read_dir(format!("/proc/{pid}/task"))
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .find(|tid| {
                std::fs::read_to_string(format!("/proc/{pid}/task/{tid}/comm"))
                    .is_ok_and(|name| name.trim() == "stderr-line")
            });
        if let Some(tid) = found {
            break tid;
        }
        assert!(Instant::now() < deadline, "the writer thread never started");
        std::thread::sleep(Duration::from_millis(5));
    };
    // Its write to stderr fails at once, so the one place it
    // can sleep is the wait for standard output, to say so.
    asleep(pid, &tid);
    // Standard error is free while it waits: this is where a
    // writer that held it would deadlock the two threads.
    drop(std::io::stderr().lock());
    drop(out);
    writer.join().unwrap();
    std::process::exit(0);
}

#[test]
#[cfg(target_os = "linux")]
fn a_line_stderr_refuses_lets_go_of_stderr_before_its_note_waits() {
    // Issue #738: a stderr line that stderr refuses is noted
    // on standard output, which another thread may hold: a
    // boot line that failed holds it while it says so on
    // stderr, and the request log's writer holds it through
    // every write. The stderr line lets go of stderr first,
    // so neither can deadlock with it. Its child holds
    // standard output while a stderr line fails, and takes
    // stderr once that line waits for standard output.
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "stderr_note_helper",
            "--ignored",
            "--test-threads=1",
            "--quiet",
        ])
        .env("CRABCROP_TEST_STDERR_NOTE", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stderr.take());
    let mut stdout = child.stdout.take().unwrap();
    let printed = std::thread::spawn(move || {
        let mut printed = String::new();
        let _ = stdout.read_to_string(&mut printed);
        printed
    });
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if started.elapsed() > LONG {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "the two threads deadlocked: the stderr line held stderr while it waited for \
                 standard output\n{}",
                printed.join().unwrap()
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let printed = printed.join().unwrap();
    assert!(status.success(), "{status}\n{printed}");
    // And once standard output was free, the note reached it.
    assert_eq!(
        printed
            .matches("crabcrop: cannot write to standard error: Broken pipe")
            .count(),
        1,
        "{printed}"
    );
}
