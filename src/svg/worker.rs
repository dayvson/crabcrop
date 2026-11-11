//! SVG parsing and rasterization in a capped child process
//! (issue #447).
//!
//! The pre-parse bound ([`super::screen`]) refuses the
//! reference-expansion shapes it models in milliseconds, but
//! some documents get past any model drawn from a document's
//! structure: a 130-byte arc whose radii make kurbo flatten
//! it into billions of segments, or whatever shape the next
//! usvg release adds. In-process, one such document aborts
//! the server and every request in flight. So the server
//! never runs usvg itself: it re-executes its own binary
//! with [`WORKER_ARG`], hands the sanitized document over
//! on stdin, and reads the intrinsic size and the RGBA
//! raster back on stdout. The child caps its own address
//! space and CPU time with `setrlimit` as it starts (on
//! itself, so the CPU cap holds on macOS too; macOS accepts
//! the address-space cap but does not enforce it), parses
//! and renders on a thread with a bounded stack, and is
//! killed with its process group at the wall-clock
//! deadline. An abort, a stack overflow, a spin, or an
//! allocation storm ends the child alone (where the cap is
//! not enforced, an allocation storm runs until the
//! deadline); the request answers with the
//! documented refusal, and every other request carries on.
//!
//! One child serves one document. A pool of long-lived
//! workers would save the spawn, but a worker that served
//! one request would carry its CPU time, heap, and any
//! state a hostile document left into the next one, and the
//! spawn is a small part of a render (`docs/url-grammar.md`
//! has the measurements).
//!
//! The exchange, integers little-endian:
//!
//! 1. The parent sends [`MAGIC`], the crate version (a
//!    length byte, then the text), and the sanitized
//!    document (a `u64` length, then the bytes).
//! 2. The child answers one frame: a tag byte (`0` ok, `1`
//!    refused), a `u64` length, and the payload. Ok carries
//!    the intrinsic size (two `f32`); refused carries the
//!    reason.
//! 3. The parent sends the raster size it picked from the
//!    intrinsic size (two `u32`), or closes stdin to stop.
//! 4. The child answers one frame, the raster (exactly
//!    `w * h * 4` bytes) or the refusal, and exits.

use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::OnceLock;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The argument that starts the binary as an SVG worker
/// instead of a server or a CLI verb. Not a user-facing
/// flag: the server passes it to itself.
pub const WORKER_ARG: &str = "--svg-worker";

/// The first bytes of every request, so a program that is
/// not a worker of this protocol fails fast and plainly.
const MAGIC: &[u8; 8] = b"crabsvg\x01";

/// Parent and worker must be the same build: a worker that
/// renders with a different usvg would serve different
/// bytes under the same cache key. On Linux the parent
/// starts `/proc/self/exe`, its own image even after an
/// upgrade replaced the file on disk; elsewhere this check
/// refuses a mismatch instead.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The worker's address-space cap. A legitimate render
/// needs at most a few hundred megabytes: the 4096 px
/// raster, a filter's intermediate rasters, the 256 MiB
/// decode budget for embedded images, and the tree. The
/// cap leaves room for that and the binary's own mappings,
/// and turns the arc path's multi-gigabyte allocation into
/// a failed allocation in the child.
pub(crate) const WORKER_MAX_ADDRESS_SPACE: u64 = 4 << 30;

/// The wall clock for isolated work whose caller has no
/// deadline of its own (upload validation): the server's
/// default processing timeout.
pub(crate) const WORKER_TIMEOUT: Duration = Duration::from_secs(30);

/// The stack of the thread that parses and renders. The
/// pre-parse bound already caps reference depth for a 2 MiB
/// stack; this is headroom, and an overflow past it ends
/// the worker, not the server.
const WORKER_STACK: usize = 8 << 20;

/// The largest document the worker reads. The parent only
/// sends sanitized sources, which the source fetch and
/// upload caps keep far below this.
const MAX_DOCUMENT: u64 = 64 << 20;

/// The longest refusal either side carries.
const MAX_MESSAGE: usize = 4096;

/// The worker's stderr kept for the diagnosis of a crash.
const STDERR_CAP: usize = 4096;

/// The largest raster side the worker renders: the engine's
/// decode dimension cap (the engine itself asks for at most
/// 4096 px a side).
const MAX_RASTER_SIDE: u32 = crate::engine::MAX_DECODE_DIMENSION;

/// The environment the worker keeps: only what the system
/// font lookup reads (fontdb and fontconfig), so text renders
/// with the fonts the server would use. Everything else, the
/// server's secrets and credentials included, is cleared, so
/// the worker is not handed them. That alone does not keep
/// them from a compromised worker, which runs as the server's
/// user: the server's non-dumpable mark (issue #605) keeps it
/// out of the server's `/proc` files, but not out of the
/// files the server can read. See SECURITY.md.
const FONT_ENV: &[&str] = &[
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_DATA_DIRS",
    "XDG_CACHE_HOME",
    "FONTCONFIG_FILE",
    "FONTCONFIG_PATH",
    "SYSTEMROOT",
    "USERPROFILE",
];

const TAG_OK: u8 = 0;
const TAG_REFUSED: u8 = 1;

/// The job byte after the version: an SVG parse and render, or
/// an AVIF decode (issue #522). One worker serves one job.
const JOB_SVG: u8 = 0;
const JOB_AVIF: u8 = 1;

/// The worker a failure names, by job, so an AVIF decode that
/// fails is not reported as an SVG render.
fn worker_name(job: u8) -> &'static str {
    if job == JOB_AVIF {
        "the AVIF decoder"
    } else {
        "the SVG renderer"
    }
}

/// The work a timeout names, by job.
fn work_name(job: u8) -> &'static str {
    if job == JOB_AVIF {
        "AVIF decoding"
    } else {
        "SVG rendering"
    }
}

/// How many bytes the worker's answer may carry: an exact
/// length the parent already knows (the SVG size frame, the
/// SVG raster), or any length up to a cap (the AVIF pixels,
/// bounded by the decode budget plus the small header).
#[derive(Clone, Copy)]
enum Want {
    Exact(usize),
    UpTo(usize),
}

/// How to start a worker: a program and the arguments that
/// put it in worker mode. The worker's own limits follow as
/// two more arguments, the address-space cap in bytes and
/// the CPU limit in seconds.
#[derive(Debug, Clone)]
pub struct WorkerCommand {
    pub program: PathBuf,
    pub args: Vec<OsString>,
}

impl WorkerCommand {
    /// The running binary in worker mode. On Linux that is
    /// `/proc/self/exe`, which resolves in the spawned child
    /// to the image the server runs, even after an upgrade
    /// replaced or removed the file on disk.
    pub fn current_binary() -> io::Result<Self> {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let program = {
            let own = PathBuf::from("/proc/self/exe");
            if own.exists() {
                own
            } else {
                std::env::current_exe()?
            }
        };
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        let program = std::env::current_exe()?;
        Ok(Self {
            program,
            args: vec![WORKER_ARG.into()],
        })
    }
}

static WORKER: OnceLock<WorkerCommand> = OnceLock::new();

/// Parse and render every SVG through workers started from
/// `command` (issue #447). The binary calls this once at
/// startup, before it serves; a library user that never
/// calls it (the unit tests, the fuzz harness) parses SVG
/// in-process. Later calls are ignored.
pub fn isolate_in_child(command: WorkerCommand) {
    let _ = WORKER.set(command);
}

/// The configured worker, when SVG work is isolated.
pub(super) fn configured() -> Option<&'static WorkerCommand> {
    WORKER.get()
}

/// Why isolated SVG work produced no result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderError {
    /// The document was refused (by the parse or one of the
    /// bounds), or the worker crashed, misbehaved, or could
    /// not start; the message says which.
    Refused(String),
    /// The wall-clock deadline passed and the worker was
    /// killed.
    TimedOut(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::Refused(why) | RenderError::TimedOut(why) => f.write_str(why),
        }
    }
}

/// The worker's intrinsic size, and its raster when the
/// caller asked for one.
pub(super) type Output = ((f32, f32), Option<image::RgbaImage>);

/// A spawned worker, killed with its process group and
/// reaped when dropped, so no early return leaves one
/// running.
struct Worker {
    child: Child,
    /// Set once the worker is reaped. From then on its
    /// process group is never signalled: an empty group's
    /// ID can be reused by an unrelated process.
    reaped: Option<Option<ExitStatus>>,
}

impl Worker {
    /// Kill the group and collect the exit status.
    fn finish(&mut self) -> Option<ExitStatus> {
        if self.reaped.is_none() {
            crate::toolrun::kill_process_group(&mut self.child);
            self.reaped = Some(self.child.wait().ok());
        }
        self.reaped.flatten()
    }

    /// The status of a worker whose stdout ended: it is
    /// exiting, so wait for that (bounded by `until`) before
    /// killing it, which keeps its own status and its last
    /// words on stderr intact for the diagnosis.
    fn exit_status(&mut self, until: Instant) -> Option<ExitStatus> {
        while self.reaped.is_none() {
            match self.child.try_wait() {
                Ok(Some(status)) => self.reaped = Some(Some(status)),
                Ok(None) if Instant::now() < until => std::thread::sleep(Duration::from_millis(2)),
                _ => return self.finish(),
            }
        }
        self.reaped.flatten()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.finish();
    }
}

/// One answer from the worker.
enum Frame {
    Ok(Vec<u8>),
    Refused(String),
}

/// Read one frame whose ok payload matches `want`: an exact
/// length, or any length up to a cap. Anything else is a
/// protocol violation; an end of stream before a whole frame
/// is `UnexpectedEof`.
fn read_frame(input: &mut impl Read, want: Want) -> io::Result<Frame> {
    let mut head = [0u8; 9];
    input.read_exact(&mut head)?;
    let len = u64::from_le_bytes(head[1..].try_into().expect("eight bytes"));
    let malformed = |why: String| io::Error::new(io::ErrorKind::InvalidData, why);
    let cap = match head[0] {
        TAG_OK => match want {
            Want::Exact(n) if len == n as u64 => n,
            Want::Exact(n) => {
                return Err(malformed(format!(
                    "a {len} byte answer where {n} were expected"
                )));
            }
            Want::UpTo(cap) if len <= cap as u64 => len as usize,
            Want::UpTo(cap) => {
                return Err(malformed(format!(
                    "a {len} byte answer past the {cap} byte cap"
                )));
            }
        },
        TAG_REFUSED if len <= MAX_MESSAGE as u64 => len as usize,
        TAG_REFUSED => return Err(malformed(format!("a {len} byte refusal"))),
        tag => return Err(malformed(format!("frame tag {tag}"))),
    };
    let mut payload = Vec::with_capacity(cap);
    input.take(cap as u64).read_to_end(&mut payload)?;
    if payload.len() != cap {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(match head[0] {
        TAG_OK => Frame::Ok(payload),
        _ => Frame::Refused(String::from_utf8_lossy(&payload).into_owned()),
    })
}

/// The parent's side of the exchange: the frame reader runs
/// on its own thread, so every wait honors the deadline.
struct Session {
    want: mpsc::Sender<Want>,
    frames: mpsc::Receiver<io::Result<Frame>>,
    stderr: mpsc::Receiver<String>,
    deadline: Instant,
    timeout: Duration,
    /// The job byte, for the words a failure uses.
    job: u8,
}

/// Why the session ended without a frame.
enum Ended {
    TimedOut,
    /// The stream ended or broke the protocol.
    Broken(io::Error),
}

impl Session {
    fn next(&self, want: Want) -> Result<Frame, Ended> {
        if self.want.send(want).is_err() {
            return Err(Ended::Broken(io::ErrorKind::UnexpectedEof.into()));
        }
        let wait = self.deadline.saturating_duration_since(Instant::now());
        match self.frames.recv_timeout(wait) {
            Ok(Ok(frame)) => Ok(frame),
            Ok(Err(e)) => Err(Ended::Broken(e)),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(Ended::TimedOut),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(Ended::Broken(io::ErrorKind::UnexpectedEof.into()))
            }
        }
    }

    /// Turn a session that ended without its frame into the
    /// caller's error, killing and reaping the worker.
    fn failure(&self, worker: &mut Worker, ended: Ended) -> RenderError {
        match ended {
            Ended::TimedOut => {
                worker.finish();
                RenderError::TimedOut(format!(
                    "{} ran past its {:.1}s deadline",
                    work_name(self.job),
                    self.timeout.as_secs_f64()
                ))
            }
            Ended::Broken(e) => {
                let until = Instant::now() + Duration::from_secs(2);
                let status = worker.exit_status(until.min(self.deadline));
                let stderr = self
                    .stderr
                    .recv_timeout(Duration::from_millis(200))
                    .unwrap_or_default();
                RenderError::Refused(diagnose(&e, status, &stderr, self.job))
            }
        }
    }
}

/// Name what happened to a worker whose answer never came:
/// its own last words on stderr say whether an allocation
/// failed or the stack overflowed.
fn diagnose(error: &io::Error, status: Option<ExitStatus>, stderr: &str, job: u8) -> String {
    let status = status.map_or_else(|| "no exit status".to_string(), |s| s.to_string());
    let worker = worker_name(job);
    if error.kind() != io::ErrorKind::UnexpectedEof {
        return format!("{worker} sent a malformed answer ({error}; {status})");
    }
    let what = if stderr.contains("memory allocation of") {
        "ran out of memory"
    } else if stderr.contains("overflowed its stack") {
        "overflowed its stack"
    } else {
        "crashed"
    };
    format!("{worker} {what} ({status})")
}

/// Build the request every job starts with: the magic, the
/// crate version, and the job byte.
fn request_head(job: u8) -> Vec<u8> {
    let mut request = Vec::with_capacity(MAGIC.len() + 2 + VERSION.len());
    request.extend_from_slice(MAGIC);
    request.push(VERSION.len() as u8);
    request.extend_from_slice(VERSION.as_bytes());
    request.push(job);
    request
}

/// Start a worker from `command` with the address-space and
/// CPU caps, its pipes open and its own process group, so the
/// whole group can be killed at the deadline. `timeout` sizes
/// the CPU cap, the backstop for a worker the parent stops
/// watching; the wall-clock kill is the primary bound.
fn spawn(command: &WorkerCommand, timeout: Duration, job: u8) -> Result<Worker, RenderError> {
    let cpu_limit = crate::toolrun::cpu_limit_for(timeout, 1);
    let mut cmd = Command::new(&command.program);
    cmd.args(&command.args)
        .arg(WORKER_MAX_ADDRESS_SPACE.to_string())
        .arg(cpu_limit.as_secs().max(1).to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // No `current_dir`: the worker reads no relative path,
        // and a chdir into a missing directory (a TMPDIR that
        // does not exist) would fail every job at spawn.
        .env_clear();
    for name in FONT_ENV {
        if let Some(value) = std::env::var_os(name) {
            cmd.env(name, value);
        }
    }
    // A test-only knob the AVIF job reads to stall past a short
    // deadline, forwarded through the cleared environment so an
    // end-to-end deadline test needs no special source file.
    if let Some(value) = std::env::var_os(AVIF_DELAY_ENV) {
        cmd.env(AVIF_DELAY_ENV, value);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let child = cmd
        .spawn()
        .map_err(|e| RenderError::Refused(format!("could not start {}: {e}", worker_name(job))))?;
    Ok(Worker {
        child,
        reaped: None,
    })
}

/// Hand `worker`'s pipes to reader threads and send `request`
/// on a writer thread (a large request can outgrow the pipe),
/// which returns stdin for a second write when the job needs
/// one. The reader threads are never joined: each ends when
/// the worker's pipes close, which the kill guarantees.
fn exchange(
    worker: &mut Worker,
    request: Vec<u8>,
    deadline: Instant,
    timeout: Duration,
    job: u8,
) -> (Session, mpsc::Receiver<std::process::ChildStdin>) {
    let mut stdin = worker.child.stdin.take().expect("stdin was piped");
    let (stdin_tx, stdin_rx) = mpsc::channel();
    std::thread::spawn(move || {
        if stdin
            .write_all(&request)
            .and_then(|()| stdin.flush())
            .is_ok()
        {
            let _ = stdin_tx.send(stdin);
        }
    });

    let mut stdout = worker.child.stdout.take().expect("stdout was piped");
    let (want_tx, want_rx) = mpsc::channel::<Want>();
    let (frame_tx, frame_rx) = mpsc::channel();
    std::thread::spawn(move || {
        for want in want_rx {
            let frame = read_frame(&mut stdout, want);
            let failed = frame.is_err();
            if frame_tx.send(frame).is_err() || failed {
                break;
            }
        }
    });

    let mut stderr = worker.child.stderr.take().expect("stderr was piped");
    let (err_tx, err_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut kept = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            match stderr.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let room = STDERR_CAP.saturating_sub(kept.len());
                    kept.extend_from_slice(&chunk[..n.min(room)]);
                }
            }
        }
        let _ = err_tx.send(String::from_utf8_lossy(&kept).into_owned());
    });

    let session = Session {
        want: want_tx,
        frames: frame_rx,
        stderr: err_rx,
        deadline,
        timeout,
        job,
    };
    (session, stdin_rx)
}

/// Decode `bytes` (an AVIF source the container screen has
/// already passed) in a worker started from `command`, under
/// `max_dimension` and `max_alloc`, killed at `deadline`. The
/// AV1 decode runs here under the worker's caps (issue #522);
/// one still running at the deadline is killed and the parent
/// answers with the timeout.
pub(super) fn run_avif(
    command: &WorkerCommand,
    bytes: &[u8],
    max_dimension: u32,
    max_alloc: u64,
    deadline: Instant,
) -> Result<image::DynamicImage, RenderError> {
    let timeout = deadline.saturating_duration_since(Instant::now());
    if timeout.is_zero() {
        return Err(RenderError::TimedOut(
            "AVIF decoding was past its deadline before it started".to_string(),
        ));
    }
    let mut worker = spawn(command, timeout, JOB_AVIF)?;
    let mut request = request_head(JOB_AVIF);
    request.extend_from_slice(&max_dimension.to_le_bytes());
    request.extend_from_slice(&max_alloc.to_le_bytes());
    request.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    request.extend_from_slice(bytes);
    // The answer carries a one-byte color tag, two u32 sizes,
    // and the pixels, which cannot exceed the decode budget.
    let cap = (max_alloc as usize).saturating_add(AVIF_HEADER);
    let (session, _stdin) = exchange(&mut worker, request, deadline, timeout, JOB_AVIF);
    match session.next(Want::UpTo(cap)) {
        Ok(Frame::Ok(payload)) => decode_avif_payload(payload).ok_or_else(|| {
            RenderError::Refused("the AVIF decoder sent a malformed image".to_string())
        }),
        Ok(Frame::Refused(why)) => Err(RenderError::Refused(why)),
        Err(ended) => Err(session.failure(&mut worker, ended)),
    }
}

/// The AVIF answer's fixed header: a color tag and two u32
/// dimensions before the pixels.
const AVIF_HEADER: usize = 1 + 4 + 4;

/// A test-only environment variable the AVIF job reads for a
/// millisecond stall, so a deadline test needs no crafted file
/// and no special build. `CRABCROP_TEST_*` names stay out of
/// the configuration docs.
const AVIF_DELAY_ENV: &str = "CRABCROP_TEST_AVIF_WORKER_DELAY_MS";

/// Rebuild the decoded image from the worker's answer: a color
/// tag, the width and height, then exactly the pixels the tag
/// and size imply. `None` on any length or tag mismatch. The
/// answer is taken by value and its header dropped in place, so
/// an 8-bit image costs no second copy of its pixels.
fn decode_avif_payload(mut payload: Vec<u8>) -> Option<image::DynamicImage> {
    let header: [u8; AVIF_HEADER] = payload.get(..AVIF_HEADER)?.try_into().ok()?;
    let color = header[0];
    let width = u32::from_le_bytes(header[1..5].try_into().ok()?);
    let height = u32::from_le_bytes(header[5..9].try_into().ok()?);
    let count = (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(avif_channels(color)? as usize)?;
    if payload.len() - AVIF_HEADER != count.checked_mul(avif_depth(color)?)? {
        return None;
    }
    payload.drain(..AVIF_HEADER);
    build_dynamic_image(color, width, height, payload)
}

/// Parse `document` (sanitized) in a worker started from
/// `command`, then render it at the size `target` picks from
/// the intrinsic size, or stop at the size when `target`
/// picks none. The worker is killed at `deadline`.
pub(super) fn run(
    command: &WorkerCommand,
    document: &str,
    deadline: Instant,
    target: impl FnOnce((f32, f32)) -> Option<(u32, u32)>,
) -> Result<Output, RenderError> {
    let timeout = deadline.saturating_duration_since(Instant::now());
    if timeout.is_zero() {
        return Err(RenderError::TimedOut(
            "SVG rendering was past its deadline before it started".to_string(),
        ));
    }
    let mut worker = spawn(command, timeout, JOB_SVG)?;
    let mut request = request_head(JOB_SVG);
    request.extend_from_slice(&(document.len() as u64).to_le_bytes());
    request.extend_from_slice(document.as_bytes());
    let (session, stdin_rx) = exchange(&mut worker, request, deadline, timeout, JOB_SVG);
    let size = match session.next(Want::Exact(8)) {
        Ok(Frame::Ok(bytes)) => (
            f32::from_le_bytes(bytes[..4].try_into().expect("four bytes")),
            f32::from_le_bytes(bytes[4..].try_into().expect("four bytes")),
        ),
        Ok(Frame::Refused(why)) => return Err(RenderError::Refused(why)),
        Err(ended) => return Err(session.failure(&mut worker, ended)),
    };
    let Some((w, h)) = target(size) else {
        // Closing stdin (dropping the channel drops it) ends
        // the worker after the size; the guard reaps it.
        return Ok((size, None));
    };
    // 0x0 would read as "stop" to the worker; refuse it the
    // way an in-process render does.
    if w == 0 || h == 0 {
        return Err(RenderError::Refused(
            "SVG raster size out of range".to_string(),
        ));
    }
    let wait = deadline.saturating_duration_since(Instant::now());
    let Ok(mut stdin) = stdin_rx.recv_timeout(wait) else {
        // The request never finished going out: the worker
        // stopped reading, or the deadline passed first.
        return Err(match session.next(Want::Exact(0)) {
            Ok(_) => RenderError::Refused("the SVG renderer answered out of turn".to_string()),
            Err(ended) => session.failure(&mut worker, ended),
        });
    };
    let mut raster_size = [0u8; 8];
    raster_size[..4].copy_from_slice(&w.to_le_bytes());
    raster_size[4..].copy_from_slice(&h.to_le_bytes());
    // A worker that died after the size shows up as the end
    // of its answer below, so a failed write is not an error
    // of its own.
    let _ = stdin.write_all(&raster_size).and_then(|()| stdin.flush());
    drop(stdin);
    let raster_bytes = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| RenderError::Refused("SVG raster size out of range".to_string()))?;
    match session.next(Want::Exact(raster_bytes)) {
        Ok(Frame::Ok(raster)) => {
            let raster = image::RgbaImage::from_raw(w, h, raster).ok_or_else(|| {
                RenderError::Refused("SVG raster buffer size mismatch".to_string())
            })?;
            Ok((size, Some(raster)))
        }
        Ok(Frame::Refused(why)) => Err(RenderError::Refused(why)),
        Err(ended) => Err(session.failure(&mut worker, ended)),
    }
}

/// Whether this process was started as an SVG worker. The
/// binary checks this first thing, before it starts a
/// runtime or reads any configuration.
pub fn is_worker_invocation() -> bool {
    std::env::args_os()
        .nth(1)
        .is_some_and(|arg| arg == WORKER_ARG)
}

/// The worker process: cap itself, serve one exchange on
/// stdin and stdout, and return the exit code. Nothing here
/// reads configuration or opens a socket.
pub fn worker_main() -> i32 {
    let mut args = std::env::args().skip(2);
    let mut limit = |default: u64| {
        args.next()
            .and_then(|arg| arg.parse().ok())
            .unwrap_or(default)
    };
    let max_address_space = limit(WORKER_MAX_ADDRESS_SPACE);
    let cpu_secs = limit(crate::toolrun::cpu_limit_for(WORKER_TIMEOUT, 1).as_secs());
    crate::toolrun::limit_self(Duration::from_secs(cpu_secs), max_address_space);
    // Should the host run short of memory anyway, the
    // kernel picks the worker over the server.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let _ = std::fs::write("/proc/self/oom_score_adj", "1000");
    let worker = std::thread::Builder::new()
        .name("svg-worker".to_string())
        .stack_size(WORKER_STACK)
        .spawn(|| {
            let mut input = io::BufReader::new(io::stdin().lock());
            let mut output = io::BufWriter::new(io::stdout().lock());
            serve(&mut input, &mut output)
        });
    match worker.map(std::thread::JoinHandle::join) {
        Ok(Ok(Ok(()))) => 0,
        // The exchange with the parent failed.
        Ok(Ok(Err(_))) => 2,
        // The parse or the render panicked.
        Ok(Err(_)) => 3,
        Err(_) => 4,
    }
}

/// Read the magic and version that open every request:
/// `Ok(Err(..))` is a request the worker refuses (and says so),
/// an I/O error one it cannot read.
fn read_header(input: &mut impl Read) -> io::Result<Result<(), String>> {
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Ok(Err("not an SVG worker request".to_string()));
    }
    let mut len = [0u8; 1];
    input.read_exact(&mut len)?;
    let mut version = vec![0u8; usize::from(len[0])];
    input.read_exact(&mut version)?;
    if version != VERSION.as_bytes() {
        return Ok(Err(format!(
            "the SVG renderer is version {VERSION}, the server {}",
            String::from_utf8_lossy(&version)
        )));
    }
    Ok(Ok(()))
}

/// Read the SVG document that follows the header for an SVG
/// job (a `u64` length, then the bytes).
fn read_document(input: &mut impl Read) -> io::Result<Result<String, String>> {
    let mut len = [0u8; 8];
    input.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    if len > MAX_DOCUMENT {
        return Ok(Err(format!(
            "the SVG document is larger than {MAX_DOCUMENT} bytes"
        )));
    }
    let mut document = Vec::with_capacity(len as usize);
    input.take(len).read_to_end(&mut document)?;
    if document.len() as u64 != len {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(String::from_utf8(document).map_err(|_| "SVG source is not valid UTF-8".to_string()))
}

fn send(output: &mut impl Write, tag: u8, payload: &[u8]) -> io::Result<()> {
    output.write_all(&[tag])?;
    output.write_all(&(payload.len() as u64).to_le_bytes())?;
    output.write_all(payload)?;
    output.flush()
}

fn refuse(output: &mut impl Write, why: &str) -> io::Result<()> {
    let mut end = why.len().min(MAX_MESSAGE);
    while !why.is_char_boundary(end) {
        end -= 1;
    }
    send(output, TAG_REFUSED, &why.as_bytes()[..end])
}

/// The worker's side of one exchange: read the header and the
/// job byte, then serve that job (see the module docs).
pub(super) fn serve(input: &mut impl Read, output: &mut impl Write) -> io::Result<()> {
    if let Err(why) = read_header(input)? {
        return refuse(output, &why);
    }
    let mut job = [0u8; 1];
    input.read_exact(&mut job)?;
    match job[0] {
        JOB_SVG => serve_svg(input, output),
        JOB_AVIF => serve_avif(input, output),
        other => refuse(output, &format!("unknown worker job {other}")),
    }
}

/// Decode one AVIF source: read the decode limits and the
/// bytes, decode on the image crate's bounded path (the same
/// one the in-process decode runs), and answer with the color
/// tag, the dimensions, and the pixels, or a refusal.
fn serve_avif(input: &mut impl Read, output: &mut impl Write) -> io::Result<()> {
    let mut limits = [0u8; 12];
    input.read_exact(&mut limits)?;
    let max_dimension = u32::from_le_bytes(limits[..4].try_into().expect("four bytes"));
    let max_alloc = u64::from_le_bytes(limits[4..].try_into().expect("eight bytes"));
    let mut len = [0u8; 8];
    input.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    // The parent only sends sources the fetch and upload caps
    // admitted, far below this; past it, say so rather than
    // leave the parent to diagnose a crash.
    let most = max_alloc.saturating_add(MAX_DOCUMENT);
    if len > most {
        return refuse(
            output,
            &format!("the AVIF source is larger than {most} bytes"),
        );
    }
    let mut bytes = Vec::with_capacity(len.min(1 << 20) as usize);
    input.take(len).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != len {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    // A test hook that stalls the job past a short deadline, so
    // the parent's kill can be exercised end to end.
    if let Ok(ms) = std::env::var(AVIF_DELAY_ENV)
        && let Ok(ms) = ms.parse::<u64>()
    {
        std::thread::sleep(Duration::from_millis(ms));
    }
    match decode_avif_bounded(&bytes, max_dimension, max_alloc) {
        Ok(image) => match encode_avif(&image) {
            Some((color, pixels)) => {
                let (w, h) = (image.width(), image.height());
                output.write_all(&[TAG_OK])?;
                output.write_all(&((AVIF_HEADER + pixels.len()) as u64).to_le_bytes())?;
                output.write_all(&[color])?;
                output.write_all(&w.to_le_bytes())?;
                output.write_all(&h.to_le_bytes())?;
                output.write_all(pixels)?;
                output.flush()
            }
            None => refuse(output, "the AVIF decoded to an unsupported color type"),
        },
        Err(why) => refuse(output, &why),
    }
}

/// The image crate's bounded still decode for an AVIF source,
/// the same decoder and limits the in-process path applies.
fn decode_avif_bounded(
    bytes: &[u8],
    max_dimension: u32,
    max_alloc: u64,
) -> Result<image::DynamicImage, String> {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(max_dimension);
    limits.max_image_height = Some(max_dimension);
    limits.max_alloc = Some(max_alloc);
    let mut reader = image::ImageReader::new(io::Cursor::new(bytes));
    reader.limits(limits);
    reader
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| e.to_string())
}

/// The color tag and raw pixels of a decoded image, for the
/// AVIF answer. `None` for a color type the AVIF decoder never
/// produces (it yields 8- or 16-bit RGBA), so the parent and
/// the worker stay a closed set.
fn encode_avif(image: &image::DynamicImage) -> Option<(u8, &[u8])> {
    use image::DynamicImage::*;
    let tag = match image {
        ImageLuma8(_) => 0,
        ImageLumaA8(_) => 1,
        ImageRgb8(_) => 2,
        ImageRgba8(_) => 3,
        ImageLuma16(_) => 4,
        ImageLumaA16(_) => 5,
        ImageRgb16(_) => 6,
        ImageRgba16(_) => 7,
        _ => return None,
    };
    Some((tag, image.as_bytes()))
}

/// The channel count of a color tag, or `None` for an unknown
/// one.
fn avif_channels(tag: u8) -> Option<u32> {
    match tag {
        0 | 4 => Some(1),
        1 | 5 => Some(2),
        2 | 6 => Some(3),
        3 | 7 => Some(4),
        _ => None,
    }
}

/// The bytes per channel of a color tag (1 for 8-bit, 2 for
/// 16-bit), or `None` for an unknown one.
fn avif_depth(tag: u8) -> Option<usize> {
    match tag {
        0..=3 => Some(1),
        4..=7 => Some(2),
        _ => None,
    }
}

/// Rebuild the decoded image from a color tag and its raw
/// pixels. `None` when the buffer does not fit the dimensions.
fn build_dynamic_image(
    tag: u8,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
) -> Option<image::DynamicImage> {
    use image::{DynamicImage, ImageBuffer};
    fn u16s(bytes: Vec<u8>) -> Vec<u16> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u16::from_ne_bytes(*p))
            .collect()
    }
    Some(match tag {
        0 => DynamicImage::ImageLuma8(ImageBuffer::from_raw(width, height, pixels)?),
        1 => DynamicImage::ImageLumaA8(ImageBuffer::from_raw(width, height, pixels)?),
        2 => DynamicImage::ImageRgb8(ImageBuffer::from_raw(width, height, pixels)?),
        3 => DynamicImage::ImageRgba8(ImageBuffer::from_raw(width, height, pixels)?),
        4 => DynamicImage::ImageLuma16(ImageBuffer::from_raw(width, height, u16s(pixels))?),
        5 => DynamicImage::ImageLumaA16(ImageBuffer::from_raw(width, height, u16s(pixels))?),
        6 => DynamicImage::ImageRgb16(ImageBuffer::from_raw(width, height, u16s(pixels))?),
        7 => DynamicImage::ImageRgba16(ImageBuffer::from_raw(width, height, u16s(pixels))?),
        _ => return None,
    })
}

/// The worker's side of an SVG exchange (see the module docs).
fn serve_svg(input: &mut impl Read, output: &mut impl Write) -> io::Result<()> {
    let document = match read_document(input)? {
        Ok(document) => document,
        Err(why) => return refuse(output, &why),
    };
    let prepared = match super::build(&document) {
        Ok(prepared) => prepared,
        Err(why) => return refuse(output, &why),
    };
    drop(document);
    let (w, h) = prepared.intrinsic();
    let mut size = [0u8; 8];
    size[..4].copy_from_slice(&w.to_le_bytes());
    size[4..].copy_from_slice(&h.to_le_bytes());
    send(output, TAG_OK, &size)?;
    let mut raster_size = [0u8; 8];
    match input.read_exact(&mut raster_size) {
        Ok(()) => {}
        // The parent wanted the size alone.
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
        Err(e) => return Err(e),
    }
    let w = u32::from_le_bytes(raster_size[..4].try_into().expect("four bytes"));
    let h = u32::from_le_bytes(raster_size[4..].try_into().expect("four bytes"));
    if w == 0 || h == 0 {
        return Ok(());
    }
    if w > MAX_RASTER_SIDE || h > MAX_RASTER_SIDE {
        return refuse(output, "SVG raster size out of range");
    }
    match prepared.render(w, h) {
        Ok(raster) => send(output, TAG_OK, raster.into_rgba8().as_raw()),
        Err(why) => refuse(output, &why),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50" viewBox="0 0 100 50"><rect width="100" height="50" fill="#00ff00"/><circle cx="50" cy="25" r="20" fill="#ff0000"/></svg>"##;

    fn request(magic: &[u8], version: &str, document: &str) -> Vec<u8> {
        let mut bytes = magic.to_vec();
        bytes.push(version.len() as u8);
        bytes.extend_from_slice(version.as_bytes());
        bytes.push(JOB_SVG);
        bytes.extend_from_slice(&(document.len() as u64).to_le_bytes());
        bytes.extend_from_slice(document.as_bytes());
        bytes
    }

    fn frames(mut output: &[u8], wants: &[usize]) -> Vec<Frame> {
        wants
            .iter()
            .map(|want| read_frame(&mut output, Want::Exact(*want)).expect("a whole frame"))
            .collect()
    }

    fn sanitized(svg: &str) -> String {
        String::from_utf8(super::super::screen(svg.as_bytes()).unwrap().into_bytes()).unwrap()
    }

    #[test]
    fn the_exchange_renders_what_the_process_renders() {
        // The worker's side, driven in memory: the size
        // frame, then exactly the raster an in-process
        // render produces, byte for byte.
        let clean = sanitized(SIMPLE);
        let mut input = request(MAGIC, VERSION, &clean);
        input.extend_from_slice(&40u32.to_le_bytes());
        input.extend_from_slice(&20u32.to_le_bytes());
        let mut output = Vec::new();
        serve(&mut input.as_slice(), &mut output).unwrap();
        let answers = frames(&output, &[8, 40 * 20 * 4]);
        let Frame::Ok(size) = &answers[0] else {
            panic!("the size was refused")
        };
        assert_eq!(f32::from_le_bytes(size[..4].try_into().unwrap()), 100.0);
        assert_eq!(f32::from_le_bytes(size[4..].try_into().unwrap()), 50.0);
        let Frame::Ok(raster) = &answers[1] else {
            panic!("the raster was refused")
        };
        let local = super::super::prepare(SIMPLE.as_bytes())
            .unwrap()
            .render(40, 20)
            .unwrap()
            .into_rgba8();
        assert_eq!(raster, local.as_raw());
    }

    #[test]
    fn the_exchange_stops_at_the_size_when_stdin_closes() {
        let mut output = Vec::new();
        let input = request(MAGIC, VERSION, &sanitized(SIMPLE));
        serve(&mut input.as_slice(), &mut output).unwrap();
        assert_eq!(output.len(), 9 + 8, "one size frame and nothing else");
        // A 0x0 size stops it the same way.
        let mut input = input;
        input.extend_from_slice(&[0; 8]);
        let mut output = Vec::new();
        serve(&mut input.as_slice(), &mut output).unwrap();
        assert_eq!(output.len(), 9 + 8);
    }

    #[test]
    fn the_worker_refuses_what_it_cannot_serve() {
        let refusal = |input: Vec<u8>| {
            let mut output = Vec::new();
            serve(&mut input.as_slice(), &mut output).unwrap();
            match frames(&output, &[8]).remove(0) {
                Frame::Refused(why) => why,
                Frame::Ok(_) => panic!("served"),
            }
        };
        let why = refusal(request(b"notsvg!\x01", VERSION, SIMPLE));
        assert!(why.contains("not an SVG worker request"), "{why}");
        let why = refusal(request(MAGIC, "0.0.0-other", SIMPLE));
        assert!(why.contains("0.0.0-other"), "{why}");
        // The expansion bound holds in the worker as it does
        // in-process: its parse is the one `prepare` runs
        // after the sanitizer (which the parent runs).
        let cycle = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1\" height=\"1\"><defs>\
             <mask id=\"m1\"><rect width=\"1\" height=\"1\" mask=\"url(#m2)\"/></mask>\
             <mask id=\"m2\"><rect width=\"1\" height=\"1\" mask=\"url(#m1)\"/></mask></defs>\
             <rect width=\"1\" height=\"1\" mask=\"url(#m1)\"/></svg>";
        let why = refusal(request(MAGIC, VERSION, cycle));
        assert!(why.contains("cycle"), "{why}");
        // A raster size past the decode dimension cap.
        let mut input = request(MAGIC, VERSION, &sanitized(SIMPLE));
        input.extend_from_slice(&(MAX_RASTER_SIDE + 1).to_le_bytes());
        input.extend_from_slice(&1u32.to_le_bytes());
        let mut output = Vec::new();
        serve(&mut input.as_slice(), &mut output).unwrap();
        assert!(
            matches!(&frames(&output, &[8, 0])[1], Frame::Refused(why) if why.contains("out of range"))
        );
    }

    #[test]
    fn frames_out_of_protocol_are_errors() {
        // An ok frame of the wrong length, an oversized
        // refusal, an unknown tag, and a truncated payload.
        let frame = |tag: u8, len: u64, payload: &[u8]| {
            let mut bytes = vec![tag];
            bytes.extend_from_slice(&len.to_le_bytes());
            bytes.extend_from_slice(payload);
            bytes
        };
        let error = |bytes: Vec<u8>, want: usize| match read_frame(
            &mut bytes.as_slice(),
            Want::Exact(want),
        ) {
            Err(e) => e.kind(),
            Ok(_) => panic!("accepted"),
        };
        assert_eq!(
            error(frame(TAG_OK, 9, &[0; 9]), 8),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            error(frame(TAG_REFUSED, MAX_MESSAGE as u64 + 1, b""), 8),
            io::ErrorKind::InvalidData
        );
        assert_eq!(error(frame(7, 0, b""), 8), io::ErrorKind::InvalidData);
        assert_eq!(
            error(frame(TAG_OK, 8, &[0; 3]), 8),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(error(Vec::new(), 8), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn a_dead_worker_is_diagnosed_from_its_last_words() {
        let eof = io::Error::from(io::ErrorKind::UnexpectedEof);
        let oom = diagnose(
            &eof,
            None,
            "memory allocation of 3758096384 bytes failed\n",
            JOB_SVG,
        );
        assert!(oom.contains("ran out of memory"), "{oom}");
        let stack = diagnose(
            &eof,
            None,
            "\nthread 'svg-worker' has overflowed its stack\nfatal runtime error: stack overflow",
            JOB_SVG,
        );
        assert!(stack.contains("overflowed its stack"), "{stack}");
        assert!(diagnose(&eof, None, "", JOB_SVG).contains("crashed"));
        let bad = io::Error::new(io::ErrorKind::InvalidData, "frame tag 7");
        assert!(diagnose(&bad, None, "", JOB_SVG).contains("malformed answer"));
        // An AVIF job's failure names the AVIF decoder.
        assert_eq!(
            diagnose(&eof, None, "", JOB_AVIF),
            "the AVIF decoder crashed (no exit status)"
        );
    }

    /// A stand-in worker: a shell script, so a test can make
    /// the child crash, hang, or misbehave on cue. (The
    /// limits `run` appends become the script's `$0` and `$1`.)
    #[cfg(unix)]
    fn shell_worker(script: &str) -> WorkerCommand {
        WorkerCommand {
            program: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), script.into()],
        }
    }

    /// The size frame for a 1x1 document, as `printf` octal
    /// escapes: tag 0, length 8, then 1.0 and 1.0 as `f32`.
    #[cfg(unix)]
    const SIZE_FRAME: &str =
        r"\000\010\000\000\000\000\000\000\000\000\000\200\077\000\000\200\077";

    #[cfg(unix)]
    fn run_shell(script: &str, timeout: Duration) -> (Result<Output, RenderError>, Duration) {
        let started = Instant::now();
        let result = run(&shell_worker(script), SIMPLE, started + timeout, |_| {
            Some((1, 1))
        });
        (result, started.elapsed())
    }

    #[cfg(unix)]
    #[test]
    fn a_crashing_worker_is_a_refusal() {
        // Before it answers, after the size, and with the
        // last words an allocation failure leaves: each is a
        // refusal naming what happened, well before the
        // deadline.
        let (result, took) = run_shell("kill -ABRT $$", Duration::from_secs(20));
        let Err(RenderError::Refused(why)) = result else {
            panic!("{result:?}")
        };
        assert!(why.contains("crashed"), "{why}");
        assert!(took < Duration::from_secs(10), "{took:?}");
        let (result, _) = run_shell(
            &format!(
                "printf '{SIZE_FRAME}'; echo 'memory allocation of 99 bytes failed' >&2; kill -ABRT $$"
            ),
            Duration::from_secs(20),
        );
        let Err(RenderError::Refused(why)) = result else {
            panic!("{result:?}")
        };
        assert!(why.contains("ran out of memory"), "{why}");
        // A program that exits without answering at all.
        let (result, _) = run_shell("exit 0", Duration::from_secs(20));
        assert!(matches!(result, Err(RenderError::Refused(_))), "{result:?}");
    }

    #[cfg(unix)]
    #[test]
    fn a_hung_worker_is_killed_at_the_deadline() {
        // Hung before the size, and hung after it, mid-render:
        // both end at the deadline as a timeout, and the
        // worker's process group is gone afterwards.
        let dir = crate::test_support::scratch_dir("svg-worker-test");
        let pid_file = dir.join("worker.pid");
        let pid_path = pid_file.display().to_string();
        for script in [
            format!("echo $$ > '{pid_path}'; exec sleep 30"),
            format!("echo $$ > '{pid_path}'; printf '{SIZE_FRAME}'; exec sleep 30"),
        ] {
            let _ = std::fs::remove_file(&pid_file);
            let (result, took) = run_shell(&script, Duration::from_millis(500));
            assert!(
                matches!(&result, Err(RenderError::TimedOut(_))),
                "{script}: {result:?}"
            );
            assert!(took >= Duration::from_millis(500), "{took:?}");
            assert!(took < Duration::from_secs(10), "{took:?}");
            let pid: i32 = std::fs::read_to_string(&pid_file)
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            let pid = rustix::process::Pid::from_raw(pid).unwrap();
            assert!(
                rustix::process::test_kill_process(pid).is_err(),
                "the worker outlived its deadline"
            );
        }
        let _ = std::fs::remove_file(&pid_file);
    }

    #[cfg(unix)]
    #[test]
    fn a_worker_that_breaks_the_protocol_is_refused() {
        // A size frame of the wrong length: the answer is
        // refused as malformed, not trusted.
        let (result, _) = run_shell(
            r"printf '\000\011\000\000\000\000\000\000\000'; exec sleep 30",
            Duration::from_secs(20),
        );
        let Err(RenderError::Refused(why)) = result else {
            panic!("{result:?}")
        };
        assert!(why.contains("malformed"), "{why}");
        // A zero-sized raster is refused before it is asked
        // for, as an in-process render refuses it.
        let result = run(
            &shell_worker(&format!("printf '{SIZE_FRAME}'; exec sleep 30")),
            SIMPLE,
            Instant::now() + Duration::from_secs(20),
            |_| Some((0, 1)),
        );
        assert!(
            matches!(&result, Err(RenderError::Refused(why)) if why.contains("out of range")),
            "{result:?}"
        );
        // A program that cannot be started at all.
        let missing = WorkerCommand {
            program: PathBuf::from("/nonexistent/crabcrop"),
            args: Vec::new(),
        };
        let result = run(
            &missing,
            SIMPLE,
            Instant::now() + Duration::from_secs(5),
            |_| None,
        );
        assert!(
            matches!(&result, Err(RenderError::Refused(why)) if why.contains("could not start")),
            "{result:?}"
        );
    }

    #[cfg(feature = "avif-decode")]
    fn avif_request(bytes: &[u8], max_dimension: u32, max_alloc: u64) -> Vec<u8> {
        let mut request = request_head(JOB_AVIF);
        request.extend_from_slice(&max_dimension.to_le_bytes());
        request.extend_from_slice(&max_alloc.to_le_bytes());
        request.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        request.extend_from_slice(bytes);
        request
    }

    #[test]
    fn the_avif_payload_round_trips_every_color_type() {
        use image::DynamicImage;
        let images = [
            DynamicImage::ImageLuma8(image::GrayImage::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap()),
            DynamicImage::ImageLumaA8(
                image::ImageBuffer::from_raw(1, 2, vec![1, 2, 3, 4]).unwrap(),
            ),
            DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(1, 2, vec![1, 2, 3, 4, 5, 6]).unwrap(),
            ),
            DynamicImage::ImageRgba8(image::RgbaImage::from_raw(1, 1, vec![9, 8, 7, 6]).unwrap()),
            DynamicImage::ImageLuma16(
                image::ImageBuffer::from_raw(2, 1, vec![40000u16, 7]).unwrap(),
            ),
            DynamicImage::ImageLumaA16(
                image::ImageBuffer::from_raw(1, 1, vec![40000u16, 9]).unwrap(),
            ),
            DynamicImage::ImageRgb16(
                image::ImageBuffer::from_raw(1, 1, vec![1u16, 2, 65535]).unwrap(),
            ),
            DynamicImage::ImageRgba16(
                image::ImageBuffer::from_raw(1, 1, vec![300u16, 1, 2, 65000]).unwrap(),
            ),
        ];
        let mut tags = Vec::new();
        for image in images {
            let (color, pixels) = encode_avif(&image).expect("a supported color type");
            tags.push(color);
            let mut payload = vec![color];
            payload.extend_from_slice(&image.width().to_le_bytes());
            payload.extend_from_slice(&image.height().to_le_bytes());
            payload.extend_from_slice(pixels);
            let rebuilt = decode_avif_payload(payload.clone()).expect("a whole image");
            assert_eq!(rebuilt.color(), image.color());
            assert_eq!(rebuilt.as_bytes(), image.as_bytes());
            // A payload a byte short of the pixels, or a byte
            // over, is refused, and so is one that names more
            // pixels than it carries.
            assert!(decode_avif_payload(payload[..payload.len() - 1].to_vec()).is_none());
            let mut over = payload.clone();
            over.push(0);
            assert!(decode_avif_payload(over).is_none());
            let mut taller = payload.clone();
            taller[5..9].copy_from_slice(&(image.height() + 1).to_le_bytes());
            assert!(decode_avif_payload(taller).is_none());
            // A size whose pixel count overflows is refused, not
            // wrapped.
            let mut vast = payload;
            vast[1..9].copy_from_slice(&[0xFF; 8]);
            assert!(decode_avif_payload(vast).is_none());
        }
        assert_eq!(tags, (0..8).collect::<Vec<u8>>(), "a closed set of tags");
        // An unknown tag, and an answer shorter than its header.
        assert!(decode_avif_payload(vec![8, 1, 0, 0, 0, 1, 0, 0, 0, 0]).is_none());
        assert!(decode_avif_payload(vec![3, 0, 0]).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_hung_avif_worker_is_killed_at_the_deadline() {
        // A worker that never answers its AVIF job is killed at
        // the deadline, its process group gone, and the parent
        // reports a timeout that names the AVIF decode, well
        // before the CPU cap.
        let dir = crate::test_support::scratch_dir("avif-worker-test");
        let pid_file = dir.join("worker.pid");
        let script = format!("echo $$ > '{}'; exec sleep 30", pid_file.display());
        let started = Instant::now();
        let result = run_avif(
            &shell_worker(&script),
            b"\x00\x00\x00\x18ftypavif",
            crate::engine::MAX_DECODE_DIMENSION,
            crate::engine::MAX_DECODE_ALLOC,
            started + Duration::from_millis(500),
        );
        let took = started.elapsed();
        assert!(
            matches!(&result, Err(RenderError::TimedOut(why)) if why.starts_with("AVIF decoding")),
            "{result:?}"
        );
        assert!(took >= Duration::from_millis(500), "{took:?}");
        assert!(took < Duration::from_secs(10), "{took:?}");
        let pid: i32 = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let pid = rustix::process::Pid::from_raw(pid).unwrap();
        assert!(
            rustix::process::test_kill_process(pid).is_err(),
            "the worker outlived its deadline"
        );
        // A worker that crashes, and one that cannot start, are
        // refusals that name the AVIF decoder, not a timeout.
        let result = run_avif(
            &shell_worker("kill -ABRT $$"),
            b"",
            1,
            1,
            Instant::now() + Duration::from_secs(20),
        );
        assert!(
            matches!(&result, Err(RenderError::Refused(why)) if why.starts_with("the AVIF decoder crashed")),
            "{result:?}"
        );
        let missing = WorkerCommand {
            program: PathBuf::from("/nonexistent/crabcrop"),
            args: Vec::new(),
        };
        let result = run_avif(&missing, b"", 1, 1, Instant::now() + Duration::from_secs(5));
        assert!(
            matches!(&result, Err(RenderError::Refused(why)) if why.starts_with("could not start the AVIF decoder")),
            "{result:?}"
        );
        // An answer past the decode budget is refused unread.
        let result = run_avif(
            &shell_worker(r"printf '\000\377\377\377\377\000\000\000\000'; exec sleep 30"),
            b"",
            1,
            1 << 20,
            Instant::now() + Duration::from_secs(20),
        );
        assert!(
            matches!(&result, Err(RenderError::Refused(why)) if why.contains("past the")),
            "{result:?}"
        );
    }

    #[cfg(feature = "avif-decode")]
    #[test]
    fn the_avif_exchange_decodes_what_the_process_decodes() {
        use image::ImageEncoder;
        let img = image::RgbImage::from_fn(20, 12, |x, y| {
            image::Rgb([(x * 11) as u8, (y * 17) as u8, ((x + y) * 7) as u8])
        });
        let mut avif = Vec::new();
        image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut avif, 10, 90)
            .write_image(img.as_raw(), 20, 12, image::ExtendedColorType::Rgb8)
            .unwrap();

        let input = avif_request(
            &avif,
            crate::engine::MAX_DECODE_DIMENSION,
            crate::engine::MAX_DECODE_ALLOC,
        );
        let mut output = Vec::new();
        serve(&mut input.as_slice(), &mut output).unwrap();
        let Frame::Ok(payload) =
            read_frame(&mut output.as_slice(), Want::UpTo(output.len())).expect("a whole frame")
        else {
            panic!("the AVIF was refused");
        };
        let decoded = decode_avif_payload(payload).expect("a decoded image");
        let in_process = crate::engine::decode_image(&avif).unwrap();
        assert_eq!(
            (decoded.width(), decoded.height()),
            (in_process.width(), in_process.height())
        );
        assert_eq!(decoded.color(), in_process.color());
        assert_eq!(decoded.as_bytes(), in_process.as_bytes());
    }
}
