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
/// server's secrets and credentials included, is cleared.
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

/// Read one frame whose ok payload must be exactly `want`
/// bytes. Anything else is a protocol violation; an end of
/// stream before a whole frame is `UnexpectedEof`.
fn read_frame(input: &mut impl Read, want: usize) -> io::Result<Frame> {
    let mut head = [0u8; 9];
    input.read_exact(&mut head)?;
    let len = u64::from_le_bytes(head[1..].try_into().expect("eight bytes"));
    let malformed = |why: String| io::Error::new(io::ErrorKind::InvalidData, why);
    let cap = match head[0] {
        TAG_OK if len == want as u64 => want,
        TAG_OK => {
            return Err(malformed(format!(
                "a {len} byte answer where {want} were expected"
            )));
        }
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
    want: mpsc::Sender<usize>,
    frames: mpsc::Receiver<io::Result<Frame>>,
    stderr: mpsc::Receiver<String>,
    deadline: Instant,
    timeout: Duration,
}

/// Why the session ended without a frame.
enum Ended {
    TimedOut,
    /// The stream ended or broke the protocol.
    Broken(io::Error),
}

impl Session {
    fn next(&self, want: usize) -> Result<Frame, Ended> {
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
                    "SVG rendering ran past its {:.1}s deadline",
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
                RenderError::Refused(diagnose(&e, status, &stderr))
            }
        }
    }
}

/// Name what happened to a worker whose answer never came:
/// its own last words on stderr say whether an allocation
/// failed or the stack overflowed.
fn diagnose(error: &io::Error, status: Option<ExitStatus>, stderr: &str) -> String {
    let status = status.map_or_else(|| "no exit status".to_string(), |s| s.to_string());
    if error.kind() != io::ErrorKind::UnexpectedEof {
        return format!("the SVG renderer sent a malformed answer ({error}; {status})");
    }
    let what = if stderr.contains("memory allocation of") {
        "ran out of memory"
    } else if stderr.contains("overflowed its stack") {
        "overflowed its stack"
    } else {
        "crashed"
    };
    format!("the SVG renderer {what} ({status})")
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
    // The worker parses and renders on one thread, so its
    // CPU time tracks the wall clock; the limit is the
    // backstop for a worker the parent no longer watches.
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
        // does not exist) would fail every SVG at spawn.
        .env_clear();
    for name in FONT_ENV {
        if let Some(value) = std::env::var_os(name) {
            cmd.env(name, value);
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let child = cmd
        .spawn()
        .map_err(|e| RenderError::Refused(format!("could not start the SVG renderer: {e}")))?;
    let mut worker = Worker {
        child,
        reaped: None,
    };

    // The request goes out on its own thread (a large
    // document can outgrow the pipe), which hands stdin
    // back for the raster size.
    let mut stdin = worker.child.stdin.take().expect("stdin was piped");
    let mut request = Vec::with_capacity(MAGIC.len() + 1 + VERSION.len() + 8 + document.len());
    request.extend_from_slice(MAGIC);
    request.push(VERSION.len() as u8);
    request.extend_from_slice(VERSION.as_bytes());
    request.extend_from_slice(&(document.len() as u64).to_le_bytes());
    request.extend_from_slice(document.as_bytes());
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
    let (want_tx, want_rx) = mpsc::channel::<usize>();
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

    // The reader threads are never joined: each ends when
    // the worker's pipes close, which the kill guarantees.
    let session = Session {
        want: want_tx,
        frames: frame_rx,
        stderr: err_rx,
        deadline,
        timeout,
    };
    let size = match session.next(8) {
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
        return Err(match session.next(0) {
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
    match session.next(raster_bytes) {
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

/// Read the request: `Ok(Err(..))` is a request the worker
/// refuses (and says so), an I/O error one it cannot read.
fn read_request(input: &mut impl Read) -> io::Result<Result<String, String>> {
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

/// The worker's side of one exchange (see the module docs).
pub(super) fn serve(input: &mut impl Read, output: &mut impl Write) -> io::Result<()> {
    let document = match read_request(input)? {
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
        bytes.extend_from_slice(&(document.len() as u64).to_le_bytes());
        bytes.extend_from_slice(document.as_bytes());
        bytes
    }

    fn frames(mut output: &[u8], wants: &[usize]) -> Vec<Frame> {
        wants
            .iter()
            .map(|want| read_frame(&mut output, *want).expect("a whole frame"))
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
        let error = |bytes: Vec<u8>, want: usize| match read_frame(&mut bytes.as_slice(), want) {
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
        let oom = diagnose(&eof, None, "memory allocation of 3758096384 bytes failed\n");
        assert!(oom.contains("ran out of memory"), "{oom}");
        let stack = diagnose(
            &eof,
            None,
            "\nthread 'svg-worker' has overflowed its stack\nfatal runtime error: stack overflow",
        );
        assert!(stack.contains("overflowed its stack"), "{stack}");
        assert!(diagnose(&eof, None, "").contains("crashed"));
        let bad = io::Error::new(io::ErrorKind::InvalidData, "frame tag 7");
        assert!(diagnose(&bad, None, "").contains("malformed answer"));
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
}
