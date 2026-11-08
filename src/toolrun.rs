//! One runner for the external tools (issue #481): the
//! frame extractor (frames.rs), the PDF tools (pdf.rs), and
//! the video encoder (video.rs) start their
//! operator-configured programs through `run_tool`, which
//! holds the whole discipline in one place: no shell, an
//! empty environment, the temporary directory as the
//! working directory, the program's own process group
//! (killed whole at the deadline or the output cap), a
//! capped stdout, a drained stderr, and the kernel limits
//! below. A tool module keeps only its argv, its input, its
//! caps, and how a `ToolError` reads in its own error
//! type.
//!
//! The resource bounds (issue #362): the wall-clock
//! timeouts kill the process group, but the children also
//! run under kernel resource limits, so a tool that
//! allocates without bound or spins on CPU meets a hard
//! stop in the kernel as well: address space is capped,
//! and CPU time is capped just past the most the
//! wall-clock budget allows on this host's cores (the wall
//! clock stays the precise mechanism; the CPU limit is the
//! backstop).
//!
//! The limits are applied with `prlimit` right after the
//! spawn: the crate forbids unsafe code, so the
//! `pre_exec` route is closed. A child therefore starts
//! un-limited for the microseconds its loader needs
//! before the parent sets the limits; it cannot do
//! meaningful work of its own in that window, and the
//! wall-clock group kill covers the whole lifetime
//! regardless. `prlimit` is a Linux syscall, so the
//! kernel limits exist only there; on other platforms
//! the wall clock and the group kill are the bounds, as
//! they were before this module.
//!
//! A child that is this binary (the SVG worker, issue
//! #447) caps itself instead, with `limit_self` as it
//! starts: `setrlimit` on oneself has no startup window
//! and works on macOS as well.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// What a tool reads on its stdin.
pub(crate) enum ToolInput<'a> {
    /// A file handed over as the tool's stdin, read from
    /// wherever its offset stands: a regular file, which the
    /// tool can seek (the frame extractor's staged source,
    /// issue #411).
    File(File),
    /// Bytes fed through a pipe from a writer thread, so a
    /// tool that reads slowly, or not at all, never blocks
    /// the caller.
    Bytes(&'a [u8]),
    /// A feeder that writes the input itself, on the writer
    /// thread, for input too large to build up front (the
    /// video encoder's frames, each repeated for its
    /// scheduled count). Stdin closes when it returns. A
    /// failed write means the tool stopped reading, and its
    /// exit status and output say why, so the feeder's error
    /// is dropped.
    Writer(Feed),
}

/// The feeder behind [`ToolInput::Writer`].
pub(crate) type Feed = Box<dyn FnOnce(&mut ChildStdin) -> std::io::Result<()> + Send>;

/// One run of an external tool: the program, its argv, its
/// input, and the bounds the run obeys.
pub(crate) struct ToolCommand<'a> {
    /// The program, started directly, never through a shell.
    pub(crate) program: &'a Path,
    /// The argv, built in full by the calling module.
    pub(crate) args: &'a [&'a str],
    /// What the tool reads on stdin.
    pub(crate) input: ToolInput<'a>,
    /// The most stdout the run accepts: one byte more kills
    /// the process group and fails the run.
    pub(crate) max_stdout: usize,
    /// How much of stderr the run keeps for the caller to
    /// read. The rest is drained, so the tool never blocks
    /// on a full pipe, and dropped.
    pub(crate) max_stderr: usize,
    /// When the process group is killed and the run fails.
    /// A caller that stacks runs (a probe, then the work)
    /// passes all of them one deadline (issue #362).
    pub(crate) deadline: Instant,
    /// The kernel's CPU-time backstop for the child:
    /// [`tool_cpu_limit`] of the tool's timeout.
    pub(crate) cpu_limit: Duration,
}

/// A run that finished inside its deadline and its stdout
/// cap, whatever its exit status.
#[derive(Debug)]
pub(crate) struct ToolRun {
    pub(crate) status: ExitStatus,
    pub(crate) stdout: Vec<u8>,
    /// The head of stderr, up to `max_stderr` bytes, as
    /// lossy UTF-8.
    pub(crate) stderr: String,
    /// Whether stderr passed `max_stderr` and was cut.
    pub(crate) stderr_truncated: bool,
    /// The end of the whole stderr stream, for
    /// [`ToolRun::stderr_excerpt`].
    stderr_tail: String,
}

/// How many characters of stderr an error message quotes.
const STDERR_EXCERPT_CHARS: usize = 500;

/// How many bytes of stderr's end a run keeps for its
/// excerpt, whatever `max_stderr` keeps of the head: room
/// for the excerpt at four bytes a character, plus a
/// character the cut splits.
const STDERR_TAIL_BYTES: usize = 4 * STDERR_EXCERPT_CHARS + 4;

impl ToolRun {
    /// The end of the tool's stderr, for error text: the
    /// last 500 characters of the whole stream, where a tool
    /// reports its failure, trimmed.
    pub(crate) fn stderr_excerpt(&self) -> String {
        excerpt(&self.stderr_tail)
    }
}

/// The last [`STDERR_EXCERPT_CHARS`] characters of `text`,
/// trimmed, or "no error output" when nothing is left.
fn excerpt(text: &str) -> String {
    let start = text
        .char_indices()
        .rev()
        .nth(STDERR_EXCERPT_CHARS - 1)
        .map_or(0, |(at, _)| at);
    let tail = text[start..].trim();
    if tail.is_empty() {
        "no error output".to_string()
    } else {
        tail.to_string()
    }
}

/// What the stderr drain hands back.
#[derive(Default)]
struct Stderr {
    head: String,
    truncated: bool,
    tail: String,
}

/// Why a run produced no [`ToolRun`]. Each tool module maps
/// these to its own error type, in its own words.
#[derive(Debug)]
pub(crate) enum ToolError {
    /// The program could not be started.
    Spawn(std::io::Error),
    /// Stdout passed the cap, carried here; the process
    /// group was killed.
    OutputCap(usize),
    /// Reading stdout failed; the process group was killed.
    Read(std::io::Error),
    /// The deadline passed; the process group was killed.
    TimedOut,
    /// Waiting on the program failed.
    Wait(std::io::Error),
}

/// Run a tool once under the standing bounds: the input on
/// stdin, stdout capped at `max_stdout` (past the cap the
/// process group is killed and the run fails), stderr kept
/// up to `max_stderr` while the rest is drained so the tool
/// never blocks on a full pipe, an empty environment, the
/// temporary directory as the working directory, its own
/// process group, the kernel limits, and the deadline.
pub(crate) fn run_tool(command: ToolCommand<'_>) -> Result<ToolRun, ToolError> {
    run_tool_in(command, &std::env::temp_dir())
}

/// [`run_tool`] with the working directory named: the
/// temporary directory, or, in tests, a directory that
/// does not exist.
fn run_tool_in(command: ToolCommand<'_>, work_dir: &Path) -> Result<ToolRun, ToolError> {
    let ToolCommand {
        program,
        args,
        input,
        max_stdout,
        max_stderr,
        deadline,
        cpu_limit,
    } = command;
    let (stdin, feed): (Stdio, Option<Feed>) = match input {
        ToolInput::File(file) => (Stdio::from(file), None),
        ToolInput::Bytes(bytes) => {
            let bytes = bytes.to_vec();
            let feed = move |stdin: &mut ChildStdin| stdin.write_all(&bytes);
            (Stdio::piped(), Some(Box::new(feed)))
        }
        ToolInput::Writer(feed) => (Stdio::piped(), Some(feed)),
    };
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(stdin)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear();
    // The temporary directory as the working directory gives
    // a tool that writes under a relative path a scratch
    // place, not the server's own directory. It is hygiene,
    // not a bound: when the directory is missing (a TMPDIR
    // naming no directory), the tool runs where the server
    // runs instead of failing every run at spawn with an
    // error that reads like a missing program.
    if work_dir.is_dir() {
        cmd.current_dir(work_dir);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd.spawn().map_err(ToolError::Spawn)?;
    limit_child(child.id(), cpu_limit, TOOL_MAX_ADDRESS_SPACE);

    let writer = feed.map(|feed| {
        let mut stdin_pipe = child.stdin.take().expect("stdin was piped");
        std::thread::spawn(move || {
            let _ = feed(&mut stdin_pipe);
        })
    });

    let mut stdout_pipe = child.stdout.take().expect("stdout was piped");
    let (out_tx, out_rx) = mpsc::channel::<Result<Vec<u8>, ToolError>>();
    let stdout_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 64 * 1024];
        let result = loop {
            match stdout_pipe.read(&mut chunk) {
                Ok(0) => break Ok(buf),
                Ok(n) => {
                    buf.extend_from_slice(&chunk[..n]);
                    if buf.len() > max_stdout {
                        break Err(ToolError::OutputCap(max_stdout));
                    }
                }
                Err(e) => break Err(ToolError::Read(e)),
            }
        };
        let _ = out_tx.send(result);
    });

    let mut stderr_pipe = child.stderr.take().expect("stderr was piped");
    let (err_tx, err_rx) = mpsc::channel::<Stderr>();
    let stderr_thread = std::thread::spawn(move || {
        let mut kept = Vec::new();
        let mut truncated = false;
        let mut tail = Vec::new();
        let mut chunk = [0u8; 16 * 1024];
        loop {
            match stderr_pipe.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let take = (max_stderr - kept.len()).min(n);
                    kept.extend_from_slice(&chunk[..take]);
                    truncated |= take < n;
                    tail.extend_from_slice(&chunk[..n]);
                    if tail.len() > 2 * STDERR_TAIL_BYTES {
                        tail.drain(..tail.len() - STDERR_TAIL_BYTES);
                    }
                }
            }
        }
        let tail = &tail[tail.len().saturating_sub(STDERR_TAIL_BYTES)..];
        let _ = err_tx.send(Stderr {
            head: String::from_utf8_lossy(&kept).into_owned(),
            truncated,
            tail: String::from_utf8_lossy(tail).into_owned(),
        });
    });

    // Wait for the exit status without ever blocking past
    // the deadline: a tool that closed its stdout and then
    // hung still gets killed on time.
    let wait_status = |child: &mut Child| -> Result<ExitStatus, ToolError> {
        loop {
            match child.try_wait() {
                Ok(Some(status)) => return Ok(status),
                Ok(None) => {
                    if Instant::now() >= deadline {
                        kill_tool(child);
                        return Err(ToolError::TimedOut);
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => return Err(ToolError::Wait(e)),
            }
        }
    };
    let (status, stdout) = loop {
        match out_rx.try_recv() {
            Ok(Ok(bytes)) => {
                let status = wait_status(&mut child)?;
                break (status, bytes);
            }
            Ok(Err(e)) => {
                kill_tool(&mut child);
                return Err(e);
            }
            Err(mpsc::TryRecvError::Empty) => {
                if Instant::now() >= deadline {
                    kill_tool(&mut child);
                    return Err(ToolError::TimedOut);
                }
                match child.try_wait() {
                    Ok(Some(status)) => {
                        // The process is gone; its stdout
                        // reader finishes at EOF. Give it
                        // a moment, then take what arrived.
                        match out_rx.recv_timeout(Duration::from_secs(2)) {
                            Ok(Ok(bytes)) => break (status, bytes),
                            Ok(Err(e)) => return Err(e),
                            Err(_) => break (status, Vec::new()),
                        }
                    }
                    Ok(None) => std::thread::sleep(Duration::from_millis(5)),
                    Err(e) => return Err(ToolError::Wait(e)),
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                let status = wait_status(&mut child)?;
                break (status, Vec::new());
            }
        }
    };
    if let Some(writer) = writer {
        let _ = writer.join();
    }
    // The reader threads are deliberately never joined
    // (issue #263): a grandchild that inherited the pipes
    // and holds them open would block a join for its whole
    // lifetime, defeating the deadline above. stderr is
    // worth a short grace through its channel, no more; the
    // stdout result already arrived through its own channel.
    let stderr = err_rx
        .recv_timeout(Duration::from_millis(200))
        .unwrap_or_default();
    drop(stdout_thread);
    drop(stderr_thread);
    Ok(ToolRun {
        status,
        stdout,
        stderr: stderr.head,
        stderr_truncated: stderr.truncated,
        stderr_tail: stderr.tail,
    })
}

/// Kill a runaway tool and reap it. On Unix the tool leads
/// its own process group, and the whole group takes the
/// signal, so the helpers a wrapper script started die with
/// it (audit R2-31). The direct kill follows: the fallback
/// there, and the whole kill on platforms without process
/// groups. The group kill is the whole mechanism on Unix
/// (issue #272): a descendant that detaches (setsid,
/// double-fork) escapes it by design of the tool contract,
/// which forbids daemonizing.
fn kill_tool(child: &mut Child) {
    #[cfg(unix)]
    {
        if let Ok(pid) = i32::try_from(child.id())
            && let Some(pid) = rustix::process::Pid::from_raw(pid)
        {
            let _ = rustix::process::kill_process_group(pid, rustix::process::Signal::KILL);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// The address-space ceiling for a tool child: generous
/// headroom over the largest legitimate workload (the
/// tools render or encode at most the shared pixel and
/// decode budgets, a few hundred megabytes of buffers)
/// while still turning a runaway allocation into a tool
/// error instead of a host-level failure.
pub(crate) const TOOL_MAX_ADDRESS_SPACE: u64 = 8 * 1024 * 1024 * 1024;

/// How far past the most CPU time a run can spend inside
/// its wall-clock budget the CPU limit sits (see
/// [`tool_cpu_limit`]).
const TOOL_CPU_GRACE: Duration = Duration::from_secs(5);

/// The CPU-time limit for a tool child whose run has
/// `timeout` of wall clock. The kernel counts CPU time
/// summed over every thread of the process, and FFmpeg
/// decodes and encodes on several, so one wall-clock
/// second can cost as many CPU seconds as the host has
/// usable cores. A limit of the timeout alone killed
/// ordinary multi-threaded extractions long before their
/// wall clock ran out (issue #406), so the limit scales
/// with the cores the child may run on. The wall clock
/// then fires first in every normal shape, and the CPU
/// limit only catches a child the wall clock somehow
/// leaves running.
pub(crate) fn tool_cpu_limit(timeout: Duration) -> Duration {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    cpu_limit_for(timeout, cores)
}

/// [`tool_cpu_limit`] for a given count of usable cores.
/// One core more than counted: `available_parallelism`
/// rounds a fractional CPU quota down (1.5 CPUs count as
/// one), and a child under that quota can still spend 1.5
/// CPU seconds per wall-clock second.
pub(crate) fn cpu_limit_for(timeout: Duration, cores: usize) -> Duration {
    let cores = u32::try_from(cores).unwrap_or(u32::MAX).saturating_add(1);
    timeout.saturating_mul(cores).saturating_add(TOOL_CPU_GRACE)
}

/// Apply the resource limits to a spawned tool child.
/// Best effort: a limit the platform refuses is skipped
/// (the wall-clock timeout and the process-group kill
/// remain the primary bounds). Linux only: `prlimit` is
/// the one call that sets another process's limits, and
/// it exists nowhere else. Elsewhere this is a no-op, as
/// before.
pub(crate) fn limit_child(pid: u32, cpu_limit: Duration, max_address_space: u64) {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let Ok(raw) = i32::try_from(pid) else {
            return;
        };
        let Some(pid) = rustix::process::Pid::from_raw(raw) else {
            return;
        };
        let cpu_secs = cpu_limit.as_secs().max(1);
        let _ = rustix::process::prlimit(
            Some(pid),
            rustix::process::Resource::Cpu,
            rustix::process::Rlimit {
                current: Some(cpu_secs),
                maximum: Some(cpu_secs),
            },
        );
        let _ = rustix::process::prlimit(
            Some(pid),
            rustix::process::Resource::As,
            rustix::process::Rlimit {
                current: Some(max_address_space),
                maximum: Some(max_address_space),
            },
        );
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = (pid, cpu_limit, max_address_space);
    }
}

/// Apply the resource limits to the calling process
/// itself (issue #447): the SVG worker caps itself as it
/// starts. `setrlimit` on oneself works on every Unix,
/// macOS included, unlike `prlimit` on another process
/// (macOS accepts the address-space limit but does not
/// enforce it; the CPU and core limits hold there).
/// A limit is never raised past the hard limit the process
/// inherited, and one the platform refuses is skipped (the
/// parent's wall-clock kill stays the bound that always
/// holds). Core dumps are switched off as well: a hostile
/// document that aborts the worker must not leave a core
/// file behind per request.
pub(crate) fn limit_self(cpu_limit: Duration, max_address_space: u64) {
    #[cfg(unix)]
    {
        use rustix::process::{Resource, Rlimit, getrlimit, setrlimit};
        let cap = |resource: Resource, value: u64| {
            let value = getrlimit(resource)
                .maximum
                .map_or(value, |max| max.min(value));
            let _ = setrlimit(
                resource,
                Rlimit {
                    current: Some(value),
                    maximum: Some(value),
                },
            );
        };
        cap(Resource::Cpu, cpu_limit.as_secs().max(1));
        cap(Resource::As, max_address_space);
        cap(Resource::Core, 0);
    }
    #[cfg(not(unix))]
    {
        let _ = (cpu_limit, max_address_space);
    }
}

/// Kill a child spawned as the leader of its own process
/// group, with the whole group, so nothing it started
/// outlives it. Best effort: a child that has already
/// exited is left for the caller's `wait` to reap.
pub(crate) fn kill_process_group(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        if let Some(pid) = i32::try_from(child.id())
            .ok()
            .and_then(rustix::process::Pid::from_raw)
        {
            let _ = rustix::process::kill_process_group(pid, rustix::process::Signal::KILL);
        }
    }
    let _ = child.kill();
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    // The kernel limits exist only where prlimit does;
    // elsewhere limit_child is a no-op and this child
    // would spin until the watchdog kills it.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn cpu_limit_stops_a_spinning_child() {
        let mut child = Command::new("/bin/sh")
            .args(["-c", "while :; do :; done"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        limit_child(child.id(), Duration::from_secs(1), TOOL_MAX_ADDRESS_SPACE);
        let started = std::time::Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if started.elapsed() > Duration::from_secs(6) {
                let _ = child.kill();
                panic!("the CPU limit did not stop the child");
            }
            std::thread::sleep(Duration::from_millis(25));
        };
        assert!(!status.success(), "{status}");
        assert!(started.elapsed() < Duration::from_secs(6));
    }

    #[test]
    fn cpu_limit_scales_with_the_usable_cores() {
        let timeout = Duration::from_secs(30);
        assert_eq!(cpu_limit_for(timeout, 1), Duration::from_secs(65));
        assert_eq!(cpu_limit_for(timeout, 4), Duration::from_secs(155));
        assert_eq!(cpu_limit_for(timeout, 64), Duration::from_secs(1955));
        // Never below what the wall clock allows on the
        // host running the test.
        let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        assert!(tool_cpu_limit(timeout) > timeout * u32::try_from(cores).unwrap());
        // Absurd inputs saturate instead of overflowing.
        assert_eq!(cpu_limit_for(Duration::MAX, 4), Duration::MAX);
        assert_eq!(
            cpu_limit_for(timeout, usize::MAX),
            timeout * u32::MAX + TOOL_CPU_GRACE
        );
    }

    /// Not a test on its own: the multi-threaded spinner
    /// that `cpu_limit_lets_a_multi_threaded_child_finish`
    /// runs as its child, by re-running this test binary.
    /// The CPU limit counts every thread of one process,
    /// and no stock tool spins several threads on demand.
    #[test]
    #[ignore = "a helper the CPU limit test runs as its child"]
    fn spin_threads_helper() {
        let Ok(spec) = std::env::var("CRABCROP_TEST_SPIN") else {
            return;
        };
        let (threads, millis) = spec.split_once(':').unwrap();
        let threads: usize = threads.parse().unwrap();
        let until = std::time::Instant::now() + Duration::from_millis(millis.parse().unwrap());
        let spinners: Vec<_> = (0..threads)
            .map(|_| {
                std::thread::spawn(move || {
                    let mut n = 0u64;
                    while std::time::Instant::now() < until {
                        n = std::hint::black_box(n.wrapping_add(1));
                    }
                })
            })
            .collect();
        for spinner in spinners {
            spinner.join().unwrap();
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn cpu_limit_lets_a_multi_threaded_child_finish() {
        // Issue #406: a child that keeps every core busy for
        // less than its wall-clock timeout must finish. With
        // a 3 s timeout, the old limit (the timeout plus the
        // grace, 8 CPU seconds) killed a 4-thread spinner on
        // a 4-core host after about 2 s of its 2.5 s run.
        let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        let threads = cores.clamp(2, 16);
        let timeout = Duration::from_secs(3);
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "toolrun::tests::spin_threads_helper",
                "--ignored",
                "--test-threads=1",
                "--quiet",
            ])
            .env("CRABCROP_TEST_SPIN", format!("{threads}:2500"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        limit_child(child.id(), tool_cpu_limit(timeout), TOOL_MAX_ADDRESS_SPACE);
        let started = std::time::Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            // Generous slack over the 2.5 s run for a loaded
            // test host; a kill shows up as a failed status
            // long before this.
            if started.elapsed() > Duration::from_secs(60) {
                let _ = child.kill();
                panic!("the spinner never finished");
            }
            std::thread::sleep(Duration::from_millis(25));
        };
        assert!(status.success(), "the CPU limit killed the child: {status}");
        // The helper really ran its spin (a renamed helper
        // would otherwise pass here without spinning).
        assert!(
            started.elapsed() >= Duration::from_secs(2),
            "the spinner ended after {:?}",
            started.elapsed()
        );
    }

    /// Not a test on its own: the process that
    /// `limit_self_caps_the_calling_process` starts, by
    /// re-running this test binary, caps itself here and
    /// reports what the caps did through its exit code.
    #[test]
    #[ignore = "a helper the self-limit test runs as its child"]
    fn limit_self_helper() {
        let Ok(mode) = std::env::var("CRABCROP_TEST_LIMIT_SELF") else {
            return;
        };
        use rustix::process::{Resource, getrlimit};
        // A test run under a tighter inherited cap keeps it:
        // limit_self never raises a limit.
        let space = getrlimit(Resource::As)
            .maximum
            .map_or(2 << 30, |max| max.min(2 << 30));
        limit_self(Duration::from_secs(1), 2 << 30);
        if getrlimit(Resource::Cpu).current != Some(1)
            || getrlimit(Resource::Core).current != Some(0)
        {
            std::process::exit(3);
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if getrlimit(Resource::As).current != Some(space) {
            std::process::exit(3);
        }
        let _ = space;
        if mode == "spin" {
            let mut n = 0u64;
            loop {
                n = std::hint::black_box(n.wrapping_add(1));
            }
        }
        // Linux enforces the address-space cap: 4 GiB no
        // longer fits. (macOS accepts the limit but does not
        // enforce it, so only the read-back above runs there.)
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if Vec::<u8>::new().try_reserve_exact(4 << 30).is_ok() {
            std::process::exit(4);
        }
        std::process::exit(0);
    }

    #[test]
    fn limit_self_caps_the_calling_process() {
        // Issue #447: the SVG worker caps itself with
        // setrlimit as it starts. In a child (the caps are
        // permanent), the CPU, core, and address-space limits
        // read back as set, a 4 GiB reservation fails under
        // the 2 GiB cap, and a spinner dies at its CPU limit.
        let helper = |mode: &str| {
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "toolrun::tests::limit_self_helper",
                    "--ignored",
                    "--test-threads=1",
                    "--quiet",
                ])
                .env("CRABCROP_TEST_LIMIT_SELF", mode)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap()
        };
        let status = helper("check").wait().unwrap();
        assert!(status.success(), "the caps did not hold: {status}");
        let mut spinner = helper("spin");
        let started = std::time::Instant::now();
        let status = loop {
            if let Some(status) = spinner.try_wait().unwrap() {
                break status;
            }
            if started.elapsed() > Duration::from_secs(30) {
                let _ = spinner.kill();
                panic!("the CPU limit did not stop the spinner");
            }
            std::thread::sleep(Duration::from_millis(25));
        };
        assert!(!status.success(), "{status}");
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn address_space_limit_stops_a_greedy_child() {
        // The greedy child allocates 3 GiB after a short
        // delay, under a 1 GiB cap set the moment it
        // spawns: the allocation fails with a memory
        // error instead of growing into the cap. The
        // delay matters: the limit lands right after
        // spawn, so the child must still be starting when
        // it does (an instant allocator can finish inside
        // the spawn-to-limit window under load, which is
        // what an earlier dd-based version of this test
        // did, flakily). python3 is the suites' usual
        // helper interpreter; skip when it is absent.
        if Command::new("python3").arg("--version").output().is_err() {
            return;
        }
        let child = Command::new("python3")
            .args([
                "-c",
                "import time; time.sleep(0.5); x = bytearray(3 * 1024**3)",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        limit_child(child.id(), Duration::from_secs(30), 1 << 30);
        let out = child.wait_with_output().unwrap();
        assert!(!out.status.success(), "{}", out.status);
    }

    /// A shell one-liner as a tool run, under generous
    /// bounds.
    fn shell<'a>(script: &'a [&'a str; 2], input: ToolInput<'a>) -> ToolCommand<'a> {
        ToolCommand {
            program: Path::new("/bin/sh"),
            args: script,
            input,
            max_stdout: 64 * 1024,
            max_stderr: 64 * 1024,
            deadline: std::time::Instant::now() + Duration::from_secs(20),
            cpu_limit: Duration::from_secs(20),
        }
    }

    /// The directory a run's stdout names, resolved the way
    /// the tool's `pwd -P` resolves it.
    fn printed_dir(run: &ToolRun) -> std::path::PathBuf {
        assert!(run.status.success(), "{}: {}", run.status, run.stderr);
        let printed = String::from_utf8(run.stdout.clone()).unwrap();
        std::fs::canonicalize(printed.trim()).unwrap()
    }

    #[test]
    fn tools_run_in_the_temporary_directory() {
        // Issue #481: every tool's working directory is the
        // temporary directory, never the server's own.
        let run = run_tool(shell(&["-c", "pwd -P"], ToolInput::Bytes(b""))).unwrap();
        assert_eq!(
            printed_dir(&run),
            std::fs::canonicalize(std::env::temp_dir()).unwrap()
        );
    }

    #[test]
    fn a_missing_temporary_directory_does_not_fail_the_run() {
        // A TMPDIR that names no directory used to fail every
        // run at spawn, with an error that blamed the program.
        // The tool now runs where the server runs.
        let missing =
            std::env::temp_dir().join(format!("crabcrop-toolrun-missing-{}", std::process::id()));
        assert!(!missing.exists());
        let run = run_tool_in(shell(&["-c", "pwd -P"], ToolInput::Bytes(b"")), &missing).unwrap();
        assert_eq!(
            printed_dir(&run),
            std::fs::canonicalize(std::env::current_dir().unwrap()).unwrap()
        );
    }

    #[test]
    fn a_timeout_kills_the_whole_process_group() {
        // Every tool, not only the encoder (audit R2-31): a
        // wrapper that forks a long-lived grandchild and waits
        // on it is killed at the deadline, grandchild and all.
        let pid_file =
            std::env::temp_dir().join(format!("crabcrop-toolrun-group-{}.pid", std::process::id()));
        // A file left by an earlier run under the same PID
        // would name some other process, alive or not.
        let _ = std::fs::remove_file(&pid_file);
        let script = format!("sleep 30 & echo $! > '{}'; wait", pid_file.display());
        let argv = ["-c", script.as_str()];
        let mut command = shell(&argv, ToolInput::Bytes(b""));
        command.deadline = std::time::Instant::now() + Duration::from_millis(300);
        let started = std::time::Instant::now();
        let err = run_tool(command).unwrap_err();
        assert!(matches!(err, ToolError::TimedOut), "{err:?}");
        assert!(started.elapsed() < Duration::from_secs(5));
        let pid = std::fs::read_to_string(&pid_file).unwrap();
        let _ = std::fs::remove_file(&pid_file);
        // Signal 0 finds nothing once the grandchild is gone;
        // give the kill a moment to land before concluding.
        let until = std::time::Instant::now() + Duration::from_secs(5);
        while Command::new("kill")
            .args(["-0", pid.trim()])
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
        {
            assert!(
                std::time::Instant::now() < until,
                "grandchild {} survived the process-group kill",
                pid.trim()
            );
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    #[test]
    fn the_excerpt_is_the_end_of_stderr() {
        // Issue #481: error text quotes the last 500
        // characters of stderr, where a tool reports its
        // failure, even past the head the run keeps.
        let argv = [
            "-c",
            "head -c 100000 /dev/zero | tr '\\000' x >&2; echo 'the real failure' >&2; exit 1",
        ];
        let mut command = shell(&argv, ToolInput::Bytes(b""));
        command.max_stderr = 1024;
        let run = run_tool(command).unwrap();
        assert!(!run.status.success());
        assert_eq!(run.stderr.len(), 1024);
        assert!(run.stderr_truncated);
        let excerpt = run.stderr_excerpt();
        assert!(excerpt.ends_with("xthe real failure"), "{excerpt}");
        // The last 500 characters, less the final newline the
        // trim drops.
        assert_eq!(excerpt.chars().count(), 499);
    }

    #[test]
    fn the_excerpt_counts_characters_and_trims() {
        assert_eq!(excerpt(""), "no error output");
        assert_eq!(excerpt(" \n\t\n"), "no error output");
        assert_eq!(excerpt("  short failure\n"), "short failure");
        // Characters, not bytes: 600 two-byte characters keep
        // their last 500, whole.
        let text = format!("{}{}", "a".repeat(10), "\u{e9}".repeat(600));
        assert_eq!(excerpt(&text), "\u{e9}".repeat(500));
        // Whitespace inside the last 500 counts; only the
        // edges are trimmed.
        let text = format!("{}end\n{}", "b".repeat(600), " ".repeat(10));
        let quoted = excerpt(&text);
        assert!(quoted.ends_with("bend"), "{quoted}");
        assert_eq!(quoted.chars().count(), 500 - 10 - 1);
    }
}
