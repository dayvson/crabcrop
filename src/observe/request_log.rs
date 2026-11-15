//! The request log on standard output (issue #642).
//!
//! Every image request leaves one JSON line on standard
//! output, and every purge and upload an audit line.
//! Through 0.12 each was a `println!` in its handler: one
//! `write` on a runtime worker before the response left.
//! With standard output a file that was about a tenth of a
//! memory hit. With a pipe, which is what a container
//! runtime or a service manager gives a process, it was
//! twice the rest of the request, because every line woke
//! the reader on another core.
//!
//! A [`RequestLog`] takes those lines. In its default
//! form, [`RequestLog::to_stdout`], a handler appends its
//! line to a queue in memory and goes on, and one thread,
//! the writer, writes what has gathered: at most about
//! once a millisecond, in one `write`. What that keeps and
//! what it changes:
//!
//! - **Text and order.** The queue is one string of whole
//!   lines, appended under a lock, and the writer is the
//!   only thing that writes to standard output once the
//!   server serves: the purge, upload and shutdown lines
//!   go through the same queue. So lines come out in the
//!   order they were logged, and no line is ever split by
//!   another.
//! - **Purge and upload lines are written before their
//!   response**, as they were: [`RequestLog::event`]
//!   returns once the writer has written the line.
//! - **A request line is written up to about a millisecond
//!   after its request**, no longer before the response.
//!   A process that is killed loses the lines still
//!   queued.
//! - **Standard output nobody reads.** The queue is full
//!   at 8 MiB of text ([`QUEUE_BYTES`]), counting what the
//!   writer has in hand. A handler asks for room before it
//!   does anything else ([`RequestLog::room`]), and while
//!   the queue is full that waits: the request is not
//!   handled, and not answered, until the writer has made
//!   room, as a request waited in `write` before. No line
//!   is dropped: [`RequestLog::line`] never refuses one,
//!   so the requests under way when the queue fills still
//!   leave theirs, and past its 8 MiB the queue takes
//!   their lines and nothing else. A request whose client
//!   gives up while it waits for room was never handled:
//!   it owes no line and holds nothing. The wait is a
//!   task's, not a thread's: the runtime's workers are
//!   never parked on the queue, so the health check,
//!   `/metrics`, timers and a shutdown signal are still
//!   served.
//! - **Standard output that fails.** A write that fails
//!   (the reader closed the pipe, the disk is full) drops
//!   the lines it had not written and counts them on
//!   stderr, and the server keeps serving; the next write
//!   tries again, so the log resumes when standard output
//!   takes lines again. A write that would block on a
//!   descriptor in non-blocking mode is not a failure: the
//!   writer waits and tries again, as a blocking write
//!   waits.
//! - **Shutdown.** [`RequestLog::finish`] queues the last
//!   line and waits for the queue to be written, for at
//!   most the limit its caller gives ([`SHUTDOWN_WAIT`]),
//!   and says how many lines were left.
//!
//! The other form, [`RequestLog::inline`], writes every
//! line with `println!` where it is logged, as before:
//! `CRABCROP_LOG_SYNC=1` selects it, and the tests' state
//! uses it, so a test's lines stay in its captured output.

use std::io::{self, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

/// The text the queue is full at, in bytes: the lines
/// waiting for the writer and the batch it is writing.
/// A request line is a few hundred bytes, so this is tens
/// of thousands of requests: room for a reader that falls
/// behind for a moment, and a bound on what a reader that
/// stops costs.
pub const QUEUE_BYTES: usize = 8 * 1024 * 1024;

/// How long a shutdown waits for the queue to be written
/// before it exits without the lines that are left. A
/// standard output that is being read takes the queue in
/// well under a millisecond; one that has not taken it in
/// a second is not being read, or not fast enough to wait
/// for.
pub const SHUTDOWN_WAIT: Duration = Duration::from_secs(1);

/// The writer starts a write no sooner than this after it
/// started the one before, so lines that arrive close
/// together share a write. The first line after a pause
/// is written at once.
const WRITE_EVERY: Duration = Duration::from_millis(1);

/// While standard output keeps failing, stderr gets one
/// line about the lines dropped at most this often.
const REPORT_EVERY: Duration = Duration::from_secs(10);

/// The longest the writer sleeps between two tries at a
/// descriptor in non-blocking mode that takes nothing.
const LONGEST_RETRY_PAUSE: Duration = Duration::from_millis(50);

/// A batch buffer keeps at most this much capacity between
/// writes, so the memory a stalled reader made the queue
/// take is given back once the queue has been written.
const KEPT_CAPACITY: usize = 64 * 1024;

/// The writer thread's name, as `ps -L` and a debugger
/// show it.
const WRITER_THREAD: &str = "crabcrop-log";

/// Where a server's log lines go: the request lines, the
/// purge and upload lines, and the shutdown lines. Cloning
/// it shares the one log. See the module documentation for
/// what each form does with a line.
#[derive(Clone)]
pub struct RequestLog(Arc<Form>);

enum Form {
    /// Every line is written where it is logged.
    Inline,
    /// [`Form::Inline`] to a stream a test reads back.
    #[cfg(test)]
    InlineTo(Mutex<Box<dyn Write + Send>>),
    /// Lines are queued for the writer thread.
    Queued(Handle),
}

/// The handlers' end of a queue. When the last clone of
/// its log is dropped, the writer writes what is queued
/// and ends.
struct Handle(Arc<Queue>);

impl Drop for Handle {
    fn drop(&mut self) {
        self.0.lock().closed = true;
        self.0.wake.notify_one();
        self.0.progress.notify_waiters();
    }
}

impl std::fmt::Debug for RequestLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match &*self.0 {
            Form::Inline => "RequestLog(inline)",
            #[cfg(test)]
            Form::InlineTo(_) => "RequestLog(inline)",
            Form::Queued(_) => "RequestLog(queued)",
        })
    }
}

impl RequestLog {
    /// A log whose lines are written with `println!` where
    /// they are logged, before the caller goes on.
    pub fn inline() -> Self {
        RequestLog(Arc::new(Form::Inline))
    }

    /// A log whose lines are queued and written to standard
    /// output by a thread of its own, which this starts.
    pub fn to_stdout() -> io::Result<Self> {
        Self::queued(Box::new(Stdout), QUEUE_BYTES, Box::new(note_on_stderr))
    }

    /// A queued log over `sink` that is full at `cap` bytes
    /// and reports what it drops through `note`.
    fn queued(sink: Box<dyn Sink>, cap: usize, note: Note) -> io::Result<Self> {
        let queue = Arc::new(Queue::new(cap));
        let writer = Writer::new(Arc::clone(&queue), sink, note);
        // A thread of its own, not a runtime worker and not
        // a blocking-pool thread: it blocks in `write` for
        // as long as standard output does, and must hold up
        // neither a request nor a render. It is detached,
        // and ends with the process.
        std::thread::Builder::new()
            .name(WRITER_THREAD.to_string())
            .spawn(move || writer.run())?;
        Ok(RequestLog(Arc::new(Form::Queued(Handle(queue)))))
    }

    /// [`RequestLog::inline`], written to `out`.
    #[cfg(test)]
    pub(crate) fn inline_to(out: impl Write + Send + 'static) -> Self {
        RequestLog(Arc::new(Form::InlineTo(Mutex::new(Box::new(out)))))
    }

    /// [`RequestLog::to_stdout`], written to `out`.
    #[cfg(test)]
    pub(crate) fn queued_to(out: impl Write + Send + 'static) -> Self {
        Self::queued(Box::new(Stream(out)), QUEUE_BYTES, Box::new(|_| {})).unwrap()
    }

    /// Wait until the log has room for a line. A handler
    /// that logs calls this before it does anything else,
    /// so while the queue is full a request waits here,
    /// unhandled and unanswered, and one whose client gives
    /// up has cost nothing and owes no line. It returns at
    /// once unless the queue is full, and always for an
    /// inline log, whose lines wait in their `write`.
    pub async fn room(&self) {
        if let Form::Queued(handle) = &*self.0 {
            // One load for every request: the queue's lock
            // is taken here only while the queue is full.
            if handle.0.full.load(Ordering::Relaxed) {
                handle.0.room().await;
            }
        }
    }

    /// Log one line: a request's, or one of the shutdown's.
    /// Queued, the line is in the queue when this returns,
    /// whatever the queue holds: a line is never refused
    /// for want of room, and nothing waits here.
    pub fn line(&self, line: &str) {
        match &*self.0 {
            Form::Inline => println!("{line}"),
            #[cfg(test)]
            Form::InlineTo(out) => write_inline(out, line),
            Form::Queued(handle) => {
                handle.0.push(line, false);
            }
        }
    }

    /// Log a line that must be on standard output before
    /// its caller goes on, a purge's or an upload's: it
    /// returns once the line is written, after every line
    /// logged before it, or once standard output has
    /// refused it.
    pub async fn event(&self, line: &str) {
        match &*self.0 {
            Form::Inline => println!("{line}"),
            #[cfg(test)]
            Form::InlineTo(out) => write_inline(out, line),
            Form::Queued(handle) => {
                if let Some(number) = handle.0.push(line, true) {
                    handle.0.settled(number).await;
                }
            }
        }
    }

    /// End the log: queue `last`, if any, as its last line,
    /// then wait for the queue to be written, for at most
    /// `limit`. Returns how many lines were not written by
    /// then, `last` among them: none, unless standard
    /// output did not take them in time. Lines logged after
    /// this are not written, so the log ends with `last`,
    /// and nothing waits for room any more.
    pub async fn finish(&self, last: Option<&str>, limit: Duration) -> u64 {
        let queue = match &*self.0 {
            Form::Inline => {
                if let Some(last) = last {
                    println!("{last}");
                }
                return 0;
            }
            #[cfg(test)]
            Form::InlineTo(out) => {
                if let Some(last) = last {
                    write_inline(out, last);
                }
                return 0;
            }
            Form::Queued(handle) => &handle.0,
        };
        let number = {
            let mut state = queue.lock();
            let ended = std::mem::replace(&mut state.finished, true);
            match last {
                Some(last) if !ended => queue.append(state, last, true),
                _ => {
                    // What is queued is written without
                    // waiting for more lines to share the
                    // write.
                    let number = state.accepted;
                    state.urgent = true;
                    drop(state);
                    queue.wake.notify_one();
                    number
                }
            }
        };
        // The requests that wait for room see that the log
        // has ended, and go on.
        queue.progress.notify_waiters();
        let _ = tokio::time::timeout(limit, queue.settled(number)).await;
        let state = queue.lock();
        state.accepted - state.settled
    }

    /// The queue's counters, for the tests.
    #[cfg(test)]
    pub(crate) fn stats(&self) -> Stats {
        match &*self.0 {
            Form::Queued(handle) => {
                let state = handle.0.lock();
                Stats {
                    held_bytes: state.held(),
                    full: handle.0.full.load(Ordering::Relaxed),
                    waiting: state.waiting,
                    dropped: state.dropped,
                    wakeups: state.wakeups,
                    writer_gone: state.writer_gone,
                }
            }
            _ => Stats::default(),
        }
    }
}

/// What [`RequestLog::stats`] reads.
#[cfg(test)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Stats {
    /// Bytes queued and in the writer's hands.
    pub(crate) held_bytes: usize,
    /// The queue is full: a request that asks for room
    /// waits.
    pub(crate) full: bool,
    /// Requests waiting for room.
    pub(crate) waiting: u64,
    /// Lines given up on.
    pub(crate) dropped: u64,
    /// How often the writer has woken from a wait.
    pub(crate) wakeups: u64,
    /// The writer thread has ended.
    pub(crate) writer_gone: bool,
}

#[cfg(test)]
fn write_inline(out: &Mutex<Box<dyn Write + Send>>, line: &str) {
    let mut out = out.lock().unwrap_or_else(PoisonError::into_inner);
    out.write_all(line.as_bytes()).unwrap();
    out.write_all(b"\n").unwrap();
}

/// The queue between the handlers and the writer.
struct Queue {
    state: Mutex<State>,
    /// Wakes the writer: a line arrived while it slept, or
    /// someone wants the queue written without delay.
    wake: Condvar,
    /// Notified after every write, when lines have settled
    /// and room has come free, and when the log ends.
    progress: Notify,
    /// The queue holds `cap` bytes or more. Set and cleared
    /// under the lock, and read without it by every request
    /// that asks for room, which takes the lock only when
    /// this says the queue is full.
    full: AtomicBool,
    /// The bytes of text the queue is full at.
    cap: usize,
}

struct State {
    /// The queued lines, each ending in a newline, in the
    /// order they were logged.
    text: String,
    /// How many lines `text` holds.
    lines: u64,
    /// The bytes of the batch the writer took and has not
    /// finished with. They count as held, so the queue is
    /// full by all the text held, not by `text` alone.
    writing: usize,
    /// How many lines that batch holds.
    writing_lines: u64,
    /// Lines queued since the start: the number of the
    /// last one.
    accepted: u64,
    /// Lines the writer is done with, written or given up
    /// on: every line numbered up to this has settled.
    settled: u64,
    /// Lines given up on since the start.
    #[cfg(test)]
    dropped: u64,
    /// Requests waiting for room.
    #[cfg(test)]
    waiting: u64,
    /// The writer sleeps until a line arrives: whoever
    /// queues one wakes it.
    asleep: bool,
    /// Someone waits for the queue to be written: the next
    /// write starts at once.
    urgent: bool,
    /// The log was finished: it takes no more lines.
    finished: bool,
    /// No handle is left: the writer writes what is queued
    /// and ends.
    closed: bool,
    /// The writer thread has ended.
    writer_gone: bool,
    /// How often the writer has woken from a wait.
    #[cfg(test)]
    wakeups: u64,
}

impl State {
    /// The bytes of text held: queued, and in the writer's
    /// hands.
    fn held(&self) -> usize {
        self.text.len() + self.writing
    }
}

impl Queue {
    fn new(cap: usize) -> Self {
        Queue {
            state: Mutex::new(State {
                text: String::new(),
                lines: 0,
                writing: 0,
                writing_lines: 0,
                accepted: 0,
                settled: 0,
                #[cfg(test)]
                dropped: 0,
                #[cfg(test)]
                waiting: 0,
                asleep: false,
                urgent: false,
                finished: false,
                closed: false,
                writer_gone: false,
                #[cfg(test)]
                wakeups: 0,
            }),
            wake: Condvar::new(),
            progress: Notify::new(),
            full: AtomicBool::new(false),
            cap,
        }
    }

    /// The state. A panic while it was held leaves nothing
    /// half done that the next holder could not carry on
    /// from, so a poisoned lock is taken all the same: the
    /// log must outlive a panic in one request.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Queue `line`. Returns its number, or `None` if the
    /// log has ended and takes no more lines.
    fn push(&self, line: &str, urgent: bool) -> Option<u64> {
        let state = self.lock();
        if state.finished || state.closed {
            return None;
        }
        Some(self.append(state, line, urgent))
    }

    /// Append `line` and wake the writer if it sleeps, or
    /// if this line is `urgent` and the writer does not
    /// know yet. Returns the line's number.
    fn append(&self, mut state: MutexGuard<'_, State>, line: &str, urgent: bool) -> u64 {
        debug_assert!(!line.contains('\n'), "a log line is one line");
        state.text.push_str(line);
        state.text.push('\n');
        state.lines += 1;
        state.accepted += 1;
        let number = state.accepted;
        if state.held() >= self.cap {
            self.full.store(true, Ordering::Relaxed);
        }
        // One wake for a sleeping writer, by the line that
        // finds it asleep: the lines after it see a writer
        // that is on its way, and cost no system call.
        let wake = state.asleep || (urgent && !state.urgent);
        state.asleep = false;
        state.urgent |= urgent;
        drop(state);
        if wake {
            self.wake.notify_one();
        }
        number
    }

    /// The writer is done with a batch: its lines have
    /// settled, `_dropped` of them unwritten, and the room
    /// it took is free.
    fn settle(&self, _dropped: u64) {
        let mut state = self.lock();
        let lines = std::mem::take(&mut state.writing_lines);
        state.writing = 0;
        state.settled += lines;
        #[cfg(test)]
        {
            state.dropped += _dropped;
        }
        if state.held() < self.cap {
            self.full.store(false, Ordering::Relaxed);
        }
        drop(state);
        self.progress.notify_waiters();
    }

    /// Wait while the queue is full.
    async fn room(&self) {
        #[cfg(test)]
        let _waiting = Waiting::new(self);
        loop {
            // Created before the look below, so a write that
            // ends in between is not missed.
            let progress = self.progress.notified();
            {
                let state = self.lock();
                let ended = state.finished || state.closed || state.writer_gone;
                if ended || state.held() < self.cap {
                    return;
                }
            }
            progress.await;
        }
    }

    /// Wait until line `number` has settled.
    async fn settled(&self, number: u64) {
        loop {
            // As in `room`.
            let progress = self.progress.notified();
            {
                let state = self.lock();
                if state.settled >= number || state.writer_gone {
                    return;
                }
            }
            progress.await;
        }
    }
}

/// One request's wait for room, counted for the tests from
/// when it starts to when it ends or is dropped.
#[cfg(test)]
struct Waiting<'a>(&'a Queue);

#[cfg(test)]
impl<'a> Waiting<'a> {
    fn new(queue: &'a Queue) -> Self {
        queue.lock().waiting += 1;
        Waiting(queue)
    }
}

#[cfg(test)]
impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.lock().waiting -= 1;
    }
}

/// Where the writer writes: standard output, or a test's
/// stand-in for it.
trait Sink: Send {
    /// Run `write` with the stream, held for as long as
    /// `write` runs.
    fn hold(&mut self, write: &mut dyn FnMut(&mut dyn Write));
}

struct Stdout;

impl Sink for Stdout {
    fn hold(&mut self, write: &mut dyn FnMut(&mut dyn Write)) {
        // The lock is held for the whole batch. Anything
        // else in the process that prints to standard
        // output waits for the batch to end, so it cannot
        // land inside a line.
        write(&mut io::stdout().lock());
    }
}

/// Any stream as a [`Sink`], for the tests.
#[cfg(test)]
struct Stream<W>(W);

#[cfg(test)]
impl<W: Write + Send> Sink for Stream<W> {
    fn hold(&mut self, write: &mut dyn FnMut(&mut dyn Write)) {
        write(&mut self.0);
    }
}

/// How the writer says what it dropped.
type Note = Box<dyn FnMut(&str) + Send>;

/// A line on stderr. Not `eprintln!`, which panics when
/// stderr is closed too: that would end the writer over a
/// report nobody can read.
fn note_on_stderr(text: &str) {
    let _ = writeln!(io::stderr().lock(), "{text}");
}

/// The writer thread's state.
struct Writer {
    queue: Arc<Queue>,
    sink: Box<dyn Sink>,
    note: Note,
    /// The buffer the queue gets when the writer takes its
    /// text: the one the batch before was written from.
    spare: String,
    /// The tail of a line that a failed write cut: written
    /// first by the next write, so the line ends where it
    /// should and the one after it starts on its own.
    carry: Vec<u8>,
    /// When the last write started.
    last_write: Option<Instant>,
    /// The last write failed.
    failing: bool,
    /// Lines dropped that stderr has not been told about.
    unreported: u64,
    /// When stderr was last told.
    reported: Option<Instant>,
}

impl Writer {
    fn new(queue: Arc<Queue>, sink: Box<dyn Sink>, note: Note) -> Self {
        Writer {
            queue,
            sink,
            note,
            spare: String::new(),
            carry: Vec::new(),
            last_write: None,
            failing: false,
            unreported: 0,
            reported: None,
        }
    }

    /// The thread's body: write batches until the log
    /// closes. A panic in a write must not end the thread:
    /// with no writer the queue would fill, and every
    /// request would wait for room that never comes. The
    /// batch in hand is given up, and the loop goes on.
    fn run(mut self) {
        loop {
            if catch_unwind(AssertUnwindSafe(|| self.write_until_closed())).is_ok() {
                break;
            }
            let lost = self.queue.lock().writing_lines;
            self.queue.settle(lost);
            self.spare = String::new();
            (self.note)(&format!(
                "request log: the writer failed and starts again; {lost} lines dropped"
            ));
            // No faster than a write, should it fail again.
            std::thread::sleep(WRITE_EVERY);
        }
        self.queue.lock().writer_gone = true;
        self.queue.progress.notify_waiters();
    }

    fn write_until_closed(&mut self) {
        loop {
            let mut state = self.queue.lock();
            // Sleep until a line arrives: no timer, no
            // polling. The line that finds `asleep` set
            // wakes this thread.
            while state.text.is_empty() && !state.closed {
                state.urgent = false;
                state.asleep = true;
                state = self.wait(state, None);
            }
            state.asleep = false;
            if state.text.is_empty() {
                // Closed, and everything is written. A line
                // cut by a failed write gets one last try.
                drop(state);
                if !self.carry.is_empty() {
                    self.write(&[]);
                }
                return;
            }
            // Lines that arrive close together share a
            // write: this one starts no sooner than
            // `WRITE_EVERY` after the last, unless someone
            // waits for it.
            if let Some(last) = self.last_write {
                let due = last + WRITE_EVERY;
                while !state.urgent && !state.closed {
                    let now = Instant::now();
                    if now >= due {
                        break;
                    }
                    state = self.wait(state, Some(due - now));
                }
            }
            let batch = std::mem::replace(&mut state.text, std::mem::take(&mut self.spare));
            state.writing = batch.len();
            state.writing_lines = std::mem::take(&mut state.lines);
            state.urgent = false;
            let finishing = state.finished;
            drop(state);

            self.last_write = Some(Instant::now());
            let dropped = self.write(batch.as_bytes());
            if finishing {
                // The last lines of a shutdown: say now what
                // was dropped, not at the next report.
                self.report(None);
            }
            self.queue.settle(dropped);

            self.spare = batch;
            self.spare.clear();
            self.spare.shrink_to(KEPT_CAPACITY);
        }
    }

    /// Wait on the queue's condition, for `limit` at most.
    fn wait<'a>(
        &self,
        state: MutexGuard<'a, State>,
        limit: Option<Duration>,
    ) -> MutexGuard<'a, State> {
        let wake = &self.queue.wake;
        #[allow(unused_mut)]
        let mut state = match limit {
            None => wake.wait(state).unwrap_or_else(PoisonError::into_inner),
            Some(limit) => {
                wake.wait_timeout(state, limit)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0
            }
        };
        #[cfg(test)]
        {
            state.wakeups += 1;
        }
        state
    }

    /// Write `batch`, which holds whole lines. Returns how
    /// many of them were given up on.
    fn write(&mut self, batch: &[u8]) -> u64 {
        let carry = &mut self.carry;
        let mut outcome = Ok(());
        self.sink.hold(&mut |out| {
            outcome = write_lines(out, carry, batch);
        });
        match outcome {
            Ok(()) => {
                if self.failing {
                    self.failing = false;
                    self.report(Some("request log: standard output takes lines again"));
                    self.reported = None;
                }
                0
            }
            Err((error, dropped)) => {
                self.failing = true;
                self.unreported += dropped;
                let due = self.reported.is_none_or(|at| at.elapsed() >= REPORT_EVERY);
                if due {
                    self.report(Some(&format!(
                        "request log: cannot write to standard output: {error}"
                    )));
                }
                dropped
            }
        }
    }

    /// Tell stderr how many lines were dropped since it was
    /// last told, after `what` happened; with no `what`,
    /// only if there are any.
    fn report(&mut self, what: Option<&str>) {
        let dropped = std::mem::take(&mut self.unreported);
        let text = match (what, dropped) {
            (None, 0) => return,
            (None, n) => format!("request log: {n} lines dropped"),
            (Some(what), 0) => what.to_string(),
            (Some(what), n) => format!("{what}; {n} lines dropped"),
        };
        (self.note)(&text);
        self.reported = Some(Instant::now());
    }
}

/// Write the tail an earlier failure left in `carry`, then
/// `batch`. On a failure: the error, and how many of the
/// batch's lines were not written at all. A line the
/// failure cut is not one of them: its tail goes to
/// `carry`, which the next write starts with, so the line
/// is whole once standard output takes writes again and
/// the line after it starts on its own.
fn write_lines(
    out: &mut dyn Write,
    carry: &mut Vec<u8>,
    batch: &[u8],
) -> Result<(), (io::Error, u64)> {
    let lines_in = |bytes: &[u8]| bytes.iter().filter(|byte| **byte == b'\n').count() as u64;
    let mut done = 0;
    while done < carry.len() {
        match write_some(out, &carry[done..]) {
            Ok(n) => done += n,
            Err(error) => {
                carry.drain(..done);
                return Err((error, lines_in(batch)));
            }
        }
    }
    carry.clear();
    let mut at = 0;
    while at < batch.len() {
        match write_some(out, &batch[at..]) {
            Ok(n) => at += n,
            Err(error) => {
                let rest = &batch[at..];
                let mut lost = lines_in(rest);
                if at != 0 && batch[at - 1] != b'\n' {
                    let end = rest.iter().position(|byte| *byte == b'\n');
                    carry.extend_from_slice(&rest[..end.map_or(rest.len(), |end| end + 1)]);
                    lost = lost.saturating_sub(1);
                }
                return Err((error, lost));
            }
        }
    }
    let mut pause = Duration::ZERO;
    loop {
        match out.flush() {
            Ok(()) => return Ok(()),
            Err(error) if retry(&error, &mut pause) => {}
            // Every line was taken. What the stream itself
            // still holds, it writes first next time.
            Err(error) => return Err((error, 0)),
        }
    }
}

/// One `write` of some of `bytes`: how many it took.
fn write_some(out: &mut dyn Write, bytes: &[u8]) -> io::Result<usize> {
    let mut pause = Duration::ZERO;
    loop {
        match out.write(bytes) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) => return Ok(n),
            Err(error) if retry(&error, &mut pause) => {}
            Err(error) => return Err(error),
        }
    }
}

/// Whether to try again after `error`: at once after an
/// interrupted call, and after a pause, which grows from
/// `pause`, when a descriptor in non-blocking mode takes
/// nothing. That one is full, not failing: its reader is
/// behind, and a blocking write would wait for it too.
fn retry(error: &io::Error, pause: &mut Duration) -> bool {
    match error.kind() {
        io::ErrorKind::Interrupted => true,
        io::ErrorKind::WouldBlock => {
            *pause = (*pause * 2).clamp(WRITE_EVERY, LONGEST_RETRY_PAUSE);
            std::thread::sleep(*pause);
            true
        }
        _ => false,
    }
}

#[cfg(test)]
pub(crate) mod fake;
#[cfg(test)]
mod tests;
