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
//! - **Standard output nobody reads.** The queue holds
//!   8 MiB of text ([`QUEUE_BYTES`]), counting what the
//!   writer has in hand. Past that a request waits for
//!   room before its response leaves, as a request waited
//!   in `write` before, and no line is dropped. The wait
//!   is a task's, not a thread's: the runtime's workers
//!   are never parked on the queue, so the health check,
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
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use tokio::sync::{Notify, Semaphore};

/// The most text the queue holds, in bytes: the lines
/// waiting for the writer and the batch it is writing.
/// A request line is a few hundred bytes, so this is tens
/// of thousands of requests: room for a reader that falls
/// behind for a moment, and a bound on what a reader that
/// stops costs.
pub const QUEUE_BYTES: usize = 8 * 1024 * 1024;

/// How long a shutdown waits for the queue to be written
/// before it exits without the lines that are left. A
/// standard output that is being read takes the queue in
/// well under a millisecond; one that takes nothing in a
/// second is not being read.
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

    /// A queued log over `sink` that holds `cap` bytes and
    /// reports what it drops through `note`.
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

    /// Log one request's line. Queued, it returns as soon
    /// as the line is in the queue, which is at once unless
    /// the queue is full: then it waits for room, so the
    /// request's response waits with it.
    pub async fn line(&self, line: &str) {
        match &*self.0 {
            Form::Inline => println!("{line}"),
            #[cfg(test)]
            Form::InlineTo(out) => write_inline(out, line),
            Form::Queued(handle) => {
                handle.0.push(line, false).await;
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
                if let Some(number) = handle.0.push(line, true).await {
                    handle.0.settled(number).await;
                }
            }
        }
    }

    /// Log a line of the shutdown, in order with the lines
    /// around it. It never waits: a shutdown must not hang
    /// on a standard output nobody reads, so the line is
    /// queued whatever the queue holds.
    pub fn note(&self, line: &str) {
        match &*self.0 {
            Form::Inline => println!("{line}"),
            #[cfg(test)]
            Form::InlineTo(out) => write_inline(out, line),
            Form::Queued(handle) => handle.0.push_past_the_cap(line),
        }
    }

    /// End the log: queue `last`, if any, as its last line,
    /// then wait for the queue to be written, for at most
    /// `limit`. Returns how many lines were not written by
    /// then, `last` among them: none, unless standard
    /// output takes nothing. Lines logged after this are
    /// not written, so the log ends with `last`.
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
        let (number, waiting) = {
            let mut state = queue.lock();
            // Lines that still wait for room are refused
            // from here on: they are left too.
            let waiting = if state.finished { 0 } else { state.waiting };
            state.finished = true;
            let number = match last {
                Some(last) => queue.append(state, last, true),
                None => {
                    let number = state.accepted;
                    state.urgent = true;
                    drop(state);
                    queue.wake.notify_one();
                    number
                }
            };
            (number, waiting)
        };
        // The lines that wait see that the log has ended,
        // and let their requests go.
        queue.progress.notify_waiters();
        let _ = tokio::time::timeout(limit, queue.settled(number)).await;
        let state = queue.lock();
        (state.accepted - state.settled) + waiting
    }

    /// The queue's counters, for the tests.
    #[cfg(test)]
    fn stats(&self) -> Stats {
        match &*self.0 {
            Form::Queued(handle) => {
                let state = handle.0.lock();
                Stats {
                    held_bytes: state.text.len() + state.writing,
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
struct Stats {
    /// Bytes queued and in the writer's hands.
    held_bytes: usize,
    /// Lines waiting for room.
    waiting: u64,
    /// Lines given up on.
    dropped: u64,
    /// How often the writer has woken from a wait.
    wakeups: u64,
    /// The writer thread has ended.
    writer_gone: bool,
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
    /// and room has come free, and when the log closes.
    progress: Notify,
    /// The lines that wait for room look for it one at a
    /// time, in the order they came.
    turn: Semaphore,
    /// The most bytes of text held at once.
    cap: usize,
}

struct State {
    /// The queued lines, each ending in a newline, in the
    /// order they were logged.
    text: String,
    /// How many lines `text` holds.
    lines: u64,
    /// The bytes of the batch the writer took and has not
    /// finished with. They count against the cap, so the
    /// cap bounds all the text held, not the queue alone.
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
    /// Lines that found the queue full and wait for room.
    waiting: u64,
    /// The writer sleeps until a line arrives: whoever
    /// queues one wakes it.
    asleep: bool,
    /// Someone waits for the queue to be written, or for
    /// room: the next write starts at once.
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

/// Why a line was not queued.
enum Refused {
    /// The queue is full, or other lines already wait for
    /// room. The line is counted as waiting.
    Full,
    /// The log was finished or closed.
    Closed,
}

/// One line's place among those that wait for room: taken
/// when the queue refuses the line, given up when the line
/// is queued or its wait ends.
struct Waiting(Arc<Queue>);

impl Drop for Waiting {
    fn drop(&mut self) {
        self.0.lock().waiting -= 1;
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
            turn: Semaphore::new(1),
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

    /// Queue `line`, waiting for room if the queue is
    /// full. Returns the line's number, or `None` if the
    /// log takes no more lines.
    async fn push(self: &Arc<Self>, line: &str, urgent: bool) -> Option<u64> {
        match self.try_push(line, urgent, false) {
            Ok(number) => Some(number),
            Err(Refused::Closed) => None,
            Err(Refused::Full) => {
                let waiting = Waiting(Arc::clone(self));
                let line = line.to_string();
                // The wait is a task of its own. A request
                // dropped while it waits, because its client
                // left, still leaves its line, as it did when
                // its handler wrote the line itself.
                tokio::spawn(async move { waiting.push_when_room(&line, urgent).await })
                    .await
                    .ok()
                    .flatten()
            }
        }
    }

    /// Queue `line` if there is room for it. `waited` says
    /// the line already holds its place among the waiting.
    fn try_push(&self, line: &str, urgent: bool, waited: bool) -> Result<u64, Refused> {
        let mut state = self.lock();
        if state.finished || state.closed {
            return Err(Refused::Closed);
        }
        // With nothing held a line always fits, so one
        // longer than the cap still gets through.
        let held = state.text.len() + state.writing;
        let room = held == 0 || held + line.len() < self.cap;
        // A line does not pass those that wait for room.
        if room && (waited || state.waiting == 0) {
            return Ok(self.append(state, line, urgent));
        }
        if !waited {
            state.waiting += 1;
        }
        // The writer may be holding back for more lines:
        // with some waiting for room it writes at once.
        let wake = !state.urgent;
        state.urgent = true;
        drop(state);
        if wake {
            self.wake.notify_one();
        }
        Err(Refused::Full)
    }

    /// Queue `line` whatever the queue holds.
    fn push_past_the_cap(&self, line: &str) {
        let state = self.lock();
        if !state.finished && !state.closed {
            self.append(state, line, false);
        }
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

    /// Wait until line `number` has settled.
    async fn settled(&self, number: u64) {
        loop {
            // Created before the look below, so a write that
            // ends in between is not missed.
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

impl Waiting {
    /// Queue `line` once there is room, after the lines
    /// that waited before it.
    async fn push_when_room(self, line: &str, urgent: bool) -> Option<u64> {
        let queue = &self.0;
        let _turn = queue.turn.acquire().await.ok()?;
        loop {
            let progress = queue.progress.notified();
            match queue.try_push(line, urgent, true) {
                Ok(number) => return Some(number),
                Err(Refused::Closed) => return None,
                Err(Refused::Full) => progress.await,
            }
        }
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
            let lost = self.give_up_batch();
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

    /// Settle the batch in hand as dropped, after a panic.
    fn give_up_batch(&mut self) -> u64 {
        let mut state = self.queue.lock();
        let lost = std::mem::take(&mut state.writing_lines);
        state.writing = 0;
        state.settled += lost;
        #[cfg(test)]
        {
            state.dropped += lost;
        }
        drop(state);
        self.queue.progress.notify_waiters();
        lost
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
                    self.write(&[], 0);
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
            let lines = std::mem::take(&mut state.lines);
            state.writing = batch.len();
            state.writing_lines = lines;
            state.urgent = false;
            let finishing = state.finished;
            drop(state);

            self.last_write = Some(Instant::now());
            let _dropped = self.write(batch.as_bytes(), lines);
            if finishing {
                // The last lines of a shutdown: say now what
                // was dropped, not at the next report.
                self.report(None);
            }

            let mut state = self.queue.lock();
            state.writing = 0;
            state.writing_lines = 0;
            state.settled += lines;
            #[cfg(test)]
            {
                state.dropped += _dropped;
            }
            drop(state);
            self.queue.progress.notify_waiters();

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

    /// Write `batch`, which holds `lines` whole lines.
    /// Returns how many of them were given up on.
    fn write(&mut self, batch: &[u8], lines: u64) -> u64 {
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
                debug_assert!(dropped <= lines);
                self.failing = true;
                self.unreported += dropped;
                let due = self
                    .reported
                    .is_none_or(|at| at.elapsed() >= REPORT_EVERY);
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
