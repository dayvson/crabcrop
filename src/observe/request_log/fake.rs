//! A stand-in for standard output that a test steers and
//! reads back: it records every `write`, and can stop
//! being read, fail, take a few bytes at a time, or panic.

use super::*;

/// The stream. Clones share it, so a test keeps one and
/// gives the log another.
#[derive(Clone, Default)]
pub(crate) struct FakeStdout(Arc<Shared>);

#[derive(Default)]
struct Shared {
    state: Mutex<FakeState>,
    changed: Condvar,
}

#[derive(Default)]
struct FakeState {
    /// The bytes each `write` took, in order.
    writes: Vec<Vec<u8>>,
    /// Nobody reads: a `write` blocks until this clears.
    stalled: bool,
    /// A `write` is blocked on `stalled`.
    blocked: bool,
    /// The most bytes one `write` takes; 0 for all it is
    /// given.
    at_most: usize,
    /// Every `write` fails with this, once the stream has
    /// taken `before_failing` more bytes.
    failing: Option<io::ErrorKind>,
    before_failing: usize,
    /// The next writes that answer "would block".
    would_block: u32,
    /// The next `write` panics.
    panics: bool,
}

impl FakeStdout {
    fn state(&self) -> MutexGuard<'_, FakeState> {
        self.0.state.lock().unwrap()
    }

    /// Everything written so far.
    pub(crate) fn text(&self) -> String {
        String::from_utf8(self.state().writes.concat()).unwrap()
    }

    /// The whole lines written so far.
    pub(crate) fn lines(&self) -> Vec<String> {
        self.text().lines().map(str::to_string).collect()
    }

    /// What each `write` took.
    pub(crate) fn writes(&self) -> Vec<String> {
        let state = self.state();
        state
            .writes
            .iter()
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .collect()
    }

    /// Stop reading: the next `write` blocks.
    pub(crate) fn stall(&self) {
        self.state().stalled = true;
    }

    /// Read again.
    pub(crate) fn resume(&self) {
        self.state().stalled = false;
        self.0.changed.notify_all();
    }

    /// Wait until a `write` is blocked on a stream nobody
    /// reads: the writer has a batch in hand.
    pub(crate) async fn blocked(&self) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !self.state().blocked {
            assert!(Instant::now() < deadline, "no write reached the stream");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// Take at most `bytes` bytes for each `write`.
    pub(crate) fn take_at_most(&self, bytes: usize) {
        self.state().at_most = bytes;
    }

    /// Fail every `write` with `kind`, after taking `first`
    /// more bytes.
    pub(crate) fn fail_after(&self, first: usize, kind: io::ErrorKind) {
        let mut state = self.state();
        state.failing = Some(kind);
        state.before_failing = first;
    }

    /// Take writes again.
    pub(crate) fn heal(&self) {
        self.state().failing = None;
    }

    /// Answer "would block" to the next `times` writes.
    pub(crate) fn would_block(&self, times: u32) {
        self.state().would_block = times;
    }

    /// Panic in the next `write`.
    pub(crate) fn panic_next(&self) {
        self.state().panics = true;
    }
}

impl Sink for FakeStdout {
    fn hold(&mut self, write: &mut dyn FnMut(&mut dyn Write)) {
        write(self);
    }
}

impl Write for FakeStdout {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut state = self.state();
        while state.stalled {
            state.blocked = true;
            state = self.0.changed.wait(state).unwrap();
        }
        state.blocked = false;
        if state.panics {
            state.panics = false;
            drop(state);
            // Unwinds like a panic, without the message a
            // panic prints.
            std::panic::resume_unwind(Box::new("a write that panics"));
        }
        if state.would_block > 0 {
            state.would_block -= 1;
            return Err(io::ErrorKind::WouldBlock.into());
        }
        let mut take = bytes.len();
        if state.at_most > 0 {
            take = take.min(state.at_most);
        }
        if let Some(kind) = state.failing {
            if state.before_failing == 0 {
                return Err(kind.into());
            }
            take = take.min(state.before_failing);
            state.before_failing -= take;
        }
        state.writes.push(bytes[..take].to_vec());
        Ok(take)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// What a log over a [`FakeStdout`] said it dropped: the
/// lines it would have printed on stderr.
#[derive(Clone, Default)]
pub(crate) struct Notes(Arc<Mutex<Vec<String>>>);

impl Notes {
    pub(crate) fn all(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

impl RequestLog {
    /// A queued log that writes to `out`, holds `cap` bytes,
    /// and reports what it drops to the notes it returns.
    pub(crate) fn over(out: &FakeStdout, cap: usize) -> (Self, Notes) {
        Self::over_stalling(out, cap, STALLED_AFTER)
    }

    /// [`RequestLog::over`], counting as stalled once its
    /// queue has been full for `after` (issue #707), so a
    /// test need not wait out [`STALLED_AFTER`].
    pub(crate) fn over_stalling(out: &FakeStdout, cap: usize, after: Duration) -> (Self, Notes) {
        let notes = Notes::default();
        let sink = notes.clone();
        let note: Note = Box::new(move |text| sink.0.lock().unwrap().push(text.to_string()));
        let log = Self::queued(Box::new(out.clone()), cap, after, note).unwrap();
        (log, notes)
    }
}
