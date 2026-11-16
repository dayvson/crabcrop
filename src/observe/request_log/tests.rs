//! Tests for the request log (issue #642): the queue and
//! its writer thread, over a stand-in for standard output
//! that each test steers and reads back.

use super::fake::{FakeStdout, Notes};
use super::*;

/// Long enough for anything that must happen to happen on
/// a loaded machine; a test that passes never waits it out.
const LONG: Duration = Duration::from_secs(60);

/// Long enough to tell "does not happen" from "has not
/// happened yet".
const SHORT: Duration = Duration::from_millis(50);

/// Wait until `done`, checking every millisecond.
async fn until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + LONG;
    while !done() {
        assert!(Instant::now() < deadline, "never happened: {what}");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// A queued log over a fresh stand-in, with the full queue.
fn queued() -> (RequestLog, FakeStdout, Notes) {
    queued_holding(QUEUE_BYTES)
}

/// A queued log over a fresh stand-in that is full at `cap`
/// bytes.
fn queued_holding(cap: usize) -> (RequestLog, FakeStdout, Notes) {
    let out = FakeStdout::default();
    let (log, notes) = RequestLog::over(&out, cap);
    (log, out, notes)
}

/// Line `n` of a test: 99 bytes, 100 with its newline.
fn numbered(n: usize) -> String {
    format!("{n:04} {}", "x".repeat(94))
}

/// Lines `from` up to `to`, which is not one of them.
fn numbered_lines(from: usize, to: usize) -> Vec<String> {
    (from..to).map(numbered).collect()
}

/// The log's queue.
fn queue_of(log: &RequestLog) -> Arc<Queue> {
    match &*log.0 {
        Form::Queued(handle) => Arc::clone(&handle.0),
        _ => panic!("not a queued log"),
    }
}

/// What a handler does: ask for room, then log its line.
async fn request(log: RequestLog, line: String) {
    log.room().await;
    log.line(&line);
}

/// Stop the stand-in being read and fill a log that is full
/// at 1000 bytes: line 0 in the writer's hands, lines 1 to
/// 9 queued.
async fn fill(log: &RequestLog, out: &FakeStdout) {
    out.stall();
    log.line(&numbered(0));
    out.blocked().await;
    for n in 1..10 {
        assert!(!log.stats().full, "full before line {n}");
        log.line(&numbered(n));
    }
    assert_eq!(log.stats().held_bytes, 1000);
    assert!(log.stats().full);
    assert_eq!(log.stats().waiting, 0);
}

/// The counts in the notes: every "N lines dropped".
fn dropped_in(notes: &Notes) -> u64 {
    notes
        .all()
        .iter()
        .filter_map(|note| {
            let before = note.strip_suffix(" lines dropped")?;
            before.rsplit(' ').next()?.parse::<u64>().ok()
        })
        .sum()
}

#[tokio::test]
async fn lines_come_out_whole_and_in_the_order_they_were_logged() {
    let (log, out, notes) = queued();
    let mut expected = String::new();
    for n in 0..600 {
        let line = format!("{{\"n\":{n},\"pad\":\"{}\"}}", "p".repeat(n % 97));
        // A request's line and an audit line go through one
        // queue.
        match n % 40 {
            0 => log.event(&line).await,
            _ => log.line(&line),
        }
        expected.push_str(&line);
        expected.push('\n');
    }
    assert_eq!(log.finish(Some("the last line"), LONG).await, 0);
    expected.push_str("the last line\n");
    assert_eq!(out.text(), expected);
    // Each write carried whole lines: the stand-in takes
    // all it is given, so a write is a batch.
    for write in out.writes() {
        assert!(write.ends_with('\n'), "{write:?}");
    }
    assert_eq!(log.stats().dropped, 0);
    assert!(notes.all().is_empty(), "{:?}", notes.all());
}

#[tokio::test]
async fn what_gathers_while_a_write_is_out_goes_in_one_write() {
    let (log, out, _) = queued();
    out.stall();
    log.line("first");
    // The writer has "first" in hand and is stuck in its
    // write. A thousand lines arrive meanwhile.
    out.blocked().await;
    for n in 0..1000 {
        log.line(&format!("line {n}"));
    }
    out.resume();
    assert_eq!(log.finish(None, LONG).await, 0);
    let writes = out.writes();
    assert_eq!(writes.len(), 2, "{} writes", writes.len());
    assert_eq!(writes[0], "first\n");
    let gathered: Vec<&str> = writes[1].lines().collect();
    assert_eq!(gathered.len(), 1000);
    assert_eq!(gathered[0], "line 0");
    assert_eq!(gathered[999], "line 999");
}

#[tokio::test]
async fn the_first_line_after_a_pause_is_written_at_once() {
    let (log, out, _) = queued();
    let queue = queue_of(&log);
    for n in 0..3 {
        until("the writer sleeps", || queue.lock().asleep).await;
        let woken = log.stats().wakeups;
        log.line(&format!("line {n}"));
        until("the line is written", || out.lines().len() == n + 1).await;
        // One wake, from the sleep it was in: it did not
        // then wait out the pace before writing.
        assert_eq!(log.stats().wakeups, woken + 1, "line {n}");
        // Longer than the pace.
        std::thread::sleep(WRITE_EVERY * 3);
    }
}

#[tokio::test]
async fn lines_that_come_close_together_share_writes() {
    let (log, out, _) = queued();
    let started = Instant::now();
    // A line every 100 µs or so, for 30 ms or more.
    for n in 0..300 {
        log.line(&format!("line {n}"));
        std::thread::sleep(Duration::from_micros(100));
    }
    assert_eq!(log.finish(None, LONG).await, 0);
    let took = started.elapsed();
    let writes = out.writes().len();
    assert_eq!(out.lines().len(), 300);
    // A write starts no sooner than `WRITE_EVERY` after the
    // one before: so many fit in the time it took, and one
    // more for the queue's last lines. A writer that wrote
    // each line as it came would have made 300.
    let most = (took.as_micros() / WRITE_EVERY.as_micros()) as usize + 2;
    assert!(
        writes <= most,
        "{writes} writes in {took:?}, more than one per {WRITE_EVERY:?}"
    );
}

#[tokio::test]
async fn the_writer_sleeps_until_a_line_arrives() {
    let (log, out, _) = queued();
    log.line("one");
    until("the line is written", || out.text() == "one\n").await;
    until("the writer sleeps", || queue_of(&log).lock().asleep).await;
    // Nothing to write: the thread must not wake at all,
    // on a timer or to look at the queue.
    let woken = log.stats().wakeups;
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(log.stats().wakeups, woken, "the writer woke with no line");
    // And a line still wakes it.
    log.line("two");
    until("the next line is written", || out.text() == "one\ntwo\n").await;
    assert_eq!(log.stats().wakeups, woken + 1);
}

#[tokio::test]
async fn a_request_asks_for_room_and_gets_it_at_once_while_there_is_some() {
    let (log, out, _) = queued_holding(1000);
    out.stall();
    // Nobody reads, and the queue fills line by line: up to
    // the last line that fits, no request waits.
    for n in 0..10 {
        tokio::time::timeout(LONG, log.room())
            .await
            .expect("a request waits though the queue has room");
        log.line(&numbered(n));
        if n == 0 {
            out.blocked().await;
        }
    }
    assert_eq!(log.stats().waiting, 0);
    assert!(log.stats().full);
    out.resume();
    assert_eq!(log.finish(None, LONG).await, 0);
    assert_eq!(out.lines(), numbered_lines(0, 10));
}

#[tokio::test]
async fn a_full_queue_makes_requests_wait_for_room_and_loses_no_line() {
    let (log, out, notes) = queued_holding(1000);
    fill(&log, &out).await;
    // The queue holds its 1000 bytes. Five more requests
    // come: none is handled, so none is answered.
    let mut requests = Vec::new();
    for n in 10..15 {
        requests.push(tokio::spawn(request(log.clone(), numbered(n))));
    }
    until("five requests wait for room", || log.stats().waiting == 5).await;
    tokio::time::sleep(SHORT).await;
    assert!(requests.iter().all(|request| !request.is_finished()));
    assert_eq!(log.stats().held_bytes, 1000);
    // The reader comes back: they are handled, and every
    // line is there, once.
    out.resume();
    for request in requests {
        tokio::time::timeout(LONG, request).await.unwrap().unwrap();
    }
    assert_eq!(log.finish(None, LONG).await, 0);
    let lines = out.lines();
    assert_eq!(lines[..10], numbered_lines(0, 10));
    // The five went on together, in no order of their own.
    let mut late = lines[10..].to_vec();
    late.sort();
    assert_eq!(late, numbered_lines(10, 15));
    assert_eq!(log.stats().dropped, 0);
    assert_eq!(log.stats().waiting, 0);
    assert!(!log.stats().full);
    assert!(notes.all().is_empty(), "{:?}", notes.all());
}

#[tokio::test]
async fn a_full_queue_holds_its_cap_and_one_line_for_each_request_under_way() {
    let (log, out, _) = queued_holding(1000);
    fill(&log, &out).await;
    // Twenty requests were under way when the queue filled:
    // they had asked for room and got it. Each still leaves
    // its line, which is all the queue takes past its cap.
    for n in 10..30 {
        log.line(&numbered(n));
    }
    assert_eq!(log.stats().held_bytes, 3000);
    // Every request after them waits for room, however many
    // come, and a client that gives up takes its request
    // with it: the queue does not grow by a byte.
    for round in 0..5 {
        let mut requests = Vec::new();
        for _ in 0..100 {
            let line = "the line of a request that was never handled".to_string();
            requests.push(tokio::spawn(request(log.clone(), line)));
        }
        until("a hundred requests wait", || log.stats().waiting == 100).await;
        for request in requests {
            request.abort();
            assert!(request.await.unwrap_err().is_cancelled());
        }
        assert_eq!(log.stats().waiting, 0, "round {round}");
        assert_eq!(log.stats().held_bytes, 3000, "round {round}");
    }
    out.resume();
    assert_eq!(log.finish(None, LONG).await, 0);
    assert_eq!(out.lines(), numbered_lines(0, 30));
}

#[tokio::test]
async fn a_line_longer_than_the_queue_gets_through() {
    let (log, out, _) = queued_holding(100);
    let long = |tag: char| tag.to_string().repeat(500);
    out.stall();
    log.line(&long('a'));
    out.blocked().await;
    // One line fills the queue five times over. The next
    // request waits until it is written, and is then
    // handled like any other.
    assert!(log.stats().full);
    let second = tokio::spawn(request(log.clone(), long('b')));
    until("the second waits", || log.stats().waiting == 1).await;
    out.resume();
    tokio::time::timeout(LONG, second).await.unwrap().unwrap();
    assert_eq!(log.finish(None, LONG).await, 0);
    assert_eq!(out.lines(), [long('a'), long('b')]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_tasks_logging_at_once_lose_and_repeat_nothing() {
    // A small queue, so requests wait for room all the
    // time, and a stream that takes a few bytes per write,
    // so every batch is many writes.
    let (log, out, notes) = queued_holding(2048);
    out.take_at_most(37);
    const TASKS: usize = 8;
    const LINES: usize = 400;
    let mut tasks = Vec::new();
    for task in 0..TASKS {
        let log = log.clone();
        tasks.push(tokio::spawn(async move {
            for n in 0..LINES {
                let line = format!("task {task} line {n} {}", "z".repeat((task * 7 + n) % 60));
                log.room().await;
                match n % 50 {
                    7 => log.event(&line).await,
                    _ => log.line(&line),
                }
                if n % 16 == 0 {
                    tokio::task::yield_now().await;
                }
            }
        }));
    }
    for task in tasks {
        tokio::time::timeout(LONG, task).await.unwrap().unwrap();
    }
    assert_eq!(log.finish(None, LONG).await, 0);
    let lines = out.lines();
    assert_eq!(lines.len(), TASKS * LINES);
    // Every line is one a task logged, whole, and each
    // task's lines are all there, once, in its order.
    let mut next = [0usize; TASKS];
    for line in &lines {
        let mut words = line.split(' ');
        assert_eq!(words.next(), Some("task"), "{line:?}");
        let task: usize = words.next().unwrap().parse().unwrap();
        assert_eq!(words.next(), Some("line"), "{line:?}");
        let n: usize = words.next().unwrap().parse().unwrap();
        assert_eq!(n, next[task], "task {task}: {line:?}");
        assert_eq!(
            words.next(),
            Some("z".repeat((task * 7 + n) % 60).as_str()),
            "{line:?}"
        );
        assert_eq!(words.next(), None, "{line:?}");
        next[task] += 1;
    }
    assert_eq!(next, [LINES; TASKS]);
    assert_eq!(log.stats().dropped, 0);
    assert_eq!(log.stats().waiting, 0);
    assert!(notes.all().is_empty(), "{:?}", notes.all());
}

#[tokio::test]
async fn an_event_returns_once_its_line_is_written() {
    let (log, out, _) = queued();
    out.stall();
    // A request's line does not wait for the write.
    log.line("a request");
    out.blocked().await;
    // A purge's line does.
    let purge = {
        let log = log.clone();
        tokio::spawn(async move { log.event("a purge").await })
    };
    tokio::time::sleep(SHORT).await;
    assert!(!purge.is_finished());
    out.resume();
    tokio::time::timeout(LONG, purge).await.unwrap().unwrap();
    assert_eq!(out.text(), "a request\na purge\n");
}

#[tokio::test]
async fn a_line_is_taken_at_once_whatever_the_queue_holds() {
    let (log, out, _) = queued_holding(1000);
    fill(&log, &out).await;
    // The queue is full and nobody reads. A shutdown's
    // line, like the line of a request under way, is queued
    // all the same, in its place: `line` does not wait.
    log.line("shutdown: draining");
    assert_eq!(log.stats().held_bytes, 1000 + "shutdown: draining\n".len());
    out.resume();
    assert_eq!(log.finish(Some("shutdown: done"), LONG).await, 0);
    let lines = out.lines();
    assert_eq!(lines.len(), 12);
    assert_eq!(lines[10], "shutdown: draining");
    assert_eq!(lines[11], "shutdown: done");
}

#[tokio::test]
async fn finish_writes_the_queue_then_its_last_line_and_ends_the_log() {
    let (log, out, _) = queued();
    out.stall();
    log.line("one");
    out.blocked().await;
    log.line("two");
    let finish = {
        let log = log.clone();
        tokio::spawn(async move { log.finish(Some("the end"), LONG).await })
    };
    // It waits for the write that is out.
    tokio::time::sleep(SHORT).await;
    assert!(!finish.is_finished());
    out.resume();
    assert_eq!(finish.await.unwrap(), 0);
    assert_eq!(out.text(), "one\ntwo\nthe end\n");
    // The log has ended: nothing is written after its last
    // line, not a second last line either, and nothing
    // waits.
    log.line("late");
    tokio::time::timeout(LONG, log.event("late")).await.unwrap();
    tokio::time::timeout(LONG, log.room()).await.unwrap();
    assert_eq!(log.finish(Some("a second end"), LONG).await, 0);
    assert_eq!(out.text(), "one\ntwo\nthe end\n");
}

#[tokio::test]
async fn finish_gives_up_at_its_limit_when_nothing_is_read() {
    let (log, out, _) = queued_holding(1000);
    fill(&log, &out).await;
    let waiting: Vec<_> = (10..12)
        .map(|n| tokio::spawn(request(log.clone(), numbered(n))))
        .collect();
    until("two requests wait", || log.stats().waiting == 2).await;
    let limit = Duration::from_millis(200);
    let started = Instant::now();
    let left = log.finish(Some("the end"), limit).await;
    let took = started.elapsed();
    // The ten held, and the last line.
    assert_eq!(left, 11);
    assert!(took >= limit, "gave up after {took:?}");
    assert!(took < LONG / 2, "gave up after {took:?}");
    // The requests that waited for room go on. The log has
    // ended, so their lines are not taken.
    for request in waiting {
        tokio::time::timeout(LONG, request).await.unwrap().unwrap();
    }
    // Nothing was thrown away: a reader that comes back
    // still gets the queue, ended by the last line.
    out.resume();
    until("the queue is written", || log.stats().held_bytes == 0).await;
    let lines = out.lines();
    assert_eq!(lines.len(), 11);
    assert_eq!(lines[10], "the end");
}

#[tokio::test]
async fn a_failing_standard_output_drops_lines_counts_them_and_holds_nothing_up() {
    let (log, out, notes) = queued_holding(1000);
    log.line("before");
    until("the line is written", || out.text() == "before\n").await;
    out.fail_after(0, io::ErrorKind::BrokenPipe);
    // Twenty times what the queue holds: no request waits
    // for room for long, and an audit line's caller is let
    // go too.
    for n in 0..200 {
        tokio::time::timeout(LONG, log.room())
            .await
            .expect("a request waits on a standard output that fails");
        log.line(&numbered(n));
    }
    tokio::time::timeout(LONG, log.event("an audit line"))
        .await
        .expect("an audit line waits on a standard output that fails");
    until("the lines are given up on", || log.stats().dropped == 201).await;
    assert_eq!(out.text(), "before\n");
    // Stderr is told at the first failure, with the reason
    // and a count, and then every ten seconds at most: not
    // once per line or per write.
    let told = notes.all();
    assert!(
        told[0].starts_with("request log: cannot write to standard output: "),
        "{told:?}"
    );
    assert!(told[0].ends_with(" lines dropped"), "{told:?}");
    assert!(told.len() <= 3, "{told:?}");
    // Standard output takes writes again: so does the log,
    // and stderr gets the rest of the count.
    out.heal();
    log.line("after");
    until("the log is back", || out.text() == "before\nafter\n").await;
    let told = notes.all();
    assert!(
        told.last()
            .unwrap()
            .starts_with("request log: standard output takes lines again"),
        "{told:?}"
    );
    assert_eq!(dropped_in(&notes), 201, "{told:?}");
    assert_eq!(log.stats().dropped, 201);
}

#[tokio::test]
async fn a_line_cut_by_a_failed_write_is_finished_before_the_next_line() {
    let (log, out, notes) = queued();
    // Five lines in one batch, behind a first line.
    out.stall();
    log.line("first");
    out.blocked().await;
    for n in 0..5 {
        log.line(&numbered(n));
    }
    // The stream takes "first", line 0, and 44 bytes of
    // line 1, then fails.
    out.fail_after("first\n".len() + 100 + 44, io::ErrorKind::StorageFull);
    out.resume();
    until("the write fails", || log.stats().dropped == 3).await;
    assert_eq!(
        out.text(),
        format!("first\n{}\n{}", numbered(0), &numbered(1)[..44])
    );
    // It takes writes again. Line 1 is finished first, so
    // the stream never holds a line with another inside.
    out.heal();
    log.line("next");
    assert_eq!(log.finish(None, LONG).await, 0);
    assert_eq!(
        out.lines(),
        [
            "first".to_string(),
            numbered(0),
            numbered(1),
            "next".to_string()
        ]
    );
    // Lines 2, 3 and 4 are the ones lost, and the ones
    // counted.
    assert_eq!(log.stats().dropped, 3);
    assert_eq!(dropped_in(&notes), 3, "{:?}", notes.all());
}

#[tokio::test]
async fn a_write_that_would_block_is_tried_again_and_loses_nothing() {
    let (log, out, notes) = queued();
    out.would_block(6);
    out.take_at_most(11);
    for n in 0..20 {
        log.line(&numbered(n));
    }
    assert_eq!(log.finish(None, LONG).await, 0);
    assert_eq!(out.lines(), numbered_lines(0, 20));
    assert_eq!(log.stats().dropped, 0);
    assert!(notes.all().is_empty(), "{:?}", notes.all());
}

#[tokio::test]
async fn a_panic_in_a_write_does_not_end_the_writer() {
    let (log, out, notes) = queued_holding(1000);
    out.panic_next();
    log.line("lost with the write that panicked");
    until("the batch is given up", || log.stats().dropped == 1).await;
    // The thread is still there, and the queue still moves:
    // far more than it holds goes through.
    for n in 0..100 {
        tokio::time::timeout(LONG, log.room())
            .await
            .expect("the queue is stuck: the writer ended");
        log.line(&numbered(n));
    }
    assert_eq!(log.finish(None, LONG).await, 0);
    assert_eq!(out.lines(), numbered_lines(0, 100));
    assert!(!log.stats().writer_gone);
    assert_eq!(
        notes.all(),
        ["request log: the writer failed and starts again; 1 lines dropped"]
    );
}

#[tokio::test]
async fn dropping_the_log_writes_what_is_queued_and_ends_the_writer() {
    let (log, out, _) = queued();
    let queue = queue_of(&log);
    out.stall();
    log.line("one");
    out.blocked().await;
    log.line("two");
    let clone = log.clone();
    drop(log);
    // A clone keeps the log open.
    clone.line("three");
    drop(clone);
    out.resume();
    until("the writer has ended", || queue.lock().writer_gone).await;
    assert_eq!(out.text(), "one\ntwo\nthree\n");
}

#[tokio::test]
async fn the_queue_gives_back_the_memory_a_stalled_reader_made_it_take() {
    let (log, out, _) = queued();
    let queue = queue_of(&log);
    out.stall();
    log.line("first");
    out.blocked().await;
    // About 2 MiB while nobody reads.
    let line = "m".repeat(1023);
    for _ in 0..2048 {
        log.line(&line);
    }
    assert!(queue.lock().text.capacity() >= 2 * 1024 * 1024);
    out.resume();
    until("the queue is written", || out.lines().len() == 2049).await;
    // The buffer that held it is the queue's again after
    // the next write, and it is small.
    log.line("after");
    until("the next line is written", || out.lines().len() == 2050).await;
    assert!(queue.lock().text.capacity() <= KEPT_CAPACITY);
}

#[tokio::test]
async fn an_inline_log_and_a_queued_one_write_the_same_bytes() {
    let inline = FakeStdout::default();
    let queued = FakeStdout::default();
    let logs = [
        RequestLog::inline_to(inline.clone()),
        RequestLog::queued_to(queued.clone()),
    ];
    for log in &logs {
        for n in 0..50 {
            let line = format!("line {n}");
            tokio::time::timeout(LONG, log.room()).await.unwrap();
            match n % 5 {
                0 => log.event(&line).await,
                _ => log.line(&line),
            }
        }
        assert_eq!(log.finish(Some("the end"), LONG).await, 0);
    }
    assert_eq!(inline.lines().len(), 51);
    assert_eq!(inline.text(), queued.text());
    // The inline log wrote each line where it was logged:
    // one write for the line, one for its newline.
    assert_eq!(inline.writes().len(), 102);
}

#[tokio::test]
async fn the_queue_shows_what_it_holds_who_waits_and_how_long_it_was_full() {
    // Issue #707: `/metrics` reads these.
    let (log, out, _) = queued_holding(1000);
    let at_rest = log.queue_state().unwrap();
    assert_eq!(
        at_rest,
        QueueState {
            held_bytes: 0,
            capacity: 1000,
            waiting: 0,
            full_time: Duration::ZERO,
        }
    );
    fill(&log, &out).await;
    let requests: Vec<_> = (10..13)
        .map(|n| tokio::spawn(request(log.clone(), numbered(n))))
        .collect();
    until("three requests wait", || log.stats().waiting == 3).await;
    let full = log.queue_state().unwrap();
    assert_eq!(full.held_bytes, 1000);
    assert_eq!(full.waiting, 3);
    // The spell under way counts, and grows.
    tokio::time::sleep(SHORT).await;
    let later = log.queue_state().unwrap();
    assert!(later.full_time >= SHORT, "{later:?}");
    assert!(later.full_time > full.full_time, "{later:?} {full:?}");
    out.resume();
    for request in requests {
        tokio::time::timeout(LONG, request).await.unwrap().unwrap();
    }
    until("the queue has room", || !log.stats().full).await;
    let after = log.queue_state().unwrap();
    assert_eq!(after.waiting, 0);
    // The spell ended: the time spent full stops growing.
    tokio::time::sleep(SHORT).await;
    assert_eq!(log.queue_state().unwrap().full_time, after.full_time);
    assert!(after.full_time >= later.full_time);
    assert_eq!(log.finish(None, LONG).await, 0);
}

#[tokio::test]
async fn the_log_stalls_once_its_queue_has_stayed_full_for_its_bound() {
    // Issue #707: what the health check reads. A queue
    // that is full counts as stalled only once it has
    // stayed full for the bound, and no longer once it has
    // room again; a new spell starts the count again.
    let bound = Duration::from_millis(500);
    let out = FakeStdout::default();
    let (log, _) = RequestLog::over_stalling(&out, 1000, bound);
    assert_eq!(log.stalled(), None);
    let filling = Instant::now();
    fill(&log, &out).await;
    if filling.elapsed() < bound / 2 {
        assert_eq!(log.stalled(), None, "stalled as soon as it filled");
    }
    until("the log stalls", || log.stalled().is_some()).await;
    let stalled = log.stalled().unwrap();
    assert!(stalled >= bound, "{stalled:?}");
    assert!(filling.elapsed() >= bound);
    out.resume();
    until("the queue has room", || !log.stats().full).await;
    assert_eq!(log.stalled(), None);
    until("the queue is written", || log.stats().held_bytes == 0).await;
    // Full again: a new spell, counted from its own start.
    let refilling = Instant::now();
    fill(&log, &out).await;
    if refilling.elapsed() < bound / 2 {
        assert_eq!(log.stalled(), None, "the spell before counted");
    }
    out.resume();
    assert_eq!(log.finish(None, LONG).await, 0);
    // A log whose default bound is STALLED_AFTER, and an
    // inline log, which has no queue to stall.
    let (default, out, _) = queued_holding(1000);
    fill(&default, &out).await;
    assert_eq!(default.stalled(), None);
    out.resume();
    assert_eq!(default.finish(None, LONG).await, 0);
    let inline = RequestLog::inline_to(FakeStdout::default());
    assert_eq!(inline.stalled(), None);
    assert_eq!(inline.queue_state(), None);
}

#[tokio::test]
async fn an_inline_log_writes_nothing_after_its_last_line() {
    // Issue #707: a request whose client went away files
    // its line when it is dropped, which at shutdown can
    // be after the last line. A queued log ends with its
    // last line whatever comes after; an inline log now
    // does too.
    let out = FakeStdout::default();
    let log = RequestLog::inline_to(out.clone());
    log.line("one");
    assert_eq!(log.finish(Some("the end"), LONG).await, 0);
    log.line("late");
    log.event("late").await;
    assert_eq!(log.finish(Some("a second end"), LONG).await, 0);
    assert_eq!(out.text(), "one\nthe end\n");
    // Finished with no last line, it ends all the same.
    let out = FakeStdout::default();
    let log = RequestLog::inline_to(out.clone());
    assert_eq!(log.finish(None, LONG).await, 0);
    log.line("late");
    assert_eq!(out.text(), "");
}

/// Line `n` of a test of pieces: 999 bytes, 1000 with its
/// newline.
fn kilo(n: usize) -> String {
    format!("{n:04} {}", "k".repeat(994))
}

#[tokio::test]
async fn a_batch_is_written_in_pieces_that_end_with_a_line() {
    // Issue #707: the writer hands standard output at most
    // `WRITE_AT_MOST` bytes of a batch at a time, a piece
    // ending with a line where one ends within it, and a
    // line longer than a piece in pieces of its own.
    let (log, out, notes) = queued();
    out.stall();
    log.line("first");
    out.blocked().await;
    // 300 lines of 1000 bytes behind it, 65 to a piece,
    // then a line longer than a piece.
    for n in 0..300 {
        log.line(&kilo(n));
    }
    let long = "l".repeat(WRITE_AT_MOST + 10);
    log.line(&long);
    out.resume();
    assert_eq!(log.finish(None, LONG).await, 0);
    let sizes: Vec<usize> = out.writes().iter().map(String::len).collect();
    assert_eq!(
        sizes,
        [6, 65_000, 65_000, 65_000, 65_000, 40_000, WRITE_AT_MOST, 11]
    );
    let mut expected = vec!["first".to_string()];
    expected.extend((0..300).map(kilo));
    expected.push(long);
    assert_eq!(out.lines(), expected);
    assert!(notes.all().is_empty(), "{:?}", notes.all());
}

#[tokio::test]
async fn a_full_queue_has_room_again_once_standard_output_takes_a_piece_of_its_batch() {
    // Issue #707: the writer handed standard output its
    // whole batch in one write, and the room the batch took
    // came back only once all of it was written. A reader
    // that was slow, not gone, then kept the queue full,
    // and every request that logs waiting, for as long as
    // it took to read the whole batch, and the health check
    // failed on it. Each piece standard output takes is now
    // room again.
    let (log, out, notes) = queued_holding(2 * WRITE_AT_MOST);
    out.stall();
    log.line(&kilo(0));
    out.blocked().await;
    // Lines 1 to 140 queue behind line 0 and fill the queue.
    for n in 1..=140 {
        log.line(&kilo(n));
    }
    assert!(log.stats().full);
    // The stream takes line 0 and stops again: the writer
    // has lines 1 to 140 in hand, one batch, and is blocked
    // in its first piece. The queue is still full.
    out.stall_after(1000);
    out.resume();
    out.blocked().await;
    assert_eq!(out.text(), format!("{}\n", kilo(0)));
    assert_eq!(log.stats().held_bytes, 140 * 1000);
    assert!(log.stats().full);
    let waiting = tokio::spawn(request(log.clone(), "waited".to_string()));
    until("a request waits for room", || log.stats().waiting == 1).await;
    // The stream takes one piece's worth and stops again.
    // What it took is room: the request goes on, though
    // most of the batch is still to be written.
    out.stall_after(WRITE_AT_MOST);
    out.resume();
    tokio::time::timeout(LONG, waiting).await.unwrap().unwrap();
    assert!(
        out.text().len() <= 1000 + WRITE_AT_MOST,
        "{} bytes written",
        out.text().len()
    );
    // Once the stream is read, everything is there, in
    // order, the request's line last.
    out.resume();
    assert_eq!(log.finish(None, LONG).await, 0);
    let mut expected: Vec<String> = (0..=140).map(kilo).collect();
    expected.push("waited".to_string());
    assert_eq!(out.lines(), expected);
    assert_eq!(log.stats().dropped, 0);
    assert!(notes.all().is_empty(), "{:?}", notes.all());
}
